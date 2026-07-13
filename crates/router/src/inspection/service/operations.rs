// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mutation capability backed by an active runtime or a bounded standalone writer.

use std::ffi::OsString;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(unix)]
use std::os::fd::OwnedFd;
#[cfg(windows)]
use std::{fs::File, os::windows::io::AsRawHandle};

use chrono::Utc;
use tokio::sync::{Mutex as AsyncMutex, watch};
use tokio::task::JoinHandle;
use uuid::Uuid;

use super::super::{
    CohortRotationRequestV1, InspectionControlRequestV1, InspectionError, LearningResetRequestV1,
    OperatorMutationReceiptV1,
};
use super::{InspectionService, map_ledger_error, map_ledger_error_class};
use crate::config::RouterConfig;
use crate::control::{
    ControlMutation, ControlMutationOptions, ControlRuntimeAuthority, RouterControlError,
    RouterControlService, RouterControlSnapshot, router_control_service_for,
};
use crate::ledger::command::{WriterFailure, WriterFailureClass};
use crate::ledger::read_pool::LedgerReadPool;
use crate::ledger::repository::inspection::InspectionAuthoritySnapshot;
use crate::ledger::repository::process::{
    HeartbeatAck, HeartbeatRenewal, ProcessCommandAck, ProcessStop,
};
use crate::ledger::repository::{LedgerRepository, OperationsLedger};
use crate::ledger::writer::{LedgerWriterClient, LedgerWriterOwner};
use crate::provider_admission::ProviderAdmissionGate;

const OPERATION_TRANSACTION_START_TIMEOUT_MS: u64 = 2_000;
const OPERATIONS_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const OPERATIONS_HEARTBEAT_WRITE_TIMEOUT: Duration = Duration::from_secs(2);
const LIFECYCLE_RUNNING: u8 = 0;
const LIFECYCLE_CLOSING: u8 = 1;
const LIFECYCLE_CLOSED: u8 = 2;
const LIFECYCLE_ABORTED: u8 = 3;

static OPERATIONS_OPEN_GATE: AsyncMutex<()> = AsyncMutex::const_new(());

pub(super) enum OperationsBackend {
    Attached(RouterControlService),
    Standalone(Box<StandaloneOperations>),
}

impl OperationsBackend {
    pub(super) async fn open(
        config: &RouterConfig,
        read_pool: &LedgerReadPool,
        authority: &InspectionAuthoritySnapshot,
        timeout_ms: u64,
    ) -> Result<Self, InspectionError> {
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(timeout_ms))
            .ok_or(InspectionError::InvalidArgument)?;
        let _open = tokio::time::timeout_at(
            tokio::time::Instant::from_std(deadline),
            OPERATIONS_OPEN_GATE.lock(),
        )
        .await
        .map_err(|_| InspectionError::Busy)?;

        if let Some(service) =
            router_control_service_for(authority.project_uuid, &authority.config_generation_id)
                .map_err(map_control_error)?
        {
            return Ok(Self::Attached(service));
        }

        let process_lock = StandaloneProcessLock::acquire(config)?;
        let config = config.clone();
        let mut start = tokio::task::spawn_blocking(move || {
            let capacity = config
                .writer_command_capacity()
                .map_err(|_| InspectionError::InvalidArgument)?;
            let operations =
                LedgerRepository::open_operations(&config).map_err(map_ledger_error)?;
            let OperationsLedger {
                repository,
                project_uuid,
                config_generation_id,
                pool_ids,
                control_snapshot,
                cohort_assignment,
                control_saturated,
            } = operations;
            let (owner, client) =
                LedgerWriterOwner::start(repository, capacity).map_err(map_writer_failure)?;
            Ok::<_, InspectionError>(PendingStartedWriter::new(StartedWriter {
                owner,
                client,
                project_uuid,
                config_generation_id,
                pool_ids,
                control_snapshot,
                cohort_assignment,
                control_saturated,
            }))
        });
        let pending = tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), &mut start)
            .await
            .map_err(|_| InspectionError::Busy)?
            .map_err(|_| InspectionError::StorageUnavailable)??;
        let started = pending.commit();

        if let Some(service) =
            router_control_service_for(started.project_uuid, &started.config_generation_id)
                .map_err(map_control_error)?
        {
            abandon_started_writer(started);
            return Ok(Self::Attached(service));
        }

        let StartedWriter {
            owner,
            client,
            project_uuid,
            config_generation_id,
            pool_ids,
            control_snapshot,
            cohort_assignment,
            control_saturated,
        } = started;
        let admission = ProviderAdmissionGate::new_pending();
        let activation = match admission.activation_token() {
            Ok(activation) => activation,
            Err(_) => {
                abandon_writer(owner, client);
                return Err(InspectionError::StorageUnavailable);
            }
        };
        let control_authority = match ControlRuntimeAuthority::new(
            client.clone(),
            read_pool.clone(),
            project_uuid,
            config_generation_id,
            pool_ids,
            admission.clone(),
            control_snapshot,
            Arc::new(cohort_assignment),
            control_saturated,
        ) {
            Ok(authority) => authority,
            Err(_) => {
                activation.rollback();
                abandon_writer(owner, client);
                return Err(InspectionError::IntegrityError);
            }
        };
        if control_authority.stage_idle_publication().is_err() {
            activation.rollback();
            abandon_writer(owner, client);
            return Err(InspectionError::Busy);
        }
        drop(activation);
        let service = match control_authority.service() {
            Ok(service) => service,
            Err(error) => {
                control_authority.unpublish();
                admission.close();
                abandon_writer(owner, client);
                return Err(map_control_error(error));
            }
        };
        let (heartbeat_stop, heartbeat_rx) = watch::channel(false);
        let heartbeat_client = client.clone();
        let failure_client = client.clone();
        let failure_authority = control_authority.clone();
        let failure_admission = admission.clone();
        let heartbeat = tokio::spawn(async move {
            let result = run_heartbeat(heartbeat_client, heartbeat_rx).await;
            if result.is_err() {
                failure_authority.unpublish();
                failure_admission.close();
                failure_client.abort();
            }
            result
        });
        Ok(Self::Standalone(Box::new(StandaloneOperations {
            service,
            control_authority,
            admission,
            writer_client: client,
            writer_owner: Mutex::new(Some(owner)),
            heartbeat_stop,
            heartbeat: Mutex::new(Some(heartbeat)),
            process_stop: Mutex::new(None),
            lifecycle: AtomicU8::new(LIFECYCLE_RUNNING),
            close_gate: AsyncMutex::new(()),
            _process_lock: process_lock,
        })))
    }

    fn service(&self) -> &RouterControlService {
        match self {
            Self::Attached(service) => service,
            Self::Standalone(standalone) => &standalone.service,
        }
    }

    pub(super) async fn close(&self, deadline: Instant) -> Result<(), InspectionError> {
        match self {
            Self::Attached(_) => Ok(()),
            Self::Standalone(standalone) => standalone.close(deadline).await,
        }
    }

    pub(super) fn abort(&self) {
        if let Self::Standalone(standalone) = self {
            standalone.abort();
        }
    }
}

pub(super) struct StandaloneOperations {
    service: RouterControlService,
    control_authority: ControlRuntimeAuthority,
    admission: ProviderAdmissionGate,
    writer_client: LedgerWriterClient,
    writer_owner: Mutex<Option<LedgerWriterOwner>>,
    heartbeat_stop: watch::Sender<bool>,
    heartbeat: Mutex<Option<JoinHandle<Result<(), InspectionError>>>>,
    process_stop: Mutex<Option<ProcessStop>>,
    lifecycle: AtomicU8,
    close_gate: AsyncMutex<()>,
    _process_lock: StandaloneProcessLock,
}

struct StandaloneProcessLock {
    #[cfg(unix)]
    _file: OwnedFd,
    #[cfg(windows)]
    _file: File,
}

impl StandaloneProcessLock {
    fn acquire(config: &RouterConfig) -> Result<Self, InspectionError> {
        let path = operations_lock_path(config);
        #[cfg(unix)]
        {
            use rustix::fs::{FlockOperation, Mode, OFlags, fchmod, fcntl_lock, open};
            use rustix::io::Errno;

            let file = open(
                &path,
                OFlags::RDWR | OFlags::CREATE | OFlags::CLOEXEC | OFlags::NOFOLLOW,
                Mode::RUSR | Mode::WUSR,
            )
            .map_err(|_| InspectionError::StorageUnavailable)?;
            fchmod(&file, Mode::RUSR | Mode::WUSR)
                .map_err(|_| InspectionError::StorageUnavailable)?;
            match fcntl_lock(&file, FlockOperation::NonBlockingLockExclusive) {
                Ok(()) => Ok(Self { _file: file }),
                Err(Errno::ACCESS | Errno::AGAIN) => Err(InspectionError::Busy),
                Err(_) => Err(InspectionError::StorageUnavailable),
            }
        }
        #[cfg(windows)]
        {
            use std::fs::OpenOptions;

            use windows_sys::Win32::Foundation::{ERROR_LOCK_VIOLATION, GetLastError, HANDLE};
            use windows_sys::Win32::Storage::FileSystem::LockFile;

            let file = OpenOptions::new()
                .create(true)
                .read(true)
                .write(true)
                .open(path)
                .map_err(|_| InspectionError::StorageUnavailable)?;
            let locked =
                unsafe { LockFile(file.as_raw_handle() as HANDLE, 0, 0, u32::MAX, u32::MAX) };
            if locked != 0 {
                Ok(Self { _file: file })
            } else if unsafe { GetLastError() } == ERROR_LOCK_VIOLATION {
                Err(InspectionError::Busy)
            } else {
                Err(InspectionError::StorageUnavailable)
            }
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = path;
            Err(InspectionError::StorageUnavailable)
        }
    }
}

fn operations_lock_path(config: &RouterConfig) -> PathBuf {
    let mut path = OsString::from(&config.database_path);
    path.push(".operations.lock");
    PathBuf::from(path)
}

impl StandaloneOperations {
    async fn close(&self, deadline: Instant) -> Result<(), InspectionError> {
        let _close = tokio::time::timeout_at(
            tokio::time::Instant::from_std(deadline),
            self.close_gate.lock(),
        )
        .await
        .map_err(|_| InspectionError::Busy)?;
        match self.lifecycle.load(Ordering::Acquire) {
            LIFECYCLE_CLOSED => return Ok(()),
            LIFECYCLE_ABORTED => return Err(InspectionError::StorageUnavailable),
            LIFECYCLE_RUNNING => self.lifecycle.store(LIFECYCLE_CLOSING, Ordering::Release),
            LIFECYCLE_CLOSING => return Err(InspectionError::Busy),
            _ => return Err(InspectionError::IntegrityError),
        }
        self.control_authority.unpublish();
        self.admission.close();
        self.heartbeat_stop.send_replace(true);

        let mut first_error = None;
        let heartbeat = self
            .heartbeat
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(mut heartbeat) = heartbeat {
            let result =
                tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), &mut heartbeat)
                    .await;
            match result {
                Ok(Ok(Ok(()))) => {}
                Ok(Ok(Err(error))) => first_error = Some(error),
                Ok(Err(_)) => first_error = Some(InspectionError::StorageUnavailable),
                Err(_) => {
                    heartbeat.abort();
                    first_error = Some(InspectionError::Busy);
                }
            }
        }

        match self.process_stop() {
            Ok(stop) => match self.writer_client.stop_process_until(stop, deadline).await {
                Ok(ProcessCommandAck::Applied | ProcessCommandAck::AlreadyApplied) => {}
                Ok(
                    ProcessCommandAck::Conflict
                    | ProcessCommandAck::OriginatingProcessNotLive
                    | ProcessCommandAck::TransactionNotStarted,
                ) => {
                    first_error.get_or_insert(InspectionError::IntegrityError);
                }
                Err(error) => {
                    first_error.get_or_insert(map_writer_failure(error));
                }
            },
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
        if let Err(error) = self.writer_client.flush_until(deadline).await {
            first_error.get_or_insert(map_writer_failure(error));
        }

        let owner = self
            .writer_owner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        match owner {
            Some(mut owner) => {
                if let Err(error) = owner.drain_until(deadline).await {
                    first_error.get_or_insert(map_writer_failure(error));
                }
            }
            None => {
                first_error.get_or_insert(InspectionError::StorageUnavailable);
            }
        }

        if let Some(error) = first_error {
            self.abort_resources();
            self.lifecycle.store(LIFECYCLE_ABORTED, Ordering::Release);
            Err(error)
        } else {
            self.lifecycle.store(LIFECYCLE_CLOSED, Ordering::Release);
            Ok(())
        }
    }

    fn process_stop(&self) -> Result<ProcessStop, InspectionError> {
        let mut stored = self
            .process_stop
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(command) = *stored {
            return Ok(command);
        }
        let command = ProcessStop::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            Utc::now().timestamp_millis().max(0),
        )
        .map_err(map_ledger_error)?;
        *stored = Some(command);
        Ok(command)
    }

    fn abort(&self) {
        let previous = self.lifecycle.swap(LIFECYCLE_ABORTED, Ordering::AcqRel);
        if matches!(previous, LIFECYCLE_CLOSED | LIFECYCLE_ABORTED) {
            return;
        }
        self.abort_resources();
    }

    fn abort_resources(&self) {
        self.control_authority.unpublish();
        self.admission.close();
        self.heartbeat_stop.send_replace(true);
        if let Some(heartbeat) = self
            .heartbeat
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            heartbeat.abort();
        }
        self.writer_client.abort();
        if let Some(owner) = self
            .writer_owner
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            owner.abort();
        }
    }
}

impl Drop for StandaloneOperations {
    fn drop(&mut self) {
        self.abort();
    }
}

struct StartedWriter {
    owner: LedgerWriterOwner,
    client: LedgerWriterClient,
    project_uuid: Uuid,
    config_generation_id: String,
    pool_ids: Vec<String>,
    control_snapshot: RouterControlSnapshot,
    cohort_assignment: crate::ledger::cohort::CohortAssignmentAuthority,
    control_saturated: bool,
}

struct PendingStartedWriter(Option<StartedWriter>);

impl PendingStartedWriter {
    fn new(started: StartedWriter) -> Self {
        Self(Some(started))
    }

    fn commit(mut self) -> StartedWriter {
        self.0.take().expect("pending operations writer must exist")
    }
}

impl Drop for PendingStartedWriter {
    fn drop(&mut self) {
        if let Some(started) = self.0.take() {
            abandon_started_writer(started);
        }
    }
}

fn abandon_started_writer(started: StartedWriter) {
    abandon_writer(started.owner, started.client);
}

fn abandon_writer(owner: LedgerWriterOwner, client: LedgerWriterClient) {
    owner.request_process_stop_on_abort();
    client.abort();
    owner.abort();
}

async fn run_heartbeat(
    writer: LedgerWriterClient,
    mut stop: watch::Receiver<bool>,
) -> Result<(), InspectionError> {
    let mut cadence = tokio::time::interval_at(
        tokio::time::Instant::now() + OPERATIONS_HEARTBEAT_INTERVAL,
        OPERATIONS_HEARTBEAT_INTERVAL,
    );
    cadence.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            biased;
            changed = stop.changed() => {
                if changed.is_err() || *stop.borrow() {
                    return Ok(());
                }
            }
            _ = cadence.tick() => {
                let renewal = HeartbeatRenewal::new(Utc::now().timestamp_millis().max(0))
                    .map_err(map_ledger_error)?;
                let deadline = Instant::now()
                    .checked_add(OPERATIONS_HEARTBEAT_WRITE_TIMEOUT)
                    .ok_or(InspectionError::Busy)?;
                match writer.renew_heartbeat_until(renewal, deadline).await {
                    Ok(HeartbeatAck::Applied { .. } | HeartbeatAck::AlreadyApplied { .. }) => {}
                    Ok(HeartbeatAck::OriginatingProcessNotLive | HeartbeatAck::TransactionNotStarted) => {
                        return Err(InspectionError::StorageUnavailable);
                    }
                    Err(error) => return Err(map_writer_failure(error)),
                }
            }
        }
    }
}

impl InspectionService {
    /// Apply or replay one durable pause or force-anchor mutation.
    pub async fn apply_control(
        &self,
        request: InspectionControlRequestV1,
    ) -> Result<RouterControlSnapshot, InspectionError> {
        let service = self.operations_service()?;
        service
            .apply_mutation(
                ControlMutation {
                    mutation_id: request.mutation_id,
                    scope: request.scope,
                    operation: request.operation,
                    expected_control_generation: request.expected_control_generation,
                    actor: request.actor,
                    reason: request.reason,
                },
                operation_options(),
            )
            .await
            .map_err(map_control_error)
    }

    /// Atomically reset one or every configured learning generation.
    pub async fn reset(
        &self,
        request: LearningResetRequestV1,
    ) -> Result<OperatorMutationReceiptV1, InspectionError> {
        self.operations_service()?
            .reset_learning(request, operation_options())
            .await
            .map_err(map_control_error)
    }

    /// Atomically rotate the protected cohort generation.
    pub async fn rotate_cohort(
        &self,
        request: CohortRotationRequestV1,
    ) -> Result<OperatorMutationReceiptV1, InspectionError> {
        self.operations_service()?
            .rotate_cohort(request, operation_options())
            .await
            .map_err(map_control_error)
    }

    fn operations_service(&self) -> Result<&RouterControlService, InspectionError> {
        if !self.inner.options.allow_operations {
            return Err(InspectionError::Forbidden);
        }
        self.current()?;
        self.inner
            .operations
            .as_ref()
            .map(OperationsBackend::service)
            .ok_or(InspectionError::StorageUnavailable)
    }
}

fn operation_options() -> ControlMutationOptions {
    ControlMutationOptions {
        transaction_start_timeout_ms: OPERATION_TRANSACTION_START_TIMEOUT_MS,
    }
}

fn map_control_error(error: RouterControlError) -> InspectionError {
    match error {
        RouterControlError::InvalidArgument => InspectionError::InvalidArgument,
        RouterControlError::Conflict { .. } => InspectionError::Conflict,
        RouterControlError::Unavailable | RouterControlError::StorageUnavailable => {
            InspectionError::StorageUnavailable
        }
        RouterControlError::Busy => InspectionError::Busy,
        RouterControlError::MutationExpired => InspectionError::MutationExpired,
        RouterControlError::CapacityExhausted => InspectionError::CapacityExhausted,
        RouterControlError::MigrationRequired => InspectionError::MigrationRequired,
        RouterControlError::IntegrityError => InspectionError::IntegrityError,
    }
}

fn map_writer_failure(error: WriterFailure) -> InspectionError {
    match error.class() {
        WriterFailureClass::Full | WriterFailureClass::Deadline => InspectionError::Busy,
        WriterFailureClass::Repository(class) => map_ledger_error_class(class),
        WriterFailureClass::Closing
        | WriterFailureClass::Exited
        | WriterFailureClass::Panicked
        | WriterFailureClass::Aborted
        | WriterFailureClass::Protocol => InspectionError::StorageUnavailable,
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use rusqlite::Connection;
    use serde_json::json;
    use tempfile::tempdir;

    use super::*;
    use crate::control::{ControlOperation, ControlScope, ControlTransactionFence};
    use crate::inspection::{
        InspectionServiceOptions, LearningResetScopeV1, OperatorHistoryKindV1, PageRequest,
    };
    use crate::ledger::repository::control::prepare_control_mutation;

    fn config(path: &std::path::Path, project_id: &str) -> RouterConfig {
        serde_json::from_value(json!({
            "version": 1,
            "mode": "shadow",
            "project_id": project_id,
            "database_path": path.to_string_lossy(),
            "retention_days": 30,
            "max_evidence_records": 1000,
            "pools": [{
                "id": "pool-a",
                "api_family": "openai_chat_completions",
                "anchor_models": ["anchor-a"],
                "anchor_revision": "2026-07-01",
                "sampling_probability": 0.25,
                "max_candidates_per_sample": 1,
                "concurrency": {"shadow": 2, "judge": 1},
                "judge": {
                    "version": 1,
                    "model": "judge-model",
                    "model_revision": "2026-07-01",
                    "prompt_version": "pairwise-equivalence-v1",
                    "rubric_version": "response-trajectory-equivalence-v1",
                    "output_schema_version": 1,
                    "response_weight": 0.5,
                    "trajectory_weight": 0.5,
                    "response_floor": 0.8,
                    "trajectory_floor": 0.8,
                    "judge_confidence_floor": 0.7,
                    "pass_threshold": 0.85,
                    "max_rationale_bytes": 4096,
                    "base_cooloff_seconds": 10,
                    "max_cooloff_seconds": 300
                },
                "candidates": [{
                    "id": "candidate-a",
                    "model": "candidate-model-a",
                    "model_revision": "2026-06-01",
                    "cost_rank": 0,
                    "capabilities": {}
                }]
            }]
        }))
        .unwrap()
    }

    fn table_counts(path: &std::path::Path) -> BTreeMap<String, i64> {
        let connection = Connection::open(path).unwrap();
        let tables = connection
            .prepare(
                "SELECT name FROM sqlite_schema
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        tables
            .into_iter()
            .map(|table| {
                let count = connection
                    .query_row(&format!("SELECT count(*) FROM \"{table}\""), [], |row| {
                        row.get(0)
                    })
                    .unwrap();
                (table, count)
            })
            .collect()
    }

    fn scalar_i64(path: &std::path::Path, sql: &str, value: &str) -> i64 {
        Connection::open(path)
            .unwrap()
            .query_row(sql, [value], |row| row.get(0))
            .unwrap()
    }

    fn live_process_ids(path: &std::path::Path) -> BTreeSet<String> {
        Connection::open(path)
            .unwrap()
            .prepare(
                "SELECT process.process_instance_id
                 FROM process_instances AS process
                 WHERE NOT EXISTS (
                    SELECT 1 FROM process_instance_state_events AS state
                    WHERE state.process_instance_id = process.process_instance_id
                      AND state.state = 'stopped'
                 )
                 ORDER BY process.process_instance_id",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<BTreeSet<_>>>()
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn standalone_operations_are_narrow_heartbeat_and_stop_cleanly() {
        let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("router.db");
        let config = config(&path, "standalone-operations");
        let mut activated = LedgerRepository::activate(&config).unwrap();
        activated.repository.stop_abandoned_process();
        drop(activated);
        let baseline = table_counts(&path);

        let readonly = InspectionService::open(config.clone(), InspectionServiceOptions::default())
            .await
            .unwrap();
        assert!(!readonly.operations_enabled());
        assert_eq!(
            readonly
                .apply_control(InspectionControlRequestV1 {
                    mutation_id: Uuid::now_v7(),
                    scope: ControlScope::All,
                    operation: ControlOperation::SetPaused { value: true },
                    expected_control_generation: 0,
                    actor: "operator".into(),
                    reason: "must remain read only".into(),
                })
                .await,
            Err(InspectionError::Forbidden)
        );
        readonly.close().await.unwrap();
        assert_eq!(table_counts(&path), baseline);

        let service = InspectionService::open(
            config,
            InspectionServiceOptions {
                allow_operations: true,
                request_timeout_ms: 10_000,
                ..InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        assert!(service.operations_enabled());
        match service.inner.operations.as_ref().unwrap() {
            OperationsBackend::Standalone(_) => {}
            OperationsBackend::Attached(_) => panic!("expected standalone operations writer"),
        }
        let process_id = Connection::open(&path)
            .unwrap()
            .query_row(
                "SELECT process.process_instance_id
                 FROM process_instances AS process
                 WHERE NOT EXISTS (
                    SELECT 1 FROM process_instance_state_events AS state
                    WHERE state.process_instance_id = process.process_instance_id
                      AND state.state = 'stopped'
                 )
                 ORDER BY process.started_at_unix_ms DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let opened = table_counts(&path);
        for (table, before) in &baseline {
            let expected = match table.as_str() {
                "process_instances"
                | "process_instance_state_events"
                | "process_writer_capabilities" => before + 1,
                _ => *before,
            };
            assert_eq!(opened[table], expected, "{table}");
        }

        let initial_expiry = scalar_i64(
            &path,
            "SELECT heartbeat_expires_at_unix_ms FROM process_instances
             WHERE process_instance_id = ?1",
            &process_id,
        );
        tokio::time::sleep(OPERATIONS_HEARTBEAT_INTERVAL + Duration::from_millis(50)).await;
        let renewed_expiry = scalar_i64(
            &path,
            "SELECT heartbeat_expires_at_unix_ms FROM process_instances
             WHERE process_instance_id = ?1",
            &process_id,
        );
        assert!(renewed_expiry > initial_expiry);

        let snapshot = service
            .apply_control(InspectionControlRequestV1 {
                mutation_id: Uuid::now_v7(),
                scope: ControlScope::All,
                operation: ControlOperation::SetPaused { value: true },
                expected_control_generation: 0,
                actor: "operator-a".into(),
                reason: "pause through standalone inspection".into(),
            })
            .await
            .unwrap();
        assert_eq!(snapshot.control_generation, 1);
        assert!(snapshot.all.paused);

        let reset_request = LearningResetRequestV1 {
            mutation_id: Uuid::now_v7(),
            scope: LearningResetScopeV1::Pool {
                pool_id: "pool-a".into(),
                expected_learning_generation_id: snapshot.pools["pool-a"].learning_generation_id,
            },
            confirm_project_id: "standalone-operations".into(),
            actor: "operator-a".into(),
            reason: "reset through standalone inspection".into(),
        };
        let reset = service.reset(reset_request.clone()).await.unwrap();
        assert_eq!(service.reset(reset_request).await.unwrap(), reset);
        assert_ne!(
            reset.prior_generations["pool-a"],
            reset.resulting_generations["pool-a"]
        );
        assert_eq!(
            service
                .reset(LearningResetRequestV1 {
                    mutation_id: Uuid::now_v7(),
                    scope: LearningResetScopeV1::Pool {
                        pool_id: "pool-a".into(),
                        expected_learning_generation_id: snapshot.pools["pool-a"]
                            .learning_generation_id,
                    },
                    confirm_project_id: "standalone-operations".into(),
                    actor: "operator-b".into(),
                    reason: "stale reset through standalone inspection".into(),
                })
                .await,
            Err(InspectionError::Conflict)
        );

        let rotation = service
            .rotate_cohort(CohortRotationRequestV1 {
                mutation_id: Uuid::now_v7(),
                expected_cohort_generation_id: snapshot.cohort_generation_id,
                confirm_project_id: "standalone-operations".into(),
                actor: "operator-a".into(),
                reason: "rotate through standalone inspection".into(),
            })
            .await
            .unwrap();
        assert_ne!(
            rotation.prior_generations["cohort"],
            rotation.resulting_generations["cohort"]
        );

        let history = service
            .list_controls(PageRequest {
                limit: 10,
                after: None,
            })
            .await
            .unwrap();
        assert_eq!(history.items.len(), 4);
        for kind in [
            OperatorHistoryKindV1::Pause,
            OperatorHistoryKindV1::ResetPool,
            OperatorHistoryKindV1::RotateCohort,
        ] {
            assert!(history.items.iter().any(|entry| entry.kind == kind));
        }

        service.close().await.unwrap();
        service.close().await.unwrap();
        assert_eq!(
            scalar_i64(
                &path,
                "SELECT count(*) FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'stopped'",
                &process_id,
            ),
            1
        );
        assert_eq!(
            service
                .apply_control(InspectionControlRequestV1 {
                    mutation_id: Uuid::now_v7(),
                    scope: ControlScope::All,
                    operation: ControlOperation::SetPaused { value: false },
                    expected_control_generation: 1,
                    actor: "operator".into(),
                    reason: "closed service".into(),
                })
                .await,
            Err(InspectionError::StorageUnavailable)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn standalone_abort_and_last_drop_invalidate_without_blocking_or_stopping() {
        let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("router.db");
        let config = config(&path, "standalone-abort");
        let mut activated = LedgerRepository::activate(&config).unwrap();
        activated.repository.stop_abandoned_process();
        drop(activated);
        let baseline_live = live_process_ids(&path);

        let options = InspectionServiceOptions {
            allow_operations: true,
            request_timeout_ms: 10_000,
            ..InspectionServiceOptions::default()
        };
        let aborted = InspectionService::open(config.clone(), options.clone())
            .await
            .unwrap();
        let after_open = live_process_ids(&path);
        let aborted_process = after_open
            .difference(&baseline_live)
            .next()
            .unwrap()
            .clone();
        aborted.abort();
        assert_eq!(
            aborted
                .apply_control(InspectionControlRequestV1 {
                    mutation_id: Uuid::now_v7(),
                    scope: ControlScope::All,
                    operation: ControlOperation::SetPaused { value: true },
                    expected_control_generation: 0,
                    actor: "operator".into(),
                    reason: "aborted service".into(),
                })
                .await,
            Err(InspectionError::StorageUnavailable)
        );
        assert_eq!(
            scalar_i64(
                &path,
                "SELECT count(*) FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'stopped'",
                &aborted_process,
            ),
            0
        );
        drop(aborted);

        let retained = InspectionService::open(config, options).await.unwrap();
        let retained_clone = retained.clone();
        let before_drop = live_process_ids(&path);
        let dropped_process = before_drop.difference(&after_open).next().unwrap().clone();
        drop(retained);
        let snapshot = retained_clone
            .apply_control(InspectionControlRequestV1 {
                mutation_id: Uuid::now_v7(),
                scope: ControlScope::All,
                operation: ControlOperation::SetForceAnchor { value: true },
                expected_control_generation: 0,
                actor: "operator".into(),
                reason: "last clone still owns operations".into(),
            })
            .await
            .unwrap();
        assert!(snapshot.all.force_anchor);
        drop(retained_clone);
        assert_eq!(
            crate::control::router_control_service().unwrap_err(),
            RouterControlError::Unavailable
        );
        assert_eq!(
            scalar_i64(
                &path,
                "SELECT count(*) FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'stopped'",
                &dropped_process,
            ),
            0
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn standalone_timeout_caller_drop_and_lost_reply_converge_by_mutation_id() {
        let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("router.db");
        let config = config(&path, "standalone-faults");
        let mut activated = LedgerRepository::activate(&config).unwrap();
        activated.repository.stop_abandoned_process();
        drop(activated);
        let service = InspectionService::open(
            config,
            InspectionServiceOptions {
                allow_operations: true,
                request_timeout_ms: 10_000,
                ..InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        let writer = match service.inner.operations.as_ref().unwrap() {
            OperationsBackend::Standalone(standalone) => standalone.writer_client.clone(),
            OperationsBackend::Attached(_) => panic!("expected standalone operations writer"),
        };

        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let pause_writer = writer.clone();
        let pause = tokio::spawn(async move {
            pause_writer
                .pause_until(
                    Instant::now() + Duration::from_secs(6),
                    started_tx,
                    release_rx,
                )
                .await
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let timed_out_id = Uuid::now_v7();
        assert_eq!(
            service
                .apply_control(InspectionControlRequestV1 {
                    mutation_id: timed_out_id,
                    scope: ControlScope::All,
                    operation: ControlOperation::SetPaused { value: true },
                    expected_control_generation: 0,
                    actor: "operator".into(),
                    reason: "fixed queue timeout".into(),
                })
                .await,
            Err(InspectionError::Busy)
        );

        let canceled_id = Uuid::now_v7();
        let canceled_service = service.clone();
        let canceled = tokio::spawn(async move {
            canceled_service
                .apply_control(InspectionControlRequestV1 {
                    mutation_id: canceled_id,
                    scope: ControlScope::All,
                    operation: ControlOperation::SetForceAnchor { value: true },
                    expected_control_generation: 0,
                    actor: "operator".into(),
                    reason: "drop queued caller".into(),
                })
                .await
        });
        tokio::time::sleep(Duration::from_millis(25)).await;
        canceled.abort();
        assert!(canceled.await.unwrap_err().is_cancelled());
        release_tx.send(()).unwrap();
        pause.await.unwrap().unwrap();
        writer
            .flush_until(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        for mutation_id in [timed_out_id, canceled_id] {
            assert_eq!(
                scalar_i64(
                    &path,
                    "SELECT count(*) FROM control_mutation_receipts WHERE mutation_id = ?1",
                    &mutation_id.to_string(),
                ),
                0
            );
        }

        let lost_request = InspectionControlRequestV1 {
            mutation_id: Uuid::now_v7(),
            scope: ControlScope::All,
            operation: ControlOperation::SetPaused { value: true },
            expected_control_generation: 0,
            actor: "operator".into(),
            reason: "retry a lost acknowledgement".into(),
        };
        let prepared = prepare_control_mutation(ControlMutation {
            mutation_id: lost_request.mutation_id,
            scope: lost_request.scope.clone(),
            operation: lost_request.operation.clone(),
            expected_control_generation: lost_request.expected_control_generation,
            actor: lost_request.actor.clone(),
            reason: lost_request.reason.clone(),
        })
        .unwrap();
        let fence = Arc::new(ControlTransactionFence::new(
            Instant::now() + Duration::from_secs(2),
        ));
        writer
            .apply_control_detached_for_test(prepared, fence)
            .await
            .unwrap();
        writer
            .flush_until(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(
            scalar_i64(
                &path,
                "SELECT count(*) FROM control_mutation_receipts WHERE mutation_id = ?1",
                &lost_request.mutation_id.to_string(),
            ),
            1
        );
        let replay = service.apply_control(lost_request).await.unwrap();
        assert_eq!(replay.control_generation, 1);
        assert!(replay.all.paused);
        service.close().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn concurrent_operations_services_share_one_standalone_writer() {
        let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("router.db");
        let config = config(&path, "standalone-shared");
        let mut activated = LedgerRepository::activate(&config).unwrap();
        activated.repository.stop_abandoned_process();
        drop(activated);
        let baseline_processes = table_counts(&path)["process_instances"];
        let options = InspectionServiceOptions {
            allow_operations: true,
            request_timeout_ms: 30_000,
            ..InspectionServiceOptions::default()
        };

        let (first, second) = tokio::join!(
            InspectionService::open(config.clone(), options.clone()),
            InspectionService::open(config, options),
        );
        let first = first.unwrap();
        let second = second.unwrap();
        assert_eq!(
            table_counts(&path)["process_instances"],
            baseline_processes + 1
        );
        let first_is_standalone = matches!(
            first.inner.operations.as_ref(),
            Some(OperationsBackend::Standalone(_))
        );
        let second_is_standalone = matches!(
            second.inner.operations.as_ref(),
            Some(OperationsBackend::Standalone(_))
        );
        assert_ne!(first_is_standalone, second_is_standalone);

        let snapshot = first
            .apply_control(InspectionControlRequestV1 {
                mutation_id: Uuid::now_v7(),
                scope: ControlScope::All,
                operation: ControlOperation::SetPaused { value: true },
                expected_control_generation: 0,
                actor: "operator".into(),
                reason: "shared standalone writer".into(),
            })
            .await
            .unwrap();
        assert!(snapshot.all.paused);
        let force_anchor = InspectionControlRequestV1 {
            mutation_id: Uuid::now_v7(),
            scope: ControlScope::All,
            operation: ControlOperation::SetForceAnchor { value: true },
            expected_control_generation: 1,
            actor: "operator".into(),
            reason: "attached close preserves owner".into(),
        };
        if first_is_standalone {
            second.close().await.unwrap();
            assert!(
                first
                    .apply_control(force_anchor)
                    .await
                    .unwrap()
                    .all
                    .force_anchor
            );
            first.close().await.unwrap();
        } else {
            first.close().await.unwrap();
            assert!(
                second
                    .apply_control(force_anchor)
                    .await
                    .unwrap()
                    .all
                    .force_anchor
            );
            second.close().await.unwrap();
        }
    }
}
