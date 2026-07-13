// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded read-only SQLite connections for ledger queries.

#![allow(dead_code)] // Task 7 consumes typed reads after the SQLite sink is installed.

use std::error::Error;
use std::fmt;
use std::path::Path;
use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use rusqlite::{Connection, InterruptHandle};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, oneshot};
use uuid::Uuid;

use super::fs::{LedgerFsError, LedgerFsErrorKind, open_secure_read_connection};
use super::repository::active_learning::{
    ACTIVE_LOOK_WORK_PAGE_MAX, ActiveLookWorkCandidate, select_active_look_work,
};
use super::repository::background_work::{
    BackfillWorkCandidate, BackfillWorkCursor, EmbeddingWorkCandidate, EmbeddingWorkCursor,
    FailurePropagationCandidate, FailurePropagationCursor, MaterializationWorkCandidate,
    MaterializationWorkCursor, select_backfill_work, select_embedding_work,
    select_failure_propagation_work, select_materialization_work,
};
use super::repository::control::{
    ControlAuthoritySnapshot, RuntimeGenerationAuthority, control_config_is_current,
    load_control_authority_snapshot, load_runtime_generation_authority,
};
use super::repository::vector_work::{
    BuildingGenerationWork, RetiredGenerationWork, VectorGenerationWorkCursor,
    VectorSpaceInspection, VectorSpaceWorkCursor, inspect_vector_spaces,
    select_building_generations, select_retired_generations,
};
use crate::sqlite_vec_extension::{
    SqliteVecStatus, verify_connection as verify_sqlite_vec_connection,
};

/// Fixed number of read-only SQLite connections in the version-1 ledger pool.
pub(crate) const READ_POOL_SIZE_V1: usize = 4;

const BUSY_TIMEOUT: Duration = Duration::from_secs(5);
const STATE_OPEN: u8 = 0;
const STATE_CLOSING: u8 = 1;
const STATE_CLOSED: u8 = 2;
const STATE_ABORTED: u8 = 3;

/// Stable, non-secret class for one bounded read-pool failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReadPoolErrorClass {
    /// The secured ledger path failed structural validation.
    InvalidFilesystem,
    /// Owner-only permissions could not be verified.
    InvalidPermissions,
    /// The platform cannot enforce the required filesystem boundary.
    UnsupportedFilesystemSecurity,
    /// A read-only SQLite connection could not be opened.
    OpenFailed,
    /// Required connection settings could not be established.
    ConfigurationFailed,
    /// The pool is draining and no longer accepts work.
    Closing,
    /// The pool completed a graceful close.
    Closed,
    /// The pool was synchronously aborted.
    Aborted,
    /// The absolute operation deadline elapsed.
    Deadline,
    /// A blocking read worker exited without returning its result.
    WorkerFailed,
    /// A typed read operation failed.
    OperationFailed,
    /// The bounded permit and connection inventories diverged.
    Invariant,
}

impl ReadPoolErrorClass {
    /// Stable code suitable for internal health reporting.
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::InvalidFilesystem => "router.ledger.read.invalid_filesystem",
            Self::InvalidPermissions => "router.ledger.read.invalid_permissions",
            Self::UnsupportedFilesystemSecurity => {
                "router.ledger.read.unsupported_filesystem_security"
            }
            Self::OpenFailed => "router.ledger.read.open_failed",
            Self::ConfigurationFailed => "router.ledger.read.configuration_failed",
            Self::Closing => "router.ledger.read.closing",
            Self::Closed => "router.ledger.read.closed",
            Self::Aborted => "router.ledger.read.aborted",
            Self::Deadline => "router.ledger.read.deadline",
            Self::WorkerFailed => "router.ledger.read.worker_failed",
            Self::OperationFailed => "router.ledger.read.operation_failed",
            Self::Invariant => "router.ledger.read.invariant",
        }
    }

    const fn message(self) -> &'static str {
        match self {
            Self::InvalidFilesystem => "Router ledger reader filesystem validation failed",
            Self::InvalidPermissions => "Router ledger reader permissions are not owner-only",
            Self::UnsupportedFilesystemSecurity => {
                "Router ledger reader filesystem security is unsupported"
            }
            Self::OpenFailed => "Router ledger reader could not be opened",
            Self::ConfigurationFailed => "Router ledger reader settings could not be verified",
            Self::Closing => "Router ledger reader pool is closing",
            Self::Closed => "Router ledger reader pool is closed",
            Self::Aborted => "Router ledger reader pool was aborted",
            Self::Deadline => "Router ledger reader deadline elapsed",
            Self::WorkerFailed => "Router ledger reader worker failed",
            Self::OperationFailed => "Router ledger read operation failed",
            Self::Invariant => "Router ledger reader pool invariant failed",
        }
    }
}

/// Sanitized read-pool error that retains no path, SQL, or SQLite error text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReadPoolError {
    class: ReadPoolErrorClass,
}

impl ReadPoolError {
    const fn new(class: ReadPoolErrorClass) -> Self {
        Self { class }
    }

    /// Construct a stable failure from inside a typed read operation.
    pub(crate) const fn operation_failed() -> Self {
        Self::new(ReadPoolErrorClass::OperationFailed)
    }

    /// Return the stable failure class.
    pub(crate) const fn class(&self) -> ReadPoolErrorClass {
        self.class
    }

    /// Return the stable machine-readable code.
    pub(crate) const fn code(&self) -> &'static str {
        self.class.code()
    }
}

impl fmt::Display for ReadPoolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.class.message())
    }
}

impl Error for ReadPoolError {}

impl From<LedgerFsError> for ReadPoolError {
    fn from(error: LedgerFsError) -> Self {
        let class = match error.kind() {
            LedgerFsErrorKind::InvalidFilesystem => ReadPoolErrorClass::InvalidFilesystem,
            LedgerFsErrorKind::InvalidPermissions => ReadPoolErrorClass::InvalidPermissions,
            LedgerFsErrorKind::OpenFailed => ReadPoolErrorClass::OpenFailed,
            LedgerFsErrorKind::UnsupportedSecurity => {
                ReadPoolErrorClass::UnsupportedFilesystemSecurity
            }
        };
        Self::new(class)
    }
}

struct ReadPoolConnection {
    connection: Connection,
    sqlite_vec_status: SqliteVecStatus,
}

struct ReadPoolShared {
    connections: Mutex<Vec<ReadPoolConnection>>,
    interrupts: Vec<InterruptHandle>,
    permits: Arc<Semaphore>,
    checkout_gate: Mutex<()>,
    state: AtomicU8,
    active: AtomicUsize,
    idle: Notify,
}

/// Cloneable handle to the fixed-size Router ledger read pool.
#[derive(Clone)]
pub(crate) struct LedgerReadPool {
    shared: Arc<ReadPoolShared>,
}

impl LedgerReadPool {
    /// Open and verify all version-1 read-only connections.
    pub(crate) fn open(database_path: &Path) -> Result<Self, ReadPoolError> {
        let mut connections = Vec::with_capacity(READ_POOL_SIZE_V1);
        let mut interrupts = Vec::with_capacity(READ_POOL_SIZE_V1);
        for _ in 0..READ_POOL_SIZE_V1 {
            let connection = open_secure_read_connection(database_path)?;
            configure_connection(&connection)?;
            let sqlite_vec_status = verify_sqlite_vec_connection(&connection);
            interrupts.push(connection.get_interrupt_handle());
            connections.push(ReadPoolConnection {
                connection,
                sqlite_vec_status,
            });
        }
        Ok(Self {
            shared: Arc::new(ReadPoolShared {
                connections: Mutex::new(connections),
                interrupts,
                permits: Arc::new(Semaphore::new(READ_POOL_SIZE_V1)),
                checkout_gate: Mutex::new(()),
                state: AtomicU8::new(STATE_OPEN),
                active: AtomicUsize::new(0),
                idle: Notify::new(),
            }),
        })
    }

    /// Run one blocking read operation without exceeding the fixed connection bound.
    pub(crate) async fn run<T, F>(
        &self,
        deadline: Instant,
        operation: F,
    ) -> Result<T, ReadPoolError>
    where
        T: Send + 'static,
        F: FnOnce(&Connection) -> Result<T, ReadPoolError> + Send + 'static,
    {
        let permit = self.acquire(deadline).await?;
        let connection = {
            let _gate = lock_unpoisoned(&self.shared.checkout_gate);
            ensure_open(self.shared.state.load(Ordering::Acquire))?;
            let connection = lock_unpoisoned(&self.shared.connections)
                .pop()
                .ok_or_else(|| ReadPoolError::new(ReadPoolErrorClass::Invariant))?;
            self.shared.active.fetch_add(1, Ordering::AcqRel);
            connection
        };
        let interrupt = connection.connection.get_interrupt_handle();
        let lease = ReadConnectionLease {
            connection: Some(connection),
            permit: Some(permit),
            shared: Arc::clone(&self.shared),
        };

        if Instant::now() >= deadline {
            drop(lease);
            return Err(ReadPoolError::new(ReadPoolErrorClass::Deadline));
        }

        let (completion_tx, mut completion_rx) = oneshot::channel();
        drop(tokio::task::spawn_blocking(move || {
            let result = if Instant::now() >= deadline {
                Err(ReadPoolError::new(ReadPoolErrorClass::Deadline))
            } else {
                operation(lease.connection())
            };
            let completed_at = Instant::now();
            let _ = completion_tx.send(ReadCompletion {
                result,
                completed_at,
                lease,
            });
        }));

        let sleep = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline));
        tokio::pin!(sleep);
        tokio::select! {
            biased;
            () = &mut sleep => {
                interrupt.interrupt();
                drop(completion_rx);
                Err(ReadPoolError::new(ReadPoolErrorClass::Deadline))
            }
            completion = &mut completion_rx => {
                let ReadCompletion { result, completed_at, lease } = completion
                    .map_err(|_| ReadPoolError::new(ReadPoolErrorClass::WorkerFailed))?;
                drop(lease);
                if completed_at >= deadline || Instant::now() >= deadline {
                    Err(ReadPoolError::new(ReadPoolErrorClass::Deadline))
                } else {
                    result
                }
            }
        }
    }

    /// Discover current experiments with one unprocessed drained outcome boundary.
    pub(crate) async fn select_active_look_work_until(
        &self,
        project_uuid: Uuid,
        current_config_generation_id: String,
        limit: usize,
        deadline: Instant,
    ) -> Result<Vec<ActiveLookWorkCandidate>, ReadPoolError> {
        if limit == 0 || limit > ACTIVE_LOOK_WORK_PAGE_MAX {
            return Err(ReadPoolError::new(ReadPoolErrorClass::Invariant));
        }
        self.run(deadline, move |connection| {
            select_active_look_work(
                connection,
                project_uuid,
                &current_config_generation_id,
                limit,
            )
            .map_err(|_| ReadPoolError::operation_failed())
        })
        .await
    }

    /// Discover one bounded deterministic page of embedding provider work.
    pub(crate) async fn select_embedding_work_until(
        &self,
        project_uuid: Uuid,
        current_config_generation_id: String,
        observed_at_unix_ms: i64,
        after: Option<EmbeddingWorkCursor>,
        limit: usize,
        deadline: Instant,
    ) -> Result<Vec<EmbeddingWorkCandidate>, ReadPoolError> {
        self.run(deadline, move |connection| {
            select_embedding_work(
                connection,
                project_uuid,
                &current_config_generation_id,
                observed_at_unix_ms,
                after.as_ref(),
                limit,
            )
            .map_err(|_| ReadPoolError::operation_failed())
        })
        .await
    }

    /// Discover one bounded deterministic page of cache-ready materializations.
    pub(crate) async fn select_materialization_work_until(
        &self,
        project_uuid: Uuid,
        observed_at_unix_ms: i64,
        after: Option<MaterializationWorkCursor>,
        limit: usize,
        deadline: Instant,
    ) -> Result<Vec<MaterializationWorkCandidate>, ReadPoolError> {
        self.run(deadline, move |connection| {
            select_materialization_work(
                connection,
                project_uuid,
                observed_at_unix_ms,
                after.as_ref(),
                limit,
            )
            .map_err(|_| ReadPoolError::operation_failed())
        })
        .await
    }

    /// Discover terminal embedding jobs with an incomplete bounded dependent window.
    pub(crate) async fn select_failure_propagation_work_until(
        &self,
        project_uuid: Uuid,
        after: Option<FailurePropagationCursor>,
        limit: usize,
        deadline: Instant,
    ) -> Result<Vec<FailurePropagationCandidate>, ReadPoolError> {
        self.run(deadline, move |connection| {
            select_failure_propagation_work(connection, project_uuid, after.as_ref(), limit)
                .map_err(|_| ReadPoolError::operation_failed())
        })
        .await
    }

    /// Discover retained terminals absent from their pool's current vector space.
    pub(crate) async fn select_backfill_work_until(
        &self,
        project_uuid: Uuid,
        current_config_generation_id: String,
        after: Option<BackfillWorkCursor>,
        limit: usize,
        deadline: Instant,
    ) -> Result<Vec<BackfillWorkCandidate>, ReadPoolError> {
        self.run(deadline, move |connection| {
            select_backfill_work(
                connection,
                project_uuid,
                &current_config_generation_id,
                after.as_ref(),
                limit,
            )
            .map_err(|_| ReadPoolError::operation_failed())
        })
        .await
    }

    /// Discover one bounded snapshot page of building vector generations.
    pub(crate) async fn select_building_generations_until(
        &self,
        project_uuid: Uuid,
        process_instance_id: Uuid,
        observed_at_unix_ms: i64,
        after: Option<VectorGenerationWorkCursor>,
        limit: usize,
        deadline: Instant,
    ) -> Result<Vec<BuildingGenerationWork>, ReadPoolError> {
        self.run(deadline, move |connection| {
            let transaction = connection
                .unchecked_transaction()
                .map_err(|_| ReadPoolError::operation_failed())?;
            select_building_generations(
                &transaction,
                project_uuid,
                process_instance_id,
                observed_at_unix_ms,
                after.as_ref(),
                limit,
            )
            .map_err(|_| ReadPoolError::operation_failed())
        })
        .await
    }

    /// Discover one bounded snapshot page of retired vector generations.
    pub(crate) async fn select_retired_generations_until(
        &self,
        project_uuid: Uuid,
        process_instance_id: Uuid,
        observed_at_unix_ms: i64,
        after: Option<VectorGenerationWorkCursor>,
        limit: usize,
        deadline: Instant,
    ) -> Result<Vec<RetiredGenerationWork>, ReadPoolError> {
        self.run(deadline, move |connection| {
            let transaction = connection
                .unchecked_transaction()
                .map_err(|_| ReadPoolError::operation_failed())?;
            select_retired_generations(
                &transaction,
                project_uuid,
                process_instance_id,
                observed_at_unix_ms,
                after.as_ref(),
                limit,
            )
            .map_err(|_| ReadPoolError::operation_failed())
        })
        .await
    }

    /// Inspect one bounded keyset page of project spaces for rebuild authority.
    pub(crate) async fn inspect_vector_spaces_until(
        &self,
        project_uuid: Uuid,
        process_instance_id: Uuid,
        observed_at_unix_ms: i64,
        after: Option<VectorSpaceWorkCursor>,
        limit: usize,
        deadline: Instant,
    ) -> Result<Vec<VectorSpaceInspection>, ReadPoolError> {
        self.run(deadline, move |connection| {
            let transaction = connection
                .unchecked_transaction()
                .map_err(|_| ReadPoolError::operation_failed())?;
            inspect_vector_spaces(
                &transaction,
                project_uuid,
                process_instance_id,
                observed_at_unix_ms,
                after.as_ref(),
                limit,
            )
            .map_err(|_| ReadPoolError::operation_failed())
        })
        .await
    }

    /// Load one verified, consistent control and learning-generation snapshot.
    pub(crate) async fn control_snapshot_until(
        &self,
        project_uuid: Uuid,
        expected_config_generation_id: String,
        pool_ids: Vec<String>,
        deadline: Instant,
    ) -> Result<Result<Option<ControlAuthoritySnapshot>, super::model::LedgerError>, ReadPoolError>
    {
        self.run(deadline, move |connection| {
            let transaction = connection
                .unchecked_transaction()
                .map_err(|_| ReadPoolError::operation_failed())?;
            match control_config_is_current(
                &transaction,
                project_uuid,
                &expected_config_generation_id,
            ) {
                Ok(true) => {}
                Ok(false) => return Ok(Ok(None)),
                Err(error) => return Ok(Err(error)),
            }
            Ok(load_control_authority_snapshot(
                &transaction,
                project_uuid,
                &expected_config_generation_id,
                &pool_ids,
            )
            .map(Some))
        })
        .await
    }

    /// Load one verified control/learning/cohort snapshot with protected assignment authority.
    pub(crate) async fn runtime_generation_snapshot_until(
        &self,
        project_uuid: Uuid,
        expected_config_generation_id: String,
        pool_ids: Vec<String>,
        deadline: Instant,
    ) -> Result<Result<Option<RuntimeGenerationAuthority>, super::model::LedgerError>, ReadPoolError>
    {
        self.run(deadline, move |connection| {
            let transaction = connection
                .unchecked_transaction()
                .map_err(|_| ReadPoolError::operation_failed())?;
            match control_config_is_current(
                &transaction,
                project_uuid,
                &expected_config_generation_id,
            ) {
                Ok(true) => {}
                Ok(false) => return Ok(Ok(None)),
                Err(error) => return Ok(Err(error)),
            }
            Ok(load_runtime_generation_authority(
                &transaction,
                project_uuid,
                &expected_config_generation_id,
                &pool_ids,
            )
            .map(Some))
        })
        .await
    }

    /// Stop new reads and close every connection once active reads finish.
    pub(crate) async fn close(&self, deadline: Instant) -> Result<(), ReadPoolError> {
        {
            let _gate = lock_unpoisoned(&self.shared.checkout_gate);
            match self.shared.state.load(Ordering::Acquire) {
                STATE_OPEN => {
                    self.shared.state.store(STATE_CLOSING, Ordering::Release);
                    self.shared.permits.close();
                }
                STATE_CLOSING => {}
                STATE_CLOSED => return Ok(()),
                STATE_ABORTED => {
                    return Err(ReadPoolError::new(ReadPoolErrorClass::Aborted));
                }
                _ => return Err(ReadPoolError::new(ReadPoolErrorClass::Invariant)),
            }
        }

        loop {
            match self.shared.state.load(Ordering::Acquire) {
                STATE_CLOSING => {}
                STATE_CLOSED => return Ok(()),
                STATE_ABORTED => {
                    return Err(ReadPoolError::new(ReadPoolErrorClass::Aborted));
                }
                _ => return Err(ReadPoolError::new(ReadPoolErrorClass::Invariant)),
            }
            if self.shared.active.load(Ordering::Acquire) == 0 {
                break;
            }
            let notified = self.shared.idle.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.shared.active.load(Ordering::Acquire) == 0 {
                break;
            }
            tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), &mut notified)
                .await
                .map_err(|_| ReadPoolError::new(ReadPoolErrorClass::Deadline))?;
        }

        let _gate = lock_unpoisoned(&self.shared.checkout_gate);
        if self.shared.state.load(Ordering::Acquire) == STATE_ABORTED {
            return Err(ReadPoolError::new(ReadPoolErrorClass::Aborted));
        }
        if self.shared.active.load(Ordering::Acquire) != 0 {
            return Err(ReadPoolError::new(ReadPoolErrorClass::Invariant));
        }
        lock_unpoisoned(&self.shared.connections).clear();
        self.shared.state.store(STATE_CLOSED, Ordering::Release);
        self.shared.idle.notify_waiters();
        Ok(())
    }

    /// Interrupt active reads and synchronously prevent all future checkouts.
    pub(crate) fn abort(&self) {
        let _gate = lock_unpoisoned(&self.shared.checkout_gate);
        match self.shared.state.load(Ordering::Acquire) {
            STATE_CLOSED | STATE_ABORTED => return,
            STATE_OPEN | STATE_CLOSING => {
                self.shared.state.store(STATE_ABORTED, Ordering::Release);
            }
            _ => return,
        }
        self.shared.permits.close();
        for interrupt in &self.shared.interrupts {
            interrupt.interrupt();
        }
        lock_unpoisoned(&self.shared.connections).clear();
        self.shared.idle.notify_waiters();
    }

    async fn acquire(&self, deadline: Instant) -> Result<OwnedSemaphorePermit, ReadPoolError> {
        ensure_open(self.shared.state.load(Ordering::Acquire))?;
        let permit = tokio::time::timeout_at(
            tokio::time::Instant::from_std(deadline),
            Arc::clone(&self.shared.permits).acquire_owned(),
        )
        .await
        .map_err(|_| ReadPoolError::new(ReadPoolErrorClass::Deadline))?
        .map_err(|_| state_error(self.shared.state.load(Ordering::Acquire)))?;
        Ok(permit)
    }

    #[cfg(test)]
    fn sqlite_vec_statuses(&self) -> Vec<SqliteVecStatus> {
        lock_unpoisoned(&self.shared.connections)
            .iter()
            .map(|connection| connection.sqlite_vec_status)
            .collect()
    }
}

struct ReadConnectionLease {
    connection: Option<ReadPoolConnection>,
    permit: Option<OwnedSemaphorePermit>,
    shared: Arc<ReadPoolShared>,
}

impl ReadConnectionLease {
    fn connection(&self) -> &Connection {
        &self
            .connection
            .as_ref()
            .expect("read connection lease retains its connection")
            .connection
    }
}

impl Drop for ReadConnectionLease {
    fn drop(&mut self) {
        if let Some(connection) = self.connection.take()
            && matches!(
                self.shared.state.load(Ordering::Acquire),
                STATE_OPEN | STATE_CLOSING
            )
        {
            lock_unpoisoned(&self.shared.connections).push(connection);
        }
        let previous = self.shared.active.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous > 0, "active read count must not underflow");
        self.permit.take();
        if previous == 1 {
            self.shared.idle.notify_waiters();
        }
    }
}

struct ReadCompletion<T> {
    result: Result<T, ReadPoolError>,
    completed_at: Instant,
    lease: ReadConnectionLease,
}

fn configure_connection(connection: &Connection) -> Result<(), ReadPoolError> {
    connection
        .busy_timeout(BUSY_TIMEOUT)
        .map_err(|_| ReadPoolError::new(ReadPoolErrorClass::ConfigurationFailed))?;
    connection
        .execute_batch("PRAGMA foreign_keys = ON; PRAGMA query_only = ON;")
        .map_err(|_| ReadPoolError::new(ReadPoolErrorClass::ConfigurationFailed))?;
    let foreign_keys: i64 = connection
        .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
        .map_err(|_| ReadPoolError::new(ReadPoolErrorClass::ConfigurationFailed))?;
    let query_only: i64 = connection
        .query_row("PRAGMA query_only", [], |row| row.get(0))
        .map_err(|_| ReadPoolError::new(ReadPoolErrorClass::ConfigurationFailed))?;
    let busy_timeout: i64 = connection
        .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
        .map_err(|_| ReadPoolError::new(ReadPoolErrorClass::ConfigurationFailed))?;
    if foreign_keys != 1 || query_only != 1 || busy_timeout != BUSY_TIMEOUT.as_millis() as i64 {
        return Err(ReadPoolError::new(ReadPoolErrorClass::ConfigurationFailed));
    }
    Ok(())
}

fn ensure_open(state: u8) -> Result<(), ReadPoolError> {
    if state == STATE_OPEN {
        Ok(())
    } else {
        Err(state_error(state))
    }
}

fn state_error(state: u8) -> ReadPoolError {
    let class = match state {
        STATE_CLOSING => ReadPoolErrorClass::Closing,
        STATE_CLOSED => ReadPoolErrorClass::Closed,
        STATE_ABORTED => ReadPoolErrorClass::Aborted,
        _ => ReadPoolErrorClass::Invariant,
    };
    ReadPoolError::new(class)
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(all(test, not(target_os = "redox"), any(unix, windows)))]
mod tests {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Condvar, Mutex};
    use std::time::{Duration, Instant};

    use tempfile::TempDir;
    use tokio::sync::{mpsc, oneshot};

    use super::{LedgerReadPool, READ_POOL_SIZE_V1, ReadPoolError, ReadPoolErrorClass};
    use crate::ledger::fs::open_secure_connection;
    use crate::sqlite_vec_extension::{SqliteVecStatus, register as register_sqlite_vec};

    #[test]
    fn every_connection_records_the_exact_sqlite_vec_status() {
        assert_eq!(register_sqlite_vec(), SqliteVecStatus::Available);
        let (_temporary, pool) = test_pool();
        assert_eq!(
            pool.sqlite_vec_statuses(),
            vec![SqliteVecStatus::Available; READ_POOL_SIZE_V1]
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn missing_sqlite_vec_keeps_the_relational_read_pool_available() {
        const CHILD_ENV: &str = "NEMO_RELAY_ROUTER_READ_POOL_WITHOUT_VEC_CHILD";
        if std::env::var_os(CHILD_ENV).is_some() {
            let (_temporary, pool) = test_pool();
            assert_eq!(
                pool.sqlite_vec_statuses(),
                vec![SqliteVecStatus::VersionMissing; READ_POOL_SIZE_V1]
            );
            let value = pool
                .run(Instant::now() + Duration::from_secs(1), |connection| {
                    connection
                        .query_row("SELECT value FROM read_test", [], |row| {
                            row.get::<_, i64>(0)
                        })
                        .map_err(|_| ReadPoolError::operation_failed())
                })
                .await
                .unwrap();
            assert_eq!(value, 7);
            return;
        }

        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "ledger::read_pool::tests::missing_sqlite_vec_keeps_the_relational_read_pool_available",
                "--nocapture",
            ])
            .env(CHILD_ENV, "1")
            .status()
            .unwrap();
        assert!(status.success());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn fixed_capacity_refuses_a_fifth_reader_at_its_deadline() {
        let (_temporary, pool) = test_pool();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (entered_tx, mut entered_rx) = mpsc::unbounded_channel();
        let mut tasks = Vec::new();
        for _ in 0..READ_POOL_SIZE_V1 {
            let pool = pool.clone();
            let gate = Arc::clone(&gate);
            let entered_tx = entered_tx.clone();
            tasks.push(tokio::spawn(async move {
                pool.run(Instant::now() + Duration::from_secs(5), move |_| {
                    entered_tx
                        .send(())
                        .expect("test receiver should remain live");
                    let (lock, wake) = &*gate;
                    let mut released = lock.lock().unwrap_or_else(|error| error.into_inner());
                    while !*released {
                        released = wake
                            .wait(released)
                            .unwrap_or_else(|error| error.into_inner());
                    }
                    Ok(())
                })
                .await
            }));
        }
        for _ in 0..READ_POOL_SIZE_V1 {
            entered_rx.recv().await.expect("all readers should enter");
        }

        let error = pool
            .run(Instant::now() + Duration::from_millis(50), |_| Ok(()))
            .await
            .expect_err("the fifth read must observe bounded capacity");
        assert_eq!(error.class(), ReadPoolErrorClass::Deadline);

        let (lock, wake) = &*gate;
        *lock.lock().unwrap_or_else(|error| error.into_inner()) = true;
        wake.notify_all();
        for task in tasks {
            task.await
                .expect("reader task should not panic")
                .expect("reader should finish");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn connections_are_configured_and_remain_read_only() {
        let (_temporary, pool) = test_pool();
        let settings = pool
            .run(Instant::now() + Duration::from_secs(1), |connection| {
                let value: i64 = connection
                    .query_row("SELECT value FROM read_test", [], |row| row.get(0))
                    .map_err(|_| ReadPoolError::operation_failed())?;
                let foreign_keys: i64 = connection
                    .query_row("PRAGMA foreign_keys", [], |row| row.get(0))
                    .map_err(|_| ReadPoolError::operation_failed())?;
                let query_only: i64 = connection
                    .query_row("PRAGMA query_only", [], |row| row.get(0))
                    .map_err(|_| ReadPoolError::operation_failed())?;
                let busy_timeout: i64 = connection
                    .query_row("PRAGMA busy_timeout", [], |row| row.get(0))
                    .map_err(|_| ReadPoolError::operation_failed())?;
                Ok((value, foreign_keys, query_only, busy_timeout))
            })
            .await
            .expect("read should succeed");
        assert_eq!(settings, (7, 1, 1, 5_000));

        let error = pool
            .run(Instant::now() + Duration::from_secs(1), |connection| {
                connection
                    .execute("INSERT INTO read_test VALUES (8)", [])
                    .map(|_| ())
                    .map_err(|_| ReadPoolError::operation_failed())
            })
            .await
            .expect_err("read-only pool must reject writes");
        assert_eq!(error.class(), ReadPoolErrorClass::OperationFailed);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn absolute_deadline_bounds_an_active_blocking_operation() {
        let (_temporary, pool) = test_pool();
        let error = pool
            .run(Instant::now() + Duration::from_millis(25), |_| {
                std::thread::sleep(Duration::from_millis(100));
                Ok(())
            })
            .await
            .expect_err("active operation must observe its absolute deadline");
        assert_eq!(error.class(), ReadPoolErrorClass::Deadline);

        tokio::time::sleep(Duration::from_millis(125)).await;
        pool.run(Instant::now() + Duration::from_secs(1), |_| Ok(()))
            .await
            .expect("the timed-out worker must eventually release its lease");
    }

    #[test]
    fn queued_worker_never_invokes_an_operation_after_its_deadline() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(1)
            .max_blocking_threads(1)
            .enable_time()
            .build()
            .expect("test runtime should build");
        runtime.block_on(async {
            let (_temporary, pool) = test_pool();
            let gate = Arc::new((Mutex::new(false), Condvar::new()));
            let blocking_gate = Arc::clone(&gate);
            let (entered_tx, entered_rx) = oneshot::channel();
            let blocker = tokio::task::spawn_blocking(move || {
                entered_tx
                    .send(())
                    .expect("blocking worker entry should be observed");
                let (lock, wake) = &*blocking_gate;
                let mut released = lock.lock().unwrap_or_else(|error| error.into_inner());
                while !*released {
                    released = wake
                        .wait(released)
                        .unwrap_or_else(|error| error.into_inner());
                }
            });
            entered_rx
                .await
                .expect("the sole blocking worker should be occupied");

            let invoked = Arc::new(AtomicBool::new(false));
            let operation_invoked = Arc::clone(&invoked);
            let error = pool
                .run(Instant::now() + Duration::from_millis(25), move |_| {
                    operation_invoked.store(true, Ordering::Release);
                    Ok(())
                })
                .await
                .expect_err("queued read must expire before its worker starts");
            assert_eq!(error.class(), ReadPoolErrorClass::Deadline);

            let (lock, wake) = &*gate;
            *lock.lock().unwrap_or_else(|error| error.into_inner()) = true;
            wake.notify_all();
            blocker.await.expect("blocking worker should exit cleanly");
            pool.close(Instant::now() + Duration::from_secs(1))
                .await
                .expect("the expired queued lease should be returned");
            assert!(!invoked.load(Ordering::Acquire));
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn close_observes_active_reads_and_can_resume_after_a_deadline() {
        let (_temporary, pool) = test_pool();
        let gate = Arc::new((Mutex::new(false), Condvar::new()));
        let (entered_tx, mut entered_rx) = mpsc::unbounded_channel();
        let active_pool = pool.clone();
        let active_gate = Arc::clone(&gate);
        let active = tokio::spawn(async move {
            active_pool
                .run(Instant::now() + Duration::from_secs(5), move |_| {
                    entered_tx
                        .send(())
                        .expect("test receiver should remain live");
                    let (lock, wake) = &*active_gate;
                    let mut released = lock.lock().unwrap_or_else(|error| error.into_inner());
                    while !*released {
                        released = wake
                            .wait(released)
                            .unwrap_or_else(|error| error.into_inner());
                    }
                    Ok(())
                })
                .await
        });
        entered_rx.recv().await.expect("reader should enter");

        let error = pool
            .close(Instant::now() + Duration::from_millis(50))
            .await
            .expect_err("active reader should hold close past the first deadline");
        assert_eq!(error.class(), ReadPoolErrorClass::Deadline);
        let refused = pool
            .run(Instant::now() + Duration::from_secs(1), |_| Ok(()))
            .await
            .expect_err("closing pool must reject new work");
        assert_eq!(refused.class(), ReadPoolErrorClass::Closing);

        let (lock, wake) = &*gate;
        *lock.lock().unwrap_or_else(|error| error.into_inner()) = true;
        wake.notify_all();
        active
            .await
            .expect("reader task should not panic")
            .expect("active reader should finish");
        pool.close(Instant::now() + Duration::from_secs(1))
            .await
            .expect("second close should finish");
        pool.close(Instant::now() + Duration::from_secs(1))
            .await
            .expect("closed pool should close idempotently");
        let closed = pool
            .run(Instant::now() + Duration::from_secs(1), |_| Ok(()))
            .await
            .expect_err("closed pool must reject new work");
        assert_eq!(closed.class(), ReadPoolErrorClass::Closed);
    }

    #[tokio::test]
    async fn abort_is_synchronous_idempotent_and_wakes_future_work() {
        let (_temporary, pool) = test_pool();
        pool.abort();
        pool.abort();

        let error = pool
            .run(Instant::now() + Duration::from_secs(1), |_| Ok(()))
            .await
            .expect_err("aborted pool must reject new work");
        assert_eq!(error.class(), ReadPoolErrorClass::Aborted);
        let close_error = pool
            .close(Instant::now() + Duration::from_secs(1))
            .await
            .expect_err("an aborted pool cannot claim graceful close");
        assert_eq!(close_error.class(), ReadPoolErrorClass::Aborted);
        assert_eq!(close_error.code(), "router.ledger.read.aborted");
    }

    fn test_pool() -> (TempDir, LedgerReadPool) {
        let temporary = tempfile::tempdir().expect("tempdir should be created");
        secure_tempdir_root(&temporary);
        let database = database_path(&temporary);
        let writer = open_secure_connection(&database).expect("secure writer should open");
        writer
            .execute_batch(
                "CREATE TABLE read_test (value INTEGER NOT NULL); INSERT INTO read_test VALUES (7);",
            )
            .expect("test schema should be written");
        drop(writer);
        let pool = LedgerReadPool::open(&database).expect("read pool should open");
        (temporary, pool)
    }

    fn database_path(temporary: &TempDir) -> PathBuf {
        temporary.path().join("ledger/router.db")
    }

    #[cfg(unix)]
    fn secure_tempdir_root(temporary: &TempDir) {
        use std::os::unix::fs::PermissionsExt;

        std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
            .expect("temporary root should be owner-only");
    }

    #[cfg(windows)]
    fn secure_tempdir_root(_temporary: &TempDir) {}
}
