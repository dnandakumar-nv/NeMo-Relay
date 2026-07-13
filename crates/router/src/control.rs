// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable process-wide Router control API.

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock, Mutex, RwLock, Weak};
use std::time::{Duration, Instant};

use serde::{Deserialize, Deserializer, Serialize};
use tokio::sync::{Notify, watch};
use uuid::Uuid;

use crate::inspection::{
    CohortRotationRequestV1, InspectionError, LearningResetRequestV1, OperatorMutationReceiptV1,
    OperatorMutationResultV1,
};
use crate::ledger::cohort::CohortAssignmentAuthority;
use crate::ledger::command::{WriterFailure, WriterFailureClass};
use crate::ledger::model::LedgerErrorClass;
use crate::ledger::read_pool::{LedgerReadPool, ReadPoolError, ReadPoolErrorClass};
use crate::ledger::repository::control::{ControlMutationAck, prepare_control_mutation};
use crate::ledger::repository::inspection::operator::{
    OperatorMutationTransactionAck, PreparedOperatorMutation, prepare_cohort_rotation,
    prepare_learning_reset,
};
use crate::ledger::writer::LedgerWriterClient;
use crate::provider_admission::{ProviderAdmissionGate, ProviderStartRefusal};

/// Default and maximum SQLite transaction-start budget for one mutation.
pub const CONTROL_TRANSACTION_START_TIMEOUT_MS_MAX: u64 = 5_000;
/// Maximum UTF-8 bytes in a control actor.
pub const CONTROL_ACTOR_MAX_BYTES: usize = 128;
/// Maximum UTF-8 bytes in a control reason.
pub const CONTROL_REASON_MAX_BYTES: usize = 512;

/// Scope whose complete durable control state is replaced by one mutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControlScope {
    /// Process-wide control state for every pool.
    All,
    /// Local state for one configured pool.
    Pool {
        /// Exact configured pool identifier.
        pool_id: String,
    },
}

impl<'de> Deserialize<'de> for ControlScope {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(rename_all = "snake_case")]
        enum Kind {
            All,
            Pool,
        }

        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WireScope {
            kind: Kind,
            pool_id: Option<String>,
        }

        let wire = WireScope::deserialize(deserializer)?;
        match (wire.kind, wire.pool_id) {
            (Kind::All, None) => Ok(Self::All),
            (Kind::Pool, Some(pool_id)) => Ok(Self::Pool { pool_id }),
            (Kind::All, Some(_)) => Err(serde::de::Error::custom(
                "all control scope cannot contain pool_id",
            )),
            (Kind::Pool, None) => Err(serde::de::Error::missing_field("pool_id")),
        }
    }
}

/// One field update applied to the complete state of a control scope.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ControlOperation {
    /// Set or clear force-anchor for the selected scope.
    SetForceAnchor {
        /// Requested force-anchor value.
        value: bool,
    },
    /// Set or clear pause for the selected scope.
    SetPaused {
        /// Requested pause value.
        value: bool,
    },
}

impl ControlOperation {
    pub(crate) const fn requested_value(&self) -> bool {
        match self {
            Self::SetForceAnchor { value } | Self::SetPaused { value } => *value,
        }
    }

    pub(crate) const fn is_safety_setting(&self) -> bool {
        self.requested_value()
    }
}

/// Idempotent compare-and-set mutation request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[serde(deny_unknown_fields)]
pub struct ControlMutation {
    /// Caller-generated RFC UUIDv7 idempotency key.
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    pub mutation_id: Uuid,
    /// Complete scope to update.
    pub scope: ControlScope,
    /// Single field update.
    pub operation: ControlOperation,
    /// Latest global generation observed by the caller.
    pub expected_control_generation: u64,
    /// Bounded nonblank operator identity.
    pub actor: String,
    /// Bounded nonblank reason for the change.
    pub reason: String,
}

/// Queue and SQLite transaction-start options for one mutation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[serde(deny_unknown_fields)]
pub struct ControlMutationOptions {
    /// Milliseconds allowed for queueing and acquiring `BEGIN IMMEDIATE`.
    pub transaction_start_timeout_ms: u64,
}

impl Default for ControlMutationOptions {
    fn default() -> Self {
        Self {
            transaction_start_timeout_ms: CONTROL_TRANSACTION_START_TIMEOUT_MS_MAX,
        }
    }
}

/// Complete local state for one durable control scope.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[serde(deny_unknown_fields)]
pub struct RouterControlState {
    /// Whether candidate serving must be disabled.
    pub force_anchor: bool,
    /// Whether new learning work must stop.
    pub paused: bool,
}

/// Local and effective state for one configured pool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[serde(deny_unknown_fields)]
pub struct RouterPoolControlSnapshot {
    /// Latest pool-local state, defaulting to false/false before first use.
    pub local: RouterControlState,
    /// Logical OR of global and local state.
    pub effective: RouterControlState,
    /// Current append-only learning generation for this pool.
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    pub learning_generation_id: Uuid,
}

/// Consistent durable control and learning-generation snapshot.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[serde(deny_unknown_fields)]
pub struct RouterControlSnapshot {
    /// Latest process-wide control generation.
    pub control_generation: u64,
    /// Current protected cohort generation; the salt is never public.
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    pub cohort_generation_id: Uuid,
    /// Latest global state.
    pub all: RouterControlState,
    /// Current configured pools in lexical key order.
    pub pools: BTreeMap<String, RouterPoolControlSnapshot>,
}

/// One immutable applied control state.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[serde(deny_unknown_fields)]
pub struct ControlRecord {
    /// Unique control UUID; an applied mutation uses its mutation UUID.
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    pub control_id: Uuid,
    /// Gap-free generation of applied state changes.
    pub control_generation: u64,
    /// Mutation-history node that produced this state, or zero for genesis.
    pub originating_history_ordinal: u64,
    /// Complete scope represented by this row.
    pub scope: ControlScope,
    /// Stored force-anchor value.
    pub force_anchor: bool,
    /// Stored pause value.
    pub paused: bool,
    /// Bounded nonblank actor.
    pub actor: String,
    /// Bounded nonblank reason.
    pub reason: String,
    /// Writer process UUID.
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    pub process_id: Uuid,
    /// Writer-owned UTC time.
    pub created_at_unix_ms: u64,
    /// Domain-separated immutable record hash.
    #[serde(deserialize_with = "deserialize_sha256")]
    pub record_hash: String,
}

/// Frozen result class of one durable mutation receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum ControlMutationResult {
    /// A new control generation was appended.
    Applied,
    /// The requested value already matched the current scope state.
    NoOp,
    /// The expected generation did not match the latest generation.
    Conflict,
}

/// One immutable mutation-history node retained for idempotent replay.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[serde(deny_unknown_fields)]
pub struct ControlMutationReceipt {
    /// Caller-provided UUIDv7 idempotency key.
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    pub mutation_id: Uuid,
    /// Canonical semantic mutation hash.
    #[serde(deserialize_with = "deserialize_sha256")]
    pub canonical_payload_hash: String,
    /// Gap-free global receipt ordinal.
    pub history_ordinal: u64,
    /// Hash immediately preceding this node.
    #[serde(deserialize_with = "deserialize_sha256")]
    pub predecessor_chain_hash: String,
    /// Domain-separated hash of this complete history node.
    #[serde(deserialize_with = "deserialize_sha256")]
    pub chain_tip_hash: String,
    /// Frozen result class.
    pub result: ControlMutationResult,
    /// Global generation observed or applied by this mutation.
    pub result_control_generation: u64,
    /// Applied control UUID, present only for `applied`.
    #[cfg_attr(feature = "schema", schemars(with = "Option<String>"))]
    pub control_id: Option<Uuid>,
    /// Applied control record hash, present only for `applied`.
    #[serde(default, deserialize_with = "deserialize_optional_sha256")]
    pub control_record_hash: Option<String>,
    /// Frozen mutation scope.
    pub scope: ControlScope,
    /// Frozen requested operation.
    pub operation: ControlOperation,
    /// Caller-observed generation.
    pub expected_control_generation: u64,
    /// Bounded nonblank actor.
    pub actor: String,
    /// Bounded nonblank reason.
    pub reason: String,
    /// Writer process UUID.
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    pub process_id: Uuid,
    /// Writer-owned UTC time.
    pub created_at_unix_ms: u64,
}

/// Stable public control-service failure.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
#[serde(tag = "class", rename_all = "snake_case", deny_unknown_fields)]
pub enum RouterControlError {
    /// The request shape, bounds, UUID, or text is invalid.
    InvalidArgument,
    /// The expected generation was stale; the current snapshot is returned.
    Conflict {
        /// Fresh consistent state after the frozen conflict result.
        snapshot: RouterControlSnapshot,
    },
    /// No committed runtime owns the requested service epoch.
    Unavailable,
    /// Queueing, lock acquisition, or a bounded snapshot exceeded its deadline.
    Busy,
    /// A missing mutation UUID is outside the bounded replay horizon.
    MutationExpired,
    /// Durable control-history capacity cannot accept the request.
    CapacityExhausted,
    /// The durable store cannot currently serve the operation.
    StorageUnavailable,
    /// Schema or live-process writer capability is below `active-v6`.
    MigrationRequired,
    /// Stored control authority failed deterministic verification.
    IntegrityError,
}

impl RouterControlError {
    /// Stable machine-readable failure class.
    pub const fn code(&self) -> &'static str {
        match self {
            Self::InvalidArgument => "invalid_argument",
            Self::Conflict { .. } => "conflict",
            Self::Unavailable => "unavailable",
            Self::Busy => "busy",
            Self::MutationExpired => "mutation_expired",
            Self::CapacityExhausted => "capacity_exhausted",
            Self::StorageUnavailable => "storage_unavailable",
            Self::MigrationRequired => "migration_required",
            Self::IntegrityError => "integrity_error",
        }
    }
}

impl fmt::Display for RouterControlError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidArgument => "Router control argument is invalid",
            Self::Conflict { .. } => "Router control generation conflicted",
            Self::Unavailable => "Router control service is unavailable",
            Self::Busy => "Router control operation remained busy",
            Self::MutationExpired => "Router control mutation replay window expired",
            Self::CapacityExhausted => "Router control history capacity is exhausted",
            Self::StorageUnavailable => "Router control storage is unavailable",
            Self::MigrationRequired => "Router control writer migration is required",
            Self::IntegrityError => "Router control history failed verification",
        })
    }
}

impl Error for RouterControlError {}

/// Cloneable weak handle to the currently committed Router control runtime.
#[derive(Clone)]
pub struct RouterControlService {
    epoch: u64,
    inner: Weak<ControlRuntimeInner>,
}

impl RouterControlService {
    pub(crate) fn matches_authority(&self, project_uuid: Uuid, config_generation_id: &str) -> bool {
        self.inner.upgrade().is_some_and(|inner| {
            inner.project_uuid == project_uuid
                && inner.config_generation_id == config_generation_id
                && inner.epoch.load(Ordering::Acquire) == self.epoch
        })
    }

    /// Load a verified consistent snapshot within the fixed five-second deadline.
    pub async fn snapshot(&self) -> Result<RouterControlSnapshot, RouterControlError> {
        let inner = resolve_service_epoch(self.epoch, &self.inner)?;
        inner.snapshot_with_timeout(Duration::from_secs(5)).await
    }

    /// Apply or replay one idempotent global-generation CAS mutation.
    pub async fn apply_mutation(
        &self,
        mutation: ControlMutation,
        options: ControlMutationOptions,
    ) -> Result<RouterControlSnapshot, RouterControlError> {
        if !(1..=CONTROL_TRANSACTION_START_TIMEOUT_MS_MAX)
            .contains(&options.transaction_start_timeout_ms)
        {
            return Err(RouterControlError::InvalidArgument);
        }
        let prepared = prepare_control_mutation(mutation)?;
        let inner = resolve_service_epoch(self.epoch, &self.inner)?;
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(options.transaction_start_timeout_ms))
            .ok_or(RouterControlError::InvalidArgument)?;
        let fence = Arc::new(ControlTransactionFence::new(deadline));
        let mut cancellation = PendingControlCancellation::new(fence.clone());
        let writer = inner.writer.clone();
        let mut operation = Box::pin(writer.apply_control_mutation(prepared, fence.clone()));
        let timeout = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline));
        tokio::pin!(timeout);
        let result = tokio::select! {
            result = &mut operation => result,
            () = &mut timeout => {
                if fence.expire() {
                    cancellation.disarm();
                    return Err(RouterControlError::Busy);
                }
                operation.await
            }
        };
        cancellation.disarm();
        let result = result.map_err(map_writer_failure)??;
        match result {
            ControlMutationAck::TransactionNotStarted => match fence.state() {
                ControlFenceState::Expired => Err(RouterControlError::Busy),
                ControlFenceState::Canceled | ControlFenceState::Aborted => {
                    Err(RouterControlError::Unavailable)
                }
                ControlFenceState::Pending | ControlFenceState::Started => {
                    Err(RouterControlError::StorageUnavailable)
                }
            },
            ControlMutationAck::Completed {
                result,
                snapshot,
                saturated,
            } => {
                let snapshot = match inner.publish_acknowledged_control(snapshot, saturated) {
                    Ok(snapshot) => snapshot,
                    Err(RouterControlError::IntegrityError) => {
                        inner
                            .refresh_snapshot_with_timeout(Duration::from_secs(5), true)
                            .await?
                    }
                    Err(error) => return Err(error),
                };
                match result {
                    ControlMutationResult::Applied | ControlMutationResult::NoOp => Ok(snapshot),
                    ControlMutationResult::Conflict => {
                        Err(RouterControlError::Conflict { snapshot })
                    }
                }
            }
        }
    }

    /// Atomically reset one or every configured learning generation.
    pub async fn reset_learning(
        &self,
        request: LearningResetRequestV1,
        options: ControlMutationOptions,
    ) -> Result<OperatorMutationReceiptV1, RouterControlError> {
        let prepared = prepare_learning_reset(request).map_err(map_inspection_error)?;
        self.apply_operator_mutation(prepared, options).await
    }

    /// Atomically rotate the protected cohort generation without returning its salt.
    pub async fn rotate_cohort(
        &self,
        request: CohortRotationRequestV1,
        options: ControlMutationOptions,
    ) -> Result<OperatorMutationReceiptV1, RouterControlError> {
        let prepared = prepare_cohort_rotation(request).map_err(map_inspection_error)?;
        self.apply_operator_mutation(prepared, options).await
    }

    async fn apply_operator_mutation(
        &self,
        prepared: PreparedOperatorMutation,
        options: ControlMutationOptions,
    ) -> Result<OperatorMutationReceiptV1, RouterControlError> {
        if !(1..=CONTROL_TRANSACTION_START_TIMEOUT_MS_MAX)
            .contains(&options.transaction_start_timeout_ms)
        {
            return Err(RouterControlError::InvalidArgument);
        }
        let inner = resolve_service_epoch(self.epoch, &self.inner)?;
        let deadline = Instant::now()
            .checked_add(Duration::from_millis(options.transaction_start_timeout_ms))
            .ok_or(RouterControlError::InvalidArgument)?;
        let fence = Arc::new(ControlTransactionFence::new(deadline));
        let mut cancellation = PendingControlCancellation::new(fence.clone());
        let writer = inner.writer.clone();
        let mut operation = Box::pin(writer.apply_operator_mutation(prepared, fence.clone()));
        let timeout = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline));
        tokio::pin!(timeout);
        let result = tokio::select! {
            result = &mut operation => result,
            () = &mut timeout => {
                if fence.expire() {
                    cancellation.disarm();
                    return Err(RouterControlError::Busy);
                }
                operation.await
            }
        };
        cancellation.disarm();
        let acknowledgement = result
            .map_err(map_writer_failure)?
            .map_err(map_inspection_error)?;
        match acknowledgement {
            OperatorMutationTransactionAck::TransactionNotStarted => match fence.state() {
                ControlFenceState::Expired => Err(RouterControlError::Busy),
                ControlFenceState::Canceled | ControlFenceState::Aborted => {
                    Err(RouterControlError::Unavailable)
                }
                ControlFenceState::Pending | ControlFenceState::Started => {
                    Err(RouterControlError::StorageUnavailable)
                }
            },
            OperatorMutationTransactionAck::Completed(acknowledgement) => {
                let snapshot = inner
                    .refresh_snapshot_with_timeout(Duration::from_secs(5), true)
                    .await?;
                match acknowledgement.receipt.result {
                    OperatorMutationResultV1::Applied => Ok(acknowledgement.receipt),
                    OperatorMutationResultV1::Conflict => {
                        Err(RouterControlError::Conflict { snapshot })
                    }
                }
            }
        }
    }
}

impl fmt::Debug for RouterControlService {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RouterControlService")
            .field("epoch", &self.epoch)
            .finish_non_exhaustive()
    }
}

/// Return the control handle for the currently committed non-Off Router runtime.
pub fn router_control_service() -> Result<RouterControlService, RouterControlError> {
    let inner = resolve_published_control()?;
    Ok(RouterControlService {
        epoch: inner.epoch.load(Ordering::Acquire),
        inner: Arc::downgrade(&inner),
    })
}

pub(crate) fn router_control_service_for(
    project_uuid: Uuid,
    config_generation_id: &str,
) -> Result<Option<RouterControlService>, RouterControlError> {
    match router_control_service() {
        Ok(service) if service.matches_authority(project_uuid, config_generation_id) => {
            Ok(Some(service))
        }
        Ok(_) | Err(RouterControlError::Unavailable) => Ok(None),
        Err(error) => Err(error),
    }
}

#[derive(Clone)]
pub(crate) struct ControlRuntimeAuthority {
    inner: Arc<ControlRuntimeInner>,
}

#[derive(Clone)]
pub(crate) struct RuntimeGenerationSnapshot {
    pub(crate) control: RouterControlSnapshot,
    pub(crate) cohort_assignment: Arc<CohortAssignmentAuthority>,
}

impl RuntimeGenerationSnapshot {
    fn new(
        control: RouterControlSnapshot,
        cohort_assignment: Arc<CohortAssignmentAuthority>,
    ) -> Result<Self, &'static str> {
        if control.cohort_generation_id != cohort_assignment.cohort_generation_id()
            || control.pools.is_empty()
        {
            return Err("Router runtime generation authority is inconsistent");
        }
        Ok(Self {
            control,
            cohort_assignment,
        })
    }
}

impl ControlRuntimeAuthority {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        writer: LedgerWriterClient,
        read_pool: LedgerReadPool,
        project_uuid: Uuid,
        config_generation_id: String,
        pool_ids: Vec<String>,
        provider_admission: ProviderAdmissionGate,
        initial_snapshot: RouterControlSnapshot,
        initial_cohort_assignment: Arc<CohortAssignmentAuthority>,
        initial_saturated: bool,
    ) -> Result<Self, &'static str> {
        let initial_generation =
            RuntimeGenerationSnapshot::new(initial_snapshot, initial_cohort_assignment)?;
        let inner = Arc::new(ControlRuntimeInner {
            epoch: AtomicU64::new(0),
            writer,
            read_pool,
            project_uuid,
            config_generation_id,
            pool_ids,
            provider_admission,
            latest: RwLock::new(Some(initial_generation)),
            fail_closed: AtomicBool::new(false),
            saturated: AtomicBool::new(initial_saturated),
            poll_wake: Notify::new(),
        });
        Ok(Self { inner })
    }

    pub(crate) fn stage_publication(&self) -> Result<u64, &'static str> {
        if self.inner.epoch.load(Ordering::Acquire) != 0 {
            return Err("Router control activation was already staged");
        }
        let epoch = stage_control_inner(&self.inner)?;
        self.inner.epoch.store(epoch, Ordering::Release);
        Ok(epoch)
    }

    pub(crate) fn stage_idle_publication(&self) -> Result<u64, &'static str> {
        if self.inner.epoch.load(Ordering::Acquire) != 0 {
            return Err("Router control activation was already staged");
        }
        let epoch = stage_idle_control_inner(&self.inner)?;
        self.inner.epoch.store(epoch, Ordering::Release);
        Ok(epoch)
    }

    pub(crate) fn service(&self) -> Result<RouterControlService, RouterControlError> {
        let epoch = self.inner.epoch.load(Ordering::Acquire);
        if epoch == 0 {
            return Err(RouterControlError::Unavailable);
        }
        Ok(RouterControlService {
            epoch,
            inner: Arc::downgrade(&self.inner),
        })
    }

    pub(crate) fn unpublish(&self) {
        self.inner.unpublish();
    }

    #[allow(dead_code)] // Focused fail-closed runtime tests inspect the effective overlay directly.
    pub(crate) fn effective_state(&self, pool_id: &str) -> RouterControlState {
        self.inner.effective_state(pool_id)
    }

    pub(crate) fn current_generation_snapshot(&self) -> Option<RuntimeGenerationSnapshot> {
        self.inner.current_generation_snapshot()
    }

    pub(crate) async fn run_monitor(&self, stop: watch::Receiver<bool>) {
        self.inner.run_monitor(stop).await;
    }

    #[cfg(test)]
    pub(crate) fn epoch(&self) -> u64 {
        self.inner.epoch.load(Ordering::Acquire)
    }
}

struct ControlRuntimeInner {
    epoch: AtomicU64,
    writer: LedgerWriterClient,
    read_pool: LedgerReadPool,
    project_uuid: Uuid,
    config_generation_id: String,
    pool_ids: Vec<String>,
    provider_admission: ProviderAdmissionGate,
    latest: RwLock<Option<RuntimeGenerationSnapshot>>,
    fail_closed: AtomicBool,
    saturated: AtomicBool,
    poll_wake: Notify,
}

impl ControlRuntimeInner {
    fn publish_acknowledged_control(
        &self,
        snapshot: RouterControlSnapshot,
        saturated: bool,
    ) -> Result<RouterControlSnapshot, RouterControlError> {
        self.provider_admission
            .start_owned(|| ())
            .map_err(|_| RouterControlError::Unavailable)?;
        if !is_current_control_epoch(self.epoch.load(Ordering::Acquire), self) {
            return Err(RouterControlError::Unavailable);
        }
        let cohort_assignment = self
            .latest
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .filter(|generation| {
                generation.cohort_assignment.cohort_generation_id() == snapshot.cohort_generation_id
            })
            .map(|generation| generation.cohort_assignment.clone())
            .ok_or(RouterControlError::IntegrityError)?;
        let generation = RuntimeGenerationSnapshot::new(snapshot.clone(), cohort_assignment)
            .map_err(|_| RouterControlError::IntegrityError)?;
        self.publish_generation(generation, saturated);
        Ok(snapshot)
    }

    async fn snapshot_with_timeout(
        &self,
        timeout: Duration,
    ) -> Result<RouterControlSnapshot, RouterControlError> {
        self.refresh_snapshot_with_timeout(timeout, true).await
    }

    async fn refresh_snapshot_with_timeout(
        &self,
        timeout: Duration,
        require_current_epoch: bool,
    ) -> Result<RouterControlSnapshot, RouterControlError> {
        self.provider_admission
            .start_owned(|| ())
            .map_err(|_| RouterControlError::Unavailable)?;
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or(RouterControlError::Busy)?;
        let authority = self
            .read_pool
            .runtime_generation_snapshot_until(
                self.project_uuid,
                self.config_generation_id.clone(),
                self.pool_ids.clone(),
                deadline,
            )
            .await
            .map_err(map_read_failure)?
            .map_err(map_snapshot_ledger_failure)?
            .ok_or(RouterControlError::Unavailable)?;
        self.provider_admission
            .start_owned(|| ())
            .map_err(|_| RouterControlError::Unavailable)?;
        if require_current_epoch
            && !is_current_control_epoch(self.epoch.load(Ordering::Acquire), self)
        {
            return Err(RouterControlError::Unavailable);
        }
        let saturated = authority.control.saturated;
        let generation = RuntimeGenerationSnapshot::new(
            authority.control.snapshot,
            Arc::new(authority.cohort_assignment),
        )
        .map_err(|_| RouterControlError::IntegrityError)?;
        let snapshot = generation.control.clone();
        self.publish_generation(generation, saturated);
        Ok(snapshot)
    }

    fn publish_generation(&self, generation: RuntimeGenerationSnapshot, saturated: bool) {
        *self
            .latest
            .write()
            .unwrap_or_else(|error| error.into_inner()) = Some(generation);
        self.fail_closed.store(false, Ordering::Release);
        self.saturated.store(saturated, Ordering::Release);
        self.poll_wake.notify_one();
    }

    #[allow(dead_code)] // Focused fail-closed runtime tests inspect the effective overlay directly.
    fn effective_state(&self, pool_id: &str) -> RouterControlState {
        if self.fail_closed.load(Ordering::Acquire) {
            return RouterControlState {
                force_anchor: true,
                paused: true,
            };
        }
        let state = self
            .latest
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .and_then(|generation| generation.control.pools.get(pool_id))
            .map(|pool| pool.effective)
            .unwrap_or(RouterControlState {
                force_anchor: true,
                paused: true,
            });
        if self.saturated.load(Ordering::Acquire) {
            RouterControlState {
                force_anchor: true,
                paused: state.paused,
            }
        } else {
            state
        }
    }

    fn current_generation_snapshot(&self) -> Option<RuntimeGenerationSnapshot> {
        if self.fail_closed.load(Ordering::Acquire) || self.saturated.load(Ordering::Acquire) {
            return None;
        }
        self.latest
            .read()
            .unwrap_or_else(|error| error.into_inner())
            .clone()
    }

    async fn run_monitor(&self, mut stop: watch::Receiver<bool>) {
        let mut cadence = tokio::time::interval_at(
            tokio::time::Instant::now() + Duration::from_secs(1),
            Duration::from_secs(1),
        );
        cadence.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                biased;
                changed = stop.changed() => {
                    if changed.is_err() || *stop.borrow() {
                        return;
                    }
                }
                _ = cadence.tick() => {
                    if self
                        .refresh_snapshot_with_timeout(Duration::from_secs(1), false)
                        .await
                        .is_err()
                    {
                        self.fail_closed.store(true, Ordering::Release);
                    }
                }
                () = self.poll_wake.notified() => {}
            }
        }
    }

    fn unpublish(&self) {
        let epoch = self.epoch.load(Ordering::Acquire);
        if epoch == 0 {
            return;
        }
        let mut publication = lock_publication();
        if publication
            .pending
            .as_ref()
            .is_some_and(|slot| slot.epoch == epoch)
        {
            publication.pending = None;
        }
        if publication
            .current
            .as_ref()
            .is_some_and(|slot| slot.epoch == epoch)
        {
            publication.current = None;
        }
    }
}

impl Drop for ControlRuntimeInner {
    fn drop(&mut self) {
        self.unpublish();
    }
}

struct PendingControlCancellation {
    fence: Arc<ControlTransactionFence>,
    armed: bool,
}

impl PendingControlCancellation {
    fn new(fence: Arc<ControlTransactionFence>) -> Self {
        Self { fence, armed: true }
    }

    fn disarm(&mut self) {
        self.armed = false;
    }
}

impl Drop for PendingControlCancellation {
    fn drop(&mut self) {
        if self.armed {
            self.fence.cancel();
        }
    }
}

#[derive(Clone)]
struct PublishedControl {
    epoch: u64,
    inner: Weak<ControlRuntimeInner>,
}

#[derive(Default)]
struct ControlPublication {
    last_epoch: u64,
    pending: Option<PublishedControl>,
    current: Option<PublishedControl>,
}

static CONTROL_PUBLICATION: LazyLock<Mutex<ControlPublication>> =
    LazyLock::new(|| Mutex::new(ControlPublication::default()));

#[cfg(test)]
pub(crate) static CONTROL_PUBLICATION_TEST_MUTEX: tokio::sync::Mutex<()> =
    tokio::sync::Mutex::const_new(());

fn stage_control_inner(inner: &Arc<ControlRuntimeInner>) -> Result<u64, &'static str> {
    let mut publication = lock_publication();
    if publication
        .pending
        .as_ref()
        .and_then(|slot| slot.inner.upgrade())
        .is_some()
    {
        return Err("Router control activation is already pending");
    }
    publication.pending = None;
    let epoch = publication
        .last_epoch
        .checked_add(1)
        .ok_or("Router control activation epoch exhausted")?;
    publication.last_epoch = epoch;
    publication.pending = Some(PublishedControl {
        epoch,
        inner: Arc::downgrade(inner),
    });
    Ok(epoch)
}

fn stage_idle_control_inner(inner: &Arc<ControlRuntimeInner>) -> Result<u64, &'static str> {
    let mut publication = lock_publication();
    if publication
        .pending
        .as_ref()
        .and_then(|slot| slot.inner.upgrade())
        .is_some()
        || publication
            .current
            .as_ref()
            .and_then(|slot| slot.inner.upgrade())
            .is_some()
    {
        return Err("Router control authority is already active");
    }
    publication.pending = None;
    publication.current = None;
    let epoch = publication
        .last_epoch
        .checked_add(1)
        .ok_or("Router control activation epoch exhausted")?;
    publication.last_epoch = epoch;
    publication.pending = Some(PublishedControl {
        epoch,
        inner: Arc::downgrade(inner),
    });
    Ok(epoch)
}

fn resolve_published_control() -> Result<Arc<ControlRuntimeInner>, RouterControlError> {
    loop {
        let pending = { lock_publication().pending.clone() };
        if let Some(pending) = pending {
            let Some(inner) = pending.inner.upgrade() else {
                clear_publication_slot(pending.epoch, true);
                continue;
            };
            match inner.provider_admission.start_owned(|| ()) {
                Ok(()) => {
                    let mut publication = lock_publication();
                    if publication
                        .pending
                        .as_ref()
                        .is_some_and(|slot| slot.epoch == pending.epoch)
                    {
                        publication.pending = None;
                        publication.current = Some(pending);
                    }
                    continue;
                }
                Err(ProviderStartRefusal::Pending) => {
                    return Err(RouterControlError::Unavailable);
                }
                Err(ProviderStartRefusal::Closed) => {
                    clear_publication_slot(pending.epoch, true);
                    continue;
                }
            }
        }

        let current =
            { lock_publication().current.clone() }.ok_or(RouterControlError::Unavailable)?;
        let inner = current.inner.upgrade().ok_or_else(|| {
            clear_publication_slot(current.epoch, false);
            RouterControlError::Unavailable
        })?;
        inner.provider_admission.start_owned(|| ()).map_err(|_| {
            clear_publication_slot(current.epoch, false);
            RouterControlError::Unavailable
        })?;
        return Ok(inner);
    }
}

fn resolve_service_epoch(
    epoch: u64,
    weak: &Weak<ControlRuntimeInner>,
) -> Result<Arc<ControlRuntimeInner>, RouterControlError> {
    let current = resolve_published_control()?;
    let inner = weak.upgrade().ok_or(RouterControlError::Unavailable)?;
    if current.epoch.load(Ordering::Acquire) != epoch || !Arc::ptr_eq(&current, &inner) {
        return Err(RouterControlError::Unavailable);
    }
    Ok(inner)
}

fn is_current_control_epoch(epoch: u64, expected: &ControlRuntimeInner) -> bool {
    let Some(current) = lock_publication().current.clone() else {
        return false;
    };
    current.epoch == epoch
        && current
            .inner
            .upgrade()
            .is_some_and(|inner| std::ptr::eq(inner.as_ref(), expected))
}

fn clear_publication_slot(epoch: u64, pending: bool) {
    let mut publication = lock_publication();
    let slot = if pending {
        &mut publication.pending
    } else {
        &mut publication.current
    };
    if slot.as_ref().is_some_and(|slot| slot.epoch == epoch) {
        *slot = None;
    }
}

fn lock_publication() -> std::sync::MutexGuard<'static, ControlPublication> {
    CONTROL_PUBLICATION
        .lock()
        .unwrap_or_else(|error| error.into_inner())
}

fn map_writer_failure(error: WriterFailure) -> RouterControlError {
    match error.class() {
        WriterFailureClass::Full | WriterFailureClass::Deadline => RouterControlError::Busy,
        WriterFailureClass::Closing | WriterFailureClass::Exited | WriterFailureClass::Aborted => {
            RouterControlError::Unavailable
        }
        WriterFailureClass::Panicked
        | WriterFailureClass::Protocol
        | WriterFailureClass::Repository(_) => RouterControlError::StorageUnavailable,
    }
}

fn map_read_failure(error: ReadPoolError) -> RouterControlError {
    match error.class() {
        ReadPoolErrorClass::Deadline => RouterControlError::Busy,
        ReadPoolErrorClass::Closing | ReadPoolErrorClass::Closed | ReadPoolErrorClass::Aborted => {
            RouterControlError::Unavailable
        }
        ReadPoolErrorClass::InvalidFilesystem
        | ReadPoolErrorClass::InvalidPermissions
        | ReadPoolErrorClass::UnsupportedFilesystemSecurity
        | ReadPoolErrorClass::OpenFailed
        | ReadPoolErrorClass::ConfigurationFailed
        | ReadPoolErrorClass::WorkerFailed
        | ReadPoolErrorClass::OperationFailed
        | ReadPoolErrorClass::Invariant => RouterControlError::StorageUnavailable,
    }
}

fn map_snapshot_ledger_failure(error: crate::ledger::model::LedgerError) -> RouterControlError {
    match error.class() {
        LedgerErrorClass::FutureSchema
        | LedgerErrorClass::MigrationChecksumMismatch
        | LedgerErrorClass::InvalidMigrationHistory
        | LedgerErrorClass::MigrationFailed => RouterControlError::MigrationRequired,
        LedgerErrorClass::IdentityInvariant | LedgerErrorClass::CorruptDatabase => {
            RouterControlError::IntegrityError
        }
        _ => RouterControlError::StorageUnavailable,
    }
}

fn map_inspection_error(error: InspectionError) -> RouterControlError {
    match error {
        InspectionError::InvalidArgument
        | InspectionError::InvalidCursor
        | InspectionError::NotFound
        | InspectionError::Unauthorized
        | InspectionError::Forbidden
        | InspectionError::EgressDenied
        | InspectionError::IncompatibleApi => RouterControlError::InvalidArgument,
        InspectionError::Conflict => RouterControlError::StorageUnavailable,
        InspectionError::Busy => RouterControlError::Busy,
        InspectionError::MutationExpired => RouterControlError::MutationExpired,
        InspectionError::CapacityExhausted => RouterControlError::CapacityExhausted,
        InspectionError::StorageUnavailable => RouterControlError::StorageUnavailable,
        InspectionError::MigrationRequired => RouterControlError::MigrationRequired,
        InspectionError::IntegrityError => RouterControlError::IntegrityError,
    }
}

fn deserialize_sha256<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let value = String::deserialize(deserializer)?;
    if is_sha256(&value) {
        Ok(value)
    } else {
        Err(serde::de::Error::custom(
            "expected lowercase hexadecimal SHA-256",
        ))
    }
}

fn deserialize_optional_sha256<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<String>::deserialize(deserializer)?;
    if value.as_deref().is_none_or(is_sha256) {
        Ok(value)
    } else {
        Err(serde::de::Error::custom(
            "expected lowercase hexadecimal SHA-256",
        ))
    }
}

pub(crate) fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

const CONTROL_FENCE_PENDING: u8 = 0;
const CONTROL_FENCE_STARTED: u8 = 1;
const CONTROL_FENCE_EXPIRED: u8 = 2;
const CONTROL_FENCE_CANCELED: u8 = 3;
const CONTROL_FENCE_ABORTED: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ControlFenceState {
    Pending,
    Started,
    Expired,
    Canceled,
    Aborted,
}

pub(crate) struct ControlTransactionFence {
    state: AtomicU8,
    start_deadline: Instant,
}

impl ControlTransactionFence {
    pub(crate) fn new(start_deadline: Instant) -> Self {
        Self {
            state: AtomicU8::new(CONTROL_FENCE_PENDING),
            start_deadline,
        }
    }

    pub(crate) const fn start_deadline(&self) -> Instant {
        self.start_deadline
    }

    pub(crate) fn try_start(&self) -> bool {
        Instant::now() < self.start_deadline
            && self
                .state
                .compare_exchange(
                    CONTROL_FENCE_PENDING,
                    CONTROL_FENCE_STARTED,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
    }

    pub(crate) fn expire(&self) -> bool {
        self.transition_pending(CONTROL_FENCE_EXPIRED)
    }

    pub(crate) fn cancel(&self) -> bool {
        self.transition_pending(CONTROL_FENCE_CANCELED)
    }

    pub(crate) fn abort(&self) -> bool {
        self.transition_pending(CONTROL_FENCE_ABORTED)
    }

    pub(crate) fn state(&self) -> ControlFenceState {
        match self.state.load(Ordering::Acquire) {
            CONTROL_FENCE_PENDING => ControlFenceState::Pending,
            CONTROL_FENCE_STARTED => ControlFenceState::Started,
            CONTROL_FENCE_EXPIRED => ControlFenceState::Expired,
            CONTROL_FENCE_CANCELED => ControlFenceState::Canceled,
            CONTROL_FENCE_ABORTED => ControlFenceState::Aborted,
            _ => unreachable!("control transaction fence state is private"),
        }
    }

    fn transition_pending(&self, target: u8) -> bool {
        self.state
            .compare_exchange(
                CONTROL_FENCE_PENDING,
                target,
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .is_ok()
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    #[test]
    fn mutation_and_options_have_exact_strict_json() {
        let mutation_id = Uuid::now_v7();
        let mutation: ControlMutation = serde_json::from_value(json!({
            "mutation_id": mutation_id,
            "scope": {"kind": "pool", "pool_id": "pool-a"},
            "operation": {"kind": "set_force_anchor", "value": true},
            "expected_control_generation": 7,
            "actor": "operator",
            "reason": "incident"
        }))
        .unwrap();
        assert_eq!(
            mutation,
            ControlMutation {
                mutation_id,
                scope: ControlScope::Pool {
                    pool_id: "pool-a".to_string(),
                },
                operation: ControlOperation::SetForceAnchor { value: true },
                expected_control_generation: 7,
                actor: "operator".to_string(),
                reason: "incident".to_string(),
            }
        );
        assert_eq!(
            serde_json::to_value(&mutation).unwrap(),
            json!({
                "mutation_id": mutation_id,
                "scope": {"kind": "pool", "pool_id": "pool-a"},
                "operation": {"kind": "set_force_anchor", "value": true},
                "expected_control_generation": 7,
                "actor": "operator",
                "reason": "incident"
            })
        );
        assert_eq!(
            ControlMutationOptions::default().transaction_start_timeout_ms,
            5_000
        );
        assert!(
            serde_json::from_value::<ControlMutation>(json!({
                "mutation_id": mutation_id,
                "scope": {"kind": "all", "pool_id": "forbidden"},
                "operation": {"kind": "set_paused", "value": true},
                "expected_control_generation": 0,
                "actor": "operator",
                "reason": "incident"
            }))
            .is_err()
        );
    }

    #[test]
    fn snapshots_and_errors_are_value_semantic_and_redacted() {
        let snapshot = RouterControlSnapshot {
            control_generation: 0,
            cohort_generation_id: Uuid::now_v7(),
            all: RouterControlState::default(),
            pools: BTreeMap::from([(
                "pool-a".to_string(),
                RouterPoolControlSnapshot {
                    local: RouterControlState::default(),
                    effective: RouterControlState::default(),
                    learning_generation_id: Uuid::now_v7(),
                },
            )]),
        };
        let error = RouterControlError::Conflict {
            snapshot: snapshot.clone(),
        };
        assert_eq!(error.code(), "conflict");
        assert_eq!(error.to_string(), "Router control generation conflicted");
        assert_eq!(
            serde_json::from_value::<RouterControlError>(serde_json::to_value(&error).unwrap())
                .unwrap(),
            error
        );
        assert!(!format!("{error:?}").contains("SELECT"));
    }

    #[test]
    fn persisted_hash_fields_reject_noncanonical_values() {
        let value = json!({
            "control_id": Uuid::nil(),
            "control_generation": 0,
            "originating_history_ordinal": 0,
            "scope": {"kind": "all"},
            "force_anchor": false,
            "paused": false,
            "actor": "nemo-relay-router",
            "reason": "initialize control state v1",
            "process_id": Uuid::now_v7(),
            "created_at_unix_ms": 0,
            "record_hash": "A".repeat(64)
        });
        assert!(serde_json::from_value::<ControlRecord>(value).is_err());
    }

    #[test]
    fn transaction_start_fence_has_one_pending_winner() {
        let started =
            ControlTransactionFence::new(Instant::now() + std::time::Duration::from_secs(1));
        assert!(started.try_start());
        assert!(!started.expire());
        assert!(!started.cancel());
        assert_eq!(started.state(), ControlFenceState::Started);

        let canceled =
            ControlTransactionFence::new(Instant::now() + std::time::Duration::from_secs(1));
        assert!(canceled.cancel());
        assert!(!canceled.try_start());
        assert_eq!(canceled.state(), ControlFenceState::Canceled);

        let expired = ControlTransactionFence::new(Instant::now());
        assert!(!expired.try_start());
        assert!(expired.expire());
        assert_eq!(expired.state(), ControlFenceState::Expired);
    }
}
