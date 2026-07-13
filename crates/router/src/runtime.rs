// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded Router runtime ownership and anchor-preserving V2 observation.

use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
#[cfg(test)]
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{Duration as ChronoDuration, Utc};
use nemo_relay::api::event::Event;
use nemo_relay::api::llm::{LlmCallRole, LlmExecutionContextSnapshot, LlmRequest};
use nemo_relay::api::runtime::{
    EventSubscriberFn, LlmExecutionNextFn, LlmExecutionV2Fn, LlmReplayTransport, flush_subscribers,
};
use nemo_relay::api::scope::ScopeType;
use nemo_relay::error::Result as FlowResult;
use nemo_relay::json::Json;
use nemo_relay::plugin::{PluginError, PluginRegistration, Result as PluginResult};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, mpsc, oneshot, watch};
use tokio::task::{AbortHandle, JoinHandle, JoinSet};
use uuid::Uuid;

use crate::active_planner::{
    ActiveAuditedQueryV2, ActiveCandidateExperimentAuthorityV2, ActiveGateResolverV2,
    ActiveWinnerAuthorityV2, evaluate_active_query_until,
};
use crate::active_runtime::{
    ActiveDeadlineObservationV2, ActiveOutcomeClientV2, ActiveOutcomeInvalidationAuthorityV2,
    ActiveOutcomeRuntimeErrorV2, ActiveOutcomeStageV2, ActiveRepresentativeObservationV2,
    start_active_outcome_runtime_v2,
};
use crate::adapter::FamilyAdapter;
use crate::background::{
    BackgroundAbortHandle, BackgroundFailure, BackgroundFailureNotifier, BackgroundResources,
    BackgroundRuntime,
};
use crate::background_driver::RouterBackgroundDriver;
use crate::canonical_json::canonical_sha256;
use crate::config::{
    OutcomeConfig, PoolConfig, RouterConfig, RouterMode, protected_outcome_policy_version_v1,
};
use crate::control::{ControlRuntimeAuthority, RouterControlSnapshot, RuntimeGenerationSnapshot};
use crate::coordinator::{
    AnchorRefusalReason, AnchorRegistration, Coordinator, CoordinatorCommand, CoordinatorEffect,
    CoordinatorLimits, LossSnapshot, PrimaryCallRegistration, WindowLimits,
};
use crate::decision_audit::{ActiveDecisionParentBindingV2, DecisionFinalReasonV1};
use crate::eligibility::IneligibilityReason;
use crate::embedder::{FrozenEmbedderClients, build_frozen_embedder_clients};
use crate::evaluator::{ActiveReplayRegistry, BackgroundStartGate, EvaluatorCancellation};
use crate::health::RouterHealth;
use crate::ledger::cohort::{
    CohortAssignmentAuthority, CohortAssignmentRequest, RandomizedCohort,
    RandomizedCohortAssignment,
};
use crate::ledger::command::{WriterFailure, WriterFailureClass};
use crate::ledger::model::{LedgerErrorClass, LedgerRuntimeIdentity};
use crate::ledger::read_pool::LedgerReadPool;
use crate::ledger::repository::active::{
    ActiveAssignmentArm, ActiveDispatchAdmission, ActiveDispatchTerminal,
    ActiveDispatchTerminalAck, ActiveDispatchTerminalState, ActiveRootAdmission,
};
use crate::ledger::repository::active_decision::{
    ActiveDecisionAdmissionAckV2, ActiveDecisionAdmissionV2, ActiveDecisionFactsV2,
    ActivePlannedRouteV2,
};
use crate::ledger::repository::active_learning::{
    ActiveExperimentCreate, ActiveExperimentCreateAck, ActiveNeighborhoodInvalidation,
    ActiveNeighborhoodMutationAck,
};
use crate::ledger::repository::process::{
    HeartbeatAck, HeartbeatRenewal, ProcessCommandAck, ProcessStop,
};
use crate::ledger::repository::retention::{RetentionAck, RetentionRequest};
use crate::ledger::repository::vector_index::{
    VectorIndexHealthMutationAck, VectorIndexHealthTarget,
};
use crate::ledger::repository::vector_registry::VectorRegistryEnsure;
use crate::ledger::repository::{ActivatedLedger, LedgerRepository, SchemaVerificationReport};
use crate::ledger::writer::{ActiveDispatchTerminalPermit, LedgerWriterClient, LedgerWriterOwner};
use crate::live_embedding::LiveEmbeddingService;
use crate::matcher::PoolMatcher;
use crate::outcome::{CompiledOutcomePolicyV1, RepresentativeResultV1, RepresentativeTerminalV1};
use crate::preflight::{EligibleCandidate, PreflightOutcome, preflight};
use crate::provider_admission::ProviderAdmissionGate;
use crate::recommendation::RecommendationErrorV1;
use crate::recommendation_delivery::{
    RecommendationDeliveryServiceV1, RecommendationFirstAckV1, RecommendationStaleReasonV1,
};
use crate::recommendation_runtime::{
    RecommendationRuntimeIdentityV1, compute_recommendation_audit_until,
    panic_safe_recommendation_future, prepare_active_runtime_query_v2, rewrite_active_winner_v2,
};
use crate::sampling::{OsSampler, Sampler, should_sample};
use crate::scheduler::{SchedulerDeadlineAuthority, SchedulerExit, ShadowScheduler};
use crate::scheduler_admission::SchedulerAdmissionPools;
use crate::sink::{DeliveryAck, SinkAck, TrajectoryDelivery, TrajectorySink};
#[cfg(test)]
use crate::sink::{InMemoryTrajectoryDelivery, InMemoryTrajectorySink};
use crate::sqlite_sink::SqliteTrajectorySink;
use crate::sqlite_vector_store::{SqliteVecStore, VectorIndexWriterAck, VectorIndexWriterCommand};
use crate::trajectory::{
    EventProjectionLimits, PendingTrajectoryWindow, ProjectedTrajectoryEvent,
    ReplayCapabilityFactsV1, TrajectoryIdentity, TrajectoryWindowSeed, project_anchor_response,
    project_candidate_facts, project_captured_event, project_owner_path,
    truncate_utc_to_milliseconds,
};

const SINK_RETRY_BASE: Duration = Duration::from_millis(25);
const SINK_RETRY_MAX: Duration = Duration::from_secs(1);
const HEARTBEAT_INTERVAL: Duration = Duration::from_secs(5);
const HEARTBEAT_WRITE_TIMEOUT: Duration = Duration::from_secs(5);
const RETENTION_INTERVAL: Duration = Duration::from_secs(60 * 60);
const RETENTION_WRITE_TIMEOUT: Duration = Duration::from_secs(5);
const RETENTION_RETRY_DELAY: Duration = Duration::from_secs(1);
const RETENTION_BATCH_ROW_LIMIT: u64 = 1_000;
const ACTIVATION_ROLLBACK_TIMEOUT: Duration = Duration::from_secs(5);
const BACKGROUND_TASK_LIMIT: usize = 256;
const HEALTH_HEARTBEAT_FAILURE: &str = "router.ledger.heartbeat_failure";
const HEALTH_RETENTION_FAILURE: &str = "router.ledger.retention_failure";
const HEALTH_RECOMMENDATION_ADMISSION_FAILURE: &str = "router.recommendation.admission_failure";
const HEALTH_RECOMMENDATION_COMPUTE_FAILURE: &str = "router.recommendation.compute_failure";
const HEALTH_RECOMMENDATION_DELIVERY_PENDING: &str = "router.recommendation.delivery_pending";
const HEALTH_RECOMMENDATION_DELIVERY_FAILURE: &str = "router.recommendation.delivery_failure";
const HEALTH_ACTIVE_ADMISSION_FAILURE: &str = "router.active.admission_failure";
const HEALTH_ACTIVE_COMPUTE_FAILURE: &str = "router.active.compute_failure";
const HEALTH_ACTIVE_OUTCOME_FAILURE: &str = "router.active.outcome_failure";
const LIFECYCLE_RUNNING: u8 = 0;
const LIFECYCLE_DRAINING: u8 = 1;
const LIFECYCLE_ABORTED: u8 = 2;
const LIFECYCLE_DRAINED: u8 = 3;

fn trajectory_identity_from_ledger(identity: &LedgerRuntimeIdentity) -> TrajectoryIdentity {
    TrajectoryIdentity {
        process_instance_id: identity.process_instance_id,
        project_uuid: identity.project_uuid,
        project_id: identity.project_id.clone(),
        policy_version_ids: identity
            .pools
            .iter()
            .map(|(pool_id, pool)| (pool_id.clone(), pool.policy_version_id.clone()))
            .collect(),
        learning_generation_ids: identity
            .pools
            .iter()
            .map(|(pool_id, pool)| (pool_id.clone(), pool.learning_generation_id))
            .collect(),
    }
}

fn trajectory_identity_for_generation(
    identity: &TrajectoryIdentity,
    generation: &RuntimeGenerationSnapshot,
) -> Option<TrajectoryIdentity> {
    if identity
        .policy_version_ids
        .keys()
        .ne(generation.control.pools.keys())
    {
        return None;
    }
    let mut current = identity.clone();
    current.learning_generation_ids = generation
        .control
        .pools
        .iter()
        .map(|(pool_id, pool)| (pool_id.clone(), pool.learning_generation_id))
        .collect();
    Some(current)
}

fn ledger_identity_for_generation(
    identity: &LedgerRuntimeIdentity,
    generation: &RuntimeGenerationSnapshot,
) -> Option<LedgerRuntimeIdentity> {
    if identity.pools.keys().ne(generation.control.pools.keys()) {
        return None;
    }
    let mut current = identity.clone();
    for (pool_id, pool) in &mut current.pools {
        pool.learning_generation_id = generation
            .control
            .pools
            .get(pool_id)?
            .learning_generation_id;
    }
    current.cohort_generation_id = generation.control.cohort_generation_id;
    Some(current)
}

struct ActivatedRepositoryGuard {
    repository: Option<LedgerRepository>,
}

impl ActivatedRepositoryGuard {
    fn new(repository: LedgerRepository) -> Self {
        Self {
            repository: Some(repository),
        }
    }

    fn take(&mut self) -> LedgerRepository {
        self.repository
            .take()
            .expect("activated repository must be owned")
    }
}

impl Drop for ActivatedRepositoryGuard {
    fn drop(&mut self) {
        if let Some(repository) = self.repository.as_mut() {
            repository.stop_abandoned_process();
        }
    }
}

type BarrierFuture = Pin<Box<dyn Future<Output = Result<(), ()>> + Send + 'static>>;

trait SubscriberBarrier: Send + Sync {
    fn flush(&self) -> BarrierFuture;
}

struct CoreSubscriberBarrier;

impl SubscriberBarrier for CoreSubscriberBarrier {
    fn flush(&self) -> BarrierFuture {
        Box::pin(async {
            tokio::task::spawn_blocking(flush_subscribers)
                .await
                .map_err(|_| ())?
                .map_err(|_| ())
        })
    }
}

#[derive(Debug)]
struct PoolRuntimeResources {
    sample_intents: Arc<Semaphore>,
    _shadow: Arc<Semaphore>,
    _judge: Arc<Semaphore>,
}

#[derive(Clone)]
struct SchedulerRuntimeControls {
    start_gate: BackgroundStartGate,
    active_replays: ActiveReplayRegistry,
    cancellation: EvaluatorCancellation,
    deadlines: SchedulerDeadlineAuthority,
    admissions: SchedulerAdmissionPools,
}

struct HeartbeatRuntime {
    stop: watch::Sender<bool>,
    task: JoinHandle<Result<(), &'static str>>,
}

struct RetentionRuntime {
    stop: watch::Sender<bool>,
    pressure: watch::Sender<Option<u64>>,
    #[allow(dead_code)] // The bounded manual trigger is exposed only to deterministic tests today.
    manual: mpsc::Sender<RetentionManualRequest>,
    task: JoinHandle<Result<(), &'static str>>,
}

struct ControlMonitorRuntime {
    stop: watch::Sender<bool>,
    task: JoinHandle<()>,
}

struct RetentionManualRequest {
    observed_at_unix_ms: i64,
    reply: oneshot::Sender<Result<RetentionAck, &'static str>>,
}

struct RuntimeAbortHandles {
    coordinator: AbortHandle,
    active_outcome: Option<AbortHandle>,
    scheduler: Option<AbortHandle>,
    background: Option<BackgroundAbortHandle>,
    retention: Option<AbortHandle>,
    heartbeat: Option<AbortHandle>,
}

impl PoolRuntimeResources {
    fn new(pool: &PoolConfig) -> Self {
        Self {
            sample_intents: Arc::new(Semaphore::new(pool.concurrency.max_pending)),
            _shadow: Arc::new(Semaphore::new(pool.concurrency.shadow)),
            _judge: Arc::new(Semaphore::new(pool.concurrency.judge)),
        }
    }
}

#[derive(Debug, Default)]
struct LossState {
    ingest_seq: AtomicU64,
    highest_dropped_ingest_seq: AtomicU64,
    classification_loss_epoch: AtomicU64,
}

impl LossState {
    fn next_ingest_seq(&self) -> Option<u64> {
        self.ingest_seq
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
                current.checked_add(1)
            })
            .ok()
            .and_then(|previous| previous.checked_add(1))
    }

    fn current_ingest_seq(&self) -> u64 {
        self.ingest_seq.load(Ordering::Acquire)
    }

    fn record_event_loss(&self, ingest_seq: u64) {
        self.highest_dropped_ingest_seq
            .fetch_max(ingest_seq, Ordering::AcqRel);
    }

    fn record_classification_loss(&self) {
        let _ = self.classification_loss_epoch.fetch_update(
            Ordering::AcqRel,
            Ordering::Acquire,
            |current| Some(current.saturating_add(1)),
        );
    }

    fn snapshot(&self) -> LossSnapshot {
        LossSnapshot {
            highest_dropped_ingest_seq: self.highest_dropped_ingest_seq.load(Ordering::Acquire),
            classification_loss_epoch: self.classification_loss_epoch.load(Ordering::Acquire),
        }
    }
}

struct RuntimeState {
    config: Arc<RouterConfig>,
    matcher: PoolMatcher,
    adapter: FamilyAdapter,
    sampler: Arc<dyn Sampler>,
    pool_resources: BTreeMap<String, Arc<PoolRuntimeResources>>,
    identity: TrajectoryIdentity,
    ledger_identity: Option<LedgerRuntimeIdentity>,
    config_generation_id: String,
    max_event_projection_bytes: usize,
    accepting: AtomicBool,
    event_ingestion: AtomicBool,
    sampling_enabled: AtomicBool,
    permanent_runtime_fault: AtomicBool,
    admission_epoch: AtomicU64,
    admission: Mutex<()>,
    loss: Arc<LossState>,
    coordinator_tx: mpsc::Sender<CoordinatorCommand>,
    health: Arc<RouterHealth>,
    writer_client: Option<LedgerWriterClient>,
    retention_pressure: Option<watch::Sender<Option<u64>>>,
    read_pool: Option<LedgerReadPool>,
    scheduler_controls: Option<SchedulerRuntimeControls>,
    provider_admission: ProviderAdmissionGate,
    foreground_admission: Arc<Semaphore>,
    subscriber_barrier: Arc<dyn SubscriberBarrier>,
    active_outcome: Option<ActiveOutcomeClientV2>,
    control_authority: Option<ControlRuntimeAuthority>,
    live_embedding: Option<LiveEmbeddingService>,
    recommendation_delivery: Option<RecommendationDeliveryServiceV1>,
    recommendation_vector_store: Option<SqliteVecStore>,
    vector_registry: Option<Arc<VectorRegistryEnsure>>,
    embedder_clients: Option<Arc<FrozenEmbedderClients>>,
    initial_vector_schema_report: SchemaVerificationReport,
    finished: AtomicBool,
    finished_notify: Notify,
    #[cfg(test)]
    ownership_probe: Arc<()>,
    #[cfg(test)]
    owned_task_count: AtomicUsize,
    #[cfg(test)]
    peak_owned_task_count: AtomicUsize,
    #[cfg(test)]
    heartbeat_ticks: AtomicUsize,
    #[cfg(test)]
    heartbeat_worker_ready: AtomicBool,
    #[cfg(test)]
    retention_ticks: AtomicUsize,
    #[cfg(test)]
    retention_already_applied_ticks: AtomicUsize,
    #[cfg(test)]
    retention_worker_ready: AtomicBool,
    #[cfg(test)]
    retention_timer_fires: AtomicUsize,
    #[cfg(test)]
    retention_recoveries: Mutex<Vec<u64>>,
}

struct PreparedSample {
    call: PreparedCallIdentity,
    pool: PreparedPool,
    envelope: crate::adapter::RouterRequestEnvelope,
    request_projection: crate::projection::RouterRequestProjectionV1,
    routing_projection: crate::projection::RouterRoutingContextProjectionV1,
    candidates: Vec<EligibleCandidate>,
    replay_transport: Arc<dyn LlmReplayTransport>,
    replay_facts: ReplayCapabilityFactsV1,
    candidate_facts: Vec<crate::trajectory::PersistedCandidateFactV1>,
    owner_path: Vec<crate::trajectory::TrajectoryOwnerScopeV1>,
}

struct PreparedPool {
    id: String,
    anchor_model_revision: String,
    sampling_probability: f64,
    requested_progress: usize,
    deadline_seconds: u64,
    lifecycle_presets: Vec<String>,
    max_events: usize,
    max_bytes: usize,
}

#[derive(Clone, Copy)]
struct PreparedCallIdentity {
    call_uuid: Uuid,
    root_uuid: Uuid,
    owner_uuid: Uuid,
    api_family: nemo_relay::api::llm::LlmApiFamily,
}

struct SampleIntent {
    prepared: PreparedSample,
    generation_snapshot: Option<RouterControlSnapshot>,
    admission_epoch: u64,
    proposal_permit: OwnedSemaphorePermit,
}

struct ActiveAdmittedCallV2 {
    active_root_window_id: Uuid,
    active_dispatch_id: Option<Uuid>,
    raw_root_uuid: Uuid,
    winner: ActiveWinnerAuthorityV2,
    relearning_cooloff_seconds: u32,
    api_family: nemo_relay::api::llm::LlmApiFamily,
}

enum PreparedActiveRouteV2 {
    Candidate {
        request: LlmRequest,
        admitted: ActiveAdmittedCallV2,
        dispatch_guard: ActiveDispatchGuardV2,
    },
    Anchor {
        admitted: ActiveAdmittedCallV2,
    },
    PostAdmissionFailure {
        dispatch_guard: Option<ActiveDispatchGuardV2>,
    },
}

pub(crate) struct ActiveDispatchGuardV2 {
    permit: Option<ActiveDispatchTerminalPermit>,
    active_dispatch_terminal_event_id: Uuid,
    active_dispatch_id: Uuid,
    active_assignment_id: Uuid,
    handed_off_at_unix_ms: Option<u64>,
}

impl ActiveDispatchGuardV2 {
    pub(crate) fn new(
        permit: ActiveDispatchTerminalPermit,
        active_dispatch_id: Uuid,
        active_assignment_id: Uuid,
    ) -> Self {
        Self {
            permit: Some(permit),
            active_dispatch_terminal_event_id: Uuid::now_v7(),
            active_dispatch_id,
            active_assignment_id,
            handed_off_at_unix_ms: None,
        }
    }

    fn mark_handed_off(&mut self, handed_off_at_unix_ms: u64) {
        self.handed_off_at_unix_ms = Some(handed_off_at_unix_ms);
    }

    async fn finish(
        mut self,
        state: ActiveDispatchTerminalState,
        stable_error_class: Option<String>,
        provider_receipt_hash: Option<String>,
        deadline: Instant,
    ) -> Result<ActiveDispatchTerminalAck, WriterFailure> {
        let permit = self
            .permit
            .take()
            .ok_or_else(|| WriterFailure::new(WriterFailureClass::Protocol))?;
        permit
            .record_until(
                self.terminal(state, stable_error_class, provider_receipt_hash),
                deadline,
            )
            .await
    }

    fn terminal(
        &self,
        state: ActiveDispatchTerminalState,
        stable_error_class: Option<String>,
        provider_receipt_hash: Option<String>,
    ) -> ActiveDispatchTerminal {
        ActiveDispatchTerminal {
            active_dispatch_terminal_event_id: self.active_dispatch_terminal_event_id,
            active_dispatch_id: self.active_dispatch_id,
            active_assignment_id: self.active_assignment_id,
            terminal_state: state,
            stable_error_class,
            provider_receipt_hash,
            handed_off_at_unix_ms: self.handed_off_at_unix_ms,
        }
    }
}

impl Drop for ActiveDispatchGuardV2 {
    fn drop(&mut self) {
        let Some(permit) = self.permit.take() else {
            return;
        };
        let state = if self.handed_off_at_unix_ms.is_some() {
            ActiveDispatchTerminalState::CancelledAfterHandoff
        } else {
            ActiveDispatchTerminalState::CancelledBeforeHandoff
        };
        permit.record_detached(self.terminal(state, None, None));
    }
}

/// Owner for all bounded state associated with one configured Router component.
pub(crate) struct RouterRuntime {
    state: Arc<RuntimeState>,
    lifecycle_gate: Mutex<()>,
    lifecycle: AtomicU8,
    process_stop: Mutex<Option<ProcessStop>>,
    abort_handles: RuntimeAbortHandles,
    coordinator: Mutex<Option<JoinHandle<()>>>,
    active_outcome: Mutex<Option<JoinHandle<()>>>,
    scheduler: Mutex<Option<JoinHandle<SchedulerExit>>>,
    background: Mutex<Option<BackgroundRuntime>>,
    retention: Mutex<Option<RetentionRuntime>>,
    heartbeat: Mutex<Option<HeartbeatRuntime>>,
    control_monitor: Mutex<Option<ControlMonitorRuntime>>,
    writer: Mutex<Option<LedgerWriterOwner>>,
    #[cfg(test)]
    drain_retained_owners_pause:
        Mutex<Option<(Arc<tokio::sync::Barrier>, Arc<tokio::sync::Barrier>)>>,
}

struct RetainedResource<'a, T> {
    slot: &'a Mutex<Option<T>>,
    lifecycle: &'a AtomicU8,
    value: Option<T>,
}

struct PanicSafeArcOwner<T: ?Sized> {
    value: Option<Arc<T>>,
}

impl<T: ?Sized> PanicSafeArcOwner<T> {
    fn new(value: Arc<T>) -> Self {
        Self { value: Some(value) }
    }

    fn get(&self) -> &Arc<T> {
        self.value
            .as_ref()
            .expect("panic-safe Arc owner used after drop")
    }

    fn take(&mut self) -> Arc<T> {
        self.value
            .take()
            .expect("panic-safe Arc owner used after drop")
    }

    fn drop_value(&mut self) {
        if let Some(value) = self.value.take() {
            let _ = catch_unwind(AssertUnwindSafe(|| drop(value)));
        }
    }
}

impl<T: ?Sized> Drop for PanicSafeArcOwner<T> {
    fn drop(&mut self) {
        self.drop_value();
    }
}

impl<'a, T> RetainedResource<'a, T> {
    fn take(slot: &'a Mutex<Option<T>>, lifecycle: &'a AtomicU8) -> Option<Self> {
        let value = slot
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()?;
        Some(Self {
            slot,
            lifecycle,
            value: Some(value),
        })
    }

    fn value(&self) -> &T {
        self.value.as_ref().expect("retained resource must exist")
    }

    fn value_mut(&mut self) -> &mut T {
        self.value.as_mut().expect("retained resource must exist")
    }

    fn finish(mut self) {
        self.value.take();
    }
}

impl<T> RetainedResource<'_, JoinHandle<T>> {
    async fn join(mut self) -> Result<T, tokio::task::JoinError> {
        let result = self.value_mut().await;
        self.value.take();
        result
    }
}

impl<T> Drop for RetainedResource<'_, T> {
    fn drop(&mut self) {
        let Some(value) = self.value.take() else {
            return;
        };
        let mut slot = self.slot.lock().unwrap_or_else(|error| error.into_inner());
        if self.lifecycle.load(Ordering::Acquire) == LIFECYCLE_ABORTED {
            drop(slot);
            drop(value);
            return;
        }
        assert!(slot.is_none(), "retained Router resource slot was replaced");
        *slot = Some(value);
    }
}

impl RouterRuntime {
    #[cfg(test)]
    pub(crate) fn start(config: RouterConfig) -> Result<Arc<Self>, String> {
        let max_evidence_records =
            usize::try_from(config.max_evidence_records).unwrap_or(usize::MAX);
        Self::start_with_dependencies(
            config,
            Arc::new(OsSampler),
            Arc::new(InMemoryTrajectorySink::new(max_evidence_records)),
            Arc::new(InMemoryTrajectoryDelivery::new(max_evidence_records)),
            Arc::new(CoreSubscriberBarrier),
        )
    }

    #[cfg_attr(test, allow(dead_code))]
    pub(crate) fn start_with_activated_ledger(
        config: RouterConfig,
        activated_ledger: ActivatedLedger,
    ) -> Result<Arc<Self>, String> {
        Self::start_with_dependencies_and_activation(
            config,
            Arc::new(OsSampler),
            None,
            Arc::new(CoreSubscriberBarrier),
            None,
            Some(activated_ledger),
        )
    }

    #[cfg(test)]
    pub(crate) fn start_with_test_storage(
        config: RouterConfig,
        sink: Arc<InMemoryTrajectorySink>,
        delivery: Arc<InMemoryTrajectoryDelivery>,
    ) -> Result<Arc<Self>, String> {
        Self::start_with_dependencies(
            config,
            Arc::new(OsSampler),
            sink,
            delivery,
            Arc::new(CoreSubscriberBarrier),
        )
    }

    #[cfg(test)]
    fn start_with_dependencies(
        config: RouterConfig,
        sampler: Arc<dyn Sampler>,
        sink: Arc<dyn TrajectorySink>,
        delivery: Arc<dyn TrajectoryDelivery>,
        barrier: Arc<dyn SubscriberBarrier>,
    ) -> Result<Arc<Self>, String> {
        Self::start_with_dependencies_and_identity(config, sampler, sink, delivery, barrier, None)
    }

    #[cfg(test)]
    fn start_with_dependencies_and_identity(
        config: RouterConfig,
        sampler: Arc<dyn Sampler>,
        sink: Arc<dyn TrajectorySink>,
        delivery: Arc<dyn TrajectoryDelivery>,
        barrier: Arc<dyn SubscriberBarrier>,
        injected_identity: Option<TrajectoryIdentity>,
    ) -> Result<Arc<Self>, String> {
        Self::start_with_dependencies_and_activation(
            config,
            sampler,
            Some((sink, delivery)),
            barrier,
            injected_identity,
            None,
        )
    }

    fn start_with_dependencies_and_activation(
        config: RouterConfig,
        sampler: Arc<dyn Sampler>,
        injected_storage: Option<(Arc<dyn TrajectorySink>, Arc<dyn TrajectoryDelivery>)>,
        barrier: Arc<dyn SubscriberBarrier>,
        injected_identity: Option<TrajectoryIdentity>,
        activated_ledger: Option<ActivatedLedger>,
    ) -> Result<Arc<Self>, String> {
        let activated_ledger = activated_ledger.map(|activated| {
            let ActivatedLedger {
                repository,
                identity,
                cohort_assignment,
                registry,
                schema_report,
                control_snapshot,
                control_saturated,
            } = activated;
            (
                ActivatedRepositoryGuard::new(repository),
                identity,
                cohort_assignment,
                registry,
                schema_report,
                control_snapshot,
                control_saturated,
            )
        });
        if injected_identity.is_some() && activated_ledger.is_some() {
            return Err("Router received conflicting runtime identities".into());
        }
        if injected_storage.is_some() == activated_ledger.is_some() {
            return Err("Router requires exactly one runtime storage authority".into());
        }
        let matcher = PoolMatcher::compile(&config).map_err(|diagnostics| {
            diagnostics
                .into_iter()
                .map(|diagnostic| diagnostic.code)
                .collect::<Vec<_>>()
                .join(", ")
        })?;
        let expected_config_generation_id = config.generation_id()?;
        let (
            config_generation_id,
            identity,
            ledger_identity,
            cohort_assignment,
            repository,
            registry,
            schema_report,
            initial_control_snapshot,
            initial_control_saturated,
        ) = if let Some((
            repository,
            ledger_identity,
            cohort_assignment,
            registry,
            schema_report,
            control_snapshot,
            control_saturated,
        )) = activated_ledger
        {
            let expected_pool_ids = config
                .pools
                .iter()
                .map(|pool| pool.id.as_str())
                .collect::<BTreeSet<_>>();
            let ledger_pool_ids = ledger_identity
                .pools
                .keys()
                .map(String::as_str)
                .collect::<BTreeSet<_>>();
            if ledger_identity.config_generation_id != expected_config_generation_id
                || ledger_pool_ids != expected_pool_ids
            {
                return Err("Router ledger runtime identity does not match configuration".into());
            }
            let trajectory_identity = trajectory_identity_from_ledger(&ledger_identity);
            (
                ledger_identity.config_generation_id.clone(),
                trajectory_identity,
                Some(ledger_identity),
                Some(cohort_assignment),
                Some(repository),
                Some(registry),
                schema_report,
                control_snapshot,
                control_saturated,
            )
        } else {
            let identity = injected_identity.unwrap_or_else(|| {
                TrajectoryIdentity::for_pools(
                    config.project_id.as_deref(),
                    config.pools.iter().map(|pool| pool.id.as_str()),
                )
            });
            (
                expected_config_generation_id,
                identity,
                None,
                None,
                None,
                None,
                SchemaVerificationReport::default(),
                None,
                false,
            )
        };
        let vector_registry = registry.map(Arc::new);
        let embedder_clients = vector_registry
            .as_deref()
            .map(|registry| build_frozen_embedder_clients(&config, registry))
            .transpose()
            .map_err(|failure| {
                format!(
                    "Router embedder initialization failed: {}",
                    failure.stable_class()
                )
            })?
            .map(Arc::new);
        let pool_limits = config
            .pools
            .iter()
            .map(|pool| (pool.id.clone(), pool.concurrency.max_pending))
            .collect::<BTreeMap<_, _>>();
        let pool_resources = config
            .pools
            .iter()
            .map(|pool| (pool.id.clone(), Arc::new(PoolRuntimeResources::new(pool))))
            .collect();
        let max_event_projection_bytes = config
            .pools
            .iter()
            .map(|pool| pool.lookahead.max_bytes_per_window)
            .max()
            .unwrap_or(1);
        let total_max_pending = pool_limits
            .values()
            .copied()
            .fold(0usize, usize::saturating_add);
        let command_capacity =
            CoordinatorLimits::from_total_max_pending(total_max_pending).command_capacity;
        let coordinator = if config.mode == RouterMode::Active {
            Coordinator::new_with_active_outcomes(pool_limits)
        } else {
            Coordinator::new(pool_limits)
        };
        let (coordinator_tx, coordinator_rx) = mpsc::channel(command_capacity);
        let observes_events = matches!(config.mode, RouterMode::Shadow | RouterMode::Active);
        let runtime_handle = tokio::runtime::Handle::try_current()
            .map_err(|error| format!("Router requires an active Tokio runtime: {error}"))?;
        let mut scheduler_actor = None;
        let (writer, writer_client, read_pool, sink, delivery, scheduler_controls) =
            if let Some(mut repository) = repository {
                let read_pool = LedgerReadPool::open(std::path::Path::new(&config.database_path))
                    .map_err(|error| error.code().to_string())?;
                let writer_capacity = config.writer_command_capacity()?;
                let active_replay_capacity = config.active_replay_capacity()?;
                let sink_plan = SqliteTrajectorySink::prepare(&config).map_err(|error| {
                    format!("Router SQLite sink initialization failed: {error:?}")
                })?;
                let scheduler_plan = ShadowScheduler::prepare(&config).map_err(|error| {
                    format!("Router scheduler initialization failed: {error:?}")
                })?;
                let active_replays = ActiveReplayRegistry::new(active_replay_capacity)
                    .map_err(|_| "Router active replay capacity is invalid".to_string())?;
                let start_gate = BackgroundStartGate::new();
                let cancellation = EvaluatorCancellation::default();
                let deadlines = SchedulerDeadlineAuthority::new();
                let (owner, client) = LedgerWriterOwner::start(repository.take(), writer_capacity)
                    .map_err(|error| error.to_string())?;
                let (sqlite_sink, scheduler_rx) = sink_plan.start(client.clone());
                let admissions = sqlite_sink.admission_pools();
                let scheduler = scheduler_plan.start(
                    client.clone(),
                    scheduler_rx,
                    start_gate.clone(),
                    active_replays.clone(),
                    cancellation.clone(),
                    deadlines.clone(),
                );
                scheduler_actor = Some(scheduler);
                let sink: Arc<dyn TrajectorySink> = Arc::new(sqlite_sink.clone());
                let delivery: Arc<dyn TrajectoryDelivery> = Arc::new(sqlite_sink);
                (
                    Some(owner),
                    Some(client),
                    Some(read_pool),
                    sink,
                    delivery,
                    Some(SchedulerRuntimeControls {
                        start_gate,
                        active_replays,
                        cancellation,
                        deadlines,
                        admissions,
                    }),
                )
            } else {
                let (sink, delivery) = injected_storage
                    .ok_or_else(|| "Router injected storage is missing".to_string())?;
                (None, None, None, sink, delivery, None)
            };
        let (retention_pressure, retention_pressure_rx) = if writer_client.is_some() {
            let (sender, receiver) = watch::channel(None);
            (Some(sender), Some(receiver))
        } else {
            (None, None)
        };
        let provider_admission = ProviderAdmissionGate::new_pending();
        let cohort_assignment = cohort_assignment.map(Arc::new);
        let control_authority = match (
            writer_client.clone(),
            read_pool.clone(),
            initial_control_snapshot,
        ) {
            (Some(writer), Some(read_pool), Some(snapshot)) => Some(
                ControlRuntimeAuthority::new(
                    writer,
                    read_pool,
                    identity.project_uuid,
                    config_generation_id.clone(),
                    config.pools.iter().map(|pool| pool.id.clone()).collect(),
                    provider_admission.clone(),
                    snapshot,
                    cohort_assignment
                        .clone()
                        .ok_or("Router cohort assignment authority is missing")?,
                    initial_control_saturated,
                )
                .map_err(str::to_string)?,
            ),
            (None, None, None) => None,
            _ => return Err("Router control resource authority is incomplete".into()),
        };
        let live_embedding = match (
            writer_client.clone(),
            read_pool.clone(),
            vector_registry.clone(),
            embedder_clients.clone(),
        ) {
            (
                Some(writer_client),
                Some(read_pool),
                Some(vector_registry),
                Some(embedder_clients),
            ) => {
                match LiveEmbeddingService::new(
                    &config,
                    vector_registry,
                    writer_client,
                    read_pool.clone(),
                    embedder_clients,
                    provider_admission.clone(),
                ) {
                    Ok(service) => Some(service),
                    Err(error) => {
                        if let Some(owner) = writer.as_ref() {
                            owner.request_process_stop_on_abort();
                            owner.abort();
                        }
                        read_pool.abort();
                        return Err(format!(
                            "Router live embedding initialization failed: {error}"
                        ));
                    }
                }
            }
            (None, None, None, None) => None,
            _ => return Err("Router live embedding resource authority is incomplete".into()),
        };
        let (recommendation_delivery, recommendation_vector_store) = if config.mode
            == RouterMode::Recommend
        {
            let writer = writer_client.clone().ok_or_else(|| {
                "Router Recommend mode requires durable writer authority".to_string()
            })?;
            let readers = read_pool.clone().ok_or_else(|| {
                "Router Recommend mode requires durable reader authority".to_string()
            })?;
            let delivery =
                RecommendationDeliveryServiceV1::new(writer.clone(), config.max_evidence_records)
                    .map_err(|_| "Router recommendation delivery configuration is invalid")?;
            (Some(delivery), Some(SqliteVecStore::new(writer, readers)))
        } else if config.mode == RouterMode::Active {
            let writer = writer_client.clone().ok_or_else(|| {
                "Router Active mode requires durable writer authority".to_string()
            })?;
            let readers = read_pool.clone().ok_or_else(|| {
                "Router Active mode requires durable reader authority".to_string()
            })?;
            (None, Some(SqliteVecStore::new(writer, readers)))
        } else {
            (None, None)
        };
        let health = Arc::new(RouterHealth::default());
        let (active_outcome, active_outcome_task) = if config.mode == RouterMode::Active {
            let writer = writer_client.clone().ok_or_else(|| {
                "Router Active mode requires durable writer authority".to_string()
            })?;
            let (client, task) =
                start_active_outcome_runtime_v2(writer, command_capacity, health.clone())
                    .map_err(|_| "Router Active outcome runtime configuration is invalid")?;
            (Some(client), Some(task))
        } else {
            (None, None)
        };
        let recommendation_retention_pressure = recommendation_delivery
            .as_ref()
            .map(RecommendationDeliveryServiceV1::subscribe_retention_pressure);
        let state = Arc::new(RuntimeState {
            config: Arc::new(config),
            matcher,
            adapter: FamilyAdapter,
            sampler,
            pool_resources,
            identity,
            ledger_identity,
            config_generation_id,
            max_event_projection_bytes,
            accepting: AtomicBool::new(true),
            event_ingestion: AtomicBool::new(observes_events),
            sampling_enabled: AtomicBool::new(observes_events),
            permanent_runtime_fault: AtomicBool::new(false),
            admission_epoch: AtomicU64::new(0),
            admission: Mutex::new(()),
            loss: Arc::new(LossState::default()),
            coordinator_tx,
            health,
            writer_client,
            retention_pressure,
            read_pool,
            scheduler_controls,
            provider_admission,
            foreground_admission: Arc::new(Semaphore::new(4)),
            subscriber_barrier: barrier.clone(),
            active_outcome,
            control_authority,
            live_embedding,
            recommendation_delivery,
            recommendation_vector_store,
            vector_registry,
            embedder_clients,
            initial_vector_schema_report: schema_report,
            finished: AtomicBool::new(false),
            finished_notify: Notify::new(),
            #[cfg(test)]
            ownership_probe: Arc::new(()),
            #[cfg(test)]
            owned_task_count: AtomicUsize::new(0),
            #[cfg(test)]
            peak_owned_task_count: AtomicUsize::new(0),
            #[cfg(test)]
            heartbeat_ticks: AtomicUsize::new(0),
            #[cfg(test)]
            heartbeat_worker_ready: AtomicBool::new(false),
            #[cfg(test)]
            retention_ticks: AtomicUsize::new(0),
            #[cfg(test)]
            retention_already_applied_ticks: AtomicUsize::new(0),
            #[cfg(test)]
            retention_worker_ready: AtomicBool::new(false),
            #[cfg(test)]
            retention_timer_fires: AtomicUsize::new(0),
            #[cfg(test)]
            retention_recoveries: Mutex::new(Vec::new()),
        });
        let background = match (
            state.writer_client.clone(),
            state.read_pool.clone(),
            state.vector_registry.clone(),
            state.embedder_clients.clone(),
        ) {
            (Some(writer), Some(read_pool), Some(vector_registry), Some(embedder_clients)) => {
                let resources = BackgroundResources::new(
                    writer,
                    read_pool,
                    state.identity.process_instance_id,
                    vector_registry,
                    embedder_clients,
                    state.initial_vector_schema_report.clone(),
                );
                Some(BackgroundRuntime::start(
                    &runtime_handle,
                    Arc::new(RouterBackgroundDriver::new(resources)),
                    state.provider_admission.clone(),
                    background_failure_notifier(state.clone()),
                    BACKGROUND_TASK_LIMIT,
                ))
            }
            (None, None, None, None) => None,
            _ => {
                return Err("Router background resource authority is incomplete".into());
            }
        };
        if let (Some(scheduler), Some(controls)) =
            (scheduler_actor.as_mut(), state.scheduler_controls.clone())
        {
            let fault_state = state.clone();
            scheduler.set_failure_notifier(Arc::new(move |_| {
                controls.start_gate.close();
                controls.cancellation.cancel();
                controls.active_replays.close_and_cancel_all();
                controls.admissions.close();
                record_permanent_runtime_fault(&fault_state, "router.scheduler.runtime_failure");
            }));
        }
        let scheduler = scheduler_actor.map(|scheduler| {
            let fault_state = state.clone();
            runtime_handle.spawn(async move {
                let exit = scheduler.run().await;
                if exit.summary().first_failure.is_some() || exit.retained_batch_count() != 0 {
                    record_permanent_runtime_fault(
                        &fault_state,
                        "router.scheduler.runtime_failure",
                    );
                }
                exit
            })
        });
        let retention =
            state
                .writer_client
                .clone()
                .zip(retention_pressure_rx)
                .map(|(writer, pressure_rx)| {
                    let (stop, stop_rx) = watch::channel(false);
                    let (manual, manual_rx) = mpsc::channel(1);
                    let task = runtime_handle.spawn(run_retention(
                        state.clone(),
                        writer,
                        stop_rx,
                        pressure_rx,
                        recommendation_retention_pressure,
                        manual_rx,
                        RETENTION_INTERVAL,
                    ));
                    RetentionRuntime {
                        stop,
                        pressure: state
                            .retention_pressure
                            .as_ref()
                            .expect("retention pressure sender must exist with writer")
                            .clone(),
                        manual,
                        task,
                    }
                });
        let heartbeat = state.writer_client.clone().map(|writer| {
            let (stop, stop_rx) = watch::channel(false);
            let task = runtime_handle.spawn(run_heartbeat(
                state.clone(),
                writer,
                stop_rx,
                HEARTBEAT_INTERVAL,
            ));
            HeartbeatRuntime { stop, task }
        });
        let coordinator = runtime_handle.spawn(run_coordinator(
            state.clone(),
            coordinator_rx,
            coordinator,
            sink,
            delivery,
            barrier,
        ));
        let abort_handles = RuntimeAbortHandles {
            coordinator: coordinator.abort_handle(),
            active_outcome: active_outcome_task.as_ref().map(JoinHandle::abort_handle),
            scheduler: scheduler.as_ref().map(JoinHandle::abort_handle),
            background: background.as_ref().map(BackgroundRuntime::abort_handle),
            retention: retention
                .as_ref()
                .map(|retention| retention.task.abort_handle()),
            heartbeat: heartbeat
                .as_ref()
                .map(|heartbeat| heartbeat.task.abort_handle()),
        };
        Ok(Arc::new(Self {
            state,
            lifecycle_gate: Mutex::new(()),
            lifecycle: AtomicU8::new(LIFECYCLE_RUNNING),
            process_stop: Mutex::new(None),
            abort_handles,
            coordinator: Mutex::new(Some(coordinator)),
            active_outcome: Mutex::new(active_outcome_task),
            scheduler: Mutex::new(scheduler),
            background: Mutex::new(background),
            retention: Mutex::new(retention),
            heartbeat: Mutex::new(heartbeat),
            control_monitor: Mutex::new(None),
            writer: Mutex::new(writer),
            #[cfg(test)]
            drain_retained_owners_pause: Mutex::new(None),
        }))
    }

    pub(crate) fn execution_callback(self: &Arc<Self>) -> LlmExecutionV2Fn {
        let runtime = self.clone();
        Arc::new(move |_, context, request, replay, next| {
            let runtime = runtime.clone();
            Box::pin(async move { runtime.execute(context, request, replay, next).await })
        })
    }

    #[cfg(test)]
    pub(crate) async fn execute_for_test(
        &self,
        context: Arc<LlmExecutionContextSnapshot>,
        request: LlmRequest,
        replay: Option<Arc<dyn LlmReplayTransport>>,
        next: LlmExecutionNextFn,
    ) -> FlowResult<Json> {
        self.execute(context, request, replay, next).await
    }

    #[cfg(test)]
    pub(crate) async fn drain_for_test(&self, deadline: Instant) -> PluginResult<()> {
        self.drain(deadline).await
    }

    #[cfg(test)]
    pub(crate) fn test_accepting(&self) -> bool {
        self.state.accepting.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(crate) fn test_state_probe(&self) -> std::sync::Weak<()> {
        Arc::downgrade(&self.state.ownership_probe)
    }

    #[cfg(test)]
    pub(crate) fn test_provider_admission(&self) -> ProviderAdmissionGate {
        self.state.provider_admission.clone()
    }

    #[cfg(test)]
    pub(crate) async fn test_saturate_foreground(&self) -> tokio::sync::OwnedSemaphorePermit {
        self.state
            .foreground_admission
            .clone()
            .acquire_many_owned(4)
            .await
            .expect("foreground admission remains open while the runtime is live")
    }

    #[allow(dead_code)] // Consumed by the later density/confidence routing stage.
    pub(crate) fn live_embedding_service(&self) -> Option<LiveEmbeddingService> {
        self.state.live_embedding.clone()
    }

    #[cfg(test)]
    pub(crate) fn test_cancel_retention_worker(&self) {
        if let Some(abort) = self.abort_handles.retention.as_ref() {
            abort.abort();
        }
    }

    #[cfg(test)]
    pub(crate) fn test_retention_worker_finished(&self) -> bool {
        self.abort_handles
            .retention
            .as_ref()
            .is_none_or(AbortHandle::is_finished)
    }

    #[cfg(test)]
    async fn test_run_heartbeat_once(&self, observed_at_unix_ms: i64) -> Result<(), &'static str> {
        let writer = self
            .state
            .writer_client
            .as_ref()
            .ok_or(HEALTH_HEARTBEAT_FAILURE)?;
        run_heartbeat_tick(&self.state, writer, observed_at_unix_ms).await
    }

    #[cfg(test)]
    async fn test_run_retention_once(
        &self,
        observed_at_unix_ms: i64,
    ) -> Result<RetentionAck, &'static str> {
        let manual = self
            .retention
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .map(|retention| retention.manual.clone())
            .ok_or(HEALTH_RETENTION_FAILURE)?;
        let (reply, receiver) = oneshot::channel();
        manual
            .send(RetentionManualRequest {
                observed_at_unix_ms,
                reply,
            })
            .await
            .map_err(|_| HEALTH_RETENTION_FAILURE)?;
        receiver.await.map_err(|_| HEALTH_RETENTION_FAILURE)?
    }

    pub(crate) fn subscriber_callback(self: &Arc<Self>) -> EventSubscriberFn {
        let runtime = self.clone();
        Arc::new(move |event| runtime.observe_event_fail_open(event))
    }

    pub(crate) fn lifecycle_registration(
        self: &Arc<Self>,
        name: String,
    ) -> PluginResult<PluginRegistration> {
        let activation_token =
            self.state
                .provider_admission
                .activation_token()
                .map_err(|error| {
                    PluginError::Internal(format!(
                        "Router provider activation token could not be issued: {error:?}"
                    ))
                })?;
        if let Some(authority) = self.state.control_authority.as_ref() {
            if let Err(error) = authority.stage_publication() {
                activation_token.rollback();
                return Err(PluginError::Internal(error.to_string()));
            }
            let (stop, stop_rx) = watch::channel(false);
            let monitor_authority = authority.clone();
            let runtime_handle = match tokio::runtime::Handle::try_current() {
                Ok(runtime_handle) => runtime_handle,
                Err(error) => {
                    authority.unpublish();
                    activation_token.rollback();
                    return Err(PluginError::Internal(format!(
                        "Router control monitor requires a Tokio runtime: {error}"
                    )));
                }
            };
            let task =
                runtime_handle.spawn(async move { monitor_authority.run_monitor(stop_rx).await });
            let mut monitor = self
                .control_monitor
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if monitor.is_some() {
                task.abort();
                authority.unpublish();
                activation_token.rollback();
                return Err(PluginError::Internal(
                    "Router control monitor was already started".into(),
                ));
            }
            *monitor = Some(ControlMonitorRuntime { stop, task });
        }
        let deregister_runtime = self.clone();
        let stop_runtime = self.clone();
        let drain_runtime = self.clone();
        let abort_runtime = self.clone();
        let rollback_runtime = self.clone();
        let mut activation_token = Some(activation_token);
        Ok(PluginRegistration::with_shutdown(
            "router",
            name,
            Box::new(move || {
                deregister_runtime.abort();
                Ok(())
            }),
            Box::new(move || {
                stop_runtime.stop_intake();
                Ok(())
            }),
            Box::new(move |deadline| {
                let runtime = drain_runtime.clone();
                Box::pin(async move { runtime.drain(deadline).await })
            }),
            Box::new(move || {
                abort_runtime.abort();
                Ok(())
            }),
        )
        .with_activation_rollback(Box::new(move || {
            if let Some(token) = activation_token.take() {
                token.rollback();
            }
            rollback_runtime.request_activation_rollback();
            Ok(())
        })))
    }

    fn request_activation_rollback(&self) {
        if let Some(writer) = self
            .writer
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
        {
            writer.request_process_stop_on_abort();
        }
    }

    async fn execute(
        &self,
        context: impl Into<Arc<LlmExecutionContextSnapshot>>,
        request: LlmRequest,
        replay: Option<Arc<dyn LlmReplayTransport>>,
        next: LlmExecutionNextFn,
    ) -> FlowResult<Json> {
        let context = context.into();
        if context.call_role != LlmCallRole::Primary {
            drop(context);
            drop(replay);
            let result = next(request).await;
            drop(next);
            return result;
        }
        if self.state.config.mode == RouterMode::Off {
            drop(context);
            drop(replay);
            let result = next(request).await;
            drop(next);
            return result;
        }
        if self.state.config.mode == RouterMode::Recommend {
            return self.execute_recommend(context, request, replay, next).await;
        }
        if self.state.config.mode == RouterMode::Active {
            return self.execute_active(context, request, replay, next).await;
        }

        let intent =
            self.prepare_sample_intent_fail_open(context.as_ref(), &request, replay.as_ref());
        drop(context);
        drop(replay);
        let result = next(request).await;
        drop(next);
        let response = match result {
            Ok(response) => response,
            Err(error) => return Err(error),
        };
        if let Some(intent) = intent {
            self.propose_anchor_fail_open(intent, &response);
        }
        Ok(response)
    }

    async fn execute_recommend(
        &self,
        context: Arc<LlmExecutionContextSnapshot>,
        request: LlmRequest,
        replay: Option<Arc<dyn LlmReplayTransport>>,
        next: LlmExecutionNextFn,
    ) -> FlowResult<Json> {
        let mut replay = replay.map(PanicSafeArcOwner::new);
        let next = PanicSafeArcOwner::new(next);
        let preprocessing = panic_safe_recommendation_future(self.preprocess_recommendation(
            context.as_ref(),
            &request,
            replay.as_mut().map(PanicSafeArcOwner::take),
        ))
        .await;
        match preprocessing {
            Ok(Ok(())) => {}
            Ok(Err(reason)) => self.record_health(reason),
            Err(()) => self.record_health(HEALTH_RECOMMENDATION_COMPUTE_FAILURE),
        }
        drop(context);
        drop(replay);
        let result = next.get()(request).await;
        drop(next);
        result
    }

    async fn execute_active(
        &self,
        context: Arc<LlmExecutionContextSnapshot>,
        request: LlmRequest,
        replay: Option<Arc<dyn LlmReplayTransport>>,
        next: LlmExecutionNextFn,
    ) -> FlowResult<Json> {
        let foreground = self
            .state
            .provider_admission
            .start_owned(|| self.state.foreground_admission.clone().try_acquire_owned())
            .ok()
            .and_then(Result::ok);
        let Some(_foreground) = foreground else {
            drop(context);
            drop(replay);
            let result = next(request).await;
            drop(next);
            return result;
        };

        let mut replay = replay.map(PanicSafeArcOwner::new);
        let next = PanicSafeArcOwner::new(next);
        let registration_epoch = self.register_primary_call(context.as_ref());
        let shadow_intent = self.prepare_sample_intent_fail_open(
            context.as_ref(),
            &request,
            replay.as_ref().map(PanicSafeArcOwner::get),
        );
        let Some(registration_epoch) = registration_epoch else {
            drop(context);
            drop(replay);
            let result = next.get()(request).await;
            drop(next);
            return result;
        };
        let prepared = panic_safe_recommendation_future(self.preprocess_active(
            context.as_ref(),
            &request,
            replay.as_mut().map(PanicSafeArcOwner::take),
            registration_epoch,
        ))
        .await;
        drop(replay);
        let route = match prepared {
            Ok(Ok(route)) => route,
            Ok(Err(reason)) => {
                self.record_health(reason);
                let result = next.get()(request).await;
                drop(next);
                return self.finish_fallback_anchor(shadow_intent, result);
            }
            Err(()) => {
                self.record_health(HEALTH_ACTIVE_COMPUTE_FAILURE);
                let result = next.get()(request).await;
                drop(next);
                return self.finish_fallback_anchor(shadow_intent, result);
            }
        };

        match route {
            PreparedActiveRouteV2::Candidate {
                request,
                admitted,
                mut dispatch_guard,
            } => {
                drop(shadow_intent);
                let handed_off_at_unix_ms = now_unix_ms_u64();
                dispatch_guard.mark_handed_off(handed_off_at_unix_ms);
                let result = panic_safe_recommendation_future(next.get()(request)).await;
                drop(next);
                match result {
                    Ok(Ok(response)) => {
                        let codec_policy_failure = project_anchor_response(
                            admitted.api_family,
                            &response,
                            self.state.max_event_projection_bytes,
                        )
                        .is_err();
                        let receipt_hash = canonical_sha256(&response).ok();
                        self.finish_active_dispatch(
                            dispatch_guard,
                            &admitted,
                            ActiveDispatchTerminalState::Completed,
                            None,
                            receipt_hash,
                            RepresentativeResultV1 {
                                terminal: RepresentativeTerminalV1::Completed,
                                codec_policy_failure,
                            },
                        )
                        .await;
                        if codec_policy_failure {
                            self.invalidate_active_winner(&admitted, "response_codec_failure")
                                .await;
                        }
                        Ok(response)
                    }
                    Ok(Err(error)) => {
                        self.finish_active_dispatch(
                            dispatch_guard,
                            &admitted,
                            ActiveDispatchTerminalState::ProviderError,
                            Some("provider_error".to_string()),
                            None,
                            RepresentativeResultV1 {
                                terminal: RepresentativeTerminalV1::Error,
                                codec_policy_failure: false,
                            },
                        )
                        .await;
                        self.invalidate_active_winner(&admitted, "provider_error")
                            .await;
                        Err(error)
                    }
                    Err(()) => {
                        self.finish_active_dispatch(
                            dispatch_guard,
                            &admitted,
                            ActiveDispatchTerminalState::PanickedAfterHandoff,
                            None,
                            None,
                            RepresentativeResultV1 {
                                terminal: RepresentativeTerminalV1::Panicked,
                                codec_policy_failure: false,
                            },
                        )
                        .await;
                        self.invalidate_active_winner(&admitted, "provider_panic")
                            .await;
                        Err(nemo_relay::error::FlowError::Internal(
                            "Router Active continuation panicked".to_string(),
                        ))
                    }
                }
            }
            PreparedActiveRouteV2::Anchor { admitted } => {
                let result = panic_safe_recommendation_future(next.get()(request)).await;
                drop(next);
                match result {
                    Ok(Ok(response)) => {
                        self.record_active_representative(
                            &admitted,
                            RepresentativeResultV1 {
                                terminal: RepresentativeTerminalV1::Completed,
                                codec_policy_failure: false,
                            },
                            None,
                        )
                        .await;
                        if let Some(intent) = shadow_intent {
                            self.propose_anchor_fail_open(intent, &response);
                        }
                        Ok(response)
                    }
                    Ok(Err(error)) => {
                        self.record_active_representative(
                            &admitted,
                            RepresentativeResultV1 {
                                terminal: RepresentativeTerminalV1::Error,
                                codec_policy_failure: false,
                            },
                            Some("provider_error".to_string()),
                        )
                        .await;
                        Err(error)
                    }
                    Err(()) => Err(nemo_relay::error::FlowError::Internal({
                        self.record_active_representative(
                            &admitted,
                            RepresentativeResultV1 {
                                terminal: RepresentativeTerminalV1::Panicked,
                                codec_policy_failure: false,
                            },
                            None,
                        )
                        .await;
                        "Router Active anchor continuation panicked".to_string()
                    })),
                }
            }
            PreparedActiveRouteV2::PostAdmissionFailure { dispatch_guard } => {
                drop(next);
                drop(shadow_intent);
                drop(dispatch_guard);
                self.record_health(HEALTH_ACTIVE_OUTCOME_FAILURE);
                Err(nemo_relay::error::FlowError::Internal(
                    "Router Active outcome ownership failed after admission".to_string(),
                ))
            }
        }
    }

    fn finish_fallback_anchor(
        &self,
        shadow_intent: Option<SampleIntent>,
        result: FlowResult<Json>,
    ) -> FlowResult<Json> {
        match result {
            Ok(response) => {
                if let Some(intent) = shadow_intent {
                    self.propose_anchor_fail_open(intent, &response);
                }
                Ok(response)
            }
            Err(error) => Err(error),
        }
    }

    async fn preprocess_recommendation(
        &self,
        context: &LlmExecutionContextSnapshot,
        request: &LlmRequest,
        replay: Option<Arc<dyn LlmReplayTransport>>,
    ) -> Result<(), &'static str> {
        let (pool, identity, live_embedding, vector_store, delivery_permit, admitted_at, deadline) = {
            let _admission = self
                .state
                .admission
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if !self.state.accepting.load(Ordering::Acquire)
                || self.state.permanent_runtime_fault.load(Ordering::Acquire)
            {
                return Err(HEALTH_RECOMMENDATION_ADMISSION_FAILURE);
            }
            let anchor_model = request
                .content
                .get("model")
                .and_then(Json::as_str)
                .ok_or(HEALTH_RECOMMENDATION_ADMISSION_FAILURE)?;
            let matched = self.state.matcher.match_pool(anchor_model, context);
            if matched.health_reason.is_some() {
                return Err(HEALTH_RECOMMENDATION_ADMISSION_FAILURE);
            }
            let pool_id = matched
                .matched_pool_id
                .ok_or(HEALTH_RECOMMENDATION_ADMISSION_FAILURE)?;
            let runtime_identity = match self.state.control_authority.as_ref() {
                Some(controls) => {
                    let generation = controls
                        .current_generation_snapshot()
                        .ok_or(HEALTH_RECOMMENDATION_ADMISSION_FAILURE)?;
                    let control = generation
                        .control
                        .pools
                        .get(pool_id)
                        .ok_or(HEALTH_RECOMMENDATION_ADMISSION_FAILURE)?;
                    if control.effective.paused {
                        return Ok(());
                    }
                    trajectory_identity_for_generation(&self.state.identity, &generation)
                        .ok_or(HEALTH_RECOMMENDATION_ADMISSION_FAILURE)?
                }
                None => self.state.identity.clone(),
            };
            let pool = self
                .state
                .config
                .pools
                .iter()
                .find(|pool| pool.id == pool_id)
                .cloned()
                .ok_or(HEALTH_RECOMMENDATION_ADMISSION_FAILURE)?;
            let live_embedding = self
                .state
                .live_embedding
                .clone()
                .ok_or(HEALTH_RECOMMENDATION_ADMISSION_FAILURE)?;
            let vector_store = self
                .state
                .recommendation_vector_store
                .clone()
                .ok_or(HEALTH_RECOMMENDATION_ADMISSION_FAILURE)?;
            let delivery = self
                .state
                .recommendation_delivery
                .as_ref()
                .ok_or(HEALTH_RECOMMENDATION_ADMISSION_FAILURE)?;
            let admitted_at = Instant::now();
            let admitted_at_unix_ms = Utc::now().timestamp_millis().max(0);
            let deadline = admitted_at
                .checked_add(
                    live_embedding
                        .timeout_for_pool(&pool.id)
                        .ok_or(HEALTH_RECOMMENDATION_ADMISSION_FAILURE)?,
                )
                .ok_or(HEALTH_RECOMMENDATION_ADMISSION_FAILURE)?;
            let delivery_permit = delivery
                .try_admit()
                .map_err(|_| HEALTH_RECOMMENDATION_ADMISSION_FAILURE)?;
            let identity = RecommendationRuntimeIdentityV1::from_runtime(
                &runtime_identity,
                self.state.config_generation_id.clone(),
                &pool.id,
                admitted_at_unix_ms,
            )
            .map_err(|_| HEALTH_RECOMMENDATION_ADMISSION_FAILURE)?;
            (
                pool,
                identity,
                live_embedding,
                vector_store,
                delivery_permit,
                admitted_at,
                deadline,
            )
        };

        let audit = compute_recommendation_audit_until(
            context,
            request,
            replay,
            &pool,
            &identity,
            &live_embedding,
            &vector_store,
            admitted_at,
            deadline,
        )
        .await
        .map_err(recommendation_compute_health_code)?;
        match delivery_permit
            .submit_until(Arc::new(audit), deadline)
            .await
        {
            Ok(
                RecommendationFirstAckV1::Applied
                | RecommendationFirstAckV1::AlreadyApplied
                | RecommendationFirstAckV1::DroppedStale(RecommendationStaleReasonV1::SourceRetiring),
            ) => Ok(()),
            Ok(RecommendationFirstAckV1::DroppedStale(
                RecommendationStaleReasonV1::AuthorityChanged
                | RecommendationStaleReasonV1::EvidenceChanged
                | RecommendationStaleReasonV1::OriginatingProcessNotLive,
            )) => Err(HEALTH_RECOMMENDATION_DELIVERY_FAILURE),
            Ok(RecommendationFirstAckV1::PendingRetry) => {
                Err(HEALTH_RECOMMENDATION_DELIVERY_PENDING)
            }
            Err(_) => Err(HEALTH_RECOMMENDATION_DELIVERY_FAILURE),
        }
    }

    async fn preprocess_active(
        &self,
        context: &LlmExecutionContextSnapshot,
        request: &LlmRequest,
        replay: Option<Arc<dyn LlmReplayTransport>>,
        registration_epoch: u64,
    ) -> Result<PreparedActiveRouteV2, &'static str> {
        let (
            pool,
            identity,
            ledger_identity,
            live_embedding,
            vector_store,
            writer,
            cohort_assignment,
            control_generation,
            admitted_at,
            deadline,
        ) = {
            let _admission = self
                .state
                .admission
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if !self.state.accepting.load(Ordering::Acquire)
                || self.state.permanent_runtime_fault.load(Ordering::Acquire)
                || self.state.admission_epoch.load(Ordering::Acquire) != registration_epoch
            {
                return Err(HEALTH_ACTIVE_ADMISSION_FAILURE);
            }
            let anchor_model = request
                .content
                .get("model")
                .and_then(Json::as_str)
                .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?;
            let matched = self.state.matcher.match_pool(anchor_model, context);
            if matched.health_reason.is_some() {
                return Err(HEALTH_ACTIVE_ADMISSION_FAILURE);
            }
            let pool_id = matched
                .matched_pool_id
                .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?;
            let controls = self
                .state
                .control_authority
                .as_ref()
                .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?;
            let generation = controls
                .current_generation_snapshot()
                .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?;
            let control_state = generation
                .control
                .pools
                .get(pool_id)
                .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?
                .effective;
            if control_state.paused || control_state.force_anchor {
                return Err(HEALTH_ACTIVE_ADMISSION_FAILURE);
            }
            let control_generation = generation.control.control_generation;
            let pool = self
                .state
                .config
                .pools
                .iter()
                .find(|pool| pool.id == pool_id)
                .cloned()
                .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?;
            let live_embedding = self
                .state
                .live_embedding
                .clone()
                .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?;
            let vector_store = self
                .state
                .recommendation_vector_store
                .clone()
                .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?;
            let writer = self
                .state
                .writer_client
                .clone()
                .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?;
            let base_ledger_identity = self
                .state
                .ledger_identity
                .as_ref()
                .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?;
            let ledger_identity = ledger_identity_for_generation(base_ledger_identity, &generation)
                .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?;
            let runtime_identity =
                trajectory_identity_for_generation(&self.state.identity, &generation)
                    .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?;
            let cohort_assignment = generation.cohort_assignment.clone();
            let admitted_at = Instant::now();
            let admitted_at_unix_ms = Utc::now().timestamp_millis().max(0);
            let deadline = admitted_at
                .checked_add(
                    live_embedding
                        .timeout_for_pool(pool_id)
                        .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?,
                )
                .ok_or(HEALTH_ACTIVE_ADMISSION_FAILURE)?;
            let identity = RecommendationRuntimeIdentityV1::from_runtime(
                &runtime_identity,
                self.state.config_generation_id.clone(),
                pool_id,
                admitted_at_unix_ms,
            )
            .map_err(|_| HEALTH_ACTIVE_ADMISSION_FAILURE)?;
            (
                pool,
                identity,
                ledger_identity,
                live_embedding,
                vector_store,
                writer,
                cohort_assignment,
                control_generation,
                admitted_at,
                deadline,
            )
        };

        let active_policy = pool
            .learning
            .as_ref()
            .and_then(|learning| learning.complete_active_policy())
            .ok_or(HEALTH_ACTIVE_COMPUTE_FAILURE)?;
        let outcome = serde_json::from_value::<OutcomeConfig>(Json::Object(
            pool.outcome.clone().into_iter().collect(),
        ))
        .map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)?;
        let protected_outcome = protected_outcome_policy_version_v1(&pool.outcome)
            .map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)?
            .ok_or(HEALTH_ACTIVE_COMPUTE_FAILURE)?;
        let compiled_outcome = Arc::new(
            CompiledOutcomePolicyV1::compile(&outcome)
                .map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)?,
        );
        let prepared = prepare_active_runtime_query_v2(
            context,
            request,
            replay,
            &pool,
            &identity,
            &live_embedding,
            admitted_at,
            deadline,
        )
        .map_err(recommendation_compute_health_code)?;
        let (prepared, envelope) = prepared.into_parts();
        let canonical_query_hash = prepared.canonical_query_hash().to_string();
        let mut experiments = Vec::new();
        for input in prepared.active_experiment_inputs() {
            let create = ActiveExperimentCreate::from_config(
                self.state.config.as_ref(),
                &ledger_identity,
                &pool.id,
                input.candidate_id.clone(),
                input.partition_hash,
                protected_outcome.outcome_policy_hash.clone(),
                control_generation,
            )
            .map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)?;
            let receipt = match writer
                .create_active_experiment_until(create, deadline)
                .await
                .map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)?
            {
                ActiveExperimentCreateAck::Applied(receipt)
                | ActiveExperimentCreateAck::AlreadyApplied(receipt) => receipt,
                ActiveExperimentCreateAck::Conflict
                | ActiveExperimentCreateAck::AuthorityChanged
                | ActiveExperimentCreateAck::TransactionNotStarted => {
                    return Err(HEALTH_ACTIVE_COMPUTE_FAILURE);
                }
            };
            experiments.push(ActiveCandidateExperimentAuthorityV2 {
                candidate_id: input.candidate_id,
                active_experiment_id: receipt.active_experiment_id,
            });
        }
        let embedding_query = prepared.query_for_embedding();
        let embedding = live_embedding
            .embed_prepared_until(embedding_query, deadline)
            .await;
        let resolver = ActiveGateResolverV2::new(
            writer.clone(),
            experiments,
            ledger_identity.config_generation_id.clone(),
            ledger_identity
                .pools
                .get(&pool.id)
                .ok_or(HEALTH_ACTIVE_COMPUTE_FAILURE)?
                .learning_generation_id,
            ledger_identity.cohort_generation_id,
            canonical_query_hash,
            active_policy.recommend.promotion_lower_bound,
            active_policy.retention_lower_bound,
            Utc::now().timestamp_millis().max(0),
            deadline,
        )
        .map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)?;
        let mut evaluated =
            evaluate_active_query_until(&vector_store, prepared, embedding, resolver)
                .await
                .map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)?;
        let winner_candidate_id = evaluated
            .winner_candidate_id()
            .map(str::to_string)
            .ok_or(HEALTH_ACTIVE_COMPUTE_FAILURE)?;
        let winner_authority = evaluated
            .authorize_winner_neighborhood_until()
            .await
            .map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)?;

        let assignment = cohort_assignment
            .assign(CohortAssignmentRequest::new(
                context.root_uuid,
                &ledger_identity.config_generation_id,
                &pool.id,
                &winner_candidate_id,
                active_policy.holdout_probability,
                active_policy.active_canary_fraction,
            ))
            .map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)?;
        let (final_reason, planned_route, arm) = active_route_facts(assignment);
        let audited = evaluated
            .into_audited_query(
                ActiveDecisionParentBindingV2 {
                    cohort_generation_id: ledger_identity.cohort_generation_id,
                    active_experiment_id: Some(winner_authority.active_experiment_id),
                    active_authorization_state_event_id: Some(
                        winner_authority.active_authorization_state_event_id,
                    ),
                    root_key: assignment.root_key().to_hex(),
                },
                final_reason,
            )
            .map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)?;
        let learning_generation_id = ledger_identity
            .pools
            .get(&pool.id)
            .ok_or(HEALTH_ACTIVE_COMPUTE_FAILURE)?
            .learning_generation_id;

        self.finish_active_preparation(
            context,
            pool,
            outcome,
            compiled_outcome,
            protected_outcome.outcome_policy_hash,
            envelope,
            audited,
            assignment,
            planned_route,
            arm,
            cohort_assignment,
            learning_generation_id,
            control_generation,
            registration_epoch,
            writer,
            deadline,
        )
        .await
    }

    #[allow(clippy::too_many_arguments)]
    async fn finish_active_preparation(
        &self,
        context: &LlmExecutionContextSnapshot,
        pool: PoolConfig,
        outcome: OutcomeConfig,
        compiled_outcome: Arc<CompiledOutcomePolicyV1>,
        outcome_policy_hash: String,
        envelope: crate::adapter::RouterRequestEnvelope,
        audited: ActiveAuditedQueryV2,
        assignment: RandomizedCohortAssignment,
        planned_route: ActivePlannedRouteV2,
        arm: ActiveAssignmentArm,
        cohort_assignment: Arc<CohortAssignmentAuthority>,
        learning_generation_id: Uuid,
        control_generation: u64,
        registration_epoch: u64,
        writer: LedgerWriterClient,
        deadline: Instant,
    ) -> Result<PreparedActiveRouteV2, &'static str> {
        let active_policy = pool
            .learning
            .as_ref()
            .and_then(|learning| learning.complete_active_policy())
            .ok_or(HEALTH_ACTIVE_COMPUTE_FAILURE)?;
        let ActiveAuditedQueryV2 {
            audit,
            winner,
            fresh_gate_audit_json,
            fresh_gate_audit_hash,
            anchor_shadow_lower_bound_bits,
        } = audited;
        let rewritten = if assignment.cohort() == RandomizedCohort::ActiveCanary {
            Some(
                rewrite_active_winner_v2(&envelope, &audit.parent.recommended_model)
                    .map_err(recommendation_compute_health_code)?,
            )
        } else {
            None
        };
        let active_root_window_id = Uuid::now_v7();
        let active_assignment_id = Uuid::now_v7();
        let active_dispatch_id = rewritten.as_ref().map(|_| Uuid::now_v7());
        let relearning_cooloff_seconds = u32::try_from(outcome.relearning_cooloff_seconds)
            .map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)?;
        let attribution_horizon = Duration::from_secs(outcome.max_attribution_seconds);
        let attribution_deadline = Instant::now()
            .checked_add(attribution_horizon)
            .ok_or(HEALTH_ACTIVE_COMPUTE_FAILURE)?;
        let attribution_deadline_unix_ms = now_unix_ms_u64()
            .checked_add(
                outcome
                    .max_attribution_seconds
                    .checked_mul(1_000)
                    .ok_or(HEALTH_ACTIVE_COMPUTE_FAILURE)?,
            )
            .ok_or(HEALTH_ACTIVE_COMPUTE_FAILURE)?;
        let opened_after_ingest_seq = self
            .stage_active_outcome_until(
                context,
                active_root_window_id,
                compiled_outcome,
                attribution_deadline,
                attribution_deadline_unix_ms,
                active_dispatch_id.map(|active_dispatch_id| ActiveOutcomeInvalidationAuthorityV2 {
                    key: winner.neighborhood_key.clone(),
                    active_dispatch_id,
                    cooloff_duration_seconds: relearning_cooloff_seconds,
                }),
                registration_epoch,
                deadline,
            )
            .await?;

        let terminal_permit = if active_dispatch_id.is_some() {
            match writer
                .reserve_active_dispatch_terminal_until(deadline)
                .await
            {
                Ok(permit) => Some(permit),
                Err(_) => {
                    self.discard_staged_active(active_root_window_id, context.root_uuid);
                    return Err(HEALTH_ACTIVE_ADMISSION_FAILURE);
                }
            }
        } else {
            None
        };
        let dispatch_guard = terminal_permit.map(|permit| {
            ActiveDispatchGuardV2::new(
                permit,
                active_dispatch_id.expect("terminal permit requires dispatch authority"),
                active_assignment_id,
            )
        });
        let request_identity_hash = rewritten
            .as_ref()
            .map(|request| {
                serde_json::to_value(request)
                    .map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)
                    .and_then(|value| {
                        canonical_sha256(&value).map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)
                    })
            })
            .transpose()?;
        let owner_relation_hash = cohort_assignment
            .protect_pinned_owner(assignment.root_key(), context.trajectory_owner_uuid)
            .to_hex();
        let propensity = assignment.propensity();
        let holdout_threshold = threshold_bytes(assignment.holdout_threshold_numerator())?;
        let canary_threshold = threshold_bytes(assignment.canary_threshold_numerator())?;
        let root = ActiveRootAdmission {
            active_root_window_id,
            active_experiment_id: winner.active_experiment_id,
            active_assignment_id,
            decision_id: audit.parent.decision_id,
            dispatch: active_dispatch_id.zip(request_identity_hash).map(
                |(active_dispatch_id, request_identity_hash)| ActiveDispatchAdmission {
                    active_dispatch_id,
                    request_identity_hash,
                },
            ),
            pool_id: pool.id.clone(),
            candidate_id: audit
                .parent
                .candidate_id
                .clone()
                .ok_or(HEALTH_ACTIVE_COMPUTE_FAILURE)?,
            root_key: assignment.root_key().to_hex(),
            owner_relation_hash,
            config_generation_id: self.state.config_generation_id.clone(),
            learning_generation_id,
            cohort_generation_id: assignment.cohort_generation_id(),
            outcome_policy_hash,
            opened_after_ingest_seq,
            attribution_deadline_unix_ms,
            tranche_ordinal: winner.open_tranche_ordinal,
            arm,
            cohort_threshold_numerator: holdout_threshold,
            selection_threshold_numerator: canary_threshold,
            configured_holdout_probability_bits: assignment.configured_holdout_probability_bits(),
            configured_canary_probability_bits: assignment.configured_active_canary_fraction_bits(),
            effective_arm_probability_bits: propensity.effective_arm_probability_bits(),
            conditional_selection_probability_bits: propensity
                .conditional_selection_probability_bits(),
            propensity_bits: propensity.propensity_bits(),
            control_generation,
        };
        let admission = ActiveDecisionAdmissionV2 {
            audit,
            facts: ActiveDecisionFactsV2 {
                active_neighborhood_state_event_id: winner.active_neighborhood_state_event_id,
                active_outcome_look_id: winner.active_outcome_look_id,
                planned_route,
                control_generation,
                promotion_lower_bound_bits: active_policy.recommend.promotion_lower_bound.to_bits(),
                retention_lower_bound_bits: active_policy.retention_lower_bound.to_bits(),
                configured_holdout_probability_bits: assignment
                    .configured_holdout_probability_bits(),
                configured_canary_probability_bits: assignment
                    .configured_active_canary_fraction_bits(),
                actual_outcome_noninferiority_lower_bits: winner.noninferiority_lower_bits,
                actual_outcome_noninferiority_upper_bits: winner.noninferiority_upper_bits,
                anchor_shadow_lower_bound_bits,
                fallback_reason: None,
                fresh_gate_audit_json,
                fresh_gate_audit_hash,
            },
            root,
        };
        let acknowledgement = writer
            .admit_active_decision_until(admission, deadline)
            .await
            .map_err(|_| HEALTH_ACTIVE_ADMISSION_FAILURE)?;
        if !matches!(
            acknowledgement,
            ActiveDecisionAdmissionAckV2::Applied(_)
                | ActiveDecisionAdmissionAckV2::AlreadyApplied(_)
        ) {
            self.discard_staged_active(active_root_window_id, context.root_uuid);
            return Err(HEALTH_ACTIVE_ADMISSION_FAILURE);
        }

        let admitted = ActiveAdmittedCallV2 {
            active_root_window_id,
            active_dispatch_id,
            raw_root_uuid: context.root_uuid,
            winner,
            relearning_cooloff_seconds,
            api_family: context.api_family,
        };
        let active_outcome = self
            .state
            .active_outcome
            .as_ref()
            .ok_or(HEALTH_ACTIVE_OUTCOME_FAILURE)?;
        if active_outcome
            .commit_until(active_root_window_id, context.root_uuid, deadline)
            .await
            .is_err()
        {
            return Ok(PreparedActiveRouteV2::PostAdmissionFailure { dispatch_guard });
        }
        match (rewritten, dispatch_guard) {
            (Some(request), Some(dispatch_guard)) => Ok(PreparedActiveRouteV2::Candidate {
                request,
                admitted,
                dispatch_guard,
            }),
            (None, None) => Ok(PreparedActiveRouteV2::Anchor { admitted }),
            _ => Err(HEALTH_ACTIVE_COMPUTE_FAILURE),
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn stage_active_outcome_until(
        &self,
        context: &LlmExecutionContextSnapshot,
        active_root_window_id: Uuid,
        policy: Arc<CompiledOutcomePolicyV1>,
        attribution_deadline: Instant,
        attribution_deadline_unix_ms: u64,
        invalidation: Option<ActiveOutcomeInvalidationAuthorityV2>,
        registration_epoch: u64,
        deadline: Instant,
    ) -> Result<u64, &'static str> {
        timeout_until(deadline, self.state.subscriber_barrier.flush())
            .await
            .map_err(|_| HEALTH_ACTIVE_OUTCOME_FAILURE)?
            .map_err(|_| HEALTH_ACTIVE_OUTCOME_FAILURE)?;
        let opened_after_ingest_seq = self.state.loss.current_ingest_seq();
        let (reply, receiver) = oneshot::channel();
        let command = CoordinatorCommand::StageActiveOutcome {
            call_uuid: context.call_uuid,
            admission_epoch: registration_epoch,
            stage: ActiveOutcomeStageV2 {
                active_root_window_id,
                raw_root_uuid: context.root_uuid,
                pinned_owner_uuid: context.trajectory_owner_uuid,
                policy,
                opened_after_ingest_seq,
                opening_loss: self.state.loss.snapshot(),
                attribution_deadline,
                attribution_deadline_unix_ms,
                invalidation,
            },
            reply,
        };
        timeout_until(deadline, self.state.coordinator_tx.send(command))
            .await
            .map_err(|_| HEALTH_ACTIVE_OUTCOME_FAILURE)?
            .map_err(|_| HEALTH_ACTIVE_OUTCOME_FAILURE)?;
        timeout_until(deadline, receiver)
            .await
            .map_err(|_| HEALTH_ACTIVE_OUTCOME_FAILURE)?
            .map_err(|_| HEALTH_ACTIVE_OUTCOME_FAILURE)?
            .map_err(|_| HEALTH_ACTIVE_OUTCOME_FAILURE)?;
        Ok(opened_after_ingest_seq)
    }

    fn discard_staged_active(&self, active_root_window_id: Uuid, raw_root_uuid: Uuid) {
        if let Some(active) = self.state.active_outcome.as_ref() {
            active.discard(active_root_window_id, raw_root_uuid);
        }
    }

    async fn finish_active_dispatch(
        &self,
        guard: ActiveDispatchGuardV2,
        admitted: &ActiveAdmittedCallV2,
        state: ActiveDispatchTerminalState,
        stable_error_class: Option<String>,
        provider_receipt_hash: Option<String>,
        representative: RepresentativeResultV1,
    ) {
        let deadline = terminal_write_deadline();
        let terminal = guard
            .finish(
                state,
                stable_error_class.clone(),
                provider_receipt_hash,
                deadline,
            )
            .await;
        if !matches!(
            terminal,
            Ok(ActiveDispatchTerminalAck::Applied | ActiveDispatchTerminalAck::AlreadyApplied)
        ) {
            self.record_health(HEALTH_ACTIVE_OUTCOME_FAILURE);
            return;
        }
        self.record_active_representative(admitted, representative, stable_error_class)
            .await;
    }

    async fn record_active_representative(
        &self,
        admitted: &ActiveAdmittedCallV2,
        result: RepresentativeResultV1,
        stable_error_class: Option<String>,
    ) {
        let deadline = terminal_write_deadline();
        let barrier = timeout_until(deadline, self.state.subscriber_barrier.flush()).await;
        if !matches!(barrier, Ok(Ok(()))) {
            self.state.loss.record_classification_loss();
        }
        let (reply, receiver) = oneshot::channel();
        let command = CoordinatorCommand::ActiveRepresentativeBarrier {
            observation: ActiveRepresentativeObservationV2 {
                active_root_window_id: admitted.active_root_window_id,
                raw_root_uuid: admitted.raw_root_uuid,
                result,
                stable_error_class,
                ingest_seq: self.state.loss.current_ingest_seq(),
                observed_at_unix_ms: now_unix_ms_u64(),
                terminal_loss: self.state.loss.snapshot(),
            },
            reply,
        };
        let sent = timeout_until(deadline, self.state.coordinator_tx.send(command)).await;
        if !matches!(sent, Ok(Ok(())))
            || !matches!(timeout_until(deadline, receiver).await, Ok(Ok(Ok(()))))
        {
            self.record_health(HEALTH_ACTIVE_OUTCOME_FAILURE);
        }
    }

    async fn invalidate_active_winner(&self, admitted: &ActiveAdmittedCallV2, reason: &str) {
        let Some(writer) = self.state.writer_client.as_ref() else {
            self.record_health(HEALTH_ACTIVE_OUTCOME_FAILURE);
            return;
        };
        let deadline = terminal_write_deadline();
        let acknowledgement = writer
            .invalidate_active_neighborhood_until(
                ActiveNeighborhoodInvalidation {
                    invalidated_state_event_id: Uuid::now_v7(),
                    cooloff_state_event_id: Uuid::now_v7(),
                    key: admitted.winner.neighborhood_key.clone(),
                    cause_active_dispatch_id: admitted.active_dispatch_id,
                    cause_outcome_id: None,
                    stable_reason: reason.to_string(),
                    cooloff_duration_seconds: admitted.relearning_cooloff_seconds,
                },
                deadline,
            )
            .await;
        if !matches!(
            acknowledgement,
            Ok(ActiveNeighborhoodMutationAck::Applied(_)
                | ActiveNeighborhoodMutationAck::AlreadyApplied(_)
                | ActiveNeighborhoodMutationAck::AlreadyCurrent(_)
                | ActiveNeighborhoodMutationAck::CoolingOff(_))
        ) {
            self.record_health(HEALTH_ACTIVE_OUTCOME_FAILURE);
        }
    }

    fn prepare_sample_intent_fail_open(
        &self,
        context: &LlmExecutionContextSnapshot,
        request: &LlmRequest,
        replay: Option<&Arc<dyn LlmReplayTransport>>,
    ) -> Option<SampleIntent> {
        if context.call_role != LlmCallRole::Primary {
            return None;
        }
        let registration_epoch = self.register_primary_call(context)?;
        if !self.state.sampling_enabled.load(Ordering::Acquire)
            || self.state.permanent_runtime_fault.load(Ordering::Acquire)
        {
            return None;
        }

        let inspected = catch_unwind(AssertUnwindSafe(
            || -> Result<Option<SampleIntent>, IneligibilityReason> {
                let prepared = self.prepare_sample(context, request, replay)?;
                let _admission = self
                    .state
                    .admission
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                if !self.state.accepting.load(Ordering::Acquire)
                    || !self.state.sampling_enabled.load(Ordering::Acquire)
                    || self.state.permanent_runtime_fault.load(Ordering::Acquire)
                    || self.state.admission_epoch.load(Ordering::Acquire) != registration_epoch
                {
                    return Ok(None);
                }
                let generation_snapshot = match self.state.control_authority.as_ref() {
                    Some(controls) => {
                        let generation = controls
                            .current_generation_snapshot()
                            .ok_or(IneligibilityReason::RuntimeFailure)?;
                        let pool = generation
                            .control
                            .pools
                            .get(&prepared.pool.id)
                            .ok_or(IneligibilityReason::RuntimeFailure)?;
                        if pool.effective.paused {
                            return Ok(None);
                        }
                        Some(generation.control)
                    }
                    None => None,
                };
                if !should_sample(
                    self.state.sampler.as_ref(),
                    prepared.pool.sampling_probability,
                )? {
                    return Ok(None);
                }
                let proposal_permit = self
                    .state
                    .pool_resources
                    .get(&prepared.pool.id)
                    .ok_or(IneligibilityReason::RuntimeFailure)?
                    .sample_intents
                    .clone()
                    .try_acquire_owned()
                    .map_err(|_| IneligibilityReason::IntentCapacity)?;
                Ok(Some(SampleIntent {
                    prepared,
                    generation_snapshot,
                    admission_epoch: registration_epoch,
                    proposal_permit,
                }))
            },
        ));
        match inspected {
            Ok(Ok(intent)) => intent,
            Ok(Err(reason)) => {
                self.record_health(reason.code());
                None
            }
            Err(_) => {
                self.record_health(IneligibilityReason::RuntimeFailure.code());
                None
            }
        }
    }

    fn register_primary_call(&self, context: &LlmExecutionContextSnapshot) -> Option<u64> {
        let admission_epoch = {
            let _admission = self
                .state
                .admission
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if !self.state.accepting.load(Ordering::Acquire) {
                return None;
            }
            self.state.admission_epoch.load(Ordering::Acquire)
        };
        let owner_path = match catch_unwind(AssertUnwindSafe(|| {
            project_owner_path(
                &context.trajectory_owner_path,
                self.state.max_event_projection_bytes,
            )
        })) {
            Ok(Ok(owner_path)) => owner_path,
            Ok(Err(reason)) => {
                self.record_registration_loss(reason.code());
                return None;
            }
            Err(_) => {
                self.record_registration_loss(IneligibilityReason::RuntimeFailure.code());
                return None;
            }
        };
        let _admission = self
            .state
            .admission
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if !self.state.accepting.load(Ordering::Acquire)
            || self.state.admission_epoch.load(Ordering::Acquire) != admission_epoch
        {
            return None;
        }
        let command = CoordinatorCommand::RegisterPrimaryCall {
            registration: PrimaryCallRegistration {
                call_uuid: context.call_uuid,
                root_uuid: context.root_uuid,
                parent_uuid: context.parent_uuid,
                owner_uuid: context.trajectory_owner_uuid,
                owner_path,
                api_family: context.api_family,
            },
            loss: self.state.loss.snapshot(),
            admission_epoch,
        };
        if self.state.coordinator_tx.try_send(command).is_err() {
            self.record_registration_loss(IneligibilityReason::QueueFull.code());
            return None;
        }
        Some(admission_epoch)
    }

    fn prepare_sample(
        &self,
        context: &LlmExecutionContextSnapshot,
        request: &LlmRequest,
        replay: Option<&Arc<dyn LlmReplayTransport>>,
    ) -> Result<PreparedSample, IneligibilityReason> {
        let anchor_model = request
            .content
            .get("model")
            .and_then(Json::as_str)
            .ok_or(IneligibilityReason::MissingModel)?;
        let matched = self.state.matcher.match_pool(anchor_model, context);
        if matched.health_reason.is_some() {
            return Err(IneligibilityReason::AmbiguousPool);
        }
        let pool_id = matched
            .matched_pool_id
            .ok_or(IneligibilityReason::NoMatchingPool)?;
        let pool = self
            .state
            .config
            .pools
            .iter()
            .find(|pool| pool.id == pool_id)
            .ok_or(IneligibilityReason::RuntimeFailure)?;
        let replay_transport = replay.cloned().ok_or(IneligibilityReason::MissingReplay)?;
        let outcome = preflight(
            context,
            request,
            Some(&replay_transport),
            pool,
            &self.state.adapter,
        )?;
        if outcome.candidates.is_empty()
            || outcome.eligible_candidate_count < outcome.candidates.len()
            || outcome.candidates.iter().any(|candidate| {
                candidate
                    .request
                    .content
                    .get("model")
                    .and_then(Json::as_str)
                    != Some(candidate.config.model.as_str())
            })
            || outcome.candidates.windows(2).any(|pair| {
                (&pair[0].config.cost_rank, &pair[0].config.id)
                    > (&pair[1].config.cost_rank, &pair[1].config.id)
            })
        {
            return Err(IneligibilityReason::RuntimeFailure);
        }
        let replay_facts = ReplayCapabilityFactsV1::from_capability(&outcome.replay_capability)?;
        let candidate_facts =
            project_candidate_facts(&outcome.candidates, &outcome.request_projection)?;
        let owner_path = project_owner_path(
            &context.trajectory_owner_path,
            pool.lookahead.max_bytes_per_window,
        )?;
        let PreflightOutcome {
            replay_capability: _,
            envelope,
            request_projection,
            routing_projection,
            candidates,
            eligible_candidate_count: _,
            eligible_candidate_facts: _,
            rejected_candidates,
        } = outcome;
        drop(rejected_candidates);
        Ok(PreparedSample {
            call: PreparedCallIdentity {
                call_uuid: context.call_uuid,
                root_uuid: context.root_uuid,
                owner_uuid: context.trajectory_owner_uuid,
                api_family: context.api_family,
            },
            pool: PreparedPool {
                id: pool.id.clone(),
                anchor_model_revision: pool.anchor_revision.clone(),
                sampling_probability: pool.sampling_probability,
                requested_progress: pool.lookahead.primary_llm_completions,
                deadline_seconds: pool.lookahead.deadline_seconds,
                lifecycle_presets: pool.lookahead.lifecycle_presets.clone(),
                max_events: pool.lookahead.max_events_per_window,
                max_bytes: pool.lookahead.max_bytes_per_window,
            },
            envelope,
            request_projection,
            routing_projection,
            candidates,
            replay_transport,
            replay_facts,
            candidate_facts,
            owner_path,
        })
    }

    fn propose_anchor_fail_open(&self, intent: SampleIntent, response: &Json) {
        let projected = catch_unwind(AssertUnwindSafe(|| {
            project_anchor_response(
                intent.prepared.call.api_family,
                response,
                intent.prepared.pool.max_bytes,
            )
        }));
        let response_projection = match projected {
            Ok(Ok(projection)) => projection,
            Ok(Err(reason)) => {
                self.record_health(reason.code());
                return;
            }
            Err(_) => {
                self.record_health(IneligibilityReason::RuntimeFailure.code());
                return;
            }
        };

        let proposed = catch_unwind(AssertUnwindSafe(|| {
            self.propose_anchor(intent, response_projection)
        }));
        match proposed {
            Ok(Ok(())) => {}
            Ok(Err(reason)) => self.record_health(reason.code()),
            Err(_) => self.record_health(IneligibilityReason::RuntimeFailure.code()),
        }
    }

    fn propose_anchor(
        &self,
        intent: SampleIntent,
        response_projection: crate::trajectory::RouterResponseProjectionV1,
    ) -> Result<(), IneligibilityReason> {
        let _admission = self
            .state
            .admission
            .lock()
            .map_err(|_| IneligibilityReason::RuntimeFailure)?;
        if !self.state.accepting.load(Ordering::Acquire)
            || !self.state.sampling_enabled.load(Ordering::Acquire)
            || self.state.permanent_runtime_fault.load(Ordering::Acquire)
            || self.state.admission_epoch.load(Ordering::Acquire) != intent.admission_epoch
        {
            return Ok(());
        }

        let learning_generation_id = match intent.generation_snapshot.as_ref() {
            Some(captured) => {
                let Some(current) = self
                    .state
                    .control_authority
                    .as_ref()
                    .and_then(ControlRuntimeAuthority::current_generation_snapshot)
                else {
                    return Ok(());
                };
                let Some(captured_pool) = captured.pools.get(&intent.prepared.pool.id) else {
                    return Err(IneligibilityReason::RuntimeFailure);
                };
                let Some(current_pool) = current.control.pools.get(&intent.prepared.pool.id) else {
                    return Err(IneligibilityReason::RuntimeFailure);
                };
                if captured.cohort_generation_id != current.control.cohort_generation_id
                    || captured_pool.learning_generation_id != current_pool.learning_generation_id
                    || current_pool.effective.paused
                {
                    return Ok(());
                }
                captured_pool.learning_generation_id
            }
            None => self
                .state
                .identity
                .learning_generation_id(&intent.prepared.pool.id)
                .ok_or(IneligibilityReason::RuntimeFailure)?,
        };

        let horizon = Duration::from_secs(intent.prepared.pool.deadline_seconds);
        let opened_monotonic = tokio::time::Instant::now();
        let monotonic_deadline = opened_monotonic
            .checked_add(horizon)
            .ok_or(IneligibilityReason::RuntimeFailure)?;
        let opened_at = truncate_utc_to_milliseconds(Utc::now());
        let deadline_delta =
            ChronoDuration::from_std(horizon).map_err(|_| IneligibilityReason::RuntimeFailure)?;
        let deadline_at = opened_at
            .checked_add_signed(deadline_delta)
            .ok_or(IneligibilityReason::RuntimeFailure)?;
        let policy_version_id = self
            .state
            .identity
            .policy_version_id(&intent.prepared.pool.id)
            .ok_or(IneligibilityReason::RuntimeFailure)?
            .to_string();
        let anchor_id = Uuid::now_v7();
        let pending = PendingTrajectoryWindow {
            schema: crate::trajectory::PENDING_TRAJECTORY_SCHEMA_V1.to_string(),
            anchor_id,
            anchor_call_uuid: intent.prepared.call.call_uuid,
            root_uuid: intent.prepared.call.root_uuid,
            owner_uuid: intent.prepared.call.owner_uuid,
            owner_path: intent.prepared.owner_path,
            pool_id: intent.prepared.pool.id.clone(),
            anchor_model_revision: intent.prepared.pool.anchor_model_revision,
            process_instance_id: self.state.identity.process_instance_id,
            project_uuid: self.state.identity.project_uuid,
            project_id: self.state.identity.project_id.clone(),
            config_generation_id: self.state.config_generation_id.clone(),
            policy_version_id,
            learning_generation_id,
            request_projection: intent.prepared.request_projection,
            routing_context_projection: intent.prepared.routing_projection,
            normalized_anchor_response: response_projection,
            replay_capability_facts: intent.prepared.replay_facts,
            candidate_facts: intent.prepared.candidate_facts,
            requested_progress: intent.prepared.pool.requested_progress,
            opened_at,
            deadline_at,
        };
        let seed = TrajectoryWindowSeed::new(
            pending,
            intent.prepared.envelope,
            intent.prepared.replay_transport,
            intent.prepared.candidates,
        );
        let registration = AnchorRegistration {
            seed,
            proposal_permit: Some(intent.proposal_permit),
            monotonic_deadline,
            opened_after_ingest_seq: self.state.loss.current_ingest_seq(),
            loss: self.state.loss.snapshot(),
            admission_epoch: intent.admission_epoch,
            limits: WindowLimits::new(
                intent.prepared.pool.requested_progress,
                intent.prepared.pool.max_events,
                intent.prepared.pool.max_bytes,
                &intent.prepared.pool.lifecycle_presets,
            ),
        };
        self.state
            .coordinator_tx
            .try_send(CoordinatorCommand::RegisterAnchor(Box::new(registration)))
            .map_err(|_| IneligibilityReason::QueueFull)
    }

    fn observe_event_fail_open(&self, event: &Event) {
        if is_internal_router_event(event) {
            return;
        }
        if !self.state.event_ingestion.load(Ordering::Acquire) {
            return;
        }
        let Some(ingest_seq) = self.state.loss.next_ingest_seq() else {
            self.state.loss.record_classification_loss();
            self.state.health.reject();
            return;
        };
        let projected = catch_unwind(AssertUnwindSafe(|| {
            project_captured_event(
                ingest_seq,
                event,
                EventProjectionLimits::for_window_bytes(self.state.max_event_projection_bytes),
            )
        }));
        let command = match projected {
            Ok(ProjectedTrajectoryEvent::Captured(event)) => CoordinatorCommand::ObservedEvent {
                event,
                loss: self.state.loss.snapshot(),
            },
            Ok(event @ ProjectedTrajectoryEvent::Oversized(_)) => {
                CoordinatorCommand::OversizedEvent {
                    event,
                    loss: self.state.loss.snapshot(),
                }
            }
            Err(_) => {
                self.state.loss.record_event_loss(ingest_seq);
                self.state.health.reject();
                return;
            }
        };
        if self.state.coordinator_tx.try_send(command).is_err() {
            self.state.loss.record_event_loss(ingest_seq);
            self.state.health.reject();
        }
    }

    fn record_health(&self, reason: &'static str) {
        if self
            .state
            .coordinator_tx
            .try_send(CoordinatorCommand::Health(reason))
            .is_err()
        {
            self.state.health.reject();
        }
    }

    fn record_registration_loss(&self, reason: &'static str) {
        self.state.loss.record_classification_loss();
        self.state.sampling_enabled.store(false, Ordering::Release);
        if self
            .state
            .coordinator_tx
            .try_send(CoordinatorCommand::RegistrationLoss {
                loss: self.state.loss.snapshot(),
                reason,
            })
            .is_err()
        {
            self.state.health.reject();
        }
    }

    fn close_admissions(&self) {
        self.state.provider_admission.close();
        if let Some(authority) = self.state.control_authority.as_ref() {
            authority.unpublish();
        }
        if let Some(monitor) = self
            .control_monitor
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
        {
            monitor.stop.send_replace(true);
        }
        if let Some(live_embedding) = self.state.live_embedding.as_ref() {
            live_embedding.close_admission();
        }
        let _admission = self
            .state
            .admission
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(controls) = self.state.scheduler_controls.as_ref() {
            controls.start_gate.close();
            controls.admissions.close();
        }
        if self.state.accepting.swap(false, Ordering::AcqRel) {
            let _ = self.state.admission_epoch.fetch_update(
                Ordering::AcqRel,
                Ordering::Acquire,
                |current| Some(current.saturating_add(1)),
            );
        }
    }

    fn stop_intake(&self) {
        self.close_admissions();
    }

    fn stop_intake_until(&self, deadline: Instant) {
        if let Some(delivery) = self.state.recommendation_delivery.as_ref() {
            delivery.close_until(deadline);
        }
        self.close_admissions();
        if let Some(live_embedding) = self.state.live_embedding.as_ref() {
            live_embedding.close_until(deadline);
        }
        if let Some(background) = self.abort_handles.background.as_ref() {
            background.stop_until(deadline);
        }
    }

    fn process_stop_command(&self) -> PluginResult<ProcessStop> {
        let mut command = self
            .process_stop
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if let Some(command) = *command {
            return Ok(command);
        }
        let created_at_unix_ms = Utc::now().timestamp_millis().max(0);
        let frozen = ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), created_at_unix_ms).map_err(
            |error| PluginError::Internal(format!("Router process stop command failed: {error}")),
        )?;
        *command = Some(frozen);
        Ok(frozen)
    }

    fn begin_drain(&self) -> PluginResult<bool> {
        let _lifecycle = self
            .lifecycle_gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        match self.lifecycle.compare_exchange(
            LIFECYCLE_RUNNING,
            LIFECYCLE_DRAINING,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => Ok(true),
            Err(LIFECYCLE_DRAINED) => Ok(false),
            Err(LIFECYCLE_ABORTED) => Err(PluginError::Internal(
                "Router runtime was aborted before drain".into(),
            )),
            Err(_) => Err(PluginError::Internal(
                "Router runtime drain is already in progress".into(),
            )),
        }
    }

    pub(crate) async fn rollback_activation(&self) -> PluginResult<()> {
        self.stop_intake();
        self.request_activation_rollback();
        let stop_result = if let Some(writer) = self.state.writer_client.as_ref() {
            async {
                let command = self.process_stop_command()?;
                let deadline = Instant::now()
                    .checked_add(ACTIVATION_ROLLBACK_TIMEOUT)
                    .ok_or_else(|| {
                        PluginError::Internal(
                            "Router activation rollback deadline overflowed".into(),
                        )
                    })?;
                match writer.stop_process_until(command, deadline).await {
                    Ok(
                        ProcessCommandAck::Applied
                        | ProcessCommandAck::AlreadyApplied
                        | ProcessCommandAck::OriginatingProcessNotLive,
                    ) => Ok(()),
                    Ok(ProcessCommandAck::Conflict | ProcessCommandAck::TransactionNotStarted) => {
                        Err(PluginError::Internal(
                            "Router activation rollback did not stop its ledger process".into(),
                        ))
                    }
                    Err(error) => Err(PluginError::Internal(format!(
                        "Router activation rollback writer failed: {}",
                        error.code()
                    ))),
                }
            }
            .await
        } else {
            Ok(())
        };
        self.abort();
        stop_result
    }

    async fn drain(&self, deadline: Instant) -> PluginResult<()> {
        self.state.provider_admission.close();
        if !self.begin_drain()? {
            return Ok(());
        }
        self.stop_intake_until(deadline);
        if let Some(controls) = self.state.scheduler_controls.as_ref() {
            controls.deadlines.install_shutdown_deadline(deadline);
        }
        let tokio_deadline = tokio::time::Instant::from_std(deadline);
        let result = tokio::time::timeout_at(tokio_deadline, async {
            if self.state.active_outcome.is_some() {
                let _foreground_quiescence = self
                    .state
                    .foreground_admission
                    .clone()
                    .acquire_many_owned(4)
                    .await
                    .map_err(|_| {
                        PluginError::Internal(
                            "Router foreground admission closed during drain".into(),
                        )
                    })?;
                self.state.subscriber_barrier.flush().await.map_err(|_| {
                    PluginError::Internal(
                        "Router subscriber flush failed during Active drain".into(),
                    )
                })?;
                let (reply, receiver) = oneshot::channel();
                self.state
                    .coordinator_tx
                    .send(CoordinatorCommand::ShutdownActiveOutcomes {
                        observed_at_unix_ms: now_unix_ms_u64(),
                        reply,
                    })
                    .await
                    .map_err(|_| {
                        PluginError::Internal(
                            "Router coordinator closed before Active outcome shutdown".into(),
                        )
                    })?;
                receiver
                    .await
                    .map_err(|_| {
                        PluginError::Internal(
                            "Router Active outcome shutdown acknowledgement was lost".into(),
                        )
                    })?
                    .map_err(|_| {
                        PluginError::Internal("Router Active outcome shutdown failed".into())
                    })?;
            }
            if !self.state.finished.load(Ordering::Acquire) {
                let shutdown = CoordinatorCommand::Shutdown {
                    admission_epoch: self.state.admission_epoch.load(Ordering::Acquire),
                    loss: self.state.loss.snapshot(),
                };
                self.state
                    .coordinator_tx
                    .send(shutdown)
                    .await
                    .map_err(|_| {
                        PluginError::Internal(
                            "Router coordinator closed before graceful shutdown".into(),
                        )
                    })?;
                loop {
                    if self.state.finished.load(Ordering::Acquire) {
                        break;
                    }
                    let notified = self.state.finished_notify.notified();
                    if self.state.finished.load(Ordering::Acquire) {
                        break;
                    }
                    notified.await;
                }
            }

            let coordinator = RetainedResource::take(&self.coordinator, &self.lifecycle);
            if let Some(coordinator) = coordinator {
                coordinator.join().await.map_err(|error| {
                    PluginError::Internal(format!("Router coordinator task failed: {error}"))
                })?;
            }
            let active_outcome = RetainedResource::take(&self.active_outcome, &self.lifecycle);
            if let Some(active_outcome) = active_outcome {
                active_outcome.join().await.map_err(|error| {
                    PluginError::Internal(format!("Router Active outcome task failed: {error}"))
                })?;
            }
            if let Some(controls) = self.state.scheduler_controls.as_ref() {
                controls.admissions.close();
            }
            let scheduler = RetainedResource::take(&self.scheduler, &self.lifecycle);
            if let Some(scheduler) = scheduler {
                let exit = scheduler.join().await.map_err(|error| {
                    PluginError::Internal(format!("Router scheduler task failed: {error}"))
                })?;
                if exit.summary().first_failure.is_some() || exit.retained_batch_count() != 0 {
                    return Err(PluginError::Internal(format!(
                        "Router scheduler stopped with failure {:?} and {} retained batches",
                        exit.summary().first_failure,
                        exit.retained_batch_count()
                    )));
                }
            }
            if let Some(controls) = self.state.scheduler_controls.as_ref()
                && controls.active_replays.close_and_cancel_all() != 0
            {
                return Err(PluginError::Internal(
                    "Router scheduler stopped with active replay calls".into(),
                ));
            }
            if let Some(delivery) = self.state.recommendation_delivery.as_ref() {
                delivery.drain_until(deadline).await.map_err(|error| {
                    PluginError::Internal(format!(
                        "Router recommendation delivery drain failed: {error:?}"
                    ))
                })?;
            }
            if let Some(live_embedding) = self.state.live_embedding.as_ref()
                && !live_embedding.drain_until(deadline).await
            {
                return Err(PluginError::Internal(
                    "Router live embedding drain deadline expired".into(),
                ));
            }
            let background = RetainedResource::take(&self.background, &self.lifecycle);
            if let Some(mut background) = background {
                background.value().stop_until(deadline);
                let exit = background.value_mut().join().await.map_err(|error| {
                    PluginError::Internal(format!(
                        "Router background supervisor task failed: {error}"
                    ))
                })?;
                if let Some(failure) = exit.first_failure() {
                    return Err(PluginError::Internal(format!(
                        "Router background supervisor failed: {}",
                        failure.code()
                    )));
                }
                background.finish();
            }
            let retention = RetainedResource::take(&self.retention, &self.lifecycle);
            let heartbeat = RetainedResource::take(&self.heartbeat, &self.lifecycle);
            let control_monitor = self
                .control_monitor
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .take();
            let writer = RetainedResource::take(&self.writer, &self.lifecycle);
            #[cfg(test)]
            let drain_retained_owners_pause = {
                self.drain_retained_owners_pause
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take()
            };
            #[cfg(test)]
            if let Some((retained, release)) = drain_retained_owners_pause {
                retained.wait().await;
                release.wait().await;
            }
            if let Some(mut retention) = retention {
                retention.value().pressure.send_replace(None);
                retention.value().stop.send_replace(true);
                (&mut retention.value_mut().task)
                    .await
                    .map_err(|error| {
                        PluginError::Internal(format!("Router retention task failed: {error}"))
                    })?
                    .map_err(|reason| {
                        PluginError::Internal(format!("Router retention failed: {reason}"))
                    })?;
                retention.finish();
            }
            if let Some(mut heartbeat) = heartbeat {
                heartbeat.value().stop.send_replace(true);
                (&mut heartbeat.value_mut().task)
                    .await
                    .map_err(|error| {
                        PluginError::Internal(format!("Router heartbeat task failed: {error}"))
                    })?
                    .map_err(|reason| {
                        PluginError::Internal(format!("Router heartbeat failed: {reason}"))
                    })?;
                heartbeat.finish();
            }
            if let Some(mut monitor) = control_monitor {
                monitor.stop.send_replace(true);
                (&mut monitor.task).await.map_err(|error| {
                    PluginError::Internal(format!("Router control monitor task failed: {error}"))
                })?;
            }
            if let Some(writer_client) = self.state.writer_client.as_ref() {
                let stop = self.process_stop_command()?;
                match writer_client.stop_process_until(stop, deadline).await {
                    Ok(ProcessCommandAck::Applied | ProcessCommandAck::AlreadyApplied) => {}
                    Ok(
                        ProcessCommandAck::Conflict
                        | ProcessCommandAck::OriginatingProcessNotLive
                        | ProcessCommandAck::TransactionNotStarted,
                    ) => {
                        return Err(PluginError::Internal(
                            "Router ledger process stop was not applied".into(),
                        ));
                    }
                    Err(error) => {
                        return Err(PluginError::Internal(format!(
                            "Router ledger process stop failed: {}",
                            error.code()
                        )));
                    }
                }
                writer_client.flush_until(deadline).await.map_err(|error| {
                    PluginError::Internal(format!(
                        "Router ledger writer flush failed: {}",
                        error.code()
                    ))
                })?;
            }
            if let Some(read_pool) = self.state.read_pool.as_ref() {
                read_pool.close(deadline).await.map_err(|error| {
                    PluginError::Internal(format!(
                        "Router ledger reader drain failed: {}",
                        error.code()
                    ))
                })?;
            }
            if let Some(mut writer) = writer {
                writer
                    .value_mut()
                    .drain_until(deadline)
                    .await
                    .map_err(|error| {
                        PluginError::Internal(format!(
                            "Router ledger writer drain failed: {}",
                            error.code()
                        ))
                    })?;
                writer.finish();
            }
            Ok(())
        })
        .await
        .map_err(|_| PluginError::Internal("Router runtime drain deadline expired".into()))?;
        if result.is_ok() {
            let _lifecycle = self
                .lifecycle_gate
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            match self.lifecycle.compare_exchange(
                LIFECYCLE_DRAINING,
                LIFECYCLE_DRAINED,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) | Err(LIFECYCLE_DRAINED) => {}
                Err(LIFECYCLE_ABORTED) => {
                    return Err(PluginError::Internal(
                        "Router runtime was aborted during drain".into(),
                    ));
                }
                Err(_) => {
                    return Err(PluginError::Internal(
                        "Router runtime lifecycle changed during drain".into(),
                    ));
                }
            }
        }
        result
    }

    fn abort(&self) {
        let _lifecycle = self
            .lifecycle_gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if matches!(
            self.lifecycle.load(Ordering::Acquire),
            LIFECYCLE_ABORTED | LIFECYCLE_DRAINED
        ) {
            return;
        }
        self.close_admissions();
        if let Some(writer) = self.state.writer_client.as_ref() {
            writer.abort();
        }
        if let Some(delivery) = self.state.recommendation_delivery.as_ref() {
            delivery.abort();
        }
        self.lifecycle.store(LIFECYCLE_ABORTED, Ordering::Release);
        if let Some(live_embedding) = self.state.live_embedding.as_ref() {
            live_embedding.abort();
        }
        if let Some(background) = self.abort_handles.background.as_ref() {
            background.abort();
        }
        if let Some(active_outcome) = self.abort_handles.active_outcome.as_ref() {
            active_outcome.abort();
        }
        let _ = self
            .background
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        self.state.event_ingestion.store(false, Ordering::Release);
        if let Some(controls) = self.state.scheduler_controls.as_ref() {
            controls.cancellation.cancel();
            controls.active_replays.close_and_cancel_all();
            controls.admissions.close();
        }
        self.abort_handles.coordinator.abort();
        if let Some(abort) = self.abort_handles.scheduler.as_ref() {
            abort.abort();
        }
        if let Some(abort) = self.abort_handles.retention.as_ref() {
            abort.abort();
        }
        if let Some(abort) = self.abort_handles.heartbeat.as_ref() {
            abort.abort();
        }
        let _ = self
            .coordinator
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        let _ = self
            .active_outcome
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(heartbeat) = self
            .heartbeat
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            heartbeat.stop.send_replace(true);
        }
        if let Some(retention) = self
            .retention
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            retention.pressure.send_replace(None);
            retention.stop.send_replace(true);
        }
        if let Some(monitor) = self
            .control_monitor
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            monitor.stop.send_replace(true);
            monitor.task.abort();
        }
        let _ = self
            .scheduler
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        if let Some(read_pool) = self.state.read_pool.as_ref() {
            read_pool.abort();
        }
        if let Some(writer) = self
            .writer
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take()
        {
            writer.abort();
        }
    }
}

fn active_route_facts(
    assignment: RandomizedCohortAssignment,
) -> (
    DecisionFinalReasonV1,
    ActivePlannedRouteV2,
    ActiveAssignmentArm,
) {
    match assignment.cohort() {
        RandomizedCohort::ActiveCanary => (
            DecisionFinalReasonV1::ActiveCandidate,
            ActivePlannedRouteV2::Candidate,
            ActiveAssignmentArm::CandidateTreatment,
        ),
        RandomizedCohort::AnchorControl => (
            DecisionFinalReasonV1::ActiveAnchorControl,
            ActivePlannedRouteV2::AnchorControl,
            ActiveAssignmentArm::AnchorControl,
        ),
        RandomizedCohort::AnchorHoldout => (
            DecisionFinalReasonV1::ActiveAnchorHoldout,
            ActivePlannedRouteV2::AnchorHoldout,
            ActiveAssignmentArm::AnchorHoldout,
        ),
    }
}

fn threshold_bytes(value: u128) -> Result<[u8; 8], &'static str> {
    u64::try_from(value)
        .map(u64::to_be_bytes)
        .map_err(|_| HEALTH_ACTIVE_COMPUTE_FAILURE)
}

fn terminal_write_deadline() -> Instant {
    Instant::now()
        .checked_add(Duration::from_secs(5))
        .unwrap_or_else(Instant::now)
}

fn now_unix_ms_u64() -> u64 {
    u64::try_from(Utc::now().timestamp_millis().max(0)).unwrap_or(0)
}

async fn timeout_until<F: Future>(deadline: Instant, future: F) -> Result<F::Output, ()> {
    tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), future)
        .await
        .map_err(|_| ())
}

fn is_internal_router_event(event: &Event) -> bool {
    matches!(
        event.scope_type(),
        Some(ScopeType::Evaluator | ScopeType::Embedder)
    ) || (event.category().map(|category| category.as_str()) == Some("llm")
        && event.llm_call_role() != Some(LlmCallRole::Primary))
}

impl Drop for RouterRuntime {
    fn drop(&mut self) {
        self.abort();
    }
}

fn recommendation_compute_health_code(error: RecommendationErrorV1) -> &'static str {
    match error {
        RecommendationErrorV1::DeadlineExceeded
        | RecommendationErrorV1::RuntimeFailure
        | RecommendationErrorV1::InvalidIdentity
        | RecommendationErrorV1::InvalidPreflight
        | RecommendationErrorV1::InvalidQuery
        | RecommendationErrorV1::InvalidPartition
        | RecommendationErrorV1::ResourceLimit
        | RecommendationErrorV1::Confidence
        | RecommendationErrorV1::DecisionAudit => HEALTH_RECOMMENDATION_COMPUTE_FAILURE,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ActorTaskCompletion {
    Driver,
    Deadline { anchor_id: Uuid, generation: u64 },
    SinkRetry { anchor_id: Uuid, generation: u64 },
    DeliveryRetry { anchor_id: Uuid, generation: u64 },
    SchedulerPressure { pool_id: String, generation: u64 },
}

struct ActorDrivers {
    state: Arc<RuntimeState>,
    sink: Arc<dyn TrajectorySink>,
    delivery: Arc<dyn TrajectoryDelivery>,
    barrier: Arc<dyn SubscriberBarrier>,
    tasks: JoinSet<ActorTaskCompletion>,
    deadlines: BTreeMap<Uuid, (u64, AbortHandle)>,
    sink_retries: BTreeMap<Uuid, (u64, AbortHandle)>,
    delivery_retries: BTreeMap<Uuid, (u64, AbortHandle)>,
    scheduler_pressure: BTreeMap<String, (u64, AbortHandle)>,
}

impl ActorDrivers {
    fn new(
        state: Arc<RuntimeState>,
        sink: Arc<dyn TrajectorySink>,
        delivery: Arc<dyn TrajectoryDelivery>,
        barrier: Arc<dyn SubscriberBarrier>,
    ) -> Self {
        Self {
            state,
            sink,
            delivery,
            barrier,
            tasks: JoinSet::new(),
            deadlines: BTreeMap::new(),
            sink_retries: BTreeMap::new(),
            delivery_retries: BTreeMap::new(),
            scheduler_pressure: BTreeMap::new(),
        }
    }

    fn apply(&mut self, effects: Vec<CoordinatorEffect>) -> bool {
        let mut drained = false;
        for effect in effects {
            match effect {
                CoordinatorEffect::StageActiveOutcome { stage, reply } => {
                    let active_root_window_id = stage.active_root_window_id;
                    let raw_root_uuid = stage.raw_root_uuid;
                    let attribution_deadline = stage.attribution_deadline;
                    let interval_end_unix_ms = stage.attribution_deadline_unix_ms;
                    let result = self
                        .state
                        .active_outcome
                        .as_ref()
                        .ok_or(ActiveOutcomeRuntimeErrorV2::Closed)
                        .and_then(|active| active.try_stage(stage, reply));
                    if result.is_err() {
                        self.state.loss.record_classification_loss();
                    } else {
                        let barrier = self.barrier.clone();
                        let sender = self.state.coordinator_tx.clone();
                        let loss = self.state.loss.clone();
                        self.tasks.spawn(async move {
                            tokio::time::sleep_until(tokio::time::Instant::from_std(
                                attribution_deadline,
                            ))
                            .await;
                            let barrier_succeeded = barrier.flush().await.is_ok();
                            let _ = sender
                                .send(CoordinatorCommand::ActiveDeadlineBarrier(
                                    ActiveDeadlineObservationV2 {
                                        active_root_window_id,
                                        raw_root_uuid,
                                        interval_end_unix_ms,
                                        terminal_loss: loss.snapshot(),
                                        barrier_succeeded,
                                    },
                                ))
                                .await;
                            ActorTaskCompletion::Driver
                        });
                    }
                }
                CoordinatorEffect::ObserveActiveOutcome(observation) => {
                    if self
                        .state
                        .active_outcome
                        .as_ref()
                        .is_some_and(|active| active.try_observe(observation).is_err())
                    {
                        self.state.loss.record_classification_loss();
                    }
                }
                CoordinatorEffect::ObserveActiveOutcomeOversized(observation) => {
                    if self
                        .state
                        .active_outcome
                        .as_ref()
                        .is_some_and(|active| active.try_observe_oversized(observation).is_err())
                    {
                        self.state.loss.record_classification_loss();
                    }
                }
                CoordinatorEffect::ActiveRepresentativeBarrier { observation, reply } => {
                    let result = self
                        .state
                        .active_outcome
                        .as_ref()
                        .ok_or(ActiveOutcomeRuntimeErrorV2::Closed)
                        .and_then(|active| active.try_representative(observation, reply));
                    if result.is_err() {
                        self.state.loss.record_classification_loss();
                    }
                }
                CoordinatorEffect::ActiveDeadlineBarrier(observation) => {
                    if self
                        .state
                        .active_outcome
                        .as_ref()
                        .is_some_and(|active| active.try_deadline(observation).is_err())
                    {
                        self.state.loss.record_classification_loss();
                    }
                }
                CoordinatorEffect::ShutdownActiveOutcomes {
                    observed_at_unix_ms,
                    reply,
                } => {
                    let result = self
                        .state
                        .active_outcome
                        .as_ref()
                        .ok_or(ActiveOutcomeRuntimeErrorV2::Closed)
                        .and_then(|active| active.try_shutdown(observed_at_unix_ms, reply));
                    if result.is_err() {
                        self.state.loss.record_classification_loss();
                    }
                }
                CoordinatorEffect::RecordPending {
                    anchor_id,
                    payload_hash,
                    payload,
                } => {
                    if payload.anchor_id() != anchor_id
                        || payload.payload_hash().as_deref() != Ok(payload_hash.as_str())
                    {
                        self.fail_internal_effect();
                        continue;
                    }
                    let sink = self.sink.clone();
                    let sender = self.state.coordinator_tx.clone();
                    self.tasks.spawn(async move {
                        let ack = sink.record_pending(Arc::new(payload)).await;
                        if let SinkAck::Failed { stable_class, .. } = &ack {
                            let _ = sender
                                .send(CoordinatorCommand::Health(stable_class.as_str()))
                                .await;
                        }
                        let _ = sender.send(CoordinatorCommand::PendingRecorded(ack)).await;
                        ActorTaskCompletion::Driver
                    });
                }
                CoordinatorEffect::RecordTerminal {
                    anchor_id,
                    payload_hash,
                    payload,
                } => {
                    if payload.anchor_id() != anchor_id
                        || payload.payload_hash().as_deref() != Ok(payload_hash.as_str())
                    {
                        self.fail_internal_effect();
                        continue;
                    }
                    let sink = self.sink.clone();
                    let sender = self.state.coordinator_tx.clone();
                    self.tasks.spawn(async move {
                        let ack = sink.record_terminal(Arc::new(payload)).await;
                        if let SinkAck::Failed { stable_class, .. } = &ack {
                            let _ = sender
                                .send(CoordinatorCommand::Health(stable_class.as_str()))
                                .await;
                        }
                        let _ = sender.send(CoordinatorCommand::TerminalRecorded(ack)).await;
                        ActorTaskCompletion::Driver
                    });
                }
                CoordinatorEffect::DeliverWindow { anchor_id, window } => {
                    if window.anchor_id() != anchor_id {
                        self.fail_internal_effect();
                        continue;
                    }
                    let delivery = self.delivery.clone();
                    let sender = self.state.coordinator_tx.clone();
                    self.tasks.spawn(async move {
                        let ack = delivery.deliver(window).await;
                        if let DeliveryAck::Failed { stable_class, .. } = &ack {
                            let _ = sender
                                .send(CoordinatorCommand::Health(stable_class.as_str()))
                                .await;
                        }
                        let _ = sender.send(CoordinatorCommand::WindowDelivered(ack)).await;
                        ActorTaskCompletion::Driver
                    });
                }
                CoordinatorEffect::ScheduleDeadline {
                    anchor_id,
                    generation,
                    deadline,
                } => self.schedule_deadline(anchor_id, generation, deadline),
                CoordinatorEffect::CancelDeadline {
                    anchor_id,
                    generation,
                } => self.cancel_deadline(anchor_id, generation),
                CoordinatorEffect::ScheduleSinkRetry {
                    anchor_id,
                    generation,
                } => self.schedule_sink_retry(anchor_id, generation),
                CoordinatorEffect::ScheduleDeliveryRetry {
                    anchor_id,
                    generation,
                } => self.schedule_delivery_retry(anchor_id, generation),
                CoordinatorEffect::SchedulerPressureClosed {
                    pool_id,
                    pressure_generation,
                } => self.schedule_scheduler_pressure(pool_id, pressure_generation),
                CoordinatorEffect::EvidenceCapacityPressureClosed {
                    pressure_generation,
                } => {
                    if pressure_generation == 0 {
                        self.fail_internal_effect();
                    } else if let Some(pressure) = self.state.retention_pressure.as_ref() {
                        pressure.send_replace(Some(pressure_generation));
                    }
                }
                CoordinatorEffect::SamplingSuppressed(reason) => {
                    self.state.sampling_enabled.store(false, Ordering::Release);
                    self.state.health.accept(reason);
                }
                CoordinatorEffect::SamplingResumed => {
                    let _admission = self
                        .state
                        .admission
                        .lock()
                        .unwrap_or_else(|error| error.into_inner());
                    if self.state.accepting.load(Ordering::Acquire)
                        && !self.state.permanent_runtime_fault.load(Ordering::Acquire)
                    {
                        self.state.sampling_enabled.store(true, Ordering::Release);
                    }
                }
                CoordinatorEffect::AnchorRefused { anchor_id, reason } => {
                    if anchor_id.is_nil() {
                        self.fail_internal_effect();
                    }
                    self.state.health.accept(anchor_refusal_code(reason));
                }
                CoordinatorEffect::Health(reason) => self.state.health.accept(reason),
                CoordinatorEffect::Drained => drained = true,
            }
        }
        self.track_task_count();
        drained
    }

    fn fail_internal_effect(&self) {
        self.state.sampling_enabled.store(false, Ordering::Release);
        self.state
            .health
            .accept(IneligibilityReason::RuntimeFailure.code());
    }

    fn schedule_deadline(
        &mut self,
        anchor_id: Uuid,
        generation: u64,
        deadline: tokio::time::Instant,
    ) {
        if let Some((_, previous)) = self.deadlines.remove(&anchor_id) {
            previous.abort();
        }
        let barrier = self.barrier.clone();
        let sender = self.state.coordinator_tx.clone();
        let loss = self.state.loss.clone();
        let handle = self.tasks.spawn(async move {
            tokio::time::sleep_until(deadline).await;
            let command = if barrier.flush().await.is_ok() {
                CoordinatorCommand::DeadlineElapsed {
                    anchor_id,
                    generation,
                    loss: loss.snapshot(),
                }
            } else {
                CoordinatorCommand::DeadlineBarrierFailed {
                    anchor_id,
                    generation,
                    loss: loss.snapshot(),
                }
            };
            let _ = sender.send(command).await;
            ActorTaskCompletion::Deadline {
                anchor_id,
                generation,
            }
        });
        self.deadlines.insert(anchor_id, (generation, handle));
    }

    fn cancel_deadline(&mut self, anchor_id: Uuid, generation: u64) {
        if self
            .deadlines
            .get(&anchor_id)
            .is_some_and(|(current, _)| *current == generation)
            && let Some((_, handle)) = self.deadlines.remove(&anchor_id)
        {
            handle.abort();
        }
    }

    fn schedule_sink_retry(&mut self, anchor_id: Uuid, generation: u64) {
        if let Some((_, previous)) = self.sink_retries.remove(&anchor_id) {
            previous.abort();
        }
        let sender = self.state.coordinator_tx.clone();
        let handle = self.tasks.spawn(async move {
            tokio::time::sleep(retry_delay(generation)).await;
            let _ = sender
                .send(CoordinatorCommand::SinkRetryElapsed {
                    anchor_id,
                    generation,
                })
                .await;
            ActorTaskCompletion::SinkRetry {
                anchor_id,
                generation,
            }
        });
        self.sink_retries.insert(anchor_id, (generation, handle));
    }

    fn schedule_delivery_retry(&mut self, anchor_id: Uuid, generation: u64) {
        if let Some((_, previous)) = self.delivery_retries.remove(&anchor_id) {
            previous.abort();
        }
        let sender = self.state.coordinator_tx.clone();
        let handle = self.tasks.spawn(async move {
            tokio::time::sleep(retry_delay(generation)).await;
            let _ = sender
                .send(CoordinatorCommand::DeliveryRetryElapsed {
                    anchor_id,
                    generation,
                })
                .await;
            ActorTaskCompletion::DeliveryRetry {
                anchor_id,
                generation,
            }
        });
        self.delivery_retries
            .insert(anchor_id, (generation, handle));
    }

    fn schedule_scheduler_pressure(&mut self, pool_id: String, generation: u64) {
        if pool_id.is_empty() || generation == 0 {
            self.fail_internal_effect();
            return;
        }
        let Some(controls) = self.state.scheduler_controls.as_ref() else {
            self.fail_internal_effect();
            return;
        };
        let Ok(pressure) = controls.admissions.subscribe_pressure(&pool_id) else {
            self.fail_internal_effect();
            return;
        };
        if let Some((_, previous)) = self.scheduler_pressure.remove(&pool_id) {
            previous.abort();
        }
        let sender = self.state.coordinator_tx.clone();
        let waiter_key = pool_id.clone();
        let completion_pool_id = pool_id.clone();
        let handle = self.tasks.spawn(await_scheduler_pressure_recovery(
            pressure,
            sender,
            pool_id,
            completion_pool_id,
            generation,
        ));
        self.scheduler_pressure
            .insert(waiter_key, (generation, handle));
    }

    fn task_completed(&mut self, completion: ActorTaskCompletion) {
        match completion {
            ActorTaskCompletion::Driver => {}
            ActorTaskCompletion::Deadline {
                anchor_id,
                generation,
            } => remove_completed_task(&mut self.deadlines, anchor_id, generation),
            ActorTaskCompletion::SinkRetry {
                anchor_id,
                generation,
            } => remove_completed_task(&mut self.sink_retries, anchor_id, generation),
            ActorTaskCompletion::DeliveryRetry {
                anchor_id,
                generation,
            } => remove_completed_task(&mut self.delivery_retries, anchor_id, generation),
            ActorTaskCompletion::SchedulerPressure {
                pool_id,
                generation,
            } => {
                if self
                    .scheduler_pressure
                    .get(&pool_id)
                    .is_some_and(|(current, _)| *current == generation)
                {
                    self.scheduler_pressure.remove(&pool_id);
                }
            }
        }
    }

    fn reap_ready(&mut self) {
        while let Some(completed) = self.tasks.try_join_next() {
            match completed {
                Ok(completion) => self.task_completed(completion),
                Err(error) if error.is_cancelled() => {}
                Err(_) => self
                    .state
                    .health
                    .accept(IneligibilityReason::RuntimeFailure.code()),
            }
        }
        self.track_task_count();
    }

    fn track_task_count(&self) {
        #[cfg(test)]
        {
            let count = self.tasks.len();
            self.state.owned_task_count.store(count, Ordering::Release);
            self.state
                .peak_owned_task_count
                .fetch_max(count, Ordering::AcqRel);
        }
    }

    async fn shutdown(mut self) {
        self.deadlines.clear();
        self.sink_retries.clear();
        self.delivery_retries.clear();
        self.scheduler_pressure.clear();
        self.tasks.abort_all();
        while self.tasks.join_next().await.is_some() {}
    }
}

async fn await_scheduler_pressure_recovery(
    mut pressure: watch::Receiver<crate::scheduler_admission::SchedulerPressureState>,
    sender: mpsc::Sender<CoordinatorCommand>,
    pool_id: String,
    completion_pool_id: String,
    generation: u64,
) -> ActorTaskCompletion {
    loop {
        if !pressure.borrow_and_update().pressure_closed {
            let _ = sender
                .send(CoordinatorCommand::SchedulerCapacityRecovered {
                    pool_id,
                    pressure_generation: generation,
                })
                .await;
            break;
        }
        if pressure.changed().await.is_err() {
            break;
        }
    }
    ActorTaskCompletion::SchedulerPressure {
        pool_id: completion_pool_id,
        generation,
    }
}

async fn run_coordinator(
    state: Arc<RuntimeState>,
    mut receiver: mpsc::Receiver<CoordinatorCommand>,
    mut coordinator: Coordinator,
    sink: Arc<dyn TrajectorySink>,
    delivery: Arc<dyn TrajectoryDelivery>,
    barrier: Arc<dyn SubscriberBarrier>,
) {
    struct Finished(Arc<RuntimeState>);
    impl Drop for Finished {
        fn drop(&mut self) {
            self.0.event_ingestion.store(false, Ordering::Release);
            self.0.finished.store(true, Ordering::Release);
            self.0.finished_notify.notify_waiters();
        }
    }

    let _finished = Finished(state.clone());
    let mut drivers = ActorDrivers::new(state.clone(), sink, delivery, barrier);
    loop {
        drivers.reap_ready();
        tokio::select! {
            command = receiver.recv() => {
                let Some(command) = command else {
                    break;
                };
                if drivers.apply(coordinator.handle(command)) {
                    break;
                }
                drivers.reap_ready();
            }
            completed = drivers.tasks.join_next(), if !drivers.tasks.is_empty() => {
                match completed {
                    Some(Ok(completion)) => drivers.task_completed(completion),
                    Some(Err(error)) if error.is_cancelled() => {}
                    Some(Err(_)) => state.health.accept(IneligibilityReason::RuntimeFailure.code()),
                    None => {}
                }
            }
        }
    }
    receiver.close();
    state.event_ingestion.store(false, Ordering::Release);
    drivers.shutdown().await;
}

async fn run_heartbeat(
    state: Arc<RuntimeState>,
    writer: LedgerWriterClient,
    mut stop: watch::Receiver<bool>,
    interval: Duration,
) -> Result<(), &'static str> {
    #[cfg(test)]
    state.heartbeat_worker_ready.store(true, Ordering::Release);
    loop {
        tokio::select! {
            biased;
            changed = stop.changed() => {
                if changed.is_err() || *stop.borrow_and_update() {
                    return Ok(());
                }
            }
            () = tokio::time::sleep(interval) => {
                run_heartbeat_tick(&state, &writer, Utc::now().timestamp_millis()).await?;
            }
        }
    }
}

async fn run_heartbeat_tick(
    state: &RuntimeState,
    writer: &LedgerWriterClient,
    observed_at_unix_ms: i64,
) -> Result<(), &'static str> {
    let renewal = HeartbeatRenewal::new(observed_at_unix_ms).map_err(|_| {
        record_permanent_runtime_fault(state, HEALTH_HEARTBEAT_FAILURE);
        HEALTH_HEARTBEAT_FAILURE
    })?;
    let deadline = Instant::now()
        .checked_add(HEARTBEAT_WRITE_TIMEOUT)
        .ok_or_else(|| {
            record_permanent_runtime_fault(state, HEALTH_HEARTBEAT_FAILURE);
            HEALTH_HEARTBEAT_FAILURE
        })?;
    match writer.renew_heartbeat_until(renewal, deadline).await {
        Ok(HeartbeatAck::Applied { .. } | HeartbeatAck::AlreadyApplied { .. }) => {
            #[cfg(test)]
            state.heartbeat_ticks.fetch_add(1, Ordering::AcqRel);
            Ok(())
        }
        Ok(HeartbeatAck::OriginatingProcessNotLive | HeartbeatAck::TransactionNotStarted)
        | Err(_) => {
            record_permanent_runtime_fault(state, HEALTH_HEARTBEAT_FAILURE);
            Err(HEALTH_HEARTBEAT_FAILURE)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetentionPassFailure {
    Transient,
    Permanent,
}

async fn run_retention(
    state: Arc<RuntimeState>,
    writer: LedgerWriterClient,
    mut stop: watch::Receiver<bool>,
    mut pressure: watch::Receiver<Option<u64>>,
    mut recommendation_pressure: Option<watch::Receiver<u64>>,
    mut manual: mpsc::Receiver<RetentionManualRequest>,
    interval: Duration,
) -> Result<(), &'static str> {
    let first_tick = tokio::time::Instant::now()
        .checked_add(interval)
        .ok_or(HEALTH_RETENTION_FAILURE)?;
    let mut periodic = tokio::time::interval_at(first_tick, interval);
    periodic.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    #[cfg(test)]
    state.retention_worker_ready.store(true, Ordering::Release);

    loop {
        let pressure_generation = tokio::select! {
            biased;
            changed = stop.changed() => {
                if changed.is_err() || *stop.borrow_and_update() {
                    return Ok(());
                }
                continue;
            }
            changed = pressure.changed() => {
                if changed.is_err() {
                    return Ok(());
                }
                let generation = *pressure.borrow_and_update();
                let Some(generation) = generation else {
                    continue;
                };
                Some(generation)
            }
            changed = async {
                match recommendation_pressure.as_mut() {
                    Some(pressure) => pressure.changed().await,
                    None => std::future::pending().await,
                }
            } => {
                if changed.is_err() {
                    recommendation_pressure = None;
                    continue;
                }
                if let Some(pressure) = recommendation_pressure.as_mut() {
                    let _ = *pressure.borrow_and_update();
                }
                None
            }
            request = manual.recv() => {
                let Some(request) = request else {
                    return Ok(());
                };
                match run_retention_pass_with_policy(
                    &state,
                    &writer,
                    request.observed_at_unix_ms,
                ).await {
                    Ok(acknowledgement) => {
                        let _ = request.reply.send(Ok(acknowledgement));
                    }
                    Err(RetentionPassFailure::Transient) => {
                        state.health.accept(HEALTH_RETENTION_FAILURE);
                        let _ = request.reply.send(Err(HEALTH_RETENTION_FAILURE));
                    }
                    Err(RetentionPassFailure::Permanent) => {
                        record_permanent_runtime_fault(&state, HEALTH_RETENTION_FAILURE);
                        let _ = request.reply.send(Err(HEALTH_RETENTION_FAILURE));
                        return Err(HEALTH_RETENTION_FAILURE);
                    }
                }
                continue;
            }
            _ = periodic.tick() => {
                #[cfg(test)]
                state.retention_timer_fires.fetch_add(1, Ordering::AcqRel);
                None
            },
        };

        run_retention_trigger(
            &state,
            &writer,
            &mut stop,
            &mut pressure,
            pressure_generation,
            RETENTION_WRITE_TIMEOUT,
            RETENTION_RETRY_DELAY,
        )
        .await?;
    }
}

async fn run_retention_trigger(
    state: &RuntimeState,
    writer: &LedgerWriterClient,
    stop: &mut watch::Receiver<bool>,
    pressure: &mut watch::Receiver<Option<u64>>,
    pressure_generation: Option<u64>,
    write_timeout: Duration,
    retry_delay: Duration,
) -> Result<(), &'static str> {
    loop {
        if *stop.borrow() {
            return Ok(());
        }
        let request = new_retention_request(Utc::now().timestamp_millis())
            .map_err(|_| HEALTH_RETENTION_FAILURE)?;
        let acknowledgement = loop {
            match run_retention_request_with_policy(state, writer, request, write_timeout).await {
                Ok(acknowledgement) => break acknowledgement,
                Err(RetentionPassFailure::Transient) => {
                    state.health.accept(HEALTH_RETENTION_FAILURE);
                    tokio::select! {
                        biased;
                        changed = stop.changed() => {
                            if changed.is_err() || *stop.borrow_and_update() {
                                return Ok(());
                            }
                        }
                        () = tokio::time::sleep(retry_delay) => {}
                    }
                }
                Err(RetentionPassFailure::Permanent) => {
                    record_permanent_runtime_fault(state, HEALTH_RETENTION_FAILURE);
                    return Err(HEALTH_RETENTION_FAILURE);
                }
            }
        };
        let (summary, observation) = match &acknowledgement {
            RetentionAck::Applied {
                summary,
                observation,
            }
            | RetentionAck::AlreadyApplied {
                summary,
                observation,
            } => (summary, observation),
            RetentionAck::Conflict
            | RetentionAck::VectorIndexUnavailable { .. }
            | RetentionAck::OriginatingProcessNotLive
            | RetentionAck::TransactionNotStarted => {
                unreachable!("retention policy accepts only committed acknowledgements")
            }
        };
        if let Some(generation) = pressure_generation
            && *pressure.borrow() == Some(generation)
            && observation.capacity_available
        {
            #[cfg(test)]
            state
                .retention_recoveries
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(generation);
            let _ = state
                .coordinator_tx
                .send(CoordinatorCommand::EvidenceCapacityRecovered {
                    pressure_generation: generation,
                })
                .await;
        }
        if summary.selected_count < RETENTION_BATCH_ROW_LIMIT && !observation.more_cleanup {
            return Ok(());
        }
        tokio::task::yield_now().await;
    }
}

async fn run_retention_pass_with_policy(
    state: &RuntimeState,
    writer: &LedgerWriterClient,
    observed_at_unix_ms: i64,
) -> Result<RetentionAck, RetentionPassFailure> {
    let request = new_retention_request(observed_at_unix_ms)?;
    run_retention_request_with_policy(state, writer, request, RETENTION_WRITE_TIMEOUT).await
}

async fn run_retention_request_with_policy(
    state: &RuntimeState,
    writer: &LedgerWriterClient,
    request: RetentionRequest,
    write_timeout: Duration,
) -> Result<RetentionAck, RetentionPassFailure> {
    #[cfg(not(test))]
    let _ = state;
    let acknowledgement = run_retention_request(writer, request, write_timeout).await?;
    match acknowledgement {
        RetentionAck::Applied { .. } => {
            #[cfg(test)]
            state.retention_ticks.fetch_add(1, Ordering::AcqRel);
            Ok(acknowledgement)
        }
        RetentionAck::AlreadyApplied { .. } => {
            #[cfg(test)]
            {
                state.retention_ticks.fetch_add(1, Ordering::AcqRel);
                state
                    .retention_already_applied_ticks
                    .fetch_add(1, Ordering::AcqRel);
            }
            Ok(acknowledgement)
        }
        RetentionAck::VectorIndexUnavailable {
            vector_space_id,
            expected_generation,
            expected_manifest_hash,
        } => {
            let deadline = Instant::now()
                .checked_add(write_timeout)
                .ok_or(RetentionPassFailure::Permanent)?;
            let health = writer
                .vector_index_until(
                    VectorIndexWriterCommand::MarkHealth {
                        vector_space_id,
                        expected_generation,
                        expected_manifest_hash,
                        target: VectorIndexHealthTarget::Unavailable,
                        stable_error_class: "router.vector.unavailable".to_string(),
                        observed_at_unix_ms: request.created_at_unix_ms,
                    },
                    deadline,
                )
                .await
                .map_err(classify_retention_writer_failure)?;
            match health {
                VectorIndexWriterAck::HealthMarked(
                    VectorIndexHealthMutationAck::Applied { .. }
                    | VectorIndexHealthMutationAck::AlreadyApplied { .. }
                    | VectorIndexHealthMutationAck::Stale,
                ) => Err(RetentionPassFailure::Transient),
                VectorIndexWriterAck::HealthMarked(
                    VectorIndexHealthMutationAck::Missing | VectorIndexHealthMutationAck::Conflict,
                )
                | VectorIndexWriterAck::GenerationAuthorized(_)
                | VectorIndexWriterAck::RebuildLeaseClaimed(_)
                | VectorIndexWriterAck::GenerationObjectsCreated(_)
                | VectorIndexWriterAck::PointMutated(_)
                | VectorIndexWriterAck::RebuildLeaseMutated(_)
                | VectorIndexWriterAck::RebuildStepped(_)
                | VectorIndexWriterAck::RebuildFlipped(_)
                | VectorIndexWriterAck::RetiredGenerationCleaned(_) => {
                    Err(RetentionPassFailure::Permanent)
                }
            }
        }
        RetentionAck::Conflict
        | RetentionAck::OriginatingProcessNotLive
        | RetentionAck::TransactionNotStarted => Err(RetentionPassFailure::Permanent),
    }
}

fn new_retention_request(
    observed_at_unix_ms: i64,
) -> Result<RetentionRequest, RetentionPassFailure> {
    RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), observed_at_unix_ms)
        .map_err(|_| RetentionPassFailure::Permanent)
}

async fn run_retention_request(
    writer: &LedgerWriterClient,
    request: RetentionRequest,
    write_timeout: Duration,
) -> Result<RetentionAck, RetentionPassFailure> {
    let deadline = Instant::now()
        .checked_add(write_timeout)
        .ok_or(RetentionPassFailure::Permanent)?;
    writer
        .run_retention_until(request, deadline)
        .await
        .map_err(classify_retention_writer_failure)
}

fn classify_retention_writer_failure(error: WriterFailure) -> RetentionPassFailure {
    match error.class() {
        WriterFailureClass::Deadline | WriterFailureClass::Repository(LedgerErrorClass::Busy) => {
            RetentionPassFailure::Transient
        }
        WriterFailureClass::Full
        | WriterFailureClass::Closing
        | WriterFailureClass::Exited
        | WriterFailureClass::Panicked
        | WriterFailureClass::Aborted
        | WriterFailureClass::Protocol
        | WriterFailureClass::Repository(_) => RetentionPassFailure::Permanent,
    }
}

fn record_permanent_runtime_fault(state: &RuntimeState, reason: &'static str) {
    {
        let _admission = state
            .admission
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.permanent_runtime_fault.store(true, Ordering::Release);
        state.sampling_enabled.store(false, Ordering::Release);
    }
    if state
        .coordinator_tx
        .try_send(CoordinatorCommand::PermanentFault(reason))
        .is_err()
    {
        state.health.reject();
    }
}

fn background_failure_notifier(state: Arc<RuntimeState>) -> BackgroundFailureNotifier {
    Arc::new(move |failure: BackgroundFailure| {
        record_permanent_runtime_fault(&state, failure.code());
    })
}

fn retry_delay(generation: u64) -> Duration {
    let shift = generation.saturating_sub(1).min(6) as u32;
    SINK_RETRY_BASE
        .saturating_mul(1u32 << shift)
        .min(SINK_RETRY_MAX)
}

fn remove_completed_task(
    tasks: &mut BTreeMap<Uuid, (u64, AbortHandle)>,
    anchor_id: Uuid,
    generation: u64,
) {
    if tasks
        .get(&anchor_id)
        .is_some_and(|(current, _)| *current == generation)
    {
        tasks.remove(&anchor_id);
    }
}

fn anchor_refusal_code(reason: AnchorRefusalReason) -> &'static str {
    match reason {
        AnchorRefusalReason::IntakeClosed => "router.anchor.intake_closed",
        AnchorRefusalReason::InvalidProposal => "router.anchor.invalid_proposal",
        AnchorRefusalReason::PoolFull => "router.anchor.pool_full",
        AnchorRefusalReason::SamplingSuppressed => "router.anchor.sampling_suppressed",
        AnchorRefusalReason::SinkRejected => "router.anchor.sink_rejected",
    }
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, VecDeque};
    #[cfg(unix)]
    use std::fs;
    use std::sync::Condvar;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use nemo_relay::api::event::{
        BaseEvent, CategoryProfile, EventCategory, ScopeCategory, ScopeEvent,
    };
    use nemo_relay::api::llm::{
        LlmApiFamily, LlmAttributes, LlmCallExecuteV2Params, LlmCallRole,
        LlmRequestInterceptOutcome, LlmTrajectoryScopeSnapshot, llm_call_execute_v2,
    };
    use nemo_relay::api::runtime::{
        LLM_REPLAY_CONTRACT_VERSION, LlmReplayCall, LlmReplayCapability, LlmReplayFactory,
        NemoRelayContextState, create_scope_stack, global_context, set_thread_scope_stack,
    };
    use nemo_relay::api::scope::{
        EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeType, event as emit_scope_event,
        pop_scope, push_scope,
    };
    use nemo_relay::codec::openai_chat::OpenAIChatCodec;
    use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
    use nemo_relay::error::FlowError;
    use nemo_relay::observability::atif::{AtifAgentInfo, AtifExportOptions, AtifExporter};
    use nemo_relay::plugin::{PluginRegistrationContext, rollback_registrations};
    use nemo_relay_adaptive::{
        AcgComponentConfig, AdaptiveConfig, AdaptiveRuntime, BackendSpec, StateConfig,
        TelemetryComponentConfig,
    };
    use serde_json::json;
    use tempfile::tempdir;
    use tokio::sync::oneshot;

    use super::*;
    use crate::config::{
        CandidateCapabilities, CandidateConfig, CanonicalizerConfig, ConcurrencyConfig,
        EmbedderConfig, JUDGE_PROMPT_VERSION_V1, JUDGE_RUBRIC_VERSION_V1, JudgeConfig,
        LearningConfig, LookaheadConfig, PoolSelectorConfig,
    };
    use crate::evaluator::{JUDGE_CALL_NAME, SHADOW_CALL_NAME};
    use crate::recommendation_delivery::{
        RecommendationAdmissionErrorV1, RecommendationDeliveryHealthV1,
    };
    use crate::sink::SinkOperation;
    use crate::trajectory::{
        TrajectoryRejectionReason, TrajectoryTerminalStateV1, TrajectoryTrigger,
    };

    struct FixedSampler {
        value: f64,
        calls: AtomicUsize,
    }

    impl FixedSampler {
        fn new(value: f64) -> Self {
            Self {
                value,
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl Sampler for FixedSampler {
        fn draw_unit(&self) -> Result<f64, IneligibilityReason> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(self.value)
        }
    }

    struct Replay {
        capability: LlmReplayCapability,
        capability_calls: AtomicUsize,
        starts: AtomicUsize,
    }

    struct PanicOnDrop;

    impl Drop for PanicOnDrop {
        fn drop(&mut self) {
            panic!("test drop panic")
        }
    }

    struct PanickingDropReplay {
        capability: LlmReplayCapability,
        _drop_bomb: PanicOnDrop,
    }

    impl LlmReplayTransport for PanickingDropReplay {
        fn capability(&self) -> &LlmReplayCapability {
            &self.capability
        }

        fn start(&self, _request: LlmRequest) -> FlowResult<LlmReplayCall> {
            unreachable!("failed Recommend preprocessing must not start replay")
        }
    }

    impl Replay {
        fn new(family: LlmApiFamily) -> Self {
            Self {
                capability: LlmReplayCapability {
                    contract_version: LLM_REPLAY_CONTRACT_VERSION,
                    api_family: family,
                    transport_identity: "router-runtime-test".into(),
                },
                capability_calls: AtomicUsize::new(0),
                starts: AtomicUsize::new(0),
            }
        }
    }

    impl LlmReplayTransport for Replay {
        fn capability(&self) -> &LlmReplayCapability {
            self.capability_calls.fetch_add(1, Ordering::SeqCst);
            &self.capability
        }

        fn start(&self, _request: LlmRequest) -> FlowResult<LlmReplayCall> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            Err(FlowError::Internal(
                "Spec 04 must never start replay work".into(),
            ))
        }
    }

    struct ProductionReplay {
        capability: LlmReplayCapability,
        responses: Mutex<BTreeMap<String, VecDeque<Json>>>,
        starts: Mutex<Vec<String>>,
    }

    impl ProductionReplay {
        fn new(responses: impl IntoIterator<Item = (String, Vec<Json>)>) -> Arc<Self> {
            Arc::new(Self {
                capability: LlmReplayCapability {
                    contract_version: LLM_REPLAY_CONTRACT_VERSION,
                    api_family: LlmApiFamily::OpenAIChatCompletions,
                    transport_identity: "router-production-runtime-test".into(),
                },
                responses: Mutex::new(
                    responses
                        .into_iter()
                        .map(|(model, responses)| (model, responses.into()))
                        .collect(),
                ),
                starts: Mutex::new(Vec::new()),
            })
        }

        fn started_models(&self) -> Vec<String> {
            self.starts
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .clone()
        }
    }

    impl LlmReplayTransport for ProductionReplay {
        fn capability(&self) -> &LlmReplayCapability {
            &self.capability
        }

        fn start(&self, request: LlmRequest) -> FlowResult<LlmReplayCall> {
            let model = request
                .content
                .get("model")
                .and_then(Json::as_str)
                .ok_or_else(|| FlowError::Internal("production replay omitted model".into()))?
                .to_string();
            let response = self
                .responses
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .get_mut(&model)
                .and_then(VecDeque::pop_front)
                .ok_or_else(|| {
                    FlowError::Internal(format!("production replay has no response for {model}"))
                })?;
            self.starts
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(model);
            Ok(LlmReplayCall::new(async move { Ok(response) }, || {}))
        }
    }

    struct PendingProductionReplay {
        capability: LlmReplayCapability,
        starts: Arc<AtomicUsize>,
        cancellations: Arc<AtomicUsize>,
    }

    impl PendingProductionReplay {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                capability: LlmReplayCapability {
                    contract_version: LLM_REPLAY_CONTRACT_VERSION,
                    api_family: LlmApiFamily::OpenAIChatCompletions,
                    transport_identity: "router-pending-runtime-test".into(),
                },
                starts: Arc::new(AtomicUsize::new(0)),
                cancellations: Arc::new(AtomicUsize::new(0)),
            })
        }
    }

    impl LlmReplayTransport for PendingProductionReplay {
        fn capability(&self) -> &LlmReplayCapability {
            &self.capability
        }

        fn start(&self, _request: LlmRequest) -> FlowResult<LlmReplayCall> {
            self.starts.fetch_add(1, Ordering::AcqRel);
            let cancellations = self.cancellations.clone();
            Ok(LlmReplayCall::new(std::future::pending(), move || {
                cancellations.fetch_add(1, Ordering::AcqRel);
            }))
        }
    }

    struct ReplayFactory {
        replay: Arc<dyn LlmReplayTransport>,
    }

    impl LlmReplayFactory for ReplayFactory {
        fn build(
            &self,
            _context: &LlmExecutionContextSnapshot,
        ) -> FlowResult<Arc<dyn LlmReplayTransport>> {
            Ok(self.replay.clone())
        }
    }

    struct ImmediateBarrier;

    impl SubscriberBarrier for ImmediateBarrier {
        fn flush(&self) -> BarrierFuture {
            Box::pin(async { Ok(()) })
        }
    }

    struct CountingBarrier {
        calls: Arc<AtomicUsize>,
        fail: bool,
    }

    type RuntimeFixture = (
        Arc<RouterRuntime>,
        Arc<InMemoryTrajectorySink>,
        Arc<InMemoryTrajectoryDelivery>,
        Arc<FixedSampler>,
        Arc<Replay>,
    );

    impl SubscriberBarrier for CountingBarrier {
        fn flush(&self) -> BarrierFuture {
            let calls = self.calls.clone();
            let fail = self.fail;
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                if fail { Err(()) } else { Ok(()) }
            })
        }
    }

    struct CountingCoreBarrier {
        calls: Arc<AtomicUsize>,
    }

    impl SubscriberBarrier for CountingCoreBarrier {
        fn flush(&self) -> BarrierFuture {
            self.calls.fetch_add(1, Ordering::SeqCst);
            CoreSubscriberBarrier.flush()
        }
    }

    struct DispatcherGate {
        entered: AtomicBool,
        released: Mutex<bool>,
        released_notify: Condvar,
    }

    impl DispatcherGate {
        fn new() -> Self {
            Self {
                entered: AtomicBool::new(false),
                released: Mutex::new(false),
                released_notify: Condvar::new(),
            }
        }

        fn block(&self) {
            self.entered.store(true, Ordering::Release);
            let mut released = self
                .released
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            while !*released {
                released = self
                    .released_notify
                    .wait(released)
                    .unwrap_or_else(|error| error.into_inner());
            }
        }

        fn release(&self) {
            *self
                .released
                .lock()
                .unwrap_or_else(|error| error.into_inner()) = true;
            self.released_notify.notify_all();
        }
    }

    struct DispatcherGateRelease(Arc<DispatcherGate>);

    impl DispatcherGateRelease {
        fn release(&self) {
            self.0.release();
        }
    }

    impl Drop for DispatcherGateRelease {
        fn drop(&mut self) {
            self.0.release();
        }
    }

    fn config(family: LlmApiFamily, deadline_seconds: u64) -> RouterConfig {
        RouterConfig {
            mode: RouterMode::Shadow,
            project_id: Some("runtime-test".into()),
            pools: vec![PoolConfig {
                id: "pool".into(),
                api_family: family,
                anchor_models: vec!["anchor".into()],
                anchor_revision: "anchor-r1".into(),
                sampling_probability: 1.0,
                max_candidates_per_sample: 1,
                selector: PoolSelectorConfig::default(),
                lookahead: LookaheadConfig {
                    primary_llm_completions: 1,
                    deadline_seconds,
                    lifecycle_presets: Vec::new(),
                    max_events_per_window: 32,
                    max_bytes_per_window: 64 * 1024,
                    unknown_fields: BTreeMap::new(),
                },
                concurrency: ConcurrencyConfig {
                    shadow: 1,
                    judge: 1,
                    max_pending: 4,
                    unknown_fields: BTreeMap::new(),
                },
                candidates: vec![CandidateConfig {
                    id: "candidate".into(),
                    model: "candidate-model".into(),
                    model_revision: "candidate-r1".into(),
                    cost_rank: 0,
                    max_context_tokens: None,
                    capabilities: CandidateCapabilities {
                        tools: true,
                        multimodal_input: true,
                        structured_output: true,
                        reasoning_controls: true,
                        unknown_fields: BTreeMap::new(),
                    },
                    unknown_fields: BTreeMap::new(),
                }],
                canonicalizer: CanonicalizerConfig::default(),
                judge: judge_config(),
                learning: None,
                outcome: BTreeMap::new(),
                unknown_fields: BTreeMap::new(),
            }],
            ..RouterConfig::default()
        }
    }

    fn judge_config() -> JudgeConfig {
        JudgeConfig {
            version: 1,
            model: "judge-model".into(),
            model_revision: "judge-r1".into(),
            prompt_version: JUDGE_PROMPT_VERSION_V1.into(),
            rubric_version: JUDGE_RUBRIC_VERSION_V1.into(),
            output_schema_version: 1,
            temperature: None,
            response_weight: 0.5,
            trajectory_weight: 0.5,
            response_floor: 0.8,
            trajectory_floor: 0.8,
            judge_confidence_floor: 0.7,
            pass_threshold: 0.85,
            max_rationale_bytes: 4_096,
            base_cooloff_seconds: 10,
            max_cooloff_seconds: 300,
            unknown_fields: BTreeMap::new(),
        }
    }

    fn context(family: LlmApiFamily) -> LlmExecutionContextSnapshot {
        let root_uuid = Uuid::now_v7();
        let owner_uuid = Uuid::now_v7();
        LlmExecutionContextSnapshot {
            call_uuid: Uuid::now_v7(),
            root_uuid,
            parent_uuid: owner_uuid,
            trajectory_owner_uuid: owner_uuid,
            trajectory_owner_path: vec![LlmTrajectoryScopeSnapshot {
                uuid: owner_uuid,
                name: "agent".into(),
                scope_type: ScopeType::Agent,
            }],
            api_family: family,
            call_role: LlmCallRole::Primary,
            attributes: LlmAttributes::empty(),
            tenant_id: None,
            agent_id: None,
            sanitized_metadata: BTreeMap::new(),
        }
    }

    fn later_context(anchor: &LlmExecutionContextSnapshot) -> LlmExecutionContextSnapshot {
        LlmExecutionContextSnapshot {
            call_uuid: Uuid::now_v7(),
            ..anchor.clone()
        }
    }

    fn request(family: LlmApiFamily) -> LlmRequest {
        let content = match family {
            LlmApiFamily::OpenAIChatCompletions => json!({
                "model": "anchor",
                "messages": [{"role": "user", "content": "hello"}]
            }),
            LlmApiFamily::OpenAIResponses => json!({
                "model": "anchor",
                "input": "hello"
            }),
            LlmApiFamily::AnthropicMessages => json!({
                "model": "anchor",
                "max_tokens": 32,
                "messages": [{"role": "user", "content": "hello"}]
            }),
        };
        LlmRequest {
            headers: serde_json::Map::new(),
            content,
        }
    }

    fn response(family: LlmApiFamily) -> Json {
        match family {
            LlmApiFamily::OpenAIChatCompletions => json!({
                "id": "chatcmpl-runtime",
                "model": "anchor",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "answer"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
            }),
            LlmApiFamily::OpenAIResponses => json!({
                "id": "resp-runtime",
                "object": "response",
                "status": "completed",
                "model": "anchor",
                "output": [{
                    "type": "message",
                    "id": "message-runtime",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "answer", "annotations": []}]
                }],
                "usage": {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}
            }),
            LlmApiFamily::AnthropicMessages => json!({
                "id": "msg-runtime",
                "type": "message",
                "role": "assistant",
                "model": "anchor",
                "content": [{"type": "text", "text": "answer"}],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 2, "output_tokens": 1}
            }),
        }
    }

    fn llm_end(call: &LlmExecutionContextSnapshot) -> Event {
        let mut profile = CategoryProfile::default();
        profile.set_llm_call_role(LlmCallRole::Primary);
        Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .name("provider.call")
                .uuid(call.call_uuid)
                .parent_uuid(call.parent_uuid)
                .build(),
            ScopeCategory::End,
            Vec::new(),
            EventCategory::llm(),
            Some(profile),
        ))
    }

    fn runtime(family: LlmApiFamily, deadline_seconds: u64) -> RuntimeFixture {
        let sink = Arc::new(InMemoryTrajectorySink::new(32));
        let delivery = Arc::new(InMemoryTrajectoryDelivery::new(32));
        let sampler = Arc::new(FixedSampler::new(0.0));
        let replay = Arc::new(Replay::new(family));
        let runtime = RouterRuntime::start_with_dependencies(
            config(family, deadline_seconds),
            sampler.clone(),
            sink.clone(),
            delivery.clone(),
            Arc::new(ImmediateBarrier),
        )
        .unwrap();
        (runtime, sink, delivery, sampler, replay)
    }

    fn available_sample_intents(runtime: &RouterRuntime) -> usize {
        runtime.state.pool_resources["pool"]
            .sample_intents
            .available_permits()
    }

    fn next_returning(response: Json, calls: Arc<AtomicUsize>) -> LlmExecutionNextFn {
        Arc::new(move |_request| {
            let response = response.clone();
            let calls = calls.clone();
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(response)
            })
        })
    }

    fn next_yielding(response: Json, calls: Arc<AtomicUsize>) -> LlmExecutionNextFn {
        Arc::new(move |_request| {
            let response = response.clone();
            let calls = calls.clone();
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                tokio::task::yield_now().await;
                Ok(response)
            })
        })
    }

    #[tokio::test]
    async fn recommend_preprocessing_failure_calls_next_once_without_shadow_or_replay() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let (runtime, sink, _delivery, sampler, replay) = runtime(family, 30);
        let context = Arc::new(context(family));
        let request = request(family);
        let original_request = request.clone();
        let response = response(family);
        let next_calls = Arc::new(AtomicUsize::new(0));
        let seen_requests = Arc::new(Mutex::new(Vec::new()));
        let next: LlmExecutionNextFn = {
            let next_calls = next_calls.clone();
            let seen_requests = seen_requests.clone();
            let response = response.clone();
            Arc::new(move |request| {
                next_calls.fetch_add(1, Ordering::SeqCst);
                seen_requests
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push(request);
                let response = response.clone();
                Box::pin(async move { Ok(response) })
            })
        };
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();

        assert_eq!(
            runtime
                .execute_recommend(context, request, Some(replay_transport), next)
                .await
                .unwrap(),
            response
        );
        assert_eq!(next_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            seen_requests
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_slice(),
            &[original_request]
        );
        assert_eq!(sampler.calls.load(Ordering::SeqCst), 0);
        assert_eq!(replay.capability_calls.load(Ordering::SeqCst), 0);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        assert_eq!(sink.operation_attempts(SinkOperation::Pending), 0);
        assert_eq!(sink.operation_attempts(SinkOperation::Terminal), 0);
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn recommend_preserves_anchor_result_when_replay_and_next_drop_panic() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let (runtime, _sink, _delivery, _sampler, _replay) = runtime(family, 30);
        let response = response(family);
        let expected = response.clone();
        let replay: Arc<dyn LlmReplayTransport> = Arc::new(PanickingDropReplay {
            capability: LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: family,
                transport_identity: "panic-drop-replay".to_string(),
            },
            _drop_bomb: PanicOnDrop,
        });
        let next: LlmExecutionNextFn = {
            let drop_bomb = PanicOnDrop;
            Arc::new(move |_request| {
                let _ = &drop_bomb;
                let response = response.clone();
                Box::pin(async move { Ok(response) })
            })
        };

        assert_eq!(
            runtime
                .execute_recommend(
                    Arc::new(context(family)),
                    request(family),
                    Some(replay),
                    next,
                )
                .await
                .unwrap(),
            expected
        );
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
    }

    fn sqlite_count(connection: &rusqlite::Connection, table: &str) -> i64 {
        connection
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap()
    }

    async fn wait_for(mut predicate: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while !predicate() {
                tokio::task::yield_now().await;
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("runtime state did not converge");
    }

    async fn yield_until(mut predicate: impl FnMut() -> bool) {
        for _ in 0..1_000 {
            if predicate() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("paused runtime state did not converge");
    }

    async fn yield_until_named(label: &str, mut predicate: impl FnMut() -> bool) {
        for _ in 0..1_000 {
            if predicate() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("paused runtime state did not converge: {label}");
    }

    async fn yield_until_blocking_work(label: &str, mut predicate: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Instant::now() < deadline {
            if predicate() {
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
            tokio::task::yield_now().await;
        }
        panic!("blocking runtime work did not converge: {label}");
    }

    #[tokio::test]
    async fn lifecycle_registration_issues_the_only_provider_activation_token() {
        let (runtime, _, _, _, _) = runtime(LlmApiFamily::OpenAIChatCompletions, 30);
        let gate = runtime.test_provider_admission();
        assert_eq!(
            gate.phase(),
            crate::provider_admission::ProviderAdmissionPhase::Pending
        );

        let first = runtime
            .lifecycle_registration("provider-token-owner".to_string())
            .unwrap();
        let repeated = runtime
            .lifecycle_registration("provider-token-duplicate".to_string())
            .unwrap_err();

        assert!(repeated.to_string().contains("AlreadyIssued"));
        assert_eq!(
            gate.phase(),
            crate::provider_admission::ProviderAdmissionPhase::Pending
        );
        let starts = AtomicUsize::new(0);
        assert_eq!(
            gate.start_owned(|| starts.fetch_add(1, Ordering::AcqRel)),
            Err(crate::provider_admission::ProviderStartRefusal::Pending)
        );

        let mut registrations = vec![first];
        rollback_registrations(&mut registrations);
        assert!(registrations.is_empty());
        assert_eq!(starts.load(Ordering::Acquire), 0);
        assert_eq!(
            gate.phase(),
            crate::provider_admission::ProviderAdmissionPhase::Closed
        );
    }

    #[tokio::test]
    async fn durable_control_service_is_pending_committed_replaced_and_epoch_scoped() {
        let _control_guard = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let first_directory = tempdir().unwrap();
        let second_directory = tempdir().unwrap();
        #[cfg(unix)]
        for directory in [&first_directory, &second_directory] {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }

        let mut first_config = config(LlmApiFamily::OpenAIChatCompletions, 30);
        first_config.project_id = Some("control-runtime-first".into());
        first_config.database_path = first_directory
            .path()
            .join("router.db")
            .to_string_lossy()
            .into_owned();
        let first_activated = LedgerRepository::activate(&first_config).unwrap();
        let first_runtime =
            RouterRuntime::start_with_activated_ledger(first_config, first_activated).unwrap();
        assert_eq!(
            crate::control::router_control_service().unwrap_err(),
            crate::control::RouterControlError::Unavailable
        );

        let first_registration = first_runtime
            .lifecycle_registration("control-runtime-first".into())
            .unwrap();
        assert!(
            first_runtime
                .state
                .control_authority
                .as_ref()
                .unwrap()
                .epoch()
                > 0
        );
        assert_eq!(
            crate::control::router_control_service().unwrap_err(),
            crate::control::RouterControlError::Unavailable
        );
        drop(first_registration);

        let first_service = crate::control::router_control_service().unwrap();
        assert_eq!(
            first_service.snapshot().await.unwrap().control_generation,
            0
        );
        let first_snapshot = first_service
            .apply_mutation(
                crate::control::ControlMutation {
                    mutation_id: Uuid::now_v7(),
                    scope: crate::control::ControlScope::All,
                    operation: crate::control::ControlOperation::SetPaused { value: true },
                    expected_control_generation: 0,
                    actor: "operator-a".into(),
                    reason: "pause first runtime".into(),
                },
                crate::control::ControlMutationOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(first_snapshot.control_generation, 1);
        assert!(first_snapshot.all.paused);

        let mut second_config = config(LlmApiFamily::OpenAIChatCompletions, 30);
        second_config.project_id = Some("control-runtime-second".into());
        second_config.database_path = second_directory
            .path()
            .join("router.db")
            .to_string_lossy()
            .into_owned();
        let second_activated = LedgerRepository::activate(&second_config).unwrap();
        let second_runtime =
            RouterRuntime::start_with_activated_ledger(second_config, second_activated).unwrap();
        let second_registration = second_runtime
            .lifecycle_registration("control-runtime-second".into())
            .unwrap();
        assert_eq!(
            crate::control::router_control_service().unwrap_err(),
            crate::control::RouterControlError::Unavailable
        );
        assert_eq!(
            first_service.snapshot().await.unwrap_err(),
            crate::control::RouterControlError::Unavailable
        );
        drop(second_registration);

        let second_service = crate::control::router_control_service().unwrap();
        assert_eq!(
            second_service.snapshot().await.unwrap().control_generation,
            0
        );
        assert_eq!(
            first_service.snapshot().await.unwrap_err(),
            crate::control::RouterControlError::Unavailable
        );
        first_runtime.stop_intake();
        assert_eq!(
            crate::control::router_control_service()
                .unwrap()
                .snapshot()
                .await
                .unwrap()
                .control_generation,
            0
        );

        second_runtime.stop_intake();
        assert_eq!(
            second_service.snapshot().await.unwrap_err(),
            crate::control::RouterControlError::Unavailable
        );
        assert_eq!(
            crate::control::router_control_service().unwrap_err(),
            crate::control::RouterControlError::Unavailable
        );
        first_runtime.abort();
        second_runtime.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn inspection_operations_attach_the_active_writer_without_starting_a_process() {
        let _control_guard = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let (_temporary, runtime) = activated_lifecycle_runtime("inspection-attached-writer");
        let registration = runtime
            .lifecycle_registration("inspection-attached-writer".into())
            .unwrap();
        drop(registration);
        let path = runtime.state.config.database_path.clone();
        let before = rusqlite::Connection::open(&path)
            .unwrap()
            .query_row("SELECT count(*) FROM process_instances", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap();

        let inspection = crate::inspection::InspectionService::open(
            runtime.state.config.as_ref().clone(),
            crate::inspection::InspectionServiceOptions {
                allow_operations: true,
                ..crate::inspection::InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        assert!(inspection.operations_enabled());
        assert_eq!(
            rusqlite::Connection::open(&path)
                .unwrap()
                .query_row("SELECT count(*) FROM process_instances", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            before
        );

        let snapshot = inspection
            .apply_control(crate::inspection::InspectionControlRequestV1 {
                mutation_id: Uuid::now_v7(),
                scope: crate::control::ControlScope::All,
                operation: crate::control::ControlOperation::SetForceAnchor { value: true },
                expected_control_generation: 0,
                actor: "inspection-operator".into(),
                reason: "attach to active runtime writer".into(),
            })
            .await
            .unwrap();
        assert_eq!(snapshot.control_generation, 1);
        assert!(snapshot.all.force_anchor);
        inspection.close().await.unwrap();

        let runtime_snapshot = crate::control::router_control_service()
            .unwrap()
            .snapshot()
            .await
            .unwrap();
        assert_eq!(runtime_snapshot.control_generation, 1);
        assert!(runtime_snapshot.all.force_anchor);
        runtime.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn control_mutation_timeout_and_cancellation_before_start_never_commit() {
        let _control_guard = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let (_temporary, runtime) = activated_lifecycle_runtime("control-start-fence");
        let registration = runtime
            .lifecycle_registration("control-start-fence".into())
            .unwrap();
        drop(registration);
        let service = crate::control::router_control_service().unwrap();
        let generation_snapshot = service.snapshot().await.unwrap();
        let pool_id = generation_snapshot.pools.keys().next().unwrap().clone();
        let writer = runtime.state.writer_client.clone().unwrap();

        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let pause_writer = writer.clone();
        let pause = tokio::spawn(async move {
            pause_writer
                .pause_until(
                    Instant::now() + Duration::from_secs(2),
                    started_tx,
                    release_rx,
                )
                .await
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let timed_out_id = Uuid::now_v7();
        let timed_out = service
            .apply_mutation(
                crate::control::ControlMutation {
                    mutation_id: timed_out_id,
                    scope: crate::control::ControlScope::All,
                    operation: crate::control::ControlOperation::SetPaused { value: true },
                    expected_control_generation: 0,
                    actor: "operator-a".into(),
                    reason: "queued timeout".into(),
                },
                crate::control::ControlMutationOptions {
                    transaction_start_timeout_ms: 10,
                },
            )
            .await;
        assert_eq!(
            timed_out.unwrap_err(),
            crate::control::RouterControlError::Busy
        );

        let canceled_id = Uuid::now_v7();
        let canceled_service = service.clone();
        let canceled = tokio::spawn(async move {
            canceled_service
                .apply_mutation(
                    crate::control::ControlMutation {
                        mutation_id: canceled_id,
                        scope: crate::control::ControlScope::All,
                        operation: crate::control::ControlOperation::SetForceAnchor { value: true },
                        expected_control_generation: 0,
                        actor: "operator-a".into(),
                        reason: "queued cancellation".into(),
                    },
                    crate::control::ControlMutationOptions::default(),
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        canceled.abort();
        assert!(canceled.await.unwrap_err().is_cancelled());

        let timed_out_reset_id = Uuid::now_v7();
        assert_eq!(
            service
                .reset_learning(
                    crate::inspection::LearningResetRequestV1 {
                        mutation_id: timed_out_reset_id,
                        scope: crate::inspection::LearningResetScopeV1::Pool {
                            pool_id: pool_id.clone(),
                            expected_learning_generation_id: generation_snapshot.pools[&pool_id]
                                .learning_generation_id,
                        },
                        confirm_project_id: runtime.state.identity.project_id.clone(),
                        actor: "operator-a".into(),
                        reason: "queued reset timeout".into(),
                    },
                    crate::control::ControlMutationOptions {
                        transaction_start_timeout_ms: 10,
                    },
                )
                .await
                .unwrap_err(),
            crate::control::RouterControlError::Busy
        );

        let canceled_rotation_id = Uuid::now_v7();
        let canceled_service = service.clone();
        let project_id = runtime.state.identity.project_id.clone();
        let expected_cohort_generation_id = generation_snapshot.cohort_generation_id;
        let canceled_rotation = tokio::spawn(async move {
            canceled_service
                .rotate_cohort(
                    crate::inspection::CohortRotationRequestV1 {
                        mutation_id: canceled_rotation_id,
                        expected_cohort_generation_id,
                        confirm_project_id: project_id,
                        actor: "operator-a".into(),
                        reason: "queued rotation cancellation".into(),
                    },
                    crate::control::ControlMutationOptions::default(),
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(10)).await;
        canceled_rotation.abort();
        assert!(canceled_rotation.await.unwrap_err().is_cancelled());

        release_tx.send(()).unwrap();
        pause.await.unwrap().unwrap();
        writer
            .flush_until(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        let connection = rusqlite::Connection::open(&runtime.state.config.database_path).unwrap();
        for mutation_id in [timed_out_id, canceled_id] {
            assert_eq!(
                connection
                    .query_row(
                        "SELECT count(*) FROM control_mutation_receipts WHERE mutation_id = ?1",
                        [mutation_id.to_string()],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap(),
                0
            );
        }
        for mutation_id in [timed_out_reset_id, canceled_rotation_id] {
            assert_eq!(
                connection
                    .query_row(
                        "SELECT count(*) FROM operator_mutation_receipts
                         WHERE mutation_id = ?1",
                        [mutation_id.to_string()],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap(),
                0
            );
        }
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM controls", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            1
        );
        runtime.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn control_saturation_force_anchors_internally_and_preserves_safety_reserve() {
        let _control_guard = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let (_temporary, runtime) = activated_lifecycle_runtime("control-saturation");
        let registration = runtime
            .lifecycle_registration("control-saturation".into())
            .unwrap();
        drop(registration);
        let service = crate::control::router_control_service().unwrap();
        let connection = rusqlite::Connection::open(&runtime.state.config.database_path).unwrap();
        let genesis_hash = connection
            .query_row(
                "SELECT record_hash FROM controls WHERE control_generation = 0",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let now = Utc::now().timestamp_millis();
        connection
            .execute(
                "WITH RECURSIVE sequence(value) AS (
                    VALUES(1)
                    UNION ALL
                    SELECT value + 1 FROM sequence WHERE value < 98973
                 )
                 INSERT INTO control_mutation_receipts (
                    mutation_id, canonical_payload_hash, history_ordinal,
                    predecessor_chain_hash, chain_tip_hash, project_uuid,
                    result, result_control_generation, control_id, control_record_hash,
                    scope_kind, pool_id, operation_kind, requested_value,
                    expected_control_generation, actor, reason,
                    process_instance_id, created_at_unix_ms
                 )
                 SELECT
                    printf('018f0000-0000-7000-8000-%012x', value),
                    printf('%064x', 200000 + value),
                    value,
                    CASE WHEN value = 1 THEN ?1 ELSE printf('%064x', value - 1) END,
                    printf('%064x', value),
                    ?2, 'no_op', 0, NULL, NULL,
                    'all', NULL, 'set_paused', 0, 0,
                    'capacity-fixture', 'capacity-fixture', ?3, ?4
                 FROM sequence",
                rusqlite::params![
                    genesis_hash,
                    runtime.state.identity.project_uuid.to_string(),
                    runtime.state.identity.process_instance_id.to_string(),
                    now,
                ],
            )
            .unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT (SELECT count(*) FROM controls)
                          + (SELECT count(*) FROM control_mutation_receipts)",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            98_974
        );

        let mutate = |operation| crate::control::ControlMutation {
            mutation_id: Uuid::now_v7(),
            scope: crate::control::ControlScope::All,
            operation,
            expected_control_generation: 0,
            actor: "operator-a".into(),
            reason: "capacity test".into(),
        };
        let snapshot = service
            .apply_mutation(
                mutate(crate::control::ControlOperation::SetPaused { value: false }),
                crate::control::ControlMutationOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(snapshot.control_generation, 0);
        assert_eq!(snapshot.all, crate::control::RouterControlState::default());
        assert_eq!(
            runtime
                .state
                .control_authority
                .as_ref()
                .unwrap()
                .effective_state("pool"),
            crate::control::RouterControlState {
                force_anchor: true,
                paused: false,
            }
        );

        service
            .apply_mutation(
                mutate(crate::control::ControlOperation::SetForceAnchor { value: false }),
                crate::control::ControlMutationOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(
            service
                .apply_mutation(
                    mutate(crate::control::ControlOperation::SetForceAnchor { value: false }),
                    crate::control::ControlMutationOptions::default(),
                )
                .await
                .unwrap_err(),
            crate::control::RouterControlError::CapacityExhausted
        );
        let safety_snapshot = service
            .apply_mutation(
                mutate(crate::control::ControlOperation::SetPaused { value: true }),
                crate::control::ControlMutationOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(safety_snapshot.control_generation, 1);
        assert!(safety_snapshot.all.paused);
        runtime.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn control_monitor_repeatedly_observes_remote_writer_commits() {
        let _control_guard = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let (temporary, first_runtime) = activated_lifecycle_runtime("control-remote-poll");
        let first_registration = first_runtime
            .lifecycle_registration("control-remote-poll-first".into())
            .unwrap();
        drop(first_registration);
        let mut second_config = config(LlmApiFamily::OpenAIChatCompletions, 30);
        second_config.project_id = Some("control-remote-poll".into());
        second_config.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        let second_activated = LedgerRepository::activate(&second_config).unwrap();
        let second_runtime =
            RouterRuntime::start_with_activated_ledger(second_config, second_activated).unwrap();
        let second_writer = second_runtime.state.writer_client.clone().unwrap();

        for (generation, paused) in [(0_u64, true), (1, false), (2, true), (3, false)] {
            let prepared = crate::ledger::repository::control::prepare_control_mutation(
                crate::control::ControlMutation {
                    mutation_id: Uuid::now_v7(),
                    scope: crate::control::ControlScope::All,
                    operation: crate::control::ControlOperation::SetPaused { value: paused },
                    expected_control_generation: generation,
                    actor: "remote-operator".into(),
                    reason: "cross-process poll test".into(),
                },
            )
            .unwrap();
            let fence = Arc::new(crate::control::ControlTransactionFence::new(
                Instant::now() + Duration::from_secs(2),
            ));
            assert!(matches!(
                second_writer
                    .apply_control_mutation(prepared, fence)
                    .await
                    .unwrap()
                    .unwrap(),
                crate::ledger::repository::control::ControlMutationAck::Completed {
                    result: crate::control::ControlMutationResult::Applied,
                    ..
                }
            ));

            tokio::time::timeout(Duration::from_secs(3), async {
                loop {
                    let observed = first_runtime
                        .state
                        .control_authority
                        .as_ref()
                        .unwrap()
                        .effective_state("pool")
                        .paused;
                    if observed == paused {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .expect("remote control commit must reach the first runtime within two cadences");
        }

        first_runtime.abort();
        second_runtime.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn operator_service_resets_rotates_replays_and_returns_fresh_conflicts() {
        let _control_guard = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let (_temporary, runtime) = activated_lifecycle_runtime("operator-service");
        let registration = runtime
            .lifecycle_registration("operator-service".into())
            .unwrap();
        drop(registration);
        let service = crate::control::router_control_service().unwrap();
        let initial = service.snapshot().await.unwrap();
        let pool_id = initial.pools.keys().next().unwrap().clone();
        let initial_learning = initial.pools[&pool_id].learning_generation_id;
        let reset_request = crate::inspection::LearningResetRequestV1 {
            mutation_id: Uuid::now_v7(),
            scope: crate::inspection::LearningResetScopeV1::Pool {
                pool_id: pool_id.clone(),
                expected_learning_generation_id: initial_learning,
            },
            confirm_project_id: runtime.state.identity.project_id.clone(),
            actor: "operator-a".into(),
            reason: "reset one pool".into(),
        };
        let reset = service
            .reset_learning(
                reset_request.clone(),
                crate::control::ControlMutationOptions::default(),
            )
            .await
            .unwrap();
        assert_eq!(
            reset.result,
            crate::inspection::OperatorMutationResultV1::Applied
        );
        let reset_learning = reset.resulting_generations[&pool_id];
        assert_ne!(reset_learning, initial_learning);
        let after_reset = service.snapshot().await.unwrap();
        assert_eq!(
            after_reset.pools[&pool_id].learning_generation_id,
            reset_learning
        );
        assert_eq!(
            after_reset.cohort_generation_id,
            initial.cohort_generation_id
        );
        assert_eq!(after_reset.control_generation, initial.control_generation);
        assert_eq!(
            service
                .reset_learning(
                    reset_request.clone(),
                    crate::control::ControlMutationOptions::default(),
                )
                .await
                .unwrap(),
            reset
        );

        let stale = crate::inspection::LearningResetRequestV1 {
            mutation_id: Uuid::now_v7(),
            reason: "stale reset".into(),
            ..reset_request
        };
        let stale_snapshot = match service
            .reset_learning(stale, crate::control::ControlMutationOptions::default())
            .await
            .unwrap_err()
        {
            crate::control::RouterControlError::Conflict { snapshot } => snapshot,
            error => panic!("unexpected stale reset error: {error:?}"),
        };
        assert_eq!(
            stale_snapshot.pools[&pool_id].learning_generation_id,
            reset_learning
        );

        let rotation_request = crate::inspection::CohortRotationRequestV1 {
            mutation_id: Uuid::now_v7(),
            expected_cohort_generation_id: initial.cohort_generation_id,
            confirm_project_id: runtime.state.identity.project_id.clone(),
            actor: "operator-a".into(),
            reason: "rotate cohort".into(),
        };
        let rotation = service
            .rotate_cohort(
                rotation_request.clone(),
                crate::control::ControlMutationOptions::default(),
            )
            .await
            .unwrap();
        let rotated_cohort = rotation.resulting_generations["cohort"];
        assert_ne!(rotated_cohort, initial.cohort_generation_id);
        let after_rotation = service.snapshot().await.unwrap();
        assert_eq!(after_rotation.cohort_generation_id, rotated_cohort);
        assert_eq!(
            after_rotation.pools[&pool_id].learning_generation_id,
            reset_learning
        );
        assert_eq!(
            service
                .rotate_cohort(
                    rotation_request,
                    crate::control::ControlMutationOptions::default(),
                )
                .await
                .unwrap(),
            rotation
        );
        let current = runtime
            .state
            .control_authority
            .as_ref()
            .unwrap()
            .current_generation_snapshot()
            .unwrap();
        assert_eq!(
            current.cohort_assignment.cohort_generation_id(),
            rotated_cohort
        );
        assert_eq!(
            sqlite_count(
                &rusqlite::Connection::open(&runtime.state.config.database_path).unwrap(),
                "operator_mutation_receipts",
            ),
            3
        );
        runtime.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shadow_request_after_reset_uses_the_refreshed_learning_generation() {
        let _control_guard = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let (_temporary, runtime) = activated_lifecycle_runtime("shadow-generation-refresh");
        let registration = runtime
            .lifecycle_registration("shadow-generation-refresh".into())
            .unwrap();
        drop(registration);
        let service = crate::control::router_control_service().unwrap();
        let initial = service.snapshot().await.unwrap();
        let pool_id = initial.pools.keys().next().unwrap().clone();
        let reset = service
            .reset_learning(
                crate::inspection::LearningResetRequestV1 {
                    mutation_id: Uuid::now_v7(),
                    scope: crate::inspection::LearningResetScopeV1::Pool {
                        pool_id: pool_id.clone(),
                        expected_learning_generation_id: initial.pools[&pool_id]
                            .learning_generation_id,
                    },
                    confirm_project_id: runtime.state.identity.project_id.clone(),
                    actor: "operator-a".into(),
                    reason: "refresh shadow generation".into(),
                },
                crate::control::ControlMutationOptions::default(),
            )
            .await
            .unwrap();
        let refreshed_generation = reset.resulting_generations[&pool_id];
        let replay: Arc<dyn LlmReplayTransport> = PendingProductionReplay::new();
        let downstream_calls = Arc::new(AtomicUsize::new(0));
        runtime
            .execute(
                context(LlmApiFamily::OpenAIChatCompletions),
                request(LlmApiFamily::OpenAIChatCompletions),
                Some(replay),
                next_returning(
                    response(LlmApiFamily::OpenAIChatCompletions),
                    downstream_calls.clone(),
                ),
            )
            .await
            .unwrap();
        assert_eq!(downstream_calls.load(Ordering::Acquire), 1);
        let connection = rusqlite::Connection::open(&runtime.state.config.database_path).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if connection
                    .query_row("SELECT count(*) FROM anchors", [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap()
                    == 1
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("post-reset Shadow anchor must persist");
        assert_eq!(
            connection
                .query_row("SELECT learning_generation_id FROM anchors", [], |row| {
                    row.get::<_, String>(0)
                })
                .unwrap(),
            refreshed_generation.to_string()
        );
        runtime.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn generation_monitor_observes_remote_reset_and_rotation_within_one_second() {
        let _control_guard = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let (temporary, first_runtime) = activated_lifecycle_runtime("generation-remote-poll");
        let first_registration = first_runtime
            .lifecycle_registration("generation-remote-poll-first".into())
            .unwrap();
        drop(first_registration);
        let initial = first_runtime
            .state
            .control_authority
            .as_ref()
            .unwrap()
            .current_generation_snapshot()
            .unwrap();
        let pool_id = initial.control.pools.keys().next().unwrap().clone();

        let mut second_config = config(LlmApiFamily::OpenAIChatCompletions, 30);
        second_config.project_id = Some("generation-remote-poll".into());
        second_config.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        let second_activated = LedgerRepository::activate(&second_config).unwrap();
        let second_runtime =
            RouterRuntime::start_with_activated_ledger(second_config, second_activated).unwrap();
        let second_writer = second_runtime.state.writer_client.clone().unwrap();

        let reset = crate::ledger::repository::inspection::operator::prepare_learning_reset(
            crate::inspection::LearningResetRequestV1 {
                mutation_id: Uuid::now_v7(),
                scope: crate::inspection::LearningResetScopeV1::Pool {
                    pool_id: pool_id.clone(),
                    expected_learning_generation_id: initial.control.pools[&pool_id]
                        .learning_generation_id,
                },
                confirm_project_id: "generation-remote-poll".into(),
                actor: "remote-operator".into(),
                reason: "remote reset".into(),
            },
        )
        .unwrap();
        let fence = Arc::new(crate::control::ControlTransactionFence::new(
            Instant::now() + Duration::from_secs(2),
        ));
        let reset_ack = second_writer
            .apply_operator_mutation(reset, fence)
            .await
            .unwrap()
            .unwrap();
        let reset_learning = match reset_ack {
            crate::ledger::repository::inspection::operator::OperatorMutationTransactionAck::Completed(
                acknowledgement,
            ) => acknowledgement.receipt.resulting_generations[&pool_id],
            crate::ledger::repository::inspection::operator::OperatorMutationTransactionAck::TransactionNotStarted => {
                panic!("remote reset transaction did not start")
            }
        };
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if first_runtime
                    .state
                    .control_authority
                    .as_ref()
                    .and_then(ControlRuntimeAuthority::current_generation_snapshot)
                    .is_some_and(|generation| {
                        generation.control.pools[&pool_id].learning_generation_id == reset_learning
                    })
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("remote learning reset must be observed within one poll cadence");

        let rotation = crate::ledger::repository::inspection::operator::prepare_cohort_rotation(
            crate::inspection::CohortRotationRequestV1 {
                mutation_id: Uuid::now_v7(),
                expected_cohort_generation_id: initial.control.cohort_generation_id,
                confirm_project_id: "generation-remote-poll".into(),
                actor: "remote-operator".into(),
                reason: "remote rotation".into(),
            },
        )
        .unwrap();
        let fence = Arc::new(crate::control::ControlTransactionFence::new(
            Instant::now() + Duration::from_secs(2),
        ));
        let rotation_ack = second_writer
            .apply_operator_mutation(rotation, fence)
            .await
            .unwrap()
            .unwrap();
        let rotated_cohort = match rotation_ack {
            crate::ledger::repository::inspection::operator::OperatorMutationTransactionAck::Completed(
                acknowledgement,
            ) => acknowledgement.receipt.resulting_generations["cohort"],
            crate::ledger::repository::inspection::operator::OperatorMutationTransactionAck::TransactionNotStarted => {
                panic!("remote rotation transaction did not start")
            }
        };
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                if first_runtime
                    .state
                    .control_authority
                    .as_ref()
                    .and_then(ControlRuntimeAuthority::current_generation_snapshot)
                    .is_some_and(|generation| {
                        generation.control.cohort_generation_id == rotated_cohort
                            && generation.cohort_assignment.cohort_generation_id() == rotated_cohort
                            && generation.control.pools[&pool_id].learning_generation_id
                                == reset_learning
                    })
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("remote cohort rotation must be observed within one poll cadence");

        first_runtime.abort();
        second_runtime.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn unreadable_control_authority_stops_background_work_and_force_anchors() {
        let _control_guard = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let (_temporary, runtime) = activated_lifecycle_runtime("control-unreadable");
        let registration = runtime
            .lifecycle_registration("control-unreadable".into())
            .unwrap();
        drop(registration);
        let service = crate::control::router_control_service().unwrap();
        assert_eq!(
            runtime
                .state
                .control_authority
                .as_ref()
                .unwrap()
                .effective_state("pool"),
            crate::control::RouterControlState::default()
        );
        runtime.state.read_pool.as_ref().unwrap().abort();
        tokio::time::timeout(Duration::from_secs(3), async {
            loop {
                if runtime
                    .state
                    .control_authority
                    .as_ref()
                    .unwrap()
                    .effective_state("pool")
                    == (crate::control::RouterControlState {
                        force_anchor: true,
                        paused: true,
                    })
                {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .expect("an unreadable authority must fail closed within two cadences");
        assert_eq!(
            service.snapshot().await.unwrap_err(),
            crate::control::RouterControlError::Unavailable
        );
        runtime.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn remote_config_replacement_makes_old_control_service_unavailable() {
        let _control_guard = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let (temporary, runtime) = activated_lifecycle_runtime("control-config-replacement");
        let registration = runtime
            .lifecycle_registration("control-config-replacement".into())
            .unwrap();
        drop(registration);
        let service = crate::control::router_control_service().unwrap();
        assert_eq!(service.snapshot().await.unwrap().control_generation, 0);

        let mut replacement = config(LlmApiFamily::OpenAIChatCompletions, 30);
        replacement.project_id = Some("control-config-replacement".into());
        replacement.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        replacement.retention_days += 1;
        let replacement_ledger = LedgerRepository::activate(&replacement).unwrap();
        assert_eq!(
            service.snapshot().await.unwrap_err(),
            crate::control::RouterControlError::Unavailable
        );
        assert_eq!(
            service
                .apply_mutation(
                    crate::control::ControlMutation {
                        mutation_id: Uuid::now_v7(),
                        scope: crate::control::ControlScope::All,
                        operation: crate::control::ControlOperation::SetPaused { value: true },
                        expected_control_generation: 0,
                        actor: "operator-a".into(),
                        reason: "stale runtime".into(),
                    },
                    crate::control::ControlMutationOptions::default(),
                )
                .await
                .unwrap_err(),
            crate::control::RouterControlError::Unavailable
        );
        drop(replacement_ledger);
        runtime.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pause_blocks_new_shadow_intents_and_recommend_preprocessing() {
        let _control_guard = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let family = LlmApiFamily::OpenAIChatCompletions;
        let mut shadow_config = config(family, 30);
        shadow_config.project_id = Some("control-shadow-pause".into());
        shadow_config.database_path = temporary
            .path()
            .join("router.db")
            .to_string_lossy()
            .into_owned();
        let activated = LedgerRepository::activate(&shadow_config).unwrap();
        let sampler = Arc::new(FixedSampler::new(0.0));
        let shadow_runtime = RouterRuntime::start_with_dependencies_and_activation(
            shadow_config,
            sampler,
            None,
            Arc::new(ImmediateBarrier),
            None,
            Some(activated),
        )
        .unwrap();
        let registration = shadow_runtime
            .lifecycle_registration("control-shadow-pause".into())
            .unwrap();
        drop(registration);
        let service = crate::control::router_control_service().unwrap();
        service
            .apply_mutation(
                crate::control::ControlMutation {
                    mutation_id: Uuid::now_v7(),
                    scope: crate::control::ControlScope::All,
                    operation: crate::control::ControlOperation::SetPaused { value: true },
                    expected_control_generation: 0,
                    actor: "operator-a".into(),
                    reason: "pause shadow".into(),
                },
                crate::control::ControlMutationOptions::default(),
            )
            .await
            .unwrap();
        let replay: Arc<dyn LlmReplayTransport> = Arc::new(Replay::new(family));
        assert!(
            shadow_runtime
                .prepare_sample_intent_fail_open(&context(family), &request(family), Some(&replay),)
                .is_none()
        );
        service
            .apply_mutation(
                crate::control::ControlMutation {
                    mutation_id: Uuid::now_v7(),
                    scope: crate::control::ControlScope::All,
                    operation: crate::control::ControlOperation::SetPaused { value: false },
                    expected_control_generation: 1,
                    actor: "operator-a".into(),
                    reason: "resume shadow".into(),
                },
                crate::control::ControlMutationOptions::default(),
            )
            .await
            .unwrap();
        assert!(
            shadow_runtime
                .prepare_sample_intent_fail_open(&context(family), &request(family), Some(&replay),)
                .is_some()
        );
        shadow_runtime.abort();

        let (_recommend_temporary, recommend_runtime) =
            activated_recommend_lifecycle_runtime("control-recommend-pause");
        let registration = recommend_runtime
            .lifecycle_registration("control-recommend-pause".into())
            .unwrap();
        drop(registration);
        let service = crate::control::router_control_service().unwrap();
        service
            .apply_mutation(
                crate::control::ControlMutation {
                    mutation_id: Uuid::now_v7(),
                    scope: crate::control::ControlScope::All,
                    operation: crate::control::ControlOperation::SetPaused { value: true },
                    expected_control_generation: 0,
                    actor: "operator-a".into(),
                    reason: "pause recommend".into(),
                },
                crate::control::ControlMutationOptions::default(),
            )
            .await
            .unwrap();
        let replay = Arc::new(Replay::new(family));
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        recommend_runtime
            .preprocess_recommendation(&context(family), &request(family), Some(replay_transport))
            .await
            .unwrap();
        assert_eq!(replay.capability_calls.load(Ordering::Acquire), 0);
        assert_eq!(replay.starts.load(Ordering::Acquire), 0);
        recommend_runtime.abort();
    }

    #[tokio::test]
    async fn runtime_stop_and_abort_close_provider_admission_before_later_starts() {
        for abort in [false, true] {
            let (runtime, _, _, _, _) = runtime(LlmApiFamily::OpenAIChatCompletions, 30);
            let gate = runtime.test_provider_admission();
            let registration = runtime
                .lifecycle_registration(format!("provider-close-{abort}"))
                .unwrap();
            drop(registration);
            assert_eq!(
                gate.phase(),
                crate::provider_admission::ProviderAdmissionPhase::Open
            );

            if abort {
                runtime.abort();
            } else {
                runtime.stop_intake();
            }

            let starts = AtomicUsize::new(0);
            assert_eq!(
                gate.start_owned(|| starts.fetch_add(1, Ordering::AcqRel)),
                Err(crate::provider_admission::ProviderStartRefusal::Closed)
            );
            assert_eq!(starts.load(Ordering::Acquire), 0);
            assert_eq!(
                gate.phase(),
                crate::provider_admission::ProviderAdmissionPhase::Closed
            );
            runtime.abort();
        }
    }

    #[tokio::test]
    async fn all_families_preserve_anchor_and_deliver_after_later_primary() {
        for family in [
            LlmApiFamily::OpenAIChatCompletions,
            LlmApiFamily::OpenAIResponses,
            LlmApiFamily::AnthropicMessages,
        ] {
            let (runtime, sink, delivery, sampler, replay) = runtime(family, 30);
            let anchor = context(family);
            let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
            let calls = Arc::new(AtomicUsize::new(0));
            let raw_response = response(family);
            let result = runtime
                .execute(
                    anchor.clone(),
                    request(family),
                    Some(replay_transport.clone()),
                    next_returning(raw_response.clone(), calls.clone()),
                )
                .await
                .unwrap();
            assert_eq!(result, raw_response);

            runtime.observe_event_fail_open(&llm_end(&anchor));
            let later = later_context(&anchor);
            assert!(runtime.register_primary_call(&later).is_some());
            runtime.observe_event_fail_open(&llm_end(&later));
            wait_for(|| delivery.delivered_anchor_ids().len() == 1).await;

            assert_eq!(calls.load(Ordering::SeqCst), 1);
            assert_eq!(sampler.calls.load(Ordering::SeqCst), 0);
            assert_eq!(replay.capability_calls.load(Ordering::SeqCst), 1);
            assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
            assert!(sink.unresolved_pending().is_empty());
            let terminal = sink.terminal_payloads();
            assert_eq!(terminal.len(), 1);
            assert_eq!(
                terminal[0].state,
                TrajectoryTerminalStateV1::Closed {
                    trigger: TrajectoryTrigger::ProgressReached
                }
            );
            runtime
                .drain(Instant::now() + Duration::from_secs(2))
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn sampled_intents_are_bounded_and_release_on_cancel_and_provider_error() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let mut runtime_config = config(family, 30);
        runtime_config.pools[0].concurrency.max_pending = 1;
        let sink = Arc::new(InMemoryTrajectorySink::new(32));
        let delivery = Arc::new(InMemoryTrajectoryDelivery::new(32));
        let runtime = RouterRuntime::start_with_dependencies(
            runtime_config,
            Arc::new(FixedSampler::new(0.0)),
            sink.clone(),
            delivery.clone(),
            Arc::new(ImmediateBarrier),
        )
        .unwrap();
        assert_eq!(available_sample_intents(&runtime), 1);

        let first_replay = Arc::new(Replay::new(family));
        let first_transport: Arc<dyn LlmReplayTransport> = first_replay.clone();
        let (first_entered_tx, first_entered_rx) = oneshot::channel();
        let first_entered = Arc::new(Mutex::new(Some(first_entered_tx)));
        let first_next: LlmExecutionNextFn = Arc::new({
            let first_entered = first_entered.clone();
            move |_| {
                let entered = first_entered
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take()
                    .expect("first provider executes once");
                Box::pin(async move {
                    let _ = entered.send(());
                    std::future::pending().await
                })
            }
        });
        let first_context = Arc::new(context(family));
        let first_context_probe = Arc::downgrade(&first_context);
        let first = tokio::spawn({
            let runtime = runtime.clone();
            async move {
                runtime
                    .execute(
                        first_context,
                        request(family),
                        Some(first_transport),
                        first_next,
                    )
                    .await
            }
        });
        first_entered_rx.await.unwrap();
        assert!(first_context_probe.upgrade().is_none());
        assert_eq!(available_sample_intents(&runtime), 0);

        let saturated_replay = Arc::new(Replay::new(family));
        let saturated_probe = Arc::downgrade(&saturated_replay);
        let saturated_transport: Arc<dyn LlmReplayTransport> = saturated_replay.clone();
        let (second_entered_tx, second_entered_rx) = oneshot::channel();
        let (second_release_tx, second_release_rx) = oneshot::channel();
        let second_entered = Arc::new(Mutex::new(Some(second_entered_tx)));
        let second_release = Arc::new(Mutex::new(Some(second_release_rx)));
        let raw_response = response(family);
        let second_next: LlmExecutionNextFn = Arc::new({
            let second_entered = second_entered.clone();
            let second_release = second_release.clone();
            let raw_response = raw_response.clone();
            move |_| {
                let entered = second_entered
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take()
                    .expect("second provider executes once");
                let release = second_release
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take()
                    .expect("second provider release exists");
                let raw_response = raw_response.clone();
                Box::pin(async move {
                    let _ = entered.send(());
                    let _ = release.await;
                    Ok(raw_response)
                })
            }
        });
        let second = tokio::spawn({
            let runtime = runtime.clone();
            async move {
                runtime
                    .execute(
                        context(family),
                        request(family),
                        Some(saturated_transport),
                        second_next,
                    )
                    .await
            }
        });
        second_entered_rx.await.unwrap();
        drop(saturated_replay);
        assert!(saturated_probe.upgrade().is_none());
        assert_eq!(available_sample_intents(&runtime), 0);
        let _ = second_release_tx.send(());
        assert_eq!(second.await.unwrap().unwrap(), raw_response);
        assert!(sink.unresolved_pending().is_empty());

        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());
        wait_for(|| available_sample_intents(&runtime) == 1).await;

        let error_transport: Arc<dyn LlmReplayTransport> = Arc::new(Replay::new(family));
        let provider_error: LlmExecutionNextFn =
            Arc::new(|_| Box::pin(async { Err(FlowError::Internal("provider sentinel".into())) }));
        let error = runtime
            .execute(
                context(family),
                request(family),
                Some(error_transport),
                provider_error,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("provider sentinel"));
        assert_eq!(available_sample_intents(&runtime), 1);
        assert_eq!(first_replay.starts.load(Ordering::SeqCst), 0);
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        assert!(delivery.delivered_anchor_ids().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn sampled_intent_permit_stays_with_queued_proposal_until_actor_handles_it() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let mut runtime_config = config(family, 30);
        runtime_config.pools[0].concurrency.max_pending = 1;
        let sink = Arc::new(InMemoryTrajectorySink::new(32));
        let delivery = Arc::new(InMemoryTrajectoryDelivery::new(32));
        let runtime = RouterRuntime::start_with_dependencies(
            runtime_config,
            Arc::new(FixedSampler::new(0.0)),
            sink.clone(),
            delivery,
            Arc::new(ImmediateBarrier),
        )
        .unwrap();
        let replay_transport: Arc<dyn LlmReplayTransport> = Arc::new(Replay::new(family));
        let intent = runtime
            .prepare_sample_intent_fail_open(
                &context(family),
                &request(family),
                Some(&replay_transport),
            )
            .expect("eligible sample acquires the only proposal permit");
        assert_eq!(available_sample_intents(&runtime), 0);

        runtime.propose_anchor_fail_open(intent, &response(family));
        assert_eq!(available_sample_intents(&runtime), 0);
        let bypass_response = response(family);
        let bypass_calls = Arc::new(AtomicUsize::new(0));
        assert_eq!(
            runtime
                .execute(
                    context(family),
                    request(family),
                    Some(replay_transport.clone()),
                    next_returning(bypass_response.clone(), bypass_calls.clone()),
                )
                .await
                .unwrap(),
            bypass_response
        );
        assert_eq!(bypass_calls.load(Ordering::SeqCst), 1);
        assert_eq!(available_sample_intents(&runtime), 0);
        assert!(sink.unresolved_pending().is_empty());

        yield_until_named("queued proposal permit release", || {
            available_sample_intents(&runtime) == 1
        })
        .await;
        yield_until_named("pending acknowledgement", || {
            sink.unresolved_pending().len() == 1
        })
        .await;
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn provider_error_and_bad_response_create_no_anchor() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let (runtime, sink, delivery, sampler, replay) = runtime(family, 30);
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let provider_error: LlmExecutionNextFn =
            Arc::new(|_| Box::pin(async { Err(FlowError::Internal("provider sentinel".into())) }));
        let error = runtime
            .execute(
                context(family),
                request(family),
                Some(replay_transport.clone()),
                provider_error,
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("provider sentinel"));
        assert_eq!(available_sample_intents(&runtime), 4);

        let malformed = json!({"choices": "invalid", "exact": "raw-response"});
        let result = runtime
            .execute(
                context(family),
                request(family),
                Some(replay_transport.clone()),
                next_returning(malformed.clone(), Arc::new(AtomicUsize::new(0))),
            )
            .await
            .unwrap();
        assert_eq!(result, malformed);
        assert_eq!(available_sample_intents(&runtime), 4);
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        assert!(sink.unresolved_pending().is_empty());
        assert!(sink.terminal_payloads().is_empty());
        assert!(delivery.delivered_anchor_ids().is_empty());
        assert_eq!(sampler.calls.load(Ordering::SeqCst), 0);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn oversized_primary_owner_path_fails_open_and_invalidates_spanning_windows() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let (runtime, sink, delivery, _, replay) = runtime(family, 30);
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let anchor = context(family);
        runtime
            .execute(
                anchor.clone(),
                request(family),
                Some(replay_transport.clone()),
                next_returning(response(family), Arc::new(AtomicUsize::new(0))),
            )
            .await
            .unwrap();
        wait_for(|| sink.unresolved_pending().len() == 1).await;
        runtime.observe_event_fail_open(&llm_end(&anchor));

        let mut oversized = later_context(&anchor);
        oversized.trajectory_owner_path = (0..65)
            .map(|_| LlmTrajectoryScopeSnapshot {
                uuid: oversized.trajectory_owner_uuid,
                name: "oversized-owner-path".into(),
                scope_type: ScopeType::Agent,
            })
            .collect();
        let raw_response = response(family);
        assert_eq!(
            runtime
                .execute(
                    oversized.clone(),
                    request(family),
                    Some(replay_transport),
                    next_returning(raw_response.clone(), Arc::new(AtomicUsize::new(0))),
                )
                .await
                .unwrap(),
            raw_response
        );
        assert_eq!(runtime.state.loss.snapshot().classification_loss_epoch, 1);
        assert!(!runtime.state.sampling_enabled.load(Ordering::Acquire));
        assert_eq!(available_sample_intents(&runtime), 4);
        assert_eq!(replay.capability_calls.load(Ordering::SeqCst), 1);

        wait_for(|| sink.terminal_payloads().len() == 1).await;
        assert_eq!(
            sink.terminal_payloads()[0].state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::EventLoss
            }
        );
        runtime.observe_event_fail_open(&llm_end(&oversized));
        assert_eq!(sink.terminal_payloads().len(), 1);
        assert!(delivery.delivered_anchor_ids().is_empty());
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn unsampled_call_returns_exact_response_without_accepting_an_anchor() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let sink = Arc::new(InMemoryTrajectorySink::new(32));
        let delivery = Arc::new(InMemoryTrajectoryDelivery::new(32));
        let sampler = Arc::new(FixedSampler::new(0.75));
        let replay = Arc::new(Replay::new(family));
        let mut runtime_config = config(family, 30);
        runtime_config.pools[0].sampling_probability = 0.5;
        let runtime = RouterRuntime::start_with_dependencies(
            runtime_config,
            sampler.clone(),
            sink.clone(),
            delivery.clone(),
            Arc::new(ImmediateBarrier),
        )
        .unwrap();
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let raw_response = response(family);

        let result = runtime
            .execute(
                context(family),
                request(family),
                Some(replay_transport.clone()),
                next_returning(raw_response.clone(), provider_calls.clone()),
            )
            .await
            .unwrap();

        assert_eq!(result, raw_response);
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
        assert_eq!(sampler.calls.load(Ordering::SeqCst), 1);
        assert_eq!(replay.capability_calls.load(Ordering::SeqCst), 1);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        assert!(sink.unresolved_pending().is_empty());
        assert!(sink.terminal_payloads().is_empty());
        assert!(delivery.delivered_anchor_ids().is_empty());
    }

    #[tokio::test]
    async fn eligible_shaped_call_without_replay_returns_exact_response_without_anchor() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let (runtime, sink, delivery, sampler, replay) = runtime(family, 30);
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let raw_response = response(family);

        let result = runtime
            .execute(
                context(family),
                request(family),
                None,
                next_returning(raw_response.clone(), provider_calls.clone()),
            )
            .await
            .unwrap();

        assert_eq!(result, raw_response);
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
        assert_eq!(sampler.calls.load(Ordering::SeqCst), 0);
        assert_eq!(replay.capability_calls.load(Ordering::SeqCst), 0);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        assert!(sink.unresolved_pending().is_empty());
        assert!(sink.terminal_payloads().is_empty());
        assert!(delivery.delivered_anchor_ids().is_empty());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn register_anchor_queue_full_is_fail_open_and_accepts_no_anchor() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let runtime_config = config(family, 30);
        let total_max_pending = runtime_config.pools[0].concurrency.max_pending;
        let command_capacity =
            CoordinatorLimits::from_total_max_pending(total_max_pending).command_capacity;
        let sink = Arc::new(InMemoryTrajectorySink::new(32));
        let delivery = Arc::new(InMemoryTrajectoryDelivery::new(32));
        let sampler = Arc::new(FixedSampler::new(0.0));
        let replay = Arc::new(Replay::new(family));
        let runtime = RouterRuntime::start_with_dependencies(
            runtime_config,
            sampler.clone(),
            sink.clone(),
            delivery.clone(),
            Arc::new(ImmediateBarrier),
        )
        .unwrap();
        assert_eq!(runtime.state.coordinator_tx.capacity(), command_capacity);
        for _ in 0..command_capacity - 1 {
            assert!(
                runtime
                    .state
                    .coordinator_tx
                    .try_send(CoordinatorCommand::Health("router.test.queue_fill"))
                    .is_ok()
            );
        }

        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let raw_response = response(family);
        let result = runtime
            .execute(
                context(family),
                request(family),
                Some(replay_transport.clone()),
                next_returning(raw_response.clone(), provider_calls.clone()),
            )
            .await
            .unwrap();

        assert_eq!(result, raw_response);
        assert_eq!(runtime.state.coordinator_tx.capacity(), 0);
        assert_eq!(available_sample_intents(&runtime), total_max_pending);
        assert!(runtime.state.health.snapshot().rejected_records > 0);
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(provider_calls.load(Ordering::SeqCst), 1);
        assert_eq!(sampler.calls.load(Ordering::SeqCst), 0);
        assert_eq!(replay.capability_calls.load(Ordering::SeqCst), 1);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        assert!(sink.unresolved_pending().is_empty());
        assert!(sink.terminal_payloads().is_empty());
        assert!(delivery.delivered_anchor_ids().is_empty());
    }

    #[tokio::test]
    async fn stop_after_next_starts_drops_pre_stop_intent() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let (runtime, sink, delivery, _, replay) = runtime(family, 30);
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let entered = Arc::new(Mutex::new(Some(entered_tx)));
        let release = Arc::new(Mutex::new(Some(release_rx)));
        let raw_response = response(family);
        let next: LlmExecutionNextFn = Arc::new({
            let entered = entered.clone();
            let release = release.clone();
            let raw_response = raw_response.clone();
            move |_| {
                let entered = entered
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take()
                    .unwrap();
                let release = release
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take()
                    .unwrap();
                let raw_response = raw_response.clone();
                Box::pin(async move {
                    let _ = entered.send(());
                    let _ = release.await;
                    Ok(raw_response)
                })
            }
        });
        let callback = runtime.execution_callback();
        let task = tokio::spawn({
            let anchor = Arc::new(context(family));
            async move {
                callback(
                    "provider",
                    anchor,
                    request(family),
                    Some(replay_transport),
                    next,
                )
                .await
            }
        });
        entered_rx.await.unwrap();
        runtime.stop_intake();
        release_tx.send(()).unwrap();
        assert_eq!(task.await.unwrap().unwrap(), raw_response);
        assert_eq!(available_sample_intents(&runtime), 4);
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        assert!(sink.unresolved_pending().is_empty());
        assert!(sink.terminal_payloads().is_empty());
        assert!(delivery.delivered_anchor_ids().is_empty());
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn permanent_fault_after_next_starts_drops_pre_fault_intent() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let (runtime, sink, delivery, _, replay) = runtime(family, 30);
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let (entered_tx, entered_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let entered = Arc::new(Mutex::new(Some(entered_tx)));
        let release = Arc::new(Mutex::new(Some(release_rx)));
        let raw_response = response(family);
        let next: LlmExecutionNextFn = Arc::new({
            let entered = entered.clone();
            let release = release.clone();
            let raw_response = raw_response.clone();
            move |_| {
                let entered = entered
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take()
                    .unwrap();
                let release = release
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .take()
                    .unwrap();
                let raw_response = raw_response.clone();
                Box::pin(async move {
                    let _ = entered.send(());
                    let _ = release.await;
                    Ok(raw_response)
                })
            }
        });
        let callback = runtime.execution_callback();
        let task = tokio::spawn({
            let anchor = Arc::new(context(family));
            async move {
                callback(
                    "provider",
                    anchor,
                    request(family),
                    Some(replay_transport),
                    next,
                )
                .await
            }
        });
        entered_rx.await.unwrap();
        record_permanent_runtime_fault(&runtime.state, "router.test.permanent_fault");
        release_tx.send(()).unwrap();

        assert_eq!(task.await.unwrap().unwrap(), raw_response);
        assert!(
            runtime
                .state
                .permanent_runtime_fault
                .load(Ordering::Acquire)
        );
        assert!(!runtime.state.sampling_enabled.load(Ordering::Acquire));
        assert_eq!(available_sample_intents(&runtime), 4);
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        assert!(sink.unresolved_pending().is_empty());
        assert!(sink.terminal_payloads().is_empty());
        assert!(delivery.delivered_anchor_ids().is_empty());
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn background_failure_notifier_records_the_stable_runtime_fault() {
        const FAILURE: BackgroundFailure =
            BackgroundFailure::new("router.test.background_runtime_failure");
        let (runtime, _, _, _, _) = runtime(LlmApiFamily::OpenAIChatCompletions, 30);

        background_failure_notifier(runtime.state.clone())(FAILURE);

        wait_for(|| runtime.state.health.snapshot().last_reason == Some(FAILURE.code())).await;
        assert!(
            runtime
                .state
                .permanent_runtime_fault
                .load(Ordering::Acquire)
        );
        assert!(!runtime.state.sampling_enabled.load(Ordering::Acquire));
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn graceful_drain_finishes_delayed_pending_and_terminal_acknowledgements() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let sink = Arc::new(InMemoryTrajectorySink::new(32));
        sink.set_delay(SinkOperation::Pending, Duration::from_millis(50));
        sink.set_delay(SinkOperation::Terminal, Duration::from_millis(50));
        let delivery = Arc::new(InMemoryTrajectoryDelivery::new(32));
        let sampler = Arc::new(FixedSampler::new(0.0));
        let replay = Arc::new(Replay::new(family));
        let runtime = RouterRuntime::start_with_dependencies(
            config(family, 30),
            sampler,
            sink.clone(),
            delivery.clone(),
            Arc::new(ImmediateBarrier),
        )
        .unwrap();
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        runtime
            .execute(
                context(family),
                request(family),
                Some(replay_transport.clone()),
                next_returning(response(family), Arc::new(AtomicUsize::new(0))),
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(10)).await;
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();

        assert!(sink.unresolved_pending().is_empty());
        let terminals = sink.terminal_payloads();
        assert_eq!(terminals.len(), 1);
        assert!(matches!(
            terminals[0].state,
            TrajectoryTerminalStateV1::Rejected { .. }
        ));
        assert!(delivery.delivered_anchor_ids().is_empty());
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(start_paused = true)]
    async fn deadline_barrier_success_and_failure_freeze_distinct_rejections() {
        for (fail, expected_reason) in [
            (false, TrajectoryRejectionReason::CanceledBeforeAnchorEnd),
            (true, TrajectoryRejectionReason::RejectedDeliveryBarrier),
        ] {
            let family = LlmApiFamily::OpenAIChatCompletions;
            let sink = Arc::new(InMemoryTrajectorySink::new(32));
            let delivery = Arc::new(InMemoryTrajectoryDelivery::new(32));
            let sampler = Arc::new(FixedSampler::new(0.0));
            let replay = Arc::new(Replay::new(family));
            let barrier_calls = Arc::new(AtomicUsize::new(0));
            let runtime = RouterRuntime::start_with_dependencies(
                config(family, 1),
                sampler,
                sink.clone(),
                delivery.clone(),
                Arc::new(CountingBarrier {
                    calls: barrier_calls.clone(),
                    fail,
                }),
            )
            .unwrap();
            let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
            runtime
                .execute(
                    context(family),
                    request(family),
                    Some(replay_transport.clone()),
                    next_returning(response(family), Arc::new(AtomicUsize::new(0))),
                )
                .await
                .unwrap();
            yield_until(|| sink.unresolved_pending().len() == 1).await;
            tokio::time::advance(Duration::from_secs(1)).await;
            yield_until(|| sink.terminal_payloads().len() == 1).await;

            assert_eq!(barrier_calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                sink.terminal_payloads()[0].state,
                TrajectoryTerminalStateV1::Rejected {
                    reason: expected_reason
                }
            );
            assert!(delivery.delivered_anchor_ids().is_empty());
            assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
            runtime
                .drain(Instant::now() + Duration::from_secs(2))
                .await
                .unwrap();
        }
    }

    #[tokio::test(start_paused = true)]
    async fn completed_window_cancels_fake_deadline_barrier_before_it_runs() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let sink = Arc::new(InMemoryTrajectorySink::new(32));
        let delivery = Arc::new(InMemoryTrajectoryDelivery::new(32));
        let sampler = Arc::new(FixedSampler::new(0.0));
        let replay = Arc::new(Replay::new(family));
        let barrier_calls = Arc::new(AtomicUsize::new(0));
        let runtime = RouterRuntime::start_with_dependencies(
            config(family, 1),
            sampler,
            sink.clone(),
            delivery.clone(),
            Arc::new(CountingBarrier {
                calls: barrier_calls.clone(),
                fail: false,
            }),
        )
        .unwrap();
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let anchor = context(family);
        runtime
            .execute(
                anchor.clone(),
                request(family),
                Some(replay_transport.clone()),
                next_returning(response(family), Arc::new(AtomicUsize::new(0))),
            )
            .await
            .unwrap();
        yield_until_named("initial pending acknowledgement", || {
            sink.unresolved_pending().len() == 1
        })
        .await;
        runtime.observe_event_fail_open(&llm_end(&anchor));
        let later = later_context(&anchor);
        assert!(runtime.register_primary_call(&later).is_some());
        runtime.observe_event_fail_open(&llm_end(&later));
        yield_until(|| delivery.delivered_anchor_ids().len() == 1).await;

        let terminals = sink.terminal_payloads();
        assert_eq!(terminals.len(), 1);
        assert_eq!(
            terminals[0].state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached
            }
        );
        assert_eq!(
            terminals[0].pending.anchor_id.get_version(),
            Some(uuid::Version::SortRand)
        );
        tokio::time::advance(Duration::from_secs(2)).await;
        for _ in 0..100 {
            tokio::task::yield_now().await;
        }
        assert_eq!(barrier_calls.load(Ordering::SeqCst), 0);
        assert_eq!(sink.terminal_payloads().len(), 1);
        assert_eq!(delivery.delivered_anchor_ids().len(), 1);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn saturated_subscriber_command_queue_records_loss_and_rejects_window() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let runtime_config = config(family, 30);
        let total_max_pending = runtime_config.pools[0].concurrency.max_pending;
        let command_capacity =
            CoordinatorLimits::from_total_max_pending(total_max_pending).command_capacity;
        let sink = Arc::new(InMemoryTrajectorySink::new(32));
        let delivery = Arc::new(InMemoryTrajectoryDelivery::new(32));
        let replay = Arc::new(Replay::new(family));
        let runtime = RouterRuntime::start_with_dependencies(
            runtime_config,
            Arc::new(FixedSampler::new(0.0)),
            sink.clone(),
            delivery.clone(),
            Arc::new(ImmediateBarrier),
        )
        .unwrap();
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let anchor = context(family);
        runtime
            .execute(
                anchor.clone(),
                request(family),
                Some(replay_transport.clone()),
                next_returning(response(family), Arc::new(AtomicUsize::new(0))),
            )
            .await
            .unwrap();
        wait_for(|| {
            sink.unresolved_pending().len() == 1
                && runtime.state.coordinator_tx.capacity() == command_capacity
        })
        .await;

        for _ in 0..command_capacity {
            assert!(
                runtime
                    .state
                    .coordinator_tx
                    .try_send(CoordinatorCommand::Health("router.test.queue_fill"))
                    .is_ok()
            );
        }
        let subscriber = runtime.subscriber_callback();
        subscriber(&llm_end(&anchor));
        assert_eq!(runtime.state.loss.snapshot().highest_dropped_ingest_seq, 1);
        assert!(runtime.state.health.snapshot().rejected_records > 0);

        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        let terminals = sink.terminal_payloads();
        assert_eq!(terminals.len(), 1);
        assert_eq!(
            terminals[0].state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::EventLoss
            }
        );
        assert!(delivery.delivered_anchor_ids().is_empty());
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn activated_runtime_persists_middleware_interference_without_labels() {
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());

        for (case, expected_starts) in [("request", 0), ("response", 1)] {
            let temporary = tempdir().unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;

                fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
            }
            let family = LlmApiFamily::OpenAIChatCompletions;
            let mut config = config(family, 30);
            config.project_id = Some(format!("middleware-interference-{case}-test"));
            config.database_path = temporary
                .path()
                .join("ledger/router.db")
                .to_string_lossy()
                .into_owned();
            let database_path = config.database_path.clone();
            let activated = LedgerRepository::activate(&config).unwrap();
            let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();

            let mut candidate_response = response(family);
            candidate_response["id"] = json!(format!("chatcmpl-{case}-candidate"));
            candidate_response["object"] = json!("chat.completion");
            candidate_response["created"] = json!(1);
            candidate_response["model"] = json!("candidate-model");
            let mut judge_response = response(family);
            judge_response["id"] = json!(format!("chatcmpl-{case}-judge"));
            judge_response["object"] = json!("chat.completion");
            judge_response["created"] = json!(1);
            judge_response["model"] = json!("judge-model");
            let replay = ProductionReplay::new([
                ("candidate-model".to_string(), vec![candidate_response]),
                ("judge-model".to_string(), vec![judge_response]),
            ]);
            let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();

            let mut middleware =
                PluginRegistrationContext::with_namespace(format!("task9-{case}-interference:"));
            match case {
                "request" => middleware
                    .register_llm_request_intercept(
                        "mutate-request",
                        0,
                        false,
                        Arc::new(|_, mut request, annotated| {
                            request.headers.insert("x-mutated".into(), json!(true));
                            Ok(LlmRequestInterceptOutcome::new(request, annotated))
                        }),
                    )
                    .unwrap(),
                "response" => {
                    let mut replacement = response(family);
                    replacement["id"] = json!("chatcmpl-middleware-replacement");
                    middleware
                        .register_llm_execution_intercept_v2(
                            "replace-response",
                            0,
                            Arc::new(move |_, _, request, _, next| {
                                let replacement = replacement.clone();
                                Box::pin(async move {
                                    next(request).await?;
                                    Ok(replacement)
                                })
                            }),
                        )
                        .unwrap();
                }
                _ => unreachable!(),
            }

            let anchor = context(family);
            let anchor_response = response(family);
            let downstream_calls = Arc::new(AtomicUsize::new(0));
            let returned = runtime
                .execute(
                    anchor.clone(),
                    request(family),
                    Some(replay_transport),
                    next_returning(anchor_response.clone(), downstream_calls.clone()),
                )
                .await
                .unwrap();
            assert_eq!(returned, anchor_response);
            assert_eq!(downstream_calls.load(Ordering::SeqCst), 1);
            runtime.observe_event_fail_open(&llm_end(&anchor));
            let later = later_context(&anchor);
            assert!(runtime.register_primary_call(&later).is_some());
            runtime.observe_event_fail_open(&llm_end(&later));

            let connection = rusqlite::Connection::open(database_path).unwrap();
            tokio::time::timeout(Duration::from_secs(5), async {
                while sqlite_count(&connection, "shadow_results") != 1 {
                    tokio::task::yield_now().await;
                }
            })
            .await
            .expect("middleware interference did not terminalize durably");
            let (terminal_class, failure_class) = connection
                .query_row(
                    "SELECT terminal_class, operational_failure_class FROM shadow_results",
                    [],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                )
                .unwrap();
            assert_eq!(terminal_class, "operational_failure");
            assert_eq!(failure_class, "router.provider.middleware_interference");
            assert_eq!(
                connection
                    .query_row(
                        "SELECT count(*) FROM sample_batch_state_events WHERE state = 'closed'",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap(),
                1
            );
            assert_eq!(sqlite_count(&connection, "judge_attempts"), 0);
            assert_eq!(sqlite_count(&connection, "evaluations"), 0);
            let expected_models = if expected_starts == 0 {
                Vec::new()
            } else {
                vec!["candidate-model".to_string()]
            };
            assert_eq!(replay.started_models(), expected_models);

            let mut middleware = middleware.into_registrations();
            rollback_registrations(&mut middleware);
            assert!(middleware.is_empty());
            runtime
                .drain(Instant::now() + Duration::from_secs(5))
                .await
                .unwrap();
        }

        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn activated_runtime_recovers_after_durable_scheduler_pressure() {
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let family = LlmApiFamily::OpenAIChatCompletions;
        let mut config = config(family, 30);
        config.project_id = Some("scheduler-pressure-runtime-test".into());
        config.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        config.pools[0].concurrency.max_pending = 1;
        let database_path = config.database_path.clone();
        let activated = LedgerRepository::activate(&config).unwrap();
        let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
        let held = runtime
            .state
            .scheduler_controls
            .as_ref()
            .unwrap()
            .admissions
            .try_acquire("pool", 1)
            .unwrap();
        let replay = PendingProductionReplay::new();
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();

        let pressured_response = response(family);
        let pressured_calls = Arc::new(AtomicUsize::new(0));
        assert_eq!(
            runtime
                .execute(
                    context(family),
                    request(family),
                    Some(replay_transport.clone()),
                    next_returning(pressured_response.clone(), pressured_calls.clone()),
                )
                .await
                .unwrap(),
            pressured_response
        );
        let connection = rusqlite::Connection::open(&database_path).unwrap();
        tokio::time::timeout(Duration::from_secs(5), async {
            while connection
                .query_row(
                    "SELECT count(*) FROM anchor_state_events WHERE state = 'not_scheduled_queue_full'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap()
                != 1
                || runtime.state.sampling_enabled.load(Ordering::Acquire)
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("queue pressure was not durably recorded and suppressed");
        assert_eq!(pressured_calls.load(Ordering::SeqCst), 1);
        assert_eq!(sqlite_count(&connection, "anchors"), 1);
        assert_eq!(sqlite_count(&connection, "sample_batches"), 0);
        assert_eq!(replay.starts.load(Ordering::Acquire), 0);

        let suppressed_response = response(family);
        let suppressed_calls = Arc::new(AtomicUsize::new(0));
        assert_eq!(
            runtime
                .execute(
                    context(family),
                    request(family),
                    Some(replay_transport.clone()),
                    next_returning(suppressed_response.clone(), suppressed_calls.clone()),
                )
                .await
                .unwrap(),
            suppressed_response
        );
        tokio::task::yield_now().await;
        assert_eq!(suppressed_calls.load(Ordering::SeqCst), 1);
        assert_eq!(sqlite_count(&connection, "anchors"), 1);
        assert_eq!(replay.starts.load(Ordering::Acquire), 0);

        drop(held);
        wait_for(|| runtime.state.sampling_enabled.load(Ordering::Acquire)).await;

        let recovered = context(family);
        let recovered_response = response(family);
        let recovered_calls = Arc::new(AtomicUsize::new(0));
        assert_eq!(
            runtime
                .execute(
                    recovered.clone(),
                    request(family),
                    Some(replay_transport),
                    next_returning(recovered_response.clone(), recovered_calls.clone()),
                )
                .await
                .unwrap(),
            recovered_response
        );
        runtime.observe_event_fail_open(&llm_end(&recovered));
        let later = later_context(&recovered);
        assert!(runtime.register_primary_call(&later).is_some());
        runtime.observe_event_fail_open(&llm_end(&later));
        wait_for(|| replay.starts.load(Ordering::Acquire) == 1).await;
        assert_eq!(recovered_calls.load(Ordering::SeqCst), 1);
        assert_eq!(sqlite_count(&connection, "anchors"), 2);
        assert_eq!(sqlite_count(&connection, "sample_batches"), 1);
        assert_eq!(sqlite_count(&connection, "shadow_attempts"), 1);

        runtime.abort();
        wait_for(|| replay.cancellations.load(Ordering::Acquire) == 1).await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());
    }

    #[tokio::test]
    async fn immediate_abort_leaves_only_acknowledged_pending_for_recovery() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let (runtime, sink, delivery, _, replay) = runtime(family, 30);
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        runtime
            .execute(
                context(family),
                request(family),
                Some(replay_transport.clone()),
                next_returning(response(family), Arc::new(AtomicUsize::new(0))),
            )
            .await
            .unwrap();
        wait_for(|| sink.unresolved_pending().len() == 1).await;
        runtime.abort();
        tokio::task::yield_now().await;

        assert_eq!(sink.unresolved_pending().len(), 1);
        assert!(sink.terminal_payloads().is_empty());
        assert!(delivery.delivered_anchor_ids().is_empty());
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn off_and_non_primary_calls_bypass_observation() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let sink = Arc::new(InMemoryTrajectorySink::new(32));
        let delivery = Arc::new(InMemoryTrajectoryDelivery::new(32));
        let sampler = Arc::new(FixedSampler::new(0.0));
        let replay = Arc::new(Replay::new(family));
        let mut off_config = config(family, 30);
        off_config.mode = RouterMode::Off;
        let off = RouterRuntime::start_with_dependencies(
            off_config,
            sampler.clone(),
            sink.clone(),
            delivery.clone(),
            Arc::new(ImmediateBarrier),
        )
        .unwrap();
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let raw_response = response(family);
        assert_eq!(
            off.execute(
                context(family),
                request(family),
                Some(replay_transport.clone()),
                next_returning(raw_response.clone(), Arc::new(AtomicUsize::new(0))),
            )
            .await
            .unwrap(),
            raw_response
        );
        off.drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(sampler.calls.load(Ordering::SeqCst), 0);
        assert_eq!(replay.capability_calls.load(Ordering::SeqCst), 0);
        assert!(sink.terminal_payloads().is_empty());

        let (runtime, sink, delivery, sampler, replay) = runtime(family, 30);
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let mut shadow = context(family);
        shadow.call_role = LlmCallRole::Shadow;
        let downstream_calls = Arc::new(AtomicUsize::new(0));
        let health_before = runtime.state.health.snapshot();
        let loss_before = runtime.state.loss.snapshot();
        let request = request(family);
        let expected_response = response(family);
        let returned = runtime
            .execute(
                shadow,
                request.clone(),
                Some(replay_transport.clone()),
                next_returning(expected_response.clone(), downstream_calls.clone()),
            )
            .await
            .unwrap();
        assert_eq!(returned, expected_response);
        assert_eq!(downstream_calls.load(Ordering::SeqCst), 1);
        assert_eq!(runtime.state.health.snapshot(), health_before);
        assert_eq!(runtime.state.loss.snapshot(), loss_before);
        assert_eq!(available_sample_intents(&runtime), 4);
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        assert_eq!(sampler.calls.load(Ordering::SeqCst), 0);
        assert_eq!(replay.capability_calls.load(Ordering::SeqCst), 0);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        assert!(sink.terminal_payloads().is_empty());
        assert!(delivery.delivered_anchor_ids().is_empty());
    }

    #[tokio::test]
    async fn internal_events_are_filtered_before_sequence_projection_and_health() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let (runtime, sink, delivery, sampler, replay) = runtime(family, 30);
        let health_before = runtime.state.health.snapshot();
        let loss_before = runtime.state.loss.snapshot();

        let evaluator = Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .name("hostile-evaluator-name")
                .uuid(Uuid::now_v7())
                .data(json!({"oversized": "x".repeat(128 * 1024)}))
                .build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::evaluator(),
            None,
        ));
        let embedder = Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .name("hostile-embedder-name")
                .uuid(Uuid::now_v7())
                .data(json!({"oversized": "x".repeat(128 * 1024)}))
                .build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::from(ScopeType::Embedder),
            None,
        ));
        let mut shadow_profile = CategoryProfile::default();
        shadow_profile.set_llm_call_role(LlmCallRole::Shadow);
        let shadow = Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .name("pretend-primary")
                .uuid(Uuid::now_v7())
                .parent_uuid(evaluator.uuid())
                .build(),
            ScopeCategory::End,
            Vec::new(),
            EventCategory::llm(),
            Some(shadow_profile),
        ));

        runtime.observe_event_fail_open(&evaluator);
        runtime.observe_event_fail_open(&embedder);
        runtime.observe_event_fail_open(&shadow);

        assert_eq!(runtime.state.loss.current_ingest_seq(), 0);
        assert_eq!(runtime.state.loss.snapshot(), loss_before);
        assert_eq!(runtime.state.health.snapshot(), health_before);
        assert_eq!(sampler.calls.load(Ordering::SeqCst), 0);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        assert!(sink.unresolved_pending().is_empty());
        assert!(sink.terminal_payloads().is_empty());
        assert!(delivery.delivered_anchor_ids().is_empty());

        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn scheduler_pressure_waiter_observes_live_and_preexisting_recovery() {
        let mut config = config(LlmApiFamily::OpenAIChatCompletions, 30);
        config.pools[0].concurrency.max_pending = 1;
        let pools = SchedulerAdmissionPools::from_config(&config).unwrap();

        let held = pools.try_acquire("pool", 1).unwrap();
        assert!(matches!(
            pools.try_acquire("pool", 1),
            Err(crate::scheduler_admission::SchedulerAdmissionError::NoPermits)
        ));
        let pressure = pools.subscribe_pressure("pool").unwrap();
        let (sender, mut receiver) = mpsc::channel(1);
        let waiter = tokio::spawn(await_scheduler_pressure_recovery(
            pressure,
            sender,
            "pool".into(),
            "pool".into(),
            7,
        ));
        tokio::task::yield_now().await;
        assert!(receiver.try_recv().is_err());
        drop(held);
        assert!(matches!(
            receiver.recv().await,
            Some(CoordinatorCommand::SchedulerCapacityRecovered {
                pool_id,
                pressure_generation: 7,
            }) if pool_id == "pool"
        ));
        assert_eq!(
            waiter.await.unwrap(),
            ActorTaskCompletion::SchedulerPressure {
                pool_id: "pool".into(),
                generation: 7,
            }
        );

        let held = pools.try_acquire("pool", 1).unwrap();
        assert!(pools.try_acquire("pool", 1).is_err());
        let pressure = pools.subscribe_pressure("pool").unwrap();
        drop(held);
        let (sender, mut receiver) = mpsc::channel(1);
        let completion =
            await_scheduler_pressure_recovery(pressure, sender, "pool".into(), "pool".into(), 8)
                .await;
        assert!(matches!(
            receiver.recv().await,
            Some(CoordinatorCommand::SchedulerCapacityRecovered {
                pool_id,
                pressure_generation: 8,
            }) if pool_id == "pool"
        ));
        assert_eq!(
            completion,
            ActorTaskCompletion::SchedulerPressure {
                pool_id: "pool".into(),
                generation: 8,
            }
        );
    }

    #[test]
    fn active_replay_registry_capacity_is_bounded_by_simultaneous_provider_work() {
        let mut config = config(LlmApiFamily::OpenAIChatCompletions, 30);
        assert_eq!(config.active_replay_capacity().unwrap(), 2);

        config.pools[0].concurrency.shadow = usize::MAX;
        config.pools[0].concurrency.judge = usize::MAX;
        assert_eq!(config.active_replay_capacity().unwrap(), 4);

        let mut second = config.pools[0].clone();
        second.id = "second-pool".into();
        second.concurrency.shadow = 1;
        second.concurrency.judge = 1;
        second.concurrency.max_pending = 1;
        config.pools.push(second);
        assert_eq!(config.active_replay_capacity().unwrap(), 5);
    }

    #[tokio::test]
    async fn runtime_accepts_fixed_private_identity_for_deterministic_tests() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let identity = TrajectoryIdentity {
            process_instance_id: Uuid::from_u128(11),
            project_uuid: Uuid::from_u128(12),
            project_id: "fixed-project".into(),
            policy_version_ids: BTreeMap::from([(
                "pool".to_string(),
                "fixed-policy-version".to_string(),
            )]),
            learning_generation_ids: BTreeMap::from([("pool".to_string(), Uuid::from_u128(13))]),
        };
        let sink = Arc::new(InMemoryTrajectorySink::new(4));
        let delivery = Arc::new(InMemoryTrajectoryDelivery::new(4));
        let runtime = RouterRuntime::start_with_dependencies_and_identity(
            config(family, 30),
            Arc::new(FixedSampler::new(0.0)),
            sink,
            delivery,
            Arc::new(ImmediateBarrier),
            Some(identity.clone()),
        )
        .unwrap();
        assert_eq!(runtime.state.identity, identity);
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn runtime_uses_and_retains_ledger_derived_identity() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mut config = config(LlmApiFamily::OpenAIChatCompletions, 30);
        config.project_id = Some("ledger-runtime-test".into());
        config.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        let activated = LedgerRepository::activate(&config).unwrap();
        let expected = activated.identity.clone();

        let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
        assert_eq!(
            runtime.state.identity.process_instance_id,
            expected.process_instance_id
        );
        assert_eq!(runtime.state.identity.project_uuid, expected.project_uuid);
        assert_eq!(runtime.state.identity.project_id, expected.project_id);
        assert_eq!(
            runtime.state.identity.policy_version_id("pool"),
            Some(expected.pools["pool"].policy_version_id.as_str())
        );
        assert_eq!(
            runtime.state.identity.learning_generation_id("pool"),
            Some(expected.pools["pool"].learning_generation_id)
        );
        assert_eq!(
            runtime.state.config_generation_id,
            expected.config_generation_id
        );
        let generation = runtime
            .state
            .control_authority
            .as_ref()
            .expect("activated runtime must retain generation authority")
            .current_generation_snapshot()
            .expect("activated generation authority must be current");
        assert_eq!(
            generation.cohort_assignment.cohort_generation_id(),
            expected.cohort_generation_id
        );
        assert!(runtime.writer.lock().unwrap().is_some());
        assert!(runtime.state.writer_client.is_some());
        assert!(runtime.state.read_pool.is_some());
        assert!(runtime.background.lock().unwrap().is_some());
        let live_embedding = runtime
            .live_embedding_service()
            .expect("activated runtime must own live embedding");
        assert!(!live_embedding.is_closed());
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        assert!(live_embedding.is_closed());
        assert!(runtime.live_embedding_service().is_some());
        assert!(runtime.background.lock().unwrap().is_none());
        assert!(runtime.writer.lock().unwrap().is_none());
        let connection = rusqlite::Connection::open(&runtime.state.config.database_path).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT group_concat(state, ',') FROM (
                         SELECT state FROM process_instance_state_events
                         WHERE process_instance_id = ?1 ORDER BY event_seq
                     )",
                    [expected.process_instance_id.to_string()],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            "started,stopped"
        );
    }

    #[tokio::test]
    async fn activated_identity_mismatch_stops_the_exact_process_before_returning() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mut activated_config = config(LlmApiFamily::OpenAIChatCompletions, 30);
        activated_config.project_id = Some("identity-mismatch-runtime-test".into());
        activated_config.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        let database_path = activated_config.database_path.clone();
        let activated = LedgerRepository::activate(&activated_config).unwrap();
        let process_instance_id = activated.identity.process_instance_id;
        let mut mismatched_config = activated_config;
        mismatched_config.pools[0].sampling_probability = 0.5;

        let result = RouterRuntime::start_with_activated_ledger(mismatched_config, activated);
        assert!(result.is_err());

        let connection = rusqlite::Connection::open(database_path).unwrap();
        let stopped = connection
            .query_row(
                "SELECT count(*) FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'stopped'",
                [process_instance_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(stopped, 1);
    }

    #[test]
    fn invalid_runtime_config_stops_activation_before_matcher_failure_returns() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mut activated_config = config(LlmApiFamily::OpenAIChatCompletions, 30);
        activated_config.project_id = Some("invalid-runtime-config-test".into());
        activated_config.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        let database_path = activated_config.database_path.clone();
        let activated = LedgerRepository::activate(&activated_config).unwrap();
        let process_instance_id = activated.identity.process_instance_id;
        let mut invalid_config = activated_config;
        invalid_config.pools.push(invalid_config.pools[0].clone());

        let result = RouterRuntime::start_with_activated_ledger(invalid_config, activated);
        assert!(result.is_err());

        let connection = rusqlite::Connection::open(database_path).unwrap();
        let stopped = connection
            .query_row(
                "SELECT count(*) FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'stopped'",
                [process_instance_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(stopped, 1);
    }

    #[test]
    fn missing_tokio_runtime_stops_the_activated_process_before_returning() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mut config = config(LlmApiFamily::OpenAIChatCompletions, 30);
        config.project_id = Some("missing-tokio-runtime-test".into());
        config.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        let database_path = config.database_path.clone();
        let activated = LedgerRepository::activate(&config).unwrap();
        let process_instance_id = activated.identity.process_instance_id;

        let result = RouterRuntime::start_with_activated_ledger(config, activated);
        assert!(result.is_err());

        let connection = rusqlite::Connection::open(database_path).unwrap();
        let stopped = connection
            .query_row(
                "SELECT count(*) FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'stopped'",
                [process_instance_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(stopped, 1);
    }

    fn activated_lifecycle_runtime(project_id: &str) -> (tempfile::TempDir, Arc<RouterRuntime>) {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mut config = config(LlmApiFamily::OpenAIChatCompletions, 30);
        config.project_id = Some(project_id.into());
        config.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        let activated = LedgerRepository::activate(&config).unwrap();
        let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
        (temporary, runtime)
    }

    fn activated_recommend_lifecycle_runtime(
        project_id: &str,
    ) -> (tempfile::TempDir, Arc<RouterRuntime>) {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mut config = config(LlmApiFamily::OpenAIChatCompletions, 30);
        config.mode = RouterMode::Recommend;
        config.project_id = Some(project_id.into());
        config.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        config.embedders = vec![EmbedderConfig {
            id: "recommend-lifecycle-embedder".into(),
            base_url: "http://127.0.0.1:9/v1".into(),
            model: "embedding-model".into(),
            provider_revision: "embedding-r1".into(),
            dimensions: 2,
            api_key_env: None,
            timeout_ms: 10_000,
            max_in_flight: 1,
            batch_size: 1,
            unknown_fields: BTreeMap::new(),
        }];
        config.pools[0].learning = Some(LearningConfig {
            version: 1,
            embedder: "recommend-lifecycle-embedder".into(),
            top_k: Some(1),
            radius: Some(1.0),
            min_points: Some(1),
            min_independent_roots: Some(1),
            min_effective_samples: Some(1.0),
            min_coverage: Some(0.0),
            time_decay_half_life_seconds: Some(3_600.0),
            prior_success: Some(1.0),
            prior_failure: Some(1.0),
            familywise_credible_level: Some(0.95),
            promotion_lower_bound: Some(0.0),
            retention_lower_bound: None,
            holdout_probability: None,
            active_canary_fraction: None,
        });
        assert!(config.validate().is_empty(), "{:?}", config.validate());
        let activated = LedgerRepository::activate(&config).unwrap();
        let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
        (temporary, runtime)
    }

    struct PausedRecommendDelivery {
        temporary: tempfile::TempDir,
        runtime: Arc<RouterRuntime>,
        delivery: RecommendationDeliveryServiceV1,
        process_instance_id: Uuid,
        release_writer: std::sync::mpsc::SyncSender<()>,
        paused_writer: JoinHandle<Result<(), WriterFailure>>,
        execution: JoinHandle<FlowResult<Json>>,
        expected_response: Json,
        next_calls: Arc<AtomicUsize>,
        replay: Arc<Replay>,
    }

    async fn paused_recommend_delivery(project_id: &str) -> PausedRecommendDelivery {
        let (temporary, runtime) = activated_recommend_lifecycle_runtime(project_id);
        runtime
            .live_embedding_service()
            .expect("Recommend runtime must own live embedding")
            .close();
        let delivery = runtime
            .state
            .recommendation_delivery
            .as_ref()
            .expect("Recommend runtime must own audit delivery")
            .clone();
        let process_instance_id = runtime.state.identity.process_instance_id;

        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_writer, release_rx) = std::sync::mpsc::sync_channel(1);
        let writer = runtime.state.writer_client.clone().unwrap();
        let paused_writer = tokio::spawn(async move {
            writer
                .pause_until(
                    Instant::now() + Duration::from_secs(30),
                    started_tx,
                    release_rx,
                )
                .await
        });
        started_rx
            .recv_timeout(Duration::from_secs(2))
            .expect("sole writer did not enter the test pause");

        let replay = Arc::new(Replay::new(LlmApiFamily::OpenAIChatCompletions));
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let expected_response = response(LlmApiFamily::OpenAIChatCompletions);
        let next_calls = Arc::new(AtomicUsize::new(0));
        let execution_runtime = runtime.clone();
        let execution_response = expected_response.clone();
        let execution_calls = next_calls.clone();
        let execution = tokio::spawn(async move {
            execution_runtime
                .execute(
                    context(LlmApiFamily::OpenAIChatCompletions),
                    request(LlmApiFamily::OpenAIChatCompletions),
                    Some(replay_transport),
                    next_returning(execution_response, execution_calls),
                )
                .await
        });
        wait_for(|| {
            matches!(
                delivery.health(),
                RecommendationDeliveryHealthV1::DegradedPending { pending: 1 }
            )
        })
        .await;

        PausedRecommendDelivery {
            temporary,
            runtime,
            delivery,
            process_instance_id,
            release_writer,
            paused_writer,
            execution,
            expected_response,
            next_calls,
            replay,
        }
    }

    fn process_stop_count(database_path: &std::path::Path, process_instance_id: Uuid) -> i64 {
        rusqlite::Connection::open(database_path)
            .unwrap()
            .query_row(
                "SELECT count(*) FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'stopped'",
                [process_instance_id.to_string()],
                |row| row.get(0),
            )
            .unwrap()
    }

    #[tokio::test]
    async fn activated_runtime_heartbeat_renews_through_the_owned_writer() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mut config = config(LlmApiFamily::OpenAIChatCompletions, 30);
        config.project_id = Some("heartbeat-runtime-test".into());
        config.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        let activated = LedgerRepository::activate(&config).unwrap();
        let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
        runtime
            .test_run_heartbeat_once(Utc::now().timestamp_millis())
            .await
            .unwrap();
        let writer = runtime.state.writer_client.clone().unwrap();
        let (stop, stop_rx) = watch::channel(false);
        let heartbeat = tokio::spawn(run_heartbeat(
            runtime.state.clone(),
            writer,
            stop_rx,
            Duration::from_millis(1),
        ));

        wait_for(|| runtime.state.heartbeat_ticks.load(Ordering::Acquire) >= 2).await;
        stop.send_replace(true);
        heartbeat.await.unwrap().unwrap();
        assert!(runtime.state.sampling_enabled.load(Ordering::Acquire));
        runtime.abort();
    }

    #[tokio::test]
    async fn activated_runtime_retention_uses_the_owned_worker_for_manual_and_pressure_ticks() {
        let (_temporary, runtime) = activated_lifecycle_runtime("retention-runtime-test");
        let acknowledgement = runtime
            .test_run_retention_once(Utc::now().timestamp_millis())
            .await
            .unwrap();
        assert!(matches!(
            acknowledgement,
            RetentionAck::Applied {
                summary,
                observation,
            } if summary.selected_count == 0 && observation.capacity_available
        ));

        let pressure = runtime.state.retention_pressure.as_ref().unwrap();
        pressure.send_replace(Some(41));
        wait_for(|| runtime.state.retention_ticks.load(Ordering::Acquire) >= 2).await;
        assert_eq!(
            *runtime
                .state
                .retention_recoveries
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            vec![41]
        );
        assert!(runtime.state.sampling_enabled.load(Ordering::Acquire));
        runtime.abort();
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn owned_heartbeat_and_retention_workers_observe_exact_timer_boundaries() {
        let (_temporary, runtime) = activated_lifecycle_runtime("runtime-timer-boundary-test");
        yield_until(|| {
            runtime.state.heartbeat_worker_ready.load(Ordering::Acquire)
                && runtime.state.retention_worker_ready.load(Ordering::Acquire)
        })
        .await;
        assert_eq!(runtime.state.heartbeat_ticks.load(Ordering::Acquire), 0);
        assert_eq!(runtime.state.retention_ticks.load(Ordering::Acquire), 0);

        tokio::time::advance(HEARTBEAT_INTERVAL - Duration::from_millis(1)).await;
        tokio::task::yield_now().await;
        assert_eq!(runtime.state.heartbeat_ticks.load(Ordering::Acquire), 0);
        assert_eq!(runtime.state.retention_ticks.load(Ordering::Acquire), 0);

        tokio::time::advance(Duration::from_millis(1)).await;
        yield_until_blocking_work("heartbeat tick", || {
            runtime.state.heartbeat_ticks.load(Ordering::Acquire) >= 1
        })
        .await;
        assert_eq!(runtime.state.retention_ticks.load(Ordering::Acquire), 0);
        let mut heartbeat = runtime.heartbeat.lock().unwrap().take().unwrap();
        heartbeat.stop.send_replace(true);
        (&mut heartbeat.task).await.unwrap().unwrap();

        tokio::time::advance(RETENTION_INTERVAL - HEARTBEAT_INTERVAL - Duration::from_millis(1))
            .await;
        tokio::task::yield_now().await;
        assert_eq!(runtime.state.retention_ticks.load(Ordering::Acquire), 0);

        tokio::time::advance(Duration::from_millis(1)).await;
        yield_until(|| runtime.state.retention_timer_fires.load(Ordering::Acquire) == 1).await;
        runtime.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn retention_deadline_retries_one_frozen_request_and_recovers_only_new_generation() {
        let (_temporary, runtime) = activated_lifecycle_runtime("retention-lost-ack-test");
        let writer = runtime.state.writer_client.clone().unwrap();
        let pause_deadline = Instant::now() + Duration::from_secs(2);
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let pause_writer = writer.clone();
        let pause = tokio::spawn(async move {
            pause_writer
                .pause_until(pause_deadline, started_tx, release_rx)
                .await
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let (stop, mut stop_rx) = watch::channel(false);
        let (pressure, mut pressure_rx) = watch::channel(Some(71));
        let trigger_state = runtime.state.clone();
        let trigger_writer = writer.clone();
        let trigger = tokio::spawn(async move {
            run_retention_trigger(
                &trigger_state,
                &trigger_writer,
                &mut stop_rx,
                &mut pressure_rx,
                Some(71),
                Duration::from_millis(40),
                Duration::from_millis(10),
            )
            .await
        });

        wait_for(|| runtime.state.health.snapshot().last_reason == Some(HEALTH_RETENTION_FAILURE))
            .await;
        pressure.send_replace(Some(72));
        tokio::task::yield_now().await;
        release_tx.send(()).unwrap();
        pause.await.unwrap().unwrap();
        trigger.await.unwrap().unwrap();

        assert!(
            runtime
                .state
                .retention_recoveries
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_empty()
        );
        let mut next_stop = stop.subscribe();
        let mut next_pressure = pressure.subscribe();
        run_retention_trigger(
            &runtime.state,
            &writer,
            &mut next_stop,
            &mut next_pressure,
            Some(72),
            Duration::from_millis(40),
            Duration::from_millis(10),
        )
        .await
        .unwrap();
        stop.send_replace(true);

        let connection = rusqlite::Connection::open(&runtime.state.config.database_path).unwrap();
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM retention_batches", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            2
        );
        assert_eq!(runtime.state.retention_ticks.load(Ordering::Acquire), 2);
        assert_eq!(
            runtime
                .state
                .retention_already_applied_ticks
                .load(Ordering::Acquire),
            1
        );
        assert_eq!(
            *runtime
                .state
                .retention_recoveries
                .lock()
                .unwrap_or_else(|error| error.into_inner()),
            vec![72]
        );
        assert_eq!(
            runtime.state.health.snapshot().last_reason,
            Some(HEALTH_RETENTION_FAILURE)
        );
        runtime.abort();
    }

    #[tokio::test]
    async fn retention_vector_unavailability_marks_health_then_retries_the_frozen_request() {
        let (_temporary, config, activated, vector_space_id, root, evidence_vector_link_id) =
            crate::ledger::repository::ready_retention_runtime_fixture();
        let database_path = config.database_path.clone();
        let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
        let writer = runtime.state.writer_client.clone().unwrap();
        let connection = rusqlite::Connection::open(database_path).unwrap();
        let (generation, active_hash, initial_state): (i64, String, String) = connection
            .query_row(
                "SELECT generation, canonical_payload_hash, state
                 FROM vector_index_manifest
                 WHERE vector_space_id = ?1 AND state = 'active'",
                [vector_space_id.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(initial_state, "active");
        connection
            .execute(
                &format!("DELETE FROM \"{root}\" WHERE record_id = ?1"),
                [evidence_vector_link_id.to_string()],
            )
            .unwrap();
        let request = RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), 40).unwrap();

        assert_eq!(
            run_retention_request_with_policy(
                &runtime.state,
                &writer,
                request,
                Duration::from_secs(2),
            )
            .await,
            Err(RetentionPassFailure::Transient)
        );
        let (unavailable_generation, unavailable_state): (i64, String) = connection
            .query_row(
                "SELECT generation, state FROM vector_index_manifest
                 WHERE vector_space_id = ?1 AND canonical_payload_hash != ?2",
                rusqlite::params![vector_space_id.as_str(), active_hash],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(unavailable_generation, generation);
        assert_eq!(unavailable_state, "unavailable");
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM retention_batches", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM evidence_vector_links", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            1
        );

        let applied = run_retention_request_with_policy(
            &runtime.state,
            &writer,
            request,
            Duration::from_secs(2),
        )
        .await
        .unwrap();
        assert!(matches!(
            applied,
            RetentionAck::Applied { ref summary, .. }
                if summary.retention_batch_id == request.retention_batch_id
                    && summary.conflict_health_event_id == request.conflict_health_event_id
        ));
        assert_eq!(
            connection
                .query_row(
                    "SELECT state FROM vector_index_manifest
                     WHERE vector_space_id = ?1 AND generation = ?2",
                    rusqlite::params![vector_space_id.as_str(), generation],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            "retired"
        );
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM retention_batches", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            1
        );
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM evidence_vector_links", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            0
        );

        assert!(matches!(
            run_retention_request_with_policy(
                &runtime.state,
                &writer,
                request,
                Duration::from_secs(2),
            )
            .await
            .unwrap(),
            RetentionAck::AlreadyApplied { ref summary, .. }
                if summary.retention_batch_id == request.retention_batch_id
        ));
        assert_eq!(runtime.state.retention_ticks.load(Ordering::Acquire), 2);
        assert_eq!(
            runtime
                .state
                .retention_already_applied_ticks
                .load(Ordering::Acquire),
            1
        );
        runtime.abort();
    }

    #[test]
    fn retention_writer_failure_classification_is_bounded() {
        assert_eq!(
            classify_retention_writer_failure(WriterFailure::new(WriterFailureClass::Deadline)),
            RetentionPassFailure::Transient
        );
        assert_eq!(
            classify_retention_writer_failure(WriterFailure::new(WriterFailureClass::Repository(
                LedgerErrorClass::Busy,
            ))),
            RetentionPassFailure::Transient
        );
        assert_eq!(
            classify_retention_writer_failure(WriterFailure::new(WriterFailureClass::Repository(
                LedgerErrorClass::DatabaseOperationFailed,
            ))),
            RetentionPassFailure::Permanent
        );
        assert_eq!(
            classify_retention_writer_failure(WriterFailure::new(WriterFailureClass::Protocol)),
            RetentionPassFailure::Permanent
        );
    }

    #[tokio::test]
    async fn retention_worker_failure_fails_graceful_drain_and_abort_writes_no_process_stop() {
        let (temporary, runtime) = activated_lifecycle_runtime("retention-worker-failure-test");
        let process_instance_id = runtime.state.identity.process_instance_id;
        assert!(runtime.test_run_retention_once(-1).await.is_err());
        assert!(
            runtime
                .state
                .permanent_runtime_fault
                .load(Ordering::Acquire)
        );

        let error = runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap_err();
        assert!(error.to_string().contains("retention"));
        assert!(runtime.retention.lock().unwrap().is_some());
        assert!(runtime.heartbeat.lock().unwrap().is_some());
        assert!(runtime.writer.lock().unwrap().is_some());
        assert_eq!(
            runtime.lifecycle.load(Ordering::Acquire),
            LIFECYCLE_DRAINING
        );
        runtime.abort();

        let connection =
            rusqlite::Connection::open(temporary.path().join("ledger/router.db")).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM process_instance_state_events
                     WHERE process_instance_id = ?1 AND state = 'stopped'",
                    [process_instance_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn heartbeat_worker_failure_fails_graceful_drain_before_process_stop() {
        let (temporary, runtime) = activated_lifecycle_runtime("heartbeat-worker-failure-test");
        let process_instance_id = runtime.state.identity.process_instance_id;
        yield_until(|| runtime.state.heartbeat_worker_ready.load(Ordering::Acquire)).await;
        runtime.state.writer_client.as_ref().unwrap().abort();
        tokio::time::sleep(HEARTBEAT_INTERVAL + Duration::from_millis(25)).await;
        wait_for(|| {
            runtime
                .heartbeat
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_ref()
                .is_some_and(|heartbeat| heartbeat.task.is_finished())
        })
        .await;

        let error = runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("heartbeat"),
            "unexpected drain error: {error}"
        );
        assert!(runtime.retention.lock().unwrap().is_none());
        assert!(runtime.heartbeat.lock().unwrap().is_some());
        assert!(runtime.writer.lock().unwrap().is_some());
        assert_eq!(
            runtime.lifecycle.load(Ordering::Acquire),
            LIFECYCLE_DRAINING
        );
        runtime.abort();

        let connection =
            rusqlite::Connection::open(temporary.path().join("ledger/router.db")).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM process_instance_state_events
                     WHERE process_instance_id = ?1 AND state = 'stopped'",
                    [process_instance_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
    }

    #[tokio::test]
    async fn writer_unavailability_suppresses_sampling_but_preserves_anchor_results() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let family = LlmApiFamily::OpenAIChatCompletions;
        let mut config = config(family, 30);
        config.project_id = Some("writer-failure-runtime-test".into());
        config.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        let activated = LedgerRepository::activate(&config).unwrap();
        let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
        runtime.writer.lock().unwrap().as_ref().unwrap().abort();
        let replay = Arc::new(Replay::new(family));
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();

        let first_response = response(family);
        let first = runtime
            .execute(
                context(family),
                request(family),
                Some(replay_transport.clone()),
                next_returning(first_response.clone(), Arc::new(AtomicUsize::new(0))),
            )
            .await
            .unwrap();
        assert_eq!(first, first_response);
        wait_for(|| !runtime.state.sampling_enabled.load(Ordering::Acquire)).await;

        let second_response = response(family);
        let second = runtime
            .execute(
                context(family),
                request(family),
                Some(replay_transport),
                next_returning(second_response.clone(), Arc::new(AtomicUsize::new(0))),
            )
            .await
            .unwrap();
        assert_eq!(second, second_response);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        assert!(runtime.state.health.snapshot().last_reason.is_some());
        runtime.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recommend_lifecycle_graceful_drain_waits_for_admitted_live_work() {
        let (temporary, runtime) =
            activated_recommend_lifecycle_runtime("recommend-live-drain-runtime-test");
        let database_path = temporary.path().join("ledger/router.db");
        let process_instance_id = runtime.state.identity.process_instance_id;
        let live_embedding = runtime
            .live_embedding_service()
            .expect("Recommend runtime must own live embedding");
        let admitted = Arc::new(tokio::sync::Barrier::new(2));
        let release = Arc::new(tokio::sync::Barrier::new(2));
        live_embedding.pause_next_call_after_admission(admitted.clone(), release.clone());

        let replay = Arc::new(Replay::new(LlmApiFamily::OpenAIChatCompletions));
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let expected_response = response(LlmApiFamily::OpenAIChatCompletions);
        let next_calls = Arc::new(AtomicUsize::new(0));
        let execution_runtime = runtime.clone();
        let execution_response = expected_response.clone();
        let execution_calls = next_calls.clone();
        let execution = tokio::spawn(async move {
            execution_runtime
                .execute(
                    context(LlmApiFamily::OpenAIChatCompletions),
                    request(LlmApiFamily::OpenAIChatCompletions),
                    Some(replay_transport),
                    next_returning(execution_response, execution_calls),
                )
                .await
        });
        tokio::time::timeout(Duration::from_secs(2), admitted.wait())
            .await
            .expect("Recommend work did not reach live-embedding admission");

        let drain_runtime = runtime.clone();
        let drain = tokio::spawn(async move {
            drain_runtime
                .drain(Instant::now() + Duration::from_secs(5))
                .await
        });
        wait_for(|| runtime.lifecycle.load(Ordering::Acquire) == LIFECYCLE_DRAINING).await;
        tokio::task::yield_now().await;
        assert!(live_embedding.is_closed());
        assert!(!drain.is_finished());
        assert_eq!(process_stop_count(&database_path, process_instance_id), 0);
        assert!(runtime.background.lock().unwrap().is_some());
        assert!(runtime.retention.lock().unwrap().is_some());
        assert!(runtime.writer.lock().unwrap().is_some());

        release.wait().await;
        assert_eq!(execution.await.unwrap().unwrap(), expected_response);
        drain.await.unwrap().unwrap();
        assert_eq!(next_calls.load(Ordering::SeqCst), 1);
        assert_eq!(replay.capability_calls.load(Ordering::SeqCst), 1);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        let connection = rusqlite::Connection::open(&database_path).unwrap();
        assert_eq!(sqlite_count(&connection, "decisions"), 1);
        assert_eq!(process_stop_count(&database_path, process_instance_id), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recommend_lifecycle_graceful_drain_commits_pending_audit_before_process_stop() {
        let PausedRecommendDelivery {
            temporary,
            runtime,
            delivery,
            process_instance_id,
            release_writer,
            paused_writer,
            execution,
            expected_response,
            next_calls,
            replay,
        } = paused_recommend_delivery("recommend-graceful-order-runtime-test").await;
        let database_path = temporary.path().join("ledger/router.db");

        let drain_runtime = runtime.clone();
        let drain = tokio::spawn(async move {
            drain_runtime
                .drain(Instant::now() + Duration::from_secs(5))
                .await
        });
        wait_for(|| runtime.lifecycle.load(Ordering::Acquire) == LIFECYCLE_DRAINING).await;
        tokio::task::yield_now().await;
        assert!(!drain.is_finished());
        assert_eq!(process_stop_count(&database_path, process_instance_id), 0);
        assert!(runtime.background.lock().unwrap().is_some());
        assert!(runtime.retention.lock().unwrap().is_some());
        assert!(runtime.writer.lock().unwrap().is_some());

        release_writer.send(()).unwrap();
        paused_writer.await.unwrap().unwrap();
        assert_eq!(execution.await.unwrap().unwrap(), expected_response);
        drain.await.unwrap().unwrap();

        assert_eq!(next_calls.load(Ordering::SeqCst), 1);
        assert_eq!(replay.capability_calls.load(Ordering::SeqCst), 1);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        assert_eq!(delivery.health(), RecommendationDeliveryHealthV1::Closed);
        let connection = rusqlite::Connection::open(&database_path).unwrap();
        assert_eq!(sqlite_count(&connection, "decisions"), 1);
        assert_eq!(process_stop_count(&database_path, process_instance_id), 1);
        assert!(runtime.background.lock().unwrap().is_none());
        assert!(runtime.retention.lock().unwrap().is_none());
        assert!(runtime.writer.lock().unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recommend_lifecycle_drain_deadline_retains_pending_audit_until_abort_fences_writer() {
        let PausedRecommendDelivery {
            temporary,
            runtime,
            delivery,
            process_instance_id,
            release_writer,
            paused_writer,
            execution,
            expected_response,
            next_calls,
            replay,
        } = paused_recommend_delivery("recommend-drain-deadline-runtime-test").await;
        let database_path = temporary.path().join("ledger/router.db");

        let error = runtime
            .drain(Instant::now() + Duration::from_millis(25))
            .await
            .unwrap_err();
        let error_message = error.to_string();
        assert!(
            error_message.to_ascii_lowercase().contains("deadline"),
            "unexpected drain error: {error_message}"
        );
        assert_eq!(
            runtime.lifecycle.load(Ordering::Acquire),
            LIFECYCLE_DRAINING
        );
        assert_eq!(
            delivery.health(),
            RecommendationDeliveryHealthV1::DegradedPending { pending: 1 }
        );
        assert_eq!(process_stop_count(&database_path, process_instance_id), 0);
        assert!(runtime.background.lock().unwrap().is_some());
        assert!(runtime.retention.lock().unwrap().is_some());
        assert!(runtime.heartbeat.lock().unwrap().is_some());
        assert!(runtime.writer.lock().unwrap().is_some());

        runtime.abort();
        assert_eq!(runtime.lifecycle.load(Ordering::Acquire), LIFECYCLE_ABORTED);
        assert_eq!(delivery.health(), RecommendationDeliveryHealthV1::Closed);
        let _ = release_writer.send(());
        let _ = paused_writer.await.unwrap();
        assert_eq!(execution.await.unwrap().unwrap(), expected_response);
        assert_eq!(next_calls.load(Ordering::SeqCst), 1);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        let connection = rusqlite::Connection::open(&database_path).unwrap();
        assert_eq!(sqlite_count(&connection, "decisions"), 0);
        assert_eq!(process_stop_count(&database_path, process_instance_id), 0);
        assert!(runtime.retention.lock().unwrap().is_none());
        assert!(runtime.heartbeat.lock().unwrap().is_none());
        assert!(runtime.writer.lock().unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recommend_lifecycle_canceled_drain_keeps_pending_delivery_owned_until_abort() {
        let PausedRecommendDelivery {
            temporary,
            runtime,
            delivery,
            process_instance_id,
            release_writer,
            paused_writer,
            execution,
            expected_response,
            next_calls,
            replay,
        } = paused_recommend_delivery("recommend-canceled-drain-runtime-test").await;
        let database_path = temporary.path().join("ledger/router.db");

        let drain_runtime = runtime.clone();
        let drain = tokio::spawn(async move {
            drain_runtime
                .drain(Instant::now() + Duration::from_secs(5))
                .await
        });
        wait_for(|| {
            matches!(
                delivery.try_admit(),
                Err(RecommendationAdmissionErrorV1::Closed)
            )
        })
        .await;
        drain.abort();
        assert!(drain.await.unwrap_err().is_cancelled());

        assert_eq!(
            runtime.lifecycle.load(Ordering::Acquire),
            LIFECYCLE_DRAINING
        );
        assert_eq!(
            delivery.health(),
            RecommendationDeliveryHealthV1::DegradedPending { pending: 1 }
        );
        assert_eq!(process_stop_count(&database_path, process_instance_id), 0);
        assert!(runtime.background.lock().unwrap().is_some());
        assert!(runtime.retention.lock().unwrap().is_some());
        assert!(runtime.heartbeat.lock().unwrap().is_some());
        assert!(runtime.writer.lock().unwrap().is_some());

        runtime.abort();
        assert_eq!(delivery.health(), RecommendationDeliveryHealthV1::Closed);
        let _ = release_writer.send(());
        let _ = paused_writer.await.unwrap();
        assert_eq!(execution.await.unwrap().unwrap(), expected_response);
        assert_eq!(next_calls.load(Ordering::SeqCst), 1);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        let connection = rusqlite::Connection::open(&database_path).unwrap();
        assert_eq!(sqlite_count(&connection, "decisions"), 0);
        assert_eq!(process_stop_count(&database_path, process_instance_id), 0);
        assert!(runtime.retention.lock().unwrap().is_none());
        assert!(runtime.heartbeat.lock().unwrap().is_none());
        assert!(runtime.writer.lock().unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn timed_out_drain_retains_scheduler_ownership_until_abort_cancels_replay() {
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let family = LlmApiFamily::OpenAIChatCompletions;
        let mut config = config(family, 30);
        config.project_id = Some("drain-timeout-runtime-test".into());
        config.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        let activated = LedgerRepository::activate(&config).unwrap();
        let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
        let replay = PendingProductionReplay::new();
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();

        let anchor = context(family);
        runtime
            .execute(
                anchor.clone(),
                request(family),
                Some(replay_transport),
                next_returning(response(family), Arc::new(AtomicUsize::new(0))),
            )
            .await
            .unwrap();
        runtime.observe_event_fail_open(&llm_end(&anchor));
        let later = later_context(&anchor);
        assert!(runtime.register_primary_call(&later).is_some());
        runtime.observe_event_fail_open(&llm_end(&later));
        wait_for(|| replay.starts.load(Ordering::Acquire) == 1).await;

        let drain = runtime
            .drain(Instant::now() + Duration::from_millis(50))
            .await;
        assert!(drain.is_err());
        assert!(runtime.scheduler.lock().unwrap().is_some());
        assert!(runtime.retention.lock().unwrap().is_some());
        assert!(runtime.heartbeat.lock().unwrap().is_some());
        assert!(runtime.writer.lock().unwrap().is_some());
        assert_eq!(replay.cancellations.load(Ordering::Acquire), 0);

        runtime.abort();
        wait_for(|| replay.cancellations.load(Ordering::Acquire) == 1).await;
        assert!(runtime.scheduler.lock().unwrap().is_none());
        assert!(runtime.retention.lock().unwrap().is_none());
        assert!(runtime.heartbeat.lock().unwrap().is_none());
        assert!(runtime.writer.lock().unwrap().is_none());
        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_interrupts_all_resources_while_drain_retains_their_owners() {
        let (_temporary, runtime) =
            activated_lifecycle_runtime("retained-writer-abort-runtime-test");
        let writer = runtime.state.writer_client.clone().unwrap();
        let read_pool = runtime.state.read_pool.as_ref().unwrap().clone();
        let (read_started_tx, read_started_rx) = std::sync::mpsc::sync_channel(1);
        let read = tokio::spawn(async move {
            read_pool
                .run(
                    Instant::now() + Duration::from_secs(30),
                    move |connection| {
                        read_started_tx.send(()).unwrap();
                        connection
                            .query_row(
                                "WITH RECURSIVE counter(value) AS (
                                 VALUES(0) UNION ALL
                                 SELECT value + 1 FROM counter WHERE value < 1000000000
                             ) SELECT sum(value) FROM counter",
                                [],
                                |row| row.get::<_, i64>(0),
                            )
                            .map(|_| ())
                            .map_err(|_| {
                                crate::ledger::read_pool::ReadPoolError::operation_failed()
                            })
                    },
                )
                .await
        });
        read_started_rx
            .recv_timeout(Duration::from_secs(5))
            .unwrap();
        let retained = Arc::new(tokio::sync::Barrier::new(2));
        let release = Arc::new(tokio::sync::Barrier::new(2));
        *runtime.drain_retained_owners_pause.lock().unwrap() =
            Some((retained.clone(), release.clone()));

        let drain_runtime = runtime.clone();
        let drain = tokio::spawn(async move {
            drain_runtime
                .drain(Instant::now() + Duration::from_secs(30))
                .await
        });
        tokio::time::timeout(Duration::from_secs(10), retained.wait())
            .await
            .expect("drain did not retain all owners before its test deadline");
        assert!(runtime.retention.lock().unwrap().is_none());
        assert!(runtime.heartbeat.lock().unwrap().is_none());
        assert!(runtime.writer.lock().unwrap().is_none());

        runtime.abort();
        assert_eq!(runtime.lifecycle.load(Ordering::Acquire), LIFECYCLE_ABORTED);
        assert!(
            writer
                .flush_until(Instant::now() + Duration::from_secs(1))
                .await
                .is_err()
        );
        yield_until(|| {
            runtime
                .abort_handles
                .retention
                .as_ref()
                .is_none_or(AbortHandle::is_finished)
                && runtime
                    .abort_handles
                    .heartbeat
                    .as_ref()
                    .is_none_or(AbortHandle::is_finished)
        })
        .await;
        assert!(
            !drain.is_finished(),
            "abort must not join retained resource owners"
        );
        assert!(
            tokio::time::timeout(Duration::from_secs(5), read)
                .await
                .unwrap()
                .unwrap()
                .is_err()
        );

        release.wait().await;
        assert!(drain.await.unwrap().is_err());
        assert!(runtime.retention.lock().unwrap().is_none());
        assert!(runtime.heartbeat.lock().unwrap().is_none());
        assert!(runtime.writer.lock().unwrap().is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn immediate_abort_does_not_append_a_process_stop() {
        let (temporary, runtime) = activated_lifecycle_runtime("abort-without-stop-runtime-test");
        let process_instance_id = runtime.state.identity.process_instance_id;

        runtime.abort();

        let connection =
            rusqlite::Connection::open(temporary.path().join("ledger/router.db")).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM process_instance_state_events
                     WHERE process_instance_id = ?1 AND state = 'stopped'",
                    [process_instance_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn process_stop_follows_owned_work_and_writer_then_closes() {
        let (temporary, runtime) = activated_lifecycle_runtime("process-stop-order-runtime-test");
        let process_instance_id = runtime.state.identity.process_instance_id;
        let writer = runtime.state.writer_client.clone().unwrap();
        let retained = Arc::new(tokio::sync::Barrier::new(2));
        let release = Arc::new(tokio::sync::Barrier::new(2));
        *runtime.drain_retained_owners_pause.lock().unwrap() =
            Some((retained.clone(), release.clone()));

        let drain_runtime = runtime.clone();
        let drain = tokio::spawn(async move {
            drain_runtime
                .drain(Instant::now() + Duration::from_secs(5))
                .await
        });
        tokio::time::timeout(Duration::from_secs(2), retained.wait())
            .await
            .expect("drain did not retain workers before ProcessStop");
        assert!(runtime.live_embedding_service().unwrap().is_closed());
        assert!(runtime.background.lock().unwrap().is_none());
        let database_path = temporary.path().join("ledger/router.db");
        assert_eq!(
            rusqlite::Connection::open(&database_path)
                .unwrap()
                .query_row(
                    "SELECT count(*) FROM process_instance_state_events
                     WHERE process_instance_id = ?1 AND state = 'stopped'",
                    [process_instance_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );

        release.wait().await;
        drain.await.unwrap().unwrap();
        assert_eq!(
            rusqlite::Connection::open(database_path)
                .unwrap()
                .query_row(
                    "SELECT count(*) FROM process_instance_state_events
                     WHERE process_instance_id = ?1 AND state = 'stopped'",
                    [process_instance_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        assert!(
            writer
                .run_retention_until(
                    new_retention_request(Utc::now().timestamp_millis()).unwrap(),
                    Instant::now() + Duration::from_secs(1),
                )
                .await
                .is_err()
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn canceled_drain_restores_all_retained_owners_until_abort() {
        let (_temporary, runtime) = activated_lifecycle_runtime("canceled-drain-runtime-test");
        let live_embedding = runtime.live_embedding_service().unwrap();
        let retained = Arc::new(tokio::sync::Barrier::new(2));
        let release = Arc::new(tokio::sync::Barrier::new(2));
        *runtime.drain_retained_owners_pause.lock().unwrap() = Some((retained.clone(), release));

        let drain_runtime = runtime.clone();
        let drain = tokio::spawn(async move {
            drain_runtime
                .drain(Instant::now() + Duration::from_secs(30))
                .await
        });
        tokio::time::timeout(Duration::from_secs(10), retained.wait())
            .await
            .expect("drain did not retain all owners before cancellation");
        assert!(runtime.retention.lock().unwrap().is_none());
        assert!(runtime.heartbeat.lock().unwrap().is_none());
        assert!(runtime.writer.lock().unwrap().is_none());

        drain.abort();
        assert!(drain.await.unwrap_err().is_cancelled());
        assert!(runtime.retention.lock().unwrap().is_some());
        assert!(runtime.heartbeat.lock().unwrap().is_some());
        assert!(runtime.writer.lock().unwrap().is_some());
        assert!(runtime.live_embedding_service().is_some());
        assert!(live_embedding.is_closed());
        assert_eq!(
            runtime.lifecycle.load(Ordering::Acquire),
            LIFECYCLE_DRAINING
        );

        runtime.abort();
        assert!(runtime.retention.lock().unwrap().is_none());
        assert!(runtime.heartbeat.lock().unwrap().is_none());
        assert!(runtime.writer.lock().unwrap().is_none());
        assert_eq!(runtime.lifecycle.load(Ordering::Acquire), LIFECYCLE_ABORTED);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn activated_runtime_persists_and_executes_complete_shadow_judge_batch() {
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let family = LlmApiFamily::OpenAIChatCompletions;
        let mut config = config(family, 30);
        config.project_id = Some("production-runtime-test".into());
        config.database_path = temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned();
        let database_path = config.database_path.clone();
        let activated = LedgerRepository::activate(&config).unwrap();

        let mut candidate_response = response(family);
        candidate_response["id"] = json!("chatcmpl-candidate-runtime");
        candidate_response["object"] = json!("chat.completion");
        candidate_response["created"] = json!(1);
        candidate_response["model"] = json!("candidate-model");
        let judge_text = json!({
            "response_equivalence": 0.95,
            "trajectory_equivalence": 0.9,
            "judge_confidence": 0.98,
            "hard_failures": [],
            "rationale": "The candidate preserves the response and trajectory behavior."
        })
        .to_string();
        let mut judge_response = response(family);
        judge_response["id"] = json!("chatcmpl-judge-runtime");
        judge_response["object"] = json!("chat.completion");
        judge_response["created"] = json!(1);
        judge_response["model"] = json!("judge-model");
        judge_response["choices"][0]["message"]["content"] = json!(judge_text);
        let replay = ProductionReplay::new([
            ("candidate-model".to_string(), vec![candidate_response]),
            ("judge-model".to_string(), vec![judge_response]),
        ]);
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
        let adaptive_agent_id = "task9-adaptive-agent";
        let mut adaptive = AdaptiveRuntime::new(AdaptiveConfig {
            agent_id: Some(adaptive_agent_id.to_string()),
            state: Some(StateConfig {
                backend: BackendSpec::in_memory(),
            }),
            telemetry: Some(TelemetryComponentConfig {
                subscriber_name: Some("task9-adaptive-consumer".to_string()),
                learners: vec!["acg".to_string()],
            }),
            acg: Some(AcgComponentConfig::default()),
            ..AdaptiveConfig::default()
        })
        .await
        .unwrap();
        adaptive.register().await.unwrap();
        let atif = AtifExporter::new(
            "task9-production-runtime".to_string(),
            AtifAgentInfo {
                name: "task9-production-runtime".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                model_name: None,
                tool_definitions: None,
                extra: None,
            },
        );
        let raw_events = Arc::new(Mutex::new(Vec::<Event>::new()));
        let event_sink = raw_events.clone();
        let mut registrations = PluginRegistrationContext::with_namespace("task9-runtime:");
        registrations
            .register_subscriber("router", runtime.subscriber_callback())
            .unwrap();
        registrations
            .register_subscriber("atif", atif.subscriber())
            .unwrap();
        registrations
            .register_subscriber(
                "raw-events",
                Arc::new(move |event| {
                    event_sink
                        .lock()
                        .unwrap_or_else(|error| error.into_inner())
                        .push(event.clone());
                }),
            )
            .unwrap();

        let anchor = context(family);
        let anchor_response = response(family);
        let downstream_calls = Arc::new(AtomicUsize::new(0));
        let returned = runtime
            .execute(
                anchor.clone(),
                request(family),
                Some(replay_transport),
                next_returning(anchor_response.clone(), downstream_calls.clone()),
            )
            .await
            .unwrap();
        assert_eq!(returned, anchor_response);
        assert_eq!(downstream_calls.load(Ordering::SeqCst), 1);
        runtime.observe_event_fail_open(&llm_end(&anchor));
        let later = later_context(&anchor);
        assert!(runtime.register_primary_call(&later).is_some());
        runtime.observe_event_fail_open(&llm_end(&later));

        let connection = rusqlite::Connection::open(database_path).unwrap();
        let converged = tokio::time::timeout(Duration::from_secs(5), async {
            while sqlite_count(&connection, "evaluations") != 1
                || sqlite_count(&connection, "shadow_results") != 1
            {
                tokio::task::yield_now().await;
            }
        })
        .await;
        assert!(
            converged.is_ok(),
            "production runtime did not converge: starts={:?}, pending={}, terminals={}, batches={}, shadows={}, judges={}, evaluations={}, vectors={}, health={:?}",
            replay.started_models(),
            sqlite_count(&connection, "anchors"),
            sqlite_count(&connection, "anchor_state_events"),
            sqlite_count(&connection, "sample_batches"),
            sqlite_count(&connection, "shadow_attempts"),
            sqlite_count(&connection, "judge_attempts"),
            sqlite_count(&connection, "evaluations"),
            sqlite_count(&connection, "shadow_results"),
            runtime.state.health.snapshot(),
        );
        assert_eq!(
            replay.started_models(),
            vec!["candidate-model".to_string(), "judge-model".to_string()]
        );
        assert_eq!(sqlite_count(&connection, "anchors"), 1);
        assert_eq!(sqlite_count(&connection, "sample_batches"), 1);
        assert_eq!(sqlite_count(&connection, "shadow_attempts"), 1);
        assert_eq!(sqlite_count(&connection, "judge_attempts"), 1);
        assert_eq!(sqlite_count(&connection, "evaluations"), 1);
        assert_eq!(sqlite_count(&connection, "shadow_results"), 1);
        flush_subscribers().unwrap();
        adaptive.wait_for_idle();
        let scheduler_events = raw_events
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .clone();
        let anchor_uuid = anchor.call_uuid.to_string();
        let evaluator_uuids = scheduler_events
            .iter()
            .filter(|event| {
                event.scope_type() == Some(ScopeType::Evaluator)
                    && event.metadata().is_some_and(|metadata| {
                        metadata.get("anchor_uuid").and_then(Json::as_str)
                            == Some(anchor_uuid.as_str())
                    })
            })
            .map(Event::uuid)
            .collect::<BTreeSet<_>>();
        assert_eq!(evaluator_uuids.len(), 2);
        for evaluator_uuid in &evaluator_uuids {
            let evaluator_events = scheduler_events
                .iter()
                .filter(|event| {
                    event.uuid() == *evaluator_uuid
                        && event.scope_type() == Some(ScopeType::Evaluator)
                })
                .collect::<Vec<_>>();
            assert_eq!(evaluator_events.len(), 2);
            assert_eq!(
                evaluator_events
                    .iter()
                    .filter(|event| event.scope_category() == Some(ScopeCategory::Start))
                    .count(),
                1
            );
            assert_eq!(
                evaluator_events
                    .iter()
                    .filter(|event| event.scope_category() == Some(ScopeCategory::End))
                    .count(),
                1
            );
        }
        let internal_llm = scheduler_events
            .iter()
            .filter(|event| {
                event
                    .parent_uuid()
                    .is_some_and(|parent| evaluator_uuids.contains(&parent))
                    && matches!(
                        event.llm_call_role(),
                        Some(LlmCallRole::Shadow | LlmCallRole::Judge)
                    )
            })
            .collect::<Vec<_>>();
        assert_eq!(internal_llm.len(), 4);
        assert!(internal_llm.iter().all(|event| {
            event
                .parent_uuid()
                .is_some_and(|parent| evaluator_uuids.contains(&parent))
                && event.parent_uuid() != Some(anchor.call_uuid)
                && event.metadata().is_some_and(|metadata| {
                    metadata.get("anchor_uuid").and_then(Json::as_str) == Some(anchor_uuid.as_str())
                })
        }));
        for expected_role in [LlmCallRole::Shadow, LlmCallRole::Judge] {
            let role_events = internal_llm
                .iter()
                .copied()
                .filter(|event| event.llm_call_role() == Some(expected_role))
                .collect::<Vec<_>>();
            assert_eq!(role_events.len(), 2);
            assert_eq!(
                role_events
                    .iter()
                    .map(|event| event.uuid())
                    .collect::<BTreeSet<_>>()
                    .len(),
                1
            );
            assert_eq!(
                role_events
                    .iter()
                    .filter(|event| event.scope_category() == Some(ScopeCategory::Start))
                    .count(),
                1
            );
            assert_eq!(
                role_events
                    .iter()
                    .filter(|event| event.scope_category() == Some(ScopeCategory::End))
                    .count(),
                1
            );
            assert_eq!(
                role_events
                    .iter()
                    .filter_map(|event| event.parent_uuid())
                    .collect::<BTreeSet<_>>()
                    .len(),
                1
            );
        }
        for evaluator_uuid in &evaluator_uuids {
            assert_eq!(
                internal_llm
                    .iter()
                    .filter(|event| event.parent_uuid() == Some(*evaluator_uuid))
                    .count(),
                2
            );
        }
        assert_eq!(runtime.state.loss.current_ingest_seq(), 2);

        let internal_default_atif = atif.export().unwrap();
        assert!(internal_default_atif.steps.is_empty());
        let internal_diagnostic_atif = atif
            .export_with_options(AtifExportOptions {
                include_non_primary_llm_calls: true,
            })
            .unwrap();
        assert_eq!(internal_diagnostic_atif.steps.len(), 4);
        for expected_name in [SHADOW_CALL_NAME, JUDGE_CALL_NAME] {
            assert_eq!(
                internal_diagnostic_atif
                    .steps
                    .iter()
                    .filter(|step| {
                        step.extra
                            .as_ref()
                            .and_then(|extra| extra.pointer("/ancestry/function_name"))
                            .and_then(Json::as_str)
                            == Some(expected_name)
                    })
                    .count(),
                2
            );
        }
        let shadow_request = internal_llm
            .iter()
            .find(|event| {
                event.llm_call_role() == Some(LlmCallRole::Shadow)
                    && event.scope_category() == Some(ScopeCategory::Start)
            })
            .and_then(|event| event.annotated_request())
            .expect("production Shadow start event must carry its annotated request");
        let internal_facts = adaptive
            .build_cache_request_facts(adaptive_agent_id, "passthrough", shadow_request)
            .expect("Adaptive cache diagnostics should accept the emitted Shadow request");
        assert_eq!(
            internal_facts.missing_facts,
            vec!["acg_stability_unavailable".to_string()]
        );

        let agent = push_scope(
            PushScopeParams::builder()
                .name(adaptive_agent_id)
                .scope_type(ScopeType::Agent)
                .build(),
        )
        .unwrap();
        let mut primary_request = request(family);
        primary_request.content["model"] = json!("primary-sibling-model");
        let primary_annotated = OpenAIChatCodec.decode(&primary_request).unwrap();
        let mut primary_response = response(family);
        primary_response["model"] = json!("primary-sibling-model");
        let expected_primary_response = primary_response.clone();
        let request_codec: Arc<dyn LlmCodec> = Arc::new(OpenAIChatCodec);
        let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);
        let returned_primary = llm_call_execute_v2(
            LlmCallExecuteV2Params::builder()
                .name("task9-primary-sibling")
                .request(primary_request)
                .func(next_returning(
                    primary_response,
                    Arc::new(AtomicUsize::new(0)),
                ))
                .api_family(family)
                .call_role(LlmCallRole::Primary)
                .sanitized_metadata(BTreeMap::new())
                .parent(agent.clone())
                .codec(request_codec)
                .response_codec(response_codec)
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(returned_primary, expected_primary_response);
        pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
        flush_subscribers().unwrap();
        adaptive.wait_for_idle();
        let primary_sibling_events = raw_events
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
            .filter(|event| event.name() == "task9-primary-sibling")
            .cloned()
            .collect::<Vec<_>>();
        assert_eq!(primary_sibling_events.len(), 2);
        assert_eq!(
            primary_sibling_events
                .iter()
                .map(Event::uuid)
                .collect::<BTreeSet<_>>()
                .len(),
            1
        );
        assert_eq!(
            primary_sibling_events
                .iter()
                .filter(|event| event.scope_category() == Some(ScopeCategory::Start))
                .count(),
            1
        );
        assert_eq!(
            primary_sibling_events
                .iter()
                .filter(|event| event.scope_category() == Some(ScopeCategory::End))
                .count(),
            1
        );
        assert!(primary_sibling_events.iter().all(|event| {
            event.llm_call_role() == Some(LlmCallRole::Primary)
                && event.parent_uuid() == Some(agent.uuid)
        }));

        let primary_facts = adaptive
            .build_cache_request_facts(adaptive_agent_id, "passthrough", &primary_annotated)
            .expect("Adaptive cache diagnostics should accept the emitted Primary request");
        assert!(
            !primary_facts
                .missing_facts
                .contains(&"acg_stability_unavailable".to_string())
        );
        let default_atif = atif.export().unwrap();
        assert_eq!(
            default_atif
                .steps
                .iter()
                .map(|step| step.source.as_str())
                .collect::<Vec<_>>(),
            vec!["user", "agent"]
        );
        assert!(default_atif.steps.iter().all(|step| {
            step.extra
                .as_ref()
                .and_then(|extra| extra.pointer("/ancestry/function_name"))
                .and_then(Json::as_str)
                == Some("task9-primary-sibling")
        }));
        let diagnostic_atif = atif
            .export_with_options(AtifExportOptions {
                include_non_primary_llm_calls: true,
            })
            .unwrap();
        assert_eq!(diagnostic_atif.steps.len(), 6);
        for expected_name in [SHADOW_CALL_NAME, JUDGE_CALL_NAME, "task9-primary-sibling"] {
            assert_eq!(
                diagnostic_atif
                    .steps
                    .iter()
                    .filter(|step| {
                        step.extra
                            .as_ref()
                            .and_then(|extra| extra.pointer("/ancestry/function_name"))
                            .and_then(Json::as_str)
                            == Some(expected_name)
                    })
                    .count(),
                2
            );
        }

        runtime
            .drain(Instant::now() + Duration::from_secs(5))
            .await
            .unwrap();
        assert!(runtime.scheduler.lock().unwrap().is_none());
        assert!(runtime.heartbeat.lock().unwrap().is_none());
        assert!(runtime.writer.lock().unwrap().is_none());
        let mut registrations = registrations.into_registrations();
        rollback_registrations(&mut registrations);
        assert!(registrations.is_empty());
        adaptive.shutdown().await.unwrap();
        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());
    }

    #[tokio::test]
    async fn completed_driver_tasks_stay_bounded_under_sustained_windows() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let (runtime, _, delivery, _, replay) = runtime(family, 30);
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        for expected in 1..=24 {
            let anchor = context(family);
            runtime
                .execute(
                    anchor.clone(),
                    request(family),
                    Some(replay_transport.clone()),
                    next_returning(response(family), Arc::new(AtomicUsize::new(0))),
                )
                .await
                .unwrap();
            runtime.observe_event_fail_open(&llm_end(&anchor));
            let later = later_context(&anchor);
            assert!(runtime.register_primary_call(&later).is_some());
            runtime.observe_event_fail_open(&llm_end(&later));
            wait_for(|| delivery.delivered_anchor_ids().len() == expected).await;
        }
        wait_for(|| runtime.state.owned_task_count.load(Ordering::Acquire) == 0).await;
        assert!(runtime.state.peak_owned_task_count.load(Ordering::Acquire) <= 8);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn concurrent_identical_name_managed_v2_calls_correlate_by_uuid() {
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());

        let family = LlmApiFamily::OpenAIChatCompletions;
        let (runtime, sink, delivery, _, replay) = runtime(family, 30);
        let mut registrations = PluginRegistrationContext::with_namespace("router-runtime-core:");
        registrations
            .register_subscriber("events", runtime.subscriber_callback())
            .unwrap();
        registrations
            .register_llm_execution_intercept_v2("execution", 0, runtime.execution_callback())
            .unwrap();
        let agent = push_scope(
            PushScopeParams::builder()
                .name("runtime-core-concurrent-agent")
                .scope_type(ScopeType::Agent)
                .build(),
        )
        .unwrap();
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let replay_factory: Arc<dyn LlmReplayFactory> = Arc::new(ReplayFactory {
            replay: replay_transport,
        });
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let mut first_response = response(family);
        first_response["id"] = json!("chatcmpl-concurrent-first");
        let mut second_response = response(family);
        second_response["id"] = json!("chatcmpl-concurrent-second");

        let first = llm_call_execute_v2(
            LlmCallExecuteV2Params::builder()
                .name("identical-name")
                .request(request(family))
                .func(next_yielding(
                    first_response.clone(),
                    provider_calls.clone(),
                ))
                .api_family(family)
                .call_role(LlmCallRole::Primary)
                .sanitized_metadata(BTreeMap::new())
                .parent(agent.clone())
                .replay_factory(replay_factory.clone())
                .build(),
        );
        let second = llm_call_execute_v2(
            LlmCallExecuteV2Params::builder()
                .name("identical-name")
                .request(request(family))
                .func(next_yielding(
                    second_response.clone(),
                    provider_calls.clone(),
                ))
                .api_family(family)
                .call_role(LlmCallRole::Primary)
                .sanitized_metadata(BTreeMap::new())
                .parent(agent.clone())
                .replay_factory(replay_factory)
                .build(),
        );
        let (first_result, second_result) = tokio::join!(first, second);
        assert_eq!(first_result.unwrap(), first_response);
        assert_eq!(second_result.unwrap(), second_response);
        flush_subscribers().unwrap();
        wait_for(|| sink.unresolved_pending().len() + sink.terminal_payloads().len() == 2).await;

        let pending = sink.unresolved_pending();
        let initial_terminals = sink.terminal_payloads();
        assert_eq!(pending.len(), 1);
        assert_eq!(initial_terminals.len(), 1);
        wait_for(|| delivery.delivered_anchor_ids().len() == 1).await;
        assert_eq!(
            initial_terminals[0].state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached
            }
        );
        let mut call_uuids = vec![
            pending[0].payload.anchor_call_uuid,
            initial_terminals[0].pending.anchor_call_uuid,
        ];
        call_uuids.sort_unstable();
        call_uuids.dedup();
        assert_eq!(call_uuids.len(), 2);
        assert_eq!(
            pending[0].payload.anchor_id.get_version(),
            Some(uuid::Version::SortRand)
        );
        assert_eq!(
            initial_terminals[0].pending.anchor_id.get_version(),
            Some(uuid::Version::SortRand)
        );

        let later_response = response(family);
        let later_result = llm_call_execute_v2(
            LlmCallExecuteV2Params::builder()
                .name("identical-name")
                .request(request(family))
                .func(next_returning(
                    later_response.clone(),
                    provider_calls.clone(),
                ))
                .api_family(family)
                .call_role(LlmCallRole::Primary)
                .sanitized_metadata(BTreeMap::new())
                .parent(agent.clone())
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(later_result, later_response);
        flush_subscribers().unwrap();
        wait_for(|| delivery.delivered_anchor_ids().len() == 2).await;
        assert_eq!(provider_calls.load(Ordering::SeqCst), 3);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        assert!(sink.terminal_payloads().iter().all(|terminal| {
            terminal.state
                == TrajectoryTerminalStateV1::Closed {
                    trigger: TrajectoryTrigger::ProgressReached,
                }
        }));

        runtime.stop_intake();
        flush_subscribers().unwrap();
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        let mut registrations = registrations.into_registrations();
        rollback_registrations(&mut registrations);
        assert!(registrations.is_empty());
        pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn real_core_deadline_barrier_flushes_queued_progress_before_deadline() {
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());
        flush_subscribers().unwrap();

        let family = LlmApiFamily::OpenAIChatCompletions;
        let sink = Arc::new(InMemoryTrajectorySink::new(32));
        let delivery = Arc::new(InMemoryTrajectoryDelivery::new(32));
        let replay = Arc::new(Replay::new(family));
        let barrier_calls = Arc::new(AtomicUsize::new(0));
        let runtime = RouterRuntime::start_with_dependencies(
            config(family, 1),
            Arc::new(FixedSampler::new(0.0)),
            sink.clone(),
            delivery.clone(),
            Arc::new(CountingCoreBarrier {
                calls: barrier_calls.clone(),
            }),
        )
        .unwrap();
        let mut registrations = PluginRegistrationContext::with_namespace("router-runtime-core:");
        registrations
            .register_subscriber("events", runtime.subscriber_callback())
            .unwrap();
        registrations
            .register_llm_execution_intercept_v2("execution", 0, runtime.execution_callback())
            .unwrap();
        let agent = push_scope(
            PushScopeParams::builder()
                .name("runtime-core-deadline-agent")
                .scope_type(ScopeType::Agent)
                .build(),
        )
        .unwrap();
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let replay_factory: Arc<dyn LlmReplayFactory> = Arc::new(ReplayFactory {
            replay: replay_transport,
        });
        let anchor_response = response(family);
        let anchor_result = llm_call_execute_v2(
            LlmCallExecuteV2Params::builder()
                .name("deadline-progress")
                .request(request(family))
                .func(next_returning(
                    anchor_response.clone(),
                    Arc::new(AtomicUsize::new(0)),
                ))
                .api_family(family)
                .call_role(LlmCallRole::Primary)
                .sanitized_metadata(BTreeMap::new())
                .parent(agent.clone())
                .replay_factory(replay_factory)
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(anchor_result, anchor_response);
        flush_subscribers().unwrap();
        yield_until_named("Core anchor pending acknowledgement", || {
            sink.unresolved_pending().len() == 1
        })
        .await;

        let gate = Arc::new(DispatcherGate::new());
        let release_gate = DispatcherGateRelease(gate.clone());
        registrations
            .register_subscriber(
                "deadline-gate",
                Arc::new({
                    let gate = gate.clone();
                    move |event| {
                        if event.name() == "router.runtime.deadline.gate" {
                            gate.block();
                        }
                    }
                }),
            )
            .unwrap();
        emit_scope_event(
            EmitMarkEventParams::builder()
                .name("router.runtime.deadline.gate")
                .build(),
        )
        .unwrap();
        yield_until_named("dispatcher gate entry", || {
            gate.entered.load(Ordering::Acquire)
        })
        .await;

        let later_response = response(family);
        let later_result = llm_call_execute_v2(
            LlmCallExecuteV2Params::builder()
                .name("deadline-progress")
                .request(request(family))
                .func(next_returning(
                    later_response.clone(),
                    Arc::new(AtomicUsize::new(0)),
                ))
                .api_family(family)
                .call_role(LlmCallRole::Primary)
                .sanitized_metadata(BTreeMap::new())
                .parent(agent.clone())
                .build(),
        )
        .await;
        tokio::time::advance(Duration::from_secs(1)).await;
        yield_until_named("deadline barrier invocation", || {
            barrier_calls.load(Ordering::SeqCst) == 1
        })
        .await;
        let terminal_before_release = sink.terminal_payloads();
        let delivery_before_release = delivery.delivered_anchor_ids();
        release_gate.release();
        flush_subscribers().unwrap();

        assert_eq!(later_result.unwrap(), later_response);
        assert!(terminal_before_release.is_empty());
        assert!(delivery_before_release.is_empty());
        yield_until_named("post-barrier trajectory delivery", || {
            delivery.delivered_anchor_ids().len() == 1
        })
        .await;
        let terminals = sink.terminal_payloads();
        assert_eq!(terminals.len(), 1);
        assert_eq!(
            terminals[0].state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached
            }
        );
        assert_eq!(barrier_calls.load(Ordering::SeqCst), 1);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);

        runtime.stop_intake();
        flush_subscribers().unwrap();
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        let mut registrations = registrations.into_registrations();
        rollback_registrations(&mut registrations);
        assert!(registrations.is_empty());
        pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn real_core_dispatcher_orders_anchor_registration_before_end_event() {
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());

        let family = LlmApiFamily::OpenAIChatCompletions;
        let (runtime, sink, delivery, _, replay) = runtime(family, 30);
        let mut registrations = PluginRegistrationContext::with_namespace("router-runtime-core:");
        registrations
            .register_subscriber("events", runtime.subscriber_callback())
            .unwrap();
        registrations
            .register_llm_execution_intercept_v2("execution", 0, runtime.execution_callback())
            .unwrap();
        let agent = push_scope(
            PushScopeParams::builder()
                .name("runtime-core-agent")
                .scope_type(ScopeType::Agent)
                .build(),
        )
        .unwrap();
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let replay_factory: Arc<dyn LlmReplayFactory> = Arc::new(ReplayFactory {
            replay: replay_transport,
        });
        let provider_calls = Arc::new(AtomicUsize::new(0));
        let raw_response = response(family);

        let anchor_result = llm_call_execute_v2(
            LlmCallExecuteV2Params::builder()
                .name("same-name")
                .request(request(family))
                .func(next_returning(raw_response.clone(), provider_calls.clone()))
                .api_family(family)
                .call_role(LlmCallRole::Primary)
                .sanitized_metadata(BTreeMap::new())
                .parent(agent.clone())
                .replay_factory(replay_factory)
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(anchor_result, raw_response);
        assert!(delivery.delivered_anchor_ids().is_empty());

        let later_response = response(family);
        let later_result = llm_call_execute_v2(
            LlmCallExecuteV2Params::builder()
                .name("same-name")
                .request(request(family))
                .func(next_returning(
                    later_response.clone(),
                    provider_calls.clone(),
                ))
                .api_family(family)
                .call_role(LlmCallRole::Primary)
                .sanitized_metadata(BTreeMap::new())
                .parent(agent.clone())
                .build(),
        )
        .await
        .unwrap();
        assert_eq!(later_result, later_response);
        flush_subscribers().unwrap();
        wait_for(|| delivery.delivered_anchor_ids().len() == 1).await;
        assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
        assert_eq!(sink.terminal_payloads().len(), 1);

        runtime.stop_intake();
        flush_subscribers().unwrap();
        runtime
            .drain(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
        let mut registrations = registrations.into_registrations();
        rollback_registrations(&mut registrations);
        assert!(registrations.is_empty());
        pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
        *global_context().write().unwrap() = NemoRelayContextState::new();
        set_thread_scope_stack(create_scope_stack());
    }
}
