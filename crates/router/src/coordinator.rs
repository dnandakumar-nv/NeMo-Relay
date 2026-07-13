// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure, serialized state machine for future-local trajectory windows.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;

use chrono::{DateTime, Utc};
use nemo_relay::api::event::ScopeCategory;
use nemo_relay::api::llm::{LlmApiFamily, LlmCallRole};
use nemo_relay::api::scope::ScopeType;
use tokio::sync::oneshot;
use tokio::time::Instant;
use uuid::Uuid;

use crate::active_runtime::{
    ActiveDeadlineObservationV2, ActiveOutcomeObservationV2, ActiveOutcomeOversizedV2,
    ActiveOutcomeRuntimeErrorV2, ActiveOutcomeStageV2, ActiveRepresentativeObservationV2,
};
use crate::sink::{DeliveryAck, DurableDeclineReason, SinkAck, TransientRefusalReason};
use crate::trajectory::{
    CapturedTrajectoryEvent, ClosedTrajectoryWindow, PendingTrajectoryWindow,
    PersistedTrajectoryTerminalV1, ProjectedTrajectoryEvent, TrajectoryOwnerScopeV1,
    TrajectoryRejectionReason, TrajectoryTrigger, TrajectoryWindowSeed,
    truncate_utc_to_milliseconds,
};

const MIN_COMMAND_CAPACITY: usize = 256;
const MAX_COMMAND_CAPACITY: usize = 65_536;
const STAGED_EVENTS_PER_CALL: usize = 16;

pub(crate) const HEALTH_CLASSIFICATION_LOSS: &str = "router.coordinator.classification_loss";
pub(crate) const HEALTH_DELIVERY_FAILURE: &str = "router.coordinator.delivery_failure";
pub(crate) const HEALTH_EVIDENCE_CAPACITY: &str = "router.coordinator.evidence_capacity";
pub(crate) const HEALTH_INTAKE_CLOSED: &str = "router.coordinator.intake_closed";
pub(crate) const HEALTH_OWNERSHIP_LOSS: &str = "router.coordinator.ownership_loss";
pub(crate) const HEALTH_POOL_FULL: &str = "router.coordinator.pool_full";
pub(crate) const HEALTH_SINK_CONFLICT: &str = "router.coordinator.sink_conflict";
pub(crate) const HEALTH_SINK_FAILURE: &str = "router.coordinator.sink_failure";
pub(crate) const HEALTH_SINK_MISMATCH: &str = "router.coordinator.sink_mismatch";
pub(crate) const HEALTH_SCHEDULER_PRESSURE: &str = "router.coordinator.scheduler_pressure";
pub(crate) const HEALTH_STAGING_PRESSURE: &str = "router.coordinator.staging_pressure";

/// Deterministic actor and index bounds derived from validated pool limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CoordinatorLimits {
    pub(crate) command_capacity: usize,
    pub(crate) primary_call_capacity: usize,
    pub(crate) ancestry_capacity: usize,
    pub(crate) staging_capacity: usize,
    pub(crate) tombstone_capacity: usize,
}

impl CoordinatorLimits {
    pub(crate) fn from_total_max_pending(total_max_pending: usize) -> Self {
        let command_capacity = total_max_pending
            .saturating_mul(4)
            .saturating_add(MIN_COMMAND_CAPACITY)
            .clamp(MIN_COMMAND_CAPACITY, MAX_COMMAND_CAPACITY);
        Self {
            command_capacity,
            primary_call_capacity: command_capacity.saturating_mul(2),
            ancestry_capacity: command_capacity.saturating_mul(4),
            staging_capacity: command_capacity,
            tombstone_capacity: command_capacity.saturating_mul(4),
        }
    }
}

/// Loss state sampled by the producer immediately before enqueueing a command.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct LossSnapshot {
    pub(crate) highest_dropped_ingest_seq: u64,
    pub(crate) classification_loss_epoch: u64,
}

/// Immutable V2 Primary identity registered before the managed continuation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PrimaryCallRegistration {
    pub(crate) call_uuid: Uuid,
    pub(crate) root_uuid: Uuid,
    pub(crate) parent_uuid: Uuid,
    pub(crate) owner_uuid: Uuid,
    pub(crate) owner_path: Vec<TrajectoryOwnerScopeV1>,
    pub(crate) api_family: LlmApiFamily,
}

/// Per-window semantic and memory bounds frozen at sample admission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WindowLimits {
    pub(crate) requested_progress: usize,
    pub(crate) max_events: usize,
    pub(crate) max_bytes: usize,
    pub(crate) handoff_progress: bool,
    pub(crate) compaction_progress: bool,
}

impl WindowLimits {
    pub(crate) fn new(
        requested_progress: usize,
        max_events: usize,
        max_bytes: usize,
        lifecycle_presets: &[String],
    ) -> Self {
        Self {
            requested_progress,
            max_events,
            max_bytes,
            handoff_progress: lifecycle_presets.iter().any(|value| value == "handoff"),
            compaction_progress: lifecycle_presets.iter().any(|value| value == "compaction"),
        }
    }
}

/// A fully projected sampled anchor proposed after its successful response.
pub(crate) struct AnchorRegistration {
    pub(crate) seed: TrajectoryWindowSeed,
    pub(crate) proposal_permit: Option<tokio::sync::OwnedSemaphorePermit>,
    pub(crate) monotonic_deadline: Instant,
    pub(crate) opened_after_ingest_seq: u64,
    pub(crate) loss: LossSnapshot,
    pub(crate) admission_epoch: u64,
    pub(crate) limits: WindowLimits,
}

/// Commands are the only way to mutate coordinator-owned state.
pub(crate) enum CoordinatorCommand {
    RegisterPrimaryCall {
        registration: PrimaryCallRegistration,
        loss: LossSnapshot,
        admission_epoch: u64,
    },
    RegistrationLoss {
        loss: LossSnapshot,
        reason: &'static str,
    },
    RegisterAnchor(Box<AnchorRegistration>),
    StageActiveOutcome {
        call_uuid: Uuid,
        admission_epoch: u64,
        stage: ActiveOutcomeStageV2,
        reply: oneshot::Sender<Result<(), ActiveOutcomeRuntimeErrorV2>>,
    },
    ActiveRepresentativeBarrier {
        observation: ActiveRepresentativeObservationV2,
        reply: oneshot::Sender<Result<(), ActiveOutcomeRuntimeErrorV2>>,
    },
    ActiveDeadlineBarrier(ActiveDeadlineObservationV2),
    ShutdownActiveOutcomes {
        observed_at_unix_ms: u64,
        reply: oneshot::Sender<Result<(), ActiveOutcomeRuntimeErrorV2>>,
    },
    PendingRecorded(SinkAck),
    ObservedEvent {
        event: Arc<CapturedTrajectoryEvent>,
        loss: LossSnapshot,
    },
    OversizedEvent {
        event: ProjectedTrajectoryEvent,
        loss: LossSnapshot,
    },
    DeadlineElapsed {
        anchor_id: Uuid,
        generation: u64,
        loss: LossSnapshot,
    },
    DeadlineBarrierFailed {
        anchor_id: Uuid,
        generation: u64,
        loss: LossSnapshot,
    },
    TerminalRecorded(SinkAck),
    WindowDelivered(DeliveryAck),
    DeliveryRetryElapsed {
        anchor_id: Uuid,
        generation: u64,
    },
    SinkRetryElapsed {
        anchor_id: Uuid,
        generation: u64,
    },
    SchedulerCapacityRecovered {
        pool_id: String,
        pressure_generation: u64,
    },
    EvidenceCapacityRecovered {
        pressure_generation: u64,
    },
    Health(&'static str),
    PermanentFault(&'static str),
    Shutdown {
        admission_epoch: u64,
        loss: LossSnapshot,
    },
}

/// Pure requests returned by the actor for runtime-owned adapters to execute.
pub(crate) enum CoordinatorEffect {
    RecordPending {
        anchor_id: Uuid,
        payload_hash: String,
        payload: PendingTrajectoryWindow,
    },
    StageActiveOutcome {
        stage: ActiveOutcomeStageV2,
        reply: oneshot::Sender<Result<(), ActiveOutcomeRuntimeErrorV2>>,
    },
    ObserveActiveOutcome(ActiveOutcomeObservationV2),
    ObserveActiveOutcomeOversized(ActiveOutcomeOversizedV2),
    ActiveRepresentativeBarrier {
        observation: ActiveRepresentativeObservationV2,
        reply: oneshot::Sender<Result<(), ActiveOutcomeRuntimeErrorV2>>,
    },
    ActiveDeadlineBarrier(ActiveDeadlineObservationV2),
    ShutdownActiveOutcomes {
        observed_at_unix_ms: u64,
        reply: oneshot::Sender<Result<(), ActiveOutcomeRuntimeErrorV2>>,
    },
    RecordTerminal {
        anchor_id: Uuid,
        payload_hash: String,
        payload: PersistedTrajectoryTerminalV1,
    },
    DeliverWindow {
        anchor_id: Uuid,
        window: Arc<ClosedTrajectoryWindow>,
    },
    ScheduleDeadline {
        anchor_id: Uuid,
        generation: u64,
        deadline: Instant,
    },
    CancelDeadline {
        anchor_id: Uuid,
        generation: u64,
    },
    ScheduleSinkRetry {
        anchor_id: Uuid,
        generation: u64,
    },
    ScheduleDeliveryRetry {
        anchor_id: Uuid,
        generation: u64,
    },
    SchedulerPressureClosed {
        pool_id: String,
        pressure_generation: u64,
    },
    EvidenceCapacityPressureClosed {
        pressure_generation: u64,
    },
    SamplingSuppressed(&'static str),
    SamplingResumed,
    AnchorRefused {
        anchor_id: Uuid,
        reason: AnchorRefusalReason,
    },
    Health(&'static str),
    Drained,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AnchorRefusalReason {
    IntakeClosed,
    InvalidProposal,
    PoolFull,
    SamplingSuppressed,
    SinkRejected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WindowPhase {
    PendingQueued,
    PendingInFlight,
    AwaitingAnchorEnd,
    Collecting,
    PersistingTerminal,
    Delivering,
}

/// Bounded, payload-free state view used by state-machine tests.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CoordinatorSnapshot {
    pub(crate) accepting: bool,
    pub(crate) sampling_enabled: bool,
    pub(crate) shutdown: bool,
    pub(crate) active_windows: usize,
    pub(crate) registered_primary_calls: usize,
    pub(crate) ancestry_entries: usize,
    pub(crate) staged_calls: usize,
    pub(crate) unmatched_pressure: usize,
    pub(crate) scheduler_pressure_pools: BTreeSet<String>,
    pub(crate) evidence_capacity_pressure: bool,
    pub(crate) pending_by_pool: BTreeMap<String, usize>,
    pub(crate) phases: BTreeMap<Uuid, WindowPhase>,
}

#[derive(Clone, PartialEq, Eq)]
struct PrimaryCallState {
    registration: PrimaryCallRegistration,
    ended: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct AncestryEntry {
    parent_uuid: Option<Uuid>,
    root_uuid: Uuid,
    owner_uuid: Uuid,
    scope_type: ScopeType,
    ended: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum UnresolvedOwnership {
    Parent {
        parent_uuid: Uuid,
        first_ingest_seq: u64,
        ended: bool,
    },
    Contradictory {
        first_ingest_seq: u64,
    },
}

enum StagedObservation {
    Event(Arc<CapturedTrajectoryEvent>, LossSnapshot),
    Oversized(ProjectedTrajectoryEvent, LossSnapshot),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct EventDedupeKey {
    uuid: Uuid,
    kind: String,
    phase: Option<String>,
    payload_hash: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FrozenOutcome {
    Closed(TrajectoryTrigger),
    Rejected(TrajectoryRejectionReason),
}

struct FrozenTerminal {
    outcome: FrozenOutcome,
    payload: PersistedTrajectoryTerminalV1,
    payload_hash: Option<String>,
    retry_generation: u64,
}

struct ProvisionalTerminal {
    outcome: FrozenOutcome,
    cutoff_ingest_seq: u64,
    uncertain_uuids: BTreeSet<Uuid>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct WindowIdentity {
    anchor_id: Uuid,
    anchor_call_uuid: Uuid,
    root_uuid: Uuid,
    owner_uuid: Uuid,
    pool_id: String,
    requested_progress: usize,
}

struct WindowState {
    identity: WindowIdentity,
    seed: Option<TrajectoryWindowSeed>,
    pending: PendingTrajectoryWindow,
    pending_hash: String,
    limits: WindowLimits,
    phase: WindowPhase,
    accepted: bool,
    deadline_generation: u64,
    opened_after_ingest_seq: u64,
    loss_at_open: LossSnapshot,
    loss_at_boundary: Option<u64>,
    capture_after_ingest_seq: Option<u64>,
    highest_observed_ingest_seq: u64,
    events: Vec<Arc<CapturedTrajectoryEvent>>,
    event_bytes: usize,
    dedupe: BTreeSet<EventDedupeKey>,
    counted_primary_calls: BTreeSet<Uuid>,
    observed_progress: usize,
    provisional: Option<ProvisionalTerminal>,
    frozen: Option<FrozenTerminal>,
    delivery: Option<Arc<ClosedTrajectoryWindow>>,
    delivery_retry_generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SinkWorkKind {
    Pending,
    Terminal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct SinkWork {
    anchor_id: Uuid,
    kind: SinkWorkKind,
}

/// The single serialized owner of all trajectory classification and window state.
pub(crate) struct Coordinator {
    limits: CoordinatorLimits,
    pool_limits: BTreeMap<String, usize>,
    pending_by_pool: BTreeMap<String, usize>,
    calls: BTreeMap<Uuid, PrimaryCallState>,
    ancestry: BTreeMap<Uuid, AncestryEntry>,
    staged: BTreeMap<Uuid, Vec<StagedObservation>>,
    dropped_staged_calls: BTreeMap<Uuid, u64>,
    pressure_calls: BTreeMap<Uuid, u64>,
    // Hard-overflow debt is anonymous but bounded. A later V2 registration or
    // unmatched V1 end consumes one unit; admission remains suppressed until
    // every unit drains, so new capture boundaries follow all lost starts.
    pressure_overflow_count: usize,
    unresolved_ownership: BTreeMap<Uuid, UnresolvedOwnership>,
    seen_lifecycle_progress: BTreeMap<Uuid, BTreeSet<Uuid>>,
    lifecycle_saturated_roots: BTreeSet<Uuid>,
    windows: BTreeMap<Uuid, WindowState>,
    sink_queue: VecDeque<SinkWork>,
    sink_inflight: Option<SinkWork>,
    delivery_queue: VecDeque<Uuid>,
    delivery_inflight: Option<Uuid>,
    tombstones: VecDeque<Uuid>,
    accepting: bool,
    shutdown: bool,
    drained_emitted: bool,
    permanent_sampling_fault: bool,
    sampling_fault_reason: Option<&'static str>,
    scheduler_pressure_pools: BTreeMap<String, u64>,
    evidence_capacity_pressure: Option<u64>,
    admission_epoch: u64,
    highest_dropped_ingest_seq: u64,
    classification_loss_epoch: u64,
    next_timer_generation: u64,
    next_pressure_generation: u64,
    active_outcomes_enabled: bool,
    now_utc: Arc<dyn Fn() -> DateTime<Utc> + Send + Sync>,
}

impl Coordinator {
    pub(crate) fn new(pool_limits: BTreeMap<String, usize>) -> Self {
        Self::new_with_clock_and_active(pool_limits, Arc::new(Utc::now), false)
    }

    pub(crate) fn new_with_active_outcomes(pool_limits: BTreeMap<String, usize>) -> Self {
        Self::new_with_clock_and_active(pool_limits, Arc::new(Utc::now), true)
    }

    #[cfg(test)]
    pub(crate) fn new_with_clock(
        pool_limits: BTreeMap<String, usize>,
        now_utc: Arc<dyn Fn() -> DateTime<Utc> + Send + Sync>,
    ) -> Self {
        Self::new_with_clock_and_active(pool_limits, now_utc, false)
    }

    fn new_with_clock_and_active(
        pool_limits: BTreeMap<String, usize>,
        now_utc: Arc<dyn Fn() -> DateTime<Utc> + Send + Sync>,
        active_outcomes_enabled: bool,
    ) -> Self {
        let total_max_pending = pool_limits
            .values()
            .copied()
            .fold(0usize, usize::saturating_add);
        let limits = CoordinatorLimits::from_total_max_pending(total_max_pending);
        let pending_by_pool = pool_limits
            .keys()
            .map(|pool_id| (pool_id.clone(), 0))
            .collect();
        Self {
            limits,
            pool_limits,
            pending_by_pool,
            calls: BTreeMap::new(),
            ancestry: BTreeMap::new(),
            staged: BTreeMap::new(),
            dropped_staged_calls: BTreeMap::new(),
            pressure_calls: BTreeMap::new(),
            pressure_overflow_count: 0,
            unresolved_ownership: BTreeMap::new(),
            seen_lifecycle_progress: BTreeMap::new(),
            lifecycle_saturated_roots: BTreeSet::new(),
            windows: BTreeMap::new(),
            sink_queue: VecDeque::new(),
            sink_inflight: None,
            delivery_queue: VecDeque::new(),
            delivery_inflight: None,
            tombstones: VecDeque::new(),
            accepting: true,
            shutdown: false,
            drained_emitted: false,
            permanent_sampling_fault: false,
            sampling_fault_reason: None,
            scheduler_pressure_pools: BTreeMap::new(),
            evidence_capacity_pressure: None,
            admission_epoch: 0,
            highest_dropped_ingest_seq: 0,
            classification_loss_epoch: 0,
            next_timer_generation: 0,
            next_pressure_generation: 0,
            active_outcomes_enabled,
            now_utc,
        }
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> CoordinatorSnapshot {
        CoordinatorSnapshot {
            accepting: self.accepting,
            sampling_enabled: self.sampling_available(),
            shutdown: self.shutdown,
            active_windows: self.windows.len(),
            registered_primary_calls: self.calls.len(),
            ancestry_entries: self.ancestry.len(),
            staged_calls: self.staged.len(),
            unmatched_pressure: self
                .pressure_calls
                .len()
                .saturating_add(self.pressure_overflow_count),
            scheduler_pressure_pools: self.scheduler_pressure_pools.keys().cloned().collect(),
            evidence_capacity_pressure: self.evidence_capacity_pressure.is_some(),
            pending_by_pool: self.pending_by_pool.clone(),
            phases: self
                .windows
                .iter()
                .map(|(anchor_id, window)| (*anchor_id, window.phase))
                .collect(),
        }
    }

    pub(crate) fn handle(&mut self, command: CoordinatorCommand) -> Vec<CoordinatorEffect> {
        let sampling_before = self.sampling_available();
        let mut effects = Vec::new();
        match command {
            CoordinatorCommand::RegisterPrimaryCall {
                registration,
                loss,
                admission_epoch,
            } => {
                self.observe_loss(loss, &mut effects);
                self.register_primary_call(registration, admission_epoch, &mut effects);
            }
            CoordinatorCommand::RegistrationLoss { loss, reason } => {
                self.observe_loss(loss, &mut effects);
                effects.push(CoordinatorEffect::Health(reason));
            }
            CoordinatorCommand::RegisterAnchor(registration) => {
                self.observe_loss(registration.loss, &mut effects);
                self.register_anchor(*registration, &mut effects);
            }
            CoordinatorCommand::StageActiveOutcome {
                call_uuid,
                admission_epoch,
                stage,
                reply,
            } => {
                let valid = self.active_outcomes_enabled
                    && self.accepting
                    && self.admission_epoch == admission_epoch
                    && self.calls.get(&call_uuid).is_some_and(|call| {
                        !call.ended
                            && call.registration.root_uuid == stage.raw_root_uuid
                            && call.registration.owner_uuid == stage.pinned_owner_uuid
                    });
                if valid {
                    effects.push(CoordinatorEffect::StageActiveOutcome { stage, reply });
                } else {
                    let _ = reply.send(Err(ActiveOutcomeRuntimeErrorV2::Conflict));
                }
            }
            CoordinatorCommand::ActiveRepresentativeBarrier { observation, reply } => {
                effects.push(CoordinatorEffect::ActiveRepresentativeBarrier { observation, reply });
            }
            CoordinatorCommand::ActiveDeadlineBarrier(observation) => {
                if self.active_outcomes_enabled {
                    effects.push(CoordinatorEffect::ActiveDeadlineBarrier(observation));
                }
            }
            CoordinatorCommand::ShutdownActiveOutcomes {
                observed_at_unix_ms,
                reply,
            } => effects.push(CoordinatorEffect::ShutdownActiveOutcomes {
                observed_at_unix_ms,
                reply,
            }),
            CoordinatorCommand::PendingRecorded(ack) => {
                self.pending_recorded(ack, &mut effects);
            }
            CoordinatorCommand::ObservedEvent { event, loss } => {
                if !event.is_internal_router_event() {
                    self.observe_loss(loss, &mut effects);
                    self.observe_event(event, loss, &mut effects);
                }
            }
            CoordinatorCommand::OversizedEvent { event, loss } => {
                if !event.is_internal_router_event() {
                    self.observe_loss(loss, &mut effects);
                    self.observe_oversized(event, loss, &mut effects);
                }
            }
            CoordinatorCommand::DeadlineElapsed {
                anchor_id,
                generation,
                loss,
            } => {
                self.observe_loss(loss, &mut effects);
                self.deadline_elapsed(anchor_id, generation, &mut effects);
            }
            CoordinatorCommand::DeadlineBarrierFailed {
                anchor_id,
                generation,
                loss,
            } => {
                self.observe_loss(loss, &mut effects);
                self.deadline_barrier_failed(anchor_id, generation, &mut effects);
            }
            CoordinatorCommand::TerminalRecorded(ack) => {
                self.terminal_recorded(ack, &mut effects);
            }
            CoordinatorCommand::WindowDelivered(ack) => {
                self.window_delivered(ack, &mut effects);
            }
            CoordinatorCommand::DeliveryRetryElapsed {
                anchor_id,
                generation,
            } => self.delivery_retry_elapsed(anchor_id, generation),
            CoordinatorCommand::SinkRetryElapsed {
                anchor_id,
                generation,
            } => self.sink_retry_elapsed(anchor_id, generation),
            CoordinatorCommand::SchedulerCapacityRecovered {
                pool_id,
                pressure_generation,
            } => {
                if self.scheduler_pressure_pools.get(&pool_id) == Some(&pressure_generation) {
                    self.scheduler_pressure_pools.remove(&pool_id);
                }
            }
            CoordinatorCommand::EvidenceCapacityRecovered {
                pressure_generation,
            } => {
                if self.evidence_capacity_pressure == Some(pressure_generation) {
                    self.evidence_capacity_pressure = None;
                }
            }
            CoordinatorCommand::Health(reason) => effects.push(CoordinatorEffect::Health(reason)),
            CoordinatorCommand::PermanentFault(reason) => {
                self.set_permanent_sampling_fault(reason, &mut effects);
            }
            CoordinatorCommand::Shutdown {
                admission_epoch,
                loss,
            } => {
                self.observe_loss(loss, &mut effects);
                self.shutdown(admission_epoch, &mut effects);
            }
        }

        self.pump_sink(&mut effects);
        self.pump_delivery(&mut effects);
        self.prune_indexes();

        let sampling_after = self.sampling_available();
        if sampling_before && !sampling_after {
            effects.push(CoordinatorEffect::SamplingSuppressed(
                self.sampling_suppression_reason(),
            ));
        } else if !sampling_before && sampling_after {
            effects.push(CoordinatorEffect::SamplingResumed);
        }
        self.maybe_emit_drained(&mut effects);
        effects
    }

    fn sampling_available(&self) -> bool {
        self.base_sampling_available()
            && (self.pool_limits.is_empty()
                || self
                    .pool_limits
                    .keys()
                    .any(|pool_id| !self.scheduler_pressure_pools.contains_key(pool_id)))
    }

    fn sampling_available_for_pool(&self, pool_id: &str) -> bool {
        self.base_sampling_available() && !self.scheduler_pressure_pools.contains_key(pool_id)
    }

    fn base_sampling_available(&self) -> bool {
        self.accepting
            && !self.shutdown
            && !self.permanent_sampling_fault
            && self.evidence_capacity_pressure.is_none()
            && self.pressure_calls.is_empty()
            && self.pressure_overflow_count == 0
    }

    fn sampling_suppression_reason(&self) -> &'static str {
        self.sampling_fault_reason.unwrap_or(if self.shutdown {
            HEALTH_INTAKE_CLOSED
        } else if self.evidence_capacity_pressure.is_some() {
            HEALTH_EVIDENCE_CAPACITY
        } else if !self.pool_limits.is_empty()
            && self
                .pool_limits
                .keys()
                .all(|pool_id| self.scheduler_pressure_pools.contains_key(pool_id))
        {
            HEALTH_SCHEDULER_PRESSURE
        } else {
            HEALTH_STAGING_PRESSURE
        })
    }

    fn observe_loss(&mut self, loss: LossSnapshot, effects: &mut Vec<CoordinatorEffect>) {
        self.highest_dropped_ingest_seq = self
            .highest_dropped_ingest_seq
            .max(loss.highest_dropped_ingest_seq);
        if loss.classification_loss_epoch > self.classification_loss_epoch {
            self.classification_loss_epoch = loss.classification_loss_epoch;
            self.set_permanent_sampling_fault(HEALTH_CLASSIFICATION_LOSS, effects);
            let affected = self
                .windows
                .iter()
                .filter_map(|(anchor_id, window)| {
                    (window.frozen.is_none() && window.provisional.is_none()).then_some(*anchor_id)
                })
                .collect::<Vec<_>>();
            for anchor_id in affected {
                self.freeze_rejected(anchor_id, TrajectoryRejectionReason::EventLoss, effects);
            }
        }

        let affected = self
            .windows
            .iter()
            .filter_map(|(anchor_id, window)| {
                let boundary = window.capture_after_ingest_seq?;
                let boundary_loss = window
                    .loss_at_boundary
                    .unwrap_or(window.loss_at_open.highest_dropped_ingest_seq);
                (window.frozen.is_none()
                    && window.provisional.is_none()
                    && loss.highest_dropped_ingest_seq > boundary
                    && loss.highest_dropped_ingest_seq > boundary_loss)
                    .then_some(*anchor_id)
            })
            .collect::<Vec<_>>();
        for anchor_id in affected {
            self.freeze_rejected(anchor_id, TrajectoryRejectionReason::EventLoss, effects);
        }
    }

    fn set_permanent_sampling_fault(
        &mut self,
        reason: &'static str,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        if !self.permanent_sampling_fault {
            self.permanent_sampling_fault = true;
            self.sampling_fault_reason = Some(reason);
            effects.push(CoordinatorEffect::Health(reason));
        }
    }

    fn allocate_pressure_generation(&mut self) -> u64 {
        self.next_pressure_generation = self.next_pressure_generation.saturating_add(1);
        self.next_pressure_generation
    }

    fn close_scheduler_pressure(&mut self, pool_id: String, effects: &mut Vec<CoordinatorEffect>) {
        let pressure_generation = self.allocate_pressure_generation();
        self.scheduler_pressure_pools
            .insert(pool_id.clone(), pressure_generation);
        effects.push(CoordinatorEffect::SchedulerPressureClosed {
            pool_id,
            pressure_generation,
        });
    }

    fn close_evidence_capacity_pressure(&mut self, effects: &mut Vec<CoordinatorEffect>) {
        let pressure_generation = self.allocate_pressure_generation();
        self.evidence_capacity_pressure = Some(pressure_generation);
        effects.push(CoordinatorEffect::EvidenceCapacityPressureClosed {
            pressure_generation,
        });
    }

    fn register_primary_call(
        &mut self,
        registration: PrimaryCallRegistration,
        admission_epoch: u64,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        if !self.accepting || admission_epoch < self.admission_epoch {
            return;
        }
        if !valid_frozen_path(&registration) {
            self.invalidate_root(
                registration.root_uuid,
                TrajectoryRejectionReason::ContradictoryOwnership,
                effects,
            );
            self.set_permanent_sampling_fault(HEALTH_OWNERSHIP_LOSS, effects);
            return;
        }
        if let Some(existing) = self.calls.get(&registration.call_uuid) {
            if existing.registration == registration {
                return;
            }
            let old_root = existing.registration.root_uuid;
            self.invalidate_root(
                old_root,
                TrajectoryRejectionReason::ContradictoryOwnership,
                effects,
            );
            self.invalidate_root(
                registration.root_uuid,
                TrajectoryRejectionReason::ContradictoryOwnership,
                effects,
            );
            self.set_permanent_sampling_fault(HEALTH_OWNERSHIP_LOSS, effects);
            return;
        }
        if self.calls.len() >= self.limits.primary_call_capacity {
            self.set_permanent_sampling_fault(HEALTH_CLASSIFICATION_LOSS, effects);
            self.invalidate_root(
                registration.root_uuid,
                TrajectoryRejectionReason::EventLoss,
                effects,
            );
            return;
        }
        if let Some(existing) = self.ancestry.get(&registration.call_uuid).copied() {
            let compatible = existing.parent_uuid == Some(registration.parent_uuid)
                && existing.root_uuid == registration.root_uuid
                && existing.owner_uuid == registration.owner_uuid
                && existing.scope_type == ScopeType::Llm
                && !existing.ended;
            if !compatible {
                self.invalidate_root(
                    existing.root_uuid,
                    TrajectoryRejectionReason::ContradictoryOwnership,
                    effects,
                );
                self.invalidate_root(
                    registration.root_uuid,
                    TrajectoryRejectionReason::ContradictoryOwnership,
                    effects,
                );
                self.set_permanent_sampling_fault(HEALTH_OWNERSHIP_LOSS, effects);
                return;
            }
        }
        if !self.seed_frozen_path(&registration, effects) {
            return;
        }

        let call_uuid = registration.call_uuid;
        let root_uuid = registration.root_uuid;
        self.calls.insert(
            call_uuid,
            PrimaryCallState {
                registration,
                ended: false,
            },
        );

        let staged_evidence_lost = self.dropped_staged_calls.remove(&call_uuid).is_some()
            | self.pressure_calls.remove(&call_uuid).is_some();
        if staged_evidence_lost {
            self.invalidate_root(root_uuid, TrajectoryRejectionReason::EventLoss, effects);
        } else if self.pressure_overflow_count > 0 {
            self.pressure_overflow_count -= 1;
        }
        let parent_uuid = self.calls[&call_uuid].registration.parent_uuid;
        self.reconcile_unresolved_registration(
            call_uuid,
            Some(parent_uuid),
            root_uuid,
            false,
            effects,
        );
        if let Some(mut staged) = self.staged.remove(&call_uuid) {
            staged.sort_by_key(staged_ingest_seq);
            self.replay_staged(call_uuid, root_uuid, staged, effects);
        }
    }

    fn seed_frozen_path(
        &mut self,
        registration: &PrimaryCallRegistration,
        effects: &mut Vec<CoordinatorEffect>,
    ) -> bool {
        let mut seeded_scopes = Vec::with_capacity(registration.owner_path.len());
        for (index, scope) in registration.owner_path.iter().enumerate() {
            let parent_uuid = index
                .checked_sub(1)
                .map(|parent_index| registration.owner_path[parent_index].uuid);
            let entry = AncestryEntry {
                parent_uuid,
                root_uuid: registration.root_uuid,
                owner_uuid: registration.owner_uuid,
                scope_type: scope.scope_type,
                ended: false,
            };
            if !self.insert_ancestry(scope.uuid, entry, registration.root_uuid, effects) {
                return false;
            }
            seeded_scopes.push((scope.uuid, parent_uuid));
        }
        for (scope_uuid, parent_uuid) in seeded_scopes {
            self.reconcile_unresolved_registration(
                scope_uuid,
                parent_uuid,
                registration.root_uuid,
                false,
                effects,
            );
            self.replay_staged_children_except(scope_uuid, Some(registration.call_uuid), effects);
        }
        true
    }

    fn insert_ancestry(
        &mut self,
        uuid: Uuid,
        entry: AncestryEntry,
        affected_root: Uuid,
        effects: &mut Vec<CoordinatorEffect>,
    ) -> bool {
        if let Some(existing) = self.ancestry.get(&uuid).copied() {
            let compatible_parent = existing.parent_uuid == entry.parent_uuid
                || existing.parent_uuid.is_none()
                || entry.parent_uuid.is_none();
            let same = compatible_parent
                && existing.root_uuid == entry.root_uuid
                && existing.owner_uuid == entry.owner_uuid
                && existing.scope_type == entry.scope_type
                && !existing.ended;
            if !same {
                self.invalidate_root(
                    existing.root_uuid,
                    TrajectoryRejectionReason::ContradictoryOwnership,
                    effects,
                );
                self.invalidate_root(
                    affected_root,
                    TrajectoryRejectionReason::ContradictoryOwnership,
                    effects,
                );
                self.set_permanent_sampling_fault(HEALTH_OWNERSHIP_LOSS, effects);
            } else if existing.parent_uuid.is_none() && entry.parent_uuid.is_some() {
                self.ancestry.insert(uuid, entry);
            }
            return same;
        }
        if self.ancestry.len() >= self.limits.ancestry_capacity {
            self.invalidate_root(
                affected_root,
                TrajectoryRejectionReason::ContradictoryOwnership,
                effects,
            );
            self.set_permanent_sampling_fault(HEALTH_OWNERSHIP_LOSS, effects);
            return false;
        }
        self.ancestry.insert(uuid, entry);
        true
    }

    fn register_anchor(
        &mut self,
        mut registration: AnchorRegistration,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let _proposal_permit = registration.proposal_permit.take();
        let pending = registration.seed.pending_projection();
        let identity = pending_identity(&pending);
        let anchor_id = identity.anchor_id;
        let invalid_limits = registration.limits.requested_progress == 0
            || registration.limits.max_events == 0
            || registration.limits.max_bytes == 0
            || identity.requested_progress != registration.limits.requested_progress;
        let invalid_call = self
            .calls
            .get(&identity.anchor_call_uuid)
            .is_none_or(|call| {
                call.registration.root_uuid != identity.root_uuid
                    || call.registration.owner_uuid != identity.owner_uuid
                    || call.registration.api_family != pending.replay_capability_facts.api_family
                    || call.registration.owner_path.len() != pending.owner_path.len()
                    || call
                        .registration
                        .owner_path
                        .iter()
                        .zip(&pending.owner_path)
                        .any(|(registered, persisted)| {
                            registered.uuid != persisted.uuid
                                || registered.name != persisted.name
                                || registered.scope_type != persisted.scope_type
                        })
            });
        let invalid_proposal = self.windows.contains_key(&anchor_id)
            || self.tombstones.contains(&anchor_id)
            || invalid_limits
            || invalid_call;

        let refusal = if !self.accepting
            || self.shutdown
            || registration.admission_epoch < self.admission_epoch
        {
            Some(AnchorRefusalReason::IntakeClosed)
        } else if !self.sampling_available_for_pool(&identity.pool_id)
            || self.lifecycle_saturated_roots.contains(&identity.root_uuid)
        {
            Some(AnchorRefusalReason::SamplingSuppressed)
        } else if invalid_proposal {
            Some(AnchorRefusalReason::InvalidProposal)
        } else {
            let Some(max_pending) = self.pool_limits.get(&identity.pool_id).copied() else {
                effects.push(CoordinatorEffect::AnchorRefused {
                    anchor_id,
                    reason: AnchorRefusalReason::InvalidProposal,
                });
                return;
            };
            let current = self
                .pending_by_pool
                .get(&identity.pool_id)
                .copied()
                .unwrap_or(0);
            (current >= max_pending).then_some(AnchorRefusalReason::PoolFull)
        };

        if let Some(reason) = refusal {
            if reason == AnchorRefusalReason::PoolFull {
                effects.push(CoordinatorEffect::Health(HEALTH_POOL_FULL));
            }
            effects.push(CoordinatorEffect::AnchorRefused { anchor_id, reason });
            return;
        }

        let Ok(pending_hash) = canonical_hash(&pending) else {
            self.set_permanent_sampling_fault(HEALTH_SINK_FAILURE, effects);
            effects.push(CoordinatorEffect::AnchorRefused {
                anchor_id,
                reason: AnchorRefusalReason::InvalidProposal,
            });
            return;
        };
        let pending_count = self
            .pending_by_pool
            .get_mut(&identity.pool_id)
            .expect("validated pool identity must have a pending counter");
        *pending_count = pending_count.saturating_add(1);

        self.next_timer_generation = self.next_timer_generation.saturating_add(1);
        let deadline_generation = self.next_timer_generation;
        self.windows.insert(
            anchor_id,
            WindowState {
                identity,
                seed: Some(registration.seed),
                pending,
                pending_hash,
                limits: registration.limits,
                phase: WindowPhase::PendingQueued,
                accepted: false,
                deadline_generation,
                opened_after_ingest_seq: registration.opened_after_ingest_seq,
                loss_at_open: registration.loss,
                loss_at_boundary: None,
                capture_after_ingest_seq: None,
                highest_observed_ingest_seq: registration.opened_after_ingest_seq,
                events: Vec::new(),
                event_bytes: 0,
                dedupe: BTreeSet::new(),
                counted_primary_calls: BTreeSet::new(),
                observed_progress: 0,
                provisional: None,
                frozen: None,
                delivery: None,
                delivery_retry_generation: 0,
            },
        );
        self.sink_queue.push_back(SinkWork {
            anchor_id,
            kind: SinkWorkKind::Pending,
        });
        effects.push(CoordinatorEffect::ScheduleDeadline {
            anchor_id,
            generation: deadline_generation,
            deadline: registration.monotonic_deadline,
        });
    }

    fn invalidate_root(
        &mut self,
        root_uuid: Uuid,
        reason: TrajectoryRejectionReason,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let anchors = self
            .windows
            .iter()
            .filter_map(|(anchor_id, window)| {
                (window.identity.root_uuid == root_uuid
                    && window.frozen.is_none()
                    && window.provisional.is_none())
                .then_some(*anchor_id)
            })
            .collect::<Vec<_>>();
        for anchor_id in anchors {
            self.freeze_rejected(anchor_id, reason, effects);
        }
    }

    fn pump_sink(&mut self, effects: &mut Vec<CoordinatorEffect>) {
        if self.sink_inflight.is_some() {
            return;
        }
        while let Some(work) = self.sink_queue.pop_front() {
            let Some(window) = self.windows.get_mut(&work.anchor_id) else {
                continue;
            };
            match work.kind {
                SinkWorkKind::Pending if window.phase == WindowPhase::PendingQueued => {
                    window.phase = WindowPhase::PendingInFlight;
                    self.sink_inflight = Some(work);
                    effects.push(CoordinatorEffect::RecordPending {
                        anchor_id: work.anchor_id,
                        payload_hash: window.pending_hash.clone(),
                        payload: window.pending.clone(),
                    });
                    return;
                }
                SinkWorkKind::Terminal
                    if window.accepted
                        && window.phase == WindowPhase::PersistingTerminal
                        && window.frozen.is_some() =>
                {
                    let terminal = window
                        .frozen
                        .as_ref()
                        .expect("checked terminal state must be present");
                    let Some(payload_hash) = terminal.payload_hash.as_ref() else {
                        continue;
                    };
                    self.sink_inflight = Some(work);
                    effects.push(CoordinatorEffect::RecordTerminal {
                        anchor_id: work.anchor_id,
                        payload_hash: payload_hash.clone(),
                        payload: terminal.payload.clone(),
                    });
                    return;
                }
                _ => {}
            }
        }
    }

    fn pump_delivery(&mut self, effects: &mut Vec<CoordinatorEffect>) {
        if self.delivery_inflight.is_some() {
            return;
        }
        while let Some(anchor_id) = self.delivery_queue.pop_front() {
            let Some(window) = self.windows.get_mut(&anchor_id) else {
                continue;
            };
            if window.phase != WindowPhase::Delivering {
                continue;
            }
            let Some(delivery) = window.delivery.as_ref().cloned() else {
                continue;
            };
            self.delivery_inflight = Some(anchor_id);
            effects.push(CoordinatorEffect::DeliverWindow {
                anchor_id,
                window: delivery,
            });
            return;
        }
    }

    fn pending_recorded(&mut self, ack: SinkAck, effects: &mut Vec<CoordinatorEffect>) {
        let Some(work) = self.sink_inflight else {
            return;
        };
        if work.kind != SinkWorkKind::Pending {
            return;
        }
        if sink_ack_anchor_id(&ack) != work.anchor_id {
            return;
        }
        let Some(window) = self.windows.get(&work.anchor_id) else {
            return;
        };
        let expected_hash = window.pending_hash.clone();
        let expected_pool_id = window.identity.pool_id.clone();

        match ack {
            SinkAck::Applied {
                anchor_id,
                payload_hash,
            }
            | SinkAck::AlreadyApplied {
                anchor_id,
                payload_hash,
            } => {
                if anchor_id != work.anchor_id || payload_hash != expected_hash {
                    self.sink_inflight = None;
                    self.set_permanent_sampling_fault(HEALTH_SINK_MISMATCH, effects);
                    self.refuse_pending_window(work.anchor_id, effects);
                    return;
                }
                self.sink_inflight = None;
                let window = self
                    .windows
                    .get_mut(&anchor_id)
                    .expect("in-flight pending anchor must remain live");
                window.accepted = true;
                window.phase = if window.capture_after_ingest_seq.is_some() {
                    WindowPhase::Collecting
                } else {
                    WindowPhase::AwaitingAnchorEnd
                };
                if window.frozen.is_some() {
                    window.phase = WindowPhase::PersistingTerminal;
                    self.enqueue_terminal_once(anchor_id);
                } else if self.shutdown {
                    self.close_for_shutdown(anchor_id, effects);
                }
            }
            SinkAck::AlreadyTerminal {
                anchor_id,
                pending_hash,
                terminal_hash,
            } => {
                if anchor_id != work.anchor_id
                    || pending_hash != expected_hash
                    || !is_canonical_sha256(&terminal_hash)
                {
                    self.sink_inflight = None;
                    self.set_permanent_sampling_fault(HEALTH_SINK_MISMATCH, effects);
                    self.refuse_pending_window(work.anchor_id, effects);
                    return;
                }
                self.sink_inflight = None;
                self.release_window(anchor_id, effects);
            }
            SinkAck::DurablyDeclined {
                anchor_id,
                pending_hash,
                terminal_hash,
                reason: DurableDeclineReason::NotScheduledQueueFull,
            } => {
                if anchor_id != work.anchor_id
                    || pending_hash != expected_hash
                    || !is_canonical_sha256(&terminal_hash)
                {
                    self.sink_inflight = None;
                    self.set_permanent_sampling_fault(HEALTH_SINK_MISMATCH, effects);
                    self.refuse_pending_window(work.anchor_id, effects);
                    return;
                }
                self.sink_inflight = None;
                self.close_scheduler_pressure(expected_pool_id, effects);
                self.release_window(anchor_id, effects);
            }
            SinkAck::TransientRefused {
                anchor_id,
                reason: TransientRefusalReason::EvidenceCapacity,
            } => {
                if anchor_id != work.anchor_id {
                    self.set_permanent_sampling_fault(HEALTH_SINK_MISMATCH, effects);
                    return;
                }
                self.sink_inflight = None;
                self.close_evidence_capacity_pressure(effects);
                self.release_window(anchor_id, effects);
            }
            SinkAck::Conflict { anchor_id } => {
                if anchor_id != work.anchor_id {
                    self.set_permanent_sampling_fault(HEALTH_SINK_MISMATCH, effects);
                    return;
                }
                self.sink_inflight = None;
                self.set_permanent_sampling_fault(HEALTH_SINK_CONFLICT, effects);
                self.refuse_pending_window(anchor_id, effects);
            }
            SinkAck::Failed { anchor_id, .. } => {
                if anchor_id != work.anchor_id {
                    self.set_permanent_sampling_fault(HEALTH_SINK_MISMATCH, effects);
                    return;
                }
                self.sink_inflight = None;
                self.set_permanent_sampling_fault(HEALTH_SINK_FAILURE, effects);
                self.refuse_pending_window(anchor_id, effects);
            }
        }
    }

    fn terminal_recorded(&mut self, ack: SinkAck, effects: &mut Vec<CoordinatorEffect>) {
        let Some(work) = self.sink_inflight else {
            return;
        };
        if work.kind != SinkWorkKind::Terminal {
            return;
        }
        if sink_ack_anchor_id(&ack) != work.anchor_id {
            return;
        }
        let Some(window) = self.windows.get(&work.anchor_id) else {
            return;
        };
        let Some(terminal) = window.frozen.as_ref() else {
            return;
        };
        let expected_pending_hash = window.pending_hash.clone();
        let Some(expected_terminal_hash) = terminal.payload_hash.clone() else {
            return;
        };

        let applied = match ack {
            SinkAck::Applied {
                anchor_id,
                payload_hash,
            }
            | SinkAck::AlreadyApplied {
                anchor_id,
                payload_hash,
            } => {
                if anchor_id != work.anchor_id || payload_hash != expected_terminal_hash {
                    self.sink_inflight = None;
                    self.set_permanent_sampling_fault(HEALTH_SINK_MISMATCH, effects);
                    self.schedule_terminal_retry(work.anchor_id, effects);
                    return;
                }
                true
            }
            SinkAck::AlreadyTerminal {
                anchor_id,
                pending_hash,
                terminal_hash,
            } => {
                if anchor_id != work.anchor_id
                    || pending_hash != expected_pending_hash
                    || terminal_hash != expected_terminal_hash
                {
                    self.sink_inflight = None;
                    self.set_permanent_sampling_fault(HEALTH_SINK_MISMATCH, effects);
                    self.schedule_terminal_retry(work.anchor_id, effects);
                    return;
                }
                true
            }
            SinkAck::DurablyDeclined { anchor_id, .. }
            | SinkAck::TransientRefused { anchor_id, .. } => {
                if anchor_id != work.anchor_id {
                    self.set_permanent_sampling_fault(HEALTH_SINK_MISMATCH, effects);
                    return;
                }
                self.sink_inflight = None;
                self.set_permanent_sampling_fault(HEALTH_SINK_MISMATCH, effects);
                self.schedule_terminal_retry(anchor_id, effects);
                return;
            }
            SinkAck::Conflict { anchor_id } => {
                if anchor_id != work.anchor_id {
                    self.set_permanent_sampling_fault(HEALTH_SINK_MISMATCH, effects);
                    return;
                }
                self.sink_inflight = None;
                self.set_permanent_sampling_fault(HEALTH_SINK_CONFLICT, effects);
                self.schedule_terminal_retry(anchor_id, effects);
                return;
            }
            SinkAck::Failed { anchor_id, .. } => {
                if anchor_id != work.anchor_id {
                    self.set_permanent_sampling_fault(HEALTH_SINK_MISMATCH, effects);
                    return;
                }
                self.sink_inflight = None;
                self.set_permanent_sampling_fault(HEALTH_SINK_FAILURE, effects);
                self.schedule_terminal_retry(anchor_id, effects);
                return;
            }
        };
        if !applied {
            return;
        }

        self.sink_inflight = None;
        let outcome = self
            .windows
            .get(&work.anchor_id)
            .and_then(|window| window.frozen.as_ref())
            .map(|terminal| terminal.outcome)
            .expect("acknowledged terminal must remain frozen");
        match outcome {
            FrozenOutcome::Rejected(_) => self.release_window(work.anchor_id, effects),
            FrozenOutcome::Closed(trigger) => {
                let window = self
                    .windows
                    .get_mut(&work.anchor_id)
                    .expect("acknowledged closed window must remain live");
                let seed = window
                    .seed
                    .take()
                    .expect("closed window must retain its memory-only seed");
                let closed_at = window
                    .frozen
                    .as_ref()
                    .expect("closed terminal must remain frozen")
                    .payload
                    .closed_at;
                let delivery = Arc::new(seed.into_closed(
                    window.events.clone(),
                    window.observed_progress,
                    trigger,
                    closed_at,
                ));
                window.delivery = Some(delivery);
                window.phase = WindowPhase::Delivering;
                self.delivery_queue.push_back(work.anchor_id);
            }
        }
    }

    fn schedule_terminal_retry(&mut self, anchor_id: Uuid, effects: &mut Vec<CoordinatorEffect>) {
        let Some(window) = self.windows.get_mut(&anchor_id) else {
            return;
        };
        let Some(terminal) = window.frozen.as_mut() else {
            return;
        };
        terminal.retry_generation = terminal.retry_generation.saturating_add(1);
        effects.push(CoordinatorEffect::ScheduleSinkRetry {
            anchor_id,
            generation: terminal.retry_generation,
        });
    }

    fn sink_retry_elapsed(&mut self, anchor_id: Uuid, generation: u64) {
        let Some(window) = self.windows.get(&anchor_id) else {
            return;
        };
        let Some(terminal) = window.frozen.as_ref() else {
            return;
        };
        if window.phase != WindowPhase::PersistingTerminal
            || terminal.retry_generation != generation
            || self
                .sink_inflight
                .is_some_and(|work| work.anchor_id == anchor_id)
            || self
                .sink_queue
                .iter()
                .any(|work| work.anchor_id == anchor_id && work.kind == SinkWorkKind::Terminal)
        {
            return;
        }
        self.sink_queue.push_back(SinkWork {
            anchor_id,
            kind: SinkWorkKind::Terminal,
        });
    }

    fn window_delivered(&mut self, ack: DeliveryAck, effects: &mut Vec<CoordinatorEffect>) {
        let anchor_id = match ack {
            DeliveryAck::Delivered { anchor_id } | DeliveryAck::AlreadyDelivered { anchor_id } => {
                anchor_id
            }
            DeliveryAck::Failed { anchor_id, .. } => {
                if self.delivery_inflight != Some(anchor_id) {
                    return;
                }
                self.delivery_inflight = None;
                self.set_permanent_sampling_fault(HEALTH_DELIVERY_FAILURE, effects);
                let window = self
                    .windows
                    .get_mut(&anchor_id)
                    .expect("failed delivery anchor must remain live");
                window.delivery_retry_generation =
                    window.delivery_retry_generation.saturating_add(1);
                effects.push(CoordinatorEffect::ScheduleDeliveryRetry {
                    anchor_id,
                    generation: window.delivery_retry_generation,
                });
                return;
            }
        };
        if self.delivery_inflight != Some(anchor_id) {
            return;
        }
        self.delivery_inflight = None;
        self.release_window(anchor_id, effects);
    }

    fn delivery_retry_elapsed(&mut self, anchor_id: Uuid, generation: u64) {
        let Some(window) = self.windows.get(&anchor_id) else {
            return;
        };
        if window.phase != WindowPhase::Delivering
            || window.delivery_retry_generation != generation
            || self.delivery_inflight == Some(anchor_id)
            || self.delivery_queue.contains(&anchor_id)
        {
            return;
        }
        self.delivery_queue.push_back(anchor_id);
    }

    fn enqueue_terminal_once(&mut self, anchor_id: Uuid) {
        if !self.windows.get(&anchor_id).is_some_and(|window| {
            window
                .frozen
                .as_ref()
                .is_some_and(|terminal| terminal.payload_hash.is_some())
        }) {
            return;
        }
        if self
            .sink_inflight
            .is_some_and(|work| work.anchor_id == anchor_id && work.kind == SinkWorkKind::Terminal)
            || self
                .sink_queue
                .iter()
                .any(|work| work.anchor_id == anchor_id && work.kind == SinkWorkKind::Terminal)
        {
            return;
        }
        self.sink_queue.push_back(SinkWork {
            anchor_id,
            kind: SinkWorkKind::Terminal,
        });
    }

    fn refuse_pending_window(&mut self, anchor_id: Uuid, effects: &mut Vec<CoordinatorEffect>) {
        effects.push(CoordinatorEffect::AnchorRefused {
            anchor_id,
            reason: AnchorRefusalReason::SinkRejected,
        });
        self.release_window(anchor_id, effects);
    }

    fn observe_event(
        &mut self,
        event: Arc<CapturedTrajectoryEvent>,
        loss: LossSnapshot,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        self.observe_event_inner(event, loss, false, effects);
    }

    fn observe_event_inner(
        &mut self,
        event: Arc<CapturedTrajectoryEvent>,
        loss: LossSnapshot,
        replaying_staged: bool,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let call_registered = self.calls.contains_key(&event.event_uuid);
        let Some((root_uuid, owner_uuid)) =
            self.resolve_captured_owner(&event, replaying_staged, effects)
        else {
            if event.category.as_deref() == Some("llm") && !call_registered {
                self.stage_unresolved(
                    event.event_uuid,
                    StagedObservation::Event(event, loss),
                    effects,
                );
            } else {
                self.record_unresolved_ownership(
                    event.event_uuid,
                    event.parent_uuid,
                    event.ingest_seq,
                    is_scope_end(&event),
                    effects,
                );
            }
            return;
        };

        let is_llm_end = is_scope_end(&event) && event.category.as_deref() == Some("llm");
        if self.active_outcomes_enabled {
            effects.push(CoordinatorEffect::ObserveActiveOutcome(
                ActiveOutcomeObservationV2 {
                    root_uuid,
                    event: Arc::clone(&event),
                },
            ));
        }
        if is_llm_end && call_registered && event.call_role != Some(LlmCallRole::Primary) {
            if let Some(call) = self.calls.get_mut(&event.event_uuid) {
                call.ended = true;
            }
            self.invalidate_root(root_uuid, TrajectoryRejectionReason::EventLoss, effects);
            return;
        }
        let registered_primary_end = is_llm_end
            && event.call_role == Some(LlmCallRole::Primary)
            && self.calls.get(&event.event_uuid).is_some_and(|call| {
                !call.ended
                    && call.registration.owner_uuid == owner_uuid
                    && call.registration.root_uuid == root_uuid
            });
        let is_owner_end = is_scope_end(&event)
            && event.scope_type == Some(ScopeType::Agent)
            && event.event_uuid == owner_uuid;
        let is_handoff = is_handoff_event(&event);
        let is_compaction = is_compaction_event(&event);
        let lifecycle_progress = is_handoff || is_compaction;
        let lifecycle_progress_first_seen = if lifecycle_progress {
            if self.lifecycle_saturated_roots.contains(&root_uuid)
                || self
                    .seen_lifecycle_progress
                    .get(&root_uuid)
                    .is_some_and(|seen| seen.contains(&event.event_uuid))
            {
                false
            } else if self
                .seen_lifecycle_progress
                .values()
                .map(BTreeSet::len)
                .sum::<usize>()
                >= self.limits.ancestry_capacity
            {
                self.lifecycle_saturated_roots.insert(root_uuid);
                self.invalidate_root(root_uuid, TrajectoryRejectionReason::EventLoss, effects);
                effects.push(CoordinatorEffect::Health(HEALTH_CLASSIFICATION_LOSS));
                false
            } else {
                self.seen_lifecycle_progress
                    .entry(root_uuid)
                    .or_default()
                    .insert(event.event_uuid);
                true
            }
        } else {
            false
        };
        let dedupe = captured_dedupe_key(&event);

        let anchor_ids = self.windows.keys().copied().collect::<Vec<_>>();
        let mut decisions = Vec::new();
        for anchor_id in anchor_ids {
            let Some(window) = self.windows.get_mut(&anchor_id) else {
                continue;
            };
            let replaying_for_window = replaying_staged
                && window.provisional.as_ref().is_none_or(|terminal| {
                    terminal.uncertain_uuids.contains(&event.event_uuid)
                        && event.ingest_seq <= terminal.cutoff_ingest_seq
                });
            if window.frozen.is_some()
                || (window.provisional.is_some() && !replaying_for_window)
                || window.identity.root_uuid != root_uuid
            {
                continue;
            }
            if is_llm_end && event.event_uuid == window.identity.anchor_call_uuid {
                if window.capture_after_ingest_seq.is_none() {
                    window.capture_after_ingest_seq = Some(event.ingest_seq);
                    window.highest_observed_ingest_seq =
                        window.highest_observed_ingest_seq.max(event.ingest_seq);
                    window.loss_at_boundary = Some(loss.highest_dropped_ingest_seq);
                    window.dedupe.insert(dedupe.clone());
                    if window.accepted {
                        window.phase = WindowPhase::Collecting;
                    }
                }
                continue;
            }
            let Some(boundary) = window.capture_after_ingest_seq else {
                continue;
            };
            if event.ingest_seq <= boundary || window.identity.owner_uuid != owner_uuid {
                continue;
            }
            if event.ingest_seq < window.highest_observed_ingest_seq && !replaying_for_window {
                decisions.push((
                    anchor_id,
                    FrozenOutcome::Rejected(TrajectoryRejectionReason::EventLoss),
                ));
                continue;
            }
            window.highest_observed_ingest_seq =
                window.highest_observed_ingest_seq.max(event.ingest_seq);
            if !window.dedupe.insert(dedupe.clone()) {
                continue;
            }
            let Some(new_bytes) = window.event_bytes.checked_add(event.canonical_size_bytes) else {
                decisions.push((
                    anchor_id,
                    FrozenOutcome::Rejected(TrajectoryRejectionReason::Overflow),
                ));
                continue;
            };
            if window.events.len() >= window.limits.max_events
                || new_bytes > window.limits.max_bytes
            {
                decisions.push((
                    anchor_id,
                    FrozenOutcome::Rejected(TrajectoryRejectionReason::Overflow),
                ));
                continue;
            }
            window.event_bytes = new_bytes;
            window.events.push(event.clone());
            if replaying_for_window {
                window.events.sort_by_key(|captured| captured.ingest_seq);
            }

            let preset_matches = (window.limits.handoff_progress && is_handoff)
                || (window.limits.compaction_progress && is_compaction);
            let preset_progress = preset_matches && lifecycle_progress_first_seen;
            let llm_progress = registered_primary_end
                && event.event_uuid != window.identity.anchor_call_uuid
                && window.counted_primary_calls.insert(event.event_uuid);
            if llm_progress || preset_progress {
                window.observed_progress = window.observed_progress.saturating_add(1);
            }
            if window.observed_progress >= window.limits.requested_progress {
                decisions.push((
                    anchor_id,
                    FrozenOutcome::Closed(TrajectoryTrigger::ProgressReached),
                ));
            } else if is_owner_end {
                decisions.push((
                    anchor_id,
                    FrozenOutcome::Closed(TrajectoryTrigger::OwnerTerminated),
                ));
            }
        }

        for (anchor_id, outcome) in decisions {
            match outcome {
                FrozenOutcome::Closed(trigger) => self.freeze_closed(anchor_id, trigger, effects),
                FrozenOutcome::Rejected(reason) => self.freeze_rejected(anchor_id, reason, effects),
            }
        }

        if is_scope_end(&event) {
            if let Some(ancestry) = self.ancestry.get_mut(&event.event_uuid) {
                ancestry.ended = true;
            }
            if let Some(call) = self.calls.get_mut(&event.event_uuid) {
                call.ended = true;
            } else if event.category.as_deref() == Some("llm") {
                self.clear_unregistered_call(event.event_uuid, Some(root_uuid), effects);
            }
        }
        if event.scope_phase == Some(ScopeCategory::Start) {
            self.replay_staged_children(event.event_uuid, effects);
        }
    }

    fn observe_oversized(
        &mut self,
        event: ProjectedTrajectoryEvent,
        loss: LossSnapshot,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        self.observe_oversized_inner(event, loss, false, effects);
    }

    fn observe_oversized_inner(
        &mut self,
        event: ProjectedTrajectoryEvent,
        loss: LossSnapshot,
        replaying_staged: bool,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let oversized = match event {
            ProjectedTrajectoryEvent::Captured(event) => {
                self.observe_event_inner(event, loss, replaying_staged, effects);
                return;
            }
            ProjectedTrajectoryEvent::Oversized(event) => event,
        };
        let call_registered = self.calls.contains_key(&oversized.event_uuid);
        if let Some(call) = self.calls.get(&oversized.event_uuid) {
            let root_uuid = call.registration.root_uuid;
            let expected_parent = call.registration.parent_uuid;
            if oversized.parent_uuid != Some(expected_parent) {
                if oversized.scope_phase == Some(ScopeCategory::End) {
                    if let Some(ancestry) = self.ancestry.get_mut(&oversized.event_uuid) {
                        ancestry.ended = true;
                    }
                    if let Some(call) = self.calls.get_mut(&oversized.event_uuid) {
                        call.ended = true;
                    }
                }
                self.invalidate_root(
                    root_uuid,
                    TrajectoryRejectionReason::ContradictoryOwnership,
                    effects,
                );
                self.set_permanent_sampling_fault(HEALTH_OWNERSHIP_LOSS, effects);
                return;
            }
            let valid_primary_scope = oversized.kind == crate::trajectory::CapturedEventKind::Scope
                && oversized.category.as_deref() == Some("llm")
                && oversized.call_role == Some(LlmCallRole::Primary);
            if !valid_primary_scope {
                if oversized.scope_phase == Some(ScopeCategory::End) {
                    if let Some(ancestry) = self.ancestry.get_mut(&oversized.event_uuid) {
                        ancestry.ended = true;
                    }
                    if let Some(call) = self.calls.get_mut(&oversized.event_uuid) {
                        call.ended = true;
                    }
                }
                self.invalidate_root(root_uuid, TrajectoryRejectionReason::EventLoss, effects);
                return;
            }
        }
        let owner = self.resolve_oversized_owner(&oversized, replaying_staged, effects);
        let Some((root_uuid, owner_uuid)) = owner else {
            if oversized.category.as_deref() == Some("llm") && !call_registered {
                self.stage_unresolved(
                    oversized.event_uuid,
                    StagedObservation::Oversized(
                        ProjectedTrajectoryEvent::Oversized(oversized),
                        loss,
                    ),
                    effects,
                );
            } else {
                self.record_unresolved_ownership(
                    oversized.event_uuid,
                    oversized.parent_uuid,
                    oversized.ingest_seq,
                    oversized.scope_phase == Some(ScopeCategory::End),
                    effects,
                );
            }
            return;
        };
        let is_llm_end = oversized.scope_phase == Some(ScopeCategory::End)
            && oversized.category.as_deref() == Some("llm");
        if self.active_outcomes_enabled {
            effects.push(CoordinatorEffect::ObserveActiveOutcomeOversized(
                ActiveOutcomeOversizedV2 {
                    root_uuid,
                    event: oversized.clone(),
                },
            ));
        }
        let anchors = self.windows.keys().copied().collect::<Vec<_>>();
        for anchor_id in anchors {
            let Some(window) = self.windows.get_mut(&anchor_id) else {
                continue;
            };
            let replaying_for_window = replaying_staged
                && window.provisional.as_ref().is_some_and(|terminal| {
                    terminal.uncertain_uuids.contains(&oversized.event_uuid)
                        && oversized.ingest_seq <= terminal.cutoff_ingest_seq
                });
            if window.frozen.is_some()
                || (window.provisional.is_some() && !replaying_for_window)
                || window.identity.root_uuid != root_uuid
            {
                continue;
            }
            if is_llm_end && oversized.event_uuid == window.identity.anchor_call_uuid {
                if window.capture_after_ingest_seq.is_none() {
                    window.capture_after_ingest_seq = Some(oversized.ingest_seq);
                    window.highest_observed_ingest_seq =
                        window.highest_observed_ingest_seq.max(oversized.ingest_seq);
                    window.loss_at_boundary = Some(loss.highest_dropped_ingest_seq);
                    if window.accepted {
                        window.phase = WindowPhase::Collecting;
                    }
                }
                continue;
            }
            if window.capture_after_ingest_seq.is_some_and(|boundary| {
                oversized.ingest_seq > boundary && window.identity.owner_uuid == owner_uuid
            }) {
                window.highest_observed_ingest_seq =
                    window.highest_observed_ingest_seq.max(oversized.ingest_seq);
                if replaying_for_window {
                    window.provisional = None;
                }
                self.freeze_rejected(anchor_id, TrajectoryRejectionReason::Overflow, effects);
            }
        }
        if oversized.scope_phase == Some(ScopeCategory::End) {
            if let Some(ancestry) = self.ancestry.get_mut(&oversized.event_uuid) {
                ancestry.ended = true;
            }
            if let Some(call) = self.calls.get_mut(&oversized.event_uuid) {
                call.ended = true;
            } else if oversized.category.as_deref() == Some("llm") {
                self.clear_unregistered_call(oversized.event_uuid, Some(root_uuid), effects);
            }
        }
    }

    fn resolve_captured_owner(
        &mut self,
        event: &CapturedTrajectoryEvent,
        replaying_staged: bool,
        effects: &mut Vec<CoordinatorEffect>,
    ) -> Option<(Uuid, Uuid)> {
        if let Some(call) = self.calls.get(&event.event_uuid) {
            let root_uuid = call.registration.root_uuid;
            let owner_uuid = call.registration.owner_uuid;
            let expected_parent = call.registration.parent_uuid;
            if event.parent_uuid != Some(expected_parent) {
                if is_scope_end(event) {
                    if let Some(ancestry) = self.ancestry.get_mut(&event.event_uuid) {
                        ancestry.ended = true;
                    }
                    if let Some(call) = self.calls.get_mut(&event.event_uuid) {
                        call.ended = true;
                    }
                }
                self.invalidate_root(
                    root_uuid,
                    TrajectoryRejectionReason::ContradictoryOwnership,
                    effects,
                );
                self.set_permanent_sampling_fault(HEALTH_OWNERSHIP_LOSS, effects);
                return None;
            }
            if event.kind != crate::trajectory::CapturedEventKind::Scope
                || event.category.as_deref() != Some("llm")
                || event.call_role != Some(LlmCallRole::Primary)
            {
                if is_scope_end(event) {
                    if let Some(ancestry) = self.ancestry.get_mut(&event.event_uuid) {
                        ancestry.ended = true;
                    }
                    if let Some(call) = self.calls.get_mut(&event.event_uuid) {
                        call.ended = true;
                    }
                }
                self.invalidate_root(root_uuid, TrajectoryRejectionReason::EventLoss, effects);
                return None;
            }
            if event.scope_phase == Some(ScopeCategory::Start) {
                let entry = AncestryEntry {
                    parent_uuid: event.parent_uuid,
                    root_uuid,
                    owner_uuid,
                    scope_type: ScopeType::Llm,
                    ended: false,
                };
                let _ = self.insert_ancestry(event.event_uuid, entry, root_uuid, effects);
            }
            return Some((root_uuid, owner_uuid));
        }

        if event.scope_phase == Some(ScopeCategory::Start) {
            let (parent_root, parent_owner) = event
                .parent_uuid
                .and_then(|uuid| self.resolve_known_owner(uuid))?;
            let scope_type = event.scope_type.unwrap_or(ScopeType::Unknown);
            let owner_uuid = if scope_type == ScopeType::Agent {
                event.event_uuid
            } else {
                parent_owner
            };
            let entry = AncestryEntry {
                parent_uuid: event.parent_uuid,
                root_uuid: parent_root,
                owner_uuid,
                scope_type,
                ended: false,
            };
            if !self.insert_ancestry(event.event_uuid, entry, parent_root, effects) {
                return None;
            }
            self.reconcile_unresolved_registration(
                event.event_uuid,
                event.parent_uuid,
                parent_root,
                replaying_staged,
                effects,
            );
            let event_owner = if scope_type == ScopeType::Agent && is_handoff_event(event) {
                parent_owner
            } else {
                owner_uuid
            };
            return Some((parent_root, event_owner));
        }

        if let Some(entry) = self.ancestry.get(&event.event_uuid).copied() {
            let contradictory_parent = entry.parent_uuid.is_some()
                && event.parent_uuid.is_some()
                && entry.parent_uuid != event.parent_uuid;
            let contradictory_type = event.scope_type.is_some_and(|scope_type| {
                scope_type != ScopeType::Unknown && scope_type != entry.scope_type
            });
            if contradictory_parent || contradictory_type {
                self.invalidate_root(
                    entry.root_uuid,
                    TrajectoryRejectionReason::ContradictoryOwnership,
                    effects,
                );
                self.set_permanent_sampling_fault(HEALTH_OWNERSHIP_LOSS, effects);
                return None;
            }
            return Some((entry.root_uuid, entry.owner_uuid));
        }
        if is_scope_end(event)
            && (event.scope_type == Some(ScopeType::Agent)
                || event.category.as_deref() == Some("agent"))
        {
            return event
                .parent_uuid
                .and_then(|uuid| self.resolve_known_owner(uuid))
                .map(|(root_uuid, _)| (root_uuid, event.event_uuid));
        }
        event
            .parent_uuid
            .and_then(|uuid| self.resolve_known_owner(uuid))
    }

    fn resolve_known_owner(&self, uuid: Uuid) -> Option<(Uuid, Uuid)> {
        self.ancestry
            .get(&uuid)
            .map(|entry| (entry.root_uuid, entry.owner_uuid))
            .or_else(|| {
                self.calls
                    .get(&uuid)
                    .map(|call| (call.registration.root_uuid, call.registration.owner_uuid))
            })
    }

    fn resolve_oversized_owner(
        &mut self,
        event: &crate::trajectory::OversizedTrajectoryEvent,
        replaying_staged: bool,
        effects: &mut Vec<CoordinatorEffect>,
    ) -> Option<(Uuid, Uuid)> {
        if let Some(call) = self.calls.get(&event.event_uuid) {
            return Some((call.registration.root_uuid, call.registration.owner_uuid));
        }
        if event.scope_phase == Some(ScopeCategory::End)
            && let Some(entry) = self.ancestry.get(&event.event_uuid)
        {
            return Some((entry.root_uuid, entry.owner_uuid));
        }
        if event.scope_phase == Some(ScopeCategory::End)
            && event.category.as_deref() == Some("agent")
        {
            return event
                .parent_uuid
                .and_then(|uuid| self.resolve_known_owner(uuid))
                .map(|(root_uuid, _)| (root_uuid, event.event_uuid));
        }
        if event.scope_phase == Some(ScopeCategory::Start) {
            let (root_uuid, parent_owner) = event
                .parent_uuid
                .and_then(|uuid| self.resolve_known_owner(uuid))?;
            let scope_type = event
                .category
                .as_deref()
                .map(scope_type_from_category)
                .unwrap_or(ScopeType::Unknown);
            let owner_uuid = if scope_type == ScopeType::Agent {
                event.event_uuid
            } else {
                parent_owner
            };
            let entry = AncestryEntry {
                parent_uuid: event.parent_uuid,
                root_uuid,
                owner_uuid,
                scope_type,
                ended: false,
            };
            if !self.insert_ancestry(event.event_uuid, entry, root_uuid, effects) {
                return None;
            }
            self.reconcile_unresolved_registration(
                event.event_uuid,
                event.parent_uuid,
                root_uuid,
                replaying_staged,
                effects,
            );
            // Metadata is absent from an oversize marker, so an Agent start is
            // conservatively attributed to its parent in case it was a handoff.
            let event_owner = if scope_type == ScopeType::Agent {
                parent_owner
            } else {
                owner_uuid
            };
            return Some((root_uuid, event_owner));
        }
        event
            .parent_uuid
            .and_then(|uuid| self.resolve_known_owner(uuid))
            .or_else(|| {
                self.ancestry
                    .get(&event.event_uuid)
                    .map(|entry| (entry.root_uuid, entry.owner_uuid))
            })
    }

    fn stage_unresolved(
        &mut self,
        call_uuid: Uuid,
        observation: StagedObservation,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let ingest_seq = staged_ingest_seq(&observation);
        if staged_is_unregistered_llm_end(&observation) {
            self.clear_unregistered_call(call_uuid, None, effects);
            return;
        }
        if let Some(staged) = self.staged.get_mut(&call_uuid) {
            if staged.len() < STAGED_EVENTS_PER_CALL {
                staged.push(observation);
            } else {
                let first_ingest_seq = staged
                    .iter()
                    .map(staged_ingest_seq)
                    .min()
                    .unwrap_or(ingest_seq)
                    .min(ingest_seq);
                self.staged.remove(&call_uuid);
                self.record_dropped_staged(call_uuid, first_ingest_seq, effects);
            }
            return;
        }
        if self.staged.len() < self.limits.staging_capacity {
            self.staged.insert(call_uuid, vec![observation]);
        } else {
            self.record_dropped_staged(call_uuid, ingest_seq, effects);
        }
    }

    fn record_unresolved_ownership(
        &mut self,
        event_uuid: Uuid,
        parent_uuid: Option<Uuid>,
        ingest_seq: u64,
        is_end: bool,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let Some(parent_uuid) = parent_uuid else {
            return;
        };
        if let Some(existing) = self.unresolved_ownership.get(&event_uuid).copied() {
            match existing {
                UnresolvedOwnership::Parent {
                    parent_uuid: existing_parent,
                    first_ingest_seq,
                    ended,
                } if existing_parent == parent_uuid => {
                    self.unresolved_ownership.insert(
                        event_uuid,
                        UnresolvedOwnership::Parent {
                            parent_uuid,
                            first_ingest_seq: first_ingest_seq.min(ingest_seq),
                            ended: ended || is_end,
                        },
                    );
                }
                UnresolvedOwnership::Parent {
                    first_ingest_seq, ..
                }
                | UnresolvedOwnership::Contradictory { first_ingest_seq } => {
                    self.unresolved_ownership.insert(
                        event_uuid,
                        UnresolvedOwnership::Contradictory {
                            first_ingest_seq: first_ingest_seq.min(ingest_seq),
                        },
                    );
                    self.set_permanent_sampling_fault(HEALTH_OWNERSHIP_LOSS, effects);
                    let anchors = self.windows.keys().copied().collect::<Vec<_>>();
                    for anchor_id in anchors {
                        self.freeze_rejected(
                            anchor_id,
                            TrajectoryRejectionReason::ContradictoryOwnership,
                            effects,
                        );
                    }
                }
            }
            return;
        }
        if self.unresolved_ownership.len() >= self.limits.ancestry_capacity {
            self.set_permanent_sampling_fault(HEALTH_OWNERSHIP_LOSS, effects);
            let anchors = self.windows.keys().copied().collect::<Vec<_>>();
            for anchor_id in anchors {
                self.freeze_rejected(
                    anchor_id,
                    TrajectoryRejectionReason::ContradictoryOwnership,
                    effects,
                );
            }
            return;
        }
        self.unresolved_ownership.insert(
            event_uuid,
            UnresolvedOwnership::Parent {
                parent_uuid,
                first_ingest_seq: ingest_seq,
                ended: is_end,
            },
        );
    }

    fn reconcile_unresolved_registration(
        &mut self,
        uuid: Uuid,
        expected_parent: Option<Uuid>,
        root_uuid: Uuid,
        defer_uuid_uncertainty: bool,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let direct = self.unresolved_ownership.remove(&uuid);
        let direct_conflict = match direct {
            Some(UnresolvedOwnership::Contradictory { .. }) => true,
            Some(UnresolvedOwnership::Parent {
                parent_uuid, ended, ..
            }) => expected_parent.map_or_else(
                || {
                    ended
                        || self
                            .resolve_known_owner(parent_uuid)
                            .is_some_and(|(parent_root, _)| parent_root != root_uuid)
                },
                |expected| ended || expected != parent_uuid,
            ),
            None => false,
        };
        let dependent = self
            .unresolved_ownership
            .iter()
            .filter_map(|(event_uuid, ownership)| match ownership {
                UnresolvedOwnership::Parent { parent_uuid, .. } if *parent_uuid == uuid => {
                    Some(*event_uuid)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if !defer_uuid_uncertainty && !self.staged.contains_key(&uuid) {
            self.resolve_uncertainty(uuid, Some(root_uuid), effects);
        }
        for event_uuid in &dependent {
            self.unresolved_ownership.remove(event_uuid);
            self.resolve_uncertainty(*event_uuid, Some(root_uuid), effects);
        }
        if direct_conflict || !dependent.is_empty() {
            self.invalidate_root(
                root_uuid,
                TrajectoryRejectionReason::ContradictoryOwnership,
                effects,
            );
        }
    }

    fn record_dropped_staged(
        &mut self,
        call_uuid: Uuid,
        ingest_seq: u64,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        if let Some(first_ingest_seq) = self.dropped_staged_calls.get_mut(&call_uuid) {
            *first_ingest_seq = (*first_ingest_seq).min(ingest_seq);
            return;
        }
        if let Some(first_ingest_seq) = self.pressure_calls.get_mut(&call_uuid) {
            *first_ingest_seq = (*first_ingest_seq).min(ingest_seq);
            return;
        }
        if self.dropped_staged_calls.len() < self.limits.primary_call_capacity {
            self.dropped_staged_calls.insert(call_uuid, ingest_seq);
        } else if self.pressure_calls.len() < self.limits.primary_call_capacity {
            self.pressure_calls.insert(call_uuid, ingest_seq);
            effects.push(CoordinatorEffect::Health(HEALTH_STAGING_PRESSURE));
        } else {
            self.pressure_overflow_count = self.pressure_overflow_count.saturating_add(1);
            effects.push(CoordinatorEffect::Health(HEALTH_STAGING_PRESSURE));
            let anchors = self.windows.keys().copied().collect::<Vec<_>>();
            for anchor_id in anchors {
                self.freeze_rejected(anchor_id, TrajectoryRejectionReason::EventLoss, effects);
            }
        }
    }

    fn clear_unregistered_call(
        &mut self,
        call_uuid: Uuid,
        resolved_root: Option<Uuid>,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let staged_losses = self
            .staged
            .remove(&call_uuid)
            .map(|observations| {
                observations
                    .iter()
                    .map(staged_ingest_seq)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let dropped_loss = self.dropped_staged_calls.remove(&call_uuid);
        let pressure_loss = self.pressure_calls.remove(&call_uuid);
        if dropped_loss.is_none() && pressure_loss.is_none() && self.pressure_overflow_count > 0 {
            self.pressure_overflow_count -= 1;
        }
        self.resolve_uncertainty(call_uuid, resolved_root, effects);
        if let Some(root_uuid) = resolved_root {
            let affected = self
                .windows
                .iter()
                .filter_map(|(anchor_id, window)| {
                    (window.identity.root_uuid == root_uuid
                        && window.frozen.is_none()
                        && window.provisional.is_none()
                        && window.capture_after_ingest_seq.is_some_and(|boundary| {
                            staged_losses
                                .iter()
                                .any(|ingest_seq| *ingest_seq > boundary)
                                || dropped_loss.is_some_and(|ingest_seq| ingest_seq > boundary)
                                || pressure_loss.is_some_and(|ingest_seq| ingest_seq > boundary)
                        }))
                    .then_some(*anchor_id)
                })
                .collect::<Vec<_>>();
            for anchor_id in affected {
                self.freeze_rejected(anchor_id, TrajectoryRejectionReason::EventLoss, effects);
            }
        }
    }

    fn replay_staged_children(&mut self, parent_uuid: Uuid, effects: &mut Vec<CoordinatorEffect>) {
        self.replay_staged_children_except(parent_uuid, None, effects);
    }

    fn replay_staged_children_except(
        &mut self,
        parent_uuid: Uuid,
        excluded_child: Option<Uuid>,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let Some((root_uuid, _)) = self.resolve_known_owner(parent_uuid) else {
            return;
        };
        let children = self
            .staged
            .iter()
            .filter_map(|(child_uuid, observations)| {
                (Some(*child_uuid) != excluded_child
                    && !observations.is_empty()
                    && observations
                        .iter()
                        .all(|observation| staged_parent_uuid(observation) == Some(parent_uuid)))
                .then_some(*child_uuid)
            })
            .collect::<Vec<_>>();
        for child_uuid in children {
            let mut staged = self
                .staged
                .remove(&child_uuid)
                .expect("selected staged child must remain present");
            staged.sort_by_key(staged_ingest_seq);
            self.replay_staged(child_uuid, root_uuid, staged, effects);
        }
    }

    fn replay_staged(
        &mut self,
        uuid: Uuid,
        root_uuid: Uuid,
        staged: Vec<StagedObservation>,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        for observation in staged {
            if let Some((ingest_seq, reason)) =
                self.staged_registration_rejection(uuid, &observation)
            {
                self.reject_provisional_observation(uuid, root_uuid, ingest_seq, reason, effects);
            }
            match observation {
                StagedObservation::Event(event, loss) => {
                    self.observe_event_inner(event, loss, true, effects)
                }
                StagedObservation::Oversized(event, loss) => {
                    self.observe_oversized_inner(event, loss, true, effects)
                }
            }
        }
        self.settle_uncertainty(uuid, Some(root_uuid), None, effects);
    }

    fn staged_registration_rejection(
        &self,
        uuid: Uuid,
        observation: &StagedObservation,
    ) -> Option<(u64, TrajectoryRejectionReason)> {
        let (ingest_seq, parent_uuid, kind, category, call_role, scope_type) = match observation {
            StagedObservation::Event(event, _)
            | StagedObservation::Oversized(ProjectedTrajectoryEvent::Captured(event), _) => (
                event.ingest_seq,
                event.parent_uuid,
                event.kind,
                event.category.as_deref(),
                event.call_role,
                event.scope_type,
            ),
            StagedObservation::Oversized(ProjectedTrajectoryEvent::Oversized(event), _) => (
                event.ingest_seq,
                event.parent_uuid,
                event.kind,
                event.category.as_deref(),
                event.call_role,
                event.category.as_deref().map(scope_type_from_category),
            ),
        };
        if let Some(call) = self.calls.get(&uuid) {
            if parent_uuid != Some(call.registration.parent_uuid) {
                return Some((
                    ingest_seq,
                    TrajectoryRejectionReason::ContradictoryOwnership,
                ));
            }
            return (kind != crate::trajectory::CapturedEventKind::Scope
                || category != Some("llm")
                || call_role != Some(LlmCallRole::Primary))
            .then_some((ingest_seq, TrajectoryRejectionReason::EventLoss));
        }
        self.ancestry.get(&uuid).and_then(|entry| {
            let contradictory_type = scope_type.is_some_and(|scope_type| {
                scope_type != ScopeType::Unknown && scope_type != entry.scope_type
            });
            (parent_uuid != entry.parent_uuid
                || kind != crate::trajectory::CapturedEventKind::Scope
                || contradictory_type)
                .then_some((
                    ingest_seq,
                    TrajectoryRejectionReason::ContradictoryOwnership,
                ))
        })
    }

    fn reject_provisional_observation(
        &mut self,
        uuid: Uuid,
        root_uuid: Uuid,
        ingest_seq: u64,
        reason: TrajectoryRejectionReason,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let affected = self
            .windows
            .iter()
            .filter_map(|(anchor_id, window)| {
                (window.identity.root_uuid == root_uuid
                    && window.provisional.as_ref().is_some_and(|terminal| {
                        terminal.uncertain_uuids.contains(&uuid)
                            && ingest_seq <= terminal.cutoff_ingest_seq
                    }))
                .then_some(*anchor_id)
            })
            .collect::<Vec<_>>();
        for anchor_id in affected {
            let window = self
                .windows
                .get_mut(&anchor_id)
                .expect("affected provisional window must remain live");
            window.provisional = None;
            self.freeze_rejected(anchor_id, reason, effects);
        }
    }

    fn deadline_elapsed(
        &mut self,
        anchor_id: Uuid,
        generation: u64,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let Some(window) = self.windows.get(&anchor_id) else {
            return;
        };
        if window.deadline_generation != generation || window.frozen.is_some() {
            return;
        }
        if window.capture_after_ingest_seq.is_some() {
            self.freeze_closed(anchor_id, TrajectoryTrigger::DeadlineElapsed, effects);
        } else {
            let reason = if self.highest_dropped_ingest_seq > window.opened_after_ingest_seq
                && self.highest_dropped_ingest_seq > window.loss_at_open.highest_dropped_ingest_seq
            {
                TrajectoryRejectionReason::EventLoss
            } else {
                TrajectoryRejectionReason::CanceledBeforeAnchorEnd
            };
            self.freeze_rejected(anchor_id, reason, effects);
        }
    }

    fn deadline_barrier_failed(
        &mut self,
        anchor_id: Uuid,
        generation: u64,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let Some(window) = self.windows.get(&anchor_id) else {
            return;
        };
        if window.deadline_generation == generation && window.frozen.is_none() {
            if window.provisional.is_some() {
                let window = self
                    .windows
                    .get_mut(&anchor_id)
                    .expect("provisional barrier window must remain live");
                window.provisional = None;
            }
            self.freeze_rejected(
                anchor_id,
                TrajectoryRejectionReason::RejectedDeliveryBarrier,
                effects,
            );
        }
    }

    fn freeze_closed(
        &mut self,
        anchor_id: Uuid,
        trigger: TrajectoryTrigger,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let Some(window) = self.windows.get(&anchor_id) else {
            return;
        };
        if window.frozen.is_some() {
            return;
        }
        let terminal_barrier = matches!(
            trigger,
            TrajectoryTrigger::DeadlineElapsed | TrajectoryTrigger::Shutdown
        );
        if window.provisional.is_some() {
            if terminal_barrier {
                let window = self
                    .windows
                    .get_mut(&anchor_id)
                    .expect("provisional barrier window must remain live");
                window.provisional = None;
                self.freeze_rejected(anchor_id, TrajectoryRejectionReason::EventLoss, effects);
            }
            return;
        }
        let cutoff = (!terminal_barrier).then_some(window.highest_observed_ingest_seq);
        let uncertain_uuids = self.uncertain_uuids_for_window(anchor_id, cutoff);
        if self.pressure_overflow_count > 0 || (terminal_barrier && !uncertain_uuids.is_empty()) {
            self.freeze_rejected(anchor_id, TrajectoryRejectionReason::EventLoss, effects);
            return;
        }
        if matches!(
            trigger,
            TrajectoryTrigger::ProgressReached | TrajectoryTrigger::OwnerTerminated
        ) && !uncertain_uuids.is_empty()
        {
            let window = self
                .windows
                .get_mut(&anchor_id)
                .expect("provisional window must remain live");
            window.provisional = Some(ProvisionalTerminal {
                outcome: FrozenOutcome::Closed(trigger),
                cutoff_ingest_seq: window.highest_observed_ingest_seq,
                uncertain_uuids,
            });
            return;
        }
        self.freeze_outcome(anchor_id, FrozenOutcome::Closed(trigger), effects);
    }

    fn freeze_rejected(
        &mut self,
        anchor_id: Uuid,
        reason: TrajectoryRejectionReason,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        self.freeze_outcome(anchor_id, FrozenOutcome::Rejected(reason), effects);
    }

    fn freeze_outcome(
        &mut self,
        anchor_id: Uuid,
        outcome: FrozenOutcome,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let Some(window) = self.windows.get(&anchor_id) else {
            return;
        };
        if window.frozen.is_some() || window.provisional.is_some() {
            return;
        }
        let closed_at = truncate_utc_to_milliseconds((self.now_utc)());
        let events = window
            .events
            .iter()
            .map(|event| event.as_ref().clone())
            .collect();
        let payload = match outcome {
            FrozenOutcome::Closed(trigger) => PersistedTrajectoryTerminalV1::closed(
                window.pending.clone(),
                events,
                window.observed_progress,
                trigger,
                closed_at,
                Vec::new(),
            ),
            FrozenOutcome::Rejected(reason) => PersistedTrajectoryTerminalV1::rejected(
                window.pending.clone(),
                events,
                window.observed_progress,
                reason,
                closed_at,
                Vec::new(),
            ),
        };
        let payload_hash = match payload.payload_hash() {
            Ok(hash) => Some(hash),
            Err(()) => {
                self.set_permanent_sampling_fault(HEALTH_SINK_FAILURE, effects);
                None
            }
        };
        let window = self
            .windows
            .get_mut(&anchor_id)
            .expect("window checked above must remain live");
        window.provisional = None;
        window.frozen = Some(FrozenTerminal {
            outcome,
            payload,
            payload_hash,
            retry_generation: 0,
        });
        effects.push(CoordinatorEffect::CancelDeadline {
            anchor_id,
            generation: window.deadline_generation,
        });
        if window.accepted {
            window.phase = WindowPhase::PersistingTerminal;
            if window
                .frozen
                .as_ref()
                .is_some_and(|terminal| terminal.payload_hash.is_some())
            {
                self.enqueue_terminal_once(anchor_id);
            }
        }
    }

    fn uncertain_uuids_for_window(&self, anchor_id: Uuid, cutoff: Option<u64>) -> BTreeSet<Uuid> {
        let Some(window) = self.windows.get(&anchor_id) else {
            return BTreeSet::new();
        };
        let Some(boundary) = window.capture_after_ingest_seq else {
            return BTreeSet::new();
        };
        let mut uncertain = self
            .staged
            .iter()
            .filter_map(|(uuid, observations)| {
                observations
                    .iter()
                    .any(|observation| {
                        let ingest_seq = staged_ingest_seq(observation);
                        ingest_seq > boundary && cutoff.is_none_or(|cutoff| ingest_seq <= cutoff)
                    })
                    .then_some(*uuid)
            })
            .collect::<BTreeSet<_>>();
        uncertain.extend(
            self.unresolved_ownership
                .iter()
                .filter_map(|(uuid, ownership)| {
                    let ingest_seq = match ownership {
                        UnresolvedOwnership::Parent {
                            first_ingest_seq, ..
                        }
                        | UnresolvedOwnership::Contradictory { first_ingest_seq } => {
                            *first_ingest_seq
                        }
                    };
                    (ingest_seq > boundary && cutoff.is_none_or(|cutoff| ingest_seq <= cutoff))
                        .then_some(*uuid)
                }),
        );
        uncertain.extend(
            self.dropped_staged_calls
                .iter()
                .filter_map(|(uuid, ingest_seq)| {
                    (*ingest_seq > boundary && cutoff.is_none_or(|cutoff| *ingest_seq <= cutoff))
                        .then_some(*uuid)
                }),
        );
        uncertain.extend(self.pressure_calls.iter().filter_map(|(uuid, ingest_seq)| {
            (*ingest_seq > boundary && cutoff.is_none_or(|cutoff| *ingest_seq <= cutoff))
                .then_some(*uuid)
        }));
        uncertain
    }

    fn resolve_uncertainty(
        &mut self,
        uuid: Uuid,
        resolved_root: Option<Uuid>,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        self.settle_uncertainty(
            uuid,
            resolved_root,
            Some(TrajectoryRejectionReason::EventLoss),
            effects,
        );
    }

    fn settle_uncertainty(
        &mut self,
        uuid: Uuid,
        resolved_root: Option<Uuid>,
        same_root_rejection: Option<TrajectoryRejectionReason>,
        effects: &mut Vec<CoordinatorEffect>,
    ) {
        let affected = self
            .windows
            .iter()
            .filter_map(|(anchor_id, window)| {
                window
                    .provisional
                    .as_ref()
                    .is_some_and(|terminal| terminal.uncertain_uuids.contains(&uuid))
                    .then_some(*anchor_id)
            })
            .collect::<Vec<_>>();
        for anchor_id in affected {
            let same_root = resolved_root.is_some_and(|root_uuid| {
                self.windows
                    .get(&anchor_id)
                    .is_some_and(|window| window.identity.root_uuid == root_uuid)
            });
            if same_root && let Some(reason) = same_root_rejection {
                let window = self
                    .windows
                    .get_mut(&anchor_id)
                    .expect("affected provisional window must remain live");
                window.provisional = None;
                self.freeze_rejected(anchor_id, reason, effects);
                continue;
            }
            let outcome = {
                let window = self
                    .windows
                    .get_mut(&anchor_id)
                    .expect("affected provisional window must remain live");
                let terminal = window
                    .provisional
                    .as_mut()
                    .expect("affected window must remain provisional");
                terminal.uncertain_uuids.remove(&uuid);
                terminal
                    .uncertain_uuids
                    .is_empty()
                    .then_some(terminal.outcome)
            };
            if let Some(outcome) = outcome {
                let window = self
                    .windows
                    .get_mut(&anchor_id)
                    .expect("completed provisional window must remain live");
                window.provisional = None;
                self.freeze_outcome(anchor_id, outcome, effects);
            }
        }
    }

    fn close_for_shutdown(&mut self, anchor_id: Uuid, effects: &mut Vec<CoordinatorEffect>) {
        let Some(window) = self.windows.get(&anchor_id) else {
            return;
        };
        if window.frozen.is_some() {
            return;
        }
        if window.capture_after_ingest_seq.is_some() {
            self.freeze_closed(anchor_id, TrajectoryTrigger::Shutdown, effects);
        } else {
            let reason = if self.highest_dropped_ingest_seq > window.opened_after_ingest_seq
                && self.highest_dropped_ingest_seq > window.loss_at_open.highest_dropped_ingest_seq
            {
                TrajectoryRejectionReason::EventLoss
            } else {
                TrajectoryRejectionReason::CanceledBeforeAnchorEnd
            };
            self.freeze_rejected(anchor_id, reason, effects);
        }
    }

    fn shutdown(&mut self, admission_epoch: u64, effects: &mut Vec<CoordinatorEffect>) {
        if self.shutdown {
            return;
        }
        self.accepting = false;
        self.shutdown = true;
        self.admission_epoch = self.admission_epoch.max(admission_epoch);

        let undispatched = self
            .windows
            .iter()
            .filter_map(|(anchor_id, window)| {
                (window.phase == WindowPhase::PendingQueued).then_some(*anchor_id)
            })
            .collect::<Vec<_>>();
        for anchor_id in undispatched {
            effects.push(CoordinatorEffect::AnchorRefused {
                anchor_id,
                reason: AnchorRefusalReason::IntakeClosed,
            });
            self.release_window(anchor_id, effects);
        }

        let remaining = self.windows.keys().copied().collect::<Vec<_>>();
        for anchor_id in remaining {
            let phase = self
                .windows
                .get(&anchor_id)
                .map(|window| window.phase)
                .expect("remaining shutdown anchor must stay live");
            if matches!(
                phase,
                WindowPhase::PendingInFlight
                    | WindowPhase::AwaitingAnchorEnd
                    | WindowPhase::Collecting
            ) {
                self.close_for_shutdown(anchor_id, effects);
            }
        }
    }

    fn release_window(&mut self, anchor_id: Uuid, effects: &mut Vec<CoordinatorEffect>) {
        let Some(window) = self.windows.remove(&anchor_id) else {
            return;
        };
        self.sink_queue.retain(|work| work.anchor_id != anchor_id);
        self.delivery_queue
            .retain(|queued_anchor| *queued_anchor != anchor_id);
        if self
            .sink_inflight
            .is_some_and(|work| work.anchor_id == anchor_id)
        {
            self.sink_inflight = None;
        }
        if self.delivery_inflight == Some(anchor_id) {
            self.delivery_inflight = None;
        }
        if let Some(count) = self.pending_by_pool.get_mut(&window.identity.pool_id) {
            *count = count.saturating_sub(1);
        }
        effects.push(CoordinatorEffect::CancelDeadline {
            anchor_id,
            generation: window.deadline_generation,
        });
        self.tombstones.push_back(anchor_id);
        while self.tombstones.len() > self.limits.tombstone_capacity {
            self.tombstones.pop_front();
        }
    }

    fn prune_indexes(&mut self) {
        let active_roots = self
            .windows
            .values()
            .map(|window| window.identity.root_uuid)
            .collect::<BTreeSet<_>>();
        self.calls
            .retain(|_, call| !call.ended || active_roots.contains(&call.registration.root_uuid));

        let call_roots = self
            .calls
            .values()
            .map(|call| call.registration.root_uuid)
            .collect::<BTreeSet<_>>();
        let parent_refs = self
            .ancestry
            .values()
            .filter_map(|entry| entry.parent_uuid)
            .collect::<BTreeSet<_>>();
        self.ancestry.retain(|uuid, entry| {
            !entry.ended
                || active_roots.contains(&entry.root_uuid)
                || call_roots.contains(&entry.root_uuid)
                || parent_refs.contains(uuid)
        });

        let live_roots = self
            .windows
            .values()
            .map(|window| window.identity.root_uuid)
            .chain(
                self.calls
                    .values()
                    .filter(|call| !call.ended)
                    .map(|call| call.registration.root_uuid),
            )
            .chain(
                self.ancestry
                    .values()
                    .filter(|entry| !entry.ended)
                    .map(|entry| entry.root_uuid),
            )
            .collect::<BTreeSet<_>>();
        self.seen_lifecycle_progress
            .retain(|root_uuid, _| live_roots.contains(root_uuid));
        self.lifecycle_saturated_roots
            .retain(|root_uuid| live_roots.contains(root_uuid));
        if self.windows.is_empty() {
            self.unresolved_ownership.clear();
        } else {
            self.unresolved_ownership.retain(|_, ownership| {
                let UnresolvedOwnership::Parent {
                    first_ingest_seq,
                    ended: true,
                    ..
                } = ownership
                else {
                    return true;
                };
                self.windows.values().any(|window| {
                    window
                        .capture_after_ingest_seq
                        .is_none_or(|boundary| *first_ingest_seq > boundary)
                })
            });
        }
    }

    fn maybe_emit_drained(&mut self, effects: &mut Vec<CoordinatorEffect>) {
        if self.shutdown
            && !self.drained_emitted
            && self.windows.is_empty()
            && self.sink_queue.is_empty()
            && self.sink_inflight.is_none()
            && self.delivery_queue.is_empty()
            && self.delivery_inflight.is_none()
        {
            self.drained_emitted = true;
            effects.push(CoordinatorEffect::Drained);
        }
    }
}

fn valid_frozen_path(registration: &PrimaryCallRegistration) -> bool {
    !registration.owner_path.is_empty()
        && registration.owner_path.first().map(|scope| scope.uuid) == Some(registration.owner_uuid)
        && registration.owner_path.last().map(|scope| scope.uuid) == Some(registration.parent_uuid)
        && registration
            .owner_path
            .iter()
            .map(|scope| scope.uuid)
            .collect::<BTreeSet<_>>()
            .len()
            == registration.owner_path.len()
}

fn pending_identity(pending: &PendingTrajectoryWindow) -> WindowIdentity {
    WindowIdentity {
        anchor_id: pending.anchor_id,
        anchor_call_uuid: pending.anchor_call_uuid,
        root_uuid: pending.root_uuid,
        owner_uuid: pending.owner_uuid,
        pool_id: pending.pool_id.clone(),
        requested_progress: pending.requested_progress,
    }
}

fn canonical_hash(pending: &PendingTrajectoryWindow) -> Result<String, ()> {
    pending.payload_hash()
}

fn sink_ack_anchor_id(ack: &SinkAck) -> Uuid {
    match ack {
        SinkAck::Applied { anchor_id, .. }
        | SinkAck::AlreadyApplied { anchor_id, .. }
        | SinkAck::AlreadyTerminal { anchor_id, .. }
        | SinkAck::DurablyDeclined { anchor_id, .. }
        | SinkAck::TransientRefused { anchor_id, .. }
        | SinkAck::Conflict { anchor_id }
        | SinkAck::Failed { anchor_id, .. } => *anchor_id,
    }
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn is_scope_end(event: &CapturedTrajectoryEvent) -> bool {
    event.scope_phase == Some(ScopeCategory::End)
}

fn scope_type_from_category(category: &str) -> ScopeType {
    match category {
        "agent" => ScopeType::Agent,
        "function" => ScopeType::Function,
        "tool" => ScopeType::Tool,
        "llm" => ScopeType::Llm,
        "retriever" => ScopeType::Retriever,
        "embedder" => ScopeType::Embedder,
        "reranker" => ScopeType::Reranker,
        "guardrail" => ScopeType::Guardrail,
        "evaluator" => ScopeType::Evaluator,
        "custom" => ScopeType::Custom,
        _ => ScopeType::Unknown,
    }
}

fn is_handoff_event(event: &CapturedTrajectoryEvent) -> bool {
    event.is_subagent_handoff()
}

fn is_compaction_event(event: &CapturedTrajectoryEvent) -> bool {
    if event.kind != crate::trajectory::CapturedEventKind::Mark {
        return false;
    }
    event.name == "compaction"
        || event
            .hook_event_name()
            .is_some_and(|name| matches!(name, "precompact" | "postcompact" | "compaction"))
}

fn captured_dedupe_key(event: &CapturedTrajectoryEvent) -> EventDedupeKey {
    EventDedupeKey {
        uuid: event.event_uuid,
        kind: format!("{:?}", event.kind),
        phase: event.scope_phase.map(|phase| format!("{phase:?}")),
        payload_hash: event.canonical_payload_hash.clone(),
    }
}

fn staged_ingest_seq(observation: &StagedObservation) -> u64 {
    match observation {
        StagedObservation::Event(event, _) => event.ingest_seq,
        StagedObservation::Oversized(ProjectedTrajectoryEvent::Captured(event), _) => {
            event.ingest_seq
        }
        StagedObservation::Oversized(ProjectedTrajectoryEvent::Oversized(event), _) => {
            event.ingest_seq
        }
    }
}

fn staged_parent_uuid(observation: &StagedObservation) -> Option<Uuid> {
    match observation {
        StagedObservation::Event(event, _)
        | StagedObservation::Oversized(ProjectedTrajectoryEvent::Captured(event), _) => {
            event.parent_uuid
        }
        StagedObservation::Oversized(ProjectedTrajectoryEvent::Oversized(event), _) => {
            event.parent_uuid
        }
    }
}

fn staged_is_unregistered_llm_end(observation: &StagedObservation) -> bool {
    match observation {
        StagedObservation::Event(event, _) => {
            is_scope_end(event) && event.category.as_deref() == Some("llm")
        }
        StagedObservation::Oversized(ProjectedTrajectoryEvent::Captured(event), _) => {
            is_scope_end(event) && event.category.as_deref() == Some("llm")
        }
        StagedObservation::Oversized(ProjectedTrajectoryEvent::Oversized(event), _) => {
            event.scope_phase == Some(ScopeCategory::End)
                && event.category.as_deref() == Some("llm")
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use chrono::{DateTime, Utc};
    use nemo_relay::api::llm::{
        LlmApiFamily, LlmAttributes, LlmCallRole, LlmExecutionContextSnapshot, LlmRequest,
        LlmTrajectoryScopeSnapshot,
    };
    use nemo_relay::api::runtime::{
        LLM_REPLAY_CONTRACT_VERSION, LlmReplayCall, LlmReplayCapability, LlmReplayTransport,
    };
    use nemo_relay::api::scope::ScopeType;
    use serde_json::json;

    use super::*;
    use crate::adapter::FamilyAdapter;
    use crate::config::{CanonicalizerConfig, PoolSelectorConfig};
    use crate::projection::{project_request, project_routing_context};
    use crate::sink::{
        DeliveryFailureClass, DurableDeclineReason, SinkFailureClass, TransientRefusalReason,
    };
    use crate::trajectory::{
        CAPTURED_EVENT_SCHEMA_V1, CapturedCodecAnnotationsV1, CapturedEventKind,
        OversizedTrajectoryEvent, PENDING_TRAJECTORY_SCHEMA_V1, ReplayCapabilityFactsV1,
        TrajectoryOwnerScopeV1, TrajectoryTerminalStateV1, project_anchor_response,
    };

    const POOL: &str = "pool";

    #[derive(Clone, Copy)]
    struct Ids {
        root: Uuid,
        owner: Uuid,
        anchor_call: Uuid,
        anchor: Uuid,
    }

    struct TestReplay {
        capability: LlmReplayCapability,
        starts: AtomicUsize,
    }

    impl TestReplay {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                capability: LlmReplayCapability {
                    contract_version: LLM_REPLAY_CONTRACT_VERSION,
                    api_family: LlmApiFamily::OpenAIChatCompletions,
                    transport_identity: "coordinator-test".into(),
                },
                starts: AtomicUsize::new(0),
            })
        }
    }

    impl LlmReplayTransport for TestReplay {
        fn capability(&self) -> &LlmReplayCapability {
            &self.capability
        }

        fn start(&self, _request: LlmRequest) -> nemo_relay::error::Result<LlmReplayCall> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            unreachable!("the Spec 04 coordinator must never start replay")
        }
    }

    fn uuid(value: u128) -> Uuid {
        Uuid::from_u128(value)
    }

    fn fixed_time() -> DateTime<Utc> {
        DateTime::parse_from_rfc3339("2026-07-08T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc)
    }

    fn ids(offset: u128) -> Ids {
        Ids {
            root: uuid(offset + 1),
            owner: uuid(offset + 2),
            anchor_call: uuid(offset + 3),
            anchor: uuid(offset + 4),
        }
    }

    fn context(ids: Ids, call_uuid: Uuid) -> LlmExecutionContextSnapshot {
        LlmExecutionContextSnapshot {
            call_uuid,
            root_uuid: ids.root,
            parent_uuid: ids.owner,
            trajectory_owner_uuid: ids.owner,
            trajectory_owner_path: vec![LlmTrajectoryScopeSnapshot {
                uuid: ids.owner,
                name: "agent".into(),
                scope_type: ScopeType::Agent,
            }],
            api_family: LlmApiFamily::OpenAIChatCompletions,
            call_role: LlmCallRole::Primary,
            attributes: LlmAttributes::empty(),
            tenant_id: None,
            agent_id: None,
            sanitized_metadata: BTreeMap::new(),
        }
    }

    fn primary_registration(ids: Ids, call_uuid: Uuid) -> PrimaryCallRegistration {
        PrimaryCallRegistration {
            call_uuid,
            root_uuid: ids.root,
            parent_uuid: ids.owner,
            owner_uuid: ids.owner,
            owner_path: vec![TrajectoryOwnerScopeV1 {
                uuid: ids.owner,
                name: "agent".into(),
                scope_type: ScopeType::Agent,
            }],
            api_family: LlmApiFamily::OpenAIChatCompletions,
        }
    }

    fn nested_primary_registration(
        ids: Ids,
        call_uuid: Uuid,
        parent_uuid: Uuid,
    ) -> PrimaryCallRegistration {
        PrimaryCallRegistration {
            call_uuid,
            root_uuid: ids.root,
            parent_uuid,
            owner_uuid: ids.owner,
            owner_path: vec![
                TrajectoryOwnerScopeV1 {
                    uuid: ids.owner,
                    name: "agent".into(),
                    scope_type: ScopeType::Agent,
                },
                TrajectoryOwnerScopeV1 {
                    uuid: parent_uuid,
                    name: "function".into(),
                    scope_type: ScopeType::Function,
                },
            ],
            api_family: LlmApiFamily::OpenAIChatCompletions,
        }
    }

    fn seed(ids: Ids, requested_progress: usize, replay: Arc<TestReplay>) -> TrajectoryWindowSeed {
        let context = context(ids, ids.anchor_call);
        let request = LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({
                "model": "anchor",
                "messages": [{"role": "user", "content": "hello"}]
            }),
        };
        let envelope = FamilyAdapter
            .decode(LlmApiFamily::OpenAIChatCompletions, &request)
            .unwrap();
        let canonicalizer = CanonicalizerConfig::default();
        let request_projection = project_request(&envelope, &canonicalizer).unwrap();
        let routing_context_projection =
            project_routing_context(&context, &PoolSelectorConfig::default(), &canonicalizer)
                .unwrap();
        let normalized_anchor_response = project_anchor_response(
            LlmApiFamily::OpenAIChatCompletions,
            &json!({
                "id": "response-1",
                "model": "anchor",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "ok"},
                    "finish_reason": "stop"
                }]
            }),
            1 << 20,
        )
        .unwrap();
        let replay_capability_facts =
            ReplayCapabilityFactsV1::from_capability(replay.capability()).unwrap();
        let pending = PendingTrajectoryWindow {
            schema: PENDING_TRAJECTORY_SCHEMA_V1.into(),
            anchor_id: ids.anchor,
            anchor_call_uuid: ids.anchor_call,
            root_uuid: ids.root,
            owner_uuid: ids.owner,
            owner_path: vec![TrajectoryOwnerScopeV1 {
                uuid: ids.owner,
                name: "agent".into(),
                scope_type: ScopeType::Agent,
            }],
            pool_id: POOL.into(),
            anchor_model_revision: "anchor-r1".into(),
            process_instance_id: uuid(1000),
            project_uuid: uuid(1001),
            project_id: "project".into(),
            config_generation_id: "config-generation".into(),
            policy_version_id: "policy-version".into(),
            learning_generation_id: uuid(1002),
            request_projection,
            routing_context_projection,
            normalized_anchor_response,
            replay_capability_facts,
            candidate_facts: Vec::new(),
            requested_progress,
            opened_at: fixed_time(),
            deadline_at: fixed_time() + chrono::Duration::seconds(300),
        };
        TrajectoryWindowSeed::new(pending, envelope, replay, Vec::new())
    }

    fn coordinator(max_pending: usize) -> Coordinator {
        Coordinator::new_with_clock(
            BTreeMap::from([(POOL.to_string(), max_pending)]),
            Arc::new(fixed_time),
        )
    }

    fn register_primary(actor: &mut Coordinator, ids: Ids, call_uuid: Uuid) {
        actor.handle(CoordinatorCommand::RegisterPrimaryCall {
            registration: primary_registration(ids, call_uuid),
            loss: LossSnapshot::default(),
            admission_epoch: 0,
        });
    }

    fn register_anchor(
        actor: &mut Coordinator,
        ids: Ids,
        requested_progress: usize,
        max_events: usize,
        presets: &[&str],
        replay: Arc<TestReplay>,
    ) -> (String, u64) {
        register_primary(actor, ids, ids.anchor_call);
        let effects = actor.handle(CoordinatorCommand::RegisterAnchor(Box::new(
            AnchorRegistration {
                seed: seed(ids, requested_progress, replay),
                proposal_permit: None,
                monotonic_deadline: Instant::now() + Duration::from_secs(300),
                opened_after_ingest_seq: 0,
                loss: LossSnapshot::default(),
                admission_epoch: 0,
                limits: WindowLimits::new(
                    requested_progress,
                    max_events,
                    1 << 20,
                    &presets
                        .iter()
                        .map(|value| (*value).to_string())
                        .collect::<Vec<_>>(),
                ),
            },
        )));
        let pending_hash = effects
            .iter()
            .find_map(|effect| match effect {
                CoordinatorEffect::RecordPending { payload_hash, .. } => Some(payload_hash.clone()),
                _ => None,
            })
            .expect("registration must emit the first pending write");
        let deadline_generation = effects
            .iter()
            .find_map(|effect| match effect {
                CoordinatorEffect::ScheduleDeadline { generation, .. } => Some(*generation),
                _ => None,
            })
            .expect("registration must emit its deadline");
        (pending_hash, deadline_generation)
    }

    fn ack_pending(actor: &mut Coordinator, anchor_id: Uuid, payload_hash: String) {
        actor.handle(CoordinatorCommand::PendingRecorded(SinkAck::Applied {
            anchor_id,
            payload_hash,
        }));
    }

    #[allow(clippy::too_many_arguments)]
    fn event(
        ingest_seq: u64,
        event_uuid: Uuid,
        parent_uuid: Option<Uuid>,
        kind: CapturedEventKind,
        scope_phase: Option<ScopeCategory>,
        category: Option<&str>,
        call_role: Option<LlmCallRole>,
        name: &str,
        scope_type: Option<ScopeType>,
        metadata: Option<serde_json::Value>,
        payload_hash: &str,
    ) -> Arc<CapturedTrajectoryEvent> {
        Arc::new(CapturedTrajectoryEvent {
            schema: CAPTURED_EVENT_SCHEMA_V1.into(),
            ingest_seq,
            event_uuid,
            parent_uuid,
            kind,
            scope_phase,
            category: category.map(str::to_string),
            call_role,
            timestamp: fixed_time(),
            name: name.into(),
            data: None,
            metadata,
            data_schema: None,
            scope_type,
            safe_scope_attributes: Vec::new(),
            codec_annotations: CapturedCodecAnnotationsV1::default(),
            canonical_payload_hash: payload_hash.into(),
            canonical_size_bytes: 16,
        })
    }

    fn llm_end(seq: u64, call_uuid: Uuid, owner: Uuid, hash: &str) -> Arc<CapturedTrajectoryEvent> {
        event(
            seq,
            call_uuid,
            Some(owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::End),
            Some("llm"),
            Some(LlmCallRole::Primary),
            "llm",
            Some(ScopeType::Llm),
            None,
            hash,
        )
    }

    fn observed(event: Arc<CapturedTrajectoryEvent>) -> CoordinatorCommand {
        CoordinatorCommand::ObservedEvent {
            event,
            loss: LossSnapshot::default(),
        }
    }

    fn terminal_effect(effects: Vec<CoordinatorEffect>) -> (String, PersistedTrajectoryTerminalV1) {
        effects
            .into_iter()
            .find_map(|effect| match effect {
                CoordinatorEffect::RecordTerminal {
                    payload_hash,
                    payload,
                    ..
                } => Some((payload_hash, payload)),
                _ => None,
            })
            .expect("a terminal write must be emitted")
    }

    #[test]
    fn derived_bounds_are_saturating_and_clamped() {
        let minimum = CoordinatorLimits::from_total_max_pending(0);
        assert_eq!(minimum.command_capacity, MIN_COMMAND_CAPACITY);
        assert_eq!(minimum.primary_call_capacity, 512);
        assert_eq!(minimum.ancestry_capacity, 1024);
        assert_eq!(minimum.tombstone_capacity, 1024);

        let maximum = CoordinatorLimits::from_total_max_pending(usize::MAX);
        assert_eq!(maximum.command_capacity, MAX_COMMAND_CAPACITY);
        assert_eq!(maximum.primary_call_capacity, 131_072);
        assert_eq!(maximum.ancestry_capacity, 262_144);
    }

    #[test]
    fn pending_ack_gates_terminal_and_physical_progress_counts_once() {
        let ids = ids(10);
        let replay = TestReplay::new();
        let mut actor = coordinator(2);
        let (pending_hash, _) =
            register_anchor(&mut actor, ids, 2, 16, &["compaction"], replay.clone());

        assert!(
            actor
                .handle(observed(llm_end(1, ids.anchor_call, ids.owner, "anchor")))
                .is_empty()
        );
        let later_call = uuid(20);
        register_primary(&mut actor, ids, later_call);
        assert!(
            actor
                .handle(observed(llm_end(2, later_call, ids.owner, "first")))
                .is_empty()
        );
        assert!(
            actor
                .handle(observed(llm_end(3, later_call, ids.owner, "changed")))
                .is_empty()
        );
        let compaction = event(
            4,
            uuid(21),
            Some(ids.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "compaction",
            None,
            None,
            "compaction",
        );
        let effects = actor.handle(observed(compaction));
        assert!(
            !effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::RecordTerminal { .. }))
        );
        assert_eq!(
            actor.snapshot().phases[&ids.anchor],
            WindowPhase::PendingInFlight
        );

        let effects = actor.handle(CoordinatorCommand::PendingRecorded(SinkAck::Applied {
            anchor_id: ids.anchor,
            payload_hash: pending_hash,
        }));
        let (terminal_hash, terminal) = terminal_effect(effects);
        assert_eq!(terminal.observed_progress, 2);
        assert_eq!(terminal.events.len(), 3);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached
            }
        );

        let effects = actor.handle(CoordinatorCommand::TerminalRecorded(SinkAck::Applied {
            anchor_id: ids.anchor,
            payload_hash: terminal_hash,
        }));
        assert!(effects.iter().any(|effect| matches!(
            effect,
            CoordinatorEffect::DeliverWindow { anchor_id, .. } if *anchor_id == ids.anchor
        )));
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);

        actor.handle(CoordinatorCommand::WindowDelivered(
            DeliveryAck::Delivered {
                anchor_id: ids.anchor,
            },
        ));
        assert_eq!(actor.snapshot().active_windows, 0);
        assert_eq!(actor.snapshot().pending_by_pool[POOL], 0);
        assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn handoff_and_compaction_count_but_child_work_is_excluded() {
        let ids = ids(100);
        let replay = TestReplay::new();
        let mut actor = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut actor, ids, 2, 16, &["handoff", "compaction"], replay);
        ack_pending(&mut actor, ids.anchor, pending_hash);
        actor.handle(observed(llm_end(1, ids.anchor_call, ids.owner, "anchor")));

        let child = uuid(110);
        actor.handle(observed(event(
            2,
            child,
            Some(ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("agent"),
            None,
            "child",
            Some(ScopeType::Agent),
            Some(json!({"nemo_relay_scope_role": "subagent"})),
            "handoff",
        )));
        actor.handle(observed(event(
            3,
            uuid(111),
            Some(child),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "child-work",
            None,
            None,
            "child-work",
        )));
        let effects = actor.handle(observed(event(
            4,
            uuid(112),
            Some(ids.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "hook",
            None,
            Some(json!({"hook_event_name": "postcompact"})),
            "compaction",
        )));
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(terminal.observed_progress, 2);
        assert_eq!(
            terminal
                .events
                .iter()
                .map(|event| event.name.as_str())
                .collect::<Vec<_>>(),
            ["child", "hook"]
        );
    }

    #[test]
    fn handoff_requires_an_exact_agent_scope_start() {
        let metadata = || Some(json!({"nemo_relay_scope_role": "subagent"}));
        let exact = event(
            1,
            uuid(113),
            Some(uuid(114)),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("agent"),
            None,
            "child",
            Some(ScopeType::Agent),
            metadata(),
            "exact",
        );
        assert!(is_handoff_event(&exact));

        let spoofed = [
            event(
                2,
                uuid(115),
                Some(uuid(114)),
                CapturedEventKind::Mark,
                None,
                None,
                None,
                "mark",
                None,
                metadata(),
                "mark",
            ),
            event(
                3,
                uuid(116),
                Some(uuid(114)),
                CapturedEventKind::Scope,
                Some(ScopeCategory::Start),
                Some("tool"),
                None,
                "tool",
                Some(ScopeType::Tool),
                metadata(),
                "tool",
            ),
            event(
                4,
                uuid(117),
                Some(uuid(114)),
                CapturedEventKind::Scope,
                Some(ScopeCategory::Start),
                Some("function"),
                None,
                "function",
                Some(ScopeType::Function),
                metadata(),
                "function",
            ),
            event(
                5,
                uuid(118),
                Some(uuid(114)),
                CapturedEventKind::Scope,
                Some(ScopeCategory::End),
                Some("agent"),
                None,
                "child",
                Some(ScopeType::Agent),
                metadata(),
                "end",
            ),
            event(
                6,
                uuid(119),
                Some(uuid(114)),
                CapturedEventKind::Scope,
                Some(ScopeCategory::Start),
                Some("tool"),
                None,
                "inconsistent-agent",
                Some(ScopeType::Agent),
                metadata(),
                "category",
            ),
        ];
        assert!(spoofed.iter().all(|event| !is_handoff_event(event)));
    }

    #[test]
    fn spoofed_handoff_metadata_is_retained_without_advancing_progress() {
        let ids = ids(200);
        let mut actor = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut actor, ids, 1, 16, &["handoff"], TestReplay::new());
        ack_pending(&mut actor, ids.anchor, pending_hash);
        actor.handle(observed(llm_end(1, ids.anchor_call, ids.owner, "anchor")));

        for (seq, kind, phase, category, scope_type) in [
            (2, CapturedEventKind::Mark, None, None, None),
            (
                3,
                CapturedEventKind::Scope,
                Some(ScopeCategory::Start),
                Some("tool"),
                Some(ScopeType::Tool),
            ),
            (
                4,
                CapturedEventKind::Scope,
                Some(ScopeCategory::Start),
                Some("function"),
                Some(ScopeType::Function),
            ),
        ] {
            let effects = actor.handle(observed(event(
                seq,
                uuid(220 + u128::from(seq)),
                Some(ids.owner),
                kind,
                phase,
                category,
                None,
                "spoof",
                scope_type,
                Some(json!({"nemo_relay_scope_role": "subagent"})),
                "spoof",
            )));
            assert!(
                !effects
                    .iter()
                    .any(|effect| matches!(effect, CoordinatorEffect::RecordTerminal { .. }))
            );
            assert_eq!(
                actor.snapshot().phases[&ids.anchor],
                WindowPhase::Collecting
            );
        }

        let effects = actor.handle(observed(event(
            5,
            uuid(230),
            Some(ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("agent"),
            None,
            "child",
            Some(ScopeType::Agent),
            Some(json!({"nemo_relay_scope_role": "subagent"})),
            "handoff",
        )));
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(terminal.observed_progress, 1);
        assert_eq!(terminal.events.len(), 4);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached
            }
        );
    }

    #[test]
    fn lifecycle_progress_is_first_seen_across_overlapping_windows() {
        let first = ids(120);
        let second = Ids {
            root: first.root,
            owner: first.owner,
            anchor_call: uuid(130),
            anchor: uuid(131),
        };
        let mut actor = coordinator(2);
        let (first_hash, _) =
            register_anchor(&mut actor, first, 2, 16, &["compaction"], TestReplay::new());
        ack_pending(&mut actor, first.anchor, first_hash);
        actor.handle(observed(llm_end(
            1,
            first.anchor_call,
            first.owner,
            "first-anchor",
        )));
        let lifecycle_uuid = uuid(132);
        actor.handle(observed(event(
            2,
            lifecycle_uuid,
            Some(first.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "compaction",
            None,
            None,
            "same-lifecycle",
        )));

        let (second_hash, _) = register_anchor(
            &mut actor,
            second,
            1,
            16,
            &["compaction"],
            TestReplay::new(),
        );
        ack_pending(&mut actor, second.anchor, second_hash);
        actor.handle(observed(llm_end(
            3,
            second.anchor_call,
            second.owner,
            "second-anchor",
        )));
        let effects = actor.handle(observed(event(
            4,
            lifecycle_uuid,
            Some(first.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "compaction",
            None,
            None,
            "same-lifecycle",
        )));
        assert!(
            effects
                .iter()
                .all(|effect| !matches!(effect, CoordinatorEffect::RecordTerminal { .. }))
        );
        assert_eq!(actor.windows[&first.anchor].observed_progress, 2);
        assert_eq!(actor.windows[&second.anchor].observed_progress, 0);
    }

    #[test]
    fn overlapping_windows_share_events_and_pool_capacity_never_evicts() {
        let first = ids(150);
        let second = Ids {
            root: first.root,
            owner: first.owner,
            anchor_call: uuid(160),
            anchor: uuid(161),
        };
        let mut actor = coordinator(2);
        let (first_hash, _) = register_anchor(&mut actor, first, 3, 16, &[], TestReplay::new());
        ack_pending(&mut actor, first.anchor, first_hash);
        actor.handle(observed(llm_end(
            1,
            first.anchor_call,
            first.owner,
            "first-anchor",
        )));
        let (second_hash, _) = register_anchor(&mut actor, second, 3, 16, &[], TestReplay::new());
        ack_pending(&mut actor, second.anchor, second_hash);
        actor.handle(observed(llm_end(
            2,
            second.anchor_call,
            second.owner,
            "second-anchor",
        )));
        let shared = event(
            3,
            uuid(162),
            Some(first.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "shared",
            None,
            None,
            "shared",
        );
        actor.handle(observed(shared));
        assert!(Arc::ptr_eq(
            actor.windows[&first.anchor].events.last().unwrap(),
            actor.windows[&second.anchor].events.last().unwrap()
        ));

        let full_ids = ids(170);
        let mut full = coordinator(1);
        let (_hash, _) = register_anchor(&mut full, full_ids, 1, 16, &[], TestReplay::new());
        let refused = Ids {
            root: full_ids.root,
            owner: full_ids.owner,
            anchor_call: uuid(180),
            anchor: uuid(181),
        };
        register_primary(&mut full, refused, refused.anchor_call);
        let effects = full.handle(CoordinatorCommand::RegisterAnchor(Box::new(
            AnchorRegistration {
                seed: seed(refused, 1, TestReplay::new()),
                proposal_permit: None,
                monotonic_deadline: Instant::now() + Duration::from_secs(300),
                opened_after_ingest_seq: 0,
                loss: LossSnapshot::default(),
                admission_epoch: 0,
                limits: WindowLimits::new(1, 16, 1 << 20, &[]),
            },
        )));
        assert!(effects.iter().any(|effect| matches!(
            effect,
            CoordinatorEffect::AnchorRefused {
                anchor_id,
                reason: AnchorRefusalReason::PoolFull,
            } if *anchor_id == refused.anchor
        )));
        assert_eq!(full.snapshot().active_windows, 1);
        assert_eq!(full.snapshot().pending_by_pool[POOL], 1);
    }

    #[test]
    fn ended_primary_duplicate_is_evidence_but_not_progress_for_new_window() {
        let first = ids(185);
        let second = Ids {
            root: first.root,
            owner: first.owner,
            anchor_call: uuid(190),
            anchor: uuid(191),
        };
        let mut actor = coordinator(2);
        let (first_hash, _) = register_anchor(&mut actor, first, 5, 16, &[], TestReplay::new());
        ack_pending(&mut actor, first.anchor, first_hash);
        actor.handle(observed(llm_end(
            1,
            first.anchor_call,
            first.owner,
            "first-anchor",
        )));
        let old_call = uuid(192);
        register_primary(&mut actor, first, old_call);
        actor.handle(observed(llm_end(2, old_call, first.owner, "old-first")));

        let (second_hash, _) = register_anchor(&mut actor, second, 1, 16, &[], TestReplay::new());
        ack_pending(&mut actor, second.anchor, second_hash);
        actor.handle(observed(llm_end(
            3,
            second.anchor_call,
            second.owner,
            "second-anchor",
        )));
        let effects = actor.handle(observed(llm_end(4, old_call, second.owner, "old-replayed")));
        assert!(
            !effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::RecordTerminal { .. }))
        );
        assert_eq!(actor.windows[&second.anchor].observed_progress, 0);
        assert_eq!(actor.windows[&second.anchor].events.len(), 1);

        let fresh_call = uuid(193);
        register_primary(&mut actor, second, fresh_call);
        let effects = actor.handle(observed(llm_end(5, fresh_call, second.owner, "fresh")));
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(terminal.observed_progress, 1);
        assert_eq!(terminal.events.len(), 2);
    }

    #[tokio::test(start_paused = true)]
    async fn stale_deadline_is_ignored_and_barrier_confirmed_deadline_is_partial() {
        let ids = ids(200);
        let replay = TestReplay::new();
        let mut actor = coordinator(1);
        let (pending_hash, generation) = register_anchor(&mut actor, ids, 2, 16, &[], replay);
        ack_pending(&mut actor, ids.anchor, pending_hash);
        actor.handle(observed(llm_end(1, ids.anchor_call, ids.owner, "anchor")));

        tokio::time::advance(Duration::from_secs(300)).await;
        let stale = actor.handle(CoordinatorCommand::DeadlineElapsed {
            anchor_id: ids.anchor,
            generation: generation + 1,
            loss: LossSnapshot::default(),
        });
        assert!(stale.is_empty());
        let effects = actor.handle(CoordinatorCommand::DeadlineElapsed {
            anchor_id: ids.anchor,
            generation,
            loss: LossSnapshot::default(),
        });
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::DeadlineElapsed
            }
        );
        assert!(terminal.is_partial);
        assert!(!terminal.promotion_eligible);
    }

    #[test]
    fn oversized_events_and_failed_deadline_barriers_are_explicit_rejections() {
        let oversized_ids = ids(250);
        let mut oversized = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut oversized, oversized_ids, 2, 16, &[], TestReplay::new());
        ack_pending(&mut oversized, oversized_ids.anchor, pending_hash);
        oversized.handle(observed(llm_end(
            1,
            oversized_ids.anchor_call,
            oversized_ids.owner,
            "anchor",
        )));
        let effects = oversized.handle(CoordinatorCommand::OversizedEvent {
            event: ProjectedTrajectoryEvent::Oversized(OversizedTrajectoryEvent {
                ingest_seq: 2,
                event_uuid: uuid(260),
                parent_uuid: Some(oversized_ids.owner),
                kind: CapturedEventKind::Mark,
                scope_phase: None,
                category: None,
                call_role: None,
                scope_type: None,
            }),
            loss: LossSnapshot::default(),
        });
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::Overflow
            }
        );

        let barrier_ids = ids(270);
        let mut barrier = coordinator(1);
        let (pending_hash, generation) =
            register_anchor(&mut barrier, barrier_ids, 2, 16, &[], TestReplay::new());
        ack_pending(&mut barrier, barrier_ids.anchor, pending_hash);
        let effects = barrier.handle(CoordinatorCommand::DeadlineBarrierFailed {
            anchor_id: barrier_ids.anchor,
            generation,
            loss: LossSnapshot::default(),
        });
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::RejectedDeliveryBarrier
            }
        );
        let effects = barrier.handle(CoordinatorCommand::Health("test.health"));
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::Health("test.health")))
        );
    }

    #[test]
    fn internal_captured_and_oversized_events_are_ignored_before_all_actor_state() {
        let test_ids = ids(25_000);
        let mut actor = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut actor, test_ids, 1, 16, &[], TestReplay::new());
        ack_pending(&mut actor, test_ids.anchor, pending_hash);
        actor.handle(observed(llm_end(
            1,
            test_ids.anchor_call,
            test_ids.owner,
            "anchor",
        )));
        let sibling_call = uuid(25_020);
        register_primary(&mut actor, test_ids, sibling_call);

        let baseline = actor.snapshot();
        let baseline_highest_dropped = actor.highest_dropped_ingest_seq;
        let baseline_classification_loss = actor.classification_loss_epoch;
        let baseline_event_count = actor.windows[&test_ids.anchor].events.len();
        let internal_loss = LossSnapshot {
            highest_dropped_ingest_seq: 99,
            classification_loss_epoch: 99,
        };

        for (index, (category, role, scope_type)) in [
            ("llm", Some(LlmCallRole::Shadow), ScopeType::Llm),
            ("llm", Some(LlmCallRole::Judge), ScopeType::Llm),
            ("evaluator", None, ScopeType::Evaluator),
        ]
        .into_iter()
        .enumerate()
        {
            let event_uuid = uuid(25_030 + index as u128);
            let captured = event(
                2 + index as u64,
                event_uuid,
                Some(test_ids.owner),
                CapturedEventKind::Scope,
                Some(ScopeCategory::Start),
                Some(category),
                role,
                "user-controlled-name",
                Some(scope_type),
                Some(json!({"user_controlled": "must-not-be-inspected"})),
                "internal-captured",
            );
            assert!(
                actor
                    .handle(CoordinatorCommand::ObservedEvent {
                        event: captured,
                        loss: internal_loss,
                    })
                    .is_empty()
            );

            assert!(
                actor
                    .handle(CoordinatorCommand::OversizedEvent {
                        event: ProjectedTrajectoryEvent::Oversized(OversizedTrajectoryEvent {
                            ingest_seq: 5 + index as u64,
                            event_uuid: uuid(25_040 + index as u128),
                            parent_uuid: Some(test_ids.owner),
                            kind: CapturedEventKind::Scope,
                            scope_phase: Some(ScopeCategory::Start),
                            category: Some(category.into()),
                            call_role: role,
                            scope_type: Some(scope_type),
                        }),
                        loss: internal_loss,
                    })
                    .is_empty()
            );
        }

        assert_eq!(actor.snapshot(), baseline);
        assert_eq!(actor.highest_dropped_ingest_seq, baseline_highest_dropped);
        assert_eq!(
            actor.classification_loss_epoch,
            baseline_classification_loss
        );
        assert_eq!(
            actor.windows[&test_ids.anchor].events.len(),
            baseline_event_count
        );
        assert!(!actor.permanent_sampling_fault);

        let (_, terminal) = terminal_effect(actor.handle(observed(llm_end(
            8,
            sibling_call,
            test_ids.owner,
            "primary-sibling",
        ))));
        assert_eq!(terminal.observed_progress, 1);
        assert_eq!(terminal.events.len(), baseline_event_count + 1);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached
            }
        );
    }

    #[test]
    fn unresolved_non_llm_is_not_replayed_and_oversized_child_end_stays_excluded() {
        let unresolved_ids = ids(280);
        let mut unresolved = coordinator(1);
        let (pending_hash, _) = register_anchor(
            &mut unresolved,
            unresolved_ids,
            2,
            16,
            &[],
            TestReplay::new(),
        );
        ack_pending(&mut unresolved, unresolved_ids.anchor, pending_hash);
        unresolved.handle(observed(llm_end(
            1,
            unresolved_ids.anchor_call,
            unresolved_ids.owner,
            "anchor",
        )));
        let future_call = uuid(290);
        assert!(
            unresolved
                .handle(observed(event(
                    2,
                    uuid(291),
                    Some(future_call),
                    CapturedEventKind::Mark,
                    None,
                    None,
                    None,
                    "unresolved-mark",
                    None,
                    None,
                    "unresolved",
                )))
                .is_empty()
        );
        assert!(unresolved.staged.is_empty());
        let effects = unresolved.handle(observed(event(
            3,
            future_call,
            Some(unresolved_ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("function"),
            None,
            "future-parent",
            Some(ScopeType::Function),
            None,
            "future-parent",
        )));
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::ContradictoryOwnership
            }
        );
        assert!(terminal.events.is_empty());

        let child_ids = ids(295);
        let mut child = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut child, child_ids, 2, 16, &[], TestReplay::new());
        ack_pending(&mut child, child_ids.anchor, pending_hash);
        child.handle(observed(llm_end(
            1,
            child_ids.anchor_call,
            child_ids.owner,
            "anchor",
        )));
        let child_uuid = uuid(305);
        child.handle(observed(event(
            2,
            child_uuid,
            Some(child_ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("agent"),
            None,
            "child",
            Some(ScopeType::Agent),
            None,
            "child-start",
        )));
        let effects = child.handle(CoordinatorCommand::OversizedEvent {
            event: ProjectedTrajectoryEvent::Oversized(OversizedTrajectoryEvent {
                ingest_seq: 3,
                event_uuid: child_uuid,
                parent_uuid: Some(child_ids.owner),
                kind: CapturedEventKind::Scope,
                scope_phase: Some(ScopeCategory::End),
                category: Some("agent".into()),
                call_role: None,
                scope_type: Some(ScopeType::Agent),
            }),
            loss: LossSnapshot::default(),
        });
        assert!(
            !effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::RecordTerminal { .. }))
        );
        assert!(child.windows[&child_ids.anchor].frozen.is_none());
        assert!(child.windows[&child_ids.anchor].events.is_empty());
    }

    #[test]
    fn real_shaped_owner_start_reconciles_but_conflicting_call_ancestry_rejects() {
        let direct_ids = ids(320);
        let mut direct = coordinator(1);
        assert!(
            direct
                .handle(observed(event(
                    1,
                    direct_ids.owner,
                    Some(direct_ids.root),
                    CapturedEventKind::Scope,
                    Some(ScopeCategory::Start),
                    Some("agent"),
                    None,
                    "agent",
                    Some(ScopeType::Agent),
                    None,
                    "agent-start",
                )))
                .is_empty()
        );
        assert!(direct.unresolved_ownership.is_empty());
        let (_pending_hash, _) =
            register_anchor(&mut direct, direct_ids, 1, 16, &[], TestReplay::new());
        assert!(direct.unresolved_ownership.is_empty());
        assert!(direct.snapshot().sampling_enabled);
        assert_eq!(direct.snapshot().active_windows, 1);

        let old_ids = ids(330);
        let mut conflict = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut conflict, old_ids, 3, 16, &[], TestReplay::new());
        ack_pending(&mut conflict, old_ids.anchor, pending_hash);
        conflict.handle(observed(llm_end(
            1,
            old_ids.anchor_call,
            old_ids.owner,
            "anchor",
        )));
        let call_uuid = uuid(340);
        conflict.handle(observed(event(
            2,
            call_uuid,
            Some(old_ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("llm"),
            Some(LlmCallRole::Primary),
            "llm",
            Some(ScopeType::Llm),
            None,
            "pre-registration",
        )));
        let claimed_ids = ids(350);
        let effects = conflict.handle(CoordinatorCommand::RegisterPrimaryCall {
            registration: primary_registration(claimed_ids, call_uuid),
            loss: LossSnapshot::default(),
            admission_epoch: 0,
        });
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::ContradictoryOwnership
            }
        );
    }

    #[test]
    fn loss_and_overflow_reject_while_internal_roles_leave_primary_authority_intact() {
        let overflow_ids = ids(300);
        let mut overflow = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut overflow, overflow_ids, 3, 1, &[], TestReplay::new());
        ack_pending(&mut overflow, overflow_ids.anchor, pending_hash);
        overflow.handle(observed(llm_end(
            1,
            overflow_ids.anchor_call,
            overflow_ids.owner,
            "anchor",
        )));
        overflow.handle(observed(event(
            2,
            uuid(310),
            Some(overflow_ids.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "one",
            None,
            None,
            "one",
        )));
        let effects = overflow.handle(observed(event(
            3,
            uuid(311),
            Some(overflow_ids.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "two",
            None,
            None,
            "two",
        )));
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::Overflow
            }
        );
        assert_eq!(terminal.events.len(), 1);

        let loss_ids = ids(350);
        let mut loss = coordinator(1);
        let (pending_hash, _) = register_anchor(&mut loss, loss_ids, 2, 16, &[], TestReplay::new());
        ack_pending(&mut loss, loss_ids.anchor, pending_hash);
        loss.handle(observed(llm_end(
            1,
            loss_ids.anchor_call,
            loss_ids.owner,
            "anchor",
        )));
        let effects = loss.handle(CoordinatorCommand::ObservedEvent {
            event: event(
                3,
                uuid(360),
                Some(loss_ids.owner),
                CapturedEventKind::Mark,
                None,
                None,
                None,
                "after-loss",
                None,
                None,
                "after-loss",
            ),
            loss: LossSnapshot {
                highest_dropped_ingest_seq: 2,
                classification_loss_epoch: 0,
            },
        });
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::EventLoss
            }
        );
        assert!(terminal.events.is_empty());

        let conflict_ids = ids(400);
        let mut conflict = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut conflict, conflict_ids, 1, 16, &[], TestReplay::new());
        ack_pending(&mut conflict, conflict_ids.anchor, pending_hash);
        conflict.handle(observed(llm_end(
            1,
            conflict_ids.anchor_call,
            conflict_ids.owner,
            "anchor",
        )));
        let later_call = uuid(410);
        register_primary(&mut conflict, conflict_ids, later_call);
        let conflicting_end = event(
            2,
            later_call,
            Some(conflict_ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::End),
            Some("llm"),
            Some(LlmCallRole::Shadow),
            "llm",
            Some(ScopeType::Llm),
            None,
            "shadow",
        );
        let effects = conflict.handle(observed(conflicting_end));
        assert!(effects.is_empty());
        assert!(conflict.windows[&conflict_ids.anchor].frozen.is_none());
        let (_, terminal) = terminal_effect(conflict.handle(observed(llm_end(
            3,
            later_call,
            conflict_ids.owner,
            "primary",
        ))));
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached
            }
        );
    }

    #[test]
    fn sink_and_delivery_failures_retry_identical_frozen_work() {
        let ids = ids(500);
        let mut actor = coordinator(1);
        let (pending_hash, _) = register_anchor(&mut actor, ids, 1, 16, &[], TestReplay::new());
        ack_pending(&mut actor, ids.anchor, pending_hash);
        actor.handle(observed(llm_end(1, ids.anchor_call, ids.owner, "anchor")));
        let later = uuid(510);
        register_primary(&mut actor, ids, later);
        let effects = actor.handle(observed(llm_end(2, later, ids.owner, "later")));
        let (terminal_hash, _) = terminal_effect(effects);

        let effects = actor.handle(CoordinatorCommand::TerminalRecorded(SinkAck::Failed {
            anchor_id: ids.anchor,
            stable_class: SinkFailureClass::InjectedTransient,
        }));
        let retry_generation = effects
            .iter()
            .find_map(|effect| match effect {
                CoordinatorEffect::ScheduleSinkRetry { generation, .. } => Some(*generation),
                _ => None,
            })
            .unwrap();
        assert!(!actor.snapshot().sampling_enabled);
        assert!(
            actor
                .handle(CoordinatorCommand::SinkRetryElapsed {
                    anchor_id: ids.anchor,
                    generation: retry_generation + 1,
                })
                .is_empty()
        );
        let effects = actor.handle(CoordinatorCommand::SinkRetryElapsed {
            anchor_id: ids.anchor,
            generation: retry_generation,
        });
        assert!(effects.iter().any(|effect| matches!(
            effect,
            CoordinatorEffect::RecordTerminal { payload_hash, .. } if payload_hash == &terminal_hash
        )));

        let effects = actor.handle(CoordinatorCommand::TerminalRecorded(SinkAck::Applied {
            anchor_id: ids.anchor,
            payload_hash: terminal_hash,
        }));
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::DeliverWindow { .. }))
        );
        let effects = actor.handle(CoordinatorCommand::WindowDelivered(DeliveryAck::Failed {
            anchor_id: ids.anchor,
            stable_class: DeliveryFailureClass::InjectedTransient,
        }));
        let delivery_generation = effects
            .iter()
            .find_map(|effect| match effect {
                CoordinatorEffect::ScheduleDeliveryRetry { generation, .. } => Some(*generation),
                _ => None,
            })
            .unwrap();
        let effects = actor.handle(CoordinatorCommand::DeliveryRetryElapsed {
            anchor_id: ids.anchor,
            generation: delivery_generation,
        });
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::DeliverWindow { .. }))
        );
        actor.handle(CoordinatorCommand::WindowDelivered(
            DeliveryAck::Delivered {
                anchor_id: ids.anchor,
            },
        ));
        assert_eq!(actor.snapshot().active_windows, 0);
    }

    #[test]
    fn terminal_canonicalization_failure_never_emits_an_empty_hash_write() {
        let ids = ids(550);
        let mut actor = coordinator(1);
        let (pending_hash, _) = register_anchor(&mut actor, ids, 1, 16, &[], TestReplay::new());
        ack_pending(&mut actor, ids.anchor, pending_hash);
        actor.handle(observed(llm_end(1, ids.anchor_call, ids.owner, "anchor")));
        let window = actor.windows.get_mut(&ids.anchor).unwrap();
        let payload = PersistedTrajectoryTerminalV1::rejected(
            window.pending.clone(),
            Vec::new(),
            0,
            TrajectoryRejectionReason::EventLoss,
            fixed_time(),
            Vec::new(),
        );
        window.frozen = Some(FrozenTerminal {
            outcome: FrozenOutcome::Rejected(TrajectoryRejectionReason::EventLoss),
            payload,
            payload_hash: None,
            retry_generation: 0,
        });
        window.phase = WindowPhase::PersistingTerminal;
        let mut fault_effects = Vec::new();
        actor.set_permanent_sampling_fault(HEALTH_SINK_FAILURE, &mut fault_effects);
        actor.enqueue_terminal_once(ids.anchor);

        let effects = actor.handle(CoordinatorCommand::Health("test.tick"));
        assert!(
            !effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::RecordTerminal { .. }))
        );
        assert!(!actor.snapshot().sampling_enabled);
        assert_eq!(
            actor.snapshot().phases[&ids.anchor],
            WindowPhase::PersistingTerminal
        );
        assert!(
            actor.windows[&ids.anchor]
                .frozen
                .as_ref()
                .unwrap()
                .payload_hash
                .is_none()
        );
    }

    #[test]
    fn shutdown_preserves_delivery_phase_and_drains_ack_races() {
        let primary_ids = ids(575);
        let mut actor = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut actor, primary_ids, 1, 16, &[], TestReplay::new());
        ack_pending(&mut actor, primary_ids.anchor, pending_hash);
        actor.handle(observed(llm_end(
            1,
            primary_ids.anchor_call,
            primary_ids.owner,
            "anchor",
        )));
        let later = uuid(585);
        register_primary(&mut actor, primary_ids, later);
        let effects = actor.handle(observed(llm_end(2, later, primary_ids.owner, "later")));
        let (terminal_hash, _) = terminal_effect(effects);
        actor.handle(CoordinatorCommand::TerminalRecorded(SinkAck::Applied {
            anchor_id: primary_ids.anchor,
            payload_hash: terminal_hash.clone(),
        }));
        assert_eq!(
            actor.snapshot().phases[&primary_ids.anchor],
            WindowPhase::Delivering
        );

        let effects = actor.handle(CoordinatorCommand::Shutdown {
            admission_epoch: 1,
            loss: LossSnapshot::default(),
        });
        assert!(
            !effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::RecordTerminal { .. }))
        );
        assert_eq!(
            actor.snapshot().phases[&primary_ids.anchor],
            WindowPhase::Delivering
        );
        assert!(
            actor
                .handle(CoordinatorCommand::TerminalRecorded(
                    SinkAck::AlreadyApplied {
                        anchor_id: primary_ids.anchor,
                        payload_hash: terminal_hash,
                    }
                ))
                .is_empty()
        );
        let effects = actor.handle(CoordinatorCommand::WindowDelivered(
            DeliveryAck::Delivered {
                anchor_id: primary_ids.anchor,
            },
        ));
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::Drained))
        );

        let retry_ids = ids(590);
        let mut retry = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut retry, retry_ids, 1, 16, &[], TestReplay::new());
        ack_pending(&mut retry, retry_ids.anchor, pending_hash);
        retry.handle(observed(llm_end(
            1,
            retry_ids.anchor_call,
            retry_ids.owner,
            "anchor",
        )));
        let later = uuid(599);
        register_primary(&mut retry, retry_ids, later);
        let (terminal_hash, _) =
            terminal_effect(retry.handle(observed(llm_end(2, later, retry_ids.owner, "later"))));
        retry.handle(CoordinatorCommand::TerminalRecorded(SinkAck::Applied {
            anchor_id: retry_ids.anchor,
            payload_hash: terminal_hash,
        }));
        let effects = retry.handle(CoordinatorCommand::WindowDelivered(DeliveryAck::Failed {
            anchor_id: retry_ids.anchor,
            stable_class: DeliveryFailureClass::InjectedTransient,
        }));
        let generation = effects
            .iter()
            .find_map(|effect| match effect {
                CoordinatorEffect::ScheduleDeliveryRetry { generation, .. } => Some(*generation),
                _ => None,
            })
            .unwrap();
        retry.handle(CoordinatorCommand::Shutdown {
            admission_epoch: 1,
            loss: LossSnapshot::default(),
        });
        let effects = retry.handle(CoordinatorCommand::DeliveryRetryElapsed {
            anchor_id: retry_ids.anchor,
            generation,
        });
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::DeliverWindow { .. }))
        );
        let effects = retry.handle(CoordinatorCommand::WindowDelivered(
            DeliveryAck::Delivered {
                anchor_id: retry_ids.anchor,
            },
        ));
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::Drained))
        );
    }

    #[test]
    fn late_registered_captured_evidence_is_sorted_but_oversized_evidence_rejects() {
        let replay_ids = ids(800);
        let mut replay = coordinator(1);
        let (pending_hash, _) = register_anchor(
            &mut replay,
            replay_ids,
            1,
            16,
            &["compaction"],
            TestReplay::new(),
        );
        ack_pending(&mut replay, replay_ids.anchor, pending_hash);
        replay.handle(observed(llm_end(
            1,
            replay_ids.anchor_call,
            replay_ids.owner,
            "anchor",
        )));
        let future_parent = uuid(810);
        let late_call = uuid(811);
        replay.handle(observed(event(
            2,
            late_call,
            Some(future_parent),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("llm"),
            Some(LlmCallRole::Primary),
            "late-llm",
            Some(ScopeType::Llm),
            None,
            "late-llm",
        )));
        let effects = replay.handle(observed(event(
            3,
            uuid(812),
            Some(replay_ids.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "compaction",
            None,
            None,
            "compaction",
        )));
        assert!(
            effects
                .iter()
                .all(|effect| !matches!(effect, CoordinatorEffect::RecordTerminal { .. }))
        );
        assert!(replay.windows[&replay_ids.anchor].provisional.is_some());
        let effects = replay.handle(CoordinatorCommand::RegisterPrimaryCall {
            registration: nested_primary_registration(replay_ids, late_call, future_parent),
            loss: LossSnapshot::default(),
            admission_epoch: 0,
        });
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal
                .events
                .iter()
                .map(|event| event.ingest_seq)
                .collect::<Vec<_>>(),
            [2, 3]
        );
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached
            }
        );

        let oversized_ids = ids(820);
        let mut oversized = coordinator(1);
        let (pending_hash, _) = register_anchor(
            &mut oversized,
            oversized_ids,
            1,
            16,
            &["compaction"],
            TestReplay::new(),
        );
        ack_pending(&mut oversized, oversized_ids.anchor, pending_hash);
        oversized.handle(observed(llm_end(
            1,
            oversized_ids.anchor_call,
            oversized_ids.owner,
            "anchor",
        )));
        let future_parent = uuid(830);
        let late_call = uuid(831);
        oversized.handle(CoordinatorCommand::OversizedEvent {
            event: ProjectedTrajectoryEvent::Oversized(OversizedTrajectoryEvent {
                ingest_seq: 2,
                event_uuid: late_call,
                parent_uuid: Some(future_parent),
                kind: CapturedEventKind::Scope,
                scope_phase: Some(ScopeCategory::Start),
                category: Some("llm".into()),
                call_role: Some(LlmCallRole::Primary),
                scope_type: Some(ScopeType::Llm),
            }),
            loss: LossSnapshot::default(),
        });
        oversized.handle(observed(event(
            3,
            uuid(832),
            Some(oversized_ids.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "compaction",
            None,
            None,
            "compaction",
        )));
        let effects = oversized.handle(CoordinatorCommand::RegisterPrimaryCall {
            registration: nested_primary_registration(oversized_ids, late_call, future_parent),
            loss: LossSnapshot::default(),
            admission_epoch: 0,
        });
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::Overflow
            }
        );
    }

    #[test]
    fn unresolved_non_llm_evidence_blocks_provisional_completion_until_ownership_resolves() {
        let test_ids = ids(840);
        let mut actor = coordinator(1);
        let (pending_hash, _) = register_anchor(
            &mut actor,
            test_ids,
            1,
            16,
            &["compaction"],
            TestReplay::new(),
        );
        ack_pending(&mut actor, test_ids.anchor, pending_hash);
        actor.handle(observed(llm_end(
            1,
            test_ids.anchor_call,
            test_ids.owner,
            "anchor",
        )));
        let future_parent = uuid(850);
        actor.handle(observed(event(
            2,
            uuid(851),
            Some(future_parent),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "unknown-mark",
            None,
            None,
            "unknown-mark",
        )));
        let effects = actor.handle(observed(event(
            3,
            uuid(852),
            Some(test_ids.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "compaction",
            None,
            None,
            "compaction",
        )));
        assert!(
            effects
                .iter()
                .all(|effect| !matches!(effect, CoordinatorEffect::RecordTerminal { .. }))
        );
        let effects = actor.handle(observed(event(
            4,
            future_parent,
            Some(test_ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("function"),
            None,
            "future-parent",
            Some(ScopeType::Function),
            None,
            "future-parent",
        )));
        let (_, terminal) = terminal_effect(effects);
        assert!(matches!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected { .. }
        ));
    }

    #[test]
    fn deadline_and_shutdown_reject_post_boundary_ambiguity() {
        for shutdown in [false, true] {
            let test_ids = ids(if shutdown { 880 } else { 860 });
            let mut actor = coordinator(1);
            let (pending_hash, generation) =
                register_anchor(&mut actor, test_ids, 2, 16, &[], TestReplay::new());
            ack_pending(&mut actor, test_ids.anchor, pending_hash);
            actor.handle(observed(llm_end(
                1,
                test_ids.anchor_call,
                test_ids.owner,
                "anchor",
            )));
            actor.handle(observed(event(
                2,
                uuid(if shutdown { 890 } else { 870 }),
                Some(uuid(if shutdown { 891 } else { 871 })),
                CapturedEventKind::Scope,
                Some(ScopeCategory::Start),
                Some("llm"),
                Some(LlmCallRole::Primary),
                "unknown-llm",
                Some(ScopeType::Llm),
                None,
                "unknown-llm",
            )));
            let effects = if shutdown {
                actor.handle(CoordinatorCommand::Shutdown {
                    admission_epoch: 1,
                    loss: LossSnapshot::default(),
                })
            } else {
                actor.handle(CoordinatorCommand::DeadlineElapsed {
                    anchor_id: test_ids.anchor,
                    generation,
                    loss: LossSnapshot::default(),
                })
            };
            let (_, terminal) = terminal_effect(effects);
            assert_eq!(
                terminal.state,
                TrajectoryTerminalStateV1::Rejected {
                    reason: TrajectoryRejectionReason::EventLoss
                }
            );
        }
    }

    #[test]
    fn byte_limits_and_loss_boundaries_are_classified_explicitly() {
        let byte_ids = ids(900);
        let mut bytes = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut bytes, byte_ids, 1, 16, &[], TestReplay::new());
        ack_pending(&mut bytes, byte_ids.anchor, pending_hash);
        bytes.handle(observed(llm_end(
            1,
            byte_ids.anchor_call,
            byte_ids.owner,
            "anchor",
        )));
        bytes
            .windows
            .get_mut(&byte_ids.anchor)
            .unwrap()
            .limits
            .max_bytes = 15;
        let later_call = uuid(910);
        register_primary(&mut bytes, byte_ids, later_call);
        let (_, terminal) = terminal_effect(bytes.handle(observed(llm_end(
            2,
            later_call,
            byte_ids.owner,
            "sixteen-bytes",
        ))));
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::Overflow
            }
        );

        let isolated_ids = ids(920);
        let mut isolated = coordinator(1);
        isolated.dropped_staged_calls.insert(uuid(929), 1);
        let (pending_hash, _) =
            register_anchor(&mut isolated, isolated_ids, 1, 16, &[], TestReplay::new());
        ack_pending(&mut isolated, isolated_ids.anchor, pending_hash);
        isolated.handle(observed(llm_end(
            2,
            isolated_ids.anchor_call,
            isolated_ids.owner,
            "anchor",
        )));
        let later_call = uuid(930);
        register_primary(&mut isolated, isolated_ids, later_call);
        let (_, terminal) = terminal_effect(isolated.handle(observed(llm_end(
            3,
            later_call,
            isolated_ids.owner,
            "later",
        ))));
        assert!(matches!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed { .. }
        ));

        let dropped_anchor_ids = ids(940);
        let mut dropped_anchor = coordinator(1);
        let (pending_hash, generation) = register_anchor(
            &mut dropped_anchor,
            dropped_anchor_ids,
            1,
            16,
            &[],
            TestReplay::new(),
        );
        ack_pending(&mut dropped_anchor, dropped_anchor_ids.anchor, pending_hash);
        let (_, terminal) =
            terminal_effect(dropped_anchor.handle(CoordinatorCommand::DeadlineElapsed {
                anchor_id: dropped_anchor_ids.anchor,
                generation,
                loss: LossSnapshot {
                    highest_dropped_ingest_seq: 1,
                    classification_loss_epoch: 0,
                },
            }));
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::EventLoss
            }
        );

        let classification_ids = ids(960);
        let mut classification = coordinator(1);
        let (pending_hash, generation) = register_anchor(
            &mut classification,
            classification_ids,
            1,
            16,
            &[],
            TestReplay::new(),
        );
        ack_pending(&mut classification, classification_ids.anchor, pending_hash);
        let effects = classification.handle(CoordinatorCommand::DeadlineElapsed {
            anchor_id: classification_ids.anchor,
            generation,
            loss: LossSnapshot {
                highest_dropped_ingest_seq: 0,
                classification_loss_epoch: 1,
            },
        });
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::EventLoss
            }
        );
        assert!(classification.permanent_sampling_fault);
        assert!(!classification.snapshot().sampling_enabled);
    }

    #[test]
    fn contradictory_unresolved_uuid_reuse_rejects_immediately() {
        let test_ids = ids(980);
        let mut actor = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut actor, test_ids, 2, 16, &[], TestReplay::new());
        ack_pending(&mut actor, test_ids.anchor, pending_hash);
        actor.handle(observed(llm_end(
            1,
            test_ids.anchor_call,
            test_ids.owner,
            "anchor",
        )));
        let reused = uuid(990);
        actor.handle(observed(event(
            2,
            reused,
            Some(uuid(991)),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "unknown",
            None,
            None,
            "first-parent",
        )));
        let effects = actor.handle(observed(event(
            3,
            reused,
            Some(uuid(992)),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "unknown",
            None,
            None,
            "second-parent",
        )));
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::ContradictoryOwnership
            }
        );
        assert!(actor.permanent_sampling_fault);
    }

    #[test]
    fn ended_generic_uncertainty_is_retained_until_parent_resolution() {
        let test_ids = ids(1_000);
        let mut actor = coordinator(1);
        let (pending_hash, _) = register_anchor(
            &mut actor,
            test_ids,
            1,
            16,
            &["compaction"],
            TestReplay::new(),
        );
        ack_pending(&mut actor, test_ids.anchor, pending_hash);
        actor.handle(observed(llm_end(
            1,
            test_ids.anchor_call,
            test_ids.owner,
            "anchor",
        )));
        let future_parent = uuid(1_010);
        let unknown_scope = uuid(1_011);
        actor.handle(observed(event(
            2,
            unknown_scope,
            Some(future_parent),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("tool"),
            None,
            "unknown-tool",
            Some(ScopeType::Tool),
            None,
            "unknown-tool-start",
        )));
        actor.handle(observed(event(
            3,
            unknown_scope,
            Some(future_parent),
            CapturedEventKind::Scope,
            Some(ScopeCategory::End),
            Some("tool"),
            None,
            "unknown-tool",
            Some(ScopeType::Tool),
            None,
            "unknown-tool-end",
        )));
        assert!(matches!(
            actor.unresolved_ownership.get(&unknown_scope),
            Some(UnresolvedOwnership::Parent { ended: true, .. })
        ));
        actor.handle(observed(event(
            4,
            uuid(1_012),
            Some(test_ids.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "compaction",
            None,
            None,
            "compaction",
        )));
        assert!(actor.windows[&test_ids.anchor].provisional.is_some());
        let effects = actor.handle(observed(event(
            5,
            future_parent,
            Some(test_ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("function"),
            None,
            "future-parent",
            Some(ScopeType::Function),
            None,
            "future-parent",
        )));
        let (_, terminal) = terminal_effect(effects);
        assert!(matches!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected { .. }
        ));
    }

    #[test]
    fn lifecycle_saturation_is_root_scoped_and_pruned_with_the_root_lifetime() {
        let first = ids(1_020);
        let mut actor = coordinator(2);
        let (pending_hash, _) =
            register_anchor(&mut actor, first, 1, 16, &["compaction"], TestReplay::new());
        ack_pending(&mut actor, first.anchor, pending_hash);
        actor.handle(observed(llm_end(
            1,
            first.anchor_call,
            first.owner,
            "anchor",
        )));
        actor.seen_lifecycle_progress.insert(
            first.root,
            (0..actor.limits.ancestry_capacity)
                .map(|index| uuid(10_000 + index as u128))
                .collect(),
        );
        let effects = actor.handle(observed(event(
            2,
            uuid(20_000),
            Some(first.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "compaction",
            None,
            None,
            "saturating-lifecycle",
        )));
        let (terminal_hash, terminal) = terminal_effect(effects);
        assert!(matches!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected { .. }
        ));
        assert!(actor.lifecycle_saturated_roots.contains(&first.root));
        assert!(!actor.permanent_sampling_fault);

        let second = Ids {
            root: first.root,
            owner: first.owner,
            anchor_call: uuid(20_010),
            anchor: uuid(20_011),
        };
        register_primary(&mut actor, second, second.anchor_call);
        let effects = actor.handle(CoordinatorCommand::RegisterAnchor(Box::new(
            AnchorRegistration {
                seed: seed(second, 1, TestReplay::new()),
                proposal_permit: None,
                monotonic_deadline: Instant::now() + Duration::from_secs(300),
                opened_after_ingest_seq: 2,
                loss: LossSnapshot::default(),
                admission_epoch: 0,
                limits: WindowLimits::new(1, 16, 1 << 20, &["compaction".into()]),
            },
        )));
        assert!(effects.iter().any(|effect| matches!(
            effect,
            CoordinatorEffect::AnchorRefused {
                anchor_id,
                reason: AnchorRefusalReason::SamplingSuppressed,
            } if *anchor_id == second.anchor
        )));

        actor.handle(observed(llm_end(
            3,
            second.anchor_call,
            second.owner,
            "refused-anchor-call",
        )));
        actor.handle(observed(event(
            4,
            first.owner,
            Some(first.root),
            CapturedEventKind::Scope,
            Some(ScopeCategory::End),
            Some("agent"),
            None,
            "agent",
            Some(ScopeType::Agent),
            None,
            "agent-end",
        )));
        actor.handle(CoordinatorCommand::TerminalRecorded(SinkAck::Applied {
            anchor_id: first.anchor,
            payload_hash: terminal_hash,
        }));
        assert!(!actor.lifecycle_saturated_roots.contains(&first.root));
        assert!(!actor.seen_lifecycle_progress.contains_key(&first.root));
    }

    #[test]
    fn stale_sink_and_delivery_acks_do_not_mutate_the_current_operation() {
        let first = ids(1_100);
        let second = ids(1_120);
        let mut actor = coordinator(2);
        let (first_hash, _) = register_anchor(&mut actor, first, 1, 16, &[], TestReplay::new());
        register_primary(&mut actor, second, second.anchor_call);
        actor.handle(CoordinatorCommand::RegisterAnchor(Box::new(
            AnchorRegistration {
                seed: seed(second, 1, TestReplay::new()),
                proposal_permit: None,
                monotonic_deadline: Instant::now() + Duration::from_secs(300),
                opened_after_ingest_seq: 0,
                loss: LossSnapshot::default(),
                admission_epoch: 0,
                limits: WindowLimits::new(1, 16, 1 << 20, &[]),
            },
        )));
        let second_hash = actor.windows[&second.anchor].pending_hash.clone();

        let stale_for_first = vec![
            SinkAck::Applied {
                anchor_id: second.anchor,
                payload_hash: second_hash.clone(),
            },
            SinkAck::AlreadyApplied {
                anchor_id: second.anchor,
                payload_hash: second_hash.clone(),
            },
            SinkAck::AlreadyTerminal {
                anchor_id: second.anchor,
                pending_hash: second_hash.clone(),
                terminal_hash: "stale-terminal".into(),
            },
            SinkAck::DurablyDeclined {
                anchor_id: second.anchor,
                pending_hash: second_hash.clone(),
                terminal_hash: "a".repeat(64),
                reason: DurableDeclineReason::NotScheduledQueueFull,
            },
            SinkAck::TransientRefused {
                anchor_id: second.anchor,
                reason: TransientRefusalReason::EvidenceCapacity,
            },
            SinkAck::Conflict {
                anchor_id: second.anchor,
            },
            SinkAck::Failed {
                anchor_id: second.anchor,
                stable_class: SinkFailureClass::InjectedTransient,
            },
        ];
        for ack in stale_for_first {
            assert!(
                actor
                    .handle(CoordinatorCommand::PendingRecorded(ack))
                    .is_empty()
            );
            assert_eq!(
                actor.sink_inflight,
                Some(SinkWork {
                    anchor_id: first.anchor,
                    kind: SinkWorkKind::Pending,
                })
            );
            assert!(!actor.permanent_sampling_fault);
        }

        let effects = actor.handle(CoordinatorCommand::PendingRecorded(SinkAck::Applied {
            anchor_id: first.anchor,
            payload_hash: first_hash.clone(),
        }));
        assert!(effects.iter().any(|effect| matches!(
            effect,
            CoordinatorEffect::RecordPending { anchor_id, .. } if *anchor_id == second.anchor
        )));
        for ack in [
            SinkAck::AlreadyApplied {
                anchor_id: first.anchor,
                payload_hash: first_hash.clone(),
            },
            SinkAck::Conflict {
                anchor_id: first.anchor,
            },
            SinkAck::Failed {
                anchor_id: first.anchor,
                stable_class: SinkFailureClass::InjectedPermanent,
            },
            SinkAck::DurablyDeclined {
                anchor_id: first.anchor,
                pending_hash: first_hash.clone(),
                terminal_hash: "a".repeat(64),
                reason: DurableDeclineReason::NotScheduledQueueFull,
            },
            SinkAck::TransientRefused {
                anchor_id: first.anchor,
                reason: TransientRefusalReason::EvidenceCapacity,
            },
        ] {
            assert!(
                actor
                    .handle(CoordinatorCommand::PendingRecorded(ack))
                    .is_empty()
            );
            assert_eq!(
                actor.sink_inflight,
                Some(SinkWork {
                    anchor_id: second.anchor,
                    kind: SinkWorkKind::Pending,
                })
            );
            assert!(!actor.permanent_sampling_fault);
        }
        actor.handle(CoordinatorCommand::PendingRecorded(
            SinkAck::AlreadyApplied {
                anchor_id: second.anchor,
                payload_hash: second_hash,
            },
        ));

        actor.handle(observed(llm_end(
            1,
            first.anchor_call,
            first.owner,
            "anchor",
        )));
        let later_call = uuid(1_130);
        register_primary(&mut actor, first, later_call);
        let (terminal_hash, _) =
            terminal_effect(actor.handle(observed(llm_end(2, later_call, first.owner, "later"))));
        actor.handle(CoordinatorCommand::TerminalRecorded(
            SinkAck::AlreadyApplied {
                anchor_id: first.anchor,
                payload_hash: terminal_hash,
            },
        ));
        for ack in [
            DeliveryAck::Delivered {
                anchor_id: second.anchor,
            },
            DeliveryAck::AlreadyDelivered {
                anchor_id: second.anchor,
            },
            DeliveryAck::Failed {
                anchor_id: second.anchor,
                stable_class: DeliveryFailureClass::InjectedTransient,
            },
        ] {
            assert!(
                actor
                    .handle(CoordinatorCommand::WindowDelivered(ack))
                    .is_empty()
            );
            assert_eq!(actor.delivery_inflight, Some(first.anchor));
            assert!(!actor.permanent_sampling_fault);
        }
    }

    #[test]
    fn same_anchor_sink_ack_outcomes_are_exhaustive_and_do_not_wedge() {
        let pending_ids = ids(1_200);
        let mut pending = coordinator(1);
        let (_pending_hash, _) =
            register_anchor(&mut pending, pending_ids, 1, 16, &[], TestReplay::new());
        let effects = pending.handle(CoordinatorCommand::PendingRecorded(
            SinkAck::AlreadyApplied {
                anchor_id: pending_ids.anchor,
                payload_hash: "wrong-hash".into(),
            },
        ));
        assert!(effects.iter().any(|effect| matches!(
            effect,
            CoordinatorEffect::AnchorRefused {
                reason: AnchorRefusalReason::SinkRejected,
                ..
            }
        )));
        assert!(pending.windows.is_empty());
        assert!(pending.sink_inflight.is_none());
        assert!(pending.permanent_sampling_fault);

        let recovered_ids = ids(1_220);
        let mut recovered = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut recovered, recovered_ids, 1, 16, &[], TestReplay::new());
        recovered.handle(CoordinatorCommand::PendingRecorded(
            SinkAck::AlreadyTerminal {
                anchor_id: recovered_ids.anchor,
                pending_hash,
                terminal_hash: "a".repeat(64),
            },
        ));
        assert!(recovered.windows.is_empty());
        assert!(recovered.sink_inflight.is_none());
        assert!(!recovered.permanent_sampling_fault);

        let terminal_ids = ids(1_240);
        let mut terminal = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut terminal, terminal_ids, 1, 16, &[], TestReplay::new());
        ack_pending(&mut terminal, terminal_ids.anchor, pending_hash.clone());
        terminal.handle(observed(llm_end(
            1,
            terminal_ids.anchor_call,
            terminal_ids.owner,
            "anchor",
        )));
        let later_call = uuid(1_250);
        register_primary(&mut terminal, terminal_ids, later_call);
        let (terminal_hash, _) = terminal_effect(terminal.handle(observed(llm_end(
            2,
            later_call,
            terminal_ids.owner,
            "later",
        ))));
        let effects = terminal.handle(CoordinatorCommand::TerminalRecorded(
            SinkAck::AlreadyTerminal {
                anchor_id: terminal_ids.anchor,
                pending_hash,
                terminal_hash,
            },
        ));
        assert!(effects.iter().any(|effect| matches!(
            effect,
            CoordinatorEffect::DeliverWindow { anchor_id, .. }
                if *anchor_id == terminal_ids.anchor
        )));

        let conflict_ids = ids(1_260);
        let mut conflict = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut conflict, conflict_ids, 1, 16, &[], TestReplay::new());
        ack_pending(&mut conflict, conflict_ids.anchor, pending_hash);
        conflict.handle(observed(llm_end(
            1,
            conflict_ids.anchor_call,
            conflict_ids.owner,
            "anchor",
        )));
        let later_call = uuid(1_270);
        register_primary(&mut conflict, conflict_ids, later_call);
        terminal_effect(conflict.handle(observed(llm_end(
            2,
            later_call,
            conflict_ids.owner,
            "later",
        ))));
        let terminal_hash = conflict.windows[&conflict_ids.anchor]
            .frozen
            .as_ref()
            .unwrap()
            .payload_hash
            .clone()
            .unwrap();
        let effects = conflict.handle(CoordinatorCommand::TerminalRecorded(SinkAck::Conflict {
            anchor_id: conflict_ids.anchor,
        }));
        assert!(effects.iter().any(|effect| matches!(
            effect,
            CoordinatorEffect::ScheduleSinkRetry { anchor_id, generation: 1 }
                if *anchor_id == conflict_ids.anchor
        )));
        assert_eq!(
            conflict.windows[&conflict_ids.anchor].phase,
            WindowPhase::PersistingTerminal
        );
        assert!(conflict.windows[&conflict_ids.anchor].frozen.is_some());
        assert!(conflict.sink_inflight.is_none());
        assert!(conflict.permanent_sampling_fault);
        let effects = conflict.handle(CoordinatorCommand::SinkRetryElapsed {
            anchor_id: conflict_ids.anchor,
            generation: 1,
        });
        assert!(effects.iter().any(|effect| matches!(
            effect,
            CoordinatorEffect::RecordTerminal {
                anchor_id,
                payload_hash,
                ..
            } if *anchor_id == conflict_ids.anchor && payload_hash == &terminal_hash
        )));
    }

    #[test]
    fn pending_refusals_release_replay_and_use_independent_temporary_gates() {
        let scheduler_ids = ids(40_000);
        let scheduler_replay = TestReplay::new();
        let mut scheduler = coordinator(1);
        let (pending_hash, _) = register_anchor(
            &mut scheduler,
            scheduler_ids,
            1,
            16,
            &[],
            scheduler_replay.clone(),
        );
        assert_eq!(Arc::strong_count(&scheduler_replay), 2);

        let effects = scheduler.handle(CoordinatorCommand::PendingRecorded(
            SinkAck::DurablyDeclined {
                anchor_id: scheduler_ids.anchor,
                pending_hash,
                terminal_hash: "a".repeat(64),
                reason: DurableDeclineReason::NotScheduledQueueFull,
            },
        ));

        assert!(scheduler.windows.is_empty());
        assert_eq!(Arc::strong_count(&scheduler_replay), 1);
        assert_eq!(
            scheduler.snapshot().scheduler_pressure_pools,
            BTreeSet::from([POOL.to_string()])
        );
        assert!(!scheduler.snapshot().evidence_capacity_pressure);
        assert!(!scheduler.permanent_sampling_fault);
        assert!(effects.iter().any(|effect| matches!(
            effect,
            CoordinatorEffect::SamplingSuppressed(HEALTH_SCHEDULER_PRESSURE)
        )));
        let scheduler_generation = effects
            .iter()
            .find_map(|effect| match effect {
                CoordinatorEffect::SchedulerPressureClosed {
                    pool_id,
                    pressure_generation,
                } if pool_id == POOL => Some(*pressure_generation),
                _ => None,
            })
            .unwrap();

        let effects = scheduler.handle(CoordinatorCommand::SchedulerCapacityRecovered {
            pool_id: POOL.to_string(),
            pressure_generation: scheduler_generation,
        });
        assert!(scheduler.snapshot().scheduler_pressure_pools.is_empty());
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::SamplingResumed))
        );

        let evidence_ids = ids(40_100);
        let evidence_replay = TestReplay::new();
        let mut evidence = coordinator(1);
        register_anchor(
            &mut evidence,
            evidence_ids,
            1,
            16,
            &[],
            evidence_replay.clone(),
        );
        let effects = evidence.handle(CoordinatorCommand::PendingRecorded(
            SinkAck::TransientRefused {
                anchor_id: evidence_ids.anchor,
                reason: TransientRefusalReason::EvidenceCapacity,
            },
        ));

        assert!(evidence.windows.is_empty());
        assert_eq!(Arc::strong_count(&evidence_replay), 1);
        assert!(evidence.snapshot().evidence_capacity_pressure);
        assert!(evidence.snapshot().scheduler_pressure_pools.is_empty());
        assert!(!evidence.permanent_sampling_fault);
        assert!(effects.iter().any(|effect| matches!(
            effect,
            CoordinatorEffect::SamplingSuppressed(HEALTH_EVIDENCE_CAPACITY)
        )));
        let evidence_generation = effects
            .iter()
            .find_map(|effect| match effect {
                CoordinatorEffect::EvidenceCapacityPressureClosed {
                    pressure_generation,
                } => Some(*pressure_generation),
                _ => None,
            })
            .unwrap();

        let effects = evidence.handle(CoordinatorCommand::EvidenceCapacityRecovered {
            pressure_generation: evidence_generation,
        });
        assert!(!evidence.snapshot().evidence_capacity_pressure);
        assert!(
            effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::SamplingResumed))
        );
    }

    #[test]
    fn scheduler_pressure_is_pool_scoped_and_decline_hash_is_validated() {
        const OTHER_POOL: &str = "other-pool";
        let mut actor = Coordinator::new_with_clock(
            BTreeMap::from([(POOL.to_string(), 2), (OTHER_POOL.to_string(), 2)]),
            Arc::new(fixed_time),
        );
        let pressured_ids = ids(40_200);
        let (pending_hash, _) =
            register_anchor(&mut actor, pressured_ids, 1, 16, &[], TestReplay::new());
        let effects = actor.handle(CoordinatorCommand::PendingRecorded(
            SinkAck::DurablyDeclined {
                anchor_id: pressured_ids.anchor,
                pending_hash,
                terminal_hash: "b".repeat(64),
                reason: DurableDeclineReason::NotScheduledQueueFull,
            },
        ));
        assert!(
            !effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::SamplingSuppressed(_)))
        );
        assert!(actor.snapshot().sampling_enabled);

        let other_ids = ids(40_300);
        register_primary(&mut actor, other_ids, other_ids.anchor_call);
        let mut other_seed = seed(other_ids, 1, TestReplay::new());
        other_seed.pending.pool_id = OTHER_POOL.to_string();
        let effects = actor.handle(CoordinatorCommand::RegisterAnchor(Box::new(
            AnchorRegistration {
                seed: other_seed,
                proposal_permit: None,
                monotonic_deadline: Instant::now() + Duration::from_secs(300),
                opened_after_ingest_seq: 0,
                loss: LossSnapshot::default(),
                admission_epoch: 0,
                limits: WindowLimits::new(1, 16, 1 << 20, &[]),
            },
        )));
        assert!(effects.iter().any(|effect| matches!(
            effect,
            CoordinatorEffect::RecordPending { anchor_id, .. } if *anchor_id == other_ids.anchor
        )));

        let refused_ids = ids(40_400);
        register_primary(&mut actor, refused_ids, refused_ids.anchor_call);
        let effects = actor.handle(CoordinatorCommand::RegisterAnchor(Box::new(
            AnchorRegistration {
                seed: seed(refused_ids, 1, TestReplay::new()),
                proposal_permit: None,
                monotonic_deadline: Instant::now() + Duration::from_secs(300),
                opened_after_ingest_seq: 0,
                loss: LossSnapshot::default(),
                admission_epoch: 0,
                limits: WindowLimits::new(1, 16, 1 << 20, &[]),
            },
        )));
        assert!(effects.iter().any(|effect| matches!(
            effect,
            CoordinatorEffect::AnchorRefused {
                anchor_id,
                reason: AnchorRefusalReason::SamplingSuppressed,
            } if *anchor_id == refused_ids.anchor
        )));
        assert!(!actor.windows.contains_key(&refused_ids.anchor));

        let invalid_ids = ids(40_500);
        let mut invalid = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut invalid, invalid_ids, 1, 16, &[], TestReplay::new());
        invalid.handle(CoordinatorCommand::PendingRecorded(
            SinkAck::DurablyDeclined {
                anchor_id: invalid_ids.anchor,
                pending_hash,
                terminal_hash: "not-a-sha256".to_string(),
                reason: DurableDeclineReason::NotScheduledQueueFull,
            },
        ));
        assert!(invalid.permanent_sampling_fault);
        assert!(invalid.scheduler_pressure_pools.is_empty());
        assert!(invalid.windows.is_empty());
    }

    #[test]
    fn temporary_recovery_does_not_clear_permanent_loss_or_shutdown_suppression() {
        for blocker in 0..3 {
            let mut actor = coordinator(1);
            actor.scheduler_pressure_pools.insert(POOL.to_string(), 1);
            actor.evidence_capacity_pressure = Some(2);
            match blocker {
                0 => {
                    actor.permanent_sampling_fault = true;
                    actor.sampling_fault_reason = Some(HEALTH_SINK_FAILURE);
                }
                1 => {
                    actor.pressure_calls.insert(uuid(41_000), 1);
                }
                _ => {
                    actor.accepting = false;
                    actor.shutdown = true;
                }
            }

            let effects = actor.handle(CoordinatorCommand::SchedulerCapacityRecovered {
                pool_id: POOL.to_string(),
                pressure_generation: 1,
            });
            assert!(
                !effects
                    .iter()
                    .any(|effect| matches!(effect, CoordinatorEffect::SamplingResumed))
            );
            let effects = actor.handle(CoordinatorCommand::EvidenceCapacityRecovered {
                pressure_generation: 2,
            });
            assert!(
                !effects
                    .iter()
                    .any(|effect| matches!(effect, CoordinatorEffect::SamplingResumed))
            );
            assert!(!actor.sampling_available());
        }
    }

    #[test]
    fn stale_capacity_recovery_cannot_clear_a_newer_pressure_generation() {
        let mut actor = coordinator(1);
        let mut effects = Vec::new();
        actor.close_scheduler_pressure(POOL.to_string(), &mut effects);
        let first_scheduler = actor.scheduler_pressure_pools[POOL];
        actor.close_scheduler_pressure(POOL.to_string(), &mut effects);
        let second_scheduler = actor.scheduler_pressure_pools[POOL];
        assert!(second_scheduler > first_scheduler);

        actor.handle(CoordinatorCommand::SchedulerCapacityRecovered {
            pool_id: POOL.to_string(),
            pressure_generation: first_scheduler,
        });
        assert_eq!(
            actor.scheduler_pressure_pools.get(POOL),
            Some(&second_scheduler)
        );
        actor.handle(CoordinatorCommand::SchedulerCapacityRecovered {
            pool_id: POOL.to_string(),
            pressure_generation: second_scheduler,
        });
        assert!(!actor.scheduler_pressure_pools.contains_key(POOL));

        actor.close_evidence_capacity_pressure(&mut effects);
        let first_evidence = actor.evidence_capacity_pressure.unwrap();
        actor.close_evidence_capacity_pressure(&mut effects);
        let second_evidence = actor.evidence_capacity_pressure.unwrap();
        assert!(second_evidence > first_evidence);

        actor.handle(CoordinatorCommand::EvidenceCapacityRecovered {
            pressure_generation: first_evidence,
        });
        assert_eq!(actor.evidence_capacity_pressure, Some(second_evidence));
        actor.handle(CoordinatorCommand::EvidenceCapacityRecovered {
            pressure_generation: second_evidence,
        });
        assert!(actor.evidence_capacity_pressure.is_none());
    }

    #[test]
    fn shutdown_waits_for_inflight_pending_ack_and_pressure_is_uuid_associated() {
        let anchor_ids = ids(600);
        let mut actor = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut actor, anchor_ids, 1, 16, &[], TestReplay::new());
        let later_call = uuid(610);
        register_primary(&mut actor, anchor_ids, later_call);
        let effects = actor.handle(CoordinatorCommand::Shutdown {
            admission_epoch: 1,
            loss: LossSnapshot::default(),
        });
        assert!(
            !effects
                .iter()
                .any(|effect| matches!(effect, CoordinatorEffect::Drained))
        );
        actor.handle(observed(llm_end(
            1,
            anchor_ids.anchor_call,
            anchor_ids.owner,
            "post-stop-anchor",
        )));
        actor.handle(observed(llm_end(
            2,
            later_call,
            anchor_ids.owner,
            "post-stop-progress",
        )));
        let effects = actor.handle(CoordinatorCommand::PendingRecorded(SinkAck::Applied {
            anchor_id: anchor_ids.anchor,
            payload_hash: pending_hash,
        }));
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::CanceledBeforeAnchorEnd
            }
        );

        let mut pressure = coordinator(1);
        for value in 0..pressure.limits.primary_call_capacity {
            pressure
                .dropped_staged_calls
                .insert(uuid(10_000 + value as u128), 1);
        }
        let duplicate_dropped = *pressure.dropped_staged_calls.keys().next().unwrap();
        let mut effects = Vec::new();
        pressure.record_dropped_staged(duplicate_dropped, 2, &mut effects);
        assert_eq!(pressure.snapshot().unmatched_pressure, 0);
        let pressured_call = uuid(99_999);
        pressure.record_dropped_staged(pressured_call, 2, &mut effects);
        assert_eq!(pressure.snapshot().unmatched_pressure, 1);
        pressure.clear_unregistered_call(uuid(99_998), None, &mut effects);
        assert_eq!(pressure.snapshot().unmatched_pressure, 1);
        pressure.clear_unregistered_call(pressured_call, None, &mut effects);
        assert_eq!(pressure.snapshot().unmatched_pressure, 0);

        for value in 0..pressure.limits.primary_call_capacity {
            pressure
                .pressure_calls
                .insert(uuid(20_000 + value as u128), 3);
        }
        let overflowed_pressure_call = uuid(199_999);
        pressure.record_dropped_staged(overflowed_pressure_call, 4, &mut effects);
        assert!(!pressure.permanent_sampling_fault);
        assert_eq!(pressure.pressure_overflow_count, 1);
        let dropped_call = *pressure.dropped_staged_calls.keys().next().unwrap();
        pressure.clear_unregistered_call(dropped_call, None, &mut effects);
        assert_eq!(pressure.pressure_overflow_count, 1);
        pressure.clear_unregistered_call(overflowed_pressure_call, None, &mut effects);
        assert_eq!(pressure.pressure_overflow_count, 0);
        pressure.pressure_overflow_count = 1;
        let pressure_ids = ids(700);
        register_primary(&mut pressure, pressure_ids, overflowed_pressure_call);
        assert_eq!(pressure.pressure_overflow_count, 0);
        assert!(!pressure.permanent_sampling_fault);
    }

    #[test]
    fn resolved_same_root_v1_end_rejects_after_staged_start_loss() {
        let ids = ids(80_000);
        let mut actor = coordinator(1);
        let (pending_hash, _) = register_anchor(&mut actor, ids, 1, 16, &[], TestReplay::new());
        ack_pending(&mut actor, ids.anchor, pending_hash);
        actor.handle(observed(llm_end(1, ids.anchor_call, ids.owner, "anchor")));

        actor.limits.staging_capacity = 0;
        let parent_uuid = uuid(80_010);
        let v1_call_uuid = uuid(80_011);
        let effects = actor.handle(observed(event(
            2,
            v1_call_uuid,
            Some(parent_uuid),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("llm"),
            Some(LlmCallRole::Primary),
            "v1",
            Some(ScopeType::Llm),
            None,
            "lost-start",
        )));
        assert!(
            effects
                .iter()
                .all(|effect| !matches!(effect, CoordinatorEffect::RecordTerminal { .. }))
        );
        assert_eq!(actor.dropped_staged_calls.get(&v1_call_uuid), Some(&2));

        actor.handle(observed(event(
            3,
            parent_uuid,
            Some(ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("function"),
            None,
            "parent",
            Some(ScopeType::Function),
            None,
            "resolved-parent",
        )));
        let effects = actor.handle(observed(event(
            4,
            v1_call_uuid,
            Some(parent_uuid),
            CapturedEventKind::Scope,
            Some(ScopeCategory::End),
            Some("llm"),
            Some(LlmCallRole::Primary),
            "v1",
            Some(ScopeType::Llm),
            None,
            "resolved-end",
        )));
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::EventLoss
            }
        );
        assert!(!actor.dropped_staged_calls.contains_key(&v1_call_uuid));
    }

    #[test]
    fn resolved_same_root_v1_end_rejects_retained_staged_evidence() {
        let ids = ids(81_000);
        let mut actor = coordinator(1);
        let (pending_hash, _) = register_anchor(&mut actor, ids, 1, 16, &[], TestReplay::new());
        ack_pending(&mut actor, ids.anchor, pending_hash);
        actor.handle(observed(llm_end(1, ids.anchor_call, ids.owner, "anchor")));

        let v1_call_uuid = uuid(81_010);
        actor.handle(observed(event(
            2,
            v1_call_uuid,
            Some(uuid(81_011)),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("llm"),
            Some(LlmCallRole::Primary),
            "v1",
            Some(ScopeType::Llm),
            None,
            "staged-start",
        )));
        assert!(actor.staged.contains_key(&v1_call_uuid));

        let effects = actor.handle(observed(event(
            3,
            v1_call_uuid,
            Some(ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::End),
            Some("llm"),
            Some(LlmCallRole::Primary),
            "v1",
            Some(ScopeType::Llm),
            None,
            "resolved-end",
        )));
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::EventLoss
            }
        );
        assert!(!actor.staged.contains_key(&v1_call_uuid));
    }

    #[test]
    fn resolved_v1_end_settles_cutoff_pressure_for_captured_and_oversized_events() {
        for (offset, oversized) in [(82_000, false), (83_000, true)] {
            let ids = ids(offset);
            let mut actor = coordinator(1);
            let (pending_hash, _) =
                register_anchor(&mut actor, ids, 1, 16, &["compaction"], TestReplay::new());
            ack_pending(&mut actor, ids.anchor, pending_hash);
            actor.handle(observed(llm_end(1, ids.anchor_call, ids.owner, "anchor")));

            let v1_call_uuid = uuid(offset + 10);
            actor.pressure_calls.insert(v1_call_uuid, 2);
            let effects = actor.handle(observed(event(
                3,
                uuid(offset + 11),
                Some(ids.owner),
                CapturedEventKind::Mark,
                None,
                None,
                None,
                "compaction",
                None,
                None,
                "compaction",
            )));
            assert!(
                effects
                    .iter()
                    .all(|effect| !matches!(effect, CoordinatorEffect::RecordTerminal { .. }))
            );
            assert!(
                actor.windows[&ids.anchor]
                    .provisional
                    .as_ref()
                    .is_some_and(|terminal| terminal.uncertain_uuids.contains(&v1_call_uuid))
            );

            let effects = if oversized {
                actor.handle(CoordinatorCommand::OversizedEvent {
                    event: ProjectedTrajectoryEvent::Oversized(OversizedTrajectoryEvent {
                        ingest_seq: 4,
                        event_uuid: v1_call_uuid,
                        parent_uuid: Some(ids.owner),
                        kind: CapturedEventKind::Scope,
                        scope_phase: Some(ScopeCategory::End),
                        category: Some("llm".into()),
                        call_role: Some(LlmCallRole::Primary),
                        scope_type: Some(ScopeType::Llm),
                    }),
                    loss: LossSnapshot::default(),
                })
            } else {
                actor.handle(observed(event(
                    4,
                    v1_call_uuid,
                    Some(ids.owner),
                    CapturedEventKind::Scope,
                    Some(ScopeCategory::End),
                    Some("llm"),
                    Some(LlmCallRole::Primary),
                    "v1",
                    Some(ScopeType::Llm),
                    None,
                    "resolved-end",
                )))
            };
            let (_, terminal) = terminal_effect(effects);
            assert_eq!(
                terminal.state,
                TrajectoryTerminalStateV1::Rejected {
                    reason: TrajectoryRejectionReason::EventLoss
                }
            );
            assert!(!actor.pressure_calls.contains_key(&v1_call_uuid));
        }
    }

    #[test]
    fn resolved_v1_end_does_not_poison_window_for_pre_boundary_pressure() {
        let ids = ids(84_000);
        let mut actor = coordinator(1);
        let (pending_hash, _) = register_anchor(&mut actor, ids, 1, 16, &[], TestReplay::new());
        ack_pending(&mut actor, ids.anchor, pending_hash);
        let v1_call_uuid = uuid(84_010);
        actor.pressure_calls.insert(v1_call_uuid, 1);
        actor.handle(observed(llm_end(2, ids.anchor_call, ids.owner, "anchor")));

        let effects = actor.handle(observed(event(
            3,
            v1_call_uuid,
            Some(ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::End),
            Some("llm"),
            Some(LlmCallRole::Primary),
            "v1",
            Some(ScopeType::Llm),
            None,
            "resolved-end",
        )));
        assert!(
            effects
                .iter()
                .all(|effect| !matches!(effect, CoordinatorEffect::RecordTerminal { .. }))
        );
        assert!(!actor.pressure_calls.contains_key(&v1_call_uuid));

        let later_call = uuid(84_011);
        register_primary(&mut actor, ids, later_call);
        let (_, terminal) =
            terminal_effect(actor.handle(observed(llm_end(4, later_call, ids.owner, "progress"))));
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached
            }
        );
    }

    #[test]
    fn terminal_ack_mismatch_and_conflict_retain_frozen_recovery_authority() {
        for (case, offset) in [
            (0, 30_000),
            (1, 30_100),
            (2, 30_200),
            (3, 30_300),
            (4, 30_400),
        ] {
            let test_ids = ids(offset);
            let mut actor = coordinator(1);
            let (pending_hash, _) =
                register_anchor(&mut actor, test_ids, 1, 16, &[], TestReplay::new());
            ack_pending(&mut actor, test_ids.anchor, pending_hash.clone());
            actor.handle(observed(llm_end(
                1,
                test_ids.anchor_call,
                test_ids.owner,
                "anchor",
            )));
            let later_call = uuid(offset + 20);
            register_primary(&mut actor, test_ids, later_call);
            let (terminal_hash, _) = terminal_effect(actor.handle(observed(llm_end(
                2,
                later_call,
                test_ids.owner,
                "later",
            ))));
            let ack = match case {
                0 => SinkAck::Applied {
                    anchor_id: test_ids.anchor,
                    payload_hash: "wrong-terminal-hash".into(),
                },
                1 => SinkAck::AlreadyTerminal {
                    anchor_id: test_ids.anchor,
                    pending_hash: "wrong-pending-hash".into(),
                    terminal_hash: terminal_hash.clone(),
                },
                2 => SinkAck::Conflict {
                    anchor_id: test_ids.anchor,
                },
                3 => SinkAck::DurablyDeclined {
                    anchor_id: test_ids.anchor,
                    pending_hash: pending_hash.clone(),
                    terminal_hash: "a".repeat(64),
                    reason: DurableDeclineReason::NotScheduledQueueFull,
                },
                _ => SinkAck::TransientRefused {
                    anchor_id: test_ids.anchor,
                    reason: TransientRefusalReason::EvidenceCapacity,
                },
            };

            let effects = actor.handle(CoordinatorCommand::TerminalRecorded(ack));

            assert!(effects.iter().any(|effect| matches!(
                effect,
                CoordinatorEffect::ScheduleSinkRetry { anchor_id, generation: 1 }
                    if *anchor_id == test_ids.anchor
            )));
            let window = &actor.windows[&test_ids.anchor];
            assert!(window.accepted);
            assert_eq!(window.phase, WindowPhase::PersistingTerminal);
            assert_eq!(
                window.frozen.as_ref().unwrap().payload_hash.as_deref(),
                Some(terminal_hash.as_str())
            );
            assert!(actor.permanent_sampling_fault);
            let effects = actor.handle(CoordinatorCommand::Shutdown {
                admission_epoch: 1,
                loss: LossSnapshot::default(),
            });
            assert!(
                !effects
                    .iter()
                    .any(|effect| matches!(effect, CoordinatorEffect::Drained))
            );
        }
    }

    #[test]
    fn provisional_cutoff_ignores_post_cutoff_loss_root_fault_and_oversize() {
        let test_ids = ids(31_000);
        let mut actor = coordinator(1);
        let (pending_hash, _) = register_anchor(
            &mut actor,
            test_ids,
            1,
            16,
            &["compaction"],
            TestReplay::new(),
        );
        ack_pending(&mut actor, test_ids.anchor, pending_hash);
        actor.handle(observed(llm_end(
            1,
            test_ids.anchor_call,
            test_ids.owner,
            "anchor",
        )));
        let future_parent = uuid(31_020);
        let late_call = uuid(31_021);
        actor.handle(observed(event(
            2,
            late_call,
            Some(future_parent),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("llm"),
            Some(LlmCallRole::Primary),
            "late-start",
            Some(ScopeType::Llm),
            None,
            "late-start",
        )));
        actor.handle(observed(event(
            3,
            uuid(31_022),
            Some(test_ids.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "compaction",
            None,
            None,
            "compaction",
        )));
        assert_eq!(
            actor.windows[&test_ids.anchor]
                .provisional
                .as_ref()
                .unwrap()
                .cutoff_ingest_seq,
            3
        );
        actor.handle(CoordinatorCommand::OversizedEvent {
            event: ProjectedTrajectoryEvent::Oversized(OversizedTrajectoryEvent {
                ingest_seq: 4,
                event_uuid: late_call,
                parent_uuid: Some(future_parent),
                kind: CapturedEventKind::Scope,
                scope_phase: Some(ScopeCategory::Start),
                category: Some("llm".into()),
                call_role: Some(LlmCallRole::Primary),
                scope_type: Some(ScopeType::Llm),
            }),
            loss: LossSnapshot::default(),
        });
        let loss_ids = ids(31_100);
        actor.handle(CoordinatorCommand::RegisterPrimaryCall {
            registration: primary_registration(loss_ids, loss_ids.anchor_call),
            loss: LossSnapshot {
                highest_dropped_ingest_seq: 4,
                classification_loss_epoch: 1,
            },
            admission_epoch: 0,
        });
        let unrelated = uuid(31_030);
        register_primary(&mut actor, test_ids, unrelated);
        actor.handle(CoordinatorCommand::RegisterPrimaryCall {
            registration: PrimaryCallRegistration {
                parent_uuid: uuid(31_031),
                ..primary_registration(test_ids, unrelated)
            },
            loss: LossSnapshot::default(),
            admission_epoch: 0,
        });
        assert!(actor.windows[&test_ids.anchor].provisional.is_some());

        let effects = actor.handle(CoordinatorCommand::RegisterPrimaryCall {
            registration: nested_primary_registration(test_ids, late_call, future_parent),
            loss: LossSnapshot::default(),
            admission_epoch: 0,
        });
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached
            }
        );
        assert_eq!(
            terminal
                .events
                .iter()
                .map(|event| event.ingest_seq)
                .collect::<Vec<_>>(),
            [2, 3]
        );
    }

    #[test]
    fn pre_cutoff_staged_parent_conflicts_reject_provisional_windows() {
        let offset = 31_200;
        let test_ids = ids(offset);
        let mut actor = coordinator(1);
        let (pending_hash, _) = register_anchor(
            &mut actor,
            test_ids,
            1,
            16,
            &["compaction"],
            TestReplay::new(),
        );
        ack_pending(&mut actor, test_ids.anchor, pending_hash);
        actor.handle(observed(llm_end(
            1,
            test_ids.anchor_call,
            test_ids.owner,
            "anchor",
        )));
        let staged_parent = uuid(offset + 20);
        let registered_parent = uuid(offset + 21);
        let late_call = uuid(offset + 22);
        actor.handle(observed(event(
            2,
            late_call,
            Some(staged_parent),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("llm"),
            Some(LlmCallRole::Primary),
            "late",
            Some(ScopeType::Llm),
            None,
            "late",
        )));
        actor.handle(observed(event(
            3,
            uuid(offset + 23),
            Some(test_ids.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "compaction",
            None,
            None,
            "compaction",
        )));
        assert!(actor.windows[&test_ids.anchor].provisional.is_some());

        let effects = actor.handle(CoordinatorCommand::RegisterPrimaryCall {
            registration: nested_primary_registration(test_ids, late_call, registered_parent),
            loss: LossSnapshot::default(),
            admission_epoch: 0,
        });
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::ContradictoryOwnership
            }
        );
    }

    #[test]
    fn newly_resolved_parent_replays_each_staged_child_by_original_sequence() {
        let test_ids = ids(31_275);
        let mut actor = coordinator(1);
        let (pending_hash, _) = register_anchor(
            &mut actor,
            test_ids,
            1,
            16,
            &["compaction"],
            TestReplay::new(),
        );
        ack_pending(&mut actor, test_ids.anchor, pending_hash);
        actor.handle(observed(llm_end(
            1,
            test_ids.anchor_call,
            test_ids.owner,
            "anchor",
        )));
        let future_parent = uuid(31_295);
        let child_call = uuid(31_296);
        actor.handle(observed(event(
            2,
            child_call,
            Some(future_parent),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("llm"),
            Some(LlmCallRole::Primary),
            "legacy-child",
            Some(ScopeType::Llm),
            None,
            "legacy-child",
        )));
        actor.handle(observed(event(
            3,
            uuid(31_297),
            Some(test_ids.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "compaction",
            None,
            None,
            "compaction",
        )));
        assert!(actor.windows[&test_ids.anchor].provisional.is_some());

        let effects = actor.handle(observed(event(
            4,
            future_parent,
            Some(test_ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("function"),
            None,
            "future-parent",
            Some(ScopeType::Function),
            None,
            "future-parent",
        )));
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached
            }
        );
        assert_eq!(
            terminal
                .events
                .iter()
                .map(|event| event.ingest_seq)
                .collect::<Vec<_>>(),
            [2, 3]
        );
        assert!(!actor.staged.contains_key(&child_call));
    }

    #[test]
    fn frozen_v2_owner_path_resolution_replays_other_staged_children() {
        let test_ids = ids(31_600);
        let mut actor = coordinator(1);
        let (pending_hash, _) = register_anchor(
            &mut actor,
            test_ids,
            1,
            16,
            &["compaction"],
            TestReplay::new(),
        );
        ack_pending(&mut actor, test_ids.anchor, pending_hash);
        actor.handle(observed(llm_end(
            1,
            test_ids.anchor_call,
            test_ids.owner,
            "anchor",
        )));
        let future_parent = uuid(31_620);
        let legacy_child = uuid(31_621);
        actor.handle(observed(event(
            2,
            legacy_child,
            Some(future_parent),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("llm"),
            Some(LlmCallRole::Primary),
            "legacy-child",
            Some(ScopeType::Llm),
            None,
            "legacy-child",
        )));
        actor.handle(observed(event(
            3,
            uuid(31_622),
            Some(test_ids.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "compaction",
            None,
            None,
            "compaction",
        )));
        assert!(actor.windows[&test_ids.anchor].provisional.is_some());

        let effects = actor.handle(CoordinatorCommand::RegisterPrimaryCall {
            registration: nested_primary_registration(test_ids, uuid(31_623), future_parent),
            loss: LossSnapshot::default(),
            admission_epoch: 0,
        });
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached
            }
        );
        assert_eq!(
            terminal
                .events
                .iter()
                .map(|event| event.ingest_seq)
                .collect::<Vec<_>>(),
            [2, 3]
        );
        assert!(!actor.staged.contains_key(&legacy_child));
    }

    #[test]
    fn full_frozen_path_validation_precedes_staged_child_replay() {
        let test_ids = ids(31_700);
        let mut actor = coordinator(1);
        let (pending_hash, _) = register_anchor(
            &mut actor,
            test_ids,
            1,
            16,
            &["compaction"],
            TestReplay::new(),
        );
        ack_pending(&mut actor, test_ids.anchor, pending_hash);
        actor.handle(observed(llm_end(
            1,
            test_ids.anchor_call,
            test_ids.owner,
            "anchor",
        )));
        let future_parent = uuid(31_720);
        let reused_scope = uuid(31_721);
        actor.handle(observed(event(
            2,
            reused_scope,
            Some(future_parent),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("llm"),
            Some(LlmCallRole::Primary),
            "legacy-child",
            Some(ScopeType::Llm),
            None,
            "legacy-child",
        )));
        actor.handle(observed(event(
            3,
            uuid(31_722),
            Some(test_ids.owner),
            CapturedEventKind::Mark,
            None,
            None,
            None,
            "compaction",
            None,
            None,
            "compaction",
        )));
        assert!(actor.windows[&test_ids.anchor].provisional.is_some());

        let effects = actor.handle(CoordinatorCommand::RegisterPrimaryCall {
            registration: PrimaryCallRegistration {
                call_uuid: uuid(31_723),
                root_uuid: test_ids.root,
                parent_uuid: reused_scope,
                owner_uuid: test_ids.owner,
                owner_path: vec![
                    TrajectoryOwnerScopeV1 {
                        uuid: test_ids.owner,
                        name: "agent".into(),
                        scope_type: ScopeType::Agent,
                    },
                    TrajectoryOwnerScopeV1 {
                        uuid: future_parent,
                        name: "future-parent".into(),
                        scope_type: ScopeType::Function,
                    },
                    TrajectoryOwnerScopeV1 {
                        uuid: reused_scope,
                        name: "reused-as-function".into(),
                        scope_type: ScopeType::Function,
                    },
                ],
                api_family: LlmApiFamily::OpenAIChatCompletions,
            },
            loss: LossSnapshot::default(),
            admission_epoch: 0,
        });
        let (_, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::ContradictoryOwnership
            }
        );
    }

    #[test]
    fn deadline_shutdown_and_failed_barrier_bound_unresolved_provisional_windows() {
        for (mode, offset) in [(0, 31_300), (1, 31_400), (2, 31_500)] {
            let test_ids = ids(offset);
            let mut actor = coordinator(1);
            let (pending_hash, generation) = register_anchor(
                &mut actor,
                test_ids,
                1,
                16,
                &["compaction"],
                TestReplay::new(),
            );
            ack_pending(&mut actor, test_ids.anchor, pending_hash);
            actor.handle(observed(llm_end(
                1,
                test_ids.anchor_call,
                test_ids.owner,
                "anchor",
            )));
            actor.handle(observed(event(
                2,
                uuid(offset + 20),
                Some(uuid(offset + 21)),
                CapturedEventKind::Scope,
                Some(ScopeCategory::Start),
                Some("llm"),
                Some(LlmCallRole::Primary),
                "unknown",
                Some(ScopeType::Llm),
                None,
                "unknown",
            )));
            actor.handle(observed(event(
                3,
                uuid(offset + 22),
                Some(test_ids.owner),
                CapturedEventKind::Mark,
                None,
                None,
                None,
                "compaction",
                None,
                None,
                "compaction",
            )));
            assert!(actor.windows[&test_ids.anchor].provisional.is_some());

            let effects = match mode {
                0 => actor.handle(CoordinatorCommand::DeadlineElapsed {
                    anchor_id: test_ids.anchor,
                    generation,
                    loss: LossSnapshot::default(),
                }),
                1 => actor.handle(CoordinatorCommand::Shutdown {
                    admission_epoch: 1,
                    loss: LossSnapshot::default(),
                }),
                _ => actor.handle(CoordinatorCommand::DeadlineBarrierFailed {
                    anchor_id: test_ids.anchor,
                    generation,
                    loss: LossSnapshot::default(),
                }),
            };
            let (_, terminal) = terminal_effect(effects);
            assert_eq!(
                terminal.state,
                TrajectoryTerminalStateV1::Rejected {
                    reason: if mode == 2 {
                        TrajectoryRejectionReason::RejectedDeliveryBarrier
                    } else {
                        TrajectoryRejectionReason::EventLoss
                    }
                }
            );
        }
    }

    #[test]
    fn oversized_registered_primary_is_validated_while_internal_role_is_ignored() {
        for (shadow_role, offset, expected) in [
            (
                false,
                32_000,
                TrajectoryRejectionReason::ContradictoryOwnership,
            ),
            (true, 32_100, TrajectoryRejectionReason::EventLoss),
        ] {
            let test_ids = ids(offset);
            let mut actor = coordinator(1);
            let (pending_hash, _) =
                register_anchor(&mut actor, test_ids, 1, 16, &[], TestReplay::new());
            ack_pending(&mut actor, test_ids.anchor, pending_hash);
            let effects = actor.handle(CoordinatorCommand::OversizedEvent {
                event: ProjectedTrajectoryEvent::Oversized(OversizedTrajectoryEvent {
                    ingest_seq: 1,
                    event_uuid: test_ids.anchor_call,
                    parent_uuid: Some(if shadow_role {
                        test_ids.owner
                    } else {
                        uuid(offset + 50)
                    }),
                    kind: CapturedEventKind::Scope,
                    scope_phase: Some(ScopeCategory::End),
                    category: Some("llm".into()),
                    call_role: Some(if shadow_role {
                        LlmCallRole::Shadow
                    } else {
                        LlmCallRole::Primary
                    }),
                    scope_type: Some(ScopeType::Llm),
                }),
                loss: LossSnapshot::default(),
            });
            if shadow_role {
                assert!(effects.is_empty());
                assert!(actor.windows[&test_ids.anchor].frozen.is_none());
                assert!(
                    actor.windows[&test_ids.anchor]
                        .capture_after_ingest_seq
                        .is_none()
                );
                continue;
            }
            let (terminal_hash, terminal) = terminal_effect(effects);
            assert_eq!(
                terminal.state,
                TrajectoryTerminalStateV1::Rejected { reason: expected }
            );
            assert!(
                actor.windows[&test_ids.anchor]
                    .capture_after_ingest_seq
                    .is_none()
            );
            actor.handle(CoordinatorCommand::TerminalRecorded(SinkAck::Applied {
                anchor_id: test_ids.anchor,
                payload_hash: terminal_hash,
            }));
            assert!(!actor.ancestry.contains_key(&test_ids.anchor_call));
        }
    }

    #[test]
    fn oversized_ends_close_ancestry_and_startless_child_ends_stay_excluded() {
        let test_ids = ids(33_000);
        let mut actor = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut actor, test_ids, 1, 32, &[], TestReplay::new());
        ack_pending(&mut actor, test_ids.anchor, pending_hash);
        actor.handle(observed(llm_end(
            1,
            test_ids.anchor_call,
            test_ids.owner,
            "anchor",
        )));
        let child = uuid(33_020);
        actor.handle(observed(event(
            2,
            child,
            Some(test_ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::End),
            Some("agent"),
            None,
            "child-end-without-start",
            Some(ScopeType::Agent),
            None,
            "child-end",
        )));
        assert!(actor.windows[&test_ids.anchor].events.is_empty());

        let tool = uuid(33_021);
        actor.handle(observed(event(
            3,
            tool,
            Some(test_ids.owner),
            CapturedEventKind::Scope,
            Some(ScopeCategory::Start),
            Some("tool"),
            None,
            "tool",
            Some(ScopeType::Tool),
            None,
            "tool-start",
        )));
        let effects = actor.handle(CoordinatorCommand::OversizedEvent {
            event: ProjectedTrajectoryEvent::Oversized(OversizedTrajectoryEvent {
                ingest_seq: 4,
                event_uuid: tool,
                parent_uuid: Some(test_ids.owner),
                kind: CapturedEventKind::Scope,
                scope_phase: Some(ScopeCategory::End),
                category: Some("tool".into()),
                call_role: None,
                scope_type: Some(ScopeType::Tool),
            }),
            loss: LossSnapshot::default(),
        });
        assert!(actor.ancestry[&tool].ended);
        let (terminal_hash, terminal) = terminal_effect(effects);
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Rejected {
                reason: TrajectoryRejectionReason::Overflow
            }
        );
        actor.handle(CoordinatorCommand::TerminalRecorded(SinkAck::Applied {
            anchor_id: test_ids.anchor,
            payload_hash: terminal_hash,
        }));
        assert!(!actor.ancestry.contains_key(&tool));
    }

    #[test]
    fn owner_shutdown_and_progress_tie_emit_exact_partial_semantics() {
        let owner_ids = ids(34_000);
        let mut owner = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut owner, owner_ids, 2, 16, &[], TestReplay::new());
        ack_pending(&mut owner, owner_ids.anchor, pending_hash);
        owner.handle(observed(llm_end(
            1,
            owner_ids.anchor_call,
            owner_ids.owner,
            "anchor",
        )));
        let (_, terminal) = terminal_effect(owner.handle(observed(event(
            2,
            owner_ids.owner,
            Some(owner_ids.root),
            CapturedEventKind::Scope,
            Some(ScopeCategory::End),
            Some("agent"),
            None,
            "owner",
            Some(ScopeType::Agent),
            None,
            "owner-end",
        ))));
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::OwnerTerminated
            }
        );
        assert!(terminal.is_partial);
        assert!(!terminal.promotion_eligible);

        let shutdown_ids = ids(34_100);
        let mut shutdown = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut shutdown, shutdown_ids, 2, 16, &[], TestReplay::new());
        ack_pending(&mut shutdown, shutdown_ids.anchor, pending_hash);
        shutdown.handle(observed(llm_end(
            1,
            shutdown_ids.anchor_call,
            shutdown_ids.owner,
            "anchor",
        )));
        let (_, terminal) = terminal_effect(shutdown.handle(CoordinatorCommand::Shutdown {
            admission_epoch: 1,
            loss: LossSnapshot::default(),
        }));
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::Shutdown
            }
        );
        assert!(terminal.is_partial);
        assert!(!terminal.promotion_eligible);

        let tie_ids = ids(34_200);
        let mut tie = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut tie, tie_ids, 1, 16, &["compaction"], TestReplay::new());
        ack_pending(&mut tie, tie_ids.anchor, pending_hash);
        tie.handle(observed(llm_end(
            1,
            tie_ids.anchor_call,
            tie_ids.owner,
            "anchor",
        )));
        let (_, terminal) = terminal_effect(tie.handle(observed(event(
            2,
            tie_ids.owner,
            Some(tie_ids.root),
            CapturedEventKind::Mark,
            Some(ScopeCategory::End),
            Some("agent"),
            None,
            "compaction",
            Some(ScopeType::Agent),
            None,
            "tie",
        ))));
        assert_eq!(
            terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached
            }
        );
        assert!(!terminal.is_partial);
    }

    #[test]
    fn ordinary_evidence_is_retained_while_sibling_and_turn_owners_are_excluded() {
        let test_ids = ids(35_000);
        let mut actor = coordinator(1);
        let (pending_hash, _) =
            register_anchor(&mut actor, test_ids, 1, 64, &[], TestReplay::new());
        ack_pending(&mut actor, test_ids.anchor, pending_hash);
        actor.handle(observed(llm_end(
            1,
            test_ids.anchor_call,
            test_ids.owner,
            "anchor",
        )));
        let tool = uuid(35_020);
        for ordinary in [
            event(
                2,
                tool,
                Some(test_ids.owner),
                CapturedEventKind::Scope,
                Some(ScopeCategory::Start),
                Some("tool"),
                None,
                "tool",
                Some(ScopeType::Tool),
                None,
                "tool-start",
            ),
            event(
                3,
                uuid(35_021),
                Some(tool),
                CapturedEventKind::Mark,
                None,
                Some("tool"),
                None,
                "tool-result",
                None,
                None,
                "tool-result",
            ),
            event(
                4,
                uuid(35_022),
                Some(test_ids.owner),
                CapturedEventKind::Mark,
                None,
                None,
                None,
                "checkpoint",
                None,
                None,
                "mark",
            ),
            event(
                5,
                uuid(35_023),
                Some(test_ids.owner),
                CapturedEventKind::Scope,
                Some(ScopeCategory::Start),
                Some("custom"),
                None,
                "custom",
                Some(ScopeType::Custom),
                None,
                "custom-start",
            ),
        ] {
            actor.handle(observed(ordinary));
        }
        assert_eq!(actor.windows[&test_ids.anchor].observed_progress, 0);

        let sibling = ids(35_100);
        register_primary(&mut actor, sibling, sibling.anchor_call);
        actor.handle(observed(llm_end(
            6,
            sibling.anchor_call,
            sibling.owner,
            "sibling",
        )));
        let turn_root = uuid(35_200);
        let turn_owner = uuid(35_201);
        let turn_call = uuid(35_202);
        actor.handle(CoordinatorCommand::RegisterPrimaryCall {
            registration: PrimaryCallRegistration {
                call_uuid: turn_call,
                root_uuid: turn_root,
                parent_uuid: turn_owner,
                owner_uuid: turn_owner,
                owner_path: vec![TrajectoryOwnerScopeV1 {
                    uuid: turn_owner,
                    name: "turn".into(),
                    scope_type: ScopeType::Function,
                }],
                api_family: LlmApiFamily::OpenAIChatCompletions,
            },
            loss: LossSnapshot::default(),
            admission_epoch: 0,
        });
        actor.handle(observed(llm_end(7, turn_call, turn_owner, "turn")));

        let later_call = uuid(35_300);
        register_primary(&mut actor, test_ids, later_call);
        let (_, terminal) = terminal_effect(actor.handle(observed(llm_end(
            8,
            later_call,
            test_ids.owner,
            "later",
        ))));
        assert_eq!(terminal.observed_progress, 1);
        let names = terminal
            .events
            .iter()
            .map(|event| event.name.as_str())
            .collect::<Vec<_>>();
        for retained in ["tool", "tool-result", "checkpoint", "custom", "llm"] {
            assert!(
                names.contains(&retained),
                "{retained} missing from retained names {names:?}"
            );
        }
        let event_uuids = terminal
            .events
            .iter()
            .map(|event| event.event_uuid)
            .collect::<BTreeSet<_>>();
        assert!(event_uuids.contains(&later_call));
        assert!(!event_uuids.contains(&sibling.anchor_call));
        assert!(!event_uuids.contains(&turn_call));
    }

    #[test]
    fn overlapping_windows_with_different_deadlines_close_independently() {
        let first = ids(36_000);
        let second = Ids {
            root: first.root,
            owner: first.owner,
            anchor_call: uuid(36_020),
            anchor: uuid(36_021),
        };
        let mut actor = coordinator(2);
        let (first_pending, first_generation) =
            register_anchor(&mut actor, first, 2, 16, &[], TestReplay::new());
        ack_pending(&mut actor, first.anchor, first_pending);
        let (second_pending, second_generation) =
            register_anchor(&mut actor, second, 2, 16, &[], TestReplay::new());
        assert_ne!(first_generation, second_generation);
        ack_pending(&mut actor, second.anchor, second_pending);
        actor.handle(observed(llm_end(
            1,
            first.anchor_call,
            first.owner,
            "first",
        )));
        actor.handle(observed(llm_end(
            2,
            second.anchor_call,
            second.owner,
            "second",
        )));

        let (first_terminal_hash, first_terminal) =
            terminal_effect(actor.handle(CoordinatorCommand::DeadlineElapsed {
                anchor_id: first.anchor,
                generation: first_generation,
                loss: LossSnapshot::default(),
            }));
        assert_eq!(
            first_terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::DeadlineElapsed
            }
        );
        assert!(actor.windows[&second.anchor].frozen.is_none());
        actor.handle(CoordinatorCommand::DeadlineElapsed {
            anchor_id: second.anchor,
            generation: second_generation,
            loss: LossSnapshot::default(),
        });
        let effects = actor.handle(CoordinatorCommand::TerminalRecorded(SinkAck::Applied {
            anchor_id: first.anchor,
            payload_hash: first_terminal_hash,
        }));
        let (_, second_terminal) = terminal_effect(effects);
        assert_eq!(
            second_terminal.state,
            TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::DeadlineElapsed
            }
        );
    }

    #[test]
    fn sustained_interleaved_v1_v2_traffic_cleans_every_bounded_index() {
        let mut actor = coordinator(1);
        let iterations = actor.limits.primary_call_capacity.saturating_mul(3);
        let mut seq = 0_u64;
        for index in 0..iterations {
            let offset = 40_000 + (index as u128).saturating_mul(10);
            let call_ids = ids(offset);
            register_primary(&mut actor, call_ids, call_ids.anchor_call);
            seq = seq.saturating_add(1);
            actor.handle(observed(llm_end(
                seq,
                call_ids.anchor_call,
                call_ids.owner,
                "v2-end",
            )));
            seq = seq.saturating_add(1);
            actor.handle(observed(event(
                seq,
                call_ids.owner,
                Some(call_ids.root),
                CapturedEventKind::Scope,
                Some(ScopeCategory::End),
                Some("agent"),
                None,
                "owner-end",
                Some(ScopeType::Agent),
                None,
                "owner-end",
            )));

            let v1_call = uuid(offset + 8);
            let unknown_parent = uuid(offset + 9);
            seq = seq.saturating_add(1);
            actor.handle(observed(event(
                seq,
                v1_call,
                Some(unknown_parent),
                CapturedEventKind::Scope,
                Some(ScopeCategory::Start),
                Some("llm"),
                Some(LlmCallRole::Primary),
                "v1-start",
                Some(ScopeType::Llm),
                None,
                "v1-start",
            )));
            seq = seq.saturating_add(1);
            actor.handle(observed(event(
                seq,
                v1_call,
                Some(unknown_parent),
                CapturedEventKind::Scope,
                Some(ScopeCategory::End),
                Some("llm"),
                Some(LlmCallRole::Primary),
                "v1-end",
                Some(ScopeType::Llm),
                None,
                "v1-end",
            )));
        }

        let snapshot = actor.snapshot();
        assert_eq!(snapshot.registered_primary_calls, 0);
        assert_eq!(snapshot.ancestry_entries, 0);
        assert_eq!(snapshot.staged_calls, 0);
        assert_eq!(snapshot.unmatched_pressure, 0);
        assert!(snapshot.sampling_enabled);
        assert!(!actor.permanent_sampling_fault);
    }
}
