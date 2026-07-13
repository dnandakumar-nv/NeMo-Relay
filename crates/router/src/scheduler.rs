// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded Shadow and Judge scheduler actor.

use std::collections::{BTreeMap, VecDeque, btree_map::VacantEntry};
use std::future::Future;
use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use chrono::Utc;
use nemo_relay::api::llm::LlmApiFamily;
use nemo_relay::json::Json;
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore, mpsc, watch};
use tokio::task::JoinSet;
use uuid::Uuid;

use crate::adapter::FamilyAdapter;
use crate::config::{
    JUDGE_OUTPUT_SCHEMA_SHA256_V1, JUDGE_PROMPT_TEMPLATE_SHA256_V1,
    JUDGE_RUBRIC_TEMPLATE_SHA256_V1, PoolConfig, RouterConfig,
};
use crate::evaluator::{
    ActiveReplayRegistry, BackgroundStartGate, EvaluatorCancellation, EvaluatorLinkage,
    ManagedReplayFailure, ManagedReplayOutcome, ManagedReplayParams, execute_judge_replay,
    execute_shadow_replay,
};
use crate::judge::{
    JudgeAttemptOutcomeV1, JudgeAttemptV1, JudgeHorizonV1, JudgePolicyIdentityV1, JudgeRegistryV1,
    PairwiseJudgeInputV1, resolve_judge_attempt_progress, validate_pairwise_judge_output,
};
use crate::ledger::command::{WriterFailure, WriterFailureClass};
use crate::ledger::repository::cooloff::{
    CandidateDependencyIdentity, DependencyCommandAck, DependencyCompletion,
    DependencyFailureClass, DependencyOperation, DependencyTransition, JudgeDependencyIdentity,
};
use crate::ledger::repository::judge::{
    EvaluationRecord, JudgeAttemptStart, JudgeAttemptTerminal, JudgeRecordAck,
    JudgeTransportFailureClass,
};
use crate::ledger::repository::shadow::{
    AtomicShadowVectorization, ReservedShadowAttempt, SampleBatchTerminalEvent,
    SampleBatchTerminalState, ShadowAttemptStarted, ShadowCommandAck,
    ShadowOperationalFailureClass, ShadowTerminalClass, ShadowTerminalRecord, ShadowVectorSourceV1,
    ShadowVectorizationHandoff,
};
use crate::ledger::writer::LedgerWriterClient;
use crate::response_validator::{
    CandidateValidationOperationalFailureV1, CandidateValidationOutcomeV1,
    JudgeResponseExtractionFailureV1, extract_judge_assistant_text, validate_candidate_response,
};
use crate::scheduler_admission::{SchedulerBatchPermit, SchedulerCandidatePermit};
use crate::sqlite_sink::ScheduledTrajectoryBatch;
use crate::trajectory::{RouterResponseProjectionV1, SanitizedResponseUsageV1};

const WRITER_RETRY_SLICE: Duration = Duration::from_millis(250);

/// Cloneable authority that installs one monotonic shutdown deadline.
/// Normal accepted work has no overall deadline and retries while the writer is live.
#[derive(Clone)]
pub(crate) struct SchedulerDeadlineAuthority {
    inner: Arc<SchedulerDeadlineInner>,
}

struct SchedulerDeadlineInner {
    shutdown_deadline: StdMutex<Option<Instant>>,
    changes: watch::Sender<Option<Instant>>,
}

enum DeadlineAttempt<T> {
    Completed(T),
    RetryShortened,
}

impl Default for SchedulerDeadlineAuthority {
    fn default() -> Self {
        let (changes, _) = watch::channel(None);
        Self {
            inner: Arc::new(SchedulerDeadlineInner {
                shutdown_deadline: StdMutex::new(None),
                changes,
            }),
        }
    }
}

impl SchedulerDeadlineAuthority {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Install or shorten the scheduler-wide shutdown deadline synchronously.
    pub(crate) fn install_shutdown_deadline(&self, deadline: Instant) -> bool {
        let mut current = self
            .inner
            .shutdown_deadline
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if current.is_some_and(|current| current <= deadline) {
            return false;
        }
        *current = Some(deadline);
        self.inner.changes.send_replace(Some(deadline));
        true
    }

    fn attempt_deadline(&self) -> Instant {
        self.attempt_deadline_for(WRITER_RETRY_SLICE)
    }

    fn attempt_deadline_for(&self, retry_slice: Duration) -> Instant {
        let slice_deadline = Instant::now()
            .checked_add(retry_slice)
            .unwrap_or_else(Instant::now);
        self.shutdown_deadline()
            .map_or(slice_deadline, |shutdown| slice_deadline.min(shutdown))
    }

    fn should_retry(&self, error: WriterFailure) -> bool {
        error.class() == WriterFailureClass::Deadline
            && self
                .shutdown_deadline()
                .is_none_or(|deadline| Instant::now() < deadline)
    }

    fn shutdown_deadline(&self) -> Option<Instant> {
        *self
            .inner
            .shutdown_deadline
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    async fn run_attempt<T, F, Fut>(&self, build: F) -> DeadlineAttempt<T>
    where
        F: FnOnce(Instant) -> Fut,
        Fut: Future<Output = T>,
    {
        let mut changes = self.inner.changes.subscribe();
        let attempt_deadline = self.attempt_deadline();
        let attempt = build(attempt_deadline);
        tokio::pin!(attempt);

        loop {
            tokio::select! {
                biased;
                changed = changes.changed() => {
                    if changed.is_err() {
                        return DeadlineAttempt::Completed(attempt.await);
                    }
                    let installed = *changes.borrow_and_update();
                    if installed.is_some_and(|deadline| deadline < attempt_deadline) {
                        return DeadlineAttempt::RetryShortened;
                    }
                }
                result = &mut attempt => return DeadlineAttempt::Completed(result),
            }
        }
    }
}

/// Stable scheduler failure classes. No provider response or writer detail is retained.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SchedulerFailureClass {
    InvalidBatch,
    InvalidConfiguration,
    DurableConflict,
    OriginatingProcessNotLive,
    Writer(WriterFailureClass),
    EvaluatorRuntime,
    JoinFailure,
}

/// Aggregate actor evidence suitable for deterministic concurrency tests.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SchedulerRunSummary {
    pub(crate) accepted_batches: usize,
    pub(crate) acknowledged_candidates: usize,
    pub(crate) retained_batches: usize,
    pub(crate) max_task_entries: usize,
    pub(crate) max_active_batches: usize,
    pub(crate) max_pending_candidates: usize,
    pub(crate) first_failure: Option<SchedulerFailureClass>,
    pub(crate) pool_gauges: BTreeMap<String, SchedulerPoolGaugeSnapshot>,
}

/// Per-pool provider and ready-queue high-water marks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SchedulerPoolGaugeSnapshot {
    pub(crate) shadow_max_in_flight: usize,
    pub(crate) judge_max_in_flight: usize,
    pub(crate) shadow_max_queued: usize,
    pub(crate) judge_max_queued: usize,
}

/// Scheduler completion. Retained batches deliberately keep admission authority alive.
/// Task 10 reconciliation decides when those permits may be released after failure.
pub(crate) struct SchedulerExit {
    summary: SchedulerRunSummary,
    retained_batches: Vec<Arc<BatchController>>,
    invalid_batches: Vec<ScheduledTrajectoryBatch>,
    retained_permits: Vec<AcknowledgedPermits>,
}

impl SchedulerExit {
    pub(crate) fn summary(&self) -> &SchedulerRunSummary {
        &self.summary
    }

    pub(crate) fn retained_batch_count(&self) -> usize {
        self.retained_batches.len() + self.invalid_batches.len() + self.retained_permits.len()
    }
}

/// Single-owner actor for the bounded scheduler channel and all ready queues.
pub(crate) struct ShadowScheduler {
    receiver: mpsc::Receiver<ScheduledTrajectoryBatch>,
    shared: SchedulerShared,
    pools: BTreeMap<String, PoolState>,
    tasks: JoinSet<TaskResult>,
    batches: BTreeMap<Uuid, Arc<BatchController>>,
    invalid_batches: Vec<ScheduledTrajectoryBatch>,
    retained_permits: Vec<AcknowledgedPermits>,
    accepted_batches: usize,
    acknowledged_candidates: usize,
    max_task_entries: usize,
    max_active_batches: usize,
    max_pending_candidates: usize,
    first_failure: Option<SchedulerFailureClass>,
    failure_notifier: Option<Arc<dyn Fn(SchedulerFailureClass) + Send + Sync>>,
}

/// Fully validated, side-effect-free production scheduler construction state.
pub(crate) struct ShadowSchedulerPlan {
    pools: BTreeMap<String, PoolState>,
}

#[derive(Clone)]
struct SchedulerShared {
    writer: LedgerWriterClient,
    start_gate: BackgroundStartGate,
    active_replays: ActiveReplayRegistry,
    cancellation: EvaluatorCancellation,
    deadlines: SchedulerDeadlineAuthority,
}

struct PoolState {
    config: Arc<PoolConfig>,
    candidate_capacity: usize,
    pending_candidates: usize,
    shadow_ready: VecDeque<CandidateWork>,
    judge_ready: VecDeque<JudgeWork>,
    shadow_semaphore: Arc<Semaphore>,
    judge_semaphore: Arc<Semaphore>,
    gauges: Arc<PoolGauges>,
}

#[derive(Default)]
struct PoolGauges {
    shadow_in_flight: AtomicUsize,
    shadow_max_in_flight: AtomicUsize,
    judge_in_flight: AtomicUsize,
    judge_max_in_flight: AtomicUsize,
    shadow_max_queued: AtomicUsize,
    judge_max_queued: AtomicUsize,
}

struct ProviderGaugeGuard {
    gauges: Arc<PoolGauges>,
    stage: ProviderStage,
}

enum ProviderStage {
    Shadow,
    Judge,
}

struct BatchController {
    window: Arc<crate::trajectory::ClosedTrajectoryWindow>,
    reservation: crate::ledger::repository::shadow::SampleBatchReservation,
    state: Mutex<BatchState>,
}

struct BatchState {
    remaining: usize,
    batch_permit: Option<SchedulerBatchPermit>,
    candidate_permits: Vec<Option<SchedulerCandidatePermit>>,
}

struct CandidateWork {
    batch: Arc<BatchController>,
    candidate_index: usize,
    attempt: ReservedShadowAttempt,
    pool: Arc<PoolConfig>,
}

struct JudgeWork {
    work: CandidateWork,
    candidate_response: RouterResponseProjectionV1,
    judge_input: PairwiseJudgeInputV1,
    latency_ms: u64,
    usage: Option<SanitizedResponseUsageV1>,
}

struct TerminalDraft {
    shadow_result_id: Uuid,
    state_event_id: Uuid,
    conflict_health_event_id: Uuid,
    terminal_class: ShadowTerminalClass,
    normalized_response: Option<RouterResponseProjectionV1>,
    deterministic_hard_failure: Option<crate::judge::DeterministicHardFailureV1>,
    operational_failure_class: Option<ShadowOperationalFailureClass>,
    latency_ms: Option<u64>,
    usage: Option<SanitizedResponseUsageV1>,
    evaluation_id: Option<Uuid>,
    created_at_unix_ms: i64,
}

#[allow(dead_code)] // Field ownership deliberately delays permit release until this value drops.
struct AcknowledgedPermits {
    candidate: SchedulerCandidatePermit,
    batch: Option<SchedulerBatchPermit>,
}

enum TaskResult {
    JudgeReady(Box<JudgeWork>),
    TerminalAcknowledged {
        batch_id: Uuid,
        pool_id: String,
        batch_closed: bool,
        permits: Box<AcknowledgedPermits>,
    },
    Failed {
        failure: SchedulerFailureClass,
    },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum DependencyClaim {
    Admitted,
    SkippedCooloff,
}

impl ShadowScheduler {
    /// Validate every fallible scheduler invariant before production work is spawned.
    pub(crate) fn prepare(
        config: &RouterConfig,
    ) -> Result<ShadowSchedulerPlan, SchedulerFailureClass> {
        let mut pools = BTreeMap::new();
        for pool in &config.pools {
            let candidate_capacity = pool
                .concurrency
                .max_pending
                .checked_mul(pool.max_candidates_per_sample)
                .ok_or(SchedulerFailureClass::InvalidConfiguration)?;
            if candidate_capacity == 0
                || pool.concurrency.shadow == 0
                || pool.concurrency.judge == 0
                || pools.contains_key(&pool.id)
            {
                return Err(SchedulerFailureClass::InvalidConfiguration);
            }
            pools.insert(
                pool.id.clone(),
                PoolState {
                    config: Arc::new(pool.clone()),
                    candidate_capacity,
                    pending_candidates: 0,
                    shadow_ready: VecDeque::with_capacity(candidate_capacity),
                    judge_ready: VecDeque::with_capacity(candidate_capacity),
                    shadow_semaphore: Arc::new(Semaphore::new(pool.concurrency.shadow)),
                    judge_semaphore: Arc::new(Semaphore::new(pool.concurrency.judge)),
                    gauges: Arc::new(PoolGauges::default()),
                },
            );
        }
        Ok(ShadowSchedulerPlan { pools })
    }

    #[allow(clippy::too_many_arguments)]
    #[cfg(test)]
    pub(crate) fn new(
        config: &RouterConfig,
        writer: LedgerWriterClient,
        receiver: mpsc::Receiver<ScheduledTrajectoryBatch>,
        start_gate: BackgroundStartGate,
        active_replays: ActiveReplayRegistry,
        cancellation: EvaluatorCancellation,
        deadlines: SchedulerDeadlineAuthority,
    ) -> Result<Self, SchedulerFailureClass> {
        Ok(Self::prepare(config)?.start(
            writer,
            receiver,
            start_gate,
            active_replays,
            cancellation,
            deadlines,
        ))
    }
}

impl ShadowSchedulerPlan {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn start(
        self,
        writer: LedgerWriterClient,
        receiver: mpsc::Receiver<ScheduledTrajectoryBatch>,
        start_gate: BackgroundStartGate,
        active_replays: ActiveReplayRegistry,
        cancellation: EvaluatorCancellation,
        deadlines: SchedulerDeadlineAuthority,
    ) -> ShadowScheduler {
        ShadowScheduler {
            receiver,
            shared: SchedulerShared {
                writer,
                start_gate,
                active_replays,
                cancellation,
                deadlines,
            },
            pools: self.pools,
            tasks: JoinSet::new(),
            batches: BTreeMap::new(),
            invalid_batches: Vec::new(),
            retained_permits: Vec::new(),
            accepted_batches: 0,
            acknowledged_candidates: 0,
            max_task_entries: 0,
            max_active_batches: 0,
            max_pending_candidates: 0,
            first_failure: None,
            failure_notifier: None,
        }
    }
}

impl ShadowScheduler {
    pub(crate) fn set_failure_notifier(
        &mut self,
        notifier: Arc<dyn Fn(SchedulerFailureClass) + Send + Sync>,
    ) {
        self.failure_notifier = Some(notifier);
    }

    pub(crate) async fn run(mut self) -> SchedulerExit {
        let mut receiver_closed = false;
        loop {
            self.reap_completed_tasks();
            self.dispatch_ready();
            self.observe_actor_state();
            if receiver_closed && self.tasks.is_empty() && self.queues_are_empty() {
                break;
            }

            tokio::select! {
                biased;
                result = self.tasks.join_next(), if !self.tasks.is_empty() => {
                    match result {
                        Some(Ok(result)) => self.handle_task_result(result),
                        Some(Err(_)) => self.record_failure(SchedulerFailureClass::JoinFailure),
                        None => {}
                    }
                }
                batch = self.receiver.recv(), if !receiver_closed => {
                    match batch {
                        Some(batch) => self.accept_batch(batch),
                        None => receiver_closed = true,
                    }
                }
            }
        }

        let pool_gauges = self
            .pools
            .iter()
            .map(|(pool_id, pool)| (pool_id.clone(), pool.gauges.snapshot()))
            .collect();
        let retained_batches = self.batches.into_values().collect::<Vec<_>>();
        let summary = SchedulerRunSummary {
            accepted_batches: self.accepted_batches,
            acknowledged_candidates: self.acknowledged_candidates,
            retained_batches: retained_batches.len()
                + self.invalid_batches.len()
                + self.retained_permits.len(),
            max_task_entries: self.max_task_entries,
            max_active_batches: self.max_active_batches,
            max_pending_candidates: self.max_pending_candidates,
            first_failure: self.first_failure,
            pool_gauges,
        };
        SchedulerExit {
            summary,
            retained_batches,
            invalid_batches: self.invalid_batches,
            retained_permits: self.retained_permits,
        }
    }

    fn accept_batch(&mut self, batch: ScheduledTrajectoryBatch) {
        if !batch_is_consistent(&batch, &self.pools) {
            self.record_failure(SchedulerFailureClass::InvalidBatch);
            self.invalid_batches.push(batch);
            return;
        }
        let pool_id = batch.reservation.pool_id.clone();
        let candidate_count = batch.reservation.attempts.len();
        let Some(pool) = self.pools.get(&pool_id) else {
            self.record_failure(SchedulerFailureClass::InvalidBatch);
            self.invalid_batches.push(batch);
            return;
        };
        let Some(next_pending) = pool.pending_candidates.checked_add(candidate_count) else {
            self.record_failure(SchedulerFailureClass::InvalidBatch);
            self.invalid_batches.push(batch);
            return;
        };
        if next_pending > pool.candidate_capacity {
            self.record_failure(SchedulerFailureClass::InvalidBatch);
            self.invalid_batches.push(batch);
            return;
        }

        let batch_id = batch.reservation.sample_batch_id;
        let entry = match reserve_unique_entry(&mut self.batches, batch_id) {
            Some(entry) => entry,
            None => {
                self.first_failure
                    .get_or_insert(SchedulerFailureClass::InvalidBatch);
                self.invalid_batches.push(batch);
                return;
            }
        };
        let attempts = batch.reservation.attempts.clone();
        let (batch_permit, candidate_permits) = batch.admission.into_parts();
        let controller = Arc::new(BatchController {
            window: batch.window,
            reservation: batch.reservation,
            state: Mutex::new(BatchState {
                remaining: candidate_count,
                batch_permit: Some(batch_permit),
                candidate_permits: candidate_permits.into_iter().map(Some).collect(),
            }),
        });
        entry.insert(controller.clone());
        let pool = self
            .pools
            .get_mut(&pool_id)
            .expect("batch consistency proved the configured pool exists");
        pool.pending_candidates = next_pending;
        for (candidate_index, attempt) in attempts.into_iter().enumerate() {
            pool.shadow_ready.push_back(CandidateWork {
                batch: controller.clone(),
                candidate_index,
                attempt,
                pool: pool.config.clone(),
            });
        }
        pool.gauges.observe_shadow_queue(pool.shadow_ready.len());
        self.accepted_batches += 1;
    }

    fn dispatch_ready(&mut self) {
        for pool in self.pools.values_mut() {
            while let Some(work) = pool.shadow_ready.pop_front() {
                if !self.shared.start_gate.is_open() {
                    self.tasks
                        .spawn(terminalize_canceled(self.shared.clone(), work));
                    continue;
                }
                let permit = match pool.shadow_semaphore.clone().try_acquire_owned() {
                    Ok(permit) => permit,
                    Err(_) => {
                        pool.shadow_ready.push_front(work);
                        break;
                    }
                };
                self.tasks.spawn(run_shadow_stage(
                    self.shared.clone(),
                    work,
                    permit,
                    pool.gauges.clone(),
                ));
            }

            while let Some(work) = pool.judge_ready.pop_front() {
                if !self.shared.start_gate.is_open() {
                    self.tasks
                        .spawn(terminalize_canceled_judge(self.shared.clone(), work));
                    continue;
                }
                let permit = match pool.judge_semaphore.clone().try_acquire_owned() {
                    Ok(permit) => permit,
                    Err(_) => {
                        pool.judge_ready.push_front(work);
                        break;
                    }
                };
                self.tasks.spawn(run_judge_stage(
                    self.shared.clone(),
                    work,
                    permit,
                    pool.gauges.clone(),
                ));
            }
        }
    }

    fn handle_task_result(&mut self, result: TaskResult) {
        match result {
            TaskResult::JudgeReady(work) => {
                let work = *work;
                let pool_id = work.work.pool.id.clone();
                let Some(pool) = self.pools.get_mut(&pool_id) else {
                    self.record_failure(SchedulerFailureClass::InvalidBatch);
                    return;
                };
                pool.judge_ready.push_back(work);
                pool.gauges.observe_judge_queue(pool.judge_ready.len());
            }
            TaskResult::TerminalAcknowledged {
                batch_id,
                pool_id,
                batch_closed,
                permits,
            } => {
                if let Some(pool) = self.pools.get_mut(&pool_id) {
                    let Some(pending) = pool.pending_candidates.checked_sub(1) else {
                        self.retained_permits.push(*permits);
                        self.record_failure(SchedulerFailureClass::InvalidBatch);
                        return;
                    };
                    pool.pending_candidates = pending;
                } else {
                    self.retained_permits.push(*permits);
                    self.record_failure(SchedulerFailureClass::InvalidBatch);
                    return;
                }
                if batch_closed {
                    if self.batches.remove(&batch_id).is_none() {
                        self.retained_permits.push(*permits);
                        self.record_failure(SchedulerFailureClass::InvalidBatch);
                        return;
                    }
                } else if !self.batches.contains_key(&batch_id) {
                    self.retained_permits.push(*permits);
                    self.record_failure(SchedulerFailureClass::InvalidBatch);
                    return;
                }
                self.acknowledged_candidates += 1;
                drop(permits);
            }
            TaskResult::Failed { failure } => self.record_failure(failure),
        }
    }

    fn queues_are_empty(&self) -> bool {
        self.pools
            .values()
            .all(|pool| pool.shadow_ready.is_empty() && pool.judge_ready.is_empty())
    }

    fn reap_completed_tasks(&mut self) {
        for result in take_completed(&mut self.tasks) {
            match result {
                Ok(result) => self.handle_task_result(result),
                Err(_) => self.record_failure(SchedulerFailureClass::JoinFailure),
            }
        }
    }

    fn record_failure(&mut self, failure: SchedulerFailureClass) {
        if self.first_failure.is_none() {
            self.first_failure = Some(failure);
            if let Some(notifier) = self.failure_notifier.as_ref() {
                notifier(failure);
            }
        }
    }

    fn observe_actor_state(&mut self) {
        self.max_task_entries = self.max_task_entries.max(self.tasks.len());
        self.max_active_batches = self.max_active_batches.max(self.batches.len());
        let pending = self
            .pools
            .values()
            .try_fold(0usize, |total, pool| {
                total.checked_add(pool.pending_candidates)
            })
            .unwrap_or(usize::MAX);
        self.max_pending_candidates = self.max_pending_candidates.max(pending);
    }
}

impl PoolGauges {
    fn enter_shadow(self: &Arc<Self>) -> ProviderGaugeGuard {
        let current = self.shadow_in_flight.fetch_add(1, Ordering::AcqRel) + 1;
        self.shadow_max_in_flight
            .fetch_max(current, Ordering::AcqRel);
        ProviderGaugeGuard {
            gauges: self.clone(),
            stage: ProviderStage::Shadow,
        }
    }

    fn enter_judge(self: &Arc<Self>) -> ProviderGaugeGuard {
        let current = self.judge_in_flight.fetch_add(1, Ordering::AcqRel) + 1;
        self.judge_max_in_flight
            .fetch_max(current, Ordering::AcqRel);
        ProviderGaugeGuard {
            gauges: self.clone(),
            stage: ProviderStage::Judge,
        }
    }

    fn observe_shadow_queue(&self, queued: usize) {
        self.shadow_max_queued.fetch_max(queued, Ordering::AcqRel);
    }

    fn observe_judge_queue(&self, queued: usize) {
        self.judge_max_queued.fetch_max(queued, Ordering::AcqRel);
    }

    fn snapshot(&self) -> SchedulerPoolGaugeSnapshot {
        SchedulerPoolGaugeSnapshot {
            shadow_max_in_flight: self.shadow_max_in_flight.load(Ordering::Acquire),
            judge_max_in_flight: self.judge_max_in_flight.load(Ordering::Acquire),
            shadow_max_queued: self.shadow_max_queued.load(Ordering::Acquire),
            judge_max_queued: self.judge_max_queued.load(Ordering::Acquire),
        }
    }
}

impl Drop for ProviderGaugeGuard {
    fn drop(&mut self) {
        match self.stage {
            ProviderStage::Shadow => {
                self.gauges.shadow_in_flight.fetch_sub(1, Ordering::AcqRel);
            }
            ProviderStage::Judge => {
                self.gauges.judge_in_flight.fetch_sub(1, Ordering::AcqRel);
            }
        }
    }
}

fn batch_is_consistent(
    batch: &ScheduledTrajectoryBatch,
    pools: &BTreeMap<String, PoolState>,
) -> bool {
    let pool_id = batch.reservation.pool_id.as_str();
    let Some(pool) = pools.get(pool_id) else {
        return false;
    };
    let attempts = &batch.reservation.attempts;
    let candidates = &batch.window.eligible_candidates;
    batch.admission.pool_id() == pool_id
        && batch.admission.candidate_count() == attempts.len()
        && batch.window.pending.pool_id == pool_id
        && batch.window.pending.anchor_id == batch.reservation.anchor_id
        && batch.window.pending.config_generation_id == batch.reservation.config_generation_id
        && batch.window.pending.policy_version_id == batch.reservation.policy_version_id
        && batch.window.pending.learning_generation_id == batch.reservation.learning_generation_id
        && attempts.len() == candidates.len()
        && attempts.len() <= pool.config.max_candidates_per_sample
        && attempts.iter().zip(candidates).all(|(attempt, candidate)| {
            attempt.candidate_id == candidate.config.id
                && attempt.candidate_model == candidate.config.model
                && attempt.candidate_model_revision == candidate.config.model_revision
                && attempt.cost_rank == candidate.config.cost_rank
                && attempt.api_family == pool.config.api_family
        })
}

fn reserve_unique_entry<K: Ord, V>(
    map: &mut BTreeMap<K, V>,
    key: K,
) -> Option<VacantEntry<'_, K, V>> {
    match map.entry(key) {
        std::collections::btree_map::Entry::Vacant(entry) => Some(entry),
        std::collections::btree_map::Entry::Occupied(_) => None,
    }
}

fn take_completed<T: Send + 'static>(
    tasks: &mut JoinSet<T>,
) -> Vec<Result<T, tokio::task::JoinError>> {
    let mut completed = Vec::new();
    while let Some(result) = tasks.try_join_next() {
        completed.push(result);
    }
    completed
}

async fn run_shadow_stage(
    shared: SchedulerShared,
    work: CandidateWork,
    provider_permit: OwnedSemaphorePermit,
    gauges: Arc<PoolGauges>,
) -> TaskResult {
    let _gauge = gauges.enter_shadow();
    let dependency = match CandidateDependencyIdentity::new(
        work.attempt.api_family,
        work.attempt.transport_identity.clone(),
        work.attempt.candidate_model.clone(),
        work.attempt.candidate_model_revision.clone(),
    ) {
        Ok(dependency) => dependency,
        Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
    };
    let operation = match DependencyOperation::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        work.batch.reservation.anchor_id,
        work.attempt.shadow_attempt_id,
        work.pool.judge.base_cooloff_seconds,
        work.pool.judge.max_cooloff_seconds,
        now_unix_ms(),
    ) {
        Ok(operation) => operation,
        Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
    };
    let claim = match claim_candidate(&shared, dependency, operation.clone()).await {
        Ok(claim) => claim,
        Err(failure) => return failed(failure),
    };
    if claim == DependencyClaim::SkippedCooloff {
        drop(provider_permit);
        return persist_terminal_result(&shared, work, TerminalDraft::skipped()).await;
    }

    let started = match ShadowAttemptStarted::new(
        work.attempt.shadow_attempt_id,
        Uuid::now_v7(),
        Uuid::now_v7(),
        now_unix_ms(),
    ) {
        Ok(started) => started,
        Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
    };
    if let Err(failure) = persist_shadow_start(&shared, started).await {
        return failed(failure);
    }

    let candidate = &work.batch.window.eligible_candidates[work.candidate_index];
    let started_at = Instant::now();
    let outcome = execute_shadow_replay(ManagedReplayParams {
        invocation_id: work.attempt.shadow_attempt_id,
        linkage: evaluator_linkage(&work),
        api_family: work.attempt.api_family,
        request: candidate.request.clone(),
        transport: work.batch.window.replay_transport.clone(),
        start_gate: shared.start_gate.clone(),
        active_replays: shared.active_replays.clone(),
        cancellation: shared.cancellation.clone(),
        max_response_bytes: work.pool.lookahead.max_bytes_per_window,
    })
    .await;
    let latency_ms = elapsed_millis(started_at);
    drop(provider_permit);
    drop(_gauge);

    match outcome {
        ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::CanceledShutdown) => {
            persist_terminal_result(&shared, work, TerminalDraft::canceled()).await
        }
        ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::Transport) => {
            finish_candidate_operational(
                &shared,
                work,
                &operation,
                "router.provider.transport",
                latency_ms,
            )
            .await
        }
        ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::MiddlewareInterference) => {
            finish_candidate_operational(
                &shared,
                work,
                &operation,
                "router.provider.middleware_interference",
                latency_ms,
            )
            .await
        }
        ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::EvidenceBound) => {
            finish_candidate_operational(
                &shared,
                work,
                &operation,
                "router.provider.evidence_bound_exceeded",
                latency_ms,
            )
            .await
        }
        ManagedReplayOutcome::OperationalFailure(
            ManagedReplayFailure::CancellationCapacity | ManagedReplayFailure::Runtime,
        ) => failed(SchedulerFailureClass::EvaluatorRuntime),
        ManagedReplayOutcome::Completed(response) => {
            finish_candidate_response(&shared, work, operation, response, latency_ms).await
        }
    }
}

async fn finish_candidate_response(
    shared: &SchedulerShared,
    work: CandidateWork,
    operation: DependencyOperation,
    response: Json,
    latency_ms: u64,
) -> TaskResult {
    let candidate = &work.batch.window.eligible_candidates[work.candidate_index];
    let validation = validate_candidate_response(
        work.attempt.api_family,
        &response,
        &candidate.response_contracts,
        work.pool.lookahead.max_bytes_per_window,
        work.batch.window.is_partial,
    );
    match validation {
        CandidateValidationOutcomeV1::Valid { response } => {
            if let Err(failure) = complete_dependency_success(shared, &operation).await {
                return failed(failure);
            }
            let usage = response.usage.clone();
            let horizon = match JudgeHorizonV1::new(
                work.batch.window.pending.requested_progress,
                work.batch.window.observed_progress,
                work.batch.window.trigger,
                work.batch.window.is_partial,
            ) {
                Ok(horizon) => horizon,
                Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
            };
            let policy = match JudgePolicyIdentityV1::from_config(&work.pool.judge) {
                Ok(policy) => policy,
                Err(_) => return failed(SchedulerFailureClass::InvalidConfiguration),
            };
            let judge_input = match PairwiseJudgeInputV1::new(
                &work.batch.window.pending.request_projection,
                &work.batch.window.pending.normalized_anchor_response,
                &response,
                &work.batch.window.events,
                horizon,
                policy,
            ) {
                Ok(input) => input,
                Err(_) => {
                    return persist_terminal_result(
                        shared,
                        work,
                        TerminalDraft::operational(
                            "router.provider.evidence_bound_exceeded",
                            None,
                            Some(latency_ms),
                            usage,
                        ),
                    )
                    .await;
                }
            };
            TaskResult::JudgeReady(Box::new(JudgeWork {
                work,
                candidate_response: response,
                judge_input,
                latency_ms,
                usage,
            }))
        }
        CandidateValidationOutcomeV1::DeterministicFailure {
            hard_failure,
            evaluation: _,
            response,
        } => {
            if let Err(failure) = complete_dependency_success(shared, &operation).await {
                return failed(failure);
            }
            let evaluation_id = Uuid::now_v7();
            let evaluation = match EvaluationRecord::deterministic(
                evaluation_id,
                work.attempt.shadow_attempt_id,
                work.attempt.evaluator_version.clone(),
                Uuid::now_v7(),
                hard_failure,
                work.batch.window.is_partial,
                &work.pool.judge,
                now_unix_ms(),
            ) {
                Ok(evaluation) => evaluation,
                Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
            };
            if let Err(failure) = persist_evaluation(shared, evaluation).await {
                return failed(failure);
            }
            let response = response.map(|response| *response);
            let usage = response
                .as_ref()
                .and_then(|response| response.usage.clone());
            persist_terminal_result(
                shared,
                work,
                TerminalDraft::deterministic(
                    hard_failure,
                    response,
                    latency_ms,
                    usage,
                    evaluation_id,
                ),
            )
            .await
        }
        CandidateValidationOutcomeV1::OperationalFailure(failure) => {
            let stable_class = candidate_validation_class(failure);
            finish_candidate_operational(shared, work, &operation, stable_class, latency_ms).await
        }
    }
}

async fn finish_candidate_operational(
    shared: &SchedulerShared,
    work: CandidateWork,
    operation: &DependencyOperation,
    stable_class: &'static str,
    latency_ms: u64,
) -> TaskResult {
    if let Err(failure) = complete_dependency_failure(shared, operation, stable_class).await {
        return failed(failure);
    }
    persist_terminal_result(
        shared,
        work,
        TerminalDraft::operational(stable_class, None, Some(latency_ms), None),
    )
    .await
}

async fn terminalize_canceled(shared: SchedulerShared, work: CandidateWork) -> TaskResult {
    persist_terminal_result(&shared, work, TerminalDraft::canceled()).await
}

async fn terminalize_canceled_judge(shared: SchedulerShared, work: JudgeWork) -> TaskResult {
    persist_terminal_result(
        &shared,
        work.work,
        TerminalDraft::canceled_with_response(work.candidate_response, work.latency_ms, work.usage),
    )
    .await
}

fn evaluator_linkage(work: &CandidateWork) -> EvaluatorLinkage {
    EvaluatorLinkage {
        anchor_uuid: work.batch.window.pending.anchor_call_uuid,
        anchor_id: work.batch.reservation.anchor_id,
        pool_id: work.batch.reservation.pool_id.clone(),
        candidate_id: work.attempt.candidate_id.clone(),
        config_generation_id: work.batch.reservation.config_generation_id.clone(),
        learning_generation_id: work.batch.reservation.learning_generation_id,
    }
}

fn candidate_validation_class(failure: CandidateValidationOperationalFailureV1) -> &'static str {
    match failure {
        CandidateValidationOperationalFailureV1::ProjectionBound => {
            "router.provider.evidence_bound_exceeded"
        }
        CandidateValidationOperationalFailureV1::Truncated => "router.provider.truncated_response",
        CandidateValidationOperationalFailureV1::Cancelled => "router.provider.canceled",
        CandidateValidationOperationalFailureV1::ProviderFailure => "router.provider.transport",
        CandidateValidationOperationalFailureV1::AmbiguousTerminal
        | CandidateValidationOperationalFailureV1::UnsafeProjection => {
            "router.provider.ambiguous_decode"
        }
    }
}

async fn run_judge_stage(
    shared: SchedulerShared,
    work: JudgeWork,
    provider_permit: OwnedSemaphorePermit,
    gauges: Arc<PoolGauges>,
) -> TaskResult {
    let gauge = gauges.enter_judge();
    let dependency = match JudgeDependencyIdentity::new(
        work.work.attempt.api_family,
        work.work.attempt.transport_identity.clone(),
        work.work.pool.judge.model.clone(),
        work.work.pool.judge.model_revision.clone(),
        work.work.pool.judge.prompt_version.clone(),
        JUDGE_PROMPT_TEMPLATE_SHA256_V1,
        work.work.pool.judge.rubric_version.clone(),
        JUDGE_RUBRIC_TEMPLATE_SHA256_V1,
        work.work.pool.judge.output_schema_version,
        JUDGE_OUTPUT_SCHEMA_SHA256_V1,
    ) {
        Ok(dependency) => dependency,
        Err(_) => return failed(SchedulerFailureClass::InvalidConfiguration),
    };
    let operation = match DependencyOperation::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        work.work.batch.reservation.anchor_id,
        work.work.attempt.shadow_attempt_id,
        work.work.pool.judge.base_cooloff_seconds,
        work.work.pool.judge.max_cooloff_seconds,
        now_unix_ms(),
    ) {
        Ok(operation) => operation,
        Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
    };
    let claim = match claim_judge(&shared, dependency, operation.clone()).await {
        Ok(claim) => claim,
        Err(failure) => return failed(failure),
    };
    if claim == DependencyClaim::SkippedCooloff {
        drop(provider_permit);
        drop(gauge);
        return persist_terminal_result(
            &shared,
            work.work,
            TerminalDraft::skipped_with_response(
                work.candidate_response,
                work.latency_ms,
                work.usage,
            ),
        )
        .await;
    }

    let registry = match JudgeRegistryV1::load(&work.work.pool.judge) {
        Ok(registry) => registry,
        Err(_) => return failed(SchedulerFailureClass::InvalidConfiguration),
    };
    let Some(judge_outer_response_bytes) =
        judge_outer_response_bound(work.work.pool.judge.max_rationale_bytes)
    else {
        return failed(SchedulerFailureClass::InvalidConfiguration);
    };
    let request = match FamilyAdapter.build_judge_request(
        work.work.attempt.api_family,
        &work.work.pool.judge,
        &registry,
        &work.judge_input,
    ) {
        Ok(request) => request,
        Err(_) => return failed(SchedulerFailureClass::InvalidConfiguration),
    };

    let initial_attempt_id = Uuid::now_v7();
    let initial_start = match JudgeAttemptStart::new(
        initial_attempt_id,
        work.work.attempt.shadow_attempt_id,
        work.work.batch.reservation.learning_generation_id,
        work.work.attempt.evaluator_version.clone(),
        &work.work.pool.judge,
        &work.judge_input,
        0,
        Uuid::now_v7(),
        Uuid::now_v7(),
        now_unix_ms(),
    ) {
        Ok(start) => start,
        Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
    };
    if let Err(failure) = persist_judge_start(&shared, initial_start).await {
        return failed(failure);
    }
    let initial_outcome = execute_judge_replay(ManagedReplayParams {
        invocation_id: initial_attempt_id,
        linkage: evaluator_linkage(&work.work),
        api_family: work.work.attempt.api_family,
        request,
        transport: work.work.batch.window.replay_transport.clone(),
        start_gate: shared.start_gate.clone(),
        active_replays: shared.active_replays.clone(),
        cancellation: shared.cancellation.clone(),
        max_response_bytes: judge_outer_response_bytes,
    })
    .await;

    let initial_text = match initial_outcome {
        ManagedReplayOutcome::Completed(response) => match extract_judge_text(
            work.work.attempt.api_family,
            &response,
            judge_outer_response_bytes,
        ) {
            Ok(text) => text,
            Err(class) => {
                drop(provider_permit);
                drop(gauge);
                return finish_judge_operational(
                    &shared,
                    work,
                    &operation,
                    initial_attempt_id,
                    class,
                )
                .await;
            }
        },
        ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::CanceledShutdown) => {
            let terminal = match JudgeAttemptTerminal::canceled_shutdown(
                initial_attempt_id,
                Uuid::now_v7(),
                Uuid::now_v7(),
                now_unix_ms(),
            ) {
                Ok(terminal) => terminal,
                Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
            };
            if let Err(failure) = persist_judge_terminal(&shared, terminal).await {
                return failed(failure);
            }
            drop(provider_permit);
            drop(gauge);
            return persist_terminal_result(
                &shared,
                work.work,
                TerminalDraft::canceled_with_response(
                    work.candidate_response,
                    work.latency_ms,
                    work.usage,
                ),
            )
            .await;
        }
        ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::Transport) => {
            drop(provider_permit);
            drop(gauge);
            return finish_judge_operational(
                &shared,
                work,
                &operation,
                initial_attempt_id,
                JudgeTransportFailureClass::Transport,
            )
            .await;
        }
        ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::MiddlewareInterference) => {
            drop(provider_permit);
            drop(gauge);
            return finish_judge_operational(
                &shared,
                work,
                &operation,
                initial_attempt_id,
                JudgeTransportFailureClass::MiddlewareInterference,
            )
            .await;
        }
        ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::EvidenceBound) => {
            drop(provider_permit);
            drop(gauge);
            return finish_judge_operational(
                &shared,
                work,
                &operation,
                initial_attempt_id,
                JudgeTransportFailureClass::EvidenceBoundExceeded,
            )
            .await;
        }
        ManagedReplayOutcome::OperationalFailure(
            ManagedReplayFailure::CancellationCapacity | ManagedReplayFailure::Runtime,
        ) => return failed(SchedulerFailureClass::EvaluatorRuntime),
    };

    let initial_validated =
        validate_pairwise_judge_output(&initial_text, work.work.pool.judge.max_rationale_bytes);
    match &initial_validated {
        JudgeAttemptOutcomeV1::Valid(_) => {
            let evaluation_id = Uuid::now_v7();
            let terminal = match JudgeAttemptTerminal::valid(
                initial_attempt_id,
                Uuid::now_v7(),
                Uuid::now_v7(),
                work.work.attempt.shadow_attempt_id,
                work.work.attempt.evaluator_version.clone(),
                &initial_text,
                &work.work.pool.judge,
                evaluation_id,
                Uuid::now_v7(),
                work.work.batch.window.is_partial,
                now_unix_ms(),
            ) {
                Ok(terminal) => terminal,
                Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
            };
            if let Err(failure) = persist_judge_terminal(&shared, terminal).await {
                return failed(failure);
            }
            if let Err(failure) = complete_dependency_success(&shared, &operation).await {
                return failed(failure);
            }
            drop(provider_permit);
            drop(gauge);
            persist_terminal_result(
                &shared,
                work.work,
                TerminalDraft::completed(
                    work.candidate_response,
                    work.latency_ms,
                    work.usage,
                    evaluation_id,
                ),
            )
            .await
        }
        JudgeAttemptOutcomeV1::Operational(_) => {
            drop(provider_permit);
            drop(gauge);
            finish_judge_operational(
                &shared,
                work,
                &operation,
                initial_attempt_id,
                JudgeTransportFailureClass::EvidenceBoundExceeded,
            )
            .await
        }
        JudgeAttemptOutcomeV1::Invalid(_) => {
            let terminal = match JudgeAttemptTerminal::invalid(
                initial_attempt_id,
                Uuid::now_v7(),
                Uuid::now_v7(),
                &initial_text,
                &work.work.pool.judge,
                now_unix_ms(),
            ) {
                Ok(terminal) => terminal,
                Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
            };
            if let Err(failure) = persist_judge_terminal(&shared, terminal).await {
                return failed(failure);
            }
            run_judge_repair(
                &shared,
                work,
                operation,
                registry,
                initial_validated,
                provider_permit,
                gauge,
            )
            .await
        }
    }
}

#[allow(clippy::too_many_arguments)]
async fn run_judge_repair(
    shared: &SchedulerShared,
    work: JudgeWork,
    operation: DependencyOperation,
    registry: JudgeRegistryV1,
    initial_outcome: JudgeAttemptOutcomeV1,
    provider_permit: OwnedSemaphorePermit,
    gauge: ProviderGaugeGuard,
) -> TaskResult {
    let progress = match resolve_judge_attempt_progress(JudgeAttemptV1::initial(initial_outcome)) {
        Ok(progress) => progress,
        Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
    };
    let (repair_evidence, repair_pending) = match progress.into_repair() {
        Ok(repair) => repair,
        Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
    };
    let request = match FamilyAdapter.build_judge_repair_request(
        work.work.attempt.api_family,
        &work.work.pool.judge,
        &registry,
        &work.judge_input,
        &repair_evidence,
    ) {
        Ok(request) => request,
        Err(_) => return failed(SchedulerFailureClass::InvalidConfiguration),
    };
    let repair_attempt_id = Uuid::now_v7();
    let repair_start = match JudgeAttemptStart::new(
        repair_attempt_id,
        work.work.attempt.shadow_attempt_id,
        work.work.batch.reservation.learning_generation_id,
        work.work.attempt.evaluator_version.clone(),
        &work.work.pool.judge,
        &work.judge_input,
        1,
        Uuid::now_v7(),
        Uuid::now_v7(),
        now_unix_ms(),
    ) {
        Ok(start) => start,
        Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
    };
    if let Err(failure) = persist_judge_start(shared, repair_start).await {
        return failed(failure);
    }
    let Some(judge_outer_response_bytes) =
        judge_outer_response_bound(work.work.pool.judge.max_rationale_bytes)
    else {
        return failed(SchedulerFailureClass::InvalidConfiguration);
    };
    let outcome = execute_judge_replay(ManagedReplayParams {
        invocation_id: repair_attempt_id,
        linkage: evaluator_linkage(&work.work),
        api_family: work.work.attempt.api_family,
        request,
        transport: work.work.batch.window.replay_transport.clone(),
        start_gate: shared.start_gate.clone(),
        active_replays: shared.active_replays.clone(),
        cancellation: shared.cancellation.clone(),
        max_response_bytes: judge_outer_response_bytes,
    })
    .await;
    let repair_text = match outcome {
        ManagedReplayOutcome::Completed(response) => match extract_judge_text(
            work.work.attempt.api_family,
            &response,
            judge_outer_response_bytes,
        ) {
            Ok(text) => text,
            Err(class) => {
                drop(provider_permit);
                drop(gauge);
                return finish_judge_operational(
                    shared,
                    work,
                    &operation,
                    repair_attempt_id,
                    class,
                )
                .await;
            }
        },
        ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::CanceledShutdown) => {
            let terminal = match JudgeAttemptTerminal::canceled_shutdown(
                repair_attempt_id,
                Uuid::now_v7(),
                Uuid::now_v7(),
                now_unix_ms(),
            ) {
                Ok(terminal) => terminal,
                Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
            };
            if let Err(failure) = persist_judge_terminal(shared, terminal).await {
                return failed(failure);
            }
            drop(provider_permit);
            drop(gauge);
            return persist_terminal_result(
                shared,
                work.work,
                TerminalDraft::canceled_with_response(
                    work.candidate_response,
                    work.latency_ms,
                    work.usage,
                ),
            )
            .await;
        }
        ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::Transport) => {
            drop(provider_permit);
            drop(gauge);
            return finish_judge_operational(
                shared,
                work,
                &operation,
                repair_attempt_id,
                JudgeTransportFailureClass::Transport,
            )
            .await;
        }
        ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::MiddlewareInterference) => {
            drop(provider_permit);
            drop(gauge);
            return finish_judge_operational(
                shared,
                work,
                &operation,
                repair_attempt_id,
                JudgeTransportFailureClass::MiddlewareInterference,
            )
            .await;
        }
        ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::EvidenceBound) => {
            drop(provider_permit);
            drop(gauge);
            return finish_judge_operational(
                shared,
                work,
                &operation,
                repair_attempt_id,
                JudgeTransportFailureClass::EvidenceBoundExceeded,
            )
            .await;
        }
        ManagedReplayOutcome::OperationalFailure(
            ManagedReplayFailure::CancellationCapacity | ManagedReplayFailure::Runtime,
        ) => return failed(SchedulerFailureClass::EvaluatorRuntime),
    };

    let repair_outcome =
        validate_pairwise_judge_output(&repair_text, work.work.pool.judge.max_rationale_bytes);
    let repair_progress = match repair_pending.resolve(JudgeAttemptV1::repair(repair_outcome)) {
        Ok(progress) => progress,
        Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
    };
    let Some(final_state) = repair_progress.final_state() else {
        return failed(SchedulerFailureClass::InvalidBatch);
    };
    if final_state.result().is_some() {
        let evaluation_id = Uuid::now_v7();
        let terminal = match JudgeAttemptTerminal::valid(
            repair_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            work.work.attempt.shadow_attempt_id,
            work.work.attempt.evaluator_version.clone(),
            &repair_text,
            &work.work.pool.judge,
            evaluation_id,
            Uuid::now_v7(),
            work.work.batch.window.is_partial,
            now_unix_ms(),
        ) {
            Ok(terminal) => terminal,
            Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
        };
        if let Err(failure) = persist_judge_terminal(shared, terminal).await {
            return failed(failure);
        }
        if let Err(failure) = complete_dependency_success(shared, &operation).await {
            return failed(failure);
        }
        drop(provider_permit);
        drop(gauge);
        return persist_terminal_result(
            shared,
            work.work,
            TerminalDraft::completed(
                work.candidate_response,
                work.latency_ms,
                work.usage,
                evaluation_id,
            ),
        )
        .await;
    }

    let terminal = if final_state.operational_failure().is_some() {
        JudgeAttemptTerminal::transport_failure(
            repair_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            JudgeTransportFailureClass::EvidenceBoundExceeded,
            now_unix_ms(),
        )
    } else {
        JudgeAttemptTerminal::invalid(
            repair_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            &repair_text,
            &work.work.pool.judge,
            now_unix_ms(),
        )
    };
    let terminal = match terminal {
        Ok(terminal) => terminal,
        Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
    };
    if let Err(failure) = persist_judge_terminal(shared, terminal).await {
        return failed(failure);
    }
    let failure_class = if final_state.operational_failure().is_some() {
        "router.provider.evidence_bound_exceeded"
    } else {
        "router.judge.output_invalid"
    };
    if let Err(failure) = complete_dependency_failure(shared, &operation, failure_class).await {
        return failed(failure);
    }
    drop(provider_permit);
    drop(gauge);
    persist_terminal_result(
        shared,
        work.work,
        TerminalDraft::operational(
            failure_class,
            Some(work.candidate_response),
            Some(work.latency_ms),
            work.usage,
        ),
    )
    .await
}

async fn finish_judge_operational(
    shared: &SchedulerShared,
    work: JudgeWork,
    operation: &DependencyOperation,
    judge_attempt_id: Uuid,
    class: JudgeTransportFailureClass,
) -> TaskResult {
    let terminal = match JudgeAttemptTerminal::transport_failure(
        judge_attempt_id,
        Uuid::now_v7(),
        Uuid::now_v7(),
        class,
        now_unix_ms(),
    ) {
        Ok(terminal) => terminal,
        Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
    };
    if let Err(failure) = persist_judge_terminal(shared, terminal).await {
        return failed(failure);
    }
    let stable_class = judge_transport_class(class);
    if let Err(failure) = complete_dependency_failure(shared, operation, stable_class).await {
        return failed(failure);
    }
    persist_terminal_result(
        shared,
        work.work,
        TerminalDraft::operational(
            stable_class,
            Some(work.candidate_response),
            Some(work.latency_ms),
            work.usage,
        ),
    )
    .await
}

impl TerminalDraft {
    fn base(terminal_class: ShadowTerminalClass) -> Self {
        Self {
            shadow_result_id: Uuid::now_v7(),
            state_event_id: Uuid::now_v7(),
            conflict_health_event_id: Uuid::now_v7(),
            terminal_class,
            normalized_response: None,
            deterministic_hard_failure: None,
            operational_failure_class: None,
            latency_ms: None,
            usage: None,
            evaluation_id: None,
            created_at_unix_ms: now_unix_ms(),
        }
    }

    fn skipped() -> Self {
        Self::base(ShadowTerminalClass::SkippedCooloff)
    }

    fn skipped_with_response(
        response: RouterResponseProjectionV1,
        latency_ms: u64,
        usage: Option<SanitizedResponseUsageV1>,
    ) -> Self {
        let mut draft = Self::skipped();
        draft.normalized_response = Some(response);
        draft.latency_ms = Some(latency_ms);
        draft.usage = usage;
        draft
    }

    fn canceled() -> Self {
        Self::base(ShadowTerminalClass::CanceledShutdown)
    }

    fn canceled_with_response(
        response: RouterResponseProjectionV1,
        latency_ms: u64,
        usage: Option<SanitizedResponseUsageV1>,
    ) -> Self {
        let mut draft = Self::canceled();
        draft.normalized_response = Some(response);
        draft.latency_ms = Some(latency_ms);
        draft.usage = usage;
        draft
    }

    fn deterministic(
        hard_failure: crate::judge::DeterministicHardFailureV1,
        response: Option<RouterResponseProjectionV1>,
        latency_ms: u64,
        usage: Option<SanitizedResponseUsageV1>,
        evaluation_id: Uuid,
    ) -> Self {
        let mut draft = Self::base(ShadowTerminalClass::DeterministicFailure);
        draft.normalized_response = response;
        draft.deterministic_hard_failure = Some(hard_failure);
        draft.latency_ms = Some(latency_ms);
        draft.usage = usage;
        draft.evaluation_id = Some(evaluation_id);
        draft
    }

    fn completed(
        response: RouterResponseProjectionV1,
        latency_ms: u64,
        usage: Option<SanitizedResponseUsageV1>,
        evaluation_id: Uuid,
    ) -> Self {
        let mut draft = Self::base(ShadowTerminalClass::Completed);
        draft.normalized_response = Some(response);
        draft.latency_ms = Some(latency_ms);
        draft.usage = usage;
        draft.evaluation_id = Some(evaluation_id);
        draft
    }

    fn operational(
        stable_class: &'static str,
        response: Option<RouterResponseProjectionV1>,
        latency_ms: Option<u64>,
        usage: Option<SanitizedResponseUsageV1>,
    ) -> Self {
        let mut draft = Self::base(ShadowTerminalClass::OperationalFailure);
        draft.normalized_response = response;
        draft.operational_failure_class = ShadowOperationalFailureClass::new(stable_class).ok();
        draft.latency_ms = latency_ms;
        draft.usage = usage;
        draft
    }
}

async fn persist_terminal_result(
    shared: &SchedulerShared,
    work: CandidateWork,
    draft: TerminalDraft,
) -> TaskResult {
    let batch_id = work.batch.reservation.sample_batch_id;
    let pool_id = work.batch.reservation.pool_id.clone();
    let mut state = work.batch.state.lock().await;
    if work.candidate_index >= state.candidate_permits.len()
        || state.candidate_permits[work.candidate_index].is_none()
        || state.remaining == 0
    {
        return failed(SchedulerFailureClass::InvalidBatch);
    }
    let attempt_terminal_state = match scheduler_batch_terminal_state(draft.terminal_class) {
        Some(state) => state,
        None => return failed(SchedulerFailureClass::InvalidBatch),
    };
    let batch_closed = state.remaining == 1;
    if batch_closed && state.batch_permit.is_none() {
        return failed(SchedulerFailureClass::InvalidBatch);
    }
    let batch_terminal = if batch_closed {
        match SampleBatchTerminalEvent::new(
            batch_id,
            Uuid::now_v7(),
            attempt_terminal_state,
            None,
            draft.created_at_unix_ms,
        ) {
            Ok(event) => Some(event),
            Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
        }
    } else {
        None
    };
    let vectorization = if work.pool.learning.is_some() {
        match AtomicShadowVectorization::new(
            work.batch.window.pending.routing_context_projection.clone(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
        ) {
            Ok(vectorization) => ShadowVectorizationHandoff::Atomic(vectorization),
            Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
        }
    } else {
        ShadowVectorizationHandoff::Disabled
    };
    let terminal = match ShadowTerminalRecord::new(
        draft.shadow_result_id,
        work.attempt.shadow_attempt_id,
        draft.state_event_id,
        draft.conflict_health_event_id,
        draft.terminal_class,
        None,
        draft.normalized_response,
        draft.deterministic_hard_failure,
        draft.operational_failure_class,
        draft.latency_ms,
        draft.usage,
        draft.evaluation_id,
        ShadowVectorSourceV1::Canonicalizable {
            query_inputs: Box::new(work.attempt.request_projection.clone()),
        },
        batch_terminal,
        draft.created_at_unix_ms,
    ) {
        Ok(terminal) => terminal.with_vectorization(vectorization),
        Err(_) => return failed(SchedulerFailureClass::InvalidBatch),
    };
    if let Err(failure) = persist_shadow_terminal(shared, terminal).await {
        return failed(failure);
    }

    let candidate_permit = state.candidate_permits[work.candidate_index].take();
    state.remaining -= 1;
    let batch_permit = if batch_closed {
        state.batch_permit.take()
    } else {
        None
    };
    drop(state);
    let Some(candidate_permit) = candidate_permit else {
        return failed(SchedulerFailureClass::InvalidBatch);
    };
    TaskResult::TerminalAcknowledged {
        batch_id,
        pool_id,
        batch_closed,
        permits: Box::new(AcknowledgedPermits {
            candidate: candidate_permit,
            batch: batch_permit,
        }),
    }
}

fn scheduler_batch_terminal_state(
    terminal_class: ShadowTerminalClass,
) -> Option<SampleBatchTerminalState> {
    match terminal_class {
        ShadowTerminalClass::Completed
        | ShadowTerminalClass::DeterministicFailure
        | ShadowTerminalClass::OperationalFailure
        | ShadowTerminalClass::SkippedCooloff => Some(SampleBatchTerminalState::Closed),
        ShadowTerminalClass::CanceledShutdown => Some(SampleBatchTerminalState::CanceledShutdown),
        ShadowTerminalClass::OrphanedBeforeSchedule | ShadowTerminalClass::OrphanedInFlight => None,
    }
}

fn failed(failure: SchedulerFailureClass) -> TaskResult {
    TaskResult::Failed { failure }
}

fn now_unix_ms() -> i64 {
    Utc::now().timestamp_millis()
}

fn elapsed_millis(started_at: Instant) -> u64 {
    u64::try_from(started_at.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn judge_transport_class(class: JudgeTransportFailureClass) -> &'static str {
    match class {
        JudgeTransportFailureClass::Transport => "router.provider.transport",
        JudgeTransportFailureClass::Authentication => "router.provider.authentication",
        JudgeTransportFailureClass::RateLimited => "router.provider.rate_limited",
        JudgeTransportFailureClass::Timeout => "router.provider.timeout",
        JudgeTransportFailureClass::ProviderCanceled => "router.provider.canceled",
        JudgeTransportFailureClass::UnreadableResponse => "router.provider.unreadable_response",
        JudgeTransportFailureClass::TruncatedResponse => "router.provider.truncated_response",
        JudgeTransportFailureClass::AmbiguousDecode => "router.provider.ambiguous_decode",
        JudgeTransportFailureClass::EvidenceBoundExceeded => {
            "router.provider.evidence_bound_exceeded"
        }
        JudgeTransportFailureClass::MiddlewareInterference => {
            "router.provider.middleware_interference"
        }
    }
}

fn extract_judge_text(
    family: LlmApiFamily,
    response: &Json,
    max_outer_response_bytes: usize,
) -> Result<String, JudgeTransportFailureClass> {
    extract_judge_assistant_text(family, response, max_outer_response_bytes).map_err(|failure| {
        match failure {
            JudgeResponseExtractionFailureV1::EvidenceBound => {
                JudgeTransportFailureClass::EvidenceBoundExceeded
            }
            JudgeResponseExtractionFailureV1::Truncated => {
                JudgeTransportFailureClass::TruncatedResponse
            }
            JudgeResponseExtractionFailureV1::ProviderCanceled => {
                JudgeTransportFailureClass::ProviderCanceled
            }
            JudgeResponseExtractionFailureV1::ProviderFailure => {
                JudgeTransportFailureClass::Transport
            }
            JudgeResponseExtractionFailureV1::Unreadable => {
                JudgeTransportFailureClass::UnreadableResponse
            }
            JudgeResponseExtractionFailureV1::AmbiguousDecode => {
                JudgeTransportFailureClass::AmbiguousDecode
            }
        }
    })
}

fn judge_outer_response_bound(max_rationale_bytes: usize) -> Option<usize> {
    max_rationale_bytes
        .checked_add(8 * 1024)?
        .checked_mul(6)?
        .checked_add(8 * 1024)
}

async fn claim_candidate(
    shared: &SchedulerShared,
    identity: CandidateDependencyIdentity,
    operation: DependencyOperation,
) -> Result<DependencyClaim, SchedulerFailureClass> {
    loop {
        let result = match shared
            .deadlines
            .run_attempt(|deadline| {
                shared.writer.claim_candidate_dependency_until(
                    identity.clone(),
                    operation.clone(),
                    deadline,
                )
            })
            .await
        {
            DeadlineAttempt::Completed(result) => result,
            DeadlineAttempt::RetryShortened => continue,
        };
        match result {
            Ok(ack) => return dependency_claim_ack(ack),
            Err(error) if shared.deadlines.should_retry(error) => continue,
            Err(error) => return Err(SchedulerFailureClass::Writer(error.class())),
        }
    }
}

async fn claim_judge(
    shared: &SchedulerShared,
    identity: JudgeDependencyIdentity,
    operation: DependencyOperation,
) -> Result<DependencyClaim, SchedulerFailureClass> {
    loop {
        let result = match shared
            .deadlines
            .run_attempt(|deadline| {
                shared.writer.claim_judge_dependency_until(
                    identity.clone(),
                    operation.clone(),
                    deadline,
                )
            })
            .await
        {
            DeadlineAttempt::Completed(result) => result,
            DeadlineAttempt::RetryShortened => continue,
        };
        match result {
            Ok(ack) => return dependency_claim_ack(ack),
            Err(error) if shared.deadlines.should_retry(error) => continue,
            Err(error) => return Err(SchedulerFailureClass::Writer(error.class())),
        }
    }
}

fn dependency_claim_ack(
    ack: DependencyCommandAck,
) -> Result<DependencyClaim, SchedulerFailureClass> {
    match ack {
        DependencyCommandAck::Applied(snapshot)
        | DependencyCommandAck::AlreadyApplied(snapshot) => match snapshot.transition {
            DependencyTransition::Admitted => Ok(DependencyClaim::Admitted),
            DependencyTransition::SkippedCooloff => Ok(DependencyClaim::SkippedCooloff),
            DependencyTransition::Success
            | DependencyTransition::Failure
            | DependencyTransition::OrphanedInFlight => Err(SchedulerFailureClass::DurableConflict),
        },
        DependencyCommandAck::Conflict => Err(SchedulerFailureClass::DurableConflict),
        DependencyCommandAck::OriginatingProcessNotLive => {
            Err(SchedulerFailureClass::OriginatingProcessNotLive)
        }
        DependencyCommandAck::TransactionNotStarted => {
            Err(SchedulerFailureClass::Writer(WriterFailureClass::Aborted))
        }
    }
}

async fn complete_dependency_success(
    shared: &SchedulerShared,
    operation: &DependencyOperation,
) -> Result<(), SchedulerFailureClass> {
    let completion = DependencyCompletion::success(
        operation.dependency_operation_id,
        Uuid::now_v7(),
        now_unix_ms(),
    )
    .map_err(|_| SchedulerFailureClass::InvalidBatch)?;
    complete_dependency(shared, completion, DependencyTransition::Success).await
}

async fn complete_dependency_failure(
    shared: &SchedulerShared,
    operation: &DependencyOperation,
    stable_class: &'static str,
) -> Result<(), SchedulerFailureClass> {
    let failure_class = DependencyFailureClass::new(stable_class)
        .map_err(|_| SchedulerFailureClass::InvalidBatch)?;
    let completion = DependencyCompletion::failure(
        operation.dependency_operation_id,
        Uuid::now_v7(),
        now_unix_ms(),
        failure_class,
    )
    .map_err(|_| SchedulerFailureClass::InvalidBatch)?;
    complete_dependency(shared, completion, DependencyTransition::Failure).await
}

async fn complete_dependency(
    shared: &SchedulerShared,
    completion: DependencyCompletion,
    expected: DependencyTransition,
) -> Result<(), SchedulerFailureClass> {
    loop {
        let result = match shared
            .deadlines
            .run_attempt(|deadline| {
                shared
                    .writer
                    .complete_dependency_until(completion.clone(), deadline)
            })
            .await
        {
            DeadlineAttempt::Completed(result) => result,
            DeadlineAttempt::RetryShortened => continue,
        };
        match result {
            Ok(DependencyCommandAck::Applied(snapshot))
            | Ok(DependencyCommandAck::AlreadyApplied(snapshot))
                if snapshot.transition == expected =>
            {
                return Ok(());
            }
            Ok(DependencyCommandAck::Applied(_))
            | Ok(DependencyCommandAck::AlreadyApplied(_))
            | Ok(DependencyCommandAck::Conflict) => {
                return Err(SchedulerFailureClass::DurableConflict);
            }
            Ok(DependencyCommandAck::OriginatingProcessNotLive) => {
                return Err(SchedulerFailureClass::OriginatingProcessNotLive);
            }
            Ok(DependencyCommandAck::TransactionNotStarted) => {
                return Err(SchedulerFailureClass::Writer(WriterFailureClass::Aborted));
            }
            Err(error) if shared.deadlines.should_retry(error) => continue,
            Err(error) => return Err(SchedulerFailureClass::Writer(error.class())),
        }
    }
}

async fn persist_shadow_start(
    shared: &SchedulerShared,
    command: ShadowAttemptStarted,
) -> Result<(), SchedulerFailureClass> {
    loop {
        let result = match shared
            .deadlines
            .run_attempt(|deadline| shared.writer.start_shadow_attempt_until(command, deadline))
            .await
        {
            DeadlineAttempt::Completed(result) => result,
            DeadlineAttempt::RetryShortened => continue,
        };
        match result {
            Ok(ack) => return shadow_ack(ack),
            Err(error) if shared.deadlines.should_retry(error) => continue,
            Err(error) => return Err(SchedulerFailureClass::Writer(error.class())),
        }
    }
}

async fn persist_shadow_terminal(
    shared: &SchedulerShared,
    command: ShadowTerminalRecord,
) -> Result<(), SchedulerFailureClass> {
    loop {
        let result = match shared
            .deadlines
            .run_attempt(|deadline| {
                shared
                    .writer
                    .record_shadow_terminal_until(command.clone(), deadline)
            })
            .await
        {
            DeadlineAttempt::Completed(result) => result,
            DeadlineAttempt::RetryShortened => continue,
        };
        match result {
            Ok(ack) => return shadow_ack(ack),
            Err(error) if shared.deadlines.should_retry(error) => continue,
            Err(error) => return Err(SchedulerFailureClass::Writer(error.class())),
        }
    }
}

fn shadow_ack(ack: ShadowCommandAck) -> Result<(), SchedulerFailureClass> {
    match ack {
        ShadowCommandAck::Applied | ShadowCommandAck::AlreadyApplied => Ok(()),
        ShadowCommandAck::Conflict => Err(SchedulerFailureClass::DurableConflict),
        ShadowCommandAck::OriginatingProcessNotLive => {
            Err(SchedulerFailureClass::OriginatingProcessNotLive)
        }
        ShadowCommandAck::TransactionNotStarted => {
            Err(SchedulerFailureClass::Writer(WriterFailureClass::Aborted))
        }
    }
}

async fn persist_judge_start(
    shared: &SchedulerShared,
    command: JudgeAttemptStart,
) -> Result<(), SchedulerFailureClass> {
    loop {
        let result = match shared
            .deadlines
            .run_attempt(|deadline| {
                shared
                    .writer
                    .record_judge_attempt_start_until(command.clone(), deadline)
            })
            .await
        {
            DeadlineAttempt::Completed(result) => result,
            DeadlineAttempt::RetryShortened => continue,
        };
        match result {
            Ok(ack) => return judge_ack(ack),
            Err(error) if shared.deadlines.should_retry(error) => continue,
            Err(error) => return Err(SchedulerFailureClass::Writer(error.class())),
        }
    }
}

async fn persist_judge_terminal(
    shared: &SchedulerShared,
    command: JudgeAttemptTerminal,
) -> Result<(), SchedulerFailureClass> {
    loop {
        let result = match shared
            .deadlines
            .run_attempt(|deadline| {
                shared
                    .writer
                    .record_judge_attempt_terminal_until(command.clone(), deadline)
            })
            .await
        {
            DeadlineAttempt::Completed(result) => result,
            DeadlineAttempt::RetryShortened => continue,
        };
        match result {
            Ok(ack) => return judge_ack(ack),
            Err(error) if shared.deadlines.should_retry(error) => continue,
            Err(error) => return Err(SchedulerFailureClass::Writer(error.class())),
        }
    }
}

async fn persist_evaluation(
    shared: &SchedulerShared,
    command: EvaluationRecord,
) -> Result<(), SchedulerFailureClass> {
    loop {
        let result = match shared
            .deadlines
            .run_attempt(|deadline| {
                shared
                    .writer
                    .record_evaluation_until(command.clone(), deadline)
            })
            .await
        {
            DeadlineAttempt::Completed(result) => result,
            DeadlineAttempt::RetryShortened => continue,
        };
        match result {
            Ok(ack) => return judge_ack(ack),
            Err(error) if shared.deadlines.should_retry(error) => continue,
            Err(error) => return Err(SchedulerFailureClass::Writer(error.class())),
        }
    }
}

fn judge_ack(ack: JudgeRecordAck) -> Result<(), SchedulerFailureClass> {
    match ack {
        JudgeRecordAck::Applied | JudgeRecordAck::AlreadyApplied => Ok(()),
        JudgeRecordAck::Conflict => Err(SchedulerFailureClass::DurableConflict),
        JudgeRecordAck::OriginatingProcessNotLive => {
            Err(SchedulerFailureClass::OriginatingProcessNotLive)
        }
        JudgeRecordAck::TransactionNotStarted => {
            Err(SchedulerFailureClass::Writer(WriterFailureClass::Aborted))
        }
    }
}

#[cfg(test)]
mod white_box_tests {
    use serde_json::json;

    use super::*;
    use crate::judge::InvalidJudgeOutputMarkerV1;

    #[test]
    fn duplicate_batch_id_never_replaces_the_original_entry() {
        let batch_id = Uuid::now_v7();
        let mut batches = BTreeMap::new();
        reserve_unique_entry(&mut batches, batch_id)
            .expect("first batch id must be vacant")
            .insert("original");

        assert!(reserve_unique_entry(&mut batches, batch_id).is_none());
        assert_eq!(batches.get(&batch_id), Some(&"original"));
        assert_eq!(batches.len(), 1);
    }

    #[tokio::test]
    async fn sustained_completed_task_churn_is_fully_reaped_each_wave() {
        const WAVES: usize = 128;
        const TASKS_PER_WAVE: usize = 32;
        let mut tasks = JoinSet::new();
        let mut completed = 0usize;

        for wave in 0..WAVES {
            for task in 0..TASKS_PER_WAVE {
                tasks.spawn(async move { wave * TASKS_PER_WAVE + task });
            }
            while !tasks.is_empty() {
                tokio::task::yield_now().await;
                for result in take_completed(&mut tasks) {
                    result.expect("churn task must join cleanly");
                    completed += 1;
                }
            }
            assert_eq!(tasks.len(), 0);
        }

        assert_eq!(completed, WAVES * TASKS_PER_WAVE);
    }

    #[test]
    fn shadow_and_judge_provider_gauges_are_independent() {
        let gauges = Arc::new(PoolGauges::default());
        let shadow_a = gauges.enter_shadow();
        let shadow_b = gauges.enter_shadow();
        let judge = gauges.enter_judge();

        assert_eq!(
            gauges.snapshot(),
            SchedulerPoolGaugeSnapshot {
                shadow_max_in_flight: 2,
                judge_max_in_flight: 1,
                shadow_max_queued: 0,
                judge_max_queued: 0,
            }
        );
        drop(shadow_a);
        drop(shadow_b);
        drop(judge);
        assert_eq!(gauges.shadow_in_flight.load(Ordering::Acquire), 0);
        assert_eq!(gauges.judge_in_flight.load(Ordering::Acquire), 0);
    }

    #[test]
    fn actual_last_candidate_class_selects_the_batch_terminal_state() {
        for class in [
            ShadowTerminalClass::Completed,
            ShadowTerminalClass::DeterministicFailure,
            ShadowTerminalClass::OperationalFailure,
            ShadowTerminalClass::SkippedCooloff,
        ] {
            assert_eq!(
                scheduler_batch_terminal_state(class),
                Some(SampleBatchTerminalState::Closed)
            );
        }
        assert_eq!(
            scheduler_batch_terminal_state(ShadowTerminalClass::CanceledShutdown),
            Some(SampleBatchTerminalState::CanceledShutdown)
        );
        assert_eq!(
            scheduler_batch_terminal_state(ShadowTerminalClass::OrphanedBeforeSchedule),
            None
        );
        assert_eq!(
            scheduler_batch_terminal_state(ShadowTerminalClass::OrphanedInFlight),
            None
        );
    }

    #[test]
    fn judge_outer_response_bound_is_derived_only_from_judge_policy() {
        assert_eq!(
            judge_outer_response_bound(1),
            Some((1 + 8 * 1024) * 6 + 8 * 1024)
        );
        assert_eq!(judge_outer_response_bound(usize::MAX), None);
    }

    #[test]
    fn judge_outer_bound_admits_raw_cap_and_oversized_marker_for_all_families() {
        let max_rationale_bytes = 1;
        let raw_bound = max_rationale_bytes + 8 * 1024;
        let outer_bound = judge_outer_response_bound(max_rationale_bytes).unwrap();
        let exact = "\u{0001}".repeat(raw_bound);
        let oversized = format!("{exact}\u{0001}");

        for family in [
            LlmApiFamily::OpenAIChatCompletions,
            LlmApiFamily::OpenAIResponses,
            LlmApiFamily::AnthropicMessages,
        ] {
            let exact_text =
                extract_judge_text(family, &judge_text_response(family, &exact), outer_bound)
                    .expect("exact raw cap must pass outer response admission");
            assert_eq!(exact_text, exact);
            let JudgeAttemptOutcomeV1::Invalid(exact_invalid) =
                validate_pairwise_judge_output(&exact_text, max_rationale_bytes)
            else {
                panic!("control-bearing exact-cap output must be invalid")
            };
            assert_eq!(
                exact_invalid.output().retained_output(),
                Some(exact.as_str())
            );

            let oversized_text = extract_judge_text(
                family,
                &judge_text_response(family, &oversized),
                outer_bound,
            )
            .expect("one-over raw cap must reach Judge output validation");
            let JudgeAttemptOutcomeV1::Invalid(oversized_invalid) =
                validate_pairwise_judge_output(&oversized_text, max_rationale_bytes)
            else {
                panic!("one-over output must be invalid")
            };
            assert_eq!(
                oversized_invalid.output().marker(),
                Some(InvalidJudgeOutputMarkerV1::Oversized)
            );
        }
    }

    fn judge_text_response(family: LlmApiFamily, text: &str) -> Json {
        match family {
            LlmApiFamily::OpenAIChatCompletions => json!({
                "id": "chatcmpl-bound",
                "object": "chat.completion",
                "created": 1,
                "model": "judge",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
            }),
            LlmApiFamily::OpenAIResponses => json!({
                "id": "resp-bound",
                "object": "response",
                "created_at": 1.0,
                "status": "completed",
                "model": "judge",
                "output": [{
                    "id": "msg-bound",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]
                }],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
            }),
            LlmApiFamily::AnthropicMessages => json!({
                "id": "msg-bound",
                "type": "message",
                "role": "assistant",
                "model": "judge",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 1, "output_tokens": 1}
            }),
        }
    }

    #[test]
    fn normal_retry_authority_outlives_prior_attempt_deadlines() {
        let authority = SchedulerDeadlineAuthority::new();
        let prior_attempt = authority.attempt_deadline_for(Duration::ZERO);

        assert!(Instant::now() >= prior_attempt);
        assert!(authority.should_retry(WriterFailure::new(WriterFailureClass::Deadline)));
        assert!(authority.attempt_deadline() > Instant::now());
        assert!(!authority.should_retry(WriterFailure::new(WriterFailureClass::Exited)));
    }

    #[test]
    fn shutdown_deadline_stops_retry_and_only_moves_earlier() {
        let authority = SchedulerDeadlineAuthority::new();
        let later = Instant::now() + Duration::from_millis(50);
        let earlier = Instant::now();
        assert!(authority.install_shutdown_deadline(later));
        assert!(authority.install_shutdown_deadline(earlier));
        assert!(!authority.install_shutdown_deadline(later));
        assert!(authority.attempt_deadline() <= earlier);

        assert!(!authority.should_retry(WriterFailure::new(WriterFailureClass::Deadline)));
    }

    #[tokio::test]
    async fn shortened_shutdown_deadline_wakes_and_retries_frozen_attempt() {
        let authority = SchedulerDeadlineAuthority::new();
        let task_authority = authority.clone();
        let frozen_command_id = Uuid::now_v7();
        let attempts = Arc::new(StdMutex::new(Vec::new()));
        let task_attempts = attempts.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        let task = tokio::spawn(async move {
            let first_attempts = task_attempts.clone();
            let first = task_authority
                .run_attempt(|deadline| {
                    first_attempts
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push((frozen_command_id, deadline));
                    let _ = started_tx.send(deadline);
                    std::future::pending::<Uuid>()
                })
                .await;
            assert!(matches!(first, DeadlineAttempt::RetryShortened));

            task_authority
                .run_attempt(|deadline| async move {
                    task_attempts
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner())
                        .push((frozen_command_id, deadline));
                    frozen_command_id
                })
                .await
        });

        let old_attempt_deadline = started_rx.await.unwrap();
        let shortened = Instant::now();
        assert!(old_attempt_deadline > shortened);
        assert!(authority.install_shutdown_deadline(shortened));
        let completed = tokio::time::timeout(Duration::from_secs(1), task)
            .await
            .expect("deadline installation must wake the old writer attempt")
            .unwrap();
        assert!(matches!(
            completed,
            DeadlineAttempt::Completed(id) if id == frozen_command_id
        ));
        let attempts = attempts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0].0, frozen_command_id);
        assert_eq!(attempts[1].0, frozen_command_id);
        assert!(attempts[1].1 <= shortened);
    }
}

#[cfg(test)]
#[path = "scheduler_tests.rs"]
mod tests;
