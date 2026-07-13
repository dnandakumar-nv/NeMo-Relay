// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Memory-only scheduler capacity reservations.

#![allow(dead_code)] // The Task 7 SQLite sink and delivery adapter consume this primitive.

use std::collections::{BTreeMap, btree_map::Entry};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError, watch};

use crate::config::{PoolConfig, RouterConfig, SCHEDULER_MAX_SLOTS, pool_scheduler_slots};

/// Invalid validated configuration supplied to the scheduler reservation pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SchedulerAdmissionBuildError {
    DuplicatePool,
    InvalidCapacity,
    CapacityOverflow,
}

/// Exhaustive nonblocking admission outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SchedulerAdmissionError {
    InvalidPool,
    InvalidCandidateCount,
    NoPermits,
    Closed,
}

/// Lossless pool-pressure state. A generation delta reveals coalesced releases.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct SchedulerPressureState {
    pub(crate) pressure_closed: bool,
    pub(crate) release_generation: u64,
}

/// Shared exact-capacity admission pools for every configured routing pool.
#[derive(Clone)]
pub(crate) struct SchedulerAdmissionPools {
    pools: Arc<BTreeMap<String, Arc<PoolAdmissionPool>>>,
    total_batch_capacity: usize,
    total_candidate_capacity: usize,
}

struct PoolAdmissionPool {
    batch: Arc<Semaphore>,
    candidates: Arc<Semaphore>,
    max_candidates_per_admission: usize,
    closed: AtomicBool,
    pressure: Mutex<PoolPressureState>,
    pressure_tx: watch::Sender<SchedulerPressureState>,
}

#[derive(Default)]
struct PoolPressureState {
    published: SchedulerPressureState,
    batch_blocked: bool,
    candidate_required: usize,
}

/// One memory-only batch reservation. It cannot be cloned or serialized.
pub(crate) struct SchedulerAdmission {
    pool_id: String,
    candidate_count: usize,
    batch_permit: Option<OwnedSemaphorePermit>,
    candidate_permits: Option<OwnedSemaphorePermit>,
    pool: Arc<PoolAdmissionPool>,
}

/// Batch capacity retained until the durable batch-terminal acknowledgement.
pub(crate) struct SchedulerBatchPermit {
    permit: Option<OwnedSemaphorePermit>,
    pool: Arc<PoolAdmissionPool>,
}

/// One candidate slot transferred between bounded scheduler stages.
pub(crate) struct SchedulerCandidatePermit {
    permit: Option<OwnedSemaphorePermit>,
    pool: Arc<PoolAdmissionPool>,
}

impl SchedulerAdmissionPools {
    /// Build exact per-pool capacities from a validated Router configuration.
    pub(crate) fn from_config(config: &RouterConfig) -> Result<Self, SchedulerAdmissionBuildError> {
        Self::from_pools(&config.pools)
    }

    fn from_pools(pools: &[PoolConfig]) -> Result<Self, SchedulerAdmissionBuildError> {
        let mut by_id = BTreeMap::new();
        let mut total_batch_capacity = 0usize;
        let mut total_candidate_capacity = 0usize;

        for pool in pools {
            let slots =
                pool_scheduler_slots(pool).ok_or(SchedulerAdmissionBuildError::CapacityOverflow)?;
            if slots.batch == 0
                || slots.candidates == 0
                || pool.max_candidates_per_sample == 0
                || slots.batch > SCHEDULER_MAX_SLOTS
                || slots.candidates > SCHEDULER_MAX_SLOTS
                || slots.batch > Semaphore::MAX_PERMITS
                || slots.candidates > Semaphore::MAX_PERMITS
            {
                return Err(SchedulerAdmissionBuildError::InvalidCapacity);
            }
            total_batch_capacity = total_batch_capacity
                .checked_add(slots.batch)
                .ok_or(SchedulerAdmissionBuildError::CapacityOverflow)?;
            total_candidate_capacity = total_candidate_capacity
                .checked_add(slots.candidates)
                .ok_or(SchedulerAdmissionBuildError::CapacityOverflow)?;
            if total_batch_capacity > SCHEDULER_MAX_SLOTS
                || total_candidate_capacity > SCHEDULER_MAX_SLOTS
            {
                return Err(SchedulerAdmissionBuildError::InvalidCapacity);
            }

            let (pressure_tx, _) = watch::channel(SchedulerPressureState::default());
            let admission_pool = Arc::new(PoolAdmissionPool {
                batch: Arc::new(Semaphore::new(slots.batch)),
                candidates: Arc::new(Semaphore::new(slots.candidates)),
                max_candidates_per_admission: pool.max_candidates_per_sample,
                closed: AtomicBool::new(false),
                pressure: Mutex::new(PoolPressureState::default()),
                pressure_tx,
            });
            match by_id.entry(pool.id.clone()) {
                Entry::Vacant(entry) => {
                    entry.insert(admission_pool);
                }
                Entry::Occupied(_) => {
                    return Err(SchedulerAdmissionBuildError::DuplicatePool);
                }
            }
        }

        Ok(Self {
            pools: Arc::new(by_id),
            total_batch_capacity,
            total_candidate_capacity,
        })
    }

    /// Acquire one batch permit and exactly `candidate_count` candidate permits.
    pub(crate) fn try_acquire(
        &self,
        pool_id: &str,
        candidate_count: usize,
    ) -> Result<SchedulerAdmission, SchedulerAdmissionError> {
        let pool = self
            .pools
            .get(pool_id)
            .cloned()
            .ok_or(SchedulerAdmissionError::InvalidPool)?;
        if candidate_count == 0 || candidate_count > pool.max_candidates_per_admission {
            return Err(SchedulerAdmissionError::InvalidCandidateCount);
        }
        let candidate_count_u32 = u32::try_from(candidate_count)
            .map_err(|_| SchedulerAdmissionError::InvalidCandidateCount)?;
        if pool.closed.load(Ordering::Acquire) {
            return Err(SchedulerAdmissionError::Closed);
        }

        let mut pressure = lock_unpoisoned(&pool.pressure);
        if pool.closed.load(Ordering::Acquire) {
            return Err(SchedulerAdmissionError::Closed);
        }
        let batch_permit = match pool.batch.clone().try_acquire_owned() {
            Ok(permit) => permit,
            Err(TryAcquireError::Closed) => return Err(SchedulerAdmissionError::Closed),
            Err(TryAcquireError::NoPermits) => {
                return Err(pool.mark_batch_pressure(&mut pressure, candidate_count));
            }
        };
        let candidate_permits = match pool
            .candidates
            .clone()
            .try_acquire_many_owned(candidate_count_u32)
        {
            Ok(permits) => permits,
            Err(TryAcquireError::Closed) => {
                drop(batch_permit);
                return Err(SchedulerAdmissionError::Closed);
            }
            Err(TryAcquireError::NoPermits) => {
                drop(batch_permit);
                return Err(pool.mark_candidate_pressure(&mut pressure, candidate_count));
            }
        };
        if pool.closed.load(Ordering::Acquire) {
            drop(candidate_permits);
            drop(batch_permit);
            return Err(SchedulerAdmissionError::Closed);
        }
        drop(pressure);

        Ok(SchedulerAdmission {
            pool_id: pool_id.to_string(),
            candidate_count,
            batch_permit: Some(batch_permit),
            candidate_permits: Some(candidate_permits),
            pool,
        })
    }

    pub(crate) fn pressure_state(
        &self,
        pool_id: &str,
    ) -> Result<SchedulerPressureState, SchedulerAdmissionError> {
        self.pools
            .get(pool_id)
            .map(|pool| pool.pressure_state())
            .ok_or(SchedulerAdmissionError::InvalidPool)
    }

    pub(crate) fn subscribe_pressure(
        &self,
        pool_id: &str,
    ) -> Result<watch::Receiver<SchedulerPressureState>, SchedulerAdmissionError> {
        self.pools
            .get(pool_id)
            .map(|pool| pool.pressure_tx.subscribe())
            .ok_or(SchedulerAdmissionError::InvalidPool)
    }

    /// Permanently close one pool without classifying shutdown as pressure.
    pub(crate) fn close_pool(&self, pool_id: &str) -> Result<(), SchedulerAdmissionError> {
        let pool = self
            .pools
            .get(pool_id)
            .ok_or(SchedulerAdmissionError::InvalidPool)?;
        pool.close();
        Ok(())
    }

    pub(crate) fn close(&self) {
        for pool in self.pools.values() {
            pool.close();
        }
    }

    pub(crate) const fn total_batch_capacity(&self) -> usize {
        self.total_batch_capacity
    }

    pub(crate) const fn total_candidate_capacity(&self) -> usize {
        self.total_candidate_capacity
    }

    #[cfg(test)]
    fn available_permits(&self, pool_id: &str) -> (usize, usize) {
        let pool = self.pools.get(pool_id).expect("test pool must exist");
        (
            pool.batch.available_permits(),
            pool.candidates.available_permits(),
        )
    }
}

impl fmt::Debug for SchedulerAdmissionPools {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SchedulerAdmissionPools")
            .field("pool_count", &self.pools.len())
            .field("total_batch_capacity", &self.total_batch_capacity)
            .field("total_candidate_capacity", &self.total_candidate_capacity)
            .finish()
    }
}

impl SchedulerAdmission {
    pub(crate) fn pool_id(&self) -> &str {
        &self.pool_id
    }

    pub(crate) const fn candidate_count(&self) -> usize {
        self.candidate_count
    }

    /// Consume this aggregate reservation into independently owned exact permits.
    pub(crate) fn into_parts(mut self) -> (SchedulerBatchPermit, Vec<SchedulerCandidatePermit>) {
        let mut candidate_permits = Vec::with_capacity(self.candidate_count);
        let aggregate = self
            .candidate_permits
            .as_ref()
            .expect("scheduler admission must own candidate permits");
        assert_eq!(
            aggregate.num_permits(),
            self.candidate_count,
            "scheduler admission candidate permit count changed"
        );
        let mut aggregate = self
            .candidate_permits
            .take()
            .expect("scheduler admission candidate permits disappeared");

        for _ in 1..self.candidate_count {
            let permit = aggregate
                .split(1)
                .expect("validated scheduler admission must split exactly");
            candidate_permits.push(SchedulerCandidatePermit {
                permit: Some(permit),
                pool: self.pool.clone(),
            });
        }
        candidate_permits.push(SchedulerCandidatePermit {
            permit: Some(aggregate),
            pool: self.pool.clone(),
        });

        let batch = SchedulerBatchPermit {
            permit: Some(
                self.batch_permit
                    .take()
                    .expect("scheduler admission must own a batch permit"),
            ),
            pool: self.pool.clone(),
        };
        (batch, candidate_permits)
    }
}

impl fmt::Debug for SchedulerAdmission {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SchedulerAdmission")
            .field("pool_id", &self.pool_id)
            .field("candidate_count", &self.candidate_count)
            .finish_non_exhaustive()
    }
}

impl Drop for SchedulerAdmission {
    fn drop(&mut self) {
        self.pool
            .record_admission_release(self.candidate_permits.take(), self.batch_permit.take());
    }
}

impl Drop for SchedulerBatchPermit {
    fn drop(&mut self) {
        if let Some(permit) = self.permit.take() {
            self.pool.record_batch_release(permit);
        }
    }
}

impl Drop for SchedulerCandidatePermit {
    fn drop(&mut self) {
        if let Some(permit) = self.permit.take() {
            self.pool.record_candidate_release(permit);
        }
    }
}

impl PoolAdmissionPool {
    fn pressure_state(&self) -> SchedulerPressureState {
        lock_unpoisoned(&self.pressure).published
    }

    fn mark_batch_pressure(
        &self,
        state: &mut PoolPressureState,
        candidate_required: usize,
    ) -> SchedulerAdmissionError {
        if self.closed.load(Ordering::Acquire) {
            return SchedulerAdmissionError::Closed;
        }
        state.batch_blocked = true;
        state.candidate_required = state.candidate_required.max(candidate_required);
        self.publish_pressure(state);
        SchedulerAdmissionError::NoPermits
    }

    fn mark_candidate_pressure(
        &self,
        state: &mut PoolPressureState,
        candidate_required: usize,
    ) -> SchedulerAdmissionError {
        if self.closed.load(Ordering::Acquire) {
            return SchedulerAdmissionError::Closed;
        }
        state.candidate_required = state.candidate_required.max(candidate_required);
        self.publish_pressure(state);
        SchedulerAdmissionError::NoPermits
    }

    fn record_admission_release(
        &self,
        candidate_permits: Option<OwnedSemaphorePermit>,
        batch_permit: Option<OwnedSemaphorePermit>,
    ) {
        if candidate_permits.is_none() && batch_permit.is_none() {
            return;
        }
        let mut state = lock_unpoisoned(&self.pressure);
        drop(candidate_permits);
        drop(batch_permit);
        self.record_release(&mut state);
    }

    fn record_batch_release(&self, permit: OwnedSemaphorePermit) {
        let mut state = lock_unpoisoned(&self.pressure);
        drop(permit);
        self.record_release(&mut state);
    }

    fn record_candidate_release(&self, permit: OwnedSemaphorePermit) {
        let mut state = lock_unpoisoned(&self.pressure);
        drop(permit);
        self.record_release(&mut state);
    }

    fn record_release(&self, state: &mut PoolPressureState) {
        state.published.release_generation = state
            .published
            .release_generation
            .checked_add(1)
            .expect("scheduler release generation exhausted");
        if !self.closed.load(Ordering::Acquire) {
            if state.batch_blocked && self.batch.available_permits() > 0 {
                state.batch_blocked = false;
            }
            if state.candidate_required > 0
                && self.candidates.available_permits() >= state.candidate_required
            {
                state.candidate_required = 0;
            }
            Self::refresh_published_pressure(state);
        }
        self.pressure_tx.send_replace(state.published);
    }

    fn publish_pressure(&self, state: &mut PoolPressureState) {
        if Self::refresh_published_pressure(state) {
            self.pressure_tx.send_replace(state.published);
        }
    }

    fn refresh_published_pressure(state: &mut PoolPressureState) -> bool {
        let pressure_closed = state.batch_blocked || state.candidate_required > 0;
        let changed = state.published.pressure_closed != pressure_closed;
        state.published.pressure_closed = pressure_closed;
        changed
    }

    fn close(&self) {
        let _pressure = lock_unpoisoned(&self.pressure);
        if !self.closed.swap(true, Ordering::AcqRel) {
            self.batch.close();
            self.candidates.close();
        }
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc as std_mpsc;
    use std::thread;
    use std::time::Duration;

    use serde_json::json;
    use tokio::sync::Barrier;

    use super::*;
    use crate::config::{
        CandidateCapabilities, CandidateConfig, CanonicalizerConfig, ConcurrencyConfig,
        JudgeConfig, LookaheadConfig, PoolSelectorConfig,
    };

    fn pool(id: &str, max_pending: usize, max_candidates_per_sample: usize) -> PoolConfig {
        let candidates = (0..max_candidates_per_sample)
            .map(|index| CandidateConfig {
                id: format!("candidate-{index}"),
                model: format!("candidate-model-{index}"),
                model_revision: "candidate-r1".to_string(),
                cost_rank: u32::try_from(index).unwrap(),
                max_context_tokens: None,
                capabilities: CandidateCapabilities::default(),
                unknown_fields: BTreeMap::new(),
            })
            .collect();
        PoolConfig {
            id: id.to_string(),
            api_family: nemo_relay::api::llm::LlmApiFamily::OpenAIChatCompletions,
            anchor_models: vec!["anchor-model".to_string()],
            anchor_revision: "anchor-r1".to_string(),
            sampling_probability: 1.0,
            max_candidates_per_sample,
            selector: PoolSelectorConfig::default(),
            lookahead: LookaheadConfig::default(),
            concurrency: ConcurrencyConfig {
                shadow: 1,
                judge: 1,
                max_pending,
                unknown_fields: BTreeMap::new(),
            },
            candidates,
            canonicalizer: CanonicalizerConfig::default(),
            judge: JudgeConfig {
                version: 1,
                model: "judge-model".to_string(),
                model_revision: "judge-r1".to_string(),
                prompt_version: "pairwise-equivalence-v1".to_string(),
                rubric_version: "response-trajectory-equivalence-v1".to_string(),
                output_schema_version: 1,
                temperature: None,
                response_weight: 0.5,
                trajectory_weight: 0.5,
                response_floor: 0.8,
                trajectory_floor: 0.8,
                judge_confidence_floor: 0.7,
                pass_threshold: 0.85,
                max_rationale_bytes: 1024,
                base_cooloff_seconds: 10,
                max_cooloff_seconds: 100,
                unknown_fields: BTreeMap::new(),
            },
            learning: None,
            outcome: BTreeMap::new(),
            unknown_fields: BTreeMap::new(),
        }
    }

    fn pools(configured: Vec<PoolConfig>) -> SchedulerAdmissionPools {
        SchedulerAdmissionPools::from_pools(&configured).unwrap()
    }

    fn assert_admission_error(
        result: Result<SchedulerAdmission, SchedulerAdmissionError>,
        expected: SchedulerAdmissionError,
    ) {
        match result {
            Ok(admission) => panic!("unexpected admission: {admission:?}"),
            Err(actual) => assert_eq!(actual, expected),
        }
    }

    #[test]
    fn capacities_are_exact_and_pool_isolated() {
        let pools = pools(vec![pool("first", 2, 3), pool("second", 1, 2)]);
        assert_eq!(pools.total_batch_capacity(), 3);
        assert_eq!(pools.total_candidate_capacity(), 8);
        assert_eq!(pools.available_permits("first"), (2, 6));
        assert_eq!(pools.available_permits("second"), (1, 2));

        let first = pools.try_acquire("first", 3).unwrap();
        let second = pools.try_acquire("first", 3).unwrap();
        assert_eq!(first.pool_id(), "first");
        assert_eq!(first.candidate_count(), 3);
        assert_eq!(pools.available_permits("first"), (0, 0));
        assert_admission_error(
            pools.try_acquire("first", 1),
            SchedulerAdmissionError::NoPermits,
        );
        assert_eq!(pools.available_permits("second"), (1, 2));
        assert!(!pools.pressure_state("second").unwrap().pressure_closed);

        drop(second);
        drop(first);
        assert_eq!(pools.available_permits("first"), (2, 6));
    }

    #[test]
    fn admission_splits_into_one_batch_and_exact_candidate_permits() {
        let pools = pools(vec![pool("pool", 1, 3)]);
        let admission = pools.try_acquire("pool", 3).unwrap();

        let (batch, candidates) = admission.into_parts();

        assert_eq!(batch.permit.as_ref().unwrap().num_permits(), 1);
        assert_eq!(candidates.len(), 3);
        assert!(
            candidates
                .iter()
                .all(|candidate| candidate.permit.as_ref().unwrap().num_permits() == 1)
        );
        assert_eq!(pools.available_permits("pool"), (0, 0));
        assert_eq!(
            pools.pressure_state("pool").unwrap(),
            SchedulerPressureState::default(),
            "consuming the aggregate must not publish a release"
        );

        drop(candidates);
        assert_eq!(pools.available_permits("pool"), (0, 3));
        assert_eq!(pools.pressure_state("pool").unwrap().release_generation, 3);
        drop(batch);
        assert_eq!(pools.available_permits("pool"), (1, 3));
        assert_eq!(pools.pressure_state("pool").unwrap().release_generation, 4);
    }

    #[test]
    fn batch_pressure_survives_independent_candidate_releases() {
        let pools = pools(vec![pool("pool", 1, 2)]);
        let admission = pools.try_acquire("pool", 2).unwrap();
        assert_admission_error(
            pools.try_acquire("pool", 1),
            SchedulerAdmissionError::NoPermits,
        );
        let (batch, mut candidates) = admission.into_parts();

        let first = candidates.pop().unwrap();
        drop(first);
        assert_eq!(pools.available_permits("pool"), (0, 1));
        assert_eq!(
            pools.pressure_state("pool").unwrap(),
            SchedulerPressureState {
                pressure_closed: true,
                release_generation: 1,
            }
        );

        drop(candidates);
        assert_eq!(pools.available_permits("pool"), (0, 2));
        assert_eq!(
            pools.pressure_state("pool").unwrap(),
            SchedulerPressureState {
                pressure_closed: true,
                release_generation: 2,
            }
        );
        drop(batch);
        assert_eq!(pools.available_permits("pool"), (1, 2));
        assert_eq!(
            pools.pressure_state("pool").unwrap(),
            SchedulerPressureState {
                pressure_closed: false,
                release_generation: 3,
            }
        );
    }

    #[test]
    fn batch_release_waits_for_the_refused_candidate_requirement() {
        let pools = pools(vec![pool("pool", 2, 2)]);
        let first = pools.try_acquire("pool", 2).unwrap();
        let second = pools.try_acquire("pool", 1).unwrap();
        assert_admission_error(
            pools.try_acquire("pool", 2),
            SchedulerAdmissionError::NoPermits,
        );
        let (first_batch, first_candidates) = first.into_parts();
        let (second_batch, second_candidates) = second.into_parts();

        drop(second_batch);
        assert_eq!(pools.available_permits("pool"), (1, 1));
        assert_eq!(
            pools.pressure_state("pool").unwrap(),
            SchedulerPressureState {
                pressure_closed: true,
                release_generation: 1,
            }
        );

        drop(second_candidates);
        assert_eq!(pools.available_permits("pool"), (1, 2));
        assert_eq!(
            pools.pressure_state("pool").unwrap(),
            SchedulerPressureState {
                pressure_closed: false,
                release_generation: 2,
            }
        );
        drop(first_candidates);
        drop(first_batch);
    }

    #[test]
    fn candidate_pressure_waits_for_the_largest_refused_slot_count() {
        let pools = pools(vec![pool("pool", 1, 3)]);
        let admission = pools.try_acquire("pool", 3).unwrap();
        let (batch, mut candidates) = admission.into_parts();
        drop(batch);

        assert_admission_error(
            pools.try_acquire("pool", 2),
            SchedulerAdmissionError::NoPermits,
        );
        assert_admission_error(
            pools.try_acquire("pool", 3),
            SchedulerAdmissionError::NoPermits,
        );

        drop(candidates.pop());
        drop(candidates.pop());
        assert_eq!(pools.available_permits("pool"), (1, 2));
        assert_eq!(
            pools.pressure_state("pool").unwrap(),
            SchedulerPressureState {
                pressure_closed: true,
                release_generation: 3,
            }
        );

        drop(candidates);
        assert_eq!(pools.available_permits("pool"), (1, 3));
        assert_eq!(
            pools.pressure_state("pool").unwrap(),
            SchedulerPressureState {
                pressure_closed: false,
                release_generation: 4,
            }
        );
    }

    #[test]
    fn unsplit_admission_drop_releases_the_bundle_once() {
        let pools = pools(vec![pool("pool", 1, 2)]);
        let admission = pools.try_acquire("pool", 2).unwrap();
        assert_admission_error(
            pools.try_acquire("pool", 1),
            SchedulerAdmissionError::NoPermits,
        );

        drop(admission);

        assert_eq!(pools.available_permits("pool"), (1, 2));
        assert_eq!(
            pools.pressure_state("pool").unwrap(),
            SchedulerPressureState {
                pressure_closed: false,
                release_generation: 1,
            }
        );
    }

    #[test]
    fn split_permit_release_and_pool_close_share_one_linearization() {
        let pools = Arc::new(pools(vec![pool("pool", 1, 2)]));
        let admission = pools.try_acquire("pool", 2).unwrap();
        assert_admission_error(
            pools.try_acquire("pool", 1),
            SchedulerAdmissionError::NoPermits,
        );
        let (batch, mut candidates) = admission.into_parts();
        let candidate = candidates.pop().unwrap();
        let pool = pools.pools.get("pool").unwrap().clone();
        let pressure = lock_unpoisoned(&pool.pressure);
        let (drop_started_tx, drop_started_rx) = std_mpsc::channel();
        let (drop_finished_tx, drop_finished_rx) = std_mpsc::channel();
        let dropper = thread::spawn(move || {
            drop_started_tx.send(()).unwrap();
            drop(candidate);
            drop_finished_tx.send(()).unwrap();
        });
        let closing = pools.clone();
        let (close_started_tx, close_started_rx) = std_mpsc::channel();
        let (close_finished_tx, close_finished_rx) = std_mpsc::channel();
        let closer = thread::spawn(move || {
            close_started_tx.send(()).unwrap();
            closing.close_pool("pool").unwrap();
            close_finished_tx.send(()).unwrap();
        });

        drop_started_rx.recv().unwrap();
        close_started_rx.recv().unwrap();
        assert!(
            drop_finished_rx
                .recv_timeout(Duration::from_millis(20))
                .is_err()
        );
        assert!(
            close_finished_rx
                .recv_timeout(Duration::from_millis(20))
                .is_err()
        );
        assert_eq!(pool.candidates.available_permits(), 0);

        drop(pressure);
        drop_finished_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        close_finished_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap();
        dropper.join().unwrap();
        closer.join().unwrap();

        assert!(pool.batch.is_closed());
        assert!(pool.candidates.is_closed());
        assert_eq!(pools.pressure_state("pool").unwrap().release_generation, 1);
        assert_admission_error(
            pools.try_acquire("pool", 1),
            SchedulerAdmissionError::Closed,
        );

        drop(candidates);
        drop(batch);
        assert_eq!(pools.pressure_state("pool").unwrap().release_generation, 3);
        assert_admission_error(
            pools.try_acquire("pool", 1),
            SchedulerAdmissionError::Closed,
        );
    }

    #[test]
    fn partial_candidate_failure_rolls_back_batch_without_fake_release() {
        let pools = pools(vec![pool("pool", 2, 2)]);
        let pool = pools.pools.get("pool").unwrap();
        let held_candidates = pool.candidates.clone().try_acquire_many_owned(4).unwrap();

        assert_admission_error(
            pools.try_acquire("pool", 2),
            SchedulerAdmissionError::NoPermits,
        );
        assert_eq!(pools.available_permits("pool"), (2, 0));
        assert_eq!(
            pools.pressure_state("pool").unwrap(),
            SchedulerPressureState {
                pressure_closed: true,
                release_generation: 0,
            }
        );

        drop(held_candidates);
        assert_eq!(pools.available_permits("pool"), (2, 4));
        assert!(pools.pressure_state("pool").unwrap().pressure_closed);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_acquisition_never_oversubscribes() {
        const CONTENDERS: usize = 24;
        let pools = Arc::new(pools(vec![pool("pool", 4, 2)]));
        let barrier = Arc::new(Barrier::new(CONTENDERS + 1));
        let release = Arc::new(Semaphore::new(0));
        let attempted = Arc::new(AtomicUsize::new(0));
        let acquired = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();

        for _ in 0..CONTENDERS {
            let pools = pools.clone();
            let barrier = barrier.clone();
            let release = release.clone();
            let attempted = attempted.clone();
            let acquired = acquired.clone();
            tasks.push(tokio::spawn(async move {
                barrier.wait().await;
                let admission = pools.try_acquire("pool", 2).ok();
                attempted.fetch_add(1, Ordering::AcqRel);
                if admission.is_some() {
                    acquired.fetch_add(1, Ordering::AcqRel);
                    let _release = release.acquire().await.unwrap();
                }
                admission
            }));
        }

        barrier.wait().await;
        while attempted.load(Ordering::Acquire) != CONTENDERS {
            tokio::task::yield_now().await;
        }
        assert_eq!(acquired.load(Ordering::Acquire), 4);
        assert_eq!(pools.available_permits("pool"), (0, 0));
        release.add_permits(acquired.load(Ordering::Acquire));
        for task in tasks {
            drop(task.await.unwrap());
        }
        assert_eq!(pools.available_permits("pool"), (4, 8));
    }

    #[tokio::test]
    async fn pressure_release_generation_is_lossless_and_reopens_on_real_drop() {
        let pools = pools(vec![pool("pool", 2, 1)]);
        let mut pressure = pools.subscribe_pressure("pool").unwrap();
        let first = pools.try_acquire("pool", 1).unwrap();
        let second = pools.try_acquire("pool", 1).unwrap();

        assert_admission_error(
            pools.try_acquire("pool", 1),
            SchedulerAdmissionError::NoPermits,
        );
        pressure.changed().await.unwrap();
        assert_eq!(
            *pressure.borrow_and_update(),
            SchedulerPressureState {
                pressure_closed: true,
                release_generation: 0,
            }
        );

        drop(first);
        drop(second);
        pressure.changed().await.unwrap();
        assert_eq!(
            *pressure.borrow_and_update(),
            SchedulerPressureState {
                pressure_closed: false,
                release_generation: 2,
            }
        );
        let recovered = pools.try_acquire("pool", 1).unwrap();
        drop(recovered);
    }

    #[test]
    fn permit_release_and_pressure_generation_share_one_linearization() {
        let pools = Arc::new(pools(vec![pool("pool", 1, 1)]));
        let admission = pools.try_acquire("pool", 1).unwrap();
        let pool = pools.pools.get("pool").unwrap().clone();
        let pressure = lock_unpoisoned(&pool.pressure);
        let (started_tx, started_rx) = std_mpsc::channel();
        let (finished_tx, finished_rx) = std_mpsc::channel();

        let dropper = thread::spawn(move || {
            started_tx.send(()).unwrap();
            drop(admission);
            finished_tx.send(()).unwrap();
        });
        started_rx.recv().unwrap();
        assert!(finished_rx.recv_timeout(Duration::from_millis(20)).is_err());
        assert_eq!(pool.batch.available_permits(), 0);
        assert_eq!(pool.candidates.available_permits(), 0);

        drop(pressure);
        finished_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        dropper.join().unwrap();
        assert_eq!(pools.available_permits("pool"), (1, 1));
        assert_eq!(pools.pressure_state("pool").unwrap().release_generation, 1);

        let held = pools.try_acquire("pool", 1).unwrap();
        assert!(matches!(
            pools.try_acquire("pool", 1),
            Err(SchedulerAdmissionError::NoPermits)
        ));
        assert!(pools.pressure_state("pool").unwrap().pressure_closed);
        drop(held);
    }

    #[test]
    fn closed_pool_is_not_pressure_and_cannot_be_reopened_by_release() {
        let pools = pools(vec![pool("pressured", 1, 1), pool("closed", 1, 1)]);
        let admission = pools.try_acquire("pressured", 1).unwrap();
        assert_admission_error(
            pools.try_acquire("pressured", 1),
            SchedulerAdmissionError::NoPermits,
        );
        pools.close_pool("pressured").unwrap();
        drop(admission);
        assert_admission_error(
            pools.try_acquire("pressured", 1),
            SchedulerAdmissionError::Closed,
        );
        assert_eq!(
            pools.pressure_state("pressured").unwrap(),
            SchedulerPressureState {
                pressure_closed: true,
                release_generation: 1,
            }
        );

        pools.close_pool("closed").unwrap();
        assert_admission_error(
            pools.try_acquire("closed", 1),
            SchedulerAdmissionError::Closed,
        );
        assert_eq!(
            pools.pressure_state("closed").unwrap(),
            SchedulerPressureState::default()
        );
    }

    #[test]
    fn pool_close_is_serialized_with_a_paused_admission_release() {
        let pools = Arc::new(pools(vec![pool("pool", 1, 1)]));
        let admission = pools.try_acquire("pool", 1).unwrap();
        assert!(matches!(
            pools.try_acquire("pool", 1),
            Err(SchedulerAdmissionError::NoPermits)
        ));
        let pool = pools.pools.get("pool").unwrap().clone();
        let pressure = lock_unpoisoned(&pool.pressure);
        let (started_tx, started_rx) = std_mpsc::channel();
        let (finished_tx, finished_rx) = std_mpsc::channel();
        let closing = pools.clone();
        let closer = thread::spawn(move || {
            started_tx.send(()).unwrap();
            closing.close_pool("pool").unwrap();
            finished_tx.send(()).unwrap();
        });

        started_rx.recv().unwrap();
        assert!(finished_rx.recv_timeout(Duration::from_millis(20)).is_err());
        assert!(!pool.batch.is_closed());
        assert!(!pool.candidates.is_closed());
        drop(pressure);
        finished_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        closer.join().unwrap();

        drop(admission);
        assert!(matches!(
            pools.try_acquire("pool", 1),
            Err(SchedulerAdmissionError::Closed)
        ));
        assert!(pools.pressure_state("pool").unwrap().pressure_closed);
    }

    #[test]
    fn invalid_pool_and_candidate_count_are_distinct_from_capacity() {
        let pools = pools(vec![pool("pool", 1, 2)]);
        assert_admission_error(
            pools.try_acquire("missing", 1),
            SchedulerAdmissionError::InvalidPool,
        );
        assert_admission_error(
            pools.try_acquire("pool", 0),
            SchedulerAdmissionError::InvalidCandidateCount,
        );
        assert_admission_error(
            pools.try_acquire("pool", 3),
            SchedulerAdmissionError::InvalidCandidateCount,
        );
        assert_eq!(pools.available_permits("pool"), (1, 2));
    }

    #[test]
    fn constructor_rejects_duplicate_and_unrepresentable_pools() {
        assert_eq!(
            SchedulerAdmissionPools::from_pools(&[pool("same", 1, 1), pool("same", 1, 1)])
                .unwrap_err(),
            SchedulerAdmissionBuildError::DuplicatePool
        );

        let mut invalid = pool("invalid", 1, 1);
        invalid.concurrency.max_pending = 0;
        assert_eq!(
            SchedulerAdmissionPools::from_pools(&[invalid]).unwrap_err(),
            SchedulerAdmissionBuildError::InvalidCapacity
        );

        let config: RouterConfig = serde_json::from_value(json!({
            "mode": "off",
            "pools": []
        }))
        .unwrap();
        let empty = SchedulerAdmissionPools::from_config(&config).unwrap();
        assert_eq!(empty.total_batch_capacity(), 0);
        assert_eq!(empty.total_candidate_capacity(), 0);
    }
}
