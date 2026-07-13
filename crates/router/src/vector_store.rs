// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral vector-store semantics and the deterministic memory oracle.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::{Mutex, MutexGuard};

use crate::config::EVIDENCE_RECORDS_MAX;
use crate::vector::{
    AuthoritativeVector, NormalizedVector, PartitionId, VectorDimensions, VectorError,
    VectorRecordId, VectorSpaceId, clamp_cosine_distance, cosine_distance,
};

pub(crate) const VECTOR_TOP_K_MAX: usize = 4_095;

/// Immutable shape of one versioned vector space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VectorSpaceSpec {
    vector_space_id: VectorSpaceId,
    dimensions: VectorDimensions,
}

impl VectorSpaceSpec {
    pub(crate) fn new(vector_space_id: VectorSpaceId, dimensions: VectorDimensions) -> Self {
        Self {
            vector_space_id,
            dimensions,
        }
    }

    pub(crate) fn vector_space_id(&self) -> &VectorSpaceId {
        &self.vector_space_id
    }

    pub(crate) const fn dimensions(&self) -> VectorDimensions {
        self.dimensions
    }
}

/// One immutable evidence-link vector record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VectorRecord {
    record_id: VectorRecordId,
    vector_space_id: VectorSpaceId,
    partition_id: PartitionId,
    vector: AuthoritativeVector,
}

impl VectorRecord {
    pub(crate) fn new(
        record_id: VectorRecordId,
        vector_space_id: VectorSpaceId,
        partition_id: PartitionId,
        vector: AuthoritativeVector,
    ) -> Result<Self, VectorError> {
        if vector.vector_space_id() != &vector_space_id {
            return Err(VectorError::VectorSpaceMismatch);
        }
        Ok(Self {
            record_id,
            vector_space_id,
            partition_id,
            vector,
        })
    }

    pub(crate) const fn record_id(&self) -> VectorRecordId {
        self.record_id
    }

    pub(crate) fn vector_space_id(&self) -> &VectorSpaceId {
        &self.vector_space_id
    }

    pub(crate) const fn partition_id(&self) -> PartitionId {
        self.partition_id
    }

    pub(crate) fn vector(&self) -> &AuthoritativeVector {
        &self.vector
    }
}

/// One detached low-level nearest-neighbor result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct VectorMatch {
    record_id: VectorRecordId,
    distance: f32,
}

impl VectorMatch {
    pub(crate) fn new(record_id: VectorRecordId, distance: f32) -> Result<Self, VectorError> {
        Ok(Self {
            record_id,
            distance: clamp_cosine_distance(distance)?,
        })
    }

    pub(crate) const fn record_id(self) -> VectorRecordId {
        self.record_id
    }

    pub(crate) const fn distance(self) -> f32 {
        self.distance
    }
}

/// Search-index health for one vector space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpaceHealthState {
    Missing,
    Healthy,
    Unavailable,
    Corrupt,
}

/// Bounded health snapshot for one vector space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SpaceHealth {
    state: SpaceHealthState,
    record_count: u64,
}

impl SpaceHealth {
    pub(crate) const fn new(state: SpaceHealthState, record_count: u64) -> Self {
        Self {
            state,
            record_count,
        }
    }

    pub(crate) const fn state(self) -> SpaceHealthState {
        self.state
    }

    pub(crate) const fn record_count(self) -> u64 {
        self.record_count
    }
}

/// Successful rebuild summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RebuildReport {
    records_rebuilt: u64,
}

impl RebuildReport {
    pub(crate) const fn new(records_rebuilt: u64) -> Self {
        Self { records_rebuilt }
    }

    pub(crate) const fn records_rebuilt(self) -> u64 {
        self.records_rebuilt
    }
}

/// Stable vector-store failures without user-controlled data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VectorStoreError {
    InvalidCapacity,
    InvalidTopK,
    InvalidVector(VectorError),
    SpaceNotFound,
    Conflict,
    CapacityExceeded,
    Corrupt,
    Unavailable,
}

impl From<VectorError> for VectorStoreError {
    fn from(error: VectorError) -> Self {
        Self::InvalidVector(error)
    }
}

/// Synchronous deterministic parity contract for the in-memory oracle.
///
/// Production uses an async sole-writer/read-pool facade and does not implement
/// this trait.
pub(crate) trait VectorStore: Send + Sync {
    fn ensure_space(&self, spec: VectorSpaceSpec) -> Result<SpaceHealth, VectorStoreError>;

    fn upsert(&self, record: VectorRecord) -> Result<(), VectorStoreError>;

    fn delete(&self, record_id: VectorRecordId) -> Result<(), VectorStoreError>;

    fn search(
        &self,
        vector_space_id: &VectorSpaceId,
        partition_id: PartitionId,
        query_vector: &NormalizedVector,
        top_k: usize,
    ) -> Result<Vec<VectorMatch>, VectorStoreError>;

    fn rebuild(&self, vector_space_id: &VectorSpaceId) -> Result<RebuildReport, VectorStoreError>;

    fn health(&self, vector_space_id: &VectorSpaceId) -> Result<SpaceHealth, VectorStoreError>;
}

/// Deterministic exact-search oracle used by backend parity tests.
pub(crate) struct BruteForceMemoryStore {
    max_records: u64,
    state: Mutex<MemoryState>,
}

impl BruteForceMemoryStore {
    pub(crate) fn new(max_records: u64) -> Result<Self, VectorStoreError> {
        if !(1..=EVIDENCE_RECORDS_MAX).contains(&max_records) {
            return Err(VectorStoreError::InvalidCapacity);
        }
        Ok(Self {
            max_records,
            state: Mutex::new(MemoryState::default()),
        })
    }

    fn lock_state(&self) -> Result<MutexGuard<'_, MemoryState>, VectorStoreError> {
        self.state.lock().map_err(|_| VectorStoreError::Unavailable)
    }
}

impl fmt::Debug for BruteForceMemoryStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BruteForceMemoryStore")
            .field("max_records", &self.max_records)
            .field("state", &"<redacted>")
            .finish()
    }
}

impl VectorStore for BruteForceMemoryStore {
    fn ensure_space(&self, spec: VectorSpaceSpec) -> Result<SpaceHealth, VectorStoreError> {
        let mut state = self.lock_state()?;
        if let Some(existing) = state.spaces.get(spec.vector_space_id()) {
            if existing.spec != spec {
                return Err(VectorStoreError::Conflict);
            }
            return space_health(existing);
        }
        state.spaces.insert(
            spec.vector_space_id().clone(),
            MemorySpace {
                spec,
                health: SpaceHealthState::Healthy,
                records: BTreeMap::new(),
            },
        );
        Ok(SpaceHealth::new(SpaceHealthState::Healthy, 0))
    }

    fn upsert(&self, record: VectorRecord) -> Result<(), VectorStoreError> {
        let mut state = self.lock_state()?;
        if let Some(existing_space_id) = state.record_spaces.get(&record.record_id()) {
            let existing = state
                .spaces
                .get(existing_space_id)
                .and_then(|space| space.records.get(&record.record_id()))
                .ok_or(VectorStoreError::Corrupt)?;
            return (existing == &record)
                .then_some(())
                .ok_or(VectorStoreError::Conflict);
        }

        let space = state
            .spaces
            .get(record.vector_space_id())
            .ok_or(VectorStoreError::SpaceNotFound)?;
        require_healthy(space.health)?;
        if record.vector().vector_space_id() != record.vector_space_id() {
            return Err(VectorError::VectorSpaceMismatch.into());
        }
        if record.vector().vector().dimensions() != space.spec.dimensions() {
            return Err(VectorError::DimensionMismatch.into());
        }
        if space.records.contains_key(&record.record_id()) {
            return Err(VectorStoreError::Corrupt);
        }
        if state.record_count >= self.max_records {
            return Err(VectorStoreError::CapacityExceeded);
        }
        let record_count = state
            .record_count
            .checked_add(1)
            .ok_or(VectorStoreError::Corrupt)?;

        let record_id = record.record_id();
        let vector_space_id = record.vector_space_id().clone();
        state
            .spaces
            .get_mut(&vector_space_id)
            .expect("space validated before insertion")
            .records
            .insert(record_id, record);
        state.record_spaces.insert(record_id, vector_space_id);
        state.record_count = record_count;
        Ok(())
    }

    fn delete(&self, record_id: VectorRecordId) -> Result<(), VectorStoreError> {
        let mut state = self.lock_state()?;
        let Some(vector_space_id) = state.record_spaces.get(&record_id).cloned() else {
            return Ok(());
        };
        let space = state
            .spaces
            .get(&vector_space_id)
            .ok_or(VectorStoreError::Corrupt)?;
        require_healthy(space.health)?;
        if !space.records.contains_key(&record_id) || state.record_count == 0 {
            return Err(VectorStoreError::Corrupt);
        }

        state
            .spaces
            .get_mut(&vector_space_id)
            .expect("space validated before deletion")
            .records
            .remove(&record_id);
        state.record_spaces.remove(&record_id);
        state.record_count -= 1;
        Ok(())
    }

    fn search(
        &self,
        vector_space_id: &VectorSpaceId,
        partition_id: PartitionId,
        query_vector: &NormalizedVector,
        top_k: usize,
    ) -> Result<Vec<VectorMatch>, VectorStoreError> {
        if !(1..=VECTOR_TOP_K_MAX).contains(&top_k) {
            return Err(VectorStoreError::InvalidTopK);
        }
        let mut state = self.lock_state()?;
        let mut matches = Vec::new();
        let distance_failed = {
            let space = state
                .spaces
                .get(vector_space_id)
                .ok_or(VectorStoreError::SpaceNotFound)?;
            require_healthy(space.health)?;
            if query_vector.dimensions() != space.spec.dimensions() {
                return Err(VectorError::DimensionMismatch.into());
            }

            let mut failed = false;
            for record in space
                .records
                .values()
                .filter(|record| record.partition_id() == partition_id)
            {
                match cosine_distance(query_vector, record.vector().vector()) {
                    Ok(distance) => match VectorMatch::new(record.record_id(), distance) {
                        Ok(vector_match) => matches.push(vector_match),
                        Err(_) => {
                            failed = true;
                            break;
                        }
                    },
                    Err(_) => {
                        failed = true;
                        break;
                    }
                }
            }
            failed
        };
        if distance_failed {
            state
                .spaces
                .get_mut(vector_space_id)
                .expect("space validated before distance calculation")
                .health = SpaceHealthState::Corrupt;
            return Err(VectorStoreError::Corrupt);
        }

        matches.sort_by(|left, right| {
            left.distance
                .total_cmp(&right.distance)
                .then_with(|| left.record_id.cmp(&right.record_id))
        });
        matches.truncate(top_k);
        Ok(matches)
    }

    fn rebuild(&self, vector_space_id: &VectorSpaceId) -> Result<RebuildReport, VectorStoreError> {
        let mut state = self.lock_state()?;
        let validation = {
            let space = state
                .spaces
                .get(vector_space_id)
                .ok_or(VectorStoreError::SpaceNotFound)?;
            if space.health == SpaceHealthState::Unavailable {
                return Err(VectorStoreError::Unavailable);
            }
            let valid = space.records.values().all(|record| {
                record.vector_space_id() == vector_space_id
                    && record.vector().vector_space_id() == vector_space_id
                    && record.vector().vector().dimensions() == space.spec.dimensions()
                    && cosine_distance(record.vector().vector(), record.vector().vector()).is_ok()
            });
            (valid, record_count(space.records.len())?)
        };
        let space = state
            .spaces
            .get_mut(vector_space_id)
            .expect("space validated before rebuild result");
        if !validation.0 {
            space.health = SpaceHealthState::Corrupt;
            return Err(VectorStoreError::Corrupt);
        }
        space.health = SpaceHealthState::Healthy;
        Ok(RebuildReport::new(validation.1))
    }

    fn health(&self, vector_space_id: &VectorSpaceId) -> Result<SpaceHealth, VectorStoreError> {
        let state = self.lock_state()?;
        match state.spaces.get(vector_space_id) {
            Some(space) => space_health(space),
            None => Ok(SpaceHealth::new(SpaceHealthState::Missing, 0)),
        }
    }
}

#[derive(Default)]
struct MemoryState {
    spaces: BTreeMap<VectorSpaceId, MemorySpace>,
    record_spaces: BTreeMap<VectorRecordId, VectorSpaceId>,
    record_count: u64,
}

struct MemorySpace {
    spec: VectorSpaceSpec,
    health: SpaceHealthState,
    records: BTreeMap<VectorRecordId, VectorRecord>,
}

fn require_healthy(health: SpaceHealthState) -> Result<(), VectorStoreError> {
    match health {
        SpaceHealthState::Healthy => Ok(()),
        SpaceHealthState::Unavailable => Err(VectorStoreError::Unavailable),
        SpaceHealthState::Corrupt | SpaceHealthState::Missing => Err(VectorStoreError::Corrupt),
    }
}

fn space_health(space: &MemorySpace) -> Result<SpaceHealth, VectorStoreError> {
    Ok(SpaceHealth::new(
        space.health,
        record_count(space.records.len())?,
    ))
}

fn record_count(count: usize) -> Result<u64, VectorStoreError> {
    u64::try_from(count).map_err(|_| VectorStoreError::Corrupt)
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use uuid::Uuid;

    use super::*;

    fn space_id(byte: char) -> VectorSpaceId {
        VectorSpaceId::new(byte.to_string().repeat(64)).unwrap()
    }

    fn record_id(value: u64) -> VectorRecordId {
        let value = format!("01890f47-6c7d-7000-8000-{value:012x}");
        VectorRecordId::new(Uuid::parse_str(&value).unwrap()).unwrap()
    }

    fn dimensions(values: &[f64]) -> VectorDimensions {
        VectorDimensions::new(u32::try_from(values.len()).unwrap()).unwrap()
    }

    fn query(values: &[f64]) -> NormalizedVector {
        NormalizedVector::from_provider_f64(values, dimensions(values)).unwrap()
    }

    fn record(id: u64, space: &VectorSpaceId, partition: i64, values: &[f64]) -> VectorRecord {
        let vector = AuthoritativeVector::from_normalized(space, query(values)).unwrap();
        VectorRecord::new(
            record_id(id),
            space.clone(),
            PartitionId::new(partition).unwrap(),
            vector,
        )
        .unwrap()
    }

    fn store(max_records: u64) -> BruteForceMemoryStore {
        BruteForceMemoryStore::new(max_records).unwrap()
    }

    fn ensure(store: &BruteForceMemoryStore, space: &VectorSpaceId, dimensions: u32) {
        store
            .ensure_space(VectorSpaceSpec::new(
                space.clone(),
                VectorDimensions::new(dimensions).unwrap(),
            ))
            .unwrap();
    }

    #[test]
    fn construction_and_space_ensure_are_strict_and_idempotent() {
        assert_eq!(
            BruteForceMemoryStore::new(0).unwrap_err(),
            VectorStoreError::InvalidCapacity
        );
        assert_eq!(
            BruteForceMemoryStore::new(EVIDENCE_RECORDS_MAX + 1).unwrap_err(),
            VectorStoreError::InvalidCapacity
        );
        assert!(BruteForceMemoryStore::new(EVIDENCE_RECORDS_MAX).is_ok());

        let first = space_id('a');
        let second = space_id('b');
        let store = store(4);
        let missing = store.health(&first).unwrap();
        assert_eq!(missing.state(), SpaceHealthState::Missing);
        assert_eq!(missing.record_count(), 0);

        ensure(&store, &first, 2);
        assert_eq!(
            store
                .ensure_space(VectorSpaceSpec::new(
                    first.clone(),
                    VectorDimensions::new(2).unwrap(),
                ))
                .unwrap()
                .state(),
            SpaceHealthState::Healthy
        );
        assert_eq!(
            store.ensure_space(VectorSpaceSpec::new(
                first.clone(),
                VectorDimensions::new(3).unwrap(),
            )),
            Err(VectorStoreError::Conflict)
        );

        let wrong_space_vector =
            AuthoritativeVector::from_normalized(&first, query(&[1.0, 0.0])).unwrap();
        assert_eq!(
            VectorRecord::new(
                record_id(1),
                second,
                PartitionId::new(1).unwrap(),
                wrong_space_vector,
            ),
            Err(VectorError::VectorSpaceMismatch)
        );
    }

    #[test]
    fn crud_is_exact_idempotent_conflict_checked_and_capacity_bounded() {
        let first = space_id('a');
        let second = space_id('b');
        let store = store(1);
        ensure(&store, &first, 2);
        ensure(&store, &second, 2);
        let original = record(1, &first, 1, &[1.0, 0.0]);
        store.upsert(original.clone()).unwrap();
        store.upsert(original.clone()).unwrap();
        assert_eq!(store.health(&first).unwrap().record_count(), 1);

        assert_eq!(
            store.upsert(record(1, &first, 2, &[1.0, 0.0])),
            Err(VectorStoreError::Conflict)
        );
        assert_eq!(
            store.upsert(record(1, &second, 1, &[1.0, 0.0])),
            Err(VectorStoreError::Conflict)
        );
        assert_eq!(
            store.upsert(record(2, &first, 1, &[0.0, 1.0])),
            Err(VectorStoreError::CapacityExceeded)
        );

        store.delete(record_id(99)).unwrap();
        store.delete(original.record_id()).unwrap();
        store.delete(original.record_id()).unwrap();
        assert_eq!(store.health(&first).unwrap().record_count(), 0);
        store.upsert(record(2, &first, 1, &[0.0, 1.0])).unwrap();
    }

    #[test]
    fn missing_space_and_dimension_mismatch_do_not_mutate_state() {
        let id = space_id('a');
        let store = store(2);
        assert_eq!(
            store.upsert(record(1, &id, 1, &[1.0, 0.0])),
            Err(VectorStoreError::SpaceNotFound)
        );
        ensure(&store, &id, 1);
        assert_eq!(
            store.upsert(record(1, &id, 1, &[1.0, 0.0])),
            Err(VectorStoreError::InvalidVector(
                VectorError::DimensionMismatch
            ))
        );
        assert_eq!(store.health(&id).unwrap().record_count(), 0);
    }

    #[test]
    fn search_prefilters_strict_partition_before_ranking() {
        let id = space_id('a');
        let store = store(4);
        ensure(&store, &id, 2);
        store.upsert(record(1, &id, 2, &[1.0, 0.0])).unwrap();
        store.upsert(record(2, &id, 1, &[0.0, 1.0])).unwrap();

        let matches = store
            .search(&id, PartitionId::new(1).unwrap(), &query(&[1.0, 0.0]), 1)
            .unwrap();
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].record_id(), record_id(2));
        assert_eq!(matches[0].distance(), 1.0);
    }

    #[test]
    fn exact_duplicate_ties_use_record_id_and_top_k_is_bounded() {
        let id = space_id('a');
        let store = store(4);
        ensure(&store, &id, 2);
        assert_eq!(
            VectorMatch::new(record_id(4), f32::NAN),
            Err(VectorError::NonFiniteValue)
        );
        assert_eq!(
            VectorMatch::new(record_id(4), 2.1),
            Err(VectorError::DistanceOutOfRange)
        );
        for record in [
            record(3, &id, 1, &[1.0, 0.0]),
            record(1, &id, 1, &[1.0, 0.0]),
            record(2, &id, 1, &[1.0, 0.0]),
        ] {
            store.upsert(record).unwrap();
        }
        let partition = PartitionId::new(1).unwrap();
        let query = query(&[1.0, 0.0]);
        assert_eq!(
            store.search(&id, partition, &query, 0),
            Err(VectorStoreError::InvalidTopK)
        );
        assert_eq!(
            store.search(&id, partition, &query, VECTOR_TOP_K_MAX + 1),
            Err(VectorStoreError::InvalidTopK)
        );
        let matches = store.search(&id, partition, &query, 2).unwrap();
        assert_eq!(
            matches
                .into_iter()
                .map(VectorMatch::record_id)
                .collect::<Vec<_>>(),
            vec![record_id(1), record_id(2)]
        );
        assert!(
            store
                .search(&id, partition, &query, VECTOR_TOP_K_MAX)
                .is_ok()
        );
    }

    #[test]
    fn randomized_fixture_is_deterministic_and_insertion_order_independent() {
        let id = space_id('a');
        let forward = store(96);
        let reverse = store(96);
        ensure(&forward, &id, 4);
        ensure(&reverse, &id, 4);

        let mut seed = 0x4d59_5df4_d0f3_3173_u64;
        let mut records = Vec::new();
        for index in 0..96_u64 {
            let mut values = [0.0_f64; 4];
            for value in &mut values {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let high = u32::try_from(seed >> 32).unwrap();
                *value = f64::from(high % 2_001) - 1_000.0;
            }
            if values.iter().all(|value| *value == 0.0) {
                values[0] = 1.0;
            }
            let partition = i64::try_from(index % 3 + 1).unwrap();
            records.push(record(index + 1, &id, partition, &values));
        }
        for record in &records {
            forward.upsert(record.clone()).unwrap();
        }
        for record in records.iter().rev() {
            reverse.upsert(record.clone()).unwrap();
        }

        let query = query(&[31.0, -17.0, 11.0, 5.0]);
        for partition in 1..=3 {
            let partition = PartitionId::new(partition).unwrap();
            let first = forward.search(&id, partition, &query, 17).unwrap();
            let second = reverse.search(&id, partition, &query, 17).unwrap();
            assert_eq!(first, second);

            let mut expected = records
                .iter()
                .filter(|record| record.partition_id() == partition)
                .map(|record| {
                    VectorMatch::new(
                        record.record_id(),
                        cosine_distance(&query, record.vector().vector()).unwrap(),
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            expected.sort_by(|left, right| {
                left.distance
                    .total_cmp(&right.distance)
                    .then_with(|| left.record_id.cmp(&right.record_id))
            });
            expected.truncate(17);
            assert_eq!(first, expected);
        }
    }

    #[test]
    fn health_rebuild_and_poisoning_have_stable_results() {
        let id = space_id('a');
        let store = store(2);
        ensure(&store, &id, 2);
        store.upsert(record(1, &id, 1, &[1.0, 0.0])).unwrap();
        store
            .state
            .lock()
            .unwrap()
            .spaces
            .get_mut(&id)
            .unwrap()
            .health = SpaceHealthState::Corrupt;
        assert_eq!(
            store.health(&id).unwrap().state(),
            SpaceHealthState::Corrupt
        );
        assert_eq!(
            store.search(&id, PartitionId::new(1).unwrap(), &query(&[1.0, 0.0]), 1,),
            Err(VectorStoreError::Corrupt)
        );
        assert_eq!(store.rebuild(&id).unwrap().records_rebuilt(), 1);
        assert_eq!(
            store.health(&id).unwrap().state(),
            SpaceHealthState::Healthy
        );

        let poisoned = catch_unwind(AssertUnwindSafe(|| {
            let _guard = store.state.lock().unwrap();
            panic!("poison memory oracle");
        }));
        assert!(poisoned.is_err());
        assert_eq!(store.health(&id), Err(VectorStoreError::Unavailable));
    }
}
