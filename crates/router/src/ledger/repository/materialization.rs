// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic evidence-link and initial vector materialization state.

use std::collections::BTreeSet;

use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use serde_json::json;
use uuid::Uuid;

use super::embedding::{
    EmbeddingFailurePropagationAck, EmbeddingFailurePropagationUpdate, EmbeddingJobCreate,
    EmbeddingJobCreateAck, create_embedding_job_in_transaction, load_verified_embedding_job,
    update_embedding_failure_propagation_in_transaction,
};
use super::process::{ProcessStatusAt, append_integrity_health, verified_process_status_at};
use super::shadow::{
    AtomicShadowVectorization, ReservedShadowAttempt, SampleBatchReservation, ShadowTerminalClass,
    ShadowTerminalRecord, ShadowVectorSourceV1, ShadowVectorizationHandoff,
    VerifiedVectorEvaluation, load_verified_backfill_vector_source,
};
use super::vector_catalog::{
    CanonicalQueryEnsureAck, RoutingPartitionEnsure, RoutingPartitionEnsureAck, SourceChangeAppend,
    SourceChangeAppendAck, VectorMappingAuthority, append_source_change, ensure_canonical_query,
    ensure_routing_partition, load_canonical_query, load_embedding_cache, load_routing_partition,
    verify_historical_routing_partition,
};
use super::vector_index::{
    ActiveGenerationResolution, GenerationRetirementAck, VectorIndexManifestState,
    VectorIndexPointMutationAck, VectorSourceChangeOperation, current_generation_manifest,
    delete_active_record, resolve_active_generation, retire_current_generation_for_retention,
    upsert_active_record, vector_source_change_payload_hash,
};
use super::vector_registry::{
    FrozenMappingKey, FrozenPoolVectorAuthority, VerifiedPoolVectorSpaceMapping,
    resolve_frozen_mapping, resolve_frozen_pool_vector_authority, resolve_vector_space,
};
use super::{LedgerRepository, TransactionStartGuard, map_fs_error};
use crate::canonical_json::canonical_sha256;
use crate::canonical_query::{CanonicalRoutingQueryArtifactV1, build_canonical_routing_query};
use crate::ledger::fs::enforce_sidecar_permissions;
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::projection::RouterRoutingContextProjectionV1;
use crate::routing_partition::{
    RoutingPartitionArtifactV1, RoutingPartitionV1, build_routing_partition_v1,
};
use crate::vector::{PartitionId, VectorRecordId, VectorSpaceId};
use crate::vector_store::VectorRecord;

const MATERIALIZATION_LEASE_MILLIS: i64 = 60_000;
pub(crate) const MATERIALIZATION_ATTEMPT_MAX: i64 = 64;
const MATERIALIZATION_ATTEMPT_LIMIT_CLASS: &str = "router.vector.materialization_attempt_limit";
const MATERIALIZATION_STATE_HISTORY_MAX: usize = 2 * MATERIALIZATION_ATTEMPT_MAX as usize + 2;
const LINK_STATE_HISTORY_MAX: usize = MATERIALIZATION_ATTEMPT_MAX as usize + 2;

#[derive(Debug)]
enum InitialMaterialization {
    PendingEmbedding {
        embedding_job_id: String,
    },
    FailedEmbedding {
        embedding_job_id: String,
        stable_error_class: String,
    },
    PendingIndex {
        embedding_id: Uuid,
    },
    Ready {
        embedding_id: Uuid,
    },
}

impl InitialMaterialization {
    fn state(&self) -> &'static str {
        match self {
            Self::PendingEmbedding { .. } => "pending_embedding",
            Self::FailedEmbedding { .. } => "failed_embedding",
            Self::PendingIndex { .. } => "pending_index",
            Self::Ready { .. } => "ready",
        }
    }

    fn embedding_job_id(&self) -> Option<&str> {
        match self {
            Self::PendingEmbedding { embedding_job_id }
            | Self::FailedEmbedding {
                embedding_job_id, ..
            } => Some(embedding_job_id),
            Self::PendingIndex { .. } | Self::Ready { .. } => None,
        }
    }

    fn embedding_id(&self) -> Option<Uuid> {
        match self {
            Self::PendingIndex { embedding_id } | Self::Ready { embedding_id } => {
                Some(*embedding_id)
            }
            Self::PendingEmbedding { .. } | Self::FailedEmbedding { .. } => None,
        }
    }

    fn stable_error_class(&self) -> Option<&str> {
        match self {
            Self::FailedEmbedding {
                stable_error_class, ..
            } => Some(stable_error_class),
            _ => None,
        }
    }
}

/// Canonical current facts for one evidence-link materialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaterializationSnapshot {
    pub(crate) materialization_job_id: String,
    pub(crate) evidence_vector_link_id: Uuid,
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) canonical_query_hash: String,
    pub(crate) embedding_job_id: Option<String>,
    pub(crate) embedding_id: Option<Uuid>,
    pub(crate) attempt_generation: i64,
    pub(crate) attempt_count: i64,
    pub(crate) next_eligible_at_unix_ms: i64,
    pub(crate) created_at_unix_ms: i64,
    pub(crate) canonical_payload_hash: String,
}

/// Complete search and index facts emitted only after their relational graph verifies.
#[derive(Clone, PartialEq)]
pub(crate) struct VerifiedVectorLinkSource {
    pub(crate) record_id: VectorRecordId,
    pub(crate) project_uuid: Uuid,
    pub(crate) pool_id: String,
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) partition_id: PartitionId,
    pub(crate) canonical_query_hash: String,
    pub(crate) canonical_query: CanonicalRoutingQueryArtifactV1,
    pub(crate) partition: RoutingPartitionV1,
    pub(crate) vector_record: Option<VectorRecord>,
    pub(crate) shadow_attempt_id: Uuid,
    pub(crate) shadow_result_id: Uuid,
    pub(crate) anchor_id: Uuid,
    pub(crate) root_uuid: Uuid,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) terminal_class: ShadowTerminalClass,
    pub(crate) evaluation: Option<VerifiedVectorEvaluation>,
    pub(crate) quality_label: Option<String>,
    pub(crate) vector_state: String,
    pub(crate) created_at_unix_ms: i64,
    pub(crate) record_hash: String,
}

/// Distinguishes a detached index ID from a present link with missing authority.
#[derive(Clone, PartialEq)]
pub(crate) enum VerifiedVectorLinkSourceLoad {
    LinkMissing,
    AuthorityMissing,
    Verified(Box<VerifiedVectorLinkSource>),
}

/// One bounded all-link scan. The cursor advances over non-ready links as well.
pub(crate) struct VerifiedVectorLinkPage {
    pub(crate) ready_records: Vec<VectorRecord>,
    pub(crate) last_scanned_record_id: Option<VectorRecordId>,
    pub(crate) exhausted: bool,
}

/// Current materialization lease authority returned only to its owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaterializationLease {
    pub(crate) job: MaterializationSnapshot,
    pub(crate) lease_owner_process_instance_id: Uuid,
    pub(crate) lease_token: Uuid,
    pub(crate) lease_expires_at_unix_ms: i64,
    pub(crate) state_event_hash: String,
}

/// Frozen request to claim or reclaim one cache-backed materialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaterializationClaim {
    pub(crate) claimed_state_event_id: Uuid,
    pub(crate) orphaned_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) materialization_job_id: String,
    pub(crate) lease_token: Uuid,
    pub(crate) expected_attempt_generation: i64,
    pub(crate) expected_canonical_payload_hash: String,
    pub(crate) observed_at_unix_ms: i64,
    pub(crate) lease_expires_at_unix_ms: i64,
}

impl MaterializationClaim {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        claimed_state_event_id: Uuid,
        orphaned_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        materialization_job_id: impl Into<String>,
        lease_token: Uuid,
        expected_attempt_generation: i64,
        expected_canonical_payload_hash: impl Into<String>,
        observed_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_distinct_uuid_v7(&[
            claimed_state_event_id,
            orphaned_state_event_id,
            conflict_health_event_id,
            lease_token,
        ])?;
        let materialization_job_id = materialization_job_id.into();
        let expected_canonical_payload_hash = expected_canonical_payload_hash.into();
        validate_sha256(&materialization_job_id)?;
        validate_sha256(&expected_canonical_payload_hash)?;
        if expected_attempt_generation < 0 || observed_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let lease_expires_at_unix_ms = observed_at_unix_ms
            .checked_add(MATERIALIZATION_LEASE_MILLIS)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        Ok(Self {
            claimed_state_event_id,
            orphaned_state_event_id,
            conflict_health_event_id,
            materialization_job_id,
            lease_token,
            expected_attempt_generation,
            expected_canonical_payload_hash,
            observed_at_unix_ms,
            lease_expires_at_unix_ms,
        })
    }
}

/// Exhaustive result of a fenced materialization claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MaterializationClaimAck {
    Claimed(MaterializationLease),
    Reclaimed(MaterializationLease),
    AlreadyApplied(MaterializationLease),
    NotFound,
    CacheNotReady,
    IndexNotReady,
    AttemptLimitTerminal(MaterializationSnapshot),
    NotEligible { next_eligible_at_unix_ms: i64 },
    LeaseHeld { lease_expires_at_unix_ms: i64 },
    Terminal,
    Stale,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

/// Frozen request to attach authoritative cache data and finish one lease.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaterializationCompletion {
    pub(crate) materialization_state_event_id: Uuid,
    pub(crate) link_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) materialization_job_id: String,
    pub(crate) lease_token: Uuid,
    pub(crate) expected_attempt_generation: i64,
    pub(crate) expected_canonical_payload_hash: String,
    pub(crate) completed_at_unix_ms: i64,
}

impl MaterializationCompletion {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        materialization_state_event_id: Uuid,
        link_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        materialization_job_id: impl Into<String>,
        lease_token: Uuid,
        expected_attempt_generation: i64,
        expected_canonical_payload_hash: impl Into<String>,
        completed_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_distinct_uuid_v7(&[
            materialization_state_event_id,
            link_state_event_id,
            conflict_health_event_id,
            lease_token,
        ])?;
        let materialization_job_id = materialization_job_id.into();
        let expected_canonical_payload_hash = expected_canonical_payload_hash.into();
        validate_sha256(&materialization_job_id)?;
        validate_sha256(&expected_canonical_payload_hash)?;
        if expected_attempt_generation <= 0 || completed_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            materialization_state_event_id,
            link_state_event_id,
            conflict_health_event_id,
            materialization_job_id,
            lease_token,
            expected_attempt_generation,
            expected_canonical_payload_hash,
            completed_at_unix_ms,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MaterializationCompletionState {
    Ready,
    PendingIndex,
}

impl MaterializationCompletionState {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::PendingIndex => "pending_index",
        }
    }
}

/// Exhaustive result of a cache attachment and index attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MaterializationCompletionAck {
    Applied {
        state: MaterializationCompletionState,
        job: MaterializationSnapshot,
    },
    AlreadyApplied {
        state: MaterializationCompletionState,
        job: MaterializationSnapshot,
    },
    NotFound,
    CacheNotReady,
    StaleLease,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MaterializationResolutionKind {
    Released,
    RetryScheduled {
        stable_error_class: String,
        next_eligible_at_unix_ms: i64,
    },
}

/// Frozen owner/token/generation/hash-fenced non-success resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaterializationResolution {
    pub(crate) materialization_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) materialization_job_id: String,
    pub(crate) lease_token: Uuid,
    pub(crate) expected_attempt_generation: i64,
    pub(crate) expected_canonical_payload_hash: String,
    pub(crate) resolved_at_unix_ms: i64,
    pub(crate) kind: MaterializationResolutionKind,
}

impl MaterializationResolution {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        materialization_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        materialization_job_id: impl Into<String>,
        lease_token: Uuid,
        expected_attempt_generation: i64,
        expected_canonical_payload_hash: impl Into<String>,
        resolved_at_unix_ms: i64,
        kind: MaterializationResolutionKind,
    ) -> Result<Self, LedgerError> {
        validate_distinct_uuid_v7(&[
            materialization_state_event_id,
            conflict_health_event_id,
            lease_token,
        ])?;
        let materialization_job_id = materialization_job_id.into();
        let expected_canonical_payload_hash = expected_canonical_payload_hash.into();
        validate_sha256(&materialization_job_id)?;
        validate_sha256(&expected_canonical_payload_hash)?;
        if expected_attempt_generation <= 0 || resolved_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        if let MaterializationResolutionKind::RetryScheduled {
            stable_error_class,
            next_eligible_at_unix_ms,
        } = &kind
        {
            validate_stable_error_class(stable_error_class)?;
            if *next_eligible_at_unix_ms < resolved_at_unix_ms {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        Ok(Self {
            materialization_state_event_id,
            conflict_health_event_id,
            materialization_job_id,
            lease_token,
            expected_attempt_generation,
            expected_canonical_payload_hash,
            resolved_at_unix_ms,
            kind,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MaterializationResolvedState {
    Released,
    RetryScheduled,
}

impl MaterializationResolvedState {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Released => "released",
            Self::RetryScheduled => "retry_scheduled",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MaterializationResolutionAck {
    Applied {
        state: MaterializationResolvedState,
        job: MaterializationSnapshot,
        state_event_hash: String,
    },
    AlreadyApplied {
        state: MaterializationResolvedState,
        job: MaterializationSnapshot,
        state_event_hash: String,
    },
    NotFound,
    StaleLease,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

/// Transaction-local retention request for one materialization graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaterializationRetentionDelete {
    pub(crate) materialization_state_event_id: Uuid,
    pub(crate) link_state_event_id: Uuid,
    pub(crate) materialization_job_id: String,
    pub(crate) expected_attempt_generation: i64,
    pub(crate) expected_canonical_payload_hash: String,
    pub(crate) deleted_at_unix_ms: i64,
}

impl MaterializationRetentionDelete {
    pub(crate) fn new(
        materialization_state_event_id: Uuid,
        link_state_event_id: Uuid,
        materialization_job_id: impl Into<String>,
        expected_attempt_generation: i64,
        expected_canonical_payload_hash: impl Into<String>,
        deleted_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_distinct_uuid_v7(&[materialization_state_event_id, link_state_event_id])?;
        let materialization_job_id = materialization_job_id.into();
        let expected_canonical_payload_hash = expected_canonical_payload_hash.into();
        validate_sha256(&materialization_job_id)?;
        validate_sha256(&expected_canonical_payload_hash)?;
        if expected_attempt_generation < 0 || deleted_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            materialization_state_event_id,
            link_state_event_id,
            materialization_job_id,
            expected_attempt_generation,
            expected_canonical_payload_hash,
            deleted_at_unix_ms,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MaterializationRetentionAck {
    Deleted,
    AlreadyAbsent,
    IndexUnavailable {
        vector_space_id: VectorSpaceId,
        expected_generation: crate::sqlite_vec_schema::VectorIndexGeneration,
        expected_manifest_hash: String,
    },
    Stale,
    Conflict,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaterializationFailurePropagationItem {
    pub(crate) evidence_vector_link_id: Uuid,
    pub(crate) materialization_state_event_id: Uuid,
    pub(crate) link_state_event_id: Uuid,
}

impl MaterializationFailurePropagationItem {
    pub(crate) fn new(
        evidence_vector_link_id: Uuid,
        materialization_state_event_id: Uuid,
        link_state_event_id: Uuid,
    ) -> Result<Self, LedgerError> {
        validate_distinct_uuid_v7(&[
            evidence_vector_link_id,
            materialization_state_event_id,
            link_state_event_id,
        ])?;
        Ok(Self {
            evidence_vector_link_id,
            materialization_state_event_id,
            link_state_event_id,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaterializationFailurePropagation {
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) embedding_job_id: String,
    pub(crate) expected_embedding_job_hash: String,
    pub(crate) expected_attempt_generation: i64,
    pub(crate) expected_cursor: Option<Uuid>,
    pub(crate) propagated_at_unix_ms: i64,
    pub(crate) items: Vec<MaterializationFailurePropagationItem>,
}

impl MaterializationFailurePropagation {
    pub(crate) fn new(
        conflict_health_event_id: Uuid,
        embedding_job_id: impl Into<String>,
        expected_embedding_job_hash: impl Into<String>,
        expected_attempt_generation: i64,
        expected_cursor: Option<Uuid>,
        propagated_at_unix_ms: i64,
        items: Vec<MaterializationFailurePropagationItem>,
    ) -> Result<Self, LedgerError> {
        let embedding_job_id = embedding_job_id.into();
        let expected_embedding_job_hash = expected_embedding_job_hash.into();
        validate_distinct_uuid_v7(&[conflict_health_event_id])?;
        validate_sha256(&embedding_job_id)?;
        validate_sha256(&expected_embedding_job_hash)?;
        if expected_attempt_generation <= 0 || propagated_at_unix_ms < 0 || items.len() > 256 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let mut links = std::collections::BTreeSet::new();
        let mut events = std::collections::BTreeSet::new();
        events.insert(conflict_health_event_id);
        if let Some(cursor) = expected_cursor {
            validate_distinct_uuid_v7(&[cursor])?;
        }
        for item in &items {
            if !links.insert(item.evidence_vector_link_id)
                || !events.insert(item.materialization_state_event_id)
                || !events.insert(item.link_state_event_id)
            {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        Ok(Self {
            conflict_health_event_id,
            embedding_job_id,
            expected_embedding_job_hash,
            expected_attempt_generation,
            expected_cursor,
            propagated_at_unix_ms,
            items,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum MaterializationFailurePropagationAck {
    Applied {
        processed: usize,
        next_cursor: Option<Uuid>,
        complete: bool,
    },
    AlreadyApplied {
        processed: usize,
        next_cursor: Option<Uuid>,
        complete: bool,
    },
    NotFound,
    NotTerminal,
    Stale,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

/// Frozen authority and fresh write identities for one retained terminal backfill.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VectorBackfillCommand {
    pub(crate) mapping: FrozenMappingKey,
    pub(crate) shadow_attempt_id: Uuid,
    pub(crate) evidence_vector_link_id: Uuid,
    pub(crate) evidence_link_state_event_id: Uuid,
    pub(crate) materialization_state_event_id: Uuid,
    pub(crate) embedding_job_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) backfilled_at_unix_ms: i64,
}

impl VectorBackfillCommand {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        mapping: FrozenMappingKey,
        shadow_attempt_id: Uuid,
        evidence_vector_link_id: Uuid,
        evidence_link_state_event_id: Uuid,
        materialization_state_event_id: Uuid,
        embedding_job_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        backfilled_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        let verified_mapping = FrozenMappingKey::new(
            mapping.project_uuid,
            mapping.config_generation_id.clone(),
            mapping.pool_id.clone(),
            mapping.policy_version_id.clone(),
        )
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        validate_distinct_uuid_v7(&[
            shadow_attempt_id,
            evidence_vector_link_id,
            evidence_link_state_event_id,
            materialization_state_event_id,
            embedding_job_state_event_id,
            conflict_health_event_id,
        ])?;
        if mapping != verified_mapping || backfilled_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            mapping,
            shadow_attempt_id,
            evidence_vector_link_id,
            evidence_link_state_event_id,
            materialization_state_event_id,
            embedding_job_state_event_id,
            conflict_health_event_id,
            backfilled_at_unix_ms,
        })
    }
}

/// Exhaustive result of an atomic current-space terminal backfill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VectorBackfillAck {
    Applied,
    AlreadyApplied,
    NotFound,
    MappingNotFound,
    MappingNotCurrent,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

impl LedgerRepository {
    pub(crate) fn backfill_vector_graph(
        &mut self,
        command: &VectorBackfillCommand,
    ) -> Result<VectorBackfillAck, LedgerError> {
        self.backfill_vector_graph_with_start_check(command, || Some(()))
    }

    pub(crate) fn backfill_vector_graph_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &VectorBackfillCommand,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<VectorBackfillAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(VectorBackfillAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(VectorBackfillAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(VectorBackfillAck::TransactionNotStarted);
        }
        drop(start_guard);
        match verified_process_status_at(
            &transaction,
            project_uuid,
            process_instance_id,
            command.backfilled_at_unix_ms,
        )? {
            ProcessStatusAt::Live => {}
            ProcessStatusAt::Expired | ProcessStatusAt::Terminal => {
                return Ok(VectorBackfillAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        transaction
            .execute_batch("SAVEPOINT vector_backfill_graph")
            .map_err(database_error)?;
        let graph_result = backfill_vector_graph_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            command,
        );
        if matches!(graph_result, Ok(VectorBackfillAck::Conflict) | Err(_)) {
            transaction
                .execute_batch("ROLLBACK TO vector_backfill_graph; RELEASE vector_backfill_graph")
                .map_err(database_error)?;
        } else {
            transaction
                .execute_batch("RELEASE vector_backfill_graph")
                .map_err(database_error)?;
        }
        let acknowledgement = graph_result?;
        if acknowledgement == VectorBackfillAck::Conflict {
            append_integrity_health(
                &transaction,
                command.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                None,
                None,
                command.backfilled_at_unix_ms,
            )?;
        }
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }

    pub(crate) fn claim_materialization(
        &mut self,
        command: &MaterializationClaim,
    ) -> Result<MaterializationClaimAck, LedgerError> {
        self.claim_materialization_with_start_check(command, || Some(()))
    }

    pub(crate) fn claim_materialization_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &MaterializationClaim,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<MaterializationClaimAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(MaterializationClaimAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(MaterializationClaimAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(MaterializationClaimAck::TransactionNotStarted);
        }
        drop(start_guard);
        match verified_process_status_at(
            &transaction,
            project_uuid,
            process_instance_id,
            command.observed_at_unix_ms,
        )? {
            ProcessStatusAt::Live => {}
            ProcessStatusAt::Expired | ProcessStatusAt::Terminal => {
                return Ok(MaterializationClaimAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        let acknowledgement = claim_materialization_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            command,
        )?;
        if acknowledgement == MaterializationClaimAck::Conflict {
            append_integrity_health(
                &transaction,
                command.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                None,
                None,
                command.observed_at_unix_ms,
            )?;
        }
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }

    pub(crate) fn complete_materialization(
        &mut self,
        command: &MaterializationCompletion,
    ) -> Result<MaterializationCompletionAck, LedgerError> {
        self.complete_materialization_with_start_check(command, || Some(()))
    }

    pub(crate) fn complete_materialization_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &MaterializationCompletion,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<MaterializationCompletionAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(MaterializationCompletionAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(MaterializationCompletionAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(MaterializationCompletionAck::TransactionNotStarted);
        }
        drop(start_guard);
        match verified_process_status_at(
            &transaction,
            project_uuid,
            process_instance_id,
            command.completed_at_unix_ms,
        )? {
            ProcessStatusAt::Live => {}
            ProcessStatusAt::Expired | ProcessStatusAt::Terminal => {
                return Ok(MaterializationCompletionAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        let acknowledgement = complete_materialization_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            command,
        )?;
        if acknowledgement == MaterializationCompletionAck::Conflict {
            append_integrity_health(
                &transaction,
                command.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                None,
                None,
                command.completed_at_unix_ms,
            )?;
        }
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }

    pub(crate) fn resolve_materialization(
        &mut self,
        command: &MaterializationResolution,
    ) -> Result<MaterializationResolutionAck, LedgerError> {
        self.resolve_materialization_with_start_check(command, || Some(()))
    }

    pub(crate) fn resolve_materialization_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &MaterializationResolution,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<MaterializationResolutionAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(MaterializationResolutionAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(MaterializationResolutionAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(MaterializationResolutionAck::TransactionNotStarted);
        }
        drop(start_guard);
        match verified_process_status_at(
            &transaction,
            project_uuid,
            process_instance_id,
            command.resolved_at_unix_ms,
        )? {
            ProcessStatusAt::Live => {}
            ProcessStatusAt::Expired | ProcessStatusAt::Terminal => {
                return Ok(MaterializationResolutionAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        let acknowledgement = resolve_materialization_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            command,
        )?;
        if acknowledgement == MaterializationResolutionAck::Conflict {
            append_integrity_health(
                &transaction,
                command.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                None,
                None,
                command.resolved_at_unix_ms,
            )?;
        }
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }

    pub(crate) fn propagate_materialization_failure(
        &mut self,
        command: &MaterializationFailurePropagation,
    ) -> Result<MaterializationFailurePropagationAck, LedgerError> {
        self.propagate_materialization_failure_with_start_check(command, || Some(()))
    }

    pub(crate) fn propagate_materialization_failure_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &MaterializationFailurePropagation,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<MaterializationFailurePropagationAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(MaterializationFailurePropagationAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(MaterializationFailurePropagationAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(MaterializationFailurePropagationAck::TransactionNotStarted);
        }
        drop(start_guard);
        match verified_process_status_at(
            &transaction,
            project_uuid,
            process_instance_id,
            command.propagated_at_unix_ms,
        )? {
            ProcessStatusAt::Live => {}
            ProcessStatusAt::Expired | ProcessStatusAt::Terminal => {
                return Ok(MaterializationFailurePropagationAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        let acknowledgement = propagate_materialization_failure_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            command,
        )?;
        if acknowledgement == MaterializationFailurePropagationAck::Conflict {
            append_integrity_health(
                &transaction,
                command.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                None,
                None,
                command.propagated_at_unix_ms,
            )?;
        }
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }
}

#[derive(Debug, Clone)]
struct StoredMaterialization {
    materialization_job_id: String,
    evidence_vector_link_id: String,
    vector_space_id: String,
    canonical_query_hash: String,
    embedding_job_id: Option<String>,
    embedding_id: Option<String>,
    lease_owner_process_instance_id: Option<String>,
    lease_token: Option<String>,
    lease_expires_at_unix_ms: Option<i64>,
    attempt_generation: i64,
    attempt_count: i64,
    next_eligible_at_unix_ms: i64,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug, Clone)]
struct StoredMaterializationState {
    state_event_id: String,
    materialization_job_id: String,
    process_instance_id: Option<String>,
    state: String,
    attempt_generation: i64,
    stable_error_class: Option<String>,
    lease_token: Option<String>,
    lease_expires_at_unix_ms: Option<i64>,
    attempt_count: i64,
    next_eligible_at_unix_ms: i64,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug)]
struct StoredEvidenceVectorLink {
    evidence_vector_link_id: String,
    vectorization_outcome_id: String,
    shadow_attempt_id: String,
    anchor_id: String,
    root_uuid: String,
    learning_generation_id: String,
    vector_space_id: String,
    partition_id: i64,
    canonical_query_hash: String,
    terminal_class: String,
    evaluation_id: Option<String>,
    quality_label: Option<String>,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug)]
struct StoredVectorizationOutcome {
    vectorization_outcome_id: String,
    shadow_attempt_id: String,
    anchor_id: String,
    learning_generation_id: String,
    vector_space_id: String,
    canonical_query_hash: Option<String>,
    outcome: String,
    stable_reason: Option<String>,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

pub(crate) fn backfill_vector_graph_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &VectorBackfillCommand,
) -> Result<VectorBackfillAck, LedgerError> {
    let validated = VectorBackfillCommand::new(
        command.mapping.clone(),
        command.shadow_attempt_id,
        command.evidence_vector_link_id,
        command.evidence_link_state_event_id,
        command.materialization_state_event_id,
        command.embedding_job_state_event_id,
        command.conflict_health_event_id,
        command.backfilled_at_unix_ms,
    )?;
    if validated != *command || command.mapping.project_uuid != project_uuid {
        return Ok(VectorBackfillAck::Conflict);
    }
    let current_config = transaction
        .query_row(
            "SELECT config_generation_id FROM process_instances
             WHERE process_instance_id = ?1 AND project_uuid = ?2",
            params![process_instance_id.to_string(), project_uuid.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if current_config != command.mapping.config_generation_id {
        return Ok(VectorBackfillAck::MappingNotCurrent);
    }
    let Some(source) =
        load_verified_backfill_vector_source(transaction, project_uuid, command.shadow_attempt_id)?
    else {
        return Ok(VectorBackfillAck::NotFound);
    };
    let source_retiring = transaction
        .query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM decision_retiring_anchors WHERE anchor_id = ?1
             )",
            [source.reservation.anchor_id.to_string()],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)?;
    if source_retiring {
        return Ok(VectorBackfillAck::NotFound);
    }
    if source.reservation.pool_id != command.mapping.pool_id
        || command.backfilled_at_unix_ms < source.terminal_at_unix_ms
    {
        return Ok(VectorBackfillAck::Conflict);
    }
    let Some(mapping) = resolve_frozen_mapping(transaction, &command.mapping)? else {
        return Ok(VectorBackfillAck::MappingNotFound);
    };
    if command.backfilled_at_unix_ms < mapping.created_at_unix_ms {
        return Ok(VectorBackfillAck::Conflict);
    }
    let vectorization = AtomicShadowVectorization::new(
        source.routing_projection.clone(),
        command.evidence_vector_link_id,
        command.evidence_link_state_event_id,
        command.materialization_state_event_id,
        command.embedding_job_state_event_id,
    )?;
    let terminal = VectorGraphTerminal {
        shadow_attempt_id: command.shadow_attempt_id,
        terminal_class: source.terminal_class,
        evaluation_id: source.evaluation_id,
        vector_source: source.vector_source.clone(),
        vectorization,
        conflict_health_event_id: command.conflict_health_event_id,
        created_at_unix_ms: command.backfilled_at_unix_ms,
    };
    let mapping_authority = VectorMappingAuthority {
        project_uuid,
        config_generation_id: command.mapping.config_generation_id.clone(),
        pool_id: command.mapping.pool_id.clone(),
        mapping_policy_version_id: command.mapping.policy_version_id.clone(),
    };
    let graph_exists = vector_graph_identity_exists(
        transaction,
        command.shadow_attempt_id,
        mapping.mapping.vector_space_id.as_str(),
    )?;
    let exact = record_enabled_graph(
        transaction,
        project_uuid,
        process_instance_id,
        &source.reservation,
        &source.attempt,
        source.root_uuid,
        &source.routing_projection,
        &terminal,
        &mapping,
        &mapping_authority,
        !graph_exists,
    )?;
    Ok(match (graph_exists, exact) {
        (false, true) => VectorBackfillAck::Applied,
        (true, true) => VectorBackfillAck::AlreadyApplied,
        (_, false) => VectorBackfillAck::Conflict,
    })
}

/// Reconstruct and verify one retained vector graph from immutable Shadow authority.
pub(crate) fn verify_retained_vector_graph_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    shadow_attempt_id: Uuid,
    vector_space_id: &VectorSpaceId,
) -> Result<bool, LedgerError> {
    let Some(source) =
        load_verified_backfill_vector_source(transaction, project_uuid, shadow_attempt_id)?
    else {
        return Ok(false);
    };
    let shadow_process_instance_id = transaction
        .query_row(
            "SELECT process_instance_id FROM shadow_attempts
             WHERE shadow_attempt_id = ?1 AND project_uuid = ?2",
            params![shadow_attempt_id.to_string(), project_uuid.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .as_deref()
        .map(parse_uuid_v7)
        .transpose()?;
    let Some(shadow_process_instance_id) = shadow_process_instance_id else {
        return Ok(false);
    };
    let graph_created_at = transaction
        .query_row(
            "SELECT created_at_unix_ms FROM vectorization_outcomes
             WHERE shadow_attempt_id = ?1 AND vector_space_id = ?2",
            params![shadow_attempt_id.to_string(), vector_space_id.as_str()],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?;
    let Some(graph_created_at) = graph_created_at else {
        return Ok(false);
    };
    if graph_created_at < source.terminal_at_unix_ms {
        return Ok(false);
    }

    let stored_link_id = transaction
        .query_row(
            "SELECT evidence_vector_link_id FROM evidence_vector_links
             WHERE shadow_attempt_id = ?1 AND vector_space_id = ?2",
            params![shadow_attempt_id.to_string(), vector_space_id.as_str()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let link_id = stored_link_id
        .as_deref()
        .map(parse_uuid_v7)
        .transpose()?
        .unwrap_or_else(Uuid::now_v7);
    let link_state_event_id = match stored_link_id.as_deref() {
        Some(stored_link_id) => transaction
            .query_row(
                "SELECT evidence_vector_link_state_event_id
                 FROM evidence_vector_link_state_events
                 WHERE evidence_vector_link_id = ?1
                 ORDER BY event_seq LIMIT 1",
                params![stored_link_id],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(database_error)?
            .as_deref()
            .map(parse_uuid_v7)
            .transpose()?
            .unwrap_or_else(Uuid::now_v7),
        None => Uuid::now_v7(),
    };
    let materialization_job_id = materialization_job_id(link_id)?;
    let initial_materialization_state = transaction
        .query_row(
            "SELECT vector_materialization_job_state_event_id, process_instance_id
             FROM vector_materialization_job_state_events
             WHERE vector_materialization_job_id = ?1
             ORDER BY event_seq LIMIT 1",
            params![materialization_job_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    if stored_link_id.is_some() && initial_materialization_state.is_none() {
        return Ok(false);
    }
    let (materialization_state_event_id, process_instance_id) = match initial_materialization_state
    {
        Some((event_id, Some(process_instance_id))) => (
            parse_uuid_v7(&event_id)?,
            parse_uuid_v7(&process_instance_id)?,
        ),
        Some((_, None)) => return Ok(false),
        None => (Uuid::now_v7(), shadow_process_instance_id),
    };
    let actor_project = transaction
        .query_row(
            "SELECT project_uuid FROM process_instances WHERE process_instance_id = ?1",
            params![process_instance_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    if actor_project.as_deref() != Some(project_uuid.to_string().as_str()) {
        return Ok(false);
    }
    // The handoff event is intentionally absent when this graph joins an existing shared job.
    let embedding_job_state_event_id = Uuid::now_v7();
    let mut occupied = BTreeSet::from([
        link_id,
        link_state_event_id,
        materialization_state_event_id,
        embedding_job_state_event_id,
    ]);
    if occupied.len() != 4 {
        return Ok(false);
    }
    let conflict_health_event_id = loop {
        let candidate = Uuid::now_v7();
        if occupied.insert(candidate) {
            break candidate;
        }
    };
    let vectorization = AtomicShadowVectorization::new(
        source.routing_projection.clone(),
        link_id,
        link_state_event_id,
        materialization_state_event_id,
        embedding_job_state_event_id,
    )?;
    let terminal = VectorGraphTerminal {
        shadow_attempt_id,
        terminal_class: source.terminal_class,
        evaluation_id: source.evaluation_id,
        vector_source: source.vector_source.clone(),
        vectorization,
        conflict_health_event_id,
        created_at_unix_ms: graph_created_at,
    };
    let mapping_keys = transaction
        .prepare(
            "SELECT config_generation_id, policy_version_id
             FROM pool_vector_space_mappings
             WHERE project_uuid = ?1 AND pool_id = ?2 AND vector_space_id = ?3
             ORDER BY config_generation_id, policy_version_id",
        )
        .map_err(database_error)?
        .query_map(
            params![
                project_uuid.to_string(),
                source.reservation.pool_id,
                vector_space_id.as_str(),
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    for (config_generation_id, policy_version_id) in mapping_keys {
        let key = FrozenMappingKey::new(
            project_uuid,
            config_generation_id.clone(),
            source.reservation.pool_id.clone(),
            policy_version_id.clone(),
        )
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        let Some(mapping) = resolve_frozen_mapping(transaction, &key)? else {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        };
        if mapping.mapping.vector_space_id != *vector_space_id {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
        let mapping_authority = VectorMappingAuthority {
            project_uuid,
            config_generation_id,
            pool_id: source.reservation.pool_id.clone(),
            mapping_policy_version_id: policy_version_id,
        };
        if record_enabled_graph(
            transaction,
            project_uuid,
            process_instance_id,
            &source.reservation,
            &source.attempt,
            source.root_uuid,
            &source.routing_projection,
            &terminal,
            &mapping,
            &mapping_authority,
            false,
        )? {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) fn claim_materialization_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &MaterializationClaim,
) -> Result<MaterializationClaimAck, LedgerError> {
    let Some(stored) =
        load_materialization(transaction, project_uuid, &command.materialization_job_id)?
    else {
        return Ok(MaterializationClaimAck::NotFound);
    };
    let states = load_materialization_states(transaction, &command.materialization_job_id)?;
    if !materialization_graph_state_is_canonical(transaction, project_uuid, &stored, &states)? {
        return Ok(MaterializationClaimAck::Conflict);
    }
    let latest = states
        .last()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if matches!(
        latest.state.as_str(),
        "ready" | "failed_embedding" | "failed_index" | "canceled_retention"
    ) {
        return Ok(MaterializationClaimAck::Terminal);
    }
    let claim_event =
        load_materialization_state_by_event(transaction, command.claimed_state_event_id)?;
    let orphan_event =
        load_materialization_state_by_event(transaction, command.orphaned_state_event_id)?;
    if let Some(event) = claim_event.as_ref() {
        if !claim_event_matches(event, process_instance_id, command)?
            || !claim_matches_current(event, &stored)?
            || match orphan_event.as_ref() {
                Some(orphan) => !orphan_event_matches(orphan, event, process_instance_id, command)?,
                None => false,
            }
        {
            return Ok(MaterializationClaimAck::Conflict);
        }
        let Some((pre_generation, pre_hash)) =
            pre_claim_materialization_hash(&stored, &states, command.claimed_state_event_id)?
        else {
            return Ok(MaterializationClaimAck::Conflict);
        };
        if command.expected_attempt_generation != pre_generation
            || command.expected_canonical_payload_hash != pre_hash
        {
            return Ok(MaterializationClaimAck::Conflict);
        }
        return Ok(MaterializationClaimAck::AlreadyApplied(
            materialization_lease(&stored, event)?,
        ));
    }
    if orphan_event.is_some() {
        return Ok(MaterializationClaimAck::Conflict);
    }
    if stored.attempt_generation != command.expected_attempt_generation
        || stored.canonical_payload_hash != command.expected_canonical_payload_hash
    {
        return Ok(MaterializationClaimAck::Stale);
    }
    if command.observed_at_unix_ms < latest.created_at_unix_ms {
        return Ok(MaterializationClaimAck::Conflict);
    }
    if stored.next_eligible_at_unix_ms > command.observed_at_unix_ms {
        return Ok(MaterializationClaimAck::NotEligible {
            next_eligible_at_unix_ms: stored.next_eligible_at_unix_ms,
        });
    }
    let Some(cache) = authoritative_materialization_cache(transaction, project_uuid, &stored)?
    else {
        return Ok(MaterializationClaimAck::CacheNotReady);
    };
    let vector_space_id = VectorSpaceId::new(stored.vector_space_id.clone())
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if !matches!(
        resolve_active_generation(transaction, &vector_space_id)?,
        ActiveGenerationResolution::Active(_)
    ) {
        return Ok(MaterializationClaimAck::IndexNotReady);
    }
    let reclaimed = match materialization_current_lease(&stored)? {
        None => None,
        Some((owner, token, expires_at)) => {
            if token == command.lease_token {
                return Ok(MaterializationClaimAck::Conflict);
            }
            let owner_status = verified_process_status_at(
                transaction,
                project_uuid,
                owner,
                command.observed_at_unix_ms,
            )?;
            if expires_at > command.observed_at_unix_ms && owner_status == ProcessStatusAt::Live {
                return Ok(MaterializationClaimAck::LeaseHeld {
                    lease_expires_at_unix_ms: expires_at,
                });
            }
            if owner_status == ProcessStatusAt::Invalid {
                return Ok(MaterializationClaimAck::Conflict);
            }
            Some((token, expires_at))
        }
    };
    if stored.attempt_count >= MATERIALIZATION_ATTEMPT_MAX {
        let terminal = terminalize_materialization_attempt_limit(
            transaction,
            project_uuid,
            process_instance_id,
            command,
            &stored,
            &cache,
            reclaimed,
        )?;
        return Ok(MaterializationClaimAck::AttemptLimitTerminal(terminal));
    }
    let attempt_generation = stored
        .attempt_generation
        .checked_add(1)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let attempt_count = stored
        .attempt_count
        .checked_add(1)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if state_transition_exists(
        transaction,
        &command.materialization_job_id,
        attempt_generation,
        "claimed",
    )? || (reclaimed.is_some()
        && state_transition_exists(
            transaction,
            &command.materialization_job_id,
            stored.attempt_generation,
            "orphaned_in_flight",
        )?)
    {
        return Ok(MaterializationClaimAck::Conflict);
    }
    if let Some((old_token, old_expiry)) = reclaimed {
        let orphan_hash = materialization_state_hash(
            command.orphaned_state_event_id,
            &command.materialization_job_id,
            Some(process_instance_id),
            "orphaned_in_flight",
            stored.attempt_generation,
            None,
            Some(old_token),
            Some(old_expiry),
            stored.attempt_count,
            stored.next_eligible_at_unix_ms,
            command.observed_at_unix_ms,
        )?;
        insert_materialization_state(
            transaction,
            command.orphaned_state_event_id,
            &command.materialization_job_id,
            Some(process_instance_id),
            "orphaned_in_flight",
            stored.attempt_generation,
            None,
            Some(old_token),
            Some(old_expiry),
            stored.attempt_count,
            stored.next_eligible_at_unix_ms,
            command.observed_at_unix_ms,
            &orphan_hash,
        )?;
    }
    let new_hash = materialization_payload_hash_current(
        &stored.materialization_job_id,
        parse_uuid_v7(&stored.evidence_vector_link_id)?,
        &stored.vector_space_id,
        &stored.canonical_query_hash,
        stored.embedding_job_id.as_deref(),
        parse_optional_uuid(stored.embedding_id.as_deref())?,
        Some(process_instance_id),
        Some(command.lease_token),
        Some(command.lease_expires_at_unix_ms),
        attempt_generation,
        attempt_count,
        stored.next_eligible_at_unix_ms,
        stored.created_at_unix_ms,
    )?;
    let changed = transaction
        .execute(
            "UPDATE vector_materialization_jobs
             SET lease_owner_process_instance_id = ?1, lease_token = ?2,
                 lease_expires_at_unix_ms = ?3, attempt_generation = ?4,
                 attempt_count = ?5, canonical_payload_hash = ?6
             WHERE vector_materialization_job_id = ?7 AND canonical_payload_hash = ?8",
            params![
                process_instance_id.to_string(),
                command.lease_token.to_string(),
                command.lease_expires_at_unix_ms,
                attempt_generation,
                attempt_count,
                new_hash,
                command.materialization_job_id,
                stored.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if changed != 1 {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let claim_hash = materialization_state_hash(
        command.claimed_state_event_id,
        &command.materialization_job_id,
        Some(process_instance_id),
        "claimed",
        attempt_generation,
        None,
        Some(command.lease_token),
        Some(command.lease_expires_at_unix_ms),
        attempt_count,
        stored.next_eligible_at_unix_ms,
        command.observed_at_unix_ms,
    )?;
    insert_materialization_state(
        transaction,
        command.claimed_state_event_id,
        &command.materialization_job_id,
        Some(process_instance_id),
        "claimed",
        attempt_generation,
        None,
        Some(command.lease_token),
        Some(command.lease_expires_at_unix_ms),
        attempt_count,
        stored.next_eligible_at_unix_ms,
        command.observed_at_unix_ms,
        &claim_hash,
    )?;
    let current = load_materialization(transaction, project_uuid, &command.materialization_job_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let event = load_materialization_state_by_event(transaction, command.claimed_state_event_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let lease = materialization_lease(&current, &event)?;
    Ok(if reclaimed.is_some() {
        MaterializationClaimAck::Reclaimed(lease)
    } else {
        MaterializationClaimAck::Claimed(lease)
    })
}

fn terminalize_materialization_attempt_limit(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &MaterializationClaim,
    stored: &StoredMaterialization,
    cache: &super::vector_catalog::EmbeddingCacheSnapshot,
    reclaimed: Option<(Uuid, i64)>,
) -> Result<MaterializationSnapshot, LedgerError> {
    if let Some((old_token, old_expiry)) = reclaimed {
        let orphan_hash = materialization_state_hash(
            command.orphaned_state_event_id,
            &command.materialization_job_id,
            Some(process_instance_id),
            "orphaned_in_flight",
            stored.attempt_generation,
            None,
            Some(old_token),
            Some(old_expiry),
            stored.attempt_count,
            stored.next_eligible_at_unix_ms,
            command.observed_at_unix_ms,
        )?;
        insert_materialization_state(
            transaction,
            command.orphaned_state_event_id,
            &command.materialization_job_id,
            Some(process_instance_id),
            "orphaned_in_flight",
            stored.attempt_generation,
            None,
            Some(old_token),
            Some(old_expiry),
            stored.attempt_count,
            stored.next_eligible_at_unix_ms,
            command.observed_at_unix_ms,
            &orphan_hash,
        )?;
    }

    let new_hash = materialization_payload_hash_current(
        &stored.materialization_job_id,
        parse_uuid_v7(&stored.evidence_vector_link_id)?,
        &stored.vector_space_id,
        &stored.canonical_query_hash,
        stored.embedding_job_id.as_deref(),
        Some(cache.embedding_id),
        None,
        None,
        None,
        stored.attempt_generation,
        stored.attempt_count,
        stored.next_eligible_at_unix_ms,
        stored.created_at_unix_ms,
    )?;
    let changed = transaction
        .execute(
            "UPDATE vector_materialization_jobs
             SET embedding_id = ?1, lease_owner_process_instance_id = NULL, lease_token = NULL,
                 lease_expires_at_unix_ms = NULL, canonical_payload_hash = ?2
             WHERE vector_materialization_job_id = ?3 AND canonical_payload_hash = ?4",
            params![
                cache.embedding_id.to_string(),
                new_hash,
                command.materialization_job_id,
                stored.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if changed != 1 {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let link_hash = link_state_hash(
        command.conflict_health_event_id,
        parse_uuid_v7(&stored.evidence_vector_link_id)?,
        Some(cache.embedding_id),
        "failed_index",
        stored.attempt_generation,
        Some(MATERIALIZATION_ATTEMPT_LIMIT_CLASS),
        command.observed_at_unix_ms,
    )?;
    insert_link_state(
        transaction,
        command.conflict_health_event_id,
        parse_uuid_v7(&stored.evidence_vector_link_id)?,
        Some(cache.embedding_id),
        "failed_index",
        stored.attempt_generation,
        Some(MATERIALIZATION_ATTEMPT_LIMIT_CLASS),
        command.observed_at_unix_ms,
        &link_hash,
    )?;
    let terminal_hash = materialization_state_hash(
        command.claimed_state_event_id,
        &command.materialization_job_id,
        Some(process_instance_id),
        "failed_index",
        stored.attempt_generation,
        Some(MATERIALIZATION_ATTEMPT_LIMIT_CLASS),
        None,
        None,
        stored.attempt_count,
        stored.next_eligible_at_unix_ms,
        command.observed_at_unix_ms,
    )?;
    insert_materialization_state(
        transaction,
        command.claimed_state_event_id,
        &command.materialization_job_id,
        Some(process_instance_id),
        "failed_index",
        stored.attempt_generation,
        Some(MATERIALIZATION_ATTEMPT_LIMIT_CLASS),
        None,
        None,
        stored.attempt_count,
        stored.next_eligible_at_unix_ms,
        command.observed_at_unix_ms,
        &terminal_hash,
    )?;
    let current = load_materialization(transaction, project_uuid, &command.materialization_job_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let states = load_materialization_states(transaction, &command.materialization_job_id)?;
    if !materialization_graph_state_is_canonical(transaction, project_uuid, &current, &states)? {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    materialization_snapshot(&current)
}

pub(crate) fn complete_materialization_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &MaterializationCompletion,
) -> Result<MaterializationCompletionAck, LedgerError> {
    let Some(stored) =
        load_materialization(transaction, project_uuid, &command.materialization_job_id)?
    else {
        return Ok(MaterializationCompletionAck::NotFound);
    };
    let states = load_materialization_states(transaction, &command.materialization_job_id)?;
    if !materialization_graph_state_is_canonical(transaction, project_uuid, &stored, &states)? {
        return Ok(MaterializationCompletionAck::Conflict);
    }
    let materialization_event =
        load_materialization_state_by_event(transaction, command.materialization_state_event_id)?;
    let link_event = load_link_state_by_event(transaction, command.link_state_event_id)?;
    if materialization_event.is_some() || link_event.is_some() {
        let (Some(materialization_event), Some(link_event)) =
            (materialization_event.as_ref(), link_event.as_ref())
        else {
            return Ok(MaterializationCompletionAck::Conflict);
        };
        let Some(state) = completion_retry_state(
            transaction,
            project_uuid,
            process_instance_id,
            command,
            &stored,
            &states,
            materialization_event,
            link_event,
        )?
        else {
            return Ok(MaterializationCompletionAck::Conflict);
        };
        let pre_embedding_id = previous_link_embedding_id(
            transaction,
            &stored.evidence_vector_link_id,
            command.link_state_event_id,
        )?;
        let Some((pre_generation, pre_hash)) = pre_owned_transition_hash(
            &stored,
            &states,
            command.materialization_state_event_id,
            pre_embedding_id,
        )?
        else {
            return Ok(MaterializationCompletionAck::Conflict);
        };
        if command.expected_attempt_generation != pre_generation
            || command.expected_canonical_payload_hash != pre_hash
        {
            return Ok(MaterializationCompletionAck::Conflict);
        }
        return Ok(MaterializationCompletionAck::AlreadyApplied {
            state,
            job: materialization_snapshot(&stored)?,
        });
    }
    if stored.attempt_generation != command.expected_attempt_generation
        || stored.canonical_payload_hash != command.expected_canonical_payload_hash
    {
        return Ok(MaterializationCompletionAck::StaleLease);
    }
    let Some((owner, token, expiry)) = materialization_current_lease(&stored)? else {
        return Ok(MaterializationCompletionAck::StaleLease);
    };
    if owner != process_instance_id
        || token != command.lease_token
        || command.completed_at_unix_ms >= expiry
        || states
            .last()
            .is_none_or(|latest| command.completed_at_unix_ms < latest.created_at_unix_ms)
    {
        return Ok(MaterializationCompletionAck::StaleLease);
    }
    let Some(cache) = authoritative_materialization_cache(transaction, project_uuid, &stored)?
    else {
        return Ok(MaterializationCompletionAck::CacheNotReady);
    };
    let record_id = VectorRecordId::new(parse_uuid_v7(&stored.evidence_vector_link_id)?)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let partition_id = materialization_partition_id(transaction, &stored)?;
    let record = VectorRecord::new(
        record_id,
        cache.vector_space_id.clone(),
        partition_id,
        cache.vector.clone(),
    )
    .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let state = match upsert_active_record(transaction, &record)? {
        VectorIndexPointMutationAck::Applied => {
            let space = resolve_vector_space(transaction, project_uuid, &cache.vector_space_id)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
            match append_source_change(
                transaction,
                &SourceChangeAppend {
                    project_uuid,
                    vector_space_id: cache.vector_space_id.clone(),
                    expected_previous_source_seq: space.source_seq,
                    operation: VectorSourceChangeOperation::Insert,
                    record_id,
                    partition_id,
                    vector_checksum: cache.vector.blob().checksum().clone(),
                    created_at_unix_ms: command.completed_at_unix_ms,
                },
            )? {
                SourceChangeAppendAck::Applied { .. } => MaterializationCompletionState::Ready,
                SourceChangeAppendAck::AlreadyExists { .. }
                | SourceChangeAppendAck::AuthorityNotFound
                | SourceChangeAppendAck::Conflict => {
                    return Err(LedgerErrorClass::CorruptDatabase.into());
                }
            }
        }
        VectorIndexPointMutationAck::Missing
        | VectorIndexPointMutationAck::Unavailable
        | VectorIndexPointMutationAck::Corrupt => MaterializationCompletionState::PendingIndex,
        VectorIndexPointMutationAck::AlreadyApplied | VectorIndexPointMutationAck::Conflict => {
            return Ok(MaterializationCompletionAck::Conflict);
        }
    };
    let next_eligible = match state {
        MaterializationCompletionState::Ready => stored.next_eligible_at_unix_ms,
        MaterializationCompletionState::PendingIndex => command
            .completed_at_unix_ms
            .checked_add(MATERIALIZATION_LEASE_MILLIS)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
    };
    let embedding_id = cache.embedding_id;
    let new_hash = materialization_payload_hash_current(
        &stored.materialization_job_id,
        parse_uuid_v7(&stored.evidence_vector_link_id)?,
        &stored.vector_space_id,
        &stored.canonical_query_hash,
        stored.embedding_job_id.as_deref(),
        Some(embedding_id),
        None,
        None,
        None,
        stored.attempt_generation,
        stored.attempt_count,
        next_eligible,
        stored.created_at_unix_ms,
    )?;
    let changed = transaction
        .execute(
            "UPDATE vector_materialization_jobs
             SET embedding_id = ?1, lease_owner_process_instance_id = NULL,
                 lease_token = NULL, lease_expires_at_unix_ms = NULL,
                 next_eligible_at_unix_ms = ?2, canonical_payload_hash = ?3
             WHERE vector_materialization_job_id = ?4 AND canonical_payload_hash = ?5",
            params![
                embedding_id.to_string(),
                next_eligible,
                new_hash,
                command.materialization_job_id,
                stored.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if changed != 1 {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let link_hash = link_state_hash(
        command.link_state_event_id,
        parse_uuid_v7(&stored.evidence_vector_link_id)?,
        Some(embedding_id),
        state.as_str(),
        stored.attempt_generation,
        None,
        command.completed_at_unix_ms,
    )?;
    insert_link_state(
        transaction,
        command.link_state_event_id,
        parse_uuid_v7(&stored.evidence_vector_link_id)?,
        Some(embedding_id),
        state.as_str(),
        stored.attempt_generation,
        None,
        command.completed_at_unix_ms,
        &link_hash,
    )?;
    let materialization_hash = materialization_state_hash(
        command.materialization_state_event_id,
        &command.materialization_job_id,
        Some(process_instance_id),
        state.as_str(),
        stored.attempt_generation,
        None,
        Some(command.lease_token),
        Some(expiry),
        stored.attempt_count,
        next_eligible,
        command.completed_at_unix_ms,
    )?;
    insert_materialization_state(
        transaction,
        command.materialization_state_event_id,
        &command.materialization_job_id,
        Some(process_instance_id),
        state.as_str(),
        stored.attempt_generation,
        None,
        Some(command.lease_token),
        Some(expiry),
        stored.attempt_count,
        next_eligible,
        command.completed_at_unix_ms,
        &materialization_hash,
    )?;
    let current = load_materialization(transaction, project_uuid, &command.materialization_job_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let current_states = load_materialization_states(transaction, &command.materialization_job_id)?;
    if !materialization_graph_state_is_canonical(
        transaction,
        project_uuid,
        &current,
        &current_states,
    )? {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(MaterializationCompletionAck::Applied {
        state,
        job: materialization_snapshot(&current)?,
    })
}

#[derive(Debug)]
struct StoredLinkState {
    state_event_id: String,
    evidence_vector_link_id: String,
    embedding_id: Option<String>,
    state: String,
    attempt_generation: i64,
    stable_error_class: Option<String>,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[allow(clippy::too_many_arguments)]
fn completion_retry_state(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &MaterializationCompletion,
    stored: &StoredMaterialization,
    states: &[StoredMaterializationState],
    materialization_event: &StoredMaterializationState,
    link_event: &StoredLinkState,
) -> Result<Option<MaterializationCompletionState>, LedgerError> {
    let state = match materialization_event.state.as_str() {
        "ready" => MaterializationCompletionState::Ready,
        "pending_index" => MaterializationCompletionState::PendingIndex,
        _ => return Ok(None),
    };
    let Some(embedding_id) = parse_optional_uuid(stored.embedding_id.as_deref())? else {
        return Ok(None);
    };
    if states.last().map(|event| event.state_event_id.as_str())
        != Some(command.materialization_state_event_id.to_string().as_str())
        || !materialization_state_is_canonical(materialization_event)?
        || materialization_event.materialization_job_id != command.materialization_job_id
        || materialization_event.process_instance_id.as_deref()
            != Some(process_instance_id.to_string().as_str())
        || materialization_event.attempt_generation != command.expected_attempt_generation
        || materialization_event.attempt_generation != stored.attempt_generation
        || materialization_event.attempt_count != stored.attempt_count
        || materialization_event.stable_error_class.is_some()
        || materialization_event.created_at_unix_ms != command.completed_at_unix_ms
        || materialization_event.lease_token.as_deref()
            != Some(command.lease_token.to_string().as_str())
        || stored.lease_owner_process_instance_id.is_some()
        || stored.lease_token.is_some()
        || stored.lease_expires_at_unix_ms.is_some()
        || link_event.state_event_id != command.link_state_event_id.to_string()
        || link_event.evidence_vector_link_id != stored.evidence_vector_link_id
        || link_event.embedding_id.as_deref() != Some(embedding_id.to_string().as_str())
        || link_event.state != state.as_str()
        || link_event.attempt_generation != stored.attempt_generation
        || link_event.stable_error_class.is_some()
        || link_event.created_at_unix_ms != command.completed_at_unix_ms
        || link_event.canonical_payload_hash
            != link_state_hash(
                command.link_state_event_id,
                parse_uuid_v7(&stored.evidence_vector_link_id)?,
                Some(embedding_id),
                state.as_str(),
                stored.attempt_generation,
                None,
                command.completed_at_unix_ms,
            )?
        || !link_event_is_latest(connection, link_event)?
    {
        return Ok(None);
    }
    let Some(cache) = authoritative_materialization_cache(connection, project_uuid, stored)? else {
        return Ok(None);
    };
    if cache.embedding_id != embedding_id {
        return Ok(None);
    }
    if state == MaterializationCompletionState::Ready {
        let partition_id = materialization_partition_id(connection, stored)?;
        let record_id = VectorRecordId::new(parse_uuid_v7(&stored.evidence_vector_link_id)?)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        let record = VectorRecord::new(
            record_id,
            cache.vector_space_id.clone(),
            partition_id,
            cache.vector.clone(),
        )
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        if !active_record_matches(connection, &record)?
            || !source_change_matches_record(
                connection,
                &cache.vector_space_id,
                VectorSourceChangeOperation::Insert,
                record_id,
                partition_id,
                cache.vector.blob().checksum(),
                command.completed_at_unix_ms,
            )?
        {
            return Ok(None);
        }
    }
    Ok(Some(state))
}

fn load_link_state_by_event(
    connection: &Connection,
    event_id: Uuid,
) -> Result<Option<StoredLinkState>, LedgerError> {
    connection
        .query_row(
            "SELECT evidence_vector_link_state_event_id, evidence_vector_link_id,
                    embedding_id, state, attempt_generation, stable_error_class,
                    created_at_unix_ms, canonical_payload_hash
             FROM evidence_vector_link_state_events
             WHERE evidence_vector_link_state_event_id = ?1",
            params![event_id.to_string()],
            |row| {
                Ok(StoredLinkState {
                    state_event_id: row.get(0)?,
                    evidence_vector_link_id: row.get(1)?,
                    embedding_id: row.get(2)?,
                    state: row.get(3)?,
                    attempt_generation: row.get(4)?,
                    stable_error_class: row.get(5)?,
                    created_at_unix_ms: row.get(6)?,
                    canonical_payload_hash: row.get(7)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn link_event_is_latest(
    connection: &Connection,
    event: &StoredLinkState,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT evidence_vector_link_state_event_id
             FROM evidence_vector_link_state_events
             WHERE evidence_vector_link_id = ?1 ORDER BY event_seq DESC LIMIT 1",
            params![event.evidence_vector_link_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)
        .map(|value| value.as_deref() == Some(event.state_event_id.as_str()))
}

fn materialization_partition_id(
    connection: &Connection,
    stored: &StoredMaterialization,
) -> Result<PartitionId, LedgerError> {
    let value = connection
        .query_row(
            "SELECT partition_id FROM evidence_vector_links
             WHERE evidence_vector_link_id = ?1",
            params![stored.evidence_vector_link_id],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    PartitionId::new(value).map_err(|_| LedgerErrorClass::CorruptDatabase.into())
}

fn active_record_matches(
    connection: &Connection,
    record: &VectorRecord,
) -> Result<bool, LedgerError> {
    let manifest = match resolve_active_generation(connection, record.vector_space_id())? {
        ActiveGenerationResolution::Active(manifest) => manifest,
        ActiveGenerationResolution::Missing
        | ActiveGenerationResolution::Unavailable
        | ActiveGenerationResolution::Corrupt => return Ok(false),
    };
    let root = manifest.authority().root().as_str();
    let sql = format!("SELECT partition_id, embedding FROM \"{root}\" WHERE record_id = ?1");
    let stored = connection
        .query_row(&sql, params![record.record_id().to_string()], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?))
        })
        .optional()
        .map_err(database_error)?;
    Ok(stored.is_some_and(|(partition, bytes)| {
        partition == record.partition_id().value()
            && bytes == record.vector().blob().native_endian_bytes()
    }))
}

fn source_change_matches_record(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
    operation: VectorSourceChangeOperation,
    record_id: VectorRecordId,
    partition_id: PartitionId,
    checksum: &crate::vector::VectorChecksum,
    created_at_unix_ms: i64,
) -> Result<bool, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT source_seq, canonical_payload_hash
             FROM vector_source_change_events
             WHERE vector_space_id = ?1 AND operation = ?2 AND record_id = ?3
               AND partition_id = ?4 AND vector_checksum = ?5
               AND created_at_unix_ms = ?6",
        )
        .map_err(database_error)?;
    let matches = statement
        .query_map(
            params![
                vector_space_id.as_str(),
                operation.as_str(),
                record_id.to_string(),
                partition_id.value(),
                checksum.as_str(),
                created_at_unix_ms,
            ],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    if matches.len() != 1 {
        return Ok(false);
    }
    let (source_seq, payload_hash) = &matches[0];
    Ok(*payload_hash
        == vector_source_change_payload_hash(
            vector_space_id,
            *source_seq,
            operation,
            record_id,
            partition_id,
            checksum,
            created_at_unix_ms,
        )?)
}

#[allow(clippy::too_many_arguments)]
fn insert_link_state(
    connection: &Connection,
    event_id: Uuid,
    evidence_vector_link_id: Uuid,
    embedding_id: Option<Uuid>,
    state: &str,
    generation: i64,
    stable_error_class: Option<&str>,
    created_at_unix_ms: i64,
    canonical_payload_hash: &str,
) -> Result<(), LedgerError> {
    connection
        .execute(
            "INSERT INTO evidence_vector_link_state_events (
                evidence_vector_link_state_event_id, evidence_vector_link_id,
                embedding_id, state, attempt_generation, stable_error_class,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                event_id.to_string(),
                evidence_vector_link_id.to_string(),
                embedding_id.map(|value| value.to_string()),
                state,
                generation,
                stable_error_class,
                created_at_unix_ms,
                canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    Ok(())
}

pub(crate) fn resolve_materialization_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &MaterializationResolution,
) -> Result<MaterializationResolutionAck, LedgerError> {
    let Some(stored) =
        load_materialization(transaction, project_uuid, &command.materialization_job_id)?
    else {
        return Ok(MaterializationResolutionAck::NotFound);
    };
    let states = load_materialization_states(transaction, &command.materialization_job_id)?;
    if !materialization_graph_state_is_canonical(transaction, project_uuid, &stored, &states)? {
        return Ok(MaterializationResolutionAck::Conflict);
    }
    let resolved_state = match command.kind {
        MaterializationResolutionKind::Released => MaterializationResolvedState::Released,
        MaterializationResolutionKind::RetryScheduled { .. } => {
            MaterializationResolvedState::RetryScheduled
        }
    };
    let stable_error = match &command.kind {
        MaterializationResolutionKind::Released => None,
        MaterializationResolutionKind::RetryScheduled {
            stable_error_class, ..
        } => Some(stable_error_class.as_str()),
    };
    let next_eligible = match &command.kind {
        MaterializationResolutionKind::Released => command.resolved_at_unix_ms,
        MaterializationResolutionKind::RetryScheduled {
            next_eligible_at_unix_ms,
            ..
        } => *next_eligible_at_unix_ms,
    };
    if let Some(event) =
        load_materialization_state_by_event(transaction, command.materialization_state_event_id)?
    {
        if states.last().map(|state| state.state_event_id.as_str())
            != Some(command.materialization_state_event_id.to_string().as_str())
            || !materialization_state_is_canonical(&event)?
            || event.materialization_job_id != command.materialization_job_id
            || event.process_instance_id.as_deref()
                != Some(process_instance_id.to_string().as_str())
            || event.state != resolved_state.as_str()
            || event.attempt_generation != command.expected_attempt_generation
            || event.attempt_generation != stored.attempt_generation
            || event.attempt_count != stored.attempt_count
            || event.stable_error_class.as_deref() != stable_error
            || event.lease_token.as_deref() != Some(command.lease_token.to_string().as_str())
            || event.next_eligible_at_unix_ms != next_eligible
            || event.created_at_unix_ms != command.resolved_at_unix_ms
            || stored.lease_owner_process_instance_id.is_some()
            || stored.lease_token.is_some()
            || stored.lease_expires_at_unix_ms.is_some()
            || stored.next_eligible_at_unix_ms != next_eligible
        {
            return Ok(MaterializationResolutionAck::Conflict);
        }
        let Some((pre_generation, pre_hash)) = pre_owned_transition_hash(
            &stored,
            &states,
            command.materialization_state_event_id,
            parse_optional_uuid(stored.embedding_id.as_deref())?,
        )?
        else {
            return Ok(MaterializationResolutionAck::Conflict);
        };
        if command.expected_attempt_generation != pre_generation
            || command.expected_canonical_payload_hash != pre_hash
        {
            return Ok(MaterializationResolutionAck::Conflict);
        }
        return Ok(MaterializationResolutionAck::AlreadyApplied {
            state: resolved_state,
            job: materialization_snapshot(&stored)?,
            state_event_hash: event.canonical_payload_hash,
        });
    }
    if stored.attempt_generation != command.expected_attempt_generation
        || stored.canonical_payload_hash != command.expected_canonical_payload_hash
    {
        return Ok(MaterializationResolutionAck::StaleLease);
    }
    let Some((owner, token, expiry)) = materialization_current_lease(&stored)? else {
        return Ok(MaterializationResolutionAck::StaleLease);
    };
    if owner != process_instance_id
        || token != command.lease_token
        || command.resolved_at_unix_ms >= expiry
        || states
            .last()
            .is_none_or(|latest| command.resolved_at_unix_ms < latest.created_at_unix_ms)
    {
        return Ok(MaterializationResolutionAck::StaleLease);
    }
    let new_hash = materialization_payload_hash_current(
        &stored.materialization_job_id,
        parse_uuid_v7(&stored.evidence_vector_link_id)?,
        &stored.vector_space_id,
        &stored.canonical_query_hash,
        stored.embedding_job_id.as_deref(),
        parse_optional_uuid(stored.embedding_id.as_deref())?,
        None,
        None,
        None,
        stored.attempt_generation,
        stored.attempt_count,
        next_eligible,
        stored.created_at_unix_ms,
    )?;
    let changed = transaction
        .execute(
            "UPDATE vector_materialization_jobs
             SET lease_owner_process_instance_id = NULL, lease_token = NULL,
                 lease_expires_at_unix_ms = NULL, next_eligible_at_unix_ms = ?1,
                 canonical_payload_hash = ?2
             WHERE vector_materialization_job_id = ?3 AND canonical_payload_hash = ?4",
            params![
                next_eligible,
                new_hash,
                command.materialization_job_id,
                stored.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if changed != 1 {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let event_hash = materialization_state_hash(
        command.materialization_state_event_id,
        &command.materialization_job_id,
        Some(process_instance_id),
        resolved_state.as_str(),
        stored.attempt_generation,
        stable_error,
        Some(command.lease_token),
        Some(expiry),
        stored.attempt_count,
        next_eligible,
        command.resolved_at_unix_ms,
    )?;
    insert_materialization_state(
        transaction,
        command.materialization_state_event_id,
        &command.materialization_job_id,
        Some(process_instance_id),
        resolved_state.as_str(),
        stored.attempt_generation,
        stable_error,
        Some(command.lease_token),
        Some(expiry),
        stored.attempt_count,
        next_eligible,
        command.resolved_at_unix_ms,
        &event_hash,
    )?;
    let current = load_materialization(transaction, project_uuid, &command.materialization_job_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let current_states = load_materialization_states(transaction, &command.materialization_job_id)?;
    if !materialization_graph_state_is_canonical(
        transaction,
        project_uuid,
        &current,
        &current_states,
    )? {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(MaterializationResolutionAck::Applied {
        state: resolved_state,
        job: materialization_snapshot(&current)?,
        state_event_hash: event_hash,
    })
}

pub(crate) fn delete_materialization_for_retention_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &MaterializationRetentionDelete,
) -> Result<MaterializationRetentionAck, LedgerError> {
    let mut retired_spaces = BTreeSet::new();
    delete_materialization_for_retention_with_retired_spaces_in_transaction(
        transaction,
        project_uuid,
        process_instance_id,
        command,
        &mut retired_spaces,
    )
}

pub(crate) fn delete_materialization_for_retention_with_retired_spaces_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &MaterializationRetentionDelete,
    retired_spaces: &mut BTreeSet<VectorSpaceId>,
) -> Result<MaterializationRetentionAck, LedgerError> {
    retire_materialization_in_transaction(
        transaction,
        project_uuid,
        process_instance_id,
        command,
        retired_spaces,
        false,
    )
}

/// Remove a retained source from search while preserving its decision-referenced graph.
pub(crate) fn deindex_materialization_for_retention_with_retired_spaces_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &MaterializationRetentionDelete,
    retired_spaces: &mut BTreeSet<VectorSpaceId>,
) -> Result<MaterializationRetentionAck, LedgerError> {
    retire_materialization_in_transaction(
        transaction,
        project_uuid,
        process_instance_id,
        command,
        retired_spaces,
        true,
    )
}

fn retire_materialization_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &MaterializationRetentionDelete,
    retired_spaces: &mut BTreeSet<VectorSpaceId>,
    preserve_relational_graph: bool,
) -> Result<MaterializationRetentionAck, LedgerError> {
    let Some(stored) =
        load_materialization(transaction, project_uuid, &command.materialization_job_id)?
    else {
        return Ok(MaterializationRetentionAck::AlreadyAbsent);
    };
    let states = load_materialization_states(transaction, &command.materialization_job_id)?;
    if !materialization_graph_state_is_canonical(transaction, project_uuid, &stored, &states)? {
        return Ok(MaterializationRetentionAck::Conflict);
    }
    if stored.attempt_generation != command.expected_attempt_generation
        || stored.canonical_payload_hash != command.expected_canonical_payload_hash
    {
        return Ok(MaterializationRetentionAck::Stale);
    }
    if command.deleted_at_unix_ms
        < states
            .last()
            .map(|state| state.created_at_unix_ms)
            .unwrap_or(stored.created_at_unix_ms)
        || load_materialization_state_by_event(transaction, command.materialization_state_event_id)?
            .is_some()
        || load_link_state_by_event(transaction, command.link_state_event_id)?.is_some()
    {
        return Ok(MaterializationRetentionAck::Conflict);
    }
    let latest = states
        .last()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if latest.state == "ready" {
        let cache = authoritative_materialization_cache(transaction, project_uuid, &stored)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        let record_id = VectorRecordId::new(parse_uuid_v7(&stored.evidence_vector_link_id)?)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        let partition_id = materialization_partition_id(transaction, &stored)?;
        let source_already_deindexed = retained_source_is_already_deindexed(
            transaction,
            &stored,
            &cache.vector_space_id,
            record_id,
            partition_id,
            cache.vector.blob().checksum(),
        )?;
        if source_already_deindexed
            && let Some(manifest) =
                current_generation_manifest(transaction, &cache.vector_space_id)?
        {
            match manifest.state() {
                VectorIndexManifestState::Active => {
                    match delete_active_record(transaction, &cache.vector_space_id, record_id)? {
                        VectorIndexPointMutationAck::AlreadyApplied => {}
                        VectorIndexPointMutationAck::Applied
                        | VectorIndexPointMutationAck::Missing
                        | VectorIndexPointMutationAck::Unavailable
                        | VectorIndexPointMutationAck::Corrupt
                        | VectorIndexPointMutationAck::Conflict => {
                            return Ok(MaterializationRetentionAck::Conflict);
                        }
                    }
                }
                VectorIndexManifestState::Unavailable
                | VectorIndexManifestState::Retired
                | VectorIndexManifestState::Dropped => {}
                VectorIndexManifestState::Building | VectorIndexManifestState::Corrupt => {
                    return Ok(MaterializationRetentionAck::Conflict);
                }
            }
        }
        if !source_already_deindexed && !retired_spaces.contains(&cache.vector_space_id) {
            let Some(manifest) = current_generation_manifest(transaction, &cache.vector_space_id)?
            else {
                return Ok(MaterializationRetentionAck::Conflict);
            };
            match manifest.state() {
                VectorIndexManifestState::Active => {
                    match delete_active_record(transaction, &cache.vector_space_id, record_id)? {
                        VectorIndexPointMutationAck::Applied => {}
                        VectorIndexPointMutationAck::Missing
                        | VectorIndexPointMutationAck::Unavailable
                        | VectorIndexPointMutationAck::AlreadyApplied => {
                            return Ok(MaterializationRetentionAck::IndexUnavailable {
                                vector_space_id: cache.vector_space_id,
                                expected_generation: manifest.generation(),
                                expected_manifest_hash: manifest
                                    .canonical_payload_hash()
                                    .to_string(),
                            });
                        }
                        VectorIndexPointMutationAck::Corrupt
                        | VectorIndexPointMutationAck::Conflict => {
                            return Ok(MaterializationRetentionAck::Conflict);
                        }
                    }
                }
                VectorIndexManifestState::Unavailable => {
                    match retire_current_generation_for_retention(
                        transaction,
                        &cache.vector_space_id,
                        manifest.generation(),
                        manifest.canonical_payload_hash(),
                        command.deleted_at_unix_ms,
                    )? {
                        GenerationRetirementAck::Applied
                        | GenerationRetirementAck::AlreadyApplied => {
                            retired_spaces.insert(cache.vector_space_id.clone());
                        }
                        GenerationRetirementAck::Missing | GenerationRetirementAck::Stale => {
                            return Ok(MaterializationRetentionAck::Stale);
                        }
                        GenerationRetirementAck::Conflict => {
                            return Ok(MaterializationRetentionAck::Conflict);
                        }
                    }
                }
                VectorIndexManifestState::Building
                | VectorIndexManifestState::Corrupt
                | VectorIndexManifestState::Retired
                | VectorIndexManifestState::Dropped => {
                    return Ok(MaterializationRetentionAck::Conflict);
                }
            }
        }
        if !source_already_deindexed {
            let space = resolve_vector_space(transaction, project_uuid, &cache.vector_space_id)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
            match append_source_change(
                transaction,
                &SourceChangeAppend {
                    project_uuid,
                    vector_space_id: cache.vector_space_id,
                    expected_previous_source_seq: space.source_seq,
                    operation: VectorSourceChangeOperation::Delete,
                    record_id,
                    partition_id,
                    vector_checksum: cache.vector.blob().checksum().clone(),
                    created_at_unix_ms: command.deleted_at_unix_ms,
                },
            )? {
                SourceChangeAppendAck::Applied { .. } => {}
                SourceChangeAppendAck::AlreadyExists { .. }
                | SourceChangeAppendAck::AuthorityNotFound
                | SourceChangeAppendAck::Conflict => {
                    return Err(LedgerErrorClass::CorruptDatabase.into());
                }
            }
        }
    }
    if !matches!(
        latest.state.as_str(),
        "ready" | "failed_embedding" | "failed_index" | "canceled_retention"
    ) {
        let current_lease = materialization_current_lease(&stored)?;
        let (state_actor, lease_token, lease_expiry) = current_lease
            .map(|(owner, token, expiry)| (owner, Some(token), Some(expiry)))
            .unwrap_or((process_instance_id, None, None));
        let embedding_id = parse_optional_uuid(stored.embedding_id.as_deref())?;
        let link_id = parse_uuid_v7(&stored.evidence_vector_link_id)?;
        let link_hash = link_state_hash(
            command.link_state_event_id,
            link_id,
            embedding_id,
            "canceled_retention",
            stored.attempt_generation,
            None,
            command.deleted_at_unix_ms,
        )?;
        insert_link_state(
            transaction,
            command.link_state_event_id,
            link_id,
            embedding_id,
            "canceled_retention",
            stored.attempt_generation,
            None,
            command.deleted_at_unix_ms,
            &link_hash,
        )?;
        let state_hash = materialization_state_hash(
            command.materialization_state_event_id,
            &stored.materialization_job_id,
            Some(state_actor),
            "canceled_retention",
            stored.attempt_generation,
            None,
            lease_token,
            lease_expiry,
            stored.attempt_count,
            command.deleted_at_unix_ms,
            command.deleted_at_unix_ms,
        )?;
        let new_hash = materialization_hash_with_facts(
            &stored,
            embedding_id,
            None,
            None,
            None,
            stored.attempt_generation,
            stored.attempt_count,
            command.deleted_at_unix_ms,
        )?;
        let changed = transaction
            .execute(
                "UPDATE vector_materialization_jobs
                 SET lease_owner_process_instance_id = NULL, lease_token = NULL,
                     lease_expires_at_unix_ms = NULL, next_eligible_at_unix_ms = ?1,
                     canonical_payload_hash = ?2
                 WHERE vector_materialization_job_id = ?3 AND canonical_payload_hash = ?4",
                params![
                    command.deleted_at_unix_ms,
                    new_hash,
                    stored.materialization_job_id,
                    stored.canonical_payload_hash,
                ],
            )
            .map_err(database_error)?;
        if changed != 1 {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
        insert_materialization_state(
            transaction,
            command.materialization_state_event_id,
            &stored.materialization_job_id,
            Some(state_actor),
            "canceled_retention",
            stored.attempt_generation,
            None,
            lease_token,
            lease_expiry,
            stored.attempt_count,
            command.deleted_at_unix_ms,
            command.deleted_at_unix_ms,
            &state_hash,
        )?;
        let canceled =
            load_materialization(transaction, project_uuid, &stored.materialization_job_id)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        let canceled_states =
            load_materialization_states(transaction, &stored.materialization_job_id)?;
        if !materialization_graph_state_is_canonical(
            transaction,
            project_uuid,
            &canceled,
            &canceled_states,
        )? {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }
    if preserve_relational_graph {
        return Ok(MaterializationRetentionAck::Deleted);
    }
    let outcome_id = transaction
        .query_row(
            "SELECT vectorization_outcome_id FROM evidence_vector_links
             WHERE evidence_vector_link_id = ?1",
            params![stored.evidence_vector_link_id],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?;
    let deleted = transaction
        .execute(
            "DELETE FROM evidence_vector_links WHERE evidence_vector_link_id = ?1",
            params![stored.evidence_vector_link_id],
        )
        .map_err(database_error)?;
    if deleted != 1 {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let deleted_outcome = transaction
        .execute(
            "DELETE FROM vectorization_outcomes
             WHERE vectorization_outcome_id = ?1
               AND NOT EXISTS (
                   SELECT 1 FROM evidence_vector_links
                   WHERE vectorization_outcome_id = ?1
               )",
            params![outcome_id],
        )
        .map_err(database_error)?;
    if deleted_outcome != 1 {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(MaterializationRetentionAck::Deleted)
}

fn retained_source_is_already_deindexed(
    transaction: &Connection,
    stored: &StoredMaterialization,
    vector_space_id: &VectorSpaceId,
    record_id: VectorRecordId,
    partition_id: PartitionId,
    vector_checksum: &crate::vector::VectorChecksum,
) -> Result<bool, LedgerError> {
    let marked = transaction
        .query_row(
            "SELECT EXISTS(
                 SELECT 1
                 FROM evidence_vector_links AS link
                 JOIN decision_retiring_anchors AS marker
                   ON marker.anchor_id = link.anchor_id
                 WHERE link.evidence_vector_link_id = ?1
             )",
            params![stored.evidence_vector_link_id],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)?;
    if !marked {
        return Ok(false);
    }
    let source_delete = transaction
        .query_row(
            "SELECT source_seq, operation, partition_id, vector_checksum,
                    created_at_unix_ms, canonical_payload_hash
             FROM vector_source_change_events
             WHERE vector_space_id = ?1 AND record_id = ?2
             ORDER BY source_seq DESC LIMIT 1",
            params![vector_space_id.as_str(), record_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, String>(5)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((source_seq, operation, stored_partition, checksum, created_at, payload_hash)) =
        source_delete
    else {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    };
    let expected_hash = vector_source_change_payload_hash(
        vector_space_id,
        source_seq,
        VectorSourceChangeOperation::Delete,
        record_id,
        partition_id,
        vector_checksum,
        created_at,
    )?;
    if operation != VectorSourceChangeOperation::Delete.as_str()
        || stored_partition != partition_id.value()
        || checksum != vector_checksum.as_str()
        || payload_hash != expected_hash
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(true)
}

/// Positively verify the preserved relational graph for one marked source.
pub(crate) fn verify_retiring_anchor_deindexed_in_transaction(
    transaction: &Connection,
    project_uuid: Uuid,
    anchor_id: Uuid,
) -> Result<bool, LedgerError> {
    let jobs = transaction
        .prepare(
            "SELECT materialization.vector_materialization_job_id
             FROM vector_materialization_jobs AS materialization
             JOIN evidence_vector_links AS link
               ON link.evidence_vector_link_id = materialization.evidence_vector_link_id
             WHERE link.anchor_id = ?1
             ORDER BY materialization.vector_materialization_job_id",
        )
        .map_err(database_error)?
        .query_map([anchor_id.to_string()], |row| row.get::<_, String>(0))
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    let link_count = transaction
        .query_row(
            "SELECT count(*) FROM evidence_vector_links WHERE anchor_id = ?1",
            [anchor_id.to_string()],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    if jobs.is_empty() || usize::try_from(link_count).ok() != Some(jobs.len()) {
        return Ok(false);
    }
    for job_id in jobs {
        let Some(stored) = load_materialization(transaction, project_uuid, &job_id)? else {
            return Ok(false);
        };
        let states = load_materialization_states(transaction, &job_id)?;
        if !materialization_graph_state_is_canonical(transaction, project_uuid, &stored, &states)? {
            return Ok(false);
        }
        let Some(latest) = states.last() else {
            return Ok(false);
        };
        match latest.state.as_str() {
            "ready" => {
                let cache =
                    authoritative_materialization_cache(transaction, project_uuid, &stored)?
                        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
                let record_id =
                    VectorRecordId::new(parse_uuid_v7(&stored.evidence_vector_link_id)?)
                        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
                let partition_id = materialization_partition_id(transaction, &stored)?;
                if !retained_source_is_already_deindexed(
                    transaction,
                    &stored,
                    &cache.vector_space_id,
                    record_id,
                    partition_id,
                    cache.vector.blob().checksum(),
                )? || !retained_record_is_physically_absent(
                    transaction,
                    &cache.vector_space_id,
                    record_id,
                )? {
                    return Ok(false);
                }
            }
            "canceled_retention" | "failed_embedding" | "failed_index" => {}
            _ => return Ok(false),
        }
    }
    Ok(true)
}

fn retained_record_is_physically_absent(
    transaction: &Connection,
    vector_space_id: &VectorSpaceId,
    record_id: VectorRecordId,
) -> Result<bool, LedgerError> {
    let Some(manifest) = current_generation_manifest(transaction, vector_space_id)? else {
        return Ok(true);
    };
    match manifest.state() {
        VectorIndexManifestState::Active => {
            let sql = format!(
                "SELECT EXISTS(SELECT 1 FROM \"{}\" WHERE record_id = ?1)",
                manifest.authority().root().as_str()
            );
            transaction
                .query_row(&sql, [record_id.to_string()], |row| row.get::<_, bool>(0))
                .map(|exists| !exists)
                .map_err(database_error)
        }
        VectorIndexManifestState::Unavailable
        | VectorIndexManifestState::Retired
        | VectorIndexManifestState::Dropped => Ok(true),
        VectorIndexManifestState::Building | VectorIndexManifestState::Corrupt => Ok(false),
    }
}

pub(crate) fn propagate_materialization_failure_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &MaterializationFailurePropagation,
) -> Result<MaterializationFailurePropagationAck, LedgerError> {
    let Some(job) =
        load_verified_embedding_job(transaction, project_uuid, &command.embedding_job_id)?
    else {
        return Ok(MaterializationFailurePropagationAck::NotFound);
    };
    let Some(stable_error_class) = job.terminal_error_class.as_deref() else {
        return Ok(MaterializationFailurePropagationAck::NotTerminal);
    };
    if job.attempt_generation != command.expected_attempt_generation {
        return Ok(MaterializationFailurePropagationAck::Stale);
    }
    let terminal_at = transaction
        .query_row(
            "SELECT created_at_unix_ms FROM embedding_job_state_events
             WHERE embedding_job_id = ?1
               AND state IN ('terminal_failure', 'quarantined')
             ORDER BY event_seq DESC LIMIT 1",
            params![command.embedding_job_id],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?;
    if terminal_at.is_none_or(|terminal_at| command.propagated_at_unix_ms < terminal_at) {
        return Ok(MaterializationFailurePropagationAck::Conflict);
    }
    let desired_current_cursor = command.expected_cursor.map(|value| value.to_string());
    let job_is_at_expected_cursor = job.failure_propagation_cursor == desired_current_cursor;
    let mut statement = transaction
        .prepare(
            "SELECT embedding_job_id, evidence_vector_link_id,
                    vector_materialization_job_id
             FROM vector_materialization_jobs
             WHERE embedding_job_id = ?1
               AND evidence_vector_link_id > COALESCE(?2, '')
             ORDER BY evidence_vector_link_id LIMIT 257",
        )
        .map_err(database_error)?;
    let rows = statement
        .query_map(
            params![
                command.embedding_job_id,
                command.expected_cursor.map(|value| value.to_string()),
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    let complete = rows.len() <= 256;
    let window = &rows[..rows.len().min(256)];
    if command.items.len() != window.len()
        || command.items.iter().zip(window).any(
            |(item, (embedding_job_id, link_id, materialization_id))| {
                embedding_job_id != &command.embedding_job_id
                    || link_id != &item.evidence_vector_link_id.to_string()
                    || materialization_job_id(item.evidence_vector_link_id)
                        .map_or(true, |expected| expected != *materialization_id)
            },
        )
    {
        return Ok(MaterializationFailurePropagationAck::Conflict);
    }
    let next_cursor = command
        .items
        .last()
        .map(|item| item.evidence_vector_link_id)
        .or(command.expected_cursor);

    #[derive(Clone, Copy, PartialEq, Eq)]
    enum FailurePlan {
        ExistingInitial,
        ExistingFanout,
        Insert,
    }

    let mut plans = Vec::with_capacity(command.items.len());
    for (item, (_, _, materialization_id)) in command.items.iter().zip(window) {
        let Some(stored) = load_materialization(transaction, project_uuid, materialization_id)?
        else {
            return Ok(MaterializationFailurePropagationAck::Conflict);
        };
        let states = load_materialization_states(transaction, materialization_id)?;
        if !materialization_graph_state_is_canonical(transaction, project_uuid, &stored, &states)?
            || stored.embedding_job_id.as_deref() != Some(command.embedding_job_id.as_str())
        {
            return Ok(MaterializationFailurePropagationAck::Conflict);
        }
        let materialization_event =
            load_materialization_state_by_event(transaction, item.materialization_state_event_id)?;
        let link_event = load_link_state_by_event(transaction, item.link_state_event_id)?;
        let latest_failed = states.last().is_some_and(|state| {
            state.state == "failed_embedding"
                && state.stable_error_class.as_deref() == Some(stable_error_class)
        });
        if latest_failed {
            let (Some(materialization_event), Some(link_event)) =
                (materialization_event.as_ref(), link_event.as_ref())
            else {
                return Ok(MaterializationFailurePropagationAck::Conflict);
            };
            if states.last().map(|state| state.state_event_id.as_str())
                != Some(item.materialization_state_event_id.to_string().as_str())
                || materialization_event.state != "failed_embedding"
                || materialization_event.stable_error_class.as_deref() != Some(stable_error_class)
                || materialization_event.attempt_generation != stored.attempt_generation
                || link_event.state_event_id != item.link_state_event_id.to_string()
                || link_event.state != "failed_embedding"
                || link_event.stable_error_class.as_deref() != Some(stable_error_class)
                || link_event.attempt_generation != stored.attempt_generation
                || link_event.created_at_unix_ms != materialization_event.created_at_unix_ms
                || !link_event_is_latest(transaction, link_event)?
            {
                return Ok(MaterializationFailurePropagationAck::Conflict);
            }
            if states.len() == 1 {
                plans.push(FailurePlan::ExistingInitial);
            } else if materialization_event.created_at_unix_ms == command.propagated_at_unix_ms
                && link_event.created_at_unix_ms == command.propagated_at_unix_ms
            {
                plans.push(FailurePlan::ExistingFanout);
            } else {
                return Ok(MaterializationFailurePropagationAck::Conflict);
            }
            continue;
        }
        if materialization_event.is_some() || link_event.is_some() {
            return Ok(MaterializationFailurePropagationAck::Conflict);
        }
        if states
            .last()
            .is_none_or(|state| command.propagated_at_unix_ms < state.created_at_unix_ms)
            || materialization_current_lease(&stored)?.is_some()
            || stored.embedding_id.is_some()
        {
            return Ok(MaterializationFailurePropagationAck::Conflict);
        }
        plans.push(FailurePlan::Insert);
    }
    let first_apply = job.canonical_payload_hash == command.expected_embedding_job_hash
        && job_is_at_expected_cursor
        && !job.failure_propagation_complete;
    if first_apply && plans.contains(&FailurePlan::ExistingFanout) {
        return Ok(MaterializationFailurePropagationAck::Conflict);
    }
    if !first_apply && plans.contains(&FailurePlan::Insert) {
        return Ok(MaterializationFailurePropagationAck::Stale);
    }
    if first_apply {
        for ((item, (_, _, materialization_id)), plan) in
            command.items.iter().zip(window).zip(&plans)
        {
            if *plan != FailurePlan::Insert {
                continue;
            }
            let stored = load_materialization(transaction, project_uuid, materialization_id)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
            let link_id = item.evidence_vector_link_id;
            let link_hash = link_state_hash(
                item.link_state_event_id,
                link_id,
                None,
                "failed_embedding",
                stored.attempt_generation,
                Some(stable_error_class),
                command.propagated_at_unix_ms,
            )?;
            insert_link_state(
                transaction,
                item.link_state_event_id,
                link_id,
                None,
                "failed_embedding",
                stored.attempt_generation,
                Some(stable_error_class),
                command.propagated_at_unix_ms,
                &link_hash,
            )?;
            let state_hash = materialization_state_hash(
                item.materialization_state_event_id,
                materialization_id,
                Some(process_instance_id),
                "failed_embedding",
                stored.attempt_generation,
                Some(stable_error_class),
                None,
                None,
                stored.attempt_count,
                stored.next_eligible_at_unix_ms,
                command.propagated_at_unix_ms,
            )?;
            insert_materialization_state(
                transaction,
                item.materialization_state_event_id,
                materialization_id,
                Some(process_instance_id),
                "failed_embedding",
                stored.attempt_generation,
                Some(stable_error_class),
                None,
                None,
                stored.attempt_count,
                stored.next_eligible_at_unix_ms,
                command.propagated_at_unix_ms,
                &state_hash,
            )?;
            let states = load_materialization_states(transaction, materialization_id)?;
            if !materialization_graph_state_is_canonical(
                transaction,
                project_uuid,
                &stored,
                &states,
            )? {
                return Err(LedgerErrorClass::CorruptDatabase.into());
            }
        }
    }
    let update = EmbeddingFailurePropagationUpdate::new(
        command.embedding_job_id.clone(),
        command.expected_embedding_job_hash.clone(),
        command.expected_attempt_generation,
        command.expected_cursor,
        next_cursor,
        complete,
    )?;
    let update_ack =
        update_embedding_failure_propagation_in_transaction(transaction, project_uuid, &update)?;
    match update_ack {
        EmbeddingFailurePropagationAck::Applied(_) if first_apply => {
            Ok(MaterializationFailurePropagationAck::Applied {
                processed: command.items.len(),
                next_cursor,
                complete,
            })
        }
        EmbeddingFailurePropagationAck::AlreadyApplied(_) if !first_apply => {
            Ok(MaterializationFailurePropagationAck::AlreadyApplied {
                processed: command.items.len(),
                next_cursor,
                complete,
            })
        }
        EmbeddingFailurePropagationAck::NotFound => {
            Ok(MaterializationFailurePropagationAck::NotFound)
        }
        EmbeddingFailurePropagationAck::NotTerminal => {
            Ok(MaterializationFailurePropagationAck::NotTerminal)
        }
        EmbeddingFailurePropagationAck::Stale => Ok(MaterializationFailurePropagationAck::Stale),
        EmbeddingFailurePropagationAck::Conflict => {
            Ok(MaterializationFailurePropagationAck::Conflict)
        }
        EmbeddingFailurePropagationAck::Applied(_)
        | EmbeddingFailurePropagationAck::AlreadyApplied(_) => {
            Err(LedgerErrorClass::CorruptDatabase.into())
        }
    }
}

fn load_materialization(
    connection: &Connection,
    project_uuid: Uuid,
    requested_job_id: &str,
) -> Result<Option<StoredMaterialization>, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT vector_materialization_job_id, evidence_vector_link_id,
                    vector_space_id, canonical_query_hash, embedding_job_id,
                    embedding_id, lease_owner_process_instance_id, lease_token,
                    lease_expires_at_unix_ms, attempt_generation, attempt_count,
                    next_eligible_at_unix_ms, created_at_unix_ms, canonical_payload_hash
             FROM vector_materialization_jobs
             WHERE vector_materialization_job_id = ?1",
            params![requested_job_id],
            |row| {
                Ok(StoredMaterialization {
                    materialization_job_id: row.get(0)?,
                    evidence_vector_link_id: row.get(1)?,
                    vector_space_id: row.get(2)?,
                    canonical_query_hash: row.get(3)?,
                    embedding_job_id: row.get(4)?,
                    embedding_id: row.get(5)?,
                    lease_owner_process_instance_id: row.get(6)?,
                    lease_token: row.get(7)?,
                    lease_expires_at_unix_ms: row.get(8)?,
                    attempt_generation: row.get(9)?,
                    attempt_count: row.get(10)?,
                    next_eligible_at_unix_ms: row.get(11)?,
                    created_at_unix_ms: row.get(12)?,
                    canonical_payload_hash: row.get(13)?,
                })
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(stored) = stored else {
        return Ok(None);
    };
    if !is_sha256(&stored.materialization_job_id)
        || !is_sha256(&stored.canonical_query_hash)
        || stored
            .embedding_job_id
            .as_deref()
            .is_some_and(|value| !is_sha256(value))
        || !(0..=MATERIALIZATION_ATTEMPT_MAX).contains(&stored.attempt_generation)
        || !(0..=MATERIALIZATION_ATTEMPT_MAX).contains(&stored.attempt_count)
        || stored.created_at_unix_ms < 0
        || stored.next_eligible_at_unix_ms < stored.created_at_unix_ms
        || stored.embedding_job_id.is_none() && stored.embedding_id.is_none()
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let link_id = parse_uuid_v7(&stored.evidence_vector_link_id)?;
    if materialization_job_id(link_id)? != stored.materialization_job_id {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let vector_space_id = VectorSpaceId::new(stored.vector_space_id.clone())
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if resolve_vector_space(connection, project_uuid, &vector_space_id)?.is_none() {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    if let Some(embedding_job_id) = stored.embedding_job_id.as_deref() {
        let embedding_job =
            load_verified_embedding_job(connection, project_uuid, embedding_job_id)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        if embedding_job.embedding_job_id != embedding_job_id
            || embedding_job.vector_space_id != stored.vector_space_id
            || embedding_job.canonical_query_hash != stored.canonical_query_hash
            || embedding_job.content_hash != stored.canonical_query_hash
        {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }
    let embedding_id = parse_optional_uuid(stored.embedding_id.as_deref())?;
    if let Some(expected_embedding_id) = embedding_id {
        let cache = load_embedding_cache(
            connection,
            project_uuid,
            &vector_space_id,
            &stored.canonical_query_hash,
        )?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        if cache.embedding_id != expected_embedding_id
            || cache.vector_space_id != vector_space_id
            || cache.canonical_query_hash != stored.canonical_query_hash
            || cache.content_hash != stored.canonical_query_hash
        {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }
    let owner = parse_optional_uuid(stored.lease_owner_process_instance_id.as_deref())?;
    let token = parse_optional_uuid(stored.lease_token.as_deref())?;
    if !matches!(
        (owner, token, stored.lease_expires_at_unix_ms),
        (None, None, None) | (Some(_), Some(_), Some(_))
    ) {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let relation_matches = connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM evidence_vector_links
                WHERE evidence_vector_link_id = ?1
                  AND vector_space_id = ?2 AND canonical_query_hash = ?3
             )",
            params![
                stored.evidence_vector_link_id,
                stored.vector_space_id,
                stored.canonical_query_hash,
            ],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)?;
    let expected_hash = materialization_payload_hash_current(
        &stored.materialization_job_id,
        link_id,
        &stored.vector_space_id,
        &stored.canonical_query_hash,
        stored.embedding_job_id.as_deref(),
        embedding_id,
        owner,
        token,
        stored.lease_expires_at_unix_ms,
        stored.attempt_generation,
        stored.attempt_count,
        stored.next_eligible_at_unix_ms,
        stored.created_at_unix_ms,
    )?;
    if !relation_matches || expected_hash != stored.canonical_payload_hash {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(Some(stored))
}

/// Load one optional materialization only after its row and full state graph verify.
pub(crate) fn load_verified_materialization_job(
    connection: &Connection,
    project_uuid: Uuid,
    materialization_job_id: &str,
) -> Result<Option<MaterializationSnapshot>, LedgerError> {
    validate_distinct_uuid_v7(&[project_uuid])?;
    validate_sha256(materialization_job_id)?;
    let Some(stored) = load_materialization(connection, project_uuid, materialization_job_id)?
    else {
        return Ok(None);
    };
    let states = load_materialization_states(connection, materialization_job_id)?;
    if !materialization_graph_state_is_canonical(connection, project_uuid, &stored, &states)? {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(Some(materialization_snapshot(&stored)?))
}

/// Scan every link in stable identity order, verify it, and retain ready records only.
pub(crate) fn load_verified_vector_link_page(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
    after: Option<VectorRecordId>,
    limit: usize,
) -> Result<VerifiedVectorLinkPage, LedgerError> {
    if limit == 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let limit =
        i64::try_from(limit).map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let after = after.map(|record_id| record_id.to_string());
    let mut statement = connection
        .prepare(
            "SELECT evidence_vector_link_id FROM evidence_vector_links
             WHERE vector_space_id = ?1
               AND (?2 IS NULL OR evidence_vector_link_id > ?2)
             ORDER BY evidence_vector_link_id LIMIT ?3",
        )
        .map_err(database_error)?;
    let stored_ids = statement
        .query_map(params![vector_space_id.as_str(), after, limit], |row| {
            row.get::<_, String>(0)
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);

    let mut ready_records = Vec::new();
    let mut last_scanned_record_id = None;
    for stored_id in &stored_ids {
        let record_id = VectorRecordId::new(parse_uuid_v7(stored_id)?)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        match load_verified_vector_link_source(connection, vector_space_id, record_id)? {
            VerifiedVectorLinkSourceLoad::Verified(source) => {
                if let Some(record) = source.vector_record {
                    ready_records.push(record);
                }
            }
            VerifiedVectorLinkSourceLoad::LinkMissing
            | VerifiedVectorLinkSourceLoad::AuthorityMissing => {
                return Err(LedgerErrorClass::CorruptDatabase.into());
            }
        }
        last_scanned_record_id = Some(record_id);
    }
    Ok(VerifiedVectorLinkPage {
        ready_records,
        last_scanned_record_id,
        exhausted: stored_ids.len() < usize::try_from(limit).unwrap_or(usize::MAX),
    })
}

/// Load one link through the canonical Shadow, materialization, catalog, and cache authority.
pub(crate) fn load_verified_vector_link_source(
    connection: &Connection,
    expected_vector_space_id: &VectorSpaceId,
    record_id: VectorRecordId,
) -> Result<VerifiedVectorLinkSourceLoad, LedgerError> {
    let Some(stored) = load_stored_evidence_vector_link(connection, record_id)? else {
        return Ok(VerifiedVectorLinkSourceLoad::LinkMissing);
    };
    let link_id = parse_uuid_v7(&stored.evidence_vector_link_id)?;
    if link_id != record_id.value() || stored.vector_space_id != expected_vector_space_id.as_str() {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let shadow_attempt_id = parse_uuid_v7(&stored.shadow_attempt_id)?;
    let anchor_id = parse_uuid_v7(&stored.anchor_id)?;
    let root_uuid = parse_uuid_v7(&stored.root_uuid)?;
    let learning_generation_id = parse_uuid_v7(&stored.learning_generation_id)?;
    let evaluation_id = stored
        .evaluation_id
        .as_deref()
        .map(parse_uuid_v7)
        .transpose()?;
    let partition_id = PartitionId::new(stored.partition_id)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if stored.created_at_unix_ms < 0
        || !is_sha256(&stored.vectorization_outcome_id)
        || !is_sha256(&stored.canonical_query_hash)
        || !is_sha256(&stored.canonical_payload_hash)
        || !matches!(
            stored.quality_label.as_deref(),
            None | Some("pass" | "fail")
        )
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let expected_link_hash = hash_json(&json!({
        "evidence_vector_link_id": link_id,
        "vectorization_outcome_id": stored.vectorization_outcome_id,
        "shadow_attempt_id": shadow_attempt_id,
        "anchor_id": anchor_id,
        "root_uuid": root_uuid,
        "learning_generation_id": learning_generation_id,
        "vector_space_id": stored.vector_space_id,
        "partition_id": partition_id.value(),
        "canonical_query_hash": stored.canonical_query_hash,
        "terminal_class": stored.terminal_class,
        "evaluation_id": evaluation_id,
        "quality_label": stored.quality_label,
        "created_at_unix_ms": stored.created_at_unix_ms,
    }))?;
    if stored.canonical_payload_hash != expected_link_hash {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }

    let Some(project_uuid) = project_uuid_for_verified_space(connection, expected_vector_space_id)?
    else {
        return Ok(VerifiedVectorLinkSourceLoad::AuthorityMissing);
    };
    if !direct_vector_link_authority_exists(connection, &stored)? {
        return Ok(VerifiedVectorLinkSourceLoad::AuthorityMissing);
    }
    let Some(source) =
        load_verified_backfill_vector_source(connection, project_uuid, shadow_attempt_id)?
    else {
        return Ok(VerifiedVectorLinkSourceLoad::AuthorityMissing);
    };
    let expected_quality_label = source.evaluation.as_ref().and_then(|evaluation| {
        if !evaluation.promotion_eligible {
            return None;
        }
        match evaluation.binary_label {
            Some(crate::judge::JudgeBinaryLabelV1::Pass) => Some("pass"),
            Some(crate::judge::JudgeBinaryLabelV1::Fail) => Some("fail"),
            None => None,
        }
    });
    if source.attempt.shadow_attempt_id != shadow_attempt_id
        || source.reservation.anchor_id != anchor_id
        || source.reservation.learning_generation_id != learning_generation_id
        || source.root_uuid != root_uuid
        || source.terminal_class.as_str() != stored.terminal_class
        || source.evaluation_id != evaluation_id
        || expected_quality_label != stored.quality_label.as_deref()
        || stored.created_at_unix_ms < source.terminal_at_unix_ms
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }

    let Some(query) = verified_source_query(
        connection,
        project_uuid,
        expected_vector_space_id,
        &source,
        &stored.canonical_query_hash,
    )?
    else {
        return Ok(VerifiedVectorLinkSourceLoad::AuthorityMissing);
    };
    let Some(stored_query) = load_canonical_query(connection, &stored.canonical_query_hash)? else {
        return Ok(VerifiedVectorLinkSourceLoad::AuthorityMissing);
    };
    if stored_query.artifact != query {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }

    let expected_partition = build_routing_partition_v1(
        &source.reservation,
        &source.attempt,
        expected_vector_space_id.as_str(),
    )
    .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if !verify_historical_routing_partition(
        connection,
        partition_id,
        project_uuid,
        expected_vector_space_id,
    )? {
        return Ok(VerifiedVectorLinkSourceLoad::AuthorityMissing);
    }
    let partition_hash = connection
        .query_row(
            "SELECT partition_hash FROM routing_partitions WHERE partition_id = ?1",
            params![partition_id.value()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    if partition_hash.as_deref() != Some(expected_partition.partition_hash.as_str()) {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    verify_vectorization_outcome(connection, &stored, &source)?;

    let job_id = materialization_job_id(link_id)?;
    let Some(materialization) = load_materialization(connection, project_uuid, &job_id)? else {
        return Ok(VerifiedVectorLinkSourceLoad::AuthorityMissing);
    };
    let materialization_states = load_materialization_states(connection, &job_id)?;
    let link_states = load_link_states(connection, &stored.evidence_vector_link_id)?;
    if materialization_states.is_empty() || link_states.is_empty() {
        return Ok(VerifiedVectorLinkSourceLoad::AuthorityMissing);
    }
    if !materialization_graph_state_is_canonical(
        connection,
        project_uuid,
        &materialization,
        &materialization_states,
    )? {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let latest_link_state = link_states
        .last()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let source_retiring = connection
        .query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM decision_retiring_anchors WHERE anchor_id = ?1
             )",
            [anchor_id.to_string()],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)?;
    if source_retiring
        && !verify_retiring_anchor_deindexed_in_transaction(connection, project_uuid, anchor_id)?
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let vector_record = if latest_link_state.state == "ready" && !source_retiring {
        let embedding_id = latest_link_state
            .embedding_id
            .as_deref()
            .map(parse_uuid_v7)
            .transpose()?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        let Some(cache) = load_embedding_cache(
            connection,
            project_uuid,
            expected_vector_space_id,
            &stored.canonical_query_hash,
        )?
        else {
            return Ok(VerifiedVectorLinkSourceLoad::AuthorityMissing);
        };
        if cache.embedding_id != embedding_id {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
        Some(
            VectorRecord::new(
                record_id,
                expected_vector_space_id.clone(),
                partition_id,
                cache.vector,
            )
            .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?,
        )
    } else {
        None
    };

    Ok(VerifiedVectorLinkSourceLoad::Verified(Box::new(
        VerifiedVectorLinkSource {
            record_id,
            project_uuid,
            pool_id: source.reservation.pool_id,
            vector_space_id: expected_vector_space_id.clone(),
            partition_id,
            canonical_query_hash: stored.canonical_query_hash,
            canonical_query: query,
            partition: expected_partition.partition,
            vector_record,
            shadow_attempt_id,
            shadow_result_id: source.shadow_result_id,
            anchor_id,
            root_uuid,
            learning_generation_id,
            terminal_class: source.terminal_class,
            evaluation: source.evaluation,
            quality_label: stored.quality_label,
            vector_state: latest_link_state.state.clone(),
            created_at_unix_ms: stored.created_at_unix_ms,
            record_hash: stored.canonical_payload_hash,
        },
    )))
}

fn load_stored_evidence_vector_link(
    connection: &Connection,
    record_id: VectorRecordId,
) -> Result<Option<StoredEvidenceVectorLink>, LedgerError> {
    connection
        .query_row(
            "SELECT evidence_vector_link_id, vectorization_outcome_id, shadow_attempt_id,
                    anchor_id, root_uuid, learning_generation_id, vector_space_id,
                    partition_id, canonical_query_hash, terminal_class, evaluation_id,
                    quality_label, created_at_unix_ms, canonical_payload_hash
             FROM evidence_vector_links WHERE evidence_vector_link_id = ?1",
            params![record_id.to_string()],
            |row| {
                Ok(StoredEvidenceVectorLink {
                    evidence_vector_link_id: row.get(0)?,
                    vectorization_outcome_id: row.get(1)?,
                    shadow_attempt_id: row.get(2)?,
                    anchor_id: row.get(3)?,
                    root_uuid: row.get(4)?,
                    learning_generation_id: row.get(5)?,
                    vector_space_id: row.get(6)?,
                    partition_id: row.get(7)?,
                    canonical_query_hash: row.get(8)?,
                    terminal_class: row.get(9)?,
                    evaluation_id: row.get(10)?,
                    quality_label: row.get(11)?,
                    created_at_unix_ms: row.get(12)?,
                    canonical_payload_hash: row.get(13)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn project_uuid_for_verified_space(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
) -> Result<Option<Uuid>, LedgerError> {
    let project_uuid = connection
        .query_row(
            "SELECT project_uuid FROM vector_spaces WHERE vector_space_id = ?1",
            params![vector_space_id.as_str()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let Some(project_uuid) = project_uuid else {
        return Ok(None);
    };
    let project_uuid = parse_uuid_v7(&project_uuid)?;
    Ok(resolve_vector_space(connection, project_uuid, vector_space_id)?.map(|_| project_uuid))
}

fn direct_vector_link_authority_exists(
    connection: &Connection,
    stored: &StoredEvidenceVectorLink,
) -> Result<bool, LedgerError> {
    let required = connection
        .query_row(
            "SELECT
                EXISTS(SELECT 1 FROM vectorization_outcomes WHERE vectorization_outcome_id = ?1),
                EXISTS(SELECT 1 FROM shadow_attempts WHERE shadow_attempt_id = ?2),
                EXISTS(SELECT 1 FROM shadow_results WHERE shadow_attempt_id = ?2),
                EXISTS(SELECT 1 FROM anchors WHERE anchor_id = ?3),
                EXISTS(SELECT 1 FROM routing_partitions WHERE partition_id = ?4),
                EXISTS(SELECT 1 FROM canonical_routing_queries WHERE canonical_query_hash = ?5),
                EXISTS(SELECT 1 FROM vector_materialization_jobs
                       WHERE evidence_vector_link_id = ?6),
                EXISTS(SELECT 1 FROM evidence_vector_link_state_events
                       WHERE evidence_vector_link_id = ?6),
                EXISTS(SELECT 1 FROM vector_materialization_job_state_events AS state
                       JOIN vector_materialization_jobs AS job
                         ON job.vector_materialization_job_id = state.vector_materialization_job_id
                       WHERE job.evidence_vector_link_id = ?6)",
            params![
                stored.vectorization_outcome_id,
                stored.shadow_attempt_id,
                stored.anchor_id,
                stored.partition_id,
                stored.canonical_query_hash,
                stored.evidence_vector_link_id,
            ],
            |row| {
                (0..9)
                    .map(|index| row.get::<_, bool>(index))
                    .collect::<rusqlite::Result<Vec<_>>>()
            },
        )
        .map_err(database_error)?;
    if required.into_iter().any(|present| !present) {
        return Ok(false);
    }
    if let Some(evaluation_id) = stored.evaluation_id.as_deref() {
        return connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM evaluations WHERE evaluation_id = ?1)",
                params![evaluation_id],
                |row| row.get(0),
            )
            .map_err(database_error);
    }
    Ok(true)
}

fn verified_source_query(
    connection: &Connection,
    project_uuid: Uuid,
    vector_space_id: &VectorSpaceId,
    source: &super::shadow::VerifiedBackfillVectorSource,
    expected_query_hash: &str,
) -> Result<Option<CanonicalRoutingQueryArtifactV1>, LedgerError> {
    let ShadowVectorSourceV1::Canonicalizable { query_inputs } = &source.vector_source else {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    };
    let mapping_keys = connection
        .prepare(
            "SELECT config_generation_id, policy_version_id
             FROM pool_vector_space_mappings
             WHERE project_uuid = ?1 AND pool_id = ?2 AND vector_space_id = ?3
             ORDER BY config_generation_id, policy_version_id",
        )
        .map_err(database_error)?
        .query_map(
            params![
                project_uuid.to_string(),
                source.reservation.pool_id,
                vector_space_id.as_str(),
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    if mapping_keys.is_empty() {
        return Ok(None);
    }
    for (config_generation_id, policy_version_id) in mapping_keys {
        let key = FrozenMappingKey::new(
            project_uuid,
            config_generation_id,
            source.reservation.pool_id.clone(),
            policy_version_id,
        )
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        let mapping = resolve_frozen_mapping(connection, &key)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        if mapping.mapping.vector_space_id != *vector_space_id {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
        if let Ok(query) = build_canonical_routing_query(
            query_inputs,
            &source.routing_projection,
            &mapping.canonicalizer,
        ) && query.canonical_query_hash == expected_query_hash
        {
            return Ok(Some(query));
        }
    }
    Err(LedgerErrorClass::CorruptDatabase.into())
}

fn verify_vectorization_outcome(
    connection: &Connection,
    link: &StoredEvidenceVectorLink,
    source: &super::shadow::VerifiedBackfillVectorSource,
) -> Result<(), LedgerError> {
    let outcome = connection
        .query_row(
            "SELECT vectorization_outcome_id, shadow_attempt_id, anchor_id,
                    learning_generation_id, vector_space_id, canonical_query_hash,
                    outcome, stable_reason, created_at_unix_ms, canonical_payload_hash
             FROM vectorization_outcomes WHERE vectorization_outcome_id = ?1",
            params![link.vectorization_outcome_id],
            |row| {
                Ok(StoredVectorizationOutcome {
                    vectorization_outcome_id: row.get(0)?,
                    shadow_attempt_id: row.get(1)?,
                    anchor_id: row.get(2)?,
                    learning_generation_id: row.get(3)?,
                    vector_space_id: row.get(4)?,
                    canonical_query_hash: row.get(5)?,
                    outcome: row.get(6)?,
                    stable_reason: row.get(7)?,
                    created_at_unix_ms: row.get(8)?,
                    canonical_payload_hash: row.get(9)?,
                })
            },
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let shadow_attempt_id = parse_uuid_v7(&outcome.shadow_attempt_id)?;
    let anchor_id = parse_uuid_v7(&outcome.anchor_id)?;
    let learning_generation_id = parse_uuid_v7(&outcome.learning_generation_id)?;
    let expected_id = vectorization_outcome_id(shadow_attempt_id, &outcome.vector_space_id)?;
    let expected_hash = hash_json(&json!({
        "vectorization_outcome_id": outcome.vectorization_outcome_id,
        "shadow_attempt_id": shadow_attempt_id,
        "anchor_id": anchor_id,
        "learning_generation_id": learning_generation_id,
        "vector_space_id": outcome.vector_space_id,
        "canonical_query_hash": outcome.canonical_query_hash,
        "outcome": outcome.outcome,
        "stable_reason": outcome.stable_reason,
        "created_at_unix_ms": outcome.created_at_unix_ms,
    }))?;
    if outcome.vectorization_outcome_id != expected_id
        || outcome.vectorization_outcome_id != link.vectorization_outcome_id
        || outcome.shadow_attempt_id != link.shadow_attempt_id
        || outcome.anchor_id != link.anchor_id
        || outcome.learning_generation_id != link.learning_generation_id
        || outcome.vector_space_id != link.vector_space_id
        || outcome.canonical_query_hash.as_deref() != Some(link.canonical_query_hash.as_str())
        || outcome.outcome != "canonicalized"
        || outcome.stable_reason.is_some()
        || outcome.created_at_unix_ms != link.created_at_unix_ms
        || outcome.created_at_unix_ms < source.terminal_at_unix_ms
        || outcome.canonical_payload_hash != expected_hash
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(())
}

fn load_materialization_states(
    connection: &Connection,
    materialization_job_id: &str,
) -> Result<Vec<StoredMaterializationState>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT vector_materialization_job_state_event_id,
                    vector_materialization_job_id, process_instance_id, state,
                    attempt_generation, stable_error_class, lease_token,
                    lease_expires_at_unix_ms, attempt_count,
                    next_eligible_at_unix_ms, created_at_unix_ms,
                    canonical_payload_hash
             FROM vector_materialization_job_state_events
             WHERE vector_materialization_job_id = ?1
             ORDER BY event_seq LIMIT ?2",
        )
        .map_err(database_error)?;
    let limit = i64::try_from(MATERIALIZATION_STATE_HISTORY_MAX + 1)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let states = statement
        .query_map(
            params![materialization_job_id, limit],
            stored_materialization_state,
        )
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    if states.len() > MATERIALIZATION_STATE_HISTORY_MAX {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(states)
}

fn load_materialization_state_by_event(
    connection: &Connection,
    event_id: Uuid,
) -> Result<Option<StoredMaterializationState>, LedgerError> {
    connection
        .query_row(
            "SELECT vector_materialization_job_state_event_id,
                    vector_materialization_job_id, process_instance_id, state,
                    attempt_generation, stable_error_class, lease_token,
                    lease_expires_at_unix_ms, attempt_count,
                    next_eligible_at_unix_ms, created_at_unix_ms,
                    canonical_payload_hash
             FROM vector_materialization_job_state_events
             WHERE vector_materialization_job_state_event_id = ?1",
            params![event_id.to_string()],
            stored_materialization_state,
        )
        .optional()
        .map_err(database_error)
}

fn stored_materialization_state(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<StoredMaterializationState> {
    Ok(StoredMaterializationState {
        state_event_id: row.get(0)?,
        materialization_job_id: row.get(1)?,
        process_instance_id: row.get(2)?,
        state: row.get(3)?,
        attempt_generation: row.get(4)?,
        stable_error_class: row.get(5)?,
        lease_token: row.get(6)?,
        lease_expires_at_unix_ms: row.get(7)?,
        attempt_count: row.get(8)?,
        next_eligible_at_unix_ms: row.get(9)?,
        created_at_unix_ms: row.get(10)?,
        canonical_payload_hash: row.get(11)?,
    })
}

fn materialization_state_is_canonical(
    state: &StoredMaterializationState,
) -> Result<bool, LedgerError> {
    let event_id = parse_uuid_v7(&state.state_event_id)?;
    let actor = parse_optional_uuid(state.process_instance_id.as_deref())?;
    let token = parse_optional_uuid(state.lease_token.as_deref())?;
    if !is_sha256(&state.materialization_job_id)
        || !(0..=MATERIALIZATION_ATTEMPT_MAX).contains(&state.attempt_generation)
        || !(0..=MATERIALIZATION_ATTEMPT_MAX).contains(&state.attempt_count)
        || state.next_eligible_at_unix_ms < 0
        || state.created_at_unix_ms < 0
        || (token.is_none() != state.lease_expires_at_unix_ms.is_none())
        || (matches!(
            state.state.as_str(),
            "retry_scheduled" | "failed_embedding" | "failed_index"
        ) != state.stable_error_class.is_some())
        || (state.state == "failed_index"
            && (state.attempt_generation != MATERIALIZATION_ATTEMPT_MAX
                || state.attempt_count != MATERIALIZATION_ATTEMPT_MAX
                || state.stable_error_class.as_deref()
                    != Some(MATERIALIZATION_ATTEMPT_LIMIT_CLASS)))
        || state
            .stable_error_class
            .as_deref()
            .is_some_and(|value| !valid_stable_error_class(value))
    {
        return Ok(false);
    }
    Ok(state.canonical_payload_hash
        == materialization_state_hash(
            event_id,
            &state.materialization_job_id,
            actor,
            &state.state,
            state.attempt_generation,
            state.stable_error_class.as_deref(),
            token,
            state.lease_expires_at_unix_ms,
            state.attempt_count,
            state.next_eligible_at_unix_ms,
            state.created_at_unix_ms,
        )?)
}

fn materialization_graph_state_is_canonical(
    connection: &Connection,
    project_uuid: Uuid,
    stored: &StoredMaterialization,
    materialization_states: &[StoredMaterializationState],
) -> Result<bool, LedgerError> {
    Ok(materialization_state_chain_is_canonical(
        connection,
        project_uuid,
        stored,
        materialization_states,
    )? && link_state_chain_is_canonical(connection, stored, materialization_states)?)
}

fn load_link_states(
    connection: &Connection,
    evidence_vector_link_id: &str,
) -> Result<Vec<StoredLinkState>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT evidence_vector_link_state_event_id, evidence_vector_link_id,
                    embedding_id, state, attempt_generation, stable_error_class,
                    created_at_unix_ms, canonical_payload_hash
             FROM evidence_vector_link_state_events
             WHERE evidence_vector_link_id = ?1
             ORDER BY event_seq LIMIT ?2",
        )
        .map_err(database_error)?;
    let limit = i64::try_from(LINK_STATE_HISTORY_MAX + 1)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let states = statement
        .query_map(params![evidence_vector_link_id, limit], |row| {
            Ok(StoredLinkState {
                state_event_id: row.get(0)?,
                evidence_vector_link_id: row.get(1)?,
                embedding_id: row.get(2)?,
                state: row.get(3)?,
                attempt_generation: row.get(4)?,
                stable_error_class: row.get(5)?,
                created_at_unix_ms: row.get(6)?,
                canonical_payload_hash: row.get(7)?,
            })
        })
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    if states.len() > LINK_STATE_HISTORY_MAX {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(states)
}

fn link_state_is_canonical(state: &StoredLinkState) -> Result<bool, LedgerError> {
    let event_id = parse_uuid_v7(&state.state_event_id)?;
    let link_id = parse_uuid_v7(&state.evidence_vector_link_id)?;
    let embedding_id = parse_optional_uuid(state.embedding_id.as_deref())?;
    if !(0..=MATERIALIZATION_ATTEMPT_MAX).contains(&state.attempt_generation)
        || state.created_at_unix_ms < 0
        || (state.state == "pending_embedding"
            && (embedding_id.is_some() || state.stable_error_class.is_some()))
        || (matches!(state.state.as_str(), "pending_index" | "ready")
            && (embedding_id.is_none() || state.stable_error_class.is_some()))
        || (state.state == "failed_embedding"
            && (embedding_id.is_some()
                || state
                    .stable_error_class
                    .as_deref()
                    .is_none_or(|value| !valid_stable_error_class(value))))
        || (state.state == "failed_index"
            && (embedding_id.is_none()
                || state.attempt_generation != MATERIALIZATION_ATTEMPT_MAX
                || state.stable_error_class.as_deref()
                    != Some(MATERIALIZATION_ATTEMPT_LIMIT_CLASS)))
        || (state.state == "canceled_retention" && state.stable_error_class.is_some())
        || !matches!(
            state.state.as_str(),
            "pending_embedding"
                | "pending_index"
                | "ready"
                | "failed_embedding"
                | "failed_index"
                | "canceled_retention"
        )
    {
        return Ok(false);
    }
    Ok(state.canonical_payload_hash
        == link_state_hash(
            event_id,
            link_id,
            embedding_id,
            &state.state,
            state.attempt_generation,
            state.stable_error_class.as_deref(),
            state.created_at_unix_ms,
        )?)
}

fn link_state_chain_is_canonical(
    connection: &Connection,
    stored: &StoredMaterialization,
    materialization_states: &[StoredMaterializationState],
) -> Result<bool, LedgerError> {
    let states = load_link_states(connection, &stored.evidence_vector_link_id)?;
    let Some(first) = states.first() else {
        return Ok(false);
    };
    if !link_state_is_canonical(first)?
        || first.evidence_vector_link_id != stored.evidence_vector_link_id
        || first.attempt_generation != 0
        || first.created_at_unix_ms != stored.created_at_unix_ms
        || !matches!(
            first.state.as_str(),
            "pending_embedding" | "pending_index" | "ready" | "failed_embedding"
        )
    {
        return Ok(false);
    }
    for link_state in &states {
        if !materialization_states.iter().any(|materialization_state| {
            materialization_state.state == link_state.state
                && materialization_state.attempt_generation == link_state.attempt_generation
                && materialization_state.created_at_unix_ms == link_state.created_at_unix_ms
                && materialization_state.stable_error_class == link_state.stable_error_class
        }) {
            return Ok(false);
        }
    }
    let mut generation = first.attempt_generation;
    let mut embedding_id = parse_optional_uuid(first.embedding_id.as_deref())?;
    let mut terminal = matches!(
        first.state.as_str(),
        "ready" | "failed_embedding" | "failed_index"
    );
    for (index, state) in states.iter().enumerate().skip(1) {
        if terminal
            || !link_state_is_canonical(state)?
            || state.evidence_vector_link_id != stored.evidence_vector_link_id
            || state.created_at_unix_ms < states[index - 1].created_at_unix_ms
            || state.attempt_generation < generation
            || state.state == "pending_embedding"
        {
            return Ok(false);
        }
        let next_embedding = parse_optional_uuid(state.embedding_id.as_deref())?;
        if let (Some(expected), Some(actual)) = (embedding_id, next_embedding)
            && expected != actual
        {
            return Ok(false);
        }
        match state.state.as_str() {
            "pending_index" => {
                let Some(next_embedding) = next_embedding else {
                    return Ok(false);
                };
                embedding_id = Some(next_embedding);
            }
            "ready" => {
                let Some(next_embedding) = next_embedding else {
                    return Ok(false);
                };
                embedding_id = Some(next_embedding);
                terminal = true;
            }
            "failed_embedding" => {
                if embedding_id.is_some() {
                    return Ok(false);
                }
                terminal = true;
            }
            "failed_index" => {
                let Some(next_embedding) = next_embedding else {
                    return Ok(false);
                };
                embedding_id = Some(next_embedding);
                terminal = true;
            }
            "canceled_retention" => terminal = true,
            "pending_embedding" => return Ok(false),
            _ => return Ok(false),
        }
        generation = state.attempt_generation;
    }
    let Some(latest_link) = states.last() else {
        return Ok(false);
    };
    let Some(latest_materialization) = materialization_states.last() else {
        return Ok(false);
    };
    let expected_link_state = match latest_materialization.state.as_str() {
        "ready" => "ready",
        "failed_embedding" => "failed_embedding",
        "failed_index" => "failed_index",
        "canceled_retention" => "canceled_retention",
        _ if stored.embedding_id.is_some() => "pending_index",
        _ => "pending_embedding",
    };
    Ok(latest_link.state == expected_link_state
        && embedding_id == parse_optional_uuid(stored.embedding_id.as_deref())?)
}

fn materialization_state_chain_is_canonical(
    connection: &Connection,
    project_uuid: Uuid,
    stored: &StoredMaterialization,
    states: &[StoredMaterializationState],
) -> Result<bool, LedgerError> {
    let Some(first) = states.first() else {
        return Ok(false);
    };
    if !materialization_state_is_canonical(first)?
        || first.materialization_job_id != stored.materialization_job_id
        || first.attempt_generation != 0
        || first.attempt_count != 0
        || first.next_eligible_at_unix_ms != stored.created_at_unix_ms
        || first.created_at_unix_ms != stored.created_at_unix_ms
        || first.lease_token.is_some()
        || first.lease_expires_at_unix_ms.is_some()
        || first.process_instance_id.is_none()
        || !matches!(
            first.state.as_str(),
            "pending_embedding" | "pending_index" | "ready" | "failed_embedding"
        )
    {
        return Ok(false);
    }
    let Some(first_actor) = first
        .process_instance_id
        .as_deref()
        .map(parse_uuid_v7)
        .transpose()?
    else {
        return Ok(false);
    };
    if verified_process_status_at(
        connection,
        project_uuid,
        first_actor,
        first.created_at_unix_ms,
    )? == ProcessStatusAt::Invalid
    {
        return Ok(false);
    }
    let mut generation = 0_i64;
    let mut attempt_count = 0_i64;
    let mut next_eligible = stored.created_at_unix_ms;
    let mut active_lease: Option<(Uuid, Uuid, i64)> = None;
    let mut orphaned_by: Option<Uuid> = None;
    let mut terminal = matches!(first.state.as_str(), "ready" | "failed_embedding");
    let mut has_embedding = matches!(first.state.as_str(), "pending_index" | "ready");

    for (index, state) in states.iter().enumerate().skip(1) {
        if !materialization_state_is_canonical(state)?
            || state.materialization_job_id != stored.materialization_job_id
            || state.created_at_unix_ms < states[index - 1].created_at_unix_ms
            || terminal
        {
            return Ok(false);
        }
        let actor = state
            .process_instance_id
            .as_deref()
            .map(parse_uuid_v7)
            .transpose()?;
        let token = state
            .lease_token
            .as_deref()
            .map(parse_uuid_v7)
            .transpose()?;
        let Some(actor) = actor else {
            return Ok(false);
        };
        if verified_process_status_at(connection, project_uuid, actor, state.created_at_unix_ms)?
            == ProcessStatusAt::Invalid
        {
            return Ok(false);
        }
        match state.state.as_str() {
            "claimed" => {
                let Some(next_generation) = generation.checked_add(1) else {
                    return Ok(false);
                };
                let Some(next_attempt_count) = attempt_count.checked_add(1) else {
                    return Ok(false);
                };
                if active_lease.is_some()
                    || token.is_none()
                    || state.lease_expires_at_unix_ms.is_none()
                    || state.attempt_generation != next_generation
                    || state.attempt_count != next_attempt_count
                    || state.next_eligible_at_unix_ms != next_eligible
                    || state
                        .lease_expires_at_unix_ms
                        .is_none_or(|expiry| expiry <= state.created_at_unix_ms)
                    || orphaned_by.is_some_and(|orphan_actor| {
                        actor != orphan_actor
                            || state.created_at_unix_ms != states[index - 1].created_at_unix_ms
                    })
                {
                    return Ok(false);
                }
                generation = next_generation;
                attempt_count = next_attempt_count;
                active_lease = Some((
                    actor,
                    token.unwrap(),
                    state.lease_expires_at_unix_ms.unwrap(),
                ));
                orphaned_by = None;
            }
            "orphaned_in_flight" => {
                let Some((_owner, active_token, active_expiry)) = active_lease else {
                    return Ok(false);
                };
                if orphaned_by.is_some()
                    || token != Some(active_token)
                    || state.lease_expires_at_unix_ms != Some(active_expiry)
                    || state.attempt_generation != generation
                    || state.attempt_count != attempt_count
                    || state.next_eligible_at_unix_ms != next_eligible
                {
                    return Ok(false);
                }
                active_lease = None;
                orphaned_by = Some(actor);
            }
            "released" | "retry_scheduled" => {
                let Some((owner, active_token, active_expiry)) = active_lease else {
                    return Ok(false);
                };
                if orphaned_by.is_some()
                    || actor != owner
                    || token != Some(active_token)
                    || state.lease_expires_at_unix_ms != Some(active_expiry)
                    || state.attempt_generation != generation
                    || state.attempt_count != attempt_count
                    || state.next_eligible_at_unix_ms < next_eligible
                {
                    return Ok(false);
                }
                next_eligible = state.next_eligible_at_unix_ms;
                active_lease = None;
            }
            "pending_index" | "ready" => {
                let Some((owner, active_token, active_expiry)) = active_lease else {
                    return Ok(false);
                };
                if orphaned_by.is_some()
                    || actor != owner
                    || token != Some(active_token)
                    || state.lease_expires_at_unix_ms != Some(active_expiry)
                    || state.attempt_generation != generation
                    || state.attempt_count != attempt_count
                    || state.next_eligible_at_unix_ms < next_eligible
                {
                    return Ok(false);
                }
                next_eligible = state.next_eligible_at_unix_ms;
                active_lease = None;
                has_embedding = true;
                terminal = state.state == "ready";
            }
            "failed_embedding" => {
                if active_lease.is_some()
                    || orphaned_by.is_some()
                    || state.attempt_generation != generation
                    || state.attempt_count != attempt_count
                    || state.next_eligible_at_unix_ms != next_eligible
                    || state.lease_token.is_some()
                    || state.lease_expires_at_unix_ms.is_some()
                {
                    return Ok(false);
                }
                terminal = true;
            }
            "failed_index" => {
                if active_lease.is_some()
                    || orphaned_by.is_some_and(|orphan_actor| orphan_actor != actor)
                    || generation != MATERIALIZATION_ATTEMPT_MAX
                    || attempt_count != MATERIALIZATION_ATTEMPT_MAX
                    || state.attempt_generation != MATERIALIZATION_ATTEMPT_MAX
                    || state.attempt_count != MATERIALIZATION_ATTEMPT_MAX
                    || state.next_eligible_at_unix_ms != next_eligible
                    || state.lease_token.is_some()
                    || state.lease_expires_at_unix_ms.is_some()
                    || state.stable_error_class.as_deref()
                        != Some(MATERIALIZATION_ATTEMPT_LIMIT_CLASS)
                {
                    return Ok(false);
                }
                orphaned_by = None;
                has_embedding = true;
                terminal = true;
            }
            "canceled_retention" => {
                if orphaned_by.is_some()
                    || state.attempt_generation != generation
                    || state.attempt_count != attempt_count
                {
                    return Ok(false);
                }
                match active_lease {
                    Some((owner, active_token, active_expiry))
                        if actor == owner
                            && token == Some(active_token)
                            && state.lease_expires_at_unix_ms == Some(active_expiry) => {}
                    None if token.is_none() && state.lease_expires_at_unix_ms.is_none() => {}
                    _ => return Ok(false),
                }
                next_eligible = state.next_eligible_at_unix_ms;
                active_lease = None;
                terminal = true;
            }
            "pending_embedding" => return Ok(false),
            _ => return Ok(false),
        }
    }
    if orphaned_by.is_some()
        || generation != stored.attempt_generation
        || attempt_count != stored.attempt_count
        || next_eligible != stored.next_eligible_at_unix_ms
    {
        return Ok(false);
    }
    let row_lease = materialization_current_lease(stored)?;
    if row_lease != active_lease {
        return Ok(false);
    }
    let latest = states.last().unwrap();
    if (latest.state == "claimed") != row_lease.is_some()
        || has_embedding != stored.embedding_id.is_some()
    {
        return Ok(false);
    }
    Ok(true)
}

fn pre_claim_materialization_hash(
    stored: &StoredMaterialization,
    states: &[StoredMaterializationState],
    claim_event_id: Uuid,
) -> Result<Option<(i64, String)>, LedgerError> {
    let Some(index) = states
        .iter()
        .position(|state| state.state_event_id == claim_event_id.to_string())
    else {
        return Ok(None);
    };
    let Some(previous) = index.checked_sub(1).and_then(|value| states.get(value)) else {
        return Ok(None);
    };
    let (owner, token, expiry, generation, count, next) = if previous.state == "orphaned_in_flight"
    {
        let Some(prior_claim) = index.checked_sub(2).and_then(|value| states.get(value)) else {
            return Ok(None);
        };
        if prior_claim.state != "claimed" {
            return Ok(None);
        }
        (
            prior_claim
                .process_instance_id
                .as_deref()
                .map(parse_uuid_v7)
                .transpose()?,
            previous
                .lease_token
                .as_deref()
                .map(parse_uuid_v7)
                .transpose()?,
            previous.lease_expires_at_unix_ms,
            previous.attempt_generation,
            previous.attempt_count,
            previous.next_eligible_at_unix_ms,
        )
    } else {
        if matches!(previous.state.as_str(), "claimed" | "orphaned_in_flight") {
            return Ok(None);
        }
        (
            None,
            None,
            None,
            previous.attempt_generation,
            previous.attempt_count,
            previous.next_eligible_at_unix_ms,
        )
    };
    let hash = materialization_hash_with_facts(
        stored,
        parse_optional_uuid(stored.embedding_id.as_deref())?,
        owner,
        token,
        expiry,
        generation,
        count,
        next,
    )?;
    Ok(Some((generation, hash)))
}

fn pre_owned_transition_hash(
    stored: &StoredMaterialization,
    states: &[StoredMaterializationState],
    transition_event_id: Uuid,
    pre_embedding_id: Option<Uuid>,
) -> Result<Option<(i64, String)>, LedgerError> {
    let Some(index) = states
        .iter()
        .position(|state| state.state_event_id == transition_event_id.to_string())
    else {
        return Ok(None);
    };
    let Some(claim) = index.checked_sub(1).and_then(|value| states.get(value)) else {
        return Ok(None);
    };
    if claim.state != "claimed" {
        return Ok(None);
    }
    let owner = claim
        .process_instance_id
        .as_deref()
        .map(parse_uuid_v7)
        .transpose()?;
    let token = claim
        .lease_token
        .as_deref()
        .map(parse_uuid_v7)
        .transpose()?;
    let hash = materialization_hash_with_facts(
        stored,
        pre_embedding_id,
        owner,
        token,
        claim.lease_expires_at_unix_ms,
        claim.attempt_generation,
        claim.attempt_count,
        claim.next_eligible_at_unix_ms,
    )?;
    Ok(Some((claim.attempt_generation, hash)))
}

#[allow(clippy::too_many_arguments)]
fn materialization_hash_with_facts(
    stored: &StoredMaterialization,
    embedding_id: Option<Uuid>,
    owner: Option<Uuid>,
    token: Option<Uuid>,
    expiry: Option<i64>,
    generation: i64,
    count: i64,
    next_eligible: i64,
) -> Result<String, LedgerError> {
    materialization_payload_hash_current(
        &stored.materialization_job_id,
        parse_uuid_v7(&stored.evidence_vector_link_id)?,
        &stored.vector_space_id,
        &stored.canonical_query_hash,
        stored.embedding_job_id.as_deref(),
        embedding_id,
        owner,
        token,
        expiry,
        generation,
        count,
        next_eligible,
        stored.created_at_unix_ms,
    )
}

fn previous_link_embedding_id(
    connection: &Connection,
    evidence_vector_link_id: &str,
    event_id: Uuid,
) -> Result<Option<Uuid>, LedgerError> {
    let target_seq = connection
        .query_row(
            "SELECT event_seq FROM evidence_vector_link_state_events
             WHERE evidence_vector_link_state_event_id = ?1
               AND evidence_vector_link_id = ?2",
            params![event_id.to_string(), evidence_vector_link_id],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?;
    let Some(target_seq) = target_seq else {
        return Ok(None);
    };
    connection
        .query_row(
            "SELECT embedding_id FROM evidence_vector_link_state_events
             WHERE evidence_vector_link_id = ?1 AND event_seq < ?2
             ORDER BY event_seq DESC LIMIT 1",
            params![evidence_vector_link_id, target_seq],
            |row| row.get::<_, Option<String>>(0),
        )
        .optional()
        .map_err(database_error)?
        .flatten()
        .as_deref()
        .map(parse_uuid_v7)
        .transpose()
}

fn authoritative_materialization_cache(
    connection: &Connection,
    project_uuid: Uuid,
    stored: &StoredMaterialization,
) -> Result<Option<super::vector_catalog::EmbeddingCacheSnapshot>, LedgerError> {
    let vector_space_id = VectorSpaceId::new(stored.vector_space_id.clone())
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let cache = load_embedding_cache(
        connection,
        project_uuid,
        &vector_space_id,
        &stored.canonical_query_hash,
    )?;
    if let (Some(expected), Some(cache)) = (stored.embedding_id.as_deref(), cache.as_ref())
        && parse_uuid_v7(expected)? != cache.embedding_id
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(cache)
}

fn materialization_current_lease(
    stored: &StoredMaterialization,
) -> Result<Option<(Uuid, Uuid, i64)>, LedgerError> {
    match (
        stored.lease_owner_process_instance_id.as_deref(),
        stored.lease_token.as_deref(),
        stored.lease_expires_at_unix_ms,
    ) {
        (None, None, None) => Ok(None),
        (Some(owner), Some(token), Some(expiry)) => {
            Ok(Some((parse_uuid_v7(owner)?, parse_uuid_v7(token)?, expiry)))
        }
        _ => Err(LedgerErrorClass::CorruptDatabase.into()),
    }
}

fn materialization_snapshot(
    stored: &StoredMaterialization,
) -> Result<MaterializationSnapshot, LedgerError> {
    Ok(MaterializationSnapshot {
        materialization_job_id: stored.materialization_job_id.clone(),
        evidence_vector_link_id: parse_uuid_v7(&stored.evidence_vector_link_id)?,
        vector_space_id: VectorSpaceId::new(stored.vector_space_id.clone())
            .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?,
        canonical_query_hash: stored.canonical_query_hash.clone(),
        embedding_job_id: stored.embedding_job_id.clone(),
        embedding_id: parse_optional_uuid(stored.embedding_id.as_deref())?,
        attempt_generation: stored.attempt_generation,
        attempt_count: stored.attempt_count,
        next_eligible_at_unix_ms: stored.next_eligible_at_unix_ms,
        created_at_unix_ms: stored.created_at_unix_ms,
        canonical_payload_hash: stored.canonical_payload_hash.clone(),
    })
}

fn materialization_lease(
    stored: &StoredMaterialization,
    event: &StoredMaterializationState,
) -> Result<MaterializationLease, LedgerError> {
    let Some((owner, token, expiry)) = materialization_current_lease(stored)? else {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    };
    Ok(MaterializationLease {
        job: materialization_snapshot(stored)?,
        lease_owner_process_instance_id: owner,
        lease_token: token,
        lease_expires_at_unix_ms: expiry,
        state_event_hash: event.canonical_payload_hash.clone(),
    })
}

fn claim_event_matches(
    event: &StoredMaterializationState,
    process_instance_id: Uuid,
    command: &MaterializationClaim,
) -> Result<bool, LedgerError> {
    Ok(materialization_state_is_canonical(event)?
        && event.materialization_job_id == command.materialization_job_id
        && event.process_instance_id.as_deref() == Some(process_instance_id.to_string().as_str())
        && event.state == "claimed"
        && event.stable_error_class.is_none()
        && event.created_at_unix_ms == command.observed_at_unix_ms
        && event.lease_token.as_deref() == Some(command.lease_token.to_string().as_str())
        && event.lease_expires_at_unix_ms == Some(command.lease_expires_at_unix_ms))
}

fn claim_matches_current(
    event: &StoredMaterializationState,
    stored: &StoredMaterialization,
) -> Result<bool, LedgerError> {
    let Some((owner, token, expiry)) = materialization_current_lease(stored)? else {
        return Ok(false);
    };
    Ok(event.attempt_generation == stored.attempt_generation
        && event.attempt_count == stored.attempt_count
        && event.next_eligible_at_unix_ms == stored.next_eligible_at_unix_ms
        && event.process_instance_id.as_deref() == Some(owner.to_string().as_str())
        && event.lease_token.as_deref() == Some(token.to_string().as_str())
        && event.lease_expires_at_unix_ms == Some(expiry))
}

fn orphan_event_matches(
    orphan: &StoredMaterializationState,
    claim: &StoredMaterializationState,
    process_instance_id: Uuid,
    command: &MaterializationClaim,
) -> Result<bool, LedgerError> {
    Ok(materialization_state_is_canonical(orphan)?
        && orphan.materialization_job_id == claim.materialization_job_id
        && orphan.process_instance_id.as_deref() == Some(process_instance_id.to_string().as_str())
        && orphan.state == "orphaned_in_flight"
        && orphan.created_at_unix_ms == command.observed_at_unix_ms
        && orphan
            .attempt_generation
            .checked_add(1)
            .is_some_and(|value| value == claim.attempt_generation)
        && orphan
            .attempt_count
            .checked_add(1)
            .is_some_and(|value| value == claim.attempt_count))
}

fn state_transition_exists(
    connection: &Connection,
    materialization_job_id: &str,
    generation: i64,
    state: &str,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM vector_materialization_job_state_events
                WHERE vector_materialization_job_id = ?1
                  AND attempt_generation = ?2 AND state = ?3
             )",
            params![materialization_job_id, generation, state],
            |row| row.get(0),
        )
        .map_err(database_error)
}

#[allow(clippy::too_many_arguments)]
fn insert_materialization_state(
    connection: &Connection,
    event_id: Uuid,
    materialization_job_id: &str,
    process_instance_id: Option<Uuid>,
    state: &str,
    generation: i64,
    stable_error_class: Option<&str>,
    lease_token: Option<Uuid>,
    lease_expires_at_unix_ms: Option<i64>,
    attempt_count: i64,
    next_eligible_at_unix_ms: i64,
    created_at_unix_ms: i64,
    canonical_payload_hash: &str,
) -> Result<(), LedgerError> {
    connection
        .execute(
            "INSERT INTO vector_materialization_job_state_events (
                vector_materialization_job_state_event_id,
                vector_materialization_job_id, process_instance_id, state,
                attempt_generation, stable_error_class, lease_token,
                lease_expires_at_unix_ms, attempt_count,
                next_eligible_at_unix_ms, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                event_id.to_string(),
                materialization_job_id,
                process_instance_id.map(|value| value.to_string()),
                state,
                generation,
                stable_error_class,
                lease_token.map(|value| value.to_string()),
                lease_expires_at_unix_ms,
                attempt_count,
                next_eligible_at_unix_ms,
                created_at_unix_ms,
                canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    Ok(())
}

#[derive(Clone, PartialEq)]
struct VectorGraphTerminal {
    shadow_attempt_id: Uuid,
    terminal_class: ShadowTerminalClass,
    evaluation_id: Option<Uuid>,
    vector_source: ShadowVectorSourceV1,
    vectorization: AtomicShadowVectorization,
    conflict_health_event_id: Uuid,
    created_at_unix_ms: i64,
}

impl VectorGraphTerminal {
    fn from_terminal(
        terminal: &ShadowTerminalRecord,
        vectorization: &AtomicShadowVectorization,
    ) -> Self {
        Self {
            shadow_attempt_id: terminal.shadow_attempt_id,
            terminal_class: terminal.terminal_class,
            evaluation_id: terminal.evaluation_id,
            vector_source: terminal.vector_source.clone(),
            vectorization: vectorization.clone(),
            conflict_health_event_id: terminal.conflict_health_event_id,
            created_at_unix_ms: terminal.created_at_unix_ms,
        }
    }
}

/// Record or verify the vector graph paired with one Shadow terminal write.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record_terminal_vector_graph(
    savepoint: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    reservation: &SampleBatchReservation,
    attempt: &ReservedShadowAttempt,
    root_uuid: Uuid,
    routing: &RouterRoutingContextProjectionV1,
    terminal: &ShadowTerminalRecord,
    shadow_applied: bool,
) -> Result<bool, LedgerError> {
    let mapping_key = FrozenMappingKey::new(
        project_uuid,
        reservation.config_generation_id.clone(),
        reservation.pool_id.clone(),
        reservation.policy_version_id.clone(),
    )
    .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let authority = resolve_frozen_pool_vector_authority(savepoint, &mapping_key)?;
    match (&terminal.vectorization, authority) {
        (ShadowVectorizationHandoff::Disabled, FrozenPoolVectorAuthority::Disabled) => {
            no_vector_graph_exists(savepoint, terminal.shadow_attempt_id)
        }
        (
            ShadowVectorizationHandoff::Atomic(vectorization),
            FrozenPoolVectorAuthority::Enabled(mapping),
        ) if vectorization.routing_projection.as_ref() == routing => {
            let mapping_authority = VectorMappingAuthority {
                project_uuid,
                config_generation_id: reservation.config_generation_id.clone(),
                pool_id: reservation.pool_id.clone(),
                mapping_policy_version_id: reservation.policy_version_id.clone(),
            };
            record_enabled_graph(
                savepoint,
                project_uuid,
                process_instance_id,
                reservation,
                attempt,
                root_uuid,
                routing,
                &VectorGraphTerminal::from_terminal(terminal, vectorization),
                &mapping,
                &mapping_authority,
                shadow_applied,
            )
        }
        (ShadowVectorizationHandoff::DeferredBackfill, _) => Ok(false),
        _ => Ok(false),
    }
}

fn no_vector_graph_exists(
    connection: &Connection,
    shadow_attempt_id: Uuid,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT NOT EXISTS(
                SELECT 1 FROM vectorization_outcomes WHERE shadow_attempt_id = ?1
             ) AND NOT EXISTS(
                SELECT 1 FROM evidence_vector_links WHERE shadow_attempt_id = ?1
             )",
            params![shadow_attempt_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn vector_graph_identity_exists(
    connection: &Connection,
    shadow_attempt_id: Uuid,
    vector_space_id: &str,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM vectorization_outcomes
                WHERE shadow_attempt_id = ?1 AND vector_space_id = ?2
             ) OR EXISTS(
                SELECT 1 FROM evidence_vector_links
                WHERE shadow_attempt_id = ?1 AND vector_space_id = ?2
             )",
            params![shadow_attempt_id.to_string(), vector_space_id],
            |row| row.get(0),
        )
        .map_err(database_error)
}

#[allow(clippy::too_many_arguments)]
fn record_enabled_graph(
    savepoint: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    reservation: &SampleBatchReservation,
    attempt: &ReservedShadowAttempt,
    root_uuid: Uuid,
    routing: &RouterRoutingContextProjectionV1,
    terminal: &VectorGraphTerminal,
    mapping: &VerifiedPoolVectorSpaceMapping,
    mapping_authority: &VectorMappingAuthority,
    shadow_applied: bool,
) -> Result<bool, LedgerError> {
    let vector_space_id = &mapping.mapping.vector_space_id;
    let outcome_id =
        vectorization_outcome_id(terminal.shadow_attempt_id, vector_space_id.as_str())?;
    let query = match &terminal.vector_source {
        ShadowVectorSourceV1::Canonicalizable { query_inputs } => {
            match build_canonical_routing_query(query_inputs, routing, &mapping.canonicalizer) {
                Ok(query) => query,
                Err(reason) => {
                    return record_noncanonical_outcome(
                        savepoint,
                        terminal,
                        reservation,
                        vector_space_id.as_str(),
                        &outcome_id,
                        reason.code(),
                        shadow_applied,
                    );
                }
            }
        }
        ShadowVectorSourceV1::Noncanonicalizable { reason } => {
            return record_noncanonical_outcome(
                savepoint,
                terminal,
                reservation,
                vector_space_id.as_str(),
                &outcome_id,
                reason.as_str(),
                shadow_applied,
            );
        }
    };
    let partition = match build_routing_partition_v1(reservation, attempt, vector_space_id.as_str())
    {
        Ok(partition) => partition,
        Err(_) => return Ok(false),
    };
    if !shadow_applied {
        return verify_canonical_graph(
            savepoint,
            process_instance_id,
            terminal,
            reservation,
            root_uuid,
            mapping,
            mapping_authority,
            &query,
            &partition,
            &outcome_id,
        );
    }

    match ensure_canonical_query(savepoint, &query, terminal.created_at_unix_ms)? {
        CanonicalQueryEnsureAck::Applied(_) | CanonicalQueryEnsureAck::AlreadyExists(_) => {}
        CanonicalQueryEnsureAck::Conflict => return Ok(false),
    }
    let partition_id = match ensure_routing_partition(
        savepoint,
        &RoutingPartitionEnsure {
            mapping: mapping_authority.clone(),
            artifact: partition.clone(),
            created_at_unix_ms: terminal.created_at_unix_ms,
        },
    )? {
        RoutingPartitionEnsureAck::Applied(snapshot)
        | RoutingPartitionEnsureAck::AlreadyExists(snapshot) => snapshot.partition_id,
        RoutingPartitionEnsureAck::MappingNotFound | RoutingPartitionEnsureAck::Conflict => {
            return Ok(false);
        }
    };
    if !insert_or_verify_outcome(
        savepoint,
        terminal,
        reservation,
        vector_space_id.as_str(),
        &outcome_id,
        Some(&query.canonical_query_hash),
        "canonicalized",
        None,
        true,
    )? {
        return Ok(false);
    }

    let Some(vectorization) = atomic_handoff(terminal) else {
        return Ok(false);
    };
    if link_identity_exists(
        savepoint,
        vectorization.evidence_vector_link_id,
        terminal.shadow_attempt_id,
        vector_space_id.as_str(),
    )? {
        return Ok(false);
    }
    let quality_label = immutable_quality_label(savepoint, terminal)?;
    let initial = initial_materialization(
        savepoint,
        project_uuid,
        process_instance_id,
        terminal,
        vector_space_id.as_str(),
        &query.canonical_query_hash,
        partition_id,
        mapping.space.source_seq,
    )?;
    let link_hash = link_payload_hash(
        vectorization.evidence_vector_link_id,
        &outcome_id,
        terminal,
        reservation,
        root_uuid,
        vector_space_id.as_str(),
        partition_id,
        &query.canonical_query_hash,
        quality_label,
    )?;
    savepoint
        .execute(
            "INSERT INTO evidence_vector_links (
                evidence_vector_link_id, vectorization_outcome_id, shadow_attempt_id,
                anchor_id, root_uuid, learning_generation_id, vector_space_id,
                partition_id, canonical_query_hash, terminal_class, evaluation_id,
                quality_label, created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)",
            params![
                vectorization.evidence_vector_link_id.to_string(),
                outcome_id,
                terminal.shadow_attempt_id.to_string(),
                reservation.anchor_id.to_string(),
                root_uuid.to_string(),
                reservation.learning_generation_id.to_string(),
                vector_space_id.as_str(),
                partition_id.value(),
                query.canonical_query_hash,
                terminal.terminal_class.as_str(),
                terminal.evaluation_id.map(|value| value.to_string()),
                quality_label,
                terminal.created_at_unix_ms,
                link_hash,
            ],
        )
        .map_err(database_error)?;
    insert_initial_materialization(
        savepoint,
        process_instance_id,
        terminal,
        vector_space_id.as_str(),
        partition_id,
        &query.canonical_query_hash,
        &initial,
    )?;
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn initial_materialization(
    savepoint: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    terminal: &VectorGraphTerminal,
    vector_space_id: &str,
    canonical_query_hash: &str,
    partition_id: PartitionId,
    expected_previous_source_seq: i64,
) -> Result<InitialMaterialization, LedgerError> {
    let vector_space_id_typed = crate::vector::VectorSpaceId::new(vector_space_id.to_string())
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if let Some(cache) = load_embedding_cache(
        savepoint,
        project_uuid,
        &vector_space_id_typed,
        canonical_query_hash,
    )? {
        let vectorization = atomic_handoff(terminal)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let record_id = VectorRecordId::new(vectorization.evidence_vector_link_id)
            .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let record = VectorRecord::new(
            record_id,
            vector_space_id_typed.clone(),
            partition_id,
            cache.vector.clone(),
        )
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        return match upsert_active_record(savepoint, &record)? {
            VectorIndexPointMutationAck::Applied => {
                match append_source_change(
                    savepoint,
                    &SourceChangeAppend {
                        project_uuid,
                        vector_space_id: vector_space_id_typed,
                        expected_previous_source_seq,
                        operation: VectorSourceChangeOperation::Insert,
                        record_id,
                        partition_id,
                        vector_checksum: cache.vector.blob().checksum().clone(),
                        created_at_unix_ms: terminal.created_at_unix_ms,
                    },
                )? {
                    SourceChangeAppendAck::Applied { .. } => Ok(InitialMaterialization::Ready {
                        embedding_id: cache.embedding_id,
                    }),
                    SourceChangeAppendAck::AlreadyExists { .. }
                    | SourceChangeAppendAck::AuthorityNotFound
                    | SourceChangeAppendAck::Conflict => {
                        Err(LedgerErrorClass::CorruptDatabase.into())
                    }
                }
            }
            VectorIndexPointMutationAck::AlreadyApplied => {
                Err(LedgerErrorClass::CorruptDatabase.into())
            }
            VectorIndexPointMutationAck::Missing
            | VectorIndexPointMutationAck::Unavailable
            | VectorIndexPointMutationAck::Corrupt => Ok(InitialMaterialization::PendingIndex {
                embedding_id: cache.embedding_id,
            }),
            VectorIndexPointMutationAck::Conflict => Err(LedgerErrorClass::CorruptDatabase.into()),
        };
    }

    let vectorization = atomic_handoff(terminal)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let create = EmbeddingJobCreate::new(
        vectorization.embedding_job_state_event_id,
        terminal.conflict_health_event_id,
        vector_space_id,
        canonical_query_hash,
        canonical_query_hash,
        terminal.created_at_unix_ms,
    )?;
    let snapshot = match create_embedding_job_in_transaction(
        savepoint,
        project_uuid,
        process_instance_id,
        &create,
    )? {
        EmbeddingJobCreateAck::Applied(snapshot)
        | EmbeddingJobCreateAck::AlreadyExists(snapshot) => snapshot,
        EmbeddingJobCreateAck::VectorSpaceNotFound
        | EmbeddingJobCreateAck::Conflict
        | EmbeddingJobCreateAck::OriginatingProcessNotLive
        | EmbeddingJobCreateAck::TransactionNotStarted => {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    };
    if let Some(stable_error_class) = snapshot.terminal_error_class {
        return Ok(InitialMaterialization::FailedEmbedding {
            embedding_job_id: snapshot.embedding_job_id,
            stable_error_class,
        });
    }
    let latest_state = savepoint
        .query_row(
            "SELECT state FROM embedding_job_state_events
             WHERE embedding_job_id = ?1 ORDER BY event_seq DESC LIMIT 1",
            params![snapshot.embedding_job_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    if latest_state.as_deref() == Some("completed") {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(InitialMaterialization::PendingEmbedding {
        embedding_job_id: snapshot.embedding_job_id,
    })
}

#[allow(clippy::too_many_arguments)]
fn insert_initial_materialization(
    savepoint: &Transaction<'_>,
    process_instance_id: Uuid,
    terminal: &VectorGraphTerminal,
    vector_space_id: &str,
    _partition_id: PartitionId,
    canonical_query_hash: &str,
    initial: &InitialMaterialization,
) -> Result<(), LedgerError> {
    let vectorization = atomic_handoff(terminal)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let materialization_job_id = materialization_job_id(vectorization.evidence_vector_link_id)?;
    let materialization_hash = materialization_payload_hash(
        &materialization_job_id,
        vectorization.evidence_vector_link_id,
        vector_space_id,
        canonical_query_hash,
        initial.embedding_job_id(),
        initial.embedding_id(),
        terminal.created_at_unix_ms,
    )?;
    savepoint
        .execute(
            "INSERT INTO vector_materialization_jobs (
                vector_materialization_job_id, evidence_vector_link_id, vector_space_id,
                canonical_query_hash, embedding_job_id, embedding_id,
                lease_owner_process_instance_id, lease_token, lease_expires_at_unix_ms,
                attempt_generation, attempt_count, next_eligible_at_unix_ms,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, NULL, NULL, NULL, 0, 0, ?7, ?7, ?8)",
            params![
                materialization_job_id,
                vectorization.evidence_vector_link_id.to_string(),
                vector_space_id,
                canonical_query_hash,
                initial.embedding_job_id(),
                initial.embedding_id().map(|value| value.to_string()),
                terminal.created_at_unix_ms,
                materialization_hash,
            ],
        )
        .map_err(database_error)?;
    let link_state_hash = link_state_payload_hash(
        vectorization.evidence_link_state_event_id,
        vectorization.evidence_vector_link_id,
        initial.embedding_id(),
        initial.state(),
        initial.stable_error_class(),
        terminal.created_at_unix_ms,
    )?;
    savepoint
        .execute(
            "INSERT INTO evidence_vector_link_state_events (
                evidence_vector_link_state_event_id, evidence_vector_link_id,
                embedding_id, state, attempt_generation, stable_error_class,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, 0, ?5, ?6, ?7)",
            params![
                vectorization.evidence_link_state_event_id.to_string(),
                vectorization.evidence_vector_link_id.to_string(),
                initial.embedding_id().map(|value| value.to_string()),
                initial.state(),
                initial.stable_error_class(),
                terminal.created_at_unix_ms,
                link_state_hash,
            ],
        )
        .map_err(database_error)?;
    let materialization_state_hash = materialization_state_payload_hash(
        vectorization.materialization_state_event_id,
        &materialization_job_id,
        process_instance_id,
        initial.state(),
        initial.stable_error_class(),
        terminal.created_at_unix_ms,
    )?;
    savepoint
        .execute(
            "INSERT INTO vector_materialization_job_state_events (
                vector_materialization_job_state_event_id, vector_materialization_job_id,
                process_instance_id, state, attempt_generation, stable_error_class,
                lease_token, lease_expires_at_unix_ms, attempt_count,
                next_eligible_at_unix_ms, created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, 0, ?5, NULL, NULL, 0, ?6, ?6, ?7)",
            params![
                vectorization.materialization_state_event_id.to_string(),
                materialization_job_id,
                process_instance_id.to_string(),
                initial.state(),
                initial.stable_error_class(),
                terminal.created_at_unix_ms,
                materialization_state_hash,
            ],
        )
        .map_err(database_error)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_canonical_graph(
    connection: &Connection,
    process_instance_id: Uuid,
    terminal: &VectorGraphTerminal,
    reservation: &SampleBatchReservation,
    root_uuid: Uuid,
    mapping: &VerifiedPoolVectorSpaceMapping,
    mapping_authority: &VectorMappingAuthority,
    query: &CanonicalRoutingQueryArtifactV1,
    partition: &RoutingPartitionArtifactV1,
    outcome_id: &str,
) -> Result<bool, LedgerError> {
    let Some(vectorization) = atomic_handoff(terminal) else {
        return Ok(false);
    };
    if load_canonical_query(connection, &query.canonical_query_hash)?
        .is_none_or(|stored| stored.artifact != *query)
    {
        return Ok(false);
    }
    let partition_id = connection
        .query_row(
            "SELECT partition_id FROM routing_partitions WHERE partition_hash = ?1",
            params![partition.partition_hash],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?
        .and_then(|value| PartitionId::new(value).ok());
    let Some(partition_id) = partition_id else {
        return Ok(false);
    };
    if load_routing_partition(connection, mapping_authority, partition_id)?
        .is_none_or(|stored| stored.artifact != *partition)
        || !insert_or_verify_outcome(
            connection,
            terminal,
            reservation,
            mapping.mapping.vector_space_id.as_str(),
            outcome_id,
            Some(&query.canonical_query_hash),
            "canonicalized",
            None,
            false,
        )?
    {
        return Ok(false);
    }
    let quality_label = immutable_quality_label(connection, terminal)?;
    let expected_link_hash = link_payload_hash(
        vectorization.evidence_vector_link_id,
        outcome_id,
        terminal,
        reservation,
        root_uuid,
        mapping.mapping.vector_space_id.as_str(),
        partition_id,
        &query.canonical_query_hash,
        quality_label,
    )?;
    let link_matches = connection
        .query_row(
            "SELECT evidence_vector_link_id, canonical_payload_hash
             FROM evidence_vector_links
             WHERE shadow_attempt_id = ?1 AND vector_space_id = ?2",
            params![
                terminal.shadow_attempt_id.to_string(),
                mapping.mapping.vector_space_id.as_str(),
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(database_error)?
        .is_some_and(|(id, hash)| {
            id == vectorization.evidence_vector_link_id.to_string() && hash == expected_link_hash
        });
    if !link_matches {
        return Ok(false);
    }
    verify_initial_states(
        connection,
        mapping.mapping.project_uuid,
        process_instance_id,
        terminal,
        mapping.mapping.vector_space_id.as_str(),
        &query.canonical_query_hash,
    )
}

fn verify_initial_states(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    terminal: &VectorGraphTerminal,
    vector_space_id: &str,
    canonical_query_hash: &str,
) -> Result<bool, LedgerError> {
    let Some(vectorization) = atomic_handoff(terminal) else {
        return Ok(false);
    };
    let link_state = connection
        .query_row(
            "SELECT embedding_id, state, attempt_generation, stable_error_class,
                    created_at_unix_ms, canonical_payload_hash
             FROM evidence_vector_link_state_events
             WHERE evidence_vector_link_state_event_id = ?1
               AND evidence_vector_link_id = ?2",
            params![
                vectorization.evidence_link_state_event_id.to_string(),
                vectorization.evidence_vector_link_id.to_string(),
            ],
            |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, String>(5)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((embedding_id, state, generation, stable_error, created_at, hash)) = link_state else {
        return Ok(false);
    };
    if generation != 0 || created_at != terminal.created_at_unix_ms {
        return Ok(false);
    }
    let parsed_embedding = embedding_id.as_deref().map(parse_uuid_v7).transpose()?;
    if hash
        != link_state_payload_hash(
            vectorization.evidence_link_state_event_id,
            vectorization.evidence_vector_link_id,
            parsed_embedding,
            &state,
            stable_error.as_deref(),
            created_at,
        )?
    {
        return Ok(false);
    }
    let materialization_job_id = materialization_job_id(vectorization.evidence_vector_link_id)?;
    if !verify_materialization_row(
        connection,
        project_uuid,
        process_instance_id,
        terminal,
        vector_space_id,
        canonical_query_hash,
        &materialization_job_id,
        &state,
        parsed_embedding,
        stable_error.as_deref(),
    )? {
        return Ok(false);
    }
    let materialization_state = connection
        .query_row(
            "SELECT process_instance_id, state, attempt_generation, stable_error_class,
                    lease_token, lease_expires_at_unix_ms, attempt_count,
                    next_eligible_at_unix_ms, created_at_unix_ms, canonical_payload_hash
             FROM vector_materialization_job_state_events
             WHERE vector_materialization_job_state_event_id = ?1
               AND vector_materialization_job_id = ?2",
            params![
                vectorization.materialization_state_event_id.to_string(),
                materialization_job_id,
            ],
            |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<i64>>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, String>(9)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((
        actor,
        mat_state,
        mat_generation,
        mat_error,
        token,
        expiry,
        count,
        next,
        at,
        mat_hash,
    )) = materialization_state
    else {
        return Ok(false);
    };
    if actor.as_deref() != Some(process_instance_id.to_string().as_str())
        || mat_state != state
        || mat_generation != 0
        || mat_error != stable_error
        || token.is_some()
        || expiry.is_some()
        || count != 0
        || next != terminal.created_at_unix_ms
        || at != terminal.created_at_unix_ms
        || mat_hash
            != materialization_state_payload_hash(
                vectorization.materialization_state_event_id,
                &materialization_job_id,
                process_instance_id,
                &state,
                stable_error.as_deref(),
                at,
            )?
    {
        return Ok(false);
    }
    Ok(true)
}

#[allow(clippy::too_many_arguments)]
fn verify_materialization_row(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    terminal: &VectorGraphTerminal,
    vector_space_id: &str,
    canonical_query_hash: &str,
    materialization_job_id: &str,
    initial_state: &str,
    initial_embedding_id: Option<Uuid>,
    initial_error: Option<&str>,
) -> Result<bool, LedgerError> {
    let Some(vectorization) = atomic_handoff(terminal) else {
        return Ok(false);
    };
    let stored = connection
        .query_row(
            "SELECT evidence_vector_link_id, vector_space_id, canonical_query_hash,
                    embedding_job_id, embedding_id, lease_owner_process_instance_id,
                    lease_token, lease_expires_at_unix_ms, attempt_generation,
                    attempt_count, next_eligible_at_unix_ms, created_at_unix_ms,
                    canonical_payload_hash
             FROM vector_materialization_jobs
             WHERE vector_materialization_job_id = ?1",
            params![materialization_job_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, Option<String>>(6)?,
                    row.get::<_, Option<i64>>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, i64>(9)?,
                    row.get::<_, i64>(10)?,
                    row.get::<_, i64>(11)?,
                    row.get::<_, String>(12)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((
        link_id,
        stored_space,
        stored_query,
        embedding_job_id,
        embedding_id,
        owner,
        token,
        expiry,
        generation,
        count,
        next_eligible,
        created_at,
        payload_hash,
    )) = stored
    else {
        return Ok(false);
    };
    let owner = owner.as_deref().map(parse_uuid_v7).transpose()?;
    let token = token.as_deref().map(parse_uuid_v7).transpose()?;
    let embedding_id = embedding_id.as_deref().map(parse_uuid_v7).transpose()?;
    let expected_payload_hash = materialization_payload_hash_current(
        materialization_job_id,
        vectorization.evidence_vector_link_id,
        vector_space_id,
        canonical_query_hash,
        embedding_job_id.as_deref(),
        embedding_id,
        owner,
        token,
        expiry,
        generation,
        count,
        next_eligible,
        created_at,
    )?;
    if link_id != vectorization.evidence_vector_link_id.to_string()
        || stored_space != vector_space_id
        || stored_query != canonical_query_hash
        || created_at != terminal.created_at_unix_ms
        || generation < 0
        || count < 0
        || next_eligible < created_at
        || !matches!(
            (owner, token, expiry),
            (None, None, None) | (Some(_), Some(_), Some(_))
        )
        || payload_hash != expected_payload_hash
    {
        return Ok(false);
    }

    match initial_state {
        "pending_embedding" | "failed_embedding" if embedding_job_id.is_some() => {}
        "pending_index" | "ready" if initial_embedding_id.is_some() => {
            if embedding_id != initial_embedding_id {
                return Ok(false);
            }
        }
        _ => return Ok(false),
    }
    if initial_state == "failed_embedding" && initial_error.is_none() {
        return Ok(false);
    }
    if let Some(job_id) = embedding_job_id {
        if !canonical_embedding_create_event_matches(
            connection,
            project_uuid,
            job_id.as_str(),
            terminal.created_at_unix_ms,
        )? || !embedding_create_event_matches(
            connection,
            vectorization.embedding_job_state_event_id,
            job_id.as_str(),
            process_instance_id,
            terminal.created_at_unix_ms,
        )? {
            return Ok(false);
        }
        let verification_event_id = loop {
            let candidate = Uuid::now_v7();
            if candidate != terminal.conflict_health_event_id {
                break candidate;
            }
        };
        let create = EmbeddingJobCreate::new(
            verification_event_id,
            terminal.conflict_health_event_id,
            vector_space_id,
            canonical_query_hash,
            canonical_query_hash,
            terminal.created_at_unix_ms,
        )?;
        if create.embedding_job_id() != job_id {
            return Ok(false);
        }
        let job_ack = create_embedding_job_in_transaction(
            connection,
            project_uuid,
            process_instance_id,
            &create,
        )?;
        let snapshot = match job_ack {
            EmbeddingJobCreateAck::AlreadyExists(snapshot) => snapshot,
            EmbeddingJobCreateAck::Applied(_)
            | EmbeddingJobCreateAck::VectorSpaceNotFound
            | EmbeddingJobCreateAck::Conflict
            | EmbeddingJobCreateAck::OriginatingProcessNotLive
            | EmbeddingJobCreateAck::TransactionNotStarted => return Ok(false),
        };
        if initial_state == "failed_embedding"
            && snapshot.terminal_error_class.as_deref() != initial_error
        {
            return Ok(false);
        }
    }
    if let Some(embedding_id) = embedding_id {
        let vector_space_id = crate::vector::VectorSpaceId::new(vector_space_id.to_string())
            .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if load_embedding_cache(
            connection,
            project_uuid,
            &vector_space_id,
            canonical_query_hash,
        )?
        .is_none_or(|cache| cache.embedding_id != embedding_id)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn canonical_embedding_create_event_matches(
    connection: &Connection,
    project_uuid: Uuid,
    embedding_job_id: &str,
    graph_created_at_unix_ms: i64,
) -> Result<bool, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT embedding_job_state_event_id, process_instance_id, created_at_unix_ms
             FROM embedding_job_state_events
             WHERE embedding_job_id = ?1
             ORDER BY event_seq LIMIT 1",
            params![embedding_job_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((event_id, Some(process_instance_id), created_at_unix_ms)) = stored else {
        return Ok(false);
    };
    let process_instance_id = parse_uuid_v7(&process_instance_id)?;
    let actor_project = connection
        .query_row(
            "SELECT project_uuid FROM process_instances WHERE process_instance_id = ?1",
            params![process_instance_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    Ok(created_at_unix_ms <= graph_created_at_unix_ms
        && actor_project.as_deref() == Some(project_uuid.to_string().as_str())
        && embedding_create_event_matches(
            connection,
            parse_uuid_v7(&event_id)?,
            embedding_job_id,
            process_instance_id,
            created_at_unix_ms,
        )?)
}

fn embedding_create_event_matches(
    connection: &Connection,
    event_id: Uuid,
    embedding_job_id: &str,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<bool, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT embedding_job_id, process_instance_id, state, attempt_generation,
                    stable_error_class, created_at_unix_ms, canonical_payload_hash,
                    lease_token, lease_expires_at_unix_ms, attempt_count,
                    next_eligible_at_unix_ms, reset_actor, reset_reason
             FROM embedding_job_state_events
             WHERE embedding_job_state_event_id = ?1",
            params![event_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, Option<String>>(7)?,
                    row.get::<_, Option<i64>>(8)?,
                    row.get::<_, i64>(9)?,
                    row.get::<_, i64>(10)?,
                    row.get::<_, Option<String>>(11)?,
                    row.get::<_, Option<String>>(12)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((
        stored_job_id,
        actor,
        state,
        generation,
        stable_error,
        event_at,
        payload_hash,
        token,
        expiry,
        count,
        next,
        reset_actor,
        reset_reason,
    )) = stored
    else {
        // A shared job created by an earlier terminal has no event for this handoff.
        return Ok(true);
    };
    if stored_job_id != embedding_job_id
        || actor.as_deref() != Some(process_instance_id.to_string().as_str())
        || state != "pending"
        || generation != 0
        || stable_error.is_some()
        || event_at != created_at_unix_ms
        || token.is_some()
        || expiry.is_some()
        || count != 0
        || next != created_at_unix_ms
        || reset_actor.is_some()
        || reset_reason.is_some()
    {
        return Ok(false);
    }
    Ok(payload_hash
        == embedding_state_payload_hash(
            event_id,
            embedding_job_id,
            process_instance_id,
            created_at_unix_ms,
        )?)
}

fn record_noncanonical_outcome(
    connection: &Connection,
    terminal: &VectorGraphTerminal,
    reservation: &SampleBatchReservation,
    vector_space_id: &str,
    outcome_id: &str,
    reason: &str,
    apply: bool,
) -> Result<bool, LedgerError> {
    if !insert_or_verify_outcome(
        connection,
        terminal,
        reservation,
        vector_space_id,
        outcome_id,
        None,
        "noncanonicalizable",
        Some(reason),
        apply,
    )? {
        return Ok(false);
    }
    connection
        .query_row(
            "SELECT NOT EXISTS(
                SELECT 1 FROM evidence_vector_links
                WHERE shadow_attempt_id = ?1 AND vector_space_id = ?2
             )",
            params![terminal.shadow_attempt_id.to_string(), vector_space_id],
            |row| row.get(0),
        )
        .map_err(database_error)
}

#[allow(clippy::too_many_arguments)]
fn insert_or_verify_outcome(
    connection: &Connection,
    terminal: &VectorGraphTerminal,
    reservation: &SampleBatchReservation,
    vector_space_id: &str,
    outcome_id: &str,
    canonical_query_hash: Option<&str>,
    outcome: &str,
    stable_reason: Option<&str>,
    apply: bool,
) -> Result<bool, LedgerError> {
    let payload_hash = outcome_payload_hash(
        outcome_id,
        terminal,
        reservation,
        vector_space_id,
        canonical_query_hash,
        outcome,
        stable_reason,
    )?;
    let stored = connection
        .query_row(
            "SELECT vectorization_outcome_id, canonical_query_hash, outcome,
                    stable_reason, canonical_payload_hash
             FROM vectorization_outcomes
             WHERE shadow_attempt_id = ?1 AND vector_space_id = ?2",
            params![terminal.shadow_attempt_id.to_string(), vector_space_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, String>(4)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    if let Some((id, query, stored_outcome, reason, hash)) = stored {
        return Ok(id == outcome_id
            && query.as_deref() == canonical_query_hash
            && stored_outcome == outcome
            && reason.as_deref() == stable_reason
            && hash == payload_hash);
    }
    if !apply {
        return Ok(false);
    }
    connection
        .execute(
            "INSERT INTO vectorization_outcomes (
                vectorization_outcome_id, shadow_attempt_id, anchor_id,
                learning_generation_id, vector_space_id, canonical_query_hash,
                outcome, stable_reason, created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                outcome_id,
                terminal.shadow_attempt_id.to_string(),
                reservation.anchor_id.to_string(),
                reservation.learning_generation_id.to_string(),
                vector_space_id,
                canonical_query_hash,
                outcome,
                stable_reason,
                terminal.created_at_unix_ms,
                payload_hash,
            ],
        )
        .map_err(database_error)?;
    Ok(true)
}

fn immutable_quality_label(
    connection: &Connection,
    terminal: &VectorGraphTerminal,
) -> Result<Option<&'static str>, LedgerError> {
    let Some(evaluation_id) = terminal.evaluation_id else {
        return Ok(None);
    };
    let stored = connection
        .query_row(
            "SELECT binary_label, promotion_eligible FROM evaluations
             WHERE evaluation_id = ?1 AND shadow_attempt_id = ?2",
            params![
                evaluation_id.to_string(),
                terminal.shadow_attempt_id.to_string()
            ],
            |row| Ok((row.get::<_, Option<String>>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    match stored {
        (Some(label), 1) if label == "pass" => Ok(Some("pass")),
        (Some(label), 1) if label == "fail" => Ok(Some("fail")),
        (_, 0) => Ok(None),
        _ => Err(LedgerErrorClass::CorruptDatabase.into()),
    }
}

fn atomic_handoff(terminal: &VectorGraphTerminal) -> Option<&AtomicShadowVectorization> {
    Some(&terminal.vectorization)
}

fn link_identity_exists(
    connection: &Connection,
    link_id: Uuid,
    shadow_attempt_id: Uuid,
    vector_space_id: &str,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM evidence_vector_links
                WHERE evidence_vector_link_id = ?1
                   OR (shadow_attempt_id = ?2 AND vector_space_id = ?3)
             )",
            params![
                link_id.to_string(),
                shadow_attempt_id.to_string(),
                vector_space_id
            ],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn vectorization_outcome_id(
    shadow_attempt_id: Uuid,
    vector_space_id: &str,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "shadow_attempt_id": shadow_attempt_id,
        "vector_space_id": vector_space_id,
    }))
}

fn materialization_job_id(evidence_vector_link_id: Uuid) -> Result<String, LedgerError> {
    hash_json(&json!({
        "evidence_vector_link_id": evidence_vector_link_id,
    }))
}

#[allow(clippy::too_many_arguments)]
fn outcome_payload_hash(
    outcome_id: &str,
    terminal: &VectorGraphTerminal,
    reservation: &SampleBatchReservation,
    vector_space_id: &str,
    canonical_query_hash: Option<&str>,
    outcome: &str,
    stable_reason: Option<&str>,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "vectorization_outcome_id": outcome_id,
        "shadow_attempt_id": terminal.shadow_attempt_id,
        "anchor_id": reservation.anchor_id,
        "learning_generation_id": reservation.learning_generation_id,
        "vector_space_id": vector_space_id,
        "canonical_query_hash": canonical_query_hash,
        "outcome": outcome,
        "stable_reason": stable_reason,
        "created_at_unix_ms": terminal.created_at_unix_ms,
    }))
}

#[allow(clippy::too_many_arguments)]
fn link_payload_hash(
    evidence_vector_link_id: Uuid,
    outcome_id: &str,
    terminal: &VectorGraphTerminal,
    reservation: &SampleBatchReservation,
    root_uuid: Uuid,
    vector_space_id: &str,
    partition_id: PartitionId,
    canonical_query_hash: &str,
    quality_label: Option<&str>,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "evidence_vector_link_id": evidence_vector_link_id,
        "vectorization_outcome_id": outcome_id,
        "shadow_attempt_id": terminal.shadow_attempt_id,
        "anchor_id": reservation.anchor_id,
        "root_uuid": root_uuid,
        "learning_generation_id": reservation.learning_generation_id,
        "vector_space_id": vector_space_id,
        "partition_id": partition_id.value(),
        "canonical_query_hash": canonical_query_hash,
        "terminal_class": terminal.terminal_class.as_str(),
        "evaluation_id": terminal.evaluation_id,
        "quality_label": quality_label,
        "created_at_unix_ms": terminal.created_at_unix_ms,
    }))
}

#[allow(clippy::too_many_arguments)]
fn materialization_payload_hash(
    materialization_job_id: &str,
    evidence_vector_link_id: Uuid,
    vector_space_id: &str,
    canonical_query_hash: &str,
    embedding_job_id: Option<&str>,
    embedding_id: Option<Uuid>,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    materialization_payload_hash_current(
        materialization_job_id,
        evidence_vector_link_id,
        vector_space_id,
        canonical_query_hash,
        embedding_job_id,
        embedding_id,
        None,
        None,
        None,
        0,
        0,
        created_at_unix_ms,
        created_at_unix_ms,
    )
}

#[allow(clippy::too_many_arguments)]
fn materialization_payload_hash_current(
    materialization_job_id: &str,
    evidence_vector_link_id: Uuid,
    vector_space_id: &str,
    canonical_query_hash: &str,
    embedding_job_id: Option<&str>,
    embedding_id: Option<Uuid>,
    lease_owner_process_instance_id: Option<Uuid>,
    lease_token: Option<Uuid>,
    lease_expires_at_unix_ms: Option<i64>,
    attempt_generation: i64,
    attempt_count: i64,
    next_eligible_at_unix_ms: i64,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "vector_materialization_job_id": materialization_job_id,
        "evidence_vector_link_id": evidence_vector_link_id,
        "vector_space_id": vector_space_id,
        "canonical_query_hash": canonical_query_hash,
        "embedding_job_id": embedding_job_id,
        "embedding_id": embedding_id,
        "lease_owner_process_instance_id": lease_owner_process_instance_id,
        "lease_token": lease_token,
        "lease_expires_at_unix_ms": lease_expires_at_unix_ms,
        "attempt_generation": attempt_generation,
        "attempt_count": attempt_count,
        "next_eligible_at_unix_ms": next_eligible_at_unix_ms,
        "created_at_unix_ms": created_at_unix_ms,
    }))
}

fn link_state_payload_hash(
    event_id: Uuid,
    evidence_vector_link_id: Uuid,
    embedding_id: Option<Uuid>,
    state: &str,
    stable_error_class: Option<&str>,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    link_state_hash(
        event_id,
        evidence_vector_link_id,
        embedding_id,
        state,
        0,
        stable_error_class,
        created_at_unix_ms,
    )
}

#[allow(clippy::too_many_arguments)]
fn link_state_hash(
    event_id: Uuid,
    evidence_vector_link_id: Uuid,
    embedding_id: Option<Uuid>,
    state: &str,
    attempt_generation: i64,
    stable_error_class: Option<&str>,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "evidence_vector_link_state_event_id": event_id,
        "evidence_vector_link_id": evidence_vector_link_id,
        "embedding_id": embedding_id,
        "state": state,
        "attempt_generation": attempt_generation,
        "stable_error_class": stable_error_class,
        "created_at_unix_ms": created_at_unix_ms,
    }))
}

fn materialization_state_payload_hash(
    event_id: Uuid,
    materialization_job_id: &str,
    process_instance_id: Uuid,
    state: &str,
    stable_error_class: Option<&str>,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    materialization_state_hash(
        event_id,
        materialization_job_id,
        Some(process_instance_id),
        state,
        0,
        stable_error_class,
        None,
        None,
        0,
        created_at_unix_ms,
        created_at_unix_ms,
    )
}

#[allow(clippy::too_many_arguments)]
fn materialization_state_hash(
    event_id: Uuid,
    materialization_job_id: &str,
    process_instance_id: Option<Uuid>,
    state: &str,
    attempt_generation: i64,
    stable_error_class: Option<&str>,
    lease_token: Option<Uuid>,
    lease_expires_at_unix_ms: Option<i64>,
    attempt_count: i64,
    next_eligible_at_unix_ms: i64,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "vector_materialization_job_state_event_id": event_id,
        "vector_materialization_job_id": materialization_job_id,
        "process_instance_id": process_instance_id,
        "state": state,
        "attempt_generation": attempt_generation,
        "stable_error_class": stable_error_class,
        "lease_token": lease_token,
        "lease_expires_at_unix_ms": lease_expires_at_unix_ms,
        "attempt_count": attempt_count,
        "next_eligible_at_unix_ms": next_eligible_at_unix_ms,
        "created_at_unix_ms": created_at_unix_ms,
    }))
}

fn embedding_state_payload_hash(
    event_id: Uuid,
    embedding_job_id: &str,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "embedding_job_state_event_id": event_id,
        "embedding_job_id": embedding_job_id,
        "process_instance_id": process_instance_id,
        "state": "pending",
        "attempt_generation": 0,
        "stable_error_class": null,
        "created_at_unix_ms": created_at_unix_ms,
        "lease_token": null,
        "lease_expires_at_unix_ms": null,
        "attempt_count": 0,
        "next_eligible_at_unix_ms": created_at_unix_ms,
        "reset_actor": null,
        "reset_reason": null,
    }))
}

fn parse_uuid_v7(value: &str) -> Result<Uuid, LedgerError> {
    let parsed = Uuid::parse_str(value).map_err(|_| LedgerErrorClass::CorruptDatabase)?;
    if parsed.to_string() != value
        || parsed.get_version_num() != 7
        || parsed.get_variant() != uuid::Variant::RFC4122
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(parsed)
}

fn parse_optional_uuid(value: Option<&str>) -> Result<Option<Uuid>, LedgerError> {
    value.map(parse_uuid_v7).transpose()
}

fn validate_distinct_uuid_v7(values: &[Uuid]) -> Result<(), LedgerError> {
    let mut seen = std::collections::BTreeSet::new();
    for value in values {
        if value.get_version_num() != 7
            || value.get_variant() != uuid::Variant::RFC4122
            || !seen.insert(*value)
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn validate_sha256(value: &str) -> Result<(), LedgerError> {
    if is_sha256(value) {
        Ok(())
    } else {
        Err(LedgerErrorClass::IdentityInvariant.into())
    }
}

fn valid_stable_error_class(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'.')
        })
}

fn validate_stable_error_class(value: &str) -> Result<(), LedgerError> {
    if valid_stable_error_class(value) {
        Ok(())
    } else {
        Err(LedgerErrorClass::IdentityInvariant.into())
    }
}

fn hash_json(value: &serde_json::Value) -> Result<String, LedgerError> {
    canonical_sha256(value).map_err(|_| LedgerErrorClass::CanonicalizationFailed.into())
}

fn database_error(error: rusqlite::Error) -> LedgerError {
    super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}
