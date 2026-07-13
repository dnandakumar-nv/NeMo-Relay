// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable embedding-job identities and cross-process leases.

use std::collections::BTreeSet;

use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use serde_json::{Value as Json, json};
use uuid::Uuid;

use super::process::{ProcessStatusAt, append_integrity_health, verified_process_status_at};
use super::vector_catalog::{
    CanonicalQueryEnsureAck, EmbeddingCacheSnapshot, EmbeddingCacheSource, EmbeddingCacheUpsertAck,
    EmbeddingCacheWrite, ensure_canonical_query, load_canonical_query, load_embedding_cache,
    upsert_embedding_cache, validate_query_artifact,
};
use super::vector_registry::{FrozenMappingKey, resolve_frozen_mapping, resolve_vector_space};
use super::{LedgerRepository, TransactionStartGuard, map_fs_error, map_sqlite_error};
use crate::canonical_json::canonical_sha256;
use crate::canonical_query::CanonicalRoutingQueryArtifactV1;
use crate::ledger::fs::enforce_sidecar_permissions;
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::vector::{AuthoritativeVector, VectorSpaceId};

/// Fixed version-1 durable job lease.
pub(crate) const EMBEDDING_JOB_LEASE_MILLIS: i64 = 60_000;

/// Frozen current-mapping authority and canonical input for one live cache lookup or job.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct LiveEmbeddingPrepare {
    pub(crate) mapping: FrozenMappingKey,
    pub(crate) artifact: CanonicalRoutingQueryArtifactV1,
    pub(crate) embedding_job_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) prepared_at_unix_ms: i64,
}

impl LiveEmbeddingPrepare {
    pub(crate) fn new(
        mapping: FrozenMappingKey,
        artifact: CanonicalRoutingQueryArtifactV1,
        embedding_job_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        prepared_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        let verified_mapping = FrozenMappingKey::new(
            mapping.project_uuid,
            mapping.config_generation_id.clone(),
            mapping.pool_id.clone(),
            mapping.policy_version_id.clone(),
        )
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        validate_query_artifact(&artifact)?;
        validate_uuid_v7(embedding_job_state_event_id)?;
        validate_uuid_v7(conflict_health_event_id)?;
        if mapping != verified_mapping
            || embedding_job_state_event_id == conflict_health_event_id
            || prepared_at_unix_ms < 0
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            mapping,
            artifact,
            embedding_job_state_event_id,
            conflict_health_event_id,
            prepared_at_unix_ms,
        })
    }
}

impl std::fmt::Debug for LiveEmbeddingPrepare {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LiveEmbeddingPrepare")
            .field("mapping", &self.mapping)
            .field("canonical_query_hash", &self.artifact.canonical_query_hash)
            .field("canonical_size_bytes", &self.artifact.canonical_bytes.len())
            .field(
                "embedding_job_state_event_id",
                &self.embedding_job_state_event_id,
            )
            .field("conflict_health_event_id", &self.conflict_health_event_id)
            .field("prepared_at_unix_ms", &self.prepared_at_unix_ms)
            .finish()
    }
}

/// Immutable request to create or reuse one shared-content embedding job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingJobCreate {
    pub(crate) embedding_job_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) vector_space_id: String,
    pub(crate) canonical_query_hash: String,
    pub(crate) content_hash: String,
    pub(crate) created_at_unix_ms: i64,
    embedding_job_id: String,
}

impl EmbeddingJobCreate {
    /// Freeze deterministic job identity and idempotency material before enqueueing.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        embedding_job_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        vector_space_id: impl Into<String>,
        canonical_query_hash: impl Into<String>,
        content_hash: impl Into<String>,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(embedding_job_state_event_id)?;
        validate_uuid_v7(conflict_health_event_id)?;
        if embedding_job_state_event_id == conflict_health_event_id || created_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let vector_space_id = vector_space_id.into();
        let canonical_query_hash = canonical_query_hash.into();
        let content_hash = content_hash.into();
        validate_sha256(&vector_space_id)?;
        validate_sha256(&canonical_query_hash)?;
        validate_sha256(&content_hash)?;
        if content_hash != canonical_query_hash {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let embedding_job_id = embedding_job_id(&vector_space_id, &canonical_query_hash)?;
        Ok(Self {
            embedding_job_state_event_id,
            conflict_health_event_id,
            vector_space_id,
            canonical_query_hash,
            content_hash,
            created_at_unix_ms,
            embedding_job_id,
        })
    }

    pub(crate) fn embedding_job_id(&self) -> &str {
        &self.embedding_job_id
    }
}

/// Frozen request to acquire or reclaim one embedding-job lease.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingJobClaim {
    pub(crate) embedding_job_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) embedding_job_id: String,
    pub(crate) lease_token: Uuid,
    pub(crate) observed_at_unix_ms: i64,
    pub(crate) lease_expires_at_unix_ms: i64,
}

#[cfg(test)]
impl EmbeddingJobClaim {
    pub(crate) fn new(
        embedding_job_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        embedding_job_id: impl Into<String>,
        lease_token: Uuid,
        observed_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(embedding_job_state_event_id)?;
        validate_uuid_v7(conflict_health_event_id)?;
        validate_uuid_v7(lease_token)?;
        if embedding_job_state_event_id == conflict_health_event_id || observed_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let embedding_job_id = embedding_job_id.into();
        validate_sha256(&embedding_job_id)?;
        let lease_expires_at_unix_ms = observed_at_unix_ms
            .checked_add(EMBEDDING_JOB_LEASE_MILLIS)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        Ok(Self {
            embedding_job_state_event_id,
            conflict_health_event_id,
            embedding_job_id,
            lease_token,
            observed_at_unix_ms,
            lease_expires_at_unix_ms,
        })
    }
}

/// Frozen successful resolution of one current embedding-job lease.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingJobCompletion {
    pub(crate) embedding_job_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) embedding_job_id: String,
    pub(crate) lease_token: Uuid,
    pub(crate) attempt_generation: i64,
    pub(crate) content_hash: String,
    pub(crate) completed_at_unix_ms: i64,
}

#[cfg(test)]
impl EmbeddingJobCompletion {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        embedding_job_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        embedding_job_id: impl Into<String>,
        lease_token: Uuid,
        attempt_generation: i64,
        content_hash: impl Into<String>,
        completed_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(embedding_job_state_event_id)?;
        validate_uuid_v7(conflict_health_event_id)?;
        validate_uuid_v7(lease_token)?;
        if embedding_job_state_event_id == conflict_health_event_id
            || attempt_generation <= 0
            || completed_at_unix_ms < 0
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let embedding_job_id = embedding_job_id.into();
        let content_hash = content_hash.into();
        validate_sha256(&embedding_job_id)?;
        validate_sha256(&content_hash)?;
        Ok(Self {
            embedding_job_state_event_id,
            conflict_health_event_id,
            embedding_job_id,
            lease_token,
            attempt_generation,
            content_hash,
            completed_at_unix_ms,
        })
    }
}

/// One immutable member of an atomic same-space lease claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingJobBatchClaimItem {
    pub(crate) embedding_job_id: String,
    pub(crate) claimed_state_event_id: Uuid,
    pub(crate) orphaned_state_event_id: Uuid,
}

impl EmbeddingJobBatchClaimItem {
    pub(crate) fn new(
        embedding_job_id: impl Into<String>,
        claimed_state_event_id: Uuid,
        orphaned_state_event_id: Uuid,
    ) -> Result<Self, LedgerError> {
        let embedding_job_id = embedding_job_id.into();
        validate_sha256(&embedding_job_id)?;
        validate_uuid_v7(claimed_state_event_id)?;
        validate_uuid_v7(orphaned_state_event_id)?;
        if claimed_state_event_id == orphaned_state_event_id {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            embedding_job_id,
            claimed_state_event_id,
            orphaned_state_event_id,
        })
    }
}

/// Frozen all-or-none request to claim one nonempty same-space job batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingJobBatchClaim {
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) lease_token: Uuid,
    pub(crate) observed_at_unix_ms: i64,
    pub(crate) lease_expires_at_unix_ms: i64,
    pub(crate) items: Vec<EmbeddingJobBatchClaimItem>,
}

impl EmbeddingJobBatchClaim {
    pub(crate) fn new(
        conflict_health_event_id: Uuid,
        vector_space_id: VectorSpaceId,
        lease_token: Uuid,
        observed_at_unix_ms: i64,
        items: Vec<EmbeddingJobBatchClaimItem>,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(conflict_health_event_id)?;
        validate_uuid_v7(lease_token)?;
        if observed_at_unix_ms < 0 || items.is_empty() {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let lease_expires_at_unix_ms = observed_at_unix_ms
            .checked_add(EMBEDDING_JOB_LEASE_MILLIS)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let mut jobs = BTreeSet::new();
        let mut event_ids = BTreeSet::from([conflict_health_event_id, lease_token]);
        for item in &items {
            if !jobs.insert(item.embedding_job_id.as_str())
                || !event_ids.insert(item.claimed_state_event_id)
                || !event_ids.insert(item.orphaned_state_event_id)
            {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        Ok(Self {
            conflict_health_event_id,
            vector_space_id,
            lease_token,
            observed_at_unix_ms,
            lease_expires_at_unix_ms,
            items,
        })
    }
}

/// One vector-backed member of an atomic embedding completion batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingJobBatchCompletionItem {
    pub(crate) embedding_job_id: String,
    pub(crate) embedding_id: Uuid,
    pub(crate) completed_state_event_id: Uuid,
    pub(crate) attempt_generation: i64,
    pub(crate) content_hash: String,
    pub(crate) vector: AuthoritativeVector,
}

impl EmbeddingJobBatchCompletionItem {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        embedding_job_id: impl Into<String>,
        embedding_id: Uuid,
        completed_state_event_id: Uuid,
        attempt_generation: i64,
        content_hash: impl Into<String>,
        vector: AuthoritativeVector,
    ) -> Result<Self, LedgerError> {
        let embedding_job_id = embedding_job_id.into();
        let content_hash = content_hash.into();
        validate_sha256(&embedding_job_id)?;
        validate_sha256(&content_hash)?;
        validate_uuid_v7(embedding_id)?;
        validate_uuid_v7(completed_state_event_id)?;
        if embedding_id == completed_state_event_id || attempt_generation <= 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            embedding_job_id,
            embedding_id,
            completed_state_event_id,
            attempt_generation,
            content_hash,
            vector,
        })
    }
}

/// Frozen all-or-none request to persist vectors and complete a claimed batch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingJobBatchCompletion {
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) lease_token: Uuid,
    pub(crate) completed_at_unix_ms: i64,
    pub(crate) items: Vec<EmbeddingJobBatchCompletionItem>,
}

impl EmbeddingJobBatchCompletion {
    pub(crate) fn new(
        conflict_health_event_id: Uuid,
        vector_space_id: VectorSpaceId,
        lease_token: Uuid,
        completed_at_unix_ms: i64,
        items: Vec<EmbeddingJobBatchCompletionItem>,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(conflict_health_event_id)?;
        validate_uuid_v7(lease_token)?;
        if completed_at_unix_ms < 0 || items.is_empty() {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let mut jobs = BTreeSet::new();
        let mut embedding_ids = BTreeSet::new();
        let mut event_ids = BTreeSet::from([conflict_health_event_id, lease_token]);
        for item in &items {
            if item.vector.vector_space_id() != &vector_space_id
                || !jobs.insert(item.embedding_job_id.as_str())
                || !embedding_ids.insert(item.embedding_id)
                || !event_ids.insert(item.completed_state_event_id)
            {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        Ok(Self {
            conflict_health_event_id,
            vector_space_id,
            lease_token,
            completed_at_unix_ms,
            items,
        })
    }
}

/// Requested non-success resolution of one currently claimed embedding job.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbeddingJobResolutionKind {
    Released,
    RetryScheduled {
        stable_error_class: String,
        next_eligible_at_unix_ms: i64,
    },
    TerminalFailure {
        stable_error_class: String,
    },
    Quarantined {
        stable_error_class: String,
    },
}

/// Frozen owner/token/generation-fenced job resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingJobResolution {
    pub(crate) embedding_job_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) embedding_job_id: String,
    pub(crate) lease_token: Uuid,
    pub(crate) attempt_generation: i64,
    pub(crate) content_hash: String,
    pub(crate) resolved_at_unix_ms: i64,
    pub(crate) kind: EmbeddingJobResolutionKind,
}

impl EmbeddingJobResolution {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        embedding_job_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        embedding_job_id: impl Into<String>,
        lease_token: Uuid,
        attempt_generation: i64,
        content_hash: impl Into<String>,
        resolved_at_unix_ms: i64,
        kind: EmbeddingJobResolutionKind,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(embedding_job_state_event_id)?;
        validate_uuid_v7(conflict_health_event_id)?;
        validate_uuid_v7(lease_token)?;
        let embedding_job_id = embedding_job_id.into();
        let content_hash = content_hash.into();
        validate_sha256(&embedding_job_id)?;
        validate_sha256(&content_hash)?;
        if embedding_job_state_event_id == conflict_health_event_id
            || attempt_generation <= 0
            || resolved_at_unix_ms < 0
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        match &kind {
            EmbeddingJobResolutionKind::Released => {}
            EmbeddingJobResolutionKind::RetryScheduled {
                stable_error_class,
                next_eligible_at_unix_ms,
            } => {
                validate_stable_error_class(stable_error_class)?;
                if *next_eligible_at_unix_ms < resolved_at_unix_ms {
                    return Err(LedgerErrorClass::IdentityInvariant.into());
                }
            }
            EmbeddingJobResolutionKind::TerminalFailure { stable_error_class }
            | EmbeddingJobResolutionKind::Quarantined { stable_error_class } => {
                validate_stable_error_class(stable_error_class)?;
            }
        }
        Ok(Self {
            embedding_job_state_event_id,
            conflict_health_event_id,
            embedding_job_id,
            lease_token,
            attempt_generation,
            content_hash,
            resolved_at_unix_ms,
            kind,
        })
    }
}

/// Frozen explicit reset of one fully propagated terminal failure.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingJobReset {
    pub(crate) embedding_job_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) embedding_job_id: String,
    pub(crate) expected_attempt_generation: i64,
    pub(crate) expected_canonical_payload_hash: String,
    pub(crate) reset_actor: String,
    pub(crate) reset_reason: String,
    pub(crate) reset_at_unix_ms: i64,
}

impl EmbeddingJobReset {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        embedding_job_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        embedding_job_id: impl Into<String>,
        expected_attempt_generation: i64,
        expected_canonical_payload_hash: impl Into<String>,
        reset_actor: impl Into<String>,
        reset_reason: impl Into<String>,
        reset_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(embedding_job_state_event_id)?;
        validate_uuid_v7(conflict_health_event_id)?;
        let embedding_job_id = embedding_job_id.into();
        let expected_canonical_payload_hash = expected_canonical_payload_hash.into();
        let reset_actor = reset_actor.into();
        let reset_reason = reset_reason.into();
        validate_sha256(&embedding_job_id)?;
        validate_sha256(&expected_canonical_payload_hash)?;
        if embedding_job_state_event_id == conflict_health_event_id
            || !(0..i64::MAX).contains(&expected_attempt_generation)
            || reset_at_unix_ms < 0
            || !valid_bounded_nonempty(&reset_actor, 128)
            || !valid_bounded_nonempty(&reset_reason, 512)
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            embedding_job_state_event_id,
            conflict_health_event_id,
            embedding_job_id,
            expected_attempt_generation,
            expected_canonical_payload_hash,
            reset_actor,
            reset_reason,
            reset_at_unix_ms,
        })
    }
}

/// Fenced progress update for bounded terminal-failure propagation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingFailurePropagationUpdate {
    pub(crate) embedding_job_id: String,
    pub(crate) expected_canonical_payload_hash: String,
    pub(crate) expected_attempt_generation: i64,
    pub(crate) expected_cursor: Option<Uuid>,
    pub(crate) next_cursor: Option<Uuid>,
    pub(crate) complete: bool,
}

impl EmbeddingFailurePropagationUpdate {
    pub(crate) fn new(
        embedding_job_id: impl Into<String>,
        expected_canonical_payload_hash: impl Into<String>,
        expected_attempt_generation: i64,
        expected_cursor: Option<Uuid>,
        next_cursor: Option<Uuid>,
        complete: bool,
    ) -> Result<Self, LedgerError> {
        let embedding_job_id = embedding_job_id.into();
        let expected_canonical_payload_hash = expected_canonical_payload_hash.into();
        validate_sha256(&embedding_job_id)?;
        validate_sha256(&expected_canonical_payload_hash)?;
        for cursor in expected_cursor.into_iter().chain(next_cursor) {
            validate_uuid_v7(cursor)?;
        }
        if expected_attempt_generation <= 0
            || expected_cursor
                .zip(next_cursor)
                .is_some_and(|(expected, next)| next < expected)
            || (expected_cursor.is_some() && next_cursor.is_none())
            || (!complete
                && next_cursor
                    .is_none_or(|next| expected_cursor.is_some_and(|expected| next <= expected)))
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            embedding_job_id,
            expected_canonical_payload_hash,
            expected_attempt_generation,
            expected_cursor,
            next_cursor,
            complete,
        })
    }
}

/// Canonical current job facts returned after a create command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingJobSnapshot {
    pub(crate) embedding_job_id: String,
    pub(crate) vector_space_id: String,
    pub(crate) canonical_query_hash: String,
    pub(crate) content_hash: String,
    pub(crate) attempt_generation: i64,
    pub(crate) attempt_count: i64,
    pub(crate) next_eligible_at_unix_ms: i64,
    pub(crate) terminal_error_class: Option<String>,
    pub(crate) failure_propagation_cursor: Option<String>,
    pub(crate) failure_propagation_complete: bool,
    pub(crate) reset_actor: Option<String>,
    pub(crate) reset_reason: Option<String>,
    pub(crate) canonical_payload_hash: String,
}

/// Current lease authority returned only to its owner.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingJobLease {
    pub(crate) job: EmbeddingJobSnapshot,
    pub(crate) lease_owner_process_instance_id: Uuid,
    pub(crate) lease_token: Uuid,
    pub(crate) lease_expires_at_unix_ms: i64,
    pub(crate) state_event_hash: String,
}

/// Exhaustive result of deterministic job creation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbeddingJobCreateAck {
    Applied(EmbeddingJobSnapshot),
    AlreadyExists(EmbeddingJobSnapshot),
    VectorSpaceNotFound,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

/// Exhaustive result of atomically preparing one cache-first live embedding.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LiveEmbeddingPrepareAck {
    Ready(EmbeddingCacheSnapshot),
    Pending(EmbeddingJobSnapshot),
    MappingNotFound,
    MappingNotCurrent,
    ProviderSpaceDegraded,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

/// Exhaustive result of a lease acquisition attempt.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbeddingJobClaimAck {
    Claimed(EmbeddingJobLease),
    Reclaimed(EmbeddingJobLease),
    AlreadyApplied(EmbeddingJobLease),
    NotFound,
    NotEligible { next_eligible_at_unix_ms: i64 },
    LeaseHeld { lease_expires_at_unix_ms: i64 },
    Terminal,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

/// Exhaustive result of resolving a lease successfully.
#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbeddingJobCompletionAck {
    Applied {
        job: EmbeddingJobSnapshot,
        state_event_hash: String,
    },
    AlreadyApplied {
        job: EmbeddingJobSnapshot,
        state_event_hash: String,
    },
    NotFound,
    StaleLease,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

/// Exhaustive result of an atomic same-space batch claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbeddingJobBatchClaimAck {
    Claimed(Vec<EmbeddingJobLease>),
    AlreadyApplied(Vec<EmbeddingJobLease>),
    VectorSpaceNotFound,
    VectorSpaceUnauthorized,
    BatchTooLarge {
        max_batch_size: usize,
    },
    JobNotFound {
        embedding_job_id: String,
    },
    NotEligible {
        embedding_job_id: String,
        next_eligible_at_unix_ms: i64,
    },
    LeaseHeld {
        embedding_job_id: String,
        lease_expires_at_unix_ms: i64,
    },
    Terminal {
        embedding_job_id: String,
    },
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

/// One vector/cache/job result from a completed batch member.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingJobBatchCompletionResult {
    pub(crate) job: EmbeddingJobSnapshot,
    pub(crate) embedding: EmbeddingCacheSnapshot,
    pub(crate) state_event_hash: String,
}

/// Exhaustive result of atomic vector persistence and job completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbeddingJobBatchCompletionAck {
    Applied(Vec<EmbeddingJobBatchCompletionResult>),
    AlreadyApplied(Vec<EmbeddingJobBatchCompletionResult>),
    VectorSpaceNotFound,
    BatchTooLarge { max_batch_size: usize },
    JobNotFound { embedding_job_id: String },
    StaleLease { embedding_job_id: String },
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

/// Canonical state actually appended by a non-success resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbeddingJobResolvedState {
    Released,
    RetryScheduled,
    TerminalFailure,
    Quarantined,
}

impl EmbeddingJobResolvedState {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Released => "released",
            Self::RetryScheduled => "retry_scheduled",
            Self::TerminalFailure => "terminal_failure",
            Self::Quarantined => "quarantined",
        }
    }
}

/// Exhaustive result of a lease-fenced non-success resolution.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbeddingJobResolutionAck {
    Applied {
        job: EmbeddingJobSnapshot,
        state: EmbeddingJobResolvedState,
        state_event_hash: String,
    },
    AlreadyApplied {
        job: EmbeddingJobSnapshot,
        state: EmbeddingJobResolvedState,
        state_event_hash: String,
    },
    NotFound,
    StaleLease,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

/// Exhaustive result of an explicit terminal reset.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbeddingJobResetAck {
    Applied {
        job: EmbeddingJobSnapshot,
        state_event_hash: String,
    },
    AlreadyApplied {
        job: EmbeddingJobSnapshot,
        state_event_hash: String,
    },
    NotFound,
    NotTerminal,
    PropagationIncomplete,
    Stale,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

/// Exhaustive result of a bounded failure-propagation cursor update.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbeddingFailurePropagationAck {
    Applied(EmbeddingJobSnapshot),
    AlreadyApplied(EmbeddingJobSnapshot),
    NotFound,
    NotTerminal,
    Stale,
    Conflict,
}

#[derive(Debug, PartialEq, Eq)]
struct StoredJob {
    embedding_job_id: String,
    vector_space_id: String,
    canonical_query_hash: String,
    content_hash: String,
    lease_owner_process_instance_id: Option<String>,
    lease_token: Option<String>,
    lease_expires_at_unix_ms: Option<i64>,
    attempt_generation: i64,
    attempt_count: i64,
    next_eligible_at_unix_ms: i64,
    terminal_error_class: Option<String>,
    failure_propagation_cursor: Option<String>,
    failure_propagation_complete: bool,
    reset_actor: Option<String>,
    reset_reason: Option<String>,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug, PartialEq, Eq)]
struct StoredJobState {
    embedding_job_state_event_id: String,
    embedding_job_id: String,
    process_instance_id: Option<String>,
    state: String,
    attempt_generation: i64,
    stable_error_class: Option<String>,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
    lease_token: Option<String>,
    lease_expires_at_unix_ms: Option<i64>,
    attempt_count: Option<i64>,
    next_eligible_at_unix_ms: Option<i64>,
    reset_actor: Option<String>,
    reset_reason: Option<String>,
}

#[derive(Debug)]
struct VerifiedJob {
    stored: StoredJob,
    states: Vec<StoredJobState>,
}

impl LedgerRepository {
    pub(crate) fn prepare_live_embedding(
        &mut self,
        command: &LiveEmbeddingPrepare,
    ) -> Result<LiveEmbeddingPrepareAck, LedgerError> {
        self.prepare_live_embedding_with_start_check(command, || Some(()))
    }

    pub(crate) fn prepare_live_embedding_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &LiveEmbeddingPrepare,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<LiveEmbeddingPrepareAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(LiveEmbeddingPrepareAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(LiveEmbeddingPrepareAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(LiveEmbeddingPrepareAck::TransactionNotStarted);
        }
        drop(start_guard);
        match verified_process_status_at(
            &transaction,
            project_uuid,
            process_instance_id,
            command.prepared_at_unix_ms,
        )? {
            ProcessStatusAt::Live => {}
            ProcessStatusAt::Expired | ProcessStatusAt::Terminal => {
                return Ok(LiveEmbeddingPrepareAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }

        transaction
            .execute_batch("SAVEPOINT live_embedding_prepare")
            .map_err(database_error)?;
        let prepare_result = prepare_live_embedding_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            command,
        );
        if matches!(
            prepare_result,
            Ok(LiveEmbeddingPrepareAck::Ready(_)) | Ok(LiveEmbeddingPrepareAck::Pending(_))
        ) {
            transaction
                .execute_batch("RELEASE live_embedding_prepare")
                .map_err(database_error)?;
        } else {
            transaction
                .execute_batch("ROLLBACK TO live_embedding_prepare; RELEASE live_embedding_prepare")
                .map_err(database_error)?;
        }
        let acknowledgement = prepare_result?;
        if acknowledgement == LiveEmbeddingPrepareAck::Conflict {
            append_integrity_health(
                &transaction,
                command.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                None,
                None,
                command.prepared_at_unix_ms,
            )?;
        }
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }

    pub(crate) fn create_embedding_job(
        &mut self,
        command: &EmbeddingJobCreate,
    ) -> Result<EmbeddingJobCreateAck, LedgerError> {
        self.create_embedding_job_with_start_check(command, || Some(()))
    }

    pub(crate) fn create_embedding_job_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &EmbeddingJobCreate,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<EmbeddingJobCreateAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(EmbeddingJobCreateAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(EmbeddingJobCreateAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(EmbeddingJobCreateAck::TransactionNotStarted);
        }
        drop(start_guard);
        match verified_process_status_at(
            &transaction,
            project_uuid,
            process_instance_id,
            command.created_at_unix_ms,
        )? {
            ProcessStatusAt::Live => {}
            ProcessStatusAt::Expired | ProcessStatusAt::Terminal => {
                return Ok(EmbeddingJobCreateAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => return Err(LedgerErrorClass::IdentityInvariant.into()),
        }

        let acknowledgement = create_embedding_job_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            command,
        )?;
        if acknowledgement == EmbeddingJobCreateAck::Conflict {
            append_integrity_health(
                &transaction,
                command.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                None,
                None,
                command.created_at_unix_ms,
            )?;
        }
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }

    #[cfg(test)]
    pub(crate) fn claim_embedding_job(
        &mut self,
        command: &EmbeddingJobClaim,
    ) -> Result<EmbeddingJobClaimAck, LedgerError> {
        self.claim_embedding_job_with_start_check(command, || Some(()))
    }

    #[cfg(test)]
    pub(crate) fn claim_embedding_job_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &EmbeddingJobClaim,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<EmbeddingJobClaimAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(EmbeddingJobClaimAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(EmbeddingJobClaimAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(EmbeddingJobClaimAck::TransactionNotStarted);
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
                return Ok(EmbeddingJobClaimAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => return Err(LedgerErrorClass::IdentityInvariant.into()),
        }

        let acknowledgement = claim_embedding_job_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            command,
        )?;
        if acknowledgement == EmbeddingJobClaimAck::Conflict {
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

    pub(crate) fn claim_embedding_job_batch(
        &mut self,
        command: &EmbeddingJobBatchClaim,
    ) -> Result<EmbeddingJobBatchClaimAck, LedgerError> {
        self.claim_embedding_job_batch_with_start_check(command, || Some(()))
    }

    pub(crate) fn claim_embedding_job_batch_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &EmbeddingJobBatchClaim,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<EmbeddingJobBatchClaimAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(EmbeddingJobBatchClaimAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(EmbeddingJobBatchClaimAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(EmbeddingJobBatchClaimAck::TransactionNotStarted);
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
                return Ok(EmbeddingJobBatchClaimAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => return Err(LedgerErrorClass::IdentityInvariant.into()),
        }
        let acknowledgement = claim_embedding_job_batch_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            command,
        )?;
        if acknowledgement == EmbeddingJobBatchClaimAck::Conflict {
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

    #[cfg(test)]
    pub(crate) fn complete_embedding_job(
        &mut self,
        command: &EmbeddingJobCompletion,
    ) -> Result<EmbeddingJobCompletionAck, LedgerError> {
        self.complete_embedding_job_with_start_check(command, || Some(()))
    }

    #[cfg(test)]
    pub(crate) fn complete_embedding_job_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &EmbeddingJobCompletion,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<EmbeddingJobCompletionAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(EmbeddingJobCompletionAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(EmbeddingJobCompletionAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(EmbeddingJobCompletionAck::TransactionNotStarted);
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
                return Ok(EmbeddingJobCompletionAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => return Err(LedgerErrorClass::IdentityInvariant.into()),
        }

        let acknowledgement = complete_embedding_job_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            command,
        )?;
        if acknowledgement == EmbeddingJobCompletionAck::Conflict {
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

    pub(crate) fn complete_embedding_job_batch(
        &mut self,
        command: &EmbeddingJobBatchCompletion,
    ) -> Result<EmbeddingJobBatchCompletionAck, LedgerError> {
        self.complete_embedding_job_batch_with_start_check(command, || Some(()))
    }

    pub(crate) fn complete_embedding_job_batch_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &EmbeddingJobBatchCompletion,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<EmbeddingJobBatchCompletionAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(EmbeddingJobBatchCompletionAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(EmbeddingJobBatchCompletionAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(EmbeddingJobBatchCompletionAck::TransactionNotStarted);
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
                return Ok(EmbeddingJobBatchCompletionAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => return Err(LedgerErrorClass::IdentityInvariant.into()),
        }
        let acknowledgement = complete_embedding_job_batch_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            command,
        )?;
        if acknowledgement == EmbeddingJobBatchCompletionAck::Conflict {
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

    pub(crate) fn resolve_embedding_job(
        &mut self,
        command: &EmbeddingJobResolution,
    ) -> Result<EmbeddingJobResolutionAck, LedgerError> {
        self.resolve_embedding_job_with_start_check(command, || Some(()))
    }

    pub(crate) fn resolve_embedding_job_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &EmbeddingJobResolution,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<EmbeddingJobResolutionAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(EmbeddingJobResolutionAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(EmbeddingJobResolutionAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(EmbeddingJobResolutionAck::TransactionNotStarted);
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
                return Ok(EmbeddingJobResolutionAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => return Err(LedgerErrorClass::IdentityInvariant.into()),
        }
        let acknowledgement = resolve_embedding_job_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            command,
        )?;
        if acknowledgement == EmbeddingJobResolutionAck::Conflict {
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

    pub(crate) fn reset_embedding_job(
        &mut self,
        command: &EmbeddingJobReset,
    ) -> Result<EmbeddingJobResetAck, LedgerError> {
        self.reset_embedding_job_with_start_check(command, || Some(()))
    }

    pub(crate) fn reset_embedding_job_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &EmbeddingJobReset,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<EmbeddingJobResetAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(EmbeddingJobResetAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(EmbeddingJobResetAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(EmbeddingJobResetAck::TransactionNotStarted);
        }
        drop(start_guard);
        match verified_process_status_at(
            &transaction,
            project_uuid,
            process_instance_id,
            command.reset_at_unix_ms,
        )? {
            ProcessStatusAt::Live => {}
            ProcessStatusAt::Expired | ProcessStatusAt::Terminal => {
                return Ok(EmbeddingJobResetAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => return Err(LedgerErrorClass::IdentityInvariant.into()),
        }
        let acknowledgement = reset_embedding_job_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            command,
        )?;
        if acknowledgement == EmbeddingJobResetAck::Conflict {
            append_integrity_health(
                &transaction,
                command.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                None,
                None,
                command.reset_at_unix_ms,
            )?;
        }
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }
}

pub(crate) fn prepare_live_embedding_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &LiveEmbeddingPrepare,
) -> Result<LiveEmbeddingPrepareAck, LedgerError> {
    let validated = LiveEmbeddingPrepare::new(
        command.mapping.clone(),
        command.artifact.clone(),
        command.embedding_job_state_event_id,
        command.conflict_health_event_id,
        command.prepared_at_unix_ms,
    )?;
    if validated != *command || command.mapping.project_uuid != project_uuid {
        return Ok(LiveEmbeddingPrepareAck::Conflict);
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
        return Ok(LiveEmbeddingPrepareAck::MappingNotCurrent);
    }
    let Some(mapping) = resolve_frozen_mapping(transaction, &command.mapping)? else {
        return Ok(LiveEmbeddingPrepareAck::MappingNotFound);
    };
    if mapping.mapping.project_uuid != project_uuid
        || mapping.mapping.config_generation_id != command.mapping.config_generation_id
        || mapping.mapping.pool_id != command.mapping.pool_id
        || mapping.mapping.policy_version_id != command.mapping.policy_version_id
        || mapping.space.space.vector_space_id != mapping.mapping.vector_space_id
        || command.prepared_at_unix_ms < mapping.created_at_unix_ms
    {
        return Ok(LiveEmbeddingPrepareAck::Conflict);
    }
    let vector_space_id = &mapping.mapping.vector_space_id;
    if let Some(embedding) = load_embedding_cache(
        transaction,
        project_uuid,
        vector_space_id,
        &command.artifact.canonical_query_hash,
    )? {
        return Ok(LiveEmbeddingPrepareAck::Ready(embedding));
    }
    if embedding_space_is_degraded(transaction, project_uuid, vector_space_id)? {
        return Ok(LiveEmbeddingPrepareAck::ProviderSpaceDegraded);
    }
    if ensure_canonical_query(transaction, &command.artifact, command.prepared_at_unix_ms)?
        == CanonicalQueryEnsureAck::Conflict
    {
        return Ok(LiveEmbeddingPrepareAck::Conflict);
    }

    let create = EmbeddingJobCreate::new(
        command.embedding_job_state_event_id,
        command.conflict_health_event_id,
        vector_space_id.as_str(),
        command.artifact.canonical_query_hash.clone(),
        command.artifact.canonical_query_hash.clone(),
        command.prepared_at_unix_ms,
    )?;
    match create_embedding_job_in_transaction(
        transaction,
        project_uuid,
        process_instance_id,
        &create,
    )? {
        EmbeddingJobCreateAck::Applied(job) | EmbeddingJobCreateAck::AlreadyExists(job) => {
            Ok(LiveEmbeddingPrepareAck::Pending(job))
        }
        EmbeddingJobCreateAck::Conflict => Ok(LiveEmbeddingPrepareAck::Conflict),
        EmbeddingJobCreateAck::VectorSpaceNotFound
        | EmbeddingJobCreateAck::OriginatingProcessNotLive
        | EmbeddingJobCreateAck::TransactionNotStarted => {
            Err(LedgerErrorClass::CorruptDatabase.into())
        }
    }
}

pub(crate) fn create_embedding_job_in_transaction(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &EmbeddingJobCreate,
) -> Result<EmbeddingJobCreateAck, LedgerError> {
    if command.content_hash != command.canonical_query_hash {
        return Ok(EmbeddingJobCreateAck::Conflict);
    }
    match verified_vector_space(connection, project_uuid, &command.vector_space_id)? {
        VectorSpaceStatus::Missing => return Ok(EmbeddingJobCreateAck::VectorSpaceNotFound),
        VectorSpaceStatus::Invalid => return Ok(EmbeddingJobCreateAck::Conflict),
        VectorSpaceStatus::Valid => {}
    }
    if load_canonical_query(connection, &command.canonical_query_hash)?.is_none() {
        return Ok(EmbeddingJobCreateAck::Conflict);
    }
    let supplied_event = load_state_by_event_id(connection, command.embedding_job_state_event_id)?;
    let matching = load_jobs_for_identity(
        connection,
        &command.embedding_job_id,
        &command.vector_space_id,
        &command.canonical_query_hash,
    )?;
    if !matching.is_empty() {
        if matching.len() != 1 {
            return Ok(EmbeddingJobCreateAck::Conflict);
        }
        let Some(verified) = verify_job(
            connection,
            project_uuid,
            matching.into_iter().next().unwrap(),
        )?
        else {
            return Ok(EmbeddingJobCreateAck::Conflict);
        };
        if verified.stored.embedding_job_id != command.embedding_job_id
            || verified.stored.vector_space_id != command.vector_space_id
            || verified.stored.canonical_query_hash != command.canonical_query_hash
            || verified.stored.content_hash != command.content_hash
        {
            return Ok(EmbeddingJobCreateAck::Conflict);
        }
        if let Some(event) = supplied_event.as_ref()
            && !create_event_matches(event, process_instance_id, command)?
        {
            return Ok(EmbeddingJobCreateAck::Conflict);
        }
        return Ok(EmbeddingJobCreateAck::AlreadyExists(snapshot(
            &verified.stored,
        )));
    }
    if supplied_event.is_some() {
        return Ok(EmbeddingJobCreateAck::Conflict);
    }

    let job_hash = job_hash(
        &command.embedding_job_id,
        &command.vector_space_id,
        &command.canonical_query_hash,
        &command.content_hash,
        None,
        None,
        None,
        0,
        0,
        command.created_at_unix_ms,
        None,
        None,
        false,
        None,
        None,
        command.created_at_unix_ms,
    )?;
    connection
        .execute(
            "INSERT INTO embedding_jobs (
                embedding_job_id, vector_space_id, canonical_query_hash, content_hash,
                lease_owner_process_instance_id, lease_token, lease_expires_at_unix_ms,
                attempt_generation, attempt_count, next_eligible_at_unix_ms,
                terminal_error_class, failure_propagation_cursor,
                failure_propagation_complete, reset_actor, reset_reason,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (
                ?1, ?2, ?3, ?4, NULL, NULL, NULL, 0, 0, ?5,
                NULL, NULL, 0, NULL, NULL, ?5, ?6
             )",
            params![
                command.embedding_job_id,
                command.vector_space_id,
                command.canonical_query_hash,
                command.content_hash,
                command.created_at_unix_ms,
                job_hash,
            ],
        )
        .map_err(database_error)?;
    let state_hash = state_hash(
        command.embedding_job_state_event_id,
        &command.embedding_job_id,
        Some(process_instance_id),
        "pending",
        0,
        None,
        command.created_at_unix_ms,
        None,
        None,
        0,
        command.created_at_unix_ms,
        None,
        None,
    )?;
    insert_state(
        connection,
        command.embedding_job_state_event_id,
        &command.embedding_job_id,
        process_instance_id,
        "pending",
        0,
        None,
        command.created_at_unix_ms,
        None,
        None,
        0,
        command.created_at_unix_ms,
        None,
        None,
        &state_hash,
    )?;
    let stored = load_job(connection, &command.embedding_job_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    Ok(EmbeddingJobCreateAck::Applied(snapshot(&stored)))
}

pub(crate) fn claim_embedding_job_batch_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &EmbeddingJobBatchClaim,
) -> Result<EmbeddingJobBatchClaimAck, LedgerError> {
    let Some(max_batch_size) =
        verified_profile_batch_size(transaction, project_uuid, &command.vector_space_id)?
    else {
        return Ok(EmbeddingJobBatchClaimAck::VectorSpaceNotFound);
    };
    if !claimant_authorizes_vector_space(
        transaction,
        project_uuid,
        process_instance_id,
        &command.vector_space_id,
    )? {
        return Ok(EmbeddingJobBatchClaimAck::VectorSpaceUnauthorized);
    }
    if command.items.len() > max_batch_size {
        return Ok(EmbeddingJobBatchClaimAck::BatchTooLarge { max_batch_size });
    }
    let Some(first_item) = command.items.first() else {
        return Ok(EmbeddingJobBatchClaimAck::Conflict);
    };
    if embedding_space_is_degraded(transaction, project_uuid, &command.vector_space_id)? {
        return Ok(EmbeddingJobBatchClaimAck::Terminal {
            embedding_job_id: first_item.embedding_job_id.clone(),
        });
    }

    let mut verified_jobs = Vec::with_capacity(command.items.len());
    let mut retry_leases = Vec::with_capacity(command.items.len());
    let mut existing_claim_count = 0_usize;
    for item in &command.items {
        let Some(stored) = load_job(transaction, &item.embedding_job_id)? else {
            return Ok(EmbeddingJobBatchClaimAck::JobNotFound {
                embedding_job_id: item.embedding_job_id.clone(),
            });
        };
        let Some(verified) = verify_job(transaction, project_uuid, stored)? else {
            return Ok(EmbeddingJobBatchClaimAck::Conflict);
        };
        if verified.stored.vector_space_id != command.vector_space_id.as_str()
            || !job_belongs_to_project(transaction, project_uuid, &verified.stored)?
        {
            return Ok(EmbeddingJobBatchClaimAck::Conflict);
        }
        let claim_event = load_state_by_event_id(transaction, item.claimed_state_event_id)?;
        let orphan_event = load_state_by_event_id(transaction, item.orphaned_state_event_id)?;
        if let Some(event) = claim_event.as_ref() {
            existing_claim_count += 1;
            if !batch_claim_event_matches(event, process_instance_id, command, item)?
                || !claim_event_matches_current(event, &verified.stored)?
            {
                return Ok(EmbeddingJobBatchClaimAck::Conflict);
            }
            if let Some(orphan) = orphan_event.as_ref()
                && !orphan_event_matches_retry(orphan, event, process_instance_id, command)?
            {
                return Ok(EmbeddingJobBatchClaimAck::Conflict);
            }
            retry_leases.push(lease_from_current(&verified.stored, event)?);
        } else if orphan_event.is_some() {
            return Ok(EmbeddingJobBatchClaimAck::Conflict);
        }
        verified_jobs.push(verified);
    }
    if existing_claim_count != 0 {
        return Ok(if existing_claim_count == command.items.len() {
            EmbeddingJobBatchClaimAck::AlreadyApplied(retry_leases)
        } else {
            EmbeddingJobBatchClaimAck::Conflict
        });
    }

    let mut plans = Vec::with_capacity(command.items.len());
    for (item, verified) in command.items.iter().zip(verified_jobs) {
        let latest = verified
            .states
            .last()
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if matches!(
            latest.state.as_str(),
            "completed" | "terminal_failure" | "quarantined"
        ) || verified.stored.attempt_count >= 5
        {
            return Ok(EmbeddingJobBatchClaimAck::Terminal {
                embedding_job_id: item.embedding_job_id.clone(),
            });
        }
        let current_lease = current_lease(&verified.stored)?;
        if let Some((owner, token, expires_at)) = current_lease {
            if token == command.lease_token {
                return Ok(EmbeddingJobBatchClaimAck::Conflict);
            }
            // A competing claim can commit while this command waits for the writer lock.
            // Classify that verified lease at its own event time before rejecting the stale
            // observation, but never let the stale command reclaim or mutate it.
            let lease_observed_at = command.observed_at_unix_ms.max(latest.created_at_unix_ms);
            let owner_status =
                verified_process_status_at(transaction, project_uuid, owner, lease_observed_at)?;
            if expires_at > lease_observed_at && owner_status == ProcessStatusAt::Live {
                return Ok(EmbeddingJobBatchClaimAck::LeaseHeld {
                    embedding_job_id: item.embedding_job_id.clone(),
                    lease_expires_at_unix_ms: expires_at,
                });
            }
            if owner_status == ProcessStatusAt::Invalid {
                return Ok(EmbeddingJobBatchClaimAck::Conflict);
            }
        }
        if command.observed_at_unix_ms < latest.created_at_unix_ms {
            return Ok(EmbeddingJobBatchClaimAck::Conflict);
        }
        if verified.stored.next_eligible_at_unix_ms > command.observed_at_unix_ms {
            return Ok(EmbeddingJobBatchClaimAck::NotEligible {
                embedding_job_id: item.embedding_job_id.clone(),
                next_eligible_at_unix_ms: verified.stored.next_eligible_at_unix_ms,
            });
        }
        let reclaimed_lease = current_lease.map(|(_, token, expires_at)| (token, expires_at));
        let attempt_generation = verified
            .stored
            .attempt_generation
            .checked_add(1)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let attempt_count = verified
            .stored
            .attempt_count
            .checked_add(1)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if attempt_count > 5
            || load_state_for_transition(
                transaction,
                &item.embedding_job_id,
                attempt_generation,
                "claimed",
            )?
            .is_some()
            || (reclaimed_lease.is_some()
                && load_state_for_transition(
                    transaction,
                    &item.embedding_job_id,
                    verified.stored.attempt_generation,
                    "orphaned_in_flight",
                )?
                .is_some())
        {
            return Ok(EmbeddingJobBatchClaimAck::Conflict);
        }
        plans.push((
            item,
            verified,
            reclaimed_lease,
            attempt_generation,
            attempt_count,
        ));
    }

    let mut leases = Vec::with_capacity(plans.len());
    for (item, verified, reclaimed_lease, attempt_generation, attempt_count) in plans {
        if let Some((old_token, old_expiry)) = reclaimed_lease {
            let orphan_hash = state_hash(
                item.orphaned_state_event_id,
                &item.embedding_job_id,
                Some(process_instance_id),
                "orphaned_in_flight",
                verified.stored.attempt_generation,
                None,
                command.observed_at_unix_ms,
                Some(old_token),
                Some(old_expiry),
                verified.stored.attempt_count,
                verified.stored.next_eligible_at_unix_ms,
                None,
                None,
            )?;
            insert_state(
                transaction,
                item.orphaned_state_event_id,
                &item.embedding_job_id,
                process_instance_id,
                "orphaned_in_flight",
                verified.stored.attempt_generation,
                None,
                command.observed_at_unix_ms,
                Some(old_token),
                Some(old_expiry),
                verified.stored.attempt_count,
                verified.stored.next_eligible_at_unix_ms,
                None,
                None,
                &orphan_hash,
            )?;
        }
        let new_hash = job_hash(
            &verified.stored.embedding_job_id,
            &verified.stored.vector_space_id,
            &verified.stored.canonical_query_hash,
            &verified.stored.content_hash,
            Some(process_instance_id),
            Some(command.lease_token),
            Some(command.lease_expires_at_unix_ms),
            attempt_generation,
            attempt_count,
            verified.stored.next_eligible_at_unix_ms,
            None,
            None,
            false,
            verified.stored.reset_actor.as_deref(),
            verified.stored.reset_reason.as_deref(),
            verified.stored.created_at_unix_ms,
        )?;
        let updated = transaction
            .execute(
                "UPDATE embedding_jobs
                 SET lease_owner_process_instance_id = ?1, lease_token = ?2,
                     lease_expires_at_unix_ms = ?3, attempt_generation = ?4,
                     attempt_count = ?5, canonical_payload_hash = ?6
                 WHERE embedding_job_id = ?7 AND canonical_payload_hash = ?8",
                params![
                    process_instance_id.to_string(),
                    command.lease_token.to_string(),
                    command.lease_expires_at_unix_ms,
                    attempt_generation,
                    attempt_count,
                    new_hash,
                    item.embedding_job_id,
                    verified.stored.canonical_payload_hash,
                ],
            )
            .map_err(database_error)?;
        if updated != 1 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let claim_hash = state_hash(
            item.claimed_state_event_id,
            &item.embedding_job_id,
            Some(process_instance_id),
            "claimed",
            attempt_generation,
            None,
            command.observed_at_unix_ms,
            Some(command.lease_token),
            Some(command.lease_expires_at_unix_ms),
            attempt_count,
            verified.stored.next_eligible_at_unix_ms,
            None,
            None,
        )?;
        insert_state(
            transaction,
            item.claimed_state_event_id,
            &item.embedding_job_id,
            process_instance_id,
            "claimed",
            attempt_generation,
            None,
            command.observed_at_unix_ms,
            Some(command.lease_token),
            Some(command.lease_expires_at_unix_ms),
            attempt_count,
            verified.stored.next_eligible_at_unix_ms,
            None,
            None,
            &claim_hash,
        )?;
        let stored = load_job(transaction, &item.embedding_job_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let event = load_state_by_event_id(transaction, item.claimed_state_event_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        leases.push(lease_from_current(&stored, &event)?);
    }
    Ok(EmbeddingJobBatchClaimAck::Claimed(leases))
}

fn claimant_authorizes_vector_space(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    vector_space_id: &VectorSpaceId,
) -> Result<bool, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT mapping.config_generation_id, mapping.pool_id,
                    mapping.policy_version_id
             FROM process_instances AS process
             JOIN pool_vector_space_mappings AS mapping
               ON mapping.project_uuid = process.project_uuid
              AND mapping.config_generation_id = process.config_generation_id
             WHERE process.process_instance_id = ?1
               AND process.project_uuid = ?2
               AND mapping.vector_space_id = ?3
             ORDER BY mapping.pool_id, mapping.policy_version_id",
        )
        .map_err(database_error)?;
    let raw_keys = statement
        .query_map(
            params![
                process_instance_id.to_string(),
                project_uuid.to_string(),
                vector_space_id.as_str(),
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
    drop(statement);
    if raw_keys.is_empty() {
        return Ok(false);
    }
    for (config_generation_id, pool_id, policy_version_id) in raw_keys {
        let key = FrozenMappingKey::new(
            project_uuid,
            config_generation_id,
            pool_id,
            policy_version_id,
        )
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        let mapping = resolve_frozen_mapping(connection, &key)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        if &mapping.mapping.vector_space_id != vector_space_id {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }
    Ok(true)
}

fn batch_claim_event_matches(
    event: &StoredJobState,
    process_instance_id: Uuid,
    command: &EmbeddingJobBatchClaim,
    item: &EmbeddingJobBatchClaimItem,
) -> Result<bool, LedgerError> {
    Ok(stored_state_is_canonical(event)?
        && event.embedding_job_id == item.embedding_job_id
        && event.process_instance_id.as_deref() == Some(process_instance_id.to_string().as_str())
        && event.state == "claimed"
        && event.stable_error_class.is_none()
        && event.created_at_unix_ms == command.observed_at_unix_ms
        && event.lease_token.as_deref() == Some(command.lease_token.to_string().as_str())
        && event.lease_expires_at_unix_ms == Some(command.lease_expires_at_unix_ms))
}

fn orphan_event_matches_retry(
    orphan: &StoredJobState,
    claim: &StoredJobState,
    process_instance_id: Uuid,
    command: &EmbeddingJobBatchClaim,
) -> Result<bool, LedgerError> {
    Ok(stored_state_is_canonical(orphan)?
        && orphan.embedding_job_id == claim.embedding_job_id
        && orphan.process_instance_id.as_deref() == Some(process_instance_id.to_string().as_str())
        && orphan.state == "orphaned_in_flight"
        && orphan.created_at_unix_ms == command.observed_at_unix_ms
        && orphan
            .attempt_generation
            .checked_add(1)
            .is_some_and(|generation| generation == claim.attempt_generation)
        && orphan.attempt_count.and_then(|count| count.checked_add(1)) == claim.attempt_count)
}

pub(crate) fn complete_embedding_job_batch_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &EmbeddingJobBatchCompletion,
) -> Result<EmbeddingJobBatchCompletionAck, LedgerError> {
    let Some(max_batch_size) =
        verified_profile_batch_size(transaction, project_uuid, &command.vector_space_id)?
    else {
        return Ok(EmbeddingJobBatchCompletionAck::VectorSpaceNotFound);
    };
    if command.items.len() > max_batch_size {
        return Ok(EmbeddingJobBatchCompletionAck::BatchTooLarge { max_batch_size });
    }

    let mut plans = Vec::with_capacity(command.items.len());
    let mut retries = Vec::with_capacity(command.items.len());
    let mut existing_event_count = 0_usize;
    for item in &command.items {
        let Some(stored) = load_job(transaction, &item.embedding_job_id)? else {
            return Ok(EmbeddingJobBatchCompletionAck::JobNotFound {
                embedding_job_id: item.embedding_job_id.clone(),
            });
        };
        let Some(verified) = verify_job(transaction, project_uuid, stored)? else {
            return Ok(EmbeddingJobBatchCompletionAck::Conflict);
        };
        if verified.stored.vector_space_id != command.vector_space_id.as_str()
            || verified.stored.content_hash != item.content_hash
            || item.vector.vector_space_id() != &command.vector_space_id
            || !job_belongs_to_project(transaction, project_uuid, &verified.stored)?
        {
            return Ok(EmbeddingJobBatchCompletionAck::Conflict);
        }
        let cache = load_embedding_cache(
            transaction,
            project_uuid,
            &command.vector_space_id,
            &verified.stored.canonical_query_hash,
        )?;
        let event = load_state_by_event_id(transaction, item.completed_state_event_id)?;
        if let Some(event) = event.as_ref() {
            existing_event_count += 1;
            let Some(cache) = cache else {
                return Ok(EmbeddingJobBatchCompletionAck::Conflict);
            };
            if !batch_completion_event_matches(
                event,
                process_instance_id,
                command,
                item,
                &verified,
            )? || !cache_matches_completion(&cache, command, item)
            {
                return Ok(EmbeddingJobBatchCompletionAck::Conflict);
            }
            retries.push(EmbeddingJobBatchCompletionResult {
                job: snapshot(&verified.stored),
                embedding: cache,
                state_event_hash: event.canonical_payload_hash.clone(),
            });
            continue;
        }
        if cache.is_some()
            || embedding_id_exists(transaction, item.embedding_id)?
            || verified
                .states
                .last()
                .is_none_or(|latest| command.completed_at_unix_ms < latest.created_at_unix_ms)
        {
            return Ok(EmbeddingJobBatchCompletionAck::Conflict);
        }
        let Some((owner, token, expires_at)) = current_lease(&verified.stored)? else {
            return Ok(EmbeddingJobBatchCompletionAck::StaleLease {
                embedding_job_id: item.embedding_job_id.clone(),
            });
        };
        if owner != process_instance_id
            || token != command.lease_token
            || verified.stored.attempt_generation != item.attempt_generation
            || command.completed_at_unix_ms >= expires_at
            || load_state_for_transition(
                transaction,
                &item.embedding_job_id,
                item.attempt_generation,
                "completed",
            )?
            .is_some()
        {
            return Ok(EmbeddingJobBatchCompletionAck::StaleLease {
                embedding_job_id: item.embedding_job_id.clone(),
            });
        }
        plans.push((item, verified, expires_at));
    }
    if existing_event_count != 0 {
        return Ok(if existing_event_count == command.items.len() {
            EmbeddingJobBatchCompletionAck::AlreadyApplied(retries)
        } else {
            EmbeddingJobBatchCompletionAck::Conflict
        });
    }

    let mut completed = Vec::with_capacity(plans.len());
    for (item, verified, expires_at) in plans {
        let cache_write = EmbeddingCacheWrite {
            embedding_id: item.embedding_id,
            project_uuid,
            vector_space_id: command.vector_space_id.clone(),
            canonical_query_hash: verified.stored.canonical_query_hash.clone(),
            content_hash: item.content_hash.clone(),
            vector: item.vector.clone(),
            source: EmbeddingCacheSource::Provider,
            created_at_unix_ms: command.completed_at_unix_ms,
        };
        let embedding = match upsert_embedding_cache(transaction, &cache_write)? {
            EmbeddingCacheUpsertAck::Applied(snapshot)
            | EmbeddingCacheUpsertAck::AlreadyExists(snapshot)
                if cache_matches_completion(&snapshot, command, item) =>
            {
                snapshot
            }
            EmbeddingCacheUpsertAck::Applied(_)
            | EmbeddingCacheUpsertAck::AlreadyExists(_)
            | EmbeddingCacheUpsertAck::AuthorityNotFound
            | EmbeddingCacheUpsertAck::Conflict => {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        };
        let new_hash = job_hash(
            &verified.stored.embedding_job_id,
            &verified.stored.vector_space_id,
            &verified.stored.canonical_query_hash,
            &verified.stored.content_hash,
            None,
            None,
            None,
            verified.stored.attempt_generation,
            verified.stored.attempt_count,
            verified.stored.next_eligible_at_unix_ms,
            None,
            None,
            false,
            verified.stored.reset_actor.as_deref(),
            verified.stored.reset_reason.as_deref(),
            verified.stored.created_at_unix_ms,
        )?;
        let updated = transaction
            .execute(
                "UPDATE embedding_jobs
                 SET lease_owner_process_instance_id = NULL, lease_token = NULL,
                     lease_expires_at_unix_ms = NULL, canonical_payload_hash = ?1
                 WHERE embedding_job_id = ?2 AND canonical_payload_hash = ?3",
                params![
                    new_hash,
                    item.embedding_job_id,
                    verified.stored.canonical_payload_hash,
                ],
            )
            .map_err(database_error)?;
        if updated != 1 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let event_hash = state_hash(
            item.completed_state_event_id,
            &item.embedding_job_id,
            Some(process_instance_id),
            "completed",
            item.attempt_generation,
            None,
            command.completed_at_unix_ms,
            Some(command.lease_token),
            Some(expires_at),
            verified.stored.attempt_count,
            verified.stored.next_eligible_at_unix_ms,
            None,
            None,
        )?;
        insert_state(
            transaction,
            item.completed_state_event_id,
            &item.embedding_job_id,
            process_instance_id,
            "completed",
            item.attempt_generation,
            None,
            command.completed_at_unix_ms,
            Some(command.lease_token),
            Some(expires_at),
            verified.stored.attempt_count,
            verified.stored.next_eligible_at_unix_ms,
            None,
            None,
            &event_hash,
        )?;
        let stored = load_job(transaction, &item.embedding_job_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        completed.push(EmbeddingJobBatchCompletionResult {
            job: snapshot(&stored),
            embedding,
            state_event_hash: event_hash,
        });
    }
    Ok(EmbeddingJobBatchCompletionAck::Applied(completed))
}

fn batch_completion_event_matches(
    event: &StoredJobState,
    process_instance_id: Uuid,
    command: &EmbeddingJobBatchCompletion,
    item: &EmbeddingJobBatchCompletionItem,
    verified: &VerifiedJob,
) -> Result<bool, LedgerError> {
    Ok(stored_state_is_canonical(event)?
        && event.embedding_job_id == item.embedding_job_id
        && event.process_instance_id.as_deref() == Some(process_instance_id.to_string().as_str())
        && event.state == "completed"
        && event.attempt_generation == item.attempt_generation
        && event.attempt_count == Some(verified.stored.attempt_count)
        && event.stable_error_class.is_none()
        && event.created_at_unix_ms == command.completed_at_unix_ms
        && event.lease_token.as_deref() == Some(command.lease_token.to_string().as_str())
        && verified.states.last().is_some_and(|latest| {
            latest.embedding_job_state_event_id == event.embedding_job_state_event_id
        }))
}

fn cache_matches_completion(
    cache: &EmbeddingCacheSnapshot,
    command: &EmbeddingJobBatchCompletion,
    item: &EmbeddingJobBatchCompletionItem,
) -> bool {
    cache.embedding_id == item.embedding_id
        && cache.vector_space_id == command.vector_space_id
        && cache.content_hash == item.content_hash
        && cache.vector.bitwise_eq(&item.vector)
        && cache.source == EmbeddingCacheSource::Provider
        && cache.created_at_unix_ms == command.completed_at_unix_ms
}

fn embedding_id_exists(connection: &Connection, embedding_id: Uuid) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM embeddings WHERE embedding_id = ?1)",
            params![embedding_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)
}

pub(crate) fn resolve_embedding_job_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &EmbeddingJobResolution,
) -> Result<EmbeddingJobResolutionAck, LedgerError> {
    let Some(stored) = load_job(transaction, &command.embedding_job_id)? else {
        return Ok(EmbeddingJobResolutionAck::NotFound);
    };
    let Some(verified) = verify_job(transaction, project_uuid, stored)? else {
        return Ok(EmbeddingJobResolutionAck::Conflict);
    };
    if verified.stored.content_hash != command.content_hash
        || !job_belongs_to_project(transaction, project_uuid, &verified.stored)?
    {
        return Ok(EmbeddingJobResolutionAck::Conflict);
    }
    let Some((resolved_state, stable_error_class, next_eligible_at_unix_ms)) = resolution_facts(
        command,
        verified.stored.attempt_count,
        verified.stored.next_eligible_at_unix_ms,
    ) else {
        return Ok(EmbeddingJobResolutionAck::Conflict);
    };
    if let Some(event) = load_state_by_event_id(transaction, command.embedding_job_state_event_id)?
    {
        if !resolution_event_matches(
            &event,
            process_instance_id,
            command,
            resolved_state,
            stable_error_class,
            next_eligible_at_unix_ms,
            &verified,
        )? {
            return Ok(EmbeddingJobResolutionAck::Conflict);
        }
        return Ok(EmbeddingJobResolutionAck::AlreadyApplied {
            job: snapshot(&verified.stored),
            state: resolved_state,
            state_event_hash: event.canonical_payload_hash,
        });
    }
    let Some((owner, token, expires_at)) = current_lease(&verified.stored)? else {
        return Ok(EmbeddingJobResolutionAck::StaleLease);
    };
    if owner != process_instance_id
        || token != command.lease_token
        || verified.stored.attempt_generation != command.attempt_generation
        || command.resolved_at_unix_ms >= expires_at
        || verified
            .states
            .last()
            .is_none_or(|latest| command.resolved_at_unix_ms < latest.created_at_unix_ms)
        || load_state_for_transition(
            transaction,
            &command.embedding_job_id,
            command.attempt_generation,
            resolved_state.as_str(),
        )?
        .is_some()
    {
        return Ok(EmbeddingJobResolutionAck::StaleLease);
    }
    let terminal_error_class = match resolved_state {
        EmbeddingJobResolvedState::TerminalFailure | EmbeddingJobResolvedState::Quarantined => {
            stable_error_class
        }
        EmbeddingJobResolvedState::Released | EmbeddingJobResolvedState::RetryScheduled => None,
    };
    let new_hash = job_hash(
        &verified.stored.embedding_job_id,
        &verified.stored.vector_space_id,
        &verified.stored.canonical_query_hash,
        &verified.stored.content_hash,
        None,
        None,
        None,
        verified.stored.attempt_generation,
        verified.stored.attempt_count,
        next_eligible_at_unix_ms,
        terminal_error_class,
        None,
        false,
        verified.stored.reset_actor.as_deref(),
        verified.stored.reset_reason.as_deref(),
        verified.stored.created_at_unix_ms,
    )?;
    let updated = transaction
        .execute(
            "UPDATE embedding_jobs
             SET lease_owner_process_instance_id = NULL, lease_token = NULL,
                 lease_expires_at_unix_ms = NULL, next_eligible_at_unix_ms = ?1,
                 terminal_error_class = ?2, failure_propagation_cursor = NULL,
                 failure_propagation_complete = 0, canonical_payload_hash = ?3
             WHERE embedding_job_id = ?4 AND canonical_payload_hash = ?5",
            params![
                next_eligible_at_unix_ms,
                terminal_error_class,
                new_hash,
                command.embedding_job_id,
                verified.stored.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if updated != 1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let event_hash = state_hash(
        command.embedding_job_state_event_id,
        &command.embedding_job_id,
        Some(process_instance_id),
        resolved_state.as_str(),
        command.attempt_generation,
        stable_error_class,
        command.resolved_at_unix_ms,
        Some(command.lease_token),
        Some(expires_at),
        verified.stored.attempt_count,
        next_eligible_at_unix_ms,
        None,
        None,
    )?;
    insert_state(
        transaction,
        command.embedding_job_state_event_id,
        &command.embedding_job_id,
        process_instance_id,
        resolved_state.as_str(),
        command.attempt_generation,
        stable_error_class,
        command.resolved_at_unix_ms,
        Some(command.lease_token),
        Some(expires_at),
        verified.stored.attempt_count,
        next_eligible_at_unix_ms,
        None,
        None,
        &event_hash,
    )?;
    let stored = load_job(transaction, &command.embedding_job_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    Ok(EmbeddingJobResolutionAck::Applied {
        job: snapshot(&stored),
        state: resolved_state,
        state_event_hash: event_hash,
    })
}

fn resolution_facts(
    command: &EmbeddingJobResolution,
    attempt_count: i64,
    current_next_eligible_at_unix_ms: i64,
) -> Option<(EmbeddingJobResolvedState, Option<&str>, i64)> {
    let facts = match &command.kind {
        EmbeddingJobResolutionKind::Released if attempt_count < 5 => (
            EmbeddingJobResolvedState::Released,
            None,
            command.resolved_at_unix_ms,
        ),
        EmbeddingJobResolutionKind::RetryScheduled {
            stable_error_class,
            next_eligible_at_unix_ms,
        } if attempt_count < 5 => (
            EmbeddingJobResolvedState::RetryScheduled,
            Some(stable_error_class.as_str()),
            *next_eligible_at_unix_ms,
        ),
        EmbeddingJobResolutionKind::RetryScheduled {
            stable_error_class, ..
        } if attempt_count == 5 => (
            EmbeddingJobResolvedState::Quarantined,
            Some(stable_error_class.as_str()),
            current_next_eligible_at_unix_ms,
        ),
        EmbeddingJobResolutionKind::TerminalFailure { stable_error_class } => (
            EmbeddingJobResolvedState::TerminalFailure,
            Some(stable_error_class.as_str()),
            current_next_eligible_at_unix_ms,
        ),
        EmbeddingJobResolutionKind::Quarantined { stable_error_class }
            if (1..=5).contains(&attempt_count) =>
        {
            (
                EmbeddingJobResolvedState::Quarantined,
                Some(stable_error_class.as_str()),
                current_next_eligible_at_unix_ms,
            )
        }
        _ => return None,
    };
    Some(facts)
}

#[allow(clippy::too_many_arguments)]
fn resolution_event_matches(
    event: &StoredJobState,
    process_instance_id: Uuid,
    command: &EmbeddingJobResolution,
    resolved_state: EmbeddingJobResolvedState,
    stable_error_class: Option<&str>,
    next_eligible_at_unix_ms: i64,
    verified: &VerifiedJob,
) -> Result<bool, LedgerError> {
    Ok(stored_state_is_canonical(event)?
        && event.embedding_job_id == command.embedding_job_id
        && event.process_instance_id.as_deref() == Some(process_instance_id.to_string().as_str())
        && event.state == resolved_state.as_str()
        && event.attempt_generation == command.attempt_generation
        && event.attempt_count == Some(verified.stored.attempt_count)
        && event.stable_error_class.as_deref() == stable_error_class
        && event.created_at_unix_ms == command.resolved_at_unix_ms
        && event.lease_token.as_deref() == Some(command.lease_token.to_string().as_str())
        && event.next_eligible_at_unix_ms == Some(next_eligible_at_unix_ms)
        && verified.states.last().is_some_and(|latest| {
            latest.embedding_job_state_event_id == event.embedding_job_state_event_id
        }))
}

pub(crate) fn update_embedding_failure_propagation_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    command: &EmbeddingFailurePropagationUpdate,
) -> Result<EmbeddingFailurePropagationAck, LedgerError> {
    let Some(stored) = load_job(transaction, &command.embedding_job_id)? else {
        return Ok(EmbeddingFailurePropagationAck::NotFound);
    };
    let Some(verified) = verify_job(transaction, project_uuid, stored)? else {
        return Ok(EmbeddingFailurePropagationAck::Conflict);
    };
    if verified.stored.canonical_payload_hash != command.expected_canonical_payload_hash {
        let expected_cursor = command.expected_cursor.map(|cursor| cursor.to_string());
        let expected_pre_update_hash = job_hash(
            &verified.stored.embedding_job_id,
            &verified.stored.vector_space_id,
            &verified.stored.canonical_query_hash,
            &verified.stored.content_hash,
            None,
            None,
            None,
            command.expected_attempt_generation,
            verified.stored.attempt_count,
            verified.stored.next_eligible_at_unix_ms,
            verified.stored.terminal_error_class.as_deref(),
            expected_cursor.as_deref(),
            false,
            verified.stored.reset_actor.as_deref(),
            verified.stored.reset_reason.as_deref(),
            verified.stored.created_at_unix_ms,
        )?;
        let desired_cursor = command.next_cursor.map(|cursor| cursor.to_string());
        return Ok(
            if expected_pre_update_hash == command.expected_canonical_payload_hash
                && verified.stored.attempt_generation == command.expected_attempt_generation
                && verified.stored.failure_propagation_cursor == desired_cursor
                && verified.stored.failure_propagation_complete == command.complete
            {
                EmbeddingFailurePropagationAck::AlreadyApplied(snapshot(&verified.stored))
            } else {
                EmbeddingFailurePropagationAck::Stale
            },
        );
    }
    if verified.stored.attempt_generation != command.expected_attempt_generation {
        return Ok(EmbeddingFailurePropagationAck::Stale);
    }
    if !matches!(
        verified.states.last().map(|state| state.state.as_str()),
        Some("terminal_failure" | "quarantined")
    ) || verified.stored.terminal_error_class.is_none()
    {
        return Ok(EmbeddingFailurePropagationAck::NotTerminal);
    }
    let expected_cursor = command.expected_cursor.map(|cursor| cursor.to_string());
    if verified.stored.failure_propagation_cursor != expected_cursor
        || verified.stored.failure_propagation_complete
    {
        return Ok(EmbeddingFailurePropagationAck::Stale);
    }
    let next_cursor = command.next_cursor.map(|cursor| cursor.to_string());
    let new_hash = job_hash(
        &verified.stored.embedding_job_id,
        &verified.stored.vector_space_id,
        &verified.stored.canonical_query_hash,
        &verified.stored.content_hash,
        None,
        None,
        None,
        verified.stored.attempt_generation,
        verified.stored.attempt_count,
        verified.stored.next_eligible_at_unix_ms,
        verified.stored.terminal_error_class.as_deref(),
        next_cursor.as_deref(),
        command.complete,
        verified.stored.reset_actor.as_deref(),
        verified.stored.reset_reason.as_deref(),
        verified.stored.created_at_unix_ms,
    )?;
    let updated = transaction
        .execute(
            "UPDATE embedding_jobs
             SET failure_propagation_cursor = ?1, failure_propagation_complete = ?2,
                 canonical_payload_hash = ?3
             WHERE embedding_job_id = ?4 AND canonical_payload_hash = ?5",
            params![
                next_cursor,
                command.complete,
                new_hash,
                command.embedding_job_id,
                command.expected_canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if updated != 1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let stored = load_job(transaction, &command.embedding_job_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    Ok(EmbeddingFailurePropagationAck::Applied(snapshot(&stored)))
}

pub(crate) fn reset_embedding_job_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &EmbeddingJobReset,
) -> Result<EmbeddingJobResetAck, LedgerError> {
    let Some(stored) = load_job(transaction, &command.embedding_job_id)? else {
        return Ok(EmbeddingJobResetAck::NotFound);
    };
    let Some(verified) = verify_job(transaction, project_uuid, stored)? else {
        return Ok(EmbeddingJobResetAck::Conflict);
    };
    if let Some(event) = load_state_by_event_id(transaction, command.embedding_job_state_event_id)?
    {
        if !reset_event_matches(&event, process_instance_id, command, &verified)? {
            return Ok(EmbeddingJobResetAck::Conflict);
        }
        return Ok(EmbeddingJobResetAck::AlreadyApplied {
            job: snapshot(&verified.stored),
            state_event_hash: event.canonical_payload_hash,
        });
    }
    if verified.stored.canonical_payload_hash != command.expected_canonical_payload_hash
        || verified.stored.attempt_generation != command.expected_attempt_generation
    {
        return Ok(EmbeddingJobResetAck::Stale);
    }
    if !matches!(
        verified.states.last().map(|state| state.state.as_str()),
        Some("terminal_failure" | "quarantined")
    ) || verified.stored.terminal_error_class.is_none()
        || current_lease(&verified.stored)?.is_some()
    {
        return Ok(EmbeddingJobResetAck::NotTerminal);
    }
    if !verified.stored.failure_propagation_complete {
        return Ok(EmbeddingJobResetAck::PropagationIncomplete);
    }
    let latest = verified
        .states
        .last()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if command.reset_at_unix_ms < latest.created_at_unix_ms {
        return Ok(EmbeddingJobResetAck::Stale);
    }
    let attempt_generation = verified
        .stored
        .attempt_generation
        .checked_add(1)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if load_state_for_transition(
        transaction,
        &command.embedding_job_id,
        attempt_generation,
        "reset",
    )?
    .is_some()
    {
        return Ok(EmbeddingJobResetAck::Conflict);
    }
    let new_hash = job_hash(
        &verified.stored.embedding_job_id,
        &verified.stored.vector_space_id,
        &verified.stored.canonical_query_hash,
        &verified.stored.content_hash,
        None,
        None,
        None,
        attempt_generation,
        0,
        command.reset_at_unix_ms,
        None,
        None,
        false,
        Some(&command.reset_actor),
        Some(&command.reset_reason),
        verified.stored.created_at_unix_ms,
    )?;
    let updated = transaction
        .execute(
            "UPDATE embedding_jobs
             SET lease_owner_process_instance_id = NULL, lease_token = NULL,
                 lease_expires_at_unix_ms = NULL, attempt_generation = ?1,
                 attempt_count = 0, next_eligible_at_unix_ms = ?2,
                 terminal_error_class = NULL, failure_propagation_cursor = NULL,
                 failure_propagation_complete = 0, reset_actor = ?3, reset_reason = ?4,
                 canonical_payload_hash = ?5
             WHERE embedding_job_id = ?6 AND canonical_payload_hash = ?7",
            params![
                attempt_generation,
                command.reset_at_unix_ms,
                command.reset_actor,
                command.reset_reason,
                new_hash,
                command.embedding_job_id,
                command.expected_canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if updated != 1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let event_hash = state_hash(
        command.embedding_job_state_event_id,
        &command.embedding_job_id,
        Some(process_instance_id),
        "reset",
        attempt_generation,
        None,
        command.reset_at_unix_ms,
        None,
        None,
        0,
        command.reset_at_unix_ms,
        Some(&command.reset_actor),
        Some(&command.reset_reason),
    )?;
    insert_state(
        transaction,
        command.embedding_job_state_event_id,
        &command.embedding_job_id,
        process_instance_id,
        "reset",
        attempt_generation,
        None,
        command.reset_at_unix_ms,
        None,
        None,
        0,
        command.reset_at_unix_ms,
        Some(&command.reset_actor),
        Some(&command.reset_reason),
        &event_hash,
    )?;
    let stored = load_job(transaction, &command.embedding_job_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    Ok(EmbeddingJobResetAck::Applied {
        job: snapshot(&stored),
        state_event_hash: event_hash,
    })
}

fn reset_event_matches(
    event: &StoredJobState,
    process_instance_id: Uuid,
    command: &EmbeddingJobReset,
    verified: &VerifiedJob,
) -> Result<bool, LedgerError> {
    Ok(stored_state_is_canonical(event)?
        && event.embedding_job_id == command.embedding_job_id
        && event.process_instance_id.as_deref() == Some(process_instance_id.to_string().as_str())
        && event.state == "reset"
        && command
            .expected_attempt_generation
            .checked_add(1)
            .is_some_and(|generation| event.attempt_generation == generation)
        && event.attempt_count == Some(0)
        && event.stable_error_class.is_none()
        && event.created_at_unix_ms == command.reset_at_unix_ms
        && event.lease_token.is_none()
        && event.lease_expires_at_unix_ms.is_none()
        && event.reset_actor.as_deref() == Some(command.reset_actor.as_str())
        && event.reset_reason.as_deref() == Some(command.reset_reason.as_str())
        && verified.states.last().is_some_and(|latest| {
            latest.embedding_job_state_event_id == event.embedding_job_state_event_id
        }))
}

#[cfg(test)]
fn claim_embedding_job_in_transaction(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &EmbeddingJobClaim,
) -> Result<EmbeddingJobClaimAck, LedgerError> {
    let Some(stored) = load_job(connection, &command.embedding_job_id)? else {
        return Ok(EmbeddingJobClaimAck::NotFound);
    };
    let Some(verified) = verify_job(connection, project_uuid, stored)? else {
        return Ok(EmbeddingJobClaimAck::Conflict);
    };
    if !job_belongs_to_project(connection, project_uuid, &verified.stored)? {
        return Ok(EmbeddingJobClaimAck::Conflict);
    }
    if let Some(event) = load_state_by_event_id(connection, command.embedding_job_state_event_id)? {
        if !claim_event_matches_command(&event, process_instance_id, command)? {
            return Ok(EmbeddingJobClaimAck::Conflict);
        }
        if claim_event_matches_current(&event, &verified.stored)? {
            return Ok(EmbeddingJobClaimAck::AlreadyApplied(lease_from_current(
                &verified.stored,
                &event,
            )?));
        }
        return Ok(
            if verified
                .states
                .last()
                .is_some_and(|latest| latest.state == "completed")
            {
                EmbeddingJobClaimAck::Terminal
            } else if let Some((_, _, lease_expires_at_unix_ms)) = current_lease(&verified.stored)?
            {
                EmbeddingJobClaimAck::LeaseHeld {
                    lease_expires_at_unix_ms,
                }
            } else {
                EmbeddingJobClaimAck::Conflict
            },
        );
    }
    if verified
        .states
        .last()
        .is_some_and(|state| state.state == "completed")
    {
        return Ok(EmbeddingJobClaimAck::Terminal);
    }
    if verified
        .states
        .last()
        .is_none_or(|state| command.observed_at_unix_ms < state.created_at_unix_ms)
    {
        return Ok(EmbeddingJobClaimAck::Conflict);
    }
    if verified.stored.next_eligible_at_unix_ms > command.observed_at_unix_ms {
        return Ok(EmbeddingJobClaimAck::NotEligible {
            next_eligible_at_unix_ms: verified.stored.next_eligible_at_unix_ms,
        });
    }

    let reclaimed = match current_lease(&verified.stored)? {
        None => None,
        Some((owner, token, expires_at)) => {
            if token == command.lease_token {
                return Ok(EmbeddingJobClaimAck::Conflict);
            }
            let owner_status = verified_process_status_at(
                connection,
                project_uuid,
                owner,
                command.observed_at_unix_ms,
            )?;
            if expires_at > command.observed_at_unix_ms && owner_status == ProcessStatusAt::Live {
                return Ok(EmbeddingJobClaimAck::LeaseHeld {
                    lease_expires_at_unix_ms: expires_at,
                });
            }
            if owner_status == ProcessStatusAt::Invalid {
                return Ok(EmbeddingJobClaimAck::Conflict);
            }
            Some((token, expires_at))
        }
    };
    if verified.states.iter().any(|state| {
        state.state == "claimed"
            && state.lease_token.as_deref() == Some(command.lease_token.to_string().as_str())
    }) {
        return Ok(EmbeddingJobClaimAck::Conflict);
    }
    let attempt_generation = verified
        .stored
        .attempt_generation
        .checked_add(1)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let attempt_count = verified
        .stored
        .attempt_count
        .checked_add(1)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if attempt_generation != attempt_count {
        return Ok(EmbeddingJobClaimAck::Conflict);
    }
    if load_state_for_transition(
        connection,
        &command.embedding_job_id,
        attempt_generation,
        "claimed",
    )?
    .is_some()
    {
        return Ok(EmbeddingJobClaimAck::Conflict);
    }
    if let Some((old_token, old_expiry)) = reclaimed {
        let orphan_event_id = Uuid::now_v7();
        let orphan_hash = state_hash(
            orphan_event_id,
            &command.embedding_job_id,
            Some(process_instance_id),
            "orphaned_in_flight",
            verified.stored.attempt_generation,
            None,
            command.observed_at_unix_ms,
            Some(old_token),
            Some(old_expiry),
            verified.stored.attempt_count,
            verified.stored.next_eligible_at_unix_ms,
            None,
            None,
        )?;
        insert_state(
            connection,
            orphan_event_id,
            &command.embedding_job_id,
            process_instance_id,
            "orphaned_in_flight",
            verified.stored.attempt_generation,
            None,
            command.observed_at_unix_ms,
            Some(old_token),
            Some(old_expiry),
            verified.stored.attempt_count,
            verified.stored.next_eligible_at_unix_ms,
            None,
            None,
            &orphan_hash,
        )?;
    }
    let new_hash = job_hash(
        &verified.stored.embedding_job_id,
        &verified.stored.vector_space_id,
        &verified.stored.canonical_query_hash,
        &verified.stored.content_hash,
        Some(process_instance_id),
        Some(command.lease_token),
        Some(command.lease_expires_at_unix_ms),
        attempt_generation,
        attempt_count,
        verified.stored.next_eligible_at_unix_ms,
        verified.stored.terminal_error_class.as_deref(),
        verified.stored.failure_propagation_cursor.as_deref(),
        verified.stored.failure_propagation_complete,
        verified.stored.reset_actor.as_deref(),
        verified.stored.reset_reason.as_deref(),
        verified.stored.created_at_unix_ms,
    )?;
    let updated = connection
        .execute(
            "UPDATE embedding_jobs
             SET lease_owner_process_instance_id = ?1, lease_token = ?2,
                 lease_expires_at_unix_ms = ?3, attempt_generation = ?4,
                 attempt_count = ?5, canonical_payload_hash = ?6
             WHERE embedding_job_id = ?7 AND canonical_payload_hash = ?8",
            params![
                process_instance_id.to_string(),
                command.lease_token.to_string(),
                command.lease_expires_at_unix_ms,
                attempt_generation,
                attempt_count,
                new_hash,
                command.embedding_job_id,
                verified.stored.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if updated != 1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let event_hash = state_hash(
        command.embedding_job_state_event_id,
        &command.embedding_job_id,
        Some(process_instance_id),
        "claimed",
        attempt_generation,
        None,
        command.observed_at_unix_ms,
        Some(command.lease_token),
        Some(command.lease_expires_at_unix_ms),
        attempt_count,
        verified.stored.next_eligible_at_unix_ms,
        None,
        None,
    )?;
    insert_state(
        connection,
        command.embedding_job_state_event_id,
        &command.embedding_job_id,
        process_instance_id,
        "claimed",
        attempt_generation,
        None,
        command.observed_at_unix_ms,
        Some(command.lease_token),
        Some(command.lease_expires_at_unix_ms),
        attempt_count,
        verified.stored.next_eligible_at_unix_ms,
        None,
        None,
        &event_hash,
    )?;
    let stored = load_job(connection, &command.embedding_job_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let event = load_state_by_event_id(connection, command.embedding_job_state_event_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let lease = lease_from_current(&stored, &event)?;
    Ok(if reclaimed.is_some() {
        EmbeddingJobClaimAck::Reclaimed(lease)
    } else {
        EmbeddingJobClaimAck::Claimed(lease)
    })
}

#[cfg(test)]
fn complete_embedding_job_in_transaction(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &EmbeddingJobCompletion,
) -> Result<EmbeddingJobCompletionAck, LedgerError> {
    let Some(stored) = load_job(connection, &command.embedding_job_id)? else {
        return Ok(EmbeddingJobCompletionAck::NotFound);
    };
    let Some(verified) = verify_job(connection, project_uuid, stored)? else {
        return Ok(EmbeddingJobCompletionAck::Conflict);
    };
    if !job_belongs_to_project(connection, project_uuid, &verified.stored)? {
        return Ok(EmbeddingJobCompletionAck::Conflict);
    }
    if let Some(event) = load_state_by_event_id(connection, command.embedding_job_state_event_id)? {
        if verified.stored.content_hash == command.content_hash
            && completion_event_matches(&event, process_instance_id, command)?
            && verified.states.last().is_some_and(|latest| {
                latest.embedding_job_state_event_id == event.embedding_job_state_event_id
            })
        {
            return Ok(EmbeddingJobCompletionAck::AlreadyApplied {
                job: snapshot(&verified.stored),
                state_event_hash: event.canonical_payload_hash,
            });
        }
        return Ok(EmbeddingJobCompletionAck::Conflict);
    }
    if verified
        .states
        .last()
        .is_some_and(|state| state.state == "completed")
    {
        return Ok(EmbeddingJobCompletionAck::StaleLease);
    }
    if verified
        .states
        .last()
        .is_none_or(|state| command.completed_at_unix_ms < state.created_at_unix_ms)
    {
        return Ok(EmbeddingJobCompletionAck::StaleLease);
    }
    let Some((owner, token, expires_at)) = current_lease(&verified.stored)? else {
        return Ok(EmbeddingJobCompletionAck::StaleLease);
    };
    if owner != process_instance_id
        || token != command.lease_token
        || verified.stored.attempt_generation != command.attempt_generation
        || verified.stored.attempt_count != command.attempt_generation
        || verified.stored.content_hash != command.content_hash
        || command.completed_at_unix_ms >= expires_at
    {
        return Ok(EmbeddingJobCompletionAck::StaleLease);
    }
    if load_state_for_transition(
        connection,
        &command.embedding_job_id,
        command.attempt_generation,
        "completed",
    )?
    .is_some()
    {
        return Ok(EmbeddingJobCompletionAck::Conflict);
    }
    let new_hash = job_hash(
        &verified.stored.embedding_job_id,
        &verified.stored.vector_space_id,
        &verified.stored.canonical_query_hash,
        &verified.stored.content_hash,
        None,
        None,
        None,
        verified.stored.attempt_generation,
        verified.stored.attempt_count,
        verified.stored.next_eligible_at_unix_ms,
        verified.stored.terminal_error_class.as_deref(),
        verified.stored.failure_propagation_cursor.as_deref(),
        verified.stored.failure_propagation_complete,
        verified.stored.reset_actor.as_deref(),
        verified.stored.reset_reason.as_deref(),
        verified.stored.created_at_unix_ms,
    )?;
    let updated = connection
        .execute(
            "UPDATE embedding_jobs
             SET lease_owner_process_instance_id = NULL, lease_token = NULL,
                 lease_expires_at_unix_ms = NULL, canonical_payload_hash = ?1
             WHERE embedding_job_id = ?2 AND canonical_payload_hash = ?3",
            params![
                new_hash,
                command.embedding_job_id,
                verified.stored.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if updated != 1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let event_hash = state_hash(
        command.embedding_job_state_event_id,
        &command.embedding_job_id,
        Some(process_instance_id),
        "completed",
        command.attempt_generation,
        None,
        command.completed_at_unix_ms,
        Some(command.lease_token),
        Some(expires_at),
        verified.stored.attempt_count,
        verified.stored.next_eligible_at_unix_ms,
        None,
        None,
    )?;
    insert_state(
        connection,
        command.embedding_job_state_event_id,
        &command.embedding_job_id,
        process_instance_id,
        "completed",
        command.attempt_generation,
        None,
        command.completed_at_unix_ms,
        Some(command.lease_token),
        Some(expires_at),
        verified.stored.attempt_count,
        verified.stored.next_eligible_at_unix_ms,
        None,
        None,
        &event_hash,
    )?;
    let stored = load_job(connection, &command.embedding_job_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    Ok(EmbeddingJobCompletionAck::Applied {
        job: snapshot(&stored),
        state_event_hash: event_hash,
    })
}

fn verify_job(
    connection: &Connection,
    project_uuid: Uuid,
    stored: StoredJob,
) -> Result<Option<VerifiedJob>, LedgerError> {
    if !stored_job_is_canonical(&stored)? {
        return Ok(None);
    }
    let states = load_job_states(connection, &stored.embedding_job_id)?;
    if states.is_empty() || !state_chain_is_canonical(connection, project_uuid, &stored, &states)? {
        return Ok(None);
    }
    Ok(Some(VerifiedJob { stored, states }))
}

/// Load one optional job only after its row, event chain, and project authority verify.
pub(crate) fn load_verified_embedding_job(
    connection: &Connection,
    project_uuid: Uuid,
    embedding_job_id: &str,
) -> Result<Option<EmbeddingJobSnapshot>, LedgerError> {
    validate_uuid_v7(project_uuid)?;
    validate_sha256(embedding_job_id)?;
    let Some(stored) = load_job(connection, embedding_job_id)? else {
        return Ok(None);
    };
    let verified = verify_job(connection, project_uuid, stored)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if !job_belongs_to_project(connection, project_uuid, &verified.stored)? {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(Some(snapshot(&verified.stored)))
}

/// Load the current verified lease only when it still carries the expected claim token.
pub(crate) fn load_verified_embedding_lease(
    connection: &Connection,
    project_uuid: Uuid,
    embedding_job_id: &str,
    expected_lease_token: Uuid,
) -> Result<Option<EmbeddingJobLease>, LedgerError> {
    validate_uuid_v7(project_uuid)?;
    validate_uuid_v7(expected_lease_token)?;
    validate_sha256(embedding_job_id)?;
    let Some(stored) = load_job(connection, embedding_job_id)? else {
        return Ok(None);
    };
    let verified = verify_job(connection, project_uuid, stored)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if !job_belongs_to_project(connection, project_uuid, &verified.stored)? {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let Some((owner, lease_token, lease_expires_at_unix_ms)) = current_lease(&verified.stored)?
    else {
        return Ok(None);
    };
    if lease_token != expected_lease_token {
        return Ok(None);
    }
    let event = verified
        .states
        .last()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if event.state != "claimed"
        || event.process_instance_id.as_deref() != Some(owner.to_string().as_str())
        || event.attempt_generation != verified.stored.attempt_generation
        || event.lease_token.as_deref() != Some(lease_token.to_string().as_str())
        || event.lease_expires_at_unix_ms != Some(lease_expires_at_unix_ms)
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(Some(lease_from_current(&verified.stored, event)?))
}

/// Return whether one exact space has a durable terminal provider failure.
pub(crate) fn embedding_space_is_degraded(
    connection: &Connection,
    project_uuid: Uuid,
    vector_space_id: &VectorSpaceId,
) -> Result<bool, LedgerError> {
    validate_uuid_v7(project_uuid)?;
    if resolve_vector_space(connection, project_uuid, vector_space_id)?.is_none() {
        return Ok(false);
    }
    let embedding_job_id = connection
        .query_row(
            "SELECT embedding_job_id
             FROM embedding_jobs
             WHERE vector_space_id = ?1 AND terminal_error_class IS NOT NULL
             ORDER BY embedding_job_id
             LIMIT 1",
            params![vector_space_id.as_str()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let Some(embedding_job_id) = embedding_job_id else {
        return Ok(false);
    };
    let job = load_verified_embedding_job(connection, project_uuid, &embedding_job_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if job.vector_space_id != vector_space_id.as_str() || job.terminal_error_class.is_none() {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(true)
}

fn state_chain_is_canonical(
    connection: &Connection,
    project_uuid: Uuid,
    job: &StoredJob,
    states: &[StoredJobState],
) -> Result<bool, LedgerError> {
    let Some(first) = states.first() else {
        return Ok(false);
    };
    let mut active_claim: Option<&StoredJobState> = None;
    let mut claim_tokens = BTreeSet::new();
    let mut generation = 0_i64;
    let mut attempt_count = 0_i64;
    let mut next_eligible_at_unix_ms = job.created_at_unix_ms;
    let mut terminal = false;
    let mut last_reset_actor: Option<&str> = None;
    let mut last_reset_reason: Option<&str> = None;
    for (index, state) in states.iter().enumerate() {
        if !stored_state_is_canonical(state)? || state.embedding_job_id != job.embedding_job_id {
            return Ok(false);
        }
        let Some(actor) = state
            .process_instance_id
            .as_deref()
            .and_then(parse_canonical_uuid_v7)
        else {
            return Ok(false);
        };
        if verified_process_status_at(connection, project_uuid, actor, state.created_at_unix_ms)?
            == ProcessStatusAt::Invalid
        {
            return Ok(false);
        }
        if index > 0 && state.created_at_unix_ms < states[index - 1].created_at_unix_ms {
            return Ok(false);
        }
        match state.state.as_str() {
            "pending" => {
                if index != 0
                    || state.attempt_generation != 0
                    || state.attempt_count != Some(0)
                    || state.lease_token.is_some()
                    || state.lease_expires_at_unix_ms.is_some()
                    || state.next_eligible_at_unix_ms != Some(job.created_at_unix_ms)
                    || state.created_at_unix_ms != job.created_at_unix_ms
                    || state.stable_error_class.is_some()
                {
                    return Ok(false);
                }
            }
            "claimed" => {
                if index == 0 || terminal || active_claim.is_some() {
                    return Ok(false);
                }
                let Some(token) = state.lease_token.as_ref() else {
                    return Ok(false);
                };
                let expected_expiry = state
                    .created_at_unix_ms
                    .checked_add(EMBEDDING_JOB_LEASE_MILLIS);
                let Some(expected_generation) = generation.checked_add(1) else {
                    return Ok(false);
                };
                let Some(expected_count) = attempt_count.checked_add(1) else {
                    return Ok(false);
                };
                if expected_count > 5
                    || state.attempt_generation != expected_generation
                    || state.attempt_count != Some(expected_count)
                    || expected_expiry.is_none()
                    || state.lease_expires_at_unix_ms != expected_expiry
                    || state.stable_error_class.is_some()
                    || state.next_eligible_at_unix_ms != Some(next_eligible_at_unix_ms)
                    || state.created_at_unix_ms < next_eligible_at_unix_ms
                    || !claim_tokens.insert(token.clone())
                {
                    return Ok(false);
                }
                generation = expected_generation;
                attempt_count = expected_count;
                active_claim = Some(state);
            }
            "orphaned_in_flight" => {
                let Some(claim) = active_claim else {
                    return Ok(false);
                };
                if !state_reproduces_claim(state, claim, false)
                    || state.stable_error_class.is_some()
                    || state.next_eligible_at_unix_ms != claim.next_eligible_at_unix_ms
                {
                    return Ok(false);
                }
                active_claim = None;
            }
            "released" => {
                let Some(claim) = active_claim else {
                    return Ok(false);
                };
                if !state_reproduces_claim(state, claim, true)
                    || state.stable_error_class.is_some()
                    || state.next_eligible_at_unix_ms != Some(state.created_at_unix_ms)
                {
                    return Ok(false);
                }
                next_eligible_at_unix_ms = state.created_at_unix_ms;
                active_claim = None;
            }
            "retry_scheduled" => {
                let Some(claim) = active_claim else {
                    return Ok(false);
                };
                if attempt_count >= 5
                    || !state_reproduces_claim(state, claim, true)
                    || state.stable_error_class.is_none()
                    || state
                        .next_eligible_at_unix_ms
                        .is_none_or(|next| next < state.created_at_unix_ms)
                {
                    return Ok(false);
                }
                next_eligible_at_unix_ms = state.next_eligible_at_unix_ms.unwrap();
                active_claim = None;
            }
            "completed" => {
                let Some(claim) = active_claim else {
                    return Ok(false);
                };
                if index + 1 != states.len()
                    || !state_reproduces_claim(state, claim, true)
                    || state.stable_error_class.is_some()
                    || state.next_eligible_at_unix_ms != Some(next_eligible_at_unix_ms)
                {
                    return Ok(false);
                }
                terminal = true;
                active_claim = None;
            }
            "terminal_failure" | "quarantined" => {
                let Some(claim) = active_claim else {
                    return Ok(false);
                };
                if !state_reproduces_claim(state, claim, true)
                    || state.stable_error_class.is_none()
                    || state.next_eligible_at_unix_ms != Some(next_eligible_at_unix_ms)
                    || (state.state == "quarantined" && !(1..=5).contains(&attempt_count))
                {
                    return Ok(false);
                }
                terminal = true;
                active_claim = None;
            }
            "reset" => {
                if !terminal
                    || index == 0
                    || !matches!(
                        states[index - 1].state.as_str(),
                        "terminal_failure" | "quarantined"
                    )
                    || active_claim.is_some()
                    || state.attempt_generation != generation.checked_add(1).unwrap_or(-1)
                    || state.attempt_count != Some(0)
                    || state.lease_token.is_some()
                    || state.lease_expires_at_unix_ms.is_some()
                    || state.stable_error_class.is_some()
                    || state.next_eligible_at_unix_ms != Some(state.created_at_unix_ms)
                {
                    return Ok(false);
                }
                generation = state.attempt_generation;
                attempt_count = 0;
                next_eligible_at_unix_ms = state.created_at_unix_ms;
                terminal = false;
                last_reset_actor = state.reset_actor.as_deref();
                last_reset_reason = state.reset_reason.as_deref();
            }
            _ => return Ok(false),
        }
    }
    debug_assert_eq!(first.state, "pending");
    let latest = states.last().expect("nonempty state chain");
    let current_matches = match latest.state.as_str() {
        "claimed" => {
            let Some((owner, token, expires_at)) = current_lease(job)? else {
                return Ok(false);
            };
            latest.process_instance_id.as_deref() == Some(owner.to_string().as_str())
                && latest.lease_token.as_deref() == Some(token.to_string().as_str())
                && latest.lease_expires_at_unix_ms == Some(expires_at)
                && latest.attempt_generation == job.attempt_generation
        }
        "pending" | "orphaned_in_flight" | "released" | "retry_scheduled" | "completed"
        | "terminal_failure" | "quarantined" | "reset" => current_lease(job)?.is_none(),
        _ => unreachable!("state variants were checked above"),
    };
    let expected_terminal_error = match latest.state.as_str() {
        "terminal_failure" | "quarantined" => latest.stable_error_class.as_deref(),
        _ => None,
    };
    Ok(current_matches
        && generation == job.attempt_generation
        && attempt_count == job.attempt_count
        && next_eligible_at_unix_ms == job.next_eligible_at_unix_ms
        && expected_terminal_error == job.terminal_error_class.as_deref()
        && (expected_terminal_error.is_some()
            || (job.failure_propagation_cursor.is_none() && !job.failure_propagation_complete))
        && last_reset_actor == job.reset_actor.as_deref()
        && last_reset_reason == job.reset_reason.as_deref())
}

fn state_reproduces_claim(
    state: &StoredJobState,
    claim: &StoredJobState,
    same_actor: bool,
) -> bool {
    state.attempt_generation == claim.attempt_generation
        && state.attempt_count == claim.attempt_count
        && (!same_actor || state.process_instance_id == claim.process_instance_id)
        && state.lease_token == claim.lease_token
        && state.lease_expires_at_unix_ms == claim.lease_expires_at_unix_ms
        && state.created_at_unix_ms >= claim.created_at_unix_ms
        && (!same_actor
            || state
                .lease_expires_at_unix_ms
                .is_some_and(|expiry| state.created_at_unix_ms < expiry))
}

fn stored_job_is_canonical(stored: &StoredJob) -> Result<bool, LedgerError> {
    if validate_sha256(&stored.embedding_job_id).is_err()
        || validate_sha256(&stored.vector_space_id).is_err()
        || validate_sha256(&stored.canonical_query_hash).is_err()
        || validate_sha256(&stored.content_hash).is_err()
        || stored.content_hash != stored.canonical_query_hash
        || embedding_job_id(&stored.vector_space_id, &stored.canonical_query_hash)?
            != stored.embedding_job_id
        || stored.attempt_generation < 0
        || stored.attempt_count < 0
        || stored.attempt_count > 5
        || stored.attempt_generation < stored.attempt_count
        || stored.next_eligible_at_unix_ms < stored.created_at_unix_ms
        || stored.created_at_unix_ms < 0
        || stored
            .terminal_error_class
            .as_deref()
            .is_some_and(|value| validate_stable_error_class(value).is_err())
        || stored
            .failure_propagation_cursor
            .as_deref()
            .is_some_and(|value| parse_canonical_uuid_v7(value).is_none())
        || (stored.terminal_error_class.is_none()
            && (stored.failure_propagation_cursor.is_some() || stored.failure_propagation_complete))
        || (stored.reset_actor.is_none() != stored.reset_reason.is_none())
        || stored
            .reset_actor
            .as_deref()
            .is_some_and(|value| !valid_bounded_nonempty(value, 128))
        || stored
            .reset_reason
            .as_deref()
            .is_some_and(|value| !valid_bounded_nonempty(value, 512))
    {
        return Ok(false);
    }
    let owner = parse_optional_uuid(stored.lease_owner_process_instance_id.as_deref())?;
    let token = parse_optional_uuid(stored.lease_token.as_deref())?;
    let lease_tuple_present = (
        owner.is_some(),
        token.is_some(),
        stored.lease_expires_at_unix_ms.is_some(),
    );
    if !matches!(
        lease_tuple_present,
        (false, false, false) | (true, true, true)
    ) {
        return Ok(false);
    }
    let expected = job_hash(
        &stored.embedding_job_id,
        &stored.vector_space_id,
        &stored.canonical_query_hash,
        &stored.content_hash,
        owner,
        token,
        stored.lease_expires_at_unix_ms,
        stored.attempt_generation,
        stored.attempt_count,
        stored.next_eligible_at_unix_ms,
        stored.terminal_error_class.as_deref(),
        stored.failure_propagation_cursor.as_deref(),
        stored.failure_propagation_complete,
        stored.reset_actor.as_deref(),
        stored.reset_reason.as_deref(),
        stored.created_at_unix_ms,
    )?;
    Ok(stored.canonical_payload_hash == expected)
}

fn stored_state_is_canonical(state: &StoredJobState) -> Result<bool, LedgerError> {
    let Some(event_id) = parse_canonical_uuid_v7(&state.embedding_job_state_event_id) else {
        return Ok(false);
    };
    let process_id = match state.process_instance_id.as_deref() {
        Some(value) => match parse_canonical_uuid_v7(value) {
            Some(value) => Some(value),
            None => return Ok(false),
        },
        None => None,
    };
    let token = match state.lease_token.as_deref() {
        Some(value) => match parse_canonical_uuid_v7(value) {
            Some(value) => Some(value),
            None => return Ok(false),
        },
        None => None,
    };
    if validate_sha256(&state.embedding_job_id).is_err()
        || state.attempt_generation < 0
        || state
            .attempt_count
            .is_none_or(|value| !(0..=5).contains(&value))
        || state.next_eligible_at_unix_ms.is_none_or(|value| value < 0)
        || state.created_at_unix_ms < 0
        || (token.is_none() != state.lease_expires_at_unix_ms.is_none())
        || state
            .stable_error_class
            .as_deref()
            .is_some_and(|value| validate_stable_error_class(value).is_err())
        || (matches!(
            state.state.as_str(),
            "retry_scheduled" | "terminal_failure" | "quarantined"
        ) != state.stable_error_class.is_some())
        || ((state.state == "reset")
            != (state.reset_actor.is_some() && state.reset_reason.is_some()))
        || state
            .reset_actor
            .as_deref()
            .is_some_and(|value| !valid_bounded_nonempty(value, 128))
        || state
            .reset_reason
            .as_deref()
            .is_some_and(|value| !valid_bounded_nonempty(value, 512))
    {
        return Ok(false);
    }
    let expected = state_hash(
        event_id,
        &state.embedding_job_id,
        process_id,
        &state.state,
        state.attempt_generation,
        state.stable_error_class.as_deref(),
        state.created_at_unix_ms,
        token,
        state.lease_expires_at_unix_ms,
        state.attempt_count.unwrap(),
        state.next_eligible_at_unix_ms.unwrap(),
        state.reset_actor.as_deref(),
        state.reset_reason.as_deref(),
    )?;
    Ok(state.canonical_payload_hash == expected)
}

#[cfg(test)]
fn claim_event_matches_command(
    event: &StoredJobState,
    process_instance_id: Uuid,
    command: &EmbeddingJobClaim,
) -> Result<bool, LedgerError> {
    Ok(stored_state_is_canonical(event)?
        && event.embedding_job_id == command.embedding_job_id
        && event.process_instance_id.as_deref() == Some(process_instance_id.to_string().as_str())
        && event.state == "claimed"
        && event.stable_error_class.is_none()
        && event.created_at_unix_ms == command.observed_at_unix_ms
        && event.lease_token.as_deref() == Some(command.lease_token.to_string().as_str())
        && event.lease_expires_at_unix_ms == Some(command.lease_expires_at_unix_ms)
        && event.attempt_count == Some(event.attempt_generation))
}

fn claim_event_matches_current(
    event: &StoredJobState,
    current: &StoredJob,
) -> Result<bool, LedgerError> {
    Ok(event.attempt_generation == current.attempt_generation
        && event.attempt_count == Some(current.attempt_count)
        && event.next_eligible_at_unix_ms == Some(current.next_eligible_at_unix_ms)
        && current.lease_owner_process_instance_id.as_deref()
            == event.process_instance_id.as_deref()
        && current.lease_token.as_deref() == event.lease_token.as_deref()
        && current.lease_expires_at_unix_ms == event.lease_expires_at_unix_ms)
}

fn create_event_matches(
    event: &StoredJobState,
    process_instance_id: Uuid,
    command: &EmbeddingJobCreate,
) -> Result<bool, LedgerError> {
    Ok(stored_state_is_canonical(event)?
        && event.embedding_job_id == command.embedding_job_id
        && event.process_instance_id.as_deref() == Some(process_instance_id.to_string().as_str())
        && event.state == "pending"
        && event.attempt_generation == 0
        && event.attempt_count == Some(0)
        && event.stable_error_class.is_none()
        && event.created_at_unix_ms == command.created_at_unix_ms
        && event.lease_token.is_none()
        && event.lease_expires_at_unix_ms.is_none()
        && event.next_eligible_at_unix_ms == Some(command.created_at_unix_ms))
}

#[cfg(test)]
fn completion_event_matches(
    event: &StoredJobState,
    process_instance_id: Uuid,
    command: &EmbeddingJobCompletion,
) -> Result<bool, LedgerError> {
    Ok(stored_state_is_canonical(event)?
        && event.embedding_job_id == command.embedding_job_id
        && event.process_instance_id.as_deref() == Some(process_instance_id.to_string().as_str())
        && event.state == "completed"
        && event.attempt_generation == command.attempt_generation
        && event.attempt_count == Some(command.attempt_generation)
        && event.stable_error_class.is_none()
        && event.created_at_unix_ms == command.completed_at_unix_ms
        && event.lease_token.as_deref() == Some(command.lease_token.to_string().as_str()))
}

fn lease_from_current(
    current: &StoredJob,
    event: &StoredJobState,
) -> Result<EmbeddingJobLease, LedgerError> {
    let Some((owner, token, expires_at)) = current_lease(current)? else {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    };
    Ok(EmbeddingJobLease {
        job: snapshot(current),
        lease_owner_process_instance_id: owner,
        lease_token: token,
        lease_expires_at_unix_ms: expires_at,
        state_event_hash: event.canonical_payload_hash.clone(),
    })
}

fn snapshot(stored: &StoredJob) -> EmbeddingJobSnapshot {
    EmbeddingJobSnapshot {
        embedding_job_id: stored.embedding_job_id.clone(),
        vector_space_id: stored.vector_space_id.clone(),
        canonical_query_hash: stored.canonical_query_hash.clone(),
        content_hash: stored.content_hash.clone(),
        attempt_generation: stored.attempt_generation,
        attempt_count: stored.attempt_count,
        next_eligible_at_unix_ms: stored.next_eligible_at_unix_ms,
        terminal_error_class: stored.terminal_error_class.clone(),
        failure_propagation_cursor: stored.failure_propagation_cursor.clone(),
        failure_propagation_complete: stored.failure_propagation_complete,
        reset_actor: stored.reset_actor.clone(),
        reset_reason: stored.reset_reason.clone(),
        canonical_payload_hash: stored.canonical_payload_hash.clone(),
    }
}

fn current_lease(stored: &StoredJob) -> Result<Option<(Uuid, Uuid, i64)>, LedgerError> {
    match (
        stored.lease_owner_process_instance_id.as_deref(),
        stored.lease_token.as_deref(),
        stored.lease_expires_at_unix_ms,
    ) {
        (None, None, None) => Ok(None),
        (Some(owner), Some(token), Some(expires_at)) => {
            let owner = parse_canonical_uuid_v7(owner)
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
            let token = parse_canonical_uuid_v7(token)
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
            Ok(Some((owner, token, expires_at)))
        }
        _ => Err(LedgerErrorClass::IdentityInvariant.into()),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VectorSpaceStatus {
    Missing,
    Valid,
    Invalid,
}

fn verified_vector_space(
    connection: &Connection,
    project_uuid: Uuid,
    vector_space_id: &str,
) -> Result<VectorSpaceStatus, LedgerError> {
    let Ok(vector_space_id) = VectorSpaceId::new(vector_space_id.to_string()) else {
        return Ok(VectorSpaceStatus::Invalid);
    };
    match resolve_vector_space(connection, project_uuid, &vector_space_id) {
        Ok(Some(_)) => Ok(VectorSpaceStatus::Valid),
        Ok(None) => Ok(VectorSpaceStatus::Missing),
        Err(error) if error.class() == LedgerErrorClass::CorruptDatabase => {
            Ok(VectorSpaceStatus::Invalid)
        }
        Err(error) => Err(error),
    }
}

fn verified_profile_batch_size(
    connection: &Connection,
    project_uuid: Uuid,
    vector_space_id: &VectorSpaceId,
) -> Result<Option<usize>, LedgerError> {
    let Some(space) = resolve_vector_space(connection, project_uuid, vector_space_id)? else {
        return Ok(None);
    };
    let batch_size = connection
        .query_row(
            "SELECT batch_size FROM embedder_profiles
             WHERE embedder_profile_version_id = ?1",
            params![space.space.embedder_profile_version_id],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let batch_size = usize::try_from(batch_size)
        .ok()
        .filter(|value| (1..=128).contains(value))
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    Ok(Some(batch_size))
}

fn job_belongs_to_project(
    connection: &Connection,
    project_uuid: Uuid,
    job: &StoredJob,
) -> Result<bool, LedgerError> {
    Ok(
        verified_vector_space(connection, project_uuid, &job.vector_space_id)?
            == VectorSpaceStatus::Valid
            && load_canonical_query(connection, &job.canonical_query_hash)?.is_some(),
    )
}

fn load_jobs_for_identity(
    connection: &Connection,
    embedding_job_id: &str,
    vector_space_id: &str,
    canonical_query_hash: &str,
) -> Result<Vec<StoredJob>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT embedding_job_id, vector_space_id, canonical_query_hash, content_hash,
                    lease_owner_process_instance_id, lease_token, lease_expires_at_unix_ms,
                    attempt_generation, attempt_count, next_eligible_at_unix_ms,
                    terminal_error_class, failure_propagation_cursor,
                    failure_propagation_complete, reset_actor, reset_reason,
                    created_at_unix_ms, canonical_payload_hash
             FROM embedding_jobs
             WHERE embedding_job_id = ?1
                OR (vector_space_id = ?2 AND canonical_query_hash = ?3)",
        )
        .map_err(database_error)?;
    statement
        .query_map(
            params![embedding_job_id, vector_space_id, canonical_query_hash],
            stored_job_from_row,
        )
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)
}

fn load_job(
    connection: &Connection,
    embedding_job_id: &str,
) -> Result<Option<StoredJob>, LedgerError> {
    connection
        .query_row(
            "SELECT embedding_job_id, vector_space_id, canonical_query_hash, content_hash,
                    lease_owner_process_instance_id, lease_token, lease_expires_at_unix_ms,
                    attempt_generation, attempt_count, next_eligible_at_unix_ms,
                    terminal_error_class, failure_propagation_cursor,
                    failure_propagation_complete, reset_actor, reset_reason,
                    created_at_unix_ms, canonical_payload_hash
             FROM embedding_jobs WHERE embedding_job_id = ?1",
            params![embedding_job_id],
            stored_job_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn stored_job_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredJob> {
    Ok(StoredJob {
        embedding_job_id: row.get(0)?,
        vector_space_id: row.get(1)?,
        canonical_query_hash: row.get(2)?,
        content_hash: row.get(3)?,
        lease_owner_process_instance_id: row.get(4)?,
        lease_token: row.get(5)?,
        lease_expires_at_unix_ms: row.get(6)?,
        attempt_generation: row.get(7)?,
        attempt_count: row.get(8)?,
        next_eligible_at_unix_ms: row.get(9)?,
        terminal_error_class: row.get(10)?,
        failure_propagation_cursor: row.get(11)?,
        failure_propagation_complete: row.get(12)?,
        reset_actor: row.get(13)?,
        reset_reason: row.get(14)?,
        created_at_unix_ms: row.get(15)?,
        canonical_payload_hash: row.get(16)?,
    })
}

fn load_job_states(
    connection: &Connection,
    embedding_job_id: &str,
) -> Result<Vec<StoredJobState>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT embedding_job_state_event_id, embedding_job_id, process_instance_id,
                    state, attempt_generation, stable_error_class, created_at_unix_ms,
                    canonical_payload_hash, lease_token, lease_expires_at_unix_ms,
                    attempt_count, next_eligible_at_unix_ms, reset_actor, reset_reason
             FROM embedding_job_state_events
             WHERE embedding_job_id = ?1 ORDER BY event_seq",
        )
        .map_err(database_error)?;
    statement
        .query_map(params![embedding_job_id], stored_state_from_row)
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)
}

fn load_state_by_event_id(
    connection: &Connection,
    event_id: Uuid,
) -> Result<Option<StoredJobState>, LedgerError> {
    connection
        .query_row(
            "SELECT embedding_job_state_event_id, embedding_job_id, process_instance_id,
                    state, attempt_generation, stable_error_class, created_at_unix_ms,
                    canonical_payload_hash, lease_token, lease_expires_at_unix_ms,
                    attempt_count, next_eligible_at_unix_ms, reset_actor, reset_reason
             FROM embedding_job_state_events WHERE embedding_job_state_event_id = ?1",
            params![event_id.to_string()],
            stored_state_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn load_state_for_transition(
    connection: &Connection,
    embedding_job_id: &str,
    attempt_generation: i64,
    state: &str,
) -> Result<Option<StoredJobState>, LedgerError> {
    connection
        .query_row(
            "SELECT embedding_job_state_event_id, embedding_job_id, process_instance_id,
                    state, attempt_generation, stable_error_class, created_at_unix_ms,
                    canonical_payload_hash, lease_token, lease_expires_at_unix_ms,
                    attempt_count, next_eligible_at_unix_ms, reset_actor, reset_reason
             FROM embedding_job_state_events
             WHERE embedding_job_id = ?1 AND attempt_generation = ?2 AND state = ?3",
            params![embedding_job_id, attempt_generation, state],
            stored_state_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn stored_state_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredJobState> {
    Ok(StoredJobState {
        embedding_job_state_event_id: row.get(0)?,
        embedding_job_id: row.get(1)?,
        process_instance_id: row.get(2)?,
        state: row.get(3)?,
        attempt_generation: row.get(4)?,
        stable_error_class: row.get(5)?,
        created_at_unix_ms: row.get(6)?,
        canonical_payload_hash: row.get(7)?,
        lease_token: row.get(8)?,
        lease_expires_at_unix_ms: row.get(9)?,
        attempt_count: row.get(10)?,
        next_eligible_at_unix_ms: row.get(11)?,
        reset_actor: row.get(12)?,
        reset_reason: row.get(13)?,
    })
}

#[allow(clippy::too_many_arguments)]
fn insert_state(
    connection: &Connection,
    event_id: Uuid,
    embedding_job_id: &str,
    process_instance_id: Uuid,
    state: &str,
    attempt_generation: i64,
    stable_error_class: Option<&str>,
    created_at_unix_ms: i64,
    lease_token: Option<Uuid>,
    lease_expires_at_unix_ms: Option<i64>,
    attempt_count: i64,
    next_eligible_at_unix_ms: i64,
    reset_actor: Option<&str>,
    reset_reason: Option<&str>,
    canonical_payload_hash: &str,
) -> Result<(), LedgerError> {
    connection
        .execute(
            "INSERT INTO embedding_job_state_events (
                embedding_job_state_event_id, embedding_job_id, process_instance_id,
                state, attempt_generation, stable_error_class, created_at_unix_ms,
                canonical_payload_hash, lease_token, lease_expires_at_unix_ms,
                attempt_count, next_eligible_at_unix_ms, reset_actor, reset_reason
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14
             )",
            params![
                event_id.to_string(),
                embedding_job_id,
                process_instance_id.to_string(),
                state,
                attempt_generation,
                stable_error_class,
                created_at_unix_ms,
                canonical_payload_hash,
                lease_token.map(|value| value.to_string()),
                lease_expires_at_unix_ms,
                attempt_count,
                next_eligible_at_unix_ms,
                reset_actor,
                reset_reason,
            ],
        )
        .map_err(database_error)?;
    Ok(())
}

fn embedding_job_id(
    vector_space_id: &str,
    canonical_query_hash: &str,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "vector_space_id": vector_space_id,
        "canonical_query_hash": canonical_query_hash,
    }))
}

#[allow(clippy::too_many_arguments)]
fn job_hash(
    embedding_job_id: &str,
    vector_space_id: &str,
    canonical_query_hash: &str,
    content_hash: &str,
    lease_owner_process_instance_id: Option<Uuid>,
    lease_token: Option<Uuid>,
    lease_expires_at_unix_ms: Option<i64>,
    attempt_generation: i64,
    attempt_count: i64,
    next_eligible_at_unix_ms: i64,
    terminal_error_class: Option<&str>,
    failure_propagation_cursor: Option<&str>,
    failure_propagation_complete: bool,
    reset_actor: Option<&str>,
    reset_reason: Option<&str>,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "embedding_job_id": embedding_job_id,
        "vector_space_id": vector_space_id,
        "canonical_query_hash": canonical_query_hash,
        "content_hash": content_hash,
        "lease_owner_process_instance_id": lease_owner_process_instance_id,
        "lease_token": lease_token,
        "lease_expires_at_unix_ms": lease_expires_at_unix_ms,
        "attempt_generation": attempt_generation,
        "attempt_count": attempt_count,
        "next_eligible_at_unix_ms": next_eligible_at_unix_ms,
        "terminal_error_class": terminal_error_class,
        "failure_propagation_cursor": failure_propagation_cursor,
        "failure_propagation_complete": failure_propagation_complete,
        "reset_actor": reset_actor,
        "reset_reason": reset_reason,
        "created_at_unix_ms": created_at_unix_ms,
    }))
}

#[allow(clippy::too_many_arguments)]
fn state_hash(
    event_id: Uuid,
    embedding_job_id: &str,
    process_instance_id: Option<Uuid>,
    state: &str,
    attempt_generation: i64,
    stable_error_class: Option<&str>,
    created_at_unix_ms: i64,
    lease_token: Option<Uuid>,
    lease_expires_at_unix_ms: Option<i64>,
    attempt_count: i64,
    next_eligible_at_unix_ms: i64,
    reset_actor: Option<&str>,
    reset_reason: Option<&str>,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "embedding_job_state_event_id": event_id,
        "embedding_job_id": embedding_job_id,
        "process_instance_id": process_instance_id,
        "state": state,
        "attempt_generation": attempt_generation,
        "stable_error_class": stable_error_class,
        "created_at_unix_ms": created_at_unix_ms,
        "lease_token": lease_token,
        "lease_expires_at_unix_ms": lease_expires_at_unix_ms,
        "attempt_count": attempt_count,
        "next_eligible_at_unix_ms": next_eligible_at_unix_ms,
        "reset_actor": reset_actor,
        "reset_reason": reset_reason,
    }))
}

fn parse_optional_uuid(value: Option<&str>) -> Result<Option<Uuid>, LedgerError> {
    value
        .map(|value| {
            parse_canonical_uuid_v7(value)
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))
        })
        .transpose()
}

fn parse_canonical_uuid_v7(value: &str) -> Option<Uuid> {
    let parsed = Uuid::parse_str(value).ok()?;
    (parsed.to_string() == value && parsed.get_version_num() == 7).then_some(parsed)
}

fn validate_uuid_v7(value: Uuid) -> Result<(), LedgerError> {
    if value.get_version_num() != 7 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), LedgerError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn validate_stable_error_class(value: &str) -> Result<(), LedgerError> {
    if value.is_empty()
        || value.len() > 128
        || !value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'.')
        })
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn valid_bounded_nonempty(value: &str, max_len: usize) -> bool {
    !value.trim().is_empty() && value.len() <= max_len
}

fn hash_json(value: &Json) -> Result<String, LedgerError> {
    canonical_sha256(value).map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn database_error(error: rusqlite::Error) -> LedgerError {
    map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Arc, Barrier};
    use std::thread;

    use rusqlite::params;
    use tempfile::tempdir;

    use super::*;
    use crate::canonical_json::canonical_json;
    use crate::canonical_query::{
        CanonicalRoutingQueryArtifactV1, CanonicalRoutingQueryV1, CanonicalTaskV1,
    };
    use crate::config::{LearningConfig, RouterConfig};
    use crate::embedding_identity::CANONICAL_ROUTING_QUERY_SCHEMA_V1;
    use crate::fingerprint::sha256_hex;
    use crate::ledger::model::{LedgerErrorClass, LedgerRuntimeIdentity};
    use crate::ledger::repository::process::{
        HeartbeatAck, HeartbeatRenewal, ProcessCommandAck, ProcessStop,
    };
    use crate::ledger::repository::tests::{config, database_path};
    use crate::ledger::repository::{ActivatedLedger, TransactionStartGuard};
    use crate::vector::NormalizedVector;

    fn learning_config(path: &Path, project_id: &str) -> RouterConfig {
        let mut config = config(path, project_id);
        config.pools[0].learning = Some(LearningConfig::minimal("embedder-a"));
        config
    }

    fn activate(path: &Path, project_id: &str) -> ActivatedLedger {
        LedgerRepository::activate(&learning_config(path, project_id)).unwrap()
    }

    fn activate_with_batch_size(
        path: &Path,
        project_id: &str,
        batch_size: usize,
    ) -> ActivatedLedger {
        let mut config = learning_config(path, project_id);
        config.embedders[0].batch_size = batch_size;
        LedgerRepository::activate(&config).unwrap()
    }

    fn process_started_at(repository: &LedgerRepository, process_instance_id: Uuid) -> i64 {
        repository
            .connection
            .query_row(
                "SELECT started_at_unix_ms FROM process_instances
                 WHERE process_instance_id = ?1",
                params![process_instance_id.to_string()],
                |row| row.get(0),
            )
            .unwrap()
    }

    fn seed_vector_space(
        repository: &LedgerRepository,
        identity: &LedgerRuntimeIdentity,
        created_at_unix_ms: i64,
    ) -> String {
        seed_query(repository, created_at_unix_ms);
        identity
            .pool("pool-a")
            .and_then(|pool| pool.vector_space.as_ref())
            .expect("learning activation should persist a vector-space mapping")
            .vector_space_id
            .as_str()
            .to_string()
    }

    fn seed_query(repository: &LedgerRepository, created_at_unix_ms: i64) {
        let (query_hash, canonical_query) = canonical_query();
        repository
            .connection
            .execute(
                "INSERT OR IGNORE INTO canonical_routing_queries (
                    canonical_query_hash, canonical_query_json, canonical_size_bytes,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, ?1)",
                params![
                    query_hash,
                    canonical_query,
                    i64::try_from(canonical_query.len()).unwrap(),
                    created_at_unix_ms,
                ],
            )
            .unwrap();
    }

    fn canonical_query() -> (String, String) {
        canonical_query_named("embedding job fixture")
    }

    fn canonical_query_named(text: &str) -> (String, String) {
        let query = CanonicalRoutingQueryV1 {
            schema: CANONICAL_ROUTING_QUERY_SCHEMA_V1.to_string(),
            instructions: Vec::new(),
            current_task: CanonicalTaskV1 {
                text: text.to_string(),
            },
            bounded_context: Vec::new(),
            tool_schema_fingerprint: "1".repeat(64),
            response_schema_fingerprint: None,
            required_capabilities: Vec::new(),
            position_features: None,
        };
        let canonical = canonical_json(&serde_json::to_value(query).unwrap()).unwrap();
        (sha256_hex(canonical.as_bytes()), canonical)
    }

    fn canonical_query_artifact_named(text: &str) -> CanonicalRoutingQueryArtifactV1 {
        let (canonical_query_hash, canonical) = canonical_query_named(text);
        CanonicalRoutingQueryArtifactV1 {
            query: serde_json::from_str(&canonical).unwrap(),
            canonical_bytes: canonical.into_bytes(),
            canonical_query_hash,
        }
    }

    fn current_mapping(identity: &LedgerRuntimeIdentity) -> FrozenMappingKey {
        let pool = identity.pool("pool-a").unwrap();
        FrozenMappingKey::new(
            identity.project_uuid,
            identity.config_generation_id.clone(),
            "pool-a",
            pool.policy_version_id.clone(),
        )
        .unwrap()
    }

    fn live_prepare(
        identity: &LedgerRuntimeIdentity,
        artifact: CanonicalRoutingQueryArtifactV1,
        prepared_at_unix_ms: i64,
    ) -> LiveEmbeddingPrepare {
        LiveEmbeddingPrepare::new(
            current_mapping(identity),
            artifact,
            Uuid::now_v7(),
            Uuid::now_v7(),
            prepared_at_unix_ms,
        )
        .unwrap()
    }

    fn create_named_job(
        repository: &mut LedgerRepository,
        vector_space_id: &str,
        created_at_unix_ms: i64,
        name: &str,
    ) -> EmbeddingJobSnapshot {
        let (query_hash, canonical_query) = canonical_query_named(name);
        repository
            .connection
            .execute(
                "INSERT INTO canonical_routing_queries (
                    canonical_query_hash, canonical_query_json, canonical_size_bytes,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, ?1)",
                params![
                    query_hash,
                    canonical_query,
                    i64::try_from(canonical_query.len()).unwrap(),
                    created_at_unix_ms,
                ],
            )
            .unwrap();
        let create = EmbeddingJobCreate::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            vector_space_id,
            query_hash.clone(),
            query_hash,
            created_at_unix_ms,
        )
        .unwrap();
        let EmbeddingJobCreateAck::Applied(job) = repository.create_embedding_job(&create).unwrap()
        else {
            panic!("named job should be created");
        };
        job
    }

    fn authoritative_vector(
        repository: &LedgerRepository,
        project_uuid: Uuid,
        vector_space_id: &str,
        component: usize,
    ) -> AuthoritativeVector {
        let vector_space_id = VectorSpaceId::new(vector_space_id.to_string()).unwrap();
        let space = resolve_vector_space(&repository.connection, project_uuid, &vector_space_id)
            .unwrap()
            .unwrap();
        let mut values = vec![0.0_f64; space.space.dimensions.as_usize()];
        let index = component % values.len();
        values[index] = 1.0;
        let normalized =
            NormalizedVector::from_provider_f64(&values, space.space.dimensions).unwrap();
        AuthoritativeVector::from_normalized(&vector_space_id, normalized).unwrap()
    }

    fn batch_claim(
        vector_space_id: &str,
        jobs: &[EmbeddingJobSnapshot],
        lease_token: Uuid,
        observed_at_unix_ms: i64,
    ) -> EmbeddingJobBatchClaim {
        EmbeddingJobBatchClaim::new(
            Uuid::now_v7(),
            VectorSpaceId::new(vector_space_id.to_string()).unwrap(),
            lease_token,
            observed_at_unix_ms,
            jobs.iter()
                .map(|job| {
                    EmbeddingJobBatchClaimItem::new(
                        job.embedding_job_id.clone(),
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap()
    }

    fn batch_completion(
        repository: &LedgerRepository,
        project_uuid: Uuid,
        vector_space_id: &str,
        leases: &[EmbeddingJobLease],
        completed_at_unix_ms: i64,
    ) -> EmbeddingJobBatchCompletion {
        EmbeddingJobBatchCompletion::new(
            Uuid::now_v7(),
            VectorSpaceId::new(vector_space_id.to_string()).unwrap(),
            leases[0].lease_token,
            completed_at_unix_ms,
            leases
                .iter()
                .enumerate()
                .map(|(index, lease)| {
                    EmbeddingJobBatchCompletionItem::new(
                        lease.job.embedding_job_id.clone(),
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                        lease.job.attempt_generation,
                        lease.job.content_hash.clone(),
                        authoritative_vector(repository, project_uuid, vector_space_id, index),
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap()
    }

    fn create_command(vector_space_id: &str, created_at_unix_ms: i64) -> EmbeddingJobCreate {
        let (query_hash, _) = canonical_query();
        EmbeddingJobCreate::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            vector_space_id,
            query_hash.clone(),
            query_hash,
            created_at_unix_ms,
        )
        .unwrap()
    }

    fn tamper_pending_job_column(project_id: &str, assignment: &str) -> StoredJob {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, project_id);
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let create = create_command(&vector_space, created_at);
        assert!(matches!(
            activated.repository.create_embedding_job(&create).unwrap(),
            EmbeddingJobCreateAck::Applied(_)
        ));
        activated
            .repository
            .connection
            .execute_batch("PRAGMA ignore_check_constraints = ON")
            .unwrap();
        activated
            .repository
            .connection
            .execute(
                &format!("UPDATE embedding_jobs SET {assignment} WHERE embedding_job_id = ?1"),
                [create.embedding_job_id()],
            )
            .unwrap();
        let stored = load_job(&activated.repository.connection, create.embedding_job_id())
            .unwrap()
            .unwrap();
        let tampered_hash = job_hash(
            &stored.embedding_job_id,
            &stored.vector_space_id,
            &stored.canonical_query_hash,
            &stored.content_hash,
            None,
            None,
            None,
            stored.attempt_generation,
            stored.attempt_count,
            stored.next_eligible_at_unix_ms,
            stored.terminal_error_class.as_deref(),
            stored.failure_propagation_cursor.as_deref(),
            stored.failure_propagation_complete,
            stored.reset_actor.as_deref(),
            stored.reset_reason.as_deref(),
            stored.created_at_unix_ms,
        )
        .unwrap();
        assert_ne!(stored.canonical_payload_hash, tampered_hash);
        assert!(
            verify_job(
                &activated.repository.connection,
                activated.identity.project_uuid,
                stored,
            )
            .unwrap()
            .is_none()
        );
        load_job(&activated.repository.connection, create.embedding_job_id())
            .unwrap()
            .unwrap()
    }

    fn claim_command(
        embedding_job_id: &str,
        lease_token: Uuid,
        observed_at_unix_ms: i64,
    ) -> EmbeddingJobClaim {
        EmbeddingJobClaim::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            embedding_job_id,
            lease_token,
            observed_at_unix_ms,
        )
        .unwrap()
    }

    fn completion(lease: &EmbeddingJobLease, completed_at_unix_ms: i64) -> EmbeddingJobCompletion {
        EmbeddingJobCompletion::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            lease.job.embedding_job_id.clone(),
            lease.lease_token,
            lease.job.attempt_generation,
            lease.job.content_hash.clone(),
            completed_at_unix_ms,
        )
        .unwrap()
    }

    fn renew(repository: &mut LedgerRepository, observed_at_unix_ms: i64) {
        assert!(matches!(
            repository
                .renew_heartbeat(HeartbeatRenewal::new(observed_at_unix_ms).unwrap())
                .unwrap(),
            HeartbeatAck::Applied { .. } | HeartbeatAck::AlreadyApplied { .. }
        ));
    }

    #[test]
    fn live_prepare_atomically_ensures_query_and_reuses_the_deterministic_job() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "live-prepare-idempotency");
        let identity = activated.identity.clone();
        let prepared_at =
            process_started_at(&activated.repository, identity.process_instance_id) + 1;
        let artifact = canonical_query_artifact_named("live prepare idempotency");
        let command = live_prepare(&identity, artifact.clone(), prepared_at);

        assert!(
            load_canonical_query(
                &activated.repository.connection,
                &artifact.canonical_query_hash,
            )
            .unwrap()
            .is_none()
        );
        let LiveEmbeddingPrepareAck::Pending(job) = activated
            .repository
            .prepare_live_embedding(&command)
            .unwrap()
        else {
            panic!("cache miss should prepare one pending job");
        };
        assert_eq!(job.canonical_query_hash, artifact.canonical_query_hash);
        assert_eq!(job.content_hash, artifact.canonical_query_hash);
        assert_eq!(
            activated
                .repository
                .prepare_live_embedding(&command)
                .unwrap(),
            LiveEmbeddingPrepareAck::Pending(job.clone())
        );
        assert_eq!(
            load_verified_embedding_job(
                &activated.repository.connection,
                identity.project_uuid,
                &job.embedding_job_id,
            )
            .unwrap(),
            Some(job)
        );
        let counts = activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT COUNT(*) FROM canonical_routing_queries
                     WHERE canonical_query_hash = ?1),
                    (SELECT COUNT(*) FROM embedding_jobs
                     WHERE canonical_query_hash = ?1)",
                params![artifact.canonical_query_hash],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
            )
            .unwrap();
        assert_eq!(counts, (1, 1));
        let diagnostics = format!("{command:?}");
        assert!(diagnostics.contains(&artifact.canonical_query_hash));
        assert!(!diagnostics.contains("live prepare idempotency"));
    }

    #[test]
    fn live_prepare_rejects_start_stale_and_missing_mapping_without_writes() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "live-prepare-mapping");
        let identity = activated.identity.clone();
        let prepared_at =
            process_started_at(&activated.repository, identity.process_instance_id) + 1;

        let refused_artifact = canonical_query_artifact_named("live prepare refused");
        let refused = live_prepare(&identity, refused_artifact.clone(), prepared_at);
        assert_eq!(
            activated
                .repository
                .prepare_live_embedding_with_start_check(&refused, || None::<()>)
                .unwrap(),
            LiveEmbeddingPrepareAck::TransactionNotStarted
        );

        let mut stale_mapping = current_mapping(&identity);
        stale_mapping.config_generation_id = "0".repeat(64);
        let stale_artifact = canonical_query_artifact_named("live prepare stale mapping");
        let stale = LiveEmbeddingPrepare::new(
            stale_mapping,
            stale_artifact.clone(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            prepared_at,
        )
        .unwrap();
        assert_eq!(
            activated.repository.prepare_live_embedding(&stale).unwrap(),
            LiveEmbeddingPrepareAck::MappingNotCurrent
        );

        let mut missing_mapping = current_mapping(&identity);
        missing_mapping.policy_version_id = "f".repeat(64);
        let missing_artifact = canonical_query_artifact_named("live prepare missing mapping");
        let missing = LiveEmbeddingPrepare::new(
            missing_mapping,
            missing_artifact.clone(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            prepared_at,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .prepare_live_embedding(&missing)
                .unwrap(),
            LiveEmbeddingPrepareAck::MappingNotFound
        );

        for artifact in [refused_artifact, stale_artifact, missing_artifact] {
            assert!(
                load_canonical_query(
                    &activated.repository.connection,
                    &artifact.canonical_query_hash,
                )
                .unwrap()
                .is_none()
            );
        }
    }

    #[test]
    fn live_prepare_uses_cache_but_suppresses_new_work_for_a_degraded_space() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "live-prepare-degraded");
        let identity = activated.identity.clone();
        let project_uuid = identity.project_uuid;
        let prepared_at =
            process_started_at(&activated.repository, identity.process_instance_id) + 1;
        let vector_space = identity
            .pool("pool-a")
            .and_then(|pool| pool.vector_space.as_ref())
            .unwrap()
            .vector_space_id
            .as_str()
            .to_string();

        let cached_artifact = canonical_query_artifact_named("cached before degradation");
        let LiveEmbeddingPrepareAck::Pending(cached_job) = activated
            .repository
            .prepare_live_embedding(&live_prepare(
                &identity,
                cached_artifact.clone(),
                prepared_at,
            ))
            .unwrap()
        else {
            panic!("cache fixture should begin as one pending job");
        };
        let claim = batch_claim(
            &vector_space,
            std::slice::from_ref(&cached_job),
            Uuid::now_v7(),
            prepared_at + 1,
        );
        let EmbeddingJobBatchClaimAck::Claimed(leases) = activated
            .repository
            .claim_embedding_job_batch(&claim)
            .unwrap()
        else {
            panic!("cache fixture should claim");
        };
        let completion = batch_completion(
            &activated.repository,
            project_uuid,
            &vector_space,
            &leases,
            prepared_at + 2,
        );
        assert!(matches!(
            activated
                .repository
                .complete_embedding_job_batch(&completion)
                .unwrap(),
            EmbeddingJobBatchCompletionAck::Applied(_)
        ));

        let failed_job = create_named_job(
            &mut activated.repository,
            &vector_space,
            prepared_at + 3,
            "provider space degradation",
        );
        let failure_claim = batch_claim(
            &vector_space,
            std::slice::from_ref(&failed_job),
            Uuid::now_v7(),
            prepared_at + 4,
        );
        let EmbeddingJobBatchClaimAck::Claimed(failure_leases) = activated
            .repository
            .claim_embedding_job_batch(&failure_claim)
            .unwrap()
        else {
            panic!("failure fixture should claim");
        };
        let failed_lease = &failure_leases[0];
        assert!(matches!(
            activated
                .repository
                .resolve_embedding_job(
                    &EmbeddingJobResolution::new(
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                        failed_lease.job.embedding_job_id.clone(),
                        failed_lease.lease_token,
                        failed_lease.job.attempt_generation,
                        failed_lease.job.content_hash.clone(),
                        prepared_at + 5,
                        EmbeddingJobResolutionKind::Quarantined {
                            stable_error_class: "embedder_authentication".to_string(),
                        },
                    )
                    .unwrap(),
                )
                .unwrap(),
            EmbeddingJobResolutionAck::Applied {
                state: EmbeddingJobResolvedState::Quarantined,
                ..
            }
        ));

        assert!(matches!(
            activated
                .repository
                .prepare_live_embedding(&live_prepare(
                    &identity,
                    cached_artifact,
                    prepared_at + 6,
                ))
                .unwrap(),
            LiveEmbeddingPrepareAck::Ready(embedding)
                if embedding.canonical_query_hash == cached_job.canonical_query_hash
        ));

        let missed_artifact = canonical_query_artifact_named("degraded cache miss");
        assert_eq!(
            activated
                .repository
                .prepare_live_embedding(&live_prepare(
                    &identity,
                    missed_artifact.clone(),
                    prepared_at + 6,
                ))
                .unwrap(),
            LiveEmbeddingPrepareAck::ProviderSpaceDegraded
        );
        assert!(
            load_canonical_query(
                &activated.repository.connection,
                &missed_artifact.canonical_query_hash,
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn live_prepare_rolls_back_query_when_job_event_identity_conflicts() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "live-prepare-conflict");
        let identity = activated.identity.clone();
        let prepared_at =
            process_started_at(&activated.repository, identity.process_instance_id) + 1;
        let vector_space = identity
            .pool("pool-a")
            .and_then(|pool| pool.vector_space.as_ref())
            .unwrap()
            .vector_space_id
            .as_str()
            .to_string();
        let existing_artifact = canonical_query_artifact_named("existing job event owner");
        let transaction = activated.repository.connection.transaction().unwrap();
        assert!(matches!(
            ensure_canonical_query(&transaction, &existing_artifact, prepared_at).unwrap(),
            CanonicalQueryEnsureAck::Applied(_)
        ));
        transaction.commit().unwrap();
        let colliding_event_id = Uuid::now_v7();
        let existing = EmbeddingJobCreate::new(
            colliding_event_id,
            Uuid::now_v7(),
            &vector_space,
            existing_artifact.canonical_query_hash.clone(),
            existing_artifact.canonical_query_hash,
            prepared_at,
        )
        .unwrap();
        assert!(matches!(
            activated
                .repository
                .create_embedding_job(&existing)
                .unwrap(),
            EmbeddingJobCreateAck::Applied(_)
        ));

        let conflicted_artifact = canonical_query_artifact_named("rolled back live query");
        let conflict_health_event_id = Uuid::now_v7();
        let conflicted = LiveEmbeddingPrepare::new(
            current_mapping(&identity),
            conflicted_artifact.clone(),
            colliding_event_id,
            conflict_health_event_id,
            prepared_at + 1,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .prepare_live_embedding(&conflicted)
                .unwrap(),
            LiveEmbeddingPrepareAck::Conflict
        );
        assert!(
            load_canonical_query(
                &activated.repository.connection,
                &conflicted_artifact.canonical_query_hash,
            )
            .unwrap()
            .is_none()
        );
        let health_count = activated
            .repository
            .connection
            .query_row(
                "SELECT COUNT(*) FROM health_events WHERE health_event_id = ?1",
                params![conflict_health_event_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(health_count, 1);
    }

    #[test]
    fn create_claim_complete_are_exactly_idempotent() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-idempotency");
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let create = create_command(&vector_space, created_at);

        let created = activated.repository.create_embedding_job(&create).unwrap();
        let EmbeddingJobCreateAck::Applied(created) = created else {
            panic!("new job should be applied");
        };
        assert_eq!(created.embedding_job_id, create.embedding_job_id());
        assert!(created.terminal_error_class.is_none());
        assert!(created.failure_propagation_cursor.is_none());
        assert!(!created.failure_propagation_complete);
        assert!(created.reset_actor.is_none());
        assert!(created.reset_reason.is_none());
        assert_eq!(
            activated.repository.create_embedding_job(&create).unwrap(),
            EmbeddingJobCreateAck::AlreadyExists(created.clone())
        );

        let claim = claim_command(&created.embedding_job_id, Uuid::now_v7(), created_at + 1);
        let EmbeddingJobClaimAck::Claimed(lease) =
            activated.repository.claim_embedding_job(&claim).unwrap()
        else {
            panic!("eligible job should be claimed");
        };
        assert_eq!(lease.job.attempt_generation, 1);
        assert_eq!(lease.job.attempt_count, 1);
        assert_eq!(
            activated.repository.claim_embedding_job(&claim).unwrap(),
            EmbeddingJobClaimAck::AlreadyApplied(lease.clone())
        );

        let completion = completion(&lease, created_at + 2);
        let applied = activated
            .repository
            .complete_embedding_job(&completion)
            .unwrap();
        let EmbeddingJobCompletionAck::Applied {
            job,
            state_event_hash,
        } = applied
        else {
            panic!("current lease should complete");
        };
        assert_eq!(
            activated
                .repository
                .complete_embedding_job(&completion)
                .unwrap(),
            EmbeddingJobCompletionAck::AlreadyApplied {
                job: job.clone(),
                state_event_hash,
            }
        );
        assert_eq!(
            activated
                .repository
                .claim_embedding_job(&claim_command(
                    &created.embedding_job_id,
                    Uuid::now_v7(),
                    created_at + 3,
                ))
                .unwrap(),
            EmbeddingJobClaimAck::Terminal
        );
        let stored = load_job(&activated.repository.connection, &created.embedding_job_id)
            .unwrap()
            .unwrap();
        assert!(current_lease(&stored).unwrap().is_none());
        assert!(
            verify_job(
                &activated.repository.connection,
                activated.identity.project_uuid,
                stored,
            )
            .unwrap()
            .is_some()
        );
    }

    #[test]
    fn completion_predating_the_active_claim_is_stale_without_job_mutation() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = learning_config(&path, "embedding-predated-completion");
        let mut activated = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let created_at = 1_001;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let create = create_command(&vector_space, created_at);
        activated.repository.create_embedding_job(&create).unwrap();
        let claim_at = created_at + 2;
        let EmbeddingJobClaimAck::Claimed(lease) = activated
            .repository
            .claim_embedding_job(&claim_command(
                create.embedding_job_id(),
                Uuid::now_v7(),
                claim_at,
            ))
            .unwrap()
        else {
            panic!("job should be claimed");
        };
        let job_before = load_job(&activated.repository.connection, create.embedding_job_id())
            .unwrap()
            .unwrap();
        let states_before =
            load_job_states(&activated.repository.connection, create.embedding_job_id()).unwrap();

        assert_eq!(
            activated
                .repository
                .complete_embedding_job(&completion(&lease, claim_at - 1))
                .unwrap(),
            EmbeddingJobCompletionAck::StaleLease
        );
        assert_eq!(
            load_job(&activated.repository.connection, create.embedding_job_id())
                .unwrap()
                .unwrap(),
            job_before
        );
        assert_eq!(
            load_job_states(&activated.repository.connection, create.embedding_job_id()).unwrap(),
            states_before
        );

        let boundary_completion = completion(&lease, claim_at);
        assert!(matches!(
            activated
                .repository
                .complete_embedding_job(&boundary_completion)
                .unwrap(),
            EmbeddingJobCompletionAck::Applied { .. }
        ));
        assert!(matches!(
            activated
                .repository
                .complete_embedding_job(&boundary_completion)
                .unwrap(),
            EmbeddingJobCompletionAck::AlreadyApplied { .. }
        ));
    }

    #[test]
    fn terminal_owner_reclaim_cannot_predate_the_previous_claim() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = learning_config(&path, "embedding-predated-terminal-owner-reclaim");
        let mut first = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let mut second = LedgerRepository::activate_at(&config, 1_001).unwrap();
        let created_at = 1_002;
        let vector_space = seed_vector_space(&first.repository, &first.identity, created_at);
        let create = create_command(&vector_space, created_at);
        first.repository.create_embedding_job(&create).unwrap();
        let claim_at = created_at + 2;
        let EmbeddingJobClaimAck::Claimed(first_lease) = first
            .repository
            .claim_embedding_job(&claim_command(
                create.embedding_job_id(),
                Uuid::now_v7(),
                claim_at,
            ))
            .unwrap()
        else {
            panic!("first process should claim the job");
        };
        assert_eq!(
            first
                .repository
                .stop_process(ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), claim_at).unwrap())
                .unwrap(),
            ProcessCommandAck::Applied
        );
        let job_before = load_job(&second.repository.connection, create.embedding_job_id())
            .unwrap()
            .unwrap();
        let states_before =
            load_job_states(&second.repository.connection, create.embedding_job_id()).unwrap();

        assert_eq!(
            second
                .repository
                .claim_embedding_job(&claim_command(
                    create.embedding_job_id(),
                    Uuid::now_v7(),
                    claim_at - 1,
                ))
                .unwrap(),
            EmbeddingJobClaimAck::Conflict
        );
        assert_eq!(
            load_job(&second.repository.connection, create.embedding_job_id())
                .unwrap()
                .unwrap(),
            job_before
        );
        assert_eq!(
            load_job_states(&second.repository.connection, create.embedding_job_id()).unwrap(),
            states_before
        );

        let EmbeddingJobClaimAck::Reclaimed(second_lease) = second
            .repository
            .claim_embedding_job(&claim_command(
                create.embedding_job_id(),
                Uuid::now_v7(),
                claim_at,
            ))
            .unwrap()
        else {
            panic!("the exact state-time boundary should remain reclaimable");
        };
        assert_eq!(second_lease.job.attempt_generation, 2);
        assert_eq!(second_lease.job.attempt_count, 2);
        assert_eq!(
            second_lease.lease_owner_process_instance_id,
            second.identity.process_instance_id
        );
        assert_ne!(second_lease.lease_token, first_lease.lease_token);
    }

    #[test]
    fn same_owner_reclaim_fences_old_completion_and_reused_tokens() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-aba");
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let create = create_command(&vector_space, created_at);
        activated.repository.create_embedding_job(&create).unwrap();

        let token_a = Uuid::now_v7();
        let claim_a = claim_command(create.embedding_job_id(), token_a, created_at + 1);
        let EmbeddingJobClaimAck::Claimed(lease_a) =
            activated.repository.claim_embedding_job(&claim_a).unwrap()
        else {
            panic!("first lease should be claimed");
        };
        let reclaim_at = lease_a.lease_expires_at_unix_ms;
        renew(&mut activated.repository, reclaim_at);
        let claim_b = claim_command(create.embedding_job_id(), Uuid::now_v7(), reclaim_at);
        let EmbeddingJobClaimAck::Reclaimed(lease_b) =
            activated.repository.claim_embedding_job(&claim_b).unwrap()
        else {
            panic!("expired lease should be reclaimed");
        };
        assert_eq!(lease_b.job.attempt_generation, 2);
        assert_ne!(lease_a.lease_token, lease_b.lease_token);
        assert_eq!(
            activated
                .repository
                .complete_embedding_job(&completion(&lease_a, reclaim_at - 1))
                .unwrap(),
            EmbeddingJobCompletionAck::StaleLease
        );

        let reuse_at = lease_b.lease_expires_at_unix_ms;
        renew(&mut activated.repository, reuse_at);
        assert_eq!(
            activated
                .repository
                .claim_embedding_job(&claim_command(create.embedding_job_id(), token_a, reuse_at,))
                .unwrap(),
            EmbeddingJobClaimAck::Conflict
        );
    }

    #[test]
    fn canonical_chain_rejects_predated_same_owner_reclaim() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-predated-reclaim");
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let create = create_command(&vector_space, created_at);
        activated.repository.create_embedding_job(&create).unwrap();
        let claim_a = claim_command(create.embedding_job_id(), Uuid::now_v7(), created_at + 1);
        let EmbeddingJobClaimAck::Claimed(lease_a) =
            activated.repository.claim_embedding_job(&claim_a).unwrap()
        else {
            panic!("first lease should be claimed");
        };
        renew(&mut activated.repository, lease_a.lease_expires_at_unix_ms);
        let claim_b = claim_command(
            create.embedding_job_id(),
            Uuid::now_v7(),
            lease_a.lease_expires_at_unix_ms,
        );
        let EmbeddingJobClaimAck::Reclaimed(lease_b) =
            activated.repository.claim_embedding_job(&claim_b).unwrap()
        else {
            panic!("expired lease should be reclaimed");
        };
        let predated_at = claim_a.observed_at_unix_ms + 1;
        let predated_expiry = predated_at + EMBEDDING_JOB_LEASE_MILLIS;
        let tampered_hash = state_hash(
            claim_b.embedding_job_state_event_id,
            create.embedding_job_id(),
            Some(activated.identity.process_instance_id),
            "claimed",
            lease_b.job.attempt_generation,
            None,
            predated_at,
            Some(lease_b.lease_token),
            Some(predated_expiry),
            lease_b.job.attempt_count,
            lease_b.job.next_eligible_at_unix_ms,
            None,
            None,
        )
        .unwrap();
        activated
            .repository
            .connection
            .execute(
                "UPDATE embedding_job_state_events
                 SET created_at_unix_ms = ?1, lease_expires_at_unix_ms = ?2,
                     canonical_payload_hash = ?3
                 WHERE embedding_job_state_event_id = ?4",
                params![
                    predated_at,
                    predated_expiry,
                    tampered_hash,
                    claim_b.embedding_job_state_event_id.to_string(),
                ],
            )
            .unwrap();
        let current = load_job(&activated.repository.connection, create.embedding_job_id())
            .unwrap()
            .unwrap();
        let current_hash = job_hash(
            &current.embedding_job_id,
            &current.vector_space_id,
            &current.canonical_query_hash,
            &current.content_hash,
            Some(activated.identity.process_instance_id),
            Some(lease_b.lease_token),
            Some(predated_expiry),
            current.attempt_generation,
            current.attempt_count,
            current.next_eligible_at_unix_ms,
            current.terminal_error_class.as_deref(),
            current.failure_propagation_cursor.as_deref(),
            current.failure_propagation_complete,
            current.reset_actor.as_deref(),
            current.reset_reason.as_deref(),
            current.created_at_unix_ms,
        )
        .unwrap();
        activated
            .repository
            .connection
            .execute(
                "UPDATE embedding_jobs
                 SET lease_expires_at_unix_ms = ?1, canonical_payload_hash = ?2
                 WHERE embedding_job_id = ?3",
                params![predated_expiry, current_hash, create.embedding_job_id()],
            )
            .unwrap();
        let stored = load_job(&activated.repository.connection, create.embedding_job_id())
            .unwrap()
            .unwrap();
        assert!(
            verify_job(
                &activated.repository.connection,
                activated.identity.project_uuid,
                stored,
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn live_owner_is_protected_and_expired_owner_is_reclaimed() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut first = activate(&path, "embedding-processes");
        let mut second = activate(&path, "embedding-processes");
        let first_started =
            process_started_at(&first.repository, first.identity.process_instance_id);
        let second_started =
            process_started_at(&second.repository, second.identity.process_instance_id);
        let created_at = first_started.max(second_started) + 1;
        let vector_space = seed_vector_space(&first.repository, &first.identity, created_at);
        let create = create_command(&vector_space, created_at);
        first.repository.create_embedding_job(&create).unwrap();
        let EmbeddingJobClaimAck::Claimed(first_lease) = first
            .repository
            .claim_embedding_job(&claim_command(
                create.embedding_job_id(),
                Uuid::now_v7(),
                created_at + 1,
            ))
            .unwrap()
        else {
            panic!("first process should claim");
        };
        assert_eq!(
            second
                .repository
                .claim_embedding_job(&claim_command(
                    create.embedding_job_id(),
                    Uuid::now_v7(),
                    created_at + 2,
                ))
                .unwrap(),
            EmbeddingJobClaimAck::LeaseHeld {
                lease_expires_at_unix_ms: first_lease.lease_expires_at_unix_ms,
            }
        );

        let first_heartbeat_expiry: i64 = second
            .repository
            .connection
            .query_row(
                "SELECT heartbeat_expires_at_unix_ms FROM process_instances
                 WHERE process_instance_id = ?1",
                params![first.identity.process_instance_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert!(first_heartbeat_expiry < first_lease.lease_expires_at_unix_ms);
        renew(&mut second.repository, first_heartbeat_expiry);
        let EmbeddingJobClaimAck::Reclaimed(second_lease) = second
            .repository
            .claim_embedding_job(&claim_command(
                create.embedding_job_id(),
                Uuid::now_v7(),
                first_heartbeat_expiry,
            ))
            .unwrap()
        else {
            panic!("expired owner should permit early takeover");
        };
        assert_eq!(
            second_lease.lease_owner_process_instance_id,
            second.identity.process_instance_id
        );
    }

    #[test]
    fn two_connection_batch_claim_race_has_one_durable_winner() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut first = activate(&path, "embedding-claim-race");
        let second = activate(&path, "embedding-claim-race");
        let first_started =
            process_started_at(&first.repository, first.identity.process_instance_id);
        let second_started =
            process_started_at(&second.repository, second.identity.process_instance_id);
        let observed_at = first_started.max(second_started) + 1;
        let vector_space = seed_vector_space(&first.repository, &first.identity, observed_at);
        let create = create_command(&vector_space, observed_at);
        let EmbeddingJobCreateAck::Applied(job) =
            first.repository.create_embedding_job(&create).unwrap()
        else {
            panic!("race fixture should create one job");
        };
        let job_id = create.embedding_job_id().to_string();
        let barrier = Arc::new(Barrier::new(3));
        let first_barrier = barrier.clone();
        let first_job = job.clone();
        let first_space = vector_space.clone();
        let first_thread = thread::spawn(move || {
            let mut repository = first.repository;
            let command = batch_claim(&first_space, &[first_job], Uuid::now_v7(), observed_at + 1);
            first_barrier.wait();
            repository.claim_embedding_job_batch(&command).unwrap()
        });
        let second_barrier = barrier.clone();
        let second_space = vector_space;
        let second_thread = thread::spawn(move || {
            let mut repository = second.repository;
            let command = batch_claim(&second_space, &[job], Uuid::now_v7(), observed_at + 1);
            second_barrier.wait();
            repository.claim_embedding_job_batch(&command).unwrap()
        });
        barrier.wait();
        let results = [first_thread.join().unwrap(), second_thread.join().unwrap()];
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, EmbeddingJobBatchClaimAck::Claimed(_)))
                .count(),
            1
        );
        assert_eq!(
            results
                .iter()
                .filter(|result| matches!(result, EmbeddingJobBatchClaimAck::LeaseHeld { .. }))
                .count(),
            1
        );
        let connection = rusqlite::Connection::open(path).unwrap();
        let row: (i64, i64, i64) = connection
            .query_row(
                "SELECT attempt_generation, attempt_count,
                        (SELECT count(*) FROM embedding_job_state_events
                         WHERE embedding_job_id = ?1 AND state = 'claimed')
                 FROM embedding_jobs WHERE embedding_job_id = ?1",
                params![job_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(row, (1, 1, 1));
    }

    #[test]
    fn predated_batch_claim_waits_for_a_newer_live_lease_without_mutation() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut first = activate(&path, "embedding-predated-batch-claim");
        let mut second = activate(&path, "embedding-predated-batch-claim");
        let first_started =
            process_started_at(&first.repository, first.identity.process_instance_id);
        let second_started =
            process_started_at(&second.repository, second.identity.process_instance_id);
        let observed_at = first_started.max(second_started) + 1;
        let vector_space = seed_vector_space(&first.repository, &first.identity, observed_at);
        let create = create_command(&vector_space, observed_at);
        let EmbeddingJobCreateAck::Applied(job) =
            first.repository.create_embedding_job(&create).unwrap()
        else {
            panic!("predated claim fixture should create one job");
        };
        let stale_claim = batch_claim(
            &vector_space,
            std::slice::from_ref(&job),
            Uuid::now_v7(),
            observed_at + 1,
        );
        let winner_claim = batch_claim(
            &vector_space,
            std::slice::from_ref(&job),
            Uuid::now_v7(),
            observed_at + 2,
        );
        let EmbeddingJobBatchClaimAck::Claimed(leases) = first
            .repository
            .claim_embedding_job_batch(&winner_claim)
            .unwrap()
        else {
            panic!("newer claim should acquire the lease");
        };

        assert_eq!(
            second
                .repository
                .claim_embedding_job_batch(&stale_claim)
                .unwrap(),
            EmbeddingJobBatchClaimAck::LeaseHeld {
                embedding_job_id: job.embedding_job_id.clone(),
                lease_expires_at_unix_ms: leases[0].lease_expires_at_unix_ms,
            }
        );
        assert_eq!(
            second
                .repository
                .connection
                .query_row(
                    "SELECT COUNT(*) FROM health_events
                     WHERE stable_class = 'router.ledger.integrity_conflict'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );

        assert_eq!(
            first
                .repository
                .stop_process(
                    ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), observed_at + 3).unwrap()
                )
                .unwrap(),
            ProcessCommandAck::Applied
        );
        assert_eq!(
            second
                .repository
                .claim_embedding_job_batch(&stale_claim)
                .unwrap(),
            EmbeddingJobBatchClaimAck::Conflict
        );
        let durable: (i64, i64, i64) = second
            .repository
            .connection
            .query_row(
                "SELECT attempt_generation, attempt_count,
                        (SELECT COUNT(*) FROM embedding_job_state_events
                         WHERE embedding_job_id = ?1 AND state = 'claimed')
                 FROM embedding_jobs WHERE embedding_job_id = ?1",
                params![job.embedding_job_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(durable, (1, 1, 1));
    }

    #[test]
    fn stopped_owner_permits_takeover_before_lease_expiry() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut first = activate(&path, "embedding-stopped-owner");
        let mut second = activate(&path, "embedding-stopped-owner");
        let first_started =
            process_started_at(&first.repository, first.identity.process_instance_id);
        let second_started =
            process_started_at(&second.repository, second.identity.process_instance_id);
        let observed_at = first_started.max(second_started) + 1;
        let vector_space = seed_vector_space(&first.repository, &first.identity, observed_at);
        let create = create_command(&vector_space, observed_at);
        first.repository.create_embedding_job(&create).unwrap();
        let EmbeddingJobClaimAck::Claimed(first_lease) = first
            .repository
            .claim_embedding_job(&claim_command(
                create.embedding_job_id(),
                Uuid::now_v7(),
                observed_at + 1,
            ))
            .unwrap()
        else {
            panic!("first process should claim");
        };
        assert_eq!(
            first
                .repository
                .stop_process(
                    ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), observed_at + 2).unwrap()
                )
                .unwrap(),
            ProcessCommandAck::Applied
        );
        renew(&mut second.repository, observed_at + 3);
        let EmbeddingJobClaimAck::Reclaimed(second_lease) = second
            .repository
            .claim_embedding_job(&claim_command(
                create.embedding_job_id(),
                Uuid::now_v7(),
                observed_at + 3,
            ))
            .unwrap()
        else {
            panic!("terminal owner should permit early takeover");
        };
        assert!(second_lease.lease_expires_at_unix_ms > observed_at + 3);
        assert!(first_lease.lease_expires_at_unix_ms > observed_at + 3);
    }

    #[test]
    fn exact_expiry_reclaim_fences_live_old_owner_completion() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut first = activate(&path, "embedding-expiry-fence");
        let mut second = activate(&path, "embedding-expiry-fence");
        let first_started =
            process_started_at(&first.repository, first.identity.process_instance_id);
        let second_started =
            process_started_at(&second.repository, second.identity.process_instance_id);
        let observed_at = first_started.max(second_started) + 1;
        let vector_space = seed_vector_space(&first.repository, &first.identity, observed_at);
        let create = create_command(&vector_space, observed_at);
        first.repository.create_embedding_job(&create).unwrap();
        let EmbeddingJobClaimAck::Claimed(first_lease) = first
            .repository
            .claim_embedding_job(&claim_command(
                create.embedding_job_id(),
                Uuid::now_v7(),
                observed_at + 1,
            ))
            .unwrap()
        else {
            panic!("first process should claim");
        };
        let expiry = first_lease.lease_expires_at_unix_ms;
        renew(&mut first.repository, expiry);
        renew(&mut second.repository, expiry);
        assert_eq!(
            first
                .repository
                .complete_embedding_job(&completion(&first_lease, expiry))
                .unwrap(),
            EmbeddingJobCompletionAck::StaleLease
        );
        let EmbeddingJobClaimAck::Reclaimed(second_lease) = second
            .repository
            .claim_embedding_job(&claim_command(
                create.embedding_job_id(),
                Uuid::now_v7(),
                expiry,
            ))
            .unwrap()
        else {
            panic!("lease should be reclaimable at the exact expiry");
        };
        assert_eq!(second_lease.job.attempt_generation, 2);
        assert_eq!(
            first
                .repository
                .complete_embedding_job(&completion(&first_lease, expiry - 1))
                .unwrap(),
            EmbeddingJobCompletionAck::StaleLease
        );
    }

    #[test]
    fn completion_and_exact_expiry_reclaim_have_one_serializable_winner() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = learning_config(&path, "embedding-completion-reclaim-race");
        let mut first = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let mut second = LedgerRepository::activate_at(&config, 1_001).unwrap();
        let project_uuid = first.identity.project_uuid;
        let second_process = second.identity.process_instance_id;
        let created_at = 1_002;
        let vector_space = seed_vector_space(&first.repository, &first.identity, created_at);
        let create = create_command(&vector_space, created_at);
        first.repository.create_embedding_job(&create).unwrap();
        let job_id = create.embedding_job_id().to_string();
        let EmbeddingJobClaimAck::Claimed(first_lease) = first
            .repository
            .claim_embedding_job(&claim_command(&job_id, Uuid::now_v7(), created_at + 1))
            .unwrap()
        else {
            panic!("first process should claim the job");
        };
        let expiry = first_lease.lease_expires_at_unix_ms;
        renew(&mut first.repository, expiry);
        renew(&mut second.repository, expiry);

        let completion = completion(&first_lease, expiry - 1);
        let reclaim = claim_command(&job_id, Uuid::now_v7(), expiry);
        let barrier = Arc::new(Barrier::new(3));
        let completion_barrier = barrier.clone();
        let completion_thread = thread::spawn(move || {
            let mut repository = first.repository;
            completion_barrier.wait();
            let acknowledgement = repository.complete_embedding_job(&completion).unwrap();
            (repository, acknowledgement)
        });
        let reclaim_barrier = barrier.clone();
        let reclaim_thread = thread::spawn(move || {
            let mut repository = second.repository;
            reclaim_barrier.wait();
            let acknowledgement = repository.claim_embedding_job(&reclaim).unwrap();
            (repository, acknowledgement)
        });
        barrier.wait();
        let (_first_repository, completion_ack) = completion_thread.join().unwrap();
        let (second_repository, reclaim_ack) = reclaim_thread.join().unwrap();

        let stored = load_job(&second_repository.connection, &job_id)
            .unwrap()
            .unwrap();
        let verified = verify_job(&second_repository.connection, project_uuid, stored)
            .unwrap()
            .expect("race winner must leave a canonical job and state chain");
        match (completion_ack, reclaim_ack) {
            (EmbeddingJobCompletionAck::Applied { .. }, EmbeddingJobClaimAck::Terminal) => {
                assert_eq!(verified.states.len(), 3);
                assert_eq!(verified.states.last().unwrap().state, "completed");
                assert_eq!(verified.stored.attempt_generation, 1);
                assert_eq!(verified.stored.attempt_count, 1);
                assert!(current_lease(&verified.stored).unwrap().is_none());
            }
            (
                EmbeddingJobCompletionAck::StaleLease,
                EmbeddingJobClaimAck::Reclaimed(second_lease),
            ) => {
                assert_eq!(verified.states.len(), 4);
                assert_eq!(verified.states.last().unwrap().state, "claimed");
                assert_eq!(verified.stored.attempt_generation, 2);
                assert_eq!(verified.stored.attempt_count, 2);
                assert_eq!(
                    current_lease(&verified.stored).unwrap(),
                    Some((
                        second_process,
                        second_lease.lease_token,
                        second_lease.lease_expires_at_unix_ms,
                    ))
                );
            }
            outcomes => panic!("lease race produced non-serializable outcomes: {outcomes:?}"),
        }
    }

    #[test]
    fn fixed_busy_timeout_leaves_claim_state_unchanged() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-busy");
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let create = create_command(&vector_space, created_at);
        activated.repository.create_embedding_job(&create).unwrap();
        let lock = rusqlite::Connection::open(&path).unwrap();
        lock.execute_batch("PRAGMA busy_timeout = 0; BEGIN IMMEDIATE;")
            .unwrap();
        let started = std::time::Instant::now();
        let error = activated
            .repository
            .claim_embedding_job(&claim_command(
                create.embedding_job_id(),
                Uuid::now_v7(),
                created_at + 1,
            ))
            .unwrap_err();
        assert_eq!(error.class(), LedgerErrorClass::Busy);
        assert!(started.elapsed() >= std::time::Duration::from_millis(4_500));
        lock.execute_batch("ROLLBACK;").unwrap();
        let row: (i64, i64, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT attempt_generation, attempt_count,
                        (SELECT count(*) FROM embedding_job_state_events
                         WHERE embedding_job_id = ?1 AND state = 'claimed')
                 FROM embedding_jobs WHERE embedding_job_id = ?1",
                params![create.embedding_job_id()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(row, (0, 0, 0));
    }

    #[test]
    fn project_mismatch_activation_cannot_mutate_a_job() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-project-a");
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let create = create_command(&vector_space, created_at);
        activated.repository.create_embedding_job(&create).unwrap();
        let error = match LedgerRepository::activate(&config(&path, "embedding-project-b")) {
            Ok(_) => panic!("project mismatch should refuse activation"),
            Err(error) => error,
        };
        assert_eq!(error.class(), LedgerErrorClass::ProjectIdMismatch);
        let row: (i64, i64, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT attempt_generation, attempt_count,
                        (SELECT count(*) FROM embedding_job_state_events
                         WHERE embedding_job_id = ?1)
                 FROM embedding_jobs WHERE embedding_job_id = ?1",
                params![create.embedding_job_id()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(row, (0, 0, 1));
    }

    #[test]
    fn malformed_vector_space_and_create_event_collisions_are_conflicts() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-conflicts");
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let missing_space = "a".repeat(64);
        let missing = create_command(&missing_space, created_at);
        assert_eq!(
            activated.repository.create_embedding_job(&missing).unwrap(),
            EmbeddingJobCreateAck::VectorSpaceNotFound
        );
        let vector_space = activated
            .identity
            .pool("pool-a")
            .and_then(|pool| pool.vector_space.as_ref())
            .unwrap()
            .vector_space_id
            .as_str()
            .to_string();
        let canonicalizer_json: String = activated
            .repository
            .connection
            .query_row(
                "SELECT canonicalizer_identity_json FROM vector_spaces
                 WHERE vector_space_id = ?1",
                [&vector_space],
                |row| row.get(0),
            )
            .unwrap();
        activated
            .repository
            .connection
            .execute(
                "UPDATE vector_spaces SET canonicalizer_identity_json = '[]'
                 WHERE vector_space_id = ?1",
                [&vector_space],
            )
            .unwrap();
        let malformed = create_command(&vector_space, created_at);
        assert_eq!(
            activated
                .repository
                .create_embedding_job(&malformed)
                .unwrap(),
            EmbeddingJobCreateAck::Conflict
        );

        activated
            .repository
            .connection
            .execute(
                "UPDATE vector_spaces SET canonicalizer_identity_json = ?1
                 WHERE vector_space_id = ?2",
                params![canonicalizer_json, vector_space],
            )
            .unwrap();
        seed_query(&activated.repository, created_at);
        let create = create_command(&vector_space, created_at);
        activated.repository.create_embedding_job(&create).unwrap();
        let query_hash = canonical_query().0;
        let collision = EmbeddingJobCreate::new(
            create.embedding_job_state_event_id,
            Uuid::now_v7(),
            vector_space,
            query_hash.clone(),
            query_hash,
            created_at + 1,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .create_embedding_job(&collision)
                .unwrap(),
            EmbeddingJobCreateAck::Conflict
        );
    }

    #[test]
    fn embedding_job_requires_a_fully_verified_canonical_query() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-query-authority");
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space = activated
            .identity
            .pool("pool-a")
            .and_then(|pool| pool.vector_space.as_ref())
            .unwrap()
            .vector_space_id
            .as_str()
            .to_string();

        assert_eq!(
            activated
                .repository
                .create_embedding_job(&create_command(&vector_space, created_at))
                .unwrap(),
            EmbeddingJobCreateAck::Conflict
        );

        seed_query(&activated.repository, created_at);
        let (query_hash, _) = canonical_query();
        activated
            .repository
            .connection
            .execute(
                "UPDATE canonical_routing_queries
                 SET canonical_query_json = '{}', canonical_size_bytes = 2
                 WHERE canonical_query_hash = ?1",
                [query_hash],
            )
            .unwrap();
        assert_eq!(
            activated
                .repository
                .create_embedding_job(&create_command(&vector_space, created_at))
                .unwrap_err()
                .class(),
            LedgerErrorClass::CorruptDatabase
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM embedding_jobs", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            0
        );
    }

    #[test]
    fn completion_retry_requires_the_original_content_hash() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-content-fence");
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let create = create_command(&vector_space, created_at);
        activated.repository.create_embedding_job(&create).unwrap();
        let EmbeddingJobClaimAck::Claimed(lease) = activated
            .repository
            .claim_embedding_job(&claim_command(
                create.embedding_job_id(),
                Uuid::now_v7(),
                created_at + 1,
            ))
            .unwrap()
        else {
            panic!("job should be claimed");
        };
        let completion = completion(&lease, created_at + 2);
        assert!(matches!(
            activated
                .repository
                .complete_embedding_job(&completion)
                .unwrap(),
            EmbeddingJobCompletionAck::Applied { .. }
        ));
        let mut mismatched = completion;
        mismatched.content_hash = "d".repeat(64);
        assert_eq!(
            activated
                .repository
                .complete_embedding_job(&mismatched)
                .unwrap(),
            EmbeddingJobCompletionAck::Conflict
        );
    }

    #[test]
    fn v1_create_rejects_content_hash_mismatch_at_both_authority_boundaries() {
        let (query_hash, _) = canonical_query();
        let error = EmbeddingJobCreate::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            "a".repeat(64),
            query_hash,
            "d".repeat(64),
            0,
        )
        .unwrap_err();
        assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);

        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-content-identity");
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let mut command = create_command(&vector_space, created_at);
        command.content_hash = "d".repeat(64);
        assert_eq!(
            activated.repository.create_embedding_job(&command).unwrap(),
            EmbeddingJobCreateAck::Conflict
        );
        assert!(
            load_job(&activated.repository.connection, command.embedding_job_id())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn stored_content_hash_mismatch_is_rejected() {
        let stored = tamper_pending_job_column(
            "embedding-stored-content-tamper",
            "content_hash = 'dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd'",
        );
        assert_ne!(stored.content_hash, stored.canonical_query_hash);
    }

    #[test]
    fn terminal_error_class_tamper_is_rejected() {
        let stored = tamper_pending_job_column(
            "embedding-terminal-error-tamper",
            "terminal_error_class = 'provider_error'",
        );
        assert_eq!(
            stored.terminal_error_class.as_deref(),
            Some("provider_error")
        );
    }

    #[test]
    fn failure_propagation_cursor_tamper_is_rejected() {
        let cursor = Uuid::now_v7().to_string();
        let stored = tamper_pending_job_column(
            "embedding-propagation-cursor-tamper",
            &format!("failure_propagation_cursor = '{cursor}'"),
        );
        assert_eq!(
            stored.failure_propagation_cursor.as_deref(),
            Some(cursor.as_str())
        );
    }

    #[test]
    fn failure_propagation_complete_tamper_is_rejected() {
        let stored = tamper_pending_job_column(
            "embedding-propagation-complete-tamper",
            "failure_propagation_complete = 1",
        );
        assert!(stored.failure_propagation_complete);
    }

    #[test]
    fn reset_actor_tamper_is_rejected() {
        let stored =
            tamper_pending_job_column("embedding-reset-actor-tamper", "reset_actor = 'operator'");
        assert_eq!(stored.reset_actor.as_deref(), Some("operator"));
    }

    #[test]
    fn reset_reason_tamper_is_rejected() {
        let stored = tamper_pending_job_column(
            "embedding-reset-reason-tamper",
            "reset_reason = 'manual reset'",
        );
        assert_eq!(stored.reset_reason.as_deref(), Some("manual reset"));
    }

    #[test]
    fn batch_claim_requires_the_claimants_current_mapping_authority() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let original = learning_config(&path, "embedding-claim-current-authority");
        let mut first = LedgerRepository::activate(&original).unwrap();
        let first_started_at =
            process_started_at(&first.repository, first.identity.process_instance_id);
        let original_space =
            seed_vector_space(&first.repository, &first.identity, first_started_at);
        let job = create_named_job(
            &mut first.repository,
            &original_space,
            first_started_at,
            "historical embedding claim",
        );
        assert_eq!(
            first
                .repository
                .stop_process(
                    ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), first_started_at).unwrap(),
                )
                .unwrap(),
            ProcessCommandAck::Applied
        );
        drop(first);

        let mut changed = original.clone();
        changed.embedders[0].model = "embedding-model-b".to_string();
        let mut current = LedgerRepository::activate(&changed).unwrap();
        let current_space = current
            .identity
            .pool("pool-a")
            .and_then(|pool| pool.vector_space.as_ref())
            .unwrap()
            .vector_space_id
            .as_str();
        assert_ne!(current_space, original_space);
        let current_started_at =
            process_started_at(&current.repository, current.identity.process_instance_id);
        let before: (i64, i64, Option<String>, i64) = current
            .repository
            .connection
            .query_row(
                "SELECT attempt_generation, attempt_count, lease_token,
                        (SELECT count(*) FROM embedding_job_state_events
                         WHERE embedding_job_id = embedding_jobs.embedding_job_id)
                 FROM embedding_jobs WHERE embedding_job_id = ?1",
                params![job.embedding_job_id.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(before, (0, 0, None, 1));
        let unauthorized = batch_claim(
            &original_space,
            std::slice::from_ref(&job),
            Uuid::now_v7(),
            current_started_at,
        );
        assert_eq!(
            current
                .repository
                .claim_embedding_job_batch(&unauthorized)
                .unwrap(),
            EmbeddingJobBatchClaimAck::VectorSpaceUnauthorized
        );
        let after: (i64, i64, Option<String>, i64) = current
            .repository
            .connection
            .query_row(
                "SELECT attempt_generation, attempt_count, lease_token,
                        (SELECT count(*) FROM embedding_job_state_events
                         WHERE embedding_job_id = embedding_jobs.embedding_job_id)
                 FROM embedding_jobs WHERE embedding_job_id = ?1",
                params![job.embedding_job_id.as_str()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(after, before);
        drop(current);

        let mut restored_config = original.clone();
        restored_config.retention_days += 1;
        let mut restored = LedgerRepository::activate(&restored_config).unwrap();
        let restored_started_at =
            process_started_at(&restored.repository, restored.identity.process_instance_id);
        let authorized = batch_claim(&original_space, &[job], Uuid::now_v7(), restored_started_at);
        assert!(matches!(
            restored
                .repository
                .claim_embedding_job_batch(&authorized)
                .unwrap(),
            EmbeddingJobBatchClaimAck::Claimed(leases) if leases.len() == 1
        ));
    }

    #[test]
    fn batch_claim_is_nonempty_profile_bounded_and_all_or_none() {
        assert!(
            EmbeddingJobBatchClaim::new(
                Uuid::now_v7(),
                VectorSpaceId::new("a".repeat(64)).unwrap(),
                Uuid::now_v7(),
                0,
                Vec::new(),
            )
            .is_err()
        );

        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate_with_batch_size(&path, "embedding-batch-bound", 2);
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let jobs = [
            create_named_job(
                &mut activated.repository,
                &vector_space,
                created_at,
                "batch bound a",
            ),
            create_named_job(
                &mut activated.repository,
                &vector_space,
                created_at,
                "batch bound b",
            ),
            create_named_job(
                &mut activated.repository,
                &vector_space,
                created_at,
                "batch bound c",
            ),
        ];
        let oversized = batch_claim(&vector_space, &jobs, Uuid::now_v7(), created_at + 1);
        assert_eq!(
            activated
                .repository
                .claim_embedding_job_batch(&oversized)
                .unwrap(),
            EmbeddingJobBatchClaimAck::BatchTooLarge { max_batch_size: 2 }
        );
        for job in &jobs {
            let stored = load_job(&activated.repository.connection, &job.embedding_job_id)
                .unwrap()
                .unwrap();
            assert!(current_lease(&stored).unwrap().is_none());
        }

        let missing = EmbeddingJobSnapshot {
            embedding_job_id: "f".repeat(64),
            ..jobs[1].clone()
        };
        let mixed = batch_claim(
            &vector_space,
            &[jobs[0].clone(), missing],
            Uuid::now_v7(),
            created_at + 1,
        );
        assert_eq!(
            activated
                .repository
                .claim_embedding_job_batch(&mixed)
                .unwrap(),
            EmbeddingJobBatchClaimAck::JobNotFound {
                embedding_job_id: "f".repeat(64)
            }
        );
        assert!(
            current_lease(
                &load_job(&activated.repository.connection, &jobs[0].embedding_job_id,)
                    .unwrap()
                    .unwrap(),
            )
            .unwrap()
            .is_none()
        );

        let claim = batch_claim(&vector_space, &jobs[..1], Uuid::now_v7(), created_at + 1);
        assert!(matches!(
            activated
                .repository
                .claim_embedding_job_batch(&claim)
                .unwrap(),
            EmbeddingJobBatchClaimAck::Claimed(ref leases) if leases.len() == 1
        ));
    }

    #[test]
    fn batch_completion_is_atomic_vector_backed_and_exactly_idempotent() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate_with_batch_size(&path, "embedding-batch-complete", 4);
        let project_uuid = activated.identity.project_uuid;
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let jobs = [
            create_named_job(
                &mut activated.repository,
                &vector_space,
                created_at,
                "batch completion a",
            ),
            create_named_job(
                &mut activated.repository,
                &vector_space,
                created_at,
                "batch completion b",
            ),
        ];
        let claim = batch_claim(&vector_space, &jobs, Uuid::now_v7(), created_at + 1);
        let EmbeddingJobBatchClaimAck::Claimed(leases) = activated
            .repository
            .claim_embedding_job_batch(&claim)
            .unwrap()
        else {
            panic!("batch should be claimed");
        };
        let completion = batch_completion(
            &activated.repository,
            project_uuid,
            &vector_space,
            &leases,
            created_at + 2,
        );
        let mut stale = completion.clone();
        stale.items[1].attempt_generation += 1;
        assert_eq!(
            activated
                .repository
                .complete_embedding_job_batch(&stale)
                .unwrap(),
            EmbeddingJobBatchCompletionAck::StaleLease {
                embedding_job_id: leases[1].job.embedding_job_id.clone()
            }
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM embeddings", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
        for lease in &leases {
            assert!(
                current_lease(
                    &load_job(
                        &activated.repository.connection,
                        &lease.job.embedding_job_id,
                    )
                    .unwrap()
                    .unwrap(),
                )
                .unwrap()
                .is_some()
            );
        }

        let applied = activated
            .repository
            .complete_embedding_job_batch(&completion)
            .unwrap();
        assert!(matches!(
            applied,
            EmbeddingJobBatchCompletionAck::Applied(ref results) if results.len() == 2
        ));
        assert!(matches!(
            activated
                .repository
                .complete_embedding_job_batch(&completion)
                .unwrap(),
            EmbeddingJobBatchCompletionAck::AlreadyApplied(ref results) if results.len() == 2
        ));

        let mut conflicting = completion.clone();
        conflicting.items[1].vector =
            authoritative_vector(&activated.repository, project_uuid, &vector_space, 3);
        assert_eq!(
            activated
                .repository
                .complete_embedding_job_batch(&conflicting)
                .unwrap(),
            EmbeddingJobBatchCompletionAck::Conflict
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM embeddings", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            2
        );
    }

    #[test]
    fn batch_reclaim_appends_orphan_before_the_next_claim() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-batch-reclaim");
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let jobs = [create_named_job(
            &mut activated.repository,
            &vector_space,
            created_at,
            "batch reclaim",
        )];
        let first = batch_claim(&vector_space, &jobs, Uuid::now_v7(), created_at + 1);
        let EmbeddingJobBatchClaimAck::Claimed(first_lease) = activated
            .repository
            .claim_embedding_job_batch(&first)
            .unwrap()
        else {
            panic!("first claim should apply");
        };
        assert!(matches!(
            activated
                .repository
                .claim_embedding_job_batch(&first)
                .unwrap(),
            EmbeddingJobBatchClaimAck::AlreadyApplied(ref leases) if leases.len() == 1
        ));
        let reclaim_at = first_lease[0].lease_expires_at_unix_ms;
        renew(&mut activated.repository, reclaim_at);
        let reclaim = batch_claim(&vector_space, &jobs, Uuid::now_v7(), reclaim_at);
        let EmbeddingJobBatchClaimAck::Claimed(second_lease) = activated
            .repository
            .claim_embedding_job_batch(&reclaim)
            .unwrap()
        else {
            panic!("expired lease should be reclaimed");
        };
        assert_eq!(second_lease[0].job.attempt_generation, 2);
        assert_eq!(second_lease[0].job.attempt_count, 2);
        let states = load_job_states(&activated.repository.connection, &jobs[0].embedding_job_id)
            .unwrap()
            .into_iter()
            .map(|state| state.state)
            .collect::<Vec<_>>();
        assert_eq!(
            states,
            ["pending", "claimed", "orphaned_in_flight", "claimed"]
        );
    }

    #[test]
    fn release_and_retries_converge_to_attempt_five_quarantine() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-attempt-limit");
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let jobs = [create_named_job(
            &mut activated.repository,
            &vector_space,
            created_at,
            "attempt limit",
        )];
        let mut claim_at = created_at + 1;
        for attempt in 1..=5 {
            let claim = batch_claim(&vector_space, &jobs, Uuid::now_v7(), claim_at);
            let EmbeddingJobBatchClaimAck::Claimed(leases) = activated
                .repository
                .claim_embedding_job_batch(&claim)
                .unwrap()
            else {
                panic!("attempt {attempt} should claim");
            };
            let lease = &leases[0];
            let resolved_at = claim_at + 1;
            let kind = if attempt == 1 {
                EmbeddingJobResolutionKind::Released
            } else {
                EmbeddingJobResolutionKind::RetryScheduled {
                    stable_error_class: "provider_timeout".to_string(),
                    next_eligible_at_unix_ms: resolved_at + 1,
                }
            };
            let resolution = EmbeddingJobResolution::new(
                Uuid::now_v7(),
                Uuid::now_v7(),
                lease.job.embedding_job_id.clone(),
                lease.lease_token,
                lease.job.attempt_generation,
                lease.job.content_hash.clone(),
                resolved_at,
                kind,
            )
            .unwrap();
            let acknowledgement = activated
                .repository
                .resolve_embedding_job(&resolution)
                .unwrap();
            let expected_state = if attempt == 1 {
                EmbeddingJobResolvedState::Released
            } else if attempt == 5 {
                EmbeddingJobResolvedState::Quarantined
            } else {
                EmbeddingJobResolvedState::RetryScheduled
            };
            assert!(matches!(
                activated
                    .repository
                    .resolve_embedding_job(&resolution)
                    .unwrap(),
                EmbeddingJobResolutionAck::AlreadyApplied { state, .. }
                    if state == expected_state
            ));
            if attempt == 5 {
                let EmbeddingJobResolutionAck::Applied { job, state, .. } = acknowledgement else {
                    panic!("fifth failure should terminalize");
                };
                assert_eq!(state, EmbeddingJobResolvedState::Quarantined);
                assert_eq!(job.attempt_count, 5);
                assert_eq!(
                    job.terminal_error_class.as_deref(),
                    Some("provider_timeout")
                );
            } else {
                let EmbeddingJobResolutionAck::Applied { job, state, .. } = acknowledgement else {
                    panic!("nonterminal resolution should apply");
                };
                assert_eq!(state, expected_state);
                claim_at = job.next_eligible_at_unix_ms;
            }
        }
    }

    #[test]
    fn permanent_provider_failure_quarantines_on_the_first_attempt() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-immediate-quarantine");
        let project_uuid = activated.identity.project_uuid;
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let jobs = [create_named_job(
            &mut activated.repository,
            &vector_space,
            created_at,
            "immediate quarantine",
        )];
        let sibling_jobs = [create_named_job(
            &mut activated.repository,
            &vector_space,
            created_at,
            "suppressed sibling",
        )];
        assert_ne!(jobs[0].embedding_job_id, sibling_jobs[0].embedding_job_id);
        let claim = batch_claim(&vector_space, &jobs, Uuid::now_v7(), created_at + 1);
        let EmbeddingJobBatchClaimAck::Claimed(leases) = activated
            .repository
            .claim_embedding_job_batch(&claim)
            .unwrap()
        else {
            panic!("job should claim");
        };
        let lease = &leases[0];
        let resolution = EmbeddingJobResolution::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            lease.job.embedding_job_id.clone(),
            lease.lease_token,
            lease.job.attempt_generation,
            lease.job.content_hash.clone(),
            created_at + 2,
            EmbeddingJobResolutionKind::Quarantined {
                stable_error_class: "embedder_authentication".to_string(),
            },
        )
        .unwrap();
        let EmbeddingJobResolutionAck::Applied { job, state, .. } = activated
            .repository
            .resolve_embedding_job(&resolution)
            .unwrap()
        else {
            panic!("permanent failure should quarantine");
        };
        assert_eq!(state, EmbeddingJobResolvedState::Quarantined);
        assert_eq!(job.attempt_count, 1);
        assert_eq!(
            job.terminal_error_class.as_deref(),
            Some("embedder_authentication")
        );
        let vector_space_id = VectorSpaceId::new(vector_space.clone()).unwrap();
        assert!(
            embedding_space_is_degraded(
                &activated.repository.connection,
                project_uuid,
                &vector_space_id,
            )
            .unwrap()
        );
        assert!(
            crate::ledger::repository::background_work::select_embedding_work(
                &activated.repository.connection,
                project_uuid,
                &activated.identity.config_generation_id,
                created_at + 3,
                None,
                256,
            )
            .unwrap()
            .is_empty()
        );
        assert!(matches!(
            activated
                .repository
                .claim_embedding_job_batch(&batch_claim(
                    &vector_space,
                    &sibling_jobs,
                    Uuid::now_v7(),
                    created_at + 3,
                ))
                .unwrap(),
            EmbeddingJobBatchClaimAck::Terminal { embedding_job_id }
                if embedding_job_id == sibling_jobs[0].embedding_job_id
        ));
        assert!(matches!(
            activated
                .repository
                .claim_embedding_job_batch(&batch_claim(
                    &vector_space,
                    &jobs,
                    Uuid::now_v7(),
                    created_at + 3,
                ))
                .unwrap(),
            EmbeddingJobBatchClaimAck::Terminal { .. }
        ));
    }

    #[test]
    fn terminal_failure_requires_propagation_before_reset_and_fences_old_lease() {
        assert!(
            EmbeddingFailurePropagationUpdate::new(
                "a".repeat(64),
                "b".repeat(64),
                0,
                None,
                None,
                true,
            )
            .is_err()
        );
        assert!(
            EmbeddingFailurePropagationUpdate::new(
                "a".repeat(64),
                "b".repeat(64),
                1,
                None,
                None,
                false,
            )
            .is_err()
        );
        let cursor = Uuid::now_v7();
        let cursor_string = cursor.to_string();
        assert!(
            EmbeddingFailurePropagationUpdate::new(
                "a".repeat(64),
                "b".repeat(64),
                1,
                Some(cursor),
                None,
                true,
            )
            .is_err()
        );
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-terminal-reset");
        let project_uuid = activated.identity.project_uuid;
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let jobs = [create_named_job(
            &mut activated.repository,
            &vector_space,
            created_at,
            "terminal reset",
        )];
        let claim = batch_claim(&vector_space, &jobs, Uuid::now_v7(), created_at + 1);
        let EmbeddingJobBatchClaimAck::Claimed(leases) = activated
            .repository
            .claim_embedding_job_batch(&claim)
            .unwrap()
        else {
            panic!("job should claim");
        };
        let old_lease = leases[0].clone();
        let resolution = EmbeddingJobResolution::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            old_lease.job.embedding_job_id.clone(),
            old_lease.lease_token,
            old_lease.job.attempt_generation,
            old_lease.job.content_hash.clone(),
            created_at + 2,
            EmbeddingJobResolutionKind::TerminalFailure {
                stable_error_class: "invalid_response".to_string(),
            },
        )
        .unwrap();
        let EmbeddingJobResolutionAck::Applied { job: terminal, .. } = activated
            .repository
            .resolve_embedding_job(&resolution)
            .unwrap()
        else {
            panic!("terminal failure should apply");
        };
        let reset_event_id = Uuid::now_v7();
        let reset = EmbeddingJobReset::new(
            reset_event_id,
            Uuid::now_v7(),
            terminal.embedding_job_id.clone(),
            terminal.attempt_generation,
            terminal.canonical_payload_hash.clone(),
            "operator",
            "reviewed provider failure",
            created_at + 3,
        )
        .unwrap();
        assert_eq!(
            activated.repository.reset_embedding_job(&reset).unwrap(),
            EmbeddingJobResetAck::PropagationIncomplete
        );

        let progress = EmbeddingFailurePropagationUpdate::new(
            terminal.embedding_job_id.clone(),
            terminal.canonical_payload_hash.clone(),
            terminal.attempt_generation,
            None,
            Some(cursor),
            false,
        )
        .unwrap();
        let partially_propagated = {
            let transaction = activated.repository.connection.transaction().unwrap();
            let EmbeddingFailurePropagationAck::Applied(job) =
                update_embedding_failure_propagation_in_transaction(
                    &transaction,
                    project_uuid,
                    &progress,
                )
                .unwrap()
            else {
                panic!("propagation completion should apply");
            };
            transaction.commit().unwrap();
            job
        };
        assert_eq!(
            partially_propagated.failure_propagation_cursor.as_deref(),
            Some(cursor_string.as_str())
        );
        assert!(!partially_propagated.failure_propagation_complete);
        assert!(matches!(
            {
                let transaction = activated.repository.connection.transaction().unwrap();
                let acknowledgement = update_embedding_failure_propagation_in_transaction(
                    &transaction,
                    project_uuid,
                    &progress,
                )
                .unwrap();
                transaction.rollback().unwrap();
                acknowledgement
            },
            EmbeddingFailurePropagationAck::AlreadyApplied(_)
        ));
        let mut wrong_generation_retry = progress.clone();
        wrong_generation_retry.expected_attempt_generation += 1;
        assert_eq!(
            {
                let transaction = activated.repository.connection.transaction().unwrap();
                let acknowledgement = update_embedding_failure_propagation_in_transaction(
                    &transaction,
                    project_uuid,
                    &wrong_generation_retry,
                )
                .unwrap();
                transaction.rollback().unwrap();
                acknowledgement
            },
            EmbeddingFailurePropagationAck::Stale
        );
        let mut wrong_hash_retry = progress.clone();
        wrong_hash_retry.expected_canonical_payload_hash = "f".repeat(64);
        assert_eq!(
            {
                let transaction = activated.repository.connection.transaction().unwrap();
                let acknowledgement = update_embedding_failure_propagation_in_transaction(
                    &transaction,
                    project_uuid,
                    &wrong_hash_retry,
                )
                .unwrap();
                transaction.rollback().unwrap();
                acknowledgement
            },
            EmbeddingFailurePropagationAck::Stale
        );
        let incomplete_reset = EmbeddingJobReset::new(
            reset_event_id,
            reset.conflict_health_event_id,
            partially_propagated.embedding_job_id.clone(),
            partially_propagated.attempt_generation,
            partially_propagated.canonical_payload_hash.clone(),
            "operator",
            "reviewed provider failure",
            created_at + 3,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .reset_embedding_job(&incomplete_reset)
                .unwrap(),
            EmbeddingJobResetAck::PropagationIncomplete
        );
        let progress = EmbeddingFailurePropagationUpdate::new(
            partially_propagated.embedding_job_id.clone(),
            partially_propagated.canonical_payload_hash.clone(),
            partially_propagated.attempt_generation,
            Some(cursor),
            Some(cursor),
            true,
        )
        .unwrap();
        let propagated = {
            let transaction = activated.repository.connection.transaction().unwrap();
            let EmbeddingFailurePropagationAck::Applied(job) =
                update_embedding_failure_propagation_in_transaction(
                    &transaction,
                    project_uuid,
                    &progress,
                )
                .unwrap()
            else {
                panic!("propagation completion should apply");
            };
            transaction.commit().unwrap();
            job
        };
        assert_eq!(
            propagated.failure_propagation_cursor.as_deref(),
            Some(cursor_string.as_str())
        );
        assert!(propagated.failure_propagation_complete);
        let reset = EmbeddingJobReset::new(
            reset_event_id,
            reset.conflict_health_event_id,
            propagated.embedding_job_id.clone(),
            propagated.attempt_generation,
            propagated.canonical_payload_hash.clone(),
            "operator",
            "reviewed provider failure",
            created_at + 3,
        )
        .unwrap();
        let EmbeddingJobResetAck::Applied { job: reset_job, .. } =
            activated.repository.reset_embedding_job(&reset).unwrap()
        else {
            panic!("propagated terminal should reset");
        };
        assert_eq!(reset_job.attempt_count, 0);
        assert_eq!(
            reset_job.attempt_generation,
            terminal.attempt_generation + 1
        );
        assert!(reset_job.terminal_error_class.is_none());
        assert_eq!(
            activated.repository.reset_embedding_job(&reset).unwrap(),
            EmbeddingJobResetAck::AlreadyApplied {
                job: reset_job.clone(),
                state_event_hash: load_state_by_event_id(
                    &activated.repository.connection,
                    reset.embedding_job_state_event_id,
                )
                .unwrap()
                .unwrap()
                .canonical_payload_hash,
            }
        );

        let old_completion = batch_completion(
            &activated.repository,
            project_uuid,
            &vector_space,
            &[old_lease],
            created_at + 4,
        );
        assert!(matches!(
            activated
                .repository
                .complete_embedding_job_batch(&old_completion)
                .unwrap(),
            EmbeddingJobBatchCompletionAck::StaleLease { .. }
        ));
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM embeddings", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
        let claim = batch_claim(&vector_space, &[reset_job], Uuid::now_v7(), created_at + 4);
        let EmbeddingJobBatchClaimAck::Claimed(leases) = activated
            .repository
            .claim_embedding_job_batch(&claim)
            .unwrap()
        else {
            panic!("reset job should become claimable");
        };
        assert_eq!(leases[0].job.attempt_count, 1);
        assert_eq!(
            leases[0].job.attempt_generation,
            terminal.attempt_generation + 2
        );
    }

    #[test]
    fn canonical_state_chain_rejects_a_null_actor() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-actor");
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let create = create_command(&vector_space, created_at);
        activated.repository.create_embedding_job(&create).unwrap();
        let event_hash = state_hash(
            create.embedding_job_state_event_id,
            create.embedding_job_id(),
            None,
            "pending",
            0,
            None,
            created_at,
            None,
            None,
            0,
            created_at,
            None,
            None,
        )
        .unwrap();
        activated
            .repository
            .connection
            .execute(
                "UPDATE embedding_job_state_events
                 SET process_instance_id = NULL, canonical_payload_hash = ?1
                 WHERE embedding_job_state_event_id = ?2",
                params![event_hash, create.embedding_job_state_event_id.to_string()],
            )
            .unwrap();
        let retry = create_command(&vector_space, created_at);
        assert_eq!(
            activated.repository.create_embedding_job(&retry).unwrap(),
            EmbeddingJobCreateAck::Conflict
        );
    }

    struct DenyStart;

    impl TransactionStartGuard for DenyStart {
        fn permits_transaction(&self) -> bool {
            false
        }
    }

    #[test]
    fn post_begin_start_guard_leaves_no_job() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "embedding-start-guard");
        let created_at = process_started_at(
            &activated.repository,
            activated.identity.process_instance_id,
        ) + 1;
        let vector_space =
            seed_vector_space(&activated.repository, &activated.identity, created_at);
        let create = create_command(&vector_space, created_at);
        assert_eq!(
            activated
                .repository
                .create_embedding_job_with_start_check(&create, || Some(DenyStart))
                .unwrap(),
            EmbeddingJobCreateAck::TransactionNotStarted
        );
        assert!(
            load_job(&activated.repository.connection, create.embedding_job_id())
                .unwrap()
                .is_none()
        );
    }

    #[test]
    fn constructors_reject_non_v7_and_lease_expiry_overflow() {
        assert!(
            EmbeddingJobClaim::new(
                Uuid::now_v7(),
                Uuid::now_v7(),
                "a".repeat(64),
                Uuid::now_v7(),
                i64::MAX,
            )
            .is_err()
        );
        assert!(
            EmbeddingJobCompletion::new(
                Uuid::nil(),
                Uuid::now_v7(),
                "a".repeat(64),
                Uuid::now_v7(),
                1,
                "c".repeat(64),
                0,
            )
            .is_err()
        );
    }
}
