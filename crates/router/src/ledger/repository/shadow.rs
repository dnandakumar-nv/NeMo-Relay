// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical sample-batch and Shadow-attempt repository operations.

use std::collections::BTreeSet;
use std::sync::Arc;

use chrono::{TimeZone, Utc};
use nemo_relay::api::llm::LlmApiFamily;
use nemo_relay::api::runtime::LLM_REPLAY_CONTRACT_VERSION;
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use serde_json::{Value as Json, json};
use uuid::Uuid;

use super::judge::{
    verified_judge_evaluation_provenance, verify_terminal_judge_graph_for_retention,
};
use super::process::{append_integrity_health, originating_process_is_live};
use super::{LedgerRepository, TransactionStartGuard, map_fs_error, map_sqlite_error};
use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::config::JudgeConfig;
use crate::judge::{
    DeterministicHardFailureV1, JudgeBinaryLabelV1, JudgeEvaluationSourceV1, JudgeEvaluationV1,
    JudgeHardFailureV1, JudgeHorizonV1, JudgeLabelV1, JudgePolicyIdentityV1, PairwiseJudgeInputV1,
    ScoredValueV1,
};
use crate::ledger::fs::enforce_sidecar_permissions;
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::projection::{
    RouterRequestProjectionV1, RouterRoutingContextProjectionV1, validate_request_projection,
};
use crate::trajectory::{
    CANDIDATE_FACT_SCHEMA_V1, CAPTURED_EVENT_SCHEMA_V1, CapturedEventKind, CapturedTrajectoryEvent,
    PENDING_TRAJECTORY_SCHEMA_V1, PendingTrajectoryWindow, PersistedCandidateFactV1,
    PersistedTrajectoryTerminalV1, REPLAY_CAPABILITY_SCHEMA_V1, RESPONSE_PROJECTION_SCHEMA_V1,
    ReplayCapabilityFactsV1, RouterResponseProjectionV1, SanitizedResponseUsageV1,
    TERMINAL_TRAJECTORY_SCHEMA_V1, TRAJECTORY_SANITIZER_VERSION, TrajectoryDiagnosticV1,
    TrajectoryOwnerScopeV1, TrajectoryTerminalStateV1, TrajectoryTrigger,
};

/// One immutable candidate reservation created with a sample batch.
#[derive(Clone, PartialEq)]
pub(crate) struct ReservedShadowAttempt {
    pub(crate) shadow_attempt_id: Uuid,
    pub(crate) reserved_state_event_id: Uuid,
    pub(crate) candidate_id: String,
    pub(crate) candidate_model: String,
    pub(crate) candidate_model_revision: String,
    pub(crate) cost_rank: u32,
    pub(crate) api_family: LlmApiFamily,
    pub(crate) transport_identity: String,
    pub(crate) anchor_model: String,
    pub(crate) anchor_model_revision: String,
    pub(crate) decoding_fingerprint: String,
    pub(crate) evaluator_version: String,
    pub(crate) tenant_policy_hash: String,
    pub(crate) agent_policy_hash: String,
    pub(crate) eligible: bool,
    pub(crate) request_projection: RouterRequestProjectionV1,
    pub(crate) created_at_unix_ms: i64,
}

impl ReservedShadowAttempt {
    /// Freeze one candidate attempt before the batch command is enqueued.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        shadow_attempt_id: Uuid,
        reserved_state_event_id: Uuid,
        candidate_id: impl Into<String>,
        candidate_model: impl Into<String>,
        candidate_model_revision: impl Into<String>,
        cost_rank: u32,
        api_family: LlmApiFamily,
        transport_identity: impl Into<String>,
        anchor_model: impl Into<String>,
        anchor_model_revision: impl Into<String>,
        decoding_fingerprint: impl Into<String>,
        evaluator_version: impl Into<String>,
        tenant_policy_hash: impl Into<String>,
        agent_policy_hash: impl Into<String>,
        eligible: bool,
        request_projection: RouterRequestProjectionV1,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(shadow_attempt_id)?;
        validate_uuid_v7(reserved_state_event_id)?;
        if shadow_attempt_id == reserved_state_event_id || created_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let candidate_id = candidate_id.into();
        let candidate_model = candidate_model.into();
        let candidate_model_revision = candidate_model_revision.into();
        let transport_identity = transport_identity.into();
        let anchor_model = anchor_model.into();
        let anchor_model_revision = anchor_model_revision.into();
        let decoding_fingerprint = decoding_fingerprint.into();
        let evaluator_version = evaluator_version.into();
        let tenant_policy_hash = tenant_policy_hash.into();
        let agent_policy_hash = agent_policy_hash.into();
        validate_bounded_text(&candidate_id, 128)?;
        validate_bounded_text(&candidate_model, 512)?;
        validate_bounded_text(&candidate_model_revision, 128)?;
        validate_bounded_text(&transport_identity, 256)?;
        validate_bounded_text(&anchor_model, 512)?;
        validate_bounded_text(&anchor_model_revision, 128)?;
        for hash in [
            &decoding_fingerprint,
            &evaluator_version,
            &tenant_policy_hash,
            &agent_policy_hash,
        ] {
            validate_sha256(hash)?;
        }
        if request_projection.family != api_family
            || !request_projection_is_canonical(&request_projection)?
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            shadow_attempt_id,
            reserved_state_event_id,
            candidate_id,
            candidate_model,
            candidate_model_revision,
            cost_rank,
            api_family,
            transport_identity,
            anchor_model,
            anchor_model_revision,
            decoding_fingerprint,
            evaluator_version,
            tenant_policy_hash,
            agent_policy_hash,
            eligible,
            request_projection,
            created_at_unix_ms,
        })
    }
}

/// Atomic reservation of one batch and every candidate attempt.
#[derive(Clone, PartialEq)]
pub(crate) struct SampleBatchReservation {
    pub(crate) sample_batch_id: Uuid,
    pub(crate) open_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) anchor_id: Uuid,
    pub(crate) config_generation_id: String,
    pub(crate) policy_version_id: String,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) pool_id: String,
    pub(crate) attempts: Vec<ReservedShadowAttempt>,
    pub(crate) created_at_unix_ms: i64,
}

impl SampleBatchReservation {
    /// Freeze one nonempty, duplicate-free candidate batch.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        sample_batch_id: Uuid,
        open_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        anchor_id: Uuid,
        config_generation_id: impl Into<String>,
        policy_version_id: impl Into<String>,
        learning_generation_id: Uuid,
        pool_id: impl Into<String>,
        attempts: Vec<ReservedShadowAttempt>,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        for id in [
            sample_batch_id,
            open_state_event_id,
            conflict_health_event_id,
            anchor_id,
            learning_generation_id,
        ] {
            validate_uuid_v7(id)?;
        }
        if created_at_unix_ms < 0
            || attempts.is_empty()
            || sample_batch_id == open_state_event_id
            || open_state_event_id == conflict_health_event_id
            || sample_batch_id == conflict_health_event_id
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let config_generation_id = config_generation_id.into();
        let policy_version_id = policy_version_id.into();
        let pool_id = pool_id.into();
        validate_sha256(&config_generation_id)?;
        validate_sha256(&policy_version_id)?;
        validate_bounded_text(&pool_id, 128)?;
        let mut attempt_ids = BTreeSet::new();
        let mut reserved_state_event_ids = BTreeSet::new();
        let mut candidate_keys = BTreeSet::new();
        for attempt in &attempts {
            if !attempt_ids.insert(attempt.shadow_attempt_id)
                || !reserved_state_event_ids.insert(attempt.reserved_state_event_id)
                || !candidate_keys.insert((
                    attempt.candidate_id.as_str(),
                    attempt.candidate_model_revision.as_str(),
                ))
            {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        Ok(Self {
            sample_batch_id,
            open_state_event_id,
            conflict_health_event_id,
            anchor_id,
            config_generation_id,
            policy_version_id,
            learning_generation_id,
            pool_id,
            attempts,
            created_at_unix_ms,
        })
    }
}

/// Immutable transition proving that a reserved attempt started.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ShadowAttemptStarted {
    pub(crate) shadow_attempt_id: Uuid,
    pub(crate) state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) created_at_unix_ms: i64,
}

impl ShadowAttemptStarted {
    pub(crate) fn new(
        shadow_attempt_id: Uuid,
        state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        for id in [shadow_attempt_id, state_event_id, conflict_health_event_id] {
            validate_uuid_v7(id)?;
        }
        if state_event_id == conflict_health_event_id || created_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            shadow_attempt_id,
            state_event_id,
            conflict_health_event_id,
            created_at_unix_ms,
        })
    }
}

/// Stable operational failure stored without provider error text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ShadowOperationalFailureClass(String);

impl ShadowOperationalFailureClass {
    pub(crate) fn new(value: impl Into<String>) -> Result<Self, LedgerError> {
        let value = value.into();
        validate_stable_class(&value)?;
        Ok(Self(value))
    }

    fn as_str(&self) -> &str {
        &self.0
    }
}

/// Explicit reason an attempt cannot produce a canonical vector source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct NoncanonicalizableReason(String);

impl NoncanonicalizableReason {
    pub(crate) fn new(value: impl Into<String>) -> Result<Self, LedgerError> {
        let value = value.into();
        if value.is_empty()
            || value.len() > 128
            || !value
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_')
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self(value))
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

/// Typed source inputs retained for Spec 06 vector materialization.
#[derive(Clone, PartialEq)]
pub(crate) enum ShadowVectorSourceV1 {
    Canonicalizable {
        query_inputs: Box<RouterRequestProjectionV1>,
    },
    Noncanonicalizable {
        reason: NoncanonicalizableReason,
    },
}

/// Exact terminal class shared by the attempt state and result row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShadowTerminalClass {
    Completed,
    DeterministicFailure,
    OperationalFailure,
    SkippedCooloff,
    CanceledShutdown,
    OrphanedBeforeSchedule,
    OrphanedInFlight,
}

impl ShadowTerminalClass {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::DeterministicFailure => "deterministic_failure",
            Self::OperationalFailure => "operational_failure",
            Self::SkippedCooloff => "skipped_cooloff",
            Self::CanceledShutdown => "canceled_shutdown",
            Self::OrphanedBeforeSchedule => "orphaned_before_schedule",
            Self::OrphanedInFlight => "orphaned_in_flight",
        }
    }

    const fn is_orphan(self) -> bool {
        matches!(self, Self::OrphanedBeforeSchedule | Self::OrphanedInFlight)
    }

    const fn requires_started_state(self) -> bool {
        matches!(
            self,
            Self::Completed | Self::DeterministicFailure | Self::OperationalFailure
        )
    }

    const fn batch_terminal_state(self) -> SampleBatchTerminalState {
        match self {
            Self::Completed
            | Self::DeterministicFailure
            | Self::OperationalFailure
            | Self::SkippedCooloff => SampleBatchTerminalState::Closed,
            Self::CanceledShutdown => SampleBatchTerminalState::CanceledShutdown,
            Self::OrphanedBeforeSchedule => SampleBatchTerminalState::OrphanedBeforeSchedule,
            Self::OrphanedInFlight => SampleBatchTerminalState::OrphanedInFlight,
        }
    }
}

/// Normal terminal vectorization authority frozen before writer admission.
#[derive(Clone, PartialEq)]
pub(crate) struct AtomicShadowVectorization {
    pub(crate) routing_projection: Box<RouterRoutingContextProjectionV1>,
    pub(crate) evidence_vector_link_id: Uuid,
    pub(crate) evidence_link_state_event_id: Uuid,
    pub(crate) materialization_state_event_id: Uuid,
    pub(crate) embedding_job_state_event_id: Uuid,
}

impl AtomicShadowVectorization {
    pub(crate) fn new(
        routing_projection: RouterRoutingContextProjectionV1,
        evidence_vector_link_id: Uuid,
        evidence_link_state_event_id: Uuid,
        materialization_state_event_id: Uuid,
        embedding_job_state_event_id: Uuid,
    ) -> Result<Self, LedgerError> {
        let ids = [
            evidence_vector_link_id,
            evidence_link_state_event_id,
            materialization_state_event_id,
            embedding_job_state_event_id,
        ];
        for id in ids {
            validate_uuid_v7(id)?;
        }
        if ids.into_iter().collect::<BTreeSet<_>>().len() != ids.len() {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            routing_projection: Box::new(routing_projection),
            evidence_vector_link_id,
            evidence_link_state_event_id,
            materialization_state_event_id,
            embedding_job_state_event_id,
        })
    }
}

/// Whether one terminal writes vectors now, is disabled, or is backfilled later.
#[derive(Clone, PartialEq)]
pub(crate) enum ShadowVectorizationHandoff {
    Disabled,
    Atomic(AtomicShadowVectorization),
    DeferredBackfill,
}

/// Optional explicit batch terminal transition coupled to one result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SampleBatchTerminalEvent {
    pub(crate) sample_batch_id: Uuid,
    pub(crate) state_event_id: Uuid,
    pub(crate) state: SampleBatchTerminalState,
    pub(crate) dead_process_instance_id: Option<Uuid>,
    pub(crate) created_at_unix_ms: i64,
}

/// Exact terminal state accepted by `sample_batch_state_events`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SampleBatchTerminalState {
    Closed,
    OrphanedBeforeSchedule,
    OrphanedInFlight,
    CanceledShutdown,
}

impl SampleBatchTerminalState {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Closed => "closed",
            Self::OrphanedBeforeSchedule => "orphaned_before_schedule",
            Self::OrphanedInFlight => "orphaned_in_flight",
            Self::CanceledShutdown => "canceled_shutdown",
        }
    }

    const fn is_orphan(self) -> bool {
        matches!(self, Self::OrphanedBeforeSchedule | Self::OrphanedInFlight)
    }
}

impl SampleBatchTerminalEvent {
    pub(crate) fn new(
        sample_batch_id: Uuid,
        state_event_id: Uuid,
        state: SampleBatchTerminalState,
        dead_process_instance_id: Option<Uuid>,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(sample_batch_id)?;
        validate_uuid_v7(state_event_id)?;
        if let Some(dead_process_instance_id) = dead_process_instance_id {
            validate_uuid_v7(dead_process_instance_id)?;
        }
        if created_at_unix_ms < 0 || state.is_orphan() != dead_process_instance_id.is_some() {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            sample_batch_id,
            state_event_id,
            state,
            dead_process_instance_id,
            created_at_unix_ms,
        })
    }
}

/// Atomic Shadow terminal result, vector source, state, and optional batch close.
#[derive(Clone, PartialEq)]
pub(crate) struct ShadowTerminalRecord {
    pub(crate) shadow_result_id: Uuid,
    pub(crate) shadow_attempt_id: Uuid,
    pub(crate) state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) terminal_class: ShadowTerminalClass,
    pub(crate) dead_process_instance_id: Option<Uuid>,
    pub(crate) normalized_response: Option<RouterResponseProjectionV1>,
    pub(crate) deterministic_hard_failure: Option<DeterministicHardFailureV1>,
    pub(crate) operational_failure_class: Option<ShadowOperationalFailureClass>,
    pub(crate) latency_ms: Option<i64>,
    pub(crate) usage: Option<SanitizedResponseUsageV1>,
    pub(crate) evaluation_id: Option<Uuid>,
    pub(crate) vector_source: ShadowVectorSourceV1,
    pub(crate) batch_terminal: Option<SampleBatchTerminalEvent>,
    pub(crate) vectorization: ShadowVectorizationHandoff,
    pub(crate) created_at_unix_ms: i64,
}

impl ShadowTerminalRecord {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        shadow_result_id: Uuid,
        shadow_attempt_id: Uuid,
        state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        terminal_class: ShadowTerminalClass,
        dead_process_instance_id: Option<Uuid>,
        normalized_response: Option<RouterResponseProjectionV1>,
        deterministic_hard_failure: Option<DeterministicHardFailureV1>,
        operational_failure_class: Option<ShadowOperationalFailureClass>,
        latency_ms: Option<u64>,
        usage: Option<SanitizedResponseUsageV1>,
        evaluation_id: Option<Uuid>,
        vector_source: ShadowVectorSourceV1,
        batch_terminal: Option<SampleBatchTerminalEvent>,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        for id in [
            shadow_result_id,
            shadow_attempt_id,
            state_event_id,
            conflict_health_event_id,
        ] {
            validate_uuid_v7(id)?;
        }
        if let Some(dead_process_instance_id) = dead_process_instance_id {
            validate_uuid_v7(dead_process_instance_id)?;
        }
        if let Some(evaluation_id) = evaluation_id {
            validate_uuid_v7(evaluation_id)?;
        }
        if state_event_id == conflict_health_event_id
            || created_at_unix_ms < 0
            || terminal_class.is_orphan() != dead_process_instance_id.is_some()
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let latency_ms = latency_ms
            .map(i64::try_from)
            .transpose()
            .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let valid_shape = match terminal_class {
            ShadowTerminalClass::Completed => {
                normalized_response.is_some()
                    && deterministic_hard_failure.is_none()
                    && operational_failure_class.is_none()
                    && evaluation_id.is_some()
            }
            ShadowTerminalClass::DeterministicFailure => {
                deterministic_hard_failure.is_some()
                    && operational_failure_class.is_none()
                    && evaluation_id.is_some()
            }
            ShadowTerminalClass::OperationalFailure => {
                deterministic_hard_failure.is_none()
                    && operational_failure_class.is_some()
                    && evaluation_id.is_none()
            }
            ShadowTerminalClass::SkippedCooloff
            | ShadowTerminalClass::CanceledShutdown
            | ShadowTerminalClass::OrphanedBeforeSchedule
            | ShadowTerminalClass::OrphanedInFlight => {
                deterministic_hard_failure.is_none()
                    && operational_failure_class.is_none()
                    && evaluation_id.is_none()
            }
        };
        if !valid_shape {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        if let Some(response) = normalized_response.as_ref()
            && !response_projection_is_canonical(response)?
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        if let ShadowVectorSourceV1::Canonicalizable { query_inputs } = &vector_source {
            validate_request_projection(query_inputs)
                .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        }
        Ok(Self {
            shadow_result_id,
            shadow_attempt_id,
            state_event_id,
            conflict_health_event_id,
            terminal_class,
            dead_process_instance_id,
            normalized_response,
            deterministic_hard_failure,
            operational_failure_class,
            latency_ms,
            usage,
            evaluation_id,
            vector_source,
            batch_terminal,
            vectorization: ShadowVectorizationHandoff::Disabled,
            created_at_unix_ms,
        })
    }

    pub(crate) fn with_vectorization(mut self, vectorization: ShadowVectorizationHandoff) -> Self {
        self.vectorization = vectorization;
        self
    }
}

/// Exhaustive stable acknowledgement for batch and Shadow commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ShadowCommandAck {
    Applied,
    AlreadyApplied,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

enum DomainWrite {
    Applied,
    AlreadyApplied,
    Conflict { anchor_id: Option<Uuid> },
}

/// Counts produced by one dead process's Shadow recovery.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(super) struct ShadowReconciliationReport {
    pub(super) reconstructed_batches: usize,
    pub(super) orphaned_attempts: usize,
    pub(super) closed_batches: usize,
}

#[derive(Debug)]
struct StoredBatch {
    sample_batch_id: String,
    anchor_id: String,
    project_uuid: String,
    process_instance_id: String,
    config_generation_id: String,
    policy_version_id: String,
    learning_generation_id: String,
    pool_id: String,
    reserved_candidate_count: i64,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug)]
struct StoredAttempt {
    shadow_attempt_id: String,
    sample_batch_id: String,
    anchor_id: String,
    project_uuid: String,
    process_instance_id: String,
    config_generation_id: String,
    policy_version_id: String,
    learning_generation_id: String,
    pool_id: String,
    candidate_id: String,
    candidate_model: String,
    candidate_model_revision: String,
    cost_rank: i64,
    api_family: String,
    transport_identity: String,
    anchor_model: String,
    anchor_model_revision: String,
    decoding_fingerprint: String,
    evaluator_version: String,
    tenant_policy_hash: String,
    agent_policy_hash: String,
    eligible: i64,
    request_projection_json: String,
    partition_inputs_json: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

/// Canonically verified Shadow identity consumed by dependent repository commands.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct VerifiedShadowAttemptContext {
    pub(super) anchor_id: Uuid,
    pub(super) process_instance_id: Uuid,
    pub(super) learning_generation_id: Uuid,
    pub(super) evaluator_version: String,
    pub(super) is_partial: bool,
    pub(super) started_at_unix_ms: i64,
}

/// Canonically reconstructed immutable inputs for one current-space backfill.
#[derive(Clone, PartialEq)]
pub(super) struct VerifiedBackfillVectorSource {
    pub(super) reservation: SampleBatchReservation,
    pub(super) attempt: ReservedShadowAttempt,
    pub(super) shadow_result_id: Uuid,
    pub(super) root_uuid: Uuid,
    pub(super) routing_projection: RouterRoutingContextProjectionV1,
    pub(super) terminal_class: ShadowTerminalClass,
    pub(super) evaluation_id: Option<Uuid>,
    pub(super) evaluation: Option<VerifiedVectorEvaluation>,
    pub(super) vector_source: ShadowVectorSourceV1,
    pub(super) terminal_at_unix_ms: i64,
}

/// Search-facing evaluation facts emitted only after the complete evaluation verifies.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VerifiedVectorEvaluation {
    pub(crate) evaluation_id: Uuid,
    pub(crate) source: JudgeEvaluationSourceV1,
    pub(crate) binary_label: Option<JudgeBinaryLabelV1>,
    pub(crate) judge_confidence: Option<ScoredValueV1>,
    pub(crate) promotion_eligible: bool,
    pub(crate) created_at_unix_ms: i64,
}

/// Canonically verified reserved Shadow identity used to bind dependency keys.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct VerifiedShadowDependencyContext {
    pub(super) anchor_id: Uuid,
    pub(super) project_uuid: Uuid,
    pub(super) process_instance_id: Uuid,
    pub(super) policy_version_id: String,
    pub(super) pool_id: String,
    pub(super) candidate_model: String,
    pub(super) candidate_model_revision: String,
    pub(super) api_family: String,
    pub(super) transport_identity: String,
    pub(super) evaluator_version: String,
}

/// Canonical immutable Judge policy bound to one reserved Shadow attempt.
#[derive(Clone, PartialEq)]
pub(super) struct VerifiedShadowJudgePolicy {
    pub(super) config: JudgeConfig,
    pub(super) evaluator_version: String,
    pub(super) prompt_sha256: String,
    pub(super) rubric_sha256: String,
    pub(super) output_schema_sha256: String,
}

/// Canonical Judge input binding reconstructed from durable Shadow evidence.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct VerifiedShadowJudgeInputContext {
    pub(super) judge_input_sha256: String,
    pub(super) candidate_response_json: String,
    pub(super) candidate_response_fingerprint: String,
}

#[derive(Debug)]
struct StoredState {
    event_id: String,
    process_instance_id: String,
    dead_process_instance_id: Option<String>,
    state: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug)]
struct StoredResult {
    shadow_result_id: String,
    shadow_attempt_id: String,
    terminal_class: String,
    normalized_response_json: Option<String>,
    response_fingerprint: Option<String>,
    deterministic_hard_failure: Option<String>,
    operational_failure_class: Option<String>,
    latency_ms: Option<i64>,
    usage_json: Option<String>,
    evaluation_id: Option<String>,
    canonicalizable: i64,
    query_inputs_json: Option<String>,
    partition_inputs_json: String,
    vector_source_hash: Option<String>,
    noncanonicalizable_reason: Option<String>,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

struct StoredEvaluation {
    evaluation_id: String,
    shadow_attempt_id: String,
    evaluator_version: String,
    source: String,
    judge_model: Option<String>,
    judge_model_revision: Option<String>,
    prompt_version: Option<String>,
    prompt_sha256: Option<String>,
    rubric_version: Option<String>,
    rubric_sha256: Option<String>,
    output_schema_version: Option<i64>,
    output_schema_sha256: Option<String>,
    response_equivalence: Option<f64>,
    response_equivalence_bits: Option<i64>,
    trajectory_equivalence: Option<f64>,
    trajectory_equivalence_bits: Option<i64>,
    judge_confidence: Option<f64>,
    judge_confidence_bits: Option<i64>,
    response_weight: Option<f64>,
    response_weight_bits: Option<i64>,
    trajectory_weight: Option<f64>,
    trajectory_weight_bits: Option<i64>,
    aggregate_score: Option<f64>,
    aggregate_score_bits: Option<i64>,
    label: String,
    binary_label: Option<String>,
    rationale: Option<String>,
    is_partial: i64,
    promotion_eligible: i64,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

struct StoredEvaluationFailure {
    value: String,
    ordinal: i64,
    canonical_payload_hash: String,
}

struct StoredAnchor {
    anchor_id: String,
    project_uuid: String,
    process_instance_id: String,
    config_generation_id: String,
    policy_version_id: String,
    learning_generation_id: String,
    pool_id: String,
    anchor_call_uuid: String,
    root_uuid: String,
    owner_uuid: String,
    owner_path_json: String,
    api_family: String,
    transport_identity: String,
    anchor_model: String,
    anchor_model_revision: String,
    replay_capability_fingerprint: String,
    decoding_fingerprint: String,
    request_projection_json: String,
    routing_context_projection_json: String,
    candidate_facts_json: String,
    requested_progress: i64,
    opened_at_unix_ms: i64,
    deadline_at_unix_ms: i64,
    non_resumable: i64,
    pending_hash: String,
    canonical_payload_hash: String,
    project_id: String,
}

struct StoredAnchorResult {
    normalized_response_json: String,
    semantic_response_fingerprint: String,
    canonical_payload_hash: String,
}

struct StoredAnchorState {
    event_id: String,
    anchor_id: String,
    process_instance_id: String,
    dead_process_instance_id: Option<String>,
    state: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

struct StoredAnchorWindow {
    anchor_id: String,
    requested_progress: i64,
    observed_progress: i64,
    terminal_kind: String,
    trigger: Option<String>,
    rejection_reason: Option<String>,
    is_partial: i64,
    promotion_eligible: i64,
    closed_at_unix_ms: i64,
    diagnostics_json: String,
    terminal_hash: String,
    canonical_payload_hash: String,
}

struct StoredTrajectoryEvent {
    anchor_id: String,
    ingest_seq: i64,
    event_uuid: String,
    parent_uuid: Option<String>,
    kind: String,
    phase: Option<String>,
    category: Option<String>,
    call_role: Option<String>,
    name: String,
    event_time_unix_ms: i64,
    schema_id: String,
    sanitized_payload_json: String,
    canonical_size_bytes: i64,
    canonical_payload_hash: String,
}

impl LedgerRepository {
    /// Atomically create one open batch and every reserved attempt.
    pub(crate) fn reserve_sample_batch(
        &mut self,
        reservation: &SampleBatchReservation,
    ) -> Result<ShadowCommandAck, LedgerError> {
        self.reserve_sample_batch_with_start_check(reservation, || Some(()))
    }

    /// Reserve a batch after retaining caller-owned transaction-start authority.
    pub(crate) fn reserve_sample_batch_with_start_check<G: TransactionStartGuard>(
        &mut self,
        reservation: &SampleBatchReservation,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ShadowCommandAck, LedgerError> {
        self.run_shadow_command(
            reservation.conflict_health_event_id,
            reservation.created_at_unix_ms,
            start_check,
            |connection, project_uuid, process_instance_id| {
                reserve_batch_in_savepoint(
                    connection,
                    project_uuid,
                    process_instance_id,
                    reservation,
                )
            },
        )
    }

    /// Append the unique started state for one reserved attempt.
    pub(crate) fn start_shadow_attempt(
        &mut self,
        command: ShadowAttemptStarted,
    ) -> Result<ShadowCommandAck, LedgerError> {
        self.start_shadow_attempt_with_start_check(command, || Some(()))
    }

    /// Append started state after retaining transaction-start authority.
    pub(crate) fn start_shadow_attempt_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: ShadowAttemptStarted,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ShadowCommandAck, LedgerError> {
        self.run_shadow_command(
            command.conflict_health_event_id,
            command.created_at_unix_ms,
            start_check,
            |connection, project_uuid, process_instance_id| {
                start_attempt_in_savepoint(connection, project_uuid, process_instance_id, command)
            },
        )
    }

    /// Atomically append a terminal result/state and optional batch terminal state.
    pub(crate) fn record_shadow_terminal(
        &mut self,
        command: &ShadowTerminalRecord,
    ) -> Result<ShadowCommandAck, LedgerError> {
        self.record_shadow_terminal_with_start_check(command, || Some(()))
    }

    /// Record a terminal result after retaining transaction-start authority.
    pub(crate) fn record_shadow_terminal_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &ShadowTerminalRecord,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ShadowCommandAck, LedgerError> {
        self.run_shadow_command(
            command.conflict_health_event_id,
            command.created_at_unix_ms,
            start_check,
            |savepoint, project_uuid, process_instance_id| {
                terminal_with_vectorization_in_savepoint(
                    savepoint,
                    project_uuid,
                    process_instance_id,
                    command,
                )
            },
        )
    }

    fn run_shadow_command<G: TransactionStartGuard>(
        &mut self,
        conflict_health_event_id: Uuid,
        created_at_unix_ms: i64,
        start_check: impl FnOnce() -> Option<G>,
        operation: impl FnOnce(&Transaction<'_>, Uuid, Uuid) -> Result<DomainWrite, LedgerError>,
    ) -> Result<ShadowCommandAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ShadowCommandAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(ShadowCommandAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ShadowCommandAck::TransactionNotStarted);
        }
        drop(start_guard);
        if !originating_process_is_live(&transaction, project_uuid, process_instance_id)? {
            return Ok(ShadowCommandAck::OriginatingProcessNotLive);
        }
        transaction
            .execute_batch("SAVEPOINT shadow_domain")
            .map_err(database_error)?;
        let domain_write = operation(&transaction, project_uuid, process_instance_id)?;
        match domain_write {
            DomainWrite::Conflict { .. } => transaction
                .execute_batch("ROLLBACK TO shadow_domain; RELEASE shadow_domain")
                .map_err(database_error)?,
            DomainWrite::Applied | DomainWrite::AlreadyApplied => transaction
                .execute_batch("RELEASE shadow_domain")
                .map_err(database_error)?,
        }
        let acknowledgement = match domain_write {
            DomainWrite::Applied => ShadowCommandAck::Applied,
            DomainWrite::AlreadyApplied => ShadowCommandAck::AlreadyApplied,
            DomainWrite::Conflict { anchor_id } => {
                append_integrity_health(
                    &transaction,
                    conflict_health_event_id,
                    project_uuid,
                    process_instance_id,
                    anchor_id,
                    None,
                    created_at_unix_ms,
                )?;
                ShadowCommandAck::Conflict
            }
        };
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }
}

fn terminal_with_vectorization_in_savepoint(
    savepoint: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &ShadowTerminalRecord,
) -> Result<DomainWrite, LedgerError> {
    if matches!(
        command.vectorization,
        ShadowVectorizationHandoff::DeferredBackfill
    ) {
        return Ok(DomainWrite::Conflict { anchor_id: None });
    }
    let shadow_write =
        terminal_in_savepoint(savepoint, project_uuid, process_instance_id, command)?;
    let applied = match shadow_write {
        DomainWrite::Applied => true,
        DomainWrite::AlreadyApplied => false,
        conflict @ DomainWrite::Conflict { .. } => return Ok(conflict),
    };

    let attempt = load_attempt_by_id(savepoint, command.shadow_attempt_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let anchor_id = parse_uuid_v7(&attempt.anchor_id)?;
    let anchor = load_stored_anchor(savepoint, anchor_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let pending = canonical_pending_anchor(savepoint, &anchor)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if let ShadowVectorizationHandoff::Atomic(vectorization) = &command.vectorization
        && canonical_serialize(vectorization.routing_projection.as_ref())?
            != anchor.routing_context_projection_json
    {
        return Ok(DomainWrite::Conflict {
            anchor_id: Some(anchor_id),
        });
    }
    let batch_id = parse_uuid_v7(&attempt.sample_batch_id)?;
    let batch = load_batch_by_id(savepoint, batch_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let reservation =
        reservation_from_stored_batch(savepoint, batch_id, &batch, std::slice::from_ref(&attempt))?;
    let reserved = reservation
        .attempts
        .first()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let vector_write = super::materialization::record_terminal_vector_graph(
        savepoint,
        project_uuid,
        process_instance_id,
        &reservation,
        reserved,
        pending.root_uuid,
        &pending.routing_context_projection,
        command,
        applied,
    )?;
    if vector_write {
        Ok(shadow_write)
    } else {
        Ok(DomainWrite::Conflict {
            anchor_id: Some(anchor_id),
        })
    }
}

/// Verify that one pending anchor is complete enough for safe orphaning.
pub(super) fn verify_pending_anchor_for_reconciliation(
    connection: &Connection,
    project_uuid: Uuid,
    dead_process_instance_id: Uuid,
    anchor_id: Uuid,
) -> Result<(), LedgerError> {
    let anchor = load_stored_anchor(connection, anchor_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if anchor.project_uuid != project_uuid.to_string()
        || anchor.process_instance_id != dead_process_instance_id.to_string()
        || canonical_pending_anchor(connection, &anchor)?.is_none()
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let pending = load_anchor_state(connection, &anchor.anchor_id, "pending")?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if !canonical_anchor_state_matches(&pending, &anchor, "pending")?
        || load_anchor_window(connection, &anchor.anchor_id)?.is_some()
        || load_batch_by_anchor(connection, anchor_id)?.is_some()
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

/// Reconstruct closed-undelivered batches and orphan every incomplete existing attempt.
pub(super) fn reconcile_shadow_work_in_transaction(
    connection: &Connection,
    project_uuid: Uuid,
    reconciler_process_instance_id: Uuid,
    dead_process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<ShadowReconciliationReport, LedgerError> {
    if reconciler_process_instance_id == dead_process_instance_id || created_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let mut report = ShadowReconciliationReport::default();

    let mut statement = connection
        .prepare(
            "SELECT a.anchor_id
             FROM anchors AS a
             JOIN anchor_state_events AS s
               ON s.anchor_id = a.anchor_id AND s.state = 'closed'
             WHERE a.project_uuid = ?1 AND a.process_instance_id = ?2
               AND NOT EXISTS (
                    SELECT 1 FROM sample_batches AS b WHERE b.anchor_id = a.anchor_id
               )
             ORDER BY a.anchor_id",
        )
        .map_err(database_error)?;
    let undelivered = statement
        .query_map(
            params![
                project_uuid.to_string(),
                dead_process_instance_id.to_string()
            ],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    for anchor_id in undelivered {
        let anchor_id = parse_uuid_v7(&anchor_id)?;
        let orphaned = reconstruct_and_orphan_closed_anchor(
            connection,
            project_uuid,
            reconciler_process_instance_id,
            dead_process_instance_id,
            anchor_id,
            created_at_unix_ms,
        )?;
        report.reconstructed_batches = report
            .reconstructed_batches
            .checked_add(1)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        report.orphaned_attempts = report
            .orphaned_attempts
            .checked_add(orphaned)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        report.closed_batches = report
            .closed_batches
            .checked_add(1)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    }

    let mut statement = connection
        .prepare(
            "SELECT b.sample_batch_id
             FROM sample_batches AS b
             JOIN sample_batch_state_events AS s
               ON s.sample_batch_id = b.sample_batch_id AND s.state = 'open'
             WHERE b.project_uuid = ?1 AND b.process_instance_id = ?2
               AND NOT EXISTS (
                    SELECT 1 FROM sample_batch_state_events AS terminal
                    WHERE terminal.sample_batch_id = b.sample_batch_id
                      AND terminal.state <> 'open'
               )
             ORDER BY b.sample_batch_id",
        )
        .map_err(database_error)?;
    let open_batches = statement
        .query_map(
            params![
                project_uuid.to_string(),
                dead_process_instance_id.to_string()
            ],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    for batch_id in open_batches {
        let batch_id = parse_uuid_v7(&batch_id)?;
        let orphaned = orphan_existing_batch(
            connection,
            project_uuid,
            reconciler_process_instance_id,
            dead_process_instance_id,
            batch_id,
            created_at_unix_ms,
        )?;
        report.orphaned_attempts = report
            .orphaned_attempts
            .checked_add(orphaned)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        report.closed_batches = report
            .closed_batches
            .checked_add(1)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    }
    validate_reconciled_shadow_work(connection, project_uuid, dead_process_instance_id)?;
    validate_reconciled_anchors(connection, project_uuid, dead_process_instance_id)?;
    Ok(report)
}

fn validate_reconciled_shadow_work(
    connection: &Connection,
    project_uuid: Uuid,
    dead_process_instance_id: Uuid,
) -> Result<(), LedgerError> {
    validate_dead_process_evaluations(connection, project_uuid, dead_process_instance_id)?;
    let mut statement = connection
        .prepare(
            "SELECT sample_batch_id FROM sample_batches
             WHERE project_uuid = ?1 AND process_instance_id = ?2
             ORDER BY sample_batch_id",
        )
        .map_err(database_error)?;
    let batch_ids = statement
        .query_map(
            params![
                project_uuid.to_string(),
                dead_process_instance_id.to_string()
            ],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    for batch_id in batch_ids {
        let batch_id = parse_uuid_v7(&batch_id)?;
        let attempts = load_attempts_by_batch(connection, batch_id)?;
        if attempts.is_empty() {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        for attempt in &attempts {
            let attempt_id = parse_uuid_v7(&attempt.shadow_attempt_id)?;
            if !stored_attempt_parent_is_canonical(connection, attempt)?
                || !canonical_terminal_attempt_matches(connection, attempt_id)?
            {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        let terminal = load_batch_terminal_state(connection, batch_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if !canonical_batch_terminal_state(connection, batch_id, &terminal)? {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
    }
    Ok(())
}

fn validate_dead_process_evaluations(
    connection: &Connection,
    project_uuid: Uuid,
    dead_process_instance_id: Uuid,
) -> Result<(), LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT e.evaluation_id, e.shadow_attempt_id
             FROM evaluations AS e
             JOIN shadow_attempts AS a ON a.shadow_attempt_id = e.shadow_attempt_id
             WHERE a.project_uuid = ?1 AND a.process_instance_id = ?2
             ORDER BY e.evaluation_id",
        )
        .map_err(database_error)?;
    let evaluations = statement
        .query_map(
            params![
                project_uuid.to_string(),
                dead_process_instance_id.to_string()
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    for (evaluation_id, shadow_attempt_id) in evaluations {
        let evaluation_id = parse_uuid_v7(&evaluation_id)?;
        let shadow_attempt_id = parse_uuid_v7(&shadow_attempt_id)?;
        if load_verified_evaluation(connection, evaluation_id, shadow_attempt_id)?.is_none() {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
    }
    Ok(())
}

/// Verify the complete canonical evidence graph for one retention candidate.
pub(super) fn verify_terminal_anchor_for_retention(
    connection: &Connection,
    project_uuid: Uuid,
    anchor_id: Uuid,
) -> Result<(), LedgerError> {
    let anchor = load_stored_anchor(connection, anchor_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if anchor.project_uuid != project_uuid.to_string()
        || !historical_anchor_identity_is_canonical(connection, &anchor)?
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let pending = canonical_pending_anchor(connection, &anchor)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let pending_state = load_anchor_state(connection, &anchor.anchor_id, "pending")?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if !canonical_anchor_state_matches(&pending_state, &anchor, "pending")?
        || pending_state.created_at_unix_ms < anchor.opened_at_unix_ms
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let mut statement = connection
        .prepare(
            "SELECT state FROM anchor_state_events
             WHERE anchor_id = ?1 AND state <> 'pending'
             ORDER BY event_seq",
        )
        .map_err(database_error)?;
    let terminal_states = statement
        .query_map(params![anchor_id.to_string()], |row| {
            row.get::<_, String>(0)
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    let [terminal_state_name] = terminal_states.as_slice() else {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    };
    let state_count: i64 = connection
        .query_row(
            "SELECT count(*) FROM anchor_state_events WHERE anchor_id = ?1",
            params![anchor_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    let terminal_state = load_anchor_state(connection, &anchor.anchor_id, terminal_state_name)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if state_count != 2 || terminal_state.created_at_unix_ms < pending_state.created_at_unix_ms {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let batch_count: i64 = connection
        .query_row(
            "SELECT count(*) FROM sample_batches WHERE anchor_id = ?1",
            params![anchor_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    match terminal_state_name.as_str() {
        "closed" => {
            if batch_count != 1
                || !canonical_terminal_anchor_states_and_window(
                    connection, &anchor, &pending, "closed",
                )?
            {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
            let batch = load_batch_by_anchor(connection, anchor_id)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
            let batch_id = parse_uuid_v7(&batch.sample_batch_id)?;
            let attempts = load_attempts_by_batch(connection, batch_id)?;
            if attempts.is_empty()
                || i64::try_from(attempts.len()).ok() != Some(batch.reserved_candidate_count)
            {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
            for attempt in &attempts {
                let attempt_id = parse_uuid_v7(&attempt.shadow_attempt_id)?;
                if !stored_attempt_parent_is_canonical(connection, attempt)?
                    || !canonical_terminal_attempt_matches(connection, attempt_id)?
                {
                    return Err(LedgerErrorClass::IdentityInvariant.into());
                }
                verify_terminal_judge_graph_for_retention(connection, project_uuid, attempt_id)?;
            }
            let batch_terminal = load_batch_terminal_state(connection, batch_id)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
            let batch_state_count: i64 = connection
                .query_row(
                    "SELECT count(*) FROM sample_batch_state_events
                     WHERE sample_batch_id = ?1",
                    params![batch_id.to_string()],
                    |row| row.get(0),
                )
                .map_err(database_error)?;
            if batch_state_count != 2
                || !canonical_batch_terminal_state(connection, batch_id, &batch_terminal)?
            {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        "rejected" => {
            if batch_count != 0
                || !canonical_terminal_anchor_states_and_window(
                    connection, &anchor, &pending, "rejected",
                )?
            {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        "not_scheduled_queue_full" => {
            if batch_count != 0
                || !canonical_anchor_state_matches(
                    &terminal_state,
                    &anchor,
                    "not_scheduled_queue_full",
                )?
                || load_anchor_window(connection, &anchor.anchor_id)?.is_some()
                || anchor_event_count(connection, anchor_id)? != 0
            {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        "orphaned_non_resumable" => {
            let actor = parse_uuid_v7(&terminal_state.process_instance_id)?;
            let dead_process_instance_id = parse_uuid_v7(&anchor.process_instance_id)?;
            let expected_hash = hash_json(&json!({
                "anchor_state_event_id": parse_uuid_v7(&terminal_state.event_id)?,
                "anchor_id": anchor_id,
                "process_instance_id": actor,
                "dead_process_instance_id": dead_process_instance_id,
                "state": "orphaned_non_resumable",
                "created_at_unix_ms": terminal_state.created_at_unix_ms,
            }))?;
            if batch_count != 0
                || actor == dead_process_instance_id
                || terminal_state.anchor_id != anchor.anchor_id
                || terminal_state.dead_process_instance_id
                    != Some(dead_process_instance_id.to_string())
                || terminal_state.canonical_payload_hash != expected_hash
                || !process_belongs_to_project(connection, actor, &anchor.project_uuid)?
                || load_anchor_window(connection, &anchor.anchor_id)?.is_some()
                || anchor_event_count(connection, anchor_id)? != 0
            {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        _ => return Err(LedgerErrorClass::IdentityInvariant.into()),
    }

    let mut statement = connection
        .prepare(
            "SELECT e.evaluation_id, e.shadow_attempt_id
             FROM evaluations AS e
             JOIN shadow_attempts AS a ON a.shadow_attempt_id = e.shadow_attempt_id
             WHERE a.anchor_id = ?1
             ORDER BY e.evaluation_id",
        )
        .map_err(database_error)?;
    let evaluations = statement
        .query_map(params![anchor_id.to_string()], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    for (evaluation_id, shadow_attempt_id) in evaluations {
        if load_verified_evaluation(
            connection,
            parse_uuid_v7(&evaluation_id)?,
            parse_uuid_v7(&shadow_attempt_id)?,
        )?
        .is_none()
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
    }
    Ok(())
}

fn validate_reconciled_anchors(
    connection: &Connection,
    project_uuid: Uuid,
    dead_process_instance_id: Uuid,
) -> Result<(), LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT anchor_id FROM anchors
             WHERE project_uuid = ?1 AND process_instance_id = ?2
             ORDER BY anchor_id",
        )
        .map_err(database_error)?;
    let anchor_ids = statement
        .query_map(
            params![
                project_uuid.to_string(),
                dead_process_instance_id.to_string()
            ],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);

    for anchor_id in anchor_ids {
        let anchor_id = parse_uuid_v7(&anchor_id)?;
        let anchor = load_stored_anchor(connection, anchor_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if !historical_anchor_identity_is_canonical(connection, &anchor)? {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let pending = canonical_pending_anchor(connection, &anchor)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let pending_state = load_anchor_state(connection, &anchor.anchor_id, "pending")?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if !canonical_anchor_state_matches(&pending_state, &anchor, "pending")? {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let terminal_state_name = connection
            .query_row(
                "SELECT state FROM anchor_state_events
                 WHERE anchor_id = ?1 AND state <> 'pending'",
                params![anchor_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(database_error)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let state_count: i64 = connection
            .query_row(
                "SELECT count(*) FROM anchor_state_events WHERE anchor_id = ?1",
                params![anchor_id.to_string()],
                |row| row.get(0),
            )
            .map_err(database_error)?;
        if state_count != 2 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let terminal_state =
            load_anchor_state(connection, &anchor.anchor_id, &terminal_state_name)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if terminal_state.created_at_unix_ms < pending_state.created_at_unix_ms {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let batch_count: i64 = connection
            .query_row(
                "SELECT count(*) FROM sample_batches WHERE anchor_id = ?1",
                params![anchor_id.to_string()],
                |row| row.get(0),
            )
            .map_err(database_error)?;
        match terminal_state_name.as_str() {
            "closed" => {
                if batch_count != 1
                    || !canonical_terminal_anchor_states_and_window(
                        connection, &anchor, &pending, "closed",
                    )?
                {
                    return Err(LedgerErrorClass::IdentityInvariant.into());
                }
            }
            "rejected" => {
                if batch_count != 0
                    || !canonical_terminal_anchor_states_and_window(
                        connection, &anchor, &pending, "rejected",
                    )?
                {
                    return Err(LedgerErrorClass::IdentityInvariant.into());
                }
            }
            "not_scheduled_queue_full" => {
                if batch_count != 0
                    || !canonical_anchor_state_matches(
                        &terminal_state,
                        &anchor,
                        "not_scheduled_queue_full",
                    )?
                    || load_anchor_window(connection, &anchor.anchor_id)?.is_some()
                    || anchor_event_count(connection, anchor_id)? != 0
                {
                    return Err(LedgerErrorClass::IdentityInvariant.into());
                }
            }
            "orphaned_non_resumable" => {
                let actor = parse_uuid_v7(&terminal_state.process_instance_id)?;
                let expected_hash = hash_json(&json!({
                    "anchor_state_event_id": parse_uuid_v7(&terminal_state.event_id)?,
                    "anchor_id": anchor_id,
                    "process_instance_id": actor,
                    "dead_process_instance_id": dead_process_instance_id,
                    "state": "orphaned_non_resumable",
                    "created_at_unix_ms": terminal_state.created_at_unix_ms,
                }))?;
                if batch_count != 0
                    || actor == dead_process_instance_id
                    || terminal_state.anchor_id != anchor.anchor_id
                    || terminal_state.dead_process_instance_id
                        != Some(dead_process_instance_id.to_string())
                    || terminal_state.canonical_payload_hash != expected_hash
                    || !process_belongs_to_project(connection, actor, &project_uuid.to_string())?
                    || load_anchor_window(connection, &anchor.anchor_id)?.is_some()
                    || anchor_event_count(connection, anchor_id)? != 0
                {
                    return Err(LedgerErrorClass::IdentityInvariant.into());
                }
            }
            _ => return Err(LedgerErrorClass::IdentityInvariant.into()),
        }
    }
    Ok(())
}

fn anchor_event_count(connection: &Connection, anchor_id: Uuid) -> Result<i64, LedgerError> {
    connection
        .query_row(
            "SELECT count(*) FROM trajectory_events WHERE anchor_id = ?1",
            params![anchor_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn historical_anchor_identity_is_canonical(
    connection: &Connection,
    anchor: &StoredAnchor,
) -> Result<bool, LedgerError> {
    let config = connection
        .query_row(
            "SELECT project_uuid, canonical_config_json, canonical_payload_hash
             FROM config_generations WHERE config_generation_id = ?1",
            params![anchor.config_generation_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((config_project, config_json, config_hash)) = config else {
        return Ok(false);
    };
    let Ok(config_value): Result<Json, _> = serde_json::from_str(&config_json) else {
        return Ok(false);
    };
    let policy = connection
        .query_row(
            "SELECT project_uuid, pool_id, canonical_policy_json, canonical_payload_hash
             FROM policy_versions
             WHERE project_uuid = ?1 AND pool_id = ?2 AND policy_version_id = ?3",
            params![
                anchor.project_uuid,
                anchor.pool_id,
                anchor.policy_version_id
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((policy_project, policy_pool, policy_json, policy_hash)) = policy else {
        return Ok(false);
    };
    let Ok(policy_value): Result<Json, _> = serde_json::from_str(&policy_json) else {
        return Ok(false);
    };
    let config_pool = config_value
        .get("pools")
        .and_then(Json::as_array)
        .and_then(|pools| {
            pools
                .iter()
                .find(|pool| pool.get("id").and_then(Json::as_str) == Some(anchor.pool_id.as_str()))
        });
    if config_project != anchor.project_uuid
        || config_hash != anchor.config_generation_id
        || canonical_json(&config_value).ok().as_deref() != Some(config_json.as_str())
        || canonical_sha256(&config_value).ok().as_deref()
            != Some(anchor.config_generation_id.as_str())
        || policy_project != anchor.project_uuid
        || policy_pool != anchor.pool_id
        || policy_hash != anchor.policy_version_id
        || canonical_json(&policy_value).ok().as_deref() != Some(policy_json.as_str())
        || canonical_sha256(&policy_value).ok().as_deref()
            != Some(anchor.policy_version_id.as_str())
        || policy_value.pointer("/pool") != config_pool
    {
        return Ok(false);
    }

    let learning = connection
        .query_row(
            "SELECT l.actor, l.reason, l.created_at_unix_ms,
                    l.canonical_payload_hash, s.learning_state_event_id,
                    s.state, s.actor, s.reason, s.created_at_unix_ms,
                    s.canonical_payload_hash
             FROM learning_generations AS l
             JOIN learning_generation_state_events AS s
               ON s.project_uuid = l.project_uuid
              AND s.pool_id = l.pool_id
              AND s.learning_generation_id = l.learning_generation_id
             WHERE l.project_uuid = ?1 AND l.pool_id = ?2
               AND l.learning_generation_id = ?3",
            params![
                anchor.project_uuid,
                anchor.pool_id,
                anchor.learning_generation_id
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, String>(9)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((
        actor,
        reason,
        created_at,
        stored_hash,
        state_event_id,
        state,
        state_actor,
        state_reason,
        state_created_at,
        stored_state_hash,
    )) = learning
    else {
        return Ok(false);
    };
    let Ok(learning_generation_id) = parse_uuid_v7(&anchor.learning_generation_id) else {
        return Ok(false);
    };
    let Ok(learning_state_event_id) = parse_uuid_v7(&state_event_id) else {
        return Ok(false);
    };
    let expected_hash = hash_json(&json!({
        "learning_generation_id": learning_generation_id,
        "project_uuid": parse_uuid_v7(&anchor.project_uuid)?,
        "pool_id": anchor.pool_id,
        "actor": actor,
        "reason": reason,
        "created_at_unix_ms": created_at,
    }))?;
    let expected_state_hash = hash_json(&json!({
        "learning_state_event_id": learning_state_event_id,
        "project_uuid": parse_uuid_v7(&anchor.project_uuid)?,
        "pool_id": anchor.pool_id,
        "learning_generation_id": learning_generation_id,
        "state": "current",
        "actor": state_actor,
        "reason": state_reason,
        "created_at_unix_ms": state_created_at,
    }))?;
    Ok(state == "current"
        && stored_hash == expected_hash
        && stored_state_hash == expected_state_hash)
}

fn reconstruct_and_orphan_closed_anchor(
    connection: &Connection,
    project_uuid: Uuid,
    reconciler_process_instance_id: Uuid,
    dead_process_instance_id: Uuid,
    anchor_id: Uuid,
    reconciled_at_unix_ms: i64,
) -> Result<usize, LedgerError> {
    let anchor = load_stored_anchor(connection, anchor_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if anchor.project_uuid != project_uuid.to_string()
        || anchor.process_instance_id != dead_process_instance_id.to_string()
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let pending = canonical_pending_anchor(connection, &anchor)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if !canonical_anchor_states_and_window(connection, &anchor, &pending)? {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let window = load_anchor_window(connection, &anchor.anchor_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if reconciled_at_unix_ms < window.closed_at_unix_ms {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let evaluator_version = historical_evaluator_version(connection, &anchor)?;
    let anchor_model = pending
        .request_projection
        .normalized_request
        .model
        .as_deref()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let mut candidates = pending.candidate_facts.clone();
    candidates.sort_by(|left, right| {
        (
            left.cost_rank,
            left.candidate_id.as_str(),
            left.model_revision.as_str(),
        )
            .cmp(&(
                right.cost_rank,
                right.candidate_id.as_str(),
                right.model_revision.as_str(),
            ))
    });
    let mut attempts = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let request_projection =
            candidate_request_projection(&pending.request_projection, &candidate.model)
                .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        attempts.push(ReservedShadowAttempt::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            candidate.candidate_id,
            candidate.model,
            candidate.model_revision,
            candidate.cost_rank,
            pending.request_projection.family,
            pending.replay_capability_facts.transport_identity.clone(),
            anchor_model,
            pending.anchor_model_revision.clone(),
            candidate.decoding_fingerprint,
            evaluator_version.clone(),
            pending
                .routing_context_projection
                .tenant_policy_hash
                .clone(),
            pending.routing_context_projection.agent_policy_hash.clone(),
            true,
            request_projection,
            window.closed_at_unix_ms,
        )?);
    }
    let reservation = SampleBatchReservation::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        anchor_id,
        pending.config_generation_id,
        pending.policy_version_id,
        pending.learning_generation_id,
        pending.pool_id,
        attempts,
        window.closed_at_unix_ms,
    )?;
    if !matches!(
        reserve_batch_in_savepoint(
            connection,
            project_uuid,
            dead_process_instance_id,
            &reservation,
        )?,
        DomainWrite::Applied
    ) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    orphan_reservation_attempts(
        connection,
        project_uuid,
        reconciler_process_instance_id,
        dead_process_instance_id,
        &reservation,
        ShadowTerminalClass::OrphanedBeforeSchedule,
        reconciled_at_unix_ms,
    )
}

fn orphan_existing_batch(
    connection: &Connection,
    project_uuid: Uuid,
    reconciler_process_instance_id: Uuid,
    dead_process_instance_id: Uuid,
    batch_id: Uuid,
    reconciled_at_unix_ms: i64,
) -> Result<usize, LedgerError> {
    let batch = load_batch_by_id(connection, batch_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if batch.project_uuid != project_uuid.to_string()
        || batch.process_instance_id != dead_process_instance_id.to_string()
        || load_batch_terminal_state(connection, batch_id)?.is_some()
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let attempts = load_attempts_by_batch(connection, batch_id)?;
    if attempts.is_empty() {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let mut incomplete = Vec::new();
    for attempt in attempts {
        let attempt_id = parse_uuid_v7(&attempt.shadow_attempt_id)?;
        if !stored_attempt_parent_is_canonical(connection, &attempt)?
            || !canonical_optional_started_state_matches(connection, &attempt)?
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        match (
            load_terminal_attempt_state(connection, attempt_id)?,
            load_result_by_attempt(connection, attempt_id)?,
        ) {
            (Some(_), Some(_)) if canonical_terminal_attempt_matches(connection, attempt_id)? => {}
            (None, None) => incomplete.push(attempt),
            _ => return Err(LedgerErrorClass::IdentityInvariant.into()),
        }
    }
    if incomplete.is_empty() {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let reservation = reservation_from_stored_batch(connection, batch_id, &batch, &incomplete)?;
    orphan_reservation_attempts(
        connection,
        project_uuid,
        reconciler_process_instance_id,
        dead_process_instance_id,
        &reservation,
        ShadowTerminalClass::OrphanedInFlight,
        reconciled_at_unix_ms,
    )
}

fn reservation_from_stored_batch(
    connection: &Connection,
    batch_id: Uuid,
    batch: &StoredBatch,
    incomplete: &[StoredAttempt],
) -> Result<SampleBatchReservation, LedgerError> {
    let open = load_batch_state(connection, batch_id, "open")?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let mut attempts = Vec::with_capacity(incomplete.len());
    for stored in incomplete {
        let reserved = load_attempt_state(
            connection,
            parse_uuid_v7(&stored.shadow_attempt_id)?,
            "reserved",
        )?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let request_projection =
            parse_canonical_json::<RouterRequestProjectionV1>(&stored.request_projection_json)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        attempts.push(ReservedShadowAttempt {
            shadow_attempt_id: parse_uuid_v7(&stored.shadow_attempt_id)?,
            reserved_state_event_id: parse_uuid_v7(&reserved.event_id)?,
            candidate_id: stored.candidate_id.clone(),
            candidate_model: stored.candidate_model.clone(),
            candidate_model_revision: stored.candidate_model_revision.clone(),
            cost_rank: u32::try_from(stored.cost_rank)
                .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
            api_family: enum_from_string(&stored.api_family)
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
            transport_identity: stored.transport_identity.clone(),
            anchor_model: stored.anchor_model.clone(),
            anchor_model_revision: stored.anchor_model_revision.clone(),
            decoding_fingerprint: stored.decoding_fingerprint.clone(),
            evaluator_version: stored.evaluator_version.clone(),
            tenant_policy_hash: stored.tenant_policy_hash.clone(),
            agent_policy_hash: stored.agent_policy_hash.clone(),
            eligible: stored.eligible == 1,
            request_projection,
            created_at_unix_ms: stored.created_at_unix_ms,
        });
    }
    Ok(SampleBatchReservation {
        sample_batch_id: batch_id,
        open_state_event_id: parse_uuid_v7(&open.event_id)?,
        conflict_health_event_id: Uuid::now_v7(),
        anchor_id: parse_uuid_v7(&batch.anchor_id)?,
        config_generation_id: batch.config_generation_id.clone(),
        policy_version_id: batch.policy_version_id.clone(),
        learning_generation_id: parse_uuid_v7(&batch.learning_generation_id)?,
        pool_id: batch.pool_id.clone(),
        attempts,
        created_at_unix_ms: batch.created_at_unix_ms,
    })
}

fn orphan_reservation_attempts(
    connection: &Connection,
    project_uuid: Uuid,
    reconciler_process_instance_id: Uuid,
    dead_process_instance_id: Uuid,
    reservation: &SampleBatchReservation,
    terminal_class: ShadowTerminalClass,
    created_at_unix_ms: i64,
) -> Result<usize, LedgerError> {
    if created_at_unix_ms < reservation.created_at_unix_ms {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let attempt_count = reservation.attempts.len();
    for (index, attempt) in reservation.attempts.iter().enumerate() {
        if created_at_unix_ms < attempt.created_at_unix_ms {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        if let Some(started) = load_attempt_state(connection, attempt.shadow_attempt_id, "started")?
            && created_at_unix_ms < started.created_at_unix_ms
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let batch_terminal = (index + 1 == attempt_count)
            .then(|| {
                SampleBatchTerminalEvent::new(
                    reservation.sample_batch_id,
                    Uuid::now_v7(),
                    terminal_class.batch_terminal_state(),
                    Some(dead_process_instance_id),
                    created_at_unix_ms,
                )
            })
            .transpose()?;
        let terminal = ShadowTerminalRecord::new(
            Uuid::now_v7(),
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            terminal_class,
            Some(dead_process_instance_id),
            None,
            None,
            None,
            None,
            None,
            None,
            ShadowVectorSourceV1::Canonicalizable {
                query_inputs: Box::new(attempt.request_projection.clone()),
            },
            batch_terminal,
            created_at_unix_ms,
        )?
        .with_vectorization(ShadowVectorizationHandoff::DeferredBackfill);
        if !matches!(
            terminal_in_savepoint(
                connection,
                project_uuid,
                reconciler_process_instance_id,
                &terminal,
            )?,
            DomainWrite::Applied
        ) {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
    }
    Ok(attempt_count)
}

fn historical_evaluator_version(
    connection: &Connection,
    anchor: &StoredAnchor,
) -> Result<String, LedgerError> {
    let policy_json: String = connection
        .query_row(
            "SELECT canonical_policy_json FROM policy_versions
             WHERE project_uuid = ?1 AND pool_id = ?2 AND policy_version_id = ?3",
            params![
                anchor.project_uuid,
                anchor.pool_id,
                anchor.policy_version_id
            ],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    let policy: Json = serde_json::from_str(&policy_json)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let policy_hash = canonical_sha256(&policy)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let evaluator_version = policy
        .pointer("/pool/judge/policy_sha256")
        .and_then(Json::as_str)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let judge: JudgeConfig = serde_json::from_value(
        policy
            .pointer("/pool/judge/config")
            .cloned()
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
    )
    .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if canonical_json(&policy).ok().as_deref() != Some(policy_json.as_str())
        || policy_hash != anchor.policy_version_id
        || judge.evaluator_version().ok().as_deref() != Some(evaluator_version)
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(evaluator_version.to_string())
}

fn reserve_batch_in_savepoint(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    reservation: &SampleBatchReservation,
) -> Result<DomainWrite, LedgerError> {
    let health_anchor_id =
        anchor_belongs_to_project(connection, project_uuid, reservation.anchor_id)?
            .then_some(reservation.anchor_id);
    let conflict = || DomainWrite::Conflict {
        anchor_id: health_anchor_id,
    };
    if !anchor_identity_matches(connection, project_uuid, process_instance_id, reservation)? {
        return Ok(conflict());
    }
    let by_id = load_batch_by_id(connection, reservation.sample_batch_id)?;
    let by_anchor = load_batch_by_anchor(connection, reservation.anchor_id)?;
    if by_id.is_some() || by_anchor.is_some() {
        let (Some(by_id), Some(by_anchor)) = (by_id, by_anchor) else {
            return Ok(conflict());
        };
        if by_id.sample_batch_id != by_anchor.sample_batch_id
            || !batch_matches(&by_id, project_uuid, process_instance_id, reservation)?
            || !open_batch_state_matches(connection, process_instance_id, reservation)?
            || !reserved_attempt_set_matches(
                connection,
                project_uuid,
                process_instance_id,
                reservation,
            )?
        {
            return Ok(conflict());
        }
        return Ok(DomainWrite::AlreadyApplied);
    }
    for attempt in &reservation.attempts {
        if load_attempt_by_id(connection, attempt.shadow_attempt_id)?.is_some()
            || load_attempt_by_candidate(
                connection,
                reservation.anchor_id,
                &attempt.candidate_id,
                &attempt.candidate_model_revision,
            )?
            .is_some()
        {
            return Ok(conflict());
        }
    }
    if batch_state_event_id_exists(connection, reservation.open_state_event_id)? {
        return Ok(conflict());
    }
    for attempt in &reservation.attempts {
        if attempt_state_event_id_exists(connection, attempt.reserved_state_event_id)? {
            return Ok(conflict());
        }
    }

    let batch_hash = batch_hash(project_uuid, process_instance_id, reservation)?;
    connection
        .execute(
            "INSERT INTO sample_batches (
                sample_batch_id, anchor_id, project_uuid, process_instance_id,
                config_generation_id, policy_version_id, learning_generation_id,
                pool_id, reserved_candidate_count, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                reservation.sample_batch_id.to_string(),
                reservation.anchor_id.to_string(),
                project_uuid.to_string(),
                process_instance_id.to_string(),
                reservation.config_generation_id,
                reservation.policy_version_id,
                reservation.learning_generation_id.to_string(),
                reservation.pool_id,
                i64::try_from(reservation.attempts.len())
                    .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
                reservation.created_at_unix_ms,
                batch_hash,
            ],
        )
        .map_err(database_error)?;
    insert_batch_state(
        connection,
        reservation.open_state_event_id,
        reservation.sample_batch_id,
        process_instance_id,
        None,
        "open",
        reservation.created_at_unix_ms,
    )?;
    for attempt in &reservation.attempts {
        insert_reserved_attempt(
            connection,
            project_uuid,
            process_instance_id,
            reservation,
            attempt,
        )?;
    }
    Ok(DomainWrite::Applied)
}

fn start_attempt_in_savepoint(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: ShadowAttemptStarted,
) -> Result<DomainWrite, LedgerError> {
    let Some(attempt) = load_attempt_by_id(connection, command.shadow_attempt_id)? else {
        return Ok(DomainWrite::Conflict { anchor_id: None });
    };
    let anchor_id = parse_uuid_v7(&attempt.anchor_id)?;
    let conflict = || DomainWrite::Conflict {
        anchor_id: Some(anchor_id),
    };
    if attempt.project_uuid != project_uuid.to_string()
        || attempt.process_instance_id != process_instance_id.to_string()
        || !stored_attempt_parent_is_canonical(connection, &attempt)?
        || command.created_at_unix_ms < attempt.created_at_unix_ms
    {
        return Ok(conflict());
    }
    let expected_hash = attempt_state_hash(
        command.state_event_id,
        command.shadow_attempt_id,
        process_instance_id,
        None,
        "started",
        command.created_at_unix_ms,
    )?;
    if let Some(stored) = load_attempt_state(connection, command.shadow_attempt_id, "started")? {
        if state_matches(
            &stored,
            command.state_event_id,
            process_instance_id,
            None,
            "started",
            command.created_at_unix_ms,
            &expected_hash,
        ) {
            return Ok(DomainWrite::AlreadyApplied);
        }
        return Ok(conflict());
    }
    if load_terminal_attempt_state(connection, command.shadow_attempt_id)?.is_some()
        || attempt_state_event_id_exists(connection, command.state_event_id)?
    {
        return Ok(conflict());
    }
    connection
        .execute(
            "INSERT INTO shadow_attempt_state_events (
                shadow_attempt_state_event_id, shadow_attempt_id,
                process_instance_id, state, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, 'started', ?4, ?5)",
            params![
                command.state_event_id.to_string(),
                command.shadow_attempt_id.to_string(),
                process_instance_id.to_string(),
                command.created_at_unix_ms,
                expected_hash,
            ],
        )
        .map_err(database_error)?;
    Ok(DomainWrite::Applied)
}

fn terminal_in_savepoint(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    command: &ShadowTerminalRecord,
) -> Result<DomainWrite, LedgerError> {
    let Some(attempt) = load_attempt_by_id(connection, command.shadow_attempt_id)? else {
        return Ok(DomainWrite::Conflict { anchor_id: None });
    };
    let anchor_id = parse_uuid_v7(&attempt.anchor_id)?;
    let conflict = || DomainWrite::Conflict {
        anchor_id: Some(anchor_id),
    };
    if attempt.project_uuid != project_uuid.to_string()
        || !stored_attempt_parent_is_canonical(connection, &attempt)?
        || (command.terminal_class.requires_started_state()
            && !canonical_started_state_matches(connection, &attempt)?)
        || (command.terminal_class.is_orphan()
            && !canonical_optional_started_state_matches(connection, &attempt)?)
        || command.created_at_unix_ms < attempt.created_at_unix_ms
        || load_attempt_state(connection, command.shadow_attempt_id, "started")?
            .is_some_and(|started| command.created_at_unix_ms < started.created_at_unix_ms)
    {
        return Ok(conflict());
    }
    let attempt_process_id = parse_uuid_v7(&attempt.process_instance_id)?;
    if command.terminal_class.is_orphan() {
        if command.dead_process_instance_id != Some(attempt_process_id)
            || attempt_process_id == process_instance_id
            || originating_process_is_live(connection, project_uuid, attempt_process_id)?
        {
            return Ok(conflict());
        }
    } else if attempt_process_id != process_instance_id
        || command.dead_process_instance_id.is_some()
    {
        return Ok(conflict());
    }
    if let ShadowVectorSourceV1::Canonicalizable { query_inputs } = &command.vector_source
        && canonical_serialize(query_inputs.as_ref())? != attempt.request_projection_json
    {
        return Ok(conflict());
    }
    if !terminal_evaluation_matches(connection, command)? {
        return Ok(conflict());
    }
    let sample_batch_id = parse_uuid_v7(&attempt.sample_batch_id)?;
    let prepared = prepare_result(command, &attempt, process_instance_id)?;
    let by_id = load_result_by_id(connection, command.shadow_result_id)?;
    let by_attempt = load_result_by_attempt(connection, command.shadow_attempt_id)?;
    if by_id.is_some() || by_attempt.is_some() {
        let (Some(by_id), Some(by_attempt)) = (by_id, by_attempt) else {
            return Ok(conflict());
        };
        if by_id.shadow_result_id != by_attempt.shadow_result_id
            || !result_matches(&by_id, command, &prepared)
            || !terminal_attempt_state_matches(
                connection,
                process_instance_id,
                command,
                &prepared.state_hash,
            )?
            || !batch_aggregate_matches(connection, process_instance_id, sample_batch_id, command)?
        {
            return Ok(conflict());
        }
        return Ok(DomainWrite::AlreadyApplied);
    }
    if load_terminal_attempt_state(connection, command.shadow_attempt_id)?.is_some()
        || attempt_state_event_id_exists(connection, command.state_event_id)?
    {
        return Ok(conflict());
    }
    if load_batch_terminal_state(connection, sample_batch_id)?.is_some() {
        return Ok(conflict());
    }
    if let Some(batch_terminal) = command.batch_terminal
        && (batch_terminal.sample_batch_id.to_string() != attempt.sample_batch_id
            || batch_state_event_id_exists(connection, batch_terminal.state_event_id)?
            || batch_terminal.state != command.terminal_class.batch_terminal_state()
            || batch_terminal.dead_process_instance_id != command.dead_process_instance_id
            || batch_terminal.created_at_unix_ms != command.created_at_unix_ms
            || max_attempt_terminal_at(connection, batch_terminal.sample_batch_id)?
                .is_some_and(|latest| batch_terminal.created_at_unix_ms < latest))
    {
        return Ok(conflict());
    }

    insert_result(connection, command, &prepared)?;
    connection
        .execute(
            "INSERT INTO shadow_attempt_state_events (
                shadow_attempt_state_event_id, shadow_attempt_id,
                process_instance_id, dead_process_instance_id, state,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                command.state_event_id.to_string(),
                command.shadow_attempt_id.to_string(),
                process_instance_id.to_string(),
                command
                    .dead_process_instance_id
                    .map(|value| value.to_string()),
                command.terminal_class.as_str(),
                command.created_at_unix_ms,
                prepared.state_hash,
            ],
        )
        .map_err(database_error)?;
    if batch_can_terminalize(connection, &attempt.sample_batch_id)?
        != command.batch_terminal.is_some()
    {
        return Ok(conflict());
    }
    if let Some(batch_terminal) = command.batch_terminal {
        insert_batch_state(
            connection,
            batch_terminal.state_event_id,
            batch_terminal.sample_batch_id,
            process_instance_id,
            batch_terminal.dead_process_instance_id,
            batch_terminal.state.as_str(),
            batch_terminal.created_at_unix_ms,
        )?;
    }
    Ok(DomainWrite::Applied)
}

struct PreparedResult {
    normalized_response_json: Option<String>,
    response_fingerprint: Option<String>,
    deterministic_hard_failure: Option<&'static str>,
    operational_failure_class: Option<String>,
    usage_json: Option<String>,
    query_inputs_json: Option<String>,
    partition_inputs_json: String,
    vector_source_hash: Option<String>,
    noncanonicalizable_reason: Option<String>,
    result_hash: String,
    state_hash: String,
}

fn prepare_result(
    command: &ShadowTerminalRecord,
    attempt: &StoredAttempt,
    process_instance_id: Uuid,
) -> Result<PreparedResult, LedgerError> {
    let normalized_response_json = command
        .normalized_response
        .as_ref()
        .map(canonical_serialize)
        .transpose()?;
    let response_fingerprint = command
        .normalized_response
        .as_ref()
        .map(|response| response.semantic_response_fingerprint.clone());
    let deterministic_hard_failure = command
        .deterministic_hard_failure
        .map(deterministic_failure_str);
    let operational_failure_class = command
        .operational_failure_class
        .as_ref()
        .map(|value| value.as_str().to_string());
    let usage_json = command
        .usage
        .as_ref()
        .map(canonical_serialize)
        .transpose()?;
    let (query_inputs_json, vector_source_hash, noncanonicalizable_reason) =
        match &command.vector_source {
            ShadowVectorSourceV1::Canonicalizable { query_inputs } => {
                let query_inputs_json = canonical_serialize(query_inputs.as_ref())?;
                let vector_source_hash = hash_json(&json!({
                    "query_inputs": serde_json::from_str::<Json>(&query_inputs_json)
                        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?,
                    "partition_inputs": serde_json::from_str::<Json>(&attempt.partition_inputs_json)
                        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
                }))?;
                (Some(query_inputs_json), Some(vector_source_hash), None)
            }
            ShadowVectorSourceV1::Noncanonicalizable { reason } => {
                (None, None, Some(reason.as_str().to_string()))
            }
        };
    let result_hash = hash_json(&json!({
        "shadow_result_id": command.shadow_result_id,
        "shadow_attempt_id": command.shadow_attempt_id,
        "terminal_class": command.terminal_class.as_str(),
        "normalized_response_json": normalized_response_json,
        "response_fingerprint": response_fingerprint,
        "deterministic_hard_failure": deterministic_hard_failure,
        "operational_failure_class": operational_failure_class,
        "latency_ms": command.latency_ms,
        "usage_json": usage_json,
        "evaluation_id": command.evaluation_id,
        "canonicalizable": matches!(&command.vector_source, ShadowVectorSourceV1::Canonicalizable { .. }),
        "query_inputs_json": query_inputs_json,
        "partition_inputs_json": attempt.partition_inputs_json,
        "vector_source_hash": vector_source_hash,
        "noncanonicalizable_reason": noncanonicalizable_reason,
        "created_at_unix_ms": command.created_at_unix_ms,
    }))?;
    let state_hash = attempt_state_hash(
        command.state_event_id,
        command.shadow_attempt_id,
        process_instance_id,
        command.dead_process_instance_id,
        command.terminal_class.as_str(),
        command.created_at_unix_ms,
    )?;
    Ok(PreparedResult {
        normalized_response_json,
        response_fingerprint,
        deterministic_hard_failure,
        operational_failure_class,
        usage_json,
        query_inputs_json,
        partition_inputs_json: attempt.partition_inputs_json.clone(),
        vector_source_hash,
        noncanonicalizable_reason,
        result_hash,
        state_hash,
    })
}

fn insert_result(
    connection: &Connection,
    command: &ShadowTerminalRecord,
    prepared: &PreparedResult,
) -> Result<(), LedgerError> {
    connection
        .execute(
            "INSERT INTO shadow_results (
                shadow_result_id, shadow_attempt_id, terminal_class,
                normalized_response_json, response_fingerprint,
                deterministic_hard_failure, operational_failure_class,
                latency_ms, usage_json, evaluation_id, canonicalizable,
                query_inputs_json, partition_inputs_json, vector_source_hash,
                noncanonicalizable_reason, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
                       ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            params![
                command.shadow_result_id.to_string(),
                command.shadow_attempt_id.to_string(),
                command.terminal_class.as_str(),
                prepared.normalized_response_json,
                prepared.response_fingerprint,
                prepared.deterministic_hard_failure,
                prepared.operational_failure_class,
                command.latency_ms,
                prepared.usage_json,
                command.evaluation_id.map(|value| value.to_string()),
                i64::from(matches!(
                    &command.vector_source,
                    ShadowVectorSourceV1::Canonicalizable { .. }
                )),
                prepared.query_inputs_json,
                prepared.partition_inputs_json,
                prepared.vector_source_hash,
                prepared.noncanonicalizable_reason,
                command.created_at_unix_ms,
                prepared.result_hash,
            ],
        )
        .map_err(database_error)?;
    Ok(())
}

fn insert_reserved_attempt(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    reservation: &SampleBatchReservation,
    attempt: &ReservedShadowAttempt,
) -> Result<(), LedgerError> {
    let request_projection_json = canonical_serialize(&attempt.request_projection)?;
    let partition_inputs_json = partition_inputs_json(reservation, attempt)?;
    let attempt_hash = attempt_hash(
        project_uuid,
        process_instance_id,
        reservation,
        attempt,
        &request_projection_json,
        &partition_inputs_json,
    )?;
    connection
        .execute(
            "INSERT INTO shadow_attempts (
                shadow_attempt_id, sample_batch_id, anchor_id, project_uuid,
                process_instance_id, config_generation_id, policy_version_id,
                learning_generation_id, pool_id, candidate_id, candidate_model,
                candidate_model_revision, cost_rank, api_family,
                transport_identity, anchor_model, anchor_model_revision,
                decoding_fingerprint, evaluator_version, tenant_policy_hash,
                agent_policy_hash, eligible, request_projection_json,
                partition_inputs_json, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
                       ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19,
                       ?20, ?21, ?22, ?23, ?24, ?25, ?26)",
            params![
                attempt.shadow_attempt_id.to_string(),
                reservation.sample_batch_id.to_string(),
                reservation.anchor_id.to_string(),
                project_uuid.to_string(),
                process_instance_id.to_string(),
                reservation.config_generation_id,
                reservation.policy_version_id,
                reservation.learning_generation_id.to_string(),
                reservation.pool_id,
                attempt.candidate_id,
                attempt.candidate_model,
                attempt.candidate_model_revision,
                i64::from(attempt.cost_rank),
                family_str(attempt.api_family),
                attempt.transport_identity,
                attempt.anchor_model,
                attempt.anchor_model_revision,
                attempt.decoding_fingerprint,
                attempt.evaluator_version,
                attempt.tenant_policy_hash,
                attempt.agent_policy_hash,
                i64::from(attempt.eligible),
                request_projection_json,
                partition_inputs_json,
                attempt.created_at_unix_ms,
                attempt_hash,
            ],
        )
        .map_err(database_error)?;
    let state_hash = attempt_state_hash(
        attempt.reserved_state_event_id,
        attempt.shadow_attempt_id,
        process_instance_id,
        None,
        "reserved",
        attempt.created_at_unix_ms,
    )?;
    connection
        .execute(
            "INSERT INTO shadow_attempt_state_events (
                shadow_attempt_state_event_id, shadow_attempt_id,
                process_instance_id, state, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, 'reserved', ?4, ?5)",
            params![
                attempt.reserved_state_event_id.to_string(),
                attempt.shadow_attempt_id.to_string(),
                process_instance_id.to_string(),
                attempt.created_at_unix_ms,
                state_hash,
            ],
        )
        .map_err(database_error)?;
    Ok(())
}

fn insert_batch_state(
    connection: &Connection,
    event_id: Uuid,
    sample_batch_id: Uuid,
    process_instance_id: Uuid,
    dead_process_instance_id: Option<Uuid>,
    state: &str,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let hash = batch_state_hash(
        event_id,
        sample_batch_id,
        process_instance_id,
        dead_process_instance_id,
        state,
        created_at_unix_ms,
    )?;
    connection
        .execute(
            "INSERT INTO sample_batch_state_events (
                sample_batch_state_event_id, sample_batch_id,
                process_instance_id, dead_process_instance_id, state,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                event_id.to_string(),
                sample_batch_id.to_string(),
                process_instance_id.to_string(),
                dead_process_instance_id.map(|value| value.to_string()),
                state,
                created_at_unix_ms,
                hash,
            ],
        )
        .map_err(database_error)?;
    Ok(())
}

fn anchor_identity_matches(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    reservation: &SampleBatchReservation,
) -> Result<bool, LedgerError> {
    let Some(anchor) = load_stored_anchor(connection, reservation.anchor_id)? else {
        return Ok(false);
    };
    if anchor.project_uuid != project_uuid.to_string()
        || anchor.process_instance_id != process_instance_id.to_string()
        || anchor.config_generation_id != reservation.config_generation_id
        || anchor.policy_version_id != reservation.policy_version_id
        || anchor.learning_generation_id != reservation.learning_generation_id.to_string()
        || anchor.pool_id != reservation.pool_id
    {
        return Ok(false);
    }
    let Some(pending) = canonical_pending_anchor(connection, &anchor)? else {
        return Ok(false);
    };
    if !canonical_anchor_states_and_window(connection, &anchor, &pending)? {
        return Ok(false);
    }
    reservation_attempts_match_anchor_and_policy(connection, &anchor, &pending, reservation)
}

fn load_stored_anchor(
    connection: &Connection,
    anchor_id: Uuid,
) -> Result<Option<StoredAnchor>, LedgerError> {
    connection
        .query_row(
            "SELECT a.anchor_id, a.project_uuid, a.process_instance_id,
                    a.config_generation_id, a.policy_version_id,
                    a.learning_generation_id, a.pool_id, a.anchor_call_uuid,
                    a.root_uuid, a.owner_uuid, a.owner_path_json, a.api_family,
                    a.transport_identity, a.anchor_model, a.anchor_model_revision,
                    a.replay_capability_fingerprint, a.decoding_fingerprint,
                    a.request_projection_json, a.routing_context_projection_json,
                    a.candidate_facts_json, a.requested_progress,
                    a.opened_at_unix_ms, a.deadline_at_unix_ms, a.non_resumable,
                    a.pending_hash, a.canonical_payload_hash, p.project_id
             FROM anchors AS a
             JOIN project_metadata AS p ON p.project_uuid = a.project_uuid
             WHERE a.anchor_id = ?1",
            params![anchor_id.to_string()],
            |row| {
                Ok(StoredAnchor {
                    anchor_id: row.get(0)?,
                    project_uuid: row.get(1)?,
                    process_instance_id: row.get(2)?,
                    config_generation_id: row.get(3)?,
                    policy_version_id: row.get(4)?,
                    learning_generation_id: row.get(5)?,
                    pool_id: row.get(6)?,
                    anchor_call_uuid: row.get(7)?,
                    root_uuid: row.get(8)?,
                    owner_uuid: row.get(9)?,
                    owner_path_json: row.get(10)?,
                    api_family: row.get(11)?,
                    transport_identity: row.get(12)?,
                    anchor_model: row.get(13)?,
                    anchor_model_revision: row.get(14)?,
                    replay_capability_fingerprint: row.get(15)?,
                    decoding_fingerprint: row.get(16)?,
                    request_projection_json: row.get(17)?,
                    routing_context_projection_json: row.get(18)?,
                    candidate_facts_json: row.get(19)?,
                    requested_progress: row.get(20)?,
                    opened_at_unix_ms: row.get(21)?,
                    deadline_at_unix_ms: row.get(22)?,
                    non_resumable: row.get(23)?,
                    pending_hash: row.get(24)?,
                    canonical_payload_hash: row.get(25)?,
                    project_id: row.get(26)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn canonical_pending_anchor(
    connection: &Connection,
    anchor: &StoredAnchor,
) -> Result<Option<PendingTrajectoryWindow>, LedgerError> {
    let Some(result) = connection
        .query_row(
            "SELECT normalized_response_json, semantic_response_fingerprint,
                    canonical_payload_hash
             FROM anchor_results WHERE anchor_id = ?1",
            params![anchor.anchor_id],
            |row| {
                Ok(StoredAnchorResult {
                    normalized_response_json: row.get(0)?,
                    semantic_response_fingerprint: row.get(1)?,
                    canonical_payload_hash: row.get(2)?,
                })
            },
        )
        .optional()
        .map_err(database_error)?
    else {
        return Ok(None);
    };
    let (Ok(anchor_id), Ok(anchor_call_uuid), Ok(root_uuid), Ok(owner_uuid)) = (
        parse_uuid_v7(&anchor.anchor_id),
        parse_uuid_v7(&anchor.anchor_call_uuid),
        parse_uuid_v7(&anchor.root_uuid),
        parse_uuid_v7(&anchor.owner_uuid),
    ) else {
        return Ok(None);
    };
    let (Ok(process_instance_id), Ok(project_uuid), Ok(learning_generation_id)) = (
        parse_uuid_v7(&anchor.process_instance_id),
        parse_uuid_v7(&anchor.project_uuid),
        parse_uuid_v7(&anchor.learning_generation_id),
    ) else {
        return Ok(None);
    };
    let Some(owner_path) =
        parse_canonical_json::<Vec<TrajectoryOwnerScopeV1>>(&anchor.owner_path_json)?
    else {
        return Ok(None);
    };
    let Some(request_projection) =
        parse_canonical_json::<RouterRequestProjectionV1>(&anchor.request_projection_json)?
    else {
        return Ok(None);
    };
    let Some(routing_context_projection) = parse_canonical_json::<RouterRoutingContextProjectionV1>(
        &anchor.routing_context_projection_json,
    )?
    else {
        return Ok(None);
    };
    let Some(candidate_facts) =
        parse_canonical_json::<Vec<PersistedCandidateFactV1>>(&anchor.candidate_facts_json)?
    else {
        return Ok(None);
    };
    let Some(normalized_anchor_response) =
        parse_canonical_json::<RouterResponseProjectionV1>(&result.normalized_response_json)?
    else {
        return Ok(None);
    };
    if !request_projection_is_canonical(&request_projection)?
        || !response_projection_is_canonical(&normalized_anchor_response)?
    {
        return Ok(None);
    }
    let Some(opened_at) = Utc.timestamp_millis_opt(anchor.opened_at_unix_ms).single() else {
        return Ok(None);
    };
    let Some(deadline_at) = Utc
        .timestamp_millis_opt(anchor.deadline_at_unix_ms)
        .single()
    else {
        return Ok(None);
    };
    let Ok(requested_progress) = usize::try_from(anchor.requested_progress) else {
        return Ok(None);
    };
    let replay_capability_facts = ReplayCapabilityFactsV1 {
        schema: REPLAY_CAPABILITY_SCHEMA_V1.to_string(),
        contract_version: LLM_REPLAY_CONTRACT_VERSION,
        api_family: request_projection.family,
        transport_identity: anchor.transport_identity.clone(),
        non_resumable: anchor.non_resumable != 0,
        capability_fingerprint: anchor.replay_capability_fingerprint.clone(),
    };
    let mut replay_value = serde_json::to_value(&replay_capability_facts)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    replay_value
        .as_object_mut()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?
        .remove("capability_fingerprint");
    let expected_replay_fingerprint = hash_json(&replay_value)?;
    let expected_decoding_fingerprint = hash_json(&json!({
        "schema": "nemo.relay.router.anchor-decoding-set@1",
        "candidate_facts": candidate_facts,
    }))?;
    let pending = PendingTrajectoryWindow {
        schema: PENDING_TRAJECTORY_SCHEMA_V1.to_string(),
        anchor_id,
        anchor_call_uuid,
        root_uuid,
        owner_uuid,
        owner_path,
        pool_id: anchor.pool_id.clone(),
        anchor_model_revision: anchor.anchor_model_revision.clone(),
        process_instance_id,
        project_uuid,
        project_id: anchor.project_id.clone(),
        config_generation_id: anchor.config_generation_id.clone(),
        policy_version_id: anchor.policy_version_id.clone(),
        learning_generation_id,
        request_projection,
        routing_context_projection,
        normalized_anchor_response,
        replay_capability_facts,
        candidate_facts,
        requested_progress,
        opened_at,
        deadline_at,
    };
    let pending_hash = pending
        .payload_hash()
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let response_hash = hash_json(
        &serde_json::to_value(&pending.normalized_anchor_response)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?,
    )?;
    let anchor_model = pending
        .request_projection
        .normalized_request
        .model
        .as_deref();
    if anchor.api_family != family_str(pending.request_projection.family)
        || anchor_model != Some(anchor.anchor_model.as_str())
        || anchor.non_resumable != 1
        || anchor.replay_capability_fingerprint != expected_replay_fingerprint
        || anchor.decoding_fingerprint != expected_decoding_fingerprint
        || anchor.pending_hash != pending_hash
        || anchor.canonical_payload_hash != pending_hash
        || result.semantic_response_fingerprint
            != pending
                .normalized_anchor_response
                .semantic_response_fingerprint
        || result.canonical_payload_hash != response_hash
    {
        return Ok(None);
    }
    Ok(Some(pending))
}

fn parse_canonical_json<T: serde::de::DeserializeOwned + serde::Serialize>(
    value: &str,
) -> Result<Option<T>, LedgerError> {
    let Ok(parsed) = serde_json::from_str::<T>(value) else {
        return Ok(None);
    };
    Ok((canonical_serialize(&parsed)? == value).then_some(parsed))
}

fn canonical_anchor_states_and_window(
    connection: &Connection,
    anchor: &StoredAnchor,
    pending: &PendingTrajectoryWindow,
) -> Result<bool, LedgerError> {
    canonical_terminal_anchor_states_and_window(connection, anchor, pending, "closed")
}

fn canonical_terminal_anchor_states_and_window(
    connection: &Connection,
    anchor: &StoredAnchor,
    pending: &PendingTrajectoryWindow,
    expected_terminal_kind: &str,
) -> Result<bool, LedgerError> {
    let pending_state = load_anchor_state(connection, &anchor.anchor_id, "pending")?;
    let terminal_state = load_anchor_state(connection, &anchor.anchor_id, expected_terminal_kind)?;
    let (Some(pending_state), Some(terminal_state)) = (pending_state, terminal_state) else {
        return Ok(false);
    };
    if !canonical_anchor_state_matches(&pending_state, anchor, "pending")?
        || !canonical_anchor_state_matches(&terminal_state, anchor, expected_terminal_kind)?
    {
        return Ok(false);
    }
    let Some(window) = load_anchor_window(connection, &anchor.anchor_id)? else {
        return Ok(false);
    };
    if window.anchor_id != anchor.anchor_id
        || window.requested_progress != anchor.requested_progress
        || window.terminal_kind != expected_terminal_kind
        || window.canonical_payload_hash != window.terminal_hash
        || pending_state.created_at_unix_ms < anchor.opened_at_unix_ms
        || terminal_state.created_at_unix_ms != window.closed_at_unix_ms
    {
        return Ok(false);
    }
    let (state, expected_partial, progress_reached) = match expected_terminal_kind {
        "closed" => {
            let Some(trigger) = window
                .trigger
                .as_deref()
                .and_then(|value| serde_json::from_value(Json::String(value.to_string())).ok())
            else {
                return Ok(false);
            };
            if window.rejection_reason.is_some() {
                return Ok(false);
            }
            (
                TrajectoryTerminalStateV1::Closed { trigger },
                trigger != TrajectoryTrigger::ProgressReached,
                trigger == TrajectoryTrigger::ProgressReached,
            )
        }
        "rejected" => {
            let Some(reason) = window
                .rejection_reason
                .as_deref()
                .and_then(|value| serde_json::from_value(Json::String(value.to_string())).ok())
            else {
                return Ok(false);
            };
            if window.trigger.is_some() {
                return Ok(false);
            }
            (TrajectoryTerminalStateV1::Rejected { reason }, true, false)
        }
        _ => return Ok(false),
    };
    let is_partial = window.is_partial != 0;
    let promotion_eligible = window.promotion_eligible != 0;
    if is_partial != expected_partial
        || (progress_reached && window.observed_progress != window.requested_progress)
        || (expected_terminal_kind == "closed"
            && !progress_reached
            && window.observed_progress >= window.requested_progress)
        || (is_partial && promotion_eligible)
        || window.closed_at_unix_ms < anchor.opened_at_unix_ms
    {
        return Ok(false);
    }
    let Some(closed_at) = Utc.timestamp_millis_opt(window.closed_at_unix_ms).single() else {
        return Ok(false);
    };
    let Ok(observed_progress) = usize::try_from(window.observed_progress) else {
        return Ok(false);
    };
    let Some(diagnostics) =
        parse_canonical_json::<Vec<TrajectoryDiagnosticV1>>(&window.diagnostics_json)?
    else {
        return Ok(false);
    };
    let Some(events) = load_canonical_trajectory_events(connection, &anchor.anchor_id)? else {
        return Ok(false);
    };
    let terminal = PersistedTrajectoryTerminalV1 {
        schema: TERMINAL_TRAJECTORY_SCHEMA_V1.to_string(),
        pending: pending.clone(),
        state,
        events,
        observed_progress,
        is_partial,
        promotion_eligible,
        closed_at,
        diagnostics,
    };
    let terminal_hash = terminal
        .payload_hash()
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    Ok(window.terminal_hash == terminal_hash)
}

fn load_anchor_state(
    connection: &Connection,
    anchor_id: &str,
    state: &str,
) -> Result<Option<StoredAnchorState>, LedgerError> {
    connection
        .query_row(
            "SELECT anchor_state_event_id, anchor_id, process_instance_id,
                    dead_process_instance_id, state, created_at_unix_ms,
                    canonical_payload_hash
             FROM anchor_state_events WHERE anchor_id = ?1 AND state = ?2",
            params![anchor_id, state],
            |row| {
                Ok(StoredAnchorState {
                    event_id: row.get(0)?,
                    anchor_id: row.get(1)?,
                    process_instance_id: row.get(2)?,
                    dead_process_instance_id: row.get(3)?,
                    state: row.get(4)?,
                    created_at_unix_ms: row.get(5)?,
                    canonical_payload_hash: row.get(6)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn canonical_anchor_state_matches(
    state: &StoredAnchorState,
    anchor: &StoredAnchor,
    expected_state: &str,
) -> Result<bool, LedgerError> {
    let (Ok(event_id), Ok(anchor_id), Ok(process_instance_id)) = (
        parse_uuid_v7(&state.event_id),
        parse_uuid_v7(&state.anchor_id),
        parse_uuid_v7(&state.process_instance_id),
    ) else {
        return Ok(false);
    };
    let expected_hash = hash_json(&json!({
        "anchor_state_event_id": event_id,
        "anchor_id": anchor_id,
        "process_instance_id": process_instance_id,
        "dead_process_instance_id": Json::Null,
        "state": expected_state,
        "created_at_unix_ms": state.created_at_unix_ms,
    }))?;
    Ok(state.anchor_id == anchor.anchor_id
        && state.process_instance_id == anchor.process_instance_id
        && state.dead_process_instance_id.is_none()
        && state.state == expected_state
        && state.canonical_payload_hash == expected_hash)
}

fn load_anchor_window(
    connection: &Connection,
    anchor_id: &str,
) -> Result<Option<StoredAnchorWindow>, LedgerError> {
    connection
        .query_row(
            "SELECT anchor_id, requested_progress, observed_progress,
                    terminal_kind, trigger, rejection_reason, is_partial,
                    promotion_eligible, closed_at_unix_ms, diagnostics_json,
                    terminal_hash, canonical_payload_hash
             FROM anchor_windows WHERE anchor_id = ?1",
            params![anchor_id],
            |row| {
                Ok(StoredAnchorWindow {
                    anchor_id: row.get(0)?,
                    requested_progress: row.get(1)?,
                    observed_progress: row.get(2)?,
                    terminal_kind: row.get(3)?,
                    trigger: row.get(4)?,
                    rejection_reason: row.get(5)?,
                    is_partial: row.get(6)?,
                    promotion_eligible: row.get(7)?,
                    closed_at_unix_ms: row.get(8)?,
                    diagnostics_json: row.get(9)?,
                    terminal_hash: row.get(10)?,
                    canonical_payload_hash: row.get(11)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn load_canonical_trajectory_events(
    connection: &Connection,
    anchor_id: &str,
) -> Result<Option<Vec<CapturedTrajectoryEvent>>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT anchor_id, ingest_seq, event_uuid, parent_uuid, kind,
                    phase, category, call_role, name, event_time_unix_ms,
                    schema_id, sanitized_payload_json, canonical_size_bytes,
                    canonical_payload_hash
             FROM trajectory_events WHERE anchor_id = ?1 ORDER BY ingest_seq",
        )
        .map_err(database_error)?;
    let rows = statement
        .query_map(params![anchor_id], |row| {
            Ok(StoredTrajectoryEvent {
                anchor_id: row.get(0)?,
                ingest_seq: row.get(1)?,
                event_uuid: row.get(2)?,
                parent_uuid: row.get(3)?,
                kind: row.get(4)?,
                phase: row.get(5)?,
                category: row.get(6)?,
                call_role: row.get(7)?,
                name: row.get(8)?,
                event_time_unix_ms: row.get(9)?,
                schema_id: row.get(10)?,
                sanitized_payload_json: row.get(11)?,
                canonical_size_bytes: row.get(12)?,
                canonical_payload_hash: row.get(13)?,
            })
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    let mut events = Vec::with_capacity(rows.len());
    for row in rows {
        let Some(event) =
            parse_canonical_json::<CapturedTrajectoryEvent>(&row.sanitized_payload_json)?
        else {
            return Ok(None);
        };
        if !trajectory_event_matches(&row, &event)? {
            return Ok(None);
        }
        events.push(event);
    }
    Ok(Some(events))
}

fn trajectory_event_matches(
    stored: &StoredTrajectoryEvent,
    event: &CapturedTrajectoryEvent,
) -> Result<bool, LedgerError> {
    let (Ok(_anchor_id), true) = (
        parse_uuid_v7(&stored.anchor_id),
        validate_uuid_v7(event.event_uuid).is_ok()
            && event
                .parent_uuid
                .is_none_or(|value| validate_uuid_v7(value).is_ok()),
    ) else {
        return Ok(false);
    };
    if validate_bounded_text(&event.name, 512).is_err()
        || validate_bounded_text(&event.schema, 256).is_err()
        || event
            .category
            .as_deref()
            .is_some_and(|value| validate_bounded_text(value, 256).is_err())
    {
        return Ok(false);
    }
    let mut semantic = serde_json::to_value(event)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let semantic = semantic
        .as_object_mut()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    for field in [
        "ingest_seq",
        "canonical_payload_hash",
        "canonical_size_bytes",
    ] {
        semantic.remove(field);
    }
    let expected_hash = hash_json(&Json::Object(semantic.clone()))?;
    let expected_kind = match event.kind {
        CapturedEventKind::Scope => "scope",
        CapturedEventKind::Mark => "mark",
    };
    let expected_phase = event.scope_phase.as_ref().map(enum_string).transpose()?;
    let expected_role = event.call_role.as_ref().map(enum_string).transpose()?;
    let Ok(ingest_seq) = i64::try_from(event.ingest_seq) else {
        return Ok(false);
    };
    let Ok(canonical_size) = i64::try_from(event.canonical_size_bytes) else {
        return Ok(false);
    };
    let actual_size = canonical_serialize(event)?.len();
    Ok(event.schema == CAPTURED_EVENT_SCHEMA_V1
        && ((expected_kind == "scope") == expected_phase.is_some())
        && stored.ingest_seq == ingest_seq
        && stored.event_uuid == event.event_uuid.to_string()
        && stored.parent_uuid == event.parent_uuid.map(|value| value.to_string())
        && stored.kind == expected_kind
        && stored.phase == expected_phase
        && stored.category == event.category
        && stored.call_role == expected_role
        && stored.name == event.name
        && stored.event_time_unix_ms == event.timestamp.timestamp_millis()
        && stored.schema_id == event.schema
        && stored.canonical_size_bytes == canonical_size
        && actual_size == event.canonical_size_bytes
        && event.canonical_payload_hash == expected_hash
        && stored.canonical_payload_hash == expected_hash)
}

fn enum_string<T: serde::Serialize>(value: &T) -> Result<String, LedgerError> {
    serde_json::to_value(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn reservation_attempts_match_anchor_and_policy(
    connection: &Connection,
    anchor: &StoredAnchor,
    pending: &PendingTrajectoryWindow,
    reservation: &SampleBatchReservation,
) -> Result<bool, LedgerError> {
    let Some(window) = load_anchor_window(connection, &anchor.anchor_id)? else {
        return Ok(false);
    };
    if reservation.created_at_unix_ms < window.closed_at_unix_ms
        || reservation
            .attempts
            .iter()
            .any(|attempt| attempt.created_at_unix_ms < reservation.created_at_unix_ms)
    {
        return Ok(false);
    }
    let stored = connection
        .query_row(
            "SELECT canonical_policy_json, canonical_payload_hash
             FROM policy_versions
             WHERE project_uuid = ?1 AND pool_id = ?2 AND policy_version_id = ?3",
            params![
                anchor.project_uuid,
                anchor.pool_id,
                anchor.policy_version_id,
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    let Some((policy_json, policy_payload_hash)) = stored else {
        return Ok(false);
    };
    let Ok(policy): Result<Json, _> = serde_json::from_str(&policy_json) else {
        return Ok(false);
    };
    if canonical_json(&policy).ok().as_deref() != Some(policy_json.as_str())
        || canonical_sha256(&policy).ok().as_deref() != Some(anchor.policy_version_id.as_str())
        || policy_payload_hash != anchor.policy_version_id
    {
        return Ok(false);
    }
    let (Some(pool), Some(config), Some(config_hash), Some(derived), Some(evaluator_version)) = (
        policy.pointer("/pool"),
        policy.pointer("/pool/judge/config"),
        policy
            .pointer("/pool/judge/config_sha256")
            .and_then(Json::as_str),
        policy.pointer("/pool/judge/derived_policy"),
        policy
            .pointer("/pool/judge/policy_sha256")
            .and_then(Json::as_str),
    ) else {
        return Ok(false);
    };
    let Ok(reproduced_config_hash) = canonical_sha256(config) else {
        return Ok(false);
    };
    let Ok(typed_config): Result<JudgeConfig, _> = serde_json::from_value(config.clone()) else {
        return Ok(false);
    };
    let Ok(reproduced_evaluator_version) = canonical_sha256(&json!({
        "config_sha256": reproduced_config_hash,
        "derived_policy": derived,
    })) else {
        return Ok(false);
    };
    if reproduced_config_hash != config_hash
        || reproduced_evaluator_version != evaluator_version
        || typed_config.evaluator_version().ok().as_deref() != Some(evaluator_version)
        || pool.get("id").and_then(Json::as_str) != Some(anchor.pool_id.as_str())
        || pool.get("api_family").and_then(Json::as_str) != Some(anchor.api_family.as_str())
        || pool.get("anchor_revision").and_then(Json::as_str)
            != Some(anchor.anchor_model_revision.as_str())
        || !pool
            .get("anchor_models")
            .and_then(Json::as_array)
            .is_some_and(|models| {
                models
                    .iter()
                    .any(|model| model.as_str() == Some(anchor.anchor_model.as_str()))
            })
    {
        return Ok(false);
    }
    let Some(policy_candidates) = pool.get("candidates").and_then(Json::as_array) else {
        return Ok(false);
    };
    let Some(max_candidates) = pool.get("max_candidates_per_sample").and_then(Json::as_u64) else {
        return Ok(false);
    };
    if reservation.attempts.is_empty()
        || u64::try_from(reservation.attempts.len()).unwrap_or(u64::MAX) > max_candidates
    {
        return Ok(false);
    }
    let mut selected_facts = pending.candidate_facts.iter().collect::<Vec<_>>();
    selected_facts.sort_by(|left, right| {
        (left.cost_rank, left.candidate_id.as_str())
            .cmp(&(right.cost_rank, right.candidate_id.as_str()))
    });
    selected_facts.truncate(usize::try_from(max_candidates).unwrap_or(usize::MAX));
    if reservation.attempts.len() != selected_facts.len()
        || reservation
            .attempts
            .iter()
            .zip(&selected_facts)
            .any(|(attempt, fact)| attempt.candidate_id != fact.candidate_id)
    {
        return Ok(false);
    }
    for attempt in &reservation.attempts {
        let Some(fact) = pending
            .candidate_facts
            .iter()
            .find(|fact| fact.candidate_id == attempt.candidate_id)
        else {
            return Ok(false);
        };
        let Some(policy_candidate) = policy_candidates.iter().find(|candidate| {
            candidate.get("id").and_then(Json::as_str) == Some(attempt.candidate_id.as_str())
        }) else {
            return Ok(false);
        };
        let capabilities = serde_json::to_value(&fact.capabilities)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let expected_request =
            candidate_request_projection(&pending.request_projection, &attempt.candidate_model)?;
        if fact.schema != CANDIDATE_FACT_SCHEMA_V1
            || fact.model != attempt.candidate_model
            || fact.model_revision != attempt.candidate_model_revision
            || fact.cost_rank != attempt.cost_rank
            || fact.decoding_fingerprint != attempt.decoding_fingerprint
            || policy_candidate.get("model").and_then(Json::as_str) != Some(fact.model.as_str())
            || policy_candidate
                .get("model_revision")
                .and_then(Json::as_str)
                != Some(fact.model_revision.as_str())
            || policy_candidate.get("cost_rank").and_then(Json::as_u64)
                != Some(u64::from(fact.cost_rank))
            || policy_candidate.get("capabilities") != Some(&capabilities)
            || attempt.api_family != pending.request_projection.family
            || attempt.transport_identity != anchor.transport_identity
            || attempt.anchor_model != anchor.anchor_model
            || attempt.anchor_model_revision != anchor.anchor_model_revision
            || attempt.evaluator_version != reproduced_evaluator_version
            || attempt.tenant_policy_hash != pending.routing_context_projection.tenant_policy_hash
            || attempt.agent_policy_hash != pending.routing_context_projection.agent_policy_hash
            || !attempt.eligible
            || !request_projection_is_canonical(&attempt.request_projection)?
            || attempt.request_projection != expected_request
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn anchor_belongs_to_project(
    connection: &Connection,
    project_uuid: Uuid,
    anchor_id: Uuid,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM anchors
                WHERE anchor_id = ?1 AND project_uuid = ?2
             )",
            params![anchor_id.to_string(), project_uuid.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn batch_state_event_id_exists(
    connection: &Connection,
    event_id: Uuid,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM sample_batch_state_events
                WHERE sample_batch_state_event_id = ?1
             )",
            params![event_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn attempt_state_event_id_exists(
    connection: &Connection,
    event_id: Uuid,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM shadow_attempt_state_events
                WHERE shadow_attempt_state_event_id = ?1
             )",
            params![event_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn reserved_attempt_set_matches(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    reservation: &SampleBatchReservation,
) -> Result<bool, LedgerError> {
    let stored_count: i64 = connection
        .query_row(
            "SELECT count(*) FROM shadow_attempts WHERE sample_batch_id = ?1",
            params![reservation.sample_batch_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    if stored_count
        != i64::try_from(reservation.attempts.len())
            .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?
    {
        return Ok(false);
    }
    for attempt in &reservation.attempts {
        let Some(stored) = load_attempt_by_id(connection, attempt.shadow_attempt_id)? else {
            return Ok(false);
        };
        let request_json = canonical_serialize(&attempt.request_projection)?;
        let partition_json = partition_inputs_json(reservation, attempt)?;
        let expected_hash = attempt_hash(
            project_uuid,
            process_instance_id,
            reservation,
            attempt,
            &request_json,
            &partition_json,
        )?;
        if !attempt_matches(
            &stored,
            project_uuid,
            process_instance_id,
            reservation,
            attempt,
            &request_json,
            &partition_json,
            &expected_hash,
        ) || !reserved_state_matches(connection, process_instance_id, attempt)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn batch_matches(
    stored: &StoredBatch,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    reservation: &SampleBatchReservation,
) -> Result<bool, LedgerError> {
    let expected_hash = batch_hash(project_uuid, process_instance_id, reservation)?;
    Ok(
        stored.sample_batch_id == reservation.sample_batch_id.to_string()
            && stored.anchor_id == reservation.anchor_id.to_string()
            && stored.project_uuid == project_uuid.to_string()
            && stored.process_instance_id == process_instance_id.to_string()
            && stored.config_generation_id == reservation.config_generation_id
            && stored.policy_version_id == reservation.policy_version_id
            && stored.learning_generation_id == reservation.learning_generation_id.to_string()
            && stored.pool_id == reservation.pool_id
            && stored.reserved_candidate_count
                == i64::try_from(reservation.attempts.len())
                    .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?
            && stored.created_at_unix_ms == reservation.created_at_unix_ms
            && stored.canonical_payload_hash == expected_hash,
    )
}

#[allow(clippy::too_many_arguments)]
fn attempt_matches(
    stored: &StoredAttempt,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    reservation: &SampleBatchReservation,
    attempt: &ReservedShadowAttempt,
    request_projection_json: &str,
    partition_inputs_json: &str,
    expected_hash: &str,
) -> bool {
    stored.shadow_attempt_id == attempt.shadow_attempt_id.to_string()
        && stored.sample_batch_id == reservation.sample_batch_id.to_string()
        && stored.anchor_id == reservation.anchor_id.to_string()
        && stored.project_uuid == project_uuid.to_string()
        && stored.process_instance_id == process_instance_id.to_string()
        && stored.config_generation_id == reservation.config_generation_id
        && stored.policy_version_id == reservation.policy_version_id
        && stored.learning_generation_id == reservation.learning_generation_id.to_string()
        && stored.pool_id == reservation.pool_id
        && stored.candidate_id == attempt.candidate_id
        && stored.candidate_model == attempt.candidate_model
        && stored.candidate_model_revision == attempt.candidate_model_revision
        && stored.cost_rank == i64::from(attempt.cost_rank)
        && stored.api_family == family_str(attempt.api_family)
        && stored.transport_identity == attempt.transport_identity
        && stored.anchor_model == attempt.anchor_model
        && stored.anchor_model_revision == attempt.anchor_model_revision
        && stored.decoding_fingerprint == attempt.decoding_fingerprint
        && stored.evaluator_version == attempt.evaluator_version
        && stored.tenant_policy_hash == attempt.tenant_policy_hash
        && stored.agent_policy_hash == attempt.agent_policy_hash
        && stored.eligible == i64::from(attempt.eligible)
        && stored.request_projection_json == request_projection_json
        && stored.partition_inputs_json == partition_inputs_json
        && stored.created_at_unix_ms == attempt.created_at_unix_ms
        && stored.canonical_payload_hash == expected_hash
}

fn result_matches(
    stored: &StoredResult,
    command: &ShadowTerminalRecord,
    prepared: &PreparedResult,
) -> bool {
    stored.shadow_result_id == command.shadow_result_id.to_string()
        && stored.shadow_attempt_id == command.shadow_attempt_id.to_string()
        && stored.terminal_class == command.terminal_class.as_str()
        && stored.normalized_response_json == prepared.normalized_response_json
        && stored.response_fingerprint == prepared.response_fingerprint
        && stored.deterministic_hard_failure
            == prepared.deterministic_hard_failure.map(str::to_string)
        && stored.operational_failure_class == prepared.operational_failure_class
        && stored.latency_ms == command.latency_ms
        && stored.usage_json == prepared.usage_json
        && stored.evaluation_id == command.evaluation_id.map(|value| value.to_string())
        && stored.canonicalizable
            == i64::from(matches!(
                &command.vector_source,
                ShadowVectorSourceV1::Canonicalizable { .. }
            ))
        && stored.query_inputs_json == prepared.query_inputs_json
        && stored.partition_inputs_json == prepared.partition_inputs_json
        && stored.vector_source_hash == prepared.vector_source_hash
        && stored.noncanonicalizable_reason == prepared.noncanonicalizable_reason
        && stored.created_at_unix_ms == command.created_at_unix_ms
        && stored.canonical_payload_hash == prepared.result_hash
}

fn terminal_evaluation_matches(
    connection: &Connection,
    command: &ShadowTerminalRecord,
) -> Result<bool, LedgerError> {
    match command.terminal_class {
        ShadowTerminalClass::Completed => {
            let (Some(evaluation_id), Some(normalized_response)) =
                (command.evaluation_id, command.normalized_response.as_ref())
            else {
                return Ok(false);
            };
            let normalized_response_json = canonical_serialize(normalized_response)?;
            Ok(
                load_verified_evaluation(connection, evaluation_id, command.shadow_attempt_id)?
                    .is_some_and(|evaluation| {
                        evaluation.matches_judge_response(
                            &normalized_response_json,
                            &normalized_response.semantic_response_fingerprint,
                        )
                    })
                    && evaluation_created_not_after(
                        connection,
                        evaluation_id,
                        command.shadow_attempt_id,
                        command.created_at_unix_ms,
                    )?,
            )
        }
        ShadowTerminalClass::DeterministicFailure => {
            let (Some(evaluation_id), Some(failure)) =
                (command.evaluation_id, command.deterministic_hard_failure)
            else {
                return Ok(false);
            };
            Ok(
                load_verified_evaluation(connection, evaluation_id, command.shadow_attempt_id)?
                    .is_some_and(|evaluation| {
                        evaluation.source == "deterministic_validator"
                            && evaluation.hard_failures.len() == 1
                            && evaluation.hard_failures[0] == deterministic_failure_str(failure)
                    })
                    && evaluation_created_not_after(
                        connection,
                        evaluation_id,
                        command.shadow_attempt_id,
                        command.created_at_unix_ms,
                    )?,
            )
        }
        ShadowTerminalClass::OperationalFailure
        | ShadowTerminalClass::SkippedCooloff
        | ShadowTerminalClass::CanceledShutdown
        | ShadowTerminalClass::OrphanedBeforeSchedule
        | ShadowTerminalClass::OrphanedInFlight => Ok(command.evaluation_id.is_none()),
    }
}

fn evaluation_created_not_after(
    connection: &Connection,
    evaluation_id: Uuid,
    shadow_attempt_id: Uuid,
    terminal_at_unix_ms: i64,
) -> Result<bool, LedgerError> {
    let created_at_unix_ms = connection
        .query_row(
            "SELECT created_at_unix_ms FROM evaluations
             WHERE evaluation_id = ?1 AND shadow_attempt_id = ?2",
            params![evaluation_id.to_string(), shadow_attempt_id.to_string()],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?;
    Ok(created_at_unix_ms.is_some_and(|created_at| created_at <= terminal_at_unix_ms))
}

struct VerifiedEvaluation {
    source: String,
    hard_failures: Vec<String>,
    judge_response_binding: Option<JudgeResponseBinding>,
    vector_projection: VerifiedVectorEvaluation,
}

struct JudgeResponseBinding {
    canonical_json: String,
    fingerprint: String,
}

impl VerifiedEvaluation {
    fn matches_judge_response(&self, canonical_json: &str, fingerprint: &str) -> bool {
        self.source == "judge"
            && self.judge_response_binding.as_ref().is_some_and(|binding| {
                binding.canonical_json == canonical_json && binding.fingerprint == fingerprint
            })
    }
}

fn load_verified_evaluation(
    connection: &Connection,
    evaluation_id: Uuid,
    shadow_attempt_id: Uuid,
) -> Result<Option<VerifiedEvaluation>, LedgerError> {
    let Some(stored) = load_evaluation(connection, evaluation_id)? else {
        return Ok(None);
    };
    if stored.shadow_attempt_id != shadow_attempt_id.to_string()
        || !stored_evaluation_is_canonical(connection, &stored)?
    {
        return Ok(None);
    }
    let Some(shadow) = verified_shadow_attempt_context(connection, shadow_attempt_id)? else {
        return Ok(None);
    };
    if stored.created_at_unix_ms < shadow.started_at_unix_ms {
        return Ok(None);
    }
    let judge_response_binding = if stored.source == "judge" {
        if !verified_judge_evaluation_provenance(connection, evaluation_id, shadow_attempt_id)? {
            return Ok(None);
        }
        let binding = connection
            .query_row(
                "SELECT candidate_response_json, candidate_response_fingerprint
                 FROM judge_attempts
                 WHERE shadow_attempt_id = ?1 AND evaluator_version = ?2
                 ORDER BY attempt_ordinal DESC LIMIT 1",
                params![shadow_attempt_id.to_string(), stored.evaluator_version],
                |row| {
                    Ok(JudgeResponseBinding {
                        canonical_json: row.get(0)?,
                        fingerprint: row.get(1)?,
                    })
                },
            )
            .optional()
            .map_err(database_error)?;
        let Some(binding) = binding else {
            return Ok(None);
        };
        Some(binding)
    } else {
        None
    };
    let hard_failures = load_evaluation_failures(connection, evaluation_id)?
        .into_iter()
        .map(|failure| failure.value)
        .collect();
    let Some(source) = enum_from_string::<JudgeEvaluationSourceV1>(&stored.source) else {
        return Ok(None);
    };
    let binary_label = match stored.binary_label.as_deref() {
        Some(value) => {
            let Some(value) = enum_from_string::<JudgeBinaryLabelV1>(value) else {
                return Ok(None);
            };
            Some(value)
        }
        None => None,
    };
    Ok(Some(VerifiedEvaluation {
        source: stored.source,
        hard_failures,
        judge_response_binding,
        vector_projection: VerifiedVectorEvaluation {
            evaluation_id,
            source,
            binary_label,
            judge_confidence: scored_value_from_bits(stored.judge_confidence_bits),
            promotion_eligible: stored.promotion_eligible != 0,
            created_at_unix_ms: stored.created_at_unix_ms,
        },
    }))
}

pub(super) fn verified_shadow_attempt_context(
    connection: &Connection,
    shadow_attempt_id: Uuid,
) -> Result<Option<VerifiedShadowAttemptContext>, LedgerError> {
    let Some(attempt) = load_attempt_by_id(connection, shadow_attempt_id)? else {
        return Ok(None);
    };
    if !stored_attempt_parent_is_canonical(connection, &attempt)?
        || !canonical_started_state_matches(connection, &attempt)?
    {
        return Ok(None);
    }
    let started = load_attempt_state(connection, shadow_attempt_id, "started")?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    validate_sha256(&attempt.evaluator_version)?;
    let Some(window) = load_anchor_window(connection, &attempt.anchor_id)? else {
        return Ok(None);
    };
    Ok(Some(VerifiedShadowAttemptContext {
        anchor_id: parse_uuid_v7(&attempt.anchor_id)?,
        process_instance_id: parse_uuid_v7(&attempt.process_instance_id)?,
        learning_generation_id: parse_uuid_v7(&attempt.learning_generation_id)?,
        evaluator_version: attempt.evaluator_version,
        is_partial: window.is_partial != 0,
        started_at_unix_ms: started.created_at_unix_ms,
    }))
}

/// Reconstruct one retained terminal without creating replay or evaluation authority.
pub(super) fn load_verified_backfill_vector_source(
    connection: &Connection,
    project_uuid: Uuid,
    shadow_attempt_id: Uuid,
) -> Result<Option<VerifiedBackfillVectorSource>, LedgerError> {
    let Some(stored_attempt) = load_attempt_by_id(connection, shadow_attempt_id)? else {
        return Ok(None);
    };
    if stored_attempt.project_uuid != project_uuid.to_string()
        || !canonical_terminal_attempt_matches(connection, shadow_attempt_id)?
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let result = load_result_by_attempt(connection, shadow_attempt_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let terminal_class = terminal_class_from_str(&result.terminal_class)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let evaluation_id = result
        .evaluation_id
        .as_deref()
        .map(parse_uuid_v7)
        .transpose()?;
    let evaluation = evaluation_id
        .map(|evaluation_id| {
            load_verified_evaluation(connection, evaluation_id, shadow_attempt_id)?
                .map(|verified| verified.vector_projection)
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))
        })
        .transpose()?;
    let vector_source = match result.canonicalizable {
        1 => {
            let query = result
                .query_inputs_json
                .as_deref()
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
            let query = parse_canonical_json::<RouterRequestProjectionV1>(query)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
            ShadowVectorSourceV1::Canonicalizable {
                query_inputs: Box::new(query),
            }
        }
        0 => ShadowVectorSourceV1::Noncanonicalizable {
            reason: NoncanonicalizableReason::new(
                result
                    .noncanonicalizable_reason
                    .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?,
            )?,
        },
        _ => return Err(LedgerErrorClass::CorruptDatabase.into()),
    };
    let anchor_id = parse_uuid_v7(&stored_attempt.anchor_id)?;
    let anchor = load_stored_anchor(connection, anchor_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let pending = canonical_pending_anchor(connection, &anchor)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let batch_id = parse_uuid_v7(&stored_attempt.sample_batch_id)?;
    let batch = load_batch_by_id(connection, batch_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let reservation = reservation_from_stored_batch(
        connection,
        batch_id,
        &batch,
        std::slice::from_ref(&stored_attempt),
    )?;
    let attempt = reservation
        .attempts
        .first()
        .cloned()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    Ok(Some(VerifiedBackfillVectorSource {
        reservation,
        attempt,
        shadow_result_id: parse_uuid_v7(&result.shadow_result_id)?,
        root_uuid: pending.root_uuid,
        routing_projection: pending.routing_context_projection,
        terminal_class,
        evaluation_id,
        evaluation,
        vector_source,
        terminal_at_unix_ms: result.created_at_unix_ms,
    }))
}

pub(super) fn verified_shadow_dependency_context(
    connection: &Connection,
    shadow_attempt_id: Uuid,
) -> Result<Option<VerifiedShadowDependencyContext>, LedgerError> {
    let Some(attempt) = load_attempt_by_id(connection, shadow_attempt_id)? else {
        return Ok(None);
    };
    if !stored_attempt_parent_is_canonical(connection, &attempt)? {
        return Ok(None);
    }
    Ok(Some(VerifiedShadowDependencyContext {
        anchor_id: parse_uuid_v7(&attempt.anchor_id)?,
        project_uuid: parse_uuid_v7(&attempt.project_uuid)?,
        process_instance_id: parse_uuid_v7(&attempt.process_instance_id)?,
        policy_version_id: attempt.policy_version_id,
        pool_id: attempt.pool_id,
        candidate_model: attempt.candidate_model,
        candidate_model_revision: attempt.candidate_model_revision,
        api_family: attempt.api_family,
        transport_identity: attempt.transport_identity,
        evaluator_version: attempt.evaluator_version,
    }))
}

pub(super) fn canonical_shadow_attempt_is_unterminalized(
    connection: &Connection,
    shadow_attempt_id: Uuid,
) -> Result<bool, LedgerError> {
    let Some(attempt) = load_attempt_by_id(connection, shadow_attempt_id)? else {
        return Ok(false);
    };
    Ok(stored_attempt_parent_is_canonical(connection, &attempt)?
        && load_result_by_attempt(connection, shadow_attempt_id)?.is_none()
        && load_terminal_attempt_state(connection, shadow_attempt_id)?.is_none())
}

pub(super) fn verified_shadow_judge_policy(
    connection: &Connection,
    shadow_attempt_id: Uuid,
) -> Result<Option<VerifiedShadowJudgePolicy>, LedgerError> {
    let Some(shadow) = verified_shadow_dependency_context(connection, shadow_attempt_id)? else {
        return Ok(None);
    };
    let stored = connection
        .query_row(
            "SELECT canonical_policy_json, canonical_payload_hash
             FROM policy_versions
             WHERE project_uuid = ?1 AND pool_id = ?2 AND policy_version_id = ?3",
            params![
                shadow.project_uuid.to_string(),
                shadow.pool_id,
                shadow.policy_version_id,
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    let Some((policy_json, policy_payload_hash)) = stored else {
        return Ok(None);
    };
    let Ok(policy): Result<Json, _> = serde_json::from_str(&policy_json) else {
        return Ok(None);
    };
    if canonical_json(&policy).ok().as_deref() != Some(policy_json.as_str())
        || canonical_sha256(&policy).ok().as_deref() != Some(shadow.policy_version_id.as_str())
        || policy_payload_hash != shadow.policy_version_id
        || policy.pointer("/pool/id").and_then(Json::as_str) != Some(shadow.pool_id.as_str())
        || policy.pointer("/pool/api_family").and_then(Json::as_str)
            != Some(shadow.api_family.as_str())
    {
        return Ok(None);
    }
    let (Some(config_value), Some(config_hash), Some(derived), Some(policy_hash)) = (
        policy.pointer("/pool/judge/config"),
        policy
            .pointer("/pool/judge/config_sha256")
            .and_then(Json::as_str),
        policy.pointer("/pool/judge/derived_policy"),
        policy
            .pointer("/pool/judge/policy_sha256")
            .and_then(Json::as_str),
    ) else {
        return Ok(None);
    };
    let Ok(config): Result<JudgeConfig, _> = serde_json::from_value(config_value.clone()) else {
        return Ok(None);
    };
    let Ok(reproduced_config_hash) = canonical_sha256(config_value) else {
        return Ok(None);
    };
    let Ok(reproduced_policy_hash) = canonical_sha256(&json!({
        "config_sha256": reproduced_config_hash,
        "derived_policy": derived,
    })) else {
        return Ok(None);
    };
    let Ok(expected_evaluator_version) = config.evaluator_version() else {
        return Ok(None);
    };
    let (Some(prompt_sha256), Some(rubric_sha256), Some(output_schema_sha256)) = (
        derived.get("prompt_template_sha256").and_then(Json::as_str),
        derived.get("rubric_template_sha256").and_then(Json::as_str),
        derived.get("output_schema_sha256").and_then(Json::as_str),
    ) else {
        return Ok(None);
    };
    if reproduced_config_hash != config_hash
        || reproduced_policy_hash != policy_hash
        || reproduced_policy_hash != shadow.evaluator_version
        || reproduced_policy_hash != expected_evaluator_version
        || [prompt_sha256, rubric_sha256, output_schema_sha256]
            .into_iter()
            .any(|value| validate_sha256(value).is_err())
    {
        return Ok(None);
    }
    Ok(Some(VerifiedShadowJudgePolicy {
        config,
        evaluator_version: reproduced_policy_hash,
        prompt_sha256: prompt_sha256.to_string(),
        rubric_sha256: rubric_sha256.to_string(),
        output_schema_sha256: output_schema_sha256.to_string(),
    }))
}

pub(super) fn verified_shadow_judge_input_context(
    connection: &Connection,
    shadow_attempt_id: Uuid,
    candidate_response: &RouterResponseProjectionV1,
) -> Result<Option<VerifiedShadowJudgeInputContext>, LedgerError> {
    if !response_projection_is_canonical(candidate_response)? {
        return Ok(None);
    }
    let Some(attempt) = load_attempt_by_id(connection, shadow_attempt_id)? else {
        return Ok(None);
    };
    if !stored_attempt_parent_is_canonical(connection, &attempt)? {
        return Ok(None);
    }
    let anchor_id = parse_uuid_v7(&attempt.anchor_id)?;
    let Some(anchor) = load_stored_anchor(connection, anchor_id)? else {
        return Ok(None);
    };
    let Some(pending) = canonical_pending_anchor(connection, &anchor)? else {
        return Ok(None);
    };
    if !canonical_anchor_states_and_window(connection, &anchor, &pending)? {
        return Ok(None);
    }
    let Some(window) = load_anchor_window(connection, &attempt.anchor_id)? else {
        return Ok(None);
    };
    let Some(trigger) = window
        .trigger
        .as_deref()
        .and_then(enum_from_string::<TrajectoryTrigger>)
    else {
        return Ok(None);
    };
    let (Ok(requested_progress), Ok(observed_progress)) = (
        usize::try_from(window.requested_progress),
        usize::try_from(window.observed_progress),
    ) else {
        return Ok(None);
    };
    let Ok(horizon) = JudgeHorizonV1::new(
        requested_progress,
        observed_progress,
        trigger,
        window.is_partial != 0,
    ) else {
        return Ok(None);
    };
    let Some(events) = load_canonical_trajectory_events(connection, &attempt.anchor_id)? else {
        return Ok(None);
    };
    let Some(policy) = verified_shadow_judge_policy(connection, shadow_attempt_id)? else {
        return Ok(None);
    };
    let Ok(policy_identity) = JudgePolicyIdentityV1::from_config(&policy.config) else {
        return Ok(None);
    };
    let events = events.into_iter().map(Arc::new).collect::<Vec<_>>();
    let Ok(input) = PairwiseJudgeInputV1::new(
        &pending.request_projection,
        &pending.normalized_anchor_response,
        candidate_response,
        &events,
        horizon,
        policy_identity,
    ) else {
        return Ok(None);
    };
    let judge_input_sha256 = input
        .canonical_sha256()
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    Ok(Some(VerifiedShadowJudgeInputContext {
        judge_input_sha256,
        candidate_response_json: canonical_serialize(candidate_response)?,
        candidate_response_fingerprint: candidate_response.semantic_response_fingerprint.clone(),
    }))
}

fn stored_attempt_parent_is_canonical(
    connection: &Connection,
    attempt: &StoredAttempt,
) -> Result<bool, LedgerError> {
    if !stored_attempt_hash_is_valid(attempt)?
        || !canonical_reserved_state_matches(connection, attempt)?
    {
        return Ok(false);
    }
    let (Ok(attempt_id), Ok(sample_batch_id), Ok(anchor_id)) = (
        parse_uuid_v7(&attempt.shadow_attempt_id),
        parse_uuid_v7(&attempt.sample_batch_id),
        parse_uuid_v7(&attempt.anchor_id),
    ) else {
        return Ok(false);
    };
    let Some(batch) = load_batch_by_id(connection, sample_batch_id)? else {
        return Ok(false);
    };
    let (Ok(batch_anchor_id), Ok(batch_project_uuid), Ok(batch_process_id)) = (
        parse_uuid_v7(&batch.anchor_id),
        parse_uuid_v7(&batch.project_uuid),
        parse_uuid_v7(&batch.process_instance_id),
    ) else {
        return Ok(false);
    };
    let Ok(batch_learning_generation_id) = parse_uuid_v7(&batch.learning_generation_id) else {
        return Ok(false);
    };
    let expected_batch_hash = hash_json(&json!({
        "sample_batch_id": sample_batch_id,
        "anchor_id": batch_anchor_id,
        "project_uuid": batch_project_uuid,
        "process_instance_id": batch_process_id,
        "config_generation_id": batch.config_generation_id,
        "policy_version_id": batch.policy_version_id,
        "learning_generation_id": batch_learning_generation_id,
        "pool_id": batch.pool_id,
        "reserved_candidate_count": batch.reserved_candidate_count,
        "created_at_unix_ms": batch.created_at_unix_ms,
    }))?;
    let actual_attempt_count: i64 = connection
        .query_row(
            "SELECT count(*) FROM shadow_attempts WHERE sample_batch_id = ?1",
            params![batch.sample_batch_id],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    let Some(open_state) = load_batch_state(connection, sample_batch_id, "open")? else {
        return Ok(false);
    };
    let Ok(open_event_id) = parse_uuid_v7(&open_state.event_id) else {
        return Ok(false);
    };
    let expected_open_hash = batch_state_hash(
        open_event_id,
        sample_batch_id,
        batch_process_id,
        None,
        "open",
        batch.created_at_unix_ms,
    )?;
    if batch.sample_batch_id != attempt.sample_batch_id
        || batch.anchor_id != attempt.anchor_id
        || batch.project_uuid != attempt.project_uuid
        || batch.process_instance_id != attempt.process_instance_id
        || batch.config_generation_id != attempt.config_generation_id
        || batch.policy_version_id != attempt.policy_version_id
        || batch.learning_generation_id != attempt.learning_generation_id
        || batch.pool_id != attempt.pool_id
        || batch.reserved_candidate_count != actual_attempt_count
        || actual_attempt_count <= 0
        || batch.canonical_payload_hash != expected_batch_hash
        || !state_matches(
            &open_state,
            open_event_id,
            batch_process_id,
            None,
            "open",
            batch.created_at_unix_ms,
            &expected_open_hash,
        )
    {
        return Ok(false);
    }
    let Some(anchor) = load_stored_anchor(connection, anchor_id)? else {
        return Ok(false);
    };
    if anchor.project_uuid != batch.project_uuid
        || anchor.process_instance_id != batch.process_instance_id
        || anchor.config_generation_id != batch.config_generation_id
        || anchor.policy_version_id != batch.policy_version_id
        || anchor.learning_generation_id != batch.learning_generation_id
        || anchor.pool_id != batch.pool_id
    {
        return Ok(false);
    }
    let Some(pending) = canonical_pending_anchor(connection, &anchor)? else {
        return Ok(false);
    };
    if !canonical_anchor_states_and_window(connection, &anchor, &pending)? {
        return Ok(false);
    }
    let stored_attempts = load_attempts_by_batch(connection, sample_batch_id)?;
    if stored_attempts.len() != usize::try_from(actual_attempt_count).unwrap_or(usize::MAX)
        || !stored_attempts
            .iter()
            .any(|stored| stored.shadow_attempt_id == attempt_id.to_string())
    {
        return Ok(false);
    }
    let mut reserved_attempts = Vec::with_capacity(stored_attempts.len());
    for stored in &stored_attempts {
        if stored.sample_batch_id != batch.sample_batch_id
            || stored.anchor_id != batch.anchor_id
            || stored.project_uuid != batch.project_uuid
            || stored.process_instance_id != batch.process_instance_id
            || stored.config_generation_id != batch.config_generation_id
            || stored.policy_version_id != batch.policy_version_id
            || stored.learning_generation_id != batch.learning_generation_id
            || stored.pool_id != batch.pool_id
            || !stored_attempt_hash_is_valid(stored)?
            || !canonical_reserved_state_matches(connection, stored)?
        {
            return Ok(false);
        }
        let Ok(stored_attempt_id) = parse_uuid_v7(&stored.shadow_attempt_id) else {
            return Ok(false);
        };
        let Some(reserved_state) = load_attempt_state(connection, stored_attempt_id, "reserved")?
        else {
            return Ok(false);
        };
        let Ok(reserved_state_event_id) = parse_uuid_v7(&reserved_state.event_id) else {
            return Ok(false);
        };
        let Some(api_family) = enum_from_string::<LlmApiFamily>(&stored.api_family) else {
            return Ok(false);
        };
        let Some(request_projection) =
            parse_canonical_json::<RouterRequestProjectionV1>(&stored.request_projection_json)?
        else {
            return Ok(false);
        };
        if !request_projection_is_canonical(&request_projection)? {
            return Ok(false);
        }
        let Ok(cost_rank) = u32::try_from(stored.cost_rank) else {
            return Ok(false);
        };
        let eligible = match stored.eligible {
            0 => false,
            1 => true,
            _ => return Ok(false),
        };
        reserved_attempts.push(ReservedShadowAttempt {
            shadow_attempt_id: stored_attempt_id,
            reserved_state_event_id,
            candidate_id: stored.candidate_id.clone(),
            candidate_model: stored.candidate_model.clone(),
            candidate_model_revision: stored.candidate_model_revision.clone(),
            cost_rank,
            api_family,
            transport_identity: stored.transport_identity.clone(),
            anchor_model: stored.anchor_model.clone(),
            anchor_model_revision: stored.anchor_model_revision.clone(),
            decoding_fingerprint: stored.decoding_fingerprint.clone(),
            evaluator_version: stored.evaluator_version.clone(),
            tenant_policy_hash: stored.tenant_policy_hash.clone(),
            agent_policy_hash: stored.agent_policy_hash.clone(),
            eligible,
            request_projection,
            created_at_unix_ms: stored.created_at_unix_ms,
        });
    }
    let reservation = SampleBatchReservation {
        sample_batch_id,
        open_state_event_id: open_event_id,
        conflict_health_event_id: open_event_id,
        anchor_id,
        config_generation_id: batch.config_generation_id,
        policy_version_id: batch.policy_version_id,
        learning_generation_id: batch_learning_generation_id,
        pool_id: batch.pool_id,
        attempts: reserved_attempts,
        created_at_unix_ms: batch.created_at_unix_ms,
    };
    for (stored, reserved) in stored_attempts.iter().zip(&reservation.attempts) {
        let expected_partition = partition_inputs_json(&reservation, reserved)?;
        let expected_attempt_hash = attempt_hash(
            batch_project_uuid,
            batch_process_id,
            &reservation,
            reserved,
            &stored.request_projection_json,
            &expected_partition,
        )?;
        if stored.partition_inputs_json != expected_partition
            || stored.canonical_payload_hash != expected_attempt_hash
        {
            return Ok(false);
        }
    }
    reservation_attempts_match_anchor_and_policy(connection, &anchor, &pending, &reservation)
}

fn stored_attempt_hash_is_valid(stored: &StoredAttempt) -> Result<bool, LedgerError> {
    let expected = hash_json(&json!({
        "shadow_attempt_id": stored.shadow_attempt_id,
        "sample_batch_id": stored.sample_batch_id,
        "anchor_id": stored.anchor_id,
        "project_uuid": stored.project_uuid,
        "process_instance_id": stored.process_instance_id,
        "config_generation_id": stored.config_generation_id,
        "policy_version_id": stored.policy_version_id,
        "learning_generation_id": stored.learning_generation_id,
        "pool_id": stored.pool_id,
        "candidate_id": stored.candidate_id,
        "candidate_model": stored.candidate_model,
        "candidate_model_revision": stored.candidate_model_revision,
        "cost_rank": stored.cost_rank,
        "api_family": stored.api_family,
        "transport_identity": stored.transport_identity,
        "anchor_model": stored.anchor_model,
        "anchor_model_revision": stored.anchor_model_revision,
        "decoding_fingerprint": stored.decoding_fingerprint,
        "evaluator_version": stored.evaluator_version,
        "tenant_policy_hash": stored.tenant_policy_hash,
        "agent_policy_hash": stored.agent_policy_hash,
        "eligible": stored.eligible != 0,
        "request_projection_json": stored.request_projection_json,
        "partition_inputs_json": stored.partition_inputs_json,
        "created_at_unix_ms": stored.created_at_unix_ms,
    }))?;
    Ok(stored.canonical_payload_hash == expected)
}

fn canonical_reserved_state_matches(
    connection: &Connection,
    attempt: &StoredAttempt,
) -> Result<bool, LedgerError> {
    let attempt_id = parse_uuid_v7(&attempt.shadow_attempt_id)?;
    let process_instance_id = parse_uuid_v7(&attempt.process_instance_id)?;
    let Some(stored) = load_attempt_state(connection, attempt_id, "reserved")? else {
        return Ok(false);
    };
    let event_id = parse_uuid_v7(&stored.event_id)?;
    let expected_hash = attempt_state_hash(
        event_id,
        attempt_id,
        process_instance_id,
        None,
        "reserved",
        attempt.created_at_unix_ms,
    )?;
    Ok(state_matches(
        &stored,
        event_id,
        process_instance_id,
        None,
        "reserved",
        attempt.created_at_unix_ms,
        &expected_hash,
    ))
}

fn canonical_started_state_matches(
    connection: &Connection,
    attempt: &StoredAttempt,
) -> Result<bool, LedgerError> {
    let attempt_id = parse_uuid_v7(&attempt.shadow_attempt_id)?;
    let process_instance_id = parse_uuid_v7(&attempt.process_instance_id)?;
    let Some(stored) = load_attempt_state(connection, attempt_id, "started")? else {
        return Ok(false);
    };
    if stored.created_at_unix_ms < attempt.created_at_unix_ms {
        return Ok(false);
    }
    let event_id = parse_uuid_v7(&stored.event_id)?;
    let expected_hash = attempt_state_hash(
        event_id,
        attempt_id,
        process_instance_id,
        None,
        "started",
        stored.created_at_unix_ms,
    )?;
    Ok(state_matches(
        &stored,
        event_id,
        process_instance_id,
        None,
        "started",
        stored.created_at_unix_ms,
        &expected_hash,
    ))
}

fn canonical_optional_started_state_matches(
    connection: &Connection,
    attempt: &StoredAttempt,
) -> Result<bool, LedgerError> {
    let attempt_id = parse_uuid_v7(&attempt.shadow_attempt_id)?;
    if load_attempt_state(connection, attempt_id, "started")?.is_none() {
        return Ok(true);
    }
    canonical_started_state_matches(connection, attempt)
}

fn open_batch_state_matches(
    connection: &Connection,
    process_instance_id: Uuid,
    reservation: &SampleBatchReservation,
) -> Result<bool, LedgerError> {
    let Some(stored) = load_batch_state(connection, reservation.sample_batch_id, "open")? else {
        return Ok(false);
    };
    let expected_hash = batch_state_hash(
        reservation.open_state_event_id,
        reservation.sample_batch_id,
        process_instance_id,
        None,
        "open",
        reservation.created_at_unix_ms,
    )?;
    Ok(state_matches(
        &stored,
        reservation.open_state_event_id,
        process_instance_id,
        None,
        "open",
        reservation.created_at_unix_ms,
        &expected_hash,
    ))
}

fn reserved_state_matches(
    connection: &Connection,
    process_instance_id: Uuid,
    attempt: &ReservedShadowAttempt,
) -> Result<bool, LedgerError> {
    let Some(stored) = load_attempt_state(connection, attempt.shadow_attempt_id, "reserved")?
    else {
        return Ok(false);
    };
    let expected_hash = attempt_state_hash(
        attempt.reserved_state_event_id,
        attempt.shadow_attempt_id,
        process_instance_id,
        None,
        "reserved",
        attempt.created_at_unix_ms,
    )?;
    Ok(state_matches(
        &stored,
        attempt.reserved_state_event_id,
        process_instance_id,
        None,
        "reserved",
        attempt.created_at_unix_ms,
        &expected_hash,
    ))
}

fn terminal_attempt_state_matches(
    connection: &Connection,
    process_instance_id: Uuid,
    command: &ShadowTerminalRecord,
    expected_hash: &str,
) -> Result<bool, LedgerError> {
    let Some(stored) = load_terminal_attempt_state(connection, command.shadow_attempt_id)? else {
        return Ok(false);
    };
    Ok(state_matches(
        &stored,
        command.state_event_id,
        process_instance_id,
        command.dead_process_instance_id,
        command.terminal_class.as_str(),
        command.created_at_unix_ms,
        expected_hash,
    ))
}

fn batch_terminal_matches(
    connection: &Connection,
    process_instance_id: Uuid,
    command: &ShadowTerminalRecord,
) -> Result<bool, LedgerError> {
    let Some(expected) = command.batch_terminal else {
        return Ok(true);
    };
    let Some(stored) = load_batch_terminal_state(connection, expected.sample_batch_id)? else {
        return Ok(false);
    };
    let expected_hash = batch_state_hash(
        expected.state_event_id,
        expected.sample_batch_id,
        process_instance_id,
        expected.dead_process_instance_id,
        expected.state.as_str(),
        expected.created_at_unix_ms,
    )?;
    Ok(state_matches(
        &stored,
        expected.state_event_id,
        process_instance_id,
        expected.dead_process_instance_id,
        expected.state.as_str(),
        expected.created_at_unix_ms,
        &expected_hash,
    ))
}

fn batch_aggregate_matches(
    connection: &Connection,
    process_instance_id: Uuid,
    sample_batch_id: Uuid,
    command: &ShadowTerminalRecord,
) -> Result<bool, LedgerError> {
    let terminal = load_batch_terminal_state(connection, sample_batch_id)?;
    let terminal_exists = terminal.is_some();
    if batch_can_terminalize(connection, &sample_batch_id.to_string())? != terminal_exists {
        return Ok(false);
    }
    if let Some(terminal) = terminal.as_ref()
        && !canonical_batch_terminal_state(connection, sample_batch_id, terminal)?
    {
        return Ok(false);
    }
    let Some(expected) = command.batch_terminal else {
        return Ok(true);
    };
    if expected.sample_batch_id != sample_batch_id {
        return Ok(false);
    }
    batch_terminal_matches(connection, process_instance_id, command)
}

fn canonical_batch_terminal_state(
    connection: &Connection,
    sample_batch_id: Uuid,
    state: &StoredState,
) -> Result<bool, LedgerError> {
    let (Ok(event_id), Ok(process_instance_id)) = (
        parse_uuid_v7(&state.event_id),
        parse_uuid_v7(&state.process_instance_id),
    ) else {
        return Ok(false);
    };
    let dead_process_instance_id = match state.dead_process_instance_id.as_deref() {
        Some(value) => match parse_uuid_v7(value) {
            Ok(value) => Some(value),
            Err(_) => return Ok(false),
        },
        None => None,
    };
    let expected_hash = batch_state_hash(
        event_id,
        sample_batch_id,
        process_instance_id,
        dead_process_instance_id,
        &state.state,
        state.created_at_unix_ms,
    )?;
    if state.canonical_payload_hash != expected_hash {
        return Ok(false);
    }
    let batch_identity = connection
        .query_row(
            "SELECT project_uuid, created_at_unix_ms
             FROM sample_batches WHERE sample_batch_id = ?1",
            params![sample_batch_id.to_string()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    let Some((project_uuid, batch_created_at_unix_ms)) = batch_identity else {
        return Ok(false);
    };
    if state.created_at_unix_ms < batch_created_at_unix_ms
        || !process_belongs_to_project(connection, process_instance_id, &project_uuid)?
    {
        return Ok(false);
    }
    let latest_terminal = connection
        .query_row(
            "SELECT s.state, s.process_instance_id, s.dead_process_instance_id,
                    s.created_at_unix_ms
             FROM shadow_attempt_state_events AS s
             JOIN shadow_attempts AS a ON a.shadow_attempt_id = s.shadow_attempt_id
             WHERE a.sample_batch_id = ?1
               AND s.state IN (
                   'completed', 'deterministic_failure', 'operational_failure',
                   'skipped_cooloff', 'canceled_shutdown',
                   'orphaned_before_schedule', 'orphaned_in_flight'
               )
             ORDER BY s.event_seq DESC LIMIT 1",
            params![sample_batch_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((terminal_state, owner, dead_owner, terminal_at_unix_ms)) = latest_terminal else {
        return Ok(false);
    };
    if max_attempt_terminal_at(connection, sample_batch_id)?
        .is_some_and(|latest| state.created_at_unix_ms < latest)
    {
        return Ok(false);
    }
    Ok(
        terminal_class_from_str(&terminal_state).is_some_and(|terminal_class| {
            terminal_class.batch_terminal_state().as_str() == state.state
                && state.process_instance_id == owner
                && state.dead_process_instance_id == dead_owner
                && state.created_at_unix_ms == terminal_at_unix_ms
        }),
    )
}

fn process_belongs_to_project(
    connection: &Connection,
    process_instance_id: Uuid,
    project_uuid: &str,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM process_instances
                WHERE process_instance_id = ?1 AND project_uuid = ?2
             )",
            params![process_instance_id.to_string(), project_uuid],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn state_matches(
    stored: &StoredState,
    event_id: Uuid,
    process_instance_id: Uuid,
    dead_process_instance_id: Option<Uuid>,
    state: &str,
    created_at_unix_ms: i64,
    expected_hash: &str,
) -> bool {
    stored.event_id == event_id.to_string()
        && stored.process_instance_id == process_instance_id.to_string()
        && stored.dead_process_instance_id
            == dead_process_instance_id.map(|value| value.to_string())
        && stored.state == state
        && stored.created_at_unix_ms == created_at_unix_ms
        && stored.canonical_payload_hash == expected_hash
}

fn batch_can_terminalize(
    connection: &Connection,
    sample_batch_id: &str,
) -> Result<bool, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT shadow_attempt_id FROM shadow_attempts
             WHERE sample_batch_id = ?1 ORDER BY shadow_attempt_id",
        )
        .map_err(database_error)?;
    let attempt_ids = statement
        .query_map(params![sample_batch_id], |row| row.get::<_, String>(0))
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    if attempt_ids.is_empty() {
        return Ok(false);
    }
    for attempt_id in attempt_ids {
        let Ok(attempt_id) = parse_uuid_v7(&attempt_id) else {
            return Ok(false);
        };
        if !canonical_terminal_attempt_matches(connection, attempt_id)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn max_attempt_terminal_at(
    connection: &Connection,
    sample_batch_id: Uuid,
) -> Result<Option<i64>, LedgerError> {
    connection
        .query_row(
            "SELECT max(s.created_at_unix_ms)
             FROM shadow_attempt_state_events AS s
             JOIN shadow_attempts AS a ON a.shadow_attempt_id = s.shadow_attempt_id
             WHERE a.sample_batch_id = ?1
               AND s.state NOT IN ('reserved', 'started')",
            params![sample_batch_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn canonical_terminal_attempt_matches(
    connection: &Connection,
    attempt_id: Uuid,
) -> Result<bool, LedgerError> {
    let Some(attempt) = load_attempt_by_id(connection, attempt_id)? else {
        return Ok(false);
    };
    if !stored_attempt_parent_is_canonical(connection, &attempt)? {
        return Ok(false);
    }
    let Some(result) = load_result_by_attempt(connection, attempt_id)? else {
        return Ok(false);
    };
    let Some(terminal_class) = terminal_class_from_str(&result.terminal_class) else {
        return Ok(false);
    };
    if terminal_class.requires_started_state()
        && !canonical_started_state_matches(connection, &attempt)?
    {
        return Ok(false);
    }
    if terminal_class.is_orphan()
        && !canonical_optional_started_state_matches(connection, &attempt)?
    {
        return Ok(false);
    }
    let Some(state) = load_terminal_attempt_state(connection, attempt_id)? else {
        return Ok(false);
    };
    if state.created_at_unix_ms < attempt.created_at_unix_ms
        || load_attempt_state(connection, attempt_id, "started")?
            .is_some_and(|started| state.created_at_unix_ms < started.created_at_unix_ms)
    {
        return Ok(false);
    }
    let (Ok(state_event_id), Ok(state_process_id), Ok(attempt_process_id)) = (
        parse_uuid_v7(&state.event_id),
        parse_uuid_v7(&state.process_instance_id),
        parse_uuid_v7(&attempt.process_instance_id),
    ) else {
        return Ok(false);
    };
    let dead_process_id = match state.dead_process_instance_id.as_deref() {
        Some(value) => match parse_uuid_v7(value) {
            Ok(value) => Some(value),
            Err(_) => return Ok(false),
        },
        None => None,
    };
    let owner_matches = if terminal_class.is_orphan() {
        dead_process_id == Some(attempt_process_id)
            && state_process_id != attempt_process_id
            && process_belongs_to_project(connection, state_process_id, &attempt.project_uuid)?
    } else {
        dead_process_id.is_none() && state_process_id == attempt_process_id
    };
    let expected_state_hash = attempt_state_hash(
        state_event_id,
        attempt_id,
        state_process_id,
        dead_process_id,
        terminal_class.as_str(),
        state.created_at_unix_ms,
    )?;
    if !owner_matches
        || state.state != terminal_class.as_str()
        || state.canonical_payload_hash != expected_state_hash
        || state.created_at_unix_ms != result.created_at_unix_ms
    {
        return Ok(false);
    }
    stored_result_is_canonical(connection, &attempt, &result, terminal_class)
}

fn stored_result_is_canonical(
    connection: &Connection,
    attempt: &StoredAttempt,
    result: &StoredResult,
    terminal_class: ShadowTerminalClass,
) -> Result<bool, LedgerError> {
    let Ok(result_id) = parse_uuid_v7(&result.shadow_result_id) else {
        return Ok(false);
    };
    let Ok(attempt_id) = parse_uuid_v7(&result.shadow_attempt_id) else {
        return Ok(false);
    };
    if result.shadow_attempt_id != attempt.shadow_attempt_id {
        return Ok(false);
    }
    let normalized_response = match result.normalized_response_json.as_deref() {
        Some(value) => match parse_canonical_json::<RouterResponseProjectionV1>(value)? {
            Some(value) if response_projection_is_canonical(&value)? => Some(value),
            None => return Ok(false),
            Some(_) => return Ok(false),
        },
        None => None,
    };
    if normalized_response
        .as_ref()
        .map(|value| value.semantic_response_fingerprint.as_str())
        != result.response_fingerprint.as_deref()
    {
        return Ok(false);
    }
    if let Some(usage) = result.usage_json.as_deref()
        && parse_canonical_json::<SanitizedResponseUsageV1>(usage)?.is_none()
    {
        return Ok(false);
    }
    let evaluation_id = match result.evaluation_id.as_deref() {
        Some(value) => match parse_uuid_v7(value) {
            Ok(value) => Some(value),
            _ => return Ok(false),
        },
        None => None,
    };
    let evaluation_matches = match terminal_class {
        ShadowTerminalClass::Completed => {
            let Some(evaluation_id) = evaluation_id else {
                return Ok(false);
            };
            let (Some(normalized_response_json), Some(response_fingerprint)) = (
                result.normalized_response_json.as_deref(),
                result.response_fingerprint.as_deref(),
            ) else {
                return Ok(false);
            };
            result.normalized_response_json.is_some()
                && result.deterministic_hard_failure.is_none()
                && result.operational_failure_class.is_none()
                && load_verified_evaluation(connection, evaluation_id, attempt_id)?.is_some_and(
                    |evaluation| {
                        evaluation
                            .matches_judge_response(normalized_response_json, response_fingerprint)
                    },
                )
                && evaluation_created_not_after(
                    connection,
                    evaluation_id,
                    attempt_id,
                    result.created_at_unix_ms,
                )?
        }
        ShadowTerminalClass::DeterministicFailure => {
            let (Some(evaluation_id), Some(failure)) =
                (evaluation_id, result.deterministic_hard_failure.as_deref())
            else {
                return Ok(false);
            };
            result.operational_failure_class.is_none()
                && load_verified_evaluation(connection, evaluation_id, attempt_id)?.is_some_and(
                    |evaluation| {
                        evaluation.source == "deterministic_validator"
                            && evaluation.hard_failures == [failure]
                    },
                )
                && evaluation_created_not_after(
                    connection,
                    evaluation_id,
                    attempt_id,
                    result.created_at_unix_ms,
                )?
        }
        ShadowTerminalClass::OperationalFailure => {
            evaluation_id.is_none()
                && result.deterministic_hard_failure.is_none()
                && result
                    .operational_failure_class
                    .as_deref()
                    .is_some_and(|value| validate_stable_class(value).is_ok())
        }
        ShadowTerminalClass::SkippedCooloff
        | ShadowTerminalClass::CanceledShutdown
        | ShadowTerminalClass::OrphanedBeforeSchedule
        | ShadowTerminalClass::OrphanedInFlight => {
            evaluation_id.is_none()
                && result.deterministic_hard_failure.is_none()
                && result.operational_failure_class.is_none()
        }
    };
    if !evaluation_matches {
        return Ok(false);
    }
    let canonicalizable = match result.canonicalizable {
        1 => {
            let Some(query_json) = result.query_inputs_json.as_deref() else {
                return Ok(false);
            };
            let Some(query) = parse_canonical_json::<RouterRequestProjectionV1>(query_json)? else {
                return Ok(false);
            };
            let Some(partition) = parse_canonical_json::<Json>(&result.partition_inputs_json)?
            else {
                return Ok(false);
            };
            let expected_vector_hash = hash_json(&json!({
                "query_inputs": query,
                "partition_inputs": partition,
            }))?;
            query_json == attempt.request_projection_json
                && result.partition_inputs_json == attempt.partition_inputs_json
                && result.vector_source_hash.as_deref() == Some(expected_vector_hash.as_str())
                && result.noncanonicalizable_reason.is_none()
        }
        0 => {
            result.query_inputs_json.is_none()
                && result.vector_source_hash.is_none()
                && result
                    .noncanonicalizable_reason
                    .as_deref()
                    .is_some_and(|value| NoncanonicalizableReason::new(value).is_ok())
                && result.partition_inputs_json == attempt.partition_inputs_json
        }
        _ => false,
    };
    if !canonicalizable {
        return Ok(false);
    }
    let expected_hash = hash_json(&json!({
        "shadow_result_id": result_id,
        "shadow_attempt_id": attempt_id,
        "terminal_class": terminal_class.as_str(),
        "normalized_response_json": result.normalized_response_json,
        "response_fingerprint": result.response_fingerprint,
        "deterministic_hard_failure": result.deterministic_hard_failure,
        "operational_failure_class": result.operational_failure_class,
        "latency_ms": result.latency_ms,
        "usage_json": result.usage_json,
        "evaluation_id": evaluation_id,
        "canonicalizable": result.canonicalizable != 0,
        "query_inputs_json": result.query_inputs_json,
        "partition_inputs_json": result.partition_inputs_json,
        "vector_source_hash": result.vector_source_hash,
        "noncanonicalizable_reason": result.noncanonicalizable_reason,
        "created_at_unix_ms": result.created_at_unix_ms,
    }))?;
    Ok(result.canonical_payload_hash == expected_hash)
}

fn terminal_class_from_str(value: &str) -> Option<ShadowTerminalClass> {
    match value {
        "completed" => Some(ShadowTerminalClass::Completed),
        "deterministic_failure" => Some(ShadowTerminalClass::DeterministicFailure),
        "operational_failure" => Some(ShadowTerminalClass::OperationalFailure),
        "skipped_cooloff" => Some(ShadowTerminalClass::SkippedCooloff),
        "canceled_shutdown" => Some(ShadowTerminalClass::CanceledShutdown),
        "orphaned_before_schedule" => Some(ShadowTerminalClass::OrphanedBeforeSchedule),
        "orphaned_in_flight" => Some(ShadowTerminalClass::OrphanedInFlight),
        _ => None,
    }
}

fn load_batch_by_id(
    connection: &Connection,
    sample_batch_id: Uuid,
) -> Result<Option<StoredBatch>, LedgerError> {
    load_batch(connection, "sample_batch_id", &sample_batch_id.to_string())
}

fn load_batch_by_anchor(
    connection: &Connection,
    anchor_id: Uuid,
) -> Result<Option<StoredBatch>, LedgerError> {
    load_batch(connection, "anchor_id", &anchor_id.to_string())
}

fn load_batch(
    connection: &Connection,
    column: &str,
    value: &str,
) -> Result<Option<StoredBatch>, LedgerError> {
    let sql = match column {
        "sample_batch_id" => {
            "SELECT sample_batch_id, anchor_id, project_uuid, process_instance_id,
                    config_generation_id, policy_version_id, learning_generation_id,
                    pool_id, reserved_candidate_count, created_at_unix_ms,
                    canonical_payload_hash
             FROM sample_batches WHERE sample_batch_id = ?1"
        }
        "anchor_id" => {
            "SELECT sample_batch_id, anchor_id, project_uuid, process_instance_id,
                    config_generation_id, policy_version_id, learning_generation_id,
                    pool_id, reserved_candidate_count, created_at_unix_ms,
                    canonical_payload_hash
             FROM sample_batches WHERE anchor_id = ?1"
        }
        _ => return Err(LedgerErrorClass::IdentityInvariant.into()),
    };
    connection
        .query_row(sql, params![value], |row| {
            Ok(StoredBatch {
                sample_batch_id: row.get(0)?,
                anchor_id: row.get(1)?,
                project_uuid: row.get(2)?,
                process_instance_id: row.get(3)?,
                config_generation_id: row.get(4)?,
                policy_version_id: row.get(5)?,
                learning_generation_id: row.get(6)?,
                pool_id: row.get(7)?,
                reserved_candidate_count: row.get(8)?,
                created_at_unix_ms: row.get(9)?,
                canonical_payload_hash: row.get(10)?,
            })
        })
        .optional()
        .map_err(database_error)
}

fn load_attempt_by_id(
    connection: &Connection,
    shadow_attempt_id: Uuid,
) -> Result<Option<StoredAttempt>, LedgerError> {
    load_attempt(
        connection,
        "SELECT shadow_attempt_id, sample_batch_id, anchor_id, project_uuid,
                process_instance_id, config_generation_id, policy_version_id,
                learning_generation_id, pool_id, candidate_id, candidate_model,
                candidate_model_revision, cost_rank, api_family, transport_identity,
                anchor_model, anchor_model_revision, decoding_fingerprint,
                evaluator_version, tenant_policy_hash, agent_policy_hash, eligible,
                request_projection_json, partition_inputs_json, created_at_unix_ms,
                canonical_payload_hash
         FROM shadow_attempts WHERE shadow_attempt_id = ?1",
        params![shadow_attempt_id.to_string()],
    )
}

fn load_attempt_by_candidate(
    connection: &Connection,
    anchor_id: Uuid,
    candidate_id: &str,
    candidate_model_revision: &str,
) -> Result<Option<StoredAttempt>, LedgerError> {
    load_attempt(
        connection,
        "SELECT shadow_attempt_id, sample_batch_id, anchor_id, project_uuid,
                process_instance_id, config_generation_id, policy_version_id,
                learning_generation_id, pool_id, candidate_id, candidate_model,
                candidate_model_revision, cost_rank, api_family, transport_identity,
                anchor_model, anchor_model_revision, decoding_fingerprint,
                evaluator_version, tenant_policy_hash, agent_policy_hash, eligible,
                request_projection_json, partition_inputs_json, created_at_unix_ms,
                canonical_payload_hash
         FROM shadow_attempts
         WHERE anchor_id = ?1 AND candidate_id = ?2 AND candidate_model_revision = ?3",
        params![
            anchor_id.to_string(),
            candidate_id,
            candidate_model_revision
        ],
    )
}

fn load_attempt<P: rusqlite::Params>(
    connection: &Connection,
    sql: &str,
    parameters: P,
) -> Result<Option<StoredAttempt>, LedgerError> {
    connection
        .query_row(sql, parameters, stored_attempt_from_row)
        .optional()
        .map_err(database_error)
}

fn load_attempts_by_batch(
    connection: &Connection,
    sample_batch_id: Uuid,
) -> Result<Vec<StoredAttempt>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT shadow_attempt_id, sample_batch_id, anchor_id, project_uuid,
                    process_instance_id, config_generation_id, policy_version_id,
                    learning_generation_id, pool_id, candidate_id, candidate_model,
                    candidate_model_revision, cost_rank, api_family, transport_identity,
                    anchor_model, anchor_model_revision, decoding_fingerprint,
                    evaluator_version, tenant_policy_hash, agent_policy_hash, eligible,
                    request_projection_json, partition_inputs_json, created_at_unix_ms,
                    canonical_payload_hash
             FROM shadow_attempts WHERE sample_batch_id = ?1
             ORDER BY cost_rank, candidate_id",
        )
        .map_err(database_error)?;
    statement
        .query_map(
            params![sample_batch_id.to_string()],
            stored_attempt_from_row,
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)
}

fn stored_attempt_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredAttempt> {
    Ok(StoredAttempt {
        shadow_attempt_id: row.get(0)?,
        sample_batch_id: row.get(1)?,
        anchor_id: row.get(2)?,
        project_uuid: row.get(3)?,
        process_instance_id: row.get(4)?,
        config_generation_id: row.get(5)?,
        policy_version_id: row.get(6)?,
        learning_generation_id: row.get(7)?,
        pool_id: row.get(8)?,
        candidate_id: row.get(9)?,
        candidate_model: row.get(10)?,
        candidate_model_revision: row.get(11)?,
        cost_rank: row.get(12)?,
        api_family: row.get(13)?,
        transport_identity: row.get(14)?,
        anchor_model: row.get(15)?,
        anchor_model_revision: row.get(16)?,
        decoding_fingerprint: row.get(17)?,
        evaluator_version: row.get(18)?,
        tenant_policy_hash: row.get(19)?,
        agent_policy_hash: row.get(20)?,
        eligible: row.get(21)?,
        request_projection_json: row.get(22)?,
        partition_inputs_json: row.get(23)?,
        created_at_unix_ms: row.get(24)?,
        canonical_payload_hash: row.get(25)?,
    })
}

fn load_result_by_id(
    connection: &Connection,
    result_id: Uuid,
) -> Result<Option<StoredResult>, LedgerError> {
    load_result(connection, "shadow_result_id", &result_id.to_string())
}

fn load_result_by_attempt(
    connection: &Connection,
    attempt_id: Uuid,
) -> Result<Option<StoredResult>, LedgerError> {
    load_result(connection, "shadow_attempt_id", &attempt_id.to_string())
}

fn load_result(
    connection: &Connection,
    column: &str,
    value: &str,
) -> Result<Option<StoredResult>, LedgerError> {
    let predicate = match column {
        "shadow_result_id" => "shadow_result_id = ?1",
        "shadow_attempt_id" => "shadow_attempt_id = ?1",
        _ => return Err(LedgerErrorClass::IdentityInvariant.into()),
    };
    let sql = format!(
        "SELECT shadow_result_id, shadow_attempt_id, terminal_class,
                normalized_response_json, response_fingerprint,
                deterministic_hard_failure, operational_failure_class,
                latency_ms, usage_json, evaluation_id, canonicalizable,
                query_inputs_json, partition_inputs_json, vector_source_hash,
                noncanonicalizable_reason, created_at_unix_ms,
                canonical_payload_hash
         FROM shadow_results WHERE {predicate}"
    );
    connection
        .query_row(&sql, params![value], |row| {
            Ok(StoredResult {
                shadow_result_id: row.get(0)?,
                shadow_attempt_id: row.get(1)?,
                terminal_class: row.get(2)?,
                normalized_response_json: row.get(3)?,
                response_fingerprint: row.get(4)?,
                deterministic_hard_failure: row.get(5)?,
                operational_failure_class: row.get(6)?,
                latency_ms: row.get(7)?,
                usage_json: row.get(8)?,
                evaluation_id: row.get(9)?,
                canonicalizable: row.get(10)?,
                query_inputs_json: row.get(11)?,
                partition_inputs_json: row.get(12)?,
                vector_source_hash: row.get(13)?,
                noncanonicalizable_reason: row.get(14)?,
                created_at_unix_ms: row.get(15)?,
                canonical_payload_hash: row.get(16)?,
            })
        })
        .optional()
        .map_err(database_error)
}

fn load_evaluation(
    connection: &Connection,
    evaluation_id: Uuid,
) -> Result<Option<StoredEvaluation>, LedgerError> {
    connection
        .query_row(
            "SELECT evaluation_id, shadow_attempt_id, evaluator_version, source,
                    judge_model, judge_model_revision, prompt_version, prompt_sha256,
                    rubric_version, rubric_sha256, output_schema_version,
                    output_schema_sha256, response_equivalence,
                    response_equivalence_bits, trajectory_equivalence,
                    trajectory_equivalence_bits, judge_confidence,
                    judge_confidence_bits, response_weight, response_weight_bits,
                    trajectory_weight, trajectory_weight_bits, aggregate_score,
                    aggregate_score_bits, label, binary_label, rationale, is_partial,
                    promotion_eligible, created_at_unix_ms, canonical_payload_hash
             FROM evaluations WHERE evaluation_id = ?1",
            params![evaluation_id.to_string()],
            |row| {
                Ok(StoredEvaluation {
                    evaluation_id: row.get(0)?,
                    shadow_attempt_id: row.get(1)?,
                    evaluator_version: row.get(2)?,
                    source: row.get(3)?,
                    judge_model: row.get(4)?,
                    judge_model_revision: row.get(5)?,
                    prompt_version: row.get(6)?,
                    prompt_sha256: row.get(7)?,
                    rubric_version: row.get(8)?,
                    rubric_sha256: row.get(9)?,
                    output_schema_version: row.get(10)?,
                    output_schema_sha256: row.get(11)?,
                    response_equivalence: row.get(12)?,
                    response_equivalence_bits: row.get(13)?,
                    trajectory_equivalence: row.get(14)?,
                    trajectory_equivalence_bits: row.get(15)?,
                    judge_confidence: row.get(16)?,
                    judge_confidence_bits: row.get(17)?,
                    response_weight: row.get(18)?,
                    response_weight_bits: row.get(19)?,
                    trajectory_weight: row.get(20)?,
                    trajectory_weight_bits: row.get(21)?,
                    aggregate_score: row.get(22)?,
                    aggregate_score_bits: row.get(23)?,
                    label: row.get(24)?,
                    binary_label: row.get(25)?,
                    rationale: row.get(26)?,
                    is_partial: row.get(27)?,
                    promotion_eligible: row.get(28)?,
                    created_at_unix_ms: row.get(29)?,
                    canonical_payload_hash: row.get(30)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn load_evaluation_failures(
    connection: &Connection,
    evaluation_id: Uuid,
) -> Result<Vec<StoredEvaluationFailure>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT hard_failure, canonical_ordinal, canonical_payload_hash
             FROM evaluation_hard_failures
             WHERE evaluation_id = ?1 ORDER BY canonical_ordinal",
        )
        .map_err(database_error)?;
    let rows = statement
        .query_map(params![evaluation_id.to_string()], |row| {
            Ok(StoredEvaluationFailure {
                value: row.get(0)?,
                ordinal: row.get(1)?,
                canonical_payload_hash: row.get(2)?,
            })
        })
        .map_err(database_error)?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)
}

fn stored_evaluation_is_canonical(
    connection: &Connection,
    stored: &StoredEvaluation,
) -> Result<bool, LedgerError> {
    let evaluation_id = parse_uuid_v7(&stored.evaluation_id)?;
    if !score_pair_matches(
        stored.response_equivalence,
        stored.response_equivalence_bits,
        false,
    ) || !score_pair_matches(
        stored.trajectory_equivalence,
        stored.trajectory_equivalence_bits,
        false,
    ) || !score_pair_matches(stored.judge_confidence, stored.judge_confidence_bits, false)
        || !score_pair_matches(stored.response_weight, stored.response_weight_bits, false)
        || !score_pair_matches(
            stored.trajectory_weight,
            stored.trajectory_weight_bits,
            false,
        )
        || !score_pair_matches(stored.aggregate_score, stored.aggregate_score_bits, true)
    {
        return Ok(false);
    }
    let failures = load_evaluation_failures(connection, evaluation_id)?;
    let mut previous_ordinal = None;
    for failure in &failures {
        if previous_ordinal.is_some_and(|previous| previous >= failure.ordinal)
            || canonical_failure_ordinal(&failure.value) != Some(failure.ordinal)
        {
            return Ok(false);
        }
        previous_ordinal = Some(failure.ordinal);
        let expected_hash = hash_json(&json!({
            "evaluation_id": evaluation_id,
            "hard_failure": failure.value,
            "canonical_ordinal": failure.ordinal,
        }))?;
        if failure.canonical_payload_hash != expected_hash {
            return Ok(false);
        }
    }
    let hard_failures = failures
        .iter()
        .map(|failure| failure.value.as_str())
        .collect::<Vec<_>>();
    if !stored_evaluation_semantics_match(connection, stored, &failures)? {
        return Ok(false);
    }
    let expected_hash = hash_json(&json!({
        "evaluation_id": evaluation_id,
        "shadow_attempt_id": parse_uuid_v7(&stored.shadow_attempt_id)?,
        "evaluator_version": stored.evaluator_version,
        "source": stored.source,
        "judge_model": stored.judge_model,
        "judge_model_revision": stored.judge_model_revision,
        "prompt_version": stored.prompt_version,
        "prompt_sha256": stored.prompt_sha256,
        "rubric_version": stored.rubric_version,
        "rubric_sha256": stored.rubric_sha256,
        "output_schema_version": stored.output_schema_version,
        "output_schema_sha256": stored.output_schema_sha256,
        "response_equivalence": score_value_from_bits(stored.response_equivalence_bits),
        "response_equivalence_bits": stored_score_bits_identity(stored.response_equivalence_bits),
        "trajectory_equivalence": score_value_from_bits(stored.trajectory_equivalence_bits),
        "trajectory_equivalence_bits": stored_score_bits_identity(stored.trajectory_equivalence_bits),
        "judge_confidence": score_value_from_bits(stored.judge_confidence_bits),
        "judge_confidence_bits": stored_score_bits_identity(stored.judge_confidence_bits),
        "response_weight": score_value_from_bits(stored.response_weight_bits),
        "response_weight_bits": stored_score_bits_identity(stored.response_weight_bits),
        "trajectory_weight": score_value_from_bits(stored.trajectory_weight_bits),
        "trajectory_weight_bits": stored_score_bits_identity(stored.trajectory_weight_bits),
        "aggregate_score": score_value_from_bits(stored.aggregate_score_bits),
        "aggregate_score_bits": stored_score_bits_identity(stored.aggregate_score_bits),
        "label": stored.label,
        "binary_label": stored.binary_label,
        "hard_failures": hard_failures,
        "rationale": stored.rationale,
        "is_partial": stored.is_partial != 0,
        "promotion_eligible": stored.promotion_eligible != 0,
        "created_at_unix_ms": stored.created_at_unix_ms,
    }))?;
    Ok(stored.canonical_payload_hash == expected_hash)
}

fn stored_evaluation_semantics_match(
    connection: &Connection,
    stored: &StoredEvaluation,
    failures: &[StoredEvaluationFailure],
) -> Result<bool, LedgerError> {
    let shadow_attempt_id = parse_uuid_v7(&stored.shadow_attempt_id)?;
    let Some(shadow) = verified_shadow_attempt_context(connection, shadow_attempt_id)? else {
        return Ok(false);
    };
    let Some(policy) = verified_shadow_judge_policy(connection, shadow_attempt_id)? else {
        return Ok(false);
    };
    if stored.evaluator_version != policy.evaluator_version
        || (stored.is_partial != 0) != shadow.is_partial
    {
        return Ok(false);
    }
    let source = match enum_from_string::<JudgeEvaluationSourceV1>(&stored.source) {
        Some(value) => value,
        None => return Ok(false),
    };
    let label = match enum_from_string::<JudgeLabelV1>(&stored.label) {
        Some(value) => value,
        None => return Ok(false),
    };
    let binary_label = match stored.binary_label.as_deref() {
        Some(value) => match enum_from_string::<JudgeBinaryLabelV1>(value) {
            Some(value) => Some(value),
            None => return Ok(false),
        },
        None => None,
    };
    let hard_failures = failures
        .iter()
        .map(|failure| enum_from_string::<JudgeHardFailureV1>(&failure.value))
        .collect::<Option<Vec<_>>>();
    let Some(hard_failures) = hard_failures else {
        return Ok(false);
    };
    let evaluation = JudgeEvaluationV1 {
        source,
        response_equivalence: scored_value_from_bits(stored.response_equivalence_bits),
        trajectory_equivalence: scored_value_from_bits(stored.trajectory_equivalence_bits),
        judge_confidence: scored_value_from_bits(stored.judge_confidence_bits),
        response_weight: scored_value_from_bits(stored.response_weight_bits),
        trajectory_weight: scored_value_from_bits(stored.trajectory_weight_bits),
        aggregate: scored_value_from_bits(stored.aggregate_score_bits),
        label,
        binary_label,
        hard_failures,
        rationale: stored.rationale.clone(),
        is_partial: stored.is_partial != 0,
        promotion_eligible: stored.promotion_eligible != 0,
    };
    let contract_matches = match source {
        JudgeEvaluationSourceV1::DeterministicValidator => {
            stored.judge_model.is_none()
                && stored.judge_model_revision.is_none()
                && stored.prompt_version.is_none()
                && stored.prompt_sha256.is_none()
                && stored.rubric_version.is_none()
                && stored.rubric_sha256.is_none()
                && stored.output_schema_version.is_none()
                && stored.output_schema_sha256.is_none()
        }
        JudgeEvaluationSourceV1::Judge => {
            stored.judge_model.as_deref() == Some(policy.config.model.as_str())
                && stored.judge_model_revision.as_deref()
                    == Some(policy.config.model_revision.as_str())
                && stored.prompt_version.as_deref() == Some(policy.config.prompt_version.as_str())
                && stored.prompt_sha256.as_deref() == Some(policy.prompt_sha256.as_str())
                && stored.rubric_version.as_deref() == Some(policy.config.rubric_version.as_str())
                && stored.rubric_sha256.as_deref() == Some(policy.rubric_sha256.as_str())
                && stored.output_schema_version
                    == Some(i64::from(policy.config.output_schema_version))
                && stored.output_schema_sha256.as_deref()
                    == Some(policy.output_schema_sha256.as_str())
        }
    };
    Ok(contract_matches && evaluation.is_consistent_with_config(&policy.config))
}

fn scored_value_from_bits(bits: Option<i64>) -> Option<ScoredValueV1> {
    bits.map(|bits| ScoredValueV1 {
        value: f64::from_bits(bits as u64),
        bits: bits as u64,
    })
}

fn enum_from_string<T: serde::de::DeserializeOwned>(value: &str) -> Option<T> {
    serde_json::from_value(Json::String(value.to_string())).ok()
}

fn score_pair_matches(value: Option<f64>, bits: Option<i64>, aggregate: bool) -> bool {
    match (value, bits) {
        (None, None) => true,
        (Some(value), Some(bits)) => {
            let canonical = f64::from_bits(bits as u64);
            let in_range = if aggregate {
                (-0.0..=1.000_000_001).contains(&canonical)
            } else {
                (0.0..=1.0).contains(&canonical)
            };
            canonical.is_finite()
                && in_range
                && (value.to_bits() == canonical.to_bits() || (value == 0.0 && canonical == 0.0))
        }
        (None, Some(_)) | (Some(_), None) => false,
    }
}

fn score_value_from_bits(bits: Option<i64>) -> Option<f64> {
    bits.map(|bits| f64::from_bits(bits as u64))
}

fn stored_score_bits_identity(bits: Option<i64>) -> Option<String> {
    bits.map(|bits| format!("{:016x}", bits as u64))
}

fn canonical_failure_ordinal(value: &str) -> Option<i64> {
    match value {
        "tool_contract" => Some(0),
        "response_schema" => Some(1),
        "safety" => Some(2),
        "malformed_candidate" => Some(3),
        _ => None,
    }
}

fn load_attempt_state(
    connection: &Connection,
    attempt_id: Uuid,
    state: &str,
) -> Result<Option<StoredState>, LedgerError> {
    connection
        .query_row(
            "SELECT shadow_attempt_state_event_id, process_instance_id,
                    dead_process_instance_id, state, created_at_unix_ms,
                    canonical_payload_hash
             FROM shadow_attempt_state_events
             WHERE shadow_attempt_id = ?1 AND state = ?2",
            params![attempt_id.to_string(), state],
            stored_state_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn load_terminal_attempt_state(
    connection: &Connection,
    attempt_id: Uuid,
) -> Result<Option<StoredState>, LedgerError> {
    connection
        .query_row(
            "SELECT shadow_attempt_state_event_id, process_instance_id,
                    dead_process_instance_id, state, created_at_unix_ms,
                    canonical_payload_hash
             FROM shadow_attempt_state_events
             WHERE shadow_attempt_id = ?1 AND state NOT IN ('reserved', 'started')",
            params![attempt_id.to_string()],
            stored_state_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn load_batch_state(
    connection: &Connection,
    batch_id: Uuid,
    state: &str,
) -> Result<Option<StoredState>, LedgerError> {
    connection
        .query_row(
            "SELECT sample_batch_state_event_id, process_instance_id,
                    dead_process_instance_id, state, created_at_unix_ms,
                    canonical_payload_hash
             FROM sample_batch_state_events
             WHERE sample_batch_id = ?1 AND state = ?2",
            params![batch_id.to_string(), state],
            stored_state_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn load_batch_terminal_state(
    connection: &Connection,
    batch_id: Uuid,
) -> Result<Option<StoredState>, LedgerError> {
    connection
        .query_row(
            "SELECT sample_batch_state_event_id, process_instance_id,
                    dead_process_instance_id, state, created_at_unix_ms,
                    canonical_payload_hash
             FROM sample_batch_state_events
             WHERE sample_batch_id = ?1 AND state <> 'open'",
            params![batch_id.to_string()],
            stored_state_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn stored_state_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredState> {
    Ok(StoredState {
        event_id: row.get(0)?,
        process_instance_id: row.get(1)?,
        dead_process_instance_id: row.get(2)?,
        state: row.get(3)?,
        created_at_unix_ms: row.get(4)?,
        canonical_payload_hash: row.get(5)?,
    })
}

fn batch_hash(
    project_uuid: Uuid,
    process_instance_id: Uuid,
    reservation: &SampleBatchReservation,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "sample_batch_id": reservation.sample_batch_id,
        "anchor_id": reservation.anchor_id,
        "project_uuid": project_uuid,
        "process_instance_id": process_instance_id,
        "config_generation_id": reservation.config_generation_id,
        "policy_version_id": reservation.policy_version_id,
        "learning_generation_id": reservation.learning_generation_id,
        "pool_id": reservation.pool_id,
        "reserved_candidate_count": reservation.attempts.len(),
        "created_at_unix_ms": reservation.created_at_unix_ms,
    }))
}

fn attempt_hash(
    project_uuid: Uuid,
    process_instance_id: Uuid,
    reservation: &SampleBatchReservation,
    attempt: &ReservedShadowAttempt,
    request_projection_json: &str,
    partition_inputs_json: &str,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "shadow_attempt_id": attempt.shadow_attempt_id,
        "sample_batch_id": reservation.sample_batch_id,
        "anchor_id": reservation.anchor_id,
        "project_uuid": project_uuid,
        "process_instance_id": process_instance_id,
        "config_generation_id": reservation.config_generation_id,
        "policy_version_id": reservation.policy_version_id,
        "learning_generation_id": reservation.learning_generation_id,
        "pool_id": reservation.pool_id,
        "candidate_id": attempt.candidate_id,
        "candidate_model": attempt.candidate_model,
        "candidate_model_revision": attempt.candidate_model_revision,
        "cost_rank": attempt.cost_rank,
        "api_family": family_str(attempt.api_family),
        "transport_identity": attempt.transport_identity,
        "anchor_model": attempt.anchor_model,
        "anchor_model_revision": attempt.anchor_model_revision,
        "decoding_fingerprint": attempt.decoding_fingerprint,
        "evaluator_version": attempt.evaluator_version,
        "tenant_policy_hash": attempt.tenant_policy_hash,
        "agent_policy_hash": attempt.agent_policy_hash,
        "eligible": attempt.eligible,
        "request_projection_json": request_projection_json,
        "partition_inputs_json": partition_inputs_json,
        "created_at_unix_ms": attempt.created_at_unix_ms,
    }))
}

fn partition_inputs_json(
    reservation: &SampleBatchReservation,
    attempt: &ReservedShadowAttempt,
) -> Result<String, LedgerError> {
    canonical_json(&json!({
        "tenant_policy_hash": attempt.tenant_policy_hash,
        "agent_policy_hash": attempt.agent_policy_hash,
        "learning_generation_id": reservation.learning_generation_id,
        "api_family": attempt.api_family,
        "transport_identity": attempt.transport_identity,
        "anchor_model": attempt.anchor_model,
        "anchor_model_revision": attempt.anchor_model_revision,
        "candidate_id": attempt.candidate_id,
        "candidate_model_revision": attempt.candidate_model_revision,
        "decoding_fingerprint": attempt.decoding_fingerprint,
        "evaluator_version": attempt.evaluator_version,
    }))
    .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn batch_state_hash(
    event_id: Uuid,
    batch_id: Uuid,
    process_instance_id: Uuid,
    dead_process_instance_id: Option<Uuid>,
    state: &str,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "sample_batch_state_event_id": event_id,
        "sample_batch_id": batch_id,
        "process_instance_id": process_instance_id,
        "dead_process_instance_id": dead_process_instance_id,
        "state": state,
        "created_at_unix_ms": created_at_unix_ms,
    }))
}

fn attempt_state_hash(
    event_id: Uuid,
    attempt_id: Uuid,
    process_instance_id: Uuid,
    dead_process_instance_id: Option<Uuid>,
    state: &str,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "shadow_attempt_state_event_id": event_id,
        "shadow_attempt_id": attempt_id,
        "process_instance_id": process_instance_id,
        "dead_process_instance_id": dead_process_instance_id,
        "state": state,
        "created_at_unix_ms": created_at_unix_ms,
    }))
}

fn batch_terminal_json(event: SampleBatchTerminalEvent) -> Json {
    json!({
        "sample_batch_id": event.sample_batch_id,
        "state_event_id": event.state_event_id,
        "state": event.state.as_str(),
        "dead_process_instance_id": event.dead_process_instance_id,
        "created_at_unix_ms": event.created_at_unix_ms,
    })
}

fn deterministic_failure_str(value: DeterministicHardFailureV1) -> &'static str {
    match value {
        DeterministicHardFailureV1::ToolContract => "tool_contract",
        DeterministicHardFailureV1::ResponseSchema => "response_schema",
        DeterministicHardFailureV1::MalformedCandidate => "malformed_candidate",
    }
}

fn family_str(value: LlmApiFamily) -> &'static str {
    match value {
        LlmApiFamily::OpenAIChatCompletions => "openai_chat_completions",
        LlmApiFamily::OpenAIResponses => "openai_responses",
        LlmApiFamily::AnthropicMessages => "anthropic_messages",
    }
}

fn canonical_serialize<T: serde::Serialize>(value: &T) -> Result<String, LedgerError> {
    let value = serde_json::to_value(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    canonical_json(&value).map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn request_projection_is_canonical(
    projection: &RouterRequestProjectionV1,
) -> Result<bool, LedgerError> {
    Ok(validate_request_projection(projection).is_ok())
}

fn response_projection_is_canonical(
    projection: &RouterResponseProjectionV1,
) -> Result<bool, LedgerError> {
    let mut value = serde_json::to_value(projection)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    value
        .as_object_mut()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?
        .remove("semantic_response_fingerprint");
    Ok(projection.schema == RESPONSE_PROJECTION_SCHEMA_V1
        && projection.sanitizer_version == TRAJECTORY_SANITIZER_VERSION
        && projection.semantic_response_fingerprint == hash_json(&value)?)
}

fn candidate_request_projection(
    anchor: &RouterRequestProjectionV1,
    candidate_model: &str,
) -> Result<RouterRequestProjectionV1, LedgerError> {
    if !request_projection_is_canonical(anchor)? {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    crate::projection::candidate_request_projection(anchor, candidate_model)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn validate_bounded_text(value: &str, max_bytes: usize) -> Result<(), LedgerError> {
    if value.is_empty() || value.len() > max_bytes || value.chars().any(char::is_control) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn validate_stable_class(value: &str) -> Result<(), LedgerError> {
    if !matches!(
        value,
        "router.provider.transport"
            | "router.provider.authentication"
            | "router.provider.rate_limited"
            | "router.provider.timeout"
            | "router.provider.canceled"
            | "router.provider.unreadable_response"
            | "router.provider.truncated_response"
            | "router.provider.ambiguous_decode"
            | "router.provider.evidence_bound_exceeded"
            | "router.provider.middleware_interference"
            | "router.judge.output_invalid"
    ) {
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

fn validate_uuid_v7(value: Uuid) -> Result<(), LedgerError> {
    if value.get_version_num() != 7 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn parse_uuid_v7(value: &str) -> Result<Uuid, LedgerError> {
    let parsed = Uuid::parse_str(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    validate_uuid_v7(parsed)?;
    if parsed.to_string() != value {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(parsed)
}

fn hash_json(value: &Json) -> Result<String, LedgerError> {
    canonical_sha256(value).map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn database_error(error: rusqlite::Error) -> LedgerError {
    map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::Path;

    use chrono::{TimeZone, Utc};
    use nemo_relay::api::llm::LlmApiFamily;
    use nemo_relay::api::runtime::{LLM_REPLAY_CONTRACT_VERSION, LlmReplayCapability};
    use rusqlite::{Connection, params};
    use tempfile::tempdir;

    use super::*;
    use crate::config::LearningConfig;
    use crate::ledger::model::LedgerRuntimeIdentity;
    use crate::ledger::repository::anchors::{
        AnchorCommandAck, FrozenPendingAnchorV1, FrozenTerminalAnchorV1,
    };
    use crate::ledger::repository::judge::{
        JudgeAttemptStart, JudgeAttemptTerminal, JudgeRecordAck,
    };
    use crate::ledger::repository::process::ProcessStop;
    use crate::ledger::repository::tests::{
        assert_reconciliation_noop_twice, config, database_path, recovery_snapshot,
    };
    use crate::projection::{
        REQUEST_PROJECTION_SCHEMA_V1, ROUTER_SANITIZER_VERSION, ROUTING_CONTEXT_SCHEMA_V1,
        RouterRoutingContextProjectionV1, SanitizedAnnotatedLlmRequest, SanitizedMessage,
        SanitizedMessageContent,
    };
    use crate::trajectory::{
        PersistedCandidateCapabilitiesV1, PersistedCandidateFactV1, PersistedTrajectoryTerminalV1,
        RESPONSE_PROJECTION_SCHEMA_V1, ReplayCapabilityFactsV1, TRAJECTORY_SANITIZER_VERSION,
        TrajectoryTrigger,
    };

    const CANDIDATE_SUFFIXES: &[&str] = &[
        "alternate",
        "corrupt-start",
        "corrupt-terminal",
        "first",
        "guard",
        "judge-binding",
        "missing-anchor",
        "noncanonical",
        "one",
        "origin",
        "orphan-binding",
        "refused",
        "required-close",
        "second",
        "start",
        "state-id-first",
        "state-id-second",
        "swapped-source",
        "terminal",
        "two",
        "wrong-failure",
    ];

    fn activate(path: &Path, project_id: &str) -> super::super::ActivatedLedger {
        LedgerRepository::activate(&shadow_config(path, project_id)).unwrap()
    }

    fn activate_learning(path: &Path, project_id: &str) -> super::super::ActivatedLedger {
        let mut config = shadow_config(path, project_id);
        config.pools[0].learning = Some(LearningConfig::minimal("embedder-a"));
        LedgerRepository::activate_at(&config, 0).unwrap()
    }

    fn shadow_config(path: &Path, project_id: &str) -> crate::config::RouterConfig {
        let mut config = config(path, project_id);
        let template = config.pools[0].candidates[0].clone();
        config.pools[0].candidates = CANDIDATE_SUFFIXES
            .iter()
            .map(|suffix| {
                let mut candidate = template.clone();
                candidate.id = format!("candidate-{suffix}");
                candidate.model = format!("candidate-model-{suffix}");
                candidate.model_revision = format!("candidate-revision-{suffix}");
                candidate.cost_rank = 0;
                candidate
            })
            .collect();
        config.pools[0].max_candidates_per_sample = 2;
        config
    }

    fn policy_evaluator_version() -> String {
        config(Path::new("router.db"), "shadow-policy").pools[0]
            .judge
            .evaluator_version()
            .unwrap()
    }

    fn request_projection() -> RouterRequestProjectionV1 {
        let mut projection = RouterRequestProjectionV1 {
            schema: REQUEST_PROJECTION_SCHEMA_V1.to_string(),
            family: LlmApiFamily::OpenAIChatCompletions,
            normalized_request: SanitizedAnnotatedLlmRequest {
                messages: Vec::new(),
                model: Some("anchor-a".to_string()),
                params: None,
                tools: None,
                tool_choice: None,
                response_format: None,
                truncation: None,
                reasoning: None,
                service_tier: None,
                parallel_tool_calls: None,
                max_output_tokens: None,
                max_tool_calls: None,
                top_logprobs: None,
            },
            ordered_instructions: Vec::new(),
            response_format: None,
            response_schema_fingerprint: None,
            required_capabilities: Vec::new(),
            sanitizer_version: ROUTER_SANITIZER_VERSION,
            semantic_request_fingerprint: String::new(),
        };
        let mut value = serde_json::to_value(&projection).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("semantic_request_fingerprint");
        projection.semantic_request_fingerprint = hash_json(&value).unwrap();
        projection
    }

    fn request_projection_with_task() -> RouterRequestProjectionV1 {
        let mut projection = request_projection();
        projection.normalized_request.messages = vec![SanitizedMessage::User {
            content: SanitizedMessageContent::Text("route this request".to_string()),
            name: None,
        }];
        projection.semantic_request_fingerprint.clear();
        let mut value = serde_json::to_value(&projection).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("semantic_request_fingerprint");
        projection.semantic_request_fingerprint = hash_json(&value).unwrap();
        projection
    }

    fn routing_projection() -> RouterRoutingContextProjectionV1 {
        RouterRoutingContextProjectionV1 {
            schema: ROUTING_CONTEXT_SCHEMA_V1.to_string(),
            tenant_policy_hash: "3".repeat(64),
            agent_policy_hash: "4".repeat(64),
            position_features: BTreeMap::new(),
        }
    }

    fn response_projection() -> RouterResponseProjectionV1 {
        let mut projection = RouterResponseProjectionV1 {
            schema: RESPONSE_PROJECTION_SCHEMA_V1.to_string(),
            sanitizer_version: TRAJECTORY_SANITIZER_VERSION,
            id: None,
            model: Some("candidate-model".to_string()),
            message: None,
            tool_calls: None,
            finish_reason: None,
            usage: None,
            semantic_response_fingerprint: String::new(),
        };
        let mut value = serde_json::to_value(&projection).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("semantic_response_fingerprint");
        projection.semantic_response_fingerprint = hash_json(&value).unwrap();
        projection
    }

    fn response_projection_for_model(model: &str) -> RouterResponseProjectionV1 {
        let mut projection = response_projection();
        projection.model = Some(model.to_string());
        projection.semantic_response_fingerprint.clear();
        let mut value = serde_json::to_value(&projection).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("semantic_response_fingerprint");
        projection.semantic_response_fingerprint = hash_json(&value).unwrap();
        projection
    }

    #[test]
    fn persisted_uuid_text_must_use_canonical_lowercase_form() {
        let value = Uuid::now_v7();
        assert_eq!(parse_uuid_v7(&value.to_string()).unwrap(), value);
        assert!(parse_uuid_v7(&value.to_string().to_uppercase()).is_err());
    }

    #[test]
    fn terminal_constructor_rejects_a_noncanonical_vector_query() {
        let mut query_inputs = request_projection();
        query_inputs.semantic_request_fingerprint = "f".repeat(64);
        assert!(
            ShadowTerminalRecord::new(
                Uuid::now_v7(),
                Uuid::now_v7(),
                Uuid::now_v7(),
                Uuid::now_v7(),
                ShadowTerminalClass::OperationalFailure,
                None,
                None,
                None,
                Some(ShadowOperationalFailureClass::new("router.provider.timeout").unwrap()),
                Some(1),
                None,
                None,
                ShadowVectorSourceV1::Canonicalizable {
                    query_inputs: Box::new(query_inputs),
                },
                None,
                1,
            )
            .is_err()
        );
    }

    #[test]
    fn operational_failure_classes_are_closed_and_exhaustive() {
        for value in [
            "router.provider.transport",
            "router.provider.authentication",
            "router.provider.rate_limited",
            "router.provider.timeout",
            "router.provider.canceled",
            "router.provider.unreadable_response",
            "router.provider.truncated_response",
            "router.provider.ambiguous_decode",
            "router.provider.evidence_bound_exceeded",
            "router.provider.middleware_interference",
            "router.judge.output_invalid",
        ] {
            assert_eq!(
                ShadowOperationalFailureClass::new(value).unwrap().as_str(),
                value
            );
        }
        assert!(ShadowOperationalFailureClass::new("router.provider.invented").is_err());
    }

    fn seed_deterministic_evaluation(
        repository: &LedgerRepository,
        attempt: &ReservedShadowAttempt,
        failure: DeterministicHardFailureV1,
    ) -> Uuid {
        let evaluation_id = Uuid::now_v7();
        let (failure, ordinal) = match failure {
            DeterministicHardFailureV1::ToolContract => ("tool_contract", 0),
            DeterministicHardFailureV1::ResponseSchema => ("response_schema", 1),
            DeterministicHardFailureV1::MalformedCandidate => ("malformed_candidate", 3),
        };
        let evaluation_hash = hash_json(&json!({
            "evaluation_id": evaluation_id,
            "shadow_attempt_id": attempt.shadow_attempt_id,
            "evaluator_version": attempt.evaluator_version,
            "source": "deterministic_validator",
            "judge_model": Json::Null,
            "judge_model_revision": Json::Null,
            "prompt_version": Json::Null,
            "prompt_sha256": Json::Null,
            "rubric_version": Json::Null,
            "rubric_sha256": Json::Null,
            "output_schema_version": Json::Null,
            "output_schema_sha256": Json::Null,
            "response_equivalence": Json::Null,
            "response_equivalence_bits": Json::Null,
            "trajectory_equivalence": Json::Null,
            "trajectory_equivalence_bits": Json::Null,
            "judge_confidence": Json::Null,
            "judge_confidence_bits": Json::Null,
            "response_weight": Json::Null,
            "response_weight_bits": Json::Null,
            "trajectory_weight": Json::Null,
            "trajectory_weight_bits": Json::Null,
            "aggregate_score": Json::Null,
            "aggregate_score_bits": Json::Null,
            "label": "fail",
            "binary_label": "fail",
            "hard_failures": [failure],
            "rationale": Json::Null,
            "is_partial": false,
            "promotion_eligible": true,
            "created_at_unix_ms": 25,
        }))
        .unwrap();
        repository
            .connection
            .execute(
                "INSERT INTO evaluations (
                    evaluation_id, shadow_attempt_id, evaluator_version, source,
                    label, binary_label, is_partial, promotion_eligible,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, 'deterministic_validator', 'fail', 'fail',
                           0, 1, 25, ?4)",
                params![
                    evaluation_id.to_string(),
                    attempt.shadow_attempt_id.to_string(),
                    attempt.evaluator_version,
                    evaluation_hash,
                ],
            )
            .unwrap();
        let failure_hash = hash_json(&json!({
            "evaluation_id": evaluation_id,
            "hard_failure": failure,
            "canonical_ordinal": ordinal,
        }))
        .unwrap();
        repository
            .connection
            .execute(
                "INSERT INTO evaluation_hard_failures (
                    evaluation_id, hard_failure, canonical_ordinal,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4)",
                params![evaluation_id.to_string(), failure, ordinal, failure_hash],
            )
            .unwrap();
        evaluation_id
    }

    fn seed_judge_evaluation_with_negative_zero(
        repository: &LedgerRepository,
        attempt: &ReservedShadowAttempt,
    ) -> Uuid {
        let evaluation_id = Uuid::now_v7();
        let scores = [-0.0_f64, 0.8, 0.9, 0.5, 0.5, 0.4];
        let bits = scores.map(|value| value.to_bits() as i64);
        let evaluation_hash = hash_json(&json!({
            "evaluation_id": evaluation_id,
            "shadow_attempt_id": attempt.shadow_attempt_id,
            "evaluator_version": attempt.evaluator_version,
            "source": "judge",
            "judge_model": "judge-model",
            "judge_model_revision": "judge-revision",
            "prompt_version": "prompt-v1",
            "prompt_sha256": "a".repeat(64),
            "rubric_version": "rubric-v1",
            "rubric_sha256": "b".repeat(64),
            "output_schema_version": 1,
            "output_schema_sha256": "c".repeat(64),
            "response_equivalence": scores[0],
            "response_equivalence_bits": format!("{:016x}", bits[0] as u64),
            "trajectory_equivalence": scores[1],
            "trajectory_equivalence_bits": format!("{:016x}", bits[1] as u64),
            "judge_confidence": scores[2],
            "judge_confidence_bits": format!("{:016x}", bits[2] as u64),
            "response_weight": scores[3],
            "response_weight_bits": format!("{:016x}", bits[3] as u64),
            "trajectory_weight": scores[4],
            "trajectory_weight_bits": format!("{:016x}", bits[4] as u64),
            "aggregate_score": scores[5],
            "aggregate_score_bits": format!("{:016x}", bits[5] as u64),
            "label": "fail",
            "binary_label": "fail",
            "hard_failures": Vec::<String>::new(),
            "rationale": "signed zero evaluation",
            "is_partial": false,
            "promotion_eligible": true,
            "created_at_unix_ms": 25,
        }))
        .unwrap();
        repository
            .connection
            .execute(
                "INSERT INTO evaluations (
                    evaluation_id, shadow_attempt_id, evaluator_version, source,
                    judge_model, judge_model_revision, prompt_version, prompt_sha256,
                    rubric_version, rubric_sha256, output_schema_version,
                    output_schema_sha256, response_equivalence,
                    response_equivalence_bits, trajectory_equivalence,
                    trajectory_equivalence_bits, judge_confidence,
                    judge_confidence_bits, response_weight, response_weight_bits,
                    trajectory_weight, trajectory_weight_bits, aggregate_score,
                    aggregate_score_bits, label, binary_label, rationale, is_partial,
                    promotion_eligible, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, ?2, ?3, 'judge', 'judge-model', 'judge-revision',
                    'prompt-v1', ?4, 'rubric-v1', ?5, 1, ?6,
                    ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16,
                    ?17, ?18, 'fail', 'fail', 'signed zero evaluation', 0, 1, 25, ?19
                 )",
                params![
                    evaluation_id.to_string(),
                    attempt.shadow_attempt_id.to_string(),
                    attempt.evaluator_version,
                    "a".repeat(64),
                    "b".repeat(64),
                    "c".repeat(64),
                    scores[0],
                    bits[0],
                    scores[1],
                    bits[1],
                    scores[2],
                    bits[2],
                    scores[3],
                    bits[3],
                    scores[4],
                    bits[4],
                    scores[5],
                    bits[5],
                    evaluation_hash,
                ],
            )
            .unwrap();
        evaluation_id
    }

    fn seed_pending_anchor(
        repository: &mut LedgerRepository,
        identity: &LedgerRuntimeIdentity,
        attempts: &[ReservedShadowAttempt],
    ) -> PendingTrajectoryWindow {
        let anchor_id = Uuid::now_v7();
        let pool = identity.pools.get("pool-a").unwrap();
        let replay_capability_facts =
            ReplayCapabilityFactsV1::from_capability(&LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: LlmApiFamily::OpenAIChatCompletions,
                transport_identity: "transport-shared".to_string(),
            })
            .unwrap();
        let mut candidate_facts = attempts
            .iter()
            .map(|attempt| PersistedCandidateFactV1 {
                schema: CANDIDATE_FACT_SCHEMA_V1.to_string(),
                candidate_id: attempt.candidate_id.clone(),
                model: attempt.candidate_model.clone(),
                model_revision: attempt.candidate_model_revision.clone(),
                cost_rank: attempt.cost_rank,
                capabilities: PersistedCandidateCapabilitiesV1 {
                    tools: true,
                    multimodal_input: false,
                    structured_output: false,
                    reasoning_controls: false,
                },
                decoding_fingerprint: attempt.decoding_fingerprint.clone(),
            })
            .collect::<Vec<_>>();
        candidate_facts.sort_by(|left, right| {
            (left.cost_rank, left.candidate_id.as_str())
                .cmp(&(right.cost_rank, right.candidate_id.as_str()))
        });
        let mut anchor_response = response_projection();
        anchor_response.model = Some("anchor-a".to_string());
        anchor_response.semantic_response_fingerprint.clear();
        let mut response_value = serde_json::to_value(&anchor_response).unwrap();
        response_value
            .as_object_mut()
            .unwrap()
            .remove("semantic_response_fingerprint");
        anchor_response.semantic_response_fingerprint = hash_json(&response_value).unwrap();
        let mut anchor_request = attempts
            .first()
            .map(|attempt| attempt.request_projection.clone())
            .unwrap_or_else(request_projection);
        anchor_request.normalized_request.model = Some("anchor-a".to_string());
        anchor_request.semantic_request_fingerprint.clear();
        let mut anchor_request_value = serde_json::to_value(&anchor_request).unwrap();
        anchor_request_value
            .as_object_mut()
            .unwrap()
            .remove("semantic_request_fingerprint");
        anchor_request.semantic_request_fingerprint = hash_json(&anchor_request_value).unwrap();
        let pending = PendingTrajectoryWindow {
            schema: PENDING_TRAJECTORY_SCHEMA_V1.to_string(),
            anchor_id,
            anchor_call_uuid: Uuid::now_v7(),
            root_uuid: Uuid::now_v7(),
            owner_uuid: Uuid::now_v7(),
            owner_path: Vec::new(),
            pool_id: "pool-a".to_string(),
            anchor_model_revision: "2026-07-01".to_string(),
            process_instance_id: identity.process_instance_id,
            project_uuid: identity.project_uuid,
            project_id: identity.project_id.clone(),
            config_generation_id: identity.config_generation_id.clone(),
            policy_version_id: pool.policy_version_id.clone(),
            learning_generation_id: pool.learning_generation_id,
            request_projection: anchor_request,
            routing_context_projection: routing_projection(),
            normalized_anchor_response: anchor_response,
            replay_capability_facts,
            candidate_facts,
            requested_progress: 1,
            opened_at: Utc.timestamp_millis_opt(0).single().unwrap(),
            deadline_at: Utc.timestamp_millis_opt(10).single().unwrap(),
        };
        let frozen_pending =
            FrozenPendingAnchorV1::new(&pending, Uuid::now_v7(), Uuid::now_v7(), 0).unwrap();
        assert!(matches!(
            repository.record_pending_anchor(&frozen_pending).unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
        pending
    }

    fn seed_closed_anchor(
        repository: &mut LedgerRepository,
        identity: &LedgerRuntimeIdentity,
        attempts: &[ReservedShadowAttempt],
    ) -> Uuid {
        let pending = seed_pending_anchor(repository, identity, attempts);
        let anchor_id = pending.anchor_id;
        let terminal = PersistedTrajectoryTerminalV1::closed(
            pending,
            Vec::new(),
            1,
            TrajectoryTrigger::ProgressReached,
            Utc.timestamp_millis_opt(10).single().unwrap(),
            Vec::new(),
        );
        let frozen_terminal =
            FrozenTerminalAnchorV1::new(&terminal, Uuid::now_v7(), Uuid::now_v7(), 10).unwrap();
        assert!(matches!(
            repository.record_terminal_anchor(&frozen_terminal).unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
        anchor_id
    }

    fn attempt(suffix: &str, created_at_unix_ms: i64) -> ReservedShadowAttempt {
        let candidate_model = format!("candidate-model-{suffix}");
        let candidate_request =
            candidate_request_projection(&request_projection(), &candidate_model).unwrap();
        ReservedShadowAttempt::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            format!("candidate-{suffix}"),
            candidate_model,
            format!("candidate-revision-{suffix}"),
            0,
            LlmApiFamily::OpenAIChatCompletions,
            "transport-shared",
            "anchor-a",
            "2026-07-01",
            "1".repeat(64),
            policy_evaluator_version(),
            "3".repeat(64),
            "4".repeat(64),
            true,
            candidate_request,
            created_at_unix_ms,
        )
        .unwrap()
    }

    fn canonicalizable_attempt(suffix: &str, created_at_unix_ms: i64) -> ReservedShadowAttempt {
        let candidate_model = format!("candidate-model-{suffix}");
        let candidate_request =
            candidate_request_projection(&request_projection_with_task(), &candidate_model)
                .unwrap();
        ReservedShadowAttempt::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            format!("candidate-{suffix}"),
            candidate_model,
            format!("candidate-revision-{suffix}"),
            0,
            LlmApiFamily::OpenAIChatCompletions,
            "transport-shared",
            "anchor-a",
            "2026-07-01",
            "1".repeat(64),
            policy_evaluator_version(),
            "3".repeat(64),
            "4".repeat(64),
            true,
            candidate_request,
            created_at_unix_ms,
        )
        .unwrap()
    }

    fn atomic_vectorization() -> ShadowVectorizationHandoff {
        ShadowVectorizationHandoff::Atomic(
            AtomicShadowVectorization::new(
                routing_projection(),
                Uuid::now_v7(),
                Uuid::now_v7(),
                Uuid::now_v7(),
                Uuid::now_v7(),
            )
            .unwrap(),
        )
    }

    fn reservation(
        identity: &LedgerRuntimeIdentity,
        anchor_id: Uuid,
        mut attempts: Vec<ReservedShadowAttempt>,
        created_at_unix_ms: i64,
    ) -> SampleBatchReservation {
        let pool = identity.pools.get("pool-a").unwrap();
        attempts.sort_by(|left, right| {
            (left.cost_rank, left.candidate_id.as_str())
                .cmp(&(right.cost_rank, right.candidate_id.as_str()))
        });
        SampleBatchReservation::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            anchor_id,
            identity.config_generation_id.clone(),
            pool.policy_version_id.clone(),
            pool.learning_generation_id,
            "pool-a",
            attempts,
            created_at_unix_ms,
        )
        .unwrap()
    }

    fn operational_terminal(
        reservation: &SampleBatchReservation,
        attempt: &ReservedShadowAttempt,
        close_batch: bool,
        created_at_unix_ms: i64,
    ) -> ShadowTerminalRecord {
        let batch_terminal = close_batch.then(|| {
            SampleBatchTerminalEvent::new(
                reservation.sample_batch_id,
                Uuid::now_v7(),
                SampleBatchTerminalState::Closed,
                None,
                created_at_unix_ms,
            )
            .unwrap()
        });
        ShadowTerminalRecord::new(
            Uuid::now_v7(),
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            ShadowTerminalClass::OperationalFailure,
            None,
            None,
            None,
            Some(ShadowOperationalFailureClass::new("router.provider.timeout").unwrap()),
            Some(12),
            None,
            None,
            ShadowVectorSourceV1::Canonicalizable {
                query_inputs: Box::new(attempt.request_projection.clone()),
            },
            batch_terminal,
            created_at_unix_ms,
        )
        .unwrap()
    }

    fn seed_started_single_attempt(
        activated: &mut super::super::ActivatedLedger,
        attempt: &ReservedShadowAttempt,
    ) -> SampleBatchReservation {
        let identity = activated.identity.clone();
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            std::slice::from_ref(attempt),
        );
        let reservation = reservation(&identity, anchor_id, vec![attempt.clone()], 20);
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .start_shadow_attempt(
                    ShadowAttemptStarted::new(
                        attempt.shadow_attempt_id,
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                        21,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ShadowCommandAck::Applied
        );
        reservation
    }

    fn health_count(repository: &LedgerRepository) -> i64 {
        repository
            .connection
            .query_row(
                "SELECT count(*) FROM health_events
                 WHERE stable_class = 'router.ledger.integrity_conflict'",
                [],
                |row| row.get(0),
            )
            .unwrap()
    }

    #[test]
    fn batch_and_all_reserved_attempts_are_atomic_exact_and_conflict_safe() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "shadow-batch");
        let identity = activated.identity.clone();
        let first = attempt("first", 20);
        let second = attempt("second", 20);
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            &[first.clone(), second.clone()],
        );
        let batch_reservation = reservation(
            &identity,
            anchor_id,
            vec![first.clone(), second.clone()],
            20,
        );

        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&batch_reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&batch_reservation)
                .unwrap(),
            ShadowCommandAck::AlreadyApplied
        );
        let counts: (i64, i64, i64, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT count(*) FROM sample_batches),
                    (SELECT count(*) FROM sample_batch_state_events WHERE state = 'open'),
                    (SELECT count(*) FROM shadow_attempts),
                    (SELECT count(*) FROM shadow_attempt_state_events WHERE state = 'reserved')",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(counts, (1, 1, 2, 2));

        let mut conflicting = batch_reservation.clone();
        conflicting.attempts[0].candidate_model = "different-model".to_string();
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&conflicting)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(health_count(&activated.repository), 1);
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM shadow_attempts", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            2
        );

        let alternate = reservation(&identity, anchor_id, vec![attempt("alternate", 21)], 21);
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&alternate)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(health_count(&activated.repository), 2);

        let missing_anchor = reservation(
            &identity,
            Uuid::now_v7(),
            vec![attempt("missing-anchor", 22)],
            22,
        );
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&missing_anchor)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(health_count(&activated.repository), 3);
    }

    #[test]
    fn started_state_is_append_only_idempotent_and_full_field_checked() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "shadow-start");
        let identity = activated.identity.clone();
        let attempt = attempt("start", 20);
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            std::slice::from_ref(&attempt),
        );
        let reservation = reservation(&identity, anchor_id, vec![attempt.clone()], 20);
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        let started = ShadowAttemptStarted::new(
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            21,
        )
        .unwrap();
        assert_eq!(
            activated.repository.start_shadow_attempt(started).unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            activated.repository.start_shadow_attempt(started).unwrap(),
            ShadowCommandAck::AlreadyApplied
        );
        let conflicting = ShadowAttemptStarted::new(
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            22,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .start_shadow_attempt(conflicting)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(health_count(&activated.repository), 1);

        let terminal = operational_terminal(&reservation, &attempt, true, 30);
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            activated.repository.start_shadow_attempt(started).unwrap(),
            ShadowCommandAck::AlreadyApplied
        );
        assert_eq!(health_count(&activated.repository), 1);
    }

    #[test]
    fn start_and_terminal_reject_noncanonical_reserved_state() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "shadow-reserved-state-integrity");
        let identity = activated.identity.clone();
        let start_attempt = attempt("corrupt-start", 20);
        let terminal_attempt = attempt("corrupt-terminal", 20);
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            &[start_attempt.clone(), terminal_attempt.clone()],
        );
        let reservation = reservation(
            &identity,
            anchor_id,
            vec![start_attempt.clone(), terminal_attempt.clone()],
            20,
        );
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert!(
            verified_shadow_attempt_context(
                &activated.repository.connection,
                start_attempt.shadow_attempt_id,
            )
            .unwrap()
            .is_none()
        );
        let started = ShadowAttemptStarted::new(
            start_attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            21,
        )
        .unwrap();
        assert_eq!(
            activated.repository.start_shadow_attempt(started).unwrap(),
            ShadowCommandAck::Applied
        );
        let verified = verified_shadow_attempt_context(
            &activated.repository.connection,
            start_attempt.shadow_attempt_id,
        )
        .unwrap()
        .unwrap();
        assert_eq!(verified.anchor_id, anchor_id);
        assert_eq!(verified.process_instance_id, identity.process_instance_id);
        assert_eq!(
            verified.learning_generation_id,
            identity.pools["pool-a"].learning_generation_id
        );
        assert_eq!(verified.evaluator_version, start_attempt.evaluator_version);

        let terminal_started = ShadowAttemptStarted::new(
            terminal_attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            21,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .start_shadow_attempt(terminal_started)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        activated
            .repository
            .connection
            .execute(
                "UPDATE shadow_attempt_state_events
                 SET canonical_payload_hash = ?1
                 WHERE shadow_attempt_id = ?2 AND state = 'started'",
                params![
                    "e".repeat(64),
                    terminal_attempt.shadow_attempt_id.to_string()
                ],
            )
            .unwrap();
        assert!(
            verified_shadow_attempt_context(
                &activated.repository.connection,
                terminal_attempt.shadow_attempt_id,
            )
            .unwrap()
            .is_none()
        );

        activated
            .repository
            .connection
            .execute(
                "UPDATE shadow_attempt_state_events
                 SET created_at_unix_ms = created_at_unix_ms + 1
                 WHERE shadow_attempt_id = ?1 AND state = 'reserved'",
                [start_attempt.shadow_attempt_id.to_string()],
            )
            .unwrap();
        assert_eq!(
            activated.repository.start_shadow_attempt(started).unwrap(),
            ShadowCommandAck::Conflict
        );

        activated
            .repository
            .connection
            .execute(
                "UPDATE shadow_attempt_state_events
                 SET canonical_payload_hash = ?1
                 WHERE shadow_attempt_id = ?2 AND state = 'reserved'",
                params![
                    "f".repeat(64),
                    terminal_attempt.shadow_attempt_id.to_string()
                ],
            )
            .unwrap();
        let terminal = operational_terminal(&reservation, &terminal_attempt, false, 22);
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        let writes: (i64, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT count(*) FROM shadow_attempt_state_events WHERE state = 'started'),
                    (SELECT count(*) FROM shadow_results)",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(writes, (2, 0));
        assert_eq!(health_count(&activated.repository), 2);
    }

    #[test]
    fn terminal_result_vector_source_state_and_batch_close_are_one_exact_transaction() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "shadow-terminal");
        let identity = activated.identity.clone();
        let attempt = attempt("terminal", 20);
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            std::slice::from_ref(&attempt),
        );
        let reservation = reservation(&identity, anchor_id, vec![attempt.clone()], 20);
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        let terminal = operational_terminal(&reservation, &attempt, true, 30);
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(
            activated
                .repository
                .start_shadow_attempt(
                    ShadowAttemptStarted::new(
                        attempt.shadow_attempt_id,
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                        21,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ShadowCommandAck::Applied
        );
        let mut wrong_query = terminal.clone();
        let ShadowVectorSourceV1::Canonicalizable { query_inputs } = &mut wrong_query.vector_source
        else {
            unreachable!();
        };
        query_inputs.semantic_request_fingerprint = "f".repeat(64);
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&wrong_query)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM shadow_results", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::AlreadyApplied
        );
        let persisted: (String, i64, Option<String>, Option<String>, String) = activated
            .repository
            .connection
            .query_row(
                "SELECT terminal_class, canonicalizable, query_inputs_json,
                        vector_source_hash, partition_inputs_json
                 FROM shadow_results WHERE shadow_attempt_id = ?1",
                params![attempt.shadow_attempt_id.to_string()],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(persisted.0, "operational_failure");
        assert_eq!(persisted.1, 1);
        assert!(persisted.2.is_some());
        assert!(persisted.3.is_some());
        assert!(persisted.4.contains("candidate-terminal"));
        let states: (i64, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT count(*) FROM shadow_attempt_state_events
                     WHERE state = 'operational_failure'),
                    (SELECT count(*) FROM sample_batch_state_events WHERE state = 'closed')",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(states, (1, 1));
        let vector_rows: i64 = activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT count(*) FROM vectorization_outcomes)
                  + (SELECT count(*) FROM evidence_vector_links)
                  + (SELECT count(*) FROM vector_materialization_jobs)",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(vector_rows, 0);

        let mut conflicting = terminal.clone();
        conflicting.latency_ms = Some(13);
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&conflicting)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(health_count(&activated.repository), 1);

        let mut omitted_close = terminal.clone();
        omitted_close.batch_terminal = None;
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&omitted_close)
                .unwrap(),
            ShadowCommandAck::AlreadyApplied
        );
        assert_eq!(health_count(&activated.repository), 1);

        let stored_attempt =
            load_attempt_by_id(&activated.repository.connection, attempt.shadow_attempt_id)
                .unwrap()
                .unwrap();
        let mut forged_command = terminal.clone();
        forged_command.operational_failure_class = Some(ShadowOperationalFailureClass(
            "router.provider.invented".to_string(),
        ));
        let forged = prepare_result(
            &forged_command,
            &stored_attempt,
            identity.process_instance_id,
        )
        .unwrap();
        activated
            .repository
            .connection
            .execute(
                "UPDATE shadow_results
                 SET operational_failure_class = ?1, canonical_payload_hash = ?2
                 WHERE shadow_attempt_id = ?3",
                params![
                    "router.provider.invented",
                    forged.result_hash,
                    attempt.shadow_attempt_id.to_string(),
                ],
            )
            .unwrap();
        assert!(
            !canonical_terminal_attempt_matches(
                &activated.repository.connection,
                attempt.shadow_attempt_id,
            )
            .unwrap()
        );
    }

    #[test]
    fn atomic_terminal_creates_one_pending_vector_graph_and_retries_exactly() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate_learning(&path, "shadow-vector-miss");
        let attempt = canonicalizable_attempt("terminal", 20);
        let reservation = seed_started_single_attempt(&mut activated, &attempt);
        let terminal = operational_terminal(&reservation, &attempt, true, 30)
            .with_vectorization(atomic_vectorization());

        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::AlreadyApplied
        );
        let counts: (i64, i64, i64, i64, i64, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT count(*) FROM canonical_routing_queries),
                    (SELECT count(*) FROM vectorization_outcomes),
                    (SELECT count(*) FROM routing_partitions),
                    (SELECT count(*) FROM embedding_jobs),
                    (SELECT count(*) FROM evidence_vector_links),
                    (SELECT count(*) FROM vector_materialization_jobs)",
                [],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(counts, (1, 1, 1, 1, 1, 1));
        let states: (String, String) = activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT state FROM evidence_vector_link_state_events),
                    (SELECT state FROM vector_materialization_job_state_events)",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            states,
            ("pending_embedding".into(), "pending_embedding".into())
        );
    }

    #[test]
    fn atomic_terminal_records_canonicalization_rejection_without_durable_work() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate_learning(&path, "shadow-vector-rejection");
        let attempt = attempt("terminal", 20);
        let reservation = seed_started_single_attempt(&mut activated, &attempt);
        let terminal = operational_terminal(&reservation, &attempt, true, 30)
            .with_vectorization(atomic_vectorization());

        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::AlreadyApplied
        );
        let outcome: (String, String) = activated
            .repository
            .connection
            .query_row(
                "SELECT outcome, stable_reason FROM vectorization_outcomes",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(outcome.0, "noncanonicalizable");
        assert_eq!(outcome.1, "router.ineligible.canonical_missing_task");
        let work: i64 = activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT count(*) FROM canonical_routing_queries)
                  + (SELECT count(*) FROM embedding_jobs)
                  + (SELECT count(*) FROM evidence_vector_links)
                  + (SELECT count(*) FROM vector_materialization_jobs)",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(work, 0);
    }

    #[test]
    fn alternate_terminal_vector_ids_roll_back_without_partial_graph_changes() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate_learning(&path, "shadow-vector-id-conflict");
        let attempt = canonicalizable_attempt("terminal", 20);
        let reservation = seed_started_single_attempt(&mut activated, &attempt);
        let terminal = operational_terminal(&reservation, &attempt, true, 30)
            .with_vectorization(atomic_vectorization());
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        let conflicting = terminal.clone().with_vectorization(atomic_vectorization());
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&conflicting)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        let counts: (i64, i64, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT count(*) FROM evidence_vector_links),
                    (SELECT count(*) FROM evidence_vector_link_state_events),
                    (SELECT count(*) FROM vector_materialization_job_state_events)",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(counts, (1, 1, 1));
        assert_eq!(health_count(&activated.repository), 1);
    }

    #[test]
    fn terminal_evaluation_source_and_deterministic_failure_must_match() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "shadow-evaluation-binding");
        let identity = activated.identity.clone();
        let completed_attempt = attempt("swapped-source", 20);
        let deterministic_attempt = attempt("wrong-failure", 20);
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            &[completed_attempt.clone(), deterministic_attempt.clone()],
        );
        let reservation = reservation(
            &identity,
            anchor_id,
            vec![completed_attempt.clone(), deterministic_attempt.clone()],
            20,
        );
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        for (attempt, created_at_unix_ms) in [
            (&completed_attempt, 21_i64),
            (&deterministic_attempt, 22_i64),
        ] {
            assert_eq!(
                activated
                    .repository
                    .start_shadow_attempt(
                        ShadowAttemptStarted::new(
                            attempt.shadow_attempt_id,
                            Uuid::now_v7(),
                            Uuid::now_v7(),
                            created_at_unix_ms,
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                ShadowCommandAck::Applied
            );
        }

        let deterministic_evaluation = seed_deterministic_evaluation(
            &activated.repository,
            &completed_attempt,
            DeterministicHardFailureV1::ToolContract,
        );
        let completed = ShadowTerminalRecord::new(
            Uuid::now_v7(),
            completed_attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            ShadowTerminalClass::Completed,
            None,
            Some(response_projection()),
            None,
            None,
            Some(10),
            None,
            Some(deterministic_evaluation),
            ShadowVectorSourceV1::Canonicalizable {
                query_inputs: Box::new(completed_attempt.request_projection.clone()),
            },
            None,
            30,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&completed)
                .unwrap(),
            ShadowCommandAck::Conflict
        );

        let wrong_failure_evaluation = seed_deterministic_evaluation(
            &activated.repository,
            &deterministic_attempt,
            DeterministicHardFailureV1::ResponseSchema,
        );
        let wrong_failure = ShadowTerminalRecord::new(
            Uuid::now_v7(),
            deterministic_attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            ShadowTerminalClass::DeterministicFailure,
            None,
            None,
            Some(DeterministicHardFailureV1::ToolContract),
            None,
            Some(10),
            None,
            Some(wrong_failure_evaluation),
            ShadowVectorSourceV1::Canonicalizable {
                query_inputs: Box::new(deterministic_attempt.request_projection.clone()),
            },
            None,
            31,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&wrong_failure)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM shadow_results", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
        assert_eq!(health_count(&activated.repository), 2);
    }

    #[test]
    fn completed_result_must_match_the_final_judge_candidate_response() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let router_config = shadow_config(&path, "shadow-judge-response-binding");
        let mut activated = LedgerRepository::activate(&router_config).unwrap();
        let identity = activated.identity.clone();
        let attempt = attempt("judge-binding", 20);
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            std::slice::from_ref(&attempt),
        );
        let reservation = reservation(&identity, anchor_id, vec![attempt.clone()], 20);
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        let shadow_start = ShadowAttemptStarted::new(
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            21,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .start_shadow_attempt(shadow_start)
                .unwrap(),
            ShadowCommandAck::Applied
        );

        let judge = &router_config.pools[0].judge;
        let candidate_response = response_projection_for_model(&attempt.candidate_model);
        let judge_input = PairwiseJudgeInputV1::new(
            &request_projection(),
            &response_projection_for_model("anchor-a"),
            &candidate_response,
            &[],
            JudgeHorizonV1::new(1, 1, TrajectoryTrigger::ProgressReached, false).unwrap(),
            JudgePolicyIdentityV1::from_config(judge).unwrap(),
        )
        .unwrap();
        let judge_start = JudgeAttemptStart::new(
            Uuid::now_v7(),
            attempt.shadow_attempt_id,
            identity.pools["pool-a"].learning_generation_id,
            attempt.evaluator_version.clone(),
            judge,
            &judge_input,
            0,
            Uuid::now_v7(),
            Uuid::now_v7(),
            22,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&judge_start)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        let evaluation_id = Uuid::now_v7();
        let judge_terminal = JudgeAttemptTerminal::valid(
            judge_start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            attempt.shadow_attempt_id,
            attempt.evaluator_version.clone(),
            &json!({
                "response_equivalence": 0.9,
                "trajectory_equivalence": 0.9,
                "judge_confidence": 0.9,
                "hard_failures": [],
                "rationale": "bounded rationale",
            })
            .to_string(),
            judge,
            evaluation_id,
            Uuid::now_v7(),
            false,
            23,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&judge_terminal)
                .unwrap(),
            JudgeRecordAck::Applied
        );

        let completed = |response, created_at_unix_ms| {
            ShadowTerminalRecord::new(
                Uuid::now_v7(),
                attempt.shadow_attempt_id,
                Uuid::now_v7(),
                Uuid::now_v7(),
                ShadowTerminalClass::Completed,
                None,
                Some(response),
                None,
                None,
                Some(10),
                None,
                Some(evaluation_id),
                ShadowVectorSourceV1::Canonicalizable {
                    query_inputs: Box::new(attempt.request_projection.clone()),
                },
                Some(
                    SampleBatchTerminalEvent::new(
                        reservation.sample_batch_id,
                        Uuid::now_v7(),
                        SampleBatchTerminalState::Closed,
                        None,
                        created_at_unix_ms,
                    )
                    .unwrap(),
                ),
                created_at_unix_ms,
            )
            .unwrap()
        };
        let canonical_started_hash: String = activated
            .repository
            .connection
            .query_row(
                "SELECT canonical_payload_hash FROM shadow_attempt_state_events
                 WHERE shadow_attempt_state_event_id = ?1",
                [shadow_start.state_event_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        activated
            .repository
            .connection
            .execute(
                "UPDATE shadow_attempt_state_events SET canonical_payload_hash = ?1
                 WHERE shadow_attempt_state_event_id = ?2",
                params!["f".repeat(64), shadow_start.state_event_id.to_string()],
            )
            .unwrap();
        let corrupt_started = completed(candidate_response.clone(), 24);
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&corrupt_started)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        activated
            .repository
            .connection
            .execute(
                "UPDATE shadow_attempt_state_events SET canonical_payload_hash = ?1
                 WHERE shadow_attempt_state_event_id = ?2",
                params![
                    canonical_started_hash,
                    shadow_start.state_event_id.to_string()
                ],
            )
            .unwrap();

        let substituted = completed(response_projection_for_model("candidate-model-b"), 25);
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&substituted)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        let exact = completed(candidate_response, 26);
        assert_eq!(
            activated.repository.record_shadow_terminal(&exact).unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            activated.repository.record_shadow_terminal(&exact).unwrap(),
            ShadowCommandAck::AlreadyApplied
        );
    }

    #[test]
    fn final_result_requires_matching_batch_terminal_on_first_write() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "shadow-required-batch-close");
        let identity = activated.identity.clone();
        let attempt = attempt("required-close", 20);
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            std::slice::from_ref(&attempt),
        );
        let reservation = reservation(&identity, anchor_id, vec![attempt.clone()], 20);
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .start_shadow_attempt(
                    ShadowAttemptStarted::new(
                        attempt.shadow_attempt_id,
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                        21,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ShadowCommandAck::Applied
        );

        let omitted = operational_terminal(&reservation, &attempt, false, 30);
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&omitted)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        let mut wrong_state = operational_terminal(&reservation, &attempt, true, 31);
        wrong_state.batch_terminal.as_mut().unwrap().state =
            SampleBatchTerminalState::CanceledShutdown;
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&wrong_state)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        let mut wrong_timestamp = operational_terminal(&reservation, &attempt, true, 32);
        wrong_timestamp
            .batch_terminal
            .as_mut()
            .unwrap()
            .created_at_unix_ms = 33;
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&wrong_timestamp)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM shadow_results", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );

        let valid = operational_terminal(&reservation, &attempt, true, 34);
        assert_eq!(
            activated.repository.record_shadow_terminal(&valid).unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(health_count(&activated.repository), 3);
    }

    #[test]
    fn orphan_batch_terminal_requires_matching_class_and_dead_process() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut origin = activate(&path, "shadow-orphan-binding");
        let origin_identity = origin.identity.clone();
        let attempt = attempt("orphan-binding", 20);
        let anchor_id = seed_closed_anchor(
            &mut origin.repository,
            &origin_identity,
            std::slice::from_ref(&attempt),
        );
        let reservation = reservation(&origin_identity, anchor_id, vec![attempt.clone()], 20);
        assert_eq!(
            origin
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        let mut reconciler = activate(&path, "shadow-orphan-binding");
        assert_eq!(
            origin
                .repository
                .stop_process(ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), 25).unwrap())
                .unwrap(),
            super::super::process::ProcessCommandAck::Applied
        );
        let batch_terminal = SampleBatchTerminalEvent::new(
            reservation.sample_batch_id,
            Uuid::now_v7(),
            SampleBatchTerminalState::OrphanedInFlight,
            Some(origin_identity.process_instance_id),
            30,
        )
        .unwrap();
        let terminal = ShadowTerminalRecord::new(
            Uuid::now_v7(),
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            ShadowTerminalClass::OrphanedInFlight,
            Some(origin_identity.process_instance_id),
            None,
            None,
            None,
            None,
            None,
            None,
            ShadowVectorSourceV1::Canonicalizable {
                query_inputs: Box::new(attempt.request_projection.clone()),
            },
            Some(batch_terminal),
            30,
        )
        .unwrap();

        let mut wrong_class = terminal.clone();
        wrong_class.batch_terminal.as_mut().unwrap().state =
            SampleBatchTerminalState::OrphanedBeforeSchedule;
        assert_eq!(
            reconciler
                .repository
                .record_shadow_terminal(&wrong_class)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        let mut wrong_process = terminal.clone();
        wrong_process
            .batch_terminal
            .as_mut()
            .unwrap()
            .dead_process_instance_id = Some(Uuid::now_v7());
        assert_eq!(
            reconciler
                .repository
                .record_shadow_terminal(&wrong_process)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(
            reconciler
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Applied
        );
    }

    #[test]
    fn startup_reconstructs_closed_undelivered_batch_without_replaying_work() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = shadow_config(&path, "shadow-reconstruct-undelivered");
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let attempts = vec![attempt("first", 20), attempt("second", 20)];
        let anchor_id = seed_closed_anchor(&mut origin.repository, &origin_identity, &attempts);
        drop(origin);

        let mut recovered = LedgerRepository::activate_at(&config, 31_000).unwrap();
        assert_eq!(
            recovered
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM sample_batches WHERE anchor_id = ?1",
                    params![anchor_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        assert_eq!(
            recovered
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM shadow_results AS r
                     JOIN shadow_attempts AS a ON a.shadow_attempt_id = r.shadow_attempt_id
                     WHERE a.anchor_id = ?1 AND r.terminal_class = 'orphaned_before_schedule'",
                    params![anchor_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            2
        );
        assert_eq!(
            recovered
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM judge_attempts AS j
                     JOIN shadow_attempts AS a ON a.shadow_attempt_id = j.shadow_attempt_id
                     WHERE a.anchor_id = ?1",
                    params![anchor_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_reconciliation_noop_twice(&mut recovered.repository, 31_000);
    }

    #[test]
    fn startup_reconstructs_from_historical_policy_and_learning_identity() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let historical_config = shadow_config(&path, "shadow-historical-reconstruction");
        let mut origin = LedgerRepository::activate_at(&historical_config, 1_000).unwrap();
        let historical_identity = origin.identity.clone();
        let historical_pool = historical_identity.pools["pool-a"].clone();
        let anchor_id = seed_closed_anchor(
            &mut origin.repository,
            &historical_identity,
            &[attempt("first", 20)],
        );
        let new_learning = origin
            .repository
            .reset_pool("pool-a", "test-rotation", "historical reconstruction")
            .unwrap();
        assert_ne!(new_learning, historical_pool.learning_generation_id);

        let mut current_config = historical_config.clone();
        current_config.pools[0].judge.pass_threshold = 0.9;
        let current = LedgerRepository::activate_at(&current_config, 1_001).unwrap();
        assert_ne!(
            current.identity.pools["pool-a"].policy_version_id,
            historical_pool.policy_version_id
        );
        drop(current);
        drop(origin);

        let mut recovered = LedgerRepository::activate_at(&current_config, 31_001).unwrap();
        let stored: (
            String,
            String,
            String,
            String,
            String,
            String,
            String,
            String,
        ) = recovered
            .repository
            .connection
            .query_row(
                "SELECT config_generation_id, process_instance_id,
                        policy_version_id, learning_generation_id,
                        evaluator_version, candidate_id, candidate_model,
                        candidate_model_revision
                 FROM shadow_attempts WHERE anchor_id = ?1",
                params![anchor_id.to_string()],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                        row.get(6)?,
                        row.get(7)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(stored.0, historical_identity.config_generation_id);
        assert_eq!(
            stored.1,
            historical_identity.process_instance_id.to_string()
        );
        assert_eq!(stored.2, historical_pool.policy_version_id);
        assert_eq!(stored.3, historical_pool.learning_generation_id.to_string());
        assert_eq!(
            stored.4,
            historical_config.pools[0]
                .judge
                .evaluator_version()
                .unwrap()
        );
        assert_eq!(stored.5, "candidate-first");
        assert_eq!(stored.6, "candidate-model-first");
        assert_eq!(stored.7, "candidate-revision-first");

        let result: (
            String,
            i64,
            Option<String>,
            String,
            Option<String>,
            Option<String>,
        ) = recovered
            .repository
            .connection
            .query_row(
                "SELECT r.terminal_class, r.canonicalizable, r.query_inputs_json,
                        r.partition_inputs_json, r.vector_source_hash, r.evaluation_id
                 FROM shadow_results AS r
                 JOIN shadow_attempts AS a ON a.shadow_attempt_id = r.shadow_attempt_id
                 WHERE a.anchor_id = ?1",
                params![anchor_id.to_string()],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(result.0, "orphaned_before_schedule");
        assert_eq!(result.1, 1);
        let query_inputs = result.2.unwrap();
        let query_value: Json = serde_json::from_str(&query_inputs).unwrap();
        assert_eq!(canonical_json(&query_value).unwrap(), query_inputs);
        let partition_value: Json = serde_json::from_str(&result.3).unwrap();
        assert_eq!(canonical_json(&partition_value).unwrap(), result.3);
        assert!(result.4.as_deref().is_some_and(|hash| hash.len() == 64));
        assert_eq!(result.5, None);

        let attempt_state: (String, String) = recovered
            .repository
            .connection
            .query_row(
                "SELECT s.process_instance_id, s.dead_process_instance_id
             FROM shadow_attempt_state_events AS s
             JOIN shadow_attempts AS a ON a.shadow_attempt_id = s.shadow_attempt_id
             WHERE a.anchor_id = ?1 AND s.state = 'orphaned_before_schedule'",
                params![anchor_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(
            attempt_state.0,
            recovered.identity.process_instance_id.to_string()
        );
        assert_eq!(
            attempt_state.1,
            historical_identity.process_instance_id.to_string()
        );
        let batch_state: (String, String, String) = recovered
            .repository
            .connection
            .query_row(
                "SELECT s.state, s.process_instance_id, s.dead_process_instance_id
             FROM sample_batch_state_events AS s
             JOIN sample_batches AS b ON b.sample_batch_id = s.sample_batch_id
             WHERE b.anchor_id = ?1 AND s.state <> 'open'",
                params![anchor_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(batch_state.0, "orphaned_before_schedule");
        assert_eq!(
            batch_state.1,
            recovered.identity.process_instance_id.to_string()
        );
        assert_eq!(
            batch_state.2,
            historical_identity.process_instance_id.to_string()
        );
        let downstream: (i64, i64) = recovered
            .repository
            .connection
            .query_row(
                "SELECT
                (SELECT count(*) FROM judge_attempts AS j
                 JOIN shadow_attempts AS a ON a.shadow_attempt_id = j.shadow_attempt_id
                 WHERE a.anchor_id = ?1),
                (SELECT count(*) FROM evaluations AS e
                 JOIN shadow_attempts AS a ON a.shadow_attempt_id = e.shadow_attempt_id
                 WHERE a.anchor_id = ?1)",
                params![anchor_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(downstream, (0, 0));
        assert_reconciliation_noop_twice(&mut recovered.repository, 31_001);
    }

    #[test]
    fn startup_orphans_pending_anchor_and_validates_the_terminal_on_second_pass() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = shadow_config(&path, "shadow-orphan-pending");
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let pending = seed_pending_anchor(
            &mut origin.repository,
            &origin_identity,
            &[attempt("first", 20)],
        );
        drop(origin);

        let mut recovered = LedgerRepository::activate_at(&config, 31_000).unwrap();
        let terminal: (String, String) = recovered
            .repository
            .connection
            .query_row(
                "SELECT state, dead_process_instance_id
                 FROM anchor_state_events
                 WHERE anchor_id = ?1 AND state <> 'pending'",
                params![pending.anchor_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(terminal.0, "orphaned_non_resumable");
        assert_eq!(terminal.1, origin_identity.process_instance_id.to_string());
        assert_eq!(
            recovered
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM sample_batches WHERE anchor_id = ?1",
                    params![pending.anchor_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_reconciliation_noop_twice(&mut recovered.repository, 31_000);
    }

    #[test]
    fn startup_recovers_pending_work_owned_by_an_already_stopped_process() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = shadow_config(&path, "shadow-stopped-owner-recovery");
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let pending = seed_pending_anchor(
            &mut origin.repository,
            &origin_identity,
            &[attempt("first", 20)],
        );
        assert_eq!(
            origin
                .repository
                .stop_process(ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), 1_001).unwrap())
                .unwrap(),
            super::super::process::ProcessCommandAck::Applied
        );
        drop(origin);

        let mut recovered = LedgerRepository::activate_at(&config, 1_002).unwrap();
        let state: String = recovered
            .repository
            .connection
            .query_row(
                "SELECT state FROM anchor_state_events
                 WHERE anchor_id = ?1 AND state <> 'pending'",
                params![pending.anchor_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(state, "orphaned_non_resumable");
        assert_eq!(
            recovered
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM process_instance_state_events
                     WHERE subject_process_instance_id = ?1 AND state = 'reconciled'",
                    params![origin_identity.process_instance_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_reconciliation_noop_twice(&mut recovered.repository, 1_002);
    }

    #[test]
    fn startup_orphans_reserved_and_started_attempts_without_replay() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = shadow_config(&path, "shadow-orphan-mixed-start");
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let reserved = attempt("first", 20);
        let started = attempt("second", 20);
        let anchor_id = seed_closed_anchor(
            &mut origin.repository,
            &origin_identity,
            &[reserved.clone(), started.clone()],
        );
        let reservation = reservation(
            &origin_identity,
            anchor_id,
            vec![reserved.clone(), started.clone()],
            20,
        );
        assert_eq!(
            origin
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            origin
                .repository
                .start_shadow_attempt(
                    ShadowAttemptStarted::new(
                        started.shadow_attempt_id,
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                        21,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ShadowCommandAck::Applied
        );
        drop(origin);

        let mut recovered = LedgerRepository::activate_at(&config, 31_000).unwrap();
        assert_eq!(
            recovered
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM shadow_results AS r
                     JOIN shadow_attempts AS a ON a.shadow_attempt_id = r.shadow_attempt_id
                     WHERE a.sample_batch_id = ?1 AND r.terminal_class = 'orphaned_in_flight'",
                    params![reservation.sample_batch_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            2
        );
        assert_eq!(
            recovered
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM shadow_attempt_state_events
                     WHERE shadow_attempt_id = ?1 AND state = 'started'",
                    params![started.shadow_attempt_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        assert_eq!(
            recovered
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM shadow_attempt_state_events
                     WHERE shadow_attempt_id = ?1 AND state = 'started'",
                    params![reserved.shadow_attempt_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_reconciliation_noop_twice(&mut recovered.repository, 31_000);
    }

    #[test]
    fn startup_preserves_terminal_sibling_and_orphans_only_incomplete_attempt() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = shadow_config(&path, "shadow-mixed-terminal-recovery");
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let terminal_attempt = attempt("first", 20);
        let incomplete_attempt = attempt("second", 20);
        let anchor_id = seed_closed_anchor(
            &mut origin.repository,
            &origin_identity,
            &[terminal_attempt.clone(), incomplete_attempt.clone()],
        );
        let reservation = reservation(
            &origin_identity,
            anchor_id,
            vec![terminal_attempt.clone(), incomplete_attempt.clone()],
            20,
        );
        origin
            .repository
            .reserve_sample_batch(&reservation)
            .unwrap();
        for attempt in [&terminal_attempt, &incomplete_attempt] {
            origin
                .repository
                .start_shadow_attempt(
                    ShadowAttemptStarted::new(
                        attempt.shadow_attempt_id,
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                        21,
                    )
                    .unwrap(),
                )
                .unwrap();
        }
        assert_eq!(
            origin
                .repository
                .record_shadow_terminal(&operational_terminal(
                    &reservation,
                    &terminal_attempt,
                    false,
                    22,
                ))
                .unwrap(),
            ShadowCommandAck::Applied
        );
        drop(origin);

        let mut recovered = LedgerRepository::activate_at(&config, 31_000).unwrap();
        let classes = recovered
            .repository
            .connection
            .prepare(
                "SELECT shadow_attempt_id, terminal_class FROM shadow_results
                 WHERE shadow_attempt_id IN (?1, ?2) ORDER BY shadow_attempt_id",
            )
            .unwrap()
            .query_map(
                params![
                    terminal_attempt.shadow_attempt_id.to_string(),
                    incomplete_attempt.shadow_attempt_id.to_string()
                ],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(classes.len(), 2);
        assert!(classes.contains(&(
            terminal_attempt.shadow_attempt_id.to_string(),
            "operational_failure".to_string()
        )));
        assert!(classes.contains(&(
            incomplete_attempt.shadow_attempt_id.to_string(),
            "orphaned_in_flight".to_string()
        )));
        assert_reconciliation_noop_twice(&mut recovered.repository, 31_000);
    }

    #[test]
    fn startup_validates_fully_terminal_batch_without_rewriting_it() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = shadow_config(&path, "shadow-terminal-batch-recovery");
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let attempt = attempt("first", 20);
        let anchor_id = seed_closed_anchor(
            &mut origin.repository,
            &origin_identity,
            std::slice::from_ref(&attempt),
        );
        let reservation = reservation(&origin_identity, anchor_id, vec![attempt.clone()], 20);
        origin
            .repository
            .reserve_sample_batch(&reservation)
            .unwrap();
        origin
            .repository
            .start_shadow_attempt(
                ShadowAttemptStarted::new(
                    attempt.shadow_attempt_id,
                    Uuid::now_v7(),
                    Uuid::now_v7(),
                    21,
                )
                .unwrap(),
            )
            .unwrap();
        origin
            .repository
            .record_shadow_terminal(&operational_terminal(&reservation, &attempt, true, 22))
            .unwrap();
        let before: (String, String, String, String, String, String) = origin
            .repository
            .connection
            .query_row(
                "SELECT r.terminal_class, r.canonical_payload_hash,
                        s.state, s.canonical_payload_hash,
                        b.state, b.canonical_payload_hash
                 FROM shadow_results AS r
                 JOIN shadow_attempt_state_events AS s
                   ON s.shadow_attempt_id = r.shadow_attempt_id
                  AND s.state NOT IN ('reserved', 'started')
                 JOIN sample_batch_state_events AS b
                   ON b.sample_batch_id = ?2 AND b.state <> 'open'
                 WHERE r.shadow_attempt_id = ?1",
                params![
                    attempt.shadow_attempt_id.to_string(),
                    reservation.sample_batch_id.to_string()
                ],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .unwrap();
        drop(origin);

        let mut recovered = LedgerRepository::activate_at(&config, 31_000).unwrap();
        let after: (String, String, String, String, String, String) = recovered
            .repository
            .connection
            .query_row(
                "SELECT r.terminal_class, r.canonical_payload_hash,
                        s.state, s.canonical_payload_hash,
                        b.state, b.canonical_payload_hash
                 FROM shadow_results AS r
                 JOIN shadow_attempt_state_events AS s
                   ON s.shadow_attempt_id = r.shadow_attempt_id
                  AND s.state NOT IN ('reserved', 'started')
                 JOIN sample_batch_state_events AS b
                   ON b.sample_batch_id = ?2 AND b.state <> 'open'
                 WHERE r.shadow_attempt_id = ?1",
                params![
                    attempt.shadow_attempt_id.to_string(),
                    reservation.sample_batch_id.to_string()
                ],
                |row| {
                    Ok((
                        row.get(0)?,
                        row.get(1)?,
                        row.get(2)?,
                        row.get(3)?,
                        row.get(4)?,
                        row.get(5)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(after, before);
        assert_reconciliation_noop_twice(&mut recovered.repository, 31_000);
    }

    #[test]
    fn startup_preserves_deterministic_evaluation_before_shadow_terminal() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = shadow_config(&path, "shadow-deterministic-evaluation-recovery");
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let attempt = attempt("first", 20);
        let anchor_id = seed_closed_anchor(
            &mut origin.repository,
            &origin_identity,
            std::slice::from_ref(&attempt),
        );
        let reservation = reservation(&origin_identity, anchor_id, vec![attempt.clone()], 20);
        origin
            .repository
            .reserve_sample_batch(&reservation)
            .unwrap();
        origin
            .repository
            .start_shadow_attempt(
                ShadowAttemptStarted::new(
                    attempt.shadow_attempt_id,
                    Uuid::now_v7(),
                    Uuid::now_v7(),
                    21,
                )
                .unwrap(),
            )
            .unwrap();
        let evaluation_id = seed_deterministic_evaluation(
            &origin.repository,
            &attempt,
            DeterministicHardFailureV1::ToolContract,
        );
        drop(origin);

        let mut recovered = LedgerRepository::activate_at(&config, 31_000).unwrap();
        assert_eq!(
            recovered
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM evaluations WHERE evaluation_id = ?1",
                    params![evaluation_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        let result: (String, Option<String>) = recovered
            .repository
            .connection
            .query_row(
                "SELECT terminal_class, evaluation_id FROM shadow_results
                 WHERE shadow_attempt_id = ?1",
                params![attempt.shadow_attempt_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(result, ("orphaned_in_flight".to_string(), None));
        assert_reconciliation_noop_twice(&mut recovered.repository, 31_000);
    }

    #[test]
    fn startup_rejects_a_malformed_started_state_without_partial_recovery() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = shadow_config(&path, "shadow-corrupt-start-recovery");
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let started = attempt("corrupt-start", 20);
        let anchor_id = seed_closed_anchor(
            &mut origin.repository,
            &origin_identity,
            std::slice::from_ref(&started),
        );
        let reservation = reservation(&origin_identity, anchor_id, vec![started.clone()], 20);
        assert_eq!(
            origin
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            origin
                .repository
                .start_shadow_attempt(
                    ShadowAttemptStarted::new(
                        started.shadow_attempt_id,
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                        21,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ShadowCommandAck::Applied
        );
        origin
            .repository
            .connection
            .execute(
                "UPDATE shadow_attempt_state_events
                 SET canonical_payload_hash = ?1
                 WHERE shadow_attempt_id = ?2 AND state = 'started'",
                params!["0".repeat(64), started.shadow_attempt_id.to_string()],
            )
            .unwrap();
        drop(origin);

        let error = match LedgerRepository::activate_at(&config, 31_000) {
            Ok(_) => panic!("malformed started state must reject activation"),
            Err(error) => error,
        };
        assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);
        let connection = Connection::open(&path).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM process_instance_state_events
                     WHERE subject_process_instance_id = ?1 AND state = 'reconciled'",
                    params![origin_identity.process_instance_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM shadow_results", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            0
        );
    }

    #[test]
    fn startup_rejects_future_shadow_reservation_and_rolls_back_recovery() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = shadow_config(&path, "shadow-future-recovery");
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let future_attempt = attempt("first", 100_000);
        let anchor_id = seed_closed_anchor(
            &mut origin.repository,
            &origin_identity,
            std::slice::from_ref(&future_attempt),
        );
        let reservation = reservation(&origin_identity, anchor_id, vec![future_attempt], 100_000);
        origin
            .repository
            .reserve_sample_batch(&reservation)
            .unwrap();
        let before = recovery_snapshot(&origin.repository.connection);
        drop(origin);

        let error = match LedgerRepository::activate_at(&config, 31_000) {
            Ok(_) => panic!("future Shadow reservation must reject recovery"),
            Err(error) => error,
        };
        assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);
        let connection = Connection::open(&path).unwrap();
        assert_eq!(recovery_snapshot(&connection), before);
    }

    #[derive(Debug, Clone, Copy)]
    enum StartupCorruption {
        ResultWithoutTerminalState,
        TerminalStateWithoutResult,
        TerminalBatchWithIncompleteAttempt,
        OpenBatchWithAllTerminalAttempts,
        AnchorWithoutResult,
        AnchorWithoutPendingState,
    }

    impl StartupCorruption {
        const fn project_suffix(self) -> &'static str {
            match self {
                Self::ResultWithoutTerminalState => "result-without-state",
                Self::TerminalStateWithoutResult => "state-without-result",
                Self::TerminalBatchWithIncompleteAttempt => "terminal-batch-incomplete",
                Self::OpenBatchWithAllTerminalAttempts => "open-batch-all-terminal",
                Self::AnchorWithoutResult => "anchor-without-result",
                Self::AnchorWithoutPendingState => "anchor-without-pending-state",
            }
        }
    }

    #[test]
    fn startup_corruption_matrix_rolls_back_every_recovery_write() {
        for corruption in [
            StartupCorruption::ResultWithoutTerminalState,
            StartupCorruption::TerminalStateWithoutResult,
            StartupCorruption::TerminalBatchWithIncompleteAttempt,
            StartupCorruption::OpenBatchWithAllTerminalAttempts,
            StartupCorruption::AnchorWithoutResult,
            StartupCorruption::AnchorWithoutPendingState,
        ] {
            let temporary = tempdir().unwrap();
            let path = database_path(&temporary);
            let project_id = format!("shadow-corruption-{}", corruption.project_suffix());
            let config = shadow_config(&path, &project_id);
            let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
            let origin_identity = origin.identity.clone();

            // This valid row is processed before final aggregate validation and
            // proves that an earlier recovery write is rolled back with the fault.
            seed_pending_anchor(
                &mut origin.repository,
                &origin_identity,
                &[attempt("first", 20)],
            );

            match corruption {
                StartupCorruption::AnchorWithoutResult
                | StartupCorruption::AnchorWithoutPendingState => {
                    let corrupt = seed_pending_anchor(
                        &mut origin.repository,
                        &origin_identity,
                        &[attempt("second", 20)],
                    );
                    let (table, predicate) = match corruption {
                        StartupCorruption::AnchorWithoutResult => {
                            ("anchor_results", "anchor_id = ?1")
                        }
                        StartupCorruption::AnchorWithoutPendingState => (
                            "anchor_state_events",
                            "anchor_id = ?1 AND state = 'pending'",
                        ),
                        _ => unreachable!(),
                    };
                    origin
                        .repository
                        .connection
                        .execute(
                            &format!("DELETE FROM {table} WHERE {predicate}"),
                            params![corrupt.anchor_id.to_string()],
                        )
                        .unwrap();
                }
                StartupCorruption::ResultWithoutTerminalState
                | StartupCorruption::TerminalStateWithoutResult
                | StartupCorruption::TerminalBatchWithIncompleteAttempt
                | StartupCorruption::OpenBatchWithAllTerminalAttempts => {
                    let first = attempt("corrupt-terminal", 20);
                    let second = attempt("second", 20);
                    let anchor_id = seed_closed_anchor(
                        &mut origin.repository,
                        &origin_identity,
                        &[first.clone(), second.clone()],
                    );
                    let reservation = reservation(
                        &origin_identity,
                        anchor_id,
                        vec![first.clone(), second.clone()],
                        20,
                    );
                    assert_eq!(
                        origin
                            .repository
                            .reserve_sample_batch(&reservation)
                            .unwrap(),
                        ShadowCommandAck::Applied
                    );

                    match corruption {
                        StartupCorruption::ResultWithoutTerminalState
                        | StartupCorruption::TerminalStateWithoutResult => {
                            origin
                                .repository
                                .start_shadow_attempt(
                                    ShadowAttemptStarted::new(
                                        first.shadow_attempt_id,
                                        Uuid::now_v7(),
                                        Uuid::now_v7(),
                                        21,
                                    )
                                    .unwrap(),
                                )
                                .unwrap();
                            assert_eq!(
                                origin
                                    .repository
                                    .record_shadow_terminal(&operational_terminal(
                                        &reservation,
                                        &first,
                                        false,
                                        23,
                                    ))
                                    .unwrap(),
                                ShadowCommandAck::Applied
                            );
                            let table = match corruption {
                                StartupCorruption::ResultWithoutTerminalState => {
                                    "shadow_attempt_state_events"
                                }
                                StartupCorruption::TerminalStateWithoutResult => "shadow_results",
                                _ => unreachable!(),
                            };
                            let predicate = if matches!(
                                corruption,
                                StartupCorruption::ResultWithoutTerminalState
                            ) {
                                "shadow_attempt_id = ?1 AND state NOT IN ('reserved', 'started')"
                            } else {
                                "shadow_attempt_id = ?1"
                            };
                            origin
                                .repository
                                .connection
                                .execute(
                                    &format!("DELETE FROM {table} WHERE {predicate}"),
                                    params![first.shadow_attempt_id.to_string()],
                                )
                                .unwrap();
                        }
                        StartupCorruption::TerminalBatchWithIncompleteAttempt => {
                            insert_batch_state(
                                &origin.repository.connection,
                                Uuid::now_v7(),
                                reservation.sample_batch_id,
                                origin_identity.process_instance_id,
                                None,
                                "closed",
                                22,
                            )
                            .unwrap();
                        }
                        StartupCorruption::OpenBatchWithAllTerminalAttempts => {
                            for (attempt, created_at_unix_ms) in
                                [(&first, 21_i64), (&second, 22_i64)]
                            {
                                origin
                                    .repository
                                    .start_shadow_attempt(
                                        ShadowAttemptStarted::new(
                                            attempt.shadow_attempt_id,
                                            Uuid::now_v7(),
                                            Uuid::now_v7(),
                                            created_at_unix_ms,
                                        )
                                        .unwrap(),
                                    )
                                    .unwrap();
                            }
                            origin
                                .repository
                                .record_shadow_terminal(&operational_terminal(
                                    &reservation,
                                    &first,
                                    false,
                                    23,
                                ))
                                .unwrap();
                            origin
                                .repository
                                .record_shadow_terminal(&operational_terminal(
                                    &reservation,
                                    &second,
                                    true,
                                    24,
                                ))
                                .unwrap();
                            origin
                                .repository
                                .connection
                                .execute(
                                    "DELETE FROM sample_batch_state_events
                                     WHERE sample_batch_id = ?1 AND state <> 'open'",
                                    params![reservation.sample_batch_id.to_string()],
                                )
                                .unwrap();
                        }
                        _ => unreachable!(),
                    }
                }
            }

            let before = recovery_snapshot(&origin.repository.connection);
            drop(origin);
            let error = match LedgerRepository::activate_at(&config, 31_000) {
                Ok(_) => panic!("{corruption:?} must reject startup recovery"),
                Err(error) => error,
            };
            assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);
            let connection = Connection::open(&path).unwrap();
            assert_eq!(
                recovery_snapshot(&connection),
                before,
                "{corruption:?} left partial recovery writes"
            );
        }
    }

    #[test]
    fn startup_rejects_corrupt_historical_config_backing_an_anchor() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let original_config = shadow_config(&path, "shadow-historical-config");
        let mut origin = LedgerRepository::activate_at(&original_config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        seed_pending_anchor(
            &mut origin.repository,
            &origin_identity,
            &[attempt("first", 20)],
        );

        let mut current_config = original_config.clone();
        current_config.retention_days = 31;
        let current = LedgerRepository::activate_at(&current_config, 1_001).unwrap();
        drop(current);
        let stored_json: String = origin
            .repository
            .connection
            .query_row(
                "SELECT canonical_config_json FROM config_generations
                 WHERE config_generation_id = ?1",
                params![origin_identity.config_generation_id],
                |row| row.get(0),
            )
            .unwrap();
        let mut value: Json = serde_json::from_str(&stored_json).unwrap();
        value["retention_days"] = json!(999);
        origin
            .repository
            .connection
            .execute(
                "UPDATE config_generations SET canonical_config_json = ?1
                 WHERE config_generation_id = ?2",
                params![
                    canonical_json(&value).unwrap(),
                    origin_identity.config_generation_id
                ],
            )
            .unwrap();
        drop(origin);

        let error = match LedgerRepository::activate_at(&current_config, 31_001) {
            Ok(_) => panic!("corrupt historical config must reject activation"),
            Err(error) => error,
        };
        assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);
    }

    #[test]
    fn startup_rejects_corrupt_historical_learning_generation() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = shadow_config(&path, "shadow-historical-learning");
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let historical_learning = origin_identity.pools["pool-a"].learning_generation_id;
        seed_pending_anchor(
            &mut origin.repository,
            &origin_identity,
            &[attempt("first", 20)],
        );
        let replacement = origin
            .repository
            .reset_pool("pool-a", "test-rotation", "historical validation")
            .unwrap();
        assert_ne!(replacement, historical_learning);
        origin
            .repository
            .connection
            .execute(
                "UPDATE learning_generations SET canonical_payload_hash = ?1
                 WHERE learning_generation_id = ?2",
                params!["0".repeat(64), historical_learning.to_string()],
            )
            .unwrap();
        drop(origin);

        let error = match LedgerRepository::activate_at(&config, 31_000) {
            Ok(_) => panic!("corrupt historical learning identity must reject activation"),
            Err(error) => error,
        };
        assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);
    }

    #[test]
    fn normal_shadow_writes_reject_inverted_stage_and_batch_timestamps() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "shadow-write-chronology");
        let identity = activated.identity.clone();

        let too_early = attempt("first", 9);
        let early_anchor = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            std::slice::from_ref(&too_early),
        );
        let early_reservation = reservation(&identity, early_anchor, vec![too_early], 9);
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&early_reservation)
                .unwrap(),
            ShadowCommandAck::Conflict
        );

        let first = attempt("first", 20);
        let second = attempt("second", 20);
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            &[first.clone(), second.clone()],
        );
        let reservation = reservation(
            &identity,
            anchor_id,
            vec![first.clone(), second.clone()],
            20,
        );
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .start_shadow_attempt(
                    ShadowAttemptStarted::new(
                        first.shadow_attempt_id,
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                        19,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        for (attempt, created_at) in [(&first, 21), (&second, 22)] {
            assert_eq!(
                activated
                    .repository
                    .start_shadow_attempt(
                        ShadowAttemptStarted::new(
                            attempt.shadow_attempt_id,
                            Uuid::now_v7(),
                            Uuid::now_v7(),
                            created_at,
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                ShadowCommandAck::Applied
            );
        }
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&operational_terminal(&reservation, &first, false, 50,))
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&operational_terminal(&reservation, &second, true, 40,))
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM shadow_results", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            1
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM sample_batch_state_events WHERE state <> 'open'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
    }

    #[test]
    fn shadow_terminal_cannot_precede_its_attached_evaluation() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "shadow-evaluation-chronology");
        let identity = activated.identity.clone();
        let attempt = attempt("first", 20);
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            std::slice::from_ref(&attempt),
        );
        let reservation = reservation(&identity, anchor_id, vec![attempt.clone()], 20);
        activated
            .repository
            .reserve_sample_batch(&reservation)
            .unwrap();
        activated
            .repository
            .start_shadow_attempt(
                ShadowAttemptStarted::new(
                    attempt.shadow_attempt_id,
                    Uuid::now_v7(),
                    Uuid::now_v7(),
                    21,
                )
                .unwrap(),
            )
            .unwrap();
        let evaluation_id = seed_deterministic_evaluation(
            &activated.repository,
            &attempt,
            DeterministicHardFailureV1::ToolContract,
        );
        let batch_terminal = SampleBatchTerminalEvent::new(
            reservation.sample_batch_id,
            Uuid::now_v7(),
            SampleBatchTerminalState::Closed,
            None,
            24,
        )
        .unwrap();
        let terminal = ShadowTerminalRecord::new(
            Uuid::now_v7(),
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            ShadowTerminalClass::DeterministicFailure,
            None,
            None,
            Some(DeterministicHardFailureV1::ToolContract),
            None,
            Some(1),
            None,
            Some(evaluation_id),
            ShadowVectorSourceV1::Canonicalizable {
                query_inputs: Box::new(attempt.request_projection.clone()),
            },
            Some(batch_terminal),
            24,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM shadow_results", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn premature_batch_close_rolls_back_result_and_last_attempt_can_close() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "shadow-last-close");
        let identity = activated.identity.clone();
        let first = attempt("one", 20);
        let second = attempt("two", 20);
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            &[first.clone(), second.clone()],
        );
        let reservation = reservation(
            &identity,
            anchor_id,
            vec![first.clone(), second.clone()],
            20,
        );
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        for (attempt, created_at_unix_ms) in [(&first, 21_i64), (&second, 22_i64)] {
            assert_eq!(
                activated
                    .repository
                    .start_shadow_attempt(
                        ShadowAttemptStarted::new(
                            attempt.shadow_attempt_id,
                            Uuid::now_v7(),
                            Uuid::now_v7(),
                            created_at_unix_ms,
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                ShadowCommandAck::Applied
            );
        }
        let premature = operational_terminal(&reservation, &first, true, 30);
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&premature)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM shadow_results", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );

        let first_terminal = operational_terminal(&reservation, &first, false, 31);
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&first_terminal)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        let second_terminal = operational_terminal(&reservation, &second, true, 32);
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&second_terminal)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&first_terminal)
                .unwrap(),
            ShadowCommandAck::AlreadyApplied
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM shadow_results", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            2
        );
    }

    #[test]
    fn noncanonicalizable_terminal_has_one_result_without_query_or_vector_hash() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "shadow-noncanonical");
        let identity = activated.identity.clone();
        let attempt = attempt("noncanonical", 20);
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            std::slice::from_ref(&attempt),
        );
        let reservation = reservation(&identity, anchor_id, vec![attempt.clone()], 20);
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        let terminal = ShadowTerminalRecord::new(
            Uuid::now_v7(),
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            ShadowTerminalClass::CanceledShutdown,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            ShadowVectorSourceV1::Noncanonicalizable {
                reason: NoncanonicalizableReason::new("missing_query_inputs").unwrap(),
            },
            Some(
                SampleBatchTerminalEvent::new(
                    reservation.sample_batch_id,
                    Uuid::now_v7(),
                    SampleBatchTerminalState::CanceledShutdown,
                    None,
                    30,
                )
                .unwrap(),
            ),
            30,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        let stored: (i64, Option<String>, Option<String>, Option<String>) = activated
            .repository
            .connection
            .query_row(
                "SELECT canonicalizable, query_inputs_json, vector_source_hash,
                        noncanonicalizable_reason FROM shadow_results",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            stored,
            (0, None, None, Some("missing_query_inputs".to_string()))
        );
    }

    #[test]
    fn terminalized_origin_is_rejected_before_duplicate_resolution() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "shadow-origin");
        let identity = activated.identity.clone();
        let attempt = attempt("origin", 20);
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            std::slice::from_ref(&attempt),
        );
        let reservation = reservation(&identity, anchor_id, vec![attempt], 20);
        activated
            .repository
            .connection
            .execute(
                "INSERT INTO process_instance_state_events (
                    process_state_event_id, process_instance_id, state,
                    subject_process_instance_id, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, 'stopped', ?2, 19, ?3)",
                params![
                    Uuid::now_v7().to_string(),
                    identity.process_instance_id.to_string(),
                    "f".repeat(64),
                ],
            )
            .unwrap();
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::OriginatingProcessNotLive
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM sample_batches", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[test]
    fn alternate_state_event_ids_conflict_without_partial_domain_writes() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "shadow-state-id-conflict");
        let identity = activated.identity.clone();

        let first_attempt = attempt("state-id-first", 20);
        let first_anchor = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            std::slice::from_ref(&first_attempt),
        );
        let first_reservation =
            reservation(&identity, first_anchor, vec![first_attempt.clone()], 20);
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&first_reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );

        let second_attempt = attempt("state-id-second", 21);
        let second_anchor = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            std::slice::from_ref(&second_attempt),
        );
        let second_reservation =
            reservation(&identity, second_anchor, vec![second_attempt.clone()], 21);
        let mut open_id_collision = second_reservation.clone();
        open_id_collision.open_state_event_id = first_reservation.open_state_event_id;
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&open_id_collision)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM sample_batches WHERE sample_batch_id = ?1",
                    params![second_reservation.sample_batch_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );

        let mut reserved_id_collision = second_reservation.clone();
        reserved_id_collision.attempts[0].reserved_state_event_id =
            first_attempt.reserved_state_event_id;
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&reserved_id_collision)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&second_reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );

        let first_started = ShadowAttemptStarted::new(
            first_attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            22,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .start_shadow_attempt(first_started)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        let start_id_collision = ShadowAttemptStarted::new(
            second_attempt.shadow_attempt_id,
            first_started.state_event_id,
            Uuid::now_v7(),
            23,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .start_shadow_attempt(start_id_collision)
                .unwrap(),
            ShadowCommandAck::Conflict
        );

        let valid_terminal = operational_terminal(&first_reservation, &first_attempt, true, 30);
        let mut attempt_state_id_collision = valid_terminal.clone();
        attempt_state_id_collision.state_event_id = first_started.state_event_id;
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&attempt_state_id_collision)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        let mut batch_state_id_collision = valid_terminal.clone();
        batch_state_id_collision
            .batch_terminal
            .as_mut()
            .unwrap()
            .state_event_id = first_reservation.open_state_event_id;
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&batch_state_id_collision)
                .unwrap(),
            ShadowCommandAck::Conflict
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM shadow_results WHERE shadow_attempt_id = ?1",
                    params![first_attempt.shadow_attempt_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&valid_terminal)
                .unwrap(),
            ShadowCommandAck::Applied
        );
    }

    #[test]
    fn transaction_start_guard_is_retained_through_begin() {
        struct Guard<'a>(&'a std::cell::Cell<bool>);
        impl TransactionStartGuard for Guard<'_> {
            fn permits_transaction(&self) -> bool {
                true
            }
        }
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                self.0.set(true);
            }
        }

        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "shadow-start-guard");
        let identity = activated.identity.clone();
        let guard_attempt = attempt("guard", 20);
        let anchor_id = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            std::slice::from_ref(&guard_attempt),
        );
        let batch_reservation = reservation(&identity, anchor_id, vec![guard_attempt], 20);
        let dropped = std::cell::Cell::new(false);
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch_with_start_check(&batch_reservation, || Some(Guard(&dropped)))
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert!(dropped.get());
        let refused_attempt = attempt("refused", 21);
        let refused_anchor = seed_closed_anchor(
            &mut activated.repository,
            &identity,
            std::slice::from_ref(&refused_attempt),
        );
        let refused = reservation(&identity, refused_anchor, vec![refused_attempt], 21);
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch_with_start_check(&refused, || None::<Guard<'_>>)
                .unwrap(),
            ShadowCommandAck::TransactionNotStarted
        );
    }
}
