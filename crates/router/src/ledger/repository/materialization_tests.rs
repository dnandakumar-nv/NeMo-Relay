// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::{TimeZone, Utc};
use nemo_relay::api::llm::{
    LlmApiFamily, LlmAttributes, LlmCallRole, LlmExecutionContextSnapshot, LlmRequest,
    LlmTrajectoryScopeSnapshot,
};
use nemo_relay::api::runtime::{
    LLM_REPLAY_CONTRACT_VERSION, LlmReplayCall, LlmReplayCapability, LlmReplayTransport,
};
use nemo_relay::api::scope::ScopeType;
use rusqlite::types::Value;
use rusqlite::{Connection, params};
use serde_json::json;
use tempfile::{TempDir, tempdir};
use uuid::Uuid;

use super::anchors::{AnchorCommandAck, FrozenPendingAnchorV1, FrozenTerminalAnchorV1};
use super::background_work::{
    select_backfill_work, select_embedding_work, select_failure_propagation_work,
    select_materialization_work,
};
use super::decision::tests as decision_test_fixtures;
use super::embedding::{
    EmbeddingJobBatchClaim, EmbeddingJobBatchClaimAck, EmbeddingJobBatchClaimItem,
    EmbeddingJobBatchCompletion, EmbeddingJobBatchCompletionAck, EmbeddingJobBatchCompletionItem,
    EmbeddingJobCreate, EmbeddingJobCreateAck, EmbeddingJobReset, EmbeddingJobResetAck,
    EmbeddingJobResolution, EmbeddingJobResolutionAck, EmbeddingJobResolutionKind,
    EmbeddingJobResolvedState, create_embedding_job_in_transaction,
};
use super::judge::{EvaluationRecord, JudgeAttemptStart, JudgeAttemptTerminal, JudgeRecordAck};
use super::materialization::{
    MATERIALIZATION_ATTEMPT_MAX, MaterializationClaim, MaterializationClaimAck,
    MaterializationCompletion, MaterializationCompletionAck, MaterializationCompletionState,
    MaterializationFailurePropagation, MaterializationFailurePropagationAck,
    MaterializationFailurePropagationItem, MaterializationResolution, MaterializationResolutionAck,
    MaterializationResolutionKind, MaterializationResolvedState, MaterializationRetentionAck,
    MaterializationRetentionDelete, MaterializationSnapshot, VectorBackfillAck,
    VectorBackfillCommand,
    deindex_materialization_for_retention_with_retired_spaces_in_transaction,
    delete_materialization_for_retention_in_transaction,
    delete_materialization_for_retention_with_retired_spaces_in_transaction,
    load_verified_materialization_job, load_verified_vector_link_page,
    verify_retiring_anchor_deindexed_in_transaction,
};
use super::process::{HeartbeatAck, HeartbeatRenewal, ProcessCommandAck, ProcessStop};
use super::retention::{RetentionAck, RetentionRequest};
use super::shadow::{
    AtomicShadowVectorization, NoncanonicalizableReason, ReservedShadowAttempt,
    SampleBatchReservation, SampleBatchTerminalEvent, SampleBatchTerminalState,
    ShadowAttemptStarted, ShadowCommandAck, ShadowOperationalFailureClass, ShadowTerminalClass,
    ShadowTerminalRecord, ShadowVectorSourceV1, ShadowVectorizationHandoff,
};
use super::tests::{config, database_path};
use super::vector_catalog::{
    CanonicalQueryEnsureAck, EmbeddingCacheSource, EmbeddingCacheUpsertAck, EmbeddingCacheWrite,
    ensure_canonical_query, upsert_embedding_cache,
};
use super::vector_index::{
    ActiveGenerationResolution, GenerationAuthorizationAck, GenerationObjectCreationAck,
    RebuildFlipAck, RebuildLeaseClaimAck, RebuildStepAck, RetiredGenerationCleanupAck,
    VectorIndexHealthMutationAck, VectorIndexHealthTarget, VectorIndexManifestState,
    VectorSourceChangeOperation, authoritative_source_fingerprint, authorize_generation,
    catch_up_rebuild_changes, claim_rebuild_lease, cleanup_retired_generation,
    create_generation_objects, flip_rebuild_generation, load_validated_manifest,
    mark_vector_index_health, populate_rebuild_chunk, resolve_active_generation,
    vector_source_change_payload_hash, vector_source_sequence_payload_hash,
};
use super::vector_search::{search_projected_neighbors, search_projected_neighbors_in_transaction};
use super::{ActivatedLedger, LedgerRepository};
use crate::adapter::FamilyAdapter;
use crate::background_jobs::{BackgroundJobOutcome, execute_materialization};
use crate::canonical_json::canonical_sha256;
use crate::canonical_query::build_canonical_routing_query;
use crate::config::LearningConfig;
use crate::judge::{
    DeterministicHardFailureV1, JudgeHorizonV1, JudgePolicyIdentityV1, PairwiseJudgeInputV1,
};
use crate::ledger::model::{LedgerErrorClass, LedgerRuntimeIdentity};
use crate::ledger::read_pool::LedgerReadPool;
use crate::ledger::writer::LedgerWriterOwner;
use crate::preflight::preflight;
use crate::projection::{
    REQUEST_PROJECTION_SCHEMA_V1, ROUTER_SANITIZER_VERSION, ROUTING_CONTEXT_SCHEMA_V1,
    RouterRequestProjectionV1, RouterRoutingContextProjectionV1, SanitizedAnnotatedLlmRequest,
    SanitizedMessage, SanitizedMessageContent, candidate_request_projection,
};
use crate::trajectory::{
    CANDIDATE_FACT_SCHEMA_V1, PENDING_TRAJECTORY_SCHEMA_V1, PendingTrajectoryWindow,
    PersistedCandidateCapabilitiesV1, PersistedCandidateFactV1, PersistedTrajectoryTerminalV1,
    RESPONSE_PROJECTION_SCHEMA_V1, ReplayCapabilityFactsV1, RouterResponseProjectionV1,
    TRAJECTORY_SANITIZER_VERSION, TrajectoryTrigger,
};
use crate::vector::{
    AuthoritativeVector, NormalizedVector, PartitionId, VectorChecksum, VectorDimensions,
    VectorRecordId, VectorSpaceId,
};

const VECTOR_GRAPH_TABLES: &[&str] = &[
    "canonical_routing_queries",
    "routing_partitions",
    "vectorization_outcomes",
    "embedding_jobs",
    "embedding_job_state_events",
    "embeddings",
    "evidence_vector_links",
    "evidence_vector_link_state_events",
    "vector_materialization_jobs",
    "vector_materialization_job_state_events",
    "vector_source_change_events",
];

fn activate(learning_enabled: bool, project_id: &str) -> (TempDir, ActivatedLedger) {
    let (temporary, _config, activated) = activate_with_config(learning_enabled, project_id);
    (temporary, activated)
}

fn activate_with_config(
    learning_enabled: bool,
    project_id: &str,
) -> (TempDir, crate::config::RouterConfig, ActivatedLedger) {
    let temporary = tempdir().unwrap();
    let path = database_path(&temporary);
    let router_config = router_config(&path, project_id, learning_enabled);
    let activated = LedgerRepository::activate_at(&router_config, 0).unwrap();
    (temporary, router_config, activated)
}

fn activate_with_decision_policy(
    project_id: &str,
) -> (TempDir, crate::config::RouterConfig, ActivatedLedger) {
    let temporary = tempdir().unwrap();
    let path = database_path(&temporary);
    let mut config = router_config(&path, project_id, true);
    config.pools[0].learning = Some(LearningConfig {
        version: 1,
        embedder: "embedder-a".to_string(),
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
        promotion_lower_bound: Some(0.2),
        retention_lower_bound: None,
        holdout_probability: None,
        active_canary_fraction: None,
    });
    let activated = LedgerRepository::activate_at(&config, 0).unwrap();
    (temporary, config, activated)
}

fn router_config(
    path: &Path,
    project_id: &str,
    learning_enabled: bool,
) -> crate::config::RouterConfig {
    let mut router_config = config(path, project_id);
    if learning_enabled {
        router_config.pools[0].learning = Some(LearningConfig::minimal("embedder-a"));
    }
    router_config
}

fn evaluator_version() -> String {
    config(Path::new("router.db"), "materialization-tests").pools[0]
        .judge
        .evaluator_version()
        .unwrap()
}

fn fingerprint_without_field<T: serde::Serialize>(value: &T, field: &str) -> String {
    let mut value = serde_json::to_value(value).unwrap();
    value.as_object_mut().unwrap().remove(field);
    canonical_sha256(&value).unwrap()
}

fn request_projection(has_task: bool) -> RouterRequestProjectionV1 {
    let messages = if has_task {
        vec![SanitizedMessage::User {
            content: SanitizedMessageContent::Text("route this request".to_string()),
            name: None,
        }]
    } else {
        Vec::new()
    };
    let mut projection = RouterRequestProjectionV1 {
        schema: REQUEST_PROJECTION_SCHEMA_V1.to_string(),
        family: LlmApiFamily::OpenAIChatCompletions,
        normalized_request: SanitizedAnnotatedLlmRequest {
            messages,
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
    projection.semantic_request_fingerprint =
        fingerprint_without_field(&projection, "semantic_request_fingerprint");
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
    response_projection_for_model("anchor-a")
}

fn response_projection_for_model(model: &str) -> RouterResponseProjectionV1 {
    let mut projection = RouterResponseProjectionV1 {
        schema: RESPONSE_PROJECTION_SCHEMA_V1.to_string(),
        sanitizer_version: TRAJECTORY_SANITIZER_VERSION,
        id: None,
        model: Some(model.to_string()),
        message: None,
        tool_calls: None,
        finish_reason: None,
        usage: None,
        semantic_response_fingerprint: String::new(),
    };
    projection.semantic_response_fingerprint =
        fingerprint_without_field(&projection, "semantic_response_fingerprint");
    projection
}

fn attempt(has_task: bool) -> ReservedShadowAttempt {
    attempt_at(has_task, 0)
}

fn attempt_at(has_task: bool, time_base_unix_ms: i64) -> ReservedShadowAttempt {
    let request =
        candidate_request_projection(&request_projection(has_task), "candidate-model-a").unwrap();
    ReservedShadowAttempt::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        "candidate-a",
        "candidate-model-a",
        "2026-06-01",
        0,
        LlmApiFamily::OpenAIChatCompletions,
        "transport-shared",
        "anchor-a",
        "2026-07-01",
        "1".repeat(64),
        evaluator_version(),
        "3".repeat(64),
        "4".repeat(64),
        true,
        request,
        time_base_unix_ms + 20,
    )
    .unwrap()
}

fn attempt_from_active_preflight(
    outcome: &crate::preflight::PreflightOutcome,
    time_base_unix_ms: i64,
) -> ReservedShadowAttempt {
    let candidate = outcome.eligible_candidate_facts.first().unwrap();
    let request =
        candidate_request_projection(&outcome.request_projection, &candidate.model).unwrap();
    ReservedShadowAttempt::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        candidate.candidate_id.clone(),
        candidate.model.clone(),
        candidate.model_revision.clone(),
        candidate.cost_rank,
        outcome.replay_capability.api_family,
        outcome.replay_capability.transport_identity.clone(),
        "anchor-a",
        "2026-07-01",
        candidate.decoding_fingerprint.clone(),
        evaluator_version(),
        outcome.routing_projection.tenant_policy_hash.clone(),
        outcome.routing_projection.agent_policy_hash.clone(),
        true,
        request,
        time_base_unix_ms + 20,
    )
    .unwrap()
}

fn seed_closed_anchor_at(
    repository: &mut LedgerRepository,
    identity: &LedgerRuntimeIdentity,
    attempt: &ReservedShadowAttempt,
    time_base_unix_ms: i64,
) -> Uuid {
    let pool = identity.pools.get("pool-a").unwrap();
    let mut anchor_request = attempt.request_projection.clone();
    anchor_request.normalized_request.model = Some("anchor-a".to_string());
    anchor_request.semantic_request_fingerprint =
        fingerprint_without_field(&anchor_request, "semantic_request_fingerprint");
    let pending = PendingTrajectoryWindow {
        schema: PENDING_TRAJECTORY_SCHEMA_V1.to_string(),
        anchor_id: Uuid::now_v7(),
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
        routing_context_projection: RouterRoutingContextProjectionV1 {
            schema: ROUTING_CONTEXT_SCHEMA_V1.to_string(),
            tenant_policy_hash: attempt.tenant_policy_hash.clone(),
            agent_policy_hash: attempt.agent_policy_hash.clone(),
            position_features: BTreeMap::new(),
        },
        normalized_anchor_response: response_projection(),
        replay_capability_facts: ReplayCapabilityFactsV1::from_capability(&LlmReplayCapability {
            contract_version: LLM_REPLAY_CONTRACT_VERSION,
            api_family: LlmApiFamily::OpenAIChatCompletions,
            transport_identity: "transport-shared".to_string(),
        })
        .unwrap(),
        candidate_facts: vec![PersistedCandidateFactV1 {
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
        }],
        requested_progress: 1,
        opened_at: Utc
            .timestamp_millis_opt(time_base_unix_ms)
            .single()
            .unwrap(),
        deadline_at: Utc
            .timestamp_millis_opt(time_base_unix_ms + 10)
            .single()
            .unwrap(),
    };
    let anchor_id = pending.anchor_id;
    let frozen_pending =
        FrozenPendingAnchorV1::new(&pending, Uuid::now_v7(), Uuid::now_v7(), time_base_unix_ms)
            .unwrap();
    assert!(matches!(
        repository.record_pending_anchor(&frozen_pending).unwrap(),
        AnchorCommandAck::Applied { .. }
    ));
    let terminal = PersistedTrajectoryTerminalV1::closed(
        pending,
        Vec::new(),
        1,
        TrajectoryTrigger::ProgressReached,
        Utc.timestamp_millis_opt(time_base_unix_ms + 10)
            .single()
            .unwrap(),
        Vec::new(),
    );
    let frozen_terminal = FrozenTerminalAnchorV1::new(
        &terminal,
        Uuid::now_v7(),
        Uuid::now_v7(),
        time_base_unix_ms + 10,
    )
    .unwrap();
    assert!(matches!(
        repository.record_terminal_anchor(&frozen_terminal).unwrap(),
        AnchorCommandAck::Applied { .. }
    ));
    anchor_id
}

fn seed_started_attempt(
    activated: &mut ActivatedLedger,
    attempt: &ReservedShadowAttempt,
) -> SampleBatchReservation {
    seed_attempt_at(activated, attempt, true, 0)
}

fn seed_started_attempt_at(
    activated: &mut ActivatedLedger,
    attempt: &ReservedShadowAttempt,
    time_base_unix_ms: i64,
) -> SampleBatchReservation {
    seed_attempt_at(activated, attempt, true, time_base_unix_ms)
}

fn seed_attempt(
    activated: &mut ActivatedLedger,
    attempt: &ReservedShadowAttempt,
    started: bool,
) -> SampleBatchReservation {
    seed_attempt_at(activated, attempt, started, 0)
}

fn seed_attempt_at(
    activated: &mut ActivatedLedger,
    attempt: &ReservedShadowAttempt,
    started: bool,
    time_base_unix_ms: i64,
) -> SampleBatchReservation {
    let identity = activated.identity.clone();
    let anchor_id = seed_closed_anchor_at(
        &mut activated.repository,
        &identity,
        attempt,
        time_base_unix_ms,
    );
    let pool = identity.pools.get("pool-a").unwrap();
    let reservation = SampleBatchReservation::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        anchor_id,
        identity.config_generation_id,
        pool.policy_version_id.clone(),
        pool.learning_generation_id,
        "pool-a",
        vec![attempt.clone()],
        time_base_unix_ms + 20,
    )
    .unwrap();
    assert_eq!(
        activated
            .repository
            .reserve_sample_batch(&reservation)
            .unwrap(),
        ShadowCommandAck::Applied
    );
    if started {
        assert_eq!(
            activated
                .repository
                .start_shadow_attempt(
                    ShadowAttemptStarted::new(
                        attempt.shadow_attempt_id,
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                        time_base_unix_ms + 21,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ShadowCommandAck::Applied
        );
    }
    reservation
}

fn terminal(
    reservation: &SampleBatchReservation,
    attempt: &ReservedShadowAttempt,
) -> ShadowTerminalRecord {
    terminal_for_class(
        reservation,
        attempt,
        ShadowTerminalClass::OperationalFailure,
        None,
        None,
        None,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn terminal_for_class(
    reservation: &SampleBatchReservation,
    attempt: &ReservedShadowAttempt,
    terminal_class: ShadowTerminalClass,
    dead_process_instance_id: Option<Uuid>,
    normalized_response: Option<RouterResponseProjectionV1>,
    deterministic_hard_failure: Option<DeterministicHardFailureV1>,
    evaluation_id: Option<Uuid>,
) -> ShadowTerminalRecord {
    terminal_for_class_at(
        reservation,
        attempt,
        terminal_class,
        dead_process_instance_id,
        normalized_response,
        deterministic_hard_failure,
        evaluation_id,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn terminal_for_class_at(
    reservation: &SampleBatchReservation,
    attempt: &ReservedShadowAttempt,
    terminal_class: ShadowTerminalClass,
    dead_process_instance_id: Option<Uuid>,
    normalized_response: Option<RouterResponseProjectionV1>,
    deterministic_hard_failure: Option<DeterministicHardFailureV1>,
    evaluation_id: Option<Uuid>,
    time_base_unix_ms: i64,
) -> ShadowTerminalRecord {
    let batch_state = match terminal_class {
        ShadowTerminalClass::Completed
        | ShadowTerminalClass::DeterministicFailure
        | ShadowTerminalClass::OperationalFailure
        | ShadowTerminalClass::SkippedCooloff => SampleBatchTerminalState::Closed,
        ShadowTerminalClass::CanceledShutdown => SampleBatchTerminalState::CanceledShutdown,
        ShadowTerminalClass::OrphanedBeforeSchedule => {
            SampleBatchTerminalState::OrphanedBeforeSchedule
        }
        ShadowTerminalClass::OrphanedInFlight => SampleBatchTerminalState::OrphanedInFlight,
    };
    let operational_failure_class = (terminal_class == ShadowTerminalClass::OperationalFailure)
        .then(|| ShadowOperationalFailureClass::new("router.provider.timeout").unwrap());
    ShadowTerminalRecord::new(
        Uuid::now_v7(),
        attempt.shadow_attempt_id,
        Uuid::now_v7(),
        Uuid::now_v7(),
        terminal_class,
        dead_process_instance_id,
        normalized_response,
        deterministic_hard_failure,
        operational_failure_class,
        matches!(
            terminal_class,
            ShadowTerminalClass::Completed
                | ShadowTerminalClass::DeterministicFailure
                | ShadowTerminalClass::OperationalFailure
        )
        .then(|| u64::try_from(time_base_unix_ms + 12).unwrap()),
        None,
        evaluation_id,
        ShadowVectorSourceV1::Canonicalizable {
            query_inputs: Box::new(attempt.request_projection.clone()),
        },
        Some(
            SampleBatchTerminalEvent::new(
                reservation.sample_batch_id,
                Uuid::now_v7(),
                batch_state,
                dead_process_instance_id,
                time_base_unix_ms + 30,
            )
            .unwrap(),
        ),
        time_base_unix_ms + 30,
    )
    .unwrap()
}

fn atomic_vectorization() -> ShadowVectorizationHandoff {
    atomic_vectorization_with_routing(routing_projection())
}

fn atomic_vectorization_for(attempt: &ReservedShadowAttempt) -> ShadowVectorizationHandoff {
    atomic_vectorization_with_routing(RouterRoutingContextProjectionV1 {
        schema: ROUTING_CONTEXT_SCHEMA_V1.to_string(),
        tenant_policy_hash: attempt.tenant_policy_hash.clone(),
        agent_policy_hash: attempt.agent_policy_hash.clone(),
        position_features: BTreeMap::new(),
    })
}

fn atomic_vectorization_with_routing(
    routing: RouterRoutingContextProjectionV1,
) -> ShadowVectorizationHandoff {
    ShadowVectorizationHandoff::Atomic(
        AtomicShadowVectorization::new(
            routing,
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
        )
        .unwrap(),
    )
}

fn seed_judge_evaluation(
    activated: &mut ActivatedLedger,
    attempt: &ReservedShadowAttempt,
    response_equivalence: f64,
    trajectory_equivalence: f64,
    judge_confidence: f64,
) -> (Uuid, RouterResponseProjectionV1) {
    seed_judge_evaluation_at(
        activated,
        attempt,
        response_equivalence,
        trajectory_equivalence,
        judge_confidence,
        0,
    )
}

fn seed_judge_evaluation_at(
    activated: &mut ActivatedLedger,
    attempt: &ReservedShadowAttempt,
    response_equivalence: f64,
    trajectory_equivalence: f64,
    judge_confidence: f64,
    time_base_unix_ms: i64,
) -> (Uuid, RouterResponseProjectionV1) {
    let router_config = config(Path::new("router.db"), "materialization-judge");
    let judge = &router_config.pools[0].judge;
    let candidate_response = response_projection_for_model(&attempt.candidate_model);
    let mut anchor_request = attempt.request_projection.clone();
    anchor_request.normalized_request.model = Some("anchor-a".to_string());
    anchor_request.semantic_request_fingerprint =
        fingerprint_without_field(&anchor_request, "semantic_request_fingerprint");
    let judge_input = PairwiseJudgeInputV1::new(
        &anchor_request,
        &response_projection(),
        &candidate_response,
        &[],
        JudgeHorizonV1::new(1, 1, TrajectoryTrigger::ProgressReached, false).unwrap(),
        JudgePolicyIdentityV1::from_config(judge).unwrap(),
    )
    .unwrap();
    let start = JudgeAttemptStart::new(
        Uuid::now_v7(),
        attempt.shadow_attempt_id,
        activated.identity.pools["pool-a"].learning_generation_id,
        attempt.evaluator_version.clone(),
        judge,
        &judge_input,
        0,
        Uuid::now_v7(),
        Uuid::now_v7(),
        time_base_unix_ms + 22,
    )
    .unwrap();
    assert_eq!(
        activated
            .repository
            .record_judge_attempt_start(&start)
            .unwrap(),
        JudgeRecordAck::Applied
    );
    let evaluation_id = Uuid::now_v7();
    let terminal = JudgeAttemptTerminal::valid(
        start.judge_attempt_id,
        Uuid::now_v7(),
        Uuid::now_v7(),
        attempt.shadow_attempt_id,
        attempt.evaluator_version.clone(),
        &json!({
            "response_equivalence": response_equivalence,
            "trajectory_equivalence": trajectory_equivalence,
            "judge_confidence": judge_confidence,
            "hard_failures": [],
            "rationale": "bounded rationale",
        })
        .to_string(),
        judge,
        evaluation_id,
        Uuid::now_v7(),
        false,
        time_base_unix_ms + 23,
    )
    .unwrap();
    assert_eq!(
        activated
            .repository
            .record_judge_attempt_terminal(&terminal)
            .unwrap(),
        JudgeRecordAck::Applied
    );
    (evaluation_id, candidate_response)
}

fn stored_link_fact(repository: &LedgerRepository) -> (String, Option<String>) {
    repository
        .connection
        .query_row(
            "SELECT terminal_class, quality_label FROM evidence_vector_links",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap()
}

#[derive(Clone)]
struct PendingMaterialization {
    materialization_job_id: String,
    evidence_vector_link_id: Uuid,
    partition_id: i64,
    attempt_generation: i64,
    canonical_payload_hash: String,
    embedding_job_id: String,
    vector_space_id: VectorSpaceId,
    canonical_query_hash: String,
}

#[derive(serde::Serialize)]
struct ManifestHashPayload {
    vector_space_id: String,
    generation: i64,
    state: String,
    root_table_name: String,
    dimensions: i64,
    expected_schema_objects_json: String,
    expected_schema_objects_sha256: String,
    base_source_seq: i64,
    applied_source_seq: i64,
    build_cursor_record_id: Option<String>,
    source_record_count: Option<i64>,
    source_fingerprint_sha256: Option<String>,
    stable_error_class: Option<String>,
    created_at_unix_ms: i64,
    activated_at_unix_ms: Option<i64>,
    retired_at_unix_ms: Option<i64>,
    dropped_at_unix_ms: Option<i64>,
}

fn rehash_vector_index_manifest(connection: &Connection, vector_space_id: &VectorSpaceId) {
    let payload = connection
        .query_row(
            "SELECT vector_space_id, generation, state, root_table_name, dimensions,
                    expected_schema_objects_json, expected_schema_objects_sha256,
                    base_source_seq, applied_source_seq, build_cursor_record_id,
                    source_record_count, source_fingerprint_sha256, stable_error_class,
                    created_at_unix_ms, activated_at_unix_ms, retired_at_unix_ms,
                    dropped_at_unix_ms
             FROM vector_index_manifest
             WHERE vector_space_id = ?1 AND state = 'active'",
            params![vector_space_id.as_str()],
            |row| {
                Ok(ManifestHashPayload {
                    vector_space_id: row.get(0)?,
                    generation: row.get(1)?,
                    state: row.get(2)?,
                    root_table_name: row.get(3)?,
                    dimensions: row.get(4)?,
                    expected_schema_objects_json: row.get(5)?,
                    expected_schema_objects_sha256: row.get(6)?,
                    base_source_seq: row.get(7)?,
                    applied_source_seq: row.get(8)?,
                    build_cursor_record_id: row.get(9)?,
                    source_record_count: row.get(10)?,
                    source_fingerprint_sha256: row.get(11)?,
                    stable_error_class: row.get(12)?,
                    created_at_unix_ms: row.get(13)?,
                    activated_at_unix_ms: row.get(14)?,
                    retired_at_unix_ms: row.get(15)?,
                    dropped_at_unix_ms: row.get(16)?,
                })
            },
        )
        .unwrap();
    let hash = canonical_sha256(&serde_json::to_value(payload).unwrap()).unwrap();
    assert_eq!(
        connection
            .execute(
                "UPDATE vector_index_manifest SET canonical_payload_hash = ?1
                 WHERE vector_space_id = ?2 AND state = 'active'",
                params![hash, vector_space_id.as_str()],
            )
            .unwrap(),
        1
    );
}

fn seed_pending_materialization(activated: &mut ActivatedLedger) -> PendingMaterialization {
    let attempt = attempt(true);
    let reservation = seed_started_attempt(activated, &attempt);
    let terminal = terminal(&reservation, &attempt).with_vectorization(atomic_vectorization());
    assert_eq!(
        activated
            .repository
            .record_shadow_terminal(&terminal)
            .unwrap(),
        ShadowCommandAck::Applied
    );
    latest_materialization(activated)
}

fn latest_materialization(activated: &ActivatedLedger) -> PendingMaterialization {
    activated
        .repository
        .connection
        .query_row(
            "SELECT m.vector_materialization_job_id, m.evidence_vector_link_id,
                    l.partition_id, m.attempt_generation, m.canonical_payload_hash,
                    m.embedding_job_id, m.vector_space_id, m.canonical_query_hash
             FROM vector_materialization_jobs AS m
             JOIN evidence_vector_links AS l
               ON l.evidence_vector_link_id = m.evidence_vector_link_id
             ORDER BY l.rowid DESC LIMIT 1",
            [],
            |row| {
                let evidence_vector_link_id = row.get::<_, String>(1)?;
                let vector_space_id = row.get::<_, String>(6)?;
                Ok(PendingMaterialization {
                    materialization_job_id: row.get(0)?,
                    evidence_vector_link_id: Uuid::parse_str(&evidence_vector_link_id).unwrap(),
                    partition_id: row.get(2)?,
                    attempt_generation: row.get(3)?,
                    canonical_payload_hash: row.get(4)?,
                    embedding_job_id: row.get(5)?,
                    vector_space_id: VectorSpaceId::new(vector_space_id).unwrap(),
                    canonical_query_hash: row.get(7)?,
                })
            },
        )
        .unwrap()
}

fn authoritative_test_vector(
    repository: &LedgerRepository,
    vector_space_id: &VectorSpaceId,
) -> AuthoritativeVector {
    let dimensions = repository
        .connection
        .query_row(
            "SELECT dimensions FROM vector_spaces WHERE vector_space_id = ?1",
            params![vector_space_id.as_str()],
            |row| row.get::<_, u32>(0),
        )
        .unwrap();
    let dimensions = VectorDimensions::new(dimensions).unwrap();
    let mut values = vec![0.0; dimensions.as_usize()];
    values[0] = 1.0;
    AuthoritativeVector::from_normalized(
        vector_space_id,
        NormalizedVector::from_provider_f64(&values, dimensions).unwrap(),
    )
    .unwrap()
}

fn complete_pending_embedding(
    activated: &mut ActivatedLedger,
    pending: &PendingMaterialization,
) -> Uuid {
    complete_pending_embedding_at(activated, pending, 0)
}

fn complete_pending_embedding_at(
    activated: &mut ActivatedLedger,
    pending: &PendingMaterialization,
    time_base_unix_ms: i64,
) -> Uuid {
    let lease_token = Uuid::now_v7();
    let claim = EmbeddingJobBatchClaim::new(
        Uuid::now_v7(),
        pending.vector_space_id.clone(),
        lease_token,
        time_base_unix_ms + 31,
        vec![
            EmbeddingJobBatchClaimItem::new(
                pending.embedding_job_id.clone(),
                Uuid::now_v7(),
                Uuid::now_v7(),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let lease = match activated
        .repository
        .claim_embedding_job_batch(&claim)
        .unwrap()
    {
        EmbeddingJobBatchClaimAck::Claimed(mut leases) if leases.len() == 1 => leases.remove(0),
        acknowledgement => panic!("unexpected embedding claim: {acknowledgement:?}"),
    };
    let embedding_id = Uuid::now_v7();
    let complete = EmbeddingJobBatchCompletion::new(
        Uuid::now_v7(),
        pending.vector_space_id.clone(),
        lease_token,
        time_base_unix_ms + 32,
        vec![
            EmbeddingJobBatchCompletionItem::new(
                pending.embedding_job_id.clone(),
                embedding_id,
                Uuid::now_v7(),
                lease.job.attempt_generation,
                pending.canonical_query_hash.clone(),
                authoritative_test_vector(&activated.repository, &pending.vector_space_id),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    match activated
        .repository
        .complete_embedding_job_batch(&complete)
        .unwrap()
    {
        EmbeddingJobBatchCompletionAck::Applied(results)
            if results.len() == 1 && results[0].embedding.embedding_id == embedding_id => {}
        acknowledgement => panic!("unexpected embedding completion: {acknowledgement:?}"),
    }
    embedding_id
}

fn activate_empty_vector_generation(
    activated: &mut ActivatedLedger,
    vector_space_id: &VectorSpaceId,
    observed_at_unix_ms: i64,
) -> String {
    let dimensions = activated
        .repository
        .connection
        .query_row(
            "SELECT dimensions FROM vector_spaces WHERE vector_space_id = ?1",
            params![vector_space_id.as_str()],
            |row| row.get::<_, u32>(0),
        )
        .unwrap();
    let owner = activated.identity.process_instance_id;
    let transaction = activated.repository.connection_mut().transaction().unwrap();
    let manifest = match authorize_generation(
        &transaction,
        vector_space_id,
        VectorDimensions::new(dimensions).unwrap(),
        observed_at_unix_ms,
    )
    .unwrap()
    {
        GenerationAuthorizationAck::Created(manifest) => manifest,
        acknowledgement => panic!("unexpected generation authorization: {acknowledgement:?}"),
    };
    let fence = match claim_rebuild_lease(&transaction, vector_space_id, owner, observed_at_unix_ms)
        .unwrap()
    {
        RebuildLeaseClaimAck::Claimed(fence) => fence,
        acknowledgement => panic!("unexpected rebuild claim: {acknowledgement:?}"),
    };
    assert_eq!(
        create_generation_objects(&transaction, &fence, observed_at_unix_ms).unwrap(),
        GenerationObjectCreationAck::Created
    );
    assert!(matches!(
        populate_rebuild_chunk(&transaction, &fence, observed_at_unix_ms + 1).unwrap(),
        RebuildStepAck::Applied {
            processed: 0,
            complete: true,
            applied_source_seq: 0,
        }
    ));
    assert!(matches!(
        catch_up_rebuild_changes(&transaction, &fence, observed_at_unix_ms + 1).unwrap(),
        RebuildStepAck::Applied {
            processed: 0,
            complete: true,
            applied_source_seq: 0,
        }
    ));
    assert_eq!(
        flip_rebuild_generation(&transaction, &fence, observed_at_unix_ms + 2).unwrap(),
        RebuildFlipAck::Activated { record_count: 0 }
    );
    let root = manifest.authority().root().as_str().to_string();
    transaction.commit().unwrap();
    root
}

fn materialization_claim(
    materialization_job_id: &str,
    expected_attempt_generation: i64,
    expected_canonical_payload_hash: &str,
    lease_token: Uuid,
    observed_at_unix_ms: i64,
) -> MaterializationClaim {
    MaterializationClaim::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        materialization_job_id,
        lease_token,
        expected_attempt_generation,
        expected_canonical_payload_hash,
        observed_at_unix_ms,
    )
    .unwrap()
}

fn materialization_completion(
    job: &MaterializationSnapshot,
    lease_token: Uuid,
    completed_at_unix_ms: i64,
) -> MaterializationCompletion {
    MaterializationCompletion::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        job.materialization_job_id.clone(),
        lease_token,
        job.attempt_generation,
        job.canonical_payload_hash.clone(),
        completed_at_unix_ms,
    )
    .unwrap()
}

fn seed_ready_materialization(
    activated: &mut ActivatedLedger,
) -> (PendingMaterialization, String, MaterializationSnapshot) {
    let pending = seed_pending_materialization(activated);
    complete_pending_embedding(activated, &pending);
    let root = activate_empty_vector_generation(activated, &pending.vector_space_id, 33);
    let lease_token = Uuid::now_v7();
    let claim = materialization_claim(
        &pending.materialization_job_id,
        pending.attempt_generation,
        &pending.canonical_payload_hash,
        lease_token,
        36,
    );
    let lease = match activated.repository.claim_materialization(&claim).unwrap() {
        MaterializationClaimAck::Claimed(lease) => lease,
        acknowledgement => panic!("unexpected ready claim: {acknowledgement:?}"),
    };
    let completion = materialization_completion(&lease.job, lease_token, 37);
    let ready = match activated
        .repository
        .complete_materialization(&completion)
        .unwrap()
    {
        MaterializationCompletionAck::Applied {
            state: MaterializationCompletionState::Ready,
            job,
        } => job,
        acknowledgement => panic!("unexpected ready completion: {acknowledgement:?}"),
    };
    (pending, root, ready)
}

pub(crate) fn ready_retention_runtime_fixture() -> (
    TempDir,
    crate::config::RouterConfig,
    ActivatedLedger,
    VectorSpaceId,
    String,
    Uuid,
) {
    let temporary = tempdir().unwrap();
    let path = database_path(&temporary);
    let mut config = router_config(&path, "retention-vector-runtime", true);
    config.max_evidence_records = 1;
    config.embedders[0].api_key_env = None;
    let mut activated = LedgerRepository::activate_at(&config, 0).unwrap();
    let (pending, root, _ready) = seed_ready_materialization(&mut activated);
    let vector_space_id = pending.vector_space_id.clone();
    let evidence_vector_link_id = pending.evidence_vector_link_id;
    (
        temporary,
        config,
        activated,
        vector_space_id,
        root,
        evidence_vector_link_id,
    )
}

pub(crate) fn ready_evaluated_runtime_fixture() -> (
    TempDir,
    crate::config::RouterConfig,
    ActivatedLedger,
    VectorSpaceId,
    String,
    Uuid,
    Uuid,
) {
    let temporary = tempdir().unwrap();
    let path = database_path(&temporary);
    let mut config = router_config(&path, "evaluated-vector-runtime", true);
    config.embedders[0].api_key_env = None;
    let mut activated = LedgerRepository::activate_at(&config, 0).unwrap();
    let attempt = attempt(true);
    let reservation = seed_started_attempt(&mut activated, &attempt);
    let (evaluation_id, candidate_response) =
        seed_judge_evaluation(&mut activated, &attempt, 0.9, 0.9, 0.9);
    let terminal = terminal_for_class(
        &reservation,
        &attempt,
        ShadowTerminalClass::Completed,
        None,
        Some(candidate_response),
        None,
        Some(evaluation_id),
    )
    .with_vectorization(atomic_vectorization());
    assert_eq!(
        activated
            .repository
            .record_shadow_terminal(&terminal)
            .unwrap(),
        ShadowCommandAck::Applied
    );
    let pending = latest_materialization(&activated);
    complete_pending_embedding(&mut activated, &pending);
    let root = activate_empty_vector_generation(&mut activated, &pending.vector_space_id, 33);
    let lease_token = Uuid::now_v7();
    let claim = materialization_claim(
        &pending.materialization_job_id,
        pending.attempt_generation,
        &pending.canonical_payload_hash,
        lease_token,
        36,
    );
    let lease = match activated.repository.claim_materialization(&claim).unwrap() {
        MaterializationClaimAck::Claimed(lease) => lease,
        acknowledgement => panic!("unexpected evaluated claim: {acknowledgement:?}"),
    };
    let completion = materialization_completion(&lease.job, lease_token, 37);
    assert!(matches!(
        activated
            .repository
            .complete_materialization(&completion)
            .unwrap(),
        MaterializationCompletionAck::Applied {
            state: MaterializationCompletionState::Ready,
            ..
        }
    ));
    (
        temporary,
        config,
        activated,
        pending.vector_space_id,
        root,
        pending.evidence_vector_link_id,
        evaluation_id,
    )
}

pub(crate) fn ready_evaluated_active_runtime_fixture() -> (
    TempDir,
    crate::config::RouterConfig,
    ActivatedLedger,
    VectorSpaceId,
    AuthoritativeVector,
    Uuid,
    Uuid,
) {
    ready_evaluated_active_runtime_fixture_with_attribution_seconds(600)
}

pub(crate) fn ready_evaluated_active_runtime_fixture_with_attribution_seconds(
    max_attribution_seconds: u64,
) -> (
    TempDir,
    crate::config::RouterConfig,
    ActivatedLedger,
    VectorSpaceId,
    AuthoritativeVector,
    Uuid,
    Uuid,
) {
    ready_evaluated_active_runtime_fixture_with_embedder_options(
        max_attribution_seconds,
        None,
        None,
        None,
    )
}

pub(crate) fn ready_evaluated_active_runtime_fixture_with_embedder(
    base_url: &str,
    api_key_env: Option<&str>,
    timeout_ms: u64,
) -> (
    TempDir,
    crate::config::RouterConfig,
    ActivatedLedger,
    VectorSpaceId,
    AuthoritativeVector,
    Uuid,
    Uuid,
) {
    ready_evaluated_active_runtime_fixture_with_embedder_options(
        600,
        Some(base_url),
        api_key_env,
        Some(timeout_ms),
    )
}

fn ready_evaluated_active_runtime_fixture_with_embedder_options(
    max_attribution_seconds: u64,
    base_url: Option<&str>,
    api_key_env: Option<&str>,
    timeout_ms: Option<u64>,
) -> (
    TempDir,
    crate::config::RouterConfig,
    ActivatedLedger,
    VectorSpaceId,
    AuthoritativeVector,
    Uuid,
    Uuid,
) {
    let temporary = tempdir().unwrap();
    let path = database_path(&temporary);
    let mut config = super::active::tests::active_config(&path, "active-evaluated-vector-runtime");
    if let Some(base_url) = base_url {
        config.embedders[0].base_url = base_url.to_string();
    }
    config.embedders[0].api_key_env = api_key_env.map(str::to_string);
    if let Some(timeout_ms) = timeout_ms {
        config.embedders[0].timeout_ms = timeout_ms;
    }
    config.pools[0].outcome.insert(
        "max_attribution_seconds".to_string(),
        json!(max_attribution_seconds),
    );
    let learning = config.pools[0].learning.as_mut().unwrap();
    learning.promotion_lower_bound = Some(0.01);
    learning.retention_lower_bound = Some(0.0);
    assert!(config.validate().is_empty(), "{:?}", config.validate());
    let time_base_unix_ms = Utc::now().timestamp_millis().max(1_000) - 1_000;
    let mut activated = LedgerRepository::activate_at(&config, time_base_unix_ms).unwrap();
    let context = active_runtime_context();
    let request = active_runtime_request();
    let replay: Arc<dyn LlmReplayTransport> = Arc::new(ActiveFixtureReplay {
        capability: LlmReplayCapability {
            contract_version: LLM_REPLAY_CONTRACT_VERSION,
            api_family: LlmApiFamily::OpenAIChatCompletions,
            transport_identity: "transport-shared".to_string(),
        },
    });
    let preflight = preflight(
        &context,
        &request,
        Some(&replay),
        &config.pools[0],
        &FamilyAdapter,
    )
    .unwrap();
    let attempt = attempt_from_active_preflight(&preflight, time_base_unix_ms);
    let reservation = seed_started_attempt_at(&mut activated, &attempt, time_base_unix_ms);
    let (evaluation_id, candidate_response) =
        seed_judge_evaluation_at(&mut activated, &attempt, 0.9, 0.9, 0.9, time_base_unix_ms);
    let terminal = terminal_for_class_at(
        &reservation,
        &attempt,
        ShadowTerminalClass::Completed,
        None,
        Some(candidate_response),
        None,
        Some(evaluation_id),
        time_base_unix_ms,
    )
    .with_vectorization(atomic_vectorization_for(&attempt));
    assert_eq!(
        activated
            .repository
            .record_shadow_terminal(&terminal)
            .unwrap(),
        ShadowCommandAck::Applied
    );
    let pending = latest_materialization(&activated);
    complete_pending_embedding_at(&mut activated, &pending, time_base_unix_ms);
    let query_vector = authoritative_test_vector(&activated.repository, &pending.vector_space_id);
    let _root = activate_empty_vector_generation(
        &mut activated,
        &pending.vector_space_id,
        time_base_unix_ms + 33,
    );
    let lease_token = Uuid::now_v7();
    let claim = materialization_claim(
        &pending.materialization_job_id,
        pending.attempt_generation,
        &pending.canonical_payload_hash,
        lease_token,
        time_base_unix_ms + 36,
    );
    let lease = match activated.repository.claim_materialization(&claim).unwrap() {
        MaterializationClaimAck::Claimed(lease) => lease,
        acknowledgement => panic!("unexpected evaluated claim: {acknowledgement:?}"),
    };
    let completion = materialization_completion(&lease.job, lease_token, time_base_unix_ms + 37);
    assert!(matches!(
        activated
            .repository
            .complete_materialization(&completion)
            .unwrap(),
        MaterializationCompletionAck::Applied {
            state: MaterializationCompletionState::Ready,
            ..
        }
    ));
    (
        temporary,
        config,
        activated,
        pending.vector_space_id,
        query_vector,
        pending.evidence_vector_link_id,
        evaluation_id,
    )
}

struct ActiveFixtureReplay {
    capability: LlmReplayCapability,
}

impl LlmReplayTransport for ActiveFixtureReplay {
    fn capability(&self) -> &LlmReplayCapability {
        &self.capability
    }

    fn start(&self, _request: LlmRequest) -> nemo_relay::error::Result<LlmReplayCall> {
        Err(nemo_relay::error::FlowError::Internal(
            "Active fixture replay must not start".to_string(),
        ))
    }
}

pub(crate) fn active_runtime_inspection_input(
    config: &crate::config::RouterConfig,
    partition: crate::routing_partition::RoutingPartitionV1,
) -> crate::inspection::RoutingInspectionInputV1 {
    let context = active_runtime_context();
    let request = active_runtime_request();
    let replay: Arc<dyn LlmReplayTransport> = Arc::new(ActiveFixtureReplay {
        capability: LlmReplayCapability {
            contract_version: LLM_REPLAY_CONTRACT_VERSION,
            api_family: LlmApiFamily::OpenAIChatCompletions,
            transport_identity: "transport-shared".to_string(),
        },
    });
    let outcome = preflight(
        &context,
        &request,
        Some(&replay),
        &config.pools[0],
        &FamilyAdapter,
    )
    .unwrap();
    crate::inspection::RoutingInspectionInputV1 {
        schema: crate::inspection::ROUTING_INSPECTION_INPUT_SCHEMA_V1.to_string(),
        pool_id: config.pools[0].id.clone(),
        partition,
        request: outcome.request_projection,
        routing_context: outcome.routing_projection,
    }
}

pub(crate) fn active_runtime_context() -> LlmExecutionContextSnapshot {
    active_runtime_context_for_root(Uuid::now_v7())
}

pub(crate) fn active_runtime_context_for_root(root_uuid: Uuid) -> LlmExecutionContextSnapshot {
    LlmExecutionContextSnapshot {
        call_uuid: Uuid::now_v7(),
        root_uuid,
        parent_uuid: root_uuid,
        trajectory_owner_uuid: root_uuid,
        trajectory_owner_path: vec![LlmTrajectoryScopeSnapshot {
            uuid: root_uuid,
            name: "active-agent".to_string(),
            scope_type: ScopeType::Agent,
        }],
        api_family: LlmApiFamily::OpenAIChatCompletions,
        call_role: LlmCallRole::Primary,
        attributes: LlmAttributes::empty(),
        tenant_id: Some("tenant-private-a".to_string()),
        agent_id: Some("agent-private-a".to_string()),
        sanitized_metadata: BTreeMap::from([("region".to_string(), json!("private-region-a"))]),
    }
}

pub(crate) fn active_runtime_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "anchor-a",
            "messages": [{"role": "user", "content": "route this request"}],
            "max_tokens": 128,
        }),
    }
}

#[test]
fn canonical_non_ready_links_advance_verified_page_cursor() {
    let (_temporary, mut activated) = activate(true, "verified-non-ready-pagination");
    let first = seed_pending_materialization(&mut activated);
    let second = seed_pending_materialization(&mut activated);
    let mut observed = std::collections::BTreeSet::new();
    let mut cursor = None;

    for _ in 0..2 {
        let page = load_verified_vector_link_page(
            &activated.repository.connection,
            &first.vector_space_id,
            cursor,
            1,
        )
        .unwrap();
        assert!(page.ready_records.is_empty());
        assert!(!page.exhausted);
        cursor = page.last_scanned_record_id;
        assert!(observed.insert(cursor.unwrap()));
    }
    let final_page = load_verified_vector_link_page(
        &activated.repository.connection,
        &first.vector_space_id,
        cursor,
        1,
    )
    .unwrap();
    assert!(final_page.ready_records.is_empty());
    assert!(final_page.last_scanned_record_id.is_none());
    assert!(final_page.exhausted);
    assert_eq!(
        observed,
        std::collections::BTreeSet::from([
            crate::vector::VectorRecordId::new(first.evidence_vector_link_id).unwrap(),
            crate::vector::VectorRecordId::new(second.evidence_vector_link_id).unwrap(),
        ])
    );
    assert_eq!(
        authoritative_source_fingerprint(&activated.repository.connection, &first.vector_space_id,)
            .unwrap()
            .record_count(),
        0
    );
}

#[test]
fn stale_hash_non_ready_state_is_corrupt_before_ready_filtering() {
    let (_temporary, mut activated) = activate(true, "stale-non-ready-state");
    let pending = seed_pending_materialization(&mut activated);
    activated
        .repository
        .connection
        .execute(
            "UPDATE evidence_vector_link_state_events
             SET state = 'canceled_retention'
             WHERE evidence_vector_link_id = ?1",
            params![pending.evidence_vector_link_id.to_string()],
        )
        .unwrap();

    let error = authoritative_source_fingerprint(
        &activated.repository.connection,
        &pending.vector_space_id,
    )
    .unwrap_err();
    assert_eq!(
        error.class(),
        crate::ledger::model::LedgerErrorClass::CorruptDatabase
    );
}

fn table_count(connection: &Connection, table: &str) -> i64 {
    connection
        .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
            row.get(0)
        })
        .unwrap()
}

fn run_single_anchor_retention(
    activated: &mut ActivatedLedger,
    created_at_unix_ms: i64,
) -> RetentionAck {
    activated.repository.max_evidence_records = 1;
    activated
        .repository
        .run_retention(
            &RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), created_at_unix_ms).unwrap(),
        )
        .unwrap()
}

fn failed_state_counts(repository: &LedgerRepository, link_id: Uuid) -> (i64, i64) {
    (
        repository
            .connection
            .query_row(
                "SELECT count(*) FROM vector_materialization_job_state_events AS s
                 JOIN vector_materialization_jobs AS m
                   ON m.vector_materialization_job_id = s.vector_materialization_job_id
                 WHERE m.evidence_vector_link_id = ?1 AND s.state = 'failed_embedding'",
                params![link_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        repository
            .connection
            .query_row(
                "SELECT count(*) FROM evidence_vector_link_state_events
                 WHERE evidence_vector_link_id = ?1 AND state = 'failed_embedding'",
                params![link_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
    )
}

#[test]
fn background_discovery_is_bounded_keyset_ordered_and_cache_gated() {
    let (_temporary, mut activated) = activate(true, "background-discovery");
    let pending = seed_pending_materialization(&mut activated);
    let project_uuid = activated.identity.project_uuid;

    let embedding = select_embedding_work(
        &activated.repository.connection,
        project_uuid,
        &activated.identity.config_generation_id,
        31,
        None,
        1,
    )
    .unwrap();
    assert_eq!(embedding.len(), 1);
    assert_eq!(embedding[0].job.embedding_job_id, pending.embedding_job_id);
    assert_eq!(
        embedding[0].canonical_query.artifact.canonical_query_hash,
        pending.canonical_query_hash
    );
    assert!(
        select_embedding_work(
            &activated.repository.connection,
            project_uuid,
            &activated.identity.config_generation_id,
            31,
            Some(&embedding[0].cursor()),
            1,
        )
        .unwrap()
        .is_empty()
    );
    assert!(
        select_materialization_work(&activated.repository.connection, project_uuid, 31, None, 1,)
            .unwrap()
            .is_empty()
    );

    complete_pending_embedding(&mut activated, &pending);
    assert!(
        select_materialization_work(&activated.repository.connection, project_uuid, 33, None, 1)
            .unwrap()
            .is_empty()
    );
    activate_empty_vector_generation(&mut activated, &pending.vector_space_id, 33);
    let materialization =
        select_materialization_work(&activated.repository.connection, project_uuid, 36, None, 1)
            .unwrap();
    assert_eq!(materialization.len(), 1);
    assert_eq!(
        materialization[0].job.materialization_job_id,
        pending.materialization_job_id
    );
    assert!(
        select_materialization_work(
            &activated.repository.connection,
            project_uuid,
            36,
            Some(&materialization[0].cursor()),
            1,
        )
        .unwrap()
        .is_empty()
    );
}

#[test]
fn background_failure_discovery_returns_exact_next_dependent_window() {
    let (_temporary, mut activated) = activate(true, "background-failure-discovery");
    let pending = seed_pending_materialization(&mut activated);
    let lease_token = Uuid::now_v7();
    let claim = EmbeddingJobBatchClaim::new(
        Uuid::now_v7(),
        pending.vector_space_id.clone(),
        lease_token,
        31,
        vec![
            EmbeddingJobBatchClaimItem::new(
                pending.embedding_job_id.clone(),
                Uuid::now_v7(),
                Uuid::now_v7(),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let lease = match activated
        .repository
        .claim_embedding_job_batch(&claim)
        .unwrap()
    {
        EmbeddingJobBatchClaimAck::Claimed(mut leases) if leases.len() == 1 => leases.remove(0),
        acknowledgement => panic!("unexpected embedding claim: {acknowledgement:?}"),
    };
    let resolution = EmbeddingJobResolution::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        lease.job.embedding_job_id.clone(),
        lease_token,
        lease.job.attempt_generation,
        lease.job.content_hash.clone(),
        32,
        EmbeddingJobResolutionKind::TerminalFailure {
            stable_error_class: "invalid_response".to_string(),
        },
    )
    .unwrap();
    assert!(matches!(
        activated
            .repository
            .resolve_embedding_job(&resolution)
            .unwrap(),
        EmbeddingJobResolutionAck::Applied {
            state: EmbeddingJobResolvedState::TerminalFailure,
            ..
        }
    ));

    let candidates = select_failure_propagation_work(
        &activated.repository.connection,
        activated.identity.project_uuid,
        None,
        1,
    )
    .unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].evidence_vector_link_ids,
        vec![pending.evidence_vector_link_id]
    );
    assert!(
        select_failure_propagation_work(
            &activated.repository.connection,
            activated.identity.project_uuid,
            Some(&candidates[0].cursor()),
            1,
        )
        .unwrap()
        .is_empty()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn supervised_materialization_reclaims_restart_work_and_attaches_the_active_index() {
    let temporary = tempdir().unwrap();
    let path = database_path(&temporary);
    let config = router_config(&path, "supervised-materialization", true);
    let mut origin = LedgerRepository::activate_at(&config, 0).unwrap();
    let pending = seed_pending_materialization(&mut origin);
    activate_empty_vector_generation(&mut origin, &pending.vector_space_id, 31);
    complete_pending_embedding(&mut origin, &pending);
    drop(origin);

    let now = Utc::now().timestamp_millis().max(60_000);
    let activated = LedgerRepository::activate_at(&config, now).unwrap();
    let candidate = select_materialization_work(
        &activated.repository.connection,
        activated.identity.project_uuid,
        now,
        None,
        1,
    )
    .unwrap()
    .pop()
    .expect("restart must discover cache-ready materialization");
    let ActivatedLedger { repository, .. } = activated;
    let (writer_owner, writer) = LedgerWriterOwner::start(repository, 16).unwrap();
    let (_cancel, cancellation) = tokio::sync::watch::channel(false);

    assert_eq!(
        execute_materialization(&writer, candidate, cancellation).await,
        BackgroundJobOutcome::Applied
    );

    let connection = Connection::open(&path).unwrap();
    let latest_state = connection
        .query_row(
            "SELECT state FROM vector_materialization_job_state_events
             WHERE vector_materialization_job_id = ?1
             ORDER BY event_seq DESC LIMIT 1",
            params![pending.materialization_job_id],
            |row| row.get::<_, String>(0),
        )
        .unwrap();
    assert_eq!(latest_state, "ready");
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM vector_source_change_events
                 WHERE vector_space_id = ?1 AND operation = 'insert'",
                params![pending.vector_space_id.as_str()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
    writer_owner.abort();
}

#[tokio::test(flavor = "current_thread")]
async fn current_space_backfill_is_bounded_atomic_and_idempotent() {
    let temporary = tempdir().unwrap();
    let path = database_path(&temporary);
    let project_id = "current-space-backfill";
    let disabled_config = router_config(&path, project_id, false);
    let mut origin = LedgerRepository::activate_at(&disabled_config, 0).unwrap();
    let attempt = attempt(true);
    let reservation = seed_started_attempt(&mut origin, &attempt);
    assert_eq!(
        origin
            .repository
            .record_shadow_terminal(&terminal(&reservation, &attempt))
            .unwrap(),
        ShadowCommandAck::Applied
    );
    assert_eq!(
        origin
            .repository
            .stop_process(ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), 31).unwrap())
            .unwrap(),
        ProcessCommandAck::Applied
    );
    drop(origin);

    let enabled_config = router_config(&path, project_id, true);
    let mut current = LedgerRepository::activate_at(&enabled_config, 40).unwrap();
    let candidates = select_backfill_work(
        &current.repository.connection,
        current.identity.project_uuid,
        &current.identity.config_generation_id,
        None,
        1,
    )
    .unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].shadow_attempt_id, attempt.shadow_attempt_id);
    assert_eq!(candidates[0].terminal_at_unix_ms, 30);
    assert_ne!(
        candidates[0].mapping.config_generation_id,
        reservation.config_generation_id
    );
    let read_pool = LedgerReadPool::open(&path).unwrap();
    assert_eq!(
        read_pool
            .select_backfill_work_until(
                current.identity.project_uuid,
                current.identity.config_generation_id.clone(),
                None,
                1,
                Instant::now() + Duration::from_secs(2),
            )
            .await
            .unwrap(),
        candidates
    );
    assert!(
        select_backfill_work(
            &current.repository.connection,
            current.identity.project_uuid,
            &current.identity.config_generation_id,
            Some(&candidates[0].cursor()),
            1,
        )
        .unwrap()
        .is_empty()
    );

    let command = VectorBackfillCommand::new(
        candidates[0].mapping.clone(),
        attempt.shadow_attempt_id,
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        41,
    )
    .unwrap();
    assert_eq!(
        current.repository.backfill_vector_graph(&command).unwrap(),
        VectorBackfillAck::Applied
    );
    let snapshot = graph_snapshot(&current.repository.connection);
    assert_eq!(
        current.repository.backfill_vector_graph(&command).unwrap(),
        VectorBackfillAck::AlreadyApplied
    );
    assert_eq!(snapshot, graph_snapshot(&current.repository.connection));
    assert!(
        select_backfill_work(
            &current.repository.connection,
            current.identity.project_uuid,
            &current.identity.config_generation_id,
            None,
            1,
        )
        .unwrap()
        .is_empty()
    );
    assert_eq!(
        table_count(&current.repository.connection, "vectorization_outcomes"),
        1
    );
    assert_eq!(
        table_count(&current.repository.connection, "evidence_vector_links"),
        1
    );
    assert_eq!(
        table_count(
            &current.repository.connection,
            "vector_materialization_jobs"
        ),
        1
    );
    let current_space = &current.identity.pools["pool-a"]
        .vector_space
        .as_ref()
        .unwrap()
        .vector_space_id;
    assert_eq!(
        authoritative_source_fingerprint(&current.repository.connection, current_space)
            .unwrap()
            .record_count(),
        0
    );
    current.repository.max_evidence_records = 1;
    assert!(matches!(
        current
            .repository
            .run_retention(&RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), 50).unwrap())
            .unwrap(),
        RetentionAck::Applied { .. }
    ));
    assert_eq!(
        table_count(&current.repository.connection, "evidence_vector_links"),
        0
    );
    read_pool
        .close(Instant::now() + Duration::from_secs(2))
        .await
        .unwrap();
}

#[test]
fn marked_anchor_is_excluded_and_rejected_before_cross_space_backfill() {
    let temporary = tempdir().unwrap();
    let path = database_path(&temporary);
    let project_id = "marked-cross-space-backfill";
    let disabled_config = router_config(&path, project_id, false);
    let mut origin = LedgerRepository::activate_at(&disabled_config, 0).unwrap();
    let attempt = attempt(true);
    let reservation = seed_started_attempt(&mut origin, &attempt);
    assert_eq!(
        origin
            .repository
            .record_shadow_terminal(&terminal(&reservation, &attempt))
            .unwrap(),
        ShadowCommandAck::Applied
    );
    assert_eq!(
        origin
            .repository
            .stop_process(ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), 31).unwrap())
            .unwrap(),
        ProcessCommandAck::Applied
    );
    drop(origin);

    let enabled_config = router_config(&path, project_id, true);
    let mut current = LedgerRepository::activate_at(&enabled_config, 40).unwrap();
    let candidate = select_backfill_work(
        &current.repository.connection,
        current.identity.project_uuid,
        &current.identity.config_generation_id,
        None,
        1,
    )
    .unwrap()
    .remove(0);
    let command = VectorBackfillCommand::new(
        candidate.mapping,
        attempt.shadow_attempt_id,
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        42,
    )
    .unwrap();
    let retention_batch_id = Uuid::now_v7();
    let marker_hash = canonical_sha256(&json!({
        "anchor_id": reservation.anchor_id,
        "project_uuid": current.identity.project_uuid,
        "first_retention_batch_id": retention_batch_id,
        "age_expired": true,
        "count_excess": false,
        "created_at_unix_ms": 41,
    }))
    .unwrap();
    current
        .repository
        .connection
        .execute(
            "INSERT INTO retention_batches (
                retention_batch_id, conflict_health_event_id, project_uuid,
                process_instance_id, summary_shape_version, age_expired,
                count_excess, selected_count, selection_lower_bound_unix_ms,
                selection_upper_bound_unix_ms, selection_hash,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, 3, 1, 0, 1, 41, 41, ?5, 41, ?6)",
            params![
                retention_batch_id.to_string(),
                Uuid::now_v7().to_string(),
                current.identity.project_uuid.to_string(),
                current.identity.process_instance_id.to_string(),
                "a".repeat(64),
                "b".repeat(64),
            ],
        )
        .unwrap();
    current
        .repository
        .connection
        .execute(
            "INSERT INTO decision_retiring_anchors (
                anchor_id, project_uuid, first_retention_batch_id, age_expired,
                count_excess, created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, 1, 0, 41, ?4)",
            params![
                reservation.anchor_id.to_string(),
                current.identity.project_uuid.to_string(),
                retention_batch_id.to_string(),
                marker_hash,
            ],
        )
        .unwrap();

    assert!(
        select_backfill_work(
            &current.repository.connection,
            current.identity.project_uuid,
            &current.identity.config_generation_id,
            None,
            1,
        )
        .unwrap()
        .is_empty()
    );
    let before = graph_snapshot(&current.repository.connection);
    assert_eq!(
        current.repository.backfill_vector_graph(&command).unwrap(),
        VectorBackfillAck::NotFound
    );
    assert_eq!(graph_snapshot(&current.repository.connection), before);
    assert_eq!(
        table_count(&current.repository.connection, "vectorization_outcomes"),
        0
    );
    assert_eq!(
        table_count(&current.repository.connection, "evidence_vector_links"),
        0
    );
}

#[test]
fn selected_backfill_with_missing_or_tampered_source_is_atomic_corruption() {
    for corruption in ["missing", "tampered"] {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let project_id = format!("backfill-source-{corruption}");
        let disabled_config = router_config(&path, &project_id, false);
        let mut origin = LedgerRepository::activate_at(&disabled_config, 0).unwrap();
        let attempt = attempt(true);
        let reservation = seed_started_attempt(&mut origin, &attempt);
        assert_eq!(
            origin
                .repository
                .record_shadow_terminal(&terminal(&reservation, &attempt))
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            origin
                .repository
                .stop_process(ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), 31).unwrap())
                .unwrap(),
            ProcessCommandAck::Applied
        );
        drop(origin);

        let enabled_config = router_config(&path, &project_id, true);
        let mut current = LedgerRepository::activate_at(&enabled_config, 40).unwrap();
        let candidate = select_backfill_work(
            &current.repository.connection,
            current.identity.project_uuid,
            &current.identity.config_generation_id,
            None,
            1,
        )
        .unwrap()
        .remove(0);
        let command = VectorBackfillCommand::new(
            candidate.mapping,
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            41,
        )
        .unwrap();
        match corruption {
            "missing" => {
                current
                    .repository
                    .connection
                    .execute(
                        "DELETE FROM shadow_results WHERE shadow_attempt_id = ?1",
                        params![attempt.shadow_attempt_id.to_string()],
                    )
                    .unwrap();
            }
            "tampered" => {
                current
                    .repository
                    .connection
                    .execute(
                        "UPDATE shadow_results SET query_inputs_json = '{}'
                         WHERE shadow_attempt_id = ?1",
                        params![attempt.shadow_attempt_id.to_string()],
                    )
                    .unwrap();
            }
            _ => unreachable!(),
        }
        let before = graph_snapshot(&current.repository.connection);

        let error = current
            .repository
            .backfill_vector_graph(&command)
            .unwrap_err();
        assert_eq!(
            error.class(),
            crate::ledger::model::LedgerErrorClass::CorruptDatabase,
            "{corruption}"
        );
        assert_eq!(
            graph_snapshot(&current.repository.connection),
            before,
            "{corruption}"
        );
        for table in [
            "vectorization_outcomes",
            "evidence_vector_links",
            "vector_materialization_jobs",
        ] {
            assert_eq!(
                table_count(&current.repository.connection, table),
                0,
                "{table}"
            );
        }
        assert_eq!(
            current
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM vectorization_outcomes
                     WHERE stable_reason = 'source_expired'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
    }
}

#[test]
fn current_space_backfill_records_noncanonical_terminal_without_link_work() {
    let temporary = tempdir().unwrap();
    let path = database_path(&temporary);
    let project_id = "noncanonical-backfill";
    let disabled_config = router_config(&path, project_id, false);
    let mut origin = LedgerRepository::activate_at(&disabled_config, 0).unwrap();
    let attempt = attempt(true);
    let reservation = seed_started_attempt(&mut origin, &attempt);
    let mut retained = terminal(&reservation, &attempt);
    retained.vector_source = ShadowVectorSourceV1::Noncanonicalizable {
        reason: NoncanonicalizableReason::new("missing_query_inputs").unwrap(),
    };
    assert_eq!(
        origin.repository.record_shadow_terminal(&retained).unwrap(),
        ShadowCommandAck::Applied
    );
    assert_eq!(
        origin
            .repository
            .stop_process(ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), 31).unwrap())
            .unwrap(),
        ProcessCommandAck::Applied
    );
    drop(origin);

    let enabled_config = router_config(&path, project_id, true);
    let mut current = LedgerRepository::activate_at(&enabled_config, 40).unwrap();
    let candidate = select_backfill_work(
        &current.repository.connection,
        current.identity.project_uuid,
        &current.identity.config_generation_id,
        None,
        1,
    )
    .unwrap()
    .remove(0);
    let command = VectorBackfillCommand::new(
        candidate.mapping,
        attempt.shadow_attempt_id,
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        41,
    )
    .unwrap();
    assert_eq!(
        current.repository.backfill_vector_graph(&command).unwrap(),
        VectorBackfillAck::Applied
    );
    assert_eq!(
        current
            .repository
            .connection
            .query_row(
                "SELECT outcome, stable_reason FROM vectorization_outcomes",
                [],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .unwrap(),
        (
            "noncanonicalizable".to_string(),
            "missing_query_inputs".to_string()
        )
    );
    assert_eq!(
        current.repository.backfill_vector_graph(&command).unwrap(),
        VectorBackfillAck::AlreadyApplied
    );
    assert_eq!(
        table_count(&current.repository.connection, "evidence_vector_links"),
        0
    );
    assert_eq!(
        table_count(
            &current.repository.connection,
            "vector_materialization_jobs"
        ),
        0
    );
}

fn graph_snapshot(connection: &Connection) -> Vec<(String, Vec<Vec<Value>>)> {
    VECTOR_GRAPH_TABLES
        .iter()
        .map(|table| {
            let mut statement = connection
                .prepare(&format!("SELECT * FROM {table} ORDER BY rowid"))
                .unwrap();
            let column_count = statement.column_count();
            let rows = statement
                .query_map([], |row| {
                    (0..column_count)
                        .map(|index| row.get::<_, Value>(index))
                        .collect::<rusqlite::Result<Vec<_>>>()
                })
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            ((*table).to_string(), rows)
        })
        .collect()
}

#[test]
fn learning_disabled_terminal_creates_no_vector_graph_rows() {
    let (_temporary, mut activated) = activate(false, "materialization-disabled");
    let attempt = attempt(true);
    let reservation = seed_started_attempt(&mut activated, &attempt);

    assert_eq!(
        activated
            .repository
            .record_shadow_terminal(&terminal(&reservation, &attempt))
            .unwrap(),
        ShadowCommandAck::Applied
    );
    for table in VECTOR_GRAPH_TABLES {
        assert_eq!(
            table_count(&activated.repository.connection, table),
            0,
            "{table}"
        );
    }
}

#[test]
fn canonical_cache_miss_creates_the_exact_initial_graph() {
    let (_temporary, mut activated) = activate(true, "materialization-cache-miss");
    let attempt = attempt(true);
    let reservation = seed_started_attempt(&mut activated, &attempt);
    let vectorization = atomic_vectorization();
    let expected = match &vectorization {
        ShadowVectorizationHandoff::Atomic(value) => value.clone(),
        ShadowVectorizationHandoff::Disabled | ShadowVectorizationHandoff::DeferredBackfill => {
            unreachable!()
        }
    };
    let terminal = terminal(&reservation, &attempt).with_vectorization(vectorization);

    assert_eq!(
        activated
            .repository
            .record_shadow_terminal(&terminal)
            .unwrap(),
        ShadowCommandAck::Applied
    );
    for table in [
        "canonical_routing_queries",
        "routing_partitions",
        "vectorization_outcomes",
        "embedding_jobs",
        "embedding_job_state_events",
        "evidence_vector_links",
        "evidence_vector_link_state_events",
        "vector_materialization_jobs",
        "vector_materialization_job_state_events",
    ] {
        assert_eq!(
            table_count(&activated.repository.connection, table),
            1,
            "{table}"
        );
    }
    assert_eq!(
        table_count(&activated.repository.connection, "embeddings"),
        0
    );
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "vector_source_change_events"
        ),
        0
    );

    let graph: (
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        Option<String>,
    ) = activated
        .repository
        .connection
        .query_row(
            "SELECT o.outcome, l.evidence_vector_link_id, ls.state,
                        es.embedding_job_state_event_id, es.state, ms.state,
                        m.embedding_job_id, m.embedding_id
                 FROM vectorization_outcomes AS o
                 JOIN evidence_vector_links AS l
                   ON l.vectorization_outcome_id = o.vectorization_outcome_id
                 JOIN evidence_vector_link_state_events AS ls
                   ON ls.evidence_vector_link_id = l.evidence_vector_link_id
                 JOIN vector_materialization_jobs AS m
                   ON m.evidence_vector_link_id = l.evidence_vector_link_id
                 JOIN vector_materialization_job_state_events AS ms
                   ON ms.vector_materialization_job_id = m.vector_materialization_job_id
                 JOIN embedding_job_state_events AS es
                   ON es.embedding_job_id = m.embedding_job_id",
            [],
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
    assert_eq!(graph.0, "canonicalized");
    assert_eq!(graph.1, expected.evidence_vector_link_id.to_string());
    assert_eq!(graph.2, "pending_embedding");
    assert_eq!(graph.3, expected.embedding_job_state_event_id.to_string());
    assert_eq!(graph.4, "pending");
    assert_eq!(graph.5, "pending_embedding");
    assert_eq!(graph.6.len(), 64);
    assert_eq!(graph.7, None);
}

#[test]
fn canonicalizer_rejection_creates_only_an_outcome() {
    let (_temporary, mut activated) = activate(true, "materialization-rejection");
    let attempt = attempt(false);
    let reservation = seed_started_attempt(&mut activated, &attempt);
    let terminal = terminal(&reservation, &attempt).with_vectorization(atomic_vectorization());

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
            .connection
            .query_row(
                "SELECT outcome || ':' || stable_reason FROM vectorization_outcomes",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "noncanonicalizable:router.ineligible.canonical_missing_task"
    );
    for table in VECTOR_GRAPH_TABLES {
        let expected = i64::from(*table == "vectorization_outcomes");
        assert_eq!(
            table_count(&activated.repository.connection, table),
            expected,
            "{table}"
        );
    }
}

#[test]
fn exact_retry_is_data_identical_and_alternate_frozen_ids_conflict() {
    let (_temporary, mut activated) = activate(true, "materialization-idempotency");
    let attempt = attempt(true);
    let reservation = seed_started_attempt(&mut activated, &attempt);
    let terminal = terminal(&reservation, &attempt).with_vectorization(atomic_vectorization());
    assert_eq!(
        activated
            .repository
            .record_shadow_terminal(&terminal)
            .unwrap(),
        ShadowCommandAck::Applied
    );
    let applied = graph_snapshot(&activated.repository.connection);

    assert_eq!(
        activated
            .repository
            .record_shadow_terminal(&terminal)
            .unwrap(),
        ShadowCommandAck::AlreadyApplied
    );
    assert_eq!(graph_snapshot(&activated.repository.connection), applied);

    let alternate = terminal.clone().with_vectorization(atomic_vectorization());
    assert_eq!(
        activated
            .repository
            .record_shadow_terminal(&alternate)
            .unwrap(),
        ShadowCommandAck::Conflict
    );
    assert_eq!(graph_snapshot(&activated.repository.connection), applied);
}

#[test]
fn tampered_graph_conflicts_without_partial_retry_rows() {
    let (_temporary, mut activated) = activate(true, "materialization-tamper");
    let attempt = attempt(true);
    let reservation = seed_started_attempt(&mut activated, &attempt);
    let terminal = terminal(&reservation, &attempt).with_vectorization(atomic_vectorization());
    assert_eq!(
        activated
            .repository
            .record_shadow_terminal(&terminal)
            .unwrap(),
        ShadowCommandAck::Applied
    );
    activated
        .repository
        .connection
        .execute(
            "UPDATE evidence_vector_links SET canonical_payload_hash = ?1",
            params!["f".repeat(64)],
        )
        .unwrap();
    let tampered = graph_snapshot(&activated.repository.connection);

    assert_eq!(
        activated
            .repository
            .record_shadow_terminal(&terminal)
            .unwrap(),
        ShadowCommandAck::Conflict
    );
    assert_eq!(graph_snapshot(&activated.repository.connection), tampered);
    assert_eq!(
        activated
            .repository
            .connection
            .query_row(
                "SELECT count(*) FROM health_events
             WHERE stable_class = 'router.ledger.integrity_conflict'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
}

#[test]
fn attempts_with_the_same_query_share_one_query_and_embedding_job() {
    let (temporary, mut first_process) = activate(true, "materialization-shared-job");
    let first_attempt = attempt(true);
    let first_reservation = seed_started_attempt(&mut first_process, &first_attempt);
    let first_terminal =
        terminal(&first_reservation, &first_attempt).with_vectorization(atomic_vectorization());
    assert_eq!(
        first_process
            .repository
            .record_shadow_terminal(&first_terminal)
            .unwrap(),
        ShadowCommandAck::Applied
    );
    let first_process_id = first_process.identity.process_instance_id;
    drop(first_process);

    let path = database_path(&temporary);
    let mut activated =
        LedgerRepository::activate_at(&router_config(&path, "materialization-shared-job", true), 0)
            .unwrap();
    assert_ne!(activated.identity.process_instance_id, first_process_id);
    let second_attempt = attempt(true);
    let second_reservation = seed_started_attempt(&mut activated, &second_attempt);
    let mut second_terminal =
        terminal(&second_reservation, &second_attempt).with_vectorization(atomic_vectorization());
    second_terminal.created_at_unix_ms = 31;
    second_terminal
        .batch_terminal
        .as_mut()
        .unwrap()
        .created_at_unix_ms = 31;
    assert_eq!(
        activated
            .repository
            .record_shadow_terminal(&second_terminal)
            .unwrap(),
        ShadowCommandAck::Applied
    );

    assert_eq!(
        table_count(
            &activated.repository.connection,
            "canonical_routing_queries"
        ),
        1
    );
    assert_eq!(
        table_count(&activated.repository.connection, "embedding_jobs"),
        1
    );
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "embedding_job_state_events"
        ),
        1
    );
    assert_eq!(
        table_count(&activated.repository.connection, "vectorization_outcomes"),
        2
    );
    assert_eq!(
        table_count(&activated.repository.connection, "evidence_vector_links"),
        2
    );
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "vector_materialization_jobs"
        ),
        2
    );
    assert_eq!(
        activated
            .repository
            .connection
            .query_row(
                "SELECT count(DISTINCT embedding_job_id) FROM vector_materialization_jobs",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
    assert!(matches!(
        run_single_anchor_retention(&mut activated, 40),
        RetentionAck::Applied { .. }
    ));
    assert_eq!(table_count(&activated.repository.connection, "anchors"), 0);
}

#[test]
fn cache_hit_without_active_manifest_starts_pending_index() {
    let (_temporary, mut activated) = activate(true, "materialization-cache-hit");
    let attempt = attempt(true);
    let reservation = seed_started_attempt(&mut activated, &attempt);
    let query = build_canonical_routing_query(
        &attempt.request_projection,
        &routing_projection(),
        &config(Path::new("router.db"), "cache-hit").pools[0].canonicalizer,
    )
    .unwrap();
    let project_uuid = activated.identity.project_uuid;
    let vector_space_id = activated.identity.pools["pool-a"]
        .vector_space
        .as_ref()
        .unwrap()
        .vector_space_id
        .clone();
    let dimensions = activated
        .repository
        .connection
        .query_row(
            "SELECT dimensions FROM vector_spaces WHERE vector_space_id = ?1",
            params![vector_space_id.as_str()],
            |row| row.get::<_, u32>(0),
        )
        .unwrap();
    let dimensions = crate::vector::VectorDimensions::new(dimensions).unwrap();
    let mut values = vec![0.0; dimensions.as_usize()];
    values[0] = 1.0;
    let vector = AuthoritativeVector::from_normalized(
        &vector_space_id,
        NormalizedVector::from_provider_f64(&values, dimensions).unwrap(),
    )
    .unwrap();
    let embedding_id = Uuid::now_v7();
    let transaction = activated.repository.connection_mut().transaction().unwrap();
    assert!(matches!(
        ensure_canonical_query(&transaction, &query, 25).unwrap(),
        CanonicalQueryEnsureAck::Applied(_)
    ));
    assert!(matches!(
        upsert_embedding_cache(
            &transaction,
            &EmbeddingCacheWrite {
                embedding_id,
                project_uuid,
                vector_space_id,
                canonical_query_hash: query.canonical_query_hash.clone(),
                content_hash: query.canonical_query_hash,
                vector,
                source: EmbeddingCacheSource::Provider,
                created_at_unix_ms: 25,
            },
        )
        .unwrap(),
        EmbeddingCacheUpsertAck::Applied(_)
    ));
    transaction.commit().unwrap();

    let terminal = terminal(&reservation, &attempt).with_vectorization(atomic_vectorization());
    assert_eq!(
        activated
            .repository
            .record_shadow_terminal(&terminal)
            .unwrap(),
        ShadowCommandAck::Applied
    );
    assert_eq!(
        table_count(&activated.repository.connection, "embedding_jobs"),
        0
    );
    assert_eq!(
        activated
            .repository
            .connection
            .query_row(
                "SELECT ls.state || ':' || ms.state
             FROM evidence_vector_link_state_events AS ls
             JOIN vector_materialization_jobs AS m
               ON m.evidence_vector_link_id = ls.evidence_vector_link_id
             JOIN vector_materialization_job_state_events AS ms
               ON ms.vector_materialization_job_id = m.vector_materialization_job_id
             WHERE ls.embedding_id = ?1 AND m.embedding_id = ?1",
                params![embedding_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "pending_index:pending_index"
    );
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "vector_source_change_events"
        ),
        0
    );
}

#[test]
fn completed_quality_labels_require_promotion_eligible_binary_evaluations() {
    for (name, response_score, trajectory_score, confidence, evaluation, expected_label) in [
        (
            "pass",
            0.9,
            0.9,
            0.9,
            ("pass", Some("pass"), 1_i64),
            Some("pass"),
        ),
        (
            "fail",
            0.6,
            0.6,
            0.9,
            ("fail", Some("fail"), 1_i64),
            Some("fail"),
        ),
        ("ambiguous", 0.9, 0.9, 0.5, ("ambiguous", None, 0_i64), None),
    ] {
        let (_temporary, mut activated) = activate(true, &format!("materialization-{name}"));
        let attempt = attempt(true);
        let reservation = seed_started_attempt(&mut activated, &attempt);
        let (evaluation_id, candidate_response) = seed_judge_evaluation(
            &mut activated,
            &attempt,
            response_score,
            trajectory_score,
            confidence,
        );
        let stored_evaluation: (String, Option<String>, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT label, binary_label, promotion_eligible
                 FROM evaluations WHERE evaluation_id = ?1",
                params![evaluation_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(stored_evaluation.0, evaluation.0, "{name}");
        assert_eq!(stored_evaluation.1.as_deref(), evaluation.1, "{name}");
        assert_eq!(stored_evaluation.2, evaluation.2, "{name}");

        let terminal = terminal_for_class(
            &reservation,
            &attempt,
            ShadowTerminalClass::Completed,
            None,
            Some(candidate_response),
            None,
            Some(evaluation_id),
        )
        .with_vectorization(atomic_vectorization());
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Applied,
            "{name}"
        );
        assert_eq!(
            stored_link_fact(&activated.repository),
            ("completed".to_string(), expected_label.map(str::to_string)),
            "{name}"
        );
    }
}

#[test]
fn non_completed_terminal_classes_persist_only_authorized_labels() {
    for (name, terminal_class, started, expected_label) in [
        (
            "deterministic-failure",
            ShadowTerminalClass::DeterministicFailure,
            true,
            Some("fail"),
        ),
        (
            "operational-failure",
            ShadowTerminalClass::OperationalFailure,
            true,
            None,
        ),
        (
            "skipped-cooloff",
            ShadowTerminalClass::SkippedCooloff,
            false,
            None,
        ),
        (
            "canceled-shutdown",
            ShadowTerminalClass::CanceledShutdown,
            false,
            None,
        ),
    ] {
        let (_temporary, mut activated) = activate(true, &format!("materialization-{name}"));
        let attempt = attempt(true);
        let reservation = seed_attempt(&mut activated, &attempt, started);
        let (deterministic_hard_failure, evaluation_id) =
            if terminal_class == ShadowTerminalClass::DeterministicFailure {
                let failure = DeterministicHardFailureV1::ResponseSchema;
                let evaluation_id = Uuid::now_v7();
                let router_config = config(Path::new("router.db"), "deterministic-label");
                let evaluation = EvaluationRecord::deterministic(
                    evaluation_id,
                    attempt.shadow_attempt_id,
                    attempt.evaluator_version.clone(),
                    Uuid::now_v7(),
                    failure,
                    false,
                    &router_config.pools[0].judge,
                    23,
                )
                .unwrap();
                assert_eq!(
                    activated.repository.record_evaluation(&evaluation).unwrap(),
                    JudgeRecordAck::Applied
                );
                (Some(failure), Some(evaluation_id))
            } else {
                (None, None)
            };
        let terminal = terminal_for_class(
            &reservation,
            &attempt,
            terminal_class,
            None,
            None,
            deterministic_hard_failure,
            evaluation_id,
        )
        .with_vectorization(atomic_vectorization());
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Applied,
            "{name}"
        );
        assert_eq!(
            stored_link_fact(&activated.repository),
            (
                terminal_class.as_str().to_string(),
                expected_label.map(str::to_string),
            ),
            "{name}"
        );
    }
}

#[test]
fn orphan_terminal_classes_create_unlabeled_vector_links_under_reconciler_authority() {
    for (name, terminal_class, started) in [
        (
            "orphaned-before-schedule",
            ShadowTerminalClass::OrphanedBeforeSchedule,
            false,
        ),
        (
            "orphaned-in-flight",
            ShadowTerminalClass::OrphanedInFlight,
            true,
        ),
    ] {
        let (temporary, mut origin) = activate(true, &format!("materialization-{name}"));
        let path = database_path(&temporary);
        let router_config = router_config(&path, &format!("materialization-{name}"), true);
        let mut reconciler = LedgerRepository::activate_at(&router_config, 0).unwrap();
        let attempt = attempt(true);
        let reservation = seed_attempt(&mut origin, &attempt, started);
        let dead_process_instance_id = origin.identity.process_instance_id;
        assert_eq!(
            origin
                .repository
                .stop_process(ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), 25).unwrap())
                .unwrap(),
            ProcessCommandAck::Applied
        );
        let terminal = terminal_for_class(
            &reservation,
            &attempt,
            terminal_class,
            Some(dead_process_instance_id),
            None,
            None,
            None,
        )
        .with_vectorization(atomic_vectorization());
        assert_eq!(
            reconciler
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Applied,
            "{name}"
        );
        assert_eq!(
            stored_link_fact(&reconciler.repository),
            (terminal_class.as_str().to_string(), None),
            "{name}"
        );
    }
}

#[test]
fn materialization_claim_waits_for_authoritative_embedding_cache() {
    let (_temporary, mut activated) = activate(true, "materialization-cache-not-ready");
    let pending = seed_pending_materialization(&mut activated);
    let claim = materialization_claim(
        &pending.materialization_job_id,
        pending.attempt_generation,
        &pending.canonical_payload_hash,
        Uuid::now_v7(),
        31,
    );

    assert_eq!(
        activated.repository.claim_materialization(&claim).unwrap(),
        MaterializationClaimAck::CacheNotReady
    );
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "vector_materialization_job_state_events"
        ),
        1
    );
}

#[test]
fn cache_backed_claim_waits_for_an_active_index_without_state_growth() {
    let (_temporary, mut activated) = activate(true, "materialization-index-not-ready");
    let pending = seed_pending_materialization(&mut activated);
    complete_pending_embedding(&mut activated, &pending);
    let claim = materialization_claim(
        &pending.materialization_job_id,
        pending.attempt_generation,
        &pending.canonical_payload_hash,
        Uuid::now_v7(),
        40,
    );
    let before = graph_snapshot(&activated.repository.connection);

    assert_eq!(
        activated.repository.claim_materialization(&claim).unwrap(),
        MaterializationClaimAck::IndexNotReady
    );
    assert_eq!(graph_snapshot(&activated.repository.connection), before);
    assert!(
        select_materialization_work(
            &activated.repository.connection,
            activated.identity.project_uuid,
            40,
            None,
            1,
        )
        .unwrap()
        .is_empty()
    );

    activate_empty_vector_generation(&mut activated, &pending.vector_space_id, 34);
    assert!(matches!(
        activated.repository.claim_materialization(&claim).unwrap(),
        MaterializationClaimAck::Claimed(_)
    ));
}

#[test]
fn materialization_discovery_filters_an_active_manifest_with_missing_objects() {
    let (_temporary, mut activated) = activate(true, "materialization-missing-index-object");
    let pending = seed_pending_materialization(&mut activated);
    complete_pending_embedding(&mut activated, &pending);
    let root = activate_empty_vector_generation(&mut activated, &pending.vector_space_id, 34);
    activated
        .repository
        .connection
        .execute_batch(&format!("DROP TABLE \"{root}_vector_chunks00\""))
        .unwrap();

    assert!(
        select_materialization_work(
            &activated.repository.connection,
            activated.identity.project_uuid,
            40,
            None,
            1,
        )
        .unwrap()
        .is_empty()
    );
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "vector_materialization_job_state_events"
        ),
        1
    );
}

#[test]
fn materialization_discovery_propagates_rehashed_manifest_corruption() {
    let (_temporary, mut activated) = activate(true, "materialization-rehashed-manifest");
    let pending = seed_pending_materialization(&mut activated);
    complete_pending_embedding(&mut activated, &pending);
    activate_empty_vector_generation(&mut activated, &pending.vector_space_id, 34);
    activated
        .repository
        .connection
        .execute_batch("PRAGMA ignore_check_constraints = ON")
        .unwrap();
    assert_eq!(
        activated
            .repository
            .connection
            .execute(
                "UPDATE vector_index_manifest
                 SET root_table_name = 'router_vec_ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff_g1'
                 WHERE vector_space_id = ?1 AND state = 'active'",
                params![pending.vector_space_id.as_str()],
            )
            .unwrap(),
        1
    );
    rehash_vector_index_manifest(&activated.repository.connection, &pending.vector_space_id);

    let error = select_materialization_work(
        &activated.repository.connection,
        activated.identity.project_uuid,
        40,
        None,
        1,
    )
    .unwrap_err();
    assert_eq!(error.class(), LedgerErrorClass::CorruptDatabase);
}

#[test]
fn rehashed_failed_index_below_attempt_limit_is_corruption() {
    const ATTEMPT_LIMIT_CLASS: &str = "router.vector.materialization_attempt_limit";

    let (_temporary, mut activated) = activate(true, "materialization-early-failed-index");
    let pending = seed_pending_materialization(&mut activated);
    let embedding_id = complete_pending_embedding(&mut activated, &pending);
    let (next_eligible_at_unix_ms, created_at_unix_ms) = activated
        .repository
        .connection
        .query_row(
            "SELECT next_eligible_at_unix_ms, created_at_unix_ms
             FROM vector_materialization_jobs
             WHERE vector_materialization_job_id = ?1",
            params![pending.materialization_job_id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        )
        .unwrap();
    let job_hash = canonical_sha256(&json!({
        "vector_materialization_job_id": pending.materialization_job_id,
        "evidence_vector_link_id": pending.evidence_vector_link_id,
        "vector_space_id": pending.vector_space_id.as_str(),
        "canonical_query_hash": pending.canonical_query_hash,
        "embedding_job_id": pending.embedding_job_id,
        "embedding_id": embedding_id,
        "lease_owner_process_instance_id": Option::<Uuid>::None,
        "lease_token": Option::<Uuid>::None,
        "lease_expires_at_unix_ms": Option::<i64>::None,
        "attempt_generation": 0,
        "attempt_count": 0,
        "next_eligible_at_unix_ms": next_eligible_at_unix_ms,
        "created_at_unix_ms": created_at_unix_ms,
    }))
    .unwrap();
    assert_eq!(
        activated
            .repository
            .connection
            .execute(
                "UPDATE vector_materialization_jobs
                 SET embedding_id = ?1, canonical_payload_hash = ?2
                 WHERE vector_materialization_job_id = ?3",
                params![
                    embedding_id.to_string(),
                    job_hash,
                    pending.materialization_job_id
                ],
            )
            .unwrap(),
        1
    );

    let link_event_id = Uuid::now_v7();
    let materialization_event_id = Uuid::now_v7();
    let terminal_at_unix_ms = 33;
    let link_hash = canonical_sha256(&json!({
        "evidence_vector_link_state_event_id": link_event_id,
        "evidence_vector_link_id": pending.evidence_vector_link_id,
        "embedding_id": embedding_id,
        "state": "failed_index",
        "attempt_generation": 0,
        "stable_error_class": ATTEMPT_LIMIT_CLASS,
        "created_at_unix_ms": terminal_at_unix_ms,
    }))
    .unwrap();
    let materialization_hash = canonical_sha256(&json!({
        "vector_materialization_job_state_event_id": materialization_event_id,
        "vector_materialization_job_id": pending.materialization_job_id,
        "process_instance_id": activated.identity.process_instance_id,
        "state": "failed_index",
        "attempt_generation": 0,
        "stable_error_class": ATTEMPT_LIMIT_CLASS,
        "lease_token": Option::<Uuid>::None,
        "lease_expires_at_unix_ms": Option::<i64>::None,
        "attempt_count": 0,
        "next_eligible_at_unix_ms": next_eligible_at_unix_ms,
        "created_at_unix_ms": terminal_at_unix_ms,
    }))
    .unwrap();

    assert!(
        activated
            .repository
            .connection
            .execute(
                "INSERT INTO evidence_vector_link_state_events (
                    evidence_vector_link_state_event_id, evidence_vector_link_id,
                    embedding_id, state, attempt_generation, stable_error_class,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, 'failed_index', 0, ?4, ?5, ?6)",
                params![
                    link_event_id.to_string(),
                    pending.evidence_vector_link_id.to_string(),
                    embedding_id.to_string(),
                    ATTEMPT_LIMIT_CLASS,
                    terminal_at_unix_ms,
                    link_hash,
                ],
            )
            .is_err()
    );
    assert!(
        activated
            .repository
            .connection
            .execute(
                "INSERT INTO vector_materialization_job_state_events (
                    vector_materialization_job_state_event_id,
                    vector_materialization_job_id, process_instance_id, state,
                    attempt_generation, stable_error_class, attempt_count,
                    next_eligible_at_unix_ms, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, 'failed_index', 0, ?4, 0, ?5, ?6, ?7)",
                params![
                    materialization_event_id.to_string(),
                    pending.materialization_job_id,
                    activated.identity.process_instance_id.to_string(),
                    ATTEMPT_LIMIT_CLASS,
                    next_eligible_at_unix_ms,
                    terminal_at_unix_ms,
                    materialization_hash,
                ],
            )
            .is_err()
    );

    activated
        .repository
        .connection
        .execute_batch("PRAGMA ignore_check_constraints = ON")
        .unwrap();
    activated
        .repository
        .connection
        .execute(
            "INSERT INTO evidence_vector_link_state_events (
                evidence_vector_link_state_event_id, evidence_vector_link_id,
                embedding_id, state, attempt_generation, stable_error_class,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, 'failed_index', 0, ?4, ?5, ?6)",
            params![
                link_event_id.to_string(),
                pending.evidence_vector_link_id.to_string(),
                embedding_id.to_string(),
                ATTEMPT_LIMIT_CLASS,
                terminal_at_unix_ms,
                link_hash,
            ],
        )
        .unwrap();
    activated
        .repository
        .connection
        .execute(
            "INSERT INTO vector_materialization_job_state_events (
                vector_materialization_job_state_event_id,
                vector_materialization_job_id, process_instance_id, state,
                attempt_generation, stable_error_class, attempt_count,
                next_eligible_at_unix_ms, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, 'failed_index', 0, ?4, 0, ?5, ?6, ?7)",
            params![
                materialization_event_id.to_string(),
                pending.materialization_job_id,
                activated.identity.process_instance_id.to_string(),
                ATTEMPT_LIMIT_CLASS,
                next_eligible_at_unix_ms,
                terminal_at_unix_ms,
                materialization_hash,
            ],
        )
        .unwrap();

    let error = load_verified_materialization_job(
        &activated.repository.connection,
        activated.identity.project_uuid,
        &pending.materialization_job_id,
    )
    .unwrap_err();
    assert_eq!(error.class(), LedgerErrorClass::CorruptDatabase);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn materialization_attempt_limit_is_a_bounded_durable_terminal() {
    let (temporary, mut activated) = activate(true, "materialization-attempt-limit");
    let pending = seed_pending_materialization(&mut activated);
    complete_pending_embedding(&mut activated, &pending);
    activate_empty_vector_generation(&mut activated, &pending.vector_space_id, 30);
    let mut job = load_verified_materialization_job(
        &activated.repository.connection,
        activated.identity.project_uuid,
        &pending.materialization_job_id,
    )
    .unwrap()
    .unwrap();

    for ordinal in 0..MATERIALIZATION_ATTEMPT_MAX {
        let claimed_at = 40 + ordinal * 2;
        let token = Uuid::now_v7();
        let claim = materialization_claim(
            &job.materialization_job_id,
            job.attempt_generation,
            &job.canonical_payload_hash,
            token,
            claimed_at,
        );
        let lease = match activated.repository.claim_materialization(&claim).unwrap() {
            MaterializationClaimAck::Claimed(lease) => lease,
            acknowledgement => panic!("unexpected bounded claim: {acknowledgement:?}"),
        };
        assert_eq!(lease.job.attempt_count, ordinal + 1);
        let release = MaterializationResolution::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            lease.job.materialization_job_id.clone(),
            token,
            lease.job.attempt_generation,
            lease.job.canonical_payload_hash,
            claimed_at + 1,
            MaterializationResolutionKind::Released,
        )
        .unwrap();
        job = match activated
            .repository
            .resolve_materialization(&release)
            .unwrap()
        {
            MaterializationResolutionAck::Applied {
                state: MaterializationResolvedState::Released,
                job,
                ..
            } => job,
            acknowledgement => panic!("unexpected bounded release: {acknowledgement:?}"),
        };
    }
    assert_eq!(job.attempt_count, MATERIALIZATION_ATTEMPT_MAX);
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "vector_materialization_job_state_events"
        ),
        1 + 2 * MATERIALIZATION_ATTEMPT_MAX
    );

    let mut candidates = select_materialization_work(
        &activated.repository.connection,
        activated.identity.project_uuid,
        200,
        None,
        1,
    )
    .unwrap();
    assert_eq!(candidates.len(), 1);
    assert_eq!(
        candidates[0].job.materialization_job_id,
        job.materialization_job_id
    );
    let claim = materialization_claim(
        &job.materialization_job_id,
        job.attempt_generation,
        &job.canonical_payload_hash,
        Uuid::now_v7(),
        200,
    );
    let observed_at_unix_ms = Utc::now().timestamp_millis();
    assert!(matches!(
        activated
            .repository
            .renew_heartbeat(HeartbeatRenewal::new(observed_at_unix_ms).unwrap())
            .unwrap(),
        HeartbeatAck::Applied { .. } | HeartbeatAck::AlreadyApplied { .. }
    ));
    let project_uuid = activated.identity.project_uuid;
    let path = database_path(&temporary);
    let config = router_config(&path, "materialization-attempt-limit", true);
    let (mut owner, client) = LedgerWriterOwner::start(activated.repository, 4).unwrap();
    let (_cancel, cancellation) = tokio::sync::watch::channel(false);
    assert_eq!(
        execute_materialization(&client, candidates.remove(0), cancellation).await,
        BackgroundJobOutcome::Applied
    );
    let connection = Connection::open(&path).unwrap();
    assert_eq!(
        table_count(&connection, "vector_materialization_job_state_events"),
        2 + 2 * MATERIALIZATION_ATTEMPT_MAX
    );
    let terminal_graph = graph_snapshot(&connection);
    assert_eq!(
        client
            .claim_materialization_until(claim, Instant::now() + Duration::from_secs(2))
            .await
            .unwrap(),
        MaterializationClaimAck::Terminal
    );
    assert_eq!(graph_snapshot(&connection), terminal_graph);
    let terminal_states: (String, String, String, String) = connection
        .query_row(
            "SELECT
                (SELECT state FROM vector_materialization_job_state_events
                 ORDER BY event_seq DESC LIMIT 1),
                (SELECT stable_error_class FROM vector_materialization_job_state_events
                 ORDER BY event_seq DESC LIMIT 1),
                (SELECT state FROM evidence_vector_link_state_events
                 ORDER BY event_seq DESC LIMIT 1),
                (SELECT stable_error_class FROM evidence_vector_link_state_events
                 ORDER BY event_seq DESC LIMIT 1)",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(terminal_states.0, "failed_index");
    assert_eq!(
        terminal_states.1,
        "router.vector.materialization_attempt_limit"
    );
    assert_eq!(terminal_states.1, terminal_states.3);
    assert_eq!(terminal_states.2, "failed_index");
    assert!(
        select_materialization_work(&connection, project_uuid, 200, None, 1,)
            .unwrap()
            .is_empty()
    );

    drop(connection);
    let RetentionAck::Applied { summary, .. } = client
        .run_retention_until(
            RetentionRequest::new(
                Uuid::now_v7(),
                Uuid::now_v7(),
                Utc::now().timestamp_millis() + 1,
            )
            .unwrap(),
            Instant::now() + Duration::from_secs(2),
        )
        .await
        .unwrap()
    else {
        panic!("attempt-limit terminal should be eligible for retention");
    };
    assert_eq!(summary.selected_count, 1);
    let retained = Connection::open(&path).unwrap();
    assert_eq!(table_count(&retained, "evidence_vector_links"), 0);
    assert_eq!(table_count(&retained, "vector_materialization_jobs"), 0);
    drop(retained);
    owner
        .drain_until(Instant::now() + Duration::from_secs(2))
        .await
        .unwrap();
    let reopened = LedgerRepository::activate_at(&config, 300).unwrap();
    assert!(
        select_materialization_work(
            &reopened.repository.connection,
            reopened.identity.project_uuid,
            300,
            None,
            1,
        )
        .unwrap()
        .is_empty()
    );
}

#[test]
fn dead_owner_at_materialization_limit_is_orphaned_before_terminal() {
    let project_id = "materialization-attempt-limit-owner";
    let (temporary, mut owner) = activate(true, project_id);
    let pending = seed_pending_materialization(&mut owner);
    complete_pending_embedding(&mut owner, &pending);
    activate_empty_vector_generation(&mut owner, &pending.vector_space_id, 30);
    let mut job = load_verified_materialization_job(
        &owner.repository.connection,
        owner.identity.project_uuid,
        &pending.materialization_job_id,
    )
    .unwrap()
    .unwrap();

    for ordinal in 0..(MATERIALIZATION_ATTEMPT_MAX - 1) {
        let claimed_at = 40 + ordinal * 2;
        let token = Uuid::now_v7();
        let claim = materialization_claim(
            &job.materialization_job_id,
            job.attempt_generation,
            &job.canonical_payload_hash,
            token,
            claimed_at,
        );
        let lease = match owner.repository.claim_materialization(&claim).unwrap() {
            MaterializationClaimAck::Claimed(lease) => lease,
            acknowledgement => panic!("unexpected bounded claim: {acknowledgement:?}"),
        };
        let release = MaterializationResolution::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            lease.job.materialization_job_id.clone(),
            token,
            lease.job.attempt_generation,
            lease.job.canonical_payload_hash,
            claimed_at + 1,
            MaterializationResolutionKind::Released,
        )
        .unwrap();
        job = match owner.repository.resolve_materialization(&release).unwrap() {
            MaterializationResolutionAck::Applied { job, .. } => job,
            acknowledgement => panic!("unexpected bounded release: {acknowledgement:?}"),
        };
    }

    let final_token = Uuid::now_v7();
    let final_claim = materialization_claim(
        &job.materialization_job_id,
        job.attempt_generation,
        &job.canonical_payload_hash,
        final_token,
        200,
    );
    let final_lease = match owner
        .repository
        .claim_materialization(&final_claim)
        .unwrap()
    {
        MaterializationClaimAck::Claimed(lease) => lease,
        acknowledgement => panic!("unexpected final owner claim: {acknowledgement:?}"),
    };
    assert_eq!(final_lease.job.attempt_count, MATERIALIZATION_ATTEMPT_MAX);
    assert_eq!(
        owner
            .repository
            .stop_process(ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), 201).unwrap())
            .unwrap(),
        ProcessCommandAck::Applied
    );
    let path = database_path(&temporary);
    let config = router_config(&path, project_id, true);
    drop(owner);

    let mut reclaimer = LedgerRepository::activate_at(&config, 300).unwrap();
    let terminal_claim = materialization_claim(
        &final_lease.job.materialization_job_id,
        final_lease.job.attempt_generation,
        &final_lease.job.canonical_payload_hash,
        Uuid::now_v7(),
        301,
    );
    assert!(matches!(
        reclaimer
            .repository
            .claim_materialization(&terminal_claim)
            .unwrap(),
        MaterializationClaimAck::AttemptLimitTerminal(_)
    ));
    assert_eq!(
        table_count(
            &reclaimer.repository.connection,
            "vector_materialization_job_state_events"
        ),
        2 + 2 * MATERIALIZATION_ATTEMPT_MAX
    );
    assert_eq!(
        reclaimer
            .repository
            .connection
            .query_row(
                "SELECT count(*) FROM vector_materialization_job_state_events
                 WHERE state = 'orphaned_in_flight'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
    assert_eq!(
        reclaimer
            .repository
            .connection
            .query_row(
                "SELECT state FROM vector_materialization_job_state_events
                 ORDER BY event_seq DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "failed_index"
    );
}

#[test]
fn cache_backed_completion_is_fenced_idempotent_with_active_manifest() {
    let (_temporary, mut activated) = activate(true, "materialization-completion");
    let pending = seed_pending_materialization(&mut activated);
    let embedding_id = complete_pending_embedding(&mut activated, &pending);
    activate_empty_vector_generation(&mut activated, &pending.vector_space_id, 30);
    let lease_token = Uuid::now_v7();
    let claim = materialization_claim(
        &pending.materialization_job_id,
        pending.attempt_generation,
        &pending.canonical_payload_hash,
        lease_token,
        33,
    );
    let lease = match activated.repository.claim_materialization(&claim).unwrap() {
        MaterializationClaimAck::Claimed(lease) => lease,
        acknowledgement => panic!("unexpected materialization claim: {acknowledgement:?}"),
    };
    assert_eq!(
        activated.repository.claim_materialization(&claim).unwrap(),
        MaterializationClaimAck::AlreadyApplied(lease.clone())
    );
    let mut claim_with_wrong_hash = claim.clone();
    claim_with_wrong_hash.expected_canonical_payload_hash = "f".repeat(64);
    let before_wrong_claim_hash = graph_snapshot(&activated.repository.connection);
    assert_eq!(
        activated
            .repository
            .claim_materialization(&claim_with_wrong_hash)
            .unwrap(),
        MaterializationClaimAck::Conflict
    );
    assert_eq!(
        graph_snapshot(&activated.repository.connection),
        before_wrong_claim_hash
    );
    assert_eq!(lease.job.embedding_id, None);
    assert_eq!(lease.job.attempt_generation, 1);

    let stale_hash = MaterializationCompletion::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        lease.job.materialization_job_id.clone(),
        lease_token,
        lease.job.attempt_generation,
        "f".repeat(64),
        34,
    )
    .unwrap();
    let stale_token = MaterializationCompletion::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        lease.job.materialization_job_id.clone(),
        Uuid::now_v7(),
        lease.job.attempt_generation,
        lease.job.canonical_payload_hash.clone(),
        34,
    )
    .unwrap();
    let stale_generation = MaterializationCompletion::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        lease.job.materialization_job_id.clone(),
        lease_token,
        lease.job.attempt_generation + 1,
        lease.job.canonical_payload_hash.clone(),
        34,
    )
    .unwrap();
    for command in [&stale_hash, &stale_token, &stale_generation] {
        assert_eq!(
            activated
                .repository
                .complete_materialization(command)
                .unwrap(),
            MaterializationCompletionAck::StaleLease
        );
    }
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "vector_materialization_job_state_events"
        ),
        2
    );

    let completion = materialization_completion(&lease.job, lease_token, 34);
    let completed_job = match activated
        .repository
        .complete_materialization(&completion)
        .unwrap()
    {
        MaterializationCompletionAck::Applied {
            state: MaterializationCompletionState::Ready,
            job,
        } => job,
        acknowledgement => panic!("unexpected materialization completion: {acknowledgement:?}"),
    };
    assert_eq!(completed_job.embedding_id, Some(embedding_id));
    assert_eq!(
        activated
            .repository
            .complete_materialization(&completion)
            .unwrap(),
        MaterializationCompletionAck::AlreadyApplied {
            state: MaterializationCompletionState::Ready,
            job: completed_job,
        }
    );
    let mut completion_with_wrong_hash = completion.clone();
    completion_with_wrong_hash.expected_canonical_payload_hash = "f".repeat(64);
    let before_wrong_completion_hash = graph_snapshot(&activated.repository.connection);
    assert_eq!(
        activated
            .repository
            .complete_materialization(&completion_with_wrong_hash)
            .unwrap(),
        MaterializationCompletionAck::Conflict
    );
    assert_eq!(
        graph_snapshot(&activated.repository.connection),
        before_wrong_completion_hash
    );
    let states: (String, String, Option<String>, Option<String>) = activated
        .repository
        .connection
        .query_row(
            "SELECT
                (SELECT state FROM evidence_vector_link_state_events
                 ORDER BY event_seq DESC LIMIT 1),
                (SELECT state FROM vector_materialization_job_state_events
                 ORDER BY event_seq DESC LIMIT 1),
                (SELECT embedding_id FROM evidence_vector_link_state_events
                 ORDER BY event_seq DESC LIMIT 1),
                (SELECT embedding_id FROM vector_materialization_jobs)",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(
        states,
        (
            "ready".to_string(),
            "ready".to_string(),
            Some(embedding_id.to_string()),
            Some(embedding_id.to_string()),
        )
    );
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "vector_source_change_events"
        ),
        1
    );
}

#[test]
fn release_retry_and_dead_owner_reclaim_append_exact_materialization_states() {
    let project_id = "materialization-resolution-reclaim";
    let (temporary, mut activated) = activate(true, project_id);
    let pending = seed_pending_materialization(&mut activated);
    complete_pending_embedding(&mut activated, &pending);
    activate_empty_vector_generation(&mut activated, &pending.vector_space_id, 30);

    let first_token = Uuid::now_v7();
    let first_claim = materialization_claim(
        &pending.materialization_job_id,
        pending.attempt_generation,
        &pending.canonical_payload_hash,
        first_token,
        33,
    );
    let first_lease = match activated
        .repository
        .claim_materialization(&first_claim)
        .unwrap()
    {
        MaterializationClaimAck::Claimed(lease) => lease,
        acknowledgement => panic!("unexpected first claim: {acknowledgement:?}"),
    };
    let released = MaterializationResolution::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        first_lease.job.materialization_job_id.clone(),
        first_token,
        first_lease.job.attempt_generation,
        first_lease.job.canonical_payload_hash.clone(),
        34,
        MaterializationResolutionKind::Released,
    )
    .unwrap();
    let released_job = match activated
        .repository
        .resolve_materialization(&released)
        .unwrap()
    {
        MaterializationResolutionAck::Applied {
            state: MaterializationResolvedState::Released,
            job,
            ..
        } => job,
        acknowledgement => panic!("unexpected release: {acknowledgement:?}"),
    };
    assert!(matches!(
        activated
            .repository
            .resolve_materialization(&released)
            .unwrap(),
        MaterializationResolutionAck::AlreadyApplied {
            state: MaterializationResolvedState::Released,
            ref job,
            ..
        } if job == &released_job
    ));
    let mut release_with_wrong_hash = released.clone();
    release_with_wrong_hash.expected_canonical_payload_hash = "f".repeat(64);
    let before_wrong_resolution_hash = graph_snapshot(&activated.repository.connection);
    assert_eq!(
        activated
            .repository
            .resolve_materialization(&release_with_wrong_hash)
            .unwrap(),
        MaterializationResolutionAck::Conflict
    );
    assert_eq!(
        graph_snapshot(&activated.repository.connection),
        before_wrong_resolution_hash
    );

    let second_token = Uuid::now_v7();
    let second_claim = materialization_claim(
        &released_job.materialization_job_id,
        released_job.attempt_generation,
        &released_job.canonical_payload_hash,
        second_token,
        35,
    );
    let second_lease = match activated
        .repository
        .claim_materialization(&second_claim)
        .unwrap()
    {
        MaterializationClaimAck::Claimed(lease) => lease,
        acknowledgement => panic!("unexpected second claim: {acknowledgement:?}"),
    };
    let retry = MaterializationResolution::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        second_lease.job.materialization_job_id.clone(),
        second_token,
        second_lease.job.attempt_generation,
        second_lease.job.canonical_payload_hash.clone(),
        36,
        MaterializationResolutionKind::RetryScheduled {
            stable_error_class: "router.provider.timeout".to_string(),
            next_eligible_at_unix_ms: 40,
        },
    )
    .unwrap();
    let retry_job = match activated
        .repository
        .resolve_materialization(&retry)
        .unwrap()
    {
        MaterializationResolutionAck::Applied {
            state: MaterializationResolvedState::RetryScheduled,
            job,
            ..
        } => job,
        acknowledgement => panic!("unexpected retry schedule: {acknowledgement:?}"),
    };
    let early_claim = materialization_claim(
        &retry_job.materialization_job_id,
        retry_job.attempt_generation,
        &retry_job.canonical_payload_hash,
        Uuid::now_v7(),
        39,
    );
    assert_eq!(
        activated
            .repository
            .claim_materialization(&early_claim)
            .unwrap(),
        MaterializationClaimAck::NotEligible {
            next_eligible_at_unix_ms: 40,
        }
    );

    let third_token = Uuid::now_v7();
    let third_claim = materialization_claim(
        &retry_job.materialization_job_id,
        retry_job.attempt_generation,
        &retry_job.canonical_payload_hash,
        third_token,
        40,
    );
    let third_lease = match activated
        .repository
        .claim_materialization(&third_claim)
        .unwrap()
    {
        MaterializationClaimAck::Claimed(lease) => lease,
        acknowledgement => panic!("unexpected third claim: {acknowledgement:?}"),
    };

    let path = database_path(&temporary);
    let router_config = router_config(&path, project_id, true);
    let mut reclaimer = LedgerRepository::activate_at(&router_config, 29_999).unwrap();
    let reclaimer_token = Uuid::now_v7();
    let reclaim = materialization_claim(
        &third_lease.job.materialization_job_id,
        third_lease.job.attempt_generation,
        &third_lease.job.canonical_payload_hash,
        reclaimer_token,
        30_001,
    );
    let reclaimed = match reclaimer
        .repository
        .claim_materialization(&reclaim)
        .unwrap()
    {
        MaterializationClaimAck::Reclaimed(lease) => lease,
        acknowledgement => panic!("unexpected reclaim: {acknowledgement:?}"),
    };
    assert_eq!(
        reclaimed.lease_owner_process_instance_id,
        reclaimer.identity.process_instance_id
    );
    assert_eq!(
        reclaimed.job.attempt_generation,
        third_lease.job.attempt_generation + 1
    );
    let orphan: (i64, String, i64) = reclaimer
        .repository
        .connection
        .query_row(
            "SELECT attempt_generation, lease_token, created_at_unix_ms
             FROM vector_materialization_job_state_events
             WHERE state = 'orphaned_in_flight'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )
        .unwrap();
    assert_eq!(
        orphan,
        (
            third_lease.job.attempt_generation,
            third_token.to_string(),
            30_001,
        )
    );
}

#[test]
fn active_generation_completion_atomically_writes_ready_point_and_source_event() {
    let (_temporary, mut activated) = activate(true, "materialization-ready-source");
    let pending = seed_pending_materialization(&mut activated);
    let embedding_id = complete_pending_embedding(&mut activated, &pending);
    let root = activate_empty_vector_generation(&mut activated, &pending.vector_space_id, 33);
    let lease_token = Uuid::now_v7();
    let claim = materialization_claim(
        &pending.materialization_job_id,
        pending.attempt_generation,
        &pending.canonical_payload_hash,
        lease_token,
        36,
    );
    let lease = match activated.repository.claim_materialization(&claim).unwrap() {
        MaterializationClaimAck::Claimed(lease) => lease,
        acknowledgement => panic!("unexpected ready claim: {acknowledgement:?}"),
    };
    let completion = materialization_completion(&lease.job, lease_token, 37);
    let completed_job = match activated
        .repository
        .complete_materialization(&completion)
        .unwrap()
    {
        MaterializationCompletionAck::Applied {
            state: MaterializationCompletionState::Ready,
            job,
        } => job,
        acknowledgement => panic!("unexpected ready completion: {acknowledgement:?}"),
    };
    assert_eq!(completed_job.embedding_id, Some(embedding_id));
    assert_eq!(
        activated
            .repository
            .complete_materialization(&completion)
            .unwrap(),
        MaterializationCompletionAck::AlreadyApplied {
            state: MaterializationCompletionState::Ready,
            job: completed_job,
        }
    );
    let indexed: (i64, Vec<u8>) = activated
        .repository
        .connection
        .query_row(
            &format!("SELECT partition_id, embedding FROM \"{root}\" WHERE record_id = ?1"),
            params![pending.evidence_vector_link_id.to_string()],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!(indexed.0, pending.partition_id);
    assert_eq!(
        indexed.1,
        authoritative_test_vector(&activated.repository, &pending.vector_space_id)
            .blob()
            .native_endian_bytes()
    );
    let source: (i64, String, String, i64) = activated
        .repository
        .connection
        .query_row(
            "SELECT source_seq, operation, record_id, partition_id
             FROM vector_source_change_events",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .unwrap();
    assert_eq!(
        source,
        (
            1,
            "insert".to_string(),
            pending.evidence_vector_link_id.to_string(),
            pending.partition_id,
        )
    );
    assert_eq!(
        activated
            .repository
            .connection
            .query_row(
                "SELECT source_seq FROM vector_space_source_sequences
             WHERE vector_space_id = ?1",
                params![pending.vector_space_id.as_str()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
}

#[test]
fn first_completion_rejects_an_exact_stray_vec0_row_without_relational_mutation() {
    let (_temporary, mut activated) = activate(true, "materialization-stray-row");
    let pending = seed_pending_materialization(&mut activated);
    complete_pending_embedding(&mut activated, &pending);
    let root = activate_empty_vector_generation(&mut activated, &pending.vector_space_id, 33);
    let lease_token = Uuid::now_v7();
    let claim = materialization_claim(
        &pending.materialization_job_id,
        pending.attempt_generation,
        &pending.canonical_payload_hash,
        lease_token,
        36,
    );
    let lease = match activated.repository.claim_materialization(&claim).unwrap() {
        MaterializationClaimAck::Claimed(lease) => lease,
        acknowledgement => panic!("unexpected stray-row claim: {acknowledgement:?}"),
    };
    let vector = authoritative_test_vector(&activated.repository, &pending.vector_space_id);
    activated
        .repository
        .connection
        .execute(
            &format!(
                "INSERT INTO \"{root}\" (record_id, embedding, partition_id)
                 VALUES (?1, ?2, ?3)"
            ),
            params![
                pending.evidence_vector_link_id.to_string(),
                vector.blob().native_endian_bytes(),
                pending.partition_id,
            ],
        )
        .unwrap();
    let before = graph_snapshot(&activated.repository.connection);
    let completion = materialization_completion(&lease.job, lease_token, 37);

    assert_eq!(
        activated
            .repository
            .complete_materialization(&completion)
            .unwrap(),
        MaterializationCompletionAck::Conflict
    );
    assert_eq!(graph_snapshot(&activated.repository.connection), before);
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "vector_source_change_events"
        ),
        0
    );
    assert_eq!(
        activated
            .repository
            .connection
            .query_row(&format!("SELECT count(*) FROM \"{root}\""), [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        1
    );
}

#[test]
fn retention_of_ready_materialization_deletes_point_and_appends_source_change() {
    let (_temporary, mut activated) = activate(true, "materialization-ready-retention");
    let pending = seed_pending_materialization(&mut activated);
    complete_pending_embedding(&mut activated, &pending);
    let root = activate_empty_vector_generation(&mut activated, &pending.vector_space_id, 33);
    let lease_token = Uuid::now_v7();
    let claim = materialization_claim(
        &pending.materialization_job_id,
        pending.attempt_generation,
        &pending.canonical_payload_hash,
        lease_token,
        36,
    );
    let lease = match activated.repository.claim_materialization(&claim).unwrap() {
        MaterializationClaimAck::Claimed(lease) => lease,
        acknowledgement => panic!("unexpected ready claim: {acknowledgement:?}"),
    };
    let completion = materialization_completion(&lease.job, lease_token, 37);
    let ready = match activated
        .repository
        .complete_materialization(&completion)
        .unwrap()
    {
        MaterializationCompletionAck::Applied {
            state: MaterializationCompletionState::Ready,
            job,
        } => job,
        acknowledgement => panic!("unexpected ready completion: {acknowledgement:?}"),
    };
    let command = MaterializationRetentionDelete::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        ready.materialization_job_id.clone(),
        ready.attempt_generation,
        ready.canonical_payload_hash.clone(),
        40,
    )
    .unwrap();
    let project_uuid = activated.identity.project_uuid;
    let process_instance_id = activated.identity.process_instance_id;
    let transaction = activated.repository.connection_mut().transaction().unwrap();
    assert_eq!(
        delete_materialization_for_retention_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            &command,
        )
        .unwrap(),
        MaterializationRetentionAck::Deleted
    );
    assert_eq!(
        delete_materialization_for_retention_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            &command,
        )
        .unwrap(),
        MaterializationRetentionAck::AlreadyAbsent
    );
    assert_eq!(
        transaction
            .query_row(
                &format!("SELECT count(*) FROM \"{root}\" WHERE record_id = ?1"),
                params![pending.evidence_vector_link_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );
    let source_changes = transaction
        .prepare(
            "SELECT source_seq, operation, record_id, partition_id
             FROM vector_source_change_events ORDER BY source_seq",
        )
        .unwrap()
        .query_map([], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(
        source_changes,
        vec![
            (
                1,
                "insert".to_string(),
                pending.evidence_vector_link_id.to_string(),
                pending.partition_id,
            ),
            (
                2,
                "delete".to_string(),
                pending.evidence_vector_link_id.to_string(),
                pending.partition_id,
            ),
        ]
    );
    assert_eq!(table_count(&transaction, "evidence_vector_links"), 0);
    assert_eq!(table_count(&transaction, "vector_materialization_jobs"), 0);
    assert_eq!(table_count(&transaction, "vectorization_outcomes"), 0);
    transaction.commit().unwrap();
}

#[test]
fn marked_ready_source_replay_verifies_deindex_and_allows_final_graph_delete() {
    let (_temporary, mut activated) = activate(true, "marked-ready-source-retention");
    let (pending, _root, ready) = seed_ready_materialization(&mut activated);
    let project_uuid = activated.identity.project_uuid;
    let process_instance_id = activated.identity.process_instance_id;
    let anchor_id = activated
        .repository
        .connection
        .query_row(
            "SELECT anchor_id FROM evidence_vector_links WHERE evidence_vector_link_id = ?1",
            [pending.evidence_vector_link_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .map(|value| Uuid::parse_str(&value).unwrap())
        .unwrap();
    let vector = authoritative_test_vector(&activated.repository, &pending.vector_space_id);
    let transaction = activated.repository.connection_mut().transaction().unwrap();
    let first = MaterializationRetentionDelete::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        ready.materialization_job_id.clone(),
        ready.attempt_generation,
        ready.canonical_payload_hash.clone(),
        40,
    )
    .unwrap();
    let mut retired_spaces = BTreeSet::new();
    assert_eq!(
        deindex_materialization_for_retention_with_retired_spaces_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            &first,
            &mut retired_spaces,
        )
        .unwrap(),
        MaterializationRetentionAck::Deleted
    );

    let batch_id = Uuid::now_v7();
    transaction
        .execute(
            "INSERT INTO retention_batches (
                retention_batch_id, conflict_health_event_id, project_uuid,
                process_instance_id, summary_shape_version, age_expired,
                count_excess, selected_count, selection_lower_bound_unix_ms,
                selection_upper_bound_unix_ms, selection_hash,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, 3, 1, 0, 1, 40, 40, ?5, 40, ?6)",
            params![
                batch_id.to_string(),
                Uuid::now_v7().to_string(),
                project_uuid.to_string(),
                process_instance_id.to_string(),
                "a".repeat(64),
                "b".repeat(64),
            ],
        )
        .unwrap();
    transaction
        .execute(
            "INSERT INTO decision_retiring_anchors (
                anchor_id, project_uuid, first_retention_batch_id, age_expired,
                count_excess, created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, 1, 0, 40, ?4)",
            params![
                anchor_id.to_string(),
                project_uuid.to_string(),
                batch_id.to_string(),
                "c".repeat(64),
            ],
        )
        .unwrap();
    assert!(
        verify_retiring_anchor_deindexed_in_transaction(&transaction, project_uuid, anchor_id)
            .unwrap()
    );

    let repeat = MaterializationRetentionDelete::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        ready.materialization_job_id.clone(),
        ready.attempt_generation,
        ready.canonical_payload_hash.clone(),
        41,
    )
    .unwrap();
    assert_eq!(
        deindex_materialization_for_retention_with_retired_spaces_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            &repeat,
            &mut retired_spaces,
        )
        .unwrap(),
        MaterializationRetentionAck::Deleted
    );

    let dimensions = transaction
        .query_row(
            "SELECT dimensions FROM vector_spaces WHERE vector_space_id = ?1",
            [pending.vector_space_id.as_str()],
            |row| row.get::<_, u32>(0),
        )
        .unwrap();
    let rebuild_manifest = match authorize_generation(
        &transaction,
        &pending.vector_space_id,
        VectorDimensions::new(dimensions).unwrap(),
        42,
    )
    .unwrap()
    {
        GenerationAuthorizationAck::Created(manifest) => manifest,
        acknowledgement => panic!("unexpected marked rebuild authorization: {acknowledgement:?}"),
    };
    let rebuild_fence = match claim_rebuild_lease(
        &transaction,
        &pending.vector_space_id,
        process_instance_id,
        42,
    )
    .unwrap()
    {
        RebuildLeaseClaimAck::Claimed(fence) => fence,
        acknowledgement => panic!("unexpected marked rebuild claim: {acknowledgement:?}"),
    };
    assert_eq!(
        create_generation_objects(&transaction, &rebuild_fence, 42).unwrap(),
        GenerationObjectCreationAck::Created
    );
    assert!(matches!(
        populate_rebuild_chunk(&transaction, &rebuild_fence, 43).unwrap(),
        RebuildStepAck::Applied {
            processed: 0,
            complete: true,
            ..
        }
    ));
    assert!(matches!(
        catch_up_rebuild_changes(&transaction, &rebuild_fence, 43).unwrap(),
        RebuildStepAck::Applied {
            processed: 0,
            complete: true,
            ..
        }
    ));
    assert_eq!(
        flip_rebuild_generation(&transaction, &rebuild_fence, 44).unwrap(),
        RebuildFlipAck::Activated { record_count: 0 }
    );
    let active_root = rebuild_manifest.authority().root().as_str();
    assert_eq!(
        transaction
            .query_row(
                &format!("SELECT count(*) FROM \"{active_root}\""),
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );

    transaction
        .execute(
            &format!(
                "INSERT INTO \"{active_root}\" (record_id, embedding, partition_id) VALUES (?1, ?2, ?3)"
            ),
            params![
                pending.evidence_vector_link_id.to_string(),
                vector.blob().native_endian_bytes(),
                pending.partition_id,
            ],
        )
        .unwrap();
    assert!(
        !verify_retiring_anchor_deindexed_in_transaction(&transaction, project_uuid, anchor_id)
            .unwrap()
    );
    transaction
        .execute(
            &format!("DELETE FROM \"{active_root}\" WHERE record_id = ?1"),
            [pending.evidence_vector_link_id.to_string()],
        )
        .unwrap();

    let final_delete = MaterializationRetentionDelete::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        ready.materialization_job_id,
        ready.attempt_generation,
        ready.canonical_payload_hash,
        45,
    )
    .unwrap();
    assert_eq!(
        delete_materialization_for_retention_with_retired_spaces_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            &final_delete,
            &mut retired_spaces,
        )
        .unwrap(),
        MaterializationRetentionAck::Deleted
    );
    assert_eq!(table_count(&transaction, "evidence_vector_links"), 0);
    assert_eq!(table_count(&transaction, "vector_materialization_jobs"), 0);
    transaction.commit().unwrap();
}

#[test]
fn high_fanout_marker_pass_blocks_then_deletes_anchor_without_partial_materialization_cleanup() {
    let (_temporary, config, mut activated) =
        activate_with_decision_policy("high-fanout-marked-source-retention");
    let (pending, root, _ready) = seed_ready_materialization(&mut activated);
    let audits = (0..=1_000)
        .map(|_| {
            decision_test_fixtures::existing_neighbor_audit(
                &activated,
                &config,
                Uuid::now_v7(),
                pending.evidence_vector_link_id,
                50,
            )
        })
        .collect::<Vec<_>>();
    decision_test_fixtures::insert_test_decision_graphs(
        activated.repository.connection_mut(),
        &audits,
    );
    activated.repository.max_evidence_records = 1;

    let first_request = RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), 60).unwrap();
    let RetentionAck::Applied {
        summary,
        observation,
    } = activated.repository.run_retention(&first_request).unwrap()
    else {
        panic!("first high-fanout retention pass must apply");
    };
    assert_eq!(summary.selected_count, 0);
    assert!(observation.more_cleanup);
    assert_eq!(
        table_count(&activated.repository.connection, "decisions"),
        1
    );
    assert_eq!(table_count(&activated.repository.connection, "anchors"), 1);
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "decision_retiring_anchors"
        ),
        1
    );
    assert_eq!(
        table_count(&activated.repository.connection, "evidence_vector_links"),
        1
    );
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "vector_materialization_jobs"
        ),
        1
    );
    assert_eq!(
        activated
            .repository
            .connection
            .query_row(
                &format!("SELECT count(*) FROM \"{root}\" WHERE record_id = ?1"),
                [pending.evidence_vector_link_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );
    let first_receipt = activated
        .repository
        .connection
        .query_row(
            "SELECT new_marker_count, deleted_decision_count, blocked_anchor_count,
                    deleted_anchor_count, more_cleanup
             FROM decision_retention_receipts WHERE retention_batch_id = ?1",
            [first_request.retention_batch_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(first_receipt, (1, 1_000, 1, 0, 1));
    assert_eq!(
        activated.repository.run_retention(&first_request).unwrap(),
        RetentionAck::AlreadyApplied {
            summary: summary.clone(),
            observation,
        }
    );
    assert_eq!(
        (
            table_count(&activated.repository.connection, "decisions"),
            table_count(&activated.repository.connection, "anchors"),
            table_count(
                &activated.repository.connection,
                "decision_retiring_anchors"
            ),
        ),
        (1, 1, 1)
    );

    let second_request = RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), 61).unwrap();
    let RetentionAck::Applied { summary, .. } =
        activated.repository.run_retention(&second_request).unwrap()
    else {
        panic!("second high-fanout retention pass must apply");
    };
    assert_eq!(summary.selected_count, 1);
    assert_eq!(
        table_count(&activated.repository.connection, "decisions"),
        0
    );
    assert_eq!(table_count(&activated.repository.connection, "anchors"), 0);
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "decision_retiring_anchors"
        ),
        0
    );
    assert_eq!(
        table_count(&activated.repository.connection, "evidence_vector_links"),
        0
    );
    let second_receipt = activated
        .repository
        .connection
        .query_row(
            "SELECT new_marker_count, deleted_decision_count, blocked_anchor_count,
                    deleted_anchor_count
             FROM decision_retention_receipts WHERE retention_batch_id = ?1",
            [second_request.retention_batch_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(second_receipt, (0, 1, 0, 1));
}

#[test]
fn top_level_retention_cancels_pending_and_claimed_materializations() {
    for claimed in [false, true] {
        let project_id = if claimed {
            "retention-claimed-materialization"
        } else {
            "retention-pending-materialization"
        };
        let (_temporary, mut activated) = activate(true, project_id);
        let pending = seed_pending_materialization(&mut activated);
        if claimed {
            complete_pending_embedding(&mut activated, &pending);
            activate_empty_vector_generation(&mut activated, &pending.vector_space_id, 33);
            let lease_token = Uuid::now_v7();
            let claim = materialization_claim(
                &pending.materialization_job_id,
                pending.attempt_generation,
                &pending.canonical_payload_hash,
                lease_token,
                36,
            );
            assert!(matches!(
                activated.repository.claim_materialization(&claim).unwrap(),
                MaterializationClaimAck::Claimed(_)
            ));
        }

        let RetentionAck::Applied { summary, .. } = run_single_anchor_retention(&mut activated, 40)
        else {
            panic!("selected materialization retention should apply");
        };
        assert_eq!(summary.selected_count, 1);
        assert_eq!(table_count(&activated.repository.connection, "anchors"), 0);
        assert_eq!(
            table_count(&activated.repository.connection, "evidence_vector_links"),
            0
        );
        assert_eq!(
            table_count(
                &activated.repository.connection,
                "vector_materialization_jobs"
            ),
            0
        );
        assert_eq!(
            table_count(
                &activated.repository.connection,
                "vector_materialization_job_state_events"
            ),
            0
        );
        assert_eq!(
            table_count(
                &activated.repository.connection,
                "vector_source_change_events"
            ),
            0
        );
        assert_eq!(
            table_count(&activated.repository.connection, "embedding_jobs"),
            if claimed { 0 } else { 1 }
        );
        assert_eq!(
            table_count(
                &activated.repository.connection,
                "canonical_routing_queries"
            ),
            if claimed { 0 } else { 1 }
        );
        assert_eq!(
            table_count(
                &activated.repository.connection,
                "pool_vector_space_mappings"
            ),
            1
        );
    }
}

#[test]
fn top_level_retention_accepts_two_links_sharing_one_pending_embedding_job() {
    let (_temporary, mut activated) = activate(true, "retention-shared-embedding-job");
    let first = seed_pending_materialization(&mut activated);
    let second = seed_pending_materialization(&mut activated);
    assert_eq!(first.embedding_job_id, second.embedding_job_id);
    assert_eq!(
        table_count(&activated.repository.connection, "embedding_jobs"),
        1
    );

    assert!(matches!(
        run_single_anchor_retention(&mut activated, 40),
        RetentionAck::Applied { .. }
    ));
    assert_eq!(table_count(&activated.repository.connection, "anchors"), 0);
    assert_eq!(
        table_count(&activated.repository.connection, "evidence_vector_links"),
        0
    );
    assert_eq!(
        table_count(&activated.repository.connection, "embedding_jobs"),
        1
    );
    assert_eq!(
        table_count(&activated.repository.connection, "embedding_jobs"),
        1
    );
}

#[test]
fn top_level_retention_deletes_ready_point_and_records_source_delete() {
    let (_temporary, mut activated) = activate(true, "top-level-ready-retention");
    let (pending, root, _ready) = seed_ready_materialization(&mut activated);

    let RetentionAck::Applied { summary, .. } = run_single_anchor_retention(&mut activated, 40)
    else {
        panic!("ready materialization retention should apply");
    };
    assert_eq!(summary.selected_count, 1);
    assert_eq!(
        activated
            .repository
            .connection
            .query_row(
                &format!("SELECT count(*) FROM \"{root}\" WHERE record_id = ?1"),
                params![pending.evidence_vector_link_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );
    let operations = activated
        .repository
        .connection
        .prepare("SELECT operation FROM vector_source_change_events ORDER BY source_seq")
        .unwrap()
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .collect::<Result<Vec<_>, _>>()
        .unwrap();
    assert_eq!(operations, vec!["insert", "delete"]);
    assert_eq!(
        table_count(&activated.repository.connection, "evidence_vector_links"),
        0
    );
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "vector_materialization_jobs"
        ),
        0
    );
    assert_eq!(
        table_count(&activated.repository.connection, "embeddings"),
        0
    );
    assert_eq!(
        table_count(&activated.repository.connection, "embedding_jobs"),
        0
    );
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "canonical_routing_queries"
        ),
        0
    );
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "pool_vector_space_mappings"
        ),
        1
    );
    assert_eq!(
        table_count(&activated.repository.connection, "vector_spaces"),
        1
    );
    assert_eq!(
        table_count(&activated.repository.connection, "embedder_profiles"),
        1
    );
}

#[test]
fn held_search_snapshot_remains_complete_across_retention_commit() {
    let (temporary, mut activated) = activate(true, "retention-search-snapshot");
    let (pending, _root, _ready) = seed_ready_materialization(&mut activated);
    let query = authoritative_test_vector(&activated.repository, &pending.vector_space_id)
        .vector()
        .clone();
    let partition_id = PartitionId::new(pending.partition_id).unwrap();
    let mut reader = Connection::open(database_path(&temporary)).unwrap();
    reader.execute_batch("PRAGMA query_only = ON").unwrap();
    let snapshot = reader.transaction().unwrap();
    assert_eq!(
        search_projected_neighbors_in_transaction(
            &snapshot,
            &pending.vector_space_id,
            partition_id,
            &query,
            1,
        )
        .unwrap()
        .len(),
        1
    );

    assert!(matches!(
        run_single_anchor_retention(&mut activated, 40),
        RetentionAck::Applied { .. }
    ));
    assert_eq!(
        search_projected_neighbors_in_transaction(
            &snapshot,
            &pending.vector_space_id,
            partition_id,
            &query,
            1,
        )
        .unwrap()
        .len(),
        1
    );
    snapshot.commit().unwrap();
    assert!(
        search_projected_neighbors(&reader, &pending.vector_space_id, partition_id, &query, 1,)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn empty_retention_sweeps_unattached_completed_cache_graph_but_keeps_current_registry() {
    let (_temporary, mut activated) = activate(true, "global-cache-retention");
    let attempt = attempt(true);
    let query = build_canonical_routing_query(
        &attempt.request_projection,
        &routing_projection(),
        &config(Path::new("router.db"), "global-cache-retention").pools[0].canonicalizer,
    )
    .unwrap();
    let project_uuid = activated.identity.project_uuid;
    let vector_space_id = activated.identity.pools["pool-a"]
        .vector_space
        .as_ref()
        .unwrap()
        .vector_space_id
        .clone();
    let query_hash = query.canonical_query_hash.clone();
    let create = EmbeddingJobCreate::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        vector_space_id.as_str(),
        query_hash.clone(),
        query_hash.clone(),
        25,
    )
    .unwrap();
    let embedding_job_id = create.embedding_job_id().to_string();
    let transaction = activated.repository.connection_mut().transaction().unwrap();
    assert!(matches!(
        ensure_canonical_query(&transaction, &query, 25).unwrap(),
        CanonicalQueryEnsureAck::Applied(_)
    ));
    assert!(matches!(
        create_embedding_job_in_transaction(
            &transaction,
            project_uuid,
            activated.identity.process_instance_id,
            &create,
        )
        .unwrap(),
        EmbeddingJobCreateAck::Applied(_)
    ));
    transaction.commit().unwrap();
    complete_pending_embedding(
        &mut activated,
        &PendingMaterialization {
            materialization_job_id: String::new(),
            evidence_vector_link_id: Uuid::now_v7(),
            partition_id: 1,
            attempt_generation: 0,
            canonical_payload_hash: String::new(),
            embedding_job_id,
            vector_space_id,
            canonical_query_hash: query_hash,
        },
    );
    assert_eq!(table_count(&activated.repository.connection, "anchors"), 0);

    let RetentionAck::Applied { summary, .. } = run_single_anchor_retention(&mut activated, 40)
    else {
        panic!("empty retention should sweep the global orphan selection");
    };
    assert_eq!(summary.selected_count, 0);
    assert_eq!(
        table_count(&activated.repository.connection, "embeddings"),
        0
    );
    assert_eq!(
        table_count(&activated.repository.connection, "embedding_jobs"),
        0
    );
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "canonical_routing_queries"
        ),
        0
    );
    assert_eq!(
        table_count(
            &activated.repository.connection,
            "pool_vector_space_mappings"
        ),
        1
    );
    assert_eq!(
        table_count(&activated.repository.connection, "vector_spaces"),
        1
    );
    assert_eq!(
        table_count(&activated.repository.connection, "embedder_profiles"),
        1
    );
}

#[test]
fn historical_active_space_converges_after_background_retired_cleanup() {
    let temporary = tempdir().unwrap();
    let path = database_path(&temporary);
    let initial_config = router_config(&path, "active-space-retention", true);
    let mut origin = LedgerRepository::activate_at(&initial_config, 0).unwrap();
    let (pending, _, _) = seed_ready_materialization(&mut origin);
    let old_space = pending.vector_space_id.clone();
    let old_manifest =
        match resolve_active_generation(&origin.repository.connection, &old_space).unwrap() {
            ActiveGenerationResolution::Active(manifest) => manifest,
            other => panic!("unexpected old generation: {other:?}"),
        };
    let old_generation = old_manifest.generation();
    let checksum = VectorChecksum::new("f".repeat(64)).unwrap();
    let partition_id = PartitionId::new(pending.partition_id).unwrap();
    for source_seq in 2..=300 {
        let record_id = VectorRecordId::new(Uuid::now_v7()).unwrap();
        let payload_hash = vector_source_change_payload_hash(
            &old_space,
            source_seq,
            VectorSourceChangeOperation::Insert,
            record_id,
            partition_id,
            &checksum,
            38,
        )
        .unwrap();
        origin
            .repository
            .connection
            .execute(
                "INSERT INTO vector_source_change_events (
                    vector_space_id, source_seq, operation, record_id, partition_id,
                    vector_checksum, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, 'insert', ?3, ?4, ?5, ?6, ?7)",
                params![
                    old_space.as_str(),
                    source_seq,
                    record_id.to_string(),
                    partition_id.value(),
                    checksum.as_str(),
                    38,
                    payload_hash,
                ],
            )
            .unwrap();
    }
    let sequence_hash = vector_source_sequence_payload_hash(&old_space, 300, 38).unwrap();
    origin
        .repository
        .connection
        .execute(
            "UPDATE vector_space_source_sequences
             SET source_seq = 300, updated_at_unix_ms = 38,
                 canonical_payload_hash = ?1
             WHERE vector_space_id = ?2",
            params![sequence_hash, old_space.as_str()],
        )
        .unwrap();
    assert_eq!(
        origin
            .repository
            .stop_process(ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), 40).unwrap())
            .unwrap(),
        ProcessCommandAck::Applied
    );
    drop(origin);

    let mut current_config = initial_config;
    current_config.embedders[0].model = "embedding-model-b".to_string();
    let mut current = LedgerRepository::activate_at(&current_config, 50).unwrap();
    let new_space = current.identity.pools["pool-a"]
        .vector_space
        .as_ref()
        .unwrap()
        .vector_space_id
        .clone();
    assert_ne!(old_space, new_space);

    assert!(matches!(
        run_single_anchor_retention(&mut current, 60),
        RetentionAck::Applied { .. }
    ));
    let retired =
        load_validated_manifest(&current.repository.connection, &old_space, old_generation)
            .unwrap()
            .unwrap();
    assert_eq!(retired.state(), VectorIndexManifestState::Retired);
    assert_eq!(
        current
            .repository
            .connection
            .query_row(
                "SELECT count(*) FROM pool_vector_space_mappings
                 WHERE vector_space_id = ?1",
                params![old_space.as_str()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );

    let old_dimensions = current
        .repository
        .connection
        .query_row(
            "SELECT dimensions FROM vector_spaces WHERE vector_space_id = ?1",
            params![old_space.as_str()],
            |row| row.get::<_, u32>(0),
        )
        .unwrap();
    let transaction = current.repository.connection_mut().transaction().unwrap();
    assert_eq!(
        cleanup_retired_generation(&transaction, &old_space, old_generation, 70).unwrap(),
        RetiredGenerationCleanupAck::Dropped
    );
    transaction.commit().unwrap();
    assert!(matches!(
        run_single_anchor_retention(&mut current, 80),
        RetentionAck::Applied { .. }
    ));
    let dropped =
        load_validated_manifest(&current.repository.connection, &old_space, old_generation)
            .unwrap()
            .unwrap();
    assert_eq!(dropped.state(), VectorIndexManifestState::Dropped);
    assert_eq!(
        current
            .repository
            .connection
            .query_row(
                "SELECT source_seq FROM vector_space_source_sequences
                 WHERE vector_space_id = ?1",
                params![old_space.as_str()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        45
    );
    let transaction = current.repository.connection_mut().transaction().unwrap();
    assert_eq!(
        authorize_generation(
            &transaction,
            &old_space,
            VectorDimensions::new(old_dimensions).unwrap(),
            81,
        )
        .unwrap(),
        GenerationAuthorizationAck::AuthorityMissing
    );
    transaction.commit().unwrap();
    assert!(matches!(
        run_single_anchor_retention(&mut current, 82),
        RetentionAck::Applied { .. }
    ));
    assert!(
        load_validated_manifest(&current.repository.connection, &old_space, old_generation)
            .unwrap()
            .is_none()
    );
    assert_eq!(
        current
            .repository
            .connection
            .query_row(
                "SELECT count(*) FROM vector_spaces WHERE vector_space_id = ?1",
                params![old_space.as_str()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        0
    );
    assert_eq!(
        table_count(&current.repository.connection, "vector_spaces"),
        1
    );
    assert_eq!(
        table_count(&current.repository.connection, "embedder_profiles"),
        1
    );
    assert_eq!(
        table_count(&current.repository.connection, "pool_vector_space_mappings"),
        1
    );
}

#[test]
fn top_level_retention_retires_unavailable_generation_before_relational_delete() {
    let (_temporary, mut activated) = activate(true, "unavailable-generation-retention");
    let (pending, root, _ready) = seed_ready_materialization(&mut activated);
    let active =
        match resolve_active_generation(&activated.repository.connection, &pending.vector_space_id)
            .unwrap()
        {
            ActiveGenerationResolution::Active(active) => active,
            other => panic!("unexpected active generation: {other:?}"),
        };
    let generation = active.generation();
    let transaction = activated.repository.connection_mut().transaction().unwrap();
    assert!(matches!(
        mark_vector_index_health(
            &transaction,
            &pending.vector_space_id,
            generation,
            active.canonical_payload_hash(),
            VectorIndexHealthTarget::Unavailable,
            "router.vector.unavailable",
            38,
        )
        .unwrap(),
        VectorIndexHealthMutationAck::Applied { .. }
    ));
    transaction.commit().unwrap();

    assert!(matches!(
        run_single_anchor_retention(&mut activated, 40),
        RetentionAck::Applied { .. }
    ));
    let retired = load_validated_manifest(
        &activated.repository.connection,
        &pending.vector_space_id,
        generation,
    )
    .unwrap()
    .unwrap();
    assert_eq!(retired.state(), VectorIndexManifestState::Retired);
    assert!(matches!(
        resolve_active_generation(&activated.repository.connection, &pending.vector_space_id)
            .unwrap(),
        ActiveGenerationResolution::Missing
    ));
    assert_eq!(
        activated
            .repository
            .connection
            .query_row(
                &format!("SELECT count(*) FROM \"{root}\" WHERE record_id = ?1"),
                params![pending.evidence_vector_link_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
    assert_eq!(
        table_count(&activated.repository.connection, "evidence_vector_links"),
        0
    );
    assert_eq!(
        activated
            .repository
            .connection
            .query_row(
                "SELECT count(*) FROM vector_source_change_events WHERE operation = 'delete'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
}

#[test]
fn vec_delete_unavailability_rolls_back_retention_and_returns_exact_health_fence() {
    let (_temporary, mut activated) = activate(true, "vec-delete-retention-rollback");
    let (pending, root, _ready) = seed_ready_materialization(&mut activated);
    let active =
        match resolve_active_generation(&activated.repository.connection, &pending.vector_space_id)
            .unwrap()
        {
            ActiveGenerationResolution::Active(active) => active,
            other => panic!("unexpected active generation: {other:?}"),
        };
    let generation = active.generation();
    let active_hash = active.canonical_payload_hash().to_string();
    activated
        .repository
        .connection
        .execute(
            &format!("DELETE FROM \"{root}\" WHERE record_id = ?1"),
            params![pending.evidence_vector_link_id.to_string()],
        )
        .unwrap();
    let before = graph_snapshot(&activated.repository.connection);

    let RetentionAck::VectorIndexUnavailable {
        vector_space_id,
        expected_generation,
        expected_manifest_hash,
    } = run_single_anchor_retention(&mut activated, 40)
    else {
        panic!("vector-local failure should return an exact health fence");
    };
    assert_eq!(vector_space_id, pending.vector_space_id);
    assert_eq!(expected_generation, generation);
    assert_eq!(expected_manifest_hash, active_hash);
    assert_eq!(graph_snapshot(&activated.repository.connection), before);
    assert_eq!(
        table_count(&activated.repository.connection, "retention_batches"),
        0
    );

    let transaction = activated.repository.connection_mut().transaction().unwrap();
    assert!(matches!(
        mark_vector_index_health(
            &transaction,
            &vector_space_id,
            expected_generation,
            &expected_manifest_hash,
            VectorIndexHealthTarget::Unavailable,
            "router.vector.unavailable",
            41,
        )
        .unwrap(),
        VectorIndexHealthMutationAck::Applied { .. }
    ));
    transaction.commit().unwrap();
    let unavailable = load_validated_manifest(
        &activated.repository.connection,
        &vector_space_id,
        expected_generation,
    )
    .unwrap()
    .unwrap();
    assert_eq!(unavailable.state(), VectorIndexManifestState::Unavailable);
    let transaction = activated.repository.connection_mut().transaction().unwrap();
    assert_eq!(
        mark_vector_index_health(
            &transaction,
            &vector_space_id,
            expected_generation,
            &expected_manifest_hash,
            VectorIndexHealthTarget::Unavailable,
            "router.vector.unavailable",
            42,
        )
        .unwrap(),
        VectorIndexHealthMutationAck::Stale
    );
    transaction.commit().unwrap();
    assert!(matches!(
        run_single_anchor_retention(&mut activated, 43),
        RetentionAck::Applied { .. }
    ));
    let retired = load_validated_manifest(
        &activated.repository.connection,
        &vector_space_id,
        expected_generation,
    )
    .unwrap()
    .unwrap();
    assert_eq!(retired.state(), VectorIndexManifestState::Retired);
    assert_eq!(
        table_count(&activated.repository.connection, "evidence_vector_links"),
        0
    );
}

#[test]
fn partial_generation_authority_blocks_and_rolls_back_retention() {
    let (_temporary, mut activated) = activate(true, "partial-retention-rollback");
    let (_pending, root, _ready) = seed_ready_materialization(&mut activated);
    activated
        .repository
        .connection
        .execute_batch(&format!("DROP TABLE \"{root}_vector_chunks00\""))
        .unwrap();
    let before = graph_snapshot(&activated.repository.connection);
    activated.repository.max_evidence_records = 1;

    let error = activated
        .repository
        .run_retention(&RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), 40).unwrap())
        .unwrap_err();
    assert_eq!(
        error.class(),
        crate::ledger::model::LedgerErrorClass::CorruptDatabase
    );
    assert_eq!(graph_snapshot(&activated.repository.connection), before);
    assert_eq!(
        table_count(&activated.repository.connection, "retention_batches"),
        0
    );
}

#[test]
fn rehashed_vector_graph_before_shadow_terminal_blocks_retention() {
    let (_temporary, mut activated) = activate(true, "retention-graph-time-fence");
    seed_pending_materialization(&mut activated);
    let stored = activated
        .repository
        .connection
        .query_row(
            "SELECT vectorization_outcome_id, shadow_attempt_id, anchor_id,
                    learning_generation_id, vector_space_id, canonical_query_hash,
                    outcome, stable_reason
             FROM vectorization_outcomes",
            [],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, Option<String>>(7)?,
                ))
            },
        )
        .unwrap();
    let tampered_at = 29;
    let tampered_hash = canonical_sha256(&json!({
        "vectorization_outcome_id": stored.0,
        "shadow_attempt_id": Uuid::parse_str(&stored.1).unwrap(),
        "anchor_id": Uuid::parse_str(&stored.2).unwrap(),
        "learning_generation_id": Uuid::parse_str(&stored.3).unwrap(),
        "vector_space_id": stored.4,
        "canonical_query_hash": stored.5,
        "outcome": stored.6,
        "stable_reason": stored.7,
        "created_at_unix_ms": tampered_at,
    }))
    .unwrap();
    activated
        .repository
        .connection
        .execute(
            "UPDATE vectorization_outcomes
             SET created_at_unix_ms = ?1, canonical_payload_hash = ?2",
            params![tampered_at, tampered_hash],
        )
        .unwrap();
    let before = graph_snapshot(&activated.repository.connection);
    activated.repository.max_evidence_records = 1;
    let error = activated
        .repository
        .run_retention(&RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), 40).unwrap())
        .unwrap_err();
    assert_eq!(
        error.class(),
        crate::ledger::model::LedgerErrorClass::CorruptDatabase
    );
    assert_eq!(graph_snapshot(&activated.repository.connection), before);
}

#[test]
fn routing_partition_pool_authority_mismatch_blocks_retention() {
    let (_temporary, mut activated) = activate(true, "retention-partition-pool-fence");
    seed_pending_materialization(&mut activated);
    activated
        .repository
        .connection
        .execute_batch("PRAGMA foreign_keys = OFF")
        .unwrap();
    activated
        .repository
        .connection
        .execute("UPDATE routing_partitions SET pool_id = 'wrong-pool'", [])
        .unwrap();
    activated
        .repository
        .connection
        .execute_batch("PRAGMA foreign_keys = ON")
        .unwrap();
    let before = graph_snapshot(&activated.repository.connection);
    activated.repository.max_evidence_records = 1;
    let error = activated
        .repository
        .run_retention(&RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), 40).unwrap())
        .unwrap_err();
    assert_eq!(
        error.class(),
        crate::ledger::model::LedgerErrorClass::CorruptDatabase
    );
    assert_eq!(graph_snapshot(&activated.repository.connection), before);
    assert_eq!(
        table_count(&activated.repository.connection, "retention_batches"),
        0
    );
}

#[test]
fn terminal_embedding_failure_fans_out_in_exact_256_item_windows() {
    let (_temporary, mut activated) = activate(true, "materialization-failure-fanout");
    let first = seed_pending_materialization(&mut activated);
    for _ in 1..256 {
        let pending = seed_pending_materialization(&mut activated);
        assert_eq!(pending.embedding_job_id, first.embedding_job_id);
    }

    let lease_token = Uuid::now_v7();
    let claim = EmbeddingJobBatchClaim::new(
        Uuid::now_v7(),
        first.vector_space_id.clone(),
        lease_token,
        31,
        vec![
            EmbeddingJobBatchClaimItem::new(
                first.embedding_job_id.clone(),
                Uuid::now_v7(),
                Uuid::now_v7(),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let lease = match activated
        .repository
        .claim_embedding_job_batch(&claim)
        .unwrap()
    {
        EmbeddingJobBatchClaimAck::Claimed(mut leases) if leases.len() == 1 => leases.remove(0),
        acknowledgement => panic!("unexpected terminal embedding claim: {acknowledgement:?}"),
    };
    let resolution = EmbeddingJobResolution::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        lease.job.embedding_job_id.clone(),
        lease_token,
        lease.job.attempt_generation,
        lease.job.content_hash.clone(),
        32,
        EmbeddingJobResolutionKind::TerminalFailure {
            stable_error_class: "invalid_response".to_string(),
        },
    )
    .unwrap();
    let terminal_job = match activated
        .repository
        .resolve_embedding_job(&resolution)
        .unwrap()
    {
        EmbeddingJobResolutionAck::Applied {
            state: EmbeddingJobResolvedState::TerminalFailure,
            job,
            ..
        } => job,
        acknowledgement => panic!("unexpected terminal embedding resolution: {acknowledgement:?}"),
    };

    // A dependent linked after terminalization starts failed and becomes the 257th row.
    let post_terminal = seed_pending_materialization(&mut activated);
    assert_eq!(post_terminal.embedding_job_id, first.embedding_job_id);
    let mut link_ids = activated
        .repository
        .connection
        .prepare(
            "SELECT evidence_vector_link_id FROM vector_materialization_jobs
             WHERE embedding_job_id = ?1 ORDER BY evidence_vector_link_id",
        )
        .unwrap()
        .query_map(params![first.embedding_job_id], |row| {
            row.get::<_, String>(0)
        })
        .unwrap()
        .map(|value| Uuid::parse_str(&value.unwrap()).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(link_ids.len(), 257);
    assert_eq!(link_ids.pop(), Some(post_terminal.evidence_vector_link_id));
    link_ids.push(post_terminal.evidence_vector_link_id);

    let post_events = activated
        .repository
        .connection
        .query_row(
            "SELECT ms.vector_materialization_job_state_event_id,
                    ls.evidence_vector_link_state_event_id
             FROM vector_materialization_jobs AS m
             JOIN vector_materialization_job_state_events AS ms
               ON ms.vector_materialization_job_id = m.vector_materialization_job_id
             JOIN evidence_vector_link_state_events AS ls
               ON ls.evidence_vector_link_id = m.evidence_vector_link_id
             WHERE m.evidence_vector_link_id = ?1
               AND ms.state = 'failed_embedding' AND ls.state = 'failed_embedding'",
            params![post_terminal.evidence_vector_link_id.to_string()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .unwrap();
    let post_events = (
        Uuid::parse_str(&post_events.0).unwrap(),
        Uuid::parse_str(&post_events.1).unwrap(),
    );

    let first_items = link_ids[..256]
        .iter()
        .map(|link_id| {
            MaterializationFailurePropagationItem::new(*link_id, Uuid::now_v7(), Uuid::now_v7())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let first_cursor = link_ids[255];
    let mut wrong_order = first_items.clone();
    wrong_order.swap(0, 1);
    let conflicting = MaterializationFailurePropagation::new(
        Uuid::now_v7(),
        terminal_job.embedding_job_id.clone(),
        terminal_job.canonical_payload_hash.clone(),
        terminal_job.attempt_generation,
        None,
        33,
        wrong_order,
    )
    .unwrap();
    let before_conflict = graph_snapshot(&activated.repository.connection);
    assert_eq!(
        activated
            .repository
            .propagate_materialization_failure(&conflicting)
            .unwrap(),
        MaterializationFailurePropagationAck::Conflict
    );
    assert_eq!(
        graph_snapshot(&activated.repository.connection),
        before_conflict
    );

    let first_window = MaterializationFailurePropagation::new(
        Uuid::now_v7(),
        terminal_job.embedding_job_id.clone(),
        terminal_job.canonical_payload_hash.clone(),
        terminal_job.attempt_generation,
        None,
        33,
        first_items,
    )
    .unwrap();
    assert_eq!(
        activated
            .repository
            .propagate_materialization_failure(&first_window)
            .unwrap(),
        MaterializationFailurePropagationAck::Applied {
            processed: 256,
            next_cursor: Some(first_cursor),
            complete: false,
        }
    );
    let after_first_window = graph_snapshot(&activated.repository.connection);
    assert_eq!(
        activated
            .repository
            .propagate_materialization_failure(&first_window)
            .unwrap(),
        MaterializationFailurePropagationAck::AlreadyApplied {
            processed: 256,
            next_cursor: Some(first_cursor),
            complete: false,
        }
    );
    assert_eq!(
        graph_snapshot(&activated.repository.connection),
        after_first_window
    );

    let mut alternate_ids = first_window.clone();
    alternate_ids.conflict_health_event_id = Uuid::now_v7();
    alternate_ids.items[0].materialization_state_event_id = Uuid::now_v7();
    let before_alternate_ids = graph_snapshot(&activated.repository.connection);
    assert_eq!(
        activated
            .repository
            .propagate_materialization_failure(&alternate_ids)
            .unwrap(),
        MaterializationFailurePropagationAck::Conflict
    );
    assert_eq!(
        graph_snapshot(&activated.repository.connection),
        before_alternate_ids
    );

    let first_window_job = activated
        .repository
        .connection
        .query_row(
            "SELECT attempt_generation, canonical_payload_hash
             FROM embedding_jobs WHERE embedding_job_id = ?1",
            params![terminal_job.embedding_job_id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .unwrap();
    let incomplete_reset = EmbeddingJobReset::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        terminal_job.embedding_job_id.clone(),
        first_window_job.0,
        first_window_job.1.clone(),
        "operator",
        "reviewed terminal embedding failure",
        34,
    )
    .unwrap();
    assert_eq!(
        activated
            .repository
            .reset_embedding_job(&incomplete_reset)
            .unwrap(),
        EmbeddingJobResetAck::PropagationIncomplete
    );

    let second_window = MaterializationFailurePropagation::new(
        Uuid::now_v7(),
        terminal_job.embedding_job_id.clone(),
        first_window_job.1,
        first_window_job.0,
        Some(first_cursor),
        35,
        vec![
            MaterializationFailurePropagationItem::new(
                post_terminal.evidence_vector_link_id,
                post_events.0,
                post_events.1,
            )
            .unwrap(),
        ],
    )
    .unwrap();
    assert_eq!(
        failed_state_counts(&activated.repository, post_terminal.evidence_vector_link_id),
        (1, 1)
    );
    assert_eq!(
        activated
            .repository
            .propagate_materialization_failure(&second_window)
            .unwrap(),
        MaterializationFailurePropagationAck::Applied {
            processed: 1,
            next_cursor: Some(post_terminal.evidence_vector_link_id),
            complete: true,
        }
    );
    assert_eq!(
        failed_state_counts(&activated.repository, post_terminal.evidence_vector_link_id),
        (1, 1)
    );
    let after_second_window = graph_snapshot(&activated.repository.connection);
    assert_eq!(
        activated
            .repository
            .propagate_materialization_failure(&second_window)
            .unwrap(),
        MaterializationFailurePropagationAck::AlreadyApplied {
            processed: 1,
            next_cursor: Some(post_terminal.evidence_vector_link_id),
            complete: true,
        }
    );
    assert_eq!(
        graph_snapshot(&activated.repository.connection),
        after_second_window
    );
    assert_eq!(
        failed_state_counts(&activated.repository, post_terminal.evidence_vector_link_id),
        (1, 1)
    );
    assert_eq!(
        activated
            .repository
            .connection
            .query_row(
                "SELECT count(*) FROM vector_materialization_job_state_events
                 WHERE state = 'failed_embedding'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        257
    );
    assert_eq!(
        activated
            .repository
            .connection
            .query_row(
                "SELECT count(*) FROM evidence_vector_link_state_events
                 WHERE state = 'failed_embedding'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        257
    );

    let fully_propagated_job = activated
        .repository
        .connection
        .query_row(
            "SELECT attempt_generation, canonical_payload_hash,
                    failure_propagation_cursor, failure_propagation_complete
             FROM embedding_jobs WHERE embedding_job_id = ?1",
            params![terminal_job.embedding_job_id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, bool>(3)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(
        fully_propagated_job.2.as_deref(),
        Some(post_terminal.evidence_vector_link_id.to_string().as_str())
    );
    assert!(fully_propagated_job.3);
    let reset = EmbeddingJobReset::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        terminal_job.embedding_job_id,
        fully_propagated_job.0,
        fully_propagated_job.1,
        "operator",
        "reviewed terminal embedding failure",
        36,
    )
    .unwrap();
    match activated.repository.reset_embedding_job(&reset).unwrap() {
        EmbeddingJobResetAck::Applied { job, .. } => {
            assert_eq!(job.attempt_generation, fully_propagated_job.0 + 1);
            assert!(job.terminal_error_class.is_none());
            assert!(job.failure_propagation_cursor.is_none());
            assert!(!job.failure_propagation_complete);
        }
        acknowledgement => panic!("unexpected fully propagated reset: {acknowledgement:?}"),
    }
}
