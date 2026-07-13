// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, anchor-observational recommendation orchestration.

#![allow(dead_code)] // Task 8 wires the prepared orchestrator into the runtime intercept.

use std::collections::BTreeSet;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use nemo_relay_types::api::llm::LlmApiFamily;
use uuid::{Uuid, Variant};

use crate::adapter::RouterRequestEnvelope;
use crate::candidate_set::{CandidateSetArtifactV1, build_candidate_set_v1};
use crate::canonical_json::canonical_sha256;
use crate::confidence::{
    AuditedNeighborV1, CandidateConfidenceInputV1, CandidateConfidenceReasonV1,
    CandidateEvidenceV1, ConfidenceBinaryLabelV1, ConfidenceDecisionReasonV1, ConfidenceDecisionV1,
    ConfidenceEngineV1, ConfidenceEvaluationInputV1, ConfidenceEvaluationSourceV1,
    ConfidenceGateResultV1, ConfidenceInputError, ConfidenceNeighborInputV1, ConfidencePolicyV1,
    ConfidenceTerminalClassV1, NeighborExclusionReasonV1,
};
use crate::decision_audit::{
    ActiveDecisionParentBindingV2, AuditF32V1, AuditF64V1, DECISION_AUDIT_BYTES_MAX,
    DecisionAuditError, DecisionAuditV1, DecisionBinaryLabelV1, DecisionCandidateInputV1,
    DecisionCandidateReasonV1, DecisionCandidateSummaryInputV1, DecisionFinalReasonV1,
    DecisionNeighborExclusionReasonV1, DecisionNeighborInputV1, DecisionParentInputV1,
    PreparedCanonicalQueryV1,
};
use crate::embedding_identity::CANONICAL_ROUTING_QUERY_SCHEMA_V1;
use crate::judge::{JudgeBinaryLabelV1, JudgeEvaluationSourceV1};
use crate::ledger::repository::shadow::ShadowTerminalClass;
use crate::ledger::repository::vector_registry::FrozenMappingKey;
use crate::ledger::repository::vector_search::{
    LiveProjectedNeighborSearch, ProjectedVectorNeighbor,
};
use crate::live_embedding::{LiveEmbeddingResult, PreparedLiveQueryV1};
use crate::preflight::{PreflightOutcome, contains_sensitive_control_material};
use crate::routing_partition::{
    RoutingPartitionArtifactV1, RoutingPartitionBaseArtifactV1, RoutingPartitionInputV1,
    build_routing_partition_base_v1, build_routing_partition_from_input_v1,
};
use crate::sqlite_vector_store::SqliteVecStore;
use crate::trajectory::PersistedCandidateFactV1;
use crate::vector::{NormalizedVector, VectorSpaceId};
use crate::vector_store::{VECTOR_TOP_K_MAX, VectorStoreError};

const AUDIT_FIXED_RESERVATION_BYTES: usize = 64 * 1024;
const AUDIT_CANDIDATE_RESERVATION_BYTES: usize = 16 * 1024;
const AUDIT_NEIGHBOR_RESERVATION_BYTES: usize = 4 * 1024;
const TRANSPORT_IDENTITY_MAX_BYTES: usize = 256;

/// Stable fail-closed outcomes that never retain user-controlled text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecommendationErrorV1 {
    DeadlineExceeded,
    RuntimeFailure,
    InvalidIdentity,
    InvalidPreflight,
    InvalidQuery,
    InvalidPartition,
    ResourceLimit,
    Confidence,
    DecisionAudit,
}

impl From<ConfidenceInputError> for RecommendationErrorV1 {
    fn from(_: ConfidenceInputError) -> Self {
        Self::Confidence
    }
}

impl From<DecisionAuditError> for RecommendationErrorV1 {
    fn from(_: DecisionAuditError) -> Self {
        Self::DecisionAudit
    }
}

/// Safe full preflight facts after executable candidate requests have been dropped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecommendationPreflightV1 {
    api_family: LlmApiFamily,
    transport_identity: String,
    candidates: Vec<PersistedCandidateFactV1>,
}

impl RecommendationPreflightV1 {
    /// Consume full preflight and discard every Shadow-only request clone.
    pub(crate) fn from_preflight(outcome: PreflightOutcome) -> Result<Self, RecommendationErrorV1> {
        let (preflight, envelope) = Self::from_active_preflight(outcome)?;
        drop(envelope);
        Ok(preflight)
    }

    /// Retain only the original lossless envelope for a later single-winner Active rewrite.
    pub(crate) fn from_active_preflight(
        outcome: PreflightOutcome,
    ) -> Result<(Self, RouterRequestEnvelope), RecommendationErrorV1> {
        let PreflightOutcome {
            replay_capability,
            envelope,
            request_projection,
            routing_projection,
            candidates,
            eligible_candidate_count,
            eligible_candidate_facts,
            rejected_candidates,
        } = outcome;
        drop((
            request_projection,
            routing_projection,
            candidates,
            rejected_candidates,
        ));
        if eligible_candidate_count != eligible_candidate_facts.len()
            || eligible_candidate_facts.is_empty()
            || replay_capability.transport_identity.is_empty()
            || replay_capability.transport_identity.len() > TRANSPORT_IDENTITY_MAX_BYTES
            || contains_sensitive_control_material(&replay_capability.transport_identity)
            || eligible_candidate_facts
                .iter()
                .map(|candidate| candidate.cost_rank)
                .collect::<BTreeSet<_>>()
                .len()
                != eligible_candidate_facts.len()
        {
            return Err(RecommendationErrorV1::InvalidPreflight);
        }
        build_candidate_set_v1(&eligible_candidate_facts)
            .map_err(|_| RecommendationErrorV1::InvalidPreflight)?;
        Ok((
            Self {
                api_family: replay_capability.api_family,
                transport_identity: replay_capability.transport_identity,
                candidates: eligible_candidate_facts,
            },
            envelope,
        ))
    }

    #[cfg(test)]
    pub(crate) fn from_safe_facts(
        api_family: LlmApiFamily,
        transport_identity: impl Into<String>,
        candidates: Vec<PersistedCandidateFactV1>,
    ) -> Result<Self, RecommendationErrorV1> {
        let transport_identity = transport_identity.into();
        if transport_identity.is_empty()
            || transport_identity.len() > TRANSPORT_IDENTITY_MAX_BYTES
            || contains_sensitive_control_material(&transport_identity)
            || candidates
                .iter()
                .map(|candidate| candidate.cost_rank)
                .collect::<BTreeSet<_>>()
                .len()
                != candidates.len()
        {
            return Err(RecommendationErrorV1::InvalidPreflight);
        }
        build_candidate_set_v1(&candidates).map_err(|_| RecommendationErrorV1::InvalidPreflight)?;
        Ok(Self {
            api_family,
            transport_identity,
            candidates,
        })
    }
}

/// Exact prepared-query authority retained until the embedding call releases its Arc.
///
/// This owner intentionally has no `Debug` implementation because it retains
/// bounded canonical application text until the audit is constructed.
pub(crate) struct RecommendationQueryV1 {
    prepared: Arc<PreparedLiveQueryV1>,
}

struct RecommendationAuditQueryV1 {
    mapping: FrozenMappingKey,
    vector_space_id: VectorSpaceId,
    prepared_query: PreparedCanonicalQueryV1,
}

impl RecommendationQueryV1 {
    pub(crate) fn from_prepared(
        prepared: Arc<PreparedLiveQueryV1>,
    ) -> Result<Self, RecommendationErrorV1> {
        let artifact = prepared.artifact();
        if artifact.query.schema != CANONICAL_ROUTING_QUERY_SCHEMA_V1
            || std::str::from_utf8(&artifact.canonical_bytes).is_err()
            || crate::ledger::repository::vector_catalog::validate_query_artifact(artifact).is_err()
        {
            return Err(RecommendationErrorV1::InvalidQuery);
        }
        Ok(Self { prepared })
    }

    pub(crate) fn for_embedding(&self) -> Arc<PreparedLiveQueryV1> {
        Arc::clone(&self.prepared)
    }

    fn mapping(&self) -> &FrozenMappingKey {
        self.prepared.mapping()
    }

    fn vector_space_id(&self) -> &VectorSpaceId {
        self.prepared.vector_space_id()
    }

    fn canonical_query_bytes_len(&self) -> usize {
        self.prepared.artifact().canonical_bytes.len()
    }

    fn into_audit_query(self) -> Result<RecommendationAuditQueryV1, RecommendationErrorV1> {
        let prepared =
            Arc::try_unwrap(self.prepared).map_err(|_| RecommendationErrorV1::InvalidQuery)?;
        let (mapping, vector_space_id, artifact) = prepared.into_audit_parts();
        Ok(RecommendationAuditQueryV1 {
            mapping,
            vector_space_id,
            prepared_query: PreparedCanonicalQueryV1::from_artifact(artifact)?,
        })
    }
}

/// Call and process facts needed to append exactly one observational decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RecommendationDecisionIdentityV1 {
    pub(crate) decision_id: Uuid,
    pub(crate) process_instance_id: Uuid,
    pub(crate) primary_call_uuid: Uuid,
    pub(crate) as_of_unix_ms: i64,
    pub(crate) created_at_unix_ms: i64,
    pub(crate) admitted_at: Instant,
}

/// Pessimistically reserved audit budget, debited before aggregate allocations.
struct RecommendationAuditBudgetV1 {
    limit: usize,
    charged: usize,
    worst_case: usize,
    top_k: usize,
}

impl RecommendationAuditBudgetV1 {
    fn reserve(
        canonical_query_bytes: usize,
        candidate_count: usize,
        top_k: usize,
    ) -> Result<Self, RecommendationErrorV1> {
        Self::reserve_with_limit(
            canonical_query_bytes,
            candidate_count,
            top_k,
            DECISION_AUDIT_BYTES_MAX,
        )
    }

    fn reserve_with_limit(
        canonical_query_bytes: usize,
        candidate_count: usize,
        top_k: usize,
        limit: usize,
    ) -> Result<Self, RecommendationErrorV1> {
        if candidate_count == 0 || candidate_count > 64 || !(1..=VECTOR_TOP_K_MAX).contains(&top_k)
        {
            return Err(RecommendationErrorV1::ResourceLimit);
        }
        let maximum_neighbors = candidate_count
            .checked_mul(top_k)
            .ok_or(RecommendationErrorV1::ResourceLimit)?;
        if maximum_neighbors > VECTOR_TOP_K_MAX {
            return Err(RecommendationErrorV1::ResourceLimit);
        }
        let charged = AUDIT_FIXED_RESERVATION_BYTES
            .checked_add(canonical_query_bytes)
            .ok_or(RecommendationErrorV1::ResourceLimit)?;
        let candidate_bytes = candidate_count
            .checked_mul(AUDIT_CANDIDATE_RESERVATION_BYTES)
            .ok_or(RecommendationErrorV1::ResourceLimit)?;
        let neighbor_bytes = maximum_neighbors
            .checked_mul(AUDIT_NEIGHBOR_RESERVATION_BYTES)
            .ok_or(RecommendationErrorV1::ResourceLimit)?;
        let worst_case = charged
            .checked_add(candidate_bytes)
            .and_then(|bytes| bytes.checked_add(neighbor_bytes))
            .ok_or(RecommendationErrorV1::ResourceLimit)?;
        if worst_case > limit {
            return Err(RecommendationErrorV1::ResourceLimit);
        }
        Ok(Self {
            limit,
            charged,
            worst_case,
            top_k,
        })
    }

    fn debit_candidate(&mut self) -> Result<(), RecommendationErrorV1> {
        self.debit(AUDIT_CANDIDATE_RESERVATION_BYTES)
    }

    fn debit_neighbors(&mut self, count: usize) -> Result<(), RecommendationErrorV1> {
        if count > self.top_k {
            return Err(RecommendationErrorV1::ResourceLimit);
        }
        let bytes = count
            .checked_mul(AUDIT_NEIGHBOR_RESERVATION_BYTES)
            .ok_or(RecommendationErrorV1::ResourceLimit)?;
        self.debit(bytes)
    }

    fn debit(&mut self, bytes: usize) -> Result<(), RecommendationErrorV1> {
        let charged = self
            .charged
            .checked_add(bytes)
            .ok_or(RecommendationErrorV1::ResourceLimit)?;
        if charged > self.limit || charged > self.worst_case {
            return Err(RecommendationErrorV1::ResourceLimit);
        }
        self.charged = charged;
        Ok(())
    }

    fn verify_exact(&self, command_size_bytes: usize) -> Result<(), RecommendationErrorV1> {
        if command_size_bytes > self.limit || command_size_bytes > self.charged {
            return Err(RecommendationErrorV1::ResourceLimit);
        }
        Ok(())
    }
}

/// Prepared recommendation state whose audit bound is reserved before embedding.
pub(crate) struct PreparedRecommendationV1 {
    identity: RecommendationDecisionIdentityV1,
    query: RecommendationQueryV1,
    confidence_policy: ConfidencePolicyV1,
    partition_base: RoutingPartitionBaseArtifactV1,
    candidates: Vec<PreparedCandidateV1>,
    candidate_set: CandidateSetArtifactV1,
    budget: RecommendationAuditBudgetV1,
    deadline: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveExperimentInputV2 {
    pub(crate) candidate_id: String,
    pub(crate) partition_hash: String,
}

struct ExecutingRecommendationV1 {
    identity: RecommendationDecisionIdentityV1,
    query: RecommendationAuditQueryV1,
    confidence_policy: ConfidencePolicyV1,
    partition_base: RoutingPartitionBaseArtifactV1,
    candidates: Vec<PreparedCandidateV1>,
    candidate_set: CandidateSetArtifactV1,
    budget: RecommendationAuditBudgetV1,
    deadline: Instant,
}

impl PreparedRecommendationV1 {
    /// Borrow the only cheap clone permitted for the live embedding call.
    pub(crate) fn query_for_embedding(&self) -> Arc<PreparedLiveQueryV1> {
        self.query.for_embedding()
    }

    pub(crate) fn canonical_query_hash(&self) -> &str {
        &self.query.prepared.artifact().canonical_query_hash
    }

    pub(crate) fn active_experiment_inputs(&self) -> Vec<ActiveExperimentInputV2> {
        self.candidates
            .iter()
            .map(|candidate| ActiveExperimentInputV2 {
                candidate_id: candidate.fact.candidate_id.clone(),
                partition_hash: candidate.partition.partition_hash.clone(),
            })
            .collect()
    }

    fn default_candidate_gates(&self) -> Vec<RecommendationCandidateGateV1> {
        self.candidates
            .iter()
            .map(|candidate| RecommendationCandidateGateV1 {
                candidate_id: candidate.fact.candidate_id.clone(),
                lower_bound_threshold: self.confidence_policy.promotion_lower_bound(),
                externally_authorized: true,
            })
            .collect()
    }

    fn into_executing(self) -> Result<ExecutingRecommendationV1, RecommendationErrorV1> {
        Ok(ExecutingRecommendationV1 {
            identity: self.identity,
            query: self.query.into_audit_query()?,
            confidence_policy: self.confidence_policy,
            partition_base: self.partition_base,
            candidates: self.candidates,
            candidate_set: self.candidate_set,
            budget: self.budget,
            deadline: self.deadline,
        })
    }
}

/// Ordered candidate-specific threshold and external-authority input for one query plan.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RecommendationCandidateGateV1 {
    pub(crate) candidate_id: String,
    pub(crate) lower_bound_threshold: f64,
    pub(crate) externally_authorized: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecommendationCandidateSearchIdentityV1 {
    pub(crate) candidate_id: String,
    pub(crate) sorted_neighbor_hash: String,
}

pub(crate) trait RecommendationCandidateGateResolverV1: Send {
    fn resolve_gate<'a>(
        &'a mut self,
        identity: RecommendationCandidateSearchIdentityV1,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<RecommendationCandidateGateV1, RecommendationErrorV1>>
                + Send
                + 'a,
        >,
    >;
}

/// Complete reusable query evaluation before mode-specific persistence and dispatch.
pub(crate) struct EvaluatedRecommendationQueryPlanV1 {
    prepared: ExecutingRecommendationV1,
    decision: ConfidenceDecisionV1,
    pub(crate) gates: Vec<RecommendationCandidateGateV1>,
    pub(crate) winner_candidate_id: Option<String>,
}

impl EvaluatedRecommendationQueryPlanV1 {
    fn into_recommendation_audit(self) -> Result<DecisionAuditV1, RecommendationErrorV1> {
        finish_audit(self.prepared, self.decision)
    }

    pub(crate) fn into_active_audit(
        self,
        binding: ActiveDecisionParentBindingV2,
        final_reason: DecisionFinalReasonV1,
    ) -> Result<DecisionAuditV1, RecommendationErrorV1> {
        finish_mode_audit(
            self.prepared,
            self.decision,
            Some(ActiveAuditFinalizationV2 {
                binding,
                final_reason,
                gates: self.gates,
            }),
        )
    }

    #[cfg(test)]
    fn summaries(&self) -> &[crate::confidence::CandidateConfidenceSummaryV1] {
        &self.decision.summaries
    }
}

struct ActiveAuditFinalizationV2 {
    binding: ActiveDecisionParentBindingV2,
    final_reason: DecisionFinalReasonV1,
    gates: Vec<RecommendationCandidateGateV1>,
}

enum CandidateGateSourceV1<'a> {
    Fixed(Vec<RecommendationCandidateGateV1>),
    Dynamic(&'a mut dyn RecommendationCandidateGateResolverV1),
}

struct PreparedCandidateV1 {
    fact: PersistedCandidateFactV1,
    partition: RoutingPartitionArtifactV1,
    partition_id: Option<i64>,
}

/// Validate all non-I/O authority and reserve the audit before embedding/search.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_recommendation_v1(
    identity: RecommendationDecisionIdentityV1,
    preflight: RecommendationPreflightV1,
    query: RecommendationQueryV1,
    confidence_policy: ConfidencePolicyV1,
    partition_base: RoutingPartitionBaseArtifactV1,
    partitions: Vec<RoutingPartitionArtifactV1>,
    deadline: Instant,
) -> Result<PreparedRecommendationV1, RecommendationErrorV1> {
    check_deadline(deadline)?;
    if ![
        identity.decision_id,
        identity.process_instance_id,
        identity.primary_call_uuid,
    ]
    .into_iter()
    .all(is_uuid_v7)
        || identity.as_of_unix_ms < 0
        || identity.created_at_unix_ms < 0
        || identity.admitted_at > deadline
    {
        return Err(RecommendationErrorV1::InvalidIdentity);
    }
    let candidate_set = build_candidate_set_v1(&preflight.candidates)
        .map_err(|_| RecommendationErrorV1::InvalidPreflight)?;
    if partitions.len() != candidate_set.candidate_count {
        return Err(RecommendationErrorV1::InvalidPartition);
    }
    ConfidenceEngineV1::new(
        &confidence_policy,
        candidate_set.candidate_count,
        identity.as_of_unix_ms,
    )?;
    validate_common_authority(&preflight, &query, &partition_base)?;
    let rebuilt_base = build_routing_partition_base_v1(&partition_base.base)
        .map_err(|_| RecommendationErrorV1::InvalidPartition)?;
    if rebuilt_base != partition_base {
        return Err(RecommendationErrorV1::InvalidPartition);
    }

    let mut candidates = Vec::with_capacity(candidate_set.candidate_count);
    for (fact, partition) in preflight.candidates.into_iter().zip(partitions) {
        let expected = build_routing_partition_from_input_v1(&RoutingPartitionInputV1 {
            base: partition_base.base.clone(),
            candidate_id: fact.candidate_id.clone(),
            candidate_model: fact.model.clone(),
            candidate_model_revision: fact.model_revision.clone(),
            decoding_fingerprint: fact.decoding_fingerprint.clone(),
        })
        .map_err(|_| RecommendationErrorV1::InvalidPartition)?;
        if expected != partition {
            return Err(RecommendationErrorV1::InvalidPartition);
        }
        candidates.push(PreparedCandidateV1 {
            fact,
            partition,
            partition_id: None,
        });
    }
    let budget = RecommendationAuditBudgetV1::reserve(
        query.canonical_query_bytes_len(),
        candidate_set.candidate_count,
        confidence_policy.top_k(),
    )?;
    check_deadline(deadline)?;
    Ok(PreparedRecommendationV1 {
        identity,
        query,
        confidence_policy,
        partition_base,
        candidates,
        candidate_set,
        budget,
        deadline,
    })
}

fn validate_common_authority(
    preflight: &RecommendationPreflightV1,
    query: &RecommendationQueryV1,
    partition_base: &RoutingPartitionBaseArtifactV1,
) -> Result<(), RecommendationErrorV1> {
    let base = &partition_base.base;
    if query.mapping().project_uuid.is_nil()
        || query.mapping().config_generation_id.len() != 64
        || query.mapping().policy_version_id.len() != 64
        || query.mapping().pool_id.is_empty()
        || query.mapping().policy_version_id != base.policy_version_id
        || query.vector_space_id().as_str() != base.vector_space_id
        || preflight.api_family != base.api_family
        || preflight.transport_identity != base.transport_identity
    {
        return Err(RecommendationErrorV1::InvalidPartition);
    }
    Ok(())
}

trait RecommendationVectorSearch {
    fn search_live_partition_until<'a>(
        &'a self,
        mapping: &'a FrozenMappingKey,
        expected_partition: &'a RoutingPartitionArtifactV1,
        query_vector: &'a NormalizedVector,
        top_k: usize,
        deadline: Instant,
    ) -> Pin<
        Box<dyn Future<Output = Result<LiveProjectedNeighborSearch, VectorStoreError>> + Send + 'a>,
    >;
}

impl RecommendationVectorSearch for SqliteVecStore {
    fn search_live_partition_until<'a>(
        &'a self,
        mapping: &'a FrozenMappingKey,
        expected_partition: &'a RoutingPartitionArtifactV1,
        query_vector: &'a NormalizedVector,
        top_k: usize,
        deadline: Instant,
    ) -> Pin<
        Box<dyn Future<Output = Result<LiveProjectedNeighborSearch, VectorStoreError>> + Send + 'a>,
    > {
        Box::pin(SqliteVecStore::search_live_partition_until(
            self,
            mapping,
            expected_partition,
            query_vector,
            top_k,
            deadline,
        ))
    }
}

/// Execute exact sequential searches and construct one immutable audit.
pub(crate) async fn run_recommendation_until(
    store: &SqliteVecStore,
    prepared: PreparedRecommendationV1,
    embedding: LiveEmbeddingResult,
) -> Result<DecisionAuditV1, RecommendationErrorV1> {
    run_with_search(store, prepared, embedding).await
}

/// Evaluate one query with ordered per-candidate hysteresis and authority gates.
pub(crate) async fn run_recommendation_query_plan_until(
    store: &SqliteVecStore,
    prepared: PreparedRecommendationV1,
    embedding: LiveEmbeddingResult,
    gates: Vec<RecommendationCandidateGateV1>,
) -> Result<EvaluatedRecommendationQueryPlanV1, RecommendationErrorV1> {
    run_with_search_gates(store, prepared, embedding, gates).await
}

pub(crate) async fn run_recommendation_query_plan_with_resolver_until(
    store: &SqliteVecStore,
    prepared: PreparedRecommendationV1,
    embedding: LiveEmbeddingResult,
    resolver: &mut dyn RecommendationCandidateGateResolverV1,
) -> Result<EvaluatedRecommendationQueryPlanV1, RecommendationErrorV1> {
    run_with_search_source(
        store,
        prepared,
        embedding,
        CandidateGateSourceV1::Dynamic(resolver),
    )
    .await
}

async fn run_with_search<S: RecommendationVectorSearch + Sync>(
    store: &S,
    prepared: PreparedRecommendationV1,
    embedding: LiveEmbeddingResult,
) -> Result<DecisionAuditV1, RecommendationErrorV1> {
    let gates = prepared.default_candidate_gates();
    run_with_search_gates(store, prepared, embedding, gates)
        .await
        .and_then(EvaluatedRecommendationQueryPlanV1::into_recommendation_audit)
}

async fn run_with_search_gates<S: RecommendationVectorSearch + Sync>(
    store: &S,
    prepared: PreparedRecommendationV1,
    embedding: LiveEmbeddingResult,
    gates: Vec<RecommendationCandidateGateV1>,
) -> Result<EvaluatedRecommendationQueryPlanV1, RecommendationErrorV1> {
    run_with_search_source(
        store,
        prepared,
        embedding,
        CandidateGateSourceV1::Fixed(gates),
    )
    .await
}

async fn run_with_search_source<S: RecommendationVectorSearch + Sync>(
    store: &S,
    prepared: PreparedRecommendationV1,
    embedding: LiveEmbeddingResult,
    mut gate_source: CandidateGateSourceV1<'_>,
) -> Result<EvaluatedRecommendationQueryPlanV1, RecommendationErrorV1> {
    check_deadline(prepared.deadline)?;
    let mut prepared = prepared.into_executing()?;
    check_deadline(prepared.deadline)?;
    if matches!(
        &gate_source,
        CandidateGateSourceV1::Fixed(gates)
            if gates.len() != prepared.candidates.len()
                || gates.iter().zip(&prepared.candidates).any(|(gate, candidate)| {
                    !candidate_gate_is_valid(gate, candidate, &prepared.confidence_policy)
                })
    ) {
        return Err(RecommendationErrorV1::InvalidIdentity);
    }
    let mut resolved_gates = Vec::with_capacity(prepared.candidates.len());
    let confidence_policy = prepared.confidence_policy.clone();
    let mut engine = ConfidenceEngineV1::new(
        &confidence_policy,
        prepared.candidates.len(),
        prepared.identity.as_of_unix_ms,
    )?;

    let vector = match embedding {
        LiveEmbeddingResult::Unavailable => {
            engine.stop_with_external_fallback(ConfidenceDecisionReasonV1::EmbeddingUnavailable)?;
            record_remaining_fallbacks(&mut prepared, &mut engine, 0, false)?;
            let decision = engine.finish()?;
            complete_unresolved_gates(&mut resolved_gates, &gate_source, &prepared);
            return finish_query_plan(prepared, decision, resolved_gates);
        }
        LiveEmbeddingResult::Ready(vector) => vector,
    };
    if vector.vector_space_id() != &prepared.query.vector_space_id {
        engine.stop_with_external_fallback(ConfidenceDecisionReasonV1::VersionMismatch)?;
        record_remaining_fallbacks(&mut prepared, &mut engine, 0, false)?;
        let decision = engine.finish()?;
        complete_unresolved_gates(&mut resolved_gates, &gate_source, &prepared);
        return finish_query_plan(prepared, decision, resolved_gates);
    }

    let mut evidence_ids = BTreeSet::new();
    let mut index = 0;
    while index < prepared.candidates.len() {
        prepared.budget.debit_candidate()?;
        if !engine.needs_evidence() {
            let candidate = &prepared.candidates[index].fact;
            engine.record_unevaluated(candidate.candidate_id.clone(), candidate.cost_rank)?;
            resolved_gates.push(gate_without_authority(
                &prepared.candidates[index],
                &prepared.confidence_policy,
            ));
            index += 1;
            continue;
        }
        check_deadline(prepared.deadline)?;
        let candidate_id = prepared.candidates[index].fact.candidate_id.clone();
        let cost_rank = prepared.candidates[index].fact.cost_rank;
        let search = store
            .search_live_partition_until(
                &prepared.query.mapping,
                &prepared.candidates[index].partition,
                vector.vector(),
                prepared.confidence_policy.top_k(),
                prepared.deadline,
            )
            .await;
        check_deadline(prepared.deadline)?;
        match search {
            Ok(LiveProjectedNeighborSearch::NoPartition) => {
                let gate = resolve_candidate_gate(
                    &mut gate_source,
                    index,
                    &prepared,
                    sorted_neighbor_hash(std::iter::empty())?,
                )
                .await?;
                engine.evaluate_candidate_with_gate(
                    CandidateConfidenceInputV1::new(
                        candidate_id,
                        cost_rank,
                        CandidateEvidenceV1::NoPartition,
                    )?,
                    gate.lower_bound_threshold,
                    gate.externally_authorized,
                )?;
                resolved_gates.push(gate);
            }
            Ok(LiveProjectedNeighborSearch::Found {
                partition_id,
                neighbors,
            }) => {
                if neighbors.len() > prepared.confidence_policy.top_k() {
                    stop_external_at(
                        &mut prepared,
                        &mut engine,
                        index,
                        ConfidenceDecisionReasonV1::VectorUnhealthy,
                    )?;
                    break;
                }
                prepared.budget.debit_neighbors(neighbors.len())?;
                if neighbors.iter().any(|neighbor| {
                    neighbor.learning_generation_id
                        != prepared.partition_base.base.learning_generation_id
                }) {
                    stop_external_at(
                        &mut prepared,
                        &mut engine,
                        index,
                        ConfidenceDecisionReasonV1::VersionMismatch,
                    )?;
                    break;
                }
                let candidate_evidence_ids = neighbors
                    .iter()
                    .map(|neighbor| neighbor.vector_match.record_id().value())
                    .collect::<BTreeSet<_>>();
                if candidate_evidence_ids.len() != neighbors.len()
                    || candidate_evidence_ids
                        .iter()
                        .any(|evidence_id| evidence_ids.contains(evidence_id))
                {
                    stop_external_at(
                        &mut prepared,
                        &mut engine,
                        index,
                        ConfidenceDecisionReasonV1::VectorUnhealthy,
                    )?;
                    break;
                }
                let neighbor_hash = sorted_neighbor_hash(
                    neighbors
                        .iter()
                        .map(|neighbor| neighbor.vector_match.record_id().value()),
                )?;
                let gate =
                    resolve_candidate_gate(&mut gate_source, index, &prepared, neighbor_hash)
                        .await?;
                let confidence_neighbors = match neighbors
                    .into_iter()
                    .map(project_confidence_neighbor)
                    .collect::<Result<Vec<_>, _>>()
                {
                    Ok(neighbors) => neighbors,
                    Err(_) => {
                        stop_external_at(
                            &mut prepared,
                            &mut engine,
                            index,
                            ConfidenceDecisionReasonV1::VectorUnhealthy,
                        )?;
                        break;
                    }
                };
                evidence_ids.extend(candidate_evidence_ids);
                prepared.candidates[index].partition_id = Some(partition_id.value());
                engine.evaluate_candidate_with_gate(
                    CandidateConfidenceInputV1::new(
                        candidate_id,
                        cost_rank,
                        CandidateEvidenceV1::Neighbors(confidence_neighbors),
                    )?,
                    gate.lower_bound_threshold,
                    gate.externally_authorized,
                )?;
                resolved_gates.push(gate);
            }
            Ok(
                LiveProjectedNeighborSearch::AuthorityNotFound
                | LiveProjectedNeighborSearch::StaleLearningGeneration,
            ) => {
                stop_external_at(
                    &mut prepared,
                    &mut engine,
                    index,
                    ConfidenceDecisionReasonV1::VersionMismatch,
                )?;
                break;
            }
            Err(error) => {
                let reason = match error {
                    VectorStoreError::SpaceNotFound
                    | VectorStoreError::InvalidVector(
                        crate::vector::VectorError::VectorSpaceMismatch
                        | crate::vector::VectorError::DimensionMismatch,
                    ) => ConfidenceDecisionReasonV1::VersionMismatch,
                    _ => ConfidenceDecisionReasonV1::VectorUnhealthy,
                };
                stop_external_at(&mut prepared, &mut engine, index, reason)?;
                break;
            }
        }
        index += 1;
    }
    let decision = engine.finish()?;
    complete_unresolved_gates(&mut resolved_gates, &gate_source, &prepared);
    finish_query_plan(prepared, decision, resolved_gates)
}

async fn resolve_candidate_gate(
    source: &mut CandidateGateSourceV1<'_>,
    index: usize,
    prepared: &ExecutingRecommendationV1,
    sorted_neighbor_hash: String,
) -> Result<RecommendationCandidateGateV1, RecommendationErrorV1> {
    let candidate = prepared
        .candidates
        .get(index)
        .ok_or(RecommendationErrorV1::InvalidIdentity)?;
    let gate = match source {
        CandidateGateSourceV1::Fixed(gates) => gates
            .get(index)
            .cloned()
            .ok_or(RecommendationErrorV1::InvalidIdentity)?,
        CandidateGateSourceV1::Dynamic(resolver) => {
            resolver
                .resolve_gate(RecommendationCandidateSearchIdentityV1 {
                    candidate_id: candidate.fact.candidate_id.clone(),
                    sorted_neighbor_hash,
                })
                .await?
        }
    };
    if !candidate_gate_is_valid(&gate, candidate, &prepared.confidence_policy) {
        return Err(RecommendationErrorV1::InvalidIdentity);
    }
    Ok(gate)
}

fn candidate_gate_is_valid(
    gate: &RecommendationCandidateGateV1,
    candidate: &PreparedCandidateV1,
    policy: &ConfidencePolicyV1,
) -> bool {
    gate.candidate_id == candidate.fact.candidate_id
        && gate.lower_bound_threshold.is_finite()
        && (0.0..=policy.promotion_lower_bound()).contains(&gate.lower_bound_threshold)
}

fn gate_without_authority(
    candidate: &PreparedCandidateV1,
    policy: &ConfidencePolicyV1,
) -> RecommendationCandidateGateV1 {
    RecommendationCandidateGateV1 {
        candidate_id: candidate.fact.candidate_id.clone(),
        lower_bound_threshold: policy.promotion_lower_bound(),
        externally_authorized: false,
    }
}

fn complete_unresolved_gates(
    resolved: &mut Vec<RecommendationCandidateGateV1>,
    source: &CandidateGateSourceV1<'_>,
    prepared: &ExecutingRecommendationV1,
) {
    while resolved.len() < prepared.candidates.len() {
        let index = resolved.len();
        let gate = match source {
            CandidateGateSourceV1::Fixed(gates) => gates[index].clone(),
            CandidateGateSourceV1::Dynamic(_) => {
                gate_without_authority(&prepared.candidates[index], &prepared.confidence_policy)
            }
        };
        resolved.push(gate);
    }
}

fn sorted_neighbor_hash(
    identifiers: impl IntoIterator<Item = Uuid>,
) -> Result<String, RecommendationErrorV1> {
    let mut identifiers = identifiers
        .into_iter()
        .map(|identifier| identifier.to_string())
        .collect::<Vec<_>>();
    identifiers.sort_unstable();
    canonical_sha256(&serde_json::json!({
        "schema": "nemo.relay.router.active-sorted-neighbors@1",
        "evidence_vector_link_ids": identifiers,
    }))
    .map_err(|_| RecommendationErrorV1::InvalidIdentity)
}

fn finish_query_plan(
    prepared: ExecutingRecommendationV1,
    decision: ConfidenceDecisionV1,
    gates: Vec<RecommendationCandidateGateV1>,
) -> Result<EvaluatedRecommendationQueryPlanV1, RecommendationErrorV1> {
    let winner_candidate_id = decision.recommended_candidate_id.clone();
    Ok(EvaluatedRecommendationQueryPlanV1 {
        prepared,
        decision,
        gates,
        winner_candidate_id,
    })
}

fn stop_external_at(
    prepared: &mut ExecutingRecommendationV1,
    engine: &mut ConfidenceEngineV1<'_>,
    index: usize,
    reason: ConfidenceDecisionReasonV1,
) -> Result<(), RecommendationErrorV1> {
    engine.stop_with_external_fallback(reason)?;
    record_remaining_fallbacks(prepared, engine, index, true)
}

fn record_remaining_fallbacks(
    prepared: &mut ExecutingRecommendationV1,
    engine: &mut ConfidenceEngineV1<'_>,
    start: usize,
    first_already_debited: bool,
) -> Result<(), RecommendationErrorV1> {
    for index in start..prepared.candidates.len() {
        if index != start || !first_already_debited {
            prepared.budget.debit_candidate()?;
        }
        let candidate = &prepared.candidates[index].fact;
        engine.record_unevaluated(candidate.candidate_id.clone(), candidate.cost_rank)?;
    }
    Ok(())
}

fn project_confidence_neighbor(
    neighbor: ProjectedVectorNeighbor,
) -> Result<ConfidenceNeighborInputV1, ConfidenceInputError> {
    let terminal_class = match neighbor.terminal_class {
        ShadowTerminalClass::Completed => ConfidenceTerminalClassV1::Completed,
        ShadowTerminalClass::DeterministicFailure => {
            ConfidenceTerminalClassV1::DeterministicFailure
        }
        ShadowTerminalClass::OperationalFailure => ConfidenceTerminalClassV1::OperationalFailure,
        ShadowTerminalClass::SkippedCooloff => ConfidenceTerminalClassV1::SkippedCooloff,
        ShadowTerminalClass::CanceledShutdown => ConfidenceTerminalClassV1::CanceledShutdown,
        ShadowTerminalClass::OrphanedBeforeSchedule => {
            ConfidenceTerminalClassV1::OrphanedBeforeSchedule
        }
        ShadowTerminalClass::OrphanedInFlight => ConfidenceTerminalClassV1::OrphanedInFlight,
    };
    let evaluation = neighbor
        .evaluation
        .map(|evaluation| {
            ConfidenceEvaluationInputV1::new(
                evaluation.evaluation_id,
                match evaluation.source {
                    JudgeEvaluationSourceV1::DeterministicValidator => {
                        ConfidenceEvaluationSourceV1::DeterministicValidator
                    }
                    JudgeEvaluationSourceV1::Judge => ConfidenceEvaluationSourceV1::Judge,
                },
                evaluation.binary_label.map(|label| match label {
                    JudgeBinaryLabelV1::Pass => ConfidenceBinaryLabelV1::Pass,
                    JudgeBinaryLabelV1::Fail => ConfidenceBinaryLabelV1::Fail,
                }),
                evaluation.judge_confidence.map(|value| value.bits),
                evaluation.promotion_eligible,
                evaluation.created_at_unix_ms,
            )
        })
        .transpose()?;
    ConfidenceNeighborInputV1::new(
        neighbor.vector_match.record_id().value(),
        neighbor.shadow_attempt_id,
        neighbor.anchor_id,
        neighbor.root_uuid,
        terminal_class,
        neighbor.vector_match.distance().to_bits(),
        evaluation,
    )
}

fn finish_audit(
    prepared: ExecutingRecommendationV1,
    decision: ConfidenceDecisionV1,
) -> Result<DecisionAuditV1, RecommendationErrorV1> {
    finish_mode_audit(prepared, decision, None)
}

fn finish_mode_audit(
    prepared: ExecutingRecommendationV1,
    decision: ConfidenceDecisionV1,
    active: Option<ActiveAuditFinalizationV2>,
) -> Result<DecisionAuditV1, RecommendationErrorV1> {
    check_deadline(prepared.deadline)?;
    if decision.summaries.len() != prepared.candidates.len() {
        return Err(RecommendationErrorV1::Confidence);
    }
    if active.as_ref().is_some_and(|active| {
        !active.final_reason.is_active()
            || active.gates.len() != prepared.candidates.len()
            || active
                .gates
                .iter()
                .zip(&prepared.candidates)
                .any(|(gate, candidate)| gate.candidate_id != candidate.fact.candidate_id)
    }) {
        return Err(RecommendationErrorV1::InvalidIdentity);
    }
    let final_reason = active.as_ref().map_or_else(
        || map_final_reason(decision.reason),
        |active| active.final_reason,
    );
    let winner_index = decision
        .recommended_candidate_id
        .as_ref()
        .and_then(|winner| {
            prepared
                .candidates
                .iter()
                .position(|candidate| &candidate.fact.candidate_id == winner)
        });
    if decision.recommended_candidate_id.is_some() != winner_index.is_some() {
        return Err(RecommendationErrorV1::Confidence);
    }
    if matches!(
        final_reason,
        DecisionFinalReasonV1::ActiveCandidate
            | DecisionFinalReasonV1::ActiveAnchorControl
            | DecisionFinalReasonV1::ActiveAnchorHoldout
    ) && winner_index.is_none()
    {
        return Err(RecommendationErrorV1::Confidence);
    }
    let base = &prepared.partition_base.base;
    let (candidate_id, recommended_model, recommended_model_revision) = winner_index.map_or_else(
        || {
            (
                None,
                base.anchor_model.clone(),
                base.anchor_revision.clone(),
            )
        },
        |index| {
            let fact = &prepared.candidates[index].fact;
            (
                Some(fact.candidate_id.clone()),
                fact.model.clone(),
                fact.model_revision.clone(),
            )
        },
    );
    let serve_candidate = final_reason == DecisionFinalReasonV1::ActiveCandidate;
    let (served_model, served_model_revision) = if serve_candidate {
        (
            recommended_model.clone(),
            recommended_model_revision.clone(),
        )
    } else {
        (base.anchor_model.clone(), base.anchor_revision.clone())
    };
    let decision_latency_ms = u64::try_from(
        Instant::now()
            .saturating_duration_since(prepared.identity.admitted_at)
            .as_millis(),
    )
    .map_err(|_| RecommendationErrorV1::InvalidIdentity)?;
    let parent = DecisionParentInputV1 {
        decision_id: prepared.identity.decision_id,
        project_uuid: prepared.query.mapping.project_uuid,
        process_instance_id: prepared.identity.process_instance_id,
        config_generation_id: prepared.query.mapping.config_generation_id.clone(),
        policy_version_id: prepared.query.mapping.policy_version_id.clone(),
        learning_generation_id: base.learning_generation_id,
        pool_id: prepared.query.mapping.pool_id.clone(),
        candidate_id,
        primary_call_uuid: prepared.identity.primary_call_uuid,
        canonical_query_hash: prepared
            .query
            .prepared_query
            .canonical_query_hash()
            .to_string(),
        partition_base_json: prepared.partition_base.canonical_json.clone(),
        partition_base_hash: prepared.partition_base.partition_base_hash.clone(),
        vector_space_id: prepared.query.vector_space_id.as_str().to_string(),
        candidate_set_hash: prepared.candidate_set.candidate_set_hash.clone(),
        recommended_model,
        recommended_model_revision,
        served_model,
        served_model_revision,
        as_of_unix_ms: prepared.identity.as_of_unix_ms,
        decision_latency_ms,
        final_reason,
        created_at_unix_ms: prepared.identity.created_at_unix_ms,
    };
    let candidate_alpha = (1.0 - prepared.confidence_policy.familywise_credible_level())
        / prepared.candidates.len() as f64;
    let mut neighbor_ordinal = 0;
    let mut candidate_inputs = Vec::with_capacity(prepared.candidates.len());
    for (rank_ordinal, (summary, candidate)) in decision
        .summaries
        .into_iter()
        .zip(prepared.candidates)
        .enumerate()
    {
        if summary.candidate_id != candidate.fact.candidate_id
            || summary.cost_rank != candidate.fact.cost_rank
        {
            return Err(RecommendationErrorV1::Confidence);
        }
        let neighbors = summary
            .neighbors
            .iter()
            .map(|neighbor| {
                let input = map_decision_neighbor(
                    neighbor,
                    &summary.candidate_id,
                    base.learning_generation_id,
                    neighbor_ordinal,
                )?;
                neighbor_ordinal += 1;
                Ok(input)
            })
            .collect::<Result<Vec<_>, RecommendationErrorV1>>()?;
        let selected_root_count = summary
            .neighbors
            .iter()
            .filter(|neighbor| neighbor.selected_root)
            .count();
        let summary_input = DecisionCandidateSummaryInputV1 {
            candidate_id: summary.candidate_id,
            rank_ordinal,
            candidate_model: candidate.fact.model,
            candidate_model_revision: candidate.fact.model_revision,
            cost_rank: summary.cost_rank,
            learning_generation_id: base.learning_generation_id,
            vector_space_id: prepared.query.vector_space_id.as_str().to_string(),
            partition_id: candidate.partition_id,
            decoding_fingerprint: candidate.fact.decoding_fingerprint,
            top_k: prepared.confidence_policy.top_k(),
            radius: AuditF64V1::new(prepared.confidence_policy.radius())?,
            min_points: prepared.confidence_policy.min_points(),
            min_independent_roots: prepared.confidence_policy.min_independent_roots(),
            min_effective_samples: AuditF64V1::new(
                prepared.confidence_policy.min_effective_samples(),
            )?,
            min_coverage: AuditF64V1::new(prepared.confidence_policy.min_coverage())?,
            time_decay_half_life_seconds: AuditF64V1::new(
                prepared.confidence_policy.time_decay_half_life_seconds(),
            )?,
            prior_success: AuditF64V1::new(prepared.confidence_policy.prior_success())?,
            prior_failure: AuditF64V1::new(prepared.confidence_policy.prior_failure())?,
            familywise_credible_level: AuditF64V1::new(
                prepared.confidence_policy.familywise_credible_level(),
            )?,
            candidate_alpha: AuditF64V1::new(candidate_alpha)?,
            promotion_lower_bound: AuditF64V1::new(active.as_ref().map_or_else(
                || prepared.confidence_policy.promotion_lower_bound(),
                |active| active.gates[rank_ordinal].lower_bound_threshold,
            ))?,
            returned_neighbor_count: summary.top_k_points,
            within_radius_count: summary.raw_points,
            labeled_point_count: summary.labeled_points,
            attempted_root_count: summary.attempted_roots,
            labeled_root_count: summary.labeled_roots,
            selected_root_count,
            coverage: audit_optional(summary.coverage)?,
            sum_weight: audit_optional(summary.sum_weight)?,
            sum_weighted_label: audit_optional(summary.sum_weighted_pass)?,
            sum_squared_weight: audit_optional(summary.sum_weight_squared)?,
            p_hat: audit_optional(summary.p_hat)?,
            effective_sample_size: audit_optional(summary.n_eff)?,
            beta_alpha: audit_optional(summary.beta_alpha)?,
            beta_beta: audit_optional(summary.beta_beta)?,
            lower_bound: audit_optional(summary.lower_bound)?,
            partition_gate_passed: map_gate(summary.gates.partition),
            points_gate_passed: map_gate(summary.gates.points),
            roots_gate_passed: map_gate(summary.gates.roots),
            coverage_gate_passed: map_gate(summary.gates.coverage),
            weight_gate_passed: map_gate(summary.gates.weight_math),
            effective_samples_gate_passed: map_gate(summary.gates.effective_samples),
            beta_quantile_gate_passed: map_gate(summary.gates.beta_quantile),
            lower_bound_gate_passed: map_gate(summary.gates.lower_bound),
            terminal_reason: map_candidate_reason(summary.reason),
        };
        candidate_inputs.push(DecisionCandidateInputV1 {
            summary: summary_input,
            partition_artifact: candidate.partition,
            neighbors,
        });
    }
    let audit = match active {
        Some(active) => DecisionAuditV1::new_active(
            parent,
            active.binding,
            candidate_inputs,
            prepared.query.prepared_query,
        )?,
        None => DecisionAuditV1::new(parent, candidate_inputs, prepared.query.prepared_query)?,
    };
    prepared.budget.verify_exact(audit.command_size_bytes)?;
    check_deadline(prepared.deadline)?;
    Ok(audit)
}

fn map_decision_neighbor(
    neighbor: &AuditedNeighborV1,
    candidate_id: &str,
    learning_generation_id: Uuid,
    neighbor_ordinal: usize,
) -> Result<DecisionNeighborInputV1, RecommendationErrorV1> {
    Ok(DecisionNeighborInputV1 {
        neighbor_ordinal,
        candidate_id: candidate_id.to_string(),
        candidate_neighbor_ordinal: neighbor.candidate_ordinal,
        evidence_vector_link_id: neighbor.evidence_vector_link_id,
        shadow_attempt_id: neighbor.shadow_attempt_id,
        anchor_id: neighbor.anchor_id,
        evaluation_id: neighbor.evaluation_id,
        learning_generation_id,
        distance: AuditF32V1::from_bits(neighbor.distance_bits)?,
        age_millis: neighbor.age_millis,
        similarity_weight: audit_optional(neighbor.similarity_weight)?,
        time_weight: audit_optional(neighbor.time_weight)?,
        final_weight: audit_optional(neighbor.final_weight)?,
        binary_label: neighbor.binary_label.map(|label| match label {
            ConfidenceBinaryLabelV1::Pass => DecisionBinaryLabelV1::Pass,
            ConfidenceBinaryLabelV1::Fail => DecisionBinaryLabelV1::Fail,
        }),
        selected_for_root: neighbor.selected_root,
        root_group_ordinal: neighbor.root_group_ordinal,
        exclusion_reason: match neighbor.exclusion_reason {
            NeighborExclusionReasonV1::OutsideRadius => {
                DecisionNeighborExclusionReasonV1::OutsideRadius
            }
            NeighborExclusionReasonV1::IneligibleQuality => {
                DecisionNeighborExclusionReasonV1::IneligibleQuality
            }
            NeighborExclusionReasonV1::DuplicateRoot => {
                DecisionNeighborExclusionReasonV1::DuplicateRoot
            }
            NeighborExclusionReasonV1::Included => DecisionNeighborExclusionReasonV1::Included,
        },
    })
}

fn audit_optional(value: Option<f64>) -> Result<Option<AuditF64V1>, RecommendationErrorV1> {
    value.map(AuditF64V1::new).transpose().map_err(Into::into)
}

fn map_gate(gate: ConfidenceGateResultV1) -> Option<bool> {
    match gate {
        ConfidenceGateResultV1::NotEvaluated => None,
        ConfidenceGateResultV1::Passed => Some(true),
        ConfidenceGateResultV1::Failed => Some(false),
    }
}

fn map_candidate_reason(reason: CandidateConfidenceReasonV1) -> DecisionCandidateReasonV1 {
    match reason {
        CandidateConfidenceReasonV1::NoPartition => DecisionCandidateReasonV1::NoPartition,
        CandidateConfidenceReasonV1::SparsePoints => DecisionCandidateReasonV1::SparsePoints,
        CandidateConfidenceReasonV1::InsufficientRoots => {
            DecisionCandidateReasonV1::InsufficientRoots
        }
        CandidateConfidenceReasonV1::LowCoverage => DecisionCandidateReasonV1::LowCoverage,
        CandidateConfidenceReasonV1::InvalidEvidenceTime => {
            DecisionCandidateReasonV1::InvalidEvidenceTime
        }
        CandidateConfidenceReasonV1::NumericError => DecisionCandidateReasonV1::NumericError,
        CandidateConfidenceReasonV1::InsufficientEffectiveSamples => {
            DecisionCandidateReasonV1::InsufficientEffectiveSamples
        }
        CandidateConfidenceReasonV1::LowerBoundBelowThreshold => {
            DecisionCandidateReasonV1::LowerBoundBelowThreshold
        }
        CandidateConfidenceReasonV1::Passed => DecisionCandidateReasonV1::Passed,
        CandidateConfidenceReasonV1::NotEvaluatedAfterWinner => {
            DecisionCandidateReasonV1::NotEvaluatedAfterWinner
        }
        CandidateConfidenceReasonV1::NotEvaluatedAfterFallback => {
            DecisionCandidateReasonV1::NotEvaluatedAfterFallback
        }
    }
}

fn map_final_reason(reason: ConfidenceDecisionReasonV1) -> DecisionFinalReasonV1 {
    match reason {
        ConfidenceDecisionReasonV1::EmbeddingUnavailable => {
            DecisionFinalReasonV1::EmbeddingUnavailable
        }
        ConfidenceDecisionReasonV1::VectorUnhealthy => DecisionFinalReasonV1::VectorUnhealthy,
        ConfidenceDecisionReasonV1::VersionMismatch => DecisionFinalReasonV1::VersionMismatch,
        ConfidenceDecisionReasonV1::RecommendObserveOnly => {
            DecisionFinalReasonV1::RecommendObserveOnly
        }
        ConfidenceDecisionReasonV1::NoCandidatePassed => DecisionFinalReasonV1::NoCandidatePassed,
        ConfidenceDecisionReasonV1::InvalidEvidenceTime => {
            DecisionFinalReasonV1::InvalidEvidenceTime
        }
        ConfidenceDecisionReasonV1::NumericError => DecisionFinalReasonV1::NumericError,
    }
}

fn check_deadline(deadline: Instant) -> Result<(), RecommendationErrorV1> {
    (Instant::now() < deadline)
        .then_some(())
        .ok_or(RecommendationErrorV1::DeadlineExceeded)
}

fn is_uuid_v7(value: Uuid) -> bool {
    value.get_version_num() == 7 && value.get_variant() == Variant::RFC4122
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::time::Duration;

    use crate::canonical_query::{
        CanonicalRoutingQueryArtifactV1, CanonicalRoutingQueryV1, CanonicalTaskV1,
    };
    use crate::embedding_identity::CANONICAL_ROUTING_QUERY_SCHEMA_V1;
    use crate::fingerprint::{canonical_serialize_bytes, sha256_hex};
    use crate::judge::ScoredValueV1;
    use crate::ledger::repository::vector_search::ProjectedNeighborEvaluation;
    use crate::routing_partition::RoutingPartitionBaseV1;
    use crate::trajectory::{CANDIDATE_FACT_SCHEMA_V1, PersistedCandidateCapabilitiesV1};
    use crate::vector::{AuthoritativeVector, PartitionId, VectorDimensions, VectorRecordId};
    use crate::vector_store::VectorMatch;

    use super::*;

    const AS_OF_UNIX_MS: i64 = 1_800_000_000_000;

    fn hash(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn uuid(suffix: u64) -> Uuid {
        Uuid::parse_str(&format!("018f1e66-0000-7000-8000-{suffix:012x}")).unwrap()
    }

    fn canonical_query_artifact() -> CanonicalRoutingQueryArtifactV1 {
        let query = CanonicalRoutingQueryV1 {
            schema: CANONICAL_ROUTING_QUERY_SCHEMA_V1.to_string(),
            instructions: Vec::new(),
            current_task: CanonicalTaskV1 {
                text: "route this request".to_string(),
            },
            bounded_context: Vec::new(),
            tool_schema_fingerprint: hash('1'),
            response_schema_fingerprint: None,
            required_capabilities: Vec::new(),
            position_features: None,
        };
        let canonical_bytes = canonical_serialize_bytes(&query).unwrap();
        let canonical_query_hash = sha256_hex(&canonical_bytes);
        CanonicalRoutingQueryArtifactV1 {
            query,
            canonical_bytes,
            canonical_query_hash,
        }
    }

    fn candidate(index: usize) -> PersistedCandidateFactV1 {
        PersistedCandidateFactV1 {
            schema: CANDIDATE_FACT_SCHEMA_V1.to_string(),
            candidate_id: format!("candidate-{index}"),
            model: format!("candidate-model-{index}"),
            model_revision: "r1".to_string(),
            cost_rank: u32::try_from(index + 1).unwrap(),
            capabilities: PersistedCandidateCapabilitiesV1 {
                tools: false,
                multimodal_input: false,
                structured_output: false,
                reasoning_controls: false,
            },
            decoding_fingerprint: hash('d'),
        }
    }

    struct Fixture {
        prepared: PreparedRecommendationV1,
        vector: AuthoritativeVector,
        learning_generation_id: Uuid,
    }

    fn fixture(candidate_count: usize) -> Fixture {
        let project_uuid = uuid(1);
        let process_instance_id = uuid(2);
        let learning_generation_id = uuid(3);
        let policy_version_id = hash('c');
        let vector_space_id = VectorSpaceId::new(hash('6')).unwrap();
        let mapping =
            FrozenMappingKey::new(project_uuid, hash('7'), "pool-a", policy_version_id.clone())
                .unwrap();
        let live_query = PreparedLiveQueryV1::from_test_parts(
            mapping,
            vector_space_id.clone(),
            canonical_query_artifact(),
            Duration::from_secs(5),
        );
        let query = RecommendationQueryV1::from_prepared(live_query).unwrap();
        let facts = (0..candidate_count).map(candidate).collect::<Vec<_>>();
        let preflight = RecommendationPreflightV1::from_safe_facts(
            LlmApiFamily::OpenAIChatCompletions,
            "transport-v1",
            facts.clone(),
        )
        .unwrap();
        let base = RoutingPartitionBaseV1 {
            tenant_policy_hash: hash('a'),
            agent_policy_hash: hash('b'),
            policy_version_id,
            learning_generation_id,
            api_family: LlmApiFamily::OpenAIChatCompletions,
            transport_identity: "transport-v1".to_string(),
            anchor_model: "anchor".to_string(),
            anchor_revision: "anchor-r1".to_string(),
            evaluator_version: hash('e'),
            vector_space_id: vector_space_id.as_str().to_string(),
        };
        let partition_base = build_routing_partition_base_v1(&base).unwrap();
        let partitions = facts
            .iter()
            .map(|fact| {
                build_routing_partition_from_input_v1(&RoutingPartitionInputV1 {
                    base: base.clone(),
                    candidate_id: fact.candidate_id.clone(),
                    candidate_model: fact.model.clone(),
                    candidate_model_revision: fact.model_revision.clone(),
                    decoding_fingerprint: fact.decoding_fingerprint.clone(),
                })
                .unwrap()
            })
            .collect();
        let confidence_policy =
            ConfidencePolicyV1::new(1, 1.0, 1, 1, 1.0, 0.0, 3_600.0, 1.0, 1.0, 0.95, 0.0, 0.7)
                .unwrap();
        let admitted_at = Instant::now();
        let prepared = prepare_recommendation_v1(
            RecommendationDecisionIdentityV1 {
                decision_id: uuid(4),
                process_instance_id,
                primary_call_uuid: uuid(5),
                as_of_unix_ms: AS_OF_UNIX_MS,
                created_at_unix_ms: AS_OF_UNIX_MS + 1,
                admitted_at,
            },
            preflight,
            query,
            confidence_policy,
            partition_base,
            partitions,
            admitted_at + Duration::from_secs(5),
        )
        .unwrap();
        let normalized =
            NormalizedVector::from_provider_f64(&[1.0, 0.0], VectorDimensions::new(2).unwrap())
                .unwrap();
        let vector = AuthoritativeVector::from_normalized(&vector_space_id, normalized).unwrap();
        Fixture {
            prepared,
            vector,
            learning_generation_id,
        }
    }

    fn passing_neighbor(sequence: u64, learning_generation_id: Uuid) -> ProjectedVectorNeighbor {
        ProjectedVectorNeighbor {
            vector_match: VectorMatch::new(VectorRecordId::new(uuid(100 + sequence)).unwrap(), 0.1)
                .unwrap(),
            vector_record: None,
            shadow_attempt_id: uuid(200 + sequence),
            shadow_result_id: uuid(300 + sequence),
            anchor_id: uuid(400 + sequence),
            root_uuid: uuid(500 + sequence),
            learning_generation_id,
            terminal_class: ShadowTerminalClass::Completed,
            evaluation: Some(ProjectedNeighborEvaluation {
                evaluation_id: uuid(600 + sequence),
                source: JudgeEvaluationSourceV1::Judge,
                binary_label: Some(JudgeBinaryLabelV1::Pass),
                judge_confidence: Some(ScoredValueV1::new(0.99)),
                promotion_eligible: true,
                created_at_unix_ms: AS_OF_UNIX_MS,
            }),
        }
    }

    type ScriptedResult = Result<LiveProjectedNeighborSearch, VectorStoreError>;

    struct ScriptedSearch {
        results: Mutex<VecDeque<ScriptedResult>>,
        calls: Mutex<Vec<String>>,
    }

    impl ScriptedSearch {
        fn new(results: impl IntoIterator<Item = ScriptedResult>) -> Self {
            Self {
                results: Mutex::new(results.into_iter().collect()),
                calls: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl RecommendationVectorSearch for ScriptedSearch {
        fn search_live_partition_until<'a>(
            &'a self,
            _mapping: &'a FrozenMappingKey,
            expected_partition: &'a RoutingPartitionArtifactV1,
            _query_vector: &'a NormalizedVector,
            _top_k: usize,
            _deadline: Instant,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<LiveProjectedNeighborSearch, VectorStoreError>>
                    + Send
                    + 'a,
            >,
        > {
            Box::pin(async move {
                self.calls
                    .lock()
                    .unwrap()
                    .push(expected_partition.partition.candidate_id.clone());
                self.results
                    .lock()
                    .unwrap()
                    .pop_front()
                    .expect("scripted search result")
            })
        }
    }

    struct ScriptedGateResolver {
        authorizations: VecDeque<bool>,
        identities: Vec<RecommendationCandidateSearchIdentityV1>,
    }

    impl RecommendationCandidateGateResolverV1 for ScriptedGateResolver {
        fn resolve_gate<'a>(
            &'a mut self,
            identity: RecommendationCandidateSearchIdentityV1,
        ) -> Pin<
            Box<
                dyn Future<Output = Result<RecommendationCandidateGateV1, RecommendationErrorV1>>
                    + Send
                    + 'a,
            >,
        > {
            let authorized = self.authorizations.pop_front().unwrap();
            self.identities.push(identity.clone());
            Box::pin(async move {
                Ok(RecommendationCandidateGateV1 {
                    candidate_id: identity.candidate_id,
                    lower_bound_threshold: 0.0,
                    externally_authorized: authorized,
                })
            })
        }
    }

    #[tokio::test]
    async fn first_winner_stops_search_and_records_later_candidate() {
        let fixture = fixture(2);
        let store = ScriptedSearch::new([
            Ok(LiveProjectedNeighborSearch::Found {
                partition_id: PartitionId::new(11).unwrap(),
                neighbors: vec![passing_neighbor(1, fixture.learning_generation_id)],
            }),
            Ok(LiveProjectedNeighborSearch::NoPartition),
        ]);

        let audit = run_with_search(
            &store,
            fixture.prepared,
            LiveEmbeddingResult::Ready(fixture.vector),
        )
        .await
        .unwrap();

        assert_eq!(store.calls(), vec!["candidate-0"]);
        assert_eq!(
            audit.parent.final_reason,
            DecisionFinalReasonV1::RecommendObserveOnly
        );
        assert_eq!(audit.parent.candidate_id.as_deref(), Some("candidate-0"));
        assert_eq!(audit.parent.served_model, "anchor");
        assert_eq!(audit.parent.recommended_model, "candidate-model-0");
        assert_eq!(
            audit.summaries[0].terminal_reason,
            DecisionCandidateReasonV1::Passed
        );
        assert_eq!(
            audit.summaries[1].terminal_reason,
            DecisionCandidateReasonV1::NotEvaluatedAfterWinner
        );
        assert_eq!(
            audit.summaries[0].candidate_alpha,
            audit.summaries[1].candidate_alpha
        );
    }

    #[tokio::test]
    async fn reusable_plan_continues_after_fresh_gate_passes_without_external_authority() {
        let fixture = fixture(2);
        let store = ScriptedSearch::new([
            Ok(LiveProjectedNeighborSearch::Found {
                partition_id: PartitionId::new(11).unwrap(),
                neighbors: vec![passing_neighbor(1, fixture.learning_generation_id)],
            }),
            Ok(LiveProjectedNeighborSearch::Found {
                partition_id: PartitionId::new(12).unwrap(),
                neighbors: vec![passing_neighbor(2, fixture.learning_generation_id)],
            }),
        ]);
        let gates = vec![
            RecommendationCandidateGateV1 {
                candidate_id: "candidate-0".into(),
                lower_bound_threshold: 0.0,
                externally_authorized: false,
            },
            RecommendationCandidateGateV1 {
                candidate_id: "candidate-1".into(),
                lower_bound_threshold: 0.0,
                externally_authorized: true,
            },
        ];

        let plan = run_with_search_gates(
            &store,
            fixture.prepared,
            LiveEmbeddingResult::Ready(fixture.vector),
            gates.clone(),
        )
        .await
        .unwrap();

        assert_eq!(store.calls(), vec!["candidate-0", "candidate-1"]);
        assert_eq!(plan.gates, gates);
        assert_eq!(plan.winner_candidate_id.as_deref(), Some("candidate-1"));
        assert_eq!(
            plan.summaries()[0].reason,
            CandidateConfidenceReasonV1::Passed
        );
        assert_eq!(
            plan.summaries()[1].reason,
            CandidateConfidenceReasonV1::Passed
        );
        let audit = plan
            .into_active_audit(
                ActiveDecisionParentBindingV2 {
                    cohort_generation_id: uuid(700),
                    active_experiment_id: Some(uuid(701)),
                    active_authorization_state_event_id: Some(uuid(702)),
                    root_key: hash('9'),
                },
                DecisionFinalReasonV1::ActiveAnchorControl,
            )
            .unwrap();
        assert_eq!(audit.parent.decision_shape_version, 2);
        assert_eq!(audit.parent.algorithm_version, 2);
        assert_eq!(audit.parent.candidate_id.as_deref(), Some("candidate-1"));
        assert_eq!(audit.parent.recommended_model, "candidate-model-1");
        assert_eq!(audit.parent.served_model, "anchor");
        assert_eq!(
            audit.summaries[0].terminal_reason,
            DecisionCandidateReasonV1::Passed
        );
        audit.validate_frozen().unwrap();
    }

    #[tokio::test]
    async fn dynamic_gate_resolution_uses_exact_neighbors_and_stops_after_first_authority() {
        let fixture = fixture(3);
        let first_neighbor = passing_neighbor(1, fixture.learning_generation_id);
        let second_neighbor = passing_neighbor(2, fixture.learning_generation_id);
        let first_id = first_neighbor.vector_match.record_id().value();
        let second_id = second_neighbor.vector_match.record_id().value();
        let store = ScriptedSearch::new([
            Ok(LiveProjectedNeighborSearch::Found {
                partition_id: PartitionId::new(11).unwrap(),
                neighbors: vec![first_neighbor],
            }),
            Ok(LiveProjectedNeighborSearch::Found {
                partition_id: PartitionId::new(12).unwrap(),
                neighbors: vec![second_neighbor],
            }),
        ]);
        let mut resolver = ScriptedGateResolver {
            authorizations: VecDeque::from([false, true]),
            identities: Vec::new(),
        };

        let plan = run_with_search_source(
            &store,
            fixture.prepared,
            LiveEmbeddingResult::Ready(fixture.vector),
            CandidateGateSourceV1::Dynamic(&mut resolver),
        )
        .await
        .unwrap();

        assert_eq!(store.calls(), vec!["candidate-0", "candidate-1"]);
        assert_eq!(plan.winner_candidate_id.as_deref(), Some("candidate-1"));
        assert_eq!(resolver.identities.len(), 2);
        assert_eq!(
            resolver.identities[0].sorted_neighbor_hash,
            sorted_neighbor_hash([first_id]).unwrap()
        );
        assert_eq!(
            resolver.identities[1].sorted_neighbor_hash,
            sorted_neighbor_hash([second_id]).unwrap()
        );
        assert_eq!(
            plan.summaries()[2].reason,
            CandidateConfidenceReasonV1::NotEvaluatedAfterWinner
        );
    }

    #[tokio::test]
    async fn no_partition_only_advances_to_the_next_exact_partition() {
        let fixture = fixture(2);
        let store = ScriptedSearch::new([
            Ok(LiveProjectedNeighborSearch::NoPartition),
            Ok(LiveProjectedNeighborSearch::Found {
                partition_id: PartitionId::new(12).unwrap(),
                neighbors: vec![passing_neighbor(2, fixture.learning_generation_id)],
            }),
        ]);

        let audit = run_with_search(
            &store,
            fixture.prepared,
            LiveEmbeddingResult::Ready(fixture.vector),
        )
        .await
        .unwrap();

        assert_eq!(store.calls(), vec!["candidate-0", "candidate-1"]);
        assert_eq!(
            audit.summaries[0].terminal_reason,
            DecisionCandidateReasonV1::NoPartition
        );
        assert_eq!(
            audit.summaries[1].terminal_reason,
            DecisionCandidateReasonV1::Passed
        );
        assert_eq!(audit.parent.candidate_id.as_deref(), Some("candidate-1"));
    }

    #[tokio::test]
    async fn embedding_unavailable_records_one_fallback_summary_per_candidate() {
        let fixture = fixture(3);
        let store = ScriptedSearch::new([]);

        let audit = run_with_search(&store, fixture.prepared, LiveEmbeddingResult::Unavailable)
            .await
            .unwrap();

        assert!(store.calls().is_empty());
        assert_eq!(
            audit.parent.final_reason,
            DecisionFinalReasonV1::EmbeddingUnavailable
        );
        assert!(audit.summaries.iter().all(|summary| {
            summary.terminal_reason == DecisionCandidateReasonV1::NotEvaluatedAfterFallback
        }));
    }

    #[tokio::test]
    async fn first_search_failure_debits_each_candidate_once_at_the_boundary() {
        let mut fixture = fixture(2);
        fixture.prepared.budget.limit = fixture
            .prepared
            .budget
            .charged
            .checked_add(2 * AUDIT_CANDIDATE_RESERVATION_BYTES)
            .unwrap();
        let store = ScriptedSearch::new([Err(VectorStoreError::Unavailable)]);

        let audit = run_with_search(
            &store,
            fixture.prepared,
            LiveEmbeddingResult::Ready(fixture.vector),
        )
        .await
        .unwrap();

        assert_eq!(store.calls(), vec!["candidate-0"]);
        assert_eq!(
            audit.parent.final_reason,
            DecisionFinalReasonV1::VectorUnhealthy
        );
        assert!(audit.summaries.iter().all(|summary| {
            summary.terminal_reason == DecisionCandidateReasonV1::NotEvaluatedAfterFallback
        }));
    }

    #[tokio::test]
    async fn leaked_embedding_arc_and_expired_deadline_fail_before_search() {
        let retained = fixture(1);
        let leaked = retained.prepared.query_for_embedding();
        let store = ScriptedSearch::new([]);
        assert!(matches!(
            run_with_search(&store, retained.prepared, LiveEmbeddingResult::Unavailable).await,
            Err(RecommendationErrorV1::InvalidQuery)
        ));
        assert!(store.calls().is_empty());
        drop(leaked);

        let mut expired = fixture(1);
        expired.prepared.deadline = Instant::now() - Duration::from_millis(1);
        let store = ScriptedSearch::new([]);
        assert!(matches!(
            run_with_search(&store, expired.prepared, LiveEmbeddingResult::Unavailable).await,
            Err(RecommendationErrorV1::DeadlineExceeded)
        ));
        assert!(store.calls().is_empty());
    }

    #[test]
    fn pessimistic_budget_accepts_exact_limits_and_rejects_one_byte_over() {
        let exact = RecommendationAuditBudgetV1::reserve_with_limit(
            1_024,
            63,
            65,
            DECISION_AUDIT_BYTES_MAX,
        )
        .unwrap();
        assert_eq!(63 * 65, VECTOR_TOP_K_MAX);
        assert!(
            RecommendationAuditBudgetV1::reserve_with_limit(1_024, 63, 65, exact.worst_case,)
                .is_ok()
        );
        assert_eq!(
            RecommendationAuditBudgetV1::reserve_with_limit(1_024, 63, 65, exact.worst_case - 1,)
                .err(),
            Some(RecommendationErrorV1::ResourceLimit)
        );
        assert_eq!(
            RecommendationAuditBudgetV1::reserve(0, 64, 64).err(),
            Some(RecommendationErrorV1::ResourceLimit)
        );
    }
}
