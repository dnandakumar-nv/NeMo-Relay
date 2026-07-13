// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compute-side Recommend integration between runtime admission and audit delivery.

#![allow(dead_code)] // The runtime branch lands after the compute and delivery halves are complete.

use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Instant;

use nemo_relay::api::llm::{LlmExecutionContextSnapshot, LlmRequest};
use nemo_relay::api::runtime::LlmReplayTransport;
use uuid::{Uuid, Variant};

use crate::adapter::FamilyAdapter;
use crate::confidence::ConfidencePolicyV1;
use crate::config::{OutcomeConfig, PoolConfig};
use crate::decision_audit::DecisionAuditV1;
use crate::eligibility::IneligibilityReason;
use crate::live_embedding::{LiveEmbeddingService, PreparedLiveQueryV1};
use crate::preflight::preflight;
use crate::recommendation::{
    PreparedRecommendationV1, RecommendationDecisionIdentityV1, RecommendationErrorV1,
    RecommendationPreflightV1, RecommendationQueryV1, prepare_recommendation_v1,
    run_recommendation_until,
};
#[cfg(test)]
use crate::routing_partition::RoutingPartitionBaseArtifactV1;
use crate::routing_partition::{
    RoutingPartitionBaseV1, RoutingPartitionInputV1, build_routing_partition_base_v1,
    build_routing_partition_from_input_v1,
};
use crate::sqlite_vector_store::SqliteVecStore;
use crate::trajectory::TrajectoryIdentity;

/// Exact process and policy authority frozen for one admitted Recommend call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RecommendationRuntimeIdentityV1 {
    project_uuid: Uuid,
    process_instance_id: Uuid,
    config_generation_id: String,
    policy_version_id: String,
    learning_generation_id: Uuid,
    admitted_at_unix_ms: i64,
}

impl RecommendationRuntimeIdentityV1 {
    /// Freeze the active identity for the already-selected exact pool.
    pub(crate) fn from_runtime(
        identity: &TrajectoryIdentity,
        config_generation_id: impl Into<String>,
        pool_id: &str,
        admitted_at_unix_ms: i64,
    ) -> Result<Self, RecommendationErrorV1> {
        let authority = Self {
            project_uuid: identity.project_uuid,
            process_instance_id: identity.process_instance_id,
            config_generation_id: config_generation_id.into(),
            policy_version_id: identity
                .policy_version_id(pool_id)
                .ok_or(RecommendationErrorV1::InvalidIdentity)?
                .to_string(),
            learning_generation_id: identity
                .learning_generation_id(pool_id)
                .ok_or(RecommendationErrorV1::InvalidIdentity)?,
            admitted_at_unix_ms,
        };
        authority.validate()?;
        Ok(authority)
    }

    fn validate(&self) -> Result<(), RecommendationErrorV1> {
        if ![
            self.project_uuid,
            self.process_instance_id,
            self.learning_generation_id,
        ]
        .into_iter()
        .all(is_uuid_v7)
            || !is_sha256(&self.config_generation_id)
            || !is_sha256(&self.policy_version_id)
            || self.admitted_at_unix_ms < 0
        {
            return Err(RecommendationErrorV1::InvalidIdentity);
        }
        Ok(())
    }
}

/// Run the complete bounded Recommend computation without invoking replay or Shadow scheduling.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn compute_recommendation_audit_until(
    context: &LlmExecutionContextSnapshot,
    request: &LlmRequest,
    replay: Option<Arc<dyn LlmReplayTransport>>,
    pool: &PoolConfig,
    identity: &RecommendationRuntimeIdentityV1,
    live_embedding: &LiveEmbeddingService,
    vector_store: &SqliteVecStore,
    admitted_at: Instant,
    deadline: Instant,
) -> Result<DecisionAuditV1, RecommendationErrorV1> {
    match panic_safe_recommendation_future(async move {
        let prepared = prepare_runtime_recommendation_with(
            context,
            request,
            replay,
            pool,
            identity,
            admitted_at,
            deadline,
            None,
            false,
            |request_projection, routing_projection| {
                live_embedding.prepare_query(&pool.id, request_projection, routing_projection)
            },
        )?
        .recommendation;
        let embedding_query = prepared.query_for_embedding();
        let embedding = live_embedding
            .embed_prepared_until(embedding_query, deadline)
            .await;
        run_recommendation_until(vector_store, prepared, embedding).await
    })
    .await
    {
        Ok(result) => result,
        Err(()) => Err(RecommendationErrorV1::RuntimeFailure),
    }
}

pub(crate) struct PreparedActiveRuntimeQueryV2 {
    pub(crate) recommendation: PreparedRecommendationV1,
    envelope: crate::adapter::RouterRequestEnvelope,
}

impl PreparedActiveRuntimeQueryV2 {
    pub(crate) fn into_parts(
        self,
    ) -> (
        PreparedRecommendationV1,
        crate::adapter::RouterRequestEnvelope,
    ) {
        (self.recommendation, self.envelope)
    }
}

/// Prepare Active confidence with the outcome policy's fresh anchor-shadow half-life.
#[allow(clippy::too_many_arguments)]
pub(crate) fn prepare_active_runtime_query_v2(
    context: &LlmExecutionContextSnapshot,
    request: &LlmRequest,
    replay: Option<Arc<dyn LlmReplayTransport>>,
    pool: &PoolConfig,
    identity: &RecommendationRuntimeIdentityV1,
    live_embedding: &LiveEmbeddingService,
    admitted_at: Instant,
    deadline: Instant,
) -> Result<PreparedActiveRuntimeQueryV2, RecommendationErrorV1> {
    pool.learning
        .as_ref()
        .and_then(|learning| learning.complete_active_policy())
        .ok_or(RecommendationErrorV1::InvalidPreflight)?;
    let outcome = serde_json::from_value::<OutcomeConfig>(serde_json::Value::Object(
        pool.outcome.clone().into_iter().collect(),
    ))
    .map_err(|_| RecommendationErrorV1::InvalidPreflight)?;
    let active_half_life = u32::try_from(outcome.anchor_shadow_half_life_seconds)
        .map(f64::from)
        .map_err(|_| RecommendationErrorV1::InvalidPreflight)?;
    let prepared = prepare_runtime_recommendation_with(
        context,
        request,
        replay,
        pool,
        identity,
        admitted_at,
        deadline,
        Some(active_half_life),
        true,
        |request_projection, routing_projection| {
            live_embedding.prepare_query(&pool.id, request_projection, routing_projection)
        },
    )?;
    Ok(PreparedActiveRuntimeQueryV2 {
        recommendation: prepared.recommendation,
        envelope: prepared
            .envelope
            .ok_or(RecommendationErrorV1::RuntimeFailure)?,
    })
}

pub(crate) fn rewrite_active_winner_v2(
    envelope: &crate::adapter::RouterRequestEnvelope,
    winner_model: &str,
) -> Result<LlmRequest, RecommendationErrorV1> {
    FamilyAdapter
        .with_model(envelope, winner_model)
        .map_err(|_| RecommendationErrorV1::InvalidPreflight)
}

pub(crate) struct PanicSafeRecommendationFuture<F> {
    future: Option<Pin<Box<F>>>,
}

impl<F> PanicSafeRecommendationFuture<F> {
    fn new(future: F) -> Self {
        Self {
            future: Some(Box::pin(future)),
        }
    }

    fn drop_future(&mut self) -> Result<(), ()> {
        let Some(future) = self.future.take() else {
            return Ok(());
        };
        catch_unwind(AssertUnwindSafe(|| drop(future))).map_err(|_| ())
    }
}

impl<F> Future for PanicSafeRecommendationFuture<F>
where
    F: Future,
{
    type Output = Result<F::Output, ()>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let polled = {
            let future = this
                .future
                .as_mut()
                .expect("recommendation future polled after completion");
            catch_unwind(AssertUnwindSafe(|| future.as_mut().poll(context)))
        };
        match polled {
            Ok(Poll::Ready(result)) if this.drop_future().is_ok() => Poll::Ready(Ok(result)),
            Ok(Poll::Ready(_)) | Err(_) => {
                let _ = this.drop_future();
                Poll::Ready(Err(()))
            }
            Ok(Poll::Pending) => Poll::Pending,
        }
    }
}

pub(crate) fn panic_safe_recommendation_future<F: Future>(
    future: F,
) -> PanicSafeRecommendationFuture<F> {
    PanicSafeRecommendationFuture::new(future)
}

impl<F> Drop for PanicSafeRecommendationFuture<F> {
    fn drop(&mut self) {
        let _ = self.drop_future();
    }
}

struct RuntimePreparedRecommendationV1 {
    recommendation: PreparedRecommendationV1,
    envelope: Option<crate::adapter::RouterRequestEnvelope>,
    #[cfg(test)]
    partition_base: RoutingPartitionBaseArtifactV1,
}

#[allow(clippy::too_many_arguments)]
fn prepare_runtime_recommendation_with<F>(
    context: &LlmExecutionContextSnapshot,
    request: &LlmRequest,
    replay: Option<Arc<dyn LlmReplayTransport>>,
    pool: &PoolConfig,
    identity: &RecommendationRuntimeIdentityV1,
    admitted_at: Instant,
    deadline: Instant,
    time_decay_half_life_seconds: Option<f64>,
    retain_envelope: bool,
    prepare_query: F,
) -> Result<RuntimePreparedRecommendationV1, RecommendationErrorV1>
where
    F: FnOnce(
        &crate::projection::RouterRequestProjectionV1,
        &crate::projection::RouterRoutingContextProjectionV1,
    ) -> Result<Arc<PreparedLiveQueryV1>, IneligibilityReason>,
{
    check_deadline(deadline)?;
    identity.validate()?;
    if admitted_at > deadline {
        return Err(RecommendationErrorV1::InvalidIdentity);
    }

    let outcome = preflight(context, request, replay.as_ref(), pool, &FamilyAdapter);
    drop(replay);
    let outcome = outcome.map_err(|_| RecommendationErrorV1::InvalidPreflight)?;
    check_deadline(deadline)?;
    let anchor_model = outcome
        .envelope
        .normalized_request
        .model
        .clone()
        .ok_or(RecommendationErrorV1::InvalidPreflight)?;
    let prepared_live = prepare_query(&outcome.request_projection, &outcome.routing_projection)
        .map_err(|_| RecommendationErrorV1::InvalidQuery)?;
    check_deadline(deadline)?;
    validate_mapping_authority(&prepared_live, pool, identity)?;

    let learning = pool
        .learning
        .as_ref()
        .and_then(|learning| learning.complete_policy())
        .ok_or(RecommendationErrorV1::InvalidPreflight)?;
    let confidence_policy = ConfidencePolicyV1::new(
        learning.top_k,
        learning.radius,
        learning.min_points,
        learning.min_independent_roots,
        learning.min_effective_samples,
        learning.min_coverage,
        time_decay_half_life_seconds.unwrap_or(learning.time_decay_half_life_seconds),
        learning.prior_success,
        learning.prior_failure,
        learning.familywise_credible_level,
        learning.promotion_lower_bound,
        pool.judge.judge_confidence_floor,
    )?;
    let partition_base = build_routing_partition_base_v1(&RoutingPartitionBaseV1 {
        tenant_policy_hash: outcome.routing_projection.tenant_policy_hash.clone(),
        agent_policy_hash: outcome.routing_projection.agent_policy_hash.clone(),
        policy_version_id: identity.policy_version_id.clone(),
        learning_generation_id: identity.learning_generation_id,
        api_family: outcome.replay_capability.api_family,
        transport_identity: outcome.replay_capability.transport_identity.clone(),
        anchor_model,
        anchor_revision: pool.anchor_revision.clone(),
        evaluator_version: pool
            .judge
            .evaluator_version()
            .map_err(|_| RecommendationErrorV1::InvalidPartition)?,
        vector_space_id: prepared_live.vector_space_id().as_str().to_string(),
    })
    .map_err(|_| RecommendationErrorV1::InvalidPartition)?;
    let partitions = outcome
        .eligible_candidate_facts
        .iter()
        .map(|candidate| {
            build_routing_partition_from_input_v1(&RoutingPartitionInputV1 {
                base: partition_base.base.clone(),
                candidate_id: candidate.candidate_id.clone(),
                candidate_model: candidate.model.clone(),
                candidate_model_revision: candidate.model_revision.clone(),
                decoding_fingerprint: candidate.decoding_fingerprint.clone(),
            })
            .map_err(|_| RecommendationErrorV1::InvalidPartition)
        })
        .collect::<Result<Vec<_>, _>>()?;
    check_deadline(deadline)?;

    let (safe_preflight, envelope) = RecommendationPreflightV1::from_active_preflight(outcome)?;
    let query = RecommendationQueryV1::from_prepared(prepared_live)?;
    #[cfg(test)]
    let test_partition_base = partition_base.clone();
    let recommendation = prepare_recommendation_v1(
        RecommendationDecisionIdentityV1 {
            decision_id: Uuid::now_v7(),
            process_instance_id: identity.process_instance_id,
            primary_call_uuid: context.call_uuid,
            as_of_unix_ms: identity.admitted_at_unix_ms,
            created_at_unix_ms: identity.admitted_at_unix_ms,
            admitted_at,
        },
        safe_preflight,
        query,
        confidence_policy,
        partition_base,
        partitions,
        deadline,
    )?;
    Ok(RuntimePreparedRecommendationV1 {
        recommendation,
        envelope: retain_envelope.then_some(envelope),
        #[cfg(test)]
        partition_base: test_partition_base,
    })
}

fn validate_mapping_authority(
    prepared: &PreparedLiveQueryV1,
    pool: &PoolConfig,
    identity: &RecommendationRuntimeIdentityV1,
) -> Result<(), RecommendationErrorV1> {
    let mapping = prepared.mapping();
    if mapping.project_uuid != identity.project_uuid
        || mapping.config_generation_id != identity.config_generation_id
        || mapping.policy_version_id != identity.policy_version_id
        || mapping.pool_id != pool.id
    {
        return Err(RecommendationErrorV1::InvalidIdentity);
    }
    Ok(())
}

fn check_deadline(deadline: Instant) -> Result<(), RecommendationErrorV1> {
    (Instant::now() < deadline)
        .then_some(())
        .ok_or(RecommendationErrorV1::DeadlineExceeded)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_uuid_v7(value: Uuid) -> bool {
    value.get_version_num() == 7 && value.get_variant() == Variant::RFC4122
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use nemo_relay::api::llm::{
        LlmApiFamily, LlmAttributes, LlmCallRole, LlmTrajectoryScopeSnapshot,
    };
    use nemo_relay::api::runtime::{
        LLM_REPLAY_CONTRACT_VERSION, LlmReplayCall, LlmReplayCapability,
    };
    use nemo_relay::api::scope::ScopeType;
    use serde_json::json;

    use super::*;
    use crate::canonical_query::build_canonical_routing_query;
    use crate::config::{
        CandidateCapabilities, CandidateConfig, CanonicalizerConfig, ConcurrencyConfig,
        JUDGE_PROMPT_VERSION_V1, JUDGE_RUBRIC_VERSION_V1, JudgeConfig, LearningConfig,
        LookaheadConfig, PoolSelectorConfig,
    };
    use crate::ledger::repository::vector_registry::FrozenMappingKey;
    use crate::vector::VectorSpaceId;

    struct CountingReplay {
        capability: LlmReplayCapability,
        capability_calls: AtomicUsize,
        start_calls: AtomicUsize,
    }

    struct DropPanicFuture {
        ready: bool,
    }

    impl Future for DropPanicFuture {
        type Output = Result<DecisionAuditV1, RecommendationErrorV1>;

        fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
            if self.ready {
                Poll::Ready(Err(RecommendationErrorV1::DeadlineExceeded))
            } else {
                Poll::Pending
            }
        }
    }

    impl Drop for DropPanicFuture {
        fn drop(&mut self) {
            panic!("test future drop panic")
        }
    }

    impl LlmReplayTransport for CountingReplay {
        fn capability(&self) -> &LlmReplayCapability {
            self.capability_calls.fetch_add(1, Ordering::Relaxed);
            &self.capability
        }

        fn start(&self, _request: LlmRequest) -> nemo_relay::error::Result<LlmReplayCall> {
            self.start_calls.fetch_add(1, Ordering::Relaxed);
            panic!("Recommend compute must not invoke replay")
        }
    }

    fn hash(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn context() -> LlmExecutionContextSnapshot {
        let root_uuid = Uuid::now_v7();
        LlmExecutionContextSnapshot {
            call_uuid: Uuid::now_v7(),
            root_uuid,
            parent_uuid: root_uuid,
            trajectory_owner_uuid: root_uuid,
            trajectory_owner_path: vec![LlmTrajectoryScopeSnapshot {
                uuid: root_uuid,
                name: "agent".to_string(),
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

    fn request() -> LlmRequest {
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({
                "model": "actual-anchor",
                "messages": [{"role": "user", "content": "hello"}],
            }),
        }
    }

    fn pool() -> PoolConfig {
        PoolConfig {
            id: "pool".to_string(),
            api_family: LlmApiFamily::OpenAIChatCompletions,
            anchor_models: vec!["other-anchor".to_string(), "actual-anchor".to_string()],
            anchor_revision: "anchor-r1".to_string(),
            sampling_probability: 1.0,
            max_candidates_per_sample: 1,
            selector: PoolSelectorConfig::default(),
            lookahead: LookaheadConfig::default(),
            concurrency: ConcurrencyConfig {
                shadow: 1,
                judge: 1,
                max_pending: 1,
                unknown_fields: BTreeMap::new(),
            },
            candidates: vec![CandidateConfig {
                id: "candidate-a".to_string(),
                model: "candidate-model".to_string(),
                model_revision: "candidate-r1".to_string(),
                cost_rank: 1,
                max_context_tokens: None,
                capabilities: CandidateCapabilities {
                    tools: false,
                    multimodal_input: false,
                    structured_output: false,
                    reasoning_controls: false,
                    unknown_fields: BTreeMap::new(),
                },
                unknown_fields: BTreeMap::new(),
            }],
            canonicalizer: CanonicalizerConfig::default(),
            judge: JudgeConfig {
                version: 1,
                model: "judge".to_string(),
                model_revision: "judge-r1".to_string(),
                prompt_version: JUDGE_PROMPT_VERSION_V1.to_string(),
                rubric_version: JUDGE_RUBRIC_VERSION_V1.to_string(),
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
            },
            learning: Some(LearningConfig {
                version: 1,
                embedder: "embedder".to_string(),
                top_k: Some(1),
                radius: Some(1.0),
                min_points: Some(1),
                min_independent_roots: Some(1),
                min_effective_samples: Some(1.0),
                min_coverage: Some(1.0),
                time_decay_half_life_seconds: Some(3_600.0),
                prior_success: Some(1.0),
                prior_failure: Some(1.0),
                familywise_credible_level: Some(0.95),
                promotion_lower_bound: Some(0.7),
                retention_lower_bound: None,
                holdout_probability: None,
                active_canary_fraction: None,
            }),
            outcome: BTreeMap::new(),
            unknown_fields: BTreeMap::new(),
        }
    }

    fn runtime_identity() -> RecommendationRuntimeIdentityV1 {
        RecommendationRuntimeIdentityV1 {
            project_uuid: Uuid::now_v7(),
            process_instance_id: Uuid::now_v7(),
            config_generation_id: hash('7'),
            policy_version_id: hash('c'),
            learning_generation_id: Uuid::now_v7(),
            admitted_at_unix_ms: 1_000,
        }
    }

    fn replay() -> Arc<CountingReplay> {
        Arc::new(CountingReplay {
            capability: LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: LlmApiFamily::OpenAIChatCompletions,
                transport_identity: "transport-v1".to_string(),
            },
            capability_calls: AtomicUsize::new(0),
            start_calls: AtomicUsize::new(0),
        })
    }

    #[test]
    fn preparation_uses_exact_authority_without_replay_or_request_mutation() {
        let context = context();
        let request = request();
        let original_request = request.clone();
        let pool = pool();
        let identity = runtime_identity();
        let replay = replay();
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let vector_space_id = VectorSpaceId::new(hash('6')).unwrap();
        let mapping = FrozenMappingKey::new(
            identity.project_uuid,
            identity.config_generation_id.clone(),
            pool.id.clone(),
            identity.policy_version_id.clone(),
        )
        .unwrap();
        let admitted_at = Instant::now();
        let prepared = prepare_runtime_recommendation_with(
            &context,
            &request,
            Some(replay_transport),
            &pool,
            &identity,
            admitted_at,
            admitted_at + Duration::from_secs(1),
            None,
            false,
            |request_projection, routing_projection| {
                assert_eq!(Arc::strong_count(&replay), 1);
                let artifact = build_canonical_routing_query(
                    request_projection,
                    routing_projection,
                    &pool.canonicalizer,
                )?;
                Ok(PreparedLiveQueryV1::from_test_parts(
                    mapping,
                    vector_space_id,
                    artifact,
                    Duration::from_secs(1),
                ))
            },
        )
        .unwrap();
        assert_eq!(prepared.partition_base.base.anchor_model, "actual-anchor");
        assert_eq!(
            prepared.partition_base.base.policy_version_id,
            identity.policy_version_id
        );
        assert_eq!(
            prepared.partition_base.base.learning_generation_id,
            identity.learning_generation_id
        );
        assert_eq!(prepared.partition_base.base.vector_space_id, hash('6'));
        let query = prepared.recommendation.query_for_embedding();
        assert_eq!(query.mapping().project_uuid, identity.project_uuid);
        assert_eq!(
            query.mapping().config_generation_id,
            identity.config_generation_id
        );
        assert_eq!(
            query.mapping().policy_version_id,
            identity.policy_version_id
        );
        assert_eq!(query.mapping().pool_id, pool.id);
        assert_eq!(request, original_request);
        assert_eq!(replay.capability_calls.load(Ordering::Relaxed), 1);
        assert_eq!(replay.start_calls.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn deadline_and_mapping_failures_stop_before_embedding_or_replay() {
        let context = context();
        let request = request();
        let pool = pool();
        let identity = runtime_identity();
        let replay = replay();
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let now = Instant::now();
        let expired = prepare_runtime_recommendation_with(
            &context,
            &request,
            Some(replay_transport.clone()),
            &pool,
            &identity,
            now,
            now,
            None,
            false,
            |_, _| unreachable!("expired work must not prepare a query"),
        );
        assert!(matches!(
            expired,
            Err(RecommendationErrorV1::DeadlineExceeded)
        ));
        assert_eq!(replay.capability_calls.load(Ordering::Relaxed), 0);
        assert_eq!(replay.start_calls.load(Ordering::Relaxed), 0);

        let wrong_mapping = FrozenMappingKey::new(
            Uuid::now_v7(),
            identity.config_generation_id.clone(),
            pool.id.clone(),
            identity.policy_version_id.clone(),
        )
        .unwrap();
        let vector_space_id = VectorSpaceId::new(hash('6')).unwrap();
        let admitted_at = Instant::now();
        let rejected = prepare_runtime_recommendation_with(
            &context,
            &request,
            Some(replay_transport),
            &pool,
            &identity,
            admitted_at,
            admitted_at + Duration::from_secs(1),
            None,
            false,
            |request_projection, routing_projection| {
                let artifact = build_canonical_routing_query(
                    request_projection,
                    routing_projection,
                    &pool.canonicalizer,
                )?;
                Ok(PreparedLiveQueryV1::from_test_parts(
                    wrong_mapping,
                    vector_space_id,
                    artifact,
                    Duration::from_secs(1),
                ))
            },
        );
        assert!(matches!(
            rejected,
            Err(RecommendationErrorV1::InvalidIdentity)
        ));
        assert_eq!(replay.start_calls.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn panic_safe_future_contains_poll_and_ready_drop_panics() {
        let poll_panic = panic_safe_recommendation_future(async {
            panic!("test future poll panic");
            #[allow(unreachable_code)]
            Err::<(), _>(RecommendationErrorV1::RuntimeFailure)
        })
        .await;
        assert!(poll_panic.is_err());

        let drop_panic = panic_safe_recommendation_future(DropPanicFuture { ready: true }).await;
        assert!(drop_panic.is_err());

        let pending_drop = catch_unwind(AssertUnwindSafe(|| {
            drop(panic_safe_recommendation_future(DropPanicFuture {
                ready: false,
            }));
        }));
        assert!(pending_drop.is_ok());
    }
}
