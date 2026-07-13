// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Activated-ledger coverage for the public Recommend execution branch.

use std::collections::BTreeMap;
#[cfg(unix)]
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nemo_relay::api::llm::{
    LlmApiFamily, LlmAttributes, LlmCallRole, LlmExecutionContextSnapshot, LlmRequest,
    LlmTrajectoryScopeSnapshot,
};
use nemo_relay::api::runtime::{
    LLM_REPLAY_CONTRACT_VERSION, LlmExecutionNextFn, LlmReplayCall, LlmReplayCapability,
    LlmReplayTransport,
};
use nemo_relay::api::scope::ScopeType;
use nemo_relay::error::{FlowError, Result as FlowResult};
use serde_json::json;
use tempfile::{TempDir, tempdir};
use tokio::sync::Barrier;
use uuid::Uuid;

use crate::config::{
    CandidateCapabilities, CandidateConfig, ConcurrencyConfig, EmbedderConfig,
    JUDGE_PROMPT_VERSION_V1, JUDGE_RUBRIC_VERSION_V1, JudgeConfig, LearningConfig, PoolConfig,
    RouterConfig, RouterMode,
};
use crate::ledger::repository::LedgerRepository;
use crate::runtime::RouterRuntime;

struct InspectOnlyReplay {
    capability: LlmReplayCapability,
    capability_calls: AtomicUsize,
    starts: AtomicUsize,
}

impl InspectOnlyReplay {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            capability: LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: LlmApiFamily::OpenAIChatCompletions,
                transport_identity: "recommend-integration-test".into(),
            },
            capability_calls: AtomicUsize::new(0),
            starts: AtomicUsize::new(0),
        })
    }
}

impl LlmReplayTransport for InspectOnlyReplay {
    fn capability(&self) -> &LlmReplayCapability {
        self.capability_calls.fetch_add(1, Ordering::SeqCst);
        &self.capability
    }

    fn start(&self, _request: LlmRequest) -> FlowResult<LlmReplayCall> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        Err(FlowError::Internal(
            "Recommend mode must not start replay work".into(),
        ))
    }
}

fn recommend_config() -> (TempDir, RouterConfig) {
    let temporary = tempdir().unwrap();
    #[cfg(unix)]
    fs::set_permissions(
        temporary.path(),
        <fs::Permissions as std::os::unix::fs::PermissionsExt>::from_mode(0o700),
    )
    .unwrap();

    let config = RouterConfig {
        mode: RouterMode::Recommend,
        project_id: Some("recommend-runtime-integration".into()),
        database_path: temporary
            .path()
            .join("ledger/router.db")
            .to_string_lossy()
            .into_owned(),
        embedders: vec![EmbedderConfig {
            id: "embedding-main".into(),
            base_url: "http://127.0.0.1:9/v1".into(),
            model: "embedding-model".into(),
            provider_revision: "embedding-r1".into(),
            dimensions: 2,
            api_key_env: None,
            timeout_ms: 1_000,
            max_in_flight: 1,
            batch_size: 1,
            unknown_fields: BTreeMap::new(),
        }],
        pools: vec![PoolConfig {
            id: "pool".into(),
            api_family: LlmApiFamily::OpenAIChatCompletions,
            anchor_models: vec!["anchor-model".into()],
            anchor_revision: "anchor-r1".into(),
            sampling_probability: 1.0,
            max_candidates_per_sample: 1,
            selector: Default::default(),
            lookahead: Default::default(),
            concurrency: ConcurrencyConfig {
                shadow: 1,
                judge: 1,
                max_pending: 4,
                unknown_fields: BTreeMap::new(),
            },
            candidates: vec![CandidateConfig {
                id: "candidate".into(),
                model: "candidate-model".into(),
                model_revision: "candidate-r1".into(),
                cost_rank: 0,
                max_context_tokens: None,
                capabilities: CandidateCapabilities {
                    tools: true,
                    multimodal_input: true,
                    structured_output: true,
                    reasoning_controls: true,
                    unknown_fields: BTreeMap::new(),
                },
                unknown_fields: BTreeMap::new(),
            }],
            canonicalizer: Default::default(),
            judge: JudgeConfig {
                version: 1,
                model: "judge-model".into(),
                model_revision: "judge-r1".into(),
                prompt_version: JUDGE_PROMPT_VERSION_V1.into(),
                rubric_version: JUDGE_RUBRIC_VERSION_V1.into(),
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
                embedder: "embedding-main".into(),
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
                promotion_lower_bound: Some(0.0),
                retention_lower_bound: None,
                holdout_probability: None,
                active_canary_fraction: None,
            }),
            outcome: BTreeMap::new(),
            unknown_fields: BTreeMap::new(),
        }],
        ..RouterConfig::default()
    };
    assert!(config.validate().is_empty(), "{:?}", config.validate());
    (temporary, config)
}

fn primary_context() -> Arc<LlmExecutionContextSnapshot> {
    let root_uuid = Uuid::now_v7();
    let owner_uuid = Uuid::now_v7();
    Arc::new(LlmExecutionContextSnapshot {
        call_uuid: Uuid::now_v7(),
        root_uuid,
        parent_uuid: owner_uuid,
        trajectory_owner_uuid: owner_uuid,
        trajectory_owner_path: vec![LlmTrajectoryScopeSnapshot {
            uuid: owner_uuid,
            name: "agent".into(),
            scope_type: ScopeType::Agent,
        }],
        api_family: LlmApiFamily::OpenAIChatCompletions,
        call_role: LlmCallRole::Primary,
        attributes: LlmAttributes::empty(),
        tenant_id: None,
        agent_id: None,
        sanitized_metadata: BTreeMap::new(),
    })
}

fn anchor_request() -> LlmRequest {
    LlmRequest {
        headers: serde_json::Map::new(),
        content: json!({
            "model": "anchor-model",
            "messages": [{"role": "user", "content": "keep the anchor request exact"}]
        }),
    }
}

#[tokio::test]
async fn activated_recommend_preserves_anchor_and_persists_one_fallback_decision() {
    let (_temporary, config) = recommend_config();
    let database_path = config.database_path.clone();
    let activated = LedgerRepository::activate(&config).unwrap();
    let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
    runtime
        .live_embedding_service()
        .expect("Recommend runtime must own live embedding")
        .close();

    let replay = InspectOnlyReplay::new();
    let request = anchor_request();
    let expected_request = request.clone();
    let observed_requests = Arc::new(Mutex::new(Vec::new()));
    let next_calls = Arc::new(AtomicUsize::new(0));
    let expected_response = json!({"id": "anchor-response", "model": "anchor-model"});
    let next: LlmExecutionNextFn = {
        let observed_requests = observed_requests.clone();
        let next_calls = next_calls.clone();
        let expected_response = expected_response.clone();
        Arc::new(move |request| {
            next_calls.fetch_add(1, Ordering::SeqCst);
            observed_requests
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(request);
            let response = expected_response.clone();
            Box::pin(async move { Ok(response) })
        })
    };

    let response = runtime
        .execute_for_test(primary_context(), request, Some(replay.clone()), next)
        .await
        .unwrap();

    assert_eq!(response, expected_response);
    assert_eq!(next_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        *observed_requests
            .lock()
            .unwrap_or_else(|error| error.into_inner()),
        vec![expected_request]
    );
    assert_eq!(replay.capability_calls.load(Ordering::SeqCst), 1);
    assert_eq!(replay.starts.load(Ordering::SeqCst), 0);

    runtime
        .drain_for_test(Instant::now() + Duration::from_secs(5))
        .await
        .unwrap();
    let connection = rusqlite::Connection::open(database_path).unwrap();
    assert_eq!(
        connection
            .query_row(
                "SELECT count(*) FROM decisions WHERE final_reason = 'embedding_unavailable'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
    for (table, expected) in [
        ("decisions", 1),
        ("decision_candidate_summaries", 1),
        ("decision_neighbors", 0),
        ("sample_batches", 0),
        ("shadow_attempts", 0),
        ("anchors", 0),
    ] {
        let count = connection
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap();
        assert_eq!(count, expected, "{table}");
    }
}

#[tokio::test]
async fn activated_recommend_returns_the_exact_anchor_error_once() {
    let (_temporary, config) = recommend_config();
    let activated = LedgerRepository::activate(&config).unwrap();
    let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
    runtime
        .live_embedding_service()
        .expect("Recommend runtime must own live embedding")
        .close();

    let replay = InspectOnlyReplay::new();
    let request = anchor_request();
    let expected_request = request.clone();
    let observed_requests = Arc::new(Mutex::new(Vec::new()));
    let next_calls = Arc::new(AtomicUsize::new(0));
    let next: LlmExecutionNextFn = {
        let observed_requests = observed_requests.clone();
        let next_calls = next_calls.clone();
        Arc::new(move |request| {
            next_calls.fetch_add(1, Ordering::SeqCst);
            observed_requests
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(request);
            Box::pin(async {
                Err(FlowError::Internal(
                    "recommend anchor error sentinel".into(),
                ))
            })
        })
    };

    let error = runtime
        .execute_for_test(primary_context(), request, Some(replay.clone()), next)
        .await
        .unwrap_err();
    match error {
        FlowError::Internal(message) => assert_eq!(message, "recommend anchor error sentinel"),
        other => panic!("unexpected anchor error: {other}"),
    }
    assert_eq!(next_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        *observed_requests
            .lock()
            .unwrap_or_else(|error| error.into_inner()),
        vec![expected_request]
    );
    assert_eq!(replay.capability_calls.load(Ordering::SeqCst), 1);
    assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
    runtime
        .drain_for_test(Instant::now() + Duration::from_secs(5))
        .await
        .unwrap();
}

#[tokio::test]
async fn canceled_recommend_preprocessing_never_invokes_a_late_continuation() {
    let (_temporary, config) = recommend_config();
    let database_path = config.database_path.clone();
    let activated = LedgerRepository::activate(&config).unwrap();
    let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
    let live_embedding = runtime
        .live_embedding_service()
        .expect("Recommend runtime must own live embedding");
    let admitted = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    live_embedding.pause_next_call_after_admission(admitted.clone(), release.clone());

    let replay = InspectOnlyReplay::new();
    let next_calls = Arc::new(AtomicUsize::new(0));
    let next: LlmExecutionNextFn = {
        let next_calls = next_calls.clone();
        Arc::new(move |_| {
            next_calls.fetch_add(1, Ordering::SeqCst);
            Box::pin(async { Ok(json!({"unexpected": "late continuation"})) })
        })
    };
    let execute_runtime = runtime.clone();
    let execution_replay = replay.clone();
    let execution = tokio::spawn(async move {
        execute_runtime
            .execute_for_test(
                primary_context(),
                anchor_request(),
                Some(execution_replay),
                next,
            )
            .await
    });

    tokio::time::timeout(Duration::from_secs(2), admitted.wait())
        .await
        .expect("Recommend preprocessing did not reach live-embedding admission");
    execution.abort();
    assert!(execution.await.unwrap_err().is_cancelled());
    tokio::time::timeout(Duration::from_secs(2), release.wait())
        .await
        .expect("canceled preprocessing did not release detached live work");

    runtime
        .drain_for_test(Instant::now() + Duration::from_secs(5))
        .await
        .unwrap();
    assert_eq!(next_calls.load(Ordering::SeqCst), 0);
    assert_eq!(replay.capability_calls.load(Ordering::SeqCst), 1);
    assert_eq!(replay.starts.load(Ordering::SeqCst), 0);
    let connection = rusqlite::Connection::open(database_path).unwrap();
    assert_eq!(
        connection
            .query_row("SELECT count(*) FROM decisions", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        0
    );
}
