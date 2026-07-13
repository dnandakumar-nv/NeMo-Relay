// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use chrono::{Duration as ChronoDuration, TimeZone, Utc};
use nemo_relay::api::llm::{LlmApiFamily, LlmRequest};
use nemo_relay::api::runtime::{
    LLM_REPLAY_CONTRACT_VERSION, LlmReplayCall, LlmReplayCapability, LlmReplayTransport,
    NemoRelayContextState, global_context,
};
use nemo_relay::error::{FlowError, Result as FlowResult};
use nemo_relay::json::Json;
use rusqlite::Connection;
use serde_json::{Map, json};
use tempfile::{TempDir, tempdir};
use tokio::sync::{Notify, mpsc};
use uuid::Uuid;

use super::*;
use crate::adapter::FamilyAdapter;
use crate::config::CanonicalizerConfig;
use crate::ledger::model::LedgerRuntimeIdentity;
use crate::ledger::repository::LedgerRepository;
use crate::ledger::writer::LedgerWriterOwner;
use crate::preflight::EligibleCandidate;
use crate::projection::{
    ROUTING_CONTEXT_SCHEMA_V1, RouterRoutingContextProjectionV1, project_request,
};
use crate::response_validator::compile_candidate_response_contracts;
use crate::scheduler_admission::{
    SchedulerAdmission, SchedulerAdmissionError, SchedulerAdmissionPools,
};
use crate::sink::{DeliveryAck, SinkAck, TrajectoryDelivery, TrajectorySink};
use crate::sqlite_sink::{ScheduledTrajectoryBatch, SqliteTrajectorySink};
use crate::trajectory::{
    PENDING_TRAJECTORY_SCHEMA_V1, PendingTrajectoryWindow, PersistedCandidateFactV1,
    ReplayCapabilityFactsV1, TrajectoryTrigger, TrajectoryWindowSeed, project_anchor_response,
};

const POOL: &str = "pool-a";
const WRITER_CAPACITY: usize = 64;

enum ReplayAction {
    Response(Json),
    Error,
    Delayed {
        response: Json,
        control: Arc<DelayedReplayControl>,
    },
    Pending {
        control: Arc<PendingReplayControl>,
    },
}

#[derive(Default)]
struct DelayedReplayControl {
    entered: Notify,
    release: Notify,
}

#[derive(Default)]
struct PendingReplayControl {
    entered: Notify,
    cancellations: Arc<AtomicUsize>,
}

struct ScriptedReplay {
    capability: LlmReplayCapability,
    scripts: Mutex<BTreeMap<String, VecDeque<ReplayAction>>>,
    starts: Mutex<Vec<String>>,
    shadow_active: Arc<AtomicUsize>,
    shadow_max_active: Arc<AtomicUsize>,
    judge_active: Arc<AtomicUsize>,
    judge_max_active: Arc<AtomicUsize>,
}

struct ReplayActiveGuard {
    active: Arc<AtomicUsize>,
}

impl Drop for ReplayActiveGuard {
    fn drop(&mut self) {
        self.active.fetch_sub(1, Ordering::AcqRel);
    }
}

impl ScriptedReplay {
    fn new(scripts: impl IntoIterator<Item = (String, Vec<ReplayAction>)>) -> Arc<Self> {
        Arc::new(Self {
            capability: LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: LlmApiFamily::OpenAIChatCompletions,
                transport_identity: "scheduler-ledger-test-transport".to_string(),
            },
            scripts: Mutex::new(
                scripts
                    .into_iter()
                    .map(|(model, actions)| (model, actions.into()))
                    .collect(),
            ),
            starts: Mutex::new(Vec::new()),
            shadow_active: Arc::new(AtomicUsize::new(0)),
            shadow_max_active: Arc::new(AtomicUsize::new(0)),
            judge_active: Arc::new(AtomicUsize::new(0)),
            judge_max_active: Arc::new(AtomicUsize::new(0)),
        })
    }

    fn started_models(&self) -> Vec<String> {
        lock_unpoisoned(&self.starts).clone()
    }

    fn remaining_actions(&self) -> usize {
        lock_unpoisoned(&self.scripts)
            .values()
            .map(VecDeque::len)
            .sum()
    }

    fn max_active_judges(&self) -> usize {
        self.judge_max_active.load(Ordering::Acquire)
    }

    fn active_shadows(&self) -> usize {
        self.shadow_active.load(Ordering::Acquire)
    }

    fn active_judges(&self) -> usize {
        self.judge_active.load(Ordering::Acquire)
    }

    fn max_active_shadows(&self) -> usize {
        self.shadow_max_active.load(Ordering::Acquire)
    }
}

impl LlmReplayTransport for ScriptedReplay {
    fn capability(&self) -> &LlmReplayCapability {
        &self.capability
    }

    fn start(&self, request: LlmRequest) -> FlowResult<LlmReplayCall> {
        let model = request
            .content
            .get("model")
            .and_then(Json::as_str)
            .ok_or_else(|| FlowError::Internal("scripted replay request omitted model".into()))?
            .to_string();
        let action = lock_unpoisoned(&self.scripts)
            .get_mut(&model)
            .and_then(VecDeque::pop_front)
            .ok_or_else(|| {
                FlowError::Internal(format!("scripted replay has no action for {model}"))
            })?;
        lock_unpoisoned(&self.starts).push(model.clone());

        let (active, max_active) = if model == "judge-model" {
            (&self.judge_active, &self.judge_max_active)
        } else {
            (&self.shadow_active, &self.shadow_max_active)
        };
        let current = active.fetch_add(1, Ordering::AcqRel) + 1;
        max_active.fetch_max(current, Ordering::AcqRel);
        let active_guard = ReplayActiveGuard {
            active: active.clone(),
        };
        let cancellation_counter = match &action {
            ReplayAction::Pending { control } => Some(control.cancellations.clone()),
            ReplayAction::Response(_) | ReplayAction::Error | ReplayAction::Delayed { .. } => None,
        };
        Ok(LlmReplayCall::new(
            async move {
                let _active_guard = active_guard;
                match action {
                    ReplayAction::Response(response) => Ok(response),
                    ReplayAction::Error => Err(FlowError::Internal(
                        "scripted provider operation failed".to_string(),
                    )),
                    ReplayAction::Delayed { response, control } => {
                        control.entered.notify_one();
                        control.release.notified().await;
                        Ok(response)
                    }
                    ReplayAction::Pending { control } => {
                        control.entered.notify_one();
                        std::future::pending().await
                    }
                }
            },
            move || {
                if let Some(counter) = cancellation_counter {
                    counter.fetch_add(1, Ordering::AcqRel);
                }
            },
        ))
    }
}

struct Harness {
    _temporary: TempDir,
    database_path: PathBuf,
    config: RouterConfig,
    identity: LedgerRuntimeIdentity,
    owner: LedgerWriterOwner,
    client: LedgerWriterClient,
    sink: Option<SqliteTrajectorySink>,
    scheduler_rx: Option<mpsc::Receiver<ScheduledTrajectoryBatch>>,
}

impl Harness {
    fn new(candidate_count: usize) -> Self {
        Self::with_limits(candidate_count, 1, 1, 1)
    }

    fn with_limits(
        candidate_count: usize,
        max_pending: usize,
        shadow_concurrency: usize,
        judge_concurrency: usize,
    ) -> Self {
        Self::with_config(|database_path| {
            config(
                database_path,
                candidate_count,
                max_pending,
                shadow_concurrency,
                judge_concurrency,
            )
        })
    }

    fn with_config(build_config: impl FnOnce(&Path) -> RouterConfig) -> Self {
        let temporary = tempdir().expect("scheduler test tempdir should be created");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700))
                .expect("scheduler test tempdir permissions should be restricted");
        }
        let database_path = temporary.path().join("ledger/router.db");
        let config = build_config(&database_path);
        let activated =
            LedgerRepository::activate(&config).expect("scheduler test repository should activate");
        let identity = activated.identity.clone();
        let (owner, client) = LedgerWriterOwner::start(activated.repository, WRITER_CAPACITY)
            .expect("scheduler test writer should start");
        let (sink, scheduler_rx) = SqliteTrajectorySink::new(&config, client.clone())
            .expect("scheduler test sink should initialize");
        Self {
            _temporary: temporary,
            database_path,
            config,
            identity,
            owner,
            client,
            sink: Some(sink),
            scheduler_rx: Some(scheduler_rx),
        }
    }

    async fn deliver(&self, replay: Arc<dyn LlmReplayTransport>) -> Uuid {
        let (pending, seed) = window_fixture(self, replay);
        let sink = self.sink.as_ref().expect("test sink must remain open");
        let pending_hash = pending
            .payload_hash()
            .expect("pending payload should be canonical");
        assert_eq!(
            sink.record_pending(pending.clone()).await,
            SinkAck::Applied {
                anchor_id: pending.anchor_id,
                payload_hash: pending_hash,
            }
        );
        let closed = Arc::new(seed.into_closed(
            Vec::new(),
            1,
            TrajectoryTrigger::ProgressReached,
            exact_now(),
        ));
        assert!(matches!(
            sink.record_terminal(Arc::new(closed.terminal())).await,
            SinkAck::Applied { anchor_id, .. } if anchor_id == pending.anchor_id
        ));
        assert_eq!(
            sink.deliver(closed).await,
            DeliveryAck::Delivered {
                anchor_id: pending.anchor_id,
            }
        );
        pending.anchor_id
    }

    fn scheduler(&mut self) -> (ShadowScheduler, SchedulerAdmissionPools) {
        let start_gate = BackgroundStartGate::new();
        let active_replays = ActiveReplayRegistry::new(self.config.max_evidence_records as usize)
            .expect("test replay registry capacity should be valid");
        let cancellation = EvaluatorCancellation::default();
        let scheduler = self.scheduler_with_controls(start_gate, active_replays, cancellation);
        self.close_sink();
        scheduler
    }

    fn scheduler_with_controls(
        &mut self,
        start_gate: BackgroundStartGate,
        active_replays: ActiveReplayRegistry,
        cancellation: EvaluatorCancellation,
    ) -> (ShadowScheduler, SchedulerAdmissionPools) {
        let admissions = self
            .sink
            .as_ref()
            .expect("test sink must remain open")
            .admission_pools();
        let scheduler = ShadowScheduler::new(
            &self.config,
            self.client.clone(),
            self.scheduler_rx
                .take()
                .expect("scheduler receiver should be owned once"),
            start_gate,
            active_replays,
            cancellation,
            SchedulerDeadlineAuthority::new(),
        )
        .expect("scheduler should accept the test configuration");
        (scheduler, admissions)
    }

    fn close_sink(&mut self) {
        drop(self.sink.take());
    }

    async fn drain_writer(&mut self) {
        self.owner
            .drain_until(Instant::now() + Duration::from_secs(5))
            .await
            .expect("scheduler test writer should drain");
    }

    fn connection(&self) -> Connection {
        Connection::open(&self.database_path).expect("scheduler test database should open")
    }
}

fn config(
    path: &Path,
    candidate_count: usize,
    max_pending: usize,
    shadow_concurrency: usize,
    judge_concurrency: usize,
) -> RouterConfig {
    let candidates = (0..candidate_count)
        .map(|index| {
            json!({
                "id": format!("candidate-{index}"),
                "model": format!("candidate-model-{index}"),
                "model_revision": "2026-06-01",
                "cost_rank": index,
                "max_context_tokens": 32768,
                "capabilities": {"tools": true}
            })
        })
        .collect::<Vec<_>>();
    serde_json::from_value(json!({
        "version": 1,
        "mode": "shadow",
        "project_id": "scheduler-ledger-tests",
        "database_path": path.to_string_lossy(),
        "retention_days": 30,
        "max_evidence_records": 1000,
        "pools": [{
            "id": POOL,
            "api_family": "openai_chat_completions",
            "anchor_models": ["anchor-a"],
            "anchor_revision": "2026-07-01",
            "sampling_probability": 1.0,
            "max_candidates_per_sample": candidate_count,
            "concurrency": {
                "shadow": shadow_concurrency,
                "judge": judge_concurrency,
                "max_pending": max_pending
            },
            "judge": {
                "version": 1,
                "model": "judge-model",
                "model_revision": "2026-07-01",
                "prompt_version": "pairwise-equivalence-v1",
                "rubric_version": "response-trajectory-equivalence-v1",
                "output_schema_version": 1,
                "response_weight": 0.5,
                "trajectory_weight": 0.5,
                "response_floor": 0.8,
                "trajectory_floor": 0.8,
                "judge_confidence_floor": 0.7,
                "pass_threshold": 0.85,
                "max_rationale_bytes": 1024,
                "base_cooloff_seconds": 10,
                "max_cooloff_seconds": 300
            },
            "candidates": candidates
        }]
    }))
    .expect("scheduler test configuration should deserialize")
}

fn window_fixture(
    harness: &Harness,
    replay: Arc<dyn LlmReplayTransport>,
) -> (Arc<PendingTrajectoryWindow>, TrajectoryWindowSeed) {
    let pool = &harness.config.pools[0];
    let family = pool.api_family;
    assert_eq!(replay.capability().api_family, family);
    let anchor_model = pool
        .anchor_models
        .first()
        .expect("scheduler test pool should have an anchor model");
    let request = family_request(family, anchor_model);
    let envelope = FamilyAdapter
        .decode(family, &request)
        .expect("anchor request should decode");
    let request_projection = project_request(&envelope, &CanonicalizerConfig::default())
        .expect("anchor request should project");
    let response_contracts = Arc::new(
        compile_candidate_response_contracts(&envelope.normalized_request, 64 * 1024)
            .expect("candidate response contracts should compile"),
    );
    let candidates = harness.config.pools[0]
        .candidates
        .iter()
        .cloned()
        .map(|candidate_config| {
            let candidate_request = FamilyAdapter
                .with_model(&envelope, &candidate_config.model)
                .expect("candidate model substitution should succeed");
            EligibleCandidate::new(
                candidate_config,
                candidate_request,
                response_contracts.clone(),
            )
        })
        .collect::<Vec<_>>();
    let candidate_facts = candidates
        .iter()
        .map(|candidate| {
            PersistedCandidateFactV1::from_eligible(candidate, &request_projection)
                .expect("candidate fact should be canonical")
        })
        .collect();
    let replay_capability_facts = ReplayCapabilityFactsV1::from_capability(replay.capability())
        .expect("replay capability facts should be valid");
    let pool_identity = harness
        .identity
        .pools
        .get(&pool.id)
        .expect("test pool identity should exist");
    let opened_at = exact_now();
    let pending = Arc::new(PendingTrajectoryWindow {
        schema: PENDING_TRAJECTORY_SCHEMA_V1.to_string(),
        anchor_id: Uuid::now_v7(),
        anchor_call_uuid: Uuid::now_v7(),
        root_uuid: Uuid::now_v7(),
        owner_uuid: Uuid::now_v7(),
        owner_path: Vec::new(),
        pool_id: pool.id.clone(),
        anchor_model_revision: pool.anchor_revision.clone(),
        process_instance_id: harness.identity.process_instance_id,
        project_uuid: harness.identity.project_uuid,
        project_id: harness.identity.project_id.clone(),
        config_generation_id: harness.identity.config_generation_id.clone(),
        policy_version_id: pool_identity.policy_version_id.clone(),
        learning_generation_id: pool_identity.learning_generation_id,
        request_projection,
        routing_context_projection: RouterRoutingContextProjectionV1 {
            schema: ROUTING_CONTEXT_SCHEMA_V1.to_string(),
            tenant_policy_hash: "3".repeat(64),
            agent_policy_hash: "4".repeat(64),
            position_features: BTreeMap::new(),
        },
        normalized_anchor_response: project_anchor_response(
            family,
            &family_response(family, anchor_model, "anchor answer"),
            64 * 1024,
        )
        .expect("anchor response should project"),
        replay_capability_facts,
        candidate_facts,
        requested_progress: 1,
        opened_at,
        deadline_at: opened_at + ChronoDuration::minutes(5),
    });
    let seed = TrajectoryWindowSeed::new(pending.as_ref().clone(), envelope, replay, candidates);
    (pending, seed)
}

fn exact_now() -> chrono::DateTime<Utc> {
    Utc.timestamp_millis_opt(Utc::now().timestamp_millis())
        .single()
        .expect("current millisecond should be representable")
}

fn chat_response(model: &str, text: &str) -> Json {
    json!({
        "id": format!("chatcmpl-{model}"),
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    })
}

fn family_request(family: LlmApiFamily, model: &str) -> LlmRequest {
    let content = match family {
        LlmApiFamily::OpenAIChatCompletions => json!({
            "model": model,
            "messages": [{"role": "user", "content": "answer safely"}]
        }),
        LlmApiFamily::OpenAIResponses => json!({
            "model": model,
            "input": "answer safely"
        }),
        LlmApiFamily::AnthropicMessages => json!({
            "model": model,
            "max_tokens": 512,
            "messages": [{"role": "user", "content": "answer safely"}]
        }),
    };
    LlmRequest {
        headers: Map::new(),
        content,
    }
}

fn family_response(family: LlmApiFamily, model: &str, text: &str) -> Json {
    match family {
        LlmApiFamily::OpenAIChatCompletions => chat_response(model, text),
        LlmApiFamily::OpenAIResponses => json!({
            "id": format!("resp-{model}"),
            "object": "response",
            "created_at": 1,
            "model": model,
            "status": "completed",
            "error": null,
            "incomplete_details": null,
            "output": [{
                "type": "message",
                "id": format!("msg-{model}"),
                "role": "assistant",
                "status": "completed",
                "content": [{
                    "type": "output_text",
                    "text": text,
                    "annotations": []
                }]
            }],
            "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
        }),
        LlmApiFamily::AnthropicMessages => json!({
            "id": format!("msg-{model}"),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": "end_turn",
            "stop_sequence": null,
            "usage": {"input_tokens": 3, "output_tokens": 2}
        }),
    }
}

fn delayed_action(response: Json) -> (ReplayAction, Arc<DelayedReplayControl>) {
    let control = Arc::new(DelayedReplayControl::default());
    (
        ReplayAction::Delayed {
            response,
            control: control.clone(),
        },
        control,
    )
}

fn pending_action() -> (ReplayAction, Arc<PendingReplayControl>) {
    let control = Arc::new(PendingReplayControl::default());
    (
        ReplayAction::Pending {
            control: control.clone(),
        },
        control,
    )
}

async fn wait_until_entered(entered: &Notify) {
    tokio::time::timeout(Duration::from_secs(3), entered.notified())
        .await
        .expect("scripted replay did not start before the test deadline");
}

async fn wait_for_admission(
    admissions: &SchedulerAdmissionPools,
    candidate_count: usize,
) -> SchedulerAdmission {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match admissions.try_acquire(POOL, candidate_count) {
                Ok(admission) => return admission,
                Err(SchedulerAdmissionError::NoPermits) => tokio::task::yield_now().await,
                Err(error) => panic!("unexpected scheduler admission error: {error:?}"),
            }
        }
    })
    .await
    .expect("scheduler admission was not released before the test deadline")
}

async fn await_scheduler(task: tokio::task::JoinHandle<SchedulerExit>) -> SchedulerExit {
    tokio::time::timeout(Duration::from_secs(5), task)
        .await
        .expect("scheduler actor did not exit before the test deadline")
        .expect("scheduler actor task should join cleanly")
}

fn assert_first_candidate_then_candidate_and_judge(models: &[String]) {
    assert_eq!(models.len(), 3);
    assert_eq!(models[0], "candidate-model-0");
    assert_eq!(
        models[1..]
            .iter()
            .filter(|model| model.as_str() == "candidate-model-1")
            .count(),
        1
    );
    assert_eq!(
        models[1..]
            .iter()
            .filter(|model| model.as_str() == "judge-model")
            .count(),
        1
    );
}

fn malformed_candidate_response(model: &str) -> Json {
    let mut response = chat_response(model, "candidate answer");
    response["object"] = json!("response");
    response
}

fn valid_judge_output() -> String {
    json!({
        "response_equivalence": 0.95,
        "trajectory_equivalence": 0.9,
        "judge_confidence": 0.98,
        "hard_failures": [],
        "rationale": "The candidate preserves the anchor response and trajectory behavior."
    })
    .to_string()
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn reset_runtime() {
    *global_context()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = NemoRelayContextState::new();
}

fn strings(connection: &Connection, sql: &str) -> Vec<String> {
    connection
        .prepare(sql)
        .expect("scheduler test query should prepare")
        .query_map([], |row| row.get(0))
        .expect("scheduler test query should execute")
        .collect::<Result<Vec<_>, _>>()
        .expect("scheduler test rows should decode")
}

fn count(connection: &Connection, table: &str) -> i64 {
    connection
        .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |row| {
            row.get(0)
        })
        .expect("scheduler test count should execute")
}

fn assert_successful_exit(exit: &SchedulerExit, candidates: usize, expected_judge_max: usize) {
    assert_eq!(exit.summary().accepted_batches, 1);
    assert_eq!(
        exit.summary().acknowledged_candidates,
        candidates,
        "scheduler summary: {:?}",
        exit.summary()
    );
    assert_eq!(exit.summary().retained_batches, 0);
    assert_eq!(exit.retained_batch_count(), 0);
    assert_eq!(exit.summary().first_failure, None);
    assert_eq!(exit.summary().max_active_batches, 1);
    assert_eq!(exit.summary().max_pending_candidates, candidates);
    let gauges = exit
        .summary()
        .pool_gauges
        .get(POOL)
        .expect("test pool gauges should exist");
    assert_eq!(gauges.shadow_max_in_flight, 1);
    assert_eq!(gauges.judge_max_in_flight, expected_judge_max);
}

fn assert_common_durable_rows(connection: &Connection, candidates: i64) {
    assert_eq!(count(connection, "sample_batches"), 1);
    assert_eq!(count(connection, "shadow_attempts"), candidates);
    assert_eq!(count(connection, "shadow_results"), candidates);
    assert_eq!(
        strings(
            connection,
            "SELECT vector_source_hash FROM shadow_results
             WHERE canonicalizable = 1 AND query_inputs_json IS NOT NULL
             ORDER BY shadow_attempt_id"
        )
        .len(),
        usize::try_from(candidates).expect("test candidate count should fit usize")
    );
    assert_eq!(count(connection, "evidence_vector_links"), 0);
    assert_eq!(
        strings(
            connection,
            "SELECT state FROM sample_batch_state_events ORDER BY event_seq"
        ),
        ["open", "closed"]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn valid_candidate_and_judge_persist_atomic_terminal_and_release_admission() {
    let _runtime_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let replay = ScriptedReplay::new([
        (
            "candidate-model-0".to_string(),
            vec![ReplayAction::Response(chat_response(
                "candidate-model-0",
                "candidate answer",
            ))],
        ),
        (
            "judge-model".to_string(),
            vec![ReplayAction::Response(chat_response(
                "judge-model",
                &valid_judge_output(),
            ))],
        ),
    ]);
    let mut harness = Harness::new(1);
    harness.deliver(replay.clone()).await;
    let (scheduler, admissions) = harness.scheduler();
    assert!(matches!(
        admissions.try_acquire(POOL, 1),
        Err(SchedulerAdmissionError::NoPermits)
    ));

    let exit = scheduler.run().await;
    assert_successful_exit(&exit, 1, 1);
    let released = admissions
        .try_acquire(POOL, 1)
        .expect("terminal acknowledgement should release batch and candidate admission");
    drop(released);
    harness.drain_writer().await;

    assert_eq!(
        replay.started_models(),
        ["candidate-model-0", "judge-model"]
    );
    assert_eq!(replay.remaining_actions(), 0);
    let connection = harness.connection();
    assert_common_durable_rows(&connection, 1);
    assert_eq!(count(&connection, "judge_attempts"), 1);
    assert_eq!(count(&connection, "evaluations"), 1);
    assert_eq!(
        strings(
            &connection,
            "SELECT state FROM shadow_attempt_state_events ORDER BY event_seq"
        ),
        ["reserved", "started", "completed"]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT state FROM judge_attempt_state_events ORDER BY event_seq"
        ),
        ["started", "valid"]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT dk.key_kind || ':' || dse.state
             FROM dependency_state_events AS dse
             JOIN dependency_keys AS dk USING (dependency_key_id)
             ORDER BY dse.event_seq"
        ),
        [
            "candidate:admitted",
            "candidate:success",
            "judge:admitted",
            "judge:success",
        ]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT sr.terminal_class || ':' || e.source
             FROM shadow_results AS sr
             JOIN evaluations AS e USING (evaluation_id)"
        ),
        ["completed:judge"]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deterministic_candidate_failure_bypasses_judge_and_closes_batch() {
    let _runtime_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let replay = ScriptedReplay::new([(
        "candidate-model-0".to_string(),
        vec![ReplayAction::Response(malformed_candidate_response(
            "candidate-model-0",
        ))],
    )]);
    let mut harness = Harness::new(1);
    harness.deliver(replay.clone()).await;
    let (scheduler, admissions) = harness.scheduler();
    let exit = scheduler.run().await;
    assert_successful_exit(&exit, 1, 0);
    drop(
        admissions
            .try_acquire(POOL, 1)
            .expect("deterministic terminal should release admission"),
    );
    harness.drain_writer().await;

    assert_eq!(replay.started_models(), ["candidate-model-0"]);
    assert_eq!(replay.remaining_actions(), 0);
    let connection = harness.connection();
    assert_common_durable_rows(&connection, 1);
    assert_eq!(count(&connection, "judge_attempts"), 0);
    assert_eq!(count(&connection, "evaluations"), 1);
    assert_eq!(count(&connection, "evaluation_hard_failures"), 1);
    assert_eq!(
        strings(
            &connection,
            "SELECT state FROM shadow_attempt_state_events ORDER BY event_seq"
        ),
        ["reserved", "started", "deterministic_failure"]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT terminal_class || ':' || deterministic_hard_failure FROM shadow_results"
        ),
        ["deterministic_failure:malformed_candidate"]
    );
    assert_eq!(
        strings(&connection, "SELECT source FROM evaluations"),
        ["deterministic_validator"]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT dk.key_kind || ':' || dse.state
             FROM dependency_state_events AS dse
             JOIN dependency_keys AS dk USING (dependency_key_id)
             ORDER BY dse.event_seq"
        ),
        ["candidate:admitted", "candidate:success"]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_initial_judge_runs_one_repair_under_one_judge_permit() {
    let _runtime_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let replay = ScriptedReplay::new([
        (
            "candidate-model-0".to_string(),
            vec![ReplayAction::Response(chat_response(
                "candidate-model-0",
                "candidate answer",
            ))],
        ),
        (
            "judge-model".to_string(),
            vec![
                ReplayAction::Response(chat_response("judge-model", "not valid judge JSON")),
                ReplayAction::Response(chat_response("judge-model", &valid_judge_output())),
            ],
        ),
    ]);
    let mut harness = Harness::new(1);
    harness.deliver(replay.clone()).await;
    let (scheduler, admissions) = harness.scheduler();
    let exit = scheduler.run().await;
    assert_successful_exit(&exit, 1, 1);
    assert_eq!(
        exit.summary().pool_gauges[POOL].judge_max_in_flight,
        1,
        "initial and repair must remain inside one Judge gauge/permit lifetime"
    );
    drop(
        admissions
            .try_acquire(POOL, 1)
            .expect("repaired terminal should release admission"),
    );
    harness.drain_writer().await;

    assert_eq!(
        replay.started_models(),
        ["candidate-model-0", "judge-model", "judge-model"]
    );
    assert_eq!(replay.remaining_actions(), 0);
    assert_eq!(replay.max_active_judges(), 1);
    let connection = harness.connection();
    assert_common_durable_rows(&connection, 1);
    assert_eq!(count(&connection, "judge_attempts"), 2);
    assert_eq!(count(&connection, "evaluations"), 1);
    assert_eq!(
        strings(
            &connection,
            "SELECT CAST(attempt_ordinal AS TEXT) FROM judge_attempts ORDER BY attempt_ordinal"
        ),
        ["0", "1"]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT state FROM judge_attempt_state_events ORDER BY event_seq"
        ),
        ["started", "invalid", "started", "valid"]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT sr.terminal_class || ':' || e.source
             FROM shadow_results AS sr
             JOIN evaluations AS e USING (evaluation_id)"
        ),
        ["completed:judge"]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn operational_candidate_failure_does_not_prevent_sibling_terminal() {
    let _runtime_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let replay = ScriptedReplay::new([
        ("candidate-model-0".to_string(), vec![ReplayAction::Error]),
        (
            "candidate-model-1".to_string(),
            vec![ReplayAction::Response(chat_response(
                "candidate-model-1",
                "sibling candidate answer",
            ))],
        ),
        (
            "judge-model".to_string(),
            vec![ReplayAction::Response(chat_response(
                "judge-model",
                &valid_judge_output(),
            ))],
        ),
    ]);
    let mut harness = Harness::new(2);
    harness.deliver(replay.clone()).await;
    let (scheduler, admissions) = harness.scheduler();
    let exit = scheduler.run().await;
    assert_successful_exit(&exit, 2, 1);
    drop(
        admissions
            .try_acquire(POOL, 2)
            .expect("both sibling terminals should release the complete admission"),
    );
    harness.drain_writer().await;

    assert_first_candidate_then_candidate_and_judge(&replay.started_models());
    assert_eq!(replay.remaining_actions(), 0);
    let connection = harness.connection();
    assert_common_durable_rows(&connection, 2);
    assert_eq!(count(&connection, "judge_attempts"), 1);
    assert_eq!(count(&connection, "evaluations"), 1);
    assert_eq!(
        strings(
            &connection,
            "SELECT sa.candidate_id || ':' || sr.terminal_class
             FROM shadow_results AS sr
             JOIN shadow_attempts AS sa USING (shadow_attempt_id)
             ORDER BY sa.cost_rank"
        ),
        ["candidate-0:operational_failure", "candidate-1:completed",]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT sa.candidate_id || ':' || sase.state
             FROM shadow_attempt_state_events AS sase
             JOIN shadow_attempts AS sa USING (shadow_attempt_id)
             WHERE sase.state NOT IN ('reserved', 'started')
             ORDER BY sa.cost_rank"
        ),
        ["candidate-0:operational_failure", "candidate-1:completed",]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT dk.key_kind || ':' || dse.state
             FROM dependency_state_events AS dse
             JOIN dependency_keys AS dk USING (dependency_key_id)
             ORDER BY dse.event_seq"
        ),
        [
            "candidate:admitted",
            "candidate:failure",
            "candidate:admitted",
            "candidate:success",
            "judge:admitted",
            "judge:success",
        ]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shadow_and_judge_overlap_without_retaining_the_shadow_permit() {
    let _runtime_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let (second_shadow, second_shadow_control) = delayed_action(chat_response(
        "candidate-model-1",
        "second candidate answer",
    ));
    let (first_judge, first_judge_control) =
        delayed_action(chat_response("judge-model", &valid_judge_output()));
    let replay = ScriptedReplay::new([
        (
            "candidate-model-0".to_string(),
            vec![ReplayAction::Response(chat_response(
                "candidate-model-0",
                "first candidate answer",
            ))],
        ),
        ("candidate-model-1".to_string(), vec![second_shadow]),
        (
            "judge-model".to_string(),
            vec![
                first_judge,
                ReplayAction::Response(chat_response("judge-model", &valid_judge_output())),
            ],
        ),
    ]);
    let mut harness = Harness::with_limits(2, 1, 1, 1);
    harness.deliver(replay.clone()).await;
    let start_gate = BackgroundStartGate::new();
    let registry = ActiveReplayRegistry::new(4).unwrap();
    let cancellation = EvaluatorCancellation::default();
    let (scheduler, admissions) =
        harness.scheduler_with_controls(start_gate, registry, cancellation);
    let actor = tokio::spawn(scheduler.run());

    wait_until_entered(&second_shadow_control.entered).await;
    wait_until_entered(&first_judge_control.entered).await;
    assert_eq!(replay.active_shadows(), 1);
    assert_eq!(replay.active_judges(), 1);
    assert_eq!(replay.max_active_shadows(), 1);
    assert_eq!(replay.max_active_judges(), 1);
    assert_first_candidate_then_candidate_and_judge(&replay.started_models());

    second_shadow_control.release.notify_one();
    tokio::time::timeout(Duration::from_secs(3), async {
        while replay.active_shadows() != 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("second Shadow replay should finish while Judge remains active");
    assert_eq!(replay.active_judges(), 1);
    assert_first_candidate_then_candidate_and_judge(&replay.started_models());
    first_judge_control.release.notify_one();

    drop(wait_for_admission(&admissions, 2).await);
    harness.close_sink();
    let exit = await_scheduler(actor).await;
    assert_successful_exit(&exit, 2, 1);
    assert_eq!(replay.max_active_shadows(), 1);
    assert_eq!(replay.max_active_judges(), 1);
    assert_eq!(replay.remaining_actions(), 0);
    harness.drain_writer().await;
    let connection = harness.connection();
    assert_common_durable_rows(&connection, 2);
    assert_eq!(count(&connection, "judge_attempts"), 2);
    assert_eq!(count(&connection, "evaluations"), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sustained_sequential_batches_remain_within_one_slot_actor_capacity() {
    const BATCHES: usize = 12;

    let _runtime_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let replay = ScriptedReplay::new([
        (
            "candidate-model-0".to_string(),
            (0..BATCHES)
                .map(|_| {
                    ReplayAction::Response(chat_response("candidate-model-0", "candidate answer"))
                })
                .collect(),
        ),
        (
            "judge-model".to_string(),
            (0..BATCHES)
                .map(|_| {
                    ReplayAction::Response(chat_response("judge-model", &valid_judge_output()))
                })
                .collect(),
        ),
    ]);
    let mut harness = Harness::with_limits(1, 1, 1, 1);
    harness.deliver(replay.clone()).await;
    let (scheduler, admissions) = harness.scheduler_with_controls(
        BackgroundStartGate::new(),
        ActiveReplayRegistry::new(4).unwrap(),
        EvaluatorCancellation::default(),
    );
    let actor = tokio::spawn(scheduler.run());

    for batch in 1..BATCHES {
        drop(wait_for_admission(&admissions, 1).await);
        harness.deliver(replay.clone()).await;
        assert_eq!(
            replay.started_models().len(),
            batch * 2,
            "the next batch must not start before the prior admission is acknowledged"
        );
    }
    drop(wait_for_admission(&admissions, 1).await);
    harness.close_sink();
    let exit = await_scheduler(actor).await;

    assert_eq!(exit.summary().accepted_batches, BATCHES);
    assert_eq!(exit.summary().acknowledged_candidates, BATCHES);
    assert_eq!(exit.summary().retained_batches, 0);
    assert_eq!(exit.summary().first_failure, None);
    assert_eq!(exit.retained_batch_count(), 0);
    assert!(exit.summary().max_task_entries <= 1);
    assert!(exit.summary().max_active_batches <= 1);
    assert!(exit.summary().max_pending_candidates <= 1);
    let gauges = exit.summary().pool_gauges[POOL];
    assert!(gauges.shadow_max_in_flight <= 1);
    assert!(gauges.judge_max_in_flight <= 1);
    assert!(gauges.shadow_max_queued <= 1);
    assert!(gauges.judge_max_queued <= 1);
    assert_eq!(replay.remaining_actions(), 0);
    assert_eq!(replay.started_models().len(), BATCHES * 2);
    harness.drain_writer().await;

    let connection = harness.connection();
    assert_eq!(count(&connection, "sample_batches"), BATCHES as i64);
    assert_eq!(count(&connection, "shadow_attempts"), BATCHES as i64);
    assert_eq!(count(&connection, "shadow_results"), BATCHES as i64);
    assert_eq!(count(&connection, "judge_attempts"), BATCHES as i64);
    assert_eq!(count(&connection, "evaluations"), BATCHES as i64);
    assert_eq!(
        strings(
            &connection,
            "SELECT state FROM sample_batch_state_events WHERE state <> 'open' ORDER BY event_seq"
        ),
        vec!["closed"; BATCHES]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_invalid_judges_cool_off_the_next_same_dependency_without_a_third_start() {
    let _runtime_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let replay = ScriptedReplay::new([
        (
            "candidate-model-0".to_string(),
            vec![
                ReplayAction::Response(chat_response(
                    "candidate-model-0",
                    "first candidate answer",
                )),
                ReplayAction::Response(chat_response(
                    "candidate-model-0",
                    "second candidate answer",
                )),
            ],
        ),
        (
            "judge-model".to_string(),
            vec![
                ReplayAction::Response(chat_response("judge-model", "invalid initial output")),
                ReplayAction::Response(chat_response("judge-model", "invalid repair output")),
            ],
        ),
    ]);
    let mut harness = Harness::with_limits(1, 1, 1, 1);
    harness.deliver(replay.clone()).await;
    let (scheduler, admissions) = harness.scheduler_with_controls(
        BackgroundStartGate::new(),
        ActiveReplayRegistry::new(4).unwrap(),
        EvaluatorCancellation::default(),
    );
    let actor = tokio::spawn(scheduler.run());

    drop(wait_for_admission(&admissions, 1).await);
    harness.deliver(replay.clone()).await;
    drop(wait_for_admission(&admissions, 1).await);
    harness.close_sink();
    let exit = await_scheduler(actor).await;

    assert_eq!(exit.summary().accepted_batches, 2);
    assert_eq!(exit.summary().acknowledged_candidates, 2);
    assert_eq!(exit.summary().retained_batches, 0);
    assert_eq!(exit.summary().first_failure, None);
    assert_eq!(
        replay.started_models(),
        [
            "candidate-model-0",
            "judge-model",
            "judge-model",
            "candidate-model-0",
        ]
    );
    assert_eq!(replay.remaining_actions(), 0);
    harness.drain_writer().await;

    let connection = harness.connection();
    assert_eq!(count(&connection, "sample_batches"), 2);
    assert_eq!(count(&connection, "shadow_results"), 2);
    assert_eq!(count(&connection, "judge_attempts"), 2);
    assert_eq!(count(&connection, "evaluations"), 0);
    assert_eq!(
        strings(
            &connection,
            "SELECT terminal_class || ':' || COALESCE(operational_failure_class, '')
             FROM shadow_results ORDER BY rowid"
        ),
        [
            "operational_failure:router.judge.output_invalid",
            "skipped_cooloff:",
        ]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT state FROM judge_attempt_state_events ORDER BY event_seq"
        ),
        ["started", "invalid", "started", "invalid"]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT dk.key_kind || ':' || dse.state
             FROM dependency_state_events AS dse
             JOIN dependency_keys AS dk USING (dependency_key_id)
             ORDER BY dse.event_seq"
        ),
        [
            "candidate:admitted",
            "candidate:success",
            "judge:admitted",
            "judge:failure",
            "candidate:admitted",
            "candidate:success",
            "judge:skipped_cooloff",
        ]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_cancels_one_active_replay_once_and_never_starts_the_queued_candidate() {
    let _runtime_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let (pending, pending_control) = pending_action();
    let replay = ScriptedReplay::new([
        ("candidate-model-0".to_string(), vec![pending]),
        (
            "candidate-model-1".to_string(),
            vec![ReplayAction::Response(chat_response(
                "candidate-model-1",
                "must never start",
            ))],
        ),
    ]);
    let mut harness = Harness::with_limits(2, 1, 1, 1);
    harness.deliver(replay.clone()).await;
    let start_gate = BackgroundStartGate::new();
    let active_replays = ActiveReplayRegistry::new(4).unwrap();
    let cancellation = EvaluatorCancellation::default();
    let (scheduler, admissions) = harness.scheduler_with_controls(
        start_gate.clone(),
        active_replays.clone(),
        cancellation.clone(),
    );
    let actor = tokio::spawn(scheduler.run());

    wait_until_entered(&pending_control.entered).await;
    assert!(matches!(
        admissions.try_acquire(POOL, 2),
        Err(SchedulerAdmissionError::NoPermits)
    ));
    assert!(start_gate.close());
    assert_eq!(active_replays.close_and_cancel_all(), 1);
    assert!(cancellation.cancel());
    harness.close_sink();

    let exit = await_scheduler(actor).await;
    assert_successful_exit(&exit, 2, 0);
    assert_eq!(pending_control.cancellations.load(Ordering::Acquire), 1);
    assert_eq!(replay.started_models(), ["candidate-model-0"]);
    assert_eq!(replay.remaining_actions(), 1);
    drop(
        admissions
            .try_acquire(POOL, 2)
            .expect("canceled terminals must release all admission permits"),
    );
    harness.drain_writer().await;

    let connection = harness.connection();
    assert_eq!(count(&connection, "sample_batches"), 1);
    assert_eq!(count(&connection, "shadow_results"), 2);
    assert_eq!(count(&connection, "judge_attempts"), 0);
    assert_eq!(count(&connection, "evaluations"), 0);
    assert_eq!(
        strings(
            &connection,
            "SELECT state FROM sample_batch_state_events ORDER BY event_seq"
        ),
        ["open", "canceled_shutdown"]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT sa.candidate_id || ':' || sase.state
             FROM shadow_attempt_state_events AS sase
             JOIN shadow_attempts AS sa USING (shadow_attempt_id)
             ORDER BY sa.cost_rank, sase.event_seq"
        ),
        [
            "candidate-0:reserved",
            "candidate-0:started",
            "candidate-0:canceled_shutdown",
            "candidate-1:reserved",
            "candidate-1:canceled_shutdown",
        ]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT terminal_class FROM shadow_results ORDER BY rowid"
        ),
        ["canceled_shutdown", "canceled_shutdown"]
    );
    assert_eq!(
        strings(
            &connection,
            "SELECT dk.key_kind || ':' || dse.state
             FROM dependency_state_events AS dse
             JOIN dependency_keys AS dk USING (dependency_key_id)
             ORDER BY dse.event_seq"
        ),
        ["candidate:admitted"]
    );
}

#[path = "scheduler_live_provider_tests.rs"]
mod live_provider_tests;
