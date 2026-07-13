// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-ledger exact-once coverage for the Active foreground execution branch.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use nemo_relay::api::event::{BaseEvent, Event, EventCategory, ScopeCategory, ScopeEvent};
use nemo_relay::api::llm::{LlmApiFamily, LlmRequest};
use nemo_relay::api::runtime::{
    LLM_REPLAY_CONTRACT_VERSION, LlmExecutionNextFn, LlmReplayCall, LlmReplayCapability,
    LlmReplayTransport,
};
use nemo_relay::error::{FlowError, Result as FlowResult};
use serde_json::json;
use uuid::Uuid;

use crate::ledger::cohort::{CohortAssignmentRequest, RandomizedCohort};
use crate::ledger::repository::active::ActiveAdmissionAck;
use crate::ledger::repository::retention::{RetentionAck, RetentionRequest};
use crate::ledger::repository::{
    ActivatedLedger, LedgerRepository, active_runtime_context_for_root, active_runtime_request,
    ready_evaluated_active_runtime_fixture,
    ready_evaluated_active_runtime_fixture_with_attribution_seconds,
};
use crate::ledger::writer::LedgerWriterOwner;
use crate::runtime::{ActiveDispatchGuardV2, RouterRuntime};

struct InspectOnlyReplay {
    capability: LlmReplayCapability,
    starts: AtomicUsize,
}

impl InspectOnlyReplay {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            capability: LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: LlmApiFamily::OpenAIChatCompletions,
                transport_identity: "transport-shared".to_string(),
            },
            starts: AtomicUsize::new(0),
        })
    }
}

impl LlmReplayTransport for InspectOnlyReplay {
    fn capability(&self) -> &LlmReplayCapability {
        &self.capability
    }

    fn start(&self, _request: LlmRequest) -> FlowResult<LlmReplayCall> {
        self.starts.fetch_add(1, Ordering::AcqRel);
        Err(FlowError::Internal(
            "Active foreground must not start replay".to_string(),
        ))
    }
}

fn cohort_root(
    config: &crate::config::RouterConfig,
    activated: &ActivatedLedger,
    desired: RandomizedCohort,
) -> Uuid {
    let active = config.pools[0]
        .learning
        .as_ref()
        .and_then(|learning| learning.complete_active_policy())
        .unwrap();
    for _ in 0..10_000 {
        let root = Uuid::now_v7();
        let assignment = activated
            .cohort_assignment
            .assign(CohortAssignmentRequest::new(
                root,
                &activated.identity.config_generation_id,
                "pool-a",
                "candidate-a",
                active.holdout_probability,
                active.active_canary_fraction,
            ))
            .unwrap();
        if assignment.cohort() == desired {
            return root;
        }
    }
    panic!("deterministic cohort search did not find a candidate root")
}

fn candidate_root(config: &crate::config::RouterConfig, activated: &ActivatedLedger) -> Uuid {
    cohort_root(config, activated, RandomizedCohort::ActiveCanary)
}

fn success_next(models: Arc<Mutex<Vec<String>>>) -> LlmExecutionNextFn {
    Arc::new(move |request| {
        let models = Arc::clone(&models);
        Box::pin(async move {
            let model = request.content["model"].as_str().unwrap().to_string();
            models.lock().unwrap().push(model.clone());
            Ok(json!({
                "id": "chatcmpl-active",
                "object": "chat.completion",
                "created": 1,
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": "active response"},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}
            }))
        })
    })
}

fn open_runtime(runtime: &Arc<RouterRuntime>, name: &str) {
    let registration = runtime.lifecycle_registration(name.to_string()).unwrap();
    drop(registration);
}

#[tokio::test]
async fn candidate_handoff_rewrites_once_and_persists_dispatch_before_return() {
    let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
    let (_temporary, config, activated, _space, _vector, _evidence, _evaluation) =
        ready_evaluated_active_runtime_fixture();
    let root = candidate_root(&config, &activated);
    let database_path = config.database_path.clone();
    let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
    open_runtime(&runtime, "active-candidate-success");
    let replay = InspectOnlyReplay::new();
    let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
    let models = Arc::new(Mutex::new(Vec::new()));

    let response = runtime
        .execute_for_test(
            Arc::new(active_runtime_context_for_root(root)),
            active_runtime_request(),
            Some(replay_transport),
            success_next(Arc::clone(&models)),
        )
        .await
        .unwrap();
    assert_eq!(response["model"], json!("candidate-model-a"));
    assert_eq!(models.lock().unwrap().as_slice(), ["candidate-model-a"]);
    assert_eq!(replay.starts.load(Ordering::Acquire), 0);

    let inspection = rusqlite::Connection::open(&database_path).unwrap();
    assert_eq!(
        inspection
            .query_row(
                "SELECT count(*) FROM decisions WHERE mode = 'active'",
                [],
                |row| { row.get::<_, i64>(0) }
            )
            .unwrap(),
        1
    );
    assert_eq!(
        inspection
            .query_row(
                "SELECT terminal_state FROM active_dispatch_terminal_events",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "completed"
    );
    assert_eq!(
        inspection
            .query_row("SELECT count(*) FROM active_root_signals", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        1
    );
    drop(inspection);

    let owner_end = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("completed")
            .uuid(root)
            .metadata(json!({"otel.status_code": "OK"}))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::new("agent"),
        None,
    ));
    runtime.subscriber_callback()(&owner_end);

    runtime
        .drain_for_test(Instant::now() + Duration::from_secs(15))
        .await
        .unwrap();
    let inspection = rusqlite::Connection::open(&database_path).unwrap();
    assert_eq!(
        inspection
            .query_row("SELECT count(*) FROM outcomes", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        1
    );
    assert_eq!(
        inspection
            .query_row(
                "SELECT state FROM active_root_window_state_events
                        ORDER BY event_seq DESC LIMIT 1",
                [],
                |row| { row.get::<_, String>(0) }
            )
            .unwrap(),
        "completed"
    );
    assert_eq!(
        inspection
            .query_row("SELECT attribution_status FROM outcomes", [], |row| {
                row.get::<_, String>(0)
            })
            .unwrap(),
        "eligible_treatment"
    );
}

#[tokio::test]
async fn attributable_tool_failure_persists_a_failed_outcome_and_query_cooloff() {
    let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
    let (_temporary, config, activated, _space, _vector, _evidence, _evaluation) =
        ready_evaluated_active_runtime_fixture();
    let root = candidate_root(&config, &activated);
    let database_path = config.database_path.clone();
    let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
    open_runtime(&runtime, "active-candidate-tool-failure");
    let replay = InspectOnlyReplay::new();
    let replay_transport: Arc<dyn LlmReplayTransport> = replay;

    runtime
        .execute_for_test(
            Arc::new(active_runtime_context_for_root(root)),
            active_runtime_request(),
            Some(replay_transport),
            success_next(Arc::new(Mutex::new(Vec::new()))),
        )
        .await
        .unwrap();

    let tool_end = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("lookup")
            .uuid(Uuid::now_v7())
            .parent_uuid(root)
            .metadata(json!({"otel.status_code": "ERROR"}))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::tool(),
        None,
    ));
    runtime.subscriber_callback()(&tool_end);
    let owner_end = Event::Scope(ScopeEvent::new(
        BaseEvent::builder()
            .name("completed")
            .uuid(root)
            .metadata(json!({"otel.status_code": "OK"}))
            .build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::new("agent"),
        None,
    ));
    runtime.subscriber_callback()(&owner_end);
    runtime
        .drain_for_test(Instant::now() + Duration::from_secs(15))
        .await
        .unwrap();

    let inspection = rusqlite::Connection::open(&database_path).unwrap();
    assert_eq!(
        inspection
            .query_row(
                "SELECT label || ':' || attribution_status FROM outcomes",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "failure:eligible_treatment"
    );
    assert_eq!(
        inspection
            .query_row(
                "SELECT state || ':' || stable_reason || ':' ||
                        (cause_outcome_id IS NOT NULL)
                 FROM active_neighborhood_state_events
                 ORDER BY event_seq DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "cooloff:candidate_outcome_failure:1"
    );
}

#[tokio::test]
async fn attribution_deadline_flushes_through_the_coordinator_before_terminalization() {
    let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
    let (_temporary, config, activated, _space, _vector, _evidence, _evaluation) =
        ready_evaluated_active_runtime_fixture_with_attribution_seconds(1);
    let root = candidate_root(&config, &activated);
    let database_path = config.database_path.clone();
    let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
    open_runtime(&runtime, "active-deadline-barrier");
    runtime
        .execute_for_test(
            Arc::new(active_runtime_context_for_root(root)),
            active_runtime_request(),
            Some(InspectOnlyReplay::new()),
            success_next(Arc::new(Mutex::new(Vec::new()))),
        )
        .await
        .unwrap();

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let outcome = rusqlite::Connection::open(&database_path)
            .unwrap()
            .query_row(
                "SELECT label || ':' || attribution_status FROM outcomes",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok();
        if outcome.as_deref() == Some("success:eligible_treatment") {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "deadline outcome was not delivered"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    runtime
        .drain_for_test(Instant::now() + Duration::from_secs(15))
        .await
        .unwrap();
}

#[tokio::test]
async fn terminal_active_graph_retires_in_bounded_receipted_steps_across_a_wal_reader() {
    let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
    let (_temporary, config, activated, _space, _vector, _evidence, _evaluation) =
        ready_evaluated_active_runtime_fixture();
    let restart_config = config.clone();
    let root = candidate_root(&config, &activated);
    let database_path = config.database_path.clone();
    let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
    open_runtime(&runtime, "active-retention-graph");
    runtime
        .execute_for_test(
            Arc::new(active_runtime_context_for_root(root)),
            active_runtime_request(),
            Some(InspectOnlyReplay::new()),
            success_next(Arc::new(Mutex::new(Vec::new()))),
        )
        .await
        .unwrap();
    runtime.subscriber_callback()(&Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("completed").uuid(root).build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::new("agent"),
        None,
    )));
    runtime
        .drain_for_test(Instant::now() + Duration::from_secs(15))
        .await
        .unwrap();
    drop(runtime);

    let now = chrono::Utc::now().timestamp_millis().max(1);
    let mut retirement = LedgerRepository::activate_at(&restart_config, now).unwrap();
    let experiment_id = rusqlite::Connection::open(&database_path)
        .unwrap()
        .query_row(
            "SELECT active_experiment_id FROM active_experiments",
            [],
            |row| row.get::<_, String>(0),
        )
        .unwrap();
    retirement
        .repository
        .force_active_experiment_terminal_for_retention(Uuid::parse_str(&experiment_id).unwrap(), 0)
        .unwrap();

    let snapshot = rusqlite::Connection::open(&database_path).unwrap();
    snapshot.execute_batch("BEGIN DEFERRED").unwrap();
    assert_eq!(
        snapshot
            .query_row("SELECT count(*) FROM active_experiments", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        1
    );
    let mut completed = false;
    for ordinal in 0..32_i64 {
        let request =
            RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), now + ordinal + 1).unwrap();
        assert!(matches!(
            retirement.repository.run_retention(&request).unwrap(),
            RetentionAck::Applied { .. }
        ));
        let inspection = rusqlite::Connection::open(&database_path).unwrap();
        let invalid_receipts = inspection
            .query_row(
                "SELECT count(*) FROM active_retirement_receipts
                 WHERE parent_rows_deleted > 1000
                    OR child_rows_deleted > 4095
                    OR bytes_deleted > 33554432",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(invalid_receipts, 0);
        if inspection
            .query_row("SELECT count(*) FROM active_experiments", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap()
            == 0
        {
            completed = true;
            break;
        }
    }
    assert!(
        completed,
        "Active retirement did not converge within 32 batches"
    );
    assert_eq!(
        snapshot
            .query_row("SELECT count(*) FROM active_experiments", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        1,
        "held WAL reader lost its original snapshot"
    );
    snapshot.execute_batch("COMMIT").unwrap();
    assert_eq!(
        snapshot
            .query_row("SELECT count(*) FROM active_experiments", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        0
    );
    let inspection = rusqlite::Connection::open(&database_path).unwrap();
    assert_eq!(
        inspection
            .query_row(
                "SELECT count(*) FROM active_retirement_receipts WHERE completed = 1",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
    assert_eq!(
        inspection
            .query_row("SELECT count(*) FROM pragma_foreign_key_check", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        0
    );
}

#[tokio::test]
async fn candidate_provider_error_is_returned_once_without_anchor_retry_and_invalidates_query() {
    let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
    let (_temporary, config, activated, _space, _vector, _evidence, _evaluation) =
        ready_evaluated_active_runtime_fixture();
    let root = candidate_root(&config, &activated);
    let database_path = config.database_path.clone();
    let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
    open_runtime(&runtime, "active-candidate-error");
    let replay = InspectOnlyReplay::new();
    let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
    let calls = Arc::new(AtomicUsize::new(0));
    let callback_calls = Arc::clone(&calls);
    let next: LlmExecutionNextFn = Arc::new(move |request| {
        let callback_calls = Arc::clone(&callback_calls);
        Box::pin(async move {
            callback_calls.fetch_add(1, Ordering::AcqRel);
            assert_eq!(request.content["model"], json!("candidate-model-a"));
            Err(FlowError::Internal(
                "candidate provider sentinel".to_string(),
            ))
        })
    });

    let error = runtime
        .execute_for_test(
            Arc::new(active_runtime_context_for_root(root)),
            active_runtime_request(),
            Some(replay_transport),
            next,
        )
        .await
        .unwrap_err();
    assert_eq!(calls.load(Ordering::Acquire), 1);
    assert_eq!(replay.starts.load(Ordering::Acquire), 0);
    assert!(matches!(
        error,
        FlowError::Internal(ref message) if message == "candidate provider sentinel"
    ));

    let inspection = rusqlite::Connection::open(&database_path).unwrap();
    assert_eq!(
        inspection
            .query_row(
                "SELECT terminal_state FROM active_dispatch_terminal_events",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "provider_error"
    );
    assert_eq!(
        inspection
            .query_row(
                "SELECT state FROM active_neighborhood_state_events
                 ORDER BY event_seq DESC LIMIT 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "cooloff"
    );
    drop(inspection);

    runtime
        .drain_for_test(Instant::now() + Duration::from_secs(15))
        .await
        .unwrap();
}

#[tokio::test]
async fn candidate_codec_failure_returns_raw_response_and_records_policy_failure() {
    let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
    let (_temporary, config, activated, _space, _vector, _evidence, _evaluation) =
        ready_evaluated_active_runtime_fixture();
    let root = candidate_root(&config, &activated);
    let database_path = config.database_path.clone();
    let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
    open_runtime(&runtime, "active-candidate-codec-failure");
    let raw = json!({"model": "candidate-model-a", "choices": "malformed"});
    let expected = raw.clone();
    let next: LlmExecutionNextFn = Arc::new(move |request| {
        let raw = raw.clone();
        Box::pin(async move {
            assert_eq!(request.content["model"], json!("candidate-model-a"));
            Ok(raw)
        })
    });

    let response = runtime
        .execute_for_test(
            Arc::new(active_runtime_context_for_root(root)),
            active_runtime_request(),
            Some(InspectOnlyReplay::new()),
            next,
        )
        .await
        .unwrap();
    assert_eq!(response, expected);
    runtime.subscriber_callback()(&Event::Scope(ScopeEvent::new(
        BaseEvent::builder().name("completed").uuid(root).build(),
        ScopeCategory::End,
        Vec::new(),
        EventCategory::new("agent"),
        None,
    )));
    runtime
        .drain_for_test(Instant::now() + Duration::from_secs(15))
        .await
        .unwrap();

    let inspection = rusqlite::Connection::open(&database_path).unwrap();
    assert_eq!(
        inspection
            .query_row(
                "SELECT terminal_state FROM active_dispatch_terminal_events",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "completed"
    );
    assert_eq!(
        inspection
            .query_row(
                "SELECT count(*) FROM active_root_signals WHERE disposition = 'failure'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap(),
        1
    );
    assert_eq!(
        inspection
            .query_row(
                "SELECT label || ':' || attribution_status FROM outcomes",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "failure:eligible_treatment"
    );
}

#[tokio::test]
async fn candidate_panic_is_contained_without_anchor_retry_and_terminalizes_dispatch() {
    let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
    let (_temporary, config, activated, _space, _vector, _evidence, _evaluation) =
        ready_evaluated_active_runtime_fixture();
    let root = candidate_root(&config, &activated);
    let database_path = config.database_path.clone();
    let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
    open_runtime(&runtime, "active-candidate-panic");
    let calls = Arc::new(AtomicUsize::new(0));
    let observed = Arc::clone(&calls);
    let next: LlmExecutionNextFn = Arc::new(move |_| {
        let observed = Arc::clone(&observed);
        Box::pin(async move {
            observed.fetch_add(1, Ordering::AcqRel);
            panic!("active continuation panic sentinel");
        })
    });

    let error = runtime
        .execute_for_test(
            Arc::new(active_runtime_context_for_root(root)),
            active_runtime_request(),
            Some(InspectOnlyReplay::new()),
            next,
        )
        .await
        .unwrap_err();
    assert_eq!(calls.load(Ordering::Acquire), 1);
    assert!(matches!(
        error,
        FlowError::Internal(ref message) if message == "Router Active continuation panicked"
    ));
    let inspection = rusqlite::Connection::open(&database_path).unwrap();
    assert_eq!(
        inspection
            .query_row(
                "SELECT terminal_state FROM active_dispatch_terminal_events",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap(),
        "panicked_after_handoff"
    );
    drop(inspection);
    runtime
        .drain_for_test(Instant::now() + Duration::from_secs(15))
        .await
        .unwrap();
}

#[tokio::test]
async fn candidate_cancellation_after_handoff_records_a_terminal_without_late_retry() {
    let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
    let (_temporary, config, activated, _space, _vector, _evidence, _evaluation) =
        ready_evaluated_active_runtime_fixture();
    let root = candidate_root(&config, &activated);
    let database_path = config.database_path.clone();
    let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
    open_runtime(&runtime, "active-candidate-cancel");
    let (started_tx, started_rx) = tokio::sync::oneshot::channel();
    let started_tx = Arc::new(Mutex::new(Some(started_tx)));
    let next: LlmExecutionNextFn = Arc::new(move |request| {
        let started = started_tx.lock().unwrap().take();
        Box::pin(async move {
            assert_eq!(request.content["model"], json!("candidate-model-a"));
            if let Some(started) = started {
                let _ = started.send(());
            }
            std::future::pending::<FlowResult<serde_json::Value>>().await
        })
    });
    let running = Arc::clone(&runtime);
    let task = tokio::spawn(async move {
        running
            .execute_for_test(
                Arc::new(active_runtime_context_for_root(root)),
                active_runtime_request(),
                Some(InspectOnlyReplay::new()),
                next,
            )
            .await
    });
    tokio::time::timeout(Duration::from_secs(10), started_rx)
        .await
        .unwrap()
        .unwrap();
    task.abort();
    assert!(task.await.unwrap_err().is_cancelled());

    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let terminal = rusqlite::Connection::open(&database_path)
            .unwrap()
            .query_row(
                "SELECT terminal_state FROM active_dispatch_terminal_events",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok();
        if terminal.as_deref() == Some("cancelled_after_handoff") {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "dispatch terminal was not delivered"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    runtime
        .drain_for_test(Instant::now() + Duration::from_secs(15))
        .await
        .unwrap();
    let inspection = rusqlite::Connection::open(&database_path).unwrap();
    assert_eq!(
        inspection
            .query_row("SELECT state || ':' || attribution_status
                        FROM active_root_window_state_events AS state_event
                        JOIN outcomes ON outcomes.active_root_window_id = state_event.active_root_window_id
                        WHERE state_event.state <> 'open'", [], |row| {
                row.get::<_, String>(0)
            })
            .unwrap(),
        "shutdown_orphaned:orphaned"
    );
}

#[tokio::test]
async fn reserved_dispatch_guard_records_cancellation_before_handoff() {
    let fixture = crate::ledger::repository::active::tests::fixture();
    let crate::ledger::repository::active::tests::Fixture {
        _directory,
        config,
        mut activated,
        admission,
    } = fixture;
    assert!(matches!(
        activated.repository.admit_active_root(&admission).unwrap(),
        ActiveAdmissionAck::Applied(_)
    ));
    let dispatch = admission.dispatch.as_ref().unwrap();
    let database_path = config.database_path.clone();
    let crate::ledger::repository::ActivatedLedger { repository, .. } = activated;
    let (mut owner, client) = LedgerWriterOwner::start(repository, 8).unwrap();
    let permit = client
        .reserve_active_dispatch_terminal_until(Instant::now() + Duration::from_secs(5))
        .await
        .unwrap();
    drop(ActiveDispatchGuardV2::new(
        permit,
        dispatch.active_dispatch_id,
        admission.active_assignment_id,
    ));

    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        let state = rusqlite::Connection::open(&database_path)
            .unwrap()
            .query_row(
                "SELECT terminal_state FROM active_dispatch_terminal_events",
                [],
                |row| row.get::<_, String>(0),
            )
            .ok();
        if state.as_deref() == Some("cancelled_before_handoff") {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "pre-handoff terminal was not delivered"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
    owner
        .drain_until(Instant::now() + Duration::from_secs(5))
        .await
        .unwrap();
}

#[tokio::test]
async fn foreground_saturation_serves_anchor_once_without_creating_an_active_graph() {
    let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
    let (_temporary, config, activated, _space, _vector, _evidence, _evaluation) =
        ready_evaluated_active_runtime_fixture();
    let root = candidate_root(&config, &activated);
    let database_path = config.database_path.clone();
    let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
    open_runtime(&runtime, "active-foreground-saturation");
    let saturation = runtime.test_saturate_foreground().await;
    let models = Arc::new(Mutex::new(Vec::new()));

    let response = runtime
        .execute_for_test(
            Arc::new(active_runtime_context_for_root(root)),
            active_runtime_request(),
            Some(InspectOnlyReplay::new()),
            success_next(Arc::clone(&models)),
        )
        .await
        .unwrap();
    assert_eq!(response["model"], json!("anchor-a"));
    assert_eq!(models.lock().unwrap().as_slice(), ["anchor-a"]);

    let inspection = rusqlite::Connection::open(&database_path).unwrap();
    for table in ["decisions", "active_root_windows", "active_dispatches"] {
        assert_eq!(
            inspection
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            0,
            "saturated foreground call created rows in {table}"
        );
    }
    drop(inspection);
    drop(saturation);
    runtime
        .drain_for_test(Instant::now() + Duration::from_secs(15))
        .await
        .unwrap();
}

#[tokio::test]
async fn active_pause_is_reversible_and_schedules_no_paused_background_work() {
    let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
    let (_temporary, config, activated, _space, _vector, _evidence, _evaluation) =
        ready_evaluated_active_runtime_fixture();
    let paused_root = candidate_root(&config, &activated);
    let resumed_root = candidate_root(&config, &activated);
    let database_path = config.database_path.clone();
    let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
    open_runtime(&runtime, "active-pause-resume");
    let service = crate::control::router_control_service().unwrap();
    service
        .apply_mutation(
            crate::control::ControlMutation {
                mutation_id: Uuid::now_v7(),
                scope: crate::control::ControlScope::All,
                operation: crate::control::ControlOperation::SetPaused { value: true },
                expected_control_generation: 0,
                actor: "operator-a".into(),
                reason: "pause active".into(),
            },
            crate::control::ControlMutationOptions::default(),
        )
        .await
        .unwrap();
    let replay = InspectOnlyReplay::new();
    let paused_models = Arc::new(Mutex::new(Vec::new()));
    let paused = runtime
        .execute_for_test(
            Arc::new(active_runtime_context_for_root(paused_root)),
            active_runtime_request(),
            Some(replay.clone()),
            success_next(Arc::clone(&paused_models)),
        )
        .await
        .unwrap();
    assert_eq!(paused["model"], json!("anchor-a"));
    assert_eq!(paused_models.lock().unwrap().as_slice(), ["anchor-a"]);
    assert_eq!(
        rusqlite::Connection::open(&database_path)
            .unwrap()
            .query_row("SELECT count(*) FROM decisions", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap(),
        0
    );

    service
        .apply_mutation(
            crate::control::ControlMutation {
                mutation_id: Uuid::now_v7(),
                scope: crate::control::ControlScope::All,
                operation: crate::control::ControlOperation::SetPaused { value: false },
                expected_control_generation: 1,
                actor: "operator-a".into(),
                reason: "resume active".into(),
            },
            crate::control::ControlMutationOptions::default(),
        )
        .await
        .unwrap();
    let resumed = runtime
        .execute_for_test(
            Arc::new(active_runtime_context_for_root(resumed_root)),
            active_runtime_request(),
            Some(replay.clone()),
            success_next(Arc::new(Mutex::new(Vec::new()))),
        )
        .await
        .unwrap();
    assert_eq!(resumed["model"], json!("candidate-model-a"));
    assert_eq!(replay.starts.load(Ordering::Acquire), 0);
    runtime
        .drain_for_test(Instant::now() + Duration::from_secs(15))
        .await
        .unwrap();
}

#[tokio::test]
async fn randomized_anchor_arms_preserve_the_request_and_create_no_dispatch() {
    let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
    for (cohort, expected_reason, expected_arm) in [
        (
            RandomizedCohort::AnchorControl,
            "active_anchor_control",
            "anchor_control",
        ),
        (
            RandomizedCohort::AnchorHoldout,
            "active_anchor_holdout",
            "anchor_holdout",
        ),
    ] {
        let (_temporary, config, activated, _space, _vector, _evidence, _evaluation) =
            ready_evaluated_active_runtime_fixture();
        let root = cohort_root(&config, &activated, cohort);
        let database_path = config.database_path.clone();
        let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
        open_runtime(&runtime, expected_reason);
        let replay = InspectOnlyReplay::new();
        let replay_transport: Arc<dyn LlmReplayTransport> = replay.clone();
        let models = Arc::new(Mutex::new(Vec::new()));
        let response = runtime
            .execute_for_test(
                Arc::new(active_runtime_context_for_root(root)),
                active_runtime_request(),
                Some(replay_transport),
                success_next(Arc::clone(&models)),
            )
            .await
            .unwrap();
        assert_eq!(response["model"], json!("anchor-a"));
        assert_eq!(models.lock().unwrap().as_slice(), ["anchor-a"]);
        assert_eq!(replay.starts.load(Ordering::Acquire), 0);

        let inspection = rusqlite::Connection::open(&database_path).unwrap();
        let (reason, arm, dispatches) = inspection
            .query_row(
                "SELECT decision.final_reason, assignment.arm,
                        (SELECT count(*) FROM active_dispatches)
                 FROM decisions AS decision
                 JOIN active_assignments AS assignment
                   ON assignment.decision_id = decision.decision_id",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(reason, expected_reason);
        assert_eq!(arm, expected_arm);
        assert_eq!(dispatches, 0);
        drop(inspection);
        runtime
            .drain_for_test(Instant::now() + Duration::from_secs(15))
            .await
            .unwrap();
    }
}
