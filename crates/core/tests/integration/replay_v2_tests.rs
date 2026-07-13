// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! V2 managed LLM replay, mixed-registry, and isolation integration tests.

#![allow(clippy::await_holding_lock)]

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::TimeDelta;
use nemo_relay::api::event::{Event, PendingMarkSpec, ScopeCategory};
use nemo_relay::api::llm::{
    LlmApiFamily, LlmAttributes, LlmCallExecuteParams, LlmCallExecuteV2Params, LlmCallRole,
    LlmExecutionContextSnapshot, LlmRequest, LlmRequestInterceptOutcome, llm_call_execute,
    llm_call_execute_v2,
};
use nemo_relay::api::registry::{
    deregister_llm_conditional_execution_guardrail, deregister_llm_execution_intercept,
    deregister_llm_request_intercept, deregister_llm_sanitize_request_guardrail,
    deregister_llm_sanitize_response_guardrail, register_llm_conditional_execution_guardrail,
    register_llm_execution_intercept, register_llm_execution_intercept_v2,
    register_llm_request_intercept, register_llm_sanitize_request_guardrail,
    register_llm_sanitize_response_guardrail, scope_register_llm_execution_intercept,
    scope_register_llm_execution_intercept_v2,
};
use nemo_relay::api::runtime::{
    LLM_REPLAY_CONTRACT_VERSION, LlmExecutionNextFn, LlmReplayCall, LlmReplayCapability,
    LlmReplayFactory, LlmReplayTransport, NemoRelayContextState, create_scope_stack,
    global_context, set_thread_scope_stack,
};
use nemo_relay::api::scope::{
    PopScopeParams, PushScopeParams, ScopeHandle, ScopeType, pop_scope, push_scope,
};
use nemo_relay::api::subscriber::deregister_subscriber;
use nemo_relay::api::subscriber::{flush_subscribers, register_subscriber};
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::error::{FlowError, Result};
use nemo_relay::json::Json;
use nemo_relay::observability::atof::{AtofExporter, AtofExporterConfig, AtofExporterMode};
use serde_json::{Map, json};
use uuid::Uuid;

static TEST_MUTEX: Mutex<()> = Mutex::new(());

type TestLlmFuture = Pin<Box<dyn Future<Output = Result<Json>> + Send>>;

fn reset_runtime() {
    *global_context().write().unwrap() = NemoRelayContextState::new();
    set_thread_scope_stack(create_scope_stack());
}

fn push(name: &str, scope_type: ScopeType) -> ScopeHandle {
    push_scope(
        PushScopeParams::builder()
            .name(name)
            .scope_type(scope_type)
            .build(),
    )
    .unwrap()
}

fn request(value: &str) -> LlmRequest {
    LlmRequest {
        headers: Map::new(),
        content: json!({"value": value}),
    }
}

fn provider(value: Json) -> LlmExecutionNextFn {
    Arc::new(move |_| {
        let value = value.clone();
        Box::pin(async move { Ok(value) })
    })
}

#[derive(Clone, Copy)]
enum FactoryBehavior {
    Valid,
    Error,
    Panic,
}

struct TestFactory {
    calls: Arc<AtomicUsize>,
    transport: Arc<dyn LlmReplayTransport>,
    behavior: FactoryBehavior,
}

impl LlmReplayFactory for TestFactory {
    fn build(&self, _context: &LlmExecutionContextSnapshot) -> Result<Arc<dyn LlmReplayTransport>> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        match self.behavior {
            FactoryBehavior::Valid => Ok(self.transport.clone()),
            FactoryBehavior::Error => Err(FlowError::Internal(
                "nvapi-secret-that-must-not-reach-events".to_string(),
            )),
            FactoryBehavior::Panic => panic!("factory secret"),
        }
    }
}

struct TestTransport {
    capability: LlmReplayCapability,
    starts: Arc<AtomicUsize>,
    cancellations: Arc<AtomicUsize>,
    delay: Duration,
    fail: bool,
}

impl TestTransport {
    fn valid(family: LlmApiFamily) -> Arc<Self> {
        Arc::new(Self {
            capability: LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: family,
                transport_identity: "gateway-A".to_string(),
            },
            starts: Arc::new(AtomicUsize::new(0)),
            cancellations: Arc::new(AtomicUsize::new(0)),
            delay: Duration::ZERO,
            fail: false,
        })
    }
}

impl LlmReplayTransport for TestTransport {
    fn capability(&self) -> &LlmReplayCapability {
        &self.capability
    }

    fn start(&self, request: LlmRequest) -> Result<LlmReplayCall> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        let delay = self.delay;
        let fail = self.fail;
        let cancellations = self.cancellations.clone();
        Ok(LlmReplayCall::new(
            async move {
                if !delay.is_zero() {
                    tokio::time::sleep(delay).await;
                }
                if fail {
                    Err(FlowError::Internal("replay failed".to_string()))
                } else {
                    Ok(request.content)
                }
            },
            move || {
                cancellations.fetch_add(1, Ordering::SeqCst);
            },
        ))
    }
}

struct PanicCapabilityFactory;

impl LlmReplayFactory for PanicCapabilityFactory {
    fn build(&self, _context: &LlmExecutionContextSnapshot) -> Result<Arc<dyn LlmReplayTransport>> {
        Ok(Arc::new(PanicCapabilityTransport))
    }
}

struct PanicCapabilityTransport;

impl LlmReplayTransport for PanicCapabilityTransport {
    fn capability(&self) -> &LlmReplayCapability {
        panic!("capability secret")
    }

    fn start(&self, _request: LlmRequest) -> Result<LlmReplayCall> {
        panic!("invalid capability must never reach replay start")
    }
}

struct ImmediateStartErrorTransport {
    capability: LlmReplayCapability,
    starts: AtomicUsize,
}

impl ImmediateStartErrorTransport {
    fn new() -> Self {
        Self {
            capability: LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: LlmApiFamily::OpenAIResponses,
                transport_identity: "gateway-start-error".to_string(),
            },
            starts: AtomicUsize::new(0),
        }
    }
}

impl LlmReplayTransport for ImmediateStartErrorTransport {
    fn capability(&self) -> &LlmReplayCapability {
        &self.capability
    }

    fn start(&self, _request: LlmRequest) -> Result<LlmReplayCall> {
        self.starts.fetch_add(1, Ordering::SeqCst);
        Err(FlowError::InvalidArgument(
            "synchronous replay start rejection".to_string(),
        ))
    }
}

fn v2_params(
    name: &str,
    func: LlmExecutionNextFn,
    parent: ScopeHandle,
    replay_factory: Option<Arc<dyn LlmReplayFactory>>,
) -> LlmCallExecuteV2Params {
    LlmCallExecuteV2Params::builder()
        .name(name)
        .request(request(name))
        .func(func)
        .api_family(LlmApiFamily::OpenAIResponses)
        .call_role(LlmCallRole::Primary)
        .sanitized_metadata(BTreeMap::new())
        .parent(parent)
        .replay_factory_opt(replay_factory)
        .build()
}

#[tokio::test]
async fn mixed_registry_preserves_order_context_and_v1_behavior() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let order = Arc::new(Mutex::new(Vec::new()));
    let v2_context = Arc::new(Mutex::new(None));
    let local_v2_context = Arc::new(Mutex::new(None));

    let v1_order = order.clone();
    register_llm_execution_intercept(
        "legacy",
        20,
        Arc::new(move |_, request, next| {
            let order = v1_order.clone();
            Box::pin(async move {
                order.lock().unwrap().push("v1");
                next(request).await
            })
        }),
    )
    .unwrap();

    let v2_order = order.clone();
    let captured = v2_context.clone();
    register_llm_execution_intercept_v2(
        "context",
        10,
        Arc::new(move |_, context, request, replay, next| {
            let order = v2_order.clone();
            let captured = captured.clone();
            Box::pin(async move {
                order.lock().unwrap().push("v2");
                assert!(replay.is_some());
                *captured.lock().unwrap() = Some(context);
                next(request).await
            })
        }),
    )
    .unwrap();
    let global_tie_order = order.clone();
    register_llm_execution_intercept(
        "z-global-tie",
        12,
        Arc::new(move |_, request, next| {
            let order = global_tie_order.clone();
            Box::pin(async move {
                order.lock().unwrap().push("z-global-tie");
                next(request).await
            })
        }),
    )
    .unwrap();
    let local_tie_order = order.clone();
    let local_captured = local_v2_context.clone();
    scope_register_llm_execution_intercept_v2(
        &agent.uuid,
        "a-local-tie",
        12,
        Arc::new(move |_, context, request, _, next| {
            let order = local_tie_order.clone();
            *local_captured.lock().unwrap() = Some(context);
            Box::pin(async move {
                order.lock().unwrap().push("a-local-tie");
                next(request).await
            })
        }),
    )
    .unwrap();
    let local_order = order.clone();
    scope_register_llm_execution_intercept_v2(
        &agent.uuid,
        "local-context",
        15,
        Arc::new(move |_, _, request, _, next| {
            let order = local_order.clone();
            Box::pin(async move {
                order.lock().unwrap().push("local-v2");
                next(request).await
            })
        }),
    )
    .unwrap();
    assert!(
        scope_register_llm_execution_intercept(
            &agent.uuid,
            "local-context",
            30,
            Arc::new(|_, request, next| next(request)),
        )
        .is_err()
    );
    assert!(
        register_llm_execution_intercept("context", 30, Arc::new(|_, request, next| next(request)))
            .is_err()
    );

    let transport = TestTransport::valid(LlmApiFamily::OpenAIResponses);
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let factory: Arc<dyn LlmReplayFactory> = Arc::new(TestFactory {
        calls: factory_calls.clone(),
        transport,
        behavior: FactoryBehavior::Valid,
    });
    let provider_order = order.clone();
    let response = llm_call_execute_v2(v2_params(
        "anchor",
        Arc::new(move |_| {
            let order = provider_order.clone();
            Box::pin(async move {
                order.lock().unwrap().push("provider");
                Ok(json!("anchor"))
            })
        }),
        agent.clone(),
        Some(factory),
    ))
    .await
    .unwrap();
    assert_eq!(response, json!("anchor"));
    assert_eq!(
        *order.lock().unwrap(),
        vec![
            "v2",
            "a-local-tie",
            "z-global-tie",
            "local-v2",
            "v1",
            "provider"
        ]
    );
    assert_eq!(factory_calls.load(Ordering::SeqCst), 1);
    let context = v2_context.lock().unwrap().clone().unwrap();
    assert_eq!(context.api_family, LlmApiFamily::OpenAIResponses);
    assert_eq!(context.call_role, LlmCallRole::Primary);
    assert_eq!(context.trajectory_owner_uuid, agent.uuid);
    let local_context = local_v2_context.lock().unwrap().clone().unwrap();
    assert!(Arc::ptr_eq(&context, &local_context));

    order.lock().unwrap().clear();
    llm_call_execute(
        LlmCallExecuteParams::builder()
            .name("legacy-call")
            .request(request("legacy"))
            .func(provider(json!("legacy")))
            .parent(agent.clone())
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(*order.lock().unwrap(), vec!["z-global-tie", "v1"]);

    deregister_llm_execution_intercept("context").unwrap();
    deregister_llm_execution_intercept("z-global-tie").unwrap();
    deregister_llm_execution_intercept("legacy").unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn scope_pop_releases_v2_execution_intercept() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let callback_owner = Arc::new(());
    let callback_owner_weak = Arc::downgrade(&callback_owner);
    let captured_owner = callback_owner.clone();
    drop(callback_owner);

    scope_register_llm_execution_intercept_v2(
        &agent.uuid,
        "scope-owned-v2",
        0,
        Arc::new(move |_, _, request, _, next| {
            let _ = &captured_owner;
            next(request)
        }),
    )
    .unwrap();
    assert!(callback_owner_weak.upgrade().is_some());

    assert_eq!(
        llm_call_execute_v2(v2_params(
            "scope-owned-v2-call",
            provider(json!("ok")),
            agent.clone(),
            None,
        ))
        .await
        .unwrap(),
        json!("ok")
    );
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();

    assert!(
        callback_owner_weak.upgrade().is_none(),
        "popping a scope retained its V2 execution intercept"
    );
}

#[tokio::test]
async fn v2_preserves_middleware_codec_sanitizer_and_pending_mark_order() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let order = Arc::new(Mutex::new(Vec::new()));
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let event_sink = events.clone();
    register_subscriber(
        "v2-pipeline-events",
        Arc::new(move |event| event_sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let guardrail_order = order.clone();
    register_llm_conditional_execution_guardrail(
        "v2-pipeline-guardrail",
        0,
        Arc::new(move |_| {
            guardrail_order.lock().unwrap().push("guardrail");
            Ok(None)
        }),
    )
    .unwrap();
    let request_order = order.clone();
    register_llm_request_intercept(
        "v2-pipeline-request",
        0,
        false,
        Arc::new(move |_, mut request, annotated| {
            request_order.lock().unwrap().push("request_intercept");
            request.headers.insert("x-v2-intercept".into(), json!(true));
            Ok(
                LlmRequestInterceptOutcome::new(request, annotated).with_pending_mark(
                    PendingMarkSpec::builder()
                        .name("v2.pipeline.pending")
                        .build(),
                ),
            )
        }),
    )
    .unwrap();
    let sanitize_request_order = order.clone();
    register_llm_sanitize_request_guardrail(
        "v2-pipeline-sanitize-request",
        0,
        Arc::new(move |mut request| {
            sanitize_request_order
                .lock()
                .unwrap()
                .push("sanitize_request");
            request.content["messages"][0]["content"] = json!("SANITIZED_REQUEST");
            request
        }),
    )
    .unwrap();
    let execution_order = order.clone();
    register_llm_execution_intercept_v2(
        "v2-pipeline-execution",
        0,
        Arc::new(move |_, _, request, _, next| {
            execution_order.lock().unwrap().push("execution_intercept");
            assert_eq!(request.headers["x-v2-intercept"], true);
            next(request)
        }),
    )
    .unwrap();
    let sanitize_response_order = order.clone();
    register_llm_sanitize_response_guardrail(
        "v2-pipeline-sanitize-response",
        0,
        Arc::new(move |mut response| {
            sanitize_response_order
                .lock()
                .unwrap()
                .push("sanitize_response");
            response["choices"][0]["message"]["content"] = json!("SANITIZED_RESPONSE");
            response
        }),
    )
    .unwrap();

    let raw_response = json!({
        "id": "chatcmpl-v2",
        "object": "chat.completion",
        "created": 1,
        "model": "test-model",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "SECRET_RESPONSE"},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
    });
    let provider_order = order.clone();
    let provider_response = raw_response.clone();
    let codec: Arc<dyn LlmCodec> = Arc::new(OpenAIChatCodec);
    let response_codec: Arc<dyn LlmResponseCodec> = Arc::new(OpenAIChatCodec);
    let response = llm_call_execute_v2(
        LlmCallExecuteV2Params::builder()
            .name("v2-pipeline")
            .request(LlmRequest {
                headers: Map::new(),
                content: json!({
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "SECRET_REQUEST"}]
                }),
            })
            .func(Arc::new(move |request| {
                provider_order.lock().unwrap().push("provider");
                assert_eq!(request.content["messages"][0]["content"], "SECRET_REQUEST");
                let response = provider_response.clone();
                Box::pin(async move { Ok(response) })
            }))
            .api_family(LlmApiFamily::OpenAIChatCompletions)
            .call_role(LlmCallRole::Primary)
            .sanitized_metadata(BTreeMap::new())
            .parent(agent.clone())
            .codec(codec)
            .response_codec(response_codec)
            .build(),
    )
    .await
    .unwrap();
    assert_eq!(response, raw_response);
    assert_eq!(
        *order.lock().unwrap(),
        [
            "guardrail",
            "request_intercept",
            "sanitize_request",
            "execution_intercept",
            "provider",
            "sanitize_response",
        ]
    );

    flush_subscribers().unwrap();
    let captured = events.lock().unwrap();
    let start_index = captured
        .iter()
        .position(|event| {
            event.name() == "v2-pipeline" && event.scope_category() == Some(ScopeCategory::Start)
        })
        .unwrap();
    let mark_index = captured
        .iter()
        .position(|event| event.name() == "v2.pipeline.pending")
        .unwrap();
    let end_index = captured
        .iter()
        .position(|event| {
            event.name() == "v2-pipeline" && event.scope_category() == Some(ScopeCategory::End)
        })
        .unwrap();
    assert!(start_index < mark_index && mark_index < end_index);
    let start = &captured[start_index];
    let end = &captured[end_index];
    assert_eq!(
        start.input().unwrap()["content"]["messages"][0]["content"],
        "SANITIZED_REQUEST"
    );
    assert!(start.annotated_request().is_some());
    assert_eq!(
        end.output().unwrap()["choices"][0]["message"]["content"],
        "SANITIZED_RESPONSE"
    );
    assert_eq!(
        end.annotated_response().unwrap().response_text(),
        Some("SANITIZED_RESPONSE")
    );
    drop(captured);

    deregister_llm_conditional_execution_guardrail("v2-pipeline-guardrail").unwrap();
    deregister_llm_request_intercept("v2-pipeline-request").unwrap();
    deregister_llm_sanitize_request_guardrail("v2-pipeline-sanitize-request").unwrap();
    deregister_llm_execution_intercept("v2-pipeline-execution").unwrap();
    deregister_llm_sanitize_response_guardrail("v2-pipeline-sanitize-response").unwrap();
    deregister_subscriber("v2-pipeline-events").unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn v2_next_is_single_use_and_expires_after_callback() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let retained = Arc::new(Mutex::new(None::<LlmExecutionNextFn>));
    let retained_from_callback = retained.clone();
    register_llm_execution_intercept_v2(
        "one-shot",
        0,
        Arc::new(move |_, _, request, _, next| {
            let retained = retained_from_callback.clone();
            Box::pin(async move {
                *retained.lock().unwrap() = Some(next.clone());
                let first = next(request.clone());
                let second = next(request);
                assert!(second.await.is_err());
                first.await
            })
        }),
    )
    .unwrap();

    let provider_owner = Arc::new(());
    let provider_weak = Arc::downgrade(&provider_owner);
    let captured_provider_owner = provider_owner.clone();
    drop(provider_owner);
    let provider: LlmExecutionNextFn = Arc::new(move |_| {
        let _ = &captured_provider_owner;
        Box::pin(async { Ok(json!("ok")) })
    });

    let result = llm_call_execute_v2(v2_params("one-shot-call", provider, agent.clone(), None))
        .await
        .unwrap();
    assert_eq!(result, json!("ok"));
    assert!(
        provider_weak.upgrade().is_none(),
        "retained V2 next kept the provider alive after its lease ended"
    );
    let expired = retained.lock().unwrap().take().unwrap();
    assert!(expired(request("late")).await.is_err());

    deregister_llm_execution_intercept("one-shot").unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn unpolled_v2_next_future_does_not_construct_or_retain_downstream() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let retained = Arc::new(Mutex::new(None::<TestLlmFuture>));
    let retained_from_callback = retained.clone();
    register_llm_execution_intercept_v2(
        "retain-unpolled-v2-next",
        0,
        Arc::new(move |_, _, request, _, next| {
            let retained = retained_from_callback.clone();
            Box::pin(async move {
                *retained.lock().unwrap() = Some(next(request));
                Ok(json!("replacement"))
            })
        }),
    )
    .unwrap();

    let provider_calls = Arc::new(AtomicUsize::new(0));
    let calls = provider_calls.clone();
    let provider_owner = Arc::new(());
    let provider_weak = Arc::downgrade(&provider_owner);
    let captured_provider_owner = provider_owner.clone();
    drop(provider_owner);
    let result = llm_call_execute_v2(v2_params(
        "retain-unpolled-v2-next-call",
        Arc::new(move |_| {
            calls.fetch_add(1, Ordering::SeqCst);
            let owner = captured_provider_owner.clone();
            Box::pin(async move {
                let _ = owner;
                Ok(json!("provider"))
            })
        }),
        agent.clone(),
        None,
    ))
    .await
    .unwrap();

    assert_eq!(result, json!("replacement"));
    assert_eq!(provider_calls.load(Ordering::SeqCst), 0);
    assert!(provider_weak.upgrade().is_none());
    let error = retained.lock().unwrap().take().unwrap().await.unwrap_err();
    assert!(
        matches!(error, FlowError::InvalidArgument(message) if message == "V2 LLM continuation is no longer active")
    );
    assert_eq!(provider_calls.load(Ordering::SeqCst), 0);

    deregister_llm_execution_intercept("retain-unpolled-v2-next").unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn downstream_future_cannot_outlive_its_v2_intercept() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let spawned = Arc::new(Mutex::new(None));
    let spawned_from_callback = spawned.clone();
    register_llm_execution_intercept_v2(
        "no-delayed-next",
        0,
        Arc::new(move |_, _, request, _, next| {
            let spawned = spawned_from_callback.clone();
            Box::pin(async move {
                *spawned.lock().unwrap() = Some(tokio::spawn(next(request)));
                Ok(json!("replacement"))
            })
        }),
    )
    .unwrap();

    let result = llm_call_execute_v2(v2_params(
        "no-delayed-next-call",
        Arc::new(|_| Box::pin(std::future::pending())),
        agent.clone(),
        None,
    ))
    .await
    .unwrap();
    assert_eq!(result, json!("replacement"));
    let delayed = spawned.lock().unwrap().take().unwrap();
    assert!(delayed.await.unwrap().is_err());

    deregister_llm_execution_intercept("no-delayed-next").unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn invalidation_wins_simultaneous_downstream_completion_for_v1_and_v2() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);

    for (use_v2, intercept_name, call_name) in [
        (true, "simultaneous-v2", "simultaneous-v2-call"),
        (false, "simultaneous-v1", "simultaneous-v1-call"),
    ] {
        let provider_started = Arc::new(tokio::sync::Notify::new());
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let result_tx = Arc::new(Mutex::new(Some(result_tx)));
        let spawned = Arc::new(Mutex::new(None));

        if use_v2 {
            let started = provider_started.clone();
            let sender = result_tx.clone();
            let spawned_from_callback = spawned.clone();
            register_llm_execution_intercept_v2(
                intercept_name,
                0,
                Arc::new(move |_, _, request, _, next| {
                    let started = started.clone();
                    let sender = sender.clone();
                    let spawned = spawned_from_callback.clone();
                    Box::pin(async move {
                        *spawned.lock().unwrap() = Some(tokio::spawn(next(request)));
                        started.notified().await;
                        sender
                            .lock()
                            .unwrap()
                            .take()
                            .unwrap()
                            .send(json!("provider"))
                            .unwrap();
                        Ok(json!("replacement"))
                    })
                }),
            )
            .unwrap();
        } else {
            let started = provider_started.clone();
            let sender = result_tx.clone();
            let spawned_from_callback = spawned.clone();
            register_llm_execution_intercept(
                intercept_name,
                0,
                Arc::new(move |_, request, next| {
                    let started = started.clone();
                    let sender = sender.clone();
                    let spawned = spawned_from_callback.clone();
                    Box::pin(async move {
                        *spawned.lock().unwrap() = Some(tokio::spawn(next(request)));
                        started.notified().await;
                        sender
                            .lock()
                            .unwrap()
                            .take()
                            .unwrap()
                            .send(json!("provider"))
                            .unwrap();
                        Ok(json!("replacement"))
                    })
                }),
            )
            .unwrap();
        }

        let receiver = Arc::new(Mutex::new(Some(result_rx)));
        let provider_receiver = receiver.clone();
        let started = provider_started.clone();
        let response = llm_call_execute_v2(v2_params(
            call_name,
            Arc::new(move |_| {
                let receiver = provider_receiver.lock().unwrap().take().unwrap();
                let started = started.clone();
                Box::pin(async move {
                    started.notify_one();
                    Ok(receiver.await.unwrap())
                })
            }),
            agent.clone(),
            None,
        ))
        .await
        .unwrap();
        assert_eq!(response, json!("replacement"));

        let error = spawned
            .lock()
            .unwrap()
            .take()
            .unwrap()
            .await
            .unwrap()
            .unwrap_err();
        assert!(
            matches!(error, FlowError::InvalidArgument(message) if message == "V2 LLM continuation is no longer active")
        );
        deregister_llm_execution_intercept(intercept_name).unwrap();
    }

    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn adapted_v1_intercept_can_retry_only_while_active() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let provider_calls = Arc::new(AtomicUsize::new(0));
    let retained = Arc::new(Mutex::new(None::<LlmExecutionNextFn>));
    let retained_from_callback = retained.clone();
    register_llm_execution_intercept(
        "retry",
        0,
        Arc::new(move |_, request, next| {
            *retained_from_callback.lock().unwrap() = Some(next.clone());
            Box::pin(async move {
                let _ = next(request.clone()).await?;
                next(request).await
            })
        }),
    )
    .unwrap();
    let calls = provider_calls.clone();
    let provider_owner = Arc::new(());
    let provider_weak = Arc::downgrade(&provider_owner);
    let captured_provider_owner = provider_owner.clone();
    drop(provider_owner);
    llm_call_execute_v2(v2_params(
        "retry-call",
        Arc::new(move |_| {
            let _ = &captured_provider_owner;
            let calls = calls.clone();
            Box::pin(async move {
                calls.fetch_add(1, Ordering::SeqCst);
                Ok(json!("ok"))
            })
        }),
        agent.clone(),
        None,
    ))
    .await
    .unwrap();
    assert_eq!(provider_calls.load(Ordering::SeqCst), 2);
    assert!(
        provider_weak.upgrade().is_none(),
        "retained adapted V1 next kept the provider alive after its lease ended"
    );
    let expired = retained.lock().unwrap().take().unwrap();
    assert!(expired(request("late-v1")).await.is_err());
    deregister_llm_execution_intercept("retry").unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn unpolled_adapted_v1_next_future_does_not_construct_or_retain_downstream() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let retained = Arc::new(Mutex::new(None::<TestLlmFuture>));
    let retained_from_callback = retained.clone();
    register_llm_execution_intercept(
        "retain-unpolled-v1-next",
        0,
        Arc::new(move |_, request, next| {
            let retained = retained_from_callback.clone();
            Box::pin(async move {
                *retained.lock().unwrap() = Some(next(request));
                Ok(json!("replacement"))
            })
        }),
    )
    .unwrap();

    let provider_calls = Arc::new(AtomicUsize::new(0));
    let calls = provider_calls.clone();
    let provider_owner = Arc::new(());
    let provider_weak = Arc::downgrade(&provider_owner);
    let captured_provider_owner = provider_owner.clone();
    drop(provider_owner);
    let result = llm_call_execute_v2(v2_params(
        "retain-unpolled-v1-next-call",
        Arc::new(move |_| {
            calls.fetch_add(1, Ordering::SeqCst);
            let owner = captured_provider_owner.clone();
            Box::pin(async move {
                let _ = owner;
                Ok(json!("provider"))
            })
        }),
        agent.clone(),
        None,
    ))
    .await
    .unwrap();

    assert_eq!(result, json!("replacement"));
    assert_eq!(provider_calls.load(Ordering::SeqCst), 0);
    assert!(provider_weak.upgrade().is_none());
    let error = retained.lock().unwrap().take().unwrap().await.unwrap_err();
    assert!(
        matches!(error, FlowError::InvalidArgument(message) if message == "V2 LLM continuation is no longer active")
    );
    assert_eq!(provider_calls.load(Ordering::SeqCst), 0);

    deregister_llm_execution_intercept("retain-unpolled-v1-next").unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn factory_failures_and_invalid_capabilities_fail_open_without_secret_diagnostics() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let sink = events.clone();
    register_subscriber(
        "replay-diagnostics",
        Arc::new(move |event| sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    let saw_replay = Arc::new(AtomicBool::new(false));
    let saw = saw_replay.clone();
    register_llm_execution_intercept_v2(
        "observe-replay",
        0,
        Arc::new(move |_, _, request, replay, next| {
            saw.store(replay.is_some(), Ordering::SeqCst);
            next(request)
        }),
    )
    .unwrap();

    let transport = TestTransport::valid(LlmApiFamily::OpenAIResponses);
    let factory: Arc<dyn LlmReplayFactory> = Arc::new(TestFactory {
        calls: Arc::new(AtomicUsize::new(0)),
        transport,
        behavior: FactoryBehavior::Error,
    });
    assert_eq!(
        llm_call_execute_v2(v2_params(
            "factory-error",
            provider(json!("anchor")),
            agent.clone(),
            Some(factory),
        ))
        .await
        .unwrap(),
        json!("anchor")
    );
    flush_subscribers().unwrap();
    assert!(!saw_replay.load(Ordering::SeqCst));
    let encoded = serde_json::to_string(&*events.lock().unwrap()).unwrap();
    assert!(encoded.contains("factory_error"));
    assert!(!encoded.contains("nvapi-secret"));
    let diagnostic = events
        .lock()
        .unwrap()
        .iter()
        .find(|event| event.name() == "nemo_relay.replay_ineligible")
        .cloned()
        .unwrap();
    assert!(diagnostic.data().is_none());
    let metadata = diagnostic.metadata().unwrap().as_object().unwrap();
    assert_eq!(metadata.len(), 2);
    assert!(metadata["call_uuid"].is_string());
    assert_eq!(metadata["reason"], "factory_error");

    events.lock().unwrap().clear();
    let invalid = Arc::new(TestTransport {
        capability: LlmReplayCapability {
            contract_version: 99,
            api_family: LlmApiFamily::OpenAIResponses,
            transport_identity: "gateway-A".to_string(),
        },
        starts: Arc::new(AtomicUsize::new(0)),
        cancellations: Arc::new(AtomicUsize::new(0)),
        delay: Duration::ZERO,
        fail: false,
    });
    let factory: Arc<dyn LlmReplayFactory> = Arc::new(TestFactory {
        calls: Arc::new(AtomicUsize::new(0)),
        transport: invalid,
        behavior: FactoryBehavior::Valid,
    });
    llm_call_execute_v2(v2_params(
        "bad-version",
        provider(json!("anchor")),
        agent.clone(),
        Some(factory),
    ))
    .await
    .unwrap();
    flush_subscribers().unwrap();
    assert!(
        serde_json::to_string(&*events.lock().unwrap())
            .unwrap()
            .contains("unsupported_contract")
    );

    for (capability, expected_reason) in [
        (
            LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: LlmApiFamily::AnthropicMessages,
                transport_identity: "gateway-A".to_string(),
            },
            "family_mismatch",
        ),
        (
            LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: LlmApiFamily::OpenAIResponses,
                transport_identity: "nvapi-abcdefghijklmnopqrstuvwxyz0123456789".to_string(),
            },
            "invalid_transport_identity",
        ),
    ] {
        events.lock().unwrap().clear();
        let invalid = Arc::new(TestTransport {
            capability,
            starts: Arc::new(AtomicUsize::new(0)),
            cancellations: Arc::new(AtomicUsize::new(0)),
            delay: Duration::ZERO,
            fail: false,
        });
        let factory: Arc<dyn LlmReplayFactory> = Arc::new(TestFactory {
            calls: Arc::new(AtomicUsize::new(0)),
            transport: invalid,
            behavior: FactoryBehavior::Valid,
        });
        llm_call_execute_v2(v2_params(
            expected_reason,
            provider(json!("anchor")),
            agent.clone(),
            Some(factory),
        ))
        .await
        .unwrap();
        flush_subscribers().unwrap();
        assert!(
            serde_json::to_string(&*events.lock().unwrap())
                .unwrap()
                .contains(expected_reason)
        );
    }

    events.lock().unwrap().clear();
    let factory: Arc<dyn LlmReplayFactory> = Arc::new(TestFactory {
        calls: Arc::new(AtomicUsize::new(0)),
        transport: TestTransport::valid(LlmApiFamily::OpenAIResponses),
        behavior: FactoryBehavior::Panic,
    });
    assert_eq!(
        llm_call_execute_v2(v2_params(
            "factory-panic",
            provider(json!("anchor")),
            agent.clone(),
            Some(factory),
        ))
        .await
        .unwrap(),
        json!("anchor")
    );
    flush_subscribers().unwrap();
    assert!(
        serde_json::to_string(&*events.lock().unwrap())
            .unwrap()
            .contains("factory_panic")
    );

    events.lock().unwrap().clear();
    assert_eq!(
        llm_call_execute_v2(v2_params(
            "capability-panic",
            provider(json!("anchor")),
            agent.clone(),
            Some(Arc::new(PanicCapabilityFactory)),
        ))
        .await
        .unwrap(),
        json!("anchor")
    );
    flush_subscribers().unwrap();
    let encoded = serde_json::to_string(&*events.lock().unwrap()).unwrap();
    assert!(encoded.contains("capability_panic"));
    assert!(!encoded.contains("capability secret"));

    deregister_llm_execution_intercept("observe-replay").unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn streaming_and_stateful_v2_calls_do_not_build_replay() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let factory: Arc<dyn LlmReplayFactory> = Arc::new(TestFactory {
        calls: factory_calls.clone(),
        transport: TestTransport::valid(LlmApiFamily::OpenAIResponses),
        behavior: FactoryBehavior::Valid,
    });

    for attributes in [LlmAttributes::STREAMING, LlmAttributes::STATEFUL] {
        let mut params = v2_params(
            "ineligible",
            provider(json!("anchor")),
            agent.clone(),
            Some(factory.clone()),
        );
        params.attributes = attributes;
        assert_eq!(llm_call_execute_v2(params).await.unwrap(), json!("anchor"));
    }
    assert_eq!(factory_calls.load(Ordering::SeqCst), 0);
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn internal_roles_require_evaluator_parent_and_anchor_uuid() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let provider_called = Arc::new(AtomicBool::new(false));
    let called = provider_called.clone();
    let invalid = LlmCallExecuteV2Params::builder()
        .name("invalid-shadow")
        .request(request("shadow"))
        .func(Arc::new(move |_| {
            called.store(true, Ordering::SeqCst);
            Box::pin(async { Ok(json!("unexpected")) })
        }))
        .api_family(LlmApiFamily::OpenAIResponses)
        .call_role(LlmCallRole::Shadow)
        .sanitized_metadata(BTreeMap::new())
        .parent(agent.clone())
        .build();
    assert!(llm_call_execute_v2(invalid).await.is_err());
    assert!(!provider_called.load(Ordering::SeqCst));

    let evaluator = push("evaluator", ScopeType::Evaluator);
    let anchor_uuid = Uuid::now_v7();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let sink = events.clone();
    register_subscriber(
        "internal-events",
        Arc::new(move |event| sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    let valid = LlmCallExecuteV2Params::builder()
        .name("shadow")
        .request(request("shadow"))
        .func(provider(json!("shadow-result")))
        .api_family(LlmApiFamily::OpenAIResponses)
        .call_role(LlmCallRole::Shadow)
        .sanitized_metadata(BTreeMap::from([(
            "anchor_uuid".to_string(),
            json!(anchor_uuid.to_string()),
        )]))
        .parent(evaluator.clone())
        .build();
    assert_eq!(
        llm_call_execute_v2(valid).await.unwrap(),
        json!("shadow-result")
    );
    flush_subscribers().unwrap();
    let shadow_lifecycle: Vec<_> = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.name() == "shadow" && event.scope_category().is_some())
        .cloned()
        .collect();
    assert_eq!(shadow_lifecycle.len(), 2);
    assert_eq!(
        shadow_lifecycle[0].llm_call_role(),
        Some(LlmCallRole::Shadow)
    );
    assert_eq!(
        shadow_lifecycle[1].llm_call_role(),
        Some(LlmCallRole::Shadow)
    );
    assert_eq!(shadow_lifecycle[0].uuid(), shadow_lifecycle[1].uuid());
    assert_eq!(
        shadow_lifecycle[0].scope_category(),
        Some(ScopeCategory::Start)
    );
    assert_eq!(
        shadow_lifecycle[1].scope_category(),
        Some(ScopeCategory::End)
    );

    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&evaluator.uuid)
            .build(),
    )
    .unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn internal_roles_exclude_recursive_replay_with_authoritative_lifecycle() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let evaluator = push("evaluator", ScopeType::Evaluator);
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let sink = events.clone();
    register_subscriber(
        "internal-recursion-events",
        Arc::new(move |event| sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    let replay_visibility = Arc::new(Mutex::new(Vec::new()));
    let visibility = replay_visibility.clone();
    register_llm_execution_intercept_v2(
        "observe-internal-recursion-replay",
        0,
        Arc::new(move |name, _, request, replay, next| {
            visibility
                .lock()
                .unwrap()
                .push((name.to_string(), replay.is_none()));
            next(request)
        }),
    )
    .unwrap();
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let factory: Arc<dyn LlmReplayFactory> = Arc::new(TestFactory {
        calls: factory_calls.clone(),
        transport: TestTransport::valid(LlmApiFamily::OpenAIResponses),
        behavior: FactoryBehavior::Valid,
    });
    let anchor_uuid = Uuid::now_v7();

    for (role, name) in [
        (LlmCallRole::Shadow, "recursive-shadow"),
        (LlmCallRole::Judge, "recursive-judge"),
    ] {
        let params = LlmCallExecuteV2Params::builder()
            .name(name)
            .request(request(name))
            .func(provider(json!({"role": name})))
            .api_family(LlmApiFamily::OpenAIResponses)
            .call_role(role)
            .sanitized_metadata(BTreeMap::from([(
                "anchor_uuid".to_string(),
                json!(anchor_uuid.to_string()),
            )]))
            .parent(evaluator.clone())
            .replay_factory(factory.clone())
            .build();
        assert_eq!(
            llm_call_execute_v2(params).await.unwrap(),
            json!({"role": name})
        );
    }

    assert_eq!(factory_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        *replay_visibility.lock().unwrap(),
        [
            ("recursive-shadow".to_string(), true),
            ("recursive-judge".to_string(), true),
        ]
    );
    flush_subscribers().unwrap();
    let captured = events.lock().unwrap();
    for (role, name) in [
        (LlmCallRole::Shadow, "recursive-shadow"),
        (LlmCallRole::Judge, "recursive-judge"),
    ] {
        let lifecycle = captured
            .iter()
            .filter(|event| event.name() == name && event.scope_category().is_some())
            .collect::<Vec<_>>();
        assert_eq!(lifecycle.len(), 2);
        assert_eq!(lifecycle[0].scope_category(), Some(ScopeCategory::Start));
        assert_eq!(lifecycle[1].scope_category(), Some(ScopeCategory::End));
        assert_eq!(lifecycle[0].uuid(), lifecycle[1].uuid());
        assert_eq!(lifecycle[0].llm_call_role(), Some(role));
        assert_eq!(lifecycle[1].llm_call_role(), Some(role));

        let expected_call_uuid = lifecycle[0].uuid().to_string();
        let diagnostics = captured
            .iter()
            .filter(|event| {
                event.name() == "nemo_relay.replay_ineligible"
                    && event.metadata().is_some_and(|metadata| {
                        metadata["call_uuid"] == expected_call_uuid
                            && metadata["reason"] == "internal_role"
                    })
            })
            .collect::<Vec<_>>();
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].data().is_none());
        assert_eq!(
            diagnostics[0]
                .metadata()
                .unwrap()
                .as_object()
                .unwrap()
                .len(),
            2
        );
    }
    drop(captured);

    deregister_llm_execution_intercept("observe-internal-recursion-replay").unwrap();
    deregister_subscriber("internal-recursion-events").unwrap();
    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&evaluator.uuid)
            .build(),
    )
    .unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn managed_internal_call_encloses_one_replay_start_latency_and_error() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let sink = events.clone();
    register_subscriber(
        "managed-replay-events",
        Arc::new(move |event| sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let transport = Arc::new(TestTransport {
        capability: LlmReplayCapability {
            contract_version: LLM_REPLAY_CONTRACT_VERSION,
            api_family: LlmApiFamily::OpenAIResponses,
            transport_identity: "gateway-A".to_string(),
        },
        starts: Arc::new(AtomicUsize::new(0)),
        cancellations: Arc::new(AtomicUsize::new(0)),
        delay: Duration::from_millis(25),
        fail: true,
    });
    let factory: Arc<dyn LlmReplayFactory> = Arc::new(TestFactory {
        calls: Arc::new(AtomicUsize::new(0)),
        transport: transport.clone(),
        behavior: FactoryBehavior::Valid,
    });
    let captured = Arc::new(Mutex::new(
        None::<(
            Arc<LlmExecutionContextSnapshot>,
            Arc<dyn LlmReplayTransport>,
        )>,
    ));
    let captured_in_intercept = captured.clone();
    register_llm_execution_intercept_v2(
        "capture-anchor-replay",
        0,
        Arc::new(move |_, context, request, replay, next| {
            let captured = captured_in_intercept.clone();
            Box::pin(async move {
                *captured.lock().unwrap() = Some((context, replay.unwrap()));
                next(request).await
            })
        }),
    )
    .unwrap();
    llm_call_execute_v2(v2_params(
        "anchor-for-replay",
        provider(json!("anchor")),
        agent.clone(),
        Some(factory),
    ))
    .await
    .unwrap();
    deregister_llm_execution_intercept("capture-anchor-replay").unwrap();
    let (anchor_context, replay) = captured.lock().unwrap().take().unwrap();

    let evaluator = push("evaluator", ScopeType::Evaluator);
    let replay_provider = replay.clone();
    let shadow = LlmCallExecuteV2Params::builder()
        .name("managed-shadow-replay")
        .request(request("candidate"))
        .func(Arc::new(move |request| {
            let replay = replay_provider.clone();
            Box::pin(async move { replay.start(request)?.await })
        }))
        .api_family(LlmApiFamily::OpenAIResponses)
        .call_role(LlmCallRole::Shadow)
        .sanitized_metadata(BTreeMap::from([(
            "anchor_uuid".to_string(),
            json!(anchor_context.call_uuid.to_string()),
        )]))
        .parent(evaluator.clone())
        .build();
    assert!(llm_call_execute_v2(shadow).await.is_err());
    assert_eq!(transport.starts.load(Ordering::SeqCst), 1);
    assert_eq!(transport.cancellations.load(Ordering::SeqCst), 0);

    flush_subscribers().unwrap();
    let lifecycle: Vec<_> = events
        .lock()
        .unwrap()
        .iter()
        .filter(|event| event.name() == "managed-shadow-replay")
        .cloned()
        .collect();
    assert_eq!(lifecycle.len(), 2);
    assert_eq!(lifecycle[0].scope_category(), Some(ScopeCategory::Start));
    assert_eq!(lifecycle[1].scope_category(), Some(ScopeCategory::End));
    assert_eq!(lifecycle[0].llm_call_role(), Some(LlmCallRole::Shadow));
    assert_eq!(lifecycle[1].llm_call_role(), Some(LlmCallRole::Shadow));
    assert_eq!(lifecycle[0].uuid(), lifecycle[1].uuid());
    assert!(*lifecycle[1].timestamp() - *lifecycle[0].timestamp() >= TimeDelta::milliseconds(15));
    assert!(
        lifecycle[1]
            .metadata()
            .unwrap()
            .to_string()
            .contains("replay failed")
    );

    pop_scope(
        PopScopeParams::builder()
            .handle_uuid(&evaluator.uuid)
            .build(),
    )
    .unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn capability_and_host_policy_are_absent_from_durable_atof_json() {
    struct OpaquePolicyTransport {
        capability: LlmReplayCapability,
        endpoint_policy: String,
        auth_policy: String,
        anchor_response_status: Arc<AtomicUsize>,
        anchor_response_headers: Arc<Mutex<Vec<String>>>,
    }

    impl LlmReplayTransport for OpaquePolicyTransport {
        fn capability(&self) -> &LlmReplayCapability {
            &self.capability
        }

        fn start(&self, request: LlmRequest) -> Result<LlmReplayCall> {
            let _immutable_host_policy = (&self.endpoint_policy, &self.auth_policy);
            let _anchor_side_channels =
                (&self.anchor_response_status, &self.anchor_response_headers);
            Ok(LlmReplayCall::new(
                async move { Ok(request.content) },
                || {},
            ))
        }
    }

    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let output = tempfile::tempdir().unwrap();
    let exporter = AtofExporter::new(AtofExporterConfig {
        output_directory: output.path().to_path_buf(),
        mode: AtofExporterMode::Overwrite,
        filename: "replay-capability.jsonl".to_string(),
        endpoints: Vec::new(),
    })
    .unwrap();
    exporter.register("replay-capability-atof").unwrap();

    let anchor_response_status = Arc::new(AtomicUsize::new(207));
    let anchor_response_headers = Arc::new(Mutex::new(vec![
        "anchor-header-side-channel-secret".to_string(),
    ]));
    let transport: Arc<dyn LlmReplayTransport> = Arc::new(OpaquePolicyTransport {
        capability: LlmReplayCapability {
            contract_version: LLM_REPLAY_CONTRACT_VERSION,
            api_family: LlmApiFamily::OpenAIResponses,
            transport_identity: "gateway-atof-policy".to_string(),
        },
        endpoint_policy: "https://private-replay-endpoint.invalid".to_string(),
        auth_policy: "Bearer replay-auth-secret".to_string(),
        anchor_response_status: anchor_response_status.clone(),
        anchor_response_headers: anchor_response_headers.clone(),
    });
    let captured = Arc::new(Mutex::new(None::<Arc<dyn LlmReplayTransport>>));
    let captured_in_intercept = captured.clone();
    register_llm_execution_intercept_v2(
        "capture-atof-replay",
        0,
        Arc::new(move |_, _, request, replay, next| {
            *captured_in_intercept.lock().unwrap() = replay;
            next(request)
        }),
    )
    .unwrap();
    let factory: Arc<dyn LlmReplayFactory> = Arc::new(TestFactory {
        calls: Arc::new(AtomicUsize::new(0)),
        transport,
        behavior: FactoryBehavior::Valid,
    });
    llm_call_execute_v2(v2_params(
        "atof-replay-anchor",
        provider(json!("anchor")),
        agent.clone(),
        Some(factory),
    ))
    .await
    .unwrap();

    let replay = captured.lock().unwrap().take().unwrap();
    assert_eq!(
        replay.start(request("delayed")).unwrap().await.unwrap(),
        json!({"value": "delayed"})
    );
    assert_eq!(anchor_response_status.load(Ordering::SeqCst), 207);
    assert_eq!(
        *anchor_response_headers.lock().unwrap(),
        vec!["anchor-header-side-channel-secret"]
    );

    exporter.force_flush().unwrap();
    assert!(exporter.deregister("replay-capability-atof").unwrap());
    exporter.shutdown().unwrap();
    deregister_llm_execution_intercept("capture-atof-replay").unwrap();
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();

    let durable_json = std::fs::read_to_string(exporter.path()).unwrap();
    assert!(durable_json.contains("atof-replay-anchor"));
    for forbidden in [
        "contract_version",
        "api_family",
        "transport_identity",
        "gateway-atof-policy",
        "https://private-replay-endpoint.invalid",
        "Bearer replay-auth-secret",
        "anchor-header-side-channel-secret",
    ] {
        assert!(
            !durable_json.contains(forbidden),
            "durable ATOF exposed replay state: {forbidden}"
        );
    }
}

#[tokio::test]
async fn transport_supports_repeated_sequential_calls_after_completion() {
    let transport = TestTransport::valid(LlmApiFamily::OpenAIResponses);

    let first = transport.start(request("first")).unwrap();
    assert_eq!(transport.starts.load(Ordering::SeqCst), 1);
    assert_eq!(first.await.unwrap(), json!({"value": "first"}));

    let second = transport.start(request("second")).unwrap();
    assert_eq!(transport.starts.load(Ordering::SeqCst), 2);
    assert_eq!(second.await.unwrap(), json!({"value": "second"}));
    assert_eq!(transport.cancellations.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn synchronous_transport_start_error_propagates_exactly() {
    let _guard = TEST_MUTEX.lock().unwrap();
    reset_runtime();
    let agent = push("agent", ScopeType::Agent);
    let transport = Arc::new(ImmediateStartErrorTransport::new());
    let factory: Arc<dyn LlmReplayFactory> = Arc::new(TestFactory {
        calls: Arc::new(AtomicUsize::new(0)),
        transport: transport.clone(),
        behavior: FactoryBehavior::Valid,
    });
    let captured = Arc::new(Mutex::new(None::<Arc<dyn LlmReplayTransport>>));
    let captured_in_intercept = captured.clone();
    register_llm_execution_intercept_v2(
        "capture-start-error-replay",
        0,
        Arc::new(move |_, _, request, replay, next| {
            *captured_in_intercept.lock().unwrap() = replay;
            next(request)
        }),
    )
    .unwrap();
    llm_call_execute_v2(v2_params(
        "start-error-anchor",
        provider(json!("anchor")),
        agent.clone(),
        Some(factory),
    ))
    .await
    .unwrap();
    deregister_llm_execution_intercept("capture-start-error-replay").unwrap();

    let replay = captured.lock().unwrap().take().unwrap();
    let replay_provider: LlmExecutionNextFn = Arc::new(move |request| {
        let replay = replay.clone();
        Box::pin(async move { replay.start(request)?.await })
    });
    let error = llm_call_execute_v2(v2_params(
        "managed-start-error",
        replay_provider,
        agent.clone(),
        None,
    ))
    .await
    .unwrap_err();
    match error {
        FlowError::InvalidArgument(message) => {
            assert_eq!(message, "synchronous replay start rejection");
        }
        other => panic!("unexpected replay start error: {other}"),
    }
    assert_eq!(transport.starts.load(Ordering::SeqCst), 1);
    pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
}

#[tokio::test]
async fn transport_supports_repeated_concurrent_calls_with_sibling_isolation() {
    let transport = TestTransport::valid(LlmApiFamily::OpenAIResponses);
    let first = transport.start(request("first")).unwrap();
    let second = transport.start(request("second")).unwrap();
    let (first, second) = tokio::join!(first, second);
    assert_eq!(first.unwrap(), json!({"value": "first"}));
    assert_eq!(second.unwrap(), json!({"value": "second"}));
    assert_eq!(transport.starts.load(Ordering::SeqCst), 2);
    assert_eq!(transport.cancellations.load(Ordering::SeqCst), 0);

    let pending = Arc::new(TestTransport {
        capability: transport.capability.clone(),
        starts: Arc::new(AtomicUsize::new(0)),
        cancellations: Arc::new(AtomicUsize::new(0)),
        delay: Duration::from_millis(10),
        fail: false,
    });
    let cancelled = pending.start(request("cancelled")).unwrap();
    let sibling = pending.start(request("sibling")).unwrap();
    drop(cancelled);
    assert_eq!(sibling.await.unwrap(), json!({"value": "sibling"}));
    assert_eq!(pending.starts.load(Ordering::SeqCst), 2);
    assert_eq!(pending.cancellations.load(Ordering::SeqCst), 1);
}
