// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::alignment::GatewayRouteKind;
use crate::config::GatewayConfig;
use crate::server::AppState;
use crate::session::{LlmGatewayStart, SessionManager};
use axum::body::Body;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, Method, Request, StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use http_body_util::BodyExt;
use nemo_relay::api::registry::{
    deregister_llm_execution_intercept, register_llm_execution_intercept_v2,
};
use nemo_relay::api::scope::ScopeType;
use reqwest::Client;
use tokio::net::TcpListener;
use uuid::Uuid;

use crate::test_support::PLUGIN_CONFIG_TEST_LOCK;

fn test_http_client() -> Client {
    Client::new()
}

struct LlmExecutionInterceptGuard(&'static str);

impl Drop for LlmExecutionInterceptGuard {
    fn drop(&mut self) {
        let _ = deregister_llm_execution_intercept(self.0);
    }
}

fn replay_context(api_family: LlmApiFamily) -> LlmExecutionContextSnapshot {
    let root_uuid = Uuid::now_v7();
    LlmExecutionContextSnapshot {
        call_uuid: Uuid::now_v7(),
        root_uuid,
        parent_uuid: root_uuid,
        trajectory_owner_uuid: root_uuid,
        trajectory_owner_path: vec![nemo_relay::api::llm::LlmTrajectoryScopeSnapshot {
            uuid: root_uuid,
            name: "gateway-replay-test".into(),
            scope_type: ScopeType::Agent,
        }],
        api_family,
        call_role: LlmCallRole::Primary,
        attributes: nemo_relay::api::llm::LlmAttributes::empty(),
        tenant_id: None,
        agent_id: None,
        sanitized_metadata: BTreeMap::new(),
    }
}

async fn spawn_replay_upstream() -> (String, Arc<Mutex<Vec<Value>>>, tokio::task::JoinHandle<()>) {
    let observed = Arc::new(Mutex::new(Vec::new()));
    let captured = observed.clone();
    let handler = move |headers: HeaderMap, Json(body): Json<Value>| {
        let captured = captured.clone();
        async move {
            let authorization = headers
                .get(header::AUTHORIZATION)
                .and_then(|value| value.to_str().ok())
                .map(ToOwned::to_owned);
            let api_key = headers
                .get("x-api-key")
                .and_then(|value| value.to_str().ok())
                .map(ToOwned::to_owned);
            let semantic_header = headers
                .get("openai-beta")
                .and_then(|value| value.to_str().ok())
                .map(ToOwned::to_owned);
            captured.lock().unwrap().push(json!({
                "body": body,
                "authorization": authorization,
                "api_key": api_key,
                "semantic_header": semantic_header,
                "x_auth_token": headers.get("x-auth-token").and_then(|value| value.to_str().ok()),
                "access_secret": headers.get("cf-access-client-secret").and_then(|value| value.to_str().ok()),
                "forwarded_host": headers.get("x-forwarded-host").and_then(|value| value.to_str().ok()),
                "original_url": headers.get("x-original-url").and_then(|value| value.to_str().ok()),
                "method_override": headers.get("x-http-method-override").and_then(|value| value.to_str().ok()),
            }));
            if body["pending"] == json!(true) {
                std::future::pending::<()>().await;
            }
            Json(json!({ "served": body }))
        }
    };
    let app = Router::new()
        .route("/v1/responses", post(handler.clone()))
        .route("/v1/chat/completions", post(handler.clone()))
        .route("/v1/messages", post(handler.clone()))
        .route("/v1/messages/count_tokens", post(handler));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let task = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    (format!("http://{address}"), observed, task)
}

#[test]
fn removes_hop_by_hop_headers() {
    assert!(!should_forward_request_header(&HeaderName::from_static(
        "connection"
    )));
    assert!(!should_forward_request_header(&HeaderName::from_static(
        "host"
    )));
    assert!(should_forward_request_header(&HeaderName::from_static(
        "authorization"
    )));
    assert!(!should_record_header(&HeaderName::from_static(
        "authorization"
    )));
    assert!(!should_record_header(&HeaderName::from_static("x-api-key")));
    assert!(!should_record_header(&HeaderName::from_static(
        "anthropic-api-key"
    )));
    // Additional credential aliases must not appear in observability metadata:
    // `cookie` carries session credentials; `api-key` is the generic alias used by some providers
    // (e.g., Azure OpenAI). Without these, secrets would leak into `LlmRequest.headers` and any
    // downstream exporter that mirrors them (ATIF, OpenInference span attributes).
    assert!(!should_record_header(&HeaderName::from_static("cookie")));
    assert!(!should_record_header(&HeaderName::from_static("api-key")));
    for private in [
        "x-auth-token",
        "cf-access-client-secret",
        "x-goog-api-key",
        "x-amz-security-token",
        "x-forwarded-host",
        "x-original-url",
        "x-rewrite-url",
        "x-http-method-override",
    ] {
        assert!(!should_record_header(&HeaderName::from_static(private)));
    }
    assert!(should_record_header(&HeaderName::from_static(
        "x-request-id"
    )));
}

#[test]
fn selects_provider_routes() {
    assert_eq!(
        ProviderRoute::from_path("/responses"),
        Some(ProviderRoute::OpenAiResponses)
    );
    assert_eq!(
        ProviderRoute::from_path("/v1/responses"),
        Some(ProviderRoute::OpenAiResponses)
    );
    assert_eq!(
        ProviderRoute::from_path("/v1/messages/count_tokens"),
        Some(ProviderRoute::AnthropicCountTokens)
    );
    assert_eq!(
        ProviderRoute::from_path("/v1/chat/completions")
            .unwrap()
            .name(),
        "openai.chat_completions"
    );
    assert_eq!(
        ProviderRoute::from_path("/models"),
        Some(ProviderRoute::OpenAiModels)
    );
    assert_eq!(ProviderRoute::OpenAiModels.name(), "openai.models");
    assert_eq!(
        ProviderRoute::AnthropicMessages.name(),
        "anthropic.messages"
    );
    assert_eq!(
        ProviderRoute::AnthropicCountTokens.name(),
        "anthropic.count_tokens"
    );
    assert_eq!(
        ProviderRoute::OpenAiResponses.alignment_route(),
        GatewayRouteKind::OpenAiResponses
    );
    assert_eq!(
        ProviderRoute::OpenAiChatCompletions.alignment_route(),
        GatewayRouteKind::OpenAiChatCompletions
    );
    assert_eq!(
        ProviderRoute::OpenAiModels.alignment_route(),
        GatewayRouteKind::OpenAiModels
    );
    assert_eq!(
        ProviderRoute::AnthropicMessages.alignment_route(),
        GatewayRouteKind::AnthropicMessages
    );
    assert_eq!(
        ProviderRoute::AnthropicCountTokens.alignment_route(),
        GatewayRouteKind::AnthropicCountTokens
    );
    assert_eq!(ProviderRoute::from_path("/unsupported"), None);
}

#[test]
fn provider_routes_expose_only_generation_families_for_v2_replay() {
    assert_eq!(
        ProviderRoute::OpenAiResponses.api_family(),
        Some(LlmApiFamily::OpenAIResponses)
    );
    assert_eq!(
        ProviderRoute::OpenAiChatCompletions.api_family(),
        Some(LlmApiFamily::OpenAIChatCompletions)
    );
    assert_eq!(
        ProviderRoute::AnthropicMessages.api_family(),
        Some(LlmApiFamily::AnthropicMessages)
    );
    for route in [
        ProviderRoute::OpenAiModels,
        ProviderRoute::AnthropicCountTokens,
    ] {
        assert_eq!(route.api_family(), None);
        assert_eq!(route.replay_transport_identity(), None);
    }

    let identities = [
        ProviderRoute::OpenAiResponses,
        ProviderRoute::OpenAiChatCompletions,
        ProviderRoute::AnthropicMessages,
    ]
    .map(|route| route.replay_transport_identity().unwrap());
    assert!(identities.iter().all(|identity| {
        identity.starts_with("nemo-relay-cli:")
            && !identity.contains("http")
            && !identity.contains("key")
            && !identity.contains("token")
    }));
}

#[test]
fn replay_headers_freeze_auth_and_reject_candidate_credential_mutation() {
    let mut inbound = HeaderMap::new();
    inbound.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.payload.signature"),
    );
    inbound.insert(header::COOKIE, HeaderValue::from_static("session=frozen"));
    inbound.insert("x-safe", HeaderValue::from_static("original"));
    inbound.insert("openai-beta", HeaderValue::from_static("original"));
    inbound.insert(
        "x-auth-token",
        HeaderValue::from_static("frozen-custom-auth"),
    );
    inbound.insert(header::HOST, HeaderValue::from_static("client.invalid"));

    let observable = observable_headers(&inbound);
    assert!(!observable.contains_key("authorization"));
    assert!(!observable.contains_key("x-auth-token"));

    let mut frozen = frozen_replay_headers_with_keys(
        &inbound,
        ProviderRoute::OpenAiResponses,
        Some("sk-frozen-openai"),
        None,
    );
    assert_eq!(
        frozen.get(header::AUTHORIZATION).unwrap(),
        "Bearer sk-frozen-openai"
    );
    assert_eq!(frozen.get(header::COOKIE).unwrap(), "session=frozen");
    assert!(frozen.get(header::HOST).is_none());

    overlay_replay_headers(
        &mut frozen,
        &Map::from_iter([
            ("authorization".into(), json!("Bearer candidate-secret")),
            ("cookie".into(), json!("session=candidate")),
            ("x-api-key".into(), json!("candidate-secret")),
            ("host".into(), json!("candidate.invalid")),
            ("x-safe".into(), json!("candidate")),
            ("openai-beta".into(), json!("candidate-beta")),
            ("x-auth-token".into(), json!("candidate-secret")),
            ("cf-access-client-secret".into(), json!("candidate-secret")),
            ("x-forwarded-host".into(), json!("candidate.invalid")),
            ("x-original-url".into(), json!("https://candidate.invalid")),
            ("x-http-method-override".into(), json!("DELETE")),
        ]),
    );

    assert_eq!(
        frozen.get(header::AUTHORIZATION).unwrap(),
        "Bearer sk-frozen-openai"
    );
    assert_eq!(frozen.get(header::COOKIE).unwrap(), "session=frozen");
    assert_eq!(frozen.get("x-safe").unwrap(), "original");
    assert_eq!(frozen.get("openai-beta").unwrap(), "candidate-beta");
    assert!(frozen.get("x-api-key").is_none());
    assert!(frozen.get(header::HOST).is_none());
    assert_eq!(frozen.get("x-auth-token").unwrap(), "frozen-custom-auth");
    for protected in [
        "cf-access-client-secret",
        "x-forwarded-host",
        "x-original-url",
        "x-http-method-override",
    ] {
        assert!(frozen.get(protected).is_none());
    }

    let anthropic = frozen_replay_headers_with_keys(
        &HeaderMap::new(),
        ProviderRoute::AnthropicMessages,
        None,
        Some("sk-ant-frozen"),
    );
    assert_eq!(anthropic.get("x-api-key").unwrap(), "sk-ant-frozen");
    assert!(anthropic.get(header::AUTHORIZATION).is_none());
}

#[tokio::test]
async fn gateway_replay_transport_is_delayed_repeatable_and_response_isolated() {
    let (base_url, observed, server) = spawn_replay_upstream().await;
    let config = GatewayConfig::default();
    let state = AppState {
        config: config.clone(),
        http: test_http_client(),
        sessions: SessionManager::new(config),
        last_activity: Arc::new(Mutex::new(std::time::Instant::now())),
    };
    let prepared = PreparedGatewayRequest {
        method: Method::POST,
        headers: HeaderMap::from_iter([
            (
                header::AUTHORIZATION,
                HeaderValue::from_static("Bearer frozen-anchor"),
            ),
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            ),
            (
                HeaderName::from_static("x-original-safe"),
                HeaderValue::from_static("kept"),
            ),
        ]),
        path: "/v1/responses".into(),
        provider: ProviderRoute::OpenAiResponses,
        upstream_url: format!("{base_url}/v1/responses"),
        body_bytes: Bytes::from_static(br#"{"model":"original"}"#),
        request_json: json!({"model": "original"}),
        streaming: false,
    };
    let cancellation_count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let factory = GatewayReplayFactory::new(
        GatewayTransportPolicy::from_prepared(&state, &prepared),
        prepared.provider,
        LlmApiFamily::OpenAIResponses,
    )
    .with_cancellation_count(cancellation_count.clone());
    let transport = factory
        .build(&replay_context(LlmApiFamily::OpenAIResponses))
        .unwrap();
    drop(factory);

    assert_eq!(
        transport.capability(),
        &LlmReplayCapability {
            contract_version: LLM_REPLAY_CONTRACT_VERSION,
            api_family: LlmApiFamily::OpenAIResponses,
            transport_identity: "nemo-relay-cli:openai-responses:v1".into(),
        }
    );

    let first = transport
        .start(LlmRequest {
            headers: Map::from_iter([
                ("authorization".into(), json!("Bearer candidate-secret")),
                ("openai-beta".into(), json!("first")),
                ("x-auth-token".into(), json!("candidate-secret")),
                ("cf-access-client-secret".into(), json!("candidate-secret")),
                ("x-forwarded-host".into(), json!("candidate.invalid")),
                ("x-original-url".into(), json!("https://candidate.invalid")),
                ("x-http-method-override".into(), json!("DELETE")),
            ]),
            content: json!({"model": "candidate-a", "id": 1}),
        })
        .unwrap()
        .await
        .unwrap();
    assert_eq!(first["served"]["model"], json!("candidate-a"));

    let second = transport
        .start(LlmRequest {
            headers: Map::from_iter([("openai-beta".into(), json!("second"))]),
            content: json!({"model": "candidate-b", "id": 2}),
        })
        .unwrap();
    let third = transport
        .start(LlmRequest {
            headers: Map::new(),
            content: json!({"model": "candidate-c", "id": 3}),
        })
        .unwrap();
    let (second, third) = tokio::join!(second, third);
    assert_eq!(second.unwrap()["served"]["id"], json!(2));
    assert_eq!(third.unwrap()["served"]["id"], json!(3));

    let pending = transport
        .start(LlmRequest {
            headers: Map::new(),
            content: json!({"pending": true, "id": "cancelled"}),
        })
        .unwrap();
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        loop {
            if observed
                .lock()
                .unwrap()
                .iter()
                .any(|request| request["body"]["pending"] == json!(true))
            {
                break;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("pending replay request did not reach the upstream");
    drop(pending);
    tokio::time::timeout(std::time::Duration::from_secs(2), async {
        while cancellation_count.load(std::sync::atomic::Ordering::SeqCst) != 1 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("dropping the in-flight replay did not cancel its task");
    assert_eq!(
        cancellation_count.load(std::sync::atomic::Ordering::SeqCst),
        1
    );
    let sibling = transport
        .start(LlmRequest {
            headers: Map::new(),
            content: json!({"id": "sibling"}),
        })
        .unwrap()
        .await
        .unwrap();
    assert_eq!(sibling["served"]["id"], json!("sibling"));
    assert_eq!(
        cancellation_count.load(std::sync::atomic::Ordering::SeqCst),
        1
    );

    let observed = observed.lock().unwrap().clone();
    assert!(observed.len() >= 5);
    for request in &observed {
        assert_eq!(request["authorization"], json!("Bearer frozen-anchor"));
        for protected in [
            "x_auth_token",
            "access_secret",
            "forwarded_host",
            "original_url",
            "method_override",
        ] {
            assert!(request[protected].is_null());
        }
    }
    assert_eq!(observed[0]["semantic_header"], json!("first"));
    server.abort();
}

#[tokio::test]
async fn buffered_generation_routes_propagate_v2_context_and_delayed_replay() {
    let _global_guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT_NAME: &str = "cli-gateway-spec09a-v2-capture";
    let _ = deregister_llm_execution_intercept(INTERCEPT_NAME);
    let captured = Arc::new(Mutex::new(Vec::<(
        LlmApiFamily,
        LlmCallRole,
        BTreeMap<String, Value>,
        Option<String>,
        Option<String>,
        LlmRequest,
        Arc<dyn LlmReplayTransport>,
    )>::new()));
    let observed = captured.clone();
    let next_calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let observed_next_calls = next_calls.clone();
    register_llm_execution_intercept_v2(
        INTERCEPT_NAME,
        -100,
        Arc::new(move |_, context, request, replay, next| {
            let replay = replay.expect("eligible CLI generation route must provide replay");
            observed.lock().unwrap().push((
                context.api_family,
                context.call_role,
                context.sanitized_metadata.clone(),
                context.tenant_id.clone(),
                context.agent_id.clone(),
                request.clone(),
                replay,
            ));
            observed_next_calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let mut effective_request = request;
            effective_request
                .headers
                .insert("openai-beta".into(), json!("anchor-rewritten"));
            effective_request
                .headers
                .insert("x-auth-token".into(), json!("candidate-secret"));
            next(effective_request)
        }),
    )
    .unwrap();
    let _intercept_guard = LlmExecutionInterceptGuard(INTERCEPT_NAME);

    let (base_url, upstream_requests, server) = spawn_replay_upstream().await;
    let config = GatewayConfig::default();
    let manager = SessionManager::new(config.clone());
    let state = AppState {
        config,
        http: test_http_client(),
        sessions: manager.clone(),
        last_activity: Arc::new(Mutex::new(std::time::Instant::now())),
    };

    for (index, (path, route, family)) in [
        (
            "/v1/responses",
            ProviderRoute::OpenAiResponses,
            LlmApiFamily::OpenAIResponses,
        ),
        (
            "/v1/chat/completions",
            ProviderRoute::OpenAiChatCompletions,
            LlmApiFamily::OpenAIChatCompletions,
        ),
        (
            "/v1/messages",
            ProviderRoute::AnthropicMessages,
            LlmApiFamily::AnthropicMessages,
        ),
    ]
    .into_iter()
    .enumerate()
    {
        let body = json!({"model": format!("anchor-{index}"), "id": index});
        let mut headers = HeaderMap::new();
        headers.insert(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        );
        headers.insert(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer frozen-anchor"),
        );
        headers.insert(
            "x-nemo-relay-session-id",
            HeaderValue::from_str(&format!("spec09a-family-{index}")).unwrap(),
        );
        let prepared = PreparedGatewayRequest {
            method: Method::POST,
            headers,
            path: path.into(),
            provider: route,
            upstream_url: format!("{base_url}{path}"),
            body_bytes: Bytes::from(serde_json::to_vec(&body).unwrap()),
            request_json: body.clone(),
            streaming: false,
        };
        let prep = manager
            .prepare_gateway_call(&prepared.headers, build_llm_gateway_start(&prepared))
            .await
            .unwrap();
        let response = run_managed_buffered(
            state.clone(),
            prepared,
            prep,
            RouteCodecs {
                streaming: None,
                response: None,
            },
        )
        .await
        .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let response: Value =
            serde_json::from_slice(&response.into_body().collect().await.unwrap().to_bytes())
                .unwrap();
        assert_eq!(response["served"], body);

        let transport = captured.lock().unwrap()[index].6.clone();
        let replay = transport
            .start(LlmRequest {
                headers: Map::new(),
                content: json!({"model": format!("candidate-{index}"), "replay": index}),
            })
            .unwrap()
            .await
            .unwrap();
        assert_eq!(replay["served"]["replay"], json!(index));
        assert_eq!(transport.capability().api_family, family);
    }

    let count_body = json!({"model": "token-counter", "input": "count me"});
    let count_headers = HeaderMap::from_iter([
        (
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        ),
        (
            HeaderName::from_static("x-nemo-relay-session-id"),
            HeaderValue::from_static("spec09a-count-tokens"),
        ),
    ]);
    let count_tokens = PreparedGatewayRequest {
        method: Method::POST,
        headers: count_headers,
        path: "/v1/messages/count_tokens".into(),
        provider: ProviderRoute::AnthropicCountTokens,
        upstream_url: format!("{base_url}/v1/messages/count_tokens"),
        body_bytes: Bytes::from(serde_json::to_vec(&count_body).unwrap()),
        request_json: count_body,
        streaming: false,
    };
    let count_prep = manager
        .prepare_gateway_call(
            &count_tokens.headers,
            build_llm_gateway_start(&count_tokens),
        )
        .await
        .unwrap();
    let count_response = run_managed_buffered(
        state,
        count_tokens,
        count_prep,
        RouteCodecs {
            streaming: None,
            response: None,
        },
    )
    .await
    .unwrap();
    assert_eq!(count_response.status(), StatusCode::OK);

    assert_eq!(next_calls.load(std::sync::atomic::Ordering::SeqCst), 3);
    let captured = captured.lock().unwrap();
    assert_eq!(captured.len(), 3);
    for (index, entry) in captured.iter().enumerate() {
        assert_eq!(entry.0, entry.6.capability().api_family);
        assert_eq!(entry.1, LlmCallRole::Primary);
        assert!(entry.2.is_empty());
        assert_eq!(entry.3, None);
        assert_eq!(entry.4, None);
        assert_eq!(entry.5.content["id"], json!(index));
    }
    drop(captured);

    let upstream_requests = upstream_requests.lock().unwrap();
    assert_eq!(upstream_requests.len(), 7);
    for request in upstream_requests.iter().filter(|request| {
        request["body"]["model"]
            .as_str()
            .is_some_and(|model| model.starts_with("anchor-"))
    }) {
        assert_eq!(request["semantic_header"], json!("anchor-rewritten"));
        assert!(request["x_auth_token"].is_null());
        assert_eq!(request["authorization"], json!("Bearer frozen-anchor"));
    }
    server.abort();
}

#[tokio::test]
async fn gateway_replay_errors_do_not_expose_endpoint_or_credentials() {
    let policy = GatewayTransportPolicy {
        http: test_http_client(),
        method: Method::POST,
        url: "http://user:super-secret@127.0.0.1:9/replay?token=hidden".into(),
        original_body: Bytes::from_static(b"{}"),
        frozen_headers: HeaderMap::from_iter([(
            header::AUTHORIZATION,
            HeaderValue::from_static("Bearer hidden-auth"),
        )]),
        max_response_bytes: 1024,
    };
    let error = policy
        .execute_replay(LlmRequest {
            headers: Map::new(),
            content: json!({"model": "candidate"}),
        })
        .await
        .unwrap_err()
        .to_string();

    assert_eq!(error, "internal error: CLI gateway replay request failed");
    for secret in ["super-secret", "token=hidden", "hidden-auth"] {
        assert!(!error.contains(secret));
    }
}

#[tokio::test]
async fn gateway_replay_rejects_non_success_and_oversized_responses_without_body_leaks() {
    let app = Router::new()
        .route(
            "/status",
            post(|| async { (StatusCode::TOO_MANY_REQUESTS, "provider-secret-body") }),
        )
        .route(
            "/large",
            post(|| async { "response-body-larger-than-eight-bytes" }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let policy = GatewayTransportPolicy {
        http: test_http_client(),
        method: Method::POST,
        url: format!("http://{address}/status"),
        original_body: Bytes::from_static(b"{}"),
        frozen_headers: HeaderMap::new(),
        max_response_bytes: 8,
    };
    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "candidate"}),
    };

    let status_error = policy
        .execute_replay(request.clone())
        .await
        .unwrap_err()
        .to_string();
    assert_eq!(
        status_error,
        "internal error: CLI gateway replay upstream returned HTTP 429"
    );
    assert!(!status_error.contains("provider-secret-body"));

    let mut oversized_policy = policy;
    oversized_policy.url = format!("http://{address}/large");
    assert_eq!(
        oversized_policy
            .execute_replay(request)
            .await
            .unwrap_err()
            .to_string(),
        "internal error: CLI gateway replay response exceeded configured byte limit"
    );
    server.abort();
}

#[tokio::test]
async fn buffered_non_success_is_an_intercept_error_but_preserves_the_provider_response() {
    let _global_guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    const INTERCEPT_NAME: &str = "cli-gateway-non-success-authority";
    let _ = deregister_llm_execution_intercept(INTERCEPT_NAME);
    let observed_error = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed = Arc::clone(&observed_error);
    register_llm_execution_intercept_v2(
        INTERCEPT_NAME,
        -100,
        Arc::new(move |_, _, request, _, next| {
            let observed = Arc::clone(&observed);
            Box::pin(async move {
                let result = next(request).await;
                observed.store(result.is_err(), std::sync::atomic::Ordering::Release);
                result
            })
        }),
    )
    .unwrap();
    let _intercept_guard = LlmExecutionInterceptGuard(INTERCEPT_NAME);

    let app = Router::new().route(
        "/provider",
        post(|| async {
            (
                StatusCode::TOO_MANY_REQUESTS,
                [("x-provider-retry", "17")],
                "provider-error-body",
            )
        }),
    );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let config = GatewayConfig::default();
    let manager = SessionManager::new(config.clone());
    let state = AppState {
        config,
        http: test_http_client(),
        sessions: manager.clone(),
        last_activity: Arc::new(Mutex::new(std::time::Instant::now())),
    };
    let request_json = json!({"model": "anchor"});
    let prepared = PreparedGatewayRequest {
        method: Method::POST,
        headers: HeaderMap::from_iter([(
            header::CONTENT_TYPE,
            HeaderValue::from_static("application/json"),
        )]),
        path: "/v1/responses".into(),
        provider: ProviderRoute::OpenAiResponses,
        upstream_url: format!("http://{address}/provider"),
        body_bytes: Bytes::from(serde_json::to_vec(&request_json).unwrap()),
        request_json,
        streaming: false,
    };
    let prep = manager
        .prepare_gateway_call(&prepared.headers, build_llm_gateway_start(&prepared))
        .await
        .unwrap();

    let response = run_managed_buffered(
        state,
        prepared,
        prep,
        RouteCodecs {
            streaming: None,
            response: None,
        },
    )
    .await
    .unwrap();
    assert!(observed_error.load(std::sync::atomic::Ordering::Acquire));
    assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(response.headers()["x-provider-retry"], "17");
    assert_eq!(
        response.into_body().collect().await.unwrap().to_bytes(),
        Bytes::from_static(b"provider-error-body")
    );
    server.abort();
}

#[test]
fn buffered_response_header_limits_apply_after_hop_by_hop_filtering() {
    let mut too_many = HeaderMap::new();
    for index in 0..=MAX_FORWARDED_RESPONSE_HEADERS {
        too_many.append(
            header::SET_COOKIE,
            HeaderValue::from_str(&format!("key-{index}=value")).unwrap(),
        );
    }
    assert!(bounded_response_headers(&too_many).is_none());

    let mut too_large = HeaderMap::new();
    too_large.insert(
        "x-large",
        HeaderValue::from_bytes(&vec![b'x'; MAX_FORWARDED_RESPONSE_HEADER_BYTES]).unwrap(),
    );
    assert!(bounded_response_headers(&too_large).is_none());

    let mut filtered = HeaderMap::new();
    for _ in 0..=MAX_FORWARDED_RESPONSE_HEADERS {
        filtered.append(header::CONNECTION, HeaderValue::from_static("close"));
    }
    assert_eq!(bounded_response_headers(&filtered).unwrap().len(), 0);
}

#[tokio::test]
async fn buffered_response_overflow_returns_stable_502_without_partial_provider_bytes() {
    let _global_guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let app = Router::new().route("/large", post(|| async { "provider-partial-secret-body" }));
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let config = GatewayConfig {
        max_passthrough_body_bytes: 8,
        ..GatewayConfig::default()
    };
    let manager = SessionManager::new(config.clone());
    let state = AppState {
        config,
        http: test_http_client(),
        sessions: manager.clone(),
        last_activity: Arc::new(Mutex::new(std::time::Instant::now())),
    };
    let request_json = json!({"model": "anchor"});
    let prepared = PreparedGatewayRequest {
        method: Method::POST,
        headers: HeaderMap::new(),
        path: "/v1/responses".into(),
        provider: ProviderRoute::OpenAiResponses,
        upstream_url: format!("http://{address}/large"),
        body_bytes: Bytes::from(serde_json::to_vec(&request_json).unwrap()),
        request_json,
        streaming: false,
    };
    let prep = manager
        .prepare_gateway_call(&prepared.headers, build_llm_gateway_start(&prepared))
        .await
        .unwrap();
    let error = run_managed_buffered(
        state,
        prepared,
        prep,
        RouteCodecs {
            streaming: None,
            response: None,
        },
    )
    .await
    .unwrap_err();
    assert!(matches!(error, CliError::UpstreamResponseTooLarge));
    let response = error.into_response();
    assert_eq!(response.status(), StatusCode::BAD_GATEWAY);
    let body = response.into_body().collect().await.unwrap().to_bytes();
    let body = std::str::from_utf8(&body).unwrap();
    assert!(body.contains("upstream_response_too_large"));
    assert!(!body.contains("provider-partial-secret-body"));
    server.abort();
}

#[tokio::test]
async fn gateway_replay_rejects_unreadable_json_without_leaks_and_preserves_anchor_bytes() {
    let _global_guard = PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let app = Router::new()
        .route(
            "/non-json-endpoint-secret",
            post(|| async { (StatusCode::CREATED, "non-json-body-secret") }),
        )
        .route(
            "/invalid-utf8-endpoint-secret",
            post(|| async {
                (
                    StatusCode::OK,
                    Bytes::from_static(b"\xffinvalid-utf8-body-secret"),
                )
            }),
        );
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    let config = GatewayConfig::default();
    let manager = SessionManager::new(config.clone());
    let state = AppState {
        config,
        http: test_http_client(),
        sessions: manager.clone(),
        last_activity: Arc::new(Mutex::new(std::time::Instant::now())),
    };
    let request_json = json!({"model": "anchor"});
    let prepared = PreparedGatewayRequest {
        method: Method::POST,
        headers: HeaderMap::from_iter([
            (
                header::AUTHORIZATION,
                HeaderValue::from_static("Bearer hidden-auth"),
            ),
            (
                header::CONTENT_TYPE,
                HeaderValue::from_static("application/json"),
            ),
        ]),
        path: "/v1/responses".into(),
        provider: ProviderRoute::OpenAiResponses,
        upstream_url: format!("http://{address}/non-json-endpoint-secret?token=hidden-query"),
        body_bytes: Bytes::from(serde_json::to_vec(&request_json).unwrap()),
        request_json,
        streaming: false,
    };
    let replay_policy = GatewayTransportPolicy::from_prepared(&state, &prepared);
    let prep = manager
        .prepare_gateway_call(&prepared.headers, build_llm_gateway_start(&prepared))
        .await
        .unwrap();

    let anchor_response = run_managed_buffered(
        state,
        prepared,
        prep,
        RouteCodecs {
            streaming: None,
            response: None,
        },
    )
    .await
    .unwrap();
    assert_eq!(anchor_response.status(), StatusCode::CREATED);
    assert_eq!(
        anchor_response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes(),
        Bytes::from_static(b"non-json-body-secret")
    );

    let request = LlmRequest {
        headers: Map::new(),
        content: json!({"model": "candidate", "credential": "candidate-secret"}),
    };
    let non_json_error = replay_policy
        .execute_replay(request.clone())
        .await
        .unwrap_err()
        .to_string();
    let mut invalid_utf8_policy = replay_policy;
    invalid_utf8_policy.url =
        format!("http://{address}/invalid-utf8-endpoint-secret?token=hidden-query");
    let invalid_utf8_error = invalid_utf8_policy
        .execute_replay(request)
        .await
        .unwrap_err()
        .to_string();

    for error in [non_json_error, invalid_utf8_error] {
        assert_eq!(
            error,
            "internal error: CLI gateway replay response was not valid JSON"
        );
        for secret in [
            "non-json-body-secret",
            "invalid-utf8-body-secret",
            "non-json-endpoint-secret",
            "invalid-utf8-endpoint-secret",
            "hidden-query",
            "hidden-auth",
            "candidate-secret",
        ] {
            assert!(!error.contains(secret));
        }
    }
    server.abort();
}

#[test]
fn provider_route_names_round_trip_through_alignment_routes() {
    for route in [
        ProviderRoute::OpenAiResponses,
        ProviderRoute::OpenAiChatCompletions,
        ProviderRoute::OpenAiModels,
        ProviderRoute::AnthropicMessages,
        ProviderRoute::AnthropicCountTokens,
    ] {
        assert_eq!(
            GatewayRouteKind::from_provider_name(route.name()),
            Some(route.alignment_route())
        );
    }
}

#[test]
fn provider_routes_preserve_path_query_and_choose_upstream() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai/v1/".into(),

        anthropic_base_url: "http://anthropic/".into(),
        metadata: None,
        plugin_config: None,
        max_hook_payload_bytes: crate::config::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        max_passthrough_body_bytes: crate::config::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
    };

    assert_eq!(
        ProviderRoute::OpenAiResponses.upstream_url(&config, "/v1/responses?x=1"),
        "http://openai/v1/responses?x=1"
    );
    assert_eq!(
        ProviderRoute::OpenAiResponses.upstream_url(&config, "/responses?x=1"),
        "http://openai/v1/responses?x=1"
    );
    assert_eq!(
        ProviderRoute::OpenAiModels.upstream_url(&config, "/models"),
        "http://openai/v1/models"
    );
    assert_eq!(
        ProviderRoute::AnthropicMessages.upstream_url(&config, "/v1/messages"),
        "http://anthropic/v1/messages"
    );
}

#[test]
fn openai_upstream_url_accepts_origin_or_v1_base() {
    let mut config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai".into(),
        anthropic_base_url: "http://anthropic".into(),
        metadata: None,
        plugin_config: None,
        max_hook_payload_bytes: crate::config::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        max_passthrough_body_bytes: crate::config::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
    };

    assert_eq!(
        ProviderRoute::OpenAiResponses.upstream_url(&config, "/responses"),
        "http://openai/v1/responses"
    );
    assert_eq!(
        ProviderRoute::OpenAiResponses.upstream_url(&config, "/v1/responses"),
        "http://openai/v1/responses"
    );

    config.openai_base_url = "http://openai/v1".into();
    assert_eq!(
        ProviderRoute::OpenAiResponses.upstream_url(&config, "/responses"),
        "http://openai/v1/responses"
    );
    assert_eq!(
        ProviderRoute::OpenAiResponses.upstream_url(&config, "/v1/responses"),
        "http://openai/v1/responses"
    );
}

#[test]
fn effective_upstream_request_overlays_runtime_body_and_headers() {
    let original_body = Bytes::from_static(br#"{"model":"original"}"#);
    let mut original_headers = HeaderMap::new();
    original_headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer original"),
    );
    let request = LlmRequest {
        headers: Map::from_iter([
            ("x-runtime".to_string(), json!("enabled")),
            ("x-runtime-json".to_string(), json!({ "enabled": true })),
        ]),
        content: json!({
            "model": "rewritten",
            "nvext": { "agent_hints": { "priority": 1 } }
        }),
    };

    let (body, headers) =
        effective_upstream_request(&original_body, &original_headers, Some(&request));
    let body: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(body["model"], json!("rewritten"));
    assert_eq!(body["nvext"]["agent_hints"]["priority"], json!(1));
    assert_eq!(
        headers.get(header::AUTHORIZATION).unwrap(),
        "Bearer original"
    );
    assert_eq!(headers.get("x-runtime").unwrap(), "enabled");
    assert_eq!(
        headers.get("x-runtime-json").unwrap(),
        r#"{"enabled":true}"#
    );
}

#[test]
fn effective_upstream_request_returns_original_without_runtime_request() {
    let original_body = Bytes::from_static(br#"{"model":"original"}"#);
    let mut original_headers = HeaderMap::new();
    original_headers.insert(
        header::AUTHORIZATION,
        HeaderValue::from_static("Bearer original"),
    );
    original_headers.insert("x-request-id", HeaderValue::from_static("request-1"));

    let (body, headers) = effective_upstream_request(&original_body, &original_headers, None);

    assert_eq!(body, original_body);
    assert_eq!(
        headers.get(header::AUTHORIZATION).unwrap(),
        "Bearer original"
    );
    assert_eq!(headers.get("x-request-id").unwrap(), "request-1");
}

#[test]
fn effective_upstream_request_preserves_original_body_for_null_runtime_content() {
    let original_body = Bytes::from_static(b"not-json-but-still-upstream-body");
    let mut original_headers = HeaderMap::new();
    original_headers.insert("x-original", HeaderValue::from_static("kept"));
    let request = LlmRequest {
        headers: Map::from_iter([("x-runtime".to_string(), json!("enabled"))]),
        content: Value::Null,
    };

    let (body, headers) =
        effective_upstream_request(&original_body, &original_headers, Some(&request));

    assert_eq!(body, original_body);
    assert_eq!(headers.get("x-original").unwrap(), "kept");
    assert_eq!(headers.get("x-runtime").unwrap(), "enabled");
}

#[test]
fn effective_upstream_request_skips_invalid_runtime_headers() {
    let original_body = Bytes::from_static(br#"{"model":"original"}"#);
    let mut original_headers = HeaderMap::new();
    original_headers.insert("x-original", HeaderValue::from_static("kept"));
    let request = LlmRequest {
        headers: Map::from_iter([
            ("bad header".to_string(), json!("skip")),
            ("x-invalid-value".to_string(), json!("line\nbreak")),
            ("x-good".to_string(), json!("ok")),
        ]),
        content: json!({ "model": "rewritten" }),
    };

    let (body, headers) =
        effective_upstream_request(&original_body, &original_headers, Some(&request));
    let body: Value = serde_json::from_slice(&body).unwrap();

    assert_eq!(body["model"], json!("rewritten"));
    assert_eq!(headers.get("x-original").unwrap(), "kept");
    assert_eq!(headers.get("x-good").unwrap(), "ok");
    assert!(headers.get("x-invalid-value").is_none());
    assert!(headers.keys().all(|name| name.as_str() != "bad header"));
}

#[test]
fn gateway_session_id_prefers_headers_and_has_fallbacks() {
    let mut headers = HeaderMap::new();
    let codex_body = json!({
        "prompt_cache_key": "codex-session",
        "client_metadata": { "x-codex-installation-id": "install-1" },
        "session_id": "body-session"
    });
    headers.insert(
        "anthropic-beta",
        HeaderValue::from_static("prompt-caching-2024-07-31"),
    );
    assert_eq!(
        gateway_session_id(&headers, &Value::Null, ProviderRoute::AnthropicMessages),
        None
    );

    headers.insert(
        "x-claude-code-session-id",
        HeaderValue::from_static("claude-session"),
    );
    assert_eq!(
        gateway_session_id(&headers, &codex_body, ProviderRoute::OpenAiResponses).as_deref(),
        Some("claude-session")
    );

    headers.insert(
        "x-nemo-relay-session-id",
        HeaderValue::from_static("explicit-session"),
    );
    assert_eq!(
        gateway_session_id(&headers, &codex_body, ProviderRoute::OpenAiResponses).as_deref(),
        Some("explicit-session")
    );

    assert_eq!(
        gateway_session_id(
            &HeaderMap::new(),
            &codex_body,
            ProviderRoute::OpenAiResponses
        )
        .as_deref(),
        Some("codex-session")
    );
    assert_eq!(
        gateway_session_id(
            &HeaderMap::new(),
            &json!({ "prompt_cache_key": "plain-cache-key" }),
            ProviderRoute::OpenAiResponses,
        ),
        None
    );
    assert_eq!(
        gateway_session_id(
            &HeaderMap::new(),
            &codex_body,
            ProviderRoute::OpenAiChatCompletions,
        )
        .as_deref(),
        Some("body-session")
    );
    assert_eq!(
        gateway_session_id(
            &HeaderMap::new(),
            &json!({ "session_id": " body-session " }),
            ProviderRoute::OpenAiResponses,
        )
        .as_deref(),
        Some("body-session")
    );
    assert_eq!(
        gateway_session_id(
            &HeaderMap::new(),
            &json!({ "session_id": "body-session" }),
            ProviderRoute::AnthropicMessages,
        ),
        None
    );
}

#[test]
fn gateway_identifiers_accept_headers_and_scalar_body_values() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-nemo-relay-request-id",
        HeaderValue::from_static("req-header"),
    );
    let body = json!({
        "conversation": { "id": 42 },
        "generation": { "id": true },
        "request": { "id": "req-body" },
        "object": { "id": { "nested": true } }
    });

    assert_eq!(
        gateway_identifier(
            &headers,
            &body,
            "x-nemo-relay-request-id",
            &[&["request", "id"]]
        )
        .as_deref(),
        Some("req-header")
    );
    assert_eq!(
        gateway_identifier(
            &HeaderMap::new(),
            &body,
            "missing",
            &[&["conversation", "id"]]
        )
        .as_deref(),
        Some("42")
    );
    assert_eq!(
        gateway_identifier(
            &HeaderMap::new(),
            &body,
            "missing",
            &[&["generation", "id"]]
        )
        .as_deref(),
        Some("true")
    );
    assert_eq!(
        gateway_identifier(&HeaderMap::new(), &body, "missing", &[&["object", "id"]]),
        None
    );
}

#[test]
fn build_llm_gateway_start_uses_alignment_identifiers_and_metadata() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-nemo-relay-subagent-id",
        HeaderValue::from_static("worker-1"),
    );
    headers.insert("x-request-id", HeaderValue::from_static("transport-req"));
    headers.insert("authorization", HeaderValue::from_static("Bearer secret"));
    let request_json = json!({
        "model": "gpt-test",
        "stream": true,
        "prompt_cache_key": "codex-thread",
        "client_metadata": { "x-codex-installation-id": "install-1" },
        "conversation_id": "conversation-1",
        "generation": { "id": "generation-1" }
    });
    let prepared = PreparedGatewayRequest {
        method: Method::POST,
        headers,
        path: "/responses".into(),
        provider: ProviderRoute::OpenAiResponses,
        upstream_url: "http://openai/v1/responses".into(),
        body_bytes: axum::body::Bytes::new(),
        request_json: request_json.clone(),
        streaming: true,
    };

    let start = build_llm_gateway_start(&prepared);

    assert_eq!(start.session_id.as_deref(), Some("codex-thread"));
    assert_eq!(start.provider, "openai.responses");
    assert_eq!(start.model_name.as_deref(), Some("gpt-test"));
    assert_eq!(start.subagent_id.as_deref(), Some("worker-1"));
    assert_eq!(start.conversation_id.as_deref(), Some("conversation-1"));
    assert_eq!(start.generation_id.as_deref(), Some("generation-1"));
    assert_eq!(start.request_id.as_deref(), Some("transport-req"));
    assert!(start.streaming);
    assert_eq!(start.metadata["gateway_path"], json!("/responses"));
    assert_eq!(start.request.content, request_json);
    assert!(
        !start.request.headers.contains_key("authorization"),
        "observable headers should not leak auth secrets"
    );
}

#[test]
fn observable_headers_omit_secrets_and_transport_headers() {
    let mut headers = HeaderMap::new();
    headers.insert("authorization", HeaderValue::from_static("Bearer secret"));
    headers.insert("x-api-key", HeaderValue::from_static("secret"));
    headers.insert("connection", HeaderValue::from_static("close"));
    headers.insert("x-request-id", HeaderValue::from_static("req-1"));

    let observed = observable_headers(&headers);

    assert_eq!(observed.get("x-request-id"), Some(&json!("req-1")));
    assert!(!observed.contains_key("authorization"));
    assert!(!observed.contains_key("x-api-key"));
    assert!(!observed.contains_key("connection"));
}

#[test]
fn strips_chatgpt_plus_jwt_from_openai_route_inbound() {
    // When OPENAI_API_KEY is set the gateway strips JWT-shaped (`Bearer eyJ...`) Authorization
    // from inbound OpenAI-route requests so the auth-injection path substitutes the env key
    // instead of forwarding the ChatGPT-Plus OAuth JWT.
    let mut inbound = HeaderMap::new();
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
    );
    let sanitized = strip_replaceable_agent_auth_headers_with_openai_key_state(
        &inbound,
        ProviderRoute::OpenAiResponses,
        true,
    );
    assert!(sanitized.get("authorization").is_none());
}

#[test]
fn preserves_real_bearer_keys_on_openai_route() {
    // Real provider keys (Hermes's `sk-...` against NVIDIA, an actual OpenAI dev key, etc.)
    // must pass through untouched — only recognized ChatGPT auth tokens are stripped.
    let mut inbound = HeaderMap::new();
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer sk-real-provider-key"),
    );
    let sanitized = strip_replaceable_agent_auth_headers_with_openai_key_state(
        &inbound,
        ProviderRoute::OpenAiResponses,
        true,
    );
    assert_eq!(
        sanitized.get("authorization").unwrap(),
        "Bearer sk-real-provider-key"
    );
}

#[test]
fn does_not_touch_anthropic_route_authorization() {
    // Defensive — the JWT shape only conflicts with OpenAI routes; Anthropic routes use
    // `x-api-key` anyway. Leaving Anthropic's Authorization alone avoids any cross-provider
    // edge cases.
    let mut inbound = HeaderMap::new();
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer eyJ.anthropic.case"),
    );
    let sanitized = strip_replaceable_agent_auth_headers_with_openai_key_state(
        &inbound,
        ProviderRoute::AnthropicMessages,
        true,
    );
    assert!(sanitized.get("authorization").is_some());
}

#[test]
fn preserves_jwt_when_no_replacement_key_available() {
    // If OPENAI_API_KEY isn't set the gateway has nothing to inject after stripping, so leave
    // the inbound bearer in place. Stripping would silently de-auth setups that point at an
    // upstream which happens to accept the ChatGPT-Plus token.
    let mut inbound = HeaderMap::new();
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
    );
    let sanitized = strip_replaceable_agent_auth_headers_with_openai_key_state(
        &inbound,
        ProviderRoute::OpenAiResponses,
        false,
    );
    assert!(sanitized.get("authorization").is_some());
}

#[test]
fn injects_openai_bearer_when_inbound_has_no_auth() {
    // NMF-86 mitigation: codex now sends no credentials, so the gateway must inject
    // `Authorization: Bearer ${OPENAI_API_KEY}` on outbound forwards to api.openai.com.
    let http = test_http_client();
    let inbound = HeaderMap::new();
    let env = |k: &str| match k {
        "OPENAI_API_KEY" => Some("sk-test-123".into()),
        _ => None,
    };
    let builder = http.get("http://upstream/v1/responses");
    let built =
        inject_provider_auth_with_env(builder, ProviderRoute::OpenAiResponses, &inbound, env)
            .build()
            .unwrap();
    assert_eq!(
        built.headers().get("authorization").unwrap(),
        "Bearer sk-test-123"
    );
}

#[test]
fn injects_anthropic_x_api_key_for_anthropic_routes() {
    let http = test_http_client();
    let inbound = HeaderMap::new();
    let env = |k: &str| match k {
        "ANTHROPIC_API_KEY" => Some("sk-ant-test".into()),
        _ => None,
    };
    let builder = http.post("http://upstream/v1/messages");
    let built =
        inject_provider_auth_with_env(builder, ProviderRoute::AnthropicMessages, &inbound, env)
            .build()
            .unwrap();
    assert_eq!(built.headers().get("x-api-key").unwrap(), "sk-ant-test");
    // Anthropic uses `x-api-key`, not Authorization. The gateway must not duplicate the secret
    // into a Bearer header — that would defeat the purpose of using the provider's standard
    // auth scheme and might trigger upstream-side rejection of the conflicting auth.
    assert!(built.headers().get("authorization").is_none());
}

#[test]
fn skips_injection_when_inbound_already_has_authorization() {
    // If the agent (e.g., a future codex version, or anyone using the gateway directly) sends
    // its own auth, we must not stomp on it.
    let http = test_http_client();
    let mut inbound = HeaderMap::new();
    inbound.insert(
        "authorization",
        HeaderValue::from_static("Bearer agent-supplied"),
    );
    let env = |_: &str| Some("sk-test-from-env".into());
    let builder = http.post("http://upstream/v1/responses");
    let built =
        inject_provider_auth_with_env(builder, ProviderRoute::OpenAiResponses, &inbound, env)
            .build()
            .unwrap();
    // The builder doesn't carry inbound headers itself (forward_upstream_request adds them in a
    // separate loop), so the only header on `built` would be the env-injected one. Since the
    // inbound had auth, we expect no injection at all.
    assert!(built.headers().get("authorization").is_none());
}

#[test]
fn skips_injection_when_env_var_unset() {
    let http = test_http_client();
    let inbound = HeaderMap::new();
    let env = |_: &str| None;
    let builder = http.post("http://upstream/v1/responses");
    let built =
        inject_provider_auth_with_env(builder, ProviderRoute::OpenAiResponses, &inbound, env)
            .build()
            .unwrap();
    assert!(built.headers().get("authorization").is_none());
}

// --- ChatGPT backend routing tests ---

#[test]
fn chatgpt_jwt_routes_to_chatgpt_backend_when_no_api_key() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
    );
    // With no OPENAI_API_KEY and a JWT, alignment returns the ChatGPT backend override.
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::OpenAiResponses,
            &headers,
            "/responses",
            false,
        )
        .as_deref(),
        Some("https://chatgpt.com/backend-api/codex/responses")
    );
}

#[test]
fn provider_key_does_not_trigger_chatgpt_backend() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer sk-real-api-key"),
    );
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::OpenAiResponses,
            &headers,
            "/responses",
            false,
        ),
        None
    );

    // Empty headers also should not trigger.
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::OpenAiResponses,
            &HeaderMap::new(),
            "/responses",
            false,
        ),
        None
    );
}

#[test]
fn anthropic_route_never_triggers_chatgpt_backend() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
    );
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::AnthropicMessages,
            &headers,
            "/v1/messages",
            false,
        ),
        None
    );
}

#[test]
fn chatgpt_backend_url_omits_v1_prefix() {
    let mut headers = HeaderMap::new();
    headers.insert(
        "authorization",
        HeaderValue::from_static("Bearer eyJhbGciOiJIUzI1NiJ9.deadbeef.signature"),
    );
    // The ChatGPT backend expects paths directly under the base, not /v1-prefixed.
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::OpenAiResponses,
            &headers,
            "/responses",
            false,
        )
        .as_deref(),
        Some("https://chatgpt.com/backend-api/codex/responses")
    );
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::OpenAiModels,
            &headers,
            "/models",
            false,
        )
        .as_deref(),
        Some("https://chatgpt.com/backend-api/codex/models")
    );
    // /v1-prefixed inbound paths are stripped
    assert_eq!(
        gateway_upstream_url_override_with_openai_key_state(
            ProviderRoute::OpenAiResponses,
            &headers,
            "/v1/responses",
            false,
        )
        .as_deref(),
        Some("https://chatgpt.com/backend-api/codex/responses")
    );
}

#[tokio::test]
async fn passthrough_rejects_unsupported_provider_path_directly() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai".into(),

        anthropic_base_url: "http://anthropic".into(),
        metadata: None,
        plugin_config: None,
        max_hook_payload_bytes: crate::config::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        max_passthrough_body_bytes: crate::config::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
    };
    let state = AppState {
        config: config.clone(),
        http: test_http_client(),
        sessions: SessionManager::new(config),
        last_activity: std::sync::Arc::new(std::sync::Mutex::new(std::time::Instant::now())),
    };
    let request = Request::builder()
        .method(Method::POST)
        .uri("/unsupported")
        .body(Body::empty())
        .unwrap();

    let error = passthrough(State(state), request).await.unwrap_err();

    assert!(error.to_string().contains("unsupported gateway path"));
}

#[tokio::test]
async fn models_rejects_non_get_requests_directly() {
    let config = GatewayConfig {
        bind: "127.0.0.1:0".parse().unwrap(),
        openai_base_url: "http://openai".into(),

        anthropic_base_url: "http://anthropic".into(),
        metadata: None,
        plugin_config: None,
        max_hook_payload_bytes: crate::config::DEFAULT_MAX_HOOK_PAYLOAD_BYTES,
        max_passthrough_body_bytes: crate::config::DEFAULT_MAX_PASSTHROUGH_BODY_BYTES,
    };
    let state = AppState {
        config: config.clone(),
        http: test_http_client(),
        sessions: SessionManager::new(config),
        last_activity: std::sync::Arc::new(std::sync::Mutex::new(std::time::Instant::now())),
    };
    let request = Request::builder()
        .method(Method::POST)
        .uri("/v1/models")
        .body(Body::empty())
        .unwrap();

    let response = models(State(state), request).await.unwrap();

    assert_eq!(response.status(), StatusCode::METHOD_NOT_ALLOWED);
    assert!(
        response
            .into_body()
            .collect()
            .await
            .unwrap()
            .to_bytes()
            .is_empty()
    );
}

#[test]
fn response_headers_preserve_duplicates() {
    let mut headers = HeaderMap::new();
    headers.append("set-cookie", HeaderValue::from_static("a=1"));
    headers.append("set-cookie", HeaderValue::from_static("b=2"));

    let copied = response_headers(&headers);

    assert_eq!(copied.get_all("set-cookie").iter().count(), 2);
}

#[tokio::test]
async fn streaming_gateway_call_guard_finishes_when_body_is_dropped() {
    let manager = SessionManager::new(GatewayConfig::default());
    let prep = manager
        .prepare_gateway_call(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("stream-drop".into()),
                provider: "openai.responses".into(),
                model_name: Some("gpt-test".into()),
                subagent_id: None,
                conversation_id: None,
                generation_id: None,
                request_id: None,
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({
                        "input": "Analyze enough text to create a stable idle-timeout test."
                    }),
                },
                streaming: true,
                metadata: json!({}),
            },
        )
        .await
        .unwrap();

    let stream: LlmJsonStream = Box::pin(futures_util::stream::pending::<
        std::result::Result<Value, FlowError>,
    >());
    let body = client_sse_body(
        stream,
        ProviderRoute::OpenAiResponses,
        manager.clone(),
        prep.session_id,
        prep.owner_subagent_id,
        Arc::new(Mutex::new(None)),
    );

    drop(body);
    tokio::task::yield_now().await;

    let closed = manager
        .close_idle_sessions_at(
            std::time::Instant::now() + std::time::Duration::from_secs(1),
            std::time::Duration::from_millis(1),
            "idle_timeout",
        )
        .await
        .unwrap();
    assert_eq!(closed, 1);
}

#[tokio::test]
async fn streaming_body_records_final_response_for_turn_output() {
    let subscriber_name = "gateway-stream-final-response-turn-output-test";
    let _ = nemo_relay::api::subscriber::deregister_subscriber(subscriber_name);
    let captured_output = Arc::new(Mutex::new(None::<Value>));
    let captured = captured_output.clone();
    nemo_relay::api::subscriber::register_subscriber(
        subscriber_name,
        Arc::new(move |event| {
            if event.scope_category() == Some(nemo_relay::api::event::ScopeCategory::End)
                && event.name() == "codex-turn"
                && event
                    .metadata()
                    .and_then(|metadata| metadata.get("session_id"))
                    .and_then(Value::as_str)
                    == Some("stream-final")
            {
                *captured.lock().unwrap() = event.output().cloned();
            }
        }),
    )
    .unwrap();

    let manager = SessionManager::new(GatewayConfig::default());
    let prep = manager
        .prepare_gateway_call(
            &HeaderMap::new(),
            LlmGatewayStart {
                session_id: Some("stream-final".into()),
                provider: "openai.responses".into(),
                model_name: Some("gpt-test".into()),
                subagent_id: None,
                conversation_id: None,
                generation_id: None,
                request_id: None,
                request: LlmRequest {
                    headers: Map::new(),
                    content: json!({
                        "input": "Stream enough text to create a final response."
                    }),
                },
                streaming: true,
                metadata: json!({}),
            },
        )
        .await
        .unwrap();
    let session_id = prep.session_id.clone();
    let owner_subagent_id = prep.owner_subagent_id.clone();
    let final_response = json!({ "output_text": "streamed final" });
    let stream: LlmJsonStream = Box::pin(futures_util::stream::empty::<
        std::result::Result<Value, FlowError>,
    >());
    let body = client_sse_body(
        stream,
        ProviderRoute::OpenAiResponses,
        manager.clone(),
        session_id,
        owner_subagent_id,
        Arc::new(Mutex::new(Some(final_response.clone()))),
    );
    let _ = body.collect().await.unwrap();

    manager
        .close_idle_sessions_at(
            std::time::Instant::now() + std::time::Duration::from_secs(1),
            std::time::Duration::from_millis(1),
            "idle_timeout",
        )
        .await
        .unwrap();

    nemo_relay::api::subscriber::flush_subscribers().unwrap();
    assert_eq!(*captured_output.lock().unwrap(), Some(final_response));
    nemo_relay::api::subscriber::deregister_subscriber(subscriber_name).unwrap();
}

// `stream_response_records_preview_and_truncation` was removed when the gateway moved to
// `llm_stream_call_execute`. The runtime now owns stream-end lifecycle (start/end events emitted
// by `LlmStreamWrapper`); core tests cover that contract, and the gateway no longer carries a
// stream preview/truncation helper.
