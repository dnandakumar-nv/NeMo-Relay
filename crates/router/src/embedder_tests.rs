// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::io::{ErrorKind, Read, Write};
use std::net::{IpAddr, Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use nemo_relay::api::event::{Event, ScopeCategory};
use nemo_relay::api::runtime::{
    NemoRelayContextState, TASK_SCOPE_STACK, create_scope_stack, global_context, task_scope_top,
};
use nemo_relay::api::scope::{PopScopeParams, PushScopeParams, ScopeType, pop_scope, push_scope};
use nemo_relay::api::subscriber::{deregister_subscriber, flush_subscribers, register_subscriber};
use serde_json::{Value as Json, json};
use uuid::Uuid;
use zeroize::Zeroizing;

use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::config::{
    EMBEDDER_REQUEST_BYTES_MAX, EmbedderConfig, RouterConfig, embedder_response_bytes_bound,
};
use crate::embedder::{
    EMBEDDER_ACTIVATION_MEMORY_BYTES_MAX, EMBEDDER_SCOPE_NAME, EmbedderBatchItem,
    EmbedderBuildFailure, EmbedderFailureClass, EmbedderFailureDisposition, EmbedderWorkKind,
    FrozenEmbedderClients, build_frozen_embedder_clients_for_test,
    embedder_memory_reservation_bytes, request_body,
};
use crate::fingerprint::sha256_hex;
use crate::ledger::repository::vector_registry::{VectorRegistryEnsure, prepare_vector_registry};
use crate::vector::VectorSpaceId;

const SECRET_ENV: &str = "ROUTER_EMBEDDING_TEST_KEY";
const SECRET: &str = "task9-secret-value";
const PAID_PROBE_RUN_ENV: &str = "NEMO_RELAY_RUN_PAID_EMBEDDING_PROBE";
const PAID_PROBE_COST_ENV: &str = "NEMO_RELAY_ACCEPT_PAID_EMBEDDING_COST";
const PAID_PROBE_ENDPOINT_ENV: &str = "NEMO_RELAY_PAID_EMBEDDING_ENDPOINT";
const PAID_PROBE_RUN_VALUE: &str = "1";
const PAID_PROBE_COST_VALUE: &str = "I_ACCEPT_ONE_PAID_EMBEDDING_REQUEST";
const PAID_PROBE_API_KEY_ENV: &str = "OPENAI_API_KEY";
const PAID_PROBE_MODEL: &str = "text-embedding-3-small";
const PAID_PROBE_DIMENSIONS: u32 = 1_536;
const PAID_PROBE_START_LIMIT: usize = 1;
const OFFICIAL_PAID_PROBE_BASE_URLS: [&str; 1] = ["https://api.openai.com/v1"];

#[derive(Default)]
struct PaidProbeRequestBudget {
    starts: AtomicUsize,
}

impl PaidProbeRequestBudget {
    fn begin(&self) -> Result<(), ()> {
        self.starts
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |starts| {
                (starts < PAID_PROBE_START_LIMIT).then_some(starts + 1)
            })
            .map(|_| ())
            .map_err(|_| ())
    }

    fn starts(&self) -> usize {
        self.starts.load(Ordering::Acquire)
    }
}

fn paid_probe_endpoint<'a>(
    run_opt_in: Option<&str>,
    cost_opt_in: Option<&str>,
    endpoint: Option<&'a str>,
) -> Result<&'a str, &'static str> {
    if run_opt_in != Some(PAID_PROBE_RUN_VALUE) {
        return Err(PAID_PROBE_RUN_ENV);
    }
    if cost_opt_in != Some(PAID_PROBE_COST_VALUE) {
        return Err(PAID_PROBE_COST_ENV);
    }
    let endpoint = endpoint.ok_or(PAID_PROBE_ENDPOINT_ENV)?;
    if !OFFICIAL_PAID_PROBE_BASE_URLS.contains(&endpoint) {
        return Err(PAID_PROBE_ENDPOINT_ENV);
    }
    Ok(endpoint)
}

pub(crate) struct RawHttpServer {
    address: SocketAddr,
    requests: Arc<Mutex<Vec<Vec<u8>>>>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl RawHttpServer {
    pub(crate) fn start(response: Vec<u8>) -> Self {
        Self::start_with_delay(response, Duration::ZERO)
    }

    pub(crate) fn start_with_delay(response: Vec<u8>, delay: Duration) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        Self::from_listener(listener, response, delay)
    }

    fn from_listener(listener: TcpListener, response: Vec<u8>, delay: Duration) -> Self {
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let request_sink = requests.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_signal = stop.clone();
        let thread = thread::spawn(move || {
            while !stop_signal.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).unwrap();
                        let request = read_http_request(&mut stream);
                        if request.is_empty() {
                            continue;
                        }
                        request_sink.lock().unwrap().push(request);
                        if !delay.is_zero() {
                            thread::sleep(delay);
                        }
                        let _ = stream.write_all(&response);
                        let _ = stream.flush();
                    }
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(2));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            address,
            requests,
            stop,
            thread: Some(thread),
        }
    }

    pub(crate) fn base_url(&self, host: &str) -> String {
        format!("http://{host}:{}/v1///", self.address.port())
    }

    pub(crate) fn request_count(&self) -> usize {
        self.requests.lock().unwrap().len()
    }

    pub(crate) fn wait_for_request(&self) -> Vec<u8> {
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if let Some(request) = self.requests.lock().unwrap().first().cloned() {
                return request;
            }
            assert!(
                Instant::now() < deadline,
                "test HTTP request did not arrive"
            );
            thread::sleep(Duration::from_millis(2));
        }
    }
}

impl Drop for RawHttpServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = TcpStream::connect_timeout(&self.address, Duration::from_millis(50));
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

fn read_http_request(stream: &mut TcpStream) -> Vec<u8> {
    stream
        .set_read_timeout(Some(Duration::from_secs(2)))
        .unwrap();
    let mut request = Vec::new();
    let mut buffer = [0_u8; 8192];
    let mut expected_len = None;
    loop {
        match stream.read(&mut buffer) {
            Ok(0) => break,
            Ok(read) => request.extend_from_slice(&buffer[..read]),
            Err(error) if matches!(error.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                break;
            }
            Err(_) => break,
        }
        if expected_len.is_none()
            && let Some(header_end) = find_bytes(&request, b"\r\n\r\n")
        {
            let headers = String::from_utf8_lossy(&request[..header_end]);
            let content_length = headers
                .lines()
                .find_map(|line| {
                    let (name, value) = line.split_once(':')?;
                    name.eq_ignore_ascii_case("content-length")
                        .then(|| value.trim().parse::<usize>().ok())
                        .flatten()
                })
                .unwrap_or_default();
            expected_len = header_end
                .checked_add(4)
                .and_then(|length| length.checked_add(content_length));
        }
        if expected_len.is_some_and(|length| request.len() >= length) {
            break;
        }
    }
    request
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

pub(crate) fn response(status: &str, headers: &[(&str, String)], body: &[u8]) -> Vec<u8> {
    let mut response = format!("HTTP/1.1 {status}\r\nConnection: close\r\n");
    for (name, value) in headers {
        response.push_str(name);
        response.push_str(": ");
        response.push_str(value);
        response.push_str("\r\n");
    }
    response.push_str("\r\n");
    let mut bytes = response.into_bytes();
    bytes.extend_from_slice(body);
    bytes
}

pub(crate) fn json_response(value: Json) -> Vec<u8> {
    let body = serde_json::to_vec(&value).unwrap();
    response(
        "200 OK",
        &[
            ("Content-Type", "application/json".to_string()),
            ("Content-Length", body.len().to_string()),
        ],
        &body,
    )
}

fn chunked_json_response(value: Json) -> Vec<u8> {
    let body = serde_json::to_vec(&value).unwrap();
    let split = body.len() / 2;
    let chunks = [&body[..split], &body[split..]];
    let mut encoded = Vec::new();
    for chunk in chunks {
        encoded.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
        encoded.extend_from_slice(chunk);
        encoded.extend_from_slice(b"\r\n");
    }
    encoded.extend_from_slice(b"0\r\n\r\n");
    response(
        "200 OK",
        &[("Transfer-Encoding", "chunked".to_string())],
        &encoded,
    )
}

fn router_config(
    base_url: String,
    credential: bool,
    timeout_ms: u64,
    max_in_flight: usize,
    batch_size: usize,
) -> RouterConfig {
    let mut profile = json!({
        "id": "embedding-main",
        "base_url": base_url,
        "model": "embed-model",
        "provider_revision": "revision-1",
        "dimensions": 2,
        "timeout_ms": timeout_ms,
        "max_in_flight": max_in_flight,
        "batch_size": batch_size,
    });
    if credential {
        profile["api_key_env"] = json!(SECRET_ENV);
    }
    serde_json::from_value(json!({
        "version": 1,
        "mode": "shadow",
        "embedders": [profile],
        "pools": [{
            "id": "pool-a",
            "api_family": "openai_chat_completions",
            "anchor_models": ["anchor-model"],
            "anchor_revision": "revision-1",
            "sampling_probability": 0.25,
            "max_candidates_per_sample": 1,
            "concurrency": {"shadow": 2, "judge": 1},
            "candidates": [{
                "id": "candidate-a",
                "model": "candidate-model",
                "model_revision": "revision-1",
                "cost_rank": 0
            }],
            "judge": {
                "version": 1,
                "model": "judge-model",
                "model_revision": "revision-1",
                "prompt_version": "pairwise-equivalence-v1",
                "rubric_version": "response-trajectory-equivalence-v1",
                "output_schema_version": 1,
                "response_weight": 0.5,
                "trajectory_weight": 0.5,
                "response_floor": 0.8,
                "trajectory_floor": 0.8,
                "judge_confidence_floor": 0.7,
                "pass_threshold": 0.85,
                "max_rationale_bytes": 4096,
                "base_cooloff_seconds": 10,
                "max_cooloff_seconds": 300
            },
            "learning": {"version": 1, "embedder": "embedding-main"}
        }]
    }))
    .unwrap()
}

fn prepared_registry(config: &RouterConfig) -> (VectorRegistryEnsure, VectorSpaceId) {
    let config_generation_id = config.generation_id().unwrap();
    let policies = config
        .policy_generation_values()
        .unwrap()
        .into_iter()
        .map(|(pool, value)| (pool, canonical_sha256(&value).unwrap()))
        .collect::<BTreeMap<_, _>>();
    let registry =
        prepare_vector_registry(config, Uuid::now_v7(), &config_generation_id, &policies, 1)
            .unwrap();
    let vector_space_id = registry.spaces.keys().next().unwrap().clone();
    (registry, vector_space_id)
}

fn clients(
    config: &RouterConfig,
    secret: Result<&str, ()>,
) -> Result<(FrozenEmbedderClients, VectorSpaceId), EmbedderBuildFailure> {
    let (registry, vector_space_id) = prepared_registry(config);
    build_frozen_embedder_clients_for_test(
        config,
        &registry,
        |name| {
            assert_eq!(name, SECRET_ENV);
            secret.map(str::to_string)
        },
        |_host, port| Ok(vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)]),
    )
    .map(|clients| (clients, vector_space_id))
}

fn item(text: &str, marker: char) -> EmbedderBatchItem {
    let canonical_query = canonical_json(&json!({"task": text})).unwrap();
    EmbedderBatchItem::new(
        sha256_hex(canonical_query.as_bytes()),
        canonical_query,
        marker.to_string().repeat(64),
    )
    .unwrap()
}

fn event_has_any_work_id(event: &Event, metadata_key: &str, expected: &[String]) -> bool {
    event
        .metadata()
        .and_then(Json::as_object)
        .and_then(|metadata| metadata.get(metadata_key))
        .and_then(Json::as_array)
        .is_some_and(|work_ids| {
            work_ids.iter().any(|work_id| {
                work_id
                    .as_str()
                    .is_some_and(|work_id| expected.iter().any(|expected| expected == work_id))
            })
        })
}

fn success_payload() -> Json {
    json!({
        "object": "list",
        "data": [
            {"object": "embedding", "embedding": [3.0, 4.0], "index": 0},
            {"object": "embedding", "embedding": [0.0, 2.0], "index": 1}
        ],
        "model": "embed-model",
        "usage": {
            "completion_tokens": 0,
            "completion_tokens_details": null,
            "prompt_tokens": 2,
            "prompt_tokens_details": {"cached_tokens": 0},
            "total_tokens": 2
        }
    })
}

fn one_vector_payload() -> Json {
    json!({
        "data": [{"object": "embedding", "embedding": [3.0, 4.0], "index": 0}]
    })
}

fn reset_runtime() {
    *global_context()
        .write()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = NemoRelayContextState::new();
}

#[derive(Debug)]
pub(crate) struct CapturedRequest {
    pub(crate) request_line: String,
    pub(crate) headers: BTreeMap<String, String>,
    pub(crate) body: Vec<u8>,
}

pub(crate) fn parse_request(request: &[u8]) -> CapturedRequest {
    let header_end = find_bytes(request, b"\r\n\r\n").unwrap();
    let headers = String::from_utf8(request[..header_end].to_vec()).unwrap();
    let mut lines = headers.lines();
    let request_line = lines.next().unwrap().to_string();
    let headers = lines
        .map(|line| {
            let (name, value) = line.split_once(':').unwrap();
            (name.to_ascii_lowercase(), value.trim().to_string())
        })
        .collect();
    CapturedRequest {
        request_line,
        headers,
        body: request[header_end + 4..].to_vec(),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn default_embeddings_endpoint_harness_is_loopback_and_credential_free() {
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let server = RawHttpServer::start(json_response(one_vector_payload()));
    assert!(server.address.ip().is_loopback());

    let config = router_config(server.base_url("127.0.0.1"), false, 500, 1, 1);
    assert!(!config.allow_remote_embedding_egress);
    assert!(config.embedders[0].api_key_env.is_none());
    let (clients, vector_space_id) = clients(&config, Err(())).unwrap();
    assert_eq!(clients.profile_count(), 1);
    assert_eq!(clients.space_count(), 1);
    assert_eq!(server.request_count(), 0);

    let vectors = clients
        .try_acquire(&vector_space_id)
        .unwrap()
        .execute_batch(
            EmbedderWorkKind::EmbeddingJob,
            vec![item("default safe endpoint probe", 'd')],
        )
        .await
        .unwrap();
    assert_eq!(vectors.len(), 1);
    assert_eq!(vectors[0].vector().values(), &[0.6, 0.8]);

    let request = parse_request(&server.wait_for_request());
    assert_eq!(request.request_line, "POST /v1/embeddings HTTP/1.1");
    assert!(!request.headers.contains_key("authorization"));
    assert_eq!(server.request_count(), 1);
}

#[test]
fn paid_embedding_probe_requires_double_opt_in_exact_destination_and_one_start() {
    let official = OFFICIAL_PAID_PROBE_BASE_URLS[0];
    assert_eq!(
        paid_probe_endpoint(None, Some(PAID_PROBE_COST_VALUE), Some(official)),
        Err(PAID_PROBE_RUN_ENV)
    );
    assert_eq!(
        paid_probe_endpoint(Some(PAID_PROBE_RUN_VALUE), None, Some(official)),
        Err(PAID_PROBE_COST_ENV)
    );
    for endpoint in [
        "http://api.openai.com/v1",
        "https://api.openai.com:443/v1",
        "https://api.openai.com/v1/",
        "https://api.openai.com/v1/embeddings",
        "https://api.openai.com/v1?destination=other",
        "https://example.com/v1",
    ] {
        assert_eq!(
            paid_probe_endpoint(
                Some(PAID_PROBE_RUN_VALUE),
                Some(PAID_PROBE_COST_VALUE),
                Some(endpoint),
            ),
            Err(PAID_PROBE_ENDPOINT_ENV)
        );
    }
    assert_eq!(
        paid_probe_endpoint(
            Some(PAID_PROBE_RUN_VALUE),
            Some(PAID_PROBE_COST_VALUE),
            Some(official),
        ),
        Ok(official)
    );

    let budget = PaidProbeRequestBudget::default();
    assert!(budget.begin().is_ok());
    assert!(budget.begin().is_err());
    assert_eq!(budget.starts(), PAID_PROBE_START_LIMIT);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires two explicit opt-ins and one paid OpenAI embeddings request"]
async fn paid_embeddings_endpoint_probe_is_bounded_and_secret_free() {
    let run_opt_in = std::env::var(PAID_PROBE_RUN_ENV).ok();
    let cost_opt_in = std::env::var(PAID_PROBE_COST_ENV).ok();
    let configured_endpoint = std::env::var(PAID_PROBE_ENDPOINT_ENV).ok();
    let endpoint = paid_probe_endpoint(
        run_opt_in.as_deref(),
        cost_opt_in.as_deref(),
        configured_endpoint.as_deref(),
    )
    .unwrap_or_else(|missing_or_invalid| {
        panic!(
            "paid embedding probe guard {missing_or_invalid} is missing or invalid; expected {PAID_PROBE_RUN_ENV}={PAID_PROBE_RUN_VALUE}, {PAID_PROBE_COST_ENV}={PAID_PROBE_COST_VALUE}, and {PAID_PROBE_ENDPOINT_ENV}={}",
            OFFICIAL_PAID_PROBE_BASE_URLS[0]
        )
    });
    let credential = Zeroizing::new(
        std::env::var(PAID_PROBE_API_KEY_ENV)
            .unwrap_or_else(|_| panic!("{PAID_PROBE_API_KEY_ENV} must be set after both opt-ins")),
    );
    assert!(
        credential.len() >= 16
            && credential.trim() == credential.as_str()
            && !credential.chars().any(char::is_control),
        "paid embedding credential must be at least 16 control-free bytes without surrounding whitespace"
    );

    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let mut config = router_config(endpoint.to_string(), true, 30_000, 1, 1);
    config.allow_remote_embedding_egress = true;
    let profile = &mut config.embedders[0];
    profile.api_key_env = Some(PAID_PROBE_API_KEY_ENV.to_string());
    profile.model = PAID_PROBE_MODEL.to_string();
    profile.provider_revision = "openai-text-embedding-3-small-default-v1".to_string();
    profile.dimensions = PAID_PROBE_DIMENSIONS;

    let (registry, vector_space_id) = prepared_registry(&config);
    let clients = build_frozen_embedder_clients_for_test(
        &config,
        &registry,
        |name| {
            assert_eq!(name, PAID_PROBE_API_KEY_ENV);
            Ok(credential.as_str().to_string())
        },
        |host, _| panic!("remote paid probe unexpectedly resolved loopback host {host}"),
    )
    .unwrap();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let event_sink = events.clone();
    register_subscriber(
        "router-paid-embedding-probe-events",
        Arc::new(move |event| event_sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let budget = PaidProbeRequestBudget::default();
    budget.begin().expect("paid probe request budget exhausted");
    let result = clients
        .try_acquire(&vector_space_id)
        .unwrap()
        .execute_batch(
            EmbedderWorkKind::EmbeddingJob,
            vec![item("synthetic paid endpoint probe", 'p')],
        )
        .await;
    flush_subscribers().unwrap();
    let captured_events = events.lock().unwrap().clone();
    deregister_subscriber("router-paid-embedding-probe-events").unwrap();

    assert_eq!(budget.starts(), PAID_PROBE_START_LIMIT);
    let credential_after = Zeroizing::new(
        std::env::var(PAID_PROBE_API_KEY_ENV)
            .expect("paid embedding credential disappeared during the probe"),
    );
    assert_eq!(credential_after.as_bytes(), credential.as_bytes());
    let inspected_surfaces = format!(
        "{}\n{config:?}\n{clients:?}\n{result:?}\n{captured_events:?}",
        serde_json::to_string(&config).unwrap()
    );
    assert!(
        !inspected_surfaces
            .as_bytes()
            .windows(credential.len())
            .any(|window| window == credential.as_bytes()),
        "paid embedding credential appeared in a post-run configuration, client, result, or event surface"
    );

    let vectors = result.expect("paid embedding probe should succeed");
    assert_eq!(vectors.len(), 1);
    assert_eq!(
        vectors[0].vector().values().len(),
        PAID_PROBE_DIMENSIONS as usize
    );
}

#[test]
fn construction_resolves_only_referenced_secrets_and_pins_localhost() {
    let server = RawHttpServer::start(json_response(success_payload()));
    let mut config = router_config(server.base_url("localhost"), true, 500, 1, 2);
    let mut unused = config.embedders[0].clone();
    unused.id = "unused-profile".to_string();
    unused.api_key_env = Some("MISSING_UNUSED_SECRET".to_string());
    config.embedders.push(unused);
    let (registry, vector_space_id) = prepared_registry(&config);
    let credential_reads = Arc::new(Mutex::new(Vec::new()));
    let reads = credential_reads.clone();
    let dns_reads = Arc::new(AtomicUsize::new(0));
    let dns = dns_reads.clone();
    let clients = build_frozen_embedder_clients_for_test(
        &config,
        &registry,
        move |name| {
            reads.lock().unwrap().push(name.to_string());
            (name == SECRET_ENV).then(|| SECRET.to_string()).ok_or(())
        },
        move |host, port| {
            assert_eq!(host, "localhost");
            dns.fetch_add(1, Ordering::AcqRel);
            Ok(vec![SocketAddr::new(Ipv4Addr::LOCALHOST.into(), port)])
        },
    )
    .unwrap();

    assert_eq!(clients.profile_count(), 1);
    assert_eq!(clients.space_count(), 1);
    assert_eq!(credential_reads.lock().unwrap().as_slice(), [SECRET_ENV]);
    assert_eq!(dns_reads.load(Ordering::Acquire), 1);
    assert_eq!(
        clients
            .try_acquire(&vector_space_id)
            .unwrap()
            .vector_space_id(),
        &vector_space_id
    );
    let debug = format!("{clients:?}");
    assert!(!debug.contains(SECRET));
    assert!(!debug.contains("MISSING_UNUSED_SECRET"));
}

#[test]
fn construction_rejects_missing_invalid_secret_and_unsafe_localhost_resolution() {
    let server = RawHttpServer::start(json_response(success_payload()));
    let config = router_config(server.base_url("localhost"), true, 500, 1, 2);
    assert!(matches!(
        clients(&config, Err(())),
        Err(EmbedderBuildFailure::MissingCredential)
    ));
    assert!(matches!(
        clients(&config, Ok("secret\nvalue")),
        Err(EmbedderBuildFailure::InvalidCredential)
    ));

    let (registry, _) = prepared_registry(&config);
    for addresses in [Vec::new(), vec!["192.0.2.1:80".parse().unwrap()]] {
        assert!(matches!(
            build_frozen_embedder_clients_for_test(
                &config,
                &registry,
                |_| Ok(SECRET.to_string()),
                move |_, _| Ok(addresses.clone()),
            ),
            Err(EmbedderBuildFailure::LocalhostResolution)
        ));
    }
}

#[test]
fn profile_permit_is_shared_and_saturation_is_retryable() {
    let server = RawHttpServer::start(json_response(success_payload()));
    let config = router_config(server.base_url("127.0.0.1"), false, 500, 1, 2);
    let (clients, vector_space_id) = clients(&config, Err(())).unwrap();
    let permit = clients.try_acquire(&vector_space_id).unwrap();
    let failure = clients.try_acquire(&vector_space_id).unwrap_err();
    assert_eq!(failure.class(), EmbedderFailureClass::Saturated);
    assert_eq!(failure.disposition(), EmbedderFailureDisposition::Retryable);
    drop(permit);
    assert!(clients.try_acquire(&vector_space_id).is_ok());
}

#[test]
fn activation_memory_budget_is_shared_across_provider_permits() {
    let server = RawHttpServer::start(json_response(success_payload()));
    let mut config = router_config(server.base_url("127.0.0.1"), false, 500, 1, 2);
    let (registry, _) = prepared_registry(&config);
    let reservation_bytes = usize::try_from(
        embedder_memory_reservation_bytes(registry.profiles.values().next().unwrap()).unwrap(),
    )
    .unwrap();
    let budget_permits = EMBEDDER_ACTIVATION_MEMORY_BYTES_MAX / reservation_bytes;
    assert!((1..crate::config::EMBEDDER_MAX_IN_FLIGHT).contains(&budget_permits));
    config.embedders[0].max_in_flight = budget_permits + 1;

    let (clients, vector_space_id) = clients(&config, Err(())).unwrap();
    let mut permits = (0..budget_permits)
        .map(|_| clients.try_acquire(&vector_space_id).unwrap())
        .collect::<Vec<_>>();
    let failure = clients.try_acquire(&vector_space_id).unwrap_err();
    assert_eq!(failure.class(), EmbedderFailureClass::Saturated);
    assert_eq!(failure.disposition(), EmbedderFailureDisposition::Retryable);

    drop(permits.pop());
    assert!(clients.try_acquire(&vector_space_id).is_ok());
}

#[test]
fn activation_memory_budget_is_shared_across_referenced_profiles() {
    let server = RawHttpServer::start(json_response(success_payload()));
    let mut config = router_config(server.base_url("127.0.0.1"), false, 500, 2, 2);
    let mut second_profile = config.embedders[0].clone();
    second_profile.id = "embedding-second".to_string();
    config.embedders.push(second_profile);
    let mut second_pool = config.pools[0].clone();
    second_pool.id = "pool-b".to_string();
    second_pool.learning.as_mut().unwrap().embedder = "embedding-second".to_string();
    config.pools.push(second_pool);

    let (registry, _) = prepared_registry(&config);
    let first_space = registry.mappings["pool-a"].vector_space_id.clone();
    let second_space = registry.mappings["pool-b"].vector_space_id.clone();
    assert_ne!(first_space, second_space);
    let reservation_bytes = usize::try_from(
        embedder_memory_reservation_bytes(
            registry
                .profiles
                .values()
                .find(|profile| profile.profile_id == "embedding-main")
                .unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    assert!(reservation_bytes * 2 <= EMBEDDER_ACTIVATION_MEMORY_BYTES_MAX);
    assert!(reservation_bytes * 3 > EMBEDDER_ACTIVATION_MEMORY_BYTES_MAX);
    let clients = build_frozen_embedder_clients_for_test(
        &config,
        &registry,
        |_| Err(()),
        |_host, port| Ok(vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)]),
    )
    .unwrap();

    let first = clients.try_acquire(&first_space).unwrap();
    let second = clients.try_acquire(&second_space).unwrap();
    let failure = clients.try_acquire(&first_space).unwrap_err();
    assert_eq!(failure.class(), EmbedderFailureClass::Saturated);
    assert_eq!(failure.disposition(), EmbedderFailureDisposition::Retryable);

    drop(second);
    assert!(clients.try_acquire(&first_space).is_ok());
    drop(first);
}

#[test]
fn maximum_valid_profile_reservation_is_exact_and_admissible() {
    let server = RawHttpServer::start(json_response(success_payload()));
    let mut config = router_config(server.base_url("127.0.0.1"), false, 500, 1, 128);
    config.embedders[0].dimensions = 8_192;
    let (registry, vector_space_id) = prepared_registry(&config);
    let profile = registry.profiles.values().next().unwrap();
    let components = profile.batch_size * profile.dimensions.as_usize();
    let expected = EMBEDDER_REQUEST_BYTES_MAX * 3
        + embedder_response_bytes_bound(profile.batch_size, profile.dimensions.value()).unwrap()
        + components * (std::mem::size_of::<f64>() + 2 * std::mem::size_of::<f32>());
    let reservation = usize::try_from(embedder_memory_reservation_bytes(profile).unwrap()).unwrap();
    assert_eq!(reservation, expected);
    assert!(reservation <= EMBEDDER_ACTIVATION_MEMORY_BYTES_MAX);

    let clients = build_frozen_embedder_clients_for_test(
        &config,
        &registry,
        |_| Err(()),
        |_host, port| Ok(vec![SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), port)]),
    )
    .unwrap();
    assert!(clients.try_acquire(&vector_space_id).is_ok());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn exact_wire_response_order_and_fresh_embedder_scope_are_preserved() {
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let server = RawHttpServer::start(chunked_json_response(success_payload()));
    let config = router_config(server.base_url("localhost"), true, 500, 2, 2);
    let (clients, vector_space_id) = clients(&config, Ok(SECRET)).unwrap();
    let items = vec![item("alpha", 'a'), item("beta", 'b')];
    let query_hashes = items
        .iter()
        .map(|item| item.canonical_query_hash().to_string())
        .collect::<Vec<_>>();
    let work_ids = items
        .iter()
        .map(|item| item.work_id().to_string())
        .collect::<Vec<_>>();

    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let event_sink = events.clone();
    register_subscriber(
        "router-task9-embedder-events",
        Arc::new(move |event| event_sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    let outer_stack = create_scope_stack();
    let outer_root = outer_stack.read().unwrap().top().uuid;
    let (outer_uuid, vectors) = TASK_SCOPE_STACK
        .scope(outer_stack, async {
            let outer = push_scope(
                PushScopeParams::builder()
                    .name("outer-agent")
                    .scope_type(ScopeType::Agent)
                    .build(),
            )
            .unwrap();
            let vectors = clients
                .try_acquire(&vector_space_id)
                .unwrap()
                .execute_batch(EmbedderWorkKind::EmbeddingJob, items)
                .await
                .unwrap();
            assert_eq!(task_scope_top().uuid, outer.uuid);
            let outer_uuid = outer.uuid;
            pop_scope(PopScopeParams::builder().handle_uuid(&outer.uuid).build()).unwrap();
            (outer_uuid, vectors)
        })
        .await;
    flush_subscribers().unwrap();

    assert_eq!(vectors.len(), 2);
    assert_eq!(vectors[0].vector_space_id(), &vector_space_id);
    assert_eq!(vectors[0].vector().values(), &[0.6, 0.8]);
    assert_eq!(vectors[1].vector().values(), &[0.0, 1.0]);

    let request = parse_request(&server.wait_for_request());
    assert_eq!(request.request_line, "POST /v1/embeddings HTTP/1.1");
    assert_eq!(request.headers["authorization"], format!("Bearer {SECRET}"));
    assert_eq!(request.headers["content-type"], "application/json");
    assert_eq!(request.headers["accept"], "application/json");
    let expected_body = canonical_json(&json!({
        "model": "embed-model",
        "input": ["{\"task\":\"alpha\"}", "{\"task\":\"beta\"}"],
        "encoding_format": "float"
    }))
    .unwrap();
    assert_eq!(request.body, expected_body.as_bytes());
    let request_json: Json = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(request_json.as_object().unwrap().len(), 3);
    assert!(request_json.get("dimensions").is_none());

    let events = events.lock().unwrap();
    let embedder = events
        .iter()
        .filter(|event| {
            event.name() == EMBEDDER_SCOPE_NAME
                && event_has_any_work_id(event, "embedding_job_ids", &work_ids)
        })
        .collect::<Vec<_>>();
    assert_eq!(embedder.len(), 2);
    assert_eq!(embedder[0].scope_category(), Some(ScopeCategory::Start));
    assert_eq!(embedder[1].scope_category(), Some(ScopeCategory::End));
    assert_eq!(embedder[0].uuid(), embedder[1].uuid());
    assert_eq!(embedder[0].scope_type(), Some(ScopeType::Embedder));
    assert_eq!(embedder[1].scope_type(), Some(ScopeType::Embedder));
    assert_eq!(embedder[0].metadata(), embedder[1].metadata());
    assert!(embedder[0].category_profile().is_none());
    assert!(embedder[0].data().is_none());
    let parent = embedder[0].parent_uuid().unwrap();
    assert_ne!(parent, outer_uuid);
    assert_ne!(parent, outer_root);
    assert!(
        !events
            .iter()
            .any(|event| event.parent_uuid() == Some(embedder[0].uuid()))
    );
    assert_eq!(
        embedder[0].metadata(),
        Some(&json!({
            "profile_id": "embedding-main",
            "vector_space_id": vector_space_id.as_str(),
            "canonical_query_hashes": query_hashes,
            "embedding_job_ids": work_ids,
        }))
    );
    drop(events);
    deregister_subscriber("router-task9-embedder-events").unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn redirects_are_not_followed_and_authorization_stays_on_the_frozen_origin() {
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let target = RawHttpServer::start(json_response(success_payload()));
    let location = format!("http://{}/stolen", target.address);
    let source = RawHttpServer::start(response("302 Found", &[("Location", location)], &[]));
    let config = router_config(source.base_url("127.0.0.1"), true, 500, 1, 2);
    let (clients, vector_space_id) = clients(&config, Ok(SECRET)).unwrap();
    let failure = clients
        .try_acquire(&vector_space_id)
        .unwrap()
        .execute_batch(EmbedderWorkKind::EmbeddingJob, vec![item("alpha", 'a')])
        .await
        .unwrap_err();
    assert_eq!(failure.class(), EmbedderFailureClass::UnexpectedStatus);
    assert_eq!(
        parse_request(&source.wait_for_request()).headers["authorization"],
        format!("Bearer {SECRET}")
    );
    thread::sleep(Duration::from_millis(50));
    assert_eq!(target.request_count(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn status_classes_are_stable_redacted_and_have_exact_retry_policy() {
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    for (status, class, disposition) in [
        (
            "401 Unauthorized",
            EmbedderFailureClass::Authentication,
            EmbedderFailureDisposition::Quarantine,
        ),
        (
            "408 Request Timeout",
            EmbedderFailureClass::RequestTimeout,
            EmbedderFailureDisposition::Retryable,
        ),
        (
            "429 Too Many Requests",
            EmbedderFailureClass::RateLimited,
            EmbedderFailureDisposition::Retryable,
        ),
        (
            "400 Bad Request",
            EmbedderFailureClass::ClientStatus,
            EmbedderFailureDisposition::Quarantine,
        ),
        (
            "503 Service Unavailable",
            EmbedderFailureClass::Server,
            EmbedderFailureDisposition::Retryable,
        ),
    ] {
        let body = format!("provider-body-{SECRET}");
        let server = RawHttpServer::start(response(
            status,
            &[("Content-Length", body.len().to_string())],
            body.as_bytes(),
        ));
        let config = router_config(server.base_url("127.0.0.1"), true, 500, 1, 2);
        let (clients, vector_space_id) = clients(&config, Ok(SECRET)).unwrap();
        let failure = clients
            .try_acquire(&vector_space_id)
            .unwrap()
            .execute_batch(EmbedderWorkKind::EmbeddingJob, vec![item("alpha", 'a')])
            .await
            .unwrap_err();
        assert_eq!(failure.class(), class);
        assert_eq!(failure.disposition(), disposition);
        assert!(!format!("{failure:?} {failure}").contains(SECRET));
        assert!(!format!("{clients:?}").contains(SECRET));
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn truncated_response_stream_is_retryable_transport_and_redacted() {
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let payload = format!("truncated-provider-payload-{SECRET}");
    let server = RawHttpServer::start(response(
        "200 OK",
        &[
            ("Content-Type", "application/json".to_string()),
            ("Content-Length", (payload.len() + 256).to_string()),
        ],
        payload.as_bytes(),
    ));
    let config = router_config(server.base_url("127.0.0.1"), true, 500, 1, 1);
    let (clients, vector_space_id) = clients(&config, Ok(SECRET)).unwrap();

    let failure = clients
        .try_acquire(&vector_space_id)
        .unwrap()
        .execute_batch(EmbedderWorkKind::EmbeddingJob, vec![item("alpha", 'a')])
        .await
        .unwrap_err();

    assert_eq!(failure.class(), EmbedderFailureClass::Transport);
    assert_eq!(failure.disposition(), EmbedderFailureDisposition::Retryable);
    let inspected = format!("{failure:?} {failure} {clients:?}");
    assert!(!inspected.contains(&payload));
    assert!(!inspected.contains(SECRET));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn strict_response_failures_cover_shape_count_index_dimensions_and_numeric_values() {
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let cases = [
        (
            b"not-json".to_vec(),
            EmbedderFailureClass::MalformedResponse,
        ),
        (
            serde_json::to_vec(&json!({"data": [], "unexpected": true})).unwrap(),
            EmbedderFailureClass::MalformedResponse,
        ),
        (
            serde_json::to_vec(&json!({
                "data": [{"embedding": [1.0, 2.0], "index": 0}],
                "usage": {
                    "completion_tokens": 2,
                    "prompt_tokens": 1,
                    "total_tokens": 1
                }
            }))
            .unwrap(),
            EmbedderFailureClass::MalformedResponse,
        ),
        (
            serde_json::to_vec(&json!({"data": []})).unwrap(),
            EmbedderFailureClass::ResponseCount,
        ),
        (
            serde_json::to_vec(&json!({"data": [{"embedding": [1.0, 2.0], "index": 1}]})).unwrap(),
            EmbedderFailureClass::ResponseIndex,
        ),
        (
            serde_json::to_vec(&json!({"data": [{"embedding": [1.0], "index": 0}]})).unwrap(),
            EmbedderFailureClass::DimensionMismatch,
        ),
        (
            br#"{"data":[{"embedding":[3.5e38,1.0],"index":0}]}"#.to_vec(),
            EmbedderFailureClass::NonFiniteVector,
        ),
        (
            serde_json::to_vec(&json!({"data": [{"embedding": [0.0, 0.0], "index": 0}]})).unwrap(),
            EmbedderFailureClass::ZeroVector,
        ),
    ];
    for (body, expected) in cases {
        let server = RawHttpServer::start(response(
            "200 OK",
            &[("Content-Length", body.len().to_string())],
            &body,
        ));
        let config = router_config(server.base_url("127.0.0.1"), false, 500, 1, 2);
        let (clients, vector_space_id) = clients(&config, Err(())).unwrap();
        let failure = clients
            .try_acquire(&vector_space_id)
            .unwrap()
            .execute_batch(EmbedderWorkKind::EmbeddingJob, vec![item("alpha", 'a')])
            .await
            .unwrap_err();
        assert_eq!(failure.class(), expected);
        assert_eq!(
            failure.disposition(),
            EmbedderFailureDisposition::Quarantine
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn response_content_length_and_operation_timeout_are_bounded_with_scope_pairs() {
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let oversized = RawHttpServer::start(response(
        "200 OK",
        &[(
            "Content-Length",
            (embedder_response_bytes_bound(1, 2).unwrap() as u64 + 1).to_string(),
        )],
        &[],
    ));
    let oversized_config = router_config(oversized.base_url("127.0.0.1"), false, 500, 1, 2);
    let (oversized_clients, vector_space_id) = clients(&oversized_config, Err(())).unwrap();
    assert_eq!(
        oversized_clients
            .try_acquire(&vector_space_id)
            .unwrap()
            .execute_batch(EmbedderWorkKind::EmbeddingJob, vec![item("alpha", 'a')])
            .await
            .unwrap_err()
            .class(),
        EmbedderFailureClass::ResponseBound
    );

    let timeout = RawHttpServer::start_with_delay(
        json_response(json!({
            "data": [{"embedding": [1.0, 2.0], "index": 0}]
        })),
        Duration::from_millis(100),
    );
    let timeout_config = router_config(timeout.base_url("127.0.0.1"), false, 20, 1, 2);
    let (clients, vector_space_id) = clients(&timeout_config, Err(())).unwrap();
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let event_sink = events.clone();
    register_subscriber(
        "router-task9-timeout-events",
        Arc::new(move |event| event_sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();
    let failure = clients
        .try_acquire(&vector_space_id)
        .unwrap()
        .execute_batch(EmbedderWorkKind::Decision, vec![item("alpha", 'a')])
        .await
        .unwrap_err();
    assert_eq!(failure.class(), EmbedderFailureClass::Timeout);
    assert_eq!(failure.disposition(), EmbedderFailureDisposition::Retryable);
    flush_subscribers().unwrap();
    let events = events.lock().unwrap();
    let timeout_work_ids = ["a".repeat(64)];
    let embedder = events
        .iter()
        .filter(|event| {
            event.name() == EMBEDDER_SCOPE_NAME
                && event_has_any_work_id(event, "decision_ids", &timeout_work_ids)
        })
        .collect::<Vec<_>>();
    assert_eq!(embedder.len(), 2);
    assert_eq!(embedder[0].scope_category(), Some(ScopeCategory::Start));
    assert_eq!(embedder[1].scope_category(), Some(ScopeCategory::End));
    let metadata = embedder[0].metadata().unwrap().as_object().unwrap();
    assert!(metadata.contains_key("decision_ids"));
    assert!(!metadata.contains_key("embedding_job_ids"));
    drop(events);
    deregister_subscriber("router-task9-timeout-events").unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn supervisor_cancellation_and_task_abort_preserve_embedder_scope_pairs() {
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let server = RawHttpServer::start_with_delay(
        json_response(json!({
            "data": [{"embedding": [1.0, 2.0], "index": 0}]
        })),
        Duration::from_millis(250),
    );
    let config = router_config(server.base_url("127.0.0.1"), false, 1_000, 1, 2);
    let (clients, vector_space_id) = clients(&config, Err(())).unwrap();
    let clients = Arc::new(clients);
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let event_sink = events.clone();
    register_subscriber(
        "router-task10-cancel-events",
        Arc::new(move |event| event_sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let (cancel, cancel_rx) = tokio::sync::watch::channel(false);
    let cancel_clients = clients.clone();
    let cancel_space = vector_space_id.clone();
    let cancelled = tokio::spawn(async move {
        cancel_clients
            .try_acquire(&cancel_space)
            .unwrap()
            .execute_batch_until(
                EmbedderWorkKind::EmbeddingJob,
                vec![item("cancelled", 'c')],
                Instant::now() + Duration::from_secs(1),
                cancel_rx,
            )
            .await
    });
    while server.request_count() == 0 {
        tokio::task::yield_now().await;
    }
    cancel.send_replace(true);
    assert_eq!(
        cancelled.await.unwrap().unwrap_err().class(),
        EmbedderFailureClass::Cancelled
    );

    let (_abort_cancel, abort_rx) = tokio::sync::watch::channel(false);
    let abort_clients = clients.clone();
    let abort_space = vector_space_id.clone();
    let aborted = tokio::spawn(async move {
        abort_clients
            .try_acquire(&abort_space)
            .unwrap()
            .execute_batch_until(
                EmbedderWorkKind::EmbeddingJob,
                vec![item("aborted", 'd')],
                Instant::now() + Duration::from_secs(1),
                abort_rx,
            )
            .await
    });
    while server.request_count() < 2 {
        tokio::task::yield_now().await;
    }
    aborted.abort();
    assert!(aborted.await.unwrap_err().is_cancelled());

    flush_subscribers().unwrap();
    let events = events.lock().unwrap();
    let cancellation_work_ids = ["c".repeat(64), "d".repeat(64)];
    let embedder = events
        .iter()
        .filter(|event| {
            event.name() == EMBEDDER_SCOPE_NAME
                && event_has_any_work_id(event, "embedding_job_ids", &cancellation_work_ids)
        })
        .collect::<Vec<_>>();
    assert_eq!(embedder.len(), 4);
    for pair in embedder.chunks_exact(2) {
        assert_eq!(pair[0].scope_category(), Some(ScopeCategory::Start));
        assert_eq!(pair[1].scope_category(), Some(ScopeCategory::End));
        assert_eq!(pair[0].uuid(), pair[1].uuid());
    }
    drop(events);
    deregister_subscriber("router-task10-cancel-events").unwrap();
}

#[test]
fn batch_items_reject_noncanonical_hashes_text_and_unsafe_work_ids() {
    let canonical = canonical_json(&json!({"task": "alpha"})).unwrap();
    assert!(EmbedderBatchItem::new("f".repeat(64), canonical.clone(), "a".repeat(64)).is_err());
    assert!(
        EmbedderBatchItem::new(
            sha256_hex(b"{ \"task\": \"alpha\" }"),
            "{ \"task\": \"alpha\" }",
            "a".repeat(64),
        )
        .is_err()
    );
    assert!(
        EmbedderBatchItem::new(
            sha256_hex(canonical.as_bytes()),
            canonical,
            "unsafe/work-id",
        )
        .is_err()
    );
}

#[test]
fn canonical_request_body_accepts_the_exact_limit_and_rejects_one_byte_over() {
    let server = RawHttpServer::start(json_response(one_vector_payload()));
    let config = router_config(server.base_url("127.0.0.1"), false, 500, 1, 2);
    let (registry, _) = prepared_registry(&config);
    let profile = registry.profiles.values().next().unwrap();
    let empty = item("", 'a');
    let fixed_bytes = request_body(profile, &[empty]).unwrap().len();
    let payload_bytes = crate::config::EMBEDDER_REQUEST_BYTES_MAX - fixed_bytes;

    let exact = item(&"x".repeat(payload_bytes), 'a');
    assert_eq!(
        request_body(profile, &[exact]).unwrap().len(),
        crate::config::EMBEDDER_REQUEST_BYTES_MAX
    );
    let over = item(&"x".repeat(payload_bytes + 1), 'a');
    assert_eq!(
        request_body(profile, &[over]).unwrap_err().class(),
        EmbedderFailureClass::RequestBound
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn chunked_response_over_the_stream_limit_is_rejected() {
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let oversized = vec![b'x'; embedder_response_bytes_bound(1, 2).unwrap() + 1];
    let mut chunked = format!("{:x}\r\n", oversized.len()).into_bytes();
    chunked.extend_from_slice(&oversized);
    chunked.extend_from_slice(b"\r\n0\r\n\r\n");
    let server = RawHttpServer::start(response(
        "200 OK",
        &[("Transfer-Encoding", "chunked".to_string())],
        &chunked,
    ));
    let config = router_config(server.base_url("127.0.0.1"), false, 2_000, 1, 2);
    let (clients, vector_space_id) = clients(&config, Err(())).unwrap();
    assert_eq!(
        clients
            .try_acquire(&vector_space_id)
            .unwrap()
            .execute_batch(EmbedderWorkKind::EmbeddingJob, vec![item("alpha", 'a')])
            .await
            .unwrap_err()
            .class(),
        EmbedderFailureClass::ResponseBound
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ipv6_literal_uses_the_frozen_path_and_sends_no_implicit_authorization() {
    let listener = match TcpListener::bind("[::1]:0") {
        Ok(listener) => listener,
        Err(_) => return,
    };
    let server = RawHttpServer::from_listener(
        listener,
        json_response(one_vector_payload()),
        Duration::ZERO,
    );
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let config = router_config(server.base_url("[::1]"), false, 500, 1, 2);
    let (clients, vector_space_id) = clients(&config, Err(())).unwrap();
    let vectors = clients
        .try_acquire(&vector_space_id)
        .unwrap()
        .execute_batch(EmbedderWorkKind::EmbeddingJob, vec![item("alpha", 'a')])
        .await
        .unwrap();
    assert_eq!(vectors.len(), 1);
    let request = parse_request(&server.wait_for_request());
    assert_eq!(request.request_line, "POST /v1/embeddings HTTP/1.1");
    assert!(!request.headers.contains_key("authorization"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_requests_have_distinct_embedder_scopes_and_fresh_roots() {
    let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
    reset_runtime();
    let server = RawHttpServer::start(json_response(one_vector_payload()));
    let config = router_config(server.base_url("127.0.0.1"), false, 500, 2, 2);
    let (clients, vector_space_id) = clients(&config, Err(())).unwrap();
    let clients = Arc::new(clients);
    let events = Arc::new(Mutex::new(Vec::<Event>::new()));
    let event_sink = events.clone();
    register_subscriber(
        "router-task9-concurrent-events",
        Arc::new(move |event| event_sink.lock().unwrap().push(event.clone())),
    )
    .unwrap();

    let first = {
        let clients = clients.clone();
        let vector_space_id = vector_space_id.clone();
        tokio::spawn(async move {
            clients
                .try_acquire(&vector_space_id)
                .unwrap()
                .execute_batch(EmbedderWorkKind::EmbeddingJob, vec![item("first", 'a')])
                .await
        })
    };
    let second = {
        let clients = clients.clone();
        let vector_space_id = vector_space_id.clone();
        tokio::spawn(async move {
            clients
                .try_acquire(&vector_space_id)
                .unwrap()
                .execute_batch(EmbedderWorkKind::EmbeddingJob, vec![item("second", 'b')])
                .await
        })
    };
    assert!(first.await.unwrap().is_ok());
    assert!(second.await.unwrap().is_ok());
    flush_subscribers().unwrap();
    let events = events.lock().unwrap();
    let concurrent_work_ids = ["a".repeat(64), "b".repeat(64)];
    let starts = events
        .iter()
        .filter(|event| {
            event.name() == EMBEDDER_SCOPE_NAME
                && event.scope_category() == Some(ScopeCategory::Start)
                && event_has_any_work_id(event, "embedding_job_ids", &concurrent_work_ids)
        })
        .collect::<Vec<_>>();
    assert_eq!(starts.len(), 2);
    assert_ne!(starts[0].uuid(), starts[1].uuid());
    assert_ne!(starts[0].parent_uuid(), starts[1].parent_uuid());
    assert_eq!(
        events
            .iter()
            .filter(|event| {
                event.name() == EMBEDDER_SCOPE_NAME
                    && event_has_any_work_id(event, "embedding_job_ids", &concurrent_work_ids)
            })
            .count(),
        4
    );
    drop(events);
    deregister_subscriber("router-task9-concurrent-events").unwrap();
}

#[test]
fn profile_config_remains_secret_name_only() {
    let server = RawHttpServer::start(json_response(success_payload()));
    let config = router_config(server.base_url("127.0.0.1"), true, 500, 1, 2);
    let serialized = serde_json::to_string(&config).unwrap();
    assert!(serialized.contains(SECRET_ENV));
    assert!(!serialized.contains(SECRET));
    let profile: &EmbedderConfig = &config.embedders[0];
    assert!(!format!("{profile:?}").contains(SECRET));
}
