// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::io::{ErrorKind, Read, Write};
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use chrono::Utc;
use nemo_relay_types::api::llm::LlmApiFamily;
use rusqlite::{Connection, OptionalExtension, params};
use serde_json::{Value as Json, json};
use tempfile::{TempDir, tempdir};
use uuid::Uuid;

use crate::background_jobs::{
    BackgroundJobOutcome, execute_embedding_batch, prepare_embedding_batch,
};
use crate::canonical_json::canonical_sha256;
use crate::canonical_query::build_canonical_routing_query;
use crate::config::RouterConfig;
use crate::embedder::{FrozenEmbedderClients, build_frozen_embedder_clients};
use crate::ledger::read_pool::LedgerReadPool;
use crate::ledger::repository::LedgerRepository;
use crate::ledger::repository::embedding::{
    EmbeddingJobBatchClaim, EmbeddingJobBatchClaimAck, EmbeddingJobBatchClaimItem,
    EmbeddingJobBatchCompletion, EmbeddingJobBatchCompletionAck, EmbeddingJobBatchCompletionItem,
    EmbeddingJobResolution, EmbeddingJobResolutionAck, EmbeddingJobResolutionKind,
    LiveEmbeddingPrepare, LiveEmbeddingPrepareAck, load_verified_embedding_job,
};
use crate::ledger::repository::process::{ProcessCommandAck, ProcessStop};
use crate::ledger::repository::vector_catalog::{
    EmbeddingCacheSource, EmbeddingCacheUpsertAck, EmbeddingCacheWrite, ensure_canonical_query,
    load_embedding_cache, upsert_embedding_cache,
};
use crate::ledger::repository::vector_index::{
    GenerationAuthorizationAck, GenerationObjectCreationAck, RebuildFlipAck, RebuildLeaseClaimAck,
    RebuildStepAck,
};
use crate::ledger::repository::vector_registry::{FrozenMappingKey, VectorRegistryEnsure};
use crate::ledger::repository::vector_search::LiveProjectedNeighborSearch;
use crate::ledger::writer::{LedgerWriterClient, LedgerWriterOwner};
use crate::live_embedding::{LiveEmbeddingResult, LiveEmbeddingService};
use crate::projection::{
    REQUEST_PROJECTION_SCHEMA_V1, ROUTER_SANITIZER_VERSION, ROUTING_CONTEXT_SCHEMA_V1,
    RouterRequestProjectionV1, RouterRoutingContextProjectionV1, SanitizedAnnotatedLlmRequest,
    SanitizedMessage, SanitizedMessageContent,
};
use crate::provider_admission::ProviderAdmissionGate;
use crate::routing_partition::{
    RoutingPartitionBaseV1, RoutingPartitionInputV1, build_routing_partition_from_input_v1,
};
use crate::sqlite_vector_store::{SqliteVecStore, VectorIndexWriterAck, VectorIndexWriterCommand};
use crate::vector::{AuthoritativeVector, NormalizedVector, VectorDimensions, VectorSpaceId};

struct LoopbackEmbeddingServer {
    address: SocketAddr,
    requests: Arc<AtomicUsize>,
    response: Arc<Mutex<Vec<u8>>>,
    hold_response: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl LoopbackEmbeddingServer {
    fn start(response: Vec<u8>, delay: Duration) -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let requests = Arc::new(AtomicUsize::new(0));
        let response = Arc::new(Mutex::new(response));
        let hold_response = Arc::new(AtomicBool::new(false));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_requests = requests.clone();
        let thread_response = response.clone();
        let thread_hold_response = hold_response.clone();
        let thread_stop = stop.clone();
        let thread = thread::spawn(move || {
            while !thread_stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((mut stream, _)) => {
                        stream.set_nonblocking(false).unwrap();
                        if read_http_request(&mut stream).is_empty() {
                            continue;
                        }
                        thread_requests.fetch_add(1, Ordering::AcqRel);
                        if !delay.is_zero() {
                            thread::sleep(delay);
                        }
                        while thread_hold_response.load(Ordering::Acquire)
                            && !thread_stop.load(Ordering::Acquire)
                        {
                            thread::sleep(Duration::from_millis(1));
                        }
                        let response = thread_response.lock().unwrap().clone();
                        let _ = stream.write_all(&response);
                        let _ = stream.flush();
                    }
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(_) => break,
                }
            }
        });
        Self {
            address,
            requests,
            response,
            hold_response,
            stop,
            thread: Some(thread),
        }
    }

    fn endpoint(&self) -> String {
        format!("http://{}/v1", self.address)
    }

    fn request_count(&self) -> usize {
        self.requests.load(Ordering::Acquire)
    }

    fn set_response(&self, response: Vec<u8>) {
        *self.response.lock().unwrap() = response;
    }

    fn hold_responses(&self) {
        self.hold_response.store(true, Ordering::Release);
    }

    fn release_responses(&self) {
        self.hold_response.store(false, Ordering::Release);
    }
}

impl Drop for LoopbackEmbeddingServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = TcpStream::connect_timeout(&self.address, Duration::from_millis(50));
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

struct FixtureOptions {
    response: Vec<u8>,
    response_delay: Duration,
    provider_timeout_ms: u64,
    active_index: bool,
    seed_cache: bool,
    max_in_flight: usize,
}

impl FixtureOptions {
    fn success() -> Self {
        Self {
            response: json_response(json!({
                "data": [{"object": "embedding", "embedding": [3.0, 4.0], "index": 0}]
            })),
            response_delay: Duration::ZERO,
            provider_timeout_ms: 2_000,
            active_index: true,
            seed_cache: false,
            max_in_flight: 2,
        }
    }
}

struct LiveServiceFixture {
    _temporary: TempDir,
    database_path: PathBuf,
    config: RouterConfig,
    registry: Arc<VectorRegistryEnsure>,
    server: LoopbackEmbeddingServer,
    service: LiveEmbeddingService,
    writer: LedgerWriterClient,
    writer_owner: LedgerWriterOwner,
    read_pool: LedgerReadPool,
    clients: Arc<FrozenEmbedderClients>,
    gate: ProviderAdmissionGate,
    project_uuid: Uuid,
    learning_generation_id: Uuid,
    vector_space_id: VectorSpaceId,
    query_hash: String,
    request: RouterRequestProjectionV1,
    routing: RouterRoutingContextProjectionV1,
}

impl LiveServiceFixture {
    async fn start(options: FixtureOptions) -> Self {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let server = LoopbackEmbeddingServer::start(options.response, options.response_delay);
        let config = live_config(
            &database_path,
            &server.endpoint(),
            options.provider_timeout_ms,
            options.max_in_flight,
        );
        let activation_at = Utc::now().timestamp_millis().saturating_sub(1_000);
        let activated = LedgerRepository::activate_at(&config, activation_at).unwrap();
        let project_uuid = activated.identity.project_uuid;
        let learning_generation_id = activated.identity.pools["pool-a"].learning_generation_id;
        let vector_space_id = activated.registry.mappings["pool-a"]
            .vector_space_id
            .clone();
        let dimensions = activated.registry.spaces[&vector_space_id].dimensions;
        let request = request_projection();
        let routing = routing_projection();
        let artifact =
            build_canonical_routing_query(&request, &routing, &config.pools[0].canonicalizer)
                .unwrap();
        let query_hash = artifact.canonical_query_hash.clone();
        if options.seed_cache {
            let mut connection = Connection::open(&database_path).unwrap();
            let transaction = connection.transaction().unwrap();
            ensure_canonical_query(&transaction, &artifact, activation_at + 1).unwrap();
            let vector = expected_vector(&vector_space_id, dimensions);
            assert!(matches!(
                upsert_embedding_cache(
                    &transaction,
                    &EmbeddingCacheWrite {
                        embedding_id: Uuid::now_v7(),
                        project_uuid,
                        vector_space_id: vector_space_id.clone(),
                        canonical_query_hash: query_hash.clone(),
                        content_hash: query_hash.clone(),
                        vector,
                        source: EmbeddingCacheSource::Cache,
                        created_at_unix_ms: activation_at + 1,
                    },
                )
                .unwrap(),
                EmbeddingCacheUpsertAck::Applied(_)
            ));
            transaction.commit().unwrap();
        }
        let registry = Arc::new(activated.registry);
        let clients = Arc::new(build_frozen_embedder_clients(&config, &registry).unwrap());
        let (writer_owner, writer) = LedgerWriterOwner::start(activated.repository, 32).unwrap();
        if options.active_index {
            activate_empty_generation(&writer, &vector_space_id, dimensions).await;
        }
        let read_pool = LedgerReadPool::open(&database_path).unwrap();
        let gate = ProviderAdmissionGate::initially_open_for_test();
        let service = LiveEmbeddingService::new(
            &config,
            registry.clone(),
            writer.clone(),
            read_pool.clone(),
            clients.clone(),
            gate.clone(),
        )
        .unwrap();
        Self {
            _temporary: temporary,
            database_path,
            config,
            registry,
            server,
            service,
            writer,
            writer_owner,
            read_pool,
            clients,
            gate,
            project_uuid,
            learning_generation_id,
            vector_space_id,
            query_hash,
            request,
            routing,
        }
    }

    async fn embed(&self, pool_id: &str, timeout: Duration) -> LiveEmbeddingResult {
        self.service
            .embed_until(
                pool_id,
                &self.request,
                &self.routing,
                Instant::now() + timeout,
            )
            .await
    }

    fn job_state(&self) -> Option<(i64, String, Option<String>)> {
        Connection::open(&self.database_path)
            .unwrap()
            .query_row(
                "SELECT job.attempt_count,
                        (SELECT state FROM embedding_job_state_events
                         WHERE embedding_job_id = job.embedding_job_id
                         ORDER BY event_seq DESC LIMIT 1),
                        job.terminal_error_class
                 FROM embedding_jobs AS job
                 WHERE job.canonical_query_hash = ?1",
                params![self.query_hash],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .optional()
            .unwrap()
    }

    fn claim_graph_state(&self) -> Option<(i64, Option<String>, String, i64, i64)> {
        Connection::open(&self.database_path)
            .unwrap()
            .query_row(
                "SELECT job.attempt_count, job.lease_token,
                        (SELECT state FROM embedding_job_state_events
                         WHERE embedding_job_id = job.embedding_job_id
                         ORDER BY event_seq DESC LIMIT 1),
                        (SELECT count(*) FROM embedding_job_state_events
                         WHERE embedding_job_id = job.embedding_job_id),
                        (SELECT count(*) FROM embeddings)
                 FROM embedding_jobs AS job
                 WHERE job.canonical_query_hash = ?1",
                params![self.query_hash],
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
            .optional()
            .unwrap()
    }
}

impl Drop for LiveServiceFixture {
    fn drop(&mut self) {
        self.service.close();
        self.read_pool.abort();
        self.writer_owner.abort();
    }
}

async fn activate_empty_generation(
    writer: &LedgerWriterClient,
    vector_space_id: &VectorSpaceId,
    dimensions: VectorDimensions,
) {
    let observed_at = Utc::now().timestamp_millis();
    let deadline = Instant::now() + Duration::from_secs(3);
    assert!(matches!(
        writer
            .vector_index_until(
                VectorIndexWriterCommand::AuthorizeGeneration {
                    vector_space_id: vector_space_id.clone(),
                    dimensions,
                    created_at_unix_ms: observed_at,
                },
                deadline,
            )
            .await
            .unwrap(),
        VectorIndexWriterAck::GenerationAuthorized(acknowledgement)
            if matches!(*acknowledgement, GenerationAuthorizationAck::Created(_))
    ));
    let fence = match writer
        .vector_index_until(
            VectorIndexWriterCommand::ClaimRebuildLease {
                vector_space_id: vector_space_id.clone(),
                observed_at_unix_ms: observed_at + 1,
            },
            deadline,
        )
        .await
        .unwrap()
    {
        VectorIndexWriterAck::RebuildLeaseClaimed(RebuildLeaseClaimAck::Claimed(fence)) => fence,
        acknowledgement => panic!("unexpected rebuild claim: {acknowledgement:?}"),
    };
    assert_eq!(
        writer
            .vector_index_until(
                VectorIndexWriterCommand::CreateGenerationObjects {
                    fence: fence.clone(),
                    observed_at_unix_ms: observed_at + 2,
                },
                deadline,
            )
            .await
            .unwrap(),
        VectorIndexWriterAck::GenerationObjectsCreated(GenerationObjectCreationAck::Created)
    );
    for command in [
        VectorIndexWriterCommand::PopulateRebuildChunk {
            fence: fence.clone(),
            observed_at_unix_ms: observed_at + 3,
        },
        VectorIndexWriterCommand::CatchUpRebuildChanges {
            fence: fence.clone(),
            observed_at_unix_ms: observed_at + 3,
        },
    ] {
        assert!(matches!(
            writer.vector_index_until(command, deadline).await.unwrap(),
            VectorIndexWriterAck::RebuildStepped(RebuildStepAck::Applied {
                processed: 0,
                complete: true,
                ..
            })
        ));
    }
    assert_eq!(
        writer
            .vector_index_until(
                VectorIndexWriterCommand::FlipRebuildGeneration {
                    fence,
                    activated_at_unix_ms: observed_at + 4,
                },
                deadline,
            )
            .await
            .unwrap(),
        VectorIndexWriterAck::RebuildFlipped(RebuildFlipAck::Activated { record_count: 0 })
    );
}

fn live_config(
    database_path: &Path,
    endpoint: &str,
    provider_timeout_ms: u64,
    max_in_flight: usize,
) -> RouterConfig {
    let mut config: RouterConfig = serde_json::from_value(json!({
        "version": 1,
        "mode": "shadow",
        "project_id": format!("live-embedding-{}", Uuid::now_v7()),
        "database_path": database_path,
        "embedders": [{
            "id": "embedding-main",
            "base_url": endpoint,
            "model": "embed-model",
            "provider_revision": "revision-1",
            "dimensions": 2,
            "timeout_ms": provider_timeout_ms,
            "max_in_flight": max_in_flight,
            "batch_size": 1
        }],
        "pools": [{
            "id": "pool-a",
            "api_family": "openai_chat_completions",
            "anchor_models": ["anchor-model"],
            "anchor_revision": "revision-1",
            "sampling_probability": 0.25,
            "max_candidates_per_sample": 1,
            "concurrency": {"shadow": 1, "judge": 1},
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
    .unwrap();
    let mut alternate = config.pools[0].clone();
    alternate.id = "pool-b".to_string();
    alternate.anchor_models = vec!["alternate-anchor".to_string()];
    alternate.canonicalizer.max_task_bytes = alternate
        .canonicalizer
        .max_task_bytes
        .checked_add(1)
        .unwrap();
    config.pools.push(alternate);
    assert!(
        config
            .validate()
            .iter()
            .all(|diagnostic| { diagnostic.level != nemo_relay::plugin::DiagnosticLevel::Error })
    );
    config
}

fn request_projection() -> RouterRequestProjectionV1 {
    request_projection_named("route this live request")
}

fn request_projection_named(task: &str) -> RouterRequestProjectionV1 {
    let mut projection = RouterRequestProjectionV1 {
        schema: REQUEST_PROJECTION_SCHEMA_V1.to_string(),
        family: LlmApiFamily::OpenAIChatCompletions,
        normalized_request: SanitizedAnnotatedLlmRequest {
            messages: vec![SanitizedMessage::User {
                content: SanitizedMessageContent::Text(task.to_string()),
                name: None,
            }],
            model: Some("anchor-model".to_string()),
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
    projection.semantic_request_fingerprint = canonical_sha256(&value).unwrap();
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

fn expected_vector(
    vector_space_id: &VectorSpaceId,
    dimensions: VectorDimensions,
) -> AuthoritativeVector {
    AuthoritativeVector::from_normalized(
        vector_space_id,
        NormalizedVector::from_provider_f64(&[3.0, 4.0], dimensions).unwrap(),
    )
    .unwrap()
}

fn json_response(value: Json) -> Vec<u8> {
    let body = serde_json::to_vec(&value).unwrap();
    let mut response = format!(
        "HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        body.len()
    )
    .into_bytes();
    response.extend_from_slice(&body);
    response
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
            && let Some(header_end) = request.windows(4).position(|window| window == b"\r\n\r\n")
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn prepared_query_freezes_the_service_authoritative_identity_once() {
    let mut options = FixtureOptions::success();
    options.seed_cache = true;
    let fixture = LiveServiceFixture::start(options).await;
    let prepared = fixture
        .service
        .prepare_query("pool-a", &fixture.request, &fixture.routing)
        .unwrap();
    assert_eq!(prepared.artifact().canonical_query_hash, fixture.query_hash);
    assert_eq!(prepared.vector_space_id(), &fixture.vector_space_id);
    assert_eq!(
        prepared.mapping().config_generation_id,
        fixture.registry.config_generation_id
    );
    assert_eq!(
        prepared.timeout(),
        Duration::from_millis(fixture.config.embedders[0].timeout_ms)
    );
    assert!(
        fixture
            .service
            .prepare_query("missing", &fixture.request, &fixture.routing)
            .is_err()
    );

    let result = fixture
        .service
        .embed_prepared_until(prepared, Instant::now() + Duration::from_secs(2))
        .await;
    assert!(matches!(result, LiveEmbeddingResult::Ready(_)));
    assert_eq!(fixture.server.request_count(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn full_live_partition_resolution_reports_no_exact_partition_in_one_snapshot() {
    let fixture = LiveServiceFixture::start(FixtureOptions::success()).await;
    let prepared = fixture
        .service
        .prepare_query("pool-a", &fixture.request, &fixture.routing)
        .unwrap();
    let mapping = &fixture.registry.mappings["pool-a"];
    let partition = build_routing_partition_from_input_v1(&RoutingPartitionInputV1 {
        base: RoutingPartitionBaseV1 {
            tenant_policy_hash: fixture.routing.tenant_policy_hash.clone(),
            agent_policy_hash: fixture.routing.agent_policy_hash.clone(),
            policy_version_id: mapping.policy_version_id.clone(),
            learning_generation_id: fixture.learning_generation_id,
            api_family: LlmApiFamily::OpenAIChatCompletions,
            transport_identity: "transport-v1".to_string(),
            anchor_model: "anchor-model".to_string(),
            anchor_revision: fixture.config.pools[0].anchor_revision.clone(),
            evaluator_version: "e".repeat(64),
            vector_space_id: fixture.vector_space_id.as_str().to_string(),
        },
        candidate_id: "candidate-a".to_string(),
        candidate_model: "candidate-model".to_string(),
        candidate_model_revision: "revision-1".to_string(),
        decoding_fingerprint: "d".repeat(64),
    })
    .unwrap();
    let query = expected_vector(&fixture.vector_space_id, VectorDimensions::new(2).unwrap());
    let store = SqliteVecStore::new(fixture.writer.clone(), fixture.read_pool.clone());
    assert_eq!(
        store
            .search_live_partition_until(
                prepared.mapping(),
                &partition,
                query.vector(),
                4,
                Instant::now() + Duration::from_secs(2),
            )
            .await
            .unwrap(),
        LiveProjectedNeighborSearch::NoPartition
    );
    assert_eq!(fixture.server.request_count(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cache_hit_returns_the_exact_persisted_vector_with_zero_http() {
    let mut options = FixtureOptions::success();
    options.seed_cache = true;
    let fixture = LiveServiceFixture::start(options).await;
    fixture
        .server
        .set_response(json_response(json!({"malformed": true})));

    let result = fixture.embed("pool-a", Duration::from_secs(2)).await;
    let LiveEmbeddingResult::Ready(vector) = result else {
        panic!("seeded cache should be ready");
    };
    assert_eq!(
        vector,
        expected_vector(&fixture.vector_space_id, VectorDimensions::new(2).unwrap())
    );
    assert_eq!(fixture.server.request_count(), 0);
    assert!(fixture.job_state().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cache_miss_calls_http_once_and_returns_only_the_persisted_vector() {
    let fixture = LiveServiceFixture::start(FixtureOptions::success()).await;

    let result = fixture.embed("pool-a", Duration::from_secs(2)).await;
    let LiveEmbeddingResult::Ready(vector) = result else {
        panic!("healthy miss should return the completed vector");
    };
    assert_eq!(fixture.server.request_count(), 1);
    assert_eq!(
        fixture.job_state(),
        Some((1, "completed".to_string(), None))
    );

    let connection = Connection::open(&fixture.database_path).unwrap();
    let cache = load_embedding_cache(
        &connection,
        fixture.project_uuid,
        &fixture.vector_space_id,
        &fixture.query_hash,
    )
    .unwrap()
    .expect("provider completion should persist the cache before return");
    assert_eq!(vector, cache.vector);
    let embedding_job_id = connection
        .query_row(
            "SELECT embedding_job_id FROM embedding_jobs WHERE canonical_query_hash = ?1",
            params![fixture.query_hash],
            |row| row.get::<_, String>(0),
        )
        .unwrap();
    assert_eq!(
        load_verified_embedding_job(&connection, fixture.project_uuid, &embedding_job_id)
            .unwrap()
            .unwrap()
            .terminal_error_class,
        None
    );

    assert_eq!(
        fixture.embed("pool-a", Duration::from_secs(2)).await,
        LiveEmbeddingResult::Ready(cache.vector)
    );
    assert_eq!(fixture.server.request_count(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_local_callers_share_one_durable_provider_request() {
    let mut options = FixtureOptions::success();
    options.response_delay = Duration::from_millis(75);
    let fixture = LiveServiceFixture::start(options).await;
    let start = Arc::new(tokio::sync::Barrier::new(2));
    let service_a = fixture.service.clone();
    let service_b = fixture.service.clone();
    let request_a = fixture.request.clone();
    let request_b = fixture.request.clone();
    let routing_a = fixture.routing.clone();
    let routing_b = fixture.routing.clone();
    let start_a = start.clone();
    let deadline = Instant::now() + Duration::from_secs(3);

    let caller_a = async move {
        start_a.wait().await;
        service_a
            .embed_until("pool-a", &request_a, &routing_a, deadline)
            .await
    };
    let caller_b = async move {
        start.wait().await;
        service_b
            .embed_until("pool-a", &request_b, &routing_b, deadline)
            .await
    };
    let (first, second) = tokio::join!(caller_a, caller_b);
    let expected = LiveEmbeddingResult::Ready(expected_vector(
        &fixture.vector_space_id,
        VectorDimensions::new(2).unwrap(),
    ));
    assert_eq!(first, expected);
    assert_eq!(second, expected);
    assert_eq!(fixture.server.request_count(), 1);
    assert_eq!(
        fixture.job_state(),
        Some((1, "completed".to_string(), None))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_processes_share_the_same_durable_provider_request() {
    let mut options = FixtureOptions::success();
    options.response_delay = Duration::from_millis(75);
    options.provider_timeout_ms = 10_000;
    let fixture = LiveServiceFixture::start(options).await;

    let second_activation =
        LedgerRepository::activate_at(&fixture.config, Utc::now().timestamp_millis()).unwrap();
    assert_eq!(
        second_activation.identity.project_uuid,
        fixture.project_uuid
    );
    let second_registry = Arc::new(second_activation.registry);
    let second_clients =
        Arc::new(build_frozen_embedder_clients(&fixture.config, &second_registry).unwrap());
    let (second_owner, second_writer) =
        LedgerWriterOwner::start(second_activation.repository, 32).unwrap();
    let second_read_pool = LedgerReadPool::open(&fixture.database_path).unwrap();
    let second_gate = ProviderAdmissionGate::initially_open_for_test();
    let second_service = LiveEmbeddingService::new(
        &fixture.config,
        second_registry,
        second_writer,
        second_read_pool.clone(),
        second_clients,
        second_gate,
    )
    .unwrap();

    let start = Arc::new(tokio::sync::Barrier::new(2));
    let first_service = fixture.service.clone();
    let first_request = fixture.request.clone();
    let first_routing = fixture.routing.clone();
    let second_request = fixture.request.clone();
    let second_routing = fixture.routing.clone();
    let first_start = start.clone();
    let deadline = Instant::now() + Duration::from_secs(10);
    let first = async move {
        first_start.wait().await;
        first_service
            .embed_until("pool-a", &first_request, &first_routing, deadline)
            .await
    };
    let second = async move {
        start.wait().await;
        second_service
            .embed_until("pool-a", &second_request, &second_routing, deadline)
            .await
    };
    let (first, second) = tokio::join!(first, second);
    let expected = LiveEmbeddingResult::Ready(expected_vector(
        &fixture.vector_space_id,
        VectorDimensions::new(2).unwrap(),
    ));
    assert_eq!(first, expected);
    assert_eq!(second, expected);
    assert_eq!(fixture.server.request_count(), 1);

    second_read_pool.abort();
    second_owner.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lease_held_loser_stays_wait_only_after_owner_becomes_reclaimable() {
    let mut options = FixtureOptions::success();
    options.provider_timeout_ms = 20_000;
    let fixture = LiveServiceFixture::start(options).await;
    fixture.server.hold_responses();

    let first_service = fixture.service.clone();
    let first_request = fixture.request.clone();
    let first_routing = fixture.routing.clone();
    let first = tokio::spawn(async move {
        first_service
            .embed_until(
                "pool-a",
                &first_request,
                &first_routing,
                Instant::now() + Duration::from_secs(10),
            )
            .await
    });
    let request_deadline = Instant::now() + Duration::from_secs(5);
    while fixture.server.request_count() == 0 && Instant::now() < request_deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(fixture.server.request_count(), 1);

    let second_activation =
        LedgerRepository::activate_at(&fixture.config, Utc::now().timestamp_millis()).unwrap();
    let second_registry = Arc::new(second_activation.registry);
    let second_clients =
        Arc::new(build_frozen_embedder_clients(&fixture.config, &second_registry).unwrap());
    let (second_owner, second_writer) =
        LedgerWriterOwner::start(second_activation.repository, 32).unwrap();
    let second_read_pool = LedgerReadPool::open(&fixture.database_path).unwrap();
    let second_service = LiveEmbeddingService::new(
        &fixture.config,
        second_registry,
        second_writer,
        second_read_pool.clone(),
        second_clients,
        ProviderAdmissionGate::initially_open_for_test(),
    )
    .unwrap();
    let losing_service = second_service.clone();
    let losing_request = fixture.request.clone();
    let losing_routing = fixture.routing.clone();
    let loser = tokio::spawn(async move {
        losing_service
            .embed_until(
                "pool-a",
                &losing_request,
                &losing_routing,
                Instant::now() + Duration::from_secs(10),
            )
            .await
    });
    let held_deadline = Instant::now() + Duration::from_secs(5);
    while second_service.lease_held_observation_count() == 0 && Instant::now() < held_deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(second_service.lease_held_observation_count(), 1);
    assert!(!loser.is_finished());
    assert_eq!(fixture.server.request_count(), 1);

    let stop = ProcessStop::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        Utc::now().timestamp_millis(),
    )
    .unwrap();
    assert_eq!(
        fixture
            .writer
            .stop_process_until(stop, Instant::now() + Duration::from_secs(5))
            .await
            .unwrap(),
        ProcessCommandAck::Applied
    );
    fixture.server.release_responses();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(fixture.server.request_count(), 1);
    second_service.close();

    assert_eq!(first.await.unwrap(), LiveEmbeddingResult::Unavailable);
    assert_eq!(loser.await.unwrap(), LiveEmbeddingResult::Unavailable);
    assert_eq!(fixture.server.request_count(), 1);
    assert_eq!(
        Connection::open(&fixture.database_path)
            .unwrap()
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );

    second_read_pool.abort();
    second_owner.abort();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stale_prepared_snapshot_accepts_the_authoritative_later_generation_lease() {
    let mut options = FixtureOptions::success();
    options.max_in_flight = 1;
    let fixture = LiveServiceFixture::start(options).await;
    let held = fixture
        .clients
        .try_acquire(&fixture.vector_space_id)
        .unwrap();
    let service = fixture.service.clone();
    let request = fixture.request.clone();
    let routing = fixture.routing.clone();
    let execution = tokio::spawn(async move {
        service
            .embed_until(
                "pool-a",
                &request,
                &routing,
                Instant::now() + Duration::from_secs(3),
            )
            .await
    });
    let job_deadline = Instant::now() + Duration::from_secs(1);
    while fixture.job_state().is_none() && Instant::now() < job_deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(fixture.job_state(), Some((0, "pending".to_string(), None)));

    let claim_at = Utc::now().timestamp_millis();
    let mut other = LedgerRepository::activate_at(&fixture.config, claim_at).unwrap();
    let embedding_job_id = Connection::open(&fixture.database_path)
        .unwrap()
        .query_row(
            "SELECT embedding_job_id FROM embedding_jobs WHERE canonical_query_hash = ?1",
            params![fixture.query_hash],
            |row| row.get::<_, String>(0),
        )
        .unwrap();
    let claim = EmbeddingJobBatchClaim::new(
        Uuid::now_v7(),
        fixture.vector_space_id.clone(),
        Uuid::now_v7(),
        claim_at + 1,
        vec![
            EmbeddingJobBatchClaimItem::new(
                embedding_job_id.clone(),
                Uuid::now_v7(),
                Uuid::now_v7(),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let EmbeddingJobBatchClaimAck::Claimed(leases) =
        other.repository.claim_embedding_job_batch(&claim).unwrap()
    else {
        panic!("intervening owner should claim the pending job");
    };
    let release = EmbeddingJobResolution::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        embedding_job_id,
        leases[0].lease_token,
        leases[0].job.attempt_generation,
        leases[0].job.content_hash.clone(),
        claim_at + 2,
        EmbeddingJobResolutionKind::Released,
    )
    .unwrap();
    assert!(matches!(
        other.repository.resolve_embedding_job(&release).unwrap(),
        EmbeddingJobResolutionAck::Applied { .. }
    ));
    drop(held);

    assert!(matches!(
        execution.await.unwrap(),
        LiveEmbeddingResult::Ready(_)
    ));
    assert_eq!(fixture.server.request_count(), 1);
    assert_eq!(
        fixture.job_state(),
        Some((2, "completed".to_string(), None))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn background_and_live_callers_race_through_one_durable_claim() {
    let mut options = FixtureOptions::success();
    options.response_delay = Duration::from_millis(75);
    let fixture = LiveServiceFixture::start(options).await;
    let mapping = &fixture.registry.mappings["pool-a"];
    let artifact = build_canonical_routing_query(
        &fixture.request,
        &fixture.routing,
        &fixture.config.pools[0].canonicalizer,
    )
    .unwrap();
    let prepare = LiveEmbeddingPrepare::new(
        FrozenMappingKey::new(
            mapping.project_uuid,
            mapping.config_generation_id.clone(),
            mapping.pool_id.clone(),
            mapping.policy_version_id.clone(),
        )
        .unwrap(),
        artifact,
        Uuid::now_v7(),
        Uuid::now_v7(),
        Utc::now().timestamp_millis(),
    )
    .unwrap();
    assert!(matches!(
        fixture
            .writer
            .prepare_live_embedding_until(prepare, Instant::now() + Duration::from_secs(2))
            .await
            .unwrap(),
        LiveEmbeddingPrepareAck::Pending(_)
    ));
    let candidate = fixture
        .read_pool
        .select_embedding_work_until(
            fixture.project_uuid,
            fixture.registry.config_generation_id.clone(),
            Utc::now().timestamp_millis(),
            None,
            1,
            Instant::now() + Duration::from_secs(2),
        )
        .await
        .unwrap()
        .pop()
        .unwrap();
    let plan = prepare_embedding_batch(
        &fixture.clients,
        vec![candidate],
        Instant::now(),
        Utc::now().timestamp_millis(),
    )
    .unwrap();
    let (_cancel, cancellation) = tokio::sync::watch::channel(false);
    let background = execute_embedding_batch(&fixture.writer, plan, cancellation);
    let live = fixture.embed("pool-a", Duration::from_secs(3));
    let (background, live) = tokio::join!(background, live);

    assert!(matches!(
        background,
        BackgroundJobOutcome::Applied
            | BackgroundJobOutcome::AlreadyApplied
            | BackgroundJobOutcome::Deferred
    ));
    assert_eq!(
        live,
        LiveEmbeddingResult::Ready(expected_vector(
            &fixture.vector_space_id,
            VectorDimensions::new(2).unwrap(),
        ))
    );
    assert_eq!(fixture.server.request_count(), 1);
    assert_eq!(
        fixture.job_state(),
        Some((1, "completed".to_string(), None))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn malformed_response_quarantines_immediately_and_degrades_the_space() {
    let mut options = FixtureOptions::success();
    options.response = json_response(json!({"data": "not-an-embedding-array"}));
    let fixture = LiveServiceFixture::start(options).await;

    assert_eq!(
        fixture.embed("pool-a", Duration::from_secs(2)).await,
        LiveEmbeddingResult::Unavailable
    );
    assert_eq!(fixture.server.request_count(), 1);
    assert_eq!(
        fixture.job_state(),
        Some((
            1,
            "quarantined".to_string(),
            Some("embedder_malformed_response".to_string())
        ))
    );

    assert_eq!(
        fixture.embed("pool-a", Duration::from_secs(2)).await,
        LiveEmbeddingResult::Unavailable
    );
    assert_eq!(fixture.server.request_count(), 1);

    let different_request = request_projection_named("a different live request");
    assert_eq!(
        fixture
            .service
            .embed_until(
                "pool-a",
                &different_request,
                &fixture.routing,
                Instant::now() + Duration::from_secs(2),
            )
            .await,
        LiveEmbeddingResult::Unavailable
    );
    assert_eq!(fixture.server.request_count(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn saturation_and_expired_deadline_start_no_claim_or_http() {
    let mut options = FixtureOptions::success();
    options.max_in_flight = 1;
    let fixture = LiveServiceFixture::start(options).await;
    let held = fixture
        .clients
        .try_acquire(&fixture.vector_space_id)
        .unwrap();

    assert_eq!(
        fixture.embed("pool-a", Duration::from_millis(40)).await,
        LiveEmbeddingResult::Unavailable
    );
    assert_eq!(fixture.server.request_count(), 0);
    assert_eq!(fixture.job_state(), Some((0, "pending".to_string(), None)));
    drop(held);

    assert_eq!(
        fixture.embed("pool-a", Duration::ZERO).await,
        LiveEmbeddingResult::Unavailable
    );
    assert_eq!(fixture.server.request_count(), 0);
    assert_eq!(fixture.job_state(), Some((0, "pending".to_string(), None)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn expired_queued_live_claim_resumes_without_graph_mutation_or_http() {
    let mut options = FixtureOptions::success();
    options.provider_timeout_ms = 75;
    options.max_in_flight = 1;
    let fixture = LiveServiceFixture::start(options).await;
    let held = fixture
        .clients
        .try_acquire(&fixture.vector_space_id)
        .unwrap();
    let service = fixture.service.clone();
    let request = fixture.request.clone();
    let routing = fixture.routing.clone();
    let live = tokio::spawn(async move {
        service
            .embed_until(
                "pool-a",
                &request,
                &routing,
                Instant::now() + Duration::from_secs(2),
            )
            .await
    });
    let job_deadline = Instant::now() + Duration::from_secs(1);
    while fixture.job_state().is_none() && Instant::now() < job_deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    let before = fixture.claim_graph_state();
    assert_eq!(before, Some((0, None, "pending".to_string(), 1, 0)));

    let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
    let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
    let pause_writer = fixture.writer.clone();
    let pause = tokio::spawn(async move {
        pause_writer
            .pause_until(
                Instant::now() + Duration::from_secs(2),
                started_tx,
                release_rx,
            )
            .await
    });
    tokio::task::spawn_blocking(move || started_rx.recv_timeout(Duration::from_secs(1)))
        .await
        .unwrap()
        .unwrap();
    drop(held);

    assert_eq!(
        tokio::time::timeout(Duration::from_secs(1), live)
            .await
            .unwrap()
            .unwrap(),
        LiveEmbeddingResult::Unavailable
    );
    assert_eq!(fixture.server.request_count(), 0);
    assert_eq!(fixture.claim_graph_state(), before);

    release_tx.send(()).unwrap();
    pause.await.unwrap().unwrap();
    fixture
        .writer
        .flush_until(Instant::now() + Duration::from_secs(1))
        .await
        .unwrap();
    assert_eq!(fixture.server.request_count(), 0);
    assert_eq!(fixture.claim_graph_state(), before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn late_provider_response_never_returns_an_unpersisted_vector() {
    let mut options = FixtureOptions::success();
    options.response_delay = Duration::from_millis(600);
    let fixture = LiveServiceFixture::start(options).await;
    let service = fixture.service.clone();
    let request = fixture.request.clone();
    let routing = fixture.routing.clone();
    let execution = tokio::spawn(async move {
        service
            .embed_until(
                "pool-a",
                &request,
                &routing,
                Instant::now() + Duration::from_millis(250),
            )
            .await
    });
    let request_deadline = Instant::now() + Duration::from_secs(1);
    while fixture.server.request_count() == 0 && Instant::now() < request_deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(fixture.server.request_count(), 1);
    assert_eq!(execution.await.unwrap(), LiveEmbeddingResult::Unavailable);
    assert_eq!(
        Connection::open(&fixture.database_path)
            .unwrap()
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dropped_caller_does_not_orphan_runtime_owned_provider_work() {
    let fixture = LiveServiceFixture::start(FixtureOptions::success()).await;
    fixture.server.hold_responses();
    let service = fixture.service.clone();
    let request = fixture.request.clone();
    let routing = fixture.routing.clone();
    let execution = tokio::spawn(async move {
        service
            .embed_until(
                "pool-a",
                &request,
                &routing,
                Instant::now() + Duration::from_secs(3),
            )
            .await
    });
    let request_deadline = Instant::now() + Duration::from_secs(1);
    while fixture.server.request_count() == 0 && Instant::now() < request_deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(fixture.server.request_count(), 1);

    execution.abort();
    assert!(execution.await.unwrap_err().is_cancelled());
    fixture.server.release_responses();
    assert!(
        fixture
            .service
            .drain_until(Instant::now() + Duration::from_secs(2))
            .await
    );
    assert_eq!(
        fixture.job_state(),
        Some((1, "completed".to_string(), None))
    );
    assert!(
        load_embedding_cache(
            &Connection::open(&fixture.database_path).unwrap(),
            fixture.project_uuid,
            &fixture.vector_space_id,
            &fixture.query_hash,
        )
        .unwrap()
        .is_some()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn close_waits_for_an_admitted_live_call_before_reporting_drained() {
    let fixture = LiveServiceFixture::start(FixtureOptions::success()).await;
    let admitted = Arc::new(tokio::sync::Barrier::new(2));
    let release = Arc::new(tokio::sync::Barrier::new(2));
    fixture
        .service
        .pause_next_call_after_admission(admitted.clone(), release.clone());
    let service = fixture.service.clone();
    let request = fixture.request.clone();
    let routing = fixture.routing.clone();
    let execution = tokio::spawn(async move {
        service
            .embed_until(
                "pool-a",
                &request,
                &routing,
                Instant::now() + Duration::from_secs(3),
            )
            .await
    });
    tokio::time::timeout(Duration::from_secs(1), admitted.wait())
        .await
        .expect("live call was not registered before its test pause");

    fixture.service.close();
    assert!(
        !fixture
            .service
            .drain_until(Instant::now() + Duration::from_millis(25))
            .await
    );
    assert!(fixture.job_state().is_none());
    assert_eq!(fixture.server.request_count(), 0);

    release.wait().await;
    assert_eq!(execution.await.unwrap(), LiveEmbeddingResult::Unavailable);
    assert!(
        fixture
            .service
            .drain_until(Instant::now() + Duration::from_secs(1))
            .await
    );
    assert!(fixture.job_state().is_none());
    assert_eq!(fixture.server.request_count(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn close_cancels_accepted_provider_work_and_drains_its_lease() {
    let fixture = LiveServiceFixture::start(FixtureOptions::success()).await;
    fixture.server.hold_responses();
    let service = fixture.service.clone();
    let request = fixture.request.clone();
    let routing = fixture.routing.clone();
    let execution = tokio::spawn(async move {
        service
            .embed_until(
                "pool-a",
                &request,
                &routing,
                Instant::now() + Duration::from_secs(3),
            )
            .await
    });
    let request_deadline = Instant::now() + Duration::from_secs(1);
    while fixture.server.request_count() == 0 && Instant::now() < request_deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(fixture.server.request_count(), 1);

    fixture.service.close();
    assert_eq!(execution.await.unwrap(), LiveEmbeddingResult::Unavailable);
    assert!(
        fixture
            .service
            .drain_until(Instant::now() + Duration::from_secs(2))
            .await
    );
    fixture.server.release_responses();
    let durable = Connection::open(&fixture.database_path)
        .unwrap()
        .query_row(
            "SELECT lease_token, (SELECT COUNT(*) FROM embeddings)
             FROM embedding_jobs WHERE canonical_query_hash = ?1",
            params![fixture.query_hash],
            |row| Ok((row.get::<_, Option<String>>(0)?, row.get::<_, i64>(1)?)),
        )
        .unwrap();
    assert_eq!(durable, (None, 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn expired_shutdown_deadline_leaves_the_live_lease_for_durable_expiry() {
    let fixture = LiveServiceFixture::start(FixtureOptions::success()).await;
    fixture.server.hold_responses();
    let service = fixture.service.clone();
    let request = fixture.request.clone();
    let routing = fixture.routing.clone();
    let execution = tokio::spawn(async move {
        service
            .embed_until(
                "pool-a",
                &request,
                &routing,
                Instant::now() + Duration::from_secs(3),
            )
            .await
    });
    let request_deadline = Instant::now() + Duration::from_secs(1);
    while fixture.server.request_count() == 0 && Instant::now() < request_deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(fixture.server.request_count(), 1);

    fixture.service.close_admission();
    fixture.service.close_until(Instant::now());
    assert_eq!(execution.await.unwrap(), LiveEmbeddingResult::Unavailable);
    assert!(
        fixture
            .service
            .drain_until(Instant::now() + Duration::from_secs(2))
            .await
    );
    fixture.server.release_responses();
    let durable = Connection::open(&fixture.database_path)
        .unwrap()
        .query_row(
            "SELECT job.lease_token,
                    (SELECT state FROM embedding_job_state_events
                     WHERE embedding_job_id = job.embedding_job_id
                     ORDER BY event_seq DESC LIMIT 1),
                    (SELECT COUNT(*) FROM embeddings)
             FROM embedding_jobs AS job WHERE canonical_query_hash = ?1",
            params![fixture.query_hash],
            |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .unwrap();
    assert!(durable.0.is_some());
    assert_eq!(durable.1, "claimed");
    assert_eq!(durable.2, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_after_writer_fence_prevents_live_cleanup_and_later_starts() {
    let fixture = LiveServiceFixture::start(FixtureOptions::success()).await;
    fixture.server.hold_responses();
    let service = fixture.service.clone();
    let request = fixture.request.clone();
    let routing = fixture.routing.clone();
    let execution = tokio::spawn(async move {
        service
            .embed_until(
                "pool-a",
                &request,
                &routing,
                Instant::now() + Duration::from_secs(3),
            )
            .await
    });
    let request_deadline = Instant::now() + Duration::from_secs(1);
    while fixture.server.request_count() == 0 && Instant::now() < request_deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(fixture.server.request_count(), 1);

    fixture.writer.abort();
    fixture.service.abort();
    assert_eq!(execution.await.unwrap(), LiveEmbeddingResult::Unavailable);
    assert!(
        fixture
            .service
            .drain_until(Instant::now() + Duration::from_secs(2))
            .await
    );
    fixture.server.release_responses();
    assert_eq!(
        fixture.embed("pool-a", Duration::from_secs(1)).await,
        LiveEmbeddingResult::Unavailable
    );
    assert_eq!(fixture.server.request_count(), 1);
    let durable = Connection::open(&fixture.database_path)
        .unwrap()
        .query_row(
            "SELECT lease_token, (SELECT COUNT(*) FROM embeddings)
             FROM embedding_jobs WHERE canonical_query_hash = ?1",
            params![fixture.query_hash],
            |row| Ok((row.get::<_, Option<String>>(0)?, row.get::<_, i64>(1)?)),
        )
        .unwrap();
    assert!(durable.0.is_some());
    assert_eq!(durable.1, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn provider_operation_timeout_is_durable_and_never_caches_late_vector() {
    let mut options = FixtureOptions::success();
    options.provider_timeout_ms = 40;
    options.response_delay = Duration::from_millis(200);
    let fixture = LiveServiceFixture::start(options).await;

    assert_eq!(
        fixture.embed("pool-a", Duration::from_secs(1)).await,
        LiveEmbeddingResult::Unavailable
    );
    tokio::time::sleep(Duration::from_millis(250)).await;

    let durable = Connection::open(&fixture.database_path)
        .unwrap()
        .query_row(
            "SELECT job.attempt_count,
                    (SELECT state FROM embedding_job_state_events
                     WHERE embedding_job_id = job.embedding_job_id
                     ORDER BY event_seq DESC LIMIT 1),
                    (SELECT stable_error_class FROM embedding_job_state_events
                     WHERE embedding_job_id = job.embedding_job_id
                     ORDER BY event_seq DESC LIMIT 1),
                    job.lease_token,
                    (SELECT COUNT(*) FROM embeddings)
             FROM embedding_jobs AS job
             WHERE job.canonical_query_hash = ?1",
            params![fixture.query_hash],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, i64>(4)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(
        durable,
        (
            1,
            "retry_scheduled".to_string(),
            Some("embedder_timeout".to_string()),
            None,
            0,
        )
    );
    assert_eq!(fixture.server.request_count(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lease_loss_never_exposes_the_stale_owners_provider_vector() {
    let mut options = FixtureOptions::success();
    options.response_delay = Duration::ZERO;
    let fixture = LiveServiceFixture::start(options).await;
    fixture.server.hold_responses();
    let service = fixture.service.clone();
    let request = fixture.request.clone();
    let routing = fixture.routing.clone();
    let execution = tokio::spawn(async move {
        service
            .embed_until(
                "pool-a",
                &request,
                &routing,
                Instant::now() + Duration::from_secs(2),
            )
            .await
    });
    let request_deadline = Instant::now() + Duration::from_secs(1);
    while fixture.server.request_count() == 0 && Instant::now() < request_deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(fixture.server.request_count(), 1);

    let takeover_at = Utc::now().timestamp_millis().saturating_add(31_000);
    let mut takeover = LedgerRepository::activate_at(&fixture.config, takeover_at).unwrap();
    let embedding_job_id = Connection::open(&fixture.database_path)
        .unwrap()
        .query_row(
            "SELECT embedding_job_id FROM embedding_jobs WHERE canonical_query_hash = ?1",
            params![fixture.query_hash],
            |row| row.get::<_, String>(0),
        )
        .unwrap();
    let claim = EmbeddingJobBatchClaim::new(
        Uuid::now_v7(),
        fixture.vector_space_id.clone(),
        Uuid::now_v7(),
        takeover_at + 1,
        vec![
            EmbeddingJobBatchClaimItem::new(
                embedding_job_id.clone(),
                Uuid::now_v7(),
                Uuid::now_v7(),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    let EmbeddingJobBatchClaimAck::Claimed(leases) = takeover
        .repository
        .claim_embedding_job_batch(&claim)
        .unwrap()
    else {
        panic!("expired owner should lose the durable lease");
    };
    let winner_vector = AuthoritativeVector::from_normalized(
        &fixture.vector_space_id,
        NormalizedVector::from_provider_f64(&[4.0, 3.0], VectorDimensions::new(2).unwrap())
            .unwrap(),
    )
    .unwrap();
    let completion = EmbeddingJobBatchCompletion::new(
        Uuid::now_v7(),
        fixture.vector_space_id.clone(),
        claim.lease_token,
        takeover_at + 2,
        vec![
            EmbeddingJobBatchCompletionItem::new(
                embedding_job_id,
                Uuid::now_v7(),
                Uuid::now_v7(),
                leases[0].job.attempt_generation,
                leases[0].job.content_hash.clone(),
                winner_vector.clone(),
            )
            .unwrap(),
        ],
    )
    .unwrap();
    assert!(matches!(
        takeover
            .repository
            .complete_embedding_job_batch(&completion)
            .unwrap(),
        EmbeddingJobBatchCompletionAck::Applied(_)
    ));
    fixture.server.release_responses();

    match execution.await.unwrap() {
        LiveEmbeddingResult::Ready(vector) => assert_eq!(vector, winner_vector),
        LiveEmbeddingResult::Unavailable => {}
    }
    let persisted = load_embedding_cache(
        &Connection::open(&fixture.database_path).unwrap(),
        fixture.project_uuid,
        &fixture.vector_space_id,
        &fixture.query_hash,
    )
    .unwrap()
    .unwrap();
    assert_eq!(persisted.vector, winner_vector);
    assert_ne!(
        persisted.vector,
        expected_vector(&fixture.vector_space_id, VectorDimensions::new(2).unwrap(),)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn writer_loss_after_http_never_returns_the_provider_vector() {
    let mut options = FixtureOptions::success();
    options.response_delay = Duration::from_millis(100);
    let fixture = LiveServiceFixture::start(options).await;
    let service = fixture.service.clone();
    let request = fixture.request.clone();
    let routing = fixture.routing.clone();
    let execution = tokio::spawn(async move {
        service
            .embed_until(
                "pool-a",
                &request,
                &routing,
                Instant::now() + Duration::from_secs(2),
            )
            .await
    });
    let request_deadline = Instant::now() + Duration::from_secs(1);
    while fixture.server.request_count() == 0 && Instant::now() < request_deadline {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }
    assert_eq!(fixture.server.request_count(), 1);
    fixture.writer_owner.abort();
    assert_eq!(execution.await.unwrap(), LiveEmbeddingResult::Unavailable);
    assert_eq!(
        Connection::open(&fixture.database_path)
            .unwrap()
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row
                .get::<_, i64>(0))
            .unwrap(),
        0
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn corrupt_cached_checksum_is_unavailable_without_provider_fallback() {
    let mut options = FixtureOptions::success();
    options.seed_cache = true;
    let fixture = LiveServiceFixture::start(options).await;
    Connection::open(&fixture.database_path)
        .unwrap()
        .execute(
            "UPDATE embeddings SET vector_checksum = ?1 WHERE vector_space_id = ?2",
            params!["f".repeat(64), fixture.vector_space_id.as_str()],
        )
        .unwrap();

    assert_eq!(
        fixture.embed("pool-a", Duration::from_secs(2)).await,
        LiveEmbeddingResult::Unavailable
    );
    assert_eq!(fixture.server.request_count(), 0);
    assert!(fixture.job_state().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn missing_exact_index_never_falls_back_to_another_ready_space() {
    let mut options = FixtureOptions::success();
    options.active_index = false;
    options.seed_cache = true;
    let fixture = LiveServiceFixture::start(options).await;
    let alternate_space = fixture.registry.mappings["pool-b"].vector_space_id.clone();
    let alternate_dimensions = fixture.registry.spaces[&alternate_space].dimensions;
    let alternate_pool = fixture
        .config
        .pools
        .iter()
        .find(|pool| pool.id == "pool-b")
        .unwrap();
    let alternate_artifact = build_canonical_routing_query(
        &fixture.request,
        &fixture.routing,
        &alternate_pool.canonicalizer,
    )
    .unwrap();
    let mut connection = Connection::open(&fixture.database_path).unwrap();
    let transaction = connection.transaction().unwrap();
    ensure_canonical_query(
        &transaction,
        &alternate_artifact,
        Utc::now().timestamp_millis(),
    )
    .unwrap();
    assert!(matches!(
        upsert_embedding_cache(
            &transaction,
            &EmbeddingCacheWrite {
                embedding_id: Uuid::now_v7(),
                project_uuid: fixture.project_uuid,
                vector_space_id: alternate_space.clone(),
                canonical_query_hash: alternate_artifact.canonical_query_hash.clone(),
                content_hash: alternate_artifact.canonical_query_hash.clone(),
                vector: expected_vector(&alternate_space, alternate_dimensions),
                source: EmbeddingCacheSource::Cache,
                created_at_unix_ms: Utc::now().timestamp_millis(),
            },
        )
        .unwrap(),
        EmbeddingCacheUpsertAck::Applied(_)
    ));
    transaction.commit().unwrap();
    activate_empty_generation(&fixture.writer, &alternate_space, alternate_dimensions).await;

    assert_eq!(
        fixture.embed("pool-a", Duration::from_secs(2)).await,
        LiveEmbeddingResult::Unavailable
    );
    assert_eq!(fixture.server.request_count(), 0);
    assert!(fixture.job_state().is_none());
    assert!(
        load_embedding_cache(
            &Connection::open(&fixture.database_path).unwrap(),
            fixture.project_uuid,
            &alternate_space,
            &alternate_artifact.canonical_query_hash,
        )
        .unwrap()
        .is_some()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn service_close_and_provider_gate_close_refuse_http_starts() {
    let closed_service = LiveServiceFixture::start(FixtureOptions::success()).await;
    closed_service.service.close();
    assert_eq!(
        closed_service.embed("pool-a", Duration::from_secs(2)).await,
        LiveEmbeddingResult::Unavailable
    );
    assert_eq!(closed_service.server.request_count(), 0);
    assert!(closed_service.job_state().is_none());

    let closed_gate = LiveServiceFixture::start(FixtureOptions::success()).await;
    assert!(closed_gate.gate.close());
    assert_eq!(
        closed_gate.embed("pool-a", Duration::from_secs(2)).await,
        LiveEmbeddingResult::Unavailable
    );
    assert_eq!(closed_gate.server.request_count(), 0);
    assert_eq!(
        closed_gate.job_state(),
        Some((1, "released".to_string(), None))
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_pool_is_unavailable_without_falling_back_to_another_space() {
    let fixture = LiveServiceFixture::start(FixtureOptions::success()).await;

    assert_eq!(
        fixture.embed("pool-missing", Duration::from_secs(2)).await,
        LiveEmbeddingResult::Unavailable
    );
    assert_eq!(fixture.server.request_count(), 0);
    assert!(fixture.job_state().is_none());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn alternate_partition_vector_is_not_used_as_live_embedding_fallback() {
    let fixture = LiveServiceFixture::start(FixtureOptions::success()).await;
    let dimensions = fixture.registry.spaces[&fixture.vector_space_id].dimensions;
    let alternate_vector = AuthoritativeVector::from_normalized(
        &fixture.vector_space_id,
        NormalizedVector::from_provider_f64(&[4.0, 3.0], dimensions).unwrap(),
    )
    .unwrap();
    let alternate_record_id = Uuid::now_v7().to_string();
    let connection = Connection::open(&fixture.database_path).unwrap();
    let root = connection
        .query_row(
            "SELECT root_table_name FROM vector_index_manifest
             WHERE vector_space_id = ?1 AND state = 'active'",
            params![fixture.vector_space_id.as_str()],
            |row| row.get::<_, String>(0),
        )
        .unwrap();
    connection
        .execute(
            &format!(
                "INSERT INTO \"{root}\" (record_id, embedding, partition_id)
                 VALUES (?1, ?2, ?3)"
            ),
            params![
                alternate_record_id,
                alternate_vector.blob().native_endian_bytes(),
                999_i64,
            ],
        )
        .unwrap();

    assert_eq!(
        fixture.embed("pool-a", Duration::from_secs(2)).await,
        LiveEmbeddingResult::Ready(expected_vector(&fixture.vector_space_id, dimensions))
    );
    assert_eq!(fixture.server.request_count(), 1);
    assert_ne!(
        load_embedding_cache(
            &Connection::open(&fixture.database_path).unwrap(),
            fixture.project_uuid,
            &fixture.vector_space_id,
            &fixture.query_hash,
        )
        .unwrap()
        .unwrap()
        .vector,
        alternate_vector
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn live_service_construction_rejects_remote_egress_without_consent() {
    let fixture = LiveServiceFixture::start(FixtureOptions::success()).await;
    let mut denied = fixture.config.clone();
    denied.allow_remote_embedding_egress = false;
    denied.embedders[0].base_url = "https://api.example.com/v1".to_string();
    assert!(denied.validate().iter().any(|diagnostic| {
        diagnostic.code == crate::diagnostics::UNSAFE_EMBEDDER_ENDPOINT
            && diagnostic.field.as_deref() == Some("embedders[0].base_url")
    }));

    let result = LiveEmbeddingService::new(
        &denied,
        fixture.registry.clone(),
        fixture.writer.clone(),
        fixture.read_pool.clone(),
        fixture.clients.clone(),
        fixture.gate.clone(),
    );
    assert!(matches!(
        result,
        Err(message) if message == "live embedding configuration is invalid"
    ));
    assert_eq!(fixture.server.request_count(), 0);
    assert!(fixture.job_state().is_none());
}
