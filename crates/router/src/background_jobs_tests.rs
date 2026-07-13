// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeSet;
use std::net::{Ipv4Addr, SocketAddr, TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use chrono::Utc;
use rusqlite::{Connection, params};
use serde_json::to_value;
use tempfile::{TempDir, tempdir};
use uuid::Uuid;

use crate::background::BackgroundCancellation;
use crate::background_jobs::{
    BackgroundJobOutcome, ExpectedEmbeddingLease, backfill_outcome,
    embedding_candidates_match_batch, embedding_failure_resolution_kind,
    embedding_leases_match_authority, embedding_resolution_outcome, execute_embedding_batch,
    execute_embedding_batch_with_cancellation, execute_rebuild, prepare_backfill_command,
    prepare_embedding_batch, prepare_embedding_completion_item, prepare_vector_space_recovery,
    provider_timeout_fits_lease,
};
use crate::canonical_json::canonical_json;
use crate::canonical_query::{
    CanonicalRoutingQueryArtifactV1, CanonicalRoutingQueryV1, CanonicalTaskV1,
};
use crate::config::RouterConfig;
use crate::embedder::{
    EmbedderBatchItem, EmbedderFailureDisposition, FrozenEmbedderClients,
    build_frozen_embedder_clients,
};
use crate::embedding_identity::CANONICAL_ROUTING_QUERY_SCHEMA_V1;
use crate::fingerprint::sha256_hex;
use crate::ledger::read_pool::LedgerReadPool;
use crate::ledger::repository::background_work::{BackfillWorkCandidate, EmbeddingWorkCandidate};
use crate::ledger::repository::embedding::{
    EmbeddingJobBatchClaim, EmbeddingJobBatchClaimItem, EmbeddingJobCreate, EmbeddingJobCreateAck,
    EmbeddingJobLease, EmbeddingJobResolutionAck, EmbeddingJobResolutionKind,
    EmbeddingJobResolvedState, EmbeddingJobSnapshot,
};
use crate::ledger::repository::materialization::VectorBackfillAck;
use crate::ledger::repository::vector_catalog::{CanonicalQuerySnapshot, ensure_canonical_query};
use crate::ledger::repository::vector_index::RebuildLeaseClaimAck;
use crate::ledger::repository::vector_registry::FrozenMappingKey;
use crate::ledger::repository::vector_work::{
    RebuildLeaseDisposition, VectorSpaceInspection, VectorSpaceRecoveryReason,
};
use crate::ledger::repository::{ActivatedLedger, LedgerRepository};
use crate::ledger::writer::{LedgerWriterClient, LedgerWriterOwner};
use crate::sqlite_vector_store::{VectorIndexWriterAck, VectorIndexWriterCommand};
use crate::vector::{AuthoritativeVector, NormalizedVector, VectorDimensions, VectorSpaceId};

struct CountingServer {
    address: SocketAddr,
    accepted: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl CountingServer {
    fn start() -> Self {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let accepted = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let thread_accepted = accepted.clone();
        let thread_stop = stop.clone();
        let thread = thread::spawn(move || {
            let mut connections = Vec::<TcpStream>::new();
            while !thread_stop.load(Ordering::Acquire) {
                match listener.accept() {
                    Ok((stream, _)) => {
                        thread_accepted.fetch_add(1, Ordering::AcqRel);
                        connections.push(stream);
                    }
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(1));
                    }
                    Err(error) => panic!("counting server failed: {error}"),
                }
            }
        });
        Self {
            address,
            accepted,
            stop,
            thread: Some(thread),
        }
    }

    fn endpoint(&self) -> String {
        format!("http://{}/v1", self.address)
    }

    fn accepted(&self) -> usize {
        self.accepted.load(Ordering::Acquire)
    }

    async fn wait_for_accept(&self) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while self.accepted() == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("provider request did not reach the loopback server");
    }
}

impl Drop for CountingServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            thread.join().unwrap();
        }
    }
}

struct DurableEmbeddingFixture {
    _temporary: TempDir,
    database_path: PathBuf,
    clients: FrozenEmbedderClients,
    writer: LedgerWriterClient,
    writer_owner: LedgerWriterOwner,
    read_pool: LedgerReadPool,
    candidate: EmbeddingWorkCandidate,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    vector_space_id: VectorSpaceId,
}

impl DurableEmbeddingFixture {
    async fn start(endpoint: &str, timeout_ms: u64, max_in_flight: usize) -> Self {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let config = embedding_config(&database_path, endpoint, timeout_ms, max_in_flight);
        let activation_at = Utc::now().timestamp_millis().saturating_sub(1_000);
        let mut activated = LedgerRepository::activate_at(&config, activation_at).unwrap();
        let project_uuid = activated.identity.project_uuid;
        let process_instance_id = activated.identity.process_instance_id;
        let vector_space_id = activated.registry.spaces.keys().next().unwrap().clone();
        let query = CanonicalRoutingQueryV1 {
            schema: CANONICAL_ROUTING_QUERY_SCHEMA_V1.to_string(),
            instructions: Vec::new(),
            current_task: CanonicalTaskV1 {
                text: "focused background job".to_string(),
            },
            bounded_context: Vec::new(),
            tool_schema_fingerprint: "0".repeat(64),
            response_schema_fingerprint: None,
            required_capabilities: Vec::new(),
            position_features: None,
        };
        let canonical_bytes = canonical_json(&to_value(&query).unwrap())
            .unwrap()
            .into_bytes();
        let canonical_query_hash = sha256_hex(&canonical_bytes);
        let artifact = CanonicalRoutingQueryArtifactV1 {
            query,
            canonical_bytes,
            canonical_query_hash: canonical_query_hash.clone(),
        };
        let mut connection = Connection::open(&database_path).unwrap();
        let transaction = connection.transaction().unwrap();
        ensure_canonical_query(&transaction, &artifact, activation_at + 1).unwrap();
        transaction.commit().unwrap();
        let create = EmbeddingJobCreate::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            vector_space_id.as_str(),
            canonical_query_hash.clone(),
            canonical_query_hash,
            activation_at + 2,
        )
        .unwrap();
        assert!(matches!(
            activated.repository.create_embedding_job(&create).unwrap(),
            EmbeddingJobCreateAck::Applied(_)
        ));
        let clients = build_frozen_embedder_clients(&config, &activated.registry).unwrap();
        let ActivatedLedger { repository, .. } = activated;
        let read_pool = LedgerReadPool::open(&database_path).unwrap();
        let (writer_owner, writer) = LedgerWriterOwner::start(repository, 16).unwrap();
        let candidate = read_pool
            .select_embedding_work_until(
                project_uuid,
                activated.identity.config_generation_id.clone(),
                Utc::now().timestamp_millis(),
                None,
                1,
                Instant::now() + Duration::from_secs(2),
            )
            .await
            .unwrap()
            .pop()
            .expect("focused embedding job must be discoverable");
        Self {
            _temporary: temporary,
            database_path,
            clients,
            writer,
            writer_owner,
            read_pool,
            candidate,
            project_uuid,
            process_instance_id,
            vector_space_id,
        }
    }

    fn job_state(&self) -> (i64, Option<String>, String, i64) {
        Connection::open(&self.database_path)
            .unwrap()
            .query_row(
                "SELECT job.attempt_count, job.lease_token,
                        (SELECT state FROM embedding_job_state_events
                         WHERE embedding_job_id = job.embedding_job_id
                         ORDER BY event_seq DESC LIMIT 1),
                        (SELECT count(*) FROM embeddings)
                 FROM embedding_jobs AS job WHERE job.embedding_job_id = ?1",
                params![self.candidate.job.embedding_job_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap()
    }
}

impl Drop for DurableEmbeddingFixture {
    fn drop(&mut self) {
        self.read_pool.abort();
        self.writer_owner.abort();
    }
}

fn embedding_config(
    database_path: &std::path::Path,
    endpoint: &str,
    timeout_ms: u64,
    max_in_flight: usize,
) -> RouterConfig {
    serde_json::from_value(serde_json::json!({
        "version": 1,
        "mode": "shadow",
        "project_id": format!("background-job-{}", Uuid::now_v7()),
        "database_path": database_path,
        "embedders": [{
            "id": "embedding-main",
            "base_url": endpoint,
            "model": "embed-model",
            "provider_revision": "revision-1",
            "dimensions": 2,
            "timeout_ms": timeout_ms,
            "max_in_flight": max_in_flight,
            "batch_size": 2
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
    .unwrap()
}

fn space(marker: char) -> VectorSpaceId {
    VectorSpaceId::new(marker.to_string().repeat(64)).unwrap()
}

fn candidate(marker: char, vector_space_id: &VectorSpaceId) -> EmbeddingWorkCandidate {
    let query = CanonicalRoutingQueryV1 {
        schema: "test-query@1".to_string(),
        instructions: Vec::new(),
        current_task: CanonicalTaskV1 {
            text: marker.to_string(),
        },
        bounded_context: Vec::new(),
        tool_schema_fingerprint: "0".repeat(64),
        response_schema_fingerprint: None,
        required_capabilities: Vec::new(),
        position_features: None,
    };
    let canonical = canonical_json(&to_value(&query).unwrap()).unwrap();
    let canonical_query_hash = sha256_hex(canonical.as_bytes());
    EmbeddingWorkCandidate {
        job: EmbeddingJobSnapshot {
            embedding_job_id: marker.to_string().repeat(64),
            vector_space_id: vector_space_id.as_str().to_string(),
            canonical_query_hash: canonical_query_hash.clone(),
            content_hash: canonical_query_hash.clone(),
            attempt_generation: 0,
            attempt_count: 0,
            next_eligible_at_unix_ms: 0,
            terminal_error_class: None,
            failure_propagation_cursor: None,
            failure_propagation_complete: false,
            reset_actor: None,
            reset_reason: None,
            canonical_payload_hash: sha256_hex(format!("payload-{marker}").as_bytes()),
        },
        canonical_query: CanonicalQuerySnapshot {
            artifact: CanonicalRoutingQueryArtifactV1 {
                query,
                canonical_bytes: canonical.into_bytes(),
                canonical_query_hash,
            },
            created_at_unix_ms: 1,
        },
        embedder_profile_version_id: "profile-version".to_string(),
        batch_size: 2,
    }
}

#[test]
fn candidate_batch_requires_one_space_profile_and_strict_job_order() {
    let vector_space_id = space('1');
    let candidates = [
        candidate('a', &vector_space_id),
        candidate('b', &vector_space_id),
    ];
    assert!(embedding_candidates_match_batch(
        &candidates,
        &vector_space_id,
        2
    ));

    let mut reversed = candidates.clone();
    reversed.reverse();
    assert!(!embedding_candidates_match_batch(
        &reversed,
        &vector_space_id,
        2
    ));

    let mut mixed_profile = candidates.clone();
    mixed_profile[1].embedder_profile_version_id = "other-profile".to_string();
    assert!(!embedding_candidates_match_batch(
        &mixed_profile,
        &vector_space_id,
        2
    ));

    let mut mixed_space = candidates.clone();
    mixed_space[1].job.vector_space_id = space('2').as_str().to_string();
    assert!(!embedding_candidates_match_batch(
        &mixed_space,
        &vector_space_id,
        2
    ));

    let mut mismatched_query = candidates;
    mismatched_query[1]
        .canonical_query
        .artifact
        .canonical_query_hash = "f".repeat(64);
    assert!(!embedding_candidates_match_batch(
        &mismatched_query,
        &vector_space_id,
        2
    ));
}

#[test]
fn lease_authority_fences_order_space_hash_token_and_expiry() {
    let vector_space_id = space('1');
    let candidates = [
        candidate('a', &vector_space_id),
        candidate('b', &vector_space_id),
    ];
    let items = candidates
        .iter()
        .map(|candidate| {
            EmbedderBatchItem::new(
                candidate.job.canonical_query_hash.clone(),
                String::from_utf8(candidate.canonical_query.artifact.canonical_bytes.clone())
                    .unwrap(),
                candidate.job.embedding_job_id.clone(),
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let claim = EmbeddingJobBatchClaim::new(
        Uuid::now_v7(),
        vector_space_id,
        Uuid::now_v7(),
        10,
        candidates
            .iter()
            .map(|candidate| {
                EmbeddingJobBatchClaimItem::new(
                    candidate.job.embedding_job_id.clone(),
                    Uuid::now_v7(),
                    Uuid::now_v7(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap();
    let expected = candidates
        .iter()
        .map(|candidate| ExpectedEmbeddingLease {
            embedding_job_id: candidate.job.embedding_job_id.clone(),
            canonical_query_hash: candidate.job.canonical_query_hash.clone(),
            content_hash: candidate.job.content_hash.clone(),
        })
        .collect::<Vec<_>>();
    let leases = candidates
        .iter()
        .map(|candidate| {
            let mut job = candidate.job.clone();
            job.attempt_generation = 1;
            job.attempt_count = 1;
            EmbeddingJobLease {
                job,
                lease_owner_process_instance_id: Uuid::now_v7(),
                lease_token: claim.lease_token,
                lease_expires_at_unix_ms: claim.lease_expires_at_unix_ms,
                state_event_hash: "e".repeat(64),
            }
        })
        .collect::<Vec<_>>();
    assert!(embedding_leases_match_authority(
        &leases, &claim, &items, &expected
    ));

    let mut reversed = leases.clone();
    reversed.reverse();
    assert!(!embedding_leases_match_authority(
        &reversed, &claim, &items, &expected
    ));
    let mut wrong_hash = leases.clone();
    wrong_hash[1].job.content_hash = "d".repeat(64);
    assert!(!embedding_leases_match_authority(
        &wrong_hash,
        &claim,
        &items,
        &expected
    ));
    let mut wrong_token = leases;
    wrong_token[0].lease_token = Uuid::now_v7();
    assert!(!embedding_leases_match_authority(
        &wrong_token,
        &claim,
        &items,
        &expected
    ));
}

#[test]
fn completion_item_uses_fresh_lease_generation_and_content_hash() {
    let vector_space_id = space('1');
    let candidate = candidate('a', &vector_space_id);
    let mut job = candidate.job;
    job.attempt_generation = 7;
    job.attempt_count = 3;
    let lease = EmbeddingJobLease {
        job,
        lease_owner_process_instance_id: Uuid::now_v7(),
        lease_token: Uuid::now_v7(),
        lease_expires_at_unix_ms: 60_000,
        state_event_hash: "e".repeat(64),
    };
    let vector = AuthoritativeVector::from_normalized(
        &vector_space_id,
        NormalizedVector::from_provider_f64(&[3.0, 4.0], VectorDimensions::new(2).unwrap())
            .unwrap(),
    )
    .unwrap();
    let item = prepare_embedding_completion_item(&lease, vector.clone()).unwrap();
    assert_eq!(item.embedding_job_id, lease.job.embedding_job_id);
    assert_eq!(item.attempt_generation, 7);
    assert_eq!(item.content_hash, lease.job.content_hash);
    assert_eq!(item.vector, vector);
    assert_ne!(item.embedding_id, item.completed_state_event_id);
}

#[test]
fn provider_timeout_preserves_completion_margin() {
    assert!(provider_timeout_fits_lease(
        Duration::from_secs(55),
        Duration::from_secs(60)
    ));
    assert!(!provider_timeout_fits_lease(
        Duration::from_millis(55_001),
        Duration::from_secs(60)
    ));
    assert!(!provider_timeout_fits_lease(Duration::MAX, Duration::MAX));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn saturated_profile_permit_leaves_the_durable_job_unclaimed() {
    let fixture = DurableEmbeddingFixture::start("http://127.0.0.1:9/v1", 250, 1).await;
    let _held = fixture
        .clients
        .try_acquire(&fixture.vector_space_id)
        .unwrap();

    assert!(matches!(
        prepare_embedding_batch(
            &fixture.clients,
            vec![fixture.candidate.clone()],
            Instant::now(),
            Utc::now().timestamp_millis(),
        ),
        Err(BackgroundJobOutcome::Deferred)
    ));
    assert_eq!(fixture.job_state(), (0, None, "pending".to_string(), 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delayed_writer_reaches_the_claim_deadline_before_starting_zero_http() {
    let server = CountingServer::start();
    let fixture = DurableEmbeddingFixture::start(&server.endpoint(), 75, 1).await;
    let plan = prepare_embedding_batch(
        &fixture.clients,
        vec![fixture.candidate.clone()],
        Instant::now(),
        Utc::now().timestamp_millis(),
    )
    .unwrap();
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
    let (_cancel, cancellation) = tokio::sync::watch::channel(false);

    assert_eq!(
        tokio::time::timeout(
            Duration::from_secs(1),
            execute_embedding_batch(&fixture.writer, plan, cancellation),
        )
        .await
        .unwrap(),
        BackgroundJobOutcome::Deferred
    );
    assert_eq!(server.accepted(), 0);
    release_tx.send(()).unwrap();
    pause.await.unwrap().unwrap();
    tokio::time::sleep(Duration::from_millis(25)).await;
    assert_eq!(server.accepted(), 0);
    assert_eq!(fixture.job_state(), (0, None, "pending".to_string(), 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_after_claim_releases_the_lease_without_completion() {
    let server = CountingServer::start();
    let fixture = DurableEmbeddingFixture::start(&server.endpoint(), 2_000, 1).await;
    let plan = prepare_embedding_batch(
        &fixture.clients,
        vec![fixture.candidate.clone()],
        Instant::now(),
        Utc::now().timestamp_millis(),
    )
    .unwrap();
    let (cancel, cancellation) = tokio::sync::watch::channel(false);
    let writer = fixture.writer.clone();
    let execution =
        tokio::spawn(async move { execute_embedding_batch(&writer, plan, cancellation).await });

    server.wait_for_accept().await;
    assert_eq!(fixture.job_state().0, 1);
    cancel.send_replace(true);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), execution)
            .await
            .unwrap()
            .unwrap(),
        BackgroundJobOutcome::Deferred
    );
    assert_eq!(fixture.job_state(), (1, None, "released".to_string(), 0));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn expired_shared_shutdown_deadline_leaves_embedding_lease_for_expiry() {
    let server = CountingServer::start();
    let fixture = DurableEmbeddingFixture::start(&server.endpoint(), 2_000, 1).await;
    let plan = prepare_embedding_batch(
        &fixture.clients,
        vec![fixture.candidate.clone()],
        Instant::now(),
        Utc::now().timestamp_millis(),
    )
    .unwrap();
    let cancellation = BackgroundCancellation::new();
    let execution_cancellation = cancellation.clone();
    let writer = fixture.writer.clone();
    let execution = tokio::spawn(async move {
        execute_embedding_batch_with_cancellation(&writer, plan, execution_cancellation).await
    });

    server.wait_for_accept().await;
    assert_eq!(fixture.job_state().0, 1);
    cancellation.cancel_until(Instant::now());
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(2), execution)
            .await
            .unwrap()
            .unwrap(),
        BackgroundJobOutcome::Deferred
    );
    let durable = fixture.job_state();
    assert_eq!(durable.0, 1);
    assert!(durable.1.is_some());
    assert_eq!(durable.2, "claimed");
    assert_eq!(durable.3, 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rebuild_shutdown_releases_the_owned_lease_within_the_shared_deadline() {
    let fixture = DurableEmbeddingFixture::start("http://127.0.0.1:9/v1", 250, 1).await;
    let observed_at_unix_ms = Utc::now().timestamp_millis();
    assert!(matches!(
        fixture
            .writer
            .vector_index_until(
                VectorIndexWriterCommand::AuthorizeGeneration {
                    vector_space_id: fixture.vector_space_id.clone(),
                    dimensions: VectorDimensions::new(2).unwrap(),
                    created_at_unix_ms: observed_at_unix_ms,
                },
                Instant::now() + Duration::from_secs(2),
            )
            .await
            .unwrap(),
        VectorIndexWriterAck::GenerationAuthorized(_)
    ));
    let work = fixture
        .read_pool
        .select_building_generations_until(
            fixture.project_uuid,
            fixture.process_instance_id,
            observed_at_unix_ms,
            None,
            1,
            Instant::now() + Duration::from_secs(2),
        )
        .await
        .unwrap()
        .pop()
        .expect("authorized generation must be discoverable");
    assert_eq!(work.lease_disposition, RebuildLeaseDisposition::Unclaimed);

    let cancellation = BackgroundCancellation::new();
    cancellation.cancel_until(Instant::now() + Duration::from_secs(2));
    assert_eq!(
        execute_rebuild(&fixture.writer, work, cancellation).await,
        BackgroundJobOutcome::Deferred
    );
    let released = fixture
        .read_pool
        .select_building_generations_until(
            fixture.project_uuid,
            fixture.process_instance_id,
            Utc::now().timestamp_millis(),
            None,
            1,
            Instant::now() + Duration::from_secs(2),
        )
        .await
        .unwrap()
        .pop()
        .expect("released generation must remain discoverable");
    assert_eq!(
        released.lease_disposition,
        RebuildLeaseDisposition::Reclaimable
    );
    assert!(released.lease.is_some());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn expired_rebuild_shutdown_deadline_leaves_the_owned_fence_for_expiry() {
    let fixture = DurableEmbeddingFixture::start("http://127.0.0.1:9/v1", 250, 1).await;
    let observed_at_unix_ms = Utc::now().timestamp_millis();
    assert!(matches!(
        fixture
            .writer
            .vector_index_until(
                VectorIndexWriterCommand::AuthorizeGeneration {
                    vector_space_id: fixture.vector_space_id.clone(),
                    dimensions: VectorDimensions::new(2).unwrap(),
                    created_at_unix_ms: observed_at_unix_ms,
                },
                Instant::now() + Duration::from_secs(2),
            )
            .await
            .unwrap(),
        VectorIndexWriterAck::GenerationAuthorized(_)
    ));
    assert!(matches!(
        fixture
            .writer
            .vector_index_until(
                VectorIndexWriterCommand::ClaimRebuildLease {
                    vector_space_id: fixture.vector_space_id.clone(),
                    observed_at_unix_ms: observed_at_unix_ms + 1,
                },
                Instant::now() + Duration::from_secs(2),
            )
            .await
            .unwrap(),
        VectorIndexWriterAck::RebuildLeaseClaimed(RebuildLeaseClaimAck::Claimed(_))
    ));
    let work = fixture
        .read_pool
        .select_building_generations_until(
            fixture.project_uuid,
            fixture.process_instance_id,
            observed_at_unix_ms + 2,
            None,
            1,
            Instant::now() + Duration::from_secs(2),
        )
        .await
        .unwrap()
        .pop()
        .expect("owned generation must be discoverable");
    assert_eq!(work.lease_disposition, RebuildLeaseDisposition::Owned);
    let before = Connection::open(&fixture.database_path)
        .unwrap()
        .query_row(
            "SELECT lease_token, lease_expires_at_unix_ms, updated_at_unix_ms,
                    canonical_payload_hash
             FROM vector_index_rebuild_leases WHERE vector_space_id = ?1",
            params![fixture.vector_space_id.as_str()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            },
        )
        .unwrap();

    let cancellation = BackgroundCancellation::new();
    cancellation.cancel_until(Instant::now());
    assert_eq!(
        execute_rebuild(&fixture.writer, work, cancellation).await,
        BackgroundJobOutcome::Deferred
    );
    let after = Connection::open(&fixture.database_path)
        .unwrap()
        .query_row(
            "SELECT lease_token, lease_expires_at_unix_ms, updated_at_unix_ms,
                    canonical_payload_hash
             FROM vector_index_rebuild_leases WHERE vector_space_id = ?1",
            params![fixture.vector_space_id.as_str()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            },
        )
        .unwrap();
    assert_eq!(after, before);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn elapsed_provider_deadline_refuses_claim_and_http() {
    let server = CountingServer::start();
    let fixture = DurableEmbeddingFixture::start(&server.endpoint(), 50, 1).await;
    let operation_origin = Instant::now()
        .checked_sub(Duration::from_millis(75))
        .unwrap();
    let plan = prepare_embedding_batch(
        &fixture.clients,
        vec![fixture.candidate.clone()],
        operation_origin,
        Utc::now().timestamp_millis(),
    )
    .unwrap();
    let (_cancel, cancellation) = tokio::sync::watch::channel(false);

    assert_eq!(
        execute_embedding_batch(&fixture.writer, plan, cancellation).await,
        BackgroundJobOutcome::Deferred
    );
    assert_eq!(server.accepted(), 0);
    assert_eq!(fixture.job_state(), (0, None, "pending".to_string(), 0));
}

#[tokio::test(start_paused = true)]
async fn retry_schedule_uses_exponential_absolute_eligibility_under_paused_time() {
    let mut resolved_at_unix_ms = 10_000;
    let started = tokio::time::Instant::now();
    for (attempt_count, delay) in [(1, 1_000), (2, 2_000), (3, 4_000), (4, 8_000)] {
        let EmbeddingJobResolutionKind::RetryScheduled {
            next_eligible_at_unix_ms,
            ..
        } = embedding_failure_resolution_kind(
            EmbedderFailureDisposition::Retryable,
            "embedder_timeout",
            attempt_count,
            resolved_at_unix_ms,
        )
        else {
            panic!("retryable attempt {attempt_count} did not schedule a retry");
        };
        assert_eq!(next_eligible_at_unix_ms, resolved_at_unix_ms + delay);
        tokio::time::advance(Duration::from_millis(delay as u64)).await;
        resolved_at_unix_ms = next_eligible_at_unix_ms;
    }
    assert_eq!(
        tokio::time::Instant::now().duration_since(started),
        Duration::from_secs(15)
    );
}

#[test]
fn provider_failure_kind_releases_cancellation_and_quarantines_permanent_failures() {
    assert_eq!(
        embedding_failure_resolution_kind(
            EmbedderFailureDisposition::Retryable,
            "embedder_cancelled",
            1,
            1_000,
        ),
        EmbeddingJobResolutionKind::Released
    );
    assert_eq!(
        embedding_failure_resolution_kind(
            EmbedderFailureDisposition::Retryable,
            "embedder_timeout",
            5,
            1_000,
        ),
        EmbeddingJobResolutionKind::RetryScheduled {
            stable_error_class: "embedder_timeout".to_string(),
            next_eligible_at_unix_ms: 17_000,
        }
    );
    assert_eq!(
        embedding_failure_resolution_kind(
            EmbedderFailureDisposition::Quarantine,
            "embedder_authentication",
            1,
            1_000,
        ),
        EmbeddingJobResolutionKind::Quarantined {
            stable_error_class: "embedder_authentication".to_string(),
        }
    );
}

#[test]
fn resolution_acknowledgements_are_not_collapsed_to_applied() {
    let job = candidate('a', &space('1')).job;
    assert_eq!(
        embedding_resolution_outcome(EmbeddingJobResolutionAck::Applied {
            job: job.clone(),
            state: EmbeddingJobResolvedState::Released,
            state_event_hash: "a".repeat(64),
        }),
        BackgroundJobOutcome::Applied
    );
    assert_eq!(
        embedding_resolution_outcome(EmbeddingJobResolutionAck::AlreadyApplied {
            job,
            state: EmbeddingJobResolvedState::Released,
            state_event_hash: "b".repeat(64),
        }),
        BackgroundJobOutcome::AlreadyApplied
    );
    assert_eq!(
        embedding_resolution_outcome(EmbeddingJobResolutionAck::StaleLease),
        BackgroundJobOutcome::Stale
    );
    assert_eq!(
        embedding_resolution_outcome(EmbeddingJobResolutionAck::TransactionNotStarted),
        BackgroundJobOutcome::Deferred
    );
    assert_eq!(
        embedding_resolution_outcome(EmbeddingJobResolutionAck::Conflict),
        BackgroundJobOutcome::PermanentFailure
    );
    assert_eq!(
        embedding_resolution_outcome(EmbeddingJobResolutionAck::OriginatingProcessNotLive),
        BackgroundJobOutcome::PermanentFailure
    );
}

#[test]
fn backfill_command_uses_fresh_identities_and_never_predates_terminal() {
    let candidate = BackfillWorkCandidate {
        mapping: FrozenMappingKey::new(Uuid::now_v7(), "a".repeat(64), "pool-a", "b".repeat(64))
            .unwrap(),
        vector_space_id: space('c'),
        shadow_attempt_id: Uuid::now_v7(),
        terminal_at_unix_ms: 50,
    };
    let command = prepare_backfill_command(&candidate, 40).unwrap();
    assert_eq!(command.backfilled_at_unix_ms, 50);
    assert_eq!(command.mapping, candidate.mapping);
    assert_eq!(command.shadow_attempt_id, candidate.shadow_attempt_id);
    let identities = BTreeSet::from([
        command.shadow_attempt_id,
        command.evidence_vector_link_id,
        command.evidence_link_state_event_id,
        command.materialization_state_event_id,
        command.embedding_job_state_event_id,
        command.conflict_health_event_id,
    ]);
    assert_eq!(identities.len(), 6);
    assert!(
        identities
            .iter()
            .all(|identity| identity.get_version_num() == 7)
    );

    let mut corrupt = candidate;
    corrupt.terminal_at_unix_ms = -1;
    assert_eq!(
        prepare_backfill_command(&corrupt, 40).unwrap_err(),
        BackgroundJobOutcome::PermanentFailure
    );
}

#[test]
fn backfill_acknowledgements_are_exhaustive() {
    assert_eq!(
        backfill_outcome(VectorBackfillAck::Applied),
        BackgroundJobOutcome::Applied
    );
    assert_eq!(
        backfill_outcome(VectorBackfillAck::AlreadyApplied),
        BackgroundJobOutcome::AlreadyApplied
    );
    for acknowledgement in [
        VectorBackfillAck::NotFound,
        VectorBackfillAck::MappingNotFound,
        VectorBackfillAck::MappingNotCurrent,
    ] {
        assert_eq!(
            backfill_outcome(acknowledgement),
            BackgroundJobOutcome::Stale
        );
    }
    assert_eq!(
        backfill_outcome(VectorBackfillAck::TransactionNotStarted),
        BackgroundJobOutcome::Deferred
    );
    for acknowledgement in [
        VectorBackfillAck::Conflict,
        VectorBackfillAck::OriginatingProcessNotLive,
    ] {
        assert_eq!(
            backfill_outcome(acknowledgement),
            BackgroundJobOutcome::PermanentFailure
        );
    }
}

#[test]
fn vector_space_recovery_rejects_partial_objects_and_authorizes_recoverable_drift() {
    let healthy = VectorSpaceInspection {
        vector_space_id: space('1'),
        dimensions: VectorDimensions::new(2).unwrap(),
        recovery: None,
    };
    assert_eq!(prepare_vector_space_recovery(healthy, 10).unwrap(), None);

    let partial = VectorSpaceInspection {
        vector_space_id: space('1'),
        dimensions: VectorDimensions::new(2).unwrap(),
        recovery: Some(VectorSpaceRecoveryReason::PartialObjects),
    };
    assert_eq!(
        prepare_vector_space_recovery(partial, 10).unwrap_err(),
        BackgroundJobOutcome::PermanentFailure
    );

    let recoverable = VectorSpaceInspection {
        vector_space_id: space('1'),
        dimensions: VectorDimensions::new(2).unwrap(),
        recovery: Some(VectorSpaceRecoveryReason::FingerprintMismatch),
    };
    assert_eq!(
        prepare_vector_space_recovery(recoverable, 10).unwrap(),
        Some(VectorIndexWriterCommand::AuthorizeGeneration {
            vector_space_id: space('1'),
            dimensions: VectorDimensions::new(2).unwrap(),
            created_at_unix_ms: 10,
        })
    );
}
