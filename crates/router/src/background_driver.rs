// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Concrete bounded discovery driver for Router background work.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::Utc;

use crate::active_evaluator::{ActiveEvaluatorOutcome, execute_active_look};
use crate::background::{
    BackgroundDriver, BackgroundFailure, BackgroundPass, BackgroundPassContext,
    BackgroundPassFuture, BackgroundResources, BackgroundWork,
};
use crate::background_jobs::{
    BackgroundJobOutcome, execute_backfill, execute_embedding_batch_with_cancellation,
    execute_failure_propagation, execute_materialization_with_cancellation, execute_rebuild,
    execute_retired_cleanup, execute_vector_space_recovery, prepare_embedding_batch,
};
use crate::ledger::read_pool::LedgerReadPool;
use crate::ledger::repository::background_work::{
    BACKGROUND_WORK_PAGE_MAX, BackfillWorkCursor, EmbeddingWorkCandidate, EmbeddingWorkCursor,
    FailurePropagationCursor, MaterializationWorkCursor,
};
use crate::ledger::repository::vector_work::{
    VECTOR_GENERATION_WORK_PAGE_MAX, VectorGenerationWorkCursor, VectorSpaceWorkCursor,
};
use crate::ledger::writer::LedgerWriterClient;

const READ_TIMEOUT: Duration = Duration::from_secs(5);
const LANE_COUNT: usize = 8;
const DISCOVERY_FAILURE: BackgroundFailure =
    BackgroundFailure::new("router.background.discovery_failure");
const BACKFILL_FAILURE: BackgroundFailure =
    BackgroundFailure::new("router.background.backfill_failure");
const EMBEDDING_FAILURE: BackgroundFailure =
    BackgroundFailure::new("router.background.embedding_failure");
const MATERIALIZATION_FAILURE: BackgroundFailure =
    BackgroundFailure::new("router.background.materialization_failure");
const FANOUT_FAILURE: BackgroundFailure =
    BackgroundFailure::new("router.background.failure_fanout_failure");
const REBUILD_FAILURE: BackgroundFailure =
    BackgroundFailure::new("router.background.rebuild_failure");
const ACTIVE_EVALUATION_FAILURE: BackgroundFailure =
    BackgroundFailure::new("router.background.active_evaluation_failure");

#[derive(Default)]
struct DriverCursors {
    vector_space: Option<VectorSpaceWorkCursor>,
    backfill: Option<BackfillWorkCursor>,
    embedding: Option<EmbeddingWorkCursor>,
    materialization: Option<MaterializationWorkCursor>,
    failure_propagation: Option<FailurePropagationCursor>,
    building: Option<VectorGenerationWorkCursor>,
    retired: Option<VectorGenerationWorkCursor>,
}

/// Stateful keyset driver. Cursors are advisory and intentionally process-local.
pub(crate) struct RouterBackgroundDriver {
    writer: LedgerWriterClient,
    read_pool: LedgerReadPool,
    vector_registry: Arc<crate::ledger::repository::vector_registry::VectorRegistryEnsure>,
    embedder_clients: Arc<crate::embedder::FrozenEmbedderClients>,
    process_instance_id: uuid::Uuid,
    startup_schema_findings: usize,
    cursors: Arc<Mutex<DriverCursors>>,
    active_evaluator_running: Arc<AtomicBool>,
}

impl RouterBackgroundDriver {
    pub(crate) fn new(resources: BackgroundResources) -> Self {
        let startup_schema_findings = resources
            .initial_schema_report()
            .missing_building
            .len()
            .saturating_add(resources.initial_schema_report().missing_active.len())
            .saturating_add(resources.initial_schema_report().missing_retired.len());
        Self {
            writer: resources.writer().clone(),
            read_pool: resources.read_pool().clone(),
            vector_registry: resources.vector_registry().clone(),
            embedder_clients: resources.embedder_clients().clone(),
            process_instance_id: resources.process_instance_id(),
            startup_schema_findings,
            cursors: Arc::new(Mutex::new(DriverCursors::default())),
            active_evaluator_running: Arc::new(AtomicBool::new(false)),
        }
    }
}

impl BackgroundDriver for RouterBackgroundDriver {
    fn run_pass(
        &self,
        pass: BackgroundPass,
        context: BackgroundPassContext,
    ) -> BackgroundPassFuture {
        let writer = self.writer.clone();
        let read_pool = self.read_pool.clone();
        let registry = self.vector_registry.clone();
        let clients = self.embedder_clients.clone();
        let process_instance_id = self.process_instance_id;
        let startup_schema_findings = self.startup_schema_findings;
        let cursors = self.cursors.clone();
        let active_evaluator_running = self.active_evaluator_running.clone();
        Box::pin(async move {
            if context.cancellation().is_cancelled() || context.work_budget() == 0 {
                return Ok(Vec::new());
            }
            let budget = context.work_budget();
            let lane_budget = budget.div_ceil(LANE_COUNT).max(1);
            let mut planned = Vec::with_capacity(budget);

            plan_active_evaluation(
                &read_pool,
                &writer,
                &registry,
                &active_evaluator_running,
                budget,
                &mut planned,
            )
            .await?;

            let vector_limit = if pass == BackgroundPass::Startup && startup_schema_findings > 0 {
                lane_budget.max(startup_schema_findings)
            } else {
                lane_budget
            }
            .min(VECTOR_GENERATION_WORK_PAGE_MAX);
            plan_vector_space_recovery(
                &read_pool,
                &writer,
                &registry,
                process_instance_id,
                &cursors,
                vector_limit,
                budget,
                &mut planned,
            )
            .await?;
            plan_backfill(
                &read_pool,
                &writer,
                &registry,
                &cursors,
                lane_budget,
                budget,
                &mut planned,
            )
            .await?;
            plan_embeddings(
                &read_pool,
                &writer,
                &registry,
                &clients,
                &cursors,
                lane_budget,
                budget,
                &mut planned,
            )
            .await?;
            plan_materializations(
                &read_pool,
                &writer,
                &registry,
                &cursors,
                lane_budget,
                budget,
                &mut planned,
            )
            .await?;
            plan_failure_propagation(
                &read_pool,
                &writer,
                &registry,
                &cursors,
                lane_budget,
                budget,
                &mut planned,
            )
            .await?;
            plan_rebuilds(
                &read_pool,
                &writer,
                &registry,
                process_instance_id,
                &cursors,
                lane_budget,
                budget,
                &mut planned,
            )
            .await?;
            plan_retired_cleanup(
                &read_pool,
                &writer,
                &registry,
                process_instance_id,
                &cursors,
                lane_budget,
                budget,
                &mut planned,
            )
            .await?;
            Ok(planned)
        })
    }
}

struct ActiveEvaluatorReservation {
    running: Arc<AtomicBool>,
}

impl ActiveEvaluatorReservation {
    fn acquire(running: &Arc<AtomicBool>) -> Option<Self> {
        running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self {
                running: running.clone(),
            })
    }
}

impl Drop for ActiveEvaluatorReservation {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
    }
}

async fn plan_active_evaluation(
    read_pool: &LedgerReadPool,
    writer: &LedgerWriterClient,
    registry: &crate::ledger::repository::vector_registry::VectorRegistryEnsure,
    running: &Arc<AtomicBool>,
    budget: usize,
    planned: &mut Vec<BackgroundWork>,
) -> Result<(), BackgroundFailure> {
    if planned.len() >= budget || running.load(Ordering::Acquire) {
        return Ok(());
    }
    let work = read_pool
        .select_active_look_work_until(
            registry.project_uuid,
            registry.config_generation_id.clone(),
            1,
            read_deadline()?,
        )
        .await
        .map_err(|_| DISCOVERY_FAILURE)?
        .into_iter()
        .next();
    let (Some(work), Some(reservation)) = (work, ActiveEvaluatorReservation::acquire(running))
    else {
        return Ok(());
    };
    let writer = writer.clone();
    planned.push(BackgroundWork::new(move |cancellation| async move {
        let _reservation = reservation;
        match execute_active_look(&writer, work, cancellation).await {
            ActiveEvaluatorOutcome::Applied
            | ActiveEvaluatorOutcome::AlreadyApplied
            | ActiveEvaluatorOutcome::Deferred
            | ActiveEvaluatorOutcome::Stale
            | ActiveEvaluatorOutcome::Cancelled => Ok(()),
            ActiveEvaluatorOutcome::PermanentFailure => Err(ACTIVE_EVALUATION_FAILURE),
        }
    }));
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn plan_vector_space_recovery(
    read_pool: &LedgerReadPool,
    writer: &LedgerWriterClient,
    registry: &crate::ledger::repository::vector_registry::VectorRegistryEnsure,
    process_instance_id: uuid::Uuid,
    cursors: &Mutex<DriverCursors>,
    limit: usize,
    budget: usize,
    planned: &mut Vec<BackgroundWork>,
) -> Result<(), BackgroundFailure> {
    if planned.len() >= budget {
        return Ok(());
    }
    let after = lock_cursors(cursors).vector_space.clone();
    let page = read_pool
        .inspect_vector_spaces_until(
            registry.project_uuid,
            process_instance_id,
            now_unix_ms(),
            after,
            limit,
            read_deadline()?,
        )
        .await
        .map_err(|_| DISCOVERY_FAILURE)?;
    lock_cursors(cursors).vector_space = page.last().map(|work| work.cursor());
    for work in page.into_iter().filter(|work| work.recovery.is_some()) {
        if planned.len() >= budget {
            break;
        }
        let writer = writer.clone();
        planned.push(BackgroundWork::new(move |_| async move {
            outcome(
                execute_vector_space_recovery(&writer, work).await,
                REBUILD_FAILURE,
            )
        }));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn plan_backfill(
    read_pool: &LedgerReadPool,
    writer: &LedgerWriterClient,
    registry: &crate::ledger::repository::vector_registry::VectorRegistryEnsure,
    cursors: &Mutex<DriverCursors>,
    limit: usize,
    budget: usize,
    planned: &mut Vec<BackgroundWork>,
) -> Result<(), BackgroundFailure> {
    if planned.len() >= budget {
        return Ok(());
    }
    let after = lock_cursors(cursors).backfill.clone();
    let page = read_pool
        .select_backfill_work_until(
            registry.project_uuid,
            registry.config_generation_id.clone(),
            after,
            limit.min(BACKGROUND_WORK_PAGE_MAX),
            read_deadline()?,
        )
        .await
        .map_err(|_| DISCOVERY_FAILURE)?;
    lock_cursors(cursors).backfill = page.last().map(|work| work.cursor());
    for work in page {
        if planned.len() >= budget {
            break;
        }
        let writer = writer.clone();
        planned.push(BackgroundWork::new(move |_| async move {
            outcome(execute_backfill(&writer, work).await, BACKFILL_FAILURE)
        }));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn plan_embeddings(
    read_pool: &LedgerReadPool,
    writer: &LedgerWriterClient,
    registry: &crate::ledger::repository::vector_registry::VectorRegistryEnsure,
    clients: &Arc<crate::embedder::FrozenEmbedderClients>,
    cursors: &Mutex<DriverCursors>,
    task_limit: usize,
    budget: usize,
    planned: &mut Vec<BackgroundWork>,
) -> Result<(), BackgroundFailure> {
    if planned.len() >= budget {
        return Ok(());
    }
    let after = lock_cursors(cursors).embedding.clone();
    let mut page = read_pool
        .select_embedding_work_until(
            registry.project_uuid,
            registry.config_generation_id.clone(),
            now_unix_ms(),
            after,
            BACKGROUND_WORK_PAGE_MAX,
            read_deadline()?,
        )
        .await
        .map_err(|_| DISCOVERY_FAILURE)?;
    lock_cursors(cursors).embedding = page.last().map(|work| work.cursor());
    let mut scheduled = 0;
    while !page.is_empty() && planned.len() < budget && scheduled < task_limit {
        let end = embedding_batch_end(&page, 0);
        let batch = page.drain(..end).collect();
        let prepared = match prepare_embedding_batch(clients, batch, Instant::now(), now_unix_ms())
        {
            Ok(prepared) => prepared,
            Err(BackgroundJobOutcome::Deferred | BackgroundJobOutcome::Stale) => continue,
            Err(BackgroundJobOutcome::PermanentFailure) => return Err(EMBEDDING_FAILURE),
            Err(BackgroundJobOutcome::Applied | BackgroundJobOutcome::AlreadyApplied) => {
                return Err(EMBEDDING_FAILURE);
            }
        };
        let writer = writer.clone();
        planned.push(BackgroundWork::new(move |cancellation| async move {
            outcome(
                execute_embedding_batch_with_cancellation(&writer, prepared, cancellation).await,
                EMBEDDING_FAILURE,
            )
        }));
        scheduled += 1;
    }
    Ok(())
}

fn embedding_batch_end(page: &[EmbeddingWorkCandidate], start: usize) -> usize {
    let first = &page[start];
    let maximum = start.saturating_add(first.batch_size).min(page.len());
    let mut end = start + 1;
    while end < maximum && page[end].job.vector_space_id == first.job.vector_space_id {
        end += 1;
    }
    end
}

#[allow(clippy::too_many_arguments)]
async fn plan_materializations(
    read_pool: &LedgerReadPool,
    writer: &LedgerWriterClient,
    registry: &crate::ledger::repository::vector_registry::VectorRegistryEnsure,
    cursors: &Mutex<DriverCursors>,
    limit: usize,
    budget: usize,
    planned: &mut Vec<BackgroundWork>,
) -> Result<(), BackgroundFailure> {
    if planned.len() >= budget {
        return Ok(());
    }
    let after = lock_cursors(cursors).materialization.clone();
    let page = read_pool
        .select_materialization_work_until(
            registry.project_uuid,
            now_unix_ms(),
            after,
            limit.min(BACKGROUND_WORK_PAGE_MAX),
            read_deadline()?,
        )
        .await
        .map_err(|_| DISCOVERY_FAILURE)?;
    lock_cursors(cursors).materialization = page.last().map(|work| work.cursor());
    for work in page {
        if planned.len() >= budget {
            break;
        }
        let writer = writer.clone();
        planned.push(BackgroundWork::new(move |cancellation| async move {
            outcome(
                execute_materialization_with_cancellation(&writer, work, cancellation).await,
                MATERIALIZATION_FAILURE,
            )
        }));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn plan_failure_propagation(
    read_pool: &LedgerReadPool,
    writer: &LedgerWriterClient,
    registry: &crate::ledger::repository::vector_registry::VectorRegistryEnsure,
    cursors: &Mutex<DriverCursors>,
    limit: usize,
    budget: usize,
    planned: &mut Vec<BackgroundWork>,
) -> Result<(), BackgroundFailure> {
    if planned.len() >= budget {
        return Ok(());
    }
    let after = lock_cursors(cursors).failure_propagation.clone();
    let page = read_pool
        .select_failure_propagation_work_until(
            registry.project_uuid,
            after,
            limit.min(BACKGROUND_WORK_PAGE_MAX),
            read_deadline()?,
        )
        .await
        .map_err(|_| DISCOVERY_FAILURE)?;
    lock_cursors(cursors).failure_propagation = page.last().map(|work| work.cursor());
    for work in page {
        if planned.len() >= budget {
            break;
        }
        let writer = writer.clone();
        planned.push(BackgroundWork::new(move |_| async move {
            outcome(
                execute_failure_propagation(&writer, work).await,
                FANOUT_FAILURE,
            )
        }));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn plan_rebuilds(
    read_pool: &LedgerReadPool,
    writer: &LedgerWriterClient,
    registry: &crate::ledger::repository::vector_registry::VectorRegistryEnsure,
    process_instance_id: uuid::Uuid,
    cursors: &Mutex<DriverCursors>,
    limit: usize,
    budget: usize,
    planned: &mut Vec<BackgroundWork>,
) -> Result<(), BackgroundFailure> {
    if planned.len() >= budget {
        return Ok(());
    }
    let after = lock_cursors(cursors).building.clone();
    let page = read_pool
        .select_building_generations_until(
            registry.project_uuid,
            process_instance_id,
            now_unix_ms(),
            after,
            limit.min(VECTOR_GENERATION_WORK_PAGE_MAX),
            read_deadline()?,
        )
        .await
        .map_err(|_| DISCOVERY_FAILURE)?;
    lock_cursors(cursors).building = page.last().map(|work| work.cursor());
    for work in page {
        if planned.len() >= budget {
            break;
        }
        let writer = writer.clone();
        planned.push(BackgroundWork::new(move |cancellation| async move {
            outcome(
                execute_rebuild(&writer, work, cancellation).await,
                REBUILD_FAILURE,
            )
        }));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn plan_retired_cleanup(
    read_pool: &LedgerReadPool,
    writer: &LedgerWriterClient,
    registry: &crate::ledger::repository::vector_registry::VectorRegistryEnsure,
    process_instance_id: uuid::Uuid,
    cursors: &Mutex<DriverCursors>,
    limit: usize,
    budget: usize,
    planned: &mut Vec<BackgroundWork>,
) -> Result<(), BackgroundFailure> {
    if planned.len() >= budget {
        return Ok(());
    }
    let after = lock_cursors(cursors).retired.clone();
    let page = read_pool
        .select_retired_generations_until(
            registry.project_uuid,
            process_instance_id,
            now_unix_ms(),
            after,
            limit.min(VECTOR_GENERATION_WORK_PAGE_MAX),
            read_deadline()?,
        )
        .await
        .map_err(|_| DISCOVERY_FAILURE)?;
    lock_cursors(cursors).retired = page.last().map(|work| work.cursor());
    for work in page {
        if planned.len() >= budget {
            break;
        }
        let writer = writer.clone();
        planned.push(BackgroundWork::new(move |_| async move {
            outcome(
                execute_retired_cleanup(&writer, work).await,
                REBUILD_FAILURE,
            )
        }));
    }
    Ok(())
}

fn outcome(
    outcome: BackgroundJobOutcome,
    permanent_failure: BackgroundFailure,
) -> Result<(), BackgroundFailure> {
    match outcome {
        BackgroundJobOutcome::Applied
        | BackgroundJobOutcome::AlreadyApplied
        | BackgroundJobOutcome::Deferred
        | BackgroundJobOutcome::Stale => Ok(()),
        BackgroundJobOutcome::PermanentFailure => Err(permanent_failure),
    }
}

fn read_deadline() -> Result<Instant, BackgroundFailure> {
    Instant::now()
        .checked_add(READ_TIMEOUT)
        .ok_or(DISCOVERY_FAILURE)
}

fn now_unix_ms() -> i64 {
    Utc::now().timestamp_millis().max(0)
}

fn lock_cursors(cursors: &Mutex<DriverCursors>) -> std::sync::MutexGuard<'_, DriverCursors> {
    cursors.lock().unwrap_or_else(|error| error.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedding_batches_never_cross_space_or_profile_batch_boundaries() {
        fn candidate(space: char, batch_size: usize, job: char) -> EmbeddingWorkCandidate {
            use crate::canonical_query::{
                CanonicalRoutingQueryArtifactV1, CanonicalRoutingQueryV1, CanonicalTaskV1,
            };
            use crate::ledger::repository::embedding::EmbeddingJobSnapshot;
            use crate::ledger::repository::vector_catalog::CanonicalQuerySnapshot;

            let canonical_bytes = format!("{{\"task\":\"{job}\"}}").into_bytes();
            let hash = crate::fingerprint::sha256_hex(&canonical_bytes);
            let query = CanonicalRoutingQueryV1 {
                schema: "test-query@1".to_string(),
                instructions: Vec::new(),
                current_task: CanonicalTaskV1 {
                    text: job.to_string(),
                },
                bounded_context: Vec::new(),
                tool_schema_fingerprint: "0".repeat(64),
                response_schema_fingerprint: None,
                required_capabilities: Vec::new(),
                position_features: None,
            };
            EmbeddingWorkCandidate {
                job: EmbeddingJobSnapshot {
                    embedding_job_id: job.to_string().repeat(64),
                    vector_space_id: space.to_string().repeat(64),
                    canonical_query_hash: hash.clone(),
                    content_hash: "c".repeat(64),
                    attempt_generation: 1,
                    attempt_count: 0,
                    next_eligible_at_unix_ms: 0,
                    terminal_error_class: None,
                    failure_propagation_cursor: None,
                    failure_propagation_complete: false,
                    reset_actor: None,
                    reset_reason: None,
                    canonical_payload_hash: "d".repeat(64),
                },
                canonical_query: CanonicalQuerySnapshot {
                    artifact: CanonicalRoutingQueryArtifactV1 {
                        query,
                        canonical_bytes,
                        canonical_query_hash: hash,
                    },
                    created_at_unix_ms: 0,
                },
                embedder_profile_version_id: "e".repeat(64),
                batch_size,
            }
        }

        let page = vec![
            candidate('a', 2, '1'),
            candidate('a', 2, '2'),
            candidate('a', 2, '3'),
            candidate('b', 2, '4'),
        ];
        assert_eq!(embedding_batch_end(&page, 0), 2);
        assert_eq!(embedding_batch_end(&page, 2), 3);
        assert_eq!(embedding_batch_end(&page, 3), 4);
    }

    #[test]
    fn only_permanent_job_outcomes_fail_the_supervisor_task() {
        for nonterminal in [
            BackgroundJobOutcome::Applied,
            BackgroundJobOutcome::AlreadyApplied,
            BackgroundJobOutcome::Deferred,
            BackgroundJobOutcome::Stale,
        ] {
            assert_eq!(outcome(nonterminal, REBUILD_FAILURE), Ok(()));
        }
        assert_eq!(
            outcome(BackgroundJobOutcome::PermanentFailure, REBUILD_FAILURE),
            Err(REBUILD_FAILURE)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn embedding_discovery_ignores_jobs_outside_the_current_mapping_authority() {
        use std::io::ErrorKind;
        use std::net::{Ipv4Addr, TcpListener};
        use std::sync::atomic::{AtomicUsize, Ordering};

        use rusqlite::params;
        use serde_json::json;
        use tempfile::tempdir;
        use uuid::Uuid;

        use crate::canonical_json::canonical_json;
        use crate::canonical_query::{
            CanonicalRoutingQueryArtifactV1, CanonicalRoutingQueryV1, CanonicalTaskV1,
        };
        use crate::config::RouterConfig;
        use crate::embedder::build_frozen_embedder_clients_for_test;
        use crate::embedding_identity::CANONICAL_ROUTING_QUERY_SCHEMA_V1;
        use crate::fingerprint::sha256_hex;
        use crate::ledger::repository::embedding::{EmbeddingJobCreate, EmbeddingJobCreateAck};
        use crate::ledger::repository::vector_catalog::ensure_canonical_query;
        use crate::ledger::repository::{ActivatedLedger, LedgerRepository};
        use crate::ledger::writer::LedgerWriterOwner;

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        listener.set_nonblocking(true).unwrap();
        let address = listener.local_addr().unwrap();
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let original: RouterConfig = serde_json::from_value(json!({
            "version": 1,
            "mode": "shadow",
            "project_id": "background-current-mapping-authority",
            "database_path": database_path,
            "embedders": [{
                "id": "embedding-main",
                "base_url": format!("http://{address}/v1"),
                "model": "embed-model-a",
                "provider_revision": "revision-1",
                "dimensions": 2,
                "api_key_env": "NEMO_RELAY_ROUTER_STALE_EMBEDDING_SECRET",
                "timeout_ms": 1000,
                "max_in_flight": 1,
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
        .unwrap();
        let mut first = LedgerRepository::activate_at(&original, 1_000).unwrap();
        let project_uuid = first.identity.project_uuid;
        let original_generation = first.identity.config_generation_id.clone();
        let original_space = first.registry.spaces.keys().next().unwrap().clone();
        let query = CanonicalRoutingQueryV1 {
            schema: CANONICAL_ROUTING_QUERY_SCHEMA_V1.to_string(),
            instructions: Vec::new(),
            current_task: CanonicalTaskV1 {
                text: "pending under the original mapping".to_string(),
            },
            bounded_context: Vec::new(),
            tool_schema_fingerprint: "0".repeat(64),
            response_schema_fingerprint: None,
            required_capabilities: Vec::new(),
            position_features: None,
        };
        let canonical_bytes = canonical_json(&serde_json::to_value(&query).unwrap())
            .unwrap()
            .into_bytes();
        let canonical_query_hash = sha256_hex(&canonical_bytes);
        let artifact = CanonicalRoutingQueryArtifactV1 {
            query,
            canonical_bytes,
            canonical_query_hash: canonical_query_hash.clone(),
        };
        let mut connection = rusqlite::Connection::open(&database_path).unwrap();
        let transaction = connection.transaction().unwrap();
        ensure_canonical_query(&transaction, &artifact, 1_001).unwrap();
        transaction.commit().unwrap();
        drop(connection);
        let create = EmbeddingJobCreate::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            original_space.as_str(),
            canonical_query_hash.clone(),
            canonical_query_hash,
            1_002,
        )
        .unwrap();
        assert!(matches!(
            first.repository.create_embedding_job(&create).unwrap(),
            EmbeddingJobCreateAck::Applied(_)
        ));
        let embedding_job_id = create.embedding_job_id().to_string();
        drop(first);

        let mut changed = original.clone();
        changed.embedders[0].model = "embed-model-b".to_string();
        changed.embedders[0].api_key_env = None;
        let current = LedgerRepository::activate_at(&changed, 2_000).unwrap();
        assert_ne!(current.identity.config_generation_id, original_generation);
        assert_ne!(
            current.registry.spaces.keys().next().unwrap(),
            &original_space
        );
        let credential_lookups = AtomicUsize::new(0);
        let clients = Arc::new(
            build_frozen_embedder_clients_for_test(
                &changed,
                &current.registry,
                |_| {
                    credential_lookups.fetch_add(1, Ordering::AcqRel);
                    Err(())
                },
                |_, _| Ok(Vec::new()),
            )
            .unwrap(),
        );
        assert_eq!(credential_lookups.load(Ordering::Acquire), 0);
        let current_registry = current.registry.clone();
        let ActivatedLedger { repository, .. } = current;
        let read_pool = LedgerReadPool::open(&database_path).unwrap();
        let (writer_owner, writer) = LedgerWriterOwner::start(repository, 16).unwrap();
        let mut planned = Vec::new();
        plan_embeddings(
            &read_pool,
            &writer,
            &current_registry,
            &clients,
            &Mutex::new(DriverCursors::default()),
            1,
            1,
            &mut planned,
        )
        .await
        .expect("historical embedding work must not become a global driver failure");
        assert!(planned.is_empty());
        assert!(matches!(
            listener.accept(),
            Err(error) if error.kind() == ErrorKind::WouldBlock
        ));
        assert_eq!(
            rusqlite::Connection::open(&database_path)
                .unwrap()
                .query_row(
                    "SELECT job.attempt_count, job.lease_token,
                            (SELECT state FROM embedding_job_state_events
                             WHERE embedding_job_id = job.embedding_job_id
                             ORDER BY event_seq DESC LIMIT 1)
                     FROM embedding_jobs AS job WHERE job.embedding_job_id = ?1",
                    params![embedding_job_id],
                    |row| Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, String>(2)?
                    )),
                )
                .unwrap(),
            (0, None, "pending".to_string())
        );
        read_pool.abort();
        writer_owner.abort();

        let mut restored_config = original.clone();
        restored_config.retention_days += 1;
        let restored = LedgerRepository::activate_at(&restored_config, 3_000).unwrap();
        assert_ne!(restored.identity.config_generation_id, original_generation);
        assert_ne!(
            restored.identity.config_generation_id,
            current_registry.config_generation_id
        );
        assert_eq!(
            restored.registry.spaces.keys().next().unwrap(),
            &original_space
        );
        let restored_connection = rusqlite::Connection::open(&database_path).unwrap();
        let discovered = crate::ledger::repository::background_work::select_embedding_work(
            &restored_connection,
            project_uuid,
            &restored.identity.config_generation_id,
            3_001,
            None,
            1,
        )
        .unwrap();
        assert_eq!(discovered.len(), 1);
        assert_eq!(discovered[0].job.embedding_job_id, embedding_job_id);
        let restored_clients = build_frozen_embedder_clients_for_test(
            &restored_config,
            &restored.registry,
            |_| Ok("restored-secret".to_string()),
            |_, _| Ok(Vec::new()),
        )
        .unwrap();
        assert!(
            prepare_embedding_batch(&restored_clients, discovered, Instant::now(), 3_001,).is_ok()
        );
        assert!(matches!(
            listener.accept(),
            Err(error) if error.kind() == ErrorKind::WouldBlock
        ));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn startup_and_completion_hints_build_the_initial_generation_source_only() {
        use rusqlite::OptionalExtension;
        use serde_json::json;
        use tempfile::tempdir;

        use crate::background::BackgroundRuntime;
        use crate::config::RouterConfig;
        use crate::embedder::build_frozen_embedder_clients;
        use crate::ledger::read_pool::LedgerReadPool;
        use crate::ledger::repository::{ActivatedLedger, LedgerRepository};
        use crate::ledger::writer::LedgerWriterOwner;
        use crate::provider_admission::ProviderAdmissionGate;

        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let config: RouterConfig = serde_json::from_value(json!({
            "version": 1,
            "mode": "shadow",
            "project_id": "background-driver-bootstrap",
            "database_path": database_path,
            "embedders": [{
                "id": "embedding-main",
                "base_url": "http://127.0.0.1:9/v1",
                "model": "embed-model",
                "provider_revision": "revision-1",
                "dimensions": 2,
                "timeout_ms": 1000,
                "max_in_flight": 1,
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
        .unwrap();
        let ActivatedLedger {
            repository,
            identity,
            registry,
            schema_report,
            ..
        } = LedgerRepository::activate(&config).unwrap();
        let vector_space_id = registry.spaces.keys().next().unwrap().clone();
        let clients = Arc::new(build_frozen_embedder_clients(&config, &registry).unwrap());
        let read_pool = LedgerReadPool::open(&database_path).unwrap();
        let (writer_owner, writer) = LedgerWriterOwner::start(repository, 32).unwrap();
        let resources = BackgroundResources::new(
            writer,
            read_pool.clone(),
            identity.process_instance_id,
            Arc::new(registry),
            clients,
            schema_report,
        );
        let driver = Arc::new(RouterBackgroundDriver::new(resources));
        let mut background = BackgroundRuntime::start(
            &tokio::runtime::Handle::current(),
            driver,
            ProviderAdmissionGate::initially_open_for_test(),
            Arc::new(|_| {}),
            8,
        );

        let wait_deadline = Instant::now() + Duration::from_secs(5);
        loop {
            let active = rusqlite::Connection::open(&database_path)
                .ok()
                .and_then(|connection| {
                    connection
                        .query_row(
                            "SELECT state FROM vector_index_manifest
                             WHERE vector_space_id = ?1 ORDER BY generation DESC LIMIT 1",
                            [vector_space_id.as_str()],
                            |row| row.get::<_, String>(0),
                        )
                        .optional()
                        .ok()
                        .flatten()
                });
            if active.as_deref() == Some("active") {
                break;
            }
            assert!(
                Instant::now() < wait_deadline,
                "background rebuild did not activate the initial generation"
            );
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        background.stop();
        let exit = background.join().await.unwrap();
        assert!(exit.first_failure().is_none());
        read_pool.abort();
        writer_owner.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn provider_batch_claims_last_and_persists_one_authoritative_vector() {
        use std::io::{Read, Write};
        use std::net::{Ipv4Addr, TcpListener};

        use serde_json::json;
        use tempfile::tempdir;
        use uuid::Uuid;

        use crate::background_jobs::{BackgroundJobOutcome, execute_embedding_batch};
        use crate::canonical_json::canonical_json;
        use crate::canonical_query::{
            CanonicalRoutingQueryArtifactV1, CanonicalRoutingQueryV1, CanonicalTaskV1,
        };
        use crate::config::RouterConfig;
        use crate::embedder::build_frozen_embedder_clients;
        use crate::embedding_identity::CANONICAL_ROUTING_QUERY_SCHEMA_V1;
        use crate::fingerprint::sha256_hex;
        use crate::ledger::read_pool::LedgerReadPool;
        use crate::ledger::repository::embedding::{EmbeddingJobCreate, EmbeddingJobCreateAck};
        use crate::ledger::repository::vector_catalog::ensure_canonical_query;
        use crate::ledger::repository::{ActivatedLedger, LedgerRepository};
        use crate::ledger::writer::LedgerWriterOwner;

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let address = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 4096];
            let mut expected = None;
            loop {
                let read = stream.read(&mut buffer).unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if expected.is_none()
                    && let Some(header_end) =
                        request.windows(4).position(|part| part == b"\r\n\r\n")
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
                    expected = Some(header_end + 4 + content_length);
                }
                if expected.is_some_and(|expected| request.len() >= expected) {
                    break;
                }
            }
            let body = br#"{"data":[{"embedding":[3.0,4.0],"index":0}]}"#;
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(body).unwrap();
            stream.flush().unwrap();
            request
        });

        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let config: RouterConfig = serde_json::from_value(json!({
            "version": 1,
            "mode": "shadow",
            "project_id": "background-embedding",
            "database_path": database_path,
            "embedders": [{
                "id": "embedding-main",
                "base_url": format!("http://{address}/v1"),
                "model": "embed-model",
                "provider_revision": "revision-1",
                "dimensions": 2,
                "timeout_ms": 1000,
                "max_in_flight": 1,
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
        .unwrap();
        let activation_at = Utc::now().timestamp_millis().saturating_sub(1_000);
        let mut activated = LedgerRepository::activate_at(&config, activation_at).unwrap();
        let project_uuid = activated.identity.project_uuid;
        let vector_space_id = activated.registry.spaces.keys().next().unwrap().clone();
        let query = CanonicalRoutingQueryV1 {
            schema: CANONICAL_ROUTING_QUERY_SCHEMA_V1.to_string(),
            instructions: Vec::new(),
            current_task: CanonicalTaskV1 {
                text: "embed this".to_string(),
            },
            bounded_context: Vec::new(),
            tool_schema_fingerprint: "0".repeat(64),
            response_schema_fingerprint: None,
            required_capabilities: Vec::new(),
            position_features: None,
        };
        let canonical_bytes = canonical_json(&serde_json::to_value(&query).unwrap())
            .unwrap()
            .into_bytes();
        let canonical_query_hash = sha256_hex(&canonical_bytes);
        let artifact = CanonicalRoutingQueryArtifactV1 {
            query,
            canonical_bytes,
            canonical_query_hash: canonical_query_hash.clone(),
        };
        let mut connection = rusqlite::Connection::open(&database_path).unwrap();
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
        let observed_at_unix_ms = Utc::now().timestamp_millis();
        let candidate = read_pool
            .select_embedding_work_until(
                project_uuid,
                activated.identity.config_generation_id.clone(),
                observed_at_unix_ms,
                None,
                1,
                Instant::now() + Duration::from_secs(2),
            )
            .await
            .unwrap()
            .pop()
            .expect("embedding job must be discoverable");
        let plan = prepare_embedding_batch(
            &clients,
            vec![candidate],
            Instant::now(),
            observed_at_unix_ms,
        )
        .unwrap();
        let (_cancel, cancellation) = tokio::sync::watch::channel(false);
        assert_eq!(
            execute_embedding_batch(&writer, plan, cancellation).await,
            BackgroundJobOutcome::Applied
        );

        let connection = rusqlite::Connection::open(&database_path).unwrap();
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM embeddings", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            1
        );
        let request = server.join().unwrap();
        assert!(String::from_utf8_lossy(&request).starts_with("POST /v1/embeddings HTTP/1.1"));
        read_pool.abort();
        writer_owner.abort();
    }
}
