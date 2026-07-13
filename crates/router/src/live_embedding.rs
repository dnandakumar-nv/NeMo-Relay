// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cache-first live embedding over the durable cross-process job protocol.

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::Utc;
#[cfg(test)]
use tokio::sync::Barrier;
use tokio::sync::{Notify, oneshot, watch};
use tokio::task::{AbortHandle, JoinHandle};
use uuid::Uuid;

use crate::background_jobs::{embedding_failure_resolution_kind, provider_timeout_fits_lease};
use crate::canonical_query::{CanonicalRoutingQueryArtifactV1, build_canonical_routing_query};
use crate::config::{CanonicalizerConfig, RouterConfig};
use crate::embedder::{
    EmbedderBatchItem, EmbedderFailureDisposition, EmbedderPermit, EmbedderWorkKind,
    FrozenEmbedderClients,
};
use crate::embedding_identity::canonicalizer_version;
use crate::ledger::read_pool::{LedgerReadPool, ReadPoolError};
use crate::ledger::repository::embedding::{
    EMBEDDING_JOB_LEASE_MILLIS, EmbeddingJobBatchClaim, EmbeddingJobBatchClaimAck,
    EmbeddingJobBatchClaimItem, EmbeddingJobBatchCompletion, EmbeddingJobBatchCompletionAck,
    EmbeddingJobBatchCompletionItem, EmbeddingJobLease, EmbeddingJobResolution,
    EmbeddingJobResolutionKind, EmbeddingJobSnapshot, LiveEmbeddingPrepare,
    LiveEmbeddingPrepareAck, embedding_space_is_degraded, load_verified_embedding_job,
    load_verified_embedding_lease,
};
use crate::ledger::repository::vector_catalog::load_embedding_cache;
use crate::ledger::repository::vector_index::{
    ActiveGenerationResolution, resolve_active_generation,
};
use crate::ledger::repository::vector_registry::{FrozenMappingKey, VectorRegistryEnsure};
use crate::ledger::writer::LedgerWriterClient;
use crate::projection::{RouterRequestProjectionV1, RouterRoutingContextProjectionV1};
use crate::provider_admission::ProviderAdmissionGate;
use crate::vector::{AuthoritativeVector, VectorSpaceId};

const PROVIDER_LEASE_MARGIN: Duration = Duration::from_secs(5);
const LIVE_POLL_INTERVAL: Duration = Duration::from_millis(10);

/// Stable live result. No provider, persistence, or index detail escapes this boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LiveEmbeddingResult {
    Ready(AuthoritativeVector),
    Unavailable,
}

#[derive(Clone)]
struct LivePoolAuthority {
    mapping: FrozenMappingKey,
    vector_space_id: VectorSpaceId,
    canonicalizer: CanonicalizerConfig,
    timeout: Duration,
}

/// Exact nonsecret query authority shared by embedding and recommendation.
///
/// This owner intentionally has no `Debug` or serde implementation because the
/// canonical artifact contains bounded application text.
#[derive(Clone)]
pub(crate) struct PreparedLiveQueryV1 {
    authority: LivePoolAuthority,
    artifact: CanonicalRoutingQueryArtifactV1,
}

impl PreparedLiveQueryV1 {
    #[cfg(test)]
    pub(crate) fn from_test_parts(
        mapping: FrozenMappingKey,
        vector_space_id: VectorSpaceId,
        artifact: CanonicalRoutingQueryArtifactV1,
        timeout: Duration,
    ) -> Arc<Self> {
        Arc::new(Self {
            authority: LivePoolAuthority {
                mapping,
                vector_space_id,
                canonicalizer: CanonicalizerConfig::default(),
                timeout,
            },
            artifact,
        })
    }

    pub(crate) fn mapping(&self) -> &FrozenMappingKey {
        &self.authority.mapping
    }

    pub(crate) fn vector_space_id(&self) -> &VectorSpaceId {
        &self.authority.vector_space_id
    }

    pub(crate) fn artifact(&self) -> &CanonicalRoutingQueryArtifactV1 {
        &self.artifact
    }

    pub(crate) const fn timeout(&self) -> Duration {
        self.authority.timeout
    }

    /// Consume the prepared owner after embedding and move its exact audit authority onward.
    pub(crate) fn into_audit_parts(
        self,
    ) -> (
        FrozenMappingKey,
        VectorSpaceId,
        CanonicalRoutingQueryArtifactV1,
    ) {
        (
            self.authority.mapping,
            self.authority.vector_space_id,
            self.artifact,
        )
    }
}

struct LiveControl {
    closed: AtomicBool,
    changed: watch::Sender<bool>,
    shutdown_deadline: Mutex<Option<Instant>>,
    next_task_id: AtomicU64,
    #[cfg(test)]
    lease_held_observations: AtomicU64,
    tasks: Mutex<HashMap<u64, AbortHandle>>,
    task_finished: Notify,
    #[cfg(test)]
    pause_after_admission: Mutex<Option<(Arc<Barrier>, Arc<Barrier>)>>,
}

struct LiveTaskGuard {
    control: Arc<LiveControl>,
    task_id: u64,
}

impl Drop for LiveTaskGuard {
    fn drop(&mut self) {
        let empty = {
            let mut tasks = self
                .control
                .tasks
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            tasks.remove(&self.task_id);
            tasks.is_empty()
        };
        if empty {
            self.control.task_finished.notify_waiters();
        }
    }
}

/// Crate-private service consumed by the later density/confidence stage.
#[derive(Clone)]
pub(crate) struct LiveEmbeddingService {
    pools: Arc<BTreeMap<String, LivePoolAuthority>>,
    writer: LedgerWriterClient,
    read_pool: LedgerReadPool,
    clients: Arc<FrozenEmbedderClients>,
    provider_gate: ProviderAdmissionGate,
    control: Arc<LiveControl>,
    wall_clock: Arc<dyn Fn() -> i64 + Send + Sync>,
}

impl LiveEmbeddingService {
    /// Return the frozen provider horizon for an already-selected exact pool.
    pub(crate) fn timeout_for_pool(&self, pool_id: &str) -> Option<Duration> {
        self.pools.get(pool_id).map(|authority| authority.timeout)
    }

    /// Freeze only current pool mappings and canonicalizers from one activation.
    pub(crate) fn new(
        config: &RouterConfig,
        registry: Arc<VectorRegistryEnsure>,
        writer: LedgerWriterClient,
        read_pool: LedgerReadPool,
        clients: Arc<FrozenEmbedderClients>,
        provider_gate: ProviderAdmissionGate,
    ) -> Result<Self, String> {
        Self::new_with_clock(
            config,
            registry,
            writer,
            read_pool,
            clients,
            provider_gate,
            Arc::new(|| Utc::now().timestamp_millis().max(0)),
        )
    }

    fn new_with_clock(
        config: &RouterConfig,
        registry: Arc<VectorRegistryEnsure>,
        writer: LedgerWriterClient,
        read_pool: LedgerReadPool,
        clients: Arc<FrozenEmbedderClients>,
        provider_gate: ProviderAdmissionGate,
        wall_clock: Arc<dyn Fn() -> i64 + Send + Sync>,
    ) -> Result<Self, String> {
        let config_generation_id = config
            .config_generation_id()
            .map_err(|_| "live embedding configuration is invalid".to_string())?;
        if registry.config_generation_id != config_generation_id {
            return Err("live embedding registry generation is stale".to_string());
        }
        let configured_pools = config
            .pools
            .iter()
            .map(|pool| (pool.id.as_str(), pool))
            .collect::<BTreeMap<_, _>>();
        let mut pools = BTreeMap::new();
        for (pool_id, mapping) in &registry.mappings {
            let pool = configured_pools
                .get(pool_id.as_str())
                .ok_or_else(|| "live embedding mapping pool is missing".to_string())?;
            if pool.learning.is_none() || mapping.pool_id != *pool_id {
                return Err("live embedding mapping is not currently enabled".to_string());
            }
            let canonicalizer = canonicalizer_version(&pool.canonicalizer)?;
            let space = registry
                .spaces
                .get(&mapping.vector_space_id)
                .ok_or_else(|| "live embedding mapping space is missing".to_string())?;
            let profile = registry
                .profiles
                .get(&mapping.embedder_profile_version_id)
                .ok_or_else(|| "live embedding mapping profile is missing".to_string())?;
            if mapping.project_uuid != registry.project_uuid
                || mapping.config_generation_id != registry.config_generation_id
                || mapping.canonicalizer_version_id != canonicalizer.canonicalizer_version_id
                || mapping.vector_space_id != space.vector_space_id
                || mapping.embedder_profile_version_id != space.embedder_profile_version_id
                || mapping.embedder_profile_version_id != profile.embedder_profile_version_id
            {
                return Err("live embedding mapping authority is inconsistent".to_string());
            }
            let key = FrozenMappingKey::new(
                mapping.project_uuid,
                mapping.config_generation_id.clone(),
                mapping.pool_id.clone(),
                mapping.policy_version_id.clone(),
            )?;
            pools.insert(
                pool_id.clone(),
                LivePoolAuthority {
                    mapping: key,
                    vector_space_id: mapping.vector_space_id.clone(),
                    canonicalizer: pool.canonicalizer.clone(),
                    timeout: Duration::from_millis(profile.timeout_ms),
                },
            );
        }
        let (changed, _) = watch::channel(false);
        Ok(Self {
            pools: Arc::new(pools),
            writer,
            read_pool,
            clients,
            provider_gate,
            control: Arc::new(LiveControl {
                closed: AtomicBool::new(false),
                changed,
                shutdown_deadline: Mutex::new(None),
                next_task_id: AtomicU64::new(1),
                #[cfg(test)]
                lease_held_observations: AtomicU64::new(0),
                tasks: Mutex::new(HashMap::new()),
                task_finished: Notify::new(),
                #[cfg(test)]
                pause_after_admission: Mutex::new(None),
            }),
            wall_clock,
        })
    }

    /// Close live admission at the task-registry linearization point.
    pub(crate) fn close_admission(&self) {
        {
            let _tasks = self
                .control
                .tasks
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            self.control.closed.store(true, Ordering::Release);
        }
    }

    /// Close live admission and cancel accepted calls.
    pub(crate) fn close(&self) {
        self.close_admission();
        self.provider_gate.close();
        self.control.changed.send_replace(true);
    }

    /// Install the shared shutdown deadline before requesting cleanup.
    pub(crate) fn close_until(&self, deadline: Instant) {
        let mut current = self
            .control
            .shutdown_deadline
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if current.is_none_or(|current| deadline < current) {
            *current = Some(deadline);
        }
        drop(current);
        self.close();
    }

    /// Synchronously fence starts and abort every accepted live task.
    pub(crate) fn abort(&self) {
        self.close();
        let tasks = self
            .control
            .tasks
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        for task in tasks.values() {
            task.abort();
        }
    }

    /// Wait for accepted live work to finish cleanup inside the shared shutdown deadline.
    pub(crate) async fn drain_until(&self, deadline: Instant) -> bool {
        loop {
            let finished = self.control.task_finished.notified();
            if self
                .control
                .tasks
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .is_empty()
            {
                return true;
            }
            tokio::select! {
                biased;
                () = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => {
                    return false;
                }
                () = finished => {}
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn lease_held_observation_count(&self) -> u64 {
        self.control.lease_held_observations.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(crate) fn is_closed(&self) -> bool {
        self.control.closed.load(Ordering::Acquire)
    }

    #[cfg(test)]
    pub(crate) fn pause_next_call_after_admission(
        &self,
        admitted: Arc<Barrier>,
        release: Arc<Barrier>,
    ) {
        *self
            .control
            .pause_after_admission
            .lock()
            .unwrap_or_else(|error| error.into_inner()) = Some((admitted, release));
    }

    /// Prepare one exact current query without persistence or provider work.
    pub(crate) fn prepare_query(
        &self,
        pool_id: &str,
        request: &RouterRequestProjectionV1,
        routing: &RouterRoutingContextProjectionV1,
    ) -> Result<Arc<PreparedLiveQueryV1>, crate::eligibility::IneligibilityReason> {
        let authority = self
            .pools
            .get(pool_id)
            .cloned()
            .ok_or(crate::eligibility::IneligibilityReason::RuntimeFailure)?;
        let artifact = build_canonical_routing_query(request, routing, &authority.canonicalizer)?;
        Ok(Arc::new(PreparedLiveQueryV1 {
            authority,
            artifact,
        }))
    }

    /// Build the current query and return only a persisted exact-space vector.
    pub(crate) async fn embed_until(
        &self,
        pool_id: &str,
        request: &RouterRequestProjectionV1,
        routing: &RouterRoutingContextProjectionV1,
        deadline: Instant,
    ) -> LiveEmbeddingResult {
        let prepared = match self.prepare_query(pool_id, request, routing) {
            Ok(prepared) => prepared,
            Err(_) => return LiveEmbeddingResult::Unavailable,
        };
        self.embed_prepared_until(prepared, deadline).await
    }

    /// Embed one service-prepared query without reconstructing its identity.
    pub(crate) async fn embed_prepared_until(
        &self,
        prepared: Arc<PreparedLiveQueryV1>,
        deadline: Instant,
    ) -> LiveEmbeddingResult {
        let service = self.clone();
        let Some(call) =
            self.spawn_tracked(async move { service.embed_owned_until(prepared, deadline).await })
        else {
            return LiveEmbeddingResult::Unavailable;
        };
        call.await.unwrap_or(LiveEmbeddingResult::Unavailable)
    }

    async fn embed_owned_until(
        &self,
        prepared: Arc<PreparedLiveQueryV1>,
        deadline: Instant,
    ) -> LiveEmbeddingResult {
        #[cfg(test)]
        let pause_after_admission = self
            .control
            .pause_after_admission
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
        #[cfg(test)]
        if let Some((admitted, release)) = pause_after_admission {
            admitted.wait().await;
            release.wait().await;
        }
        if self.is_closed_or_expired(deadline) {
            return LiveEmbeddingResult::Unavailable;
        }
        let authority = &prepared.authority;
        let artifact = &prepared.artifact;
        let canonical_query_hash = artifact.canonical_query_hash.clone();
        match self
            .observe(authority, &canonical_query_hash, None, deadline)
            .await
        {
            LiveObservation::Ready(vector) => return LiveEmbeddingResult::Ready(vector),
            LiveObservation::Unavailable => return LiveEmbeddingResult::Unavailable,
            LiveObservation::Pending => {}
        }

        let prepare = match LiveEmbeddingPrepare::new(
            authority.mapping.clone(),
            artifact.clone(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            (self.wall_clock)(),
        ) {
            Ok(prepare) => prepare,
            Err(_) => return LiveEmbeddingResult::Unavailable,
        };
        let job = match self
            .writer
            .prepare_live_embedding_until(prepare, deadline)
            .await
        {
            Ok(LiveEmbeddingPrepareAck::Ready(_)) => {
                return self
                    .ready_from_cache(authority, &canonical_query_hash, deadline)
                    .await;
            }
            Ok(LiveEmbeddingPrepareAck::Pending(job)) => job,
            Ok(
                LiveEmbeddingPrepareAck::MappingNotFound
                | LiveEmbeddingPrepareAck::MappingNotCurrent
                | LiveEmbeddingPrepareAck::ProviderSpaceDegraded
                | LiveEmbeddingPrepareAck::Conflict
                | LiveEmbeddingPrepareAck::OriginatingProcessNotLive
                | LiveEmbeddingPrepareAck::TransactionNotStarted,
            )
            | Err(_) => return LiveEmbeddingResult::Unavailable,
        };
        if job.vector_space_id != authority.vector_space_id.as_str()
            || job.canonical_query_hash != canonical_query_hash
            || job.content_hash != canonical_query_hash
        {
            return LiveEmbeddingResult::Unavailable;
        }

        let canonical_query = match String::from_utf8(artifact.canonical_bytes.clone()) {
            Ok(query) => query,
            Err(_) => return LiveEmbeddingResult::Unavailable,
        };
        let mut cancellation = self.control.changed.subscribe();
        loop {
            if self.is_closed_or_expired(deadline) {
                return LiveEmbeddingResult::Unavailable;
            }
            match self
                .observe(
                    authority,
                    &canonical_query_hash,
                    Some(&job.embedding_job_id),
                    deadline,
                )
                .await
            {
                LiveObservation::Ready(vector) => return LiveEmbeddingResult::Ready(vector),
                LiveObservation::Unavailable => return LiveEmbeddingResult::Unavailable,
                LiveObservation::Pending => {}
            }

            let permit = match self.clients.try_acquire(&authority.vector_space_id) {
                Ok(permit) => permit,
                Err(failure) if failure.disposition() == EmbedderFailureDisposition::Retryable => {
                    if !wait_for_progress(&mut cancellation, deadline).await {
                        return LiveEmbeddingResult::Unavailable;
                    }
                    continue;
                }
                Err(_) => return LiveEmbeddingResult::Unavailable,
            };
            match self
                .try_winning_call(
                    authority,
                    &job,
                    &canonical_query_hash,
                    &canonical_query,
                    permit,
                    deadline,
                    &mut cancellation,
                )
                .await
            {
                LiveAttempt::Ready(vector) => return LiveEmbeddingResult::Ready(vector),
                LiveAttempt::Lost => {
                    return self
                        .wait_for_persisted_winner(
                            authority,
                            &canonical_query_hash,
                            &job.embedding_job_id,
                            deadline,
                            &mut cancellation,
                        )
                        .await;
                }
                LiveAttempt::RetryObservation => {
                    if !wait_for_progress(&mut cancellation, deadline).await {
                        return LiveEmbeddingResult::Unavailable;
                    }
                }
                LiveAttempt::Unavailable => return LiveEmbeddingResult::Unavailable,
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_winning_call(
        &self,
        authority: &LivePoolAuthority,
        job: &EmbeddingJobSnapshot,
        canonical_query_hash: &str,
        canonical_query: &str,
        permit: EmbedderPermit,
        deadline: Instant,
        cancellation: &mut watch::Receiver<bool>,
    ) -> LiveAttempt {
        if self.is_closed_or_expired(deadline) || *cancellation.borrow() {
            return LiveAttempt::Unavailable;
        }
        let service = self.clone();
        let authority = authority.clone();
        let job = job.clone();
        let canonical_query_hash = canonical_query_hash.to_string();
        let canonical_query = canonical_query.to_string();
        let Some(mut attempt) = self.spawn_tracked(async move {
            service
                .run_winning_call(
                    authority,
                    job,
                    canonical_query_hash,
                    canonical_query,
                    permit,
                    deadline,
                )
                .await
        }) else {
            return LiveAttempt::Unavailable;
        };

        tokio::select! {
            biased;
            changed = cancellation.changed() => {
                let _ = changed;
                LiveAttempt::Unavailable
            }
            () = tokio::time::sleep_until(tokio::time::Instant::from_std(deadline)) => {
                LiveAttempt::Unavailable
            }
            joined = &mut attempt => joined.unwrap_or(LiveAttempt::Unavailable),
        }
    }

    async fn run_winning_call(
        &self,
        authority: LivePoolAuthority,
        job: EmbeddingJobSnapshot,
        canonical_query_hash: String,
        canonical_query: String,
        permit: EmbedderPermit,
        deadline: Instant,
    ) -> LiveAttempt {
        let operation_origin = Instant::now();
        let timeout = permit.operation_timeout();
        if !provider_timeout_fits_lease(
            timeout,
            Duration::from_millis(EMBEDDING_JOB_LEASE_MILLIS as u64),
        ) {
            return LiveAttempt::Unavailable;
        }
        let lease_provider_limit = match operation_origin.checked_add(
            Duration::from_millis(EMBEDDING_JOB_LEASE_MILLIS as u64)
                .saturating_sub(PROVIDER_LEASE_MARGIN),
        ) {
            Some(limit) => limit,
            None => return LiveAttempt::Unavailable,
        };
        let lease_deadline = match operation_origin
            .checked_add(Duration::from_millis(EMBEDDING_JOB_LEASE_MILLIS as u64))
        {
            Some(limit) => limit,
            None => return LiveAttempt::Unavailable,
        };
        let timeout_deadline = match operation_origin.checked_add(timeout) {
            Some(limit) => limit,
            None => return LiveAttempt::Unavailable,
        };
        let provider_deadline = deadline.min(timeout_deadline).min(lease_provider_limit);
        if Instant::now() >= provider_deadline {
            return LiveAttempt::Unavailable;
        }
        if permit.vector_space_id() != &authority.vector_space_id {
            return LiveAttempt::Unavailable;
        }
        if job.vector_space_id != authority.vector_space_id.as_str()
            || job.canonical_query_hash != canonical_query_hash
            || job.content_hash != canonical_query_hash
        {
            return LiveAttempt::Unavailable;
        }
        let item = match EmbedderBatchItem::from_verified_artifact(
            canonical_query_hash.clone(),
            canonical_query,
            job.embedding_job_id.clone(),
        ) {
            Ok(item) => item,
            Err(_) => return LiveAttempt::Unavailable,
        };
        let observed_at_unix_ms = (self.wall_clock)();
        let claim = match EmbeddingJobBatchClaim::new(
            Uuid::now_v7(),
            authority.vector_space_id.clone(),
            Uuid::now_v7(),
            observed_at_unix_ms,
            vec![match EmbeddingJobBatchClaimItem::new(
                job.embedding_job_id.clone(),
                Uuid::now_v7(),
                Uuid::now_v7(),
            ) {
                Ok(item) => item,
                Err(_) => return LiveAttempt::Unavailable,
            }],
        ) {
            Ok(claim) => claim,
            Err(_) => return LiveAttempt::Unavailable,
        };
        let leases = match self
            .writer
            .claim_embedding_job_batch_until(claim.clone(), provider_deadline)
            .await
        {
            Ok(EmbeddingJobBatchClaimAck::Claimed(leases))
            | Ok(EmbeddingJobBatchClaimAck::AlreadyApplied(leases)) => leases,
            Ok(EmbeddingJobBatchClaimAck::NotEligible { .. }) => {
                return LiveAttempt::RetryObservation;
            }
            Ok(EmbeddingJobBatchClaimAck::LeaseHeld { .. }) => {
                #[cfg(test)]
                self.control
                    .lease_held_observations
                    .fetch_add(1, Ordering::AcqRel);
                return LiveAttempt::Lost;
            }
            Ok(
                EmbeddingJobBatchClaimAck::JobNotFound { .. }
                | EmbeddingJobBatchClaimAck::Terminal { .. },
            ) => return LiveAttempt::Lost,
            Ok(
                EmbeddingJobBatchClaimAck::VectorSpaceNotFound
                | EmbeddingJobBatchClaimAck::VectorSpaceUnauthorized
                | EmbeddingJobBatchClaimAck::BatchTooLarge { .. }
                | EmbeddingJobBatchClaimAck::Conflict
                | EmbeddingJobBatchClaimAck::OriginatingProcessNotLive
                | EmbeddingJobBatchClaimAck::TransactionNotStarted,
            )
            | Err(_) => {
                self.release_unconfirmed_claim(&authority, &job, claim.lease_token, lease_deadline)
                    .await;
                return LiveAttempt::Unavailable;
            }
        };
        let Some(lease) = leases.as_slice().first() else {
            self.release_unconfirmed_claim(&authority, &job, claim.lease_token, lease_deadline)
                .await;
            return LiveAttempt::Unavailable;
        };
        if leases.len() != 1
            || lease.job.embedding_job_id != job.embedding_job_id
            || lease.job.vector_space_id != authority.vector_space_id.as_str()
            || lease.job.canonical_query_hash != canonical_query_hash
            || lease.job.content_hash != canonical_query_hash
            || lease.lease_token != claim.lease_token
            || lease.lease_expires_at_unix_ms != claim.lease_expires_at_unix_ms
        {
            self.release_unconfirmed_claim(&authority, &job, claim.lease_token, lease_deadline)
                .await;
            return LiveAttempt::Unavailable;
        }
        let lease = lease.clone();
        let mut cancellation = self.control.changed.subscribe();
        if self.is_closed_or_expired(provider_deadline) || *cancellation.borrow() {
            self.release_lease(&lease, lease_deadline).await;
            return LiveAttempt::Unavailable;
        }
        let provider_service = self.clone();
        let execution = match self.provider_gate.start_owned(|| {
            provider_service.spawn_tracked(permit.execute_batch_until(
                EmbedderWorkKind::EmbeddingJob,
                vec![item],
                provider_deadline,
                cancellation.clone(),
            ))
        }) {
            Ok(Some(execution)) => execution,
            Ok(None) | Err(_) => {
                self.release_lease(&lease, lease_deadline).await;
                return LiveAttempt::Unavailable;
            }
        };
        let vectors = match execution.await {
            Ok(Ok(vectors)) => vectors,
            Ok(Err(failure)) => {
                self.resolve_failure(
                    &lease,
                    failure.disposition(),
                    failure.class().stable_class(),
                    lease_deadline,
                )
                .await;
                return LiveAttempt::Unavailable;
            }
            Err(_) => {
                self.release_lease(&lease, lease_deadline).await;
                return LiveAttempt::Unavailable;
            }
        };
        let [vector] = vectors.as_slice() else {
            self.release_lease(&lease, lease_deadline).await;
            return LiveAttempt::Unavailable;
        };
        if self.is_closed_or_expired(deadline) || *cancellation.borrow_and_update() {
            self.release_lease(&lease, lease_deadline).await;
            return LiveAttempt::Unavailable;
        }
        let completion_item = match EmbeddingJobBatchCompletionItem::new(
            job.embedding_job_id.clone(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            lease.job.attempt_generation,
            lease.job.content_hash.clone(),
            vector.clone(),
        ) {
            Ok(item) => item,
            Err(_) => {
                self.release_lease(&lease, lease_deadline).await;
                return LiveAttempt::Unavailable;
            }
        };
        let completion = match EmbeddingJobBatchCompletion::new(
            Uuid::now_v7(),
            authority.vector_space_id.clone(),
            lease.lease_token,
            (self.wall_clock)(),
            vec![completion_item],
        ) {
            Ok(completion) => completion,
            Err(_) => {
                self.release_lease(&lease, lease_deadline).await;
                return LiveAttempt::Unavailable;
            }
        };
        let result = match self
            .writer
            .complete_embedding_job_batch_until(completion, lease_deadline)
            .await
        {
            Ok(EmbeddingJobBatchCompletionAck::Applied(results))
            | Ok(EmbeddingJobBatchCompletionAck::AlreadyApplied(results))
                if results.len() == 1
                    && results[0].job.embedding_job_id == job.embedding_job_id
                    && results[0].embedding.vector_space_id == authority.vector_space_id
                    && results[0].embedding.canonical_query_hash == canonical_query_hash =>
            {
                match self
                    .observe(
                        &authority,
                        &canonical_query_hash,
                        Some(&job.embedding_job_id),
                        deadline,
                    )
                    .await
                {
                    LiveObservation::Ready(vector) => LiveAttempt::Ready(vector),
                    LiveObservation::Pending | LiveObservation::Unavailable => {
                        LiveAttempt::Unavailable
                    }
                }
            }
            Ok(EmbeddingJobBatchCompletionAck::StaleLease { .. }) => LiveAttempt::Lost,
            Ok(
                EmbeddingJobBatchCompletionAck::Applied(_)
                | EmbeddingJobBatchCompletionAck::AlreadyApplied(_),
            ) => LiveAttempt::Unavailable,
            Ok(
                EmbeddingJobBatchCompletionAck::VectorSpaceNotFound
                | EmbeddingJobBatchCompletionAck::BatchTooLarge { .. }
                | EmbeddingJobBatchCompletionAck::JobNotFound { .. }
                | EmbeddingJobBatchCompletionAck::Conflict
                | EmbeddingJobBatchCompletionAck::OriginatingProcessNotLive
                | EmbeddingJobBatchCompletionAck::TransactionNotStarted,
            )
            | Err(_) => LiveAttempt::Unavailable,
        };
        if matches!(result, LiveAttempt::Unavailable) {
            self.release_lease(&lease, lease_deadline).await;
        }
        result
    }

    async fn ready_from_cache(
        &self,
        authority: &LivePoolAuthority,
        canonical_query_hash: &str,
        deadline: Instant,
    ) -> LiveEmbeddingResult {
        match self
            .observe(authority, canonical_query_hash, None, deadline)
            .await
        {
            LiveObservation::Ready(vector) => LiveEmbeddingResult::Ready(vector),
            LiveObservation::Pending | LiveObservation::Unavailable => {
                LiveEmbeddingResult::Unavailable
            }
        }
    }

    async fn wait_for_persisted_winner(
        &self,
        authority: &LivePoolAuthority,
        canonical_query_hash: &str,
        embedding_job_id: &str,
        deadline: Instant,
        cancellation: &mut watch::Receiver<bool>,
    ) -> LiveEmbeddingResult {
        loop {
            match self
                .observe(
                    authority,
                    canonical_query_hash,
                    Some(embedding_job_id),
                    deadline,
                )
                .await
            {
                LiveObservation::Ready(vector) => return LiveEmbeddingResult::Ready(vector),
                LiveObservation::Unavailable => return LiveEmbeddingResult::Unavailable,
                LiveObservation::Pending => {}
            }
            if !wait_for_progress(cancellation, deadline).await {
                return LiveEmbeddingResult::Unavailable;
            }
        }
    }

    async fn observe(
        &self,
        authority: &LivePoolAuthority,
        canonical_query_hash: &str,
        embedding_job_id: Option<&str>,
        deadline: Instant,
    ) -> LiveObservation {
        if self.is_closed_or_expired(deadline) {
            return LiveObservation::Unavailable;
        }
        let project_uuid = authority.mapping.project_uuid;
        let vector_space_id = authority.vector_space_id.clone();
        let canonical_query_hash = canonical_query_hash.to_string();
        let embedding_job_id = embedding_job_id.map(str::to_string);
        let observation = self
            .read_pool
            .run(deadline, move |connection| {
                let transaction = connection
                    .unchecked_transaction()
                    .map_err(|_| ReadPoolError::operation_failed())?;
                let cache = load_embedding_cache(
                    &transaction,
                    project_uuid,
                    &vector_space_id,
                    &canonical_query_hash,
                )
                .map_err(|_| ReadPoolError::operation_failed())?;
                let healthy = matches!(
                    resolve_active_generation(&transaction, &vector_space_id)
                        .map_err(|_| ReadPoolError::operation_failed())?,
                    ActiveGenerationResolution::Active(_)
                );
                let observation = if let Some(cache) = cache {
                    if healthy {
                        LiveObservation::Ready(cache.vector)
                    } else {
                        LiveObservation::Unavailable
                    }
                } else if !healthy
                    || embedding_space_is_degraded(&transaction, project_uuid, &vector_space_id)
                        .map_err(|_| ReadPoolError::operation_failed())?
                {
                    LiveObservation::Unavailable
                } else if let Some(embedding_job_id) = embedding_job_id {
                    match load_verified_embedding_job(&transaction, project_uuid, &embedding_job_id)
                        .map_err(|_| ReadPoolError::operation_failed())?
                    {
                        Some(job)
                            if job.vector_space_id == vector_space_id.as_str()
                                && job.canonical_query_hash == canonical_query_hash
                                && job.content_hash == canonical_query_hash
                                && job.terminal_error_class.is_none()
                                && job.attempt_count < 5 =>
                        {
                            LiveObservation::Pending
                        }
                        _ => LiveObservation::Unavailable,
                    }
                } else {
                    LiveObservation::Pending
                };
                transaction
                    .commit()
                    .map_err(|_| ReadPoolError::operation_failed())?;
                Ok(observation)
            })
            .await
            .unwrap_or(LiveObservation::Unavailable);
        if self.is_closed_or_expired(deadline) {
            LiveObservation::Unavailable
        } else {
            observation
        }
    }

    async fn release_lease(&self, lease: &EmbeddingJobLease, deadline: Instant) {
        self.resolve_lease(lease, EmbeddingJobResolutionKind::Released, deadline)
            .await;
    }

    async fn resolve_failure(
        &self,
        lease: &EmbeddingJobLease,
        disposition: EmbedderFailureDisposition,
        stable_error_class: &str,
        deadline: Instant,
    ) {
        let resolved_at_unix_ms = (self.wall_clock)();
        let kind = embedding_failure_resolution_kind(
            disposition,
            stable_error_class,
            lease.job.attempt_count,
            resolved_at_unix_ms,
        );
        self.resolve_lease(lease, kind, deadline).await;
    }

    async fn release_unconfirmed_claim(
        &self,
        authority: &LivePoolAuthority,
        job: &EmbeddingJobSnapshot,
        lease_token: Uuid,
        deadline: Instant,
    ) {
        if Instant::now() >= deadline {
            return;
        }
        let project_uuid = authority.mapping.project_uuid;
        let embedding_job_id = job.embedding_job_id.clone();
        let lease = self
            .read_pool
            .run(deadline, move |connection| {
                load_verified_embedding_lease(
                    connection,
                    project_uuid,
                    &embedding_job_id,
                    lease_token,
                )
                .map_err(|_| ReadPoolError::operation_failed())
            })
            .await
            .ok()
            .flatten();
        if let Some(lease) = lease {
            self.release_lease(&lease, deadline).await;
        }
    }

    async fn resolve_lease(
        &self,
        lease: &EmbeddingJobLease,
        kind: EmbeddingJobResolutionKind,
        deadline: Instant,
    ) {
        self.resolve_job(
            lease.job.embedding_job_id.clone(),
            lease.lease_token,
            lease.job.attempt_generation,
            lease.job.content_hash.clone(),
            kind,
            deadline,
        )
        .await;
    }

    async fn resolve_job(
        &self,
        embedding_job_id: String,
        lease_token: Uuid,
        attempt_generation: i64,
        content_hash: String,
        kind: EmbeddingJobResolutionKind,
        deadline: Instant,
    ) {
        let deadline = self.bounded_shutdown_deadline(deadline);
        if Instant::now() >= deadline {
            return;
        }
        let command = EmbeddingJobResolution::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            embedding_job_id,
            lease_token,
            attempt_generation,
            content_hash,
            (self.wall_clock)(),
            kind,
        );
        if let Ok(command) = command {
            let _ = self
                .writer
                .resolve_embedding_job_until(command, deadline)
                .await;
        }
    }

    fn is_closed_or_expired(&self, deadline: Instant) -> bool {
        self.control.closed.load(Ordering::Acquire) || Instant::now() >= deadline
    }

    fn bounded_shutdown_deadline(&self, default: Instant) -> Instant {
        self.control
            .shutdown_deadline
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .map(|deadline| deadline.min(default))
            .unwrap_or(default)
    }

    fn spawn_tracked<T>(
        &self,
        future: impl Future<Output = T> + Send + 'static,
    ) -> Option<JoinHandle<T>>
    where
        T: Send + 'static,
    {
        let mut tasks = self
            .control
            .tasks
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if self.control.closed.load(Ordering::Acquire) {
            return None;
        }
        let task_id = self.control.next_task_id.fetch_add(1, Ordering::AcqRel);
        let control = self.control.clone();
        let (start, started) = oneshot::channel();
        let task = tokio::spawn(async move {
            started
                .await
                .expect("live task registry must publish before task start");
            let _guard = LiveTaskGuard { control, task_id };
            future.await
        });
        tasks.insert(task_id, task.abort_handle());
        drop(tasks);
        if start.send(()).is_err() {
            let empty = {
                let mut tasks = self
                    .control
                    .tasks
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                tasks.remove(&task_id);
                tasks.is_empty()
            };
            if empty {
                self.control.task_finished.notify_waiters();
            }
        }
        Some(task)
    }
}

impl std::fmt::Debug for LiveEmbeddingService {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("LiveEmbeddingService")
            .field("pool_count", &self.pools.len())
            .field("closed", &self.control.closed.load(Ordering::Acquire))
            .finish_non_exhaustive()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LiveObservation {
    Ready(AuthoritativeVector),
    Pending,
    Unavailable,
}

enum LiveAttempt {
    Ready(AuthoritativeVector),
    Lost,
    RetryObservation,
    Unavailable,
}

async fn wait_for_progress(cancellation: &mut watch::Receiver<bool>, deadline: Instant) -> bool {
    if *cancellation.borrow_and_update() || Instant::now() >= deadline {
        return false;
    }
    let wake = Instant::now()
        .checked_add(LIVE_POLL_INTERVAL)
        .unwrap_or(deadline)
        .min(deadline);
    tokio::select! {
        biased;
        changed = cancellation.changed() => changed.is_ok() && !*cancellation.borrow_and_update(),
        () = tokio::time::sleep_until(tokio::time::Instant::from_std(wake)) => {
            Instant::now() < deadline
        }
    }
}
