// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fenced operations executed by the Router background supervisor.

use std::time::{Duration, Instant};

use chrono::Utc;
use tokio::sync::watch;
use uuid::Uuid;

use crate::background::BackgroundCancellation;
use crate::embedder::{
    EmbedderBatchItem, EmbedderFailureDisposition, EmbedderPermit, EmbedderWorkKind,
    FrozenEmbedderClients,
};
use crate::ledger::repository::background_work::{
    BackfillWorkCandidate, EmbeddingWorkCandidate, FailurePropagationCandidate,
    MaterializationWorkCandidate,
};
use crate::ledger::repository::embedding::{
    EmbeddingJobBatchClaim, EmbeddingJobBatchClaimAck, EmbeddingJobBatchClaimItem,
    EmbeddingJobBatchCompletion, EmbeddingJobBatchCompletionAck, EmbeddingJobBatchCompletionItem,
    EmbeddingJobLease, EmbeddingJobResolution, EmbeddingJobResolutionAck,
    EmbeddingJobResolutionKind,
};
use crate::ledger::repository::materialization::{
    MaterializationClaim, MaterializationClaimAck, MaterializationCompletion,
    MaterializationCompletionAck, MaterializationFailurePropagation,
    MaterializationFailurePropagationAck, MaterializationFailurePropagationItem,
    MaterializationLease, MaterializationResolution, MaterializationResolutionAck,
    MaterializationResolutionKind, VectorBackfillAck, VectorBackfillCommand,
};
use crate::ledger::repository::vector_index::{
    GenerationAuthorizationAck, GenerationObjectCreationAck, GenerationObjectsStatus,
    RebuildFlipAck, RebuildLeaseClaimAck, RebuildLeaseFence, RebuildLeaseMutationAck,
    RebuildStepAck, RetiredGenerationCleanupAck,
};
use crate::ledger::repository::vector_work::{
    BuildingGenerationWork, RebuildLeaseDisposition, RetiredGenerationWork, VectorSpaceInspection,
    VectorSpaceRecoveryReason,
};
use crate::ledger::writer::LedgerWriterClient;
use crate::sqlite_vector_store::{VectorIndexWriterAck, VectorIndexWriterCommand};
use crate::vector::{AuthoritativeVector, VectorSpaceId};

const PROVIDER_LEASE_MARGIN: Duration = Duration::from_secs(5);
const WRITER_OPERATION_TIMEOUT: Duration = Duration::from_secs(5);
const RETRY_BASE: Duration = Duration::from_secs(1);
const RETRY_MAX: Duration = Duration::from_secs(5 * 60);

/// Stable result of one bounded, advisory background operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BackgroundJobOutcome {
    Applied,
    AlreadyApplied,
    Deferred,
    Stale,
    PermanentFailure,
}

/// Fully assembled provider batch. Creating this value acquires the profile permit.
pub(crate) struct PreparedEmbeddingBatch {
    permit: EmbedderPermit,
    items: Vec<EmbedderBatchItem>,
    claim: EmbeddingJobBatchClaim,
    expected_leases: Vec<ExpectedEmbeddingLease>,
    provider_deadline: Instant,
    completion_deadline: Instant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpectedEmbeddingLease {
    pub(crate) embedding_job_id: String,
    pub(crate) canonical_query_hash: String,
    pub(crate) content_hash: String,
}

pub(crate) fn embedding_candidates_match_batch(
    candidates: &[EmbeddingWorkCandidate],
    vector_space_id: &VectorSpaceId,
    max_batch_size: usize,
) -> bool {
    let Some(first) = candidates.first() else {
        return false;
    };
    if candidates.len() > max_batch_size || first.batch_size != max_batch_size {
        return false;
    }
    candidates.iter().all(|candidate| {
        candidate.job.vector_space_id == vector_space_id.as_str()
            && candidate.embedder_profile_version_id == first.embedder_profile_version_id
            && candidate.batch_size == first.batch_size
            && candidate.job.canonical_query_hash
                == candidate.canonical_query.artifact.canonical_query_hash
    }) && candidates
        .windows(2)
        .all(|pair| pair[0].job.embedding_job_id.as_str() < pair[1].job.embedding_job_id.as_str())
}

/// Assemble one exact same-space batch and acquire its provider permit before claim.
pub(crate) fn prepare_embedding_batch(
    clients: &FrozenEmbedderClients,
    candidates: Vec<EmbeddingWorkCandidate>,
    operation_origin: Instant,
    observed_at_unix_ms: i64,
) -> Result<PreparedEmbeddingBatch, BackgroundJobOutcome> {
    let first = candidates
        .first()
        .ok_or(BackgroundJobOutcome::PermanentFailure)?;
    let vector_space_id = VectorSpaceId::new(first.job.vector_space_id.clone())
        .map_err(|_| BackgroundJobOutcome::PermanentFailure)?;
    let permit =
        clients
            .try_acquire(&vector_space_id)
            .map_err(|failure| match failure.disposition() {
                EmbedderFailureDisposition::Retryable => BackgroundJobOutcome::Deferred,
                EmbedderFailureDisposition::Quarantine => BackgroundJobOutcome::PermanentFailure,
            })?;
    if !embedding_candidates_match_batch(&candidates, &vector_space_id, permit.max_batch_size()) {
        return Err(BackgroundJobOutcome::PermanentFailure);
    }

    let mut items = Vec::with_capacity(candidates.len());
    let mut claim_items = Vec::with_capacity(candidates.len());
    let mut expected_leases = Vec::with_capacity(candidates.len());
    for candidate in candidates {
        let canonical_query = String::from_utf8(candidate.canonical_query.artifact.canonical_bytes)
            .map_err(|_| BackgroundJobOutcome::PermanentFailure)?;
        items.push(
            EmbedderBatchItem::from_verified_artifact(
                candidate.job.canonical_query_hash.clone(),
                canonical_query,
                candidate.job.embedding_job_id.clone(),
            )
            .map_err(|_| BackgroundJobOutcome::PermanentFailure)?,
        );
        claim_items.push(
            EmbeddingJobBatchClaimItem::new(
                candidate.job.embedding_job_id.clone(),
                Uuid::now_v7(),
                Uuid::now_v7(),
            )
            .map_err(|_| BackgroundJobOutcome::PermanentFailure)?,
        );
        expected_leases.push(ExpectedEmbeddingLease {
            embedding_job_id: candidate.job.embedding_job_id,
            canonical_query_hash: candidate.job.canonical_query_hash,
            content_hash: candidate.job.content_hash,
        });
    }
    let claim = EmbeddingJobBatchClaim::new(
        Uuid::now_v7(),
        vector_space_id,
        Uuid::now_v7(),
        observed_at_unix_ms,
        claim_items,
    )
    .map_err(|_| BackgroundJobOutcome::PermanentFailure)?;
    let timeout = permit.operation_timeout();
    let lease_duration_ms = claim
        .lease_expires_at_unix_ms
        .checked_sub(claim.observed_at_unix_ms)
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(BackgroundJobOutcome::PermanentFailure)?;
    let lease_duration = Duration::from_millis(lease_duration_ms);
    if !provider_timeout_fits_lease(timeout, lease_duration) {
        return Err(BackgroundJobOutcome::PermanentFailure);
    }
    let provider_deadline = operation_origin
        .checked_add(timeout)
        .ok_or(BackgroundJobOutcome::PermanentFailure)?;
    let completion_deadline = operation_origin
        .checked_add(lease_duration)
        .ok_or(BackgroundJobOutcome::PermanentFailure)?;
    Ok(PreparedEmbeddingBatch {
        permit,
        items,
        claim,
        expected_leases,
        provider_deadline,
        completion_deadline,
    })
}

pub(crate) fn provider_timeout_fits_lease(timeout: Duration, lease_duration: Duration) -> bool {
    timeout
        .checked_add(PROVIDER_LEASE_MARGIN)
        .is_some_and(|required| required <= lease_duration)
}

/// Claim last, execute outside SQLite, then complete or resolve every batch member.
pub(crate) async fn execute_embedding_batch(
    writer: &LedgerWriterClient,
    plan: PreparedEmbeddingBatch,
    cancellation: watch::Receiver<bool>,
) -> BackgroundJobOutcome {
    execute_embedding_batch_inner(writer, plan, cancellation, None).await
}

pub(crate) async fn execute_embedding_batch_with_cancellation(
    writer: &LedgerWriterClient,
    plan: PreparedEmbeddingBatch,
    cancellation: BackgroundCancellation,
) -> BackgroundJobOutcome {
    execute_embedding_batch_inner(writer, plan, cancellation.subscribe(), Some(cancellation)).await
}

async fn execute_embedding_batch_inner(
    writer: &LedgerWriterClient,
    plan: PreparedEmbeddingBatch,
    cancellation: watch::Receiver<bool>,
    shutdown: Option<BackgroundCancellation>,
) -> BackgroundJobOutcome {
    if *cancellation.borrow() || Instant::now() >= plan.provider_deadline {
        return BackgroundJobOutcome::Deferred;
    }
    let claim_deadline = bounded_shutdown_deadline(plan.provider_deadline, shutdown.as_ref());
    if Instant::now() >= claim_deadline {
        return BackgroundJobOutcome::Deferred;
    }
    let acknowledgement = match writer
        .claim_embedding_job_batch_until(plan.claim.clone(), claim_deadline)
        .await
    {
        Ok(acknowledgement) => acknowledgement,
        Err(_) => return BackgroundJobOutcome::Deferred,
    };
    let leases = match acknowledgement {
        EmbeddingJobBatchClaimAck::Claimed(leases) => leases,
        EmbeddingJobBatchClaimAck::AlreadyApplied(leases) => leases,
        EmbeddingJobBatchClaimAck::NotEligible { .. }
        | EmbeddingJobBatchClaimAck::LeaseHeld { .. }
        | EmbeddingJobBatchClaimAck::Terminal { .. }
        | EmbeddingJobBatchClaimAck::VectorSpaceUnauthorized
        | EmbeddingJobBatchClaimAck::TransactionNotStarted => {
            return BackgroundJobOutcome::Deferred;
        }
        EmbeddingJobBatchClaimAck::JobNotFound { .. } => return BackgroundJobOutcome::Stale,
        EmbeddingJobBatchClaimAck::VectorSpaceNotFound
        | EmbeddingJobBatchClaimAck::BatchTooLarge { .. }
        | EmbeddingJobBatchClaimAck::Conflict
        | EmbeddingJobBatchClaimAck::OriginatingProcessNotLive => {
            return BackgroundJobOutcome::PermanentFailure;
        }
    };
    if !leases_match_plan(&leases, &plan) {
        release_embedding_leases(
            writer,
            &leases,
            bounded_shutdown_deadline(plan.completion_deadline, shutdown.as_ref()),
        )
        .await;
        return BackgroundJobOutcome::PermanentFailure;
    }
    if *cancellation.borrow() || Instant::now() >= plan.provider_deadline {
        release_embedding_leases(
            writer,
            &leases,
            bounded_shutdown_deadline(plan.completion_deadline, shutdown.as_ref()),
        )
        .await;
        return BackgroundJobOutcome::Deferred;
    }

    let result = plan
        .permit
        .execute_batch_until(
            EmbedderWorkKind::EmbeddingJob,
            plan.items,
            plan.provider_deadline,
            cancellation.clone(),
        )
        .await;
    match result {
        Ok(vectors) => {
            if *cancellation.borrow()
                || vectors.len() != leases.len()
                || Instant::now() >= plan.completion_deadline
            {
                release_embedding_leases(
                    writer,
                    &leases,
                    bounded_shutdown_deadline(plan.completion_deadline, shutdown.as_ref()),
                )
                .await;
                return BackgroundJobOutcome::Deferred;
            }
            let completed_at_unix_ms = now_unix_ms();
            let mut completion_items = Vec::with_capacity(leases.len());
            for (lease, vector) in leases.iter().zip(vectors) {
                let item = match prepare_embedding_completion_item(lease, vector) {
                    Ok(item) => item,
                    Err(_) => {
                        release_embedding_leases(
                            writer,
                            &leases,
                            bounded_shutdown_deadline(plan.completion_deadline, shutdown.as_ref()),
                        )
                        .await;
                        return BackgroundJobOutcome::PermanentFailure;
                    }
                };
                completion_items.push(item);
            }
            let completion = match EmbeddingJobBatchCompletion::new(
                Uuid::now_v7(),
                plan.claim.vector_space_id,
                plan.claim.lease_token,
                completed_at_unix_ms,
                completion_items,
            ) {
                Ok(completion) => completion,
                Err(_) => {
                    release_embedding_leases(
                        writer,
                        &leases,
                        bounded_shutdown_deadline(plan.completion_deadline, shutdown.as_ref()),
                    )
                    .await;
                    return BackgroundJobOutcome::PermanentFailure;
                }
            };
            let completion_deadline =
                bounded_shutdown_deadline(plan.completion_deadline, shutdown.as_ref());
            match writer
                .complete_embedding_job_batch_until(completion, completion_deadline)
                .await
            {
                Ok(EmbeddingJobBatchCompletionAck::Applied(_)) => BackgroundJobOutcome::Applied,
                Ok(EmbeddingJobBatchCompletionAck::AlreadyApplied(_)) => {
                    BackgroundJobOutcome::AlreadyApplied
                }
                Ok(EmbeddingJobBatchCompletionAck::StaleLease { .. })
                | Ok(EmbeddingJobBatchCompletionAck::JobNotFound { .. }) => {
                    BackgroundJobOutcome::Stale
                }
                Ok(EmbeddingJobBatchCompletionAck::TransactionNotStarted) | Err(_) => {
                    BackgroundJobOutcome::Deferred
                }
                Ok(
                    EmbeddingJobBatchCompletionAck::VectorSpaceNotFound
                    | EmbeddingJobBatchCompletionAck::BatchTooLarge { .. }
                    | EmbeddingJobBatchCompletionAck::Conflict
                    | EmbeddingJobBatchCompletionAck::OriginatingProcessNotLive,
                ) => BackgroundJobOutcome::PermanentFailure,
            }
        }
        Err(failure) => {
            let disposition = failure.disposition();
            let stable_error_class = failure.class().stable_class().to_string();
            let resolved_at_unix_ms = now_unix_ms();
            let mut permanent_failure = false;
            for lease in &leases {
                let kind = embedding_failure_resolution_kind(
                    disposition,
                    &stable_error_class,
                    lease.job.attempt_count,
                    resolved_at_unix_ms,
                );
                permanent_failure |= resolve_embedding_lease(
                    writer,
                    lease,
                    kind,
                    resolved_at_unix_ms,
                    bounded_shutdown_deadline(plan.completion_deadline, shutdown.as_ref()),
                )
                .await
                    == BackgroundJobOutcome::PermanentFailure;
            }
            if permanent_failure {
                BackgroundJobOutcome::PermanentFailure
            } else {
                BackgroundJobOutcome::Deferred
            }
        }
    }
}

pub(crate) fn prepare_embedding_completion_item(
    lease: &EmbeddingJobLease,
    vector: AuthoritativeVector,
) -> Result<EmbeddingJobBatchCompletionItem, BackgroundJobOutcome> {
    EmbeddingJobBatchCompletionItem::new(
        lease.job.embedding_job_id.clone(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        lease.job.attempt_generation,
        lease.job.content_hash.clone(),
        vector,
    )
    .map_err(|_| BackgroundJobOutcome::PermanentFailure)
}

/// Claim and attach one cache-ready materialization.
pub(crate) async fn execute_materialization(
    writer: &LedgerWriterClient,
    candidate: MaterializationWorkCandidate,
    cancellation: watch::Receiver<bool>,
) -> BackgroundJobOutcome {
    execute_materialization_inner(writer, candidate, cancellation, None).await
}

pub(crate) async fn execute_materialization_with_cancellation(
    writer: &LedgerWriterClient,
    candidate: MaterializationWorkCandidate,
    cancellation: BackgroundCancellation,
) -> BackgroundJobOutcome {
    execute_materialization_inner(
        writer,
        candidate,
        cancellation.subscribe(),
        Some(cancellation),
    )
    .await
}

async fn execute_materialization_inner(
    writer: &LedgerWriterClient,
    candidate: MaterializationWorkCandidate,
    cancellation: watch::Receiver<bool>,
    shutdown: Option<BackgroundCancellation>,
) -> BackgroundJobOutcome {
    if *cancellation.borrow() {
        return BackgroundJobOutcome::Deferred;
    }
    let observed_at_unix_ms = now_unix_ms();
    let deadline = match Instant::now().checked_add(WRITER_OPERATION_TIMEOUT) {
        Some(deadline) => bounded_shutdown_deadline(deadline, shutdown.as_ref()),
        None => return BackgroundJobOutcome::PermanentFailure,
    };
    if Instant::now() >= deadline {
        return BackgroundJobOutcome::Deferred;
    }
    let claim = match MaterializationClaim::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        candidate.job.materialization_job_id,
        Uuid::now_v7(),
        candidate.job.attempt_generation,
        candidate.job.canonical_payload_hash,
        observed_at_unix_ms,
    ) {
        Ok(claim) => claim,
        Err(_) => return BackgroundJobOutcome::PermanentFailure,
    };
    let lease = match writer.claim_materialization_until(claim, deadline).await {
        Ok(MaterializationClaimAck::Claimed(lease))
        | Ok(MaterializationClaimAck::Reclaimed(lease))
        | Ok(MaterializationClaimAck::AlreadyApplied(lease)) => lease,
        Ok(MaterializationClaimAck::AttemptLimitTerminal(_)) => {
            return BackgroundJobOutcome::Applied;
        }
        Ok(
            MaterializationClaimAck::CacheNotReady
            | MaterializationClaimAck::IndexNotReady
            | MaterializationClaimAck::NotEligible { .. }
            | MaterializationClaimAck::LeaseHeld { .. }
            | MaterializationClaimAck::Terminal
            | MaterializationClaimAck::TransactionNotStarted,
        )
        | Err(_) => return BackgroundJobOutcome::Deferred,
        Ok(MaterializationClaimAck::NotFound | MaterializationClaimAck::Stale) => {
            return BackgroundJobOutcome::Stale;
        }
        Ok(
            MaterializationClaimAck::Conflict | MaterializationClaimAck::OriginatingProcessNotLive,
        ) => return BackgroundJobOutcome::PermanentFailure,
    };
    if *cancellation.borrow() {
        return release_materialization(
            writer,
            &lease,
            bounded_shutdown_deadline(deadline, shutdown.as_ref()),
        )
        .await;
    }
    let completion = match MaterializationCompletion::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        lease.job.materialization_job_id.clone(),
        lease.lease_token,
        lease.job.attempt_generation,
        lease.job.canonical_payload_hash.clone(),
        now_unix_ms(),
    ) {
        Ok(completion) => completion,
        Err(_) => {
            let _ = release_materialization(
                writer,
                &lease,
                bounded_shutdown_deadline(deadline, shutdown.as_ref()),
            )
            .await;
            return BackgroundJobOutcome::PermanentFailure;
        }
    };
    let completion_deadline = bounded_shutdown_deadline(deadline, shutdown.as_ref());
    match writer
        .complete_materialization_until(completion, completion_deadline)
        .await
    {
        Ok(MaterializationCompletionAck::Applied { .. }) => BackgroundJobOutcome::Applied,
        Ok(MaterializationCompletionAck::AlreadyApplied { .. }) => {
            BackgroundJobOutcome::AlreadyApplied
        }
        Ok(MaterializationCompletionAck::NotFound | MaterializationCompletionAck::StaleLease) => {
            BackgroundJobOutcome::Stale
        }
        Ok(MaterializationCompletionAck::CacheNotReady)
        | Ok(MaterializationCompletionAck::TransactionNotStarted)
        | Err(_) => BackgroundJobOutcome::Deferred,
        Ok(
            MaterializationCompletionAck::Conflict
            | MaterializationCompletionAck::OriginatingProcessNotLive,
        ) => BackgroundJobOutcome::PermanentFailure,
    }
}

/// Advance one exact, at-most-256 dependent terminal-failure window.
pub(crate) async fn execute_failure_propagation(
    writer: &LedgerWriterClient,
    candidate: FailurePropagationCandidate,
) -> BackgroundJobOutcome {
    let expected_cursor = match candidate.job.failure_propagation_cursor.as_deref() {
        Some(value) => match Uuid::parse_str(value) {
            Ok(value) if value.get_version_num() == 7 => Some(value),
            _ => return BackgroundJobOutcome::PermanentFailure,
        },
        None => None,
    };
    let mut items = Vec::with_capacity(candidate.evidence_vector_link_ids.len());
    for evidence_vector_link_id in candidate.evidence_vector_link_ids {
        let item = match MaterializationFailurePropagationItem::new(
            evidence_vector_link_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
        ) {
            Ok(item) => item,
            Err(_) => return BackgroundJobOutcome::PermanentFailure,
        };
        items.push(item);
    }
    let command = match MaterializationFailurePropagation::new(
        Uuid::now_v7(),
        candidate.job.embedding_job_id,
        candidate.job.canonical_payload_hash,
        candidate.job.attempt_generation,
        expected_cursor,
        now_unix_ms(),
        items,
    ) {
        Ok(command) => command,
        Err(_) => return BackgroundJobOutcome::PermanentFailure,
    };
    let deadline = match Instant::now().checked_add(WRITER_OPERATION_TIMEOUT) {
        Some(deadline) => deadline,
        None => return BackgroundJobOutcome::PermanentFailure,
    };
    match writer
        .propagate_materialization_failure_until(command, deadline)
        .await
    {
        Ok(MaterializationFailurePropagationAck::Applied { .. }) => BackgroundJobOutcome::Applied,
        Ok(MaterializationFailurePropagationAck::AlreadyApplied { .. }) => {
            BackgroundJobOutcome::AlreadyApplied
        }
        Ok(
            MaterializationFailurePropagationAck::NotFound
            | MaterializationFailurePropagationAck::NotTerminal
            | MaterializationFailurePropagationAck::Stale,
        ) => BackgroundJobOutcome::Stale,
        Ok(MaterializationFailurePropagationAck::TransactionNotStarted) | Err(_) => {
            BackgroundJobOutcome::Deferred
        }
        Ok(
            MaterializationFailurePropagationAck::Conflict
            | MaterializationFailurePropagationAck::OriginatingProcessNotLive,
        ) => BackgroundJobOutcome::PermanentFailure,
    }
}

/// Atomically materialize one retained terminal into its pool's current vector space.
pub(crate) async fn execute_backfill(
    writer: &LedgerWriterClient,
    candidate: BackfillWorkCandidate,
) -> BackgroundJobOutcome {
    let command = match prepare_backfill_command(&candidate, now_unix_ms()) {
        Ok(command) => command,
        Err(outcome) => return outcome,
    };
    let deadline = match Instant::now().checked_add(WRITER_OPERATION_TIMEOUT) {
        Some(deadline) => deadline,
        None => return BackgroundJobOutcome::PermanentFailure,
    };
    match writer.backfill_vector_graph_until(command, deadline).await {
        Ok(acknowledgement) => backfill_outcome(acknowledgement),
        Err(_) => BackgroundJobOutcome::Deferred,
    }
}

pub(crate) fn prepare_backfill_command(
    candidate: &BackfillWorkCandidate,
    observed_at_unix_ms: i64,
) -> Result<VectorBackfillCommand, BackgroundJobOutcome> {
    if candidate.terminal_at_unix_ms < 0 || observed_at_unix_ms < 0 {
        return Err(BackgroundJobOutcome::PermanentFailure);
    }
    VectorBackfillCommand::new(
        candidate.mapping.clone(),
        candidate.shadow_attempt_id,
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        Uuid::now_v7(),
        observed_at_unix_ms.max(candidate.terminal_at_unix_ms),
    )
    .map_err(|_| BackgroundJobOutcome::PermanentFailure)
}

pub(crate) const fn backfill_outcome(acknowledgement: VectorBackfillAck) -> BackgroundJobOutcome {
    match acknowledgement {
        VectorBackfillAck::Applied => BackgroundJobOutcome::Applied,
        VectorBackfillAck::AlreadyApplied => BackgroundJobOutcome::AlreadyApplied,
        VectorBackfillAck::NotFound
        | VectorBackfillAck::MappingNotFound
        | VectorBackfillAck::MappingNotCurrent => BackgroundJobOutcome::Stale,
        VectorBackfillAck::TransactionNotStarted => BackgroundJobOutcome::Deferred,
        VectorBackfillAck::Conflict | VectorBackfillAck::OriginatingProcessNotLive => {
            BackgroundJobOutcome::PermanentFailure
        }
    }
}

/// Authorize a fresh building generation when inspection finds recoverable drift.
pub(crate) async fn execute_vector_space_recovery(
    writer: &LedgerWriterClient,
    inspection: VectorSpaceInspection,
) -> BackgroundJobOutcome {
    let command = match prepare_vector_space_recovery(inspection, now_unix_ms()) {
        Ok(Some(command)) => command,
        Ok(None) => return BackgroundJobOutcome::AlreadyApplied,
        Err(outcome) => return outcome,
    };
    match vector_mutation(writer, command).await {
        Ok(VectorIndexWriterAck::GenerationAuthorized(acknowledgement)) => {
            generation_authorization_outcome(*acknowledgement)
        }
        Ok(_) => BackgroundJobOutcome::PermanentFailure,
        Err(outcome) => outcome,
    }
}

pub(crate) fn prepare_vector_space_recovery(
    inspection: VectorSpaceInspection,
    observed_at_unix_ms: i64,
) -> Result<Option<VectorIndexWriterCommand>, BackgroundJobOutcome> {
    if observed_at_unix_ms < 0 {
        return Err(BackgroundJobOutcome::PermanentFailure);
    }
    match inspection.recovery {
        None => Ok(None),
        Some(VectorSpaceRecoveryReason::PartialObjects) => {
            Err(BackgroundJobOutcome::PermanentFailure)
        }
        Some(
            VectorSpaceRecoveryReason::MissingCurrent
            | VectorSpaceRecoveryReason::MissingObjects
            | VectorSpaceRecoveryReason::Unavailable
            | VectorSpaceRecoveryReason::Corrupt
            | VectorSpaceRecoveryReason::FingerprintMismatch,
        ) => Ok(Some(VectorIndexWriterCommand::AuthorizeGeneration {
            vector_space_id: inspection.vector_space_id,
            dimensions: inspection.dimensions,
            created_at_unix_ms: observed_at_unix_ms,
        })),
    }
}

pub(crate) fn generation_authorization_outcome(
    acknowledgement: GenerationAuthorizationAck,
) -> BackgroundJobOutcome {
    match acknowledgement {
        GenerationAuthorizationAck::Created(_) => BackgroundJobOutcome::Applied,
        GenerationAuthorizationAck::AlreadyExists(_) => BackgroundJobOutcome::AlreadyApplied,
        GenerationAuthorizationAck::AuthorityMissing => BackgroundJobOutcome::AlreadyApplied,
    }
}

/// Resume one bounded source-only generation rebuild.
pub(crate) async fn execute_rebuild(
    writer: &LedgerWriterClient,
    work: BuildingGenerationWork,
    cancellation: BackgroundCancellation,
) -> BackgroundJobOutcome {
    if work.objects == GenerationObjectsStatus::Partial {
        return BackgroundJobOutcome::PermanentFailure;
    }
    if work.lease_disposition == RebuildLeaseDisposition::Held {
        return BackgroundJobOutcome::Deferred;
    }
    let mut fence = match work.lease_disposition {
        RebuildLeaseDisposition::Owned => match work.lease {
            Some(fence) => fence,
            None => return BackgroundJobOutcome::PermanentFailure,
        },
        RebuildLeaseDisposition::Unclaimed | RebuildLeaseDisposition::Reclaimable => {
            let acknowledgement = match vector_mutation_until(
                writer,
                VectorIndexWriterCommand::ClaimRebuildLease {
                    vector_space_id: work.manifest.vector_space_id().clone(),
                    observed_at_unix_ms: now_unix_ms(),
                },
                cancellation.shutdown_deadline(),
            )
            .await
            {
                Ok(VectorIndexWriterAck::RebuildLeaseClaimed(acknowledgement)) => acknowledgement,
                Ok(_) => return BackgroundJobOutcome::PermanentFailure,
                Err(outcome) => return outcome,
            };
            match acknowledgement {
                RebuildLeaseClaimAck::Claimed(fence)
                | RebuildLeaseClaimAck::Reclaimed(fence)
                | RebuildLeaseClaimAck::AlreadyOwned(fence) => fence,
                RebuildLeaseClaimAck::Held { .. }
                | RebuildLeaseClaimAck::MissingGeneration
                | RebuildLeaseClaimAck::Stale => return BackgroundJobOutcome::Deferred,
                RebuildLeaseClaimAck::ClaimantNotLive => {
                    return BackgroundJobOutcome::PermanentFailure;
                }
            }
        }
        RebuildLeaseDisposition::Held => unreachable!("held rebuild work returned above"),
    };

    if cancellation.is_cancelled() {
        return release_rebuild(writer, &fence, cancellation.shutdown_deadline()).await;
    }
    if work.objects == GenerationObjectsStatus::Missing {
        let creation = vector_mutation_until(
            writer,
            VectorIndexWriterCommand::CreateGenerationObjects {
                fence: fence.clone(),
                observed_at_unix_ms: now_unix_ms(),
            },
            cancellation.shutdown_deadline(),
        )
        .await;
        if cancellation.is_cancelled() {
            return release_rebuild(writer, &fence, cancellation.shutdown_deadline()).await;
        }
        match creation {
            Ok(VectorIndexWriterAck::GenerationObjectsCreated(
                GenerationObjectCreationAck::Created | GenerationObjectCreationAck::AlreadyExists,
            )) => {}
            Ok(VectorIndexWriterAck::GenerationObjectsCreated(
                GenerationObjectCreationAck::Unavailable,
            )) => return BackgroundJobOutcome::Deferred,
            Ok(VectorIndexWriterAck::GenerationObjectsCreated(
                GenerationObjectCreationAck::ManifestMissing,
            )) => return BackgroundJobOutcome::Stale,
            Ok(VectorIndexWriterAck::GenerationObjectsCreated(
                GenerationObjectCreationAck::Partial | GenerationObjectCreationAck::Conflict,
            ))
            | Ok(_) => return BackgroundJobOutcome::PermanentFailure,
            Err(outcome) => return outcome,
        }
    }
    let renewed = renew_rebuild(writer, &mut fence, cancellation.shutdown_deadline()).await;
    if cancellation.is_cancelled() {
        return release_rebuild(writer, &fence, cancellation.shutdown_deadline()).await;
    }
    if !renewed {
        return BackgroundJobOutcome::Stale;
    }

    let population_result = vector_mutation_until(
        writer,
        VectorIndexWriterCommand::PopulateRebuildChunk {
            fence: fence.clone(),
            observed_at_unix_ms: now_unix_ms(),
        },
        cancellation.shutdown_deadline(),
    )
    .await;
    if cancellation.is_cancelled() {
        return release_rebuild(writer, &fence, cancellation.shutdown_deadline()).await;
    }
    let population = match population_result {
        Ok(VectorIndexWriterAck::RebuildStepped(acknowledgement)) => acknowledgement,
        Ok(_) => return BackgroundJobOutcome::PermanentFailure,
        Err(outcome) => return outcome,
    };
    match population {
        RebuildStepAck::Applied {
            complete: false, ..
        } => return BackgroundJobOutcome::Applied,
        RebuildStepAck::Applied { complete: true, .. } => {}
        RebuildStepAck::Stale => return BackgroundJobOutcome::Stale,
        RebuildStepAck::Unavailable => return BackgroundJobOutcome::Deferred,
        RebuildStepAck::Conflict => return BackgroundJobOutcome::PermanentFailure,
    }
    let renewed = renew_rebuild(writer, &mut fence, cancellation.shutdown_deadline()).await;
    if cancellation.is_cancelled() {
        return release_rebuild(writer, &fence, cancellation.shutdown_deadline()).await;
    }
    if !renewed {
        return BackgroundJobOutcome::Stale;
    }

    let catch_up_result = vector_mutation_until(
        writer,
        VectorIndexWriterCommand::CatchUpRebuildChanges {
            fence: fence.clone(),
            observed_at_unix_ms: now_unix_ms(),
        },
        cancellation.shutdown_deadline(),
    )
    .await;
    if cancellation.is_cancelled() {
        return release_rebuild(writer, &fence, cancellation.shutdown_deadline()).await;
    }
    let catch_up = match catch_up_result {
        Ok(VectorIndexWriterAck::RebuildStepped(acknowledgement)) => acknowledgement,
        Ok(_) => return BackgroundJobOutcome::PermanentFailure,
        Err(outcome) => return outcome,
    };
    match catch_up {
        RebuildStepAck::Applied {
            complete: false, ..
        } => return BackgroundJobOutcome::Applied,
        RebuildStepAck::Applied { complete: true, .. } => {}
        RebuildStepAck::Stale => return BackgroundJobOutcome::Stale,
        RebuildStepAck::Unavailable => return BackgroundJobOutcome::Deferred,
        RebuildStepAck::Conflict => return BackgroundJobOutcome::PermanentFailure,
    }
    if cancellation.is_cancelled() {
        return release_rebuild(writer, &fence, cancellation.shutdown_deadline()).await;
    }

    match vector_mutation_until(
        writer,
        VectorIndexWriterCommand::FlipRebuildGeneration {
            fence,
            activated_at_unix_ms: now_unix_ms(),
        },
        cancellation.shutdown_deadline(),
    )
    .await
    {
        Ok(VectorIndexWriterAck::RebuildFlipped(RebuildFlipAck::Activated { .. })) => {
            BackgroundJobOutcome::Applied
        }
        Ok(VectorIndexWriterAck::RebuildFlipped(RebuildFlipAck::NotReady)) => {
            BackgroundJobOutcome::Deferred
        }
        Ok(VectorIndexWriterAck::RebuildFlipped(RebuildFlipAck::Stale)) => {
            BackgroundJobOutcome::Stale
        }
        Ok(VectorIndexWriterAck::RebuildFlipped(
            RebuildFlipAck::FingerprintMismatch | RebuildFlipAck::Unavailable,
        ))
        | Ok(_) => BackgroundJobOutcome::PermanentFailure,
        Err(outcome) => outcome,
    }
}

/// Drop one exact retired generation through the sole writer.
pub(crate) async fn execute_retired_cleanup(
    writer: &LedgerWriterClient,
    work: RetiredGenerationWork,
) -> BackgroundJobOutcome {
    if work.objects == GenerationObjectsStatus::Partial {
        return BackgroundJobOutcome::PermanentFailure;
    }
    match vector_mutation(
        writer,
        VectorIndexWriterCommand::CleanupRetiredGeneration {
            vector_space_id: work.manifest.vector_space_id().clone(),
            generation: work.manifest.generation(),
            dropped_at_unix_ms: now_unix_ms(),
        },
    )
    .await
    {
        Ok(VectorIndexWriterAck::RetiredGenerationCleaned(
            RetiredGenerationCleanupAck::Dropped,
        )) => BackgroundJobOutcome::Applied,
        Ok(VectorIndexWriterAck::RetiredGenerationCleaned(
            RetiredGenerationCleanupAck::ReconciledMissing
            | RetiredGenerationCleanupAck::AlreadyDropped,
        )) => BackgroundJobOutcome::AlreadyApplied,
        Ok(VectorIndexWriterAck::RetiredGenerationCleaned(
            RetiredGenerationCleanupAck::Unavailable,
        )) => BackgroundJobOutcome::Deferred,
        Ok(VectorIndexWriterAck::RetiredGenerationCleaned(
            RetiredGenerationCleanupAck::Partial | RetiredGenerationCleanupAck::Conflict,
        ))
        | Ok(_) => BackgroundJobOutcome::PermanentFailure,
        Err(outcome) => outcome,
    }
}

fn leases_match_plan(leases: &[EmbeddingJobLease], plan: &PreparedEmbeddingBatch) -> bool {
    embedding_leases_match_authority(leases, &plan.claim, &plan.items, &plan.expected_leases)
}

pub(crate) fn embedding_leases_match_authority(
    leases: &[EmbeddingJobLease],
    claim: &EmbeddingJobBatchClaim,
    items: &[EmbedderBatchItem],
    expected_leases: &[ExpectedEmbeddingLease],
) -> bool {
    if leases.len() != claim.items.len()
        || leases.len() != items.len()
        || leases.len() != expected_leases.len()
    {
        return false;
    }
    leases
        .iter()
        .zip(&claim.items)
        .zip(items)
        .zip(expected_leases)
        .all(|(((lease, claim_item), item), expected)| {
            lease.job.embedding_job_id == claim_item.embedding_job_id
                && lease.job.embedding_job_id == item.work_id()
                && lease.job.embedding_job_id == expected.embedding_job_id
                && lease.job.vector_space_id == claim.vector_space_id.as_str()
                && lease.job.canonical_query_hash == item.canonical_query_hash()
                && lease.job.canonical_query_hash == expected.canonical_query_hash
                && lease.job.content_hash == expected.content_hash
                && lease.lease_token == claim.lease_token
                && lease.lease_expires_at_unix_ms == claim.lease_expires_at_unix_ms
        })
}

pub(crate) fn embedding_failure_resolution_kind(
    disposition: EmbedderFailureDisposition,
    stable_error_class: &str,
    attempt_count: i64,
    resolved_at_unix_ms: i64,
) -> EmbeddingJobResolutionKind {
    match disposition {
        EmbedderFailureDisposition::Retryable if stable_error_class == "embedder_cancelled" => {
            EmbeddingJobResolutionKind::Released
        }
        EmbedderFailureDisposition::Retryable => EmbeddingJobResolutionKind::RetryScheduled {
            stable_error_class: stable_error_class.to_string(),
            next_eligible_at_unix_ms: resolved_at_unix_ms.saturating_add(
                i64::try_from(retry_delay(attempt_count).as_millis()).unwrap_or(i64::MAX),
            ),
        },
        EmbedderFailureDisposition::Quarantine => EmbeddingJobResolutionKind::Quarantined {
            stable_error_class: stable_error_class.to_string(),
        },
    }
}

async fn release_embedding_leases(
    writer: &LedgerWriterClient,
    leases: &[EmbeddingJobLease],
    deadline: Instant,
) {
    let resolved_at_unix_ms = now_unix_ms();
    for lease in leases {
        let _ = resolve_embedding_lease(
            writer,
            lease,
            EmbeddingJobResolutionKind::Released,
            resolved_at_unix_ms,
            deadline,
        )
        .await;
    }
}

async fn resolve_embedding_lease(
    writer: &LedgerWriterClient,
    lease: &EmbeddingJobLease,
    kind: EmbeddingJobResolutionKind,
    resolved_at_unix_ms: i64,
    deadline: Instant,
) -> BackgroundJobOutcome {
    let resolution = match EmbeddingJobResolution::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        lease.job.embedding_job_id.clone(),
        lease.lease_token,
        lease.job.attempt_generation,
        lease.job.content_hash.clone(),
        resolved_at_unix_ms,
        kind,
    ) {
        Ok(resolution) => resolution,
        Err(_) => return BackgroundJobOutcome::PermanentFailure,
    };
    match writer
        .resolve_embedding_job_until(resolution, deadline)
        .await
    {
        Ok(acknowledgement) => embedding_resolution_outcome(acknowledgement),
        Err(_) => BackgroundJobOutcome::Deferred,
    }
}

pub(crate) fn embedding_resolution_outcome(
    acknowledgement: EmbeddingJobResolutionAck,
) -> BackgroundJobOutcome {
    match acknowledgement {
        EmbeddingJobResolutionAck::Applied { .. } => BackgroundJobOutcome::Applied,
        EmbeddingJobResolutionAck::AlreadyApplied { .. } => BackgroundJobOutcome::AlreadyApplied,
        EmbeddingJobResolutionAck::NotFound | EmbeddingJobResolutionAck::StaleLease => {
            BackgroundJobOutcome::Stale
        }
        EmbeddingJobResolutionAck::TransactionNotStarted => BackgroundJobOutcome::Deferred,
        EmbeddingJobResolutionAck::Conflict
        | EmbeddingJobResolutionAck::OriginatingProcessNotLive => {
            BackgroundJobOutcome::PermanentFailure
        }
    }
}

async fn release_materialization(
    writer: &LedgerWriterClient,
    lease: &MaterializationLease,
    deadline: Instant,
) -> BackgroundJobOutcome {
    let resolution = match MaterializationResolution::new(
        Uuid::now_v7(),
        Uuid::now_v7(),
        lease.job.materialization_job_id.clone(),
        lease.lease_token,
        lease.job.attempt_generation,
        lease.job.canonical_payload_hash.clone(),
        now_unix_ms(),
        MaterializationResolutionKind::Released,
    ) {
        Ok(resolution) => resolution,
        Err(_) => return BackgroundJobOutcome::PermanentFailure,
    };
    match writer
        .resolve_materialization_until(resolution, deadline)
        .await
    {
        Ok(
            MaterializationResolutionAck::Applied { .. }
            | MaterializationResolutionAck::AlreadyApplied { .. }
            | MaterializationResolutionAck::TransactionNotStarted,
        ) => BackgroundJobOutcome::Deferred,
        Ok(MaterializationResolutionAck::NotFound | MaterializationResolutionAck::StaleLease) => {
            BackgroundJobOutcome::Stale
        }
        Ok(
            MaterializationResolutionAck::Conflict
            | MaterializationResolutionAck::OriginatingProcessNotLive,
        ) => BackgroundJobOutcome::PermanentFailure,
        Err(_) => BackgroundJobOutcome::Deferred,
    }
}

fn bounded_shutdown_deadline(
    default: Instant,
    cancellation: Option<&BackgroundCancellation>,
) -> Instant {
    cancellation
        .and_then(BackgroundCancellation::shutdown_deadline)
        .map(|deadline| deadline.min(default))
        .unwrap_or(default)
}

async fn renew_rebuild(
    writer: &LedgerWriterClient,
    fence: &mut RebuildLeaseFence,
    shutdown_deadline: Option<Instant>,
) -> bool {
    match vector_mutation_until(
        writer,
        VectorIndexWriterCommand::RenewRebuildLease {
            fence: fence.clone(),
            observed_at_unix_ms: now_unix_ms(),
        },
        shutdown_deadline,
    )
    .await
    {
        Ok(VectorIndexWriterAck::RebuildLeaseMutated(RebuildLeaseMutationAck::Applied)) => true,
        Ok(_) | Err(_) => false,
    }
}

async fn release_rebuild(
    writer: &LedgerWriterClient,
    fence: &RebuildLeaseFence,
    shutdown_deadline: Option<Instant>,
) -> BackgroundJobOutcome {
    match vector_mutation_until(
        writer,
        VectorIndexWriterCommand::ReleaseRebuildLease {
            fence: fence.clone(),
            released_at_unix_ms: now_unix_ms(),
        },
        shutdown_deadline,
    )
    .await
    {
        Ok(VectorIndexWriterAck::RebuildLeaseMutated(RebuildLeaseMutationAck::Applied)) => {
            BackgroundJobOutcome::Deferred
        }
        Ok(VectorIndexWriterAck::RebuildLeaseMutated(RebuildLeaseMutationAck::Stale)) => {
            BackgroundJobOutcome::Stale
        }
        Ok(_) => BackgroundJobOutcome::PermanentFailure,
        Err(outcome) => outcome,
    }
}

async fn vector_mutation(
    writer: &LedgerWriterClient,
    command: VectorIndexWriterCommand,
) -> Result<VectorIndexWriterAck, BackgroundJobOutcome> {
    vector_mutation_until(writer, command, None).await
}

async fn vector_mutation_until(
    writer: &LedgerWriterClient,
    command: VectorIndexWriterCommand,
    shutdown_deadline: Option<Instant>,
) -> Result<VectorIndexWriterAck, BackgroundJobOutcome> {
    let operation_deadline = Instant::now()
        .checked_add(WRITER_OPERATION_TIMEOUT)
        .ok_or(BackgroundJobOutcome::PermanentFailure)?;
    let deadline = shutdown_deadline
        .map(|deadline| deadline.min(operation_deadline))
        .unwrap_or(operation_deadline);
    if Instant::now() >= deadline {
        return Err(BackgroundJobOutcome::Deferred);
    }
    writer
        .vector_index_until(command, deadline)
        .await
        .map_err(|_| BackgroundJobOutcome::Deferred)
}

fn retry_delay(attempt_count: i64) -> Duration {
    let exponent = u32::try_from(attempt_count.saturating_sub(1))
        .unwrap_or(u32::MAX)
        .min(9);
    RETRY_BASE
        .checked_mul(1_u32 << exponent)
        .unwrap_or(RETRY_MAX)
        .min(RETRY_MAX)
}

fn now_unix_ms() -> i64 {
    Utc::now().timestamp_millis().max(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retry_delay_is_positive_bounded_and_generation_monotonic() {
        assert_eq!(retry_delay(1), Duration::from_secs(1));
        assert_eq!(retry_delay(2), Duration::from_secs(2));
        assert_eq!(retry_delay(5), Duration::from_secs(16));
        assert_eq!(retry_delay(6), Duration::from_secs(32));
        assert_eq!(retry_delay(10), RETRY_MAX);
        assert_eq!(retry_delay(i64::MAX), RETRY_MAX);
    }

    #[test]
    fn configured_provider_bound_fits_the_exact_lease_margin() {
        assert_eq!(
            Duration::from_millis(
                crate::ledger::repository::embedding::EMBEDDING_JOB_LEASE_MILLIS as u64,
            ),
            Duration::from_secs(60)
        );
        assert_eq!(
            Duration::from_millis(crate::config::EMBEDDER_TIMEOUT_MS_MAX) + PROVIDER_LEASE_MARGIN,
            Duration::from_millis(
                crate::ledger::repository::embedding::EMBEDDING_JOB_LEASE_MILLIS as u64,
            )
        );
    }
}
