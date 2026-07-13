// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded sole-writer transport for the Router ledger.

use std::ops::{Deref, DerefMut};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

use rusqlite::InterruptHandle;
use tokio::sync::{mpsc, oneshot, watch};

use super::command::{DrainAck, FlushAck, LedgerWriterCommand, WriterFailure, WriterFailureClass};
use super::repository::active::{
    ActiveAdmissionAck, ActiveDispatchTerminal, ActiveDispatchTerminalAck, ActiveRootAdmission,
    ActiveRootTerminal, ActiveRootTerminalAck, ActiveSignalBatch, ActiveSignalBatchAck,
};
use super::repository::active_decision::{ActiveDecisionAdmissionAckV2, ActiveDecisionAdmissionV2};
use super::repository::active_learning::{
    ActiveAuthorizationObserveAck, ActiveExperimentCreate, ActiveExperimentCreateAck,
    ActiveLookClaimAck, ActiveLookClaimRequest, ActiveLookCommit, ActiveLookCommitAck,
    ActiveLookFailure, ActiveLookFailureAck, ActiveLookLeaseAck, ActiveLookLeaseRenewal,
    ActiveNeighborhoodInvalidation, ActiveNeighborhoodKey, ActiveNeighborhoodMutationAck,
    ActiveNeighborhoodObserveAck, ActiveNeighborhoodPromotion,
};
use super::repository::anchors::{
    AnchorCommandAck, AnchorProbe, FrozenPendingAnchorV1, FrozenTerminalAnchorV1,
    NotScheduledQueueFull, NotScheduledQueueFullAck, PendingAnchorCapacityAck,
};
use super::repository::control::{ControlMutationAck, PreparedControlMutation};
use super::repository::cooloff::{
    CandidateDependencyIdentity, DependencyCommandAck, DependencyCompletion, DependencyOperation,
    JudgeDependencyIdentity,
};
use super::repository::decision::DecisionAuditAck;
use super::repository::embedding::{
    EmbeddingJobBatchClaim, EmbeddingJobBatchClaimAck, EmbeddingJobBatchCompletion,
    EmbeddingJobBatchCompletionAck, EmbeddingJobCreate, EmbeddingJobCreateAck, EmbeddingJobReset,
    EmbeddingJobResetAck, EmbeddingJobResolution, EmbeddingJobResolutionAck, LiveEmbeddingPrepare,
    LiveEmbeddingPrepareAck,
};
use super::repository::inspection::operator::{
    OperatorMutationTransactionAck, PreparedOperatorMutation,
};
use super::repository::judge::{
    EvaluationRecord, JudgeAttemptStart, JudgeAttemptTerminal, JudgeRecordAck,
};
use super::repository::materialization::{
    MaterializationClaim, MaterializationClaimAck, MaterializationCompletion,
    MaterializationCompletionAck, MaterializationFailurePropagation,
    MaterializationFailurePropagationAck, MaterializationResolution, MaterializationResolutionAck,
    VectorBackfillAck, VectorBackfillCommand,
};
use super::repository::process::{
    HeartbeatAck, HeartbeatRenewal, LedgerHealthEvent, ProcessCommandAck, ProcessStop,
};
use super::repository::retention::{RetentionAck, RetentionRequest};
use super::repository::shadow::{
    SampleBatchReservation, ShadowAttemptStarted, ShadowCommandAck, ShadowTerminalRecord,
};
use super::repository::vector_registry::{RegistryEnsureAck, VectorRegistryEnsure};
use super::repository::vector_registry_writer::RegistryEnsureTransactionAck;
use super::repository::{LedgerRepository, TransactionStartGuard};
use crate::control::{ControlTransactionFence, RouterControlError};
use crate::decision_audit::DecisionAuditV1;
use crate::inspection::InspectionError;
use crate::sqlite_vector_store::{
    VectorIndexTransactionAck, VectorIndexWriterAck, VectorIndexWriterCommand,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterExitClass {
    Clean,
    Aborted,
    Panicked,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WriterLifecycleState {
    Running,
    Draining,
    Aborted,
    Exited(WriterExitClass),
}

struct WriterEnvelope {
    abort_epoch: u64,
    transaction_start_deadline: Option<Instant>,
    command: LedgerWriterCommand,
}

struct WriterShared {
    sender: Mutex<Option<mpsc::Sender<WriterEnvelope>>>,
    lifecycle_gate: Mutex<()>,
    lifecycle: watch::Sender<WriterLifecycleState>,
    abort_epoch: AtomicU64,
    accepting: AtomicBool,
    stop_process_on_abort: AtomicBool,
    interrupt: Arc<InterruptHandle>,
}

impl WriterShared {
    fn sender_snapshot(&self) -> Result<(mpsc::Sender<WriterEnvelope>, u64), WriterFailure> {
        if !self.accepting.load(Ordering::Acquire) {
            return Err(failure_for_state(*self.lifecycle.borrow()));
        }
        let abort_epoch = self.abort_epoch.load(Ordering::Acquire);
        let sender = self
            .sender
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .as_ref()
            .cloned()
            .ok_or_else(|| WriterFailure::new(WriterFailureClass::Closing))?;
        if !self.accepting.load(Ordering::Acquire)
            || self.abort_epoch.load(Ordering::Acquire) != abort_epoch
        {
            return Err(failure_for_state(*self.lifecycle.borrow()));
        }
        Ok((sender, abort_epoch))
    }

    fn reservation_is_current(&self, abort_epoch: u64) -> Result<(), WriterFailure> {
        if self.abort_epoch.load(Ordering::Acquire) != abort_epoch {
            return Err(WriterFailureClass::Aborted.into());
        }
        if !self.accepting.load(Ordering::Acquire) {
            return Err(failure_for_state(*self.lifecycle.borrow()));
        }
        Ok(())
    }

    fn abort(&self) {
        let _lifecycle = self
            .lifecycle_gate
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if self.abort_epoch.load(Ordering::Acquire) != 0
            || matches!(*self.lifecycle.borrow(), WriterLifecycleState::Exited(_))
        {
            return;
        }
        self.accepting.store(false, Ordering::Release);
        self.interrupt.interrupt();
        self.abort_epoch.fetch_add(1, Ordering::AcqRel);
        self.lifecycle.send_replace(WriterLifecycleState::Aborted);
        self.sender
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();
    }
}

/// Cloneable bounded command client. It never owns the writer thread.
#[derive(Clone)]
pub(crate) struct LedgerWriterClient {
    shared: Arc<WriterShared>,
}

impl LedgerWriterClient {
    /// Synchronously close writer admission and interrupt any SQLite operation.
    pub(crate) fn abort(&self) {
        self.shared.abort();
    }

    /// Probe one frozen pending anchor through the sole writer.
    #[allow(dead_code)] // Task 7 scheduler wiring consumes this typed command.
    pub(crate) async fn probe_pending_anchor_until(
        &self,
        pending: FrozenPendingAnchorV1,
        deadline: Instant,
    ) -> Result<AnchorProbe, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        self.probe_pending_anchor_with_permit_until(permit, pending, deadline)
            .await
    }

    /// Consume pre-accept writer capacity to probe one frozen pending anchor.
    #[allow(dead_code)] // Task 7 sink wiring consumes the pre-accept authority.
    pub(crate) async fn probe_pending_anchor_with_permit_until(
        &self,
        permit: WriterCommandPermit,
        pending: FrozenPendingAnchorV1,
        deadline: Instant,
    ) -> Result<AnchorProbe, WriterFailure> {
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::ProbePendingAnchor {
            pending: Box::new(pending),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Probe with pre-accept authority until reply or writer lifecycle termination.
    #[allow(dead_code)] // Task 7 sink owns this pre-accept command to acknowledgement.
    pub(crate) async fn probe_pending_anchor_with_permit(
        &self,
        permit: WriterCommandPermit,
        pending: FrozenPendingAnchorV1,
    ) -> Result<AnchorProbe, WriterFailure> {
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::ProbePendingAnchor {
            pending: Box::new(pending),
            reply,
        });
        await_reply_while_running(receiver, self.shared.lifecycle.subscribe()).await
    }

    /// Persist one frozen pending anchor through the sole writer.
    #[allow(dead_code)] // Task 7 scheduler wiring consumes this typed command.
    pub(crate) async fn record_pending_anchor_until(
        &self,
        pending: FrozenPendingAnchorV1,
        deadline: Instant,
    ) -> Result<AnchorCommandAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        self.record_pending_anchor_with_permit_until(permit, pending, deadline)
            .await
    }

    /// Consume pre-accept writer capacity to persist one frozen pending anchor.
    #[allow(dead_code)] // Task 7 sink wiring consumes the pre-accept authority.
    pub(crate) async fn record_pending_anchor_with_permit_until(
        &self,
        permit: WriterCommandPermit,
        pending: FrozenPendingAnchorV1,
        deadline: Instant,
    ) -> Result<AnchorCommandAck, WriterFailure> {
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordPendingAnchor {
            pending: Box::new(pending),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Consume pre-accept capacity and enforce terminal-evidence capacity in SQLite.
    #[allow(dead_code)] // Task 7 sink wiring consumes this capacity-aware command.
    pub(crate) async fn record_pending_anchor_with_capacity_and_permit_until(
        &self,
        permit: WriterCommandPermit,
        pending: FrozenPendingAnchorV1,
        max_evidence_records: u64,
        deadline: Instant,
    ) -> Result<PendingAnchorCapacityAck, WriterFailure> {
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordPendingAnchorWithCapacity {
            pending: Box::new(pending),
            max_evidence_records,
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Persist capacity-aware pending work until reply or lifecycle termination.
    #[allow(dead_code)] // Task 7 sink owns this pre-accept command to acknowledgement.
    pub(crate) async fn record_pending_anchor_with_capacity_and_permit(
        &self,
        permit: WriterCommandPermit,
        pending: FrozenPendingAnchorV1,
        max_evidence_records: u64,
    ) -> Result<PendingAnchorCapacityAck, WriterFailure> {
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordPendingAnchorWithCapacity {
            pending: Box::new(pending),
            max_evidence_records,
            reply,
        });
        await_reply_while_running(receiver, self.shared.lifecycle.subscribe()).await
    }

    /// Persist an atomic pending plus queue-full terminal through the sole writer.
    #[allow(dead_code)] // Task 7 sink wiring consumes this convenience path.
    pub(crate) async fn record_not_scheduled_queue_full_until(
        &self,
        decline: NotScheduledQueueFull,
        deadline: Instant,
    ) -> Result<NotScheduledQueueFullAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        self.record_not_scheduled_queue_full_with_permit_until(permit, decline, deadline)
            .await
    }

    /// Consume pre-accept writer capacity to persist an atomic queue-full decline.
    #[allow(dead_code)] // Task 7 sink wiring consumes the pre-accept authority.
    pub(crate) async fn record_not_scheduled_queue_full_with_permit_until(
        &self,
        permit: WriterCommandPermit,
        decline: NotScheduledQueueFull,
        deadline: Instant,
    ) -> Result<NotScheduledQueueFullAck, WriterFailure> {
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordNotScheduledQueueFull {
            decline: Box::new(decline),
            max_evidence_records: None,
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Consume pre-accept capacity and enforce terminal-evidence capacity in SQLite.
    #[allow(dead_code)] // Task 7 sink wiring consumes this capacity-aware command.
    pub(crate) async fn record_not_scheduled_queue_full_with_capacity_and_permit_until(
        &self,
        permit: WriterCommandPermit,
        decline: NotScheduledQueueFull,
        max_evidence_records: u64,
        deadline: Instant,
    ) -> Result<NotScheduledQueueFullAck, WriterFailure> {
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordNotScheduledQueueFull {
            decline: Box::new(decline),
            max_evidence_records: Some(max_evidence_records),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Persist a capacity-aware decline until reply or lifecycle termination.
    #[allow(dead_code)] // Task 7 sink owns this pre-accept command to acknowledgement.
    pub(crate) async fn record_not_scheduled_queue_full_with_capacity_and_permit(
        &self,
        permit: WriterCommandPermit,
        decline: NotScheduledQueueFull,
        max_evidence_records: u64,
    ) -> Result<NotScheduledQueueFullAck, WriterFailure> {
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordNotScheduledQueueFull {
            decline: Box::new(decline),
            max_evidence_records: Some(max_evidence_records),
            reply,
        });
        await_reply_while_running(receiver, self.shared.lifecycle.subscribe()).await
    }

    /// Persist one frozen terminal anchor through the sole writer.
    #[allow(dead_code)] // Task 7 scheduler wiring consumes this typed command.
    pub(crate) async fn record_terminal_anchor_until(
        &self,
        terminal: FrozenTerminalAnchorV1,
        deadline: Instant,
    ) -> Result<AnchorCommandAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordTerminalAnchor {
            terminal: Box::new(terminal),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Reserve one sample batch and its attempts through the sole writer.
    #[allow(dead_code)] // Task 7 scheduler wiring consumes this typed command.
    pub(crate) async fn reserve_sample_batch_until(
        &self,
        reservation: SampleBatchReservation,
        deadline: Instant,
    ) -> Result<ShadowCommandAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::ReserveSampleBatch {
            reservation: Box::new(reservation),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Mark one reserved Shadow attempt started through the sole writer.
    #[allow(dead_code)] // Task 7 scheduler wiring consumes this typed command.
    pub(crate) async fn start_shadow_attempt_until(
        &self,
        command: ShadowAttemptStarted,
        deadline: Instant,
    ) -> Result<ShadowCommandAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::StartShadowAttempt { command, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Persist one Shadow terminal aggregate through the sole writer.
    #[allow(dead_code)] // Task 7 scheduler wiring consumes this typed command.
    pub(crate) async fn record_shadow_terminal_until(
        &self,
        command: ShadowTerminalRecord,
        deadline: Instant,
    ) -> Result<ShadowCommandAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordShadowTerminal {
            command: Box::new(command),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Persist one Judge-attempt start through the sole writer.
    #[allow(dead_code)] // Task 8 Judge wiring consumes this typed command.
    pub(crate) async fn record_judge_attempt_start_until(
        &self,
        start: JudgeAttemptStart,
        deadline: Instant,
    ) -> Result<JudgeRecordAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordJudgeAttemptStart {
            start: Box::new(start),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Persist one Judge-attempt terminal through the sole writer.
    #[allow(dead_code)] // Task 8 Judge wiring consumes this typed command.
    pub(crate) async fn record_judge_attempt_terminal_until(
        &self,
        terminal: JudgeAttemptTerminal,
        deadline: Instant,
    ) -> Result<JudgeRecordAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordJudgeAttemptTerminal {
            terminal: Box::new(terminal),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Persist one deterministic evaluation through the sole writer.
    #[allow(dead_code)] // Task 7 deterministic validation wiring consumes this command.
    pub(crate) async fn record_evaluation_until(
        &self,
        evaluation: EvaluationRecord,
        deadline: Instant,
    ) -> Result<JudgeRecordAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordEvaluation {
            evaluation: Box::new(evaluation),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Append one immutable recommendation decision through the sole writer.
    #[allow(dead_code)] // Spec 07 Task 8 owns foreground delivery and exact retry.
    pub(crate) async fn record_decision_audit_until(
        &self,
        audit: Arc<DecisionAuditV1>,
        max_evidence_records: u64,
        conflict_health_event_id: uuid::Uuid,
        deadline: Instant,
    ) -> Result<DecisionAuditAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        self.record_decision_audit_with_permit_until(
            permit,
            audit,
            max_evidence_records,
            conflict_health_event_id,
            deadline,
        )
        .await
    }

    /// Fence transaction start independently while retaining the same attempt
    /// until its definitive reply or writer lifecycle termination.
    pub(crate) async fn record_decision_audit_with_start_deadline(
        &self,
        audit: Arc<DecisionAuditV1>,
        max_evidence_records: u64,
        conflict_health_event_id: uuid::Uuid,
        transaction_start_deadline: Instant,
    ) -> Result<DecisionAuditAck, WriterFailure> {
        let permit = self
            .reserve_accepted_until(transaction_start_deadline)
            .await?;
        let (reply, receiver) = oneshot::channel();
        permit.send_with_transaction_start_deadline(
            LedgerWriterCommand::RecordDecisionAudit {
                audit,
                max_evidence_records,
                conflict_health_event_id,
                reply,
            },
            transaction_start_deadline,
        );
        await_reply_while_running(receiver, self.shared.lifecycle.subscribe()).await
    }

    /// Consume pre-accept capacity and fence the decision transaction start.
    #[allow(dead_code)] // Spec 07 Task 8 reserves delivery ownership before computation.
    pub(crate) async fn record_decision_audit_with_permit_until(
        &self,
        permit: WriterCommandPermit,
        audit: Arc<DecisionAuditV1>,
        max_evidence_records: u64,
        conflict_health_event_id: uuid::Uuid,
        deadline: Instant,
    ) -> Result<DecisionAuditAck, WriterFailure> {
        let (reply, receiver) = oneshot::channel();
        permit.send_with_transaction_start_deadline(
            LedgerWriterCommand::RecordDecisionAudit {
                audit,
                max_evidence_records,
                conflict_health_event_id,
                reply,
            },
            deadline,
        );
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Claim one candidate-provider dependency through the sole writer.
    #[allow(dead_code)] // Task 7 scheduler wiring consumes this typed command.
    pub(crate) async fn claim_candidate_dependency_until(
        &self,
        identity: CandidateDependencyIdentity,
        operation: DependencyOperation,
        deadline: Instant,
    ) -> Result<DependencyCommandAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::ClaimCandidateDependency {
            identity,
            operation,
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Claim one judge-provider dependency through the sole writer.
    #[allow(dead_code)] // Task 7 scheduler wiring consumes this typed command.
    pub(crate) async fn claim_judge_dependency_until(
        &self,
        identity: JudgeDependencyIdentity,
        operation: DependencyOperation,
        deadline: Instant,
    ) -> Result<DependencyCommandAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::ClaimJudgeDependency {
            identity,
            operation,
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Complete one admitted dependency through the sole writer.
    #[allow(dead_code)] // Task 7 scheduler wiring consumes this typed command.
    pub(crate) async fn complete_dependency_until(
        &self,
        completion: DependencyCompletion,
        deadline: Instant,
    ) -> Result<DependencyCommandAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::CompleteDependency { completion, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Atomically recheck cache/space health and create or reuse one current-space live job.
    #[allow(dead_code)] // Task 11 live embedding consumes this bounded writer command.
    pub(crate) async fn prepare_live_embedding_until(
        &self,
        command: LiveEmbeddingPrepare,
        deadline: Instant,
    ) -> Result<LiveEmbeddingPrepareAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::PrepareLiveEmbedding {
            command: Box::new(command),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Create or reuse one deterministic embedding job through the sole writer.
    #[allow(dead_code)] // Spec 06 consumes the Task 10 durable job transport.
    pub(crate) async fn create_embedding_job_until(
        &self,
        command: EmbeddingJobCreate,
        deadline: Instant,
    ) -> Result<EmbeddingJobCreateAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::CreateEmbeddingJob { command, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Atomically claim or reclaim one bounded same-space embedding batch.
    #[allow(dead_code)] // Task 9 provider workers consume the bounded batch transport.
    pub(crate) async fn claim_embedding_job_batch_until(
        &self,
        command: EmbeddingJobBatchClaim,
        deadline: Instant,
    ) -> Result<EmbeddingJobBatchClaimAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send_with_transaction_start_deadline(
            LedgerWriterCommand::ClaimEmbeddingJobBatch { command, reply },
            deadline,
        );
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Atomically persist vectors and complete one claimed embedding batch.
    #[allow(dead_code)] // Task 9 provider workers consume the bounded batch transport.
    pub(crate) async fn complete_embedding_job_batch_until(
        &self,
        command: EmbeddingJobBatchCompletion,
        deadline: Instant,
    ) -> Result<EmbeddingJobBatchCompletionAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::CompleteEmbeddingJobBatch { command, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Resolve one claimed embedding job without producing a vector.
    #[allow(dead_code)] // Task 9 provider workers consume the bounded failure transport.
    pub(crate) async fn resolve_embedding_job_until(
        &self,
        command: EmbeddingJobResolution,
        deadline: Instant,
    ) -> Result<EmbeddingJobResolutionAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::ResolveEmbeddingJob { command, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Apply one explicit generation-fenced reset to a propagated terminal job.
    #[allow(dead_code)] // Task 11 administrative reconciliation consumes explicit reset.
    pub(crate) async fn reset_embedding_job_until(
        &self,
        command: EmbeddingJobReset,
        deadline: Instant,
    ) -> Result<EmbeddingJobResetAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::ResetEmbeddingJob { command, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Claim or reclaim one cache-backed vector materialization.
    #[allow(dead_code)] // Task 10 materializer workers consume this transport.
    pub(crate) async fn claim_materialization_until(
        &self,
        command: MaterializationClaim,
        deadline: Instant,
    ) -> Result<MaterializationClaimAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send_with_transaction_start_deadline(
            LedgerWriterCommand::ClaimMaterialization { command, reply },
            deadline,
        );
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Attach authoritative cache data and advance one materialization.
    #[allow(dead_code)] // Task 10 materializer workers consume this transport.
    pub(crate) async fn complete_materialization_until(
        &self,
        command: MaterializationCompletion,
        deadline: Instant,
    ) -> Result<MaterializationCompletionAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::CompleteMaterialization { command, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Release or defer one current materialization lease.
    #[allow(dead_code)] // Task 10 materializer workers consume this transport.
    pub(crate) async fn resolve_materialization_until(
        &self,
        command: MaterializationResolution,
        deadline: Instant,
    ) -> Result<MaterializationResolutionAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::ResolveMaterialization { command, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Propagate one bounded window of a shared embedding terminal failure.
    #[allow(dead_code)] // Task 10 reconciliation consumes bounded failure fanout.
    pub(crate) async fn propagate_materialization_failure_until(
        &self,
        command: MaterializationFailurePropagation,
        deadline: Instant,
    ) -> Result<MaterializationFailurePropagationAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::PropagateMaterializationFailure {
            command: Box::new(command),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Atomically materialize one retained terminal into its current pool vector space.
    #[allow(dead_code)] // Task 10 backfill workers consume this transport.
    pub(crate) async fn backfill_vector_graph_until(
        &self,
        command: VectorBackfillCommand,
        deadline: Instant,
    ) -> Result<VectorBackfillAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::BackfillVectorGraph {
            command: Box::new(command),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Serialize one sqlite-vec mutation through the existing ledger writer.
    pub(crate) async fn vector_index_until(
        &self,
        command: VectorIndexWriterCommand,
        deadline: Instant,
    ) -> Result<VectorIndexWriterAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let claim_start_deadline =
            matches!(command, VectorIndexWriterCommand::ClaimRebuildLease { .. })
                .then_some(deadline);
        let (reply, receiver) = oneshot::channel();
        let command = LedgerWriterCommand::VectorIndex {
            command: Box::new(command),
            reply,
        };
        if let Some(start_deadline) = claim_start_deadline {
            permit.send_with_transaction_start_deadline(command, start_deadline);
        } else {
            permit.send(command);
        }
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Ensure one exact referenced vector registry through the sole writer.
    #[allow(dead_code)] // Task 7 registry wiring consumes this typed command after activation.
    pub(crate) async fn ensure_vector_registry_until(
        &self,
        ensure: VectorRegistryEnsure,
        deadline: Instant,
    ) -> Result<RegistryEnsureAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::EnsureVectorRegistry {
            ensure: Box::new(ensure),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Create or reload one immutable Active experiment through the sole writer.
    #[allow(dead_code)] // Task 10 creates experiments from planner admission.
    pub(crate) async fn create_active_experiment_until(
        &self,
        create: ActiveExperimentCreate,
        deadline: Instant,
    ) -> Result<ActiveExperimentCreateAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::CreateActiveExperiment {
            create: Box::new(create),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Claim the next fully drained Active outcome boundary through the sole writer.
    pub(crate) async fn claim_active_look_until(
        &self,
        request: ActiveLookClaimRequest,
        deadline: Instant,
    ) -> Result<ActiveLookClaimAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::ClaimActiveLook { request, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Renew one owned Active outcome-look lease through the sole writer.
    pub(crate) async fn renew_active_look_lease_until(
        &self,
        renewal: ActiveLookLeaseRenewal,
        deadline: Instant,
    ) -> Result<ActiveLookLeaseAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RenewActiveLookLease { renewal, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Commit one completed Active outcome look through the sole writer.
    pub(crate) async fn commit_active_look_until(
        &self,
        commit: ActiveLookCommit,
        deadline: Instant,
    ) -> Result<ActiveLookCommitAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::CommitActiveLook { commit, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Persist one failed Active look evaluation through the sole writer.
    pub(crate) async fn fail_active_look_until(
        &self,
        failure: ActiveLookFailure,
        deadline: Instant,
    ) -> Result<ActiveLookFailureAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::FailActiveLook { failure, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Observe expiry or supersession and return current experiment authorization.
    #[allow(dead_code)] // Task 10 consumes fresh authorization in the Active planner.
    pub(crate) async fn observe_active_authorization_until(
        &self,
        active_experiment_id: uuid::Uuid,
        observed_at_unix_ms: i64,
        deadline: Instant,
    ) -> Result<ActiveAuthorizationObserveAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::ObserveActiveAuthorization {
            active_experiment_id,
            observed_at_unix_ms,
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Promote or restore one query-local Active neighborhood through the sole writer.
    #[allow(dead_code)] // Task 10 consumes promotion hysteresis in the Active planner.
    pub(crate) async fn promote_active_neighborhood_until(
        &self,
        promotion: ActiveNeighborhoodPromotion,
        deadline: Instant,
    ) -> Result<ActiveNeighborhoodMutationAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::PromoteActiveNeighborhood {
            promotion: Box::new(promotion),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Atomically invalidate and cool off one query-local Active neighborhood.
    #[allow(dead_code)] // Task 11 wires candidate failure outcomes to local invalidation.
    pub(crate) async fn invalidate_active_neighborhood_until(
        &self,
        invalidation: ActiveNeighborhoodInvalidation,
        deadline: Instant,
    ) -> Result<ActiveNeighborhoodMutationAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::InvalidateActiveNeighborhood {
            invalidation: Box::new(invalidation),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Observe current query-local Active neighborhood authority through the sole writer.
    #[allow(dead_code)] // Task 10 consumes fresh neighborhood authority in the planner.
    pub(crate) async fn observe_active_neighborhood_until(
        &self,
        key: ActiveNeighborhoodKey,
        observed_at_unix_ms: i64,
        deadline: Instant,
    ) -> Result<ActiveNeighborhoodObserveAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::ObserveActiveNeighborhood {
            key: Box::new(key),
            observed_at_unix_ms,
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Atomically persist one randomized root assignment through the sole writer.
    #[allow(dead_code)] // Task 11 connects active foreground admission to this command.
    pub(crate) async fn admit_active_root_until(
        &self,
        admission: ActiveRootAdmission,
        deadline: Instant,
    ) -> Result<ActiveAdmissionAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::AdmitActiveRoot {
            admission: Box::new(admission),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Atomically persist one shape-2 decision, randomized assignment, and dispatch authority.
    #[allow(dead_code)] // Task 11 submits foreground Active plans through this boundary.
    pub(crate) async fn admit_active_decision_until(
        &self,
        admission: ActiveDecisionAdmissionV2,
        deadline: Instant,
    ) -> Result<ActiveDecisionAdmissionAckV2, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::AdmitActiveDecision {
            admission: Box::new(admission),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Append one bounded protected-signal batch through the sole writer.
    #[allow(dead_code)] // Task 11 connects the outcome projector to this command.
    pub(crate) async fn append_active_signals_until(
        &self,
        batch: ActiveSignalBatch,
        deadline: Instant,
    ) -> Result<ActiveSignalBatchAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::AppendActiveSignals {
            batch: Box::new(batch),
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Persist one candidate-dispatch terminal fact through the sole writer.
    #[allow(dead_code)] // Task 11 connects active dispatch lifecycle to this command.
    pub(crate) async fn record_active_dispatch_terminal_until(
        &self,
        terminal: ActiveDispatchTerminal,
        deadline: Instant,
    ) -> Result<ActiveDispatchTerminalAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordActiveDispatchTerminal { terminal, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    pub(crate) async fn reserve_active_dispatch_terminal_until(
        &self,
        deadline: Instant,
    ) -> Result<ActiveDispatchTerminalPermit, WriterFailure> {
        Ok(ActiveDispatchTerminalPermit {
            permit: Some(self.reserve_accepted_until(deadline).await?),
            lifecycle: self.shared.lifecycle.subscribe(),
        })
    }

    /// Rescan and terminalize one active root through the sole writer.
    #[allow(dead_code)] // Task 11 connects active root closure to this command.
    pub(crate) async fn terminalize_active_root_until(
        &self,
        terminal: ActiveRootTerminal,
        deadline: Instant,
    ) -> Result<ActiveRootTerminalAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::TerminalizeActiveRoot { terminal, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Run one deterministic retention transaction through the sole writer.
    pub(crate) async fn run_retention_until(
        &self,
        request: RetentionRequest,
        deadline: Instant,
    ) -> Result<RetentionAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RunRetention { request, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Renew this process's durable heartbeat through the sole writer.
    pub(crate) async fn renew_heartbeat_until(
        &self,
        renewal: HeartbeatRenewal,
        deadline: Instant,
    ) -> Result<HeartbeatAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RenewHeartbeat { renewal, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Append this process's terminal state through the sole writer.
    #[allow(dead_code)] // Task 10 graceful lifecycle wiring consumes this typed command.
    pub(crate) async fn stop_process_until(
        &self,
        command: ProcessStop,
        deadline: Instant,
    ) -> Result<ProcessCommandAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::StopProcess { command, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Append one stable ledger health event through the sole writer.
    #[allow(dead_code)] // Task 7 and Task 10 health wiring consume this typed command.
    pub(crate) async fn append_health_event_until(
        &self,
        event: LedgerHealthEvent,
        deadline: Instant,
    ) -> Result<ProcessCommandAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::AppendHealthEvent { event, reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    /// Apply one control mutation with a transaction-start-only deadline.
    pub(crate) async fn apply_control_mutation(
        &self,
        mutation: PreparedControlMutation,
        fence: Arc<ControlTransactionFence>,
    ) -> Result<Result<ControlMutationAck, RouterControlError>, WriterFailure> {
        let deadline = fence.start_deadline();
        let permit = match self.reserve_accepted_until(deadline).await {
            Ok(permit) => permit,
            Err(error) => {
                fence.expire();
                return Err(error);
            }
        };
        let (reply, receiver) = oneshot::channel();
        permit.send_with_transaction_start_deadline(
            LedgerWriterCommand::ApplyControlMutation {
                mutation,
                fence,
                reply,
            },
            deadline,
        );
        await_reply_while_running(receiver, self.shared.lifecycle.subscribe()).await
    }

    #[cfg(test)]
    pub(crate) async fn apply_control_detached_for_test(
        &self,
        mutation: PreparedControlMutation,
        fence: Arc<ControlTransactionFence>,
    ) -> Result<(), WriterFailure> {
        let deadline = fence.start_deadline();
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send_with_transaction_start_deadline(
            LedgerWriterCommand::ApplyControlMutation {
                mutation,
                fence,
                reply,
            },
            deadline,
        );
        drop(receiver);
        Ok(())
    }

    /// Apply one reset or cohort rotation with a transaction-start-only deadline.
    pub(crate) async fn apply_operator_mutation(
        &self,
        mutation: PreparedOperatorMutation,
        fence: Arc<ControlTransactionFence>,
    ) -> Result<Result<OperatorMutationTransactionAck, InspectionError>, WriterFailure> {
        let deadline = fence.start_deadline();
        let permit = match self.reserve_accepted_until(deadline).await {
            Ok(permit) => permit,
            Err(error) => {
                fence.expire();
                return Err(error);
            }
        };
        let (reply, receiver) = oneshot::channel();
        permit.send_with_transaction_start_deadline(
            LedgerWriterCommand::ApplyOperatorMutation {
                mutation,
                fence,
                reply,
            },
            deadline,
        );
        await_reply_while_running(receiver, self.shared.lifecycle.subscribe()).await
    }

    /// Reserve one writer slot without waiting before accepting evidence.
    #[allow(dead_code)] // Task 7 consumes pre-accept reservations.
    pub(crate) fn try_reserve_pre_accept(&self) -> Result<WriterCommandPermit, WriterFailure> {
        let (sender, abort_epoch) = self.shared.sender_snapshot()?;
        let permit = sender.try_reserve_owned().map_err(|error| match error {
            mpsc::error::TrySendError::Full(_) => WriterFailure::new(WriterFailureClass::Full),
            mpsc::error::TrySendError::Closed(_) => {
                failure_for_state(*self.shared.lifecycle.borrow())
            }
        })?;
        self.shared.reservation_is_current(abort_epoch)?;
        Ok(WriterCommandPermit {
            permit,
            abort_epoch,
        })
    }

    /// Wait for one writer slot while observing liveness and an absolute deadline.
    pub(crate) async fn reserve_accepted_until(
        &self,
        deadline: Instant,
    ) -> Result<WriterCommandPermit, WriterFailure> {
        if Instant::now() >= deadline {
            return Err(WriterFailureClass::Deadline.into());
        }
        let (sender, abort_epoch) = self.shared.sender_snapshot()?;
        let mut lifecycle = self.shared.lifecycle.subscribe();
        let tokio_deadline = tokio::time::Instant::from_std(deadline);
        let permit = loop {
            tokio::select! {
                biased;
                result = sender.clone().reserve_owned() => {
                    break result.map_err(|_| failure_for_state(*lifecycle.borrow()))?;
                }
                changed = lifecycle.changed() => {
                    if changed.is_err() || lifecycle_is_terminal(*lifecycle.borrow()) {
                        return Err(failure_for_state(*lifecycle.borrow()));
                    }
                    if matches!(*lifecycle.borrow(), WriterLifecycleState::Draining) {
                        return Err(WriterFailureClass::Closing.into());
                    }
                }
                () = tokio::time::sleep_until(tokio_deadline) => {
                    return Err(WriterFailureClass::Deadline.into());
                }
            }
        };
        if Instant::now() >= deadline {
            return Err(WriterFailureClass::Deadline.into());
        }
        self.shared.reservation_is_current(abort_epoch)?;
        if Instant::now() >= deadline {
            return Err(WriterFailureClass::Deadline.into());
        }
        Ok(WriterCommandPermit {
            permit,
            abort_epoch,
        })
    }

    /// Enqueue and await a FIFO flush acknowledgement.
    pub(crate) async fn flush_until(&self, deadline: Instant) -> Result<FlushAck, WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::Flush { reply });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }

    #[cfg(test)]
    pub(crate) async fn pause_until(
        &self,
        deadline: Instant,
        started: std::sync::mpsc::SyncSender<()>,
        release: std::sync::mpsc::Receiver<()>,
    ) -> Result<(), WriterFailure> {
        let permit = self.reserve_accepted_until(deadline).await?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::Pause {
            started,
            release,
            reply,
        });
        await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await
    }
}

/// One already-accounted writer slot. Dropping it releases capacity.
pub(crate) struct WriterCommandPermit {
    permit: mpsc::OwnedPermit<WriterEnvelope>,
    abort_epoch: u64,
}

pub(crate) struct ActiveDispatchTerminalPermit {
    permit: Option<WriterCommandPermit>,
    lifecycle: watch::Receiver<WriterLifecycleState>,
}

impl ActiveDispatchTerminalPermit {
    pub(crate) async fn record_until(
        mut self,
        terminal: ActiveDispatchTerminal,
        deadline: Instant,
    ) -> Result<ActiveDispatchTerminalAck, WriterFailure> {
        let permit = self
            .permit
            .take()
            .ok_or_else(|| WriterFailure::new(WriterFailureClass::Protocol))?;
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordActiveDispatchTerminal { terminal, reply });
        await_reply(receiver, self.lifecycle, deadline).await
    }

    pub(crate) fn record_detached(mut self, terminal: ActiveDispatchTerminal) {
        let Some(permit) = self.permit.take() else {
            return;
        };
        let (reply, _receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordActiveDispatchTerminal { terminal, reply });
    }
}

impl WriterCommandPermit {
    pub(super) fn send(self, command: LedgerWriterCommand) {
        self.send_with_optional_transaction_start_deadline(command, None);
    }

    fn send_with_transaction_start_deadline(
        self,
        command: LedgerWriterCommand,
        transaction_start_deadline: Instant,
    ) {
        self.send_with_optional_transaction_start_deadline(
            command,
            Some(transaction_start_deadline),
        );
    }

    fn send_with_optional_transaction_start_deadline(
        self,
        command: LedgerWriterCommand,
        transaction_start_deadline: Option<Instant>,
    ) {
        drop(self.permit.send(WriterEnvelope {
            abort_epoch: self.abort_epoch,
            transaction_start_deadline,
            command,
        }));
    }
}

/// Sole lifecycle and join owner for one ledger writer thread.
pub(crate) struct LedgerWriterOwner {
    shared: Arc<WriterShared>,
    join: Option<JoinHandle<()>>,
}

struct WriterRepository<'a> {
    repository: LedgerRepository,
    shared: &'a WriterShared,
}

impl Deref for WriterRepository<'_> {
    type Target = LedgerRepository;

    fn deref(&self) -> &Self::Target {
        &self.repository
    }
}

impl DerefMut for WriterRepository<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.repository
    }
}

impl Drop for WriterRepository<'_> {
    fn drop(&mut self) {
        if std::thread::panicking()
            || self
                .shared
                .stop_process_on_abort
                .swap(false, Ordering::AcqRel)
        {
            self.repository.stop_abandoned_process();
        }
    }
}

struct WriterStartupRepository {
    repository: Option<LedgerRepository>,
}

impl WriterStartupRepository {
    fn new(repository: LedgerRepository) -> Self {
        Self {
            repository: Some(repository),
        }
    }

    fn take(mut self) -> LedgerRepository {
        self.repository
            .take()
            .expect("writer startup repository must exist")
    }

    fn connection_mut(&mut self) -> &mut rusqlite::Connection {
        self.repository
            .as_mut()
            .expect("writer startup repository must exist")
            .connection_mut()
    }
}

impl Drop for WriterStartupRepository {
    fn drop(&mut self) {
        if let Some(repository) = self.repository.as_mut() {
            repository.stop_abandoned_process();
        }
    }
}

impl LedgerWriterOwner {
    /// Move the exact activation repository into one named writer thread.
    pub(crate) fn start(
        repository: LedgerRepository,
        capacity: usize,
    ) -> Result<(Self, LedgerWriterClient), WriterFailure> {
        let mut startup_repository = WriterStartupRepository::new(repository);
        if capacity == 0 {
            return Err(WriterFailureClass::Protocol.into());
        }
        let interrupt = Arc::new(startup_repository.connection_mut().get_interrupt_handle());
        let (sender, receiver) = mpsc::channel(capacity);
        let (lifecycle, _) = watch::channel(WriterLifecycleState::Running);
        let shared = Arc::new(WriterShared {
            sender: Mutex::new(Some(sender)),
            lifecycle_gate: Mutex::new(()),
            lifecycle,
            abort_epoch: AtomicU64::new(0),
            accepting: AtomicBool::new(true),
            stop_process_on_abort: AtomicBool::new(false),
            interrupt,
        });
        let worker_shared = shared.clone();
        let join = std::thread::Builder::new()
            .name("nemo-relay-router-ledger".to_string())
            .spawn(move || {
                let repository = startup_repository.take();
                let outcome = catch_unwind(AssertUnwindSafe(|| {
                    run_writer(repository, receiver, &worker_shared)
                }));
                let _lifecycle = worker_shared
                    .lifecycle_gate
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                let exit = match outcome {
                    Ok(())
                        if matches!(
                            *worker_shared.lifecycle.borrow(),
                            WriterLifecycleState::Aborted
                        ) =>
                    {
                        WriterExitClass::Aborted
                    }
                    Ok(()) => WriterExitClass::Clean,
                    Err(_) => WriterExitClass::Panicked,
                };
                worker_shared
                    .lifecycle
                    .send_replace(WriterLifecycleState::Exited(exit));
            })
            .map_err(|_| WriterFailure::new(WriterFailureClass::Protocol))?;
        let client = LedgerWriterClient {
            shared: shared.clone(),
        };
        Ok((
            Self {
                shared,
                join: Some(join),
            },
            client,
        ))
    }

    /// Close admission, drain queued and already-permitted commands, and join.
    pub(crate) async fn drain_until(
        &mut self,
        deadline: Instant,
    ) -> Result<DrainAck, WriterFailure> {
        let (sender, abort_epoch) = {
            let _lifecycle = self
                .shared
                .lifecycle_gate
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if !self.shared.accepting.swap(false, Ordering::AcqRel) {
                if self.shared.abort_epoch.load(Ordering::Acquire) != 0 {
                    return Err(WriterFailureClass::Aborted.into());
                }
                return Err(failure_for_state(*self.shared.lifecycle.borrow()));
            }
            self.shared
                .lifecycle
                .send_replace(WriterLifecycleState::Draining);
            let sender = self
                .shared
                .sender
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_ref()
                .cloned()
                .ok_or_else(|| WriterFailure::new(WriterFailureClass::Closing))?;
            (sender, self.shared.abort_epoch.load(Ordering::Acquire))
        };
        let tokio_deadline = tokio::time::Instant::from_std(deadline);
        let permit = tokio::time::timeout_at(tokio_deadline, sender.reserve_owned())
            .await
            .map_err(|_| WriterFailure::new(WriterFailureClass::Deadline))?
            .map_err(|_| failure_for_state(*self.shared.lifecycle.borrow()))?;
        let (reply, receiver) = oneshot::channel();
        drop(permit.send(WriterEnvelope {
            abort_epoch,
            transaction_start_deadline: None,
            command: LedgerWriterCommand::Drain { reply },
        }));
        self.shared
            .sender
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .take();

        let acknowledgement =
            await_reply(receiver, self.shared.lifecycle.subscribe(), deadline).await?;
        wait_for_exit(self.shared.lifecycle.subscribe(), deadline).await?;
        self.join_exited()?;
        Ok(acknowledgement)
    }

    /// Synchronously invalidate future commands and interrupt SQLite.
    pub(crate) fn abort(&self) {
        self.shared.abort();
    }

    /// Mark failed activation so the writer terminalizes its process while aborting.
    pub(crate) fn request_process_stop_on_abort(&self) {
        self.shared
            .stop_process_on_abort
            .store(true, Ordering::Release);
    }

    fn join_exited(&mut self) -> Result<(), WriterFailure> {
        if let Some(join) = self.join.take() {
            join.join()
                .map_err(|_| WriterFailure::new(WriterFailureClass::Panicked))?;
        }
        match *self.shared.lifecycle.borrow() {
            WriterLifecycleState::Exited(WriterExitClass::Clean) => Ok(()),
            state => Err(failure_for_state(state)),
        }
    }

    #[cfg(test)]
    fn lifecycle(&self) -> WriterLifecycleState {
        *self.shared.lifecycle.borrow()
    }
}

impl Drop for LedgerWriterOwner {
    fn drop(&mut self) {
        if self.join.is_some() {
            self.shared.abort();
        }
    }
}

fn run_writer(
    repository: LedgerRepository,
    mut receiver: mpsc::Receiver<WriterEnvelope>,
    shared: &WriterShared,
) {
    let mut repository = WriterRepository { repository, shared };
    enum NextEvent {
        Envelope(Box<WriterEnvelope>),
        ChannelClosed,
        LifecycleChanged,
    }

    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("ledger writer wake runtime must initialize");
    let mut lifecycle = shared.lifecycle.subscribe();
    let mut drain_reply: Option<oneshot::Sender<Result<DrainAck, WriterFailure>>> = None;
    loop {
        if matches!(
            *lifecycle.borrow_and_update(),
            WriterLifecycleState::Aborted | WriterLifecycleState::Exited(_)
        ) {
            fail_pending_after_abort(&mut receiver);
            if let Some(reply) = drain_reply {
                let _ = reply.send(Err(WriterFailureClass::Aborted.into()));
            }
            return;
        }
        let envelope = loop {
            let next = runtime.block_on(async {
                tokio::select! {
                    biased;
                    changed = lifecycle.changed() => {
                        let _ = changed;
                        NextEvent::LifecycleChanged
                    }
                    envelope = receiver.recv() => match envelope {
                        Some(envelope) => NextEvent::Envelope(Box::new(envelope)),
                        None => NextEvent::ChannelClosed,
                    }
                }
            });
            match next {
                NextEvent::Envelope(envelope) => break *envelope,
                NextEvent::ChannelClosed => {
                    if let Some(reply) = drain_reply {
                        let _ = reply.send(Ok(DrainAck));
                    }
                    return;
                }
                NextEvent::LifecycleChanged
                    if matches!(
                        *lifecycle.borrow_and_update(),
                        WriterLifecycleState::Aborted | WriterLifecycleState::Exited(_)
                    ) =>
                {
                    fail_pending_after_abort(&mut receiver);
                    if let Some(reply) = drain_reply {
                        let _ = reply.send(Err(WriterFailureClass::Aborted.into()));
                    }
                    return;
                }
                NextEvent::LifecycleChanged => {
                    lifecycle.borrow_and_update();
                }
            }
        };
        if !transaction_epoch_is_current(shared, envelope.abort_epoch) {
            envelope.command.fail(WriterFailureClass::Aborted.into());
            fail_pending_after_abort(&mut receiver);
            return;
        }
        let abort_epoch = envelope.abort_epoch;
        let transaction_start_deadline = envelope.transaction_start_deadline;
        match envelope.command {
            LedgerWriterCommand::ProbePendingAnchor { pending, reply } => {
                let result = repository
                    .probe_pending_anchor_with_start_check(&pending, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_anchor_probe);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::RecordPendingAnchor { pending, reply } => {
                let result = repository
                    .record_pending_anchor_with_start_check(&pending, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_anchor_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::RecordPendingAnchorWithCapacity {
                pending,
                max_evidence_records,
                reply,
            } => {
                let result = repository
                    .record_pending_anchor_with_capacity_and_start_check(
                        &pending,
                        max_evidence_records,
                        || retain_transaction_start_guard(shared, abort_epoch),
                    )
                    .map_err(WriterFailure::from)
                    .and_then(map_pending_anchor_capacity_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::RecordNotScheduledQueueFull {
                decline,
                max_evidence_records,
                reply,
            } => {
                let result = match max_evidence_records {
                    Some(max_evidence_records) => repository
                        .record_not_scheduled_queue_full_with_capacity_and_start_check(
                            &decline,
                            max_evidence_records,
                            || retain_transaction_start_guard(shared, abort_epoch),
                        ),
                    None => repository
                        .record_not_scheduled_queue_full_with_start_check(&decline, || {
                            retain_transaction_start_guard(shared, abort_epoch)
                        }),
                }
                .map_err(WriterFailure::from)
                .and_then(map_not_scheduled_queue_full_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::RecordTerminalAnchor { terminal, reply } => {
                let result = repository
                    .record_terminal_anchor_with_start_check(&terminal, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_anchor_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::ReserveSampleBatch { reservation, reply } => {
                let result = repository
                    .reserve_sample_batch_with_start_check(&reservation, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_shadow_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::StartShadowAttempt { command, reply } => {
                let result = repository
                    .start_shadow_attempt_with_start_check(command, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_shadow_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::RecordShadowTerminal { command, reply } => {
                let result = repository
                    .record_shadow_terminal_with_start_check(&command, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_shadow_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::RecordJudgeAttemptStart { start, reply } => {
                let result = repository
                    .record_judge_attempt_start_with_start_check(&start, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_judge_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::RecordJudgeAttemptTerminal { terminal, reply } => {
                let result = repository
                    .record_judge_attempt_terminal_with_start_check(&terminal, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_judge_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::RecordEvaluation { evaluation, reply } => {
                let result = repository
                    .record_evaluation_with_start_check(&evaluation, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_judge_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::RecordDecisionAudit {
                audit,
                max_evidence_records,
                conflict_health_event_id,
                reply,
            } => {
                let result = repository
                    .record_decision_audit_with_start_check(
                        audit.as_ref(),
                        max_evidence_records,
                        conflict_health_event_id,
                        || {
                            retain_transaction_start_guard_with_deadline(
                                shared,
                                abort_epoch,
                                transaction_start_deadline,
                            )
                        },
                    )
                    .map_err(WriterFailure::from)
                    .and_then(map_decision_audit_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::ClaimCandidateDependency {
                identity,
                operation,
                reply,
            } => {
                let result = repository
                    .claim_candidate_dependency_with_start_check(&identity, &operation, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_dependency_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::ClaimJudgeDependency {
                identity,
                operation,
                reply,
            } => {
                let result = repository
                    .claim_judge_dependency_with_start_check(&identity, &operation, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_dependency_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::CompleteDependency { completion, reply } => {
                let result = repository
                    .complete_dependency_with_start_check(&completion, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_dependency_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::PrepareLiveEmbedding { command, reply } => {
                let result = repository
                    .prepare_live_embedding_with_start_check(&command, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_live_embedding_prepare_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::CreateEmbeddingJob { command, reply } => {
                let result = repository
                    .create_embedding_job_with_start_check(&command, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_embedding_create_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::ClaimEmbeddingJobBatch { command, reply } => {
                let result = repository
                    .claim_embedding_job_batch_with_start_check(&command, || {
                        retain_transaction_start_guard_with_deadline(
                            shared,
                            abort_epoch,
                            transaction_start_deadline,
                        )
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_embedding_batch_claim_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::CompleteEmbeddingJobBatch { command, reply } => {
                let result = repository
                    .complete_embedding_job_batch_with_start_check(&command, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_embedding_batch_completion_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::ResolveEmbeddingJob { command, reply } => {
                let result = repository
                    .resolve_embedding_job_with_start_check(&command, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_embedding_resolution_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::ResetEmbeddingJob { command, reply } => {
                let result = repository
                    .reset_embedding_job_with_start_check(&command, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_embedding_reset_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::ClaimMaterialization { command, reply } => {
                let result = repository
                    .claim_materialization_with_start_check(&command, || {
                        retain_transaction_start_guard_with_deadline(
                            shared,
                            abort_epoch,
                            transaction_start_deadline,
                        )
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_materialization_claim_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::CompleteMaterialization { command, reply } => {
                let result = repository
                    .complete_materialization_with_start_check(&command, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_materialization_completion_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::ResolveMaterialization { command, reply } => {
                let result = repository
                    .resolve_materialization_with_start_check(&command, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_materialization_resolution_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::PropagateMaterializationFailure { command, reply } => {
                let result = repository
                    .propagate_materialization_failure_with_start_check(&command, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_materialization_failure_propagation_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::BackfillVectorGraph { command, reply } => {
                let result = repository
                    .backfill_vector_graph_with_start_check(&command, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_vector_backfill_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::VectorIndex { command, reply } => {
                let result = repository
                    .execute_vector_index_command_with_start_check(&command, || {
                        retain_transaction_start_guard_with_deadline(
                            shared,
                            abort_epoch,
                            transaction_start_deadline,
                        )
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_vector_index_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::EnsureVectorRegistry { ensure, reply } => {
                let result = repository
                    .ensure_vector_registry_with_start_check(&ensure, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_registry_ensure_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::CreateActiveExperiment { create, reply } => {
                let result = repository
                    .create_active_experiment_with_start_check(&create, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_active_experiment_create_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::ClaimActiveLook { request, reply } => {
                let result = repository
                    .claim_next_active_look_with_start_check(&request, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_active_look_claim_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::RenewActiveLookLease { renewal, reply } => {
                let result = repository
                    .renew_active_look_lease_with_start_check(&renewal, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_active_look_lease_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::CommitActiveLook { commit, reply } => {
                let result = repository
                    .commit_active_look_with_start_check(&commit, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_active_look_commit_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::FailActiveLook { failure, reply } => {
                let result = repository
                    .fail_active_look_with_start_check(&failure, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_active_look_failure_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::ObserveActiveAuthorization {
                active_experiment_id,
                observed_at_unix_ms,
                reply,
            } => {
                let result = repository
                    .observe_active_authorization_with_start_check(
                        active_experiment_id,
                        observed_at_unix_ms,
                        || retain_transaction_start_guard(shared, abort_epoch),
                    )
                    .map_err(WriterFailure::from)
                    .and_then(map_active_authorization_observe_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::PromoteActiveNeighborhood { promotion, reply } => {
                let result = repository
                    .promote_active_neighborhood_with_start_check(&promotion, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_active_neighborhood_mutation_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::InvalidateActiveNeighborhood {
                invalidation,
                reply,
            } => {
                let result = repository
                    .invalidate_active_neighborhood_with_start_check(&invalidation, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_active_neighborhood_mutation_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::ObserveActiveNeighborhood {
                key,
                observed_at_unix_ms,
                reply,
            } => {
                let result = repository
                    .observe_active_neighborhood_with_start_check(&key, observed_at_unix_ms, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_active_neighborhood_observe_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::AdmitActiveRoot { admission, reply } => {
                let result = repository
                    .admit_active_root_with_start_check(&admission, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_active_admission_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::AdmitActiveDecision { admission, reply } => {
                let result = repository
                    .admit_active_decision_with_start_check(&admission, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_active_decision_admission_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::AppendActiveSignals { batch, reply } => {
                let result = repository
                    .append_active_signals_with_start_check(&batch, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_active_signal_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::RecordActiveDispatchTerminal { terminal, reply } => {
                let result = repository
                    .record_active_dispatch_terminal_with_start_check(&terminal, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_active_dispatch_terminal_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::TerminalizeActiveRoot { terminal, reply } => {
                let result = repository
                    .terminalize_active_root_with_start_check(&terminal, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_active_root_terminal_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::RunRetention { request, reply } => {
                let result = repository
                    .run_retention_with_start_check(&request, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_retention_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::RenewHeartbeat { renewal, reply } => {
                let result = repository
                    .renew_heartbeat_with_start_check(renewal, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_heartbeat_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::StopProcess { command, reply } => {
                let result = repository
                    .stop_process_with_start_check(command, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_process_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::AppendHealthEvent { event, reply } => {
                let result = repository
                    .append_health_event_with_start_check(&event, || {
                        retain_transaction_start_guard(shared, abort_epoch)
                    })
                    .map_err(WriterFailure::from)
                    .and_then(map_process_ack);
                let _ = reply.send(result);
            }
            LedgerWriterCommand::ApplyControlMutation {
                mutation,
                fence,
                reply,
            } => {
                let result =
                    repository.apply_control_mutation_with_start_check(&mutation, &fence, || {
                        retain_transaction_start_guard_with_deadline(
                            shared,
                            abort_epoch,
                            transaction_start_deadline,
                        )
                    });
                let _ = reply.send(Ok(result));
            }
            LedgerWriterCommand::ApplyOperatorMutation {
                mutation,
                fence,
                reply,
            } => {
                let result =
                    repository.apply_operator_mutation_with_start_check(&mutation, &fence, || {
                        retain_transaction_start_guard_with_deadline(
                            shared,
                            abort_epoch,
                            transaction_start_deadline,
                        )
                    });
                let _ = reply.send(Ok(result));
            }
            LedgerWriterCommand::Flush { reply } => {
                let _ = reply.send(Ok(FlushAck));
            }
            LedgerWriterCommand::Drain { reply } => {
                if drain_reply.is_some() {
                    let _ = reply.send(Err(WriterFailureClass::Protocol.into()));
                } else {
                    drain_reply = Some(reply);
                    receiver.close();
                }
            }
            #[cfg(test)]
            LedgerWriterCommand::Pause {
                started,
                release,
                reply,
            } => {
                let _ = started.send(());
                let result = release
                    .recv()
                    .map_err(|_| WriterFailure::new(WriterFailureClass::Protocol));
                let _ = reply.send(result);
            }
            #[cfg(test)]
            LedgerWriterCommand::Panic { reply } => {
                drop(reply);
                panic!("injected ledger writer panic");
            }
        }
        if matches!(*shared.lifecycle.borrow(), WriterLifecycleState::Aborted) {
            fail_pending_after_abort(&mut receiver);
            if let Some(reply) = drain_reply {
                let _ = reply.send(Err(WriterFailureClass::Aborted.into()));
            }
            return;
        }
    }
}

fn fail_pending_after_abort(receiver: &mut mpsc::Receiver<WriterEnvelope>) {
    receiver.close();
    while let Ok(envelope) = receiver.try_recv() {
        envelope.command.fail(WriterFailureClass::Aborted.into());
    }
}

fn transaction_epoch_is_current(shared: &WriterShared, abort_epoch: u64) -> bool {
    shared.abort_epoch.load(Ordering::Acquire) == abort_epoch
        && !matches!(
            *shared.lifecycle.borrow(),
            WriterLifecycleState::Aborted | WriterLifecycleState::Exited(_)
        )
}

struct TransactionStartAuthority<'a> {
    shared: &'a WriterShared,
    abort_epoch: u64,
    start_deadline: Option<Instant>,
}

impl TransactionStartGuard for TransactionStartAuthority<'_> {
    fn permits_transaction(&self) -> bool {
        transaction_start_is_permitted(self.shared, self.abort_epoch, self.start_deadline)
    }
}

fn retain_transaction_start_guard<'a>(
    shared: &'a WriterShared,
    abort_epoch: u64,
) -> Option<TransactionStartAuthority<'a>> {
    retain_transaction_start_guard_with_deadline(shared, abort_epoch, None)
}

fn retain_transaction_start_guard_with_deadline<'a>(
    shared: &'a WriterShared,
    abort_epoch: u64,
    start_deadline: Option<Instant>,
) -> Option<TransactionStartAuthority<'a>> {
    if transaction_start_is_permitted(shared, abort_epoch, start_deadline) {
        Some(TransactionStartAuthority {
            shared,
            abort_epoch,
            start_deadline,
        })
    } else {
        None
    }
}

fn transaction_start_is_permitted(
    shared: &WriterShared,
    abort_epoch: u64,
    start_deadline: Option<Instant>,
) -> bool {
    shared.abort_epoch.load(Ordering::Acquire) == abort_epoch
        && !matches!(
            *shared.lifecycle.borrow(),
            WriterLifecycleState::Aborted | WriterLifecycleState::Exited(_)
        )
        && start_deadline.is_none_or(|deadline| Instant::now() < deadline)
}

fn map_dependency_ack(
    acknowledgement: DependencyCommandAck,
) -> Result<DependencyCommandAck, WriterFailure> {
    match acknowledgement {
        DependencyCommandAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_decision_audit_ack(
    acknowledgement: DecisionAuditAck,
) -> Result<DecisionAuditAck, WriterFailure> {
    match acknowledgement {
        DecisionAuditAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_vector_index_ack(
    acknowledgement: VectorIndexTransactionAck,
) -> Result<VectorIndexWriterAck, WriterFailure> {
    match acknowledgement {
        VectorIndexTransactionAck::Completed(acknowledgement) => Ok(acknowledgement),
        VectorIndexTransactionAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
    }
}

fn map_registry_ensure_ack(
    acknowledgement: RegistryEnsureTransactionAck,
) -> Result<RegistryEnsureAck, WriterFailure> {
    match acknowledgement {
        RegistryEnsureTransactionAck::Completed(acknowledgement) => Ok(acknowledgement),
        RegistryEnsureTransactionAck::TransactionNotStarted => {
            Err(WriterFailureClass::Aborted.into())
        }
    }
}

fn map_active_experiment_create_ack(
    acknowledgement: ActiveExperimentCreateAck,
) -> Result<ActiveExperimentCreateAck, WriterFailure> {
    match acknowledgement {
        ActiveExperimentCreateAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_active_look_claim_ack(
    acknowledgement: ActiveLookClaimAck,
) -> Result<ActiveLookClaimAck, WriterFailure> {
    match acknowledgement {
        ActiveLookClaimAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_active_look_lease_ack(
    acknowledgement: ActiveLookLeaseAck,
) -> Result<ActiveLookLeaseAck, WriterFailure> {
    match acknowledgement {
        ActiveLookLeaseAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_active_look_commit_ack(
    acknowledgement: ActiveLookCommitAck,
) -> Result<ActiveLookCommitAck, WriterFailure> {
    match acknowledgement {
        ActiveLookCommitAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_active_look_failure_ack(
    acknowledgement: ActiveLookFailureAck,
) -> Result<ActiveLookFailureAck, WriterFailure> {
    match acknowledgement {
        ActiveLookFailureAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_active_authorization_observe_ack(
    acknowledgement: ActiveAuthorizationObserveAck,
) -> Result<ActiveAuthorizationObserveAck, WriterFailure> {
    match acknowledgement {
        ActiveAuthorizationObserveAck::TransactionNotStarted => {
            Err(WriterFailureClass::Aborted.into())
        }
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_active_neighborhood_mutation_ack(
    acknowledgement: ActiveNeighborhoodMutationAck,
) -> Result<ActiveNeighborhoodMutationAck, WriterFailure> {
    match acknowledgement {
        ActiveNeighborhoodMutationAck::TransactionNotStarted => {
            Err(WriterFailureClass::Aborted.into())
        }
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_active_neighborhood_observe_ack(
    acknowledgement: ActiveNeighborhoodObserveAck,
) -> Result<ActiveNeighborhoodObserveAck, WriterFailure> {
    match acknowledgement {
        ActiveNeighborhoodObserveAck::TransactionNotStarted => {
            Err(WriterFailureClass::Aborted.into())
        }
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_active_admission_ack(
    acknowledgement: ActiveAdmissionAck,
) -> Result<ActiveAdmissionAck, WriterFailure> {
    match acknowledgement {
        ActiveAdmissionAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_active_decision_admission_ack(
    acknowledgement: ActiveDecisionAdmissionAckV2,
) -> Result<ActiveDecisionAdmissionAckV2, WriterFailure> {
    match acknowledgement {
        ActiveDecisionAdmissionAckV2::TransactionNotStarted => {
            Err(WriterFailureClass::Aborted.into())
        }
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_active_signal_ack(
    acknowledgement: ActiveSignalBatchAck,
) -> Result<ActiveSignalBatchAck, WriterFailure> {
    match acknowledgement {
        ActiveSignalBatchAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_active_dispatch_terminal_ack(
    acknowledgement: ActiveDispatchTerminalAck,
) -> Result<ActiveDispatchTerminalAck, WriterFailure> {
    match acknowledgement {
        ActiveDispatchTerminalAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_active_root_terminal_ack(
    acknowledgement: ActiveRootTerminalAck,
) -> Result<ActiveRootTerminalAck, WriterFailure> {
    match acknowledgement {
        ActiveRootTerminalAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_embedding_create_ack(
    acknowledgement: EmbeddingJobCreateAck,
) -> Result<EmbeddingJobCreateAck, WriterFailure> {
    match acknowledgement {
        EmbeddingJobCreateAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_live_embedding_prepare_ack(
    acknowledgement: LiveEmbeddingPrepareAck,
) -> Result<LiveEmbeddingPrepareAck, WriterFailure> {
    match acknowledgement {
        LiveEmbeddingPrepareAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_embedding_batch_claim_ack(
    acknowledgement: EmbeddingJobBatchClaimAck,
) -> Result<EmbeddingJobBatchClaimAck, WriterFailure> {
    match acknowledgement {
        EmbeddingJobBatchClaimAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_embedding_batch_completion_ack(
    acknowledgement: EmbeddingJobBatchCompletionAck,
) -> Result<EmbeddingJobBatchCompletionAck, WriterFailure> {
    match acknowledgement {
        EmbeddingJobBatchCompletionAck::TransactionNotStarted => {
            Err(WriterFailureClass::Aborted.into())
        }
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_embedding_resolution_ack(
    acknowledgement: EmbeddingJobResolutionAck,
) -> Result<EmbeddingJobResolutionAck, WriterFailure> {
    match acknowledgement {
        EmbeddingJobResolutionAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_embedding_reset_ack(
    acknowledgement: EmbeddingJobResetAck,
) -> Result<EmbeddingJobResetAck, WriterFailure> {
    match acknowledgement {
        EmbeddingJobResetAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_materialization_claim_ack(
    acknowledgement: MaterializationClaimAck,
) -> Result<MaterializationClaimAck, WriterFailure> {
    match acknowledgement {
        MaterializationClaimAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_materialization_completion_ack(
    acknowledgement: MaterializationCompletionAck,
) -> Result<MaterializationCompletionAck, WriterFailure> {
    match acknowledgement {
        MaterializationCompletionAck::TransactionNotStarted => {
            Err(WriterFailureClass::Aborted.into())
        }
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_materialization_resolution_ack(
    acknowledgement: MaterializationResolutionAck,
) -> Result<MaterializationResolutionAck, WriterFailure> {
    match acknowledgement {
        MaterializationResolutionAck::TransactionNotStarted => {
            Err(WriterFailureClass::Aborted.into())
        }
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_materialization_failure_propagation_ack(
    acknowledgement: MaterializationFailurePropagationAck,
) -> Result<MaterializationFailurePropagationAck, WriterFailure> {
    match acknowledgement {
        MaterializationFailurePropagationAck::TransactionNotStarted => {
            Err(WriterFailureClass::Aborted.into())
        }
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_vector_backfill_ack(
    acknowledgement: VectorBackfillAck,
) -> Result<VectorBackfillAck, WriterFailure> {
    match acknowledgement {
        VectorBackfillAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_retention_ack(acknowledgement: RetentionAck) -> Result<RetentionAck, WriterFailure> {
    match acknowledgement {
        RetentionAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_anchor_probe(acknowledgement: AnchorProbe) -> Result<AnchorProbe, WriterFailure> {
    match acknowledgement {
        AnchorProbe::TransactionNotStarted { .. } => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_anchor_ack(acknowledgement: AnchorCommandAck) -> Result<AnchorCommandAck, WriterFailure> {
    match acknowledgement {
        AnchorCommandAck::TransactionNotStarted { .. } => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_pending_anchor_capacity_ack(
    acknowledgement: PendingAnchorCapacityAck,
) -> Result<PendingAnchorCapacityAck, WriterFailure> {
    match acknowledgement {
        PendingAnchorCapacityAck::TransactionNotStarted { .. } => {
            Err(WriterFailureClass::Aborted.into())
        }
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_not_scheduled_queue_full_ack(
    acknowledgement: NotScheduledQueueFullAck,
) -> Result<NotScheduledQueueFullAck, WriterFailure> {
    match acknowledgement {
        NotScheduledQueueFullAck::TransactionNotStarted { .. } => {
            Err(WriterFailureClass::Aborted.into())
        }
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_shadow_ack(acknowledgement: ShadowCommandAck) -> Result<ShadowCommandAck, WriterFailure> {
    match acknowledgement {
        ShadowCommandAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_judge_ack(acknowledgement: JudgeRecordAck) -> Result<JudgeRecordAck, WriterFailure> {
    match acknowledgement {
        JudgeRecordAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_heartbeat_ack(acknowledgement: HeartbeatAck) -> Result<HeartbeatAck, WriterFailure> {
    match acknowledgement {
        HeartbeatAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

fn map_process_ack(acknowledgement: ProcessCommandAck) -> Result<ProcessCommandAck, WriterFailure> {
    match acknowledgement {
        ProcessCommandAck::TransactionNotStarted => Err(WriterFailureClass::Aborted.into()),
        acknowledgement => Ok(acknowledgement),
    }
}

async fn await_reply<T>(
    mut receiver: oneshot::Receiver<Result<T, WriterFailure>>,
    mut lifecycle: watch::Receiver<WriterLifecycleState>,
    deadline: Instant,
) -> Result<T, WriterFailure> {
    let tokio_deadline = tokio::time::Instant::from_std(deadline);
    loop {
        if Instant::now() >= deadline {
            return Err(WriterFailureClass::Deadline.into());
        }
        tokio::select! {
            biased;
            () = tokio::time::sleep_until(tokio_deadline) => {
                return Err(WriterFailureClass::Deadline.into());
            }
            result = &mut receiver => {
                if Instant::now() >= deadline {
                    return Err(WriterFailureClass::Deadline.into());
                }
                match result {
                    Ok(result) => return result,
                    Err(_) => return Err(
                        wait_for_terminal_failure(lifecycle, deadline).await
                    ),
                }
            }
            changed = lifecycle.changed() => {
                if changed.is_err() || lifecycle_is_terminal(*lifecycle.borrow()) {
                    return Err(failure_for_state(*lifecycle.borrow()));
                }
            }
        }
    }
}

async fn await_reply_while_running<T>(
    receiver: oneshot::Receiver<Result<T, WriterFailure>>,
    lifecycle: watch::Receiver<WriterLifecycleState>,
) -> Result<T, WriterFailure> {
    match receiver.await {
        Ok(result) => result,
        Err(_) => Err(wait_for_terminal_failure_while_running(lifecycle).await),
    }
}

async fn wait_for_terminal_failure_while_running(
    mut lifecycle: watch::Receiver<WriterLifecycleState>,
) -> WriterFailure {
    loop {
        if lifecycle_is_terminal(*lifecycle.borrow()) {
            return failure_for_state(*lifecycle.borrow());
        }
        if lifecycle.changed().await.is_err() {
            return WriterFailureClass::Exited.into();
        }
    }
}

async fn wait_for_terminal_failure(
    mut lifecycle: watch::Receiver<WriterLifecycleState>,
    deadline: Instant,
) -> WriterFailure {
    let tokio_deadline = tokio::time::Instant::from_std(deadline);
    loop {
        if Instant::now() >= deadline {
            return WriterFailureClass::Deadline.into();
        }
        if lifecycle_is_terminal(*lifecycle.borrow()) {
            return failure_for_state(*lifecycle.borrow());
        }
        tokio::select! {
            biased;
            () = tokio::time::sleep_until(tokio_deadline) => {
                return WriterFailureClass::Deadline.into();
            }
            changed = lifecycle.changed() => {
                if Instant::now() >= deadline {
                    return WriterFailureClass::Deadline.into();
                }
                if changed.is_err() {
                    return WriterFailureClass::Exited.into();
                }
            }
        }
    }
}

async fn wait_for_exit(
    mut lifecycle: watch::Receiver<WriterLifecycleState>,
    deadline: Instant,
) -> Result<(), WriterFailure> {
    let tokio_deadline = tokio::time::Instant::from_std(deadline);
    loop {
        if Instant::now() >= deadline {
            return Err(WriterFailureClass::Deadline.into());
        }
        if matches!(*lifecycle.borrow(), WriterLifecycleState::Exited(_)) {
            return Ok(());
        }
        tokio::select! {
            biased;
            () = tokio::time::sleep_until(tokio_deadline) => {
                return Err(WriterFailureClass::Deadline.into());
            }
            changed = lifecycle.changed() => {
                if Instant::now() >= deadline {
                    return Err(WriterFailureClass::Deadline.into());
                }
                if changed.is_err() {
                    return Err(WriterFailureClass::Exited.into());
                }
            }
        }
    }
}

fn lifecycle_is_terminal(state: WriterLifecycleState) -> bool {
    matches!(
        state,
        WriterLifecycleState::Aborted | WriterLifecycleState::Exited(_)
    )
}

fn failure_for_state(state: WriterLifecycleState) -> WriterFailure {
    match state {
        WriterLifecycleState::Running => WriterFailureClass::Exited.into(),
        WriterLifecycleState::Draining => WriterFailureClass::Closing.into(),
        WriterLifecycleState::Aborted | WriterLifecycleState::Exited(WriterExitClass::Aborted) => {
            WriterFailureClass::Aborted.into()
        }
        WriterLifecycleState::Exited(WriterExitClass::Panicked) => {
            WriterFailureClass::Panicked.into()
        }
        WriterLifecycleState::Exited(WriterExitClass::Clean) => WriterFailureClass::Exited.into(),
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};

    use chrono::{TimeZone, Utc};
    use nemo_relay::api::llm::LlmApiFamily;
    use nemo_relay::api::runtime::{LLM_REPLAY_CONTRACT_VERSION, LlmReplayCapability};
    use rusqlite::params;
    use serde_json::json;
    use tempfile::tempdir;
    use tokio::sync::oneshot;
    use uuid::Uuid;

    use super::{
        LedgerWriterCommand, LedgerWriterOwner, WriterFailureClass, map_decision_audit_ack,
        map_embedding_batch_claim_ack, map_embedding_batch_completion_ack, map_embedding_reset_ack,
        map_embedding_resolution_ack, map_live_embedding_prepare_ack,
        map_materialization_claim_ack, map_materialization_completion_ack,
        map_materialization_failure_propagation_ack, map_materialization_resolution_ack,
        map_vector_backfill_ack,
    };
    use crate::canonical_json::{canonical_json, canonical_sha256};
    use crate::canonical_query::{
        CanonicalRoutingQueryArtifactV1, CanonicalRoutingQueryV1, CanonicalTaskV1,
    };
    use crate::config::{
        JUDGE_OUTPUT_SCHEMA_SHA256_V1, JUDGE_PROMPT_TEMPLATE_SHA256_V1,
        JUDGE_RUBRIC_TEMPLATE_SHA256_V1, RouterConfig,
    };
    use crate::embedding_identity::CANONICAL_ROUTING_QUERY_SCHEMA_V1;
    use crate::fingerprint::sha256_hex;
    use crate::judge::{
        DeterministicHardFailureV1, JudgeHorizonV1, JudgePolicyIdentityV1, PairwiseJudgeInputV1,
    };
    use crate::ledger::model::LedgerRuntimeIdentity;
    use crate::ledger::repository::anchors::{
        AnchorCommandAck, AnchorProbe, FrozenPendingAnchorV1, FrozenTerminalAnchorV1,
        NotScheduledQueueFull, NotScheduledQueueFullAck, PendingAnchorCapacityAck,
    };
    use crate::ledger::repository::cooloff::{
        CandidateDependencyIdentity, DependencyCommandAck, DependencyCompletion,
        DependencyOperation, DependencyTransition, JudgeDependencyIdentity,
    };
    use crate::ledger::repository::decision::DecisionAuditAck;
    use crate::ledger::repository::embedding::{
        EmbeddingJobBatchClaim, EmbeddingJobBatchClaimAck, EmbeddingJobBatchClaimItem,
        EmbeddingJobBatchCompletion, EmbeddingJobBatchCompletionAck,
        EmbeddingJobBatchCompletionItem, EmbeddingJobResetAck, EmbeddingJobResolutionAck,
        LiveEmbeddingPrepare, LiveEmbeddingPrepareAck,
    };
    use crate::ledger::repository::judge::{
        EvaluationRecord, JudgeAttemptStart, JudgeAttemptTerminal, JudgeRecordAck,
        JudgeTransportFailureClass,
    };
    use crate::ledger::repository::materialization::{
        MaterializationClaim, MaterializationClaimAck, MaterializationCompletion,
        MaterializationCompletionAck, MaterializationCompletionState,
        MaterializationFailurePropagationAck, MaterializationResolutionAck, VectorBackfillAck,
    };
    use crate::ledger::repository::process::{
        HeartbeatAck, HeartbeatRenewal, LedgerHealthEvent, LedgerHealthSeverity, ProcessCommandAck,
        ProcessStop,
    };
    use crate::ledger::repository::retention::{RetentionAck, RetentionRequest};
    use crate::ledger::repository::shadow::{
        AtomicShadowVectorization, ReservedShadowAttempt, SampleBatchReservation,
        SampleBatchTerminalEvent, SampleBatchTerminalState, ShadowAttemptStarted, ShadowCommandAck,
        ShadowOperationalFailureClass, ShadowTerminalClass, ShadowTerminalRecord,
        ShadowVectorSourceV1, ShadowVectorizationHandoff,
    };
    use crate::ledger::repository::vector_index::{
        GenerationAuthorizationAck, GenerationObjectCreationAck, RebuildFlipAck,
        RebuildLeaseClaimAck, RebuildStepAck, authorize_generation, catch_up_rebuild_changes,
        claim_rebuild_lease, create_generation_objects, flip_rebuild_generation,
        populate_rebuild_chunk,
    };
    use crate::ledger::repository::vector_registry::FrozenMappingKey;
    use crate::ledger::repository::{ActivatedLedger, LedgerRepository};
    use crate::projection::{
        REQUEST_PROJECTION_SCHEMA_V1, ROUTER_SANITIZER_VERSION, RouterRequestProjectionV1,
        SanitizedAnnotatedLlmRequest, SanitizedMessage, SanitizedMessageContent,
    };
    use crate::trajectory::test_fixtures::pending_window;
    use crate::trajectory::{
        CANDIDATE_FACT_SCHEMA_V1, PendingTrajectoryWindow, PersistedCandidateCapabilitiesV1,
        PersistedCandidateFactV1, PersistedTrajectoryTerminalV1, ReplayCapabilityFactsV1,
        RouterResponseProjectionV1, TrajectoryTrigger,
    };
    use crate::vector::{AuthoritativeVector, NormalizedVector, VectorDimensions, VectorSpaceId};

    const CREATED_AT: i64 = 1_700_000_001_000;

    fn activate_empty_vector_generation(
        activated: &mut ActivatedLedger,
        vector_space_id: &VectorSpaceId,
        observed_at_unix_ms: i64,
    ) {
        let owner = activated.identity.process_instance_id;
        let dimensions = activated
            .repository
            .connection_mut()
            .query_row(
                "SELECT dimensions FROM vector_spaces WHERE vector_space_id = ?1",
                params![vector_space_id.as_str()],
                |row| row.get::<_, u32>(0),
            )
            .unwrap();
        let transaction = activated.repository.connection_mut().transaction().unwrap();
        let manifest = match authorize_generation(
            &transaction,
            vector_space_id,
            VectorDimensions::new(dimensions).unwrap(),
            observed_at_unix_ms,
        )
        .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest) => manifest,
            acknowledgement => panic!("unexpected generation authorization: {acknowledgement:?}"),
        };
        let fence =
            match claim_rebuild_lease(&transaction, vector_space_id, owner, observed_at_unix_ms)
                .unwrap()
            {
                RebuildLeaseClaimAck::Claimed(fence) => fence,
                acknowledgement => panic!("unexpected rebuild claim: {acknowledgement:?}"),
            };
        assert_eq!(
            create_generation_objects(&transaction, &fence, observed_at_unix_ms).unwrap(),
            GenerationObjectCreationAck::Created
        );
        assert!(matches!(
            populate_rebuild_chunk(&transaction, &fence, observed_at_unix_ms + 1).unwrap(),
            RebuildStepAck::Applied {
                processed: 0,
                complete: true,
                applied_source_seq: 0,
            }
        ));
        assert!(matches!(
            catch_up_rebuild_changes(&transaction, &fence, observed_at_unix_ms + 1).unwrap(),
            RebuildStepAck::Applied {
                processed: 0,
                complete: true,
                applied_source_seq: 0,
            }
        ));
        assert_eq!(
            flip_rebuild_generation(&transaction, &fence, observed_at_unix_ms + 2).unwrap(),
            RebuildFlipAck::Activated { record_count: 0 }
        );
        assert_eq!(manifest.vector_space_id(), vector_space_id);
        transaction.commit().unwrap();
    }

    #[derive(Clone, Copy)]
    struct DependencyTarget {
        anchor_id: Uuid,
        shadow_attempt_id: Uuid,
        learning_generation_id: Uuid,
    }

    fn config(path: &Path) -> RouterConfig {
        serde_json::from_value(json!({
            "version": 1,
            "mode": "shadow",
            "project_id": "writer-project",
            "database_path": path.to_string_lossy(),
            "retention_days": 30,
            "max_evidence_records": 1000,
            "pools": [{
                "id": "pool-a",
                "api_family": "openai_chat_completions",
                "anchor_models": ["anchor-a"],
                "anchor_revision": "2026-07-01",
                "sampling_probability": 0.25,
                "max_candidates_per_sample": 1,
                "concurrency": {"shadow": 1, "judge": 1, "max_pending": 1},
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
                    "max_rationale_bytes": 4096,
                    "base_cooloff_seconds": 10,
                    "max_cooloff_seconds": 300
                },
                "candidates": [{
                    "id": "candidate-a",
                    "model": "candidate-model-a",
                    "model_revision": "2026-06-01",
                    "cost_rank": 0,
                    "max_context_tokens": 32768,
                    "capabilities": {"tools": true}
                }]
            }]
        }))
        .unwrap()
    }

    fn evaluator_version() -> String {
        config(Path::new("unused.db")).pools[0]
            .judge
            .evaluator_version()
            .unwrap()
    }

    fn writer(
        capacity: usize,
    ) -> (
        tempfile::TempDir,
        LedgerWriterOwner,
        super::LedgerWriterClient,
    ) {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let activated = LedgerRepository::activate(&config(&path)).unwrap();
        let (owner, client) = LedgerWriterOwner::start(activated.repository, capacity).unwrap();
        (temporary, owner, client)
    }

    fn recovery_snapshot(connection: &rusqlite::Connection) -> Vec<(String, String)> {
        use std::fmt::Write as _;

        let tables = connection
            .prepare(
                "SELECT name FROM sqlite_schema
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                   AND name <> 'schema_migrations'
                 ORDER BY name",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        let mut snapshot = Vec::new();
        for table in tables {
            let mut statement = connection
                .prepare(&format!("SELECT * FROM \"{table}\" ORDER BY rowid"))
                .unwrap();
            let column_count = statement.column_count();
            let mut rows = statement.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                let mut encoded = String::new();
                for index in 0..column_count {
                    if index != 0 {
                        encoded.push('|');
                    }
                    match row.get_ref(index).unwrap() {
                        rusqlite::types::ValueRef::Null => encoded.push('n'),
                        rusqlite::types::ValueRef::Integer(value) => {
                            write!(encoded, "i{value}").unwrap();
                        }
                        rusqlite::types::ValueRef::Real(value) => {
                            write!(encoded, "r{:016x}", value.to_bits()).unwrap();
                        }
                        rusqlite::types::ValueRef::Text(value) => {
                            encoded.push('t');
                            for byte in value {
                                write!(encoded, "{byte:02x}").unwrap();
                            }
                        }
                        rusqlite::types::ValueRef::Blob(value) => {
                            encoded.push('b');
                            for byte in value {
                                write!(encoded, "{byte:02x}").unwrap();
                            }
                        }
                    }
                }
                snapshot.push((table.clone(), encoded));
            }
        }
        snapshot
    }

    #[test]
    fn transaction_start_refusals_map_to_stable_writer_abort() {
        let failures = [
            map_decision_audit_ack(DecisionAuditAck::TransactionNotStarted).unwrap_err(),
            map_live_embedding_prepare_ack(LiveEmbeddingPrepareAck::TransactionNotStarted)
                .unwrap_err(),
            map_embedding_batch_claim_ack(EmbeddingJobBatchClaimAck::TransactionNotStarted)
                .unwrap_err(),
            map_embedding_batch_completion_ack(
                EmbeddingJobBatchCompletionAck::TransactionNotStarted,
            )
            .unwrap_err(),
            map_embedding_resolution_ack(EmbeddingJobResolutionAck::TransactionNotStarted)
                .unwrap_err(),
            map_embedding_reset_ack(EmbeddingJobResetAck::TransactionNotStarted).unwrap_err(),
            map_materialization_claim_ack(MaterializationClaimAck::TransactionNotStarted)
                .unwrap_err(),
            map_materialization_completion_ack(MaterializationCompletionAck::TransactionNotStarted)
                .unwrap_err(),
            map_materialization_resolution_ack(MaterializationResolutionAck::TransactionNotStarted)
                .unwrap_err(),
            map_materialization_failure_propagation_ack(
                MaterializationFailurePropagationAck::TransactionNotStarted,
            )
            .unwrap_err(),
            map_vector_backfill_ack(VectorBackfillAck::TransactionNotStarted).unwrap_err(),
        ];
        assert!(
            failures
                .into_iter()
                .all(|failure| failure.class() == WriterFailureClass::Aborted)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn live_prepare_and_embedding_job_commands_roundtrip_through_the_sole_writer() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let mut writer_config = config(&path);
        writer_config.embedders.push(
            serde_json::from_value(json!({
                "id": "embedder-a",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "embedding-model-a",
                "provider_revision": "2026-07-01",
                "dimensions": 2,
                "api_key_env": "ROUTER_TEST_EMBEDDING_SECRET",
                "timeout_ms": 10000
            }))
            .unwrap(),
        );
        writer_config.pools[0].learning =
            Some(crate::config::LearningConfig::minimal("embedder-a"));
        let activated = LedgerRepository::activate(&writer_config).unwrap();
        let pool = activated.identity.pool("pool-a").unwrap();
        let vector_space_id = activated
            .identity
            .pool("pool-a")
            .and_then(|identity| identity.vector_space.as_ref())
            .expect("learning pool should have one verified vector space")
            .vector_space_id
            .clone();
        let mapping = FrozenMappingKey::new(
            activated.identity.project_uuid,
            activated.identity.config_generation_id.clone(),
            "pool-a",
            pool.policy_version_id.clone(),
        )
        .unwrap();
        let created_at: i64 = rusqlite::Connection::open(&path)
            .unwrap()
            .query_row(
                "SELECT started_at_unix_ms FROM process_instances
                 WHERE process_instance_id = ?1",
                params![activated.identity.process_instance_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        let query = CanonicalRoutingQueryV1 {
            schema: CANONICAL_ROUTING_QUERY_SCHEMA_V1.to_string(),
            instructions: Vec::new(),
            current_task: CanonicalTaskV1 {
                text: "writer embedding job".to_string(),
            },
            bounded_context: Vec::new(),
            tool_schema_fingerprint: "1".repeat(64),
            response_schema_fingerprint: None,
            required_capabilities: Vec::new(),
            position_features: None,
        };
        let canonical_bytes = canonical_json(&serde_json::to_value(&query).unwrap())
            .unwrap()
            .into_bytes();
        let query_artifact = CanonicalRoutingQueryArtifactV1 {
            query,
            canonical_query_hash: sha256_hex(&canonical_bytes),
            canonical_bytes,
        };
        let content_hash = query_artifact.canonical_query_hash.clone();
        let (mut owner, client) = LedgerWriterOwner::start(activated.repository, 4).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        let prepare = LiveEmbeddingPrepare::new(
            mapping,
            query_artifact,
            Uuid::now_v7(),
            Uuid::now_v7(),
            created_at + 1,
        )
        .unwrap();
        let retry = prepare.clone();
        let LiveEmbeddingPrepareAck::Pending(job) = client
            .prepare_live_embedding_until(prepare, deadline)
            .await
            .unwrap()
        else {
            panic!("writer should prepare the job");
        };
        assert_eq!(
            client
                .prepare_live_embedding_until(retry, deadline)
                .await
                .unwrap(),
            LiveEmbeddingPrepareAck::Pending(job.clone())
        );
        let token = Uuid::now_v7();
        let claim = EmbeddingJobBatchClaim::new(
            Uuid::now_v7(),
            vector_space_id.clone(),
            token,
            created_at + 2,
            vec![
                EmbeddingJobBatchClaimItem::new(
                    job.embedding_job_id.clone(),
                    Uuid::now_v7(),
                    Uuid::now_v7(),
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let EmbeddingJobBatchClaimAck::Claimed(leases) = client
            .claim_embedding_job_batch_until(claim, deadline)
            .await
            .unwrap()
        else {
            panic!("writer should claim the batch");
        };
        let lease = leases.into_iter().next().unwrap();
        let vector = AuthoritativeVector::from_normalized(
            &vector_space_id,
            NormalizedVector::from_provider_f64(&[1.0, 0.0], VectorDimensions::new(2).unwrap())
                .unwrap(),
        )
        .unwrap();
        let completion = EmbeddingJobBatchCompletion::new(
            Uuid::now_v7(),
            vector_space_id,
            token,
            created_at + 3,
            vec![
                EmbeddingJobBatchCompletionItem::new(
                    job.embedding_job_id,
                    Uuid::now_v7(),
                    Uuid::now_v7(),
                    lease.job.attempt_generation,
                    content_hash,
                    vector,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        assert!(matches!(
            client
                .complete_embedding_job_batch_until(completion, deadline)
                .await
                .unwrap(),
            EmbeddingJobBatchCompletionAck::Applied(results) if results.len() == 1
        ));
        owner.drain_until(deadline).await.unwrap();
    }

    struct CacheBackedMaterializationWriter {
        temporary: tempfile::TempDir,
        owner: LedgerWriterOwner,
        client: super::LedgerWriterClient,
        deadline: Instant,
        process_started_at_unix_ms: i64,
        materialization_job_id: String,
        attempt_generation: i64,
        canonical_payload_hash: String,
        embedding_id: Uuid,
    }

    async fn cache_backed_materialization_writer() -> CacheBackedMaterializationWriter {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let mut writer_config = config(&path);
        writer_config.embedders.push(
            serde_json::from_value(json!({
                "id": "embedder-a",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "embedding-model-a",
                "provider_revision": "2026-07-01",
                "dimensions": 2,
                "api_key_env": "ROUTER_TEST_EMBEDDING_SECRET",
                "timeout_ms": 10000
            }))
            .unwrap(),
        );
        writer_config.pools[0].learning =
            Some(crate::config::LearningConfig::minimal("embedder-a"));
        let mut activated = LedgerRepository::activate_at(&writer_config, CREATED_AT).unwrap();
        let identity = activated.identity.clone();
        let vector_space_id = identity
            .pool("pool-a")
            .and_then(|pool| pool.vector_space.as_ref())
            .expect("learning pool should have one verified vector space")
            .vector_space_id
            .clone();
        let process_started_at_unix_ms = rusqlite::Connection::open(&path)
            .unwrap()
            .query_row(
                "SELECT started_at_unix_ms FROM process_instances
                 WHERE process_instance_id = ?1",
                params![identity.process_instance_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        let attempt = vectorizable_shadow_attempt(CREATED_AT);
        let anchor_id =
            seed_closed_anchor_for_attempt(&mut activated.repository, &identity, &attempt);
        let reservation = shadow_reservation(&identity, anchor_id, attempt.clone(), CREATED_AT);
        let start = ShadowAttemptStarted::new(
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            CREATED_AT + 1,
        )
        .unwrap();
        let terminal = operational_terminal(&reservation, &attempt, CREATED_AT + 2)
            .with_vectorization(atomic_vectorization(anchor_id));
        activate_empty_vector_generation(
            &mut activated,
            &vector_space_id,
            process_started_at_unix_ms + 1,
        );
        let (owner, client) = LedgerWriterOwner::start(activated.repository, 4).unwrap();
        let deadline = Instant::now() + Duration::from_secs(5);
        assert_eq!(
            client
                .reserve_sample_batch_until(reservation, deadline)
                .await
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            client
                .start_shadow_attempt_until(start, deadline)
                .await
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            client
                .record_shadow_terminal_until(terminal, deadline)
                .await
                .unwrap(),
            ShadowCommandAck::Applied
        );

        let connection = rusqlite::Connection::open(&path).unwrap();
        let (
            materialization_job_id,
            embedding_job_id,
            attempt_generation,
            canonical_payload_hash,
            canonical_query_hash,
        ): (String, String, i64, String, String) = connection
            .query_row(
                "SELECT vector_materialization_job_id, embedding_job_id,
                        attempt_generation, canonical_payload_hash, canonical_query_hash
                 FROM vector_materialization_jobs",
                [],
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
            .unwrap();
        drop(connection);

        let embedding_lease_token = Uuid::now_v7();
        let embedding_claim = EmbeddingJobBatchClaim::new(
            Uuid::now_v7(),
            vector_space_id.clone(),
            embedding_lease_token,
            process_started_at_unix_ms + 3,
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
        let embedding_claim_ack = client
            .claim_embedding_job_batch_until(embedding_claim, deadline)
            .await
            .unwrap();
        let EmbeddingJobBatchClaimAck::Claimed(mut embedding_leases) = embedding_claim_ack else {
            panic!("unexpected embedding claim: {embedding_claim_ack:?}");
        };
        let embedding_lease = embedding_leases.remove(0);
        let embedding_id = Uuid::now_v7();
        let vector = AuthoritativeVector::from_normalized(
            &vector_space_id,
            NormalizedVector::from_provider_f64(&[1.0, 0.0], VectorDimensions::new(2).unwrap())
                .unwrap(),
        )
        .unwrap();
        let embedding_completion = EmbeddingJobBatchCompletion::new(
            Uuid::now_v7(),
            vector_space_id,
            embedding_lease_token,
            process_started_at_unix_ms + 4,
            vec![
                EmbeddingJobBatchCompletionItem::new(
                    embedding_job_id,
                    embedding_id,
                    Uuid::now_v7(),
                    embedding_lease.job.attempt_generation,
                    canonical_query_hash,
                    vector,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        assert!(matches!(
            client
                .complete_embedding_job_batch_until(embedding_completion, deadline)
                .await
                .unwrap(),
            EmbeddingJobBatchCompletionAck::Applied(results)
                if results.len() == 1 && results[0].embedding.embedding_id == embedding_id
        ));

        CacheBackedMaterializationWriter {
            temporary,
            owner,
            client,
            deadline,
            process_started_at_unix_ms,
            materialization_job_id,
            attempt_generation,
            canonical_payload_hash,
            embedding_id,
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn materialization_commands_roundtrip_through_the_sole_writer() {
        let mut fixture = cache_backed_materialization_writer().await;
        let lease_token = Uuid::now_v7();
        let claim = MaterializationClaim::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            fixture.materialization_job_id.clone(),
            lease_token,
            fixture.attempt_generation,
            fixture.canonical_payload_hash.clone(),
            fixture.process_started_at_unix_ms + 5,
        )
        .unwrap();
        let MaterializationClaimAck::Claimed(lease) = fixture
            .client
            .claim_materialization_until(claim, fixture.deadline)
            .await
            .unwrap()
        else {
            panic!("writer should claim the cache-backed materialization");
        };
        let completion = MaterializationCompletion::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            lease.job.materialization_job_id.clone(),
            lease_token,
            lease.job.attempt_generation,
            lease.job.canonical_payload_hash.clone(),
            fixture.process_started_at_unix_ms + 6,
        )
        .unwrap();
        let completed_job = match fixture
            .client
            .complete_materialization_until(completion, fixture.deadline)
            .await
            .unwrap()
        {
            MaterializationCompletionAck::Applied {
                state: MaterializationCompletionState::Ready,
                job,
            } => job,
            acknowledgement => panic!("unexpected materialization completion: {acknowledgement:?}"),
        };
        assert_eq!(completed_job.embedding_id, Some(fixture.embedding_id));

        fixture.owner.drain_until(fixture.deadline).await.unwrap();
        let connection =
            rusqlite::Connection::open(fixture.temporary.path().join("ledger/router.db")).unwrap();
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM vector_index_manifest", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            1
        );
        let states: (String, String) = connection
            .query_row(
                "SELECT
                    (SELECT state FROM evidence_vector_link_state_events
                     ORDER BY event_seq DESC LIMIT 1),
                    (SELECT state FROM vector_materialization_job_state_events
                     ORDER BY event_seq DESC LIMIT 1)",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(states, ("ready".to_string(), "ready".to_string()));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_rejects_queued_materialization_claim_before_begin() {
        let mut fixture = cache_backed_materialization_writer().await;
        let path = fixture.temporary.path().join("ledger/router.db");
        let before: (i64, String, Option<String>, i64) = rusqlite::Connection::open(&path)
            .unwrap()
            .query_row(
                "SELECT attempt_generation, canonical_payload_hash, lease_token,
                        (SELECT count(*) FROM vector_materialization_job_state_events)
                 FROM vector_materialization_jobs
                 WHERE vector_materialization_job_id = ?1",
                params![fixture.materialization_job_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let pause_client = fixture.client.clone();
        let deadline = fixture.deadline;
        let pause = tokio::spawn(async move {
            pause_client
                .pause_until(deadline, started_tx, release_rx)
                .await
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let claim = MaterializationClaim::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            fixture.materialization_job_id.clone(),
            Uuid::now_v7(),
            fixture.attempt_generation,
            fixture.canonical_payload_hash.clone(),
            fixture.process_started_at_unix_ms + 5,
        )
        .unwrap();
        let permit = fixture
            .client
            .reserve_accepted_until(fixture.deadline)
            .await
            .unwrap();
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::ClaimMaterialization {
            command: claim,
            reply,
        });

        fixture.client.abort();
        release_tx.send(()).unwrap();
        assert_eq!(
            receiver.await.unwrap().unwrap_err().class(),
            WriterFailureClass::Aborted
        );
        let _ = pause.await;
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if matches!(
                    fixture.owner.lifecycle(),
                    super::WriterLifecycleState::Exited(super::WriterExitClass::Aborted)
                ) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            fixture.owner.join_exited().unwrap_err().class(),
            WriterFailureClass::Aborted
        );

        let after: (i64, String, Option<String>, i64) = rusqlite::Connection::open(path)
            .unwrap()
            .query_row(
                "SELECT attempt_generation, canonical_payload_hash, lease_token,
                        (SELECT count(*) FROM vector_materialization_job_state_events)
                 FROM vector_materialization_jobs
                 WHERE vector_materialization_job_id = ?1",
                params![fixture.materialization_job_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(after, before);
    }

    fn dependency_writer(
        capacity: usize,
    ) -> (
        tempfile::TempDir,
        LedgerWriterOwner,
        super::LedgerWriterClient,
        DependencyTarget,
    ) {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let mut activated = LedgerRepository::activate(&config(&path)).unwrap();
        let target = seed_dependency_target(&mut activated.repository, &activated.identity);
        let (owner, client) = LedgerWriterOwner::start(activated.repository, capacity).unwrap();
        (temporary, owner, client, target)
    }

    fn judge_writer(
        capacity: usize,
    ) -> (
        tempfile::TempDir,
        LedgerWriterOwner,
        super::LedgerWriterClient,
        DependencyTarget,
        DependencyTarget,
    ) {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let mut activated = LedgerRepository::activate(&config(&path)).unwrap();
        let deterministic_target =
            seed_dependency_target(&mut activated.repository, &activated.identity);
        let judge_target = seed_dependency_target(&mut activated.repository, &activated.identity);
        let (owner, client) = LedgerWriterOwner::start(activated.repository, capacity).unwrap();
        (temporary, owner, client, deterministic_target, judge_target)
    }

    fn seed_dependency_target(
        repository: &mut LedgerRepository,
        identity: &LedgerRuntimeIdentity,
    ) -> DependencyTarget {
        let anchor_id = seed_closed_anchor(repository, identity);
        let pool = identity.pools.get("pool-a").unwrap();
        let attempt = shadow_attempt(CREATED_AT);
        let reservation = shadow_reservation(identity, anchor_id, attempt.clone(), CREATED_AT);
        assert_eq!(
            repository.reserve_sample_batch(&reservation).unwrap(),
            ShadowCommandAck::Applied
        );
        let started = ShadowAttemptStarted::new(
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            CREATED_AT + 1,
        )
        .unwrap();
        assert_eq!(
            repository.start_shadow_attempt(started).unwrap(),
            ShadowCommandAck::Applied
        );
        DependencyTarget {
            anchor_id,
            shadow_attempt_id: attempt.shadow_attempt_id,
            learning_generation_id: pool.learning_generation_id,
        }
    }

    fn pending_anchor(
        identity: &LedgerRuntimeIdentity,
        anchor_id: Uuid,
        anchor_call_uuid: Uuid,
    ) -> PendingTrajectoryWindow {
        let pool = identity.pools.get("pool-a").unwrap();
        let mut pending = pending_window(anchor_id);
        pending.anchor_call_uuid = anchor_call_uuid;
        pending.root_uuid = Uuid::now_v7();
        pending.owner_uuid = Uuid::now_v7();
        pending.pool_id = "pool-a".into();
        pending.anchor_model_revision = "2026-07-01".into();
        pending.process_instance_id = identity.process_instance_id;
        pending.project_uuid = identity.project_uuid;
        pending.project_id = identity.project_id.clone();
        pending.config_generation_id = identity.config_generation_id.clone();
        pending.policy_version_id = pool.policy_version_id.clone();
        pending.learning_generation_id = pool.learning_generation_id;
        pending.request_projection = request_projection();
        pending.routing_context_projection.tenant_policy_hash = "3".repeat(64);
        pending.routing_context_projection.agent_policy_hash = "4".repeat(64);
        pending.replay_capability_facts =
            ReplayCapabilityFactsV1::from_capability(&LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: LlmApiFamily::OpenAIChatCompletions,
                transport_identity: "transport-shared".to_string(),
            })
            .unwrap();
        pending.candidate_facts = vec![PersistedCandidateFactV1 {
            schema: CANDIDATE_FACT_SCHEMA_V1.to_string(),
            candidate_id: "candidate-a".to_string(),
            model: "candidate-model-a".to_string(),
            model_revision: "2026-06-01".to_string(),
            cost_rank: 0,
            capabilities: PersistedCandidateCapabilitiesV1 {
                tools: true,
                multimodal_input: false,
                structured_output: false,
                reasoning_controls: false,
            },
            decoding_fingerprint: "1".repeat(64),
        }];
        pending
            .normalized_anchor_response
            .semantic_response_fingerprint
            .clear();
        let mut response_value = serde_json::to_value(&pending.normalized_anchor_response).unwrap();
        response_value
            .as_object_mut()
            .unwrap()
            .remove("semantic_response_fingerprint");
        pending
            .normalized_anchor_response
            .semantic_response_fingerprint = canonical_sha256(&response_value).unwrap();
        pending
    }

    fn seed_closed_anchor(
        repository: &mut LedgerRepository,
        identity: &LedgerRuntimeIdentity,
    ) -> Uuid {
        let anchor_id = Uuid::now_v7();
        let pending = pending_anchor(identity, anchor_id, Uuid::now_v7());
        let frozen_pending =
            FrozenPendingAnchorV1::new(&pending, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT)
                .unwrap();
        assert!(matches!(
            repository.record_pending_anchor(&frozen_pending).unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
        let terminal = PersistedTrajectoryTerminalV1::closed(
            pending,
            Vec::new(),
            1,
            TrajectoryTrigger::ProgressReached,
            Utc.timestamp_millis_opt(CREATED_AT).unwrap(),
            Vec::new(),
        );
        let frozen_terminal =
            FrozenTerminalAnchorV1::new(&terminal, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT)
                .unwrap();
        assert!(matches!(
            repository.record_terminal_anchor(&frozen_terminal).unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
        anchor_id
    }

    fn seed_closed_anchor_for_attempt(
        repository: &mut LedgerRepository,
        identity: &LedgerRuntimeIdentity,
        attempt: &ReservedShadowAttempt,
    ) -> Uuid {
        let anchor_id = Uuid::now_v7();
        let mut pending = pending_anchor(identity, anchor_id, Uuid::now_v7());
        let mut anchor_request = attempt.request_projection.clone();
        anchor_request.normalized_request.model = Some("anchor-a".to_string());
        anchor_request.semantic_request_fingerprint.clear();
        let mut value = serde_json::to_value(&anchor_request).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("semantic_request_fingerprint");
        anchor_request.semantic_request_fingerprint = canonical_sha256(&value).unwrap();
        pending.request_projection = anchor_request;
        let frozen_pending =
            FrozenPendingAnchorV1::new(&pending, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT)
                .unwrap();
        assert!(matches!(
            repository.record_pending_anchor(&frozen_pending).unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
        let terminal = PersistedTrajectoryTerminalV1::closed(
            pending,
            Vec::new(),
            1,
            TrajectoryTrigger::ProgressReached,
            Utc.timestamp_millis_opt(CREATED_AT).unwrap(),
            Vec::new(),
        );
        let frozen_terminal =
            FrozenTerminalAnchorV1::new(&terminal, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT)
                .unwrap();
        assert!(matches!(
            repository.record_terminal_anchor(&frozen_terminal).unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
        anchor_id
    }

    fn request_projection() -> RouterRequestProjectionV1 {
        let mut projection = RouterRequestProjectionV1 {
            schema: REQUEST_PROJECTION_SCHEMA_V1.to_string(),
            family: LlmApiFamily::OpenAIChatCompletions,
            normalized_request: SanitizedAnnotatedLlmRequest {
                messages: Vec::new(),
                model: Some("anchor-a".to_string()),
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

    fn candidate_request_projection() -> RouterRequestProjectionV1 {
        let mut projection = request_projection();
        projection.normalized_request.model = Some("candidate-model-a".to_string());
        projection.semantic_request_fingerprint.clear();
        let mut value = serde_json::to_value(&projection).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("semantic_request_fingerprint");
        projection.semantic_request_fingerprint = canonical_sha256(&value).unwrap();
        projection
    }

    fn response_projection(model: &str) -> RouterResponseProjectionV1 {
        let mut projection = pending_window(Uuid::now_v7()).normalized_anchor_response;
        projection.model = Some(model.to_string());
        projection.semantic_response_fingerprint.clear();
        let mut value = serde_json::to_value(&projection).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("semantic_response_fingerprint");
        projection.semantic_response_fingerprint = canonical_sha256(&value).unwrap();
        projection
    }

    fn judge_input(judge: &crate::config::JudgeConfig) -> PairwiseJudgeInputV1 {
        PairwiseJudgeInputV1::new(
            &request_projection(),
            &response_projection("anchor"),
            &response_projection("candidate-model-a"),
            &[],
            JudgeHorizonV1::new(1, 1, TrajectoryTrigger::ProgressReached, false).unwrap(),
            JudgePolicyIdentityV1::from_config(judge).unwrap(),
        )
        .unwrap()
    }

    fn shadow_attempt(created_at_unix_ms: i64) -> ReservedShadowAttempt {
        ReservedShadowAttempt::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            "candidate-a",
            "candidate-model-a",
            "2026-06-01",
            0,
            LlmApiFamily::OpenAIChatCompletions,
            "transport-shared",
            "anchor-a",
            "2026-07-01",
            "1".repeat(64),
            evaluator_version(),
            "3".repeat(64),
            "4".repeat(64),
            true,
            candidate_request_projection(),
            created_at_unix_ms,
        )
        .unwrap()
    }

    fn vectorizable_shadow_attempt(created_at_unix_ms: i64) -> ReservedShadowAttempt {
        let mut projection = candidate_request_projection();
        projection.normalized_request.messages = vec![SanitizedMessage::User {
            content: SanitizedMessageContent::Text("route this writer request".to_string()),
            name: None,
        }];
        projection.semantic_request_fingerprint.clear();
        let mut value = serde_json::to_value(&projection).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("semantic_request_fingerprint");
        projection.semantic_request_fingerprint = canonical_sha256(&value).unwrap();
        ReservedShadowAttempt::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            "candidate-a",
            "candidate-model-a",
            "2026-06-01",
            0,
            LlmApiFamily::OpenAIChatCompletions,
            "transport-shared",
            "anchor-a",
            "2026-07-01",
            "1".repeat(64),
            evaluator_version(),
            "3".repeat(64),
            "4".repeat(64),
            true,
            projection,
            created_at_unix_ms,
        )
        .unwrap()
    }

    fn shadow_reservation(
        identity: &LedgerRuntimeIdentity,
        anchor_id: Uuid,
        attempt: ReservedShadowAttempt,
        created_at_unix_ms: i64,
    ) -> SampleBatchReservation {
        let pool = identity.pools.get("pool-a").unwrap();
        SampleBatchReservation::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            anchor_id,
            identity.config_generation_id.clone(),
            pool.policy_version_id.clone(),
            pool.learning_generation_id,
            "pool-a",
            vec![attempt],
            created_at_unix_ms,
        )
        .unwrap()
    }

    fn atomic_vectorization(anchor_id: Uuid) -> ShadowVectorizationHandoff {
        let mut routing_projection = pending_window(anchor_id).routing_context_projection;
        routing_projection.tenant_policy_hash = "3".repeat(64);
        routing_projection.agent_policy_hash = "4".repeat(64);
        ShadowVectorizationHandoff::Atomic(
            AtomicShadowVectorization::new(
                routing_projection,
                Uuid::now_v7(),
                Uuid::now_v7(),
                Uuid::now_v7(),
                Uuid::now_v7(),
            )
            .unwrap(),
        )
    }

    fn operational_terminal(
        reservation: &SampleBatchReservation,
        attempt: &ReservedShadowAttempt,
        created_at_unix_ms: i64,
    ) -> ShadowTerminalRecord {
        let batch_terminal = SampleBatchTerminalEvent::new(
            reservation.sample_batch_id,
            Uuid::now_v7(),
            SampleBatchTerminalState::Closed,
            None,
            created_at_unix_ms,
        )
        .unwrap();
        ShadowTerminalRecord::new(
            Uuid::now_v7(),
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            ShadowTerminalClass::OperationalFailure,
            None,
            None,
            None,
            Some(ShadowOperationalFailureClass::new("router.provider.timeout").unwrap()),
            Some(12),
            None,
            None,
            ShadowVectorSourceV1::Canonicalizable {
                query_inputs: Box::new(attempt.request_projection.clone()),
            },
            Some(batch_terminal),
            created_at_unix_ms,
        )
        .unwrap()
    }

    fn candidate_identity() -> CandidateDependencyIdentity {
        CandidateDependencyIdentity::new(
            LlmApiFamily::OpenAIChatCompletions,
            "transport-shared",
            "candidate-model-a",
            "2026-06-01",
        )
        .unwrap()
    }

    fn judge_identity() -> JudgeDependencyIdentity {
        JudgeDependencyIdentity::new(
            LlmApiFamily::OpenAIChatCompletions,
            "transport-shared",
            "judge-model",
            "2026-07-01",
            "pairwise-equivalence-v1",
            JUDGE_PROMPT_TEMPLATE_SHA256_V1,
            "response-trajectory-equivalence-v1",
            JUDGE_RUBRIC_TEMPLATE_SHA256_V1,
            1,
            JUDGE_OUTPUT_SCHEMA_SHA256_V1,
        )
        .unwrap()
    }

    fn dependency_operation(target: DependencyTarget, created_at: i64) -> DependencyOperation {
        DependencyOperation::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            target.anchor_id,
            target.shadow_attempt_id,
            10,
            300,
            created_at,
        )
        .unwrap()
    }

    fn transition(acknowledgement: &DependencyCommandAck) -> DependencyTransition {
        match acknowledgement {
            DependencyCommandAck::Applied(snapshot)
            | DependencyCommandAck::AlreadyApplied(snapshot) => snapshot.transition,
            other => panic!("expected durable dependency state, got {other:?}"),
        }
    }

    #[test]
    fn zero_capacity_start_stops_the_exact_activated_process() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let activated = LedgerRepository::activate(&config(&path)).unwrap();
        let process_instance_id = activated.identity.process_instance_id;

        let result = LedgerWriterOwner::start(activated.repository, 0);
        assert!(matches!(result, Err(error) if error.class() == WriterFailureClass::Protocol));

        let connection = rusqlite::Connection::open(path).unwrap();
        let stopped = connection
            .query_row(
                "SELECT count(*) FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'stopped'",
                [process_instance_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(stopped, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn repeated_abort_is_one_shot_and_preserves_activation_failure_stop() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let activated = LedgerRepository::activate(&config(&path)).unwrap();
        let process_instance_id = activated.identity.process_instance_id;
        let (owner, client) = LedgerWriterOwner::start(activated.repository, 2).unwrap();
        let shared = owner.shared.clone();
        let lifecycle = shared.lifecycle.subscribe();

        owner.request_process_stop_on_abort();
        client.abort();
        owner.abort();
        drop(owner);

        assert_eq!(shared.abort_epoch.load(Ordering::Acquire), 1);
        super::wait_for_exit(lifecycle, Instant::now() + Duration::from_secs(3))
            .await
            .unwrap();
        assert_eq!(shared.abort_epoch.load(Ordering::Acquire), 1);
        drop(client);
        drop(shared);

        let connection = rusqlite::Connection::open(path).unwrap();
        let stopped = connection
            .query_row(
                "SELECT count(*) FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'stopped'",
                [process_instance_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(stopped, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_cannot_overwrite_a_clean_writer_exit() {
        let (_temporary, mut owner, client) = writer(2);
        let deadline = Instant::now() + Duration::from_secs(3);

        owner.drain_until(deadline).await.unwrap();
        assert!(matches!(
            owner.lifecycle(),
            super::WriterLifecycleState::Exited(super::WriterExitClass::Clean)
        ));

        client.abort();
        assert!(matches!(
            owner.lifecycle(),
            super::WriterLifecycleState::Exited(super::WriterExitClass::Clean)
        ));
        assert_eq!(owner.shared.abort_epoch.load(Ordering::Acquire), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn typed_anchor_commands_roundtrip_through_the_sole_writer() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let activated = LedgerRepository::activate(&config(&path)).unwrap();
        let pending = pending_anchor(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen_pending =
            FrozenPendingAnchorV1::new(&pending, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT)
                .unwrap();
        let terminal = PersistedTrajectoryTerminalV1::closed(
            pending.clone(),
            Vec::new(),
            1,
            TrajectoryTrigger::ProgressReached,
            Utc.timestamp_millis_opt(CREATED_AT + 10_000).unwrap(),
            Vec::new(),
        );
        let frozen_terminal = FrozenTerminalAnchorV1::new(
            &terminal,
            Uuid::now_v7(),
            Uuid::now_v7(),
            CREATED_AT + 10_000,
        )
        .unwrap();
        let (mut owner, client) = LedgerWriterOwner::start(activated.repository, 4).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);

        let probe_permit = client.try_reserve_pre_accept().unwrap();
        assert_eq!(
            client
                .probe_pending_anchor_with_permit(probe_permit, frozen_pending.clone())
                .await
                .unwrap(),
            AnchorProbe::Missing {
                anchor_id: pending.anchor_id
            }
        );
        let pending_permit = client.try_reserve_pre_accept().unwrap();
        assert!(matches!(
            client
                .record_pending_anchor_with_capacity_and_permit(
                    pending_permit,
                    frozen_pending.clone(),
                    100,
                )
                .await
                .unwrap(),
            PendingAnchorCapacityAck::Applied { anchor_id, .. } if anchor_id == pending.anchor_id
        ));
        assert!(matches!(
            client
                .probe_pending_anchor_until(frozen_pending.clone(), deadline)
                .await
                .unwrap(),
            AnchorProbe::Pending { anchor_id, .. } if anchor_id == pending.anchor_id
        ));
        assert!(matches!(
            client
                .record_terminal_anchor_until(frozen_terminal, deadline)
                .await
                .unwrap(),
            AnchorCommandAck::Applied { anchor_id, .. } if anchor_id == pending.anchor_id
        ));
        assert!(matches!(
            client
                .probe_pending_anchor_until(frozen_pending, deadline)
                .await
                .unwrap(),
            AnchorProbe::Terminal { anchor_id, .. } if anchor_id == pending.anchor_id
        ));
        owner.drain_until(deadline).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abrupt_writer_abort_recovers_an_acknowledged_pending_anchor() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let config = config(&path);
        let activated = LedgerRepository::activate_at(&config, CREATED_AT).unwrap();
        let dead_process = activated.identity.process_instance_id;
        let anchor_id = Uuid::now_v7();
        let pending = pending_anchor(&activated.identity, anchor_id, Uuid::now_v7());
        let frozen =
            FrozenPendingAnchorV1::new(&pending, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT)
                .unwrap();
        let heartbeat_expiry: i64 = rusqlite::Connection::open(&path)
            .unwrap()
            .query_row(
                "SELECT heartbeat_expires_at_unix_ms FROM process_instances
                 WHERE process_instance_id = ?1",
                params![dead_process.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        let (mut owner, client) = LedgerWriterOwner::start(activated.repository, 2).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        assert!(matches!(
            client
                .record_pending_anchor_until(frozen, deadline)
                .await
                .unwrap(),
            AnchorCommandAck::Applied {
                anchor_id: applied,
                ..
            } if applied == anchor_id
        ));

        let lifecycle = owner.shared.lifecycle.subscribe();
        owner.abort();
        super::wait_for_exit(lifecycle, deadline).await.unwrap();
        assert_eq!(
            owner.join_exited().unwrap_err().class(),
            WriterFailureClass::Aborted
        );
        drop(client);
        drop(owner);

        let connection = rusqlite::Connection::open(&path).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM process_instance_state_events
                     WHERE subject_process_instance_id = ?1
                       AND state IN ('stopped', 'reconciled')",
                    params![dead_process.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        drop(connection);

        let mut recovered = LedgerRepository::activate_at(&config, heartbeat_expiry).unwrap();
        let reconciler_process = recovered.identity.process_instance_id;
        let connection = rusqlite::Connection::open(&path).unwrap();
        let process_fence: (String, String) = connection
            .query_row(
                "SELECT process_instance_id, subject_process_instance_id
                 FROM process_instance_state_events
                 WHERE subject_process_instance_id = ?1 AND state = 'reconciled'",
                params![dead_process.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(process_fence.0, reconciler_process.to_string());
        assert_eq!(process_fence.1, dead_process.to_string());
        let anchor_terminal: (String, String, String) = connection
            .query_row(
                "SELECT state, process_instance_id, dead_process_instance_id
                 FROM anchor_state_events
                 WHERE anchor_id = ?1 AND state <> 'pending'",
                params![anchor_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(anchor_terminal.0, "orphaned_non_resumable");
        assert_eq!(anchor_terminal.1, reconciler_process.to_string());
        assert_eq!(anchor_terminal.2, dead_process.to_string());
        let recovery_counts: (i64, i64, i64, i64) = connection
            .query_row(
                "SELECT
                    (SELECT count(*) FROM process_instance_state_events
                     WHERE subject_process_instance_id = ?1 AND state = 'stopped'),
                    (SELECT count(*) FROM anchor_state_events WHERE anchor_id = ?2),
                    (SELECT count(*) FROM anchor_windows WHERE anchor_id = ?2),
                    (SELECT count(*) FROM sample_batches WHERE anchor_id = ?2)",
                params![dead_process.to_string(), anchor_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(recovery_counts, (0, 2, 0, 0));

        let snapshot = recovery_snapshot(&connection);
        for _ in 0..2 {
            let report = recovered.repository.reconcile_at(heartbeat_expiry).unwrap();
            assert_eq!(report, Default::default());
            assert_eq!(recovery_snapshot(&connection), snapshot);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queue_full_decline_consumes_pre_accept_permit_and_roundtrips() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let activated = LedgerRepository::activate(&config(&path)).unwrap();
        let pending = pending_anchor(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen =
            FrozenPendingAnchorV1::new(&pending, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT)
                .unwrap();
        let decline =
            NotScheduledQueueFull::new(frozen.clone(), Uuid::now_v7(), CREATED_AT + 1).unwrap();
        let (mut owner, client) = LedgerWriterOwner::start(activated.repository, 2).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        let permit = client.try_reserve_pre_accept().unwrap();

        assert_eq!(
            client
                .record_not_scheduled_queue_full_with_capacity_and_permit(
                    permit,
                    decline.clone(),
                    100,
                )
                .await
                .unwrap(),
            NotScheduledQueueFullAck::Applied {
                anchor_id: pending.anchor_id,
                pending_hash: decline.pending_hash().to_string(),
                terminal_hash: decline.terminal_hash().to_string(),
            }
        );
        let retry_permit = client.try_reserve_pre_accept().unwrap();
        assert_eq!(
            client
                .record_not_scheduled_queue_full_with_capacity_and_permit(
                    retry_permit,
                    decline.clone(),
                    0,
                )
                .await
                .unwrap(),
            NotScheduledQueueFullAck::AlreadyApplied {
                anchor_id: pending.anchor_id,
                pending_hash: decline.pending_hash().to_string(),
                terminal_hash: decline.terminal_hash().to_string(),
            }
        );
        assert_eq!(
            client
                .probe_pending_anchor_until(frozen, deadline)
                .await
                .unwrap(),
            AnchorProbe::NotScheduledQueueFull {
                anchor_id: pending.anchor_id,
                pending_hash: decline.pending_hash().to_string(),
                terminal_hash: decline.terminal_hash().to_string(),
            }
        );
        owner.drain_until(deadline).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_rejects_queued_queue_full_decline_before_begin() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let activated = LedgerRepository::activate(&config(&path)).unwrap();
        let pending = pending_anchor(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen =
            FrozenPendingAnchorV1::new(&pending, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT)
                .unwrap();
        let decline = NotScheduledQueueFull::new(frozen, Uuid::now_v7(), CREATED_AT + 1).unwrap();
        let (mut owner, client) = LedgerWriterOwner::start(activated.repository, 2).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let pause_client = client.clone();
        let pause = tokio::spawn(async move {
            pause_client
                .pause_until(deadline, started_tx, release_rx)
                .await
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let permit = client.try_reserve_pre_accept().unwrap();
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RecordNotScheduledQueueFull {
            decline: Box::new(decline),
            max_evidence_records: Some(100),
            reply,
        });

        owner.abort();
        release_tx.send(()).unwrap();
        assert_eq!(
            receiver.await.unwrap().unwrap_err().class(),
            WriterFailureClass::Aborted
        );
        let _ = pause.await;
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if matches!(
                    owner.lifecycle(),
                    super::WriterLifecycleState::Exited(super::WriterExitClass::Aborted)
                ) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            owner.join_exited().unwrap_err().class(),
            WriterFailureClass::Aborted
        );
        let connection = rusqlite::Connection::open(path).unwrap();
        let count: i64 = connection
            .query_row("SELECT count(*) FROM anchors", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queued_pre_accept_pending_waits_for_its_definitive_acknowledgement() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let activated = LedgerRepository::activate(&config(&path)).unwrap();
        let pending = pending_anchor(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen =
            FrozenPendingAnchorV1::new(&pending, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT)
                .unwrap();
        let (mut owner, client) = LedgerWriterOwner::start(activated.repository, 2).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let pause_client = client.clone();
        let pause = tokio::spawn(async move {
            pause_client
                .pause_until(deadline, started_tx, release_rx)
                .await
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let permit = client.try_reserve_pre_accept().unwrap();
        let write_client = client.clone();
        let write = tokio::spawn(async move {
            write_client
                .record_pending_anchor_with_capacity_and_permit(permit, frozen, 100)
                .await
        });
        tokio::time::sleep(Duration::from_millis(75)).await;
        assert!(!write.is_finished());

        release_tx.send(()).unwrap();
        pause.await.unwrap().unwrap();
        assert!(matches!(
            write.await.unwrap().unwrap(),
            PendingAnchorCapacityAck::Applied { anchor_id, .. }
                if anchor_id == pending.anchor_id
        ));
        owner.drain_until(deadline).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn typed_shadow_commands_roundtrip_through_the_sole_writer() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let mut activated = LedgerRepository::activate(&config(&path)).unwrap();
        let identity = activated.identity.clone();
        let anchor_id = seed_closed_anchor(&mut activated.repository, &identity);
        let attempt = shadow_attempt(CREATED_AT);
        let reservation = shadow_reservation(&identity, anchor_id, attempt.clone(), CREATED_AT);
        let start = ShadowAttemptStarted::new(
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            CREATED_AT + 1,
        )
        .unwrap();
        let terminal = operational_terminal(&reservation, &attempt, CREATED_AT + 2);
        let (mut owner, client) = LedgerWriterOwner::start(activated.repository, 4).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);

        assert_eq!(
            client
                .reserve_sample_batch_until(reservation.clone(), deadline)
                .await
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            client
                .reserve_sample_batch_until(reservation, deadline)
                .await
                .unwrap(),
            ShadowCommandAck::AlreadyApplied
        );
        assert_eq!(
            client
                .start_shadow_attempt_until(start, deadline)
                .await
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            client
                .record_shadow_terminal_until(terminal.clone(), deadline)
                .await
                .unwrap(),
            ShadowCommandAck::Applied
        );
        assert_eq!(
            client
                .record_shadow_terminal_until(terminal, deadline)
                .await
                .unwrap(),
            ShadowCommandAck::AlreadyApplied
        );
        owner.drain_until(deadline).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn typed_judge_commands_roundtrip_through_the_sole_writer() {
        let (temporary, mut owner, client, deterministic_target, judge_target) = judge_writer(4);
        let judge = config(&temporary.path().join("unused.db")).pools[0]
            .judge
            .clone();
        let deadline = Instant::now() + Duration::from_secs(3);
        let evaluation = EvaluationRecord::deterministic(
            Uuid::now_v7(),
            deterministic_target.shadow_attempt_id,
            evaluator_version(),
            Uuid::now_v7(),
            DeterministicHardFailureV1::ResponseSchema,
            false,
            &judge,
            CREATED_AT + 2,
        )
        .unwrap();
        let start = JudgeAttemptStart::new(
            Uuid::now_v7(),
            judge_target.shadow_attempt_id,
            judge_target.learning_generation_id,
            evaluator_version(),
            &judge,
            &judge_input(&judge),
            0,
            Uuid::now_v7(),
            Uuid::now_v7(),
            CREATED_AT + 1,
        )
        .unwrap();
        let terminal = JudgeAttemptTerminal::transport_failure(
            start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            JudgeTransportFailureClass::Timeout,
            CREATED_AT + 2,
        )
        .unwrap();

        assert_eq!(
            client
                .record_evaluation_until(evaluation.clone(), deadline)
                .await
                .unwrap(),
            JudgeRecordAck::Applied
        );
        assert_eq!(
            client
                .record_evaluation_until(evaluation, deadline)
                .await
                .unwrap(),
            JudgeRecordAck::AlreadyApplied
        );
        assert_eq!(
            client
                .record_judge_attempt_start_until(start, deadline)
                .await
                .unwrap(),
            JudgeRecordAck::Applied
        );
        assert_eq!(
            client
                .record_judge_attempt_terminal_until(terminal.clone(), deadline)
                .await
                .unwrap(),
            JudgeRecordAck::Applied
        );
        assert_eq!(
            client
                .record_judge_attempt_terminal_until(terminal, deadline)
                .await
                .unwrap(),
            JudgeRecordAck::AlreadyApplied
        );
        owner.drain_until(deadline).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn typed_dependency_commands_roundtrip_through_the_sole_writer() {
        let (_temporary, mut owner, client, target) = dependency_writer(4);
        let deadline = Instant::now() + Duration::from_secs(3);
        let candidate_operation = dependency_operation(target, 1_000);
        let candidate = candidate_identity();

        let claimed = client
            .claim_candidate_dependency_until(
                candidate.clone(),
                candidate_operation.clone(),
                deadline,
            )
            .await
            .unwrap();
        assert!(matches!(claimed, DependencyCommandAck::Applied(_)));
        assert_eq!(transition(&claimed), DependencyTransition::Admitted);
        let duplicate = client
            .claim_candidate_dependency_until(candidate, candidate_operation.clone(), deadline)
            .await
            .unwrap();
        assert!(matches!(duplicate, DependencyCommandAck::AlreadyApplied(_)));

        let completed = client
            .complete_dependency_until(
                DependencyCompletion::success(
                    candidate_operation.dependency_operation_id,
                    Uuid::now_v7(),
                    1_001,
                )
                .unwrap(),
                deadline,
            )
            .await
            .unwrap();
        assert_eq!(transition(&completed), DependencyTransition::Success);

        let judge = client
            .claim_judge_dependency_until(
                judge_identity(),
                dependency_operation(target, 1_002),
                deadline,
            )
            .await
            .unwrap();
        assert_eq!(transition(&judge), DependencyTransition::Admitted);
        owner.drain_until(deadline).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn typed_process_and_health_commands_roundtrip_through_the_sole_writer() {
        let (_temporary, mut owner, client) = writer(4);
        let deadline = Instant::now() + Duration::from_secs(3);
        let observed_at = chrono::Utc::now().timestamp_millis() + 60_000;
        let renewal = HeartbeatRenewal::new(observed_at).unwrap();
        let renewed = client
            .renew_heartbeat_until(renewal, deadline)
            .await
            .unwrap();
        assert_eq!(
            renewed,
            HeartbeatAck::Applied {
                expires_at_unix_ms: renewal.expires_at_unix_ms
            }
        );
        assert_eq!(
            client
                .renew_heartbeat_until(renewal, deadline)
                .await
                .unwrap(),
            HeartbeatAck::AlreadyApplied {
                expires_at_unix_ms: renewal.expires_at_unix_ms
            }
        );

        let health = LedgerHealthEvent::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            None,
            None,
            "router.writer.test",
            LedgerHealthSeverity::Warning,
            observed_at + 1,
        )
        .unwrap();
        assert_eq!(
            client
                .append_health_event_until(health.clone(), deadline)
                .await
                .unwrap(),
            ProcessCommandAck::Applied
        );
        assert_eq!(
            client
                .append_health_event_until(health, deadline)
                .await
                .unwrap(),
            ProcessCommandAck::AlreadyApplied
        );

        let stop = ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), observed_at + 2).unwrap();
        assert_eq!(
            client.stop_process_until(stop, deadline).await.unwrap(),
            ProcessCommandAck::Applied
        );
        assert_eq!(
            client.stop_process_until(stop, deadline).await.unwrap(),
            ProcessCommandAck::AlreadyApplied
        );
        owner.drain_until(deadline).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn zero_selection_retention_receipt_retries_through_the_sole_writer() {
        let (_temporary, mut owner, client) = writer(4);
        let deadline = Instant::now() + Duration::from_secs(3);
        let request = RetentionRequest::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            Utc::now().timestamp_millis(),
        )
        .unwrap();

        let RetentionAck::Applied {
            summary,
            observation,
        } = client.run_retention_until(request, deadline).await.unwrap()
        else {
            panic!("first retention request should apply");
        };
        assert_eq!(summary.retention_batch_id, request.retention_batch_id);
        assert_eq!(summary.created_at_unix_ms, request.created_at_unix_ms);
        assert_eq!(summary.selected_count, 0);
        assert_eq!(observation.terminal_anchor_count, 0);
        assert!(observation.capacity_available);
        assert_eq!(
            client.run_retention_until(request, deadline).await.unwrap(),
            RetentionAck::AlreadyApplied {
                summary: summary.clone(),
                observation,
            }
        );

        owner.drain_until(deadline).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn retention_runs_after_accepted_terminal_evidence_in_writer_fifo() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let activated = LedgerRepository::activate(&config(&path)).unwrap();
        let pending = pending_anchor(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen =
            FrozenPendingAnchorV1::new(&pending, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT)
                .unwrap();
        let decline = NotScheduledQueueFull::new(frozen, Uuid::now_v7(), CREATED_AT + 1).unwrap();
        let (mut owner, client) = LedgerWriterOwner::start(activated.repository, 4).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let pause_client = client.clone();
        let pause = tokio::spawn(async move {
            pause_client
                .pause_until(deadline, started_tx, release_rx)
                .await
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let evidence_permit = client.reserve_accepted_until(deadline).await.unwrap();
        let retention_permit = client.reserve_accepted_until(deadline).await.unwrap();
        let (evidence_reply, evidence_receiver) = oneshot::channel();
        evidence_permit.send(LedgerWriterCommand::RecordNotScheduledQueueFull {
            decline: Box::new(decline),
            max_evidence_records: None,
            reply: evidence_reply,
        });
        let request = RetentionRequest::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            Utc::now().timestamp_millis(),
        )
        .unwrap();
        let (retention_reply, retention_receiver) = oneshot::channel();
        retention_permit.send(LedgerWriterCommand::RunRetention {
            request,
            reply: retention_reply,
        });

        release_tx.send(()).unwrap();
        pause.await.unwrap().unwrap();
        assert!(matches!(
            evidence_receiver.await.unwrap().unwrap(),
            NotScheduledQueueFullAck::Applied { anchor_id, .. } if anchor_id == pending.anchor_id
        ));
        let RetentionAck::Applied {
            summary,
            observation,
        } = retention_receiver.await.unwrap().unwrap()
        else {
            panic!("retention should run after terminal evidence");
        };
        assert_eq!(summary.selected_count, 1);
        assert!(summary.age_expired);
        assert_eq!(observation.terminal_anchor_count, 0);
        assert!(observation.capacity_available);

        owner.drain_until(deadline).await.unwrap();
        let connection = rusqlite::Connection::open(path).unwrap();
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM anchors", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            0
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_rejects_queued_retention_before_transaction_start() {
        let (temporary, mut owner, client) = writer(2);
        let deadline = Instant::now() + Duration::from_secs(3);
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let pause_client = client.clone();
        let pause = tokio::spawn(async move {
            pause_client
                .pause_until(deadline, started_tx, release_rx)
                .await
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let permit = client.reserve_accepted_until(deadline).await.unwrap();
        let request = RetentionRequest::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            Utc::now().timestamp_millis(),
        )
        .unwrap();
        let (reply, receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::RunRetention { request, reply });

        client.abort();
        release_tx.send(()).unwrap();
        assert_eq!(
            receiver.await.unwrap().unwrap_err().class(),
            WriterFailureClass::Aborted
        );
        let _ = pause.await;
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if matches!(
                    owner.lifecycle(),
                    super::WriterLifecycleState::Exited(super::WriterExitClass::Aborted)
                ) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            owner.join_exited().unwrap_err().class(),
            WriterFailureClass::Aborted
        );
        let connection =
            rusqlite::Connection::open(temporary.path().join("ledger/router.db")).unwrap();
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM retention_batches", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            0
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn lost_process_stop_reply_retries_the_byte_identical_command() {
        let (temporary, mut owner, client) = writer(4);
        let deadline = Instant::now() + Duration::from_secs(3);
        let stop = ProcessStop::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            chrono::Utc::now().timestamp_millis() + 60_000,
        )
        .unwrap();
        let permit = client.reserve_accepted_until(deadline).await.unwrap();
        let (first_reply, first_receiver) = oneshot::channel();
        permit.send(LedgerWriterCommand::StopProcess {
            command: stop,
            reply: first_reply,
        });
        drop(first_receiver);

        assert_eq!(
            client.stop_process_until(stop, deadline).await.unwrap(),
            ProcessCommandAck::AlreadyApplied
        );
        owner.drain_until(deadline).await.unwrap();

        let connection =
            rusqlite::Connection::open(temporary.path().join("ledger/router.db")).unwrap();
        let stored = connection
            .query_row(
                "SELECT process_state_event_id, created_at_unix_ms
                 FROM process_instance_state_events
                 WHERE state = 'stopped'",
                [],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
            )
            .unwrap();
        assert_eq!(stored.0, stop.process_state_event_id.to_string());
        assert_eq!(stored.1, stop.created_at_unix_ms);
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM process_instance_state_events WHERE state = 'stopped'",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn accepted_command_waits_for_capacity_and_completes_after_release() {
        let (_temporary, mut owner, client) = writer(1);
        let deadline = Instant::now() + Duration::from_secs(3);
        let held = client.try_reserve_pre_accept().unwrap();
        let waiting_client = client.clone();
        let waiting = tokio::spawn(async move { waiting_client.flush_until(deadline).await });
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(!waiting.is_finished());

        let (first_reply, first_receiver) = oneshot::channel();
        held.send(LedgerWriterCommand::Flush { reply: first_reply });
        first_receiver.await.unwrap().unwrap();
        waiting.await.unwrap().unwrap();
        owner.drain_until(deadline).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn issued_permits_are_bounded_and_graceful_drain_waits_for_them() {
        let (_temporary, mut owner, client) = writer(1);
        let deadline = Instant::now() + Duration::from_secs(3);
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let pause_client = client.clone();
        let pause = tokio::spawn(async move {
            pause_client
                .pause_until(deadline, started_tx, release_rx)
                .await
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let issued = client.try_reserve_pre_accept().unwrap();
        assert_eq!(
            client.try_reserve_pre_accept().err().unwrap().class(),
            WriterFailureClass::Full
        );
        let drain = tokio::spawn(async move { owner.drain_until(deadline).await });
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(!drain.is_finished());

        let (flush_tx, flush_rx) = oneshot::channel();
        issued.send(LedgerWriterCommand::Flush { reply: flush_tx });
        release_tx.send(()).unwrap();
        pause.await.unwrap().unwrap();
        flush_rx.await.unwrap().unwrap();
        drain.await.unwrap().unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn panic_wakes_an_existing_capacity_and_acknowledgement_waiter() {
        let (temporary, owner, client) = writer(1);
        let deadline = Instant::now() + Duration::from_secs(3);
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let pause_client = client.clone();
        let pause = tokio::spawn(async move {
            pause_client
                .pause_until(deadline, started_tx, release_rx)
                .await
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let panic_permit = client.try_reserve_pre_accept().unwrap();
        let waiting_client = client.clone();
        let waiting = tokio::spawn(async move { waiting_client.flush_until(deadline).await });
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(!waiting.is_finished());

        let (panic_reply, _panic_receiver) = oneshot::channel();
        panic_permit.send(LedgerWriterCommand::Panic { reply: panic_reply });
        release_tx.send(()).unwrap();
        pause.await.unwrap().unwrap();
        let error = waiting.await.unwrap().unwrap_err();
        assert_eq!(error.class(), WriterFailureClass::Panicked);
        assert!(matches!(
            owner.lifecycle(),
            super::WriterLifecycleState::Exited(super::WriterExitClass::Panicked)
        ));
        client.abort();
        assert!(matches!(
            owner.lifecycle(),
            super::WriterLifecycleState::Exited(super::WriterExitClass::Panicked)
        ));
        assert_eq!(owner.shared.abort_epoch.load(Ordering::Acquire), 0);
        let connection =
            rusqlite::Connection::open(temporary.path().join("ledger/router.db")).unwrap();
        let stopped = connection
            .query_row(
                "SELECT count(*) FROM process_instance_state_events WHERE state = 'stopped'",
                [],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(stopped, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn client_abort_invalidates_a_preexisting_permit_without_waiting() {
        let (_temporary, mut owner, client) = writer(1);
        let issued = client.try_reserve_pre_accept().unwrap();
        client.abort();
        let target = DependencyTarget {
            anchor_id: Uuid::now_v7(),
            shadow_attempt_id: Uuid::now_v7(),
            learning_generation_id: Uuid::now_v7(),
        };
        let (reply, receiver) = oneshot::channel();
        issued.send(LedgerWriterCommand::ClaimCandidateDependency {
            identity: candidate_identity(),
            operation: dependency_operation(target, 1_000),
            reply,
        });
        if let Ok(result) = receiver.await {
            assert_eq!(result.unwrap_err().class(), WriterFailureClass::Aborted);
        }
        assert_eq!(
            client.try_reserve_pre_accept().err().unwrap().class(),
            WriterFailureClass::Aborted
        );
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if matches!(
                    owner.lifecycle(),
                    super::WriterLifecycleState::Exited(super::WriterExitClass::Aborted)
                ) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            owner.join_exited().unwrap_err().class(),
            WriterFailureClass::Aborted
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn abort_wakes_idle_writer_while_every_slot_is_an_unsent_permit() {
        let (_temporary, mut owner, client) = writer(1);
        let issued = client.try_reserve_pre_accept().unwrap();
        owner.abort();
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if matches!(
                    owner.lifecycle(),
                    super::WriterLifecycleState::Exited(super::WriterExitClass::Aborted)
                ) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            owner.join_exited().unwrap_err().class(),
            WriterFailureClass::Aborted
        );
        drop(issued);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn accepted_reservation_refuses_an_already_expired_deadline() {
        let (_temporary, mut owner, client) = writer(1);
        let error = client
            .reserve_accepted_until(Instant::now() - Duration::from_millis(1))
            .await
            .err()
            .expect("an expired deadline must not issue writer authority");
        assert_eq!(error.class(), WriterFailureClass::Deadline);
        let permit = client.try_reserve_pre_accept().unwrap();
        drop(permit);
        owner
            .drain_until(Instant::now() + Duration::from_secs(1))
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn expired_deadline_wins_over_a_ready_reply_and_clean_exit() {
        let deadline = Instant::now() - Duration::from_millis(1);
        let (reply, receiver) = oneshot::channel();
        reply
            .send(Ok::<super::FlushAck, super::WriterFailure>(super::FlushAck))
            .unwrap();
        let (_lifecycle, lifecycle) =
            tokio::sync::watch::channel(super::WriterLifecycleState::Running);
        assert_eq!(
            super::await_reply(receiver, lifecycle, deadline)
                .await
                .unwrap_err()
                .class(),
            WriterFailureClass::Deadline
        );

        let (_lifecycle, lifecycle) = tokio::sync::watch::channel(
            super::WriterLifecycleState::Exited(super::WriterExitClass::Clean),
        );
        assert_eq!(
            super::wait_for_exit(lifecycle, deadline)
                .await
                .unwrap_err()
                .class(),
            WriterFailureClass::Deadline
        );
    }

    #[tokio::test]
    async fn definitive_pre_accept_reply_wins_over_terminal_lifecycle_state() {
        let (reply, receiver) = oneshot::channel();
        reply
            .send(Ok::<super::FlushAck, super::WriterFailure>(super::FlushAck))
            .unwrap();
        let (_lifecycle, lifecycle) = tokio::sync::watch::channel(
            super::WriterLifecycleState::Exited(super::WriterExitClass::Aborted),
        );
        assert_eq!(
            super::await_reply_while_running(receiver, lifecycle)
                .await
                .unwrap(),
            super::FlushAck
        );

        let (reply, receiver) = oneshot::channel::<Result<super::FlushAck, super::WriterFailure>>();
        drop(reply);
        let (_lifecycle, lifecycle) = tokio::sync::watch::channel(
            super::WriterLifecycleState::Exited(super::WriterExitClass::Panicked),
        );
        assert_eq!(
            super::await_reply_while_running(receiver, lifecycle)
                .await
                .unwrap_err()
                .class(),
            WriterFailureClass::Panicked
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sqlite_contention_does_not_block_reservation_or_abort_callers() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let activated = LedgerRepository::activate(&config(&path)).unwrap();
        let blocker = rusqlite::Connection::open(&path).unwrap();
        blocker.execute_batch("BEGIN IMMEDIATE").unwrap();
        let (mut owner, client) = LedgerWriterOwner::start(activated.repository, 4).unwrap();
        let event = LedgerHealthEvent::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            None,
            None,
            "router.writer.contention",
            LedgerHealthSeverity::Warning,
            CREATED_AT,
        )
        .unwrap();
        let waiting_client = client.clone();
        let write = tokio::spawn(async move {
            waiting_client
                .append_health_event_until(event, Instant::now() + Duration::from_secs(10))
                .await
        });
        tokio::time::sleep(Duration::from_millis(50)).await;

        let reserve_started = Instant::now();
        drop(client.try_reserve_pre_accept().unwrap());
        assert!(reserve_started.elapsed() < Duration::from_millis(250));
        let abort_started = Instant::now();
        owner.abort();
        assert!(abort_started.elapsed() < Duration::from_millis(250));

        blocker.execute_batch("ROLLBACK").unwrap();
        assert_eq!(
            write.await.unwrap().unwrap_err().class(),
            WriterFailureClass::Aborted
        );
        tokio::time::timeout(Duration::from_secs(1), async {
            loop {
                if matches!(
                    owner.lifecycle(),
                    super::WriterLifecycleState::Exited(super::WriterExitClass::Aborted)
                ) {
                    break;
                }
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        assert_eq!(
            owner.join_exited().unwrap_err().class(),
            WriterFailureClass::Aborted
        );
    }

    #[test]
    fn writer_capacity_uses_candidate_slots_not_batch_slots() {
        let temporary = tempdir().unwrap();
        let mut config = config(&temporary.path().join("router.db"));
        config.pools[0].concurrency.max_pending = 2;
        config.pools[0].max_candidates_per_sample = 3;
        assert_eq!(config.writer_command_capacity().unwrap(), 280);
    }
}
