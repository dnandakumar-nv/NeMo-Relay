// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed commands and sanitized acknowledgements for the ledger writer.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

use tokio::sync::oneshot;

use super::model::{LedgerError, LedgerErrorClass};
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
use crate::control::{ControlTransactionFence, RouterControlError};
use crate::decision_audit::DecisionAuditV1;
use crate::inspection::InspectionError;
use crate::sqlite_vector_store::{VectorIndexWriterAck, VectorIndexWriterCommand};

/// Stable failure classes exposed by the bounded writer transport.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WriterFailureClass {
    /// The nonblocking pre-accept channel reservation found no capacity.
    #[allow(dead_code)] // Task 7 maps this exact pre-accept refusal into sink pressure.
    Full,
    /// Writer admission is closing or already closed.
    Closing,
    /// The caller's absolute deadline expired.
    Deadline,
    /// The writer exited before acknowledging the command.
    Exited,
    /// The writer thread panicked.
    Panicked,
    /// Synchronous abort invalidated the command before its transaction began.
    Aborted,
    /// A command/reply pairing violated the internal protocol.
    Protocol,
    /// The sole writer repository rejected a typed operation.
    Repository(LedgerErrorClass),
}

impl WriterFailureClass {
    /// Stable non-secret code suitable for health reporting.
    pub(crate) const fn code(self) -> &'static str {
        match self {
            Self::Full => "router.writer.full",
            Self::Closing => "router.writer.closing",
            Self::Deadline => "router.writer.deadline",
            Self::Exited => "router.writer.exited",
            Self::Panicked => "router.writer.panicked",
            Self::Aborted => "router.writer.aborted",
            Self::Protocol => "router.writer.protocol",
            Self::Repository(class) => class.code(),
        }
    }
}

/// Sanitized writer error that never retains SQL, paths, or provider content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct WriterFailure {
    class: WriterFailureClass,
}

impl WriterFailure {
    pub(crate) const fn new(class: WriterFailureClass) -> Self {
        Self { class }
    }

    #[allow(dead_code)] // Focused transport tests and later health mapping use the class.
    pub(crate) const fn class(self) -> WriterFailureClass {
        self.class
    }

    pub(crate) const fn code(self) -> &'static str {
        self.class.code()
    }
}

impl From<WriterFailureClass> for WriterFailure {
    fn from(class: WriterFailureClass) -> Self {
        Self::new(class)
    }
}

impl fmt::Display for WriterFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self.class {
            WriterFailureClass::Full => "Router ledger writer capacity is full",
            WriterFailureClass::Closing => "Router ledger writer is closing",
            WriterFailureClass::Deadline => "Router ledger writer deadline expired",
            WriterFailureClass::Exited => "Router ledger writer exited",
            WriterFailureClass::Panicked => "Router ledger writer failed",
            WriterFailureClass::Aborted => "Router ledger writer was aborted",
            WriterFailureClass::Protocol => "Router ledger writer protocol failed",
            WriterFailureClass::Repository(_) => "Router ledger writer repository operation failed",
        })
    }
}

impl Error for WriterFailure {}

impl From<LedgerError> for WriterFailure {
    fn from(error: LedgerError) -> Self {
        Self::new(WriterFailureClass::Repository(error.class()))
    }
}

/// FIFO proof that every command sent before this command was handled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FlushAck;

/// Proof that writer admission closed and every issued permit was resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DrainAck;

#[allow(dead_code)] // Task 7 and Task 8 consume the typed domain commands through adapters.
pub(super) enum LedgerWriterCommand {
    ProbePendingAnchor {
        pending: Box<FrozenPendingAnchorV1>,
        reply: oneshot::Sender<Result<AnchorProbe, WriterFailure>>,
    },
    RecordPendingAnchor {
        pending: Box<FrozenPendingAnchorV1>,
        reply: oneshot::Sender<Result<AnchorCommandAck, WriterFailure>>,
    },
    RecordPendingAnchorWithCapacity {
        pending: Box<FrozenPendingAnchorV1>,
        max_evidence_records: u64,
        reply: oneshot::Sender<Result<PendingAnchorCapacityAck, WriterFailure>>,
    },
    RecordNotScheduledQueueFull {
        decline: Box<NotScheduledQueueFull>,
        max_evidence_records: Option<u64>,
        reply: oneshot::Sender<Result<NotScheduledQueueFullAck, WriterFailure>>,
    },
    RecordTerminalAnchor {
        terminal: Box<FrozenTerminalAnchorV1>,
        reply: oneshot::Sender<Result<AnchorCommandAck, WriterFailure>>,
    },
    ReserveSampleBatch {
        reservation: Box<SampleBatchReservation>,
        reply: oneshot::Sender<Result<ShadowCommandAck, WriterFailure>>,
    },
    StartShadowAttempt {
        command: ShadowAttemptStarted,
        reply: oneshot::Sender<Result<ShadowCommandAck, WriterFailure>>,
    },
    RecordShadowTerminal {
        command: Box<ShadowTerminalRecord>,
        reply: oneshot::Sender<Result<ShadowCommandAck, WriterFailure>>,
    },
    RecordJudgeAttemptStart {
        start: Box<JudgeAttemptStart>,
        reply: oneshot::Sender<Result<JudgeRecordAck, WriterFailure>>,
    },
    RecordJudgeAttemptTerminal {
        terminal: Box<JudgeAttemptTerminal>,
        reply: oneshot::Sender<Result<JudgeRecordAck, WriterFailure>>,
    },
    RecordEvaluation {
        evaluation: Box<EvaluationRecord>,
        reply: oneshot::Sender<Result<JudgeRecordAck, WriterFailure>>,
    },
    RecordDecisionAudit {
        audit: Arc<DecisionAuditV1>,
        max_evidence_records: u64,
        conflict_health_event_id: uuid::Uuid,
        reply: oneshot::Sender<Result<DecisionAuditAck, WriterFailure>>,
    },
    ClaimCandidateDependency {
        identity: CandidateDependencyIdentity,
        operation: DependencyOperation,
        reply: oneshot::Sender<Result<DependencyCommandAck, WriterFailure>>,
    },
    ClaimJudgeDependency {
        identity: JudgeDependencyIdentity,
        operation: DependencyOperation,
        reply: oneshot::Sender<Result<DependencyCommandAck, WriterFailure>>,
    },
    CompleteDependency {
        completion: DependencyCompletion,
        reply: oneshot::Sender<Result<DependencyCommandAck, WriterFailure>>,
    },
    PrepareLiveEmbedding {
        command: Box<LiveEmbeddingPrepare>,
        reply: oneshot::Sender<Result<LiveEmbeddingPrepareAck, WriterFailure>>,
    },
    CreateEmbeddingJob {
        command: EmbeddingJobCreate,
        reply: oneshot::Sender<Result<EmbeddingJobCreateAck, WriterFailure>>,
    },
    ClaimEmbeddingJobBatch {
        command: EmbeddingJobBatchClaim,
        reply: oneshot::Sender<Result<EmbeddingJobBatchClaimAck, WriterFailure>>,
    },
    CompleteEmbeddingJobBatch {
        command: EmbeddingJobBatchCompletion,
        reply: oneshot::Sender<Result<EmbeddingJobBatchCompletionAck, WriterFailure>>,
    },
    ResolveEmbeddingJob {
        command: EmbeddingJobResolution,
        reply: oneshot::Sender<Result<EmbeddingJobResolutionAck, WriterFailure>>,
    },
    ResetEmbeddingJob {
        command: EmbeddingJobReset,
        reply: oneshot::Sender<Result<EmbeddingJobResetAck, WriterFailure>>,
    },
    ClaimMaterialization {
        command: MaterializationClaim,
        reply: oneshot::Sender<Result<MaterializationClaimAck, WriterFailure>>,
    },
    CompleteMaterialization {
        command: MaterializationCompletion,
        reply: oneshot::Sender<Result<MaterializationCompletionAck, WriterFailure>>,
    },
    ResolveMaterialization {
        command: MaterializationResolution,
        reply: oneshot::Sender<Result<MaterializationResolutionAck, WriterFailure>>,
    },
    PropagateMaterializationFailure {
        command: Box<MaterializationFailurePropagation>,
        reply: oneshot::Sender<Result<MaterializationFailurePropagationAck, WriterFailure>>,
    },
    BackfillVectorGraph {
        command: Box<VectorBackfillCommand>,
        reply: oneshot::Sender<Result<VectorBackfillAck, WriterFailure>>,
    },
    VectorIndex {
        command: Box<VectorIndexWriterCommand>,
        reply: oneshot::Sender<Result<VectorIndexWriterAck, WriterFailure>>,
    },
    EnsureVectorRegistry {
        ensure: Box<VectorRegistryEnsure>,
        reply: oneshot::Sender<Result<RegistryEnsureAck, WriterFailure>>,
    },
    CreateActiveExperiment {
        create: Box<ActiveExperimentCreate>,
        reply: oneshot::Sender<Result<ActiveExperimentCreateAck, WriterFailure>>,
    },
    ClaimActiveLook {
        request: ActiveLookClaimRequest,
        reply: oneshot::Sender<Result<ActiveLookClaimAck, WriterFailure>>,
    },
    RenewActiveLookLease {
        renewal: ActiveLookLeaseRenewal,
        reply: oneshot::Sender<Result<ActiveLookLeaseAck, WriterFailure>>,
    },
    CommitActiveLook {
        commit: ActiveLookCommit,
        reply: oneshot::Sender<Result<ActiveLookCommitAck, WriterFailure>>,
    },
    FailActiveLook {
        failure: ActiveLookFailure,
        reply: oneshot::Sender<Result<ActiveLookFailureAck, WriterFailure>>,
    },
    ObserveActiveAuthorization {
        active_experiment_id: uuid::Uuid,
        observed_at_unix_ms: i64,
        reply: oneshot::Sender<Result<ActiveAuthorizationObserveAck, WriterFailure>>,
    },
    PromoteActiveNeighborhood {
        promotion: Box<ActiveNeighborhoodPromotion>,
        reply: oneshot::Sender<Result<ActiveNeighborhoodMutationAck, WriterFailure>>,
    },
    InvalidateActiveNeighborhood {
        invalidation: Box<ActiveNeighborhoodInvalidation>,
        reply: oneshot::Sender<Result<ActiveNeighborhoodMutationAck, WriterFailure>>,
    },
    ObserveActiveNeighborhood {
        key: Box<ActiveNeighborhoodKey>,
        observed_at_unix_ms: i64,
        reply: oneshot::Sender<Result<ActiveNeighborhoodObserveAck, WriterFailure>>,
    },
    AdmitActiveRoot {
        admission: Box<ActiveRootAdmission>,
        reply: oneshot::Sender<Result<ActiveAdmissionAck, WriterFailure>>,
    },
    AdmitActiveDecision {
        admission: Box<ActiveDecisionAdmissionV2>,
        reply: oneshot::Sender<Result<ActiveDecisionAdmissionAckV2, WriterFailure>>,
    },
    AppendActiveSignals {
        batch: Box<ActiveSignalBatch>,
        reply: oneshot::Sender<Result<ActiveSignalBatchAck, WriterFailure>>,
    },
    RecordActiveDispatchTerminal {
        terminal: ActiveDispatchTerminal,
        reply: oneshot::Sender<Result<ActiveDispatchTerminalAck, WriterFailure>>,
    },
    TerminalizeActiveRoot {
        terminal: ActiveRootTerminal,
        reply: oneshot::Sender<Result<ActiveRootTerminalAck, WriterFailure>>,
    },
    RunRetention {
        request: RetentionRequest,
        reply: oneshot::Sender<Result<RetentionAck, WriterFailure>>,
    },
    RenewHeartbeat {
        renewal: HeartbeatRenewal,
        reply: oneshot::Sender<Result<HeartbeatAck, WriterFailure>>,
    },
    StopProcess {
        command: ProcessStop,
        reply: oneshot::Sender<Result<ProcessCommandAck, WriterFailure>>,
    },
    AppendHealthEvent {
        event: LedgerHealthEvent,
        reply: oneshot::Sender<Result<ProcessCommandAck, WriterFailure>>,
    },
    ApplyControlMutation {
        mutation: PreparedControlMutation,
        fence: Arc<ControlTransactionFence>,
        reply:
            oneshot::Sender<Result<Result<ControlMutationAck, RouterControlError>, WriterFailure>>,
    },
    ApplyOperatorMutation {
        mutation: PreparedOperatorMutation,
        fence: Arc<ControlTransactionFence>,
        reply: oneshot::Sender<
            Result<Result<OperatorMutationTransactionAck, InspectionError>, WriterFailure>,
        >,
    },
    Flush {
        reply: oneshot::Sender<Result<FlushAck, WriterFailure>>,
    },
    Drain {
        reply: oneshot::Sender<Result<DrainAck, WriterFailure>>,
    },
    #[cfg(test)]
    Pause {
        started: std::sync::mpsc::SyncSender<()>,
        release: std::sync::mpsc::Receiver<()>,
        reply: oneshot::Sender<Result<(), WriterFailure>>,
    },
    #[cfg(test)]
    Panic {
        reply: oneshot::Sender<Result<(), WriterFailure>>,
    },
}

impl LedgerWriterCommand {
    pub(super) fn fail(self, failure: WriterFailure) {
        match self {
            Self::ProbePendingAnchor { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::RecordPendingAnchor { reply, .. } | Self::RecordTerminalAnchor { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::RecordPendingAnchorWithCapacity { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::RecordNotScheduledQueueFull { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::ReserveSampleBatch { reply, .. }
            | Self::StartShadowAttempt { reply, .. }
            | Self::RecordShadowTerminal { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::RecordJudgeAttemptStart { reply, .. }
            | Self::RecordJudgeAttemptTerminal { reply, .. }
            | Self::RecordEvaluation { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::RecordDecisionAudit { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::ClaimCandidateDependency { reply, .. }
            | Self::ClaimJudgeDependency { reply, .. }
            | Self::CompleteDependency { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::PrepareLiveEmbedding { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::CreateEmbeddingJob { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::ClaimEmbeddingJobBatch { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::CompleteEmbeddingJobBatch { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::ResolveEmbeddingJob { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::ResetEmbeddingJob { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::ClaimMaterialization { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::CompleteMaterialization { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::ResolveMaterialization { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::PropagateMaterializationFailure { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::BackfillVectorGraph { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::VectorIndex { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::EnsureVectorRegistry { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::CreateActiveExperiment { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::ClaimActiveLook { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::RenewActiveLookLease { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::CommitActiveLook { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::FailActiveLook { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::ObserveActiveAuthorization { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::PromoteActiveNeighborhood { reply, .. }
            | Self::InvalidateActiveNeighborhood { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::ObserveActiveNeighborhood { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::AdmitActiveRoot { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::AdmitActiveDecision { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::AppendActiveSignals { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::RecordActiveDispatchTerminal { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::TerminalizeActiveRoot { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::RunRetention { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::RenewHeartbeat { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::StopProcess { reply, .. } | Self::AppendHealthEvent { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::ApplyControlMutation { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::ApplyOperatorMutation { reply, .. } => {
                let _ = reply.send(Err(failure));
            }
            Self::Flush { reply } => {
                let _ = reply.send(Err(failure));
            }
            Self::Drain { reply } => {
                let _ = reply.send(Err(failure));
            }
            #[cfg(test)]
            Self::Pause { reply, .. } | Self::Panic { reply } => {
                let _ = reply.send(Err(failure));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{WriterFailure, WriterFailureClass};
    use crate::ledger::model::LedgerErrorClass;

    #[test]
    fn writer_failures_expose_only_stable_codes_and_messages() {
        for class in [
            WriterFailureClass::Full,
            WriterFailureClass::Closing,
            WriterFailureClass::Deadline,
            WriterFailureClass::Exited,
            WriterFailureClass::Panicked,
            WriterFailureClass::Aborted,
            WriterFailureClass::Protocol,
            WriterFailureClass::Repository(LedgerErrorClass::Busy),
        ] {
            let failure = WriterFailure::new(class);
            assert_eq!(failure.class(), class);
            let expected_prefix = if matches!(class, WriterFailureClass::Repository(_)) {
                "router.ledger."
            } else {
                "router.writer."
            };
            assert!(failure.code().starts_with(expected_prefix));
            assert!(!failure.to_string().contains('/'));
            assert!(!failure.to_string().to_ascii_lowercase().contains("select"));
        }
    }

    #[test]
    fn repository_errors_preserve_only_the_stable_ledger_class() {
        for ledger_class in [
            LedgerErrorClass::Busy,
            LedgerErrorClass::InvalidPermissions,
            LedgerErrorClass::IdentityInvariant,
            LedgerErrorClass::DatabaseOperationFailed,
        ] {
            let failure =
                WriterFailure::from(crate::ledger::model::LedgerError::from(ledger_class));
            assert_eq!(
                failure.class(),
                WriterFailureClass::Repository(ledger_class)
            );
            assert_eq!(failure.code(), ledger_class.code());
            assert_eq!(
                failure.to_string(),
                "Router ledger writer repository operation failed"
            );
        }
    }
}
