// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SQLite-backed trajectory persistence and scheduler handoff.

use std::collections::{BTreeMap, VecDeque, btree_map::Entry};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use chrono::Utc;
use tokio::sync::{Mutex as AsyncMutex, mpsc};
use uuid::Uuid;

use crate::config::RouterConfig;
use crate::ledger::command::WriterFailure;
use crate::ledger::repository::anchors::{
    AnchorCommandAck, AnchorProbe, FrozenPendingAnchorV1, FrozenTerminalAnchorV1,
    NotScheduledQueueFull, NotScheduledQueueFullAck, PendingAnchorCapacityAck,
};
use crate::ledger::repository::shadow::{
    ReservedShadowAttempt, SampleBatchReservation, ShadowCommandAck,
};
use crate::ledger::writer::LedgerWriterClient;
use crate::projection::candidate_request_projection;
use crate::scheduler_admission::{
    SchedulerAdmission, SchedulerAdmissionError, SchedulerAdmissionPools,
};
use crate::sink::{
    DeliveryAck, DeliveryFailureClass, DeliveryFuture, DurableDeclineReason, SinkAck,
    SinkFailureClass, SinkFuture, TrajectoryDelivery, TrajectorySink, TransientRefusalReason,
};
use crate::trajectory::{
    ClosedTrajectoryWindow, PendingTrajectoryWindow, PersistedCandidateFactV1,
    PersistedTrajectoryTerminalV1, TrajectoryTerminalStateV1,
};

const WRITE_TIMEOUT: Duration = Duration::from_secs(5);

/// One capacity-accounted batch transferred to the scheduler actor.
pub(crate) struct ScheduledTrajectoryBatch {
    pub(crate) window: Arc<ClosedTrajectoryWindow>,
    pub(crate) reservation: SampleBatchReservation,
    pub(crate) admission: SchedulerAdmission,
}

/// Production SQLite implementation of both acknowledged trajectory contracts.
#[derive(Clone)]
pub(crate) struct SqliteTrajectorySink {
    inner: Arc<SqliteTrajectorySinkInner>,
}

/// Fully validated, side-effect-free production sink construction state.
pub(crate) struct SqliteTrajectorySinkPlan {
    admissions: SchedulerAdmissionPools,
    channel_capacity: usize,
    max_evidence_records: u64,
    evaluator_versions: BTreeMap<String, String>,
}

struct SqliteTrajectorySinkInner {
    writer: LedgerWriterClient,
    admissions: SchedulerAdmissionPools,
    max_evidence_records: u64,
    evaluator_versions: BTreeMap<String, String>,
    scheduler_tx: mpsc::Sender<ScheduledTrajectoryBatch>,
    state: Mutex<SinkState>,
    operation_serial: AsyncMutex<()>,
    delivered_receipt_capacity: usize,
    #[cfg(test)]
    pending_probe_hook: Mutex<Option<PendingProbeHook>>,
}

#[cfg(test)]
struct PendingProbeHook {
    reached: tokio::sync::oneshot::Sender<()>,
    resume: tokio::sync::oneshot::Receiver<()>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReservationIdentity {
    pending_hash: String,
    pool_id: String,
    policy_version_id: String,
    candidate_count: usize,
}

struct ActiveEntry {
    identity: ReservationIdentity,
    admission: Option<SchedulerAdmission>,
    pending: FrozenPendingAnchorV1,
    stage: ReservationStage,
}

struct CompletedEntry {
    identity: ReservationIdentity,
    terminal_hash: String,
    kind: CompletionKind,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum CompletionKind {
    Rejected,
    Transferred,
}

enum ReservationEntry {
    Active(Box<ActiveEntry>),
    Completed(CompletedEntry),
}

enum ReservationStage {
    PendingWrite,
    PendingDurable,
    TerminalWrite {
        terminal: FrozenTerminal,
        rejected: bool,
    },
    ClosedDurable {
        terminal: FrozenTerminal,
        reservation: Option<SampleBatchReservation>,
    },
}

#[derive(Clone)]
struct FrozenTerminal {
    payload_hash: String,
    command: FrozenTerminalAnchorV1,
}

#[derive(Default)]
struct SinkState {
    entries: BTreeMap<Uuid, ReservationEntry>,
    rejected_order: VecDeque<Uuid>,
    transferred_order: VecDeque<Uuid>,
}

impl SqliteTrajectorySink {
    /// Validate every fallible sink invariant before production work is spawned.
    pub(crate) fn prepare(
        config: &RouterConfig,
    ) -> Result<SqliteTrajectorySinkPlan, SinkFailureClass> {
        let admissions = SchedulerAdmissionPools::from_config(config)
            .map_err(|_| SinkFailureClass::InvalidReservation)?;
        let channel_capacity = admissions.total_batch_capacity();
        if channel_capacity == 0 {
            return Err(SinkFailureClass::InvalidReservation);
        }
        let evaluator_versions = config
            .pools
            .iter()
            .map(|pool| {
                pool.judge
                    .evaluator_version()
                    .map(|version| (pool.id.clone(), version))
                    .map_err(|_| SinkFailureClass::CanonicalizationFailed)
            })
            .collect::<Result<BTreeMap<_, _>, _>>()?;
        Ok(SqliteTrajectorySinkPlan {
            admissions,
            channel_capacity,
            max_evidence_records: config.max_evidence_records,
            evaluator_versions,
        })
    }

    /// Build exact scheduler capacity and its matching bounded handoff channel.
    #[cfg(test)]
    pub(crate) fn new(
        config: &RouterConfig,
        writer: LedgerWriterClient,
    ) -> Result<(Self, mpsc::Receiver<ScheduledTrajectoryBatch>), SinkFailureClass> {
        Ok(Self::prepare(config)?.start(writer))
    }
}

impl SqliteTrajectorySinkPlan {
    /// Consume prevalidated state without another fallible configuration step.
    pub(crate) fn start(
        self,
        writer: LedgerWriterClient,
    ) -> (
        SqliteTrajectorySink,
        mpsc::Receiver<ScheduledTrajectoryBatch>,
    ) {
        let (scheduler_tx, scheduler_rx) = mpsc::channel(self.channel_capacity);
        let inner = Arc::new(SqliteTrajectorySinkInner {
            writer,
            admissions: self.admissions,
            max_evidence_records: self.max_evidence_records,
            evaluator_versions: self.evaluator_versions,
            scheduler_tx,
            state: Mutex::new(SinkState::default()),
            operation_serial: AsyncMutex::new(()),
            delivered_receipt_capacity: self.channel_capacity,
            #[cfg(test)]
            pending_probe_hook: Mutex::new(None),
        });
        (SqliteTrajectorySink { inner }, scheduler_rx)
    }
}

impl SqliteTrajectorySink {
    pub(crate) fn admission_pools(&self) -> SchedulerAdmissionPools {
        self.inner.admissions.clone()
    }

    #[cfg(test)]
    fn install_pending_probe_hook(
        &self,
        reached: tokio::sync::oneshot::Sender<()>,
        resume: tokio::sync::oneshot::Receiver<()>,
    ) {
        *lock_unpoisoned(&self.inner.pending_probe_hook) =
            Some(PendingProbeHook { reached, resume });
    }

    #[cfg(test)]
    async fn wait_after_pending_probe(&self) {
        let hook = lock_unpoisoned(&self.inner.pending_probe_hook).take();
        if let Some(PendingProbeHook { reached, resume }) = hook {
            let _ = reached.send(());
            let _ = resume.await;
        }
    }

    async fn record_pending_impl(&self, pending: Arc<PendingTrajectoryWindow>) -> SinkAck {
        let _serial = self.inner.operation_serial.lock().await;
        let anchor_id = pending.anchor_id();
        let identity = match ReservationIdentity::from_pending(&pending) {
            Ok(identity) => identity,
            Err(class) => {
                return SinkAck::Failed {
                    anchor_id,
                    stable_class: class,
                };
            }
        };
        let provisional = {
            let state = lock_unpoisoned(&self.inner.state);
            match state.entries.get(&anchor_id) {
                Some(ReservationEntry::Active(active)) if active.identity == identity => {
                    match &active.stage {
                        ReservationStage::PendingWrite => Some(active.pending.clone()),
                        ReservationStage::PendingDurable
                        | ReservationStage::TerminalWrite { .. } => {
                            return SinkAck::AlreadyApplied {
                                anchor_id,
                                payload_hash: identity.pending_hash,
                            };
                        }
                        ReservationStage::ClosedDurable { terminal, .. } => {
                            return SinkAck::AlreadyTerminal {
                                anchor_id,
                                pending_hash: identity.pending_hash,
                                terminal_hash: terminal.payload_hash.clone(),
                            };
                        }
                    }
                }
                Some(ReservationEntry::Completed(completed)) if completed.identity == identity => {
                    return SinkAck::AlreadyTerminal {
                        anchor_id,
                        pending_hash: identity.pending_hash,
                        terminal_hash: completed.terminal_hash.clone(),
                    };
                }
                Some(ReservationEntry::Active(_) | ReservationEntry::Completed(_)) => {
                    return SinkAck::Conflict { anchor_id };
                }
                None => None,
            }
        };
        if let Some(frozen) = provisional {
            let permit = match self
                .inner
                .writer
                .reserve_accepted_until(write_deadline())
                .await
            {
                Ok(permit) => permit,
                Err(error) => return sink_writer_failure(anchor_id, error),
            };
            return self
                .persist_pending(anchor_id, identity, frozen, permit)
                .await;
        }

        let created_at_unix_ms = Utc::now()
            .timestamp_millis()
            .max(pending.opened_at.timestamp_millis());
        let frozen = match FrozenPendingAnchorV1::new(
            &pending,
            Uuid::now_v7(),
            Uuid::now_v7(),
            created_at_unix_ms,
        ) {
            Ok(frozen) => frozen,
            Err(_) => {
                return SinkAck::Failed {
                    anchor_id,
                    stable_class: SinkFailureClass::CanonicalizationFailed,
                };
            }
        };
        let probe_permit = match self.inner.writer.try_reserve_pre_accept() {
            Ok(permit) => permit,
            Err(error) => return sink_writer_failure(anchor_id, error),
        };
        let probe = match self
            .inner
            .writer
            .probe_pending_anchor_with_permit(probe_permit, frozen.clone())
            .await
        {
            Ok(probe) => probe,
            Err(error) => return sink_writer_failure(anchor_id, error),
        };
        match probe {
            AnchorProbe::Missing {
                anchor_id: acknowledged,
            } if acknowledged == anchor_id => {}
            AnchorProbe::Pending {
                anchor_id: acknowledged,
                pending_hash,
            } if acknowledged == anchor_id && pending_hash == identity.pending_hash => {}
            AnchorProbe::Terminal {
                anchor_id: acknowledged,
                pending_hash,
                terminal_hash,
            } if acknowledged == anchor_id && pending_hash == identity.pending_hash => {
                return SinkAck::AlreadyTerminal {
                    anchor_id,
                    pending_hash,
                    terminal_hash,
                };
            }
            AnchorProbe::NotScheduledQueueFull {
                anchor_id: acknowledged,
                pending_hash,
                terminal_hash,
            } if acknowledged == anchor_id && pending_hash == identity.pending_hash => {
                return SinkAck::DurablyDeclined {
                    anchor_id,
                    pending_hash,
                    terminal_hash,
                    reason: DurableDeclineReason::NotScheduledQueueFull,
                };
            }
            AnchorProbe::Conflict {
                anchor_id: acknowledged,
            } if acknowledged == anchor_id => return SinkAck::Conflict { anchor_id },
            AnchorProbe::OriginatingProcessNotLive {
                anchor_id: acknowledged,
            } if acknowledged == anchor_id => {
                return SinkAck::Failed {
                    anchor_id,
                    stable_class: SinkFailureClass::RepositoryRejected,
                };
            }
            AnchorProbe::TransactionNotStarted { .. } => {
                return SinkAck::Failed {
                    anchor_id,
                    stable_class: SinkFailureClass::RepositoryRejected,
                };
            }
            _ => return SinkAck::Conflict { anchor_id },
        }
        #[cfg(test)]
        self.wait_after_pending_probe().await;

        let write_permit = match self.inner.writer.try_reserve_pre_accept() {
            Ok(permit) => permit,
            Err(error) => return sink_writer_failure(anchor_id, error),
        };
        let admission = match self
            .inner
            .admissions
            .try_acquire(&identity.pool_id, identity.candidate_count)
        {
            Ok(admission) => admission,
            Err(SchedulerAdmissionError::NoPermits) => {
                let decline =
                    match NotScheduledQueueFull::new(frozen, Uuid::now_v7(), created_at_unix_ms) {
                        Ok(decline) => decline,
                        Err(_) => {
                            return SinkAck::Failed {
                                anchor_id,
                                stable_class: SinkFailureClass::CanonicalizationFailed,
                            };
                        }
                    };
                return self.record_queue_full(write_permit, decline).await;
            }
            Err(SchedulerAdmissionError::Closed) => {
                return SinkAck::Failed {
                    anchor_id,
                    stable_class: SinkFailureClass::SchedulerClosed,
                };
            }
            Err(
                SchedulerAdmissionError::InvalidPool
                | SchedulerAdmissionError::InvalidCandidateCount,
            ) => {
                return SinkAck::Failed {
                    anchor_id,
                    stable_class: SinkFailureClass::InvalidReservation,
                };
            }
        };
        {
            let mut state = lock_unpoisoned(&self.inner.state);
            match state.entries.entry(anchor_id) {
                Entry::Vacant(entry) => {
                    entry.insert(ReservationEntry::Active(Box::new(ActiveEntry {
                        identity: identity.clone(),
                        admission: Some(admission),
                        pending: frozen.clone(),
                        stage: ReservationStage::PendingWrite,
                    })));
                }
                Entry::Occupied(_) => {
                    drop(admission);
                    return SinkAck::Conflict { anchor_id };
                }
            }
        }
        self.persist_pending(anchor_id, identity, frozen, write_permit)
            .await
    }

    async fn record_queue_full(
        &self,
        permit: crate::ledger::writer::WriterCommandPermit,
        decline: NotScheduledQueueFull,
    ) -> SinkAck {
        let anchor_id = decline.anchor_id();
        let pending_hash = decline.pending_hash().to_string();
        let terminal_hash = decline.terminal_hash().to_string();
        match self
            .inner
            .writer
            .record_not_scheduled_queue_full_with_capacity_and_permit(
                permit,
                decline,
                self.inner.max_evidence_records,
            )
            .await
        {
            Ok(acknowledgement) => {
                map_queue_full_ack(acknowledgement, anchor_id, &pending_hash, &terminal_hash)
            }
            Err(error) => sink_writer_failure(anchor_id, error),
        }
    }

    async fn persist_pending(
        &self,
        anchor_id: Uuid,
        identity: ReservationIdentity,
        frozen: FrozenPendingAnchorV1,
        permit: crate::ledger::writer::WriterCommandPermit,
    ) -> SinkAck {
        let acknowledgement = match self
            .inner
            .writer
            .record_pending_anchor_with_capacity_and_permit(
                permit,
                frozen,
                self.inner.max_evidence_records,
            )
            .await
        {
            Ok(acknowledgement) => acknowledgement,
            Err(error) => {
                self.discard_pending_write(anchor_id, &identity);
                return sink_writer_failure(anchor_id, error);
            }
        };
        self.finish_pending_write(anchor_id, identity, acknowledgement)
    }

    fn finish_pending_write(
        &self,
        expected_anchor_id: Uuid,
        identity: ReservationIdentity,
        acknowledgement: PendingAnchorCapacityAck,
    ) -> SinkAck {
        let anchor_id = acknowledgement.anchor_id();
        if anchor_id != expected_anchor_id {
            self.discard_pending_write(expected_anchor_id, &identity);
            return SinkAck::Conflict {
                anchor_id: expected_anchor_id,
            };
        }
        match acknowledgement {
            PendingAnchorCapacityAck::Applied {
                anchor_id,
                canonical_payload_hash,
            } if canonical_payload_hash == identity.pending_hash => {
                self.mark_pending_durable(anchor_id, identity, true)
            }
            PendingAnchorCapacityAck::AlreadyApplied {
                anchor_id,
                canonical_payload_hash,
            } if canonical_payload_hash == identity.pending_hash => {
                self.mark_pending_durable(anchor_id, identity, false)
            }
            PendingAnchorCapacityAck::AlreadyTerminal {
                anchor_id,
                pending_hash,
                terminal_hash,
            } if pending_hash == identity.pending_hash => {
                self.discard_pending_write(anchor_id, &identity);
                SinkAck::AlreadyTerminal {
                    anchor_id,
                    pending_hash,
                    terminal_hash,
                }
            }
            PendingAnchorCapacityAck::NotScheduledQueueFull {
                anchor_id,
                pending_hash,
                terminal_hash,
            } if pending_hash == identity.pending_hash => {
                self.discard_pending_write(anchor_id, &identity);
                SinkAck::DurablyDeclined {
                    anchor_id,
                    pending_hash,
                    terminal_hash,
                    reason: DurableDeclineReason::NotScheduledQueueFull,
                }
            }
            PendingAnchorCapacityAck::EvidenceCapacity { anchor_id } => {
                self.discard_pending_write(anchor_id, &identity);
                SinkAck::TransientRefused {
                    anchor_id,
                    reason: TransientRefusalReason::EvidenceCapacity,
                }
            }
            PendingAnchorCapacityAck::Conflict { anchor_id } => {
                self.discard_pending_write(anchor_id, &identity);
                SinkAck::Conflict { anchor_id }
            }
            PendingAnchorCapacityAck::OriginatingProcessNotLive { anchor_id }
            | PendingAnchorCapacityAck::TransactionNotStarted { anchor_id } => {
                self.discard_pending_write(anchor_id, &identity);
                SinkAck::Failed {
                    anchor_id,
                    stable_class: SinkFailureClass::RepositoryRejected,
                }
            }
            _ => {
                self.discard_pending_write(anchor_id, &identity);
                SinkAck::Conflict { anchor_id }
            }
        }
    }

    fn mark_pending_durable(
        &self,
        anchor_id: Uuid,
        identity: ReservationIdentity,
        applied: bool,
    ) -> SinkAck {
        let payload_hash = identity.pending_hash.clone();
        let mut state = lock_unpoisoned(&self.inner.state);
        match state.entries.get_mut(&anchor_id) {
            Some(ReservationEntry::Active(active))
                if active.identity == identity
                    && matches!(active.stage, ReservationStage::PendingWrite) =>
            {
                active.stage = ReservationStage::PendingDurable;
                if applied {
                    SinkAck::Applied {
                        anchor_id,
                        payload_hash,
                    }
                } else {
                    SinkAck::AlreadyApplied {
                        anchor_id,
                        payload_hash,
                    }
                }
            }
            _ => SinkAck::Failed {
                anchor_id,
                stable_class: SinkFailureClass::MissingPending,
            },
        }
    }

    fn discard_pending_write(&self, anchor_id: Uuid, identity: &ReservationIdentity) {
        let mut state = lock_unpoisoned(&self.inner.state);
        let remove = matches!(
            state.entries.get(&anchor_id),
            Some(ReservationEntry::Active(active))
                if active.identity == *identity
                    && matches!(active.stage, ReservationStage::PendingWrite)
        );
        if remove {
            state.entries.remove(&anchor_id);
        }
    }

    async fn record_terminal_impl(&self, terminal: Arc<PersistedTrajectoryTerminalV1>) -> SinkAck {
        let _serial = self.inner.operation_serial.lock().await;
        let anchor_id = terminal.anchor_id();
        let identity = match ReservationIdentity::from_pending(&terminal.pending) {
            Ok(identity) => identity,
            Err(class) => {
                return SinkAck::Failed {
                    anchor_id,
                    stable_class: class,
                };
            }
        };
        let payload_hash = match terminal.payload_hash() {
            Ok(hash) => hash,
            Err(()) => {
                return SinkAck::Failed {
                    anchor_id,
                    stable_class: SinkFailureClass::CanonicalizationFailed,
                };
            }
        };
        let rejected = matches!(terminal.state, TrajectoryTerminalStateV1::Rejected { .. });
        let existing = {
            let state = lock_unpoisoned(&self.inner.state);
            match state.entries.get(&anchor_id) {
                Some(ReservationEntry::Active(active)) if active.identity == identity => {
                    match &active.stage {
                        ReservationStage::PendingWrite => {
                            return SinkAck::Failed {
                                anchor_id,
                                stable_class: SinkFailureClass::MissingPending,
                            };
                        }
                        ReservationStage::PendingDurable => None,
                        ReservationStage::TerminalWrite {
                            terminal: existing,
                            rejected: existing_rejected,
                        } if existing.payload_hash == payload_hash
                            && *existing_rejected == rejected =>
                        {
                            Some(existing.clone())
                        }
                        ReservationStage::ClosedDurable {
                            terminal: existing, ..
                        } if existing.payload_hash == payload_hash && !rejected => {
                            return SinkAck::AlreadyApplied {
                                anchor_id,
                                payload_hash,
                            };
                        }
                        ReservationStage::TerminalWrite { .. }
                        | ReservationStage::ClosedDurable { .. } => {
                            return SinkAck::Conflict { anchor_id };
                        }
                    }
                }
                Some(ReservationEntry::Completed(completed))
                    if completed.identity == identity
                        && completed.terminal_hash == payload_hash =>
                {
                    return SinkAck::AlreadyApplied {
                        anchor_id,
                        payload_hash,
                    };
                }
                Some(ReservationEntry::Active(_) | ReservationEntry::Completed(_)) => {
                    return SinkAck::Conflict { anchor_id };
                }
                None => {
                    return SinkAck::Failed {
                        anchor_id,
                        stable_class: SinkFailureClass::MissingPending,
                    };
                }
            }
        };
        let frozen = match existing {
            Some(existing) if existing.payload_hash == payload_hash => existing,
            Some(_) => return SinkAck::Conflict { anchor_id },
            None => {
                let command = match FrozenTerminalAnchorV1::new(
                    &terminal,
                    Uuid::now_v7(),
                    Uuid::now_v7(),
                    terminal.closed_at.timestamp_millis(),
                ) {
                    Ok(command) => command,
                    Err(_) => {
                        return SinkAck::Failed {
                            anchor_id,
                            stable_class: SinkFailureClass::CanonicalizationFailed,
                        };
                    }
                };
                let frozen = FrozenTerminal {
                    payload_hash: payload_hash.clone(),
                    command,
                };
                let mut state = lock_unpoisoned(&self.inner.state);
                match state.entries.get_mut(&anchor_id) {
                    Some(ReservationEntry::Active(active))
                        if active.identity == identity
                            && matches!(active.stage, ReservationStage::PendingDurable) =>
                    {
                        active.stage = ReservationStage::TerminalWrite {
                            terminal: frozen.clone(),
                            rejected,
                        };
                    }
                    _ => return SinkAck::Conflict { anchor_id },
                }
                frozen
            }
        };

        let acknowledgement = match self
            .inner
            .writer
            .record_terminal_anchor_until(frozen.command.clone(), write_deadline())
            .await
        {
            Ok(acknowledgement) => acknowledgement,
            Err(error) => return sink_writer_failure(anchor_id, error),
        };
        let applied = match acknowledgement {
            AnchorCommandAck::Applied {
                anchor_id: acknowledged,
                canonical_payload_hash,
            } if acknowledged == anchor_id && canonical_payload_hash == payload_hash => true,
            AnchorCommandAck::AlreadyApplied {
                anchor_id: acknowledged,
                canonical_payload_hash,
            } if acknowledged == anchor_id && canonical_payload_hash == payload_hash => false,
            AnchorCommandAck::Conflict {
                anchor_id: acknowledged,
            } if acknowledged == anchor_id => return SinkAck::Conflict { anchor_id },
            AnchorCommandAck::OriginatingProcessNotLive {
                anchor_id: acknowledged,
            } if acknowledged == anchor_id => {
                return SinkAck::Failed {
                    anchor_id,
                    stable_class: SinkFailureClass::RepositoryRejected,
                };
            }
            AnchorCommandAck::TransactionNotStarted { .. } => {
                return SinkAck::Failed {
                    anchor_id,
                    stable_class: SinkFailureClass::RepositoryRejected,
                };
            }
            _ => return SinkAck::Conflict { anchor_id },
        };
        if rejected {
            self.complete_entry(
                anchor_id,
                identity,
                payload_hash.clone(),
                CompletionKind::Rejected,
            );
        } else {
            let mut state = lock_unpoisoned(&self.inner.state);
            match state.entries.get_mut(&anchor_id) {
                Some(ReservationEntry::Active(active)) if active.identity == identity => {
                    match &active.stage {
                        ReservationStage::TerminalWrite {
                            terminal,
                            rejected: false,
                        } if terminal.payload_hash == payload_hash => {
                            active.stage = ReservationStage::ClosedDurable {
                                terminal: frozen,
                                reservation: None,
                            };
                        }
                        _ => return SinkAck::Conflict { anchor_id },
                    }
                }
                _ => return SinkAck::Conflict { anchor_id },
            }
        }
        if applied {
            SinkAck::Applied {
                anchor_id,
                payload_hash,
            }
        } else {
            SinkAck::AlreadyApplied {
                anchor_id,
                payload_hash,
            }
        }
    }

    fn complete_entry(
        &self,
        anchor_id: Uuid,
        identity: ReservationIdentity,
        terminal_hash: String,
        kind: CompletionKind,
    ) {
        let mut state = lock_unpoisoned(&self.inner.state);
        state.entries.insert(
            anchor_id,
            ReservationEntry::Completed(CompletedEntry {
                identity,
                terminal_hash,
                kind,
            }),
        );
        match kind {
            CompletionKind::Rejected => state.rejected_order.push_back(anchor_id),
            CompletionKind::Transferred => state.transferred_order.push_back(anchor_id),
        }
        trim_completed_receipts(&mut state, self.inner.delivered_receipt_capacity, kind);
    }

    async fn deliver_impl(&self, window: Arc<ClosedTrajectoryWindow>) -> DeliveryAck {
        let _serial = self.inner.operation_serial.lock().await;
        let anchor_id = window.anchor_id();
        let identity = match ReservationIdentity::from_pending(&window.pending) {
            Ok(identity) => identity,
            Err(_) => {
                return delivery_failure(anchor_id, DeliveryFailureClass::InvalidWindow);
            }
        };
        let terminal_hash = match window.terminal().payload_hash() {
            Ok(hash) => hash,
            Err(()) => return delivery_failure(anchor_id, DeliveryFailureClass::InvalidWindow),
        };
        let cached_reservation = {
            let state = lock_unpoisoned(&self.inner.state);
            match state.entries.get(&anchor_id) {
                Some(ReservationEntry::Completed(completed))
                    if completed.identity == identity
                        && completed.terminal_hash == terminal_hash =>
                {
                    return if completed.kind == CompletionKind::Transferred {
                        DeliveryAck::AlreadyDelivered { anchor_id }
                    } else {
                        delivery_failure(anchor_id, DeliveryFailureClass::InvalidWindow)
                    };
                }
                Some(ReservationEntry::Active(active)) if active.identity == identity => {
                    match &active.stage {
                        ReservationStage::ClosedDurable {
                            terminal,
                            reservation,
                        } if terminal.payload_hash == terminal_hash => reservation.clone(),
                        _ => {
                            return delivery_failure(
                                anchor_id,
                                DeliveryFailureClass::InvalidWindow,
                            );
                        }
                    }
                }
                Some(ReservationEntry::Active(_) | ReservationEntry::Completed(_)) => {
                    return delivery_failure(anchor_id, DeliveryFailureClass::InvalidWindow);
                }
                None => {
                    return delivery_failure(anchor_id, DeliveryFailureClass::MissingAdmission);
                }
            }
        };
        let reservation = match cached_reservation {
            Some(reservation) => reservation,
            None => {
                let reservation = match self.build_reservation(&window) {
                    Ok(reservation) => reservation,
                    Err(class) => return delivery_failure(anchor_id, class),
                };
                let mut state = lock_unpoisoned(&self.inner.state);
                match state.entries.get_mut(&anchor_id) {
                    Some(ReservationEntry::Active(active)) if active.identity == identity => {
                        match &mut active.stage {
                            ReservationStage::ClosedDurable {
                                terminal,
                                reservation: cached,
                            } if terminal.payload_hash == terminal_hash => {
                                *cached = Some(reservation.clone());
                            }
                            _ => {
                                return delivery_failure(
                                    anchor_id,
                                    DeliveryFailureClass::InvalidWindow,
                                );
                            }
                        }
                    }
                    _ => {
                        return delivery_failure(anchor_id, DeliveryFailureClass::InvalidWindow);
                    }
                }
                reservation
            }
        };
        match self
            .inner
            .writer
            .reserve_sample_batch_until(reservation.clone(), write_deadline())
            .await
        {
            Ok(ShadowCommandAck::Applied | ShadowCommandAck::AlreadyApplied) => {}
            Ok(ShadowCommandAck::Conflict) => {
                return delivery_failure(anchor_id, DeliveryFailureClass::RepositoryConflict);
            }
            Ok(
                ShadowCommandAck::OriginatingProcessNotLive
                | ShadowCommandAck::TransactionNotStarted,
            ) => {
                return delivery_failure(anchor_id, DeliveryFailureClass::RepositoryRejected);
            }
            Err(error) => {
                return delivery_failure(anchor_id, DeliveryFailureClass::Writer(error.class()));
            }
        }

        let admission = {
            let mut state = lock_unpoisoned(&self.inner.state);
            match state.entries.get_mut(&anchor_id) {
                Some(ReservationEntry::Active(active)) if active.identity == identity => {
                    match active.admission.take() {
                        Some(admission) => admission,
                        None => {
                            return delivery_failure(
                                anchor_id,
                                DeliveryFailureClass::MissingAdmission,
                            );
                        }
                    }
                }
                _ => return delivery_failure(anchor_id, DeliveryFailureClass::InvalidWindow),
            }
        };
        let batch = ScheduledTrajectoryBatch {
            window,
            reservation: reservation.clone(),
            admission,
        };
        match self.inner.scheduler_tx.try_send(batch) {
            Ok(()) => {
                self.complete_entry(
                    anchor_id,
                    identity,
                    terminal_hash,
                    CompletionKind::Transferred,
                );
                DeliveryAck::Delivered { anchor_id }
            }
            Err(mpsc::error::TrySendError::Full(batch)) => {
                self.restore_admission(anchor_id, &identity, batch.admission);
                delivery_failure(anchor_id, DeliveryFailureClass::SchedulerFull)
            }
            Err(mpsc::error::TrySendError::Closed(batch)) => {
                self.restore_admission(anchor_id, &identity, batch.admission);
                delivery_failure(anchor_id, DeliveryFailureClass::SchedulerClosed)
            }
        }
    }

    fn restore_admission(
        &self,
        anchor_id: Uuid,
        identity: &ReservationIdentity,
        admission: SchedulerAdmission,
    ) {
        let mut state = lock_unpoisoned(&self.inner.state);
        if let Some(ReservationEntry::Active(active)) = state.entries.get_mut(&anchor_id)
            && active.identity == *identity
            && active.admission.is_none()
        {
            active.admission = Some(admission);
        }
    }

    fn build_reservation(
        &self,
        window: &ClosedTrajectoryWindow,
    ) -> Result<SampleBatchReservation, DeliveryFailureClass> {
        let pending = &window.pending;
        let evaluator_version = self
            .inner
            .evaluator_versions
            .get(&pending.pool_id)
            .ok_or(DeliveryFailureClass::InvalidWindow)?;
        let anchor_model = pending
            .request_projection
            .normalized_request
            .model
            .as_deref()
            .ok_or(DeliveryFailureClass::InvalidWindow)?;
        if pending.candidate_facts.is_empty()
            || pending.candidate_facts.len() != window.eligible_candidates.len()
            || pending.replay_capability_facts.api_family != pending.request_projection.family
        {
            return Err(DeliveryFailureClass::InvalidWindow);
        }
        let created_at_unix_ms = window.closed_at.timestamp_millis();
        let mut attempts = Vec::with_capacity(pending.candidate_facts.len());
        for (fact, candidate) in pending
            .candidate_facts
            .iter()
            .zip(&window.eligible_candidates)
        {
            let projected_fact =
                PersistedCandidateFactV1::from_eligible(candidate, &pending.request_projection)
                    .map_err(|_| DeliveryFailureClass::InvalidWindow)?;
            if projected_fact != *fact
                || candidate
                    .request
                    .content
                    .get("model")
                    .and_then(serde_json::Value::as_str)
                    != Some(fact.model.as_str())
            {
                return Err(DeliveryFailureClass::InvalidWindow);
            }
            let request_projection =
                candidate_request_projection(&pending.request_projection, &fact.model)
                    .map_err(|_| DeliveryFailureClass::InvalidWindow)?;
            attempts.push(
                ReservedShadowAttempt::new(
                    Uuid::now_v7(),
                    Uuid::now_v7(),
                    fact.candidate_id.clone(),
                    fact.model.clone(),
                    fact.model_revision.clone(),
                    fact.cost_rank,
                    pending.request_projection.family,
                    pending.replay_capability_facts.transport_identity.clone(),
                    anchor_model,
                    pending.anchor_model_revision.clone(),
                    fact.decoding_fingerprint.clone(),
                    evaluator_version.clone(),
                    pending
                        .routing_context_projection
                        .tenant_policy_hash
                        .clone(),
                    pending.routing_context_projection.agent_policy_hash.clone(),
                    true,
                    request_projection,
                    created_at_unix_ms,
                )
                .map_err(|_| DeliveryFailureClass::InvalidWindow)?,
            );
        }
        SampleBatchReservation::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            pending.anchor_id,
            pending.config_generation_id.clone(),
            pending.policy_version_id.clone(),
            pending.learning_generation_id,
            pending.pool_id.clone(),
            attempts,
            created_at_unix_ms,
        )
        .map_err(|_| DeliveryFailureClass::InvalidWindow)
    }
}

impl TrajectorySink for SqliteTrajectorySink {
    fn record_pending(&self, pending: Arc<PendingTrajectoryWindow>) -> SinkFuture {
        let sink = self.clone();
        Box::pin(async move { sink.record_pending_impl(pending).await })
    }

    fn record_terminal(&self, terminal: Arc<PersistedTrajectoryTerminalV1>) -> SinkFuture {
        let sink = self.clone();
        Box::pin(async move { sink.record_terminal_impl(terminal).await })
    }
}

impl TrajectoryDelivery for SqliteTrajectorySink {
    fn deliver(&self, window: Arc<ClosedTrajectoryWindow>) -> DeliveryFuture {
        let sink = self.clone();
        Box::pin(async move { sink.deliver_impl(window).await })
    }
}

impl ReservationIdentity {
    fn from_pending(pending: &PendingTrajectoryWindow) -> Result<Self, SinkFailureClass> {
        let pending_hash = pending
            .payload_hash()
            .map_err(|_| SinkFailureClass::CanonicalizationFailed)?;
        if pending.pool_id.is_empty()
            || pending.policy_version_id.is_empty()
            || pending.candidate_facts.is_empty()
        {
            return Err(SinkFailureClass::InvalidReservation);
        }
        Ok(Self {
            pending_hash,
            pool_id: pending.pool_id.clone(),
            policy_version_id: pending.policy_version_id.clone(),
            candidate_count: pending.candidate_facts.len(),
        })
    }
}

fn map_queue_full_ack(
    acknowledgement: NotScheduledQueueFullAck,
    expected_anchor_id: Uuid,
    expected_pending_hash: &str,
    expected_terminal_hash: &str,
) -> SinkAck {
    match acknowledgement {
        NotScheduledQueueFullAck::Applied {
            anchor_id,
            pending_hash,
            terminal_hash,
        }
        | NotScheduledQueueFullAck::AlreadyApplied {
            anchor_id,
            pending_hash,
            terminal_hash,
        } if anchor_id == expected_anchor_id
            && pending_hash == expected_pending_hash
            && terminal_hash == expected_terminal_hash
            && is_canonical_sha256(&terminal_hash) =>
        {
            SinkAck::DurablyDeclined {
                anchor_id,
                pending_hash,
                terminal_hash,
                reason: DurableDeclineReason::NotScheduledQueueFull,
            }
        }
        NotScheduledQueueFullAck::Conflict { anchor_id } if anchor_id == expected_anchor_id => {
            SinkAck::Conflict { anchor_id }
        }
        NotScheduledQueueFullAck::EvidenceCapacity { anchor_id }
            if anchor_id == expected_anchor_id =>
        {
            SinkAck::TransientRefused {
                anchor_id,
                reason: TransientRefusalReason::EvidenceCapacity,
            }
        }
        NotScheduledQueueFullAck::OriginatingProcessNotLive { anchor_id }
        | NotScheduledQueueFullAck::TransactionNotStarted { anchor_id }
            if anchor_id == expected_anchor_id =>
        {
            SinkAck::Failed {
                anchor_id,
                stable_class: SinkFailureClass::RepositoryRejected,
            }
        }
        _ => SinkAck::Conflict {
            anchor_id: expected_anchor_id,
        },
    }
}

fn is_canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn sink_writer_failure(anchor_id: Uuid, error: WriterFailure) -> SinkAck {
    SinkAck::Failed {
        anchor_id,
        stable_class: SinkFailureClass::Writer(error.class()),
    }
}

fn delivery_failure(anchor_id: Uuid, stable_class: DeliveryFailureClass) -> DeliveryAck {
    DeliveryAck::Failed {
        anchor_id,
        stable_class,
    }
}

fn trim_completed_receipts(state: &mut SinkState, capacity: usize, kind: CompletionKind) {
    loop {
        let anchor_id = match kind {
            CompletionKind::Rejected if state.rejected_order.len() > capacity => {
                state.rejected_order.pop_front()
            }
            CompletionKind::Transferred if state.transferred_order.len() > capacity => {
                state.transferred_order.pop_front()
            }
            CompletionKind::Rejected | CompletionKind::Transferred => None,
        };
        let Some(anchor_id) = anchor_id else {
            break;
        };
        if matches!(
            state.entries.get(&anchor_id),
            Some(ReservationEntry::Completed(completed)) if completed.kind == kind
        ) {
            state.entries.remove(&anchor_id);
        }
    }
}

fn write_deadline() -> Instant {
    Instant::now()
        .checked_add(WRITE_TIMEOUT)
        .unwrap_or_else(Instant::now)
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use chrono::{Duration as ChronoDuration, TimeZone};
    use nemo_relay::api::llm::{LlmApiFamily, LlmRequest};
    use nemo_relay::api::runtime::{
        LLM_REPLAY_CONTRACT_VERSION, LlmReplayCall, LlmReplayCapability, LlmReplayTransport,
    };
    use serde_json::{Map, json};
    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::adapter::FamilyAdapter;
    use crate::canonical_json::canonical_sha256;
    use crate::config::CanonicalizerConfig;
    use crate::ledger::model::LedgerRuntimeIdentity;
    use crate::ledger::repository::LedgerRepository;
    use crate::ledger::writer::LedgerWriterOwner;
    use crate::preflight::EligibleCandidate;
    use crate::projection::{
        ROUTING_CONTEXT_SCHEMA_V1, RouterRoutingContextProjectionV1, project_request,
    };
    use crate::response_validator::compile_candidate_response_contracts;
    use crate::trajectory::{
        PENDING_TRAJECTORY_SCHEMA_V1, PersistedCandidateFactV1, ReplayCapabilityFactsV1,
        TrajectoryRejectionReason, TrajectoryTrigger, TrajectoryWindowSeed,
        project_anchor_response,
    };

    const POOL: &str = "pool-a";
    const WRITER_CAPACITY: usize = 8;

    struct CountingReplay {
        capability: LlmReplayCapability,
        starts: Arc<AtomicUsize>,
    }

    impl LlmReplayTransport for CountingReplay {
        fn capability(&self) -> &LlmReplayCapability {
            &self.capability
        }

        fn start(&self, _request: LlmRequest) -> nemo_relay::error::Result<LlmReplayCall> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            Ok(LlmReplayCall::new(async { Ok(json!({"ok": true})) }, || {}))
        }
    }

    struct WindowFixture {
        pending: Arc<PendingTrajectoryWindow>,
        seed: TrajectoryWindowSeed,
        starts: Arc<AtomicUsize>,
    }

    struct Harness {
        _temporary: TempDir,
        config: RouterConfig,
        identity: LedgerRuntimeIdentity,
        owner: LedgerWriterOwner,
        client: LedgerWriterClient,
        sink: SqliteTrajectorySink,
        scheduler_rx: Option<mpsc::Receiver<ScheduledTrajectoryBatch>>,
    }

    fn config(path: &Path, max_pending: usize, max_evidence_records: u64) -> RouterConfig {
        serde_json::from_value(json!({
            "version": 1,
            "mode": "shadow",
            "project_id": "sqlite-sink-tests",
            "database_path": path.to_string_lossy(),
            "retention_days": 30,
            "max_evidence_records": max_evidence_records,
            "pools": [{
                "id": POOL,
                "api_family": "openai_chat_completions",
                "anchor_models": ["anchor-a"],
                "anchor_revision": "2026-07-01",
                "sampling_probability": 1.0,
                "max_candidates_per_sample": 1,
                "concurrency": {
                    "shadow": 1,
                    "judge": 1,
                    "max_pending": max_pending
                },
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
                    "max_rationale_bytes": 1024,
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

    fn harness(max_pending: usize, max_evidence_records: u64) -> Harness {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger/router.db");
        let config = config(&path, max_pending, max_evidence_records);
        let activated = LedgerRepository::activate(&config).unwrap();
        let identity = activated.identity.clone();
        let (owner, client) =
            LedgerWriterOwner::start(activated.repository, WRITER_CAPACITY).unwrap();
        let (sink, scheduler_rx) = SqliteTrajectorySink::new(&config, client.clone()).unwrap();
        Harness {
            _temporary: temporary,
            config,
            identity,
            owner,
            client,
            sink,
            scheduler_rx: Some(scheduler_rx),
        }
    }

    fn exact_now() -> chrono::DateTime<Utc> {
        Utc.timestamp_millis_opt(Utc::now().timestamp_millis())
            .single()
            .unwrap()
    }

    fn window_fixture(harness: &Harness) -> WindowFixture {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let request = LlmRequest {
            headers: Map::new(),
            content: json!({"model": "anchor-a", "messages": []}),
        };
        let envelope = FamilyAdapter.decode(family, &request).unwrap();
        let request_projection =
            project_request(&envelope, &CanonicalizerConfig::default()).unwrap();
        let candidate_config = harness.config.pools[0].candidates[0].clone();
        let candidate_request = FamilyAdapter
            .with_model(&envelope, &candidate_config.model)
            .unwrap();
        let response_contracts = Arc::new(
            compile_candidate_response_contracts(&envelope.normalized_request, 64 * 1024).unwrap(),
        );
        let candidate =
            EligibleCandidate::new(candidate_config, candidate_request, response_contracts);
        let candidate_facts =
            vec![PersistedCandidateFactV1::from_eligible(&candidate, &request_projection).unwrap()];
        let capability = LlmReplayCapability {
            contract_version: LLM_REPLAY_CONTRACT_VERSION,
            api_family: family,
            transport_identity: "sqlite-sink-test-transport".to_string(),
        };
        let replay_capability_facts =
            ReplayCapabilityFactsV1::from_capability(&capability).unwrap();
        let starts = Arc::new(AtomicUsize::new(0));
        let replay_transport: Arc<dyn LlmReplayTransport> = Arc::new(CountingReplay {
            capability,
            starts: starts.clone(),
        });
        let pool_identity = harness.identity.pools.get(POOL).unwrap();
        let opened_at = exact_now();
        let pending = PendingTrajectoryWindow {
            schema: PENDING_TRAJECTORY_SCHEMA_V1.to_string(),
            anchor_id: Uuid::now_v7(),
            anchor_call_uuid: Uuid::now_v7(),
            root_uuid: Uuid::now_v7(),
            owner_uuid: Uuid::now_v7(),
            owner_path: Vec::new(),
            pool_id: POOL.to_string(),
            anchor_model_revision: "2026-07-01".to_string(),
            process_instance_id: harness.identity.process_instance_id,
            project_uuid: harness.identity.project_uuid,
            project_id: harness.identity.project_id.clone(),
            config_generation_id: harness.identity.config_generation_id.clone(),
            policy_version_id: pool_identity.policy_version_id.clone(),
            learning_generation_id: pool_identity.learning_generation_id,
            request_projection,
            routing_context_projection: RouterRoutingContextProjectionV1 {
                schema: ROUTING_CONTEXT_SCHEMA_V1.to_string(),
                tenant_policy_hash: "3".repeat(64),
                agent_policy_hash: "4".repeat(64),
                position_features: BTreeMap::new(),
            },
            normalized_anchor_response: project_anchor_response(
                family,
                &json!({
                    "id": "response-a",
                    "model": "anchor-a",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop"
                    }]
                }),
                64 * 1024,
            )
            .unwrap(),
            replay_capability_facts,
            candidate_facts,
            requested_progress: 1,
            opened_at,
            deadline_at: opened_at + ChronoDuration::minutes(5),
        };
        let pending = Arc::new(pending);
        let seed = TrajectoryWindowSeed::new(
            pending.as_ref().clone(),
            envelope,
            replay_transport,
            vec![candidate],
        );
        WindowFixture {
            pending,
            seed,
            starts,
        }
    }

    async fn drain(harness: &mut Harness) {
        harness
            .owner
            .drain_until(Instant::now() + Duration::from_secs(3))
            .await
            .unwrap();
    }

    fn assert_pending_applied(acknowledgement: SinkAck, pending: &PendingTrajectoryWindow) {
        assert_eq!(
            acknowledgement,
            SinkAck::Applied {
                anchor_id: pending.anchor_id,
                payload_hash: pending.payload_hash().unwrap(),
            }
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn canceled_pending_write_retains_admission_and_fifo_retry_resolves_commit() {
        let mut harness = harness(1, 10);
        let fixture = window_fixture(&harness);
        let (probe_reached, wait_for_probe) = tokio::sync::oneshot::channel();
        let (resume_after_probe, probe_resume) = tokio::sync::oneshot::channel();
        harness
            .sink
            .install_pending_probe_hook(probe_reached, probe_resume);

        let sink = harness.sink.clone();
        let pending = fixture.pending.clone();
        let write = tokio::spawn(async move { sink.record_pending(pending).await });
        wait_for_probe.await.unwrap();
        let blocker = rusqlite::Connection::open(&harness.config.database_path).unwrap();
        blocker.execute_batch("BEGIN IMMEDIATE").unwrap();
        resume_after_probe.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                let pending_write = {
                    let state = lock_unpoisoned(&harness.sink.inner.state);
                    matches!(
                        state.entries.get(&fixture.pending.anchor_id),
                        Some(ReservationEntry::Active(active))
                            if matches!(active.stage, ReservationStage::PendingWrite)
                    )
                };
                if pending_write {
                    break;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap();
        write.abort();
        assert!(write.await.unwrap_err().is_cancelled());
        blocker.execute_batch("ROLLBACK").unwrap();

        assert_eq!(
            harness.sink.record_pending(fixture.pending.clone()).await,
            SinkAck::AlreadyApplied {
                anchor_id: fixture.pending.anchor_id,
                payload_hash: fixture.pending.payload_hash().unwrap(),
            }
        );
        assert_eq!(fixture.starts.load(Ordering::SeqCst), 0);
        drain(&mut harness).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn recovered_pending_queue_full_is_atomic_exact_and_timestamp_safe() {
        let mut harness = harness(1, 10);
        let retained = window_fixture(&harness);
        assert_pending_applied(
            harness.sink.record_pending(retained.pending.clone()).await,
            &retained.pending,
        );

        let recovered = window_fixture(&harness);
        let mut recovered_pending = recovered.pending.as_ref().clone();
        let now = exact_now();
        recovered_pending.opened_at = now - ChronoDuration::seconds(10);
        recovered_pending.deadline_at = now + ChronoDuration::minutes(5);
        let recovered_pending = Arc::new(recovered_pending);
        let frozen = FrozenPendingAnchorV1::new(
            &recovered_pending,
            Uuid::now_v7(),
            Uuid::now_v7(),
            (now - ChronoDuration::seconds(1)).timestamp_millis(),
        )
        .unwrap();
        let permit = harness.client.try_reserve_pre_accept().unwrap();
        assert!(matches!(
            harness
                .client
                .record_pending_anchor_with_capacity_and_permit(permit, frozen, 10)
                .await
                .unwrap(),
            PendingAnchorCapacityAck::Applied { .. }
        ));

        let first = harness.sink.record_pending(recovered_pending.clone()).await;
        let (pending_hash, terminal_hash) = match first {
            SinkAck::DurablyDeclined {
                anchor_id,
                pending_hash,
                terminal_hash,
                reason: DurableDeclineReason::NotScheduledQueueFull,
            } if anchor_id == recovered_pending.anchor_id => (pending_hash, terminal_hash),
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        assert_eq!(pending_hash, recovered_pending.payload_hash().unwrap());
        assert!(is_canonical_sha256(&terminal_hash));
        assert_eq!(
            harness.sink.record_pending(recovered_pending.clone()).await,
            SinkAck::DurablyDeclined {
                anchor_id: recovered_pending.anchor_id,
                pending_hash,
                terminal_hash,
                reason: DurableDeclineReason::NotScheduledQueueFull,
            }
        );
        assert_eq!(retained.starts.load(Ordering::SeqCst), 0);
        assert_eq!(recovered.starts.load(Ordering::SeqCst), 0);
        drain(&mut harness).await;
    }

    #[test]
    fn queue_full_mapper_rejects_alternate_valid_hashes() {
        let anchor_id = Uuid::now_v7();
        let expected_pending = "1".repeat(64);
        let expected_terminal = "2".repeat(64);
        assert_eq!(
            map_queue_full_ack(
                NotScheduledQueueFullAck::Applied {
                    anchor_id,
                    pending_hash: expected_pending.clone(),
                    terminal_hash: "3".repeat(64),
                },
                anchor_id,
                &expected_pending,
                &expected_terminal,
            ),
            SinkAck::Conflict { anchor_id }
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rejected_terminal_is_idempotent_and_releases_scheduler_admission() {
        let mut harness = harness(1, 10);
        let rejected = window_fixture(&harness);
        assert_pending_applied(
            harness.sink.record_pending(rejected.pending.clone()).await,
            &rejected.pending,
        );
        let terminal = Arc::new(PersistedTrajectoryTerminalV1::rejected(
            rejected.pending.as_ref().clone(),
            Vec::new(),
            0,
            TrajectoryRejectionReason::CanceledBeforeAnchorEnd,
            exact_now(),
            Vec::new(),
        ));
        let terminal_hash = terminal.payload_hash().unwrap();
        assert_eq!(
            harness.sink.record_terminal(terminal.clone()).await,
            SinkAck::Applied {
                anchor_id: rejected.pending.anchor_id,
                payload_hash: terminal_hash.clone(),
            }
        );
        assert_eq!(
            harness.sink.record_terminal(terminal).await,
            SinkAck::AlreadyApplied {
                anchor_id: rejected.pending.anchor_id,
                payload_hash: terminal_hash,
            }
        );

        let next = window_fixture(&harness);
        assert_pending_applied(
            harness.sink.record_pending(next.pending.clone()).await,
            &next.pending,
        );
        assert_eq!(rejected.starts.load(Ordering::SeqCst), 0);
        assert_eq!(next.starts.load(Ordering::SeqCst), 0);
        drain(&mut harness).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn evidence_capacity_refuses_without_retaining_an_admission() {
        let mut harness = harness(1, 1);
        let retained = window_fixture(&harness);
        assert_pending_applied(
            harness.sink.record_pending(retained.pending.clone()).await,
            &retained.pending,
        );
        let terminal = Arc::new(PersistedTrajectoryTerminalV1::rejected(
            retained.pending.as_ref().clone(),
            Vec::new(),
            0,
            TrajectoryRejectionReason::EventLoss,
            exact_now(),
            Vec::new(),
        ));
        assert!(matches!(
            harness.sink.record_terminal(terminal).await,
            SinkAck::Applied { .. }
        ));

        let refused = window_fixture(&harness);
        assert_eq!(
            harness.sink.record_pending(refused.pending.clone()).await,
            SinkAck::TransientRefused {
                anchor_id: refused.pending.anchor_id,
                reason: TransientRefusalReason::EvidenceCapacity,
            }
        );
        assert!(
            !lock_unpoisoned(&harness.sink.inner.state)
                .entries
                .contains_key(&refused.pending.anchor_id)
        );
        assert_eq!(refused.starts.load(Ordering::SeqCst), 0);
        drain(&mut harness).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn closed_delivery_transfers_stable_reservation_and_never_starts_replay() {
        let mut harness = harness(1, 10);
        let fixture = window_fixture(&harness);
        assert_pending_applied(
            harness.sink.record_pending(fixture.pending.clone()).await,
            &fixture.pending,
        );
        let closed = Arc::new(fixture.seed.into_closed(
            Vec::new(),
            1,
            TrajectoryTrigger::ProgressReached,
            exact_now(),
        ));
        let terminal = Arc::new(closed.terminal());
        assert!(matches!(
            harness.sink.record_terminal(terminal).await,
            SinkAck::Applied { .. }
        ));
        assert_eq!(
            harness.sink.deliver(closed.clone()).await,
            DeliveryAck::Delivered {
                anchor_id: closed.anchor_id(),
            }
        );
        assert_eq!(
            harness.sink.deliver(closed.clone()).await,
            DeliveryAck::AlreadyDelivered {
                anchor_id: closed.anchor_id(),
            }
        );
        let batch = harness.scheduler_rx.as_mut().unwrap().recv().await.unwrap();
        assert_eq!(batch.window.anchor_id(), closed.anchor_id());
        assert_eq!(batch.reservation.anchor_id, closed.anchor_id());
        assert_eq!(batch.reservation.attempts.len(), 1);
        assert_eq!(batch.admission.pool_id(), POOL);
        assert_eq!(batch.admission.candidate_count(), 1);
        assert_eq!(fixture.starts.load(Ordering::SeqCst), 0);
        drop(batch);

        let next = window_fixture(&harness);
        assert_pending_applied(
            harness.sink.record_pending(next.pending.clone()).await,
            &next.pending,
        );
        drain(&mut harness).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn rejected_receipt_does_not_evict_transferred_receipt() {
        let mut harness = harness(1, 10);
        let transferred = window_fixture(&harness);
        assert_pending_applied(
            harness
                .sink
                .record_pending(transferred.pending.clone())
                .await,
            &transferred.pending,
        );
        let closed = Arc::new(transferred.seed.into_closed(
            Vec::new(),
            1,
            TrajectoryTrigger::ProgressReached,
            exact_now(),
        ));
        assert!(matches!(
            harness
                .sink
                .record_terminal(Arc::new(closed.terminal()))
                .await,
            SinkAck::Applied { .. }
        ));
        assert!(matches!(
            harness.sink.deliver(closed.clone()).await,
            DeliveryAck::Delivered { .. }
        ));
        drop(harness.scheduler_rx.as_mut().unwrap().recv().await.unwrap());

        let rejected = window_fixture(&harness);
        assert_pending_applied(
            harness.sink.record_pending(rejected.pending.clone()).await,
            &rejected.pending,
        );
        let rejected_terminal = Arc::new(PersistedTrajectoryTerminalV1::rejected(
            rejected.pending.as_ref().clone(),
            Vec::new(),
            0,
            TrajectoryRejectionReason::EventLoss,
            exact_now(),
            Vec::new(),
        ));
        assert!(matches!(
            harness.sink.record_terminal(rejected_terminal).await,
            SinkAck::Applied { .. }
        ));

        assert_eq!(
            harness.sink.deliver(closed.clone()).await,
            DeliveryAck::AlreadyDelivered {
                anchor_id: closed.anchor_id(),
            }
        );
        assert_eq!(transferred.starts.load(Ordering::SeqCst), 0);
        assert_eq!(rejected.starts.load(Ordering::SeqCst), 0);
        drain(&mut harness).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_receiver_restores_admission_and_reuses_batch_ids() {
        let mut harness = harness(1, 10);
        let fixture = window_fixture(&harness);
        assert_pending_applied(
            harness.sink.record_pending(fixture.pending.clone()).await,
            &fixture.pending,
        );
        let closed = Arc::new(fixture.seed.into_closed(
            Vec::new(),
            1,
            TrajectoryTrigger::ProgressReached,
            exact_now(),
        ));
        assert!(matches!(
            harness
                .sink
                .record_terminal(Arc::new(closed.terminal()))
                .await,
            SinkAck::Applied { .. }
        ));
        drop(harness.scheduler_rx.take());

        assert_eq!(
            harness.sink.deliver(closed.clone()).await,
            DeliveryAck::Failed {
                anchor_id: closed.anchor_id(),
                stable_class: DeliveryFailureClass::SchedulerClosed,
            }
        );
        let first = cached_reservation(&harness.sink, closed.anchor_id());
        assert_eq!(
            harness.sink.deliver(closed.clone()).await,
            DeliveryAck::Failed {
                anchor_id: closed.anchor_id(),
                stable_class: DeliveryFailureClass::SchedulerClosed,
            }
        );
        let second = cached_reservation(&harness.sink, closed.anchor_id());
        assert_eq!(first.sample_batch_id, second.sample_batch_id);
        assert_eq!(first.open_state_event_id, second.open_state_event_id);
        assert_eq!(
            first.attempts[0].shadow_attempt_id,
            second.attempts[0].shadow_attempt_id
        );
        assert_eq!(fixture.starts.load(Ordering::SeqCst), 0);
        drain(&mut harness).await;
    }

    fn cached_reservation(sink: &SqliteTrajectorySink, anchor_id: Uuid) -> SampleBatchReservation {
        let state = lock_unpoisoned(&sink.inner.state);
        match state.entries.get(&anchor_id) {
            Some(ReservationEntry::Active(active)) => match &active.stage {
                ReservationStage::ClosedDurable {
                    reservation: Some(reservation),
                    ..
                } => reservation.clone(),
                _ => panic!("reservation was not cached"),
            },
            _ => panic!("active reservation entry was not retained"),
        }
    }

    #[test]
    fn pending_conflict_uses_hash_pool_policy_and_candidate_identity() {
        let mut harness = harness(1, 10);
        let runtime = tokio::runtime::Runtime::new().unwrap();
        let fixture = window_fixture(&harness);
        runtime.block_on(async {
            assert_pending_applied(
                harness.sink.record_pending(fixture.pending.clone()).await,
                &fixture.pending,
            );
            let mut conflict = fixture.pending.as_ref().clone();
            conflict.policy_version_id = canonical_sha256(&json!({"alternate": true})).unwrap();
            assert_eq!(
                harness.sink.record_pending(Arc::new(conflict)).await,
                SinkAck::Conflict {
                    anchor_id: fixture.pending.anchor_id,
                }
            );
            drain(&mut harness).await;
        });
    }
}
