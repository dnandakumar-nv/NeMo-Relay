// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private acknowledged persistence and delivery adapters for trajectory windows.

#![cfg_attr(not(test), allow(dead_code))]

use std::collections::{BTreeMap, btree_map::Entry};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use uuid::Uuid;

use crate::ledger::command::WriterFailureClass;
use crate::trajectory::{
    ClosedTrajectoryWindow, PendingTrajectoryWindow, PersistedTrajectoryTerminalV1,
};

/// Boxed result future returned by a trajectory sink.
pub(crate) type SinkFuture = Pin<Box<dyn Future<Output = SinkAck> + Send + 'static>>;

/// Boxed result future returned by a trajectory delivery adapter.
pub(crate) type DeliveryFuture = Pin<Box<dyn Future<Output = DeliveryAck> + Send + 'static>>;

/// Stable sink failure classes suitable for health reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SinkFailureClass {
    /// The safe payload could not be serialized canonically.
    CanonicalizationFailed,
    /// A terminal payload arrived before its pending fact was acknowledged.
    MissingPending,
    /// The pending identity cannot be represented by a configured scheduler pool.
    InvalidReservation,
    /// The scheduler admission pool has closed.
    SchedulerClosed,
    /// The ledger rejected the originating process or command state.
    RepositoryRejected,
    /// The bounded writer transport failed with a stable class.
    Writer(WriterFailureClass),
    /// A deterministic transient test failure was injected.
    InjectedTransient,
    /// A deterministic permanent test failure was injected.
    InjectedPermanent,
}

impl SinkFailureClass {
    /// Stable non-secret code for health and diagnostics.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::CanonicalizationFailed => "router.sink.canonicalization_failed",
            Self::MissingPending => "router.sink.missing_pending",
            Self::InvalidReservation => "router.sink.invalid_reservation",
            Self::SchedulerClosed => "router.sink.scheduler_closed",
            Self::RepositoryRejected => "router.sink.repository_rejected",
            Self::Writer(class) => class.code(),
            Self::InjectedTransient => "router.sink.injected_transient",
            Self::InjectedPermanent => "router.sink.injected_permanent",
        }
    }
}

/// Durable reasons a pending proposal was terminalized before acceptance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DurableDeclineReason {
    /// Scheduler capacity was unavailable for the matching pool.
    NotScheduledQueueFull,
}

/// Transient reasons a pending proposal was refused without creating a fact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TransientRefusalReason {
    /// The configured durable evidence bound cannot currently be freed.
    EvidenceCapacity,
}

/// Exhaustive acknowledgement for a pending or terminal sink operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum SinkAck {
    /// This exact payload was applied.
    Applied {
        /// Trajectory anchor identity.
        anchor_id: Uuid,
        /// SHA-256 of the canonical payload bytes.
        payload_hash: String,
    },
    /// This exact payload had already been applied.
    AlreadyApplied {
        /// Trajectory anchor identity.
        anchor_id: Uuid,
        /// SHA-256 of the canonical payload bytes.
        payload_hash: String,
    },
    /// The matching pending payload has already advanced to a terminal payload.
    AlreadyTerminal {
        /// Trajectory anchor identity.
        anchor_id: Uuid,
        /// SHA-256 of the canonical pending payload bytes.
        pending_hash: String,
        /// SHA-256 of the canonical terminal payload bytes.
        terminal_hash: String,
    },
    /// The pending fact and a pre-accept terminal fact were atomically persisted.
    DurablyDeclined {
        /// Trajectory anchor identity.
        anchor_id: Uuid,
        /// SHA-256 of the canonical pending payload bytes.
        pending_hash: String,
        /// SHA-256 of the canonical terminal payload bytes.
        terminal_hash: String,
        /// Typed durable-decline reason.
        reason: DurableDeclineReason,
    },
    /// The pending proposal was refused without creating a durable fact.
    TransientRefused {
        /// Trajectory anchor identity.
        anchor_id: Uuid,
        /// Typed transient-refusal reason.
        reason: TransientRefusalReason,
    },
    /// The anchor exists with different canonical payload bytes.
    Conflict {
        /// Trajectory anchor identity.
        anchor_id: Uuid,
    },
    /// The operation failed without exposing an internal error payload.
    Failed {
        /// Trajectory anchor identity.
        anchor_id: Uuid,
        /// Stable non-secret failure class.
        stable_class: SinkFailureClass,
    },
}

/// Private asynchronous trajectory persistence contract.
pub(crate) trait TrajectorySink: Send + Sync {
    /// Record the immutable pending fact before the proposal is accepted.
    fn record_pending(&self, pending: Arc<PendingTrajectoryWindow>) -> SinkFuture;

    /// Record one terminal fact for a previously acknowledged pending fact.
    fn record_terminal(&self, terminal: Arc<PersistedTrajectoryTerminalV1>) -> SinkFuture;
}

/// Sink operation selected by deterministic test controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SinkOperation {
    /// Pending-fact operation.
    Pending,
    /// Terminal-fact operation.
    Terminal,
}

#[derive(Default)]
struct OperationControl {
    delay: Duration,
    transient_failures: usize,
    permanent_failure: bool,
    #[cfg(test)]
    attempts: usize,
}

impl OperationControl {
    fn take_failure(&mut self) -> Option<SinkFailureClass> {
        if self.permanent_failure {
            return Some(SinkFailureClass::InjectedPermanent);
        }
        if self.transient_failures > 0 {
            self.transient_failures -= 1;
            return Some(SinkFailureClass::InjectedTransient);
        }
        None
    }
}

struct PendingEvidence {
    #[allow(dead_code)] // Spec 05 recovery reads the safe pending payload.
    payload: Arc<PendingTrajectoryWindow>,
    canonical_bytes: Arc<[u8]>,
    payload_hash: String,
}

struct TerminalEvidence {
    pending_hash: String,
    pending_canonical_bytes: Arc<[u8]>,
    canonical_bytes: Arc<[u8]>,
    payload_hash: String,
}

enum EvidenceRecord {
    Pending(PendingEvidence),
    Terminal(TerminalEvidence),
}

#[derive(Default)]
struct SinkState {
    records: BTreeMap<Uuid, EvidenceRecord>,
    pending_control: OperationControl,
    terminal_control: OperationControl,
}

impl SinkState {
    fn control(&self, operation: SinkOperation) -> &OperationControl {
        match operation {
            SinkOperation::Pending => &self.pending_control,
            SinkOperation::Terminal => &self.terminal_control,
        }
    }

    #[cfg(test)]
    fn control_mut(&mut self, operation: SinkOperation) -> &mut OperationControl {
        match operation {
            SinkOperation::Pending => &mut self.pending_control,
            SinkOperation::Terminal => &mut self.terminal_control,
        }
    }
}

/// Safe snapshot of one unresolved pending fact for recovery inspection.
#[cfg(test)]
pub(crate) struct UnresolvedPending {
    /// Original safe serializable pending payload.
    pub(crate) payload: Arc<PendingTrajectoryWindow>,
    /// Exact RFC 8785 bytes retained by the sink.
    pub(crate) canonical_bytes: Arc<[u8]>,
    /// SHA-256 of `canonical_bytes`.
    pub(crate) payload_hash: String,
}

/// Bounded in-memory implementation of the durable sink contract.
#[derive(Clone)]
pub(crate) struct InMemoryTrajectorySink {
    max_evidence_records: usize,
    state: Arc<Mutex<SinkState>>,
}

impl InMemoryTrajectorySink {
    /// Construct a sink that refuses new anchors at the configured bound.
    pub(crate) fn new(max_evidence_records: usize) -> Self {
        Self {
            max_evidence_records,
            state: Arc::new(Mutex::new(SinkState::default())),
        }
    }

    /// Configure a deterministic acknowledgement delay for one operation kind.
    #[cfg(test)]
    pub(crate) fn set_delay(&self, operation: SinkOperation, delay: Duration) {
        lock_unpoisoned(&self.state).control_mut(operation).delay = delay;
    }

    /// Fail exactly the next `count` calls of one operation kind.
    #[cfg(test)]
    pub(crate) fn inject_transient_failures(&self, operation: SinkOperation, count: usize) {
        lock_unpoisoned(&self.state)
            .control_mut(operation)
            .transient_failures = count;
    }

    /// Enable or disable a permanent deterministic failure for one operation kind.
    #[cfg(test)]
    pub(crate) fn set_permanent_failure(&self, operation: SinkOperation, enabled: bool) {
        lock_unpoisoned(&self.state)
            .control_mut(operation)
            .permanent_failure = enabled;
    }

    /// Return the number of sink futures started for one operation kind.
    #[cfg(test)]
    pub(crate) fn operation_attempts(&self, operation: SinkOperation) -> usize {
        lock_unpoisoned(&self.state).control(operation).attempts
    }

    /// Return unresolved pending facts in deterministic anchor order.
    #[cfg(test)]
    pub(crate) fn unresolved_pending(&self) -> Vec<UnresolvedPending> {
        lock_unpoisoned(&self.state)
            .records
            .values()
            .filter_map(|record| match record {
                EvidenceRecord::Pending(pending) => Some(UnresolvedPending {
                    payload: pending.payload.clone(),
                    canonical_bytes: pending.canonical_bytes.clone(),
                    payload_hash: pending.payload_hash.clone(),
                }),
                EvidenceRecord::Terminal(_) => None,
            })
            .collect()
    }

    #[cfg(test)]
    pub(crate) fn terminal_payloads(&self) -> Vec<PersistedTrajectoryTerminalV1> {
        lock_unpoisoned(&self.state)
            .records
            .values()
            .filter_map(|record| match record {
                EvidenceRecord::Terminal(terminal) => {
                    serde_json::from_slice(terminal.canonical_bytes.as_ref()).ok()
                }
                EvidenceRecord::Pending(_) => None,
            })
            .collect()
    }

    fn delay(&self, operation: SinkOperation) -> Duration {
        lock_unpoisoned(&self.state).control(operation).delay
    }
}

impl fmt::Debug for InMemoryTrajectorySink {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = lock_unpoisoned(&self.state);
        formatter
            .debug_struct("InMemoryTrajectorySink")
            .field("max_evidence_records", &self.max_evidence_records)
            .field("evidence_records", &state.records.len())
            .finish_non_exhaustive()
    }
}

impl TrajectorySink for InMemoryTrajectorySink {
    fn record_pending(&self, pending: Arc<PendingTrajectoryWindow>) -> SinkFuture {
        let anchor_id = pending.anchor_id();
        let canonical_bytes = match pending.canonical_bytes() {
            Ok(bytes) => Arc::<[u8]>::from(bytes),
            Err(()) => {
                return Box::pin(async move {
                    SinkAck::Failed {
                        anchor_id,
                        stable_class: SinkFailureClass::CanonicalizationFailed,
                    }
                });
            }
        };
        let payload_hash = match pending.payload_hash() {
            Ok(hash) => hash,
            Err(()) => {
                return Box::pin(async move {
                    SinkAck::Failed {
                        anchor_id,
                        stable_class: SinkFailureClass::CanonicalizationFailed,
                    }
                });
            }
        };
        #[cfg(test)]
        {
            lock_unpoisoned(&self.state)
                .control_mut(SinkOperation::Pending)
                .attempts += 1;
        }
        let delay = self.delay(SinkOperation::Pending);
        let max_evidence_records = self.max_evidence_records;
        let state = self.state.clone();

        Box::pin(async move {
            delay_ack(delay).await;
            let mut state = lock_unpoisoned(&state);
            if let Some(stable_class) = state.pending_control.take_failure() {
                return SinkAck::Failed {
                    anchor_id,
                    stable_class,
                };
            }

            let at_capacity = state.records.len() >= max_evidence_records;
            match state.records.entry(anchor_id) {
                Entry::Vacant(_) if at_capacity => SinkAck::TransientRefused {
                    anchor_id,
                    reason: TransientRefusalReason::EvidenceCapacity,
                },
                Entry::Vacant(entry) => {
                    entry.insert(EvidenceRecord::Pending(PendingEvidence {
                        payload: pending,
                        canonical_bytes,
                        payload_hash: payload_hash.clone(),
                    }));
                    SinkAck::Applied {
                        anchor_id,
                        payload_hash,
                    }
                }
                Entry::Occupied(entry) => match entry.get() {
                    EvidenceRecord::Pending(existing)
                        if existing.canonical_bytes.as_ref() == canonical_bytes.as_ref()
                            && existing.payload_hash == payload_hash =>
                    {
                        SinkAck::AlreadyApplied {
                            anchor_id,
                            payload_hash,
                        }
                    }
                    EvidenceRecord::Pending(_) => SinkAck::Conflict { anchor_id },
                    EvidenceRecord::Terminal(existing)
                        if existing.pending_hash == payload_hash
                            && existing.pending_canonical_bytes.as_ref()
                                == canonical_bytes.as_ref() =>
                    {
                        SinkAck::AlreadyTerminal {
                            anchor_id,
                            pending_hash: existing.pending_hash.clone(),
                            terminal_hash: existing.payload_hash.clone(),
                        }
                    }
                    EvidenceRecord::Terminal(_) => SinkAck::Conflict { anchor_id },
                },
            }
        })
    }

    fn record_terminal(&self, terminal: Arc<PersistedTrajectoryTerminalV1>) -> SinkFuture {
        let anchor_id = terminal.anchor_id();
        let pending_bytes = match terminal.pending.canonical_bytes() {
            Ok(bytes) => bytes,
            Err(()) => {
                return Box::pin(async move {
                    SinkAck::Failed {
                        anchor_id,
                        stable_class: SinkFailureClass::CanonicalizationFailed,
                    }
                });
            }
        };
        let pending_hash = match terminal.pending.payload_hash() {
            Ok(hash) => hash,
            Err(()) => {
                return Box::pin(async move {
                    SinkAck::Failed {
                        anchor_id,
                        stable_class: SinkFailureClass::CanonicalizationFailed,
                    }
                });
            }
        };
        let canonical_bytes = match terminal.canonical_bytes() {
            Ok(bytes) => Arc::<[u8]>::from(bytes),
            Err(()) => {
                return Box::pin(async move {
                    SinkAck::Failed {
                        anchor_id,
                        stable_class: SinkFailureClass::CanonicalizationFailed,
                    }
                });
            }
        };
        let payload_hash = match terminal.payload_hash() {
            Ok(hash) => hash,
            Err(()) => {
                return Box::pin(async move {
                    SinkAck::Failed {
                        anchor_id,
                        stable_class: SinkFailureClass::CanonicalizationFailed,
                    }
                });
            }
        };
        #[cfg(test)]
        {
            lock_unpoisoned(&self.state)
                .control_mut(SinkOperation::Terminal)
                .attempts += 1;
        }
        let delay = self.delay(SinkOperation::Terminal);
        let state = self.state.clone();

        Box::pin(async move {
            delay_ack(delay).await;
            let mut state = lock_unpoisoned(&state);
            if let Some(stable_class) = state.terminal_control.take_failure() {
                return SinkAck::Failed {
                    anchor_id,
                    stable_class,
                };
            }

            match state.records.entry(anchor_id) {
                Entry::Vacant(_) => SinkAck::Failed {
                    anchor_id,
                    stable_class: SinkFailureClass::MissingPending,
                },
                Entry::Occupied(mut entry) => match entry.get() {
                    EvidenceRecord::Pending(existing)
                        if existing.canonical_bytes.as_ref() == pending_bytes.as_slice()
                            && existing.payload_hash == pending_hash =>
                    {
                        entry.insert(EvidenceRecord::Terminal(TerminalEvidence {
                            pending_hash,
                            pending_canonical_bytes: Arc::from(pending_bytes.clone()),
                            canonical_bytes,
                            payload_hash: payload_hash.clone(),
                        }));
                        SinkAck::Applied {
                            anchor_id,
                            payload_hash,
                        }
                    }
                    EvidenceRecord::Pending(_) => SinkAck::Conflict { anchor_id },
                    EvidenceRecord::Terminal(existing)
                        if existing.pending_hash == pending_hash
                            && existing.pending_canonical_bytes.as_ref()
                                == pending_bytes.as_slice()
                            && existing.canonical_bytes.as_ref() == canonical_bytes.as_ref()
                            && existing.payload_hash == payload_hash =>
                    {
                        SinkAck::AlreadyApplied {
                            anchor_id,
                            payload_hash,
                        }
                    }
                    EvidenceRecord::Terminal(_) => SinkAck::Conflict { anchor_id },
                },
            }
        })
    }
}

/// Stable delivery failure classes suitable for health reporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeliveryFailureClass {
    /// The configured delivered-anchor bound has been reached.
    CapacityExceeded,
    /// The closed window does not match its retained pending identity.
    InvalidWindow,
    /// No retained scheduler admission exists for the closed window.
    MissingAdmission,
    /// Durable batch creation found conflicting canonical bytes.
    RepositoryConflict,
    /// The ledger rejected the originating process or command state.
    RepositoryRejected,
    /// The bounded scheduler channel was unexpectedly full despite admission.
    SchedulerFull,
    /// The bounded scheduler channel or admission pool has closed.
    SchedulerClosed,
    /// The bounded writer transport failed with a stable class.
    Writer(WriterFailureClass),
    /// A deterministic transient test failure was injected.
    InjectedTransient,
    /// A deterministic permanent test failure was injected.
    InjectedPermanent,
}

impl DeliveryFailureClass {
    /// Stable non-secret code for health and diagnostics.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::CapacityExceeded => "router.delivery.capacity_exceeded",
            Self::InvalidWindow => "router.delivery.invalid_window",
            Self::MissingAdmission => "router.delivery.missing_admission",
            Self::RepositoryConflict => "router.delivery.repository_conflict",
            Self::RepositoryRejected => "router.delivery.repository_rejected",
            Self::SchedulerFull => "router.delivery.scheduler_full",
            Self::SchedulerClosed => "router.delivery.scheduler_closed",
            Self::Writer(class) => class.code(),
            Self::InjectedTransient => "router.delivery.injected_transient",
            Self::InjectedPermanent => "router.delivery.injected_permanent",
        }
    }
}

/// Exhaustive acknowledgement for in-memory closed-window delivery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DeliveryAck {
    /// The anchor was recorded as delivered.
    Delivered {
        /// Trajectory anchor identity.
        anchor_id: Uuid,
    },
    /// The anchor had already been recorded as delivered.
    AlreadyDelivered {
        /// Trajectory anchor identity.
        anchor_id: Uuid,
    },
    /// Delivery failed without exposing the window or an internal error.
    Failed {
        /// Trajectory anchor identity.
        anchor_id: Uuid,
        /// Stable non-secret failure class.
        stable_class: DeliveryFailureClass,
    },
}

/// Private asynchronous handoff contract for evaluable closed windows.
pub(crate) trait TrajectoryDelivery: Send + Sync {
    /// Record a closed window as delivered without invoking replay.
    fn deliver(&self, window: Arc<ClosedTrajectoryWindow>) -> DeliveryFuture;
}

#[derive(Default)]
struct DeliveryState {
    delivered: BTreeMap<Uuid, ()>,
    delay: Duration,
    transient_failures: usize,
    permanent_failure: bool,
}

impl DeliveryState {
    fn take_failure(&mut self) -> Option<DeliveryFailureClass> {
        if self.permanent_failure {
            return Some(DeliveryFailureClass::InjectedPermanent);
        }
        if self.transient_failures > 0 {
            self.transient_failures -= 1;
            return Some(DeliveryFailureClass::InjectedTransient);
        }
        None
    }
}

/// Bounded idempotent delivery recorder that never invokes replay.
#[derive(Clone)]
pub(crate) struct InMemoryTrajectoryDelivery {
    max_delivered_records: usize,
    state: Arc<Mutex<DeliveryState>>,
}

impl InMemoryTrajectoryDelivery {
    /// Construct a delivery recorder that refuses new anchors at the bound.
    pub(crate) fn new(max_delivered_records: usize) -> Self {
        Self {
            max_delivered_records,
            state: Arc::new(Mutex::new(DeliveryState::default())),
        }
    }

    /// Configure a deterministic acknowledgement delay.
    #[cfg(test)]
    pub(crate) fn set_delay(&self, delay: Duration) {
        lock_unpoisoned(&self.state).delay = delay;
    }

    /// Fail exactly the next `count` delivery attempts.
    #[cfg(test)]
    pub(crate) fn inject_transient_failures(&self, count: usize) {
        lock_unpoisoned(&self.state).transient_failures = count;
    }

    /// Enable or disable a permanent deterministic delivery failure.
    #[cfg(test)]
    pub(crate) fn set_permanent_failure(&self, enabled: bool) {
        lock_unpoisoned(&self.state).permanent_failure = enabled;
    }

    /// Return delivered anchor IDs in deterministic order.
    #[cfg(test)]
    pub(crate) fn delivered_anchor_ids(&self) -> Vec<Uuid> {
        lock_unpoisoned(&self.state)
            .delivered
            .keys()
            .copied()
            .collect()
    }
}

impl fmt::Debug for InMemoryTrajectoryDelivery {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let state = lock_unpoisoned(&self.state);
        formatter
            .debug_struct("InMemoryTrajectoryDelivery")
            .field("max_delivered_records", &self.max_delivered_records)
            .field("delivered_records", &state.delivered.len())
            .finish_non_exhaustive()
    }
}

impl TrajectoryDelivery for InMemoryTrajectoryDelivery {
    fn deliver(&self, window: Arc<ClosedTrajectoryWindow>) -> DeliveryFuture {
        // Read only the safe identity and release all replay authority before waiting.
        let anchor_id = window.anchor_id();
        drop(window);

        let delay = lock_unpoisoned(&self.state).delay;
        let max_delivered_records = self.max_delivered_records;
        let state = self.state.clone();
        Box::pin(async move {
            delay_ack(delay).await;
            let mut state = lock_unpoisoned(&state);
            if let Some(stable_class) = state.take_failure() {
                return DeliveryAck::Failed {
                    anchor_id,
                    stable_class,
                };
            }

            let at_capacity = state.delivered.len() >= max_delivered_records;
            match state.delivered.entry(anchor_id) {
                Entry::Occupied(_) => DeliveryAck::AlreadyDelivered { anchor_id },
                Entry::Vacant(_) if at_capacity => DeliveryAck::Failed {
                    anchor_id,
                    stable_class: DeliveryFailureClass::CapacityExceeded,
                },
                Entry::Vacant(entry) => {
                    entry.insert(());
                    DeliveryAck::Delivered { anchor_id }
                }
            }
        })
    }
}

async fn delay_ack(delay: Duration) {
    if !delay.is_zero() {
        tokio::time::sleep(delay).await;
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::Ordering;

    use super::*;
    use crate::trajectory::test_fixtures::{closed_window, pending_window, terminal_window};

    fn pending(anchor: u128) -> Arc<PendingTrajectoryWindow> {
        Arc::new(pending_window(Uuid::from_u128(anchor)))
    }

    fn terminal(pending: &Arc<PendingTrajectoryWindow>) -> Arc<PersistedTrajectoryTerminalV1> {
        Arc::new(terminal_window(pending.anchor_id()))
    }

    #[tokio::test]
    async fn sink_enforces_pending_before_terminal_and_canonical_idempotency() {
        let sink = InMemoryTrajectorySink::new(2);
        let pending = pending(1);
        let terminal = terminal(&pending);
        let pending_hash = pending.payload_hash().unwrap();
        let terminal_hash = terminal.payload_hash().unwrap();
        let anchor_id = pending.anchor_id();

        assert_eq!(
            sink.record_terminal(terminal.clone()).await,
            SinkAck::Failed {
                anchor_id,
                stable_class: SinkFailureClass::MissingPending,
            }
        );
        assert_eq!(
            sink.record_pending(pending.clone()).await,
            SinkAck::Applied {
                anchor_id,
                payload_hash: pending_hash.clone(),
            }
        );
        assert_eq!(
            sink.record_pending(pending.clone()).await,
            SinkAck::AlreadyApplied {
                anchor_id,
                payload_hash: pending_hash.clone(),
            }
        );

        let mut conflicting_pending = pending.as_ref().clone();
        conflicting_pending.project_id = "different-project".to_string();
        assert_eq!(
            sink.record_pending(Arc::new(conflicting_pending)).await,
            SinkAck::Conflict { anchor_id }
        );
        assert_eq!(
            sink.record_terminal(terminal.clone()).await,
            SinkAck::Applied {
                anchor_id,
                payload_hash: terminal_hash.clone(),
            }
        );
        assert_eq!(
            sink.record_terminal(terminal.clone()).await,
            SinkAck::AlreadyApplied {
                anchor_id,
                payload_hash: terminal_hash.clone(),
            }
        );
        assert_eq!(
            sink.record_pending(pending.clone()).await,
            SinkAck::AlreadyTerminal {
                anchor_id,
                pending_hash,
                terminal_hash,
            }
        );

        let mut conflicting_terminal = terminal.as_ref().clone();
        conflicting_terminal.observed_progress = 2;
        assert_eq!(
            sink.record_terminal(Arc::new(conflicting_terminal)).await,
            SinkAck::Conflict { anchor_id }
        );
        assert!(sink.unresolved_pending().is_empty());
    }

    #[tokio::test]
    async fn sink_refuses_capacity_without_evicting_or_masking_retries() {
        let sink = InMemoryTrajectorySink::new(1);
        let first = pending(1);
        let second = pending(2);
        let first_hash = first.payload_hash().unwrap();

        assert!(matches!(
            sink.record_pending(first.clone()).await,
            SinkAck::Applied { .. }
        ));
        assert_eq!(
            sink.record_pending(second.clone()).await,
            SinkAck::TransientRefused {
                anchor_id: second.anchor_id(),
                reason: TransientRefusalReason::EvidenceCapacity,
            }
        );
        assert_eq!(
            sink.record_pending(first.clone()).await,
            SinkAck::AlreadyApplied {
                anchor_id: first.anchor_id(),
                payload_hash: first_hash.clone(),
            }
        );

        let unresolved = sink.unresolved_pending();
        assert_eq!(unresolved.len(), 1);
        assert_eq!(unresolved[0].payload.anchor_id(), first.anchor_id());
        assert_eq!(
            unresolved[0].canonical_bytes.as_ref(),
            first.canonical_bytes().unwrap()
        );
        assert_eq!(unresolved[0].payload_hash, first_hash);

        assert!(matches!(
            sink.record_terminal(terminal(&first)).await,
            SinkAck::Applied { .. }
        ));
        assert_eq!(
            sink.record_pending(second.clone()).await,
            SinkAck::TransientRefused {
                anchor_id: second.anchor_id(),
                reason: TransientRefusalReason::EvidenceCapacity,
            }
        );
        assert!(sink.unresolved_pending().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn sink_delay_and_failure_injection_are_deterministic() {
        let sink = InMemoryTrajectorySink::new(1);
        let pending = pending(1);
        sink.set_delay(SinkOperation::Pending, Duration::from_secs(5));
        sink.inject_transient_failures(SinkOperation::Pending, 1);

        let attempt = tokio::spawn(sink.record_pending(pending.clone()));
        tokio::task::yield_now().await;
        assert!(!attempt.is_finished());
        assert!(sink.unresolved_pending().is_empty());
        tokio::time::advance(Duration::from_secs(5)).await;
        assert_eq!(
            attempt.await.unwrap(),
            SinkAck::Failed {
                anchor_id: pending.anchor_id(),
                stable_class: SinkFailureClass::InjectedTransient,
            }
        );
        assert!(sink.unresolved_pending().is_empty());

        sink.set_delay(SinkOperation::Pending, Duration::ZERO);
        assert!(matches!(
            sink.record_pending(pending.clone()).await,
            SinkAck::Applied { .. }
        ));
        sink.set_permanent_failure(SinkOperation::Terminal, true);
        assert_eq!(
            sink.record_terminal(terminal(&pending)).await,
            SinkAck::Failed {
                anchor_id: pending.anchor_id(),
                stable_class: SinkFailureClass::InjectedPermanent,
            }
        );
        assert_eq!(sink.unresolved_pending().len(), 1);
        sink.set_permanent_failure(SinkOperation::Terminal, false);
        assert!(matches!(
            sink.record_terminal(terminal(&pending)).await,
            SinkAck::Applied { .. }
        ));
    }

    #[tokio::test(start_paused = true)]
    async fn canceling_a_delayed_sink_future_leaves_no_fact_or_owned_payload() {
        let sink = InMemoryTrajectorySink::new(1);
        let pending = pending(1);
        sink.set_delay(SinkOperation::Pending, Duration::from_secs(60));

        let attempt = tokio::spawn(sink.record_pending(pending.clone()));
        tokio::task::yield_now().await;
        assert_eq!(Arc::strong_count(&pending), 2);
        attempt.abort();
        assert!(attempt.await.unwrap_err().is_cancelled());

        assert_eq!(Arc::strong_count(&pending), 1);
        assert!(sink.unresolved_pending().is_empty());
    }

    #[tokio::test(start_paused = true)]
    async fn delivery_is_bounded_idempotent_delayed_and_never_starts_replay() {
        let delivery = InMemoryTrajectoryDelivery::new(1);
        let (first, starts) = closed_window(Uuid::from_u128(1));
        delivery.set_delay(Duration::from_secs(5));

        let future = delivery.deliver(first.clone());
        assert_eq!(Arc::strong_count(&first), 1);
        let attempt = tokio::spawn(future);
        tokio::task::yield_now().await;
        assert!(!attempt.is_finished());
        tokio::time::advance(Duration::from_secs(5)).await;
        assert_eq!(
            attempt.await.unwrap(),
            DeliveryAck::Delivered {
                anchor_id: first.anchor_id(),
            }
        );
        assert_eq!(starts.load(Ordering::SeqCst), 0);

        delivery.set_delay(Duration::ZERO);
        assert_eq!(
            delivery.deliver(first.clone()).await,
            DeliveryAck::AlreadyDelivered {
                anchor_id: first.anchor_id(),
            }
        );
        let (second, second_starts) = closed_window(Uuid::from_u128(2));
        assert_eq!(
            delivery.deliver(second.clone()).await,
            DeliveryAck::Failed {
                anchor_id: second.anchor_id(),
                stable_class: DeliveryFailureClass::CapacityExceeded,
            }
        );
        assert_eq!(delivery.delivered_anchor_ids(), vec![first.anchor_id()]);
        assert_eq!(starts.load(Ordering::SeqCst), 0);
        assert_eq!(second_starts.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn delivery_failure_injection_uses_only_stable_classes() {
        let delivery = InMemoryTrajectoryDelivery::new(1);
        let (window, starts) = closed_window(Uuid::from_u128(1));
        delivery.inject_transient_failures(1);

        let transient = delivery.deliver(window.clone()).await;
        assert_eq!(
            transient,
            DeliveryAck::Failed {
                anchor_id: window.anchor_id(),
                stable_class: DeliveryFailureClass::InjectedTransient,
            }
        );
        assert!(delivery.delivered_anchor_ids().is_empty());
        delivery.set_permanent_failure(true);
        let permanent = delivery.deliver(window.clone()).await;
        assert_eq!(
            permanent,
            DeliveryAck::Failed {
                anchor_id: window.anchor_id(),
                stable_class: DeliveryFailureClass::InjectedPermanent,
            }
        );
        assert!(delivery.delivered_anchor_ids().is_empty());
        delivery.set_permanent_failure(false);
        assert_eq!(
            delivery.deliver(window.clone()).await,
            DeliveryAck::Delivered {
                anchor_id: window.anchor_id(),
            }
        );
        assert_eq!(starts.load(Ordering::SeqCst), 0);
        assert_eq!(
            SinkFailureClass::InjectedPermanent.as_str(),
            "router.sink.injected_permanent"
        );
        assert_eq!(
            DeliveryFailureClass::InjectedPermanent.as_str(),
            "router.delivery.injected_permanent"
        );
        assert!(!format!("{permanent:?}").contains("fixture"));
    }
}
