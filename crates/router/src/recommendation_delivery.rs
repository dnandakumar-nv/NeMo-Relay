// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded ownership and exact retry for immutable recommendation audits.

#![allow(dead_code)] // Task 8 runtime wiring consumes this owner after its focused gate.

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, oneshot, watch};
use tokio::task::AbortHandle;
use uuid::Uuid;

use crate::decision_audit::DecisionAuditV1;
use crate::ledger::command::{WriterFailure, WriterFailureClass};
use crate::ledger::model::LedgerErrorClass;
use crate::ledger::repository::decision::DecisionAuditAck;
use crate::ledger::writer::LedgerWriterClient;

const RECOMMENDATION_DELIVERY_SLOTS_V1: usize = 4;
const TRANSACTION_START_SLICE: Duration = Duration::from_millis(250);
const RETRY_BACKOFF_INITIAL: Duration = Duration::from_millis(10);
const RETRY_BACKOFF_MAX: Duration = Duration::from_secs(1);

type WriterAttemptFuture =
    Pin<Box<dyn Future<Output = Result<DecisionAuditAck, WriterFailure>> + Send + 'static>>;

trait RecommendationAuditWriterV1: Send + Sync + 'static {
    fn record_decision_audit(
        &self,
        audit: Arc<DecisionAuditV1>,
        max_evidence_records: u64,
        conflict_health_event_id: Uuid,
        transaction_start_deadline: Instant,
    ) -> WriterAttemptFuture;
}

impl RecommendationAuditWriterV1 for LedgerWriterClient {
    fn record_decision_audit(
        &self,
        audit: Arc<DecisionAuditV1>,
        max_evidence_records: u64,
        conflict_health_event_id: Uuid,
        transaction_start_deadline: Instant,
    ) -> WriterAttemptFuture {
        let writer = self.clone();
        Box::pin(async move {
            writer
                .record_decision_audit_with_start_deadline(
                    audit,
                    max_evidence_records,
                    conflict_health_event_id,
                    transaction_start_deadline,
                )
                .await
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecommendationDeliveryHealthV1 {
    Healthy,
    DegradedPending { pending: usize },
    PermanentFault { retained: usize },
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecommendationAdmissionErrorV1 {
    Closed,
    DegradedPending,
    PermanentFault,
    Saturated,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecommendationStaleReasonV1 {
    AuthorityChanged,
    SourceRetiring,
    EvidenceChanged,
    OriginatingProcessNotLive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecommendationFirstAckV1 {
    Applied,
    AlreadyApplied,
    DroppedStale(RecommendationStaleReasonV1),
    PendingRetry,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecommendationDeliveryErrorV1 {
    InvalidAudit,
    Deadline,
    Aborted,
    PermanentFault,
    RuntimeUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RecommendationDrainErrorV1 {
    Deadline,
    PermanentFault,
    Aborted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeliveryLifecycleV1 {
    Running,
    Draining { deadline: Instant },
    Aborted,
}

struct PendingAuditV1 {
    audit: Arc<DecisionAuditV1>,
    max_evidence_records: u64,
    conflict_health_event_id: Uuid,
    _slot: OwnedSemaphorePermit,
}

struct DeliveryStateV1 {
    lifecycle: DeliveryLifecycleV1,
    permanent_fault: bool,
    pending: BTreeMap<Uuid, PendingAuditV1>,
    tasks: HashMap<Uuid, AbortHandle>,
}

struct DeliverySharedV1 {
    writer: Arc<dyn RecommendationAuditWriterV1>,
    slots: Arc<Semaphore>,
    state: Mutex<DeliveryStateV1>,
    lifecycle: watch::Sender<DeliveryLifecycleV1>,
    retention_pressure_epoch: AtomicU64,
    retention_pressure: watch::Sender<u64>,
    changed: Notify,
}

/// Cloneable admission, delivery-health, and retry authority.
#[derive(Clone)]
pub(crate) struct RecommendationDeliveryServiceV1 {
    shared: Arc<DeliverySharedV1>,
    max_evidence_records: u64,
}

/// One of the four bounded recommendation/audit ownership slots.
pub(crate) struct RecommendationDeliveryPermitV1 {
    shared: Arc<DeliverySharedV1>,
    max_evidence_records: u64,
    slot: Option<OwnedSemaphorePermit>,
}

struct DeliveryTaskGuardV1 {
    shared: Arc<DeliverySharedV1>,
    decision_id: Uuid,
    started: bool,
}

impl Drop for DeliveryTaskGuardV1 {
    fn drop(&mut self) {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        state.tasks.remove(&self.decision_id);
        if !self.started {
            state.pending.remove(&self.decision_id);
        }
        drop(state);
        self.shared.changed.notify_waiters();
    }
}

impl Drop for RecommendationDeliveryPermitV1 {
    fn drop(&mut self) {
        if self.slot.take().is_some() {
            self.shared.changed.notify_waiters();
        }
    }
}

impl RecommendationDeliveryServiceV1 {
    pub(crate) fn new(
        writer: LedgerWriterClient,
        max_evidence_records: u64,
    ) -> Result<Self, RecommendationDeliveryErrorV1> {
        Self::new_with_writer(Arc::new(writer), max_evidence_records)
    }

    fn new_with_writer(
        writer: Arc<dyn RecommendationAuditWriterV1>,
        max_evidence_records: u64,
    ) -> Result<Self, RecommendationDeliveryErrorV1> {
        if max_evidence_records == 0 {
            return Err(RecommendationDeliveryErrorV1::InvalidAudit);
        }
        let (lifecycle, _) = watch::channel(DeliveryLifecycleV1::Running);
        let (retention_pressure, _) = watch::channel(0);
        Ok(Self {
            shared: Arc::new(DeliverySharedV1 {
                writer,
                slots: Arc::new(Semaphore::new(RECOMMENDATION_DELIVERY_SLOTS_V1)),
                state: Mutex::new(DeliveryStateV1 {
                    lifecycle: DeliveryLifecycleV1::Running,
                    permanent_fault: false,
                    pending: BTreeMap::new(),
                    tasks: HashMap::new(),
                }),
                lifecycle,
                retention_pressure_epoch: AtomicU64::new(0),
                retention_pressure,
                changed: Notify::new(),
            }),
            max_evidence_records,
        })
    }

    pub(crate) fn try_admit(
        &self,
    ) -> Result<RecommendationDeliveryPermitV1, RecommendationAdmissionErrorV1> {
        let state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state.lifecycle != DeliveryLifecycleV1::Running {
            return Err(RecommendationAdmissionErrorV1::Closed);
        }
        if state.permanent_fault {
            return Err(RecommendationAdmissionErrorV1::PermanentFault);
        }
        if !state.pending.is_empty() {
            return Err(RecommendationAdmissionErrorV1::DegradedPending);
        }
        let slot = self
            .shared
            .slots
            .clone()
            .try_acquire_owned()
            .map_err(|_| RecommendationAdmissionErrorV1::Saturated)?;
        drop(state);
        Ok(RecommendationDeliveryPermitV1 {
            shared: self.shared.clone(),
            max_evidence_records: self.max_evidence_records,
            slot: Some(slot),
        })
    }

    pub(crate) fn health(&self) -> RecommendationDeliveryHealthV1 {
        let state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        if state.lifecycle == DeliveryLifecycleV1::Aborted {
            RecommendationDeliveryHealthV1::Closed
        } else if state.permanent_fault {
            RecommendationDeliveryHealthV1::PermanentFault {
                retained: state.pending.len(),
            }
        } else if !state.pending.is_empty() {
            RecommendationDeliveryHealthV1::DegradedPending {
                pending: state.pending.len(),
            }
        } else if state.lifecycle == DeliveryLifecycleV1::Running {
            RecommendationDeliveryHealthV1::Healthy
        } else {
            RecommendationDeliveryHealthV1::Closed
        }
    }

    pub(crate) fn subscribe_retention_pressure(&self) -> watch::Receiver<u64> {
        self.shared.retention_pressure.subscribe()
    }

    pub(crate) fn close_until(&self, deadline: Instant) {
        let lifecycle = {
            let mut state = self
                .shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            state.lifecycle = match state.lifecycle {
                DeliveryLifecycleV1::Running => DeliveryLifecycleV1::Draining { deadline },
                DeliveryLifecycleV1::Draining { deadline: current } => {
                    DeliveryLifecycleV1::Draining {
                        deadline: current.min(deadline),
                    }
                }
                DeliveryLifecycleV1::Aborted => DeliveryLifecycleV1::Aborted,
            };
            state.lifecycle
        };
        self.shared.lifecycle.send_replace(lifecycle);
        self.shared.changed.notify_waiters();
    }

    pub(crate) async fn drain_until(
        &self,
        deadline: Instant,
    ) -> Result<(), RecommendationDrainErrorV1> {
        self.close_until(deadline);
        let tokio_deadline = tokio::time::Instant::from_std(deadline);
        loop {
            let changed = self.shared.changed.notified();
            let (lifecycle, permanent_fault, pending_empty) = {
                let state = self
                    .shared
                    .state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner());
                (
                    state.lifecycle,
                    state.permanent_fault,
                    state.pending.is_empty(),
                )
            };
            if lifecycle == DeliveryLifecycleV1::Aborted {
                return Err(RecommendationDrainErrorV1::Aborted);
            }
            if pending_empty
                && self.shared.slots.available_permits() == RECOMMENDATION_DELIVERY_SLOTS_V1
            {
                return Ok(());
            }
            tokio::select! {
                biased;
                () = tokio::time::sleep_until(tokio_deadline) => {
                    return Err(if permanent_fault {
                        RecommendationDrainErrorV1::PermanentFault
                    } else {
                        RecommendationDrainErrorV1::Deadline
                    });
                }
                () = changed => {}
            }
        }
    }

    /// The writer must be fenced before this synchronous delivery abort.
    pub(crate) fn abort(&self) {
        let tasks = {
            let mut state = self
                .shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if state.lifecycle == DeliveryLifecycleV1::Aborted {
                return;
            }
            state.lifecycle = DeliveryLifecycleV1::Aborted;
            state.pending.clear();
            state
                .tasks
                .drain()
                .map(|(_, task)| task)
                .collect::<Vec<_>>()
        };
        self.shared.slots.close();
        self.shared
            .lifecycle
            .send_replace(DeliveryLifecycleV1::Aborted);
        for task in tasks {
            task.abort();
        }
        self.shared.changed.notify_waiters();
    }
}

impl RecommendationDeliveryPermitV1 {
    pub(crate) async fn submit_until(
        mut self,
        audit: Arc<DecisionAuditV1>,
        foreground_deadline: Instant,
    ) -> Result<RecommendationFirstAckV1, RecommendationDeliveryErrorV1> {
        if audit.validate_frozen().is_err() {
            self.shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .permanent_fault = true;
            self.shared.changed.notify_waiters();
            return Err(RecommendationDeliveryErrorV1::InvalidAudit);
        }
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| RecommendationDeliveryErrorV1::RuntimeUnavailable)?;
        let decision_id = audit.parent.decision_id;
        let conflict_health_event_id = Uuid::now_v7();
        let (first_send, first_receive) = oneshot::channel();
        let (start, started) = oneshot::channel();

        {
            let mut state = self
                .shared
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            if state.lifecycle == DeliveryLifecycleV1::Aborted {
                return Err(RecommendationDeliveryErrorV1::Aborted);
            }
            if state.permanent_fault {
                return Err(RecommendationDeliveryErrorV1::PermanentFault);
            }
            if state.pending.contains_key(&decision_id) {
                state.permanent_fault = true;
                self.shared.changed.notify_waiters();
                return Err(RecommendationDeliveryErrorV1::PermanentFault);
            }
            let slot = self
                .slot
                .take()
                .ok_or(RecommendationDeliveryErrorV1::InvalidAudit)?;
            state.pending.insert(
                decision_id,
                PendingAuditV1 {
                    audit,
                    max_evidence_records: self.max_evidence_records,
                    conflict_health_event_id,
                    _slot: slot,
                },
            );
            let task_shared = self.shared.clone();
            let task = runtime.spawn(async move {
                let mut guard = DeliveryTaskGuardV1 {
                    shared: task_shared.clone(),
                    decision_id,
                    started: false,
                };
                if started.await.is_err() {
                    return;
                }
                guard.started = true;
                run_pending_audit(task_shared, decision_id, foreground_deadline, first_send).await;
            });
            state.tasks.insert(decision_id, task.abort_handle());
        }
        self.shared.changed.notify_waiters();
        if start.send(()).is_err() {
            cleanup_unstarted_submission(&self.shared, decision_id);
            return Err(RecommendationDeliveryErrorV1::Aborted);
        }
        wait_for_first_result(&self.shared, first_receive, foreground_deadline).await
    }
}

fn cleanup_unstarted_submission(shared: &DeliverySharedV1, decision_id: Uuid) {
    let mut state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    state.tasks.remove(&decision_id);
    state.pending.remove(&decision_id);
    drop(state);
    shared.changed.notify_waiters();
}

async fn wait_for_first_result(
    shared: &DeliverySharedV1,
    mut receiver: oneshot::Receiver<RecommendationFirstAckV1>,
    deadline: Instant,
) -> Result<RecommendationFirstAckV1, RecommendationDeliveryErrorV1> {
    let mut lifecycle = shared.lifecycle.subscribe();
    let tokio_deadline = tokio::time::Instant::from_std(deadline);
    loop {
        if *lifecycle.borrow() == DeliveryLifecycleV1::Aborted {
            return Err(RecommendationDeliveryErrorV1::Aborted);
        }
        tokio::select! {
            biased;
            result = &mut receiver => {
                return match result {
                    Ok(result) => Ok(result),
                    Err(_) => {
                        let permanent_fault = shared
                            .state
                            .lock()
                            .unwrap_or_else(|error| error.into_inner())
                            .permanent_fault;
                        if permanent_fault {
                            Err(RecommendationDeliveryErrorV1::PermanentFault)
                        } else {
                            Ok(RecommendationFirstAckV1::PendingRetry)
                        }
                    }
                };
            }
            () = tokio::time::sleep_until(tokio_deadline) => {
                return Ok(RecommendationFirstAckV1::PendingRetry);
            }
            changed = lifecycle.changed() => {
                if changed.is_err() || *lifecycle.borrow() == DeliveryLifecycleV1::Aborted {
                    return Err(RecommendationDeliveryErrorV1::Aborted);
                }
            }
        }
    }
}

async fn run_pending_audit(
    shared: Arc<DeliverySharedV1>,
    decision_id: Uuid,
    foreground_deadline: Instant,
    first_send: oneshot::Sender<RecommendationFirstAckV1>,
) {
    let mut first_send = Some(first_send);
    let mut first_attempt = true;
    let mut retention_notified = false;
    let mut backoff = RETRY_BACKOFF_INITIAL;
    loop {
        let Some(start_deadline) =
            attempt_start_deadline(&shared, first_attempt.then_some(foreground_deadline))
        else {
            send_first(&mut first_send, RecommendationFirstAckV1::PendingRetry);
            wait_for_abort(&shared).await;
            return;
        };
        if Instant::now() >= start_deadline {
            if first_attempt {
                first_attempt = false;
                send_first(&mut first_send, RecommendationFirstAckV1::PendingRetry);
                continue;
            }
            wait_for_abort(&shared).await;
            return;
        }
        let Some((audit, max_evidence_records, conflict_health_event_id)) =
            pending_command(&shared, decision_id)
        else {
            return;
        };
        let attempt = match catch_unwind(AssertUnwindSafe(|| {
            shared.writer.record_decision_audit(
                audit,
                max_evidence_records,
                conflict_health_event_id,
                start_deadline,
            )
        })) {
            Ok(attempt) => PanicCaughtWriterFuture {
                inner: Some(attempt),
            },
            Err(_) => {
                mark_permanent(&shared);
                send_permanent(&mut first_send);
                return;
            }
        };
        let result = await_attempt_or_abort(&shared, attempt).await;
        first_attempt = false;
        match classify_attempt(result) {
            AttemptDispositionV1::Applied => {
                remove_pending(&shared, decision_id);
                send_first(&mut first_send, RecommendationFirstAckV1::Applied);
                return;
            }
            AttemptDispositionV1::AlreadyApplied => {
                remove_pending(&shared, decision_id);
                send_first(&mut first_send, RecommendationFirstAckV1::AlreadyApplied);
                return;
            }
            AttemptDispositionV1::Dropped(reason) => {
                remove_pending(&shared, decision_id);
                send_first(
                    &mut first_send,
                    RecommendationFirstAckV1::DroppedStale(reason),
                );
                return;
            }
            AttemptDispositionV1::RetentionRequired => {
                if !retention_notified {
                    signal_retention_pressure(&shared);
                    retention_notified = true;
                }
                send_first(&mut first_send, RecommendationFirstAckV1::PendingRetry);
            }
            AttemptDispositionV1::Retry => {
                send_first(&mut first_send, RecommendationFirstAckV1::PendingRetry);
            }
            AttemptDispositionV1::Permanent => {
                mark_permanent(&shared);
                send_permanent(&mut first_send);
                return;
            }
            AttemptDispositionV1::Aborted => return,
        }
        if !wait_retry_backoff(&shared, backoff).await {
            return;
        }
        backoff = next_backoff(backoff);
    }
}

fn pending_command(
    shared: &DeliverySharedV1,
    decision_id: Uuid,
) -> Option<(Arc<DecisionAuditV1>, u64, Uuid)> {
    let state = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    let pending = state.pending.get(&decision_id)?;
    Some((
        pending.audit.clone(),
        pending.max_evidence_records,
        pending.conflict_health_event_id,
    ))
}

fn attempt_start_deadline(
    shared: &DeliverySharedV1,
    foreground_deadline: Option<Instant>,
) -> Option<Instant> {
    let lifecycle = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .lifecycle;
    if lifecycle == DeliveryLifecycleV1::Aborted {
        return None;
    }
    let mut deadline = Instant::now().checked_add(TRANSACTION_START_SLICE)?;
    if let Some(foreground_deadline) = foreground_deadline {
        deadline = deadline.min(foreground_deadline);
    }
    if let DeliveryLifecycleV1::Draining { deadline: shutdown } = lifecycle {
        deadline = deadline.min(shutdown);
    }
    Some(deadline)
}

struct PanicCaughtWriterFuture {
    inner: Option<WriterAttemptFuture>,
}

impl Future for PanicCaughtWriterFuture {
    type Output = Result<Result<DecisionAuditAck, WriterFailure>, ()>;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.as_mut().get_mut();
        let Some(inner) = this.inner.as_mut() else {
            return Poll::Ready(Err(()));
        };
        match catch_unwind(AssertUnwindSafe(|| inner.as_mut().poll(context))) {
            Ok(Poll::Ready(result)) => {
                if drop_writer_attempt(&mut this.inner) {
                    Poll::Ready(Ok(result))
                } else {
                    Poll::Ready(Err(()))
                }
            }
            Ok(Poll::Pending) => Poll::Pending,
            Err(_) => {
                drop_writer_attempt(&mut this.inner);
                Poll::Ready(Err(()))
            }
        }
    }
}

impl Drop for PanicCaughtWriterFuture {
    fn drop(&mut self) {
        drop_writer_attempt(&mut self.inner);
    }
}

fn drop_writer_attempt(attempt: &mut Option<WriterAttemptFuture>) -> bool {
    let Some(attempt) = attempt.take() else {
        return true;
    };
    catch_unwind(AssertUnwindSafe(|| drop(attempt))).is_ok()
}

async fn await_attempt_or_abort(
    shared: &DeliverySharedV1,
    mut attempt: PanicCaughtWriterFuture,
) -> AttemptResultV1 {
    let mut lifecycle = shared.lifecycle.subscribe();
    loop {
        tokio::select! {
            biased;
            result = &mut attempt => return match result {
                Ok(result) => AttemptResultV1::Writer(result),
                Err(()) => AttemptResultV1::Panicked,
            },
            changed = lifecycle.changed() => {
                if changed.is_err() || *lifecycle.borrow() == DeliveryLifecycleV1::Aborted {
                    return AttemptResultV1::Aborted;
                }
            }
        }
    }
}

enum AttemptResultV1 {
    Writer(Result<DecisionAuditAck, WriterFailure>),
    Panicked,
    Aborted,
}

enum AttemptDispositionV1 {
    Applied,
    AlreadyApplied,
    Dropped(RecommendationStaleReasonV1),
    RetentionRequired,
    Retry,
    Permanent,
    Aborted,
}

fn classify_attempt(result: AttemptResultV1) -> AttemptDispositionV1 {
    match result {
        AttemptResultV1::Writer(Ok(DecisionAuditAck::Applied)) => AttemptDispositionV1::Applied,
        AttemptResultV1::Writer(Ok(DecisionAuditAck::AlreadyApplied)) => {
            AttemptDispositionV1::AlreadyApplied
        }
        AttemptResultV1::Writer(Ok(DecisionAuditAck::RetentionRequired)) => {
            AttemptDispositionV1::RetentionRequired
        }
        AttemptResultV1::Writer(Ok(DecisionAuditAck::AuthorityChanged)) => {
            AttemptDispositionV1::Dropped(RecommendationStaleReasonV1::AuthorityChanged)
        }
        AttemptResultV1::Writer(Ok(DecisionAuditAck::SourceRetiring)) => {
            AttemptDispositionV1::Dropped(RecommendationStaleReasonV1::SourceRetiring)
        }
        AttemptResultV1::Writer(Ok(DecisionAuditAck::EvidenceChanged)) => {
            AttemptDispositionV1::Dropped(RecommendationStaleReasonV1::EvidenceChanged)
        }
        AttemptResultV1::Writer(Ok(DecisionAuditAck::OriginatingProcessNotLive)) => {
            AttemptDispositionV1::Dropped(RecommendationStaleReasonV1::OriginatingProcessNotLive)
        }
        AttemptResultV1::Writer(Ok(DecisionAuditAck::Conflict)) | AttemptResultV1::Panicked => {
            AttemptDispositionV1::Permanent
        }
        AttemptResultV1::Writer(Ok(DecisionAuditAck::TransactionNotStarted)) => {
            AttemptDispositionV1::Retry
        }
        AttemptResultV1::Writer(Err(error)) => match error.class() {
            WriterFailureClass::Protocol
            | WriterFailureClass::Panicked
            | WriterFailureClass::Repository(
                LedgerErrorClass::CorruptDatabase
                | LedgerErrorClass::IdentityInvariant
                | LedgerErrorClass::CanonicalizationFailed,
            ) => AttemptDispositionV1::Permanent,
            WriterFailureClass::Full
            | WriterFailureClass::Closing
            | WriterFailureClass::Deadline
            | WriterFailureClass::Exited
            | WriterFailureClass::Aborted
            | WriterFailureClass::Repository(_) => AttemptDispositionV1::Retry,
        },
        AttemptResultV1::Aborted => AttemptDispositionV1::Aborted,
    }
}

fn remove_pending(shared: &DeliverySharedV1, decision_id: Uuid) {
    shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .pending
        .remove(&decision_id);
    shared.changed.notify_waiters();
}

fn mark_permanent(shared: &DeliverySharedV1) {
    shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .permanent_fault = true;
    shared.changed.notify_waiters();
}

fn send_permanent(first_send: &mut Option<oneshot::Sender<RecommendationFirstAckV1>>) {
    if let Some(sender) = first_send.take() {
        drop(sender);
    }
}

fn send_first(
    sender: &mut Option<oneshot::Sender<RecommendationFirstAckV1>>,
    outcome: RecommendationFirstAckV1,
) {
    if let Some(sender) = sender.take() {
        let _ = sender.send(outcome);
    }
}

fn signal_retention_pressure(shared: &DeliverySharedV1) {
    let next = shared
        .retention_pressure_epoch
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |current| {
            Some(current.saturating_add(1))
        })
        .unwrap_or(u64::MAX)
        .saturating_add(1);
    shared.retention_pressure.send_replace(next);
}

async fn wait_retry_backoff(shared: &DeliverySharedV1, backoff: Duration) -> bool {
    let lifecycle = shared
        .state
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .lifecycle;
    if lifecycle == DeliveryLifecycleV1::Aborted {
        return false;
    }
    let wake = Instant::now().checked_add(backoff);
    let wake = match (wake, lifecycle) {
        (Some(wake), DeliveryLifecycleV1::Draining { deadline }) => wake.min(deadline),
        (Some(wake), _) => wake,
        (None, _) => return false,
    };
    let mut lifecycle = shared.lifecycle.subscribe();
    loop {
        tokio::select! {
            biased;
            () = tokio::time::sleep_until(tokio::time::Instant::from_std(wake)) => {
                return match *lifecycle.borrow() {
                    DeliveryLifecycleV1::Running => true,
                    DeliveryLifecycleV1::Draining { deadline } => Instant::now() < deadline,
                    DeliveryLifecycleV1::Aborted => false,
                };
            }
            changed = lifecycle.changed() => {
                if changed.is_err() || *lifecycle.borrow() == DeliveryLifecycleV1::Aborted {
                    return false;
                }
            }
        }
    }
}

async fn wait_for_abort(shared: &DeliverySharedV1) {
    let mut lifecycle = shared.lifecycle.subscribe();
    while *lifecycle.borrow() != DeliveryLifecycleV1::Aborted {
        if lifecycle.changed().await.is_err() {
            return;
        }
    }
}

fn next_backoff(current: Duration) -> Duration {
    current
        .checked_mul(2)
        .unwrap_or(RETRY_BACKOFF_MAX)
        .min(RETRY_BACKOFF_MAX)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use nemo_relay_types::api::llm::LlmApiFamily;

    use crate::candidate_set::{CandidateSetMemberInputV1, build_candidate_set_from_members_v1};
    use crate::canonical_query::{
        CanonicalRoutingQueryArtifactV1, CanonicalRoutingQueryV1, CanonicalTaskV1,
    };
    use crate::decision_audit::{
        AuditF64V1, DecisionCandidateInputV1, DecisionCandidateReasonV1,
        DecisionCandidateSummaryInputV1, DecisionFinalReasonV1, DecisionParentInputV1,
        PreparedCanonicalQueryV1,
    };
    use crate::embedding_identity::CANONICAL_ROUTING_QUERY_SCHEMA_V1;
    use crate::fingerprint::{canonical_serialize_bytes, sha256_hex};
    use crate::routing_partition::{
        RoutingPartitionBaseV1, RoutingPartitionInputV1, build_routing_partition_base_v1,
        build_routing_partition_from_input_v1,
    };

    use super::*;

    fn hash(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn uuid(suffix: u64) -> Uuid {
        Uuid::parse_str(&format!("018f1e66-0000-7000-8000-{suffix:012x}")).unwrap()
    }

    fn audit(suffix: u64) -> Arc<DecisionAuditV1> {
        let query = CanonicalRoutingQueryV1 {
            schema: CANONICAL_ROUTING_QUERY_SCHEMA_V1.to_string(),
            instructions: Vec::new(),
            current_task: CanonicalTaskV1 {
                text: format!("route request {suffix}"),
            },
            bounded_context: Vec::new(),
            tool_schema_fingerprint: hash('1'),
            response_schema_fingerprint: None,
            required_capabilities: Vec::new(),
            position_features: None,
        };
        let canonical_bytes = canonical_serialize_bytes(&query).unwrap();
        let canonical_query_hash = sha256_hex(&canonical_bytes);
        let prepared_query =
            PreparedCanonicalQueryV1::from_artifact(CanonicalRoutingQueryArtifactV1 {
                query,
                canonical_bytes,
                canonical_query_hash: canonical_query_hash.clone(),
            })
            .unwrap();
        let learning_generation_id = uuid(100 + suffix);
        let base = RoutingPartitionBaseV1 {
            tenant_policy_hash: hash('a'),
            agent_policy_hash: hash('b'),
            policy_version_id: hash('c'),
            learning_generation_id,
            api_family: LlmApiFamily::OpenAIChatCompletions,
            transport_identity: "transport-v1".to_string(),
            anchor_model: "anchor".to_string(),
            anchor_revision: "anchor-r1".to_string(),
            evaluator_version: hash('e'),
            vector_space_id: hash('6'),
        };
        let base_artifact = build_routing_partition_base_v1(&base).unwrap();
        let partition = build_routing_partition_from_input_v1(&RoutingPartitionInputV1 {
            base,
            candidate_id: "candidate-a".to_string(),
            candidate_model: "model-a".to_string(),
            candidate_model_revision: "r1".to_string(),
            decoding_fingerprint: hash('d'),
        })
        .unwrap();
        let candidate_set = build_candidate_set_from_members_v1(&[CandidateSetMemberInputV1 {
            candidate_id: "candidate-a".to_string(),
            model: "model-a".to_string(),
            model_revision: "r1".to_string(),
            cost_rank: 1,
        }])
        .unwrap();
        let parent = DecisionParentInputV1 {
            decision_id: uuid(suffix),
            project_uuid: uuid(10),
            process_instance_id: uuid(11),
            config_generation_id: hash('7'),
            policy_version_id: hash('c'),
            learning_generation_id,
            pool_id: "pool-a".to_string(),
            candidate_id: None,
            primary_call_uuid: uuid(12 + suffix),
            canonical_query_hash,
            partition_base_json: base_artifact.canonical_json,
            partition_base_hash: base_artifact.partition_base_hash,
            vector_space_id: hash('6'),
            candidate_set_hash: candidate_set.candidate_set_hash,
            recommended_model: "anchor".to_string(),
            recommended_model_revision: "anchor-r1".to_string(),
            served_model: "anchor".to_string(),
            served_model_revision: "anchor-r1".to_string(),
            as_of_unix_ms: 1_800_000_000_000,
            decision_latency_ms: 1,
            final_reason: DecisionFinalReasonV1::EmbeddingUnavailable,
            created_at_unix_ms: 1_800_000_000_001,
        };
        let summary = DecisionCandidateSummaryInputV1 {
            candidate_id: "candidate-a".to_string(),
            rank_ordinal: 0,
            candidate_model: "model-a".to_string(),
            candidate_model_revision: "r1".to_string(),
            cost_rank: 1,
            learning_generation_id,
            vector_space_id: hash('6'),
            partition_id: None,
            decoding_fingerprint: hash('d'),
            top_k: 1,
            radius: AuditF64V1::new(1.0).unwrap(),
            min_points: 1,
            min_independent_roots: 1,
            min_effective_samples: AuditF64V1::new(1.0).unwrap(),
            min_coverage: AuditF64V1::new(0.0).unwrap(),
            time_decay_half_life_seconds: AuditF64V1::new(3_600.0).unwrap(),
            prior_success: AuditF64V1::new(1.0).unwrap(),
            prior_failure: AuditF64V1::new(1.0).unwrap(),
            familywise_credible_level: AuditF64V1::new(0.95).unwrap(),
            candidate_alpha: AuditF64V1::new(0.05).unwrap(),
            promotion_lower_bound: AuditF64V1::new(0.0).unwrap(),
            returned_neighbor_count: 0,
            within_radius_count: 0,
            labeled_point_count: 0,
            attempted_root_count: 0,
            labeled_root_count: 0,
            selected_root_count: 0,
            coverage: None,
            sum_weight: None,
            sum_weighted_label: None,
            sum_squared_weight: None,
            p_hat: None,
            effective_sample_size: None,
            beta_alpha: None,
            beta_beta: None,
            lower_bound: None,
            partition_gate_passed: None,
            points_gate_passed: None,
            roots_gate_passed: None,
            coverage_gate_passed: None,
            weight_gate_passed: None,
            effective_samples_gate_passed: None,
            beta_quantile_gate_passed: None,
            lower_bound_gate_passed: None,
            terminal_reason: DecisionCandidateReasonV1::NotEvaluatedAfterFallback,
        };
        Arc::new(
            DecisionAuditV1::new(
                parent,
                vec![DecisionCandidateInputV1 {
                    summary,
                    partition_artifact: partition,
                    neighbors: Vec::new(),
                }],
                prepared_query,
            )
            .unwrap(),
        )
    }

    enum FakeAction {
        Immediate(Result<DecisionAuditAck, WriterFailure>),
        Wait(Arc<Notify>, Result<DecisionAuditAck, WriterFailure>),
        Panic,
        DropPanic,
    }

    #[derive(Debug, Clone, Copy)]
    struct RecordedAttempt {
        audit_address: usize,
        conflict_health_event_id: Uuid,
        start_deadline: Instant,
        started_at: Instant,
    }

    struct FakeWriterState {
        actions: VecDeque<FakeAction>,
        attempts: Vec<RecordedAttempt>,
    }

    struct FakeWriter {
        state: Arc<Mutex<FakeWriterState>>,
        in_flight: Arc<AtomicUsize>,
        max_in_flight: Arc<AtomicUsize>,
    }

    struct FakeAttemptGuard {
        in_flight: Arc<AtomicUsize>,
    }

    impl Drop for FakeAttemptGuard {
        fn drop(&mut self) {
            self.in_flight.fetch_sub(1, Ordering::AcqRel);
        }
    }

    impl FakeWriter {
        fn new(actions: impl IntoIterator<Item = FakeAction>) -> Self {
            Self {
                state: Arc::new(Mutex::new(FakeWriterState {
                    actions: actions.into_iter().collect(),
                    attempts: Vec::new(),
                })),
                in_flight: Arc::new(AtomicUsize::new(0)),
                max_in_flight: Arc::new(AtomicUsize::new(0)),
            }
        }

        fn attempts(&self) -> Vec<RecordedAttempt> {
            self.state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .attempts
                .clone()
        }

        fn max_in_flight(&self) -> usize {
            self.max_in_flight.load(Ordering::Acquire)
        }
    }

    impl RecommendationAuditWriterV1 for FakeWriter {
        fn record_decision_audit(
            &self,
            audit: Arc<DecisionAuditV1>,
            _max_evidence_records: u64,
            conflict_health_event_id: Uuid,
            transaction_start_deadline: Instant,
        ) -> WriterAttemptFuture {
            let action = self
                .state
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .actions
                .pop_front()
                .unwrap_or(FakeAction::Immediate(Ok(
                    DecisionAuditAck::RetentionRequired,
                )));
            let state = self.state.clone();
            let in_flight = self.in_flight.clone();
            let max_in_flight = self.max_in_flight.clone();
            if matches!(&action, FakeAction::DropPanic) {
                return Box::pin(PanicOnDropFuture);
            }
            Box::pin(async move {
                let started_at = Instant::now();
                state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .attempts
                    .push(RecordedAttempt {
                        audit_address: Arc::as_ptr(&audit) as usize,
                        conflict_health_event_id,
                        start_deadline: transaction_start_deadline,
                        started_at,
                    });
                let current = in_flight.fetch_add(1, Ordering::AcqRel) + 1;
                max_in_flight.fetch_max(current, Ordering::AcqRel);
                let _guard = FakeAttemptGuard { in_flight };
                match action {
                    FakeAction::Immediate(result) => result,
                    FakeAction::Wait(release, result) => {
                        release.notified().await;
                        result
                    }
                    FakeAction::Panic => panic!("injected writer panic"),
                    FakeAction::DropPanic => unreachable!("handled before async construction"),
                }
            })
        }
    }

    struct PanicOnDropFuture;

    impl Future for PanicOnDropFuture {
        type Output = Result<DecisionAuditAck, WriterFailure>;

        fn poll(self: Pin<&mut Self>, _context: &mut Context<'_>) -> Poll<Self::Output> {
            Poll::Ready(Ok(DecisionAuditAck::Applied))
        }
    }

    impl Drop for PanicOnDropFuture {
        fn drop(&mut self) {
            panic!("injected writer future drop panic");
        }
    }

    fn service(writer: Arc<FakeWriter>) -> RecommendationDeliveryServiceV1 {
        RecommendationDeliveryServiceV1::new_with_writer(writer, 10_000).unwrap()
    }

    async fn wait_for_health(
        service: &RecommendationDeliveryServiceV1,
        expected: RecommendationDeliveryHealthV1,
    ) {
        let deadline = tokio::time::Instant::now() + Duration::from_secs(2);
        loop {
            if service.health() == expected {
                return;
            }
            assert!(tokio::time::Instant::now() < deadline);
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }

    #[test]
    fn backoff_doubles_and_caps_at_one_second() {
        let mut current = RETRY_BACKOFF_INITIAL;
        let expected = [20, 40, 80, 160, 320, 640, 1_000, 1_000];
        for millis in expected {
            current = next_backoff(current);
            assert_eq!(current, Duration::from_millis(millis));
        }
    }

    #[tokio::test]
    async fn admission_is_bounded_to_four_and_drain_waits_for_unsubmitted_work() {
        let writer = Arc::new(FakeWriter::new([]));
        let service = service(writer);
        let mut permits = (0..RECOMMENDATION_DELIVERY_SLOTS_V1)
            .map(|_| service.try_admit().unwrap())
            .collect::<Vec<_>>();
        assert!(matches!(
            service.try_admit(),
            Err(RecommendationAdmissionErrorV1::Saturated)
        ));

        let draining = service.clone();
        let drain = tokio::spawn(async move {
            draining
                .drain_until(Instant::now() + Duration::from_secs(1))
                .await
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!drain.is_finished());
        permits.clear();
        assert_eq!(drain.await.unwrap(), Ok(()));
        assert!(matches!(
            service.try_admit(),
            Err(RecommendationAdmissionErrorV1::Closed)
        ));
    }

    #[tokio::test]
    async fn transient_retry_reuses_exact_arc_and_conflict_identity() {
        let writer = Arc::new(FakeWriter::new([
            FakeAction::Immediate(Err(WriterFailure::new(WriterFailureClass::Deadline))),
            FakeAction::Immediate(Ok(DecisionAuditAck::Applied)),
        ]));
        let service = service(writer.clone());
        let audit = audit(2);
        let expected_address = Arc::as_ptr(&audit) as usize;

        assert_eq!(
            service
                .try_admit()
                .unwrap()
                .submit_until(audit, Instant::now() + Duration::from_secs(1))
                .await
                .unwrap(),
            RecommendationFirstAckV1::PendingRetry
        );
        wait_for_health(&service, RecommendationDeliveryHealthV1::Healthy).await;

        let attempts = writer.attempts();
        assert_eq!(attempts.len(), 2);
        assert!(
            attempts
                .iter()
                .all(|attempt| attempt.audit_address == expected_address)
        );
        assert_eq!(
            attempts[0].conflict_health_event_id,
            attempts[1].conflict_health_event_id
        );
        assert!(attempts.iter().all(|attempt| {
            attempt.start_deadline > attempt.started_at
                && attempt.start_deadline.duration_since(attempt.started_at)
                    <= TRANSACTION_START_SLICE
        }));
        assert_eq!(writer.max_in_flight(), 1);
    }

    #[tokio::test]
    async fn foreground_timeout_preserves_the_single_definitive_attempt() {
        let release = Arc::new(Notify::new());
        let writer = Arc::new(FakeWriter::new([FakeAction::Wait(
            release.clone(),
            Ok(DecisionAuditAck::Applied),
        )]));
        let service = service(writer.clone());

        assert_eq!(
            service
                .try_admit()
                .unwrap()
                .submit_until(audit(3), Instant::now() + Duration::from_millis(25))
                .await
                .unwrap(),
            RecommendationFirstAckV1::PendingRetry
        );
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert_eq!(writer.attempts().len(), 1);
        assert_eq!(writer.max_in_flight(), 1);
        assert_eq!(
            service.health(),
            RecommendationDeliveryHealthV1::DegradedPending { pending: 1 }
        );
        release.notify_waiters();
        wait_for_health(&service, RecommendationDeliveryHealthV1::Healthy).await;
        assert_eq!(writer.attempts().len(), 1);
    }

    #[tokio::test]
    async fn preadmitted_audits_can_be_pending_concurrently_and_recover_together() {
        let release = Arc::new(Notify::new());
        let writer = Arc::new(FakeWriter::new([
            FakeAction::Wait(release.clone(), Ok(DecisionAuditAck::Applied)),
            FakeAction::Wait(release.clone(), Ok(DecisionAuditAck::Applied)),
        ]));
        let service = service(writer.clone());
        let first_permit = service.try_admit().unwrap();
        let second_permit = service.try_admit().unwrap();

        let first = tokio::spawn(async move {
            first_permit
                .submit_until(audit(40), Instant::now() + Duration::from_secs(2))
                .await
        });
        let second = tokio::spawn(async move {
            second_permit
                .submit_until(audit(41), Instant::now() + Duration::from_secs(2))
                .await
        });
        let deadline = tokio::time::Instant::now() + Duration::from_secs(1);
        while writer.attempts().len() != 2 {
            assert!(tokio::time::Instant::now() < deadline);
            tokio::task::yield_now().await;
        }
        assert_eq!(
            service.health(),
            RecommendationDeliveryHealthV1::DegradedPending { pending: 2 }
        );
        assert!(matches!(
            service.try_admit(),
            Err(RecommendationAdmissionErrorV1::DegradedPending)
        ));

        release.notify_waiters();
        assert_eq!(
            first.await.unwrap().unwrap(),
            RecommendationFirstAckV1::Applied
        );
        assert_eq!(
            second.await.unwrap().unwrap(),
            RecommendationFirstAckV1::Applied
        );
        wait_for_health(&service, RecommendationDeliveryHealthV1::Healthy).await;
        assert_eq!(writer.attempts().len(), 2);
        drop(service.try_admit().unwrap());
    }

    #[tokio::test]
    async fn expired_foreground_deadline_still_retains_the_valid_audit() {
        let writer = Arc::new(FakeWriter::new([FakeAction::Immediate(Ok(
            DecisionAuditAck::Applied,
        ))]));
        let service = service(writer.clone());

        assert_eq!(
            service
                .try_admit()
                .unwrap()
                .submit_until(audit(30), Instant::now() - Duration::from_millis(1))
                .await
                .unwrap(),
            RecommendationFirstAckV1::PendingRetry
        );
        wait_for_health(&service, RecommendationDeliveryHealthV1::Healthy).await;
        assert_eq!(writer.attempts().len(), 1);
    }

    #[tokio::test]
    async fn invalid_frozen_audit_latches_permanent_fault_without_retention() {
        let writer = Arc::new(FakeWriter::new([]));
        let service = service(writer.clone());
        let mut invalid = audit(31);
        Arc::get_mut(&mut invalid).unwrap().command_size_bytes += 1;

        assert_eq!(
            service
                .try_admit()
                .unwrap()
                .submit_until(invalid, Instant::now() + Duration::from_secs(1))
                .await,
            Err(RecommendationDeliveryErrorV1::InvalidAudit)
        );
        assert_eq!(
            service.health(),
            RecommendationDeliveryHealthV1::PermanentFault { retained: 0 }
        );
        assert!(writer.attempts().is_empty());
    }

    #[tokio::test]
    async fn retention_pressure_is_signaled_before_exact_retry() {
        let writer = Arc::new(FakeWriter::new([
            FakeAction::Immediate(Ok(DecisionAuditAck::RetentionRequired)),
            FakeAction::Immediate(Ok(DecisionAuditAck::RetentionRequired)),
            FakeAction::Immediate(Ok(DecisionAuditAck::AlreadyApplied)),
        ]));
        let service = service(writer.clone());
        let mut pressure = service.subscribe_retention_pressure();

        assert_eq!(
            service
                .try_admit()
                .unwrap()
                .submit_until(audit(4), Instant::now() + Duration::from_secs(1))
                .await
                .unwrap(),
            RecommendationFirstAckV1::PendingRetry
        );
        tokio::time::timeout(Duration::from_secs(1), pressure.changed())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(*pressure.borrow(), 1);
        wait_for_health(&service, RecommendationDeliveryHealthV1::Healthy).await;
        assert_eq!(*pressure.borrow(), 1);
        assert_eq!(writer.attempts().len(), 3);
    }

    #[tokio::test]
    async fn source_retiring_discards_without_latching_degraded_health() {
        let writer = Arc::new(FakeWriter::new([FakeAction::Immediate(Ok(
            DecisionAuditAck::SourceRetiring,
        ))]));
        let service = service(writer);

        assert_eq!(
            service
                .try_admit()
                .unwrap()
                .submit_until(audit(5), Instant::now() + Duration::from_secs(1))
                .await
                .unwrap(),
            RecommendationFirstAckV1::DroppedStale(RecommendationStaleReasonV1::SourceRetiring)
        );
        assert_eq!(service.health(), RecommendationDeliveryHealthV1::Healthy);
        drop(service.try_admit().unwrap());
    }

    #[tokio::test]
    async fn conflict_and_writer_panic_latch_permanent_fault() {
        for (suffix, action) in [
            (6, FakeAction::Immediate(Ok(DecisionAuditAck::Conflict))),
            (7, FakeAction::Panic),
            (8, FakeAction::DropPanic),
            (
                9,
                FakeAction::Immediate(Err(WriterFailure::new(WriterFailureClass::Panicked))),
            ),
        ] {
            let writer = Arc::new(FakeWriter::new([action]));
            let service = service(writer);
            assert_eq!(
                service
                    .try_admit()
                    .unwrap()
                    .submit_until(audit(suffix), Instant::now() + Duration::from_secs(1))
                    .await,
                Err(RecommendationDeliveryErrorV1::PermanentFault)
            );
            assert_eq!(
                service.health(),
                RecommendationDeliveryHealthV1::PermanentFault { retained: 1 }
            );
            assert!(matches!(
                service.try_admit(),
                Err(RecommendationAdmissionErrorV1::PermanentFault)
            ));
            assert_eq!(
                service
                    .drain_until(Instant::now() + Duration::from_millis(20))
                    .await,
                Err(RecommendationDrainErrorV1::PermanentFault)
            );
            service.abort();
        }
    }

    #[tokio::test]
    async fn cancellation_keeps_delivery_owned_and_abort_stops_later_attempts() {
        let release = Arc::new(Notify::new());
        let writer = Arc::new(FakeWriter::new([FakeAction::Wait(
            release.clone(),
            Err(WriterFailure::new(WriterFailureClass::Deadline)),
        )]));
        let service = service(writer.clone());
        let permit = service.try_admit().unwrap();
        let foreground = tokio::spawn(async move {
            permit
                .submit_until(audit(8), Instant::now() + Duration::from_secs(1))
                .await
        });
        while writer.attempts().is_empty() {
            tokio::task::yield_now().await;
        }
        foreground.abort();
        assert_eq!(
            service.health(),
            RecommendationDeliveryHealthV1::DegradedPending { pending: 1 }
        );
        release.notify_waiters();
        tokio::time::sleep(Duration::from_millis(2)).await;
        service.abort();
        let attempts = writer.attempts().len();
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert_eq!(writer.attempts().len(), attempts);
        assert_eq!(service.health(), RecommendationDeliveryHealthV1::Closed);
    }
}
