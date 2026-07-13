// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded foreground Active outcome ownership and durable terminal delivery.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use nemo_relay::api::event::ScopeCategory;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::canonical_json::canonical_json;
use crate::coordinator::LossSnapshot;
use crate::health::RouterHealth;
use crate::ledger::repository::active::{
    ActiveProtectedSignal, ActiveRepresentativeStatus, ActiveRootClosure, ActiveRootTerminal,
    ActiveRootTerminalAck, ActiveSignalBatch, ActiveSignalBatchAck, ActiveSignalDisposition,
};
use crate::ledger::repository::active_learning::{
    ActiveNeighborhoodInvalidation, ActiveNeighborhoodKey,
};
use crate::ledger::writer::LedgerWriterClient;
use crate::outcome::{
    CompiledOutcomePolicyV1, OutcomeCollectionFaultV1, OutcomeEventContextV1,
    OutcomeSignalAccumulatorV1, PinnedOwnerRelationV1, ProtectedOutcomeSignalV1,
    RepresentativeResultV1, RepresentativeTerminalV1, RootOutcomeLabelV1,
};
use crate::trajectory::{CapturedTrajectoryEvent, OversizedTrajectoryEvent};

const ACTIVE_OUTCOME_WRITE_TIMEOUT: Duration = Duration::from_secs(5);
const HEALTH_ACTIVE_OUTCOME_FAILURE: &str = "router.active.outcome_failure";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveOutcomeRuntimeErrorV2 {
    Closed,
    Capacity,
    Conflict,
    Storage,
    Deadline,
}

pub(crate) struct ActiveOutcomeStageV2 {
    pub(crate) active_root_window_id: Uuid,
    pub(crate) raw_root_uuid: Uuid,
    pub(crate) pinned_owner_uuid: Uuid,
    pub(crate) policy: Arc<CompiledOutcomePolicyV1>,
    pub(crate) opened_after_ingest_seq: u64,
    pub(crate) opening_loss: LossSnapshot,
    pub(crate) attribution_deadline: Instant,
    pub(crate) attribution_deadline_unix_ms: u64,
    pub(crate) invalidation: Option<ActiveOutcomeInvalidationAuthorityV2>,
}

pub(crate) struct ActiveOutcomeInvalidationAuthorityV2 {
    pub(crate) key: ActiveNeighborhoodKey,
    pub(crate) active_dispatch_id: Uuid,
    pub(crate) cooloff_duration_seconds: u32,
}

pub(crate) struct ActiveOutcomeObservationV2 {
    pub(crate) root_uuid: Uuid,
    pub(crate) event: Arc<CapturedTrajectoryEvent>,
}

pub(crate) struct ActiveOutcomeOversizedV2 {
    pub(crate) root_uuid: Uuid,
    pub(crate) event: OversizedTrajectoryEvent,
}

pub(crate) struct ActiveRepresentativeObservationV2 {
    pub(crate) active_root_window_id: Uuid,
    pub(crate) raw_root_uuid: Uuid,
    pub(crate) result: RepresentativeResultV1,
    pub(crate) stable_error_class: Option<String>,
    pub(crate) ingest_seq: u64,
    pub(crate) observed_at_unix_ms: u64,
    pub(crate) terminal_loss: LossSnapshot,
}

pub(crate) struct ActiveDeadlineObservationV2 {
    pub(crate) active_root_window_id: Uuid,
    pub(crate) raw_root_uuid: Uuid,
    pub(crate) interval_end_unix_ms: u64,
    pub(crate) terminal_loss: LossSnapshot,
    pub(crate) barrier_succeeded: bool,
}

pub(crate) enum ActiveOutcomeCommandV2 {
    Stage {
        stage: ActiveOutcomeStageV2,
        reply: oneshot::Sender<Result<(), ActiveOutcomeRuntimeErrorV2>>,
    },
    Commit {
        active_root_window_id: Uuid,
        raw_root_uuid: Uuid,
        reply: oneshot::Sender<Result<(), ActiveOutcomeRuntimeErrorV2>>,
    },
    Discard {
        active_root_window_id: Uuid,
        raw_root_uuid: Uuid,
    },
    Observe(ActiveOutcomeObservationV2),
    Oversized(ActiveOutcomeOversizedV2),
    Representative {
        observation: ActiveRepresentativeObservationV2,
        reply: oneshot::Sender<Result<(), ActiveOutcomeRuntimeErrorV2>>,
    },
    Deadline(ActiveDeadlineObservationV2),
    Shutdown {
        observed_at_unix_ms: u64,
        reply: oneshot::Sender<Result<(), ActiveOutcomeRuntimeErrorV2>>,
    },
}

#[derive(Clone)]
pub(crate) struct ActiveOutcomeClientV2 {
    sender: mpsc::Sender<ActiveOutcomeCommandV2>,
}

impl ActiveOutcomeClientV2 {
    pub(crate) fn try_observe(
        &self,
        observation: ActiveOutcomeObservationV2,
    ) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
        self.sender
            .try_send(ActiveOutcomeCommandV2::Observe(observation))
            .map_err(map_try_send)
    }

    pub(crate) fn try_observe_oversized(
        &self,
        observation: ActiveOutcomeOversizedV2,
    ) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
        self.sender
            .try_send(ActiveOutcomeCommandV2::Oversized(observation))
            .map_err(map_try_send)
    }

    pub(crate) fn try_stage(
        &self,
        stage: ActiveOutcomeStageV2,
        reply: oneshot::Sender<Result<(), ActiveOutcomeRuntimeErrorV2>>,
    ) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
        self.sender
            .try_send(ActiveOutcomeCommandV2::Stage { stage, reply })
            .map_err(map_try_send)
    }

    pub(crate) async fn commit_until(
        &self,
        active_root_window_id: Uuid,
        raw_root_uuid: Uuid,
        deadline: Instant,
    ) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
        let (reply, receiver) = oneshot::channel();
        send_until(
            &self.sender,
            ActiveOutcomeCommandV2::Commit {
                active_root_window_id,
                raw_root_uuid,
                reply,
            },
            deadline,
        )
        .await?;
        receive_until(receiver, deadline).await?
    }

    pub(crate) fn discard(&self, active_root_window_id: Uuid, raw_root_uuid: Uuid) {
        let _ = self.sender.try_send(ActiveOutcomeCommandV2::Discard {
            active_root_window_id,
            raw_root_uuid,
        });
    }

    pub(crate) fn try_representative(
        &self,
        observation: ActiveRepresentativeObservationV2,
        reply: oneshot::Sender<Result<(), ActiveOutcomeRuntimeErrorV2>>,
    ) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
        self.sender
            .try_send(ActiveOutcomeCommandV2::Representative { observation, reply })
            .map_err(map_try_send)
    }

    pub(crate) fn try_deadline(
        &self,
        observation: ActiveDeadlineObservationV2,
    ) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
        self.sender
            .try_send(ActiveOutcomeCommandV2::Deadline(observation))
            .map_err(map_try_send)
    }

    pub(crate) fn try_shutdown(
        &self,
        observed_at_unix_ms: u64,
        reply: oneshot::Sender<Result<(), ActiveOutcomeRuntimeErrorV2>>,
    ) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
        self.sender
            .try_send(ActiveOutcomeCommandV2::Shutdown {
                observed_at_unix_ms,
                reply,
            })
            .map_err(map_try_send)
    }
}

pub(crate) fn start_active_outcome_runtime_v2(
    writer: LedgerWriterClient,
    command_capacity: usize,
    health: Arc<RouterHealth>,
) -> Result<(ActiveOutcomeClientV2, JoinHandle<()>), ActiveOutcomeRuntimeErrorV2> {
    if command_capacity == 0 {
        return Err(ActiveOutcomeRuntimeErrorV2::Capacity);
    }
    let (sender, receiver) = mpsc::channel(command_capacity);
    let client = ActiveOutcomeClientV2 { sender };
    let task = tokio::spawn(run_active_outcome_actor(writer, health, receiver));
    Ok((client, task))
}

struct ActiveOutcomeWindowV2 {
    active_root_window_id: Uuid,
    pinned_owner_uuid: Uuid,
    accumulator: OutcomeSignalAccumulatorV1,
    persisted_signal_ids: BTreeSet<String>,
    pending_signals: BTreeMap<String, ActiveProtectedSignal>,
    opening_loss: LossSnapshot,
    opened_after_ingest_seq: u64,
    committed: bool,
    owner_ended: bool,
    representative: Option<(RepresentativeResultV1, Option<String>)>,
    terminal: Option<ActiveRootTerminal>,
    invalidation: Option<ActiveOutcomeInvalidationAuthorityV2>,
}

async fn run_active_outcome_actor(
    writer: LedgerWriterClient,
    health: Arc<RouterHealth>,
    mut receiver: mpsc::Receiver<ActiveOutcomeCommandV2>,
) {
    let mut windows = BTreeMap::<Uuid, ActiveOutcomeWindowV2>::new();
    while let Some(command) = receiver.recv().await {
        if handle_command(&writer, &health, &mut windows, command).await {
            break;
        }
    }
    if !windows.is_empty() {
        health.accept(HEALTH_ACTIVE_OUTCOME_FAILURE);
    }
}

async fn handle_command(
    writer: &LedgerWriterClient,
    health: &RouterHealth,
    windows: &mut BTreeMap<Uuid, ActiveOutcomeWindowV2>,
    command: ActiveOutcomeCommandV2,
) -> bool {
    match command {
        ActiveOutcomeCommandV2::Stage { stage, reply } => {
            let result = stage_window(windows, stage);
            let _ = reply.send(result);
        }
        ActiveOutcomeCommandV2::Commit {
            active_root_window_id,
            raw_root_uuid,
            reply,
        } => {
            let result = commit_window(writer, windows, raw_root_uuid, active_root_window_id).await;
            let _ = reply.send(result);
        }
        ActiveOutcomeCommandV2::Discard {
            active_root_window_id,
            raw_root_uuid,
        } => {
            if windows.get(&raw_root_uuid).is_some_and(|window| {
                !window.committed && window.active_root_window_id == active_root_window_id
            }) {
                windows.remove(&raw_root_uuid);
            }
        }
        ActiveOutcomeCommandV2::Observe(observation) => {
            if observe_event(writer, windows, observation).await.is_err() {
                health.accept(HEALTH_ACTIVE_OUTCOME_FAILURE);
            }
        }
        ActiveOutcomeCommandV2::Oversized(observation) => {
            if observe_oversized(writer, windows, observation)
                .await
                .is_err()
            {
                health.accept(HEALTH_ACTIVE_OUTCOME_FAILURE);
            }
        }
        ActiveOutcomeCommandV2::Representative { observation, reply } => {
            let result = observe_representative(writer, windows, observation).await;
            let _ = reply.send(result);
        }
        ActiveOutcomeCommandV2::Deadline(observation) => {
            if observe_deadline(writer, windows, observation)
                .await
                .is_err()
            {
                health.accept(HEALTH_ACTIVE_OUTCOME_FAILURE);
            }
        }
        ActiveOutcomeCommandV2::Shutdown {
            observed_at_unix_ms,
            reply,
        } => {
            let result = shutdown_windows(writer, windows, observed_at_unix_ms).await;
            let _ = reply.send(result);
            return true;
        }
    }
    false
}

fn stage_window(
    windows: &mut BTreeMap<Uuid, ActiveOutcomeWindowV2>,
    stage: ActiveOutcomeStageV2,
) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
    if windows.len() >= crate::outcome::OUTCOME_OPEN_ROOT_WINDOWS_MAX {
        return Err(ActiveOutcomeRuntimeErrorV2::Capacity);
    }
    if stage.attribution_deadline <= Instant::now()
        || windows.contains_key(&stage.raw_root_uuid)
        || windows
            .values()
            .any(|window| window.active_root_window_id == stage.active_root_window_id)
    {
        return Err(ActiveOutcomeRuntimeErrorV2::Conflict);
    }
    windows.insert(
        stage.raw_root_uuid,
        ActiveOutcomeWindowV2 {
            active_root_window_id: stage.active_root_window_id,
            pinned_owner_uuid: stage.pinned_owner_uuid,
            accumulator: OutcomeSignalAccumulatorV1::new(
                stage.policy,
                stage.opened_after_ingest_seq,
            ),
            persisted_signal_ids: BTreeSet::new(),
            pending_signals: BTreeMap::new(),
            opening_loss: stage.opening_loss,
            opened_after_ingest_seq: stage.opened_after_ingest_seq,
            committed: false,
            owner_ended: false,
            representative: None,
            terminal: None,
            invalidation: stage.invalidation,
        },
    );
    Ok(())
}

async fn commit_window(
    writer: &LedgerWriterClient,
    windows: &mut BTreeMap<Uuid, ActiveOutcomeWindowV2>,
    raw_root_uuid: Uuid,
    active_root_window_id: Uuid,
) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
    let window = windows
        .get_mut(&raw_root_uuid)
        .ok_or(ActiveOutcomeRuntimeErrorV2::Conflict)?;
    if window.active_root_window_id != active_root_window_id {
        return Err(ActiveOutcomeRuntimeErrorV2::Conflict);
    }
    window.committed = true;
    persist_pending_signals(writer, window).await?;
    maybe_close_owner_end(writer, windows, raw_root_uuid).await
}

async fn observe_event(
    writer: &LedgerWriterClient,
    windows: &mut BTreeMap<Uuid, ActiveOutcomeWindowV2>,
    observation: ActiveOutcomeObservationV2,
) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
    let Some(window) = windows.get_mut(&observation.root_uuid) else {
        return Ok(());
    };
    let exact_owner_end = observation.event.event_uuid == window.pinned_owner_uuid
        && observation.event.scope_phase == Some(ScopeCategory::End);
    // The coordinator emits only observations resolved under this raw root.
    // Nested owners remain inside the pinned external owner subtree, and
    // internal Router roles were removed before this command was emitted.
    let relation = PinnedOwnerRelationV1::ExternalPinnedOwnerSubtree;
    let before = window
        .accumulator
        .signals()
        .iter()
        .map(|signal| signal.signal_sha256().to_string())
        .collect::<BTreeSet<_>>();
    window.accumulator.observe_event(
        &observation.event,
        &OutcomeEventContextV1 {
            owner_relation: relation,
            exact_pinned_owner_end: exact_owner_end,
        },
    );
    capture_new_signals(window, &before, event_signal_context(&observation.event));
    if exact_owner_end {
        window.owner_ended = true;
    }
    if window.committed {
        persist_pending_signals(writer, window).await?;
        maybe_close_owner_end(writer, windows, observation.root_uuid).await?;
    }
    Ok(())
}

async fn observe_oversized(
    writer: &LedgerWriterClient,
    windows: &mut BTreeMap<Uuid, ActiveOutcomeWindowV2>,
    observation: ActiveOutcomeOversizedV2,
) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
    let Some(window) = windows.get_mut(&observation.root_uuid) else {
        return Ok(());
    };
    if observation.event.ingest_seq > window.opened_after_ingest_seq {
        window
            .accumulator
            .mark_fault(OutcomeCollectionFaultV1::OversizedEvent);
    }
    if observation.event.event_uuid == window.pinned_owner_uuid
        && observation.event.scope_phase == Some(ScopeCategory::End)
    {
        window.owner_ended = true;
    }
    if window.committed {
        maybe_close_owner_end(writer, windows, observation.root_uuid).await?;
    }
    Ok(())
}

async fn observe_representative(
    writer: &LedgerWriterClient,
    windows: &mut BTreeMap<Uuid, ActiveOutcomeWindowV2>,
    observation: ActiveRepresentativeObservationV2,
) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
    let window = windows
        .get_mut(&observation.raw_root_uuid)
        .ok_or(ActiveOutcomeRuntimeErrorV2::Conflict)?;
    if !window.committed || window.active_root_window_id != observation.active_root_window_id {
        return Err(ActiveOutcomeRuntimeErrorV2::Conflict);
    }
    if observation.terminal_loss.classification_loss_epoch
        != window.opening_loss.classification_loss_epoch
        || observation.terminal_loss.highest_dropped_ingest_seq
            > window.opening_loss.highest_dropped_ingest_seq
    {
        window
            .accumulator
            .mark_fault(OutcomeCollectionFaultV1::EventLoss);
    }
    let before = window
        .accumulator
        .signals()
        .iter()
        .map(|signal| signal.signal_sha256().to_string())
        .collect::<BTreeSet<_>>();
    window
        .accumulator
        .record_representative_result(observation.result, observation.ingest_seq);
    capture_new_signals(
        window,
        &before,
        SignalContextV2 {
            event_kind: "representative".to_string(),
            scope_phase: "terminal".to_string(),
            category: "llm".to_string(),
            name: representative_name(observation.result.terminal).to_string(),
            observed_at_unix_ms: observation.observed_at_unix_ms,
        },
    );
    window.representative = Some((observation.result, observation.stable_error_class));
    persist_pending_signals(writer, window).await?;
    maybe_close_owner_end(writer, windows, observation.raw_root_uuid).await
}

async fn maybe_close_owner_end(
    writer: &LedgerWriterClient,
    windows: &mut BTreeMap<Uuid, ActiveOutcomeWindowV2>,
    raw_root_uuid: Uuid,
) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
    let should_close = windows.get(&raw_root_uuid).is_some_and(|window| {
        window.committed && window.owner_ended && window.representative.is_some()
    });
    if should_close {
        close_window(
            writer,
            windows,
            raw_root_uuid,
            ActiveRootClosure::OwnerEnd,
            now_unix_ms(),
        )
        .await
    } else {
        Ok(())
    }
}

async fn observe_deadline(
    writer: &LedgerWriterClient,
    windows: &mut BTreeMap<Uuid, ActiveOutcomeWindowV2>,
    observation: ActiveDeadlineObservationV2,
) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
    let Some(window) = windows.get_mut(&observation.raw_root_uuid) else {
        return Ok(());
    };
    if window.active_root_window_id != observation.active_root_window_id {
        return Err(ActiveOutcomeRuntimeErrorV2::Conflict);
    }
    if !window.committed {
        windows.remove(&observation.raw_root_uuid);
        return Ok(());
    }
    if !observation.barrier_succeeded
        || observation.terminal_loss.classification_loss_epoch
            != window.opening_loss.classification_loss_epoch
        || observation.terminal_loss.highest_dropped_ingest_seq
            > window.opening_loss.highest_dropped_ingest_seq
    {
        window
            .accumulator
            .mark_fault(OutcomeCollectionFaultV1::EventLoss);
    }
    close_window(
        writer,
        windows,
        observation.raw_root_uuid,
        ActiveRootClosure::Deadline,
        observation.interval_end_unix_ms,
    )
    .await
}

async fn shutdown_windows(
    writer: &LedgerWriterClient,
    windows: &mut BTreeMap<Uuid, ActiveOutcomeWindowV2>,
    observed_at_unix_ms: u64,
) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
    let roots = windows.keys().copied().collect::<Vec<_>>();
    for root in roots {
        if windows.get(&root).is_some_and(|window| window.committed) {
            close_window(
                writer,
                windows,
                root,
                ActiveRootClosure::ShutdownOrphaned,
                observed_at_unix_ms,
            )
            .await?;
        } else {
            windows.remove(&root);
        }
    }
    Ok(())
}

async fn close_window(
    writer: &LedgerWriterClient,
    windows: &mut BTreeMap<Uuid, ActiveOutcomeWindowV2>,
    raw_root_uuid: Uuid,
    closure: ActiveRootClosure,
    interval_end_unix_ms: u64,
) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
    let window = windows
        .get_mut(&raw_root_uuid)
        .ok_or(ActiveOutcomeRuntimeErrorV2::Conflict)?;
    persist_pending_signals(writer, window).await?;
    let terminal = window.terminal.get_or_insert_with(|| {
        let representative = window.representative.as_ref();
        let outcome_id = Uuid::now_v7();
        let representative_status = representative.and_then(|(result, _)| match result.terminal {
            RepresentativeTerminalV1::Completed => Some(ActiveRepresentativeStatus::Completed),
            RepresentativeTerminalV1::Error => Some(ActiveRepresentativeStatus::Error),
            _ => None,
        });
        ActiveRootTerminal {
            active_root_window_state_event_id: Uuid::now_v7(),
            outcome_id,
            active_root_window_id: window.active_root_window_id,
            closure: if window.accumulator.label_complete() {
                closure
            } else {
                ActiveRootClosure::ProjectionFault
            },
            label_complete: window.accumulator.label_complete(),
            representative_status,
            stable_terminal_error_class: representative.and_then(|(_, error)| error.clone()),
            interval_end_unix_ms,
            neighborhood_invalidation: (window.accumulator.label_complete()
                && window.accumulator.prefix_label() == RootOutcomeLabelV1::Failure)
                .then(|| {
                    window
                        .invalidation
                        .as_ref()
                        .map(|authority| ActiveNeighborhoodInvalidation {
                            invalidated_state_event_id: Uuid::now_v7(),
                            cooloff_state_event_id: Uuid::now_v7(),
                            key: authority.key.clone(),
                            cause_active_dispatch_id: Some(authority.active_dispatch_id),
                            cause_outcome_id: Some(outcome_id),
                            stable_reason: "candidate_outcome_failure".to_string(),
                            cooloff_duration_seconds: authority.cooloff_duration_seconds,
                        })
                })
                .flatten(),
        }
    });
    let deadline = Instant::now()
        .checked_add(ACTIVE_OUTCOME_WRITE_TIMEOUT)
        .ok_or(ActiveOutcomeRuntimeErrorV2::Deadline)?;
    match writer
        .terminalize_active_root_until(terminal.clone(), deadline)
        .await
        .map_err(|_| ActiveOutcomeRuntimeErrorV2::Storage)?
    {
        ActiveRootTerminalAck::Applied | ActiveRootTerminalAck::AlreadyApplied => {
            windows.remove(&raw_root_uuid);
            Ok(())
        }
        ActiveRootTerminalAck::AuthorityChanged | ActiveRootTerminalAck::TransactionNotStarted => {
            Err(ActiveOutcomeRuntimeErrorV2::Storage)
        }
    }
}

async fn persist_pending_signals(
    writer: &LedgerWriterClient,
    window: &mut ActiveOutcomeWindowV2,
) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
    let signals = window
        .pending_signals
        .iter()
        .filter_map(|(identity, signal)| {
            (!window.persisted_signal_ids.contains(identity)).then_some(signal.clone())
        })
        .collect::<Vec<_>>();
    if signals.is_empty() {
        return Ok(());
    }
    let identities = signals
        .iter()
        .map(|signal| signal.signal_identity_hash.clone())
        .collect::<Vec<_>>();
    let deadline = Instant::now()
        .checked_add(ACTIVE_OUTCOME_WRITE_TIMEOUT)
        .ok_or(ActiveOutcomeRuntimeErrorV2::Deadline)?;
    match writer
        .append_active_signals_until(
            ActiveSignalBatch {
                active_root_window_id: window.active_root_window_id,
                signals,
            },
            deadline,
        )
        .await
        .map_err(|_| ActiveOutcomeRuntimeErrorV2::Storage)?
    {
        ActiveSignalBatchAck::Applied { .. } | ActiveSignalBatchAck::AlreadyApplied { .. } => {
            window.persisted_signal_ids.extend(identities);
            Ok(())
        }
        ActiveSignalBatchAck::CapacityExceeded => {
            window
                .accumulator
                .mark_fault(OutcomeCollectionFaultV1::SignalBytesExceeded);
            Err(ActiveOutcomeRuntimeErrorV2::Capacity)
        }
        ActiveSignalBatchAck::WindowClosed
        | ActiveSignalBatchAck::AuthorityChanged
        | ActiveSignalBatchAck::TransactionNotStarted => Err(ActiveOutcomeRuntimeErrorV2::Storage),
    }
}

struct SignalContextV2 {
    event_kind: String,
    scope_phase: String,
    category: String,
    name: String,
    observed_at_unix_ms: u64,
}

fn event_signal_context(event: &CapturedTrajectoryEvent) -> SignalContextV2 {
    SignalContextV2 {
        event_kind: match event.kind {
            crate::trajectory::CapturedEventKind::Scope => "scope",
            crate::trajectory::CapturedEventKind::Mark => "mark",
        }
        .to_string(),
        scope_phase: match event.scope_phase {
            Some(ScopeCategory::Start) => "start",
            Some(ScopeCategory::End) => "end",
            None => "mark",
        }
        .to_string(),
        category: event
            .category
            .clone()
            .unwrap_or_else(|| "uncategorized".to_string()),
        name: event.name.clone(),
        observed_at_unix_ms: u64::try_from(event.timestamp.timestamp_millis()).unwrap_or(0),
    }
}

fn capture_new_signals(
    window: &mut ActiveOutcomeWindowV2,
    before: &BTreeSet<String>,
    context: SignalContextV2,
) {
    for signal in window.accumulator.signals() {
        if before.contains(signal.signal_sha256())
            || window.pending_signals.contains_key(signal.signal_sha256())
        {
            continue;
        }
        if let Ok(persisted) = active_signal(signal, &context) {
            window
                .pending_signals
                .insert(signal.signal_sha256().to_string(), persisted);
        } else {
            window
                .accumulator
                .mark_fault(OutcomeCollectionFaultV1::Projection);
            break;
        }
    }
}

fn active_signal(
    signal: &ProtectedOutcomeSignalV1,
    context: &SignalContextV2,
) -> Result<ActiveProtectedSignal, ActiveOutcomeRuntimeErrorV2> {
    let value = serde_json::to_value(signal).map_err(|_| ActiveOutcomeRuntimeErrorV2::Conflict)?;
    let canonical_signal_json =
        canonical_json(&value).map_err(|_| ActiveOutcomeRuntimeErrorV2::Conflict)?;
    Ok(ActiveProtectedSignal {
        active_root_signal_id: Uuid::now_v7(),
        signal_identity_hash: signal.signal_sha256().to_string(),
        event_kind: context.event_kind.clone(),
        scope_phase: context.scope_phase.clone(),
        category: context.category.clone(),
        name: context.name.clone(),
        disposition: match signal.disposition() {
            crate::config::OutcomeDisposition::Success => ActiveSignalDisposition::Success,
            crate::config::OutcomeDisposition::Failure => ActiveSignalDisposition::Failure,
            crate::config::OutcomeDisposition::Ignore => ActiveSignalDisposition::Ignored,
        },
        observed_at_unix_ms: context.observed_at_unix_ms,
        canonical_signal_json,
    })
}

fn representative_name(terminal: RepresentativeTerminalV1) -> &'static str {
    match terminal {
        RepresentativeTerminalV1::Completed => "completed",
        RepresentativeTerminalV1::Error => "error",
        RepresentativeTerminalV1::Cancelled => "cancelled",
        RepresentativeTerminalV1::Panicked => "panicked",
        RepresentativeTerminalV1::Aborted => "aborted",
        RepresentativeTerminalV1::Missing => "missing",
        RepresentativeTerminalV1::UnknownAfterCrash => "unknown_after_crash",
    }
}

async fn send_until(
    sender: &mpsc::Sender<ActiveOutcomeCommandV2>,
    command: ActiveOutcomeCommandV2,
    deadline: Instant,
) -> Result<(), ActiveOutcomeRuntimeErrorV2> {
    tokio::time::timeout_at(
        tokio::time::Instant::from_std(deadline),
        sender.send(command),
    )
    .await
    .map_err(|_| ActiveOutcomeRuntimeErrorV2::Deadline)?
    .map_err(|_| ActiveOutcomeRuntimeErrorV2::Closed)
}

async fn receive_until<T>(
    receiver: oneshot::Receiver<T>,
    deadline: Instant,
) -> Result<T, ActiveOutcomeRuntimeErrorV2> {
    tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), receiver)
        .await
        .map_err(|_| ActiveOutcomeRuntimeErrorV2::Deadline)?
        .map_err(|_| ActiveOutcomeRuntimeErrorV2::Closed)
}

fn map_try_send(
    error: mpsc::error::TrySendError<ActiveOutcomeCommandV2>,
) -> ActiveOutcomeRuntimeErrorV2 {
    match error {
        mpsc::error::TrySendError::Full(_) => ActiveOutcomeRuntimeErrorV2::Capacity,
        mpsc::error::TrySendError::Closed(_) => ActiveOutcomeRuntimeErrorV2::Closed,
    }
}

fn now_unix_ms() -> u64 {
    u64::try_from(chrono::Utc::now().timestamp_millis().max(0)).unwrap_or(0)
}
