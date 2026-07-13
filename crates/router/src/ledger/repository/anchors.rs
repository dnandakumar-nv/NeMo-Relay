// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical pending and terminal trajectory aggregates.

use std::collections::BTreeSet;

use chrono::{TimeZone, Utc};
use nemo_relay::api::runtime::LLM_REPLAY_CONTRACT_VERSION;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde::Serialize;
use serde::de::DeserializeOwned;
use serde_json::{Value as Json, json};
use uuid::Uuid;

use super::process::{append_integrity_health, originating_process_is_live};
use super::{LedgerRepository, TransactionStartGuard, map_fs_error, map_sqlite_error};
use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::fingerprint::canonical_serialize_bytes;
use crate::ledger::fs::enforce_sidecar_permissions;
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::projection::{ROUTING_CONTEXT_SCHEMA_V1, validate_request_projection};
use crate::trajectory::{
    CANDIDATE_FACT_SCHEMA_V1, CAPTURED_EVENT_SCHEMA_V1, CapturedEventKind, CapturedTrajectoryEvent,
    PENDING_TRAJECTORY_SCHEMA_V1, PendingTrajectoryWindow, PersistedTrajectoryTerminalV1,
    REPLAY_CAPABILITY_SCHEMA_V1, RESPONSE_PROJECTION_SCHEMA_V1, TERMINAL_TRAJECTORY_SCHEMA_V1,
    TRAJECTORY_SANITIZER_VERSION, TrajectoryDiagnosticV1, TrajectoryRejectionReason,
    TrajectoryTerminalStateV1, TrajectoryTrigger,
};

/// Result of a canonical pending-anchor probe.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AnchorProbe {
    Missing {
        anchor_id: Uuid,
    },
    Pending {
        anchor_id: Uuid,
        pending_hash: String,
    },
    Terminal {
        anchor_id: Uuid,
        pending_hash: String,
        terminal_hash: String,
    },
    NotScheduledQueueFull {
        anchor_id: Uuid,
        pending_hash: String,
        terminal_hash: String,
    },
    Conflict {
        anchor_id: Uuid,
    },
    OriginatingProcessNotLive {
        anchor_id: Uuid,
    },
    TransactionNotStarted {
        anchor_id: Uuid,
    },
}

/// Exhaustive acknowledgement for pending and terminal anchor writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AnchorCommandAck {
    Applied {
        anchor_id: Uuid,
        canonical_payload_hash: String,
    },
    AlreadyApplied {
        anchor_id: Uuid,
        canonical_payload_hash: String,
    },
    Conflict {
        anchor_id: Uuid,
    },
    OriginatingProcessNotLive {
        anchor_id: Uuid,
    },
    TransactionNotStarted {
        anchor_id: Uuid,
    },
}

/// Capacity-aware pending acknowledgement that preserves a racing terminal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PendingAnchorCapacityAck {
    Applied {
        anchor_id: Uuid,
        canonical_payload_hash: String,
    },
    AlreadyApplied {
        anchor_id: Uuid,
        canonical_payload_hash: String,
    },
    AlreadyTerminal {
        anchor_id: Uuid,
        pending_hash: String,
        terminal_hash: String,
    },
    NotScheduledQueueFull {
        anchor_id: Uuid,
        pending_hash: String,
        terminal_hash: String,
    },
    EvidenceCapacity {
        anchor_id: Uuid,
    },
    Conflict {
        anchor_id: Uuid,
    },
    OriginatingProcessNotLive {
        anchor_id: Uuid,
    },
    TransactionNotStarted {
        anchor_id: Uuid,
    },
}

impl PendingAnchorCapacityAck {
    pub(crate) const fn anchor_id(&self) -> Uuid {
        match self {
            Self::Applied { anchor_id, .. }
            | Self::AlreadyApplied { anchor_id, .. }
            | Self::AlreadyTerminal { anchor_id, .. }
            | Self::NotScheduledQueueFull { anchor_id, .. }
            | Self::EvidenceCapacity { anchor_id }
            | Self::Conflict { anchor_id }
            | Self::OriginatingProcessNotLive { anchor_id }
            | Self::TransactionNotStarted { anchor_id } => *anchor_id,
        }
    }
}

/// Exhaustive acknowledgement for an atomic queue-full pending decline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum NotScheduledQueueFullAck {
    Applied {
        anchor_id: Uuid,
        pending_hash: String,
        terminal_hash: String,
    },
    AlreadyApplied {
        anchor_id: Uuid,
        pending_hash: String,
        terminal_hash: String,
    },
    Conflict {
        anchor_id: Uuid,
    },
    EvidenceCapacity {
        anchor_id: Uuid,
    },
    OriginatingProcessNotLive {
        anchor_id: Uuid,
    },
    TransactionNotStarted {
        anchor_id: Uuid,
    },
}

/// Canonical, queue-safe pending aggregate with all caller-owned identities frozen.
#[derive(Clone)]
pub(crate) struct FrozenPendingAnchorV1 {
    rows: PendingRows,
    state: AnchorStateRow,
    conflict_health_event_id: Uuid,
    created_at_unix_ms: i64,
}

/// Canonical, queue-safe terminal aggregate with all caller-owned identities frozen.
#[derive(Clone)]
pub(crate) struct FrozenTerminalAnchorV1 {
    pending: PendingRows,
    events: Vec<TrajectoryEventRow>,
    window: AnchorWindowRow,
    state: AnchorStateRow,
    terminal_hash: String,
    conflict_health_event_id: Uuid,
    created_at_unix_ms: i64,
}

/// Frozen pending aggregate declined before scheduling because capacity was full.
#[derive(Clone)]
pub(crate) struct NotScheduledQueueFull {
    pending: FrozenPendingAnchorV1,
    terminal_state: AnchorStateRow,
    terminal_hash: String,
}

#[derive(Clone)]
struct PendingRows {
    anchor_id: Uuid,
    project_uuid: Uuid,
    project_id: String,
    process_instance_id: Uuid,
    pending_hash: String,
    pending_dto: PendingTrajectoryWindow,
    anchor: AnchorRow,
    result: AnchorResultRow,
}

#[derive(Clone, PartialEq, Eq)]
struct AnchorRow {
    anchor_id: String,
    project_uuid: String,
    process_instance_id: String,
    config_generation_id: String,
    policy_version_id: String,
    learning_generation_id: String,
    pool_id: String,
    anchor_call_uuid: String,
    root_uuid: String,
    owner_uuid: String,
    owner_path_json: String,
    api_family: String,
    transport_identity: String,
    anchor_model: String,
    anchor_model_revision: String,
    replay_capability_fingerprint: String,
    decoding_fingerprint: String,
    request_projection_json: String,
    routing_context_projection_json: String,
    candidate_facts_json: String,
    requested_progress: i64,
    opened_at_unix_ms: i64,
    deadline_at_unix_ms: i64,
    non_resumable: i64,
    pending_hash: String,
    canonical_payload_hash: String,
}

#[derive(Clone, PartialEq, Eq)]
struct AnchorResultRow {
    anchor_id: String,
    normalized_response_json: String,
    semantic_response_fingerprint: String,
    canonical_payload_hash: String,
}

#[derive(Clone, PartialEq, Eq)]
struct AnchorStateRow {
    anchor_state_event_id: String,
    anchor_id: String,
    process_instance_id: String,
    dead_process_instance_id: Option<String>,
    state: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Clone, PartialEq, Eq)]
struct AnchorWindowRow {
    anchor_id: String,
    requested_progress: i64,
    observed_progress: i64,
    terminal_kind: String,
    trigger: Option<String>,
    rejection_reason: Option<String>,
    is_partial: i64,
    promotion_eligible: i64,
    closed_at_unix_ms: i64,
    diagnostics_json: String,
    terminal_hash: String,
    canonical_payload_hash: String,
}

#[derive(Clone, PartialEq, Eq)]
struct TrajectoryEventRow {
    anchor_id: String,
    ingest_seq: i64,
    event_uuid: String,
    parent_uuid: Option<String>,
    kind: String,
    phase: Option<String>,
    category: Option<String>,
    call_role: Option<String>,
    name: String,
    event_time_unix_ms: i64,
    schema_id: String,
    sanitized_payload_json: String,
    canonical_size_bytes: i64,
    canonical_payload_hash: String,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InsertStatus {
    Inserted,
    Existing,
}

enum DomainWrite {
    Applied,
    AlreadyApplied,
    Conflict(ConflictContext),
}

enum NotScheduledQueueFullWrite {
    Applied,
    AlreadyApplied { terminal_hash: String },
    Conflict(ConflictContext),
}

struct ConflictContext {
    anchor_id: Option<Uuid>,
}

enum InsertResult {
    Status(InsertStatus),
    Conflict(ConflictContext),
}

enum ProbeResult {
    Missing,
    Pending,
    Terminal(String),
    NotScheduledQueueFull(String),
    Conflict(ConflictContext),
}

impl FrozenPendingAnchorV1 {
    /// Freeze a pending DTO and caller-owned state/health identities before enqueue.
    pub(crate) fn new(
        pending: &PendingTrajectoryWindow,
        pending_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_command_metadata(
            pending_state_event_id,
            conflict_health_event_id,
            created_at_unix_ms,
        )?;
        let rows = prepare_pending_rows(pending)?;
        if created_at_unix_ms < rows.anchor.opened_at_unix_ms {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let state = prepare_anchor_state(
            pending_state_event_id,
            rows.anchor_id,
            rows.process_instance_id,
            "pending",
            created_at_unix_ms,
        )?;
        Ok(Self {
            rows,
            state,
            conflict_health_event_id,
            created_at_unix_ms,
        })
    }

    pub(crate) fn anchor_id(&self) -> Uuid {
        self.rows.anchor_id
    }

    pub(crate) fn pending_hash(&self) -> &str {
        &self.rows.pending_hash
    }
}

impl NotScheduledQueueFull {
    /// Freeze the special terminal state onto an already-frozen pending identity.
    pub(crate) fn new(
        pending: FrozenPendingAnchorV1,
        terminal_state_event_id: Uuid,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_command_metadata(
            terminal_state_event_id,
            pending.conflict_health_event_id,
            created_at_unix_ms,
        )?;
        if terminal_state_event_id.to_string() == pending.state.anchor_state_event_id
            || created_at_unix_ms < pending.created_at_unix_ms
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let terminal_state = prepare_anchor_state(
            terminal_state_event_id,
            pending.rows.anchor_id,
            pending.rows.process_instance_id,
            "not_scheduled_queue_full",
            created_at_unix_ms,
        )?;
        let terminal_hash = terminal_state.canonical_payload_hash.clone();
        Ok(Self {
            pending,
            terminal_state,
            terminal_hash,
        })
    }

    pub(crate) fn anchor_id(&self) -> Uuid {
        self.pending.anchor_id()
    }

    pub(crate) fn pending_hash(&self) -> &str {
        self.pending.pending_hash()
    }

    pub(crate) fn terminal_hash(&self) -> &str {
        &self.terminal_hash
    }
}

impl FrozenTerminalAnchorV1 {
    /// Freeze a terminal DTO and caller-owned state/health identities before enqueue.
    pub(crate) fn new(
        terminal: &PersistedTrajectoryTerminalV1,
        terminal_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_command_metadata(
            terminal_state_event_id,
            conflict_health_event_id,
            created_at_unix_ms,
        )?;
        if terminal.schema != TERMINAL_TRAJECTORY_SCHEMA_V1 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let pending = prepare_pending_rows(&terminal.pending)?;
        let terminal_hash = terminal
            .payload_hash()
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let requested_progress = usize_to_i64(terminal.pending.requested_progress)?;
        let observed_progress = usize_to_i64(terminal.observed_progress)?;
        validate_exact_millisecond(terminal.closed_at)?;
        let closed_at_unix_ms = nonnegative_timestamp(terminal.closed_at.timestamp_millis())?;
        if closed_at_unix_ms < pending.anchor.opened_at_unix_ms
            || created_at_unix_ms != closed_at_unix_ms
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let (terminal_kind, trigger, rejection_reason) = match &terminal.state {
            TrajectoryTerminalStateV1::Closed { trigger } => {
                let trigger = serialized_enum(trigger)?;
                let expected_partial = trigger != "progress_reached";
                if terminal.is_partial != expected_partial
                    || (!expected_partial && observed_progress != requested_progress)
                    || (expected_partial && observed_progress >= requested_progress)
                {
                    return Err(LedgerErrorClass::IdentityInvariant.into());
                }
                ("closed".to_string(), Some(trigger), None)
            }
            TrajectoryTerminalStateV1::Rejected { reason } => {
                if !terminal.is_partial {
                    return Err(LedgerErrorClass::IdentityInvariant.into());
                }
                ("rejected".to_string(), None, Some(serialized_enum(reason)?))
            }
            TrajectoryTerminalStateV1::OrphanedNonResumable => {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        };
        if terminal.is_partial && terminal.promotion_eligible {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let mut events = Vec::with_capacity(terminal.events.len());
        let mut previous = None;
        for event in &terminal.events {
            if previous.is_some_and(|value| event.ingest_seq <= value) {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
            previous = Some(event.ingest_seq);
            events.push(prepare_trajectory_event(pending.anchor_id, event)?);
        }
        let diagnostics_json = canonical_text(&terminal.diagnostics)?;
        let window = AnchorWindowRow {
            anchor_id: pending.anchor.anchor_id.clone(),
            requested_progress,
            observed_progress,
            terminal_kind: terminal_kind.clone(),
            trigger,
            rejection_reason,
            is_partial: i64::from(terminal.is_partial),
            promotion_eligible: i64::from(terminal.promotion_eligible),
            closed_at_unix_ms,
            diagnostics_json,
            terminal_hash: terminal_hash.clone(),
            canonical_payload_hash: terminal_hash.clone(),
        };
        let state = prepare_anchor_state(
            terminal_state_event_id,
            pending.anchor_id,
            pending.process_instance_id,
            &terminal_kind,
            created_at_unix_ms,
        )?;
        Ok(Self {
            pending,
            events,
            window,
            state,
            terminal_hash,
            conflict_health_event_id,
            created_at_unix_ms,
        })
    }

    pub(crate) fn anchor_id(&self) -> Uuid {
        self.pending.anchor_id
    }

    pub(crate) fn terminal_hash(&self) -> &str {
        &self.terminal_hash
    }
}

impl LedgerRepository {
    /// Inspect one canonical pending identity before scheduler capacity is acquired.
    pub(crate) fn probe_pending_anchor(
        &mut self,
        pending: &FrozenPendingAnchorV1,
    ) -> Result<AnchorProbe, LedgerError> {
        self.probe_pending_anchor_with_start_check(pending, || Some(()))
    }

    /// Probe after atomically retaining caller-owned transaction-start authority.
    pub(crate) fn probe_pending_anchor_with_start_check<G>(
        &mut self,
        pending: &FrozenPendingAnchorV1,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<AnchorProbe, LedgerError>
    where
        G: TransactionStartGuard,
    {
        self.verify_anchor_origin(&pending.rows)?;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(AnchorProbe::TransactionNotStarted {
                anchor_id: pending.anchor_id(),
            });
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_) if !start_guard.permits_transaction() => {
                return Ok(AnchorProbe::TransactionNotStarted {
                    anchor_id: pending.anchor_id(),
                });
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(AnchorProbe::TransactionNotStarted {
                anchor_id: pending.anchor_id(),
            });
        }
        drop(start_guard);
        verify_project_id(&transaction, &pending.rows)?;
        if !originating_process_is_live(
            &transaction,
            pending.rows.project_uuid,
            pending.rows.process_instance_id,
        )? {
            return Ok(AnchorProbe::OriginatingProcessNotLive {
                anchor_id: pending.anchor_id(),
            });
        }
        if !canonical_policy_version_matches(&transaction, &pending.rows)? {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let result = probe_pending_in_transaction(&transaction, &pending.rows)?;
        if matches!(result, ProbeResult::Missing)
            && !current_learning_generation_matches(&transaction, &pending.rows)?
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let probe = match result {
            ProbeResult::Missing => AnchorProbe::Missing {
                anchor_id: pending.anchor_id(),
            },
            ProbeResult::Pending => AnchorProbe::Pending {
                anchor_id: pending.anchor_id(),
                pending_hash: pending.pending_hash().to_string(),
            },
            ProbeResult::Terminal(terminal_hash) => AnchorProbe::Terminal {
                anchor_id: pending.anchor_id(),
                pending_hash: pending.pending_hash().to_string(),
                terminal_hash,
            },
            ProbeResult::NotScheduledQueueFull(terminal_hash) => {
                AnchorProbe::NotScheduledQueueFull {
                    anchor_id: pending.anchor_id(),
                    pending_hash: pending.pending_hash().to_string(),
                    terminal_hash,
                }
            }
            ProbeResult::Conflict(context) => {
                append_integrity_health(
                    &transaction,
                    pending.conflict_health_event_id,
                    pending.rows.project_uuid,
                    pending.rows.process_instance_id,
                    context.anchor_id,
                    None,
                    pending.created_at_unix_ms,
                )?;
                AnchorProbe::Conflict {
                    anchor_id: pending.anchor_id(),
                }
            }
        };
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(probe)
    }

    /// Atomically append an immutable anchor, normalized result, and pending state.
    pub(crate) fn record_pending_anchor(
        &mut self,
        pending: &FrozenPendingAnchorV1,
    ) -> Result<AnchorCommandAck, LedgerError> {
        self.record_pending_anchor_with_start_check(pending, || Some(()))
    }

    /// Record pending after atomically retaining caller-owned start authority.
    pub(crate) fn record_pending_anchor_with_start_check<G>(
        &mut self,
        pending: &FrozenPendingAnchorV1,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<AnchorCommandAck, LedgerError>
    where
        G: TransactionStartGuard,
    {
        let acknowledgement =
            self.record_pending_anchor_with_optional_capacity(pending, None, start_check)?;
        pending_capacity_ack_into_anchor_ack(acknowledgement)
    }

    /// Record pending with a transaction-time terminal-evidence capacity check.
    pub(crate) fn record_pending_anchor_with_capacity_and_start_check<G>(
        &mut self,
        pending: &FrozenPendingAnchorV1,
        max_evidence_records: u64,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<PendingAnchorCapacityAck, LedgerError>
    where
        G: TransactionStartGuard,
    {
        self.record_pending_anchor_with_optional_capacity(
            pending,
            Some(max_evidence_records),
            start_check,
        )
    }

    fn record_pending_anchor_with_optional_capacity<G>(
        &mut self,
        pending: &FrozenPendingAnchorV1,
        max_evidence_records: Option<u64>,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<PendingAnchorCapacityAck, LedgerError>
    where
        G: TransactionStartGuard,
    {
        self.verify_anchor_origin(&pending.rows)?;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(PendingAnchorCapacityAck::TransactionNotStarted {
                anchor_id: pending.anchor_id(),
            });
        };
        let mut transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_) if !start_guard.permits_transaction() => {
                return Ok(PendingAnchorCapacityAck::TransactionNotStarted {
                    anchor_id: pending.anchor_id(),
                });
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(PendingAnchorCapacityAck::TransactionNotStarted {
                anchor_id: pending.anchor_id(),
            });
        }
        drop(start_guard);
        verify_project_id(&transaction, &pending.rows)?;
        if !originating_process_is_live(
            &transaction,
            pending.rows.project_uuid,
            pending.rows.process_instance_id,
        )? {
            return Ok(PendingAnchorCapacityAck::OriginatingProcessNotLive {
                anchor_id: pending.anchor_id(),
            });
        }
        if !canonical_policy_version_matches(&transaction, &pending.rows)? {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        if let Some(max_evidence_records) = max_evidence_records {
            let existing = probe_pending_in_transaction(&transaction, &pending.rows)?;
            let capacity_acknowledgement = match existing {
                ProbeResult::Missing => {
                    if !current_learning_generation_matches(&transaction, &pending.rows)? {
                        return Err(LedgerErrorClass::IdentityInvariant.into());
                    }
                    (terminal_anchor_count(&transaction)? >= max_evidence_records).then_some(
                        PendingAnchorCapacityAck::EvidenceCapacity {
                            anchor_id: pending.anchor_id(),
                        },
                    )
                }
                ProbeResult::Pending => Some(PendingAnchorCapacityAck::AlreadyApplied {
                    anchor_id: pending.anchor_id(),
                    canonical_payload_hash: pending.pending_hash().to_string(),
                }),
                ProbeResult::Terminal(terminal_hash) => {
                    Some(PendingAnchorCapacityAck::AlreadyTerminal {
                        anchor_id: pending.anchor_id(),
                        pending_hash: pending.pending_hash().to_string(),
                        terminal_hash,
                    })
                }
                ProbeResult::NotScheduledQueueFull(terminal_hash) => {
                    Some(PendingAnchorCapacityAck::NotScheduledQueueFull {
                        anchor_id: pending.anchor_id(),
                        pending_hash: pending.pending_hash().to_string(),
                        terminal_hash,
                    })
                }
                ProbeResult::Conflict(context) => {
                    Some(pending_capacity_ack_from_anchor_ack(finish_domain_write(
                        &transaction,
                        pending.anchor_id(),
                        pending.pending_hash(),
                        pending.conflict_health_event_id,
                        pending.rows.project_uuid,
                        pending.rows.process_instance_id,
                        pending.created_at_unix_ms,
                        DomainWrite::Conflict(context),
                    )?))
                }
            };
            if let Some(acknowledgement) = capacity_acknowledgement {
                enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
                transaction.commit().map_err(database_error)?;
                return Ok(acknowledgement);
            }
        }
        if !anchor_id_exists(&transaction, pending.anchor_id())?
            && !current_learning_generation_matches(&transaction, &pending.rows)?
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let domain_write = {
            let mut savepoint = transaction.savepoint().map_err(database_error)?;
            let result = record_pending_in_savepoint(&savepoint, pending)?;
            match result {
                DomainWrite::Conflict(_) => {
                    savepoint.rollback().map_err(database_error)?;
                    savepoint.commit().map_err(database_error)?;
                }
                DomainWrite::Applied | DomainWrite::AlreadyApplied => {
                    savepoint.commit().map_err(database_error)?;
                }
            }
            result
        };
        let acknowledgement = pending_capacity_ack_from_anchor_ack(finish_domain_write(
            &transaction,
            pending.anchor_id(),
            pending.pending_hash(),
            pending.conflict_health_event_id,
            pending.rows.project_uuid,
            pending.rows.process_instance_id,
            pending.created_at_unix_ms,
            domain_write,
        )?);
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }

    /// Atomically append a pending anchor and its queue-full terminal state.
    pub(crate) fn record_not_scheduled_queue_full(
        &mut self,
        decline: &NotScheduledQueueFull,
    ) -> Result<NotScheduledQueueFullAck, LedgerError> {
        self.record_not_scheduled_queue_full_with_start_check(decline, || Some(()))
    }

    /// Record a queue-full decline after retaining transaction-start authority.
    pub(crate) fn record_not_scheduled_queue_full_with_start_check<G>(
        &mut self,
        decline: &NotScheduledQueueFull,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<NotScheduledQueueFullAck, LedgerError>
    where
        G: TransactionStartGuard,
    {
        self.record_not_scheduled_queue_full_with_optional_capacity(decline, None, start_check)
    }

    /// Record a queue-full decline with transaction-time evidence capacity.
    pub(crate) fn record_not_scheduled_queue_full_with_capacity_and_start_check<G>(
        &mut self,
        decline: &NotScheduledQueueFull,
        max_evidence_records: u64,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<NotScheduledQueueFullAck, LedgerError>
    where
        G: TransactionStartGuard,
    {
        self.record_not_scheduled_queue_full_with_optional_capacity(
            decline,
            Some(max_evidence_records),
            start_check,
        )
    }

    fn record_not_scheduled_queue_full_with_optional_capacity<G>(
        &mut self,
        decline: &NotScheduledQueueFull,
        max_evidence_records: Option<u64>,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<NotScheduledQueueFullAck, LedgerError>
    where
        G: TransactionStartGuard,
    {
        self.verify_anchor_origin(&decline.pending.rows)?;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(NotScheduledQueueFullAck::TransactionNotStarted {
                anchor_id: decline.anchor_id(),
            });
        };
        let mut transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_) if !start_guard.permits_transaction() => {
                return Ok(NotScheduledQueueFullAck::TransactionNotStarted {
                    anchor_id: decline.anchor_id(),
                });
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(NotScheduledQueueFullAck::TransactionNotStarted {
                anchor_id: decline.anchor_id(),
            });
        }
        drop(start_guard);
        verify_project_id(&transaction, &decline.pending.rows)?;
        if !originating_process_is_live(
            &transaction,
            decline.pending.rows.project_uuid,
            decline.pending.rows.process_instance_id,
        )? {
            return Ok(NotScheduledQueueFullAck::OriginatingProcessNotLive {
                anchor_id: decline.anchor_id(),
            });
        }
        if !canonical_policy_version_matches(&transaction, &decline.pending.rows)? {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        if let Some(max_evidence_records) = max_evidence_records {
            let existing = probe_pending_in_transaction(&transaction, &decline.pending.rows)?;
            if matches!(existing, ProbeResult::Missing) {
                if !current_learning_generation_matches(&transaction, &decline.pending.rows)? {
                    return Err(LedgerErrorClass::IdentityInvariant.into());
                }
                if terminal_anchor_count(&transaction)? >= max_evidence_records {
                    enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
                    transaction.commit().map_err(database_error)?;
                    return Ok(NotScheduledQueueFullAck::EvidenceCapacity {
                        anchor_id: decline.anchor_id(),
                    });
                }
            }
        }
        if !anchor_id_exists(&transaction, decline.anchor_id())?
            && !current_learning_generation_matches(&transaction, &decline.pending.rows)?
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let domain_write = {
            let mut savepoint = transaction.savepoint().map_err(database_error)?;
            let result = record_not_scheduled_queue_full_in_savepoint(&savepoint, decline)?;
            match result {
                NotScheduledQueueFullWrite::Conflict(_) => {
                    savepoint.rollback().map_err(database_error)?;
                    savepoint.commit().map_err(database_error)?;
                }
                NotScheduledQueueFullWrite::Applied
                | NotScheduledQueueFullWrite::AlreadyApplied { .. } => {
                    savepoint.commit().map_err(database_error)?;
                }
            }
            result
        };
        let acknowledgement = finish_not_scheduled_queue_full(&transaction, decline, domain_write)?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }

    /// Atomically append ordered events, the terminal window, and terminal state.
    pub(crate) fn record_terminal_anchor(
        &mut self,
        terminal: &FrozenTerminalAnchorV1,
    ) -> Result<AnchorCommandAck, LedgerError> {
        self.record_terminal_anchor_with_start_check(terminal, || Some(()))
    }

    /// Record terminal after atomically retaining caller-owned start authority.
    pub(crate) fn record_terminal_anchor_with_start_check<G>(
        &mut self,
        terminal: &FrozenTerminalAnchorV1,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<AnchorCommandAck, LedgerError>
    where
        G: TransactionStartGuard,
    {
        self.verify_anchor_origin(&terminal.pending)?;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(AnchorCommandAck::TransactionNotStarted {
                anchor_id: terminal.anchor_id(),
            });
        };
        let mut transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_) if !start_guard.permits_transaction() => {
                return Ok(AnchorCommandAck::TransactionNotStarted {
                    anchor_id: terminal.anchor_id(),
                });
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(AnchorCommandAck::TransactionNotStarted {
                anchor_id: terminal.anchor_id(),
            });
        }
        drop(start_guard);
        verify_project_id(&transaction, &terminal.pending)?;
        if !originating_process_is_live(
            &transaction,
            terminal.pending.project_uuid,
            terminal.pending.process_instance_id,
        )? {
            return Ok(AnchorCommandAck::OriginatingProcessNotLive {
                anchor_id: terminal.anchor_id(),
            });
        }
        if !canonical_policy_version_matches(&transaction, &terminal.pending)? {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let domain_write = {
            let mut savepoint = transaction.savepoint().map_err(database_error)?;
            let result = record_terminal_in_savepoint(&savepoint, terminal)?;
            match result {
                DomainWrite::Conflict(_) => {
                    savepoint.rollback().map_err(database_error)?;
                    savepoint.commit().map_err(database_error)?;
                }
                DomainWrite::Applied | DomainWrite::AlreadyApplied => {
                    savepoint.commit().map_err(database_error)?;
                }
            }
            result
        };
        let acknowledgement = finish_domain_write(
            &transaction,
            terminal.anchor_id(),
            terminal.terminal_hash(),
            terminal.conflict_health_event_id,
            terminal.pending.project_uuid,
            terminal.pending.process_instance_id,
            terminal.created_at_unix_ms,
            domain_write,
        )?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }

    fn verify_anchor_origin(&self, pending: &PendingRows) -> Result<(), LedgerError> {
        if pending.project_uuid != self.project_uuid
            || pending.process_instance_id != self.process_instance_id
            || self
                .active_policy_versions
                .get(&pending.pending_dto.pool_id)
                .is_none_or(|policy| policy != &pending.pending_dto.policy_version_id)
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(())
    }
}

fn anchor_id_exists(connection: &Connection, anchor_id: Uuid) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM anchors WHERE anchor_id = ?1)",
            params![anchor_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn terminal_anchor_count(connection: &Connection) -> Result<u64, LedgerError> {
    let count = connection
        .query_row(
            "SELECT count(*) FROM anchor_state_events WHERE state <> 'pending'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    u64::try_from(count).map_err(|_| LedgerErrorClass::IdentityInvariant.into())
}

fn canonical_policy_version_matches(
    connection: &Connection,
    pending: &PendingRows,
) -> Result<bool, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT canonical_policy_json, canonical_payload_hash
             FROM policy_versions
             WHERE project_uuid = ?1 AND pool_id = ?2 AND policy_version_id = ?3",
            params![
                pending.project_uuid.to_string(),
                pending.pending_dto.pool_id,
                pending.pending_dto.policy_version_id,
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    let Some((policy_json, payload_hash)) = stored else {
        return Ok(false);
    };
    let Ok(policy): Result<Json, _> = serde_json::from_str(&policy_json) else {
        return Ok(false);
    };
    if canonical_json(&policy).ok().as_deref() != Some(policy_json.as_str())
        || canonical_sha256(&policy).ok().as_deref()
            != Some(pending.pending_dto.policy_version_id.as_str())
        || payload_hash != pending.pending_dto.policy_version_id
    {
        return Ok(false);
    }
    let Some(pool) = policy.pointer("/pool") else {
        return Ok(false);
    };
    let Some(anchor_model) = pending
        .pending_dto
        .request_projection
        .normalized_request
        .model
        .as_deref()
    else {
        return Ok(false);
    };
    let api_family = serialized_enum(&pending.pending_dto.request_projection.family)?;
    let Some(policy_candidates) = pool.get("candidates").and_then(Json::as_array) else {
        return Ok(false);
    };
    let Some(max_candidates) = pool.get("max_candidates_per_sample").and_then(Json::as_u64) else {
        return Ok(false);
    };
    if pool.get("id").and_then(Json::as_str) != Some(pending.pending_dto.pool_id.as_str())
        || pool.get("api_family").and_then(Json::as_str) != Some(api_family.as_str())
        || pool.get("anchor_revision").and_then(Json::as_str)
            != Some(pending.pending_dto.anchor_model_revision.as_str())
        || !pool
            .get("anchor_models")
            .and_then(Json::as_array)
            .is_some_and(|models| {
                models
                    .iter()
                    .any(|model| model.as_str() == Some(anchor_model))
            })
        || u64::try_from(pending.pending_dto.candidate_facts.len()).unwrap_or(u64::MAX)
            > max_candidates
    {
        return Ok(false);
    }
    for fact in &pending.pending_dto.candidate_facts {
        let Some(candidate) = policy_candidates.iter().find(|candidate| {
            candidate.get("id").and_then(Json::as_str) == Some(fact.candidate_id.as_str())
        }) else {
            return Ok(false);
        };
        let capabilities = serde_json::to_value(&fact.capabilities)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        if candidate.get("model").and_then(Json::as_str) != Some(fact.model.as_str())
            || candidate.get("model_revision").and_then(Json::as_str)
                != Some(fact.model_revision.as_str())
            || candidate.get("cost_rank").and_then(Json::as_u64) != Some(u64::from(fact.cost_rank))
            || candidate.get("capabilities") != Some(&capabilities)
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn current_learning_generation_matches(
    connection: &Connection,
    pending: &PendingRows,
) -> Result<bool, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT g.learning_generation_id, g.project_uuid, g.pool_id,
                    g.actor, g.reason, g.created_at_unix_ms,
                    g.canonical_payload_hash, s.learning_state_event_id,
                    s.state, s.actor, s.reason, s.created_at_unix_ms,
                    s.canonical_payload_hash
             FROM learning_generation_state_events AS s
             JOIN learning_generations AS g
               ON g.project_uuid = s.project_uuid
              AND g.pool_id = s.pool_id
              AND g.learning_generation_id = s.learning_generation_id
             WHERE s.project_uuid = ?1 AND s.pool_id = ?2
             ORDER BY s.event_seq DESC LIMIT 1",
            params![
                pending.project_uuid.to_string(),
                pending.pending_dto.pool_id
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                    row.get::<_, String>(9)?,
                    row.get::<_, String>(10)?,
                    row.get::<_, i64>(11)?,
                    row.get::<_, String>(12)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((
        generation_id,
        project_uuid,
        pool_id,
        actor,
        reason,
        generation_created_at,
        generation_hash,
        state_event_id,
        state,
        state_actor,
        state_reason,
        state_created_at,
        state_hash,
    )) = stored
    else {
        return Ok(false);
    };
    let (Ok(generation_uuid), Ok(project_uuid), Ok(state_event_uuid)) = (
        parse_canonical_uuid_v7(&generation_id),
        parse_canonical_uuid_v7(&project_uuid),
        parse_canonical_uuid_v7(&state_event_id),
    ) else {
        return Ok(false);
    };
    if state != "current" {
        return Ok(false);
    }
    let expected_generation_hash = canonical_sha256(&json!({
        "learning_generation_id": generation_uuid,
        "project_uuid": project_uuid,
        "pool_id": &pool_id,
        "actor": &actor,
        "reason": &reason,
        "created_at_unix_ms": generation_created_at,
    }))
    .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let expected_state_hash = canonical_sha256(&json!({
        "learning_state_event_id": state_event_uuid,
        "project_uuid": project_uuid,
        "pool_id": &pool_id,
        "learning_generation_id": generation_uuid,
        "state": "current",
        "actor": &state_actor,
        "reason": &state_reason,
        "created_at_unix_ms": state_created_at,
    }))
    .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    Ok(
        generation_uuid == pending.pending_dto.learning_generation_id
            && project_uuid == pending.project_uuid
            && generation_hash == expected_generation_hash
            && state_hash == expected_state_hash,
    )
}

#[allow(clippy::too_many_arguments)]
fn finish_domain_write(
    connection: &Connection,
    anchor_id: Uuid,
    canonical_payload_hash: &str,
    conflict_health_event_id: Uuid,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
    domain_write: DomainWrite,
) -> Result<AnchorCommandAck, LedgerError> {
    match domain_write {
        DomainWrite::Applied => Ok(AnchorCommandAck::Applied {
            anchor_id,
            canonical_payload_hash: canonical_payload_hash.to_string(),
        }),
        DomainWrite::AlreadyApplied => Ok(AnchorCommandAck::AlreadyApplied {
            anchor_id,
            canonical_payload_hash: canonical_payload_hash.to_string(),
        }),
        DomainWrite::Conflict(context) => {
            append_integrity_health(
                connection,
                conflict_health_event_id,
                project_uuid,
                process_instance_id,
                context.anchor_id,
                None,
                created_at_unix_ms,
            )?;
            Ok(AnchorCommandAck::Conflict { anchor_id })
        }
    }
}

fn pending_capacity_ack_from_anchor_ack(
    acknowledgement: AnchorCommandAck,
) -> PendingAnchorCapacityAck {
    match acknowledgement {
        AnchorCommandAck::Applied {
            anchor_id,
            canonical_payload_hash,
        } => PendingAnchorCapacityAck::Applied {
            anchor_id,
            canonical_payload_hash,
        },
        AnchorCommandAck::AlreadyApplied {
            anchor_id,
            canonical_payload_hash,
        } => PendingAnchorCapacityAck::AlreadyApplied {
            anchor_id,
            canonical_payload_hash,
        },
        AnchorCommandAck::Conflict { anchor_id } => {
            PendingAnchorCapacityAck::Conflict { anchor_id }
        }
        AnchorCommandAck::OriginatingProcessNotLive { anchor_id } => {
            PendingAnchorCapacityAck::OriginatingProcessNotLive { anchor_id }
        }
        AnchorCommandAck::TransactionNotStarted { anchor_id } => {
            PendingAnchorCapacityAck::TransactionNotStarted { anchor_id }
        }
    }
}

fn pending_capacity_ack_into_anchor_ack(
    acknowledgement: PendingAnchorCapacityAck,
) -> Result<AnchorCommandAck, LedgerError> {
    match acknowledgement {
        PendingAnchorCapacityAck::Applied {
            anchor_id,
            canonical_payload_hash,
        } => Ok(AnchorCommandAck::Applied {
            anchor_id,
            canonical_payload_hash,
        }),
        PendingAnchorCapacityAck::AlreadyApplied {
            anchor_id,
            canonical_payload_hash,
        } => Ok(AnchorCommandAck::AlreadyApplied {
            anchor_id,
            canonical_payload_hash,
        }),
        PendingAnchorCapacityAck::Conflict { anchor_id } => {
            Ok(AnchorCommandAck::Conflict { anchor_id })
        }
        PendingAnchorCapacityAck::OriginatingProcessNotLive { anchor_id } => {
            Ok(AnchorCommandAck::OriginatingProcessNotLive { anchor_id })
        }
        PendingAnchorCapacityAck::TransactionNotStarted { anchor_id } => {
            Ok(AnchorCommandAck::TransactionNotStarted { anchor_id })
        }
        PendingAnchorCapacityAck::AlreadyTerminal { .. }
        | PendingAnchorCapacityAck::NotScheduledQueueFull { .. }
        | PendingAnchorCapacityAck::EvidenceCapacity { .. } => {
            Err(LedgerErrorClass::IdentityInvariant.into())
        }
    }
}

fn finish_not_scheduled_queue_full(
    connection: &Connection,
    decline: &NotScheduledQueueFull,
    domain_write: NotScheduledQueueFullWrite,
) -> Result<NotScheduledQueueFullAck, LedgerError> {
    match domain_write {
        NotScheduledQueueFullWrite::Applied => Ok(NotScheduledQueueFullAck::Applied {
            anchor_id: decline.anchor_id(),
            pending_hash: decline.pending_hash().to_string(),
            terminal_hash: decline.terminal_hash().to_string(),
        }),
        NotScheduledQueueFullWrite::AlreadyApplied { terminal_hash } => {
            Ok(NotScheduledQueueFullAck::AlreadyApplied {
                anchor_id: decline.anchor_id(),
                pending_hash: decline.pending_hash().to_string(),
                terminal_hash,
            })
        }
        NotScheduledQueueFullWrite::Conflict(context) => {
            append_integrity_health(
                connection,
                decline.pending.conflict_health_event_id,
                decline.pending.rows.project_uuid,
                decline.pending.rows.process_instance_id,
                context.anchor_id,
                None,
                decline.pending.created_at_unix_ms,
            )?;
            Ok(NotScheduledQueueFullAck::Conflict {
                anchor_id: decline.anchor_id(),
            })
        }
    }
}

fn record_pending_in_savepoint(
    connection: &Connection,
    pending: &FrozenPendingAnchorV1,
) -> Result<DomainWrite, LedgerError> {
    let mut statuses = Vec::with_capacity(3);
    match insert_or_verify_anchor(connection, &pending.rows.anchor)? {
        InsertResult::Status(status) => statuses.push(status),
        InsertResult::Conflict(context) => return Ok(DomainWrite::Conflict(context)),
    }
    match insert_or_verify_result(connection, &pending.rows.result)? {
        InsertResult::Status(status) => statuses.push(status),
        InsertResult::Conflict(context) => return Ok(DomainWrite::Conflict(context)),
    }
    match insert_or_verify_state(connection, &pending.state)? {
        InsertResult::Status(status) => statuses.push(status),
        InsertResult::Conflict(context) => return Ok(DomainWrite::Conflict(context)),
    }
    Ok(statuses_to_domain_write(&statuses, pending.rows.anchor_id))
}

fn record_not_scheduled_queue_full_in_savepoint(
    connection: &Connection,
    decline: &NotScheduledQueueFull,
) -> Result<NotScheduledQueueFullWrite, LedgerError> {
    let pending_existing = match probe_pending_in_transaction(connection, &decline.pending.rows)? {
        ProbeResult::Missing => match record_pending_in_savepoint(connection, &decline.pending)? {
            DomainWrite::Applied => false,
            DomainWrite::AlreadyApplied | DomainWrite::Conflict(_) => {
                return Ok(NotScheduledQueueFullWrite::Conflict(ConflictContext {
                    anchor_id: Some(decline.anchor_id()),
                }));
            }
        },
        ProbeResult::Pending => {
            let durable_pending =
                load_pending_state(connection, &decline.pending.rows.anchor.anchor_id)?;
            if durable_pending.as_ref().is_none_or(|state| {
                decline.terminal_state.created_at_unix_ms < state.created_at_unix_ms
            }) {
                return Ok(NotScheduledQueueFullWrite::Conflict(ConflictContext {
                    anchor_id: Some(decline.anchor_id()),
                }));
            }
            true
        }
        ProbeResult::NotScheduledQueueFull(_) => true,
        ProbeResult::Terminal(_) => {
            return Ok(NotScheduledQueueFullWrite::Conflict(ConflictContext {
                anchor_id: Some(decline.anchor_id()),
            }));
        }
        ProbeResult::Conflict(context) => {
            return Ok(NotScheduledQueueFullWrite::Conflict(context));
        }
    };
    let terminal_status = match insert_or_verify_state(connection, &decline.terminal_state)? {
        InsertResult::Status(status) => status,
        InsertResult::Conflict(context) => {
            return Ok(NotScheduledQueueFullWrite::Conflict(context));
        }
    };
    match (pending_existing, terminal_status) {
        (false, InsertStatus::Inserted) | (true, InsertStatus::Inserted) => {
            Ok(NotScheduledQueueFullWrite::Applied)
        }
        (true, InsertStatus::Existing) => Ok(NotScheduledQueueFullWrite::AlreadyApplied {
            terminal_hash: decline.terminal_hash().to_string(),
        }),
        (false, InsertStatus::Existing) => {
            Ok(NotScheduledQueueFullWrite::Conflict(ConflictContext {
                anchor_id: Some(decline.anchor_id()),
            }))
        }
    }
}

fn record_terminal_in_savepoint(
    connection: &Connection,
    terminal: &FrozenTerminalAnchorV1,
) -> Result<DomainWrite, LedgerError> {
    if let Some(context) = pending_prerequisite_conflict(connection, &terminal.pending)? {
        return Ok(DomainWrite::Conflict(context));
    }
    let expected_sequences = terminal
        .events
        .iter()
        .map(|event| event.ingest_seq)
        .collect::<BTreeSet<_>>();
    let mut statement = connection
        .prepare("SELECT ingest_seq FROM trajectory_events WHERE anchor_id = ?1")
        .map_err(database_error)?;
    let existing_sequences = statement
        .query_map(params![terminal.pending.anchor.anchor_id], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    if existing_sequences
        .iter()
        .any(|sequence| !expected_sequences.contains(sequence))
    {
        return Ok(DomainWrite::Conflict(ConflictContext {
            anchor_id: Some(terminal.anchor_id()),
        }));
    }

    let mut statuses = Vec::with_capacity(terminal.events.len().saturating_add(2));
    for event in &terminal.events {
        match insert_or_verify_event(connection, event)? {
            InsertResult::Status(status) => statuses.push(status),
            InsertResult::Conflict(context) => return Ok(DomainWrite::Conflict(context)),
        }
    }
    match insert_or_verify_window(connection, &terminal.window)? {
        InsertResult::Status(status) => statuses.push(status),
        InsertResult::Conflict(context) => return Ok(DomainWrite::Conflict(context)),
    }
    match insert_or_verify_state(connection, &terminal.state)? {
        InsertResult::Status(status) => statuses.push(status),
        InsertResult::Conflict(context) => return Ok(DomainWrite::Conflict(context)),
    }
    Ok(statuses_to_domain_write(
        &statuses,
        terminal.pending.anchor_id,
    ))
}

fn statuses_to_domain_write(statuses: &[InsertStatus], anchor_id: Uuid) -> DomainWrite {
    if statuses
        .iter()
        .all(|status| *status == InsertStatus::Inserted)
    {
        DomainWrite::Applied
    } else if statuses
        .iter()
        .all(|status| *status == InsertStatus::Existing)
    {
        DomainWrite::AlreadyApplied
    } else {
        DomainWrite::Conflict(ConflictContext {
            anchor_id: Some(anchor_id),
        })
    }
}

fn prepare_pending_rows(pending: &PendingTrajectoryWindow) -> Result<PendingRows, LedgerError> {
    if pending.schema != PENDING_TRAJECTORY_SCHEMA_V1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    for value in [
        pending.anchor_id,
        pending.anchor_call_uuid,
        pending.root_uuid,
        pending.owner_uuid,
        pending.process_instance_id,
        pending.project_uuid,
        pending.learning_generation_id,
    ] {
        validate_uuid_v7(value)?;
    }
    for owner in &pending.owner_path {
        validate_uuid_v7(owner.uuid)?;
        validate_bounded_text(&owner.name, 512)?;
    }
    validate_bounded_text(&pending.project_id, 128)?;
    validate_bounded_text(&pending.pool_id, 128)?;
    validate_bounded_text(&pending.anchor_model_revision, 128)?;
    validate_bounded_text(&pending.replay_capability_facts.transport_identity, 256)?;
    validate_sha256(&pending.config_generation_id)?;
    validate_sha256(&pending.policy_version_id)?;
    validate_sha256(&pending.replay_capability_facts.capability_fingerprint)?;
    validate_sha256(
        &pending
            .normalized_anchor_response
            .semantic_response_fingerprint,
    )?;
    validate_sha256(&pending.routing_context_projection.tenant_policy_hash)?;
    validate_sha256(&pending.routing_context_projection.agent_policy_hash)?;
    if validate_request_projection(&pending.request_projection).is_err()
        || pending.routing_context_projection.schema != ROUTING_CONTEXT_SCHEMA_V1
        || pending.normalized_anchor_response.schema != RESPONSE_PROJECTION_SCHEMA_V1
        || pending.normalized_anchor_response.sanitizer_version != TRAJECTORY_SANITIZER_VERSION
        || pending
            .normalized_anchor_response
            .semantic_response_fingerprint
            != canonical_hash_without_field(
                &pending.normalized_anchor_response,
                "semantic_response_fingerprint",
            )?
        || pending.replay_capability_facts.capability_fingerprint
            != canonical_hash_without_field(
                &pending.replay_capability_facts,
                "capability_fingerprint",
            )?
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    if pending.request_projection.family != pending.replay_capability_facts.api_family
        || !pending.replay_capability_facts.non_resumable
        || pending.replay_capability_facts.schema != REPLAY_CAPABILITY_SCHEMA_V1
        || pending.replay_capability_facts.contract_version != LLM_REPLAY_CONTRACT_VERSION
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let anchor_model = pending
        .request_projection
        .normalized_request
        .model
        .as_deref()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    validate_bounded_text(anchor_model, 512)?;
    if pending.candidate_facts.is_empty() {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let mut candidate_ids = BTreeSet::new();
    for candidate in &pending.candidate_facts {
        if candidate.schema != CANDIDATE_FACT_SCHEMA_V1
            || !candidate_ids.insert(candidate.candidate_id.as_str())
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        validate_bounded_text(&candidate.candidate_id, 128)?;
        validate_bounded_text(&candidate.model, 512)?;
        validate_bounded_text(&candidate.model_revision, 128)?;
        validate_sha256(&candidate.decoding_fingerprint)?;
    }
    let requested_progress = usize_to_i64(pending.requested_progress)?;
    if requested_progress == 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    validate_exact_millisecond(pending.opened_at)?;
    validate_exact_millisecond(pending.deadline_at)?;
    let opened_at_unix_ms = nonnegative_timestamp(pending.opened_at.timestamp_millis())?;
    let deadline_at_unix_ms = nonnegative_timestamp(pending.deadline_at.timestamp_millis())?;
    if deadline_at_unix_ms < opened_at_unix_ms {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let pending_hash = pending
        .payload_hash()
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let owner_path_json = canonical_text(&pending.owner_path)?;
    let request_projection_json = canonical_text(&pending.request_projection)?;
    let routing_context_projection_json = canonical_text(&pending.routing_context_projection)?;
    let candidate_facts_json = canonical_text(&pending.candidate_facts)?;
    let normalized_response_json = canonical_text(&pending.normalized_anchor_response)?;
    let response_hash = canonical_hash(&pending.normalized_anchor_response)?;
    let decoding_fingerprint = canonical_hash(&json!({
        "schema": "nemo.relay.router.anchor-decoding-set@1",
        "candidate_facts": pending.candidate_facts,
    }))?;
    let api_family = serialized_enum(&pending.request_projection.family)?;
    let anchor_id = pending.anchor_id.to_string();
    let anchor = AnchorRow {
        anchor_id: anchor_id.clone(),
        project_uuid: pending.project_uuid.to_string(),
        process_instance_id: pending.process_instance_id.to_string(),
        config_generation_id: pending.config_generation_id.clone(),
        policy_version_id: pending.policy_version_id.clone(),
        learning_generation_id: pending.learning_generation_id.to_string(),
        pool_id: pending.pool_id.clone(),
        anchor_call_uuid: pending.anchor_call_uuid.to_string(),
        root_uuid: pending.root_uuid.to_string(),
        owner_uuid: pending.owner_uuid.to_string(),
        owner_path_json,
        api_family,
        transport_identity: pending.replay_capability_facts.transport_identity.clone(),
        anchor_model: anchor_model.to_string(),
        anchor_model_revision: pending.anchor_model_revision.clone(),
        replay_capability_fingerprint: pending
            .replay_capability_facts
            .capability_fingerprint
            .clone(),
        decoding_fingerprint,
        request_projection_json,
        routing_context_projection_json,
        candidate_facts_json,
        requested_progress,
        opened_at_unix_ms,
        deadline_at_unix_ms,
        non_resumable: i64::from(pending.replay_capability_facts.non_resumable),
        pending_hash: pending_hash.clone(),
        canonical_payload_hash: pending_hash.clone(),
    };
    let result = AnchorResultRow {
        anchor_id,
        normalized_response_json,
        semantic_response_fingerprint: pending
            .normalized_anchor_response
            .semantic_response_fingerprint
            .clone(),
        canonical_payload_hash: response_hash,
    };
    Ok(PendingRows {
        anchor_id: pending.anchor_id,
        project_uuid: pending.project_uuid,
        project_id: pending.project_id.clone(),
        process_instance_id: pending.process_instance_id,
        pending_hash,
        pending_dto: pending.clone(),
        anchor,
        result,
    })
}

fn prepare_anchor_state(
    state_event_id: Uuid,
    anchor_id: Uuid,
    process_instance_id: Uuid,
    state: &str,
    created_at_unix_ms: i64,
) -> Result<AnchorStateRow, LedgerError> {
    let hash = anchor_state_hash(
        state_event_id,
        anchor_id,
        process_instance_id,
        None,
        state,
        created_at_unix_ms,
    )?;
    Ok(AnchorStateRow {
        anchor_state_event_id: state_event_id.to_string(),
        anchor_id: anchor_id.to_string(),
        process_instance_id: process_instance_id.to_string(),
        dead_process_instance_id: None,
        state: state.to_string(),
        created_at_unix_ms,
        canonical_payload_hash: hash,
    })
}

fn prepare_trajectory_event(
    anchor_id: Uuid,
    event: &CapturedTrajectoryEvent,
) -> Result<TrajectoryEventRow, LedgerError> {
    if event.schema != CAPTURED_EVENT_SCHEMA_V1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    validate_uuid_v7(event.event_uuid)?;
    if let Some(parent_uuid) = event.parent_uuid {
        validate_uuid_v7(parent_uuid)?;
    }
    validate_bounded_text(&event.name, 512)?;
    validate_bounded_text(&event.schema, 256)?;
    if let Some(category) = event.category.as_deref() {
        validate_bounded_text(category, 256)?;
    }
    validate_sha256(&event.canonical_payload_hash)?;
    let expected_hash = event_semantic_hash(event)?;
    if event.canonical_payload_hash != expected_hash {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let canonical_bytes = canonical_serialize_bytes(event)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    if canonical_bytes.len() != event.canonical_size_bytes {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let kind = match event.kind {
        CapturedEventKind::Scope => "scope",
        CapturedEventKind::Mark => "mark",
    };
    let phase = event
        .scope_phase
        .as_ref()
        .map(serialized_enum)
        .transpose()?;
    if (kind == "scope") != phase.is_some() {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let call_role = event.call_role.as_ref().map(serialized_enum).transpose()?;
    Ok(TrajectoryEventRow {
        anchor_id: anchor_id.to_string(),
        ingest_seq: u64_to_i64(event.ingest_seq)?,
        event_uuid: event.event_uuid.to_string(),
        parent_uuid: event.parent_uuid.map(|value| value.to_string()),
        kind: kind.to_string(),
        phase,
        category: event.category.clone(),
        call_role,
        name: event.name.clone(),
        event_time_unix_ms: nonnegative_timestamp(event.timestamp.timestamp_millis())?,
        schema_id: event.schema.clone(),
        sanitized_payload_json: canonical_text(event)?,
        canonical_size_bytes: usize_to_i64(event.canonical_size_bytes)?,
        canonical_payload_hash: event.canonical_payload_hash.clone(),
    })
}

fn event_semantic_hash(event: &CapturedTrajectoryEvent) -> Result<String, LedgerError> {
    let mut value = serde_json::to_value(event)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    for field in [
        "ingest_seq",
        "canonical_payload_hash",
        "canonical_size_bytes",
    ] {
        object.remove(field);
    }
    canonical_hash(&value)
}

fn validate_command_metadata(
    state_event_id: Uuid,
    conflict_health_event_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    validate_uuid_v7(state_event_id)?;
    validate_uuid_v7(conflict_health_event_id)?;
    if state_event_id == conflict_health_event_id || created_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn validate_uuid_v7(value: Uuid) -> Result<(), LedgerError> {
    if value.get_version_num() != 7 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn parse_canonical_uuid_v7(value: &str) -> Result<Uuid, LedgerError> {
    let parsed = Uuid::parse_str(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    validate_uuid_v7(parsed)?;
    if parsed.to_string() != value {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(parsed)
}

fn validate_sha256(value: &str) -> Result<(), LedgerError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn validate_bounded_text(value: &str, max_bytes: usize) -> Result<(), LedgerError> {
    if value.is_empty() || value.len() > max_bytes || value.chars().any(char::is_control) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn nonnegative_timestamp(value: i64) -> Result<i64, LedgerError> {
    if value < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(value)
}

fn validate_exact_millisecond(value: chrono::DateTime<Utc>) -> Result<(), LedgerError> {
    if !value.timestamp_subsec_nanos().is_multiple_of(1_000_000) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn usize_to_i64(value: usize) -> Result<i64, LedgerError> {
    i64::try_from(value).map_err(|_| LedgerErrorClass::IdentityInvariant.into())
}

fn u64_to_i64(value: u64) -> Result<i64, LedgerError> {
    i64::try_from(value).map_err(|_| LedgerErrorClass::IdentityInvariant.into())
}

fn serialized_enum<T: Serialize>(value: &T) -> Result<String, LedgerError> {
    serde_json::to_value(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?
        .as_str()
        .map(str::to_string)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn canonical_text<T: Serialize>(value: &T) -> Result<String, LedgerError> {
    let value = serde_json::to_value(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    canonical_json(&value).map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn canonical_hash<T: Serialize>(value: &T) -> Result<String, LedgerError> {
    let value = serde_json::to_value(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    canonical_sha256(&value).map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn canonical_hash_without_field<T: Serialize>(
    value: &T,
    field: &str,
) -> Result<String, LedgerError> {
    let mut value = serde_json::to_value(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    value
        .as_object_mut()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?
        .remove(field);
    canonical_sha256(&value).map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn anchor_state_hash(
    state_event_id: Uuid,
    anchor_id: Uuid,
    process_instance_id: Uuid,
    dead_process_instance_id: Option<Uuid>,
    state: &str,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    canonical_hash(&json!({
        "anchor_state_event_id": state_event_id,
        "anchor_id": anchor_id,
        "process_instance_id": process_instance_id,
        "dead_process_instance_id": dead_process_instance_id,
        "state": state,
        "created_at_unix_ms": created_at_unix_ms,
    }))
}

/// Append the only legal terminal for one canonical pending non-resumable anchor.
pub(super) fn orphan_pending_anchor_in_transaction(
    connection: &Connection,
    project_uuid: Uuid,
    reconciler_process_instance_id: Uuid,
    dead_process_instance_id: Uuid,
    anchor_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<bool, LedgerError> {
    if reconciler_process_instance_id == dead_process_instance_id || created_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let anchor = load_anchor_by_id(connection, &anchor_id.to_string())?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if anchor.project_uuid != project_uuid.to_string()
        || anchor.process_instance_id != dead_process_instance_id.to_string()
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let pending = load_pending_state(connection, &anchor.anchor_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let pending_event_id = parse_canonical_uuid_v7(&pending.anchor_state_event_id)?;
    let expected_pending_hash = anchor_state_hash(
        pending_event_id,
        anchor_id,
        dead_process_instance_id,
        None,
        "pending",
        pending.created_at_unix_ms,
    )?;
    if pending.process_instance_id != dead_process_instance_id.to_string()
        || pending.dead_process_instance_id.is_some()
        || pending.state != "pending"
        || pending.canonical_payload_hash != expected_pending_hash
        || pending.created_at_unix_ms < anchor.opened_at_unix_ms
        || load_window(connection, &anchor.anchor_id)?.is_some()
        || created_at_unix_ms < pending.created_at_unix_ms
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let batch_exists: bool = connection
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM sample_batches WHERE anchor_id = ?1)",
            params![anchor_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    if batch_exists {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    if let Some(terminal) = load_terminal_state(connection, &anchor.anchor_id)? {
        let terminal_event_id = parse_canonical_uuid_v7(&terminal.anchor_state_event_id)?;
        let stored_reconciler = parse_canonical_uuid_v7(&terminal.process_instance_id)?;
        let expected_hash = anchor_state_hash(
            terminal_event_id,
            anchor_id,
            stored_reconciler,
            Some(dead_process_instance_id),
            "orphaned_non_resumable",
            terminal.created_at_unix_ms,
        )?;
        let reconciler_belongs_to_project: bool = connection
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM process_instances
                    WHERE project_uuid = ?1 AND process_instance_id = ?2
                 )",
                params![project_uuid.to_string(), stored_reconciler.to_string()],
                |row| row.get(0),
            )
            .map_err(database_error)?;
        if stored_reconciler != dead_process_instance_id
            && reconciler_belongs_to_project
            && terminal.dead_process_instance_id == Some(dead_process_instance_id.to_string())
            && terminal.state == "orphaned_non_resumable"
            && terminal.created_at_unix_ms >= pending.created_at_unix_ms
            && terminal.canonical_payload_hash == expected_hash
        {
            return Ok(false);
        }
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let event_id = Uuid::now_v7();
    let payload_hash = anchor_state_hash(
        event_id,
        anchor_id,
        reconciler_process_instance_id,
        Some(dead_process_instance_id),
        "orphaned_non_resumable",
        created_at_unix_ms,
    )?;
    connection
        .execute(
            "INSERT INTO anchor_state_events (
                anchor_state_event_id, anchor_id, process_instance_id,
                dead_process_instance_id, state, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, 'orphaned_non_resumable', ?5, ?6)",
            params![
                event_id.to_string(),
                anchor_id.to_string(),
                reconciler_process_instance_id.to_string(),
                dead_process_instance_id.to_string(),
                created_at_unix_ms,
                payload_hash,
            ],
        )
        .map_err(database_error)?;
    Ok(true)
}

fn database_error(error: rusqlite::Error) -> LedgerError {
    map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}

fn verify_project_id(connection: &Connection, pending: &PendingRows) -> Result<(), LedgerError> {
    let stored = connection
        .query_row(
            "SELECT project_id FROM project_metadata WHERE project_uuid = ?1",
            params![pending.project_uuid.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    if stored.as_deref() != Some(pending.project_id.as_str()) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn probe_pending_in_transaction(
    connection: &Connection,
    pending: &PendingRows,
) -> Result<ProbeResult, LedgerError> {
    let by_id = load_anchor_by_id(connection, &pending.anchor.anchor_id)?;
    let by_alternate = load_anchor_by_alternate(connection, &pending.anchor)?;
    if by_id.is_none() && by_alternate.is_none() {
        return Ok(ProbeResult::Missing);
    }
    if by_id.as_ref() != Some(&pending.anchor) || by_alternate.as_ref() != Some(&pending.anchor) {
        return Ok(ProbeResult::Conflict(conflict_context(
            by_id.as_ref().or(by_alternate.as_ref()),
        )?));
    }
    if let Some(context) = pending_prerequisite_conflict(connection, pending)? {
        return Ok(ProbeResult::Conflict(context));
    }

    let terminal_state = load_terminal_state(connection, &pending.anchor.anchor_id)?;
    let window = load_window(connection, &pending.anchor.anchor_id)?;
    let events = load_events(connection, &pending.anchor.anchor_id)?;
    match (terminal_state, window) {
        (None, None) if events.is_empty() => Ok(ProbeResult::Pending),
        (Some(state), None) if events.is_empty() => {
            let pending_state = load_pending_state(connection, &pending.anchor.anchor_id)?;
            if stored_state_is_canonical(&state)
                && state.anchor_id == pending.anchor.anchor_id
                && state.process_instance_id == pending.anchor.process_instance_id
                && state.dead_process_instance_id.is_none()
                && state.state == "not_scheduled_queue_full"
                && pending_state.as_ref().is_some_and(|pending_state| {
                    pending_state.created_at_unix_ms >= pending.anchor.opened_at_unix_ms
                        && state.created_at_unix_ms >= pending_state.created_at_unix_ms
                })
            {
                Ok(ProbeResult::NotScheduledQueueFull(
                    state.canonical_payload_hash,
                ))
            } else {
                Ok(ProbeResult::Conflict(ConflictContext {
                    anchor_id: Some(pending.anchor_id),
                }))
            }
        }
        (Some(state), Some(window)) => {
            let structurally_canonical = stored_state_is_canonical(&state)
                && state.anchor_id == pending.anchor.anchor_id
                && state.process_instance_id == pending.anchor.process_instance_id
                && state.dead_process_instance_id.is_none()
                && state.state == window.terminal_kind
                && matches!(state.state.as_str(), "closed" | "rejected")
                && state.created_at_unix_ms == window.closed_at_unix_ms
                && stored_window_is_canonical(&window)
                && window.anchor_id == pending.anchor.anchor_id
                && window.requested_progress == pending.anchor.requested_progress;
            let reproduced_hash = structurally_canonical
                .then(|| reproduce_terminal_hash(pending, &window, &events))
                .transpose()?
                .flatten();
            if reproduced_hash.as_deref() == Some(window.terminal_hash.as_str()) {
                Ok(ProbeResult::Terminal(window.terminal_hash))
            } else {
                Ok(ProbeResult::Conflict(ConflictContext {
                    anchor_id: Some(pending.anchor_id),
                }))
            }
        }
        _ => Ok(ProbeResult::Conflict(ConflictContext {
            anchor_id: Some(pending.anchor_id),
        })),
    }
}

fn pending_prerequisite_conflict(
    connection: &Connection,
    pending: &PendingRows,
) -> Result<Option<ConflictContext>, LedgerError> {
    let by_id = load_anchor_by_id(connection, &pending.anchor.anchor_id)?;
    let by_alternate = load_anchor_by_alternate(connection, &pending.anchor)?;
    if by_id.as_ref() != Some(&pending.anchor) || by_alternate.as_ref() != Some(&pending.anchor) {
        return Ok(Some(conflict_context(
            by_id.as_ref().or(by_alternate.as_ref()),
        )?));
    }
    if load_result(connection, &pending.anchor.anchor_id)?.as_ref() != Some(&pending.result) {
        return Ok(Some(ConflictContext {
            anchor_id: Some(pending.anchor_id),
        }));
    }
    let state = load_pending_state(connection, &pending.anchor.anchor_id)?;
    if !state.as_ref().is_some_and(|state| {
        state.anchor_id == pending.anchor.anchor_id
            && state.process_instance_id == pending.anchor.process_instance_id
            && state.dead_process_instance_id.is_none()
            && state.created_at_unix_ms >= pending.anchor.opened_at_unix_ms
            && stored_state_is_canonical(state)
    }) {
        return Ok(Some(ConflictContext {
            anchor_id: Some(pending.anchor_id),
        }));
    }
    Ok(None)
}

fn insert_or_verify_anchor(
    connection: &Connection,
    expected: &AnchorRow,
) -> Result<InsertResult, LedgerError> {
    let by_id = load_anchor_by_id(connection, &expected.anchor_id)?;
    let by_alternate = load_anchor_by_alternate(connection, expected)?;
    match (by_id.as_ref(), by_alternate.as_ref()) {
        (None, None) => {
            let inserted = connection
                .execute(
                    "INSERT INTO anchors (
                        anchor_id, project_uuid, process_instance_id,
                        config_generation_id, policy_version_id,
                        learning_generation_id, pool_id, anchor_call_uuid,
                        root_uuid, owner_uuid, owner_path_json, api_family,
                        transport_identity, anchor_model, anchor_model_revision,
                        replay_capability_fingerprint, decoding_fingerprint,
                        request_projection_json, routing_context_projection_json,
                        candidate_facts_json, requested_progress, opened_at_unix_ms,
                        deadline_at_unix_ms, non_resumable, pending_hash,
                        canonical_payload_hash
                     ) VALUES (
                        ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                        ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22,
                        ?23, ?24, ?25, ?26
                     ) ON CONFLICT(anchor_id) DO NOTHING",
                    params![
                        expected.anchor_id,
                        expected.project_uuid,
                        expected.process_instance_id,
                        expected.config_generation_id,
                        expected.policy_version_id,
                        expected.learning_generation_id,
                        expected.pool_id,
                        expected.anchor_call_uuid,
                        expected.root_uuid,
                        expected.owner_uuid,
                        expected.owner_path_json,
                        expected.api_family,
                        expected.transport_identity,
                        expected.anchor_model,
                        expected.anchor_model_revision,
                        expected.replay_capability_fingerprint,
                        expected.decoding_fingerprint,
                        expected.request_projection_json,
                        expected.routing_context_projection_json,
                        expected.candidate_facts_json,
                        expected.requested_progress,
                        expected.opened_at_unix_ms,
                        expected.deadline_at_unix_ms,
                        expected.non_resumable,
                        expected.pending_hash,
                        expected.canonical_payload_hash,
                    ],
                )
                .map_err(database_error)?;
            if inserted == 1 {
                Ok(InsertResult::Status(InsertStatus::Inserted))
            } else {
                Ok(InsertResult::Conflict(ConflictContext { anchor_id: None }))
            }
        }
        (Some(by_id), Some(by_alternate)) if by_id == expected && by_alternate == expected => {
            Ok(InsertResult::Status(InsertStatus::Existing))
        }
        _ => Ok(InsertResult::Conflict(conflict_context(
            by_id.as_ref().or(by_alternate.as_ref()),
        )?)),
    }
}

fn insert_or_verify_result(
    connection: &Connection,
    expected: &AnchorResultRow,
) -> Result<InsertResult, LedgerError> {
    if let Some(stored) = load_result(connection, &expected.anchor_id)? {
        return if stored == *expected {
            Ok(InsertResult::Status(InsertStatus::Existing))
        } else {
            Ok(InsertResult::Conflict(context_for_expected_anchor(
                &expected.anchor_id,
            )?))
        };
    }
    let inserted = connection
        .execute(
            "INSERT INTO anchor_results (
                anchor_id, normalized_response_json,
                semantic_response_fingerprint, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(anchor_id) DO NOTHING",
            params![
                expected.anchor_id,
                expected.normalized_response_json,
                expected.semantic_response_fingerprint,
                expected.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if inserted == 1 {
        Ok(InsertResult::Status(InsertStatus::Inserted))
    } else {
        Ok(InsertResult::Conflict(context_for_expected_anchor(
            &expected.anchor_id,
        )?))
    }
}

fn insert_or_verify_state(
    connection: &Connection,
    expected: &AnchorStateRow,
) -> Result<InsertResult, LedgerError> {
    let by_id = load_state_by_id(connection, &expected.anchor_state_event_id)?;
    let by_key = if expected.state == "pending" {
        load_pending_state(connection, &expected.anchor_id)?
    } else {
        load_terminal_state(connection, &expected.anchor_id)?
    };
    match (by_id.as_ref(), by_key.as_ref()) {
        (None, None) => {
            let inserted = connection
                .execute(
                    "INSERT INTO anchor_state_events (
                        anchor_state_event_id, anchor_id, process_instance_id,
                        dead_process_instance_id, state, created_at_unix_ms,
                        canonical_payload_hash
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                     ON CONFLICT(anchor_state_event_id) DO NOTHING",
                    params![
                        expected.anchor_state_event_id,
                        expected.anchor_id,
                        expected.process_instance_id,
                        expected.dead_process_instance_id,
                        expected.state,
                        expected.created_at_unix_ms,
                        expected.canonical_payload_hash,
                    ],
                )
                .map_err(database_error)?;
            if inserted == 1 {
                Ok(InsertResult::Status(InsertStatus::Inserted))
            } else {
                let collision = load_state_by_id(connection, &expected.anchor_state_event_id)?;
                Ok(InsertResult::Conflict(state_conflict_context(
                    collision.as_ref(),
                    None,
                )?))
            }
        }
        (Some(by_id), Some(by_key)) if by_id == expected && by_key == expected => {
            Ok(InsertResult::Status(InsertStatus::Existing))
        }
        _ => Ok(InsertResult::Conflict(state_conflict_context(
            by_id.as_ref(),
            by_key.as_ref(),
        )?)),
    }
}

fn insert_or_verify_window(
    connection: &Connection,
    expected: &AnchorWindowRow,
) -> Result<InsertResult, LedgerError> {
    if let Some(stored) = load_window(connection, &expected.anchor_id)? {
        return if stored == *expected {
            Ok(InsertResult::Status(InsertStatus::Existing))
        } else {
            Ok(InsertResult::Conflict(context_for_expected_anchor(
                &expected.anchor_id,
            )?))
        };
    }
    let inserted = connection
        .execute(
            "INSERT INTO anchor_windows (
                anchor_id, requested_progress, observed_progress, terminal_kind,
                trigger, rejection_reason, is_partial, promotion_eligible,
                closed_at_unix_ms, diagnostics_json, terminal_hash,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)
             ON CONFLICT(anchor_id) DO NOTHING",
            params![
                expected.anchor_id,
                expected.requested_progress,
                expected.observed_progress,
                expected.terminal_kind,
                expected.trigger,
                expected.rejection_reason,
                expected.is_partial,
                expected.promotion_eligible,
                expected.closed_at_unix_ms,
                expected.diagnostics_json,
                expected.terminal_hash,
                expected.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if inserted == 1 {
        Ok(InsertResult::Status(InsertStatus::Inserted))
    } else {
        Ok(InsertResult::Conflict(context_for_expected_anchor(
            &expected.anchor_id,
        )?))
    }
}

fn insert_or_verify_event(
    connection: &Connection,
    expected: &TrajectoryEventRow,
) -> Result<InsertResult, LedgerError> {
    if let Some(stored) = load_event(connection, &expected.anchor_id, expected.ingest_seq)? {
        return if stored == *expected {
            Ok(InsertResult::Status(InsertStatus::Existing))
        } else {
            Ok(InsertResult::Conflict(context_for_expected_anchor(
                &expected.anchor_id,
            )?))
        };
    }
    let inserted = connection
        .execute(
            "INSERT INTO trajectory_events (
                anchor_id, ingest_seq, event_uuid, parent_uuid, kind, phase,
                category, call_role, name, event_time_unix_ms, schema_id,
                sanitized_payload_json, canonical_size_bytes,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14)
             ON CONFLICT(anchor_id, ingest_seq) DO NOTHING",
            params![
                expected.anchor_id,
                expected.ingest_seq,
                expected.event_uuid,
                expected.parent_uuid,
                expected.kind,
                expected.phase,
                expected.category,
                expected.call_role,
                expected.name,
                expected.event_time_unix_ms,
                expected.schema_id,
                expected.sanitized_payload_json,
                expected.canonical_size_bytes,
                expected.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if inserted == 1 {
        Ok(InsertResult::Status(InsertStatus::Inserted))
    } else {
        Ok(InsertResult::Conflict(context_for_expected_anchor(
            &expected.anchor_id,
        )?))
    }
}

fn load_anchor_by_id(
    connection: &Connection,
    anchor_id: &str,
) -> Result<Option<AnchorRow>, LedgerError> {
    connection
        .query_row(
            "SELECT anchor_id, project_uuid, process_instance_id,
                    config_generation_id, policy_version_id,
                    learning_generation_id, pool_id, anchor_call_uuid,
                    root_uuid, owner_uuid, owner_path_json, api_family,
                    transport_identity, anchor_model, anchor_model_revision,
                    replay_capability_fingerprint, decoding_fingerprint,
                    request_projection_json, routing_context_projection_json,
                    candidate_facts_json, requested_progress, opened_at_unix_ms,
                    deadline_at_unix_ms, non_resumable, pending_hash,
                    canonical_payload_hash
             FROM anchors WHERE anchor_id = ?1",
            params![anchor_id],
            anchor_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn load_anchor_by_alternate(
    connection: &Connection,
    expected: &AnchorRow,
) -> Result<Option<AnchorRow>, LedgerError> {
    connection
        .query_row(
            "SELECT anchor_id, project_uuid, process_instance_id,
                    config_generation_id, policy_version_id,
                    learning_generation_id, pool_id, anchor_call_uuid,
                    root_uuid, owner_uuid, owner_path_json, api_family,
                    transport_identity, anchor_model, anchor_model_revision,
                    replay_capability_fingerprint, decoding_fingerprint,
                    request_projection_json, routing_context_projection_json,
                    candidate_facts_json, requested_progress, opened_at_unix_ms,
                    deadline_at_unix_ms, non_resumable, pending_hash,
                    canonical_payload_hash
             FROM anchors
             WHERE config_generation_id = ?1 AND learning_generation_id = ?2
               AND pool_id = ?3 AND anchor_call_uuid = ?4",
            params![
                expected.config_generation_id,
                expected.learning_generation_id,
                expected.pool_id,
                expected.anchor_call_uuid,
            ],
            anchor_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn anchor_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AnchorRow> {
    Ok(AnchorRow {
        anchor_id: row.get(0)?,
        project_uuid: row.get(1)?,
        process_instance_id: row.get(2)?,
        config_generation_id: row.get(3)?,
        policy_version_id: row.get(4)?,
        learning_generation_id: row.get(5)?,
        pool_id: row.get(6)?,
        anchor_call_uuid: row.get(7)?,
        root_uuid: row.get(8)?,
        owner_uuid: row.get(9)?,
        owner_path_json: row.get(10)?,
        api_family: row.get(11)?,
        transport_identity: row.get(12)?,
        anchor_model: row.get(13)?,
        anchor_model_revision: row.get(14)?,
        replay_capability_fingerprint: row.get(15)?,
        decoding_fingerprint: row.get(16)?,
        request_projection_json: row.get(17)?,
        routing_context_projection_json: row.get(18)?,
        candidate_facts_json: row.get(19)?,
        requested_progress: row.get(20)?,
        opened_at_unix_ms: row.get(21)?,
        deadline_at_unix_ms: row.get(22)?,
        non_resumable: row.get(23)?,
        pending_hash: row.get(24)?,
        canonical_payload_hash: row.get(25)?,
    })
}

fn load_result(
    connection: &Connection,
    anchor_id: &str,
) -> Result<Option<AnchorResultRow>, LedgerError> {
    connection
        .query_row(
            "SELECT anchor_id, normalized_response_json,
                    semantic_response_fingerprint, canonical_payload_hash
             FROM anchor_results WHERE anchor_id = ?1",
            params![anchor_id],
            |row| {
                Ok(AnchorResultRow {
                    anchor_id: row.get(0)?,
                    normalized_response_json: row.get(1)?,
                    semantic_response_fingerprint: row.get(2)?,
                    canonical_payload_hash: row.get(3)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn load_state_by_id(
    connection: &Connection,
    state_event_id: &str,
) -> Result<Option<AnchorStateRow>, LedgerError> {
    connection
        .query_row(
            "SELECT anchor_state_event_id, anchor_id, process_instance_id,
                    dead_process_instance_id, state, created_at_unix_ms,
                    canonical_payload_hash
             FROM anchor_state_events WHERE anchor_state_event_id = ?1",
            params![state_event_id],
            anchor_state_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn load_pending_state(
    connection: &Connection,
    anchor_id: &str,
) -> Result<Option<AnchorStateRow>, LedgerError> {
    connection
        .query_row(
            "SELECT anchor_state_event_id, anchor_id, process_instance_id,
                    dead_process_instance_id, state, created_at_unix_ms,
                    canonical_payload_hash
             FROM anchor_state_events
             WHERE anchor_id = ?1 AND state = 'pending'",
            params![anchor_id],
            anchor_state_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn load_terminal_state(
    connection: &Connection,
    anchor_id: &str,
) -> Result<Option<AnchorStateRow>, LedgerError> {
    connection
        .query_row(
            "SELECT anchor_state_event_id, anchor_id, process_instance_id,
                    dead_process_instance_id, state, created_at_unix_ms,
                    canonical_payload_hash
             FROM anchor_state_events
             WHERE anchor_id = ?1 AND state <> 'pending'",
            params![anchor_id],
            anchor_state_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn anchor_state_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AnchorStateRow> {
    Ok(AnchorStateRow {
        anchor_state_event_id: row.get(0)?,
        anchor_id: row.get(1)?,
        process_instance_id: row.get(2)?,
        dead_process_instance_id: row.get(3)?,
        state: row.get(4)?,
        created_at_unix_ms: row.get(5)?,
        canonical_payload_hash: row.get(6)?,
    })
}

fn load_window(
    connection: &Connection,
    anchor_id: &str,
) -> Result<Option<AnchorWindowRow>, LedgerError> {
    connection
        .query_row(
            "SELECT anchor_id, requested_progress, observed_progress,
                    terminal_kind, trigger, rejection_reason, is_partial,
                    promotion_eligible, closed_at_unix_ms, diagnostics_json,
                    terminal_hash, canonical_payload_hash
             FROM anchor_windows WHERE anchor_id = ?1",
            params![anchor_id],
            |row| {
                Ok(AnchorWindowRow {
                    anchor_id: row.get(0)?,
                    requested_progress: row.get(1)?,
                    observed_progress: row.get(2)?,
                    terminal_kind: row.get(3)?,
                    trigger: row.get(4)?,
                    rejection_reason: row.get(5)?,
                    is_partial: row.get(6)?,
                    promotion_eligible: row.get(7)?,
                    closed_at_unix_ms: row.get(8)?,
                    diagnostics_json: row.get(9)?,
                    terminal_hash: row.get(10)?,
                    canonical_payload_hash: row.get(11)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn load_event(
    connection: &Connection,
    anchor_id: &str,
    ingest_seq: i64,
) -> Result<Option<TrajectoryEventRow>, LedgerError> {
    connection
        .query_row(
            "SELECT anchor_id, ingest_seq, event_uuid, parent_uuid, kind,
                    phase, category, call_role, name, event_time_unix_ms,
                    schema_id, sanitized_payload_json, canonical_size_bytes,
                    canonical_payload_hash
             FROM trajectory_events WHERE anchor_id = ?1 AND ingest_seq = ?2",
            params![anchor_id, ingest_seq],
            trajectory_event_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn load_events(
    connection: &Connection,
    anchor_id: &str,
) -> Result<Vec<TrajectoryEventRow>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT anchor_id, ingest_seq, event_uuid, parent_uuid, kind,
                    phase, category, call_role, name, event_time_unix_ms,
                    schema_id, sanitized_payload_json, canonical_size_bytes,
                    canonical_payload_hash
             FROM trajectory_events WHERE anchor_id = ?1 ORDER BY ingest_seq",
        )
        .map_err(database_error)?;
    statement
        .query_map(params![anchor_id], trajectory_event_from_row)
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)
}

fn trajectory_event_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<TrajectoryEventRow> {
    Ok(TrajectoryEventRow {
        anchor_id: row.get(0)?,
        ingest_seq: row.get(1)?,
        event_uuid: row.get(2)?,
        parent_uuid: row.get(3)?,
        kind: row.get(4)?,
        phase: row.get(5)?,
        category: row.get(6)?,
        call_role: row.get(7)?,
        name: row.get(8)?,
        event_time_unix_ms: row.get(9)?,
        schema_id: row.get(10)?,
        sanitized_payload_json: row.get(11)?,
        canonical_size_bytes: row.get(12)?,
        canonical_payload_hash: row.get(13)?,
    })
}

fn stored_state_is_canonical(state: &AnchorStateRow) -> bool {
    let Ok(state_event_id) = parse_canonical_uuid_v7(&state.anchor_state_event_id) else {
        return false;
    };
    let Ok(anchor_id) = parse_canonical_uuid_v7(&state.anchor_id) else {
        return false;
    };
    let Ok(process_instance_id) = parse_canonical_uuid_v7(&state.process_instance_id) else {
        return false;
    };
    let dead_process_instance_id = match state.dead_process_instance_id.as_deref() {
        Some(value) => match parse_canonical_uuid_v7(value) {
            Ok(value) => Some(value),
            Err(_) => return false,
        },
        None => None,
    };
    anchor_state_hash(
        state_event_id,
        anchor_id,
        process_instance_id,
        dead_process_instance_id,
        &state.state,
        state.created_at_unix_ms,
    )
    .is_ok_and(|hash| hash == state.canonical_payload_hash)
}

fn stored_window_is_canonical(window: &AnchorWindowRow) -> bool {
    validate_sha256(&window.terminal_hash).is_ok()
        && window.canonical_payload_hash == window.terminal_hash
        && matches!(window.terminal_kind.as_str(), "closed" | "rejected")
        && ((window.terminal_kind == "closed"
            && window.trigger.is_some()
            && window.rejection_reason.is_none())
            || (window.terminal_kind == "rejected"
                && window.trigger.is_none()
                && window.rejection_reason.is_some()))
}

fn reproduce_terminal_hash(
    pending: &PendingRows,
    window: &AnchorWindowRow,
    events: &[TrajectoryEventRow],
) -> Result<Option<String>, LedgerError> {
    let mut captured_events = Vec::with_capacity(events.len());
    for event in events {
        if !stored_event_is_canonical(event) {
            return Ok(None);
        }
        let Ok(captured) =
            serde_json::from_str::<CapturedTrajectoryEvent>(&event.sanitized_payload_json)
        else {
            return Ok(None);
        };
        captured_events.push(captured);
    }
    let (state, expected_partial, progress_reached) = match window.terminal_kind.as_str() {
        "closed" => {
            let Some(trigger) = window
                .trigger
                .as_deref()
                .and_then(deserialize_enum::<TrajectoryTrigger>)
            else {
                return Ok(None);
            };
            (
                TrajectoryTerminalStateV1::Closed { trigger },
                trigger != TrajectoryTrigger::ProgressReached,
                trigger == TrajectoryTrigger::ProgressReached,
            )
        }
        "rejected" => {
            let Some(reason) = window
                .rejection_reason
                .as_deref()
                .and_then(deserialize_enum::<TrajectoryRejectionReason>)
            else {
                return Ok(None);
            };
            (TrajectoryTerminalStateV1::Rejected { reason }, true, false)
        }
        _ => return Ok(None),
    };
    let Some(closed_at) = Utc.timestamp_millis_opt(window.closed_at_unix_ms).single() else {
        return Ok(None);
    };
    let Ok(observed_progress) = usize::try_from(window.observed_progress) else {
        return Ok(None);
    };
    let diagnostics =
        match serde_json::from_str::<Vec<TrajectoryDiagnosticV1>>(&window.diagnostics_json) {
            Ok(diagnostics) => diagnostics,
            Err(_) => return Ok(None),
        };
    if canonical_text(&diagnostics)? != window.diagnostics_json {
        return Ok(None);
    }
    let is_partial = match window.is_partial {
        0 => false,
        1 => true,
        _ => return Ok(None),
    };
    let promotion_eligible = match window.promotion_eligible {
        0 => false,
        1 => true,
        _ => return Ok(None),
    };
    if is_partial != expected_partial
        || (progress_reached && window.observed_progress != window.requested_progress)
        || (window.terminal_kind == "closed"
            && !progress_reached
            && window.observed_progress >= window.requested_progress)
        || (is_partial && promotion_eligible)
        || window.closed_at_unix_ms < pending.anchor.opened_at_unix_ms
    {
        return Ok(None);
    }
    let terminal = PersistedTrajectoryTerminalV1 {
        schema: TERMINAL_TRAJECTORY_SCHEMA_V1.to_string(),
        pending: pending.pending_dto.clone(),
        state,
        events: captured_events,
        observed_progress,
        is_partial,
        promotion_eligible,
        closed_at,
        diagnostics,
    };
    terminal
        .payload_hash()
        .map(Some)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn deserialize_enum<T: DeserializeOwned>(value: &str) -> Option<T> {
    serde_json::from_value(Json::String(value.to_string())).ok()
}

fn stored_event_is_canonical(event: &TrajectoryEventRow) -> bool {
    let Ok(captured) =
        serde_json::from_str::<CapturedTrajectoryEvent>(&event.sanitized_payload_json)
    else {
        return false;
    };
    let Ok(anchor_id) = parse_canonical_uuid_v7(&event.anchor_id) else {
        return false;
    };
    prepare_trajectory_event(anchor_id, &captured).is_ok_and(|prepared| prepared == *event)
}

fn context_for_expected_anchor(anchor_id: &str) -> Result<ConflictContext, LedgerError> {
    let anchor_id = parse_canonical_uuid_v7(anchor_id)?;
    Ok(ConflictContext {
        anchor_id: Some(anchor_id),
    })
}

fn conflict_context(stored: Option<&AnchorRow>) -> Result<ConflictContext, LedgerError> {
    let anchor_id = stored
        .map(|anchor| parse_canonical_uuid_v7(&anchor.anchor_id))
        .transpose()?;
    Ok(ConflictContext { anchor_id })
}

fn state_conflict_context(
    by_id: Option<&AnchorStateRow>,
    by_key: Option<&AnchorStateRow>,
) -> Result<ConflictContext, LedgerError> {
    let anchor_id = by_id
        .or(by_key)
        .map(|state| parse_canonical_uuid_v7(&state.anchor_id))
        .transpose()?;
    Ok(ConflictContext { anchor_id })
}

#[cfg(test)]
mod tests {
    use chrono::{TimeZone, Utc};
    use nemo_relay::api::event::ScopeCategory;
    use nemo_relay::api::llm::LlmCallRole;
    use rusqlite::params;
    use serde_json::json;
    use tempfile::tempdir;

    use super::*;
    use crate::ledger::model::LedgerRuntimeIdentity;
    use crate::ledger::repository::process::{ProcessCommandAck, ProcessStop};
    use crate::ledger::repository::tests::{config, database_path};
    use crate::trajectory::test_fixtures::pending_window;
    use crate::trajectory::{
        CAPTURED_EVENT_SCHEMA_V1, CapturedCodecAnnotationsV1, CapturedEventKind,
        CapturedTrajectoryEvent, PersistedCandidateCapabilitiesV1, PersistedCandidateFactV1,
        PersistedTrajectoryTerminalV1, TrajectoryTrigger,
    };

    const CREATED_AT: i64 = 1_700_000_001_000;

    fn activate() -> (tempfile::TempDir, super::super::ActivatedLedger) {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let activated = LedgerRepository::activate(&config(&path, "anchor-project")).unwrap();
        (temporary, activated)
    }

    fn pending(
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
        pending.request_projection.normalized_request.model = Some("anchor-a".into());
        pending.request_projection.semantic_request_fingerprint = canonical_hash_without_field(
            &pending.request_projection,
            "semantic_request_fingerprint",
        )
        .unwrap();
        pending.routing_context_projection.tenant_policy_hash = "2".repeat(64);
        pending.routing_context_projection.agent_policy_hash = "3".repeat(64);
        pending.normalized_anchor_response.model = Some("anchor-a".into());
        pending
            .normalized_anchor_response
            .semantic_response_fingerprint = canonical_hash_without_field(
            &pending.normalized_anchor_response,
            "semantic_response_fingerprint",
        )
        .unwrap();
        pending.candidate_facts = vec![PersistedCandidateFactV1 {
            schema: CANDIDATE_FACT_SCHEMA_V1.into(),
            candidate_id: "candidate-a".into(),
            model: "candidate-model-a".into(),
            model_revision: "2026-06-01".into(),
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
    }

    fn frozen_pending(pending: &PendingTrajectoryWindow) -> FrozenPendingAnchorV1 {
        FrozenPendingAnchorV1::new(pending, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT).unwrap()
    }

    fn captured_event(ingest_seq: u64) -> CapturedTrajectoryEvent {
        let mut event = CapturedTrajectoryEvent {
            schema: CAPTURED_EVENT_SCHEMA_V1.into(),
            ingest_seq,
            event_uuid: Uuid::now_v7(),
            parent_uuid: Some(Uuid::now_v7()),
            kind: CapturedEventKind::Scope,
            scope_phase: Some(ScopeCategory::Start),
            category: Some("llm".into()),
            call_role: Some(LlmCallRole::Primary),
            timestamp: Utc
                .timestamp_millis_opt(CREATED_AT + ingest_seq as i64)
                .unwrap(),
            name: format!("event-{ingest_seq}"),
            data: Some(json!({"safe": ingest_seq})),
            metadata: Some(json!({"source": "test"})),
            data_schema: None,
            scope_type: None,
            safe_scope_attributes: vec!["bounded".into()],
            codec_annotations: CapturedCodecAnnotationsV1::default(),
            canonical_payload_hash: String::new(),
            canonical_size_bytes: 0,
        };
        event.canonical_payload_hash = event_semantic_hash(&event).unwrap();
        for _ in 0..8 {
            let size = canonical_serialize_bytes(&event).unwrap().len();
            if size == event.canonical_size_bytes {
                break;
            }
            event.canonical_size_bytes = size;
        }
        assert_eq!(
            event.canonical_size_bytes,
            canonical_serialize_bytes(&event).unwrap().len()
        );
        event
    }

    fn terminal(
        pending: PendingTrajectoryWindow,
        events: Vec<CapturedTrajectoryEvent>,
    ) -> PersistedTrajectoryTerminalV1 {
        PersistedTrajectoryTerminalV1::closed(
            pending,
            events,
            1,
            TrajectoryTrigger::ProgressReached,
            Utc.timestamp_millis_opt(CREATED_AT + 10_000).unwrap(),
            Vec::new(),
        )
    }

    fn scalar(connection: &Connection, sql: &str) -> i64 {
        connection.query_row(sql, [], |row| row.get(0)).unwrap()
    }

    #[test]
    fn pending_probe_and_exact_retry_are_idempotent() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen = frozen_pending(&pending);

        assert_eq!(
            activated.repository.probe_pending_anchor(&frozen).unwrap(),
            AnchorProbe::Missing {
                anchor_id: pending.anchor_id
            }
        );
        assert_eq!(
            activated.repository.record_pending_anchor(&frozen).unwrap(),
            AnchorCommandAck::Applied {
                anchor_id: pending.anchor_id,
                canonical_payload_hash: frozen.pending_hash().to_string(),
            }
        );
        assert_eq!(
            activated.repository.record_pending_anchor(&frozen).unwrap(),
            AnchorCommandAck::AlreadyApplied {
                anchor_id: pending.anchor_id,
                canonical_payload_hash: frozen.pending_hash().to_string(),
            }
        );
        assert_eq!(
            activated.repository.probe_pending_anchor(&frozen).unwrap(),
            AnchorProbe::Pending {
                anchor_id: pending.anchor_id,
                pending_hash: frozen.pending_hash().to_string(),
            }
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchors"
            ),
            1
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchor_results"
            ),
            1
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchor_state_events WHERE state = 'pending'"
            ),
            1
        );
    }

    #[test]
    fn queue_full_decline_is_atomic_exact_and_probeable_without_a_window() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen = frozen_pending(&pending);
        let decline =
            NotScheduledQueueFull::new(frozen.clone(), Uuid::now_v7(), CREATED_AT + 1).unwrap();

        assert_eq!(
            activated
                .repository
                .record_not_scheduled_queue_full(&decline)
                .unwrap(),
            NotScheduledQueueFullAck::Applied {
                anchor_id: pending.anchor_id,
                pending_hash: decline.pending_hash().to_string(),
                terminal_hash: decline.terminal_hash().to_string(),
            }
        );
        assert_eq!(
            activated.repository.probe_pending_anchor(&frozen).unwrap(),
            AnchorProbe::NotScheduledQueueFull {
                anchor_id: pending.anchor_id,
                pending_hash: decline.pending_hash().to_string(),
                terminal_hash: decline.terminal_hash().to_string(),
            }
        );
        assert_eq!(
            activated
                .repository
                .record_not_scheduled_queue_full(&decline)
                .unwrap(),
            NotScheduledQueueFullAck::AlreadyApplied {
                anchor_id: pending.anchor_id,
                pending_hash: decline.pending_hash().to_string(),
                terminal_hash: decline.terminal_hash().to_string(),
            }
        );
        let recovered = frozen_pending(&pending);
        assert_eq!(
            activated
                .repository
                .record_pending_anchor_with_capacity_and_start_check(&recovered, 0, || Some(()))
                .unwrap(),
            PendingAnchorCapacityAck::NotScheduledQueueFull {
                anchor_id: pending.anchor_id,
                pending_hash: decline.pending_hash().to_string(),
                terminal_hash: decline.terminal_hash().to_string(),
            }
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchor_state_events"
            ),
            2
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchor_windows"
            ),
            0
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM trajectory_events"
            ),
            0
        );
    }

    #[test]
    fn queue_full_decline_can_terminalize_an_exact_existing_pending() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen = frozen_pending(&pending);
        assert!(matches!(
            activated.repository.record_pending_anchor(&frozen).unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
        let recovered_pending = frozen_pending(&pending);
        assert_ne!(
            recovered_pending.state.anchor_state_event_id,
            frozen.state.anchor_state_event_id
        );
        let decline =
            NotScheduledQueueFull::new(recovered_pending, Uuid::now_v7(), CREATED_AT + 1).unwrap();

        assert_eq!(
            activated
                .repository
                .record_not_scheduled_queue_full_with_capacity_and_start_check(
                    &decline,
                    0,
                    || Some(()),
                )
                .unwrap(),
            NotScheduledQueueFullAck::Applied {
                anchor_id: pending.anchor_id,
                pending_hash: decline.pending_hash().to_string(),
                terminal_hash: decline.terminal_hash().to_string(),
            }
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchor_state_events
                 WHERE anchor_id IN (SELECT anchor_id FROM anchors)"
            ),
            2
        );
    }

    #[test]
    fn evidence_capacity_refuses_only_missing_aggregates_without_writes() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen = frozen_pending(&pending);
        assert_eq!(
            activated
                .repository
                .record_pending_anchor_with_capacity_and_start_check(&frozen, 0, || Some(()))
                .unwrap(),
            PendingAnchorCapacityAck::EvidenceCapacity {
                anchor_id: pending.anchor_id,
            }
        );
        let decline = NotScheduledQueueFull::new(frozen, Uuid::now_v7(), CREATED_AT + 1).unwrap();
        assert_eq!(
            activated
                .repository
                .record_not_scheduled_queue_full_with_capacity_and_start_check(
                    &decline,
                    0,
                    || Some(()),
                )
                .unwrap(),
            NotScheduledQueueFullAck::EvidenceCapacity {
                anchor_id: pending.anchor_id,
            }
        );
        for table in ["anchors", "anchor_results", "anchor_state_events"] {
            let count = activated
                .repository
                .connection
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap();
            assert_eq!(count, 0, "capacity refusal wrote {table}");
        }
    }

    #[test]
    fn capacity_aware_pending_write_preserves_a_racing_terminal_result() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen = frozen_pending(&pending);
        activated.repository.record_pending_anchor(&frozen).unwrap();
        let recovered = frozen_pending(&pending);
        assert!(matches!(
            activated
                .repository
                .probe_pending_anchor(&recovered)
                .unwrap(),
            AnchorProbe::Pending { .. }
        ));
        let terminal = terminal(pending.clone(), Vec::new());
        let frozen_terminal = FrozenTerminalAnchorV1::new(
            &terminal,
            Uuid::now_v7(),
            Uuid::now_v7(),
            CREATED_AT + 10_000,
        )
        .unwrap();
        activated
            .repository
            .record_terminal_anchor(&frozen_terminal)
            .unwrap();

        assert_eq!(
            activated
                .repository
                .record_pending_anchor_with_capacity_and_start_check(&recovered, 0, || Some(()))
                .unwrap(),
            PendingAnchorCapacityAck::AlreadyTerminal {
                anchor_id: pending.anchor_id,
                pending_hash: recovered.pending_hash().to_string(),
                terminal_hash: frozen_terminal.terminal_hash().to_string(),
            }
        );
    }

    #[test]
    fn queue_full_decline_alternate_terminal_identity_conflicts_without_overwrite() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen = frozen_pending(&pending);
        let original =
            NotScheduledQueueFull::new(frozen.clone(), Uuid::now_v7(), CREATED_AT + 1).unwrap();
        assert!(matches!(
            activated
                .repository
                .record_not_scheduled_queue_full(&original)
                .unwrap(),
            NotScheduledQueueFullAck::Applied { .. }
        ));
        let alternate = NotScheduledQueueFull::new(frozen, Uuid::now_v7(), CREATED_AT + 1).unwrap();

        assert_eq!(
            activated
                .repository
                .record_not_scheduled_queue_full(&alternate)
                .unwrap(),
            NotScheduledQueueFullAck::Conflict {
                anchor_id: pending.anchor_id,
            }
        );
        assert_eq!(
            activated
                .repository
                .record_not_scheduled_queue_full(&alternate)
                .unwrap(),
            NotScheduledQueueFullAck::Conflict {
                anchor_id: pending.anchor_id,
            }
        );
        let stored_hash: String = activated
            .repository
            .connection
            .query_row(
                "SELECT canonical_payload_hash FROM anchor_state_events
                 WHERE anchor_id = ?1 AND state = 'not_scheduled_queue_full'",
                [pending.anchor_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stored_hash, original.terminal_hash());
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM health_events
                 WHERE stable_class = 'router.ledger.integrity_conflict'"
            ),
            1
        );
    }

    #[test]
    fn new_pending_requires_active_policy_and_latest_learning_generation() {
        let (_temporary, mut activated) = activate();
        let stale = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let stale_frozen = frozen_pending(&stale);
        let current_generation = activated
            .repository
            .reset_pool("pool-a", "test", "advance generation")
            .unwrap();
        assert_eq!(
            activated
                .repository
                .probe_pending_anchor(&stale_frozen)
                .unwrap_err()
                .class(),
            LedgerErrorClass::IdentityInvariant
        );
        assert_eq!(
            activated
                .repository
                .record_pending_anchor(&stale_frozen)
                .unwrap_err()
                .class(),
            LedgerErrorClass::IdentityInvariant
        );

        let mut current = stale.clone();
        current.anchor_id = Uuid::now_v7();
        current.anchor_call_uuid = Uuid::now_v7();
        current.learning_generation_id = current_generation;
        let current_frozen = frozen_pending(&current);
        assert!(matches!(
            activated
                .repository
                .record_pending_anchor(&current_frozen)
                .unwrap(),
            AnchorCommandAck::Applied { .. }
        ));

        let stored_policy_json: String = activated
            .repository
            .connection
            .query_row(
                "SELECT canonical_policy_json FROM policy_versions
                 WHERE project_uuid = ?1 AND pool_id = 'pool-a'",
                params![activated.identity.project_uuid.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        let mut historical_policy: Json = serde_json::from_str(&stored_policy_json).unwrap();
        historical_policy["pool"]["sampling_probability"] = json!(0.5);
        let historical_policy_id = canonical_sha256(&historical_policy).unwrap();
        let historical_policy_json = canonical_json(&historical_policy).unwrap();
        activated
            .repository
            .connection
            .execute(
                "INSERT INTO policy_versions (
                    policy_version_id, project_uuid, pool_id,
                    canonical_policy_json, canonical_payload_hash,
                    created_at_unix_ms
                 ) VALUES (?1, ?2, 'pool-a', ?3, ?1, ?4)",
                params![
                    historical_policy_id,
                    activated.identity.project_uuid.to_string(),
                    historical_policy_json,
                    CREATED_AT,
                ],
            )
            .unwrap();
        let mut historical = current;
        historical.anchor_id = Uuid::now_v7();
        historical.anchor_call_uuid = Uuid::now_v7();
        historical.policy_version_id = historical_policy_id;
        let historical_frozen = frozen_pending(&historical);
        assert_eq!(
            activated
                .repository
                .record_pending_anchor(&historical_frozen)
                .unwrap_err()
                .class(),
            LedgerErrorClass::IdentityInvariant
        );
    }

    #[test]
    fn accepted_anchor_can_terminalize_after_learning_generation_advances() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen_pending = frozen_pending(&pending);
        activated
            .repository
            .record_pending_anchor(&frozen_pending)
            .unwrap();
        activated
            .repository
            .reset_pool("pool-a", "test", "advance after acceptance")
            .unwrap();
        let terminal = terminal(pending, Vec::new());
        let frozen_terminal = FrozenTerminalAnchorV1::new(
            &terminal,
            Uuid::now_v7(),
            Uuid::now_v7(),
            CREATED_AT + 10_000,
        )
        .unwrap();
        assert!(matches!(
            activated
                .repository
                .record_terminal_anchor(&frozen_terminal)
                .unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
    }

    #[test]
    fn terminal_events_are_ordered_reproducible_and_exactly_retryable() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen_pending = frozen_pending(&pending);
        activated
            .repository
            .record_pending_anchor(&frozen_pending)
            .unwrap();

        let first = captured_event(10);
        let second = captured_event(20);
        let unordered = terminal(pending.clone(), vec![second.clone(), first.clone()]);
        assert!(
            FrozenTerminalAnchorV1::new(
                &unordered,
                Uuid::now_v7(),
                Uuid::now_v7(),
                CREATED_AT + 10_000,
            )
            .is_err()
        );

        let terminal = terminal(pending.clone(), vec![first.clone(), second.clone()]);
        let frozen_terminal = FrozenTerminalAnchorV1::new(
            &terminal,
            Uuid::now_v7(),
            Uuid::now_v7(),
            CREATED_AT + 10_000,
        )
        .unwrap();
        assert!(matches!(
            activated
                .repository
                .record_terminal_anchor(&frozen_terminal)
                .unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
        assert!(matches!(
            activated
                .repository
                .record_terminal_anchor(&frozen_terminal)
                .unwrap(),
            AnchorCommandAck::AlreadyApplied { .. }
        ));
        assert_eq!(
            activated
                .repository
                .probe_pending_anchor(&frozen_pending)
                .unwrap(),
            AnchorProbe::Terminal {
                anchor_id: pending.anchor_id,
                pending_hash: frozen_pending.pending_hash().to_string(),
                terminal_hash: frozen_terminal.terminal_hash().to_string(),
            }
        );
        let stored: String = activated
            .repository
            .connection
            .query_row(
                "SELECT sanitized_payload_json FROM trajectory_events
                 WHERE anchor_id = ?1 AND ingest_seq = 10",
                params![pending.anchor_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            serde_json::from_str::<CapturedTrajectoryEvent>(&stored).unwrap(),
            first
        );
    }

    #[test]
    fn terminal_probe_rejects_rehashed_invalid_semantics_and_noncanonical_json() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen_pending = frozen_pending(&pending);
        activated
            .repository
            .record_pending_anchor(&frozen_pending)
            .unwrap();
        let terminal = terminal(pending.clone(), vec![captured_event(1)]);
        let frozen_terminal = FrozenTerminalAnchorV1::new(
            &terminal,
            Uuid::now_v7(),
            Uuid::now_v7(),
            CREATED_AT + 10_000,
        )
        .unwrap();
        activated
            .repository
            .record_terminal_anchor(&frozen_terminal)
            .unwrap();
        let mut invalid_terminal = terminal.clone();
        invalid_terminal.state = TrajectoryTerminalStateV1::Closed {
            trigger: TrajectoryTrigger::Shutdown,
        };
        invalid_terminal.is_partial = false;
        let invalid_hash = invalid_terminal.payload_hash().unwrap();
        activated
            .repository
            .connection
            .execute(
                "UPDATE anchor_windows
                 SET trigger = 'shutdown', is_partial = 0,
                     terminal_hash = ?1, canonical_payload_hash = ?1
                 WHERE anchor_id = ?2",
                params![invalid_hash, pending.anchor_id.to_string()],
            )
            .unwrap();

        assert_eq!(
            activated
                .repository
                .probe_pending_anchor(&frozen_pending)
                .unwrap(),
            AnchorProbe::Conflict {
                anchor_id: pending.anchor_id
            }
        );

        activated
            .repository
            .connection
            .execute(
                "UPDATE anchor_windows
                 SET trigger = 'progress_reached', diagnostics_json = '[ ]',
                     terminal_hash = ?1, canonical_payload_hash = ?1
                 WHERE anchor_id = ?2",
                params![
                    frozen_terminal.terminal_hash(),
                    pending.anchor_id.to_string()
                ],
            )
            .unwrap();
        assert!(matches!(
            activated
                .repository
                .probe_pending_anchor(&frozen_pending)
                .unwrap(),
            AnchorProbe::Conflict { .. }
        ));
    }

    #[test]
    fn pending_mismatch_records_one_idempotent_conflict_health_fact() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let original = frozen_pending(&pending);
        activated
            .repository
            .record_pending_anchor(&original)
            .unwrap();

        let mut mismatch = pending.clone();
        mismatch.normalized_anchor_response.id = Some("different-response".into());
        mismatch
            .normalized_anchor_response
            .semantic_response_fingerprint = canonical_hash_without_field(
            &mismatch.normalized_anchor_response,
            "semantic_response_fingerprint",
        )
        .unwrap();
        let conflict = frozen_pending(&mismatch);
        assert_eq!(
            activated
                .repository
                .record_pending_anchor(&conflict)
                .unwrap(),
            AnchorCommandAck::Conflict {
                anchor_id: pending.anchor_id
            }
        );
        assert_eq!(
            activated
                .repository
                .record_pending_anchor(&conflict)
                .unwrap(),
            AnchorCommandAck::Conflict {
                anchor_id: pending.anchor_id
            }
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM health_events
                 WHERE stable_class = 'router.ledger.integrity_conflict'"
            ),
            1
        );
        let stored_hash: String = activated
            .repository
            .connection
            .query_row(
                "SELECT pending_hash FROM anchors WHERE anchor_id = ?1",
                params![pending.anchor_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stored_hash, original.pending_hash());
    }

    #[test]
    fn changed_child_identity_is_a_conflict() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        activated
            .repository
            .record_pending_anchor(&frozen_pending(&pending))
            .unwrap();

        let changed_pending_state = frozen_pending(&pending);
        assert!(matches!(
            activated
                .repository
                .record_pending_anchor(&changed_pending_state)
                .unwrap(),
            AnchorCommandAck::Conflict { .. }
        ));

        let terminal = terminal(pending.clone(), Vec::new());
        let original_terminal = FrozenTerminalAnchorV1::new(
            &terminal,
            Uuid::now_v7(),
            Uuid::now_v7(),
            CREATED_AT + 10_000,
        )
        .unwrap();
        activated
            .repository
            .record_terminal_anchor(&original_terminal)
            .unwrap();
        let changed_terminal_state = FrozenTerminalAnchorV1::new(
            &terminal,
            Uuid::now_v7(),
            Uuid::now_v7(),
            CREATED_AT + 10_000,
        )
        .unwrap();
        assert!(matches!(
            activated
                .repository
                .record_terminal_anchor(&changed_terminal_state)
                .unwrap(),
            AnchorCommandAck::Conflict { .. }
        ));
    }

    #[test]
    fn partial_pending_aggregate_rolls_back_repairs_but_commits_health() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen = frozen_pending(&pending);
        assert!(matches!(
            insert_or_verify_anchor(&activated.repository.connection, &frozen.rows.anchor).unwrap(),
            InsertResult::Status(InsertStatus::Inserted)
        ));

        assert!(matches!(
            activated.repository.record_pending_anchor(&frozen).unwrap(),
            AnchorCommandAck::Conflict { .. }
        ));
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchor_results"
            ),
            0
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchor_state_events"
            ),
            0
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM health_events"
            ),
            1
        );
    }

    #[test]
    fn transaction_start_refusal_opens_no_transaction_or_domain_rows() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen = frozen_pending(&pending);

        assert_eq!(
            activated
                .repository
                .probe_pending_anchor_with_start_check(&frozen, || None::<()>)
                .unwrap(),
            AnchorProbe::TransactionNotStarted {
                anchor_id: pending.anchor_id
            }
        );
        assert_eq!(
            activated
                .repository
                .record_pending_anchor_with_start_check(&frozen, || None::<()>)
                .unwrap(),
            AnchorCommandAck::TransactionNotStarted {
                anchor_id: pending.anchor_id
            }
        );
        assert_eq!(
            activated
                .repository
                .record_pending_anchor_with_capacity_and_start_check(&frozen, 100, || None::<()>,)
                .unwrap(),
            PendingAnchorCapacityAck::TransactionNotStarted {
                anchor_id: pending.anchor_id
            }
        );
        let decline = NotScheduledQueueFull::new(frozen, Uuid::now_v7(), CREATED_AT + 1).unwrap();
        assert_eq!(
            activated
                .repository
                .record_not_scheduled_queue_full_with_capacity_and_start_check(
                    &decline,
                    100,
                    || None::<()>,
                )
                .unwrap(),
            NotScheduledQueueFullAck::TransactionNotStarted {
                anchor_id: pending.anchor_id
            }
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchors"
            ),
            0
        );
    }

    #[test]
    fn project_label_must_match_immutable_project_metadata() {
        let (_temporary, mut activated) = activate();
        let mut pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        pending.project_id = "wrong-project-label".into();
        let frozen = frozen_pending(&pending);

        assert!(activated.repository.record_pending_anchor(&frozen).is_err());
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchors"
            ),
            0
        );
    }

    #[test]
    fn state_event_id_collision_rolls_back_new_anchor_and_health_references_existing_anchor() {
        let (_temporary, mut activated) = activate();
        let collision_id = Uuid::now_v7();
        let first_pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let first =
            FrozenPendingAnchorV1::new(&first_pending, collision_id, Uuid::now_v7(), CREATED_AT)
                .unwrap();
        activated.repository.record_pending_anchor(&first).unwrap();

        let second_pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let second = FrozenPendingAnchorV1::new(
            &second_pending,
            collision_id,
            Uuid::now_v7(),
            CREATED_AT + 1,
        )
        .unwrap();
        assert_eq!(
            activated.repository.record_pending_anchor(&second).unwrap(),
            AnchorCommandAck::Conflict {
                anchor_id: second_pending.anchor_id
            }
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchors"
            ),
            1
        );
        let health_anchor: String = activated
            .repository
            .connection
            .query_row("SELECT anchor_id FROM health_events", [], |row| row.get(0))
            .unwrap();
        assert_eq!(health_anchor, first_pending.anchor_id.to_string());
    }

    #[test]
    fn canonical_hashes_reject_unreproducible_submillisecond_timestamps() {
        let (_temporary, activated) = activate();
        let mut pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        pending.opened_at += chrono::Duration::nanoseconds(1);
        assert!(
            FrozenPendingAnchorV1::new(&pending, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT,)
                .is_err()
        );

        pending.opened_at -= chrono::Duration::nanoseconds(1);
        let mut terminal = terminal(pending, Vec::new());
        terminal.closed_at += chrono::Duration::nanoseconds(1);
        assert!(
            FrozenTerminalAnchorV1::new(
                &terminal,
                Uuid::now_v7(),
                Uuid::now_v7(),
                CREATED_AT + 10_000,
            )
            .is_err()
        );
    }

    #[test]
    fn anchor_state_timestamps_must_follow_the_persisted_window() {
        let (_temporary, activated) = activate();
        let mut pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        pending.opened_at = Utc.timestamp_millis_opt(CREATED_AT + 1).unwrap();
        pending.deadline_at = Utc.timestamp_millis_opt(CREATED_AT + 20_000).unwrap();
        assert!(
            FrozenPendingAnchorV1::new(&pending, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT,)
                .is_err()
        );

        let terminal = PersistedTrajectoryTerminalV1::closed(
            pending,
            Vec::new(),
            1,
            TrajectoryTrigger::ProgressReached,
            Utc.timestamp_millis_opt(CREATED_AT + 10_000).unwrap(),
            Vec::new(),
        );
        assert!(
            FrozenTerminalAnchorV1::new(
                &terminal,
                Uuid::now_v7(),
                Uuid::now_v7(),
                CREATED_AT + 9_999,
            )
            .is_err()
        );
    }

    #[test]
    fn pending_boundary_rejects_noncanonical_projections_and_policy_drift() {
        let (_temporary, mut activated) = activate();
        let base = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());

        let mut stale_request = base.clone();
        stale_request.request_projection.normalized_request.model = Some("different".into());
        assert!(
            FrozenPendingAnchorV1::new(&stale_request, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT,)
                .is_err()
        );

        let mut stale_response = base.clone();
        stale_response.normalized_anchor_response.id = Some("different".into());
        assert!(
            FrozenPendingAnchorV1::new(
                &stale_response,
                Uuid::now_v7(),
                Uuid::now_v7(),
                CREATED_AT,
            )
            .is_err()
        );

        let mut invalid_context = base.clone();
        invalid_context.routing_context_projection.schema = "wrong".into();
        assert!(
            FrozenPendingAnchorV1::new(
                &invalid_context,
                Uuid::now_v7(),
                Uuid::now_v7(),
                CREATED_AT,
            )
            .is_err()
        );

        let mut duplicate_candidate = base.clone();
        duplicate_candidate
            .candidate_facts
            .push(duplicate_candidate.candidate_facts[0].clone());
        assert!(
            FrozenPendingAnchorV1::new(
                &duplicate_candidate,
                Uuid::now_v7(),
                Uuid::now_v7(),
                CREATED_AT,
            )
            .is_err()
        );

        let mut policy_drift = base;
        policy_drift.candidate_facts[0].model = "unconfigured-model".into();
        let policy_drift =
            FrozenPendingAnchorV1::new(&policy_drift, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT)
                .unwrap();
        assert!(
            activated
                .repository
                .record_pending_anchor(&policy_drift)
                .is_err()
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchors"
            ),
            0
        );
    }

    #[test]
    fn closed_horizon_must_match_the_judge_input_contract() {
        let (_temporary, activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let progress_overrun = PersistedTrajectoryTerminalV1::closed(
            pending.clone(),
            Vec::new(),
            2,
            TrajectoryTrigger::ProgressReached,
            Utc.timestamp_millis_opt(CREATED_AT + 10_000).unwrap(),
            Vec::new(),
        );
        assert!(
            FrozenTerminalAnchorV1::new(
                &progress_overrun,
                Uuid::now_v7(),
                Uuid::now_v7(),
                CREATED_AT + 10_000,
            )
            .is_err()
        );

        let partial_at_horizon = PersistedTrajectoryTerminalV1::closed(
            pending,
            Vec::new(),
            1,
            TrajectoryTrigger::DeadlineElapsed,
            Utc.timestamp_millis_opt(CREATED_AT + 10_000).unwrap(),
            Vec::new(),
        );
        assert!(
            FrozenTerminalAnchorV1::new(
                &partial_at_horizon,
                Uuid::now_v7(),
                Uuid::now_v7(),
                CREATED_AT + 10_000,
            )
            .is_err()
        );
    }

    #[test]
    fn stopped_origin_never_receives_an_applied_ack() {
        let (_temporary, mut activated) = activate();
        let pending = pending(&activated.identity, Uuid::now_v7(), Uuid::now_v7());
        let frozen = frozen_pending(&pending);
        let stop = ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), CREATED_AT).unwrap();
        assert_eq!(
            activated.repository.stop_process(stop).unwrap(),
            ProcessCommandAck::Applied
        );
        assert_eq!(
            activated.repository.record_pending_anchor(&frozen).unwrap(),
            AnchorCommandAck::OriginatingProcessNotLive {
                anchor_id: pending.anchor_id
            }
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchors"
            ),
            0
        );
    }

    #[test]
    fn alternate_identity_conflict_is_visible_across_processes() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut first = LedgerRepository::activate(&config(&path, "multi-process")).unwrap();
        let mut second = LedgerRepository::activate(&config(&path, "multi-process")).unwrap();
        let anchor_call_uuid = Uuid::now_v7();
        let first_pending = pending(&first.identity, Uuid::now_v7(), anchor_call_uuid);
        let first_frozen = frozen_pending(&first_pending);
        assert!(matches!(
            first
                .repository
                .record_pending_anchor(&first_frozen)
                .unwrap(),
            AnchorCommandAck::Applied { .. }
        ));

        let second_pending = pending(&second.identity, Uuid::now_v7(), anchor_call_uuid);
        let second_frozen = frozen_pending(&second_pending);
        assert_eq!(
            second
                .repository
                .record_pending_anchor(&second_frozen)
                .unwrap(),
            AnchorCommandAck::Conflict {
                anchor_id: second_pending.anchor_id
            }
        );
        assert_eq!(
            scalar(
                &second.repository.connection,
                "SELECT count(*) FROM anchors"
            ),
            1
        );
        assert!(matches!(
            first
                .repository
                .record_pending_anchor(&first_frozen)
                .unwrap(),
            AnchorCommandAck::AlreadyApplied { .. }
        ));
    }
}
