// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable process liveness and append-only health operations.

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde_json::{Value as Json, json};
use uuid::Uuid;

use super::{
    LedgerRepository, PROCESS_HEARTBEAT_MILLIS, TransactionStartGuard, map_fs_error,
    map_sqlite_error,
};
use crate::canonical_json::canonical_sha256;
use crate::ledger::fs::enforce_sidecar_permissions;
use crate::ledger::model::{LedgerError, LedgerErrorClass};

const INTEGRITY_CONFLICT_CLASS: &str = "router.ledger.integrity_conflict";

/// Immutable request to append this writer process's terminal state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ProcessStop {
    pub(crate) process_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) created_at_unix_ms: i64,
}

impl ProcessStop {
    pub(crate) fn new(
        process_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(process_state_event_id)?;
        validate_uuid_v7(conflict_health_event_id)?;
        if process_state_event_id == conflict_health_event_id || created_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            process_state_event_id,
            conflict_health_event_id,
            created_at_unix_ms,
        })
    }
}

/// Fixed version-1 heartbeat renewal request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct HeartbeatRenewal {
    pub(crate) observed_at_unix_ms: i64,
    pub(crate) expires_at_unix_ms: i64,
}

impl HeartbeatRenewal {
    pub(crate) fn new(observed_at_unix_ms: i64) -> Result<Self, LedgerError> {
        if observed_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let expires_at_unix_ms = observed_at_unix_ms
            .checked_add(PROCESS_HEARTBEAT_MILLIS)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        Ok(Self {
            observed_at_unix_ms,
            expires_at_unix_ms,
        })
    }
}

/// Stable health severity persisted in the private ledger.
#[allow(dead_code)] // Task 10 wires informational and warning inspection health events.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LedgerHealthSeverity {
    Info,
    Warning,
    Degraded,
}

impl LedgerHealthSeverity {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Degraded => "degraded",
        }
    }
}

/// One bounded, non-secret append-only health fact.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LedgerHealthEvent {
    pub(crate) health_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) anchor_id: Option<Uuid>,
    pub(crate) dependency_key_id: Option<String>,
    pub(crate) stable_class: String,
    pub(crate) severity: LedgerHealthSeverity,
    pub(crate) created_at_unix_ms: i64,
}

impl LedgerHealthEvent {
    #[allow(dead_code)] // Task 10 constructs explicit inspection and reconciliation health facts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        health_event_id: Uuid,
        conflict_health_event_id: Uuid,
        anchor_id: Option<Uuid>,
        dependency_key_id: Option<String>,
        stable_class: impl Into<String>,
        severity: LedgerHealthSeverity,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(health_event_id)?;
        validate_uuid_v7(conflict_health_event_id)?;
        if health_event_id == conflict_health_event_id || created_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        if let Some(anchor_id) = anchor_id {
            validate_uuid_v7(anchor_id)?;
        }
        if dependency_key_id
            .as_deref()
            .is_some_and(|value| validate_sha256(value).is_err())
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let stable_class = stable_class.into();
        validate_stable_class(&stable_class)?;
        Ok(Self {
            health_event_id,
            conflict_health_event_id,
            anchor_id,
            dependency_key_id,
            stable_class,
            severity,
            created_at_unix_ms,
        })
    }
}

/// Exhaustive acknowledgement for append-only process and health commands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProcessCommandAck {
    Applied,
    AlreadyApplied,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

/// Monotonic heartbeat result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum HeartbeatAck {
    Applied { expires_at_unix_ms: i64 },
    AlreadyApplied { expires_at_unix_ms: i64 },
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

#[derive(Debug)]
struct StoredProcessInstance {
    process_instance_id: String,
    project_uuid: String,
    config_generation_id: String,
    application_version: String,
    sqlite_version: String,
    started_at_unix_ms: i64,
    heartbeat_expires_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProcessLifecycle {
    Live,
    Terminal,
}

/// Canonically verified process state at one frozen reconciliation timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ProcessStatusAt {
    Live,
    Expired,
    Terminal,
    Invalid,
}

#[derive(Debug)]
struct StoredProcessState {
    process_state_event_id: String,
    process_instance_id: String,
    state: String,
    subject_process_instance_id: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug)]
struct StoredHealthEvent {
    project_uuid: String,
    process_instance_id: Option<String>,
    requested_anchor_id: Option<String>,
    requested_dependency_key_id: Option<String>,
    anchor_id: Option<String>,
    dependency_key_id: Option<String>,
    stable_class: String,
    severity: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

impl LedgerRepository {
    /// Best-effort terminalization used only when activation ownership cannot be delivered.
    pub(crate) fn stop_abandoned_process(&mut self) {
        let created_at_unix_ms = Utc::now().timestamp_millis().max(0);
        if let Ok(command) = ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), created_at_unix_ms) {
            let _ = self.stop_process(command);
        }
    }

    /// Renew only the mutable process lease; no unbounded event is appended.
    #[allow(dead_code)] // Production renewal enters through the sole writer.
    pub(crate) fn renew_heartbeat(
        &mut self,
        renewal: HeartbeatRenewal,
    ) -> Result<HeartbeatAck, LedgerError> {
        self.renew_heartbeat_with_start_check(renewal, || Some(()))
    }

    /// Renew the process lease after atomically retaining caller-owned start authority.
    pub(crate) fn renew_heartbeat_with_start_check<G: TransactionStartGuard>(
        &mut self,
        renewal: HeartbeatRenewal,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<HeartbeatAck, LedgerError> {
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(HeartbeatAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(HeartbeatAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(HeartbeatAck::TransactionNotStarted);
        }
        drop(start_guard);
        let Some(process) =
            load_process_instance(&transaction, self.project_uuid, self.process_instance_id)?
        else {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        };
        if !originating_process_is_live(&transaction, self.project_uuid, self.process_instance_id)?
        {
            return Ok(HeartbeatAck::OriginatingProcessNotLive);
        }
        if renewal.observed_at_unix_ms < process.started_at_unix_ms {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        if process.heartbeat_expires_at_unix_ms >= renewal.expires_at_unix_ms {
            transaction.commit().map_err(database_error)?;
            return Ok(HeartbeatAck::AlreadyApplied {
                expires_at_unix_ms: process.heartbeat_expires_at_unix_ms,
            });
        }
        let payload_hash = process_instance_hash(
            self.process_instance_id,
            self.project_uuid,
            &process.config_generation_id,
            &process.application_version,
            &process.sqlite_version,
            process.started_at_unix_ms,
            renewal.expires_at_unix_ms,
        )?;
        let updated = transaction
            .execute(
                "UPDATE process_instances
                 SET heartbeat_expires_at_unix_ms = ?1, canonical_payload_hash = ?2
                 WHERE project_uuid = ?3 AND process_instance_id = ?4
                   AND heartbeat_expires_at_unix_ms = ?5
                   AND canonical_payload_hash = ?6",
                params![
                    renewal.expires_at_unix_ms,
                    payload_hash,
                    self.project_uuid.to_string(),
                    self.process_instance_id.to_string(),
                    process.heartbeat_expires_at_unix_ms,
                    process.canonical_payload_hash,
                ],
            )
            .map_err(database_error)?;
        if updated != 1 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(HeartbeatAck::Applied {
            expires_at_unix_ms: renewal.expires_at_unix_ms,
        })
    }

    /// Append this process's terminal state with exact retry semantics.
    pub(crate) fn stop_process(
        &mut self,
        command: ProcessStop,
    ) -> Result<ProcessCommandAck, LedgerError> {
        self.stop_process_with_start_check(command, || Some(()))
    }

    /// Append process stop after atomically retaining caller-owned start authority.
    pub(crate) fn stop_process_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: ProcessStop,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ProcessCommandAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ProcessCommandAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(ProcessCommandAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ProcessCommandAck::TransactionNotStarted);
        }
        drop(start_guard);
        let expected_hash = process_stop_hash(process_instance_id, command)?;

        if let Some(stored) = load_process_state(&transaction, command.process_state_event_id)? {
            let acknowledgement =
                if process_stop_matches(
                    &stored,
                    command.process_state_event_id,
                    process_instance_id,
                    command.created_at_unix_ms,
                    &expected_hash,
                ) && verified_process_lifecycle(&transaction, project_uuid, process_instance_id)?
                    == Some(ProcessLifecycle::Terminal)
                {
                    ProcessCommandAck::AlreadyApplied
                } else {
                    append_integrity_health(
                        &transaction,
                        command.conflict_health_event_id,
                        project_uuid,
                        process_instance_id,
                        None,
                        None,
                        command.created_at_unix_ms,
                    )?;
                    ProcessCommandAck::Conflict
                };
            transaction.commit().map_err(database_error)?;
            return Ok(acknowledgement);
        }

        if load_terminal_process_state(&transaction, process_instance_id)?.is_some() {
            append_integrity_health(
                &transaction,
                command.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                None,
                None,
                command.created_at_unix_ms,
            )?;
            transaction.commit().map_err(database_error)?;
            return Ok(ProcessCommandAck::Conflict);
        }
        if !originating_process_is_live(&transaction, project_uuid, process_instance_id)? {
            return Ok(ProcessCommandAck::OriginatingProcessNotLive);
        }
        transaction
            .execute(
                "INSERT INTO process_instance_state_events (
                    process_state_event_id, process_instance_id, state,
                    subject_process_instance_id, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, 'stopped', ?2, ?3, ?4)",
                params![
                    command.process_state_event_id.to_string(),
                    process_instance_id.to_string(),
                    command.created_at_unix_ms,
                    expected_hash,
                ],
            )
            .map_err(database_error)?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(ProcessCommandAck::Applied)
    }

    /// Append one stable health fact and degrade on a mismatched duplicate key.
    #[allow(dead_code)] // Task 10 wires explicit runtime health recording through the writer.
    pub(crate) fn append_health_event(
        &mut self,
        event: &LedgerHealthEvent,
    ) -> Result<ProcessCommandAck, LedgerError> {
        self.append_health_event_with_start_check(event, || Some(()))
    }

    /// Append health after atomically retaining caller-owned start authority.
    pub(crate) fn append_health_event_with_start_check<G: TransactionStartGuard>(
        &mut self,
        event: &LedgerHealthEvent,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ProcessCommandAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ProcessCommandAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(ProcessCommandAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ProcessCommandAck::TransactionNotStarted);
        }
        drop(start_guard);
        if !originating_process_is_live(&transaction, project_uuid, process_instance_id)? {
            return Ok(ProcessCommandAck::OriginatingProcessNotLive);
        }
        let payload_hash = health_event_hash(project_uuid, process_instance_id, event)?;
        if let Some(stored) = load_health_event(&transaction, event.health_event_id)? {
            let acknowledgement = if health_event_matches(
                &stored,
                project_uuid,
                process_instance_id,
                event,
                &payload_hash,
            ) {
                ProcessCommandAck::AlreadyApplied
            } else {
                append_integrity_health(
                    &transaction,
                    event.conflict_health_event_id,
                    project_uuid,
                    process_instance_id,
                    event.anchor_id,
                    event.dependency_key_id.as_deref(),
                    event.created_at_unix_ms,
                )?;
                ProcessCommandAck::Conflict
            };
            transaction.commit().map_err(database_error)?;
            return Ok(acknowledgement);
        }
        transaction
            .execute(
                "INSERT INTO health_events (
                    health_event_id, project_uuid, process_instance_id,
                    requested_anchor_id, requested_dependency_key_id,
                    anchor_id, dependency_key_id, stable_class, severity,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    event.health_event_id.to_string(),
                    project_uuid.to_string(),
                    process_instance_id.to_string(),
                    event.anchor_id.map(|value| value.to_string()),
                    event.dependency_key_id,
                    event.stable_class,
                    event.severity.as_str(),
                    event.created_at_unix_ms,
                    payload_hash,
                ],
            )
            .map_err(database_error)?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(ProcessCommandAck::Applied)
    }
}

fn load_process_instance(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
) -> Result<Option<StoredProcessInstance>, LedgerError> {
    connection
        .query_row(
            "SELECT process_instance_id, project_uuid, config_generation_id,
                    application_version, sqlite_version, started_at_unix_ms,
                    heartbeat_expires_at_unix_ms, canonical_payload_hash
             FROM process_instances
             WHERE project_uuid = ?1 AND process_instance_id = ?2",
            params![project_uuid.to_string(), process_instance_id.to_string()],
            |row| {
                Ok(StoredProcessInstance {
                    process_instance_id: row.get(0)?,
                    project_uuid: row.get(1)?,
                    config_generation_id: row.get(2)?,
                    application_version: row.get(3)?,
                    sqlite_version: row.get(4)?,
                    started_at_unix_ms: row.get(5)?,
                    heartbeat_expires_at_unix_ms: row.get(6)?,
                    canonical_payload_hash: row.get(7)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn load_process_state(
    connection: &Connection,
    event_id: Uuid,
) -> Result<Option<StoredProcessState>, LedgerError> {
    connection
        .query_row(
            "SELECT process_state_event_id, process_instance_id, state,
                    subject_process_instance_id,
                    created_at_unix_ms, canonical_payload_hash
             FROM process_instance_state_events
             WHERE process_state_event_id = ?1",
            params![event_id.to_string()],
            |row| {
                Ok(StoredProcessState {
                    process_state_event_id: row.get(0)?,
                    process_instance_id: row.get(1)?,
                    state: row.get(2)?,
                    subject_process_instance_id: row.get(3)?,
                    created_at_unix_ms: row.get(4)?,
                    canonical_payload_hash: row.get(5)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn load_terminal_process_state(
    connection: &Connection,
    process_instance_id: Uuid,
) -> Result<Option<StoredProcessState>, LedgerError> {
    connection
        .query_row(
            "SELECT process_state_event_id, process_instance_id, state,
                    subject_process_instance_id, created_at_unix_ms,
                    canonical_payload_hash
             FROM process_instance_state_events
             WHERE subject_process_instance_id = ?1
               AND state IN ('stopped', 'reconciled')",
            params![process_instance_id.to_string()],
            |row| {
                Ok(StoredProcessState {
                    process_state_event_id: row.get(0)?,
                    process_instance_id: row.get(1)?,
                    state: row.get(2)?,
                    subject_process_instance_id: row.get(3)?,
                    created_at_unix_ms: row.get(4)?,
                    canonical_payload_hash: row.get(5)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn load_started_process_state(
    connection: &Connection,
    process_instance_id: Uuid,
) -> Result<Option<StoredProcessState>, LedgerError> {
    connection
        .query_row(
            "SELECT process_state_event_id, process_instance_id, state,
                    subject_process_instance_id, created_at_unix_ms,
                    canonical_payload_hash
             FROM process_instance_state_events
             WHERE subject_process_instance_id = ?1 AND state = 'started'",
            params![process_instance_id.to_string()],
            |row| {
                Ok(StoredProcessState {
                    process_state_event_id: row.get(0)?,
                    process_instance_id: row.get(1)?,
                    state: row.get(2)?,
                    subject_process_instance_id: row.get(3)?,
                    created_at_unix_ms: row.get(4)?,
                    canonical_payload_hash: row.get(5)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn process_stop_matches(
    stored: &StoredProcessState,
    process_state_event_id: Uuid,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
    expected_hash: &str,
) -> bool {
    let process_instance_id = process_instance_id.to_string();
    stored.process_state_event_id == process_state_event_id.to_string()
        && stored.process_instance_id == process_instance_id
        && stored.state == "stopped"
        && stored.subject_process_instance_id == process_instance_id
        && stored.created_at_unix_ms == created_at_unix_ms
        && stored.canonical_payload_hash == expected_hash
}

fn load_health_event(
    connection: &Connection,
    event_id: Uuid,
) -> Result<Option<StoredHealthEvent>, LedgerError> {
    connection
        .query_row(
            "SELECT project_uuid, process_instance_id, requested_anchor_id,
                    requested_dependency_key_id, anchor_id, dependency_key_id, stable_class, severity,
                    created_at_unix_ms, canonical_payload_hash
             FROM health_events WHERE health_event_id = ?1",
            params![event_id.to_string()],
            |row| {
                Ok(StoredHealthEvent {
                    project_uuid: row.get(0)?,
                    process_instance_id: row.get(1)?,
                    requested_anchor_id: row.get(2)?,
                    requested_dependency_key_id: row.get(3)?,
                    anchor_id: row.get(4)?,
                    dependency_key_id: row.get(5)?,
                    stable_class: row.get(6)?,
                    severity: row.get(7)?,
                    created_at_unix_ms: row.get(8)?,
                    canonical_payload_hash: row.get(9)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn health_event_matches(
    stored: &StoredHealthEvent,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    event: &LedgerHealthEvent,
    expected_hash: &str,
) -> bool {
    stored.project_uuid == project_uuid.to_string()
        && stored.process_instance_id.as_deref() == Some(process_instance_id.to_string().as_str())
        && stored.requested_anchor_id == event.anchor_id.map(|value| value.to_string())
        && stored.requested_dependency_key_id == event.dependency_key_id
        && stored.anchor_id == event.anchor_id.map(|value| value.to_string())
        && stored.dependency_key_id == event.dependency_key_id
        && stored.stable_class == event.stable_class
        && stored.severity == event.severity.as_str()
        && stored.created_at_unix_ms == event.created_at_unix_ms
        && stored.canonical_payload_hash == expected_hash
}

pub(super) fn originating_process_is_live(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
) -> Result<bool, LedgerError> {
    Ok(
        verified_process_lifecycle(connection, project_uuid, process_instance_id)?
            == Some(ProcessLifecycle::Live),
    )
}

/// Verify both append-only lifecycle and mutable heartbeat state at one instant.
pub(super) fn verified_process_status_at(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    observed_at_unix_ms: i64,
) -> Result<ProcessStatusAt, LedgerError> {
    if observed_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let Some(process) = load_process_instance(connection, project_uuid, process_instance_id)?
    else {
        return Ok(ProcessStatusAt::Invalid);
    };
    if observed_at_unix_ms < process.started_at_unix_ms {
        return Ok(ProcessStatusAt::Invalid);
    }
    match verified_process_lifecycle(connection, project_uuid, process_instance_id)? {
        Some(ProcessLifecycle::Terminal) => Ok(ProcessStatusAt::Terminal),
        Some(ProcessLifecycle::Live)
            if process.heartbeat_expires_at_unix_ms <= observed_at_unix_ms =>
        {
            Ok(ProcessStatusAt::Expired)
        }
        Some(ProcessLifecycle::Live) => Ok(ProcessStatusAt::Live),
        None => Ok(ProcessStatusAt::Invalid),
    }
}

/// Refresh the process created by activation after potentially long recovery work.
pub(super) fn refresh_activation_heartbeat(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    observed_at_unix_ms: i64,
) -> Result<i64, LedgerError> {
    let process = load_process_instance(connection, project_uuid, process_instance_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if observed_at_unix_ms < process.started_at_unix_ms
        || !stored_process_instance_is_canonical(&process, project_uuid, process_instance_id)?
        || verified_process_lifecycle(connection, project_uuid, process_instance_id)?
            != Some(ProcessLifecycle::Live)
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let expires_at_unix_ms = observed_at_unix_ms
        .checked_add(PROCESS_HEARTBEAT_MILLIS)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if expires_at_unix_ms < process.heartbeat_expires_at_unix_ms {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    if expires_at_unix_ms == process.heartbeat_expires_at_unix_ms {
        return Ok(expires_at_unix_ms);
    }
    let payload_hash = process_instance_hash(
        process_instance_id,
        project_uuid,
        &process.config_generation_id,
        &process.application_version,
        &process.sqlite_version,
        process.started_at_unix_ms,
        expires_at_unix_ms,
    )?;
    let updated = connection
        .execute(
            "UPDATE process_instances
             SET heartbeat_expires_at_unix_ms = ?1, canonical_payload_hash = ?2
             WHERE project_uuid = ?3 AND process_instance_id = ?4
               AND heartbeat_expires_at_unix_ms = ?5
               AND canonical_payload_hash = ?6",
            params![
                expires_at_unix_ms,
                payload_hash,
                project_uuid.to_string(),
                process_instance_id.to_string(),
                process.heartbeat_expires_at_unix_ms,
                process.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if updated != 1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(expires_at_unix_ms)
}

/// Fence one expired process before any of its non-resumable work is orphaned.
pub(super) fn fence_expired_process(
    connection: &Connection,
    project_uuid: Uuid,
    reconciler_process_instance_id: Uuid,
    expired_process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<bool, LedgerError> {
    if reconciler_process_instance_id == expired_process_instance_id
        || verified_process_status_at(
            connection,
            project_uuid,
            reconciler_process_instance_id,
            created_at_unix_ms,
        )? != ProcessStatusAt::Live
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    match verified_process_status_at(
        connection,
        project_uuid,
        expired_process_instance_id,
        created_at_unix_ms,
    )? {
        ProcessStatusAt::Terminal => return Ok(false),
        ProcessStatusAt::Live | ProcessStatusAt::Invalid => {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        ProcessStatusAt::Expired => {}
    }

    let event_id = Uuid::now_v7();
    let payload_hash = process_state_hash(
        event_id,
        reconciler_process_instance_id,
        "reconciled",
        expired_process_instance_id,
        created_at_unix_ms,
    )?;
    let inserted = connection
        .execute(
            "INSERT INTO process_instance_state_events (
                process_state_event_id, process_instance_id, state,
                subject_process_instance_id, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, 'reconciled', ?3, ?4, ?5)",
            params![
                event_id.to_string(),
                reconciler_process_instance_id.to_string(),
                expired_process_instance_id.to_string(),
                created_at_unix_ms,
                payload_hash,
            ],
        )
        .map_err(database_error)?;
    if inserted != 1
        || verified_process_status_at(
            connection,
            project_uuid,
            expired_process_instance_id,
            created_at_unix_ms,
        )? != ProcessStatusAt::Terminal
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(true)
}

fn verified_process_lifecycle(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
) -> Result<Option<ProcessLifecycle>, LedgerError> {
    let Some(process) = load_process_instance(connection, project_uuid, process_instance_id)?
    else {
        return Ok(None);
    };
    if !stored_process_instance_is_canonical(&process, project_uuid, process_instance_id)? {
        return Ok(None);
    }

    let Some(started) = load_started_process_state(connection, process_instance_id)? else {
        return Ok(None);
    };
    if !stored_process_state_is_canonical(
        &started,
        "started",
        process_instance_id,
        process_instance_id,
        Some(process.started_at_unix_ms),
    )? {
        return Ok(None);
    }

    let Some(terminal) = load_terminal_process_state(connection, process_instance_id)? else {
        return Ok(Some(ProcessLifecycle::Live));
    };
    let Some(actor_process_instance_id) = parse_uuid_v7(&terminal.process_instance_id) else {
        return Ok(None);
    };
    let terminal_semantics_match = match terminal.state.as_str() {
        "stopped" => actor_process_instance_id == process_instance_id,
        "reconciled" => {
            actor_process_instance_id != process_instance_id
                && process_belongs_to_project(connection, project_uuid, actor_process_instance_id)?
        }
        _ => false,
    };
    if !terminal_semantics_match
        || !stored_process_state_is_canonical(
            &terminal,
            terminal.state.as_str(),
            actor_process_instance_id,
            process_instance_id,
            None,
        )?
    {
        return Ok(None);
    }
    Ok(Some(ProcessLifecycle::Terminal))
}

fn stored_process_instance_is_canonical(
    stored: &StoredProcessInstance,
    project_uuid: Uuid,
    process_instance_id: Uuid,
) -> Result<bool, LedgerError> {
    let Some(initial_expiry) = stored
        .started_at_unix_ms
        .checked_add(PROCESS_HEARTBEAT_MILLIS)
    else {
        return Ok(false);
    };
    if stored.process_instance_id != process_instance_id.to_string()
        || stored.project_uuid != project_uuid.to_string()
        || validate_sha256(&stored.config_generation_id).is_err()
        || stored.application_version.is_empty()
        || stored.sqlite_version.is_empty()
        || stored.started_at_unix_ms < 0
        || stored.heartbeat_expires_at_unix_ms < initial_expiry
    {
        return Ok(false);
    }
    let expected_hash = process_instance_hash(
        process_instance_id,
        project_uuid,
        &stored.config_generation_id,
        &stored.application_version,
        &stored.sqlite_version,
        stored.started_at_unix_ms,
        stored.heartbeat_expires_at_unix_ms,
    )?;
    Ok(stored.canonical_payload_hash == expected_hash)
}

fn stored_process_state_is_canonical(
    stored: &StoredProcessState,
    expected_state: &str,
    process_instance_id: Uuid,
    subject_process_instance_id: Uuid,
    expected_created_at_unix_ms: Option<i64>,
) -> Result<bool, LedgerError> {
    let Some(process_state_event_id) = parse_uuid_v7(&stored.process_state_event_id) else {
        return Ok(false);
    };
    if stored.process_instance_id != process_instance_id.to_string()
        || stored.state != expected_state
        || stored.subject_process_instance_id != subject_process_instance_id.to_string()
        || stored.created_at_unix_ms < 0
        || expected_created_at_unix_ms.is_some_and(|expected| stored.created_at_unix_ms != expected)
    {
        return Ok(false);
    }
    let expected_hash = process_state_hash(
        process_state_event_id,
        process_instance_id,
        expected_state,
        subject_process_instance_id,
        stored.created_at_unix_ms,
    )?;
    Ok(stored.canonical_payload_hash == expected_hash)
}

fn process_belongs_to_project(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM process_instances
                WHERE project_uuid = ?1 AND process_instance_id = ?2
             )",
            params![project_uuid.to_string(), process_instance_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)
}

pub(super) fn append_integrity_health(
    connection: &Connection,
    health_event_id: Uuid,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    anchor_id: Option<Uuid>,
    dependency_key_id: Option<&str>,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let requested_anchor_id = anchor_id.map(|value| value.to_string());
    let requested_dependency_key_id = dependency_key_id;
    if let Some(stored) = load_health_event(connection, health_event_id)? {
        let stored_hash = health_payload_hash(
            health_event_id,
            project_uuid,
            process_instance_id,
            stored.requested_anchor_id.as_deref(),
            stored.requested_dependency_key_id.as_deref(),
            stored.anchor_id.as_deref(),
            stored.dependency_key_id.as_deref(),
            INTEGRITY_CONFLICT_CLASS,
            LedgerHealthSeverity::Degraded.as_str(),
            created_at_unix_ms,
        )?;
        if stored.project_uuid != project_uuid.to_string()
            || stored.process_instance_id.as_deref()
                != Some(process_instance_id.to_string().as_str())
            || stored.requested_anchor_id.as_deref() != requested_anchor_id.as_deref()
            || stored.requested_dependency_key_id.as_deref() != requested_dependency_key_id
            || stored
                .anchor_id
                .as_ref()
                .is_some_and(|stored| Some(stored) != requested_anchor_id.as_ref())
            || stored
                .dependency_key_id
                .as_deref()
                .is_some_and(|stored| Some(stored) != requested_dependency_key_id)
            || stored.stable_class != INTEGRITY_CONFLICT_CLASS
            || stored.severity != LedgerHealthSeverity::Degraded.as_str()
            || stored.created_at_unix_ms != created_at_unix_ms
            || stored.canonical_payload_hash != stored_hash
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        return Ok(());
    }

    let anchor_id = resolve_health_anchor(connection, project_uuid, anchor_id)?;
    let dependency_key_id = resolve_health_dependency(connection, project_uuid, dependency_key_id)?;
    let anchor_id = anchor_id.map(|value| value.to_string());
    let payload_hash = health_payload_hash(
        health_event_id,
        project_uuid,
        process_instance_id,
        requested_anchor_id.as_deref(),
        requested_dependency_key_id,
        anchor_id.as_deref(),
        dependency_key_id.as_deref(),
        INTEGRITY_CONFLICT_CLASS,
        LedgerHealthSeverity::Degraded.as_str(),
        created_at_unix_ms,
    )?;
    let inserted = connection
        .execute(
            "INSERT INTO health_events (
                health_event_id, project_uuid, process_instance_id,
                requested_anchor_id, requested_dependency_key_id,
                anchor_id, dependency_key_id, stable_class, severity,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 'degraded', ?9, ?10)
             ON CONFLICT(health_event_id) DO NOTHING",
            params![
                health_event_id.to_string(),
                project_uuid.to_string(),
                process_instance_id.to_string(),
                requested_anchor_id,
                requested_dependency_key_id,
                anchor_id,
                dependency_key_id,
                INTEGRITY_CONFLICT_CLASS,
                created_at_unix_ms,
                payload_hash,
            ],
        )
        .map_err(database_error)?;
    if inserted != 1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn health_payload_hash(
    health_event_id: Uuid,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    requested_anchor_id: Option<&str>,
    requested_dependency_key_id: Option<&str>,
    resolved_anchor_id: Option<&str>,
    resolved_dependency_key_id: Option<&str>,
    stable_class: &str,
    severity: &str,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "health_event_id": health_event_id,
        "project_uuid": project_uuid,
        "process_instance_id": process_instance_id,
        "requested_anchor_id": requested_anchor_id,
        "requested_dependency_key_id": requested_dependency_key_id,
        "anchor_id": resolved_anchor_id,
        "dependency_key_id": resolved_dependency_key_id,
        "stable_class": stable_class,
        "severity": severity,
        "created_at_unix_ms": created_at_unix_ms,
    }))
}

fn resolve_health_anchor(
    connection: &Connection,
    project_uuid: Uuid,
    anchor_id: Option<Uuid>,
) -> Result<Option<Uuid>, LedgerError> {
    let Some(anchor_id) = anchor_id else {
        return Ok(None);
    };
    let exists = connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM anchors WHERE anchor_id = ?1 AND project_uuid = ?2
             )",
            params![anchor_id.to_string(), project_uuid.to_string()],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)?;
    Ok(exists.then_some(anchor_id))
}

fn resolve_health_dependency(
    connection: &Connection,
    project_uuid: Uuid,
    dependency_key_id: Option<&str>,
) -> Result<Option<String>, LedgerError> {
    let Some(dependency_key_id) = dependency_key_id else {
        return Ok(None);
    };
    let exists = connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM dependency_keys
                WHERE dependency_key_id = ?1 AND project_uuid = ?2
             )",
            params![dependency_key_id, project_uuid.to_string()],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)?;
    Ok(exists.then(|| dependency_key_id.to_string()))
}

fn process_stop_hash(
    process_instance_id: Uuid,
    command: ProcessStop,
) -> Result<String, LedgerError> {
    process_state_hash(
        command.process_state_event_id,
        process_instance_id,
        "stopped",
        process_instance_id,
        command.created_at_unix_ms,
    )
}

#[allow(clippy::too_many_arguments)]
fn process_instance_hash(
    process_instance_id: Uuid,
    project_uuid: Uuid,
    config_generation_id: &str,
    application_version: &str,
    sqlite_version: &str,
    started_at_unix_ms: i64,
    heartbeat_expires_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "process_instance_id": process_instance_id,
        "project_uuid": project_uuid,
        "config_generation_id": config_generation_id,
        "application_version": application_version,
        "sqlite_version": sqlite_version,
        "started_at_unix_ms": started_at_unix_ms,
        "heartbeat_expires_at_unix_ms": heartbeat_expires_at_unix_ms,
    }))
}

fn process_state_hash(
    process_state_event_id: Uuid,
    process_instance_id: Uuid,
    state: &str,
    subject_process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "process_state_event_id": process_state_event_id,
        "process_instance_id": process_instance_id,
        "state": state,
        "subject_process_instance_id": subject_process_instance_id,
        "created_at_unix_ms": created_at_unix_ms,
    }))
}

fn health_event_hash(
    project_uuid: Uuid,
    process_instance_id: Uuid,
    event: &LedgerHealthEvent,
) -> Result<String, LedgerError> {
    let anchor_id = event.anchor_id.map(|value| value.to_string());
    health_payload_hash(
        event.health_event_id,
        project_uuid,
        process_instance_id,
        anchor_id.as_deref(),
        event.dependency_key_id.as_deref(),
        anchor_id.as_deref(),
        event.dependency_key_id.as_deref(),
        &event.stable_class,
        event.severity.as_str(),
        event.created_at_unix_ms,
    )
}

#[allow(dead_code)] // Task 10 explicit health construction consumes this validator.
fn validate_stable_class(value: &str) -> Result<(), LedgerError> {
    let accepted = matches!(
        value,
        "router.dependency.cooloff_active" | "router.dependency.recovered"
    );
    #[cfg(test)]
    let accepted = accepted || matches!(value, "router.writer.test" | "router.writer.contention");
    if !accepted {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), LedgerError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
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

fn parse_uuid_v7(value: &str) -> Option<Uuid> {
    let parsed = Uuid::parse_str(value).ok()?;
    (parsed.get_version_num() == 7 && value == parsed.to_string()).then_some(parsed)
}

fn hash_json(value: &Json) -> Result<String, LedgerError> {
    canonical_sha256(value).map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn database_error(error: rusqlite::Error) -> LedgerError {
    map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use rusqlite::params;
    use serde_json::json;
    use tempfile::tempdir;
    use uuid::Uuid;

    use super::{
        HeartbeatAck, HeartbeatRenewal, LedgerHealthEvent, LedgerHealthSeverity, ProcessCommandAck,
        ProcessLifecycle, ProcessStatusAt, ProcessStop, append_integrity_health,
        originating_process_is_live, process_instance_hash, process_state_hash,
        verified_process_lifecycle, verified_process_status_at,
    };
    use crate::config::RouterConfig;
    use crate::ledger::model::LedgerErrorClass;
    use crate::ledger::repository::LedgerRepository;

    const NOW: i64 = 2_000_000_000_000;

    fn config(path: &Path) -> RouterConfig {
        serde_json::from_value(json!({
            "version": 1,
            "mode": "shadow",
            "project_id": "process-project",
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

    fn repository() -> (tempfile::TempDir, LedgerRepository) {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let activated =
            LedgerRepository::activate(&config(&temporary.path().join("ledger.db"))).unwrap();
        (temporary, activated.repository)
    }

    #[test]
    fn heartbeat_is_monotonic_and_stop_is_exactly_idempotent() {
        let (_temporary, mut repository) = repository();
        let renewal = HeartbeatRenewal::new(NOW).unwrap();
        assert_eq!(
            repository.renew_heartbeat(renewal).unwrap(),
            HeartbeatAck::Applied {
                expires_at_unix_ms: NOW + 30_000,
            }
        );
        assert_eq!(
            repository.renew_heartbeat(renewal).unwrap(),
            HeartbeatAck::AlreadyApplied {
                expires_at_unix_ms: NOW + 30_000,
            }
        );
        assert!(
            originating_process_is_live(
                &repository.connection,
                repository.project_uuid,
                repository.process_instance_id,
            )
            .unwrap()
        );
        let (config_generation_id, application_version, sqlite_version, started_at, stored_hash): (
            String,
            String,
            String,
            i64,
            String,
        ) = repository
            .connection
            .query_row(
                "SELECT config_generation_id, application_version, sqlite_version,
                        started_at_unix_ms, canonical_payload_hash
                 FROM process_instances WHERE process_instance_id = ?1",
                [repository.process_instance_id.to_string()],
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
        assert_eq!(
            stored_hash,
            process_instance_hash(
                repository.process_instance_id,
                repository.project_uuid,
                &config_generation_id,
                &application_version,
                &sqlite_version,
                started_at,
                NOW + 30_000,
            )
            .unwrap()
        );

        let stop = ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), NOW + 1).unwrap();
        assert_eq!(
            repository.stop_process(stop).unwrap(),
            ProcessCommandAck::Applied
        );
        assert_eq!(
            repository.stop_process(stop).unwrap(),
            ProcessCommandAck::AlreadyApplied
        );
        assert_eq!(
            repository
                .renew_heartbeat(HeartbeatRenewal::new(NOW + 2).unwrap())
                .unwrap(),
            HeartbeatAck::OriginatingProcessNotLive
        );
    }

    #[test]
    fn timestamped_process_status_rejects_observation_before_start() {
        let (_temporary, mut repository) = repository();
        let project_uuid = repository.project_uuid;
        let process_instance_id = repository.process_instance_id;
        let started_at_unix_ms: i64 = repository
            .connection
            .query_row(
                "SELECT started_at_unix_ms FROM process_instances
                 WHERE process_instance_id = ?1",
                [process_instance_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();

        assert_eq!(
            verified_process_status_at(
                &repository.connection,
                project_uuid,
                process_instance_id,
                started_at_unix_ms - 1,
            )
            .unwrap(),
            ProcessStatusAt::Invalid
        );
        assert_eq!(
            verified_process_status_at(
                &repository.connection,
                project_uuid,
                process_instance_id,
                started_at_unix_ms,
            )
            .unwrap(),
            ProcessStatusAt::Live
        );

        let stop = ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), started_at_unix_ms).unwrap();
        assert_eq!(
            repository.stop_process(stop).unwrap(),
            ProcessCommandAck::Applied
        );
        assert_eq!(
            verified_process_status_at(
                &repository.connection,
                project_uuid,
                process_instance_id,
                started_at_unix_ms - 1,
            )
            .unwrap(),
            ProcessStatusAt::Invalid
        );
    }

    #[test]
    fn liveness_requires_canonical_process_start_and_heartbeat_aggregate() {
        let (_temporary, mut repository) = repository();
        let project_uuid = repository.project_uuid;
        let process_instance_id = repository.process_instance_id;
        assert!(
            originating_process_is_live(&repository.connection, project_uuid, process_instance_id)
                .unwrap()
        );

        let original_process_hash: String = repository
            .connection
            .query_row(
                "SELECT canonical_payload_hash FROM process_instances
                 WHERE process_instance_id = ?1",
                [process_instance_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        repository
            .connection
            .execute(
                "UPDATE process_instances SET canonical_payload_hash = ?1
                 WHERE process_instance_id = ?2",
                params!["a".repeat(64), process_instance_id.to_string()],
            )
            .unwrap();
        assert!(
            !originating_process_is_live(
                &repository.connection,
                project_uuid,
                process_instance_id,
            )
            .unwrap()
        );
        repository
            .connection
            .execute(
                "UPDATE process_instances SET canonical_payload_hash = ?1
                 WHERE process_instance_id = ?2",
                params![original_process_hash, process_instance_id.to_string()],
            )
            .unwrap();

        let (start_event_id, started_at, original_start_hash): (String, i64, String) = repository
            .connection
            .query_row(
                "SELECT process_state_event_id, created_at_unix_ms, canonical_payload_hash
                 FROM process_instance_state_events
                 WHERE subject_process_instance_id = ?1 AND state = 'started'",
                [process_instance_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        repository
            .connection
            .execute(
                "UPDATE process_instance_state_events SET canonical_payload_hash = ?1
                 WHERE process_state_event_id = ?2",
                params!["b".repeat(64), start_event_id],
            )
            .unwrap();
        assert!(
            !originating_process_is_live(
                &repository.connection,
                project_uuid,
                process_instance_id,
            )
            .unwrap()
        );
        repository
            .connection
            .execute(
                "UPDATE process_instance_state_events SET canonical_payload_hash = ?1
                 WHERE process_state_event_id = ?2",
                params![&original_start_hash, &start_event_id],
            )
            .unwrap();
        let start_event_uuid = Uuid::parse_str(&start_event_id).unwrap();
        let mismatched_start_time = started_at + 1;
        let mismatched_start_hash = process_state_hash(
            start_event_uuid,
            process_instance_id,
            "started",
            process_instance_id,
            mismatched_start_time,
        )
        .unwrap();
        repository
            .connection
            .execute(
                "UPDATE process_instance_state_events
                 SET created_at_unix_ms = ?1, canonical_payload_hash = ?2
                 WHERE process_state_event_id = ?3",
                params![
                    mismatched_start_time,
                    mismatched_start_hash,
                    &start_event_id
                ],
            )
            .unwrap();
        assert!(
            !originating_process_is_live(
                &repository.connection,
                project_uuid,
                process_instance_id,
            )
            .unwrap()
        );
        repository
            .connection
            .execute(
                "UPDATE process_instance_state_events
                 SET created_at_unix_ms = ?1, canonical_payload_hash = ?2
                 WHERE process_state_event_id = ?3",
                params![started_at, original_start_hash, start_event_id],
            )
            .unwrap();

        let (config_generation_id, application_version, sqlite_version): (String, String, String) =
            repository
                .connection
                .query_row(
                    "SELECT config_generation_id, application_version, sqlite_version
                 FROM process_instances WHERE process_instance_id = ?1",
                    [process_instance_id.to_string()],
                    |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
                )
                .unwrap();
        let invalid_expiry = started_at + 29_999;
        let invalid_but_exact_hash = process_instance_hash(
            process_instance_id,
            project_uuid,
            &config_generation_id,
            &application_version,
            &sqlite_version,
            started_at,
            invalid_expiry,
        )
        .unwrap();
        repository
            .connection
            .execute(
                "UPDATE process_instances
                 SET heartbeat_expires_at_unix_ms = ?1, canonical_payload_hash = ?2
                 WHERE process_instance_id = ?3",
                params![
                    invalid_expiry,
                    invalid_but_exact_hash,
                    process_instance_id.to_string()
                ],
            )
            .unwrap();
        assert!(
            !originating_process_is_live(
                &repository.connection,
                project_uuid,
                process_instance_id,
            )
            .unwrap()
        );
        assert_eq!(
            repository
                .renew_heartbeat(HeartbeatRenewal::new(NOW).unwrap())
                .unwrap(),
            HeartbeatAck::OriginatingProcessNotLive
        );
    }

    #[test]
    fn terminal_lifecycle_requires_the_full_canonical_event() {
        let (_temporary, mut repository) = repository();
        let project_uuid = repository.project_uuid;
        let process_instance_id = repository.process_instance_id;
        let stop = ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), NOW).unwrap();
        assert_eq!(
            repository.stop_process(stop).unwrap(),
            ProcessCommandAck::Applied
        );
        assert_eq!(
            verified_process_lifecycle(&repository.connection, project_uuid, process_instance_id,)
                .unwrap(),
            Some(ProcessLifecycle::Terminal)
        );

        repository
            .connection
            .execute(
                "UPDATE process_instance_state_events SET canonical_payload_hash = ?1
                 WHERE process_state_event_id = ?2",
                params!["c".repeat(64), stop.process_state_event_id.to_string()],
            )
            .unwrap();
        assert_eq!(
            verified_process_lifecycle(&repository.connection, project_uuid, process_instance_id,)
                .unwrap(),
            None
        );
        assert!(
            !originating_process_is_live(
                &repository.connection,
                project_uuid,
                process_instance_id,
            )
            .unwrap()
        );

        let exact_hash = process_state_hash(
            stop.process_state_event_id,
            process_instance_id,
            "stopped",
            process_instance_id,
            stop.created_at_unix_ms,
        )
        .unwrap();
        repository
            .connection
            .execute(
                "UPDATE process_instance_state_events SET canonical_payload_hash = ?1
                 WHERE process_state_event_id = ?2",
                params![exact_hash, stop.process_state_event_id.to_string()],
            )
            .unwrap();
        assert_eq!(
            repository.stop_process(stop).unwrap(),
            ProcessCommandAck::AlreadyApplied
        );
    }

    #[test]
    fn a_different_stop_identity_never_overwrites_and_degrades_health() {
        let (_temporary, mut repository) = repository();
        let first = ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), NOW).unwrap();
        assert_eq!(
            repository.stop_process(first).unwrap(),
            ProcessCommandAck::Applied
        );
        let second = ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), NOW + 1).unwrap();
        assert_eq!(
            repository.stop_process(second).unwrap(),
            ProcessCommandAck::Conflict
        );
        let (stopped, health): (i64, i64) = repository
            .connection
            .query_row(
                "SELECT
                    (SELECT COUNT(*) FROM process_instance_state_events WHERE state = 'stopped'),
                    (SELECT COUNT(*) FROM health_events
                     WHERE stable_class = 'router.ledger.integrity_conflict')",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!((stopped, health), (1, 1));
    }

    #[test]
    fn exact_stop_retry_rejects_a_corrupt_process_parent() {
        let (_temporary, mut repository) = repository();
        let stop = ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), NOW).unwrap();
        assert_eq!(
            repository.stop_process(stop).unwrap(),
            ProcessCommandAck::Applied
        );
        repository
            .connection
            .execute(
                "UPDATE process_instances SET canonical_payload_hash = ?1
                 WHERE process_instance_id = ?2",
                params!["f".repeat(64), repository.process_instance_id.to_string()],
            )
            .unwrap();
        assert_eq!(
            repository.stop_process(stop).unwrap(),
            ProcessCommandAck::Conflict
        );
    }

    #[test]
    fn health_events_require_full_duplicate_equality() {
        let (_temporary, mut repository) = repository();
        let event_id = Uuid::now_v7();
        let event = LedgerHealthEvent::new(
            event_id,
            Uuid::now_v7(),
            None,
            None,
            "router.dependency.cooloff_active",
            LedgerHealthSeverity::Warning,
            NOW,
        )
        .unwrap();
        assert_eq!(
            repository.append_health_event(&event).unwrap(),
            ProcessCommandAck::Applied
        );
        assert_eq!(
            repository.append_health_event(&event).unwrap(),
            ProcessCommandAck::AlreadyApplied
        );

        let mismatch = LedgerHealthEvent::new(
            event_id,
            Uuid::now_v7(),
            Some(Uuid::now_v7()),
            Some("a".repeat(64)),
            "router.dependency.recovered",
            LedgerHealthSeverity::Info,
            NOW,
        )
        .unwrap();
        assert_eq!(
            repository.append_health_event(&mismatch).unwrap(),
            ProcessCommandAck::Conflict
        );
        let rows: i64 = repository
            .connection
            .query_row("SELECT COUNT(*) FROM health_events", [], |row| row.get(0))
            .unwrap();
        assert_eq!(rows, 2);
        let conflict_refs: (Option<String>, Option<String>) = repository
            .connection
            .query_row(
                "SELECT anchor_id, dependency_key_id FROM health_events
                 WHERE health_event_id = ?1",
                [mismatch.conflict_health_event_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(conflict_refs, (None, None));
    }

    #[test]
    fn nullable_integrity_context_remains_exactly_idempotent() {
        let (_temporary, repository) = repository();
        let health_event_id = Uuid::now_v7();
        let first_anchor = Uuid::now_v7();
        let first_dependency = "a".repeat(64);
        append_integrity_health(
            &repository.connection,
            health_event_id,
            repository.project_uuid,
            repository.process_instance_id,
            Some(first_anchor),
            Some(&first_dependency),
            NOW,
        )
        .unwrap();
        let stored: (
            Option<String>,
            Option<String>,
            Option<String>,
            Option<String>,
        ) = repository
            .connection
            .query_row(
                "SELECT requested_anchor_id, requested_dependency_key_id,
                        anchor_id, dependency_key_id
                 FROM health_events WHERE health_event_id = ?1",
                [health_event_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(
            stored,
            (
                Some(first_anchor.to_string()),
                Some(first_dependency.clone()),
                None,
                None,
            )
        );
        append_integrity_health(
            &repository.connection,
            health_event_id,
            repository.project_uuid,
            repository.process_instance_id,
            Some(first_anchor),
            Some(&first_dependency),
            NOW,
        )
        .unwrap();

        let error = append_integrity_health(
            &repository.connection,
            health_event_id,
            repository.project_uuid,
            repository.process_instance_id,
            Some(Uuid::now_v7()),
            Some(&"b".repeat(64)),
            NOW,
        )
        .unwrap_err();
        assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);
    }

    #[test]
    fn health_and_process_inputs_fail_closed() {
        assert!(HeartbeatRenewal::new(-1).is_err());
        assert!(ProcessStop::new(Uuid::nil(), Uuid::now_v7(), NOW).is_err());
        assert!(
            LedgerHealthEvent::new(
                Uuid::now_v7(),
                Uuid::now_v7(),
                None,
                None,
                "contains-hyphen",
                LedgerHealthSeverity::Degraded,
                NOW,
            )
            .is_err()
        );
    }
}
