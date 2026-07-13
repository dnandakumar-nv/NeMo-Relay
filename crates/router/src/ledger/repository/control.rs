// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Verified activation and transactional storage for durable Router controls.

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use rusqlite::{
    Connection, ErrorCode, OptionalExtension, Transaction, TransactionBehavior, params,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::{Uuid, Variant};

use super::{LedgerError, LedgerErrorClass};
use crate::control::{
    CONTROL_ACTOR_MAX_BYTES, CONTROL_REASON_MAX_BYTES, ControlMutation, ControlMutationReceipt,
    ControlMutationResult, ControlOperation, ControlScope, ControlTransactionFence,
    RouterControlError, RouterControlSnapshot, RouterControlState, RouterPoolControlSnapshot,
    is_sha256,
};
use crate::ledger::cohort::CohortAssignmentAuthority;
use crate::ledger::repository::inspection::operator::{
    CurrentGenerationAuthority, load_current_generation_authority,
};

const ACTIVE_WRITER_PROTOCOL: &str = "active-v6";
const ACTIVE_WRITER_SCHEMA_VERSION: i64 = 6;
const INITIAL_CONTROL_ACTOR: &str = "nemo-relay-router";
const INITIAL_CONTROL_REASON: &str = "initialize control state v1";
const CURRENT_CONFIG_EVENT_DOMAIN: &[u8] = b"nemo-relay-router/current-config-event/v1\0";
const WRITER_CAPABILITY_DOMAIN: &[u8] = b"nemo-relay-router/writer-capability/v1\0";
const CONTROL_RECORD_DOMAIN: &[u8] = b"nemo-relay-router/control-record/v1\0";
const CONTROL_MUTATION_DOMAIN: &[u8] = b"nemo-relay-router/control-mutation/v1\0";
const CONTROL_HISTORY_ENTRY_DOMAIN: &[u8] = b"nemo-relay-router/control-history-entry/v1\0";
const CONTROL_HISTORY_RANGE_DOMAIN: &[u8] = b"nemo-relay-router/control-history-range/v1\0";
const CONTROL_ENDING_STATES_DOMAIN: &[u8] = b"nemo-relay-router/control-ending-states/v1\0";
const CONTROL_CHECKPOINT_DOMAIN: &[u8] = b"nemo-relay-router/control-checkpoint/v1\0";
const CONTROL_CHECKPOINT_MERGE_DOMAIN: &[u8] = b"nemo-relay-router/control-checkpoint-merge/v1\0";
const CONTROL_HISTORY_LIVE_MAX: i64 = 100_000;
const CONTROL_HISTORY_ORDINARY_MAX: i64 = 98_976;
const CONTROL_HISTORY_SATURATION_THRESHOLD: i64 = 98_974;
const CONTROL_MUTATION_CLOCK_SKEW_MS: i128 = 300_000;
const CONTROL_COMPACTION_BATCH_MAX: usize = 512;
const CONTROL_CHECKPOINT_MAX: usize = 1_024;
const CONTROL_CHECKPOINT_MERGE_COUNT: usize = 512;
const CONTROL_ENDING_STATES_MAX_BYTES: usize = 32 * 1024 * 1024;
const MILLIS_PER_DAY: i64 = 86_400_000;

#[derive(Debug, Clone)]
pub(crate) struct PreparedControlMutation {
    mutation: ControlMutation,
    canonical_payload_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ControlMutationAck {
    Completed {
        result: ControlMutationResult,
        snapshot: RouterControlSnapshot,
        saturated: bool,
    },
    TransactionNotStarted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ControlAuthoritySnapshot {
    pub(crate) snapshot: RouterControlSnapshot,
    pub(crate) saturated: bool,
}

pub(crate) struct RuntimeGenerationAuthority {
    pub(crate) control: ControlAuthoritySnapshot,
    pub(crate) cohort_assignment: CohortAssignmentAuthority,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedControlHistoryEntry {
    pub(crate) receipt: ControlMutationReceipt,
    pub(crate) prior_value: Option<bool>,
    pub(crate) new_value: Option<bool>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedControlHistoryPage {
    pub(crate) entries: Vec<VerifiedControlHistoryEntry>,
    pub(crate) maximum_history_ordinal: u64,
}

#[derive(Debug)]
struct StoredReceipt {
    mutation_id: String,
    canonical_payload_hash: String,
    history_ordinal: i64,
    predecessor_chain_hash: String,
    chain_tip_hash: String,
    project_uuid: String,
    result: String,
    result_control_generation: i64,
    control_id: Option<String>,
    control_record_hash: Option<String>,
    scope_kind: String,
    pool_id: Option<String>,
    operation_kind: String,
    requested_value: i64,
    expected_control_generation: i64,
    actor: String,
    reason: String,
    process_instance_id: String,
    created_at_unix_ms: i64,
}

#[derive(Debug)]
struct StoredConfigEvent {
    event_id: String,
    epoch: i64,
    project_uuid: String,
    config_generation_id: String,
    predecessor_event_hash: Option<String>,
    process_instance_id: String,
    created_at_unix_ms: i64,
    event_hash: String,
    canonical_payload_hash: String,
}

#[derive(Debug)]
struct StoredControl {
    control_id: String,
    control_generation: i64,
    originating_history_ordinal: i64,
    project_uuid: String,
    scope_kind: String,
    pool_id: Option<String>,
    force_anchor: i64,
    paused: i64,
    actor: String,
    reason: String,
    process_instance_id: String,
    created_at_unix_ms: i64,
    record_hash: String,
    canonical_payload_hash: String,
}

#[derive(Debug)]
struct StoredCheckpoint {
    checkpoint_hash: String,
    project_uuid: String,
    level: i64,
    first_history_ordinal: i64,
    last_history_ordinal: i64,
    first_predecessor_hash: String,
    covered_chain_tip_hash: String,
    receipt_count: i64,
    applied_control_count: i64,
    range_hash: String,
    ending_states_json: String,
    ending_states_hash: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CheckpointEndingState {
    scope: ControlScope,
    control_generation: u64,
    control_id: Uuid,
    force_anchor: bool,
    paused: bool,
    record_hash: String,
}

#[derive(Debug)]
struct VerifiedCheckpoint {
    checkpoint_hash: String,
    level: u32,
    first_history_ordinal: u64,
    last_history_ordinal: u64,
    first_predecessor_hash: String,
    covered_chain_tip_hash: String,
    receipt_count: u64,
    applied_control_count: u64,
    ending_states: Vec<CheckpointEndingState>,
}

pub(super) fn initialize_v6_authority(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    config_generation_id: &str,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    if created_at_unix_ms < 0 || decode_sha256(config_generation_id).is_none() {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    ensure_writer_capability(
        transaction,
        project_uuid,
        process_instance_id,
        created_at_unix_ms,
    )?;
    ensure_current_config_event(
        transaction,
        project_uuid,
        process_instance_id,
        config_generation_id,
        created_at_unix_ms,
    )?;
    ensure_genesis_control(
        transaction,
        project_uuid,
        process_instance_id,
        created_at_unix_ms,
    )
}

pub(crate) fn load_control_snapshot(
    connection: &Transaction<'_>,
    project_uuid: Uuid,
    expected_config_generation_id: &str,
    pool_ids: &[String],
    generations: &CurrentGenerationAuthority,
) -> Result<RouterControlSnapshot, LedgerError> {
    let events = load_verified_config_events(connection, project_uuid)?;
    let current = events
        .last()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if current.config_generation_id != expected_config_generation_id {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let controls = load_latest_controls(connection, project_uuid)?;
    let all = controls
        .get(&ControlKey::All)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if all.control_generation < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let control_generation = controls
        .values()
        .map(|control| control.control_generation)
        .max()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let all_state = RouterControlState {
        force_anchor: parse_bool(all.force_anchor)?,
        paused: parse_bool(all.paused)?,
    };

    let mut pools = BTreeMap::new();
    for pool_id in pool_ids {
        let learning_generation_id = generations
            .learning_generation_ids
            .get(pool_id)
            .copied()
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let local = controls
            .get(&ControlKey::Pool(pool_id.clone()))
            .map(|control| -> Result<RouterControlState, LedgerError> {
                Ok(RouterControlState {
                    force_anchor: parse_bool(control.force_anchor)?,
                    paused: parse_bool(control.paused)?,
                })
            })
            .transpose()?
            .unwrap_or_default();
        pools.insert(
            pool_id.clone(),
            RouterPoolControlSnapshot {
                local,
                effective: RouterControlState {
                    force_anchor: all_state.force_anchor || local.force_anchor,
                    paused: all_state.paused || local.paused,
                },
                learning_generation_id,
            },
        );
    }

    Ok(RouterControlSnapshot {
        control_generation: u64::try_from(control_generation)
            .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
        cohort_generation_id: generations.cohort_generation_id,
        all: all_state,
        pools,
    })
}

pub(crate) fn control_config_is_current(
    connection: &Connection,
    project_uuid: Uuid,
    expected_config_generation_id: &str,
) -> Result<bool, LedgerError> {
    Ok(load_verified_config_events(connection, project_uuid)?
        .last()
        .is_some_and(|event| event.config_generation_id == expected_config_generation_id))
}

pub(crate) fn load_control_authority_snapshot(
    connection: &Transaction<'_>,
    project_uuid: Uuid,
    expected_config_generation_id: &str,
    pool_ids: &[String],
) -> Result<ControlAuthoritySnapshot, LedgerError> {
    let generations = load_current_generation_authority(connection, project_uuid, pool_ids)?;
    load_control_authority_with_generations(
        connection,
        project_uuid,
        expected_config_generation_id,
        pool_ids,
        &generations,
    )
}

pub(crate) fn load_runtime_generation_authority(
    connection: &Transaction<'_>,
    project_uuid: Uuid,
    expected_config_generation_id: &str,
    pool_ids: &[String],
) -> Result<RuntimeGenerationAuthority, LedgerError> {
    let generations = load_current_generation_authority(connection, project_uuid, pool_ids)?;
    let control = load_control_authority_with_generations(
        connection,
        project_uuid,
        expected_config_generation_id,
        pool_ids,
        &generations,
    )?;
    let CurrentGenerationAuthority {
        cohort_generation_id,
        cohort_salt,
        ..
    } = generations;
    Ok(RuntimeGenerationAuthority {
        control,
        cohort_assignment: CohortAssignmentAuthority::new(cohort_generation_id, cohort_salt),
    })
}

fn load_control_authority_with_generations(
    connection: &Transaction<'_>,
    project_uuid: Uuid,
    expected_config_generation_id: &str,
    pool_ids: &[String],
    generations: &CurrentGenerationAuthority,
) -> Result<ControlAuthoritySnapshot, LedgerError> {
    let snapshot = load_control_snapshot(
        connection,
        project_uuid,
        expected_config_generation_id,
        pool_ids,
        generations,
    )?;
    let live_count = connection
        .query_row(
            "SELECT (SELECT count(*) FROM controls WHERE project_uuid = ?1)
                  + (SELECT count(*) FROM control_mutation_receipts WHERE project_uuid = ?1)",
            [project_uuid.to_string()],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    if !(1..=CONTROL_HISTORY_LIVE_MAX).contains(&live_count) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(ControlAuthoritySnapshot {
        snapshot,
        saturated: live_count > CONTROL_HISTORY_SATURATION_THRESHOLD,
    })
}

pub(crate) fn load_verified_control_history_page(
    connection: &Connection,
    project_uuid: Uuid,
    snapshot_time_unix_ms: i64,
    maximum_history_ordinal: Option<u64>,
    after: Option<(i64, &str)>,
    limit: usize,
) -> Result<VerifiedControlHistoryPage, RouterControlError> {
    if snapshot_time_unix_ms < 0 || limit == 0 {
        return Err(RouterControlError::InvalidArgument);
    }
    let (live_start_ordinal, live_predecessor_hash, actual_maximum) =
        verified_live_history_authority(connection, project_uuid)?;
    let maximum = maximum_history_ordinal.unwrap_or(actual_maximum);
    if maximum > actual_maximum {
        return Err(RouterControlError::IntegrityError);
    }
    let after_time = after.map(|value| value.0);
    let after_id = after.map(|value| value.1);
    let mut statement = connection
        .prepare(
            "SELECT mutation_id, canonical_payload_hash, history_ordinal,
                    predecessor_chain_hash, chain_tip_hash, project_uuid,
                    result, result_control_generation, control_id, control_record_hash,
                    scope_kind, pool_id, operation_kind, requested_value,
                    expected_control_generation, actor, reason,
                    process_instance_id, created_at_unix_ms
             FROM control_mutation_receipts
             WHERE project_uuid = ?1 AND history_ordinal <= ?2
               AND created_at_unix_ms <= ?3
               AND (?4 IS NULL OR created_at_unix_ms < ?4
                    OR (created_at_unix_ms = ?4 AND mutation_id < ?5))
             ORDER BY created_at_unix_ms DESC, mutation_id DESC
             LIMIT ?6",
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    let stored = statement
        .query_map(
            params![
                project_uuid.to_string(),
                i64::try_from(maximum).map_err(|_| RouterControlError::IntegrityError)?,
                snapshot_time_unix_ms,
                after_time,
                after_id,
                i64::try_from(limit).map_err(|_| RouterControlError::InvalidArgument)?,
            ],
            stored_receipt_from_row,
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    drop(statement);

    let mut entries = Vec::with_capacity(stored.len());
    for stored in stored {
        let prepared = prepared_from_stored_receipt(&stored)?;
        let result = verify_replayed_receipt(connection, project_uuid, &prepared, &stored)?;
        let history_ordinal = u64::try_from(stored.history_ordinal)
            .map_err(|_| RouterControlError::IntegrityError)?;
        let expected_predecessor = if history_ordinal == live_start_ordinal {
            live_predecessor_hash.clone()
        } else {
            let predecessor_ordinal = history_ordinal
                .checked_sub(1)
                .ok_or(RouterControlError::IntegrityError)?;
            connection
                .query_row(
                    "SELECT chain_tip_hash FROM control_mutation_receipts
                     WHERE project_uuid = ?1 AND history_ordinal = ?2",
                    params![
                        project_uuid.to_string(),
                        i64::try_from(predecessor_ordinal)
                            .map_err(|_| RouterControlError::IntegrityError)?,
                    ],
                    |row| row.get::<_, String>(0),
                )
                .optional()
                .map_err(|_| RouterControlError::StorageUnavailable)?
                .ok_or(RouterControlError::IntegrityError)?
        };
        if stored.predecessor_chain_hash != expected_predecessor {
            return Err(RouterControlError::IntegrityError);
        }

        let result_control_generation = u64::try_from(stored.result_control_generation)
            .map_err(|_| RouterControlError::IntegrityError)?;
        let prior_value = match result {
            ControlMutationResult::Applied => {
                let prior = load_prior_scope_state(
                    connection,
                    project_uuid,
                    &prepared.mutation.scope,
                    result_control_generation,
                )?;
                Some(operation_field(&prepared.mutation.operation, prior))
            }
            ControlMutationResult::NoOp => Some(prepared.mutation.operation.requested_value()),
            ControlMutationResult::Conflict => None,
        };
        let new_value = match result {
            ControlMutationResult::Applied | ControlMutationResult::NoOp => {
                Some(prepared.mutation.operation.requested_value())
            }
            ControlMutationResult::Conflict => None,
        };
        entries.push(VerifiedControlHistoryEntry {
            receipt: ControlMutationReceipt {
                mutation_id: prepared.mutation.mutation_id,
                canonical_payload_hash: stored.canonical_payload_hash,
                history_ordinal,
                predecessor_chain_hash: stored.predecessor_chain_hash,
                chain_tip_hash: stored.chain_tip_hash,
                result,
                result_control_generation,
                control_id: stored
                    .control_id
                    .as_deref()
                    .map(parse_uuid)
                    .transpose()
                    .map_err(|_| RouterControlError::IntegrityError)?,
                control_record_hash: stored.control_record_hash,
                scope: prepared.mutation.scope,
                operation: prepared.mutation.operation,
                expected_control_generation: prepared.mutation.expected_control_generation,
                actor: prepared.mutation.actor,
                reason: prepared.mutation.reason,
                process_id: parse_uuid_v7(&stored.process_instance_id)
                    .map_err(|_| RouterControlError::IntegrityError)?,
                created_at_unix_ms: u64::try_from(stored.created_at_unix_ms)
                    .map_err(|_| RouterControlError::IntegrityError)?,
            },
            prior_value,
            new_value,
        });
    }
    Ok(VerifiedControlHistoryPage {
        entries,
        maximum_history_ordinal: maximum,
    })
}

fn verified_live_history_authority(
    connection: &Connection,
    project_uuid: Uuid,
) -> Result<(u64, String, u64), RouterControlError> {
    let checkpoints = load_verified_checkpoints(connection, project_uuid)?;
    let genesis = load_control_by_generation(connection, project_uuid, 0)
        .map_err(|_| RouterControlError::IntegrityError)?
        .ok_or(RouterControlError::IntegrityError)?;
    verify_stored_control(&genesis, project_uuid)
        .map_err(|_| RouterControlError::IntegrityError)?;
    let (covered_ordinal, predecessor_hash) = checkpoints
        .last()
        .map(|checkpoint| {
            (
                checkpoint.last_history_ordinal,
                checkpoint.covered_chain_tip_hash.clone(),
            )
        })
        .unwrap_or((0, genesis.record_hash));
    let live = connection
        .query_row(
            "SELECT min(history_ordinal), max(history_ordinal), count(*)
             FROM control_mutation_receipts WHERE project_uuid = ?1",
            [project_uuid.to_string()],
            |row| {
                Ok((
                    row.get::<_, Option<i64>>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    let live_start = covered_ordinal
        .checked_add(1)
        .ok_or(RouterControlError::IntegrityError)?;
    let actual_maximum = match live {
        (None, None, 0) => covered_ordinal,
        (Some(first), Some(last), count) => {
            let first = u64::try_from(first).map_err(|_| RouterControlError::IntegrityError)?;
            let last = u64::try_from(last).map_err(|_| RouterControlError::IntegrityError)?;
            let count = u64::try_from(count).map_err(|_| RouterControlError::IntegrityError)?;
            if first != live_start
                || last.checked_sub(first).and_then(|span| span.checked_add(1)) != Some(count)
            {
                return Err(RouterControlError::IntegrityError);
            }
            let first_predecessor = connection
                .query_row(
                    "SELECT predecessor_chain_hash FROM control_mutation_receipts
                     WHERE project_uuid = ?1 AND history_ordinal = ?2",
                    params![
                        project_uuid.to_string(),
                        i64::try_from(first).map_err(|_| RouterControlError::IntegrityError)?,
                    ],
                    |row| row.get::<_, String>(0),
                )
                .map_err(|_| RouterControlError::StorageUnavailable)?;
            if first_predecessor != predecessor_hash {
                return Err(RouterControlError::IntegrityError);
            }
            last
        }
        _ => return Err(RouterControlError::IntegrityError),
    };
    Ok((live_start, predecessor_hash, actual_maximum))
}

fn load_prior_scope_state(
    connection: &Connection,
    project_uuid: Uuid,
    scope: &ControlScope,
    before_generation: u64,
) -> Result<RouterControlState, RouterControlError> {
    let (scope_kind, pool_id) = scope_columns(scope);
    let stored = connection
        .query_row(
            "SELECT control_id, control_generation, originating_history_ordinal,
                    project_uuid, scope_kind, pool_id, force_anchor, paused,
                    actor, reason, process_instance_id, created_at_unix_ms,
                    record_hash, canonical_payload_hash
             FROM controls
             WHERE project_uuid = ?1 AND control_generation < ?2
               AND scope_kind = ?3
               AND ((?4 IS NULL AND pool_id IS NULL) OR pool_id = ?4)
             ORDER BY control_generation DESC LIMIT 1",
            params![
                project_uuid.to_string(),
                i64::try_from(before_generation).map_err(|_| RouterControlError::IntegrityError)?,
                scope_kind,
                pool_id,
            ],
            stored_control_from_row,
        )
        .optional()
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    let live = stored
        .map(|stored| {
            verify_stored_control(&stored, project_uuid)
                .map_err(|_| RouterControlError::IntegrityError)?;
            Ok((
                u64::try_from(stored.control_generation)
                    .map_err(|_| RouterControlError::IntegrityError)?,
                RouterControlState {
                    force_anchor: parse_bool(stored.force_anchor)
                        .map_err(|_| RouterControlError::IntegrityError)?,
                    paused: parse_bool(stored.paused)
                        .map_err(|_| RouterControlError::IntegrityError)?,
                },
            ))
        })
        .transpose()?;
    let checkpoint = load_verified_checkpoints(connection, project_uuid)?
        .into_iter()
        .flat_map(|checkpoint| checkpoint.ending_states)
        .filter(|state| state.scope == *scope && state.control_generation < before_generation)
        .max_by_key(|state| state.control_generation)
        .map(|state| {
            (
                state.control_generation,
                RouterControlState {
                    force_anchor: state.force_anchor,
                    paused: state.paused,
                },
            )
        });
    live.into_iter()
        .chain(checkpoint)
        .max_by_key(|(generation, _)| *generation)
        .map(|(_, state)| state)
        .or_else(|| match scope {
            ControlScope::Pool { .. } => Some(RouterControlState::default()),
            ControlScope::All => None,
        })
        .ok_or(RouterControlError::IntegrityError)
}

fn operation_field(operation: &ControlOperation, state: RouterControlState) -> bool {
    match operation {
        ControlOperation::SetForceAnchor { .. } => state.force_anchor,
        ControlOperation::SetPaused { .. } => state.paused,
    }
}

pub(crate) fn prepare_control_mutation(
    mutation: ControlMutation,
) -> Result<PreparedControlMutation, RouterControlError> {
    if mutation.mutation_id.get_variant() != Variant::RFC4122
        || mutation.mutation_id.get_version_num() != 7
        || mutation.expected_control_generation > i64::MAX as u64
        || !valid_operator_text(&mutation.actor, CONTROL_ACTOR_MAX_BYTES)
        || !valid_operator_text(&mutation.reason, CONTROL_REASON_MAX_BYTES)
        || matches!(
            &mutation.scope,
            ControlScope::Pool { pool_id }
                if pool_id.is_empty() || pool_id.len() > 128
        )
    {
        return Err(RouterControlError::InvalidArgument);
    }
    let canonical_payload_hash =
        control_mutation_hash(&mutation).map_err(|_| RouterControlError::InvalidArgument)?;
    Ok(PreparedControlMutation {
        mutation,
        canonical_payload_hash,
    })
}

impl super::LedgerRepository {
    pub(crate) fn apply_control_mutation_with_start_check<G: super::TransactionStartGuard>(
        &mut self,
        prepared: &PreparedControlMutation,
        fence: &Arc<ControlTransactionFence>,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ControlMutationAck, RouterControlError> {
        let database_path = self.database_path.clone();
        super::enforce_sidecar_permissions(&database_path)
            .map_err(|_| RouterControlError::StorageUnavailable)?;
        let Some(start_guard) = start_check() else {
            fence.abort();
            return Ok(ControlMutationAck::TransactionNotStarted);
        };
        let now = Instant::now();
        let Some(remaining) = fence.start_deadline().checked_duration_since(now) else {
            fence.expire();
            return Ok(ControlMutationAck::TransactionNotStarted);
        };
        self.connection
            .busy_timeout(remaining)
            .map_err(|_| RouterControlError::StorageUnavailable)?;
        let active_pool_ids = self.active_pool_ids.iter().cloned().collect::<Vec<_>>();
        let retention_days = self.retention_days;
        let outcome = execute_control_transaction(
            &mut self.connection,
            &database_path,
            self.project_uuid,
            self.process_instance_id,
            &self.config_generation_id,
            &active_pool_ids,
            retention_days,
            prepared,
            fence,
            start_guard,
        );
        self.connection
            .busy_timeout(super::BUSY_TIMEOUT)
            .map_err(|_| RouterControlError::StorageUnavailable)?;
        outcome
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_control_transaction<G: super::TransactionStartGuard>(
    connection: &mut Connection,
    database_path: &std::path::Path,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    expected_config_generation_id: &str,
    active_pool_ids: &[String],
    retention_days: u32,
    prepared: &PreparedControlMutation,
    fence: &Arc<ControlTransactionFence>,
    start_guard: G,
) -> Result<ControlMutationAck, RouterControlError> {
    let transaction = match connection.transaction_with_behavior(TransactionBehavior::Immediate) {
        Ok(transaction) => transaction,
        Err(error) => {
            if !start_guard.permits_transaction() {
                fence.abort();
                return Ok(ControlMutationAck::TransactionNotStarted);
            }
            if matches!(
                error.sqlite_error_code(),
                Some(ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked)
            ) {
                fence.expire();
                return Ok(ControlMutationAck::TransactionNotStarted);
            }
            return Err(RouterControlError::StorageUnavailable);
        }
    };
    if !start_guard.permits_transaction() {
        drop(transaction);
        fence.abort();
        return Ok(ControlMutationAck::TransactionNotStarted);
    }
    if !fence.try_start() {
        drop(transaction);
        return Ok(ControlMutationAck::TransactionNotStarted);
    }
    drop(start_guard);

    let created_at_unix_ms = Utc::now().timestamp_millis();
    if created_at_unix_ms < 0 {
        return Err(RouterControlError::StorageUnavailable);
    }
    let outcome = apply_control_mutation_in_transaction(
        &transaction,
        project_uuid,
        process_instance_id,
        expected_config_generation_id,
        active_pool_ids,
        retention_days,
        prepared,
        created_at_unix_ms,
    )?;
    super::enforce_sidecar_permissions(database_path)
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    transaction
        .commit()
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    Ok(outcome)
}

#[allow(clippy::too_many_arguments)]
fn apply_control_mutation_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    expected_config_generation_id: &str,
    active_pool_ids: &[String],
    retention_days: u32,
    prepared: &PreparedControlMutation,
    created_at_unix_ms: i64,
) -> Result<ControlMutationAck, RouterControlError> {
    if let Some(stored) =
        load_existing_receipt(transaction, project_uuid, prepared.mutation.mutation_id)?
    {
        let result = verify_replayed_receipt(transaction, project_uuid, prepared, &stored)?;
        let authority = load_control_authority_snapshot(
            transaction,
            project_uuid,
            expected_config_generation_id,
            active_pool_ids,
        )
        .map_err(|_| RouterControlError::IntegrityError)?;
        return Ok(ControlMutationAck::Completed {
            result,
            snapshot: authority.snapshot,
            saturated: authority.saturated,
        });
    }

    validate_uuid_time(prepared.mutation.mutation_id, created_at_unix_ms)?;
    verify_current_config(transaction, project_uuid, expected_config_generation_id)?;
    verify_writer_process(transaction, project_uuid, process_instance_id)?;
    verify_live_writer_capabilities(transaction, project_uuid)?;
    if let ControlScope::Pool { pool_id } = &prepared.mutation.scope
        && (!active_pool_ids.iter().any(|active| active == pool_id)
            || load_current_learning_generation(transaction, project_uuid, pool_id).is_err())
    {
        return Err(RouterControlError::InvalidArgument);
    }

    compact_control_history(
        transaction,
        project_uuid,
        retention_days,
        created_at_unix_ms,
    )?;

    let latest = load_latest_controls(transaction, project_uuid)
        .map_err(|_| RouterControlError::IntegrityError)?;
    let latest_generation = latest
        .values()
        .map(|control| control.control_generation)
        .max()
        .ok_or(RouterControlError::IntegrityError)?;
    let latest_generation_u64 =
        u64::try_from(latest_generation).map_err(|_| RouterControlError::IntegrityError)?;
    let scope_key = control_key(&prepared.mutation.scope);
    let current_scope = latest.get(&scope_key);
    let current_state = current_scope
        .map(|control| {
            Ok(RouterControlState {
                force_anchor: parse_bool(control.force_anchor)
                    .map_err(|_| RouterControlError::IntegrityError)?,
                paused: parse_bool(control.paused)
                    .map_err(|_| RouterControlError::IntegrityError)?,
            })
        })
        .transpose()?
        .unwrap_or_default();
    let result = if prepared.mutation.expected_control_generation != latest_generation_u64 {
        ControlMutationResult::Conflict
    } else if operation_already_applied(&prepared.mutation.operation, current_state) {
        ControlMutationResult::NoOp
    } else {
        ControlMutationResult::Applied
    };
    let added_rows = if result == ControlMutationResult::Applied {
        2
    } else {
        1
    };
    enforce_control_capacity(transaction, &prepared.mutation, added_rows)?;

    let (history_ordinal, predecessor_chain_hash) = next_history_node(transaction, project_uuid)?;
    let result_control_generation = if result == ControlMutationResult::Applied {
        latest_generation_u64
            .checked_add(1)
            .filter(|value| *value <= i64::MAX as u64)
            .ok_or(RouterControlError::CapacityExhausted)?
    } else {
        latest_generation_u64
    };
    let (control_id, control_record_hash) = if result == ControlMutationResult::Applied {
        let (force_anchor, paused) = apply_operation(&prepared.mutation.operation, current_state);
        let record_hash = control_record_hash(
            prepared.mutation.mutation_id,
            result_control_generation,
            history_ordinal,
            project_uuid,
            &prepared.mutation.scope,
            force_anchor,
            paused,
            &prepared.mutation.actor,
            &prepared.mutation.reason,
            process_instance_id,
            created_at_unix_ms,
        )
        .map_err(|_| RouterControlError::IntegrityError)?;
        insert_control(
            transaction,
            project_uuid,
            process_instance_id,
            prepared,
            history_ordinal,
            result_control_generation,
            force_anchor,
            paused,
            created_at_unix_ms,
            &record_hash,
        )?;
        (Some(prepared.mutation.mutation_id), Some(record_hash))
    } else {
        (None, None)
    };
    let chain_tip_hash = control_history_entry_hash(
        history_ordinal,
        &predecessor_chain_hash,
        project_uuid,
        &prepared.mutation,
        &prepared.canonical_payload_hash,
        result,
        result_control_generation,
        control_id.zip(control_record_hash.as_deref()),
        process_instance_id,
        created_at_unix_ms,
    )
    .map_err(|_| RouterControlError::IntegrityError)?;
    insert_receipt(
        transaction,
        project_uuid,
        process_instance_id,
        prepared,
        history_ordinal,
        &predecessor_chain_hash,
        &chain_tip_hash,
        result,
        result_control_generation,
        control_id,
        control_record_hash.as_deref(),
        created_at_unix_ms,
    )?;

    let authority = load_control_authority_snapshot(
        transaction,
        project_uuid,
        expected_config_generation_id,
        active_pool_ids,
    )
    .map_err(|_| RouterControlError::IntegrityError)?;
    Ok(ControlMutationAck::Completed {
        result,
        snapshot: authority.snapshot,
        saturated: authority.saturated,
    })
}

fn valid_operator_text(value: &str, max_bytes: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= max_bytes
        && value.chars().all(|character| !character.is_control())
}

fn control_mutation_hash(mutation: &ControlMutation) -> Result<String, LedgerError> {
    let mut digest = Sha256::new();
    digest.update(CONTROL_MUTATION_DOMAIN);
    encode_scope(&mut digest, &mutation.scope)?;
    encode_operation(&mut digest, &mutation.operation);
    digest.update(mutation.expected_control_generation.to_be_bytes());
    encode_string(&mut digest, &mutation.actor)?;
    encode_string(&mut digest, &mutation.reason)?;
    Ok(hex_digest(digest.finalize()))
}

fn validate_uuid_time(
    mutation_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), RouterControlError> {
    let timestamp = mutation_id
        .get_timestamp()
        .ok_or(RouterControlError::InvalidArgument)?;
    let (seconds, subsec_nanos) = timestamp.to_unix();
    let mutation_millis = u128::from(seconds)
        .checked_mul(1_000)
        .and_then(|value| value.checked_add(u128::from(subsec_nanos / 1_000_000)))
        .and_then(|value| i128::try_from(value).ok())
        .ok_or(RouterControlError::InvalidArgument)?;
    let writer_millis = i128::from(created_at_unix_ms);
    let difference = mutation_millis - writer_millis;
    if (-CONTROL_MUTATION_CLOCK_SKEW_MS..=CONTROL_MUTATION_CLOCK_SKEW_MS).contains(&difference) {
        Ok(())
    } else {
        Err(RouterControlError::MutationExpired)
    }
}

fn verify_current_config(
    connection: &Connection,
    project_uuid: Uuid,
    expected_config_generation_id: &str,
) -> Result<(), RouterControlError> {
    let events = load_verified_config_events(connection, project_uuid)
        .map_err(|_| RouterControlError::IntegrityError)?;
    match events.last() {
        Some(event) if event.config_generation_id == expected_config_generation_id => Ok(()),
        Some(_) => Err(RouterControlError::Unavailable),
        None => Err(RouterControlError::IntegrityError),
    }
}

fn verify_writer_process(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
) -> Result<(), RouterControlError> {
    match super::process::originating_process_is_live(connection, project_uuid, process_instance_id)
    {
        Ok(true) => Ok(()),
        Ok(false) => Err(RouterControlError::Unavailable),
        Err(_) => Err(RouterControlError::IntegrityError),
    }
}

fn verify_live_writer_capabilities(
    connection: &Connection,
    project_uuid: Uuid,
) -> Result<(), RouterControlError> {
    let mut statement = connection
        .prepare(
            "SELECT p.process_instance_id, c.project_uuid, c.schema_version,
                    c.verified_at_unix_ms, c.canonical_payload_hash
             FROM process_instances AS p
             LEFT JOIN process_writer_capabilities AS c
               ON c.process_instance_id = p.process_instance_id
              AND c.writer_protocol = 'active-v6'
             WHERE p.project_uuid = ?1
               AND NOT EXISTS (
                    SELECT 1 FROM process_instance_state_events AS s
                    WHERE s.subject_process_instance_id = p.process_instance_id
                      AND s.state IN ('stopped', 'reconciled')
               )
             ORDER BY p.process_instance_id",
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    let rows = statement
        .query_map([project_uuid.to_string()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?,
                row.get::<_, Option<i64>>(2)?,
                row.get::<_, Option<i64>>(3)?,
                row.get::<_, Option<String>>(4)?,
            ))
        })
        .map_err(|_| RouterControlError::StorageUnavailable)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    for (process_id, stored_project, schema_version, verified_at, stored_hash) in rows {
        let process_id =
            parse_uuid_v7(&process_id).map_err(|_| RouterControlError::IntegrityError)?;
        let (Some(stored_project), Some(schema_version), Some(verified_at), Some(stored_hash)) =
            (stored_project, schema_version, verified_at, stored_hash)
        else {
            return Err(RouterControlError::MigrationRequired);
        };
        let expected_hash = writer_capability_hash(project_uuid, process_id, verified_at)
            .map_err(|_| RouterControlError::IntegrityError)?;
        if stored_project != project_uuid.to_string()
            || schema_version != ACTIVE_WRITER_SCHEMA_VERSION
            || verified_at < 0
            || stored_hash != expected_hash
        {
            return Err(RouterControlError::IntegrityError);
        }
    }
    Ok(())
}

fn control_key(scope: &ControlScope) -> ControlKey {
    match scope {
        ControlScope::All => ControlKey::All,
        ControlScope::Pool { pool_id } => ControlKey::Pool(pool_id.clone()),
    }
}

fn operation_already_applied(operation: &ControlOperation, state: RouterControlState) -> bool {
    match operation {
        ControlOperation::SetForceAnchor { value } => state.force_anchor == *value,
        ControlOperation::SetPaused { value } => state.paused == *value,
    }
}

fn apply_operation(operation: &ControlOperation, state: RouterControlState) -> (bool, bool) {
    match operation {
        ControlOperation::SetForceAnchor { value } => (*value, state.paused),
        ControlOperation::SetPaused { value } => (state.force_anchor, *value),
    }
}

fn enforce_control_capacity(
    connection: &Connection,
    mutation: &ControlMutation,
    added_rows: i64,
) -> Result<(), RouterControlError> {
    let live_count = connection
        .query_row(
            "SELECT (SELECT count(*) FROM controls)
                  + (SELECT count(*) FROM control_mutation_receipts)",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    let maximum = if mutation.operation.is_safety_setting() {
        CONTROL_HISTORY_LIVE_MAX
    } else {
        CONTROL_HISTORY_ORDINARY_MAX
    };
    if live_count < 0
        || added_rows <= 0
        || live_count
            .checked_add(added_rows)
            .is_none_or(|next| next > maximum)
    {
        return Err(RouterControlError::CapacityExhausted);
    }
    Ok(())
}

fn next_history_node(
    connection: &Connection,
    project_uuid: Uuid,
) -> Result<(u64, String), RouterControlError> {
    let live = connection
        .query_row(
            "SELECT history_ordinal, chain_tip_hash
             FROM control_mutation_receipts
             WHERE project_uuid = ?1
             ORDER BY history_ordinal DESC LIMIT 1",
            [project_uuid.to_string()],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    let checkpoint = connection
        .query_row(
            "SELECT last_history_ordinal, covered_chain_tip_hash
             FROM control_history_checkpoints
             WHERE project_uuid = ?1
             ORDER BY last_history_ordinal DESC LIMIT 1",
            [project_uuid.to_string()],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    let genesis = load_control_by_generation(connection, project_uuid, 0)
        .map_err(|_| RouterControlError::IntegrityError)?
        .ok_or(RouterControlError::IntegrityError)?;
    verify_stored_control(&genesis, project_uuid)
        .map_err(|_| RouterControlError::IntegrityError)?;
    let (last_ordinal, predecessor) = match (live, checkpoint) {
        (Some(live), Some(checkpoint)) if checkpoint.0 >= live.0 => checkpoint,
        (Some(live), _) => live,
        (None, Some(checkpoint)) => checkpoint,
        (None, None) => (0, genesis.record_hash),
    };
    if last_ordinal < 0 || !is_sha256(&predecessor) {
        return Err(RouterControlError::IntegrityError);
    }
    let next = u64::try_from(last_ordinal)
        .ok()
        .and_then(|value| value.checked_add(1))
        .filter(|value| *value <= i64::MAX as u64)
        .ok_or(RouterControlError::CapacityExhausted)?;
    Ok((next, predecessor))
}

#[allow(clippy::too_many_arguments)]
fn insert_control(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    prepared: &PreparedControlMutation,
    history_ordinal: u64,
    control_generation: u64,
    force_anchor: bool,
    paused: bool,
    created_at_unix_ms: i64,
    record_hash: &str,
) -> Result<(), RouterControlError> {
    let (scope_kind, pool_id) = scope_columns(&prepared.mutation.scope);
    let inserted = transaction
        .execute(
            "INSERT INTO controls (
                control_id, control_generation, originating_history_ordinal,
                project_uuid, scope_kind, pool_id, force_anchor, paused,
                actor, reason, process_instance_id, created_at_unix_ms,
                record_hash, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?13)",
            params![
                prepared.mutation.mutation_id.to_string(),
                i64::try_from(control_generation)
                    .map_err(|_| RouterControlError::CapacityExhausted)?,
                i64::try_from(history_ordinal)
                    .map_err(|_| RouterControlError::CapacityExhausted)?,
                project_uuid.to_string(),
                scope_kind,
                pool_id,
                i64::from(force_anchor),
                i64::from(paused),
                prepared.mutation.actor,
                prepared.mutation.reason,
                process_instance_id.to_string(),
                created_at_unix_ms,
                record_hash,
            ],
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    if inserted != 1 {
        return Err(RouterControlError::IntegrityError);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn insert_receipt(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    prepared: &PreparedControlMutation,
    history_ordinal: u64,
    predecessor_chain_hash: &str,
    chain_tip_hash: &str,
    result: ControlMutationResult,
    result_control_generation: u64,
    control_id: Option<Uuid>,
    control_record_hash: Option<&str>,
    created_at_unix_ms: i64,
) -> Result<(), RouterControlError> {
    let (scope_kind, pool_id) = scope_columns(&prepared.mutation.scope);
    let (operation_kind, requested_value) = operation_columns(&prepared.mutation.operation);
    let inserted = transaction
        .execute(
            "INSERT INTO control_mutation_receipts (
                mutation_id, canonical_payload_hash, history_ordinal,
                predecessor_chain_hash, chain_tip_hash, project_uuid,
                result, result_control_generation, control_id, control_record_hash,
                scope_kind, pool_id, operation_kind, requested_value,
                expected_control_generation, actor, reason,
                process_instance_id, created_at_unix_ms
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
                ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19
             )",
            params![
                prepared.mutation.mutation_id.to_string(),
                prepared.canonical_payload_hash,
                i64::try_from(history_ordinal)
                    .map_err(|_| RouterControlError::CapacityExhausted)?,
                predecessor_chain_hash,
                chain_tip_hash,
                project_uuid.to_string(),
                result_name(result),
                i64::try_from(result_control_generation)
                    .map_err(|_| RouterControlError::CapacityExhausted)?,
                control_id.map(|value| value.to_string()),
                control_record_hash,
                scope_kind,
                pool_id,
                operation_kind,
                i64::from(requested_value),
                i64::try_from(prepared.mutation.expected_control_generation)
                    .map_err(|_| RouterControlError::InvalidArgument)?,
                prepared.mutation.actor,
                prepared.mutation.reason,
                process_instance_id.to_string(),
                created_at_unix_ms,
            ],
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    if inserted != 1 {
        return Err(RouterControlError::IntegrityError);
    }
    let history_inserted = transaction
        .execute(
            "INSERT INTO operator_history_entries (
                project_uuid, audit_id, entry_kind, control_mutation_id,
                operator_mutation_id, created_at_unix_ms
             ) VALUES (?1, ?2, 'control', ?2, NULL, ?3)",
            params![
                project_uuid.to_string(),
                prepared.mutation.mutation_id.to_string(),
                created_at_unix_ms,
            ],
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    if history_inserted != 1 {
        return Err(RouterControlError::IntegrityError);
    }
    Ok(())
}

fn load_existing_receipt(
    connection: &Connection,
    project_uuid: Uuid,
    mutation_id: Uuid,
) -> Result<Option<StoredReceipt>, RouterControlError> {
    connection
        .query_row(
            "SELECT mutation_id, canonical_payload_hash, history_ordinal,
                    predecessor_chain_hash, chain_tip_hash, project_uuid,
                    result, result_control_generation, control_id, control_record_hash,
                    scope_kind, pool_id, operation_kind, requested_value,
                    expected_control_generation, actor, reason,
                    process_instance_id, created_at_unix_ms
             FROM control_mutation_receipts
             WHERE project_uuid = ?1 AND mutation_id = ?2",
            params![project_uuid.to_string(), mutation_id.to_string()],
            stored_receipt_from_row,
        )
        .optional()
        .map_err(|_| RouterControlError::StorageUnavailable)
}

fn stored_receipt_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredReceipt> {
    Ok(StoredReceipt {
        mutation_id: row.get(0)?,
        canonical_payload_hash: row.get(1)?,
        history_ordinal: row.get(2)?,
        predecessor_chain_hash: row.get(3)?,
        chain_tip_hash: row.get(4)?,
        project_uuid: row.get(5)?,
        result: row.get(6)?,
        result_control_generation: row.get(7)?,
        control_id: row.get(8)?,
        control_record_hash: row.get(9)?,
        scope_kind: row.get(10)?,
        pool_id: row.get(11)?,
        operation_kind: row.get(12)?,
        requested_value: row.get(13)?,
        expected_control_generation: row.get(14)?,
        actor: row.get(15)?,
        reason: row.get(16)?,
        process_instance_id: row.get(17)?,
        created_at_unix_ms: row.get(18)?,
    })
}

fn verify_replayed_receipt(
    connection: &Connection,
    project_uuid: Uuid,
    prepared: &PreparedControlMutation,
    stored: &StoredReceipt,
) -> Result<ControlMutationResult, RouterControlError> {
    let mutation_id =
        parse_uuid_v7(&stored.mutation_id).map_err(|_| RouterControlError::IntegrityError)?;
    let stored_project_uuid =
        parse_uuid(&stored.project_uuid).map_err(|_| RouterControlError::IntegrityError)?;
    let process_instance_id = parse_uuid_v7(&stored.process_instance_id)
        .map_err(|_| RouterControlError::IntegrityError)?;
    let history_ordinal =
        u64::try_from(stored.history_ordinal).map_err(|_| RouterControlError::IntegrityError)?;
    let result_generation = u64::try_from(stored.result_control_generation)
        .map_err(|_| RouterControlError::IntegrityError)?;
    let expected_generation = u64::try_from(stored.expected_control_generation)
        .map_err(|_| RouterControlError::IntegrityError)?;
    let scope = parse_scope(&stored.scope_kind, stored.pool_id.as_deref())?;
    let operation = parse_operation(&stored.operation_kind, stored.requested_value)?;
    let result = parse_result(&stored.result)?;
    let control_id = stored
        .control_id
        .as_deref()
        .map(parse_uuid)
        .transpose()
        .map_err(|_| RouterControlError::IntegrityError)?;
    let applied_pair = match (control_id, stored.control_record_hash.as_deref()) {
        (Some(control_id), Some(record_hash)) if is_sha256(record_hash) => {
            Some((control_id, record_hash))
        }
        (None, None) => None,
        _ => return Err(RouterControlError::IntegrityError),
    };
    if (result == ControlMutationResult::Applied) != applied_pair.is_some()
        || stored_project_uuid != project_uuid
        || stored.created_at_unix_ms < 0
        || !is_sha256(&stored.predecessor_chain_hash)
    {
        return Err(RouterControlError::IntegrityError);
    }
    let stored_mutation = ControlMutation {
        mutation_id,
        scope,
        operation,
        expected_control_generation: expected_generation,
        actor: stored.actor.clone(),
        reason: stored.reason.clone(),
    };
    let payload_hash =
        control_mutation_hash(&stored_mutation).map_err(|_| RouterControlError::IntegrityError)?;
    let chain_hash = control_history_entry_hash(
        history_ordinal,
        &stored.predecessor_chain_hash,
        project_uuid,
        &stored_mutation,
        &payload_hash,
        result,
        result_generation,
        applied_pair,
        process_instance_id,
        stored.created_at_unix_ms,
    )
    .map_err(|_| RouterControlError::IntegrityError)?;
    if payload_hash != stored.canonical_payload_hash || chain_hash != stored.chain_tip_hash {
        return Err(RouterControlError::IntegrityError);
    }
    verify_control_history_index_row(
        connection,
        project_uuid,
        mutation_id,
        stored.created_at_unix_ms,
    )?;
    if stored_mutation != prepared.mutation
        || prepared.canonical_payload_hash != stored.canonical_payload_hash
    {
        return Err(RouterControlError::InvalidArgument);
    }
    if let Some((control_id, record_hash)) = applied_pair {
        let control = connection
            .query_row(
                "SELECT control_id, control_generation, originating_history_ordinal,
                        project_uuid, scope_kind, pool_id, force_anchor, paused,
                        actor, reason, process_instance_id, created_at_unix_ms,
                        record_hash, canonical_payload_hash
                 FROM controls WHERE control_id = ?1",
                [control_id.to_string()],
                stored_control_from_row,
            )
            .optional()
            .map_err(|_| RouterControlError::StorageUnavailable)?
            .ok_or(RouterControlError::IntegrityError)?;
        verify_stored_control(&control, project_uuid)
            .map_err(|_| RouterControlError::IntegrityError)?;
        if control.record_hash != record_hash
            || control.control_generation != stored.result_control_generation
            || control.originating_history_ordinal != stored.history_ordinal
        {
            return Err(RouterControlError::IntegrityError);
        }
    }
    Ok(result)
}

fn verify_control_history_index_row(
    connection: &Connection,
    project_uuid: Uuid,
    mutation_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), RouterControlError> {
    let stored = connection
        .query_row(
            "SELECT project_uuid, audit_id, entry_kind, control_mutation_id,
                    operator_mutation_id, created_at_unix_ms
             FROM operator_history_entries WHERE control_mutation_id = ?1",
            [mutation_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            },
        )
        .optional()
        .map_err(|_| RouterControlError::StorageUnavailable)?
        .ok_or(RouterControlError::IntegrityError)?;
    let mutation_id = mutation_id.to_string();
    if stored
        != (
            project_uuid.to_string(),
            mutation_id.clone(),
            "control".to_string(),
            Some(mutation_id),
            None,
            created_at_unix_ms,
        )
    {
        return Err(RouterControlError::IntegrityError);
    }
    Ok(())
}

fn compact_control_history(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    retention_days: u32,
    now_unix_ms: i64,
) -> Result<(), RouterControlError> {
    let retention_millis = i64::from(retention_days)
        .checked_mul(MILLIS_PER_DAY)
        .ok_or(RouterControlError::IntegrityError)?;
    let Some(cutoff_unix_ms) = now_unix_ms.checked_sub(retention_millis) else {
        return Ok(());
    };

    let mut checkpoints = load_verified_checkpoints(transaction, project_uuid)?;

    let genesis = load_control_by_generation(transaction, project_uuid, 0)
        .map_err(|_| RouterControlError::IntegrityError)?
        .ok_or(RouterControlError::IntegrityError)?;
    verify_stored_control(&genesis, project_uuid)
        .map_err(|_| RouterControlError::IntegrityError)?;
    let (last_covered_ordinal, mut predecessor_hash) = checkpoints
        .last()
        .map(|checkpoint| {
            (
                checkpoint.last_history_ordinal,
                checkpoint.covered_chain_tip_hash.clone(),
            )
        })
        .unwrap_or((0, genesis.record_hash));
    let first_uncovered = last_covered_ordinal
        .checked_add(1)
        .filter(|ordinal| *ordinal <= i64::MAX as u64)
        .ok_or(RouterControlError::CapacityExhausted)?;

    let mut statement = transaction
        .prepare(
            "SELECT mutation_id, canonical_payload_hash, history_ordinal,
                    predecessor_chain_hash, chain_tip_hash, project_uuid,
                    result, result_control_generation, control_id, control_record_hash,
                    scope_kind, pool_id, operation_kind, requested_value,
                    expected_control_generation, actor, reason,
                    process_instance_id, created_at_unix_ms
             FROM control_mutation_receipts
             WHERE project_uuid = ?1 AND history_ordinal >= ?2
             ORDER BY history_ordinal
             LIMIT ?3",
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    let candidates = statement
        .query_map(
            params![
                project_uuid.to_string(),
                i64::try_from(first_uncovered)
                    .map_err(|_| RouterControlError::CapacityExhausted)?,
                i64::try_from(CONTROL_COMPACTION_BATCH_MAX)
                    .map_err(|_| RouterControlError::IntegrityError)?,
            ],
            stored_receipt_from_row,
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    drop(statement);

    let mut expected_ordinal = first_uncovered;
    let mut covered = Vec::new();
    for stored in candidates {
        let ordinal = u64::try_from(stored.history_ordinal)
            .map_err(|_| RouterControlError::IntegrityError)?;
        if ordinal != expected_ordinal || stored.predecessor_chain_hash != predecessor_hash {
            return Err(RouterControlError::IntegrityError);
        }
        if stored.created_at_unix_ms > cutoff_unix_ms {
            break;
        }
        let prepared = prepared_from_stored_receipt(&stored)?;
        verify_replayed_receipt(transaction, project_uuid, &prepared, &stored)?;
        predecessor_hash.clone_from(&stored.chain_tip_hash);
        expected_ordinal = expected_ordinal
            .checked_add(1)
            .ok_or(RouterControlError::CapacityExhausted)?;
        covered.push(stored);
    }
    if covered.is_empty() {
        return Ok(());
    }
    while checkpoints.len() >= CONTROL_CHECKPOINT_MAX {
        merge_oldest_checkpoints(transaction, project_uuid, &checkpoints, now_unix_ms)?;
        checkpoints = load_verified_checkpoints(transaction, project_uuid)?;
    }

    let first = covered.first().ok_or(RouterControlError::IntegrityError)?;
    let last = covered.last().ok_or(RouterControlError::IntegrityError)?;
    let first_ordinal =
        u64::try_from(first.history_ordinal).map_err(|_| RouterControlError::IntegrityError)?;
    let last_ordinal =
        u64::try_from(last.history_ordinal).map_err(|_| RouterControlError::IntegrityError)?;

    let mut applied_controls = Vec::new();
    let mut ending_by_scope = BTreeMap::new();
    for receipt in &covered {
        if parse_result(&receipt.result)? != ControlMutationResult::Applied {
            continue;
        }
        let control_id = receipt
            .control_id
            .as_deref()
            .ok_or(RouterControlError::IntegrityError)?;
        let control = load_control_by_id(transaction, control_id)?
            .ok_or(RouterControlError::IntegrityError)?;
        verify_stored_control(&control, project_uuid)
            .map_err(|_| RouterControlError::IntegrityError)?;
        if control.originating_history_ordinal != receipt.history_ordinal
            || Some(control.record_hash.as_str()) != receipt.control_record_hash.as_deref()
        {
            return Err(RouterControlError::IntegrityError);
        }
        let state = checkpoint_ending_state(&control)?;
        ending_by_scope.insert(control_key(&state.scope), state);
        applied_controls.push(control);
    }
    applied_controls.sort_by_key(|control| control.control_generation);
    let ending_states = ending_by_scope.into_values().collect::<Vec<_>>();
    let ending_states_json = canonical_ending_states_json(&ending_states)?;
    let ending_states_hash = control_ending_states_hash(&ending_states)?;
    let range_hash = control_history_range_hash(&covered, &applied_controls)?;
    let receipt_count =
        u64::try_from(covered.len()).map_err(|_| RouterControlError::IntegrityError)?;
    let applied_control_count =
        u64::try_from(applied_controls.len()).map_err(|_| RouterControlError::IntegrityError)?;
    let checkpoint_hash = control_checkpoint_hash(
        0,
        first_ordinal,
        last_ordinal,
        &first.predecessor_chain_hash,
        &last.chain_tip_hash,
        receipt_count,
        applied_control_count,
        &range_hash,
        &ending_states_hash,
    )?;
    insert_checkpoint(
        transaction,
        project_uuid,
        0,
        first_ordinal,
        last_ordinal,
        &first.predecessor_chain_hash,
        &last.chain_tip_hash,
        receipt_count,
        applied_control_count,
        &range_hash,
        &ending_states_json,
        &ending_states_hash,
        now_unix_ms,
        &checkpoint_hash,
    )?;

    let deleted_receipts = transaction
        .execute(
            "DELETE FROM control_mutation_receipts
             WHERE project_uuid = ?1
               AND history_ordinal BETWEEN ?2 AND ?3",
            params![
                project_uuid.to_string(),
                i64::try_from(first_ordinal).map_err(|_| RouterControlError::IntegrityError)?,
                i64::try_from(last_ordinal).map_err(|_| RouterControlError::IntegrityError)?,
            ],
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    if deleted_receipts != covered.len() {
        return Err(RouterControlError::IntegrityError);
    }
    delete_compacted_controls(transaction, project_uuid, last_ordinal, cutoff_unix_ms)?;
    Ok(())
}

fn prepared_from_stored_receipt(
    stored: &StoredReceipt,
) -> Result<PreparedControlMutation, RouterControlError> {
    let mutation = ControlMutation {
        mutation_id: parse_uuid_v7(&stored.mutation_id)
            .map_err(|_| RouterControlError::IntegrityError)?,
        scope: parse_scope(&stored.scope_kind, stored.pool_id.as_deref())?,
        operation: parse_operation(&stored.operation_kind, stored.requested_value)?,
        expected_control_generation: u64::try_from(stored.expected_control_generation)
            .map_err(|_| RouterControlError::IntegrityError)?,
        actor: stored.actor.clone(),
        reason: stored.reason.clone(),
    };
    let canonical_payload_hash =
        control_mutation_hash(&mutation).map_err(|_| RouterControlError::IntegrityError)?;
    Ok(PreparedControlMutation {
        mutation,
        canonical_payload_hash,
    })
}

fn load_control_by_id(
    connection: &Connection,
    control_id: &str,
) -> Result<Option<StoredControl>, RouterControlError> {
    connection
        .query_row(
            "SELECT control_id, control_generation, originating_history_ordinal,
                    project_uuid, scope_kind, pool_id, force_anchor, paused,
                    actor, reason, process_instance_id, created_at_unix_ms,
                    record_hash, canonical_payload_hash
             FROM controls WHERE control_id = ?1",
            [control_id],
            stored_control_from_row,
        )
        .optional()
        .map_err(|_| RouterControlError::StorageUnavailable)
}

fn checkpoint_ending_state(
    control: &StoredControl,
) -> Result<CheckpointEndingState, RouterControlError> {
    Ok(CheckpointEndingState {
        scope: parse_scope(&control.scope_kind, control.pool_id.as_deref())?,
        control_generation: u64::try_from(control.control_generation)
            .map_err(|_| RouterControlError::IntegrityError)?,
        control_id: parse_uuid(&control.control_id)
            .map_err(|_| RouterControlError::IntegrityError)?,
        force_anchor: parse_bool(control.force_anchor)
            .map_err(|_| RouterControlError::IntegrityError)?,
        paused: parse_bool(control.paused).map_err(|_| RouterControlError::IntegrityError)?,
        record_hash: control.record_hash.clone(),
    })
}

fn canonical_ending_states_json(
    ending_states: &[CheckpointEndingState],
) -> Result<String, RouterControlError> {
    let json =
        serde_json::to_string(ending_states).map_err(|_| RouterControlError::IntegrityError)?;
    if !(2..=CONTROL_ENDING_STATES_MAX_BYTES).contains(&json.len()) {
        return Err(RouterControlError::CapacityExhausted);
    }
    Ok(json)
}

fn control_history_range_hash(
    receipts: &[StoredReceipt],
    applied_controls: &[StoredControl],
) -> Result<String, RouterControlError> {
    let first = receipts.first().ok_or(RouterControlError::IntegrityError)?;
    let last = receipts.last().ok_or(RouterControlError::IntegrityError)?;
    let mut digest = Sha256::new();
    digest.update(CONTROL_HISTORY_RANGE_DOMAIN);
    digest.update(
        u64::try_from(first.history_ordinal)
            .map_err(|_| RouterControlError::IntegrityError)?
            .to_be_bytes(),
    );
    digest.update(
        u64::try_from(last.history_ordinal)
            .map_err(|_| RouterControlError::IntegrityError)?
            .to_be_bytes(),
    );
    digest.update(
        decode_sha256(&first.predecessor_chain_hash).ok_or(RouterControlError::IntegrityError)?,
    );
    digest.update(decode_sha256(&last.chain_tip_hash).ok_or(RouterControlError::IntegrityError)?);
    digest.update(
        u32::try_from(receipts.len())
            .map_err(|_| RouterControlError::IntegrityError)?
            .to_be_bytes(),
    );
    for receipt in receipts {
        digest.update(
            u64::try_from(receipt.history_ordinal)
                .map_err(|_| RouterControlError::IntegrityError)?
                .to_be_bytes(),
        );
        digest.update(
            decode_sha256(&receipt.chain_tip_hash).ok_or(RouterControlError::IntegrityError)?,
        );
    }
    digest.update(
        u32::try_from(applied_controls.len())
            .map_err(|_| RouterControlError::IntegrityError)?
            .to_be_bytes(),
    );
    for control in applied_controls {
        digest.update(
            u64::try_from(control.control_generation)
                .map_err(|_| RouterControlError::IntegrityError)?
                .to_be_bytes(),
        );
        digest
            .update(decode_sha256(&control.record_hash).ok_or(RouterControlError::IntegrityError)?);
    }
    Ok(hex_digest(digest.finalize()))
}

fn control_ending_states_hash(
    ending_states: &[CheckpointEndingState],
) -> Result<String, RouterControlError> {
    let mut digest = Sha256::new();
    digest.update(CONTROL_ENDING_STATES_DOMAIN);
    digest.update(
        u32::try_from(ending_states.len())
            .map_err(|_| RouterControlError::IntegrityError)?
            .to_be_bytes(),
    );
    for state in ending_states {
        encode_scope(&mut digest, &state.scope).map_err(|_| RouterControlError::IntegrityError)?;
        digest.update(state.control_generation.to_be_bytes());
        digest.update(state.control_id.as_bytes());
        digest.update([u8::from(state.force_anchor), u8::from(state.paused)]);
        digest.update(decode_sha256(&state.record_hash).ok_or(RouterControlError::IntegrityError)?);
    }
    Ok(hex_digest(digest.finalize()))
}

#[allow(clippy::too_many_arguments)]
fn control_checkpoint_hash(
    level: u32,
    first_history_ordinal: u64,
    last_history_ordinal: u64,
    first_predecessor_hash: &str,
    covered_chain_tip_hash: &str,
    receipt_count: u64,
    applied_control_count: u64,
    range_hash: &str,
    ending_states_hash: &str,
) -> Result<String, RouterControlError> {
    let mut digest = Sha256::new();
    digest.update(CONTROL_CHECKPOINT_DOMAIN);
    digest.update(level.to_be_bytes());
    digest.update(first_history_ordinal.to_be_bytes());
    digest.update(last_history_ordinal.to_be_bytes());
    for hash in [first_predecessor_hash, covered_chain_tip_hash] {
        digest.update(decode_sha256(hash).ok_or(RouterControlError::IntegrityError)?);
    }
    digest.update(receipt_count.to_be_bytes());
    digest.update(applied_control_count.to_be_bytes());
    digest.update(decode_sha256(range_hash).ok_or(RouterControlError::IntegrityError)?);
    digest.update(decode_sha256(ending_states_hash).ok_or(RouterControlError::IntegrityError)?);
    Ok(hex_digest(digest.finalize()))
}

#[allow(clippy::too_many_arguments)]
fn insert_checkpoint(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    level: u32,
    first_history_ordinal: u64,
    last_history_ordinal: u64,
    first_predecessor_hash: &str,
    covered_chain_tip_hash: &str,
    receipt_count: u64,
    applied_control_count: u64,
    range_hash: &str,
    ending_states_json: &str,
    ending_states_hash: &str,
    created_at_unix_ms: i64,
    checkpoint_hash: &str,
) -> Result<(), RouterControlError> {
    let inserted = transaction
        .execute(
            "INSERT INTO control_history_checkpoints (
                checkpoint_hash, project_uuid, level,
                first_history_ordinal, last_history_ordinal,
                first_predecessor_hash, covered_chain_tip_hash,
                receipt_count, applied_control_count, range_hash,
                ending_states_json, ending_states_hash,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7,
                ?8, ?9, ?10, ?11, ?12, ?13, ?1
             )",
            params![
                checkpoint_hash,
                project_uuid.to_string(),
                i64::from(level),
                i64::try_from(first_history_ordinal)
                    .map_err(|_| RouterControlError::CapacityExhausted)?,
                i64::try_from(last_history_ordinal)
                    .map_err(|_| RouterControlError::CapacityExhausted)?,
                first_predecessor_hash,
                covered_chain_tip_hash,
                i64::try_from(receipt_count).map_err(|_| RouterControlError::CapacityExhausted)?,
                i64::try_from(applied_control_count)
                    .map_err(|_| RouterControlError::CapacityExhausted)?,
                range_hash,
                ending_states_json,
                ending_states_hash,
                created_at_unix_ms,
            ],
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    if inserted != 1 {
        return Err(RouterControlError::IntegrityError);
    }
    Ok(())
}

fn delete_compacted_controls(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    last_covered_ordinal: u64,
    cutoff_unix_ms: i64,
) -> Result<(), RouterControlError> {
    let mut statement = transaction
        .prepare(
            "SELECT candidate.control_id
             FROM controls AS candidate
             WHERE candidate.project_uuid = ?1
               AND candidate.control_generation > 0
               AND candidate.originating_history_ordinal <= ?2
               AND candidate.created_at_unix_ms <= ?3
               AND EXISTS (
                    SELECT 1 FROM controls AS newer
                    WHERE newer.project_uuid = candidate.project_uuid
                      AND newer.scope_kind = candidate.scope_kind
                      AND newer.pool_id IS candidate.pool_id
                      AND newer.control_generation > candidate.control_generation
               )
             ORDER BY candidate.control_generation",
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    let control_ids = statement
        .query_map(
            params![
                project_uuid.to_string(),
                i64::try_from(last_covered_ordinal)
                    .map_err(|_| RouterControlError::IntegrityError)?,
                cutoff_unix_ms,
            ],
            |row| row.get::<_, String>(0),
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    drop(statement);
    for control_id in control_ids {
        if transaction
            .execute("DELETE FROM controls WHERE control_id = ?1", [control_id])
            .map_err(|_| RouterControlError::StorageUnavailable)?
            != 1
        {
            return Err(RouterControlError::IntegrityError);
        }
    }
    Ok(())
}

fn load_verified_checkpoints(
    connection: &Connection,
    project_uuid: Uuid,
) -> Result<Vec<VerifiedCheckpoint>, RouterControlError> {
    let mut statement = connection
        .prepare(
            "SELECT checkpoint_hash, project_uuid, level,
                    first_history_ordinal, last_history_ordinal,
                    first_predecessor_hash, covered_chain_tip_hash,
                    receipt_count, applied_control_count, range_hash,
                    ending_states_json, ending_states_hash,
                    created_at_unix_ms, canonical_payload_hash
             FROM control_history_checkpoints
             WHERE project_uuid = ?1
             ORDER BY first_history_ordinal, last_history_ordinal, level, checkpoint_hash",
        )
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    let stored = statement
        .query_map([project_uuid.to_string()], |row| {
            Ok(StoredCheckpoint {
                checkpoint_hash: row.get(0)?,
                project_uuid: row.get(1)?,
                level: row.get(2)?,
                first_history_ordinal: row.get(3)?,
                last_history_ordinal: row.get(4)?,
                first_predecessor_hash: row.get(5)?,
                covered_chain_tip_hash: row.get(6)?,
                receipt_count: row.get(7)?,
                applied_control_count: row.get(8)?,
                range_hash: row.get(9)?,
                ending_states_json: row.get(10)?,
                ending_states_hash: row.get(11)?,
                created_at_unix_ms: row.get(12)?,
                canonical_payload_hash: row.get(13)?,
            })
        })
        .map_err(|_| RouterControlError::StorageUnavailable)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|_| RouterControlError::StorageUnavailable)?;
    if stored.len() > CONTROL_CHECKPOINT_MAX {
        return Err(RouterControlError::IntegrityError);
    }
    let mut verified = Vec::with_capacity(stored.len());
    let mut expected_ordinal = 1_u64;
    let mut expected_predecessor = load_control_by_generation(connection, project_uuid, 0)
        .map_err(|_| RouterControlError::IntegrityError)?
        .ok_or(RouterControlError::IntegrityError)?
        .record_hash;
    for checkpoint in stored {
        let checkpoint = verify_checkpoint(checkpoint, project_uuid)?;
        if checkpoint.first_history_ordinal != expected_ordinal
            || checkpoint.first_predecessor_hash != expected_predecessor
        {
            return Err(RouterControlError::IntegrityError);
        }
        expected_ordinal = checkpoint
            .last_history_ordinal
            .checked_add(1)
            .ok_or(RouterControlError::IntegrityError)?;
        expected_predecessor.clone_from(&checkpoint.covered_chain_tip_hash);
        verified.push(checkpoint);
    }
    Ok(verified)
}

fn verify_checkpoint(
    stored: StoredCheckpoint,
    project_uuid: Uuid,
) -> Result<VerifiedCheckpoint, RouterControlError> {
    let level = u32::try_from(stored.level).map_err(|_| RouterControlError::IntegrityError)?;
    let first = u64::try_from(stored.first_history_ordinal)
        .map_err(|_| RouterControlError::IntegrityError)?;
    let last = u64::try_from(stored.last_history_ordinal)
        .map_err(|_| RouterControlError::IntegrityError)?;
    let receipt_count =
        u64::try_from(stored.receipt_count).map_err(|_| RouterControlError::IntegrityError)?;
    let applied_control_count = u64::try_from(stored.applied_control_count)
        .map_err(|_| RouterControlError::IntegrityError)?;
    let expected_span = last
        .checked_sub(first)
        .and_then(|span| span.checked_add(1))
        .ok_or(RouterControlError::IntegrityError)?;
    let ending_states: Vec<CheckpointEndingState> =
        serde_json::from_str(&stored.ending_states_json)
            .map_err(|_| RouterControlError::IntegrityError)?;
    if stored.project_uuid != project_uuid.to_string()
        || stored.created_at_unix_ms < 0
        || receipt_count != expected_span
        || canonical_ending_states_json(&ending_states)? != stored.ending_states_json
        || control_ending_states_hash(&ending_states)? != stored.ending_states_hash
        || stored.canonical_payload_hash != stored.checkpoint_hash
    {
        return Err(RouterControlError::IntegrityError);
    }
    let mut previous_key = None;
    for state in &ending_states {
        if !is_sha256(&state.record_hash) {
            return Err(RouterControlError::IntegrityError);
        }
        let key = control_key(&state.scope);
        if previous_key
            .as_ref()
            .is_some_and(|previous| previous >= &key)
        {
            return Err(RouterControlError::IntegrityError);
        }
        previous_key = Some(key);
    }
    let expected_hash = control_checkpoint_hash(
        level,
        first,
        last,
        &stored.first_predecessor_hash,
        &stored.covered_chain_tip_hash,
        receipt_count,
        applied_control_count,
        &stored.range_hash,
        &stored.ending_states_hash,
    )?;
    if expected_hash != stored.checkpoint_hash
        || ending_states.len() as u64 > applied_control_count
        || (applied_control_count == 0) != ending_states.is_empty()
    {
        return Err(RouterControlError::IntegrityError);
    }
    Ok(VerifiedCheckpoint {
        checkpoint_hash: stored.checkpoint_hash,
        level,
        first_history_ordinal: first,
        last_history_ordinal: last,
        first_predecessor_hash: stored.first_predecessor_hash,
        covered_chain_tip_hash: stored.covered_chain_tip_hash,
        receipt_count,
        applied_control_count,
        ending_states,
    })
}

fn merge_oldest_checkpoints(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    checkpoints: &[VerifiedCheckpoint],
    created_at_unix_ms: i64,
) -> Result<(), RouterControlError> {
    let children = checkpoints
        .get(..CONTROL_CHECKPOINT_MERGE_COUNT)
        .ok_or(RouterControlError::IntegrityError)?;
    let first = children.first().ok_or(RouterControlError::IntegrityError)?;
    let last = children.last().ok_or(RouterControlError::IntegrityError)?;
    let level = children
        .iter()
        .map(|child| child.level)
        .max()
        .and_then(|level| level.checked_add(1))
        .filter(|level| *level <= i32::MAX as u32)
        .ok_or(RouterControlError::CapacityExhausted)?;
    let receipt_count = children
        .iter()
        .try_fold(0_u64, |total, child| total.checked_add(child.receipt_count))
        .ok_or(RouterControlError::CapacityExhausted)?;
    let applied_control_count = children
        .iter()
        .try_fold(0_u64, |total, child| {
            total.checked_add(child.applied_control_count)
        })
        .ok_or(RouterControlError::CapacityExhausted)?;
    let mut ending_by_scope = BTreeMap::new();
    for child in children {
        for state in &child.ending_states {
            ending_by_scope.insert(control_key(&state.scope), state.clone());
        }
    }
    let ending_states = ending_by_scope.into_values().collect::<Vec<_>>();
    let ending_states_json = canonical_ending_states_json(&ending_states)?;
    let ending_states_hash = control_ending_states_hash(&ending_states)?;
    let range_hash = control_checkpoint_merge_hash(children)?;
    let checkpoint_hash = control_checkpoint_hash(
        level,
        first.first_history_ordinal,
        last.last_history_ordinal,
        &first.first_predecessor_hash,
        &last.covered_chain_tip_hash,
        receipt_count,
        applied_control_count,
        &range_hash,
        &ending_states_hash,
    )?;

    for child in children {
        if transaction
            .execute(
                "DELETE FROM control_history_checkpoints WHERE checkpoint_hash = ?1",
                [&child.checkpoint_hash],
            )
            .map_err(|_| RouterControlError::StorageUnavailable)?
            != 1
        {
            return Err(RouterControlError::IntegrityError);
        }
    }
    insert_checkpoint(
        transaction,
        project_uuid,
        level,
        first.first_history_ordinal,
        last.last_history_ordinal,
        &first.first_predecessor_hash,
        &last.covered_chain_tip_hash,
        receipt_count,
        applied_control_count,
        &range_hash,
        &ending_states_json,
        &ending_states_hash,
        created_at_unix_ms,
        &checkpoint_hash,
    )
}

fn control_checkpoint_merge_hash(
    children: &[VerifiedCheckpoint],
) -> Result<String, RouterControlError> {
    if children.len() != CONTROL_CHECKPOINT_MERGE_COUNT {
        return Err(RouterControlError::IntegrityError);
    }
    let mut digest = Sha256::new();
    digest.update(CONTROL_CHECKPOINT_MERGE_DOMAIN);
    digest.update(
        u32::try_from(children.len())
            .map_err(|_| RouterControlError::IntegrityError)?
            .to_be_bytes(),
    );
    for child in children {
        digest.update(child.level.to_be_bytes());
        digest.update(child.first_history_ordinal.to_be_bytes());
        digest.update(child.last_history_ordinal.to_be_bytes());
        digest.update(
            decode_sha256(&child.checkpoint_hash).ok_or(RouterControlError::IntegrityError)?,
        );
    }
    Ok(hex_digest(digest.finalize()))
}

#[allow(clippy::too_many_arguments)]
fn control_history_entry_hash(
    history_ordinal: u64,
    predecessor_chain_hash: &str,
    project_uuid: Uuid,
    mutation: &ControlMutation,
    canonical_payload_hash: &str,
    result: ControlMutationResult,
    result_control_generation: u64,
    applied_control: Option<(Uuid, &str)>,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    let predecessor = decode_sha256(predecessor_chain_hash)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let payload = decode_sha256(canonical_payload_hash)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let created_at = u64::try_from(created_at_unix_ms)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let mut digest = Sha256::new();
    digest.update(CONTROL_HISTORY_ENTRY_DOMAIN);
    digest.update(history_ordinal.to_be_bytes());
    digest.update(predecessor);
    digest.update(project_uuid.as_bytes());
    digest.update(mutation.mutation_id.as_bytes());
    digest.update(payload);
    digest.update([result_tag(result)]);
    digest.update(result_control_generation.to_be_bytes());
    match applied_control {
        Some((control_id, record_hash)) => {
            digest.update([1]);
            digest.update(control_id.as_bytes());
            digest.update(
                decode_sha256(record_hash)
                    .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
            );
        }
        None => digest.update([0]),
    }
    encode_scope(&mut digest, &mutation.scope)?;
    encode_operation(&mut digest, &mutation.operation);
    digest.update(mutation.expected_control_generation.to_be_bytes());
    encode_string(&mut digest, &mutation.actor)?;
    encode_string(&mut digest, &mutation.reason)?;
    digest.update(process_instance_id.as_bytes());
    digest.update(created_at.to_be_bytes());
    Ok(hex_digest(digest.finalize()))
}

fn scope_columns(scope: &ControlScope) -> (&'static str, Option<&str>) {
    match scope {
        ControlScope::All => ("all", None),
        ControlScope::Pool { pool_id } => ("pool", Some(pool_id)),
    }
}

fn operation_columns(operation: &ControlOperation) -> (&'static str, bool) {
    match operation {
        ControlOperation::SetForceAnchor { value } => ("set_force_anchor", *value),
        ControlOperation::SetPaused { value } => ("set_paused", *value),
    }
}

fn parse_scope(kind: &str, pool_id: Option<&str>) -> Result<ControlScope, RouterControlError> {
    match (kind, pool_id) {
        ("all", None) => Ok(ControlScope::All),
        ("pool", Some(pool_id)) if !pool_id.is_empty() && pool_id.len() <= 128 => {
            Ok(ControlScope::Pool {
                pool_id: pool_id.to_string(),
            })
        }
        _ => Err(RouterControlError::IntegrityError),
    }
}

fn parse_operation(kind: &str, value: i64) -> Result<ControlOperation, RouterControlError> {
    let value = parse_bool(value).map_err(|_| RouterControlError::IntegrityError)?;
    match kind {
        "set_force_anchor" => Ok(ControlOperation::SetForceAnchor { value }),
        "set_paused" => Ok(ControlOperation::SetPaused { value }),
        _ => Err(RouterControlError::IntegrityError),
    }
}

fn parse_result(value: &str) -> Result<ControlMutationResult, RouterControlError> {
    match value {
        "applied" => Ok(ControlMutationResult::Applied),
        "no_op" => Ok(ControlMutationResult::NoOp),
        "conflict" => Ok(ControlMutationResult::Conflict),
        _ => Err(RouterControlError::IntegrityError),
    }
}

const fn result_name(value: ControlMutationResult) -> &'static str {
    match value {
        ControlMutationResult::Applied => "applied",
        ControlMutationResult::NoOp => "no_op",
        ControlMutationResult::Conflict => "conflict",
    }
}

const fn result_tag(value: ControlMutationResult) -> u8 {
    match value {
        ControlMutationResult::Applied => 0,
        ControlMutationResult::NoOp => 1,
        ControlMutationResult::Conflict => 2,
    }
}

fn encode_operation(digest: &mut Sha256, operation: &ControlOperation) {
    match operation {
        ControlOperation::SetForceAnchor { value } => {
            digest.update([0, u8::from(*value)]);
        }
        ControlOperation::SetPaused { value } => {
            digest.update([1, u8::from(*value)]);
        }
    }
}

fn ensure_writer_capability(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    verified_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let payload_hash =
        writer_capability_hash(project_uuid, process_instance_id, verified_at_unix_ms)?;
    let inserted = transaction
        .execute(
            "INSERT INTO process_writer_capabilities (
                process_instance_id, project_uuid, writer_protocol,
                schema_version, verified_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)
             ON CONFLICT(process_instance_id, writer_protocol) DO NOTHING",
            params![
                process_instance_id.to_string(),
                project_uuid.to_string(),
                ACTIVE_WRITER_PROTOCOL,
                ACTIVE_WRITER_SCHEMA_VERSION,
                verified_at_unix_ms,
                payload_hash,
            ],
        )
        .map_err(database_error)?;
    if inserted == 1 {
        return Ok(());
    }
    let stored = transaction
        .query_row(
            "SELECT project_uuid, schema_version, verified_at_unix_ms,
                    canonical_payload_hash
             FROM process_writer_capabilities
             WHERE process_instance_id = ?1 AND writer_protocol = ?2",
            params![process_instance_id.to_string(), ACTIVE_WRITER_PROTOCOL],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let expected = writer_capability_hash(project_uuid, process_instance_id, stored.2)?;
    if stored.0 != project_uuid.to_string()
        || stored.1 != ACTIVE_WRITER_SCHEMA_VERSION
        || stored.2 < 0
        || stored.3 != expected
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn ensure_current_config_event(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    config_generation_id: &str,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let events = load_verified_config_events(transaction, project_uuid)?;
    if events
        .last()
        .is_some_and(|event| event.config_generation_id == config_generation_id)
    {
        return Ok(());
    }
    if events
        .iter()
        .any(|event| event.config_generation_id == config_generation_id)
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let epoch = i64::try_from(events.len())
        .ok()
        .and_then(|value| value.checked_add(1))
        .filter(|value| *value > 0)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let predecessor = events.last().map(|event| event.event_hash.as_str());
    let event_id = Uuid::now_v7();
    let event_hash = current_config_event_hash(
        u64::try_from(epoch).map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
        project_uuid,
        config_generation_id,
        predecessor,
        process_instance_id,
        created_at_unix_ms,
    )?;
    let inserted = transaction
        .execute(
            "INSERT INTO config_generation_state_events (
                config_generation_state_event_id, config_epoch, project_uuid,
                config_generation_id, predecessor_event_hash,
                process_instance_id, created_at_unix_ms,
                event_hash, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?8)",
            params![
                event_id.to_string(),
                epoch,
                project_uuid.to_string(),
                config_generation_id,
                predecessor,
                process_instance_id.to_string(),
                created_at_unix_ms,
                event_hash,
            ],
        )
        .map_err(database_error)?;
    if inserted != 1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn load_verified_config_events(
    connection: &Connection,
    project_uuid: Uuid,
) -> Result<Vec<StoredConfigEvent>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT config_generation_state_event_id, config_epoch, project_uuid,
                    config_generation_id, predecessor_event_hash,
                    process_instance_id, created_at_unix_ms,
                    event_hash, canonical_payload_hash
             FROM config_generation_state_events
             WHERE project_uuid = ?1 ORDER BY config_epoch",
        )
        .map_err(database_error)?;
    let events = statement
        .query_map([project_uuid.to_string()], |row| {
            Ok(StoredConfigEvent {
                event_id: row.get(0)?,
                epoch: row.get(1)?,
                project_uuid: row.get(2)?,
                config_generation_id: row.get(3)?,
                predecessor_event_hash: row.get(4)?,
                process_instance_id: row.get(5)?,
                created_at_unix_ms: row.get(6)?,
                event_hash: row.get(7)?,
                canonical_payload_hash: row.get(8)?,
            })
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    let mut predecessor: Option<&str> = None;
    for (index, event) in events.iter().enumerate() {
        let event_id = parse_uuid_v7(&event.event_id)?;
        let event_project_uuid = parse_uuid(&event.project_uuid)?;
        let process_instance_id = parse_uuid_v7(&event.process_instance_id)?;
        let expected_epoch = i64::try_from(index)
            .ok()
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let expected_hash = current_config_event_hash(
            u64::try_from(event.epoch)
                .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
            event_project_uuid,
            &event.config_generation_id,
            predecessor,
            process_instance_id,
            event.created_at_unix_ms,
        )?;
        if event_id.to_string() != event.event_id
            || event_project_uuid != project_uuid
            || event.epoch != expected_epoch
            || event.predecessor_event_hash.as_deref() != predecessor
            || event.created_at_unix_ms < 0
            || event.event_hash != expected_hash
            || event.canonical_payload_hash != expected_hash
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        predecessor = Some(&event.event_hash);
    }
    Ok(events)
}

fn ensure_genesis_control(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let stored = load_control_by_generation(transaction, project_uuid, 0)?;
    if let Some(stored) = stored {
        verify_stored_control(&stored, project_uuid)?;
        if stored.control_id != Uuid::nil().to_string()
            || stored.originating_history_ordinal != 0
            || stored.scope_kind != "all"
            || stored.pool_id.is_some()
            || stored.force_anchor != 0
            || stored.paused != 0
            || stored.actor != INITIAL_CONTROL_ACTOR
            || stored.reason != INITIAL_CONTROL_REASON
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        return Ok(());
    }

    let record_hash = control_record_hash(
        Uuid::nil(),
        0,
        0,
        project_uuid,
        &ControlScope::All,
        false,
        false,
        INITIAL_CONTROL_ACTOR,
        INITIAL_CONTROL_REASON,
        process_instance_id,
        created_at_unix_ms,
    )?;
    let inserted = transaction
        .execute(
            "INSERT INTO controls (
                control_id, control_generation, originating_history_ordinal,
                project_uuid, scope_kind, pool_id, force_anchor, paused,
                actor, reason, process_instance_id, created_at_unix_ms,
                record_hash, canonical_payload_hash
             ) VALUES (?1, 0, 0, ?2, 'all', NULL, 0, 0, ?3, ?4, ?5, ?6, ?7, ?7)",
            params![
                Uuid::nil().to_string(),
                project_uuid.to_string(),
                INITIAL_CONTROL_ACTOR,
                INITIAL_CONTROL_REASON,
                process_instance_id.to_string(),
                created_at_unix_ms,
                record_hash,
            ],
        )
        .map_err(database_error)?;
    if inserted != 1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum ControlKey {
    All,
    Pool(String),
}

fn load_latest_controls(
    connection: &Connection,
    project_uuid: Uuid,
) -> Result<BTreeMap<ControlKey, StoredControl>, LedgerError> {
    let genesis = load_control_by_generation(connection, project_uuid, 0)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    verify_stored_control(&genesis, project_uuid)?;
    if genesis.control_id != Uuid::nil().to_string()
        || genesis.originating_history_ordinal != 0
        || genesis.scope_kind != "all"
        || genesis.pool_id.is_some()
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let mut statement = connection
        .prepare(
            "SELECT control_id, control_generation, originating_history_ordinal,
                    project_uuid, scope_kind, pool_id, force_anchor, paused,
                    actor, reason, process_instance_id, created_at_unix_ms,
                    record_hash, canonical_payload_hash
             FROM controls AS candidate
             WHERE candidate.project_uuid = ?1
               AND NOT EXISTS (
                    SELECT 1 FROM controls AS newer
                    WHERE newer.project_uuid = candidate.project_uuid
                      AND newer.scope_kind = candidate.scope_kind
                      AND newer.pool_id IS candidate.pool_id
                      AND newer.control_generation > candidate.control_generation
               )
             ORDER BY control_generation",
        )
        .map_err(database_error)?;
    let stored = statement
        .query_map([project_uuid.to_string()], stored_control_from_row)
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    let mut latest = BTreeMap::new();
    for control in stored {
        verify_stored_control(&control, project_uuid)?;
        let key = match control.scope_kind.as_str() {
            "all" if control.pool_id.is_none() => ControlKey::All,
            "pool" => ControlKey::Pool(
                control
                    .pool_id
                    .clone()
                    .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
            ),
            _ => return Err(LedgerErrorClass::IdentityInvariant.into()),
        };
        if latest.insert(key, control).is_some() {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
    }
    if !latest.contains_key(&ControlKey::All) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(latest)
}

fn load_control_by_generation(
    connection: &Connection,
    project_uuid: Uuid,
    generation: i64,
) -> Result<Option<StoredControl>, LedgerError> {
    connection
        .query_row(
            "SELECT control_id, control_generation, originating_history_ordinal,
                    project_uuid, scope_kind, pool_id, force_anchor, paused,
                    actor, reason, process_instance_id, created_at_unix_ms,
                    record_hash, canonical_payload_hash
             FROM controls
             WHERE project_uuid = ?1 AND control_generation = ?2",
            params![project_uuid.to_string(), generation],
            stored_control_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn stored_control_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredControl> {
    Ok(StoredControl {
        control_id: row.get(0)?,
        control_generation: row.get(1)?,
        originating_history_ordinal: row.get(2)?,
        project_uuid: row.get(3)?,
        scope_kind: row.get(4)?,
        pool_id: row.get(5)?,
        force_anchor: row.get(6)?,
        paused: row.get(7)?,
        actor: row.get(8)?,
        reason: row.get(9)?,
        process_instance_id: row.get(10)?,
        created_at_unix_ms: row.get(11)?,
        record_hash: row.get(12)?,
        canonical_payload_hash: row.get(13)?,
    })
}

fn verify_stored_control(
    stored: &StoredControl,
    expected_project_uuid: Uuid,
) -> Result<(), LedgerError> {
    let control_id = parse_uuid(&stored.control_id)?;
    let project_uuid = parse_uuid(&stored.project_uuid)?;
    let process_instance_id = parse_uuid_v7(&stored.process_instance_id)?;
    let generation = u64::try_from(stored.control_generation)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let history_ordinal = u64::try_from(stored.originating_history_ordinal)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let scope = match stored.scope_kind.as_str() {
        "all" if stored.pool_id.is_none() => ControlScope::All,
        "pool" => ControlScope::Pool {
            pool_id: stored
                .pool_id
                .clone()
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
        },
        _ => return Err(LedgerErrorClass::IdentityInvariant.into()),
    };
    let force_anchor = parse_bool(stored.force_anchor)?;
    let paused = parse_bool(stored.paused)?;
    let expected_hash = control_record_hash(
        control_id,
        generation,
        history_ordinal,
        project_uuid,
        &scope,
        force_anchor,
        paused,
        &stored.actor,
        &stored.reason,
        process_instance_id,
        stored.created_at_unix_ms,
    )?;
    if project_uuid != expected_project_uuid
        || stored.created_at_unix_ms < 0
        || stored.record_hash != expected_hash
        || stored.canonical_payload_hash != expected_hash
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn load_current_learning_generation(
    connection: &Connection,
    project_uuid: Uuid,
    pool_id: &str,
) -> Result<Uuid, LedgerError> {
    let value = connection
        .query_row(
            "SELECT learning_generation_id
             FROM learning_generation_state_events
             WHERE project_uuid = ?1 AND pool_id = ?2
             ORDER BY event_seq DESC LIMIT 1",
            params![project_uuid.to_string(), pool_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    parse_uuid_v7(&value)
}

fn current_config_event_hash(
    epoch: u64,
    project_uuid: Uuid,
    config_generation_id: &str,
    predecessor_event_hash: Option<&str>,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    let config_hash = decode_sha256(config_generation_id)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let predecessor_hash = predecessor_event_hash
        .map(|value| {
            decode_sha256(value)
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))
        })
        .transpose()?;
    let created_at = u64::try_from(created_at_unix_ms)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let mut digest = Sha256::new();
    digest.update(CURRENT_CONFIG_EVENT_DOMAIN);
    digest.update(epoch.to_be_bytes());
    digest.update(project_uuid.as_bytes());
    digest.update(config_hash);
    match predecessor_hash {
        Some(hash) => {
            digest.update([1]);
            digest.update(hash);
        }
        None => digest.update([0]),
    }
    digest.update(process_instance_id.as_bytes());
    digest.update(created_at.to_be_bytes());
    Ok(hex_digest(digest.finalize()))
}

fn writer_capability_hash(
    project_uuid: Uuid,
    process_instance_id: Uuid,
    verified_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    let verified_at = u64::try_from(verified_at_unix_ms)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let mut digest = Sha256::new();
    digest.update(WRITER_CAPABILITY_DOMAIN);
    digest.update(project_uuid.as_bytes());
    digest.update(process_instance_id.as_bytes());
    encode_string(&mut digest, ACTIVE_WRITER_PROTOCOL)?;
    digest.update((ACTIVE_WRITER_SCHEMA_VERSION as u64).to_be_bytes());
    digest.update(verified_at.to_be_bytes());
    Ok(hex_digest(digest.finalize()))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn control_record_hash(
    control_id: Uuid,
    control_generation: u64,
    originating_history_ordinal: u64,
    project_uuid: Uuid,
    scope: &ControlScope,
    force_anchor: bool,
    paused: bool,
    actor: &str,
    reason: &str,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    let created_at = u64::try_from(created_at_unix_ms)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let mut digest = Sha256::new();
    digest.update(CONTROL_RECORD_DOMAIN);
    digest.update(control_id.as_bytes());
    digest.update(control_generation.to_be_bytes());
    digest.update(originating_history_ordinal.to_be_bytes());
    digest.update(project_uuid.as_bytes());
    encode_scope(&mut digest, scope)?;
    digest.update([u8::from(force_anchor), u8::from(paused)]);
    encode_string(&mut digest, actor)?;
    encode_string(&mut digest, reason)?;
    digest.update(process_instance_id.as_bytes());
    digest.update(created_at.to_be_bytes());
    Ok(hex_digest(digest.finalize()))
}

fn encode_scope(digest: &mut Sha256, scope: &ControlScope) -> Result<(), LedgerError> {
    match scope {
        ControlScope::All => digest.update([0]),
        ControlScope::Pool { pool_id } => {
            digest.update([1]);
            encode_string(digest, pool_id)?;
        }
    }
    Ok(())
}

fn encode_string(digest: &mut Sha256, value: &str) -> Result<(), LedgerError> {
    let length = u32::try_from(value.len())
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    digest.update(length.to_be_bytes());
    digest.update(value.as_bytes());
    Ok(())
}

fn decode_sha256(value: &str) -> Option<[u8; 32]> {
    if !is_sha256(value) {
        return None;
    }
    let mut bytes = [0_u8; 32];
    for (index, pair) in value.as_bytes().chunks_exact(2).enumerate() {
        bytes[index] = decode_nibble(pair[0])? << 4 | decode_nibble(pair[1])?;
    }
    Some(bytes)
}

fn decode_nibble(value: u8) -> Option<u8> {
    match value {
        b'0'..=b'9' => Some(value - b'0'),
        b'a'..=b'f' => Some(value - b'a' + 10),
        _ => None,
    }
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn parse_bool(value: i64) -> Result<bool, LedgerError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(LedgerErrorClass::IdentityInvariant.into()),
    }
}

fn parse_uuid(value: &str) -> Result<Uuid, LedgerError> {
    let parsed = Uuid::parse_str(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if parsed.to_string() != value
        || (parsed != Uuid::nil() && parsed.get_variant() != Variant::RFC4122)
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(parsed)
}

fn parse_uuid_v7(value: &str) -> Result<Uuid, LedgerError> {
    let parsed = parse_uuid(value)?;
    if parsed.get_version_num() != 7 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(parsed)
}

fn database_error(_error: rusqlite::Error) -> LedgerError {
    LedgerErrorClass::DatabaseOperationFailed.into()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use tempfile::tempdir;

    use super::*;
    use crate::ledger::repository::LedgerRepository;

    fn mutation(
        mutation_id: Uuid,
        operation: ControlOperation,
        expected_control_generation: u64,
    ) -> ControlMutation {
        ControlMutation {
            mutation_id,
            scope: ControlScope::All,
            operation,
            expected_control_generation,
            actor: "operator-a".into(),
            reason: "control test".into(),
        }
    }

    fn apply(
        repository: &mut LedgerRepository,
        mutation: ControlMutation,
    ) -> Result<ControlMutationAck, RouterControlError> {
        let prepared = prepare_control_mutation(mutation)?;
        let fence = Arc::new(ControlTransactionFence::new(
            Instant::now() + Duration::from_secs(1),
        ));
        repository.apply_control_mutation_with_start_check(&prepared, &fence, || Some(()))
    }

    fn apply_at(
        repository: &mut LedgerRepository,
        mutation: ControlMutation,
        created_at_unix_ms: i64,
    ) -> Result<ControlMutationAck, RouterControlError> {
        let prepared = prepare_control_mutation(mutation)?;
        let project_uuid = repository.project_uuid;
        let process_instance_id = repository.process_instance_id;
        let config_generation_id = repository.config_generation_id.clone();
        let pool_ids = repository
            .active_pool_ids
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let retention_days = repository.retention_days;
        let transaction = repository
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|_| RouterControlError::StorageUnavailable)?;
        let result = apply_control_mutation_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            &config_generation_id,
            &pool_ids,
            retention_days,
            &prepared,
            created_at_unix_ms,
        )?;
        transaction
            .commit()
            .map_err(|_| RouterControlError::StorageUnavailable)?;
        Ok(result)
    }

    fn uuid_v7_at(unix_ms: u64, discriminator: u64) -> Uuid {
        assert!(unix_ms < (1_u64 << 48));
        let timestamp = unix_ms.to_be_bytes();
        let random = discriminator.to_be_bytes();
        let mut bytes = [0_u8; 16];
        bytes[..6].copy_from_slice(&timestamp[2..]);
        bytes[6] = 0x70 | (random[0] & 0x0f);
        bytes[7] = random[1];
        bytes[8] = 0x80 | (random[2] & 0x3f);
        bytes[9..].copy_from_slice(&random[1..]);
        Uuid::from_bytes(bytes)
    }

    #[test]
    fn fixed_hash_encodings_are_stable_and_domain_separated() {
        let project = Uuid::parse_str("018f0000-0000-7000-8000-000000000001").unwrap();
        let process = Uuid::parse_str("018f0000-0000-7000-8000-000000000002").unwrap();
        let config = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
        assert_eq!(
            current_config_event_hash(1, project, config, None, process, 7).unwrap(),
            "f1e9c58fb8e483f703ab5d61c8693475daa0989e8f5a4d6668f34088558feea0"
        );
        assert_eq!(
            control_record_hash(
                Uuid::nil(),
                0,
                0,
                project,
                &ControlScope::All,
                false,
                false,
                INITIAL_CONTROL_ACTOR,
                INITIAL_CONTROL_REASON,
                process,
                7,
            )
            .unwrap(),
            "c86166808f21a75851c1804cc24181df47f057062b91bb752dc33e37fef1ab49"
        );
        assert_ne!(
            writer_capability_hash(project, process, 7).unwrap(),
            current_config_event_hash(1, project, config, None, process, 7).unwrap()
        );

        let ending_states_hash = control_ending_states_hash(&[]).unwrap();
        assert_eq!(
            ending_states_hash,
            "efb2f875d7218bd694af6291a28c6d52575f90c86be30c331ba2086905ea0831"
        );
        let receipt = StoredReceipt {
            mutation_id: Uuid::now_v7().to_string(),
            canonical_payload_hash: "22".repeat(32),
            history_ordinal: 1,
            predecessor_chain_hash: "00".repeat(32),
            chain_tip_hash: "11".repeat(32),
            project_uuid: project.to_string(),
            result: "no_op".into(),
            result_control_generation: 0,
            control_id: None,
            control_record_hash: None,
            scope_kind: "all".into(),
            pool_id: None,
            operation_kind: "set_paused".into(),
            requested_value: 0,
            expected_control_generation: 0,
            actor: "operator".into(),
            reason: "golden".into(),
            process_instance_id: process.to_string(),
            created_at_unix_ms: 7,
        };
        let range_hash = control_history_range_hash(&[receipt], &[]).unwrap();
        assert_eq!(
            range_hash,
            "3e63344438c0b0df0c250b9ab2281be2a9250f9230df9e6c27a829876907a63f"
        );
        assert_eq!(
            control_checkpoint_hash(
                0,
                1,
                1,
                &"00".repeat(32),
                &"11".repeat(32),
                1,
                0,
                &range_hash,
                &ending_states_hash,
            )
            .unwrap(),
            "addd098aeca43cdcde88fe62deda09340be5ede3bdaee85d65cdd1605a61bb00"
        );
        let children = (1..=CONTROL_CHECKPOINT_MERGE_COUNT)
            .map(|ordinal| VerifiedCheckpoint {
                checkpoint_hash: hex_digest(Sha256::digest(format!("child-{ordinal}").as_bytes())),
                level: 0,
                first_history_ordinal: ordinal as u64,
                last_history_ordinal: ordinal as u64,
                first_predecessor_hash: "00".repeat(32),
                covered_chain_tip_hash: "11".repeat(32),
                receipt_count: 1,
                applied_control_count: 0,
                ending_states: Vec::new(),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            control_checkpoint_merge_hash(&children).unwrap(),
            "93673f9a00d76e36b90991d4f43ba0da62229203784cd14df12d11d18becd0ab"
        );
    }

    #[test]
    fn mutations_apply_no_op_conflict_and_replay_without_duplicate_rows() {
        let directory = tempdir().unwrap();
        let path = crate::ledger::repository::tests::database_path(&directory);
        let config = crate::ledger::repository::tests::config(&path, "control-replay");
        let mut activated = LedgerRepository::activate(&config).unwrap();

        let applied = mutation(
            Uuid::now_v7(),
            ControlOperation::SetPaused { value: true },
            0,
        );
        assert!(matches!(
            apply(&mut activated.repository, applied.clone()).unwrap(),
            ControlMutationAck::Completed {
                result: ControlMutationResult::Applied,
                ..
            }
        ));
        assert!(matches!(
            apply(&mut activated.repository, applied.clone()).unwrap(),
            ControlMutationAck::Completed {
                result: ControlMutationResult::Applied,
                ..
            }
        ));

        let no_op = mutation(
            Uuid::now_v7(),
            ControlOperation::SetPaused { value: true },
            1,
        );
        assert!(matches!(
            apply(&mut activated.repository, no_op.clone()).unwrap(),
            ControlMutationAck::Completed {
                result: ControlMutationResult::NoOp,
                ..
            }
        ));
        assert!(matches!(
            apply(&mut activated.repository, no_op).unwrap(),
            ControlMutationAck::Completed {
                result: ControlMutationResult::NoOp,
                ..
            }
        ));

        let conflict = mutation(
            Uuid::now_v7(),
            ControlOperation::SetForceAnchor { value: true },
            0,
        );
        let conflict_id = conflict.mutation_id;
        assert!(matches!(
            apply(&mut activated.repository, conflict.clone()).unwrap(),
            ControlMutationAck::Completed {
                result: ControlMutationResult::Conflict,
                ..
            }
        ));
        assert!(matches!(
            apply(&mut activated.repository, conflict).unwrap(),
            ControlMutationAck::Completed {
                result: ControlMutationResult::Conflict,
                ..
            }
        ));

        let mismatched = ControlMutation {
            actor: "operator-b".into(),
            ..mutation(
                conflict_id,
                ControlOperation::SetForceAnchor { value: true },
                0,
            )
        };
        assert_eq!(
            apply(&mut activated.repository, mismatched).unwrap_err(),
            RouterControlError::InvalidArgument
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM controls", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            2
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM control_mutation_receipts",
                    [],
                    |row| { row.get::<_, i64>(0) }
                )
                .unwrap(),
            3
        );
    }

    #[test]
    fn pool_snapshots_apply_global_or_and_keep_current_learning_generation() {
        let directory = tempdir().unwrap();
        let path = crate::ledger::repository::tests::database_path(&directory);
        let config = crate::ledger::repository::tests::config(&path, "control-pool-or");
        let mut activated = LedgerRepository::activate(&config).unwrap();
        let learning_generation_id = activated.identity.pools["pool-a"].learning_generation_id;
        let pool_mutation = |operation, expected_control_generation| ControlMutation {
            mutation_id: Uuid::now_v7(),
            scope: ControlScope::Pool {
                pool_id: "pool-a".into(),
            },
            operation,
            expected_control_generation,
            actor: "operator-a".into(),
            reason: "pool control test".into(),
        };

        let ControlMutationAck::Completed { snapshot, .. } = apply(
            &mut activated.repository,
            pool_mutation(ControlOperation::SetPaused { value: true }, 0),
        )
        .unwrap() else {
            panic!("pool mutation must begin");
        };
        assert_eq!(snapshot.control_generation, 1);
        assert!(!snapshot.all.paused);
        assert_eq!(
            snapshot.pools["pool-a"],
            RouterPoolControlSnapshot {
                local: RouterControlState {
                    force_anchor: false,
                    paused: true,
                },
                effective: RouterControlState {
                    force_anchor: false,
                    paused: true,
                },
                learning_generation_id,
            }
        );

        let ControlMutationAck::Completed { snapshot, .. } = apply(
            &mut activated.repository,
            mutation(
                Uuid::now_v7(),
                ControlOperation::SetForceAnchor { value: true },
                1,
            ),
        )
        .unwrap() else {
            panic!("global mutation must begin");
        };
        assert!(snapshot.pools["pool-a"].effective.force_anchor);
        assert!(!snapshot.pools["pool-a"].local.force_anchor);

        let ControlMutationAck::Completed {
            result, snapshot, ..
        } = apply(
            &mut activated.repository,
            pool_mutation(ControlOperation::SetForceAnchor { value: false }, 2),
        )
        .unwrap()
        else {
            panic!("pool no-op must begin");
        };
        assert_eq!(result, ControlMutationResult::NoOp);
        assert_eq!(snapshot.control_generation, 2);
        assert!(snapshot.pools["pool-a"].effective.force_anchor);

        let invalid_pool = ControlMutation {
            mutation_id: Uuid::now_v7(),
            scope: ControlScope::Pool {
                pool_id: "missing-pool".into(),
            },
            operation: ControlOperation::SetPaused { value: true },
            expected_control_generation: 2,
            actor: "operator-a".into(),
            reason: "invalid pool".into(),
        };
        assert_eq!(
            apply(&mut activated.repository, invalid_pool).unwrap_err(),
            RouterControlError::InvalidArgument
        );
    }

    #[test]
    fn activation_is_idempotent_advances_config_once_and_rejects_resurrection() {
        let directory = tempdir().unwrap();
        let path = crate::ledger::repository::tests::database_path(&directory);
        let original = crate::ledger::repository::tests::config(&path, "control-activation");
        let first = LedgerRepository::activate(&original).unwrap();
        let original_generation = first.identity.config_generation_id.clone();
        assert_eq!(
            first.control_snapshot.as_ref().unwrap().control_generation,
            0
        );
        let genesis_hash = first
            .repository
            .connection
            .query_row(
                "SELECT record_hash FROM controls WHERE control_generation = 0",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        drop(first);

        let repeated = LedgerRepository::activate(&original).unwrap();
        assert_eq!(repeated.identity.config_generation_id, original_generation);
        assert_eq!(
            repeated
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM config_generation_state_events",
                    [],
                    |row| { row.get::<_, i64>(0) }
                )
                .unwrap(),
            1
        );
        assert_eq!(
            repeated
                .repository
                .connection
                .query_row("SELECT count(*) FROM controls", [], |row| row
                    .get::<_, i64>(0))
                .unwrap(),
            1
        );
        drop(repeated);

        let mut changed = original.clone();
        changed.retention_days += 1;
        let advanced = LedgerRepository::activate(&changed).unwrap();
        assert_ne!(advanced.identity.config_generation_id, original_generation);
        assert_eq!(
            advanced
                .control_snapshot
                .as_ref()
                .unwrap()
                .control_generation,
            0
        );
        assert_eq!(
            advanced
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM config_generation_state_events",
                    [],
                    |row| { row.get::<_, i64>(0) }
                )
                .unwrap(),
            2
        );
        assert_eq!(
            advanced
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM process_writer_capabilities",
                    [],
                    |row| { row.get::<_, i64>(0) }
                )
                .unwrap(),
            3
        );
        assert_eq!(
            advanced
                .repository
                .connection
                .query_row(
                    "SELECT record_hash FROM controls WHERE control_generation = 0",
                    [],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            genesis_hash
        );
        drop(advanced);

        assert_eq!(
            LedgerRepository::activate(&original).err().unwrap().class(),
            LedgerErrorClass::IdentityInvariant
        );
        let connection = Connection::open(&path).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM config_generation_state_events",
                    [],
                    |row| { row.get::<_, i64>(0) }
                )
                .unwrap(),
            2
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM process_writer_capabilities",
                    [],
                    |row| { row.get::<_, i64>(0) }
                )
                .unwrap(),
            3
        );
    }

    #[test]
    fn receipt_replay_precedes_live_writer_capability_fence() {
        let directory = tempdir().unwrap();
        let path = crate::ledger::repository::tests::database_path(&directory);
        let config = crate::ledger::repository::tests::config(&path, "control-capability");
        let mut activated = LedgerRepository::activate(&config).unwrap();
        let replayed = mutation(
            Uuid::now_v7(),
            ControlOperation::SetPaused { value: false },
            0,
        );
        assert!(matches!(
            apply(&mut activated.repository, replayed.clone()).unwrap(),
            ControlMutationAck::Completed {
                result: ControlMutationResult::NoOp,
                ..
            }
        ));
        assert_eq!(
            activated
                .repository
                .connection
                .execute(
                    "DELETE FROM process_writer_capabilities
                     WHERE process_instance_id = ?1",
                    [activated.repository.process_instance_id.to_string()],
                )
                .unwrap(),
            1
        );
        assert!(matches!(
            apply(&mut activated.repository, replayed).unwrap(),
            ControlMutationAck::Completed {
                result: ControlMutationResult::NoOp,
                ..
            }
        ));
        assert_eq!(
            apply(
                &mut activated.repository,
                mutation(
                    Uuid::now_v7(),
                    ControlOperation::SetPaused { value: true },
                    0,
                ),
            )
            .unwrap_err(),
            RouterControlError::MigrationRequired
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM control_mutation_receipts",
                    [],
                    |row| { row.get::<_, i64>(0) }
                )
                .unwrap(),
            1
        );
    }

    #[test]
    fn compaction_checkpoints_old_receipts_preserves_latest_state_and_reopens() {
        let directory = tempdir().unwrap();
        let path = crate::ledger::repository::tests::database_path(&directory);
        let config = crate::ledger::repository::tests::config(&path, "control-compaction");
        let mut activated = LedgerRepository::activate(&config).unwrap();
        let now = Utc::now().timestamp_millis();
        let first_time = now - 62 * MILLIS_PER_DAY;
        let second_time = now - 31 * MILLIS_PER_DAY;
        let first = mutation(
            uuid_v7_at(first_time as u64, 1),
            ControlOperation::SetPaused { value: true },
            0,
        );
        let second = mutation(
            uuid_v7_at(second_time as u64, 2),
            ControlOperation::SetForceAnchor { value: true },
            1,
        );
        assert!(matches!(
            apply_at(&mut activated.repository, first.clone(), first_time).unwrap(),
            ControlMutationAck::Completed {
                result: ControlMutationResult::Applied,
                ..
            }
        ));
        assert!(matches!(
            apply_at(&mut activated.repository, second, second_time).unwrap(),
            ControlMutationAck::Completed {
                result: ControlMutationResult::Applied,
                ..
            }
        ));
        assert!(matches!(
            apply(
                &mut activated.repository,
                mutation(
                    Uuid::now_v7(),
                    ControlOperation::SetForceAnchor { value: true },
                    2,
                ),
            )
            .unwrap(),
            ControlMutationAck::Completed {
                result: ControlMutationResult::NoOp,
                snapshot: RouterControlSnapshot {
                    control_generation: 2,
                    all: RouterControlState {
                        force_anchor: true,
                        paused: true,
                    },
                    ..
                },
                ..
            }
        ));

        let generations = activated
            .repository
            .connection
            .prepare("SELECT control_generation FROM controls ORDER BY control_generation")
            .unwrap()
            .query_map([], |row| row.get::<_, i64>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(generations, vec![0, 2]);
        assert_eq!(
            load_verified_checkpoints(
                &activated.repository.connection,
                activated.repository.project_uuid,
            )
            .unwrap()
            .len(),
            2
        );
        assert_eq!(
            activated
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM control_mutation_receipts",
                    [],
                    |row| { row.get::<_, i64>(0) }
                )
                .unwrap(),
            1
        );
        let history = load_verified_control_history_page(
            &activated.repository.connection,
            activated.repository.project_uuid,
            Utc::now().timestamp_millis(),
            None,
            None,
            10,
        )
        .unwrap();
        assert_eq!(history.maximum_history_ordinal, 3);
        assert_eq!(history.entries.len(), 1);
        assert_eq!(
            history.entries[0].receipt.result,
            ControlMutationResult::NoOp
        );
        assert_eq!(history.entries[0].prior_value, Some(true));
        assert_eq!(history.entries[0].new_value, Some(true));
        assert_eq!(
            apply(&mut activated.repository, first).unwrap_err(),
            RouterControlError::MutationExpired
        );

        drop(activated);
        let reopened = LedgerRepository::activate(&config).unwrap();
        let snapshot = reopened.control_snapshot.unwrap();
        assert_eq!(snapshot.control_generation, 2);
        assert_eq!(
            snapshot.all,
            RouterControlState {
                force_anchor: true,
                paused: true,
            }
        );
    }

    #[test]
    fn fine_compaction_covers_at_most_512_and_stops_at_first_ineligible_receipt() {
        let directory = tempdir().unwrap();
        let path = crate::ledger::repository::tests::database_path(&directory);
        let config = crate::ledger::repository::tests::config(&path, "control-fine-batch");
        let mut activated = LedgerRepository::activate(&config).unwrap();
        let now = Utc::now().timestamp_millis();
        let old_base = now - 31 * MILLIS_PER_DAY;
        let project_uuid = activated.repository.project_uuid;
        let process_instance_id = activated.repository.process_instance_id;
        let config_generation_id = activated.repository.config_generation_id.clone();
        let pool_ids = activated
            .repository
            .active_pool_ids
            .iter()
            .cloned()
            .collect::<Vec<_>>();
        let first_old_id = uuid_v7_at(old_base as u64, 1);
        let transaction = activated
            .repository
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        for index in 0..513_u64 {
            let created_at_unix_ms = old_base + index as i64;
            let prepared = prepare_control_mutation(mutation(
                uuid_v7_at(created_at_unix_ms as u64, index + 1),
                ControlOperation::SetPaused { value: false },
                0,
            ))
            .unwrap();
            assert!(matches!(
                apply_control_mutation_in_transaction(
                    &transaction,
                    project_uuid,
                    process_instance_id,
                    &config_generation_id,
                    &pool_ids,
                    u32::MAX,
                    &prepared,
                    created_at_unix_ms,
                )
                .unwrap(),
                ControlMutationAck::Completed {
                    result: ControlMutationResult::NoOp,
                    ..
                }
            ));
        }
        transaction.commit().unwrap();

        apply(
            &mut activated.repository,
            mutation(
                Uuid::now_v7(),
                ControlOperation::SetPaused { value: false },
                0,
            ),
        )
        .unwrap();
        let checkpoint_counts = activated
            .repository
            .connection
            .prepare(
                "SELECT receipt_count FROM control_history_checkpoints
                 ORDER BY first_history_ordinal",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, i64>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(checkpoint_counts, vec![512]);
        assert_eq!(
            activated
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM control_mutation_receipts",
                    [],
                    |row| { row.get::<_, i64>(0) }
                )
                .unwrap(),
            2
        );

        apply(
            &mut activated.repository,
            mutation(
                Uuid::now_v7(),
                ControlOperation::SetPaused { value: false },
                0,
            ),
        )
        .unwrap();
        let checkpoints = load_verified_checkpoints(
            &activated.repository.connection,
            activated.repository.project_uuid,
        )
        .unwrap();
        assert_eq!(
            checkpoints
                .iter()
                .map(|checkpoint| checkpoint.receipt_count)
                .collect::<Vec<_>>(),
            vec![512, 1]
        );
        assert_eq!(
            apply(
                &mut activated.repository,
                mutation(
                    first_old_id,
                    ControlOperation::SetPaused { value: false },
                    0,
                ),
            )
            .unwrap_err(),
            RouterControlError::MutationExpired
        );
    }

    #[test]
    fn checkpoint_limit_merges_exactly_the_oldest_512_children() {
        let directory = tempdir().unwrap();
        let path = crate::ledger::repository::tests::database_path(&directory);
        let config = crate::ledger::repository::tests::config(&path, "control-merge");
        let mut activated = LedgerRepository::activate(&config).unwrap();
        let project_uuid = activated.repository.project_uuid;
        let now = Utc::now().timestamp_millis();
        let ending_states = Vec::new();
        let ending_states_json = canonical_ending_states_json(&ending_states).unwrap();
        let ending_states_hash = control_ending_states_hash(&ending_states).unwrap();
        let mut predecessor = activated
            .repository
            .connection
            .query_row(
                "SELECT record_hash FROM controls WHERE control_generation = 0",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let transaction = activated
            .repository
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        for ordinal in 1..=CONTROL_CHECKPOINT_MAX as u64 {
            let covered_tip = hex_digest(Sha256::digest(format!("tip-{ordinal}").as_bytes()));
            let range_hash = hex_digest(Sha256::digest(format!("range-{ordinal}").as_bytes()));
            let checkpoint_hash = control_checkpoint_hash(
                0,
                ordinal,
                ordinal,
                &predecessor,
                &covered_tip,
                1,
                0,
                &range_hash,
                &ending_states_hash,
            )
            .unwrap();
            insert_checkpoint(
                &transaction,
                project_uuid,
                0,
                ordinal,
                ordinal,
                &predecessor,
                &covered_tip,
                1,
                0,
                &range_hash,
                &ending_states_json,
                &ending_states_hash,
                now,
                &checkpoint_hash,
            )
            .unwrap();
            predecessor = covered_tip;
        }
        transaction.commit().unwrap();
        let checkpoints =
            load_verified_checkpoints(&activated.repository.connection, project_uuid).unwrap();
        assert_eq!(checkpoints.len(), CONTROL_CHECKPOINT_MAX);

        let transaction = activated
            .repository
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        merge_oldest_checkpoints(&transaction, project_uuid, &checkpoints, now + 1).unwrap();
        transaction.commit().unwrap();
        let merged =
            load_verified_checkpoints(&activated.repository.connection, project_uuid).unwrap();
        assert_eq!(merged.len(), 513);
        assert_eq!(merged[0].level, 1);
        assert_eq!(merged[0].first_history_ordinal, 1);
        assert_eq!(merged[0].last_history_ordinal, 512);
        assert_eq!(merged[0].receipt_count, 512);
        assert_eq!(merged[1].level, 0);
        assert_eq!(merged[1].first_history_ordinal, 513);
        assert_eq!(
            merged[1].first_predecessor_hash,
            merged[0].covered_chain_tip_hash
        );
    }
}
