// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transactional root, dispatch, signal, and outcome persistence for Active routing.

use std::collections::BTreeSet;

use chrono::Utc;
use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use serde_json::{Value as Json, json};
use uuid::{Uuid, Variant};

use super::{LedgerError, LedgerErrorClass, TransactionStartGuard};
use crate::active_math::active_math_algorithm_identity_v1;
use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::config::{
    OUTCOME_DECAY_ID_V1, OUTCOME_REDUCER_ID_V1, RouterConfig, protected_outcome_policy_version_v1,
};
use crate::ledger::repository::active_learning::{
    ActiveNeighborhoodInvalidation, ActiveNeighborhoodMutationAck,
};

pub(crate) const ACTIVE_OPEN_ROOT_WINDOWS_MAX: i64 = 4_096;
pub(crate) const ACTIVE_ROOT_SIGNALS_MAX: usize = 64;
pub(crate) const ACTIVE_ROOT_SIGNAL_BYTES_MAX: usize = 65_536;
const OUTCOME_MATCHER_ALGORITHM_ID_V1: &str = "protected_outcome_matcher_v1";

pub(super) fn ensure_outcome_policy_versions(
    transaction: &Transaction<'_>,
    config: &RouterConfig,
    project_uuid: Uuid,
    config_generation_id: &str,
    policy_versions: &std::collections::BTreeMap<String, String>,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    for pool in &config.pools {
        let Some(policy) = protected_outcome_policy_version_v1(&pool.outcome)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?
        else {
            continue;
        };
        let policy_version_id = policy_versions.get(&pool.id).ok_or_else(corrupt)?;
        let active_math_id = active_math_algorithm_identity_v1();
        let inserted = transaction
            .execute(
                "INSERT INTO outcome_policy_versions (
                    outcome_policy_hash, project_uuid, pool_id,
                    config_generation_id, policy_version_id, canonical_policy_json,
                    matcher_algorithm_id, reducer_algorithm_id, decay_algorithm_id,
                    active_math_algorithm_id_sha256, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?1)
                 ON CONFLICT(outcome_policy_hash) DO NOTHING",
                params![
                    policy.outcome_policy_hash,
                    project_uuid.to_string(),
                    pool.id,
                    config_generation_id,
                    policy_version_id,
                    policy.canonical_policy_json,
                    OUTCOME_MATCHER_ALGORITHM_ID_V1,
                    OUTCOME_REDUCER_ID_V1,
                    OUTCOME_DECAY_ID_V1,
                    active_math_id,
                    created_at_unix_ms,
                ],
            )
            .map_err(database_error)?;
        if inserted == 1 {
            continue;
        }
        let stored = transaction
            .query_row(
                "SELECT project_uuid, pool_id, config_generation_id,
                        policy_version_id, canonical_policy_json,
                        matcher_algorithm_id, reducer_algorithm_id,
                        decay_algorithm_id, active_math_algorithm_id_sha256,
                        canonical_payload_hash
                 FROM outcome_policy_versions WHERE outcome_policy_hash = ?1",
                [policy.outcome_policy_hash.clone()],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, String>(6)?,
                        row.get::<_, String>(7)?,
                        row.get::<_, String>(8)?,
                        row.get::<_, String>(9)?,
                    ))
                },
            )
            .optional()
            .map_err(database_error)?
            .ok_or_else(corrupt)?;
        if stored
            != (
                project_uuid.to_string(),
                pool.id.clone(),
                config_generation_id.to_string(),
                policy_version_id.clone(),
                policy.canonical_policy_json,
                OUTCOME_MATCHER_ALGORITHM_ID_V1.to_string(),
                OUTCOME_REDUCER_ID_V1.to_string(),
                OUTCOME_DECAY_ID_V1.to_string(),
                active_math_id,
                policy.outcome_policy_hash,
            )
        {
            return Err(corrupt());
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveAssignmentArm {
    CandidateTreatment,
    AnchorControl,
    AnchorHoldout,
    NonLearning,
}

impl ActiveAssignmentArm {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::CandidateTreatment => "candidate_treatment",
            Self::AnchorControl => "anchor_control",
            Self::AnchorHoldout => "anchor_holdout",
            Self::NonLearning => "non_learning",
        }
    }

    const fn is_nonholdout(self) -> bool {
        matches!(self, Self::CandidateTreatment | Self::AnchorControl)
    }

    pub(crate) fn matches_decision_reason(self, reason: &str) -> bool {
        match self {
            Self::CandidateTreatment => matches!(reason, "active_candidate"),
            Self::AnchorControl => matches!(reason, "active_anchor_control"),
            Self::AnchorHoldout => matches!(reason, "active_anchor_holdout"),
            Self::NonLearning => matches!(
                reason,
                "active_force_anchor"
                    | "active_paused"
                    | "active_ineligible"
                    | "active_exhausted"
                    | "active_storage_fallback"
                    | "active_authorization_stale"
                    | "active_cap_reached"
            ),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveDispatchAdmission {
    pub(crate) active_dispatch_id: Uuid,
    pub(crate) request_identity_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveRootAdmission {
    pub(crate) active_root_window_id: Uuid,
    pub(crate) active_experiment_id: Uuid,
    pub(crate) active_assignment_id: Uuid,
    pub(crate) decision_id: Uuid,
    pub(crate) dispatch: Option<ActiveDispatchAdmission>,
    pub(crate) pool_id: String,
    pub(crate) candidate_id: String,
    pub(crate) root_key: String,
    pub(crate) owner_relation_hash: String,
    pub(crate) config_generation_id: String,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) cohort_generation_id: Uuid,
    pub(crate) outcome_policy_hash: String,
    pub(crate) opened_after_ingest_seq: u64,
    pub(crate) attribution_deadline_unix_ms: u64,
    pub(crate) tranche_ordinal: u64,
    pub(crate) arm: ActiveAssignmentArm,
    pub(crate) cohort_threshold_numerator: [u8; 8],
    pub(crate) selection_threshold_numerator: [u8; 8],
    pub(crate) configured_holdout_probability_bits: u64,
    pub(crate) configured_canary_probability_bits: u64,
    pub(crate) effective_arm_probability_bits: u64,
    pub(crate) conditional_selection_probability_bits: u64,
    pub(crate) propensity_bits: u64,
    pub(crate) control_generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveAdmissionReceipt {
    pub(crate) total_ordinal: u64,
    pub(crate) cap_ordinal: Option<u64>,
    pub(crate) tranche_total_ordinal: u64,
    pub(crate) tranche_nonholdout_ordinal: Option<u64>,
    pub(crate) admitted_at_unix_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveAdmissionAck {
    Applied(ActiveAdmissionReceipt),
    AlreadyApplied(ActiveAdmissionReceipt),
    RootAlreadyAssigned,
    AuthorityChanged,
    OpenWindowCapacity,
    TrancheFull,
    ExperimentCapReached,
    TransactionNotStarted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveSignalDisposition {
    Success,
    Failure,
    Ignored,
}

impl ActiveSignalDisposition {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Success => "success",
            Self::Failure => "failure",
            Self::Ignored => "ignored",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveProtectedSignal {
    pub(crate) active_root_signal_id: Uuid,
    pub(crate) signal_identity_hash: String,
    pub(crate) event_kind: String,
    pub(crate) scope_phase: String,
    pub(crate) category: String,
    pub(crate) name: String,
    pub(crate) disposition: ActiveSignalDisposition,
    pub(crate) observed_at_unix_ms: u64,
    pub(crate) canonical_signal_json: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveSignalBatch {
    pub(crate) active_root_window_id: Uuid,
    pub(crate) signals: Vec<ActiveProtectedSignal>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveSignalBatchAck {
    Applied {
        signal_count: u64,
        signal_size_bytes: u64,
    },
    AlreadyApplied {
        signal_count: u64,
        signal_size_bytes: u64,
    },
    WindowClosed,
    CapacityExceeded,
    AuthorityChanged,
    TransactionNotStarted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveDispatchTerminalState {
    Completed,
    ProviderError,
    CancelledBeforeHandoff,
    CancelledAfterHandoff,
    PanickedAfterHandoff,
    AbortedBeforeHandoff,
    UnknownAfterCrash,
}

impl ActiveDispatchTerminalState {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::ProviderError => "provider_error",
            Self::CancelledBeforeHandoff => "cancelled_before_handoff",
            Self::CancelledAfterHandoff => "cancelled_after_handoff",
            Self::PanickedAfterHandoff => "panicked_after_handoff",
            Self::AbortedBeforeHandoff => "aborted_before_handoff",
            Self::UnknownAfterCrash => "unknown_after_crash",
        }
    }

    const fn representative_status(self) -> Option<&'static str> {
        match self {
            Self::Completed => Some("completed"),
            Self::ProviderError => Some("error"),
            _ => None,
        }
    }

    const fn requires_handoff(self) -> bool {
        matches!(
            self,
            Self::Completed
                | Self::ProviderError
                | Self::CancelledAfterHandoff
                | Self::PanickedAfterHandoff
        )
    }

    const fn forbids_handoff(self) -> bool {
        matches!(
            self,
            Self::CancelledBeforeHandoff | Self::AbortedBeforeHandoff
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveDispatchTerminal {
    pub(crate) active_dispatch_terminal_event_id: Uuid,
    pub(crate) active_dispatch_id: Uuid,
    pub(crate) active_assignment_id: Uuid,
    pub(crate) terminal_state: ActiveDispatchTerminalState,
    pub(crate) stable_error_class: Option<String>,
    pub(crate) provider_receipt_hash: Option<String>,
    pub(crate) handed_off_at_unix_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveDispatchTerminalAck {
    Applied,
    AlreadyApplied,
    WindowClosed,
    AuthorityChanged,
    TransactionNotStarted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveRepresentativeStatus {
    Completed,
    Error,
}

impl ActiveRepresentativeStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Completed => "completed",
            Self::Error => "error",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveRootClosure {
    OwnerEnd,
    Deadline,
    ProjectionFault,
    Orphaned,
    ShutdownOrphaned,
    AmbiguousExposure,
}

impl ActiveRootClosure {
    const fn as_str(self) -> &'static str {
        match self {
            Self::OwnerEnd => "owner_end",
            Self::Deadline => "deadline",
            Self::ProjectionFault => "projection_fault",
            Self::Orphaned => "orphaned",
            Self::ShutdownOrphaned => "shutdown_orphaned",
            Self::AmbiguousExposure => "ambiguous_exposure",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveRootTerminal {
    pub(crate) active_root_window_state_event_id: Uuid,
    pub(crate) outcome_id: Uuid,
    pub(crate) active_root_window_id: Uuid,
    pub(crate) closure: ActiveRootClosure,
    pub(crate) label_complete: bool,
    pub(crate) representative_status: Option<ActiveRepresentativeStatus>,
    pub(crate) stable_terminal_error_class: Option<String>,
    pub(crate) interval_end_unix_ms: u64,
    pub(crate) neighborhood_invalidation: Option<ActiveNeighborhoodInvalidation>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveRootTerminalAck {
    Applied,
    AlreadyApplied,
    AuthorityChanged,
    TransactionNotStarted,
}

impl super::LedgerRepository {
    pub(crate) fn admit_active_root(
        &mut self,
        admission: &ActiveRootAdmission,
    ) -> Result<ActiveAdmissionAck, LedgerError> {
        self.admit_active_root_with_start_check(admission, || Some(()))
    }

    pub(crate) fn admit_active_root_with_start_check<G: TransactionStartGuard>(
        &mut self,
        admission: &ActiveRootAdmission,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveAdmissionAck, LedgerError> {
        validate_active_admission(admission)?;
        let database_path = self.database_path.clone();
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveAdmissionAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(ActiveAdmissionAck::TransactionNotStarted);
            }
            Err(error) => {
                return Err(super::map_sqlite_error(
                    &error,
                    LedgerErrorClass::DatabaseOperationFailed,
                ));
            }
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ActiveAdmissionAck::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = admit_active_root_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            admission,
            now_unix_ms()?,
        )?;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }

    pub(crate) fn append_active_signals(
        &mut self,
        batch: &ActiveSignalBatch,
    ) -> Result<ActiveSignalBatchAck, LedgerError> {
        self.append_active_signals_with_start_check(batch, || Some(()))
    }

    pub(crate) fn append_active_signals_with_start_check<G: TransactionStartGuard>(
        &mut self,
        batch: &ActiveSignalBatch,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveSignalBatchAck, LedgerError> {
        let prepared = prepare_signal_batch(batch)?;
        let database_path = self.database_path.clone();
        let project_uuid = self.project_uuid;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveSignalBatchAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(ActiveSignalBatchAck::TransactionNotStarted);
            }
            Err(error) => {
                return Err(super::map_sqlite_error(
                    &error,
                    LedgerErrorClass::DatabaseOperationFailed,
                ));
            }
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ActiveSignalBatchAck::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = append_active_signals_in_transaction(
            &transaction,
            project_uuid,
            batch.active_root_window_id,
            &prepared,
        )?;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }

    pub(crate) fn record_active_dispatch_terminal(
        &mut self,
        terminal: &ActiveDispatchTerminal,
    ) -> Result<ActiveDispatchTerminalAck, LedgerError> {
        self.record_active_dispatch_terminal_with_start_check(terminal, || Some(()))
    }

    pub(crate) fn record_active_dispatch_terminal_with_start_check<G: TransactionStartGuard>(
        &mut self,
        terminal: &ActiveDispatchTerminal,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveDispatchTerminalAck, LedgerError> {
        validate_dispatch_terminal(terminal)?;
        let database_path = self.database_path.clone();
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveDispatchTerminalAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(ActiveDispatchTerminalAck::TransactionNotStarted);
            }
            Err(error) => {
                return Err(super::map_sqlite_error(
                    &error,
                    LedgerErrorClass::DatabaseOperationFailed,
                ));
            }
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ActiveDispatchTerminalAck::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = record_dispatch_terminal_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            terminal,
            now_unix_ms()?,
        )?;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }

    pub(crate) fn terminalize_active_root(
        &mut self,
        terminal: &ActiveRootTerminal,
    ) -> Result<ActiveRootTerminalAck, LedgerError> {
        self.terminalize_active_root_with_start_check(terminal, || Some(()))
    }

    pub(crate) fn terminalize_active_root_with_start_check<G: TransactionStartGuard>(
        &mut self,
        terminal: &ActiveRootTerminal,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveRootTerminalAck, LedgerError> {
        validate_root_terminal(terminal)?;
        let database_path = self.database_path.clone();
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveRootTerminalAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(ActiveRootTerminalAck::TransactionNotStarted);
            }
            Err(error) => {
                return Err(super::map_sqlite_error(
                    &error,
                    LedgerErrorClass::DatabaseOperationFailed,
                ));
            }
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ActiveRootTerminalAck::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = terminalize_root_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            terminal,
            now_unix_ms()?,
        )?;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }
}

pub(crate) fn admit_active_root_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    admission: &ActiveRootAdmission,
    admitted_at_unix_ms: i64,
) -> Result<ActiveAdmissionAck, LedgerError> {
    validate_active_admission(admission)?;
    if let Some(existing) = load_existing_admission(transaction, project_uuid, admission)? {
        return Ok(existing);
    }
    if !active_admission_authority_matches(
        transaction,
        project_uuid,
        process_instance_id,
        admission,
    )? {
        return Ok(ActiveAdmissionAck::AuthorityChanged);
    }
    if let Some(acknowledgement) = active_admission_lifecycle_ack(transaction, admission)? {
        return Ok(acknowledgement);
    }
    let open_windows = transaction
        .query_row(
            "SELECT count(*)
             FROM active_root_windows AS window
             WHERE window.process_instance_id = ?1
               AND NOT EXISTS (
                    SELECT 1 FROM active_root_window_state_events AS state
                    WHERE state.active_root_window_id = window.active_root_window_id
                      AND state.state <> 'open'
               )",
            [process_instance_id.to_string()],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    if !(0..=ACTIVE_OPEN_ROOT_WINDOWS_MAX).contains(&open_windows) {
        return Err(corrupt());
    }
    if open_windows == ACTIVE_OPEN_ROOT_WINDOWS_MAX {
        return Ok(ActiveAdmissionAck::OpenWindowCapacity);
    }

    let limits = load_admission_limits(transaction, admission)?;
    let ordinals = allocate_ordinals(transaction, admission, limits)?;
    let Some(ordinals) = ordinals else {
        return Ok(if limits.cap_full {
            ActiveAdmissionAck::ExperimentCapReached
        } else {
            ActiveAdmissionAck::TrancheFull
        });
    };
    insert_admission_graph(
        transaction,
        project_uuid,
        process_instance_id,
        admission,
        ordinals,
        admitted_at_unix_ms,
    )?;
    super::active_learning::advance_after_admission(
        transaction,
        process_instance_id,
        admission.active_experiment_id,
        admission.tranche_ordinal,
        admitted_at_unix_ms,
    )?;
    Ok(ActiveAdmissionAck::Applied(ActiveAdmissionReceipt {
        total_ordinal: ordinals.total,
        cap_ordinal: ordinals.cap,
        tranche_total_ordinal: ordinals.tranche_total,
        tranche_nonholdout_ordinal: ordinals.tranche_nonholdout,
        admitted_at_unix_ms: u64::try_from(admitted_at_unix_ms).map_err(|_| corrupt())?,
    }))
}

pub(super) fn reconcile_active_startup(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    observed_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let dispatches = transaction
        .prepare(
            "SELECT dispatch.active_dispatch_id, dispatch.active_assignment_id,
                    dispatch.process_instance_id
             FROM active_dispatches AS dispatch
             LEFT JOIN active_dispatch_terminal_events AS terminal
               ON terminal.active_dispatch_id = dispatch.active_dispatch_id
             WHERE terminal.active_dispatch_id IS NULL
             ORDER BY dispatch.admitted_at_unix_ms, dispatch.active_dispatch_id",
        )
        .map_err(database_error)?
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    for (dispatch_id, assignment_id, owner_process_id) in dispatches {
        let owner_process_id = parse_uuid_v7(&owner_process_id)?;
        if super::process::originating_process_is_live(transaction, project_uuid, owner_process_id)?
        {
            continue;
        }
        let acknowledgement = record_dispatch_terminal_in_transaction(
            transaction,
            project_uuid,
            process_instance_id,
            &ActiveDispatchTerminal {
                active_dispatch_terminal_event_id: Uuid::now_v7(),
                active_dispatch_id: parse_uuid_v7(&dispatch_id)?,
                active_assignment_id: parse_uuid_v7(&assignment_id)?,
                terminal_state: ActiveDispatchTerminalState::UnknownAfterCrash,
                stable_error_class: None,
                provider_receipt_hash: None,
                handed_off_at_unix_ms: None,
            },
            observed_at_unix_ms,
        )?;
        if acknowledgement != ActiveDispatchTerminalAck::Applied {
            return Err(corrupt());
        }
    }

    let windows = transaction
        .prepare(
            "SELECT window.active_root_window_id, window.process_instance_id,
                    window.attribution_deadline_unix_ms
             FROM active_root_windows AS window
             WHERE NOT EXISTS (
                SELECT 1 FROM active_root_window_state_events AS state
                WHERE state.active_root_window_id = window.active_root_window_id
                  AND state.state <> 'open'
             )
             ORDER BY window.created_at_unix_ms, window.active_root_window_id",
        )
        .map_err(database_error)?
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    for (root_window_id, owner_process_id, deadline) in windows {
        let owner_process_id = parse_uuid_v7(&owner_process_id)?;
        if super::process::originating_process_is_live(transaction, project_uuid, owner_process_id)?
        {
            continue;
        }
        let deadline = u64::try_from(deadline).map_err(|_| corrupt())?;
        let observed = u64::try_from(observed_at_unix_ms).map_err(|_| corrupt())?;
        let acknowledgement = terminalize_root_in_transaction(
            transaction,
            project_uuid,
            process_instance_id,
            &ActiveRootTerminal {
                active_root_window_state_event_id: Uuid::now_v7(),
                outcome_id: Uuid::now_v7(),
                active_root_window_id: parse_uuid_v7(&root_window_id)?,
                closure: ActiveRootClosure::Orphaned,
                label_complete: false,
                representative_status: None,
                stable_terminal_error_class: None,
                interval_end_unix_ms: observed.min(deadline),
                neighborhood_invalidation: None,
            },
            observed_at_unix_ms,
        )?;
        if acknowledgement != ActiveRootTerminalAck::Applied {
            return Err(corrupt());
        }
    }
    super::active_learning::reconcile_stale_active_look_claims(
        transaction,
        project_uuid,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    Ok(())
}

fn validate_dispatch_terminal(terminal: &ActiveDispatchTerminal) -> Result<(), LedgerError> {
    if !is_uuid_v7(terminal.active_dispatch_terminal_event_id)
        || !is_uuid_v7(terminal.active_dispatch_id)
        || !is_uuid_v7(terminal.active_assignment_id)
        || terminal
            .stable_error_class
            .as_deref()
            .is_some_and(|value| !valid_stable_class(value))
        || terminal
            .provider_receipt_hash
            .as_deref()
            .is_some_and(|value| !is_hash(value))
        || terminal
            .handed_off_at_unix_ms
            .is_some_and(|value| value > i64::MAX as u64)
        || (terminal.terminal_state == ActiveDispatchTerminalState::ProviderError)
            != terminal.stable_error_class.is_some()
        || (terminal.terminal_state.requires_handoff() && terminal.handed_off_at_unix_ms.is_none())
        || (terminal.terminal_state.forbids_handoff() && terminal.handed_off_at_unix_ms.is_some())
    {
        return Err(invariant());
    }
    Ok(())
}

fn record_dispatch_terminal_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    terminal: &ActiveDispatchTerminal,
    created_at_unix_ms: i64,
) -> Result<ActiveDispatchTerminalAck, LedgerError> {
    validate_dispatch_terminal(terminal)?;
    if let Some(existing) = load_existing_dispatch_terminal(transaction, terminal)? {
        return Ok(existing);
    }
    if !super::process::originating_process_is_live(transaction, project_uuid, process_instance_id)?
        || !all_live_processes_support_active_v6(transaction, project_uuid)?
    {
        return Ok(ActiveDispatchTerminalAck::AuthorityChanged);
    }
    let dispatch = transaction
        .query_row(
            "SELECT dispatch.active_assignment_id, assignment.active_root_window_id,
                    root.project_uuid, dispatch.admitted_at_unix_ms
             FROM active_dispatches AS dispatch
             JOIN active_assignments AS assignment
               ON assignment.active_assignment_id = dispatch.active_assignment_id
             JOIN active_root_windows AS root
               ON root.active_root_window_id = assignment.active_root_window_id
             WHERE dispatch.active_dispatch_id = ?1",
            [terminal.active_dispatch_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(dispatch) = dispatch else {
        return Ok(ActiveDispatchTerminalAck::AuthorityChanged);
    };
    if dispatch.0 != terminal.active_assignment_id.to_string()
        || dispatch.2 != project_uuid.to_string()
    {
        return Ok(ActiveDispatchTerminalAck::AuthorityChanged);
    }
    let root_window_id = parse_uuid_v7(&dispatch.1)?;
    if !root_window_is_open(transaction, root_window_id)? {
        return Ok(ActiveDispatchTerminalAck::WindowClosed);
    }
    if terminal.handed_off_at_unix_ms.is_some_and(|handed_off| {
        let created = u64::try_from(created_at_unix_ms).ok();
        u64::try_from(dispatch.3).map_or(true, |admitted| {
            handed_off < admitted || created.is_none_or(|created| handed_off > created)
        })
    }) {
        return Err(invariant());
    }
    let payload_hash =
        dispatch_terminal_payload_hash(process_instance_id, terminal, created_at_unix_ms)?;
    execute_one(
        transaction,
        "INSERT INTO active_dispatch_terminal_events (
            active_dispatch_terminal_event_id, active_dispatch_id,
            active_assignment_id, terminal_state, representative_status,
            stable_error_class, provider_receipt_hash, handed_off_at_unix_ms,
            process_instance_id, created_at_unix_ms, canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            terminal.active_dispatch_terminal_event_id.to_string(),
            terminal.active_dispatch_id.to_string(),
            terminal.active_assignment_id.to_string(),
            terminal.terminal_state.as_str(),
            terminal.terminal_state.representative_status(),
            terminal.stable_error_class,
            terminal.provider_receipt_hash,
            terminal
                .handed_off_at_unix_ms
                .map(i64::try_from)
                .transpose()
                .map_err(|_| invariant())?,
            process_instance_id.to_string(),
            created_at_unix_ms,
            payload_hash,
        ],
    )?;
    Ok(ActiveDispatchTerminalAck::Applied)
}

fn load_existing_dispatch_terminal(
    connection: &Connection,
    terminal: &ActiveDispatchTerminal,
) -> Result<Option<ActiveDispatchTerminalAck>, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT active_dispatch_terminal_event_id, active_assignment_id,
                    terminal_state, representative_status, stable_error_class,
                    provider_receipt_hash, handed_off_at_unix_ms,
                    process_instance_id, created_at_unix_ms, canonical_payload_hash
             FROM active_dispatch_terminal_events WHERE active_dispatch_id = ?1",
            [terminal.active_dispatch_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, Option<i64>>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, String>(9)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(stored) = stored else {
        return Ok(None);
    };
    let stored_process = parse_uuid_v7(&stored.7)?;
    let expected_hash = dispatch_terminal_payload_hash(stored_process, terminal, stored.8)?;
    if stored.0 != terminal.active_dispatch_terminal_event_id.to_string()
        || stored.1 != terminal.active_assignment_id.to_string()
        || stored.2 != terminal.terminal_state.as_str()
        || stored.3.as_deref() != terminal.terminal_state.representative_status()
        || stored.4 != terminal.stable_error_class
        || stored.5 != terminal.provider_receipt_hash
        || stored.6.and_then(|value| u64::try_from(value).ok()) != terminal.handed_off_at_unix_ms
        || stored.9 != expected_hash
    {
        return Err(corrupt());
    }
    Ok(Some(ActiveDispatchTerminalAck::AlreadyApplied))
}

fn dispatch_terminal_payload_hash(
    process_instance_id: Uuid,
    terminal: &ActiveDispatchTerminal,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(json!({
        "shape": "active_dispatch_terminal_v1",
        "active_dispatch_terminal_event_id": terminal.active_dispatch_terminal_event_id,
        "active_dispatch_id": terminal.active_dispatch_id,
        "active_assignment_id": terminal.active_assignment_id,
        "terminal_state": terminal.terminal_state.as_str(),
        "representative_status": terminal.terminal_state.representative_status(),
        "stable_error_class": terminal.stable_error_class,
        "provider_receipt_hash": terminal.provider_receipt_hash,
        "handed_off_at_unix_ms": terminal.handed_off_at_unix_ms.map(|value| value.to_string()),
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": created_at_unix_ms.to_string(),
    }))
}

fn validate_root_terminal(terminal: &ActiveRootTerminal) -> Result<(), LedgerError> {
    if !is_uuid_v7(terminal.active_root_window_state_event_id)
        || !is_uuid_v7(terminal.outcome_id)
        || !is_uuid_v7(terminal.active_root_window_id)
        || terminal.interval_end_unix_ms > i64::MAX as u64
        || terminal
            .stable_terminal_error_class
            .as_deref()
            .is_some_and(|value| !valid_stable_class(value))
        || (terminal.representative_status == Some(ActiveRepresentativeStatus::Error))
            != terminal.stable_terminal_error_class.is_some()
        || (terminal.closure == ActiveRootClosure::ProjectionFault && terminal.label_complete)
        || terminal
            .neighborhood_invalidation
            .as_ref()
            .is_some_and(|invalidation| {
                invalidation.cause_active_dispatch_id.is_none()
                    || invalidation.cause_outcome_id != Some(terminal.outcome_id)
            })
    {
        return Err(invariant());
    }
    Ok(())
}

#[derive(Debug)]
struct RootOutcomeFacts {
    active_experiment_id: Uuid,
    active_assignment_id: Uuid,
    decision_id: Uuid,
    root_key: String,
    outcome_policy_hash: String,
    arm: ActiveAssignmentArm,
    tranche_ordinal: u64,
    total_ordinal: u64,
    cap_ordinal: Option<u64>,
    admission_unix_ms: u64,
    attribution_deadline_unix_ms: u64,
    signal_count: u64,
    signal_size_bytes: u64,
    signal_aggregate_hash: String,
    reduced_label: Option<&'static str>,
    exposure: ExposureFacts,
}

#[derive(Debug)]
struct RootOutcomeInputs {
    admission: ActiveRootAdmission,
    receipt: ActiveAdmissionReceipt,
    structural_ambiguity: bool,
    signals: SignalReduction,
    dispatch: Option<ExposureDispatch>,
}

/// One stored Active outcome emitted only after its complete authority verifies.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VerifiedStoredOutcome {
    pub(crate) outcome_id: Uuid,
    pub(crate) project_uuid: Uuid,
    pub(crate) pool_id: String,
    pub(crate) active_experiment_id: Uuid,
    pub(crate) decision_id: Option<Uuid>,
    pub(crate) arm: String,
    pub(crate) label: Option<String>,
    pub(crate) attribution_status: String,
    pub(crate) latency_ms: f64,
    pub(crate) created_at_unix_ms: i64,
    pub(crate) canonical_payload_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ExposureFacts {
    attribution_status: &'static str,
    root_state: &'static str,
    label: Option<&'static str>,
    stable_terminal_error_class: Option<String>,
}

fn terminalize_root_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    terminal: &ActiveRootTerminal,
    created_at_unix_ms: i64,
) -> Result<ActiveRootTerminalAck, LedgerError> {
    validate_root_terminal(terminal)?;
    if let Some(existing) =
        load_existing_root_terminal(transaction, project_uuid, process_instance_id, terminal)?
    {
        return Ok(existing);
    }
    if !super::process::originating_process_is_live(transaction, project_uuid, process_instance_id)?
        || !all_live_processes_support_active_v6(transaction, project_uuid)?
    {
        return Ok(ActiveRootTerminalAck::AuthorityChanged);
    }
    if !root_window_is_open(transaction, terminal.active_root_window_id)? {
        return Err(corrupt());
    }
    let facts = derive_root_outcome_facts(transaction, project_uuid, terminal)?;
    let retiring: bool = transaction
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM active_retirement_markers WHERE active_experiment_id = ?1
             )",
            [facts.active_experiment_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    if retiring {
        return Ok(ActiveRootTerminalAck::AuthorityChanged);
    }
    let stored_label_complete = terminal.label_complete && facts.exposure.root_state == "completed";
    let root_state_hash = root_state_payload_hash(
        terminal.active_root_window_state_event_id,
        terminal.active_root_window_id,
        facts.exposure.root_state,
        stored_label_complete,
        facts.signal_count,
        facts.signal_size_bytes,
        process_instance_id,
        created_at_unix_ms,
    )?;
    execute_one(
        transaction,
        "INSERT INTO active_root_window_state_events (
            active_root_window_state_event_id, active_root_window_id, state,
            label_complete, signal_count, signal_size_bytes,
            process_instance_id, created_at_unix_ms, canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            terminal.active_root_window_state_event_id.to_string(),
            terminal.active_root_window_id.to_string(),
            facts.exposure.root_state,
            i64::from(stored_label_complete),
            i64::try_from(facts.signal_count).map_err(|_| corrupt())?,
            i64::try_from(facts.signal_size_bytes).map_err(|_| corrupt())?,
            process_instance_id.to_string(),
            created_at_unix_ms,
            root_state_hash,
        ],
    )?;
    let interval_start = facts.admission_unix_ms;
    let interval_end = terminal.interval_end_unix_ms;
    let latency_ms = (interval_end - interval_start) as f64;
    let outcome_hash = outcome_payload_hash(
        process_instance_id,
        terminal,
        &facts,
        latency_ms.to_bits(),
        created_at_unix_ms,
    )?;
    execute_one(
        transaction,
        "INSERT INTO outcomes (
            outcome_id, active_experiment_id, active_root_window_id,
            active_assignment_id, representative_decision_id, root_key,
            outcome_policy_hash, source_signal_aggregate_hash, arm,
            total_ordinal, cap_ordinal, label, attribution_status,
            interval_start_unix_ms, interval_end_unix_ms,
            latency_ms, latency_ms_bits, stable_terminal_error_class,
            process_instance_id, created_at_unix_ms, canonical_payload_hash
         ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
            ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21
         )",
        params![
            terminal.outcome_id.to_string(),
            facts.active_experiment_id.to_string(),
            terminal.active_root_window_id.to_string(),
            facts.active_assignment_id.to_string(),
            facts.decision_id.to_string(),
            facts.root_key,
            facts.outcome_policy_hash,
            facts.signal_aggregate_hash,
            facts.arm.as_str(),
            i64::try_from(facts.total_ordinal).map_err(|_| corrupt())?,
            facts
                .cap_ordinal
                .map(i64::try_from)
                .transpose()
                .map_err(|_| corrupt())?,
            facts.exposure.label,
            facts.exposure.attribution_status,
            i64::try_from(interval_start).map_err(|_| corrupt())?,
            i64::try_from(interval_end).map_err(|_| invariant())?,
            latency_ms,
            latency_ms.to_bits() as i64,
            facts.exposure.stable_terminal_error_class,
            process_instance_id.to_string(),
            created_at_unix_ms,
            outcome_hash,
        ],
    )?;
    if facts.exposure.attribution_status == "eligible_treatment"
        && facts.exposure.label == Some("failure")
    {
        let invalidation = terminal
            .neighborhood_invalidation
            .as_ref()
            .ok_or_else(corrupt)?;
        let acknowledgement =
            super::active_learning::invalidate_active_neighborhood_in_transaction(
                transaction,
                project_uuid,
                process_instance_id,
                invalidation,
                created_at_unix_ms,
            )?;
        if !matches!(
            acknowledgement,
            ActiveNeighborhoodMutationAck::Applied(_)
                | ActiveNeighborhoodMutationAck::AlreadyApplied(_)
                | ActiveNeighborhoodMutationAck::AlreadyCurrent(_)
                | ActiveNeighborhoodMutationAck::CoolingOff(_)
        ) {
            return Err(corrupt());
        }
    }
    super::active_learning::advance_after_terminalization(
        transaction,
        process_instance_id,
        facts.active_experiment_id,
        facts.tranche_ordinal,
        created_at_unix_ms,
    )?;
    Ok(ActiveRootTerminalAck::Applied)
}

fn derive_root_outcome_facts(
    connection: &Connection,
    project_uuid: Uuid,
    terminal: &ActiveRootTerminal,
) -> Result<RootOutcomeFacts, LedgerError> {
    let inputs =
        load_root_outcome_inputs(connection, project_uuid, terminal.active_root_window_id)?;
    root_outcome_facts(&inputs, terminal)
}

fn load_root_outcome_inputs(
    connection: &Connection,
    project_uuid: Uuid,
    active_root_window_id: Uuid,
) -> Result<RootOutcomeInputs, LedgerError> {
    let (admission, receipt) =
        load_verified_admission_for_outcome(connection, project_uuid, active_root_window_id)?;
    let links = connection
        .query_row(
            "SELECT count(*),
                    sum(CASE WHEN decision_id = ?2 THEN 1 ELSE 0 END)
             FROM active_root_decision_links WHERE active_root_window_id = ?1",
            params![
                active_root_window_id.to_string(),
                admission.decision_id.to_string()
            ],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        )
        .map_err(database_error)?;
    let structural_ambiguity = links != (1, 1);
    let signals = load_signal_reduction(connection, active_root_window_id)?;
    let dispatch = load_exposure_dispatch(connection, admission.active_assignment_id)?;
    Ok(RootOutcomeInputs {
        admission,
        receipt,
        structural_ambiguity,
        signals,
        dispatch,
    })
}

fn root_outcome_facts(
    inputs: &RootOutcomeInputs,
    terminal: &ActiveRootTerminal,
) -> Result<RootOutcomeFacts, LedgerError> {
    let admission = &inputs.admission;
    let receipt = inputs.receipt;
    if terminal.interval_end_unix_ms < receipt.admitted_at_unix_ms
        || terminal.interval_end_unix_ms > admission.attribution_deadline_unix_ms
    {
        return Err(invariant());
    }
    let exposure = classify_exposure(
        terminal,
        admission.arm,
        inputs.signals.reduced_label,
        inputs.structural_ambiguity,
        inputs.dispatch.clone(),
    );
    Ok(RootOutcomeFacts {
        active_experiment_id: admission.active_experiment_id,
        active_assignment_id: admission.active_assignment_id,
        decision_id: admission.decision_id,
        root_key: admission.root_key.clone(),
        outcome_policy_hash: admission.outcome_policy_hash.clone(),
        arm: admission.arm,
        tranche_ordinal: admission.tranche_ordinal,
        total_ordinal: receipt.total_ordinal,
        cap_ordinal: receipt.cap_ordinal,
        admission_unix_ms: receipt.admitted_at_unix_ms,
        attribution_deadline_unix_ms: admission.attribution_deadline_unix_ms,
        signal_count: inputs.signals.count,
        signal_size_bytes: inputs.signals.size_bytes,
        signal_aggregate_hash: inputs.signals.aggregate_hash.clone(),
        reduced_label: inputs.signals.reduced_label,
        exposure,
    })
}

fn load_verified_admission_for_outcome(
    connection: &Connection,
    project_uuid: Uuid,
    active_root_window_id: Uuid,
) -> Result<(ActiveRootAdmission, ActiveAdmissionReceipt), LedgerError> {
    let root = connection
        .query_row(
            "SELECT active_experiment_id, project_uuid, pool_id, candidate_id,
                    root_key, owner_relation_hash, config_generation_id,
                    learning_generation_id, cohort_generation_id, outcome_policy_hash,
                    opened_after_ingest_seq, attribution_deadline_unix_ms
             FROM active_root_windows WHERE active_root_window_id = ?1",
            [active_root_window_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, String>(8)?,
                    row.get::<_, String>(9)?,
                    row.get::<_, i64>(10)?,
                    row.get::<_, i64>(11)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    if root.1 != project_uuid.to_string() {
        return Err(corrupt());
    }
    let assignment = connection
        .query_row(
            "SELECT active_assignment_id, decision_id, tranche_ordinal, arm,
                    cohort_threshold_numerator, selection_threshold_numerator,
                    configured_holdout_probability_bits,
                    configured_canary_probability_bits,
                    effective_arm_probability_bits,
                    conditional_selection_probability_bits, propensity_bits,
                    control_generation
             FROM active_assignments WHERE active_root_window_id = ?1",
            [active_root_window_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Vec<u8>>(4)?,
                    row.get::<_, Vec<u8>>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, i64>(9)?,
                    row.get::<_, i64>(10)?,
                    row.get::<_, i64>(11)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    let active_assignment_id = parse_uuid_v7(&assignment.0)?;
    let dispatch = connection
        .query_row(
            "SELECT active_dispatch_id, request_identity_hash
             FROM active_dispatches WHERE active_assignment_id = ?1",
            [active_assignment_id.to_string()],
            |row| {
                Ok(ActiveDispatchAdmission {
                    active_dispatch_id: parse_uuid_v7(&row.get::<_, String>(0)?)
                        .map_err(|_| rusqlite::Error::InvalidQuery)?,
                    request_identity_hash: row.get(1)?,
                })
            },
        )
        .optional()
        .map_err(database_error)?;
    let admission = ActiveRootAdmission {
        active_root_window_id,
        active_experiment_id: parse_uuid_v7(&root.0)?,
        active_assignment_id,
        decision_id: parse_uuid_v7(&assignment.1)?,
        dispatch,
        pool_id: root.2,
        candidate_id: root.3,
        root_key: root.4,
        owner_relation_hash: root.5,
        config_generation_id: root.6,
        learning_generation_id: parse_uuid_v7(&root.7)?,
        cohort_generation_id: parse_uuid_v7(&root.8)?,
        outcome_policy_hash: root.9,
        opened_after_ingest_seq: u64::try_from(root.10).map_err(|_| corrupt())?,
        attribution_deadline_unix_ms: u64::try_from(root.11).map_err(|_| corrupt())?,
        tranche_ordinal: u64::try_from(assignment.2).map_err(|_| corrupt())?,
        arm: parse_arm(&assignment.3)?,
        cohort_threshold_numerator: assignment.4.try_into().map_err(|_| corrupt())?,
        selection_threshold_numerator: assignment.5.try_into().map_err(|_| corrupt())?,
        configured_holdout_probability_bits: assignment.6 as u64,
        configured_canary_probability_bits: assignment.7 as u64,
        effective_arm_probability_bits: assignment.8 as u64,
        conditional_selection_probability_bits: assignment.9 as u64,
        propensity_bits: assignment.10 as u64,
        control_generation: u64::try_from(assignment.11).map_err(|_| corrupt())?,
    };
    let receipt = match load_existing_admission(connection, project_uuid, &admission)? {
        Some(ActiveAdmissionAck::AlreadyApplied(receipt)) => receipt,
        _ => return Err(corrupt()),
    };
    Ok((admission, receipt))
}

#[derive(Debug)]
struct SignalReduction {
    count: u64,
    size_bytes: u64,
    aggregate_hash: String,
    reduced_label: Option<&'static str>,
}

#[derive(Debug)]
struct StoredSignal {
    active_root_signal_id: String,
    signal_ordinal: i64,
    signal_identity_hash: String,
    event_kind: String,
    scope_phase: String,
    category: String,
    name: String,
    disposition: String,
    observed_at_unix_ms: i64,
    canonical_signal_json: String,
    canonical_payload_hash: String,
}

fn load_signal_reduction(
    connection: &Connection,
    root_window_id: Uuid,
) -> Result<SignalReduction, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT active_root_signal_id, signal_ordinal, signal_identity_hash,
                    event_kind, scope_phase, category, name, disposition,
                    observed_at_unix_ms, canonical_signal_json, canonical_payload_hash
             FROM active_root_signals
             WHERE active_root_window_id = ?1 ORDER BY signal_ordinal",
        )
        .map_err(database_error)?;
    let rows = statement
        .query_map([root_window_id.to_string()], |row| {
            Ok(StoredSignal {
                active_root_signal_id: row.get(0)?,
                signal_ordinal: row.get(1)?,
                signal_identity_hash: row.get(2)?,
                event_kind: row.get(3)?,
                scope_phase: row.get(4)?,
                category: row.get(5)?,
                name: row.get(6)?,
                disposition: row.get(7)?,
                observed_at_unix_ms: row.get(8)?,
                canonical_signal_json: row.get(9)?,
                canonical_payload_hash: row.get(10)?,
            })
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    if rows.len() > ACTIVE_ROOT_SIGNALS_MAX {
        return Err(corrupt());
    }
    let mut size_bytes = 0_usize;
    let mut has_success = false;
    let mut has_failure = false;
    let mut aggregate = Vec::with_capacity(rows.len());
    for (index, row) in rows.iter().enumerate() {
        let disposition = match row.disposition.as_str() {
            "success" => ActiveSignalDisposition::Success,
            "failure" => ActiveSignalDisposition::Failure,
            "ignored" => ActiveSignalDisposition::Ignored,
            _ => return Err(corrupt()),
        };
        let signal = ActiveProtectedSignal {
            active_root_signal_id: parse_uuid_v7(&row.active_root_signal_id)?,
            signal_identity_hash: row.signal_identity_hash.clone(),
            event_kind: row.event_kind.clone(),
            scope_phase: row.scope_phase.clone(),
            category: row.category.clone(),
            name: row.name.clone(),
            disposition,
            observed_at_unix_ms: u64::try_from(row.observed_at_unix_ms).map_err(|_| corrupt())?,
            canonical_signal_json: row.canonical_signal_json.clone(),
        };
        let prepared = prepare_signal_batch(&ActiveSignalBatch {
            active_root_window_id: root_window_id,
            signals: vec![signal],
        })
        .map_err(|_| corrupt())?;
        if row.signal_ordinal != index as i64
            || prepared[0].canonical_payload_hash != row.canonical_payload_hash
        {
            return Err(corrupt());
        }
        size_bytes = size_bytes
            .checked_add(prepared[0].size_bytes)
            .ok_or_else(corrupt)?;
        match row.disposition.as_str() {
            "success" => has_success = true,
            "failure" => has_failure = true,
            "ignored" => {}
            _ => return Err(corrupt()),
        }
        aggregate.push(json!({
            "ordinal": index.to_string(),
            "signal_identity_hash": row.signal_identity_hash,
            "disposition": row.disposition,
            "canonical_payload_hash": row.canonical_payload_hash,
        }));
    }
    if size_bytes > ACTIVE_ROOT_SIGNAL_BYTES_MAX {
        return Err(corrupt());
    }
    let reduced_label = if has_failure {
        Some("failure")
    } else if has_success {
        Some("success")
    } else {
        None
    };
    Ok(SignalReduction {
        count: rows.len() as u64,
        size_bytes: size_bytes as u64,
        aggregate_hash: hash_json(json!({
            "shape": "active_signal_aggregate_v1",
            "signals": aggregate,
        }))?,
        reduced_label,
    })
}

#[derive(Debug, Clone)]
struct ExposureDispatch {
    terminal_state: Option<String>,
    representative_status: Option<String>,
    stable_error_class: Option<String>,
}

#[derive(Debug)]
struct StoredExposureDispatch {
    active_dispatch_id: String,
    active_assignment_id: String,
    terminal_event_id: Option<String>,
    terminal_state: Option<String>,
    representative_status: Option<String>,
    stable_error_class: Option<String>,
    provider_receipt_hash: Option<String>,
    handed_off_at_unix_ms: Option<i64>,
}

fn load_exposure_dispatch(
    connection: &Connection,
    assignment_id: Uuid,
) -> Result<Option<ExposureDispatch>, LedgerError> {
    let rows = connection
        .prepare(
            "SELECT dispatch.active_dispatch_id, dispatch.active_assignment_id,
                    terminal.active_dispatch_terminal_event_id,
                    terminal.terminal_state, terminal.representative_status,
                    terminal.stable_error_class, terminal.provider_receipt_hash,
                    terminal.handed_off_at_unix_ms
             FROM active_dispatches AS dispatch
             LEFT JOIN active_dispatch_terminal_events AS terminal
               ON terminal.active_dispatch_id = dispatch.active_dispatch_id
             WHERE dispatch.active_assignment_id = ?1",
        )
        .map_err(database_error)?
        .query_map([assignment_id.to_string()], |row| {
            Ok(StoredExposureDispatch {
                active_dispatch_id: row.get(0)?,
                active_assignment_id: row.get(1)?,
                terminal_event_id: row.get(2)?,
                terminal_state: row.get(3)?,
                representative_status: row.get(4)?,
                stable_error_class: row.get(5)?,
                provider_receipt_hash: row.get(6)?,
                handed_off_at_unix_ms: row.get(7)?,
            })
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    let Some(stored) = (match rows.len() {
        0 => return Ok(None),
        1 => rows.into_iter().next(),
        _ => return Err(corrupt()),
    }) else {
        return Err(corrupt());
    };
    if stored.active_assignment_id != assignment_id.to_string() {
        return Err(corrupt());
    }
    let Some(terminal_event_id) = stored.terminal_event_id.as_deref() else {
        if stored.terminal_state.is_some()
            || stored.representative_status.is_some()
            || stored.stable_error_class.is_some()
            || stored.provider_receipt_hash.is_some()
            || stored.handed_off_at_unix_ms.is_some()
        {
            return Err(corrupt());
        }
        return Ok(Some(ExposureDispatch {
            terminal_state: None,
            representative_status: None,
            stable_error_class: None,
        }));
    };
    let terminal_state = stored.terminal_state.as_deref().ok_or_else(corrupt)?;
    let terminal = ActiveDispatchTerminal {
        active_dispatch_terminal_event_id: parse_uuid_v7(terminal_event_id)?,
        active_dispatch_id: parse_uuid_v7(&stored.active_dispatch_id)?,
        active_assignment_id: assignment_id,
        terminal_state: parse_dispatch_terminal_state(terminal_state)?,
        stable_error_class: stored.stable_error_class.clone(),
        provider_receipt_hash: stored.provider_receipt_hash,
        handed_off_at_unix_ms: stored
            .handed_off_at_unix_ms
            .map(u64::try_from)
            .transpose()
            .map_err(|_| corrupt())?,
    };
    if load_existing_dispatch_terminal(connection, &terminal)?
        != Some(ActiveDispatchTerminalAck::AlreadyApplied)
        || stored.representative_status.as_deref()
            != terminal.terminal_state.representative_status()
    {
        return Err(corrupt());
    }
    Ok(Some(ExposureDispatch {
        terminal_state: stored.terminal_state,
        representative_status: stored.representative_status,
        stable_error_class: stored.stable_error_class,
    }))
}

fn parse_dispatch_terminal_state(value: &str) -> Result<ActiveDispatchTerminalState, LedgerError> {
    match value {
        "completed" => Ok(ActiveDispatchTerminalState::Completed),
        "provider_error" => Ok(ActiveDispatchTerminalState::ProviderError),
        "cancelled_before_handoff" => Ok(ActiveDispatchTerminalState::CancelledBeforeHandoff),
        "cancelled_after_handoff" => Ok(ActiveDispatchTerminalState::CancelledAfterHandoff),
        "panicked_after_handoff" => Ok(ActiveDispatchTerminalState::PanickedAfterHandoff),
        "aborted_before_handoff" => Ok(ActiveDispatchTerminalState::AbortedBeforeHandoff),
        "unknown_after_crash" => Ok(ActiveDispatchTerminalState::UnknownAfterCrash),
        _ => Err(corrupt()),
    }
}

fn classify_exposure(
    terminal: &ActiveRootTerminal,
    arm: ActiveAssignmentArm,
    reduced_label: Option<&'static str>,
    structural_ambiguity: bool,
    dispatch: Option<ExposureDispatch>,
) -> ExposureFacts {
    if terminal.closure == ActiveRootClosure::Orphaned {
        return nonlearning_exposure("orphaned", "orphaned");
    }
    if terminal.closure == ActiveRootClosure::ShutdownOrphaned {
        return nonlearning_exposure("orphaned", "shutdown_orphaned");
    }
    if terminal.closure == ActiveRootClosure::ProjectionFault || !terminal.label_complete {
        return nonlearning_exposure("unattributed", "unattributed");
    }
    if terminal.closure == ActiveRootClosure::AmbiguousExposure || structural_ambiguity {
        return nonlearning_exposure("ambiguous_exposure", "ambiguous_exposure");
    }
    match arm {
        ActiveAssignmentArm::CandidateTreatment => {
            let Some(dispatch) = dispatch else {
                return nonlearning_exposure("unattributed", "unattributed");
            };
            match dispatch.terminal_state.as_deref() {
                Some("unknown_after_crash") => nonlearning_exposure("orphaned", "orphaned"),
                Some("completed" | "provider_error") => {
                    let expected = terminal.representative_status.map(|status| status.as_str());
                    if expected != dispatch.representative_status.as_deref()
                        || terminal.stable_terminal_error_class != dispatch.stable_error_class
                    {
                        nonlearning_exposure("ambiguous_exposure", "ambiguous_exposure")
                    } else if let Some(label) = reduced_label {
                        ExposureFacts {
                            attribution_status: "eligible_treatment",
                            root_state: "completed",
                            label: Some(label),
                            stable_terminal_error_class: dispatch.stable_error_class,
                        }
                    } else {
                        nonlearning_exposure("unattributed", "unattributed")
                    }
                }
                _ => nonlearning_exposure("unattributed", "unattributed"),
            }
        }
        ActiveAssignmentArm::AnchorControl | ActiveAssignmentArm::AnchorHoldout => {
            if dispatch.is_some() {
                return nonlearning_exposure("ambiguous_exposure", "ambiguous_exposure");
            }
            if terminal.representative_status.is_none() {
                return nonlearning_exposure("unattributed", "unattributed");
            }
            let Some(label) = reduced_label else {
                return nonlearning_exposure("unattributed", "unattributed");
            };
            ExposureFacts {
                attribution_status: if arm == ActiveAssignmentArm::AnchorControl {
                    "eligible_control"
                } else {
                    "monitoring_only"
                },
                root_state: "completed",
                label: Some(label),
                stable_terminal_error_class: terminal.stable_terminal_error_class.clone(),
            }
        }
        ActiveAssignmentArm::NonLearning => nonlearning_exposure("unattributed", "unattributed"),
    }
}

fn nonlearning_exposure(
    attribution_status: &'static str,
    root_state: &'static str,
) -> ExposureFacts {
    ExposureFacts {
        attribution_status,
        root_state,
        label: None,
        stable_terminal_error_class: None,
    }
}

fn parse_arm(value: &str) -> Result<ActiveAssignmentArm, LedgerError> {
    match value {
        "candidate_treatment" => Ok(ActiveAssignmentArm::CandidateTreatment),
        "anchor_control" => Ok(ActiveAssignmentArm::AnchorControl),
        "anchor_holdout" => Ok(ActiveAssignmentArm::AnchorHoldout),
        "non_learning" => Ok(ActiveAssignmentArm::NonLearning),
        _ => Err(corrupt()),
    }
}

#[derive(Debug)]
struct StoredRootTerminalState {
    state_event_id: String,
    state: String,
    label_complete: i64,
    signal_count: i64,
    signal_size_bytes: i64,
    process_instance_id: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug)]
struct StoredOutcomeRow {
    outcome_id: String,
    active_experiment_id: String,
    active_root_window_id: String,
    active_assignment_id: String,
    representative_decision_id: Option<String>,
    root_key: String,
    outcome_policy_hash: String,
    source_signal_aggregate_hash: String,
    arm: String,
    total_ordinal: i64,
    cap_ordinal: Option<i64>,
    label: Option<String>,
    attribution_status: String,
    interval_start_unix_ms: i64,
    interval_end_unix_ms: i64,
    latency_ms: f64,
    latency_ms_bits: i64,
    stable_terminal_error_class: Option<String>,
    process_instance_id: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

fn load_existing_root_terminal(
    connection: &Connection,
    project_uuid: Uuid,
    _current_process_instance_id: Uuid,
    terminal: &ActiveRootTerminal,
) -> Result<Option<ActiveRootTerminalAck>, LedgerError> {
    load_verified_root_outcome(
        connection,
        project_uuid,
        terminal.active_root_window_id,
        Some(terminal),
    )
    .map(|outcome| outcome.map(|_| ActiveRootTerminalAck::AlreadyApplied))
}

/// Load one outcome by immutable ID and verify its complete Active authority.
pub(crate) fn load_verified_outcome(
    connection: &Connection,
    project_uuid: Uuid,
    outcome_id: Uuid,
) -> Result<Option<VerifiedStoredOutcome>, LedgerError> {
    if !is_uuid_v7(outcome_id) {
        return Err(invariant());
    }
    let root = connection
        .query_row(
            "SELECT outcome.active_root_window_id, root.project_uuid
             FROM outcomes AS outcome
             JOIN active_root_windows AS root
               ON root.active_root_window_id = outcome.active_root_window_id
             WHERE outcome.outcome_id = ?1",
            [outcome_id.to_string()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    let Some((active_root_window_id, stored_project_uuid)) = root else {
        return Ok(None);
    };
    if parse_uuid_v7(&stored_project_uuid)? != project_uuid {
        return Ok(None);
    }
    let active_root_window_id = parse_uuid_v7(&active_root_window_id)?;
    let outcome =
        load_verified_root_outcome(connection, project_uuid, active_root_window_id, None)?
            .ok_or_else(corrupt)?;
    if outcome.outcome_id != outcome_id {
        return Err(corrupt());
    }
    Ok(Some(outcome))
}

fn load_verified_root_outcome(
    connection: &Connection,
    project_uuid: Uuid,
    active_root_window_id: Uuid,
    expected_terminal: Option<&ActiveRootTerminal>,
) -> Result<Option<VerifiedStoredOutcome>, LedgerError> {
    let state = connection
        .query_row(
            "SELECT active_root_window_state_event_id, state, label_complete,
                    signal_count, signal_size_bytes, process_instance_id,
                    created_at_unix_ms, canonical_payload_hash
             FROM active_root_window_state_events
             WHERE active_root_window_id = ?1 AND state <> 'open'",
            [active_root_window_id.to_string()],
            |row| {
                Ok(StoredRootTerminalState {
                    state_event_id: row.get(0)?,
                    state: row.get(1)?,
                    label_complete: row.get(2)?,
                    signal_count: row.get(3)?,
                    signal_size_bytes: row.get(4)?,
                    process_instance_id: row.get(5)?,
                    created_at_unix_ms: row.get(6)?,
                    canonical_payload_hash: row.get(7)?,
                })
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(state) = state else {
        let outcome_collision: bool = connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM outcomes WHERE outcome_id = ?1)",
                [expected_terminal
                    .map(|terminal| terminal.outcome_id.to_string())
                    .unwrap_or_default()],
                |row| row.get(0),
            )
            .map_err(database_error)?;
        return if outcome_collision {
            Err(corrupt())
        } else {
            Ok(None)
        };
    };
    let outcome = connection
        .query_row(
            "SELECT outcome_id, active_experiment_id, active_root_window_id,
                    active_assignment_id, representative_decision_id, root_key,
                    outcome_policy_hash, source_signal_aggregate_hash, arm,
                    total_ordinal, cap_ordinal, label, attribution_status,
                    interval_start_unix_ms, interval_end_unix_ms, latency_ms,
                    latency_ms_bits, stable_terminal_error_class,
                    process_instance_id, created_at_unix_ms, canonical_payload_hash
             FROM outcomes WHERE active_root_window_id = ?1",
            [active_root_window_id.to_string()],
            |row| {
                Ok(StoredOutcomeRow {
                    outcome_id: row.get(0)?,
                    active_experiment_id: row.get(1)?,
                    active_root_window_id: row.get(2)?,
                    active_assignment_id: row.get(3)?,
                    representative_decision_id: row.get(4)?,
                    root_key: row.get(5)?,
                    outcome_policy_hash: row.get(6)?,
                    source_signal_aggregate_hash: row.get(7)?,
                    arm: row.get(8)?,
                    total_ordinal: row.get(9)?,
                    cap_ordinal: row.get(10)?,
                    label: row.get(11)?,
                    attribution_status: row.get(12)?,
                    interval_start_unix_ms: row.get(13)?,
                    interval_end_unix_ms: row.get(14)?,
                    latency_ms: row.get(15)?,
                    latency_ms_bits: row.get(16)?,
                    stable_terminal_error_class: row.get(17)?,
                    process_instance_id: row.get(18)?,
                    created_at_unix_ms: row.get(19)?,
                    canonical_payload_hash: row.get(20)?,
                })
            },
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    let inputs = load_root_outcome_inputs(connection, project_uuid, active_root_window_id)?;
    let candidates = match expected_terminal {
        Some(terminal) => vec![terminal.clone()],
        None => stored_terminal_candidates(&state, &outcome)?,
    };
    for terminal in candidates {
        if validate_root_terminal(&terminal).is_err() {
            continue;
        }
        let facts = root_outcome_facts(&inputs, &terminal)?;
        if stored_outcome_matches(&state, &outcome, &terminal, &facts)? {
            let latency_bits = outcome.latency_ms_bits as u64;
            if !outcome.latency_ms.is_finite()
                || outcome.latency_ms < 0.0
                || outcome.latency_ms.to_bits() != latency_bits
                || outcome.created_at_unix_ms < 0
            {
                return Err(corrupt());
            }
            return Ok(Some(VerifiedStoredOutcome {
                outcome_id: parse_uuid_v7(&outcome.outcome_id)?,
                project_uuid,
                pool_id: inputs.admission.pool_id.clone(),
                active_experiment_id: parse_uuid_v7(&outcome.active_experiment_id)?,
                decision_id: outcome
                    .representative_decision_id
                    .as_deref()
                    .map(parse_uuid_v7)
                    .transpose()?,
                arm: outcome.arm.clone(),
                label: outcome.label.clone(),
                attribution_status: outcome.attribution_status.clone(),
                latency_ms: outcome.latency_ms,
                created_at_unix_ms: outcome.created_at_unix_ms,
                canonical_payload_hash: outcome.canonical_payload_hash.clone(),
            }));
        }
    }
    Err(corrupt())
}

fn stored_terminal_candidates(
    state: &StoredRootTerminalState,
    outcome: &StoredOutcomeRow,
) -> Result<Vec<ActiveRootTerminal>, LedgerError> {
    let representatives = if outcome.stable_terminal_error_class.is_some() {
        vec![Some(ActiveRepresentativeStatus::Error)]
    } else {
        vec![None, Some(ActiveRepresentativeStatus::Completed)]
    };
    let mut candidates = Vec::with_capacity(24);
    for closure in [
        ActiveRootClosure::OwnerEnd,
        ActiveRootClosure::Deadline,
        ActiveRootClosure::ProjectionFault,
        ActiveRootClosure::Orphaned,
        ActiveRootClosure::ShutdownOrphaned,
        ActiveRootClosure::AmbiguousExposure,
    ] {
        for label_complete in [false, true] {
            for representative_status in &representatives {
                candidates.push(ActiveRootTerminal {
                    active_root_window_state_event_id: parse_uuid_v7(&state.state_event_id)?,
                    outcome_id: parse_uuid_v7(&outcome.outcome_id)?,
                    active_root_window_id: parse_uuid_v7(&outcome.active_root_window_id)?,
                    closure,
                    label_complete,
                    representative_status: *representative_status,
                    stable_terminal_error_class: outcome.stable_terminal_error_class.clone(),
                    interval_end_unix_ms: u64::try_from(outcome.interval_end_unix_ms)
                        .map_err(|_| corrupt())?,
                    neighborhood_invalidation: None,
                });
            }
        }
    }
    Ok(candidates)
}

fn stored_outcome_matches(
    state: &StoredRootTerminalState,
    outcome: &StoredOutcomeRow,
    terminal: &ActiveRootTerminal,
    facts: &RootOutcomeFacts,
) -> Result<bool, LedgerError> {
    let state_process = parse_uuid_v7(&state.process_instance_id)?;
    let stored_label_complete = terminal.label_complete && facts.exposure.root_state == "completed";
    let expected_state_hash = root_state_payload_hash(
        terminal.active_root_window_state_event_id,
        terminal.active_root_window_id,
        facts.exposure.root_state,
        stored_label_complete,
        facts.signal_count,
        facts.signal_size_bytes,
        state_process,
        state.created_at_unix_ms,
    )?;
    let outcome_process = parse_uuid_v7(&outcome.process_instance_id)?;
    let latency_bits = outcome.latency_ms_bits as u64;
    let expected_outcome_hash = outcome_payload_hash(
        outcome_process,
        terminal,
        facts,
        latency_bits,
        outcome.created_at_unix_ms,
    )?;
    let decision_id = facts.decision_id.to_string();
    Ok(
        state.state_event_id == terminal.active_root_window_state_event_id.to_string()
            && state.state == facts.exposure.root_state
            && state.label_complete == i64::from(stored_label_complete)
            && u64::try_from(state.signal_count).ok() == Some(facts.signal_count)
            && u64::try_from(state.signal_size_bytes).ok() == Some(facts.signal_size_bytes)
            && state.canonical_payload_hash == expected_state_hash
            && outcome.outcome_id == terminal.outcome_id.to_string()
            && outcome.active_experiment_id == facts.active_experiment_id.to_string()
            && outcome.active_root_window_id == terminal.active_root_window_id.to_string()
            && outcome.active_assignment_id == facts.active_assignment_id.to_string()
            && outcome.representative_decision_id.as_deref() == Some(decision_id.as_str())
            && outcome.root_key == facts.root_key
            && outcome.outcome_policy_hash == facts.outcome_policy_hash
            && outcome.source_signal_aggregate_hash == facts.signal_aggregate_hash
            && outcome.arm == facts.arm.as_str()
            && u64::try_from(outcome.total_ordinal).ok() == Some(facts.total_ordinal)
            && outcome
                .cap_ordinal
                .and_then(|value| u64::try_from(value).ok())
                == facts.cap_ordinal
            && outcome.label.as_deref() == facts.exposure.label
            && outcome.attribution_status == facts.exposure.attribution_status
            && u64::try_from(outcome.interval_start_unix_ms).ok() == Some(facts.admission_unix_ms)
            && u64::try_from(outcome.interval_end_unix_ms).ok()
                == Some(terminal.interval_end_unix_ms)
            && outcome.latency_ms.to_bits() == latency_bits
            && outcome.stable_terminal_error_class == facts.exposure.stable_terminal_error_class
            && outcome.canonical_payload_hash == expected_outcome_hash,
    )
}

fn outcome_payload_hash(
    process_instance_id: Uuid,
    terminal: &ActiveRootTerminal,
    facts: &RootOutcomeFacts,
    latency_ms_bits: u64,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(json!({
        "shape": "active_outcome_v1",
        "outcome_id": terminal.outcome_id,
        "active_experiment_id": facts.active_experiment_id,
        "active_root_window_id": terminal.active_root_window_id,
        "active_assignment_id": facts.active_assignment_id,
        "representative_decision_id": facts.decision_id,
        "root_key": facts.root_key,
        "outcome_policy_hash": facts.outcome_policy_hash,
        "source_signal_aggregate_hash": facts.signal_aggregate_hash,
        "arm": facts.arm.as_str(),
        "total_ordinal": facts.total_ordinal.to_string(),
        "cap_ordinal": facts.cap_ordinal.map(|value| value.to_string()),
        "label": facts.exposure.label,
        "reduced_label": facts.reduced_label,
        "label_complete": terminal.label_complete,
        "closure": terminal.closure.as_str(),
        "attribution_status": facts.exposure.attribution_status,
        "interval_start_unix_ms": facts.admission_unix_ms.to_string(),
        "interval_end_unix_ms": terminal.interval_end_unix_ms.to_string(),
        "attribution_deadline_unix_ms": facts.attribution_deadline_unix_ms.to_string(),
        "latency_ms_bits": format!("{latency_ms_bits:016x}"),
        "stable_terminal_error_class": facts.exposure.stable_terminal_error_class,
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": created_at_unix_ms.to_string(),
    }))
}

#[derive(Debug, Clone)]
struct PreparedSignal {
    signal: ActiveProtectedSignal,
    canonical_payload_hash: String,
    size_bytes: usize,
}

fn prepare_signal_batch(batch: &ActiveSignalBatch) -> Result<Vec<PreparedSignal>, LedgerError> {
    if !is_uuid_v7(batch.active_root_window_id)
        || batch.signals.is_empty()
        || batch.signals.len() > ACTIVE_ROOT_SIGNALS_MAX
    {
        return Err(invariant());
    }
    let mut identities = BTreeSet::new();
    let mut ids = BTreeSet::new();
    let mut prepared = Vec::with_capacity(batch.signals.len());
    for signal in &batch.signals {
        if !is_uuid_v7(signal.active_root_signal_id)
            || !is_hash(&signal.signal_identity_hash)
            || !valid_text(&signal.event_kind, 64)
            || !valid_text(&signal.scope_phase, 64)
            || !valid_text(&signal.category, 256)
            || !valid_text(&signal.name, 256)
            || signal.observed_at_unix_ms > i64::MAX as u64
            || !identities.insert(signal.signal_identity_hash.clone())
            || !ids.insert(signal.active_root_signal_id)
        {
            return Err(invariant());
        }
        let parsed: Json =
            serde_json::from_str(&signal.canonical_signal_json).map_err(|_| invariant())?;
        let canonical = canonical_json(&parsed)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let size_bytes = canonical.len();
        if canonical != signal.canonical_signal_json
            || !(2..=ACTIVE_ROOT_SIGNAL_BYTES_MAX).contains(&size_bytes)
        {
            return Err(invariant());
        }
        prepared.push(PreparedSignal {
            canonical_payload_hash: signal_payload_hash(signal)?,
            signal: signal.clone(),
            size_bytes,
        });
    }
    Ok(prepared)
}

fn append_active_signals_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    active_root_window_id: Uuid,
    prepared: &[PreparedSignal],
) -> Result<ActiveSignalBatchAck, LedgerError> {
    let window_project = transaction
        .query_row(
            "SELECT project_uuid FROM active_root_windows WHERE active_root_window_id = ?1",
            [active_root_window_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    if window_project != project_uuid.to_string() {
        return Err(corrupt());
    }
    if !root_window_is_open(transaction, active_root_window_id)? {
        return Ok(ActiveSignalBatchAck::WindowClosed);
    }

    let (existing_count, existing_bytes) = load_signal_totals(transaction, active_root_window_id)?;
    let mut new_signals = Vec::new();
    for signal in prepared {
        let existing = transaction
            .query_row(
                "SELECT active_root_signal_id, event_kind, scope_phase, category, name,
                        disposition, observed_at_unix_ms, canonical_signal_json,
                        canonical_payload_hash
                 FROM active_root_signals
                 WHERE active_root_window_id = ?1 AND signal_identity_hash = ?2",
                params![
                    active_root_window_id.to_string(),
                    signal.signal.signal_identity_hash,
                ],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, String>(5)?,
                        row.get::<_, i64>(6)?,
                        row.get::<_, String>(7)?,
                        row.get::<_, String>(8)?,
                    ))
                },
            )
            .optional()
            .map_err(database_error)?;
        if let Some(existing) = existing {
            if existing.0 != signal.signal.active_root_signal_id.to_string()
                || existing.1 != signal.signal.event_kind
                || existing.2 != signal.signal.scope_phase
                || existing.3 != signal.signal.category
                || existing.4 != signal.signal.name
                || existing.5 != signal.signal.disposition.as_str()
                || u64::try_from(existing.6).ok() != Some(signal.signal.observed_at_unix_ms)
                || existing.7 != signal.signal.canonical_signal_json
                || existing.8 != signal.canonical_payload_hash
            {
                return Err(corrupt());
            }
        } else {
            new_signals.push(signal);
        }
    }
    let new_bytes = new_signals.iter().try_fold(0_usize, |total, signal| {
        total.checked_add(signal.size_bytes)
    });
    let Some(final_count) = existing_count.checked_add(new_signals.len()) else {
        return Ok(ActiveSignalBatchAck::CapacityExceeded);
    };
    let Some(final_bytes) = existing_bytes.checked_add(new_bytes.unwrap_or(usize::MAX)) else {
        return Ok(ActiveSignalBatchAck::CapacityExceeded);
    };
    if final_count > ACTIVE_ROOT_SIGNALS_MAX || final_bytes > ACTIVE_ROOT_SIGNAL_BYTES_MAX {
        return Ok(ActiveSignalBatchAck::CapacityExceeded);
    }
    for (offset, signal) in new_signals.iter().enumerate() {
        let ordinal = existing_count.checked_add(offset).ok_or_else(corrupt)?;
        transaction
            .execute(
                "INSERT INTO active_root_signals (
                    active_root_signal_id, active_root_window_id, signal_ordinal,
                    signal_identity_hash, event_kind, scope_phase, category, name,
                    disposition, observed_at_unix_ms, canonical_signal_json,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                params![
                    signal.signal.active_root_signal_id.to_string(),
                    active_root_window_id.to_string(),
                    i64::try_from(ordinal).map_err(|_| corrupt())?,
                    signal.signal.signal_identity_hash,
                    signal.signal.event_kind,
                    signal.signal.scope_phase,
                    signal.signal.category,
                    signal.signal.name,
                    signal.signal.disposition.as_str(),
                    i64::try_from(signal.signal.observed_at_unix_ms).map_err(|_| corrupt())?,
                    signal.signal.canonical_signal_json,
                    signal.canonical_payload_hash,
                ],
            )
            .map_err(database_error)?;
    }
    let acknowledgement = if new_signals.is_empty() {
        ActiveSignalBatchAck::AlreadyApplied {
            signal_count: final_count as u64,
            signal_size_bytes: final_bytes as u64,
        }
    } else {
        ActiveSignalBatchAck::Applied {
            signal_count: final_count as u64,
            signal_size_bytes: final_bytes as u64,
        }
    };
    Ok(acknowledgement)
}

#[derive(Debug, Clone, Copy)]
struct AdmissionLimits {
    max_canary_roots: u64,
    tranche_nonholdout_limit: u64,
    tranche_total_limit: u64,
    cap_full: bool,
}

#[derive(Debug, Clone, Copy)]
struct AdmissionOrdinals {
    total: u64,
    cap: Option<u64>,
    tranche_total: u64,
    tranche_nonholdout: Option<u64>,
}

fn load_admission_limits(
    transaction: &Transaction<'_>,
    admission: &ActiveRootAdmission,
) -> Result<AdmissionLimits, LedgerError> {
    let limits = transaction
        .query_row(
            "SELECT experiment.max_canary_roots,
                    tranche.nonholdout_limit, tranche.total_limit
             FROM active_experiments AS experiment
             JOIN active_experiment_tranches AS tranche
               ON tranche.active_experiment_id = experiment.active_experiment_id
             WHERE experiment.active_experiment_id = ?1
               AND tranche.tranche_ordinal = ?2",
            params![
                admission.active_experiment_id.to_string(),
                i64::try_from(admission.tranche_ordinal).map_err(|_| invariant())?,
            ],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    let max_canary_roots = u64::try_from(limits.0).map_err(|_| corrupt())?;
    let cap_count = assignment_count(transaction, admission.active_experiment_id, true, None)?.0;
    Ok(AdmissionLimits {
        max_canary_roots,
        tranche_nonholdout_limit: u64::try_from(limits.1).map_err(|_| corrupt())?,
        tranche_total_limit: u64::try_from(limits.2).map_err(|_| corrupt())?,
        cap_full: admission.arm.is_nonholdout() && cap_count >= max_canary_roots,
    })
}

fn allocate_ordinals(
    transaction: &Transaction<'_>,
    admission: &ActiveRootAdmission,
    limits: AdmissionLimits,
) -> Result<Option<AdmissionOrdinals>, LedgerError> {
    let (total_count, total_max) =
        assignment_count(transaction, admission.active_experiment_id, false, None)?;
    let (cap_count, cap_max) =
        assignment_count(transaction, admission.active_experiment_id, true, None)?;
    let (tranche_total_count, tranche_total_max) = assignment_count(
        transaction,
        admission.active_experiment_id,
        false,
        Some(admission.tranche_ordinal),
    )?;
    let (tranche_cap_count, tranche_cap_max) = assignment_count(
        transaction,
        admission.active_experiment_id,
        true,
        Some(admission.tranche_ordinal),
    )?;
    if total_count != total_max
        || cap_count != cap_max
        || tranche_total_count != tranche_total_max
        || tranche_cap_count != tranche_cap_max
    {
        return Err(corrupt());
    }
    let next_total = checked_successor(total_max)?;
    let next_tranche_total = checked_successor(tranche_total_max)?;
    let next_cap = admission
        .arm
        .is_nonholdout()
        .then(|| checked_successor(cap_max))
        .transpose()?;
    let next_tranche_cap = admission
        .arm
        .is_nonholdout()
        .then(|| checked_successor(tranche_cap_max))
        .transpose()?;
    let lifetime_total_limit = limits.max_canary_roots.checked_mul(2).ok_or_else(corrupt)?;
    if next_total > lifetime_total_limit
        || next_tranche_total > limits.tranche_total_limit
        || next_cap.is_some_and(|value| value > limits.max_canary_roots)
        || next_tranche_cap.is_some_and(|value| value > limits.tranche_nonholdout_limit)
    {
        return Ok(None);
    }
    Ok(Some(AdmissionOrdinals {
        total: next_total,
        cap: next_cap,
        tranche_total: next_tranche_total,
        tranche_nonholdout: next_tranche_cap,
    }))
}

fn assignment_count(
    connection: &Connection,
    experiment_id: Uuid,
    nonholdout: bool,
    tranche: Option<u64>,
) -> Result<(u64, u64), LedgerError> {
    let column = match (nonholdout, tranche.is_some()) {
        (false, false) => "total_ordinal",
        (true, false) => "cap_ordinal",
        (false, true) => "tranche_total_ordinal",
        (true, true) => "tranche_nonholdout_ordinal",
    };
    let sql = format!(
        "SELECT count({column}), coalesce(max({column}), 0)
         FROM active_assignments
         WHERE active_experiment_id = ?1
           AND (?2 IS NULL OR tranche_ordinal = ?2)"
    );
    let tranche = tranche
        .map(i64::try_from)
        .transpose()
        .map_err(|_| invariant())?;
    let values = connection
        .query_row(&sql, params![experiment_id.to_string(), tranche], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?))
        })
        .map_err(database_error)?;
    Ok((
        u64::try_from(values.0).map_err(|_| corrupt())?,
        u64::try_from(values.1).map_err(|_| corrupt())?,
    ))
}

fn checked_successor(value: u64) -> Result<u64, LedgerError> {
    value
        .checked_add(1)
        .filter(|value| *value <= i64::MAX as u64)
        .ok_or_else(corrupt)
}

fn validate_active_admission(admission: &ActiveRootAdmission) -> Result<(), LedgerError> {
    let identifiers = [
        admission.active_root_window_id,
        admission.active_experiment_id,
        admission.active_assignment_id,
        admission.decision_id,
    ];
    if identifiers.into_iter().any(|value| !is_uuid_v7(value))
        || admission
            .dispatch
            .as_ref()
            .is_some_and(|dispatch| !is_uuid_v7(dispatch.active_dispatch_id))
        || !valid_text(&admission.pool_id, 128)
        || !valid_text(&admission.candidate_id, 128)
        || !is_hash(&admission.root_key)
        || !is_hash(&admission.owner_relation_hash)
        || !is_hash(&admission.config_generation_id)
        || !is_hash(&admission.outcome_policy_hash)
        || admission.opened_after_ingest_seq > i64::MAX as u64
        || admission.attribution_deadline_unix_ms > i64::MAX as u64
        || admission.tranche_ordinal == 0
        || admission.tranche_ordinal > i64::MAX as u64
        || admission.control_generation > i64::MAX as u64
        || admission
            .dispatch
            .as_ref()
            .is_some_and(|dispatch| !is_hash(&dispatch.request_identity_hash))
        || (admission.arm == ActiveAssignmentArm::CandidateTreatment)
            != admission.dispatch.is_some()
    {
        return Err(invariant());
    }
    let holdout = probability(admission.configured_holdout_probability_bits, true)?;
    let canary = probability(admission.configured_canary_probability_bits, true)?;
    let effective = probability(admission.effective_arm_probability_bits, false)?;
    let conditional = probability(admission.conditional_selection_probability_bits, false)?;
    let propensity = probability(admission.propensity_bits, false)?;
    if holdout + canary > 1.0
        || (effective * conditional).to_bits() != propensity.to_bits()
        || (admission.arm == ActiveAssignmentArm::NonLearning
            && (effective != 1.0 || conditional != 1.0 || propensity != 1.0))
    {
        return Err(invariant());
    }
    if admission.arm != ActiveAssignmentArm::NonLearning {
        for threshold in [
            admission.cohort_threshold_numerator,
            admission.selection_threshold_numerator,
        ] {
            let value = u64::from_be_bytes(threshold);
            if value == 0 {
                return Err(invariant());
            }
        }
    }
    Ok(())
}

fn probability(bits: u64, allow_zero: bool) -> Result<f64, LedgerError> {
    let value = f64::from_bits(bits);
    if !value.is_finite()
        || value.is_sign_negative()
        || value > 1.0
        || (!allow_zero && value == 0.0)
        || (value == 0.0 && bits != 0)
    {
        return Err(invariant());
    }
    Ok(value)
}

fn active_admission_authority_matches(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    admission: &ActiveRootAdmission,
) -> Result<bool, LedgerError> {
    if !super::process::originating_process_is_live(connection, project_uuid, process_instance_id)?
        || !all_live_processes_support_active_v6(connection, project_uuid)?
    {
        return Ok(false);
    }
    let current_config = connection
        .query_row(
            "SELECT config_generation_id FROM config_generation_state_events
             WHERE project_uuid = ?1 ORDER BY config_epoch DESC LIMIT 1",
            [project_uuid.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let current_learning = connection
        .query_row(
            "SELECT learning_generation_id FROM learning_generation_state_events
             WHERE project_uuid = ?1 AND pool_id = ?2
             ORDER BY event_seq DESC LIMIT 1",
            params![project_uuid.to_string(), admission.pool_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let current_cohort = connection
        .query_row(
            "SELECT cohort_generation_id FROM cohort_generation_state_events
             WHERE project_uuid = ?1 ORDER BY event_seq DESC LIMIT 1",
            [project_uuid.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let current_control = connection
        .query_row(
            "SELECT max(control_generation) FROM controls WHERE project_uuid = ?1",
            [project_uuid.to_string()],
            |row| row.get::<_, Option<i64>>(0),
        )
        .map_err(database_error)?;
    let learning_generation_id = admission.learning_generation_id.to_string();
    let cohort_generation_id = admission.cohort_generation_id.to_string();
    if current_config.as_deref() != Some(admission.config_generation_id.as_str())
        || current_learning.as_deref() != Some(learning_generation_id.as_str())
        || current_cohort.as_deref() != Some(cohort_generation_id.as_str())
        || current_control.and_then(|value| u64::try_from(value).ok())
            != Some(admission.control_generation)
    {
        return Ok(false);
    }

    let experiment = connection
        .query_row(
            "SELECT project_uuid, pool_id, candidate_id, config_generation_id,
                    learning_generation_id, cohort_generation_id, outcome_policy_hash
             FROM active_experiments WHERE active_experiment_id = ?1",
            [admission.active_experiment_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(experiment) = experiment else {
        return Ok(false);
    };
    if experiment
        != (
            project_uuid.to_string(),
            admission.pool_id.clone(),
            admission.candidate_id.clone(),
            admission.config_generation_id.clone(),
            admission.learning_generation_id.to_string(),
            admission.cohort_generation_id.to_string(),
            admission.outcome_policy_hash.clone(),
        )
    {
        return Ok(false);
    }
    let retiring: bool = connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM active_retirement_markers WHERE active_experiment_id = ?1
             )",
            [admission.active_experiment_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    if retiring {
        return Ok(false);
    }

    let decision = connection
        .query_row(
            "SELECT decision_shape_version, project_uuid, config_generation_id,
                    learning_generation_id, cohort_generation_id, active_experiment_id,
                    pool_id, candidate_id, root_key, mode, final_reason
             FROM decisions WHERE decision_id = ?1",
            [admission.decision_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, Option<String>>(7)?,
                    row.get::<_, Option<String>>(8)?,
                    row.get::<_, String>(9)?,
                    row.get::<_, String>(10)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    Ok(matches!(
        decision,
        Some((
            2,
            ref stored_project,
            ref config,
            ref learning,
            Some(ref cohort),
            Some(ref experiment_id),
            ref pool,
            Some(ref candidate),
            Some(ref root_key),
            ref mode,
            ref final_reason,
        )) if stored_project == &project_uuid.to_string()
            && config == &admission.config_generation_id
            && learning == &admission.learning_generation_id.to_string()
            && cohort == &admission.cohort_generation_id.to_string()
            && experiment_id == &admission.active_experiment_id.to_string()
            && pool == &admission.pool_id
            && candidate == &admission.candidate_id
            && root_key == &admission.root_key
            && mode == "active"
            && admission.arm.matches_decision_reason(final_reason)
    ))
}

fn active_admission_lifecycle_ack(
    connection: &Connection,
    admission: &ActiveRootAdmission,
) -> Result<Option<ActiveAdmissionAck>, LedgerError> {
    let experiment_state = latest_string_state(
        connection,
        "active_experiment_state_events",
        "active_experiment_id",
        &admission.active_experiment_id.to_string(),
        None,
    )?;
    match experiment_state.as_deref() {
        Some("collecting") => {}
        Some("cap_draining") => return Ok(Some(ActiveAdmissionAck::ExperimentCapReached)),
        Some("terminal") => return Ok(Some(ActiveAdmissionAck::AuthorityChanged)),
        _ => return Err(corrupt()),
    }
    let tranche_state = latest_string_state(
        connection,
        "active_tranche_state_events",
        "active_experiment_id",
        &admission.active_experiment_id.to_string(),
        Some(admission.tranche_ordinal),
    )?;
    match tranche_state.as_deref() {
        Some("open") => Ok(None),
        Some("closed" | "drained" | "evaluating" | "complete") => {
            Ok(Some(ActiveAdmissionAck::TrancheFull))
        }
        _ => Err(corrupt()),
    }
}

pub(super) fn all_live_processes_support_active_v6(
    connection: &Connection,
    project_uuid: Uuid,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT NOT EXISTS(
                SELECT 1 FROM process_instances AS process
                WHERE process.project_uuid = ?1
                  AND NOT EXISTS (
                    SELECT 1 FROM process_instance_state_events AS state
                    WHERE state.subject_process_instance_id = process.process_instance_id
                      AND state.state IN ('stopped', 'reconciled')
                  )
                  AND NOT EXISTS (
                    SELECT 1 FROM process_writer_capabilities AS capability
                    WHERE capability.process_instance_id = process.process_instance_id
                      AND capability.project_uuid = process.project_uuid
                      AND capability.writer_protocol = 'active-v6'
                      AND capability.schema_version = 6
                  )
             )",
            [project_uuid.to_string()],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)
}

fn latest_string_state(
    connection: &Connection,
    table: &str,
    key_column: &str,
    key: &str,
    tranche_ordinal: Option<u64>,
) -> Result<Option<String>, LedgerError> {
    let sql = if tranche_ordinal.is_some() {
        format!(
            "SELECT state FROM {table}
             WHERE {key_column} = ?1 AND tranche_ordinal = ?2
             ORDER BY event_seq DESC LIMIT 1"
        )
    } else {
        format!(
            "SELECT state FROM {table}
             WHERE {key_column} = ?1 ORDER BY event_seq DESC LIMIT 1"
        )
    };
    let tranche = tranche_ordinal
        .map(i64::try_from)
        .transpose()
        .map_err(|_| invariant())?;
    match tranche {
        Some(tranche) => connection
            .query_row(&sql, params![key, tranche], |row| row.get::<_, String>(0))
            .optional()
            .map_err(database_error),
        None => connection
            .query_row(&sql, [key], |row| row.get::<_, String>(0))
            .optional()
            .map_err(database_error),
    }
}

#[allow(clippy::too_many_arguments)]
fn insert_admission_graph(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    admission: &ActiveRootAdmission,
    ordinals: AdmissionOrdinals,
    admitted_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let root_hash = root_payload_hash(
        project_uuid,
        process_instance_id,
        admission,
        admitted_at_unix_ms,
    )?;
    execute_one(
        transaction,
        "INSERT INTO active_root_windows (
            active_root_window_id, active_experiment_id, project_uuid,
            pool_id, candidate_id, root_key, owner_relation_hash,
            config_generation_id, learning_generation_id, cohort_generation_id,
            outcome_policy_hash, opened_after_ingest_seq,
            attribution_deadline_unix_ms, process_instance_id,
            created_at_unix_ms, canonical_payload_hash
         ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8,
            ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16
         )",
        params![
            admission.active_root_window_id.to_string(),
            admission.active_experiment_id.to_string(),
            project_uuid.to_string(),
            admission.pool_id,
            admission.candidate_id,
            admission.root_key,
            admission.owner_relation_hash,
            admission.config_generation_id,
            admission.learning_generation_id.to_string(),
            admission.cohort_generation_id.to_string(),
            admission.outcome_policy_hash,
            i64::try_from(admission.opened_after_ingest_seq).map_err(|_| invariant())?,
            i64::try_from(admission.attribution_deadline_unix_ms).map_err(|_| invariant())?,
            process_instance_id.to_string(),
            admitted_at_unix_ms,
            root_hash,
        ],
    )?;

    let root_state_event_id = Uuid::now_v7();
    let root_state_hash = root_state_payload_hash(
        root_state_event_id,
        admission.active_root_window_id,
        "open",
        false,
        0,
        0,
        process_instance_id,
        admitted_at_unix_ms,
    )?;
    execute_one(
        transaction,
        "INSERT INTO active_root_window_state_events (
            active_root_window_state_event_id, active_root_window_id, state,
            label_complete, signal_count, signal_size_bytes,
            process_instance_id, created_at_unix_ms, canonical_payload_hash
         ) VALUES (?1, ?2, 'open', 0, 0, 0, ?3, ?4, ?5)",
        params![
            root_state_event_id.to_string(),
            admission.active_root_window_id.to_string(),
            process_instance_id.to_string(),
            admitted_at_unix_ms,
            root_state_hash,
        ],
    )?;

    let assignment_hash = assignment_payload_hash(
        process_instance_id,
        admission,
        ordinals,
        admitted_at_unix_ms,
    )?;
    let holdout = f64::from_bits(admission.configured_holdout_probability_bits);
    let canary = f64::from_bits(admission.configured_canary_probability_bits);
    let effective = f64::from_bits(admission.effective_arm_probability_bits);
    let conditional = f64::from_bits(admission.conditional_selection_probability_bits);
    let propensity = f64::from_bits(admission.propensity_bits);
    execute_one(
        transaction,
        "INSERT INTO active_assignments (
            active_assignment_id, active_experiment_id, active_root_window_id,
            decision_id, tranche_ordinal, total_ordinal, cap_ordinal,
            tranche_total_ordinal, tranche_nonholdout_ordinal, arm,
            cohort_threshold_numerator, selection_threshold_numerator,
            configured_holdout_probability, configured_holdout_probability_bits,
            configured_canary_probability, configured_canary_probability_bits,
            effective_arm_probability, effective_arm_probability_bits,
            conditional_selection_probability, conditional_selection_probability_bits,
            propensity, propensity_bits, control_generation,
            admission_unix_ms, process_instance_id, canonical_payload_hash
         ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10,
            ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20,
            ?21, ?22, ?23, ?24, ?25, ?26
         )",
        params![
            admission.active_assignment_id.to_string(),
            admission.active_experiment_id.to_string(),
            admission.active_root_window_id.to_string(),
            admission.decision_id.to_string(),
            i64::try_from(admission.tranche_ordinal).map_err(|_| invariant())?,
            i64::try_from(ordinals.total).map_err(|_| corrupt())?,
            ordinals
                .cap
                .map(i64::try_from)
                .transpose()
                .map_err(|_| corrupt())?,
            i64::try_from(ordinals.tranche_total).map_err(|_| corrupt())?,
            ordinals
                .tranche_nonholdout
                .map(i64::try_from)
                .transpose()
                .map_err(|_| corrupt())?,
            admission.arm.as_str(),
            admission.cohort_threshold_numerator.as_slice(),
            admission.selection_threshold_numerator.as_slice(),
            holdout,
            admission.configured_holdout_probability_bits as i64,
            canary,
            admission.configured_canary_probability_bits as i64,
            effective,
            admission.effective_arm_probability_bits as i64,
            conditional,
            admission.conditional_selection_probability_bits as i64,
            propensity,
            admission.propensity_bits as i64,
            i64::try_from(admission.control_generation).map_err(|_| invariant())?,
            admitted_at_unix_ms,
            process_instance_id.to_string(),
            assignment_hash,
        ],
    )?;

    let link_hash = decision_link_payload_hash(admission, admitted_at_unix_ms)?;
    execute_one(
        transaction,
        "INSERT INTO active_root_decision_links (
            active_root_window_id, decision_id, owner_relation_hash,
            linked_at_unix_ms, canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            admission.active_root_window_id.to_string(),
            admission.decision_id.to_string(),
            admission.owner_relation_hash,
            admitted_at_unix_ms,
            link_hash,
        ],
    )?;

    if let Some(dispatch) = &admission.dispatch {
        let dispatch_hash = dispatch_payload_hash(
            process_instance_id,
            admission,
            dispatch,
            admitted_at_unix_ms,
        )?;
        execute_one(
            transaction,
            "INSERT INTO active_dispatches (
                active_dispatch_id, active_experiment_id, active_assignment_id,
                decision_id, candidate_id, request_identity_hash,
                process_instance_id, admitted_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                dispatch.active_dispatch_id.to_string(),
                admission.active_experiment_id.to_string(),
                admission.active_assignment_id.to_string(),
                admission.decision_id.to_string(),
                admission.candidate_id,
                dispatch.request_identity_hash,
                process_instance_id.to_string(),
                admitted_at_unix_ms,
                dispatch_hash,
            ],
        )?;
    }
    Ok(())
}

#[derive(Debug)]
struct StoredRoot {
    active_root_window_id: String,
    active_experiment_id: String,
    pool_id: String,
    candidate_id: String,
    root_key: String,
    owner_relation_hash: String,
    config_generation_id: String,
    learning_generation_id: String,
    cohort_generation_id: String,
    outcome_policy_hash: String,
    opened_after_ingest_seq: i64,
    attribution_deadline_unix_ms: i64,
    process_instance_id: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

fn load_existing_admission(
    connection: &Connection,
    project_uuid: Uuid,
    admission: &ActiveRootAdmission,
) -> Result<Option<ActiveAdmissionAck>, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT active_root_window_id, active_experiment_id, pool_id, candidate_id,
                    root_key, owner_relation_hash, config_generation_id,
                    learning_generation_id, cohort_generation_id, outcome_policy_hash,
                    opened_after_ingest_seq, attribution_deadline_unix_ms,
                    process_instance_id, created_at_unix_ms, canonical_payload_hash
             FROM active_root_windows
             WHERE project_uuid = ?1 AND cohort_generation_id = ?2 AND root_key = ?3",
            params![
                project_uuid.to_string(),
                admission.cohort_generation_id.to_string(),
                admission.root_key,
            ],
            |row| {
                Ok(StoredRoot {
                    active_root_window_id: row.get(0)?,
                    active_experiment_id: row.get(1)?,
                    pool_id: row.get(2)?,
                    candidate_id: row.get(3)?,
                    root_key: row.get(4)?,
                    owner_relation_hash: row.get(5)?,
                    config_generation_id: row.get(6)?,
                    learning_generation_id: row.get(7)?,
                    cohort_generation_id: row.get(8)?,
                    outcome_policy_hash: row.get(9)?,
                    opened_after_ingest_seq: row.get(10)?,
                    attribution_deadline_unix_ms: row.get(11)?,
                    process_instance_id: row.get(12)?,
                    created_at_unix_ms: row.get(13)?,
                    canonical_payload_hash: row.get(14)?,
                })
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(stored) = stored else {
        let id_collision: bool = connection
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM active_root_windows WHERE active_root_window_id = ?1
                 )",
                [admission.active_root_window_id.to_string()],
                |row| row.get(0),
            )
            .map_err(database_error)?;
        return if id_collision {
            Err(corrupt())
        } else {
            Ok(None)
        };
    };
    let stored_process = parse_uuid_v7(&stored.process_instance_id)?;
    let expected_root_hash = root_payload_hash(
        project_uuid,
        stored_process,
        &ActiveRootAdmission {
            active_root_window_id: parse_uuid_v7(&stored.active_root_window_id)?,
            active_experiment_id: parse_uuid_v7(&stored.active_experiment_id)?,
            active_assignment_id: admission.active_assignment_id,
            decision_id: admission.decision_id,
            dispatch: admission.dispatch.clone(),
            pool_id: stored.pool_id.clone(),
            candidate_id: stored.candidate_id.clone(),
            root_key: stored.root_key.clone(),
            owner_relation_hash: stored.owner_relation_hash.clone(),
            config_generation_id: stored.config_generation_id.clone(),
            learning_generation_id: parse_uuid_v7(&stored.learning_generation_id)?,
            cohort_generation_id: parse_uuid_v7(&stored.cohort_generation_id)?,
            outcome_policy_hash: stored.outcome_policy_hash.clone(),
            opened_after_ingest_seq: u64::try_from(stored.opened_after_ingest_seq)
                .map_err(|_| corrupt())?,
            attribution_deadline_unix_ms: u64::try_from(stored.attribution_deadline_unix_ms)
                .map_err(|_| corrupt())?,
            tranche_ordinal: admission.tranche_ordinal,
            arm: admission.arm,
            cohort_threshold_numerator: admission.cohort_threshold_numerator,
            selection_threshold_numerator: admission.selection_threshold_numerator,
            configured_holdout_probability_bits: admission.configured_holdout_probability_bits,
            configured_canary_probability_bits: admission.configured_canary_probability_bits,
            effective_arm_probability_bits: admission.effective_arm_probability_bits,
            conditional_selection_probability_bits: admission
                .conditional_selection_probability_bits,
            propensity_bits: admission.propensity_bits,
            control_generation: admission.control_generation,
        },
        stored.created_at_unix_ms,
    )?;
    if expected_root_hash != stored.canonical_payload_hash {
        return Err(corrupt());
    }
    let semantic_match = stored.active_root_window_id
        == admission.active_root_window_id.to_string()
        && stored.active_experiment_id == admission.active_experiment_id.to_string()
        && stored.pool_id == admission.pool_id
        && stored.candidate_id == admission.candidate_id
        && stored.root_key == admission.root_key
        && stored.owner_relation_hash == admission.owner_relation_hash
        && stored.config_generation_id == admission.config_generation_id
        && stored.learning_generation_id == admission.learning_generation_id.to_string()
        && stored.cohort_generation_id == admission.cohort_generation_id.to_string()
        && stored.outcome_policy_hash == admission.outcome_policy_hash
        && u64::try_from(stored.opened_after_ingest_seq).ok()
            == Some(admission.opened_after_ingest_seq)
        && u64::try_from(stored.attribution_deadline_unix_ms).ok()
            == Some(admission.attribution_deadline_unix_ms);
    if !semantic_match {
        return Ok(Some(ActiveAdmissionAck::RootAlreadyAssigned));
    }
    let assignment = load_and_verify_assignment(connection, admission)?;
    verify_decision_link(connection, admission)?;
    verify_dispatch(
        connection,
        admission,
        stored_process,
        stored.created_at_unix_ms,
    )?;
    let _ = root_window_is_open(connection, admission.active_root_window_id)?;
    Ok(Some(ActiveAdmissionAck::AlreadyApplied(assignment)))
}

#[derive(Debug)]
struct StoredAssignment {
    active_assignment_id: String,
    active_experiment_id: String,
    active_root_window_id: String,
    decision_id: String,
    tranche_ordinal: i64,
    total_ordinal: i64,
    cap_ordinal: Option<i64>,
    tranche_total_ordinal: i64,
    tranche_nonholdout_ordinal: Option<i64>,
    arm: String,
    cohort_threshold_numerator: Vec<u8>,
    selection_threshold_numerator: Vec<u8>,
    configured_holdout_probability: f64,
    configured_holdout_probability_bits: i64,
    configured_canary_probability: f64,
    configured_canary_probability_bits: i64,
    effective_arm_probability: f64,
    effective_arm_probability_bits: i64,
    conditional_selection_probability: f64,
    conditional_selection_probability_bits: i64,
    propensity: f64,
    propensity_bits: i64,
    control_generation: i64,
    admission_unix_ms: i64,
    process_instance_id: String,
    canonical_payload_hash: String,
}

fn load_and_verify_assignment(
    connection: &Connection,
    admission: &ActiveRootAdmission,
) -> Result<ActiveAdmissionReceipt, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT active_assignment_id, active_experiment_id, active_root_window_id,
                    decision_id, tranche_ordinal, total_ordinal, cap_ordinal,
                    tranche_total_ordinal, tranche_nonholdout_ordinal, arm,
                    cohort_threshold_numerator, selection_threshold_numerator,
                    configured_holdout_probability, configured_holdout_probability_bits,
                    configured_canary_probability, configured_canary_probability_bits,
                    effective_arm_probability, effective_arm_probability_bits,
                    conditional_selection_probability, conditional_selection_probability_bits,
                    propensity, propensity_bits, control_generation,
                    admission_unix_ms, process_instance_id, canonical_payload_hash
             FROM active_assignments WHERE active_root_window_id = ?1",
            [admission.active_root_window_id.to_string()],
            |row| {
                Ok(StoredAssignment {
                    active_assignment_id: row.get(0)?,
                    active_experiment_id: row.get(1)?,
                    active_root_window_id: row.get(2)?,
                    decision_id: row.get(3)?,
                    tranche_ordinal: row.get(4)?,
                    total_ordinal: row.get(5)?,
                    cap_ordinal: row.get(6)?,
                    tranche_total_ordinal: row.get(7)?,
                    tranche_nonholdout_ordinal: row.get(8)?,
                    arm: row.get(9)?,
                    cohort_threshold_numerator: row.get(10)?,
                    selection_threshold_numerator: row.get(11)?,
                    configured_holdout_probability: row.get(12)?,
                    configured_holdout_probability_bits: row.get(13)?,
                    configured_canary_probability: row.get(14)?,
                    configured_canary_probability_bits: row.get(15)?,
                    effective_arm_probability: row.get(16)?,
                    effective_arm_probability_bits: row.get(17)?,
                    conditional_selection_probability: row.get(18)?,
                    conditional_selection_probability_bits: row.get(19)?,
                    propensity: row.get(20)?,
                    propensity_bits: row.get(21)?,
                    control_generation: row.get(22)?,
                    admission_unix_ms: row.get(23)?,
                    process_instance_id: row.get(24)?,
                    canonical_payload_hash: row.get(25)?,
                })
            },
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    let ordinals = AdmissionOrdinals {
        total: u64::try_from(stored.total_ordinal).map_err(|_| corrupt())?,
        cap: stored
            .cap_ordinal
            .map(u64::try_from)
            .transpose()
            .map_err(|_| corrupt())?,
        tranche_total: u64::try_from(stored.tranche_total_ordinal).map_err(|_| corrupt())?,
        tranche_nonholdout: stored
            .tranche_nonholdout_ordinal
            .map(u64::try_from)
            .transpose()
            .map_err(|_| corrupt())?,
    };
    let process_instance_id = parse_uuid_v7(&stored.process_instance_id)?;
    let expected_hash = assignment_payload_hash(
        process_instance_id,
        admission,
        ordinals,
        stored.admission_unix_ms,
    )?;
    let semantic_match = stored.active_assignment_id == admission.active_assignment_id.to_string()
        && stored.active_experiment_id == admission.active_experiment_id.to_string()
        && stored.active_root_window_id == admission.active_root_window_id.to_string()
        && stored.decision_id == admission.decision_id.to_string()
        && u64::try_from(stored.tranche_ordinal).ok() == Some(admission.tranche_ordinal)
        && stored.arm == admission.arm.as_str()
        && stored.cohort_threshold_numerator == admission.cohort_threshold_numerator
        && stored.selection_threshold_numerator == admission.selection_threshold_numerator
        && stored.configured_holdout_probability_bits
            == admission.configured_holdout_probability_bits as i64
        && stored.configured_canary_probability_bits
            == admission.configured_canary_probability_bits as i64
        && stored.effective_arm_probability_bits == admission.effective_arm_probability_bits as i64
        && stored.conditional_selection_probability_bits
            == admission.conditional_selection_probability_bits as i64
        && stored.propensity_bits == admission.propensity_bits as i64
        && u64::try_from(stored.control_generation).ok() == Some(admission.control_generation)
        && stored.configured_holdout_probability.to_bits()
            == admission.configured_holdout_probability_bits
        && stored.configured_canary_probability.to_bits()
            == admission.configured_canary_probability_bits
        && stored.effective_arm_probability.to_bits() == admission.effective_arm_probability_bits
        && stored.conditional_selection_probability.to_bits()
            == admission.conditional_selection_probability_bits
        && stored.propensity.to_bits() == admission.propensity_bits;
    if !semantic_match || expected_hash != stored.canonical_payload_hash {
        return Err(corrupt());
    }
    Ok(ActiveAdmissionReceipt {
        total_ordinal: ordinals.total,
        cap_ordinal: ordinals.cap,
        tranche_total_ordinal: ordinals.tranche_total,
        tranche_nonholdout_ordinal: ordinals.tranche_nonholdout,
        admitted_at_unix_ms: u64::try_from(stored.admission_unix_ms).map_err(|_| corrupt())?,
    })
}

fn verify_decision_link(
    connection: &Connection,
    admission: &ActiveRootAdmission,
) -> Result<(), LedgerError> {
    let stored = connection
        .query_row(
            "SELECT decision_id, owner_relation_hash, linked_at_unix_ms,
                    canonical_payload_hash
             FROM active_root_decision_links WHERE active_root_window_id = ?1",
            [admission.active_root_window_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    if stored.0 != admission.decision_id.to_string()
        || stored.1 != admission.owner_relation_hash
        || decision_link_payload_hash(admission, stored.2)? != stored.3
    {
        return Err(corrupt());
    }
    Ok(())
}

fn verify_dispatch(
    connection: &Connection,
    admission: &ActiveRootAdmission,
    _root_process_instance_id: Uuid,
    _root_created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let stored = connection
        .query_row(
            "SELECT active_dispatch_id, active_experiment_id, active_assignment_id,
                    decision_id, candidate_id, request_identity_hash,
                    process_instance_id, admitted_at_unix_ms, canonical_payload_hash
             FROM active_dispatches WHERE active_assignment_id = ?1",
            [admission.active_assignment_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, String>(8)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    match (&admission.dispatch, stored) {
        (None, None) => Ok(()),
        (Some(expected), Some(stored)) => {
            let process_instance_id = parse_uuid_v7(&stored.6)?;
            let expected_hash =
                dispatch_payload_hash(process_instance_id, admission, expected, stored.7)?;
            if stored.0 != expected.active_dispatch_id.to_string()
                || stored.1 != admission.active_experiment_id.to_string()
                || stored.2 != admission.active_assignment_id.to_string()
                || stored.3 != admission.decision_id.to_string()
                || stored.4 != admission.candidate_id
                || stored.5 != expected.request_identity_hash
                || stored.8 != expected_hash
            {
                return Err(corrupt());
            }
            Ok(())
        }
        _ => Err(corrupt()),
    }
}

fn root_payload_hash(
    project_uuid: Uuid,
    process_instance_id: Uuid,
    admission: &ActiveRootAdmission,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(json!({
        "shape": "active_root_window_v1",
        "active_root_window_id": admission.active_root_window_id,
        "active_experiment_id": admission.active_experiment_id,
        "project_uuid": project_uuid,
        "pool_id": admission.pool_id,
        "candidate_id": admission.candidate_id,
        "root_key": admission.root_key,
        "owner_relation_hash": admission.owner_relation_hash,
        "config_generation_id": admission.config_generation_id,
        "learning_generation_id": admission.learning_generation_id,
        "cohort_generation_id": admission.cohort_generation_id,
        "outcome_policy_hash": admission.outcome_policy_hash,
        "opened_after_ingest_seq": admission.opened_after_ingest_seq.to_string(),
        "attribution_deadline_unix_ms": admission.attribution_deadline_unix_ms.to_string(),
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": created_at_unix_ms.to_string(),
    }))
}

#[allow(clippy::too_many_arguments)]
fn root_state_payload_hash(
    event_id: Uuid,
    root_window_id: Uuid,
    state: &str,
    label_complete: bool,
    signal_count: u64,
    signal_size_bytes: u64,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(json!({
        "shape": "active_root_window_state_v1",
        "event_id": event_id,
        "active_root_window_id": root_window_id,
        "state": state,
        "label_complete": label_complete,
        "signal_count": signal_count.to_string(),
        "signal_size_bytes": signal_size_bytes.to_string(),
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": created_at_unix_ms.to_string(),
    }))
}

fn assignment_payload_hash(
    process_instance_id: Uuid,
    admission: &ActiveRootAdmission,
    ordinals: AdmissionOrdinals,
    admitted_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(json!({
        "shape": "active_assignment_v1",
        "active_assignment_id": admission.active_assignment_id,
        "active_experiment_id": admission.active_experiment_id,
        "active_root_window_id": admission.active_root_window_id,
        "decision_id": admission.decision_id,
        "tranche_ordinal": admission.tranche_ordinal.to_string(),
        "total_ordinal": ordinals.total.to_string(),
        "cap_ordinal": ordinals.cap.map(|value| value.to_string()),
        "tranche_total_ordinal": ordinals.tranche_total.to_string(),
        "tranche_nonholdout_ordinal": ordinals.tranche_nonholdout.map(|value| value.to_string()),
        "arm": admission.arm.as_str(),
        "cohort_threshold_numerator": hex_bytes(&admission.cohort_threshold_numerator),
        "selection_threshold_numerator": hex_bytes(&admission.selection_threshold_numerator),
        "configured_holdout_probability_bits": format!("{:016x}", admission.configured_holdout_probability_bits),
        "configured_canary_probability_bits": format!("{:016x}", admission.configured_canary_probability_bits),
        "effective_arm_probability_bits": format!("{:016x}", admission.effective_arm_probability_bits),
        "conditional_selection_probability_bits": format!("{:016x}", admission.conditional_selection_probability_bits),
        "propensity_bits": format!("{:016x}", admission.propensity_bits),
        "control_generation": admission.control_generation.to_string(),
        "admission_unix_ms": admitted_at_unix_ms.to_string(),
        "process_instance_id": process_instance_id,
    }))
}

fn decision_link_payload_hash(
    admission: &ActiveRootAdmission,
    linked_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(json!({
        "shape": "active_root_decision_link_v1",
        "active_root_window_id": admission.active_root_window_id,
        "decision_id": admission.decision_id,
        "owner_relation_hash": admission.owner_relation_hash,
        "linked_at_unix_ms": linked_at_unix_ms.to_string(),
    }))
}

fn dispatch_payload_hash(
    process_instance_id: Uuid,
    admission: &ActiveRootAdmission,
    dispatch: &ActiveDispatchAdmission,
    admitted_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(json!({
        "shape": "active_dispatch_v1",
        "active_dispatch_id": dispatch.active_dispatch_id,
        "active_experiment_id": admission.active_experiment_id,
        "active_assignment_id": admission.active_assignment_id,
        "decision_id": admission.decision_id,
        "candidate_id": admission.candidate_id,
        "request_identity_hash": dispatch.request_identity_hash,
        "process_instance_id": process_instance_id,
        "admitted_at_unix_ms": admitted_at_unix_ms.to_string(),
    }))
}

fn signal_payload_hash(signal: &ActiveProtectedSignal) -> Result<String, LedgerError> {
    hash_json(json!({
        "shape": "active_root_signal_v1",
        "active_root_signal_id": signal.active_root_signal_id,
        "signal_identity_hash": signal.signal_identity_hash,
        "event_kind": signal.event_kind,
        "scope_phase": signal.scope_phase,
        "category": signal.category,
        "name": signal.name,
        "disposition": signal.disposition.as_str(),
        "observed_at_unix_ms": signal.observed_at_unix_ms.to_string(),
        "canonical_signal_json": serde_json::from_str::<Json>(&signal.canonical_signal_json)
            .map_err(|_| invariant())?,
    }))
}

fn hash_json(value: Json) -> Result<String, LedgerError> {
    canonical_sha256(&value).map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn hex_bytes(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn execute_one(
    connection: &Connection,
    sql: &str,
    parameters: impl rusqlite::Params,
) -> Result<(), LedgerError> {
    if connection
        .execute(sql, parameters)
        .map_err(database_error)?
        != 1
    {
        return Err(corrupt());
    }
    Ok(())
}

fn root_window_is_open(
    connection: &Connection,
    active_root_window_id: Uuid,
) -> Result<bool, LedgerError> {
    let rows = connection
        .prepare(
            "SELECT active_root_window_state_event_id, state, label_complete,
                    signal_count, signal_size_bytes, process_instance_id,
                    created_at_unix_ms, canonical_payload_hash
             FROM active_root_window_state_events
             WHERE active_root_window_id = ?1 ORDER BY event_seq",
        )
        .map_err(database_error)?
        .query_map([active_root_window_id.to_string()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, i64>(6)?,
                row.get::<_, String>(7)?,
            ))
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    for row in &rows {
        let label_complete = match row.2 {
            0 => false,
            1 => true,
            _ => return Err(corrupt()),
        };
        let signal_count = u64::try_from(row.3).map_err(|_| corrupt())?;
        let signal_size_bytes = u64::try_from(row.4).map_err(|_| corrupt())?;
        let process_instance_id = parse_uuid_v7(&row.5)?;
        if root_state_payload_hash(
            parse_uuid_v7(&row.0)?,
            active_root_window_id,
            &row.1,
            label_complete,
            signal_count,
            signal_size_bytes,
            process_instance_id,
            row.6,
        )? != row.7
        {
            return Err(corrupt());
        }
    }
    match rows.as_slice() {
        [open] if open.1 == "open" && open.2 == 0 && open.3 == 0 && open.4 == 0 => Ok(true),
        [open, terminal]
            if open.1 == "open"
                && open.2 == 0
                && open.3 == 0
                && open.4 == 0
                && terminal.1 != "open" =>
        {
            Ok(false)
        }
        _ => Err(corrupt()),
    }
}

fn load_signal_totals(
    connection: &Connection,
    active_root_window_id: Uuid,
) -> Result<(usize, usize), LedgerError> {
    let values = connection
        .query_row(
            "SELECT count(*), coalesce(sum(length(CAST(canonical_signal_json AS BLOB))), 0),
                    coalesce(max(signal_ordinal), -1)
             FROM active_root_signals WHERE active_root_window_id = ?1",
            [active_root_window_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .map_err(database_error)?;
    if values.0 < 0 || values.1 < 0 || values.2 != values.0 - 1 {
        return Err(corrupt());
    }
    Ok((
        usize::try_from(values.0).map_err(|_| corrupt())?,
        usize::try_from(values.1).map_err(|_| corrupt())?,
    ))
}

fn now_unix_ms() -> Result<i64, LedgerError> {
    let now = Utc::now().timestamp_millis();
    if now < 0 {
        Err(LedgerErrorClass::DatabaseOperationFailed.into())
    } else {
        Ok(now)
    }
}

fn is_uuid_v7(value: Uuid) -> bool {
    value.get_variant() == Variant::RFC4122 && value.get_version_num() == 7
}

fn parse_uuid_v7(value: &str) -> Result<Uuid, LedgerError> {
    let parsed = Uuid::parse_str(value).map_err(|_| corrupt())?;
    if parsed.to_string() != value || !is_uuid_v7(parsed) {
        return Err(corrupt());
    }
    Ok(parsed)
}

fn is_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_text(value: &str, max_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= max_bytes
        && value.chars().all(|character| !character.is_control())
}

fn valid_stable_class(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
        })
}

fn invariant() -> LedgerError {
    LedgerErrorClass::IdentityInvariant.into()
}

fn corrupt() -> LedgerError {
    LedgerErrorClass::CorruptDatabase.into()
}

fn database_error(_error: rusqlite::Error) -> LedgerError {
    LedgerErrorClass::DatabaseOperationFailed.into()
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::{Arc, Barrier};
    use std::time::{Duration, Instant};

    use tempfile::tempdir;

    use super::*;
    use crate::config::RouterConfig;
    use crate::diagnostics::validate_router_config;
    use crate::ledger::cohort::NonLearningCohort;
    use crate::ledger::command::WriterFailureClass;
    use crate::ledger::repository::LedgerRepository;
    use crate::ledger::repository::active_learning::{
        ActiveExperimentCreate, ActiveExperimentCreateAck,
    };
    use crate::ledger::writer::LedgerWriterOwner;

    pub(crate) struct Fixture {
        pub(crate) _directory: tempfile::TempDir,
        pub(crate) config: RouterConfig,
        pub(crate) activated: super::super::ActivatedLedger,
        pub(crate) admission: ActiveRootAdmission,
    }

    pub(crate) fn active_config(path: &std::path::Path, project_id: &str) -> RouterConfig {
        let base = crate::ledger::repository::tests::config(path, project_id);
        let mut value = serde_json::to_value(base).unwrap();
        value["mode"] = json!("active");
        value["embedders"] = json!([{
            "id": "active-embedder",
            "base_url": "http://127.0.0.1:8080/v1",
            "model": "embedder-model",
            "provider_revision": "embedder-r1",
            "dimensions": 16,
            "timeout_ms": 1_000
        }]);
        value["pools"][0]["learning"] = json!({
            "version": 1,
            "embedder": "active-embedder",
            "top_k": 1,
            "radius": 1.0,
            "min_points": 1,
            "min_independent_roots": 1,
            "min_effective_samples": 1.0,
            "min_coverage": 0.0,
            "time_decay_half_life_seconds": 3_600.0,
            "prior_success": 1.0,
            "prior_failure": 1.0,
            "familywise_credible_level": 0.95,
            "promotion_lower_bound": 0.9,
            "retention_lower_bound": 0.8,
            "holdout_probability": 0.2,
            "active_canary_fraction": 0.4
        });
        value["pools"][0]["outcome"] = json!({
            "version": 1,
            "success_matchers": [{
                "event_kind": "scope_end",
                "category": "agent",
                "name": "completed",
                "terminal_status": "ok",
                "metadata_equals": {}
            }],
            "failure_matchers": [{
                "event_kind": "scope_end",
                "category": "agent",
                "name": "completed",
                "terminal_status": "error",
                "metadata_equals": {}
            }],
            "completion_disposition": "success",
            "error_disposition": "failure",
            "tool_failure_disposition": "failure",
            "end_of_run_disposition": "ignore",
            "max_attribution_seconds": 600,
            "actual_outcome_half_life_seconds": 1_800,
            "anchor_shadow_half_life_seconds": 1_800,
            "relearning_cooloff_seconds": 300,
            "min_treatment_roots": 32,
            "min_control_roots": 32,
            "min_treatment_effective_weight": 16.0,
            "min_control_effective_weight": 16.0,
            "noninferiority_margin": 0.1,
            "noninferiority_probability": 0.99,
            "rollback_probability": 0.95,
            "outcome_evaluation_batch_size": 64,
            "max_canary_roots": 64,
            "authorization_ttl_seconds": 600
        });
        let report = validate_router_config(value.as_object().unwrap());
        assert!(!report.has_errors(), "{:?}", report.diagnostics);
        report.config.unwrap()
    }

    pub(crate) fn fixture() -> Fixture {
        fixture_with_max_canary_roots(64)
    }

    pub(crate) fn fixture_with_max_canary_roots(max_canary_roots: u64) -> Fixture {
        let directory = tempdir().unwrap();
        let path = crate::ledger::repository::tests::database_path(&directory);
        let mut config = active_config(&path, "active-persistence");
        config.pools[0]
            .outcome
            .insert("max_canary_roots".into(), json!(max_canary_roots));
        let mut activated = LedgerRepository::activate(&config).unwrap();
        let project_uuid = activated.identity.project_uuid;
        let process_instance_id = activated.identity.process_instance_id;
        let pool = &activated.identity.pools["pool-a"];
        let vector_space_id = pool.vector_space.as_ref().unwrap().vector_space_id.clone();
        let active_assignment_id = Uuid::now_v7();
        let active_root_window_id = Uuid::now_v7();
        let decision_id = Uuid::now_v7();
        let active_dispatch_id = Uuid::now_v7();
        let raw_root = Uuid::now_v7();
        let raw_owner = Uuid::now_v7();
        let protected = activated
            .cohort_assignment
            .assign_non_learning(raw_root, NonLearningCohort::Ineligible);
        let root_key = protected.root_key();
        let owner_relation_hash = activated
            .cohort_assignment
            .protect_pinned_owner(root_key, raw_owner)
            .to_hex();
        let outcome_policy_hash = activated
            .repository
            .connection
            .query_row(
                "SELECT outcome_policy_hash FROM outcome_policy_versions WHERE pool_id = 'pool-a'",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let create = ActiveExperimentCreate::from_config(
            &config,
            &activated.identity,
            "pool-a",
            "candidate-a",
            "11".repeat(32),
            outcome_policy_hash.clone(),
            0,
        )
        .unwrap();
        let experiment_receipt = match activated
            .repository
            .create_active_experiment(&create)
            .unwrap()
        {
            ActiveExperimentCreateAck::Applied(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        let active_experiment_id = experiment_receipt.active_experiment_id;
        let authorization_id = experiment_receipt.initial_authorization_state_event_id;
        let now = Utc::now().timestamp_millis();
        let canonical_query_hash = canonical_sha256(&json!({})).unwrap();
        activated
            .repository
            .connection
            .execute(
                "INSERT INTO canonical_routing_queries (
                    canonical_query_hash, canonical_query_json, canonical_size_bytes,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, '{}', 2, ?2, ?1)",
                params![canonical_query_hash, now],
            )
            .unwrap();
        activated
            .repository
            .connection
            .execute(
                "INSERT INTO decisions (
                    decision_id, decision_shape_version, algorithm_version,
                    project_uuid, process_instance_id, config_generation_id,
                    policy_version_id, learning_generation_id, cohort_generation_id,
                    active_experiment_id, active_authorization_state_event_id,
                    pool_id, candidate_id, root_key, primary_call_uuid, mode,
                    canonical_query_hash, partition_base_json, partition_base_hash,
                    vector_space_id, candidate_set_hash, candidate_count,
                    recommended_model, recommended_model_revision,
                    served_model, served_model_revision, as_of_unix_ms,
                    decision_latency_ms, final_reason, summary_count,
                    summary_aggregate_hash, neighbor_count, neighbor_aggregate_hash,
                    aggregate_size_bytes, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, 2, 2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
                    'pool-a', 'candidate-a', ?10, ?11, 'active', ?12,
                    '{}', ?12, ?13, ?14, 1, 'candidate-model-a', '2026-06-01',
                    'candidate-model-a', '2026-06-01', ?15, 1,
                    'active_candidate', 1, ?16, 0, ?17, 1, ?15, ?18
                 )",
                params![
                    decision_id.to_string(),
                    project_uuid.to_string(),
                    process_instance_id.to_string(),
                    activated.identity.config_generation_id,
                    pool.policy_version_id,
                    pool.learning_generation_id.to_string(),
                    activated.identity.cohort_generation_id.to_string(),
                    active_experiment_id.to_string(),
                    authorization_id.to_string(),
                    root_key.to_hex(),
                    Uuid::now_v7().to_string(),
                    canonical_query_hash,
                    vector_space_id.as_str(),
                    "77".repeat(32),
                    now,
                    "88".repeat(32),
                    "99".repeat(32),
                    "aa".repeat(32),
                ],
            )
            .unwrap();

        let admission = ActiveRootAdmission {
            active_root_window_id,
            active_experiment_id,
            active_assignment_id,
            decision_id,
            dispatch: Some(ActiveDispatchAdmission {
                active_dispatch_id,
                request_identity_hash: "bb".repeat(32),
            }),
            pool_id: "pool-a".into(),
            candidate_id: "candidate-a".into(),
            root_key: root_key.to_hex(),
            owner_relation_hash,
            config_generation_id: activated.identity.config_generation_id.clone(),
            learning_generation_id: pool.learning_generation_id,
            cohort_generation_id: activated.identity.cohort_generation_id,
            outcome_policy_hash,
            opened_after_ingest_seq: 7,
            attribution_deadline_unix_ms: (now + 600_000) as u64,
            tranche_ordinal: 1,
            arm: ActiveAssignmentArm::CandidateTreatment,
            cohort_threshold_numerator: 1_u64.to_be_bytes(),
            selection_threshold_numerator: 2_u64.to_be_bytes(),
            configured_holdout_probability_bits: 0.2_f64.to_bits(),
            configured_canary_probability_bits: 0.4_f64.to_bits(),
            effective_arm_probability_bits: 0.4_f64.to_bits(),
            conditional_selection_probability_bits: 1.0_f64.to_bits(),
            propensity_bits: 0.4_f64.to_bits(),
            control_generation: 0,
        };
        Fixture {
            _directory: directory,
            config,
            activated,
            admission,
        }
    }

    fn decision_reason(arm: ActiveAssignmentArm) -> &'static str {
        match arm {
            ActiveAssignmentArm::CandidateTreatment => "active_candidate",
            ActiveAssignmentArm::AnchorControl => "active_anchor_control",
            ActiveAssignmentArm::AnchorHoldout => "active_anchor_holdout",
            ActiveAssignmentArm::NonLearning => "active_ineligible",
        }
    }

    fn insert_cloned_decision(fixture: &Fixture, admission: &ActiveRootAdmission) {
        let (served_model, served_revision) =
            if admission.arm == ActiveAssignmentArm::CandidateTreatment {
                ("candidate-model-a", "2026-06-01")
            } else {
                ("anchor-a", "2026-07-01")
            };
        fixture
            .activated
            .repository
            .connection
            .execute(
                "INSERT INTO decisions (
                    decision_id, decision_shape_version, algorithm_version,
                    project_uuid, process_instance_id, config_generation_id,
                    policy_version_id, learning_generation_id, cohort_generation_id,
                    active_experiment_id, active_authorization_state_event_id,
                    pool_id, candidate_id, root_key, primary_call_uuid, mode,
                    canonical_query_hash, partition_base_json, partition_base_hash,
                    vector_space_id, candidate_set_hash, candidate_count,
                    recommended_model, recommended_model_revision,
                    served_model, served_model_revision, as_of_unix_ms,
                    decision_latency_ms, final_reason, summary_count,
                    summary_aggregate_hash, neighbor_count, neighbor_aggregate_hash,
                    aggregate_size_bytes, created_at_unix_ms, canonical_payload_hash
                 )
                 SELECT ?1, decision_shape_version, algorithm_version,
                    project_uuid, process_instance_id, config_generation_id,
                    policy_version_id, learning_generation_id, cohort_generation_id,
                    active_experiment_id, active_authorization_state_event_id,
                    pool_id, candidate_id, ?2, ?3, mode,
                    canonical_query_hash, partition_base_json, partition_base_hash,
                    vector_space_id, candidate_set_hash, candidate_count,
                    recommended_model, recommended_model_revision,
                    ?4, ?5, as_of_unix_ms, decision_latency_ms, ?6,
                    summary_count, summary_aggregate_hash, neighbor_count,
                    neighbor_aggregate_hash, aggregate_size_bytes,
                    created_at_unix_ms, ?7
                 FROM decisions WHERE decision_id = ?8",
                params![
                    admission.decision_id.to_string(),
                    admission.root_key,
                    Uuid::now_v7().to_string(),
                    served_model,
                    served_revision,
                    decision_reason(admission.arm),
                    hash_json(json!({"decision": admission.decision_id})).unwrap(),
                    fixture.admission.decision_id.to_string(),
                ],
            )
            .unwrap();
    }

    pub(crate) fn new_admission(
        fixture: &Fixture,
        arm: ActiveAssignmentArm,
    ) -> ActiveRootAdmission {
        let raw_root = Uuid::now_v7();
        let protected = fixture
            .activated
            .cohort_assignment
            .assign_non_learning(raw_root, NonLearningCohort::Ineligible);
        let root_key = protected.root_key();
        let effective_probability = match arm {
            ActiveAssignmentArm::AnchorHoldout => 0.2_f64,
            ActiveAssignmentArm::NonLearning => 1.0_f64,
            ActiveAssignmentArm::CandidateTreatment | ActiveAssignmentArm::AnchorControl => 0.4_f64,
        };
        let mut admission = fixture.admission.clone();
        admission.active_root_window_id = Uuid::now_v7();
        admission.active_assignment_id = Uuid::now_v7();
        admission.decision_id = Uuid::now_v7();
        admission.root_key = root_key.to_hex();
        admission.owner_relation_hash = fixture
            .activated
            .cohort_assignment
            .protect_pinned_owner(root_key, Uuid::now_v7())
            .to_hex();
        admission.arm = arm;
        admission.effective_arm_probability_bits = effective_probability.to_bits();
        admission.conditional_selection_probability_bits = 1.0_f64.to_bits();
        admission.propensity_bits = effective_probability.to_bits();
        admission.dispatch =
            (arm == ActiveAssignmentArm::CandidateTreatment).then(|| ActiveDispatchAdmission {
                active_dispatch_id: Uuid::now_v7(),
                request_identity_hash: hash_json(json!({"request": admission.decision_id}))
                    .unwrap(),
            });
        insert_cloned_decision(fixture, &admission);
        admission
    }

    pub(crate) fn protected_signal(
        seed: u64,
        disposition: ActiveSignalDisposition,
        observed_at_unix_ms: u64,
        canonical_signal_json: String,
    ) -> ActiveProtectedSignal {
        ActiveProtectedSignal {
            active_root_signal_id: Uuid::now_v7(),
            signal_identity_hash: hash_json(json!({"signal": seed})).unwrap(),
            event_kind: "scope_end".into(),
            scope_phase: "end".into(),
            category: "agent".into(),
            name: "completed".into(),
            disposition,
            observed_at_unix_ms,
            canonical_signal_json,
        }
    }

    pub(crate) fn terminal_for(
        admission: &ActiveRootAdmission,
        receipt: ActiveAdmissionReceipt,
        closure: ActiveRootClosure,
        label_complete: bool,
        representative_status: Option<ActiveRepresentativeStatus>,
    ) -> ActiveRootTerminal {
        ActiveRootTerminal {
            active_root_window_state_event_id: Uuid::now_v7(),
            outcome_id: Uuid::now_v7(),
            active_root_window_id: admission.active_root_window_id,
            closure,
            label_complete,
            representative_status,
            stable_terminal_error_class: None,
            interval_end_unix_ms: receipt.admitted_at_unix_ms + 10,
            neighborhood_invalidation: None,
        }
    }

    #[test]
    fn admission_signal_dispatch_and_outcome_replay_exactly() {
        let mut fixture = fixture();
        let receipt = match fixture
            .activated
            .repository
            .admit_active_root(&fixture.admission)
            .unwrap()
        {
            ActiveAdmissionAck::Applied(receipt) => receipt,
            acknowledgement => panic!("unexpected admission acknowledgement: {acknowledgement:?}"),
        };
        assert_eq!(receipt.total_ordinal, 1);
        assert_eq!(receipt.cap_ordinal, Some(1));
        assert_eq!(
            fixture
                .activated
                .repository
                .admit_active_root(&fixture.admission)
                .unwrap(),
            ActiveAdmissionAck::AlreadyApplied(receipt)
        );

        let signal = ActiveProtectedSignal {
            active_root_signal_id: Uuid::now_v7(),
            signal_identity_hash: "cc".repeat(32),
            event_kind: "representative_result".into(),
            scope_phase: "end".into(),
            category: "llm".into(),
            name: "primary".into(),
            disposition: ActiveSignalDisposition::Success,
            observed_at_unix_ms: receipt.admitted_at_unix_ms + 1,
            canonical_signal_json: "{\"status\":\"completed\"}".into(),
        };
        let batch = ActiveSignalBatch {
            active_root_window_id: fixture.admission.active_root_window_id,
            signals: vec![signal],
        };
        assert_eq!(
            fixture
                .activated
                .repository
                .append_active_signals(&batch)
                .unwrap(),
            ActiveSignalBatchAck::Applied {
                signal_count: 1,
                signal_size_bytes: 22,
            }
        );
        assert!(matches!(
            fixture
                .activated
                .repository
                .append_active_signals(&batch)
                .unwrap(),
            ActiveSignalBatchAck::AlreadyApplied {
                signal_count: 1,
                ..
            }
        ));

        let dispatch = fixture.admission.dispatch.as_ref().unwrap();
        let dispatch_terminal = ActiveDispatchTerminal {
            active_dispatch_terminal_event_id: Uuid::now_v7(),
            active_dispatch_id: dispatch.active_dispatch_id,
            active_assignment_id: fixture.admission.active_assignment_id,
            terminal_state: ActiveDispatchTerminalState::Completed,
            stable_error_class: None,
            provider_receipt_hash: Some("dd".repeat(32)),
            handed_off_at_unix_ms: Some(receipt.admitted_at_unix_ms),
        };
        assert_eq!(
            fixture
                .activated
                .repository
                .record_active_dispatch_terminal(&dispatch_terminal)
                .unwrap(),
            ActiveDispatchTerminalAck::Applied
        );
        assert_eq!(
            fixture
                .activated
                .repository
                .record_active_dispatch_terminal(&dispatch_terminal)
                .unwrap(),
            ActiveDispatchTerminalAck::AlreadyApplied
        );

        let root_terminal = ActiveRootTerminal {
            active_root_window_state_event_id: Uuid::now_v7(),
            outcome_id: Uuid::now_v7(),
            active_root_window_id: fixture.admission.active_root_window_id,
            closure: ActiveRootClosure::OwnerEnd,
            label_complete: true,
            representative_status: Some(ActiveRepresentativeStatus::Completed),
            stable_terminal_error_class: None,
            interval_end_unix_ms: receipt.admitted_at_unix_ms + 10,
            neighborhood_invalidation: None,
        };
        assert_eq!(
            fixture
                .activated
                .repository
                .terminalize_active_root(&root_terminal)
                .unwrap(),
            ActiveRootTerminalAck::Applied
        );
        assert_eq!(
            fixture
                .activated
                .repository
                .terminalize_active_root(&root_terminal)
                .unwrap(),
            ActiveRootTerminalAck::AlreadyApplied
        );
        let outcome = fixture
            .activated
            .repository
            .connection
            .query_row(
                "SELECT arm, label, attribution_status FROM outcomes",
                [],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(
            outcome,
            (
                "candidate_treatment".into(),
                Some("success".into()),
                "eligible_treatment".into(),
            )
        );
        assert_eq!(
            fixture
                .activated
                .repository
                .append_active_signals(&ActiveSignalBatch {
                    active_root_window_id: fixture.admission.active_root_window_id,
                    signals: vec![ActiveProtectedSignal {
                        active_root_signal_id: Uuid::now_v7(),
                        signal_identity_hash: "ee".repeat(32),
                        event_kind: "late".into(),
                        scope_phase: "end".into(),
                        category: "agent".into(),
                        name: "late".into(),
                        disposition: ActiveSignalDisposition::Failure,
                        observed_at_unix_ms: receipt.admitted_at_unix_ms + 11,
                        canonical_signal_json: "{}".into(),
                    }],
                })
                .unwrap(),
            ActiveSignalBatchAck::WindowClosed
        );
    }

    #[test]
    fn same_root_different_command_is_refused_without_consuming_an_ordinal() {
        let mut fixture = fixture();
        assert!(matches!(
            fixture
                .activated
                .repository
                .admit_active_root(&fixture.admission)
                .unwrap(),
            ActiveAdmissionAck::Applied(_)
        ));
        let mut conflicting = fixture.admission.clone();
        conflicting.active_root_window_id = Uuid::now_v7();
        conflicting.active_assignment_id = Uuid::now_v7();
        conflicting.decision_id = Uuid::now_v7();
        conflicting.dispatch.as_mut().unwrap().active_dispatch_id = Uuid::now_v7();
        assert_eq!(
            fixture
                .activated
                .repository
                .admit_active_root(&conflicting)
                .unwrap(),
            ActiveAdmissionAck::RootAlreadyAssigned
        );
        let assignment_count = fixture
            .activated
            .repository
            .connection
            .query_row("SELECT count(*) FROM active_assignments", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap();
        assert_eq!(assignment_count, 1);
    }

    #[test]
    fn signal_count_and_byte_limits_reject_whole_batches() {
        let mut count_fixture = fixture();
        let receipt = match count_fixture
            .activated
            .repository
            .admit_active_root(&count_fixture.admission)
            .unwrap()
        {
            ActiveAdmissionAck::Applied(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        let signals = (0..ACTIVE_ROOT_SIGNALS_MAX)
            .map(|seed| {
                protected_signal(
                    seed as u64,
                    ActiveSignalDisposition::Ignored,
                    receipt.admitted_at_unix_ms,
                    "{}".into(),
                )
            })
            .collect();
        assert_eq!(
            count_fixture
                .activated
                .repository
                .append_active_signals(&ActiveSignalBatch {
                    active_root_window_id: count_fixture.admission.active_root_window_id,
                    signals,
                })
                .unwrap(),
            ActiveSignalBatchAck::Applied {
                signal_count: 64,
                signal_size_bytes: 128,
            }
        );
        assert_eq!(
            count_fixture
                .activated
                .repository
                .append_active_signals(&ActiveSignalBatch {
                    active_root_window_id: count_fixture.admission.active_root_window_id,
                    signals: vec![protected_signal(
                        64,
                        ActiveSignalDisposition::Ignored,
                        receipt.admitted_at_unix_ms,
                        "{}".into(),
                    )],
                })
                .unwrap(),
            ActiveSignalBatchAck::CapacityExceeded
        );

        let mut byte_fixture = fixture();
        let receipt = match byte_fixture
            .activated
            .repository
            .admit_active_root(&byte_fixture.admission)
            .unwrap()
        {
            ActiveAdmissionAck::Applied(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        let exact_limit = format!("\"{}\"", "x".repeat(ACTIVE_ROOT_SIGNAL_BYTES_MAX - 2));
        assert_eq!(exact_limit.len(), ACTIVE_ROOT_SIGNAL_BYTES_MAX);
        assert_eq!(
            byte_fixture
                .activated
                .repository
                .append_active_signals(&ActiveSignalBatch {
                    active_root_window_id: byte_fixture.admission.active_root_window_id,
                    signals: vec![protected_signal(
                        0,
                        ActiveSignalDisposition::Ignored,
                        receipt.admitted_at_unix_ms,
                        exact_limit,
                    )],
                })
                .unwrap(),
            ActiveSignalBatchAck::Applied {
                signal_count: 1,
                signal_size_bytes: ACTIVE_ROOT_SIGNAL_BYTES_MAX as u64,
            }
        );
        assert_eq!(
            byte_fixture
                .activated
                .repository
                .append_active_signals(&ActiveSignalBatch {
                    active_root_window_id: byte_fixture.admission.active_root_window_id,
                    signals: vec![protected_signal(
                        1,
                        ActiveSignalDisposition::Ignored,
                        receipt.admitted_at_unix_ms,
                        "{}".into(),
                    )],
                })
                .unwrap(),
            ActiveSignalBatchAck::CapacityExceeded
        );
    }

    #[test]
    fn terminalization_rejects_corrupt_admission_signal_and_terminal_hashes() {
        for corrupt_table in [
            "active_assignments",
            "active_root_signals",
            "active_dispatch_terminal_events",
        ] {
            let mut fixture = fixture();
            let receipt = match fixture
                .activated
                .repository
                .admit_active_root(&fixture.admission)
                .unwrap()
            {
                ActiveAdmissionAck::Applied(receipt) => receipt,
                acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
            };
            fixture
                .activated
                .repository
                .append_active_signals(&ActiveSignalBatch {
                    active_root_window_id: fixture.admission.active_root_window_id,
                    signals: vec![protected_signal(
                        0,
                        ActiveSignalDisposition::Success,
                        receipt.admitted_at_unix_ms,
                        "{}".into(),
                    )],
                })
                .unwrap();
            let dispatch = fixture.admission.dispatch.as_ref().unwrap();
            fixture
                .activated
                .repository
                .record_active_dispatch_terminal(&ActiveDispatchTerminal {
                    active_dispatch_terminal_event_id: Uuid::now_v7(),
                    active_dispatch_id: dispatch.active_dispatch_id,
                    active_assignment_id: fixture.admission.active_assignment_id,
                    terminal_state: ActiveDispatchTerminalState::Completed,
                    stable_error_class: None,
                    provider_receipt_hash: None,
                    handed_off_at_unix_ms: Some(receipt.admitted_at_unix_ms),
                })
                .unwrap();
            fixture
                .activated
                .repository
                .connection
                .execute(
                    &format!("UPDATE {corrupt_table} SET canonical_payload_hash = ?1"),
                    ["f".repeat(64)],
                )
                .unwrap();
            let error = fixture
                .activated
                .repository
                .terminalize_active_root(&terminal_for(
                    &fixture.admission,
                    receipt,
                    ActiveRootClosure::OwnerEnd,
                    true,
                    Some(ActiveRepresentativeStatus::Completed),
                ))
                .unwrap_err();
            assert_eq!(error.class(), LedgerErrorClass::CorruptDatabase);
            let outcomes = fixture
                .activated
                .repository
                .connection
                .query_row("SELECT count(*) FROM outcomes", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap();
            assert_eq!(outcomes, 0);
        }
    }

    #[test]
    fn control_and_holdout_outcomes_preserve_distinct_eligibility() {
        for (arm, expected_status, expected_cap) in [
            (
                ActiveAssignmentArm::AnchorControl,
                "eligible_control",
                Some(1_i64),
            ),
            (ActiveAssignmentArm::AnchorHoldout, "monitoring_only", None),
        ] {
            let mut fixture = fixture();
            let admission = new_admission(&fixture, arm);
            let receipt = match fixture
                .activated
                .repository
                .admit_active_root(&admission)
                .unwrap()
            {
                ActiveAdmissionAck::Applied(receipt) => receipt,
                acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
            };
            fixture
                .activated
                .repository
                .append_active_signals(&ActiveSignalBatch {
                    active_root_window_id: admission.active_root_window_id,
                    signals: vec![protected_signal(
                        0,
                        ActiveSignalDisposition::Success,
                        receipt.admitted_at_unix_ms,
                        "{}".into(),
                    )],
                })
                .unwrap();
            fixture
                .activated
                .repository
                .terminalize_active_root(&terminal_for(
                    &admission,
                    receipt,
                    ActiveRootClosure::OwnerEnd,
                    true,
                    Some(ActiveRepresentativeStatus::Completed),
                ))
                .unwrap();
            let outcome = fixture
                .activated
                .repository
                .connection
                .query_row(
                    "SELECT attribution_status, label, cap_ordinal
                     FROM outcomes WHERE active_root_window_id = ?1",
                    [admission.active_root_window_id.to_string()],
                    |row| {
                        Ok((
                            row.get::<_, String>(0)?,
                            row.get::<_, Option<String>>(1)?,
                            row.get::<_, Option<i64>>(2)?,
                        ))
                    },
                )
                .unwrap();
            assert_eq!(
                outcome,
                (expected_status.into(), Some("success".into()), expected_cap)
            );
        }
    }

    #[test]
    fn closure_precedence_and_late_dispatch_are_nonlearning() {
        let mut fixture = fixture();
        let receipt = match fixture
            .activated
            .repository
            .admit_active_root(&fixture.admission)
            .unwrap()
        {
            ActiveAdmissionAck::Applied(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        let terminal = terminal_for(
            &fixture.admission,
            receipt,
            ActiveRootClosure::Orphaned,
            false,
            None,
        );
        fixture
            .activated
            .repository
            .terminalize_active_root(&terminal)
            .unwrap();
        let outcome = fixture
            .activated
            .repository
            .connection
            .query_row(
                "SELECT attribution_status, label FROM outcomes",
                [],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
            )
            .unwrap();
        assert_eq!(outcome, ("orphaned".into(), None));
        let dispatch = fixture.admission.dispatch.as_ref().unwrap();
        assert_eq!(
            fixture
                .activated
                .repository
                .record_active_dispatch_terminal(&ActiveDispatchTerminal {
                    active_dispatch_terminal_event_id: Uuid::now_v7(),
                    active_dispatch_id: dispatch.active_dispatch_id,
                    active_assignment_id: fixture.admission.active_assignment_id,
                    terminal_state: ActiveDispatchTerminalState::Completed,
                    stable_error_class: None,
                    provider_receipt_hash: None,
                    handed_off_at_unix_ms: Some(receipt.admitted_at_unix_ms),
                })
                .unwrap(),
            ActiveDispatchTerminalAck::WindowClosed
        );
    }

    #[test]
    fn concurrent_same_root_admission_applies_once_and_replays_once() {
        let Fixture {
            _directory,
            config,
            activated,
            admission,
        } = fixture();
        let second = LedgerRepository::activate(&config).unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let first_barrier = barrier.clone();
        let first_admission = admission.clone();
        let mut first_repository = activated.repository;
        let first = std::thread::spawn(move || {
            first_barrier.wait();
            first_repository
                .admit_active_root(&first_admission)
                .unwrap()
        });
        let second_barrier = barrier.clone();
        let second_admission = admission;
        let mut second_repository = second.repository;
        let second = std::thread::spawn(move || {
            second_barrier.wait();
            second_repository
                .admit_active_root(&second_admission)
                .unwrap()
        });
        let acknowledgements = [first.join().unwrap(), second.join().unwrap()];
        assert_eq!(
            acknowledgements
                .iter()
                .filter(|ack| matches!(ack, ActiveAdmissionAck::Applied(_)))
                .count(),
            1
        );
        assert_eq!(
            acknowledgements
                .iter()
                .filter(|ack| matches!(ack, ActiveAdmissionAck::AlreadyApplied(_)))
                .count(),
            1
        );
        let receipts = acknowledgements.map(|ack| match ack {
            ActiveAdmissionAck::Applied(receipt) | ActiveAdmissionAck::AlreadyApplied(receipt) => {
                receipt
            }
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        });
        assert_eq!(receipts[0], receipts[1]);
    }

    #[test]
    fn concurrent_cap_admission_never_exceeds_the_experiment_limit() {
        let mut fixture = fixture();
        for expected_cap_ordinal in 1..64 {
            let admission = new_admission(&fixture, ActiveAssignmentArm::CandidateTreatment);
            let acknowledgement = fixture
                .activated
                .repository
                .admit_active_root(&admission)
                .unwrap();
            assert!(matches!(
                acknowledgement,
                ActiveAdmissionAck::Applied(ActiveAdmissionReceipt {
                    cap_ordinal: Some(value),
                    ..
                }) if value == expected_cap_ordinal
            ));
        }
        let first_admission = new_admission(&fixture, ActiveAssignmentArm::CandidateTreatment);
        let second_admission = new_admission(&fixture, ActiveAssignmentArm::CandidateTreatment);
        let second = LedgerRepository::activate(&fixture.config).unwrap();
        let barrier = Arc::new(Barrier::new(2));
        let first_barrier = barrier.clone();
        let mut first_repository = fixture.activated.repository;
        let first = std::thread::spawn(move || {
            first_barrier.wait();
            first_repository
                .admit_active_root(&first_admission)
                .unwrap()
        });
        let second_barrier = barrier.clone();
        let mut second_repository = second.repository;
        let second = std::thread::spawn(move || {
            second_barrier.wait();
            second_repository
                .admit_active_root(&second_admission)
                .unwrap()
        });
        let acknowledgements = [first.join().unwrap(), second.join().unwrap()];
        assert_eq!(
            acknowledgements
                .iter()
                .filter(|ack| matches!(
                    ack,
                    ActiveAdmissionAck::Applied(ActiveAdmissionReceipt {
                        cap_ordinal: Some(64),
                        ..
                    })
                ))
                .count(),
            1
        );
        assert_eq!(
            acknowledgements
                .iter()
                .filter(|ack| matches!(ack, ActiveAdmissionAck::ExperimentCapReached))
                .count(),
            1
        );
    }

    #[test]
    fn process_open_window_limit_is_enforced_before_assignment() {
        let mut fixture = fixture();
        let process_instance_id = fixture.activated.identity.process_instance_id;
        let project_uuid = fixture.activated.identity.project_uuid;
        let created_at_unix_ms = Utc::now().timestamp_millis();
        let transaction = fixture
            .activated
            .repository
            .connection
            .transaction()
            .unwrap();
        for index in 0..ACTIVE_OPEN_ROOT_WINDOWS_MAX {
            let mut root = fixture.admission.clone();
            root.active_root_window_id = Uuid::now_v7();
            root.root_key = hash_json(json!({"capacity_root": index})).unwrap();
            root.owner_relation_hash = hash_json(json!({"capacity_owner": index})).unwrap();
            let root_hash =
                root_payload_hash(project_uuid, process_instance_id, &root, created_at_unix_ms)
                    .unwrap();
            transaction
                .execute(
                    "INSERT INTO active_root_windows (
                        active_root_window_id, active_experiment_id, project_uuid,
                        pool_id, candidate_id, root_key, owner_relation_hash,
                        config_generation_id, learning_generation_id,
                        cohort_generation_id, outcome_policy_hash,
                        opened_after_ingest_seq, attribution_deadline_unix_ms,
                        process_instance_id, created_at_unix_ms,
                        canonical_payload_hash
                     ) VALUES (
                        ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8,
                        ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16
                     )",
                    params![
                        root.active_root_window_id.to_string(),
                        root.active_experiment_id.to_string(),
                        project_uuid.to_string(),
                        root.pool_id,
                        root.candidate_id,
                        root.root_key,
                        root.owner_relation_hash,
                        root.config_generation_id,
                        root.learning_generation_id.to_string(),
                        root.cohort_generation_id.to_string(),
                        root.outcome_policy_hash,
                        i64::try_from(root.opened_after_ingest_seq).unwrap(),
                        i64::try_from(root.attribution_deadline_unix_ms).unwrap(),
                        process_instance_id.to_string(),
                        created_at_unix_ms,
                        root_hash,
                    ],
                )
                .unwrap();
        }
        transaction.commit().unwrap();
        assert_eq!(
            fixture
                .activated
                .repository
                .admit_active_root(&fixture.admission)
                .unwrap(),
            ActiveAdmissionAck::OpenWindowCapacity
        );
    }

    #[test]
    fn startup_reconciles_each_root_dispatch_crash_point_without_eligibility() {
        for (arm, record_terminal, expected_dispatch_state) in [
            (
                ActiveAssignmentArm::CandidateTreatment,
                false,
                Some("unknown_after_crash"),
            ),
            (
                ActiveAssignmentArm::CandidateTreatment,
                true,
                Some("completed"),
            ),
            (ActiveAssignmentArm::AnchorControl, false, None),
        ] {
            let mut fixture = fixture();
            let admission = if arm == ActiveAssignmentArm::CandidateTreatment {
                fixture.admission.clone()
            } else {
                new_admission(&fixture, arm)
            };
            let receipt = match fixture
                .activated
                .repository
                .admit_active_root(&admission)
                .unwrap()
            {
                ActiveAdmissionAck::Applied(receipt) => receipt,
                acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
            };
            if record_terminal {
                let dispatch = admission.dispatch.as_ref().unwrap();
                fixture
                    .activated
                    .repository
                    .record_active_dispatch_terminal(&ActiveDispatchTerminal {
                        active_dispatch_terminal_event_id: Uuid::now_v7(),
                        active_dispatch_id: dispatch.active_dispatch_id,
                        active_assignment_id: admission.active_assignment_id,
                        terminal_state: ActiveDispatchTerminalState::Completed,
                        stable_error_class: None,
                        provider_receipt_hash: None,
                        handed_off_at_unix_ms: Some(receipt.admitted_at_unix_ms),
                    })
                    .unwrap();
            }
            let Fixture {
                _directory,
                config,
                activated,
                ..
            } = fixture;
            drop(activated);
            let restart_at = i64::try_from(admission.attribution_deadline_unix_ms).unwrap() + 1;
            let restarted = LedgerRepository::activate_at(&config, restart_at).unwrap();
            let dispatch_state = restarted
                .repository
                .connection
                .query_row(
                    "SELECT terminal.terminal_state
                     FROM active_dispatches AS dispatch
                     LEFT JOIN active_dispatch_terminal_events AS terminal
                       ON terminal.active_dispatch_id = dispatch.active_dispatch_id
                     WHERE dispatch.active_assignment_id = ?1",
                    [admission.active_assignment_id.to_string()],
                    |row| row.get::<_, Option<String>>(0),
                )
                .optional()
                .unwrap()
                .flatten();
            assert_eq!(dispatch_state.as_deref(), expected_dispatch_state);
            let outcome = restarted
                .repository
                .connection
                .query_row(
                    "SELECT attribution_status, label
                     FROM outcomes WHERE active_root_window_id = ?1",
                    [admission.active_root_window_id.to_string()],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
                )
                .unwrap();
            assert_eq!(outcome, ("orphaned".into(), None));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn writer_timeout_loses_only_the_acknowledgement_and_retry_replays() {
        let Fixture {
            _directory,
            activated,
            admission,
            ..
        } = fixture();
        let (mut owner, client) = LedgerWriterOwner::start(activated.repository, 4).unwrap();
        let pause_deadline = Instant::now() + Duration::from_secs(3);
        let (started_tx, started_rx) = std::sync::mpsc::sync_channel(1);
        let (release_tx, release_rx) = std::sync::mpsc::sync_channel(1);
        let pause_client = client.clone();
        let pause = tokio::spawn(async move {
            pause_client
                .pause_until(pause_deadline, started_tx, release_rx)
                .await
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let error = client
            .admit_active_root_until(
                admission.clone(),
                Instant::now() + Duration::from_millis(20),
            )
            .await
            .unwrap_err();
        assert_eq!(error.class(), WriterFailureClass::Deadline);
        release_tx.send(()).unwrap();
        pause.await.unwrap().unwrap();
        client
            .flush_until(Instant::now() + Duration::from_secs(1))
            .await
            .unwrap();
        assert!(matches!(
            client
                .admit_active_root_until(admission, Instant::now() + Duration::from_secs(1))
                .await
                .unwrap(),
            ActiveAdmissionAck::AlreadyApplied(_)
        ));
        owner
            .drain_until(Instant::now() + Duration::from_secs(2))
            .await
            .unwrap();
    }
}
