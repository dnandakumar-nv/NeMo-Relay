// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic, atomic retention for fully terminal Router evidence.

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use serde_json::{Value as Json, json};
use uuid::Uuid;

use super::cooloff::{
    carry_forward_latest_dependency_states_in_transaction,
    verify_retained_dependency_states_in_transaction,
};
use super::embedding::load_verified_embedding_job;
use super::materialization::{
    MaterializationRetentionAck, MaterializationRetentionDelete,
    deindex_materialization_for_retention_with_retired_spaces_in_transaction,
    delete_materialization_for_retention_with_retired_spaces_in_transaction,
    verify_retained_vector_graph_in_transaction, verify_retiring_anchor_deindexed_in_transaction,
};
use super::process::{ProcessStatusAt, append_integrity_health, verified_process_status_at};
use super::shadow::verify_terminal_anchor_for_retention;
use super::vector_catalog::{
    load_canonical_query, load_embedding_cache, verify_historical_routing_partition,
};
use super::vector_index::{
    GenerationObjectsStatus, GenerationRetirementAck, SourceHistoryRetentionAck,
    VectorIndexManifestState, current_generation_manifest, load_validated_manifest,
    load_validated_rebuild_lease, prune_vector_source_history_for_retention,
    retire_current_generation_for_retention, verify_generation_objects,
};
use super::vector_registry::{
    FrozenMappingKey, resolve_embedder_profile, resolve_frozen_mapping, resolve_vector_space,
};
use super::{LedgerRepository, TransactionStartGuard, map_fs_error, map_sqlite_error};
use crate::canonical_json::canonical_sha256;
use crate::ledger::fs::enforce_sidecar_permissions;
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::sqlite_vec_schema::VectorIndexGeneration;
use crate::vector::VectorSpaceId;

const RETENTION_BATCH_LIMIT: usize = 1_000;
const DECISION_CHILD_RETENTION_LIMIT: usize = 4_159;
const DECISION_RETENTION_BYTES_LIMIT: usize = 32 * 1024 * 1024;
const ACTIVE_RETIREMENT_CHILD_LIMIT: usize = 4_095;
const ACTIVE_RETIREMENT_BYTES_LIMIT: usize = 32 * 1024 * 1024;
const ACTIVE_RETIREMENT_RECEIPT_MAX: usize = 8_192;
const ACTIVE_RETIREMENT_CHECKPOINT_MAX: usize = 1_024;
const ACTIVE_RETIREMENT_COMPACTION_COUNT: usize = 512;
const MILLIS_PER_DAY: i64 = 86_400_000;
const RETENTION_SUMMARY_SHAPE_VERSION: i64 = 3;

/// One immutable request for a retention transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RetentionRequest {
    pub(crate) retention_batch_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) created_at_unix_ms: i64,
}

impl RetentionRequest {
    pub(crate) fn new(
        retention_batch_id: Uuid,
        conflict_health_event_id: Uuid,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(retention_batch_id)?;
        validate_uuid_v7(conflict_health_event_id)?;
        if retention_batch_id == conflict_health_event_id || created_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            retention_batch_id,
            conflict_health_event_id,
            created_at_unix_ms,
        })
    }
}

/// One member of the immutable, globally ordered retention selection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RetentionSelection {
    pub(crate) anchor_id: Uuid,
    pub(crate) closed_at_unix_ms: i64,
    pub(crate) age_expired: bool,
    pub(crate) count_excess: bool,
}

/// Durable audit summary for one retention request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RetentionSummary {
    pub(crate) retention_batch_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) summary_shape_version: i64,
    pub(crate) project_uuid: Uuid,
    pub(crate) process_instance_id: Uuid,
    pub(crate) age_expired: bool,
    pub(crate) count_excess: bool,
    pub(crate) selected_count: u64,
    pub(crate) selection_lower_bound_unix_ms: Option<i64>,
    pub(crate) selection_upper_bound_unix_ms: Option<i64>,
    pub(crate) selection_hash: String,
    pub(crate) created_at_unix_ms: i64,
    pub(crate) canonical_payload_hash: String,
}

/// Transaction-adjacent capacity observation, intentionally outside the immutable summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RetentionCapacityObservation {
    pub(crate) terminal_anchor_count: u64,
    pub(crate) capacity_available: bool,
    pub(crate) more_cleanup: bool,
}

/// Exhaustive acknowledgement for one retention writer command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RetentionAck {
    Applied {
        summary: RetentionSummary,
        observation: RetentionCapacityObservation,
    },
    AlreadyApplied {
        summary: RetentionSummary,
        observation: RetentionCapacityObservation,
    },
    Conflict,
    VectorIndexUnavailable {
        vector_space_id: VectorSpaceId,
        expected_generation: VectorIndexGeneration,
        expected_manifest_hash: String,
    },
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RetentionFaultPoint {
    Selection,
    CarryForward,
    Deletion,
    Summary,
}

#[derive(Debug)]
struct TerminalAnchor {
    anchor_id: Uuid,
    closed_at_unix_ms: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct RetiringAnchorMarker {
    anchor_id: Uuid,
    first_retention_batch_id: Uuid,
    age_expired: bool,
    count_excess: bool,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AnchorRetentionCandidate {
    selection: RetentionSelection,
    existing_marker: bool,
}

#[derive(Debug, Clone)]
struct ActiveRetirementMarkerRow {
    marker_id: Uuid,
    experiment_id: Uuid,
}

#[derive(Debug, Clone, Copy)]
struct ActiveRetirementStep {
    parent_rows_deleted: usize,
    child_rows_deleted: usize,
    bytes_deleted: usize,
    completed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DecisionRetentionSelection {
    decision_id: Uuid,
    created_at_unix_ms: i64,
    age_expired: bool,
    count_excess: bool,
    source_forced: bool,
    summary_count: usize,
    neighbor_count: usize,
    aggregate_size_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct DecisionRetentionReceipt {
    retention_batch_id: Uuid,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    decision_age_expired: bool,
    decision_count_excess: bool,
    source_forced: bool,
    new_marker_count: u64,
    new_marker_hash: String,
    deleted_decision_count: u64,
    deleted_summary_count: u64,
    deleted_neighbor_count: u64,
    deleted_child_count: u64,
    verified_aggregate_bytes: u64,
    selection_lower_bound_unix_ms: Option<i64>,
    selection_upper_bound_unix_ms: Option<i64>,
    decision_selection_hash: String,
    blocked_anchor_count: u64,
    blocked_anchor_hash: String,
    deleted_anchor_count: u64,
    deleted_anchor_hash: String,
    more_cleanup: bool,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug, Default)]
struct CapturedVectorReferences {
    embedding_ids: BTreeSet<String>,
    embedding_job_ids: BTreeSet<String>,
    canonical_query_hashes: BTreeSet<String>,
    vector_space_ids: BTreeSet<String>,
    partition_ids: BTreeSet<i64>,
    profile_versions: BTreeSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum RetentionCleanupKind {
    Embedding,
    EmbeddingJob,
    CanonicalQuery,
    VectorSpace,
    Profile,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct RetentionCleanupKey {
    kind: RetentionCleanupKind,
    key: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AttemptTerminality {
    Nonterminal,
    InFlightChild,
    Terminal,
}

#[derive(Debug)]
struct StoredRetentionSummary {
    retention_batch_id: String,
    conflict_health_event_id: Option<String>,
    summary_shape_version: i64,
    project_uuid: String,
    process_instance_id: String,
    age_expired: i64,
    count_excess: i64,
    selected_count: i64,
    selection_lower_bound_unix_ms: Option<i64>,
    selection_upper_bound_unix_ms: Option<i64>,
    selection_hash: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug)]
struct StoredDecisionRetentionReceipt {
    retention_batch_id: String,
    project_uuid: String,
    process_instance_id: String,
    decision_age_expired: i64,
    decision_count_excess: i64,
    source_forced: i64,
    new_marker_count: i64,
    new_marker_hash: String,
    deleted_decision_count: i64,
    deleted_summary_count: i64,
    deleted_neighbor_count: i64,
    deleted_child_count: i64,
    verified_aggregate_bytes: i64,
    selection_lower_bound_unix_ms: Option<i64>,
    selection_upper_bound_unix_ms: Option<i64>,
    decision_selection_hash: String,
    blocked_anchor_count: i64,
    blocked_anchor_hash: String,
    deleted_anchor_count: i64,
    deleted_anchor_hash: String,
    more_cleanup: i64,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

impl LedgerRepository {
    /// Atomically retain only the configured fully terminal evidence set.
    pub(crate) fn run_retention(
        &mut self,
        request: &RetentionRequest,
    ) -> Result<RetentionAck, LedgerError> {
        self.run_retention_with_start_check(request, || Some(()))
    }

    /// Run retention after retaining caller-owned writer start authority.
    pub(crate) fn run_retention_with_start_check<G: TransactionStartGuard>(
        &mut self,
        request: &RetentionRequest,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<RetentionAck, LedgerError> {
        self.run_retention_with_hooks(request, start_check, |_| Ok(()))
    }

    fn run_retention_with_hooks<G, Fault>(
        &mut self,
        request: &RetentionRequest,
        start_check: impl FnOnce() -> Option<G>,
        mut fault: Fault,
    ) -> Result<RetentionAck, LedgerError>
    where
        G: TransactionStartGuard,
        Fault: FnMut(RetentionFaultPoint) -> Result<(), LedgerError>,
    {
        if self.retention_days == 0 || self.max_evidence_records == 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let database_path = self.database_path.clone();
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let retention_days = self.retention_days;
        let max_evidence_records = self.max_evidence_records;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(RetentionAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_) if !start_guard.permits_transaction() => {
                return Ok(RetentionAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(RetentionAck::TransactionNotStarted);
        }
        drop(start_guard);

        match verified_process_status_at(
            &transaction,
            project_uuid,
            process_instance_id,
            request.created_at_unix_ms,
        )? {
            ProcessStatusAt::Live => {}
            ProcessStatusAt::Expired | ProcessStatusAt::Terminal => {
                return Ok(RetentionAck::OriginatingProcessNotLive);
            }
            ProcessStatusAt::Invalid => {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        let stored_summary = load_retention_summary(&transaction, request.retention_batch_id)?;
        let stored_decision_receipt =
            load_decision_retention_receipt(&transaction, request.retention_batch_id)?;
        if let Some(stored) = stored_summary {
            let acknowledgement = if let (Some(summary), Some(decision_receipt)) = (
                verify_stored_summary(stored),
                stored_decision_receipt.and_then(verify_stored_decision_retention_receipt),
            ) && summary_matches_request(
                &summary,
                request,
                project_uuid,
                process_instance_id,
            ) && decision_receipt_matches_request(
                &decision_receipt,
                request,
                project_uuid,
                process_instance_id,
            ) && decision_receipt.deleted_anchor_count
                == summary.selected_count
                && decision_receipt.deleted_anchor_hash == summary.selection_hash
            {
                RetentionAck::AlreadyApplied {
                    summary,
                    observation: capacity_observation(
                        &transaction,
                        project_uuid,
                        max_evidence_records,
                        request.created_at_unix_ms,
                        decision_receipt.more_cleanup,
                    )?,
                }
            } else {
                append_integrity_health(
                    &transaction,
                    request.conflict_health_event_id,
                    project_uuid,
                    process_instance_id,
                    None,
                    None,
                    request.created_at_unix_ms,
                )?;
                RetentionAck::Conflict
            };
            enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
            transaction.commit().map_err(database_error)?;
            return Ok(acknowledgement);
        }
        if stored_decision_receipt.is_some() {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }

        let terminal_count = terminal_anchor_count(&transaction, project_uuid)?;
        let eligible = fully_terminal_anchors(&transaction, project_uuid)?;
        let retention_millis = i64::from(retention_days)
            .checked_mul(MILLIS_PER_DAY)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let age_cutoff = request
            .created_at_unix_ms
            .checked_sub(retention_millis)
            .filter(|cutoff| *cutoff >= 0);
        let retiring_markers = load_retiring_anchor_markers(&transaction, project_uuid)?;
        let (anchor_candidates, anchor_selection_truncated) = select_anchor_retention_candidates(
            eligible,
            terminal_count,
            max_evidence_records,
            age_cutoff,
            &retiring_markers,
        )?;
        for candidate in &anchor_candidates {
            verify_terminal_anchor_for_retention(
                &transaction,
                project_uuid,
                candidate.selection.anchor_id,
            )?;
        }
        fault(RetentionFaultPoint::Selection)?;

        let referenced_anchor_ids = anchor_ids_with_decision_references(
            &transaction,
            anchor_candidates
                .iter()
                .map(|candidate| candidate.selection.anchor_id),
        )?;
        let referenced_anchor_id_strings = referenced_anchor_ids
            .iter()
            .map(Uuid::to_string)
            .collect::<BTreeSet<_>>();
        let preserving_references =
            captured_vector_references(&transaction, &referenced_anchor_id_strings)?;
        verify_selected_vector_graph(
            &transaction,
            project_uuid,
            &referenced_anchor_id_strings,
            &preserving_references,
        )?;
        let preserving_deletes = selected_materialization_deletes(
            &transaction,
            &referenced_anchor_id_strings,
            request.created_at_unix_ms,
        )?;
        let mut retired_spaces = BTreeSet::new();
        for command in &preserving_deletes {
            let acknowledgement =
                deindex_materialization_for_retention_with_retired_spaces_in_transaction(
                    &transaction,
                    project_uuid,
                    process_instance_id,
                    command,
                    &mut retired_spaces,
                )?;
            match acknowledgement {
                MaterializationRetentionAck::Deleted
                | MaterializationRetentionAck::AlreadyAbsent => {}
                MaterializationRetentionAck::IndexUnavailable {
                    vector_space_id,
                    expected_generation,
                    expected_manifest_hash,
                } => {
                    return Ok(RetentionAck::VectorIndexUnavailable {
                        vector_space_id,
                        expected_generation,
                        expected_manifest_hash,
                    });
                }
                MaterializationRetentionAck::Stale | MaterializationRetentionAck::Conflict => {
                    return Err(LedgerErrorClass::IdentityInvariant.into());
                }
            }
        }

        let mut new_markers = Vec::new();
        for candidate in &anchor_candidates {
            if candidate.existing_marker
                || !referenced_anchor_ids.contains(&candidate.selection.anchor_id)
            {
                continue;
            }
            let marker = build_retiring_anchor_marker(request, project_uuid, &candidate.selection)?;
            insert_retiring_anchor_marker(&transaction, &marker)?;
            new_markers.push(marker);
        }

        let (decision_selection, decision_selection_truncated) = select_decisions_for_retention(
            &transaction,
            project_uuid,
            max_evidence_records,
            age_cutoff,
        )?;
        for selected in &decision_selection {
            let deleted = transaction
                .execute(
                    "DELETE FROM decisions WHERE decision_id = ?1 AND project_uuid = ?2",
                    params![selected.decision_id.to_string(), project_uuid.to_string()],
                )
                .map_err(database_error)?;
            if deleted != 1 {
                return Err(LedgerErrorClass::CorruptDatabase.into());
            }
        }

        let mut deleted_anchors = Vec::new();
        let mut blocked_anchors = Vec::new();
        for candidate in &anchor_candidates {
            if anchor_has_decision_references(&transaction, candidate.selection.anchor_id)? {
                blocked_anchors.push(candidate.selection.clone());
                continue;
            }
            deleted_anchors.push(candidate.selection.clone());
        }
        let deleted_anchor_ids = deleted_anchors
            .iter()
            .map(|selection| selection.anchor_id.to_string())
            .collect::<BTreeSet<_>>();
        let carried_dependency_keys = carry_forward_latest_dependency_states_in_transaction(
            &transaction,
            project_uuid,
            &deleted_anchor_ids,
            request.created_at_unix_ms,
        )?;
        fault(RetentionFaultPoint::CarryForward)?;

        let mut captured_vector_references =
            captured_vector_references(&transaction, &deleted_anchor_ids)?;
        let cleanup_limit = RETENTION_BATCH_LIMIT.saturating_sub(deleted_anchors.len());
        let cleanup_selection = select_global_cleanup_keys(
            &transaction,
            project_uuid,
            request.created_at_unix_ms,
            cleanup_limit,
        )?;
        merge_global_cleanup_references(
            &transaction,
            &cleanup_selection,
            &mut captured_vector_references,
        )?;
        verify_global_cleanup_selection(&transaction, project_uuid, &cleanup_selection)?;
        verify_selected_vector_graph(
            &transaction,
            project_uuid,
            &deleted_anchor_ids,
            &captured_vector_references,
        )?;
        let materialization_deletes = selected_materialization_deletes(
            &transaction,
            &deleted_anchor_ids,
            request.created_at_unix_ms,
        )?;
        for command in &materialization_deletes {
            match delete_materialization_for_retention_with_retired_spaces_in_transaction(
                &transaction,
                project_uuid,
                process_instance_id,
                command,
                &mut retired_spaces,
            )? {
                MaterializationRetentionAck::Deleted
                | MaterializationRetentionAck::AlreadyAbsent => {}
                MaterializationRetentionAck::IndexUnavailable {
                    vector_space_id,
                    expected_generation,
                    expected_manifest_hash,
                } => {
                    return Ok(RetentionAck::VectorIndexUnavailable {
                        vector_space_id,
                        expected_generation,
                        expected_manifest_hash,
                    });
                }
                MaterializationRetentionAck::Stale | MaterializationRetentionAck::Conflict => {
                    return Err(LedgerErrorClass::IdentityInvariant.into());
                }
            }
        }
        for selected in &deleted_anchors {
            let deleted = transaction
                .execute(
                    "DELETE FROM anchors WHERE anchor_id = ?1 AND project_uuid = ?2",
                    params![selected.anchor_id.to_string(), project_uuid.to_string()],
                )
                .map_err(database_error)?;
            if deleted != 1 {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
        verify_retained_dependency_states_in_transaction(
            &transaction,
            project_uuid,
            &carried_dependency_keys,
        )?;
        delete_captured_unreferenced_embeddings(
            &transaction,
            &captured_vector_references.embedding_ids,
        )?;
        delete_captured_terminal_embedding_jobs(
            &transaction,
            &captured_vector_references.embedding_job_ids,
        )?;
        delete_captured_unreferenced_queries(
            &transaction,
            &captured_vector_references.canonical_query_hashes,
        )?;
        delete_captured_historical_vector_authority(
            &transaction,
            project_uuid,
            request.created_at_unix_ms,
            &captured_vector_references,
        )?;
        fault(RetentionFaultPoint::Deletion)?;

        let decision_more_cleanup = decision_selection_truncated
            || has_pending_decision_retention(
                &transaction,
                project_uuid,
                max_evidence_records,
                age_cutoff,
            )?;
        let global_more_cleanup =
            !select_global_cleanup_keys(&transaction, project_uuid, request.created_at_unix_ms, 1)?
                .is_empty();
        let ordinary_more_cleanup = anchor_selection_truncated
            || !blocked_anchors.is_empty()
            || decision_more_cleanup
            || global_more_cleanup;
        let summary = build_summary(request, project_uuid, process_instance_id, &deleted_anchors)?;
        insert_retention_summary(&transaction, &summary)?;
        let active_more_cleanup = run_active_retention_batch(
            &transaction,
            request,
            project_uuid,
            process_instance_id,
            retention_days,
            max_evidence_records,
        )?;
        let more_cleanup = ordinary_more_cleanup || active_more_cleanup;
        let decision_receipt = build_decision_retention_receipt(
            request,
            project_uuid,
            process_instance_id,
            &new_markers,
            &decision_selection,
            &blocked_anchors,
            &deleted_anchors,
            more_cleanup,
        )?;
        insert_decision_retention_receipt(&transaction, &decision_receipt)?;
        fault(RetentionFaultPoint::Summary)?;
        verify_no_foreign_key_violations(&transaction)?;
        let observation = capacity_observation(
            &transaction,
            project_uuid,
            max_evidence_records,
            request.created_at_unix_ms,
            decision_receipt.more_cleanup,
        )?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(RetentionAck::Applied {
            summary,
            observation,
        })
    }

    #[cfg(test)]
    fn run_retention_with_fault(
        &mut self,
        request: &RetentionRequest,
        fault_point: RetentionFaultPoint,
    ) -> Result<RetentionAck, LedgerError> {
        self.run_retention_with_hooks(
            request,
            || Some(()),
            |point| {
                if point == fault_point {
                    Err(LedgerErrorClass::DatabaseOperationFailed.into())
                } else {
                    Ok(())
                }
            },
        )
    }
}

fn invariant() -> LedgerError {
    LedgerErrorClass::IdentityInvariant.into()
}

fn corrupt() -> LedgerError {
    LedgerErrorClass::CorruptDatabase.into()
}

fn execute_active_one<P: rusqlite::Params>(
    connection: &Connection,
    sql: &str,
    params: P,
) -> Result<(), LedgerError> {
    if connection.execute(sql, params).map_err(database_error)? != 1 {
        return Err(corrupt());
    }
    Ok(())
}

fn run_active_retention_batch(
    transaction: &Transaction<'_>,
    request: &RetentionRequest,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    retention_days: u32,
    max_evidence_records: u64,
) -> Result<bool, LedgerError> {
    transaction
        .execute_batch("PRAGMA defer_foreign_keys = ON")
        .map_err(database_error)?;
    let marker = match load_oldest_active_retirement_marker(transaction, project_uuid)? {
        Some(marker) => marker,
        None => {
            let Some((experiment_id, age_expired, count_pressure)) =
                select_active_experiment_for_retirement(
                    transaction,
                    project_uuid,
                    request.created_at_unix_ms,
                    retention_days,
                    max_evidence_records,
                )?
            else {
                return Ok(false);
            };
            insert_active_retirement_marker(
                transaction,
                request,
                project_uuid,
                process_instance_id,
                experiment_id,
                age_expired,
                count_pressure,
            )?
        }
    };
    let step = delete_active_retirement_step(transaction, marker.experiment_id)?;
    let next_phase = active_retirement_phase(transaction, marker.experiment_id)?;
    insert_active_retirement_receipt(
        transaction,
        request,
        process_instance_id,
        &marker,
        step,
        next_phase,
    )?;
    compact_active_retirement_history(transaction, project_uuid, request.created_at_unix_ms)?;
    let another_marker = load_oldest_active_retirement_marker(transaction, project_uuid)?.is_some();
    let another_candidate = select_active_experiment_for_retirement(
        transaction,
        project_uuid,
        request.created_at_unix_ms,
        retention_days,
        max_evidence_records,
    )?
    .is_some();
    Ok(!step.completed || another_marker || another_candidate)
}

fn load_oldest_active_retirement_marker(
    connection: &Connection,
    project_uuid: Uuid,
) -> Result<Option<ActiveRetirementMarkerRow>, LedgerError> {
    connection
        .query_row(
            "SELECT active_retirement_marker_id, active_experiment_id
             FROM active_retirement_markers
             WHERE project_uuid = ?1
             ORDER BY marked_at_unix_ms, active_experiment_id
             LIMIT 1",
            [project_uuid.to_string()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(database_error)?
        .map(|(marker_id, experiment_id)| {
            Ok(ActiveRetirementMarkerRow {
                marker_id: parse_uuid_v7(&marker_id)?,
                experiment_id: parse_uuid_v7(&experiment_id)?,
            })
        })
        .transpose()
}

fn select_active_experiment_for_retirement(
    connection: &Connection,
    project_uuid: Uuid,
    observed_at_unix_ms: i64,
    retention_days: u32,
    max_evidence_records: u64,
) -> Result<Option<(Uuid, bool, bool)>, LedgerError> {
    let retention_millis = i64::from(retention_days)
        .checked_mul(MILLIS_PER_DAY)
        .ok_or_else(invariant)?;
    let age_cutoff = observed_at_unix_ms
        .checked_sub(retention_millis)
        .filter(|value| *value >= 0);
    let count_pressure =
        terminal_anchor_count(connection, project_uuid)? > max_evidence_records.saturating_sub(1);
    let rows = connection
        .prepare(
            "SELECT experiment.active_experiment_id, terminal.created_at_unix_ms
             FROM active_experiments AS experiment
             JOIN active_experiment_state_events AS terminal
               ON terminal.active_experiment_id = experiment.active_experiment_id
              AND terminal.state = 'terminal'
             WHERE experiment.project_uuid = ?1
               AND NOT EXISTS (
                   SELECT 1 FROM active_retirement_markers AS marker
                   WHERE marker.active_experiment_id = experiment.active_experiment_id
               )
               AND NOT EXISTS (
                   SELECT 1 FROM active_root_windows AS window
                   WHERE window.active_experiment_id = experiment.active_experiment_id
                     AND NOT EXISTS (
                         SELECT 1 FROM active_root_window_state_events AS state
                         WHERE state.active_root_window_id = window.active_root_window_id
                           AND state.state <> 'open'
                     )
               )
               AND NOT EXISTS (
                   SELECT 1 FROM active_dispatches AS dispatch
                   WHERE dispatch.active_experiment_id = experiment.active_experiment_id
                     AND NOT EXISTS (
                         SELECT 1 FROM active_dispatch_terminal_events AS terminal_dispatch
                         WHERE terminal_dispatch.active_dispatch_id = dispatch.active_dispatch_id
                     )
               )
               AND NOT EXISTS (
                   SELECT 1 FROM active_authorization_state_events AS authorization
                   WHERE authorization.active_experiment_id = experiment.active_experiment_id
                     AND authorization.state = 'passed'
                     AND authorization.valid_until_unix_ms > ?2
               )
               AND NOT EXISTS (
                   SELECT 1 FROM active_experiment_tranches AS tranche
                   WHERE tranche.active_experiment_id = experiment.active_experiment_id
                     AND coalesce((
                         SELECT state.state FROM active_tranche_state_events AS state
                         WHERE state.active_experiment_id = tranche.active_experiment_id
                           AND state.tranche_ordinal = tranche.tranche_ordinal
                         ORDER BY state.event_seq DESC LIMIT 1
                     ), '') <> 'complete'
               )
             ORDER BY terminal.created_at_unix_ms, experiment.active_experiment_id",
        )
        .map_err(database_error)?
        .query_map(
            params![project_uuid.to_string(), observed_at_unix_ms],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    for (experiment_id, terminal_at) in rows {
        if terminal_at < 0 {
            return Err(corrupt());
        }
        let age_expired = age_cutoff.is_some_and(|cutoff| terminal_at <= cutoff);
        if age_expired || count_pressure {
            return Ok(Some((
                parse_uuid_v7(&experiment_id)?,
                age_expired,
                count_pressure,
            )));
        }
    }
    Ok(None)
}

#[allow(clippy::too_many_arguments)]
fn insert_active_retirement_marker(
    connection: &Connection,
    request: &RetentionRequest,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    experiment_id: Uuid,
    age_expired: bool,
    count_pressure: bool,
) -> Result<ActiveRetirementMarkerRow, LedgerError> {
    if !age_expired && !count_pressure {
        return Err(invariant());
    }
    let marker = ActiveRetirementMarkerRow {
        marker_id: Uuid::now_v7(),
        experiment_id,
    };
    let hash = hash_json(&json!({
        "shape": "active_retirement_marker_v1",
        "active_retirement_marker_id": marker.marker_id,
        "active_experiment_id": experiment_id,
        "project_uuid": project_uuid,
        "first_retention_batch_id": request.retention_batch_id,
        "age_expired": age_expired,
        "count_pressure": count_pressure,
        "marked_by_process_instance_id": process_instance_id,
        "marked_at_unix_ms": request.created_at_unix_ms,
    }))?;
    execute_active_one(
        connection,
        "INSERT INTO active_retirement_markers (
            active_retirement_marker_id, active_experiment_id, project_uuid,
            first_retention_batch_id, age_expired, count_pressure,
            marked_by_process_instance_id, marked_at_unix_ms, canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            marker.marker_id.to_string(),
            experiment_id.to_string(),
            project_uuid.to_string(),
            request.retention_batch_id.to_string(),
            i64::from(age_expired),
            i64::from(count_pressure),
            process_instance_id.to_string(),
            request.created_at_unix_ms,
            hash,
        ],
    )?;
    Ok(marker)
}

fn delete_active_retirement_step(
    transaction: &Transaction<'_>,
    experiment_id: Uuid,
) -> Result<ActiveRetirementStep, LedgerError> {
    let experiment = experiment_id.to_string();
    for (table, selection) in [
        (
            "active_decision_facts",
            "SELECT rowid, length(CAST(fresh_gate_audit_json AS BLOB)) + 64
             FROM active_decision_facts WHERE active_experiment_id = ?1
             ORDER BY rowid",
        ),
        (
            "decision_neighbors",
            "SELECT neighbor.rowid, 64
             FROM decision_neighbors AS neighbor
             JOIN decisions AS decision ON decision.decision_id = neighbor.decision_id
             WHERE decision.active_experiment_id = ?1 ORDER BY neighbor.rowid",
        ),
        (
            "decision_candidate_summaries",
            "SELECT summary.rowid, 64
             FROM decision_candidate_summaries AS summary
             JOIN decisions AS decision ON decision.decision_id = summary.decision_id
             WHERE decision.active_experiment_id = ?1 ORDER BY summary.rowid",
        ),
        (
            "active_neighborhood_state_events",
            "SELECT rowid,
                    64 + length(CAST(stable_reason AS BLOB))
             FROM active_neighborhood_state_events
             WHERE active_experiment_id = ?1 ORDER BY event_seq DESC",
        ),
        (
            "active_look_members",
            "SELECT rowid, 64 FROM active_look_members
             WHERE active_experiment_id = ?1 ORDER BY rowid",
        ),
        (
            "active_outcome_look_audits",
            "SELECT audit.rowid,
                    length(CAST(audit.canonical_audit_json AS BLOB)) + 64
             FROM active_outcome_look_audits AS audit
             JOIN active_outcome_looks AS look
               ON look.active_outcome_look_id = audit.active_outcome_look_id
             WHERE look.active_experiment_id = ?1 ORDER BY audit.rowid",
        ),
        (
            "active_look_claim_state_events",
            "SELECT state.rowid, 64
             FROM active_look_claim_state_events AS state
             WHERE state.active_experiment_id = ?1 ORDER BY state.event_seq DESC",
        ),
        (
            "active_tranche_state_events",
            "SELECT state.rowid, 64
             FROM active_tranche_state_events AS state
             WHERE state.active_experiment_id = ?1 ORDER BY state.event_seq DESC",
        ),
    ] {
        if table_has_active_rows(transaction, selection, &experiment)? {
            let (deleted, bytes) =
                delete_bounded_active_rows(transaction, table, selection, &experiment)?;
            return Ok(ActiveRetirementStep {
                parent_rows_deleted: 0,
                child_rows_deleted: deleted,
                bytes_deleted: bytes,
                completed: false,
            });
        }
    }
    if let Some(step) = delete_active_root_batch(transaction, experiment_id)? {
        return Ok(step);
    }
    delete_active_experiment_core(transaction, experiment_id)
}

fn table_has_active_rows(
    connection: &Connection,
    selection: &str,
    experiment_id: &str,
) -> Result<bool, LedgerError> {
    let query = format!("SELECT EXISTS(SELECT 1 FROM ({selection}) LIMIT 1)");
    connection
        .query_row(&query, [experiment_id], |row| row.get(0))
        .map_err(database_error)
}

fn delete_bounded_active_rows(
    connection: &Connection,
    table: &str,
    selection: &str,
    experiment_id: &str,
) -> Result<(usize, usize), LedgerError> {
    let mut statement = connection.prepare(selection).map_err(database_error)?;
    let mut rows = statement.query([experiment_id]).map_err(database_error)?;
    let mut selected = Vec::new();
    let mut bytes = 0_usize;
    while let Some(row) = rows.next().map_err(database_error)? {
        if selected.len() == ACTIVE_RETIREMENT_CHILD_LIMIT {
            break;
        }
        let rowid = row.get::<_, i64>(0).map_err(database_error)?;
        let row_bytes = row.get::<_, i64>(1).map_err(database_error)?;
        let row_bytes = usize::try_from(row_bytes).map_err(|_| corrupt())?;
        let next_bytes = bytes.checked_add(row_bytes).ok_or_else(corrupt)?;
        if next_bytes > ACTIVE_RETIREMENT_BYTES_LIMIT {
            if selected.is_empty() {
                return Err(corrupt());
            }
            break;
        }
        selected.push(rowid);
        bytes = next_bytes;
    }
    drop(rows);
    drop(statement);
    if selected.is_empty() {
        return Err(corrupt());
    }
    let delete = format!("DELETE FROM {table} WHERE rowid = ?1");
    for rowid in &selected {
        if connection
            .execute(&delete, [rowid])
            .map_err(database_error)?
            != 1
        {
            return Err(corrupt());
        }
    }
    Ok((selected.len(), bytes))
}

fn delete_active_root_batch(
    connection: &Connection,
    experiment_id: Uuid,
) -> Result<Option<ActiveRetirementStep>, LedgerError> {
    let roots = connection
        .prepare(
            "SELECT active_root_window_id FROM active_root_windows
             WHERE active_experiment_id = ?1
             ORDER BY created_at_unix_ms, active_root_window_id
             LIMIT 1000",
        )
        .map_err(database_error)?
        .query_map([experiment_id.to_string()], |row| row.get::<_, String>(0))
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    if roots.is_empty() {
        return Ok(None);
    }
    let mut selected = Vec::new();
    let mut parent_rows = 0_usize;
    let mut child_rows = 0_usize;
    let mut bytes = 0_usize;
    for root in roots {
        let cost = active_root_delete_cost(connection, &root)?;
        let next_parents = parent_rows
            .checked_add(cost.parent_rows_deleted)
            .ok_or_else(corrupt)?;
        let next_children = child_rows
            .checked_add(cost.child_rows_deleted)
            .ok_or_else(corrupt)?;
        let next_bytes = bytes.checked_add(cost.bytes_deleted).ok_or_else(corrupt)?;
        if next_parents > RETENTION_BATCH_LIMIT
            || next_children > ACTIVE_RETIREMENT_CHILD_LIMIT
            || next_bytes > ACTIVE_RETIREMENT_BYTES_LIMIT
        {
            if selected.is_empty() {
                return Err(corrupt());
            }
            break;
        }
        selected.push(root);
        parent_rows = next_parents;
        child_rows = next_children;
        bytes = next_bytes;
    }
    for root in selected {
        delete_active_root_rows(connection, &root)?;
    }
    Ok(Some(ActiveRetirementStep {
        parent_rows_deleted: parent_rows,
        child_rows_deleted: child_rows,
        bytes_deleted: bytes,
        completed: false,
    }))
}

fn active_root_delete_cost(
    connection: &Connection,
    root: &str,
) -> Result<ActiveRetirementStep, LedgerError> {
    let signal_bytes = connection
        .query_row(
            "SELECT coalesce(sum(length(CAST(canonical_signal_json AS BLOB)) + 64), 0)
             FROM active_root_signals WHERE active_root_window_id = ?1",
            [root],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    let mut child_rows = 0_usize;
    let mut parent_rows = 1_usize;
    let mut bytes = usize::try_from(signal_bytes)
        .map_err(|_| corrupt())?
        .checked_add(64)
        .ok_or_else(corrupt)?;
    for (table, parent) in [
        ("outcomes", true),
        ("active_root_signals", false),
        ("active_root_window_state_events", false),
        ("active_root_decision_links", false),
        ("active_assignments", false),
    ] {
        let count = connection
            .query_row(
                &format!("SELECT count(*) FROM {table} WHERE active_root_window_id = ?1"),
                [root],
                |row| row.get::<_, i64>(0),
            )
            .map_err(database_error)?;
        let count = usize::try_from(count).map_err(|_| corrupt())?;
        if parent {
            parent_rows = parent_rows.checked_add(count).ok_or_else(corrupt)?;
        } else {
            child_rows = child_rows.checked_add(count).ok_or_else(corrupt)?;
        }
        bytes = bytes
            .checked_add(count.checked_mul(64).ok_or_else(corrupt)?)
            .ok_or_else(corrupt)?;
    }
    for sql in [
        "SELECT count(*) FROM active_dispatches AS dispatch
         JOIN active_assignments AS assignment
           ON assignment.active_assignment_id = dispatch.active_assignment_id
         WHERE assignment.active_root_window_id = ?1",
        "SELECT count(*) FROM active_dispatch_terminal_events AS terminal
         JOIN active_dispatches AS dispatch
           ON dispatch.active_dispatch_id = terminal.active_dispatch_id
         JOIN active_assignments AS assignment
           ON assignment.active_assignment_id = dispatch.active_assignment_id
         WHERE assignment.active_root_window_id = ?1",
        "SELECT count(*) FROM decisions AS decision
         JOIN active_root_decision_links AS link ON link.decision_id = decision.decision_id
         WHERE link.active_root_window_id = ?1",
    ] {
        let count = connection
            .query_row(sql, [root], |row| row.get::<_, i64>(0))
            .map_err(database_error)?;
        let count = usize::try_from(count).map_err(|_| corrupt())?;
        child_rows = child_rows.checked_add(count).ok_or_else(corrupt)?;
        bytes = bytes
            .checked_add(count.checked_mul(64).ok_or_else(corrupt)?)
            .ok_or_else(corrupt)?;
    }
    Ok(ActiveRetirementStep {
        parent_rows_deleted: parent_rows,
        child_rows_deleted: child_rows,
        bytes_deleted: bytes,
        completed: false,
    })
}

fn delete_active_root_rows(connection: &Connection, root: &str) -> Result<(), LedgerError> {
    let decisions = connection
        .prepare(
            "SELECT decision_id FROM active_root_decision_links
             WHERE active_root_window_id = ?1",
        )
        .map_err(database_error)?
        .query_map([root], |row| row.get::<_, String>(0))
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    for sql in [
        "DELETE FROM outcomes WHERE active_root_window_id = ?1",
        "DELETE FROM active_dispatch_terminal_events WHERE active_dispatch_id IN (
             SELECT dispatch.active_dispatch_id FROM active_dispatches AS dispatch
             JOIN active_assignments AS assignment
               ON assignment.active_assignment_id = dispatch.active_assignment_id
             WHERE assignment.active_root_window_id = ?1
         )",
        "DELETE FROM active_dispatches WHERE active_assignment_id IN (
             SELECT active_assignment_id FROM active_assignments
             WHERE active_root_window_id = ?1
         )",
        "DELETE FROM active_assignments WHERE active_root_window_id = ?1",
        "DELETE FROM active_root_decision_links WHERE active_root_window_id = ?1",
        "DELETE FROM active_root_signals WHERE active_root_window_id = ?1",
        "DELETE FROM active_root_window_state_events WHERE active_root_window_id = ?1",
    ] {
        connection.execute(sql, [root]).map_err(database_error)?;
    }
    for decision_id in decisions {
        execute_active_one(
            connection,
            "DELETE FROM decisions WHERE decision_id = ?1",
            [decision_id],
        )?;
    }
    execute_active_one(
        connection,
        "DELETE FROM active_root_windows WHERE active_root_window_id = ?1",
        [root],
    )
}

fn delete_active_experiment_core(
    connection: &Connection,
    experiment_id: Uuid,
) -> Result<ActiveRetirementStep, LedgerError> {
    let experiment = experiment_id.to_string();
    let tables = [
        "active_authorization_state_events",
        "active_outcome_look_audits",
        "active_look_failures",
        "active_outcome_looks",
        "active_look_claim_state_events",
        "active_look_claims",
        "active_tranche_state_events",
        "active_experiment_tranches",
        "active_experiment_state_events",
    ];
    let mut child_rows = 0_usize;
    let mut bytes = 0_usize;
    for table in tables {
        let predicate = if table == "active_outcome_look_audits" {
            "active_outcome_look_id IN (
                SELECT active_outcome_look_id FROM active_outcome_looks
                WHERE active_experiment_id = ?1
             )"
        } else {
            "active_experiment_id = ?1"
        };
        let count = connection
            .query_row(
                &format!("SELECT count(*) FROM {table} WHERE {predicate}"),
                [&experiment],
                |row| row.get::<_, i64>(0),
            )
            .map_err(database_error)?;
        let count = usize::try_from(count).map_err(|_| corrupt())?;
        child_rows = child_rows.checked_add(count).ok_or_else(corrupt)?;
        bytes = bytes
            .checked_add(count.checked_mul(64).ok_or_else(corrupt)?)
            .ok_or_else(corrupt)?;
    }
    if child_rows > ACTIVE_RETIREMENT_CHILD_LIMIT || bytes > ACTIVE_RETIREMENT_BYTES_LIMIT {
        return Err(corrupt());
    }
    for sql in [
        "DELETE FROM active_authorization_state_events WHERE active_experiment_id = ?1",
        "DELETE FROM active_outcome_look_audits WHERE active_outcome_look_id IN (
             SELECT active_outcome_look_id FROM active_outcome_looks
             WHERE active_experiment_id = ?1
         )",
        "DELETE FROM active_look_failures WHERE active_experiment_id = ?1",
        "DELETE FROM active_outcome_looks WHERE active_experiment_id = ?1",
        "DELETE FROM active_look_claim_state_events WHERE active_experiment_id = ?1",
        "DELETE FROM active_look_claims WHERE active_experiment_id = ?1",
        "DELETE FROM active_tranche_state_events WHERE active_experiment_id = ?1",
        "DELETE FROM active_experiment_tranches WHERE active_experiment_id = ?1",
        "DELETE FROM active_experiment_state_events WHERE active_experiment_id = ?1",
    ] {
        connection
            .execute(sql, [&experiment])
            .map_err(database_error)?;
    }
    execute_active_one(
        connection,
        "DELETE FROM active_experiments WHERE active_experiment_id = ?1",
        [&experiment],
    )?;
    Ok(ActiveRetirementStep {
        parent_rows_deleted: 1,
        child_rows_deleted: child_rows,
        bytes_deleted: bytes.checked_add(64).ok_or_else(corrupt)?,
        completed: true,
    })
}

fn active_retirement_phase(
    connection: &Connection,
    experiment_id: Uuid,
) -> Result<&'static str, LedgerError> {
    let experiment = experiment_id.to_string();
    for (phase, sql) in [
        (
            "decision_facts",
            "SELECT EXISTS(SELECT 1 FROM active_decision_facts
             WHERE active_experiment_id = ?1)",
        ),
        (
            "decision_neighbors",
            "SELECT EXISTS(
                 SELECT 1 FROM decision_neighbors AS neighbor
                 JOIN decisions AS decision ON decision.decision_id = neighbor.decision_id
                 WHERE decision.active_experiment_id = ?1
             )",
        ),
        (
            "decision_summaries",
            "SELECT EXISTS(
                 SELECT 1 FROM decision_candidate_summaries AS summary
                 JOIN decisions AS decision ON decision.decision_id = summary.decision_id
                 WHERE decision.active_experiment_id = ?1
             )",
        ),
        (
            "neighborhoods",
            "SELECT EXISTS(SELECT 1 FROM active_neighborhood_state_events
             WHERE active_experiment_id = ?1)",
        ),
        (
            "look_members",
            "SELECT EXISTS(SELECT 1 FROM active_look_members
             WHERE active_experiment_id = ?1)",
        ),
        (
            "roots",
            "SELECT EXISTS(SELECT 1 FROM active_root_windows
             WHERE active_experiment_id = ?1)",
        ),
        (
            "look_audits",
            "SELECT EXISTS(
                 SELECT 1 FROM active_outcome_look_audits AS audit
                 JOIN active_outcome_looks AS look
                   ON look.active_outcome_look_id = audit.active_outcome_look_id
                 WHERE look.active_experiment_id = ?1
             )",
        ),
        (
            "claim_states",
            "SELECT EXISTS(SELECT 1 FROM active_look_claim_state_events
             WHERE active_experiment_id = ?1)",
        ),
        (
            "tranche_states",
            "SELECT EXISTS(SELECT 1 FROM active_tranche_state_events
             WHERE active_experiment_id = ?1)",
        ),
        (
            "experiment",
            "SELECT EXISTS(SELECT 1 FROM active_experiments
             WHERE active_experiment_id = ?1)",
        ),
    ] {
        let exists = connection
            .query_row(sql, [&experiment], |row| row.get::<_, bool>(0))
            .map_err(database_error)?;
        if exists {
            return Ok(phase);
        }
    }
    Ok("completed")
}

fn insert_active_retirement_receipt(
    connection: &Connection,
    request: &RetentionRequest,
    process_instance_id: Uuid,
    marker: &ActiveRetirementMarkerRow,
    step: ActiveRetirementStep,
    next_phase: &str,
) -> Result<(), LedgerError> {
    if step.parent_rows_deleted > RETENTION_BATCH_LIMIT
        || step.child_rows_deleted > ACTIVE_RETIREMENT_CHILD_LIMIT
        || step.bytes_deleted > ACTIVE_RETIREMENT_BYTES_LIMIT
    {
        return Err(invariant());
    }
    let predecessor = connection
        .query_row(
            "SELECT chain_tip_hash FROM active_retirement_receipts
             ORDER BY receipt_ordinal DESC LIMIT 1",
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .unwrap_or_else(|| "00".repeat(32));
    let receipt_id = Uuid::now_v7();
    let cursor = json!({
        "shape": "active_retirement_cursor_v1",
        "active_experiment_id": marker.experiment_id,
        "next_phase": next_phase,
    });
    let cursor_json = crate::canonical_json::canonical_json(&cursor)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let cursor_hash = hash_json(&cursor)?;
    let chain_tip = hash_json(&json!({
        "shape": "active_retirement_receipt_v1",
        "active_retirement_receipt_id": receipt_id,
        "active_retirement_marker_id": marker.marker_id,
        "active_experiment_id": marker.experiment_id,
        "retention_batch_id": request.retention_batch_id,
        "predecessor_chain_hash": predecessor,
        "cursor_hash": cursor_hash,
        "parent_rows_deleted": step.parent_rows_deleted,
        "child_rows_deleted": step.child_rows_deleted,
        "bytes_deleted": step.bytes_deleted,
        "completed": step.completed,
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": request.created_at_unix_ms,
    }))?;
    execute_active_one(
        connection,
        "INSERT INTO active_retirement_receipts (
            active_retirement_receipt_id, active_retirement_marker_id,
            active_experiment_id, retention_batch_id, predecessor_chain_hash,
            chain_tip_hash, cursor_json, cursor_hash, parent_rows_deleted,
            child_rows_deleted, bytes_deleted, completed, process_instance_id,
            created_at_unix_ms, canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?6)",
        params![
            receipt_id.to_string(),
            marker.marker_id.to_string(),
            marker.experiment_id.to_string(),
            request.retention_batch_id.to_string(),
            predecessor,
            chain_tip,
            cursor_json,
            cursor_hash,
            i64::try_from(step.parent_rows_deleted).map_err(|_| invariant())?,
            i64::try_from(step.child_rows_deleted).map_err(|_| invariant())?,
            i64::try_from(step.bytes_deleted).map_err(|_| invariant())?,
            i64::from(step.completed),
            process_instance_id.to_string(),
            request.created_at_unix_ms,
        ],
    )?;
    Ok(())
}

fn compact_active_retirement_history(
    connection: &Connection,
    project_uuid: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let receipt_count = connection
        .query_row(
            "SELECT count(*) FROM active_retirement_receipts",
            [],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    if usize::try_from(receipt_count).map_err(|_| corrupt())? > ACTIVE_RETIREMENT_RECEIPT_MAX {
        compact_completed_active_receipts(connection, project_uuid, created_at_unix_ms)?;
    }
    let checkpoint_count = connection
        .query_row(
            "SELECT count(*) FROM active_retirement_checkpoints WHERE project_uuid = ?1",
            [project_uuid.to_string()],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    if usize::try_from(checkpoint_count).map_err(|_| corrupt())? > ACTIVE_RETIREMENT_CHECKPOINT_MAX
    {
        compact_active_checkpoints(connection, project_uuid, created_at_unix_ms)?;
    }
    Ok(())
}

fn compact_completed_active_receipts(
    connection: &Connection,
    project_uuid: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let rows = connection
        .prepare(
            "SELECT receipt.receipt_ordinal, receipt.predecessor_chain_hash,
                    receipt.chain_tip_hash, receipt.parent_rows_deleted,
                    receipt.child_rows_deleted, receipt.bytes_deleted
             FROM active_retirement_receipts AS receipt
             WHERE EXISTS (
                 SELECT 1 FROM active_retirement_receipts AS terminal
                 WHERE terminal.active_experiment_id = receipt.active_experiment_id
                   AND terminal.completed = 1
             )
             ORDER BY receipt.receipt_ordinal
             LIMIT ?1",
        )
        .map_err(database_error)?
        .query_map(
            [i64::try_from(ACTIVE_RETIREMENT_COMPACTION_COUNT).map_err(|_| invariant())?],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            },
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    if rows.len() != ACTIVE_RETIREMENT_COMPACTION_COUNT {
        return Err(corrupt());
    }
    for pair in rows.windows(2) {
        if pair[1].1 != pair[0].2 || pair[1].0 != pair[0].0 + 1 {
            return Err(corrupt());
        }
    }
    let first_ordinal = rows.first().ok_or_else(corrupt)?.0;
    let last_ordinal = rows.last().ok_or_else(corrupt)?.0;
    let first_predecessor = rows.first().ok_or_else(corrupt)?.1.clone();
    let covered_tip = rows.last().ok_or_else(corrupt)?.2.clone();
    let parent_rows = checked_i64_sum(rows.iter().map(|row| row.3))?;
    let child_rows = checked_i64_sum(rows.iter().map(|row| row.4))?;
    let bytes = checked_i64_sum(rows.iter().map(|row| row.5))?;
    let range_hash = hash_json(&json!({
        "shape": "active_retirement_receipt_range_v1",
        "receipts": rows.iter().map(|row| json!({
            "receipt_ordinal": row.0,
            "chain_tip_hash": row.2,
        })).collect::<Vec<_>>(),
    }))?;
    insert_active_retirement_checkpoint(
        connection,
        project_uuid,
        0,
        first_ordinal,
        last_ordinal,
        first_predecessor,
        covered_tip,
        i64::try_from(rows.len()).map_err(|_| invariant())?,
        parent_rows,
        child_rows,
        bytes,
        range_hash,
        created_at_unix_ms,
    )?;
    for row in rows {
        execute_active_one(
            connection,
            "DELETE FROM active_retirement_receipts WHERE receipt_ordinal = ?1",
            [row.0],
        )?;
    }
    Ok(())
}

fn compact_active_checkpoints(
    connection: &Connection,
    project_uuid: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let rows = connection
        .prepare(
            "SELECT checkpoint_hash, level, first_receipt_ordinal,
                    last_receipt_ordinal, first_predecessor_hash,
                    covered_chain_tip_hash, receipt_count,
                    parent_rows_deleted, child_rows_deleted, bytes_deleted
             FROM active_retirement_checkpoints
             WHERE project_uuid = ?1
             ORDER BY first_receipt_ordinal, last_receipt_ordinal, level
             LIMIT ?2",
        )
        .map_err(database_error)?
        .query_map(
            params![
                project_uuid.to_string(),
                i64::try_from(ACTIVE_RETIREMENT_COMPACTION_COUNT).map_err(|_| invariant())?,
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, i64>(9)?,
                ))
            },
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    if rows.len() != ACTIVE_RETIREMENT_COMPACTION_COUNT {
        return Err(corrupt());
    }
    for pair in rows.windows(2) {
        if pair[1].4 != pair[0].5 || pair[1].2 != pair[0].3 + 1 {
            return Err(corrupt());
        }
    }
    let level = rows
        .iter()
        .map(|row| row.1)
        .max()
        .ok_or_else(corrupt)?
        .checked_add(1)
        .ok_or_else(corrupt)?;
    let first_ordinal = rows.first().ok_or_else(corrupt)?.2;
    let last_ordinal = rows.last().ok_or_else(corrupt)?.3;
    let first_predecessor = rows.first().ok_or_else(corrupt)?.4.clone();
    let covered_tip = rows.last().ok_or_else(corrupt)?.5.clone();
    let receipt_count = checked_i64_sum(rows.iter().map(|row| row.6))?;
    let parent_rows = checked_i64_sum(rows.iter().map(|row| row.7))?;
    let child_rows = checked_i64_sum(rows.iter().map(|row| row.8))?;
    let bytes = checked_i64_sum(rows.iter().map(|row| row.9))?;
    let range_hash = hash_json(&json!({
        "shape": "active_retirement_checkpoint_range_v1",
        "checkpoints": rows.iter().map(|row| &row.0).collect::<Vec<_>>(),
    }))?;
    for row in &rows {
        execute_active_one(
            connection,
            "DELETE FROM active_retirement_checkpoints WHERE checkpoint_hash = ?1",
            [&row.0],
        )?;
    }
    insert_active_retirement_checkpoint(
        connection,
        project_uuid,
        level,
        first_ordinal,
        last_ordinal,
        first_predecessor,
        covered_tip,
        receipt_count,
        parent_rows,
        child_rows,
        bytes,
        range_hash,
        created_at_unix_ms,
    )
}

#[allow(clippy::too_many_arguments)]
fn insert_active_retirement_checkpoint(
    connection: &Connection,
    project_uuid: Uuid,
    level: i64,
    first_receipt_ordinal: i64,
    last_receipt_ordinal: i64,
    first_predecessor_hash: String,
    covered_chain_tip_hash: String,
    receipt_count: i64,
    parent_rows_deleted: i64,
    child_rows_deleted: i64,
    bytes_deleted: i64,
    range_hash: String,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let checkpoint_hash = hash_json(&json!({
        "shape": "active_retirement_checkpoint_v1",
        "project_uuid": project_uuid,
        "level": level,
        "first_receipt_ordinal": first_receipt_ordinal,
        "last_receipt_ordinal": last_receipt_ordinal,
        "first_predecessor_hash": first_predecessor_hash,
        "covered_chain_tip_hash": covered_chain_tip_hash,
        "receipt_count": receipt_count,
        "parent_rows_deleted": parent_rows_deleted,
        "child_rows_deleted": child_rows_deleted,
        "bytes_deleted": bytes_deleted,
        "range_hash": range_hash,
        "created_at_unix_ms": created_at_unix_ms,
    }))?;
    execute_active_one(
        connection,
        "INSERT INTO active_retirement_checkpoints (
            checkpoint_hash, project_uuid, level, first_receipt_ordinal,
            last_receipt_ordinal, first_predecessor_hash,
            covered_chain_tip_hash, receipt_count, parent_rows_deleted,
            child_rows_deleted, bytes_deleted, range_hash,
            created_at_unix_ms, canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?1)",
        params![
            checkpoint_hash,
            project_uuid.to_string(),
            level,
            first_receipt_ordinal,
            last_receipt_ordinal,
            first_predecessor_hash,
            covered_chain_tip_hash,
            receipt_count,
            parent_rows_deleted,
            child_rows_deleted,
            bytes_deleted,
            range_hash,
            created_at_unix_ms,
        ],
    )
}

fn checked_i64_sum(mut values: impl Iterator<Item = i64>) -> Result<i64, LedgerError> {
    values.try_fold(0_i64, |total, value| {
        if value < 0 {
            return Err(corrupt());
        }
        total.checked_add(value).ok_or_else(corrupt)
    })
}

fn select_retention_anchors(
    eligible: Vec<TerminalAnchor>,
    terminal_count: u64,
    max_evidence_records: u64,
    age_cutoff_unix_ms: Option<i64>,
) -> Vec<RetentionSelection> {
    retention_anchor_candidates(
        eligible,
        terminal_count,
        max_evidence_records,
        age_cutoff_unix_ms,
    )
    .into_iter()
    .take(RETENTION_BATCH_LIMIT)
    .collect()
}

fn retention_anchor_candidates(
    mut eligible: Vec<TerminalAnchor>,
    terminal_count: u64,
    max_evidence_records: u64,
    age_cutoff_unix_ms: Option<i64>,
) -> Vec<RetentionSelection> {
    eligible.sort_by_key(|anchor| (anchor.closed_at_unix_ms, anchor.anchor_id));
    let target = max_evidence_records.saturating_sub(1);
    let count_excess = terminal_count.saturating_sub(target);
    eligible
        .into_iter()
        .enumerate()
        .filter_map(|(index, anchor)| {
            let count_reason = u64::try_from(index).is_ok_and(|value| value < count_excess);
            let age_reason =
                age_cutoff_unix_ms.is_some_and(|cutoff| anchor.closed_at_unix_ms <= cutoff);
            (count_reason || age_reason).then_some(RetentionSelection {
                anchor_id: anchor.anchor_id,
                closed_at_unix_ms: anchor.closed_at_unix_ms,
                age_expired: age_reason,
                count_excess: count_reason,
            })
        })
        .collect()
}

fn select_anchor_retention_candidates(
    eligible: Vec<TerminalAnchor>,
    terminal_count: u64,
    max_evidence_records: u64,
    age_cutoff_unix_ms: Option<i64>,
    retiring_markers: &[RetiringAnchorMarker],
) -> Result<(Vec<AnchorRetentionCandidate>, bool), LedgerError> {
    let terminal_times = eligible
        .iter()
        .map(|anchor| (anchor.anchor_id, anchor.closed_at_unix_ms))
        .collect::<BTreeMap<_, _>>();
    let existing_ids = retiring_markers
        .iter()
        .map(|marker| marker.anchor_id)
        .collect::<BTreeSet<_>>();
    let ordinary = retention_anchor_candidates(
        eligible,
        terminal_count,
        max_evidence_records,
        age_cutoff_unix_ms,
    );
    let mut candidates = Vec::new();
    for marker in retiring_markers {
        let closed_at_unix_ms = terminal_times
            .get(&marker.anchor_id)
            .copied()
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        candidates.push(AnchorRetentionCandidate {
            selection: RetentionSelection {
                anchor_id: marker.anchor_id,
                closed_at_unix_ms,
                age_expired: marker.age_expired,
                count_excess: marker.count_excess,
            },
            existing_marker: true,
        });
    }
    candidates.extend(
        ordinary
            .into_iter()
            .filter(|selection| !existing_ids.contains(&selection.anchor_id))
            .map(|selection| AnchorRetentionCandidate {
                selection,
                existing_marker: false,
            }),
    );
    let truncated = candidates.len() > RETENTION_BATCH_LIMIT;
    candidates.truncate(RETENTION_BATCH_LIMIT);
    Ok((candidates, truncated))
}

fn load_retiring_anchor_markers(
    connection: &Connection,
    project_uuid: Uuid,
) -> Result<Vec<RetiringAnchorMarker>, LedgerError> {
    let rows = connection
        .prepare(
            "SELECT anchor_id, first_retention_batch_id, age_expired, count_excess,
                    created_at_unix_ms, canonical_payload_hash
             FROM decision_retiring_anchors
             WHERE project_uuid = ?1
             ORDER BY created_at_unix_ms, anchor_id",
        )
        .map_err(database_error)?
        .query_map(params![project_uuid.to_string()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, String>(5)?,
            ))
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    rows.into_iter()
        .map(|(anchor, batch, age, count, created_at, payload_hash)| {
            if !matches!(age, 0 | 1)
                || !matches!(count, 0 | 1)
                || age == 0 && count == 0
                || created_at < 0
                || !is_sha256(&payload_hash)
            {
                return Err(LedgerErrorClass::CorruptDatabase.into());
            }
            let marker = RetiringAnchorMarker {
                anchor_id: parse_uuid_v7(&anchor)?,
                first_retention_batch_id: parse_uuid_v7(&batch)?,
                age_expired: age == 1,
                count_excess: count == 1,
                created_at_unix_ms: created_at,
                canonical_payload_hash: payload_hash,
            };
            if retiring_anchor_marker_hash(project_uuid, &marker)? != marker.canonical_payload_hash
            {
                return Err(LedgerErrorClass::CorruptDatabase.into());
            }
            Ok(marker)
        })
        .collect()
}

fn build_retiring_anchor_marker(
    request: &RetentionRequest,
    project_uuid: Uuid,
    selection: &RetentionSelection,
) -> Result<RetiringAnchorMarker, LedgerError> {
    let mut marker = RetiringAnchorMarker {
        anchor_id: selection.anchor_id,
        first_retention_batch_id: request.retention_batch_id,
        age_expired: selection.age_expired,
        count_excess: selection.count_excess,
        created_at_unix_ms: request.created_at_unix_ms,
        canonical_payload_hash: String::new(),
    };
    marker.canonical_payload_hash = retiring_anchor_marker_hash(project_uuid, &marker)?;
    Ok(marker)
}

fn retiring_anchor_marker_hash(
    project_uuid: Uuid,
    marker: &RetiringAnchorMarker,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "anchor_id": marker.anchor_id,
        "project_uuid": project_uuid,
        "first_retention_batch_id": marker.first_retention_batch_id,
        "age_expired": marker.age_expired,
        "count_excess": marker.count_excess,
        "created_at_unix_ms": marker.created_at_unix_ms,
    }))
}

fn insert_retiring_anchor_marker(
    connection: &Connection,
    marker: &RetiringAnchorMarker,
) -> Result<(), LedgerError> {
    let project_uuid = connection
        .query_row(
            "SELECT project_uuid FROM anchors WHERE anchor_id = ?1",
            [marker.anchor_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?;
    let inserted = connection
        .execute(
            "INSERT INTO decision_retiring_anchors (
                anchor_id, project_uuid, first_retention_batch_id,
                age_expired, count_excess, created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                marker.anchor_id.to_string(),
                project_uuid,
                marker.first_retention_batch_id.to_string(),
                i64::from(marker.age_expired),
                i64::from(marker.count_excess),
                marker.created_at_unix_ms,
                marker.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if inserted != 1 {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(())
}

fn anchor_ids_with_decision_references(
    connection: &Connection,
    anchor_ids: impl IntoIterator<Item = Uuid>,
) -> Result<BTreeSet<Uuid>, LedgerError> {
    anchor_ids
        .into_iter()
        .try_fold(BTreeSet::new(), |mut referenced, anchor_id| {
            if anchor_has_decision_references(connection, anchor_id)? {
                referenced.insert(anchor_id);
            }
            Ok(referenced)
        })
}

fn anchor_has_decision_references(
    connection: &Connection,
    anchor_id: Uuid,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM decision_neighbors WHERE anchor_id = ?1
             )",
            [anchor_id.to_string()],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)
}

fn load_decision_retention_candidates(
    connection: &Connection,
    project_uuid: Uuid,
    max_evidence_records: u64,
    age_cutoff_unix_ms: Option<i64>,
) -> Result<Vec<DecisionRetentionSelection>, LedgerError> {
    let rows = connection
        .prepare(
            "SELECT decision.decision_id, decision.created_at_unix_ms,
                    decision.summary_count, decision.neighbor_count,
                    decision.aggregate_size_bytes,
                    EXISTS(
                        SELECT 1
                        FROM decision_neighbors AS neighbor
                        JOIN decision_retiring_anchors AS marker
                          ON marker.anchor_id = neighbor.anchor_id
                        WHERE neighbor.decision_id = decision.decision_id
                    )
             FROM decisions AS decision
             WHERE decision.project_uuid = ?1
               AND decision.mode = 'recommend'
               AND decision.decision_shape_version = 1
               AND decision.active_experiment_id IS NULL
               AND NOT EXISTS (
                   SELECT 1 FROM active_decision_facts AS active
                   WHERE active.decision_id = decision.decision_id
               )
             ORDER BY decision.created_at_unix_ms, decision.decision_id",
        )
        .map_err(database_error)?
        .query_map(params![project_uuid.to_string()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, bool>(5)?,
            ))
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    let decision_count = u64::try_from(rows.len())
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let count_excess = decision_count.saturating_sub(max_evidence_records.saturating_sub(1));
    let mut candidates = rows
        .into_iter()
        .enumerate()
        .filter_map(
            |(index, (decision_id, created_at, summaries, neighbors, bytes, source_forced))| {
                let count_reason = u64::try_from(index).is_ok_and(|index| index < count_excess);
                let age_reason = age_cutoff_unix_ms.is_some_and(|cutoff| created_at <= cutoff);
                (count_reason || age_reason || source_forced).then_some((
                    decision_id,
                    created_at,
                    summaries,
                    neighbors,
                    bytes,
                    age_reason,
                    count_reason,
                    source_forced,
                ))
            },
        )
        .map(
            |(
                decision_id,
                created_at,
                summaries,
                neighbors,
                bytes,
                age_expired,
                count_excess,
                source_forced,
            )| {
                let summary_count = usize::try_from(summaries)
                    .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
                let neighbor_count = usize::try_from(neighbors)
                    .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
                let aggregate_size_bytes = usize::try_from(bytes)
                    .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
                let child_count = summary_count
                    .checked_add(neighbor_count)
                    .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
                if created_at < 0
                    || summary_count == 0
                    || child_count > DECISION_CHILD_RETENTION_LIMIT
                    || aggregate_size_bytes == 0
                    || aggregate_size_bytes > DECISION_RETENTION_BYTES_LIMIT
                {
                    return Err(LedgerErrorClass::CorruptDatabase.into());
                }
                Ok(DecisionRetentionSelection {
                    decision_id: parse_uuid_v7(&decision_id)?,
                    created_at_unix_ms: created_at,
                    age_expired,
                    count_excess,
                    source_forced,
                    summary_count,
                    neighbor_count,
                    aggregate_size_bytes,
                })
            },
        )
        .collect::<Result<Vec<_>, LedgerError>>()?;
    prioritize_decision_retention_candidates(&mut candidates);
    Ok(candidates)
}

fn prioritize_decision_retention_candidates(candidates: &mut [DecisionRetentionSelection]) {
    candidates.sort_by_key(|candidate| {
        (
            !candidate.source_forced,
            candidate.created_at_unix_ms,
            candidate.decision_id,
        )
    });
}

fn next_decision_retention_totals(
    parent_count: usize,
    child_count: usize,
    aggregate_bytes: usize,
    candidate: &DecisionRetentionSelection,
) -> Result<Option<(usize, usize, usize)>, LedgerError> {
    let next_parents = parent_count
        .checked_add(1)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let next_children = child_count
        .checked_add(candidate.summary_count)
        .and_then(|count| count.checked_add(candidate.neighbor_count))
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let next_bytes = aggregate_bytes
        .checked_add(candidate.aggregate_size_bytes)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    Ok((next_parents <= RETENTION_BATCH_LIMIT
        && next_children <= DECISION_CHILD_RETENTION_LIMIT
        && next_bytes <= DECISION_RETENTION_BYTES_LIMIT)
        .then_some((next_parents, next_children, next_bytes)))
}

fn select_decisions_for_retention(
    connection: &Connection,
    project_uuid: Uuid,
    max_evidence_records: u64,
    age_cutoff_unix_ms: Option<i64>,
) -> Result<(Vec<DecisionRetentionSelection>, bool), LedgerError> {
    let candidates = load_decision_retention_candidates(
        connection,
        project_uuid,
        max_evidence_records,
        age_cutoff_unix_ms,
    )?;
    let mut selection = Vec::new();
    let mut child_count = 0_usize;
    let mut aggregate_bytes = 0_usize;
    let mut truncated = false;
    for candidate in candidates {
        let Some((_, next_children, next_bytes)) = next_decision_retention_totals(
            selection.len(),
            child_count,
            aggregate_bytes,
            &candidate,
        )?
        else {
            truncated = true;
            break;
        };
        let graph = super::decision::load_decision_graph(connection, candidate.decision_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        if graph.graph.parent.project_uuid != project_uuid
            || graph.graph.parent.created_at_unix_ms != candidate.created_at_unix_ms
            || graph.graph.parent.summary_count != candidate.summary_count
            || graph.graph.parent.neighbor_count != candidate.neighbor_count
            || graph.graph.parent.aggregate_size_bytes != candidate.aggregate_size_bytes
            || graph.graph.summaries.len() != candidate.summary_count
            || graph.graph.neighbors.len() != candidate.neighbor_count
        {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
        child_count = next_children;
        aggregate_bytes = next_bytes;
        selection.push(candidate);
    }
    Ok((selection, truncated))
}

fn has_pending_decision_retention(
    connection: &Connection,
    project_uuid: Uuid,
    max_evidence_records: u64,
    age_cutoff_unix_ms: Option<i64>,
) -> Result<bool, LedgerError> {
    Ok(!load_decision_retention_candidates(
        connection,
        project_uuid,
        max_evidence_records,
        age_cutoff_unix_ms,
    )?
    .is_empty())
}

fn fully_terminal_anchors(
    connection: &Connection,
    project_uuid: Uuid,
) -> Result<Vec<TerminalAnchor>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT a.anchor_id, terminal.state, terminal.created_at_unix_ms,
                    window.terminal_kind, window.closed_at_unix_ms
             FROM anchors AS a
             JOIN anchor_state_events AS terminal
               ON terminal.anchor_id = a.anchor_id AND terminal.state <> 'pending'
             LEFT JOIN anchor_windows AS window ON window.anchor_id = a.anchor_id
             WHERE a.project_uuid = ?1
             ORDER BY coalesce(window.closed_at_unix_ms, terminal.created_at_unix_ms),
                      a.anchor_id",
        )
        .map_err(database_error)?;
    let rows = statement
        .query_map(params![project_uuid.to_string()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<i64>>(4)?,
            ))
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);

    let mut eligible = Vec::new();
    for (anchor_id, terminal_state, terminal_created_at, window_kind, window_closed_at) in rows {
        let anchor_id = parse_uuid_v7(&anchor_id)?;
        let closed_at_unix_ms = window_closed_at.unwrap_or(terminal_created_at);
        if closed_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        if terminal_anchor_is_fully_terminal(
            connection,
            anchor_id,
            &terminal_state,
            terminal_created_at,
            window_kind.as_deref(),
            window_closed_at,
        )? {
            eligible.push(TerminalAnchor {
                anchor_id,
                closed_at_unix_ms,
            });
        }
    }
    Ok(eligible)
}

fn terminal_anchor_is_fully_terminal(
    connection: &Connection,
    anchor_id: Uuid,
    terminal_state: &str,
    terminal_created_at: i64,
    window_kind: Option<&str>,
    window_closed_at: Option<i64>,
) -> Result<bool, LedgerError> {
    if terminal_created_at < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let (pending_states, terminal_states, total_states) = connection
        .query_row(
            "SELECT coalesce(sum(state = 'pending'), 0),
                    coalesce(sum(state <> 'pending'), 0), count(*)
             FROM anchor_state_events WHERE anchor_id = ?1",
            params![anchor_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .map_err(database_error)?;
    if (pending_states, terminal_states, total_states) != (1, 1, 2) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let batch = connection
        .query_row(
            "SELECT sample_batch_id, reserved_candidate_count
             FROM sample_batches WHERE anchor_id = ?1",
            params![anchor_id.to_string()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    match terminal_state {
        "not_scheduled_queue_full" | "orphaned_non_resumable" => {
            if window_kind.is_some() || window_closed_at.is_some() || batch.is_some() {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
            Ok(true)
        }
        "rejected" => {
            if window_kind != Some("rejected") || window_closed_at.is_none() || batch.is_some() {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
            Ok(true)
        }
        "closed" => {
            if window_kind != Some("closed") || window_closed_at.is_none() {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
            let Some((batch_id, reserved_candidate_count)) = batch else {
                return Ok(false);
            };
            validate_batch_terminality(connection, &batch_id, anchor_id, reserved_candidate_count)
        }
        _ => Err(LedgerErrorClass::IdentityInvariant.into()),
    }
}

fn validate_batch_terminality(
    connection: &Connection,
    batch_id: &str,
    anchor_id: Uuid,
    reserved_candidate_count: i64,
) -> Result<bool, LedgerError> {
    if reserved_candidate_count <= 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let (open_states, terminal_states, state_count) = connection
        .query_row(
            "SELECT coalesce(sum(state = 'open'), 0),
                    coalesce(sum(state <> 'open'), 0), count(*)
             FROM sample_batch_state_events WHERE sample_batch_id = ?1",
            params![batch_id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .map_err(database_error)?;
    if open_states != 1 || terminal_states > 1 || state_count != open_states + terminal_states {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let mut statement = connection
        .prepare(
            "SELECT shadow_attempt_id FROM shadow_attempts
             WHERE sample_batch_id = ?1 AND anchor_id = ?2
             ORDER BY shadow_attempt_id",
        )
        .map_err(database_error)?;
    let attempt_ids = statement
        .query_map(params![batch_id, anchor_id.to_string()], |row| {
            row.get::<_, String>(0)
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    if i64::try_from(attempt_ids.len()).ok() != Some(reserved_candidate_count) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let mut all_attempts_terminal = true;
    let mut has_nonterminal_attempt = false;
    for attempt_id in &attempt_ids {
        match validate_attempt_terminality(connection, attempt_id)? {
            AttemptTerminality::Terminal => {}
            AttemptTerminality::InFlightChild => all_attempts_terminal = false,
            AttemptTerminality::Nonterminal => {
                all_attempts_terminal = false;
                has_nonterminal_attempt = true;
            }
        }
    }
    match terminal_states {
        0 if all_attempts_terminal => Err(LedgerErrorClass::IdentityInvariant.into()),
        0 => Ok(false),
        1 if all_attempts_terminal => Ok(true),
        1 if !has_nonterminal_attempt => Ok(false),
        1 => Err(LedgerErrorClass::IdentityInvariant.into()),
        _ => Err(LedgerErrorClass::IdentityInvariant.into()),
    }
}

fn validate_attempt_terminality(
    connection: &Connection,
    attempt_id: &str,
) -> Result<AttemptTerminality, LedgerError> {
    let (reserved_states, started_states, terminal_states, state_count) = connection
        .query_row(
            "SELECT coalesce(sum(state = 'reserved'), 0),
                    coalesce(sum(state = 'started'), 0),
                    coalesce(sum(state NOT IN ('reserved', 'started')), 0),
                    count(*)
             FROM shadow_attempt_state_events WHERE shadow_attempt_id = ?1",
            params![attempt_id],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )
        .map_err(database_error)?;
    if reserved_states != 1
        || started_states > 1
        || terminal_states > 1
        || state_count != reserved_states + started_states + terminal_states
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let terminal_state = connection
        .query_row(
            "SELECT state FROM shadow_attempt_state_events
             WHERE shadow_attempt_id = ?1 AND state NOT IN ('reserved', 'started')",
            params![attempt_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let result_class = connection
        .query_row(
            "SELECT terminal_class FROM shadow_results WHERE shadow_attempt_id = ?1",
            params![attempt_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    if terminal_state.is_some() != result_class.is_some()
        || terminal_state
            .as_deref()
            .zip(result_class.as_deref())
            .is_some_and(|(state, class)| state != class)
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let terminal = terminal_states == 1;
    let judges_terminal = validate_judge_terminalities(connection, attempt_id)?;
    let dependencies_terminal = validate_dependency_terminalities(connection, attempt_id)?;
    Ok(match (terminal, judges_terminal && dependencies_terminal) {
        (true, true) => AttemptTerminality::Terminal,
        (true, false) => AttemptTerminality::InFlightChild,
        (false, _) => AttemptTerminality::Nonterminal,
    })
}

fn validate_judge_terminalities(
    connection: &Connection,
    attempt_id: &str,
) -> Result<bool, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT judge_attempt_id FROM judge_attempts
             WHERE shadow_attempt_id = ?1 ORDER BY attempt_ordinal",
        )
        .map_err(database_error)?;
    let judge_ids = statement
        .query_map(params![attempt_id], |row| row.get::<_, String>(0))
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    let mut all_terminal = true;
    for judge_id in judge_ids {
        let (started, terminal, total) = connection
            .query_row(
                "SELECT coalesce(sum(state = 'started'), 0),
                        coalesce(sum(state <> 'started'), 0), count(*)
                 FROM judge_attempt_state_events WHERE judge_attempt_id = ?1",
                params![judge_id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .map_err(database_error)?;
        if started != 1 || terminal > 1 || total != started + terminal {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        all_terminal &= terminal == 1;
    }
    Ok(all_terminal)
}

fn validate_dependency_terminalities(
    connection: &Connection,
    attempt_id: &str,
) -> Result<bool, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT dependency_operation_id FROM dependency_operations
             WHERE shadow_attempt_id = ?1 ORDER BY dependency_operation_id",
        )
        .map_err(database_error)?;
    let operation_ids = statement
        .query_map(params![attempt_id], |row| row.get::<_, String>(0))
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    let mut all_terminal = true;
    for operation_id in operation_ids {
        let (admitted, skipped, terminal, total) = connection
            .query_row(
                "SELECT coalesce(sum(state = 'admitted'), 0),
                        coalesce(sum(state = 'skipped_cooloff'), 0),
                        coalesce(sum(state IN ('success', 'failure', 'orphaned_in_flight')), 0),
                        count(*)
                 FROM dependency_state_events WHERE dependency_operation_id = ?1",
                params![operation_id],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .map_err(database_error)?;
        let skipped_shape = (admitted, skipped, terminal, total) == (0, 1, 0, 1);
        let admitted_shape =
            admitted == 1 && skipped == 0 && terminal <= 1 && total == 1 + terminal;
        if !skipped_shape && !admitted_shape {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        all_terminal &= skipped_shape || terminal == 1;
    }
    Ok(all_terminal)
}

fn build_summary(
    request: &RetentionRequest,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    selection: &[RetentionSelection],
) -> Result<RetentionSummary, LedgerError> {
    let selection_hash = selection_hash(selection)?;
    let age_expired = selection.iter().any(|selected| selected.age_expired);
    let count_excess = selection.iter().any(|selected| selected.count_excess);
    let selected_count = u64::try_from(selection.len())
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let selection_lower_bound_unix_ms =
        selection.first().map(|selected| selected.closed_at_unix_ms);
    let selection_upper_bound_unix_ms = selection.last().map(|selected| selected.closed_at_unix_ms);
    let mut summary = RetentionSummary {
        retention_batch_id: request.retention_batch_id,
        conflict_health_event_id: request.conflict_health_event_id,
        summary_shape_version: RETENTION_SUMMARY_SHAPE_VERSION,
        project_uuid,
        process_instance_id,
        age_expired,
        count_excess,
        selected_count,
        selection_lower_bound_unix_ms,
        selection_upper_bound_unix_ms,
        selection_hash,
        created_at_unix_ms: request.created_at_unix_ms,
        canonical_payload_hash: String::new(),
    };
    summary.canonical_payload_hash = retention_summary_hash(&summary)?;
    Ok(summary)
}

fn selection_hash(selection: &[RetentionSelection]) -> Result<String, LedgerError> {
    let value = Json::Array(
        selection
            .iter()
            .map(|selected| {
                json!({
                    "anchor_id": selected.anchor_id,
                    "closed_at_unix_ms": selected.closed_at_unix_ms,
                    "age_expired": selected.age_expired,
                    "count_excess": selected.count_excess,
                })
            })
            .collect(),
    );
    hash_json(&value)
}

fn retention_summary_hash(summary: &RetentionSummary) -> Result<String, LedgerError> {
    hash_json(&json!({
        "retention_batch_id": summary.retention_batch_id,
        "conflict_health_event_id": summary.conflict_health_event_id,
        "summary_shape_version": summary.summary_shape_version,
        "project_uuid": summary.project_uuid,
        "process_instance_id": summary.process_instance_id,
        "age_expired": summary.age_expired,
        "count_excess": summary.count_excess,
        "selected_count": summary.selected_count,
        "selection_lower_bound_unix_ms": summary.selection_lower_bound_unix_ms,
        "selection_upper_bound_unix_ms": summary.selection_upper_bound_unix_ms,
        "selection_hash": summary.selection_hash,
        "created_at_unix_ms": summary.created_at_unix_ms,
    }))
}

fn insert_retention_summary(
    connection: &Connection,
    summary: &RetentionSummary,
) -> Result<(), LedgerError> {
    let inserted = connection
        .execute(
            "INSERT INTO retention_batches (
                retention_batch_id, conflict_health_event_id,
                project_uuid, process_instance_id,
                summary_shape_version, age_expired, count_excess, selected_count,
                selection_lower_bound_unix_ms, selection_upper_bound_unix_ms,
                selection_hash, created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)
             ON CONFLICT(retention_batch_id) DO NOTHING",
            params![
                summary.retention_batch_id.to_string(),
                summary.conflict_health_event_id.to_string(),
                summary.project_uuid.to_string(),
                summary.process_instance_id.to_string(),
                summary.summary_shape_version,
                i64::from(summary.age_expired),
                i64::from(summary.count_excess),
                i64::try_from(summary.selected_count)
                    .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
                summary.selection_lower_bound_unix_ms,
                summary.selection_upper_bound_unix_ms,
                summary.selection_hash,
                summary.created_at_unix_ms,
                summary.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if inserted != 1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn load_retention_summary(
    connection: &Connection,
    retention_batch_id: Uuid,
) -> Result<Option<StoredRetentionSummary>, LedgerError> {
    connection
        .query_row(
            "SELECT retention_batch_id, summary_shape_version, project_uuid,
                    conflict_health_event_id, process_instance_id,
                    age_expired, count_excess, selected_count,
                    selection_lower_bound_unix_ms, selection_upper_bound_unix_ms,
                    selection_hash, created_at_unix_ms, canonical_payload_hash
             FROM retention_batches WHERE retention_batch_id = ?1",
            params![retention_batch_id.to_string()],
            |row| {
                Ok(StoredRetentionSummary {
                    retention_batch_id: row.get(0)?,
                    summary_shape_version: row.get(1)?,
                    project_uuid: row.get(2)?,
                    conflict_health_event_id: row.get(3)?,
                    process_instance_id: row.get(4)?,
                    age_expired: row.get(5)?,
                    count_excess: row.get(6)?,
                    selected_count: row.get(7)?,
                    selection_lower_bound_unix_ms: row.get(8)?,
                    selection_upper_bound_unix_ms: row.get(9)?,
                    selection_hash: row.get(10)?,
                    created_at_unix_ms: row.get(11)?,
                    canonical_payload_hash: row.get(12)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn verify_stored_summary(stored: StoredRetentionSummary) -> Option<RetentionSummary> {
    if stored.summary_shape_version != RETENTION_SUMMARY_SHAPE_VERSION
        || !matches!(stored.age_expired, 0 | 1)
        || !matches!(stored.count_excess, 0 | 1)
        || !(0..=i64::try_from(RETENTION_BATCH_LIMIT).ok()?).contains(&stored.selected_count)
        || !is_sha256(&stored.selection_hash)
        || !is_sha256(&stored.canonical_payload_hash)
        || stored.created_at_unix_ms < 0
    {
        return None;
    }
    let selected_count = u64::try_from(stored.selected_count).ok()?;
    let age_expired = stored.age_expired == 1;
    let count_excess = stored.count_excess == 1;
    let valid_shape = if selected_count == 0 {
        !age_expired
            && !count_excess
            && stored.selection_lower_bound_unix_ms.is_none()
            && stored.selection_upper_bound_unix_ms.is_none()
    } else {
        (age_expired || count_excess)
            && stored.selection_lower_bound_unix_ms.is_some()
            && stored.selection_upper_bound_unix_ms.is_some()
            && stored.selection_lower_bound_unix_ms <= stored.selection_upper_bound_unix_ms
    };
    if !valid_shape {
        return None;
    }
    let retention_batch_id = parse_uuid_v7(&stored.retention_batch_id).ok()?;
    let conflict_health_event_id =
        parse_uuid_v7(stored.conflict_health_event_id.as_deref()?).ok()?;
    if retention_batch_id == conflict_health_event_id {
        return None;
    }
    let summary = RetentionSummary {
        retention_batch_id,
        conflict_health_event_id,
        summary_shape_version: stored.summary_shape_version,
        project_uuid: parse_uuid_v7(&stored.project_uuid).ok()?,
        process_instance_id: parse_uuid_v7(&stored.process_instance_id).ok()?,
        age_expired,
        count_excess,
        selected_count,
        selection_lower_bound_unix_ms: stored.selection_lower_bound_unix_ms,
        selection_upper_bound_unix_ms: stored.selection_upper_bound_unix_ms,
        selection_hash: stored.selection_hash,
        created_at_unix_ms: stored.created_at_unix_ms,
        canonical_payload_hash: stored.canonical_payload_hash,
    };
    (retention_summary_hash(&summary).ok().as_deref()
        == Some(summary.canonical_payload_hash.as_str()))
    .then_some(summary)
}

fn summary_matches_request(
    summary: &RetentionSummary,
    request: &RetentionRequest,
    project_uuid: Uuid,
    process_instance_id: Uuid,
) -> bool {
    summary.retention_batch_id == request.retention_batch_id
        && summary.conflict_health_event_id == request.conflict_health_event_id
        && summary.summary_shape_version == RETENTION_SUMMARY_SHAPE_VERSION
        && summary.project_uuid == project_uuid
        && summary.process_instance_id == process_instance_id
        && summary.created_at_unix_ms == request.created_at_unix_ms
}

#[allow(clippy::too_many_arguments)]
fn build_decision_retention_receipt(
    request: &RetentionRequest,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    new_markers: &[RetiringAnchorMarker],
    decisions: &[DecisionRetentionSelection],
    blocked_anchors: &[RetentionSelection],
    deleted_anchors: &[RetentionSelection],
    more_cleanup: bool,
) -> Result<DecisionRetentionReceipt, LedgerError> {
    let deleted_summary_count = decisions.iter().try_fold(0_usize, |count, decision| {
        count
            .checked_add(decision.summary_count)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))
    })?;
    let deleted_neighbor_count = decisions.iter().try_fold(0_usize, |count, decision| {
        count
            .checked_add(decision.neighbor_count)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))
    })?;
    let deleted_child_count = deleted_summary_count
        .checked_add(deleted_neighbor_count)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let verified_aggregate_bytes = decisions.iter().try_fold(0_usize, |bytes, decision| {
        bytes
            .checked_add(decision.aggregate_size_bytes)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))
    })?;
    let mut receipt = DecisionRetentionReceipt {
        retention_batch_id: request.retention_batch_id,
        project_uuid,
        process_instance_id,
        decision_age_expired: decisions.iter().any(|decision| decision.age_expired),
        decision_count_excess: decisions.iter().any(|decision| decision.count_excess),
        source_forced: decisions.iter().any(|decision| decision.source_forced),
        new_marker_count: to_u64(new_markers.len())?,
        new_marker_hash: new_marker_selection_hash(new_markers)?,
        deleted_decision_count: to_u64(decisions.len())?,
        deleted_summary_count: to_u64(deleted_summary_count)?,
        deleted_neighbor_count: to_u64(deleted_neighbor_count)?,
        deleted_child_count: to_u64(deleted_child_count)?,
        verified_aggregate_bytes: to_u64(verified_aggregate_bytes)?,
        selection_lower_bound_unix_ms: decisions
            .iter()
            .map(|decision| decision.created_at_unix_ms)
            .min(),
        selection_upper_bound_unix_ms: decisions
            .iter()
            .map(|decision| decision.created_at_unix_ms)
            .max(),
        decision_selection_hash: decision_selection_hash(decisions)?,
        blocked_anchor_count: to_u64(blocked_anchors.len())?,
        blocked_anchor_hash: selection_hash(blocked_anchors)?,
        deleted_anchor_count: to_u64(deleted_anchors.len())?,
        deleted_anchor_hash: selection_hash(deleted_anchors)?,
        more_cleanup,
        created_at_unix_ms: request.created_at_unix_ms,
        canonical_payload_hash: String::new(),
    };
    receipt.canonical_payload_hash = decision_retention_receipt_hash(&receipt)?;
    Ok(receipt)
}

fn new_marker_selection_hash(markers: &[RetiringAnchorMarker]) -> Result<String, LedgerError> {
    hash_json(&Json::Array(
        markers
            .iter()
            .map(|marker| {
                json!({
                    "anchor_id": marker.anchor_id,
                    "first_retention_batch_id": marker.first_retention_batch_id,
                    "age_expired": marker.age_expired,
                    "count_excess": marker.count_excess,
                    "created_at_unix_ms": marker.created_at_unix_ms,
                    "canonical_payload_hash": marker.canonical_payload_hash,
                })
            })
            .collect(),
    ))
}

fn decision_selection_hash(
    decisions: &[DecisionRetentionSelection],
) -> Result<String, LedgerError> {
    hash_json(&Json::Array(
        decisions
            .iter()
            .map(|decision| {
                json!({
                    "decision_id": decision.decision_id,
                    "created_at_unix_ms": decision.created_at_unix_ms,
                    "age_expired": decision.age_expired,
                    "count_excess": decision.count_excess,
                    "source_forced": decision.source_forced,
                    "summary_count": decision.summary_count,
                    "neighbor_count": decision.neighbor_count,
                    "aggregate_size_bytes": decision.aggregate_size_bytes,
                })
            })
            .collect(),
    ))
}

fn decision_retention_receipt_hash(
    receipt: &DecisionRetentionReceipt,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "retention_batch_id": receipt.retention_batch_id,
        "project_uuid": receipt.project_uuid,
        "process_instance_id": receipt.process_instance_id,
        "decision_age_expired": receipt.decision_age_expired,
        "decision_count_excess": receipt.decision_count_excess,
        "source_forced": receipt.source_forced,
        "new_marker_count": receipt.new_marker_count,
        "new_marker_hash": receipt.new_marker_hash,
        "deleted_decision_count": receipt.deleted_decision_count,
        "deleted_summary_count": receipt.deleted_summary_count,
        "deleted_neighbor_count": receipt.deleted_neighbor_count,
        "deleted_child_count": receipt.deleted_child_count,
        "verified_aggregate_bytes": receipt.verified_aggregate_bytes,
        "selection_lower_bound_unix_ms": receipt.selection_lower_bound_unix_ms,
        "selection_upper_bound_unix_ms": receipt.selection_upper_bound_unix_ms,
        "decision_selection_hash": receipt.decision_selection_hash,
        "blocked_anchor_count": receipt.blocked_anchor_count,
        "blocked_anchor_hash": receipt.blocked_anchor_hash,
        "deleted_anchor_count": receipt.deleted_anchor_count,
        "deleted_anchor_hash": receipt.deleted_anchor_hash,
        "more_cleanup": receipt.more_cleanup,
        "created_at_unix_ms": receipt.created_at_unix_ms,
    }))
}

fn insert_decision_retention_receipt(
    connection: &Connection,
    receipt: &DecisionRetentionReceipt,
) -> Result<(), LedgerError> {
    let inserted = connection
        .execute(
            "INSERT INTO decision_retention_receipts (
                retention_batch_id, project_uuid, process_instance_id,
                decision_age_expired, decision_count_excess, source_forced,
                new_marker_count, new_marker_hash, deleted_decision_count,
                deleted_summary_count, deleted_neighbor_count, deleted_child_count,
                verified_aggregate_bytes, selection_lower_bound_unix_ms,
                selection_upper_bound_unix_ms, decision_selection_hash,
                blocked_anchor_count, blocked_anchor_hash, deleted_anchor_count,
                deleted_anchor_hash, more_cleanup, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
                ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23
             )",
            params![
                receipt.retention_batch_id.to_string(),
                receipt.project_uuid.to_string(),
                receipt.process_instance_id.to_string(),
                i64::from(receipt.decision_age_expired),
                i64::from(receipt.decision_count_excess),
                i64::from(receipt.source_forced),
                to_i64(receipt.new_marker_count)?,
                receipt.new_marker_hash,
                to_i64(receipt.deleted_decision_count)?,
                to_i64(receipt.deleted_summary_count)?,
                to_i64(receipt.deleted_neighbor_count)?,
                to_i64(receipt.deleted_child_count)?,
                to_i64(receipt.verified_aggregate_bytes)?,
                receipt.selection_lower_bound_unix_ms,
                receipt.selection_upper_bound_unix_ms,
                receipt.decision_selection_hash,
                to_i64(receipt.blocked_anchor_count)?,
                receipt.blocked_anchor_hash,
                to_i64(receipt.deleted_anchor_count)?,
                receipt.deleted_anchor_hash,
                i64::from(receipt.more_cleanup),
                receipt.created_at_unix_ms,
                receipt.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if inserted != 1 {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(())
}

fn load_decision_retention_receipt(
    connection: &Connection,
    retention_batch_id: Uuid,
) -> Result<Option<StoredDecisionRetentionReceipt>, LedgerError> {
    connection
        .query_row(
            "SELECT retention_batch_id, project_uuid, process_instance_id,
                    decision_age_expired, decision_count_excess, source_forced,
                    new_marker_count, new_marker_hash, deleted_decision_count,
                    deleted_summary_count, deleted_neighbor_count, deleted_child_count,
                    verified_aggregate_bytes, selection_lower_bound_unix_ms,
                    selection_upper_bound_unix_ms, decision_selection_hash,
                    blocked_anchor_count, blocked_anchor_hash, deleted_anchor_count,
                    deleted_anchor_hash, more_cleanup, created_at_unix_ms,
                    canonical_payload_hash
             FROM decision_retention_receipts WHERE retention_batch_id = ?1",
            [retention_batch_id.to_string()],
            |row| {
                Ok(StoredDecisionRetentionReceipt {
                    retention_batch_id: row.get(0)?,
                    project_uuid: row.get(1)?,
                    process_instance_id: row.get(2)?,
                    decision_age_expired: row.get(3)?,
                    decision_count_excess: row.get(4)?,
                    source_forced: row.get(5)?,
                    new_marker_count: row.get(6)?,
                    new_marker_hash: row.get(7)?,
                    deleted_decision_count: row.get(8)?,
                    deleted_summary_count: row.get(9)?,
                    deleted_neighbor_count: row.get(10)?,
                    deleted_child_count: row.get(11)?,
                    verified_aggregate_bytes: row.get(12)?,
                    selection_lower_bound_unix_ms: row.get(13)?,
                    selection_upper_bound_unix_ms: row.get(14)?,
                    decision_selection_hash: row.get(15)?,
                    blocked_anchor_count: row.get(16)?,
                    blocked_anchor_hash: row.get(17)?,
                    deleted_anchor_count: row.get(18)?,
                    deleted_anchor_hash: row.get(19)?,
                    more_cleanup: row.get(20)?,
                    created_at_unix_ms: row.get(21)?,
                    canonical_payload_hash: row.get(22)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

fn verify_stored_decision_retention_receipt(
    stored: StoredDecisionRetentionReceipt,
) -> Option<DecisionRetentionReceipt> {
    let booleans = [
        stored.decision_age_expired,
        stored.decision_count_excess,
        stored.source_forced,
        stored.more_cleanup,
    ];
    if booleans.into_iter().any(|value| !matches!(value, 0 | 1))
        || !(0..=i64::try_from(RETENTION_BATCH_LIMIT).ok()?).contains(&stored.new_marker_count)
        || !(0..=i64::try_from(RETENTION_BATCH_LIMIT).ok()?)
            .contains(&stored.deleted_decision_count)
        || !(0..=i64::try_from(DECISION_CHILD_RETENTION_LIMIT).ok()?)
            .contains(&stored.deleted_summary_count)
        || !(0..=i64::try_from(DECISION_CHILD_RETENTION_LIMIT).ok()?)
            .contains(&stored.deleted_neighbor_count)
        || !(0..=i64::try_from(DECISION_CHILD_RETENTION_LIMIT).ok()?)
            .contains(&stored.deleted_child_count)
        || !(0..=i64::try_from(DECISION_RETENTION_BYTES_LIMIT).ok()?)
            .contains(&stored.verified_aggregate_bytes)
        || !(0..=i64::try_from(RETENTION_BATCH_LIMIT).ok()?).contains(&stored.blocked_anchor_count)
        || !(0..=i64::try_from(RETENTION_BATCH_LIMIT).ok()?).contains(&stored.deleted_anchor_count)
        || stored.deleted_child_count
            != stored.deleted_summary_count + stored.deleted_neighbor_count
        || stored.created_at_unix_ms < 0
        || stored
            .selection_lower_bound_unix_ms
            .is_some_and(|value| value < 0)
        || stored
            .selection_upper_bound_unix_ms
            .is_some_and(|value| value < 0)
        || [
            &stored.new_marker_hash,
            &stored.decision_selection_hash,
            &stored.blocked_anchor_hash,
            &stored.deleted_anchor_hash,
            &stored.canonical_payload_hash,
        ]
        .into_iter()
        .any(|hash| !is_sha256(hash))
    {
        return None;
    }
    let empty = stored.deleted_decision_count == 0;
    if (empty
        && (stored.deleted_child_count != 0
            || stored.verified_aggregate_bytes != 0
            || stored.selection_lower_bound_unix_ms.is_some()
            || stored.selection_upper_bound_unix_ms.is_some()
            || stored.decision_age_expired != 0
            || stored.decision_count_excess != 0
            || stored.source_forced != 0))
        || (!empty
            && (stored.deleted_summary_count < stored.deleted_decision_count
                || stored.decision_age_expired == 0
                    && stored.decision_count_excess == 0
                    && stored.source_forced == 0
                || stored.verified_aggregate_bytes == 0
                || stored.selection_lower_bound_unix_ms.is_none()
                || stored.selection_upper_bound_unix_ms.is_none()
                || stored.selection_lower_bound_unix_ms > stored.selection_upper_bound_unix_ms))
    {
        return None;
    }
    let receipt = DecisionRetentionReceipt {
        retention_batch_id: parse_uuid_v7(&stored.retention_batch_id).ok()?,
        project_uuid: parse_uuid_v7(&stored.project_uuid).ok()?,
        process_instance_id: parse_uuid_v7(&stored.process_instance_id).ok()?,
        decision_age_expired: stored.decision_age_expired == 1,
        decision_count_excess: stored.decision_count_excess == 1,
        source_forced: stored.source_forced == 1,
        new_marker_count: u64::try_from(stored.new_marker_count).ok()?,
        new_marker_hash: stored.new_marker_hash,
        deleted_decision_count: u64::try_from(stored.deleted_decision_count).ok()?,
        deleted_summary_count: u64::try_from(stored.deleted_summary_count).ok()?,
        deleted_neighbor_count: u64::try_from(stored.deleted_neighbor_count).ok()?,
        deleted_child_count: u64::try_from(stored.deleted_child_count).ok()?,
        verified_aggregate_bytes: u64::try_from(stored.verified_aggregate_bytes).ok()?,
        selection_lower_bound_unix_ms: stored.selection_lower_bound_unix_ms,
        selection_upper_bound_unix_ms: stored.selection_upper_bound_unix_ms,
        decision_selection_hash: stored.decision_selection_hash,
        blocked_anchor_count: u64::try_from(stored.blocked_anchor_count).ok()?,
        blocked_anchor_hash: stored.blocked_anchor_hash,
        deleted_anchor_count: u64::try_from(stored.deleted_anchor_count).ok()?,
        deleted_anchor_hash: stored.deleted_anchor_hash,
        more_cleanup: stored.more_cleanup == 1,
        created_at_unix_ms: stored.created_at_unix_ms,
        canonical_payload_hash: stored.canonical_payload_hash,
    };
    (decision_retention_receipt_hash(&receipt).ok().as_deref()
        == Some(receipt.canonical_payload_hash.as_str()))
    .then_some(receipt)
}

fn decision_receipt_matches_request(
    receipt: &DecisionRetentionReceipt,
    request: &RetentionRequest,
    project_uuid: Uuid,
    process_instance_id: Uuid,
) -> bool {
    receipt.retention_batch_id == request.retention_batch_id
        && receipt.project_uuid == project_uuid
        && receipt.process_instance_id == process_instance_id
        && receipt.created_at_unix_ms == request.created_at_unix_ms
}

fn to_u64(value: usize) -> Result<u64, LedgerError> {
    u64::try_from(value).map_err(|_| LedgerErrorClass::IdentityInvariant.into())
}

fn to_i64(value: u64) -> Result<i64, LedgerError> {
    i64::try_from(value).map_err(|_| LedgerErrorClass::IdentityInvariant.into())
}

fn terminal_anchor_count(connection: &Connection, project_uuid: Uuid) -> Result<u64, LedgerError> {
    let count = connection
        .query_row(
            "SELECT (
                 SELECT count(DISTINCT state.anchor_id)
                 FROM anchor_state_events AS state
                 JOIN anchors AS anchor ON anchor.anchor_id = state.anchor_id
                 WHERE anchor.project_uuid = ?1 AND state.state <> 'pending'
             ) + (
                 SELECT count(*) FROM outcomes AS outcome
                 JOIN active_experiments AS experiment
                   ON experiment.active_experiment_id = outcome.active_experiment_id
                 WHERE experiment.project_uuid = ?1
             )",
            params![project_uuid.to_string()],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    u64::try_from(count).map_err(|_| LedgerErrorClass::IdentityInvariant.into())
}

fn capacity_observation(
    connection: &Connection,
    project_uuid: Uuid,
    max_evidence_records: u64,
    observed_at_unix_ms: i64,
    decision_more_cleanup: bool,
) -> Result<RetentionCapacityObservation, LedgerError> {
    let terminal_anchor_count = terminal_anchor_count(connection, project_uuid)?;
    Ok(RetentionCapacityObservation {
        terminal_anchor_count,
        capacity_available: terminal_anchor_count < max_evidence_records,
        more_cleanup: decision_more_cleanup
            || !select_global_cleanup_keys(connection, project_uuid, observed_at_unix_ms, 1)?
                .is_empty(),
    })
}

fn select_global_cleanup_keys(
    connection: &Connection,
    project_uuid: Uuid,
    observed_at_unix_ms: i64,
    limit: usize,
) -> Result<BTreeSet<RetentionCleanupKey>, LedgerError> {
    if limit == 0 {
        return Ok(BTreeSet::new());
    }
    let sql_limit =
        i64::try_from(limit).map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let mut candidates = BTreeSet::new();

    let embedding_ids = connection
        .prepare(
            "SELECT embedding_id FROM embeddings AS embedding
             WHERE NOT EXISTS (
                 SELECT 1 FROM evidence_vector_link_state_events
                 WHERE embedding_id = embedding.embedding_id
             )
               AND NOT EXISTS (
                   SELECT 1 FROM vector_materialization_jobs
                   WHERE embedding_id = embedding.embedding_id
               )
             ORDER BY embedding_id LIMIT ?1",
        )
        .map_err(database_error)?
        .query_map(params![sql_limit], |row| row.get::<_, String>(0))
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    candidates.extend(embedding_ids.into_iter().map(|key| RetentionCleanupKey {
        kind: RetentionCleanupKind::Embedding,
        key,
    }));

    let embedding_job_ids = connection
        .prepare(
            "SELECT embedding_job_id FROM embedding_jobs AS job
             WHERE job.lease_owner_process_instance_id IS NULL
               AND NOT EXISTS (
                   SELECT 1 FROM vector_materialization_jobs
                   WHERE embedding_job_id = job.embedding_job_id
               )
               AND (
                   (
                       SELECT state FROM embedding_job_state_events
                       WHERE embedding_job_id = job.embedding_job_id
                       ORDER BY event_seq DESC LIMIT 1
                   ) = 'completed'
                   OR (
                       job.failure_propagation_complete = 1
                       AND (
                           SELECT state FROM embedding_job_state_events
                           WHERE embedding_job_id = job.embedding_job_id
                           ORDER BY event_seq DESC LIMIT 1
                       ) IN ('terminal_failure', 'quarantined')
                   )
               )
             ORDER BY embedding_job_id LIMIT ?1",
        )
        .map_err(database_error)?
        .query_map(params![sql_limit], |row| row.get::<_, String>(0))
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    candidates.extend(
        embedding_job_ids
            .into_iter()
            .map(|key| RetentionCleanupKey {
                kind: RetentionCleanupKind::EmbeddingJob,
                key,
            }),
    );

    let query_hashes = connection
        .prepare(
            "SELECT canonical_query_hash FROM canonical_routing_queries AS query
             WHERE NOT EXISTS (
                 SELECT 1 FROM vectorization_outcomes
                 WHERE canonical_query_hash = query.canonical_query_hash
             )
               AND NOT EXISTS (
                   SELECT 1 FROM embedding_jobs
                   WHERE canonical_query_hash = query.canonical_query_hash
               )
               AND NOT EXISTS (
                   SELECT 1 FROM embeddings
                   WHERE canonical_query_hash = query.canonical_query_hash
               )
               AND NOT EXISTS (
                   SELECT 1 FROM evidence_vector_links
                   WHERE canonical_query_hash = query.canonical_query_hash
               )
               AND NOT EXISTS (
                   SELECT 1 FROM vector_materialization_jobs
                   WHERE canonical_query_hash = query.canonical_query_hash
               )
               AND NOT EXISTS (
                   SELECT 1 FROM decisions
                   WHERE canonical_query_hash = query.canonical_query_hash
               )
             ORDER BY canonical_query_hash LIMIT ?1",
        )
        .map_err(database_error)?
        .query_map(params![sql_limit], |row| row.get::<_, String>(0))
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    candidates.extend(query_hashes.into_iter().map(|key| RetentionCleanupKey {
        kind: RetentionCleanupKind::CanonicalQuery,
        key,
    }));

    let vector_space_ids = connection
        .prepare(
            "SELECT vector_space_id FROM (
                 SELECT DISTINCT mapping.vector_space_id AS vector_space_id
                 FROM pool_vector_space_mappings AS mapping
                 WHERE mapping.project_uuid = ?1
                   AND NOT EXISTS (
                       SELECT 1 FROM process_instances AS process
                       WHERE process.project_uuid = mapping.project_uuid
                         AND process.config_generation_id = mapping.config_generation_id
                         AND process.heartbeat_expires_at_unix_ms > ?2
                         AND (
                             SELECT state FROM process_instance_state_events
                             WHERE subject_process_instance_id = process.process_instance_id
                             ORDER BY event_seq DESC LIMIT 1
                         ) = 'started'
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM shadow_attempts AS attempt
                       WHERE attempt.config_generation_id = mapping.config_generation_id
                         AND attempt.pool_id = mapping.pool_id
                         AND attempt.policy_version_id = mapping.policy_version_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM embedding_jobs AS job
                       WHERE job.vector_space_id = mapping.vector_space_id
                         AND (
                             job.lease_owner_process_instance_id IS NOT NULL
                             OR coalesce((
                                 SELECT state FROM embedding_job_state_events
                                 WHERE embedding_job_id = job.embedding_job_id
                                 ORDER BY event_seq DESC LIMIT 1
                             ), '') NOT IN ('completed', 'terminal_failure', 'quarantined')
                             OR job.failure_propagation_complete = 0
                                 AND coalesce((
                                     SELECT state FROM embedding_job_state_events
                                     WHERE embedding_job_id = job.embedding_job_id
                                     ORDER BY event_seq DESC LIMIT 1
                                 ), '') IN ('terminal_failure', 'quarantined')
                         )
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM vector_index_manifest
                       WHERE vector_space_id = mapping.vector_space_id
                         AND state = 'building'
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM vectorization_outcomes
                       WHERE vector_space_id = mapping.vector_space_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM embedding_jobs
                       WHERE vector_space_id = mapping.vector_space_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM embeddings
                       WHERE vector_space_id = mapping.vector_space_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM evidence_vector_links
                       WHERE vector_space_id = mapping.vector_space_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM vector_materialization_jobs
                       WHERE vector_space_id = mapping.vector_space_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM decisions
                       WHERE project_uuid = mapping.project_uuid
                         AND config_generation_id = mapping.config_generation_id
                         AND pool_id = mapping.pool_id
                         AND policy_version_id = mapping.policy_version_id
                         AND vector_space_id = mapping.vector_space_id
                   )
                 UNION
                 SELECT space.vector_space_id
                 FROM vector_spaces AS space
                 WHERE space.project_uuid = ?1
                   AND NOT EXISTS (
                       SELECT 1 FROM pool_vector_space_mappings
                       WHERE vector_space_id = space.vector_space_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM vectorization_outcomes
                       WHERE vector_space_id = space.vector_space_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM embedding_jobs
                       WHERE vector_space_id = space.vector_space_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM embeddings
                       WHERE vector_space_id = space.vector_space_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM evidence_vector_links
                       WHERE vector_space_id = space.vector_space_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM vector_materialization_jobs
                       WHERE vector_space_id = space.vector_space_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM decisions
                       WHERE vector_space_id = space.vector_space_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM decision_candidate_summaries
                       WHERE vector_space_id = space.vector_space_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM vector_index_manifest
                       WHERE vector_space_id = space.vector_space_id AND state <> 'dropped'
                   )
             ) ORDER BY vector_space_id LIMIT ?3",
        )
        .map_err(database_error)?
        .query_map(
            params![project_uuid.to_string(), observed_at_unix_ms, sql_limit],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    candidates.extend(vector_space_ids.into_iter().map(|key| RetentionCleanupKey {
        kind: RetentionCleanupKind::VectorSpace,
        key,
    }));

    let profile_versions = connection
        .prepare(
            "SELECT embedder_profile_version_id FROM embedder_profiles AS profile
             WHERE NOT EXISTS (
                 SELECT 1 FROM vector_spaces
                 WHERE embedder_profile_version_id = profile.embedder_profile_version_id
             )
               AND NOT EXISTS (
                   SELECT 1 FROM pool_vector_space_mappings
                   WHERE embedder_profile_version_id = profile.embedder_profile_version_id
               )
             ORDER BY embedder_profile_version_id LIMIT ?1",
        )
        .map_err(database_error)?
        .query_map(params![sql_limit], |row| row.get::<_, String>(0))
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    candidates.extend(profile_versions.into_iter().map(|key| RetentionCleanupKey {
        kind: RetentionCleanupKind::Profile,
        key,
    }));

    Ok(candidates.into_iter().take(limit).collect())
}

fn merge_global_cleanup_references(
    connection: &Connection,
    cleanup_selection: &BTreeSet<RetentionCleanupKey>,
    captured: &mut CapturedVectorReferences,
) -> Result<(), LedgerError> {
    for selected in cleanup_selection {
        match selected.kind {
            RetentionCleanupKind::Embedding => {
                let (query_hash, vector_space_id) = connection
                    .query_row(
                        "SELECT canonical_query_hash, vector_space_id FROM embeddings
                         WHERE embedding_id = ?1",
                        params![selected.key],
                        |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                    )
                    .map_err(database_error)?;
                captured.embedding_ids.insert(selected.key.clone());
                captured.canonical_query_hashes.insert(query_hash);
                captured.vector_space_ids.insert(vector_space_id);
            }
            RetentionCleanupKind::EmbeddingJob => {
                let (query_hash, vector_space_id) = connection
                    .query_row(
                        "SELECT canonical_query_hash, vector_space_id FROM embedding_jobs
                         WHERE embedding_job_id = ?1",
                        params![selected.key],
                        |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                    )
                    .map_err(database_error)?;
                captured.embedding_job_ids.insert(selected.key.clone());
                captured.canonical_query_hashes.insert(query_hash);
                captured.vector_space_ids.insert(vector_space_id);
            }
            RetentionCleanupKind::CanonicalQuery => {
                captured.canonical_query_hashes.insert(selected.key.clone());
            }
            RetentionCleanupKind::VectorSpace => {
                captured.vector_space_ids.insert(selected.key.clone());
            }
            RetentionCleanupKind::Profile => {
                captured.profile_versions.insert(selected.key.clone());
            }
        }
    }
    Ok(())
}

fn verify_global_cleanup_selection(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    cleanup_selection: &BTreeSet<RetentionCleanupKey>,
) -> Result<(), LedgerError> {
    for selected in cleanup_selection {
        match selected.kind {
            RetentionCleanupKind::Embedding => {
                let (embedding_id, vector_space_id, query_hash) = transaction
                    .query_row(
                        "SELECT embedding_id, vector_space_id, canonical_query_hash
                         FROM embeddings WHERE embedding_id = ?1",
                        params![selected.key],
                        |row| {
                            Ok((
                                row.get::<_, String>(0)?,
                                row.get::<_, String>(1)?,
                                row.get::<_, String>(2)?,
                            ))
                        },
                    )
                    .map_err(database_error)?;
                let vector_space_id = VectorSpaceId::new(vector_space_id)
                    .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
                let cache =
                    load_embedding_cache(transaction, project_uuid, &vector_space_id, &query_hash)?
                        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
                if cache.embedding_id.to_string() != embedding_id {
                    return Err(LedgerErrorClass::CorruptDatabase.into());
                }
            }
            RetentionCleanupKind::EmbeddingJob => {
                if load_verified_embedding_job(transaction, project_uuid, &selected.key)?.is_none()
                {
                    return Err(LedgerErrorClass::CorruptDatabase.into());
                }
            }
            RetentionCleanupKind::CanonicalQuery => {
                if load_canonical_query(transaction, &selected.key)?.is_none() {
                    return Err(LedgerErrorClass::CorruptDatabase.into());
                }
            }
            RetentionCleanupKind::VectorSpace => {
                verify_global_vector_space_authority(transaction, project_uuid, &selected.key)?;
            }
            RetentionCleanupKind::Profile => {
                if resolve_embedder_profile(transaction, &selected.key)?.is_none() {
                    return Err(LedgerErrorClass::CorruptDatabase.into());
                }
            }
        }
    }
    Ok(())
}

fn verify_global_vector_space_authority(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    vector_space_id: &str,
) -> Result<(), LedgerError> {
    let vector_space_id = VectorSpaceId::new(vector_space_id.to_string())
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let space = resolve_vector_space(transaction, project_uuid, &vector_space_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if resolve_embedder_profile(transaction, &space.space.embedder_profile_version_id)?.is_none() {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let mappings = transaction
        .prepare(
            "SELECT config_generation_id, pool_id, policy_version_id
             FROM pool_vector_space_mappings
             WHERE project_uuid = ?1 AND vector_space_id = ?2
             ORDER BY config_generation_id, pool_id, policy_version_id",
        )
        .map_err(database_error)?
        .query_map(
            params![project_uuid.to_string(), vector_space_id.as_str()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    for (config_generation_id, pool_id, policy_version_id) in mappings {
        let key = FrozenMappingKey::new(
            project_uuid,
            config_generation_id,
            pool_id,
            policy_version_id,
        )
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        let mapping = resolve_frozen_mapping(transaction, &key)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        if mapping.mapping.vector_space_id != vector_space_id {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }

    let generations = transaction
        .prepare(
            "SELECT generation FROM vector_index_manifest
             WHERE vector_space_id = ?1 ORDER BY generation",
        )
        .map_err(database_error)?
        .query_map(params![vector_space_id.as_str()], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    for generation in generations {
        let generation = VectorIndexGeneration::new(generation)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        let manifest = load_validated_manifest(transaction, &vector_space_id, generation)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        let objects = verify_generation_objects(transaction, manifest.authority())?;
        if objects == GenerationObjectsStatus::Partial
            || manifest.state() == VectorIndexManifestState::Dropped
                && objects != GenerationObjectsStatus::Missing
        {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }
    if transaction
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM vector_index_rebuild_leases
                            WHERE vector_space_id = ?1)",
            params![vector_space_id.as_str()],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)?
        && load_validated_rebuild_lease(transaction, &vector_space_id)?.is_none()
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let partition_ids = transaction
        .prepare(
            "SELECT partition_id FROM routing_partitions
             WHERE vector_space_id = ?1 ORDER BY partition_id",
        )
        .map_err(database_error)?
        .query_map(params![vector_space_id.as_str()], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    for partition_id in partition_ids {
        let partition_id = crate::vector::PartitionId::new(partition_id)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        if !verify_historical_routing_partition(
            transaction,
            partition_id,
            project_uuid,
            &vector_space_id,
        )? {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }
    Ok(())
}

fn captured_vector_references(
    connection: &Connection,
    selected_anchor_ids: &BTreeSet<String>,
) -> Result<CapturedVectorReferences, LedgerError> {
    let mut captured = CapturedVectorReferences::default();
    for anchor_id in selected_anchor_ids {
        let embedding_ids = connection
            .prepare(
                "SELECT DISTINCT state.embedding_id
                 FROM evidence_vector_links AS link
                 JOIN evidence_vector_link_state_events AS state
                   ON state.evidence_vector_link_id = link.evidence_vector_link_id
                 WHERE link.anchor_id = ?1 AND state.embedding_id IS NOT NULL
                 ORDER BY state.embedding_id",
            )
            .map_err(database_error)?
            .query_map(params![anchor_id], |row| row.get::<_, String>(0))
            .map_err(database_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(database_error)?;
        captured.embedding_ids.extend(embedding_ids);

        let materializations = connection
            .prepare(
                "SELECT materialization.embedding_job_id, materialization.embedding_id,
                        materialization.canonical_query_hash,
                        materialization.vector_space_id
                 FROM vector_materialization_jobs AS materialization
                 JOIN evidence_vector_links AS link
                   ON link.evidence_vector_link_id = materialization.evidence_vector_link_id
                 WHERE link.anchor_id = ?1
                 ORDER BY materialization.vector_materialization_job_id",
            )
            .map_err(database_error)?
            .query_map(params![anchor_id], |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            })
            .map_err(database_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(database_error)?;
        for (embedding_job_id, embedding_id, query_hash, vector_space_id) in materializations {
            captured.embedding_job_ids.extend(embedding_job_id);
            captured.embedding_ids.extend(embedding_id);
            captured.canonical_query_hashes.insert(query_hash);
            captured.vector_space_ids.insert(vector_space_id);
        }

        let links = connection
            .prepare(
                "SELECT vector_space_id, canonical_query_hash, partition_id
                 FROM evidence_vector_links
                 WHERE anchor_id = ?1
                 ORDER BY evidence_vector_link_id",
            )
            .map_err(database_error)?
            .query_map(params![anchor_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            })
            .map_err(database_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(database_error)?;
        for (vector_space_id, query_hash, partition_id) in links {
            captured.vector_space_ids.insert(vector_space_id);
            captured.canonical_query_hashes.insert(query_hash);
            captured.partition_ids.insert(partition_id);
        }

        let outcomes = connection
            .prepare(
                "SELECT vector_space_id, canonical_query_hash
                 FROM vectorization_outcomes
                 WHERE anchor_id = ?1
                 ORDER BY vectorization_outcome_id",
            )
            .map_err(database_error)?
            .query_map(params![anchor_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?))
            })
            .map_err(database_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(database_error)?;
        for (vector_space_id, query_hash) in outcomes {
            captured.vector_space_ids.insert(vector_space_id);
            captured.canonical_query_hashes.extend(query_hash);
        }
    }
    Ok(captured)
}

fn selected_materialization_deletes(
    connection: &Connection,
    selected_anchor_ids: &BTreeSet<String>,
    deleted_at_unix_ms: i64,
) -> Result<Vec<MaterializationRetentionDelete>, LedgerError> {
    let mut selected = BTreeSet::new();
    for anchor_id in selected_anchor_ids {
        let rows = connection
            .prepare(
                "SELECT materialization.vector_materialization_job_id,
                        materialization.attempt_generation,
                        materialization.canonical_payload_hash
                 FROM vector_materialization_jobs AS materialization
                 JOIN evidence_vector_links AS link
                   ON link.evidence_vector_link_id = materialization.evidence_vector_link_id
                 WHERE link.anchor_id = ?1
                 ORDER BY materialization.vector_materialization_job_id",
            )
            .map_err(database_error)?
            .query_map(params![anchor_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(database_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(database_error)?;
        selected.extend(rows);
    }
    selected
        .into_iter()
        .map(
            |(materialization_job_id, attempt_generation, canonical_payload_hash)| {
                MaterializationRetentionDelete::new(
                    Uuid::now_v7(),
                    Uuid::now_v7(),
                    materialization_job_id,
                    attempt_generation,
                    canonical_payload_hash,
                    deleted_at_unix_ms,
                )
            },
        )
        .collect()
}

fn verify_selected_vector_graph(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    selected_anchor_ids: &BTreeSet<String>,
    captured: &CapturedVectorReferences,
) -> Result<(), LedgerError> {
    for anchor_id in selected_anchor_ids {
        let source_retiring = transaction
            .query_row(
                "SELECT EXISTS(
                     SELECT 1 FROM decision_retiring_anchors WHERE anchor_id = ?1
                 )",
                [anchor_id],
                |row| row.get::<_, bool>(0),
            )
            .map_err(database_error)?;
        let outcomes = transaction
            .prepare(
                "SELECT vectorization_outcome_id, shadow_attempt_id, anchor_id,
                        learning_generation_id, vector_space_id, canonical_query_hash,
                        outcome, stable_reason, created_at_unix_ms, canonical_payload_hash
                 FROM vectorization_outcomes WHERE anchor_id = ?1
                 ORDER BY vectorization_outcome_id",
            )
            .map_err(database_error)?
            .query_map(params![anchor_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, Option<String>>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, String>(9)?,
                ))
            })
            .map_err(database_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(database_error)?;
        for (id, attempt, anchor, learning, space, query, outcome, reason, created_at, hash) in
            outcomes
        {
            let expected = hash_json(&json!({
                "vectorization_outcome_id": id,
                "shadow_attempt_id": parse_uuid_v7(&attempt)?,
                "anchor_id": parse_uuid_v7(&anchor)?,
                "learning_generation_id": parse_uuid_v7(&learning)?,
                "vector_space_id": space,
                "canonical_query_hash": query,
                "outcome": outcome,
                "stable_reason": reason,
                "created_at_unix_ms": created_at,
            }))?;
            if created_at < 0 || expected != hash {
                return Err(LedgerErrorClass::CorruptDatabase.into());
            }
            let attempt = parse_uuid_v7(&attempt)?;
            let space = VectorSpaceId::new(space)
                .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
            if !source_retiring
                && !verify_retained_vector_graph_in_transaction(
                    transaction,
                    project_uuid,
                    attempt,
                    &space,
                )?
            {
                return Err(LedgerErrorClass::CorruptDatabase.into());
            }
        }

        let links = transaction
            .prepare(
                "SELECT evidence_vector_link_id, vectorization_outcome_id,
                        shadow_attempt_id, anchor_id, root_uuid, learning_generation_id,
                        vector_space_id, partition_id, canonical_query_hash,
                        terminal_class, evaluation_id, quality_label,
                        created_at_unix_ms, canonical_payload_hash
                 FROM evidence_vector_links WHERE anchor_id = ?1
                 ORDER BY evidence_vector_link_id",
            )
            .map_err(database_error)?
            .query_map(params![anchor_id], |row| {
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
                    row.get::<_, String>(9)?,
                    row.get::<_, Option<String>>(10)?,
                    row.get::<_, Option<String>>(11)?,
                    row.get::<_, i64>(12)?,
                    row.get::<_, String>(13)?,
                ))
            })
            .map_err(database_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(database_error)?;
        for (
            id,
            outcome_id,
            attempt,
            anchor,
            root,
            learning,
            space,
            partition,
            query,
            terminal,
            evaluation,
            quality,
            created_at,
            hash,
        ) in links
        {
            let attempt_id = parse_uuid_v7(&attempt)?;
            let vector_space_id = VectorSpaceId::new(space.clone())
                .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
            let link_id = parse_uuid_v7(&id)?;
            let expected = hash_json(&json!({
                "evidence_vector_link_id": link_id,
                "vectorization_outcome_id": outcome_id,
                "shadow_attempt_id": parse_uuid_v7(&attempt)?,
                "anchor_id": parse_uuid_v7(&anchor)?,
                "root_uuid": parse_uuid_v7(&root)?,
                "learning_generation_id": parse_uuid_v7(&learning)?,
                "vector_space_id": space,
                "partition_id": partition,
                "canonical_query_hash": query,
                "terminal_class": terminal,
                "evaluation_id": evaluation
                    .as_deref()
                    .map(parse_uuid_v7)
                    .transpose()?,
                "quality_label": quality,
                "created_at_unix_ms": created_at,
            }))?;
            let materialization_count = transaction
                .query_row(
                    "SELECT count(*) FROM vector_materialization_jobs
                     WHERE evidence_vector_link_id = ?1",
                    params![id],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(database_error)?;
            if created_at < 0 || partition <= 0 || expected != hash || materialization_count != 1 {
                return Err(LedgerErrorClass::CorruptDatabase.into());
            }
            if !source_retiring
                && !verify_retained_vector_graph_in_transaction(
                    transaction,
                    project_uuid,
                    attempt_id,
                    &vector_space_id,
                )?
            {
                return Err(LedgerErrorClass::CorruptDatabase.into());
            }
        }
        if source_retiring
            && !verify_retiring_anchor_deindexed_in_transaction(
                transaction,
                project_uuid,
                parse_uuid_v7(anchor_id)?,
            )?
        {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }

    for query_hash in &captured.canonical_query_hashes {
        if load_canonical_query(transaction, query_hash)?.is_none() {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }
    for embedding_job_id in &captured.embedding_job_ids {
        if load_verified_embedding_job(transaction, project_uuid, embedding_job_id)?.is_none() {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }
    for embedding_id in &captured.embedding_ids {
        let (space, query) = transaction
            .query_row(
                "SELECT vector_space_id, canonical_query_hash FROM embeddings
                 WHERE embedding_id = ?1",
                params![embedding_id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .map_err(database_error)?;
        let space = VectorSpaceId::new(space)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        let cache = load_embedding_cache(transaction, project_uuid, &space, &query)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
        if cache.embedding_id.to_string() != *embedding_id {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }
    for vector_space_id in &captured.vector_space_ids {
        verify_global_vector_space_authority(transaction, project_uuid, vector_space_id)?;
    }
    Ok(())
}

fn delete_captured_unreferenced_embeddings(
    connection: &Connection,
    captured_embedding_ids: &BTreeSet<String>,
) -> Result<(), LedgerError> {
    for embedding_id in captured_embedding_ids {
        connection
            .execute(
                "DELETE FROM embeddings
                 WHERE embedding_id = ?1
                   AND NOT EXISTS (
                       SELECT 1 FROM evidence_vector_link_state_events
                       WHERE embedding_id = ?1
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM vector_materialization_jobs
                       WHERE embedding_id = ?1
                   )",
                params![embedding_id],
            )
            .map_err(database_error)?;
    }
    Ok(())
}

fn delete_captured_terminal_embedding_jobs(
    connection: &Connection,
    captured_embedding_job_ids: &BTreeSet<String>,
) -> Result<(), LedgerError> {
    for embedding_job_id in captured_embedding_job_ids {
        connection
            .execute(
                "DELETE FROM embedding_jobs
                 WHERE embedding_job_id = ?1
                   AND lease_owner_process_instance_id IS NULL
                   AND NOT EXISTS (
                       SELECT 1 FROM vector_materialization_jobs
                       WHERE embedding_job_id = ?1
                   )
                   AND (
                       (
                           SELECT state FROM embedding_job_state_events
                           WHERE embedding_job_id = ?1
                           ORDER BY event_seq DESC LIMIT 1
                       ) = 'completed'
                       OR (
                           failure_propagation_complete = 1
                           AND (
                               SELECT state FROM embedding_job_state_events
                               WHERE embedding_job_id = ?1
                               ORDER BY event_seq DESC LIMIT 1
                           ) IN ('terminal_failure', 'quarantined')
                       )
                   )",
                params![embedding_job_id],
            )
            .map_err(database_error)?;
    }
    Ok(())
}

fn delete_captured_unreferenced_queries(
    connection: &Connection,
    captured_query_hashes: &BTreeSet<String>,
) -> Result<(), LedgerError> {
    for query_hash in captured_query_hashes {
        connection
            .execute(
                "DELETE FROM canonical_routing_queries
                 WHERE canonical_query_hash = ?1
                   AND NOT EXISTS (
                       SELECT 1 FROM vectorization_outcomes
                       WHERE canonical_query_hash = ?1
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM embedding_jobs WHERE canonical_query_hash = ?1
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM embeddings WHERE canonical_query_hash = ?1
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM evidence_vector_links WHERE canonical_query_hash = ?1
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM vector_materialization_jobs
                       WHERE canonical_query_hash = ?1
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM decisions WHERE canonical_query_hash = ?1
                   )",
                params![query_hash],
            )
            .map_err(database_error)?;
    }
    Ok(())
}

fn delete_captured_historical_vector_authority(
    connection: &Transaction<'_>,
    project_uuid: Uuid,
    observed_at_unix_ms: i64,
    captured: &CapturedVectorReferences,
) -> Result<(), LedgerError> {
    let mut profile_versions = captured.profile_versions.clone();
    for vector_space_id in &captured.vector_space_ids {
        let Some(profile_version) = connection
            .query_row(
                "SELECT embedder_profile_version_id FROM vector_spaces
                 WHERE vector_space_id = ?1 AND project_uuid = ?2",
                params![vector_space_id, project_uuid.to_string()],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(database_error)?
        else {
            continue;
        };
        profile_versions.insert(profile_version);
        let mappings = connection
            .prepare(
                "SELECT config_generation_id, pool_id, policy_version_id
                 FROM pool_vector_space_mappings
                 WHERE project_uuid = ?1 AND vector_space_id = ?2
                 ORDER BY config_generation_id, pool_id, policy_version_id",
            )
            .map_err(database_error)?
            .query_map(params![project_uuid.to_string(), vector_space_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(database_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(database_error)?;
        for (config_generation_id, pool_id, policy_version_id) in mappings {
            if mapping_has_retained_attempt(
                connection,
                &config_generation_id,
                &pool_id,
                &policy_version_id,
                vector_space_id,
            )? || vector_space_has_unfinished_embedding_work(connection, vector_space_id)?
                || vector_space_has_building_generation(connection, vector_space_id)?
                || vector_space_has_retained_graph(connection, vector_space_id)?
                || config_has_verified_live_process(
                    connection,
                    project_uuid,
                    &config_generation_id,
                    observed_at_unix_ms,
                )?
            {
                continue;
            }
            connection
                .execute(
                    "DELETE FROM pool_vector_space_mappings
                     WHERE project_uuid = ?1 AND config_generation_id = ?2
                       AND pool_id = ?3 AND policy_version_id = ?4
                       AND vector_space_id = ?5",
                    params![
                        project_uuid.to_string(),
                        config_generation_id,
                        pool_id,
                        policy_version_id,
                        vector_space_id,
                    ],
                )
                .map_err(database_error)?;
        }
    }

    for partition_id in &captured.partition_ids {
        connection
            .execute(
                "DELETE FROM routing_partitions
                 WHERE partition_id = ?1
                   AND NOT EXISTS (
                       SELECT 1 FROM evidence_vector_links WHERE partition_id = ?1
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM vector_source_change_events WHERE partition_id = ?1
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM decision_candidate_summaries WHERE partition_id = ?1
                   )",
                params![partition_id],
            )
            .map_err(database_error)?;
    }

    for vector_space_id in &captured.vector_space_ids {
        if vector_space_has_retained_graph(connection, vector_space_id)? {
            continue;
        }
        let has_mapping_or_rebuild_authority = connection
            .query_row(
                "SELECT
                    EXISTS(SELECT 1 FROM pool_vector_space_mappings
                           WHERE vector_space_id = ?1)
                    OR EXISTS(SELECT 1 FROM vector_index_rebuild_leases
                              WHERE vector_space_id = ?1)",
                params![vector_space_id],
                |row| row.get::<_, bool>(0),
            )
            .map_err(database_error)?;
        if has_mapping_or_rebuild_authority {
            continue;
        }
        let vector_space = VectorSpaceId::new(vector_space_id.clone())
            .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if let Some(current) = current_generation_manifest(connection, &vector_space)? {
            if !matches!(
                current.state(),
                VectorIndexManifestState::Active | VectorIndexManifestState::Unavailable
            ) {
                return Err(LedgerErrorClass::CorruptDatabase.into());
            }
            match retire_current_generation_for_retention(
                connection,
                &vector_space,
                current.generation(),
                current.canonical_payload_hash(),
                observed_at_unix_ms,
            )? {
                GenerationRetirementAck::Applied | GenerationRetirementAck::AlreadyApplied => {
                    continue;
                }
                GenerationRetirementAck::Missing | GenerationRetirementAck::Stale => {
                    return Err(LedgerErrorClass::IdentityInvariant.into());
                }
                GenerationRetirementAck::Conflict => {
                    return Err(LedgerErrorClass::CorruptDatabase.into());
                }
            }
        }
        let has_live_generation = connection
            .query_row(
                "SELECT EXISTS(
                     SELECT 1 FROM vector_index_manifest
                     WHERE vector_space_id = ?1 AND state <> 'dropped'
                 )",
                params![vector_space_id],
                |row| row.get::<_, bool>(0),
            )
            .map_err(database_error)?;
        if has_live_generation {
            continue;
        }
        if matches!(
            prune_vector_source_history_for_retention(connection, &vector_space)?,
            SourceHistoryRetentionAck::More { .. }
        ) {
            continue;
        }
        connection
            .execute(
                "DELETE FROM vector_index_manifest
                 WHERE vector_space_id = ?1 AND state = 'dropped'",
                params![vector_space_id],
            )
            .map_err(database_error)?;
        connection
            .execute(
                "DELETE FROM routing_partitions
                 WHERE vector_space_id = ?1
                   AND NOT EXISTS (
                       SELECT 1 FROM evidence_vector_links
                       WHERE partition_id = routing_partitions.partition_id
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM decision_candidate_summaries
                       WHERE partition_id = routing_partitions.partition_id
                   )",
                params![vector_space_id],
            )
            .map_err(database_error)?;
        connection
            .execute(
                "DELETE FROM vector_space_source_sequences WHERE vector_space_id = ?1",
                params![vector_space_id],
            )
            .map_err(database_error)?;
        connection
            .execute(
                "DELETE FROM vector_spaces
                 WHERE vector_space_id = ?1 AND project_uuid = ?2",
                params![vector_space_id, project_uuid.to_string()],
            )
            .map_err(database_error)?;
    }

    for profile_version in profile_versions {
        connection
            .execute(
                "DELETE FROM embedder_profiles
                 WHERE embedder_profile_version_id = ?1
                   AND NOT EXISTS (
                       SELECT 1 FROM vector_spaces
                       WHERE embedder_profile_version_id = ?1
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM pool_vector_space_mappings
                       WHERE embedder_profile_version_id = ?1
                   )",
                params![profile_version],
            )
            .map_err(database_error)?;
    }
    Ok(())
}

fn vector_space_has_retained_graph(
    connection: &Connection,
    vector_space_id: &str,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT
                EXISTS(SELECT 1 FROM vectorization_outcomes WHERE vector_space_id = ?1)
                OR EXISTS(SELECT 1 FROM embedding_jobs WHERE vector_space_id = ?1)
                OR EXISTS(SELECT 1 FROM embeddings WHERE vector_space_id = ?1)
                OR EXISTS(SELECT 1 FROM evidence_vector_links WHERE vector_space_id = ?1)
                OR EXISTS(SELECT 1 FROM vector_materialization_jobs
                          WHERE vector_space_id = ?1)
                OR EXISTS(SELECT 1 FROM decisions WHERE vector_space_id = ?1)
                OR EXISTS(SELECT 1 FROM decision_candidate_summaries
                          WHERE vector_space_id = ?1)",
            params![vector_space_id],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)
}

fn mapping_has_retained_attempt(
    connection: &Connection,
    config_generation_id: &str,
    pool_id: &str,
    policy_version_id: &str,
    vector_space_id: &str,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM shadow_attempts AS attempt
                 WHERE attempt.config_generation_id = ?1
                   AND attempt.pool_id = ?2
                   AND attempt.policy_version_id = ?3
             ) OR EXISTS(
                 SELECT 1 FROM decisions AS decision
                 WHERE decision.config_generation_id = ?1
                   AND decision.pool_id = ?2
                   AND decision.policy_version_id = ?3
                   AND decision.vector_space_id = ?4
             )",
            params![
                config_generation_id,
                pool_id,
                policy_version_id,
                vector_space_id
            ],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)
}

fn vector_space_has_unfinished_embedding_work(
    connection: &Connection,
    vector_space_id: &str,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM embedding_jobs AS job
                 WHERE job.vector_space_id = ?1
                   AND (
                       job.lease_owner_process_instance_id IS NOT NULL
                       OR coalesce((
                           SELECT state FROM embedding_job_state_events
                           WHERE embedding_job_id = job.embedding_job_id
                           ORDER BY event_seq DESC LIMIT 1
                       ), '') NOT IN ('completed', 'terminal_failure', 'quarantined')
                       OR (
                           coalesce((
                               SELECT state FROM embedding_job_state_events
                               WHERE embedding_job_id = job.embedding_job_id
                               ORDER BY event_seq DESC LIMIT 1
                           ), '') IN ('terminal_failure', 'quarantined')
                           AND job.failure_propagation_complete = 0
                       )
                   )
             )",
            params![vector_space_id],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)
}

fn vector_space_has_building_generation(
    connection: &Connection,
    vector_space_id: &str,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                 SELECT 1 FROM vector_index_manifest
                 WHERE vector_space_id = ?1 AND state = 'building'
             )",
            params![vector_space_id],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)
}

fn config_has_verified_live_process(
    connection: &Connection,
    project_uuid: Uuid,
    config_generation_id: &str,
    observed_at_unix_ms: i64,
) -> Result<bool, LedgerError> {
    let process_ids = connection
        .prepare(
            "SELECT process_instance_id FROM process_instances
             WHERE project_uuid = ?1 AND config_generation_id = ?2
             ORDER BY process_instance_id",
        )
        .map_err(database_error)?
        .query_map(
            params![project_uuid.to_string(), config_generation_id],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    for process_id in process_ids {
        match verified_process_status_at(
            connection,
            project_uuid,
            parse_uuid_v7(&process_id)?,
            observed_at_unix_ms,
        )? {
            ProcessStatusAt::Live => return Ok(true),
            ProcessStatusAt::Expired | ProcessStatusAt::Terminal => {}
            ProcessStatusAt::Invalid => {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }
    }
    Ok(false)
}

fn verify_no_foreign_key_violations(connection: &Connection) -> Result<(), LedgerError> {
    let mut statement = connection
        .prepare("PRAGMA foreign_key_check")
        .map_err(database_error)?;
    let mut rows = statement.query([]).map_err(database_error)?;
    if rows.next().map_err(database_error)?.is_some() {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn parse_uuid_v7(value: &str) -> Result<Uuid, LedgerError> {
    let parsed = Uuid::parse_str(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if parsed.to_string() != value {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    validate_uuid_v7(parsed)?;
    Ok(parsed)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn hash_json(value: &Json) -> Result<String, LedgerError> {
    canonical_sha256(value).map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn database_error(error: rusqlite::Error) -> LedgerError {
    map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}

fn validate_uuid_v7(value: Uuid) -> Result<(), LedgerError> {
    if value.get_version_num() != 7 || value.get_variant() != uuid::Variant::RFC4122 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::{Duration, Instant};

    use chrono::{TimeZone, Utc};
    use nemo_relay::api::llm::LlmApiFamily;
    use rusqlite::params;
    use serde::Serialize;
    use serde_json::json;
    use tempfile::tempdir;

    use super::*;
    use crate::canonical_query::build_canonical_routing_query;
    use crate::config::LearningConfig;
    use crate::ledger::model::{LedgerErrorClass, LedgerRuntimeIdentity};
    use crate::ledger::repository::anchors::{
        AnchorCommandAck, FrozenPendingAnchorV1, FrozenTerminalAnchorV1, NotScheduledQueueFull,
        NotScheduledQueueFullAck,
    };
    use crate::ledger::repository::cooloff::{
        CandidateDependencyIdentity, DependencyCommandAck, DependencyCompletion,
        DependencyFailureClass, DependencyOperation, DependencyTransition,
    };
    use crate::ledger::repository::decision::{DecisionAuditAck, tests as decision_test_fixtures};
    use crate::ledger::repository::process::{HeartbeatRenewal, ProcessStop};
    use crate::ledger::repository::shadow::{
        ReservedShadowAttempt, SampleBatchReservation, SampleBatchTerminalEvent,
        SampleBatchTerminalState, ShadowAttemptStarted, ShadowCommandAck,
        ShadowOperationalFailureClass, ShadowTerminalClass, ShadowTerminalRecord,
        ShadowVectorSourceV1,
    };
    use crate::ledger::repository::tests::{config, database_path, recovery_snapshot};
    use crate::ledger::repository::vector_catalog::{
        CanonicalQueryEnsureAck, ensure_canonical_query,
    };
    use crate::projection::{SanitizedMessage, SanitizedMessageContent};
    use crate::trajectory::test_fixtures::pending_window;
    use crate::trajectory::{
        CANDIDATE_FACT_SCHEMA_V1, PendingTrajectoryWindow, PersistedCandidateCapabilitiesV1,
        PersistedCandidateFactV1, PersistedTrajectoryTerminalV1, TrajectoryTrigger,
    };

    const CREATED_AT: i64 = 1_700_000_001_000;

    fn activate(
        max_evidence_records: u64,
        retention_days: u32,
    ) -> (tempfile::TempDir, super::super::ActivatedLedger) {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut config = config(&path, "retention-project");
        config.max_evidence_records = max_evidence_records;
        config.retention_days = retention_days;
        let activated = LedgerRepository::activate_at(&config, CREATED_AT).unwrap();
        (temporary, activated)
    }

    fn canonical_hash_without_field<T: Serialize>(value: &T, field: &str) -> String {
        let mut value = serde_json::to_value(value).unwrap();
        value.as_object_mut().unwrap().remove(field);
        hash_json(&value).unwrap()
    }

    fn pending(identity: &LedgerRuntimeIdentity, anchor_id: Uuid) -> PendingTrajectoryWindow {
        let pool = identity.pools.get("pool-a").unwrap();
        let mut pending = pending_window(anchor_id);
        pending.anchor_call_uuid = Uuid::now_v7();
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
        pending.request_projection.normalized_request.messages = vec![SanitizedMessage::User {
            content: SanitizedMessageContent::Text("route this request".into()),
            name: None,
        }];
        pending.request_projection.semantic_request_fingerprint = canonical_hash_without_field(
            &pending.request_projection,
            "semantic_request_fingerprint",
        );
        pending.routing_context_projection.tenant_policy_hash = "2".repeat(64);
        pending.routing_context_projection.agent_policy_hash = "3".repeat(64);
        pending.normalized_anchor_response.model = Some("anchor-a".into());
        pending
            .normalized_anchor_response
            .semantic_response_fingerprint = canonical_hash_without_field(
            &pending.normalized_anchor_response,
            "semantic_response_fingerprint",
        );
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

    fn seed_queue_full_anchor(
        activated: &mut super::super::ActivatedLedger,
        created_at_unix_ms: i64,
    ) -> Uuid {
        let anchor_id = Uuid::now_v7();
        let pending = pending(&activated.identity, anchor_id);
        let frozen = FrozenPendingAnchorV1::new(
            &pending,
            Uuid::now_v7(),
            Uuid::now_v7(),
            created_at_unix_ms,
        )
        .unwrap();
        let decline =
            NotScheduledQueueFull::new(frozen, Uuid::now_v7(), created_at_unix_ms + 1).unwrap();
        assert!(matches!(
            activated
                .repository
                .record_not_scheduled_queue_full(&decline)
                .unwrap(),
            NotScheduledQueueFullAck::Applied { .. }
        ));
        anchor_id
    }

    fn seed_closed_anchor_without_batch(
        activated: &mut super::super::ActivatedLedger,
        created_at_unix_ms: i64,
    ) -> (Uuid, PendingTrajectoryWindow) {
        let anchor_id = Uuid::now_v7();
        let pending = pending(&activated.identity, anchor_id);
        let frozen_pending = FrozenPendingAnchorV1::new(
            &pending,
            Uuid::now_v7(),
            Uuid::now_v7(),
            created_at_unix_ms,
        )
        .unwrap();
        assert!(matches!(
            activated
                .repository
                .record_pending_anchor(&frozen_pending)
                .unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
        let terminal = PersistedTrajectoryTerminalV1::closed(
            pending.clone(),
            Vec::new(),
            1,
            TrajectoryTrigger::ProgressReached,
            Utc.timestamp_millis_opt(created_at_unix_ms + 1)
                .single()
                .unwrap(),
            Vec::new(),
        );
        let frozen_terminal = FrozenTerminalAnchorV1::new(
            &terminal,
            Uuid::now_v7(),
            Uuid::now_v7(),
            created_at_unix_ms + 1,
        )
        .unwrap();
        assert!(matches!(
            activated
                .repository
                .record_terminal_anchor(&frozen_terminal)
                .unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
        (anchor_id, pending)
    }

    fn reserve_open_batch(
        activated: &mut super::super::ActivatedLedger,
        anchor_id: Uuid,
        pending: &PendingTrajectoryWindow,
        created_at_unix_ms: i64,
    ) -> (ReservedShadowAttempt, SampleBatchReservation) {
        let pool = activated.identity.pools.get("pool-a").unwrap();
        let evaluator_version = config(Path::new("router.db"), "retention-policy").pools[0]
            .judge
            .evaluator_version()
            .unwrap();
        let mut candidate_request = pending.request_projection.clone();
        candidate_request.normalized_request.model = Some("candidate-model-a".into());
        candidate_request.semantic_request_fingerprint =
            canonical_hash_without_field(&candidate_request, "semantic_request_fingerprint");
        let attempt = ReservedShadowAttempt::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            "candidate-a",
            "candidate-model-a",
            "2026-06-01",
            0,
            LlmApiFamily::OpenAIChatCompletions,
            "fixture",
            "anchor-a",
            "2026-07-01",
            "1".repeat(64),
            evaluator_version,
            "2".repeat(64),
            "3".repeat(64),
            true,
            candidate_request,
            created_at_unix_ms,
        )
        .unwrap();
        let reservation = SampleBatchReservation::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            anchor_id,
            activated.identity.config_generation_id.clone(),
            pool.policy_version_id.clone(),
            pool.learning_generation_id,
            "pool-a",
            vec![attempt.clone()],
            created_at_unix_ms,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .reserve_sample_batch(&reservation)
                .unwrap(),
            ShadowCommandAck::Applied
        );
        (attempt, reservation)
    }

    fn terminalize_attempt(
        activated: &mut super::super::ActivatedLedger,
        attempt: &ReservedShadowAttempt,
        reservation: &SampleBatchReservation,
        created_at_unix_ms: i64,
    ) {
        assert_eq!(
            activated
                .repository
                .start_shadow_attempt(
                    ShadowAttemptStarted::new(
                        attempt.shadow_attempt_id,
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                        created_at_unix_ms - 1,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ShadowCommandAck::Applied
        );
        let terminal = ShadowTerminalRecord::new(
            Uuid::now_v7(),
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            ShadowTerminalClass::OperationalFailure,
            None,
            None,
            None,
            Some(ShadowOperationalFailureClass::new("router.provider.timeout").unwrap()),
            Some(1),
            None,
            None,
            ShadowVectorSourceV1::Canonicalizable {
                query_inputs: Box::new(attempt.request_projection.clone()),
            },
            Some(
                SampleBatchTerminalEvent::new(
                    reservation.sample_batch_id,
                    Uuid::now_v7(),
                    SampleBatchTerminalState::Closed,
                    None,
                    created_at_unix_ms,
                )
                .unwrap(),
            ),
            created_at_unix_ms,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Applied
        );
    }

    fn retention_request(retention_batch_id: Uuid, created_at_unix_ms: i64) -> RetentionRequest {
        RetentionRequest::new(retention_batch_id, Uuid::now_v7(), created_at_unix_ms).unwrap()
    }

    #[test]
    fn completed_active_receipt_prefix_compacts_to_a_verifiable_bridge_checkpoint() {
        let (_temporary, mut activated) = activate(100, 30);
        let batch_id = Uuid::now_v7();
        assert!(matches!(
            activated
                .repository
                .run_retention(&retention_request(batch_id, CREATED_AT + 1))
                .unwrap(),
            RetentionAck::Applied { .. }
        ));
        let project_uuid = activated.identity.project_uuid;
        let process_instance_id = activated.identity.process_instance_id;
        let experiment_id = Uuid::now_v7();
        let marker_id = Uuid::now_v7();
        let transaction = activated.repository.connection.transaction().unwrap();
        let mut predecessor = "00".repeat(32);
        for ordinal in 1..=ACTIVE_RETIREMENT_RECEIPT_MAX + 1 {
            let chain_tip = format!("{ordinal:064x}");
            transaction
                .execute(
                    "INSERT INTO active_retirement_receipts (
                        active_retirement_receipt_id, active_retirement_marker_id,
                        active_experiment_id, retention_batch_id,
                        predecessor_chain_hash, chain_tip_hash, cursor_json,
                        cursor_hash, parent_rows_deleted, child_rows_deleted,
                        bytes_deleted, completed, process_instance_id,
                        created_at_unix_ms, canonical_payload_hash
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, '{}', ?7, 0, 0, 0, ?8, ?9, ?10, ?6)",
                    params![
                        Uuid::now_v7().to_string(),
                        marker_id.to_string(),
                        experiment_id.to_string(),
                        batch_id.to_string(),
                        predecessor,
                        chain_tip,
                        "11".repeat(32),
                        i64::from(ordinal == ACTIVE_RETIREMENT_RECEIPT_MAX + 1),
                        process_instance_id.to_string(),
                        CREATED_AT + 2,
                    ],
                )
                .unwrap();
            predecessor = format!("{ordinal:064x}");
        }
        compact_active_retirement_history(&transaction, project_uuid, CREATED_AT + 3).unwrap();
        assert_eq!(
            transaction
                .query_row(
                    "SELECT count(*) FROM active_retirement_receipts",
                    [],
                    |row| { row.get::<_, i64>(0) }
                )
                .unwrap(),
            i64::try_from(ACTIVE_RETIREMENT_RECEIPT_MAX + 1 - ACTIVE_RETIREMENT_COMPACTION_COUNT)
                .unwrap()
        );
        let checkpoint = transaction
            .query_row(
                "SELECT first_receipt_ordinal, last_receipt_ordinal,
                        first_predecessor_hash, covered_chain_tip_hash,
                        receipt_count
                 FROM active_retirement_checkpoints",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, i64>(4)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(checkpoint.0, 1);
        assert_eq!(checkpoint.1, 512);
        assert_eq!(checkpoint.2, "00".repeat(32));
        assert_eq!(checkpoint.3, format!("{:064x}", 512));
        assert_eq!(checkpoint.4, 512);
        assert_eq!(
            transaction
                .query_row(
                    "SELECT predecessor_chain_hash FROM active_retirement_receipts
                     ORDER BY receipt_ordinal LIMIT 1",
                    [],
                    |row| row.get::<_, String>(0),
                )
                .unwrap(),
            checkpoint.3
        );
        transaction.commit().unwrap();
    }

    fn decision_selection(
        suffix: u32,
        created_at_unix_ms: i64,
        source_forced: bool,
        summary_count: usize,
        neighbor_count: usize,
        aggregate_size_bytes: usize,
    ) -> DecisionRetentionSelection {
        let decision_id =
            Uuid::parse_str(&format!("018f0000-0000-7000-8000-{suffix:012x}")).unwrap();
        DecisionRetentionSelection {
            decision_id,
            created_at_unix_ms,
            age_expired: !source_forced,
            count_excess: false,
            source_forced,
            summary_count,
            neighbor_count,
            aggregate_size_bytes,
        }
    }

    #[test]
    fn source_forced_decisions_precede_ordinary_candidates_deterministically() {
        let mut candidates = vec![
            decision_selection(3, 30, false, 1, 0, 1),
            decision_selection(2, 20, true, 1, 0, 1),
            decision_selection(1, 10, true, 1, 0, 1),
            decision_selection(4, 5, false, 1, 0, 1),
        ];

        prioritize_decision_retention_candidates(&mut candidates);

        assert_eq!(
            candidates
                .iter()
                .map(|candidate| (candidate.source_forced, candidate.created_at_unix_ms))
                .collect::<Vec<_>>(),
            vec![(true, 10), (true, 20), (false, 5), (false, 30)]
        );
    }

    #[test]
    fn cumulative_decision_limits_admit_exact_maximum_and_stop_before_overflow() {
        let maximum = decision_selection(1, 1, true, 64, 4_095, DECISION_RETENTION_BYTES_LIMIT);
        assert_eq!(
            next_decision_retention_totals(0, 0, 0, &maximum).unwrap(),
            Some((
                1,
                DECISION_CHILD_RETENTION_LIMIT,
                DECISION_RETENTION_BYTES_LIMIT
            ))
        );
        let smallest = decision_selection(2, 2, true, 1, 0, 1);
        assert_eq!(
            next_decision_retention_totals(
                1,
                DECISION_CHILD_RETENTION_LIMIT,
                DECISION_RETENTION_BYTES_LIMIT,
                &smallest,
            )
            .unwrap(),
            None
        );
        assert_eq!(
            next_decision_retention_totals(
                RETENTION_BATCH_LIMIT,
                RETENTION_BATCH_LIMIT,
                RETENTION_BATCH_LIMIT,
                &smallest,
            )
            .unwrap(),
            None
        );
    }

    #[test]
    fn zero_neighbor_decision_count_retention_is_whole_and_preserves_live_authority() {
        let (_temporary, config, mut activated) = decision_test_fixtures::activate(0.0, 2);
        let first = decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        let first_id = first.parent.decision_id;
        let first_bytes = first.parent.aggregate_size_bytes;
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&first, 2, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::Applied
        );
        let second =
            decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        let second_id = second.parent.decision_id;
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&second, 2, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::Applied
        );
        let expected_deleted = first_id.min(second_id);
        let expected_retained = first_id.max(second_id);
        let request = retention_request(Uuid::now_v7(), 103);

        assert!(matches!(
            activated.repository.run_retention(&request).unwrap(),
            RetentionAck::Applied { .. }
        ));

        assert!(
            super::super::decision::load_decision_graph(
                &activated.repository.connection,
                expected_deleted,
            )
            .unwrap()
            .is_none()
        );
        assert!(
            super::super::decision::load_decision_graph(
                &activated.repository.connection,
                expected_retained,
            )
            .unwrap()
            .is_some()
        );
        let receipt = load_decision_retention_receipt(
            &activated.repository.connection,
            request.retention_batch_id,
        )
        .unwrap()
        .and_then(verify_stored_decision_retention_receipt)
        .unwrap();
        assert_eq!(receipt.deleted_decision_count, 1);
        assert_eq!(receipt.deleted_summary_count, 1);
        assert_eq!(receipt.deleted_neighbor_count, 0);
        assert_eq!(receipt.deleted_child_count, 1);
        assert_eq!(
            receipt.verified_aggregate_bytes,
            to_u64(first_bytes).unwrap()
        );
        assert!(receipt.decision_count_excess);
        assert!(!receipt.source_forced);
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM canonical_routing_queries"
            ),
            1
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM pool_vector_space_mappings"
            ),
            1
        );
    }

    fn scalar(connection: &Connection, sql: &str) -> i64 {
        connection.query_row(sql, [], |row| row.get(0)).unwrap()
    }

    fn sqlite_sequence_snapshot(connection: &Connection) -> Vec<(String, i64)> {
        connection
            .prepare("SELECT name, seq FROM sqlite_sequence ORDER BY name")
            .unwrap()
            .query_map([], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap()
    }

    fn seed_vector_space(
        connection: &Connection,
        project_uuid: Uuid,
        vector_space_id: &str,
        created_at_unix_ms: i64,
    ) {
        connection
            .execute(
                "INSERT OR IGNORE INTO embedder_profiles (
                    embedder_profile_version_id, profile_id, protocol, endpoint_url,
                    endpoint_identity_sha256, model, provider_revision, dimensions,
                    credential_env_name_sha256, timeout_ms, max_in_flight, batch_size,
                    egress_class, canonical_profile_json, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (
                    ?1, 'retention-test-profile', 'openai-embeddings-v1',
                    'http://127.0.0.1/v1/embeddings', ?2, 'retention-test-model',
                    'retention-test-revision', 2, NULL, 1000, 1, 1,
                    'loopback_http', '{}', ?3, ?1
                 )",
                params!["f".repeat(64), "9".repeat(64), created_at_unix_ms,],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO vector_spaces (
                    vector_space_id, project_uuid, embedder_profile_version_id,
                    canonicalizer_version_id, canonicalizer_identity_json,
                    endpoint_identity_sha256, model, provider_revision,
                    dimensions, metric, normalization, canonical_space_json,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, ?2, ?3, ?4, '{}', ?5, 'retention-test-model',
                    'retention-test-revision', 2, 'cosine', 'l2_f32_v1', '{}', ?6, ?1
                 )",
                params![
                    vector_space_id,
                    project_uuid.to_string(),
                    "f".repeat(64),
                    "8".repeat(64),
                    "9".repeat(64),
                    created_at_unix_ms,
                ],
            )
            .unwrap();
    }

    fn seed_embedding(
        connection: &Connection,
        embedding_id: Uuid,
        vector_space_id: &str,
        canonical_query_hash: &str,
        content_hash: &str,
        created_at_unix_ms: i64,
    ) {
        connection
            .execute(
                "INSERT OR IGNORE INTO canonical_routing_queries (
                    canonical_query_hash, canonical_query_json, canonical_size_bytes,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, '{}', 2, ?2, ?1)",
                params![canonical_query_hash, created_at_unix_ms],
            )
            .unwrap();
        let payload_hash = hash_json(&json!({
            "embedding_id": embedding_id,
            "vector_space_id": vector_space_id,
            "canonical_query_hash": canonical_query_hash,
            "content_hash": content_hash,
            "created_at_unix_ms": created_at_unix_ms,
        }))
        .unwrap();
        connection
            .execute(
                "INSERT INTO embeddings (
                    embedding_id, vector_space_id, canonical_query_hash,
                    content_hash, dimensions, vector_blob, vector_checksum,
                    source, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, 2, ?5, ?6, 'provider', ?7, ?8)",
                params![
                    embedding_id.to_string(),
                    vector_space_id,
                    canonical_query_hash,
                    content_hash,
                    [1.0_f32.to_le_bytes(), 0.0_f32.to_le_bytes()].concat(),
                    "7".repeat(64),
                    created_at_unix_ms,
                    payload_hash,
                ],
            )
            .unwrap();
    }

    #[allow(clippy::too_many_arguments)]
    fn seed_vector_link(
        connection: &Connection,
        evidence_vector_link_id: Uuid,
        state_event_id: Uuid,
        shadow_attempt_id: Uuid,
        anchor_id: Uuid,
        root_uuid: Uuid,
        learning_generation_id: Uuid,
        vector_space_id: &str,
        embedding_id: Uuid,
        created_at_unix_ms: i64,
    ) {
        let canonical_query_hash = connection
            .query_row(
                "SELECT canonical_query_hash FROM embeddings WHERE embedding_id = ?1",
                params![embedding_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let (
            project_uuid,
            pool_id,
            tenant_policy_hash,
            agent_policy_hash,
            policy_version_id,
            api_family,
            transport_identity,
            anchor_model,
            anchor_revision,
            candidate_id,
            candidate_model,
            candidate_model_revision,
            decoding_fingerprint,
            evaluator_version,
        ) = connection
            .query_row(
                "SELECT project_uuid, pool_id, tenant_policy_hash, agent_policy_hash,
                        policy_version_id, api_family, transport_identity,
                        anchor_model, anchor_model_revision, candidate_id,
                        candidate_model, candidate_model_revision,
                        decoding_fingerprint, evaluator_version
                 FROM shadow_attempts WHERE shadow_attempt_id = ?1",
                params![shadow_attempt_id.to_string()],
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
                        row.get::<_, String>(10)?,
                        row.get::<_, String>(11)?,
                        row.get::<_, String>(12)?,
                        row.get::<_, String>(13)?,
                    ))
                },
            )
            .unwrap();
        let partition_hash = hash_json(&json!({
            "shadow_attempt_id": shadow_attempt_id,
            "vector_space_id": vector_space_id,
        }))
        .unwrap();
        connection
            .execute(
                "INSERT INTO routing_partitions (
                    partition_hash, canonical_partition_json, project_uuid, pool_id,
                    tenant_policy_hash, agent_policy_hash, policy_version_id,
                    learning_generation_id, api_family, transport_identity,
                    anchor_model, anchor_revision, candidate_id, candidate_model,
                    candidate_model_revision, decoding_fingerprint, evaluator_version,
                    vector_space_id, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, '{}', ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
                    ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?1
                 )",
                params![
                    partition_hash,
                    project_uuid,
                    pool_id,
                    tenant_policy_hash,
                    agent_policy_hash,
                    policy_version_id,
                    learning_generation_id.to_string(),
                    api_family,
                    transport_identity,
                    anchor_model,
                    anchor_revision,
                    candidate_id,
                    candidate_model,
                    candidate_model_revision,
                    decoding_fingerprint,
                    evaluator_version,
                    vector_space_id,
                    created_at_unix_ms,
                ],
            )
            .unwrap();
        let partition_id = connection.last_insert_rowid();
        let vectorization_outcome_id = hash_json(&json!({
            "shadow_attempt_id": shadow_attempt_id,
            "vector_space_id": vector_space_id,
            "canonical_query_hash": canonical_query_hash,
        }))
        .unwrap();
        connection
            .execute(
                "INSERT INTO vectorization_outcomes (
                    vectorization_outcome_id, shadow_attempt_id, anchor_id,
                    learning_generation_id, vector_space_id, canonical_query_hash,
                    outcome, stable_reason, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'canonicalized', NULL, ?7, ?1)",
                params![
                    vectorization_outcome_id,
                    shadow_attempt_id.to_string(),
                    anchor_id.to_string(),
                    learning_generation_id.to_string(),
                    vector_space_id,
                    canonical_query_hash,
                    created_at_unix_ms,
                ],
            )
            .unwrap();
        let link_hash = hash_json(&json!({
            "evidence_vector_link_id": evidence_vector_link_id,
            "shadow_attempt_id": shadow_attempt_id,
            "anchor_id": anchor_id,
            "root_uuid": root_uuid,
            "learning_generation_id": learning_generation_id,
            "vector_space_id": vector_space_id,
            "evaluation_id": Json::Null,
            "quality_label": Json::Null,
            "created_at_unix_ms": created_at_unix_ms,
        }))
        .unwrap();
        connection
            .execute(
                "INSERT INTO evidence_vector_links (
                    evidence_vector_link_id, vectorization_outcome_id,
                    shadow_attempt_id, anchor_id, root_uuid,
                    learning_generation_id, vector_space_id, partition_id,
                    canonical_query_hash, terminal_class, evaluation_id,
                    quality_label, created_at_unix_ms, canonical_payload_hash
                ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
                    'operational_failure', NULL, NULL, ?10, ?11
                )",
                params![
                    evidence_vector_link_id.to_string(),
                    vectorization_outcome_id,
                    shadow_attempt_id.to_string(),
                    anchor_id.to_string(),
                    root_uuid.to_string(),
                    learning_generation_id.to_string(),
                    vector_space_id,
                    partition_id,
                    canonical_query_hash,
                    created_at_unix_ms,
                    link_hash,
                ],
            )
            .unwrap();
        let state_hash = hash_json(&json!({
            "evidence_vector_link_state_event_id": state_event_id,
            "evidence_vector_link_id": evidence_vector_link_id,
            "embedding_id": embedding_id,
            "state": "ready",
            "created_at_unix_ms": created_at_unix_ms,
        }))
        .unwrap();
        connection
            .execute(
                "INSERT INTO evidence_vector_link_state_events (
                    evidence_vector_link_state_event_id, evidence_vector_link_id,
                    embedding_id, state, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, 'ready', ?4, ?5)",
                params![
                    state_event_id.to_string(),
                    evidence_vector_link_id.to_string(),
                    embedding_id.to_string(),
                    created_at_unix_ms,
                    state_hash,
                ],
            )
            .unwrap();
    }

    #[test]
    fn selection_orders_equal_close_ties_unions_reasons_and_caps_at_one_thousand() {
        let closed_at_unix_ms = 500;
        let mut anchor_ids = (0..1_002).map(|_| Uuid::now_v7()).collect::<Vec<_>>();
        anchor_ids.sort_unstable();
        let eligible = anchor_ids
            .iter()
            .rev()
            .map(|anchor_id| TerminalAnchor {
                anchor_id: *anchor_id,
                closed_at_unix_ms,
            })
            .collect();

        let selection = select_retention_anchors(eligible, 1_002, 1_001, Some(500));
        assert_eq!(selection.len(), RETENTION_BATCH_LIMIT);
        assert_eq!(
            selection
                .iter()
                .map(|selected| selected.anchor_id)
                .collect::<Vec<_>>(),
            anchor_ids[..RETENTION_BATCH_LIMIT]
        );
        assert!(selection[0].age_expired && selection[0].count_excess);
        assert!(selection[1].age_expired && selection[1].count_excess);
        assert!(
            selection[2..]
                .iter()
                .all(|selected| selected.age_expired && !selected.count_excess)
        );
        assert!(!selection.iter().any(|selected| {
            selected.anchor_id == anchor_ids[RETENTION_BATCH_LIMIT]
                || selected.anchor_id == anchor_ids[RETENTION_BATCH_LIMIT + 1]
        }));
    }

    #[test]
    fn empty_retention_reports_and_drains_multiple_global_cleanup_pages() {
        let (_temporary, mut activated) = activate(1_000, 30);
        let base = pending(&activated.identity, Uuid::now_v7());
        let canonicalizer = config(Path::new("router.db"), "retention-cleanup-pages").pools[0]
            .canonicalizer
            .clone();
        let transaction = activated.repository.connection_mut().transaction().unwrap();
        for index in 0..=RETENTION_BATCH_LIMIT {
            let mut request = base.request_projection.clone();
            request.normalized_request.messages = vec![SanitizedMessage::User {
                content: SanitizedMessageContent::Text(format!("route request {index}")),
                name: None,
            }];
            request.semantic_request_fingerprint =
                canonical_hash_without_field(&request, "semantic_request_fingerprint");
            let query = build_canonical_routing_query(
                &request,
                &base.routing_context_projection,
                &canonicalizer,
            )
            .unwrap();
            assert!(matches!(
                ensure_canonical_query(&transaction, &query, CREATED_AT).unwrap(),
                CanonicalQueryEnsureAck::Applied(_)
            ));
        }
        transaction.commit().unwrap();

        let RetentionAck::Applied {
            summary: first,
            observation: first_observation,
        } = activated
            .repository
            .run_retention(&retention_request(Uuid::now_v7(), CREATED_AT + 10))
            .unwrap()
        else {
            panic!("first global cleanup page should apply");
        };
        assert_eq!(first.selected_count, 0);
        assert!(first_observation.more_cleanup);
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM canonical_routing_queries"
            ),
            1
        );

        let RetentionAck::Applied {
            summary: second,
            observation: second_observation,
        } = activated
            .repository
            .run_retention(&retention_request(Uuid::now_v7(), CREATED_AT + 11))
            .unwrap()
        else {
            panic!("second global cleanup page should apply");
        };
        assert_eq!(second.selected_count, 0);
        assert!(!second_observation.more_cleanup);
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM canonical_routing_queries"
            ),
            0
        );
    }

    #[test]
    fn unfinished_attempt_preserves_exact_mapping_without_vectorization_outcome() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut router_config = config(&path, "retention-unfinished-mapping");
        router_config.pools[0].learning = Some(LearningConfig::minimal("embedder-a"));
        let mut activated = LedgerRepository::activate_at(&router_config, CREATED_AT).unwrap();
        let (anchor_id, pending) = seed_closed_anchor_without_batch(&mut activated, CREATED_AT + 1);
        let (_attempt, _reservation) =
            reserve_open_batch(&mut activated, anchor_id, &pending, CREATED_AT + 3);
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM vectorization_outcomes"
            ),
            0
        );
        let vector_space_id = activated.identity.pools["pool-a"]
            .vector_space
            .as_ref()
            .unwrap()
            .vector_space_id
            .as_str()
            .to_string();
        assert_eq!(
            activated
                .repository
                .stop_process(
                    ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), CREATED_AT + 5).unwrap(),
                )
                .unwrap(),
            super::super::process::ProcessCommandAck::Applied
        );
        let mut captured = CapturedVectorReferences::default();
        captured.vector_space_ids.insert(vector_space_id);
        let project_uuid = activated.identity.project_uuid;
        let transaction = activated.repository.connection_mut().transaction().unwrap();
        delete_captured_historical_vector_authority(
            &transaction,
            project_uuid,
            CREATED_AT + 6,
            &captured,
        )
        .unwrap();
        transaction.commit().unwrap();
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM pool_vector_space_mappings"
            ),
            1
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM vector_spaces"
            ),
            1
        );
    }

    #[test]
    fn unclaimed_building_generation_preserves_mapping_and_space() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut router_config = config(&path, "retention-building-mapping");
        router_config.pools[0].learning = Some(LearningConfig::minimal("embedder-a"));
        let mut activated = LedgerRepository::activate_at(&router_config, CREATED_AT).unwrap();
        let vector_space_id = activated.identity.pools["pool-a"]
            .vector_space
            .as_ref()
            .unwrap()
            .vector_space_id
            .clone();
        let dimensions = activated
            .repository
            .connection
            .query_row(
                "SELECT dimensions FROM vector_spaces WHERE vector_space_id = ?1",
                params![vector_space_id.as_str()],
                |row| row.get::<_, u32>(0),
            )
            .unwrap();
        let transaction = activated.repository.connection_mut().transaction().unwrap();
        assert!(matches!(
            crate::ledger::repository::vector_index::authorize_generation(
                &transaction,
                &vector_space_id,
                crate::vector::VectorDimensions::new(dimensions).unwrap(),
                CREATED_AT + 1,
            )
            .unwrap(),
            crate::ledger::repository::vector_index::GenerationAuthorizationAck::Created(_)
        ));
        transaction.commit().unwrap();
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM vector_index_rebuild_leases"
            ),
            0
        );
        assert_eq!(
            activated
                .repository
                .stop_process(
                    ProcessStop::new(Uuid::now_v7(), Uuid::now_v7(), CREATED_AT + 2).unwrap(),
                )
                .unwrap(),
            super::super::process::ProcessCommandAck::Applied
        );
        let mut captured = CapturedVectorReferences::default();
        captured
            .vector_space_ids
            .insert(vector_space_id.as_str().to_string());
        let project_uuid = activated.identity.project_uuid;
        let transaction = activated.repository.connection_mut().transaction().unwrap();
        delete_captured_historical_vector_authority(
            &transaction,
            project_uuid,
            CREATED_AT + 3,
            &captured,
        )
        .unwrap();
        transaction.commit().unwrap();
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM pool_vector_space_mappings"
            ),
            1
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM vector_spaces"
            ),
            1
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM vector_index_manifest WHERE state = 'building'"
            ),
            1
        );
    }

    #[test]
    fn stored_summary_rejects_rehashed_equal_batch_and_conflict_ids() {
        let retention_batch_id = Uuid::now_v7();
        let request = retention_request(retention_batch_id, CREATED_AT);
        let mut summary = build_summary(&request, Uuid::now_v7(), Uuid::now_v7(), &[]).unwrap();
        summary.conflict_health_event_id = retention_batch_id;
        summary.canonical_payload_hash = retention_summary_hash(&summary).unwrap();
        let stored = StoredRetentionSummary {
            retention_batch_id: summary.retention_batch_id.to_string(),
            conflict_health_event_id: Some(summary.conflict_health_event_id.to_string()),
            summary_shape_version: summary.summary_shape_version,
            project_uuid: summary.project_uuid.to_string(),
            process_instance_id: summary.process_instance_id.to_string(),
            age_expired: i64::from(summary.age_expired),
            count_excess: i64::from(summary.count_excess),
            selected_count: i64::try_from(summary.selected_count).unwrap(),
            selection_lower_bound_unix_ms: summary.selection_lower_bound_unix_ms,
            selection_upper_bound_unix_ms: summary.selection_upper_bound_unix_ms,
            selection_hash: summary.selection_hash,
            created_at_unix_ms: summary.created_at_unix_ms,
            canonical_payload_hash: summary.canonical_payload_hash,
        };
        assert!(verify_stored_summary(stored).is_none());
    }

    #[test]
    fn pending_closed_undelivered_and_open_batch_aggregates_are_ineligible() {
        let (_temporary, mut activated) = activate(2, 30);
        let pending_only = pending(&activated.identity, Uuid::now_v7());
        let frozen_pending =
            FrozenPendingAnchorV1::new(&pending_only, Uuid::now_v7(), Uuid::now_v7(), CREATED_AT)
                .unwrap();
        assert!(matches!(
            activated
                .repository
                .record_pending_anchor(&frozen_pending)
                .unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
        seed_closed_anchor_without_batch(&mut activated, CREATED_AT + 2);
        let (open_anchor_id, open_pending) =
            seed_closed_anchor_without_batch(&mut activated, CREATED_AT + 4);
        let _ = reserve_open_batch(
            &mut activated,
            open_anchor_id,
            &open_pending,
            CREATED_AT + 6,
        );
        let (started_anchor_id, started_pending) =
            seed_closed_anchor_without_batch(&mut activated, CREATED_AT + 7);
        let (started_attempt, _) = reserve_open_batch(
            &mut activated,
            started_anchor_id,
            &started_pending,
            CREATED_AT + 8,
        );
        assert_eq!(
            activated
                .repository
                .start_shadow_attempt(
                    ShadowAttemptStarted::new(
                        started_attempt.shadow_attempt_id,
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                        CREATED_AT + 9,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ShadowCommandAck::Applied
        );

        let RetentionAck::Applied {
            summary,
            observation,
        } = activated
            .repository
            .run_retention(&retention_request(Uuid::now_v7(), CREATED_AT + 10))
            .unwrap()
        else {
            panic!("valid ineligible aggregates should produce an empty receipt");
        };
        assert_eq!(summary.selected_count, 0);
        assert_eq!(observation.terminal_anchor_count, 3);
        assert!(!observation.capacity_available);
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchors"
            ),
            4
        );
    }

    #[test]
    fn terminal_attempt_with_admitted_dependency_is_ineligible_not_corrupt() {
        let (_temporary, mut activated) = activate(1, 30);
        let (anchor_id, pending) = seed_closed_anchor_without_batch(&mut activated, CREATED_AT);
        let (attempt, reservation) =
            reserve_open_batch(&mut activated, anchor_id, &pending, CREATED_AT + 2);
        let dependency = CandidateDependencyIdentity::new(
            LlmApiFamily::OpenAIChatCompletions,
            "fixture",
            "candidate-model-a",
            "2026-06-01",
        )
        .unwrap();
        let operation = DependencyOperation::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            anchor_id,
            attempt.shadow_attempt_id,
            10,
            300,
            CREATED_AT + 3,
        )
        .unwrap();
        assert!(matches!(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &operation)
                .unwrap(),
            DependencyCommandAck::Applied(snapshot)
                if snapshot.transition == DependencyTransition::Admitted
        ));
        terminalize_attempt(&mut activated, &attempt, &reservation, CREATED_AT + 5);

        let RetentionAck::Applied { summary, .. } = activated
            .repository
            .run_retention(&retention_request(Uuid::now_v7(), CREATED_AT + 6))
            .unwrap()
        else {
            panic!("admitted-only dependency should make evidence ineligible");
        };
        assert_eq!(summary.selected_count, 0);
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchors"
            ),
            1
        );
    }

    #[test]
    fn terminal_attempt_with_started_judge_is_ineligible_not_corrupt() {
        let (_temporary, mut activated) = activate(1, 30);
        let (anchor_id, pending) = seed_closed_anchor_without_batch(&mut activated, CREATED_AT);
        let (attempt, reservation) =
            reserve_open_batch(&mut activated, anchor_id, &pending, CREATED_AT + 2);
        terminalize_attempt(&mut activated, &attempt, &reservation, CREATED_AT + 5);
        let judge_attempt_id = Uuid::now_v7();
        activated
            .repository
            .connection
            .execute(
                "INSERT INTO judge_attempts (
                    judge_attempt_id, shadow_attempt_id, process_instance_id,
                    learning_generation_id, evaluator_version, judge_input_sha256,
                    candidate_response_json, candidate_response_fingerprint,
                    judge_model, judge_model_revision, prompt_version, prompt_sha256,
                    rubric_version, rubric_sha256, output_schema_version,
                    output_schema_sha256, attempt_ordinal, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, '{}', ?7, 'judge-model',
                           '2026-07-01', 'prompt-v1', ?8, 'rubric-v1', ?9, 1,
                           ?10, 0, ?11, ?12)",
                params![
                    judge_attempt_id.to_string(),
                    attempt.shadow_attempt_id.to_string(),
                    activated.identity.process_instance_id.to_string(),
                    activated.identity.pools["pool-a"]
                        .learning_generation_id
                        .to_string(),
                    attempt.evaluator_version,
                    "0".repeat(64),
                    "1".repeat(64),
                    "2".repeat(64),
                    "3".repeat(64),
                    "4".repeat(64),
                    CREATED_AT + 3,
                    "5".repeat(64),
                ],
            )
            .unwrap();
        activated
            .repository
            .connection
            .execute(
                "INSERT INTO judge_attempt_state_events (
                    judge_attempt_state_event_id, judge_attempt_id,
                    process_instance_id, state, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, 'started', ?4, ?5)",
                params![
                    Uuid::now_v7().to_string(),
                    judge_attempt_id.to_string(),
                    activated.identity.process_instance_id.to_string(),
                    CREATED_AT + 3,
                    "6".repeat(64),
                ],
            )
            .unwrap();

        let RetentionAck::Applied { summary, .. } = activated
            .repository
            .run_retention(&retention_request(Uuid::now_v7(), CREATED_AT + 6))
            .unwrap()
        else {
            panic!("started Judge should make evidence ineligible");
        };
        assert_eq!(summary.selected_count, 0);
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchors"
            ),
            1
        );
    }

    #[test]
    fn malformed_anchor_state_and_result_halves_are_corruption() {
        for missing_result in [true, false] {
            let (_temporary, mut activated) = activate(1, 30);
            let anchor_id = seed_queue_full_anchor(&mut activated, CREATED_AT);
            let table = if missing_result {
                "anchor_results"
            } else {
                "anchor_state_events"
            };
            let predicate = if missing_result {
                "anchor_id = ?1"
            } else {
                "anchor_id = ?1 AND state = 'pending'"
            };
            activated
                .repository
                .connection
                .execute(
                    &format!("DELETE FROM {table} WHERE {predicate}"),
                    params![anchor_id.to_string()],
                )
                .unwrap();
            let before = recovery_snapshot(&activated.repository.connection);

            let error = activated
                .repository
                .run_retention(&retention_request(Uuid::now_v7(), CREATED_AT + 10))
                .unwrap_err();
            assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);
            assert_eq!(recovery_snapshot(&activated.repository.connection), before);
            assert_eq!(
                scalar(
                    &activated.repository.connection,
                    "SELECT count(*) FROM retention_batches"
                ),
                0
            );
        }
    }

    #[test]
    fn shadow_and_batch_aggregate_halves_are_corruption() {
        for corruption in [
            "shadow_result_missing",
            "shadow_terminal_state_missing",
            "terminal_batch_open_attempt",
            "open_batch_terminal_attempt",
            "reserved_candidate_mismatch",
        ] {
            let (_temporary, mut activated) = activate(1, 30);
            let (anchor_id, pending) = seed_closed_anchor_without_batch(&mut activated, CREATED_AT);
            let (attempt, reservation) =
                reserve_open_batch(&mut activated, anchor_id, &pending, CREATED_AT + 2);
            match corruption {
                "terminal_batch_open_attempt" => {
                    activated
                        .repository
                        .connection
                        .execute(
                            "INSERT INTO sample_batch_state_events (
                                sample_batch_state_event_id, sample_batch_id,
                                process_instance_id, state, created_at_unix_ms,
                                canonical_payload_hash
                             ) VALUES (?1, ?2, ?3, 'closed', ?4, ?5)",
                            params![
                                Uuid::now_v7().to_string(),
                                reservation.sample_batch_id.to_string(),
                                activated.identity.process_instance_id.to_string(),
                                CREATED_AT + 5,
                                "a".repeat(64),
                            ],
                        )
                        .unwrap();
                }
                "shadow_result_missing" => {
                    terminalize_attempt(&mut activated, &attempt, &reservation, CREATED_AT + 5);
                    activated
                        .repository
                        .connection
                        .execute(
                            "DELETE FROM shadow_results WHERE shadow_attempt_id = ?1",
                            params![attempt.shadow_attempt_id.to_string()],
                        )
                        .unwrap();
                }
                "shadow_terminal_state_missing" => {
                    terminalize_attempt(&mut activated, &attempt, &reservation, CREATED_AT + 5);
                    activated
                        .repository
                        .connection
                        .execute(
                            "DELETE FROM shadow_attempt_state_events
                             WHERE shadow_attempt_id = ?1
                               AND state NOT IN ('reserved', 'started')",
                            params![attempt.shadow_attempt_id.to_string()],
                        )
                        .unwrap();
                }
                "open_batch_terminal_attempt" => {
                    terminalize_attempt(&mut activated, &attempt, &reservation, CREATED_AT + 5);
                    activated
                        .repository
                        .connection
                        .execute(
                            "DELETE FROM sample_batch_state_events
                             WHERE sample_batch_id = ?1 AND state <> 'open'",
                            params![reservation.sample_batch_id.to_string()],
                        )
                        .unwrap();
                }
                "reserved_candidate_mismatch" => {
                    terminalize_attempt(&mut activated, &attempt, &reservation, CREATED_AT + 5);
                    activated
                        .repository
                        .connection
                        .execute(
                            "UPDATE sample_batches SET reserved_candidate_count = 2
                             WHERE sample_batch_id = ?1",
                            params![reservation.sample_batch_id.to_string()],
                        )
                        .unwrap();
                }
                _ => unreachable!(),
            }
            let before = recovery_snapshot(&activated.repository.connection);

            let error = activated
                .repository
                .run_retention(&retention_request(Uuid::now_v7(), CREATED_AT + 10))
                .unwrap_err();
            assert_eq!(
                error.class(),
                LedgerErrorClass::IdentityInvariant,
                "{corruption}"
            );
            assert_eq!(
                recovery_snapshot(&activated.repository.connection),
                before,
                "{corruption}"
            );
            assert_eq!(
                scalar(
                    &activated.repository.connection,
                    "SELECT count(*) FROM retention_batches"
                ),
                0,
                "{corruption}"
            );
        }
    }

    #[test]
    fn count_pressure_targets_max_minus_one_and_retries_exactly() {
        let (_temporary, mut activated) = activate(2, 30);
        let oldest = seed_queue_full_anchor(&mut activated, CREATED_AT);
        let newest = seed_queue_full_anchor(&mut activated, CREATED_AT + 10);
        let retention_batch_id = Uuid::now_v7();
        let request = retention_request(retention_batch_id, CREATED_AT + 20);

        let RetentionAck::Applied {
            summary,
            observation,
        } = activated.repository.run_retention(&request).unwrap()
        else {
            panic!("retention should apply");
        };
        assert_eq!(
            summary.summary_shape_version,
            RETENTION_SUMMARY_SHAPE_VERSION
        );
        assert_eq!(summary.selected_count, 1);
        assert!(!summary.age_expired);
        assert!(summary.count_excess);
        assert_eq!(observation.terminal_anchor_count, 1);
        assert!(observation.capacity_available);
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("SELECT anchor_id FROM anchors", [], |row| {
                    row.get::<_, String>(0)
                })
                .unwrap(),
            newest.to_string()
        );
        assert_ne!(oldest, newest);

        assert_eq!(
            activated.repository.run_retention(&request).unwrap(),
            RetentionAck::AlreadyApplied {
                summary: summary.clone(),
                observation,
            }
        );
        let conflicting = retention_request(retention_batch_id, CREATED_AT + 20);
        assert_eq!(
            activated.repository.run_retention(&conflicting).unwrap(),
            RetentionAck::Conflict
        );
        assert_eq!(
            activated.repository.run_retention(&conflicting).unwrap(),
            RetentionAck::Conflict
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM retention_batches"
            ),
            1
        );
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
    fn synthetic_vector_rows_without_canonical_authority_block_retention() {
        let (_temporary, mut activated) = activate(2, 30);
        let (selected_anchor_id, selected_pending) =
            seed_closed_anchor_without_batch(&mut activated, CREATED_AT);
        let (selected_attempt, selected_reservation) = reserve_open_batch(
            &mut activated,
            selected_anchor_id,
            &selected_pending,
            CREATED_AT + 2,
        );
        terminalize_attempt(
            &mut activated,
            &selected_attempt,
            &selected_reservation,
            CREATED_AT + 5,
        );
        let (retained_anchor_id, retained_pending) =
            seed_closed_anchor_without_batch(&mut activated, CREATED_AT + 10);
        let (retained_attempt, _) = reserve_open_batch(
            &mut activated,
            retained_anchor_id,
            &retained_pending,
            CREATED_AT + 12,
        );

        let selected_root_uuid = activated
            .repository
            .connection
            .query_row(
                "SELECT root_uuid FROM anchors WHERE anchor_id = ?1",
                params![selected_anchor_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let retained_root_uuid = activated
            .repository
            .connection
            .query_row(
                "SELECT root_uuid FROM anchors WHERE anchor_id = ?1",
                params![retained_anchor_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let selected_root_uuid = Uuid::parse_str(&selected_root_uuid).unwrap();
        let retained_root_uuid = Uuid::parse_str(&retained_root_uuid).unwrap();
        let captured_space = "a".repeat(64);
        let shared_space = "b".repeat(64);
        let captured_query = "c".repeat(64);
        let shared_query = "d".repeat(64);
        let content_hash = "e".repeat(64);
        seed_vector_space(
            &activated.repository.connection,
            activated.identity.project_uuid,
            &captured_space,
            CREATED_AT,
        );
        seed_vector_space(
            &activated.repository.connection,
            activated.identity.project_uuid,
            &shared_space,
            CREATED_AT,
        );
        let captured_embedding_id = Uuid::now_v7();
        let shared_embedding_id = Uuid::now_v7();
        seed_embedding(
            &activated.repository.connection,
            captured_embedding_id,
            &captured_space,
            &captured_query,
            &content_hash,
            CREATED_AT,
        );
        seed_embedding(
            &activated.repository.connection,
            shared_embedding_id,
            &shared_space,
            &shared_query,
            &content_hash,
            CREATED_AT,
        );
        let learning_generation_id = activated.identity.pools["pool-a"].learning_generation_id;
        seed_vector_link(
            &activated.repository.connection,
            Uuid::now_v7(),
            Uuid::now_v7(),
            selected_attempt.shadow_attempt_id,
            selected_anchor_id,
            selected_root_uuid,
            learning_generation_id,
            &captured_space,
            captured_embedding_id,
            CREATED_AT + 6,
        );
        seed_vector_link(
            &activated.repository.connection,
            Uuid::now_v7(),
            Uuid::now_v7(),
            selected_attempt.shadow_attempt_id,
            selected_anchor_id,
            selected_root_uuid,
            learning_generation_id,
            &shared_space,
            shared_embedding_id,
            CREATED_AT + 6,
        );
        seed_vector_link(
            &activated.repository.connection,
            Uuid::now_v7(),
            Uuid::now_v7(),
            retained_attempt.shadow_attempt_id,
            retained_anchor_id,
            retained_root_uuid,
            learning_generation_id,
            &shared_space,
            shared_embedding_id,
            CREATED_AT + 13,
        );
        append_integrity_health(
            &activated.repository.connection,
            Uuid::now_v7(),
            activated.identity.project_uuid,
            activated.identity.process_instance_id,
            Some(selected_anchor_id),
            None,
            CREATED_AT + 14,
        )
        .unwrap();
        append_integrity_health(
            &activated.repository.connection,
            Uuid::now_v7(),
            activated.identity.project_uuid,
            activated.identity.process_instance_id,
            None,
            None,
            CREATED_AT + 14,
        )
        .unwrap();

        let error = activated
            .repository
            .run_retention(&retention_request(Uuid::now_v7(), CREATED_AT + 20))
            .unwrap_err();
        assert_eq!(error.class(), LedgerErrorClass::CorruptDatabase);
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchors"
            ),
            2
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM retention_batches"
            ),
            0
        );
    }

    #[test]
    fn age_cutoff_is_inclusive_and_empty_receipts_are_durable() {
        let (_temporary, mut activated) = activate(1_000, 1);
        let anchor_id = seed_queue_full_anchor(&mut activated, CREATED_AT);
        let closed_at_unix_ms = CREATED_AT + 1;
        let just_before = closed_at_unix_ms + MILLIS_PER_DAY - 1;
        activated
            .repository
            .renew_heartbeat(HeartbeatRenewal::new(just_before - 1).unwrap())
            .unwrap();

        let empty_request = retention_request(Uuid::now_v7(), just_before);
        let RetentionAck::Applied {
            summary: empty_summary,
            observation: empty_observation,
        } = activated.repository.run_retention(&empty_request).unwrap()
        else {
            panic!("empty retention should still commit a receipt");
        };
        assert_eq!(empty_summary.selected_count, 0);
        assert!(!empty_summary.age_expired);
        assert!(!empty_summary.count_excess);
        assert_eq!(empty_summary.selection_hash, selection_hash(&[]).unwrap());
        assert_eq!(empty_summary.selection_lower_bound_unix_ms, None);
        assert_eq!(empty_summary.selection_upper_bound_unix_ms, None);
        assert_eq!(empty_observation.terminal_anchor_count, 1);
        assert!(empty_observation.capacity_available);

        let exact_cutoff = retention_request(Uuid::now_v7(), just_before + 1);
        let RetentionAck::Applied { summary, .. } =
            activated.repository.run_retention(&exact_cutoff).unwrap()
        else {
            panic!("inclusive cutoff should retain the anchor");
        };
        assert_eq!(summary.selected_count, 1);
        assert!(summary.age_expired);
        assert!(!summary.count_excess);
        assert_eq!(
            summary.selection_lower_bound_unix_ms,
            Some(closed_at_unix_ms)
        );
        assert_eq!(
            summary.selection_upper_bound_unix_ms,
            Some(closed_at_unix_ms)
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchors"
            ),
            0
        );
        assert_eq!(
            scalar(
                &activated.repository.connection,
                "SELECT count(*) FROM anchor_results"
            ),
            0
        );
        assert_eq!(
            activated.repository.run_retention(&empty_request).unwrap(),
            RetentionAck::AlreadyApplied {
                summary: empty_summary,
                observation: RetentionCapacityObservation {
                    terminal_anchor_count: 0,
                    capacity_available: true,
                    more_cleanup: false,
                },
            }
        );
        assert_ne!(anchor_id, Uuid::nil());
    }

    #[test]
    fn every_fault_point_rolls_back_deletion_and_receipt() {
        let (_temporary, mut activated) = activate(1, 30);
        seed_queue_full_anchor(&mut activated, CREATED_AT);
        let request = retention_request(Uuid::now_v7(), CREATED_AT + 10);
        let before = recovery_snapshot(&activated.repository.connection);

        for fault_point in [
            RetentionFaultPoint::Selection,
            RetentionFaultPoint::CarryForward,
            RetentionFaultPoint::Deletion,
            RetentionFaultPoint::Summary,
        ] {
            let error = activated
                .repository
                .run_retention_with_fault(&request, fault_point)
                .unwrap_err();
            assert_eq!(error.class(), LedgerErrorClass::DatabaseOperationFailed);
            assert_eq!(recovery_snapshot(&activated.repository.connection), before);
        }

        assert!(matches!(
            activated.repository.run_retention(&request).unwrap(),
            RetentionAck::Applied { .. }
        ));
    }

    #[test]
    fn checkpoint_faults_restore_every_table_and_sqlite_sequence() {
        let (_temporary, mut activated) = activate(1, 30);
        let (anchor_id, pending) = seed_closed_anchor_without_batch(&mut activated, CREATED_AT);
        let (attempt, reservation) =
            reserve_open_batch(&mut activated, anchor_id, &pending, CREATED_AT + 2);
        let dependency = CandidateDependencyIdentity::new(
            LlmApiFamily::OpenAIChatCompletions,
            "fixture",
            "candidate-model-a",
            "2026-06-01",
        )
        .unwrap();
        let operation = DependencyOperation::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            anchor_id,
            attempt.shadow_attempt_id,
            10,
            300,
            CREATED_AT + 3,
        )
        .unwrap();
        assert!(matches!(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &operation)
                .unwrap(),
            DependencyCommandAck::Applied(snapshot)
                if snapshot.transition == DependencyTransition::Admitted
        ));
        let completion = DependencyCompletion::failure(
            operation.dependency_operation_id,
            Uuid::now_v7(),
            CREATED_AT + 4,
            DependencyFailureClass::new("router.provider.timeout").unwrap(),
        )
        .unwrap();
        assert!(matches!(
            activated
                .repository
                .complete_dependency(&completion)
                .unwrap(),
            DependencyCommandAck::Applied(snapshot)
                if snapshot.transition == DependencyTransition::Failure
        ));
        terminalize_attempt(&mut activated, &attempt, &reservation, CREATED_AT + 5);
        let request = retention_request(Uuid::now_v7(), CREATED_AT + 6);
        let before = recovery_snapshot(&activated.repository.connection);
        let sequence_before = sqlite_sequence_snapshot(&activated.repository.connection);

        for fault_point in [
            RetentionFaultPoint::Selection,
            RetentionFaultPoint::CarryForward,
            RetentionFaultPoint::Deletion,
            RetentionFaultPoint::Summary,
        ] {
            let error = activated
                .repository
                .run_retention_with_fault(&request, fault_point)
                .unwrap_err();
            assert_eq!(error.class(), LedgerErrorClass::DatabaseOperationFailed);
            assert_eq!(recovery_snapshot(&activated.repository.connection), before);
            assert_eq!(
                sqlite_sequence_snapshot(&activated.repository.connection),
                sequence_before
            );
            assert_eq!(
                activated
                    .repository
                    .connection
                    .query_row(
                        "SELECT count(*) FROM dependency_state_events
                         WHERE dependency_operation_id IS NULL AND anchor_id IS NULL",
                        [],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap(),
                0
            );
        }
    }

    #[test]
    fn two_wal_processes_serialize_retention_without_double_selection() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut config = config(&path, "retention-wal-serialization");
        config.max_evidence_records = 2;
        config.retention_days = 30;
        let mut first = LedgerRepository::activate_at(&config, CREATED_AT).unwrap();
        let second = LedgerRepository::activate_at(&config, CREATED_AT).unwrap();
        seed_queue_full_anchor(&mut first, CREATED_AT);
        seed_queue_full_anchor(&mut first, CREATED_AT + 10);
        let barrier = Arc::new(Barrier::new(3));
        let first_request = retention_request(Uuid::now_v7(), CREATED_AT + 20);
        let second_request = retention_request(Uuid::now_v7(), CREATED_AT + 20);
        let first_barrier = Arc::clone(&barrier);
        let second_barrier = Arc::clone(&barrier);
        let first_thread = thread::spawn(move || {
            let mut repository = first.repository;
            first_barrier.wait();
            let ack = repository.run_retention(&first_request).unwrap();
            (repository, ack)
        });
        let second_thread = thread::spawn(move || {
            let mut repository = second.repository;
            second_barrier.wait();
            let ack = repository.run_retention(&second_request).unwrap();
            (repository, ack)
        });
        barrier.wait();
        let (first_repository, first_ack) = first_thread.join().unwrap();
        let (_second_repository, second_ack) = second_thread.join().unwrap();
        let selected_counts = [first_ack, second_ack].map(|ack| match ack {
            RetentionAck::Applied { summary, .. } => summary.selected_count,
            other => panic!("both serialized requests must apply: {other:?}"),
        });
        assert_eq!(selected_counts.iter().sum::<u64>(), 1);
        assert!(selected_counts.contains(&0));
        assert!(selected_counts.contains(&1));
        assert_eq!(
            scalar(&first_repository.connection, "SELECT count(*) FROM anchors"),
            1
        );
        assert_eq!(
            scalar(
                &first_repository.connection,
                "SELECT count(*) FROM retention_batches"
            ),
            2
        );
        assert_eq!(
            first_repository
                .connection
                .query_row("PRAGMA journal_mode", [], |row| row.get::<_, String>(0))
                .unwrap(),
            "wal"
        );
    }

    #[test]
    fn held_wal_reader_keeps_its_snapshot_across_retention_commit() {
        let (temporary, mut activated) = activate(1, 30);
        seed_queue_full_anchor(&mut activated, CREATED_AT);
        let reader = Connection::open(database_path(&temporary)).unwrap();
        reader
            .execute_batch("PRAGMA query_only = ON; BEGIN DEFERRED")
            .unwrap();
        assert_eq!(scalar(&reader, "SELECT count(*) FROM anchors"), 1);

        let RetentionAck::Applied { summary, .. } = activated
            .repository
            .run_retention(&retention_request(Uuid::now_v7(), CREATED_AT + 10))
            .unwrap()
        else {
            panic!("retention should commit while a WAL reader holds a snapshot");
        };
        assert_eq!(summary.selected_count, 1);
        assert_eq!(scalar(&reader, "SELECT count(*) FROM anchors"), 1);
        reader.execute_batch("COMMIT").unwrap();
        assert_eq!(scalar(&reader, "SELECT count(*) FROM anchors"), 0);
        assert_eq!(scalar(&reader, "SELECT count(*) FROM retention_batches"), 1);
    }

    #[test]
    fn busy_timeout_has_no_partial_retention_effect() {
        let (temporary, mut activated) = activate(1, 30);
        seed_queue_full_anchor(&mut activated, CREATED_AT);
        let request = retention_request(Uuid::now_v7(), CREATED_AT + 10);
        let before = recovery_snapshot(&activated.repository.connection);
        let sequence_before = sqlite_sequence_snapshot(&activated.repository.connection);
        assert_eq!(
            activated
                .repository
                .connection
                .query_row("PRAGMA busy_timeout", [], |row| row.get::<_, i64>(0))
                .unwrap(),
            5_000
        );
        let blocker = Connection::open(database_path(&temporary)).unwrap();
        blocker.execute_batch("BEGIN IMMEDIATE").unwrap();

        let started = Instant::now();
        let error = activated.repository.run_retention(&request).unwrap_err();
        let elapsed = started.elapsed();
        assert_eq!(error.class(), LedgerErrorClass::Busy);
        assert!(
            elapsed >= Duration::from_millis(4_500),
            "elapsed: {elapsed:?}"
        );
        assert!(elapsed < Duration::from_secs(8), "elapsed: {elapsed:?}");
        assert_eq!(recovery_snapshot(&activated.repository.connection), before);
        assert_eq!(
            sqlite_sequence_snapshot(&activated.repository.connection),
            sequence_before
        );
        blocker.execute_batch("ROLLBACK").unwrap();

        assert!(matches!(
            activated.repository.run_retention(&request).unwrap(),
            RetentionAck::Applied { .. }
        ));
    }

    #[test]
    fn transaction_start_refusal_has_no_durable_effect() {
        let (_temporary, mut activated) = activate(1, 30);
        seed_queue_full_anchor(&mut activated, CREATED_AT);
        let request = retention_request(Uuid::now_v7(), CREATED_AT + 10);
        let before = recovery_snapshot(&activated.repository.connection);

        assert_eq!(
            activated
                .repository
                .run_retention_with_start_check(&request, || None::<()>)
                .unwrap(),
            RetentionAck::TransactionNotStarted
        );
        assert_eq!(recovery_snapshot(&activated.repository.connection), before);
    }

    #[test]
    fn rehashed_canonical_corruption_blocks_retention_atomically() {
        let (_temporary, mut activated) = activate(1, 30);
        let anchor_id = seed_queue_full_anchor(&mut activated, CREATED_AT);
        let stored_json: String = activated
            .repository
            .connection
            .query_row(
                "SELECT normalized_response_json FROM anchor_results WHERE anchor_id = ?1",
                params![anchor_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        let mut response: Json = serde_json::from_str(&stored_json).unwrap();
        response["model"] = json!("tampered-anchor");
        let semantic_hash = {
            let mut semantic = response.clone();
            semantic
                .as_object_mut()
                .unwrap()
                .remove("semantic_response_fingerprint");
            hash_json(&semantic).unwrap()
        };
        response["semantic_response_fingerprint"] = json!(semantic_hash);
        let canonical_response = crate::canonical_json::canonical_json(&response).unwrap();
        let row_hash = hash_json(&response).unwrap();
        activated
            .repository
            .connection
            .execute(
                "UPDATE anchor_results
                 SET normalized_response_json = ?1,
                     semantic_response_fingerprint = ?2,
                     canonical_payload_hash = ?3
                 WHERE anchor_id = ?4",
                params![
                    canonical_response,
                    response["semantic_response_fingerprint"].as_str().unwrap(),
                    row_hash,
                    anchor_id.to_string(),
                ],
            )
            .unwrap();
        let before = recovery_snapshot(&activated.repository.connection);

        let error = activated
            .repository
            .run_retention(&retention_request(Uuid::now_v7(), CREATED_AT + 10))
            .unwrap_err();
        assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);
        assert_eq!(recovery_snapshot(&activated.repository.connection), before);
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
                "SELECT count(*) FROM retention_batches"
            ),
            0
        );
    }
}
