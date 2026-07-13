// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic persistence and replay verification for recommendation decision audits.

use std::collections::BTreeSet;

use rusqlite::{
    Connection, Error as SqliteError, ErrorCode, OptionalExtension, Row, Transaction,
    TransactionBehavior, named_params, params, types::Type,
};
use serde_json::Value as Json;
use uuid::{Uuid, Variant};

use super::process::{append_integrity_health, originating_process_is_live};
use super::vector_catalog::{
    CanonicalQueryEnsureAck, LiveRoutingPartitionResolution, ensure_canonical_query,
    load_embedding_cache, resolve_live_routing_partition,
};
use super::vector_registry::{FrozenMappingKey, resolve_frozen_mapping};
use super::vector_search::{ProjectedVectorNeighbor, search_projected_neighbors_in_transaction};
use super::{LedgerRepository, TransactionStartGuard, map_sqlite_error};
use crate::candidate_set::{CandidateSetMemberInputV1, build_candidate_set_from_members_v1};
use crate::canonical_json::canonical_json;
use crate::canonical_query::CanonicalRoutingQueryArtifactV1;
use crate::confidence::{
    AuditedNeighborV1, CandidateConfidenceInputV1, CandidateConfidenceReasonV1,
    CandidateConfidenceSummaryV1, CandidateEvidenceV1, ConfidenceBinaryLabelV1,
    ConfidenceDecisionReasonV1, ConfidenceEngineV1, ConfidenceEvaluationInputV1,
    ConfidenceEvaluationSourceV1, ConfidenceNeighborInputV1, ConfidencePolicyV1,
    ConfidenceTerminalClassV1, NeighborExclusionReasonV1,
};
use crate::decision_audit::{
    ACTIVE_DECISION_SHAPE_VERSION_V2, AuditF32V1, AuditF64V1, DECISION_SHAPE_VERSION_V1,
    DecisionAuditV1, DecisionBinaryLabelV1, DecisionCandidateReasonV1, DecisionCandidateSummaryV1,
    DecisionFinalReasonV1, DecisionNeighborExclusionReasonV1, DecisionNeighborV1, DecisionRecordV1,
    StoredDecisionGraphV1, VerifiedStoredDecisionGraphV1, verify_decision_parent_record,
};
use crate::fingerprint::sha256_hex;
use crate::judge::{JudgeBinaryLabelV1, JudgeEvaluationSourceV1};
use crate::ledger::fs::enforce_sidecar_permissions;
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::ledger::repository::shadow::ShadowTerminalClass;
use crate::vector::{PartitionId, VectorSpaceId};

/// Exhaustive result of one immutable decision-audit append attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecisionAuditAck {
    Applied,
    AlreadyApplied,
    RetentionRequired,
    AuthorityChanged,
    SourceRetiring,
    EvidenceChanged,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

impl LedgerRepository {
    #[cfg(test)]
    pub(crate) fn record_decision_audit(
        &mut self,
        audit: &DecisionAuditV1,
        max_evidence_records: u64,
        conflict_health_event_id: Uuid,
    ) -> Result<DecisionAuditAck, LedgerError> {
        self.record_decision_audit_with_start_check(
            audit,
            max_evidence_records,
            conflict_health_event_id,
            || Some(()),
        )
    }

    pub(crate) fn record_decision_audit_with_start_check<G: TransactionStartGuard>(
        &mut self,
        audit: &DecisionAuditV1,
        max_evidence_records: u64,
        conflict_health_event_id: Uuid,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<DecisionAuditAck, LedgerError> {
        validate_audit_command(audit)?;
        if max_evidence_records == 0 || !is_uuid_v7(conflict_health_event_id) {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let database_path = self.database_path.clone();
        let project_uuid = self.project_uuid;
        let writer_process_instance_id = self.process_instance_id;
        let active_policy_version = self
            .active_policy_versions
            .get(&audit.parent.pool_id)
            .cloned();
        enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(DecisionAuditAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(DecisionAuditAck::TransactionNotStarted);
            }
            Err(error) => {
                return Err(map_sqlite_error(
                    &error,
                    LedgerErrorClass::DatabaseOperationFailed,
                ));
            }
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(DecisionAuditAck::TransactionNotStarted);
        }
        drop(start_guard);

        let acknowledgement = append_decision_audit_in_transaction(
            &transaction,
            audit,
            project_uuid,
            active_policy_version.as_deref(),
            max_evidence_records,
        )?;
        if acknowledgement == DecisionAuditAck::Conflict {
            append_integrity_health(
                &transaction,
                conflict_health_event_id,
                project_uuid,
                writer_process_instance_id,
                None,
                None,
                audit.parent.created_at_unix_ms,
            )?;
        }
        enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        transaction
            .commit()
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed))?;
        Ok(acknowledgement)
    }
}

fn is_uuid_v7(value: Uuid) -> bool {
    value.get_version_num() == 7 && value.get_variant() == Variant::RFC4122
}

fn append_decision_audit_in_transaction(
    transaction: &Transaction<'_>,
    audit: &DecisionAuditV1,
    project_uuid: Uuid,
    active_policy_version: Option<&str>,
    max_evidence_records: u64,
) -> Result<DecisionAuditAck, LedgerError> {
    if audit.parent.decision_shape_version != DECISION_SHAPE_VERSION_V1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    if let Some(existing) = load_existing_decision(transaction, audit)? {
        return Ok(existing);
    }

    let decision_count = transaction
        .query_row(
            "SELECT count(*) FROM decisions WHERE project_uuid = ?1",
            [project_uuid.to_string()],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    if u64::try_from(decision_count).map_err(|_| corrupt())? >= max_evidence_records {
        return Ok(DecisionAuditAck::RetentionRequired);
    }

    if audit.parent.project_uuid != project_uuid
        || active_policy_version != Some(audit.parent.policy_version_id.as_str())
    {
        return Ok(DecisionAuditAck::AuthorityChanged);
    }
    if !originating_process_is_live(
        transaction,
        audit.parent.project_uuid,
        audit.parent.process_instance_id,
    )? {
        return Ok(DecisionAuditAck::OriginatingProcessNotLive);
    }
    if !originating_config_matches(transaction, audit, max_evidence_records)? {
        return Ok(DecisionAuditAck::AuthorityChanged);
    }

    let mapping = FrozenMappingKey::new(
        audit.parent.project_uuid,
        audit.parent.config_generation_id.clone(),
        audit.parent.pool_id.clone(),
        audit.parent.policy_version_id.clone(),
    )
    .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let Some(resolved_mapping) = resolve_frozen_mapping(transaction, &mapping)? else {
        return Ok(DecisionAuditAck::AuthorityChanged);
    };
    if resolved_mapping.mapping.vector_space_id.as_str() != audit.parent.vector_space_id
        || !candidate_policy_matches(transaction, audit)?
    {
        return Ok(DecisionAuditAck::AuthorityChanged);
    }

    for (summary, artifact) in audit.summaries.iter().zip(&audit.partition_artifacts) {
        match resolve_live_routing_partition(transaction, &mapping, artifact)? {
            LiveRoutingPartitionResolution::AuthorityNotFound
            | LiveRoutingPartitionResolution::StaleLearningGeneration => {
                return Ok(DecisionAuditAck::AuthorityChanged);
            }
            LiveRoutingPartitionResolution::NoPartition => {
                if summary.terminal_reason != DecisionCandidateReasonV1::NoPartition
                    && !is_unevaluated(summary.terminal_reason)
                {
                    return Ok(DecisionAuditAck::AuthorityChanged);
                }
            }
            LiveRoutingPartitionResolution::Found(found) => {
                if summary
                    .partition_id
                    .is_some_and(|id| id != found.partition_id.value())
                    || (summary.partition_id.is_none() && !is_unevaluated(summary.terminal_reason))
                {
                    return Ok(DecisionAuditAck::AuthorityChanged);
                }
            }
        }
    }

    let retiring = audit
        .neighbors
        .iter()
        .map(|neighbor| neighbor.anchor_id)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .try_fold(false, |found, anchor_id| {
            if found {
                return Ok(true);
            }
            transaction
                .query_row(
                    "SELECT EXISTS(
                        SELECT 1 FROM decision_retiring_anchors WHERE anchor_id = ?1
                     )",
                    [anchor_id.to_string()],
                    |row| row.get::<_, bool>(0),
                )
                .map_err(database_error)
        })?;
    if retiring {
        return Ok(DecisionAuditAck::SourceRetiring);
    }
    if !evidence_matches(transaction, audit)? {
        return Ok(DecisionAuditAck::EvidenceChanged);
    }

    insert_decision_graph(transaction, audit)
}

fn load_existing_decision(
    connection: &Connection,
    audit: &DecisionAuditV1,
) -> Result<Option<DecisionAuditAck>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT decision_id FROM decisions
             WHERE decision_id = ?1 OR (project_uuid = ?2 AND primary_call_uuid = ?3)
             ORDER BY decision_id",
        )
        .map_err(database_error)?;
    let ids = statement
        .query_map(
            params![
                audit.parent.decision_id.to_string(),
                audit.parent.project_uuid.to_string(),
                audit.parent.primary_call_uuid.to_string(),
            ],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    if ids.is_empty() {
        return Ok(None);
    }
    if ids.len() != 1 {
        return Ok(Some(DecisionAuditAck::Conflict));
    }
    let Ok(stored_id) = parse_uuid(&ids[0]) else {
        return Ok(Some(DecisionAuditAck::Conflict));
    };
    let stored = match load_decision_graph(connection, stored_id) {
        Ok(Some(stored)) => stored,
        Ok(None) => return Ok(Some(DecisionAuditAck::Conflict)),
        Err(error) if error.class() == LedgerErrorClass::CorruptDatabase => {
            return Ok(Some(DecisionAuditAck::Conflict));
        }
        Err(error) => return Err(error),
    };
    let exact = audit.persisted_eq(stored.graph).unwrap_or(false);
    Ok(Some(if exact {
        DecisionAuditAck::AlreadyApplied
    } else {
        DecisionAuditAck::Conflict
    }))
}

/// Insert or exactly replay one shape-2 decision inside an already-authorized transaction.
pub(super) fn insert_active_decision_audit_in_transaction(
    transaction: &Transaction<'_>,
    audit: &DecisionAuditV1,
) -> Result<DecisionAuditAck, LedgerError> {
    validate_audit_command(audit)?;
    if audit.parent.decision_shape_version != ACTIVE_DECISION_SHAPE_VERSION_V2 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    if let Some(existing) = load_existing_decision(transaction, audit)? {
        return Ok(existing);
    }
    insert_decision_graph(transaction, audit)
}

fn originating_config_matches(
    connection: &Connection,
    audit: &DecisionAuditV1,
    max_evidence_records: u64,
) -> Result<bool, LedgerError> {
    let process_config = connection
        .query_row(
            "SELECT config_generation_id FROM process_instances
             WHERE project_uuid = ?1 AND process_instance_id = ?2",
            params![
                audit.parent.project_uuid.to_string(),
                audit.parent.process_instance_id.to_string(),
            ],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    if process_config.as_deref() != Some(audit.parent.config_generation_id.as_str()) {
        return Ok(false);
    }
    let stored = connection
        .query_row(
            "SELECT canonical_config_json, canonical_payload_hash
             FROM config_generations
             WHERE project_uuid = ?1 AND config_generation_id = ?2",
            params![
                audit.parent.project_uuid.to_string(),
                audit.parent.config_generation_id,
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    let Some((document, payload_hash)) = stored else {
        return Ok(false);
    };
    let value: Json = serde_json::from_str(&document).map_err(|_| corrupt())?;
    if payload_hash != audit.parent.config_generation_id
        || sha256_hex(document.as_bytes()) != audit.parent.config_generation_id
        || canonical_json(&value).map_err(|_| corrupt())? != document
    {
        return Err(corrupt());
    }
    Ok(value.get("max_evidence_records").and_then(Json::as_u64) == Some(max_evidence_records))
}

fn candidate_policy_matches(
    connection: &Connection,
    audit: &DecisionAuditV1,
) -> Result<bool, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT canonical_policy_json, canonical_payload_hash
             FROM policy_versions
             WHERE project_uuid = ?1 AND pool_id = ?2 AND policy_version_id = ?3",
            params![
                audit.parent.project_uuid.to_string(),
                audit.parent.pool_id,
                audit.parent.policy_version_id,
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    let Some((document, payload_hash)) = stored else {
        return Ok(false);
    };
    let value: Json = serde_json::from_str(&document).map_err(|_| corrupt())?;
    if payload_hash != audit.parent.policy_version_id
        || sha256_hex(document.as_bytes()) != audit.parent.policy_version_id
        || canonical_json(&value).map_err(|_| corrupt())? != document
    {
        return Err(corrupt());
    }
    let Some(candidates) = value.pointer("/pool/candidates").and_then(Json::as_array) else {
        return Err(corrupt());
    };
    for summary in &audit.summaries {
        let matches = candidates
            .iter()
            .filter(|candidate| {
                candidate.get("id").and_then(Json::as_str) == Some(&summary.candidate_id)
                    && candidate.get("model").and_then(Json::as_str)
                        == Some(&summary.candidate_model)
                    && candidate.get("model_revision").and_then(Json::as_str)
                        == Some(&summary.candidate_model_revision)
                    && candidate.get("cost_rank").and_then(Json::as_u64)
                        == Some(u64::from(summary.cost_rank))
            })
            .count();
        if matches != 1 {
            return Ok(false);
        }
    }
    let members = audit
        .summaries
        .iter()
        .map(|summary| CandidateSetMemberInputV1 {
            candidate_id: summary.candidate_id.clone(),
            model: summary.candidate_model.clone(),
            model_revision: summary.candidate_model_revision.clone(),
            cost_rank: summary.cost_rank,
        })
        .collect::<Vec<_>>();
    let candidate_set = build_candidate_set_from_members_v1(&members).map_err(|_| corrupt())?;
    Ok(
        candidate_set.candidate_count == audit.parent.candidate_count
            && candidate_set.candidate_set_hash == audit.parent.candidate_set_hash,
    )
}

fn evidence_matches(
    connection: &Transaction<'_>,
    audit: &DecisionAuditV1,
) -> Result<bool, LedgerError> {
    let vector_space_id =
        VectorSpaceId::new(audit.parent.vector_space_id.clone()).map_err(|_| corrupt())?;
    let policy_document = load_policy_document(connection, audit)?;
    let policy = confidence_policy(audit, &policy_document)?;

    let mut engine =
        ConfidenceEngineV1::new(&policy, audit.summaries.len(), audit.parent.as_of_unix_ms)
            .map_err(|_| corrupt())?;
    let external_fallback = match audit.parent.final_reason {
        DecisionFinalReasonV1::EmbeddingUnavailable => {
            Some(ConfidenceDecisionReasonV1::EmbeddingUnavailable)
        }
        DecisionFinalReasonV1::VectorUnhealthy => Some(ConfidenceDecisionReasonV1::VectorUnhealthy),
        DecisionFinalReasonV1::VersionMismatch => Some(ConfidenceDecisionReasonV1::VersionMismatch),
        _ => None,
    };
    let mut cached_query_vector = None;

    for summary in &audit.summaries {
        if is_unevaluated(summary.terminal_reason) {
            if engine.needs_evidence() {
                let Some(reason) = external_fallback else {
                    return Ok(false);
                };
                if engine.stop_with_external_fallback(reason).is_err() {
                    return Ok(false);
                }
            }
            if engine
                .record_unevaluated(summary.candidate_id.clone(), summary.cost_rank)
                .is_err()
            {
                return Ok(false);
            }
            continue;
        }
        let evidence = if summary.terminal_reason == DecisionCandidateReasonV1::NoPartition {
            CandidateEvidenceV1::NoPartition
        } else {
            let Some(partition_id) = summary
                .partition_id
                .and_then(|value| PartitionId::new(value).ok())
            else {
                return Ok(false);
            };
            if cached_query_vector.is_none() {
                cached_query_vector = load_embedding_cache(
                    connection,
                    audit.parent.project_uuid,
                    &vector_space_id,
                    &audit.parent.canonical_query_hash,
                )?;
            }
            let Some(cache) = cached_query_vector.as_ref() else {
                return Ok(false);
            };
            let projected = match search_projected_neighbors_in_transaction(
                connection,
                &vector_space_id,
                partition_id,
                cache.vector.vector(),
                summary.top_k,
            ) {
                Ok(projected) => projected,
                Err(crate::vector_store::VectorStoreError::Corrupt) => return Err(corrupt()),
                Err(_) => return Ok(false),
            };
            let stored_neighbors = audit
                .neighbors
                .iter()
                .filter(|neighbor| neighbor.candidate_id == summary.candidate_id)
                .collect::<Vec<_>>();
            if projected.len() != stored_neighbors.len()
                || projected
                    .iter()
                    .zip(&stored_neighbors)
                    .any(|(projected, stored)| !projected_source_matches(projected, stored))
            {
                return Ok(false);
            }
            CandidateEvidenceV1::Neighbors(
                projected
                    .into_iter()
                    .map(project_confidence_neighbor)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| corrupt())?,
            )
        };
        let input = CandidateConfidenceInputV1::new(
            summary.candidate_id.clone(),
            summary.cost_rank,
            evidence,
        )
        .map_err(|_| corrupt())?;
        if engine.evaluate_candidate(input).is_err() {
            return Ok(false);
        }
    }
    let replay = engine.finish().map_err(|_| corrupt())?;
    if replay.recommended_candidate_id != audit.parent.candidate_id
        || replay.summaries.len() != audit.summaries.len()
    {
        return Ok(false);
    }
    for (replayed, stored) in replay.summaries.iter().zip(&audit.summaries) {
        let stored_neighbors = audit
            .neighbors
            .iter()
            .filter(|neighbor| neighbor.candidate_id == stored.candidate_id)
            .collect::<Vec<_>>();
        if !confidence_summary_matches(replayed, stored, &stored_neighbors) {
            return Ok(false);
        }
    }
    Ok(confidence_final_reason_matches(
        replay.reason,
        audit.parent.final_reason,
    ))
}

pub(super) fn active_decision_evidence_matches(
    connection: &Transaction<'_>,
    audit: &DecisionAuditV1,
) -> Result<bool, LedgerError> {
    if audit.parent.decision_shape_version != ACTIVE_DECISION_SHAPE_VERSION_V2 {
        return Ok(false);
    }
    let mapping = FrozenMappingKey::new(
        audit.parent.project_uuid,
        audit.parent.config_generation_id.clone(),
        audit.parent.pool_id.clone(),
        audit.parent.policy_version_id.clone(),
    )
    .map_err(|_| corrupt())?;
    let Some(resolved_mapping) = resolve_frozen_mapping(connection, &mapping)? else {
        return Ok(false);
    };
    if resolved_mapping.mapping.vector_space_id.as_str() != audit.parent.vector_space_id
        || !candidate_policy_matches(connection, audit)?
    {
        return Ok(false);
    }
    for (summary, artifact) in audit.summaries.iter().zip(&audit.partition_artifacts) {
        match resolve_live_routing_partition(connection, &mapping, artifact)? {
            LiveRoutingPartitionResolution::Found(found)
                if summary.partition_id == Some(found.partition_id.value()) => {}
            LiveRoutingPartitionResolution::NoPartition
                if summary.terminal_reason == DecisionCandidateReasonV1::NoPartition => {}
            _ => return Ok(false),
        }
    }
    let vector_space_id =
        VectorSpaceId::new(audit.parent.vector_space_id.clone()).map_err(|_| corrupt())?;
    let policy = active_confidence_policy(connection, audit)?;
    let mut engine =
        ConfidenceEngineV1::new(&policy, audit.summaries.len(), audit.parent.as_of_unix_ms)
            .map_err(|_| corrupt())?;
    let mut cached_query_vector = None;

    for summary in &audit.summaries {
        if is_unevaluated(summary.terminal_reason) {
            if summary.terminal_reason != DecisionCandidateReasonV1::NotEvaluatedAfterWinner
                || engine.needs_evidence()
                || engine
                    .record_unevaluated(summary.candidate_id.clone(), summary.cost_rank)
                    .is_err()
            {
                return Ok(false);
            }
            continue;
        }
        let evidence = if summary.terminal_reason == DecisionCandidateReasonV1::NoPartition {
            CandidateEvidenceV1::NoPartition
        } else {
            let Some(partition_id) = summary
                .partition_id
                .and_then(|value| PartitionId::new(value).ok())
            else {
                return Ok(false);
            };
            if cached_query_vector.is_none() {
                cached_query_vector = load_embedding_cache(
                    connection,
                    audit.parent.project_uuid,
                    &vector_space_id,
                    &audit.parent.canonical_query_hash,
                )?;
            }
            let Some(cache) = cached_query_vector.as_ref() else {
                return Ok(false);
            };
            let projected = match search_projected_neighbors_in_transaction(
                connection,
                &vector_space_id,
                partition_id,
                cache.vector.vector(),
                summary.top_k,
            ) {
                Ok(projected) => projected,
                Err(crate::vector_store::VectorStoreError::Corrupt) => return Err(corrupt()),
                Err(_) => return Ok(false),
            };
            let stored_neighbors = audit
                .neighbors
                .iter()
                .filter(|neighbor| neighbor.candidate_id == summary.candidate_id)
                .collect::<Vec<_>>();
            if projected.len() != stored_neighbors.len()
                || projected
                    .iter()
                    .zip(&stored_neighbors)
                    .any(|(projected, stored)| !projected_source_matches(projected, stored))
            {
                return Ok(false);
            }
            CandidateEvidenceV1::Neighbors(
                projected
                    .into_iter()
                    .map(project_confidence_neighbor)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|_| corrupt())?,
            )
        };
        let input = CandidateConfidenceInputV1::new(
            summary.candidate_id.clone(),
            summary.cost_rank,
            evidence,
        )
        .map_err(|_| corrupt())?;
        if engine
            .evaluate_candidate_with_gate(
                input,
                summary.promotion_lower_bound.value(),
                audit.parent.candidate_id.as_deref() == Some(summary.candidate_id.as_str()),
            )
            .is_err()
        {
            return Ok(false);
        }
    }
    let replay = engine.finish().map_err(|_| corrupt())?;
    if replay.recommended_candidate_id != audit.parent.candidate_id
        || replay.summaries.len() != audit.summaries.len()
    {
        return Ok(false);
    }
    Ok(replay
        .summaries
        .iter()
        .zip(&audit.summaries)
        .all(|(replayed, stored)| {
            let stored_neighbors = audit
                .neighbors
                .iter()
                .filter(|neighbor| neighbor.candidate_id == stored.candidate_id)
                .collect::<Vec<_>>();
            confidence_summary_matches(replayed, stored, &stored_neighbors)
        }))
}

fn active_confidence_policy(
    connection: &Connection,
    audit: &DecisionAuditV1,
) -> Result<ConfidencePolicyV1, LedgerError> {
    let first = audit.summaries.first().ok_or_else(corrupt)?;
    let policy_document = load_policy_document(connection, audit)?;
    let learning = policy_document
        .pointer("/pool/learning")
        .ok_or_else(corrupt)?;
    let judge_floor = policy_document
        .pointer("/pool/judge/config/judge_confidence_floor")
        .and_then(Json::as_f64)
        .ok_or_else(corrupt)?;
    let experiment_id = audit.parent.active_experiment_id.ok_or_else(corrupt)?;
    let active_half_life = connection
        .query_row(
            "SELECT anchor_shadow_half_life_seconds FROM active_experiments
             WHERE active_experiment_id = ?1",
            [experiment_id.to_string()],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    let active_half_life = u32::try_from(active_half_life).map_err(|_| corrupt())?;
    let configured = (
        learning.get("top_k").and_then(Json::as_u64),
        learning.get("radius").and_then(Json::as_f64),
        learning.get("min_points").and_then(Json::as_u64),
        learning.get("min_independent_roots").and_then(Json::as_u64),
        learning.get("min_effective_samples").and_then(Json::as_f64),
        learning.get("min_coverage").and_then(Json::as_f64),
        learning.get("prior_success").and_then(Json::as_f64),
        learning.get("prior_failure").and_then(Json::as_f64),
        learning
            .get("familywise_credible_level")
            .and_then(Json::as_f64),
    );
    let expected = (
        Some(u64::try_from(first.top_k).map_err(|_| corrupt())?),
        Some(first.radius.value()),
        Some(u64::try_from(first.min_points).map_err(|_| corrupt())?),
        Some(u64::try_from(first.min_independent_roots).map_err(|_| corrupt())?),
        Some(first.min_effective_samples.value()),
        Some(first.min_coverage.value()),
        Some(first.prior_success.value()),
        Some(first.prior_failure.value()),
        Some(first.familywise_credible_level.value()),
    );
    if configured != expected
        || first.time_decay_half_life_seconds.value() != f64::from(active_half_life)
        || audit.summaries.iter().any(|summary| {
            summary.top_k != first.top_k
                || summary.radius != first.radius
                || summary.min_points != first.min_points
                || summary.min_independent_roots != first.min_independent_roots
                || summary.min_effective_samples != first.min_effective_samples
                || summary.min_coverage != first.min_coverage
                || summary.time_decay_half_life_seconds != first.time_decay_half_life_seconds
                || summary.prior_success != first.prior_success
                || summary.prior_failure != first.prior_failure
                || summary.familywise_credible_level != first.familywise_credible_level
        })
    {
        return Err(corrupt());
    }
    ConfidencePolicyV1::new(
        first.top_k,
        first.radius.value(),
        first.min_points,
        first.min_independent_roots,
        first.min_effective_samples.value(),
        first.min_coverage.value(),
        first.time_decay_half_life_seconds.value(),
        first.prior_success.value(),
        first.prior_failure.value(),
        first.familywise_credible_level.value(),
        1.0,
        judge_floor,
    )
    .map_err(|_| corrupt())
}

fn load_policy_document(
    connection: &Connection,
    audit: &DecisionAuditV1,
) -> Result<Json, LedgerError> {
    let document = connection
        .query_row(
            "SELECT canonical_policy_json FROM policy_versions
             WHERE project_uuid = ?1 AND pool_id = ?2 AND policy_version_id = ?3",
            params![
                audit.parent.project_uuid.to_string(),
                audit.parent.pool_id,
                audit.parent.policy_version_id,
            ],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    serde_json::from_str(&document).map_err(|_| corrupt())
}

fn confidence_policy(
    audit: &DecisionAuditV1,
    policy: &Json,
) -> Result<ConfidencePolicyV1, LedgerError> {
    let first = audit.summaries.first().ok_or_else(corrupt)?;
    let learning = policy.pointer("/pool/learning").ok_or_else(corrupt)?;
    let judge_floor = policy
        .pointer("/pool/judge/config/judge_confidence_floor")
        .and_then(Json::as_f64)
        .ok_or_else(corrupt)?;
    let configured = (
        learning.get("top_k").and_then(Json::as_u64),
        learning.get("radius").and_then(Json::as_f64),
        learning.get("min_points").and_then(Json::as_u64),
        learning.get("min_independent_roots").and_then(Json::as_u64),
        learning.get("min_effective_samples").and_then(Json::as_f64),
        learning.get("min_coverage").and_then(Json::as_f64),
        learning
            .get("time_decay_half_life_seconds")
            .and_then(Json::as_f64),
        learning.get("prior_success").and_then(Json::as_f64),
        learning.get("prior_failure").and_then(Json::as_f64),
        learning
            .get("familywise_credible_level")
            .and_then(Json::as_f64),
        learning.get("promotion_lower_bound").and_then(Json::as_f64),
    );
    let expected = (
        Some(u64::try_from(first.top_k).map_err(|_| corrupt())?),
        Some(first.radius.value()),
        Some(u64::try_from(first.min_points).map_err(|_| corrupt())?),
        Some(u64::try_from(first.min_independent_roots).map_err(|_| corrupt())?),
        Some(first.min_effective_samples.value()),
        Some(first.min_coverage.value()),
        Some(first.time_decay_half_life_seconds.value()),
        Some(first.prior_success.value()),
        Some(first.prior_failure.value()),
        Some(first.familywise_credible_level.value()),
        Some(first.promotion_lower_bound.value()),
    );
    if configured != expected
        || audit.summaries.iter().any(|summary| {
            summary.top_k != first.top_k
                || summary.radius != first.radius
                || summary.min_points != first.min_points
                || summary.min_independent_roots != first.min_independent_roots
                || summary.min_effective_samples != first.min_effective_samples
                || summary.min_coverage != first.min_coverage
                || summary.time_decay_half_life_seconds != first.time_decay_half_life_seconds
                || summary.prior_success != first.prior_success
                || summary.prior_failure != first.prior_failure
                || summary.familywise_credible_level != first.familywise_credible_level
                || summary.promotion_lower_bound != first.promotion_lower_bound
        })
    {
        return Err(corrupt());
    }
    let expected_alpha =
        (1.0 - first.familywise_credible_level.value()) / audit.summaries.len() as f64;
    if !expected_alpha.is_finite()
        || audit
            .summaries
            .iter()
            .any(|summary| (summary.candidate_alpha.value() - expected_alpha).abs() > 1.0e-12)
    {
        return Err(corrupt());
    }
    ConfidencePolicyV1::new(
        first.top_k,
        first.radius.value(),
        first.min_points,
        first.min_independent_roots,
        first.min_effective_samples.value(),
        first.min_coverage.value(),
        first.time_decay_half_life_seconds.value(),
        first.prior_success.value(),
        first.prior_failure.value(),
        first.familywise_credible_level.value(),
        first.promotion_lower_bound.value(),
        judge_floor,
    )
    .map_err(|_| corrupt())
}

fn projected_source_matches(
    projected: &ProjectedVectorNeighbor,
    stored: &DecisionNeighborV1,
) -> bool {
    projected.vector_match.record_id().value() == stored.evidence_vector_link_id
        && projected.vector_match.distance().to_bits() == stored.distance.bits()
        && projected.shadow_attempt_id == stored.shadow_attempt_id
        && projected.anchor_id == stored.anchor_id
        && projected.learning_generation_id == stored.learning_generation_id
        && projected
            .evaluation
            .as_ref()
            .map(|value| value.evaluation_id)
            == stored.evaluation_id
        && projected
            .evaluation
            .as_ref()
            .and_then(|value| value.binary_label)
            .map(map_binary_label)
            == stored.binary_label
}

fn project_confidence_neighbor(
    neighbor: ProjectedVectorNeighbor,
) -> Result<ConfidenceNeighborInputV1, crate::confidence::ConfidenceInputError> {
    let terminal_class = match neighbor.terminal_class {
        ShadowTerminalClass::Completed => ConfidenceTerminalClassV1::Completed,
        ShadowTerminalClass::DeterministicFailure => {
            ConfidenceTerminalClassV1::DeterministicFailure
        }
        ShadowTerminalClass::OperationalFailure => ConfidenceTerminalClassV1::OperationalFailure,
        ShadowTerminalClass::SkippedCooloff => ConfidenceTerminalClassV1::SkippedCooloff,
        ShadowTerminalClass::CanceledShutdown => ConfidenceTerminalClassV1::CanceledShutdown,
        ShadowTerminalClass::OrphanedBeforeSchedule => {
            ConfidenceTerminalClassV1::OrphanedBeforeSchedule
        }
        ShadowTerminalClass::OrphanedInFlight => ConfidenceTerminalClassV1::OrphanedInFlight,
    };
    let evaluation = neighbor
        .evaluation
        .map(|evaluation| {
            ConfidenceEvaluationInputV1::new(
                evaluation.evaluation_id,
                match evaluation.source {
                    JudgeEvaluationSourceV1::DeterministicValidator => {
                        ConfidenceEvaluationSourceV1::DeterministicValidator
                    }
                    JudgeEvaluationSourceV1::Judge => ConfidenceEvaluationSourceV1::Judge,
                },
                evaluation.binary_label.map(|label| match label {
                    JudgeBinaryLabelV1::Pass => ConfidenceBinaryLabelV1::Pass,
                    JudgeBinaryLabelV1::Fail => ConfidenceBinaryLabelV1::Fail,
                }),
                evaluation.judge_confidence.map(|value| value.bits),
                evaluation.promotion_eligible,
                evaluation.created_at_unix_ms,
            )
        })
        .transpose()?;
    ConfidenceNeighborInputV1::new(
        neighbor.vector_match.record_id().value(),
        neighbor.shadow_attempt_id,
        neighbor.anchor_id,
        neighbor.root_uuid,
        terminal_class,
        neighbor.vector_match.distance().to_bits(),
        evaluation,
    )
}

fn confidence_summary_matches(
    replayed: &CandidateConfidenceSummaryV1,
    stored: &DecisionCandidateSummaryV1,
    stored_neighbors: &[&DecisionNeighborV1],
) -> bool {
    replayed.candidate_id == stored.candidate_id
        && replayed.cost_rank == stored.cost_rank
        && replayed.top_k_points == stored.returned_neighbor_count
        && replayed.raw_points == stored.within_radius_count
        && replayed.labeled_points == stored.labeled_point_count
        && replayed.attempted_roots == stored.attempted_root_count
        && replayed.labeled_roots == stored.labeled_root_count
        && replayed
            .neighbors
            .iter()
            .filter(|neighbor| neighbor.selected_root)
            .count()
            == stored.selected_root_count
        && map_confidence_reason(replayed.reason) == stored.terminal_reason
        && float_option_matches(replayed.coverage, stored.coverage)
        && float_option_matches(replayed.sum_weight, stored.sum_weight)
        && float_option_matches(replayed.sum_weighted_pass, stored.sum_weighted_label)
        && float_option_matches(replayed.sum_weight_squared, stored.sum_squared_weight)
        && float_option_matches(replayed.p_hat, stored.p_hat)
        && float_option_matches(replayed.n_eff, stored.effective_sample_size)
        && float_option_matches(replayed.beta_alpha, stored.beta_alpha)
        && float_option_matches(replayed.beta_beta, stored.beta_beta)
        && replayed
            .candidate_alpha
            .is_none_or(|value| float_matches(value, stored.candidate_alpha))
        && float_option_matches(replayed.lower_bound, stored.lower_bound)
        && replayed.neighbors.len() == stored_neighbors.len()
        && replayed
            .neighbors
            .iter()
            .zip(stored_neighbors)
            .all(|(replayed, stored)| confidence_neighbor_matches(replayed, stored))
}

fn confidence_neighbor_matches(replayed: &AuditedNeighborV1, stored: &DecisionNeighborV1) -> bool {
    replayed.candidate_ordinal == stored.candidate_neighbor_ordinal
        && replayed.evidence_vector_link_id == stored.evidence_vector_link_id
        && replayed.shadow_attempt_id == stored.shadow_attempt_id
        && replayed.anchor_id == stored.anchor_id
        && replayed.evaluation_id == stored.evaluation_id
        && replayed.distance_bits == stored.distance.bits()
        && age_matches(replayed.age_millis, stored.age_seconds)
        && float_option_matches(replayed.similarity_weight, stored.similarity_weight)
        && float_option_matches(replayed.time_weight, stored.time_weight)
        && float_option_matches(replayed.final_weight, stored.final_weight)
        && replayed.binary_label.map(|label| match label {
            ConfidenceBinaryLabelV1::Pass => DecisionBinaryLabelV1::Pass,
            ConfidenceBinaryLabelV1::Fail => DecisionBinaryLabelV1::Fail,
        }) == stored.binary_label
        && replayed.selected_root == stored.selected_for_root
        && replayed.root_group_ordinal == stored.root_group_ordinal
        && map_exclusion_reason(replayed.exclusion_reason) == stored.exclusion_reason
}

fn float_option_matches(replayed: Option<f64>, stored: Option<AuditF64V1>) -> bool {
    match (replayed, stored) {
        (Some(replayed), Some(stored)) => float_matches(replayed, stored),
        (None, None) => true,
        _ => false,
    }
}

fn float_matches(replayed: f64, stored: AuditF64V1) -> bool {
    replayed.is_finite()
        && stored.value().is_finite()
        && (replayed - stored.value()).abs() <= 1.0e-12
}

#[allow(clippy::cast_precision_loss)]
fn age_matches(replayed_millis: Option<u64>, stored: Option<AuditF64V1>) -> bool {
    match (replayed_millis, stored) {
        (Some(millis), Some(stored)) => {
            (millis as f64 / 1_000.0).to_bits() == stored.value().to_bits()
        }
        (None, None) => true,
        _ => false,
    }
}

fn map_binary_label(label: JudgeBinaryLabelV1) -> DecisionBinaryLabelV1 {
    match label {
        JudgeBinaryLabelV1::Pass => DecisionBinaryLabelV1::Pass,
        JudgeBinaryLabelV1::Fail => DecisionBinaryLabelV1::Fail,
    }
}

fn map_confidence_reason(reason: CandidateConfidenceReasonV1) -> DecisionCandidateReasonV1 {
    match reason {
        CandidateConfidenceReasonV1::NoPartition => DecisionCandidateReasonV1::NoPartition,
        CandidateConfidenceReasonV1::SparsePoints => DecisionCandidateReasonV1::SparsePoints,
        CandidateConfidenceReasonV1::InsufficientRoots => {
            DecisionCandidateReasonV1::InsufficientRoots
        }
        CandidateConfidenceReasonV1::LowCoverage => DecisionCandidateReasonV1::LowCoverage,
        CandidateConfidenceReasonV1::InvalidEvidenceTime => {
            DecisionCandidateReasonV1::InvalidEvidenceTime
        }
        CandidateConfidenceReasonV1::NumericError => DecisionCandidateReasonV1::NumericError,
        CandidateConfidenceReasonV1::InsufficientEffectiveSamples => {
            DecisionCandidateReasonV1::InsufficientEffectiveSamples
        }
        CandidateConfidenceReasonV1::LowerBoundBelowThreshold => {
            DecisionCandidateReasonV1::LowerBoundBelowThreshold
        }
        CandidateConfidenceReasonV1::Passed => DecisionCandidateReasonV1::Passed,
        CandidateConfidenceReasonV1::NotEvaluatedAfterWinner => {
            DecisionCandidateReasonV1::NotEvaluatedAfterWinner
        }
        CandidateConfidenceReasonV1::NotEvaluatedAfterFallback => {
            DecisionCandidateReasonV1::NotEvaluatedAfterFallback
        }
    }
}

fn map_exclusion_reason(reason: NeighborExclusionReasonV1) -> DecisionNeighborExclusionReasonV1 {
    match reason {
        NeighborExclusionReasonV1::OutsideRadius => {
            DecisionNeighborExclusionReasonV1::OutsideRadius
        }
        NeighborExclusionReasonV1::IneligibleQuality => {
            DecisionNeighborExclusionReasonV1::IneligibleQuality
        }
        NeighborExclusionReasonV1::DuplicateRoot => {
            DecisionNeighborExclusionReasonV1::DuplicateRoot
        }
        NeighborExclusionReasonV1::Included => DecisionNeighborExclusionReasonV1::Included,
    }
}

fn confidence_final_reason_matches(
    replayed: ConfidenceDecisionReasonV1,
    stored: DecisionFinalReasonV1,
) -> bool {
    match replayed {
        ConfidenceDecisionReasonV1::RecommendObserveOnly => {
            stored == DecisionFinalReasonV1::RecommendObserveOnly
        }
        ConfidenceDecisionReasonV1::InvalidEvidenceTime => {
            stored == DecisionFinalReasonV1::InvalidEvidenceTime
        }
        ConfidenceDecisionReasonV1::NumericError => stored == DecisionFinalReasonV1::NumericError,
        ConfidenceDecisionReasonV1::NoCandidatePassed => !matches!(
            stored,
            DecisionFinalReasonV1::RecommendObserveOnly
                | DecisionFinalReasonV1::EmbeddingUnavailable
                | DecisionFinalReasonV1::VectorUnhealthy
                | DecisionFinalReasonV1::VersionMismatch
                | DecisionFinalReasonV1::InvalidEvidenceTime
                | DecisionFinalReasonV1::NumericError
        ),
        ConfidenceDecisionReasonV1::EmbeddingUnavailable => {
            stored == DecisionFinalReasonV1::EmbeddingUnavailable
        }
        ConfidenceDecisionReasonV1::VectorUnhealthy => {
            stored == DecisionFinalReasonV1::VectorUnhealthy
        }
        ConfidenceDecisionReasonV1::VersionMismatch => {
            stored == DecisionFinalReasonV1::VersionMismatch
        }
    }
}

fn is_unevaluated(reason: DecisionCandidateReasonV1) -> bool {
    matches!(
        reason,
        DecisionCandidateReasonV1::NotEvaluatedAfterWinner
            | DecisionCandidateReasonV1::NotEvaluatedAfterFallback
    )
}

fn validate_audit_command(audit: &DecisionAuditV1) -> Result<(), LedgerError> {
    audit
        .validate_frozen()
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    prepared_query_artifact(audit)?;
    Ok(())
}

fn prepared_query_artifact(
    audit: &DecisionAuditV1,
) -> Result<&CanonicalRoutingQueryArtifactV1, LedgerError> {
    if audit.prepared_query.canonical_query_hash() != audit.parent.canonical_query_hash {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(audit.prepared_query.artifact())
}

enum GraphInsertError {
    Conflict,
    Ledger(LedgerError),
}

fn insert_decision_graph(
    transaction: &Transaction<'_>,
    audit: &DecisionAuditV1,
) -> Result<DecisionAuditAck, LedgerError> {
    transaction
        .execute_batch("SAVEPOINT decision_audit_graph")
        .map_err(database_error)?;
    let result = insert_decision_graph_after_savepoint(transaction, audit);
    match result {
        Ok(()) => {
            transaction
                .execute_batch("RELEASE decision_audit_graph")
                .map_err(database_error)?;
            Ok(DecisionAuditAck::Applied)
        }
        Err(GraphInsertError::Conflict) => {
            transaction
                .execute_batch("ROLLBACK TO decision_audit_graph; RELEASE decision_audit_graph;")
                .map_err(database_error)?;
            Ok(DecisionAuditAck::Conflict)
        }
        Err(GraphInsertError::Ledger(error)) => {
            let _ = transaction
                .execute_batch("ROLLBACK TO decision_audit_graph; RELEASE decision_audit_graph;");
            Err(error)
        }
    }
}

fn insert_decision_graph_after_savepoint(
    transaction: &Transaction<'_>,
    audit: &DecisionAuditV1,
) -> Result<(), GraphInsertError> {
    let query = prepared_query_artifact(audit).map_err(GraphInsertError::Ledger)?;
    if ensure_canonical_query(transaction, query, audit.parent.created_at_unix_ms)
        .map_err(GraphInsertError::Ledger)?
        == CanonicalQueryEnsureAck::Conflict
    {
        return Err(GraphInsertError::Conflict);
    }
    insert_parent(transaction, &audit.parent)?;
    for summary in &audit.summaries {
        insert_summary(transaction, summary)?;
    }
    for neighbor in &audit.neighbors {
        insert_neighbor(transaction, neighbor)?;
    }
    Ok(())
}

fn insert_parent(
    transaction: &Transaction<'_>,
    parent: &DecisionRecordV1,
) -> Result<(), GraphInsertError> {
    let mode = match parent.decision_shape_version {
        DECISION_SHAPE_VERSION_V1 => "recommend",
        ACTIVE_DECISION_SHAPE_VERSION_V2 => "active",
        _ => {
            return Err(GraphInsertError::Ledger(
                LedgerErrorClass::IdentityInvariant.into(),
            ));
        }
    };
    let inserted = transaction
        .execute(
            "INSERT INTO decisions (
                decision_id, decision_shape_version, algorithm_version, project_uuid,
                process_instance_id, config_generation_id, policy_version_id,
                learning_generation_id, cohort_generation_id, active_experiment_id,
                active_authorization_state_event_id, pool_id, candidate_id, root_key,
                primary_call_uuid, mode, canonical_query_hash, partition_base_json,
                partition_base_hash, vector_space_id, candidate_set_hash, candidate_count,
                recommended_model, recommended_model_revision, served_model,
                served_model_revision, as_of_unix_ms, decision_latency_ms, final_reason,
                summary_count, summary_aggregate_hash, neighbor_count,
                neighbor_aggregate_hash, aggregate_size_bytes, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (
                :decision_id, :shape, :algorithm, :project, :process, :config, :policy,
                :learning, :cohort, :experiment, :authorization, :pool, :candidate,
                :root_key, :primary_call, :mode, :query, :partition_base_json,
                :partition_base_hash, :space,
                :candidate_set_hash, :candidate_count, :recommended_model,
                :recommended_revision, :served_model, :served_revision, :as_of,
                :latency, :final_reason, :summary_count, :summary_hash, :neighbor_count,
                :neighbor_hash, :aggregate_size, :created_at, :payload_hash
             )",
            named_params! {
                ":decision_id": parent.decision_id.to_string(),
                ":shape": to_i64(parent.decision_shape_version)?,
                ":algorithm": to_i64(parent.algorithm_version)?,
                ":project": parent.project_uuid.to_string(),
                ":process": parent.process_instance_id.to_string(),
                ":config": parent.config_generation_id,
                ":policy": parent.policy_version_id,
                ":learning": parent.learning_generation_id.to_string(),
                ":cohort": parent.cohort_generation_id.map(|value| value.to_string()),
                ":experiment": parent.active_experiment_id.map(|value| value.to_string()),
                ":authorization": parent
                    .active_authorization_state_event_id
                    .map(|value| value.to_string()),
                ":pool": parent.pool_id,
                ":candidate": parent.candidate_id,
                ":root_key": parent.root_key,
                ":primary_call": parent.primary_call_uuid.to_string(),
                ":mode": mode,
                ":query": parent.canonical_query_hash,
                ":partition_base_json": parent.partition_base_json,
                ":partition_base_hash": parent.partition_base_hash,
                ":space": parent.vector_space_id,
                ":candidate_set_hash": parent.candidate_set_hash,
                ":candidate_count": to_i64(parent.candidate_count)?,
                ":recommended_model": parent.recommended_model,
                ":recommended_revision": parent.recommended_model_revision,
                ":served_model": parent.served_model,
                ":served_revision": parent.served_model_revision,
                ":as_of": parent.as_of_unix_ms,
                ":latency": to_i64(parent.decision_latency_ms)?,
                ":final_reason": parent.final_reason.as_str(),
                ":summary_count": to_i64(parent.summary_count)?,
                ":summary_hash": parent.summary_aggregate_hash,
                ":neighbor_count": to_i64(parent.neighbor_count)?,
                ":neighbor_hash": parent.neighbor_aggregate_hash,
                ":aggregate_size": to_i64(parent.aggregate_size_bytes)?,
                ":created_at": parent.created_at_unix_ms,
                ":payload_hash": parent.canonical_payload_hash,
            },
        )
        .map_err(graph_sql_error)?;
    if inserted != 1 {
        return Err(GraphInsertError::Conflict);
    }
    Ok(())
}

fn insert_summary(
    transaction: &Transaction<'_>,
    summary: &DecisionCandidateSummaryV1,
) -> Result<(), GraphInsertError> {
    let inserted = transaction
        .execute(
            "INSERT INTO decision_candidate_summaries (
                decision_id, candidate_id, rank_ordinal, candidate_model,
                candidate_model_revision, cost_rank, learning_generation_id,
                vector_space_id, partition_hash, partition_id, decoding_fingerprint,
                top_k, radius, radius_bits, min_points, min_independent_roots,
                min_effective_samples, min_effective_samples_bits, min_coverage,
                min_coverage_bits, time_decay_half_life_seconds,
                time_decay_half_life_seconds_bits, prior_success, prior_success_bits,
                prior_failure, prior_failure_bits, familywise_credible_level,
                familywise_credible_level_bits, candidate_alpha, candidate_alpha_bits,
                promotion_lower_bound, promotion_lower_bound_bits,
                returned_neighbor_count, within_radius_count, labeled_point_count,
                attempted_root_count, labeled_root_count, selected_root_count,
                coverage, coverage_bits, sum_weight, sum_weight_bits,
                sum_weighted_label, sum_weighted_label_bits, sum_squared_weight,
                sum_squared_weight_bits, p_hat, p_hat_bits, effective_sample_size,
                effective_sample_size_bits, beta_alpha, beta_alpha_bits, beta_beta,
                beta_beta_bits, lower_bound, lower_bound_bits, partition_gate_passed,
                points_gate_passed, roots_gate_passed, coverage_gate_passed,
                weight_gate_passed, effective_samples_gate_passed,
                beta_quantile_gate_passed, lower_bound_gate_passed, terminal_reason,
                neighbor_count, neighbor_aggregate_hash, canonical_payload_hash
             ) VALUES (
                :decision, :candidate, :rank, :model, :revision, :cost, :learning,
                :space, :partition_hash, :partition_id, :decoding, :top_k, :radius,
                :radius_bits, :min_points, :min_roots, :min_effective,
                :min_effective_bits, :min_coverage, :min_coverage_bits, :half_life,
                :half_life_bits, :prior_success, :prior_success_bits, :prior_failure,
                :prior_failure_bits, :credible, :credible_bits, :candidate_alpha,
                :candidate_alpha_bits, :promotion, :promotion_bits, :returned, :within,
                :labeled_points, :attempted_roots, :labeled_roots, :selected_roots,
                :coverage, :coverage_bits, :sum_weight, :sum_weight_bits,
                :sum_weighted, :sum_weighted_bits, :sum_squared, :sum_squared_bits,
                :p_hat, :p_hat_bits, :effective, :effective_bits, :beta_alpha,
                :beta_alpha_bits, :beta_beta, :beta_beta_bits, :lower_bound,
                :lower_bound_bits, :partition_gate, :points_gate, :roots_gate,
                :coverage_gate, :weight_gate, :effective_gate, :beta_gate, :lower_gate,
                :reason, :neighbor_count, :neighbor_hash, :payload_hash
             )",
            named_params! {
                ":decision": summary.decision_id.to_string(),
                ":candidate": summary.candidate_id,
                ":rank": to_i64(summary.rank_ordinal)?,
                ":model": summary.candidate_model,
                ":revision": summary.candidate_model_revision,
                ":cost": i64::from(summary.cost_rank),
                ":learning": summary.learning_generation_id.to_string(),
                ":space": summary.vector_space_id,
                ":partition_hash": summary.partition_hash,
                ":partition_id": summary.partition_id,
                ":decoding": summary.decoding_fingerprint,
                ":top_k": to_i64(summary.top_k)?,
                ":radius": summary.radius.value(),
                ":radius_bits": float64_bits(summary.radius)?,
                ":min_points": to_i64(summary.min_points)?,
                ":min_roots": to_i64(summary.min_independent_roots)?,
                ":min_effective": summary.min_effective_samples.value(),
                ":min_effective_bits": float64_bits(summary.min_effective_samples)?,
                ":min_coverage": summary.min_coverage.value(),
                ":min_coverage_bits": float64_bits(summary.min_coverage)?,
                ":half_life": summary.time_decay_half_life_seconds.value(),
                ":half_life_bits": float64_bits(summary.time_decay_half_life_seconds)?,
                ":prior_success": summary.prior_success.value(),
                ":prior_success_bits": float64_bits(summary.prior_success)?,
                ":prior_failure": summary.prior_failure.value(),
                ":prior_failure_bits": float64_bits(summary.prior_failure)?,
                ":credible": summary.familywise_credible_level.value(),
                ":credible_bits": float64_bits(summary.familywise_credible_level)?,
                ":candidate_alpha": summary.candidate_alpha.value(),
                ":candidate_alpha_bits": float64_bits(summary.candidate_alpha)?,
                ":promotion": summary.promotion_lower_bound.value(),
                ":promotion_bits": float64_bits(summary.promotion_lower_bound)?,
                ":returned": to_i64(summary.returned_neighbor_count)?,
                ":within": to_i64(summary.within_radius_count)?,
                ":labeled_points": to_i64(summary.labeled_point_count)?,
                ":attempted_roots": to_i64(summary.attempted_root_count)?,
                ":labeled_roots": to_i64(summary.labeled_root_count)?,
                ":selected_roots": to_i64(summary.selected_root_count)?,
                ":coverage": optional_float_value(summary.coverage),
                ":coverage_bits": optional_float_bits(summary.coverage)?,
                ":sum_weight": optional_float_value(summary.sum_weight),
                ":sum_weight_bits": optional_float_bits(summary.sum_weight)?,
                ":sum_weighted": optional_float_value(summary.sum_weighted_label),
                ":sum_weighted_bits": optional_float_bits(summary.sum_weighted_label)?,
                ":sum_squared": optional_float_value(summary.sum_squared_weight),
                ":sum_squared_bits": optional_float_bits(summary.sum_squared_weight)?,
                ":p_hat": optional_float_value(summary.p_hat),
                ":p_hat_bits": optional_float_bits(summary.p_hat)?,
                ":effective": optional_float_value(summary.effective_sample_size),
                ":effective_bits": optional_float_bits(summary.effective_sample_size)?,
                ":beta_alpha": optional_float_value(summary.beta_alpha),
                ":beta_alpha_bits": optional_float_bits(summary.beta_alpha)?,
                ":beta_beta": optional_float_value(summary.beta_beta),
                ":beta_beta_bits": optional_float_bits(summary.beta_beta)?,
                ":lower_bound": optional_float_value(summary.lower_bound),
                ":lower_bound_bits": optional_float_bits(summary.lower_bound)?,
                ":partition_gate": optional_bool(summary.partition_gate_passed),
                ":points_gate": optional_bool(summary.points_gate_passed),
                ":roots_gate": optional_bool(summary.roots_gate_passed),
                ":coverage_gate": optional_bool(summary.coverage_gate_passed),
                ":weight_gate": optional_bool(summary.weight_gate_passed),
                ":effective_gate": optional_bool(summary.effective_samples_gate_passed),
                ":beta_gate": optional_bool(summary.beta_quantile_gate_passed),
                ":lower_gate": optional_bool(summary.lower_bound_gate_passed),
                ":reason": summary.terminal_reason.as_str(),
                ":neighbor_count": to_i64(summary.neighbor_count)?,
                ":neighbor_hash": summary.neighbor_aggregate_hash,
                ":payload_hash": summary.canonical_payload_hash,
            },
        )
        .map_err(graph_sql_error)?;
    if inserted != 1 {
        return Err(GraphInsertError::Conflict);
    }
    Ok(())
}

fn insert_neighbor(
    transaction: &Transaction<'_>,
    neighbor: &DecisionNeighborV1,
) -> Result<(), GraphInsertError> {
    let inserted = transaction
        .execute(
            "INSERT INTO decision_neighbors (
                decision_id, neighbor_ordinal, candidate_id, candidate_neighbor_ordinal,
                evidence_vector_link_id, shadow_attempt_id, anchor_id, evaluation_id,
                learning_generation_id, distance, distance_f32_bits, age_seconds,
                age_seconds_bits, similarity_weight, similarity_weight_bits, time_weight,
                time_weight_bits, final_weight, final_weight_bits, binary_label,
                selected_for_root, root_group_ordinal, exclusion_reason,
                canonical_payload_hash
             ) VALUES (
                :decision, :ordinal, :candidate, :candidate_ordinal, :link, :attempt,
                :anchor, :evaluation, :learning, :distance, :distance_bits, :age,
                :age_bits, :similarity, :similarity_bits, :time_weight, :time_bits,
                :final_weight, :final_bits, :label, :selected, :root_ordinal,
                :reason, :payload_hash
             )",
            named_params! {
                ":decision": neighbor.decision_id.to_string(),
                ":ordinal": to_i64(neighbor.neighbor_ordinal)?,
                ":candidate": neighbor.candidate_id,
                ":candidate_ordinal": to_i64(neighbor.candidate_neighbor_ordinal)?,
                ":link": neighbor.evidence_vector_link_id.to_string(),
                ":attempt": neighbor.shadow_attempt_id.to_string(),
                ":anchor": neighbor.anchor_id.to_string(),
                ":evaluation": neighbor.evaluation_id.map(|value| value.to_string()),
                ":learning": neighbor.learning_generation_id.to_string(),
                ":distance": f64::from(neighbor.distance.value()),
                ":distance_bits": i64::from(neighbor.distance.bits()),
                ":age": optional_float_value(neighbor.age_seconds),
                ":age_bits": optional_float_bits(neighbor.age_seconds)?,
                ":similarity": optional_float_value(neighbor.similarity_weight),
                ":similarity_bits": optional_float_bits(neighbor.similarity_weight)?,
                ":time_weight": optional_float_value(neighbor.time_weight),
                ":time_bits": optional_float_bits(neighbor.time_weight)?,
                ":final_weight": optional_float_value(neighbor.final_weight),
                ":final_bits": optional_float_bits(neighbor.final_weight)?,
                ":label": neighbor.binary_label.map(DecisionBinaryLabelV1::as_str),
                ":selected": i64::from(neighbor.selected_for_root),
                ":root_ordinal": to_i64(neighbor.root_group_ordinal)?,
                ":reason": neighbor.exclusion_reason.as_str(),
                ":payload_hash": neighbor.canonical_payload_hash,
            },
        )
        .map_err(graph_sql_error)?;
    if inserted != 1 {
        return Err(GraphInsertError::Conflict);
    }
    Ok(())
}

fn graph_sql_error(error: SqliteError) -> GraphInsertError {
    if error.sqlite_error_code() == Some(ErrorCode::ConstraintViolation) {
        GraphInsertError::Conflict
    } else {
        GraphInsertError::Ledger(map_sqlite_error(
            &error,
            LedgerErrorClass::DatabaseOperationFailed,
        ))
    }
}

fn to_i64<T>(value: T) -> Result<i64, GraphInsertError>
where
    i64: TryFrom<T>,
{
    i64::try_from(value).map_err(|_| {
        GraphInsertError::Ledger(LedgerError::new(LedgerErrorClass::IdentityInvariant))
    })
}

fn float64_bits(value: AuditF64V1) -> Result<i64, GraphInsertError> {
    Ok(i64::from_ne_bytes(value.bits().to_ne_bytes()))
}

fn optional_float_value(value: Option<AuditF64V1>) -> Option<f64> {
    value.map(AuditF64V1::value)
}

fn optional_float_bits(value: Option<AuditF64V1>) -> Result<Option<i64>, GraphInsertError> {
    value.map(float64_bits).transpose()
}

fn optional_bool(value: Option<bool>) -> Option<i64> {
    value.map(i64::from)
}

/// Reload and verify every immutable row in one persisted decision aggregate.
pub(crate) fn load_decision_graph(
    connection: &Connection,
    decision_id: Uuid,
) -> Result<Option<VerifiedStoredDecisionGraphV1>, LedgerError> {
    if !is_uuid_v7(decision_id) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let Some(parent) = load_decision_parent(connection, decision_id)? else {
        return Ok(None);
    };
    let summaries = load_decision_summaries(connection, decision_id)?;
    let neighbors = load_decision_neighbors(connection, decision_id)?;
    StoredDecisionGraphV1 {
        parent,
        summaries,
        neighbors,
    }
    .verify()
    .map(Some)
    .map_err(|_| corrupt())
}

/// Reload and verify the immutable parent row without loading bounded child audits.
pub(crate) fn load_verified_decision_parent(
    connection: &Connection,
    decision_id: Uuid,
) -> Result<Option<DecisionRecordV1>, LedgerError> {
    if !is_uuid_v7(decision_id) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let parent = load_decision_parent(connection, decision_id)?;
    if let Some(parent) = &parent {
        verify_decision_parent_record(parent).map_err(|_| corrupt())?;
    }
    Ok(parent)
}

fn load_decision_parent(
    connection: &Connection,
    decision_id: Uuid,
) -> Result<Option<DecisionRecordV1>, LedgerError> {
    connection
        .query_row(
            "SELECT decision_id, decision_shape_version, algorithm_version, project_uuid,
                    process_instance_id, config_generation_id, policy_version_id,
                    learning_generation_id, cohort_generation_id, active_experiment_id,
                    active_authorization_state_event_id, pool_id, candidate_id, root_key,
                    primary_call_uuid, mode, canonical_query_hash, partition_base_json,
                    partition_base_hash, vector_space_id, candidate_set_hash, candidate_count,
                    recommended_model, recommended_model_revision, served_model,
                    served_model_revision, as_of_unix_ms, decision_latency_ms, final_reason,
                    summary_count, summary_aggregate_hash, neighbor_count,
                    neighbor_aggregate_hash, aggregate_size_bytes, created_at_unix_ms,
                    canonical_payload_hash
             FROM decisions WHERE decision_id = ?1",
            [decision_id.to_string()],
            |row| {
                let decision_shape_version = row_u32(row, 1)?;
                let mode = row.get::<_, String>(15)?;
                if !matches!(
                    (decision_shape_version, mode.as_str()),
                    (DECISION_SHAPE_VERSION_V1, "recommend")
                        | (ACTIVE_DECISION_SHAPE_VERSION_V2, "active")
                ) {
                    return Err(row_corrupt(15, Type::Text));
                }
                Ok(DecisionRecordV1 {
                    decision_id: row_uuid(row, 0)?,
                    decision_shape_version,
                    algorithm_version: row_u32(row, 2)?,
                    project_uuid: row_uuid(row, 3)?,
                    process_instance_id: row_uuid(row, 4)?,
                    config_generation_id: row.get(5)?,
                    policy_version_id: row.get(6)?,
                    learning_generation_id: row_uuid(row, 7)?,
                    cohort_generation_id: row_optional_uuid(row, 8)?,
                    active_experiment_id: row_optional_uuid(row, 9)?,
                    active_authorization_state_event_id: row_optional_uuid(row, 10)?,
                    pool_id: row.get(11)?,
                    candidate_id: row.get(12)?,
                    root_key: row.get(13)?,
                    primary_call_uuid: row_uuid(row, 14)?,
                    canonical_query_hash: row.get(16)?,
                    partition_base_json: row.get(17)?,
                    partition_base_hash: row.get(18)?,
                    vector_space_id: row.get(19)?,
                    candidate_set_hash: row.get(20)?,
                    candidate_count: row_usize(row, 21)?,
                    recommended_model: row.get(22)?,
                    recommended_model_revision: row.get(23)?,
                    served_model: row.get(24)?,
                    served_model_revision: row.get(25)?,
                    as_of_unix_ms: row.get(26)?,
                    decision_latency_ms: row_u64(row, 27)?,
                    final_reason: DecisionFinalReasonV1::parse(&row.get::<_, String>(28)?)
                        .map_err(|_| row_corrupt(28, Type::Text))?,
                    summary_count: row_usize(row, 29)?,
                    summary_aggregate_hash: row.get(30)?,
                    neighbor_count: row_usize(row, 31)?,
                    neighbor_aggregate_hash: row.get(32)?,
                    aggregate_size_bytes: row_usize(row, 33)?,
                    created_at_unix_ms: row.get(34)?,
                    canonical_payload_hash: row.get(35)?,
                })
            },
        )
        .optional()
        .map_err(|_| corrupt())
}

fn load_decision_summaries(
    connection: &Connection,
    decision_id: Uuid,
) -> Result<Vec<DecisionCandidateSummaryV1>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT decision_id, candidate_id, rank_ordinal, candidate_model,
                    candidate_model_revision, cost_rank, learning_generation_id,
                    vector_space_id, partition_hash, partition_id, decoding_fingerprint,
                    top_k, radius, radius_bits, min_points, min_independent_roots,
                    min_effective_samples, min_effective_samples_bits, min_coverage,
                    min_coverage_bits, time_decay_half_life_seconds,
                    time_decay_half_life_seconds_bits, prior_success, prior_success_bits,
                    prior_failure, prior_failure_bits, familywise_credible_level,
                    familywise_credible_level_bits, candidate_alpha, candidate_alpha_bits,
                    promotion_lower_bound, promotion_lower_bound_bits,
                    returned_neighbor_count, within_radius_count, labeled_point_count,
                    attempted_root_count, labeled_root_count, selected_root_count,
                    coverage, coverage_bits, sum_weight, sum_weight_bits,
                    sum_weighted_label, sum_weighted_label_bits, sum_squared_weight,
                    sum_squared_weight_bits, p_hat, p_hat_bits, effective_sample_size,
                    effective_sample_size_bits, beta_alpha, beta_alpha_bits, beta_beta,
                    beta_beta_bits, lower_bound, lower_bound_bits, partition_gate_passed,
                    points_gate_passed, roots_gate_passed, coverage_gate_passed,
                    weight_gate_passed, effective_samples_gate_passed,
                    beta_quantile_gate_passed, lower_bound_gate_passed, terminal_reason,
                    neighbor_count, neighbor_aggregate_hash, canonical_payload_hash
             FROM decision_candidate_summaries
             WHERE decision_id = ?1 ORDER BY rank_ordinal",
        )
        .map_err(database_error)?;
    statement
        .query_map([decision_id.to_string()], |row| {
            Ok(DecisionCandidateSummaryV1 {
                decision_id: row_uuid(row, 0)?,
                candidate_id: row.get(1)?,
                rank_ordinal: row_usize(row, 2)?,
                candidate_model: row.get(3)?,
                candidate_model_revision: row.get(4)?,
                cost_rank: row_u32(row, 5)?,
                learning_generation_id: row_uuid(row, 6)?,
                vector_space_id: row.get(7)?,
                partition_hash: row.get(8)?,
                partition_id: row.get(9)?,
                decoding_fingerprint: row.get(10)?,
                top_k: row_usize(row, 11)?,
                radius: row_float64(row, 12, 13)?,
                min_points: row_usize(row, 14)?,
                min_independent_roots: row_usize(row, 15)?,
                min_effective_samples: row_float64(row, 16, 17)?,
                min_coverage: row_float64(row, 18, 19)?,
                time_decay_half_life_seconds: row_float64(row, 20, 21)?,
                prior_success: row_float64(row, 22, 23)?,
                prior_failure: row_float64(row, 24, 25)?,
                familywise_credible_level: row_float64(row, 26, 27)?,
                candidate_alpha: row_float64(row, 28, 29)?,
                promotion_lower_bound: row_float64(row, 30, 31)?,
                returned_neighbor_count: row_usize(row, 32)?,
                within_radius_count: row_usize(row, 33)?,
                labeled_point_count: row_usize(row, 34)?,
                attempted_root_count: row_usize(row, 35)?,
                labeled_root_count: row_usize(row, 36)?,
                selected_root_count: row_usize(row, 37)?,
                coverage: row_optional_float64(row, 38, 39)?,
                sum_weight: row_optional_float64(row, 40, 41)?,
                sum_weighted_label: row_optional_float64(row, 42, 43)?,
                sum_squared_weight: row_optional_float64(row, 44, 45)?,
                p_hat: row_optional_float64(row, 46, 47)?,
                effective_sample_size: row_optional_float64(row, 48, 49)?,
                beta_alpha: row_optional_float64(row, 50, 51)?,
                beta_beta: row_optional_float64(row, 52, 53)?,
                lower_bound: row_optional_float64(row, 54, 55)?,
                partition_gate_passed: row_optional_bool(row, 56)?,
                points_gate_passed: row_optional_bool(row, 57)?,
                roots_gate_passed: row_optional_bool(row, 58)?,
                coverage_gate_passed: row_optional_bool(row, 59)?,
                weight_gate_passed: row_optional_bool(row, 60)?,
                effective_samples_gate_passed: row_optional_bool(row, 61)?,
                beta_quantile_gate_passed: row_optional_bool(row, 62)?,
                lower_bound_gate_passed: row_optional_bool(row, 63)?,
                terminal_reason: DecisionCandidateReasonV1::parse(&row.get::<_, String>(64)?)
                    .map_err(|_| row_corrupt(64, Type::Text))?,
                neighbor_count: row_usize(row, 65)?,
                neighbor_aggregate_hash: row.get(66)?,
                canonical_payload_hash: row.get(67)?,
            })
        })
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| corrupt())
}

fn load_decision_neighbors(
    connection: &Connection,
    decision_id: Uuid,
) -> Result<Vec<DecisionNeighborV1>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT decision_id, neighbor_ordinal, candidate_id,
                    candidate_neighbor_ordinal, evidence_vector_link_id,
                    shadow_attempt_id, anchor_id, evaluation_id, learning_generation_id,
                    distance, distance_f32_bits, age_seconds, age_seconds_bits,
                    similarity_weight, similarity_weight_bits, time_weight,
                    time_weight_bits, final_weight, final_weight_bits, binary_label,
                    selected_for_root, root_group_ordinal, exclusion_reason,
                    canonical_payload_hash
             FROM decision_neighbors WHERE decision_id = ?1 ORDER BY neighbor_ordinal",
        )
        .map_err(database_error)?;
    statement
        .query_map([decision_id.to_string()], |row| {
            Ok(DecisionNeighborV1 {
                decision_id: row_uuid(row, 0)?,
                neighbor_ordinal: row_usize(row, 1)?,
                candidate_id: row.get(2)?,
                candidate_neighbor_ordinal: row_usize(row, 3)?,
                evidence_vector_link_id: row_uuid(row, 4)?,
                shadow_attempt_id: row_uuid(row, 5)?,
                anchor_id: row_uuid(row, 6)?,
                evaluation_id: row_optional_uuid(row, 7)?,
                learning_generation_id: row_uuid(row, 8)?,
                distance: row_float32(row, 9, 10)?,
                age_seconds: row_optional_float64(row, 11, 12)?,
                similarity_weight: row_optional_float64(row, 13, 14)?,
                time_weight: row_optional_float64(row, 15, 16)?,
                final_weight: row_optional_float64(row, 17, 18)?,
                binary_label: row
                    .get::<_, Option<String>>(19)?
                    .map(|value| {
                        DecisionBinaryLabelV1::parse(&value)
                            .map_err(|_| row_corrupt(19, Type::Text))
                    })
                    .transpose()?,
                selected_for_root: row_bool(row, 20)?,
                root_group_ordinal: row_usize(row, 21)?,
                exclusion_reason: DecisionNeighborExclusionReasonV1::parse(
                    &row.get::<_, String>(22)?,
                )
                .map_err(|_| row_corrupt(22, Type::Text))?,
                canonical_payload_hash: row.get(23)?,
            })
        })
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| corrupt())
}

fn row_uuid(row: &Row<'_>, index: usize) -> rusqlite::Result<Uuid> {
    let value = row.get::<_, String>(index)?;
    Uuid::parse_str(&value)
        .ok()
        .filter(|value| is_uuid_v7(*value))
        .ok_or_else(|| row_corrupt(index, Type::Text))
}

fn row_optional_uuid(row: &Row<'_>, index: usize) -> rusqlite::Result<Option<Uuid>> {
    row.get::<_, Option<String>>(index)?
        .map(|value| {
            Uuid::parse_str(&value)
                .ok()
                .filter(|value| is_uuid_v7(*value))
                .ok_or_else(|| row_corrupt(index, Type::Text))
        })
        .transpose()
}

fn row_usize(row: &Row<'_>, index: usize) -> rusqlite::Result<usize> {
    usize::try_from(row.get::<_, i64>(index)?).map_err(|_| row_corrupt(index, Type::Integer))
}

fn row_u64(row: &Row<'_>, index: usize) -> rusqlite::Result<u64> {
    u64::try_from(row.get::<_, i64>(index)?).map_err(|_| row_corrupt(index, Type::Integer))
}

fn row_u32(row: &Row<'_>, index: usize) -> rusqlite::Result<u32> {
    u32::try_from(row.get::<_, i64>(index)?).map_err(|_| row_corrupt(index, Type::Integer))
}

fn row_float64(
    row: &Row<'_>,
    value_index: usize,
    bits_index: usize,
) -> rusqlite::Result<AuditF64V1> {
    let value = row.get::<_, f64>(value_index)?;
    let bits = u64::from_ne_bytes(row.get::<_, i64>(bits_index)?.to_ne_bytes());
    stored_float64(value, bits).map_err(|_| row_corrupt(bits_index, Type::Integer))
}

fn row_optional_float64(
    row: &Row<'_>,
    value_index: usize,
    bits_index: usize,
) -> rusqlite::Result<Option<AuditF64V1>> {
    match (
        row.get::<_, Option<f64>>(value_index)?,
        row.get::<_, Option<i64>>(bits_index)?,
    ) {
        (Some(value), Some(bits)) => stored_float64(value, u64::from_ne_bytes(bits.to_ne_bytes()))
            .map(Some)
            .map_err(|_| row_corrupt(bits_index, Type::Integer)),
        (None, None) => Ok(None),
        _ => Err(row_corrupt(value_index, Type::Real)),
    }
}

fn row_float32(
    row: &Row<'_>,
    value_index: usize,
    bits_index: usize,
) -> rusqlite::Result<AuditF32V1> {
    let value = row.get::<_, f64>(value_index)?;
    let bits = u32::try_from(row.get::<_, i64>(bits_index)?)
        .map_err(|_| row_corrupt(bits_index, Type::Integer))?;
    let audited =
        AuditF32V1::from_bits(bits).map_err(|_| row_corrupt(bits_index, Type::Integer))?;
    if f64::from(audited.value()) == value {
        Ok(audited)
    } else {
        Err(row_corrupt(bits_index, Type::Integer))
    }
}

fn stored_float64(stored_value: f64, authoritative_bits: u64) -> Result<AuditF64V1, ()> {
    let audited = AuditF64V1::from_bits(authoritative_bits).map_err(|_| ())?;
    if audited.value() == stored_value {
        Ok(audited)
    } else {
        Err(())
    }
}

fn row_bool(row: &Row<'_>, index: usize) -> rusqlite::Result<bool> {
    match row.get::<_, i64>(index)? {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(row_corrupt(index, Type::Integer)),
    }
}

fn row_optional_bool(row: &Row<'_>, index: usize) -> rusqlite::Result<Option<bool>> {
    match row.get::<_, Option<i64>>(index)? {
        None => Ok(None),
        Some(0) => Ok(Some(false)),
        Some(1) => Ok(Some(true)),
        Some(_) => Err(row_corrupt(index, Type::Integer)),
    }
}

fn row_corrupt(index: usize, data_type: Type) -> SqliteError {
    SqliteError::FromSqlConversionFailure(
        index,
        data_type,
        Box::new(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "decision audit row failed canonical decoding",
        )),
    )
}

fn parse_uuid(value: &str) -> Result<Uuid, LedgerError> {
    Uuid::parse_str(value)
        .ok()
        .filter(|value| is_uuid_v7(*value))
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))
}

fn database_error(error: SqliteError) -> LedgerError {
    map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}

fn corrupt() -> LedgerError {
    LedgerErrorClass::CorruptDatabase.into()
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use nemo_relay_types::api::llm::LlmApiFamily;
    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::canonical_query::{CanonicalRoutingQueryV1, CanonicalTaskV1};
    use crate::config::{LearningConfig, RouterConfig, RouterMode};
    use crate::decision_audit::{
        DecisionCandidateInputV1, DecisionCandidateSummaryInputV1, DecisionNeighborInputV1,
        DecisionParentInputV1, PreparedCanonicalQueryV1,
    };
    use crate::embedding_identity::CANONICAL_ROUTING_QUERY_SCHEMA_V1;
    use crate::fingerprint::canonical_serialize_bytes;
    use crate::ledger::repository::tests::{config, database_path};
    use crate::ledger::repository::vector_catalog::{
        RoutingPartitionEnsure, RoutingPartitionEnsureAck, VectorMappingAuthority,
        ensure_routing_partition,
    };
    use crate::ledger::repository::{ActivatedLedger, LedgerRepository};
    use crate::ledger::writer::LedgerWriterOwner;
    use crate::routing_partition::{
        RoutingPartitionArtifactV1, RoutingPartitionBaseV1, RoutingPartitionInputV1,
        RoutingPartitionV1, build_routing_partition_base_v1, build_routing_partition_from_input_v1,
    };

    fn hash(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    pub(crate) fn activate(
        min_coverage: f64,
        max_evidence_records: u64,
    ) -> (TempDir, RouterConfig, ActivatedLedger) {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut config = config(&path, "decision-repository-tests");
        config.mode = RouterMode::Recommend;
        config.max_evidence_records = max_evidence_records;
        config.embedders[0].api_key_env = None;
        config.pools[0].learning = Some(LearningConfig {
            version: 1,
            embedder: "embedder-a".to_string(),
            top_k: Some(1),
            radius: Some(1.0),
            min_points: Some(1),
            min_independent_roots: Some(1),
            min_effective_samples: Some(1.0),
            min_coverage: Some(min_coverage),
            time_decay_half_life_seconds: Some(3_600.0),
            prior_success: Some(1.0),
            prior_failure: Some(1.0),
            familywise_credible_level: Some(0.95),
            promotion_lower_bound: Some(0.2),
            retention_lower_bound: None,
            holdout_probability: None,
            active_canary_fraction: None,
        });
        assert!(config.validate().is_empty(), "{:?}", config.validate());
        let activated = LedgerRepository::activate_at(&config, 0).unwrap();
        (temporary, config, activated)
    }

    fn prepared_query() -> PreparedCanonicalQueryV1 {
        let query = CanonicalRoutingQueryV1 {
            schema: CANONICAL_ROUTING_QUERY_SCHEMA_V1.to_string(),
            instructions: Vec::new(),
            current_task: CanonicalTaskV1 {
                text: "route this request".to_string(),
            },
            bounded_context: Vec::new(),
            tool_schema_fingerprint: hash('1'),
            response_schema_fingerprint: None,
            required_capabilities: Vec::new(),
            position_features: None,
        };
        let canonical_bytes = canonical_serialize_bytes(&query).unwrap();
        let canonical_query_hash = sha256_hex(&canonical_bytes);
        PreparedCanonicalQueryV1::from_artifact(CanonicalRoutingQueryArtifactV1 {
            query,
            canonical_bytes,
            canonical_query_hash,
        })
        .unwrap()
    }

    pub(crate) fn no_partition_audit(
        activated: &ActivatedLedger,
        config: &RouterConfig,
        primary_call_uuid: Uuid,
    ) -> DecisionAuditV1 {
        let pool_config = &config.pools[0];
        let pool_identity = activated.identity.pool(&pool_config.id).unwrap();
        let learning = pool_config
            .learning
            .as_ref()
            .and_then(LearningConfig::complete_policy)
            .unwrap();
        let vector_space_id = pool_identity
            .vector_space
            .as_ref()
            .unwrap()
            .vector_space_id
            .as_str()
            .to_string();
        let base = RoutingPartitionBaseV1 {
            tenant_policy_hash: hash('a'),
            agent_policy_hash: hash('b'),
            policy_version_id: pool_identity.policy_version_id.clone(),
            learning_generation_id: pool_identity.learning_generation_id,
            api_family: LlmApiFamily::OpenAIChatCompletions,
            transport_identity: "transport-v1".to_string(),
            anchor_model: pool_config.anchor_models[0].clone(),
            anchor_revision: pool_config.anchor_revision.clone(),
            evaluator_version: pool_config.judge.evaluator_version().unwrap(),
            vector_space_id: vector_space_id.clone(),
        };
        let base_artifact = build_routing_partition_base_v1(&base).unwrap();
        let candidate = &pool_config.candidates[0];
        let decoding_fingerprint = hash('d');
        let partition = build_routing_partition_from_input_v1(&RoutingPartitionInputV1 {
            base,
            candidate_id: candidate.id.clone(),
            candidate_model: candidate.model.clone(),
            candidate_model_revision: candidate.model_revision.clone(),
            decoding_fingerprint: decoding_fingerprint.clone(),
        })
        .unwrap();
        let candidate_set = build_candidate_set_from_members_v1(&[CandidateSetMemberInputV1 {
            candidate_id: candidate.id.clone(),
            model: candidate.model.clone(),
            model_revision: candidate.model_revision.clone(),
            cost_rank: candidate.cost_rank,
        }])
        .unwrap();
        let prepared_query = prepared_query();
        let parent = DecisionParentInputV1 {
            decision_id: Uuid::now_v7(),
            project_uuid: activated.identity.project_uuid,
            process_instance_id: activated.identity.process_instance_id,
            config_generation_id: activated.identity.config_generation_id.clone(),
            policy_version_id: pool_identity.policy_version_id.clone(),
            learning_generation_id: pool_identity.learning_generation_id,
            pool_id: pool_config.id.clone(),
            candidate_id: None,
            primary_call_uuid,
            canonical_query_hash: prepared_query.canonical_query_hash().to_string(),
            partition_base_json: base_artifact.canonical_json,
            partition_base_hash: base_artifact.partition_base_hash,
            vector_space_id: vector_space_id.clone(),
            candidate_set_hash: candidate_set.candidate_set_hash,
            recommended_model: pool_config.anchor_models[0].clone(),
            recommended_model_revision: pool_config.anchor_revision.clone(),
            served_model: pool_config.anchor_models[0].clone(),
            served_model_revision: pool_config.anchor_revision.clone(),
            as_of_unix_ms: 100,
            decision_latency_ms: 2,
            final_reason: DecisionFinalReasonV1::NoPartition,
            created_at_unix_ms: 102,
        };
        let summary = DecisionCandidateSummaryInputV1 {
            candidate_id: candidate.id.clone(),
            rank_ordinal: 0,
            candidate_model: candidate.model.clone(),
            candidate_model_revision: candidate.model_revision.clone(),
            cost_rank: candidate.cost_rank,
            learning_generation_id: pool_identity.learning_generation_id,
            vector_space_id,
            partition_id: None,
            decoding_fingerprint,
            top_k: learning.top_k,
            radius: AuditF64V1::new(learning.radius).unwrap(),
            min_points: learning.min_points,
            min_independent_roots: learning.min_independent_roots,
            min_effective_samples: AuditF64V1::new(learning.min_effective_samples).unwrap(),
            min_coverage: AuditF64V1::new(learning.min_coverage).unwrap(),
            time_decay_half_life_seconds: AuditF64V1::new(learning.time_decay_half_life_seconds)
                .unwrap(),
            prior_success: AuditF64V1::new(learning.prior_success).unwrap(),
            prior_failure: AuditF64V1::new(learning.prior_failure).unwrap(),
            familywise_credible_level: AuditF64V1::new(learning.familywise_credible_level).unwrap(),
            candidate_alpha: AuditF64V1::new(1.0 - learning.familywise_credible_level).unwrap(),
            promotion_lower_bound: AuditF64V1::new(learning.promotion_lower_bound).unwrap(),
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
            partition_gate_passed: Some(false),
            points_gate_passed: None,
            roots_gate_passed: None,
            coverage_gate_passed: None,
            weight_gate_passed: None,
            effective_samples_gate_passed: None,
            beta_quantile_gate_passed: None,
            lower_bound_gate_passed: None,
            terminal_reason: DecisionCandidateReasonV1::NoPartition,
        };
        DecisionAuditV1::new(
            parent,
            vec![DecisionCandidateInputV1 {
                summary,
                partition_artifact: partition,
                neighbors: Vec::new(),
            }],
            prepared_query,
        )
        .unwrap()
    }

    pub(crate) fn existing_neighbor_audit(
        activated: &ActivatedLedger,
        config: &RouterConfig,
        primary_call_uuid: Uuid,
        evidence_vector_link_id: Uuid,
        created_at_unix_ms: i64,
    ) -> DecisionAuditV1 {
        let (
            shadow_attempt_id,
            anchor_id,
            learning_generation_id,
            vector_space_id,
            partition_id,
            canonical_query_hash,
        ) = activated
            .repository
            .connection
            .query_row(
                "SELECT shadow_attempt_id, anchor_id, learning_generation_id,
                        vector_space_id, partition_id, canonical_query_hash
                 FROM evidence_vector_links WHERE evidence_vector_link_id = ?1",
                [evidence_vector_link_id.to_string()],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, i64>(4)?,
                        row.get::<_, String>(5)?,
                    ))
                },
            )
            .unwrap();
        let (partition_hash, canonical_partition_json) = activated
            .repository
            .connection
            .query_row(
                "SELECT partition_hash, canonical_partition_json
                 FROM routing_partitions WHERE partition_id = ?1",
                [partition_id],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .unwrap();
        let partition: RoutingPartitionV1 =
            serde_json::from_str(&canonical_partition_json).unwrap();
        let partition_artifact = RoutingPartitionArtifactV1 {
            partition: partition.clone(),
            canonical_json: canonical_partition_json,
            partition_hash: partition_hash.clone(),
        };
        let base_artifact = build_routing_partition_base_v1(&RoutingPartitionBaseV1 {
            tenant_policy_hash: partition.tenant_policy_hash.clone(),
            agent_policy_hash: partition.agent_policy_hash.clone(),
            policy_version_id: partition.policy_version_id.clone(),
            learning_generation_id: partition.learning_generation_id,
            api_family: partition.api_family,
            transport_identity: partition.transport_identity.clone(),
            anchor_model: partition.anchor_model.clone(),
            anchor_revision: partition.anchor_revision.clone(),
            evaluator_version: partition.evaluator_version.clone(),
            vector_space_id: partition.vector_space_id.clone(),
        })
        .unwrap();
        let canonical_query_json = activated
            .repository
            .connection
            .query_row(
                "SELECT canonical_query_json FROM canonical_routing_queries
                 WHERE canonical_query_hash = ?1",
                [&canonical_query_hash],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let prepared_query =
            PreparedCanonicalQueryV1::from_artifact(CanonicalRoutingQueryArtifactV1 {
                query: serde_json::from_str::<CanonicalRoutingQueryV1>(&canonical_query_json)
                    .unwrap(),
                canonical_bytes: canonical_query_json.as_bytes().to_vec(),
                canonical_query_hash: canonical_query_hash.clone(),
            })
            .unwrap();
        let pool_config = &config.pools[0];
        let candidate = pool_config
            .candidates
            .iter()
            .find(|candidate| candidate.id == partition.candidate_id)
            .unwrap();
        let learning = pool_config
            .learning
            .as_ref()
            .and_then(LearningConfig::complete_policy)
            .unwrap();
        let candidate_set = build_candidate_set_from_members_v1(&[CandidateSetMemberInputV1 {
            candidate_id: candidate.id.clone(),
            model: candidate.model.clone(),
            model_revision: candidate.model_revision.clone(),
            cost_rank: candidate.cost_rank,
        }])
        .unwrap();
        let parent = DecisionParentInputV1 {
            decision_id: Uuid::now_v7(),
            project_uuid: activated.identity.project_uuid,
            process_instance_id: activated.identity.process_instance_id,
            config_generation_id: activated.identity.config_generation_id.clone(),
            policy_version_id: partition.policy_version_id.clone(),
            learning_generation_id: Uuid::parse_str(&learning_generation_id).unwrap(),
            pool_id: pool_config.id.clone(),
            candidate_id: None,
            primary_call_uuid,
            canonical_query_hash,
            partition_base_json: base_artifact.canonical_json,
            partition_base_hash: base_artifact.partition_base_hash,
            vector_space_id: vector_space_id.clone(),
            candidate_set_hash: candidate_set.candidate_set_hash,
            recommended_model: pool_config.anchor_models[0].clone(),
            recommended_model_revision: pool_config.anchor_revision.clone(),
            served_model: pool_config.anchor_models[0].clone(),
            served_model_revision: pool_config.anchor_revision.clone(),
            as_of_unix_ms: created_at_unix_ms - 1,
            decision_latency_ms: 1,
            final_reason: DecisionFinalReasonV1::SparsePoints,
            created_at_unix_ms,
        };
        let summary = DecisionCandidateSummaryInputV1 {
            candidate_id: candidate.id.clone(),
            rank_ordinal: 0,
            candidate_model: candidate.model.clone(),
            candidate_model_revision: candidate.model_revision.clone(),
            cost_rank: candidate.cost_rank,
            learning_generation_id: Uuid::parse_str(&learning_generation_id).unwrap(),
            vector_space_id,
            partition_id: Some(partition_id),
            decoding_fingerprint: partition.decoding_fingerprint,
            top_k: learning.top_k,
            radius: AuditF64V1::new(learning.radius).unwrap(),
            min_points: learning.min_points,
            min_independent_roots: learning.min_independent_roots,
            min_effective_samples: AuditF64V1::new(learning.min_effective_samples).unwrap(),
            min_coverage: AuditF64V1::new(learning.min_coverage).unwrap(),
            time_decay_half_life_seconds: AuditF64V1::new(learning.time_decay_half_life_seconds)
                .unwrap(),
            prior_success: AuditF64V1::new(learning.prior_success).unwrap(),
            prior_failure: AuditF64V1::new(learning.prior_failure).unwrap(),
            familywise_credible_level: AuditF64V1::new(learning.familywise_credible_level).unwrap(),
            candidate_alpha: AuditF64V1::new(1.0 - learning.familywise_credible_level).unwrap(),
            promotion_lower_bound: AuditF64V1::new(learning.promotion_lower_bound).unwrap(),
            returned_neighbor_count: 1,
            within_radius_count: 1,
            labeled_point_count: 0,
            attempted_root_count: 1,
            labeled_root_count: 0,
            selected_root_count: 0,
            coverage: Some(AuditF64V1::new(0.0).unwrap()),
            sum_weight: None,
            sum_weighted_label: None,
            sum_squared_weight: None,
            p_hat: None,
            effective_sample_size: None,
            beta_alpha: None,
            beta_beta: None,
            lower_bound: None,
            partition_gate_passed: Some(true),
            points_gate_passed: Some(false),
            roots_gate_passed: None,
            coverage_gate_passed: None,
            weight_gate_passed: None,
            effective_samples_gate_passed: None,
            beta_quantile_gate_passed: None,
            lower_bound_gate_passed: None,
            terminal_reason: DecisionCandidateReasonV1::SparsePoints,
        };
        let neighbor = DecisionNeighborInputV1 {
            neighbor_ordinal: 0,
            candidate_id: candidate.id.clone(),
            candidate_neighbor_ordinal: 0,
            evidence_vector_link_id,
            shadow_attempt_id: Uuid::parse_str(&shadow_attempt_id).unwrap(),
            anchor_id: Uuid::parse_str(&anchor_id).unwrap(),
            evaluation_id: None,
            learning_generation_id: Uuid::parse_str(&learning_generation_id).unwrap(),
            distance: AuditF32V1::new(0.0).unwrap(),
            age_millis: None,
            similarity_weight: None,
            time_weight: None,
            final_weight: None,
            binary_label: None,
            selected_for_root: false,
            root_group_ordinal: 0,
            exclusion_reason: DecisionNeighborExclusionReasonV1::IneligibleQuality,
        };
        DecisionAuditV1::new(
            parent,
            vec![DecisionCandidateInputV1 {
                summary,
                partition_artifact,
                neighbors: vec![neighbor],
            }],
            prepared_query,
        )
        .unwrap()
    }

    pub(crate) fn insert_test_decision_graphs(
        connection: &mut Connection,
        audits: &[DecisionAuditV1],
    ) {
        let transaction = connection.transaction().unwrap();
        for audit in audits {
            assert_eq!(
                insert_decision_graph(&transaction, audit).unwrap(),
                DecisionAuditAck::Applied
            );
        }
        transaction.commit().unwrap();
    }

    fn missing_evidence_audit(
        activated: &mut ActivatedLedger,
        config: &RouterConfig,
    ) -> (DecisionAuditV1, Uuid) {
        let seed = no_partition_audit(activated, config, Uuid::now_v7());
        let seed_parent = &seed.parent;
        let seed_summary = &seed.summaries[0];
        let partition_artifact = seed.partition_artifacts[0].clone();
        let transaction = activated.repository.connection.transaction().unwrap();
        let partition_id = match ensure_routing_partition(
            &transaction,
            &RoutingPartitionEnsure {
                mapping: VectorMappingAuthority {
                    project_uuid: seed_parent.project_uuid,
                    config_generation_id: seed_parent.config_generation_id.clone(),
                    pool_id: seed_parent.pool_id.clone(),
                    mapping_policy_version_id: seed_parent.policy_version_id.clone(),
                },
                artifact: partition_artifact.clone(),
                created_at_unix_ms: 90,
            },
        )
        .unwrap()
        {
            RoutingPartitionEnsureAck::Applied(snapshot)
            | RoutingPartitionEnsureAck::AlreadyExists(snapshot) => snapshot.partition_id.value(),
            acknowledgement => panic!("unexpected partition acknowledgement: {acknowledgement:?}"),
        };
        transaction.commit().unwrap();

        let anchor_id = Uuid::now_v7();
        let parent = DecisionParentInputV1 {
            decision_id: seed_parent.decision_id,
            project_uuid: seed_parent.project_uuid,
            process_instance_id: seed_parent.process_instance_id,
            config_generation_id: seed_parent.config_generation_id.clone(),
            policy_version_id: seed_parent.policy_version_id.clone(),
            learning_generation_id: seed_parent.learning_generation_id,
            pool_id: seed_parent.pool_id.clone(),
            candidate_id: None,
            primary_call_uuid: seed_parent.primary_call_uuid,
            canonical_query_hash: seed_parent.canonical_query_hash.clone(),
            partition_base_json: seed_parent.partition_base_json.clone(),
            partition_base_hash: seed_parent.partition_base_hash.clone(),
            vector_space_id: seed_parent.vector_space_id.clone(),
            candidate_set_hash: seed_parent.candidate_set_hash.clone(),
            recommended_model: seed_parent.recommended_model.clone(),
            recommended_model_revision: seed_parent.recommended_model_revision.clone(),
            served_model: seed_parent.served_model.clone(),
            served_model_revision: seed_parent.served_model_revision.clone(),
            as_of_unix_ms: seed_parent.as_of_unix_ms,
            decision_latency_ms: seed_parent.decision_latency_ms,
            final_reason: DecisionFinalReasonV1::SparsePoints,
            created_at_unix_ms: seed_parent.created_at_unix_ms,
        };
        let summary = DecisionCandidateSummaryInputV1 {
            candidate_id: seed_summary.candidate_id.clone(),
            rank_ordinal: seed_summary.rank_ordinal,
            candidate_model: seed_summary.candidate_model.clone(),
            candidate_model_revision: seed_summary.candidate_model_revision.clone(),
            cost_rank: seed_summary.cost_rank,
            learning_generation_id: seed_summary.learning_generation_id,
            vector_space_id: seed_summary.vector_space_id.clone(),
            partition_id: Some(partition_id),
            decoding_fingerprint: seed_summary.decoding_fingerprint.clone(),
            top_k: seed_summary.top_k,
            radius: seed_summary.radius,
            min_points: seed_summary.min_points,
            min_independent_roots: seed_summary.min_independent_roots,
            min_effective_samples: seed_summary.min_effective_samples,
            min_coverage: seed_summary.min_coverage,
            time_decay_half_life_seconds: seed_summary.time_decay_half_life_seconds,
            prior_success: seed_summary.prior_success,
            prior_failure: seed_summary.prior_failure,
            familywise_credible_level: seed_summary.familywise_credible_level,
            candidate_alpha: seed_summary.candidate_alpha,
            promotion_lower_bound: seed_summary.promotion_lower_bound,
            returned_neighbor_count: 1,
            within_radius_count: 1,
            labeled_point_count: 0,
            attempted_root_count: 1,
            labeled_root_count: 0,
            selected_root_count: 0,
            coverage: Some(AuditF64V1::new(0.0).unwrap()),
            sum_weight: None,
            sum_weighted_label: None,
            sum_squared_weight: None,
            p_hat: None,
            effective_sample_size: None,
            beta_alpha: None,
            beta_beta: None,
            lower_bound: None,
            partition_gate_passed: Some(true),
            points_gate_passed: Some(false),
            roots_gate_passed: None,
            coverage_gate_passed: None,
            weight_gate_passed: None,
            effective_samples_gate_passed: None,
            beta_quantile_gate_passed: None,
            lower_bound_gate_passed: None,
            terminal_reason: DecisionCandidateReasonV1::SparsePoints,
        };
        let neighbor = DecisionNeighborInputV1 {
            neighbor_ordinal: 0,
            candidate_id: seed_summary.candidate_id.clone(),
            candidate_neighbor_ordinal: 0,
            evidence_vector_link_id: Uuid::now_v7(),
            shadow_attempt_id: Uuid::now_v7(),
            anchor_id,
            evaluation_id: None,
            learning_generation_id: seed_summary.learning_generation_id,
            distance: AuditF32V1::new(0.1).unwrap(),
            age_millis: None,
            similarity_weight: None,
            time_weight: None,
            final_weight: None,
            binary_label: None,
            selected_for_root: false,
            root_group_ordinal: 0,
            exclusion_reason: DecisionNeighborExclusionReasonV1::IneligibleQuality,
        };
        let prepared_query =
            PreparedCanonicalQueryV1::from_artifact(seed.prepared_query.artifact().clone())
                .unwrap();
        (
            DecisionAuditV1::new(
                parent,
                vec![DecisionCandidateInputV1 {
                    summary,
                    partition_artifact,
                    neighbors: vec![neighbor],
                }],
                prepared_query,
            )
            .unwrap(),
            anchor_id,
        )
    }

    fn count(connection: &Connection, table: &str) -> i64 {
        connection
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap()
    }

    #[test]
    fn decision_append_reloads_exactly_and_retry_precedes_capacity() {
        let (_temporary, config, mut activated) = activate(0.0, 1);
        let audit = no_partition_audit(&activated, &config, Uuid::now_v7());
        let decision_id = audit.parent.decision_id;
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&audit, 1, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::Applied
        );
        let loaded = load_decision_graph(&activated.repository.connection, decision_id)
            .unwrap()
            .unwrap();
        assert!(audit.persisted_eq(loaded.graph).unwrap());
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&audit, 1, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::AlreadyApplied
        );
        let second = no_partition_audit(&activated, &config, Uuid::now_v7());
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&second, 1, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::RetentionRequired
        );
        assert_eq!(count(&activated.repository.connection, "decisions"), 1);
        assert_eq!(
            count(
                &activated.repository.connection,
                "decision_candidate_summaries"
            ),
            1
        );
        assert_eq!(
            count(&activated.repository.connection, "decision_neighbors"),
            0
        );
    }

    #[test]
    fn duplicate_primary_call_conflicts_without_partial_graph() {
        let (_temporary, config, mut activated) = activate(0.0, 10);
        let primary_call_uuid = Uuid::now_v7();
        let first = no_partition_audit(&activated, &config, primary_call_uuid);
        let second = no_partition_audit(&activated, &config, primary_call_uuid);
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&first, 10, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&second, 10, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::Conflict
        );
        assert_eq!(count(&activated.repository.connection, "decisions"), 1);
        assert_eq!(count(&activated.repository.connection, "health_events"), 1);
    }

    #[test]
    fn transaction_start_fence_prevents_all_mutation() {
        let (_temporary, config, mut activated) = activate(0.0, 10);
        let audit = no_partition_audit(&activated, &config, Uuid::now_v7());
        assert_eq!(
            activated
                .repository
                .record_decision_audit_with_start_check::<()>(&audit, 10, Uuid::now_v7(), || None,)
                .unwrap(),
            DecisionAuditAck::TransactionNotStarted
        );
        assert_eq!(count(&activated.repository.connection, "decisions"), 0);
    }

    #[test]
    fn corrupted_child_is_not_an_exact_retry_and_appends_health() {
        let (_temporary, config, mut activated) = activate(0.0, 10);
        let audit = no_partition_audit(&activated, &config, Uuid::now_v7());
        let decision_id = audit.parent.decision_id;
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&audit, 10, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::Applied
        );
        activated
            .repository
            .connection
            .execute(
                "UPDATE decision_candidate_summaries
                 SET candidate_model = 'tampered-model'
                 WHERE decision_id = ?1",
                [decision_id.to_string()],
            )
            .unwrap();
        assert_eq!(
            load_decision_graph(&activated.repository.connection, decision_id)
                .unwrap_err()
                .class(),
            LedgerErrorClass::CorruptDatabase
        );
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&audit, 10, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::Conflict
        );
        assert_eq!(count(&activated.repository.connection, "health_events"), 1);
    }

    #[test]
    fn signed_zero_float_bits_round_trip_through_sqlite_integer() {
        let (_temporary, config, mut activated) = activate(-0.0, 10);
        let audit = no_partition_audit(&activated, &config, Uuid::now_v7());
        let decision_id = audit.parent.decision_id;
        assert_eq!(audit.summaries[0].min_coverage.bits(), (-0.0_f64).to_bits());
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&audit, 10, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::Applied
        );
        let loaded = load_decision_graph(&activated.repository.connection, decision_id)
            .unwrap()
            .unwrap();
        assert_eq!(
            loaded.graph.summaries[0].min_coverage.bits(),
            (-0.0_f64).to_bits()
        );
    }

    #[test]
    fn stale_learning_generation_is_refused_before_mutation() {
        let (_temporary, config, mut activated) = activate(0.0, 10);
        let audit = no_partition_audit(&activated, &config, Uuid::now_v7());
        activated
            .repository
            .reset_pool("pool-a", "test", "generation-change")
            .unwrap();
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&audit, 10, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::AuthorityChanged
        );
        assert_eq!(count(&activated.repository.connection, "decisions"), 0);
    }

    #[test]
    fn missing_evidence_and_retiring_source_are_distinct_expected_races() {
        let (_temporary, config, mut activated) = activate(0.0, 10);
        let (missing, _) = missing_evidence_audit(&mut activated, &config);
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&missing, 10, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::EvidenceChanged
        );
        assert_eq!(count(&activated.repository.connection, "decisions"), 0);

        let (retiring, anchor_id) = missing_evidence_audit(&mut activated, &config);
        activated
            .repository
            .connection
            .execute_batch("PRAGMA foreign_keys = OFF")
            .unwrap();
        activated
            .repository
            .connection
            .execute(
                "INSERT INTO decision_retiring_anchors (
                    anchor_id, project_uuid, first_retention_batch_id,
                    age_expired, count_excess, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, 1, 0, 100, ?4)",
                params![
                    anchor_id.to_string(),
                    activated.identity.project_uuid.to_string(),
                    Uuid::now_v7().to_string(),
                    hash('f'),
                ],
            )
            .unwrap();
        activated
            .repository
            .connection
            .execute_batch("PRAGMA foreign_keys = ON")
            .unwrap();
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&retiring, 10, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::SourceRetiring
        );
        assert_eq!(count(&activated.repository.connection, "decisions"), 0);
    }

    #[test]
    fn insertion_fault_rolls_back_query_parent_and_children() {
        let (_temporary, config, mut activated) = activate(0.0, 10);
        let audit = no_partition_audit(&activated, &config, Uuid::now_v7());
        activated
            .repository
            .connection
            .execute_batch(
                "CREATE TRIGGER decision_summary_fault
                 BEFORE INSERT ON decision_candidate_summaries
                 BEGIN
                    SELECT RAISE(ABORT, 'injected decision summary fault');
                 END;",
            )
            .unwrap();
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&audit, 10, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::Conflict
        );
        assert_eq!(count(&activated.repository.connection, "decisions"), 0);
        assert_eq!(
            count(
                &activated.repository.connection,
                "decision_candidate_summaries"
            ),
            0
        );
        assert_eq!(
            count(
                &activated.repository.connection,
                "canonical_routing_queries"
            ),
            0
        );
        assert_eq!(count(&activated.repository.connection, "health_events"), 1);
    }

    #[tokio::test]
    async fn writer_arc_round_trip_supports_exact_lost_ack_retry() {
        let (_temporary, config, activated) = activate(0.0, 10);
        let audit = Arc::new(no_partition_audit(&activated, &config, Uuid::now_v7()));
        let (mut owner, client) = LedgerWriterOwner::start(activated.repository, 2).unwrap();
        let deadline = Instant::now() + Duration::from_secs(3);
        assert_eq!(
            client
                .record_decision_audit_until(audit.clone(), 10, Uuid::now_v7(), deadline)
                .await
                .unwrap(),
            DecisionAuditAck::Applied
        );
        assert_eq!(
            client
                .record_decision_audit_until(audit, 10, Uuid::now_v7(), deadline)
                .await
                .unwrap(),
            DecisionAuditAck::AlreadyApplied
        );
        owner
            .drain_until(Instant::now() + Duration::from_secs(3))
            .await
            .unwrap();
    }
}
