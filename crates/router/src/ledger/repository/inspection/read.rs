// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Consistent typed reads over one verified Router ledger snapshot.

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde_json::json;
use uuid::Uuid;

use super::{InspectionAuthoritySnapshot, load_inspection_authority_in_transaction};
use crate::config::{PoolConfig, RouterConfig, RouterMode};
use crate::control::{ControlMutationResult, ControlOperation, RouterControlError};
use crate::decision_audit::VerifiedStoredDecisionGraphV1;
use crate::inspection::content::inspect_content;
use crate::inspection::{
    ActiveDecisionExposureV1, CandidateSummaryV1, ContentPolicy, ControlStatusV1,
    DECISION_EXPOSURE_SCHEMA_V1,
    DecisionCandidateSummaryV1 as InspectionDecisionCandidateSummaryV1, DecisionDetailV1,
    DecisionExposureV1, DecisionFilterV1, DecisionNeighborV1 as InspectionDecisionNeighborV1,
    DecisionSummaryV1, EvidenceDetailV1, EvidenceFilterV1, EvidenceSummaryV1, FreshnessStatusV1,
    HealthEventV1, HealthSummaryV1, InspectionError, LeaseStatusV1, MigrationSummaryV1,
    OperatorHistoryEntryV1, OperatorHistoryKindV1, OutcomeFilterV1, OutcomeSummaryV1,
    OverviewDecisionCountsV1, OverviewExposureCountsV1, OverviewOutcomeCountsV1,
    POOL_DETAIL_SCHEMA_V1, PoolConcurrencyDetailV1, PoolDetailV1, PoolLearningPolicyV1,
    PoolSelectorDetailV1, PoolSummaryV1, PoolSupportPrerequisitesV1, PoolSupportV1, QueueStatusV1,
    VectorIndexStatusV1,
};
use crate::ledger::migrations::MIGRATIONS;
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::ledger::repository::active::load_verified_outcome;
use crate::ledger::repository::active_decision::load_verified_active_decision_facts;
use crate::ledger::repository::control::load_verified_control_history_page;
use crate::ledger::repository::decision::{load_decision_graph, load_verified_decision_parent};
use crate::ledger::repository::inspection::operator::load_verified_operator_history_entry;
use crate::ledger::repository::materialization::{
    VerifiedVectorLinkSourceLoad, load_verified_vector_link_source,
};
use crate::ledger::repository::vector_index::{
    GenerationObjectsStatus, VectorIndexManifestState, load_validated_manifest,
    verify_generation_objects,
};
use crate::ledger::repository::vector_registry::FrozenPoolVectorAuthority;
use crate::sqlite_vec_schema::VectorIndexGeneration;
use crate::vector::{VectorRecordId, VectorSpaceId};

/// Current authority plus aggregate status derived in the same read transaction.
pub(crate) struct StatusInspectionSnapshot {
    pub(crate) authority: InspectionAuthoritySnapshot,
    pub(crate) queues: QueueStatusV1,
    pub(crate) leases: LeaseStatusV1,
    pub(crate) vector_index: VectorIndexStatusV1,
    pub(crate) freshness: FreshnessStatusV1,
    pub(crate) health: HealthSummaryV1,
}

/// Current authority, status, and fixed-window aggregates from one transaction.
pub(crate) struct OverviewInspectionSnapshot {
    pub(crate) authority: InspectionAuthoritySnapshot,
    pub(crate) queues: QueueStatusV1,
    pub(crate) leases: LeaseStatusV1,
    pub(crate) vector_index: VectorIndexStatusV1,
    pub(crate) freshness: FreshnessStatusV1,
    pub(crate) health: HealthSummaryV1,
    pub(crate) decisions: OverviewDecisionCountsV1,
    pub(crate) exposures: OverviewExposureCountsV1,
    pub(crate) outcomes: OverviewOutcomeCountsV1,
}

/// One lexical pool page and its verified authority.
pub(crate) struct PoolPageRead {
    pub(crate) authority: InspectionAuthoritySnapshot,
    pub(crate) items: Vec<PoolSummaryV1>,
}

/// One newest-first health page and its frozen rowid boundary.
pub(crate) struct HealthPageRead {
    pub(crate) items: Vec<HealthEventV1>,
    pub(crate) maximum_insertion_sequence: u64,
}

/// One newest-first migration page and its frozen version boundary.
pub(crate) struct MigrationPageRead {
    pub(crate) items: Vec<MigrationSummaryV1>,
    pub(crate) maximum_insertion_sequence: u64,
}

/// One newest-first control page and its frozen history boundary.
pub(crate) struct ControlPageRead {
    pub(crate) items: Vec<OperatorHistoryEntryV1>,
    pub(crate) maximum_insertion_sequence: u64,
}

#[derive(Debug)]
struct StoredOperatorHistoryEntry {
    history_sequence: i64,
    project_uuid: String,
    audit_id: String,
    entry_kind: String,
    control_mutation_id: Option<String>,
    operator_mutation_id: Option<String>,
    created_at_unix_ms: i64,
}

#[derive(Debug, Clone, Copy)]
enum OperatorHistoryReference {
    Control(Uuid),
    Operator(Uuid),
}

/// One newest-first evidence page and its frozen rowid boundary.
pub(crate) struct EvidencePageRead {
    pub(crate) items: Vec<EvidenceSummaryV1>,
    pub(crate) maximum_insertion_sequence: u64,
}

/// One newest-first decision page and its frozen rowid boundary.
pub(crate) struct DecisionPageRead {
    pub(crate) items: Vec<DecisionSummaryV1>,
    pub(crate) maximum_insertion_sequence: u64,
}

/// One forward decision tail page and its next durable row boundary.
pub(crate) struct DecisionFollowPageRead {
    pub(crate) items: Vec<DecisionSummaryV1>,
    pub(crate) next_insertion_sequence: u64,
}

/// One newest-first Active outcome page and its frozen rowid boundary.
pub(crate) struct OutcomePageRead {
    pub(crate) items: Vec<OutcomeSummaryV1>,
    pub(crate) maximum_insertion_sequence: u64,
}

pub(crate) fn load_status_snapshot(
    connection: &Connection,
    config: &RouterConfig,
    snapshot_time_unix_ms: u64,
) -> Result<StatusInspectionSnapshot, LedgerError> {
    let transaction = read_transaction(connection)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let snapshot_time = to_i64(snapshot_time_unix_ms)?;
    let queues = load_queue_status(&transaction, config, authority.project_uuid)?;
    let leases = load_lease_status(&transaction, authority.project_uuid, snapshot_time)?;
    let vector_index = load_vector_index_status(&transaction, &authority)?;
    let freshness = load_freshness(&transaction, authority.project_uuid)?;
    let health = load_health_summary(&transaction, authority.project_uuid, None)?;
    commit(transaction)?;
    Ok(StatusInspectionSnapshot {
        authority,
        queues,
        leases,
        vector_index,
        freshness,
        health,
    })
}

pub(crate) fn load_overview_snapshot(
    connection: &Connection,
    config: &RouterConfig,
    window_start_unix_ms: u64,
    snapshot_time_unix_ms: u64,
) -> Result<OverviewInspectionSnapshot, LedgerError> {
    if window_start_unix_ms > snapshot_time_unix_ms {
        return Err(corrupt());
    }
    let transaction = read_transaction(connection)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let snapshot_time = to_i64(snapshot_time_unix_ms)?;
    let window_start = to_i64(window_start_unix_ms)?;
    let queues = load_queue_status(&transaction, config, authority.project_uuid)?;
    let leases = load_lease_status(&transaction, authority.project_uuid, snapshot_time)?;
    let vector_index = load_vector_index_status(&transaction, &authority)?;
    let freshness = load_freshness(&transaction, authority.project_uuid)?;
    let health = load_health_summary(&transaction, authority.project_uuid, None)?;
    let (decisions, exposures) = load_overview_decisions(
        &transaction,
        authority.project_uuid,
        window_start,
        snapshot_time,
    )?;
    let outcomes = load_overview_outcomes(
        &transaction,
        authority.project_uuid,
        window_start,
        snapshot_time,
    )?;
    commit(transaction)?;
    Ok(OverviewInspectionSnapshot {
        authority,
        queues,
        leases,
        vector_index,
        freshness,
        health,
        decisions,
        exposures,
        outcomes,
    })
}

pub(crate) fn load_pool_page(
    connection: &Connection,
    config: &RouterConfig,
    after_pool_id: Option<&str>,
    limit: usize,
) -> Result<PoolPageRead, LedgerError> {
    if limit == 0 {
        return Err(corrupt());
    }
    let transaction = read_transaction(connection)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let support = load_pool_support(&transaction, &authority)?;
    let mut pools = config.pools.iter().collect::<Vec<_>>();
    pools.sort_by(|left, right| left.id.cmp(&right.id));
    let mut items = Vec::with_capacity(limit.min(pools.len()));
    for pool in pools
        .into_iter()
        .filter(|pool| after_pool_id.is_none_or(|after| pool.id.as_str() > after))
        .take(limit)
    {
        items.push(load_pool_summary(&transaction, &authority, &support, pool)?);
    }
    commit(transaction)?;
    Ok(PoolPageRead { authority, items })
}

pub(crate) fn load_pool_detail(
    connection: &Connection,
    config: &RouterConfig,
    pool_id: &str,
) -> Result<Option<PoolDetailV1>, LedgerError> {
    let transaction = read_transaction(connection)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let support = load_pool_support(&transaction, &authority)?;
    let detail = config
        .pools
        .iter()
        .find(|pool| pool.id == pool_id)
        .map(|pool| {
            let summary = load_pool_summary(&transaction, &authority, &support, pool)?;
            inspection_pool_detail(pool, summary)
        })
        .transpose()?;
    commit(transaction)?;
    Ok(detail)
}

fn load_pool_summary(
    transaction: &Transaction<'_>,
    authority: &InspectionAuthoritySnapshot,
    support: &BTreeMap<String, PoolSupportV1>,
    pool: &PoolConfig,
) -> Result<PoolSummaryV1, LedgerError> {
    let policy_version_id = authority
        .policy_version_ids
        .get(&pool.id)
        .ok_or_else(corrupt)?
        .clone();
    let learning_generation_id = *authority
        .learning_generation_ids
        .get(&pool.id)
        .ok_or_else(corrupt)?;
    let controls = match authority.control_snapshot.as_ref() {
        Some(snapshot) => {
            let effective = snapshot.pools.get(&pool.id).ok_or_else(corrupt)?.effective;
            Some(ControlStatusV1 {
                control_generation: snapshot.control_generation,
                all: snapshot.all,
                effective,
            })
        }
        None => None,
    };
    let mut candidates = pool
        .candidates
        .iter()
        .map(|candidate| CandidateSummaryV1 {
            id: candidate.id.clone(),
            model: candidate.model.clone(),
            model_revision: candidate.model_revision.clone(),
            cost_rank: candidate.cost_rank,
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        left.cost_rank
            .cmp(&right.cost_rank)
            .then_with(|| left.id.cmp(&right.id))
    });
    let health = load_health_summary(transaction, authority.project_uuid, Some(&pool.id))?;
    Ok(PoolSummaryV1 {
        id: pool.id.clone(),
        api_family: pool.api_family,
        anchor_models: pool.anchor_models.clone(),
        anchor_revision: pool.anchor_revision.clone(),
        candidates,
        policy_version_id,
        learning_generation_id,
        active_canary_fraction: pool
            .learning
            .as_ref()
            .and_then(|learning| learning.active_canary_fraction),
        holdout_probability: pool
            .learning
            .as_ref()
            .and_then(|learning| learning.holdout_probability),
        support: support.get(&pool.id).cloned().unwrap_or_default(),
        controls,
        health,
    })
}

fn inspection_pool_detail(
    pool: &PoolConfig,
    summary: PoolSummaryV1,
) -> Result<PoolDetailV1, LedgerError> {
    let selector = PoolSelectorDetailV1 {
        tenant_ids: pool.selector.tenant_ids.clone(),
        agent_ids: pool.selector.agent_ids.clone(),
        owner_scope_types: pool.selector.owner_scope_types.clone(),
        metadata_equals: pool.selector.metadata_equals.clone(),
        scope_path_patterns: pool.selector.scope_path_patterns.clone(),
    };
    let concurrency = PoolConcurrencyDetailV1 {
        shadow: u64::try_from(pool.concurrency.shadow).map_err(|_| corrupt())?,
        judge: u64::try_from(pool.concurrency.judge).map_err(|_| corrupt())?,
        max_pending: u64::try_from(pool.concurrency.max_pending).map_err(|_| corrupt())?,
    };
    let learning = pool
        .learning
        .as_ref()
        .map(|learning| -> Result<PoolLearningPolicyV1, LedgerError> {
            Ok(PoolLearningPolicyV1 {
                version: learning.version,
                embedder: learning.embedder.clone(),
                top_k: learning
                    .top_k
                    .map(|value| u64::try_from(value).map_err(|_| corrupt()))
                    .transpose()?,
                radius: learning.radius,
                min_points: learning
                    .min_points
                    .map(|value| u64::try_from(value).map_err(|_| corrupt()))
                    .transpose()?,
                min_independent_roots: learning
                    .min_independent_roots
                    .map(|value| u64::try_from(value).map_err(|_| corrupt()))
                    .transpose()?,
                min_effective_samples: learning.min_effective_samples,
                min_coverage: learning.min_coverage,
                time_decay_half_life_seconds: learning.time_decay_half_life_seconds,
                prior_success: learning.prior_success,
                prior_failure: learning.prior_failure,
                familywise_credible_level: learning.familywise_credible_level,
                promotion_lower_bound: learning.promotion_lower_bound,
                retention_lower_bound: learning.retention_lower_bound,
                holdout_probability: learning.holdout_probability,
                active_canary_fraction: learning.active_canary_fraction,
            })
        })
        .transpose()?;
    let ready_evidence = pool
        .learning
        .as_ref()
        .and_then(|learning| learning.min_points)
        .map(|minimum| {
            u64::try_from(minimum)
                .map(|minimum| summary.support.ready_evidence >= minimum)
                .map_err(|_| corrupt())
        })
        .transpose()?;
    let independent_roots = pool
        .learning
        .as_ref()
        .and_then(|learning| learning.min_independent_roots)
        .map(|minimum| {
            u64::try_from(minimum)
                .map(|minimum| summary.support.independent_roots >= minimum)
                .map_err(|_| corrupt())
        })
        .transpose()?;
    let support_prerequisites = PoolSupportPrerequisitesV1 {
        ready_evidence,
        independent_roots,
        coverage: pool
            .learning
            .as_ref()
            .and_then(|learning| learning.min_coverage)
            .map(|minimum| {
                summary
                    .support
                    .coverage
                    .is_some_and(|value| value >= minimum)
            }),
        query_local_evaluation_required: pool
            .learning
            .as_ref()
            .is_some_and(|learning| learning.complete_policy().is_some()),
    };
    Ok(PoolDetailV1 {
        schema: POOL_DETAIL_SCHEMA_V1.into(),
        summary,
        sampling_probability: pool.sampling_probability,
        max_candidates_per_sample: u64::try_from(pool.max_candidates_per_sample)
            .map_err(|_| corrupt())?,
        selector,
        concurrency,
        learning,
        support_prerequisites,
    })
}

pub(crate) fn load_health_page(
    connection: &Connection,
    config: &RouterConfig,
    snapshot_time_unix_ms: u64,
    maximum_insertion_sequence: Option<u64>,
    after: Option<(u64, &str)>,
    limit: usize,
) -> Result<HealthPageRead, LedgerError> {
    if limit == 0 {
        return Err(corrupt());
    }
    let transaction = read_transaction(connection)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let maximum = match maximum_insertion_sequence {
        Some(value) => value,
        None => nonnegative_u64(
            transaction
                .query_row(
                    "SELECT coalesce(max(rowid), 0) FROM health_events",
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(database_error)?,
        )?,
    };
    let snapshot_time = to_i64(snapshot_time_unix_ms)?;
    let after_time = after.map(|value| to_i64(value.0)).transpose()?;
    let after_id = after.map(|value| value.1);
    let mut statement = transaction
        .prepare(
            "SELECT rowid, health_event_id, project_uuid, process_instance_id,
                    requested_anchor_id, requested_dependency_key_id,
                    anchor_id, dependency_key_id, stable_class, severity,
                    created_at_unix_ms, canonical_payload_hash
             FROM health_events
             WHERE project_uuid = ?1 AND rowid <= ?2 AND created_at_unix_ms <= ?3
               AND (?4 IS NULL OR created_at_unix_ms < ?4
                    OR (created_at_unix_ms = ?4 AND health_event_id < ?5))
             ORDER BY created_at_unix_ms DESC, health_event_id DESC
             LIMIT ?6",
        )
        .map_err(database_error)?;
    let stored = statement
        .query_map(
            params![
                authority.project_uuid.to_string(),
                to_i64(maximum)?,
                snapshot_time,
                after_time,
                after_id,
                i64::try_from(limit).map_err(|_| corrupt())?,
            ],
            |row| {
                Ok(StoredHealthEvent {
                    rowid: row.get(0)?,
                    health_event_id: row.get(1)?,
                    project_uuid: row.get(2)?,
                    process_instance_id: row.get(3)?,
                    requested_anchor_id: row.get(4)?,
                    requested_dependency_key_id: row.get(5)?,
                    anchor_id: row.get(6)?,
                    dependency_key_id: row.get(7)?,
                    stable_class: row.get(8)?,
                    severity: row.get(9)?,
                    created_at_unix_ms: row.get(10)?,
                    canonical_payload_hash: row.get(11)?,
                })
            },
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    let items = stored
        .into_iter()
        .map(|event| verify_health_event(event, authority.project_uuid, maximum))
        .collect::<Result<Vec<_>, _>>()?;
    commit(transaction)?;
    Ok(HealthPageRead {
        items,
        maximum_insertion_sequence: maximum,
    })
}

pub(crate) fn load_migration_page(
    connection: &Connection,
    config: &RouterConfig,
    snapshot_time_unix_ms: u64,
    maximum_insertion_sequence: Option<u64>,
    after: Option<(u64, &str)>,
    limit: usize,
) -> Result<MigrationPageRead, LedgerError> {
    if limit == 0 {
        return Err(corrupt());
    }
    let transaction = read_transaction(connection)?;
    let _authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let supported_maximum = MIGRATIONS
        .last()
        .and_then(|migration| u64::try_from(migration.version).ok())
        .ok_or_else(corrupt)?;
    let maximum = maximum_insertion_sequence.unwrap_or(supported_maximum);
    if maximum == 0 || maximum > supported_maximum {
        return Err(corrupt());
    }
    let snapshot_time = to_i64(snapshot_time_unix_ms)?;
    let mut stored_items = Vec::new();
    for migration in MIGRATIONS {
        if migration.version > to_i64(maximum)? {
            continue;
        }
        let stored = transaction
            .query_row(
                "SELECT name, checksum_sha256, applied_at_unix_ms
                 FROM schema_migrations WHERE version = ?1",
                [migration.version],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .optional()
            .map_err(database_error)?;
        let (name, sha256, applied_at) = stored.ok_or_else(corrupt)?;
        if name != migration.name
            || sha256 != migration.checksum_sha256
            || applied_at < 0
            || applied_at > snapshot_time
        {
            return Err(corrupt());
        }
        let id = migration_sort_id(migration.version)?;
        let created_at = nonnegative_u64(applied_at)?;
        stored_items.push((
            created_at,
            id,
            MigrationSummaryV1 {
                version: migration.version,
                name,
                sha256,
                applied_at_unix_ms: Some(created_at),
                verified: true,
            },
        ));
    }
    stored_items.sort_by(|left, right| right.0.cmp(&left.0).then_with(|| right.1.cmp(&left.1)));
    let items = stored_items
        .into_iter()
        .filter(|(created_at, id, _)| {
            after.is_none_or(|(after_time, after_id)| {
                *created_at < after_time || (*created_at == after_time && id.as_str() < after_id)
            })
        })
        .take(limit)
        .map(|(_, _, summary)| summary)
        .collect();
    commit(transaction)?;
    Ok(MigrationPageRead {
        items,
        maximum_insertion_sequence: maximum,
    })
}

pub(crate) fn load_control_page(
    connection: &Connection,
    config: &RouterConfig,
    snapshot_time_unix_ms: u64,
    maximum_insertion_sequence: Option<u64>,
    after: Option<(u64, &str)>,
    limit: usize,
) -> Result<ControlPageRead, LedgerError> {
    if limit == 0 {
        return Err(corrupt());
    }
    let transaction = read_transaction(connection)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    if authority.control_snapshot.is_none() {
        commit(transaction)?;
        return Ok(ControlPageRead {
            items: Vec::new(),
            maximum_insertion_sequence: 0,
        });
    }
    let actual_maximum = nonnegative_u64(
        transaction
            .query_row(
                "SELECT coalesce(max(history_sequence), 0)
                 FROM operator_history_entries WHERE project_uuid = ?1",
                [authority.project_uuid.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .map_err(database_error)?,
    )?;
    let maximum = maximum_insertion_sequence.unwrap_or(actual_maximum);
    if maximum > actual_maximum {
        return Err(corrupt());
    }
    verify_operator_history_coverage(&transaction, authority.project_uuid)?;
    let snapshot_time = to_i64(snapshot_time_unix_ms)?;
    let after = after
        .map(|(created_at, id)| to_i64(created_at).map(|created_at| (created_at, id)))
        .transpose()?;
    let after_time = after.map(|value| value.0);
    let after_id = after.map(|value| value.1);
    let mut statement = transaction
        .prepare(
            "SELECT history_sequence, project_uuid, audit_id, entry_kind,
                    control_mutation_id, operator_mutation_id, created_at_unix_ms
             FROM operator_history_entries
             WHERE project_uuid = ?1 AND history_sequence <= ?2
               AND created_at_unix_ms <= ?3
               AND (?4 IS NULL OR created_at_unix_ms < ?4
                    OR (created_at_unix_ms = ?4 AND audit_id < ?5))
             ORDER BY created_at_unix_ms DESC, audit_id DESC
             LIMIT ?6",
        )
        .map_err(database_error)?;
    let stored = statement
        .query_map(
            params![
                authority.project_uuid.to_string(),
                to_i64(maximum)?,
                snapshot_time,
                after_time,
                after_id,
                i64::try_from(limit).map_err(|_| corrupt())?,
            ],
            |row| {
                Ok(StoredOperatorHistoryEntry {
                    history_sequence: row.get(0)?,
                    project_uuid: row.get(1)?,
                    audit_id: row.get(2)?,
                    entry_kind: row.get(3)?,
                    control_mutation_id: row.get(4)?,
                    operator_mutation_id: row.get(5)?,
                    created_at_unix_ms: row.get(6)?,
                })
            },
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);

    let mut references = Vec::with_capacity(stored.len());
    let mut control_ids = Vec::new();
    for entry in &stored {
        let history_sequence = nonnegative_u64(entry.history_sequence)?;
        let audit_id = parse_uuid_v7(&entry.audit_id)?;
        if history_sequence == 0
            || history_sequence > maximum
            || entry.project_uuid != authority.project_uuid.to_string()
            || audit_id.to_string() != entry.audit_id
            || entry.created_at_unix_ms < 0
        {
            return Err(corrupt());
        }
        let reference = match (
            entry.entry_kind.as_str(),
            entry.control_mutation_id.as_deref(),
            entry.operator_mutation_id.as_deref(),
        ) {
            ("control", Some(control_id), None) if control_id == entry.audit_id => {
                control_ids.push(audit_id);
                OperatorHistoryReference::Control(audit_id)
            }
            ("operator", None, Some(operator_id)) if operator_id == entry.audit_id => {
                OperatorHistoryReference::Operator(audit_id)
            }
            _ => return Err(corrupt()),
        };
        references.push(reference);
    }

    let mut controls = BTreeMap::new();
    if !control_ids.is_empty() {
        let maximum_control_ordinal = nonnegative_u64(
            transaction
                .query_row(
                    "SELECT coalesce(max(receipt.history_ordinal), 0)
                     FROM operator_history_entries AS history
                     JOIN control_mutation_receipts AS receipt
                       ON receipt.project_uuid = history.project_uuid
                      AND receipt.mutation_id = history.control_mutation_id
                     WHERE history.project_uuid = ?1 AND history.history_sequence <= ?2",
                    params![authority.project_uuid.to_string(), to_i64(maximum)?],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(database_error)?,
        )?;
        if maximum_control_ordinal == 0 {
            return Err(corrupt());
        }
        let page = load_verified_control_history_page(
            &transaction,
            authority.project_uuid,
            snapshot_time,
            Some(maximum_control_ordinal),
            after,
            limit,
        )
        .map_err(map_control_error)?;
        controls.extend(
            page.entries
                .into_iter()
                .map(|entry| (entry.receipt.mutation_id, entry)),
        );
    }

    let mut items = Vec::with_capacity(stored.len());
    for (stored, reference) in stored.into_iter().zip(references) {
        let item = match reference {
            OperatorHistoryReference::Control(mutation_id) => {
                let entry = controls.remove(&mutation_id).ok_or_else(corrupt)?;
                let kind = match entry.receipt.operation {
                    ControlOperation::SetPaused { .. } => OperatorHistoryKindV1::Pause,
                    ControlOperation::SetForceAnchor { .. } => OperatorHistoryKindV1::ForceAnchor,
                };
                let result = match entry.receipt.result {
                    ControlMutationResult::Applied => "applied",
                    ControlMutationResult::NoOp => "no_op",
                    ControlMutationResult::Conflict => "conflict",
                };
                OperatorHistoryEntryV1 {
                    audit_id: entry.receipt.mutation_id,
                    kind,
                    result: result.into(),
                    control_generation: Some(entry.receipt.result_control_generation),
                    scope: Some(entry.receipt.scope),
                    prior_value: entry.prior_value,
                    new_value: entry.new_value,
                    prior_generations: BTreeMap::new(),
                    new_generations: BTreeMap::new(),
                    superseded_ids: Vec::new(),
                    actor: entry.receipt.actor,
                    reason: entry.receipt.reason,
                    created_at_unix_ms: entry.receipt.created_at_unix_ms,
                }
            }
            OperatorHistoryReference::Operator(mutation_id) => {
                load_verified_operator_history_entry(
                    &transaction,
                    authority.project_uuid,
                    mutation_id,
                )
                .map_err(map_operator_history_error)?
            }
        };
        if item.audit_id.to_string() != stored.audit_id
            || item.created_at_unix_ms
                != u64::try_from(stored.created_at_unix_ms).map_err(|_| corrupt())?
        {
            return Err(corrupt());
        }
        items.push(item);
    }
    commit(transaction)?;
    Ok(ControlPageRead {
        items,
        maximum_insertion_sequence: maximum,
    })
}

fn verify_operator_history_coverage(
    connection: &Connection,
    project_uuid: Uuid,
) -> Result<(), LedgerError> {
    let (history_count, control_count, operator_count) = connection
        .query_row(
            "SELECT
                (SELECT count(*) FROM operator_history_entries WHERE project_uuid = ?1),
                (SELECT count(*) FROM control_mutation_receipts WHERE project_uuid = ?1),
                (SELECT count(*) FROM operator_mutation_receipts WHERE project_uuid = ?1)",
            [project_uuid.to_string()],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .map_err(database_error)?;
    if history_count < 0
        || control_count < 0
        || operator_count < 0
        || control_count.checked_add(operator_count) != Some(history_count)
    {
        return Err(corrupt());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn load_evidence_page(
    connection: &Connection,
    config: &RouterConfig,
    content_policy: ContentPolicy,
    filter: &EvidenceFilterV1,
    snapshot_time_unix_ms: u64,
    maximum_insertion_sequence: Option<u64>,
    after: Option<(u64, &str)>,
    limit: usize,
) -> Result<EvidencePageRead, LedgerError> {
    if limit == 0 {
        return Err(corrupt());
    }
    let transaction = read_transaction(connection)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let maximum = match maximum_insertion_sequence {
        Some(value) => value,
        None => nonnegative_u64(
            transaction
                .query_row(
                    "SELECT coalesce(max(link.rowid), 0)
                     FROM evidence_vector_links AS link
                     JOIN routing_partitions AS partition
                       ON partition.partition_id = link.partition_id
                     WHERE partition.project_uuid = ?1",
                    [authority.project_uuid.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(database_error)?,
        )?,
    };
    let after_time = after
        .map(|(created_at, _)| to_i64(created_at))
        .transpose()?;
    let after_id = after.map(|(_, id)| id);
    let mut statement = transaction
        .prepare(
            "SELECT link.rowid, link.evidence_vector_link_id,
                    link.vector_space_id, link.created_at_unix_ms
             FROM evidence_vector_links AS link
             JOIN routing_partitions AS partition
               ON partition.partition_id = link.partition_id
             WHERE partition.project_uuid = ?1
               AND link.rowid <= ?2 AND link.created_at_unix_ms <= ?3
               AND (?4 IS NULL OR partition.pool_id = ?4)
               AND (?5 IS NULL OR partition.candidate_id = ?5)
               AND (?6 IS NULL OR link.terminal_class = ?6)
               AND (?7 IS NULL OR link.quality_label = ?7)
               AND (?8 IS NULL OR link.learning_generation_id = ?8)
               AND (?9 IS NULL OR link.created_at_unix_ms < ?9
                    OR (link.created_at_unix_ms = ?9
                        AND link.evidence_vector_link_id < ?10))
             ORDER BY link.created_at_unix_ms DESC, link.evidence_vector_link_id DESC
             LIMIT ?11",
        )
        .map_err(database_error)?;
    let selected = statement
        .query_map(
            params![
                authority.project_uuid.to_string(),
                to_i64(maximum)?,
                to_i64(snapshot_time_unix_ms)?,
                filter.pool_id.as_deref(),
                filter.candidate_id.as_deref(),
                filter.terminal_class.as_deref(),
                filter.quality_label.as_deref(),
                filter
                    .learning_generation_id
                    .map(|generation_id| generation_id.to_string()),
                after_time,
                after_id,
                i64::try_from(limit).map_err(|_| corrupt())?,
            ],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    let mut items = Vec::with_capacity(selected.len());
    for (rowid, evidence_id, vector_space_id, created_at_unix_ms) in selected {
        let rowid = nonnegative_u64(rowid)?;
        if rowid == 0 || rowid > maximum {
            return Err(corrupt());
        }
        let evidence_id = parse_uuid_v7(&evidence_id)?;
        let vector_space_id = VectorSpaceId::new(vector_space_id).map_err(|_| corrupt())?;
        let detail = verified_evidence_detail(
            &transaction,
            authority.project_uuid,
            evidence_id,
            &vector_space_id,
            content_policy,
        )?
        .ok_or_else(corrupt)?;
        if detail.summary.created_at_unix_ms != nonnegative_u64(created_at_unix_ms)? {
            return Err(corrupt());
        }
        items.push(detail.summary);
    }
    commit(transaction)?;
    Ok(EvidencePageRead {
        items,
        maximum_insertion_sequence: maximum,
    })
}

pub(crate) fn load_evidence_detail(
    connection: &Connection,
    config: &RouterConfig,
    content_policy: ContentPolicy,
    evidence_id: Uuid,
) -> Result<Option<EvidenceDetailV1>, LedgerError> {
    let transaction = read_transaction(connection)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let vector_space_id = transaction
        .query_row(
            "SELECT link.vector_space_id
             FROM evidence_vector_links AS link
             JOIN routing_partitions AS partition
               ON partition.partition_id = link.partition_id
             WHERE link.evidence_vector_link_id = ?1 AND partition.project_uuid = ?2",
            params![evidence_id.to_string(), authority.project_uuid.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .map(VectorSpaceId::new)
        .transpose()
        .map_err(|_| corrupt())?;
    let detail = match vector_space_id {
        Some(vector_space_id) => verified_evidence_detail(
            &transaction,
            authority.project_uuid,
            evidence_id,
            &vector_space_id,
            content_policy,
        )?,
        None => None,
    };
    commit(transaction)?;
    Ok(detail)
}

fn verified_evidence_detail(
    connection: &Connection,
    project_uuid: Uuid,
    evidence_id: Uuid,
    vector_space_id: &VectorSpaceId,
    content_policy: ContentPolicy,
) -> Result<Option<EvidenceDetailV1>, LedgerError> {
    let record_id = VectorRecordId::new(evidence_id).map_err(|_| corrupt())?;
    let source = match load_verified_vector_link_source(connection, vector_space_id, record_id)? {
        VerifiedVectorLinkSourceLoad::LinkMissing => return Ok(None),
        VerifiedVectorLinkSourceLoad::AuthorityMissing => return Err(corrupt()),
        VerifiedVectorLinkSourceLoad::Verified(source) => source,
    };
    if source.project_uuid != project_uuid || source.vector_space_id != *vector_space_id {
        return Err(corrupt());
    }
    let query = serde_json::to_value(&source.canonical_query.query).map_err(|_| corrupt())?;
    let content = inspect_content(&query, content_policy).map_err(|_| corrupt())?;
    let evaluation_id = source
        .evaluation
        .as_ref()
        .map(|evaluation| evaluation.evaluation_id);
    Ok(Some(EvidenceDetailV1 {
        summary: EvidenceSummaryV1 {
            evidence_id,
            pool_id: source.pool_id,
            candidate_id: source.partition.candidate_id.clone(),
            terminal_class: source.terminal_class.as_str().into(),
            quality_label: source.quality_label,
            canonical_query_hash: source.canonical_query_hash,
            learning_generation_id: source.learning_generation_id,
            vector_state: source.vector_state,
            created_at_unix_ms: nonnegative_u64(source.created_at_unix_ms)?,
            content,
        },
        partition: source.partition,
        anchor_id: source.anchor_id,
        shadow_attempt_id: source.shadow_attempt_id,
        evaluation_id,
        record_hash: source.record_hash,
    }))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn load_decision_page(
    connection: &Connection,
    config: &RouterConfig,
    filter: &DecisionFilterV1,
    snapshot_time_unix_ms: u64,
    maximum_insertion_sequence: Option<u64>,
    after: Option<(u64, &str)>,
    limit: usize,
) -> Result<DecisionPageRead, LedgerError> {
    if limit == 0 {
        return Err(corrupt());
    }
    let transaction = read_transaction(connection)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let maximum = match maximum_insertion_sequence {
        Some(value) => value,
        None => nonnegative_u64(
            transaction
                .query_row(
                    "SELECT coalesce(max(rowid), 0) FROM decisions WHERE project_uuid = ?1",
                    [authority.project_uuid.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(database_error)?,
        )?,
    };
    let after_time = after
        .map(|(created_at, _)| to_i64(created_at))
        .transpose()?;
    let after_id = after.map(|(_, id)| id);
    let mut statement = transaction
        .prepare(
            "SELECT rowid, decision_id, created_at_unix_ms
             FROM decisions
             WHERE project_uuid = ?1 AND rowid <= ?2 AND created_at_unix_ms <= ?3
               AND (?4 IS NULL OR pool_id = ?4)
               AND (?5 IS NULL OR mode = ?5)
               AND (?6 IS NULL OR candidate_id = ?6)
               AND (?7 IS NULL OR final_reason = ?7)
               AND (?8 IS NULL OR created_at_unix_ms < ?8
                    OR (created_at_unix_ms = ?8 AND decision_id < ?9))
             ORDER BY created_at_unix_ms DESC, decision_id DESC
             LIMIT ?10",
        )
        .map_err(database_error)?;
    let selected = statement
        .query_map(
            params![
                authority.project_uuid.to_string(),
                to_i64(maximum)?,
                to_i64(snapshot_time_unix_ms)?,
                filter.pool_id.as_deref(),
                filter.mode.as_deref(),
                filter.candidate_id.as_deref(),
                filter.final_reason.as_deref(),
                after_time,
                after_id,
                i64::try_from(limit).map_err(|_| corrupt())?,
            ],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    let mut items = Vec::with_capacity(selected.len());
    for (rowid, decision_id, created_at_unix_ms) in selected {
        let rowid = nonnegative_u64(rowid)?;
        if rowid == 0 || rowid > maximum {
            return Err(corrupt());
        }
        let decision_id = parse_uuid_v7(&decision_id)?;
        let detail = load_decision_graph(&transaction, decision_id)?
            .ok_or_else(corrupt)
            .and_then(|graph| inspection_decision_detail(graph, authority.project_uuid))?;
        if detail.summary.created_at_unix_ms != nonnegative_u64(created_at_unix_ms)? {
            return Err(corrupt());
        }
        items.push(detail.summary);
    }
    commit(transaction)?;
    Ok(DecisionPageRead {
        items,
        maximum_insertion_sequence: maximum,
    })
}

pub(crate) fn load_decision_follow_page(
    connection: &Connection,
    config: &RouterConfig,
    filter: &DecisionFilterV1,
    after_insertion_sequence: Option<u64>,
    limit: usize,
) -> Result<DecisionFollowPageRead, LedgerError> {
    if limit == 0 {
        return Err(corrupt());
    }
    let transaction = read_transaction(connection)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let project_uuid = authority.project_uuid.to_string();
    let maximum = nonnegative_u64(
        transaction
            .query_row(
                "SELECT coalesce(max(rowid), 0) FROM decisions WHERE project_uuid = ?1",
                [&project_uuid],
                |row| row.get::<_, i64>(0),
            )
            .map_err(database_error)?,
    )?;
    let fetch_limit = limit.checked_add(1).ok_or_else(corrupt)?;
    let selected = if let Some(after) = after_insertion_sequence {
        let mut statement = transaction
            .prepare(
                "SELECT rowid, decision_id, created_at_unix_ms
                 FROM decisions
                 WHERE project_uuid = ?1 AND rowid > ?2 AND rowid <= ?3
                   AND (?4 IS NULL OR pool_id = ?4)
                   AND (?5 IS NULL OR mode = ?5)
                   AND (?6 IS NULL OR candidate_id = ?6)
                   AND (?7 IS NULL OR final_reason = ?7)
                 ORDER BY rowid ASC
                 LIMIT ?8",
            )
            .map_err(database_error)?;
        statement
            .query_map(
                params![
                    project_uuid,
                    to_i64(after)?,
                    to_i64(maximum)?,
                    filter.pool_id.as_deref(),
                    filter.mode.as_deref(),
                    filter.candidate_id.as_deref(),
                    filter.final_reason.as_deref(),
                    i64::try_from(fetch_limit).map_err(|_| corrupt())?,
                ],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .map_err(database_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(database_error)?
    } else {
        let mut statement = transaction
            .prepare(
                "SELECT rowid, decision_id, created_at_unix_ms
                 FROM decisions
                 WHERE project_uuid = ?1 AND rowid <= ?2
                   AND (?3 IS NULL OR pool_id = ?3)
                   AND (?4 IS NULL OR mode = ?4)
                   AND (?5 IS NULL OR candidate_id = ?5)
                   AND (?6 IS NULL OR final_reason = ?6)
                 ORDER BY rowid DESC
                 LIMIT ?7",
            )
            .map_err(database_error)?;
        let mut selected = statement
            .query_map(
                params![
                    project_uuid,
                    to_i64(maximum)?,
                    filter.pool_id.as_deref(),
                    filter.mode.as_deref(),
                    filter.candidate_id.as_deref(),
                    filter.final_reason.as_deref(),
                    i64::try_from(limit).map_err(|_| corrupt())?,
                ],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                    ))
                },
            )
            .map_err(database_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(database_error)?;
        selected.reverse();
        selected
    };
    let has_more = after_insertion_sequence.is_some() && selected.len() > limit;
    let mut selected = selected;
    selected.truncate(limit);
    let next_insertion_sequence = if has_more {
        selected
            .last()
            .map(|(rowid, _, _)| nonnegative_u64(*rowid))
            .transpose()?
            .ok_or_else(corrupt)?
    } else {
        maximum.max(after_insertion_sequence.unwrap_or(0))
    };
    let mut items = Vec::with_capacity(selected.len());
    for (rowid, decision_id, created_at_unix_ms) in selected {
        let rowid = nonnegative_u64(rowid)?;
        if rowid == 0 || rowid > maximum {
            return Err(corrupt());
        }
        let decision_id = parse_uuid_v7(&decision_id)?;
        let detail = load_decision_graph(&transaction, decision_id)?
            .ok_or_else(corrupt)
            .and_then(|graph| inspection_decision_detail(graph, authority.project_uuid))?;
        if detail.summary.created_at_unix_ms != nonnegative_u64(created_at_unix_ms)? {
            return Err(corrupt());
        }
        items.push(detail.summary);
    }
    commit(transaction)?;
    Ok(DecisionFollowPageRead {
        items,
        next_insertion_sequence,
    })
}

pub(crate) fn load_decision_detail(
    connection: &Connection,
    config: &RouterConfig,
    decision_id: Uuid,
) -> Result<Option<DecisionDetailV1>, LedgerError> {
    let transaction = read_transaction(connection)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let detail = load_decision_graph(&transaction, decision_id)?
        .filter(|graph| graph.graph.parent.project_uuid == authority.project_uuid)
        .map(|graph| inspection_decision_detail(graph, authority.project_uuid))
        .transpose()?;
    commit(transaction)?;
    Ok(detail)
}

pub(crate) fn load_decision_exposure(
    connection: &Connection,
    config: &RouterConfig,
    decision_id: Uuid,
) -> Result<Option<DecisionExposureV1>, LedgerError> {
    let transaction = read_transaction(connection)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let Some(parent) = load_verified_decision_parent(&transaction, decision_id)? else {
        commit(transaction)?;
        return Ok(None);
    };
    if parent.project_uuid != authority.project_uuid {
        commit(transaction)?;
        return Ok(None);
    }
    let stored_active = load_verified_active_decision_facts(&transaction, decision_id)?;
    let active = match (parent.cohort_generation_id, stored_active) {
        (None, None) => None,
        (Some(_), Some(facts))
            if facts.decision_id == decision_id
                && facts.active_experiment_id == parent.active_experiment_id
                && facts
                    .assignment_arm
                    .matches_decision_reason(parent.final_reason.as_str()) =>
        {
            Some(ActiveDecisionExposureV1 {
                active_experiment_id: facts.active_experiment_id,
                active_assignment_id: facts.active_assignment_id,
                planned_route: facts.planned_route.as_str().into(),
                assignment_arm: facts.assignment_arm.as_str().into(),
                control_generation: facts.control_generation,
                promotion_lower_bound: facts.promotion_lower_bound,
                retention_lower_bound: facts.retention_lower_bound,
                configured_holdout_probability: facts.configured_holdout_probability,
                configured_canary_probability: facts.configured_canary_probability,
                effective_arm_probability: facts.effective_arm_probability,
                conditional_selection_probability: facts.conditional_selection_probability,
                propensity: facts.propensity,
                fallback_reason: facts.fallback_reason,
            })
        }
        _ => return Err(corrupt()),
    };
    let outcome_id = transaction
        .query_row(
            "SELECT outcome_id FROM outcomes WHERE representative_decision_id = ?1",
            [decision_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let outcome = outcome_id
        .map(|outcome_id| {
            let outcome_id = parse_uuid_v7(&outcome_id)?;
            let outcome = load_verified_outcome(&transaction, authority.project_uuid, outcome_id)?
                .ok_or_else(corrupt)?;
            if outcome.decision_id != Some(decision_id)
                || parent
                    .active_experiment_id
                    .is_none_or(|expected| expected != outcome.active_experiment_id)
            {
                return Err(corrupt());
            }
            inspection_outcome_summary(outcome)
        })
        .transpose()?;
    if active.is_none() && outcome.is_some() {
        return Err(corrupt());
    }
    commit(transaction)?;
    Ok(Some(DecisionExposureV1 {
        schema: DECISION_EXPOSURE_SCHEMA_V1.into(),
        decision_id,
        active,
        outcome,
    }))
}

fn inspection_decision_detail(
    verified: VerifiedStoredDecisionGraphV1,
    project_uuid: Uuid,
) -> Result<DecisionDetailV1, LedgerError> {
    let graph = verified.graph;
    let parent = graph.parent;
    if parent.project_uuid != project_uuid {
        return Err(corrupt());
    }
    let mode = if parent.cohort_generation_id.is_some() {
        "active"
    } else {
        "recommend"
    };
    let summary = DecisionSummaryV1 {
        decision_id: parent.decision_id,
        pool_id: parent.pool_id.clone(),
        mode: mode.into(),
        candidate_id: parent.candidate_id.clone(),
        recommended_model: parent.recommended_model.clone(),
        served_model: parent.served_model.clone(),
        final_reason: parent.final_reason.as_str().into(),
        canonical_query_hash: parent.canonical_query_hash.clone(),
        cohort_generation_id: parent.cohort_generation_id,
        created_at_unix_ms: nonnegative_u64(parent.created_at_unix_ms)?,
    };
    let candidates = graph
        .summaries
        .into_iter()
        .map(|candidate| {
            Ok(InspectionDecisionCandidateSummaryV1 {
                candidate_id: candidate.candidate_id,
                rank_ordinal: u32::try_from(candidate.rank_ordinal).map_err(|_| corrupt())?,
                partition_hash: candidate.partition_id.map(|_| candidate.partition_hash),
                neighbor_count: u32::try_from(candidate.neighbor_count).map_err(|_| corrupt())?,
                effective_sample_size: candidate.effective_sample_size.map(|value| value.value()),
                lower_bound: candidate.lower_bound.map(|value| value.value()),
                reason: candidate.terminal_reason.as_str().into(),
            })
        })
        .collect::<Result<Vec<_>, LedgerError>>()?;
    let neighbors = graph
        .neighbors
        .into_iter()
        .map(|neighbor| {
            Ok(InspectionDecisionNeighborV1 {
                neighbor_ordinal: u32::try_from(neighbor.neighbor_ordinal)
                    .map_err(|_| corrupt())?,
                candidate_id: neighbor.candidate_id,
                evidence_id: neighbor.evidence_vector_link_id,
                distance: f64::from(neighbor.distance.value()),
                final_weight: neighbor.final_weight.map(|value| value.value()),
                binary_label: neighbor.binary_label.map(|label| label.as_str().into()),
                exclusion_reason: neighbor.exclusion_reason.as_str().into(),
            })
        })
        .collect::<Result<Vec<_>, LedgerError>>()?;
    Ok(DecisionDetailV1 {
        summary,
        candidates,
        neighbors,
        record_hash: parent.canonical_payload_hash,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn load_outcome_page(
    connection: &Connection,
    config: &RouterConfig,
    filter: &OutcomeFilterV1,
    snapshot_time_unix_ms: u64,
    maximum_insertion_sequence: Option<u64>,
    after: Option<(u64, &str)>,
    limit: usize,
) -> Result<OutcomePageRead, LedgerError> {
    if limit == 0 {
        return Err(corrupt());
    }
    let transaction = read_transaction(connection)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let maximum = match maximum_insertion_sequence {
        Some(value) => value,
        None => nonnegative_u64(
            transaction
                .query_row(
                    "SELECT coalesce(max(outcome.rowid), 0)
                     FROM outcomes AS outcome
                     JOIN active_root_windows AS root
                       ON root.active_root_window_id = outcome.active_root_window_id
                     WHERE root.project_uuid = ?1",
                    [authority.project_uuid.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .map_err(database_error)?,
        )?,
    };
    let after_time = after
        .map(|(created_at, _)| to_i64(created_at))
        .transpose()?;
    let after_id = after.map(|(_, id)| id);
    let mut statement = transaction
        .prepare(
            "SELECT outcome.rowid, outcome.outcome_id, outcome.created_at_unix_ms
             FROM outcomes AS outcome
             JOIN active_root_windows AS root
               ON root.active_root_window_id = outcome.active_root_window_id
             WHERE root.project_uuid = ?1 AND outcome.rowid <= ?2
               AND outcome.created_at_unix_ms <= ?3
               AND (?4 IS NULL OR root.pool_id = ?4)
               AND (?5 IS NULL OR outcome.arm = ?5)
               AND (?6 IS NULL OR outcome.label = ?6)
               AND (?7 IS NULL OR outcome.attribution_status = ?7)
               AND (?8 IS NULL OR outcome.created_at_unix_ms < ?8
                    OR (outcome.created_at_unix_ms = ?8 AND outcome.outcome_id < ?9))
             ORDER BY outcome.created_at_unix_ms DESC, outcome.outcome_id DESC
             LIMIT ?10",
        )
        .map_err(database_error)?;
    let selected = statement
        .query_map(
            params![
                authority.project_uuid.to_string(),
                to_i64(maximum)?,
                to_i64(snapshot_time_unix_ms)?,
                filter.pool_id.as_deref(),
                filter.arm.as_deref(),
                filter.label.as_deref(),
                filter.attribution_status.as_deref(),
                after_time,
                after_id,
                i64::try_from(limit).map_err(|_| corrupt())?,
            ],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    let mut items = Vec::with_capacity(selected.len());
    for (rowid, outcome_id, created_at_unix_ms) in selected {
        let rowid = nonnegative_u64(rowid)?;
        if rowid == 0 || rowid > maximum {
            return Err(corrupt());
        }
        let outcome_id = parse_uuid_v7(&outcome_id)?;
        let outcome = load_verified_outcome(&transaction, authority.project_uuid, outcome_id)?
            .ok_or_else(corrupt)?;
        let created_at_unix_ms = nonnegative_u64(created_at_unix_ms)?;
        if created_at_unix_ms != nonnegative_u64(outcome.created_at_unix_ms)? {
            return Err(corrupt());
        }
        let summary = inspection_outcome_summary(outcome)?;
        if summary.created_at_unix_ms != created_at_unix_ms {
            return Err(corrupt());
        }
        items.push(summary);
    }
    commit(transaction)?;
    Ok(OutcomePageRead {
        items,
        maximum_insertion_sequence: maximum,
    })
}

fn inspection_outcome_summary(
    outcome: crate::ledger::repository::active::VerifiedStoredOutcome,
) -> Result<OutcomeSummaryV1, LedgerError> {
    Ok(OutcomeSummaryV1 {
        outcome_id: outcome.outcome_id,
        active_experiment_id: outcome.active_experiment_id,
        decision_id: outcome.decision_id,
        arm: outcome.arm,
        label: outcome.label,
        attribution_status: outcome.attribution_status,
        latency_ms: outcome.latency_ms,
        created_at_unix_ms: nonnegative_u64(outcome.created_at_unix_ms)?,
    })
}

fn load_overview_decisions(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    window_start_unix_ms: i64,
    snapshot_time_unix_ms: i64,
) -> Result<(OverviewDecisionCountsV1, OverviewExposureCountsV1), LedgerError> {
    let mut statement = transaction
        .prepare(
            "SELECT decision_id, created_at_unix_ms FROM decisions
             WHERE project_uuid = ?1 AND created_at_unix_ms >= ?2
               AND created_at_unix_ms <= ?3
             ORDER BY created_at_unix_ms, decision_id",
        )
        .map_err(database_error)?;
    let selected = statement
        .query_map(
            params![
                project_uuid.to_string(),
                window_start_unix_ms,
                snapshot_time_unix_ms
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);

    let mut decisions = OverviewDecisionCountsV1::default();
    let mut exposures = OverviewExposureCountsV1::default();
    for (decision_id, created_at_unix_ms) in selected {
        let decision_id = parse_uuid_v7(&decision_id)?;
        let parent =
            load_verified_decision_parent(transaction, decision_id)?.ok_or_else(corrupt)?;
        if parent.project_uuid != project_uuid
            || parent.created_at_unix_ms != created_at_unix_ms
            || created_at_unix_ms < window_start_unix_ms
            || created_at_unix_ms > snapshot_time_unix_ms
        {
            return Err(corrupt());
        }
        increment(&mut decisions.total)?;
        increment_map(&mut decisions.final_reasons, parent.final_reason.as_str())?;
        match parent.cohort_generation_id {
            None => {
                if load_verified_active_decision_facts(transaction, decision_id)?.is_some() {
                    return Err(corrupt());
                }
                increment(&mut decisions.recommend)?;
            }
            Some(_) => {
                let facts = load_verified_active_decision_facts(transaction, decision_id)?
                    .ok_or_else(corrupt)?;
                if facts.decision_id != decision_id
                    || facts.active_experiment_id != parent.active_experiment_id
                    || !facts
                        .assignment_arm
                        .matches_decision_reason(parent.final_reason.as_str())
                {
                    return Err(corrupt());
                }
                increment(&mut decisions.active)?;
                match facts.planned_route.as_str() {
                    "candidate" => increment(&mut decisions.candidate_served)?,
                    "anchor_fallback" => increment(&mut decisions.fallback)?,
                    "anchor_control" | "anchor_holdout" | "anchor_forced" | "anchor_paused" => {}
                    _ => return Err(corrupt()),
                }
                match facts.assignment_arm.as_str() {
                    "candidate_treatment" => increment(&mut exposures.candidate_treatment)?,
                    "anchor_control" => increment(&mut exposures.anchor_control)?,
                    "anchor_holdout" => increment(&mut exposures.anchor_holdout)?,
                    "non_learning" => increment(&mut exposures.non_learning)?,
                    _ => return Err(corrupt()),
                }
            }
        }
    }
    decisions.anchor_served = decisions
        .total
        .checked_sub(decisions.candidate_served)
        .ok_or_else(corrupt)?;
    let exposure_total = exposures
        .candidate_treatment
        .checked_add(exposures.anchor_control)
        .and_then(|value| value.checked_add(exposures.anchor_holdout))
        .and_then(|value| value.checked_add(exposures.non_learning))
        .ok_or_else(corrupt)?;
    if decisions.total
        != decisions
            .recommend
            .checked_add(decisions.active)
            .ok_or_else(corrupt)?
        || decisions.active != exposure_total
    {
        return Err(corrupt());
    }
    Ok((decisions, exposures))
}

fn load_overview_outcomes(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    window_start_unix_ms: i64,
    snapshot_time_unix_ms: i64,
) -> Result<OverviewOutcomeCountsV1, LedgerError> {
    let mut statement = transaction
        .prepare(
            "SELECT outcome.outcome_id, outcome.created_at_unix_ms
             FROM outcomes AS outcome
             JOIN active_root_windows AS root
               ON root.active_root_window_id = outcome.active_root_window_id
             WHERE root.project_uuid = ?1 AND outcome.created_at_unix_ms >= ?2
               AND outcome.created_at_unix_ms <= ?3
             ORDER BY outcome.created_at_unix_ms, outcome.outcome_id",
        )
        .map_err(database_error)?;
    let selected = statement
        .query_map(
            params![
                project_uuid.to_string(),
                window_start_unix_ms,
                snapshot_time_unix_ms
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);

    let mut counts = OverviewOutcomeCountsV1::default();
    for (outcome_id, created_at_unix_ms) in selected {
        let outcome_id = parse_uuid_v7(&outcome_id)?;
        let outcome =
            load_verified_outcome(transaction, project_uuid, outcome_id)?.ok_or_else(corrupt)?;
        if outcome.project_uuid != project_uuid
            || outcome.created_at_unix_ms != created_at_unix_ms
            || created_at_unix_ms < window_start_unix_ms
            || created_at_unix_ms > snapshot_time_unix_ms
        {
            return Err(corrupt());
        }
        let arm = match outcome.arm.as_str() {
            "candidate_treatment" => &mut counts.candidate_treatment,
            "anchor_control" => &mut counts.anchor_control,
            "anchor_holdout" => &mut counts.anchor_holdout,
            "non_learning" => &mut counts.non_learning,
            _ => return Err(corrupt()),
        };
        increment(&mut arm.total)?;
        match outcome.label.as_deref() {
            Some("success") => increment(&mut arm.success)?,
            Some("failure") => increment(&mut arm.failure)?,
            None => increment(&mut arm.unlabeled)?,
            _ => return Err(corrupt()),
        }
    }
    for arm in [
        &counts.candidate_treatment,
        &counts.anchor_control,
        &counts.anchor_holdout,
        &counts.non_learning,
    ] {
        if arm.total
            != arm
                .success
                .checked_add(arm.failure)
                .and_then(|value| value.checked_add(arm.unlabeled))
                .ok_or_else(corrupt)?
        {
            return Err(corrupt());
        }
    }
    Ok(counts)
}

fn increment(value: &mut u64) -> Result<(), LedgerError> {
    *value = value.checked_add(1).ok_or_else(corrupt)?;
    Ok(())
}

fn increment_map(counts: &mut BTreeMap<String, u64>, key: &str) -> Result<(), LedgerError> {
    increment(counts.entry(key.to_string()).or_default())
}

fn load_queue_status(
    transaction: &Transaction<'_>,
    config: &RouterConfig,
    project_uuid: Uuid,
) -> Result<QueueStatusV1, LedgerError> {
    let pending = transaction
        .query_row(
            "SELECT
                (SELECT count(*) FROM sample_batches AS batch
                 JOIN sample_batch_state_events AS state
                   ON state.sample_batch_id = batch.sample_batch_id
                 WHERE batch.project_uuid = ?1
                   AND state.event_seq = (
                     SELECT max(latest.event_seq) FROM sample_batch_state_events AS latest
                     WHERE latest.sample_batch_id = batch.sample_batch_id)
                   AND state.state = 'open')
              + (SELECT count(*) FROM embedding_jobs AS job
                 JOIN vector_spaces AS space ON space.vector_space_id = job.vector_space_id
                 JOIN embedding_job_state_events AS state
                   ON state.embedding_job_id = job.embedding_job_id
                 WHERE space.project_uuid = ?1
                   AND state.event_seq = (
                     SELECT max(latest.event_seq) FROM embedding_job_state_events AS latest
                     WHERE latest.embedding_job_id = job.embedding_job_id)
                   AND state.state IN ('pending', 'claimed', 'released', 'retry_scheduled'))
              + (SELECT count(*) FROM vector_materialization_jobs AS job
                 JOIN vector_spaces AS space ON space.vector_space_id = job.vector_space_id
                 JOIN vector_materialization_job_state_events AS state
                   ON state.vector_materialization_job_id = job.vector_materialization_job_id
                 WHERE space.project_uuid = ?1
                   AND state.event_seq = (
                     SELECT max(latest.event_seq)
                     FROM vector_materialization_job_state_events AS latest
                     WHERE latest.vector_materialization_job_id = job.vector_materialization_job_id)
                   AND state.state IN (
                     'pending_embedding', 'claimed', 'released',
                     'retry_scheduled', 'pending_index'))
              + (SELECT count(*) FROM active_look_claims AS claim
                 JOIN active_experiments AS experiment
                   ON experiment.active_experiment_id = claim.active_experiment_id
                 JOIN active_look_claim_state_events AS state
                   ON state.active_look_claim_id = claim.active_look_claim_id
                 WHERE experiment.project_uuid = ?1
                   AND state.event_seq = (
                     SELECT max(latest.event_seq) FROM active_look_claim_state_events AS latest
                     WHERE latest.active_look_claim_id = claim.active_look_claim_id)
                   AND state.state IN ('claimed', 'renewed', 'reclaimed'))",
            [project_uuid.to_string()],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    let rejected = transaction
        .query_row(
            "SELECT count(*) FROM anchors AS anchor
             JOIN anchor_state_events AS state ON state.anchor_id = anchor.anchor_id
             WHERE anchor.project_uuid = ?1
               AND state.event_seq = (
                 SELECT max(latest.event_seq) FROM anchor_state_events AS latest
                 WHERE latest.anchor_id = anchor.anchor_id)
               AND state.state = 'not_scheduled_queue_full'",
            [project_uuid.to_string()],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    let capacity = if config.mode == RouterMode::Off {
        0
    } else {
        u64::try_from(config.writer_command_capacity().map_err(|_| corrupt())?)
            .map_err(|_| corrupt())?
    };
    Ok(QueueStatusV1 {
        pending: nonnegative_u64(pending)?,
        capacity,
        rejected: nonnegative_u64(rejected)?,
    })
}

fn load_lease_status(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    snapshot_time_unix_ms: i64,
) -> Result<LeaseStatusV1, LedgerError> {
    let (active, expired) = transaction
        .query_row(
            "SELECT
                coalesce(sum(CASE WHEN lease_expires_at_unix_ms > ?2 THEN 1 ELSE 0 END), 0),
                coalesce(sum(CASE WHEN lease_expires_at_unix_ms <= ?2 THEN 1 ELSE 0 END), 0)
             FROM (
                SELECT job.lease_expires_at_unix_ms
                FROM embedding_jobs AS job
                JOIN vector_spaces AS space ON space.vector_space_id = job.vector_space_id
                WHERE space.project_uuid = ?1 AND job.lease_expires_at_unix_ms IS NOT NULL
                UNION ALL
                SELECT job.lease_expires_at_unix_ms
                FROM vector_materialization_jobs AS job
                JOIN vector_spaces AS space ON space.vector_space_id = job.vector_space_id
                WHERE space.project_uuid = ?1 AND job.lease_expires_at_unix_ms IS NOT NULL
                UNION ALL
                SELECT lease.lease_expires_at_unix_ms
                FROM vector_index_rebuild_leases AS lease
                JOIN vector_spaces AS space ON space.vector_space_id = lease.vector_space_id
                WHERE space.project_uuid = ?1
                UNION ALL
                SELECT state.lease_expires_at_unix_ms
                FROM active_look_claim_state_events AS state
                JOIN active_look_claims AS claim
                  ON claim.active_look_claim_id = state.active_look_claim_id
                JOIN active_experiments AS experiment
                  ON experiment.active_experiment_id = claim.active_experiment_id
                WHERE experiment.project_uuid = ?1
                  AND state.event_seq = (
                    SELECT max(latest.event_seq) FROM active_look_claim_state_events AS latest
                    WHERE latest.active_look_claim_id = state.active_look_claim_id)
                  AND state.state IN ('claimed', 'renewed', 'reclaimed')
                  AND state.lease_expires_at_unix_ms IS NOT NULL
             )",
            params![project_uuid.to_string(), snapshot_time_unix_ms],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        )
        .map_err(database_error)?;
    Ok(LeaseStatusV1 {
        active: nonnegative_u64(active)?,
        expired: nonnegative_u64(expired)?,
    })
}

fn load_vector_index_status(
    transaction: &Transaction<'_>,
    authority: &InspectionAuthoritySnapshot,
) -> Result<VectorIndexStatusV1, LedgerError> {
    let mut spaces = BTreeMap::new();
    for vector_authority in authority.vector_authorities.values() {
        let FrozenPoolVectorAuthority::Enabled(mapping) = vector_authority else {
            continue;
        };
        let space_id = mapping.space.space.vector_space_id.clone();
        match spaces.insert(space_id, mapping.space.source_seq) {
            Some(existing) if existing != mapping.space.source_seq => return Err(corrupt()),
            _ => {}
        }
    }

    let mut active_spaces = 0_u64;
    let mut rebuilding_spaces = 0_u64;
    let mut degraded_spaces = 0_u64;
    let mut stale_spaces = 0_u64;
    for (vector_space_id, source_seq) in spaces {
        let building = load_manifest_keys(
            transaction,
            &vector_space_id,
            &[VectorIndexManifestState::Building],
        )?;
        if building.len() > 1 {
            return Err(corrupt());
        }
        if let Some(generation) = building.first() {
            let manifest = load_validated_manifest(transaction, &vector_space_id, *generation)?
                .ok_or_else(corrupt)?;
            if manifest.state() != VectorIndexManifestState::Building {
                return Err(corrupt());
            }
            rebuilding_spaces = rebuilding_spaces.checked_add(1).ok_or_else(corrupt)?;
        }

        let current = load_manifest_keys(
            transaction,
            &vector_space_id,
            &[
                VectorIndexManifestState::Active,
                VectorIndexManifestState::Unavailable,
                VectorIndexManifestState::Corrupt,
            ],
        )?;
        if current.len() > 1 {
            return Err(corrupt());
        }
        let Some(generation) = current.first() else {
            degraded_spaces = degraded_spaces.checked_add(1).ok_or_else(corrupt)?;
            continue;
        };
        let manifest = load_validated_manifest(transaction, &vector_space_id, *generation)?
            .ok_or_else(corrupt)?;
        match manifest.state() {
            VectorIndexManifestState::Active => {
                active_spaces = active_spaces.checked_add(1).ok_or_else(corrupt)?;
                if verify_generation_objects(transaction, manifest.authority())?
                    != GenerationObjectsStatus::Complete
                {
                    degraded_spaces = degraded_spaces.checked_add(1).ok_or_else(corrupt)?;
                }
                if manifest.applied_source_seq() > source_seq {
                    return Err(corrupt());
                }
                if manifest.applied_source_seq() < source_seq {
                    stale_spaces = stale_spaces.checked_add(1).ok_or_else(corrupt)?;
                }
            }
            VectorIndexManifestState::Unavailable | VectorIndexManifestState::Corrupt => {
                degraded_spaces = degraded_spaces.checked_add(1).ok_or_else(corrupt)?;
            }
            VectorIndexManifestState::Building
            | VectorIndexManifestState::Retired
            | VectorIndexManifestState::Dropped => return Err(corrupt()),
        }
    }
    Ok(VectorIndexStatusV1 {
        active_spaces,
        rebuilding_spaces,
        degraded_spaces,
        stale_spaces,
    })
}

fn load_manifest_keys(
    transaction: &Transaction<'_>,
    vector_space_id: &VectorSpaceId,
    states: &[VectorIndexManifestState],
) -> Result<Vec<VectorIndexGeneration>, LedgerError> {
    let accepted = states
        .iter()
        .map(|state| state.as_str())
        .collect::<BTreeSet<_>>();
    let mut statement = transaction
        .prepare(
            "SELECT generation, state FROM vector_index_manifest
             WHERE vector_space_id = ?1
               AND state IN ('building', 'active', 'unavailable', 'corrupt')
             ORDER BY generation",
        )
        .map_err(database_error)?;
    let rows = statement
        .query_map([vector_space_id.as_str()], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    rows.into_iter()
        .filter(|(_, state)| accepted.contains(state.as_str()))
        .map(|(generation, _)| VectorIndexGeneration::new(generation).map_err(|_| corrupt()))
        .collect()
}

fn load_freshness(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
) -> Result<FreshnessStatusV1, LedgerError> {
    let project = project_uuid.to_string();
    let latest_evidence = transaction
        .query_row(
            "SELECT max(link.created_at_unix_ms)
             FROM evidence_vector_links AS link
             JOIN routing_partitions AS partition
               ON partition.partition_id = link.partition_id
             WHERE partition.project_uuid = ?1",
            [&project],
            |row| row.get::<_, Option<i64>>(0),
        )
        .map_err(database_error)?;
    let latest_decision = transaction
        .query_row(
            "SELECT max(created_at_unix_ms) FROM decisions WHERE project_uuid = ?1",
            [&project],
            |row| row.get::<_, Option<i64>>(0),
        )
        .map_err(database_error)?;
    let latest_outcome = transaction
        .query_row(
            "SELECT max(outcome.created_at_unix_ms)
             FROM outcomes AS outcome
             JOIN active_experiments AS experiment
               ON experiment.active_experiment_id = outcome.active_experiment_id
             WHERE experiment.project_uuid = ?1",
            [&project],
            |row| row.get::<_, Option<i64>>(0),
        )
        .map_err(database_error)?;
    Ok(FreshnessStatusV1 {
        latest_evidence_unix_ms: optional_nonnegative_u64(latest_evidence)?,
        latest_decision_unix_ms: optional_nonnegative_u64(latest_decision)?,
        latest_outcome_unix_ms: optional_nonnegative_u64(latest_outcome)?,
    })
}

fn load_pool_support(
    transaction: &Transaction<'_>,
    authority: &InspectionAuthoritySnapshot,
) -> Result<BTreeMap<String, PoolSupportV1>, LedgerError> {
    let mut statement = transaction
        .prepare(
            "SELECT partition.pool_id, partition.learning_generation_id,
                    count(*), count(DISTINCT link.root_uuid),
                    sum(CASE WHEN link.quality_label IS NOT NULL THEN 1 ELSE 0 END)
             FROM evidence_vector_links AS link
             JOIN routing_partitions AS partition
               ON partition.partition_id = link.partition_id
             JOIN evidence_vector_link_state_events AS state
               ON state.evidence_vector_link_id = link.evidence_vector_link_id
             WHERE partition.project_uuid = ?1
               AND state.event_seq = (
                 SELECT max(latest.event_seq)
                 FROM evidence_vector_link_state_events AS latest
                 WHERE latest.evidence_vector_link_id = link.evidence_vector_link_id)
               AND state.state = 'ready'
             GROUP BY partition.pool_id, partition.learning_generation_id",
        )
        .map_err(database_error)?;
    let rows = statement
        .query_map([authority.project_uuid.to_string()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, i64>(3)?,
                row.get::<_, i64>(4)?,
            ))
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    let mut support = BTreeMap::new();
    for (pool_id, generation_id, ready, roots, labeled) in rows {
        if authority
            .learning_generation_ids
            .get(&pool_id)
            .is_none_or(|expected| expected.to_string() != generation_id)
        {
            continue;
        }
        let ready = nonnegative_u64(ready)?;
        let labeled = nonnegative_u64(labeled)?;
        if labeled > ready {
            return Err(corrupt());
        }
        support.insert(
            pool_id,
            PoolSupportV1 {
                ready_evidence: ready,
                independent_roots: nonnegative_u64(roots)?,
                coverage: (ready > 0).then_some(labeled as f64 / ready as f64),
            },
        );
    }
    Ok(support)
}

fn load_health_summary(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    pool_id: Option<&str>,
) -> Result<HealthSummaryV1, LedgerError> {
    let maximum = nonnegative_u64(
        transaction
            .query_row(
                "SELECT coalesce(max(rowid), 0) FROM health_events",
                [],
                |row| row.get::<_, i64>(0),
            )
            .map_err(database_error)?,
    )?;
    let mut statement = transaction
        .prepare(
            "WITH candidate AS (
                SELECT health.rowid, health.health_event_id, health.project_uuid,
                       health.process_instance_id, health.requested_anchor_id,
                       health.requested_dependency_key_id, health.anchor_id,
                       health.dependency_key_id, health.stable_class, health.severity,
                       health.created_at_unix_ms, health.canonical_payload_hash,
                       CASE
                         WHEN health.requested_dependency_key_id IS NOT NULL
                           THEN 'dependency:' || health.requested_dependency_key_id
                         WHEN health.dependency_key_id IS NOT NULL
                           THEN 'dependency:' || health.dependency_key_id
                         WHEN health.requested_anchor_id IS NOT NULL
                           THEN 'anchor:' || health.requested_anchor_id
                         WHEN health.anchor_id IS NOT NULL THEN 'anchor:' || health.anchor_id
                         ELSE 'class:' || health.stable_class
                       END AS subject_key
                FROM health_events AS health
                LEFT JOIN anchors AS anchor
                  ON anchor.anchor_id = coalesce(health.anchor_id, health.requested_anchor_id)
                WHERE health.project_uuid = ?1
                  AND (?2 IS NULL OR anchor.pool_id = ?2)
             ), ranked AS (
                SELECT *, row_number() OVER (
                    PARTITION BY subject_key
                    ORDER BY created_at_unix_ms DESC, health_event_id DESC
                ) AS subject_rank
                FROM candidate
             )
             SELECT rowid, health_event_id, project_uuid, process_instance_id,
                    requested_anchor_id, requested_dependency_key_id,
                    anchor_id, dependency_key_id, stable_class, severity,
                    created_at_unix_ms, canonical_payload_hash
             FROM ranked
             WHERE subject_rank = 1",
        )
        .map_err(database_error)?;
    let rows = statement
        .query_map(params![project_uuid.to_string(), pool_id], |row| {
            Ok(StoredHealthEvent {
                rowid: row.get(0)?,
                health_event_id: row.get(1)?,
                project_uuid: row.get(2)?,
                process_instance_id: row.get(3)?,
                requested_anchor_id: row.get(4)?,
                requested_dependency_key_id: row.get(5)?,
                anchor_id: row.get(6)?,
                dependency_key_id: row.get(7)?,
                stable_class: row.get(8)?,
                severity: row.get(9)?,
                created_at_unix_ms: row.get(10)?,
                canonical_payload_hash: row.get(11)?,
            })
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    let mut summary = HealthSummaryV1::default();
    let mut worst_rank = 0_u8;
    for stored in rows {
        let event = verify_health_event(stored, project_uuid, maximum)?;
        let rank = match event.severity.as_str() {
            "info" => 1,
            "warning" => 2,
            "degraded" => 3,
            _ => return Err(corrupt()),
        };
        if event.severity == "degraded" {
            summary.degraded = summary.degraded.checked_add(1).ok_or_else(corrupt)?;
        }
        summary.latest_event_unix_ms = Some(
            summary
                .latest_event_unix_ms
                .unwrap_or_default()
                .max(event.created_at_unix_ms),
        );
        if rank > worst_rank {
            worst_rank = rank;
            summary.worst = event.severity;
        }
    }
    Ok(summary)
}

struct StoredHealthEvent {
    rowid: i64,
    health_event_id: String,
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

fn verify_health_event(
    stored: StoredHealthEvent,
    expected_project_uuid: Uuid,
    maximum_insertion_sequence: u64,
) -> Result<HealthEventV1, LedgerError> {
    let rowid = nonnegative_u64(stored.rowid)?;
    let health_event_id = parse_uuid_v7(&stored.health_event_id)?;
    let project_uuid = Uuid::parse_str(&stored.project_uuid).map_err(|_| corrupt())?;
    let process_instance_id = stored
        .process_instance_id
        .as_deref()
        .map(parse_uuid_v7)
        .transpose()?
        .ok_or_else(corrupt)?;
    for value in [
        stored.requested_anchor_id.as_deref(),
        stored.anchor_id.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        parse_uuid_v7(value)?;
    }
    for value in [
        stored.requested_dependency_key_id.as_deref(),
        stored.dependency_key_id.as_deref(),
    ]
    .into_iter()
    .flatten()
    {
        if !is_sha256(value) {
            return Err(corrupt());
        }
    }
    if rowid == 0
        || rowid > maximum_insertion_sequence
        || project_uuid != expected_project_uuid
        || !valid_stable_class(&stored.stable_class)
        || !matches!(stored.severity.as_str(), "info" | "warning" | "degraded")
    {
        return Err(corrupt());
    }
    let expected_hash = crate::ledger::repository::hash_json(&json!({
        "health_event_id": health_event_id,
        "project_uuid": project_uuid,
        "process_instance_id": process_instance_id,
        "requested_anchor_id": stored.requested_anchor_id,
        "requested_dependency_key_id": stored.requested_dependency_key_id,
        "anchor_id": stored.anchor_id,
        "dependency_key_id": stored.dependency_key_id,
        "stable_class": stored.stable_class,
        "severity": stored.severity,
        "created_at_unix_ms": stored.created_at_unix_ms,
    }))?;
    if expected_hash != stored.canonical_payload_hash {
        return Err(corrupt());
    }
    let (subject_kind, subject_id) = if let Some(value) = stored
        .requested_dependency_key_id
        .or(stored.dependency_key_id)
    {
        ("dependency".to_string(), Some(value))
    } else if let Some(value) = stored.requested_anchor_id.or(stored.anchor_id) {
        ("anchor".to_string(), Some(value))
    } else {
        ("process".to_string(), Some(process_instance_id.to_string()))
    };
    Ok(HealthEventV1 {
        health_event_id,
        subject_kind,
        subject_id,
        severity: stored.severity,
        reason: stored.stable_class,
        created_at_unix_ms: nonnegative_u64(stored.created_at_unix_ms)?,
    })
}

fn read_transaction(connection: &Connection) -> Result<Transaction<'_>, LedgerError> {
    connection.unchecked_transaction().map_err(database_error)
}

fn commit(transaction: Transaction<'_>) -> Result<(), LedgerError> {
    transaction.commit().map_err(database_error)
}

fn parse_uuid_v7(value: &str) -> Result<Uuid, LedgerError> {
    let uuid = Uuid::parse_str(value).map_err(|_| corrupt())?;
    if uuid.get_variant() != uuid::Variant::RFC4122 || uuid.get_version_num() != 7 {
        return Err(corrupt());
    }
    Ok(uuid)
}

fn valid_stable_class(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'.' || byte == b'_'
        })
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn optional_nonnegative_u64(value: Option<i64>) -> Result<Option<u64>, LedgerError> {
    value.map(nonnegative_u64).transpose()
}

fn nonnegative_u64(value: i64) -> Result<u64, LedgerError> {
    u64::try_from(value).map_err(|_| corrupt())
}

fn to_i64(value: u64) -> Result<i64, LedgerError> {
    i64::try_from(value).map_err(|_| corrupt())
}

fn migration_sort_id(version: i64) -> Result<String, LedgerError> {
    if version <= 0 {
        return Err(corrupt());
    }
    Ok(format!("{version:020}"))
}

fn corrupt() -> LedgerError {
    LedgerError::new(LedgerErrorClass::IdentityInvariant)
}

fn database_error(_: rusqlite::Error) -> LedgerError {
    LedgerError::new(LedgerErrorClass::DatabaseOperationFailed)
}

fn map_control_error(error: RouterControlError) -> LedgerError {
    match error {
        RouterControlError::Busy => LedgerError::new(LedgerErrorClass::Busy),
        RouterControlError::MigrationRequired => LedgerError::new(LedgerErrorClass::FutureSchema),
        RouterControlError::StorageUnavailable | RouterControlError::Unavailable => {
            LedgerError::new(LedgerErrorClass::DatabaseOperationFailed)
        }
        RouterControlError::InvalidArgument
        | RouterControlError::Conflict { .. }
        | RouterControlError::MutationExpired
        | RouterControlError::CapacityExhausted
        | RouterControlError::IntegrityError => {
            LedgerError::new(LedgerErrorClass::IdentityInvariant)
        }
    }
}

fn map_operator_history_error(error: InspectionError) -> LedgerError {
    match error {
        InspectionError::Busy => LedgerError::new(LedgerErrorClass::Busy),
        InspectionError::StorageUnavailable => {
            LedgerError::new(LedgerErrorClass::DatabaseOperationFailed)
        }
        InspectionError::MigrationRequired => LedgerError::new(LedgerErrorClass::FutureSchema),
        InspectionError::InvalidArgument
        | InspectionError::InvalidCursor
        | InspectionError::NotFound
        | InspectionError::Unauthorized
        | InspectionError::Forbidden
        | InspectionError::Conflict
        | InspectionError::MutationExpired
        | InspectionError::CapacityExhausted
        | InspectionError::EgressDenied
        | InspectionError::IntegrityError
        | InspectionError::IncompatibleApi => LedgerError::new(LedgerErrorClass::IdentityInvariant),
    }
}
