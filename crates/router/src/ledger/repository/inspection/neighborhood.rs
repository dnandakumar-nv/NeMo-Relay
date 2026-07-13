// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Provider-free persisted neighborhood inspection in one verified snapshot.

use rusqlite::{Connection, OptionalExtension, params};
use uuid::Uuid;

use super::{InspectionAuthoritySnapshot, load_inspection_authority_in_transaction};
use crate::confidence::{
    CandidateConfidenceInputV1, CandidateConfidenceReasonV1, CandidateConfidenceSummaryV1,
    CandidateEvidenceV1, ConfidenceBinaryLabelV1, ConfidenceEvaluationInputV1,
    ConfidenceEvaluationSourceV1, ConfidenceGateResultV1, ConfidenceNeighborInputV1,
    ConfidencePolicyV1, ConfidenceTerminalClassV1, evaluate_single_candidate_v1,
};
use crate::config::{PoolConfig, RouterConfig};
use crate::inspection::projection::{DiagnosticVector, project_pca_2};
use crate::inspection::{
    DiagnosticPointKindV1, NeighborhoodGatesV1, NeighborhoodLookupV1, NeighborhoodNeighborV1,
    NeighborhoodRecommendationV1, NeighborhoodReportV1, NeighborhoodSupportV1,
};
use crate::judge::{JudgeBinaryLabelV1, JudgeEvaluationSourceV1};
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::ledger::repository::materialization::{
    VerifiedVectorLinkSource, VerifiedVectorLinkSourceLoad, load_verified_vector_link_source,
};
use crate::ledger::repository::shadow::ShadowTerminalClass;
use crate::ledger::repository::vector_catalog::{
    LiveRoutingPartitionResolution, load_canonical_query, load_embedding_cache,
    resolve_live_routing_partition,
};
use crate::ledger::repository::vector_registry::{FrozenMappingKey, FrozenPoolVectorAuthority};
use crate::ledger::repository::vector_search::{
    ProjectedVectorNeighbor, search_live_partition_in_transaction,
};
use crate::routing_partition::{
    RoutingPartitionArtifactV1, RoutingPartitionV1, artifact_from_routing_partition_v1,
};
use crate::vector::{AuthoritativeVector, VectorRecordId, VectorSpaceId};
use crate::vector_store::VectorStoreError;

pub(crate) enum PersistedNeighborhoodRead {
    Report(Box<NeighborhoodReportV1>),
    NotFound,
    InvalidArgument,
    NeedsEmbedding,
}

enum QueryVectorSource {
    Persisted,
    Request(Option<AuthoritativeVector>),
}

pub(crate) fn load_persisted_neighborhood(
    connection: &Connection,
    config: &RouterConfig,
    lookup: &NeighborhoodLookupV1,
    snapshot_time_unix_ms: u64,
) -> Result<PersistedNeighborhoodRead, LedgerError> {
    let transaction = connection.unchecked_transaction().map_err(database_error)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let source = match lookup {
        NeighborhoodLookupV1::Evidence { evidence_id } => {
            match evidence_lookup(&transaction, authority.project_uuid, *evidence_id)? {
                Some(source) => source,
                None => return commit_read(transaction, PersistedNeighborhoodRead::NotFound),
            }
        }
        NeighborhoodLookupV1::QueryHash {
            canonical_query_hash,
            partition,
        } => {
            let artifact = artifact_from_routing_partition_v1(partition)
                .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
            let Some(pool) = infer_pool(config, &authority, partition) else {
                return commit_read(transaction, PersistedNeighborhoodRead::InvalidArgument);
            };
            NeighborhoodSource {
                pool_id: pool.id.clone(),
                canonical_query_hash: canonical_query_hash.clone(),
                partition: artifact,
            }
        }
        NeighborhoodLookupV1::Request { .. } => {
            return commit_read(transaction, PersistedNeighborhoodRead::InvalidArgument);
        }
    };

    let result = load_neighborhood_in_transaction(
        &transaction,
        config,
        &authority,
        source,
        QueryVectorSource::Persisted,
        snapshot_time_unix_ms,
    )?;
    commit_read(transaction, result)
}

pub(crate) fn load_request_neighborhood(
    connection: &Connection,
    config: &RouterConfig,
    pool_id: String,
    canonical_query_hash: String,
    partition: RoutingPartitionArtifactV1,
    provided_vector: Option<AuthoritativeVector>,
    snapshot_time_unix_ms: u64,
) -> Result<PersistedNeighborhoodRead, LedgerError> {
    let transaction = connection.unchecked_transaction().map_err(database_error)?;
    let authority = load_inspection_authority_in_transaction(&transaction, config)?;
    let result = load_neighborhood_in_transaction(
        &transaction,
        config,
        &authority,
        NeighborhoodSource {
            pool_id,
            canonical_query_hash,
            partition,
        },
        QueryVectorSource::Request(provided_vector),
        snapshot_time_unix_ms,
    )?;
    commit_read(transaction, result)
}

fn load_neighborhood_in_transaction(
    transaction: &rusqlite::Transaction<'_>,
    config: &RouterConfig,
    authority: &InspectionAuthoritySnapshot,
    source: NeighborhoodSource,
    vector_source: QueryVectorSource,
    snapshot_time_unix_ms: u64,
) -> Result<PersistedNeighborhoodRead, LedgerError> {
    let Some(pool) = config.pools.iter().find(|pool| pool.id == source.pool_id) else {
        return Err(corrupt());
    };
    let partition = &source.partition.partition;
    let current_policy = authority
        .policy_version_ids
        .get(&pool.id)
        .ok_or_else(corrupt)?;
    let current_learning = authority
        .learning_generation_ids
        .get(&pool.id)
        .ok_or_else(corrupt)?;
    let Some(vector_space_id) = current_vector_space(authority, &pool.id) else {
        let report = fallback_report(
            pool,
            &source,
            "version_mismatch",
            false,
            snapshot_time_unix_ms,
        );
        return Ok(PersistedNeighborhoodRead::Report(Box::new(report)));
    };
    if partition.policy_version_id != *current_policy
        || partition.learning_generation_id != *current_learning
        || partition.vector_space_id != vector_space_id.as_str()
    {
        let report = fallback_report(
            pool,
            &source,
            "version_mismatch",
            false,
            snapshot_time_unix_ms,
        );
        return Ok(PersistedNeighborhoodRead::Report(Box::new(report)));
    }
    let mapping = FrozenMappingKey::new(
        authority.project_uuid,
        authority.config_generation_id.clone(),
        pool.id.clone(),
        current_policy.clone(),
    )
    .map_err(|_| corrupt())?;
    let partition_id =
        match resolve_live_routing_partition(transaction, &mapping, &source.partition)? {
            LiveRoutingPartitionResolution::Found(found) => found.partition_id,
            LiveRoutingPartitionResolution::NoPartition => {
                let report =
                    fallback_report(pool, &source, "no_partition", false, snapshot_time_unix_ms);
                return Ok(PersistedNeighborhoodRead::Report(Box::new(report)));
            }
            LiveRoutingPartitionResolution::AuthorityNotFound
            | LiveRoutingPartitionResolution::StaleLearningGeneration => {
                let report = fallback_report(
                    pool,
                    &source,
                    "version_mismatch",
                    false,
                    snapshot_time_unix_ms,
                );
                return Ok(PersistedNeighborhoodRead::Report(Box::new(report)));
            }
        };
    let Some(learning) = pool
        .learning
        .as_ref()
        .and_then(|value| value.complete_policy())
    else {
        let report = fallback_report(
            pool,
            &source,
            "version_mismatch",
            false,
            snapshot_time_unix_ms,
        );
        return Ok(PersistedNeighborhoodRead::Report(Box::new(report)));
    };
    let Some(candidate) = pool.candidates.iter().find(|candidate| {
        candidate.id == partition.candidate_id
            && candidate.model == partition.candidate_model
            && candidate.model_revision == partition.candidate_model_revision
    }) else {
        return Ok(PersistedNeighborhoodRead::InvalidArgument);
    };
    let policy = ConfidencePolicyV1::new(
        learning.top_k,
        learning.radius,
        learning.min_points,
        learning.min_independent_roots,
        learning.min_effective_samples,
        learning.min_coverage,
        learning.time_decay_half_life_seconds,
        learning.prior_success,
        learning.prior_failure,
        learning.familywise_credible_level,
        learning.promotion_lower_bound,
        pool.judge.judge_confidence_floor,
    )
    .map_err(|_| corrupt())?;
    let cached = || {
        load_embedding_cache(
            transaction,
            authority.project_uuid,
            &vector_space_id,
            &source.canonical_query_hash,
        )
    };
    let query_vector = match vector_source {
        QueryVectorSource::Persisted => {
            if load_canonical_query(transaction, &source.canonical_query_hash)?.is_none() {
                return Ok(PersistedNeighborhoodRead::NotFound);
            }
            let Some(cache) = cached()? else {
                let report = fallback_report(
                    pool,
                    &source,
                    "embedding_unavailable",
                    true,
                    snapshot_time_unix_ms,
                );
                return Ok(PersistedNeighborhoodRead::Report(Box::new(report)));
            };
            cache.vector
        }
        QueryVectorSource::Request(provided) => match cached()? {
            Some(cache) => cache.vector,
            None => match provided {
                Some(vector) if vector.vector_space_id() == &vector_space_id => vector,
                Some(_) => return Err(corrupt()),
                None => return Ok(PersistedNeighborhoodRead::NeedsEmbedding),
            },
        },
    };
    let projected = match search_live_partition_in_transaction(
        transaction,
        &vector_space_id,
        partition_id,
        query_vector.vector(),
        learning.top_k,
    ) {
        Ok(neighbors) => neighbors,
        Err(VectorStoreError::SpaceNotFound | VectorStoreError::InvalidVector(_)) => {
            let report = fallback_report(
                pool,
                &source,
                "version_mismatch",
                false,
                snapshot_time_unix_ms,
            );
            return Ok(PersistedNeighborhoodRead::Report(Box::new(report)));
        }
        Err(VectorStoreError::Unavailable) => {
            let report = fallback_report(
                pool,
                &source,
                "vector_unhealthy",
                true,
                snapshot_time_unix_ms,
            );
            return Ok(PersistedNeighborhoodRead::Report(Box::new(report)));
        }
        Err(
            VectorStoreError::Corrupt
            | VectorStoreError::InvalidTopK
            | VectorStoreError::InvalidCapacity
            | VectorStoreError::Conflict
            | VectorStoreError::CapacityExceeded,
        ) => return Err(corrupt()),
    };
    let confidence_neighbors = projected
        .iter()
        .cloned()
        .map(project_confidence_neighbor)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|_| corrupt())?;
    let as_of_unix_ms = i64::try_from(snapshot_time_unix_ms).map_err(|_| corrupt())?;
    let summary = evaluate_single_candidate_v1(
        &policy,
        pool.candidates.len(),
        as_of_unix_ms,
        CandidateConfidenceInputV1::new(
            candidate.id.clone(),
            candidate.cost_rank,
            CandidateEvidenceV1::Neighbors(confidence_neighbors),
        )
        .map_err(|_| corrupt())?,
    )
    .map_err(|_| corrupt())?;
    let report = found_report(
        pool,
        source,
        summary,
        &query_vector,
        &projected,
        snapshot_time_unix_ms,
    )?;
    Ok(PersistedNeighborhoodRead::Report(Box::new(report)))
}

struct NeighborhoodSource {
    pool_id: String,
    canonical_query_hash: String,
    partition: RoutingPartitionArtifactV1,
}

fn evidence_lookup(
    connection: &Connection,
    project_uuid: Uuid,
    evidence_id: Uuid,
) -> Result<Option<NeighborhoodSource>, LedgerError> {
    let vector_space_id = connection
        .query_row(
            "SELECT link.vector_space_id
             FROM evidence_vector_links AS link
             JOIN routing_partitions AS partition
               ON partition.partition_id = link.partition_id
             WHERE link.evidence_vector_link_id = ?1 AND partition.project_uuid = ?2",
            params![evidence_id.to_string(), project_uuid.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .map(VectorSpaceId::new)
        .transpose()
        .map_err(|_| corrupt())?;
    let Some(vector_space_id) = vector_space_id else {
        return Ok(None);
    };
    let record_id = VectorRecordId::new(evidence_id).map_err(|_| corrupt())?;
    let source = match load_verified_vector_link_source(connection, &vector_space_id, record_id)? {
        VerifiedVectorLinkSourceLoad::LinkMissing => return Ok(None),
        VerifiedVectorLinkSourceLoad::AuthorityMissing => return Err(corrupt()),
        VerifiedVectorLinkSourceLoad::Verified(source) => source,
    };
    if source.project_uuid != project_uuid {
        return Err(corrupt());
    }
    source_to_lookup(*source)
}

fn source_to_lookup(
    source: VerifiedVectorLinkSource,
) -> Result<Option<NeighborhoodSource>, LedgerError> {
    let partition = artifact_from_routing_partition_v1(&source.partition).map_err(|_| corrupt())?;
    Ok(Some(NeighborhoodSource {
        pool_id: source.pool_id,
        canonical_query_hash: source.canonical_query_hash,
        partition,
    }))
}

fn infer_pool<'a>(
    config: &'a RouterConfig,
    authority: &InspectionAuthoritySnapshot,
    partition: &RoutingPartitionV1,
) -> Option<&'a PoolConfig> {
    let mut matches = config.pools.iter().filter(|pool| {
        authority.policy_version_ids.get(&pool.id) == Some(&partition.policy_version_id)
            && current_vector_space(authority, &pool.id)
                .is_some_and(|space| space.as_str() == partition.vector_space_id)
            && pool.api_family == partition.api_family
            && pool.anchor_revision == partition.anchor_revision
            && pool.anchor_models.contains(&partition.anchor_model)
            && pool.candidates.iter().any(|candidate| {
                candidate.id == partition.candidate_id
                    && candidate.model == partition.candidate_model
                    && candidate.model_revision == partition.candidate_model_revision
            })
    });
    let found = matches.next()?;
    matches.next().is_none().then_some(found)
}

fn current_vector_space(
    authority: &InspectionAuthoritySnapshot,
    pool_id: &str,
) -> Option<VectorSpaceId> {
    match authority.vector_authorities.get(pool_id)? {
        FrozenPoolVectorAuthority::Enabled(mapping) => {
            Some(mapping.mapping.vector_space_id.clone())
        }
        FrozenPoolVectorAuthority::Disabled => None,
    }
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

fn found_report(
    pool: &PoolConfig,
    source: NeighborhoodSource,
    summary: CandidateConfidenceSummaryV1,
    query_vector: &crate::vector::AuthoritativeVector,
    projected: &[ProjectedVectorNeighbor],
    snapshot_time_unix_ms: u64,
) -> Result<NeighborhoodReportV1, LedgerError> {
    let projection = if projected
        .iter()
        .all(|neighbor| neighbor.vector_record.is_some())
    {
        let mut vectors = Vec::with_capacity(projected.len() + 1);
        vectors.push(DiagnosticVector {
            record_id: source.canonical_query_hash.clone(),
            kind: DiagnosticPointKindV1::Query,
            vector: query_vector.vector(),
        });
        for neighbor in projected {
            let record = neighbor.vector_record.as_ref().ok_or_else(corrupt)?;
            vectors.push(DiagnosticVector {
                record_id: neighbor.vector_match.record_id().to_string(),
                kind: DiagnosticPointKindV1::Evidence,
                vector: record.vector().vector(),
            });
        }
        project_pca_2(&source.partition.partition.vector_space_id, vectors)
            .map_err(|_| corrupt())?
    } else {
        None
    };
    let neighbors = summary
        .neighbors
        .iter()
        .map(|neighbor| {
            Ok(NeighborhoodNeighborV1 {
                evidence_id: neighbor.evidence_vector_link_id,
                ordinal: u32::try_from(neighbor.candidate_ordinal).map_err(|_| corrupt())?,
                distance: f64::from(neighbor.distance),
                age_seconds: neighbor.age_millis.map(|value| value as f64 / 1_000.0),
                similarity_weight: neighbor.similarity_weight,
                time_weight: neighbor.time_weight,
                final_weight: neighbor.final_weight,
                binary_label: neighbor.binary_label.map(|value| value.as_str().into()),
                inclusion: neighbor.exclusion_reason.as_str().into(),
            })
        })
        .collect::<Result<Vec<_>, LedgerError>>()?;
    let passed = summary.reason == CandidateConfidenceReasonV1::Passed;
    Ok(NeighborhoodReportV1 {
        schema: crate::inspection::NEIGHBORHOOD_REPORT_SCHEMA_V1.into(),
        pool_id: pool.id.clone(),
        canonical_query_hash: source.canonical_query_hash,
        partition: source.partition.partition,
        neighbors,
        support: NeighborhoodSupportV1 {
            returned_neighbors: u32::try_from(summary.top_k_points).map_err(|_| corrupt())?,
            within_radius: u32::try_from(summary.raw_points).map_err(|_| corrupt())?,
            attempted_roots: u32::try_from(summary.attempted_roots).map_err(|_| corrupt())?,
            selected_roots: u32::try_from(
                summary
                    .neighbors
                    .iter()
                    .filter(|neighbor| neighbor.selected_root)
                    .count(),
            )
            .map_err(|_| corrupt())?,
            coverage: summary.coverage,
            effective_sample_size: summary.n_eff,
        },
        credible_lower_bound: summary.lower_bound,
        gates: NeighborhoodGatesV1 {
            partition: gate(summary.gates.partition),
            points: gate(summary.gates.points),
            roots: gate(summary.gates.roots),
            coverage: gate(summary.gates.coverage),
            weight_math: gate(summary.gates.weight_math),
            effective_samples: gate(summary.gates.effective_samples),
            beta_quantile: gate(summary.gates.beta_quantile),
            lower_bound: gate(summary.gates.lower_bound),
        },
        recommendation: NeighborhoodRecommendationV1 {
            candidate_id: passed.then(|| summary.candidate_id.clone()),
            reason: summary.reason.as_str().into(),
            anchor_fallback: !passed,
        },
        projection,
        snapshot_time_unix_ms,
    })
}

fn fallback_report(
    pool: &PoolConfig,
    source: &NeighborhoodSource,
    reason: &str,
    partition_resolved: bool,
    snapshot_time_unix_ms: u64,
) -> NeighborhoodReportV1 {
    NeighborhoodReportV1 {
        schema: crate::inspection::NEIGHBORHOOD_REPORT_SCHEMA_V1.into(),
        pool_id: pool.id.clone(),
        canonical_query_hash: source.canonical_query_hash.clone(),
        partition: source.partition.partition.clone(),
        neighbors: Vec::new(),
        support: NeighborhoodSupportV1::default(),
        credible_lower_bound: None,
        gates: NeighborhoodGatesV1 {
            partition: Some(partition_resolved),
            ..NeighborhoodGatesV1::default()
        },
        recommendation: NeighborhoodRecommendationV1 {
            candidate_id: None,
            reason: reason.into(),
            anchor_fallback: true,
        },
        projection: None,
        snapshot_time_unix_ms,
    }
}

fn gate(value: ConfidenceGateResultV1) -> Option<bool> {
    match value {
        ConfidenceGateResultV1::NotEvaluated => None,
        ConfidenceGateResultV1::Passed => Some(true),
        ConfidenceGateResultV1::Failed => Some(false),
    }
}

fn commit_read(
    transaction: rusqlite::Transaction<'_>,
    result: PersistedNeighborhoodRead,
) -> Result<PersistedNeighborhoodRead, LedgerError> {
    transaction.commit().map_err(database_error)?;
    Ok(result)
}

fn corrupt() -> LedgerError {
    LedgerError::new(LedgerErrorClass::CorruptDatabase)
}

fn database_error(_: rusqlite::Error) -> LedgerError {
    LedgerError::new(LedgerErrorClass::DatabaseOperationFailed)
}
