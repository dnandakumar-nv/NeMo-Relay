// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-snapshot sqlite-vec search and complete relational neighbor projection.

#[cfg(test)]
use std::collections::BTreeMap;
use std::collections::BTreeSet;

use rusqlite::{Connection, Transaction, params};
use uuid::{Uuid, Variant};

use super::materialization::{VerifiedVectorLinkSourceLoad, load_verified_vector_link_source};
use super::vector_catalog::{LiveRoutingPartitionResolution, resolve_live_routing_partition};
use super::vector_index::{ActiveGenerationResolution, resolve_active_generation};
use super::vector_registry::FrozenMappingKey;
use crate::judge::{JudgeBinaryLabelV1, JudgeEvaluationSourceV1, ScoredValueV1};
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::ledger::repository::shadow::ShadowTerminalClass;
use crate::routing_partition::RoutingPartitionArtifactV1;
use crate::sqlite_vec_schema::Vec0SchemaAuthority;
use crate::vector::{NormalizedVector, PartitionId, VectorRecordId, VectorSpaceId};
use crate::vector_store::{VECTOR_TOP_K_MAX, VectorMatch, VectorRecord, VectorStoreError};

/// Optional final evaluation projected with one vector neighbor.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProjectedNeighborEvaluation {
    pub(crate) evaluation_id: Uuid,
    pub(crate) source: JudgeEvaluationSourceV1,
    pub(crate) binary_label: Option<JudgeBinaryLabelV1>,
    pub(crate) judge_confidence: Option<ScoredValueV1>,
    pub(crate) promotion_eligible: bool,
    pub(crate) created_at_unix_ms: i64,
}

/// Complete evidence record returned before the SQLite read snapshot closes.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ProjectedVectorNeighbor {
    pub(crate) vector_match: VectorMatch,
    pub(crate) vector_record: Option<VectorRecord>,
    pub(crate) shadow_attempt_id: Uuid,
    pub(crate) shadow_result_id: Uuid,
    pub(crate) anchor_id: Uuid,
    pub(crate) root_uuid: Uuid,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) terminal_class: ShadowTerminalClass,
    pub(crate) evaluation: Option<ProjectedNeighborEvaluation>,
}

/// Exact live-partition resolution and one-snapshot neighbor projection.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum LiveProjectedNeighborSearch {
    AuthorityNotFound,
    StaleLearningGeneration,
    NoPartition,
    Found {
        partition_id: PartitionId,
        neighbors: Vec<ProjectedVectorNeighbor>,
    },
}

/// Resolve a full live partition artifact and search it in one read snapshot.
pub(crate) fn search_live_projected_neighbors(
    connection: &Connection,
    mapping: &FrozenMappingKey,
    expected_partition: &RoutingPartitionArtifactV1,
    query_vector: &NormalizedVector,
    top_k: usize,
) -> Result<LiveProjectedNeighborSearch, VectorStoreError> {
    validate_top_k(top_k)?;
    let transaction = connection
        .unchecked_transaction()
        .map_err(|_| VectorStoreError::Unavailable)?;
    let result = match resolve_live_routing_partition(&transaction, mapping, expected_partition)
        .map_err(map_resolver_error)?
    {
        LiveRoutingPartitionResolution::AuthorityNotFound => {
            LiveProjectedNeighborSearch::AuthorityNotFound
        }
        LiveRoutingPartitionResolution::StaleLearningGeneration => {
            LiveProjectedNeighborSearch::StaleLearningGeneration
        }
        LiveRoutingPartitionResolution::NoPartition => LiveProjectedNeighborSearch::NoPartition,
        LiveRoutingPartitionResolution::Found(partition) => {
            let vector_space_id =
                VectorSpaceId::new(expected_partition.partition.vector_space_id.clone())
                    .map_err(|_| VectorStoreError::Corrupt)?;
            let neighbors = search_projected_neighbors_in_transaction(
                &transaction,
                &vector_space_id,
                partition.partition_id,
                query_vector,
                top_k,
            )?;
            verify_no_retiring_sources(&transaction, &neighbors)?;
            LiveProjectedNeighborSearch::Found {
                partition_id: partition.partition_id,
                neighbors,
            }
        }
    };
    transaction
        .commit()
        .map_err(|_| VectorStoreError::Unavailable)?;
    Ok(result)
}

/// Search one exact partition in a new deferred transaction.
pub(crate) fn search_projected_neighbors(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
    partition_id: PartitionId,
    query_vector: &NormalizedVector,
    top_k: usize,
) -> Result<Vec<ProjectedVectorNeighbor>, VectorStoreError> {
    validate_top_k(top_k)?;
    let transaction = connection
        .unchecked_transaction()
        .map_err(|_| VectorStoreError::Unavailable)?;
    let result = search_projected_neighbors_in_transaction(
        &transaction,
        vector_space_id,
        partition_id,
        query_vector,
        top_k,
    );
    if result.is_ok() {
        transaction
            .commit()
            .map_err(|_| VectorStoreError::Unavailable)?;
    }
    result
}

/// Search inside a caller-owned deferred transaction without detaching IDs.
pub(crate) fn search_projected_neighbors_in_transaction(
    transaction: &Transaction<'_>,
    vector_space_id: &VectorSpaceId,
    partition_id: PartitionId,
    query_vector: &NormalizedVector,
    top_k: usize,
) -> Result<Vec<ProjectedVectorNeighbor>, VectorStoreError> {
    validate_top_k(top_k)?;
    let manifest = match resolve_active_generation(transaction, vector_space_id)
        .map_err(map_resolver_error)?
    {
        ActiveGenerationResolution::Active(manifest) => manifest,
        ActiveGenerationResolution::Missing => return Err(VectorStoreError::SpaceNotFound),
        ActiveGenerationResolution::Unavailable => return Err(VectorStoreError::Unavailable),
        ActiveGenerationResolution::Corrupt => return Err(VectorStoreError::Corrupt),
    };
    let authority = manifest.authority();
    if authority.root().vector_space_id() != vector_space_id {
        return Err(VectorStoreError::Corrupt);
    }
    if authority.dimensions() != query_vector.dimensions() {
        return Err(VectorStoreError::InvalidVector(
            crate::vector::VectorError::DimensionMismatch,
        ));
    }

    search_resolved_generation(
        transaction,
        authority,
        vector_space_id,
        partition_id,
        query_vector,
        top_k,
    )
}

/// Search one live partition and reject sources already fenced for retirement.
pub(crate) fn search_live_partition_in_transaction(
    transaction: &Transaction<'_>,
    vector_space_id: &VectorSpaceId,
    partition_id: PartitionId,
    query_vector: &NormalizedVector,
    top_k: usize,
) -> Result<Vec<ProjectedVectorNeighbor>, VectorStoreError> {
    let neighbors = search_projected_neighbors_in_transaction(
        transaction,
        vector_space_id,
        partition_id,
        query_vector,
        top_k,
    )?;
    verify_no_retiring_sources(transaction, &neighbors)?;
    Ok(neighbors)
}

fn search_resolved_generation(
    connection: &Connection,
    authority: &Vec0SchemaAuthority,
    vector_space_id: &VectorSpaceId,
    partition_id: PartitionId,
    query_vector: &NormalizedVector,
    top_k: usize,
) -> Result<Vec<ProjectedVectorNeighbor>, VectorStoreError> {
    let ready_count = authoritative_ready_count(connection, vector_space_id, partition_id)?;
    let index_count = indexed_partition_count(connection, authority, partition_id)?;
    if ready_count != index_count {
        return Err(VectorStoreError::Unavailable);
    }
    if ready_count == 0 {
        return Ok(Vec::new());
    }

    let fast_k = ready_count.min(top_k.checked_add(1).ok_or(VectorStoreError::InvalidTopK)?);
    let query_blob = native_vector_blob(query_vector);
    let mut matches = query_matches(
        connection,
        &knn_sql(authority),
        &query_blob,
        partition_id,
        fast_k,
    )?;
    if matches.len() != fast_k {
        return Err(VectorStoreError::Unavailable);
    }
    if matches
        .windows(2)
        .any(|pair| pair[0].distance().total_cmp(&pair[1].distance()).is_gt())
    {
        return Err(VectorStoreError::Corrupt);
    }

    if needs_scalar_fallback(&matches, top_k, ready_count) {
        matches = query_matches(
            connection,
            &scalar_sql(authority),
            &query_blob,
            partition_id,
            top_k,
        )?;
        if matches.len() != top_k.min(ready_count) {
            return Err(VectorStoreError::Unavailable);
        }
    }
    matches.sort_by(|left, right| {
        left.distance()
            .total_cmp(&right.distance())
            .then_with(|| left.record_id().cmp(&right.record_id()))
    });
    matches.truncate(top_k);
    project_matches(connection, vector_space_id, partition_id, &matches)
}

fn validate_top_k(top_k: usize) -> Result<(), VectorStoreError> {
    if (1..=VECTOR_TOP_K_MAX).contains(&top_k) {
        Ok(())
    } else {
        Err(VectorStoreError::InvalidTopK)
    }
}

fn verify_no_retiring_sources(
    connection: &Connection,
    neighbors: &[ProjectedVectorNeighbor],
) -> Result<(), VectorStoreError> {
    for neighbor in neighbors {
        let retiring = connection
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM decision_retiring_anchors WHERE anchor_id = ?1
                 )",
                [neighbor.anchor_id.to_string()],
                |row| row.get::<_, bool>(0),
            )
            .map_err(|_| VectorStoreError::Unavailable)?;
        if retiring {
            return Err(VectorStoreError::Corrupt);
        }
    }
    Ok(())
}

fn authoritative_ready_count(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
    partition_id: PartitionId,
) -> Result<usize, VectorStoreError> {
    let count = connection
        .query_row(
            "SELECT count(*)
             FROM evidence_vector_links AS link
             JOIN evidence_vector_link_state_events AS state
               ON state.event_seq = (
                    SELECT max(latest.event_seq)
                    FROM evidence_vector_link_state_events AS latest
                    WHERE latest.evidence_vector_link_id = link.evidence_vector_link_id
               )
             WHERE link.vector_space_id = ?1
               AND link.partition_id = ?2
               AND state.state = 'ready'
               AND NOT EXISTS (
                   SELECT 1 FROM decision_retiring_anchors AS marker
                   WHERE marker.anchor_id = link.anchor_id
               )",
            params![vector_space_id.as_str(), partition_id.value()],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|_| VectorStoreError::Unavailable)?;
    usize::try_from(count).map_err(|_| VectorStoreError::Corrupt)
}

fn indexed_partition_count(
    connection: &Connection,
    authority: &Vec0SchemaAuthority,
    partition_id: PartitionId,
) -> Result<usize, VectorStoreError> {
    let sql = format!(
        "SELECT count(*) FROM \"{}\" WHERE partition_id = ?1",
        authority.root().as_str()
    );
    let count = connection
        .query_row(&sql, [partition_id.value()], |row| row.get::<_, i64>(0))
        .map_err(|_| VectorStoreError::Unavailable)?;
    usize::try_from(count).map_err(|_| VectorStoreError::Corrupt)
}

fn knn_sql(authority: &Vec0SchemaAuthority) -> String {
    format!(
        "SELECT record_id, distance FROM \"{}\" \
         WHERE embedding MATCH ?1 AND k = ?2 AND partition_id = ?3 \
         ORDER BY distance",
        authority.root().as_str()
    )
}

fn scalar_sql(authority: &Vec0SchemaAuthority) -> String {
    format!(
        "SELECT record_id, vec_distance_cosine(embedding, ?1) AS distance \
         FROM \"{}\" WHERE partition_id = ?3 \
         ORDER BY distance, record_id LIMIT ?2",
        authority.root().as_str()
    )
}

fn native_vector_blob(vector: &NormalizedVector) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(vector.values()));
    for value in vector.values() {
        bytes.extend_from_slice(&value.to_ne_bytes());
    }
    bytes
}

fn query_matches(
    connection: &Connection,
    sql: &str,
    query_blob: &[u8],
    partition_id: PartitionId,
    limit: usize,
) -> Result<Vec<VectorMatch>, VectorStoreError> {
    let limit = i64::try_from(limit).map_err(|_| VectorStoreError::InvalidTopK)?;
    let mut statement = connection
        .prepare(sql)
        .map_err(|_| VectorStoreError::Unavailable)?;
    let rows = statement
        .query_map(params![query_blob, limit, partition_id.value()], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, f64>(1)?))
        })
        .map_err(|_| VectorStoreError::Unavailable)?;
    let mut matches = Vec::new();
    let mut ids = BTreeSet::new();
    for row in rows {
        let (record_id, distance) = row.map_err(|_| VectorStoreError::Unavailable)?;
        let record_id = parse_record_id(&record_id)?;
        if !distance.is_finite() || !ids.insert(record_id) {
            return Err(VectorStoreError::Corrupt);
        }
        let distance = distance as f32;
        matches.push(VectorMatch::new(record_id, distance).map_err(|_| VectorStoreError::Corrupt)?);
    }
    Ok(matches)
}

fn needs_scalar_fallback(matches: &[VectorMatch], top_k: usize, ready_count: usize) -> bool {
    ready_count > top_k
        && matches.len() > top_k
        && matches[top_k - 1]
            .distance()
            .total_cmp(&matches[top_k].distance())
            .is_eq()
}

fn map_resolver_error(error: LedgerError) -> VectorStoreError {
    if error.class() == LedgerErrorClass::CorruptDatabase {
        VectorStoreError::Corrupt
    } else {
        VectorStoreError::Unavailable
    }
}

fn parse_record_id(value: &str) -> Result<VectorRecordId, VectorStoreError> {
    let value = parse_uuid_v7(value)?;
    VectorRecordId::new(value).map_err(|_| VectorStoreError::Corrupt)
}

fn parse_uuid_v7(value: &str) -> Result<Uuid, VectorStoreError> {
    let value = Uuid::parse_str(value).map_err(|_| VectorStoreError::Corrupt)?;
    if value.get_version_num() != 7 || value.get_variant() != Variant::RFC4122 {
        return Err(VectorStoreError::Corrupt);
    }
    Ok(value)
}

fn project_matches(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
    partition_id: PartitionId,
    matches: &[VectorMatch],
) -> Result<Vec<ProjectedVectorNeighbor>, VectorStoreError> {
    #[cfg(test)]
    if synthetic_projection_fixture(connection)? {
        return project_matches_synthetic(connection, vector_space_id, partition_id, matches);
    }

    matches
        .iter()
        .map(|vector_match| {
            let source = match load_verified_vector_link_source(
                connection,
                vector_space_id,
                vector_match.record_id(),
            )
            .map_err(map_resolver_error)?
            {
                VerifiedVectorLinkSourceLoad::LinkMissing
                | VerifiedVectorLinkSourceLoad::AuthorityMissing => {
                    return Err(VectorStoreError::Unavailable);
                }
                VerifiedVectorLinkSourceLoad::Verified(source) => source,
            };
            if source.record_id != vector_match.record_id()
                || source.vector_space_id != *vector_space_id
                || source.partition_id != partition_id
                || source.vector_record.as_ref().map(VectorRecord::record_id)
                    != Some(vector_match.record_id())
            {
                return Err(VectorStoreError::Corrupt);
            }
            let evaluation = source
                .evaluation
                .map(|evaluation| ProjectedNeighborEvaluation {
                    evaluation_id: evaluation.evaluation_id,
                    source: evaluation.source,
                    binary_label: evaluation.binary_label,
                    judge_confidence: evaluation.judge_confidence,
                    promotion_eligible: evaluation.promotion_eligible,
                    created_at_unix_ms: evaluation.created_at_unix_ms,
                });
            Ok(ProjectedVectorNeighbor {
                vector_match: *vector_match,
                vector_record: source.vector_record,
                shadow_attempt_id: source.shadow_attempt_id,
                shadow_result_id: source.shadow_result_id,
                anchor_id: source.anchor_id,
                root_uuid: source.root_uuid,
                learning_generation_id: source.learning_generation_id,
                terminal_class: source.terminal_class,
                evaluation,
            })
        })
        .collect()
}

#[cfg(test)]
fn synthetic_projection_fixture(connection: &Connection) -> Result<bool, VectorStoreError> {
    connection
        .query_row(
            "SELECT NOT EXISTS(
                SELECT 1 FROM pragma_table_info('evidence_vector_links')
                WHERE name = 'canonical_payload_hash'
             )",
            [],
            |row| row.get(0),
        )
        .map_err(|_| VectorStoreError::Unavailable)
}

#[cfg(test)]
fn project_matches_synthetic(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
    partition_id: PartitionId,
    matches: &[VectorMatch],
) -> Result<Vec<ProjectedVectorNeighbor>, VectorStoreError> {
    let requested = matches
        .iter()
        .map(|vector_match| vector_match.record_id().to_string())
        .collect::<Vec<_>>();
    let requested_json =
        serde_json::to_string(&requested).map_err(|_| VectorStoreError::Corrupt)?;
    let mut statement = connection
        .prepare(
            "WITH requested(record_id) AS (
                SELECT CAST(value AS TEXT) FROM json_each(?1)
             )
             SELECT requested.record_id,
                    link.evidence_vector_link_id, link.shadow_attempt_id,
                    link.anchor_id, link.root_uuid, link.learning_generation_id,
                    link.terminal_class, link.evaluation_id, link.quality_label,
                    link.vector_space_id, link.partition_id, link.canonical_query_hash,
                    state.state, state.embedding_id,
                    attempt.shadow_attempt_id, attempt.anchor_id,
                    attempt.learning_generation_id,
                    result.shadow_result_id, result.shadow_attempt_id,
                    result.terminal_class, result.evaluation_id, result.canonicalizable,
                    anchor.anchor_id, anchor.root_uuid,
                    embedding.embedding_id, embedding.vector_space_id,
                    embedding.canonical_query_hash,
                    evaluation.evaluation_id, evaluation.shadow_attempt_id,
                    evaluation.source, evaluation.judge_confidence,
                    evaluation.judge_confidence_bits, evaluation.binary_label,
                    evaluation.promotion_eligible, evaluation.created_at_unix_ms
             FROM requested
             LEFT JOIN evidence_vector_links AS link
               ON link.evidence_vector_link_id = requested.record_id
             LEFT JOIN evidence_vector_link_state_events AS state
               ON state.event_seq = (
                    SELECT max(latest.event_seq)
                    FROM evidence_vector_link_state_events AS latest
                    WHERE latest.evidence_vector_link_id = link.evidence_vector_link_id
               )
             LEFT JOIN shadow_attempts AS attempt
               ON attempt.shadow_attempt_id = link.shadow_attempt_id
             LEFT JOIN shadow_results AS result
               ON result.shadow_attempt_id = link.shadow_attempt_id
             LEFT JOIN anchors AS anchor ON anchor.anchor_id = link.anchor_id
             LEFT JOIN embeddings AS embedding ON embedding.embedding_id = state.embedding_id
             LEFT JOIN evaluations AS evaluation ON evaluation.evaluation_id = link.evaluation_id",
        )
        .map_err(|_| VectorStoreError::Unavailable)?;
    let rows = statement
        .query_map([requested_json], raw_projection)
        .map_err(|_| VectorStoreError::Unavailable)?;
    let match_by_id = matches
        .iter()
        .map(|vector_match| (vector_match.record_id(), *vector_match))
        .collect::<BTreeMap<_, _>>();
    let mut projected = BTreeMap::new();
    for row in rows {
        let raw = row.map_err(|_| VectorStoreError::Unavailable)?;
        let requested_id = parse_record_id(&raw.requested_record_id)?;
        let vector_match = match_by_id
            .get(&requested_id)
            .copied()
            .ok_or(VectorStoreError::Corrupt)?;
        let neighbor = validate_projection(raw, vector_match, vector_space_id, partition_id)?;
        if projected.insert(requested_id, neighbor).is_some() {
            return Err(VectorStoreError::Corrupt);
        }
    }
    if projected.len() != matches.len() {
        return Err(VectorStoreError::Unavailable);
    }
    matches
        .iter()
        .map(|vector_match| {
            projected
                .remove(&vector_match.record_id())
                .ok_or(VectorStoreError::Unavailable)
        })
        .collect()
}

#[derive(Debug)]
struct RawProjection {
    requested_record_id: String,
    link_id: Option<String>,
    link_shadow_attempt_id: Option<String>,
    link_anchor_id: Option<String>,
    link_root_uuid: Option<String>,
    link_learning_generation_id: Option<String>,
    link_terminal_class: Option<String>,
    link_evaluation_id: Option<String>,
    link_quality_label: Option<String>,
    link_vector_space_id: Option<String>,
    link_partition_id: Option<i64>,
    link_query_hash: Option<String>,
    state: Option<String>,
    state_embedding_id: Option<String>,
    attempt_id: Option<String>,
    attempt_anchor_id: Option<String>,
    attempt_learning_generation_id: Option<String>,
    result_id: Option<String>,
    result_attempt_id: Option<String>,
    result_terminal_class: Option<String>,
    result_evaluation_id: Option<String>,
    result_canonicalizable: Option<i64>,
    anchor_id: Option<String>,
    anchor_root_uuid: Option<String>,
    embedding_id: Option<String>,
    embedding_vector_space_id: Option<String>,
    embedding_query_hash: Option<String>,
    evaluation_id: Option<String>,
    evaluation_attempt_id: Option<String>,
    evaluation_source: Option<String>,
    evaluation_confidence: Option<f64>,
    evaluation_confidence_bits: Option<i64>,
    evaluation_binary_label: Option<String>,
    evaluation_promotion_eligible: Option<i64>,
    evaluation_created_at_unix_ms: Option<i64>,
}

fn raw_projection(row: &rusqlite::Row<'_>) -> rusqlite::Result<RawProjection> {
    Ok(RawProjection {
        requested_record_id: row.get(0)?,
        link_id: row.get(1)?,
        link_shadow_attempt_id: row.get(2)?,
        link_anchor_id: row.get(3)?,
        link_root_uuid: row.get(4)?,
        link_learning_generation_id: row.get(5)?,
        link_terminal_class: row.get(6)?,
        link_evaluation_id: row.get(7)?,
        link_quality_label: row.get(8)?,
        link_vector_space_id: row.get(9)?,
        link_partition_id: row.get(10)?,
        link_query_hash: row.get(11)?,
        state: row.get(12)?,
        state_embedding_id: row.get(13)?,
        attempt_id: row.get(14)?,
        attempt_anchor_id: row.get(15)?,
        attempt_learning_generation_id: row.get(16)?,
        result_id: row.get(17)?,
        result_attempt_id: row.get(18)?,
        result_terminal_class: row.get(19)?,
        result_evaluation_id: row.get(20)?,
        result_canonicalizable: row.get(21)?,
        anchor_id: row.get(22)?,
        anchor_root_uuid: row.get(23)?,
        embedding_id: row.get(24)?,
        embedding_vector_space_id: row.get(25)?,
        embedding_query_hash: row.get(26)?,
        evaluation_id: row.get(27)?,
        evaluation_attempt_id: row.get(28)?,
        evaluation_source: row.get(29)?,
        evaluation_confidence: row.get(30)?,
        evaluation_confidence_bits: row.get(31)?,
        evaluation_binary_label: row.get(32)?,
        evaluation_promotion_eligible: row.get(33)?,
        evaluation_created_at_unix_ms: row.get(34)?,
    })
}

fn validate_projection(
    raw: RawProjection,
    vector_match: VectorMatch,
    vector_space_id: &VectorSpaceId,
    partition_id: PartitionId,
) -> Result<ProjectedVectorNeighbor, VectorStoreError> {
    let link_id = required(raw.link_id.as_deref())?;
    if parse_record_id(link_id)? != vector_match.record_id()
        || raw.requested_record_id != link_id
        || required(raw.link_vector_space_id.as_deref())? != vector_space_id.as_str()
        || required(raw.link_partition_id)? != partition_id.value()
        || required(raw.state.as_deref())? != "ready"
    {
        return Err(VectorStoreError::Corrupt);
    }

    let shadow_attempt_id = parse_uuid_v7(required(raw.link_shadow_attempt_id.as_deref())?)?;
    let anchor_id = parse_uuid_v7(required(raw.link_anchor_id.as_deref())?)?;
    let root_uuid = parse_uuid_v7(required(raw.link_root_uuid.as_deref())?)?;
    let learning_generation_id =
        parse_uuid_v7(required(raw.link_learning_generation_id.as_deref())?)?;
    let terminal_class = parse_terminal_class(required(raw.link_terminal_class.as_deref())?)?;
    let evaluation = validate_evaluation(&raw, shadow_attempt_id, terminal_class)?;
    let query_hash = required(raw.link_query_hash.as_deref())?;
    let state_embedding_id = required(raw.state_embedding_id.as_deref())?;

    if parse_uuid_v7(required(raw.attempt_id.as_deref())?)? != shadow_attempt_id
        || parse_uuid_v7(required(raw.attempt_anchor_id.as_deref())?)? != anchor_id
        || parse_uuid_v7(required(raw.attempt_learning_generation_id.as_deref())?)?
            != learning_generation_id
        || parse_uuid_v7(required(raw.result_attempt_id.as_deref())?)? != shadow_attempt_id
        || parse_terminal_class(required(raw.result_terminal_class.as_deref())?)? != terminal_class
        || required(raw.result_canonicalizable)? != 1
        || parse_uuid_v7(required(raw.anchor_id.as_deref())?)? != anchor_id
        || parse_uuid_v7(required(raw.anchor_root_uuid.as_deref())?)? != root_uuid
        || required(raw.embedding_id.as_deref())? != state_embedding_id
        || required(raw.embedding_vector_space_id.as_deref())? != vector_space_id.as_str()
        || required(raw.embedding_query_hash.as_deref())? != query_hash
    {
        return Err(VectorStoreError::Corrupt);
    }
    let shadow_result_id = parse_uuid_v7(required(raw.result_id.as_deref())?)?;
    Ok(ProjectedVectorNeighbor {
        vector_match,
        vector_record: None,
        shadow_attempt_id,
        shadow_result_id,
        anchor_id,
        root_uuid,
        learning_generation_id,
        terminal_class,
        evaluation,
    })
}

fn validate_evaluation(
    raw: &RawProjection,
    shadow_attempt_id: Uuid,
    terminal_class: ShadowTerminalClass,
) -> Result<Option<ProjectedNeighborEvaluation>, VectorStoreError> {
    if raw.link_evaluation_id != raw.result_evaluation_id {
        return Err(VectorStoreError::Corrupt);
    }
    let Some(link_evaluation_id) = raw.link_evaluation_id.as_deref() else {
        if raw.link_quality_label.is_some()
            || raw.evaluation_id.is_some()
            || raw.evaluation_attempt_id.is_some()
            || raw.evaluation_source.is_some()
            || raw.evaluation_confidence.is_some()
            || raw.evaluation_confidence_bits.is_some()
            || raw.evaluation_binary_label.is_some()
            || raw.evaluation_promotion_eligible.is_some()
            || raw.evaluation_created_at_unix_ms.is_some()
            || matches!(
                terminal_class,
                ShadowTerminalClass::Completed | ShadowTerminalClass::DeterministicFailure
            )
        {
            return Err(VectorStoreError::Corrupt);
        }
        return Ok(None);
    };

    let evaluation_id = parse_uuid_v7(link_evaluation_id)?;
    if parse_uuid_v7(&required(raw.evaluation_id.clone())?)? != evaluation_id
        || parse_uuid_v7(&required(raw.evaluation_attempt_id.clone())?)? != shadow_attempt_id
    {
        return Err(VectorStoreError::Corrupt);
    }
    let source = parse_evaluation_source(&required(raw.evaluation_source.clone())?)?;
    let binary_label = raw
        .evaluation_binary_label
        .as_deref()
        .map(parse_binary_label)
        .transpose()?;
    let link_quality_label = raw
        .link_quality_label
        .as_deref()
        .map(parse_binary_label)
        .transpose()?;
    let promotion_eligible = parse_bool(required(raw.evaluation_promotion_eligible)?)?;
    validate_link_quality(link_quality_label, binary_label, promotion_eligible)?;
    if !matches!(
        terminal_class,
        ShadowTerminalClass::Completed | ShadowTerminalClass::DeterministicFailure
    ) {
        return Err(VectorStoreError::Corrupt);
    }
    let judge_confidence = parse_score(raw.evaluation_confidence, raw.evaluation_confidence_bits)?;
    if matches!(source, JudgeEvaluationSourceV1::Judge) != judge_confidence.is_some() {
        return Err(VectorStoreError::Corrupt);
    }
    let created_at_unix_ms = required(raw.evaluation_created_at_unix_ms)?;
    if created_at_unix_ms < 0 {
        return Err(VectorStoreError::Corrupt);
    }
    Ok(Some(ProjectedNeighborEvaluation {
        evaluation_id,
        source,
        binary_label,
        judge_confidence,
        promotion_eligible,
        created_at_unix_ms,
    }))
}

fn validate_link_quality(
    link_quality_label: Option<JudgeBinaryLabelV1>,
    evaluation_binary_label: Option<JudgeBinaryLabelV1>,
    promotion_eligible: bool,
) -> Result<(), VectorStoreError> {
    let valid = if promotion_eligible {
        evaluation_binary_label.is_some() && link_quality_label == evaluation_binary_label
    } else {
        link_quality_label.is_none()
    };
    valid.then_some(()).ok_or(VectorStoreError::Corrupt)
}

fn parse_terminal_class(value: &str) -> Result<ShadowTerminalClass, VectorStoreError> {
    match value {
        "completed" => Ok(ShadowTerminalClass::Completed),
        "deterministic_failure" => Ok(ShadowTerminalClass::DeterministicFailure),
        "operational_failure" => Ok(ShadowTerminalClass::OperationalFailure),
        "skipped_cooloff" => Ok(ShadowTerminalClass::SkippedCooloff),
        "canceled_shutdown" => Ok(ShadowTerminalClass::CanceledShutdown),
        "orphaned_before_schedule" => Ok(ShadowTerminalClass::OrphanedBeforeSchedule),
        "orphaned_in_flight" => Ok(ShadowTerminalClass::OrphanedInFlight),
        _ => Err(VectorStoreError::Corrupt),
    }
}

fn parse_evaluation_source(value: &str) -> Result<JudgeEvaluationSourceV1, VectorStoreError> {
    match value {
        "deterministic_validator" => Ok(JudgeEvaluationSourceV1::DeterministicValidator),
        "judge" => Ok(JudgeEvaluationSourceV1::Judge),
        _ => Err(VectorStoreError::Corrupt),
    }
}

fn parse_binary_label(value: &str) -> Result<JudgeBinaryLabelV1, VectorStoreError> {
    match value {
        "pass" => Ok(JudgeBinaryLabelV1::Pass),
        "fail" => Ok(JudgeBinaryLabelV1::Fail),
        _ => Err(VectorStoreError::Corrupt),
    }
}

fn parse_score(
    value: Option<f64>,
    bits: Option<i64>,
) -> Result<Option<ScoredValueV1>, VectorStoreError> {
    match (value, bits) {
        (None, None) => Ok(None),
        (Some(value), Some(bits)) if value.is_finite() && (0.0..=1.0).contains(&value) => {
            let score = ScoredValueV1::new(value);
            if score.bits as i64 != bits {
                return Err(VectorStoreError::Corrupt);
            }
            Ok(Some(score))
        }
        _ => Err(VectorStoreError::Corrupt),
    }
}

fn parse_bool(value: i64) -> Result<bool, VectorStoreError> {
    match value {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(VectorStoreError::Corrupt),
    }
}

fn required<T>(value: Option<T>) -> Result<T, VectorStoreError> {
    value.ok_or(VectorStoreError::Unavailable)
}

#[cfg(test)]
mod tests {
    use rusqlite::{Connection, named_params};
    use serde_json::json;

    use super::*;
    use crate::canonical_json::canonical_sha256;
    use crate::ledger::repository::{
        ready_evaluated_runtime_fixture, ready_retention_runtime_fixture,
    };
    use crate::sqlite_vec_extension::{SqliteVecStatus, register as register_sqlite_vec};
    use crate::sqlite_vec_schema::{Vec0RootName, VectorIndexGeneration};
    use crate::vector::{AuthoritativeVector, VectorDimensions};
    use crate::vector_store::{BruteForceMemoryStore, VectorRecord, VectorSpaceSpec, VectorStore};

    struct SearchFixture {
        connection: Connection,
        space_id: VectorSpaceId,
        authority: Vec0SchemaAuthority,
        dimensions: VectorDimensions,
    }

    impl SearchFixture {
        fn in_memory(dimensions: u32) -> Self {
            assert_eq!(register_sqlite_vec(), SqliteVecStatus::Available);
            Self::from_connection(Connection::open_in_memory().unwrap(), dimensions)
        }

        fn from_connection(connection: Connection, dimensions: u32) -> Self {
            connection
                .execute_batch(
                    "PRAGMA journal_mode = WAL;
                     CREATE TABLE vector_spaces (
                        vector_space_id TEXT PRIMARY KEY,
                        dimensions INTEGER NOT NULL
                     );
                     CREATE TABLE vector_index_manifest (
                        vector_space_id TEXT NOT NULL,
                        generation INTEGER NOT NULL,
                        state TEXT NOT NULL,
                        root_table_name TEXT NOT NULL,
                        dimensions INTEGER NOT NULL,
                        expected_schema_objects_json TEXT NOT NULL,
                        expected_schema_objects_sha256 TEXT NOT NULL,
                        base_source_seq INTEGER NOT NULL,
                        applied_source_seq INTEGER NOT NULL,
                        build_cursor_record_id TEXT,
                        source_record_count INTEGER,
                        source_fingerprint_sha256 TEXT,
                        stable_error_class TEXT,
                        created_at_unix_ms INTEGER NOT NULL,
                        activated_at_unix_ms INTEGER,
                        retired_at_unix_ms INTEGER,
                        dropped_at_unix_ms INTEGER,
                        canonical_payload_hash TEXT NOT NULL,
                        PRIMARY KEY (vector_space_id, generation)
                     );
                     CREATE TABLE evidence_vector_links (
                        evidence_vector_link_id TEXT PRIMARY KEY,
                        shadow_attempt_id TEXT NOT NULL,
                        anchor_id TEXT NOT NULL,
                        root_uuid TEXT NOT NULL,
                        learning_generation_id TEXT NOT NULL,
                        terminal_class TEXT NOT NULL,
                        evaluation_id TEXT,
                        quality_label TEXT,
                        vector_space_id TEXT NOT NULL,
                        partition_id INTEGER NOT NULL,
                        canonical_query_hash TEXT NOT NULL
                     );
                     CREATE INDEX fixture_links_partition
                        ON evidence_vector_links(vector_space_id, partition_id);
                     CREATE TABLE evidence_vector_link_state_events (
                        event_seq INTEGER PRIMARY KEY AUTOINCREMENT,
                        evidence_vector_link_id TEXT NOT NULL,
                        state TEXT NOT NULL,
                        embedding_id TEXT
                     );
                     CREATE INDEX fixture_link_state_latest
                        ON evidence_vector_link_state_events(
                            evidence_vector_link_id, event_seq DESC
                        );
                     CREATE TABLE shadow_attempts (
                        shadow_attempt_id TEXT PRIMARY KEY,
                        anchor_id TEXT NOT NULL,
                        learning_generation_id TEXT NOT NULL
                     );
                     CREATE TABLE shadow_results (
                        shadow_result_id TEXT PRIMARY KEY,
                        shadow_attempt_id TEXT NOT NULL,
                        terminal_class TEXT NOT NULL,
                        evaluation_id TEXT,
                        canonicalizable INTEGER NOT NULL
                     );
                     CREATE TABLE anchors (
                        anchor_id TEXT PRIMARY KEY,
                        root_uuid TEXT NOT NULL
                     );
                     CREATE TABLE decision_retiring_anchors (
                        anchor_id TEXT PRIMARY KEY
                     );
                     CREATE TABLE embeddings (
                        embedding_id TEXT PRIMARY KEY,
                        vector_space_id TEXT NOT NULL,
                        canonical_query_hash TEXT NOT NULL
                     );
                     CREATE TABLE evaluations (
                        evaluation_id TEXT PRIMARY KEY,
                        shadow_attempt_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        judge_confidence REAL,
                        judge_confidence_bits INTEGER,
                        binary_label TEXT,
                        promotion_eligible INTEGER NOT NULL,
                        created_at_unix_ms INTEGER NOT NULL
                     );",
                )
                .unwrap();
            let dimensions = VectorDimensions::new(dimensions).unwrap();
            let space_id = VectorSpaceId::new("a".repeat(64)).unwrap();
            let authority = Vec0SchemaAuthority::new(
                Vec0RootName::new(space_id.clone(), VectorIndexGeneration::new(1).unwrap()),
                dimensions,
            );
            connection
                .execute(
                    "INSERT INTO vector_spaces VALUES (?1, ?2)",
                    params![space_id.as_str(), i64::from(dimensions.value())],
                )
                .unwrap();
            connection.execute_batch(authority.create_sql()).unwrap();
            let fixture = Self {
                connection,
                space_id,
                authority,
                dimensions,
            };
            fixture.set_manifest_count(0);
            fixture
        }

        fn set_manifest_count(&self, count: u64) {
            let fingerprint = "f".repeat(64);
            let payload_hash = canonical_sha256(&json!({
                "vector_space_id": self.space_id.as_str(),
                "generation": 1,
                "state": "active",
                "root_table_name": self.authority.root().as_str(),
                "dimensions": self.dimensions.value(),
                "expected_schema_objects_json": self.authority.manifest_json(),
                "expected_schema_objects_sha256": self.authority.manifest_sha256(),
                "base_source_seq": 0,
                "applied_source_seq": 0,
                "build_cursor_record_id": null,
                "source_record_count": count,
                "source_fingerprint_sha256": fingerprint,
                "stable_error_class": null,
                "created_at_unix_ms": 0,
                "activated_at_unix_ms": 1,
                "retired_at_unix_ms": null,
                "dropped_at_unix_ms": null,
            }))
            .unwrap();
            self.connection
                .execute(
                    "INSERT OR REPLACE INTO vector_index_manifest (
                        vector_space_id, generation, state, root_table_name, dimensions,
                        expected_schema_objects_json, expected_schema_objects_sha256,
                        base_source_seq, applied_source_seq, build_cursor_record_id,
                        source_record_count, source_fingerprint_sha256, stable_error_class,
                        created_at_unix_ms, activated_at_unix_ms, retired_at_unix_ms,
                        dropped_at_unix_ms, canonical_payload_hash
                     ) VALUES (
                        :space, 1, 'active', :root, :dimensions,
                        :objects, :objects_hash, 0, 0, NULL,
                        :record_count, :fingerprint, NULL, 0, 1, NULL, NULL, :payload_hash
                     )",
                    named_params! {
                        ":space": self.space_id.as_str(),
                        ":root": self.authority.root().as_str(),
                        ":dimensions": i64::from(self.dimensions.value()),
                        ":objects": self.authority.manifest_json(),
                        ":objects_hash": self.authority.manifest_sha256(),
                        ":record_count": i64::try_from(count).unwrap(),
                        ":fingerprint": fingerprint,
                        ":payload_hash": payload_hash,
                    },
                )
                .unwrap();
        }

        fn seed_operational(&self, index: u64, partition_id: i64, values: &[f64]) -> VectorRecord {
            let record_id = vector_record_id(index);
            let shadow_attempt_id = fixture_uuid(100_000 + index);
            let shadow_result_id = fixture_uuid(200_000 + index);
            let anchor_id = fixture_uuid(300_000 + index);
            let root_uuid = fixture_uuid(400_000 + index);
            let learning_generation_id = fixture_uuid(500_000);
            let embedding_id = fixture_uuid(600_000 + index);
            let query_hash = format!("{index:064x}");
            let partition_id = PartitionId::new(partition_id).unwrap();
            let normalized = NormalizedVector::from_provider_f64(values, self.dimensions).unwrap();
            let vector = AuthoritativeVector::from_normalized(&self.space_id, normalized).unwrap();
            let insert_vec = format!(
                "INSERT INTO \"{}\" (record_id, embedding, partition_id)
                 VALUES (?1, ?2, ?3)",
                self.authority.root().as_str()
            );
            self.connection
                .execute(
                    &insert_vec,
                    params![
                        record_id.to_string(),
                        vector.blob().native_endian_bytes(),
                        partition_id.value(),
                    ],
                )
                .unwrap();
            self.connection
                .execute(
                    "INSERT INTO anchors VALUES (?1, ?2)",
                    params![anchor_id.to_string(), root_uuid.to_string()],
                )
                .unwrap();
            self.connection
                .execute(
                    "INSERT INTO shadow_attempts VALUES (?1, ?2, ?3)",
                    params![
                        shadow_attempt_id.to_string(),
                        anchor_id.to_string(),
                        learning_generation_id.to_string(),
                    ],
                )
                .unwrap();
            self.connection
                .execute(
                    "INSERT INTO shadow_results VALUES (?1, ?2, 'operational_failure', NULL, 1)",
                    params![shadow_result_id.to_string(), shadow_attempt_id.to_string()],
                )
                .unwrap();
            self.connection
                .execute(
                    "INSERT INTO embeddings VALUES (?1, ?2, ?3)",
                    params![
                        embedding_id.to_string(),
                        self.space_id.as_str(),
                        &query_hash,
                    ],
                )
                .unwrap();
            self.connection
                .execute(
                    "INSERT INTO evidence_vector_links VALUES (
                        ?1, ?2, ?3, ?4, ?5, 'operational_failure', NULL, NULL,
                        ?6, ?7, ?8
                     )",
                    params![
                        record_id.to_string(),
                        shadow_attempt_id.to_string(),
                        anchor_id.to_string(),
                        root_uuid.to_string(),
                        learning_generation_id.to_string(),
                        self.space_id.as_str(),
                        partition_id.value(),
                        query_hash,
                    ],
                )
                .unwrap();
            self.connection
                .execute(
                    "INSERT INTO evidence_vector_link_state_events (
                        evidence_vector_link_id, state, embedding_id
                     ) VALUES (?1, 'ready', ?2)",
                    params![record_id.to_string(), embedding_id.to_string()],
                )
                .unwrap();
            VectorRecord::new(record_id, self.space_id.clone(), partition_id, vector).unwrap()
        }

        fn refresh_manifest_count(&self) {
            let sql = format!(
                "SELECT count(*) FROM \"{}\"",
                self.authority.root().as_str()
            );
            let count = self
                .connection
                .query_row(&sql, [], |row| row.get::<_, i64>(0))
                .unwrap();
            self.set_manifest_count(u64::try_from(count).unwrap());
        }
    }

    fn fixture_uuid(index: u64) -> Uuid {
        Uuid::parse_str(&format!("01890f47-6c7d-7000-8000-{index:012x}")).unwrap()
    }

    fn vector_record_id(index: u64) -> VectorRecordId {
        VectorRecordId::new(fixture_uuid(index)).unwrap()
    }

    fn authority() -> Vec0SchemaAuthority {
        Vec0SchemaAuthority::new(
            Vec0RootName::new(
                VectorSpaceId::new("a".repeat(64)).unwrap(),
                VectorIndexGeneration::new(7).unwrap(),
            ),
            crate::vector::VectorDimensions::new(3).unwrap(),
        )
    }

    fn vector_match(index: u64, distance: f32) -> VectorMatch {
        let uuid = Uuid::parse_str(&format!("01890f47-6c7d-7000-8000-{index:012x}")).unwrap();
        VectorMatch::new(VectorRecordId::new(uuid).unwrap(), distance).unwrap()
    }

    fn query(values: &[f64], dimensions: VectorDimensions) -> NormalizedVector {
        NormalizedVector::from_provider_f64(values, dimensions).unwrap()
    }

    #[test]
    fn strict_partition_excludes_a_closer_wrong_partition_record() {
        let fixture = SearchFixture::in_memory(2);
        fixture.seed_operational(1, 2, &[1.0, 0.0]);
        fixture.seed_operational(2, 1, &[0.0, 1.0]);
        fixture.refresh_manifest_count();

        let neighbors = search_projected_neighbors(
            &fixture.connection,
            &fixture.space_id,
            PartitionId::new(1).unwrap(),
            &query(&[1.0, 0.0], fixture.dimensions),
            1,
        )
        .unwrap();
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].vector_match.record_id(), vector_record_id(2));
        assert_eq!(neighbors[0].vector_match.distance(), 1.0);
        assert_eq!(
            search_projected_neighbors(
                &fixture.connection,
                &fixture.space_id,
                PartitionId::new(1).unwrap(),
                &query(&[1.0, 0.0], fixture.dimensions),
                VECTOR_TOP_K_MAX + 1,
            ),
            Err(VectorStoreError::InvalidTopK)
        );
    }

    #[test]
    fn empty_partition_does_not_fall_back_to_exact_match_in_another_partition() {
        let fixture = SearchFixture::in_memory(2);
        fixture.seed_operational(1, 2, &[1.0, 0.0]);
        fixture.refresh_manifest_count();

        let neighbors = search_projected_neighbors(
            &fixture.connection,
            &fixture.space_id,
            PartitionId::new(1).unwrap(),
            &query(&[1.0, 0.0], fixture.dimensions),
            1,
        )
        .unwrap();

        assert!(neighbors.is_empty());
    }

    #[test]
    fn randomized_memory_and_sqlite_searches_agree() {
        let fixture = SearchFixture::in_memory(4);
        let memory = BruteForceMemoryStore::new(96).unwrap();
        memory
            .ensure_space(VectorSpaceSpec::new(
                fixture.space_id.clone(),
                fixture.dimensions,
            ))
            .unwrap();
        let mut seed = 0x4d59_5df4_d0f3_3173_u64;
        for index in 1..=96_u64 {
            let mut values = [0.0_f64; 4];
            for value in &mut values {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                *value = f64::from(u32::try_from(seed >> 32).unwrap() % 2_001) - 1_000.0;
            }
            if values.iter().all(|value| *value == 0.0) {
                values[0] = 1.0;
            }
            let partition = i64::try_from((index - 1) % 3 + 1).unwrap();
            let record = fixture.seed_operational(index, partition, &values);
            memory.upsert(record).unwrap();
        }
        fixture.refresh_manifest_count();
        let query = query(&[31.0, -17.0, 11.0, 5.0], fixture.dimensions);
        for partition in 1..=3 {
            let partition = PartitionId::new(partition).unwrap();
            let expected = memory
                .search(&fixture.space_id, partition, &query, 17)
                .unwrap();
            let actual = search_projected_neighbors(
                &fixture.connection,
                &fixture.space_id,
                partition,
                &query,
                17,
            )
            .unwrap();
            assert_eq!(actual.len(), expected.len());
            for (actual, expected) in actual.iter().zip(expected) {
                assert_eq!(actual.vector_match.record_id(), expected.record_id());
                assert!(
                    (actual.vector_match.distance() - expected.distance()).abs() <= 1.0e-5,
                    "SQLite={} memory={}",
                    actual.vector_match.distance(),
                    expected.distance(),
                );
            }
        }
    }

    #[test]
    fn wide_cutoff_tie_falls_back_to_stable_record_id_order() {
        const RECORDS: u64 = 4_100;

        let fixture = SearchFixture::in_memory(2);
        for index in 1..=RECORDS {
            fixture.seed_operational(index, 1, &[1.0, 0.0]);
        }
        fixture.refresh_manifest_count();
        let neighbors = search_projected_neighbors(
            &fixture.connection,
            &fixture.space_id,
            PartitionId::new(1).unwrap(),
            &query(&[1.0, 0.0], fixture.dimensions),
            VECTOR_TOP_K_MAX,
        )
        .unwrap();
        assert_eq!(neighbors.len(), VECTOR_TOP_K_MAX);
        assert_eq!(neighbors[0].vector_match.record_id(), vector_record_id(1));
        assert_eq!(
            neighbors.last().unwrap().vector_match.record_id(),
            vector_record_id(u64::try_from(VECTOR_TOP_K_MAX).unwrap())
        );
        assert!(
            neighbors
                .iter()
                .all(|neighbor| neighbor.vector_match.distance() == 0.0)
        );
    }

    #[test]
    fn a_missing_projected_relation_invalidates_the_whole_neighborhood() {
        let fixture = SearchFixture::in_memory(2);
        fixture.seed_operational(1, 1, &[1.0, 0.0]);
        fixture.seed_operational(2, 1, &[0.0, 1.0]);
        fixture.refresh_manifest_count();
        fixture
            .connection
            .execute(
                "DELETE FROM shadow_results WHERE shadow_attempt_id = ?1",
                [fixture_uuid(100_002).to_string()],
            )
            .unwrap();

        let result = search_projected_neighbors(
            &fixture.connection,
            &fixture.space_id,
            PartitionId::new(1).unwrap(),
            &query(&[1.0, 0.0], fixture.dimensions),
            2,
        );
        assert_eq!(result, Err(VectorStoreError::Unavailable));
    }

    #[test]
    fn production_result_id_with_stale_hash_is_corrupt() {
        let (_temporary, _config, activated, vector_space_id, _root, link_id) =
            ready_retention_runtime_fixture();
        let source = match load_verified_vector_link_source(
            &activated.repository.connection,
            &vector_space_id,
            VectorRecordId::new(link_id).unwrap(),
        )
        .unwrap()
        {
            VerifiedVectorLinkSourceLoad::Verified(source) => source,
            _ => panic!("ready production source did not verify"),
        };
        let query = source
            .vector_record
            .as_ref()
            .unwrap()
            .vector()
            .vector()
            .clone();
        assert_eq!(
            search_projected_neighbors(
                &activated.repository.connection,
                &vector_space_id,
                source.partition_id,
                &query,
                1,
            )
            .unwrap()
            .len(),
            1
        );
        activated
            .repository
            .connection
            .execute(
                "UPDATE shadow_results SET shadow_result_id = ?1
                 WHERE shadow_attempt_id = ?2",
                params![
                    Uuid::now_v7().to_string(),
                    source.shadow_attempt_id.to_string()
                ],
            )
            .unwrap();

        assert_eq!(
            search_projected_neighbors(
                &activated.repository.connection,
                &vector_space_id,
                source.partition_id,
                &query,
                1,
            ),
            Err(VectorStoreError::Corrupt)
        );
    }

    #[test]
    fn production_evaluation_timestamp_with_stale_hash_is_corrupt() {
        let (_temporary, _config, activated, vector_space_id, _root, link_id, evaluation_id) =
            ready_evaluated_runtime_fixture();
        let source = match load_verified_vector_link_source(
            &activated.repository.connection,
            &vector_space_id,
            VectorRecordId::new(link_id).unwrap(),
        )
        .unwrap()
        {
            VerifiedVectorLinkSourceLoad::Verified(source) => source,
            _ => panic!("evaluated production source did not verify"),
        };
        assert_eq!(
            source.evaluation.as_ref().map(|value| value.evaluation_id),
            Some(evaluation_id)
        );
        let query = source
            .vector_record
            .as_ref()
            .unwrap()
            .vector()
            .vector()
            .clone();
        assert_eq!(
            search_projected_neighbors(
                &activated.repository.connection,
                &vector_space_id,
                source.partition_id,
                &query,
                1,
            )
            .unwrap()
            .len(),
            1
        );
        activated
            .repository
            .connection
            .execute(
                "UPDATE evaluations SET created_at_unix_ms = created_at_unix_ms + 1
                 WHERE evaluation_id = ?1",
                params![evaluation_id.to_string()],
            )
            .unwrap();

        assert_eq!(
            search_projected_neighbors(
                &activated.repository.connection,
                &vector_space_id,
                source.partition_id,
                &query,
                1,
            ),
            Err(VectorStoreError::Corrupt)
        );
    }

    #[test]
    fn missing_production_result_is_unavailable() {
        let (_temporary, _config, activated, vector_space_id, _root, link_id) =
            ready_retention_runtime_fixture();
        let source = match load_verified_vector_link_source(
            &activated.repository.connection,
            &vector_space_id,
            VectorRecordId::new(link_id).unwrap(),
        )
        .unwrap()
        {
            VerifiedVectorLinkSourceLoad::Verified(source) => source,
            _ => panic!("ready production source did not verify"),
        };
        let query = source
            .vector_record
            .as_ref()
            .unwrap()
            .vector()
            .vector()
            .clone();
        activated
            .repository
            .connection
            .execute(
                "DELETE FROM shadow_results WHERE shadow_attempt_id = ?1",
                params![source.shadow_attempt_id.to_string()],
            )
            .unwrap();

        assert_eq!(
            search_projected_neighbors(
                &activated.repository.connection,
                &vector_space_id,
                source.partition_id,
                &query,
                1,
            ),
            Err(VectorStoreError::Unavailable)
        );
    }

    #[test]
    fn partial_evaluation_is_projected_without_a_link_quality_label() {
        let fixture = SearchFixture::in_memory(2);
        fixture.seed_operational(1, 1, &[1.0, 0.0]);
        let evaluation_id = fixture_uuid(700_001);
        let attempt_id = fixture_uuid(100_001);
        fixture
            .connection
            .execute(
                "INSERT INTO evaluations VALUES (
                    ?1, ?2, 'judge', ?3, ?4, 'pass', 0, 10
                 )",
                params![
                    evaluation_id.to_string(),
                    attempt_id.to_string(),
                    0.9_f64,
                    0.9_f64.to_bits() as i64,
                ],
            )
            .unwrap();
        fixture
            .connection
            .execute(
                "UPDATE evidence_vector_links
                 SET terminal_class = 'completed', evaluation_id = ?1, quality_label = NULL
                 WHERE evidence_vector_link_id = ?2",
                params![evaluation_id.to_string(), vector_record_id(1).to_string()],
            )
            .unwrap();
        fixture
            .connection
            .execute(
                "UPDATE shadow_results
                 SET terminal_class = 'completed', evaluation_id = ?1
                 WHERE shadow_attempt_id = ?2",
                params![evaluation_id.to_string(), attempt_id.to_string()],
            )
            .unwrap();
        fixture.refresh_manifest_count();

        let neighbors = search_projected_neighbors(
            &fixture.connection,
            &fixture.space_id,
            PartitionId::new(1).unwrap(),
            &query(&[1.0, 0.0], fixture.dimensions),
            1,
        )
        .unwrap();
        let evaluation = neighbors[0].evaluation.as_ref().unwrap();
        assert_eq!(evaluation.binary_label, Some(JudgeBinaryLabelV1::Pass));
        assert!(!evaluation.promotion_eligible);
    }

    #[test]
    fn caller_owned_transaction_retains_projection_snapshot() {
        assert_eq!(register_sqlite_vec(), SqliteVecStatus::Available);
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("vector-search.db");
        let fixture = SearchFixture::from_connection(Connection::open(&path).unwrap(), 2);
        fixture.seed_operational(1, 1, &[1.0, 0.0]);
        fixture.refresh_manifest_count();

        let mut reader = Connection::open(&path).unwrap();
        reader.execute_batch("PRAGMA query_only = ON").unwrap();
        let transaction = reader.transaction().unwrap();
        assert!(matches!(
            resolve_active_generation(&transaction, &fixture.space_id).unwrap(),
            ActiveGenerationResolution::Active(_)
        ));
        fixture
            .connection
            .execute(
                "DELETE FROM shadow_results WHERE shadow_attempt_id = ?1",
                [fixture_uuid(100_001).to_string()],
            )
            .unwrap();

        let snapshot_neighbors = search_projected_neighbors_in_transaction(
            &transaction,
            &fixture.space_id,
            PartitionId::new(1).unwrap(),
            &query(&[1.0, 0.0], fixture.dimensions),
            1,
        )
        .unwrap();
        assert_eq!(snapshot_neighbors.len(), 1);
        transaction.commit().unwrap();

        assert_eq!(
            search_projected_neighbors(
                &reader,
                &fixture.space_id,
                PartitionId::new(1).unwrap(),
                &query(&[1.0, 0.0], fixture.dimensions),
                1,
            ),
            Err(VectorStoreError::Unavailable)
        );
    }

    #[test]
    fn generated_search_sql_keeps_the_exact_partition_inside_both_queries() {
        let authority = authority();
        assert_eq!(
            knn_sql(&authority),
            format!(
                "SELECT record_id, distance FROM \"{}\" \
                 WHERE embedding MATCH ?1 AND k = ?2 AND partition_id = ?3 \
                 ORDER BY distance",
                authority.root().as_str()
            )
        );
        assert_eq!(
            scalar_sql(&authority),
            format!(
                "SELECT record_id, vec_distance_cosine(embedding, ?1) AS distance \
                 FROM \"{}\" WHERE partition_id = ?3 \
                 ORDER BY distance, record_id LIMIT ?2",
                authority.root().as_str()
            )
        );
    }

    #[test]
    fn cutoff_tie_detection_is_exact_and_top_k_is_bounded() {
        assert_eq!(validate_top_k(0), Err(VectorStoreError::InvalidTopK));
        assert!(validate_top_k(VECTOR_TOP_K_MAX).is_ok());
        assert_eq!(
            validate_top_k(VECTOR_TOP_K_MAX + 1),
            Err(VectorStoreError::InvalidTopK)
        );

        let tied = [
            vector_match(1, 0.1),
            vector_match(2, 0.2),
            vector_match(3, 0.2),
        ];
        assert!(needs_scalar_fallback(&tied, 2, 3));
        let separated = [
            vector_match(1, 0.1),
            vector_match(2, 0.2),
            vector_match(3, 0.3),
        ];
        assert!(!needs_scalar_fallback(&separated, 2, 3));
        assert!(!needs_scalar_fallback(&tied[..2], 2, 2));
    }

    #[test]
    fn query_blob_uses_native_f32_component_order() {
        let dimensions = crate::vector::VectorDimensions::new(3).unwrap();
        let vector = NormalizedVector::from_provider_f64(&[3.0, -4.0, 12.0], dimensions).unwrap();
        let expected = vector
            .values()
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        assert_eq!(native_vector_blob(&vector), expected);
    }

    #[test]
    fn partial_binary_evaluation_does_not_become_a_link_quality_label() {
        assert!(validate_link_quality(None, Some(JudgeBinaryLabelV1::Pass), false).is_ok());
        assert_eq!(
            validate_link_quality(
                Some(JudgeBinaryLabelV1::Pass),
                Some(JudgeBinaryLabelV1::Pass),
                false,
            ),
            Err(VectorStoreError::Corrupt)
        );
        assert!(
            validate_link_quality(
                Some(JudgeBinaryLabelV1::Fail),
                Some(JudgeBinaryLabelV1::Fail),
                true,
            )
            .is_ok()
        );
    }
}
