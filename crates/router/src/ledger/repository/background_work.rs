// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded, advisory discovery of durable vector work.

use rusqlite::{Connection, params};
use uuid::Uuid;

use super::embedding::{EmbeddingJobSnapshot, load_verified_embedding_job};
use super::materialization::{
    MATERIALIZATION_ATTEMPT_MAX, MaterializationSnapshot, load_verified_materialization_job,
};
use super::shadow::load_verified_backfill_vector_source;
use super::vector_catalog::{CanonicalQuerySnapshot, load_canonical_query};
use super::vector_index::{ActiveGenerationResolution, resolve_active_generation};
use super::vector_registry::{
    FrozenMappingKey, resolve_embedder_profile, resolve_frozen_mapping, resolve_vector_space,
};
use super::{LedgerError, LedgerErrorClass, map_sqlite_error};
use crate::config::EMBEDDER_REQUEST_BYTES_MAX;
use crate::vector::VectorSpaceId;

/// Maximum rows returned by one background discovery query.
pub(crate) const BACKGROUND_WORK_PAGE_MAX: usize = 256;

/// Stable keyset position for same-space embedding discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingWorkCursor {
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) embedding_job_id: String,
}

/// One verified provider-work candidate, including the exact canonical bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingWorkCandidate {
    pub(crate) job: EmbeddingJobSnapshot,
    pub(crate) canonical_query: CanonicalQuerySnapshot,
    pub(crate) embedder_profile_version_id: String,
    pub(crate) batch_size: usize,
}

impl EmbeddingWorkCandidate {
    pub(crate) fn cursor(&self) -> EmbeddingWorkCursor {
        EmbeddingWorkCursor {
            vector_space_id: VectorSpaceId::new(self.job.vector_space_id.clone())
                .expect("verified embedding work has a valid vector space"),
            embedding_job_id: self.job.embedding_job_id.clone(),
        }
    }
}

/// Stable keyset position for materialization discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaterializationWorkCursor {
    pub(crate) materialization_job_id: String,
}

/// One cache-ready materialization that may be claimed by the writer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct MaterializationWorkCandidate {
    pub(crate) job: MaterializationSnapshot,
}

impl MaterializationWorkCandidate {
    pub(crate) fn cursor(&self) -> MaterializationWorkCursor {
        MaterializationWorkCursor {
            materialization_job_id: self.job.materialization_job_id.clone(),
        }
    }
}

/// Stable keyset position for terminal-failure propagation discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FailurePropagationCursor {
    pub(crate) embedding_job_id: String,
}

/// One verified terminal job and its exact next bounded dependent window.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FailurePropagationCandidate {
    pub(crate) job: EmbeddingJobSnapshot,
    pub(crate) evidence_vector_link_ids: Vec<Uuid>,
}

/// Stable keyset position for retained terminal backfill discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackfillWorkCursor {
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) shadow_attempt_id: Uuid,
}

/// One verified retained terminal absent from its current pool vector space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BackfillWorkCandidate {
    pub(crate) mapping: FrozenMappingKey,
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) shadow_attempt_id: Uuid,
    pub(crate) terminal_at_unix_ms: i64,
}

impl BackfillWorkCandidate {
    pub(crate) fn cursor(&self) -> BackfillWorkCursor {
        BackfillWorkCursor {
            vector_space_id: self.vector_space_id.clone(),
            shadow_attempt_id: self.shadow_attempt_id,
        }
    }
}

impl FailurePropagationCandidate {
    pub(crate) fn cursor(&self) -> FailurePropagationCursor {
        FailurePropagationCursor {
            embedding_job_id: self.job.embedding_job_id.clone(),
        }
    }
}

/// Select claimable embedding hints in deterministic same-space order.
pub(crate) fn select_embedding_work(
    connection: &Connection,
    project_uuid: Uuid,
    current_config_generation_id: &str,
    observed_at_unix_ms: i64,
    after: Option<&EmbeddingWorkCursor>,
    limit: usize,
) -> Result<Vec<EmbeddingWorkCandidate>, LedgerError> {
    validate_request(project_uuid, observed_at_unix_ms, limit)?;
    validate_sha256(current_config_generation_id)?;
    if let Some(cursor) = after {
        validate_sha256(&cursor.embedding_job_id)?;
    }
    let limit = limit_i64(limit)?;
    let after_space = after.map(|cursor| cursor.vector_space_id.as_str());
    let after_job = after.map(|cursor| cursor.embedding_job_id.as_str());
    let mut statement = connection
        .prepare(
            "SELECT job.vector_space_id, job.embedding_job_id,
                    space.embedder_profile_version_id, profile.batch_size,
                    query.canonical_size_bytes
             FROM embedding_jobs AS job
             JOIN vector_spaces AS space ON space.vector_space_id = job.vector_space_id
             JOIN embedder_profiles AS profile
               ON profile.embedder_profile_version_id = space.embedder_profile_version_id
             LEFT JOIN canonical_routing_queries AS query
               ON query.canonical_query_hash = job.canonical_query_hash
             JOIN embedding_job_state_events AS state
               ON state.embedding_job_id = job.embedding_job_id
              AND state.event_seq = (
                  SELECT max(latest.event_seq)
                  FROM embedding_job_state_events AS latest
                  WHERE latest.embedding_job_id = job.embedding_job_id
              )
             LEFT JOIN process_instances AS owner
               ON owner.process_instance_id = job.lease_owner_process_instance_id
             WHERE space.project_uuid = ?1
               AND EXISTS (
                    SELECT 1
                    FROM pool_vector_space_mappings AS mapping
                    WHERE mapping.project_uuid = ?1
                      AND mapping.config_generation_id = ?2
                      AND mapping.vector_space_id = job.vector_space_id
               )
               AND job.next_eligible_at_unix_ms <= ?3
               AND job.attempt_count < 5
               AND state.state NOT IN ('completed', 'terminal_failure', 'quarantined')
               AND NOT EXISTS (
                    SELECT 1
                    FROM embedding_jobs AS degraded
                    WHERE degraded.vector_space_id = job.vector_space_id
                      AND degraded.terminal_error_class IS NOT NULL
               )
               AND (job.vector_space_id > COALESCE(?4, '')
                    OR (job.vector_space_id = COALESCE(?4, '')
                        AND job.embedding_job_id > COALESCE(?5, '')))
               AND (
                    job.lease_owner_process_instance_id IS NULL
                    OR job.lease_expires_at_unix_ms <= ?3
                    OR owner.heartbeat_expires_at_unix_ms <= ?3
                    OR EXISTS (
                        SELECT 1 FROM process_instance_state_events AS terminal
                        WHERE terminal.subject_process_instance_id =
                              job.lease_owner_process_instance_id
                          AND terminal.state IN ('stopped', 'reconciled')
                    )
             )
             ORDER BY job.vector_space_id, job.embedding_job_id
             LIMIT ?6",
        )
        .map_err(database_error)?;
    let raw = statement
        .query_map(
            params![
                project_uuid.to_string(),
                current_config_generation_id,
                observed_at_unix_ms,
                after_space,
                after_job,
                limit,
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, Option<i64>>(4)?,
                ))
            },
        )
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    drop(statement);

    let selected_space = raw.first().map(|row| row.0.clone());
    let prefix_len = embedding_work_prefix_len(
        raw.iter()
            .take_while(|row| Some(row.0.as_str()) == selected_space.as_deref())
            .map(|row| row.4),
        EMBEDDER_REQUEST_BYTES_MAX,
    )?;
    let mut selected = Vec::new();
    for (
        selected_vector_space_id,
        embedding_job_id,
        profile_version_id,
        batch_size,
        canonical_size_bytes,
    ) in raw.into_iter().take(prefix_len)
    {
        let canonical_size_bytes = canonical_size_bytes
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| (1..=EMBEDDER_REQUEST_BYTES_MAX).contains(value))
            .ok_or_else(corrupt)?;
        let job = load_verified_embedding_job(connection, project_uuid, &embedding_job_id)?
            .ok_or_else(corrupt)?;
        if job.vector_space_id != selected_vector_space_id {
            return Err(corrupt());
        }
        let vector_space_id =
            VectorSpaceId::new(job.vector_space_id.clone()).map_err(|_| corrupt())?;
        let space = resolve_vector_space(connection, project_uuid, &vector_space_id)?
            .ok_or_else(corrupt)?;
        let profile =
            resolve_embedder_profile(connection, &profile_version_id)?.ok_or_else(corrupt)?;
        let canonical_query =
            load_canonical_query(connection, &job.canonical_query_hash)?.ok_or_else(corrupt)?;
        let batch_size = usize::try_from(batch_size)
            .ok()
            .filter(|value| (1..=128).contains(value))
            .ok_or_else(corrupt)?;
        if space.space.embedder_profile_version_id != profile_version_id
            || profile.profile.embedder_profile_version_id != profile_version_id
            || profile.profile.batch_size != batch_size
            || canonical_query.artifact.canonical_query_hash != job.canonical_query_hash
            || canonical_query.artifact.canonical_bytes.len() != canonical_size_bytes
        {
            return Err(corrupt());
        }
        selected.push(EmbeddingWorkCandidate {
            job,
            canonical_query,
            embedder_profile_version_id: profile_version_id,
            batch_size,
        });
    }
    Ok(selected)
}

fn embedding_work_prefix_len(
    sizes: impl IntoIterator<Item = Option<i64>>,
    byte_limit: usize,
) -> Result<usize, LedgerError> {
    if byte_limit == 0 || byte_limit > EMBEDDER_REQUEST_BYTES_MAX {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let mut count = 0_usize;
    let mut total = 0_usize;
    for size in sizes {
        let size = size
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| (1..=EMBEDDER_REQUEST_BYTES_MAX).contains(value))
            .ok_or_else(corrupt)?;
        let next = total.checked_add(size).ok_or_else(corrupt)?;
        if next > byte_limit {
            break;
        }
        total = next;
        count += 1;
    }
    Ok(count)
}

/// Select cache-ready materialization hints in deterministic job order.
pub(crate) fn select_materialization_work(
    connection: &Connection,
    project_uuid: Uuid,
    observed_at_unix_ms: i64,
    after: Option<&MaterializationWorkCursor>,
    limit: usize,
) -> Result<Vec<MaterializationWorkCandidate>, LedgerError> {
    validate_request(project_uuid, observed_at_unix_ms, limit)?;
    if let Some(cursor) = after {
        validate_sha256(&cursor.materialization_job_id)?;
    }
    let mut statement = connection
        .prepare(
            "SELECT job.vector_materialization_job_id
             FROM vector_materialization_jobs AS job
             JOIN vector_spaces AS space ON space.vector_space_id = job.vector_space_id
             JOIN embeddings AS embedding
               ON embedding.vector_space_id = job.vector_space_id
              AND embedding.canonical_query_hash = job.canonical_query_hash
             JOIN vector_materialization_job_state_events AS state
               ON state.vector_materialization_job_id = job.vector_materialization_job_id
              AND state.event_seq = (
                  SELECT max(latest.event_seq)
                  FROM vector_materialization_job_state_events AS latest
                  WHERE latest.vector_materialization_job_id =
                        job.vector_materialization_job_id
              )
             LEFT JOIN process_instances AS owner
               ON owner.process_instance_id = job.lease_owner_process_instance_id
             WHERE space.project_uuid = ?1
               AND job.vector_materialization_job_id > COALESCE(?2, '')
               AND job.next_eligible_at_unix_ms <= ?3
               AND job.attempt_count <= ?4
               AND EXISTS (
                    SELECT 1 FROM vector_index_manifest AS manifest
                    WHERE manifest.vector_space_id = job.vector_space_id
                      AND manifest.state = 'active'
               )
               AND state.state NOT IN (
                    'ready', 'failed_embedding', 'failed_index', 'canceled_retention'
               )
               AND (
                    job.lease_owner_process_instance_id IS NULL
                    OR job.lease_expires_at_unix_ms <= ?3
                    OR owner.heartbeat_expires_at_unix_ms <= ?3
                    OR EXISTS (
                        SELECT 1 FROM process_instance_state_events AS terminal
                        WHERE terminal.subject_process_instance_id =
                              job.lease_owner_process_instance_id
                          AND terminal.state IN ('stopped', 'reconciled')
                    )
               )
             ORDER BY job.vector_materialization_job_id
             LIMIT ?5",
        )
        .map_err(database_error)?;
    let ids = statement
        .query_map(
            params![
                project_uuid.to_string(),
                after.map(|cursor| cursor.materialization_job_id.as_str()),
                observed_at_unix_ms,
                MATERIALIZATION_ATTEMPT_MAX,
                limit_i64(limit)?,
            ],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    drop(statement);
    let mut selected = Vec::with_capacity(ids.len());
    for materialization_job_id in ids {
        let job =
            load_verified_materialization_job(connection, project_uuid, &materialization_job_id)?
                .ok_or_else(corrupt)?;
        match resolve_active_generation(connection, &job.vector_space_id)? {
            ActiveGenerationResolution::Active(_) => {
                selected.push(MaterializationWorkCandidate { job });
            }
            ActiveGenerationResolution::Missing
            | ActiveGenerationResolution::Unavailable
            | ActiveGenerationResolution::Corrupt => {}
        }
    }
    Ok(selected)
}

/// Select terminal embedding jobs whose bounded dependent fanout is incomplete.
pub(crate) fn select_failure_propagation_work(
    connection: &Connection,
    project_uuid: Uuid,
    after: Option<&FailurePropagationCursor>,
    limit: usize,
) -> Result<Vec<FailurePropagationCandidate>, LedgerError> {
    validate_request(project_uuid, 0, limit)?;
    if let Some(cursor) = after {
        validate_sha256(&cursor.embedding_job_id)?;
    }
    let mut statement = connection
        .prepare(
            "SELECT job.embedding_job_id
             FROM embedding_jobs AS job
             JOIN vector_spaces AS space ON space.vector_space_id = job.vector_space_id
             WHERE space.project_uuid = ?1
               AND job.embedding_job_id > COALESCE(?2, '')
               AND job.terminal_error_class IS NOT NULL
               AND job.failure_propagation_complete = 0
             ORDER BY job.embedding_job_id
             LIMIT ?3",
        )
        .map_err(database_error)?;
    let job_ids = statement
        .query_map(
            params![
                project_uuid.to_string(),
                after.map(|cursor| cursor.embedding_job_id.as_str()),
                limit_i64(limit)?,
            ],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    drop(statement);

    job_ids
        .into_iter()
        .map(|embedding_job_id| {
            let job = load_verified_embedding_job(connection, project_uuid, &embedding_job_id)?
                .ok_or_else(corrupt)?;
            if job.terminal_error_class.is_none() || job.failure_propagation_complete {
                return Err(corrupt());
            }
            let mut dependents = connection
                .prepare(
                    "SELECT evidence_vector_link_id
                     FROM vector_materialization_jobs
                     WHERE embedding_job_id = ?1
                       AND evidence_vector_link_id > COALESCE(?2, '')
                     ORDER BY evidence_vector_link_id
                     LIMIT 256",
                )
                .map_err(database_error)?;
            let ids = dependents
                .query_map(
                    params![embedding_job_id, job.failure_propagation_cursor.as_deref()],
                    |row| row.get::<_, String>(0),
                )
                .map_err(database_error)?
                .map(|value| {
                    value
                        .map_err(database_error)
                        .and_then(|value| parse_uuid_v7(&value))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Ok(FailurePropagationCandidate {
                job,
                evidence_vector_link_ids: ids,
            })
        })
        .collect()
}

/// Select retained terminals that lack an outcome in their pool's current vector space.
pub(crate) fn select_backfill_work(
    connection: &Connection,
    project_uuid: Uuid,
    current_config_generation_id: &str,
    after: Option<&BackfillWorkCursor>,
    limit: usize,
) -> Result<Vec<BackfillWorkCandidate>, LedgerError> {
    validate_request(project_uuid, 0, limit)?;
    validate_sha256(current_config_generation_id)?;
    let after_space = after.map(|cursor| cursor.vector_space_id.as_str());
    let after_attempt = after.map(|cursor| cursor.shadow_attempt_id.to_string());
    let mut statement = connection
        .prepare(
            "SELECT mapping.vector_space_id, attempt.shadow_attempt_id,
                    mapping.pool_id, mapping.policy_version_id
             FROM pool_vector_space_mappings AS mapping
             JOIN shadow_attempts AS attempt
               ON attempt.project_uuid = mapping.project_uuid
              AND attempt.pool_id = mapping.pool_id
             JOIN shadow_results AS result
               ON result.shadow_attempt_id = attempt.shadow_attempt_id
             LEFT JOIN vectorization_outcomes AS outcome
               ON outcome.shadow_attempt_id = attempt.shadow_attempt_id
              AND outcome.vector_space_id = mapping.vector_space_id
             WHERE mapping.project_uuid = ?1
               AND mapping.config_generation_id = ?2
               AND outcome.vectorization_outcome_id IS NULL
               AND NOT EXISTS (
                   SELECT 1 FROM decision_retiring_anchors AS marker
                   WHERE marker.anchor_id = attempt.anchor_id
               )
               AND (
                    mapping.vector_space_id > COALESCE(?3, '')
                    OR (mapping.vector_space_id = COALESCE(?3, '')
                        AND attempt.shadow_attempt_id > COALESCE(?4, ''))
               )
             ORDER BY mapping.vector_space_id, attempt.shadow_attempt_id
             LIMIT ?5",
        )
        .map_err(database_error)?;
    let raw = statement
        .query_map(
            params![
                project_uuid.to_string(),
                current_config_generation_id,
                after_space,
                after_attempt,
                limit_i64(limit)?,
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            },
        )
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    drop(statement);

    raw.into_iter()
        .map(
            |(vector_space_id, shadow_attempt_id, pool_id, policy_version_id)| {
                let vector_space_id = VectorSpaceId::new(vector_space_id).map_err(|_| corrupt())?;
                let shadow_attempt_id = parse_uuid_v7(&shadow_attempt_id)?;
                let mapping = FrozenMappingKey::new(
                    project_uuid,
                    current_config_generation_id,
                    pool_id,
                    policy_version_id,
                )
                .map_err(|_| corrupt())?;
                let verified_mapping =
                    resolve_frozen_mapping(connection, &mapping)?.ok_or_else(corrupt)?;
                let source = load_verified_backfill_vector_source(
                    connection,
                    project_uuid,
                    shadow_attempt_id,
                )?
                .ok_or_else(corrupt)?;
                if verified_mapping.mapping.vector_space_id != vector_space_id
                    || source.reservation.pool_id != mapping.pool_id
                {
                    return Err(corrupt());
                }
                Ok(BackfillWorkCandidate {
                    mapping,
                    vector_space_id,
                    shadow_attempt_id,
                    terminal_at_unix_ms: source.terminal_at_unix_ms,
                })
            },
        )
        .collect()
}

fn validate_request(
    project_uuid: Uuid,
    observed_at_unix_ms: i64,
    limit: usize,
) -> Result<(), LedgerError> {
    if project_uuid.get_version_num() != 7
        || project_uuid.get_variant() != uuid::Variant::RFC4122
        || observed_at_unix_ms < 0
        || !(1..=BACKGROUND_WORK_PAGE_MAX).contains(&limit)
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn limit_i64(limit: usize) -> Result<i64, LedgerError> {
    i64::try_from(limit).map_err(|_| LedgerErrorClass::IdentityInvariant.into())
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

fn parse_uuid_v7(value: &str) -> Result<Uuid, LedgerError> {
    let parsed =
        Uuid::parse_str(value).map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if parsed.get_version_num() != 7
        || parsed.get_variant() != uuid::Variant::RFC4122
        || parsed.to_string() != value
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(parsed)
}

fn corrupt() -> LedgerError {
    LedgerErrorClass::CorruptDatabase.into()
}

fn database_error(error: rusqlite::Error) -> LedgerError {
    map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}

#[cfg(test)]
mod tests {
    use crate::config::EMBEDDER_REQUEST_BYTES_MAX;

    use super::embedding_work_prefix_len;

    #[test]
    fn embedding_work_prefix_is_byte_bounded_and_resumable() {
        let limit = i64::try_from(EMBEDDER_REQUEST_BYTES_MAX).unwrap();
        let sizes = [Some(limit - 1), Some(1), Some(1)];
        assert_eq!(
            embedding_work_prefix_len(sizes, EMBEDDER_REQUEST_BYTES_MAX).unwrap(),
            2
        );
        assert_eq!(
            embedding_work_prefix_len([sizes[2]], EMBEDDER_REQUEST_BYTES_MAX).unwrap(),
            1
        );
        assert_eq!(
            embedding_work_prefix_len([Some(limit)], EMBEDDER_REQUEST_BYTES_MAX).unwrap(),
            1
        );
        assert!(embedding_work_prefix_len([Some(limit + 1)], EMBEDDER_REQUEST_BYTES_MAX).is_err());
        assert!(embedding_work_prefix_len([Some(0)], EMBEDDER_REQUEST_BYTES_MAX).is_err());
        assert!(embedding_work_prefix_len([None], EMBEDDER_REQUEST_BYTES_MAX).is_err());
    }
}
