// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transaction-local canonical query, partition, cache, and source-change APIs.

use nemo_relay_types::api::llm::LlmApiFamily;
use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde_json::json;
use uuid::Uuid;

use super::vector_index::{
    VectorSourceChangeOperation, vector_source_change_payload_hash,
    vector_source_sequence_payload_hash,
};
use super::vector_registry::{FrozenMappingKey, resolve_frozen_mapping, resolve_vector_space};
use super::{is_sha256, map_sqlite_error};
use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::canonical_query::{CanonicalRoutingQueryArtifactV1, CanonicalRoutingQueryV1};
use crate::fingerprint::sha256_hex;
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::routing_partition::{RoutingPartitionArtifactV1, RoutingPartitionV1};
use crate::vector::{
    AuthoritativeVector, PartitionId, VectorChecksum, VectorDimensions, VectorRecordId,
    VectorSpaceId,
};

/// Frozen authority required to use one pool-to-space mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VectorMappingAuthority {
    pub(crate) project_uuid: Uuid,
    pub(crate) config_generation_id: String,
    pub(crate) pool_id: String,
    pub(crate) mapping_policy_version_id: String,
}

/// Verified canonical query row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanonicalQuerySnapshot {
    pub(crate) artifact: CanonicalRoutingQueryArtifactV1,
    pub(crate) created_at_unix_ms: i64,
}

/// Idempotent canonical-query result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CanonicalQueryEnsureAck {
    Applied(CanonicalQuerySnapshot),
    AlreadyExists(CanonicalQuerySnapshot),
    Conflict,
}

/// Immutable partition allocation request.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RoutingPartitionEnsure {
    pub(crate) mapping: VectorMappingAuthority,
    pub(crate) artifact: RoutingPartitionArtifactV1,
    pub(crate) created_at_unix_ms: i64,
}

/// Verified partition row including its positive SQLite key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RoutingPartitionSnapshot {
    pub(crate) partition_id: PartitionId,
    pub(crate) mapping: VectorMappingAuthority,
    pub(crate) artifact: RoutingPartitionArtifactV1,
    pub(crate) created_at_unix_ms: i64,
}

/// Exact live lookup result before any vector search begins.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LiveRoutingPartitionResolution {
    AuthorityNotFound,
    StaleLearningGeneration,
    NoPartition,
    Found(Box<RoutingPartitionSnapshot>),
}

/// Exhaustive partition allocation result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RoutingPartitionEnsureAck {
    Applied(RoutingPartitionSnapshot),
    AlreadyExists(RoutingPartitionSnapshot),
    MappingNotFound,
    Conflict,
}

/// Stable persisted origin of one authoritative embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum EmbeddingCacheSource {
    Provider,
    Cache,
}

impl EmbeddingCacheSource {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Provider => "provider",
            Self::Cache => "cache",
        }
    }

    fn parse(value: &str) -> Result<Self, LedgerError> {
        match value {
            "provider" => Ok(Self::Provider),
            "cache" => Ok(Self::Cache),
            _ => Err(corrupt()),
        }
    }
}

/// Proposed first-writer cache row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingCacheWrite {
    pub(crate) embedding_id: Uuid,
    pub(crate) project_uuid: Uuid,
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) canonical_query_hash: String,
    pub(crate) content_hash: String,
    pub(crate) vector: AuthoritativeVector,
    pub(crate) source: EmbeddingCacheSource,
    pub(crate) created_at_unix_ms: i64,
}

/// Fully verified authoritative embedding row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct EmbeddingCacheSnapshot {
    pub(crate) embedding_id: Uuid,
    pub(crate) project_uuid: Uuid,
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) canonical_query_hash: String,
    pub(crate) content_hash: String,
    pub(crate) vector: AuthoritativeVector,
    pub(crate) source: EmbeddingCacheSource,
    pub(crate) created_at_unix_ms: i64,
    pub(crate) canonical_payload_hash: String,
}

/// Exhaustive cache write result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum EmbeddingCacheUpsertAck {
    Applied(EmbeddingCacheSnapshot),
    AlreadyExists(EmbeddingCacheSnapshot),
    AuthorityNotFound,
    Conflict,
}

/// One expected-sequence source mutation appended beside a ready-row change.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SourceChangeAppend {
    pub(crate) project_uuid: Uuid,
    pub(crate) vector_space_id: VectorSpaceId,
    pub(crate) expected_previous_source_seq: i64,
    pub(crate) operation: VectorSourceChangeOperation,
    pub(crate) record_id: VectorRecordId,
    pub(crate) partition_id: PartitionId,
    pub(crate) vector_checksum: VectorChecksum,
    pub(crate) created_at_unix_ms: i64,
}

/// Idempotent source-cursor append result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceChangeAppendAck {
    Applied { source_seq: i64 },
    AlreadyExists { source_seq: i64 },
    AuthorityNotFound,
    Conflict,
}

/// Create or verify one exact canonical query document.
pub(crate) fn ensure_canonical_query(
    transaction: &Transaction<'_>,
    artifact: &CanonicalRoutingQueryArtifactV1,
    created_at_unix_ms: i64,
) -> Result<CanonicalQueryEnsureAck, LedgerError> {
    validate_query_artifact(artifact)?;
    if created_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    if let Some(existing) = load_canonical_query(transaction, &artifact.canonical_query_hash)? {
        return Ok(if existing.artifact == *artifact {
            CanonicalQueryEnsureAck::AlreadyExists(existing)
        } else {
            CanonicalQueryEnsureAck::Conflict
        });
    }
    let canonical_json = std::str::from_utf8(&artifact.canonical_bytes)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let inserted = transaction
        .execute(
            "INSERT INTO canonical_routing_queries (
                canonical_query_hash, canonical_query_json, canonical_size_bytes,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?1)
             ON CONFLICT(canonical_query_hash) DO NOTHING",
            params![
                artifact.canonical_query_hash,
                canonical_json,
                i64::try_from(artifact.canonical_bytes.len())
                    .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
                created_at_unix_ms,
            ],
        )
        .map_err(database_error)?;
    let existing =
        load_canonical_query(transaction, &artifact.canonical_query_hash)?.ok_or_else(corrupt)?;
    Ok(if existing.artifact != *artifact {
        CanonicalQueryEnsureAck::Conflict
    } else if inserted == 1 {
        CanonicalQueryEnsureAck::Applied(existing)
    } else {
        CanonicalQueryEnsureAck::AlreadyExists(existing)
    })
}

/// Read and cryptographically verify one canonical query row.
pub(crate) fn load_canonical_query(
    connection: &Connection,
    canonical_query_hash: &str,
) -> Result<Option<CanonicalQuerySnapshot>, LedgerError> {
    if !is_sha256(canonical_query_hash) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let stored = connection
        .query_row(
            "SELECT canonical_query_hash, canonical_query_json, canonical_size_bytes,
                    created_at_unix_ms, canonical_payload_hash
             FROM canonical_routing_queries WHERE canonical_query_hash = ?1",
            params![canonical_query_hash],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, String>(4)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    stored
        .map(|(hash, canonical, size, created_at, payload_hash)| {
            if hash != canonical_query_hash
                || payload_hash != hash
                || size <= 0
                || usize::try_from(size).ok() != Some(canonical.len())
                || created_at < 0
                || sha256_hex(canonical.as_bytes()) != hash
            {
                return Err(corrupt());
            }
            let query: CanonicalRoutingQueryV1 =
                serde_json::from_str(&canonical).map_err(|_| corrupt())?;
            let value = serde_json::to_value(&query).map_err(|_| corrupt())?;
            if canonical_json(&value).ok().as_deref() != Some(canonical.as_str()) {
                return Err(corrupt());
            }
            Ok(CanonicalQuerySnapshot {
                artifact: CanonicalRoutingQueryArtifactV1 {
                    query,
                    canonical_bytes: canonical.into_bytes(),
                    canonical_query_hash: hash,
                },
                created_at_unix_ms: created_at,
            })
        })
        .transpose()
}

/// Allocate or verify one exact 14-field routing partition.
pub(crate) fn ensure_routing_partition(
    transaction: &Transaction<'_>,
    input: &RoutingPartitionEnsure,
) -> Result<RoutingPartitionEnsureAck, LedgerError> {
    validate_partition_artifact(&input.artifact)?;
    if input.created_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    if !mapping_authorizes_partition(transaction, &input.mapping, &input.artifact.partition)? {
        return Ok(RoutingPartitionEnsureAck::MappingNotFound);
    }
    if let Some(existing_id) = partition_id_for_hash(transaction, &input.artifact.partition_hash)? {
        let existing = load_routing_partition(transaction, &input.mapping, existing_id)?
            .ok_or_else(corrupt)?;
        return Ok(if existing.artifact == input.artifact {
            RoutingPartitionEnsureAck::AlreadyExists(existing)
        } else {
            RoutingPartitionEnsureAck::Conflict
        });
    }

    let partition = &input.artifact.partition;
    let inserted_id = transaction
        .query_row(
            "INSERT INTO routing_partitions (
                partition_hash, canonical_partition_json, project_uuid, pool_id,
                tenant_policy_hash, agent_policy_hash, policy_version_id,
                learning_generation_id, api_family, transport_identity,
                anchor_model, anchor_revision, candidate_id, candidate_model,
                candidate_model_revision, decoding_fingerprint, evaluator_version,
                vector_space_id, created_at_unix_ms, canonical_payload_hash
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13,
                ?14, ?15, ?16, ?17, ?18, ?19, ?1
             )
             ON CONFLICT(partition_hash) DO NOTHING
             RETURNING partition_id",
            params![
                input.artifact.partition_hash,
                input.artifact.canonical_json,
                input.mapping.project_uuid.to_string(),
                input.mapping.pool_id,
                partition.tenant_policy_hash,
                partition.agent_policy_hash,
                partition.policy_version_id,
                partition.learning_generation_id.to_string(),
                api_family_text(partition.api_family),
                partition.transport_identity,
                partition.anchor_model,
                partition.anchor_revision,
                partition.candidate_id,
                partition.candidate_model,
                partition.candidate_model_revision,
                partition.decoding_fingerprint,
                partition.evaluator_version,
                partition.vector_space_id,
                input.created_at_unix_ms,
            ],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?;
    let partition_id = match inserted_id {
        Some(value) => PartitionId::new(value).map_err(|_| corrupt())?,
        None => partition_id_for_hash(transaction, &input.artifact.partition_hash)?
            .ok_or_else(corrupt)?,
    };
    let snapshot =
        load_routing_partition(transaction, &input.mapping, partition_id)?.ok_or_else(corrupt)?;
    Ok(if snapshot.artifact != input.artifact {
        RoutingPartitionEnsureAck::Conflict
    } else if inserted_id.is_some() {
        RoutingPartitionEnsureAck::Applied(snapshot)
    } else {
        RoutingPartitionEnsureAck::AlreadyExists(snapshot)
    })
}

/// Read and verify one partition against the caller's frozen mapping authority.
pub(crate) fn load_routing_partition(
    connection: &Connection,
    mapping: &VectorMappingAuthority,
    partition_id: PartitionId,
) -> Result<Option<RoutingPartitionSnapshot>, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT partition_hash, canonical_partition_json, project_uuid, pool_id,
                    tenant_policy_hash, agent_policy_hash, policy_version_id,
                    learning_generation_id, api_family, transport_identity,
                    anchor_model, anchor_revision, candidate_id, candidate_model,
                    candidate_model_revision, decoding_fingerprint, evaluator_version,
                    vector_space_id, created_at_unix_ms, canonical_payload_hash
             FROM routing_partitions WHERE partition_id = ?1",
            params![partition_id.value()],
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
                    row.get::<_, String>(14)?,
                    row.get::<_, String>(15)?,
                    row.get::<_, String>(16)?,
                    row.get::<_, String>(17)?,
                    row.get::<_, i64>(18)?,
                    row.get::<_, String>(19)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    stored
        .map(
            |(
                hash,
                canonical,
                project_uuid,
                pool_id,
                tenant,
                agent,
                policy,
                learning,
                family,
                transport,
                anchor_model,
                anchor_revision,
                candidate_id,
                candidate_model,
                candidate_revision,
                decoding,
                evaluator,
                vector_space,
                created_at,
                payload_hash,
            )| {
                if project_uuid != mapping.project_uuid.to_string()
                    || pool_id != mapping.pool_id
                    || payload_hash != hash
                    || created_at < 0
                {
                    return Err(corrupt());
                }
                let partition: RoutingPartitionV1 =
                    serde_json::from_str(&canonical).map_err(|_| corrupt())?;
                let artifact = RoutingPartitionArtifactV1 {
                    partition,
                    canonical_json: canonical,
                    partition_hash: hash,
                };
                validate_partition_artifact(&artifact).map_err(|_| corrupt())?;
                let expected = &artifact.partition;
                if expected.tenant_policy_hash != tenant
                    || expected.agent_policy_hash != agent
                    || expected.policy_version_id != policy
                    || expected.learning_generation_id.to_string() != learning
                    || api_family_text(expected.api_family) != family
                    || expected.transport_identity != transport
                    || expected.anchor_model != anchor_model
                    || expected.anchor_revision != anchor_revision
                    || expected.candidate_id != candidate_id
                    || expected.candidate_model != candidate_model
                    || expected.candidate_model_revision != candidate_revision
                    || expected.decoding_fingerprint != decoding
                    || expected.evaluator_version != evaluator
                    || expected.vector_space_id != vector_space
                {
                    return Err(corrupt());
                }
                if !mapping_authorizes_partition(connection, mapping, expected)? {
                    return Err(corrupt());
                }
                Ok(RoutingPartitionSnapshot {
                    partition_id,
                    mapping: mapping.clone(),
                    artifact,
                    created_at_unix_ms: created_at,
                })
            },
        )
        .transpose()
}

/// Resolve a live expected partition by hash and full canonical document.
pub(crate) fn resolve_live_routing_partition(
    connection: &Connection,
    mapping: &FrozenMappingKey,
    expected: &RoutingPartitionArtifactV1,
) -> Result<LiveRoutingPartitionResolution, LedgerError> {
    validate_partition_artifact(expected)?;
    let Some(resolved) = resolve_frozen_mapping(connection, mapping)? else {
        return Ok(LiveRoutingPartitionResolution::AuthorityNotFound);
    };
    if resolved.mapping.project_uuid != mapping.project_uuid
        || resolved.mapping.config_generation_id != mapping.config_generation_id
        || resolved.mapping.pool_id != mapping.pool_id
        || resolved.mapping.policy_version_id != mapping.policy_version_id
        || resolved.mapping.vector_space_id.as_str() != expected.partition.vector_space_id
        || expected.partition.policy_version_id != mapping.policy_version_id
    {
        return Ok(LiveRoutingPartitionResolution::AuthorityNotFound);
    }
    verify_partition_learning_authority(
        connection,
        mapping.project_uuid,
        &mapping.pool_id,
        &expected.partition.learning_generation_id.to_string(),
    )?;
    let latest_generation = connection
        .query_row(
            "SELECT learning_generation_id
             FROM learning_generation_state_events
             WHERE project_uuid = ?1 AND pool_id = ?2
             ORDER BY event_seq DESC LIMIT 1",
            params![mapping.project_uuid.to_string(), mapping.pool_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    if latest_generation != expected.partition.learning_generation_id.to_string() {
        return Ok(LiveRoutingPartitionResolution::StaleLearningGeneration);
    }

    let Some(partition_id) = partition_id_for_hash(connection, &expected.partition_hash)? else {
        return Ok(LiveRoutingPartitionResolution::NoPartition);
    };
    let authority = VectorMappingAuthority {
        project_uuid: mapping.project_uuid,
        config_generation_id: mapping.config_generation_id.clone(),
        pool_id: mapping.pool_id.clone(),
        mapping_policy_version_id: mapping.policy_version_id.clone(),
    };
    let stored =
        load_routing_partition(connection, &authority, partition_id)?.ok_or_else(corrupt)?;
    if stored.artifact != *expected {
        return Err(corrupt());
    }
    Ok(LiveRoutingPartitionResolution::Found(Box::new(stored)))
}

/// Verify the self-contained canonical portion of a historical partition row.
pub(crate) fn verify_historical_routing_partition(
    connection: &Connection,
    partition_id: PartitionId,
    expected_project_uuid: Uuid,
    expected_vector_space_id: &VectorSpaceId,
) -> Result<bool, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT partition_hash, canonical_partition_json, project_uuid, pool_id,
                    tenant_policy_hash, agent_policy_hash, policy_version_id,
                    learning_generation_id, api_family, transport_identity,
                    anchor_model, anchor_revision, candidate_id, candidate_model,
                    candidate_model_revision, decoding_fingerprint, evaluator_version,
                    vector_space_id, created_at_unix_ms, canonical_payload_hash
             FROM routing_partitions WHERE partition_id = ?1",
            params![partition_id.value()],
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
                    row.get::<_, String>(14)?,
                    row.get::<_, String>(15)?,
                    row.get::<_, String>(16)?,
                    row.get::<_, String>(17)?,
                    row.get::<_, i64>(18)?,
                    row.get::<_, String>(19)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((
        hash,
        canonical,
        project_uuid,
        pool_id,
        tenant,
        agent,
        policy,
        learning,
        family,
        transport,
        anchor_model,
        anchor_revision,
        candidate_id,
        candidate_model,
        candidate_revision,
        decoding,
        evaluator,
        vector_space,
        created_at,
        payload_hash,
    )) = stored
    else {
        return Ok(false);
    };
    if payload_hash != hash
        || created_at < 0
        || project_uuid != expected_project_uuid.to_string()
        || vector_space != expected_vector_space_id.as_str()
    {
        return Err(corrupt());
    }
    let partition: RoutingPartitionV1 = serde_json::from_str(&canonical).map_err(|_| corrupt())?;
    let artifact = RoutingPartitionArtifactV1 {
        partition,
        canonical_json: canonical,
        partition_hash: hash,
    };
    validate_partition_artifact(&artifact).map_err(|_| corrupt())?;
    let expected = &artifact.partition;
    if expected.tenant_policy_hash != tenant
        || expected.agent_policy_hash != agent
        || expected.policy_version_id != policy
        || expected.learning_generation_id.to_string() != learning
        || api_family_text(expected.api_family) != family
        || expected.transport_identity != transport
        || expected.anchor_model != anchor_model
        || expected.anchor_revision != anchor_revision
        || expected.candidate_id != candidate_id
        || expected.candidate_model != candidate_model
        || expected.candidate_model_revision != candidate_revision
        || expected.decoding_fingerprint != decoding
        || expected.evaluator_version != evaluator
        || expected.vector_space_id != vector_space
    {
        return Err(corrupt());
    }
    verify_partition_policy_authority(connection, expected_project_uuid, &pool_id, &policy)?;
    verify_partition_learning_authority(connection, expected_project_uuid, &pool_id, &learning)?;
    Ok(true)
}

fn verify_partition_policy_authority(
    connection: &Connection,
    project_uuid: Uuid,
    pool_id: &str,
    policy_version_id: &str,
) -> Result<(), LedgerError> {
    let (canonical, payload_hash) = connection
        .query_row(
            "SELECT canonical_policy_json, canonical_payload_hash
             FROM policy_versions
             WHERE project_uuid = ?1 AND pool_id = ?2 AND policy_version_id = ?3",
            params![project_uuid.to_string(), pool_id, policy_version_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    let value: serde_json::Value = serde_json::from_str(&canonical).map_err(|_| corrupt())?;
    if canonical_json(&value).map_err(|_| corrupt())? != canonical
        || canonical_sha256(&value).map_err(|_| corrupt())? != policy_version_id
        || payload_hash != policy_version_id
        || value
            .pointer("/pool/id")
            .and_then(serde_json::Value::as_str)
            != Some(pool_id)
    {
        return Err(corrupt());
    }
    Ok(())
}

fn verify_partition_learning_authority(
    connection: &Connection,
    project_uuid: Uuid,
    pool_id: &str,
    learning_generation_id: &str,
) -> Result<(), LedgerError> {
    let generation_uuid = Uuid::parse_str(learning_generation_id).map_err(|_| corrupt())?;
    if generation_uuid.to_string() != learning_generation_id
        || generation_uuid.get_version_num() != 7
        || generation_uuid.get_variant() != uuid::Variant::RFC4122
    {
        return Err(corrupt());
    }
    let stored = connection
        .query_row(
            "SELECT generation.actor, generation.reason, generation.created_at_unix_ms,
                    generation.canonical_payload_hash, state.learning_state_event_id,
                    state.actor, state.reason, state.created_at_unix_ms,
                    state.canonical_payload_hash
             FROM learning_generations AS generation
             JOIN learning_generation_state_events AS state
               ON state.project_uuid = generation.project_uuid
              AND state.pool_id = generation.pool_id
              AND state.learning_generation_id = generation.learning_generation_id
             WHERE generation.project_uuid = ?1 AND generation.pool_id = ?2
               AND generation.learning_generation_id = ?3",
            params![project_uuid.to_string(), pool_id, learning_generation_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
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
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    let (
        actor,
        reason,
        created_at,
        payload_hash,
        event_id,
        state_actor,
        state_reason,
        state_at,
        state_hash,
    ) = stored;
    let event_uuid = Uuid::parse_str(&event_id).map_err(|_| corrupt())?;
    if event_uuid.to_string() != event_id
        || canonical_sha256(&json!({
            "learning_generation_id": generation_uuid,
            "project_uuid": project_uuid,
            "pool_id": pool_id,
            "actor": actor,
            "reason": reason,
            "created_at_unix_ms": created_at,
        }))
        .map_err(|_| corrupt())?
            != payload_hash
        || canonical_sha256(&json!({
            "learning_state_event_id": event_uuid,
            "project_uuid": project_uuid,
            "pool_id": pool_id,
            "learning_generation_id": generation_uuid,
            "state": "current",
            "actor": state_actor,
            "reason": state_reason,
            "created_at_unix_ms": state_at,
        }))
        .map_err(|_| corrupt())?
            != state_hash
    {
        return Err(corrupt());
    }
    Ok(())
}

/// Read and verify one authoritative cache row for a space/query pair.
pub(crate) fn load_embedding_cache(
    connection: &Connection,
    project_uuid: Uuid,
    vector_space_id: &VectorSpaceId,
    canonical_query_hash: &str,
) -> Result<Option<EmbeddingCacheSnapshot>, LedgerError> {
    validate_uuid_v7(project_uuid)?;
    if !is_sha256(canonical_query_hash) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let Some(space) = resolve_vector_space(connection, project_uuid, vector_space_id)? else {
        return Ok(None);
    };
    if load_canonical_query(connection, canonical_query_hash)?.is_none() {
        return Ok(None);
    }
    let expected_dimensions = space.space.dimensions;
    let stored = connection
        .query_row(
            "SELECT embedding_id, vector_space_id, canonical_query_hash, content_hash,
                    dimensions, vector_blob, vector_checksum, source,
                    created_at_unix_ms, canonical_payload_hash
             FROM embeddings
             WHERE vector_space_id = ?1 AND canonical_query_hash = ?2",
            params![vector_space_id.as_str(), canonical_query_hash],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, Vec<u8>>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, String>(9)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    stored
        .map(
            |(
                embedding_id,
                stored_space,
                stored_query,
                content_hash,
                dimensions,
                vector_blob,
                vector_checksum,
                source,
                created_at,
                payload_hash,
            )| {
                let embedding_id = parse_uuid_v7(&embedding_id)?;
                let dimensions = u32::try_from(dimensions)
                    .ok()
                    .and_then(|value| VectorDimensions::new(value).ok())
                    .ok_or_else(corrupt)?;
                let checksum = VectorChecksum::new(vector_checksum).map_err(|_| corrupt())?;
                let source = EmbeddingCacheSource::parse(&source)?;
                if stored_space != vector_space_id.as_str()
                    || stored_query != canonical_query_hash
                    || content_hash != canonical_query_hash
                    || dimensions != expected_dimensions
                    || !is_sha256(&content_hash)
                    || created_at < 0
                    || !is_sha256(&payload_hash)
                {
                    return Err(corrupt());
                }
                let vector = AuthoritativeVector::from_blob_verified(
                    vector_space_id,
                    dimensions,
                    vector_blob,
                    checksum,
                )
                .map_err(|_| corrupt())?;
                let expected_hash = embedding_payload_hash(
                    embedding_id,
                    vector_space_id,
                    canonical_query_hash,
                    &content_hash,
                    &vector,
                    source,
                    created_at,
                )?;
                if payload_hash != expected_hash {
                    return Err(corrupt());
                }
                Ok(EmbeddingCacheSnapshot {
                    embedding_id,
                    project_uuid,
                    vector_space_id: vector_space_id.clone(),
                    canonical_query_hash: canonical_query_hash.to_string(),
                    content_hash,
                    vector,
                    source,
                    created_at_unix_ms: created_at,
                    canonical_payload_hash: payload_hash,
                })
            },
        )
        .transpose()
}

/// Apply first-writer cache semantics without replacing a conflicting vector.
pub(crate) fn upsert_embedding_cache(
    transaction: &Transaction<'_>,
    input: &EmbeddingCacheWrite,
) -> Result<EmbeddingCacheUpsertAck, LedgerError> {
    validate_embedding_write(input)?;
    let Some(space) =
        resolve_vector_space(transaction, input.project_uuid, &input.vector_space_id)?
    else {
        return Ok(EmbeddingCacheUpsertAck::AuthorityNotFound);
    };
    if space.space.dimensions != input.vector.vector().dimensions()
        || load_canonical_query(transaction, &input.canonical_query_hash)?.is_none()
    {
        return Ok(EmbeddingCacheUpsertAck::AuthorityNotFound);
    }
    if let Some(existing) = load_embedding_cache(
        transaction,
        input.project_uuid,
        &input.vector_space_id,
        &input.canonical_query_hash,
    )? {
        return Ok(
            if existing.content_hash == input.content_hash
                && existing.vector.bitwise_eq(&input.vector)
            {
                EmbeddingCacheUpsertAck::AlreadyExists(existing)
            } else {
                EmbeddingCacheUpsertAck::Conflict
            },
        );
    }
    let id_collision: bool = transaction
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM embeddings WHERE embedding_id = ?1)",
            params![input.embedding_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    if id_collision {
        return Ok(EmbeddingCacheUpsertAck::Conflict);
    }
    let payload_hash = embedding_payload_hash(
        input.embedding_id,
        &input.vector_space_id,
        &input.canonical_query_hash,
        &input.content_hash,
        &input.vector,
        input.source,
        input.created_at_unix_ms,
    )?;
    transaction
        .execute(
            "INSERT INTO embeddings (
                embedding_id, vector_space_id, canonical_query_hash, content_hash,
                dimensions, vector_blob, vector_checksum, source,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                input.embedding_id.to_string(),
                input.vector_space_id.as_str(),
                input.canonical_query_hash,
                input.content_hash,
                i64::from(input.vector.vector().dimensions().value()),
                input.vector.blob().bytes(),
                input.vector.blob().checksum().as_str(),
                input.source.as_str(),
                input.created_at_unix_ms,
                payload_hash,
            ],
        )
        .map_err(database_error)?;
    let snapshot = load_embedding_cache(
        transaction,
        input.project_uuid,
        &input.vector_space_id,
        &input.canonical_query_hash,
    )?
    .ok_or_else(corrupt)?;
    Ok(EmbeddingCacheUpsertAck::Applied(snapshot))
}

/// Atomically append one contiguous source change and advance its per-space cursor.
pub(crate) fn append_source_change(
    transaction: &Transaction<'_>,
    input: &SourceChangeAppend,
) -> Result<SourceChangeAppendAck, LedgerError> {
    validate_source_change_input(input)?;
    let Some(space) =
        resolve_vector_space(transaction, input.project_uuid, &input.vector_space_id)?
    else {
        return Ok(SourceChangeAppendAck::AuthorityNotFound);
    };
    if !partition_belongs_to_space(transaction, input.partition_id, &input.vector_space_id)? {
        return Ok(SourceChangeAppendAck::AuthorityNotFound);
    }
    let current = load_source_sequence(transaction, &input.vector_space_id)?.ok_or_else(corrupt)?;
    if current.0 != space.source_seq || current.1 != space.source_updated_at_unix_ms {
        return Err(corrupt());
    }
    let retry_seq = input
        .expected_previous_source_seq
        .checked_add(1)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if let Some(exact_match) = source_change_matches(transaction, input, retry_seq)? {
        return Ok(if exact_match {
            SourceChangeAppendAck::AlreadyExists {
                source_seq: retry_seq,
            }
        } else {
            SourceChangeAppendAck::Conflict
        });
    }
    let (current_seq, current_updated_at, previous_hash) = current;
    if current_seq != input.expected_previous_source_seq {
        return Ok(SourceChangeAppendAck::Conflict);
    }
    if input.created_at_unix_ms < current_updated_at {
        return Ok(SourceChangeAppendAck::Conflict);
    }
    let source_seq = retry_seq;
    let change_hash = vector_source_change_payload_hash(
        &input.vector_space_id,
        source_seq,
        input.operation,
        input.record_id,
        input.partition_id,
        &input.vector_checksum,
        input.created_at_unix_ms,
    )?;
    transaction
        .execute(
            "INSERT INTO vector_source_change_events (
                vector_space_id, source_seq, operation, record_id, partition_id,
                vector_checksum, created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                input.vector_space_id.as_str(),
                source_seq,
                input.operation.as_str(),
                input.record_id.to_string(),
                input.partition_id.value(),
                input.vector_checksum.as_str(),
                input.created_at_unix_ms,
                change_hash,
            ],
        )
        .map_err(database_error)?;
    let sequence_hash = vector_source_sequence_payload_hash(
        &input.vector_space_id,
        source_seq,
        input.created_at_unix_ms,
    )?;
    let changed = transaction
        .execute(
            "UPDATE vector_space_source_sequences
             SET source_seq = ?1, updated_at_unix_ms = ?2,
                 canonical_payload_hash = ?3
             WHERE vector_space_id = ?4 AND source_seq = ?5
               AND canonical_payload_hash = ?6",
            params![
                source_seq,
                input.created_at_unix_ms,
                sequence_hash,
                input.vector_space_id.as_str(),
                current_seq,
                previous_hash,
            ],
        )
        .map_err(database_error)?;
    if changed != 1 {
        return Err(corrupt());
    }
    Ok(SourceChangeAppendAck::Applied { source_seq })
}

pub(crate) fn validate_query_artifact(
    artifact: &CanonicalRoutingQueryArtifactV1,
) -> Result<(), LedgerError> {
    if !is_sha256(&artifact.canonical_query_hash)
        || artifact.canonical_bytes.is_empty()
        || artifact.canonical_bytes.len() > 33_554_432
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let value = serde_json::to_value(&artifact.query)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let canonical = canonical_json(&value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    if artifact.canonical_bytes != canonical.as_bytes()
        || sha256_hex(&artifact.canonical_bytes) != artifact.canonical_query_hash
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn validate_partition_artifact(artifact: &RoutingPartitionArtifactV1) -> Result<(), LedgerError> {
    if !is_sha256(&artifact.partition_hash) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let value = serde_json::to_value(&artifact.partition)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let canonical = canonical_json(&value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let hash = canonical_sha256(&value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    if canonical != artifact.canonical_json || hash != artifact.partition_hash {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    VectorSpaceId::new(artifact.partition.vector_space_id.clone())
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    validate_uuid_v7(artifact.partition.learning_generation_id)?;
    Ok(())
}

fn mapping_authorizes_partition(
    connection: &Connection,
    mapping: &VectorMappingAuthority,
    partition: &RoutingPartitionV1,
) -> Result<bool, LedgerError> {
    validate_mapping(mapping)?;
    let key = FrozenMappingKey::new(
        mapping.project_uuid,
        mapping.config_generation_id.clone(),
        mapping.pool_id.clone(),
        mapping.mapping_policy_version_id.clone(),
    )
    .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let Some(resolved) = resolve_frozen_mapping(connection, &key)? else {
        return Ok(false);
    };
    let mapping_matches = resolved.mapping.project_uuid == mapping.project_uuid
        && resolved.mapping.config_generation_id == mapping.config_generation_id
        && resolved.mapping.pool_id == mapping.pool_id
        && resolved.mapping.policy_version_id == mapping.mapping_policy_version_id
        && resolved.mapping.vector_space_id.as_str() == partition.vector_space_id
        && resolved.space.space.vector_space_id == resolved.mapping.vector_space_id;
    Ok(mapping_matches
        && connection
            .query_row(
                "SELECT EXISTS(
                SELECT 1 FROM learning_generations
                WHERE project_uuid = ?1 AND pool_id = ?2
                  AND learning_generation_id = ?3
             )",
                params![
                    mapping.project_uuid.to_string(),
                    mapping.pool_id,
                    partition.learning_generation_id.to_string(),
                ],
                |row| row.get(0),
            )
            .map_err(database_error)?)
}

fn validate_mapping(mapping: &VectorMappingAuthority) -> Result<(), LedgerError> {
    validate_uuid_v7(mapping.project_uuid)?;
    if !is_sha256(&mapping.config_generation_id)
        || !is_sha256(&mapping.mapping_policy_version_id)
        || mapping.pool_id.is_empty()
        || mapping.pool_id.len() > 128
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn partition_id_for_hash(
    connection: &Connection,
    partition_hash: &str,
) -> Result<Option<PartitionId>, LedgerError> {
    let value = connection
        .query_row(
            "SELECT partition_id FROM routing_partitions WHERE partition_hash = ?1",
            params![partition_hash],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?;
    value
        .map(|value| PartitionId::new(value).map_err(|_| corrupt()))
        .transpose()
}

fn validate_embedding_write(input: &EmbeddingCacheWrite) -> Result<(), LedgerError> {
    validate_uuid_v7(input.embedding_id)?;
    validate_uuid_v7(input.project_uuid)?;
    if !is_sha256(&input.canonical_query_hash)
        || !is_sha256(&input.content_hash)
        || input.content_hash != input.canonical_query_hash
        || input.vector.vector_space_id() != &input.vector_space_id
        || input.created_at_unix_ms < 0
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn embedding_payload_hash(
    embedding_id: Uuid,
    vector_space_id: &VectorSpaceId,
    canonical_query_hash: &str,
    content_hash: &str,
    vector: &AuthoritativeVector,
    source: EmbeddingCacheSource,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    canonical_sha256(&json!({
        "embedding_id": embedding_id,
        "vector_space_id": vector_space_id.as_str(),
        "canonical_query_hash": canonical_query_hash,
        "content_hash": content_hash,
        "dimensions": vector.vector().dimensions().value(),
        "vector_blob_sha256": sha256_hex(vector.blob().bytes()),
        "vector_checksum": vector.blob().checksum().as_str(),
        "source": source.as_str(),
        "created_at_unix_ms": created_at_unix_ms,
    }))
    .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn validate_source_change_input(input: &SourceChangeAppend) -> Result<(), LedgerError> {
    validate_uuid_v7(input.project_uuid)?;
    if input.expected_previous_source_seq < 0 || input.created_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn partition_belongs_to_space(
    connection: &Connection,
    partition_id: PartitionId,
    vector_space_id: &VectorSpaceId,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM routing_partitions
                WHERE partition_id = ?1 AND vector_space_id = ?2
             )",
            params![partition_id.value(), vector_space_id.as_str()],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn load_source_sequence(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
) -> Result<Option<(i64, i64, String)>, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT source_seq, updated_at_unix_ms, canonical_payload_hash
             FROM vector_space_source_sequences WHERE vector_space_id = ?1",
            params![vector_space_id.as_str()],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    stored
        .map(|(source_seq, updated_at, payload_hash)| {
            if source_seq < 0
                || updated_at < 0
                || !is_sha256(&payload_hash)
                || vector_source_sequence_payload_hash(vector_space_id, source_seq, updated_at)?
                    != payload_hash
            {
                return Err(corrupt());
            }
            Ok((source_seq, updated_at, payload_hash))
        })
        .transpose()
}

fn source_change_matches(
    connection: &Connection,
    input: &SourceChangeAppend,
    source_seq: i64,
) -> Result<Option<bool>, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT operation, record_id, partition_id, vector_checksum,
                    created_at_unix_ms, canonical_payload_hash
             FROM vector_source_change_events
             WHERE vector_space_id = ?1 AND source_seq = ?2",
            params![input.vector_space_id.as_str(), source_seq],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, String>(5)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((operation, record_id, partition_id, checksum, created_at, payload_hash)) = stored
    else {
        return Ok(None);
    };
    let expected_hash = vector_source_change_payload_hash(
        &input.vector_space_id,
        source_seq,
        input.operation,
        input.record_id,
        input.partition_id,
        &input.vector_checksum,
        input.created_at_unix_ms,
    )?;
    Ok(Some(
        operation == input.operation.as_str()
            && record_id == input.record_id.to_string()
            && partition_id == input.partition_id.value()
            && checksum == input.vector_checksum.as_str()
            && created_at == input.created_at_unix_ms
            && payload_hash == expected_hash,
    ))
}

fn api_family_text(family: LlmApiFamily) -> &'static str {
    match family {
        LlmApiFamily::OpenAIChatCompletions => "openai_chat_completions",
        LlmApiFamily::OpenAIResponses => "openai_responses",
        LlmApiFamily::AnthropicMessages => "anthropic_messages",
    }
}

fn validate_uuid_v7(value: Uuid) -> Result<(), LedgerError> {
    if value.get_version_num() != 7 || value.get_variant() != uuid::Variant::RFC4122 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn parse_uuid_v7(value: &str) -> Result<Uuid, LedgerError> {
    let parsed = Uuid::parse_str(value).map_err(|_| corrupt())?;
    if parsed.to_string() != value
        || parsed.get_version_num() != 7
        || parsed.get_variant() != uuid::Variant::RFC4122
    {
        return Err(corrupt());
    }
    Ok(parsed)
}

fn database_error(error: rusqlite::Error) -> LedgerError {
    map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}

fn corrupt() -> LedgerError {
    LedgerErrorClass::CorruptDatabase.into()
}

#[cfg(test)]
mod tests {
    use tempfile::TempDir;

    use super::*;
    use crate::canonical_query::{CanonicalRoutingQueryV1, CanonicalTaskV1};
    use crate::embedding_identity::CANONICAL_ROUTING_QUERY_SCHEMA_V1;
    use crate::ledger::repository::LedgerRepository;
    use crate::ledger::repository::tests::{config, database_path};
    use crate::vector::NormalizedVector;

    struct Fixture {
        _temporary: TempDir,
        activated: crate::ledger::repository::ActivatedLedger,
        mapping: VectorMappingAuthority,
        vector_space_id: VectorSpaceId,
        learning_generation_id: Uuid,
        policy_version_id: String,
    }

    fn fixture() -> Fixture {
        let temporary = tempfile::tempdir().unwrap();
        let path = database_path(&temporary);
        let mut runtime_config = config(&path, "vector-catalog-fixture");
        runtime_config.embedders[0].dimensions = 3;
        runtime_config.pools[0].learning =
            Some(crate::config::LearningConfig::minimal("embedder-a"));
        let activated = LedgerRepository::activate_at(&runtime_config, 1_000).unwrap();
        let project_uuid = activated.identity.project_uuid;
        let config_generation_id = activated.identity.config_generation_id.clone();
        let pool = activated.identity.pool("pool-a").unwrap();
        let policy_version_id = pool.policy_version_id.clone();
        let learning_generation_id = pool.learning_generation_id;
        let vector_space_id = pool
            .vector_space
            .as_ref()
            .expect("learning-enabled fixture should expose registry authority")
            .vector_space_id
            .clone();
        Fixture {
            _temporary: temporary,
            activated,
            mapping: VectorMappingAuthority {
                project_uuid,
                config_generation_id,
                pool_id: "pool-a".to_string(),
                mapping_policy_version_id: policy_version_id.clone(),
            },
            vector_space_id,
            learning_generation_id,
            policy_version_id,
        }
    }

    fn query_artifact(task: &str) -> CanonicalRoutingQueryArtifactV1 {
        let query = CanonicalRoutingQueryV1 {
            schema: CANONICAL_ROUTING_QUERY_SCHEMA_V1.to_string(),
            instructions: Vec::new(),
            current_task: CanonicalTaskV1 {
                text: task.to_string(),
            },
            bounded_context: Vec::new(),
            tool_schema_fingerprint: "1".repeat(64),
            response_schema_fingerprint: None,
            required_capabilities: Vec::new(),
            position_features: None,
        };
        let value = serde_json::to_value(&query).unwrap();
        let canonical_bytes = canonical_json(&value).unwrap().into_bytes();
        let canonical_query_hash = sha256_hex(&canonical_bytes);
        CanonicalRoutingQueryArtifactV1 {
            query,
            canonical_bytes,
            canonical_query_hash,
        }
    }

    fn partition_artifact(fixture: &Fixture, candidate: &str) -> RoutingPartitionArtifactV1 {
        let partition = RoutingPartitionV1 {
            tenant_policy_hash: "1".repeat(64),
            agent_policy_hash: "2".repeat(64),
            policy_version_id: fixture.policy_version_id.clone(),
            learning_generation_id: fixture.learning_generation_id,
            api_family: LlmApiFamily::OpenAIChatCompletions,
            transport_identity: "transport-v1".to_string(),
            anchor_model: "anchor-model".to_string(),
            anchor_revision: "anchor-r1".to_string(),
            candidate_id: candidate.to_string(),
            candidate_model: "candidate-model".to_string(),
            candidate_model_revision: "candidate-r1".to_string(),
            decoding_fingerprint: "3".repeat(64),
            evaluator_version: "4".repeat(64),
            vector_space_id: fixture.vector_space_id.as_str().to_string(),
        };
        let value = serde_json::to_value(&partition).unwrap();
        RoutingPartitionArtifactV1 {
            partition,
            canonical_json: canonical_json(&value).unwrap(),
            partition_hash: canonical_sha256(&value).unwrap(),
        }
    }

    fn vector(vector_space_id: &VectorSpaceId) -> AuthoritativeVector {
        let normalized = NormalizedVector::from_provider_f64(
            &[1.0, 2.0, 3.0],
            VectorDimensions::new(3).unwrap(),
        )
        .unwrap();
        AuthoritativeVector::from_normalized(vector_space_id, normalized).unwrap()
    }

    #[test]
    fn canonical_queries_are_exact_idempotent_and_corruption_checked() {
        let mut fixture = fixture();
        let artifact = query_artifact("route this request");
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        assert!(matches!(
            ensure_canonical_query(&transaction, &artifact, 1_001).unwrap(),
            CanonicalQueryEnsureAck::Applied(_)
        ));
        assert!(matches!(
            ensure_canonical_query(&transaction, &artifact, 1_002).unwrap(),
            CanonicalQueryEnsureAck::AlreadyExists(_)
        ));
        transaction.commit().unwrap();

        fixture
            .activated
            .repository
            .connection_mut()
            .execute(
                "UPDATE canonical_routing_queries
                 SET canonical_query_json = '{}', canonical_size_bytes = 2
                 WHERE canonical_query_hash = ?1",
                params![artifact.canonical_query_hash],
            )
            .unwrap();
        assert_eq!(
            load_canonical_query(
                fixture.activated.repository.connection_mut(),
                &artifact.canonical_query_hash,
            )
            .unwrap_err()
            .class(),
            LedgerErrorClass::CorruptDatabase
        );
    }

    #[test]
    fn partition_allocation_uses_returning_and_full_collision_checks() {
        let mut fixture = fixture();
        let first = partition_artifact(&fixture, "candidate-a");
        let second = partition_artifact(&fixture, "candidate-b");
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        let first_id = match ensure_routing_partition(
            &transaction,
            &RoutingPartitionEnsure {
                mapping: fixture.mapping.clone(),
                artifact: first.clone(),
                created_at_unix_ms: 1_001,
            },
        )
        .unwrap()
        {
            RoutingPartitionEnsureAck::Applied(snapshot) => snapshot.partition_id,
            other => panic!("unexpected allocation: {other:?}"),
        };
        assert!(matches!(
            ensure_routing_partition(
                &transaction,
                &RoutingPartitionEnsure {
                    mapping: fixture.mapping.clone(),
                    artifact: first,
                    created_at_unix_ms: 1_002,
                },
            )
            .unwrap(),
            RoutingPartitionEnsureAck::AlreadyExists(_)
        ));
        let second_id = match ensure_routing_partition(
            &transaction,
            &RoutingPartitionEnsure {
                mapping: fixture.mapping.clone(),
                artifact: second,
                created_at_unix_ms: 1_003,
            },
        )
        .unwrap()
        {
            RoutingPartitionEnsureAck::Applied(snapshot) => snapshot.partition_id,
            other => panic!("unexpected second allocation: {other:?}"),
        };
        assert!(second_id > first_id);
        transaction.commit().unwrap();
        let connection = fixture.activated.repository.connection_mut();
        connection
            .execute(
                "UPDATE pool_vector_space_mappings
                 SET canonical_mapping_json = '{}'
                 WHERE project_uuid = ?1 AND config_generation_id = ?2
                   AND pool_id = ?3 AND policy_version_id = ?4",
                params![
                    fixture.mapping.project_uuid.to_string(),
                    fixture.mapping.config_generation_id,
                    fixture.mapping.pool_id,
                    fixture.policy_version_id,
                ],
            )
            .unwrap();
        assert_eq!(
            load_routing_partition(connection, &fixture.mapping, first_id)
                .unwrap_err()
                .class(),
            LedgerErrorClass::CorruptDatabase
        );
    }

    #[test]
    fn live_partition_resolution_uses_frozen_mapping_hash_document_and_current_generation() {
        let mut fixture = fixture();
        let artifact = partition_artifact(&fixture, "candidate-a");
        let missing = partition_artifact(&fixture, "candidate-missing");
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        let partition_id = match ensure_routing_partition(
            &transaction,
            &RoutingPartitionEnsure {
                mapping: fixture.mapping.clone(),
                artifact: artifact.clone(),
                created_at_unix_ms: 1_001,
            },
        )
        .unwrap()
        {
            RoutingPartitionEnsureAck::Applied(snapshot) => snapshot.partition_id,
            other => panic!("unexpected allocation: {other:?}"),
        };
        transaction.commit().unwrap();
        let key = FrozenMappingKey::new(
            fixture.mapping.project_uuid,
            fixture.mapping.config_generation_id.clone(),
            fixture.mapping.pool_id.clone(),
            fixture.mapping.mapping_policy_version_id.clone(),
        )
        .unwrap();
        assert!(matches!(
            resolve_live_routing_partition(
                fixture.activated.repository.connection_mut(),
                &key,
                &artifact,
            )
            .unwrap(),
            LiveRoutingPartitionResolution::Found(snapshot)
                if snapshot.partition_id == partition_id
        ));
        assert_eq!(
            resolve_live_routing_partition(
                fixture.activated.repository.connection_mut(),
                &key,
                &missing,
            )
            .unwrap(),
            LiveRoutingPartitionResolution::NoPartition
        );

        fixture
            .activated
            .repository
            .reset_pool("pool-a", "operator", "test generation fence")
            .unwrap();
        assert_eq!(
            resolve_live_routing_partition(
                fixture.activated.repository.connection_mut(),
                &key,
                &artifact,
            )
            .unwrap(),
            LiveRoutingPartitionResolution::StaleLearningGeneration
        );

        let mut wrong_mapping = key;
        wrong_mapping.policy_version_id = "f".repeat(64);
        assert_eq!(
            resolve_live_routing_partition(
                fixture.activated.repository.connection_mut(),
                &wrong_mapping,
                &artifact,
            )
            .unwrap(),
            LiveRoutingPartitionResolution::AuthorityNotFound
        );
    }

    #[test]
    fn cache_is_first_writer_idempotent_and_rejects_vector_conflicts() {
        let mut fixture = fixture();
        let query = query_artifact("embed once");
        assert_eq!(
            load_embedding_cache(
                fixture.activated.repository.connection_mut(),
                fixture.mapping.project_uuid,
                &fixture.vector_space_id,
                &"f".repeat(64),
            )
            .unwrap(),
            None
        );
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        ensure_canonical_query(&transaction, &query, 1_001).unwrap();
        let write = EmbeddingCacheWrite {
            embedding_id: Uuid::now_v7(),
            project_uuid: fixture.mapping.project_uuid,
            vector_space_id: fixture.vector_space_id.clone(),
            canonical_query_hash: query.canonical_query_hash.clone(),
            content_hash: query.canonical_query_hash.clone(),
            vector: vector(&fixture.vector_space_id),
            source: EmbeddingCacheSource::Provider,
            created_at_unix_ms: 1_002,
        };
        assert!(matches!(
            upsert_embedding_cache(&transaction, &write).unwrap(),
            EmbeddingCacheUpsertAck::Applied(_)
        ));
        let mut same = write.clone();
        same.embedding_id = Uuid::now_v7();
        assert!(matches!(
            upsert_embedding_cache(&transaction, &same).unwrap(),
            EmbeddingCacheUpsertAck::AlreadyExists(_)
        ));
        let mut conflict = same;
        conflict.vector = AuthoritativeVector::from_normalized(
            &fixture.vector_space_id,
            NormalizedVector::from_provider_f64(
                &[3.0, 2.0, 1.0],
                VectorDimensions::new(3).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
        assert_eq!(
            upsert_embedding_cache(&transaction, &conflict).unwrap(),
            EmbeddingCacheUpsertAck::Conflict
        );
        let mut invalid_content_hash = write.clone();
        invalid_content_hash.content_hash = "6".repeat(64);
        assert_eq!(
            upsert_embedding_cache(&transaction, &invalid_content_hash)
                .unwrap_err()
                .class(),
            LedgerErrorClass::IdentityInvariant
        );
        transaction.commit().unwrap();
        let connection = fixture.activated.repository.connection_mut();
        let mismatched_content_hash = "6".repeat(64);
        let mismatched_payload_hash = embedding_payload_hash(
            write.embedding_id,
            &write.vector_space_id,
            &write.canonical_query_hash,
            &mismatched_content_hash,
            &write.vector,
            write.source,
            write.created_at_unix_ms,
        )
        .unwrap();
        connection
            .execute(
                "UPDATE embeddings
                 SET content_hash = ?1, canonical_payload_hash = ?2
                 WHERE embedding_id = ?3",
                params![
                    mismatched_content_hash,
                    mismatched_payload_hash,
                    write.embedding_id.to_string(),
                ],
            )
            .unwrap();
        assert_eq!(
            load_embedding_cache(
                connection,
                fixture.mapping.project_uuid,
                &fixture.vector_space_id,
                &query.canonical_query_hash,
            )
            .unwrap_err()
            .class(),
            LedgerErrorClass::CorruptDatabase
        );
        let restored_payload_hash = embedding_payload_hash(
            write.embedding_id,
            &write.vector_space_id,
            &write.canonical_query_hash,
            &write.content_hash,
            &write.vector,
            write.source,
            write.created_at_unix_ms,
        )
        .unwrap();
        connection
            .execute(
                "UPDATE embeddings
                 SET content_hash = ?1, canonical_payload_hash = ?2
                 WHERE embedding_id = ?3",
                params![
                    write.content_hash,
                    restored_payload_hash,
                    write.embedding_id.to_string(),
                ],
            )
            .unwrap();
        connection
            .execute(
                "UPDATE canonical_routing_queries
                 SET canonical_query_json = '{}', canonical_size_bytes = 2
                 WHERE canonical_query_hash = ?1",
                params![query.canonical_query_hash],
            )
            .unwrap();
        assert_eq!(
            load_embedding_cache(
                connection,
                fixture.mapping.project_uuid,
                &fixture.vector_space_id,
                &query.canonical_query_hash,
            )
            .unwrap_err()
            .class(),
            LedgerErrorClass::CorruptDatabase
        );
    }

    #[test]
    fn source_append_is_contiguous_idempotent_and_rolls_back_as_one_unit() {
        let mut fixture = fixture();
        let partition = partition_artifact(&fixture, "candidate-a");
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        let partition_id = match ensure_routing_partition(
            &transaction,
            &RoutingPartitionEnsure {
                mapping: fixture.mapping.clone(),
                artifact: partition,
                created_at_unix_ms: 1_001,
            },
        )
        .unwrap()
        {
            RoutingPartitionEnsureAck::Applied(snapshot) => snapshot.partition_id,
            other => panic!("unexpected partition: {other:?}"),
        };
        transaction.commit().unwrap();
        let change = SourceChangeAppend {
            project_uuid: fixture.mapping.project_uuid,
            vector_space_id: fixture.vector_space_id.clone(),
            expected_previous_source_seq: 0,
            operation: VectorSourceChangeOperation::Insert,
            record_id: VectorRecordId::new(Uuid::now_v7()).unwrap(),
            partition_id,
            vector_checksum: vector(&fixture.vector_space_id).blob().checksum().clone(),
            created_at_unix_ms: 1_002,
        };

        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        assert_eq!(
            append_source_change(&transaction, &change).unwrap(),
            SourceChangeAppendAck::Applied { source_seq: 1 }
        );
        drop(transaction);
        let connection = fixture.activated.repository.connection_mut();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM vector_source_change_events",
                    [],
                    |row| { row.get::<_, i64>(0) }
                )
                .unwrap(),
            0
        );

        let transaction = connection.transaction().unwrap();
        assert_eq!(
            append_source_change(&transaction, &change).unwrap(),
            SourceChangeAppendAck::Applied { source_seq: 1 }
        );
        transaction.commit().unwrap();
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        let mut next = change.clone();
        next.expected_previous_source_seq = 1;
        next.record_id = VectorRecordId::new(Uuid::now_v7()).unwrap();
        next.created_at_unix_ms = 1_003;
        assert_eq!(
            append_source_change(&transaction, &next).unwrap(),
            SourceChangeAppendAck::Applied { source_seq: 2 }
        );
        assert_eq!(
            append_source_change(&transaction, &change).unwrap(),
            SourceChangeAppendAck::AlreadyExists { source_seq: 1 }
        );
        let mut conflicting_retry = change.clone();
        conflicting_retry.record_id = VectorRecordId::new(Uuid::now_v7()).unwrap();
        assert_eq!(
            append_source_change(&transaction, &conflicting_retry).unwrap(),
            SourceChangeAppendAck::Conflict
        );
        transaction.commit().unwrap();

        let connection = fixture.activated.repository.connection_mut();
        connection
            .execute(
                "DELETE FROM vector_space_source_sequences WHERE vector_space_id = ?1",
                params![fixture.vector_space_id.as_str()],
            )
            .unwrap();
        let transaction = connection.transaction().unwrap();
        assert_eq!(
            append_source_change(&transaction, &next)
                .unwrap_err()
                .class(),
            LedgerErrorClass::CorruptDatabase
        );
    }
}
