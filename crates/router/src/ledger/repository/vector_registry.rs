// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable nonsecret embedding-profile, vector-space, and pool-mapping authority.

use std::collections::{BTreeMap, BTreeSet};

use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde::Deserialize;
use serde_json::Value as Json;
use uuid::{Uuid, Variant};

use super::map_sqlite_error;
use super::vector_index::vector_source_sequence_payload_hash;
use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::config::{
    CanonicalizerConfig, EMBEDDER_BATCH_SIZE_MAX, EMBEDDER_MAX_IN_FLIGHT, EMBEDDER_TIMEOUT_MS_MAX,
    ID_MAX_BYTES, MODEL_ID_MAX_BYTES, REVISION_MAX_BYTES, RouterConfig,
};
use crate::embedding_identity::{
    CANONICAL_ROUTING_QUERY_RULES_V1, CANONICAL_ROUTING_QUERY_SCHEMA_V1,
    CANONICAL_TEXT_NORMALIZATION_V1, EmbedderEgressClass, PreparedEmbedderProfile,
    PreparedPoolVectorSpaceMapping, PreparedVectorSpace, canonicalizer_version,
    normalize_embedder_endpoint, propose_vector_space_registry,
};
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::vector::{VectorDimensions, VectorSpaceId};

const CANONICALIZER_IDENTITY_SCHEMA_V1: &str = "nemo.relay.router.canonicalizer-identity@1";
const CONFIG_GENERATION_SCHEMA_V1: &str = "nemo.relay.router.config-generation@1";
const EMBEDDER_PROFILE_SCHEMA_V1: &str = "nemo.relay.router.embedder-profile@1";
const POLICY_SCHEMA_V1: &str = "nemo.relay.router.policy@1";
const POOL_VECTOR_SPACE_MAPPING_SCHEMA_V1: &str = "nemo.relay.router.pool-vector-space-mapping@1";
const VECTOR_SPACE_SCHEMA_V1: &str = "nemo.relay.router.vector-space@1";
const OPENAI_EMBEDDINGS_PROTOCOL_V1: &str = "openai-embeddings-v1";
const VECTOR_DISTANCE_METRIC_V1: &str = "cosine";
const VECTOR_NORMALIZATION_V1: &str = "l2_f32_v1";

/// Exact referenced registry prepared for one project/configuration activation.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct VectorRegistryEnsure {
    pub(crate) project_uuid: Uuid,
    pub(crate) config_generation_id: String,
    pub(crate) profiles: BTreeMap<String, PreparedEmbedderProfile>,
    pub(crate) spaces: BTreeMap<VectorSpaceId, PreparedVectorSpace>,
    pub(crate) mappings: BTreeMap<String, PreparedPoolVectorSpaceMapping>,
    pub(crate) created_at_unix_ms: i64,
}

impl std::fmt::Debug for VectorRegistryEnsure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("VectorRegistryEnsure")
            .field("project_uuid", &self.project_uuid)
            .field("config_generation_id", &self.config_generation_id)
            .field("profiles", &self.profiles)
            .field("spaces", &self.spaces)
            .field("mappings", &self.mappings)
            .field("created_at_unix_ms", &self.created_at_unix_ms)
            .finish()
    }
}

/// One verified persisted profile and its first-writer timestamp.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedEmbedderProfile {
    pub(crate) profile: PreparedEmbedderProfile,
    pub(crate) created_at_unix_ms: i64,
}

/// One verified persisted space and its authoritative source cursor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedVectorSpace {
    pub(crate) space: PreparedVectorSpace,
    pub(crate) created_at_unix_ms: i64,
    pub(crate) source_seq: i64,
    pub(crate) source_updated_at_unix_ms: i64,
}

/// Exact durable key for a delayed attempt's frozen pool mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FrozenMappingKey {
    pub(crate) project_uuid: Uuid,
    pub(crate) config_generation_id: String,
    pub(crate) pool_id: String,
    pub(crate) policy_version_id: String,
}

impl FrozenMappingKey {
    pub(crate) fn new(
        project_uuid: Uuid,
        config_generation_id: impl Into<String>,
        pool_id: impl Into<String>,
        policy_version_id: impl Into<String>,
    ) -> Result<Self, String> {
        let config_generation_id = config_generation_id.into();
        let pool_id = pool_id.into();
        let policy_version_id = policy_version_id.into();
        validate_uuid_v7(project_uuid, "mapping project UUID")?;
        validate_sha256(&config_generation_id, "configuration generation")?;
        validate_sha256(&policy_version_id, "policy version")?;
        if pool_id.is_empty() || pool_id.len() > ID_MAX_BYTES {
            return Err("mapping pool ID is invalid".to_string());
        }
        Ok(Self {
            project_uuid,
            config_generation_id,
            pool_id,
            policy_version_id,
        })
    }
}

/// Fully verified mapping authority consumed by delayed Task 8 terminalization.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VerifiedPoolVectorSpaceMapping {
    pub(crate) mapping: PreparedPoolVectorSpaceMapping,
    pub(crate) profile: VerifiedEmbedderProfile,
    pub(crate) space: VerifiedVectorSpace,
    pub(crate) canonicalizer: CanonicalizerConfig,
    pub(crate) created_at_unix_ms: i64,
}

/// Config-aware vector authority for one historical pool identity.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum FrozenPoolVectorAuthority {
    Disabled,
    Enabled(Box<VerifiedPoolVectorSpaceMapping>),
}

/// Read-back proof for every row ensured by one registry operation.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VerifiedVectorRegistrySnapshot {
    pub(crate) profiles: BTreeMap<String, VerifiedEmbedderProfile>,
    pub(crate) spaces: BTreeMap<VectorSpaceId, VerifiedVectorSpace>,
    pub(crate) mappings: BTreeMap<String, VerifiedPoolVectorSpaceMapping>,
}

/// Exhaustive idempotent result of ensuring one referenced registry.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RegistryEnsureAck {
    Applied(VerifiedVectorRegistrySnapshot),
    AlreadyApplied(VerifiedVectorRegistrySnapshot),
    Conflict,
}

/// Prepare exact referenced durable rows without reading any environment value.
pub(crate) fn prepare_vector_registry(
    config: &RouterConfig,
    project_uuid: Uuid,
    config_generation_id: &str,
    policies: &BTreeMap<String, String>,
    created_at_unix_ms: i64,
) -> Result<VectorRegistryEnsure, String> {
    validate_uuid_v7(project_uuid, "registry project UUID")?;
    if created_at_unix_ms < 0 {
        return Err("registry creation timestamp is invalid".to_string());
    }
    validate_sha256(config_generation_id, "configuration generation")?;
    if config.generation_id()? != config_generation_id {
        return Err("registry configuration generation does not match config".to_string());
    }
    let expected_policies = config
        .policy_generation_values()?
        .into_iter()
        .map(|(pool_id, value)| canonical_sha256(&value).map(|hash| (pool_id, hash)))
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    if &expected_policies != policies {
        return Err("registry policy versions do not match config".to_string());
    }

    let proposed = propose_vector_space_registry(config)?;
    let profile_configs = config
        .embedders
        .iter()
        .map(|profile| (profile.id.as_str(), profile))
        .collect::<BTreeMap<_, _>>();
    let pool_configs = config
        .pools
        .iter()
        .map(|pool| (pool.id.as_str(), pool))
        .collect::<BTreeMap<_, _>>();
    let mut profiles = BTreeMap::new();
    for (version_id, artifact) in &proposed.profiles {
        let profile_config = profile_configs
            .get(artifact.profile_id.as_str())
            .ok_or_else(|| "proposed registry profile is absent from config".to_string())?;
        let prepared = PreparedEmbedderProfile::from_validated(
            profile_config,
            config.allow_remote_embedding_egress,
            artifact,
        )?;
        if prepared.embedder_profile_version_id != *version_id {
            return Err("prepared profile map key is inconsistent".to_string());
        }
        profiles.insert(version_id.clone(), prepared);
    }

    let mut spaces = BTreeMap::new();
    let mut mappings = BTreeMap::new();
    for (pool_id, mapping_artifact) in &proposed.pools {
        let pool = pool_configs
            .get(pool_id.as_str())
            .ok_or_else(|| "proposed registry pool is absent from config".to_string())?;
        let profile_config = profile_configs
            .get(mapping_artifact.profile_id.as_str())
            .ok_or_else(|| "proposed mapping profile is absent from config".to_string())?;
        let profile_artifact = proposed
            .profiles
            .get(&mapping_artifact.embedder_profile_version_id)
            .ok_or_else(|| "proposed mapping profile artifact is absent".to_string())?;
        let canonicalizer_artifact = canonicalizer_version(&pool.canonicalizer)?;
        let space_artifact = proposed
            .spaces
            .get(&mapping_artifact.vector_space_id)
            .ok_or_else(|| "proposed mapping space artifact is absent".to_string())?;
        let prepared_space = PreparedVectorSpace::from_validated(
            profile_config,
            &pool.canonicalizer,
            config.allow_remote_embedding_egress,
            profile_artifact,
            &canonicalizer_artifact,
            space_artifact,
        )?;
        let prepared_profile = profiles
            .get(&mapping_artifact.embedder_profile_version_id)
            .ok_or_else(|| "prepared mapping profile is absent".to_string())?;
        let policy_version_id = policies
            .get(pool_id)
            .ok_or_else(|| "prepared mapping policy is absent".to_string())?;
        let prepared_mapping = PreparedPoolVectorSpaceMapping::from_validated(
            project_uuid,
            config_generation_id,
            policy_version_id,
            mapping_artifact,
            prepared_profile,
            &prepared_space,
        )?;
        if let Some(existing) = spaces.insert(
            prepared_space.vector_space_id.clone(),
            prepared_space.clone(),
        ) && existing != prepared_space
        {
            return Err("prepared vector-space identity collision".to_string());
        }
        mappings.insert(pool_id.clone(), prepared_mapping);
    }
    if spaces.len() != proposed.spaces.len() || mappings.len() != proposed.pools.len() {
        return Err("prepared registry did not preserve proposed identities".to_string());
    }
    Ok(VectorRegistryEnsure {
        project_uuid,
        config_generation_id: config_generation_id.to_string(),
        profiles,
        spaces,
        mappings,
        created_at_unix_ms,
    })
}

/// Idempotently persist all referenced registry rows in the caller's transaction.
pub(crate) fn ensure_vector_registry_in_transaction(
    transaction: &Transaction<'_>,
    ensure: &VectorRegistryEnsure,
) -> Result<RegistryEnsureAck, LedgerError> {
    if !registry_shape_is_valid(ensure) || !configuration_authorizes_registry(transaction, ensure)?
    {
        return Ok(RegistryEnsureAck::Conflict);
    }

    let mut missing_profiles = Vec::new();
    for (version_id, proposed) in &ensure.profiles {
        match load_profile(transaction, version_id)? {
            Stored::Missing => missing_profiles.push(proposed),
            Stored::Valid(stored) if stored.profile == *proposed => {}
            Stored::Valid(_) | Stored::Invalid => return Ok(RegistryEnsureAck::Conflict),
        }
    }

    let mut missing_spaces = Vec::new();
    for (space_id, proposed) in &ensure.spaces {
        if semantic_space_collision(transaction, proposed)? {
            return Ok(RegistryEnsureAck::Conflict);
        }
        match load_space(transaction, ensure.project_uuid, space_id)? {
            Stored::Missing => missing_spaces.push(proposed),
            Stored::Valid(stored) if stored.space == *proposed => {}
            Stored::Valid(_) | Stored::Invalid => return Ok(RegistryEnsureAck::Conflict),
        }
    }

    let mut missing_mappings = Vec::new();
    for (pool_id, proposed) in &ensure.mappings {
        let key = FrozenMappingKey {
            project_uuid: ensure.project_uuid,
            config_generation_id: ensure.config_generation_id.clone(),
            pool_id: pool_id.clone(),
            policy_version_id: proposed.policy_version_id.clone(),
        };
        match load_mapping_row(transaction, &key)? {
            Stored::Missing => missing_mappings.push(proposed),
            Stored::Valid(stored) if stored.mapping == *proposed => {}
            Stored::Valid(_) | Stored::Invalid => return Ok(RegistryEnsureAck::Conflict),
        }
    }

    for profile in &missing_profiles {
        insert_profile(transaction, profile, ensure.created_at_unix_ms)?;
    }
    for space in &missing_spaces {
        insert_space(
            transaction,
            ensure.project_uuid,
            space,
            ensure.created_at_unix_ms,
        )?;
    }
    for mapping in &missing_mappings {
        insert_mapping(transaction, mapping, ensure.created_at_unix_ms)?;
    }

    let snapshot = load_registry_snapshot(transaction, ensure)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if missing_profiles.is_empty() && missing_spaces.is_empty() && missing_mappings.is_empty() {
        Ok(RegistryEnsureAck::AlreadyApplied(snapshot))
    } else {
        Ok(RegistryEnsureAck::Applied(snapshot))
    }
}

/// Resolve one fully verified embedder profile for provider-work consumers.
pub(crate) fn resolve_embedder_profile(
    connection: &Connection,
    embedder_profile_version_id: &str,
) -> Result<Option<VerifiedEmbedderProfile>, LedgerError> {
    match load_profile(connection, embedder_profile_version_id)? {
        Stored::Missing => Ok(None),
        Stored::Valid(profile) => Ok(Some(profile)),
        Stored::Invalid => Err(corrupt()),
    }
}

/// Resolve one fully verified vector space for catalog and embedding consumers.
pub(crate) fn resolve_vector_space(
    connection: &Connection,
    project_uuid: Uuid,
    vector_space_id: &VectorSpaceId,
) -> Result<Option<VerifiedVectorSpace>, LedgerError> {
    let space = match load_space(connection, project_uuid, vector_space_id)? {
        Stored::Missing => return Ok(None),
        Stored::Valid(space) => space,
        Stored::Invalid => return Err(corrupt()),
    };
    if semantic_space_collision(connection, &space.space)? {
        return Err(corrupt());
    }
    Ok(Some(space))
}

/// Resolve and fully verify a historical mapping without consulting current config.
pub(crate) fn resolve_frozen_mapping(
    connection: &Connection,
    key: &FrozenMappingKey,
) -> Result<Option<VerifiedPoolVectorSpaceMapping>, LedgerError> {
    let stored_mapping = match load_mapping_row(connection, key)? {
        Stored::Missing => return Ok(None),
        Stored::Valid(mapping) => mapping,
        Stored::Invalid => return Err(corrupt()),
    };
    let profile = match load_profile(
        connection,
        &stored_mapping.mapping.embedder_profile_version_id,
    )? {
        Stored::Valid(profile) => profile,
        Stored::Missing | Stored::Invalid => return Err(corrupt()),
    };
    if !configuration_authorizes_mapping(connection, &stored_mapping.mapping, &profile.profile)? {
        return Err(corrupt());
    }
    let space = match load_space(
        connection,
        key.project_uuid,
        &stored_mapping.mapping.vector_space_id,
    )? {
        Stored::Valid(space) => space,
        Stored::Missing | Stored::Invalid => return Err(corrupt()),
    };
    if profile.profile.profile_id != stored_mapping.mapping.profile_id
        || profile.profile.embedder_profile_version_id
            != stored_mapping.mapping.embedder_profile_version_id
        || space.space.embedder_profile_version_id
            != stored_mapping.mapping.embedder_profile_version_id
        || space.space.canonicalizer_version_id != stored_mapping.mapping.canonicalizer_version_id
        || semantic_space_collision(connection, &space.space)?
    {
        return Err(corrupt());
    }
    let canonicalizer =
        canonicalizer_config(&space.space.canonicalizer_identity_json).map_err(|_| corrupt())?;
    Ok(Some(VerifiedPoolVectorSpaceMapping {
        mapping: stored_mapping.mapping,
        profile,
        space,
        canonicalizer,
        created_at_unix_ms: stored_mapping.created_at_unix_ms,
    }))
}

/// Resolve whether one fully verified historical pool disabled or enabled vectors.
pub(crate) fn resolve_frozen_pool_vector_authority(
    connection: &Connection,
    key: &FrozenMappingKey,
) -> Result<FrozenPoolVectorAuthority, LedgerError> {
    if !frozen_mapping_key_is_valid(key) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let pool = load_verified_historical_pool(connection, key)?.ok_or_else(corrupt)?;
    let mapping_keys = load_frozen_pool_mapping_keys(connection, key)?;
    for mapping_key in &mapping_keys {
        if !matches!(load_mapping_row(connection, mapping_key)?, Stored::Valid(_)) {
            return Err(corrupt());
        }
    }
    if pool.learning_profile_id.is_none() {
        return if mapping_keys.is_empty() {
            Ok(FrozenPoolVectorAuthority::Disabled)
        } else {
            Err(corrupt())
        };
    }
    if mapping_keys.as_slice() != std::slice::from_ref(key) {
        return Err(corrupt());
    }
    let mapping = resolve_frozen_mapping(connection, key)?;
    match (pool.learning_profile_id, mapping) {
        (Some(profile_id), Some(mapping)) if mapping.mapping.profile_id == profile_id => {
            Ok(FrozenPoolVectorAuthority::Enabled(Box::new(mapping)))
        }
        (None, _) | (Some(_), None) | (Some(_), Some(_)) => Err(corrupt()),
    }
}

fn load_frozen_pool_mapping_keys(
    connection: &Connection,
    key: &FrozenMappingKey,
) -> Result<Vec<FrozenMappingKey>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT policy_version_id FROM pool_vector_space_mappings
             WHERE project_uuid = ?1 AND config_generation_id = ?2 AND pool_id = ?3
             ORDER BY policy_version_id",
        )
        .map_err(database_error)?;
    let policy_versions = statement
        .query_map(
            params![
                key.project_uuid.to_string(),
                key.config_generation_id,
                key.pool_id,
            ],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    policy_versions
        .into_iter()
        .map(|policy_version_id| {
            FrozenMappingKey::new(
                key.project_uuid,
                key.config_generation_id.clone(),
                key.pool_id.clone(),
                policy_version_id,
            )
            .map_err(|_| corrupt())
        })
        .collect()
}

fn load_registry_snapshot(
    connection: &Connection,
    ensure: &VectorRegistryEnsure,
) -> Result<Option<VerifiedVectorRegistrySnapshot>, LedgerError> {
    let mut profiles = BTreeMap::new();
    for version_id in ensure.profiles.keys() {
        let Stored::Valid(profile) = load_profile(connection, version_id)? else {
            return Ok(None);
        };
        profiles.insert(version_id.clone(), profile);
    }
    let mut spaces = BTreeMap::new();
    for space_id in ensure.spaces.keys() {
        let Stored::Valid(space) = load_space(connection, ensure.project_uuid, space_id)? else {
            return Ok(None);
        };
        spaces.insert(space_id.clone(), space);
    }
    let mut mappings = BTreeMap::new();
    for (pool_id, proposed) in &ensure.mappings {
        let key = FrozenMappingKey {
            project_uuid: ensure.project_uuid,
            config_generation_id: ensure.config_generation_id.clone(),
            pool_id: pool_id.clone(),
            policy_version_id: proposed.policy_version_id.clone(),
        };
        let Some(mapping) = resolve_frozen_mapping(connection, &key)? else {
            return Ok(None);
        };
        mappings.insert(pool_id.clone(), mapping);
    }
    Ok(Some(VerifiedVectorRegistrySnapshot {
        profiles,
        spaces,
        mappings,
    }))
}

#[derive(Debug)]
enum Stored<T> {
    Missing,
    Valid(T),
    Invalid,
}

struct VerifiedHistoricalPool {
    config: Json,
    pool: Json,
    learning_profile_id: Option<String>,
}

#[derive(Debug)]
struct StoredMapping {
    mapping: PreparedPoolVectorSpaceMapping,
    created_at_unix_ms: i64,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ProfileIdentity {
    schema: String,
    profile_id: String,
    protocol: String,
    endpoint_identity_sha256: String,
    model: String,
    provider_revision: String,
    dimensions: u32,
    api_key_env_sha256: Option<String>,
    timeout_ms: u64,
    max_in_flight: usize,
    batch_size: usize,
    egress_class: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct SpaceIdentity {
    schema: String,
    embedder_profile_version_id: String,
    endpoint_identity_sha256: String,
    model: String,
    provider_revision: String,
    dimensions: u32,
    distance_metric: String,
    normalization: String,
    canonicalizer_version_id: String,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct CanonicalizerIdentity {
    schema: String,
    query_schema: String,
    query_rules: String,
    text_normalization: String,
    version: u32,
    max_instruction_bytes: usize,
    max_task_bytes: usize,
    max_context_messages: usize,
    max_context_bytes: usize,
    max_position_features_bytes: usize,
    position_features: Vec<String>,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct MappingIdentity {
    schema: String,
    project_uuid: Uuid,
    config_generation_id: String,
    pool_id: String,
    policy_version_id: String,
    profile_id: String,
    embedder_profile_version_id: String,
    canonicalizer_version_id: String,
    vector_space_id: String,
}

#[derive(Debug)]
struct ProfileRow {
    version_id: String,
    profile_id: String,
    protocol: String,
    endpoint_url: String,
    endpoint_identity_sha256: String,
    model: String,
    provider_revision: String,
    dimensions: i64,
    credential_env_name_sha256: Option<String>,
    timeout_ms: i64,
    max_in_flight: i64,
    batch_size: i64,
    egress_class: String,
    canonical_profile_json: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

fn load_profile(
    connection: &Connection,
    version_id: &str,
) -> Result<Stored<VerifiedEmbedderProfile>, LedgerError> {
    let row = connection
        .query_row(
            "SELECT embedder_profile_version_id, profile_id, protocol, endpoint_url,
                    endpoint_identity_sha256, model, provider_revision, dimensions,
                    credential_env_name_sha256, timeout_ms, max_in_flight, batch_size,
                    egress_class, canonical_profile_json, created_at_unix_ms,
                    canonical_payload_hash
             FROM embedder_profiles WHERE embedder_profile_version_id = ?1",
            params![version_id],
            |row| {
                Ok(ProfileRow {
                    version_id: row.get(0)?,
                    profile_id: row.get(1)?,
                    protocol: row.get(2)?,
                    endpoint_url: row.get(3)?,
                    endpoint_identity_sha256: row.get(4)?,
                    model: row.get(5)?,
                    provider_revision: row.get(6)?,
                    dimensions: row.get(7)?,
                    credential_env_name_sha256: row.get(8)?,
                    timeout_ms: row.get(9)?,
                    max_in_flight: row.get(10)?,
                    batch_size: row.get(11)?,
                    egress_class: row.get(12)?,
                    canonical_profile_json: row.get(13)?,
                    created_at_unix_ms: row.get(14)?,
                    canonical_payload_hash: row.get(15)?,
                })
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(row) = row else {
        return Ok(Stored::Missing);
    };
    match verified_profile(row) {
        Ok(profile) => Ok(Stored::Valid(profile)),
        Err(()) => Ok(Stored::Invalid),
    }
}

fn verified_profile(row: ProfileRow) -> Result<VerifiedEmbedderProfile, ()> {
    let value = exact_canonical_value(&row.canonical_profile_json)?;
    let identity: ProfileIdentity = serde_json::from_value(value.clone()).map_err(|_| ())?;
    let dimensions =
        VectorDimensions::new(u32::try_from(row.dimensions).map_err(|_| ())?).map_err(|_| ())?;
    let timeout_ms = u64::try_from(row.timeout_ms).map_err(|_| ())?;
    let max_in_flight = usize::try_from(row.max_in_flight).map_err(|_| ())?;
    let batch_size = usize::try_from(row.batch_size).map_err(|_| ())?;
    let egress_class = parse_egress_class(&row.egress_class).ok_or(())?;
    let endpoint = canonical_endpoint_from_request_url(&row.endpoint_url).ok_or(())?;
    if row.version_id.len() != 64
        || row.profile_id.is_empty()
        || row.profile_id.len() > ID_MAX_BYTES
        || row.protocol != OPENAI_EMBEDDINGS_PROTOCOL_V1
        || row.model.is_empty()
        || row.model.len() > MODEL_ID_MAX_BYTES
        || row.provider_revision.is_empty()
        || row.provider_revision.len() > REVISION_MAX_BYTES
        || timeout_ms == 0
        || timeout_ms > EMBEDDER_TIMEOUT_MS_MAX
        || max_in_flight == 0
        || max_in_flight > EMBEDDER_MAX_IN_FLIGHT
        || batch_size == 0
        || batch_size > EMBEDDER_BATCH_SIZE_MAX
        || row.created_at_unix_ms < 0
        || row.canonical_payload_hash != row.version_id
        || canonical_sha256(&value).map_err(|_| ())? != row.version_id
        || endpoint.request_url.as_str() != row.endpoint_url
        || endpoint.endpoint_identity_sha256 != row.endpoint_identity_sha256
        || endpoint.egress_class != egress_class
        || row
            .credential_env_name_sha256
            .as_deref()
            .is_some_and(|hash| !is_sha256(hash))
        || identity
            != (ProfileIdentity {
                schema: EMBEDDER_PROFILE_SCHEMA_V1.to_string(),
                profile_id: row.profile_id.clone(),
                protocol: row.protocol.clone(),
                endpoint_identity_sha256: row.endpoint_identity_sha256.clone(),
                model: row.model.clone(),
                provider_revision: row.provider_revision.clone(),
                dimensions: dimensions.value(),
                api_key_env_sha256: row.credential_env_name_sha256.clone(),
                timeout_ms,
                max_in_flight,
                batch_size,
                egress_class: row.egress_class.clone(),
            })
    {
        return Err(());
    }
    Ok(VerifiedEmbedderProfile {
        profile: PreparedEmbedderProfile {
            profile_id: row.profile_id,
            embedder_profile_version_id: row.version_id,
            protocol: OPENAI_EMBEDDINGS_PROTOCOL_V1,
            endpoint_url: row.endpoint_url,
            endpoint_identity_sha256: row.endpoint_identity_sha256,
            model: row.model,
            provider_revision: row.provider_revision,
            dimensions,
            credential_env_name_sha256: row.credential_env_name_sha256,
            timeout_ms,
            max_in_flight,
            batch_size,
            egress_class,
            canonical_profile_json: row.canonical_profile_json,
        },
        created_at_unix_ms: row.created_at_unix_ms,
    })
}

#[derive(Debug)]
struct SpaceRow {
    vector_space_id: String,
    project_uuid: String,
    profile_version_id: String,
    canonicalizer_version_id: String,
    canonicalizer_identity_json: String,
    endpoint_identity_sha256: String,
    model: String,
    provider_revision: String,
    dimensions: i64,
    metric: String,
    normalization: String,
    canonical_space_json: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

fn load_space(
    connection: &Connection,
    project_uuid: Uuid,
    vector_space_id: &VectorSpaceId,
) -> Result<Stored<VerifiedVectorSpace>, LedgerError> {
    let row = connection
        .query_row(
            "SELECT vector_space_id, project_uuid, embedder_profile_version_id,
                    canonicalizer_version_id, canonicalizer_identity_json,
                    endpoint_identity_sha256, model, provider_revision, dimensions,
                    metric, normalization, canonical_space_json, created_at_unix_ms,
                    canonical_payload_hash
             FROM vector_spaces WHERE vector_space_id = ?1",
            params![vector_space_id.as_str()],
            |row| {
                Ok(SpaceRow {
                    vector_space_id: row.get(0)?,
                    project_uuid: row.get(1)?,
                    profile_version_id: row.get(2)?,
                    canonicalizer_version_id: row.get(3)?,
                    canonicalizer_identity_json: row.get(4)?,
                    endpoint_identity_sha256: row.get(5)?,
                    model: row.get(6)?,
                    provider_revision: row.get(7)?,
                    dimensions: row.get(8)?,
                    metric: row.get(9)?,
                    normalization: row.get(10)?,
                    canonical_space_json: row.get(11)?,
                    created_at_unix_ms: row.get(12)?,
                    canonical_payload_hash: row.get(13)?,
                })
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(row) = row else {
        return Ok(Stored::Missing);
    };
    match verified_space(connection, project_uuid, row) {
        Ok(space) => Ok(Stored::Valid(space)),
        Err(()) => Ok(Stored::Invalid),
    }
}

fn verified_space(
    connection: &Connection,
    project_uuid: Uuid,
    row: SpaceRow,
) -> Result<VerifiedVectorSpace, ()> {
    let space_value = exact_canonical_value(&row.canonical_space_json)?;
    let space_identity: SpaceIdentity =
        serde_json::from_value(space_value.clone()).map_err(|_| ())?;
    let canonicalizer_value = exact_canonical_value(&row.canonicalizer_identity_json)?;
    let canonicalizer_identity: CanonicalizerIdentity =
        serde_json::from_value(canonicalizer_value.clone()).map_err(|_| ())?;
    let vector_space_id = VectorSpaceId::new(row.vector_space_id.clone()).map_err(|_| ())?;
    let dimensions =
        VectorDimensions::new(u32::try_from(row.dimensions).map_err(|_| ())?).map_err(|_| ())?;
    let Stored::Valid(profile) =
        load_profile(connection, &row.profile_version_id).map_err(|_| ())?
    else {
        return Err(());
    };
    if row.project_uuid != project_uuid.to_string()
        || row.created_at_unix_ms < 0
        || row.canonical_payload_hash != row.vector_space_id
        || canonical_sha256(&space_value).map_err(|_| ())? != row.vector_space_id
        || canonical_sha256(&canonicalizer_value).map_err(|_| ())? != row.canonicalizer_version_id
        || canonicalizer_identity.schema != CANONICALIZER_IDENTITY_SCHEMA_V1
        || canonicalizer_identity.query_schema != CANONICAL_ROUTING_QUERY_SCHEMA_V1
        || canonicalizer_identity.query_rules != CANONICAL_ROUTING_QUERY_RULES_V1
        || canonicalizer_identity.text_normalization != CANONICAL_TEXT_NORMALIZATION_V1
        || row.endpoint_identity_sha256 != profile.profile.endpoint_identity_sha256
        || row.model != profile.profile.model
        || row.provider_revision != profile.profile.provider_revision
        || dimensions != profile.profile.dimensions
        || row.metric != VECTOR_DISTANCE_METRIC_V1
        || row.normalization != VECTOR_NORMALIZATION_V1
        || space_identity
            != (SpaceIdentity {
                schema: VECTOR_SPACE_SCHEMA_V1.to_string(),
                embedder_profile_version_id: row.profile_version_id.clone(),
                endpoint_identity_sha256: row.endpoint_identity_sha256.clone(),
                model: row.model.clone(),
                provider_revision: row.provider_revision.clone(),
                dimensions: dimensions.value(),
                distance_metric: row.metric.clone(),
                normalization: row.normalization.clone(),
                canonicalizer_version_id: row.canonicalizer_version_id.clone(),
            })
    {
        return Err(());
    }
    let cursor = connection
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
        .map_err(|_| ())?
        .ok_or(())?;
    if cursor.0 < 0
        || cursor.1 < 0
        || vector_source_sequence_payload_hash(&vector_space_id, cursor.0, cursor.1)
            .map_err(|_| ())?
            != cursor.2
    {
        return Err(());
    }
    Ok(VerifiedVectorSpace {
        space: PreparedVectorSpace {
            vector_space_id,
            embedder_profile_version_id: row.profile_version_id,
            canonicalizer_version_id: row.canonicalizer_version_id,
            canonicalizer_identity_json: row.canonicalizer_identity_json,
            endpoint_identity_sha256: row.endpoint_identity_sha256,
            model: row.model,
            provider_revision: row.provider_revision,
            dimensions,
            metric: VECTOR_DISTANCE_METRIC_V1,
            normalization: VECTOR_NORMALIZATION_V1,
            canonical_space_json: row.canonical_space_json,
        },
        created_at_unix_ms: row.created_at_unix_ms,
        source_seq: cursor.0,
        source_updated_at_unix_ms: cursor.1,
    })
}

fn load_mapping_row(
    connection: &Connection,
    key: &FrozenMappingKey,
) -> Result<Stored<StoredMapping>, LedgerError> {
    let row = connection
        .query_row(
            "SELECT project_uuid, config_generation_id, pool_id, policy_version_id,
                    profile_id, embedder_profile_version_id, canonicalizer_version_id,
                    vector_space_id, canonical_mapping_json, created_at_unix_ms,
                    canonical_payload_hash
             FROM pool_vector_space_mappings
             WHERE project_uuid = ?1 AND config_generation_id = ?2
               AND pool_id = ?3 AND policy_version_id = ?4",
            params![
                key.project_uuid.to_string(),
                key.config_generation_id,
                key.pool_id,
                key.policy_version_id,
            ],
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
                    row.get::<_, i64>(9)?,
                    row.get::<_, String>(10)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(row) = row else {
        return Ok(Stored::Missing);
    };
    match verified_mapping_row(row) {
        Ok(mapping) => Ok(Stored::Valid(mapping)),
        Err(()) => Ok(Stored::Invalid),
    }
}

#[allow(clippy::type_complexity)]
fn verified_mapping_row(
    row: (
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        i64,
        String,
    ),
) -> Result<StoredMapping, ()> {
    let (
        project_uuid,
        config_generation_id,
        pool_id,
        policy_version_id,
        profile_id,
        profile_version_id,
        canonicalizer_version_id,
        vector_space_id,
        canonical_mapping_json,
        created_at_unix_ms,
        canonical_payload_hash,
    ) = row;
    let value = exact_canonical_value(&canonical_mapping_json)?;
    let identity: MappingIdentity = serde_json::from_value(value.clone()).map_err(|_| ())?;
    let project_uuid_value = Uuid::parse_str(&project_uuid).map_err(|_| ())?;
    let vector_space_id_value = VectorSpaceId::new(vector_space_id.clone()).map_err(|_| ())?;
    if project_uuid_value.get_version_num() != 7
        || project_uuid_value.get_variant() != Variant::RFC4122
        || created_at_unix_ms < 0
        || canonical_sha256(&value).map_err(|_| ())? != canonical_payload_hash
        || identity
            != (MappingIdentity {
                schema: POOL_VECTOR_SPACE_MAPPING_SCHEMA_V1.to_string(),
                project_uuid: project_uuid_value,
                config_generation_id: config_generation_id.clone(),
                pool_id: pool_id.clone(),
                policy_version_id: policy_version_id.clone(),
                profile_id: profile_id.clone(),
                embedder_profile_version_id: profile_version_id.clone(),
                canonicalizer_version_id: canonicalizer_version_id.clone(),
                vector_space_id: vector_space_id.clone(),
            })
    {
        return Err(());
    }
    Ok(StoredMapping {
        mapping: PreparedPoolVectorSpaceMapping {
            project_uuid: project_uuid_value,
            config_generation_id,
            pool_id,
            policy_version_id,
            profile_id,
            embedder_profile_version_id: profile_version_id,
            canonicalizer_version_id,
            vector_space_id: vector_space_id_value,
            canonical_mapping_json,
            canonical_payload_hash,
        },
        created_at_unix_ms,
    })
}

fn registry_shape_is_valid(ensure: &VectorRegistryEnsure) -> bool {
    ensure.created_at_unix_ms >= 0
        && ensure.project_uuid.get_version_num() == 7
        && ensure.project_uuid.get_variant() == Variant::RFC4122
        && is_sha256(&ensure.config_generation_id)
        && ensure
            .profiles
            .iter()
            .all(|(key, value)| key == &value.embedder_profile_version_id)
        && ensure
            .spaces
            .iter()
            .all(|(key, value)| key == &value.vector_space_id)
        && ensure.mappings.iter().all(|(pool_id, mapping)| {
            pool_id == &mapping.pool_id
                && mapping.project_uuid == ensure.project_uuid
                && mapping.config_generation_id == ensure.config_generation_id
                && ensure
                    .profiles
                    .contains_key(&mapping.embedder_profile_version_id)
                && ensure.spaces.contains_key(&mapping.vector_space_id)
        })
}

fn configuration_authorizes_registry(
    connection: &Connection,
    ensure: &VectorRegistryEnsure,
) -> Result<bool, LedgerError> {
    if !project_and_config_are_valid(
        connection,
        ensure.project_uuid,
        &ensure.config_generation_id,
    )? {
        return Ok(false);
    }
    for mapping in ensure.mappings.values() {
        let Some(profile) = ensure.profiles.get(&mapping.embedder_profile_version_id) else {
            return Ok(false);
        };
        if !configuration_authorizes_mapping(connection, mapping, profile)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn project_and_config_are_valid(
    connection: &Connection,
    project_uuid: Uuid,
    config_generation_id: &str,
) -> Result<bool, LedgerError> {
    let project_matches: bool = connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM project_metadata
                WHERE singleton_key = 1 AND project_uuid = ?1
             )",
            params![project_uuid.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    if !project_matches {
        return Ok(false);
    }
    let config = connection
        .query_row(
            "SELECT project_uuid, canonical_config_json, canonical_payload_hash
             FROM config_generations WHERE config_generation_id = ?1",
            params![config_generation_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((stored_project, canonical, payload_hash)) = config else {
        return Ok(false);
    };
    let Ok(value) = exact_canonical_value(&canonical) else {
        return Ok(false);
    };
    Ok(stored_project == project_uuid.to_string()
        && payload_hash == config_generation_id
        && canonical_sha256(&value).ok().as_deref() == Some(config_generation_id))
}

fn configuration_authorizes_mapping(
    connection: &Connection,
    mapping: &PreparedPoolVectorSpaceMapping,
    expected_profile: &PreparedEmbedderProfile,
) -> Result<bool, LedgerError> {
    let Ok(key) = FrozenMappingKey::new(
        mapping.project_uuid,
        mapping.config_generation_id.clone(),
        mapping.pool_id.clone(),
        mapping.policy_version_id.clone(),
    ) else {
        return Ok(false);
    };
    let Some(authority) = load_verified_historical_pool(connection, &key)? else {
        return Ok(false);
    };
    let config_profile = authority
        .config
        .get("embedders")
        .and_then(Json::as_array)
        .and_then(|profiles| {
            profiles.iter().find(|profile| {
                profile.get("profile_id").and_then(Json::as_str) == Some(&mapping.profile_id)
            })
        });
    let expected_profile_value =
        exact_canonical_value(&expected_profile.canonical_profile_json).ok();
    let profile_matches = config_profile.is_some_and(|profile| {
        canonical_sha256(profile).ok().as_deref() == Some(&mapping.embedder_profile_version_id)
            && expected_profile_value.as_ref() == Some(profile)
            && expected_profile.profile_id == mapping.profile_id
            && expected_profile.embedder_profile_version_id == mapping.embedder_profile_version_id
    });
    if !profile_matches
        || authority.learning_profile_id.as_deref() != Some(&mapping.profile_id)
        || authority
            .pool
            .pointer("/canonicalizer/canonicalizer_version_id")
            .and_then(Json::as_str)
            != Some(&mapping.canonicalizer_version_id)
    {
        return Ok(false);
    }
    Ok(true)
}

fn load_verified_historical_pool(
    connection: &Connection,
    key: &FrozenMappingKey,
) -> Result<Option<VerifiedHistoricalPool>, LedgerError> {
    if !frozen_mapping_key_is_valid(key)
        || !project_and_config_are_valid(connection, key.project_uuid, &key.config_generation_id)?
    {
        return Ok(None);
    }
    let config_json = connection
        .query_row(
            "SELECT canonical_config_json FROM config_generations
             WHERE project_uuid = ?1 AND config_generation_id = ?2",
            params![key.project_uuid.to_string(), key.config_generation_id,],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let Some(config_json) = config_json else {
        return Ok(None);
    };
    let Ok(config) = exact_canonical_value(&config_json) else {
        return Ok(None);
    };
    if config.get("schema").and_then(Json::as_str) != Some(CONFIG_GENERATION_SCHEMA_V1) {
        return Ok(None);
    }
    let Some(pools) = config.get("pools").and_then(Json::as_array) else {
        return Ok(None);
    };
    let mut matching_pools = pools
        .iter()
        .filter(|pool| pool.get("id").and_then(Json::as_str) == Some(&key.pool_id));
    let Some(pool) = matching_pools.next() else {
        return Ok(None);
    };
    if matching_pools.next().is_some() {
        return Ok(None);
    }
    let pool = pool.clone();
    let Ok(learning_profile_id) = historical_learning_profile(&pool) else {
        return Ok(None);
    };
    let policy = connection
        .query_row(
            "SELECT canonical_policy_json, canonical_payload_hash
             FROM policy_versions
             WHERE project_uuid = ?1 AND pool_id = ?2 AND policy_version_id = ?3",
            params![
                key.project_uuid.to_string(),
                key.pool_id,
                key.policy_version_id,
            ],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    let Some((policy_json, payload_hash)) = policy else {
        return Ok(None);
    };
    let Ok(policy_value) = exact_canonical_value(&policy_json) else {
        return Ok(None);
    };
    if payload_hash != key.policy_version_id
        || canonical_sha256(&policy_value).ok().as_deref() != Some(&key.policy_version_id)
        || policy_value.get("schema").and_then(Json::as_str) != Some(POLICY_SCHEMA_V1)
        || policy_value.pointer("/pool") != Some(&pool)
    {
        return Ok(None);
    }
    Ok(Some(VerifiedHistoricalPool {
        config,
        pool,
        learning_profile_id,
    }))
}

fn historical_learning_profile(pool: &Json) -> Result<Option<String>, ()> {
    let learning = pool.get("learning").and_then(Json::as_object).ok_or(())?;
    if learning.is_empty() {
        return Ok(None);
    }
    const MINIMAL_KEYS: [&str; 2] = ["embedder", "version"];
    const COMPLETE_KEYS: [&str; 14] = [
        "algorithms",
        "embedder",
        "familywise_credible_level",
        "min_coverage",
        "min_effective_samples",
        "min_independent_roots",
        "min_points",
        "prior_failure",
        "prior_success",
        "promotion_lower_bound",
        "radius",
        "time_decay_half_life_seconds",
        "top_k",
        "version",
    ];
    const ACTIVE_KEYS: [&str; 17] = [
        "active_canary_fraction",
        "algorithms",
        "embedder",
        "familywise_credible_level",
        "holdout_probability",
        "min_coverage",
        "min_effective_samples",
        "min_independent_roots",
        "min_points",
        "prior_failure",
        "prior_success",
        "promotion_lower_bound",
        "radius",
        "retention_lower_bound",
        "time_decay_half_life_seconds",
        "top_k",
        "version",
    ];
    let keys = learning.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let minimal = MINIMAL_KEYS.into_iter().collect::<BTreeSet<_>>();
    let complete = COMPLETE_KEYS.into_iter().collect::<BTreeSet<_>>();
    let active = ACTIVE_KEYS.into_iter().collect::<BTreeSet<_>>();
    if (keys != minimal && keys != complete && keys != active)
        || learning.get("version").and_then(Json::as_u64) != Some(1)
    {
        return Err(());
    }
    let profile_id = learning.get("embedder").and_then(Json::as_str).ok_or(())?;
    if profile_id.is_empty() || profile_id.len() > ID_MAX_BYTES {
        return Err(());
    }
    Ok(Some(profile_id.to_string()))
}

fn frozen_mapping_key_is_valid(key: &FrozenMappingKey) -> bool {
    key.project_uuid.get_version_num() == 7
        && key.project_uuid.get_variant() == Variant::RFC4122
        && is_sha256(&key.config_generation_id)
        && !key.pool_id.is_empty()
        && key.pool_id.len() <= ID_MAX_BYTES
        && is_sha256(&key.policy_version_id)
}

fn semantic_space_collision(
    connection: &Connection,
    proposed: &PreparedVectorSpace,
) -> Result<bool, LedgerError> {
    let alternate: Option<String> = connection
        .query_row(
            "SELECT vector_space_id FROM vector_spaces
             WHERE embedder_profile_version_id = ?1 AND canonicalizer_version_id = ?2
               AND vector_space_id <> ?3
             LIMIT 1",
            params![
                proposed.embedder_profile_version_id,
                proposed.canonicalizer_version_id,
                proposed.vector_space_id.as_str(),
            ],
            |row| row.get(0),
        )
        .optional()
        .map_err(database_error)?;
    Ok(alternate.is_some())
}

fn insert_profile(
    transaction: &Transaction<'_>,
    profile: &PreparedEmbedderProfile,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    transaction
        .execute(
            "INSERT INTO embedder_profiles (
                embedder_profile_version_id, profile_id, protocol, endpoint_url,
                endpoint_identity_sha256, model, provider_revision, dimensions,
                credential_env_name_sha256, timeout_ms, max_in_flight, batch_size,
                egress_class, canonical_profile_json, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?1)",
            params![
                profile.embedder_profile_version_id,
                profile.profile_id,
                profile.protocol,
                profile.endpoint_url,
                profile.endpoint_identity_sha256,
                profile.model,
                profile.provider_revision,
                i64::from(profile.dimensions.value()),
                profile.credential_env_name_sha256,
                i64::try_from(profile.timeout_ms)
                    .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
                i64::try_from(profile.max_in_flight)
                    .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
                i64::try_from(profile.batch_size)
                    .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
                profile.egress_class.as_str(),
                profile.canonical_profile_json,
                created_at_unix_ms,
            ],
        )
        .map_err(database_error)?;
    Ok(())
}

fn insert_space(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    space: &PreparedVectorSpace,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    transaction
        .execute(
            "INSERT INTO vector_spaces (
                vector_space_id, project_uuid, embedder_profile_version_id,
                canonicalizer_version_id, canonicalizer_identity_json,
                endpoint_identity_sha256, model, provider_revision, dimensions,
                metric, normalization, canonical_space_json, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?1)",
            params![
                space.vector_space_id.as_str(),
                project_uuid.to_string(),
                space.embedder_profile_version_id,
                space.canonicalizer_version_id,
                space.canonicalizer_identity_json,
                space.endpoint_identity_sha256,
                space.model,
                space.provider_revision,
                i64::from(space.dimensions.value()),
                space.metric,
                space.normalization,
                space.canonical_space_json,
                created_at_unix_ms,
            ],
        )
        .map_err(database_error)?;
    let cursor_hash =
        vector_source_sequence_payload_hash(&space.vector_space_id, 0, created_at_unix_ms)?;
    transaction
        .execute(
            "INSERT INTO vector_space_source_sequences (
                vector_space_id, source_seq, updated_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, 0, ?2, ?3)",
            params![
                space.vector_space_id.as_str(),
                created_at_unix_ms,
                cursor_hash,
            ],
        )
        .map_err(database_error)?;
    Ok(())
}

fn insert_mapping(
    transaction: &Transaction<'_>,
    mapping: &PreparedPoolVectorSpaceMapping,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    transaction
        .execute(
            "INSERT INTO pool_vector_space_mappings (
                project_uuid, config_generation_id, pool_id, policy_version_id,
                profile_id, embedder_profile_version_id, canonicalizer_version_id,
                vector_space_id, canonical_mapping_json, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            params![
                mapping.project_uuid.to_string(),
                mapping.config_generation_id,
                mapping.pool_id,
                mapping.policy_version_id,
                mapping.profile_id,
                mapping.embedder_profile_version_id,
                mapping.canonicalizer_version_id,
                mapping.vector_space_id.as_str(),
                mapping.canonical_mapping_json,
                created_at_unix_ms,
                mapping.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    Ok(())
}

fn canonicalizer_config(value: &str) -> Result<CanonicalizerConfig, ()> {
    let parsed = exact_canonical_value(value)?;
    let identity: CanonicalizerIdentity = serde_json::from_value(parsed).map_err(|_| ())?;
    if identity.schema != CANONICALIZER_IDENTITY_SCHEMA_V1
        || identity.query_schema != CANONICAL_ROUTING_QUERY_SCHEMA_V1
        || identity.query_rules != CANONICAL_ROUTING_QUERY_RULES_V1
        || identity.text_normalization != CANONICAL_TEXT_NORMALIZATION_V1
        || identity.version != 1
        || identity
            .position_features
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        || identity
            .position_features
            .iter()
            .any(|name| name != "turn_index")
    {
        return Err(());
    }
    Ok(CanonicalizerConfig {
        version: identity.version,
        max_instruction_bytes: identity.max_instruction_bytes,
        max_task_bytes: identity.max_task_bytes,
        max_context_messages: identity.max_context_messages,
        max_context_bytes: identity.max_context_bytes,
        max_position_features_bytes: identity.max_position_features_bytes,
        position_features: identity.position_features,
        unknown_fields: BTreeMap::new(),
    })
}

fn canonical_endpoint_from_request_url(
    endpoint_url: &str,
) -> Option<crate::embedding_identity::CanonicalEmbedderEndpoint> {
    let base_url = endpoint_url.strip_suffix("/embeddings")?;
    let endpoint = normalize_embedder_endpoint(base_url, true).ok()?;
    (endpoint.request_url.as_str() == endpoint_url).then_some(endpoint)
}

fn parse_egress_class(value: &str) -> Option<EmbedderEgressClass> {
    match value {
        "loopback_http" => Some(EmbedderEgressClass::LoopbackHttp),
        "loopback_https" => Some(EmbedderEgressClass::LoopbackHttps),
        "remote_https" => Some(EmbedderEgressClass::RemoteHttps),
        _ => None,
    }
}

fn exact_canonical_value(value: &str) -> Result<Json, ()> {
    let parsed = serde_json::from_str::<Json>(value).map_err(|_| ())?;
    if canonical_json(&parsed).map_err(|_| ())? != value {
        return Err(());
    }
    Ok(parsed)
}

fn validate_uuid_v7(value: Uuid, kind: &str) -> Result<(), String> {
    if value.get_version_num() != 7 || value.get_variant() != Variant::RFC4122 {
        return Err(format!("{kind} is not UUIDv7"));
    }
    Ok(())
}

fn validate_sha256(value: &str, kind: &str) -> Result<(), String> {
    if !is_sha256(value) {
        return Err(format!("{kind} is not lowercase SHA-256"));
    }
    Ok(())
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn database_error(error: rusqlite::Error) -> LedgerError {
    map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}

fn corrupt() -> LedgerError {
    LedgerErrorClass::CorruptDatabase.into()
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::{Arc, Barrier};
    use std::thread;

    use tempfile::TempDir;

    use super::*;
    use crate::config::LearningConfig;
    use crate::ledger::repository::tests::{config as base_config, database_path};
    use crate::ledger::repository::{ActivatedLedger, LedgerRepository};

    const ACTIVATED_AT: i64 = 1_000;
    const RETRY_AT: i64 = 2_000;

    struct Fixture {
        _temporary: TempDir,
        path: PathBuf,
        config: RouterConfig,
        activated: ActivatedLedger,
        ensure: VectorRegistryEnsure,
    }

    fn learning_config(path: &Path, project_id: &str) -> RouterConfig {
        let mut config = base_config(path, project_id);
        config.pools[0].learning = Some(LearningConfig::minimal(config.embedders[0].id.clone()));
        config
    }

    fn prepare_for(
        config: &RouterConfig,
        activated: &ActivatedLedger,
        created_at_unix_ms: i64,
    ) -> VectorRegistryEnsure {
        let policies = activated
            .identity
            .pools
            .iter()
            .map(|(pool_id, identity)| (pool_id.clone(), identity.policy_version_id.clone()))
            .collect();
        prepare_vector_registry(
            config,
            activated.identity.project_uuid,
            &activated.identity.config_generation_id,
            &policies,
            created_at_unix_ms,
        )
        .unwrap()
    }

    fn fixture() -> Fixture {
        let temporary = tempfile::tempdir().unwrap();
        let path = database_path(&temporary);
        let config = learning_config(&path, "vector-registry-fixture");
        let activated = LedgerRepository::activate_at(&config, ACTIVATED_AT).unwrap();
        let ensure = prepare_for(&config, &activated, RETRY_AT);
        Fixture {
            _temporary: temporary,
            path,
            config,
            activated,
            ensure,
        }
    }

    #[test]
    fn historical_learning_profile_accepts_only_generated_minimal_or_complete_shapes() {
        let minimal = serde_json::json!({
            "learning": {"version": 1, "embedder": "embedder-a"}
        });
        assert_eq!(
            historical_learning_profile(&minimal),
            Ok(Some("embedder-a".to_string()))
        );

        let complete = serde_json::json!({
            "learning": {
                "version": 1,
                "embedder": "embedder-a",
                "top_k": 1,
                "radius": 1.0,
                "min_points": 1,
                "min_independent_roots": 1,
                "min_effective_samples": 1.0,
                "min_coverage": 0.0,
                "time_decay_half_life_seconds": 3600.0,
                "prior_success": 1.0,
                "prior_failure": 1.0,
                "familywise_credible_level": 0.95,
                "promotion_lower_bound": 0.2,
                "algorithms": {},
            }
        });
        assert_eq!(
            historical_learning_profile(&complete),
            Ok(Some("embedder-a".to_string()))
        );

        let mut active = complete.clone();
        active["learning"]["retention_lower_bound"] = serde_json::json!(0.1);
        active["learning"]["holdout_probability"] = serde_json::json!(0.2);
        active["learning"]["active_canary_fraction"] = serde_json::json!(0.4);
        assert_eq!(
            historical_learning_profile(&active),
            Ok(Some("embedder-a".to_string()))
        );

        let mut partial = complete.clone();
        partial["learning"]
            .as_object_mut()
            .unwrap()
            .remove("algorithms");
        assert_eq!(historical_learning_profile(&partial), Err(()));
        let mut unknown = complete;
        unknown["learning"]["unexpected"] = Json::Bool(true);
        assert_eq!(historical_learning_profile(&unknown), Err(()));
    }

    fn scalar(connection: &Connection, table: &str) -> i64 {
        connection
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap()
    }

    fn mapping_key(ensure: &VectorRegistryEnsure) -> FrozenMappingKey {
        let mapping = &ensure.mappings["pool-a"];
        FrozenMappingKey::new(
            ensure.project_uuid,
            ensure.config_generation_id.clone(),
            "pool-a",
            mapping.policy_version_id.clone(),
        )
        .unwrap()
    }

    fn identity_mapping_key(activated: &ActivatedLedger) -> FrozenMappingKey {
        let pool = activated.identity.pool("pool-a").unwrap();
        FrozenMappingKey::new(
            activated.identity.project_uuid,
            activated.identity.config_generation_id.clone(),
            "pool-a",
            pool.policy_version_id.clone(),
        )
        .unwrap()
    }

    #[test]
    fn exact_retry_retains_first_timestamps_and_resolves_frozen_authority() {
        let mut fixture = fixture();
        let key = mapping_key(&fixture.ensure);
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        let snapshot =
            match ensure_vector_registry_in_transaction(&transaction, &fixture.ensure).unwrap() {
                RegistryEnsureAck::AlreadyApplied(snapshot) => snapshot,
                other => panic!("unexpected retry result: {other:?}"),
            };
        assert!(
            snapshot
                .profiles
                .values()
                .all(|profile| profile.created_at_unix_ms == ACTIVATED_AT)
        );
        assert!(snapshot.spaces.values().all(|space| {
            space.created_at_unix_ms == ACTIVATED_AT
                && space.source_seq == 0
                && space.source_updated_at_unix_ms == ACTIVATED_AT
        }));
        assert!(
            snapshot
                .mappings
                .values()
                .all(|mapping| mapping.created_at_unix_ms == ACTIVATED_AT)
        );

        let resolved = resolve_frozen_mapping(&transaction, &key)
            .unwrap()
            .expect("mapping should resolve");
        assert_eq!(resolved.mapping, fixture.ensure.mappings["pool-a"]);
        assert_eq!(resolved.created_at_unix_ms, ACTIVATED_AT);
        assert_eq!(
            resolve_vector_space(
                &transaction,
                fixture.ensure.project_uuid,
                &resolved.mapping.vector_space_id,
            )
            .unwrap()
            .unwrap(),
            resolved.space
        );
        assert_eq!(
            resolve_frozen_pool_vector_authority(&transaction, &key).unwrap(),
            FrozenPoolVectorAuthority::Enabled(Box::new(resolved))
        );
        transaction.commit().unwrap();
    }

    #[test]
    fn frozen_pool_authority_distinguishes_disabled_from_enabled() {
        let temporary = tempfile::tempdir().unwrap();
        let path = database_path(&temporary);
        let mut disabled =
            LedgerRepository::activate_at(&base_config(&path, "vector-registry-disabled"), 500)
                .unwrap();
        let disabled_key = identity_mapping_key(&disabled);
        let disabled_config: String = disabled
            .repository
            .connection_mut()
            .query_row(
                "SELECT canonical_config_json FROM config_generations
                 WHERE config_generation_id = ?1",
                params![disabled_key.config_generation_id],
                |row| row.get(0),
            )
            .unwrap();
        assert!(
            exact_canonical_value(&disabled_config)
                .unwrap()
                .pointer("/pools/0/learning")
                .and_then(Json::as_object)
                .is_some_and(serde_json::Map::is_empty)
        );
        assert_eq!(
            resolve_frozen_pool_vector_authority(
                disabled.repository.connection_mut(),
                &disabled_key,
            )
            .unwrap(),
            FrozenPoolVectorAuthority::Disabled
        );

        let enabled_config = learning_config(&path, "vector-registry-disabled");
        let mut enabled = LedgerRepository::activate_at(&enabled_config, ACTIVATED_AT).unwrap();
        let enabled_key = identity_mapping_key(&enabled);
        assert!(matches!(
            resolve_frozen_pool_vector_authority(
                enabled.repository.connection_mut(),
                &enabled_key,
            )
            .unwrap(),
            FrozenPoolVectorAuthority::Enabled(_)
        ));
    }

    fn assert_frozen_authority_corrupt(update: &str) {
        let mut fixture = fixture();
        let key = mapping_key(&fixture.ensure);
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        transaction.execute_batch(update).unwrap();
        assert_eq!(
            resolve_frozen_pool_vector_authority(&transaction, &key)
                .unwrap_err()
                .class(),
            LedgerErrorClass::CorruptDatabase
        );
    }

    #[test]
    fn enabled_frozen_pool_rejects_missing_or_tampered_authority() {
        for update in [
            "DELETE FROM pool_vector_space_mappings",
            "UPDATE pool_vector_space_mappings SET canonical_mapping_json = '{}'",
            "UPDATE config_generations SET canonical_config_json = '{}'",
            "UPDATE policy_versions SET canonical_policy_json = '{}'",
        ] {
            assert_frozen_authority_corrupt(update);
        }
    }

    #[test]
    fn registry_creation_rolls_back_as_one_unit() {
        let mut fixture = fixture();
        let connection = fixture.activated.repository.connection_mut();
        connection
            .execute_batch(
                "DELETE FROM pool_vector_space_mappings;
                 DELETE FROM vector_space_source_sequences;
                 DELETE FROM vector_spaces;
                 DELETE FROM embedder_profiles;",
            )
            .unwrap();
        let transaction = connection.transaction().unwrap();
        assert!(matches!(
            ensure_vector_registry_in_transaction(&transaction, &fixture.ensure).unwrap(),
            RegistryEnsureAck::Applied(_)
        ));
        assert_eq!(scalar(&transaction, "embedder_profiles"), 1);
        assert_eq!(scalar(&transaction, "vector_spaces"), 1);
        assert_eq!(scalar(&transaction, "vector_space_source_sequences"), 1);
        assert_eq!(scalar(&transaction, "pool_vector_space_mappings"), 1);
        drop(transaction);
        assert_eq!(scalar(connection, "embedder_profiles"), 0);
        assert_eq!(scalar(connection, "vector_spaces"), 0);
        assert_eq!(scalar(connection, "vector_space_source_sequences"), 0);
        assert_eq!(scalar(connection, "pool_vector_space_mappings"), 0);
    }

    fn assert_update_conflicts(fixture: &mut Fixture, update: &str) {
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        transaction.execute_batch(update).unwrap();
        assert!(matches!(
            ensure_vector_registry_in_transaction(&transaction, &fixture.ensure).unwrap(),
            RegistryEnsureAck::Conflict
        ));
    }

    #[test]
    fn denormalized_and_full_payload_tampering_returns_typed_conflict() {
        let mut fixture = fixture();
        for update in [
            "UPDATE embedder_profiles SET model = 'tampered-profile-model'",
            "UPDATE vector_spaces SET provider_revision = 'tampered-space-revision'",
            "UPDATE pool_vector_space_mappings SET canonical_mapping_json = '{}'",
            "UPDATE vector_space_source_sequences SET canonical_payload_hash = '0000000000000000000000000000000000000000000000000000000000000000'",
        ] {
            assert_update_conflicts(&mut fixture, update);
        }
    }

    #[test]
    fn alternate_semantic_space_is_a_collision() {
        let mut fixture = fixture();
        let proposed = fixture.ensure.spaces.values().next().unwrap();
        let alternate_id = "0".repeat(64);
        assert_ne!(alternate_id, proposed.vector_space_id.as_str());
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        transaction
            .execute(
                "INSERT INTO vector_spaces (
                    vector_space_id, project_uuid, embedder_profile_version_id,
                    canonicalizer_version_id, canonicalizer_identity_json,
                    endpoint_identity_sha256, model, provider_revision, dimensions,
                    metric, normalization, canonical_space_json, created_at_unix_ms,
                    canonical_payload_hash
                 )
                 SELECT ?1, project_uuid, embedder_profile_version_id,
                    canonicalizer_version_id, canonicalizer_identity_json,
                    endpoint_identity_sha256, model, provider_revision, dimensions,
                    metric, normalization, canonical_space_json, created_at_unix_ms, ?1
                 FROM vector_spaces WHERE vector_space_id = ?2",
                params![alternate_id, proposed.vector_space_id.as_str()],
            )
            .unwrap();
        assert!(matches!(
            ensure_vector_registry_in_transaction(&transaction, &fixture.ensure).unwrap(),
            RegistryEnsureAck::Conflict
        ));
        assert_eq!(
            resolve_vector_space(
                &transaction,
                fixture.ensure.project_uuid,
                &proposed.vector_space_id,
            )
            .unwrap_err()
            .class(),
            LedgerErrorClass::CorruptDatabase
        );
    }

    #[test]
    fn vector_space_resolver_rejects_denormalized_tampering() {
        let mut fixture = fixture();
        let space_id = fixture.ensure.spaces.keys().next().unwrap().clone();
        let connection = fixture.activated.repository.connection_mut();
        assert!(
            resolve_vector_space(connection, fixture.ensure.project_uuid, &space_id)
                .unwrap()
                .is_some()
        );
        connection
            .execute(
                "UPDATE vector_spaces SET model = 'tampered-model'
                 WHERE vector_space_id = ?1",
                params![space_id.as_str()],
            )
            .unwrap();
        assert_eq!(
            resolve_vector_space(connection, fixture.ensure.project_uuid, &space_id)
                .unwrap_err()
                .class(),
            LedgerErrorClass::CorruptDatabase
        );
    }

    #[test]
    fn concurrent_activations_converge_and_a_live_new_generation_coexists() {
        let temporary = tempfile::tempdir().unwrap();
        let path = database_path(&temporary);
        let base = base_config(&path, "vector-registry-concurrent-fixture");
        drop(LedgerRepository::activate_at(&base, 500).unwrap());
        let config = learning_config(&path, "vector-registry-concurrent-fixture");
        let barrier = Arc::new(Barrier::new(3));
        let handles = [1_000_i64, 1_000_i64].map(|activated_at| {
            let config = config.clone();
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                barrier.wait();
                let mut busy_retries = 0;
                loop {
                    match LedgerRepository::activate_at(&config, activated_at) {
                        Ok(activated) => break Ok(activated),
                        Err(error)
                            if error.class() == LedgerErrorClass::Busy && busy_retries < 2 =>
                        {
                            busy_retries += 1;
                        }
                        Err(error) => break Err(error),
                    }
                }
            })
        });
        barrier.wait();
        let [first_handle, second_handle] = handles;
        let mut first = first_handle.join().unwrap().unwrap();
        let second = second_handle.join().unwrap().unwrap();
        assert_eq!(first.identity.project_uuid, second.identity.project_uuid);
        assert_eq!(
            first.identity.config_generation_id,
            second.identity.config_generation_id
        );
        assert_eq!(
            first.identity.pools["pool-a"].vector_space,
            second.identity.pools["pool-a"].vector_space
        );

        let first_ensure = prepare_for(&config, &first, 9_000);
        let first_key = mapping_key(&first_ensure);
        let connection = first.repository.connection_mut();
        assert_eq!(scalar(connection, "embedder_profiles"), 1);
        assert_eq!(scalar(connection, "vector_spaces"), 1);
        assert_eq!(scalar(connection, "vector_space_source_sequences"), 1);
        assert_eq!(scalar(connection, "pool_vector_space_mappings"), 1);
        let timestamps = connection
            .query_row(
                "SELECT
                    (SELECT created_at_unix_ms FROM embedder_profiles),
                    (SELECT created_at_unix_ms FROM vector_spaces),
                    (SELECT updated_at_unix_ms FROM vector_space_source_sequences),
                    (SELECT created_at_unix_ms FROM pool_vector_space_mappings)",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, i64>(3)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(timestamps.0, 1_000);
        assert_eq!(
            timestamps,
            (timestamps.0, timestamps.0, timestamps.0, timestamps.0)
        );
        let historical = resolve_frozen_mapping(connection, &first_key)
            .unwrap()
            .unwrap();
        assert_eq!(historical.created_at_unix_ms, timestamps.0);
        assert_eq!(historical.mapping, first_ensure.mappings["pool-a"]);

        let mut next_config = config.clone();
        next_config.retention_days += 1;
        let mut next = LedgerRepository::activate_at(&next_config, 3_000).unwrap();
        let next_ensure = prepare_for(&next_config, &next, 9_001);
        let next_key = mapping_key(&next_ensure);
        assert_ne!(
            first_ensure.config_generation_id,
            next_ensure.config_generation_id
        );
        assert_eq!(first_ensure.spaces, next_ensure.spaces);
        let connection = next.repository.connection_mut();
        assert_eq!(scalar(connection, "embedder_profiles"), 1);
        assert_eq!(scalar(connection, "vector_spaces"), 1);
        assert_eq!(scalar(connection, "vector_space_source_sequences"), 1);
        assert_eq!(scalar(connection, "pool_vector_space_mappings"), 2);
        let historical_after = resolve_frozen_mapping(connection, &first_key)
            .unwrap()
            .unwrap();
        let current = resolve_frozen_mapping(connection, &next_key)
            .unwrap()
            .unwrap();
        assert_eq!(historical_after, historical);
        assert_eq!(current.mapping, next_ensure.mappings["pool-a"]);
        assert_eq!(current.created_at_unix_ms, 3_000);
        assert_eq!(current.profile.created_at_unix_ms, timestamps.0);
        assert_eq!(current.space.created_at_unix_ms, timestamps.0);

        // Keep both original process connections alive through the cross-generation checks.
        assert_eq!(second.identity.project_uuid, next.identity.project_uuid);
        drop(temporary);
    }

    #[test]
    fn frozen_mappings_remain_resolvable_across_config_generations() {
        let Fixture {
            _temporary,
            path,
            config,
            activated,
            ensure: first,
        } = fixture();
        let first_key = mapping_key(&first);
        drop(activated);

        let mut second_config = config;
        second_config.retention_days += 1;
        let mut second = LedgerRepository::activate_at(&second_config, 3_000).unwrap();
        let second_ensure = prepare_for(&second_config, &second, 4_000);
        let second_key = mapping_key(&second_ensure);
        assert_ne!(
            first.config_generation_id,
            second_ensure.config_generation_id
        );
        assert_eq!(first.spaces, second_ensure.spaces);
        assert_eq!(
            scalar(second.repository.connection_mut(), "embedder_profiles"),
            1
        );
        assert_eq!(
            scalar(second.repository.connection_mut(), "vector_spaces"),
            1
        );
        assert_eq!(
            scalar(
                second.repository.connection_mut(),
                "pool_vector_space_mappings"
            ),
            2
        );
        let first_mapping = resolve_frozen_mapping(second.repository.connection_mut(), &first_key)
            .unwrap()
            .unwrap();
        let second_mapping =
            resolve_frozen_mapping(second.repository.connection_mut(), &second_key)
                .unwrap()
                .unwrap();
        assert_eq!(first_mapping.mapping, first.mappings["pool-a"]);
        assert_eq!(second_mapping.mapping, second_ensure.mappings["pool-a"]);
        assert_eq!(first_mapping.space, second_mapping.space);
        assert!(path.exists());
        drop(_temporary);
    }

    #[test]
    fn historical_enabled_pool_survives_current_disable_and_rejects_an_unexpected_mapping() {
        let Fixture {
            _temporary,
            path,
            mut config,
            activated,
            ensure,
        } = fixture();
        let historical_key = mapping_key(&ensure);
        drop(activated);

        config.pools[0].learning = None;
        let mut current = LedgerRepository::activate_at(&config, 3_000).unwrap();
        let current_key = identity_mapping_key(&current);
        assert_ne!(
            historical_key.config_generation_id,
            current_key.config_generation_id
        );
        assert!(matches!(
            resolve_frozen_pool_vector_authority(
                current.repository.connection_mut(),
                &historical_key,
            )
            .unwrap(),
            FrozenPoolVectorAuthority::Enabled(_)
        ));
        assert_eq!(
            resolve_frozen_pool_vector_authority(
                current.repository.connection_mut(),
                &current_key,
            )
            .unwrap(),
            FrozenPoolVectorAuthority::Disabled
        );
        assert_eq!(
            scalar(
                current.repository.connection_mut(),
                "pool_vector_space_mappings"
            ),
            1
        );

        let transaction = current.repository.connection_mut().transaction().unwrap();
        transaction
            .execute(
                "INSERT INTO pool_vector_space_mappings (
                    project_uuid, config_generation_id, pool_id, policy_version_id,
                    profile_id, embedder_profile_version_id, canonicalizer_version_id,
                    vector_space_id, canonical_mapping_json, created_at_unix_ms,
                    canonical_payload_hash
                 )
                 SELECT project_uuid, ?1, pool_id, ?2, profile_id,
                    embedder_profile_version_id, canonicalizer_version_id,
                    vector_space_id, canonical_mapping_json, 3_000,
                    canonical_payload_hash
                 FROM pool_vector_space_mappings
                 WHERE config_generation_id = ?3",
                params![
                    current_key.config_generation_id,
                    current_key.policy_version_id,
                    historical_key.config_generation_id,
                ],
            )
            .unwrap();
        assert_eq!(
            resolve_frozen_pool_vector_authority(&transaction, &current_key)
                .unwrap_err()
                .class(),
            LedgerErrorClass::CorruptDatabase
        );
        drop(transaction);
        assert!(path.exists());
        drop(_temporary);
    }

    fn artifact_path(path: &Path, suffix: &str) -> PathBuf {
        let mut value = path.as_os_str().to_os_string();
        value.push(suffix);
        value.into()
    }

    #[test]
    fn registry_never_persists_or_debugs_secret_material() {
        const SECRET_ENV_NAME: &str = "ROUTER_REGISTRY_SECRET_SENTINEL";
        const SECRET_VALUE: &str = "secret-value-that-must-never-reach-sqlite";

        let temporary = tempfile::tempdir().unwrap();
        let path = database_path(&temporary);
        let mut config = learning_config(&path, "vector-registry-secret-fixture");
        config.embedders[0].api_key_env = Some(SECRET_ENV_NAME.to_string());
        let mut activated = LedgerRepository::activate_at(&config, ACTIVATED_AT).unwrap();
        let ensure = prepare_for(&config, &activated, RETRY_AT);
        let debug = format!("{ensure:?}");
        assert!(!debug.contains(SECRET_ENV_NAME));
        assert!(!debug.contains(SECRET_VALUE));
        activated
            .repository
            .connection_mut()
            .execute_batch("PRAGMA wal_checkpoint(FULL);")
            .unwrap();
        for artifact in [
            path.clone(),
            artifact_path(&path, "-wal"),
            artifact_path(&path, "-shm"),
        ] {
            let Ok(bytes) = fs::read(&artifact) else {
                continue;
            };
            for forbidden in [SECRET_ENV_NAME.as_bytes(), SECRET_VALUE.as_bytes()] {
                assert!(
                    !bytes
                        .windows(forbidden.len())
                        .any(|window| window == forbidden),
                    "secret material reached {}",
                    artifact.display()
                );
            }
        }
    }

    fn profile_row(profile: &PreparedEmbedderProfile) -> ProfileRow {
        ProfileRow {
            version_id: profile.embedder_profile_version_id.clone(),
            profile_id: profile.profile_id.clone(),
            protocol: profile.protocol.to_string(),
            endpoint_url: profile.endpoint_url.clone(),
            endpoint_identity_sha256: profile.endpoint_identity_sha256.clone(),
            model: profile.model.clone(),
            provider_revision: profile.provider_revision.clone(),
            dimensions: i64::from(profile.dimensions.value()),
            credential_env_name_sha256: profile.credential_env_name_sha256.clone(),
            timeout_ms: i64::try_from(profile.timeout_ms).unwrap(),
            max_in_flight: i64::try_from(profile.max_in_flight).unwrap(),
            batch_size: i64::try_from(profile.batch_size).unwrap(),
            egress_class: profile.egress_class.as_str().to_string(),
            canonical_profile_json: profile.canonical_profile_json.clone(),
            created_at_unix_ms: ACTIVATED_AT,
            canonical_payload_hash: profile.embedder_profile_version_id.clone(),
        }
    }

    fn space_row(project_uuid: Uuid, space: &PreparedVectorSpace) -> SpaceRow {
        SpaceRow {
            vector_space_id: space.vector_space_id.as_str().to_string(),
            project_uuid: project_uuid.to_string(),
            profile_version_id: space.embedder_profile_version_id.clone(),
            canonicalizer_version_id: space.canonicalizer_version_id.clone(),
            canonicalizer_identity_json: space.canonicalizer_identity_json.clone(),
            endpoint_identity_sha256: space.endpoint_identity_sha256.clone(),
            model: space.model.clone(),
            provider_revision: space.provider_revision.clone(),
            dimensions: i64::from(space.dimensions.value()),
            metric: space.metric.to_string(),
            normalization: space.normalization.to_string(),
            canonical_space_json: space.canonical_space_json.clone(),
            created_at_unix_ms: ACTIVATED_AT,
            canonical_payload_hash: space.vector_space_id.as_str().to_string(),
        }
    }

    type RawMappingRow = (
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        String,
        i64,
        String,
    );

    fn mapping_row(mapping: &PreparedPoolVectorSpaceMapping) -> RawMappingRow {
        (
            mapping.project_uuid.to_string(),
            mapping.config_generation_id.clone(),
            mapping.pool_id.clone(),
            mapping.policy_version_id.clone(),
            mapping.profile_id.clone(),
            mapping.embedder_profile_version_id.clone(),
            mapping.canonicalizer_version_id.clone(),
            mapping.vector_space_id.as_str().to_string(),
            mapping.canonical_mapping_json.clone(),
            ACTIVATED_AT,
            mapping.canonical_payload_hash.clone(),
        )
    }

    #[test]
    fn full_row_verifiers_reject_every_column_mismatch() {
        let mut fixture = fixture();
        let profile = fixture.ensure.profiles.values().next().unwrap();
        let space = fixture.ensure.spaces.values().next().unwrap();
        let mapping = fixture.ensure.mappings.values().next().unwrap();
        assert!(verified_profile(profile_row(profile)).is_ok());
        assert!(
            verified_space(
                fixture.activated.repository.connection_mut(),
                fixture.ensure.project_uuid,
                space_row(fixture.ensure.project_uuid, space),
            )
            .is_ok()
        );

        macro_rules! profile_mismatch {
            ($field:ident, $value:expr) => {{
                let mut row = profile_row(profile);
                row.$field = $value;
                assert!(verified_profile(row).is_err(), stringify!($field));
            }};
        }
        profile_mismatch!(version_id, "0".repeat(64));
        profile_mismatch!(profile_id, "other-profile".to_string());
        profile_mismatch!(protocol, "other-protocol".to_string());
        profile_mismatch!(endpoint_url, "not-a-url".to_string());
        profile_mismatch!(endpoint_identity_sha256, "0".repeat(64));
        profile_mismatch!(model, "other-model".to_string());
        profile_mismatch!(provider_revision, "other-revision".to_string());
        profile_mismatch!(dimensions, 0);
        profile_mismatch!(credential_env_name_sha256, None);
        profile_mismatch!(timeout_ms, 0);
        profile_mismatch!(max_in_flight, 0);
        profile_mismatch!(batch_size, 0);
        profile_mismatch!(egress_class, "remote_https".to_string());
        profile_mismatch!(canonical_profile_json, "{}".to_string());
        profile_mismatch!(created_at_unix_ms, -1);
        profile_mismatch!(canonical_payload_hash, "0".repeat(64));

        macro_rules! space_mismatch {
            ($field:ident, $value:expr) => {{
                let mut row = space_row(fixture.ensure.project_uuid, space);
                row.$field = $value;
                assert!(
                    verified_space(
                        fixture.activated.repository.connection_mut(),
                        fixture.ensure.project_uuid,
                        row,
                    )
                    .is_err(),
                    stringify!($field)
                );
            }};
        }
        space_mismatch!(vector_space_id, "0".repeat(64));
        space_mismatch!(project_uuid, Uuid::nil().to_string());
        space_mismatch!(profile_version_id, "0".repeat(64));
        space_mismatch!(canonicalizer_version_id, "0".repeat(64));
        space_mismatch!(canonicalizer_identity_json, "{}".to_string());
        space_mismatch!(endpoint_identity_sha256, "0".repeat(64));
        space_mismatch!(model, "other-model".to_string());
        space_mismatch!(provider_revision, "other-revision".to_string());
        space_mismatch!(dimensions, 0);
        space_mismatch!(metric, "other-metric".to_string());
        space_mismatch!(normalization, "other-normalization".to_string());
        space_mismatch!(canonical_space_json, "{}".to_string());
        space_mismatch!(created_at_unix_ms, -1);
        space_mismatch!(canonical_payload_hash, "0".repeat(64));

        assert!(verified_mapping_row(mapping_row(mapping)).is_ok());
        macro_rules! mapping_mismatch {
            ($index:tt, $value:expr) => {{
                let mut row = mapping_row(mapping);
                row.$index = $value;
                assert!(verified_mapping_row(row).is_err(), stringify!($index));
            }};
        }
        mapping_mismatch!(0, Uuid::nil().to_string());
        mapping_mismatch!(1, "0".repeat(64));
        mapping_mismatch!(2, "other-pool".to_string());
        mapping_mismatch!(3, "0".repeat(64));
        mapping_mismatch!(4, "other-profile".to_string());
        mapping_mismatch!(5, "0".repeat(64));
        mapping_mismatch!(6, "0".repeat(64));
        mapping_mismatch!(7, "0".repeat(64));
        mapping_mismatch!(8, "{}".to_string());
        mapping_mismatch!(9, -1);
        mapping_mismatch!(10, "0".repeat(64));
    }
}
