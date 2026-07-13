// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Manifest authority and generated-schema validation for sqlite-vec indexes.

use std::collections::BTreeMap;

use rusqlite::{Connection, OptionalExtension, Transaction, params};
use serde_json::json;
use sha2::{Digest, Sha256};
use uuid::Uuid;

use super::materialization::{
    VerifiedVectorLinkSourceLoad, load_verified_vector_link_page, load_verified_vector_link_source,
};
use super::process::{ProcessStatusAt, verified_process_status_at};
use super::{is_sha256, map_sqlite_error, parse_uuid_v7};
use crate::canonical_json::canonical_sha256;
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::sqlite_vec_extension::{SqliteVecStatus, verify_connection as verify_sqlite_vec};
use crate::sqlite_vec_schema::{Vec0RootName, Vec0SchemaAuthority, VectorIndexGeneration};
use crate::vector::{
    AuthoritativeVector, PartitionId, VectorChecksum, VectorDimensions, VectorRecordId,
    VectorSpaceId,
};
use crate::vector_store::{SpaceHealth, SpaceHealthState, VectorRecord};

const SOURCE_FINGERPRINT_DOMAIN_V1: &[u8] = b"nemo.relay.router.vector-source-fingerprint@1\0";
const REBUILD_LEASE_MILLIS: i64 = 60_000;
pub(crate) const REBUILD_CHUNK_MAX: usize = 256;

/// Lifecycle state persisted for one immutable sqlite-vec generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VectorIndexManifestState {
    Building,
    Active,
    Unavailable,
    Corrupt,
    Retired,
    Dropped,
}

impl VectorIndexManifestState {
    fn parse(value: &str) -> Result<Self, LedgerError> {
        match value {
            "building" => Ok(Self::Building),
            "active" => Ok(Self::Active),
            "unavailable" => Ok(Self::Unavailable),
            "corrupt" => Ok(Self::Corrupt),
            "retired" => Ok(Self::Retired),
            "dropped" => Ok(Self::Dropped),
            _ => Err(corrupt()),
        }
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Building => "building",
            Self::Active => "active",
            Self::Unavailable => "unavailable",
            Self::Corrupt => "corrupt",
            Self::Retired => "retired",
            Self::Dropped => "dropped",
        }
    }
}

/// Fully validated mutable manifest row and its exact generated-schema authority.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ValidatedVectorIndexManifest {
    vector_space_id: VectorSpaceId,
    generation: VectorIndexGeneration,
    state: VectorIndexManifestState,
    authority: Vec0SchemaAuthority,
    base_source_seq: i64,
    applied_source_seq: i64,
    build_cursor_record_id: Option<VectorRecordId>,
    source_record_count: Option<u64>,
    source_fingerprint_sha256: Option<String>,
    stable_error_class: Option<String>,
    created_at_unix_ms: i64,
    activated_at_unix_ms: Option<i64>,
    retired_at_unix_ms: Option<i64>,
    dropped_at_unix_ms: Option<i64>,
    canonical_payload_hash: String,
}

impl ValidatedVectorIndexManifest {
    pub(crate) fn vector_space_id(&self) -> &VectorSpaceId {
        &self.vector_space_id
    }

    pub(crate) const fn generation(&self) -> VectorIndexGeneration {
        self.generation
    }

    pub(crate) const fn state(&self) -> VectorIndexManifestState {
        self.state
    }

    pub(crate) fn authority(&self) -> &Vec0SchemaAuthority {
        &self.authority
    }

    pub(crate) const fn base_source_seq(&self) -> i64 {
        self.base_source_seq
    }

    pub(crate) const fn applied_source_seq(&self) -> i64 {
        self.applied_source_seq
    }

    pub(crate) const fn build_cursor_record_id(&self) -> Option<VectorRecordId> {
        self.build_cursor_record_id
    }

    pub(crate) const fn source_record_count(&self) -> Option<u64> {
        self.source_record_count
    }

    pub(crate) fn source_fingerprint_sha256(&self) -> Option<&str> {
        self.source_fingerprint_sha256.as_deref()
    }

    pub(crate) fn stable_error_class(&self) -> Option<&str> {
        self.stable_error_class.as_deref()
    }

    pub(crate) fn canonical_payload_hash(&self) -> &str {
        &self.canonical_payload_hash
    }
}

/// Exact-object presence for one manifest-authorized generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GenerationObjectsStatus {
    Complete,
    Missing,
    Partial,
}

/// Vector-local result of resolving the current generation for search.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ActiveGenerationResolution {
    Missing,
    Unavailable,
    Corrupt,
    Active(Box<ValidatedVectorIndexManifest>),
}

/// Idempotent result of authorizing a new inactive generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GenerationAuthorizationAck {
    Created(ValidatedVectorIndexManifest),
    AlreadyExists(ValidatedVectorIndexManifest),
    AuthorityMissing,
}

/// Idempotent result of creating one manifest-authorized vec0 generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GenerationObjectCreationAck {
    Created,
    AlreadyExists,
    Partial,
    ManifestMissing,
    Conflict,
    Unavailable,
}

/// Exact point-mutation result without widening vector-local failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VectorIndexPointMutationAck {
    Applied,
    AlreadyApplied,
    Missing,
    Unavailable,
    Corrupt,
    Conflict,
}

/// Monotonic vector-local health target for one current generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VectorIndexHealthTarget {
    Unavailable,
    Corrupt,
}

/// Manifest-hash-fenced result of one vector-local health transition.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VectorIndexHealthMutationAck {
    Applied { manifest_hash: String },
    AlreadyApplied { manifest_hash: String },
    Missing,
    Stale,
    Conflict,
}

/// Manifest-hash-fenced result of retiring an already-unavailable generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GenerationRetirementAck {
    Applied,
    AlreadyApplied,
    Missing,
    Stale,
    Conflict,
}

/// Stable pure failures while constructing a source fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VectorSourceFingerprintError {
    OutOfOrder,
    InvalidChecksum,
    CountOverflow,
}

/// Exact source count and domain-separated SHA-256 fingerprint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VectorSourceFingerprint {
    record_count: u64,
    sha256: String,
}

impl VectorSourceFingerprint {
    pub(crate) const fn record_count(&self) -> u64 {
        self.record_count
    }

    pub(crate) fn sha256(&self) -> &str {
        &self.sha256
    }
}

/// Streaming fixed-width encoder for sorted authoritative source tuples.
pub(crate) struct VectorSourceFingerprintBuilder {
    digest: Sha256,
    previous_record_id: Option<VectorRecordId>,
    record_count: u64,
}

impl VectorSourceFingerprintBuilder {
    pub(crate) fn new() -> Self {
        let mut digest = Sha256::new();
        digest.update(SOURCE_FINGERPRINT_DOMAIN_V1);
        Self {
            digest,
            previous_record_id: None,
            record_count: 0,
        }
    }

    pub(crate) fn push(
        &mut self,
        record_id: VectorRecordId,
        partition_id: PartitionId,
        checksum: &VectorChecksum,
    ) -> Result<(), VectorSourceFingerprintError> {
        if self
            .previous_record_id
            .is_some_and(|previous| previous >= record_id)
        {
            return Err(VectorSourceFingerprintError::OutOfOrder);
        }
        let checksum = decode_checksum(checksum.as_str())?;
        self.record_count = self
            .record_count
            .checked_add(1)
            .ok_or(VectorSourceFingerprintError::CountOverflow)?;
        self.digest.update(record_id.value().as_bytes());
        self.digest.update(partition_id.value().to_be_bytes());
        self.digest.update(checksum);
        self.previous_record_id = Some(record_id);
        Ok(())
    }

    pub(crate) fn finish(self) -> VectorSourceFingerprint {
        VectorSourceFingerprint {
            record_count: self.record_count,
            sha256: self
                .digest
                .finalize()
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect(),
        }
    }
}

impl Default for VectorSourceFingerprintBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Current fenced ownership of one building generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RebuildLeaseFence {
    vector_space_id: VectorSpaceId,
    generation: VectorIndexGeneration,
    lease_generation: i64,
    owner_process_instance_id: Uuid,
    lease_token: Uuid,
    lease_expires_at_unix_ms: i64,
}

impl RebuildLeaseFence {
    pub(crate) fn vector_space_id(&self) -> &VectorSpaceId {
        &self.vector_space_id
    }

    pub(crate) const fn generation(&self) -> VectorIndexGeneration {
        self.generation
    }

    pub(crate) const fn lease_generation(&self) -> i64 {
        self.lease_generation
    }

    pub(crate) const fn owner_process_instance_id(&self) -> Uuid {
        self.owner_process_instance_id
    }

    pub(crate) const fn lease_token(&self) -> Uuid {
        self.lease_token
    }

    pub(crate) const fn lease_expires_at_unix_ms(&self) -> i64 {
        self.lease_expires_at_unix_ms
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RebuildLeaseClaimAck {
    Claimed(RebuildLeaseFence),
    Reclaimed(RebuildLeaseFence),
    AlreadyOwned(RebuildLeaseFence),
    Held { lease_expires_at_unix_ms: i64 },
    MissingGeneration,
    ClaimantNotLive,
    Stale,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RebuildLeaseMutationAck {
    Applied,
    Stale,
    OwnerNotLive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RebuildStepAck {
    Applied {
        processed: usize,
        complete: bool,
        applied_source_seq: i64,
    },
    Stale,
    Unavailable,
    Conflict,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RebuildFlipAck {
    Activated { record_count: u64 },
    NotReady,
    FingerprintMismatch,
    Stale,
    Unavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RetiredGenerationCleanupAck {
    Dropped,
    ReconciledMissing,
    AlreadyDropped,
    Partial,
    Conflict,
    Unavailable,
}

/// Durable operation recorded for one authoritative source-sequence change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum VectorSourceChangeOperation {
    Insert,
    Delete,
}

/// Bounded progress while removing source history for an otherwise dead space.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SourceHistoryRetentionAck {
    Drained,
    More { remaining_source_seq: i64 },
}

impl VectorSourceChangeOperation {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Insert => "insert",
            Self::Delete => "delete",
        }
    }

    fn parse(value: &str) -> Result<Self, LedgerError> {
        match value {
            "insert" => Ok(Self::Insert),
            "delete" => Ok(Self::Delete),
            _ => Err(corrupt()),
        }
    }
}

#[derive(Debug, Clone)]
struct StoredManifest {
    vector_space_id: String,
    generation: i64,
    state: String,
    root_table_name: String,
    dimensions: i64,
    expected_schema_objects_json: String,
    expected_schema_objects_sha256: String,
    base_source_seq: i64,
    applied_source_seq: i64,
    build_cursor_record_id: Option<String>,
    source_record_count: Option<i64>,
    source_fingerprint_sha256: Option<String>,
    stable_error_class: Option<String>,
    created_at_unix_ms: i64,
    activated_at_unix_ms: Option<i64>,
    retired_at_unix_ms: Option<i64>,
    dropped_at_unix_ms: Option<i64>,
    canonical_payload_hash: String,
}

#[derive(Debug, Clone)]
struct StoredRebuildLease {
    vector_space_id: String,
    generation: i64,
    lease_generation: i64,
    lease_owner_process_instance_id: String,
    lease_token: String,
    lease_expires_at_unix_ms: i64,
    base_source_seq: i64,
    applied_source_seq: i64,
    build_cursor_record_id: Option<String>,
    created_at_unix_ms: i64,
    updated_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug, Clone)]
struct StoredSourceSequence {
    vector_space_id: String,
    source_seq: i64,
    updated_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug, Clone)]
struct StoredSourceChange {
    vector_space_id: String,
    source_seq: i64,
    operation: String,
    record_id: String,
    partition_id: i64,
    vector_checksum: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

#[derive(Debug, Clone)]
struct ValidatedSourceChange {
    source_seq: i64,
    #[allow(dead_code)] // Parsing locks the persisted operation grammar for Task 7 producers.
    operation: VectorSourceChangeOperation,
    record_id: VectorRecordId,
    partition_id: PartitionId,
    vector_checksum: VectorChecksum,
    created_at_unix_ms: i64,
}

/// Load and fully validate one manifest row, including its mutable payload hash.
pub(crate) fn load_validated_manifest(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
    generation: VectorIndexGeneration,
) -> Result<Option<ValidatedVectorIndexManifest>, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT vector_space_id, generation, state, root_table_name, dimensions,
                    expected_schema_objects_json, expected_schema_objects_sha256,
                    base_source_seq, applied_source_seq, build_cursor_record_id,
                    source_record_count, source_fingerprint_sha256, stable_error_class,
                    created_at_unix_ms, activated_at_unix_ms, retired_at_unix_ms,
                    dropped_at_unix_ms, canonical_payload_hash
             FROM vector_index_manifest
             WHERE vector_space_id = ?1 AND generation = ?2",
            params![vector_space_id.as_str(), generation.value()],
            stored_manifest_from_row,
        )
        .optional()
        .map_err(database_error)?;
    stored
        .map(|stored| validate_manifest(connection, stored))
        .transpose()
}

/// Verify every generated object byte-for-byte without requiring vec_version().
pub(crate) fn verify_generation_objects(
    connection: &Connection,
    authority: &Vec0SchemaAuthority,
) -> Result<GenerationObjectsStatus, LedgerError> {
    let root = authority.root().as_str();
    let namespace_prefix = format!("{root}_");
    let expected = authority
        .objects()
        .iter()
        .map(|object| (object.name(), object))
        .collect::<BTreeMap<_, _>>();
    let mut actual = BTreeMap::new();
    let mut statement = connection
        .prepare(
            "SELECT type, name, tbl_name, sql
             FROM sqlite_schema
             WHERE type IN ('table', 'index', 'trigger', 'view')
               AND name NOT LIKE 'sqlite_%'",
        )
        .map_err(database_error)?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, Option<String>>(3)?,
            ))
        })
        .map_err(database_error)?;
    for row in rows {
        let (object_type, name, table_name, sql) = row.map_err(database_error)?;
        if name != root && !name.starts_with(&namespace_prefix) {
            continue;
        }
        let Some(expected_object) = expected.get(name.as_str()) else {
            return Err(corrupt());
        };
        if object_type != expected_object.object_type()
            || table_name != expected_object.table_name()
            || sql.as_deref() != Some(expected_object.sql())
            || crate::fingerprint::sha256_hex(expected_object.sql().as_bytes())
                != expected_object.sql_sha256()
            || actual.insert(name, ()).is_some()
        {
            return Err(corrupt());
        }
    }
    match actual.len() {
        0 => Ok(GenerationObjectsStatus::Missing),
        count if count == expected.len() => Ok(GenerationObjectsStatus::Complete),
        _ => Ok(GenerationObjectsStatus::Partial),
    }
}

/// Resolve one current generation and classify only vector-local unavailability.
pub(crate) fn resolve_active_generation(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
) -> Result<ActiveGenerationResolution, LedgerError> {
    let Some(manifest) = current_generation_manifest(connection, vector_space_id)? else {
        return Ok(ActiveGenerationResolution::Missing);
    };
    match manifest.state() {
        VectorIndexManifestState::Unavailable => {
            return Ok(ActiveGenerationResolution::Unavailable);
        }
        VectorIndexManifestState::Corrupt => {
            return Ok(ActiveGenerationResolution::Corrupt);
        }
        VectorIndexManifestState::Active => {}
        VectorIndexManifestState::Building
        | VectorIndexManifestState::Retired
        | VectorIndexManifestState::Dropped => return Err(corrupt()),
    }
    let objects_status = verify_generation_objects(connection, manifest.authority())?;
    if objects_status != GenerationObjectsStatus::Complete
        || verify_sqlite_vec(connection) != SqliteVecStatus::Available
    {
        return Ok(ActiveGenerationResolution::Unavailable);
    }
    Ok(ActiveGenerationResolution::Active(Box::new(manifest)))
}

/// Load the one exact current manifest without collapsing its persisted health state.
pub(crate) fn current_generation_manifest(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
) -> Result<Option<ValidatedVectorIndexManifest>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT generation
             FROM vector_index_manifest
             WHERE vector_space_id = ?1
               AND state IN ('active', 'unavailable', 'corrupt')
             ORDER BY generation",
        )
        .map_err(database_error)?;
    let generations = statement
        .query_map(params![vector_space_id.as_str()], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    if generations.is_empty() {
        return Ok(None);
    }
    if generations.len() != 1 {
        return Err(corrupt());
    }
    let generation = VectorIndexGeneration::new(generations[0]).map_err(|_| corrupt())?;
    let manifest =
        load_validated_manifest(connection, vector_space_id, generation)?.ok_or_else(corrupt)?;
    Ok(Some(manifest))
}

/// Persist manifest authority before any generated sqlite-vec object is created.
pub(crate) fn authorize_generation(
    transaction: &Transaction<'_>,
    vector_space_id: &VectorSpaceId,
    dimensions: VectorDimensions,
    created_at_unix_ms: i64,
) -> Result<GenerationAuthorizationAck, LedgerError> {
    if created_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let stored_dimensions = transaction
        .query_row(
            "SELECT dimensions FROM vector_spaces WHERE vector_space_id = ?1",
            params![vector_space_id.as_str()],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if stored_dimensions != i64::from(dimensions.value()) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let existing_generation = transaction
        .query_row(
            "SELECT generation FROM vector_index_manifest
             WHERE vector_space_id = ?1 AND state = 'building'",
            params![vector_space_id.as_str()],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?;
    if let Some(existing_generation) = existing_generation {
        let generation = VectorIndexGeneration::new(existing_generation).map_err(|_| corrupt())?;
        let manifest = load_validated_manifest(transaction, vector_space_id, generation)?
            .ok_or_else(corrupt)?;
        if manifest.authority().dimensions() != dimensions {
            return Err(corrupt());
        }
        return Ok(GenerationAuthorizationAck::AlreadyExists(manifest));
    }

    let has_live_authority = transaction
        .query_row(
            "SELECT
                EXISTS(SELECT 1 FROM pool_vector_space_mappings
                       WHERE vector_space_id = ?1)
                OR EXISTS(SELECT 1 FROM vectorization_outcomes
                          WHERE vector_space_id = ?1)
                OR EXISTS(SELECT 1 FROM embedding_jobs
                          WHERE vector_space_id = ?1)
                OR EXISTS(SELECT 1 FROM embeddings
                          WHERE vector_space_id = ?1)
                OR EXISTS(SELECT 1 FROM evidence_vector_links
                          WHERE vector_space_id = ?1)
                OR EXISTS(SELECT 1 FROM vector_materialization_jobs
                          WHERE vector_space_id = ?1)",
            params![vector_space_id.as_str()],
            |row| row.get::<_, bool>(0),
        )
        .map_err(database_error)?;
    let previous_generation = transaction
        .query_row(
            "SELECT max(generation) FROM vector_index_manifest WHERE vector_space_id = ?1",
            params![vector_space_id.as_str()],
            |row| row.get::<_, Option<i64>>(0),
        )
        .map_err(database_error)?
        .unwrap_or(0);
    if previous_generation > 0 && !has_live_authority {
        return Ok(GenerationAuthorizationAck::AuthorityMissing);
    }
    let next_generation = previous_generation
        .checked_add(1)
        .and_then(|value| VectorIndexGeneration::new(value).ok())
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let base_source_seq = load_validated_source_sequence(transaction, vector_space_id)?.source_seq;

    let authority = Vec0SchemaAuthority::new(
        Vec0RootName::new(vector_space_id.clone(), next_generation),
        dimensions,
    );
    let stored = StoredManifest {
        vector_space_id: vector_space_id.as_str().to_string(),
        generation: next_generation.value(),
        state: VectorIndexManifestState::Building.as_str().to_string(),
        root_table_name: authority.root().as_str().to_string(),
        dimensions: i64::from(dimensions.value()),
        expected_schema_objects_json: authority.manifest_json().to_string(),
        expected_schema_objects_sha256: authority.manifest_sha256().to_string(),
        base_source_seq,
        applied_source_seq: base_source_seq,
        build_cursor_record_id: None,
        source_record_count: None,
        source_fingerprint_sha256: None,
        stable_error_class: None,
        created_at_unix_ms,
        activated_at_unix_ms: None,
        retired_at_unix_ms: None,
        dropped_at_unix_ms: None,
        canonical_payload_hash: String::new(),
    };
    let canonical_payload_hash = manifest_payload_hash(&stored)?;
    transaction
        .execute(
            "INSERT INTO vector_index_manifest (
                vector_space_id, generation, state, root_table_name, dimensions,
                expected_schema_objects_json, expected_schema_objects_sha256,
                base_source_seq, applied_source_seq, build_cursor_record_id,
                source_record_count, source_fingerprint_sha256, stable_error_class,
                created_at_unix_ms, activated_at_unix_ms, retired_at_unix_ms,
                dropped_at_unix_ms, canonical_payload_hash
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, NULL, NULL, NULL, NULL,
                ?10, NULL, NULL, NULL, ?11
             )",
            params![
                stored.vector_space_id,
                stored.generation,
                stored.state,
                stored.root_table_name,
                stored.dimensions,
                stored.expected_schema_objects_json,
                stored.expected_schema_objects_sha256,
                stored.base_source_seq,
                stored.applied_source_seq,
                stored.created_at_unix_ms,
                canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    let manifest = load_validated_manifest(transaction, vector_space_id, next_generation)?
        .ok_or_else(corrupt)?;
    Ok(GenerationAuthorizationAck::Created(manifest))
}

/// Create all generated objects only after their canonical manifest is durable.
pub(crate) fn create_generation_objects(
    transaction: &Transaction<'_>,
    fence: &RebuildLeaseFence,
    observed_at_unix_ms: i64,
) -> Result<GenerationObjectCreationAck, LedgerError> {
    if !validate_live_fence(transaction, fence, observed_at_unix_ms)? {
        return Ok(GenerationObjectCreationAck::Conflict);
    }
    create_generation_objects_for_manifest(transaction, fence.vector_space_id(), fence.generation())
}

fn create_generation_objects_for_manifest(
    transaction: &Transaction<'_>,
    vector_space_id: &VectorSpaceId,
    generation: VectorIndexGeneration,
) -> Result<GenerationObjectCreationAck, LedgerError> {
    let Some(manifest) = load_validated_manifest(transaction, vector_space_id, generation)? else {
        return Ok(GenerationObjectCreationAck::ManifestMissing);
    };
    if manifest.state() != VectorIndexManifestState::Building {
        return Ok(GenerationObjectCreationAck::Conflict);
    }
    match verify_generation_objects(transaction, manifest.authority())? {
        GenerationObjectsStatus::Complete => {
            if verify_sqlite_vec(transaction) == SqliteVecStatus::Available {
                Ok(GenerationObjectCreationAck::AlreadyExists)
            } else {
                Ok(GenerationObjectCreationAck::Unavailable)
            }
        }
        GenerationObjectsStatus::Partial => Ok(GenerationObjectCreationAck::Partial),
        GenerationObjectsStatus::Missing => {
            if verify_sqlite_vec(transaction) != SqliteVecStatus::Available {
                return Ok(GenerationObjectCreationAck::Unavailable);
            }
            transaction
                .execute_batch(manifest.authority().create_sql())
                .map_err(database_error)?;
            if verify_generation_objects(transaction, manifest.authority())?
                != GenerationObjectsStatus::Complete
            {
                return Err(corrupt());
            }
            Ok(GenerationObjectCreationAck::Created)
        }
    }
}

/// Insert one immutable active-generation point or prove exact idempotency.
pub(crate) fn upsert_active_record(
    transaction: &Transaction<'_>,
    record: &VectorRecord,
) -> Result<VectorIndexPointMutationAck, LedgerError> {
    let manifest = match resolve_active_generation(transaction, record.vector_space_id())? {
        ActiveGenerationResolution::Missing => {
            return Ok(VectorIndexPointMutationAck::Missing);
        }
        ActiveGenerationResolution::Unavailable => {
            return Ok(VectorIndexPointMutationAck::Unavailable);
        }
        ActiveGenerationResolution::Corrupt => {
            return Ok(VectorIndexPointMutationAck::Corrupt);
        }
        ActiveGenerationResolution::Active(manifest) => manifest,
    };
    if record.vector().vector().dimensions() != manifest.authority().dimensions() {
        return Ok(VectorIndexPointMutationAck::Conflict);
    }

    let root = manifest.authority().root().as_str();
    let select_sql = format!("SELECT partition_id, embedding FROM \"{root}\" WHERE record_id = ?1");
    let existing = transaction
        .query_row(
            &select_sql,
            params![record.record_id().to_string()],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    let expected_bytes = record.vector().blob().native_endian_bytes();
    if let Some((partition_id, vector_bytes)) = existing {
        return Ok(
            if partition_id == record.partition_id().value() && vector_bytes == expected_bytes {
                VectorIndexPointMutationAck::AlreadyApplied
            } else {
                VectorIndexPointMutationAck::Conflict
            },
        );
    }

    let insert_sql =
        format!("INSERT INTO \"{root}\" (record_id, embedding, partition_id) VALUES (?1, ?2, ?3)");
    transaction
        .execute(
            &insert_sql,
            params![
                record.record_id().to_string(),
                expected_bytes,
                record.partition_id().value(),
            ],
        )
        .map_err(database_error)?;
    Ok(VectorIndexPointMutationAck::Applied)
}

/// Delete one point from the resolved active generation without touching relations.
pub(crate) fn delete_active_record(
    transaction: &Transaction<'_>,
    vector_space_id: &VectorSpaceId,
    record_id: VectorRecordId,
) -> Result<VectorIndexPointMutationAck, LedgerError> {
    let manifest = match resolve_active_generation(transaction, vector_space_id)? {
        ActiveGenerationResolution::Missing => {
            return Ok(VectorIndexPointMutationAck::Missing);
        }
        ActiveGenerationResolution::Unavailable => {
            return Ok(VectorIndexPointMutationAck::Unavailable);
        }
        ActiveGenerationResolution::Corrupt => {
            return Ok(VectorIndexPointMutationAck::Corrupt);
        }
        ActiveGenerationResolution::Active(manifest) => manifest,
    };
    let root = manifest.authority().root().as_str();
    let exists_sql = format!("SELECT 1 FROM \"{root}\" WHERE record_id = ?1");
    let exists = transaction
        .query_row(&exists_sql, params![record_id.to_string()], |_| Ok(()))
        .optional()
        .map_err(database_error)?
        .is_some();
    if !exists {
        return Ok(VectorIndexPointMutationAck::AlreadyApplied);
    }
    let delete_sql = format!("DELETE FROM \"{root}\" WHERE record_id = ?1");
    transaction
        .execute(&delete_sql, params![record_id.to_string()])
        .map_err(database_error)?;
    Ok(VectorIndexPointMutationAck::Applied)
}

/// Count rows in one already-validated generation.
pub(crate) fn generation_record_count(
    connection: &Connection,
    manifest: &ValidatedVectorIndexManifest,
) -> Result<u64, LedgerError> {
    if verify_generation_objects(connection, manifest.authority())?
        != GenerationObjectsStatus::Complete
        || verify_sqlite_vec(connection) != SqliteVecStatus::Available
    {
        return Err(LedgerErrorClass::DatabaseOperationFailed.into());
    }
    let root = manifest.authority().root().as_str();
    let sql = format!("SELECT count(*) FROM \"{root}\"");
    let count = connection
        .query_row(&sql, [], |row| row.get::<_, i64>(0))
        .map_err(database_error)?;
    u64::try_from(count).map_err(|_| corrupt())
}

/// Return bounded vector-local health and exact generated-row count.
pub(crate) fn vector_index_health(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
) -> Result<SpaceHealth, LedgerError> {
    match resolve_active_generation(connection, vector_space_id)? {
        ActiveGenerationResolution::Missing => Ok(SpaceHealth::new(SpaceHealthState::Missing, 0)),
        ActiveGenerationResolution::Unavailable => {
            Ok(SpaceHealth::new(SpaceHealthState::Unavailable, 0))
        }
        ActiveGenerationResolution::Corrupt => Ok(SpaceHealth::new(SpaceHealthState::Corrupt, 0)),
        ActiveGenerationResolution::Active(manifest) => Ok(SpaceHealth::new(
            SpaceHealthState::Healthy,
            generation_record_count(connection, &manifest)?,
        )),
    }
}

/// Mark only the exact current generation unavailable or corrupt.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mark_vector_index_health(
    transaction: &Transaction<'_>,
    vector_space_id: &VectorSpaceId,
    expected_generation: VectorIndexGeneration,
    expected_manifest_hash: &str,
    target: VectorIndexHealthTarget,
    stable_error_class: &str,
    observed_at_unix_ms: i64,
) -> Result<VectorIndexHealthMutationAck, LedgerError> {
    if !is_sha256(expected_manifest_hash)
        || !is_stable_error_class(stable_error_class)
        || observed_at_unix_ms < 0
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let current_generations = transaction
        .prepare(
            "SELECT generation FROM vector_index_manifest
             WHERE vector_space_id = ?1
               AND state IN ('active', 'unavailable', 'corrupt')
             ORDER BY generation",
        )
        .map_err(database_error)?
        .query_map(params![vector_space_id.as_str()], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    let current_generation = match current_generations.as_slice() {
        [] => {
            return match load_stored_manifest(transaction, vector_space_id, expected_generation)? {
                Some(stored) => {
                    validate_manifest(transaction, stored)?;
                    Ok(VectorIndexHealthMutationAck::Conflict)
                }
                None => Ok(VectorIndexHealthMutationAck::Missing),
            };
        }
        [generation] => VectorIndexGeneration::new(*generation).map_err(|_| corrupt())?,
        _ => return Err(corrupt()),
    };
    if current_generation != expected_generation {
        return Ok(VectorIndexHealthMutationAck::Stale);
    }

    let mut stored = load_stored_manifest(transaction, vector_space_id, expected_generation)?
        .ok_or_else(corrupt)?;
    let manifest = validate_manifest(transaction, stored.clone())?;
    if manifest.canonical_payload_hash() != expected_manifest_hash {
        return Ok(VectorIndexHealthMutationAck::Stale);
    }
    verify_generation_objects(transaction, manifest.authority())?;
    if stored
        .activated_at_unix_ms
        .is_none_or(|activated| observed_at_unix_ms < activated)
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let transition = match (manifest.state(), target) {
        (VectorIndexManifestState::Active, VectorIndexHealthTarget::Unavailable) => {
            VectorIndexManifestState::Unavailable
        }
        (VectorIndexManifestState::Active, VectorIndexHealthTarget::Corrupt)
        | (VectorIndexManifestState::Unavailable, VectorIndexHealthTarget::Corrupt) => {
            VectorIndexManifestState::Corrupt
        }
        (VectorIndexManifestState::Unavailable, VectorIndexHealthTarget::Unavailable)
        | (VectorIndexManifestState::Corrupt, VectorIndexHealthTarget::Corrupt) => {
            return Ok(
                if manifest.stable_error_class() == Some(stable_error_class) {
                    VectorIndexHealthMutationAck::AlreadyApplied {
                        manifest_hash: manifest.canonical_payload_hash().to_string(),
                    }
                } else {
                    VectorIndexHealthMutationAck::Conflict
                },
            );
        }
        (VectorIndexManifestState::Corrupt, VectorIndexHealthTarget::Unavailable) => {
            return Ok(VectorIndexHealthMutationAck::Conflict);
        }
        (VectorIndexManifestState::Building, _)
        | (VectorIndexManifestState::Retired, _)
        | (VectorIndexManifestState::Dropped, _) => return Err(corrupt()),
    };

    stored.state = transition.as_str().to_string();
    stored.stable_error_class = Some(stable_error_class.to_string());
    stored.canonical_payload_hash = manifest_payload_hash(&stored)?;
    validate_manifest(transaction, stored.clone())?;
    if update_manifest(transaction, &stored, expected_manifest_hash)? != 1 {
        return Ok(VectorIndexHealthMutationAck::Stale);
    }
    Ok(VectorIndexHealthMutationAck::Applied {
        manifest_hash: stored.canonical_payload_hash,
    })
}

/// Retire an exact healthy or unavailable current generation after its caller proves
/// that no retained graph, mapping, or rebuild authority still needs the space.
pub(crate) fn retire_current_generation_for_retention(
    transaction: &Transaction<'_>,
    vector_space_id: &VectorSpaceId,
    expected_generation: VectorIndexGeneration,
    expected_manifest_hash: &str,
    retired_at_unix_ms: i64,
) -> Result<GenerationRetirementAck, LedgerError> {
    if !is_sha256(expected_manifest_hash) || retired_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let Some(mut stored) = load_stored_manifest(transaction, vector_space_id, expected_generation)?
    else {
        return Ok(GenerationRetirementAck::Missing);
    };
    let manifest = validate_manifest(transaction, stored.clone())?;
    if manifest.state() == VectorIndexManifestState::Retired {
        return Ok(if stored.retired_at_unix_ms == Some(retired_at_unix_ms) {
            GenerationRetirementAck::AlreadyApplied
        } else {
            GenerationRetirementAck::Conflict
        });
    }
    if !matches!(
        manifest.state(),
        VectorIndexManifestState::Active | VectorIndexManifestState::Unavailable
    ) {
        return Ok(GenerationRetirementAck::Conflict);
    }
    if manifest.canonical_payload_hash() != expected_manifest_hash {
        return Ok(GenerationRetirementAck::Stale);
    }
    if manifest
        .activated_at_unix_ms
        .is_none_or(|activated| retired_at_unix_ms < activated)
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    if verify_generation_objects(transaction, manifest.authority())?
        == GenerationObjectsStatus::Partial
    {
        return Ok(GenerationRetirementAck::Conflict);
    }
    stored.state = VectorIndexManifestState::Retired.as_str().to_string();
    stored.retired_at_unix_ms = Some(retired_at_unix_ms);
    stored.canonical_payload_hash = manifest_payload_hash(&stored)?;
    if update_manifest(transaction, &stored, expected_manifest_hash)? != 1 {
        return Ok(GenerationRetirementAck::Stale);
    }
    Ok(GenerationRetirementAck::Applied)
}

/// Verify and prune one bounded tail page after all live space authority is gone.
pub(crate) fn prune_vector_source_history_for_retention(
    transaction: &Transaction<'_>,
    vector_space_id: &VectorSpaceId,
) -> Result<SourceHistoryRetentionAck, LedgerError> {
    let sequence = load_validated_source_sequence(transaction, vector_space_id)?;
    if sequence.source_seq == 0 {
        return Ok(SourceHistoryRetentionAck::Drained);
    }
    let retained_source_seq = sequence
        .source_seq
        .saturating_sub(i64::try_from(REBUILD_CHUNK_MAX).expect("chunk bound fits i64"))
        .max(0);
    let changes = load_validated_source_changes(
        transaction,
        vector_space_id,
        retained_source_seq,
        sequence.source_seq,
    )?;
    let expected_count = usize::try_from(sequence.source_seq - retained_source_seq)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if changes.len() != expected_count {
        return Err(corrupt());
    }
    let deleted = transaction
        .execute(
            "DELETE FROM vector_source_change_events
             WHERE vector_space_id = ?1 AND source_seq > ?2 AND source_seq <= ?3",
            params![
                vector_space_id.as_str(),
                retained_source_seq,
                sequence.source_seq,
            ],
        )
        .map_err(database_error)?;
    if deleted != expected_count {
        return Err(corrupt());
    }
    let new_hash = vector_source_sequence_payload_hash(
        vector_space_id,
        retained_source_seq,
        sequence.updated_at_unix_ms,
    )?;
    let updated = transaction
        .execute(
            "UPDATE vector_space_source_sequences
             SET source_seq = ?1, canonical_payload_hash = ?2
             WHERE vector_space_id = ?3 AND source_seq = ?4
               AND canonical_payload_hash = ?5",
            params![
                retained_source_seq,
                new_hash,
                vector_space_id.as_str(),
                sequence.source_seq,
                sequence.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if updated != 1 {
        return Err(corrupt());
    }
    Ok(if retained_source_seq == 0 {
        SourceHistoryRetentionAck::Drained
    } else {
        SourceHistoryRetentionAck::More {
            remaining_source_seq: retained_source_seq,
        }
    })
}

/// Claim one building generation, fencing every expired or dead prior owner.
pub(crate) fn claim_rebuild_lease(
    transaction: &Transaction<'_>,
    vector_space_id: &VectorSpaceId,
    owner_process_instance_id: Uuid,
    observed_at_unix_ms: i64,
) -> Result<RebuildLeaseClaimAck, LedgerError> {
    if observed_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let project_uuid = project_uuid_for_space(transaction, vector_space_id)?;
    if verified_process_status_at(
        transaction,
        project_uuid,
        owner_process_instance_id,
        observed_at_unix_ms,
    )? != ProcessStatusAt::Live
    {
        return Ok(RebuildLeaseClaimAck::ClaimantNotLive);
    }
    let Some(manifest) = load_building_manifest(transaction, vector_space_id)? else {
        return Ok(RebuildLeaseClaimAck::MissingGeneration);
    };
    let lease_expires_at_unix_ms = observed_at_unix_ms
        .checked_add(REBUILD_LEASE_MILLIS)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;

    if let Some(mut stored) = load_stored_rebuild_lease(transaction, vector_space_id)? {
        let existing = validate_rebuild_lease(transaction, &stored)?;
        if existing.owner_process_instance_id() == owner_process_instance_id
            && existing.lease_expires_at_unix_ms() > observed_at_unix_ms
        {
            return Ok(RebuildLeaseClaimAck::AlreadyOwned(existing));
        }
        let existing_owner_status = verified_process_status_at(
            transaction,
            project_uuid,
            existing.owner_process_instance_id(),
            observed_at_unix_ms,
        )?;
        if existing.lease_expires_at_unix_ms() > observed_at_unix_ms
            && existing_owner_status == ProcessStatusAt::Live
        {
            return Ok(RebuildLeaseClaimAck::Held {
                lease_expires_at_unix_ms: existing.lease_expires_at_unix_ms(),
            });
        }
        if existing_owner_status == ProcessStatusAt::Invalid {
            return Err(corrupt());
        }
        let previous_hash = stored.canonical_payload_hash.clone();
        stored.lease_generation = stored
            .lease_generation
            .checked_add(1)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        stored.lease_owner_process_instance_id = owner_process_instance_id.to_string();
        stored.lease_token = Uuid::now_v7().to_string();
        stored.lease_expires_at_unix_ms = lease_expires_at_unix_ms;
        stored.base_source_seq = manifest.base_source_seq();
        stored.applied_source_seq = manifest.applied_source_seq();
        stored.build_cursor_record_id = manifest
            .build_cursor_record_id()
            .map(|record_id| record_id.to_string());
        stored.updated_at_unix_ms = observed_at_unix_ms;
        stored.canonical_payload_hash = rebuild_lease_payload_hash(&stored)?;
        if update_rebuild_lease(transaction, &stored, &previous_hash)? != 1 {
            return Ok(RebuildLeaseClaimAck::Stale);
        }
        return Ok(RebuildLeaseClaimAck::Reclaimed(validate_rebuild_lease(
            transaction,
            &stored,
        )?));
    }

    let mut stored = StoredRebuildLease {
        vector_space_id: vector_space_id.as_str().to_string(),
        generation: manifest.generation().value(),
        lease_generation: 1,
        lease_owner_process_instance_id: owner_process_instance_id.to_string(),
        lease_token: Uuid::now_v7().to_string(),
        lease_expires_at_unix_ms,
        base_source_seq: manifest.base_source_seq(),
        applied_source_seq: manifest.applied_source_seq(),
        build_cursor_record_id: manifest
            .build_cursor_record_id()
            .map(|record_id| record_id.to_string()),
        created_at_unix_ms: observed_at_unix_ms,
        updated_at_unix_ms: observed_at_unix_ms,
        canonical_payload_hash: String::new(),
    };
    stored.canonical_payload_hash = rebuild_lease_payload_hash(&stored)?;
    transaction
        .execute(
            "INSERT INTO vector_index_rebuild_leases (
                vector_space_id, generation, lease_generation,
                lease_owner_process_instance_id, lease_token,
                lease_expires_at_unix_ms, base_source_seq, applied_source_seq,
                build_cursor_record_id, created_at_unix_ms, updated_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                stored.vector_space_id,
                stored.generation,
                stored.lease_generation,
                stored.lease_owner_process_instance_id,
                stored.lease_token,
                stored.lease_expires_at_unix_ms,
                stored.base_source_seq,
                stored.applied_source_seq,
                stored.build_cursor_record_id,
                stored.created_at_unix_ms,
                stored.updated_at_unix_ms,
                stored.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    Ok(RebuildLeaseClaimAck::Claimed(validate_rebuild_lease(
        transaction,
        &stored,
    )?))
}

/// Renew one current fence without changing its generation or token.
pub(crate) fn renew_rebuild_lease(
    transaction: &Transaction<'_>,
    fence: &RebuildLeaseFence,
    observed_at_unix_ms: i64,
) -> Result<RebuildLeaseMutationAck, LedgerError> {
    if observed_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let Some(mut stored) = load_stored_rebuild_lease(transaction, fence.vector_space_id())? else {
        return Ok(RebuildLeaseMutationAck::Stale);
    };
    let current = validate_rebuild_lease(transaction, &stored)?;
    if !fence_matches(fence, &current)
        || current.lease_expires_at_unix_ms() <= observed_at_unix_ms
        || observed_at_unix_ms < stored.updated_at_unix_ms
    {
        return Ok(RebuildLeaseMutationAck::Stale);
    }
    let project_uuid = project_uuid_for_space(transaction, fence.vector_space_id())?;
    if verified_process_status_at(
        transaction,
        project_uuid,
        fence.owner_process_instance_id(),
        observed_at_unix_ms,
    )? != ProcessStatusAt::Live
    {
        return Ok(RebuildLeaseMutationAck::OwnerNotLive);
    }
    let previous_hash = stored.canonical_payload_hash.clone();
    let proposed_expiry = observed_at_unix_ms
        .checked_add(REBUILD_LEASE_MILLIS)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    stored.lease_expires_at_unix_ms = stored.lease_expires_at_unix_ms.max(proposed_expiry);
    stored.updated_at_unix_ms = observed_at_unix_ms;
    stored.canonical_payload_hash = rebuild_lease_payload_hash(&stored)?;
    if update_rebuild_lease(transaction, &stored, &previous_hash)? != 1 {
        return Ok(RebuildLeaseMutationAck::Stale);
    }
    Ok(RebuildLeaseMutationAck::Applied)
}

/// Release one exact fence; a newer owner can never be removed.
pub(crate) fn release_rebuild_lease(
    transaction: &Transaction<'_>,
    fence: &RebuildLeaseFence,
    released_at_unix_ms: i64,
) -> Result<RebuildLeaseMutationAck, LedgerError> {
    if released_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let Some(mut stored) = load_stored_rebuild_lease(transaction, fence.vector_space_id())? else {
        return Ok(RebuildLeaseMutationAck::Stale);
    };
    let current = validate_rebuild_lease(transaction, &stored)?;
    if !fence_matches(fence, &current) {
        return Ok(RebuildLeaseMutationAck::Stale);
    }
    if released_at_unix_ms < stored.updated_at_unix_ms {
        return Ok(RebuildLeaseMutationAck::Stale);
    }
    let previous_hash = stored.canonical_payload_hash.clone();
    stored.lease_expires_at_unix_ms = released_at_unix_ms;
    stored.updated_at_unix_ms = released_at_unix_ms;
    stored.canonical_payload_hash = rebuild_lease_payload_hash(&stored)?;
    Ok(
        if update_rebuild_lease(transaction, &stored, &previous_hash)? == 1 {
            RebuildLeaseMutationAck::Applied
        } else {
            RebuildLeaseMutationAck::Stale
        },
    )
}

/// Stream the authoritative ready-link tuple set in deterministic record order.
pub(crate) fn authoritative_source_fingerprint(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
) -> Result<VectorSourceFingerprint, LedgerError> {
    let mut builder = VectorSourceFingerprintBuilder::new();
    let mut cursor = None;
    loop {
        let page =
            load_verified_vector_link_page(connection, vector_space_id, cursor, REBUILD_CHUNK_MAX)?;
        for record in page.ready_records {
            builder
                .push(
                    record.record_id(),
                    record.partition_id(),
                    record.vector().blob().checksum(),
                )
                .map_err(|_| corrupt())?;
        }
        if page.exhausted {
            break;
        }
        cursor = page.last_scanned_record_id;
        if cursor.is_none() {
            return Err(corrupt());
        }
    }
    Ok(builder.finish())
}

/// Populate one bounded deterministic chunk from the current relational source.
pub(crate) fn populate_rebuild_chunk(
    transaction: &Transaction<'_>,
    fence: &RebuildLeaseFence,
    observed_at_unix_ms: i64,
) -> Result<RebuildStepAck, LedgerError> {
    if !validate_live_fence(transaction, fence, observed_at_unix_ms)? {
        return Ok(RebuildStepAck::Stale);
    }
    let manifest =
        load_validated_manifest(transaction, fence.vector_space_id(), fence.generation())?
            .ok_or_else(corrupt)?;
    if !generation_is_available(transaction, &manifest)? {
        return Ok(RebuildStepAck::Unavailable);
    }
    let page = load_verified_vector_link_page(
        transaction,
        fence.vector_space_id(),
        manifest.build_cursor_record_id(),
        REBUILD_CHUNK_MAX,
    )?;
    for record in &page.ready_records {
        match upsert_generation_record(transaction, &manifest, record)? {
            VectorIndexPointMutationAck::Applied | VectorIndexPointMutationAck::AlreadyApplied => {}
            VectorIndexPointMutationAck::Conflict
            | VectorIndexPointMutationAck::Missing
            | VectorIndexPointMutationAck::Corrupt => return Ok(RebuildStepAck::Conflict),
            VectorIndexPointMutationAck::Unavailable => {
                return Ok(RebuildStepAck::Unavailable);
            }
        }
    }
    let processed = page.ready_records.len();
    let next_cursor = page
        .last_scanned_record_id
        .or(manifest.build_cursor_record_id());
    if next_cursor != manifest.build_cursor_record_id() {
        update_build_progress(
            transaction,
            fence,
            Some(next_cursor),
            None,
            observed_at_unix_ms,
        )?;
    }
    Ok(RebuildStepAck::Applied {
        processed,
        complete: page.exhausted,
        applied_source_seq: manifest.applied_source_seq(),
    })
}

/// Apply one bounded contiguous source-change chunk to the inactive generation.
pub(crate) fn catch_up_rebuild_changes(
    transaction: &Transaction<'_>,
    fence: &RebuildLeaseFence,
    observed_at_unix_ms: i64,
) -> Result<RebuildStepAck, LedgerError> {
    if !validate_live_fence(transaction, fence, observed_at_unix_ms)? {
        return Ok(RebuildStepAck::Stale);
    }
    let manifest =
        load_validated_manifest(transaction, fence.vector_space_id(), fence.generation())?
            .ok_or_else(corrupt)?;
    if !generation_is_available(transaction, &manifest)? {
        return Ok(RebuildStepAck::Unavailable);
    }
    let source_sequence = load_validated_source_sequence(transaction, fence.vector_space_id())?;
    let current_source_seq = source_sequence.source_seq;
    if manifest.applied_source_seq() > current_source_seq {
        return Err(corrupt());
    }
    if manifest.applied_source_seq() == current_source_seq {
        return Ok(RebuildStepAck::Applied {
            processed: 0,
            complete: true,
            applied_source_seq: current_source_seq,
        });
    }

    let changes = load_validated_source_changes(
        transaction,
        fence.vector_space_id(),
        manifest.applied_source_seq(),
        current_source_seq,
    )?;
    if changes.is_empty() {
        return Err(corrupt());
    }
    let mut expected_seq = manifest
        .applied_source_seq()
        .checked_add(1)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let mut applied_source_seq = manifest.applied_source_seq();
    for change in &changes {
        if change.source_seq != expected_seq {
            return Err(corrupt());
        }
        if change.created_at_unix_ms > source_sequence.updated_at_unix_ms {
            return Err(corrupt());
        }
        let current =
            load_ready_source_record(transaction, fence.vector_space_id(), change.record_id)?;
        let mutation = if let Some(record) = current {
            if record.partition_id() != change.partition_id
                || record.vector().blob().checksum() != &change.vector_checksum
            {
                return Err(corrupt());
            }
            upsert_generation_record(transaction, &manifest, &record)?
        } else {
            delete_generation_record(transaction, &manifest, change.record_id)?
        };
        match mutation {
            VectorIndexPointMutationAck::Applied | VectorIndexPointMutationAck::AlreadyApplied => {}
            VectorIndexPointMutationAck::Conflict
            | VectorIndexPointMutationAck::Missing
            | VectorIndexPointMutationAck::Corrupt => return Ok(RebuildStepAck::Conflict),
            VectorIndexPointMutationAck::Unavailable => {
                return Ok(RebuildStepAck::Unavailable);
            }
        }
        applied_source_seq = change.source_seq;
        expected_seq = expected_seq
            .checked_add(1)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    }
    update_build_progress(
        transaction,
        fence,
        None,
        Some(applied_source_seq),
        observed_at_unix_ms,
    )?;
    Ok(RebuildStepAck::Applied {
        processed: changes.len(),
        complete: applied_source_seq == current_source_seq,
        applied_source_seq,
    })
}

/// Verify exact source equality and atomically publish one immutable generation.
pub(crate) fn flip_rebuild_generation(
    transaction: &Transaction<'_>,
    fence: &RebuildLeaseFence,
    activated_at_unix_ms: i64,
) -> Result<RebuildFlipAck, LedgerError> {
    if !validate_live_fence(transaction, fence, activated_at_unix_ms)? {
        return Ok(RebuildFlipAck::Stale);
    }
    let manifest =
        load_validated_manifest(transaction, fence.vector_space_id(), fence.generation())?
            .ok_or_else(corrupt)?;
    if !generation_is_available(transaction, &manifest)? {
        return Ok(RebuildFlipAck::Unavailable);
    }
    let current_source_seq =
        load_validated_source_sequence(transaction, fence.vector_space_id())?.source_seq;
    if manifest.applied_source_seq() != current_source_seq {
        return Ok(RebuildFlipAck::NotReady);
    }
    let authoritative = authoritative_source_fingerprint(transaction, fence.vector_space_id())?;
    let Some(indexed) = generation_source_fingerprint(transaction, &manifest)? else {
        return Ok(RebuildFlipAck::FingerprintMismatch);
    };
    if authoritative != indexed {
        return Ok(RebuildFlipAck::FingerprintMismatch);
    }
    if activated_at_unix_ms < manifest.created_at_unix_ms {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let mut current_generations = transaction
        .prepare(
            "SELECT generation FROM vector_index_manifest
             WHERE vector_space_id = ?1
               AND state IN ('active', 'unavailable', 'corrupt')
             ORDER BY generation",
        )
        .map_err(database_error)?
        .query_map(params![fence.vector_space_id().as_str()], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    if current_generations.len() > 1 {
        return Err(corrupt());
    }
    if let Some(current_generation) = current_generations.pop() {
        let current_generation =
            VectorIndexGeneration::new(current_generation).map_err(|_| corrupt())?;
        if current_generation == fence.generation() {
            return Err(corrupt());
        }
        let mut stored =
            load_stored_manifest(transaction, fence.vector_space_id(), current_generation)?
                .ok_or_else(corrupt)?;
        let current_manifest = validate_manifest(transaction, stored.clone())?;
        if verify_generation_objects(transaction, current_manifest.authority())?
            == GenerationObjectsStatus::Partial
        {
            return Ok(RebuildFlipAck::Unavailable);
        }
        if stored
            .activated_at_unix_ms
            .is_none_or(|activated| activated > activated_at_unix_ms)
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let previous_hash = stored.canonical_payload_hash.clone();
        stored.state = VectorIndexManifestState::Retired.as_str().to_string();
        stored.retired_at_unix_ms = Some(activated_at_unix_ms);
        stored.canonical_payload_hash = manifest_payload_hash(&stored)?;
        if update_manifest(transaction, &stored, &previous_hash)? != 1 {
            return Err(corrupt());
        }
    }

    let mut stored =
        load_stored_manifest(transaction, fence.vector_space_id(), fence.generation())?
            .ok_or_else(corrupt)?;
    let previous_hash = stored.canonical_payload_hash.clone();
    stored.state = VectorIndexManifestState::Active.as_str().to_string();
    stored.source_record_count = Some(
        i64::try_from(authoritative.record_count())
            .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
    );
    stored.source_fingerprint_sha256 = Some(authoritative.sha256().to_string());
    stored.activated_at_unix_ms = Some(activated_at_unix_ms);
    stored.canonical_payload_hash = manifest_payload_hash(&stored)?;
    if update_manifest(transaction, &stored, &previous_hash)? != 1 {
        return Err(corrupt());
    }

    let lease =
        load_stored_rebuild_lease(transaction, fence.vector_space_id())?.ok_or_else(corrupt)?;
    let deleted = transaction
        .execute(
            "DELETE FROM vector_index_rebuild_leases
             WHERE vector_space_id = ?1 AND generation = ?2
               AND lease_generation = ?3
               AND lease_owner_process_instance_id = ?4
               AND lease_token = ?5 AND canonical_payload_hash = ?6",
            params![
                fence.vector_space_id().as_str(),
                fence.generation().value(),
                fence.lease_generation(),
                fence.owner_process_instance_id().to_string(),
                fence.lease_token().to_string(),
                lease.canonical_payload_hash,
            ],
        )
        .map_err(database_error)?;
    if deleted != 1 {
        return Err(corrupt());
    }
    Ok(RebuildFlipAck::Activated {
        record_count: authoritative.record_count(),
    })
}

/// Drop one exact retired generation or reconcile an already-absent root.
pub(crate) fn cleanup_retired_generation(
    transaction: &Transaction<'_>,
    vector_space_id: &VectorSpaceId,
    generation: VectorIndexGeneration,
    dropped_at_unix_ms: i64,
) -> Result<RetiredGenerationCleanupAck, LedgerError> {
    if dropped_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let Some(mut stored) = load_stored_manifest(transaction, vector_space_id, generation)? else {
        return Ok(RetiredGenerationCleanupAck::Conflict);
    };
    let manifest = validate_manifest(transaction, stored.clone())?;
    let objects = verify_generation_objects(transaction, manifest.authority())?;
    if manifest.state() == VectorIndexManifestState::Dropped {
        return if objects == GenerationObjectsStatus::Missing {
            Ok(RetiredGenerationCleanupAck::AlreadyDropped)
        } else {
            Err(corrupt())
        };
    }
    if manifest.state() != VectorIndexManifestState::Retired {
        return Ok(RetiredGenerationCleanupAck::Conflict);
    }
    if objects == GenerationObjectsStatus::Partial {
        return Ok(RetiredGenerationCleanupAck::Partial);
    }
    if stored
        .retired_at_unix_ms
        .is_none_or(|retired| retired > dropped_at_unix_ms)
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let acknowledgement = if objects == GenerationObjectsStatus::Complete {
        if verify_sqlite_vec(transaction) != SqliteVecStatus::Available {
            return Ok(RetiredGenerationCleanupAck::Unavailable);
        }
        transaction
            .execute_batch(manifest.authority().drop_sql())
            .map_err(database_error)?;
        if verify_generation_objects(transaction, manifest.authority())?
            != GenerationObjectsStatus::Missing
        {
            return Err(corrupt());
        }
        RetiredGenerationCleanupAck::Dropped
    } else {
        RetiredGenerationCleanupAck::ReconciledMissing
    };
    let previous_hash = stored.canonical_payload_hash.clone();
    stored.state = VectorIndexManifestState::Dropped.as_str().to_string();
    stored.dropped_at_unix_ms = Some(dropped_at_unix_ms);
    stored.canonical_payload_hash = manifest_payload_hash(&stored)?;
    if update_manifest(transaction, &stored, &previous_hash)? != 1 {
        return Err(corrupt());
    }
    Ok(acknowledgement)
}

/// Canonical hash authority for one per-space source cursor row.
pub(crate) fn vector_source_sequence_payload_hash(
    vector_space_id: &VectorSpaceId,
    source_seq: i64,
    updated_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    canonical_sha256(&json!({
        "vector_space_id": vector_space_id.as_str(),
        "source_seq": source_seq,
        "updated_at_unix_ms": updated_at_unix_ms,
    }))
    .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

/// Canonical hash authority shared by Task 7 source-change producers and rebuilds.
#[allow(clippy::too_many_arguments)]
pub(crate) fn vector_source_change_payload_hash(
    vector_space_id: &VectorSpaceId,
    source_seq: i64,
    operation: VectorSourceChangeOperation,
    record_id: VectorRecordId,
    partition_id: PartitionId,
    vector_checksum: &VectorChecksum,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    canonical_sha256(&json!({
        "vector_space_id": vector_space_id.as_str(),
        "source_seq": source_seq,
        "operation": operation.as_str(),
        "record_id": record_id.to_string(),
        "partition_id": partition_id.value(),
        "vector_checksum": vector_checksum.as_str(),
        "created_at_unix_ms": created_at_unix_ms,
    }))
    .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn load_validated_source_sequence(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
) -> Result<StoredSourceSequence, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT vector_space_id, source_seq, updated_at_unix_ms,
                    canonical_payload_hash
             FROM vector_space_source_sequences WHERE vector_space_id = ?1",
            params![vector_space_id.as_str()],
            |row| {
                Ok(StoredSourceSequence {
                    vector_space_id: row.get(0)?,
                    source_seq: row.get(1)?,
                    updated_at_unix_ms: row.get(2)?,
                    canonical_payload_hash: row.get(3)?,
                })
            },
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if stored.vector_space_id != vector_space_id.as_str()
        || stored.source_seq < 0
        || stored.updated_at_unix_ms < 0
        || !is_sha256(&stored.canonical_payload_hash)
        || vector_source_sequence_payload_hash(
            vector_space_id,
            stored.source_seq,
            stored.updated_at_unix_ms,
        )
        .map_err(|_| corrupt())?
            != stored.canonical_payload_hash
    {
        return Err(corrupt());
    }
    Ok(stored)
}

fn load_validated_source_changes(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
    after_source_seq: i64,
    through_source_seq: i64,
) -> Result<Vec<ValidatedSourceChange>, LedgerError> {
    if after_source_seq < 0 || through_source_seq < after_source_seq {
        return Err(corrupt());
    }
    let limit = i64::try_from(REBUILD_CHUNK_MAX)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let mut statement = connection
        .prepare(
            "SELECT vector_space_id, source_seq, operation, record_id,
                    partition_id, vector_checksum, created_at_unix_ms,
                    canonical_payload_hash
             FROM vector_source_change_events
             WHERE vector_space_id = ?1 AND source_seq > ?2 AND source_seq <= ?3
             ORDER BY source_seq LIMIT ?4",
        )
        .map_err(database_error)?;
    let rows = statement
        .query_map(
            params![
                vector_space_id.as_str(),
                after_source_seq,
                through_source_seq,
                limit,
            ],
            |row| {
                Ok(StoredSourceChange {
                    vector_space_id: row.get(0)?,
                    source_seq: row.get(1)?,
                    operation: row.get(2)?,
                    record_id: row.get(3)?,
                    partition_id: row.get(4)?,
                    vector_checksum: row.get(5)?,
                    created_at_unix_ms: row.get(6)?,
                    canonical_payload_hash: row.get(7)?,
                })
            },
        )
        .map_err(database_error)?;
    let mut changes = Vec::new();
    for row in rows {
        let stored = row.map_err(database_error)?;
        let operation = VectorSourceChangeOperation::parse(&stored.operation)?;
        let record_id = parse_record_id(&stored.record_id)?;
        let partition_id = PartitionId::new(stored.partition_id).map_err(|_| corrupt())?;
        let vector_checksum =
            VectorChecksum::new(stored.vector_checksum.clone()).map_err(|_| corrupt())?;
        if stored.vector_space_id != vector_space_id.as_str()
            || stored.source_seq <= 0
            || stored.created_at_unix_ms < 0
            || !is_sha256(&stored.canonical_payload_hash)
            || vector_source_change_payload_hash(
                vector_space_id,
                stored.source_seq,
                operation,
                record_id,
                partition_id,
                &vector_checksum,
                stored.created_at_unix_ms,
            )
            .map_err(|_| corrupt())?
                != stored.canonical_payload_hash
        {
            return Err(corrupt());
        }
        changes.push(ValidatedSourceChange {
            source_seq: stored.source_seq,
            operation,
            record_id,
            partition_id,
            vector_checksum,
            created_at_unix_ms: stored.created_at_unix_ms,
        });
    }
    Ok(changes)
}

fn load_ready_source_record(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
    record_id: VectorRecordId,
) -> Result<Option<VectorRecord>, LedgerError> {
    match load_verified_vector_link_source(connection, vector_space_id, record_id)? {
        VerifiedVectorLinkSourceLoad::LinkMissing => Ok(None),
        VerifiedVectorLinkSourceLoad::AuthorityMissing => Err(corrupt()),
        VerifiedVectorLinkSourceLoad::Verified(source) => Ok(source.vector_record),
    }
}

fn generation_is_available(
    connection: &Connection,
    manifest: &ValidatedVectorIndexManifest,
) -> Result<bool, LedgerError> {
    if manifest.state() != VectorIndexManifestState::Building {
        return Err(corrupt());
    }
    if verify_generation_objects(connection, manifest.authority())?
        != GenerationObjectsStatus::Complete
    {
        return Ok(false);
    }
    Ok(verify_sqlite_vec(connection) == SqliteVecStatus::Available)
}

fn upsert_generation_record(
    transaction: &Transaction<'_>,
    manifest: &ValidatedVectorIndexManifest,
    record: &VectorRecord,
) -> Result<VectorIndexPointMutationAck, LedgerError> {
    if record.vector_space_id() != manifest.vector_space_id()
        || record.vector().vector().dimensions() != manifest.authority().dimensions()
    {
        return Ok(VectorIndexPointMutationAck::Conflict);
    }
    let root = manifest.authority().root().as_str();
    let select_sql = format!("SELECT partition_id, embedding FROM \"{root}\" WHERE record_id = ?1");
    let existing = transaction
        .query_row(
            &select_sql,
            params![record.record_id().to_string()],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    let expected_bytes = record.vector().blob().native_endian_bytes();
    if let Some((partition_id, vector_bytes)) = existing {
        return Ok(
            if partition_id == record.partition_id().value() && vector_bytes == expected_bytes {
                VectorIndexPointMutationAck::AlreadyApplied
            } else {
                VectorIndexPointMutationAck::Conflict
            },
        );
    }
    let insert_sql =
        format!("INSERT INTO \"{root}\" (record_id, embedding, partition_id) VALUES (?1, ?2, ?3)");
    transaction
        .execute(
            &insert_sql,
            params![
                record.record_id().to_string(),
                expected_bytes,
                record.partition_id().value(),
            ],
        )
        .map_err(database_error)?;
    Ok(VectorIndexPointMutationAck::Applied)
}

fn delete_generation_record(
    transaction: &Transaction<'_>,
    manifest: &ValidatedVectorIndexManifest,
    record_id: VectorRecordId,
) -> Result<VectorIndexPointMutationAck, LedgerError> {
    let root = manifest.authority().root().as_str();
    let delete_sql = format!("DELETE FROM \"{root}\" WHERE record_id = ?1");
    let deleted = transaction
        .execute(&delete_sql, params![record_id.to_string()])
        .map_err(database_error)?;
    Ok(if deleted == 0 {
        VectorIndexPointMutationAck::AlreadyApplied
    } else if deleted == 1 {
        VectorIndexPointMutationAck::Applied
    } else {
        return Err(corrupt());
    })
}

pub(crate) fn generation_source_fingerprint(
    connection: &Connection,
    manifest: &ValidatedVectorIndexManifest,
) -> Result<Option<VectorSourceFingerprint>, LedgerError> {
    let root = manifest.authority().root().as_str();
    let sql =
        format!("SELECT record_id, partition_id, embedding FROM \"{root}\" ORDER BY record_id");
    let mut statement = connection.prepare(&sql).map_err(database_error)?;
    let mut rows = statement.query([]).map_err(database_error)?;
    let mut builder = VectorSourceFingerprintBuilder::new();
    while let Some(row) = rows.next().map_err(database_error)? {
        let record_id = match row
            .get::<_, String>(0)
            .map_err(database_error)
            .and_then(|value| parse_record_id(&value))
        {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };
        let partition_id = match row
            .get::<_, i64>(1)
            .map_err(database_error)
            .and_then(|value| PartitionId::new(value).map_err(|_| corrupt()))
        {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };
        let native_blob = row.get::<_, Vec<u8>>(2).map_err(database_error)?;
        let vector = match AuthoritativeVector::from_native_blob(
            manifest.vector_space_id(),
            manifest.authority().dimensions(),
            &native_blob,
        ) {
            Ok(value) => value,
            Err(_) => return Ok(None),
        };
        if builder
            .push(record_id, partition_id, vector.blob().checksum())
            .is_err()
        {
            return Ok(None);
        }
    }
    Ok(Some(builder.finish()))
}

fn load_stored_manifest(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
    generation: VectorIndexGeneration,
) -> Result<Option<StoredManifest>, LedgerError> {
    connection
        .query_row(
            "SELECT vector_space_id, generation, state, root_table_name, dimensions,
                    expected_schema_objects_json, expected_schema_objects_sha256,
                    base_source_seq, applied_source_seq, build_cursor_record_id,
                    source_record_count, source_fingerprint_sha256, stable_error_class,
                    created_at_unix_ms, activated_at_unix_ms, retired_at_unix_ms,
                    dropped_at_unix_ms, canonical_payload_hash
             FROM vector_index_manifest
             WHERE vector_space_id = ?1 AND generation = ?2",
            params![vector_space_id.as_str(), generation.value()],
            stored_manifest_from_row,
        )
        .optional()
        .map_err(database_error)
}

fn update_manifest(
    transaction: &Transaction<'_>,
    stored: &StoredManifest,
    previous_hash: &str,
) -> Result<usize, LedgerError> {
    transaction
        .execute(
            "UPDATE vector_index_manifest
             SET state = ?1, root_table_name = ?2, dimensions = ?3,
                 expected_schema_objects_json = ?4,
                 expected_schema_objects_sha256 = ?5, base_source_seq = ?6,
                 applied_source_seq = ?7, build_cursor_record_id = ?8,
                 source_record_count = ?9, source_fingerprint_sha256 = ?10,
                 stable_error_class = ?11, created_at_unix_ms = ?12,
                 activated_at_unix_ms = ?13, retired_at_unix_ms = ?14,
                 dropped_at_unix_ms = ?15, canonical_payload_hash = ?16
             WHERE vector_space_id = ?17 AND generation = ?18
               AND canonical_payload_hash = ?19",
            params![
                stored.state,
                stored.root_table_name,
                stored.dimensions,
                stored.expected_schema_objects_json,
                stored.expected_schema_objects_sha256,
                stored.base_source_seq,
                stored.applied_source_seq,
                stored.build_cursor_record_id,
                stored.source_record_count,
                stored.source_fingerprint_sha256,
                stored.stable_error_class,
                stored.created_at_unix_ms,
                stored.activated_at_unix_ms,
                stored.retired_at_unix_ms,
                stored.dropped_at_unix_ms,
                stored.canonical_payload_hash,
                stored.vector_space_id,
                stored.generation,
                previous_hash,
            ],
        )
        .map_err(database_error)
}

fn update_build_progress(
    transaction: &Transaction<'_>,
    fence: &RebuildLeaseFence,
    build_cursor_record_id: Option<Option<VectorRecordId>>,
    applied_source_seq: Option<i64>,
    updated_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let mut manifest =
        load_stored_manifest(transaction, fence.vector_space_id(), fence.generation())?
            .ok_or_else(corrupt)?;
    let mut lease =
        load_stored_rebuild_lease(transaction, fence.vector_space_id())?.ok_or_else(corrupt)?;
    let current_fence = validate_rebuild_lease(transaction, &lease)?;
    if !fence_matches(fence, &current_fence) || updated_at_unix_ms < lease.updated_at_unix_ms {
        return Err(corrupt());
    }
    if let Some(cursor) = build_cursor_record_id {
        let cursor = cursor.map(|record_id| record_id.to_string());
        manifest.build_cursor_record_id = cursor.clone();
        lease.build_cursor_record_id = cursor;
    }
    if let Some(source_seq) = applied_source_seq {
        if source_seq < manifest.applied_source_seq || source_seq < manifest.base_source_seq {
            return Err(corrupt());
        }
        manifest.applied_source_seq = source_seq;
        lease.applied_source_seq = source_seq;
    }
    lease.updated_at_unix_ms = updated_at_unix_ms;
    let previous_manifest_hash = manifest.canonical_payload_hash.clone();
    let previous_lease_hash = lease.canonical_payload_hash.clone();
    manifest.canonical_payload_hash = manifest_payload_hash(&manifest)?;
    lease.canonical_payload_hash = rebuild_lease_payload_hash(&lease)?;
    if update_manifest(transaction, &manifest, &previous_manifest_hash)? != 1
        || update_rebuild_lease(transaction, &lease, &previous_lease_hash)? != 1
    {
        return Err(corrupt());
    }
    Ok(())
}

fn stored_manifest_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredManifest> {
    Ok(StoredManifest {
        vector_space_id: row.get(0)?,
        generation: row.get(1)?,
        state: row.get(2)?,
        root_table_name: row.get(3)?,
        dimensions: row.get(4)?,
        expected_schema_objects_json: row.get(5)?,
        expected_schema_objects_sha256: row.get(6)?,
        base_source_seq: row.get(7)?,
        applied_source_seq: row.get(8)?,
        build_cursor_record_id: row.get(9)?,
        source_record_count: row.get(10)?,
        source_fingerprint_sha256: row.get(11)?,
        stable_error_class: row.get(12)?,
        created_at_unix_ms: row.get(13)?,
        activated_at_unix_ms: row.get(14)?,
        retired_at_unix_ms: row.get(15)?,
        dropped_at_unix_ms: row.get(16)?,
        canonical_payload_hash: row.get(17)?,
    })
}

fn load_building_manifest(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
) -> Result<Option<ValidatedVectorIndexManifest>, LedgerError> {
    let generation = connection
        .query_row(
            "SELECT generation FROM vector_index_manifest
             WHERE vector_space_id = ?1 AND state = 'building'",
            params![vector_space_id.as_str()],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?;
    generation
        .map(|generation| {
            let generation = VectorIndexGeneration::new(generation).map_err(|_| corrupt())?;
            load_validated_manifest(connection, vector_space_id, generation)?.ok_or_else(corrupt)
        })
        .transpose()
}

fn project_uuid_for_space(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
) -> Result<Uuid, LedgerError> {
    let project_uuid = connection
        .query_row(
            "SELECT project_uuid FROM vector_spaces WHERE vector_space_id = ?1",
            params![vector_space_id.as_str()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    parse_uuid_v7(&project_uuid)
}

fn load_stored_rebuild_lease(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
) -> Result<Option<StoredRebuildLease>, LedgerError> {
    connection
        .query_row(
            "SELECT vector_space_id, generation, lease_generation,
                    lease_owner_process_instance_id, lease_token,
                    lease_expires_at_unix_ms, base_source_seq, applied_source_seq,
                    build_cursor_record_id, created_at_unix_ms, updated_at_unix_ms,
                    canonical_payload_hash
             FROM vector_index_rebuild_leases WHERE vector_space_id = ?1",
            params![vector_space_id.as_str()],
            |row| {
                Ok(StoredRebuildLease {
                    vector_space_id: row.get(0)?,
                    generation: row.get(1)?,
                    lease_generation: row.get(2)?,
                    lease_owner_process_instance_id: row.get(3)?,
                    lease_token: row.get(4)?,
                    lease_expires_at_unix_ms: row.get(5)?,
                    base_source_seq: row.get(6)?,
                    applied_source_seq: row.get(7)?,
                    build_cursor_record_id: row.get(8)?,
                    created_at_unix_ms: row.get(9)?,
                    updated_at_unix_ms: row.get(10)?,
                    canonical_payload_hash: row.get(11)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

/// Load one optional rebuild lease only after its row, manifest, and project authority verify.
pub(crate) fn load_validated_rebuild_lease(
    connection: &Connection,
    vector_space_id: &VectorSpaceId,
) -> Result<Option<RebuildLeaseFence>, LedgerError> {
    load_stored_rebuild_lease(connection, vector_space_id)?
        .as_ref()
        .map(|stored| validate_rebuild_lease(connection, stored))
        .transpose()
}

fn validate_rebuild_lease(
    connection: &Connection,
    stored: &StoredRebuildLease,
) -> Result<RebuildLeaseFence, LedgerError> {
    let vector_space_id =
        VectorSpaceId::new(stored.vector_space_id.clone()).map_err(|_| corrupt())?;
    let generation = VectorIndexGeneration::new(stored.generation).map_err(|_| corrupt())?;
    let owner_process_instance_id = parse_uuid_v7(&stored.lease_owner_process_instance_id)?;
    let lease_token = parse_uuid_v7(&stored.lease_token)?;
    if stored.lease_generation <= 0
        || stored.base_source_seq < 0
        || stored.applied_source_seq < stored.base_source_seq
        || stored.created_at_unix_ms < 0
        || stored.updated_at_unix_ms < stored.created_at_unix_ms
        || stored.lease_expires_at_unix_ms < stored.created_at_unix_ms
        || !is_sha256(&stored.canonical_payload_hash)
        || rebuild_lease_payload_hash(stored).map_err(|_| corrupt())?
            != stored.canonical_payload_hash
    {
        return Err(corrupt());
    }
    if let Some(cursor) = stored.build_cursor_record_id.as_deref() {
        parse_record_id(cursor)?;
    }
    let manifest =
        load_validated_manifest(connection, &vector_space_id, generation)?.ok_or_else(corrupt)?;
    if manifest.state() != VectorIndexManifestState::Building
        || manifest.base_source_seq() != stored.base_source_seq
        || manifest.applied_source_seq() != stored.applied_source_seq
        || manifest
            .build_cursor_record_id()
            .map(|value| value.to_string())
            != stored.build_cursor_record_id
    {
        return Err(corrupt());
    }
    let project_uuid = project_uuid_for_space(connection, &vector_space_id)?;
    let owner_project = connection
        .query_row(
            "SELECT project_uuid FROM process_instances WHERE process_instance_id = ?1",
            params![owner_process_instance_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    if owner_project != project_uuid.to_string() {
        return Err(corrupt());
    }
    Ok(RebuildLeaseFence {
        vector_space_id,
        generation,
        lease_generation: stored.lease_generation,
        owner_process_instance_id,
        lease_token,
        lease_expires_at_unix_ms: stored.lease_expires_at_unix_ms,
    })
}

fn rebuild_lease_payload_hash(stored: &StoredRebuildLease) -> Result<String, LedgerError> {
    canonical_sha256(&json!({
        "vector_space_id": stored.vector_space_id,
        "generation": stored.generation,
        "lease_generation": stored.lease_generation,
        "lease_owner_process_instance_id": stored.lease_owner_process_instance_id,
        "lease_token": stored.lease_token,
        "lease_expires_at_unix_ms": stored.lease_expires_at_unix_ms,
        "base_source_seq": stored.base_source_seq,
        "applied_source_seq": stored.applied_source_seq,
        "build_cursor_record_id": stored.build_cursor_record_id,
        "created_at_unix_ms": stored.created_at_unix_ms,
        "updated_at_unix_ms": stored.updated_at_unix_ms,
    }))
    .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn update_rebuild_lease(
    transaction: &Transaction<'_>,
    stored: &StoredRebuildLease,
    previous_hash: &str,
) -> Result<usize, LedgerError> {
    transaction
        .execute(
            "UPDATE vector_index_rebuild_leases
             SET generation = ?1, lease_generation = ?2,
                 lease_owner_process_instance_id = ?3, lease_token = ?4,
                 lease_expires_at_unix_ms = ?5, base_source_seq = ?6,
                 applied_source_seq = ?7, build_cursor_record_id = ?8,
                 created_at_unix_ms = ?9, updated_at_unix_ms = ?10,
                 canonical_payload_hash = ?11
             WHERE vector_space_id = ?12 AND canonical_payload_hash = ?13",
            params![
                stored.generation,
                stored.lease_generation,
                stored.lease_owner_process_instance_id,
                stored.lease_token,
                stored.lease_expires_at_unix_ms,
                stored.base_source_seq,
                stored.applied_source_seq,
                stored.build_cursor_record_id,
                stored.created_at_unix_ms,
                stored.updated_at_unix_ms,
                stored.canonical_payload_hash,
                stored.vector_space_id,
                previous_hash,
            ],
        )
        .map_err(database_error)
}

fn fence_matches(expected: &RebuildLeaseFence, actual: &RebuildLeaseFence) -> bool {
    expected.vector_space_id() == actual.vector_space_id()
        && expected.generation() == actual.generation()
        && expected.lease_generation() == actual.lease_generation()
        && expected.owner_process_instance_id() == actual.owner_process_instance_id()
        && expected.lease_token() == actual.lease_token()
}

fn validate_live_fence(
    connection: &Connection,
    fence: &RebuildLeaseFence,
    observed_at_unix_ms: i64,
) -> Result<bool, LedgerError> {
    if observed_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let Some(stored) = load_stored_rebuild_lease(connection, fence.vector_space_id())? else {
        return Ok(false);
    };
    let current = validate_rebuild_lease(connection, &stored)?;
    if !fence_matches(fence, &current)
        || current.lease_expires_at_unix_ms() <= observed_at_unix_ms
        || observed_at_unix_ms < stored.updated_at_unix_ms
    {
        return Ok(false);
    }
    let project_uuid = project_uuid_for_space(connection, fence.vector_space_id())?;
    Ok(verified_process_status_at(
        connection,
        project_uuid,
        fence.owner_process_instance_id(),
        observed_at_unix_ms,
    )? == ProcessStatusAt::Live)
}

fn validate_manifest(
    connection: &Connection,
    stored: StoredManifest,
) -> Result<ValidatedVectorIndexManifest, LedgerError> {
    let vector_space_id =
        VectorSpaceId::new(stored.vector_space_id.clone()).map_err(|_| corrupt())?;
    let generation = VectorIndexGeneration::new(stored.generation).map_err(|_| corrupt())?;
    let dimensions = u32::try_from(stored.dimensions)
        .ok()
        .and_then(|value| VectorDimensions::new(value).ok())
        .ok_or_else(corrupt)?;
    let root = Vec0RootName::parse(&stored.root_table_name).map_err(|_| corrupt())?;
    if root.vector_space_id() != &vector_space_id || root.generation() != generation {
        return Err(corrupt());
    }
    let authority = Vec0SchemaAuthority::new(root, dimensions);
    if stored.expected_schema_objects_json != authority.manifest_json()
        || stored.expected_schema_objects_sha256 != authority.manifest_sha256()
    {
        return Err(corrupt());
    }
    let space_dimensions = connection
        .query_row(
            "SELECT dimensions FROM vector_spaces WHERE vector_space_id = ?1",
            params![vector_space_id.as_str()],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    if space_dimensions != stored.dimensions {
        return Err(corrupt());
    }
    if stored.base_source_seq < 0
        || stored.applied_source_seq < stored.base_source_seq
        || stored.created_at_unix_ms < 0
        || !is_sha256(&stored.canonical_payload_hash)
        || manifest_payload_hash(&stored).map_err(|_| corrupt())? != stored.canonical_payload_hash
    {
        return Err(corrupt());
    }
    let build_cursor_record_id = stored
        .build_cursor_record_id
        .as_deref()
        .map(parse_record_id)
        .transpose()?;
    let source_record_count = stored
        .source_record_count
        .map(|count| u64::try_from(count).map_err(|_| corrupt()))
        .transpose()?;
    if source_record_count.is_some() != stored.source_fingerprint_sha256.is_some()
        || stored
            .source_fingerprint_sha256
            .as_deref()
            .is_some_and(|value| !is_sha256(value))
        || stored
            .stable_error_class
            .as_deref()
            .is_some_and(|value| !is_stable_error_class(value))
    {
        return Err(corrupt());
    }
    let state = VectorIndexManifestState::parse(&stored.state)?;
    validate_lifecycle(&stored, state, source_record_count)?;
    Ok(ValidatedVectorIndexManifest {
        vector_space_id,
        generation,
        state,
        authority,
        base_source_seq: stored.base_source_seq,
        applied_source_seq: stored.applied_source_seq,
        build_cursor_record_id,
        source_record_count,
        source_fingerprint_sha256: stored.source_fingerprint_sha256,
        stable_error_class: stored.stable_error_class,
        created_at_unix_ms: stored.created_at_unix_ms,
        activated_at_unix_ms: stored.activated_at_unix_ms,
        retired_at_unix_ms: stored.retired_at_unix_ms,
        dropped_at_unix_ms: stored.dropped_at_unix_ms,
        canonical_payload_hash: stored.canonical_payload_hash,
    })
}

fn validate_lifecycle(
    stored: &StoredManifest,
    state: VectorIndexManifestState,
    source_record_count: Option<u64>,
) -> Result<(), LedgerError> {
    let valid = match state {
        VectorIndexManifestState::Building => {
            stored.activated_at_unix_ms.is_none()
                && stored.retired_at_unix_ms.is_none()
                && stored.dropped_at_unix_ms.is_none()
                && stored.stable_error_class.is_none()
        }
        VectorIndexManifestState::Active => {
            stored.activated_at_unix_ms.is_some()
                && stored.retired_at_unix_ms.is_none()
                && stored.dropped_at_unix_ms.is_none()
                && source_record_count.is_some()
                && stored.stable_error_class.is_none()
        }
        VectorIndexManifestState::Unavailable | VectorIndexManifestState::Corrupt => {
            stored.activated_at_unix_ms.is_some()
                && stored.retired_at_unix_ms.is_none()
                && stored.dropped_at_unix_ms.is_none()
                && source_record_count.is_some()
                && stored.stable_error_class.is_some()
        }
        VectorIndexManifestState::Retired => {
            stored.activated_at_unix_ms.is_some()
                && stored.retired_at_unix_ms.is_some()
                && stored.dropped_at_unix_ms.is_none()
                && source_record_count.is_some()
        }
        VectorIndexManifestState::Dropped => {
            stored.activated_at_unix_ms.is_some()
                && stored.retired_at_unix_ms.is_some()
                && stored.dropped_at_unix_ms.is_some()
                && source_record_count.is_some()
        }
    };
    if !valid
        || stored
            .activated_at_unix_ms
            .is_some_and(|value| value < stored.created_at_unix_ms)
        || stored
            .retired_at_unix_ms
            .is_some_and(|value| value < stored.created_at_unix_ms)
        || stored
            .dropped_at_unix_ms
            .is_some_and(|value| value < stored.created_at_unix_ms)
        || matches!(
            (stored.retired_at_unix_ms, stored.dropped_at_unix_ms),
            (Some(retired), Some(dropped)) if dropped < retired
        )
        || matches!(
            (stored.activated_at_unix_ms, stored.retired_at_unix_ms),
            (Some(activated), Some(retired)) if retired < activated
        )
    {
        return Err(corrupt());
    }
    Ok(())
}

fn manifest_payload_hash(stored: &StoredManifest) -> Result<String, LedgerError> {
    canonical_sha256(&json!({
        "vector_space_id": stored.vector_space_id,
        "generation": stored.generation,
        "state": stored.state,
        "root_table_name": stored.root_table_name,
        "dimensions": stored.dimensions,
        "expected_schema_objects_json": stored.expected_schema_objects_json,
        "expected_schema_objects_sha256": stored.expected_schema_objects_sha256,
        "base_source_seq": stored.base_source_seq,
        "applied_source_seq": stored.applied_source_seq,
        "build_cursor_record_id": stored.build_cursor_record_id,
        "source_record_count": stored.source_record_count,
        "source_fingerprint_sha256": stored.source_fingerprint_sha256,
        "stable_error_class": stored.stable_error_class,
        "created_at_unix_ms": stored.created_at_unix_ms,
        "activated_at_unix_ms": stored.activated_at_unix_ms,
        "retired_at_unix_ms": stored.retired_at_unix_ms,
        "dropped_at_unix_ms": stored.dropped_at_unix_ms,
    }))
    .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn parse_record_id(value: &str) -> Result<VectorRecordId, LedgerError> {
    let parsed = uuid::Uuid::parse_str(value).map_err(|_| corrupt())?;
    if parsed.to_string() != value {
        return Err(corrupt());
    }
    VectorRecordId::new(parsed).map_err(|_| corrupt())
}

fn decode_checksum(value: &str) -> Result<[u8; 32], VectorSourceFingerprintError> {
    if value.len() != 64 {
        return Err(VectorSourceFingerprintError::InvalidChecksum);
    }
    let mut decoded = [0_u8; 32];
    for (output, pair) in decoded.iter_mut().zip(value.as_bytes().chunks_exact(2)) {
        let high = decode_hex_nibble(pair[0])?;
        let low = decode_hex_nibble(pair[1])?;
        *output = (high << 4) | low;
    }
    Ok(decoded)
}

fn decode_hex_nibble(value: u8) -> Result<u8, VectorSourceFingerprintError> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        _ => Err(VectorSourceFingerprintError::InvalidChecksum),
    }
}

fn is_stable_error_class(value: &str) -> bool {
    (1..=128).contains(&value.len())
        && value
            .bytes()
            .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"_.".contains(&byte))
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
    use std::path::PathBuf;
    use std::time::Duration;

    use rusqlite::Connection;
    use tempfile::TempDir;

    use super::*;
    use crate::config::LearningConfig;
    use crate::ledger::migrations::MIGRATIONS;
    use crate::ledger::repository::retention::{RetentionAck, RetentionRequest};
    use crate::ledger::repository::tests::{config, database_path};
    use crate::ledger::repository::{LedgerRepository, ready_retention_runtime_fixture};
    use crate::sqlite_vec_extension::register as register_sqlite_vec;
    use crate::vector::{AuthoritativeVector, NormalizedVector, VectorError};

    fn space_id() -> VectorSpaceId {
        VectorSpaceId::new("a".repeat(64)).unwrap()
    }

    fn fixture() -> Connection {
        assert_eq!(register_sqlite_vec(), SqliteVecStatus::Available);
        let connection = Connection::open_in_memory().unwrap();
        for migration in MIGRATIONS {
            connection.execute_batch(migration.sql).unwrap();
        }
        let project_uuid = uuid::Uuid::now_v7().to_string();
        connection
            .execute(
                "INSERT INTO project_metadata (
                    singleton_key, project_uuid, project_id, created_at_unix_ms,
                    application_version, canonical_payload_hash
                 ) VALUES (1, ?1, 'vector-index-fixture', 0, 'test', ?2)",
                params![&project_uuid, "e".repeat(64)],
            )
            .unwrap();
        seed_vector_space(&connection, &project_uuid);
        connection
    }

    fn seed_vector_space(connection: &Connection, project_uuid: &str) {
        connection
            .execute(
                "INSERT INTO embedder_profiles (
                    embedder_profile_version_id, profile_id, protocol, endpoint_url,
                    endpoint_identity_sha256, model, provider_revision, dimensions,
                    credential_env_name_sha256, timeout_ms, max_in_flight, batch_size,
                    egress_class, canonical_profile_json, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, 'fixture', 'openai-embeddings-v1',
                    'http://127.0.0.1/v1/embeddings', ?2, 'fixture-model', 'r1', 3,
                    NULL, 1000, 1, 1, 'loopback_http', '{}', 0, ?1)",
                params!["b".repeat(64), "c".repeat(64)],
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO vector_spaces (
                    vector_space_id, project_uuid, embedder_profile_version_id,
                    canonicalizer_version_id, canonicalizer_identity_json,
                    endpoint_identity_sha256, model, provider_revision, dimensions,
                    metric, normalization, canonical_space_json, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, '{}', ?5, 'fixture-model', 'r1', 3,
                    'cosine', 'l2_f32_v1', '{}', 0, ?1)",
                params![
                    space_id().as_str(),
                    project_uuid,
                    "b".repeat(64),
                    "d".repeat(64),
                    "c".repeat(64),
                ],
            )
            .unwrap();
        let sequence_hash = vector_source_sequence_payload_hash(&space_id(), 0, 0).unwrap();
        connection
            .execute(
                "INSERT INTO vector_space_source_sequences (
                    vector_space_id, source_seq, updated_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, 0, 0, ?2)",
                params![space_id().as_str(), sequence_hash],
            )
            .unwrap();
    }

    struct ActivatedFixture {
        _temporary: TempDir,
        path: PathBuf,
        activated: crate::ledger::repository::ActivatedLedger,
    }

    fn activated_fixture(activated_at_unix_ms: i64) -> ActivatedFixture {
        let temporary = tempfile::tempdir().unwrap();
        let path = database_path(&temporary);
        let mut router_config = config(&path, "vector-index-rebuild-fixture");
        router_config.embedders[0].dimensions = 3;
        router_config.pools[0].learning = Some(LearningConfig::minimal("embedder-a"));
        let mut activated =
            LedgerRepository::activate_at(&router_config, activated_at_unix_ms).unwrap();
        seed_vector_space(
            activated.repository.connection_mut(),
            &activated.identity.project_uuid.to_string(),
        );
        ActivatedFixture {
            _temporary: temporary,
            path,
            activated,
        }
    }

    fn mapped_space_id(fixture: &ActivatedFixture) -> VectorSpaceId {
        fixture.activated.identity.pools["pool-a"]
            .vector_space
            .as_ref()
            .expect("learning fixture has mapped vector authority")
            .vector_space_id
            .clone()
    }

    fn authorize(connection: &mut Connection) -> ValidatedVectorIndexManifest {
        let transaction = connection.transaction().unwrap();
        let manifest = match authorize_generation(
            &transaction,
            &space_id(),
            VectorDimensions::new(3).unwrap(),
            10,
        )
        .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest) => manifest,
            GenerationAuthorizationAck::AlreadyExists(_) => panic!("unexpected existing manifest"),
            GenerationAuthorizationAck::AuthorityMissing => panic!("missing generation authority"),
        };
        transaction.commit().unwrap();
        manifest
    }

    fn mark_active(connection: &Connection) {
        mark_active_for(connection, &space_id());
    }

    fn mark_active_for(connection: &Connection, vector_space_id: &VectorSpaceId) {
        let mut stored = connection
            .query_row(
                "SELECT vector_space_id, generation, state, root_table_name, dimensions,
                        expected_schema_objects_json, expected_schema_objects_sha256,
                        base_source_seq, applied_source_seq, build_cursor_record_id,
                        source_record_count, source_fingerprint_sha256, stable_error_class,
                        created_at_unix_ms, activated_at_unix_ms, retired_at_unix_ms,
                        dropped_at_unix_ms, canonical_payload_hash
                 FROM vector_index_manifest
                 WHERE vector_space_id = ?1 AND generation = 1",
                params![vector_space_id.as_str()],
                stored_manifest_from_row,
            )
            .unwrap();
        stored.state = VectorIndexManifestState::Active.as_str().to_string();
        stored.source_record_count = Some(0);
        stored.source_fingerprint_sha256 = Some("f".repeat(64));
        stored.activated_at_unix_ms = Some(20);
        stored.canonical_payload_hash = manifest_payload_hash(&stored).unwrap();
        connection
            .execute(
                "UPDATE vector_index_manifest
                 SET state = ?1, source_record_count = ?2,
                     source_fingerprint_sha256 = ?3, activated_at_unix_ms = ?4,
                     canonical_payload_hash = ?5
                 WHERE vector_space_id = ?6 AND generation = 1",
                params![
                    stored.state,
                    stored.source_record_count,
                    stored.source_fingerprint_sha256,
                    stored.activated_at_unix_ms,
                    stored.canonical_payload_hash,
                    vector_space_id.as_str(),
                ],
            )
            .unwrap();
    }

    fn create_objects(connection: &mut Connection, manifest: &ValidatedVectorIndexManifest) {
        let transaction = connection.transaction().unwrap();
        assert_eq!(
            create_generation_objects_for_manifest(
                &transaction,
                manifest.vector_space_id(),
                manifest.generation(),
            )
            .unwrap(),
            GenerationObjectCreationAck::Created
        );
        transaction.commit().unwrap();
    }

    fn record(id: u64, partition: i64, values: &[f64]) -> VectorRecord {
        let record_id = format!("01890f47-6c7d-7000-8000-{id:012x}");
        let dimensions = VectorDimensions::new(u32::try_from(values.len()).unwrap()).unwrap();
        let vector = NormalizedVector::from_provider_f64(values, dimensions).unwrap();
        let vector = AuthoritativeVector::from_normalized(&space_id(), vector).unwrap();
        VectorRecord::new(
            VectorRecordId::new(uuid::Uuid::parse_str(&record_id).unwrap()).unwrap(),
            space_id(),
            PartitionId::new(partition).unwrap(),
            vector,
        )
        .unwrap()
    }

    fn append_source_change(
        connection: &Connection,
        record: &VectorRecord,
        source_seq: i64,
        operation: VectorSourceChangeOperation,
        created_at_unix_ms: i64,
    ) {
        let change_hash = vector_source_change_payload_hash(
            record.vector_space_id(),
            source_seq,
            operation,
            record.record_id(),
            record.partition_id(),
            record.vector().blob().checksum(),
            created_at_unix_ms,
        )
        .unwrap();
        connection
            .execute(
                "INSERT INTO vector_source_change_events (
                    vector_space_id, source_seq, operation, record_id,
                    partition_id, vector_checksum, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    record.vector_space_id().as_str(),
                    source_seq,
                    operation.as_str(),
                    record.record_id().to_string(),
                    record.partition_id().value(),
                    record.vector().blob().checksum().as_str(),
                    created_at_unix_ms,
                    change_hash,
                ],
            )
            .unwrap();
        let sequence_hash = vector_source_sequence_payload_hash(
            record.vector_space_id(),
            source_seq,
            created_at_unix_ms,
        )
        .unwrap();
        assert_eq!(
            connection
                .execute(
                    "UPDATE vector_space_source_sequences
                     SET source_seq = ?1, updated_at_unix_ms = ?2,
                         canonical_payload_hash = ?3
                     WHERE vector_space_id = ?4",
                    params![
                        source_seq,
                        created_at_unix_ms,
                        sequence_hash,
                        record.vector_space_id().as_str(),
                    ],
                )
                .unwrap(),
            1
        );
    }

    fn clone_partition_with_different_candidate(connection: &Connection, partition_id: i64) -> i64 {
        let canonical: String = connection
            .query_row(
                "SELECT canonical_partition_json FROM routing_partitions
                 WHERE partition_id = ?1",
                params![partition_id],
                |row| row.get(0),
            )
            .unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&canonical).unwrap();
        value["candidate_id"] = json!("candidate-b");
        let canonical = crate::canonical_json::canonical_json(&value).unwrap();
        let partition_hash = canonical_sha256(&value).unwrap();
        connection
            .query_row(
                "INSERT INTO routing_partitions (
                    partition_hash, canonical_partition_json, project_uuid, pool_id,
                    tenant_policy_hash, agent_policy_hash, policy_version_id,
                    learning_generation_id, api_family, transport_identity,
                    anchor_model, anchor_revision, candidate_id, candidate_model,
                    candidate_model_revision, decoding_fingerprint, evaluator_version,
                    vector_space_id, created_at_unix_ms, canonical_payload_hash
                 )
                 SELECT ?1, ?2, project_uuid, pool_id, tenant_policy_hash,
                        agent_policy_hash, policy_version_id, learning_generation_id,
                        api_family, transport_identity, anchor_model, anchor_revision,
                        'candidate-b', candidate_model, candidate_model_revision,
                        decoding_fingerprint, evaluator_version, vector_space_id,
                        created_at_unix_ms, ?1
                 FROM routing_partitions WHERE partition_id = ?3
                 RETURNING partition_id",
                params![partition_hash, canonical, partition_id],
                |row| row.get(0),
            )
            .unwrap()
    }

    fn seed_ready_source(
        connection: &Connection,
        record: &VectorRecord,
        source_seq: i64,
        created_at_unix_ms: i64,
    ) {
        let project_uuid = connection
            .query_row(
                "SELECT project_uuid FROM project_metadata WHERE singleton_key = 1",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let query_hash = crate::fingerprint::sha256_hex(record.record_id().to_string().as_bytes());
        connection
            .execute(
                "INSERT OR IGNORE INTO canonical_routing_queries (
                    canonical_query_hash, canonical_query_json,
                    canonical_size_bytes, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, '{}', 2, ?2, ?1)",
                params![query_hash, created_at_unix_ms],
            )
            .unwrap();
        let partition_hash = crate::fingerprint::sha256_hex(
            format!("partition:{}", record.partition_id().value()).as_bytes(),
        );
        connection
            .execute(
                "INSERT OR IGNORE INTO routing_partitions (
                    partition_id, partition_hash, canonical_partition_json,
                    project_uuid, pool_id, tenant_policy_hash, agent_policy_hash,
                    policy_version_id, learning_generation_id, api_family,
                    transport_identity, anchor_model, anchor_revision,
                    candidate_id, candidate_model, candidate_model_revision,
                    decoding_fingerprint, evaluator_version, vector_space_id,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, ?2, '{}', ?3, 'pool-a', ?4, ?5, ?6, ?7,
                    'openai_chat_completions', 'fixture', 'anchor', 'r1',
                    'candidate', 'candidate-model', 'r1', ?8, ?9, ?10, ?11, ?2
                 )",
                params![
                    record.partition_id().value(),
                    partition_hash,
                    project_uuid,
                    "1".repeat(64),
                    "2".repeat(64),
                    "3".repeat(64),
                    "01890f47-6c7d-7000-8000-000000000100",
                    "4".repeat(64),
                    "5".repeat(64),
                    record.vector_space_id().as_str(),
                    created_at_unix_ms,
                ],
            )
            .unwrap();
        let embedding_id = uuid::Uuid::now_v7();
        connection
            .execute(
                "INSERT INTO embeddings (
                    embedding_id, vector_space_id, canonical_query_hash,
                    content_hash, dimensions, vector_blob, vector_checksum,
                    source, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'provider', ?8, ?9)",
                params![
                    embedding_id.to_string(),
                    record.vector_space_id().as_str(),
                    query_hash,
                    "6".repeat(64),
                    i64::from(record.vector().vector().dimensions().value()),
                    record.vector().blob().bytes(),
                    record.vector().blob().checksum().as_str(),
                    created_at_unix_ms,
                    crate::fingerprint::sha256_hex(embedding_id.as_bytes()),
                ],
            )
            .unwrap();
        let link_hash = crate::fingerprint::sha256_hex(record.record_id().to_string().as_bytes());
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
                    record.record_id().to_string(),
                    crate::fingerprint::sha256_hex(
                        format!("outcome:{}", record.record_id()).as_bytes(),
                    ),
                    uuid::Uuid::now_v7().to_string(),
                    uuid::Uuid::now_v7().to_string(),
                    uuid::Uuid::now_v7().to_string(),
                    uuid::Uuid::now_v7().to_string(),
                    record.vector_space_id().as_str(),
                    record.partition_id().value(),
                    query_hash,
                    created_at_unix_ms,
                    link_hash,
                ],
            )
            .unwrap();
        let state_id = uuid::Uuid::now_v7();
        connection
            .execute(
                "INSERT INTO evidence_vector_link_state_events (
                    evidence_vector_link_state_event_id, evidence_vector_link_id,
                    embedding_id, state, attempt_generation,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, 'ready', 0, ?4, ?5)",
                params![
                    state_id.to_string(),
                    record.record_id().to_string(),
                    embedding_id.to_string(),
                    created_at_unix_ms,
                    crate::fingerprint::sha256_hex(state_id.as_bytes()),
                ],
            )
            .unwrap();
        append_source_change(
            connection,
            record,
            source_seq,
            VectorSourceChangeOperation::Insert,
            created_at_unix_ms,
        );
    }

    fn cancel_ready_source(
        connection: &Connection,
        record: &VectorRecord,
        source_seq: i64,
        created_at_unix_ms: i64,
    ) {
        let state_id = uuid::Uuid::now_v7();
        connection
            .execute(
                "INSERT INTO evidence_vector_link_state_events (
                    evidence_vector_link_state_event_id, evidence_vector_link_id,
                    embedding_id, state, attempt_generation,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, NULL, 'canceled_retention', 1, ?3, ?4)",
                params![
                    state_id.to_string(),
                    record.record_id().to_string(),
                    created_at_unix_ms,
                    crate::fingerprint::sha256_hex(state_id.as_bytes()),
                ],
            )
            .unwrap();
        append_source_change(
            connection,
            record,
            source_seq,
            VectorSourceChangeOperation::Delete,
            created_at_unix_ms,
        );
    }

    #[test]
    fn retention_prunes_source_history_in_bounded_tail_pages() {
        let mut connection = fixture();
        connection
            .execute_batch("PRAGMA foreign_keys = OFF")
            .unwrap();
        for source_seq in 1..=300 {
            let record = record(source_seq as u64, 7, &[1.0, 0.0, 0.0]);
            append_source_change(
                &connection,
                &record,
                source_seq,
                VectorSourceChangeOperation::Insert,
                source_seq,
            );
        }
        connection
            .execute_batch("PRAGMA foreign_keys = ON")
            .unwrap();

        let transaction = connection.transaction().unwrap();
        assert_eq!(
            prune_vector_source_history_for_retention(&transaction, &space_id()).unwrap(),
            SourceHistoryRetentionAck::More {
                remaining_source_seq: 44,
            }
        );
        transaction.commit().unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM vector_source_change_events
                     WHERE vector_space_id = ?1",
                    params![space_id().as_str()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            44
        );
        assert_eq!(
            load_validated_source_sequence(&connection, &space_id())
                .unwrap()
                .source_seq,
            44
        );

        let transaction = connection.transaction().unwrap();
        assert_eq!(
            prune_vector_source_history_for_retention(&transaction, &space_id()).unwrap(),
            SourceHistoryRetentionAck::Drained
        );
        transaction.commit().unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM vector_source_change_events
                     WHERE vector_space_id = ?1",
                    params![space_id().as_str()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        assert_eq!(
            load_validated_source_sequence(&connection, &space_id())
                .unwrap()
                .source_seq,
            0
        );
    }

    #[test]
    fn manifest_first_authorization_is_exact_and_idempotent() {
        let mut connection = fixture();
        let created = authorize(&mut connection);
        assert_eq!(created.state(), VectorIndexManifestState::Building);
        assert_eq!(created.generation().value(), 1);
        assert_eq!(created.base_source_seq(), 0);
        assert_eq!(created.applied_source_seq(), 0);
        assert_eq!(
            verify_generation_objects(&connection, created.authority()).unwrap(),
            GenerationObjectsStatus::Missing
        );

        let transaction = connection.transaction().unwrap();
        let existing = authorize_generation(
            &transaction,
            &space_id(),
            VectorDimensions::new(3).unwrap(),
            11,
        )
        .unwrap();
        assert!(matches!(
            existing,
            GenerationAuthorizationAck::AlreadyExists(ref manifest)
                if manifest.canonical_payload_hash() == created.canonical_payload_hash()
        ));
        transaction.commit().unwrap();
        assert!(matches!(
            resolve_active_generation(&connection, &space_id()).unwrap(),
            ActiveGenerationResolution::Missing
        ));
    }

    #[test]
    fn mutable_manifest_hash_and_exact_authority_are_verified() {
        let mut connection = fixture();
        let manifest = authorize(&mut connection);
        connection
            .execute(
                "UPDATE vector_index_manifest SET canonical_payload_hash = ?1
                 WHERE vector_space_id = ?2 AND generation = 1",
                params!["f".repeat(64), space_id().as_str()],
            )
            .unwrap();
        let error = load_validated_manifest(
            &connection,
            &space_id(),
            VectorIndexGeneration::new(1).unwrap(),
        )
        .unwrap_err();
        assert_eq!(error.class(), LedgerErrorClass::CorruptDatabase);
        assert_eq!(
            manifest.authority().root().as_str(),
            "router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1"
        );
    }

    #[test]
    fn generated_objects_are_checked_exactly_and_reject_namespace_extras() {
        let mut connection = fixture();
        let manifest = authorize(&mut connection);
        connection
            .execute_batch(manifest.authority().create_sql())
            .unwrap();
        assert_eq!(
            verify_generation_objects(&connection, manifest.authority()).unwrap(),
            GenerationObjectsStatus::Complete
        );
        connection
            .execute_batch(&format!(
                "CREATE TABLE \"{}_unexpected\" (value INTEGER)",
                manifest.authority().root().as_str()
            ))
            .unwrap();
        let error = verify_generation_objects(&connection, manifest.authority()).unwrap_err();
        assert_eq!(error.class(), LedgerErrorClass::CorruptDatabase);
    }

    #[test]
    fn a_missing_authorized_object_is_vector_local_unavailability() {
        let mut connection = fixture();
        let manifest = authorize(&mut connection);
        connection
            .execute_batch(manifest.authority().create_sql())
            .unwrap();
        connection
            .execute_batch(&format!(
                "DROP TABLE \"{}_vector_chunks00\"",
                manifest.authority().root().as_str()
            ))
            .unwrap();
        assert_eq!(
            verify_generation_objects(&connection, manifest.authority()).unwrap(),
            GenerationObjectsStatus::Partial
        );
    }

    #[test]
    fn active_resolution_requires_exact_present_objects() {
        let mut connection = fixture();
        let manifest = authorize(&mut connection);
        mark_active(&connection);
        assert!(matches!(
            resolve_active_generation(&connection, &space_id()).unwrap(),
            ActiveGenerationResolution::Unavailable
        ));

        connection
            .execute_batch(manifest.authority().create_sql())
            .unwrap();
        let resolved = resolve_active_generation(&connection, &space_id()).unwrap();
        assert!(matches!(
            resolved,
            ActiveGenerationResolution::Active(ref active)
                if active.authority() == manifest.authority()
        ));
    }

    #[test]
    fn health_transitions_are_hash_fenced_idempotent_and_monotonic() {
        let mut fixture = activated_fixture(0);
        let vector_space_id = mapped_space_id(&fixture);
        let connection = fixture.activated.repository.connection_mut();
        let transaction = connection.transaction().unwrap();
        let building = match authorize_generation(
            &transaction,
            &vector_space_id,
            VectorDimensions::new(3).unwrap(),
            10,
        )
        .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest) => manifest,
            other => panic!("unexpected authorization: {other:?}"),
        };
        transaction.commit().unwrap();
        let transaction = connection.transaction().unwrap();
        assert_eq!(
            mark_vector_index_health(
                &transaction,
                &vector_space_id,
                building.generation(),
                building.canonical_payload_hash(),
                VectorIndexHealthTarget::Unavailable,
                "router.vector.unavailable",
                11,
            )
            .unwrap(),
            VectorIndexHealthMutationAck::Conflict
        );
        transaction.commit().unwrap();

        mark_active_for(connection, &vector_space_id);
        let active = load_validated_manifest(connection, &vector_space_id, building.generation())
            .unwrap()
            .unwrap();
        let active_hash = active.canonical_payload_hash().to_string();
        for (error_class, observed_at_unix_ms) in [("BAD", 20), ("router.vector.unavailable", 19)] {
            let transaction = connection.transaction().unwrap();
            let error = mark_vector_index_health(
                &transaction,
                &vector_space_id,
                active.generation(),
                &active_hash,
                VectorIndexHealthTarget::Unavailable,
                error_class,
                observed_at_unix_ms,
            )
            .unwrap_err();
            assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);
        }

        let transaction = connection.transaction().unwrap();
        let error = mark_vector_index_health(
            &transaction,
            &vector_space_id,
            active.generation(),
            "not-a-sha256",
            VectorIndexHealthTarget::Unavailable,
            "router.vector.unavailable",
            20,
        )
        .unwrap_err();
        assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);
        drop(transaction);

        let transaction = connection.transaction().unwrap();
        let applied_hash = match mark_vector_index_health(
            &transaction,
            &vector_space_id,
            active.generation(),
            &active_hash,
            VectorIndexHealthTarget::Unavailable,
            "router.vector.unavailable",
            21,
        )
        .unwrap()
        {
            VectorIndexHealthMutationAck::Applied { manifest_hash } => manifest_hash,
            other => panic!("unexpected unavailable acknowledgement: {other:?}"),
        };
        transaction.commit().unwrap();
        let unavailable =
            load_validated_manifest(connection, &vector_space_id, active.generation())
                .unwrap()
                .unwrap();
        assert_eq!(unavailable.state(), VectorIndexManifestState::Unavailable);
        assert_eq!(
            unavailable.stable_error_class(),
            Some("router.vector.unavailable")
        );
        assert_eq!(unavailable.canonical_payload_hash(), applied_hash);
        assert_ne!(unavailable.canonical_payload_hash(), active_hash);

        let transaction = connection.transaction().unwrap();
        assert_eq!(
            mark_vector_index_health(
                &transaction,
                &vector_space_id,
                active.generation(),
                &active_hash,
                VectorIndexHealthTarget::Corrupt,
                "router.vector.corrupt",
                22,
            )
            .unwrap(),
            VectorIndexHealthMutationAck::Stale
        );
        assert_eq!(
            mark_vector_index_health(
                &transaction,
                &vector_space_id,
                active.generation(),
                unavailable.canonical_payload_hash(),
                VectorIndexHealthTarget::Unavailable,
                "router.vector.unavailable",
                22,
            )
            .unwrap(),
            VectorIndexHealthMutationAck::AlreadyApplied {
                manifest_hash: unavailable.canonical_payload_hash().to_string(),
            }
        );
        assert_eq!(
            mark_vector_index_health(
                &transaction,
                &vector_space_id,
                active.generation(),
                unavailable.canonical_payload_hash(),
                VectorIndexHealthTarget::Unavailable,
                "router.vector.other",
                22,
            )
            .unwrap(),
            VectorIndexHealthMutationAck::Conflict
        );
        let corrupt_hash = match mark_vector_index_health(
            &transaction,
            &vector_space_id,
            active.generation(),
            unavailable.canonical_payload_hash(),
            VectorIndexHealthTarget::Corrupt,
            "router.vector.corrupt",
            22,
        )
        .unwrap()
        {
            VectorIndexHealthMutationAck::Applied { manifest_hash } => manifest_hash,
            other => panic!("unexpected corrupt acknowledgement: {other:?}"),
        };
        transaction.commit().unwrap();

        let corrupt = load_validated_manifest(connection, &vector_space_id, active.generation())
            .unwrap()
            .unwrap();
        assert_eq!(corrupt.state(), VectorIndexManifestState::Corrupt);
        assert_eq!(corrupt.canonical_payload_hash(), corrupt_hash);
        let transaction = connection.transaction().unwrap();
        assert_eq!(
            mark_vector_index_health(
                &transaction,
                &vector_space_id,
                corrupt.generation(),
                corrupt.canonical_payload_hash(),
                VectorIndexHealthTarget::Unavailable,
                "router.vector.unavailable",
                23,
            )
            .unwrap(),
            VectorIndexHealthMutationAck::Conflict
        );
        assert_eq!(
            mark_vector_index_health(
                &transaction,
                &vector_space_id,
                corrupt.generation(),
                corrupt.canonical_payload_hash(),
                VectorIndexHealthTarget::Corrupt,
                "router.vector.corrupt",
                23,
            )
            .unwrap(),
            VectorIndexHealthMutationAck::AlreadyApplied {
                manifest_hash: corrupt.canonical_payload_hash().to_string(),
            }
        );
        transaction.commit().unwrap();

        let transaction = connection.transaction().unwrap();
        let replacement = match authorize_generation(
            &transaction,
            &vector_space_id,
            VectorDimensions::new(3).unwrap(),
            30,
        )
        .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest) => manifest,
            other => panic!("unexpected replacement authorization: {other:?}"),
        };
        transaction.commit().unwrap();

        let transaction = connection.transaction().unwrap();
        let mut old = load_stored_manifest(&transaction, &vector_space_id, corrupt.generation())
            .unwrap()
            .unwrap();
        let previous_old_hash = old.canonical_payload_hash.clone();
        old.state = VectorIndexManifestState::Retired.as_str().to_string();
        old.retired_at_unix_ms = Some(40);
        old.canonical_payload_hash = manifest_payload_hash(&old).unwrap();
        validate_manifest(&transaction, old.clone()).unwrap();
        assert_eq!(
            update_manifest(&transaction, &old, &previous_old_hash).unwrap(),
            1
        );

        let mut new =
            load_stored_manifest(&transaction, &vector_space_id, replacement.generation())
                .unwrap()
                .unwrap();
        let previous_new_hash = new.canonical_payload_hash.clone();
        new.state = VectorIndexManifestState::Active.as_str().to_string();
        new.source_record_count = Some(0);
        new.source_fingerprint_sha256 = Some("e".repeat(64));
        new.activated_at_unix_ms = Some(40);
        new.canonical_payload_hash = manifest_payload_hash(&new).unwrap();
        validate_manifest(&transaction, new.clone()).unwrap();
        assert_eq!(
            update_manifest(&transaction, &new, &previous_new_hash).unwrap(),
            1
        );
        transaction.commit().unwrap();

        let replacement =
            load_validated_manifest(connection, &vector_space_id, replacement.generation())
                .unwrap()
                .unwrap();
        let replacement_hash = replacement.canonical_payload_hash().to_string();
        let transaction = connection.transaction().unwrap();
        assert_eq!(
            mark_vector_index_health(
                &transaction,
                &vector_space_id,
                corrupt.generation(),
                corrupt.canonical_payload_hash(),
                VectorIndexHealthTarget::Corrupt,
                "router.vector.corrupt",
                41,
            )
            .unwrap(),
            VectorIndexHealthMutationAck::Stale
        );
        assert!(matches!(
            mark_vector_index_health(
                &transaction,
                &vector_space_id,
                replacement.generation(),
                &replacement_hash,
                VectorIndexHealthTarget::Corrupt,
                "router.vector.corrupt",
                41,
            )
            .unwrap(),
            VectorIndexHealthMutationAck::Applied { ref manifest_hash }
                if is_sha256(manifest_hash)
        ));
        transaction.commit().unwrap();

        let missing_space = VectorSpaceId::new("b".repeat(64)).unwrap();
        let transaction = connection.transaction().unwrap();
        assert_eq!(
            mark_vector_index_health(
                &transaction,
                &missing_space,
                VectorIndexGeneration::new(1).unwrap(),
                &"a".repeat(64),
                VectorIndexHealthTarget::Unavailable,
                "router.vector.unavailable",
                41,
            )
            .unwrap(),
            VectorIndexHealthMutationAck::Missing
        );
    }

    #[test]
    fn health_transition_accepts_partial_but_rejects_extra_generated_schema() {
        let mut partial_connection = fixture();
        let partial = authorize(&mut partial_connection);
        partial_connection
            .execute_batch(partial.authority().objects()[1].sql())
            .unwrap();
        mark_active(&partial_connection);
        let active = load_validated_manifest(
            &partial_connection,
            partial.vector_space_id(),
            partial.generation(),
        )
        .unwrap()
        .unwrap();
        let transaction = partial_connection.transaction().unwrap();
        assert!(matches!(
            mark_vector_index_health(
                &transaction,
                active.vector_space_id(),
                active.generation(),
                active.canonical_payload_hash(),
                VectorIndexHealthTarget::Unavailable,
                "router.vector.partial",
                21,
            )
            .unwrap(),
            VectorIndexHealthMutationAck::Applied { .. }
        ));
        transaction.commit().unwrap();

        let mut extra_connection = fixture();
        let complete = authorize(&mut extra_connection);
        create_objects(&mut extra_connection, &complete);
        mark_active(&extra_connection);
        extra_connection
            .execute_batch(&format!(
                "CREATE TABLE \"{}_unexpected\" (value INTEGER)",
                complete.authority().root().as_str()
            ))
            .unwrap();
        let active = load_validated_manifest(
            &extra_connection,
            complete.vector_space_id(),
            complete.generation(),
        )
        .unwrap()
        .unwrap();
        let transaction = extra_connection.transaction().unwrap();
        let error = mark_vector_index_health(
            &transaction,
            active.vector_space_id(),
            active.generation(),
            active.canonical_payload_hash(),
            VectorIndexHealthTarget::Unavailable,
            "router.vector.extra_schema",
            21,
        )
        .unwrap_err();
        assert_eq!(error.class(), LedgerErrorClass::CorruptDatabase);
    }

    #[test]
    fn object_creation_is_fenced_idempotent_and_leaves_partials_recoverable() {
        let mut connection = fixture();
        let manifest = authorize(&mut connection);
        create_objects(&mut connection, &manifest);
        let transaction = connection.transaction().unwrap();
        assert_eq!(
            create_generation_objects_for_manifest(
                &transaction,
                &space_id(),
                manifest.generation(),
            )
            .unwrap(),
            GenerationObjectCreationAck::AlreadyExists
        );
        transaction.commit().unwrap();

        let mut partial_connection = fixture();
        let partial = authorize(&mut partial_connection);
        partial_connection
            .execute_batch(partial.authority().objects()[1].sql())
            .unwrap();
        assert_eq!(
            verify_generation_objects(&partial_connection, partial.authority()).unwrap(),
            GenerationObjectsStatus::Partial
        );
        let transaction = partial_connection.transaction().unwrap();
        assert_eq!(
            create_generation_objects_for_manifest(
                &transaction,
                &space_id(),
                partial.generation(),
            )
            .unwrap(),
            GenerationObjectCreationAck::Partial
        );
        transaction.commit().unwrap();
    }

    #[test]
    fn active_point_crud_is_text_keyed_exact_and_never_updates_conflicts() {
        let mut connection = fixture();
        let manifest = authorize(&mut connection);
        create_objects(&mut connection, &manifest);
        mark_active(&connection);

        let original = record(1, 7, &[1.0, 0.0, 0.0]);
        let transaction = connection.transaction().unwrap();
        assert_eq!(
            upsert_active_record(&transaction, &original).unwrap(),
            VectorIndexPointMutationAck::Applied
        );
        assert_eq!(
            upsert_active_record(&transaction, &original).unwrap(),
            VectorIndexPointMutationAck::AlreadyApplied
        );
        assert_eq!(
            upsert_active_record(&transaction, &record(1, 8, &[1.0, 0.0, 0.0])).unwrap(),
            VectorIndexPointMutationAck::Conflict
        );
        assert_eq!(
            upsert_active_record(&transaction, &record(1, 7, &[0.0, 1.0, 0.0])).unwrap(),
            VectorIndexPointMutationAck::Conflict
        );
        transaction.commit().unwrap();

        let root = manifest.authority().root().as_str();
        let stored_id: String = connection
            .query_row(&format!("SELECT record_id FROM \"{root}\""), [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(stored_id, original.record_id().to_string());
        assert_eq!(
            vector_index_health(&connection, &space_id()).unwrap(),
            SpaceHealth::new(SpaceHealthState::Healthy, 1)
        );

        let transaction = connection.transaction().unwrap();
        assert_eq!(
            delete_active_record(&transaction, &space_id(), original.record_id()).unwrap(),
            VectorIndexPointMutationAck::Applied
        );
        assert_eq!(
            delete_active_record(&transaction, &space_id(), original.record_id()).unwrap(),
            VectorIndexPointMutationAck::AlreadyApplied
        );
        transaction.commit().unwrap();
        assert_eq!(
            vector_index_health(&connection, &space_id())
                .unwrap()
                .record_count(),
            0
        );
    }

    #[test]
    fn checked_vectors_reject_corrupt_blob_dimensions_and_checksum_before_dml() {
        let dimensions = VectorDimensions::new(3).unwrap();
        let checksum = VectorChecksum::new("0".repeat(64)).unwrap();
        assert_eq!(
            AuthoritativeVector::from_blob_verified(
                &space_id(),
                dimensions,
                vec![0; 8],
                checksum.clone(),
            ),
            Err(VectorError::InvalidBlobLength)
        );
        assert_eq!(
            AuthoritativeVector::from_blob_verified(&space_id(), dimensions, vec![0; 12], checksum,),
            Err(VectorError::ChecksumMismatch)
        );
        assert_eq!(
            VectorDimensions::new(8_193),
            Err(VectorError::InvalidDimensions)
        );
    }

    #[test]
    fn source_fingerprint_uses_fixed_width_sorted_binary_tuples() {
        let first = VectorRecordId::new(
            uuid::Uuid::parse_str("01890f47-6c7d-7000-8000-000000000001").unwrap(),
        )
        .unwrap();
        let second = VectorRecordId::new(
            uuid::Uuid::parse_str("01890f47-6c7d-7000-8000-000000000002").unwrap(),
        )
        .unwrap();
        let mut builder = VectorSourceFingerprintBuilder::new();
        builder
            .push(
                first,
                PartitionId::new(1).unwrap(),
                &VectorChecksum::new("00".repeat(32)).unwrap(),
            )
            .unwrap();
        builder
            .push(
                second,
                PartitionId::new(42).unwrap(),
                &VectorChecksum::new("ff".repeat(32)).unwrap(),
            )
            .unwrap();
        let fingerprint = builder.finish();
        assert_eq!(fingerprint.record_count(), 2);
        assert_eq!(
            fingerprint.sha256(),
            "941abb6856dd7ecad415d93bc95acd93b99e4033e875bc866d3e9b6384c4256e"
        );

        let mut out_of_order = VectorSourceFingerprintBuilder::new();
        out_of_order
            .push(
                second,
                PartitionId::new(1).unwrap(),
                &VectorChecksum::new("00".repeat(32)).unwrap(),
            )
            .unwrap();
        assert_eq!(
            out_of_order.push(
                first,
                PartitionId::new(1).unwrap(),
                &VectorChecksum::new("00".repeat(32)).unwrap(),
            ),
            Err(VectorSourceFingerprintError::OutOfOrder)
        );
        assert_eq!(
            decode_checksum(&"A0".repeat(32)),
            Err(VectorSourceFingerprintError::InvalidChecksum)
        );
    }

    #[test]
    fn rebuild_lease_fences_expired_and_dead_owners_monotonically() {
        let mut fixture = activated_fixture(1_000);
        let first_owner = fixture.activated.identity.process_instance_id;
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        let manifest = match authorize_generation(
            &transaction,
            &space_id(),
            VectorDimensions::new(3).unwrap(),
            1_000,
        )
        .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest) => manifest,
            GenerationAuthorizationAck::AlreadyExists(_) => panic!("unexpected manifest"),
            GenerationAuthorizationAck::AuthorityMissing => panic!("missing generation authority"),
        };
        let first =
            match claim_rebuild_lease(&transaction, &space_id(), first_owner, 1_000).unwrap() {
                RebuildLeaseClaimAck::Claimed(fence) => fence,
                other => panic!("unexpected claim: {other:?}"),
            };
        assert_eq!(first.lease_generation(), 1);
        assert_eq!(
            create_generation_objects(&transaction, &first, 1_000).unwrap(),
            GenerationObjectCreationAck::Created
        );
        transaction.commit().unwrap();

        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        assert!(matches!(
            claim_rebuild_lease(&transaction, &space_id(), first_owner, 1_001).unwrap(),
            RebuildLeaseClaimAck::AlreadyOwned(ref current)
                if current.lease_token() == first.lease_token()
        ));
        transaction.commit().unwrap();

        let mut second = LedgerRepository::activate_at(
            &config(&fixture.path, "vector-index-rebuild-fixture"),
            1_002,
        )
        .unwrap();
        let second_owner = second.identity.process_instance_id;
        let transaction = second.repository.connection_mut().transaction().unwrap();
        assert!(matches!(
            claim_rebuild_lease(&transaction, &space_id(), second_owner, 1_003).unwrap(),
            RebuildLeaseClaimAck::Held { .. }
        ));
        transaction.commit().unwrap();

        let transaction = second.repository.connection_mut().transaction().unwrap();
        let reclaimed =
            match claim_rebuild_lease(&transaction, &space_id(), second_owner, 31_000).unwrap() {
                RebuildLeaseClaimAck::Reclaimed(fence) => fence,
                other => panic!("unexpected reclaim: {other:?}"),
            };
        assert_eq!(reclaimed.lease_generation(), 2);
        assert_ne!(reclaimed.lease_token(), first.lease_token());
        assert_eq!(
            populate_rebuild_chunk(&transaction, &first, 31_000).unwrap(),
            RebuildStepAck::Stale
        );
        assert_eq!(
            renew_rebuild_lease(&transaction, &first, 31_000).unwrap(),
            RebuildLeaseMutationAck::Stale
        );
        assert_eq!(
            release_rebuild_lease(&transaction, &reclaimed, 31_001).unwrap(),
            RebuildLeaseMutationAck::Applied
        );
        transaction.commit().unwrap();

        let transaction = second.repository.connection_mut().transaction().unwrap();
        let reclaimed_again =
            match claim_rebuild_lease(&transaction, &space_id(), second_owner, 31_001).unwrap() {
                RebuildLeaseClaimAck::Reclaimed(fence) => fence,
                other => panic!("unexpected post-release reclaim: {other:?}"),
            };
        assert_eq!(reclaimed_again.lease_generation(), 3);
        assert_eq!(manifest.generation(), reclaimed_again.generation());
        transaction.commit().unwrap();
    }

    #[test]
    fn empty_rebuild_flips_immutable_generations_and_cleans_retired_root() {
        let mut fixture = activated_fixture(1_000);
        let owner = fixture.activated.identity.process_instance_id;
        let vector_space_id = mapped_space_id(&fixture);
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        let first_manifest = match authorize_generation(
            &transaction,
            &vector_space_id,
            VectorDimensions::new(3).unwrap(),
            1_000,
        )
        .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest) => manifest,
            other => panic!("unexpected authorization: {other:?}"),
        };
        let first_fence =
            match claim_rebuild_lease(&transaction, &vector_space_id, owner, 1_000).unwrap() {
                RebuildLeaseClaimAck::Claimed(fence) => fence,
                other => panic!("unexpected claim: {other:?}"),
            };
        assert_eq!(
            create_generation_objects(&transaction, &first_fence, 1_000).unwrap(),
            GenerationObjectCreationAck::Created
        );
        assert!(matches!(
            populate_rebuild_chunk(&transaction, &first_fence, 1_001).unwrap(),
            RebuildStepAck::Applied {
                processed: 0,
                complete: true,
                applied_source_seq: 0,
            }
        ));
        assert!(matches!(
            catch_up_rebuild_changes(&transaction, &first_fence, 1_001).unwrap(),
            RebuildStepAck::Applied {
                processed: 0,
                complete: true,
                applied_source_seq: 0,
            }
        ));
        assert_eq!(
            flip_rebuild_generation(&transaction, &first_fence, 1_002).unwrap(),
            RebuildFlipAck::Activated { record_count: 0 }
        );
        transaction.commit().unwrap();

        let reader_connection = Connection::open(&fixture.path).unwrap();
        let reader_transaction = reader_connection.unchecked_transaction().unwrap();
        let first_root = first_manifest.authority().root().as_str().to_string();
        assert_eq!(
            reader_transaction
                .query_row(
                    &format!("SELECT count(*) FROM \"{first_root}\""),
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );

        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        let second_manifest = match authorize_generation(
            &transaction,
            &vector_space_id,
            VectorDimensions::new(3).unwrap(),
            1_003,
        )
        .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest) => manifest,
            other => panic!("unexpected authorization: {other:?}"),
        };
        let second_fence =
            match claim_rebuild_lease(&transaction, &vector_space_id, owner, 1_003).unwrap() {
                RebuildLeaseClaimAck::Claimed(fence) => fence,
                other => panic!("unexpected claim: {other:?}"),
            };
        assert_eq!(
            create_generation_objects(&transaction, &second_fence, 1_003).unwrap(),
            GenerationObjectCreationAck::Created
        );
        assert!(matches!(
            populate_rebuild_chunk(&transaction, &second_fence, 1_004).unwrap(),
            RebuildStepAck::Applied { complete: true, .. }
        ));
        assert_eq!(
            flip_rebuild_generation(&transaction, &second_fence, 1_005).unwrap(),
            RebuildFlipAck::Activated { record_count: 0 }
        );
        transaction.commit().unwrap();

        let connection = fixture.activated.repository.connection_mut();
        let first =
            load_validated_manifest(connection, &vector_space_id, first_manifest.generation())
                .unwrap()
                .unwrap();
        let second =
            load_validated_manifest(connection, &vector_space_id, second_manifest.generation())
                .unwrap()
                .unwrap();
        assert_eq!(first.state(), VectorIndexManifestState::Retired);
        assert_eq!(second.state(), VectorIndexManifestState::Active);

        let first_cleanup: Result<RetiredGenerationCleanupAck, LedgerError> = (|| {
            let transaction = connection.transaction().map_err(database_error)?;
            let acknowledgement = cleanup_retired_generation(
                &transaction,
                &vector_space_id,
                first_manifest.generation(),
                1_006,
            )?;
            transaction.commit().map_err(database_error)?;
            Ok(acknowledgement)
        })();
        assert_eq!(
            reader_transaction
                .query_row(
                    &format!("SELECT count(*) FROM \"{first_root}\""),
                    [],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        reader_transaction.commit().unwrap();
        match first_cleanup {
            Ok(acknowledgement) => {
                assert_eq!(acknowledgement, RetiredGenerationCleanupAck::Dropped);
            }
            Err(error) => {
                assert_eq!(error.class(), LedgerErrorClass::Busy);
                let transaction = connection.transaction().unwrap();
                assert_eq!(
                    cleanup_retired_generation(
                        &transaction,
                        &vector_space_id,
                        first_manifest.generation(),
                        1_007,
                    )
                    .unwrap(),
                    RetiredGenerationCleanupAck::Dropped
                );
                transaction.commit().unwrap();
            }
        }
        let first =
            load_validated_manifest(connection, &vector_space_id, first_manifest.generation())
                .unwrap()
                .unwrap();
        assert_eq!(first.state(), VectorIndexManifestState::Dropped);
        assert_eq!(
            verify_generation_objects(connection, first.authority()).unwrap(),
            GenerationObjectsStatus::Missing
        );
    }

    #[test]
    fn flip_rejects_partial_previous_active_generation() {
        let temporary = tempfile::tempdir().unwrap();
        let path = database_path(&temporary);
        let mut router_config = config(&path, "partial-previous-generation");
        router_config.pools[0].learning = Some(LearningConfig::minimal("embedder-a"));
        let mut activated = LedgerRepository::activate_at(&router_config, 1_000).unwrap();
        let owner = activated.identity.process_instance_id;
        let vector_space_id = activated.identity.pools["pool-a"]
            .vector_space
            .as_ref()
            .unwrap()
            .vector_space_id
            .clone();
        let dimensions = VectorDimensions::new(
            activated
                .repository
                .connection
                .query_row(
                    "SELECT dimensions FROM vector_spaces WHERE vector_space_id = ?1",
                    params![vector_space_id.as_str()],
                    |row| row.get::<_, u32>(0),
                )
                .unwrap(),
        )
        .unwrap();
        let connection = activated.repository.connection_mut();

        let transaction = connection.transaction().unwrap();
        let first = match authorize_generation(&transaction, &vector_space_id, dimensions, 1_000)
            .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest) => manifest,
            other => panic!("unexpected first authorization: {other:?}"),
        };
        let first_fence =
            match claim_rebuild_lease(&transaction, &vector_space_id, owner, 1_000).unwrap() {
                RebuildLeaseClaimAck::Claimed(fence) => fence,
                other => panic!("unexpected first claim: {other:?}"),
            };
        assert_eq!(
            create_generation_objects(&transaction, &first_fence, 1_000).unwrap(),
            GenerationObjectCreationAck::Created
        );
        assert!(matches!(
            populate_rebuild_chunk(&transaction, &first_fence, 1_001).unwrap(),
            RebuildStepAck::Applied { complete: true, .. }
        ));
        assert_eq!(
            flip_rebuild_generation(&transaction, &first_fence, 1_002).unwrap(),
            RebuildFlipAck::Activated { record_count: 0 }
        );
        transaction.commit().unwrap();

        let transaction = connection.transaction().unwrap();
        let second = match authorize_generation(&transaction, &vector_space_id, dimensions, 1_003)
            .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest) => manifest,
            other => panic!("unexpected second authorization: {other:?}"),
        };
        let second_fence =
            match claim_rebuild_lease(&transaction, &vector_space_id, owner, 1_003).unwrap() {
                RebuildLeaseClaimAck::Claimed(fence) => fence,
                other => panic!("unexpected second claim: {other:?}"),
            };
        assert_eq!(
            create_generation_objects(&transaction, &second_fence, 1_003).unwrap(),
            GenerationObjectCreationAck::Created
        );
        assert!(matches!(
            populate_rebuild_chunk(&transaction, &second_fence, 1_004).unwrap(),
            RebuildStepAck::Applied { complete: true, .. }
        ));
        transaction.commit().unwrap();

        connection
            .execute_batch(&format!(
                "DROP TABLE \"{}_vector_chunks00\"",
                first.authority().root().as_str()
            ))
            .unwrap();
        let transaction = connection.transaction().unwrap();
        assert_eq!(
            flip_rebuild_generation(&transaction, &second_fence, 1_005).unwrap(),
            RebuildFlipAck::Unavailable
        );
        transaction.commit().unwrap();
        assert_eq!(
            load_validated_manifest(connection, &vector_space_id, first.generation())
                .unwrap()
                .unwrap()
                .state(),
            VectorIndexManifestState::Active
        );
        assert_eq!(
            load_validated_manifest(connection, &vector_space_id, second.generation())
                .unwrap()
                .unwrap()
                .state(),
            VectorIndexManifestState::Building
        );
    }

    #[test]
    fn rebuild_population_and_deltas_converge_before_fingerprint_flip() {
        let (_temporary, _config, mut activated, vector_space_id, _active_root, first_link_id) =
            ready_retention_runtime_fixture();
        let owner = activated.identity.process_instance_id;
        let first_record_id = VectorRecordId::new(first_link_id).unwrap();
        let first = match load_verified_vector_link_source(
            &activated.repository.connection,
            &vector_space_id,
            first_record_id,
        )
        .unwrap()
        {
            VerifiedVectorLinkSourceLoad::Verified(source) => source.vector_record.unwrap(),
            _ => panic!("ready source did not verify"),
        };
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
        let manifest = match authorize_generation(
            &transaction,
            &vector_space_id,
            VectorDimensions::new(dimensions).unwrap(),
            40,
        )
        .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest) => manifest,
            other => panic!("unexpected authorization: {other:?}"),
        };
        let fence = match claim_rebuild_lease(&transaction, &vector_space_id, owner, 40).unwrap() {
            RebuildLeaseClaimAck::Claimed(fence) => fence,
            other => panic!("unexpected claim: {other:?}"),
        };
        assert_eq!(
            create_generation_objects(&transaction, &fence, 40).unwrap(),
            GenerationObjectCreationAck::Created
        );
        assert!(matches!(
            populate_rebuild_chunk(&transaction, &fence, 41).unwrap(),
            RebuildStepAck::Applied {
                processed: 1,
                complete: true,
                ..
            }
        ));
        transaction.commit().unwrap();

        let root = manifest.authority().root().as_str();
        let transaction = activated.repository.connection_mut().transaction().unwrap();
        transaction
            .execute(
                &format!("DELETE FROM \"{root}\" WHERE record_id = ?1"),
                params![first.record_id().to_string()],
            )
            .unwrap();
        assert_eq!(
            flip_rebuild_generation(&transaction, &fence, 42).unwrap(),
            RebuildFlipAck::FingerprintMismatch
        );
        let current_manifest =
            load_validated_manifest(&transaction, &vector_space_id, manifest.generation())
                .unwrap()
                .unwrap();
        assert_eq!(
            upsert_generation_record(&transaction, &current_manifest, &first).unwrap(),
            VectorIndexPointMutationAck::Applied
        );
        transaction.commit().unwrap();

        let retention = activated
            .repository
            .run_retention(&RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), 44).unwrap())
            .unwrap();
        assert!(matches!(
            retention,
            RetentionAck::Applied { ref summary, .. } if summary.selected_count == 1
        ));
        assert!(
            !activated
                .repository
                .connection
                .query_row(
                    "SELECT EXISTS(SELECT 1 FROM evidence_vector_links
                                   WHERE evidence_vector_link_id = ?1)",
                    params![first_link_id.to_string()],
                    |row| row.get::<_, bool>(0),
                )
                .unwrap()
        );
        let transaction = activated.repository.connection_mut().transaction().unwrap();
        assert!(matches!(
            catch_up_rebuild_changes(&transaction, &fence, 45).unwrap(),
            RebuildStepAck::Applied {
                processed: 1,
                complete: true,
                ..
            }
        ));
        transaction.commit().unwrap();

        let transaction = activated.repository.connection_mut().transaction().unwrap();
        assert_eq!(
            flip_rebuild_generation(&transaction, &fence, 46).unwrap(),
            RebuildFlipAck::Activated { record_count: 0 }
        );
        transaction.commit().unwrap();

        let stored_ids = activated
            .repository
            .connection
            .prepare(&format!(
                "SELECT record_id FROM \"{root}\" ORDER BY record_id"
            ))
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert!(stored_ids.is_empty());
        let fingerprint =
            authoritative_source_fingerprint(&activated.repository.connection, &vector_space_id)
                .unwrap();
        assert_eq!(fingerprint.record_count(), 0);
        let active =
            resolve_active_generation(&activated.repository.connection, &vector_space_id).unwrap();
        assert!(matches!(
            active,
            ActiveGenerationResolution::Active(ref active)
                if active.source_fingerprint_sha256() == Some(fingerprint.sha256())
        ));
    }

    #[test]
    fn valid_partition_retarget_with_stale_link_hash_blocks_rebuild() {
        let (_temporary, _config, mut activated, vector_space_id, _root, link_id) =
            ready_retention_runtime_fixture();
        let original_partition = activated
            .repository
            .connection
            .query_row(
                "SELECT partition_id FROM evidence_vector_links
                 WHERE evidence_vector_link_id = ?1",
                params![link_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        let alternate_partition = clone_partition_with_different_candidate(
            &activated.repository.connection,
            original_partition,
        );
        activated
            .repository
            .connection
            .execute(
                "UPDATE evidence_vector_links SET partition_id = ?1
                 WHERE evidence_vector_link_id = ?2",
                params![alternate_partition, link_id.to_string()],
            )
            .unwrap();

        let error =
            authoritative_source_fingerprint(&activated.repository.connection, &vector_space_id)
                .unwrap_err();
        assert_eq!(error.class(), LedgerErrorClass::CorruptDatabase);

        let dimensions = activated
            .repository
            .connection
            .query_row(
                "SELECT dimensions FROM vector_spaces WHERE vector_space_id = ?1",
                params![vector_space_id.as_str()],
                |row| row.get::<_, u32>(0),
            )
            .unwrap();
        let owner = activated.identity.process_instance_id;
        let transaction = activated.repository.connection_mut().transaction().unwrap();
        let manifest = match authorize_generation(
            &transaction,
            &vector_space_id,
            VectorDimensions::new(dimensions).unwrap(),
            40,
        )
        .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest) => manifest,
            other => panic!("unexpected authorization: {other:?}"),
        };
        let fence = match claim_rebuild_lease(&transaction, &vector_space_id, owner, 40).unwrap() {
            RebuildLeaseClaimAck::Claimed(fence) => fence,
            other => panic!("unexpected rebuild claim: {other:?}"),
        };
        assert_eq!(
            create_generation_objects(&transaction, &fence, 40).unwrap(),
            GenerationObjectCreationAck::Created
        );
        let error = populate_rebuild_chunk(&transaction, &fence, 41).unwrap_err();
        assert_eq!(error.class(), LedgerErrorClass::CorruptDatabase);
        assert_eq!(manifest.state(), VectorIndexManifestState::Building);
        transaction.rollback().unwrap();
        assert!(matches!(
            resolve_active_generation(&activated.repository.connection, &vector_space_id).unwrap(),
            ActiveGenerationResolution::Active(_)
        ));
    }

    #[test]
    fn stale_embedding_hash_blocks_authoritative_fingerprint() {
        let (_temporary, _config, activated, vector_space_id, _root, _link_id) =
            ready_retention_runtime_fixture();
        activated
            .repository
            .connection
            .execute(
                "UPDATE embeddings SET canonical_payload_hash = ?1
                 WHERE vector_space_id = ?2",
                params!["0".repeat(64), vector_space_id.as_str()],
            )
            .unwrap();

        let error =
            authoritative_source_fingerprint(&activated.repository.connection, &vector_space_id)
                .unwrap_err();
        assert_eq!(error.class(), LedgerErrorClass::CorruptDatabase);
    }

    #[test]
    fn retired_partial_generation_is_a_permanent_integrity_blocker() {
        let mut connection = fixture();
        let manifest = authorize(&mut connection);
        create_objects(&mut connection, &manifest);
        mark_active(&connection);
        let transaction = connection.transaction().unwrap();
        let mut stored = load_stored_manifest(&transaction, &space_id(), manifest.generation())
            .unwrap()
            .unwrap();
        let previous_hash = stored.canonical_payload_hash.clone();
        stored.state = VectorIndexManifestState::Retired.as_str().to_string();
        stored.retired_at_unix_ms = Some(30);
        stored.canonical_payload_hash = manifest_payload_hash(&stored).unwrap();
        assert_eq!(
            update_manifest(&transaction, &stored, &previous_hash).unwrap(),
            1
        );
        transaction.commit().unwrap();

        connection
            .execute_batch(&format!(
                "DROP TABLE \"{}_vector_chunks00\"",
                manifest.authority().root().as_str()
            ))
            .unwrap();
        assert_eq!(
            verify_generation_objects(&connection, manifest.authority()).unwrap(),
            GenerationObjectsStatus::Partial
        );
        let transaction = connection.transaction().unwrap();
        assert_eq!(
            cleanup_retired_generation(&transaction, &space_id(), manifest.generation(), 40,)
                .unwrap(),
            RetiredGenerationCleanupAck::Partial
        );
        transaction.commit().unwrap();
        let stored = load_validated_manifest(&connection, &space_id(), manifest.generation())
            .unwrap()
            .unwrap();
        assert_eq!(stored.state(), VectorIndexManifestState::Retired);
    }

    #[test]
    fn retired_generation_with_absent_objects_reconciles_to_dropped() {
        let mut connection = fixture();
        let manifest = authorize(&mut connection);
        create_objects(&mut connection, &manifest);
        mark_active(&connection);
        let transaction = connection.transaction().unwrap();
        let mut stored = load_stored_manifest(&transaction, &space_id(), manifest.generation())
            .unwrap()
            .unwrap();
        let previous_hash = stored.canonical_payload_hash.clone();
        stored.state = VectorIndexManifestState::Retired.as_str().to_string();
        stored.retired_at_unix_ms = Some(30);
        stored.canonical_payload_hash = manifest_payload_hash(&stored).unwrap();
        assert_eq!(
            update_manifest(&transaction, &stored, &previous_hash).unwrap(),
            1
        );
        transaction.commit().unwrap();
        connection
            .execute_batch(manifest.authority().drop_sql())
            .unwrap();

        let transaction = connection.transaction().unwrap();
        assert_eq!(
            cleanup_retired_generation(&transaction, &space_id(), manifest.generation(), 40,)
                .unwrap(),
            RetiredGenerationCleanupAck::ReconciledMissing
        );
        transaction.commit().unwrap();
        assert_eq!(
            load_validated_manifest(&connection, &space_id(), manifest.generation())
                .unwrap()
                .unwrap()
                .state(),
            VectorIndexManifestState::Dropped
        );
    }

    #[test]
    fn cross_process_reader_survives_generation_flip_and_cleanup() {
        const DB_ENV: &str = "NEMO_RELAY_ROUTER_VEC_READER_DB";
        const ROOT_ENV: &str = "NEMO_RELAY_ROUTER_VEC_READER_ROOT";
        const READY_ENV: &str = "NEMO_RELAY_ROUTER_VEC_READER_READY";
        const RELEASE_ENV: &str = "NEMO_RELAY_ROUTER_VEC_READER_RELEASE";

        if let Some(database_path) = std::env::var_os(DB_ENV) {
            assert_eq!(register_sqlite_vec(), SqliteVecStatus::Available);
            let root = std::env::var(ROOT_ENV).unwrap();
            let ready = PathBuf::from(std::env::var_os(READY_ENV).unwrap());
            let release = PathBuf::from(std::env::var_os(RELEASE_ENV).unwrap());
            let connection = Connection::open(database_path).unwrap();
            let transaction = connection.unchecked_transaction().unwrap();
            let count = || {
                transaction
                    .query_row(&format!("SELECT count(*) FROM \"{root}\""), [], |row| {
                        row.get::<_, i64>(0)
                    })
                    .unwrap()
            };
            assert_eq!(count(), 0);
            fs::write(&ready, b"ready").unwrap();
            for _ in 0..500 {
                if release.exists() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            assert!(release.exists());
            assert_eq!(count(), 0);
            transaction.commit().unwrap();
            return;
        }

        let mut fixture = activated_fixture(3_000);
        let owner = fixture.activated.identity.process_instance_id;
        let vector_space_id = mapped_space_id(&fixture);
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        let first_manifest = match authorize_generation(
            &transaction,
            &vector_space_id,
            VectorDimensions::new(3).unwrap(),
            3_000,
        )
        .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest) => manifest,
            other => panic!("unexpected authorization: {other:?}"),
        };
        let first_fence =
            match claim_rebuild_lease(&transaction, &vector_space_id, owner, 3_000).unwrap() {
                RebuildLeaseClaimAck::Claimed(fence) => fence,
                other => panic!("unexpected claim: {other:?}"),
            };
        assert_eq!(
            create_generation_objects(&transaction, &first_fence, 3_000).unwrap(),
            GenerationObjectCreationAck::Created
        );
        assert_eq!(
            flip_rebuild_generation(&transaction, &first_fence, 3_001).unwrap(),
            RebuildFlipAck::Activated { record_count: 0 }
        );
        transaction.commit().unwrap();

        let ready = fixture._temporary.path().join("reader.ready");
        let release = fixture._temporary.path().join("reader.release");
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "ledger::repository::vector_index::tests::cross_process_reader_survives_generation_flip_and_cleanup",
                "--nocapture",
            ])
            .env(DB_ENV, &fixture.path)
            .env(ROOT_ENV, first_manifest.authority().root().as_str())
            .env(READY_ENV, &ready)
            .env(RELEASE_ENV, &release)
            .spawn()
            .unwrap();
        for _ in 0..500 {
            if ready.exists() {
                break;
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(ready.exists());

        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        let second_manifest = match authorize_generation(
            &transaction,
            &vector_space_id,
            VectorDimensions::new(3).unwrap(),
            3_002,
        )
        .unwrap()
        {
            GenerationAuthorizationAck::Created(manifest) => manifest,
            other => panic!("unexpected authorization: {other:?}"),
        };
        let second_fence =
            match claim_rebuild_lease(&transaction, &vector_space_id, owner, 3_002).unwrap() {
                RebuildLeaseClaimAck::Claimed(fence) => fence,
                other => panic!("unexpected claim: {other:?}"),
            };
        assert_eq!(
            create_generation_objects(&transaction, &second_fence, 3_002).unwrap(),
            GenerationObjectCreationAck::Created
        );
        assert_eq!(
            flip_rebuild_generation(&transaction, &second_fence, 3_003).unwrap(),
            RebuildFlipAck::Activated { record_count: 0 }
        );
        transaction.commit().unwrap();
        assert_eq!(second_manifest.generation().value(), 2);

        let first_cleanup: Result<RetiredGenerationCleanupAck, LedgerError> = (|| {
            let transaction = fixture
                .activated
                .repository
                .connection_mut()
                .transaction()
                .map_err(database_error)?;
            let acknowledgement = cleanup_retired_generation(
                &transaction,
                &vector_space_id,
                first_manifest.generation(),
                3_004,
            )?;
            transaction.commit().map_err(database_error)?;
            Ok(acknowledgement)
        })();
        fs::write(&release, b"release").unwrap();
        assert!(child.wait().unwrap().success());
        match first_cleanup {
            Ok(acknowledgement) => {
                assert_eq!(acknowledgement, RetiredGenerationCleanupAck::Dropped);
            }
            Err(error) => {
                assert_eq!(error.class(), LedgerErrorClass::Busy);
                let transaction = fixture
                    .activated
                    .repository
                    .connection_mut()
                    .transaction()
                    .unwrap();
                assert_eq!(
                    cleanup_retired_generation(
                        &transaction,
                        &vector_space_id,
                        first_manifest.generation(),
                        3_005,
                    )
                    .unwrap(),
                    RetiredGenerationCleanupAck::Dropped
                );
                transaction.commit().unwrap();
            }
        }
    }
}
