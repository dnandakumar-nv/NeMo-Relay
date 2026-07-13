// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validated sqlite-vec generation names and exact 0.1.9 schema authority.

#![allow(dead_code)] // Task 6 repository commands consume this schema authority.

use std::fmt;

use serde_json::{Value as Json, json};

use crate::canonical_json::canonical_json;
use crate::fingerprint::sha256_hex;
use crate::vector::{VectorDimensions, VectorSpaceId};

pub(crate) const VEC0_SCHEMA_OBJECT_COUNT: usize = 5;

const ROOT_PREFIX: &str = "router_vec_";
const GENERATION_SEPARATOR: &str = "_g";
const TABLE_OBJECT_TYPE: &str = "table";

/// Stable failures for generation-name and schema-authority construction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SqliteVecSchemaError {
    InvalidGeneration,
    InvalidRootName,
}

/// Positive sqlite-vec index generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct VectorIndexGeneration(i64);

impl VectorIndexGeneration {
    pub(crate) fn new(value: i64) -> Result<Self, SqliteVecSchemaError> {
        if value <= 0 {
            return Err(SqliteVecSchemaError::InvalidGeneration);
        }
        Ok(Self(value))
    }

    pub(crate) const fn value(self) -> i64 {
        self.0
    }
}

/// Exact generated vec0 root name derived from typed authority.
#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct Vec0RootName {
    vector_space_id: VectorSpaceId,
    generation: VectorIndexGeneration,
    value: String,
}

impl Vec0RootName {
    pub(crate) fn new(vector_space_id: VectorSpaceId, generation: VectorIndexGeneration) -> Self {
        let value = format!(
            "{ROOT_PREFIX}{}{GENERATION_SEPARATOR}{}",
            vector_space_id.as_str(),
            generation.value()
        );
        Self {
            vector_space_id,
            generation,
            value,
        }
    }

    /// Parse only the exact canonical grammar emitted by [`Self::new`].
    pub(crate) fn parse(value: &str) -> Result<Self, SqliteVecSchemaError> {
        if !value.is_ascii() {
            return Err(SqliteVecSchemaError::InvalidRootName);
        }
        let remainder = value
            .strip_prefix(ROOT_PREFIX)
            .ok_or(SqliteVecSchemaError::InvalidRootName)?;
        let vector_space_text = remainder
            .get(..64)
            .ok_or(SqliteVecSchemaError::InvalidRootName)?;
        let generation_text = remainder
            .get(64..)
            .and_then(|suffix| suffix.strip_prefix(GENERATION_SEPARATOR))
            .ok_or(SqliteVecSchemaError::InvalidRootName)?;
        if generation_text.is_empty()
            || generation_text.starts_with('0')
            || !generation_text.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(SqliteVecSchemaError::InvalidRootName);
        }
        let generation_value = generation_text
            .parse::<i64>()
            .map_err(|_| SqliteVecSchemaError::InvalidRootName)?;
        let generation = VectorIndexGeneration::new(generation_value)
            .map_err(|_| SqliteVecSchemaError::InvalidRootName)?;
        let vector_space_id = VectorSpaceId::new(vector_space_text.to_string())
            .map_err(|_| SqliteVecSchemaError::InvalidRootName)?;
        let parsed = Self::new(vector_space_id, generation);
        if parsed.as_str() != value {
            return Err(SqliteVecSchemaError::InvalidRootName);
        }
        Ok(parsed)
    }

    pub(crate) fn vector_space_id(&self) -> &VectorSpaceId {
        &self.vector_space_id
    }

    pub(crate) const fn generation(&self) -> VectorIndexGeneration {
        self.generation
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.value
    }
}

impl fmt::Debug for Vec0RootName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_tuple("Vec0RootName")
            .field(&self.value)
            .finish()
    }
}

impl fmt::Display for Vec0RootName {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.value)
    }
}

/// One exact sqlite_schema tuple authorized by a generation manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ExpectedVec0SchemaObject {
    object_type: &'static str,
    name: String,
    table_name: String,
    sql: String,
    sql_sha256: String,
}

impl ExpectedVec0SchemaObject {
    pub(crate) const fn object_type(&self) -> &'static str {
        self.object_type
    }

    pub(crate) fn name(&self) -> &str {
        &self.name
    }

    pub(crate) fn table_name(&self) -> &str {
        &self.table_name
    }

    pub(crate) fn sql(&self) -> &str {
        &self.sql
    }

    pub(crate) fn sql_sha256(&self) -> &str {
        &self.sql_sha256
    }

    fn manifest_value(&self) -> Json {
        json!({
            "type": self.object_type,
            "name": self.name,
            "table_name": self.table_name,
            "sql_sha256": self.sql_sha256,
        })
    }
}

/// Exact DDL and manifest authority for one immutable sqlite-vec generation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Vec0SchemaAuthority {
    root: Vec0RootName,
    dimensions: VectorDimensions,
    objects: Vec<ExpectedVec0SchemaObject>,
    manifest_json: String,
    manifest_sha256: String,
    drop_sql: String,
}

impl Vec0SchemaAuthority {
    pub(crate) fn new(root: Vec0RootName, dimensions: VectorDimensions) -> Self {
        let objects = expected_schema_objects(&root, dimensions);
        let manifest_value = Json::Array(
            objects
                .iter()
                .map(ExpectedVec0SchemaObject::manifest_value)
                .collect(),
        );
        let manifest_json = canonical_json(&manifest_value)
            .expect("fixed sqlite-vec schema authority is canonical JSON");
        let manifest_sha256 = sha256_hex(manifest_json.as_bytes());
        let drop_sql = format!("DROP TABLE \"{}\"", root.as_str());
        Self {
            root,
            dimensions,
            objects,
            manifest_json,
            manifest_sha256,
            drop_sql,
        }
    }

    pub(crate) fn root(&self) -> &Vec0RootName {
        &self.root
    }

    pub(crate) const fn dimensions(&self) -> VectorDimensions {
        self.dimensions
    }

    pub(crate) fn create_sql(&self) -> &str {
        self.objects[0].sql()
    }

    pub(crate) fn drop_sql(&self) -> &str {
        &self.drop_sql
    }

    pub(crate) fn objects(&self) -> &[ExpectedVec0SchemaObject] {
        &self.objects
    }

    pub(crate) fn manifest_json(&self) -> &str {
        &self.manifest_json
    }

    pub(crate) fn manifest_sha256(&self) -> &str {
        &self.manifest_sha256
    }
}

fn expected_schema_objects(
    root: &Vec0RootName,
    dimensions: VectorDimensions,
) -> Vec<ExpectedVec0SchemaObject> {
    let root = root.as_str();
    let definitions = [
        (
            root.to_string(),
            format!(
                "CREATE VIRTUAL TABLE \"{root}\" USING vec0(record_id TEXT PRIMARY KEY, \
                 embedding FLOAT[{}] distance_metric=cosine, partition_id INTEGER PARTITION KEY)",
                dimensions.value()
            ),
        ),
        (
            format!("{root}_chunks"),
            format!(
                "CREATE TABLE \"{root}_chunks\"(chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,\
                 size INTEGER NOT NULL,sequence_id integer,partition00,validity BLOB NOT NULL, \
                 rowids BLOB NOT NULL)"
            ),
        ),
        (
            format!("{root}_info"),
            format!("CREATE TABLE \"{root}_info\" (key text primary key, value any)"),
        ),
        (
            format!("{root}_rowids"),
            format!(
                "CREATE TABLE \"{root}_rowids\"(rowid INTEGER PRIMARY KEY AUTOINCREMENT,\
                 id TEXT UNIQUE NOT NULL,chunk_id INTEGER,chunk_offset INTEGER)"
            ),
        ),
        (
            format!("{root}_vector_chunks00"),
            format!(
                "CREATE TABLE \"{root}_vector_chunks00\"(rowid PRIMARY KEY,vectors BLOB NOT NULL)"
            ),
        ),
    ];

    definitions
        .into_iter()
        .map(|(name, sql)| ExpectedVec0SchemaObject {
            object_type: TABLE_OBJECT_TYPE,
            table_name: name.clone(),
            name,
            sql_sha256: sha256_hex(sql.as_bytes()),
            sql,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_root() -> Vec0RootName {
        Vec0RootName::new(
            VectorSpaceId::new("a".repeat(64)).unwrap(),
            VectorIndexGeneration::new(1).unwrap(),
        )
    }

    fn fixture_authority() -> Vec0SchemaAuthority {
        Vec0SchemaAuthority::new(fixture_root(), VectorDimensions::new(3).unwrap())
    }

    #[test]
    fn generation_and_root_grammar_are_exact_and_canonical() {
        assert_eq!(
            VectorIndexGeneration::new(0),
            Err(SqliteVecSchemaError::InvalidGeneration)
        );
        assert_eq!(
            VectorIndexGeneration::new(-1),
            Err(SqliteVecSchemaError::InvalidGeneration)
        );

        let root = fixture_root();
        assert_eq!(
            root.as_str(),
            "router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1"
        );
        assert_eq!(Vec0RootName::parse(root.as_str()).unwrap(), root);
        assert_eq!(root.generation().value(), 1);
        assert_eq!(root.vector_space_id().as_str(), "a".repeat(64));

        let maximum = Vec0RootName::new(
            VectorSpaceId::new("f".repeat(64)).unwrap(),
            VectorIndexGeneration::new(i64::MAX).unwrap(),
        );
        assert_eq!(maximum.as_str().len(), 96);
        assert_eq!(Vec0RootName::parse(maximum.as_str()).unwrap(), maximum);
    }

    #[test]
    fn root_parser_rejects_every_injection_surface() {
        let hash = "a".repeat(64);
        for invalid in [
            String::new(),
            format!("router_vec_{hash}_g0"),
            format!("router_vec_{hash}_g01"),
            format!("router_vec_{hash}_g-1"),
            format!("router_vec_{}_g1", "A".repeat(64)),
            format!("router_vec_{hash}_g1;DROP TABLE vector_spaces"),
            format!("router_vec_{hash}_g1\""),
            format!("router_vec_{hash}_g9223372036854775808"),
        ] {
            assert_eq!(
                Vec0RootName::parse(&invalid),
                Err(SqliteVecSchemaError::InvalidRootName),
                "accepted invalid root: {invalid:?}"
            );
        }
        assert!(VectorSpaceId::new(format!("{};DROP", "a".repeat(64))).is_err());
    }

    #[test]
    fn create_and_drop_ddl_use_only_the_validated_root_and_dimensions() {
        let authority = fixture_authority();
        assert_eq!(
            authority.create_sql(),
            "CREATE VIRTUAL TABLE \"router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1\" USING vec0(record_id TEXT PRIMARY KEY, embedding FLOAT[3] distance_metric=cosine, partition_id INTEGER PARTITION KEY)"
        );
        assert_eq!(
            authority.drop_sql(),
            "DROP TABLE \"router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1\""
        );
        assert_eq!(authority.dimensions().value(), 3);
        assert_eq!(authority.root(), &fixture_root());
    }

    #[test]
    fn pinned_019_schema_strings_and_hashes_are_exact() {
        let authority = fixture_authority();
        let expected = [
            (
                "router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1",
                "CREATE VIRTUAL TABLE \"router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1\" USING vec0(record_id TEXT PRIMARY KEY, embedding FLOAT[3] distance_metric=cosine, partition_id INTEGER PARTITION KEY)",
                "d77ed4b97150e04f41b07ed1a5aa3eed8a69eff0f0a685c18e3886921dbc47e7",
            ),
            (
                "router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1_chunks",
                "CREATE TABLE \"router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1_chunks\"(chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,size INTEGER NOT NULL,sequence_id integer,partition00,validity BLOB NOT NULL, rowids BLOB NOT NULL)",
                "799ec3810b6a100fb7126f71e5b92f9d5bda6e92b03b568530fb5829ae9cd82b",
            ),
            (
                "router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1_info",
                "CREATE TABLE \"router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1_info\" (key text primary key, value any)",
                "473a406dd6370745c39c6fba04cf80612f490e847e52d804065f6ebb5416f465",
            ),
            (
                "router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1_rowids",
                "CREATE TABLE \"router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1_rowids\"(rowid INTEGER PRIMARY KEY AUTOINCREMENT,id TEXT UNIQUE NOT NULL,chunk_id INTEGER,chunk_offset INTEGER)",
                "2ba1913ced9f22e74d72a117af694db38d9d4c12a86320ec031940efae8d94c7",
            ),
            (
                "router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1_vector_chunks00",
                "CREATE TABLE \"router_vec_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa_g1_vector_chunks00\"(rowid PRIMARY KEY,vectors BLOB NOT NULL)",
                "f4d57f6467d8adec0c487a9c1ff0a0252bd9739c2e243001c642ccece5e524f8",
            ),
        ];
        assert_eq!(authority.objects().len(), VEC0_SCHEMA_OBJECT_COUNT);
        for (actual, (name, sql, hash)) in authority.objects().iter().zip(expected) {
            assert_eq!(actual.object_type(), "table");
            assert_eq!(actual.name(), name);
            assert_eq!(actual.table_name(), name);
            assert_eq!(actual.sql(), sql);
            assert_eq!(actual.sql_sha256(), hash);
        }
    }

    #[test]
    fn manifest_json_and_hash_are_canonical_and_pinned() {
        let authority = fixture_authority();
        let parsed: Json = serde_json::from_str(authority.manifest_json()).unwrap();
        assert_eq!(canonical_json(&parsed).unwrap(), authority.manifest_json());
        assert_eq!(
            sha256_hex(authority.manifest_json().as_bytes()),
            authority.manifest_sha256()
        );
        assert_eq!(
            authority.manifest_sha256(),
            "4ea9b3b32b92e41313a7ac5a58b6b4536990d9f1d70ec5e353c9087615bd1fbd"
        );
        assert_eq!(parsed.as_array().unwrap().len(), VEC0_SCHEMA_OBJECT_COUNT);
    }

    #[test]
    fn dimension_and_generation_change_every_derived_authority_surface() {
        let base = fixture_authority();
        let different_dimension =
            Vec0SchemaAuthority::new(fixture_root(), VectorDimensions::new(4).unwrap());
        let different_generation = Vec0SchemaAuthority::new(
            Vec0RootName::new(
                VectorSpaceId::new("a".repeat(64)).unwrap(),
                VectorIndexGeneration::new(2).unwrap(),
            ),
            VectorDimensions::new(3).unwrap(),
        );

        assert_ne!(base.create_sql(), different_dimension.create_sql());
        assert_ne!(base.manifest_json(), different_dimension.manifest_json());
        assert_ne!(
            base.manifest_sha256(),
            different_dimension.manifest_sha256()
        );
        assert_ne!(base.root(), different_generation.root());
        assert_ne!(base.manifest_json(), different_generation.manifest_json());
        assert_ne!(
            base.manifest_sha256(),
            different_generation.manifest_sha256()
        );
    }
}
