// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded native-package and ledger diagnostics for first-party hosts.

use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use rusqlite::{Connection, OpenFlags, params};
use serde::Serialize;
use uuid::Uuid;

use crate::ledger::migrations::{CURRENT_SCHEMA_VERSION, LEDGER_APPLICATION_ID};
use crate::ledger::repository::inspect_existing_database;
use crate::sqlite_vec_extension::{
    EXPECTED_VEC_VERSION, SqliteVecStatus, register as register_sqlite_vec,
    verify_connection as verify_sqlite_vec,
};

/// Highest Router ledger schema version understood by this build.
pub const ROUTER_LEDGER_SCHEMA_VERSION: i64 = CURRENT_SCHEMA_VERSION;

/// Read-only classification of an existing Router ledger path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RouterDatabaseSchemaState {
    /// The configured path does not exist.
    Missing,
    /// The path is a readable empty SQLite database.
    Empty,
    /// The ledger matches this build's complete schema.
    Current,
    /// The ledger is valid but requires one or more known migrations.
    UpgradeRequired,
    /// The ledger was created by a newer Router schema.
    Newer,
    /// The file is SQLite data but not a valid Router ledger at its claimed version.
    Incompatible,
    /// The path cannot be opened or its SQLite metadata cannot be read.
    Unreadable,
}

/// Bounded schema metadata returned without migrating or starting Router work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RouterDatabaseSchemaReport {
    /// Stable database classification.
    pub state: RouterDatabaseSchemaState,
    /// SQLite application ID when it could be read.
    pub application_id: Option<i64>,
    /// SQLite user/schema version when it could be read.
    pub schema_version: Option<i64>,
    /// Highest schema version understood by this Router build.
    pub supported_schema_version: i64,
}

impl RouterDatabaseSchemaReport {
    fn unavailable(state: RouterDatabaseSchemaState) -> Self {
        Self {
            state,
            application_id: None,
            schema_version: None,
            supported_schema_version: ROUTER_LEDGER_SCHEMA_VERSION,
        }
    }

    fn observed(
        state: RouterDatabaseSchemaState,
        application_id: i64,
        schema_version: i64,
    ) -> Self {
        Self {
            state,
            application_id: Some(application_id),
            schema_version: Some(schema_version),
            supported_schema_version: ROUTER_LEDGER_SCHEMA_VERSION,
        }
    }
}

/// Successful native SQLite and sqlite-vec package probe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RouterNativeVectorCapability {
    /// Runtime SQLite version linked into this artifact.
    pub sqlite_version: String,
    /// Runtime sqlite-vec version linked into this artifact.
    pub vector_version: String,
}

/// Stable failure classes for the native vector package probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouterNativeVectorCapabilityError {
    /// A private temporary probe directory could not be created.
    TemporaryDirectory,
    /// The statically linked sqlite-vec extension could not be registered.
    Registration,
    /// The temporary SQLite database could not be opened.
    DatabaseOpen,
    /// The runtime sqlite-vec version was missing or did not match the pin.
    Version,
    /// The pinned `vec0` schema could not be created.
    Schema,
    /// A fixed probe vector could not be inserted.
    Insert,
    /// The fixed nearest-neighbor query could not be completed.
    Query,
    /// The query returned a result other than the deterministic nearest row.
    UnexpectedNeighbor,
    /// Probe files could not be removed after the SQLite connection closed.
    Cleanup,
}

impl RouterNativeVectorCapabilityError {
    /// Returns the stable diagnostic code for this failure class.
    pub const fn code(self) -> &'static str {
        match self {
            Self::TemporaryDirectory => "router.native_vector.temporary_directory",
            Self::Registration => "router.native_vector.registration",
            Self::DatabaseOpen => "router.native_vector.database_open",
            Self::Version => "router.native_vector.version",
            Self::Schema => "router.native_vector.schema",
            Self::Insert => "router.native_vector.insert",
            Self::Query => "router.native_vector.query",
            Self::UnexpectedNeighbor => "router.native_vector.unexpected_neighbor",
            Self::Cleanup => "router.native_vector.cleanup",
        }
    }
}

impl fmt::Display for RouterNativeVectorCapabilityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

impl Error for RouterNativeVectorCapabilityError {}

/// Inspects an existing Router database without migrating or modifying it.
pub fn inspect_router_database(path: impl AsRef<Path>) -> RouterDatabaseSchemaReport {
    let path = path.as_ref();
    if !path.exists() {
        return RouterDatabaseSchemaReport::unavailable(RouterDatabaseSchemaState::Missing);
    }

    let flags = OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX;
    let Ok(connection) = Connection::open_with_flags(path, flags) else {
        return RouterDatabaseSchemaReport::unavailable(RouterDatabaseSchemaState::Unreadable);
    };
    let metadata = (|| {
        let application_id = connection.query_row("PRAGMA application_id", [], |row| row.get(0))?;
        let schema_version = connection.query_row("PRAGMA user_version", [], |row| row.get(0))?;
        let user_tables = connection.query_row(
            "SELECT count(*) FROM sqlite_schema \
             WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        Ok::<_, rusqlite::Error>((application_id, schema_version, user_tables))
    })();
    let Ok((application_id, schema_version, user_tables)) = metadata else {
        return RouterDatabaseSchemaReport::unavailable(RouterDatabaseSchemaState::Unreadable);
    };

    if application_id == 0 && schema_version == 0 && user_tables == 0 {
        return RouterDatabaseSchemaReport::observed(
            RouterDatabaseSchemaState::Empty,
            application_id,
            schema_version,
        );
    }
    if application_id != i64::from(LEDGER_APPLICATION_ID)
        || (application_id == 0 && user_tables != 0)
    {
        return RouterDatabaseSchemaReport::observed(
            RouterDatabaseSchemaState::Incompatible,
            application_id,
            schema_version,
        );
    }
    if schema_version > CURRENT_SCHEMA_VERSION {
        return RouterDatabaseSchemaReport::observed(
            RouterDatabaseSchemaState::Newer,
            application_id,
            schema_version,
        );
    }
    if inspect_existing_database(&connection).is_err() {
        return RouterDatabaseSchemaReport::observed(
            RouterDatabaseSchemaState::Incompatible,
            application_id,
            schema_version,
        );
    }
    let state = if schema_version < CURRENT_SCHEMA_VERSION {
        RouterDatabaseSchemaState::UpgradeRequired
    } else {
        RouterDatabaseSchemaState::Current
    };
    RouterDatabaseSchemaReport::observed(state, application_id, schema_version)
}

/// Verifies this artifact's statically linked SQLite and sqlite-vec runtime.
///
/// The probe creates and removes a private temporary directory, opens a file
/// database, creates a fixed `vec0` table, inserts two vectors, and verifies the
/// deterministic nearest neighbor. It performs no network or provider I/O.
pub fn probe_native_vector_capability()
-> Result<RouterNativeVectorCapability, RouterNativeVectorCapabilityError> {
    let directory = create_probe_directory()?;
    let database_path = directory.join("probe.sqlite3");
    let result = run_native_vector_probe(&database_path);
    let cleanup =
        fs::remove_dir_all(&directory).map_err(|_| RouterNativeVectorCapabilityError::Cleanup);
    match (result, cleanup) {
        (Ok(report), Ok(())) => Ok(report),
        (Err(error), _) => Err(error),
        (Ok(_), Err(error)) => Err(error),
    }
}

fn create_probe_directory() -> Result<PathBuf, RouterNativeVectorCapabilityError> {
    let temporary_root = std::env::temp_dir();
    for _ in 0..4 {
        let path = temporary_root.join(format!(
            "nemo-relay-router-native-probe-{}-{}",
            std::process::id(),
            Uuid::now_v7()
        ));
        match fs::create_dir(&path) {
            Ok(()) => return Ok(path),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(_) => return Err(RouterNativeVectorCapabilityError::TemporaryDirectory),
        }
    }
    Err(RouterNativeVectorCapabilityError::TemporaryDirectory)
}

fn run_native_vector_probe(
    database_path: &Path,
) -> Result<RouterNativeVectorCapability, RouterNativeVectorCapabilityError> {
    if register_sqlite_vec() != SqliteVecStatus::Available {
        return Err(RouterNativeVectorCapabilityError::Registration);
    }
    let connection = Connection::open(database_path)
        .map_err(|_| RouterNativeVectorCapabilityError::DatabaseOpen)?;
    if verify_sqlite_vec(&connection) != SqliteVecStatus::Available {
        return Err(RouterNativeVectorCapabilityError::Version);
    }
    connection
        .execute_batch(
            "CREATE VIRTUAL TABLE native_vector_probe USING vec0(
                record_id TEXT PRIMARY KEY,
                embedding FLOAT[2] distance_metric=cosine,
                partition_id INTEGER PARTITION KEY
             );",
        )
        .map_err(|_| RouterNativeVectorCapabilityError::Schema)?;
    let insert = "INSERT INTO native_vector_probe(record_id, embedding, partition_id) \
                  VALUES (?1, ?2, 1)";
    for (record_id, vector) in [("near", [1.0_f32, 0.0]), ("far", [0.0_f32, 1.0])] {
        connection
            .execute(insert, params![record_id, vector_blob(&vector)])
            .map_err(|_| RouterNativeVectorCapabilityError::Insert)?;
    }
    let neighbor = connection
        .query_row(
            "SELECT record_id FROM native_vector_probe \
             WHERE embedding MATCH ?1 AND k = 1 AND partition_id = 1 \
             ORDER BY distance",
            params![vector_blob(&[1.0_f32, 0.0])],
            |row| row.get::<_, String>(0),
        )
        .map_err(|_| RouterNativeVectorCapabilityError::Query)?;
    if neighbor != "near" {
        return Err(RouterNativeVectorCapabilityError::UnexpectedNeighbor);
    }
    let sqlite_version = connection
        .query_row("SELECT sqlite_version()", [], |row| row.get::<_, String>(0))
        .map_err(|_| RouterNativeVectorCapabilityError::Query)?;
    let vector_version = connection
        .query_row("SELECT vec_version()", [], |row| row.get::<_, String>(0))
        .map_err(|_| RouterNativeVectorCapabilityError::Query)?;
    if vector_version != EXPECTED_VEC_VERSION {
        return Err(RouterNativeVectorCapabilityError::Version);
    }
    drop(connection);
    Ok(RouterNativeVectorCapability {
        sqlite_version,
        vector_version,
    })
}

fn vector_blob(values: &[f32]) -> Vec<u8> {
    let mut blob = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        blob.extend_from_slice(&value.to_ne_bytes());
    }
    blob
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use rusqlite::Connection;
    use tempfile::tempdir;

    use super::{
        ROUTER_LEDGER_SCHEMA_VERSION, RouterDatabaseSchemaState, inspect_router_database,
        probe_native_vector_capability,
    };
    use crate::RouterConfig;
    use crate::ledger::migrations::{LEDGER_APPLICATION_ID, MIGRATIONS};
    use crate::ledger::repository::LedgerRepository;

    #[test]
    fn native_vector_probe_is_repeatable_and_reports_pinned_versions() {
        for _ in 0..2 {
            let report = probe_native_vector_capability().unwrap();
            assert_eq!(report.sqlite_version, rusqlite::version());
            assert_eq!(report.vector_version, "v0.1.9");
        }
    }

    #[test]
    fn database_inspection_classifies_missing_empty_and_unreadable_paths() {
        let temporary = tempdir().unwrap();
        let missing = temporary.path().join("missing.sqlite3");
        assert_eq!(
            inspect_router_database(&missing).state,
            RouterDatabaseSchemaState::Missing
        );

        let empty = temporary.path().join("empty.sqlite3");
        File::create(&empty).unwrap();
        assert_eq!(
            inspect_router_database(&empty).state,
            RouterDatabaseSchemaState::Empty
        );
        assert_eq!(
            inspect_router_database(temporary.path()).state,
            RouterDatabaseSchemaState::Unreadable
        );
    }

    #[test]
    fn database_inspection_classifies_current_upgrade_newer_and_incompatible() {
        let temporary = tempdir().unwrap();

        let current = temporary.path().join("current.sqlite3");
        let current_config = RouterConfig {
            database_path: current.to_string_lossy().into_owned(),
            ..RouterConfig::default()
        };
        drop(LedgerRepository::activate(&current_config).unwrap());
        let report = inspect_router_database(&current);
        assert_eq!(report.state, RouterDatabaseSchemaState::Current);
        assert_eq!(report.schema_version, Some(ROUTER_LEDGER_SCHEMA_VERSION));

        let older = temporary.path().join("older.sqlite3");
        let connection = Connection::open(&older).unwrap();
        connection.execute_batch(MIGRATIONS[0].sql).unwrap();
        connection
            .execute(
                "INSERT INTO schema_migrations (
                    version, name, checksum_sha256, applied_at_unix_ms,
                    application_version, sqlite_version
                 ) VALUES (?1, ?2, ?3, 0, 'test', ?4)",
                rusqlite::params![
                    MIGRATIONS[0].version,
                    MIGRATIONS[0].name,
                    MIGRATIONS[0].checksum_sha256,
                    rusqlite::version(),
                ],
            )
            .unwrap();
        connection
            .pragma_update(None, "user_version", MIGRATIONS[0].version)
            .unwrap();
        drop(connection);
        assert_eq!(
            inspect_router_database(&older).state,
            RouterDatabaseSchemaState::UpgradeRequired
        );

        let newer = temporary.path().join("newer.sqlite3");
        let connection = Connection::open(&newer).unwrap();
        connection
            .pragma_update(None, "application_id", LEDGER_APPLICATION_ID)
            .unwrap();
        connection
            .pragma_update(None, "user_version", ROUTER_LEDGER_SCHEMA_VERSION + 1)
            .unwrap();
        drop(connection);
        assert_eq!(
            inspect_router_database(&newer).state,
            RouterDatabaseSchemaState::Newer
        );

        let incompatible = temporary.path().join("incompatible.sqlite3");
        let connection = Connection::open(&incompatible).unwrap();
        connection
            .execute_batch("CREATE TABLE unrelated(value INTEGER);")
            .unwrap();
        drop(connection);
        assert_eq!(
            inspect_router_database(&incompatible).state,
            RouterDatabaseSchemaState::Incompatible
        );
    }

    #[test]
    fn database_inspection_rejects_corrupt_current_migration_history() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("corrupt.sqlite3");
        let config = RouterConfig {
            database_path: path.to_string_lossy().into_owned(),
            ..RouterConfig::default()
        };
        drop(LedgerRepository::activate(&config).unwrap());
        let connection = Connection::open(&path).unwrap();
        connection
            .execute(
                "UPDATE schema_migrations
                 SET checksum_sha256 = '0000000000000000000000000000000000000000000000000000000000000000'
                 WHERE version = 1",
                [],
            )
            .unwrap();
        drop(connection);
        assert_eq!(
            inspect_router_database(&path).state,
            RouterDatabaseSchemaState::Incompatible
        );
    }
}
