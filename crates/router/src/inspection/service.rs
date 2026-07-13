// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::sync::watch;

use crate::config::RouterConfig;
use crate::embedder::FrozenEmbedderClients;
use crate::host::{RouterDatabaseSchemaState, inspect_router_database};
use crate::inspection::cursor::CursorCodecV1;
use crate::inspection::{
    ContentPolicy, DatabaseStatusV1, InspectionDatabaseStateV1, InspectionError,
    InspectionServiceOptions,
};
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::ledger::read_pool::{LedgerReadPool, ReadPoolError, ReadPoolErrorClass};
use crate::ledger::repository::inspection::{
    InspectionAuthoritySnapshot, load_inspection_authority_snapshot,
};
use crate::sqlite_vec_extension::{SqliteVecStatus, register as register_sqlite_vec};

mod neighborhood;
mod operations;
mod reads;
mod request;

/// Cloneable async access to one configured Router ledger.
#[derive(Clone)]
pub struct InspectionService {
    inner: Arc<InspectionServiceInner>,
}

struct InspectionServiceInner {
    #[allow(dead_code)] // Tasks 3-6 use the resolved config for typed reads.
    config: Arc<RouterConfig>,
    options: InspectionServiceOptions,
    database: DatabaseStatusV1,
    current: Option<CurrentInspectionBackend>,
    operations: Option<operations::OperationsBackend>,
    embedder_clients: Mutex<BTreeMap<String, Arc<FrozenEmbedderClients>>>,
    cancellation: watch::Sender<bool>,
}

#[allow(dead_code)] // Tasks 3-6 consume the verified authority and cursor codec.
pub(super) struct CurrentInspectionBackend {
    pub(super) read_pool: LedgerReadPool,
    pub(super) authority: InspectionAuthoritySnapshot,
    pub(super) cursor: CursorCodecV1,
}

impl InspectionService {
    /// Open an existing Router ledger without creating, migrating, or activating it.
    pub async fn open(
        config: RouterConfig,
        options: InspectionServiceOptions,
    ) -> Result<Self, InspectionError> {
        options.validate()?;
        config
            .config_generation_id()
            .map_err(|_| InspectionError::InvalidArgument)?;
        let report = inspect_router_database(Path::new(&config.database_path));
        let database = DatabaseStatusV1 {
            state: map_database_state(report.state),
            application_id: report.application_id,
            schema_version: report.schema_version,
            supported_schema_version: report.supported_schema_version,
        };
        let (cancellation, _) = watch::channel(false);
        if report.state != RouterDatabaseSchemaState::Current {
            return Ok(Self {
                inner: Arc::new(InspectionServiceInner {
                    config: Arc::new(config),
                    options,
                    database,
                    current: None,
                    operations: None,
                    embedder_clients: Mutex::new(BTreeMap::new()),
                    cancellation,
                }),
            });
        }

        if register_sqlite_vec() != SqliteVecStatus::Available {
            return Err(InspectionError::StorageUnavailable);
        }
        let read_pool =
            LedgerReadPool::open(Path::new(&config.database_path)).map_err(map_read_pool_error)?;
        let deadline = inspection_deadline(options.request_timeout_ms)?;
        let authority_config = config.clone();
        let authority = match read_pool
            .run(deadline, move |connection| {
                Ok(load_inspection_authority_snapshot(
                    connection,
                    &authority_config,
                ))
            })
            .await
        {
            Ok(Ok(authority)) => authority,
            Ok(Err(error)) => {
                read_pool.abort();
                return Err(map_ledger_error(error));
            }
            Err(error) => {
                read_pool.abort();
                return Err(map_read_pool_error(error));
            }
        };
        let cursor = CursorCodecV1::new(
            authority.cursor_key.clone(),
            report.supported_schema_version,
            authority.project_uuid,
            authority.cohort_generation_id,
        )?;
        let operations = if options.allow_operations {
            match operations::OperationsBackend::open(
                &config,
                &read_pool,
                &authority,
                options.request_timeout_ms,
            )
            .await
            {
                Ok(backend) => Some(backend),
                Err(error) => {
                    read_pool.abort();
                    return Err(error);
                }
            }
        } else {
            None
        };
        Ok(Self {
            inner: Arc::new(InspectionServiceInner {
                config: Arc::new(config),
                options,
                database,
                current: Some(CurrentInspectionBackend {
                    read_pool,
                    authority,
                    cursor,
                }),
                operations,
                embedder_clients: Mutex::new(BTreeMap::new()),
                cancellation,
            }),
        })
    }

    /// Return the non-secret database classification captured at open time.
    pub fn database_status(&self) -> &DatabaseStatusV1 {
        &self.inner.database
    }

    /// Return the host-fixed effective content policy.
    pub fn content_policy(&self) -> ContentPolicy {
        self.inner.options.content_policy
    }

    /// Return whether this host permits provider-backed request inspection.
    pub fn request_embedding_enabled(&self) -> bool {
        self.inner.options.allow_request_embedding
    }

    /// Return whether this host fixed mutation capability at construction time.
    pub fn operations_enabled(&self) -> bool {
        self.inner.options.allow_operations
    }

    #[cfg(feature = "http")]
    pub(crate) fn request_timeout_ms(&self) -> u64 {
        self.inner.options.request_timeout_ms
    }

    #[cfg(feature = "http")]
    pub(crate) fn cancellation_receiver(&self) -> watch::Receiver<bool> {
        self.inner.cancellation.subscribe()
    }

    pub(super) fn export_chunk_bytes(&self) -> usize {
        self.inner.options.export_chunk_bytes
    }

    /// Drain active reads and close every read-only connection by the host deadline.
    pub async fn close(&self) -> Result<(), InspectionError> {
        self.inner.cancellation.send_replace(true);
        self.inner
            .embedder_clients
            .lock()
            .map_err(|_| InspectionError::IntegrityError)?
            .clear();
        let deadline = inspection_deadline(self.inner.options.request_timeout_ms)?;
        let operations_result = match &self.inner.operations {
            Some(operations) => operations.close(deadline).await,
            None => Ok(()),
        };
        let Some(current) = &self.inner.current else {
            return operations_result;
        };
        let read_result = current
            .read_pool
            .close(deadline)
            .await
            .map_err(map_read_pool_error);
        operations_result.and(read_result)
    }

    /// Interrupt active reads and synchronously reject every future checkout.
    pub fn abort(&self) {
        self.inner.cancellation.send_replace(true);
        if let Ok(mut clients) = self.inner.embedder_clients.lock() {
            clients.clear();
        }
        if let Some(current) = &self.inner.current {
            current.read_pool.abort();
        }
        if let Some(operations) = &self.inner.operations {
            operations.abort();
        }
    }

    #[allow(dead_code)] // Tasks 3-6 consume the current-only backend.
    pub(super) fn current(&self) -> Result<&CurrentInspectionBackend, InspectionError> {
        self.inner
            .current
            .as_ref()
            .ok_or_else(|| match self.inner.database.state {
                InspectionDatabaseStateV1::UpgradeRequired | InspectionDatabaseStateV1::Newer => {
                    InspectionError::MigrationRequired
                }
                InspectionDatabaseStateV1::Incompatible => InspectionError::IntegrityError,
                InspectionDatabaseStateV1::Missing
                | InspectionDatabaseStateV1::Empty
                | InspectionDatabaseStateV1::Unreadable
                | InspectionDatabaseStateV1::Current => InspectionError::StorageUnavailable,
            })
    }
}

impl fmt::Debug for InspectionService {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("InspectionService")
            .field("database_state", &self.inner.database.state)
            .field("content_policy", &self.inner.options.content_policy)
            .field(
                "request_embedding_enabled",
                &self.inner.options.allow_request_embedding,
            )
            .finish_non_exhaustive()
    }
}

impl Drop for InspectionServiceInner {
    fn drop(&mut self) {
        self.cancellation.send_replace(true);
        if let Some(current) = &self.current {
            current.read_pool.abort();
        }
        if let Some(operations) = &self.operations {
            operations.abort();
        }
    }
}

fn inspection_deadline(timeout_ms: u64) -> Result<Instant, InspectionError> {
    Instant::now()
        .checked_add(Duration::from_millis(timeout_ms))
        .ok_or(InspectionError::InvalidArgument)
}

fn map_database_state(state: RouterDatabaseSchemaState) -> InspectionDatabaseStateV1 {
    match state {
        RouterDatabaseSchemaState::Missing => InspectionDatabaseStateV1::Missing,
        RouterDatabaseSchemaState::Empty => InspectionDatabaseStateV1::Empty,
        RouterDatabaseSchemaState::Current => InspectionDatabaseStateV1::Current,
        RouterDatabaseSchemaState::UpgradeRequired => InspectionDatabaseStateV1::UpgradeRequired,
        RouterDatabaseSchemaState::Newer => InspectionDatabaseStateV1::Newer,
        RouterDatabaseSchemaState::Incompatible => InspectionDatabaseStateV1::Incompatible,
        RouterDatabaseSchemaState::Unreadable => InspectionDatabaseStateV1::Unreadable,
    }
}

fn map_read_pool_error(error: ReadPoolError) -> InspectionError {
    match error.class() {
        ReadPoolErrorClass::Deadline => InspectionError::Busy,
        ReadPoolErrorClass::OperationFailed
        | ReadPoolErrorClass::WorkerFailed
        | ReadPoolErrorClass::Invariant => InspectionError::IntegrityError,
        ReadPoolErrorClass::InvalidFilesystem
        | ReadPoolErrorClass::InvalidPermissions
        | ReadPoolErrorClass::UnsupportedFilesystemSecurity
        | ReadPoolErrorClass::OpenFailed
        | ReadPoolErrorClass::ConfigurationFailed
        | ReadPoolErrorClass::Closing
        | ReadPoolErrorClass::Closed
        | ReadPoolErrorClass::Aborted => InspectionError::StorageUnavailable,
    }
}

fn map_ledger_error(error: LedgerError) -> InspectionError {
    map_ledger_error_class(error.class())
}

pub(super) fn map_ledger_error_class(class: LedgerErrorClass) -> InspectionError {
    match class {
        LedgerErrorClass::Busy => InspectionError::Busy,
        LedgerErrorClass::FutureSchema
        | LedgerErrorClass::MigrationChecksumMismatch
        | LedgerErrorClass::InvalidMigrationHistory
        | LedgerErrorClass::MigrationFailed => InspectionError::MigrationRequired,
        LedgerErrorClass::ProjectIdMismatch => InspectionError::InvalidArgument,
        LedgerErrorClass::CorruptDatabase
        | LedgerErrorClass::IdentityInvariant
        | LedgerErrorClass::CanonicalizationFailed
        | LedgerErrorClass::PragmaMismatch
        | LedgerErrorClass::SqliteVersionMismatch => InspectionError::IntegrityError,
        LedgerErrorClass::InvalidFilesystem
        | LedgerErrorClass::InvalidPermissions
        | LedgerErrorClass::UnsupportedFilesystemSecurity
        | LedgerErrorClass::OpenFailed
        | LedgerErrorClass::DatabaseOperationFailed
        | LedgerErrorClass::RandomnessUnavailable => InspectionError::StorageUnavailable,
    }
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::fs::{self, File};

    use rusqlite::Connection;
    use tempfile::tempdir;

    use super::*;
    use crate::host::ROUTER_LEDGER_SCHEMA_VERSION;
    use crate::ledger::migrations::{LEDGER_APPLICATION_ID, MIGRATIONS};
    use crate::ledger::repository::LedgerRepository;

    fn config(path: &Path, project_id: &str) -> RouterConfig {
        RouterConfig {
            project_id: Some(project_id.into()),
            database_path: path.to_string_lossy().into_owned(),
            ..RouterConfig::default()
        }
    }

    fn domain_counts(path: &Path) -> Vec<i64> {
        let connection = Connection::open(path).unwrap();
        [
            "project_metadata",
            "config_generations",
            "process_instances",
            "controls",
            "learning_generations",
            "cohort_generations",
        ]
        .iter()
        .map(|table| {
            connection
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                    row.get(0)
                })
                .unwrap()
        })
        .collect()
    }

    fn directory_snapshot(path: &Path) -> Vec<(OsString, Vec<u8>)> {
        let mut snapshot = fs::read_dir(path)
            .unwrap()
            .map(|entry| entry.unwrap())
            .filter(|entry| entry.file_type().unwrap().is_file())
            .map(|entry| (entry.file_name(), fs::read(entry.path()).unwrap()))
            .collect::<Vec<_>>();
        snapshot.sort_by(|left, right| left.0.cmp(&right.0));
        snapshot
    }

    #[tokio::test]
    async fn missing_and_empty_databases_remain_status_only() {
        let temporary = tempdir().unwrap();
        let missing = temporary.path().join("missing.sqlite3");
        let missing_service = InspectionService::open(
            config(&missing, "inspection-missing"),
            InspectionServiceOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            missing_service.database_status().state,
            InspectionDatabaseStateV1::Missing
        );
        assert_eq!(
            missing_service.current().err(),
            Some(InspectionError::StorageUnavailable)
        );
        assert!(!missing.exists());

        let empty = temporary.path().join("empty.sqlite3");
        File::create(&empty).unwrap();
        let empty_service = InspectionService::open(
            config(&empty, "inspection-empty"),
            InspectionServiceOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            empty_service.database_status().state,
            InspectionDatabaseStateV1::Empty
        );
        assert_eq!(File::open(&empty).unwrap().metadata().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn noncurrent_databases_remain_status_only_and_unchanged() {
        let temporary = tempdir().unwrap();

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

        let newer = temporary.path().join("newer.sqlite3");
        let connection = Connection::open(&newer).unwrap();
        connection
            .pragma_update(None, "application_id", LEDGER_APPLICATION_ID)
            .unwrap();
        connection
            .pragma_update(None, "user_version", ROUTER_LEDGER_SCHEMA_VERSION + 1)
            .unwrap();
        drop(connection);

        let incompatible = temporary.path().join("incompatible.sqlite3");
        Connection::open(&incompatible)
            .unwrap()
            .execute_batch("CREATE TABLE unrelated(value INTEGER);")
            .unwrap();

        for (path, state, error) in [
            (
                older,
                InspectionDatabaseStateV1::UpgradeRequired,
                InspectionError::MigrationRequired,
            ),
            (
                newer,
                InspectionDatabaseStateV1::Newer,
                InspectionError::MigrationRequired,
            ),
            (
                incompatible,
                InspectionDatabaseStateV1::Incompatible,
                InspectionError::IntegrityError,
            ),
        ] {
            let before = fs::read(&path).unwrap();
            let service = InspectionService::open(
                config(&path, "inspection-status-only"),
                InspectionServiceOptions::default(),
            )
            .await
            .unwrap();
            assert_eq!(service.database_status().state, state);
            assert_eq!(service.current().err(), Some(error));
            service.close().await.unwrap();
            assert_eq!(fs::read(&path).unwrap(), before);
        }

        let service = InspectionService::open(
            config(temporary.path(), "inspection-unreadable"),
            InspectionServiceOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(
            service.database_status().state,
            InspectionDatabaseStateV1::Unreadable
        );
        assert_eq!(
            service.current().err(),
            Some(InspectionError::StorageUnavailable)
        );
    }

    #[tokio::test]
    async fn current_open_verifies_authority_without_domain_writes() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("router.sqlite3");
        let config = config(&path, "inspection-current");
        drop(LedgerRepository::activate(&config).unwrap());
        let before = domain_counts(&path);
        let before_files = directory_snapshot(temporary.path());

        let service = InspectionService::open(
            config.clone(),
            InspectionServiceOptions {
                content_policy: ContentPolicy::Full,
                ..InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        assert_eq!(
            service.database_status().state,
            InspectionDatabaseStateV1::Current
        );
        assert_eq!(service.content_policy(), ContentPolicy::Full);
        let current = service.current().unwrap();
        assert_eq!(current.authority.project_id, "inspection-current");
        assert_eq!(
            current.authority.config_generation_id,
            config.config_generation_id().unwrap()
        );
        assert!(current.authority.policy_version_ids.is_empty());
        assert!(current.authority.learning_generation_ids.is_empty());
        assert!(current.authority.control_snapshot.is_none());
        assert!(current.authority.vector_authorities.is_empty());
        service.close().await.unwrap();
        assert_eq!(domain_counts(&path), before);
        assert_eq!(directory_snapshot(temporary.path()), before_files);
    }

    #[tokio::test]
    async fn current_open_rejects_config_and_project_mismatch() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("router.sqlite3");
        let original = config(&path, "inspection-match");
        drop(LedgerRepository::activate(&original).unwrap());

        let mut changed = original.clone();
        changed.retention_days += 1;
        assert_eq!(
            InspectionService::open(changed, InspectionServiceOptions::default())
                .await
                .unwrap_err(),
            InspectionError::IntegrityError
        );
        let wrong_project = config(&path, "inspection-other-project");
        assert_eq!(
            InspectionService::open(wrong_project, InspectionServiceOptions::default())
                .await
                .unwrap_err(),
            InspectionError::InvalidArgument
        );
    }

    #[tokio::test]
    async fn close_and_abort_are_bounded_and_shared_by_clones() {
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("router.sqlite3");
        let config = config(&path, "inspection-close");
        drop(LedgerRepository::activate(&config).unwrap());
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        let clone = service.clone();
        clone.close().await.unwrap();
        service.close().await.unwrap();
        service.abort();
    }
}
