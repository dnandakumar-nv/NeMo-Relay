// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Async sqlite-vec facade over the ledger's sole writer and bounded read pool.

use std::fmt;
use std::time::Instant;

use crate::ledger::command::{WriterFailure, WriterFailureClass};
use crate::ledger::model::LedgerErrorClass;
use crate::ledger::read_pool::LedgerReadPool;
use crate::ledger::repository::vector_index::{
    GenerationAuthorizationAck, GenerationObjectCreationAck, RebuildFlipAck, RebuildLeaseClaimAck,
    RebuildLeaseFence, RebuildLeaseMutationAck, RebuildStepAck, RetiredGenerationCleanupAck,
    VectorIndexHealthMutationAck, VectorIndexHealthTarget, VectorIndexPointMutationAck,
};
use crate::ledger::repository::vector_registry::FrozenMappingKey;
use crate::ledger::repository::vector_search::{
    LiveProjectedNeighborSearch, ProjectedVectorNeighbor, search_live_projected_neighbors,
    search_projected_neighbors,
};
use crate::ledger::writer::LedgerWriterClient;
use crate::routing_partition::RoutingPartitionArtifactV1;
use crate::sqlite_vec_schema::VectorIndexGeneration;
use crate::vector::{
    NormalizedVector, PartitionId, VectorDimensions, VectorRecordId, VectorSpaceId,
};
use crate::vector_store::{SpaceHealth, VectorRecord, VectorStoreError};

/// One mutation serialized through the ledger's sole writer connection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VectorIndexWriterCommand {
    AuthorizeGeneration {
        vector_space_id: VectorSpaceId,
        dimensions: VectorDimensions,
        created_at_unix_ms: i64,
    },
    ClaimRebuildLease {
        vector_space_id: VectorSpaceId,
        observed_at_unix_ms: i64,
    },
    CreateGenerationObjects {
        fence: RebuildLeaseFence,
        observed_at_unix_ms: i64,
    },
    UpsertActiveRecord {
        record: Box<VectorRecord>,
    },
    DeleteActiveRecord {
        vector_space_id: VectorSpaceId,
        record_id: VectorRecordId,
    },
    RenewRebuildLease {
        fence: RebuildLeaseFence,
        observed_at_unix_ms: i64,
    },
    ReleaseRebuildLease {
        fence: RebuildLeaseFence,
        released_at_unix_ms: i64,
    },
    PopulateRebuildChunk {
        fence: RebuildLeaseFence,
        observed_at_unix_ms: i64,
    },
    CatchUpRebuildChanges {
        fence: RebuildLeaseFence,
        observed_at_unix_ms: i64,
    },
    FlipRebuildGeneration {
        fence: RebuildLeaseFence,
        activated_at_unix_ms: i64,
    },
    CleanupRetiredGeneration {
        vector_space_id: VectorSpaceId,
        generation: VectorIndexGeneration,
        dropped_at_unix_ms: i64,
    },
    MarkHealth {
        vector_space_id: VectorSpaceId,
        expected_generation: VectorIndexGeneration,
        expected_manifest_hash: String,
        target: VectorIndexHealthTarget,
        stable_error_class: String,
        observed_at_unix_ms: i64,
    },
}

/// Exact vector-local acknowledgement returned by one writer mutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VectorIndexWriterAck {
    GenerationAuthorized(Box<GenerationAuthorizationAck>),
    RebuildLeaseClaimed(RebuildLeaseClaimAck),
    GenerationObjectsCreated(GenerationObjectCreationAck),
    PointMutated(VectorIndexPointMutationAck),
    RebuildLeaseMutated(RebuildLeaseMutationAck),
    RebuildStepped(RebuildStepAck),
    RebuildFlipped(RebuildFlipAck),
    RetiredGenerationCleaned(RetiredGenerationCleanupAck),
    HealthMarked(VectorIndexHealthMutationAck),
}

/// Repository-only acknowledgement used to map a fenced transaction refusal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VectorIndexTransactionAck {
    Completed(VectorIndexWriterAck),
    TransactionNotStarted,
}

/// Production vector-index handle without independent SQLite ownership.
#[derive(Clone)]
pub(crate) struct SqliteVecStore {
    writer: LedgerWriterClient,
    read_pool: LedgerReadPool,
}

impl SqliteVecStore {
    pub(crate) fn new(writer: LedgerWriterClient, read_pool: LedgerReadPool) -> Self {
        Self { writer, read_pool }
    }

    /// Serialize one vec0 DDL, DML, lease, or manifest transition.
    pub(crate) async fn mutate_until(
        &self,
        command: VectorIndexWriterCommand,
        deadline: Instant,
    ) -> Result<VectorIndexWriterAck, VectorStoreError> {
        self.writer
            .vector_index_until(command, deadline)
            .await
            .map_err(map_writer_failure)
    }

    /// Search and project complete neighbors before the read snapshot closes.
    pub(crate) async fn search_until(
        &self,
        vector_space_id: &VectorSpaceId,
        partition_id: PartitionId,
        query_vector: &NormalizedVector,
        top_k: usize,
        deadline: Instant,
    ) -> Result<Vec<ProjectedVectorNeighbor>, VectorStoreError> {
        let vector_space_id = vector_space_id.clone();
        let query_vector = query_vector.clone();
        self.read_pool
            .run(deadline, move |connection| {
                Ok(search_projected_neighbors(
                    connection,
                    &vector_space_id,
                    partition_id,
                    &query_vector,
                    top_k,
                ))
            })
            .await
            .map_err(|_| VectorStoreError::Unavailable)?
    }

    /// Resolve a full live partition and project neighbors in one snapshot.
    pub(crate) async fn search_live_partition_until(
        &self,
        mapping: &FrozenMappingKey,
        expected_partition: &RoutingPartitionArtifactV1,
        query_vector: &NormalizedVector,
        top_k: usize,
        deadline: Instant,
    ) -> Result<LiveProjectedNeighborSearch, VectorStoreError> {
        let mapping = mapping.clone();
        let expected_partition = expected_partition.clone();
        let query_vector = query_vector.clone();
        self.read_pool
            .run(deadline, move |connection| {
                Ok(search_live_projected_neighbors(
                    connection,
                    &mapping,
                    &expected_partition,
                    &query_vector,
                    top_k,
                ))
            })
            .await
            .map_err(|_| VectorStoreError::Unavailable)?
    }

    /// Read one bounded health snapshot through a pooled deferred transaction.
    pub(crate) async fn health_until(
        &self,
        vector_space_id: &VectorSpaceId,
        deadline: Instant,
    ) -> Result<SpaceHealth, VectorStoreError> {
        let vector_space_id = vector_space_id.clone();
        self.read_pool
            .run(deadline, move |connection| {
                Ok(read_health_in_transaction(connection, &vector_space_id))
            })
            .await
            .map_err(|_| VectorStoreError::Unavailable)?
    }
}

impl fmt::Debug for SqliteVecStore {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SqliteVecStore")
            .finish_non_exhaustive()
    }
}

fn read_health_in_transaction(
    connection: &rusqlite::Connection,
    vector_space_id: &VectorSpaceId,
) -> Result<SpaceHealth, VectorStoreError> {
    let transaction = connection
        .unchecked_transaction()
        .map_err(|_| VectorStoreError::Unavailable)?;
    let health =
        crate::ledger::repository::vector_index::vector_index_health(&transaction, vector_space_id)
            .map_err(|error| map_ledger_failure(error.class()))?;
    transaction
        .commit()
        .map_err(|_| VectorStoreError::Unavailable)?;
    Ok(health)
}

fn map_writer_failure(failure: WriterFailure) -> VectorStoreError {
    match failure.class() {
        WriterFailureClass::Repository(class) => map_ledger_failure(class),
        WriterFailureClass::Full
        | WriterFailureClass::Closing
        | WriterFailureClass::Deadline
        | WriterFailureClass::Exited
        | WriterFailureClass::Panicked
        | WriterFailureClass::Aborted
        | WriterFailureClass::Protocol => VectorStoreError::Unavailable,
    }
}

fn map_ledger_failure(class: LedgerErrorClass) -> VectorStoreError {
    match class {
        LedgerErrorClass::ProjectIdMismatch | LedgerErrorClass::IdentityInvariant => {
            VectorStoreError::Conflict
        }
        LedgerErrorClass::CorruptDatabase
        | LedgerErrorClass::PragmaMismatch
        | LedgerErrorClass::SqliteVersionMismatch
        | LedgerErrorClass::FutureSchema
        | LedgerErrorClass::MigrationChecksumMismatch
        | LedgerErrorClass::InvalidMigrationHistory
        | LedgerErrorClass::MigrationFailed => VectorStoreError::Corrupt,
        LedgerErrorClass::InvalidFilesystem
        | LedgerErrorClass::InvalidPermissions
        | LedgerErrorClass::UnsupportedFilesystemSecurity
        | LedgerErrorClass::OpenFailed
        | LedgerErrorClass::Busy
        | LedgerErrorClass::RandomnessUnavailable
        | LedgerErrorClass::CanonicalizationFailed
        | LedgerErrorClass::DatabaseOperationFailed => VectorStoreError::Unavailable,
    }
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::mpsc;
    use std::time::Duration;

    use rusqlite::params;
    use serde_json::json;
    use tempfile::{TempDir, tempdir};
    use uuid::Uuid;

    use super::*;
    use crate::config::RouterConfig;
    use crate::ledger::repository::LedgerRepository;
    use crate::ledger::writer::LedgerWriterOwner;
    use crate::vector::VectorDimensions;
    use crate::vector_store::SpaceHealthState;

    fn database_path(temporary: &TempDir) -> PathBuf {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        temporary.path().join("ledger/router.db")
    }

    fn config(path: &Path) -> RouterConfig {
        serde_json::from_value(json!({
            "version": 1,
            "mode": "shadow",
            "project_id": "sqlite-vector-store-project",
            "database_path": path.to_string_lossy(),
            "retention_days": 30,
            "max_evidence_records": 1000,
            "pools": [{
                "id": "pool-a",
                "api_family": "openai_chat_completions",
                "anchor_models": ["anchor-a"],
                "anchor_revision": "2026-07-01",
                "sampling_probability": 0.25,
                "max_candidates_per_sample": 1,
                "concurrency": {"shadow": 1, "judge": 1, "max_pending": 1},
                "judge": {
                    "version": 1,
                    "model": "judge-model",
                    "model_revision": "2026-07-01",
                    "prompt_version": "pairwise-equivalence-v1",
                    "rubric_version": "response-trajectory-equivalence-v1",
                    "output_schema_version": 1,
                    "response_weight": 0.5,
                    "trajectory_weight": 0.5,
                    "response_floor": 0.8,
                    "trajectory_floor": 0.8,
                    "judge_confidence_floor": 0.7,
                    "pass_threshold": 0.85,
                    "max_rationale_bytes": 4096,
                    "base_cooloff_seconds": 10,
                    "max_cooloff_seconds": 300
                },
                "candidates": [{
                    "id": "candidate-a",
                    "model": "candidate-model-a",
                    "model_revision": "2026-06-01",
                    "cost_rank": 0,
                    "max_context_tokens": 32768,
                    "capabilities": {"tools": true}
                }]
            }]
        }))
        .unwrap()
    }

    fn space_id() -> VectorSpaceId {
        VectorSpaceId::new("a".repeat(64)).unwrap()
    }

    fn record_id() -> VectorRecordId {
        VectorRecordId::new(Uuid::now_v7()).unwrap()
    }

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(5)
    }

    #[test]
    fn writer_deadline_and_abort_are_vector_local_unavailability() {
        for class in [WriterFailureClass::Deadline, WriterFailureClass::Aborted] {
            assert_eq!(
                map_writer_failure(WriterFailure::new(class)),
                VectorStoreError::Unavailable
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn facade_uses_the_sole_writer_and_bounded_read_pool() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let activated = LedgerRepository::activate(&config(&path)).unwrap();
        let read_pool = LedgerReadPool::open(&path).unwrap();
        let (mut owner, writer) = LedgerWriterOwner::start(activated.repository, 4).unwrap();
        let store = SqliteVecStore::new(writer.clone(), read_pool.clone());

        assert_eq!(format!("{store:?}"), "SqliteVecStore { .. }");
        assert_eq!(
            store
                .mutate_until(
                    VectorIndexWriterCommand::DeleteActiveRecord {
                        vector_space_id: space_id(),
                        record_id: record_id(),
                    },
                    Instant::now(),
                )
                .await,
            Err(VectorStoreError::Unavailable)
        );

        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let pause_writer = writer.clone();
        let pause = tokio::spawn(async move {
            pause_writer
                .pause_until(deadline(), started_tx, release_rx)
                .await
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let mutation_store = store.clone();
        let mutation = tokio::spawn(async move {
            mutation_store
                .mutate_until(
                    VectorIndexWriterCommand::DeleteActiveRecord {
                        vector_space_id: space_id(),
                        record_id: record_id(),
                    },
                    deadline(),
                )
                .await
        });
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(!mutation.is_finished());
        release_tx.send(()).unwrap();
        pause.await.unwrap().unwrap();
        assert_eq!(
            mutation.await.unwrap().unwrap(),
            VectorIndexWriterAck::PointMutated(VectorIndexPointMutationAck::Missing)
        );

        let health = store.health_until(&space_id(), deadline()).await.unwrap();
        assert_eq!(health.state(), SpaceHealthState::Missing);
        assert_eq!(health.record_count(), 0);

        let query =
            NormalizedVector::from_provider_f64(&[1.0, 0.0], VectorDimensions::new(2).unwrap())
                .unwrap();
        assert_eq!(
            store
                .search_until(
                    &space_id(),
                    PartitionId::new(1).unwrap(),
                    &query,
                    1,
                    deadline(),
                )
                .await,
            Err(VectorStoreError::SpaceNotFound)
        );

        read_pool.abort();
        assert_eq!(
            store.health_until(&space_id(), deadline()).await,
            Err(VectorStoreError::Unavailable)
        );
        owner.drain_until(deadline()).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn facade_roundtrips_an_empty_fenced_rebuild() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let activated = LedgerRepository::activate_at(&config(&path), 1_000).unwrap();
        let project_uuid = activated.identity.project_uuid.to_string();
        let connection = rusqlite::Connection::open(&path).unwrap();
        connection
            .execute(
                "INSERT INTO embedder_profiles (
                    embedder_profile_version_id, profile_id, protocol, endpoint_url,
                    endpoint_identity_sha256, model, provider_revision, dimensions,
                    credential_env_name_sha256, timeout_ms, max_in_flight, batch_size,
                    egress_class, canonical_profile_json, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, 'fixture', 'openai-embeddings-v1',
                    'http://127.0.0.1/v1/embeddings', ?2, 'fixture-model', 'r1', 2,
                    NULL, 1000, 1, 1, 'loopback_http', '{}', 1000, ?1)",
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
                 ) VALUES (?1, ?2, ?3, ?4, '{}', ?5, 'fixture-model', 'r1', 2,
                    'cosine', 'l2_f32_v1', '{}', 1000, ?1)",
                params![
                    space_id().as_str(),
                    project_uuid,
                    "b".repeat(64),
                    "d".repeat(64),
                    "c".repeat(64),
                ],
            )
            .unwrap();
        let sequence_hash =
            crate::ledger::repository::vector_index::vector_source_sequence_payload_hash(
                &space_id(),
                0,
                1_000,
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO vector_space_source_sequences (
                    vector_space_id, source_seq, updated_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, 0, 1000, ?2)",
                params![space_id().as_str(), sequence_hash],
            )
            .unwrap();
        drop(connection);

        let read_pool = LedgerReadPool::open(&path).unwrap();
        let (mut owner, writer) = LedgerWriterOwner::start(activated.repository, 8).unwrap();
        let store = SqliteVecStore::new(writer, read_pool);
        let generation = match store
            .mutate_until(
                VectorIndexWriterCommand::AuthorizeGeneration {
                    vector_space_id: space_id(),
                    dimensions: VectorDimensions::new(2).unwrap(),
                    created_at_unix_ms: 1_001,
                },
                deadline(),
            )
            .await
            .unwrap()
        {
            VectorIndexWriterAck::GenerationAuthorized(acknowledgement) => match *acknowledgement {
                GenerationAuthorizationAck::Created(manifest) => manifest.generation(),
                other => panic!("unexpected authorization acknowledgement: {other:?}"),
            },
            other => panic!("unexpected writer acknowledgement: {other:?}"),
        };
        let fence = match store
            .mutate_until(
                VectorIndexWriterCommand::ClaimRebuildLease {
                    vector_space_id: space_id(),
                    observed_at_unix_ms: 1_002,
                },
                deadline(),
            )
            .await
            .unwrap()
        {
            VectorIndexWriterAck::RebuildLeaseClaimed(RebuildLeaseClaimAck::Claimed(fence)) => {
                fence
            }
            other => panic!("unexpected claim acknowledgement: {other:?}"),
        };
        assert_eq!(
            store
                .mutate_until(
                    VectorIndexWriterCommand::CreateGenerationObjects {
                        fence: fence.clone(),
                        observed_at_unix_ms: 1_002,
                    },
                    deadline(),
                )
                .await
                .unwrap(),
            VectorIndexWriterAck::GenerationObjectsCreated(GenerationObjectCreationAck::Created)
        );
        assert_eq!(
            store
                .mutate_until(
                    VectorIndexWriterCommand::RenewRebuildLease {
                        fence: fence.clone(),
                        observed_at_unix_ms: 1_003,
                    },
                    deadline(),
                )
                .await
                .unwrap(),
            VectorIndexWriterAck::RebuildLeaseMutated(RebuildLeaseMutationAck::Applied)
        );
        assert!(matches!(
            store
                .mutate_until(
                    VectorIndexWriterCommand::PopulateRebuildChunk {
                        fence: fence.clone(),
                        observed_at_unix_ms: 1_004,
                    },
                    deadline(),
                )
                .await
                .unwrap(),
            VectorIndexWriterAck::RebuildStepped(RebuildStepAck::Applied {
                processed: 0,
                complete: true,
                applied_source_seq: 0,
            })
        ));
        assert!(matches!(
            store
                .mutate_until(
                    VectorIndexWriterCommand::CatchUpRebuildChanges {
                        fence: fence.clone(),
                        observed_at_unix_ms: 1_004,
                    },
                    deadline(),
                )
                .await
                .unwrap(),
            VectorIndexWriterAck::RebuildStepped(RebuildStepAck::Applied {
                processed: 0,
                complete: true,
                applied_source_seq: 0,
            })
        ));
        assert_eq!(
            store
                .mutate_until(
                    VectorIndexWriterCommand::FlipRebuildGeneration {
                        fence,
                        activated_at_unix_ms: 1_005,
                    },
                    deadline(),
                )
                .await
                .unwrap(),
            VectorIndexWriterAck::RebuildFlipped(RebuildFlipAck::Activated { record_count: 0 })
        );
        let health = store.health_until(&space_id(), deadline()).await.unwrap();
        assert_eq!(health.state(), SpaceHealthState::Healthy);
        assert_eq!(health.record_count(), 0);

        let connection = rusqlite::Connection::open(&path).unwrap();
        let active_hash = connection
            .query_row(
                "SELECT canonical_payload_hash FROM vector_index_manifest
                 WHERE vector_space_id = ?1 AND generation = ?2 AND state = 'active'",
                params![space_id().as_str(), generation.value()],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        drop(connection);
        let unavailable_hash = match store
            .mutate_until(
                VectorIndexWriterCommand::MarkHealth {
                    vector_space_id: space_id(),
                    expected_generation: generation,
                    expected_manifest_hash: active_hash.clone(),
                    target: VectorIndexHealthTarget::Unavailable,
                    stable_error_class: "router.vector.unavailable".to_string(),
                    observed_at_unix_ms: 1_006,
                },
                deadline(),
            )
            .await
            .unwrap()
        {
            VectorIndexWriterAck::HealthMarked(VectorIndexHealthMutationAck::Applied {
                manifest_hash,
            }) => manifest_hash,
            other => panic!("unexpected health acknowledgement: {other:?}"),
        };
        assert_ne!(unavailable_hash, active_hash);
        assert_eq!(
            store.health_until(&space_id(), deadline()).await.unwrap(),
            SpaceHealth::new(SpaceHealthState::Unavailable, 0)
        );
        assert_eq!(
            store
                .mutate_until(
                    VectorIndexWriterCommand::MarkHealth {
                        vector_space_id: space_id(),
                        expected_generation: generation,
                        expected_manifest_hash: unavailable_hash.clone(),
                        target: VectorIndexHealthTarget::Unavailable,
                        stable_error_class: "router.vector.unavailable".to_string(),
                        observed_at_unix_ms: 1_007,
                    },
                    deadline(),
                )
                .await
                .unwrap(),
            VectorIndexWriterAck::HealthMarked(VectorIndexHealthMutationAck::AlreadyApplied {
                manifest_hash: unavailable_hash.clone(),
            })
        );
        assert_eq!(
            store
                .mutate_until(
                    VectorIndexWriterCommand::MarkHealth {
                        vector_space_id: space_id(),
                        expected_generation: generation,
                        expected_manifest_hash: active_hash,
                        target: VectorIndexHealthTarget::Corrupt,
                        stable_error_class: "router.vector.corrupt".to_string(),
                        observed_at_unix_ms: 1_008,
                    },
                    deadline(),
                )
                .await
                .unwrap(),
            VectorIndexWriterAck::HealthMarked(VectorIndexHealthMutationAck::Stale)
        );
        assert!(matches!(
            store
                .mutate_until(
                    VectorIndexWriterCommand::MarkHealth {
                        vector_space_id: space_id(),
                        expected_generation: generation,
                        expected_manifest_hash: unavailable_hash,
                        target: VectorIndexHealthTarget::Corrupt,
                        stable_error_class: "router.vector.corrupt".to_string(),
                        observed_at_unix_ms: 1_009,
                    },
                    deadline(),
                )
                .await
                .unwrap(),
            VectorIndexWriterAck::HealthMarked(VectorIndexHealthMutationAck::Applied {
                ref manifest_hash,
            }) if manifest_hash.len() == 64
        ));
        assert_eq!(
            store.health_until(&space_id(), deadline()).await.unwrap(),
            SpaceHealth::new(SpaceHealthState::Corrupt, 0)
        );
        owner.drain_until(deadline()).await.unwrap();
    }
}
