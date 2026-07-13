// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sole-writer transaction adapter for vector-registry ensures.

use rusqlite::TransactionBehavior;

use super::vector_registry::{
    RegistryEnsureAck, VectorRegistryEnsure, ensure_vector_registry_in_transaction,
};
use super::{LedgerRepository, TransactionStartGuard, map_fs_error, map_sqlite_error};
use crate::ledger::fs::enforce_sidecar_permissions;
use crate::ledger::model::{LedgerError, LedgerErrorClass};

/// Internal result used to preserve the writer's abort linearization point.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum RegistryEnsureTransactionAck {
    Completed(RegistryEnsureAck),
    TransactionNotStarted,
}

impl LedgerRepository {
    pub(crate) fn ensure_vector_registry_with_start_check<G: TransactionStartGuard>(
        &mut self,
        ensure: &VectorRegistryEnsure,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<RegistryEnsureTransactionAck, LedgerError> {
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(RegistryEnsureTransactionAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(RegistryEnsureTransactionAck::TransactionNotStarted);
            }
            Err(error) => {
                return Err(map_sqlite_error(
                    &error,
                    LedgerErrorClass::DatabaseOperationFailed,
                ));
            }
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(RegistryEnsureTransactionAck::TransactionNotStarted);
        }
        drop(start_guard);

        let acknowledgement = ensure_vector_registry_in_transaction(&transaction, ensure)?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction
            .commit()
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed))?;
        Ok(RegistryEnsureTransactionAck::Completed(acknowledgement))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::mpsc;
    use std::time::{Duration, Instant};

    use rusqlite::Connection;
    use tempfile::{TempDir, tempdir};

    use super::*;
    use crate::ledger::command::WriterFailureClass;
    use crate::ledger::repository::tests::{config, database_path};
    use crate::ledger::writer::{LedgerWriterClient, LedgerWriterOwner};

    struct RegistryWriterFixture {
        _temporary: TempDir,
        path: std::path::PathBuf,
        owner: LedgerWriterOwner,
        client: LedgerWriterClient,
        ensure: VectorRegistryEnsure,
    }

    fn deadline() -> Instant {
        Instant::now() + Duration::from_secs(5)
    }

    fn registry_writer_fixture() -> RegistryWriterFixture {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut config = config(&path, "registry-writer-project");
        config.pools[0].learning = Some(crate::config::LearningConfig::minimal("embedder-a"));
        let mut activated = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let policies = activated
            .identity
            .pools
            .iter()
            .map(|(pool_id, identity)| (pool_id.clone(), identity.policy_version_id.clone()))
            .collect::<BTreeMap<_, _>>();
        let ensure = super::super::vector_registry::prepare_vector_registry(
            &config,
            activated.identity.project_uuid,
            &activated.identity.config_generation_id,
            &policies,
            1_001,
        )
        .unwrap();
        activated
            .repository
            .connection_mut()
            .execute_batch(
                "DELETE FROM pool_vector_space_mappings;
                 DELETE FROM vector_space_source_sequences;
                 DELETE FROM vector_spaces;
                 DELETE FROM embedder_profiles;",
            )
            .unwrap();
        assert_eq!(registry_row_count(&path), 0);
        let (owner, client) = LedgerWriterOwner::start(activated.repository, 4).unwrap();
        RegistryWriterFixture {
            _temporary: temporary,
            path,
            owner,
            client,
            ensure,
        }
    }

    fn registry_row_count(path: &std::path::Path) -> i64 {
        Connection::open(path)
            .unwrap()
            .query_row(
                "SELECT
                    (SELECT count(*) FROM embedder_profiles) +
                    (SELECT count(*) FROM vector_spaces) +
                    (SELECT count(*) FROM vector_space_source_sequences) +
                    (SELECT count(*) FROM pool_vector_space_mappings)",
                [],
                |row| row.get(0),
            )
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn writer_roundtrips_registry_ensure_and_deadline() {
        let mut fixture = registry_writer_fixture();
        let failure = fixture
            .client
            .ensure_vector_registry_until(fixture.ensure.clone(), Instant::now())
            .await
            .unwrap_err();
        assert_eq!(failure.class(), WriterFailureClass::Deadline);
        assert_eq!(registry_row_count(&fixture.path), 0);

        let first = fixture
            .client
            .ensure_vector_registry_until(fixture.ensure.clone(), deadline())
            .await
            .unwrap();
        let snapshot = match first {
            RegistryEnsureAck::Applied(snapshot) => snapshot,
            other => panic!("unexpected first registry acknowledgement: {other:?}"),
        };
        assert_eq!(snapshot.profiles.len(), 1);
        assert_eq!(snapshot.spaces.len(), 1);
        assert_eq!(snapshot.mappings.len(), 1);
        assert_eq!(registry_row_count(&fixture.path), 4);

        assert_eq!(
            fixture
                .client
                .ensure_vector_registry_until(fixture.ensure.clone(), deadline())
                .await
                .unwrap(),
            RegistryEnsureAck::AlreadyApplied(snapshot)
        );
        fixture.owner.drain_until(deadline()).await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn queued_abort_rejects_registry_ensure_without_mutation() {
        let fixture = registry_writer_fixture();
        let (started_tx, started_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let pause_client = fixture.client.clone();
        let pause = tokio::spawn(async move {
            pause_client
                .pause_until(deadline(), started_tx, release_rx)
                .await
        });
        started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

        let ensure_client = fixture.client.clone();
        let ensure = fixture.ensure.clone();
        let queued = tokio::spawn(async move {
            ensure_client
                .ensure_vector_registry_until(ensure, deadline())
                .await
        });
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(!queued.is_finished());

        fixture.client.abort();
        release_tx.send(()).unwrap();
        let failure = queued.await.unwrap().unwrap_err();
        assert_eq!(failure.class(), WriterFailureClass::Aborted);
        let _ = pause.await.unwrap();
        tokio::time::sleep(Duration::from_millis(25)).await;
        assert_eq!(registry_row_count(&fixture.path), 0);
        drop(fixture.owner);
    }
}
