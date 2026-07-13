// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One-transaction startup recovery for non-resumable Router work.

use rusqlite::{Connection, Transaction, params};
use uuid::Uuid;

use super::anchors::orphan_pending_anchor_in_transaction;
use super::cooloff::orphan_inflight_dependency_operations_in_transaction;
use super::judge::orphan_inflight_judge_attempts_in_transaction;
use super::process::{ProcessStatusAt, fence_expired_process, verified_process_status_at};
use super::shadow::{
    reconcile_shadow_work_in_transaction, verify_pending_anchor_for_reconciliation,
};
use super::{LedgerRepository, map_sqlite_error};
use crate::ledger::model::{LedgerError, LedgerErrorClass};

/// Exact counts applied by one committed reconciliation transaction.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ReconciliationReport {
    pub(crate) processes_reconciled: usize,
    pub(crate) anchors_orphaned: usize,
    pub(crate) batches_reconstructed: usize,
    pub(crate) shadow_attempts_orphaned: usize,
    pub(crate) judge_attempts_orphaned: usize,
    pub(crate) dependency_operations_orphaned: usize,
    pub(crate) batches_closed: usize,
}

impl ReconciliationReport {
    fn checked_add(&mut self, target: ReportField, amount: usize) -> Result<(), LedgerError> {
        let slot = match target {
            ReportField::Processes => &mut self.processes_reconciled,
            ReportField::Anchors => &mut self.anchors_orphaned,
            ReportField::ReconstructedBatches => &mut self.batches_reconstructed,
            ReportField::ShadowAttempts => &mut self.shadow_attempts_orphaned,
            ReportField::JudgeAttempts => &mut self.judge_attempts_orphaned,
            ReportField::DependencyOperations => &mut self.dependency_operations_orphaned,
            ReportField::ClosedBatches => &mut self.batches_closed,
        };
        *slot = slot
            .checked_add(amount)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        Ok(())
    }
}

enum ReportField {
    Processes,
    Anchors,
    ReconstructedBatches,
    ShadowAttempts,
    JudgeAttempts,
    DependencyOperations,
    ClosedBatches,
}

pub(super) fn reconcile_startup_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    reconciler_process_instance_id: Uuid,
    observed_at_unix_ms: i64,
) -> Result<ReconciliationReport, LedgerError> {
    if observed_at_unix_ms < 0
        || verified_process_status_at(
            transaction,
            project_uuid,
            reconciler_process_instance_id,
            observed_at_unix_ms,
        )? != ProcessStatusAt::Live
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let mut statement = transaction
        .prepare(
            "SELECT process_instance_id FROM process_instances
             WHERE project_uuid = ?1 AND process_instance_id <> ?2
             ORDER BY process_instance_id",
        )
        .map_err(database_error)?;
    let process_ids = statement
        .query_map(
            params![
                project_uuid.to_string(),
                reconciler_process_instance_id.to_string()
            ],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);

    let mut report = ReconciliationReport::default();
    for process_id in process_ids {
        let process_id = parse_uuid_v7(&process_id)?;
        match verified_process_status_at(
            transaction,
            project_uuid,
            process_id,
            observed_at_unix_ms,
        )? {
            ProcessStatusAt::Live => continue,
            ProcessStatusAt::Invalid => return Err(LedgerErrorClass::IdentityInvariant.into()),
            ProcessStatusAt::Expired => {
                if fence_expired_process(
                    transaction,
                    project_uuid,
                    reconciler_process_instance_id,
                    process_id,
                    observed_at_unix_ms,
                )? {
                    report.checked_add(ReportField::Processes, 1)?;
                }
            }
            ProcessStatusAt::Terminal => {}
        }

        let dependency_count = orphan_inflight_dependency_operations_in_transaction(
            transaction,
            project_uuid,
            process_id,
            observed_at_unix_ms,
        )?;
        report.checked_add(ReportField::DependencyOperations, dependency_count)?;

        let judge_count = orphan_inflight_judge_attempts_in_transaction(
            transaction,
            project_uuid,
            reconciler_process_instance_id,
            process_id,
            observed_at_unix_ms,
        )?;
        report.checked_add(ReportField::JudgeAttempts, judge_count)?;

        let pending_anchors = pending_anchor_ids(transaction, project_uuid, process_id)?;
        for anchor_id in pending_anchors {
            verify_pending_anchor_for_reconciliation(
                transaction,
                project_uuid,
                process_id,
                anchor_id,
            )?;
            if orphan_pending_anchor_in_transaction(
                transaction,
                project_uuid,
                reconciler_process_instance_id,
                process_id,
                anchor_id,
                observed_at_unix_ms,
            )? {
                report.checked_add(ReportField::Anchors, 1)?;
            }
        }
        for anchor_id in orphaned_pending_anchor_ids(transaction, project_uuid, process_id)? {
            if orphan_pending_anchor_in_transaction(
                transaction,
                project_uuid,
                reconciler_process_instance_id,
                process_id,
                anchor_id,
                observed_at_unix_ms,
            )? {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }

        let shadow = reconcile_shadow_work_in_transaction(
            transaction,
            project_uuid,
            reconciler_process_instance_id,
            process_id,
            observed_at_unix_ms,
        )?;
        report.checked_add(
            ReportField::ReconstructedBatches,
            shadow.reconstructed_batches,
        )?;
        report.checked_add(ReportField::ShadowAttempts, shadow.orphaned_attempts)?;
        report.checked_add(ReportField::ClosedBatches, shadow.closed_batches)?;
    }
    Ok(report)
}

fn orphaned_pending_anchor_ids(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
) -> Result<Vec<Uuid>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT a.anchor_id
             FROM anchors AS a
             JOIN anchor_state_events AS terminal
               ON terminal.anchor_id = a.anchor_id
              AND terminal.state = 'orphaned_non_resumable'
             WHERE a.project_uuid = ?1 AND a.process_instance_id = ?2
             ORDER BY a.anchor_id",
        )
        .map_err(database_error)?;
    statement
        .query_map(
            params![project_uuid.to_string(), process_instance_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .map(|value| {
            value
                .map_err(database_error)
                .and_then(|value| parse_uuid_v7(&value))
        })
        .collect()
}

fn pending_anchor_ids(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
) -> Result<Vec<Uuid>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT a.anchor_id
             FROM anchors AS a
             JOIN anchor_state_events AS pending
               ON pending.anchor_id = a.anchor_id AND pending.state = 'pending'
             WHERE a.project_uuid = ?1 AND a.process_instance_id = ?2
               AND NOT EXISTS (
                    SELECT 1 FROM anchor_state_events AS terminal
                    WHERE terminal.anchor_id = a.anchor_id
                      AND terminal.state <> 'pending'
               )
             ORDER BY a.anchor_id",
        )
        .map_err(database_error)?;
    statement
        .query_map(
            params![project_uuid.to_string(), process_instance_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .map(|value| {
            value
                .map_err(database_error)
                .and_then(|value| parse_uuid_v7(&value))
        })
        .collect()
}

fn parse_uuid_v7(value: &str) -> Result<Uuid, LedgerError> {
    let parsed = Uuid::parse_str(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if parsed.get_version_num() != 7 || parsed.to_string() != value {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(parsed)
}

fn database_error(error: rusqlite::Error) -> LedgerError {
    map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}

impl LedgerRepository {
    #[cfg(test)]
    pub(crate) fn reconcile_at(
        &mut self,
        observed_at_unix_ms: i64,
    ) -> Result<ReconciliationReport, LedgerError> {
        let transaction = self
            .connection
            .transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)
            .map_err(database_error)?;
        let report = reconcile_startup_in_transaction(
            &transaction,
            self.project_uuid,
            self.process_instance_id,
            observed_at_unix_ms,
        )?;
        transaction.commit().map_err(database_error)?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier, mpsc};
    use std::thread;
    use std::time::{Duration, Instant};

    use serde_json::json;
    use tempfile::tempdir;

    use super::*;
    use crate::canonical_json::canonical_sha256;
    use crate::ledger::read_pool::{LedgerReadPool, ReadPoolError};
    use crate::ledger::repository::LedgerRepository;
    use crate::ledger::repository::process::{
        HeartbeatAck, HeartbeatRenewal, ProcessStatusAt, verified_process_status_at,
    };
    use crate::ledger::repository::tests::{
        assert_reconciliation_noop_twice, config, database_path,
    };
    use crate::ledger::writer::LedgerWriterOwner;

    #[test]
    fn activation_fences_at_exact_expiry_and_second_pass_is_a_noop() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = config(&path, "reconciliation-exact-expiry");
        let first = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let first_process = first.identity.process_instance_id;
        drop(first);

        let mut second = LedgerRepository::activate_between(&config, 31_000, 70_000).unwrap();
        let second_process = second.identity.process_instance_id;
        let reconciled: (String, String, i64) = second
            .repository
            .connection
            .query_row(
                "SELECT process_instance_id, state, created_at_unix_ms
                 FROM process_instance_state_events
                 WHERE subject_process_instance_id = ?1 AND state = 'reconciled'",
                params![first_process.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(reconciled.0, second_process.to_string());
        assert_eq!(reconciled.1, "reconciled");
        assert_eq!(reconciled.2, 31_000);
        let second_expiry: i64 = second
            .repository
            .connection
            .query_row(
                "SELECT heartbeat_expires_at_unix_ms FROM process_instances
                 WHERE process_instance_id = ?1",
                params![second_process.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(second_expiry, 100_000);

        assert_reconciliation_noop_twice(&mut second.repository, 70_000);
    }

    #[test]
    fn renewal_before_exact_expiry_activation_keeps_the_process_live() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = config(&path, "reconciliation-renewal-before-activation");
        let mut first = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let first_process = first.identity.process_instance_id;
        let project_uuid = first.identity.project_uuid;

        assert_eq!(
            first
                .repository
                .renew_heartbeat(HeartbeatRenewal::new(31_000).unwrap())
                .unwrap(),
            HeartbeatAck::Applied {
                expires_at_unix_ms: 61_000,
            }
        );
        drop(first);

        let mut second = LedgerRepository::activate_at(&config, 31_000).unwrap();
        assert_ne!(second.identity.process_instance_id, first_process);
        assert_eq!(
            heartbeat_expiry(&second.repository.connection, first_process),
            61_000
        );
        assert_eq!(
            reconciliation_fence_count(&second.repository.connection, first_process),
            0
        );
        assert_eq!(
            verified_process_status_at(
                &second.repository.connection,
                project_uuid,
                first_process,
                31_000,
            )
            .unwrap(),
            ProcessStatusAt::Live
        );
        assert_reconciliation_noop_twice(&mut second.repository, 31_000);
    }

    #[test]
    fn renewal_racing_exact_expiry_activation_has_only_two_linearized_outcomes() {
        const COMPLETION_TIMEOUT: Duration = Duration::from_secs(8);

        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = config(&path, "reconciliation-renewal-activation-race");
        let first = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let first_process = first.identity.process_instance_id;
        let project_uuid = first.identity.project_uuid;
        let mut renewal_repository = first.repository;

        let barrier = Arc::new(Barrier::new(3));
        let (renewal_tx, renewal_rx) = mpsc::sync_channel(1);
        let renewal_barrier = Arc::clone(&barrier);
        let renewal_thread = thread::spawn(move || {
            renewal_barrier.wait();
            let result = renewal_repository.renew_heartbeat(HeartbeatRenewal::new(31_000).unwrap());
            let _ = renewal_tx.send(result);
        });

        let (activation_tx, activation_rx) = mpsc::sync_channel(1);
        let activation_barrier = Arc::clone(&barrier);
        let activation_thread = thread::spawn(move || {
            activation_barrier.wait();
            let result = LedgerRepository::activate_at(&config, 31_000);
            let _ = activation_tx.send(result);
        });

        barrier.wait();
        let renewal_delivery = renewal_rx.recv_timeout(COMPLETION_TIMEOUT);
        let activation_delivery = activation_rx.recv_timeout(COMPLETION_TIMEOUT);
        let renewal_join = renewal_thread.join();
        let activation_join = activation_thread.join();

        assert!(renewal_join.is_ok(), "renewal thread must not panic");
        assert!(activation_join.is_ok(), "activation thread must not panic");
        let renewal = renewal_delivery
            .expect("renewal must complete within the bounded timeout")
            .unwrap_or_else(|error| {
                assert_ne!(error.class(), LedgerErrorClass::Busy);
                panic!("renewal failed with {:?}", error.class());
            });
        let mut activated = activation_delivery
            .expect("activation must complete within the bounded timeout")
            .unwrap_or_else(|error| {
                assert_ne!(error.class(), LedgerErrorClass::Busy);
                panic!("activation failed with {:?}", error.class());
            });
        let reconciler_process = activated.identity.process_instance_id;

        match renewal {
            HeartbeatAck::Applied { expires_at_unix_ms } => {
                assert_eq!(expires_at_unix_ms, 61_000);
                assert_eq!(
                    heartbeat_expiry(&activated.repository.connection, first_process),
                    61_000
                );
                assert_eq!(
                    reconciliation_fence_count(&activated.repository.connection, first_process,),
                    0
                );
                assert_eq!(
                    verified_process_status_at(
                        &activated.repository.connection,
                        project_uuid,
                        first_process,
                        31_000,
                    )
                    .unwrap(),
                    ProcessStatusAt::Live
                );
            }
            HeartbeatAck::OriginatingProcessNotLive => {
                assert_eq!(
                    heartbeat_expiry(&activated.repository.connection, first_process),
                    31_000
                );
                assert_exact_reconciliation_fence(
                    &activated.repository.connection,
                    project_uuid,
                    reconciler_process,
                    first_process,
                    31_000,
                );
            }
            other => panic!("unexpected heartbeat race outcome: {other:?}"),
        }
        assert_reconciliation_noop_twice(&mut activated.repository, 31_000);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn wal_reader_keeps_its_snapshot_while_recovery_commits() {
        const ENTRY_TIMEOUT: Duration = Duration::from_secs(2);
        const ACTIVATION_TIMEOUT: Duration = Duration::from_secs(8);
        const READER_TIMEOUT: Duration = Duration::from_secs(12);

        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = config(&path, "reconciliation-wal-reader");
        let first = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let first_process = first.identity.process_instance_id;
        drop(first);

        let pool = LedgerReadPool::open(&path).unwrap();
        let reader_pool = pool.clone();
        let (entered_tx, entered_rx) = mpsc::sync_channel(1);
        let (release_tx, release_rx) = mpsc::sync_channel(1);
        let mut reader = tokio::spawn(async move {
            reader_pool
                .run(Instant::now() + Duration::from_secs(8), move |connection| {
                    connection
                        .execute_batch("BEGIN DEFERRED")
                        .map_err(|_| ReadPoolError::operation_failed())?;
                    let initial = read_process_snapshot(connection)?;
                    if entered_tx.send(initial).is_err() {
                        let _ = connection.execute_batch("ROLLBACK");
                        return Err(ReadPoolError::operation_failed());
                    }
                    if release_rx.recv_timeout(Duration::from_secs(10)).is_err() {
                        let _ = connection.execute_batch("ROLLBACK");
                        return Err(ReadPoolError::operation_failed());
                    }
                    let retained = read_process_snapshot(connection)?;
                    connection
                        .execute_batch("ROLLBACK")
                        .map_err(|_| ReadPoolError::operation_failed())?;
                    let refreshed = read_process_snapshot(connection)?;
                    Ok((initial, retained, refreshed))
                })
                .await
        });

        let initial = match entered_rx.recv_timeout(ENTRY_TIMEOUT) {
            Ok(initial) => initial,
            Err(error) => {
                let _ = release_tx.send(());
                if tokio::time::timeout(READER_TIMEOUT, &mut reader)
                    .await
                    .is_err()
                {
                    pool.abort();
                    reader.abort();
                    let _ = reader.await;
                }
                panic!("reader did not establish its snapshot: {error:?}");
            }
        };

        let (activation_tx, activation_rx) = mpsc::sync_channel(1);
        let activation_thread = thread::spawn(move || {
            let result = LedgerRepository::activate_at(&config, 31_000);
            let _ = activation_tx.send(result);
        });
        let activation_delivery = activation_rx.recv_timeout(ACTIVATION_TIMEOUT);

        let _ = release_tx.send(());
        let reader_delivery = tokio::time::timeout(READER_TIMEOUT, &mut reader).await;
        let activation_join = activation_thread.join();
        let close_result = if reader_delivery.is_ok() {
            pool.close(Instant::now() + Duration::from_secs(1)).await
        } else {
            pool.abort();
            reader.abort();
            let _ = reader.await;
            Err(ReadPoolError::operation_failed())
        };

        assert!(activation_join.is_ok(), "activation thread must not panic");
        let activated = activation_delivery
            .expect("WAL reader must not prevent activation from completing")
            .unwrap_or_else(|error| {
                assert_ne!(error.class(), LedgerErrorClass::Busy);
                panic!("activation failed with {:?}", error.class());
            });
        let snapshots = reader_delivery
            .expect("released WAL reader must finish within the bounded timeout")
            .expect("reader task must not panic")
            .expect("reader operation must succeed");
        close_result.expect("read pool must close after the reader finishes");

        assert_eq!(initial, (1, 0));
        assert_eq!(snapshots.0, initial);
        assert_eq!(snapshots.1, (1, 0));
        assert_eq!(snapshots.2, (2, 1));
        assert_exact_reconciliation_fence(
            &activated.repository.connection,
            activated.identity.project_uuid,
            activated.identity.process_instance_id,
            first_process,
            31_000,
        );
    }

    #[test]
    fn activation_recovers_a_process_after_abrupt_writer_abort() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = config(&path, "reconciliation-writer-abort");
        let activated = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let dead_process = activated.identity.process_instance_id;
        let (owner, client) = LedgerWriterOwner::start(activated.repository, 4).unwrap();
        owner.abort();
        drop(client);
        drop(owner);

        let mut recovered = LedgerRepository::activate_at(&config, 31_000).unwrap();
        let reconciler: String = recovered
            .repository
            .connection
            .query_row(
                "SELECT process_instance_id FROM process_instance_state_events
                 WHERE subject_process_instance_id = ?1 AND state = 'reconciled'",
                params![dead_process.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(
            reconciler,
            recovered.identity.process_instance_id.to_string()
        );
        assert_reconciliation_noop_twice(&mut recovered.repository, 31_000);
    }

    fn heartbeat_expiry(connection: &Connection, process_instance_id: Uuid) -> i64 {
        connection
            .query_row(
                "SELECT heartbeat_expires_at_unix_ms FROM process_instances
                 WHERE process_instance_id = ?1",
                params![process_instance_id.to_string()],
                |row| row.get(0),
            )
            .unwrap()
    }

    fn reconciliation_fence_count(
        connection: &Connection,
        subject_process_instance_id: Uuid,
    ) -> i64 {
        connection
            .query_row(
                "SELECT count(*) FROM process_instance_state_events
                 WHERE subject_process_instance_id = ?1 AND state = 'reconciled'",
                params![subject_process_instance_id.to_string()],
                |row| row.get(0),
            )
            .unwrap()
    }

    fn assert_exact_reconciliation_fence(
        connection: &Connection,
        project_uuid: Uuid,
        reconciler_process_instance_id: Uuid,
        subject_process_instance_id: Uuid,
        created_at_unix_ms: i64,
    ) {
        let fences = connection
            .prepare(
                "SELECT process_state_event_id, process_instance_id, state,
                        subject_process_instance_id, created_at_unix_ms,
                        canonical_payload_hash
                 FROM process_instance_state_events
                 WHERE subject_process_instance_id = ?1 AND state = 'reconciled'
                 ORDER BY event_seq",
            )
            .unwrap()
            .query_map(params![subject_process_instance_id.to_string()], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, String>(5)?,
                ))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(fences.len(), 1);
        let fence = &fences[0];
        let event_id = Uuid::parse_str(&fence.0).unwrap();
        assert_eq!(event_id.get_version_num(), 7);
        assert_eq!(fence.1, reconciler_process_instance_id.to_string());
        assert_eq!(fence.2, "reconciled");
        assert_eq!(fence.3, subject_process_instance_id.to_string());
        assert_eq!(fence.4, created_at_unix_ms);
        assert_eq!(
            fence.5,
            canonical_sha256(&json!({
                "process_state_event_id": event_id,
                "process_instance_id": reconciler_process_instance_id,
                "state": "reconciled",
                "subject_process_instance_id": subject_process_instance_id,
                "created_at_unix_ms": created_at_unix_ms,
            }))
            .unwrap()
        );
        assert_eq!(
            verified_process_status_at(
                connection,
                project_uuid,
                subject_process_instance_id,
                created_at_unix_ms,
            )
            .unwrap(),
            ProcessStatusAt::Terminal
        );
    }

    fn read_process_snapshot(connection: &Connection) -> Result<(i64, i64), ReadPoolError> {
        let process_count = connection
            .query_row("SELECT count(*) FROM process_instances", [], |row| {
                row.get(0)
            })
            .map_err(|_| ReadPoolError::operation_failed())?;
        let fence_count = connection
            .query_row(
                "SELECT count(*) FROM process_instance_state_events WHERE state = 'reconciled'",
                [],
                |row| row.get(0),
            )
            .map_err(|_| ReadPoolError::operation_failed())?;
        Ok((process_count, fence_count))
    }
}
