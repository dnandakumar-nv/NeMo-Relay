// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Sole-writer transaction adapter for sqlite-vec mutations.

use rusqlite::TransactionBehavior;

use super::{LedgerRepository, TransactionStartGuard, map_fs_error, map_sqlite_error};
use crate::ledger::fs::enforce_sidecar_permissions;
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::sqlite_vector_store::{
    VectorIndexTransactionAck, VectorIndexWriterAck, VectorIndexWriterCommand,
};

impl LedgerRepository {
    pub(crate) fn execute_vector_index_command_with_start_check<G: TransactionStartGuard>(
        &mut self,
        command: &VectorIndexWriterCommand,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<VectorIndexTransactionAck, LedgerError> {
        let database_path = self.database_path.clone();
        let process_instance_id = self.process_instance_id;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(VectorIndexTransactionAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(VectorIndexTransactionAck::TransactionNotStarted);
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
            return Ok(VectorIndexTransactionAck::TransactionNotStarted);
        }
        drop(start_guard);

        let acknowledgement = match command {
            VectorIndexWriterCommand::AuthorizeGeneration {
                vector_space_id,
                dimensions,
                created_at_unix_ms,
            } => VectorIndexWriterAck::GenerationAuthorized(Box::new(
                super::vector_index::authorize_generation(
                    &transaction,
                    vector_space_id,
                    *dimensions,
                    *created_at_unix_ms,
                )?,
            )),
            VectorIndexWriterCommand::ClaimRebuildLease {
                vector_space_id,
                observed_at_unix_ms,
            } => {
                VectorIndexWriterAck::RebuildLeaseClaimed(super::vector_index::claim_rebuild_lease(
                    &transaction,
                    vector_space_id,
                    process_instance_id,
                    *observed_at_unix_ms,
                )?)
            }
            VectorIndexWriterCommand::CreateGenerationObjects {
                fence,
                observed_at_unix_ms,
            } => VectorIndexWriterAck::GenerationObjectsCreated(
                super::vector_index::create_generation_objects(
                    &transaction,
                    fence,
                    *observed_at_unix_ms,
                )?,
            ),
            VectorIndexWriterCommand::UpsertActiveRecord { record } => {
                VectorIndexWriterAck::PointMutated(super::vector_index::upsert_active_record(
                    &transaction,
                    record,
                )?)
            }
            VectorIndexWriterCommand::DeleteActiveRecord {
                vector_space_id,
                record_id,
            } => VectorIndexWriterAck::PointMutated(super::vector_index::delete_active_record(
                &transaction,
                vector_space_id,
                *record_id,
            )?),
            VectorIndexWriterCommand::RenewRebuildLease {
                fence,
                observed_at_unix_ms,
            } => {
                VectorIndexWriterAck::RebuildLeaseMutated(super::vector_index::renew_rebuild_lease(
                    &transaction,
                    fence,
                    *observed_at_unix_ms,
                )?)
            }
            VectorIndexWriterCommand::ReleaseRebuildLease {
                fence,
                released_at_unix_ms,
            } => VectorIndexWriterAck::RebuildLeaseMutated(
                super::vector_index::release_rebuild_lease(
                    &transaction,
                    fence,
                    *released_at_unix_ms,
                )?,
            ),
            VectorIndexWriterCommand::PopulateRebuildChunk {
                fence,
                observed_at_unix_ms,
            } => VectorIndexWriterAck::RebuildStepped(super::vector_index::populate_rebuild_chunk(
                &transaction,
                fence,
                *observed_at_unix_ms,
            )?),
            VectorIndexWriterCommand::CatchUpRebuildChanges {
                fence,
                observed_at_unix_ms,
            } => {
                VectorIndexWriterAck::RebuildStepped(super::vector_index::catch_up_rebuild_changes(
                    &transaction,
                    fence,
                    *observed_at_unix_ms,
                )?)
            }
            VectorIndexWriterCommand::FlipRebuildGeneration {
                fence,
                activated_at_unix_ms,
            } => {
                VectorIndexWriterAck::RebuildFlipped(super::vector_index::flip_rebuild_generation(
                    &transaction,
                    fence,
                    *activated_at_unix_ms,
                )?)
            }
            VectorIndexWriterCommand::CleanupRetiredGeneration {
                vector_space_id,
                generation,
                dropped_at_unix_ms,
            } => VectorIndexWriterAck::RetiredGenerationCleaned(
                super::vector_index::cleanup_retired_generation(
                    &transaction,
                    vector_space_id,
                    *generation,
                    *dropped_at_unix_ms,
                )?,
            ),
            VectorIndexWriterCommand::MarkHealth {
                vector_space_id,
                expected_generation,
                expected_manifest_hash,
                target,
                stable_error_class,
                observed_at_unix_ms,
            } => VectorIndexWriterAck::HealthMarked(super::vector_index::mark_vector_index_health(
                &transaction,
                vector_space_id,
                *expected_generation,
                expected_manifest_hash,
                *target,
                stable_error_class,
                *observed_at_unix_ms,
            )?),
        };

        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction
            .commit()
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed))?;
        Ok(VectorIndexTransactionAck::Completed(acknowledgement))
    }
}
