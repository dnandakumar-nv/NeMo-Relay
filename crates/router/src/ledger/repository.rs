// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transactional migration, identity initialization, and reset operations.

#[allow(dead_code)] // Spec 08 Tasks 8-11 consume the active persistence authority.
pub(crate) mod active;
#[allow(dead_code)] // Spec 08 Tasks 10-11 consume atomic shape-2 admission.
pub(crate) mod active_decision;
#[allow(dead_code)] // Spec 08 Tasks 9-11 consume Active experiment and look authority.
pub(crate) mod active_learning;
#[allow(dead_code)] // Task 7 wires these typed Task 6 anchor commands to the sink.
pub(crate) mod anchors;
#[allow(dead_code)] // Task 10 background workers consume bounded durable work discovery.
pub(crate) mod background_work;
#[allow(dead_code)] // Spec 08 Task 7 consumes durable control reads and mutations.
pub(crate) mod control;
#[allow(dead_code)] // The Task 8 evaluator consumes these typed Task 6 commands.
pub(crate) mod cooloff;
#[allow(dead_code)] // Spec 07 routes immutable recommendation audits through the sole writer.
pub(crate) mod decision;
#[allow(dead_code)] // Spec 06 consumes the Task 10 durable embedding-job lease API.
pub(crate) mod embedding;
#[allow(dead_code)] // Spec 10 consumes bounded read-only inspection authority.
pub(crate) mod inspection;
#[allow(dead_code)] // Task 8 consumes Judge attempt and evaluation commands.
pub(crate) mod judge;
#[allow(dead_code)] // Task 8 owns atomic terminal vector graphs and materialization facts.
pub(crate) mod materialization;
#[cfg(test)]
mod materialization_tests;
#[cfg(test)]
pub(crate) use materialization_tests::{
    active_runtime_context_for_root, active_runtime_inspection_input, active_runtime_request,
    ready_evaluated_active_runtime_fixture,
    ready_evaluated_active_runtime_fixture_with_attribution_seconds,
    ready_evaluated_active_runtime_fixture_with_embedder, ready_evaluated_runtime_fixture,
    ready_retention_runtime_fixture,
};
pub(crate) mod process;
mod reconciliation;
#[allow(dead_code)] // Task 11 routes retention through the sole writer and runtime timer.
pub(crate) mod retention;
#[cfg(test)]
mod retention_race_tests;
#[allow(dead_code)] // Task 7 and Task 8 consume batch and Shadow commands.
pub(crate) mod shadow;
#[allow(dead_code)] // Task 8 consumes transaction-local vector catalog primitives.
pub(crate) mod vector_catalog;
#[allow(dead_code)] // Task 6 writer and read facades consume vector-index primitives.
pub(crate) mod vector_index;
#[allow(dead_code)] // Task 7 activation and Task 8 consume verified vector registry authority.
pub(crate) mod vector_registry;
#[allow(dead_code)] // Task 7 routes registry ensures through the sole writer.
pub(crate) mod vector_registry_writer;
#[allow(dead_code)] // Spec 07 consumes the one-snapshot projected neighbor service.
pub(crate) mod vector_search;
#[allow(dead_code)] // Task 6 owns the sole-writer sqlite-vec transaction adapter.
mod vector_store;
#[allow(dead_code)] // Task 10 rebuild ownership consumes bounded durable work discovery.
pub(crate) mod vector_work;

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::thread;
use std::time::{Duration, Instant};

use chrono::Utc;
use rusqlite::{
    Connection, Error as SqliteError, ErrorCode, OptionalExtension, Transaction,
    TransactionBehavior, params,
};
use serde::Deserialize;
use serde_json::{Value as Json, json};
use uuid::Uuid;
use zeroize::Zeroizing;

use super::cohort::CohortAssignmentAuthority;
use super::fs::{LedgerFsErrorKind, enforce_sidecar_permissions, open_secure_connection};
use super::migrations::{CURRENT_SCHEMA_VERSION, LEDGER_APPLICATION_ID, MIGRATIONS, Migration};
use super::model::{
    COHORT_ASSIGNMENT_ALGORITHM_V1, COHORT_SALT_BYTES, CohortSalt, LedgerError, LedgerErrorClass,
    LedgerRuntimeIdentity, PoolRuntimeIdentity, PoolVectorSpaceRuntimeIdentity,
};
use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::config::{RouterConfig, RouterMode};
use crate::fingerprint::sha256_hex;
use crate::sqlite_vec_extension::{
    register as register_sqlite_vec, verify_connection as verify_sqlite_vec,
};
use crate::sqlite_vec_schema::{Vec0RootName, Vec0SchemaAuthority, VectorIndexGeneration};
use crate::vector::{VectorDimensions, VectorSpaceId};

use self::vector_registry::{
    RegistryEnsureAck, VectorRegistryEnsure, ensure_vector_registry_in_transaction,
    prepare_vector_registry,
};

const BUSY_TIMEOUT: Duration = Duration::from_secs(5);
const PROCESS_HEARTBEAT_MILLIS: i64 = 30_000;
const INITIAL_ACTOR: &str = "router_activation";
const INITIAL_REASON: &str = "initialization";
const DYNAMIC_SCHEMA_VERSION: i64 = 4;
const VEC0_SCHEMA_OBJECT_COUNT: usize = 5;

/// One activated ledger and its immutable runtime identity snapshot.
pub(crate) struct ActivatedLedger {
    pub(crate) repository: LedgerRepository,
    pub(crate) identity: LedgerRuntimeIdentity,
    pub(crate) cohort_assignment: CohortAssignmentAuthority,
    pub(crate) registry: VectorRegistryEnsure,
    pub(crate) schema_report: SchemaVerificationReport,
    pub(crate) control_snapshot: Option<crate::control::RouterControlSnapshot>,
    pub(crate) control_saturated: bool,
}

/// Current-schema writer authority for inspection operations without runtime activation.
pub(crate) struct OperationsLedger {
    pub(crate) repository: LedgerRepository,
    pub(crate) project_uuid: Uuid,
    pub(crate) config_generation_id: String,
    pub(crate) pool_ids: Vec<String>,
    pub(crate) control_snapshot: crate::control::RouterControlSnapshot,
    pub(crate) cohort_assignment: CohortAssignmentAuthority,
    pub(crate) control_saturated: bool,
}

/// Sole owner of the activation SQLite connection and protected cohort access.
///
/// This type deliberately has no `Debug` implementation because it owns both a
/// filesystem location and assignment key material.
pub(crate) struct LedgerRepository {
    connection: Connection,
    database_path: PathBuf,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    config_generation_id: String,
    active_pool_ids: BTreeSet<String>,
    active_policy_versions: BTreeMap<String, String>,
    retention_days: u32,
    max_evidence_records: u64,
}

/// Authority retained by the writer while SQLite attempts to begin a transaction.
///
/// The second check after `BEGIN` is the domain linearization point. It lets a
/// synchronous abort invalidate a contended start without blocking on SQLite's
/// busy timeout.
pub(crate) trait TransactionStartGuard {
    fn permits_transaction(&self) -> bool;
}

impl TransactionStartGuard for () {
    fn permits_transaction(&self) -> bool {
        true
    }
}

struct PreparedLedgerMaterial {
    config_generation_id: String,
    canonical_config_json: String,
    policies: BTreeMap<String, PreparedPolicy>,
}

struct PreparedPolicy {
    policy_version_id: String,
    canonical_policy_json: String,
}

impl LedgerRepository {
    /// Securely opens, migrates, and initializes the configured Router ledger.
    pub(crate) fn activate(config: &RouterConfig) -> Result<ActivatedLedger, LedgerError> {
        Self::activate_with_clock(config, None)
    }

    /// Open one current ledger for bounded control operations without full runtime activation.
    pub(crate) fn open_operations(config: &RouterConfig) -> Result<OperationsLedger, LedgerError> {
        if config.mode == RouterMode::Off {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let prepared = PreparedLedgerMaterial::from_config(config)?;
        let database_path = PathBuf::from(&config.database_path);
        let mut connection = open_secure_connection(&database_path).map_err(map_fs_error)?;
        configure_connection(&connection)?;
        inspect_existing_database(&connection)?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;

        let now = now_unix_millis()?;
        let sqlite_version = verified_sqlite_version(&connection)?;
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed))?;
        let (project_uuid, _project_id, project_created) =
            initialize_project(&transaction, config, now, false)?;
        if project_created {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        inspection::verify_config_and_policy_authority(&transaction, project_uuid, &prepared)?;
        let process_instance_id = initialize_process(
            &transaction,
            project_uuid,
            &prepared.config_generation_id,
            &sqlite_version,
            now,
        )?;
        control::initialize_v6_authority(
            &transaction,
            project_uuid,
            process_instance_id,
            &prepared.config_generation_id,
            now,
        )?;
        let pool_ids = prepared.policies.keys().cloned().collect::<Vec<_>>();
        let authority = control::load_runtime_generation_authority(
            &transaction,
            project_uuid,
            &prepared.config_generation_id,
            &pool_ids,
        )?;
        verify_schema(&transaction, CURRENT_SCHEMA_VERSION)?;
        verify_required_pragmas(&transaction)?;
        verified_sqlite_version(&transaction)?;
        verify_integrity(&transaction)?;
        process::refresh_activation_heartbeat(
            &transaction,
            project_uuid,
            process_instance_id,
            now_unix_millis()?,
        )?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction
            .commit()
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed))?;

        let active_pool_ids = pool_ids.iter().cloned().collect();
        let active_policy_versions = prepared
            .policies
            .iter()
            .map(|(pool_id, policy)| (pool_id.clone(), policy.policy_version_id.clone()))
            .collect();
        Ok(OperationsLedger {
            repository: Self {
                connection,
                database_path,
                project_uuid,
                process_instance_id,
                config_generation_id: prepared.config_generation_id.clone(),
                active_pool_ids,
                active_policy_versions,
                retention_days: config.retention_days,
                max_evidence_records: config.max_evidence_records,
            },
            project_uuid,
            config_generation_id: prepared.config_generation_id,
            pool_ids,
            control_snapshot: authority.control.snapshot,
            cohort_assignment: authority.cohort_assignment,
            control_saturated: authority.control.saturated,
        })
    }

    #[cfg(test)]
    pub(crate) fn activate_at(
        config: &RouterConfig,
        activation_time_unix_ms: i64,
    ) -> Result<ActivatedLedger, LedgerError> {
        if activation_time_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Self::activate_with_clock(
            config,
            Some((activation_time_unix_ms, activation_time_unix_ms)),
        )
    }

    #[cfg(test)]
    pub(crate) fn activate_between(
        config: &RouterConfig,
        activation_start_unix_ms: i64,
        activation_end_unix_ms: i64,
    ) -> Result<ActivatedLedger, LedgerError> {
        if activation_start_unix_ms < 0 || activation_end_unix_ms < activation_start_unix_ms {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Self::activate_with_clock(
            config,
            Some((activation_start_unix_ms, activation_end_unix_ms)),
        )
    }

    fn activate_with_clock(
        config: &RouterConfig,
        fixed_times_unix_ms: Option<(i64, i64)>,
    ) -> Result<ActivatedLedger, LedgerError> {
        Self::activate_with_clock_and_hooks(config, fixed_times_unix_ms, || {}, |_| {})
    }

    fn activate_with_clock_and_hooks<AfterMigration, AfterActivationCutoff>(
        config: &RouterConfig,
        fixed_times_unix_ms: Option<(i64, i64)>,
        after_migration: AfterMigration,
        after_activation_cutoff: AfterActivationCutoff,
    ) -> Result<ActivatedLedger, LedgerError>
    where
        AfterMigration: FnOnce(),
        AfterActivationCutoff: FnOnce(i64),
    {
        let prepared = PreparedLedgerMaterial::from_config(config)?;
        if config.mode != RouterMode::Off {
            let _ = register_sqlite_vec();
        }
        let database_path = PathBuf::from(&config.database_path);
        let mut connection = open_secure_connection(&database_path).map_err(map_fs_error)?;
        configure_connection(&connection)?;
        if config.mode != RouterMode::Off {
            let _ = verify_sqlite_vec(&connection);
        }
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;

        let now = match fixed_times_unix_ms {
            Some((start, _)) => start,
            None => now_unix_millis()?,
        };
        let sqlite_version = verified_sqlite_version(&connection)?;
        let migration_transaction = connection
            .transaction_with_behavior(TransactionBehavior::Exclusive)
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed))?;
        let fresh_ledger = migrate(&migration_transaction, now, &sqlite_version)?;
        let (migration_project_uuid, _, migration_project_created) =
            initialize_project(&migration_transaction, config, now, fresh_ledger)?;
        if fresh_ledger {
            let new_pool_ids = initialize_config_and_policies(
                &migration_transaction,
                migration_project_uuid,
                &prepared,
                now,
                migration_project_created,
            )?;
            load_or_initialize_cohort(
                &migration_transaction,
                migration_project_uuid,
                now,
                migration_project_created,
            )?;
            for pool_id in prepared.policies.keys() {
                load_or_initialize_learning(
                    &migration_transaction,
                    migration_project_uuid,
                    pool_id,
                    now,
                    new_pool_ids.contains(pool_id),
                )?;
            }
        }
        verify_schema(&migration_transaction, CURRENT_SCHEMA_VERSION)?;
        verify_required_pragmas(&migration_transaction)?;
        verified_sqlite_version(&migration_transaction)?;
        verify_integrity(&migration_transaction)?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        migration_transaction
            .commit()
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed))?;

        after_migration();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed))?;
        let activation_now = match fixed_times_unix_ms {
            Some((start, _)) => start,
            None => now_unix_millis()?,
        };
        after_activation_cutoff(activation_now);
        let (project_uuid, project_id, project_created) =
            initialize_project(&transaction, config, activation_now, false)?;
        if project_created {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let new_pool_ids = initialize_config_and_policies(
            &transaction,
            project_uuid,
            &prepared,
            activation_now,
            project_created,
        )?;
        let cohort_generation_id =
            load_or_initialize_cohort(&transaction, project_uuid, activation_now, project_created)?;
        let (verified_cohort_generation_id, cohort_salt) = load_verified_cohort_generation(
            &transaction,
            project_uuid,
            &cohort_generation_id.to_string(),
        )?;
        if verified_cohort_generation_id != cohort_generation_id {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let cohort_assignment = CohortAssignmentAuthority::new(cohort_generation_id, cohort_salt);

        let mut pools = BTreeMap::new();
        for (pool_id, policy) in &prepared.policies {
            let learning_generation_id = load_or_initialize_learning(
                &transaction,
                project_uuid,
                pool_id,
                activation_now,
                new_pool_ids.contains(pool_id),
            )?;
            pools.insert(
                pool_id.clone(),
                PoolRuntimeIdentity {
                    policy_version_id: policy.policy_version_id.clone(),
                    learning_generation_id,
                    vector_space: None,
                },
            );
        }

        let policy_versions = prepared
            .policies
            .iter()
            .map(|(pool_id, policy)| (pool_id.clone(), policy.policy_version_id.clone()))
            .collect::<BTreeMap<_, _>>();
        active::ensure_outcome_policy_versions(
            &transaction,
            config,
            project_uuid,
            &prepared.config_generation_id,
            &policy_versions,
            activation_now,
        )?;
        let registry = prepare_vector_registry(
            config,
            project_uuid,
            &prepared.config_generation_id,
            &policy_versions,
            activation_now,
        )
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let registry_snapshot =
            match ensure_vector_registry_in_transaction(&transaction, &registry)? {
                RegistryEnsureAck::Applied(snapshot)
                | RegistryEnsureAck::AlreadyApplied(snapshot) => snapshot,
                RegistryEnsureAck::Conflict => {
                    return Err(LedgerErrorClass::CorruptDatabase.into());
                }
            };
        for (pool_id, mapping) in registry_snapshot.mappings {
            let pool = pools
                .get_mut(&pool_id)
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
            if mapping.mapping.policy_version_id != pool.policy_version_id {
                return Err(LedgerErrorClass::CorruptDatabase.into());
            }
            pool.vector_space = Some(PoolVectorSpaceRuntimeIdentity {
                profile_id: mapping.mapping.profile_id,
                embedder_profile_version_id: mapping.mapping.embedder_profile_version_id,
                canonicalizer_version_id: mapping.mapping.canonicalizer_version_id,
                vector_space_id: mapping.mapping.vector_space_id,
            });
        }
        for pool in &config.pools {
            let has_mapping = pools
                .get(&pool.id)
                .and_then(|identity| identity.vector_space.as_ref())
                .is_some();
            if has_mapping != pool.learning.is_some() {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }

        let process_instance_id = initialize_process(
            &transaction,
            project_uuid,
            &prepared.config_generation_id,
            &sqlite_version,
            activation_now,
        )?;
        reconciliation::reconcile_startup_in_transaction(
            &transaction,
            project_uuid,
            process_instance_id,
            activation_now,
        )?;
        let (control_snapshot, control_saturated) = if config.mode != RouterMode::Off {
            control::initialize_v6_authority(
                &transaction,
                project_uuid,
                process_instance_id,
                &prepared.config_generation_id,
                activation_now,
            )?;
            active::reconcile_active_startup(
                &transaction,
                project_uuid,
                process_instance_id,
                activation_now,
            )?;
            let pool_ids = pools.keys().cloned().collect::<Vec<_>>();
            let authority = control::load_control_authority_snapshot(
                &transaction,
                project_uuid,
                &prepared.config_generation_id,
                &pool_ids,
            )?;
            (Some(authority.snapshot), authority.saturated)
        } else {
            (None, false)
        };
        let schema_report = verify_schema(&transaction, CURRENT_SCHEMA_VERSION)?;
        verify_required_pragmas(&transaction)?;
        verified_sqlite_version(&transaction)?;
        verify_integrity(&transaction)?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let activation_end = match fixed_times_unix_ms {
            Some((_, end)) => end,
            None => now_unix_millis()?,
        };
        process::refresh_activation_heartbeat(
            &transaction,
            project_uuid,
            process_instance_id,
            activation_end,
        )?;
        transaction
            .commit()
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed))?;

        let identity = LedgerRuntimeIdentity {
            process_instance_id,
            project_uuid,
            project_id,
            cohort_generation_id,
            config_generation_id: prepared.config_generation_id,
            pools,
        };
        let active_pool_ids = identity.pools.keys().cloned().collect();
        let active_policy_versions = identity
            .pools
            .iter()
            .map(|(pool_id, pool)| (pool_id.clone(), pool.policy_version_id.clone()))
            .collect();
        Ok(ActivatedLedger {
            repository: Self {
                connection,
                database_path,
                project_uuid,
                process_instance_id,
                config_generation_id: identity.config_generation_id.clone(),
                active_pool_ids,
                active_policy_versions,
                retention_days: config.retention_days,
                max_evidence_records: config.max_evidence_records,
            },
            identity,
            cohort_assignment,
            registry,
            schema_report,
            control_snapshot,
            control_saturated,
        })
    }

    /// Append a new current learning generation for one configured pool.
    #[allow(dead_code)] // Spec 08 wires this control operation to the runtime.
    pub(crate) fn reset_pool(
        &mut self,
        pool_id: &str,
        actor: &str,
        reason: &str,
    ) -> Result<Uuid, LedgerError> {
        validate_reset_text(actor, reason)?;
        if !self.active_pool_ids.contains(pool_id) {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        enforce_sidecar_permissions(&self.database_path).map_err(map_fs_error)?;
        let now = now_unix_millis()?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed))?;
        let generation = append_learning_generation(
            &transaction,
            self.project_uuid,
            pool_id,
            actor,
            reason,
            now,
        )?;
        enforce_sidecar_permissions(&self.database_path).map_err(map_fs_error)?;
        transaction
            .commit()
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed))?;
        Ok(generation)
    }

    /// Atomically append new current learning generations for all configured pools.
    #[allow(dead_code)] // Spec 08 wires this control operation to the runtime.
    pub(crate) fn reset_project(
        &mut self,
        actor: &str,
        reason: &str,
    ) -> Result<BTreeMap<String, Uuid>, LedgerError> {
        validate_reset_text(actor, reason)?;
        enforce_sidecar_permissions(&self.database_path).map_err(map_fs_error)?;
        let now = now_unix_millis()?;
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed))?;
        let mut generations = BTreeMap::new();
        for pool_id in &self.active_pool_ids {
            let generation = append_learning_generation(
                &transaction,
                self.project_uuid,
                pool_id,
                actor,
                reason,
                now,
            )?;
            generations.insert(pool_id.clone(), generation);
        }
        enforce_sidecar_permissions(&self.database_path).map_err(map_fs_error)?;
        transaction
            .commit()
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed))?;
        Ok(generations)
    }

    /// Load one protected assignment salt inside the ledger implementation.
    #[allow(dead_code)] // Spec 08 consumes the salt for pseudonymous root assignment.
    pub(super) fn cohort_salt(
        &self,
        cohort_generation_id: Uuid,
    ) -> Result<CohortSalt, LedgerError> {
        let (stored_generation_id, salt) = load_verified_cohort_generation(
            &self.connection,
            self.project_uuid,
            &cohort_generation_id.to_string(),
        )?;
        if stored_generation_id != cohort_generation_id {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(salt)
    }

    /// Borrow the activation connection for the Task 6 sole-writer handoff.
    #[allow(dead_code)] // Task 6 transfers this exact connection to the writer.
    pub(super) fn connection_mut(&mut self) -> &mut Connection {
        &mut self.connection
    }

    #[cfg(test)]
    pub(crate) fn test_connection_mut(&mut self) -> &mut Connection {
        &mut self.connection
    }
}

impl PreparedLedgerMaterial {
    fn from_config(config: &RouterConfig) -> Result<Self, LedgerError> {
        let config_value = config
            .generation_value()
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let config_generation_id = canonical_sha256(&config_value)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let canonical_config_json = canonical_json(&config_value)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let policies = config
            .policy_generation_values()
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?
            .into_iter()
            .map(|(pool_id, value)| {
                let policy_version_id = canonical_sha256(&value)
                    .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
                let canonical_policy_json = canonical_json(&value)
                    .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
                Ok((
                    pool_id,
                    PreparedPolicy {
                        policy_version_id,
                        canonical_policy_json,
                    },
                ))
            })
            .collect::<Result<_, LedgerError>>()?;
        Ok(Self {
            config_generation_id,
            canonical_config_json,
            policies,
        })
    }
}

fn configure_connection(connection: &Connection) -> Result<(), LedgerError> {
    connection
        .busy_timeout(BUSY_TIMEOUT)
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::PragmaMismatch))?;
    connection
        .execute_batch("PRAGMA foreign_keys = ON;")
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::PragmaMismatch))?;
    verified_sqlite_version(connection)?;
    preflight_database(connection)?;

    let journal_mode = set_wal_mode_with_retry(connection)?;
    if !journal_mode.eq_ignore_ascii_case("wal") {
        return Err(LedgerErrorClass::PragmaMismatch.into());
    }
    connection
        .execute_batch("PRAGMA synchronous = FULL;")
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::PragmaMismatch))?;
    verify_required_pragmas(connection)?;
    Ok(())
}

fn set_wal_mode_with_retry(connection: &Connection) -> Result<String, LedgerError> {
    let deadline = Instant::now() + BUSY_TIMEOUT;
    loop {
        match connection.query_row("PRAGMA journal_mode = WAL", [], |row| row.get(0)) {
            Ok(mode) => return Ok(mode),
            Err(error)
                if matches!(
                    error.sqlite_error_code(),
                    Some(ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked)
                ) && Instant::now() < deadline =>
            {
                thread::sleep(Duration::from_millis(10));
            }
            Err(error) => {
                return Err(map_sqlite_error(&error, LedgerErrorClass::PragmaMismatch));
            }
        }
    }
}

fn preflight_database(connection: &Connection) -> Result<(), LedgerError> {
    let application_id = pragma_i64(connection, "application_id")?;
    let user_tables = user_table_count(connection)?;
    let user_version = pragma_i64(connection, "user_version")?;
    if application_id != 0 && application_id != i64::from(LEDGER_APPLICATION_ID) {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    if application_id == 0 && user_tables != 0 {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    if user_version > CURRENT_SCHEMA_VERSION {
        return Err(LedgerErrorClass::FutureSchema.into());
    }
    let empty = application_id == 0 && user_tables == 0 && user_version == 0;
    if !empty {
        let applied = load_migration_history(connection)?;
        validate_migration_history(&applied, user_version)?;
        verify_schema(connection, user_version)?;
    }
    verify_integrity(connection)
}

pub(crate) fn inspect_existing_database(connection: &Connection) -> Result<(), LedgerError> {
    preflight_database(connection)
}

fn verify_required_pragmas(connection: &Connection) -> Result<(), LedgerError> {
    if pragma_i64(connection, "foreign_keys")? != 1
        || pragma_i64(connection, "synchronous")? != 2
        || pragma_i64(connection, "busy_timeout")? != BUSY_TIMEOUT.as_millis() as i64
    {
        return Err(LedgerErrorClass::PragmaMismatch.into());
    }
    let journal_mode: String = connection
        .query_row("PRAGMA journal_mode", [], |row| row.get(0))
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::PragmaMismatch))?;
    if !journal_mode.eq_ignore_ascii_case("wal") {
        return Err(LedgerErrorClass::PragmaMismatch.into());
    }
    Ok(())
}

fn verified_sqlite_version(connection: &Connection) -> Result<String, LedgerError> {
    let runtime: String = connection
        .query_row("SELECT sqlite_version()", [], |row| row.get(0))
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::SqliteVersionMismatch))?;
    if runtime != rusqlite::version() {
        return Err(LedgerErrorClass::SqliteVersionMismatch.into());
    }
    Ok(runtime)
}

fn migrate(
    transaction: &Transaction<'_>,
    now: i64,
    sqlite_version: &str,
) -> Result<bool, LedgerError> {
    let application_id = pragma_i64(transaction, "application_id")?;
    let user_tables = user_table_count(transaction)?;
    let user_version = pragma_i64(transaction, "user_version")?;
    if user_version > CURRENT_SCHEMA_VERSION {
        return Err(LedgerErrorClass::FutureSchema.into());
    }

    let empty = user_tables == 0 && application_id == 0 && user_version == 0;
    if !empty && application_id != i64::from(LEDGER_APPLICATION_ID) {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }

    let applied = if empty {
        Vec::new()
    } else {
        load_migration_history(transaction)?
    };
    validate_migration_history(&applied, user_version)?;

    for migration in MIGRATIONS
        .iter()
        .filter(|migration| migration.version > user_version)
    {
        apply_migration(transaction, *migration, now, sqlite_version)?;
    }

    if pragma_i64(transaction, "application_id")? != i64::from(LEDGER_APPLICATION_ID)
        || pragma_i64(transaction, "user_version")? != CURRENT_SCHEMA_VERSION
    {
        return Err(LedgerErrorClass::MigrationFailed.into());
    }
    let final_history = load_migration_history(transaction)?;
    validate_migration_history(&final_history, CURRENT_SCHEMA_VERSION)?;
    Ok(empty)
}

fn load_migration_history(
    connection: &Connection,
) -> Result<Vec<(i64, String, String)>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT version, name, checksum_sha256 FROM schema_migrations ORDER BY version ASC",
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::InvalidMigrationHistory))?;
    let rows = statement
        .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::InvalidMigrationHistory))?;
    rows.collect::<Result<Vec<_>, _>>()
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::InvalidMigrationHistory))
}

fn validate_migration_history(
    applied: &[(i64, String, String)],
    user_version: i64,
) -> Result<(), LedgerError> {
    if applied
        .iter()
        .any(|(version, _, _)| *version > CURRENT_SCHEMA_VERSION)
    {
        return Err(LedgerErrorClass::FutureSchema.into());
    }
    let expected_len = usize::try_from(user_version)
        .map_err(|_| LedgerError::new(LedgerErrorClass::InvalidMigrationHistory))?;
    if applied.len() != expected_len {
        return Err(LedgerErrorClass::InvalidMigrationHistory.into());
    }
    for (index, (version, name, checksum)) in applied.iter().enumerate() {
        let expected_version = i64::try_from(index + 1)
            .map_err(|_| LedgerError::new(LedgerErrorClass::InvalidMigrationHistory))?;
        if *version != expected_version {
            return Err(LedgerErrorClass::InvalidMigrationHistory.into());
        }
        let migration = MIGRATIONS
            .get(index)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::FutureSchema))?;
        if name != migration.name {
            return Err(LedgerErrorClass::InvalidMigrationHistory.into());
        }
        if checksum != migration.checksum_sha256 {
            return Err(LedgerErrorClass::MigrationChecksumMismatch.into());
        }
    }
    Ok(())
}

fn apply_migration(
    transaction: &Transaction<'_>,
    migration: Migration,
    now: i64,
    sqlite_version: &str,
) -> Result<(), LedgerError> {
    transaction
        .execute_batch(migration.sql)
        .map_err(|error| map_migration_error(&error))?;
    transaction
        .execute(
            "INSERT INTO schema_migrations (
                version, name, checksum_sha256, applied_at_unix_ms,
                application_version, sqlite_version
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![
                migration.version,
                migration.name,
                migration.checksum_sha256,
                now,
                env!("CARGO_PKG_VERSION"),
                sqlite_version,
            ],
        )
        .map_err(|error| map_migration_error(&error))?;
    transaction
        .pragma_update(None, "user_version", migration.version)
        .map_err(|error| map_migration_error(&error))?;
    Ok(())
}

fn initialize_project(
    transaction: &Transaction<'_>,
    config: &RouterConfig,
    now: i64,
    allow_create: bool,
) -> Result<(Uuid, String, bool), LedgerError> {
    let stored = transaction
        .query_row(
            "SELECT project_uuid, project_id, created_at_unix_ms,
                    application_version, canonical_payload_hash
             FROM project_metadata WHERE singleton_key = 1",
            [],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            },
        )
        .optional()
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    if let Some((uuid, project_id, created_at, application_version, stored_hash)) = stored {
        let project_uuid = parse_uuid_v7(&uuid)?;
        if config
            .project_id
            .as_deref()
            .is_some_and(|configured| configured != project_id)
        {
            return Err(LedgerErrorClass::ProjectIdMismatch.into());
        }
        let expected_hash = hash_json(&json!({
            "project_uuid": uuid,
            "project_id": project_id,
            "created_at_unix_ms": created_at,
            "application_version": application_version,
        }))?;
        if stored_hash != expected_hash {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        return Ok((project_uuid, project_id, false));
    }
    if !allow_create {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let project_uuid = Uuid::now_v7();
    let project_id = config
        .project_id
        .clone()
        .unwrap_or_else(|| project_uuid.to_string());
    let payload_hash = hash_json(&json!({
        "project_uuid": project_uuid,
        "project_id": project_id,
        "created_at_unix_ms": now,
        "application_version": env!("CARGO_PKG_VERSION"),
    }))?;
    transaction
        .execute(
            "INSERT INTO project_metadata (
                singleton_key, project_uuid, project_id, created_at_unix_ms,
                application_version, canonical_payload_hash
             ) VALUES (1, ?1, ?2, ?3, ?4, ?5)",
            params![
                project_uuid.to_string(),
                project_id,
                now,
                env!("CARGO_PKG_VERSION"),
                payload_hash,
            ],
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    Ok((project_uuid, project_id, true))
}

fn initialize_config_and_policies(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    prepared: &PreparedLedgerMaterial,
    now: i64,
    project_created: bool,
) -> Result<BTreeSet<String>, LedgerError> {
    let config_history_exists: bool = transaction
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM config_generations WHERE project_uuid = ?1
             )",
            params![project_uuid.to_string()],
            |row| row.get(0),
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    if config_history_exists == project_created {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let mut new_pool_ids = BTreeSet::new();
    for pool_id in prepared.policies.keys() {
        let seen_in_config: bool = transaction
            .query_row(
                "SELECT EXISTS(
                    SELECT 1
                    FROM config_generations AS c, json_each(c.canonical_config_json, '$.pools') AS p
                    WHERE c.project_uuid = ?1
                      AND json_extract(p.value, '$.id') = ?2
                 )",
                params![project_uuid.to_string(), pool_id],
                |row| row.get(0),
            )
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
        let policy_history_exists: bool = transaction
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM policy_versions
                    WHERE project_uuid = ?1 AND pool_id = ?2
                 )",
                params![project_uuid.to_string(), pool_id],
                |row| row.get(0),
            )
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
        if seen_in_config != policy_history_exists {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        if !seen_in_config {
            new_pool_ids.insert(pool_id.clone());
        }
    }

    transaction
        .execute(
            "INSERT INTO config_generations (
                config_generation_id, project_uuid, canonical_config_json,
                canonical_payload_hash, created_at_unix_ms, application_version
             ) VALUES (?1, ?2, ?3, ?1, ?4, ?5)
             ON CONFLICT(config_generation_id) DO NOTHING",
            params![
                prepared.config_generation_id,
                project_uuid.to_string(),
                prepared.canonical_config_json,
                now,
                env!("CARGO_PKG_VERSION"),
            ],
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    let stored_config = transaction
        .query_row(
            "SELECT project_uuid, canonical_config_json, canonical_payload_hash
             FROM config_generations WHERE config_generation_id = ?1",
            params![prepared.config_generation_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    if stored_config
        != (
            project_uuid.to_string(),
            prepared.canonical_config_json.clone(),
            prepared.config_generation_id.clone(),
        )
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    for (pool_id, policy) in &prepared.policies {
        transaction
            .execute(
                "INSERT INTO policy_versions (
                    policy_version_id, project_uuid, pool_id,
                    canonical_policy_json, canonical_payload_hash, created_at_unix_ms
                 ) VALUES (?1, ?2, ?3, ?4, ?1, ?5)
                 ON CONFLICT(project_uuid, pool_id, policy_version_id) DO NOTHING",
                params![
                    policy.policy_version_id,
                    project_uuid.to_string(),
                    pool_id,
                    policy.canonical_policy_json,
                    now,
                ],
            )
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
        let stored_policy = transaction
            .query_row(
                "SELECT canonical_policy_json, canonical_payload_hash
                 FROM policy_versions
                 WHERE project_uuid = ?1 AND pool_id = ?2 AND policy_version_id = ?3",
                params![project_uuid.to_string(), pool_id, policy.policy_version_id,],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
        if stored_policy
            != (
                policy.canonical_policy_json.clone(),
                policy.policy_version_id.clone(),
            )
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
    }
    Ok(new_pool_ids)
}

fn load_or_initialize_cohort(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    now: i64,
    allow_create: bool,
) -> Result<Uuid, LedgerError> {
    let orphan_exists: bool = transaction
        .query_row(
            "SELECT EXISTS(
                SELECT 1
                FROM cohort_generations AS c
                LEFT JOIN cohort_generation_state_events AS s
                  ON s.project_uuid = c.project_uuid
                 AND s.cohort_generation_id = c.cohort_generation_id
                WHERE c.project_uuid = ?1 AND s.cohort_state_event_id IS NULL
             )",
            params![project_uuid.to_string()],
            |row| row.get(0),
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    if orphan_exists {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let stored = transaction
        .query_row(
            "SELECT s.cohort_generation_id, s.cohort_state_event_id,
                    s.actor, s.reason, s.created_at_unix_ms,
                    s.canonical_payload_hash
             FROM cohort_generation_state_events AS s
             WHERE s.project_uuid = ?1
             ORDER BY s.event_seq DESC LIMIT 1",
            params![project_uuid.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, String>(5)?,
                ))
            },
        )
        .optional()
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    if let Some((id, state_event_id, state_actor, state_reason, state_created_at, state_hash)) =
        stored
    {
        let generation_id = parse_uuid_v7(&id)?;
        let state_event_uuid = parse_uuid_v7(&state_event_id)?;
        let expected_state_hash = hash_json(&json!({
            "cohort_state_event_id": state_event_uuid,
            "project_uuid": project_uuid,
            "cohort_generation_id": generation_id,
            "state": "current",
            "actor": state_actor,
            "reason": state_reason,
            "created_at_unix_ms": state_created_at,
        }))?;
        if expected_state_hash != state_hash {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let (verified_generation_id, _salt) =
            load_verified_cohort_generation(transaction, project_uuid, &id)?;
        if verified_generation_id != generation_id {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        return Ok(generation_id);
    }

    let generations_exist: bool = transaction
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM cohort_generations WHERE project_uuid = ?1
             )",
            params![project_uuid.to_string()],
            |row| row.get(0),
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    if generations_exist {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    if !allow_create {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let mut bytes = Zeroizing::new([0_u8; COHORT_SALT_BYTES]);
    getrandom::fill(bytes.as_mut())
        .map_err(|_| LedgerError::new(LedgerErrorClass::RandomnessUnavailable))?;
    let salt = CohortSalt::new(bytes);
    let generation_id = Uuid::now_v7();
    let fingerprint = sha256_hex(salt.as_bytes());
    let payload_hash = hash_json(&json!({
        "cohort_generation_id": generation_id,
        "project_uuid": project_uuid,
        "assignment_algorithm": COHORT_ASSIGNMENT_ALGORITHM_V1,
        "salt_fingerprint_sha256": fingerprint,
        "actor": INITIAL_ACTOR,
        "reason": INITIAL_REASON,
        "created_at_unix_ms": now,
    }))?;
    transaction
        .execute(
            "INSERT INTO cohort_generations (
                cohort_generation_id, project_uuid, cohort_salt, assignment_algorithm,
                salt_fingerprint_sha256, actor, reason, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                generation_id.to_string(),
                project_uuid.to_string(),
                salt.as_bytes().as_slice(),
                COHORT_ASSIGNMENT_ALGORITHM_V1,
                fingerprint,
                INITIAL_ACTOR,
                INITIAL_REASON,
                now,
                payload_hash,
            ],
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    append_cohort_state_event(
        transaction,
        project_uuid,
        generation_id,
        INITIAL_ACTOR,
        INITIAL_REASON,
        now,
    )?;
    Ok(generation_id)
}

fn load_verified_cohort_generation(
    connection: &Connection,
    project_uuid: Uuid,
    cohort_generation_id: &str,
) -> Result<(Uuid, CohortSalt), LedgerError> {
    let (id, assignment_algorithm, fingerprint, actor, reason, created_at, stored_hash, bytes) =
        connection
            .query_row(
                "SELECT cohort_generation_id, assignment_algorithm,
                    salt_fingerprint_sha256, actor, reason,
                    created_at_unix_ms, canonical_payload_hash, cohort_salt
             FROM cohort_generations
             WHERE project_uuid = ?1 AND cohort_generation_id = ?2",
                params![project_uuid.to_string(), cohort_generation_id],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, i64>(5)?,
                        row.get::<_, String>(6)?,
                        row.get::<_, Vec<u8>>(7)?,
                    ))
                },
            )
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    let salt = CohortSalt::from_vec(bytes)?;
    let generation_id = parse_uuid_v7(&id)?;
    if id != cohort_generation_id
        || assignment_algorithm != COHORT_ASSIGNMENT_ALGORITHM_V1
        || !is_sha256(&fingerprint)
        || sha256_hex(salt.as_bytes()) != fingerprint
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let expected_hash = hash_json(&json!({
        "cohort_generation_id": generation_id,
        "project_uuid": project_uuid,
        "assignment_algorithm": assignment_algorithm,
        "salt_fingerprint_sha256": fingerprint,
        "actor": actor,
        "reason": reason,
        "created_at_unix_ms": created_at,
    }))?;
    if expected_hash != stored_hash {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok((generation_id, salt))
}

fn append_cohort_state_event(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    generation_id: Uuid,
    actor: &str,
    reason: &str,
    now: i64,
) -> Result<(), LedgerError> {
    let event_id = Uuid::now_v7();
    let payload_hash = hash_json(&json!({
        "cohort_state_event_id": event_id,
        "project_uuid": project_uuid,
        "cohort_generation_id": generation_id,
        "state": "current",
        "actor": actor,
        "reason": reason,
        "created_at_unix_ms": now,
    }))?;
    transaction
        .execute(
            "INSERT INTO cohort_generation_state_events (
                cohort_state_event_id, project_uuid, cohort_generation_id, state,
                actor, reason, created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, 'current', ?4, ?5, ?6, ?7)",
            params![
                event_id.to_string(),
                project_uuid.to_string(),
                generation_id.to_string(),
                actor,
                reason,
                now,
                payload_hash,
            ],
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    Ok(())
}

fn load_or_initialize_learning(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    pool_id: &str,
    now: i64,
    allow_create: bool,
) -> Result<Uuid, LedgerError> {
    let orphan_exists: bool = transaction
        .query_row(
            "SELECT EXISTS(
                SELECT 1
                FROM learning_generations AS l
                LEFT JOIN learning_generation_state_events AS s
                  ON s.project_uuid = l.project_uuid
                 AND s.pool_id = l.pool_id
                 AND s.learning_generation_id = l.learning_generation_id
                WHERE l.project_uuid = ?1 AND l.pool_id = ?2
                  AND s.learning_state_event_id IS NULL
             )",
            params![project_uuid.to_string(), pool_id],
            |row| row.get(0),
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    if orphan_exists {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let stored = transaction
        .query_row(
            "SELECT l.learning_generation_id, l.actor, l.reason,
                    l.created_at_unix_ms, l.canonical_payload_hash,
                    s.learning_state_event_id, s.actor, s.reason,
                    s.created_at_unix_ms, s.canonical_payload_hash
             FROM learning_generation_state_events AS s
             JOIN learning_generations AS l
               ON l.project_uuid = s.project_uuid
              AND l.pool_id = s.pool_id
              AND l.learning_generation_id = s.learning_generation_id
             WHERE s.project_uuid = ?1 AND s.pool_id = ?2
             ORDER BY s.event_seq DESC LIMIT 1",
            params![project_uuid.to_string(), pool_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, String>(9)?,
                ))
            },
        )
        .optional()
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    if let Some((
        id,
        actor,
        reason,
        created_at,
        stored_hash,
        state_event_id,
        state_actor,
        state_reason,
        state_created_at,
        state_stored_hash,
    )) = stored
    {
        let generation_id = parse_uuid_v7(&id)?;
        let state_event_uuid = parse_uuid_v7(&state_event_id)?;
        let expected_hash = hash_json(&json!({
            "learning_generation_id": id,
            "project_uuid": project_uuid,
            "pool_id": pool_id,
            "actor": actor,
            "reason": reason,
            "created_at_unix_ms": created_at,
        }))?;
        if expected_hash != stored_hash {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let expected_state_hash = hash_json(&json!({
            "learning_state_event_id": state_event_uuid,
            "project_uuid": project_uuid,
            "pool_id": pool_id,
            "learning_generation_id": generation_id,
            "state": "current",
            "actor": state_actor,
            "reason": state_reason,
            "created_at_unix_ms": state_created_at,
        }))?;
        if expected_state_hash != state_stored_hash {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        return Ok(generation_id);
    }
    let generations_exist: bool = transaction
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM learning_generations
                WHERE project_uuid = ?1 AND pool_id = ?2
             )",
            params![project_uuid.to_string(), pool_id],
            |row| row.get(0),
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    if generations_exist {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    if !allow_create {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    append_learning_generation(
        transaction,
        project_uuid,
        pool_id,
        INITIAL_ACTOR,
        INITIAL_REASON,
        now,
    )
}

fn append_learning_generation(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    pool_id: &str,
    actor: &str,
    reason: &str,
    now: i64,
) -> Result<Uuid, LedgerError> {
    let generation_id = Uuid::now_v7();
    let payload_hash = hash_json(&json!({
        "learning_generation_id": generation_id,
        "project_uuid": project_uuid,
        "pool_id": pool_id,
        "actor": actor,
        "reason": reason,
        "created_at_unix_ms": now,
    }))?;
    transaction
        .execute(
            "INSERT INTO learning_generations (
                learning_generation_id, project_uuid, pool_id, actor, reason,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                generation_id.to_string(),
                project_uuid.to_string(),
                pool_id,
                actor,
                reason,
                now,
                payload_hash,
            ],
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    let event_id = Uuid::now_v7();
    let event_hash = hash_json(&json!({
        "learning_state_event_id": event_id,
        "project_uuid": project_uuid,
        "pool_id": pool_id,
        "learning_generation_id": generation_id,
        "state": "current",
        "actor": actor,
        "reason": reason,
        "created_at_unix_ms": now,
    }))?;
    transaction
        .execute(
            "INSERT INTO learning_generation_state_events (
                learning_state_event_id, project_uuid, pool_id,
                learning_generation_id, state, actor, reason,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, 'current', ?5, ?6, ?7, ?8)",
            params![
                event_id.to_string(),
                project_uuid.to_string(),
                pool_id,
                generation_id.to_string(),
                actor,
                reason,
                now,
                event_hash,
            ],
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    Ok(generation_id)
}

fn initialize_process(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    config_generation_id: &str,
    sqlite_version: &str,
    now: i64,
) -> Result<Uuid, LedgerError> {
    let process_instance_id = Uuid::now_v7();
    let heartbeat_expires_at = now
        .checked_add(PROCESS_HEARTBEAT_MILLIS)
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let payload_hash = hash_json(&json!({
        "process_instance_id": process_instance_id,
        "project_uuid": project_uuid,
        "config_generation_id": config_generation_id,
        "application_version": env!("CARGO_PKG_VERSION"),
        "sqlite_version": sqlite_version,
        "started_at_unix_ms": now,
        "heartbeat_expires_at_unix_ms": heartbeat_expires_at,
    }))?;
    transaction
        .execute(
            "INSERT INTO process_instances (
                process_instance_id, project_uuid, config_generation_id,
                application_version, sqlite_version, started_at_unix_ms,
                heartbeat_expires_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                process_instance_id.to_string(),
                project_uuid.to_string(),
                config_generation_id,
                env!("CARGO_PKG_VERSION"),
                sqlite_version,
                now,
                heartbeat_expires_at,
                payload_hash,
            ],
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    let event_id = Uuid::now_v7();
    let event_hash = hash_json(&json!({
        "process_state_event_id": event_id,
        "process_instance_id": process_instance_id,
        "state": "started",
        "subject_process_instance_id": process_instance_id,
        "created_at_unix_ms": now,
    }))?;
    transaction
        .execute(
            "INSERT INTO process_instance_state_events (
                process_state_event_id, process_instance_id, state,
                subject_process_instance_id, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (?1, ?2, 'started', ?2, ?3, ?4)",
            params![
                event_id.to_string(),
                process_instance_id.to_string(),
                now,
                event_hash,
            ],
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::IdentityInvariant))?;
    Ok(process_instance_id)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SchemaObject {
    object_type: String,
    name: String,
    table_name: String,
    sql: Option<String>,
}

impl SchemaObject {
    fn key(&self) -> (String, String) {
        (self.object_type.clone(), self.name.clone())
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct ManifestSchemaObject {
    #[serde(rename = "type")]
    object_type: String,
    name: String,
    table_name: String,
    sql_sha256: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ManifestLifecycle {
    Building,
    Active,
    Unavailable,
    Corrupt,
    Retired,
    Dropped,
}

impl ManifestLifecycle {
    fn parse(value: &str) -> Result<Self, LedgerError> {
        match value {
            "building" => Ok(Self::Building),
            "active" => Ok(Self::Active),
            "unavailable" => Ok(Self::Unavailable),
            "corrupt" => Ok(Self::Corrupt),
            "retired" => Ok(Self::Retired),
            "dropped" => Ok(Self::Dropped),
            _ => Err(LedgerErrorClass::CorruptDatabase.into()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct MissingDynamicGeneration {
    pub(crate) vector_space_id: String,
    pub(crate) generation: i64,
}

#[derive(Debug, Clone)]
struct DynamicSchemaManifest {
    generation: MissingDynamicGeneration,
    lifecycle: ManifestLifecycle,
    objects: Vec<ManifestSchemaObject>,
}

/// Extension-independent dynamic schema findings consumed by Task 6 recovery.
#[allow(dead_code)] // Task 6 consumes the per-lifecycle generation findings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct SchemaVerificationReport {
    pub(crate) missing_building: Vec<MissingDynamicGeneration>,
    pub(crate) missing_active: Vec<MissingDynamicGeneration>,
    pub(crate) missing_retired: Vec<MissingDynamicGeneration>,
}

fn verify_schema(
    connection: &Connection,
    version: i64,
) -> Result<SchemaVerificationReport, LedgerError> {
    let actual = schema_objects(connection)?;
    let expected_static = reference_schema_objects(version)?;
    let manifests = if version >= DYNAMIC_SCHEMA_VERSION {
        load_dynamic_schema_manifests(connection)?
    } else {
        Vec::new()
    };
    let report = verify_schema_objects(actual, expected_static, manifests)?;

    let mut foreign_keys = connection
        .prepare("PRAGMA foreign_key_check")
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::CorruptDatabase))?;
    let mut rows = foreign_keys
        .query([])
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::CorruptDatabase))?;
    if rows
        .next()
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::CorruptDatabase))?
        .is_some()
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(report)
}

fn verify_schema_objects(
    actual: Vec<SchemaObject>,
    expected_static: Vec<SchemaObject>,
    manifests: Vec<DynamicSchemaManifest>,
) -> Result<SchemaVerificationReport, LedgerError> {
    let mut remaining = schema_object_map(actual)?;
    let mut static_keys = BTreeSet::new();
    for expected in expected_static {
        let key = expected.key();
        static_keys.insert(key.clone());
        if remaining.remove(&key).as_ref() != Some(&expected) {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }
    verify_dynamic_schema(remaining.into_values().collect(), manifests, &static_keys)
}

fn verify_dynamic_schema(
    actual: Vec<SchemaObject>,
    manifests: Vec<DynamicSchemaManifest>,
    static_keys: &BTreeSet<(String, String)>,
) -> Result<SchemaVerificationReport, LedgerError> {
    let actual = schema_object_map(actual)?;
    let actual_keys = actual.keys().cloned().collect::<BTreeSet<_>>();
    let mut authorized = BTreeMap::new();
    for (manifest_index, manifest) in manifests.iter().enumerate() {
        for expected in &manifest.objects {
            let key = (expected.object_type.clone(), expected.name.clone());
            if static_keys.contains(&key)
                || authorized
                    .insert(key, (manifest_index, expected.clone()))
                    .is_some()
            {
                return Err(LedgerErrorClass::CorruptDatabase.into());
            }
        }
    }

    for (key, object) in &actual {
        let Some((manifest_index, expected)) = authorized.get(key) else {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        };
        if manifests[*manifest_index].lifecycle == ManifestLifecycle::Dropped
            || object.table_name != expected.table_name
            || object
                .sql
                .as_deref()
                .is_none_or(|sql| sha256_hex(sql.as_bytes()) != expected.sql_sha256)
        {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }

    let mut report = SchemaVerificationReport::default();
    for manifest in manifests {
        let present = manifest
            .objects
            .iter()
            .filter(|object| {
                actual_keys.contains(&(object.object_type.clone(), object.name.clone()))
            })
            .count();
        if present == manifest.objects.len() {
            continue;
        }
        if present != 0 {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
        match manifest.lifecycle {
            ManifestLifecycle::Building => report.missing_building.push(manifest.generation),
            ManifestLifecycle::Active
            | ManifestLifecycle::Unavailable
            | ManifestLifecycle::Corrupt => report.missing_active.push(manifest.generation),
            ManifestLifecycle::Retired => report.missing_retired.push(manifest.generation),
            ManifestLifecycle::Dropped => {}
        }
    }
    Ok(report)
}

fn schema_object_map(
    objects: Vec<SchemaObject>,
) -> Result<BTreeMap<(String, String), SchemaObject>, LedgerError> {
    let mut mapped = BTreeMap::new();
    for object in objects {
        if mapped.insert(object.key(), object).is_some() {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }
    Ok(mapped)
}

fn load_dynamic_schema_manifests(
    connection: &Connection,
) -> Result<Vec<DynamicSchemaManifest>, LedgerError> {
    // The expected-object document is immutable schema authority. Task 6 owns
    // complete validation of the manifest's mutable lifecycle/progress payload.
    let mut statement = connection
        .prepare(
            "SELECT vector_space_id, generation, state, root_table_name, dimensions,
                    expected_schema_objects_json, expected_schema_objects_sha256
             FROM vector_index_manifest
             ORDER BY vector_space_id, generation",
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::CorruptDatabase))?;
    let rows = statement
        .query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, i64>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
            ))
        })
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::CorruptDatabase))?;
    rows.map(|row| {
        let (
            vector_space_id,
            generation,
            state,
            root_table_name,
            dimensions,
            objects_json,
            objects_sha256,
        ) = row.map_err(|error| map_sqlite_error(&error, LedgerErrorClass::CorruptDatabase))?;
        parse_dynamic_schema_manifest(
            vector_space_id,
            generation,
            &state,
            &root_table_name,
            dimensions,
            &objects_json,
            &objects_sha256,
        )
    })
    .collect()
}

fn parse_dynamic_schema_manifest(
    vector_space_id: String,
    generation: i64,
    state: &str,
    root_table_name: &str,
    dimensions: i64,
    objects_json: &str,
    objects_sha256: &str,
) -> Result<DynamicSchemaManifest, LedgerError> {
    if !is_sha256(objects_sha256) {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let typed_space = VectorSpaceId::new(vector_space_id.clone())
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let typed_generation = VectorIndexGeneration::new(generation)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let typed_dimensions = u32::try_from(dimensions)
        .ok()
        .and_then(|value| VectorDimensions::new(value).ok())
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    let authority = Vec0SchemaAuthority::new(
        Vec0RootName::new(typed_space, typed_generation),
        typed_dimensions,
    );
    if root_table_name != authority.root().as_str()
        || objects_json != authority.manifest_json()
        || objects_sha256 != authority.manifest_sha256()
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let value: Json = serde_json::from_str(objects_json)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if canonical_json(&value).ok().as_deref() != Some(objects_json)
        || canonical_sha256(&value).ok().as_deref() != Some(objects_sha256)
    {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    let objects: Vec<ManifestSchemaObject> = serde_json::from_value(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CorruptDatabase))?;
    if objects.len() != VEC0_SCHEMA_OBJECT_COUNT {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }

    let root = authority.root().as_str().to_string();
    let expected_names = [
        root.clone(),
        format!("{root}_info"),
        format!("{root}_chunks"),
        format!("{root}_rowids"),
        format!("{root}_vector_chunks00"),
    ]
    .into_iter()
    .collect::<BTreeSet<_>>();
    let mut names = BTreeSet::new();
    for object in &objects {
        if object.object_type != "table"
            || object.table_name != object.name
            || !is_sha256(&object.sql_sha256)
            || !names.insert(object.name.clone())
        {
            return Err(LedgerErrorClass::CorruptDatabase.into());
        }
    }
    if names != expected_names {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }

    Ok(DynamicSchemaManifest {
        generation: MissingDynamicGeneration {
            vector_space_id,
            generation,
        },
        lifecycle: ManifestLifecycle::parse(state)?,
        objects,
    })
}

fn schema_objects(connection: &Connection) -> Result<Vec<SchemaObject>, LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT type, name, tbl_name, sql FROM sqlite_schema
             WHERE type IN ('table', 'index', 'trigger', 'view')
               AND name NOT LIKE 'sqlite_%'
             ORDER BY type ASC, name ASC",
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::CorruptDatabase))?;
    statement
        .query_map([], |row| {
            Ok(SchemaObject {
                object_type: row.get(0)?,
                name: row.get(1)?,
                table_name: row.get(2)?,
                sql: row.get(3)?,
            })
        })
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::CorruptDatabase))?
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::CorruptDatabase))
}

fn reference_schema_objects(version: i64) -> Result<Vec<SchemaObject>, LedgerError> {
    if !(0..=CURRENT_SCHEMA_VERSION).contains(&version) {
        return Err(LedgerErrorClass::InvalidMigrationHistory.into());
    }
    let reference = Connection::open_in_memory()
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::CorruptDatabase))?;
    for migration in MIGRATIONS
        .iter()
        .filter(|migration| migration.version <= version)
    {
        reference
            .execute_batch(migration.sql)
            .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::CorruptDatabase))?;
    }
    schema_objects(&reference)
}

fn verify_integrity(connection: &Connection) -> Result<(), LedgerError> {
    let result: String = connection
        .query_row("PRAGMA integrity_check(1)", [], |row| row.get(0))
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::CorruptDatabase))?;
    if result != "ok" {
        return Err(LedgerErrorClass::CorruptDatabase.into());
    }
    Ok(())
}

fn pragma_i64(connection: &Connection, name: &str) -> Result<i64, LedgerError> {
    let sql = match name {
        "application_id" => "PRAGMA application_id",
        "busy_timeout" => "PRAGMA busy_timeout",
        "foreign_keys" => "PRAGMA foreign_keys",
        "synchronous" => "PRAGMA synchronous",
        "user_version" => "PRAGMA user_version",
        _ => return Err(LedgerErrorClass::PragmaMismatch.into()),
    };
    connection
        .query_row(sql, [], |row| row.get(0))
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::PragmaMismatch))
}

fn user_table_count(connection: &Connection) -> Result<i64, LedgerError> {
    connection
        .query_row(
            "SELECT count(*) FROM sqlite_schema
             WHERE type = 'table' AND name NOT LIKE 'sqlite_%'",
            [],
            |row| row.get(0),
        )
        .map_err(|error| map_sqlite_error(&error, LedgerErrorClass::CorruptDatabase))
}

fn hash_json(value: &Json) -> Result<String, LedgerError> {
    canonical_sha256(value).map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn parse_uuid_v7(value: &str) -> Result<Uuid, LedgerError> {
    let parsed = Uuid::parse_str(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if parsed.get_version_num() != 7 || parsed.to_string() != value {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(parsed)
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn validate_reset_text(actor: &str, reason: &str) -> Result<(), LedgerError> {
    let valid_actor =
        !actor.trim().is_empty() && actor.len() <= 128 && !actor.chars().any(char::is_control);
    let valid_reason =
        !reason.trim().is_empty() && reason.len() <= 512 && !reason.chars().any(char::is_control);
    if !valid_actor || !valid_reason {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn now_unix_millis() -> Result<i64, LedgerError> {
    let now = Utc::now().timestamp_millis();
    if now < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(now)
}

fn map_fs_error(error: super::fs::LedgerFsError) -> LedgerError {
    match error.kind() {
        LedgerFsErrorKind::InvalidFilesystem => LedgerErrorClass::InvalidFilesystem.into(),
        LedgerFsErrorKind::InvalidPermissions => LedgerErrorClass::InvalidPermissions.into(),
        LedgerFsErrorKind::UnsupportedSecurity => {
            LedgerErrorClass::UnsupportedFilesystemSecurity.into()
        }
        LedgerFsErrorKind::OpenFailed => LedgerErrorClass::OpenFailed.into(),
    }
}

fn map_migration_error(error: &SqliteError) -> LedgerError {
    let class = match error.sqlite_error_code() {
        Some(ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked) => LedgerErrorClass::Busy,
        Some(ErrorCode::DatabaseCorrupt | ErrorCode::NotADatabase) => {
            LedgerErrorClass::CorruptDatabase
        }
        Some(ErrorCode::CannotOpen) => LedgerErrorClass::OpenFailed,
        Some(ErrorCode::PermissionDenied | ErrorCode::ReadOnly) => {
            LedgerErrorClass::InvalidPermissions
        }
        _ => LedgerErrorClass::MigrationFailed,
    };
    class.into()
}

fn map_sqlite_error(error: &SqliteError, fallback: LedgerErrorClass) -> LedgerError {
    let class = match error.sqlite_error_code() {
        Some(ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked) => LedgerErrorClass::Busy,
        Some(ErrorCode::DatabaseCorrupt | ErrorCode::NotADatabase) => {
            LedgerErrorClass::CorruptDatabase
        }
        Some(ErrorCode::CannotOpen) => LedgerErrorClass::OpenFailed,
        Some(ErrorCode::PermissionDenied | ErrorCode::ReadOnly) => {
            LedgerErrorClass::InvalidPermissions
        }
        Some(ErrorCode::ConstraintViolation) => LedgerErrorClass::IdentityInvariant,
        _ => fallback,
    };
    class.into()
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::Path;
    use std::process::Command;
    use std::sync::{Arc, Barrier, mpsc};
    use std::thread;

    use rusqlite::Connection;
    use serde_json::json;
    use tempfile::tempdir;

    use super::*;

    const COHORT_CHILD_DATABASE_ENV: &str = "NEMO_RELAY_ROUTER_COHORT_CHILD_DATABASE";
    const COHORT_CHILD_RESULT_PREFIX: &str = "NEMO_RELAY_ROUTER_COHORT_RESULT=";

    pub(super) fn config(path: &Path, project_id: &str) -> RouterConfig {
        serde_json::from_value(json!({
            "version": 1,
            "mode": "shadow",
            "project_id": project_id,
            "database_path": path.to_string_lossy(),
            "retention_days": 30,
            "max_evidence_records": 1000,
            "embedders": [{
                "id": "embedder-a",
                "base_url": "http://127.0.0.1:8080/v1",
                "model": "embedding-model-a",
                "provider_revision": "2026-07-01",
                "dimensions": 1024,
                "api_key_env": "ROUTER_TEST_EMBEDDING_SECRET",
                "timeout_ms": 10000
            }],
            "pools": [{
                "id": "pool-a",
                "api_family": "openai_chat_completions",
                "anchor_models": ["anchor-a"],
                "anchor_revision": "2026-07-01",
                "sampling_probability": 0.25,
                "max_candidates_per_sample": 1,
                "selector": {
                    "tenant_ids": ["tenant-private-a"],
                    "agent_ids": ["agent-private-a"],
                    "metadata_equals": {"region": "private-region-a"}
                },
                "concurrency": {"shadow": 2, "judge": 1},
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
        .expect("ledger test configuration should deserialize")
    }

    fn two_pool_config(path: &Path, project_id: &str) -> RouterConfig {
        let mut config = config(path, project_id);
        let mut second = config.pools[0].clone();
        second.id = "pool-b".into();
        second.anchor_models = vec!["anchor-b".into()];
        second.candidates[0].id = "candidate-b".into();
        second.candidates[0].model = "candidate-model-b".into();
        config.pools.push(second);
        config
    }

    pub(super) fn database_path(directory: &tempfile::TempDir) -> PathBuf {
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(directory.path(), fs::Permissions::from_mode(0o700))
                .expect("temporary ledger root should be owner-only");
        }
        directory.path().join("ledger/router.db")
    }

    fn scalar_i64(connection: &Connection, sql: &str) -> i64 {
        connection
            .query_row(sql, [], |row| row.get(0))
            .expect("ledger scalar query should succeed")
    }

    fn all_table_counts(connection: &Connection) -> Vec<(String, i64)> {
        let table_names = connection
            .prepare(
                "SELECT name FROM sqlite_schema
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                 ORDER BY name",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        table_names
            .into_iter()
            .map(|table_name| {
                let quoted = table_name.replace('"', "\"\"");
                let count = connection
                    .query_row(&format!("SELECT count(*) FROM \"{quoted}\""), [], |row| {
                        row.get(0)
                    })
                    .unwrap();
                (table_name, count)
            })
            .collect()
    }

    pub(super) fn recovery_snapshot(connection: &Connection) -> Vec<(String, String)> {
        use std::fmt::Write as _;

        let tables = connection
            .prepare(
                "SELECT name FROM sqlite_schema
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                   AND name <> 'schema_migrations'
                 ORDER BY name",
            )
            .unwrap()
            .query_map([], |row| row.get::<_, String>(0))
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        let mut snapshot = Vec::new();
        for table in tables {
            let mut statement = connection
                .prepare(&format!("SELECT * FROM \"{table}\" ORDER BY rowid"))
                .unwrap();
            let column_count = statement.column_count();
            let mut rows = statement.query([]).unwrap();
            while let Some(row) = rows.next().unwrap() {
                let mut encoded = String::new();
                for index in 0..column_count {
                    if index != 0 {
                        encoded.push('|');
                    }
                    match row.get_ref(index).unwrap() {
                        rusqlite::types::ValueRef::Null => encoded.push('n'),
                        rusqlite::types::ValueRef::Integer(value) => {
                            write!(encoded, "i{value}").unwrap();
                        }
                        rusqlite::types::ValueRef::Real(value) => {
                            write!(encoded, "r{:016x}", value.to_bits()).unwrap();
                        }
                        rusqlite::types::ValueRef::Text(value) => {
                            encoded.push('t');
                            for byte in value {
                                write!(encoded, "{byte:02x}").unwrap();
                            }
                        }
                        rusqlite::types::ValueRef::Blob(value) => {
                            encoded.push('b');
                            for byte in value {
                                write!(encoded, "{byte:02x}").unwrap();
                            }
                        }
                    }
                }
                snapshot.push((table.clone(), encoded));
            }
        }
        snapshot
    }

    pub(super) fn assert_reconciliation_noop_twice(
        repository: &mut LedgerRepository,
        observed_at_unix_ms: i64,
    ) {
        let before = recovery_snapshot(&repository.connection);
        for _ in 0..2 {
            assert_eq!(
                repository.reconcile_at(observed_at_unix_ms).unwrap(),
                super::reconciliation::ReconciliationReport::default()
            );
            assert_eq!(recovery_snapshot(&repository.connection), before);
        }
    }

    fn schema_snapshot(connection: &Connection) -> Vec<(String, String, Option<String>)> {
        connection
            .prepare(
                "SELECT type, name, sql FROM sqlite_schema
                 WHERE name NOT LIKE 'sqlite_%'
                 ORDER BY type, name",
            )
            .unwrap()
            .query_map([], |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)))
            .unwrap()
            .collect::<rusqlite::Result<_>>()
            .unwrap()
    }

    fn set_delete_journal_mode(connection: &Connection) {
        let mode: String = connection
            .query_row("PRAGMA journal_mode = DELETE", [], |row| row.get(0))
            .unwrap();
        assert_eq!(mode, "delete");
    }

    fn assert_refusal_left_delete_journal_untouched(path: &Path) {
        let connection = Connection::open(path).unwrap();
        let mode: String = connection
            .query_row("PRAGMA journal_mode", [], |row| row.get(0))
            .unwrap();
        assert_eq!(mode, "delete");
        drop(connection);
        assert!(!path.with_file_name("router.db-wal").exists());
        assert!(!path.with_file_name("router.db-shm").exists());
    }

    fn delete_all_domain_rows(connection: &Connection) {
        connection
            .execute_batch("PRAGMA foreign_keys = OFF;")
            .unwrap();
        let tables = {
            let mut statement = connection
                .prepare(
                    "SELECT name FROM sqlite_schema
                     WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                       AND name <> 'schema_migrations'",
                )
                .unwrap();
            statement
                .query_map([], |row| row.get::<_, String>(0))
                .unwrap()
                .collect::<Result<Vec<_>, _>>()
                .unwrap()
        };
        for table in tables {
            let quoted = table.replace('"', "\"\"");
            connection
                .execute(&format!("DELETE FROM \"{quoted}\""), [])
                .unwrap();
        }
    }

    #[test]
    fn fresh_and_reopen_preserve_stable_identity_and_cohort_assignment() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = config(&path, "stable-project");
        let raw_root_uuid = Uuid::parse_str("00112233-4455-6677-8899-aabbccddeeff").unwrap();

        let first = LedgerRepository::activate(&config).unwrap();
        let first_identity = first.identity.clone();
        let first_assignment = first
            .cohort_assignment
            .assign(crate::ledger::cohort::CohortAssignmentRequest::new(
                raw_root_uuid,
                &first_identity.config_generation_id,
                "pool-a",
                "candidate-a",
                0.125,
                0.25,
            ))
            .unwrap();
        assert_eq!(
            pragma_i64(&first.repository.connection, "foreign_keys").unwrap(),
            1
        );
        assert_eq!(
            pragma_i64(&first.repository.connection, "synchronous").unwrap(),
            2
        );
        assert_eq!(
            pragma_i64(&first.repository.connection, "busy_timeout").unwrap(),
            5000
        );
        drop(first);

        let second = LedgerRepository::activate(&config).unwrap();
        let second_assignment = second
            .cohort_assignment
            .assign(crate::ledger::cohort::CohortAssignmentRequest::new(
                raw_root_uuid,
                &second.identity.config_generation_id,
                "pool-a",
                "candidate-a",
                0.125,
                0.25,
            ))
            .unwrap();
        assert_eq!(second.identity.project_uuid, first_identity.project_uuid);
        assert_eq!(second.identity.project_id, first_identity.project_id);
        assert_eq!(
            second.identity.cohort_generation_id,
            first_identity.cohort_generation_id
        );
        assert_eq!(
            second.identity.config_generation_id,
            first_identity.config_generation_id
        );
        assert_eq!(second.identity.pools, first_identity.pools);
        assert_ne!(
            second.identity.process_instance_id,
            first_identity.process_instance_id
        );
        assert_eq!(second_assignment, first_assignment);
        assert_eq!(
            scalar_i64(
                &second.repository.connection,
                "SELECT count(*) FROM process_instances"
            ),
            2
        );
        assert_eq!(
            scalar_i64(
                &second.repository.connection,
                "SELECT count(*) FROM schema_migrations"
            ),
            CURRENT_SCHEMA_VERSION
        );
        assert_eq!(
            scalar_i64(
                &second.repository.connection,
                "SELECT count(*) FROM sqlite_schema
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ),
            84
        );
        assert_eq!(
            scalar_i64(
                &second.repository.connection,
                "SELECT count(*) FROM sqlite_schema
                 WHERE type = 'index' AND name NOT LIKE 'sqlite_autoindex_%'"
            ),
            109
        );
        assert_eq!(
            scalar_i64(
                &second.repository.connection,
                "SELECT count(*) FROM sqlite_schema
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                   AND sql NOT LIKE '%STRICT'"
            ),
            0
        );
        assert_eq!(
            pragma_i64(&second.repository.connection, "application_id").unwrap(),
            i64::from(LEDGER_APPLICATION_ID)
        );
        assert_eq!(
            pragma_i64(&second.repository.connection, "user_version").unwrap(),
            CURRENT_SCHEMA_VERSION
        );
        let canonical_config_json: String = second
            .repository
            .connection
            .query_row(
                "SELECT canonical_config_json FROM config_generations LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let canonical_policy_json: String = second
            .repository
            .connection
            .query_row(
                "SELECT canonical_policy_json FROM policy_versions LIMIT 1",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let path_text = path.to_string_lossy();
        for forbidden in [
            path_text.as_ref(),
            "http://127.0.0.1:8080/v1",
            "ROUTER_TEST_EMBEDDING_SECRET",
            "tenant-private-a",
            "agent-private-a",
            "private-region-a",
        ] {
            assert!(!canonical_config_json.contains(forbidden), "{forbidden}");
            assert!(!canonical_policy_json.contains(forbidden), "{forbidden}");
        }
        assert!(canonical_config_json.contains("database_path_sha256"));
        assert!(canonical_config_json.contains("endpoint_identity_sha256"));
        assert!(canonical_policy_json.contains("tenant_selector"));
    }

    #[test]
    fn referenced_vector_registry_is_atomic_idempotent_and_config_frozen() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut first_config = config(&path, "vector-registry-project");
        first_config.pools[0].learning = Some(crate::config::LearningConfig::minimal("embedder-a"));
        first_config.embedders[0].api_key_env =
            Some("NEMO_RELAY_ROUTER_TASK7_MISSING_SECRET_DO_NOT_SET".to_string());

        let first = LedgerRepository::activate_at(&first_config, 1_000).unwrap();
        let first_identity = first.identity.clone();
        let first_vector = first_identity
            .pool("pool-a")
            .unwrap()
            .vector_space
            .clone()
            .unwrap();
        assert_eq!(first_vector.profile_id, "embedder-a");
        assert_eq!(
            scalar_i64(
                &first.repository.connection,
                "SELECT count(*) FROM embedder_profiles"
            ),
            1
        );
        assert_eq!(
            scalar_i64(
                &first.repository.connection,
                "SELECT count(*) FROM vector_spaces"
            ),
            1
        );
        assert_eq!(
            scalar_i64(
                &first.repository.connection,
                "SELECT count(*) FROM pool_vector_space_mappings"
            ),
            1
        );
        assert_eq!(
            scalar_i64(
                &first.repository.connection,
                "SELECT count(*) FROM vector_space_source_sequences"
            ),
            1
        );
        let canonical_profile: String = first
            .repository
            .connection
            .query_row(
                "SELECT canonical_profile_json FROM embedder_profiles",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(!canonical_profile.contains("NEMO_RELAY_ROUTER_TASK7_MISSING_SECRET_DO_NOT_SET"));
        drop(first);

        let second = LedgerRepository::activate_at(&first_config, 1_001).unwrap();
        assert_eq!(
            second
                .identity
                .pool("pool-a")
                .unwrap()
                .vector_space
                .as_ref(),
            Some(&first_vector)
        );
        assert_eq!(
            scalar_i64(
                &second.repository.connection,
                "SELECT count(*) FROM pool_vector_space_mappings"
            ),
            1
        );
        drop(second);

        let mut changed_config = first_config.clone();
        changed_config.retention_days = 31;
        let changed = LedgerRepository::activate_at(&changed_config, 1_002).unwrap();
        assert_ne!(
            changed.identity.config_generation_id,
            first_identity.config_generation_id
        );
        assert_eq!(
            changed.identity.pool("pool-a").unwrap().policy_version_id,
            first_identity.pool("pool-a").unwrap().policy_version_id
        );
        assert_eq!(
            changed
                .identity
                .pool("pool-a")
                .unwrap()
                .vector_space
                .as_ref(),
            Some(&first_vector)
        );
        assert_eq!(
            scalar_i64(
                &changed.repository.connection,
                "SELECT count(*) FROM pool_vector_space_mappings"
            ),
            2
        );
        assert_eq!(
            scalar_i64(
                &changed.repository.connection,
                "SELECT count(*) FROM vector_spaces"
            ),
            1
        );

        let key = vector_registry::FrozenMappingKey::new(
            changed.identity.project_uuid,
            changed.identity.config_generation_id.clone(),
            "pool-a",
            changed
                .identity
                .pool("pool-a")
                .unwrap()
                .policy_version_id
                .clone(),
        )
        .unwrap();
        let resolved =
            vector_registry::resolve_frozen_mapping(&changed.repository.connection, &key)
                .unwrap()
                .unwrap();
        assert_eq!(
            resolved.mapping.vector_space_id,
            first_vector.vector_space_id
        );
    }

    #[test]
    fn cohort_assignment_child_process_helper() {
        let Some(database_path) = std::env::var_os(COHORT_CHILD_DATABASE_ENV) else {
            return;
        };
        let config = config(Path::new(&database_path), "cohort-child-project");
        let activated = LedgerRepository::activate(&config).unwrap();
        let assignment = activated
            .cohort_assignment
            .assign(crate::ledger::cohort::CohortAssignmentRequest::new(
                Uuid::parse_str("00112233-4455-6677-8899-aabbccddeeff").unwrap(),
                &activated.identity.config_generation_id,
                "pool-a",
                "candidate-a",
                0.125,
                0.25,
            ))
            .unwrap();
        let safe_assignment = serde_json::to_string(&assignment).unwrap();
        println!(
            "{COHORT_CHILD_RESULT_PREFIX}{}|{safe_assignment}",
            activated.identity.process_instance_id
        );
    }

    #[test]
    fn child_processes_share_same_database_cohort_assignment_without_sharing_salt() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let executable = std::env::current_exe().unwrap();
        let run_child = || {
            let output = Command::new(&executable)
                .args([
                    "--exact",
                    "ledger::repository::tests::cohort_assignment_child_process_helper",
                    "--nocapture",
                ])
                .env(COHORT_CHILD_DATABASE_ENV, &path)
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "cohort child failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let stdout = String::from_utf8(output.stdout).unwrap();
            let result = stdout
                .lines()
                .find_map(|line| line.strip_prefix(COHORT_CHILD_RESULT_PREFIX))
                .expect("cohort child did not emit its safe result");
            let (process_instance_id, assignment) = result
                .split_once('|')
                .expect("cohort child result is malformed");
            (
                Uuid::parse_str(process_instance_id).unwrap(),
                assignment.to_owned(),
            )
        };

        let first = run_child();
        let second = run_child();
        assert_ne!(first.0, second.0);
        assert_eq!(first.1, second.1);
        assert!(!first.1.contains("00112233-4455-6677-8899-aabbccddeeff"));
        assert!(!first.1.contains("00112233445566778899aabbccddeeff"));
    }

    #[test]
    fn wrong_project_rolls_back_pending_v2_migration_without_domain_writes() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let original = config(&path, "canonical-v1-project");
        let prepared = PreparedLedgerMaterial::from_config(&original).unwrap();
        let mut connection = open_secure_connection(&path).unwrap();
        configure_connection(&connection).unwrap();
        enforce_sidecar_permissions(&path).unwrap();
        let sqlite_version = verified_sqlite_version(&connection).unwrap();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Exclusive)
            .unwrap();
        apply_migration(&transaction, MIGRATIONS[0], 1_000, &sqlite_version).unwrap();
        let (project_uuid, _, project_created) =
            initialize_project(&transaction, &original, 1_000, true).unwrap();
        let new_pool_ids = initialize_config_and_policies(
            &transaction,
            project_uuid,
            &prepared,
            1_000,
            project_created,
        )
        .unwrap();
        load_or_initialize_cohort(&transaction, project_uuid, 1_000, project_created).unwrap();
        for pool_id in prepared.policies.keys() {
            load_or_initialize_learning(
                &transaction,
                project_uuid,
                pool_id,
                1_000,
                new_pool_ids.contains(pool_id),
            )
            .unwrap();
        }
        transaction.commit().unwrap();
        assert_eq!(pragma_i64(&connection, "user_version").unwrap(), 1);
        let counts_before = all_table_counts(&connection);
        let schema_before = schema_snapshot(&connection);
        drop(connection);

        let mismatched = config(&path, "wrong-project");
        let error = match LedgerRepository::activate_at(&mismatched, 31_000) {
            Ok(_) => panic!("wrong project must be rejected"),
            Err(error) => error,
        };
        assert_eq!(error.class(), LedgerErrorClass::ProjectIdMismatch);

        let connection = Connection::open(&path).unwrap();
        assert_eq!(pragma_i64(&connection, "user_version").unwrap(), 1);
        assert_eq!(
            scalar_i64(&connection, "SELECT count(*) FROM schema_migrations"),
            1
        );
        assert_eq!(all_table_counts(&connection), counts_before);
        assert_eq!(schema_snapshot(&connection), schema_before);
    }

    #[test]
    fn nonempty_v3_vector_placeholder_refuses_upgrade_without_changes() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = config(&path, "nonempty-v3-vector-project");
        let prepared = PreparedLedgerMaterial::from_config(&config).unwrap();
        let mut connection = open_secure_connection(&path).unwrap();
        configure_connection(&connection).unwrap();
        enforce_sidecar_permissions(&path).unwrap();
        let sqlite_version = verified_sqlite_version(&connection).unwrap();
        let transaction = connection
            .transaction_with_behavior(TransactionBehavior::Exclusive)
            .unwrap();
        for migration in &MIGRATIONS[..3] {
            apply_migration(&transaction, *migration, 1_000, &sqlite_version).unwrap();
        }
        let (project_uuid, _, project_created) =
            initialize_project(&transaction, &config, 1_000, true).unwrap();
        let new_pool_ids = initialize_config_and_policies(
            &transaction,
            project_uuid,
            &prepared,
            1_000,
            project_created,
        )
        .unwrap();
        load_or_initialize_cohort(&transaction, project_uuid, 1_000, project_created).unwrap();
        for pool_id in prepared.policies.keys() {
            load_or_initialize_learning(
                &transaction,
                project_uuid,
                pool_id,
                1_000,
                new_pool_ids.contains(pool_id),
            )
            .unwrap();
        }
        transaction
            .execute(
                "INSERT INTO vector_spaces (
                    vector_space_id, project_uuid, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, 1, ?3)",
                params!["a".repeat(64), project_uuid.to_string(), "b".repeat(64)],
            )
            .unwrap();
        transaction.commit().unwrap();
        assert_eq!(pragma_i64(&connection, "user_version").unwrap(), 3);
        let counts_before = all_table_counts(&connection);
        let schema_before = schema_snapshot(&connection);
        drop(connection);

        let error = LedgerRepository::activate_at(&config, 2_000)
            .err()
            .expect("nonempty Spec 05 placeholders must refuse migration 0004");
        assert_eq!(error.class(), LedgerErrorClass::MigrationFailed);

        let connection = Connection::open(&path).unwrap();
        assert_eq!(pragma_i64(&connection, "user_version").unwrap(), 3);
        assert_eq!(
            scalar_i64(&connection, "SELECT count(*) FROM schema_migrations"),
            3
        );
        assert_eq!(all_table_counts(&connection), counts_before);
        assert_eq!(schema_snapshot(&connection), schema_before);
    }

    #[test]
    fn omitted_project_id_resolves_to_the_stable_project_uuid() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut config = config(&path, "discarded-project-id");
        config.project_id = None;

        let first = LedgerRepository::activate(&config).unwrap();
        assert_eq!(
            first.identity.project_id,
            first.identity.project_uuid.to_string()
        );
        let project_uuid = first.identity.project_uuid;
        drop(first);
        let second = LedgerRepository::activate(&config).unwrap();
        assert_eq!(second.identity.project_uuid, project_uuid);
        assert_eq!(second.identity.project_id, project_uuid.to_string());
    }

    #[test]
    fn failed_initialization_rolls_back_the_initial_migration() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut invalid = config(&path, "temporary");
        invalid.project_id = Some("x".repeat(129));

        assert_eq!(
            LedgerRepository::activate(&invalid).err().unwrap().class(),
            LedgerErrorClass::IdentityInvariant
        );
        let connection = Connection::open(&path).unwrap();
        assert_eq!(pragma_i64(&connection, "application_id").unwrap(), 0);
        assert_eq!(pragma_i64(&connection, "user_version").unwrap(), 0);
        assert_eq!(
            scalar_i64(
                &connection,
                "SELECT count(*) FROM sqlite_schema
                 WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            ),
            0
        );
    }

    #[test]
    fn config_changes_preserve_learning_and_semantically_unchanged_policy() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut config = config(&path, "config-project");
        let first = LedgerRepository::activate(&config).unwrap();
        let first_identity = first.identity.clone();
        drop(first);

        config.retention_days += 1;
        let second = LedgerRepository::activate(&config).unwrap();
        let second_identity = second.identity.clone();
        assert_ne!(
            second_identity.config_generation_id,
            first_identity.config_generation_id
        );
        assert_eq!(
            second_identity.pools["pool-a"],
            first_identity.pools["pool-a"]
        );
        drop(second);

        config.pools[0].sampling_probability = 0.5;
        let third = LedgerRepository::activate(&config).unwrap();
        assert_eq!(
            third.identity.pools["pool-a"].learning_generation_id,
            first_identity.pools["pool-a"].learning_generation_id
        );
        assert_ne!(
            third.identity.pools["pool-a"].policy_version_id,
            first_identity.pools["pool-a"].policy_version_id
        );
    }

    #[test]
    fn resets_are_append_only_isolated_and_atomic() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated =
            LedgerRepository::activate(&two_pool_config(&path, "reset-project")).unwrap();
        let original_a = activated.identity.pools["pool-a"].learning_generation_id;
        let original_b = activated.identity.pools["pool-b"].learning_generation_id;

        let reset_a = activated
            .repository
            .reset_pool("pool-a", "operator", "targeted reset")
            .unwrap();
        assert_ne!(reset_a, original_a);
        assert_eq!(
            scalar_i64(
                &activated.repository.connection,
                "SELECT count(*) FROM learning_generations WHERE pool_id = 'pool-a'"
            ),
            2
        );
        assert_eq!(
            scalar_i64(
                &activated.repository.connection,
                "SELECT count(*) FROM learning_generations WHERE pool_id = 'pool-b'"
            ),
            1
        );

        let before_unknown = scalar_i64(
            &activated.repository.connection,
            "SELECT count(*) FROM learning_generations",
        );
        assert_eq!(
            activated
                .repository
                .reset_pool("missing", "operator", "must rollback")
                .unwrap_err()
                .class(),
            LedgerErrorClass::IdentityInvariant
        );
        assert_eq!(
            scalar_i64(
                &activated.repository.connection,
                "SELECT count(*) FROM learning_generations"
            ),
            before_unknown
        );

        activated
            .repository
            .connection
            .execute_batch(
                "CREATE TEMP TRIGGER fail_second_pool_reset
                 BEFORE INSERT ON learning_generations
                 WHEN NEW.pool_id = 'pool-b'
                 BEGIN
                     SELECT RAISE(ABORT, 'injected reset failure');
                 END;",
            )
            .unwrap();
        let before_injected_failure = scalar_i64(
            &activated.repository.connection,
            "SELECT count(*) FROM learning_generations",
        );
        assert!(
            activated
                .repository
                .reset_project("operator", "injected atomic rollback")
                .is_err()
        );
        assert_eq!(
            scalar_i64(
                &activated.repository.connection,
                "SELECT count(*) FROM learning_generations"
            ),
            before_injected_failure
        );
        activated
            .repository
            .connection
            .execute_batch("DROP TRIGGER temp.fail_second_pool_reset;")
            .unwrap();

        let project_reset = activated
            .repository
            .reset_project("operator", "project reset")
            .unwrap();
        assert_eq!(project_reset.len(), 2);
        assert_ne!(project_reset["pool-a"], reset_a);
        assert_ne!(project_reset["pool-b"], original_b);
        assert_eq!(
            scalar_i64(
                &activated.repository.connection,
                "SELECT count(*) FROM learning_generations"
            ),
            5
        );

        let before_invalid = scalar_i64(
            &activated.repository.connection,
            "SELECT count(*) FROM learning_generations",
        );
        assert!(
            activated
                .repository
                .reset_project("", "invalid actor")
                .is_err()
        );
        assert_eq!(
            scalar_i64(
                &activated.repository.connection,
                "SELECT count(*) FROM learning_generations"
            ),
            before_invalid
        );
    }

    #[test]
    fn configured_project_mismatch_is_refused_without_leaking_the_path() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        drop(LedgerRepository::activate(&config(&path, "project-a")).unwrap());

        let error = LedgerRepository::activate(&config(&path, "project-b"))
            .err()
            .expect("project mismatch should fail");
        assert_eq!(error.class(), LedgerErrorClass::ProjectIdMismatch);
        assert!(!error.to_string().contains(path.to_string_lossy().as_ref()));
        assert!(!error.to_string().to_ascii_lowercase().contains("select"));
    }

    #[test]
    fn missing_generation_pointers_are_refused_instead_of_rotating_identity() {
        let cohort_temp = tempdir().unwrap();
        let cohort_path = database_path(&cohort_temp);
        let cohort_config = config(&cohort_path, "orphaned-cohort-project");
        let first = LedgerRepository::activate(&cohort_config).unwrap();
        let cohort_generation_id = first.identity.cohort_generation_id;
        drop(first);
        let connection = Connection::open(&cohort_path).unwrap();
        connection
            .execute("DELETE FROM cohort_generation_state_events", [])
            .unwrap();
        drop(connection);
        assert_eq!(
            LedgerRepository::activate(&cohort_config)
                .err()
                .unwrap()
                .class(),
            LedgerErrorClass::IdentityInvariant
        );
        let connection = Connection::open(&cohort_path).unwrap();
        let stored_generation_id: String = connection
            .query_row(
                "SELECT cohort_generation_id FROM cohort_generations",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stored_generation_id, cohort_generation_id.to_string());
        assert_eq!(
            scalar_i64(&connection, "SELECT count(*) FROM cohort_generations"),
            1
        );

        let learning_temp = tempdir().unwrap();
        let learning_path = database_path(&learning_temp);
        let learning_config = config(&learning_path, "orphaned-learning-project");
        let mut first = LedgerRepository::activate(&learning_config).unwrap();
        let learning_generation_id = first
            .repository
            .reset_pool("pool-a", "operator", "pointer-loss-test")
            .unwrap();
        drop(first);
        let connection = Connection::open(&learning_path).unwrap();
        connection
            .execute(
                "DELETE FROM learning_generation_state_events
                 WHERE learning_generation_id = ?1",
                params![learning_generation_id.to_string()],
            )
            .unwrap();
        drop(connection);
        assert_eq!(
            LedgerRepository::activate(&learning_config)
                .err()
                .unwrap()
                .class(),
            LedgerErrorClass::IdentityInvariant
        );
        let connection = Connection::open(&learning_path).unwrap();
        let stored_generation_id: String = connection
            .query_row(
                "SELECT learning_generation_id FROM learning_generations
                 WHERE learning_generation_id = ?1",
                params![learning_generation_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(stored_generation_id, learning_generation_id.to_string());
        assert_eq!(
            scalar_i64(&connection, "SELECT count(*) FROM learning_generations"),
            2
        );
        assert_eq!(
            scalar_i64(
                &connection,
                "SELECT count(*) FROM learning_generation_state_events"
            ),
            1
        );
    }

    #[test]
    fn deleted_identity_rows_cannot_masquerade_as_first_activation() {
        let cohort_temp = tempdir().unwrap();
        let cohort_path = database_path(&cohort_temp);
        let cohort_config = config(&cohort_path, "deleted-cohort-project");
        drop(LedgerRepository::activate(&cohort_config).unwrap());
        let connection = Connection::open(&cohort_path).unwrap();
        connection
            .execute("DELETE FROM cohort_generation_state_events", [])
            .unwrap();
        connection
            .execute("DELETE FROM cohort_generations", [])
            .unwrap();
        drop(connection);
        assert_eq!(
            LedgerRepository::activate(&cohort_config)
                .err()
                .unwrap()
                .class(),
            LedgerErrorClass::IdentityInvariant
        );

        let learning_temp = tempdir().unwrap();
        let learning_path = database_path(&learning_temp);
        let learning_config = config(&learning_path, "deleted-learning-project");
        drop(LedgerRepository::activate(&learning_config).unwrap());
        let connection = Connection::open(&learning_path).unwrap();
        connection
            .execute("DELETE FROM learning_generation_state_events", [])
            .unwrap();
        connection
            .execute("DELETE FROM learning_generations", [])
            .unwrap();
        drop(connection);
        assert_eq!(
            LedgerRepository::activate(&learning_config)
                .err()
                .unwrap()
                .class(),
            LedgerErrorClass::IdentityInvariant
        );

        let project_temp = tempdir().unwrap();
        let project_path = database_path(&project_temp);
        let project_config = config(&project_path, "deleted-project");
        drop(LedgerRepository::activate(&project_config).unwrap());
        let connection = Connection::open(&project_path).unwrap();
        delete_all_domain_rows(&connection);
        drop(connection);
        assert_eq!(
            LedgerRepository::activate(&project_config)
                .err()
                .unwrap()
                .class(),
            LedgerErrorClass::IdentityInvariant
        );
        let connection = Connection::open(&project_path).unwrap();
        assert_eq!(
            scalar_i64(&connection, "SELECT count(*) FROM schema_migrations"),
            CURRENT_SCHEMA_VERSION
        );
        assert_eq!(
            scalar_i64(&connection, "SELECT count(*) FROM project_metadata"),
            0
        );
    }

    #[test]
    fn cohort_salt_reads_revalidate_fingerprint_and_payload_hash() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = config(&path, "salt-tamper-project");
        let activated = LedgerRepository::activate(&config).unwrap();
        let cohort_generation_id = activated.identity.cohort_generation_id;
        activated
            .repository
            .connection
            .execute(
                "UPDATE cohort_generations SET cohort_salt = ?1
                 WHERE cohort_generation_id = ?2",
                params![
                    vec![0_u8; COHORT_SALT_BYTES],
                    cohort_generation_id.to_string()
                ],
            )
            .unwrap();

        assert_eq!(
            activated
                .repository
                .cohort_salt(cohort_generation_id)
                .err()
                .unwrap()
                .class(),
            LedgerErrorClass::IdentityInvariant
        );
        drop(activated);
        assert_eq!(
            LedgerRepository::activate(&config).err().unwrap().class(),
            LedgerErrorClass::IdentityInvariant
        );
    }

    #[test]
    fn stable_code_columns_accept_namespaced_runtime_classes() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let activated = LedgerRepository::activate(&config(&path, "stable-code-project")).unwrap();
        let classes = [
            LedgerErrorClass::InvalidFilesystem,
            LedgerErrorClass::InvalidPermissions,
            LedgerErrorClass::UnsupportedFilesystemSecurity,
            LedgerErrorClass::OpenFailed,
            LedgerErrorClass::Busy,
            LedgerErrorClass::CorruptDatabase,
            LedgerErrorClass::PragmaMismatch,
            LedgerErrorClass::SqliteVersionMismatch,
            LedgerErrorClass::FutureSchema,
            LedgerErrorClass::MigrationChecksumMismatch,
            LedgerErrorClass::InvalidMigrationHistory,
            LedgerErrorClass::MigrationFailed,
            LedgerErrorClass::ProjectIdMismatch,
            LedgerErrorClass::IdentityInvariant,
            LedgerErrorClass::RandomnessUnavailable,
            LedgerErrorClass::CanonicalizationFailed,
            LedgerErrorClass::DatabaseOperationFailed,
        ];
        for stable_class in classes
            .iter()
            .map(|class| class.code())
            .chain(["router.ineligible.contract_schema_draft"])
        {
            activated
                .repository
                .connection
                .execute(
                    "INSERT INTO health_events (
                        health_event_id, project_uuid, stable_class, severity,
                        created_at_unix_ms, canonical_payload_hash
                     ) VALUES (?1, ?2, ?3, 'degraded', 0, ?4)",
                    params![
                        Uuid::now_v7().to_string(),
                        activated.identity.project_uuid.to_string(),
                        stable_class,
                        "0".repeat(64),
                    ],
                )
                .unwrap();
        }
    }

    #[test]
    fn terminal_schema_keeps_started_attempts_immutable_and_excludes_operational_evaluations() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        drop(LedgerRepository::activate(&config(&path, "terminal-schema-project")).unwrap());
        let connection = Connection::open(&path).unwrap();
        connection
            .execute_batch("PRAGMA foreign_keys = OFF;")
            .unwrap();

        let columns = |table: &str| {
            let mut statement = connection
                .prepare(&format!("PRAGMA table_info(\"{table}\")"))
                .unwrap();
            statement
                .query_map([], |row| row.get::<_, String>(1))
                .unwrap()
                .collect::<Result<BTreeSet<_>, _>>()
                .unwrap()
        };
        let attempt_columns = columns("judge_attempts");
        let state_columns = columns("judge_attempt_state_events");
        for terminal_column in [
            "parse_result",
            "safe_output_json",
            "raw_output_sha256",
            "raw_output_bytes",
            "stable_error_class",
        ] {
            assert!(!attempt_columns.contains(terminal_column));
            assert!(state_columns.contains(terminal_column));
        }

        let judge_attempt_id = Uuid::now_v7().to_string();
        let process_instance_id = Uuid::now_v7().to_string();
        assert!(
            connection
                .execute(
                    "INSERT INTO judge_attempt_state_events (
                        judge_attempt_state_event_id, judge_attempt_id,
                        process_instance_id, state, parse_result, safe_output_json,
                        raw_output_sha256, raw_output_bytes,
                        created_at_unix_ms, canonical_payload_hash
                     ) VALUES (?1, ?2, ?3, 'started', 'valid', '{}', ?4, 2, 0, ?5)",
                    params![
                        Uuid::now_v7().to_string(),
                        judge_attempt_id,
                        process_instance_id,
                        "1".repeat(64),
                        "2".repeat(64),
                    ],
                )
                .is_err()
        );
        connection
            .execute(
                "INSERT INTO judge_attempt_state_events (
                    judge_attempt_state_event_id, judge_attempt_id,
                    process_instance_id, state, parse_result, safe_output_json,
                    raw_output_sha256, raw_output_bytes,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, 'valid', 'valid', '{}', ?4, 2, 0, ?5)",
                params![
                    Uuid::now_v7().to_string(),
                    judge_attempt_id,
                    process_instance_id,
                    "1".repeat(64),
                    "2".repeat(64),
                ],
            )
            .unwrap();

        let transport_attempt_id = Uuid::now_v7().to_string();
        let insert_transport = |stable_error_class: Option<&str>| {
            connection.execute(
                "INSERT INTO judge_attempt_state_events (
                    judge_attempt_state_event_id, judge_attempt_id,
                    process_instance_id, state, parse_result, stable_error_class,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, 'transport_failure', 'operational', ?4, 0, ?5)",
                params![
                    Uuid::now_v7().to_string(),
                    transport_attempt_id,
                    process_instance_id,
                    stable_error_class,
                    "4".repeat(64),
                ],
            )
        };
        assert!(insert_transport(None).is_err());
        insert_transport(Some("router.judge.transport_failure")).unwrap();

        assert!(
            connection
                .execute(
                    "INSERT INTO judge_attempt_state_events (
                        judge_attempt_state_event_id, judge_attempt_id,
                        process_instance_id, state, parse_result, safe_output_json,
                        stable_error_class, created_at_unix_ms, canonical_payload_hash
                     ) VALUES (?1, ?2, ?3, 'valid', 'valid', '{}',
                               'router.judge.unexpected', 0, ?4)",
                    params![
                        Uuid::now_v7().to_string(),
                        Uuid::now_v7().to_string(),
                        process_instance_id,
                        "5".repeat(64),
                    ],
                )
                .is_err()
        );

        let shadow_attempt_id = Uuid::now_v7().to_string();
        let insert_operational = |evaluation_id: Option<&str>| {
            connection.execute(
                "INSERT INTO shadow_results (
                    shadow_result_id, shadow_attempt_id, terminal_class,
                    operational_failure_class, evaluation_id, canonicalizable,
                    partition_inputs_json, noncanonicalizable_reason,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, 'operational_failure', 'router.provider.timeout',
                           ?3, 0, '{}', 'operational_failure', 0, ?4)",
                params![
                    Uuid::now_v7().to_string(),
                    shadow_attempt_id,
                    evaluation_id,
                    "3".repeat(64),
                ],
            )
        };
        assert!(insert_operational(Some(&Uuid::now_v7().to_string())).is_err());
        insert_operational(None).unwrap();
    }

    fn dynamic_schema_fixture(
        space_byte: char,
        generation: i64,
        state: &str,
    ) -> (
        DynamicSchemaManifest,
        Vec<SchemaObject>,
        String,
        String,
        String,
    ) {
        let vector_space_id = VectorSpaceId::new(space_byte.to_string().repeat(64)).unwrap();
        let authority = Vec0SchemaAuthority::new(
            Vec0RootName::new(
                vector_space_id.clone(),
                VectorIndexGeneration::new(generation).unwrap(),
            ),
            VectorDimensions::new(3).unwrap(),
        );
        let root = authority.root().as_str().to_string();
        let actual = authority
            .objects()
            .iter()
            .map(|object| SchemaObject {
                object_type: object.object_type().to_string(),
                name: object.name().to_string(),
                table_name: object.table_name().to_string(),
                sql: Some(object.sql().to_string()),
            })
            .collect::<Vec<_>>();
        let objects_json = authority.manifest_json().to_string();
        let objects_sha256 = authority.manifest_sha256().to_string();
        let manifest = parse_dynamic_schema_manifest(
            vector_space_id.as_str().to_string(),
            generation,
            state,
            &root,
            3,
            &objects_json,
            &objects_sha256,
        )
        .unwrap();
        (manifest, actual, root, objects_json, objects_sha256)
    }

    fn assert_corrupt<T>(result: Result<T, LedgerError>) {
        assert_eq!(
            result.err().unwrap().class(),
            LedgerErrorClass::CorruptDatabase
        );
    }

    #[test]
    fn dynamic_schema_manifest_loading_and_missing_lifecycles_are_exact() {
        let (building, _, _, _, _) = dynamic_schema_fixture('a', 1, "building");
        let (active, actual, root, objects_json, objects_sha256) =
            dynamic_schema_fixture('b', 2, "active");
        let (unavailable, _, _, _, _) = dynamic_schema_fixture('c', 3, "unavailable");
        let (corrupt, _, _, _, _) = dynamic_schema_fixture('d', 4, "corrupt");
        let (retired, _, _, _, _) = dynamic_schema_fixture('e', 5, "retired");
        let (dropped, _, _, _, _) = dynamic_schema_fixture('f', 6, "dropped");

        let report = verify_dynamic_schema(
            Vec::new(),
            vec![
                building.clone(),
                active.clone(),
                unavailable.clone(),
                corrupt.clone(),
                retired.clone(),
                dropped,
            ],
            &BTreeSet::new(),
        )
        .unwrap();
        assert_eq!(report.missing_building, vec![building.generation.clone()]);
        assert_eq!(
            report.missing_active,
            vec![
                active.generation.clone(),
                unavailable.generation.clone(),
                corrupt.generation.clone()
            ]
        );
        assert_eq!(report.missing_retired, vec![retired.generation.clone()]);
        assert_eq!(
            verify_dynamic_schema(actual, vec![active], &BTreeSet::new()).unwrap(),
            SchemaVerificationReport::default()
        );

        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE vector_index_manifest (
                    vector_space_id TEXT NOT NULL,
                    generation INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    root_table_name TEXT NOT NULL,
                    dimensions INTEGER NOT NULL,
                    expected_schema_objects_json TEXT NOT NULL,
                    expected_schema_objects_sha256 TEXT NOT NULL
                 ) STRICT;",
            )
            .unwrap();
        connection
            .execute(
                "INSERT INTO vector_index_manifest VALUES (?1, 2, 'active', ?2, 3, ?3, ?4)",
                params!["b".repeat(64), root, objects_json, objects_sha256],
            )
            .unwrap();
        let loaded = load_dynamic_schema_manifests(&connection).unwrap();
        assert_eq!(loaded.len(), 1);
        assert_eq!(loaded[0].generation.generation, 2);
        assert_eq!(loaded[0].lifecycle, ManifestLifecycle::Active);
    }

    #[test]
    fn dynamic_schema_rejects_malformed_or_nonbijective_authority() {
        let (active, actual, root, objects_json, objects_sha256) =
            dynamic_schema_fixture('a', 1, "active");
        let space = "a".repeat(64);

        assert_corrupt(parse_dynamic_schema_manifest(
            space.clone(),
            1,
            "active",
            "wrong-root",
            3,
            &objects_json,
            &objects_sha256,
        ));
        for dimensions in [0, i64::from(crate::config::EMBEDDING_DIMENSIONS_MAX) + 1] {
            assert_corrupt(parse_dynamic_schema_manifest(
                space.clone(),
                1,
                "active",
                &root,
                dimensions,
                &objects_json,
                &objects_sha256,
            ));
        }
        assert_corrupt(parse_dynamic_schema_manifest(
            space.clone(),
            1,
            "unknown",
            &root,
            3,
            &objects_json,
            &objects_sha256,
        ));
        assert_corrupt(parse_dynamic_schema_manifest(
            space.clone(),
            1,
            "active",
            &root,
            3,
            &format!(" {objects_json}"),
            &objects_sha256,
        ));
        assert_corrupt(parse_dynamic_schema_manifest(
            space.clone(),
            1,
            "active",
            &root,
            3,
            &objects_json,
            &"0".repeat(64),
        ));

        let mut malformed: Json = serde_json::from_str(&objects_json).unwrap();
        malformed[0]["name"] = json!("unowned");
        let malformed_json = canonical_json(&malformed).unwrap();
        let malformed_hash = canonical_sha256(&malformed).unwrap();
        assert_corrupt(parse_dynamic_schema_manifest(
            space,
            1,
            "active",
            &root,
            3,
            &malformed_json,
            &malformed_hash,
        ));

        let mut mismatched = actual.clone();
        mismatched[0].sql = Some("CREATE TABLE wrong (value INTEGER)".to_string());
        assert_corrupt(verify_dynamic_schema(
            mismatched,
            vec![active.clone()],
            &BTreeSet::new(),
        ));

        let mut extra = actual.clone();
        extra.push(SchemaObject {
            object_type: "table".to_string(),
            name: "unexpected".to_string(),
            table_name: "unexpected".to_string(),
            sql: Some("CREATE TABLE unexpected (value INTEGER)".to_string()),
        });
        assert_corrupt(verify_dynamic_schema(
            extra,
            vec![active.clone()],
            &BTreeSet::new(),
        ));
        assert_corrupt(verify_dynamic_schema(
            actual.clone(),
            vec![active.clone(), active.clone()],
            &BTreeSet::new(),
        ));
        assert_corrupt(verify_dynamic_schema(
            actual[..1].to_vec(),
            vec![active.clone()],
            &BTreeSet::new(),
        ));
        let mut static_keys = BTreeSet::new();
        static_keys.insert((
            active.objects[0].object_type.clone(),
            active.objects[0].name.clone(),
        ));
        assert_corrupt(verify_dynamic_schema(
            actual.clone(),
            vec![active],
            &static_keys,
        ));

        let (dropped, _, _, _, _) = dynamic_schema_fixture('a', 1, "dropped");
        assert_corrupt(verify_dynamic_schema(
            actual,
            vec![dropped],
            &BTreeSet::new(),
        ));
    }

    #[test]
    fn migration_constraint_errors_remain_migration_failures() {
        let connection = Connection::open_in_memory().unwrap();
        connection
            .execute_batch(
                "CREATE TABLE migration_guard (
                    all_placeholders_empty INTEGER NOT NULL
                        CHECK (all_placeholders_empty = 1)
                 ) STRICT;",
            )
            .unwrap();
        let error = connection
            .execute(
                "INSERT INTO migration_guard (all_placeholders_empty) VALUES (0)",
                [],
            )
            .unwrap_err();
        assert_eq!(
            map_migration_error(&error).class(),
            LedgerErrorClass::MigrationFailed
        );
    }

    #[test]
    fn checksum_gap_future_and_extra_schema_are_refused() {
        let checksum_temp = tempdir().unwrap();
        let checksum_path = database_path(&checksum_temp);
        let checksum_config = config(&checksum_path, "checksum-project");
        drop(LedgerRepository::activate(&checksum_config).unwrap());
        let connection = Connection::open(&checksum_path).unwrap();
        connection
            .execute(
                "UPDATE schema_migrations SET checksum_sha256 = ?1 WHERE version = 1",
                params!["0".repeat(64)],
            )
            .unwrap();
        set_delete_journal_mode(&connection);
        drop(connection);
        assert_eq!(
            LedgerRepository::activate(&checksum_config)
                .err()
                .unwrap()
                .class(),
            LedgerErrorClass::MigrationChecksumMismatch
        );
        assert_refusal_left_delete_journal_untouched(&checksum_path);

        let gap_temp = tempdir().unwrap();
        let gap_path = database_path(&gap_temp);
        let gap_config = config(&gap_path, "gap-project");
        drop(LedgerRepository::activate(&gap_config).unwrap());
        let connection = Connection::open(&gap_path).unwrap();
        connection
            .execute("DELETE FROM schema_migrations", [])
            .unwrap();
        drop(connection);
        assert_eq!(
            LedgerRepository::activate(&gap_config)
                .err()
                .unwrap()
                .class(),
            LedgerErrorClass::InvalidMigrationHistory
        );

        let future_temp = tempdir().unwrap();
        let future_path = database_path(&future_temp);
        let future_config = config(&future_path, "future-project");
        drop(LedgerRepository::activate(&future_config).unwrap());
        let connection = Connection::open(&future_path).unwrap();
        connection
            .pragma_update(None, "user_version", CURRENT_SCHEMA_VERSION + 1)
            .unwrap();
        set_delete_journal_mode(&connection);
        drop(connection);
        assert_eq!(
            LedgerRepository::activate(&future_config)
                .err()
                .unwrap()
                .class(),
            LedgerErrorClass::FutureSchema
        );
        assert_refusal_left_delete_journal_untouched(&future_path);

        let extra_temp = tempdir().unwrap();
        let extra_path = database_path(&extra_temp);
        let extra_config = config(&extra_path, "extra-project");
        drop(LedgerRepository::activate(&extra_config).unwrap());
        let connection = Connection::open(&extra_path).unwrap();
        connection
            .execute("CREATE TABLE unexpected (id INTEGER)", [])
            .unwrap();
        drop(connection);
        assert_eq!(
            LedgerRepository::activate(&extra_config)
                .err()
                .unwrap()
                .class(),
            LedgerErrorClass::CorruptDatabase
        );

        let missing_index_temp = tempdir().unwrap();
        let missing_index_path = database_path(&missing_index_temp);
        let missing_index_config = config(&missing_index_path, "missing-index-project");
        drop(LedgerRepository::activate(&missing_index_config).unwrap());
        let connection = Connection::open(&missing_index_path).unwrap();
        connection
            .execute("DROP INDEX uq_anchor_terminal_state", [])
            .unwrap();
        set_delete_journal_mode(&connection);
        drop(connection);
        assert_eq!(
            LedgerRepository::activate(&missing_index_config)
                .err()
                .unwrap()
                .class(),
            LedgerErrorClass::CorruptDatabase
        );
        assert_refusal_left_delete_journal_untouched(&missing_index_path);
    }

    #[test]
    fn corrupt_and_foreign_sqlite_files_are_refused() {
        let corrupt_temp = tempdir().unwrap();
        let corrupt_path = database_path(&corrupt_temp);
        fs::create_dir_all(corrupt_path.parent().unwrap()).unwrap();
        fs::write(&corrupt_path, b"not a sqlite database").unwrap();
        assert_eq!(
            LedgerRepository::activate(&config(&corrupt_path, "corrupt-project"))
                .err()
                .unwrap()
                .class(),
            LedgerErrorClass::CorruptDatabase
        );

        let foreign_temp = tempdir().unwrap();
        let foreign_path = database_path(&foreign_temp);
        fs::create_dir_all(foreign_path.parent().unwrap()).unwrap();
        let connection = Connection::open(&foreign_path).unwrap();
        connection
            .execute("CREATE TABLE foreign_data (id INTEGER)", [])
            .unwrap();
        drop(connection);
        assert_eq!(
            LedgerRepository::activate(&config(&foreign_path, "foreign-project"))
                .err()
                .unwrap()
                .class(),
            LedgerErrorClass::CorruptDatabase
        );
    }

    #[test]
    fn concurrent_openers_share_stable_facts_and_get_distinct_processes() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = config(&path, "concurrent-project");
        let barrier = Arc::new(Barrier::new(2));
        let handles = (0..2)
            .map(|_| {
                let config = config.clone();
                let barrier = barrier.clone();
                thread::spawn(move || {
                    barrier.wait();
                    let mut busy_retries = 0;
                    loop {
                        match LedgerRepository::activate(&config) {
                            Ok(activated) => break Ok((activated.identity, busy_retries)),
                            Err(error)
                                if error.class() == LedgerErrorClass::Busy && busy_retries < 2 =>
                            {
                                busy_retries += 1;
                            }
                            Err(error) => break Err(error),
                        }
                    }
                })
            })
            .collect::<Vec<_>>();
        let opened = handles
            .into_iter()
            .map(|handle| handle.join().unwrap().unwrap())
            .collect::<Vec<_>>();
        assert!(opened.iter().any(|(_, busy_retries)| *busy_retries == 0));
        let identities = opened
            .into_iter()
            .map(|(identity, _)| identity)
            .collect::<Vec<_>>();

        assert!(
            identities
                .iter()
                .all(|identity| identity.project_uuid == identities[0].project_uuid)
        );
        assert!(
            identities
                .iter()
                .all(|identity| identity.cohort_generation_id
                    == identities[0].cohort_generation_id)
        );
        assert!(
            identities
                .iter()
                .all(|identity| identity.pools == identities[0].pools)
        );
        assert_eq!(
            identities
                .iter()
                .map(|identity| identity.process_instance_id)
                .collect::<BTreeSet<_>>()
                .len(),
            2
        );
    }

    #[test]
    fn activation_samples_cutoff_after_waiting_for_startup_lock() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = config(&path, "activation-cutoff-project");
        let first = LedgerRepository::activate(&config).unwrap();
        let project_uuid = first.identity.project_uuid;
        let config_generation_id = first.identity.config_generation_id.clone();
        drop(first);

        let (migration_ready_tx, migration_ready_rx) = mpsc::sync_channel(0);
        let release_activation = Arc::new(Barrier::new(2));
        let thread_release = release_activation.clone();
        let (cutoff_tx, cutoff_rx) = mpsc::channel();
        let opener_config = config.clone();
        let opener = thread::spawn(move || {
            LedgerRepository::activate_with_clock_and_hooks(
                &opener_config,
                None,
                move || {
                    migration_ready_tx.send(()).unwrap();
                    thread_release.wait();
                },
                move |cutoff| cutoff_tx.send(cutoff).unwrap(),
            )
        });

        migration_ready_rx.recv().unwrap();
        let mut blocker = Connection::open(&path).unwrap();
        configure_connection(&blocker).unwrap();
        let sqlite_version = verified_sqlite_version(&blocker).unwrap();
        let blocker_transaction = blocker
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        release_activation.wait();

        assert!(matches!(
            cutoff_rx.recv_timeout(Duration::from_millis(100)),
            Err(mpsc::RecvTimeoutError::Timeout)
        ));
        let competing_start = now_unix_millis().unwrap();
        initialize_process(
            &blocker_transaction,
            project_uuid,
            &config_generation_id,
            &sqlite_version,
            competing_start,
        )
        .unwrap();
        blocker_transaction.commit().unwrap();

        let activation_cutoff = cutoff_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(activation_cutoff >= competing_start);
        opener.join().unwrap().unwrap();
    }

    #[test]
    fn busy_writer_is_refused_after_the_fixed_timeout_without_partial_identity() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let config = config(&path, "busy-project");
        let activated = LedgerRepository::activate(&config).unwrap();
        let process_count = scalar_i64(
            &activated.repository.connection,
            "SELECT count(*) FROM process_instances",
        );
        drop(activated);

        let blocker = Connection::open(&path).unwrap();
        blocker.execute_batch("BEGIN IMMEDIATE").unwrap();
        let started = std::time::Instant::now();
        let error = LedgerRepository::activate(&config).err().unwrap();
        let elapsed = started.elapsed();
        assert_eq!(error.class(), LedgerErrorClass::Busy);
        assert!(elapsed >= Duration::from_secs(4), "elapsed: {elapsed:?}");
        assert!(elapsed < Duration::from_secs(8), "elapsed: {elapsed:?}");
        blocker.execute_batch("ROLLBACK").unwrap();
        drop(blocker);

        let reopened = LedgerRepository::activate(&config).unwrap();
        assert_eq!(
            scalar_i64(
                &reopened.repository.connection,
                "SELECT count(*) FROM process_instances"
            ),
            process_count + 1
        );
    }

    #[test]
    fn values_are_bound_even_when_identifiers_contain_sql_syntax() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut config = config(&path, "project'); DROP TABLE project_metadata; --");
        config.pools[0].id = "pool'); DROP TABLE policy_versions; --".into();
        let activated = LedgerRepository::activate(&config).unwrap();
        assert_eq!(
            scalar_i64(
                &activated.repository.connection,
                "SELECT count(*) FROM project_metadata"
            ),
            1
        );
        assert_eq!(
            scalar_i64(
                &activated.repository.connection,
                "SELECT count(*) FROM policy_versions"
            ),
            1
        );
    }
}
