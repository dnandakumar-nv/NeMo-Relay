// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Durable Active experiments, tranche boundaries, and outcome-look authority.

mod neighborhood;

pub(crate) use neighborhood::*;

use std::collections::BTreeMap;

use chrono::Utc;
use rusqlite::{
    Connection, OptionalExtension, Transaction, TransactionBehavior, named_params, params,
};
use serde_json::{Value as Json, json};
use uuid::{Uuid, Variant};

use super::{LedgerRepository, TransactionStartGuard};
use crate::active_math::{
    ActiveLookAuditV1, ActiveLookMemberV1, ActiveLookPolicyV1, ActiveOutcomeArmV1,
    ActiveOutcomeLabelV1, LookProducedStateV1, active_look_input_identity_v1,
    active_math_algorithm_identity_v1,
};
use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::config::{OutcomeConfig, RouterConfig, RouterMode};
use crate::ledger::model::{LedgerError, LedgerErrorClass, LedgerRuntimeIdentity};

pub(crate) const ACTIVE_LOOK_LEASE_MAX_MILLIS: u64 = 30_000;
pub(crate) const ACTIVE_LOOK_WORK_PAGE_MAX: usize = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveLookWorkCandidate {
    pub(crate) active_experiment_id: Uuid,
}

pub(crate) fn select_active_look_work(
    connection: &Connection,
    project_uuid: Uuid,
    current_config_generation_id: &str,
    limit: usize,
) -> Result<Vec<ActiveLookWorkCandidate>, LedgerError> {
    if project_uuid.get_version_num() != 7
        || project_uuid.get_variant() != Variant::RFC4122
        || !is_hash(current_config_generation_id)
        || limit == 0
        || limit > ACTIVE_LOOK_WORK_PAGE_MAX
    {
        return Err(invariant());
    }
    connection
        .prepare(
            "SELECT experiment.active_experiment_id
             FROM active_experiments AS experiment
             JOIN active_experiment_state_events AS experiment_state
               ON experiment_state.active_experiment_id = experiment.active_experiment_id
             WHERE experiment.project_uuid = ?1
               AND experiment.config_generation_id = ?2
               AND experiment_state.event_seq = (
                    SELECT max(latest.event_seq)
                    FROM active_experiment_state_events AS latest
                    WHERE latest.active_experiment_id = experiment.active_experiment_id
               )
               AND experiment_state.state IN ('collecting', 'cap_draining')
               AND experiment.learning_generation_id = (
                    SELECT learning.learning_generation_id
                    FROM learning_generation_state_events AS learning
                    WHERE learning.project_uuid = experiment.project_uuid
                      AND learning.pool_id = experiment.pool_id
                    ORDER BY learning.event_seq DESC LIMIT 1
               )
               AND experiment.cohort_generation_id = (
                    SELECT cohort.cohort_generation_id
                    FROM cohort_generation_state_events AS cohort
                    WHERE cohort.project_uuid = experiment.project_uuid
                    ORDER BY cohort.event_seq DESC LIMIT 1
               )
               AND NOT EXISTS (
                    SELECT 1 FROM active_retirement_markers AS retirement
                    WHERE retirement.active_experiment_id = experiment.active_experiment_id
               )
               AND EXISTS (
                    SELECT 1
                    FROM active_experiment_tranches AS tranche
                    JOIN active_tranche_state_events AS tranche_state
                      ON tranche_state.active_experiment_id = tranche.active_experiment_id
                     AND tranche_state.tranche_ordinal = tranche.tranche_ordinal
                    WHERE tranche.active_experiment_id = experiment.active_experiment_id
                      AND tranche_state.event_seq = (
                           SELECT max(latest.event_seq)
                           FROM active_tranche_state_events AS latest
                           WHERE latest.active_experiment_id = tranche.active_experiment_id
                             AND latest.tranche_ordinal = tranche.tranche_ordinal
                      )
                      AND tranche_state.state IN ('drained', 'evaluating')
                      AND NOT EXISTS (
                           SELECT 1 FROM active_outcome_looks AS look
                           WHERE look.active_experiment_id = tranche.active_experiment_id
                             AND look.boundary_tranche_ordinal = tranche.tranche_ordinal
                      )
                      AND NOT EXISTS (
                           SELECT 1 FROM active_look_failures AS failure
                           WHERE failure.active_experiment_id = tranche.active_experiment_id
                             AND failure.boundary_tranche_ordinal = tranche.tranche_ordinal
                      )
               )
             ORDER BY experiment.created_at_unix_ms, experiment.active_experiment_id
             LIMIT ?3",
        )
        .map_err(database_error)?
        .query_map(
            params![
                project_uuid.to_string(),
                current_config_generation_id,
                i64::try_from(limit).map_err(|_| invariant())?,
            ],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .map(|row| {
            row.map_err(database_error)
                .and_then(|value| parse_uuid_v7(&value))
                .map(|active_experiment_id| ActiveLookWorkCandidate {
                    active_experiment_id,
                })
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveExperimentCreate {
    pub(crate) active_experiment_id: Uuid,
    pub(crate) initial_experiment_state_event_id: Uuid,
    pub(crate) initial_tranche_state_event_id: Uuid,
    pub(crate) initial_authorization_state_event_id: Uuid,
    pub(crate) pool_id: String,
    pub(crate) candidate_id: String,
    pub(crate) partition_hash: String,
    pub(crate) config_generation_id: String,
    pub(crate) policy_version_id: String,
    pub(crate) outcome_policy_hash: String,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) cohort_generation_id: Uuid,
    pub(crate) vector_space_id: String,
    pub(crate) control_generation: u64,
    pub(crate) outcome_evaluation_batch_size: u32,
    pub(crate) max_canary_roots: u32,
    pub(crate) max_looks: u32,
    pub(crate) min_treatment_roots: u32,
    pub(crate) min_control_roots: u32,
    pub(crate) min_treatment_effective_weight_bits: u64,
    pub(crate) min_control_effective_weight_bits: u64,
    pub(crate) noninferiority_margin_bits: u64,
    pub(crate) noninferiority_probability_bits: u64,
    pub(crate) rollback_probability_bits: u64,
    pub(crate) promotion_lower_bound_bits: u64,
    pub(crate) retention_lower_bound_bits: u64,
    pub(crate) holdout_probability_bits: u64,
    pub(crate) active_canary_fraction_bits: u64,
    pub(crate) actual_outcome_half_life_seconds: u32,
    pub(crate) anchor_shadow_half_life_seconds: u32,
    pub(crate) authorization_ttl_seconds: u32,
}

impl ActiveExperimentCreate {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_config(
        config: &RouterConfig,
        identity: &LedgerRuntimeIdentity,
        pool_id: &str,
        candidate_id: impl Into<String>,
        partition_hash: impl Into<String>,
        outcome_policy_hash: impl Into<String>,
        control_generation: u64,
    ) -> Result<Self, LedgerError> {
        if config.mode != RouterMode::Active {
            return Err(invariant());
        }
        let pool = config
            .pools
            .iter()
            .find(|pool| pool.id == pool_id)
            .ok_or_else(invariant)?;
        let active = pool
            .learning
            .as_ref()
            .and_then(|learning| learning.complete_active_policy())
            .ok_or_else(invariant)?;
        let outcome = serde_json::from_value::<OutcomeConfig>(Json::Object(
            pool.outcome.clone().into_iter().collect(),
        ))
        .map_err(|_| invariant())?;
        let runtime = identity.pools.get(pool_id).ok_or_else(invariant)?;
        let vector_space_id = runtime
            .vector_space
            .as_ref()
            .ok_or_else(invariant)?
            .vector_space_id
            .as_str()
            .to_string();
        let max_looks = outcome
            .max_canary_roots
            .checked_div(outcome.outcome_evaluation_batch_size)
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(invariant)?;
        let create = Self {
            active_experiment_id: Uuid::now_v7(),
            initial_experiment_state_event_id: Uuid::now_v7(),
            initial_tranche_state_event_id: Uuid::now_v7(),
            initial_authorization_state_event_id: Uuid::now_v7(),
            pool_id: pool_id.to_string(),
            candidate_id: candidate_id.into(),
            partition_hash: partition_hash.into(),
            config_generation_id: identity.config_generation_id.clone(),
            policy_version_id: runtime.policy_version_id.clone(),
            outcome_policy_hash: outcome_policy_hash.into(),
            learning_generation_id: runtime.learning_generation_id,
            cohort_generation_id: identity.cohort_generation_id,
            vector_space_id,
            control_generation,
            outcome_evaluation_batch_size: u32::try_from(outcome.outcome_evaluation_batch_size)
                .map_err(|_| invariant())?,
            max_canary_roots: u32::try_from(outcome.max_canary_roots).map_err(|_| invariant())?,
            max_looks,
            min_treatment_roots: u32::try_from(outcome.min_treatment_roots)
                .map_err(|_| invariant())?,
            min_control_roots: u32::try_from(outcome.min_control_roots).map_err(|_| invariant())?,
            min_treatment_effective_weight_bits: outcome.min_treatment_effective_weight.to_bits(),
            min_control_effective_weight_bits: outcome.min_control_effective_weight.to_bits(),
            noninferiority_margin_bits: outcome.noninferiority_margin.to_bits(),
            noninferiority_probability_bits: outcome.noninferiority_probability.to_bits(),
            rollback_probability_bits: outcome.rollback_probability.to_bits(),
            promotion_lower_bound_bits: active.recommend.promotion_lower_bound.to_bits(),
            retention_lower_bound_bits: active.retention_lower_bound.to_bits(),
            holdout_probability_bits: active.holdout_probability.to_bits(),
            active_canary_fraction_bits: active.active_canary_fraction.to_bits(),
            actual_outcome_half_life_seconds: u32::try_from(
                outcome.actual_outcome_half_life_seconds,
            )
            .map_err(|_| invariant())?,
            anchor_shadow_half_life_seconds: u32::try_from(outcome.anchor_shadow_half_life_seconds)
                .map_err(|_| invariant())?,
            authorization_ttl_seconds: u32::try_from(outcome.authorization_ttl_seconds)
                .map_err(|_| invariant())?,
        };
        validate_experiment_create(&create)?;
        Ok(create)
    }

    pub(crate) fn look_policy(&self) -> Result<ActiveLookPolicyV1, LedgerError> {
        ActiveLookPolicyV1::new(
            self.actual_outcome_half_life_seconds,
            self.min_treatment_roots,
            self.min_control_roots,
            f64::from_bits(self.min_treatment_effective_weight_bits),
            f64::from_bits(self.min_control_effective_weight_bits),
            f64::from_bits(self.noninferiority_margin_bits),
            f64::from_bits(self.noninferiority_probability_bits),
            f64::from_bits(self.rollback_probability_bits),
            self.max_looks,
        )
        .map_err(|_| invariant())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveExperimentReceipt {
    pub(crate) active_experiment_id: Uuid,
    pub(crate) initial_experiment_state_event_id: Uuid,
    pub(crate) initial_tranche_state_event_id: Uuid,
    pub(crate) initial_authorization_state_event_id: Uuid,
    pub(crate) created_at_unix_ms: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveExperimentCreateAck {
    Applied(ActiveExperimentReceipt),
    AlreadyApplied(ActiveExperimentReceipt),
    Conflict,
    AuthorityChanged,
    TransactionNotStarted,
}

impl LedgerRepository {
    pub(crate) fn create_active_experiment(
        &mut self,
        create: &ActiveExperimentCreate,
    ) -> Result<ActiveExperimentCreateAck, LedgerError> {
        self.create_active_experiment_with_start_check(create, || Some(()))
    }

    pub(crate) fn create_active_experiment_with_start_check<G: TransactionStartGuard>(
        &mut self,
        create: &ActiveExperimentCreate,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveExperimentCreateAck, LedgerError> {
        validate_experiment_create(create)?;
        let database_path = self.database_path.clone();
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveExperimentCreateAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(ActiveExperimentCreateAck::TransactionNotStarted);
            }
            Err(error) => {
                return Err(super::map_sqlite_error(
                    &error,
                    LedgerErrorClass::DatabaseOperationFailed,
                ));
            }
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ActiveExperimentCreateAck::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = create_active_experiment_in_transaction(
            &transaction,
            self.project_uuid,
            self.process_instance_id,
            create,
            now_unix_ms()?,
        )?;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }
}

pub(crate) fn create_active_experiment_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    create: &ActiveExperimentCreate,
    created_at_unix_ms: i64,
) -> Result<ActiveExperimentCreateAck, LedgerError> {
    validate_experiment_create(create)?;
    if let Some(existing) = load_existing_experiment(transaction, project_uuid, create)? {
        return Ok(existing);
    }
    if !experiment_authority_matches(transaction, project_uuid, process_instance_id, create)? {
        return Ok(ActiveExperimentCreateAck::AuthorityChanged);
    }
    insert_experiment(
        transaction,
        project_uuid,
        process_instance_id,
        create,
        created_at_unix_ms,
    )?;
    Ok(ActiveExperimentCreateAck::Applied(
        ActiveExperimentReceipt {
            active_experiment_id: create.active_experiment_id,
            initial_experiment_state_event_id: create.initial_experiment_state_event_id,
            initial_tranche_state_event_id: create.initial_tranche_state_event_id,
            initial_authorization_state_event_id: create.initial_authorization_state_event_id,
            created_at_unix_ms: u64::try_from(created_at_unix_ms).map_err(|_| corrupt())?,
        },
    ))
}

fn validate_experiment_create(create: &ActiveExperimentCreate) -> Result<(), LedgerError> {
    let ids = [
        create.active_experiment_id,
        create.initial_experiment_state_event_id,
        create.initial_tranche_state_event_id,
        create.initial_authorization_state_event_id,
        create.learning_generation_id,
        create.cohort_generation_id,
    ];
    let holdout = finite_probability(create.holdout_probability_bits, false)?;
    let canary = finite_probability(create.active_canary_fraction_bits, false)?;
    let promotion = finite_probability(create.promotion_lower_bound_bits, true)?;
    let retention = finite_probability(create.retention_lower_bound_bits, true)?;
    if ids.into_iter().any(|id| !is_uuid_v7(id))
        || !valid_text(&create.pool_id, 128)
        || !valid_text(&create.candidate_id, 128)
        || !is_hash(&create.partition_hash)
        || !is_hash(&create.config_generation_id)
        || !is_hash(&create.policy_version_id)
        || !is_hash(&create.outcome_policy_hash)
        || !is_hash(&create.vector_space_id)
        || !(64..=2_048).contains(&create.outcome_evaluation_batch_size)
        || create.max_canary_roots < create.outcome_evaluation_batch_size
        || !create
            .max_canary_roots
            .is_multiple_of(create.outcome_evaluation_batch_size)
        || create.max_looks != create.max_canary_roots / create.outcome_evaluation_batch_size
        || !(1..=256).contains(&create.max_looks)
        || create.control_generation > i64::MAX as u64
        || holdout > 0.25
        || holdout + canary >= 1.0
        || retention >= promotion
        || create.authorization_ttl_seconds > create.actual_outcome_half_life_seconds
    {
        return Err(invariant());
    }
    create.look_policy()?;
    Ok(())
}

fn finite_probability(bits: u64, allow_zero: bool) -> Result<f64, LedgerError> {
    let value = f64::from_bits(bits);
    if !value.is_finite()
        || value.is_sign_negative()
        || value > 1.0
        || (!allow_zero && value == 0.0)
        || (value == 0.0 && bits != 0)
    {
        return Err(invariant());
    }
    Ok(value)
}

#[derive(Debug)]
struct StoredExperiment {
    active_experiment_id: String,
    process_instance_id: String,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
    values: BTreeMap<&'static str, String>,
}

fn load_existing_experiment(
    connection: &Connection,
    project_uuid: Uuid,
    create: &ActiveExperimentCreate,
) -> Result<Option<ActiveExperimentCreateAck>, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT active_experiment_id, process_instance_id, created_at_unix_ms,
                    canonical_payload_hash, policy_version_id, vector_space_id,
                    control_generation, outcome_evaluation_batch_size,
                    max_canary_roots, max_looks, min_treatment_roots,
                    min_control_roots, min_treatment_effective_weight_bits,
                    min_control_effective_weight_bits, noninferiority_margin_bits,
                    noninferiority_probability_bits, rollback_probability_bits,
                    promotion_lower_bound_bits, retention_lower_bound_bits,
                    holdout_probability_bits, active_canary_fraction_bits,
                    actual_outcome_half_life_seconds,
                    anchor_shadow_half_life_seconds, authorization_ttl_seconds
             FROM active_experiments
             WHERE pool_id = ?1 AND candidate_id = ?2 AND partition_hash = ?3
               AND learning_generation_id = ?4 AND config_generation_id = ?5
               AND cohort_generation_id = ?6 AND outcome_policy_hash = ?7",
            params![
                create.pool_id,
                create.candidate_id,
                create.partition_hash,
                create.learning_generation_id.to_string(),
                create.config_generation_id,
                create.cohort_generation_id.to_string(),
                create.outcome_policy_hash,
            ],
            |row| {
                let mut values = BTreeMap::new();
                for (name, index) in [
                    ("policy_version_id", 4),
                    ("vector_space_id", 5),
                    ("control_generation", 6),
                    ("outcome_evaluation_batch_size", 7),
                    ("max_canary_roots", 8),
                    ("max_looks", 9),
                    ("min_treatment_roots", 10),
                    ("min_control_roots", 11),
                    ("min_treatment_effective_weight_bits", 12),
                    ("min_control_effective_weight_bits", 13),
                    ("noninferiority_margin_bits", 14),
                    ("noninferiority_probability_bits", 15),
                    ("rollback_probability_bits", 16),
                    ("promotion_lower_bound_bits", 17),
                    ("retention_lower_bound_bits", 18),
                    ("holdout_probability_bits", 19),
                    ("active_canary_fraction_bits", 20),
                    ("actual_outcome_half_life_seconds", 21),
                    ("anchor_shadow_half_life_seconds", 22),
                    ("authorization_ttl_seconds", 23),
                ] {
                    let value = match index {
                        4 | 5 => row.get::<_, String>(index)?,
                        _ => row.get::<_, i64>(index)?.to_string(),
                    };
                    values.insert(name, value);
                }
                Ok(StoredExperiment {
                    active_experiment_id: row.get(0)?,
                    process_instance_id: row.get(1)?,
                    created_at_unix_ms: row.get(2)?,
                    canonical_payload_hash: row.get(3)?,
                    values,
                })
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(stored) = stored else {
        let id_collision: bool = connection
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM active_experiments WHERE active_experiment_id = ?1
                 )",
                [create.active_experiment_id.to_string()],
                |row| row.get(0),
            )
            .map_err(database_error)?;
        return Ok(id_collision.then_some(ActiveExperimentCreateAck::Conflict));
    };
    if !stored_experiment_matches(&stored, create) {
        return Ok(Some(ActiveExperimentCreateAck::Conflict));
    }
    let active_experiment_id = parse_uuid_v7(&stored.active_experiment_id)?;
    let process_instance_id = parse_uuid_v7(&stored.process_instance_id)?;
    if experiment_payload_hash(
        project_uuid,
        process_instance_id,
        active_experiment_id,
        create,
        stored.created_at_unix_ms,
    )? != stored.canonical_payload_hash
    {
        return Err(corrupt());
    }
    let initial_experiment_state_event_id = load_initial_event_id(
        connection,
        "active_experiment_state_events",
        "active_experiment_state_event_id",
        active_experiment_id,
    )?;
    let initial_tranche_state_event_id = connection
        .query_row(
            "SELECT active_tranche_state_event_id
             FROM active_tranche_state_events
             WHERE active_experiment_id = ?1 AND tranche_ordinal = 1
             ORDER BY event_seq LIMIT 1",
            [active_experiment_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)
        .and_then(|value| parse_uuid_v7(&value))?;
    let initial_authorization_state_event_id = load_initial_event_id(
        connection,
        "active_authorization_state_events",
        "active_authorization_state_event_id",
        active_experiment_id,
    )?;
    Ok(Some(ActiveExperimentCreateAck::AlreadyApplied(
        ActiveExperimentReceipt {
            active_experiment_id,
            initial_experiment_state_event_id,
            initial_tranche_state_event_id,
            initial_authorization_state_event_id,
            created_at_unix_ms: u64::try_from(stored.created_at_unix_ms).map_err(|_| corrupt())?,
        },
    )))
}

fn stored_experiment_matches(stored: &StoredExperiment, create: &ActiveExperimentCreate) -> bool {
    let expected = BTreeMap::from([
        ("policy_version_id", create.policy_version_id.clone()),
        ("vector_space_id", create.vector_space_id.clone()),
        ("control_generation", create.control_generation.to_string()),
        (
            "outcome_evaluation_batch_size",
            create.outcome_evaluation_batch_size.to_string(),
        ),
        ("max_canary_roots", create.max_canary_roots.to_string()),
        ("max_looks", create.max_looks.to_string()),
        (
            "min_treatment_roots",
            create.min_treatment_roots.to_string(),
        ),
        ("min_control_roots", create.min_control_roots.to_string()),
        (
            "min_treatment_effective_weight_bits",
            (create.min_treatment_effective_weight_bits as i64).to_string(),
        ),
        (
            "min_control_effective_weight_bits",
            (create.min_control_effective_weight_bits as i64).to_string(),
        ),
        (
            "noninferiority_margin_bits",
            (create.noninferiority_margin_bits as i64).to_string(),
        ),
        (
            "noninferiority_probability_bits",
            (create.noninferiority_probability_bits as i64).to_string(),
        ),
        (
            "rollback_probability_bits",
            (create.rollback_probability_bits as i64).to_string(),
        ),
        (
            "promotion_lower_bound_bits",
            (create.promotion_lower_bound_bits as i64).to_string(),
        ),
        (
            "retention_lower_bound_bits",
            (create.retention_lower_bound_bits as i64).to_string(),
        ),
        (
            "holdout_probability_bits",
            (create.holdout_probability_bits as i64).to_string(),
        ),
        (
            "active_canary_fraction_bits",
            (create.active_canary_fraction_bits as i64).to_string(),
        ),
        (
            "actual_outcome_half_life_seconds",
            create.actual_outcome_half_life_seconds.to_string(),
        ),
        (
            "anchor_shadow_half_life_seconds",
            create.anchor_shadow_half_life_seconds.to_string(),
        ),
        (
            "authorization_ttl_seconds",
            create.authorization_ttl_seconds.to_string(),
        ),
    ]);
    stored.values == expected
}

fn load_initial_event_id(
    connection: &Connection,
    table: &str,
    id_column: &str,
    active_experiment_id: Uuid,
) -> Result<Uuid, LedgerError> {
    let sql = format!(
        "SELECT {id_column} FROM {table}
         WHERE active_experiment_id = ?1 ORDER BY event_seq LIMIT 1"
    );
    connection
        .query_row(&sql, [active_experiment_id.to_string()], |row| {
            row.get::<_, String>(0)
        })
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)
        .and_then(|value| parse_uuid_v7(&value))
}

fn experiment_authority_matches(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    create: &ActiveExperimentCreate,
) -> Result<bool, LedgerError> {
    if !super::process::originating_process_is_live(connection, project_uuid, process_instance_id)?
        || !super::active::all_live_processes_support_active_v6(connection, project_uuid)?
    {
        return Ok(false);
    }
    let current_config = connection
        .query_row(
            "SELECT config_generation_id FROM config_generation_state_events
             WHERE project_uuid = ?1 ORDER BY config_epoch DESC LIMIT 1",
            [project_uuid.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let current_learning = connection
        .query_row(
            "SELECT learning_generation_id FROM learning_generation_state_events
             WHERE project_uuid = ?1 AND pool_id = ?2
             ORDER BY event_seq DESC LIMIT 1",
            params![project_uuid.to_string(), create.pool_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let current_cohort = connection
        .query_row(
            "SELECT cohort_generation_id FROM cohort_generation_state_events
             WHERE project_uuid = ?1 ORDER BY event_seq DESC LIMIT 1",
            [project_uuid.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let current_control = connection
        .query_row(
            "SELECT max(control_generation) FROM controls WHERE project_uuid = ?1",
            [project_uuid.to_string()],
            |row| row.get::<_, Option<i64>>(0),
        )
        .map_err(database_error)?;
    let learning_generation_id = create.learning_generation_id.to_string();
    let cohort_generation_id = create.cohort_generation_id.to_string();
    if current_config.as_deref() != Some(create.config_generation_id.as_str())
        || current_learning.as_deref() != Some(learning_generation_id.as_str())
        || current_cohort.as_deref() != Some(cohort_generation_id.as_str())
        || current_control.and_then(|value| u64::try_from(value).ok())
            != Some(create.control_generation)
    {
        return Ok(false);
    }
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1
                FROM outcome_policy_versions AS outcome
                JOIN pool_vector_space_mappings AS mapping
                  ON mapping.project_uuid = outcome.project_uuid
                 AND mapping.config_generation_id = outcome.config_generation_id
                 AND mapping.pool_id = outcome.pool_id
                 AND mapping.policy_version_id = outcome.policy_version_id
                WHERE outcome.project_uuid = ?1 AND outcome.pool_id = ?2
                  AND outcome.outcome_policy_hash = ?3
                  AND outcome.config_generation_id = ?4
                  AND outcome.policy_version_id = ?5
                  AND mapping.vector_space_id = ?6
             )",
            params![
                project_uuid.to_string(),
                create.pool_id,
                create.outcome_policy_hash,
                create.config_generation_id,
                create.policy_version_id,
                create.vector_space_id,
            ],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn insert_experiment(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    create: &ActiveExperimentCreate,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let active_math_algorithm_id_sha256 = active_math_algorithm_identity_v1();
    let experiment_hash = experiment_payload_hash(
        project_uuid,
        process_instance_id,
        create.active_experiment_id,
        create,
        created_at_unix_ms,
    )?;
    let treatment_weight = f64::from_bits(create.min_treatment_effective_weight_bits);
    let control_weight = f64::from_bits(create.min_control_effective_weight_bits);
    let margin = f64::from_bits(create.noninferiority_margin_bits);
    let noninferiority = f64::from_bits(create.noninferiority_probability_bits);
    let rollback = f64::from_bits(create.rollback_probability_bits);
    let promotion = f64::from_bits(create.promotion_lower_bound_bits);
    let retention = f64::from_bits(create.retention_lower_bound_bits);
    let holdout = f64::from_bits(create.holdout_probability_bits);
    let canary = f64::from_bits(create.active_canary_fraction_bits);
    execute_one(
        transaction,
        "INSERT INTO active_experiments (
            active_experiment_id, experiment_shape_version, project_uuid,
            pool_id, candidate_id, partition_hash, config_generation_id,
            policy_version_id, outcome_policy_hash, learning_generation_id,
            cohort_generation_id, vector_space_id, control_generation,
            assignment_algorithm_id, active_math_algorithm_id_sha256,
            outcome_evaluation_batch_size, max_canary_roots, max_looks,
            min_treatment_roots, min_control_roots,
            min_treatment_effective_weight, min_treatment_effective_weight_bits,
            min_control_effective_weight, min_control_effective_weight_bits,
            noninferiority_margin, noninferiority_margin_bits,
            noninferiority_probability, noninferiority_probability_bits,
            rollback_probability, rollback_probability_bits,
            promotion_lower_bound, promotion_lower_bound_bits,
            retention_lower_bound, retention_lower_bound_bits,
            holdout_probability, holdout_probability_bits,
            active_canary_fraction, active_canary_fraction_bits,
            actual_outcome_half_life_seconds, anchor_shadow_half_life_seconds,
            authorization_ttl_seconds, process_instance_id,
            created_at_unix_ms, canonical_payload_hash
         ) VALUES (
            ?1, 1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12,
            'cohort_assignment_v1', ?13, ?14, ?15, ?16, ?17, ?18,
            ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26, ?27, ?28,
            ?29, ?30, ?31, ?32, ?33, ?34, ?35, ?36,
            ?37, ?38, ?39, ?40, ?41, ?42
         )",
        params![
            create.active_experiment_id.to_string(),
            project_uuid.to_string(),
            create.pool_id,
            create.candidate_id,
            create.partition_hash,
            create.config_generation_id,
            create.policy_version_id,
            create.outcome_policy_hash,
            create.learning_generation_id.to_string(),
            create.cohort_generation_id.to_string(),
            create.vector_space_id,
            i64::try_from(create.control_generation).map_err(|_| invariant())?,
            active_math_algorithm_id_sha256,
            i64::from(create.outcome_evaluation_batch_size),
            i64::from(create.max_canary_roots),
            i64::from(create.max_looks),
            i64::from(create.min_treatment_roots),
            i64::from(create.min_control_roots),
            treatment_weight,
            create.min_treatment_effective_weight_bits as i64,
            control_weight,
            create.min_control_effective_weight_bits as i64,
            margin,
            create.noninferiority_margin_bits as i64,
            noninferiority,
            create.noninferiority_probability_bits as i64,
            rollback,
            create.rollback_probability_bits as i64,
            promotion,
            create.promotion_lower_bound_bits as i64,
            retention,
            create.retention_lower_bound_bits as i64,
            holdout,
            create.holdout_probability_bits as i64,
            canary,
            create.active_canary_fraction_bits as i64,
            i64::from(create.actual_outcome_half_life_seconds),
            i64::from(create.anchor_shadow_half_life_seconds),
            i64::from(create.authorization_ttl_seconds),
            process_instance_id.to_string(),
            created_at_unix_ms,
            experiment_hash,
        ],
    )?;
    insert_experiment_state(
        transaction,
        create.initial_experiment_state_event_id,
        create.active_experiment_id,
        "collecting",
        None,
        process_instance_id,
        created_at_unix_ms,
    )?;
    insert_tranche(
        transaction,
        create.active_experiment_id,
        1,
        create.outcome_evaluation_batch_size,
        process_instance_id,
        created_at_unix_ms,
    )?;
    insert_tranche_state(
        transaction,
        create.initial_tranche_state_event_id,
        create.active_experiment_id,
        1,
        "open",
        0,
        0,
        0,
        process_instance_id,
        created_at_unix_ms,
    )?;
    insert_authorization_state(
        transaction,
        create.initial_authorization_state_event_id,
        create.active_experiment_id,
        None,
        None,
        None,
        "collecting",
        None,
        create.control_generation,
        process_instance_id,
        created_at_unix_ms,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn experiment_payload_hash(
    project_uuid: Uuid,
    process_instance_id: Uuid,
    active_experiment_id: Uuid,
    create: &ActiveExperimentCreate,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(json!({
        "shape": "active_experiment_v1",
        "active_experiment_id": active_experiment_id,
        "project_uuid": project_uuid,
        "pool_id": create.pool_id,
        "candidate_id": create.candidate_id,
        "partition_hash": create.partition_hash,
        "config_generation_id": create.config_generation_id,
        "policy_version_id": create.policy_version_id,
        "outcome_policy_hash": create.outcome_policy_hash,
        "learning_generation_id": create.learning_generation_id,
        "cohort_generation_id": create.cohort_generation_id,
        "vector_space_id": create.vector_space_id,
        "control_generation": create.control_generation.to_string(),
        "assignment_algorithm_id": "cohort_assignment_v1",
        "active_math_algorithm_id_sha256": active_math_algorithm_identity_v1(),
        "outcome_evaluation_batch_size": create.outcome_evaluation_batch_size.to_string(),
        "max_canary_roots": create.max_canary_roots.to_string(),
        "max_looks": create.max_looks.to_string(),
        "min_treatment_roots": create.min_treatment_roots.to_string(),
        "min_control_roots": create.min_control_roots.to_string(),
        "min_treatment_effective_weight_bits": format!("{:016x}", create.min_treatment_effective_weight_bits),
        "min_control_effective_weight_bits": format!("{:016x}", create.min_control_effective_weight_bits),
        "noninferiority_margin_bits": format!("{:016x}", create.noninferiority_margin_bits),
        "noninferiority_probability_bits": format!("{:016x}", create.noninferiority_probability_bits),
        "rollback_probability_bits": format!("{:016x}", create.rollback_probability_bits),
        "promotion_lower_bound_bits": format!("{:016x}", create.promotion_lower_bound_bits),
        "retention_lower_bound_bits": format!("{:016x}", create.retention_lower_bound_bits),
        "holdout_probability_bits": format!("{:016x}", create.holdout_probability_bits),
        "active_canary_fraction_bits": format!("{:016x}", create.active_canary_fraction_bits),
        "actual_outcome_half_life_seconds": create.actual_outcome_half_life_seconds.to_string(),
        "anchor_shadow_half_life_seconds": create.anchor_shadow_half_life_seconds.to_string(),
        "authorization_ttl_seconds": create.authorization_ttl_seconds.to_string(),
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": created_at_unix_ms.to_string(),
    }))
}

#[allow(clippy::too_many_arguments)]
fn insert_experiment_state(
    connection: &Connection,
    event_id: Uuid,
    experiment_id: Uuid,
    state: &str,
    terminal_reason: Option<&str>,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let payload_hash = hash_json(json!({
        "shape": "active_experiment_state_v1",
        "event_id": event_id,
        "active_experiment_id": experiment_id,
        "state": state,
        "terminal_reason": terminal_reason,
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": created_at_unix_ms.to_string(),
    }))?;
    execute_one(
        connection,
        "INSERT INTO active_experiment_state_events (
            active_experiment_state_event_id, active_experiment_id, state,
            terminal_reason, process_instance_id, created_at_unix_ms,
            canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            event_id.to_string(),
            experiment_id.to_string(),
            state,
            terminal_reason,
            process_instance_id.to_string(),
            created_at_unix_ms,
            payload_hash,
        ],
    )
}

fn insert_tranche(
    connection: &Connection,
    experiment_id: Uuid,
    tranche_ordinal: u64,
    nonholdout_limit: u32,
    process_instance_id: Uuid,
    opened_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let total_limit = u64::from(nonholdout_limit)
        .checked_mul(2)
        .ok_or_else(invariant)?;
    let payload_hash = hash_json(json!({
        "shape": "active_experiment_tranche_v1",
        "active_experiment_id": experiment_id,
        "tranche_ordinal": tranche_ordinal.to_string(),
        "nonholdout_limit": nonholdout_limit.to_string(),
        "total_limit": total_limit.to_string(),
        "opened_by_process_instance_id": process_instance_id,
        "opened_at_unix_ms": opened_at_unix_ms.to_string(),
    }))?;
    execute_one(
        connection,
        "INSERT INTO active_experiment_tranches (
            active_experiment_id, tranche_ordinal, nonholdout_limit,
            total_limit, opened_by_process_instance_id, opened_at_unix_ms,
            canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            experiment_id.to_string(),
            i64::try_from(tranche_ordinal).map_err(|_| invariant())?,
            i64::from(nonholdout_limit),
            i64::try_from(total_limit).map_err(|_| invariant())?,
            process_instance_id.to_string(),
            opened_at_unix_ms,
            payload_hash,
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn insert_tranche_state(
    connection: &Connection,
    event_id: Uuid,
    experiment_id: Uuid,
    tranche_ordinal: u64,
    state: &str,
    total_assignment_count: u64,
    nonholdout_assignment_count: u64,
    unresolved_assignment_count: u64,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let payload_hash = hash_json(json!({
        "shape": "active_tranche_state_v1",
        "event_id": event_id,
        "active_experiment_id": experiment_id,
        "tranche_ordinal": tranche_ordinal.to_string(),
        "state": state,
        "total_assignment_count": total_assignment_count.to_string(),
        "nonholdout_assignment_count": nonholdout_assignment_count.to_string(),
        "unresolved_assignment_count": unresolved_assignment_count.to_string(),
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": created_at_unix_ms.to_string(),
    }))?;
    execute_one(
        connection,
        "INSERT INTO active_tranche_state_events (
            active_tranche_state_event_id, active_experiment_id,
            tranche_ordinal, state, total_assignment_count,
            nonholdout_assignment_count, unresolved_assignment_count,
            process_instance_id, created_at_unix_ms, canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            event_id.to_string(),
            experiment_id.to_string(),
            i64::try_from(tranche_ordinal).map_err(|_| invariant())?,
            state,
            i64::try_from(total_assignment_count).map_err(|_| invariant())?,
            i64::try_from(nonholdout_assignment_count).map_err(|_| invariant())?,
            i64::try_from(unresolved_assignment_count).map_err(|_| invariant())?,
            process_instance_id.to_string(),
            created_at_unix_ms,
            payload_hash,
        ],
    )
}

#[allow(clippy::too_many_arguments)]
fn insert_authorization_state(
    connection: &Connection,
    event_id: Uuid,
    experiment_id: Uuid,
    predecessor_id: Option<Uuid>,
    outcome_look_id: Option<Uuid>,
    look_claim_id: Option<Uuid>,
    state: &str,
    valid_until_unix_ms: Option<u64>,
    control_generation: u64,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let transition_identity_hash = hash_json(json!({
        "shape": "active_authorization_transition_identity_v1",
        "active_experiment_id": experiment_id,
        "predecessor_id": predecessor_id,
        "active_outcome_look_id": outcome_look_id,
        "active_look_claim_id": look_claim_id,
        "state": state,
        "valid_until_unix_ms": valid_until_unix_ms.map(|value| value.to_string()),
        "control_generation": control_generation.to_string(),
    }))?;
    let payload_hash = hash_json(json!({
        "shape": "active_authorization_state_v1",
        "event_id": event_id,
        "active_experiment_id": experiment_id,
        "predecessor_id": predecessor_id,
        "active_outcome_look_id": outcome_look_id,
        "active_look_claim_id": look_claim_id,
        "state": state,
        "valid_until_unix_ms": valid_until_unix_ms.map(|value| value.to_string()),
        "control_generation": control_generation.to_string(),
        "transition_identity_hash": transition_identity_hash,
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": created_at_unix_ms.to_string(),
    }))?;
    execute_one(
        connection,
        "INSERT INTO active_authorization_state_events (
            active_authorization_state_event_id, active_experiment_id,
            predecessor_authorization_state_event_id, active_outcome_look_id,
            active_look_claim_id, state, valid_until_unix_ms,
            control_generation, transition_identity_hash,
            process_instance_id, created_at_unix_ms, canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
        params![
            event_id.to_string(),
            experiment_id.to_string(),
            predecessor_id.map(|value| value.to_string()),
            outcome_look_id.map(|value| value.to_string()),
            look_claim_id.map(|value| value.to_string()),
            state,
            valid_until_unix_ms
                .map(i64::try_from)
                .transpose()
                .map_err(|_| invariant())?,
            i64::try_from(control_generation).map_err(|_| invariant())?,
            transition_identity_hash,
            process_instance_id.to_string(),
            created_at_unix_ms,
            payload_hash,
        ],
    )
}

pub(super) fn advance_after_admission(
    transaction: &Transaction<'_>,
    process_instance_id: Uuid,
    active_experiment_id: Uuid,
    tranche_ordinal: u64,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let counts = tranche_counts(transaction, active_experiment_id, tranche_ordinal)?;
    let limits = transaction
        .query_row(
            "SELECT tranche.nonholdout_limit, tranche.total_limit,
                    experiment.max_canary_roots
             FROM active_experiment_tranches AS tranche
             JOIN active_experiments AS experiment
               ON experiment.active_experiment_id = tranche.active_experiment_id
             WHERE tranche.active_experiment_id = ?1 AND tranche.tranche_ordinal = ?2",
            params![
                active_experiment_id.to_string(),
                i64::try_from(tranche_ordinal).map_err(|_| invariant())?
            ],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .map_err(database_error)?;
    let experiment_counts = transaction
        .query_row(
            "SELECT count(*), count(cap_ordinal)
             FROM active_assignments WHERE active_experiment_id = ?1",
            [active_experiment_id.to_string()],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        )
        .map_err(database_error)?;
    let tranche_full = counts.0 >= limits.1 || counts.1 >= limits.0;
    let cap_full = experiment_counts.0 >= limits.2 * 2 || experiment_counts.1 >= limits.2;
    if tranche_full || cap_full {
        append_tranche_state_if_new(
            transaction,
            active_experiment_id,
            tranche_ordinal,
            "closed",
            counts,
            process_instance_id,
            created_at_unix_ms,
        )?;
    }
    if cap_full {
        append_experiment_state_if_new(
            transaction,
            active_experiment_id,
            "cap_draining",
            None,
            process_instance_id,
            created_at_unix_ms,
        )?;
    }
    Ok(())
}

pub(super) fn advance_after_terminalization(
    transaction: &Transaction<'_>,
    process_instance_id: Uuid,
    active_experiment_id: Uuid,
    tranche_ordinal: u64,
    created_at_unix_ms: i64,
) -> Result<bool, LedgerError> {
    let latest = latest_tranche_state(transaction, active_experiment_id, tranche_ordinal)?
        .ok_or_else(corrupt)?;
    if latest == "open" {
        return Ok(false);
    }
    let counts = tranche_counts(transaction, active_experiment_id, tranche_ordinal)?;
    if counts.2 != 0 {
        return Ok(false);
    }
    append_tranche_state_if_new(
        transaction,
        active_experiment_id,
        tranche_ordinal,
        "drained",
        counts,
        process_instance_id,
        created_at_unix_ms,
    )?;
    Ok(true)
}

fn tranche_counts(
    connection: &Connection,
    active_experiment_id: Uuid,
    tranche_ordinal: u64,
) -> Result<(i64, i64, i64), LedgerError> {
    connection
        .query_row(
            "SELECT count(*), count(assignment.cap_ordinal),
                    coalesce(sum(CASE WHEN terminal.active_root_window_id IS NULL
                                      THEN 1 ELSE 0 END), 0)
             FROM active_assignments AS assignment
             LEFT JOIN (
                SELECT DISTINCT active_root_window_id
                FROM active_root_window_state_events WHERE state <> 'open'
             ) AS terminal
               ON terminal.active_root_window_id = assignment.active_root_window_id
             WHERE assignment.active_experiment_id = ?1
               AND assignment.tranche_ordinal = ?2",
            params![
                active_experiment_id.to_string(),
                i64::try_from(tranche_ordinal).map_err(|_| invariant())?
            ],
            |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                ))
            },
        )
        .map_err(database_error)
}

fn latest_tranche_state(
    connection: &Connection,
    active_experiment_id: Uuid,
    tranche_ordinal: u64,
) -> Result<Option<String>, LedgerError> {
    connection
        .query_row(
            "SELECT state FROM active_tranche_state_events
             WHERE active_experiment_id = ?1 AND tranche_ordinal = ?2
             ORDER BY event_seq DESC LIMIT 1",
            params![
                active_experiment_id.to_string(),
                i64::try_from(tranche_ordinal).map_err(|_| invariant())?
            ],
            |row| row.get(0),
        )
        .optional()
        .map_err(database_error)
}

fn append_tranche_state_if_new(
    connection: &Connection,
    active_experiment_id: Uuid,
    tranche_ordinal: u64,
    state: &str,
    counts: (i64, i64, i64),
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    if latest_tranche_state(connection, active_experiment_id, tranche_ordinal)?.as_deref()
        == Some(state)
    {
        return Ok(());
    }
    insert_tranche_state(
        connection,
        Uuid::now_v7(),
        active_experiment_id,
        tranche_ordinal,
        state,
        u64::try_from(counts.0).map_err(|_| corrupt())?,
        u64::try_from(counts.1).map_err(|_| corrupt())?,
        u64::try_from(counts.2).map_err(|_| corrupt())?,
        process_instance_id,
        created_at_unix_ms,
    )
}

fn append_experiment_state_if_new(
    connection: &Connection,
    active_experiment_id: Uuid,
    state: &str,
    terminal_reason: Option<&str>,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let latest = connection
        .query_row(
            "SELECT state, terminal_reason FROM active_experiment_state_events
             WHERE active_experiment_id = ?1 ORDER BY event_seq DESC LIMIT 1",
            [active_experiment_id.to_string()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, Option<String>>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    if latest
        .as_ref()
        .is_some_and(|latest| latest.0 == state && latest.1.as_deref() == terminal_reason)
    {
        return Ok(());
    }
    insert_experiment_state(
        connection,
        Uuid::now_v7(),
        active_experiment_id,
        state,
        terminal_reason,
        process_instance_id,
        created_at_unix_ms,
    )
}

#[cfg(test)]
impl super::LedgerRepository {
    pub(crate) fn force_active_experiment_terminal_for_retention(
        &mut self,
        active_experiment_id: Uuid,
        created_at_unix_ms: i64,
    ) -> Result<(), LedgerError> {
        let tranches = self
            .connection
            .prepare(
                "SELECT tranche_ordinal FROM active_experiment_tranches
                 WHERE active_experiment_id = ?1 ORDER BY tranche_ordinal",
            )
            .map_err(database_error)?
            .query_map([active_experiment_id.to_string()], |row| {
                row.get::<_, i64>(0)
            })
            .map_err(database_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(database_error)?;
        for tranche_ordinal in tranches {
            let tranche_ordinal = u64::try_from(tranche_ordinal).map_err(|_| corrupt())?;
            let counts = tranche_counts(&self.connection, active_experiment_id, tranche_ordinal)?;
            if counts.2 != 0 {
                return Err(invariant());
            }
            for state in ["closed", "drained", "complete"] {
                append_tranche_state_if_new(
                    &self.connection,
                    active_experiment_id,
                    tranche_ordinal,
                    state,
                    counts,
                    self.process_instance_id,
                    created_at_unix_ms,
                )?;
            }
        }
        append_experiment_state_if_new(
            &self.connection,
            active_experiment_id,
            "terminal",
            Some("closed_passed"),
            self.process_instance_id,
            created_at_unix_ms,
        )
    }
}

fn hash_json(value: Json) -> Result<String, LedgerError> {
    canonical_sha256(&value).map_err(|_| LedgerErrorClass::CanonicalizationFailed.into())
}

fn execute_one(
    connection: &Connection,
    sql: &str,
    parameters: impl rusqlite::Params,
) -> Result<(), LedgerError> {
    if connection
        .execute(sql, parameters)
        .map_err(database_error)?
        != 1
    {
        return Err(corrupt());
    }
    Ok(())
}

fn is_uuid_v7(value: Uuid) -> bool {
    value.get_version_num() == 7 && value.get_variant() == Variant::RFC4122
}

fn parse_uuid_v7(value: &str) -> Result<Uuid, LedgerError> {
    let parsed = Uuid::parse_str(value).map_err(|_| corrupt())?;
    if parsed.to_string() != value || !is_uuid_v7(parsed) {
        return Err(corrupt());
    }
    Ok(parsed)
}

fn is_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_text(value: &str, max_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= max_bytes
        && value.chars().all(|character| !character.is_control())
}

fn valid_stable_reason(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 128
        && value.bytes().all(|byte| {
            byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"._-".contains(&byte)
        })
}

fn now_unix_ms() -> Result<i64, LedgerError> {
    let now = Utc::now().timestamp_millis();
    if now < 0 {
        Err(LedgerErrorClass::DatabaseOperationFailed.into())
    } else {
        Ok(now)
    }
}

fn invariant() -> LedgerError {
    LedgerErrorClass::IdentityInvariant.into()
}

fn corrupt() -> LedgerError {
    LedgerErrorClass::CorruptDatabase.into()
}

fn database_error(_error: rusqlite::Error) -> LedgerError {
    LedgerErrorClass::DatabaseOperationFailed.into()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveLookInclusionStatus {
    Eligible,
    Unattributed,
    Orphaned,
    AmbiguousExposure,
}

impl ActiveLookInclusionStatus {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Eligible => "eligible",
            Self::Unattributed => "unattributed",
            Self::Orphaned => "orphaned",
            Self::AmbiguousExposure => "ambiguous_exposure",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveFrozenLookMember {
    pub(crate) active_assignment_id: Uuid,
    pub(crate) outcome_id: Uuid,
    pub(crate) cap_ordinal: u64,
    pub(crate) admission_unix_ms: i64,
    pub(crate) arm: ActiveOutcomeArmV1,
    pub(crate) label: Option<ActiveOutcomeLabelV1>,
    pub(crate) inclusion_status: ActiveLookInclusionStatus,
}

impl ActiveFrozenLookMember {
    pub(crate) const fn math_member(self) -> ActiveLookMemberV1 {
        ActiveLookMemberV1 {
            cap_ordinal: self.cap_ordinal,
            admission_unix_ms: self.admission_unix_ms,
            arm: self.arm,
            label: self.label,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ActiveLookClaimReceipt {
    pub(crate) active_look_claim_id: Uuid,
    pub(crate) active_experiment_id: Uuid,
    pub(crate) boundary_tranche_ordinal: u64,
    pub(crate) boundary_nonholdout_count: u64,
    pub(crate) expected_look_ordinal: u64,
    pub(crate) expected_prior_authorization_state_event_id: Uuid,
    pub(crate) evaluating_authorization_state_event_id: Uuid,
    pub(crate) input_aggregate_hash: String,
    pub(crate) as_of_unix_ms: i64,
    pub(crate) treatment_denominator: u32,
    pub(crate) control_denominator: u32,
    pub(crate) lease_token_hash: String,
    pub(crate) lease_expires_at_unix_ms: i64,
    pub(crate) policy: ActiveLookPolicyV1,
    pub(crate) members: Vec<ActiveFrozenLookMember>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveLookClaimRequest {
    pub(crate) active_look_claim_id: Uuid,
    pub(crate) claim_state_event_id: Uuid,
    pub(crate) evaluating_authorization_state_event_id: Uuid,
    pub(crate) skipped_failure_id: Uuid,
    pub(crate) active_experiment_id: Uuid,
    pub(crate) lease_token_hash: String,
    pub(crate) lease_duration_millis: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveBoundarySkipReceipt {
    pub(crate) active_look_failure_id: Uuid,
    pub(crate) active_experiment_id: Uuid,
    pub(crate) boundary_tranche_ordinal: u64,
    pub(crate) boundary_nonholdout_count: u64,
    pub(crate) treatment_labeled: u32,
    pub(crate) control_labeled: u32,
    pub(crate) exhausted: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ActiveLookClaimAck {
    Claimed(ActiveLookClaimReceipt),
    AlreadyOwned(ActiveLookClaimReceipt),
    Reclaimed(ActiveLookClaimReceipt),
    Skipped(ActiveBoundarySkipReceipt),
    Busy { lease_expires_at_unix_ms: i64 },
    NoBoundary,
    AuthorityChanged,
    TransactionNotStarted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveLookLeaseRenewal {
    pub(crate) active_look_claim_id: Uuid,
    pub(crate) claim_state_event_id: Uuid,
    pub(crate) lease_token_hash: String,
    pub(crate) lease_duration_millis: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveLookLeaseAck {
    Renewed { lease_expires_at_unix_ms: i64 },
    AlreadyRenewed { lease_expires_at_unix_ms: i64 },
    LeaseLost,
    AuthorityChanged,
    TransactionNotStarted,
}

impl LedgerRepository {
    pub(crate) fn claim_next_active_look(
        &mut self,
        request: &ActiveLookClaimRequest,
    ) -> Result<ActiveLookClaimAck, LedgerError> {
        self.claim_next_active_look_with_start_check(request, || Some(()))
    }

    pub(crate) fn claim_next_active_look_with_start_check<G: TransactionStartGuard>(
        &mut self,
        request: &ActiveLookClaimRequest,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveLookClaimAck, LedgerError> {
        validate_claim_request(request)?;
        let database_path = self.database_path.clone();
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveLookClaimAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(ActiveLookClaimAck::TransactionNotStarted);
            }
            Err(error) => {
                return Err(super::map_sqlite_error(
                    &error,
                    LedgerErrorClass::DatabaseOperationFailed,
                ));
            }
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ActiveLookClaimAck::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = claim_next_active_look_in_transaction(
            &transaction,
            self.project_uuid,
            self.process_instance_id,
            request,
            now_unix_ms()?,
        )?;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }

    pub(crate) fn renew_active_look_lease(
        &mut self,
        renewal: &ActiveLookLeaseRenewal,
    ) -> Result<ActiveLookLeaseAck, LedgerError> {
        self.renew_active_look_lease_with_start_check(renewal, || Some(()))
    }

    pub(crate) fn renew_active_look_lease_with_start_check<G: TransactionStartGuard>(
        &mut self,
        renewal: &ActiveLookLeaseRenewal,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveLookLeaseAck, LedgerError> {
        validate_lease_renewal(renewal)?;
        let database_path = self.database_path.clone();
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveLookLeaseAck::TransactionNotStarted);
        };
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| {
                super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
            })?;
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ActiveLookLeaseAck::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = renew_active_look_lease_in_transaction(
            &transaction,
            self.project_uuid,
            self.process_instance_id,
            renewal,
            now_unix_ms()?,
        )?;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }
}

fn validate_claim_request(request: &ActiveLookClaimRequest) -> Result<(), LedgerError> {
    if [
        request.active_look_claim_id,
        request.claim_state_event_id,
        request.evaluating_authorization_state_event_id,
        request.skipped_failure_id,
        request.active_experiment_id,
    ]
    .into_iter()
    .any(|value| !is_uuid_v7(value))
        || !is_hash(&request.lease_token_hash)
        || !(1..=ACTIVE_LOOK_LEASE_MAX_MILLIS).contains(&request.lease_duration_millis)
    {
        return Err(invariant());
    }
    Ok(())
}

fn validate_lease_renewal(renewal: &ActiveLookLeaseRenewal) -> Result<(), LedgerError> {
    if !is_uuid_v7(renewal.active_look_claim_id)
        || !is_uuid_v7(renewal.claim_state_event_id)
        || !is_hash(&renewal.lease_token_hash)
        || !(1..=ACTIVE_LOOK_LEASE_MAX_MILLIS).contains(&renewal.lease_duration_millis)
    {
        return Err(invariant());
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct LoadedExperimentPolicy {
    create: ActiveExperimentCreate,
    experiment_state: String,
}

fn claim_next_active_look_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    request: &ActiveLookClaimRequest,
    observed_at_unix_ms: i64,
) -> Result<ActiveLookClaimAck, LedgerError> {
    validate_claim_request(request)?;
    if let Some(receipt) = load_skipped_boundary_by_id(transaction, request)? {
        return Ok(ActiveLookClaimAck::Skipped(receipt));
    }
    if let Some(receipt) = load_claim_by_id(
        transaction,
        project_uuid,
        process_instance_id,
        request.active_look_claim_id,
    )? {
        if receipt.lease_token_hash == request.lease_token_hash
            && receipt.lease_expires_at_unix_ms > observed_at_unix_ms
        {
            return Ok(ActiveLookClaimAck::AlreadyOwned(receipt));
        }
        return Ok(ActiveLookClaimAck::Busy {
            lease_expires_at_unix_ms: receipt.lease_expires_at_unix_ms,
        });
    }
    let policy = load_experiment_policy(transaction, project_uuid, request.active_experiment_id)?
        .ok_or_else(corrupt)?;
    if !claim_authority_matches(transaction, project_uuid, process_instance_id, &policy)? {
        return Ok(ActiveLookClaimAck::AuthorityChanged);
    }
    let Some(boundary) = next_drained_boundary(
        transaction,
        request.active_experiment_id,
        policy.create.outcome_evaluation_batch_size,
        &policy.experiment_state,
        process_instance_id,
        observed_at_unix_ms,
    )?
    else {
        return Ok(ActiveLookClaimAck::NoBoundary);
    };
    let members = load_frozen_members(
        transaction,
        request.active_experiment_id,
        boundary.tranche_ordinal,
    )?;
    let input_aggregate_hash = frozen_member_aggregate_hash(&members)?;
    let (treatment_denominator, control_denominator, treatment_labeled, control_labeled) =
        member_counts(&members)?;
    if treatment_labeled < policy.create.min_treatment_roots
        || control_labeled < policy.create.min_control_roots
    {
        let reason = match (
            treatment_labeled < policy.create.min_treatment_roots,
            control_labeled < policy.create.min_control_roots,
        ) {
            (true, true) => "insufficient_both_arms",
            (true, false) => "insufficient_treatment_roots",
            (false, true) => "insufficient_control_roots",
            (false, false) => return Err(corrupt()),
        };
        let receipt = record_skipped_boundary(
            transaction,
            process_instance_id,
            request,
            &policy,
            boundary,
            &input_aggregate_hash,
            treatment_labeled,
            control_labeled,
            reason,
            observed_at_unix_ms,
        )?;
        return Ok(ActiveLookClaimAck::Skipped(receipt));
    }
    if let Some(existing) = load_active_boundary_claim(
        transaction,
        project_uuid,
        process_instance_id,
        request.active_experiment_id,
        boundary.tranche_ordinal,
        boundary.nonholdout_count,
    )? {
        if existing.lease_expires_at_unix_ms > observed_at_unix_ms {
            return Ok(ActiveLookClaimAck::Busy {
                lease_expires_at_unix_ms: existing.lease_expires_at_unix_ms,
            });
        }
        let lease_expires_at_unix_ms =
            checked_lease_expiry(observed_at_unix_ms, request.lease_duration_millis)?;
        insert_claim_state(
            transaction,
            request.claim_state_event_id,
            existing.active_look_claim_id,
            request.active_experiment_id,
            "reclaimed",
            Some(&request.lease_token_hash),
            Some(lease_expires_at_unix_ms),
            process_instance_id,
            observed_at_unix_ms,
        )?;
        let receipt = load_claim_by_id(
            transaction,
            project_uuid,
            process_instance_id,
            existing.active_look_claim_id,
        )?
        .ok_or_else(corrupt)?;
        return Ok(ActiveLookClaimAck::Reclaimed(receipt));
    }
    let latest_authorization =
        latest_authorization(transaction, request.active_experiment_id)?.ok_or_else(corrupt)?;
    if latest_authorization.state == "evaluating"
        || matches!(
            latest_authorization.state.as_str(),
            "rollback"
                | "invalidated"
                | "superseded_draining"
                | "superseded"
                | "closed_passed"
                | "exhausted"
        )
    {
        return Ok(ActiveLookClaimAck::AuthorityChanged);
    }
    let expected_look_ordinal = next_look_ordinal(transaction, request.active_experiment_id)?;
    let lease_expires_at_unix_ms =
        checked_lease_expiry(observed_at_unix_ms, request.lease_duration_millis)?;
    let claim_hash = claim_payload_hash(
        request.active_look_claim_id,
        request.active_experiment_id,
        boundary,
        expected_look_ordinal,
        latest_authorization.event_id,
        &input_aggregate_hash,
        observed_at_unix_ms,
        treatment_denominator,
        control_denominator,
        process_instance_id,
    )?;
    execute_one(
        transaction,
        "INSERT INTO active_look_claims (
            active_look_claim_id, active_experiment_id,
            boundary_tranche_ordinal, boundary_nonholdout_count,
            expected_look_ordinal, expected_prior_authorization_state_event_id,
            input_aggregate_hash, as_of_unix_ms, treatment_denominator,
            control_denominator, owner_process_instance_id,
            created_at_unix_ms, canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
        params![
            request.active_look_claim_id.to_string(),
            request.active_experiment_id.to_string(),
            i64::try_from(boundary.tranche_ordinal).map_err(|_| invariant())?,
            i64::try_from(boundary.nonholdout_count).map_err(|_| invariant())?,
            i64::try_from(expected_look_ordinal).map_err(|_| invariant())?,
            latest_authorization.event_id.to_string(),
            input_aggregate_hash,
            observed_at_unix_ms,
            i64::from(treatment_denominator),
            i64::from(control_denominator),
            process_instance_id.to_string(),
            observed_at_unix_ms,
            claim_hash,
        ],
    )?;
    insert_claim_state(
        transaction,
        request.claim_state_event_id,
        request.active_look_claim_id,
        request.active_experiment_id,
        "claimed",
        Some(&request.lease_token_hash),
        Some(lease_expires_at_unix_ms),
        process_instance_id,
        observed_at_unix_ms,
    )?;
    let counts = tranche_counts(
        transaction,
        request.active_experiment_id,
        boundary.tranche_ordinal,
    )?;
    insert_tranche_state(
        transaction,
        Uuid::now_v7(),
        request.active_experiment_id,
        boundary.tranche_ordinal,
        "evaluating",
        u64::try_from(counts.0).map_err(|_| corrupt())?,
        u64::try_from(counts.1).map_err(|_| corrupt())?,
        0,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    insert_authorization_state(
        transaction,
        request.evaluating_authorization_state_event_id,
        request.active_experiment_id,
        Some(latest_authorization.event_id),
        None,
        Some(request.active_look_claim_id),
        "evaluating",
        None,
        policy.create.control_generation,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    let receipt = load_claim_by_id(
        transaction,
        project_uuid,
        process_instance_id,
        request.active_look_claim_id,
    )?
    .ok_or_else(corrupt)?;
    Ok(ActiveLookClaimAck::Claimed(receipt))
}

fn load_skipped_boundary_by_id(
    connection: &Connection,
    request: &ActiveLookClaimRequest,
) -> Result<Option<ActiveBoundarySkipReceipt>, LedgerError> {
    let row = connection
        .query_row(
            "SELECT active_experiment_id, boundary_tranche_ordinal,
                    boundary_nonholdout_count, stable_reason,
                    input_aggregate_hash, process_instance_id,
                    created_at_unix_ms, canonical_payload_hash
             FROM active_look_failures
             WHERE active_look_failure_id = ?1 AND failure_kind = 'skipped'",
            [request.skipped_failure_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, String>(7)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(row) = row else {
        return Ok(None);
    };
    let active_experiment_id = parse_uuid_v7(&row.0)?;
    if active_experiment_id != request.active_experiment_id {
        return Err(corrupt());
    }
    let boundary = ActiveBoundary {
        tranche_ordinal: u64::try_from(row.1).map_err(|_| corrupt())?,
        nonholdout_count: u64::try_from(row.2).map_err(|_| corrupt())?,
    };
    let members = load_frozen_members(connection, active_experiment_id, boundary.tranche_ordinal)?;
    let (_, _, treatment_labeled, control_labeled) = member_counts(&members)?;
    if frozen_member_aggregate_hash(&members)? != row.4 {
        return Err(corrupt());
    }
    let process_instance_id = parse_uuid_v7(&row.5)?;
    let expected_hash = hash_json(json!({
        "shape": "active_look_failure_v1",
        "active_look_failure_id": request.skipped_failure_id,
        "active_experiment_id": active_experiment_id,
        "active_look_claim_id": None::<Uuid>,
        "boundary_tranche_ordinal": boundary.tranche_ordinal.to_string(),
        "boundary_nonholdout_count": boundary.nonholdout_count.to_string(),
        "failure_kind": "skipped",
        "stable_reason": row.3,
        "input_aggregate_hash": row.4,
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": row.6.to_string(),
    }))?;
    if expected_hash != row.7 {
        return Err(corrupt());
    }
    let exhausted: bool = connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM active_experiment_state_events
                WHERE active_experiment_id = ?1 AND state = 'terminal'
                  AND terminal_reason = 'exhausted'
             )",
            [active_experiment_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    Ok(Some(ActiveBoundarySkipReceipt {
        active_look_failure_id: request.skipped_failure_id,
        active_experiment_id,
        boundary_tranche_ordinal: boundary.tranche_ordinal,
        boundary_nonholdout_count: boundary.nonholdout_count,
        treatment_labeled,
        control_labeled,
        exhausted,
    }))
}

#[derive(Debug, Clone, Copy)]
struct ActiveBoundary {
    tranche_ordinal: u64,
    nonholdout_count: u64,
}

#[derive(Debug, Clone)]
struct LatestAuthorization {
    event_id: Uuid,
    active_outcome_look_id: Option<Uuid>,
    state: String,
    valid_until_unix_ms: Option<i64>,
}

fn load_experiment_policy(
    connection: &Connection,
    project_uuid: Uuid,
    active_experiment_id: Uuid,
) -> Result<Option<LoadedExperimentPolicy>, LedgerError> {
    let row = connection
        .query_row(
            "SELECT project_uuid, pool_id, candidate_id, partition_hash,
                    config_generation_id, policy_version_id, outcome_policy_hash,
                    learning_generation_id, cohort_generation_id, vector_space_id,
                    control_generation, outcome_evaluation_batch_size,
                    max_canary_roots, max_looks, min_treatment_roots,
                    min_control_roots, min_treatment_effective_weight_bits,
                    min_control_effective_weight_bits, noninferiority_margin_bits,
                    noninferiority_probability_bits, rollback_probability_bits,
                    promotion_lower_bound_bits, retention_lower_bound_bits,
                    holdout_probability_bits, active_canary_fraction_bits,
                    actual_outcome_half_life_seconds,
                    anchor_shadow_half_life_seconds, authorization_ttl_seconds,
                    active_math_algorithm_id_sha256
             FROM active_experiments WHERE active_experiment_id = ?1",
            [active_experiment_id.to_string()],
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
                    row.get::<_, String>(9)?,
                    row.get::<_, i64>(10)?,
                    row.get::<_, i64>(11)?,
                    row.get::<_, i64>(12)?,
                    row.get::<_, i64>(13)?,
                    row.get::<_, i64>(14)?,
                    row.get::<_, i64>(15)?,
                    row.get::<_, i64>(16)?,
                    row.get::<_, i64>(17)?,
                    row.get::<_, i64>(18)?,
                    row.get::<_, i64>(19)?,
                    row.get::<_, i64>(20)?,
                    row.get::<_, i64>(21)?,
                    row.get::<_, i64>(22)?,
                    row.get::<_, i64>(23)?,
                    row.get::<_, i64>(24)?,
                    row.get::<_, i64>(25)?,
                    row.get::<_, i64>(26)?,
                    row.get::<_, i64>(27)?,
                    row.get::<_, String>(28)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(row) = row else {
        return Ok(None);
    };
    if row.0 != project_uuid.to_string() || row.28 != active_math_algorithm_identity_v1() {
        return Err(corrupt());
    }
    let create = ActiveExperimentCreate {
        active_experiment_id,
        initial_experiment_state_event_id: Uuid::now_v7(),
        initial_tranche_state_event_id: Uuid::now_v7(),
        initial_authorization_state_event_id: Uuid::now_v7(),
        pool_id: row.1,
        candidate_id: row.2,
        partition_hash: row.3,
        config_generation_id: row.4,
        policy_version_id: row.5,
        outcome_policy_hash: row.6,
        learning_generation_id: parse_uuid_v7(&row.7)?,
        cohort_generation_id: parse_uuid_v7(&row.8)?,
        vector_space_id: row.9,
        control_generation: u64::try_from(row.10).map_err(|_| corrupt())?,
        outcome_evaluation_batch_size: u32::try_from(row.11).map_err(|_| corrupt())?,
        max_canary_roots: u32::try_from(row.12).map_err(|_| corrupt())?,
        max_looks: u32::try_from(row.13).map_err(|_| corrupt())?,
        min_treatment_roots: u32::try_from(row.14).map_err(|_| corrupt())?,
        min_control_roots: u32::try_from(row.15).map_err(|_| corrupt())?,
        min_treatment_effective_weight_bits: row.16 as u64,
        min_control_effective_weight_bits: row.17 as u64,
        noninferiority_margin_bits: row.18 as u64,
        noninferiority_probability_bits: row.19 as u64,
        rollback_probability_bits: row.20 as u64,
        promotion_lower_bound_bits: row.21 as u64,
        retention_lower_bound_bits: row.22 as u64,
        holdout_probability_bits: row.23 as u64,
        active_canary_fraction_bits: row.24 as u64,
        actual_outcome_half_life_seconds: u32::try_from(row.25).map_err(|_| corrupt())?,
        anchor_shadow_half_life_seconds: u32::try_from(row.26).map_err(|_| corrupt())?,
        authorization_ttl_seconds: u32::try_from(row.27).map_err(|_| corrupt())?,
    };
    validate_experiment_create(&create).map_err(|_| corrupt())?;
    if !matches!(
        load_existing_experiment(connection, project_uuid, &create)?,
        Some(ActiveExperimentCreateAck::AlreadyApplied(_))
    ) {
        return Err(corrupt());
    }
    let experiment_state = connection
        .query_row(
            "SELECT state FROM active_experiment_state_events
             WHERE active_experiment_id = ?1 ORDER BY event_seq DESC LIMIT 1",
            [active_experiment_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    Ok(Some(LoadedExperimentPolicy {
        create,
        experiment_state,
    }))
}

fn claim_authority_matches(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    policy: &LoadedExperimentPolicy,
) -> Result<bool, LedgerError> {
    if !matches!(
        policy.experiment_state.as_str(),
        "collecting" | "cap_draining"
    ) || !experiment_authority_matches(
        connection,
        project_uuid,
        process_instance_id,
        &policy.create,
    )? {
        return Ok(false);
    }
    connection
        .query_row(
            "SELECT NOT EXISTS(
                SELECT 1 FROM active_retirement_markers
                WHERE active_experiment_id = ?1
             )",
            [policy.create.active_experiment_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn next_drained_boundary(
    connection: &Connection,
    active_experiment_id: Uuid,
    batch_size: u32,
    experiment_state: &str,
    process_instance_id: Uuid,
    observed_at_unix_ms: i64,
) -> Result<Option<ActiveBoundary>, LedgerError> {
    loop {
        let tranche_ordinal = connection
            .query_row(
                "SELECT tranche.tranche_ordinal
                 FROM active_experiment_tranches AS tranche
                 JOIN active_tranche_state_events AS state
                   ON state.active_experiment_id = tranche.active_experiment_id
                  AND state.tranche_ordinal = tranche.tranche_ordinal
                 WHERE tranche.active_experiment_id = ?1
                   AND state.event_seq = (
                        SELECT max(latest.event_seq)
                        FROM active_tranche_state_events AS latest
                        WHERE latest.active_experiment_id = tranche.active_experiment_id
                          AND latest.tranche_ordinal = tranche.tranche_ordinal
                   )
                   AND state.state IN ('drained', 'evaluating')
                   AND NOT EXISTS (
                        SELECT 1 FROM active_outcome_looks AS look
                        WHERE look.active_experiment_id = tranche.active_experiment_id
                          AND look.boundary_tranche_ordinal = tranche.tranche_ordinal
                   )
                   AND NOT EXISTS (
                        SELECT 1 FROM active_look_failures AS failure
                        WHERE failure.active_experiment_id = tranche.active_experiment_id
                          AND failure.boundary_tranche_ordinal = tranche.tranche_ordinal
                   )
                 ORDER BY tranche.tranche_ordinal LIMIT 1",
                [active_experiment_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .map_err(database_error)?;
        let Some(tranche_ordinal) = tranche_ordinal else {
            return Ok(None);
        };
        let tranche_ordinal = u64::try_from(tranche_ordinal).map_err(|_| corrupt())?;
        let nonholdout_count = connection
            .query_row(
                "SELECT count(cap_ordinal) FROM active_assignments
                 WHERE active_experiment_id = ?1 AND tranche_ordinal <= ?2",
                params![
                    active_experiment_id.to_string(),
                    i64::try_from(tranche_ordinal).map_err(|_| corrupt())?
                ],
                |row| row.get::<_, i64>(0),
            )
            .map_err(database_error)?;
        let nonholdout_count = u64::try_from(nonholdout_count).map_err(|_| corrupt())?;
        let last_boundary = connection
            .query_row(
                "SELECT max(boundary_nonholdout_count) FROM (
                    SELECT boundary_nonholdout_count FROM active_outcome_looks
                    WHERE active_experiment_id = ?1
                    UNION ALL
                    SELECT boundary_nonholdout_count FROM active_look_failures
                    WHERE active_experiment_id = ?1
                 )",
                [active_experiment_id.to_string()],
                |row| row.get::<_, Option<i64>>(0),
            )
            .map_err(database_error)?
            .unwrap_or(0);
        let last_boundary = u64::try_from(last_boundary).map_err(|_| corrupt())?;
        let next_target = last_boundary
            .checked_div(u64::from(batch_size))
            .and_then(|value| value.checked_add(1))
            .and_then(|value| value.checked_mul(u64::from(batch_size)))
            .ok_or_else(corrupt)?;
        let latest_state = latest_tranche_state(connection, active_experiment_id, tranche_ordinal)?
            .ok_or_else(corrupt)?;
        if latest_state == "evaluating"
            || nonholdout_count >= next_target
            || experiment_state == "cap_draining"
        {
            return Ok(Some(ActiveBoundary {
                tranche_ordinal,
                nonholdout_count,
            }));
        }
        let counts = tranche_counts(connection, active_experiment_id, tranche_ordinal)?;
        append_tranche_state_if_new(
            connection,
            active_experiment_id,
            tranche_ordinal,
            "complete",
            counts,
            process_instance_id,
            observed_at_unix_ms,
        )?;
        open_next_tranche(
            connection,
            active_experiment_id,
            tranche_ordinal,
            batch_size,
            process_instance_id,
            observed_at_unix_ms,
        )?;
    }
}

fn load_frozen_members(
    connection: &Connection,
    active_experiment_id: Uuid,
    boundary_tranche_ordinal: u64,
) -> Result<Vec<ActiveFrozenLookMember>, LedgerError> {
    let rows = connection
        .prepare(
            "SELECT assignment.active_assignment_id, assignment.cap_ordinal,
                    assignment.admission_unix_ms, assignment.arm,
                    outcome.outcome_id, outcome.label, outcome.attribution_status
             FROM active_assignments AS assignment
             LEFT JOIN outcomes AS outcome
               ON outcome.active_assignment_id = assignment.active_assignment_id
             WHERE assignment.active_experiment_id = ?1
               AND assignment.tranche_ordinal <= ?2
               AND assignment.cap_ordinal IS NOT NULL
             ORDER BY assignment.cap_ordinal",
        )
        .map_err(database_error)?
        .query_map(
            params![
                active_experiment_id.to_string(),
                i64::try_from(boundary_tranche_ordinal).map_err(|_| invariant())?
            ],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, Option<String>>(5)?,
                    row.get::<_, Option<String>>(6)?,
                ))
            },
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    let mut members = Vec::with_capacity(rows.len());
    for (index, row) in rows.into_iter().enumerate() {
        let cap_ordinal = u64::try_from(row.1).map_err(|_| corrupt())?;
        if cap_ordinal != index as u64 + 1 {
            return Err(corrupt());
        }
        let arm = match row.3.as_str() {
            "candidate_treatment" => ActiveOutcomeArmV1::Treatment,
            "anchor_control" => ActiveOutcomeArmV1::Control,
            _ => return Err(corrupt()),
        };
        let outcome_id = row
            .4
            .as_deref()
            .ok_or_else(corrupt)
            .and_then(parse_uuid_v7)?;
        let (label, inclusion_status) = match (row.5.as_deref(), row.6.as_deref(), arm) {
            (Some("success"), Some("eligible_treatment"), ActiveOutcomeArmV1::Treatment)
            | (Some("success"), Some("eligible_control"), ActiveOutcomeArmV1::Control) => (
                Some(ActiveOutcomeLabelV1::Success),
                ActiveLookInclusionStatus::Eligible,
            ),
            (Some("failure"), Some("eligible_treatment"), ActiveOutcomeArmV1::Treatment)
            | (Some("failure"), Some("eligible_control"), ActiveOutcomeArmV1::Control) => (
                Some(ActiveOutcomeLabelV1::Failure),
                ActiveLookInclusionStatus::Eligible,
            ),
            (None, Some("unattributed"), _) => (None, ActiveLookInclusionStatus::Unattributed),
            (None, Some("orphaned"), _) => (None, ActiveLookInclusionStatus::Orphaned),
            (None, Some("ambiguous_exposure"), _) => {
                (None, ActiveLookInclusionStatus::AmbiguousExposure)
            }
            _ => return Err(corrupt()),
        };
        members.push(ActiveFrozenLookMember {
            active_assignment_id: parse_uuid_v7(&row.0)?,
            outcome_id,
            cap_ordinal,
            admission_unix_ms: row.2,
            arm,
            label,
            inclusion_status,
        });
    }
    Ok(members)
}

fn member_counts(members: &[ActiveFrozenLookMember]) -> Result<(u32, u32, u32, u32), LedgerError> {
    let mut treatment_denominator = 0_u32;
    let mut control_denominator = 0_u32;
    let mut treatment_labeled = 0_u32;
    let mut control_labeled = 0_u32;
    for member in members {
        match member.arm {
            ActiveOutcomeArmV1::Treatment => {
                treatment_denominator = treatment_denominator.checked_add(1).ok_or_else(corrupt)?;
                if member.label.is_some() {
                    treatment_labeled = treatment_labeled.checked_add(1).ok_or_else(corrupt)?;
                }
            }
            ActiveOutcomeArmV1::Control => {
                control_denominator = control_denominator.checked_add(1).ok_or_else(corrupt)?;
                if member.label.is_some() {
                    control_labeled = control_labeled.checked_add(1).ok_or_else(corrupt)?;
                }
            }
        }
    }
    Ok((
        treatment_denominator,
        control_denominator,
        treatment_labeled,
        control_labeled,
    ))
}

fn frozen_member_aggregate_hash(members: &[ActiveFrozenLookMember]) -> Result<String, LedgerError> {
    let values = members
        .iter()
        .map(|member| {
            json!({
                "active_assignment_id": member.active_assignment_id,
                "outcome_id": member.outcome_id,
                "cap_ordinal": member.cap_ordinal.to_string(),
                "admission_unix_ms": member.admission_unix_ms.to_string(),
                "arm": match member.arm {
                    ActiveOutcomeArmV1::Treatment => "treatment",
                    ActiveOutcomeArmV1::Control => "control",
                },
                "label": match member.label {
                    Some(ActiveOutcomeLabelV1::Success) => Some("success"),
                    Some(ActiveOutcomeLabelV1::Failure) => Some("failure"),
                    None => None,
                },
                "inclusion_status": member.inclusion_status.as_str(),
            })
        })
        .collect::<Vec<_>>();
    hash_json(json!({
        "shape": "active_look_input_aggregate_v1",
        "members": values,
    }))
}

#[allow(clippy::too_many_arguments)]
fn record_skipped_boundary(
    connection: &Connection,
    process_instance_id: Uuid,
    request: &ActiveLookClaimRequest,
    policy: &LoadedExperimentPolicy,
    boundary: ActiveBoundary,
    input_aggregate_hash: &str,
    treatment_labeled: u32,
    control_labeled: u32,
    reason: &str,
    observed_at_unix_ms: i64,
) -> Result<ActiveBoundarySkipReceipt, LedgerError> {
    let failure_hash = hash_json(json!({
        "shape": "active_look_failure_v1",
        "active_look_failure_id": request.skipped_failure_id,
        "active_experiment_id": request.active_experiment_id,
        "active_look_claim_id": None::<Uuid>,
        "boundary_tranche_ordinal": boundary.tranche_ordinal.to_string(),
        "boundary_nonholdout_count": boundary.nonholdout_count.to_string(),
        "failure_kind": "skipped",
        "stable_reason": reason,
        "input_aggregate_hash": input_aggregate_hash,
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": observed_at_unix_ms.to_string(),
    }))?;
    execute_one(
        connection,
        "INSERT INTO active_look_failures (
            active_look_failure_id, active_experiment_id, active_look_claim_id,
            boundary_tranche_ordinal, boundary_nonholdout_count, failure_kind,
            stable_reason, input_aggregate_hash, process_instance_id,
            created_at_unix_ms, canonical_payload_hash
         ) VALUES (?1, ?2, NULL, ?3, ?4, 'skipped', ?5, ?6, ?7, ?8, ?9)",
        params![
            request.skipped_failure_id.to_string(),
            request.active_experiment_id.to_string(),
            i64::try_from(boundary.tranche_ordinal).map_err(|_| invariant())?,
            i64::try_from(boundary.nonholdout_count).map_err(|_| invariant())?,
            reason,
            input_aggregate_hash,
            process_instance_id.to_string(),
            observed_at_unix_ms,
            failure_hash,
        ],
    )?;
    let exhausted = finish_boundary(
        connection,
        process_instance_id,
        &policy.create,
        boundary.tranche_ordinal,
        "collecting",
        observed_at_unix_ms,
    )?;
    Ok(ActiveBoundarySkipReceipt {
        active_look_failure_id: request.skipped_failure_id,
        active_experiment_id: request.active_experiment_id,
        boundary_tranche_ordinal: boundary.tranche_ordinal,
        boundary_nonholdout_count: boundary.nonholdout_count,
        treatment_labeled,
        control_labeled,
        exhausted,
    })
}

fn finish_boundary(
    connection: &Connection,
    process_instance_id: Uuid,
    create: &ActiveExperimentCreate,
    tranche_ordinal: u64,
    current_authorization_state: &str,
    observed_at_unix_ms: i64,
) -> Result<bool, LedgerError> {
    let counts = tranche_counts(connection, create.active_experiment_id, tranche_ordinal)?;
    if counts.2 != 0 {
        return Err(corrupt());
    }
    append_tranche_state_if_new(
        connection,
        create.active_experiment_id,
        tranche_ordinal,
        "complete",
        counts,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    if current_authorization_state == "rollback" {
        append_experiment_state_if_new(
            connection,
            create.active_experiment_id,
            "terminal",
            Some("rollback"),
            process_instance_id,
            observed_at_unix_ms,
        )?;
        return Ok(true);
    }
    let latest_experiment_state = connection
        .query_row(
            "SELECT state FROM active_experiment_state_events
             WHERE active_experiment_id = ?1 ORDER BY event_seq DESC LIMIT 1",
            [create.active_experiment_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?;
    if latest_experiment_state == "cap_draining" {
        let latest_authorization =
            latest_authorization(connection, create.active_experiment_id)?.ok_or_else(corrupt)?;
        let final_state = if current_authorization_state == "passed" {
            "closed_passed"
        } else {
            "exhausted"
        };
        insert_authorization_state(
            connection,
            Uuid::now_v7(),
            create.active_experiment_id,
            Some(latest_authorization.event_id),
            None,
            None,
            final_state,
            None,
            create.control_generation,
            process_instance_id,
            observed_at_unix_ms,
        )?;
        append_experiment_state_if_new(
            connection,
            create.active_experiment_id,
            "terminal",
            Some(final_state),
            process_instance_id,
            observed_at_unix_ms,
        )?;
        return Ok(true);
    }
    if latest_experiment_state != "collecting" {
        return Err(corrupt());
    }
    open_next_tranche(
        connection,
        create.active_experiment_id,
        tranche_ordinal,
        create.outcome_evaluation_batch_size,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    Ok(false)
}

fn open_next_tranche(
    connection: &Connection,
    active_experiment_id: Uuid,
    completed_tranche_ordinal: u64,
    batch_size: u32,
    process_instance_id: Uuid,
    observed_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let next = completed_tranche_ordinal
        .checked_add(1)
        .ok_or_else(corrupt)?;
    let exists: bool = connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM active_experiment_tranches
                WHERE active_experiment_id = ?1 AND tranche_ordinal = ?2
             )",
            params![
                active_experiment_id.to_string(),
                i64::try_from(next).map_err(|_| corrupt())?
            ],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    if exists {
        return Ok(());
    }
    insert_tranche(
        connection,
        active_experiment_id,
        next,
        batch_size,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    insert_tranche_state(
        connection,
        Uuid::now_v7(),
        active_experiment_id,
        next,
        "open",
        0,
        0,
        0,
        process_instance_id,
        observed_at_unix_ms,
    )
}

fn latest_authorization(
    connection: &Connection,
    active_experiment_id: Uuid,
) -> Result<Option<LatestAuthorization>, LedgerError> {
    connection
        .query_row(
            "SELECT active_authorization_state_event_id, active_outcome_look_id,
                    state, valid_until_unix_ms
             FROM active_authorization_state_events
             WHERE active_experiment_id = ?1 ORDER BY event_seq DESC LIMIT 1",
            [active_experiment_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, Option<String>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<i64>>(3)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?
        .map(|row| {
            Ok(LatestAuthorization {
                event_id: parse_uuid_v7(&row.0)?,
                active_outcome_look_id: row.1.as_deref().map(parse_uuid_v7).transpose()?,
                state: row.2,
                valid_until_unix_ms: row.3,
            })
        })
        .transpose()
}

fn next_look_ordinal(
    connection: &Connection,
    active_experiment_id: Uuid,
) -> Result<u64, LedgerError> {
    let values = connection
        .query_row(
            "SELECT count(*), coalesce(max(look_ordinal), 0)
             FROM active_outcome_looks WHERE active_experiment_id = ?1",
            [active_experiment_id.to_string()],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        )
        .map_err(database_error)?;
    if values.0 != values.1 {
        return Err(corrupt());
    }
    u64::try_from(values.1)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or_else(corrupt)
}

#[allow(clippy::too_many_arguments)]
fn claim_payload_hash(
    active_look_claim_id: Uuid,
    active_experiment_id: Uuid,
    boundary: ActiveBoundary,
    expected_look_ordinal: u64,
    expected_prior_authorization_state_event_id: Uuid,
    input_aggregate_hash: &str,
    as_of_unix_ms: i64,
    treatment_denominator: u32,
    control_denominator: u32,
    owner_process_instance_id: Uuid,
) -> Result<String, LedgerError> {
    hash_json(json!({
        "shape": "active_look_claim_v1",
        "active_look_claim_id": active_look_claim_id,
        "active_experiment_id": active_experiment_id,
        "boundary_tranche_ordinal": boundary.tranche_ordinal.to_string(),
        "boundary_nonholdout_count": boundary.nonholdout_count.to_string(),
        "expected_look_ordinal": expected_look_ordinal.to_string(),
        "expected_prior_authorization_state_event_id": expected_prior_authorization_state_event_id,
        "input_aggregate_hash": input_aggregate_hash,
        "as_of_unix_ms": as_of_unix_ms.to_string(),
        "treatment_denominator": treatment_denominator.to_string(),
        "control_denominator": control_denominator.to_string(),
        "owner_process_instance_id": owner_process_instance_id,
        "created_at_unix_ms": as_of_unix_ms.to_string(),
    }))
}

#[allow(clippy::too_many_arguments)]
fn insert_claim_state(
    connection: &Connection,
    event_id: Uuid,
    claim_id: Uuid,
    experiment_id: Uuid,
    state: &str,
    lease_token_hash: Option<&str>,
    lease_expires_at_unix_ms: Option<i64>,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let payload_hash = hash_json(json!({
        "shape": "active_look_claim_state_v1",
        "event_id": event_id,
        "active_look_claim_id": claim_id,
        "active_experiment_id": experiment_id,
        "state": state,
        "lease_token_hash": lease_token_hash,
        "lease_expires_at_unix_ms": lease_expires_at_unix_ms.map(|value| value.to_string()),
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": created_at_unix_ms.to_string(),
    }))?;
    execute_one(
        connection,
        "INSERT INTO active_look_claim_state_events (
            active_look_claim_state_event_id, active_look_claim_id,
            active_experiment_id, state, lease_token_hash,
            lease_expires_at_unix_ms, process_instance_id,
            created_at_unix_ms, canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            event_id.to_string(),
            claim_id.to_string(),
            experiment_id.to_string(),
            state,
            lease_token_hash,
            lease_expires_at_unix_ms,
            process_instance_id.to_string(),
            created_at_unix_ms,
            payload_hash,
        ],
    )
}

fn checked_lease_expiry(
    observed_at_unix_ms: i64,
    lease_duration_millis: u64,
) -> Result<i64, LedgerError> {
    let duration = i64::try_from(lease_duration_millis).map_err(|_| invariant())?;
    observed_at_unix_ms
        .checked_add(duration)
        .ok_or_else(invariant)
}

fn load_claim_by_id(
    connection: &Connection,
    project_uuid: Uuid,
    _process_instance_id: Uuid,
    active_look_claim_id: Uuid,
) -> Result<Option<ActiveLookClaimReceipt>, LedgerError> {
    let row = connection
        .query_row(
            "SELECT claim.active_experiment_id, claim.boundary_tranche_ordinal,
                    claim.boundary_nonholdout_count, claim.expected_look_ordinal,
                    claim.expected_prior_authorization_state_event_id,
                    claim.input_aggregate_hash, claim.as_of_unix_ms,
                    claim.treatment_denominator, claim.control_denominator,
                    claim.owner_process_instance_id, claim.created_at_unix_ms,
                    claim.canonical_payload_hash,
                    state.state, state.lease_token_hash,
                    state.lease_expires_at_unix_ms,
                    state.process_instance_id, state.created_at_unix_ms,
                    state.active_look_claim_state_event_id,
                    state.canonical_payload_hash
             FROM active_look_claims AS claim
             JOIN active_look_claim_state_events AS state
               ON state.active_look_claim_id = claim.active_look_claim_id
             WHERE claim.active_look_claim_id = ?1
               AND state.event_seq = (
                    SELECT max(latest.event_seq)
                    FROM active_look_claim_state_events AS latest
                    WHERE latest.active_look_claim_id = claim.active_look_claim_id
               )",
            [active_look_claim_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                    row.get::<_, i64>(8)?,
                    row.get::<_, String>(9)?,
                    row.get::<_, i64>(10)?,
                    row.get::<_, String>(11)?,
                    row.get::<_, String>(12)?,
                    row.get::<_, Option<String>>(13)?,
                    row.get::<_, Option<i64>>(14)?,
                    row.get::<_, String>(15)?,
                    row.get::<_, i64>(16)?,
                    row.get::<_, String>(17)?,
                    row.get::<_, String>(18)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(row) = row else {
        return Ok(None);
    };
    if !matches!(row.12.as_str(), "claimed" | "renewed" | "reclaimed") {
        return Ok(None);
    }
    let active_experiment_id = parse_uuid_v7(&row.0)?;
    let boundary = ActiveBoundary {
        tranche_ordinal: u64::try_from(row.1).map_err(|_| corrupt())?,
        nonholdout_count: u64::try_from(row.2).map_err(|_| corrupt())?,
    };
    let expected_look_ordinal = u64::try_from(row.3).map_err(|_| corrupt())?;
    let expected_prior = parse_uuid_v7(&row.4)?;
    let treatment_denominator = u32::try_from(row.7).map_err(|_| corrupt())?;
    let control_denominator = u32::try_from(row.8).map_err(|_| corrupt())?;
    let owner_process_instance_id = parse_uuid_v7(&row.9)?;
    if row.10 != row.6
        || claim_payload_hash(
            active_look_claim_id,
            active_experiment_id,
            boundary,
            expected_look_ordinal,
            expected_prior,
            &row.5,
            row.6,
            treatment_denominator,
            control_denominator,
            owner_process_instance_id,
        )? != row.11
    {
        return Err(corrupt());
    }
    let lease_token_hash = row.13.ok_or_else(corrupt)?;
    let lease_expires_at_unix_ms = row.14.ok_or_else(corrupt)?;
    let state_process = parse_uuid_v7(&row.15)?;
    let state_event_id = parse_uuid_v7(&row.17)?;
    let expected_state_hash = hash_json(json!({
        "shape": "active_look_claim_state_v1",
        "event_id": state_event_id,
        "active_look_claim_id": active_look_claim_id,
        "active_experiment_id": active_experiment_id,
        "state": row.12,
        "lease_token_hash": lease_token_hash,
        "lease_expires_at_unix_ms": lease_expires_at_unix_ms.to_string(),
        "process_instance_id": state_process,
        "created_at_unix_ms": row.16.to_string(),
    }))?;
    if expected_state_hash != row.18 {
        return Err(corrupt());
    }
    let policy = load_experiment_policy(connection, project_uuid, active_experiment_id)?
        .ok_or_else(corrupt)?;
    let members = load_frozen_members(connection, active_experiment_id, boundary.tranche_ordinal)?;
    let aggregate_hash = frozen_member_aggregate_hash(&members)?;
    let (current_treatment, current_control, _, _) = member_counts(&members)?;
    if aggregate_hash != row.5
        || current_treatment != treatment_denominator
        || current_control != control_denominator
        || members.len() as u64 != boundary.nonholdout_count
    {
        return Err(corrupt());
    }
    let evaluating_authorization_state_event_id = connection
        .query_row(
            "SELECT active_authorization_state_event_id
             FROM active_authorization_state_events
             WHERE active_experiment_id = ?1 AND active_look_claim_id = ?2
               AND state = 'evaluating'",
            params![
                active_experiment_id.to_string(),
                active_look_claim_id.to_string()
            ],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)
        .and_then(|value| parse_uuid_v7(&value))?;
    Ok(Some(ActiveLookClaimReceipt {
        active_look_claim_id,
        active_experiment_id,
        boundary_tranche_ordinal: boundary.tranche_ordinal,
        boundary_nonholdout_count: boundary.nonholdout_count,
        expected_look_ordinal,
        expected_prior_authorization_state_event_id: expected_prior,
        evaluating_authorization_state_event_id,
        input_aggregate_hash: row.5,
        as_of_unix_ms: row.6,
        treatment_denominator,
        control_denominator,
        lease_token_hash,
        lease_expires_at_unix_ms,
        policy: policy.create.look_policy()?,
        members,
    }))
}

fn load_active_boundary_claim(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    active_experiment_id: Uuid,
    boundary_tranche_ordinal: u64,
    boundary_nonholdout_count: u64,
) -> Result<Option<ActiveLookClaimReceipt>, LedgerError> {
    let claim_id = connection
        .query_row(
            "SELECT active_look_claim_id FROM active_look_claims
             WHERE active_experiment_id = ?1
               AND boundary_tranche_ordinal = ?2
               AND boundary_nonholdout_count = ?3
             ORDER BY created_at_unix_ms, active_look_claim_id LIMIT 1",
            params![
                active_experiment_id.to_string(),
                i64::try_from(boundary_tranche_ordinal).map_err(|_| invariant())?,
                i64::try_from(boundary_nonholdout_count).map_err(|_| invariant())?
            ],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    claim_id
        .map(|value| {
            load_claim_by_id(
                connection,
                project_uuid,
                process_instance_id,
                parse_uuid_v7(&value)?,
            )
        })
        .transpose()
        .map(Option::flatten)
}

fn renew_active_look_lease_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    renewal: &ActiveLookLeaseRenewal,
    observed_at_unix_ms: i64,
) -> Result<ActiveLookLeaseAck, LedgerError> {
    validate_lease_renewal(renewal)?;
    let replay = transaction
        .query_row(
            "SELECT active_look_claim_id, state, lease_token_hash,
                    lease_expires_at_unix_ms
             FROM active_look_claim_state_events
             WHERE active_look_claim_state_event_id = ?1",
            [renewal.claim_state_event_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<i64>>(3)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    if let Some(replay) = replay {
        if replay.0 != renewal.active_look_claim_id.to_string()
            || replay.1 != "renewed"
            || replay.2.as_deref() != Some(renewal.lease_token_hash.as_str())
        {
            return Err(corrupt());
        }
        return Ok(ActiveLookLeaseAck::AlreadyRenewed {
            lease_expires_at_unix_ms: replay.3.ok_or_else(corrupt)?,
        });
    }
    let Some(receipt) = load_claim_by_id(
        transaction,
        project_uuid,
        process_instance_id,
        renewal.active_look_claim_id,
    )?
    else {
        return Ok(ActiveLookLeaseAck::LeaseLost);
    };
    let policy = load_experiment_policy(transaction, project_uuid, receipt.active_experiment_id)?
        .ok_or_else(corrupt)?;
    if !claim_authority_matches(transaction, project_uuid, process_instance_id, &policy)? {
        return Ok(ActiveLookLeaseAck::AuthorityChanged);
    }
    let latest_owner = transaction
        .query_row(
            "SELECT process_instance_id FROM active_look_claim_state_events
             WHERE active_look_claim_id = ?1 ORDER BY event_seq DESC LIMIT 1",
            [renewal.active_look_claim_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?;
    if receipt.lease_token_hash != renewal.lease_token_hash
        || receipt.lease_expires_at_unix_ms <= observed_at_unix_ms
        || latest_owner != process_instance_id.to_string()
    {
        return Ok(ActiveLookLeaseAck::LeaseLost);
    }
    let lease_expires_at_unix_ms =
        checked_lease_expiry(observed_at_unix_ms, renewal.lease_duration_millis)?;
    insert_claim_state(
        transaction,
        renewal.claim_state_event_id,
        renewal.active_look_claim_id,
        receipt.active_experiment_id,
        "renewed",
        Some(&renewal.lease_token_hash),
        Some(lease_expires_at_unix_ms),
        process_instance_id,
        observed_at_unix_ms,
    )?;
    Ok(ActiveLookLeaseAck::Renewed {
        lease_expires_at_unix_ms,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveLookCommit {
    pub(crate) active_look_claim_id: Uuid,
    pub(crate) lease_token_hash: String,
    pub(crate) active_outcome_look_id: Uuid,
    pub(crate) claim_terminal_state_event_id: Uuid,
    pub(crate) authorization_state_event_id: Uuid,
    pub(crate) audit: Box<ActiveLookAuditV1>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveLookCommitReceipt {
    pub(crate) active_outcome_look_id: Uuid,
    pub(crate) authorization_state_event_id: Uuid,
    pub(crate) result_state: LookProducedStateV1,
    pub(crate) valid_until_unix_ms: Option<u64>,
    pub(crate) experiment_terminal: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveLookCommitAck {
    Applied(ActiveLookCommitReceipt),
    AlreadyApplied(ActiveLookCommitReceipt),
    LeaseLost,
    AuthorityChanged,
    Conflict,
    TransactionNotStarted,
}

impl LedgerRepository {
    pub(crate) fn commit_active_look(
        &mut self,
        commit: &ActiveLookCommit,
    ) -> Result<ActiveLookCommitAck, LedgerError> {
        self.commit_active_look_with_start_check(commit, || Some(()))
    }

    pub(crate) fn commit_active_look_with_start_check<G: TransactionStartGuard>(
        &mut self,
        commit: &ActiveLookCommit,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveLookCommitAck, LedgerError> {
        validate_look_commit(commit)?;
        let database_path = self.database_path.clone();
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveLookCommitAck::TransactionNotStarted);
        };
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| {
                super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
            })?;
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ActiveLookCommitAck::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = commit_active_look_in_transaction(
            &transaction,
            self.project_uuid,
            self.process_instance_id,
            commit,
            now_unix_ms()?,
        )?;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }
}

fn validate_look_commit(commit: &ActiveLookCommit) -> Result<(), LedgerError> {
    if !is_uuid_v7(commit.active_look_claim_id)
        || !is_uuid_v7(commit.active_outcome_look_id)
        || !is_uuid_v7(commit.claim_terminal_state_event_id)
        || !is_uuid_v7(commit.authorization_state_event_id)
        || !is_hash(&commit.lease_token_hash)
    {
        return Err(invariant());
    }
    Ok(())
}

fn commit_active_look_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    commit: &ActiveLookCommit,
    observed_at_unix_ms: i64,
) -> Result<ActiveLookCommitAck, LedgerError> {
    validate_look_commit(commit)?;
    if let Some(existing) = load_existing_look_commit(transaction, commit)? {
        return Ok(existing);
    }
    let Some(claim) = load_claim_by_id(
        transaction,
        project_uuid,
        process_instance_id,
        commit.active_look_claim_id,
    )?
    else {
        return Ok(ActiveLookCommitAck::LeaseLost);
    };
    let policy = load_experiment_policy(transaction, project_uuid, claim.active_experiment_id)?
        .ok_or_else(corrupt)?;
    if !claim_authority_matches(transaction, project_uuid, process_instance_id, &policy)? {
        return Ok(ActiveLookCommitAck::AuthorityChanged);
    }
    let latest_owner = transaction
        .query_row(
            "SELECT process_instance_id FROM active_look_claim_state_events
             WHERE active_look_claim_id = ?1 ORDER BY event_seq DESC LIMIT 1",
            [commit.active_look_claim_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?;
    if claim.lease_token_hash != commit.lease_token_hash
        || claim.lease_expires_at_unix_ms <= observed_at_unix_ms
        || latest_owner != process_instance_id.to_string()
    {
        return Ok(ActiveLookCommitAck::LeaseLost);
    }
    let latest_authorization =
        latest_authorization(transaction, claim.active_experiment_id)?.ok_or_else(corrupt)?;
    if latest_authorization.event_id != claim.evaluating_authorization_state_event_id
        || latest_authorization.state != "evaluating"
    {
        return Ok(ActiveLookCommitAck::AuthorityChanged);
    }
    validate_audit_for_claim(&claim, &commit.audit)?;
    let audit_value = serde_json::to_value(commit.audit.as_ref())
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let canonical_audit_json = canonical_json(&audit_value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    if canonical_audit_json.len() > 1_048_576 {
        return Err(invariant());
    }
    let audit_hash = canonical_sha256(&audit_value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let result_state = commit.audit.state;
    let result_state_str = look_state_str(result_state);
    let valid_until_unix_ms = if result_state == LookProducedStateV1::Passed {
        let ttl_millis = u64::from(policy.create.authorization_ttl_seconds)
            .checked_mul(1_000)
            .ok_or_else(invariant)?;
        Some(
            u64::try_from(claim.as_of_unix_ms)
                .map_err(|_| invariant())?
                .checked_add(ttl_millis)
                .ok_or_else(invariant)?,
        )
    } else {
        None
    };
    if valid_until_unix_ms.is_some_and(|value| {
        u64::try_from(observed_at_unix_ms).map_or(true, |observed| value <= observed)
    }) {
        return Ok(ActiveLookCommitAck::AuthorityChanged);
    }
    let previous_boundary = transaction
        .query_row(
            "SELECT coalesce(max(boundary_nonholdout_count), 0)
             FROM active_outcome_looks WHERE active_experiment_id = ?1",
            [claim.active_experiment_id.to_string()],
            |row| row.get::<_, i64>(0),
        )
        .map_err(database_error)?;
    let previous_boundary = u64::try_from(previous_boundary).map_err(|_| corrupt())?;
    let look_hash = look_payload_hash(
        commit.active_outcome_look_id,
        &claim,
        result_state,
        process_instance_id,
        observed_at_unix_ms,
        &audit_hash,
    )?;
    execute_one(
        transaction,
        "INSERT INTO active_outcome_looks (
            active_outcome_look_id, active_experiment_id, active_look_claim_id,
            look_ordinal, boundary_tranche_ordinal,
            boundary_nonholdout_count, as_of_unix_ms,
            treatment_denominator, treatment_labeled,
            control_denominator, control_labeled, input_aggregate_hash,
            result_state, process_instance_id, created_at_unix_ms,
            canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16)",
        params![
            commit.active_outcome_look_id.to_string(),
            claim.active_experiment_id.to_string(),
            claim.active_look_claim_id.to_string(),
            i64::try_from(claim.expected_look_ordinal).map_err(|_| invariant())?,
            i64::try_from(claim.boundary_tranche_ordinal).map_err(|_| invariant())?,
            i64::try_from(claim.boundary_nonholdout_count).map_err(|_| invariant())?,
            claim.as_of_unix_ms,
            i64::from(claim.treatment_denominator),
            i64::from(commit.audit.treatment.labeled),
            i64::from(claim.control_denominator),
            i64::from(commit.audit.control.labeled),
            claim.input_aggregate_hash,
            result_state_str,
            process_instance_id.to_string(),
            observed_at_unix_ms,
            look_hash,
        ],
    )?;
    insert_look_audit(
        transaction,
        commit.active_outcome_look_id,
        &commit.audit,
        &canonical_audit_json,
        &audit_hash,
    )?;
    insert_delta_members(
        transaction,
        commit.active_outcome_look_id,
        claim.active_experiment_id,
        previous_boundary,
        &claim.members,
    )?;
    insert_claim_state(
        transaction,
        commit.claim_terminal_state_event_id,
        commit.active_look_claim_id,
        claim.active_experiment_id,
        "committed",
        None,
        None,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    insert_authorization_state(
        transaction,
        commit.authorization_state_event_id,
        claim.active_experiment_id,
        Some(claim.evaluating_authorization_state_event_id),
        Some(commit.active_outcome_look_id),
        None,
        result_state_str,
        valid_until_unix_ms,
        policy.create.control_generation,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    let experiment_terminal = finish_boundary(
        transaction,
        process_instance_id,
        &policy.create,
        claim.boundary_tranche_ordinal,
        result_state_str,
        observed_at_unix_ms,
    )?;
    Ok(ActiveLookCommitAck::Applied(ActiveLookCommitReceipt {
        active_outcome_look_id: commit.active_outcome_look_id,
        authorization_state_event_id: commit.authorization_state_event_id,
        result_state,
        valid_until_unix_ms,
        experiment_terminal,
    }))
}

fn validate_audit_for_claim(
    claim: &ActiveLookClaimReceipt,
    audit: &ActiveLookAuditV1,
) -> Result<(), LedgerError> {
    let members = claim
        .members
        .iter()
        .copied()
        .map(ActiveFrozenLookMember::math_member)
        .collect::<Vec<_>>();
    let input_identity = active_look_input_identity_v1(claim.policy, claim.as_of_unix_ms, &members)
        .map_err(|_| invariant())?;
    if audit.shape_version != 1
        || audit.active_math_algorithm_id_sha256 != active_math_algorithm_identity_v1()
        || audit.input_identity_sha256 != input_identity
        || audit.as_of_unix_ms != claim.as_of_unix_ms
        || audit.treatment.denominator != claim.treatment_denominator
        || audit.control.denominator != claim.control_denominator
        || audit.policy.actual_outcome_half_life_seconds
            != claim.policy.actual_outcome_half_life_seconds()
        || audit
            .policy
            .per_look_noninferiority_threshold
            .value()
            .to_bits()
            != claim.policy.per_look_noninferiority_threshold().to_bits()
    {
        return Err(invariant());
    }
    Ok(())
}

const fn look_state_str(state: LookProducedStateV1) -> &'static str {
    match state {
        LookProducedStateV1::Collecting => "collecting",
        LookProducedStateV1::Passed => "passed",
        LookProducedStateV1::Rollback => "rollback",
    }
}

fn look_payload_hash(
    active_outcome_look_id: Uuid,
    claim: &ActiveLookClaimReceipt,
    result_state: LookProducedStateV1,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
    audit_hash: &str,
) -> Result<String, LedgerError> {
    hash_json(json!({
        "shape": "active_outcome_look_v1",
        "active_outcome_look_id": active_outcome_look_id,
        "active_experiment_id": claim.active_experiment_id,
        "active_look_claim_id": claim.active_look_claim_id,
        "look_ordinal": claim.expected_look_ordinal.to_string(),
        "boundary_tranche_ordinal": claim.boundary_tranche_ordinal.to_string(),
        "boundary_nonholdout_count": claim.boundary_nonholdout_count.to_string(),
        "as_of_unix_ms": claim.as_of_unix_ms.to_string(),
        "treatment_denominator": claim.treatment_denominator.to_string(),
        "treatment_labeled": claim.members.iter().filter(|member| {
            member.arm == ActiveOutcomeArmV1::Treatment && member.label.is_some()
        }).count().to_string(),
        "control_denominator": claim.control_denominator.to_string(),
        "control_labeled": claim.members.iter().filter(|member| {
            member.arm == ActiveOutcomeArmV1::Control && member.label.is_some()
        }).count().to_string(),
        "input_aggregate_hash": claim.input_aggregate_hash,
        "result_state": look_state_str(result_state),
        "audit_hash": audit_hash,
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": created_at_unix_ms.to_string(),
    }))
}

fn insert_look_audit(
    connection: &Connection,
    active_outcome_look_id: Uuid,
    audit: &ActiveLookAuditV1,
    canonical_audit_json: &str,
    canonical_payload_hash: &str,
) -> Result<(), LedgerError> {
    let changed = connection
        .execute(
            "INSERT INTO active_outcome_look_audits (
                active_outcome_look_id, audit_shape_version,
                active_math_build_id, active_math_algorithm_id_sha256,
                input_identity_sha256, actual_outcome_half_life_seconds,
                min_treatment_roots, min_control_roots,
                min_treatment_effective_weight, min_treatment_effective_weight_bits,
                min_control_effective_weight, min_control_effective_weight_bits,
                noninferiority_margin, noninferiority_margin_bits,
                noninferiority_probability, noninferiority_probability_bits,
                per_look_noninferiority_threshold,
                per_look_noninferiority_threshold_bits,
                rollback_probability, rollback_probability_bits, max_looks,
                treatment_denominator, treatment_labeled,
                treatment_successes, treatment_failures,
                treatment_success_weight, treatment_success_weight_bits,
                treatment_failure_weight, treatment_failure_weight_bits,
                treatment_effective_weight, treatment_effective_weight_bits,
                treatment_beta_alpha, treatment_beta_alpha_bits,
                treatment_beta_beta, treatment_beta_beta_bits,
                treatment_label_rate, treatment_label_rate_bits,
                control_denominator, control_labeled,
                control_successes, control_failures,
                control_success_weight, control_success_weight_bits,
                control_failure_weight, control_failure_weight_bits,
                control_effective_weight, control_effective_weight_bits,
                control_beta_alpha, control_beta_alpha_bits,
                control_beta_beta, control_beta_beta_bits,
                control_label_rate, control_label_rate_bits,
                noninferiority_g_lower, noninferiority_g_lower_bits,
                noninferiority_g_upper, noninferiority_g_upper_bits,
                noninferiority_h_lower, noninferiority_h_lower_bits,
                noninferiority_h_upper, noninferiority_h_upper_bits,
                noninferiority_lower, noninferiority_lower_bits,
                noninferiority_upper, noninferiority_upper_bits,
                rollback_lower, rollback_lower_bits,
                treatment_raw_roots_gate, control_raw_roots_gate,
                treatment_effective_weight_gate, control_effective_weight_gate,
                treatment_attribution_gate, control_attribution_gate,
                differential_attribution_gate, noninferiority_gate,
                rollback_gate, result_state, raw_beta_cdf_calls,
                operational_cdf_queries, max_symmetry_disagreement,
                max_symmetry_disagreement_bits, max_monotonic_repair,
                max_monotonic_repair_bits, max_quantile_width,
                max_quantile_width_bits, cdf_transcript_sha256,
                canonical_audit_json, canonical_payload_hash
             ) VALUES (
                :look_id, :shape, :build, :algorithm, :input,
                :half_life, :min_treatment, :min_control,
                :min_treatment_weight, :min_treatment_weight_bits,
                :min_control_weight, :min_control_weight_bits,
                :margin, :margin_bits, :noninferiority_probability,
                :noninferiority_probability_bits, :per_look, :per_look_bits,
                :rollback_probability, :rollback_probability_bits, :max_looks,
                :td, :tl, :ts, :tf, :tsw, :tswb, :tfw, :tfwb,
                :tew, :tewb, :ta, :tab, :tb, :tbb, :tr, :trb,
                :cd, :cl, :cs, :cf, :csw, :cswb, :cfw, :cfwb,
                :cew, :cewb, :ca, :cab, :cb, :cbb, :cr, :crb,
                :gl, :glb, :gu, :gub, :hl, :hlb, :hu, :hub,
                :nl, :nlb, :nu, :nub, :rl, :rlb,
                :trg, :crg, :tewg, :cewg, :tag, :cag, :dag, :nig, :rbg,
                :result, :raw_calls, :queries, :disagreement, :disagreement_bits,
                :repair, :repair_bits, :width, :width_bits, :transcript,
                :audit_json, :payload_hash
             )",
            named_params! {
                ":look_id": active_outcome_look_id.to_string(),
                ":shape": i64::from(audit.shape_version),
                ":build": audit.active_math_build_id,
                ":algorithm": audit.active_math_algorithm_id_sha256,
                ":input": audit.input_identity_sha256,
                ":half_life": i64::from(audit.policy.actual_outcome_half_life_seconds),
                ":min_treatment": i64::from(audit.policy.min_treatment_roots),
                ":min_control": i64::from(audit.policy.min_control_roots),
                ":min_treatment_weight": audit.policy.min_treatment_effective_weight.value(),
                ":min_treatment_weight_bits": audit.policy.min_treatment_effective_weight.bits() as i64,
                ":min_control_weight": audit.policy.min_control_effective_weight.value(),
                ":min_control_weight_bits": audit.policy.min_control_effective_weight.bits() as i64,
                ":margin": audit.policy.noninferiority_margin.value(),
                ":margin_bits": audit.policy.noninferiority_margin.bits() as i64,
                ":noninferiority_probability": audit.policy.noninferiority_probability.value(),
                ":noninferiority_probability_bits": audit.policy.noninferiority_probability.bits() as i64,
                ":per_look": audit.policy.per_look_noninferiority_threshold.value(),
                ":per_look_bits": audit.policy.per_look_noninferiority_threshold.bits() as i64,
                ":rollback_probability": audit.policy.rollback_probability.value(),
                ":rollback_probability_bits": audit.policy.rollback_probability.bits() as i64,
                ":max_looks": i64::from(audit.policy.max_looks),
                ":td": i64::from(audit.treatment.denominator),
                ":tl": i64::from(audit.treatment.labeled),
                ":ts": i64::from(audit.treatment.successes),
                ":tf": i64::from(audit.treatment.failures),
                ":tsw": audit.treatment.success_weight.value(),
                ":tswb": audit.treatment.success_weight.bits() as i64,
                ":tfw": audit.treatment.failure_weight.value(),
                ":tfwb": audit.treatment.failure_weight.bits() as i64,
                ":tew": audit.treatment.effective_weight.value(),
                ":tewb": audit.treatment.effective_weight.bits() as i64,
                ":ta": audit.treatment.beta_alpha.value(),
                ":tab": audit.treatment.beta_alpha.bits() as i64,
                ":tb": audit.treatment.beta_beta.value(),
                ":tbb": audit.treatment.beta_beta.bits() as i64,
                ":tr": audit.treatment.label_rate.value(),
                ":trb": audit.treatment.label_rate.bits() as i64,
                ":cd": i64::from(audit.control.denominator),
                ":cl": i64::from(audit.control.labeled),
                ":cs": i64::from(audit.control.successes),
                ":cf": i64::from(audit.control.failures),
                ":csw": audit.control.success_weight.value(),
                ":cswb": audit.control.success_weight.bits() as i64,
                ":cfw": audit.control.failure_weight.value(),
                ":cfwb": audit.control.failure_weight.bits() as i64,
                ":cew": audit.control.effective_weight.value(),
                ":cewb": audit.control.effective_weight.bits() as i64,
                ":ca": audit.control.beta_alpha.value(),
                ":cab": audit.control.beta_alpha.bits() as i64,
                ":cb": audit.control.beta_beta.value(),
                ":cbb": audit.control.beta_beta.bits() as i64,
                ":cr": audit.control.label_rate.value(),
                ":crb": audit.control.label_rate.bits() as i64,
                ":gl": audit.noninferiority_g.lower.value(),
                ":glb": audit.noninferiority_g.lower.bits() as i64,
                ":gu": audit.noninferiority_g.upper.value(),
                ":gub": audit.noninferiority_g.upper.bits() as i64,
                ":hl": audit.noninferiority_h.lower.value(),
                ":hlb": audit.noninferiority_h.lower.bits() as i64,
                ":hu": audit.noninferiority_h.upper.value(),
                ":hub": audit.noninferiority_h.upper.bits() as i64,
                ":nl": audit.noninferiority.lower.value(),
                ":nlb": audit.noninferiority.lower.bits() as i64,
                ":nu": audit.noninferiority.upper.value(),
                ":nub": audit.noninferiority.upper.bits() as i64,
                ":rl": audit.rollback_lower.value(),
                ":rlb": audit.rollback_lower.bits() as i64,
                ":trg": i64::from(audit.gates.treatment_raw_roots),
                ":crg": i64::from(audit.gates.control_raw_roots),
                ":tewg": i64::from(audit.gates.treatment_effective_weight),
                ":cewg": i64::from(audit.gates.control_effective_weight),
                ":tag": i64::from(audit.gates.treatment_attribution),
                ":cag": i64::from(audit.gates.control_attribution),
                ":dag": i64::from(audit.gates.differential_attribution),
                ":nig": i64::from(audit.gates.noninferiority),
                ":rbg": i64::from(audit.gates.rollback),
                ":result": look_state_str(audit.state),
                ":raw_calls": i64::from(audit.raw_beta_cdf_calls),
                ":queries": i64::from(audit.operational_cdf_queries),
                ":disagreement": audit.max_symmetry_disagreement.value(),
                ":disagreement_bits": audit.max_symmetry_disagreement.bits() as i64,
                ":repair": audit.max_monotonic_repair.value(),
                ":repair_bits": audit.max_monotonic_repair.bits() as i64,
                ":width": audit.max_quantile_width.value(),
                ":width_bits": audit.max_quantile_width.bits() as i64,
                ":transcript": audit.cdf_transcript_sha256,
                ":audit_json": canonical_audit_json,
                ":payload_hash": canonical_payload_hash,
            },
        )
        .map_err(database_error)?;
    if changed != 1 {
        return Err(corrupt());
    }
    Ok(())
}

fn insert_delta_members(
    connection: &Connection,
    active_outcome_look_id: Uuid,
    active_experiment_id: Uuid,
    previous_boundary: u64,
    members: &[ActiveFrozenLookMember],
) -> Result<(), LedgerError> {
    let mut delta_ordinal = 0_u64;
    for member in members
        .iter()
        .filter(|member| member.cap_ordinal > previous_boundary)
    {
        delta_ordinal = delta_ordinal.checked_add(1).ok_or_else(corrupt)?;
        let arm = match member.arm {
            ActiveOutcomeArmV1::Treatment => "candidate_treatment",
            ActiveOutcomeArmV1::Control => "anchor_control",
        };
        let label = match member.label {
            Some(ActiveOutcomeLabelV1::Success) => Some("success"),
            Some(ActiveOutcomeLabelV1::Failure) => Some("failure"),
            None => None,
        };
        let payload_hash = hash_json(json!({
            "shape": "active_look_member_v1",
            "active_outcome_look_id": active_outcome_look_id,
            "delta_member_ordinal": delta_ordinal.to_string(),
            "active_experiment_id": active_experiment_id,
            "active_assignment_id": member.active_assignment_id,
            "outcome_id": member.outcome_id,
            "cap_ordinal": member.cap_ordinal.to_string(),
            "arm": arm,
            "label": label,
            "inclusion_status": member.inclusion_status.as_str(),
        }))?;
        execute_one(
            connection,
            "INSERT INTO active_look_members (
                active_outcome_look_id, delta_member_ordinal,
                active_experiment_id, active_assignment_id, outcome_id,
                cap_ordinal, arm, label, inclusion_status,
                canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                active_outcome_look_id.to_string(),
                i64::try_from(delta_ordinal).map_err(|_| corrupt())?,
                active_experiment_id.to_string(),
                member.active_assignment_id.to_string(),
                member.outcome_id.to_string(),
                i64::try_from(member.cap_ordinal).map_err(|_| corrupt())?,
                arm,
                label,
                member.inclusion_status.as_str(),
                payload_hash,
            ],
        )?;
    }
    Ok(())
}

fn load_existing_look_commit(
    connection: &Connection,
    commit: &ActiveLookCommit,
) -> Result<Option<ActiveLookCommitAck>, LedgerError> {
    let row = connection
        .query_row(
            "SELECT look.active_outcome_look_id, look.result_state,
                    audit.canonical_audit_json, audit.canonical_payload_hash,
                    authorization.active_authorization_state_event_id,
                    authorization.valid_until_unix_ms,
                    claim_state.active_look_claim_state_event_id,
                    experiment_state.state, experiment_state.terminal_reason
             FROM active_outcome_looks AS look
             JOIN active_outcome_look_audits AS audit
               ON audit.active_outcome_look_id = look.active_outcome_look_id
             JOIN active_authorization_state_events AS authorization
               ON authorization.active_outcome_look_id = look.active_outcome_look_id
             JOIN active_look_claim_state_events AS claim_state
               ON claim_state.active_look_claim_id = look.active_look_claim_id
              AND claim_state.state = 'committed'
             JOIN active_experiment_state_events AS experiment_state
               ON experiment_state.active_experiment_id = look.active_experiment_id
              AND experiment_state.event_seq = (
                    SELECT max(latest.event_seq)
                    FROM active_experiment_state_events AS latest
                    WHERE latest.active_experiment_id = look.active_experiment_id
               )
             WHERE look.active_look_claim_id = ?1",
            [commit.active_look_claim_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, Option<i64>>(5)?,
                    row.get::<_, String>(6)?,
                    row.get::<_, String>(7)?,
                    row.get::<_, Option<String>>(8)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(row) = row else {
        let look_collision: bool = connection
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM active_outcome_looks
                    WHERE active_outcome_look_id = ?1
                 )",
                [commit.active_outcome_look_id.to_string()],
                |row| row.get(0),
            )
            .map_err(database_error)?;
        return Ok(look_collision.then_some(ActiveLookCommitAck::Conflict));
    };
    let stored_value: Json = serde_json::from_str(&row.2).map_err(|_| corrupt())?;
    if canonical_json(&stored_value).map_err(|_| corrupt())? != row.2
        || canonical_sha256(&stored_value).map_err(|_| corrupt())? != row.3
    {
        return Err(corrupt());
    }
    let requested_value = serde_json::to_value(commit.audit.as_ref())
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let requested_json = canonical_json(&requested_value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let state = match row.1.as_str() {
        "collecting" => LookProducedStateV1::Collecting,
        "passed" => LookProducedStateV1::Passed,
        "rollback" => LookProducedStateV1::Rollback,
        _ => return Err(corrupt()),
    };
    if row.0 != commit.active_outcome_look_id.to_string()
        || row.2 != requested_json
        || row.4 != commit.authorization_state_event_id.to_string()
        || row.6 != commit.claim_terminal_state_event_id.to_string()
        || state != commit.audit.state
    {
        return Ok(Some(ActiveLookCommitAck::Conflict));
    }
    let valid_until_unix_ms = row
        .5
        .map(u64::try_from)
        .transpose()
        .map_err(|_| corrupt())?;
    let experiment_terminal = row.7 == "terminal";
    if experiment_terminal
        && !matches!(
            row.8.as_deref(),
            Some("closed_passed" | "exhausted" | "rollback")
        )
    {
        return Err(corrupt());
    }
    Ok(Some(ActiveLookCommitAck::AlreadyApplied(
        ActiveLookCommitReceipt {
            active_outcome_look_id: commit.active_outcome_look_id,
            authorization_state_event_id: commit.authorization_state_event_id,
            result_state: state,
            valid_until_unix_ms,
            experiment_terminal,
        },
    )))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveLookFailureKind {
    NumericFailure,
    IntegrityFailure,
}

impl ActiveLookFailureKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::NumericFailure => "numeric_failure",
            Self::IntegrityFailure => "integrity_failure",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveLookFailure {
    pub(crate) active_look_failure_id: Uuid,
    pub(crate) active_look_claim_id: Uuid,
    pub(crate) claim_terminal_state_event_id: Uuid,
    pub(crate) authorization_state_event_id: Uuid,
    pub(crate) lease_token_hash: String,
    pub(crate) failure_kind: ActiveLookFailureKind,
    pub(crate) stable_reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveLookFailureAck {
    Applied,
    AlreadyApplied,
    LeaseLost,
    AuthorityChanged,
    Conflict,
    TransactionNotStarted,
}

impl LedgerRepository {
    pub(crate) fn fail_active_look(
        &mut self,
        failure: &ActiveLookFailure,
    ) -> Result<ActiveLookFailureAck, LedgerError> {
        self.fail_active_look_with_start_check(failure, || Some(()))
    }

    pub(crate) fn fail_active_look_with_start_check<G: TransactionStartGuard>(
        &mut self,
        failure: &ActiveLookFailure,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveLookFailureAck, LedgerError> {
        validate_look_failure(failure)?;
        let database_path = self.database_path.clone();
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveLookFailureAck::TransactionNotStarted);
        };
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| {
                super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
            })?;
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ActiveLookFailureAck::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = fail_active_look_in_transaction(
            &transaction,
            self.project_uuid,
            self.process_instance_id,
            failure,
            now_unix_ms()?,
        )?;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }
}

fn validate_look_failure(failure: &ActiveLookFailure) -> Result<(), LedgerError> {
    if !is_uuid_v7(failure.active_look_failure_id)
        || !is_uuid_v7(failure.active_look_claim_id)
        || !is_uuid_v7(failure.claim_terminal_state_event_id)
        || !is_uuid_v7(failure.authorization_state_event_id)
        || !is_hash(&failure.lease_token_hash)
        || !valid_stable_reason(&failure.stable_reason)
    {
        return Err(invariant());
    }
    Ok(())
}

fn fail_active_look_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    failure: &ActiveLookFailure,
    observed_at_unix_ms: i64,
) -> Result<ActiveLookFailureAck, LedgerError> {
    validate_look_failure(failure)?;
    let replay = transaction
        .query_row(
            "SELECT active_look_claim_id, failure_kind, stable_reason
             FROM active_look_failures WHERE active_look_failure_id = ?1",
            [failure.active_look_failure_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, Option<String>>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    if let Some(replay) = replay {
        let active_look_claim_id = failure.active_look_claim_id.to_string();
        return Ok(
            if replay.0.as_deref() == Some(active_look_claim_id.as_str())
                && replay.1 == failure.failure_kind.as_str()
                && replay.2 == failure.stable_reason
            {
                ActiveLookFailureAck::AlreadyApplied
            } else {
                ActiveLookFailureAck::Conflict
            },
        );
    }
    let Some(claim) = load_claim_by_id(
        transaction,
        project_uuid,
        process_instance_id,
        failure.active_look_claim_id,
    )?
    else {
        return Ok(ActiveLookFailureAck::LeaseLost);
    };
    let policy = load_experiment_policy(transaction, project_uuid, claim.active_experiment_id)?
        .ok_or_else(corrupt)?;
    if !claim_authority_matches(transaction, project_uuid, process_instance_id, &policy)? {
        return Ok(ActiveLookFailureAck::AuthorityChanged);
    }
    let latest_owner = transaction
        .query_row(
            "SELECT process_instance_id FROM active_look_claim_state_events
             WHERE active_look_claim_id = ?1 ORDER BY event_seq DESC LIMIT 1",
            [failure.active_look_claim_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?;
    if claim.lease_token_hash != failure.lease_token_hash
        || claim.lease_expires_at_unix_ms <= observed_at_unix_ms
        || latest_owner != process_instance_id.to_string()
    {
        return Ok(ActiveLookFailureAck::LeaseLost);
    }
    let failure_hash = hash_json(json!({
        "shape": "active_look_failure_v1",
        "active_look_failure_id": failure.active_look_failure_id,
        "active_experiment_id": claim.active_experiment_id,
        "active_look_claim_id": failure.active_look_claim_id,
        "boundary_tranche_ordinal": claim.boundary_tranche_ordinal.to_string(),
        "boundary_nonholdout_count": claim.boundary_nonholdout_count.to_string(),
        "failure_kind": failure.failure_kind.as_str(),
        "stable_reason": failure.stable_reason,
        "input_aggregate_hash": claim.input_aggregate_hash,
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": observed_at_unix_ms.to_string(),
    }))?;
    execute_one(
        transaction,
        "INSERT INTO active_look_failures (
            active_look_failure_id, active_experiment_id, active_look_claim_id,
            boundary_tranche_ordinal, boundary_nonholdout_count, failure_kind,
            stable_reason, input_aggregate_hash, process_instance_id,
            created_at_unix_ms, canonical_payload_hash
         ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            failure.active_look_failure_id.to_string(),
            claim.active_experiment_id.to_string(),
            failure.active_look_claim_id.to_string(),
            i64::try_from(claim.boundary_tranche_ordinal).map_err(|_| invariant())?,
            i64::try_from(claim.boundary_nonholdout_count).map_err(|_| invariant())?,
            failure.failure_kind.as_str(),
            failure.stable_reason,
            claim.input_aggregate_hash,
            process_instance_id.to_string(),
            observed_at_unix_ms,
            failure_hash,
        ],
    )?;
    insert_claim_state(
        transaction,
        failure.claim_terminal_state_event_id,
        failure.active_look_claim_id,
        claim.active_experiment_id,
        "failed",
        None,
        None,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    insert_authorization_state(
        transaction,
        failure.authorization_state_event_id,
        claim.active_experiment_id,
        Some(claim.evaluating_authorization_state_event_id),
        None,
        None,
        "invalidated",
        None,
        policy.create.control_generation,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    let counts = tranche_counts(
        transaction,
        claim.active_experiment_id,
        claim.boundary_tranche_ordinal,
    )?;
    append_tranche_state_if_new(
        transaction,
        claim.active_experiment_id,
        claim.boundary_tranche_ordinal,
        "complete",
        counts,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    append_experiment_state_if_new(
        transaction,
        claim.active_experiment_id,
        "terminal",
        Some("invalidated"),
        process_instance_id,
        observed_at_unix_ms,
    )?;
    Ok(ActiveLookFailureAck::Applied)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveAuthorizationState {
    Collecting,
    Evaluating,
    Passed,
    Rollback,
    Expired,
    Invalidated,
    SupersededDraining,
    Superseded,
    ClosedPassed,
    Exhausted,
}

impl ActiveAuthorizationState {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Collecting => "collecting",
            Self::Evaluating => "evaluating",
            Self::Passed => "passed",
            Self::Rollback => "rollback",
            Self::Expired => "expired",
            Self::Invalidated => "invalidated",
            Self::SupersededDraining => "superseded_draining",
            Self::Superseded => "superseded",
            Self::ClosedPassed => "closed_passed",
            Self::Exhausted => "exhausted",
        }
    }

    fn parse(value: &str) -> Result<Self, LedgerError> {
        match value {
            "collecting" => Ok(Self::Collecting),
            "evaluating" => Ok(Self::Evaluating),
            "passed" => Ok(Self::Passed),
            "rollback" => Ok(Self::Rollback),
            "expired" => Ok(Self::Expired),
            "invalidated" => Ok(Self::Invalidated),
            "superseded_draining" => Ok(Self::SupersededDraining),
            "superseded" => Ok(Self::Superseded),
            "closed_passed" => Ok(Self::ClosedPassed),
            "exhausted" => Ok(Self::Exhausted),
            _ => Err(corrupt()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveAuthorizationSnapshot {
    pub(crate) active_experiment_id: Uuid,
    pub(crate) authorization_state_event_id: Uuid,
    pub(crate) active_outcome_look_id: Option<Uuid>,
    pub(crate) open_tranche_ordinal: Option<u64>,
    pub(crate) noninferiority_lower_bits: Option<u64>,
    pub(crate) noninferiority_upper_bits: Option<u64>,
    pub(crate) state: ActiveAuthorizationState,
    pub(crate) valid_until_unix_ms: Option<u64>,
    pub(crate) control_generation: u64,
    pub(crate) authorizing: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveAuthorizationObserveAck {
    Current(ActiveAuthorizationSnapshot),
    Unavailable,
    TransactionNotStarted,
}

impl LedgerRepository {
    pub(crate) fn observe_active_authorization(
        &mut self,
        active_experiment_id: Uuid,
    ) -> Result<ActiveAuthorizationObserveAck, LedgerError> {
        self.observe_active_authorization_at(active_experiment_id, now_unix_ms()?)
    }

    pub(crate) fn observe_active_authorization_at(
        &mut self,
        active_experiment_id: Uuid,
        observed_at_unix_ms: i64,
    ) -> Result<ActiveAuthorizationObserveAck, LedgerError> {
        self.observe_active_authorization_with_start_check(
            active_experiment_id,
            observed_at_unix_ms,
            || Some(()),
        )
    }

    pub(crate) fn observe_active_authorization_with_start_check<G: TransactionStartGuard>(
        &mut self,
        active_experiment_id: Uuid,
        observed_at_unix_ms: i64,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveAuthorizationObserveAck, LedgerError> {
        if !is_uuid_v7(active_experiment_id) || observed_at_unix_ms < 0 {
            return Err(invariant());
        }
        let database_path = self.database_path.clone();
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveAuthorizationObserveAck::TransactionNotStarted);
        };
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| {
                super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
            })?;
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ActiveAuthorizationObserveAck::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = observe_active_authorization_in_transaction(
            &transaction,
            self.project_uuid,
            self.process_instance_id,
            active_experiment_id,
            observed_at_unix_ms,
        )?;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }
}

fn observe_active_authorization_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    active_experiment_id: Uuid,
    observed_at_unix_ms: i64,
) -> Result<ActiveAuthorizationObserveAck, LedgerError> {
    if !super::process::originating_process_is_live(transaction, project_uuid, process_instance_id)?
        || !super::active::all_live_processes_support_active_v6(transaction, project_uuid)?
    {
        return Ok(ActiveAuthorizationObserveAck::Unavailable);
    }
    let Some(policy) = load_experiment_policy(transaction, project_uuid, active_experiment_id)?
    else {
        return Ok(ActiveAuthorizationObserveAck::Unavailable);
    };
    let retiring: bool = transaction
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM active_retirement_markers
                WHERE active_experiment_id = ?1
             )",
            [active_experiment_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    if retiring {
        return Ok(ActiveAuthorizationObserveAck::Unavailable);
    }
    let generations_current =
        experiment_generations_are_current(transaction, project_uuid, &policy.create)?;
    if !generations_current {
        observe_supersession(
            transaction,
            process_instance_id,
            &policy.create,
            observed_at_unix_ms,
        )?;
    }
    let mut latest =
        latest_authorization(transaction, active_experiment_id)?.ok_or_else(corrupt)?;
    if generations_current
        && latest.state == "passed"
        && latest
            .valid_until_unix_ms
            .is_some_and(|valid_until| observed_at_unix_ms >= valid_until)
    {
        insert_authorization_state(
            transaction,
            Uuid::now_v7(),
            active_experiment_id,
            Some(latest.event_id),
            None,
            None,
            "expired",
            None,
            policy.create.control_generation,
            process_instance_id,
            observed_at_unix_ms,
        )?;
        close_open_tranches(
            transaction,
            active_experiment_id,
            process_instance_id,
            observed_at_unix_ms,
        )?;
        latest = latest_authorization(transaction, active_experiment_id)?.ok_or_else(corrupt)?;
    }
    let current_control = current_control_state(transaction, project_uuid, &policy.create.pool_id)?;
    let state = ActiveAuthorizationState::parse(&latest.state)?;
    let open_tranche_ordinal = transaction
        .query_row(
            "SELECT tranche.tranche_ordinal
             FROM active_experiment_tranches AS tranche
             JOIN active_tranche_state_events AS state
               ON state.active_experiment_id = tranche.active_experiment_id
              AND state.tranche_ordinal = tranche.tranche_ordinal
             WHERE tranche.active_experiment_id = ?1
               AND state.event_seq = (
                   SELECT max(latest.event_seq)
                   FROM active_tranche_state_events AS latest
                   WHERE latest.active_experiment_id = tranche.active_experiment_id
                     AND latest.tranche_ordinal = tranche.tranche_ordinal
               )
               AND state.state = 'open'
             ORDER BY tranche.tranche_ordinal
             LIMIT 1",
            [active_experiment_id.to_string()],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?
        .map(u64::try_from)
        .transpose()
        .map_err(|_| corrupt())?;
    let (noninferiority_lower_bits, noninferiority_upper_bits) = match latest.active_outcome_look_id
    {
        Some(look_id) => {
            let interval = transaction
                .query_row(
                    "SELECT audit.noninferiority_lower_bits,
                            audit.noninferiority_upper_bits
                     FROM active_outcome_looks AS look
                     JOIN active_outcome_look_audits AS audit
                       ON audit.active_outcome_look_id = look.active_outcome_look_id
                     WHERE look.active_outcome_look_id = ?1
                       AND look.active_experiment_id = ?2",
                    params![look_id.to_string(), active_experiment_id.to_string()],
                    |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
                )
                .optional()
                .map_err(database_error)?
                .ok_or_else(corrupt)?;
            (Some(interval.0 as u64), Some(interval.1 as u64))
        }
        None => (None, None),
    };
    let valid_until_unix_ms = latest
        .valid_until_unix_ms
        .map(u64::try_from)
        .transpose()
        .map_err(|_| corrupt())?;
    let authorizing = generations_current
        && state == ActiveAuthorizationState::Passed
        && valid_until_unix_ms.is_some_and(|valid_until| {
            u64::try_from(observed_at_unix_ms).is_ok_and(|observed| observed < valid_until)
        })
        && current_control.generation == policy.create.control_generation
        && !current_control.paused
        && !current_control.force_anchor;
    Ok(ActiveAuthorizationObserveAck::Current(
        ActiveAuthorizationSnapshot {
            active_experiment_id,
            authorization_state_event_id: latest.event_id,
            active_outcome_look_id: latest.active_outcome_look_id,
            open_tranche_ordinal,
            noninferiority_lower_bits,
            noninferiority_upper_bits,
            state,
            valid_until_unix_ms,
            control_generation: current_control.generation,
            authorizing,
        },
    ))
}

fn experiment_generations_are_current(
    connection: &Connection,
    project_uuid: Uuid,
    create: &ActiveExperimentCreate,
) -> Result<bool, LedgerError> {
    let current_config = connection
        .query_row(
            "SELECT config_generation_id FROM config_generation_state_events
             WHERE project_uuid = ?1 ORDER BY config_epoch DESC LIMIT 1",
            [project_uuid.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let current_learning = connection
        .query_row(
            "SELECT learning_generation_id FROM learning_generation_state_events
             WHERE project_uuid = ?1 AND pool_id = ?2
             ORDER BY event_seq DESC LIMIT 1",
            params![project_uuid.to_string(), create.pool_id],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let current_cohort = connection
        .query_row(
            "SELECT cohort_generation_id FROM cohort_generation_state_events
             WHERE project_uuid = ?1 ORDER BY event_seq DESC LIMIT 1",
            [project_uuid.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    let learning_generation_id = create.learning_generation_id.to_string();
    let cohort_generation_id = create.cohort_generation_id.to_string();
    Ok(
        current_config.as_deref() == Some(create.config_generation_id.as_str())
            && current_learning.as_deref() == Some(learning_generation_id.as_str())
            && current_cohort.as_deref() == Some(cohort_generation_id.as_str()),
    )
}

#[derive(Debug, Clone, Copy)]
struct EffectiveControl {
    generation: u64,
    paused: bool,
    force_anchor: bool,
}

fn current_control_state(
    connection: &Connection,
    project_uuid: Uuid,
    pool_id: &str,
) -> Result<EffectiveControl, LedgerError> {
    let row = connection
        .query_row(
            "WITH latest_all AS (
                SELECT paused, force_anchor FROM controls
                WHERE project_uuid = ?1 AND scope_kind = 'all'
                ORDER BY control_generation DESC LIMIT 1
             ), latest_pool AS (
                SELECT paused, force_anchor FROM controls
                WHERE project_uuid = ?1 AND scope_kind = 'pool' AND pool_id = ?2
                ORDER BY control_generation DESC LIMIT 1
             )
             SELECT (SELECT max(control_generation) FROM controls WHERE project_uuid = ?1),
                    max(coalesce((SELECT paused FROM latest_all), 0),
                        coalesce((SELECT paused FROM latest_pool), 0)),
                    max(coalesce((SELECT force_anchor FROM latest_all), 0),
                        coalesce((SELECT force_anchor FROM latest_pool), 0))",
            params![project_uuid.to_string(), pool_id],
            |row| {
                Ok((
                    row.get::<_, Option<i64>>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                ))
            },
        )
        .map_err(database_error)?;
    Ok(EffectiveControl {
        generation: row
            .0
            .and_then(|value| u64::try_from(value).ok())
            .ok_or_else(corrupt)?,
        paused: row.1 == Some(1),
        force_anchor: row.2 == Some(1),
    })
}

fn observe_supersession(
    connection: &Connection,
    process_instance_id: Uuid,
    create: &ActiveExperimentCreate,
    observed_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let latest =
        latest_authorization(connection, create.active_experiment_id)?.ok_or_else(corrupt)?;
    if matches!(
        latest.state.as_str(),
        "superseded" | "closed_passed" | "exhausted"
    ) {
        return Ok(());
    }
    cancel_active_claims(
        connection,
        create.active_experiment_id,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    if latest.state != "superseded_draining" {
        insert_authorization_state(
            connection,
            Uuid::now_v7(),
            create.active_experiment_id,
            Some(latest.event_id),
            None,
            None,
            "superseded_draining",
            None,
            create.control_generation,
            process_instance_id,
            observed_at_unix_ms,
        )?;
    }
    close_open_tranches(
        connection,
        create.active_experiment_id,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    let unresolved: i64 = connection
        .query_row(
            "SELECT count(*)
             FROM active_assignments AS assignment
             WHERE assignment.active_experiment_id = ?1
               AND NOT EXISTS (
                    SELECT 1 FROM active_root_window_state_events AS state
                    WHERE state.active_root_window_id = assignment.active_root_window_id
                      AND state.state <> 'open'
               )",
            [create.active_experiment_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    if unresolved == 0 {
        let predecessor =
            latest_authorization(connection, create.active_experiment_id)?.ok_or_else(corrupt)?;
        if predecessor.state != "superseded" {
            insert_authorization_state(
                connection,
                Uuid::now_v7(),
                create.active_experiment_id,
                Some(predecessor.event_id),
                None,
                None,
                "superseded",
                None,
                create.control_generation,
                process_instance_id,
                observed_at_unix_ms,
            )?;
        }
        append_experiment_state_if_new(
            connection,
            create.active_experiment_id,
            "terminal",
            Some("superseded"),
            process_instance_id,
            observed_at_unix_ms,
        )?;
    }
    Ok(())
}

fn cancel_active_claims(
    connection: &Connection,
    active_experiment_id: Uuid,
    process_instance_id: Uuid,
    observed_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let claims = connection
        .prepare(
            "SELECT claim.active_look_claim_id
             FROM active_look_claims AS claim
             JOIN active_look_claim_state_events AS state
               ON state.active_look_claim_id = claim.active_look_claim_id
             WHERE claim.active_experiment_id = ?1
               AND state.event_seq = (
                    SELECT max(latest.event_seq)
                    FROM active_look_claim_state_events AS latest
                    WHERE latest.active_look_claim_id = claim.active_look_claim_id
               )
               AND state.state IN ('claimed', 'renewed', 'reclaimed')",
        )
        .map_err(database_error)?
        .query_map([active_experiment_id.to_string()], |row| {
            row.get::<_, String>(0)
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    for claim in claims {
        insert_claim_state(
            connection,
            Uuid::now_v7(),
            parse_uuid_v7(&claim)?,
            active_experiment_id,
            "cancelled",
            None,
            None,
            process_instance_id,
            observed_at_unix_ms,
        )?;
    }
    Ok(())
}

pub(super) fn reconcile_stale_active_look_claims(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    observed_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let claims = connection
        .prepare(
            "SELECT claim.active_look_claim_id, claim.active_experiment_id,
                    state.process_instance_id
             FROM active_look_claims AS claim
             JOIN active_experiments AS experiment
               ON experiment.active_experiment_id = claim.active_experiment_id
             JOIN active_look_claim_state_events AS state
               ON state.active_look_claim_id = claim.active_look_claim_id
             WHERE experiment.project_uuid = ?1
               AND state.event_seq = (
                    SELECT max(latest.event_seq)
                    FROM active_look_claim_state_events AS latest
                    WHERE latest.active_look_claim_id = claim.active_look_claim_id
               )
               AND state.state IN ('claimed', 'renewed', 'reclaimed')
             ORDER BY claim.created_at_unix_ms, claim.active_look_claim_id",
        )
        .map_err(database_error)?
        .query_map([project_uuid.to_string()], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    for (claim_id, experiment_id, owner_process_id) in claims {
        let owner_process_id = parse_uuid_v7(&owner_process_id)?;
        if super::process::originating_process_is_live(connection, project_uuid, owner_process_id)?
        {
            continue;
        }
        let claim_id = parse_uuid_v7(&claim_id)?;
        let experiment_id = parse_uuid_v7(&experiment_id)?;
        let event_id = Uuid::now_v7();
        let lease_token_hash = hash_json(json!({
            "shape": "active_look_startup_reclaim_v1",
            "active_look_claim_id": claim_id,
            "active_experiment_id": experiment_id,
            "active_look_claim_state_event_id": event_id,
            "process_instance_id": process_instance_id,
            "observed_at_unix_ms": observed_at_unix_ms.to_string(),
        }))?;
        let lease_expires_at_unix_ms = observed_at_unix_ms.checked_add(1).ok_or_else(invariant)?;
        insert_claim_state(
            connection,
            event_id,
            claim_id,
            experiment_id,
            "reclaimed",
            Some(&lease_token_hash),
            Some(lease_expires_at_unix_ms),
            process_instance_id,
            observed_at_unix_ms,
        )?;
    }
    Ok(())
}

fn close_open_tranches(
    connection: &Connection,
    active_experiment_id: Uuid,
    process_instance_id: Uuid,
    observed_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let tranches = connection
        .prepare(
            "SELECT tranche.tranche_ordinal
             FROM active_experiment_tranches AS tranche
             JOIN active_tranche_state_events AS state
               ON state.active_experiment_id = tranche.active_experiment_id
              AND state.tranche_ordinal = tranche.tranche_ordinal
             WHERE tranche.active_experiment_id = ?1
               AND state.event_seq = (
                    SELECT max(latest.event_seq)
                    FROM active_tranche_state_events AS latest
                    WHERE latest.active_experiment_id = tranche.active_experiment_id
                      AND latest.tranche_ordinal = tranche.tranche_ordinal
               )
               AND state.state = 'open'",
        )
        .map_err(database_error)?
        .query_map([active_experiment_id.to_string()], |row| {
            row.get::<_, i64>(0)
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    for tranche in tranches {
        let tranche = u64::try_from(tranche).map_err(|_| corrupt())?;
        let counts = tranche_counts(connection, active_experiment_id, tranche)?;
        append_tranche_state_if_new(
            connection,
            active_experiment_id,
            tranche,
            "closed",
            counts,
            process_instance_id,
            observed_at_unix_ms,
        )?;
        if counts.2 == 0 {
            append_tranche_state_if_new(
                connection,
                active_experiment_id,
                tranche,
                "drained",
                counts,
                process_instance_id,
                observed_at_unix_ms,
            )?;
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::{Arc, Barrier};
    use std::time::Duration;

    use tempfile::tempdir;

    use super::super::active::{
        ActiveAdmissionAck, ActiveAssignmentArm, ActiveDispatchTerminal, ActiveDispatchTerminalAck,
        ActiveDispatchTerminalState, ActiveRepresentativeStatus, ActiveRootClosure,
        ActiveSignalBatch, ActiveSignalDisposition,
    };
    use super::*;
    use crate::active_math::{ActiveLookEvaluationV1, evaluate_active_look_v1};
    use crate::ledger::repository::process::ProcessStop;

    struct Fixture {
        _directory: tempfile::TempDir,
        config: RouterConfig,
        activated: super::super::ActivatedLedger,
        create: ActiveExperimentCreate,
    }

    fn fixture(project_id: &str) -> Fixture {
        let directory = tempdir().unwrap();
        let path = crate::ledger::repository::tests::database_path(&directory);
        let config = super::super::active::tests::active_config(&path, project_id);
        let activated = LedgerRepository::activate(&config).unwrap();
        let outcome_policy_hash = activated
            .repository
            .connection
            .query_row(
                "SELECT outcome_policy_hash FROM outcome_policy_versions
                 WHERE pool_id = 'pool-a'",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let create = ActiveExperimentCreate::from_config(
            &config,
            &activated.identity,
            "pool-a",
            "candidate-a",
            "11".repeat(32),
            outcome_policy_hash,
            0,
        )
        .unwrap();
        Fixture {
            _directory: directory,
            config,
            activated,
            create,
        }
    }

    fn claim_request(active_experiment_id: Uuid, seed: &str) -> ActiveLookClaimRequest {
        ActiveLookClaimRequest {
            active_look_claim_id: Uuid::now_v7(),
            claim_state_event_id: Uuid::now_v7(),
            evaluating_authorization_state_event_id: Uuid::now_v7(),
            skipped_failure_id: Uuid::now_v7(),
            active_experiment_id,
            lease_token_hash: hash_json(json!({"lease": seed})).unwrap(),
            lease_duration_millis: ACTIVE_LOOK_LEASE_MAX_MILLIS,
        }
    }

    fn fill_drained_boundary(
        fixture: &mut super::super::active::tests::Fixture,
        treatment_count: usize,
        control_count: usize,
    ) {
        fill_drained_boundary_with_labels(
            fixture,
            treatment_count,
            control_count,
            ActiveSignalDisposition::Success,
            ActiveSignalDisposition::Success,
        );
    }

    pub(crate) fn fill_drained_boundary_with_labels(
        fixture: &mut super::super::active::tests::Fixture,
        treatment_count: usize,
        control_count: usize,
        treatment_disposition: ActiveSignalDisposition,
        control_disposition: ActiveSignalDisposition,
    ) {
        assert_eq!(treatment_count + control_count, 64);
        let arms =
            std::iter::repeat_n(ActiveAssignmentArm::CandidateTreatment, treatment_count).chain(
                std::iter::repeat_n(ActiveAssignmentArm::AnchorControl, control_count),
            );
        for (index, arm) in arms.enumerate() {
            let disposition = if arm == ActiveAssignmentArm::CandidateTreatment {
                treatment_disposition
            } else {
                control_disposition
            };
            let admission = super::super::active::tests::new_admission(fixture, arm);
            let receipt = match fixture
                .activated
                .repository
                .admit_active_root(&admission)
                .unwrap()
            {
                ActiveAdmissionAck::Applied(receipt) => receipt,
                acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
            };
            fixture
                .activated
                .repository
                .append_active_signals(&ActiveSignalBatch {
                    active_root_window_id: admission.active_root_window_id,
                    signals: vec![super::super::active::tests::protected_signal(
                        index as u64,
                        disposition,
                        receipt.admitted_at_unix_ms,
                        "{}".into(),
                    )],
                })
                .unwrap();
            if let Some(dispatch) = admission.dispatch.as_ref() {
                assert_eq!(
                    fixture
                        .activated
                        .repository
                        .record_active_dispatch_terminal(&ActiveDispatchTerminal {
                            active_dispatch_terminal_event_id: Uuid::now_v7(),
                            active_dispatch_id: dispatch.active_dispatch_id,
                            active_assignment_id: admission.active_assignment_id,
                            terminal_state: ActiveDispatchTerminalState::Completed,
                            stable_error_class: None,
                            provider_receipt_hash: None,
                            handed_off_at_unix_ms: Some(receipt.admitted_at_unix_ms),
                        })
                        .unwrap(),
                    ActiveDispatchTerminalAck::Applied
                );
            }
            fixture
                .activated
                .repository
                .terminalize_active_root(&super::super::active::tests::terminal_for(
                    &admission,
                    receipt,
                    ActiveRootClosure::OwnerEnd,
                    true,
                    Some(ActiveRepresentativeStatus::Completed),
                ))
                .unwrap();
        }
        let state = fixture
            .activated
            .repository
            .connection
            .query_row(
                "SELECT state FROM active_tranche_state_events
                 WHERE active_experiment_id = ?1 AND tranche_ordinal = 1
                 ORDER BY event_seq DESC LIMIT 1",
                [fixture.admission.active_experiment_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        assert_eq!(state, "drained");
    }

    fn complete_passing_look(
        fixture: &mut super::super::active::tests::Fixture,
    ) -> ActiveLookCommitReceipt {
        fill_drained_boundary_with_labels(
            fixture,
            32,
            32,
            ActiveSignalDisposition::Success,
            ActiveSignalDisposition::Failure,
        );
        let request = claim_request(fixture.admission.active_experiment_id, "passing-look");
        let claim = match fixture
            .activated
            .repository
            .claim_next_active_look(&request)
            .unwrap()
        {
            ActiveLookClaimAck::Claimed(claim) => claim,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        let members = claim
            .members
            .iter()
            .copied()
            .map(ActiveFrozenLookMember::math_member)
            .collect::<Vec<_>>();
        let audit =
            match evaluate_active_look_v1(claim.policy, claim.as_of_unix_ms, &members).unwrap() {
                ActiveLookEvaluationV1::Completed(audit) => audit,
                evaluation => panic!("unexpected evaluation: {evaluation:?}"),
            };
        assert_eq!(audit.state, LookProducedStateV1::Passed);
        match fixture
            .activated
            .repository
            .commit_active_look(&ActiveLookCommit {
                active_look_claim_id: claim.active_look_claim_id,
                lease_token_hash: claim.lease_token_hash,
                active_outcome_look_id: Uuid::now_v7(),
                claim_terminal_state_event_id: Uuid::now_v7(),
                authorization_state_event_id: Uuid::now_v7(),
                audit,
            })
            .unwrap()
        {
            ActiveLookCommitAck::Applied(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        }
    }

    #[test]
    fn experiment_creation_reloads_exact_identity_and_rejects_policy_conflict() {
        let mut fixture = fixture("active-experiment-create");
        let receipt = match fixture
            .activated
            .repository
            .create_active_experiment(&fixture.create)
            .unwrap()
        {
            ActiveExperimentCreateAck::Applied(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        let mut equivalent = fixture.create.clone();
        equivalent.active_experiment_id = Uuid::now_v7();
        equivalent.initial_experiment_state_event_id = Uuid::now_v7();
        equivalent.initial_tranche_state_event_id = Uuid::now_v7();
        equivalent.initial_authorization_state_event_id = Uuid::now_v7();
        assert_eq!(
            fixture
                .activated
                .repository
                .create_active_experiment(&equivalent)
                .unwrap(),
            ActiveExperimentCreateAck::AlreadyApplied(receipt)
        );
        let mut conflicting = equivalent;
        conflicting.max_canary_roots = 128;
        conflicting.max_looks = 2;
        assert_eq!(
            fixture
                .activated
                .repository
                .create_active_experiment(&conflicting)
                .unwrap(),
            ActiveExperimentCreateAck::Conflict
        );
        let states = fixture
            .activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT count(*) FROM active_experiment_state_events),
                    (SELECT count(*) FROM active_experiment_tranches),
                    (SELECT count(*) FROM active_tranche_state_events),
                    (SELECT count(*) FROM active_authorization_state_events)",
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
        assert_eq!(states, (1, 1, 1, 1));
    }

    #[test]
    fn concurrent_experiment_creation_converges_on_one_durable_identity() {
        let Fixture {
            _directory,
            config,
            activated,
            create,
        } = fixture("active-experiment-race");
        let second = LedgerRepository::activate(&config).unwrap();
        let mut second_create = create.clone();
        second_create.active_experiment_id = Uuid::now_v7();
        second_create.initial_experiment_state_event_id = Uuid::now_v7();
        second_create.initial_tranche_state_event_id = Uuid::now_v7();
        second_create.initial_authorization_state_event_id = Uuid::now_v7();
        let barrier = Arc::new(Barrier::new(2));
        let first_barrier = barrier.clone();
        let mut first_repository = activated.repository;
        let first = std::thread::spawn(move || {
            first_barrier.wait();
            first_repository.create_active_experiment(&create).unwrap()
        });
        let second_barrier = barrier.clone();
        let mut second_repository = second.repository;
        let second = std::thread::spawn(move || {
            second_barrier.wait();
            second_repository
                .create_active_experiment(&second_create)
                .unwrap()
        });
        let acknowledgements = [first.join().unwrap(), second.join().unwrap()];
        assert_eq!(
            acknowledgements
                .iter()
                .filter(|ack| matches!(ack, ActiveExperimentCreateAck::Applied(_)))
                .count(),
            1
        );
        assert_eq!(
            acknowledgements
                .iter()
                .filter(|ack| matches!(ack, ActiveExperimentCreateAck::AlreadyApplied(_)))
                .count(),
            1
        );
    }

    #[test]
    fn drained_boundary_claim_is_renewable_busy_and_reclaimable_cross_process() {
        let mut fixture = super::super::active::tests::fixture();
        fill_drained_boundary(&mut fixture, 32, 32);
        let request = claim_request(fixture.admission.active_experiment_id, "first");
        let receipt = match fixture
            .activated
            .repository
            .claim_next_active_look(&request)
            .unwrap()
        {
            ActiveLookClaimAck::Claimed(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        assert_eq!(receipt.boundary_tranche_ordinal, 1);
        assert_eq!(receipt.boundary_nonholdout_count, 64);
        assert_eq!(receipt.expected_look_ordinal, 1);
        assert_eq!(receipt.treatment_denominator, 32);
        assert_eq!(receipt.control_denominator, 32);
        assert_eq!(receipt.members.len(), 64);
        assert!(matches!(
            fixture
                .activated
                .repository
                .claim_next_active_look(&request)
                .unwrap(),
            ActiveLookClaimAck::AlreadyOwned(replayed) if replayed == receipt
        ));
        let competing = claim_request(fixture.admission.active_experiment_id, "competing");
        assert!(matches!(
            fixture
                .activated
                .repository
                .claim_next_active_look(&competing)
                .unwrap(),
            ActiveLookClaimAck::Busy { .. }
        ));
        let renewal = ActiveLookLeaseRenewal {
            active_look_claim_id: receipt.active_look_claim_id,
            claim_state_event_id: Uuid::now_v7(),
            lease_token_hash: receipt.lease_token_hash.clone(),
            lease_duration_millis: ACTIVE_LOOK_LEASE_MAX_MILLIS,
        };
        let renewed_until = match fixture
            .activated
            .repository
            .renew_active_look_lease(&renewal)
            .unwrap()
        {
            ActiveLookLeaseAck::Renewed {
                lease_expires_at_unix_ms,
            } => lease_expires_at_unix_ms,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        assert_eq!(
            fixture
                .activated
                .repository
                .renew_active_look_lease(&renewal)
                .unwrap(),
            ActiveLookLeaseAck::AlreadyRenewed {
                lease_expires_at_unix_ms: renewed_until,
            }
        );

        let mut second = LedgerRepository::activate(&fixture.config).unwrap();
        let reclaim = claim_request(fixture.admission.active_experiment_id, "reclaimed");
        let transaction = second
            .repository
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .unwrap();
        let reclaimed = claim_next_active_look_in_transaction(
            &transaction,
            second.identity.project_uuid,
            second.identity.process_instance_id,
            &reclaim,
            renewed_until + 1,
        )
        .unwrap();
        transaction.commit().unwrap();
        assert!(matches!(
            reclaimed,
            ActiveLookClaimAck::Reclaimed(ref reclaimed)
                if reclaimed.active_look_claim_id == receipt.active_look_claim_id
                    && reclaimed.lease_token_hash == reclaim.lease_token_hash
        ));
        assert_eq!(
            fixture
                .activated
                .repository
                .renew_active_look_lease(&ActiveLookLeaseRenewal {
                    active_look_claim_id: receipt.active_look_claim_id,
                    claim_state_event_id: Uuid::now_v7(),
                    lease_token_hash: receipt.lease_token_hash,
                    lease_duration_millis: ACTIVE_LOOK_LEASE_MAX_MILLIS,
                })
                .unwrap(),
            ActiveLookLeaseAck::LeaseLost
        );
    }

    #[test]
    fn startup_makes_a_dead_process_look_claim_immediately_reclaimable() {
        let mut fixture = super::super::active::tests::fixture_with_max_canary_roots(128);
        fill_drained_boundary(&mut fixture, 32, 32);
        let first_request = claim_request(fixture.admission.active_experiment_id, "dead-owner");
        let first_claim = match fixture
            .activated
            .repository
            .claim_next_active_look(&first_request)
            .unwrap()
        {
            ActiveLookClaimAck::Claimed(claim) => claim,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        let config = fixture.config.clone();
        fixture
            .activated
            .repository
            .stop_process(
                ProcessStop::new(
                    Uuid::now_v7(),
                    Uuid::now_v7(),
                    Utc::now().timestamp_millis(),
                )
                .unwrap(),
            )
            .unwrap();
        drop(fixture.activated);
        let mut restarted = LedgerRepository::activate(&config).unwrap();
        let startup_reclaims: i64 = restarted
            .repository
            .connection
            .query_row(
                "SELECT count(*) FROM active_look_claim_state_events
                 WHERE active_look_claim_id = ?1 AND state = 'reclaimed'
                   AND process_instance_id = ?2",
                params![
                    first_claim.active_look_claim_id.to_string(),
                    restarted.identity.process_instance_id.to_string(),
                ],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(startup_reclaims, 1);
        std::thread::sleep(Duration::from_millis(2));
        let second_request = claim_request(first_claim.active_experiment_id, "new-owner");
        let reclaimed = restarted
            .repository
            .claim_next_active_look(&second_request)
            .unwrap();
        assert!(matches!(
            reclaimed,
            ActiveLookClaimAck::Reclaimed(ActiveLookClaimReceipt {
                active_look_claim_id,
                ..
            }) if active_look_claim_id == first_claim.active_look_claim_id
        ));
    }

    #[test]
    fn raw_minimum_skip_consumes_no_look_and_exhausts_final_cap() {
        let mut fixture = super::super::active::tests::fixture();
        fill_drained_boundary(&mut fixture, 64, 0);
        let request = claim_request(fixture.admission.active_experiment_id, "skip");
        let receipt = match fixture
            .activated
            .repository
            .claim_next_active_look(&request)
            .unwrap()
        {
            ActiveLookClaimAck::Skipped(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        assert_eq!(receipt.treatment_labeled, 64);
        assert_eq!(receipt.control_labeled, 0);
        assert!(receipt.exhausted);
        assert_eq!(
            fixture
                .activated
                .repository
                .claim_next_active_look(&request)
                .unwrap(),
            ActiveLookClaimAck::Skipped(receipt)
        );
        let state = fixture
            .activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT count(*) FROM active_outcome_looks),
                    (SELECT state FROM active_authorization_state_events
                     WHERE active_experiment_id = ?1 ORDER BY event_seq DESC LIMIT 1),
                    (SELECT terminal_reason FROM active_experiment_state_events
                     WHERE active_experiment_id = ?1 ORDER BY event_seq DESC LIMIT 1)",
                [fixture.admission.active_experiment_id.to_string()],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(state, (0, "exhausted".into(), Some("exhausted".into())));
    }

    #[test]
    fn completed_look_persists_full_audit_delta_and_final_transition() {
        let mut fixture = super::super::active::tests::fixture();
        fill_drained_boundary_with_labels(
            &mut fixture,
            32,
            32,
            ActiveSignalDisposition::Success,
            ActiveSignalDisposition::Failure,
        );
        let request = claim_request(fixture.admission.active_experiment_id, "commit");
        let claim = match fixture
            .activated
            .repository
            .claim_next_active_look(&request)
            .unwrap()
        {
            ActiveLookClaimAck::Claimed(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        let math_members = claim
            .members
            .iter()
            .copied()
            .map(ActiveFrozenLookMember::math_member)
            .collect::<Vec<_>>();
        let audit = match evaluate_active_look_v1(claim.policy, claim.as_of_unix_ms, &math_members)
            .unwrap()
        {
            ActiveLookEvaluationV1::Completed(audit) => audit,
            evaluation => panic!("expected a completed look: {evaluation:?}"),
        };
        assert_eq!(audit.state, LookProducedStateV1::Passed);
        let commit = ActiveLookCommit {
            active_look_claim_id: claim.active_look_claim_id,
            lease_token_hash: claim.lease_token_hash,
            active_outcome_look_id: Uuid::now_v7(),
            claim_terminal_state_event_id: Uuid::now_v7(),
            authorization_state_event_id: Uuid::now_v7(),
            audit,
        };
        let receipt = match fixture
            .activated
            .repository
            .commit_active_look(&commit)
            .unwrap()
        {
            ActiveLookCommitAck::Applied(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        assert_eq!(receipt.result_state, LookProducedStateV1::Passed);
        assert!(receipt.valid_until_unix_ms.is_some());
        assert!(receipt.experiment_terminal);
        assert_eq!(
            fixture
                .activated
                .repository
                .commit_active_look(&commit)
                .unwrap(),
            ActiveLookCommitAck::AlreadyApplied(receipt)
        );
        let counts = fixture
            .activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT count(*) FROM active_outcome_looks),
                    (SELECT count(*) FROM active_outcome_look_audits),
                    (SELECT count(*) FROM active_look_members),
                    (SELECT state FROM active_authorization_state_events
                     WHERE active_experiment_id = ?1 ORDER BY event_seq DESC LIMIT 1),
                    (SELECT terminal_reason FROM active_experiment_state_events
                     WHERE active_experiment_id = ?1 ORDER BY event_seq DESC LIMIT 1)",
                [fixture.admission.active_experiment_id.to_string()],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, Option<String>>(4)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(
            counts,
            (
                1,
                1,
                64,
                "closed_passed".into(),
                Some("closed_passed".into())
            )
        );
        let mut conflicting = commit;
        conflicting.authorization_state_event_id = Uuid::now_v7();
        assert_eq!(
            fixture
                .activated
                .repository
                .commit_active_look(&conflicting)
                .unwrap(),
            ActiveLookCommitAck::Conflict
        );
    }

    #[test]
    fn numeric_failure_invalidates_only_its_experiment_and_replays() {
        let mut fixture = super::super::active::tests::fixture();
        let outcome_policy_hash = fixture
            .activated
            .repository
            .connection
            .query_row(
                "SELECT outcome_policy_hash FROM outcome_policy_versions
                 WHERE pool_id = 'pool-a'",
                [],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let isolated = ActiveExperimentCreate::from_config(
            &fixture.config,
            &fixture.activated.identity,
            "pool-a",
            "candidate-a",
            "22".repeat(32),
            outcome_policy_hash,
            0,
        )
        .unwrap();
        fixture
            .activated
            .repository
            .create_active_experiment(&isolated)
            .unwrap();
        fill_drained_boundary(&mut fixture, 32, 32);
        let request = claim_request(fixture.admission.active_experiment_id, "failure");
        let claim = match fixture
            .activated
            .repository
            .claim_next_active_look(&request)
            .unwrap()
        {
            ActiveLookClaimAck::Claimed(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        let failure = ActiveLookFailure {
            active_look_failure_id: Uuid::now_v7(),
            active_look_claim_id: claim.active_look_claim_id,
            claim_terminal_state_event_id: Uuid::now_v7(),
            authorization_state_event_id: Uuid::now_v7(),
            lease_token_hash: claim.lease_token_hash,
            failure_kind: ActiveLookFailureKind::NumericFailure,
            stable_reason: "posterior_numeric_failure".into(),
        };
        assert_eq!(
            fixture
                .activated
                .repository
                .fail_active_look(&failure)
                .unwrap(),
            ActiveLookFailureAck::Applied
        );
        assert_eq!(
            fixture
                .activated
                .repository
                .fail_active_look(&failure)
                .unwrap(),
            ActiveLookFailureAck::AlreadyApplied
        );
        let states = fixture
            .activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT state FROM active_authorization_state_events
                     WHERE active_experiment_id = ?1 ORDER BY event_seq DESC LIMIT 1),
                    (SELECT terminal_reason FROM active_experiment_state_events
                     WHERE active_experiment_id = ?1 ORDER BY event_seq DESC LIMIT 1),
                    (SELECT state FROM active_authorization_state_events
                     WHERE active_experiment_id = ?2 ORDER BY event_seq DESC LIMIT 1),
                    (SELECT state FROM active_experiment_state_events
                     WHERE active_experiment_id = ?2 ORDER BY event_seq DESC LIMIT 1)",
                params![
                    fixture.admission.active_experiment_id.to_string(),
                    isolated.active_experiment_id.to_string()
                ],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, String>(3)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(
            states,
            (
                "invalidated".into(),
                Some("invalidated".into()),
                "collecting".into(),
                "collecting".into()
            )
        );
    }

    #[test]
    fn observed_expiry_is_durable_across_clock_rollback() {
        let mut fixture = super::super::active::tests::fixture_with_max_canary_roots(128);
        fill_drained_boundary_with_labels(
            &mut fixture,
            32,
            32,
            ActiveSignalDisposition::Success,
            ActiveSignalDisposition::Failure,
        );
        let request = claim_request(fixture.admission.active_experiment_id, "expiry");
        let claim = match fixture
            .activated
            .repository
            .claim_next_active_look(&request)
            .unwrap()
        {
            ActiveLookClaimAck::Claimed(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        let members = claim
            .members
            .iter()
            .copied()
            .map(ActiveFrozenLookMember::math_member)
            .collect::<Vec<_>>();
        let audit =
            match evaluate_active_look_v1(claim.policy, claim.as_of_unix_ms, &members).unwrap() {
                ActiveLookEvaluationV1::Completed(audit) => audit,
                evaluation => panic!("expected completed look: {evaluation:?}"),
            };
        assert_eq!(audit.state, LookProducedStateV1::Passed);
        let commit = ActiveLookCommit {
            active_look_claim_id: claim.active_look_claim_id,
            lease_token_hash: claim.lease_token_hash,
            active_outcome_look_id: Uuid::now_v7(),
            claim_terminal_state_event_id: Uuid::now_v7(),
            authorization_state_event_id: Uuid::now_v7(),
            audit,
        };
        let receipt = match fixture
            .activated
            .repository
            .commit_active_look(&commit)
            .unwrap()
        {
            ActiveLookCommitAck::Applied(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        assert!(!receipt.experiment_terminal);
        let valid_until = receipt.valid_until_unix_ms.unwrap();
        let before = fixture
            .activated
            .repository
            .observe_active_authorization_at(
                fixture.admission.active_experiment_id,
                i64::try_from(valid_until - 1).unwrap(),
            )
            .unwrap();
        assert!(matches!(
            before,
            ActiveAuthorizationObserveAck::Current(ActiveAuthorizationSnapshot {
                state: ActiveAuthorizationState::Passed,
                authorizing: true,
                ..
            })
        ));
        let expired = fixture
            .activated
            .repository
            .observe_active_authorization_at(
                fixture.admission.active_experiment_id,
                i64::try_from(valid_until).unwrap(),
            )
            .unwrap();
        assert!(matches!(
            expired,
            ActiveAuthorizationObserveAck::Current(ActiveAuthorizationSnapshot {
                state: ActiveAuthorizationState::Expired,
                authorizing: false,
                ..
            })
        ));
        let rollback = fixture
            .activated
            .repository
            .observe_active_authorization_at(
                fixture.admission.active_experiment_id,
                i64::try_from(valid_until - 60_000).unwrap(),
            )
            .unwrap();
        assert!(matches!(
            rollback,
            ActiveAuthorizationObserveAck::Current(ActiveAuthorizationSnapshot {
                state: ActiveAuthorizationState::Expired,
                authorizing: false,
                ..
            })
        ));
    }

    #[test]
    fn initial_collecting_authority_supports_bounded_neighborhood_promotion() {
        let mut fixture = super::super::active::tests::fixture();
        let authorization_id = fixture
            .activated
            .repository
            .connection
            .query_row(
                "SELECT active_authorization_state_event_id
                 FROM active_authorization_state_events
                 WHERE active_experiment_id = ?1 ORDER BY event_seq DESC LIMIT 1",
                [fixture.admission.active_experiment_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let promotion = ActiveNeighborhoodPromotion {
            active_neighborhood_state_event_id: Uuid::now_v7(),
            key: ActiveNeighborhoodKey {
                active_experiment_id: fixture.admission.active_experiment_id,
                canonical_query_hash: canonical_sha256(&json!({})).unwrap(),
                sorted_neighbor_hash: "22".repeat(32),
                config_generation_id: fixture.admission.config_generation_id.clone(),
                learning_generation_id: fixture.admission.learning_generation_id,
                cohort_generation_id: fixture.admission.cohort_generation_id,
                active_authorization_state_event_id: Uuid::parse_str(&authorization_id).unwrap(),
            },
            stable_reason: "fresh_bootstrap_support".into(),
        };
        assert!(matches!(
            fixture
                .activated
                .repository
                .promote_active_neighborhood(&promotion)
                .unwrap(),
            ActiveNeighborhoodMutationAck::Applied(ActiveNeighborhoodSnapshot {
                state: ActiveNeighborhoodState::Promoted,
                authorizing: true,
                ..
            })
        ));
    }

    #[test]
    fn neighborhood_invalidation_cooloff_and_restoration_are_query_local() {
        let mut fixture = super::super::active::tests::fixture_with_max_canary_roots(128);
        let look = complete_passing_look(&mut fixture);
        let valid_until = i64::try_from(look.valid_until_unix_ms.unwrap()).unwrap();
        let observed_at = valid_until - 500_000;
        let second_query = json!({"query": "second"});
        let second_query_json = canonical_json(&second_query).unwrap();
        let second_query_hash = canonical_sha256(&second_query).unwrap();
        fixture
            .activated
            .repository
            .connection
            .execute(
                "INSERT INTO canonical_routing_queries (
                    canonical_query_hash, canonical_query_json, canonical_size_bytes,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, ?1)",
                params![
                    second_query_hash,
                    second_query_json,
                    i64::try_from(second_query_json.len()).unwrap(),
                    observed_at,
                ],
            )
            .unwrap();
        let key = ActiveNeighborhoodKey {
            active_experiment_id: fixture.admission.active_experiment_id,
            canonical_query_hash: canonical_sha256(&json!({})).unwrap(),
            sorted_neighbor_hash: "22".repeat(32),
            config_generation_id: fixture.admission.config_generation_id.clone(),
            learning_generation_id: fixture.admission.learning_generation_id,
            cohort_generation_id: fixture.admission.cohort_generation_id,
            active_authorization_state_event_id: look.authorization_state_event_id,
        };
        let other_key = ActiveNeighborhoodKey {
            canonical_query_hash: second_query_hash,
            sorted_neighbor_hash: "33".repeat(32),
            ..key.clone()
        };
        let promotion = ActiveNeighborhoodPromotion {
            active_neighborhood_state_event_id: Uuid::now_v7(),
            key: key.clone(),
            stable_reason: "fresh_promotion_support".into(),
        };
        let other_promotion = ActiveNeighborhoodPromotion {
            active_neighborhood_state_event_id: Uuid::now_v7(),
            key: other_key.clone(),
            stable_reason: "fresh_promotion_support".into(),
        };
        assert!(matches!(
            fixture
                .activated
                .repository
                .promote_active_neighborhood_at(&promotion, observed_at)
                .unwrap(),
            ActiveNeighborhoodMutationAck::Applied(ActiveNeighborhoodSnapshot {
                state: ActiveNeighborhoodState::Promoted,
                authorizing: true,
                ..
            })
        ));
        assert!(matches!(
            fixture
                .activated
                .repository
                .promote_active_neighborhood_at(&other_promotion, observed_at)
                .unwrap(),
            ActiveNeighborhoodMutationAck::Applied(_)
        ));
        let cause_active_dispatch_id = fixture
            .activated
            .repository
            .connection
            .query_row(
                "SELECT active_dispatch_id FROM active_dispatches
                 WHERE active_experiment_id = ?1
                 ORDER BY admitted_at_unix_ms, active_dispatch_id LIMIT 1",
                [fixture.admission.active_experiment_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .map(|value| Uuid::parse_str(&value).unwrap())
            .unwrap();
        let invalidation = ActiveNeighborhoodInvalidation {
            invalidated_state_event_id: Uuid::now_v7(),
            cooloff_state_event_id: Uuid::now_v7(),
            key: key.clone(),
            cause_active_dispatch_id: Some(cause_active_dispatch_id),
            cause_outcome_id: None,
            stable_reason: "candidate_provider_error".into(),
            cooloff_duration_seconds: 300,
        };
        let cooloff = match fixture
            .activated
            .repository
            .invalidate_active_neighborhood_at(&invalidation, observed_at + 1_000)
            .unwrap()
        {
            ActiveNeighborhoodMutationAck::Applied(snapshot) => snapshot,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        assert_eq!(cooloff.state, ActiveNeighborhoodState::Cooloff);
        assert!(!cooloff.authorizing);
        assert!(!cooloff.restorable);
        assert!(matches!(
            fixture
                .activated
                .repository
                .invalidate_active_neighborhood_at(&invalidation, observed_at + 2_000)
                .unwrap(),
            ActiveNeighborhoodMutationAck::AlreadyApplied(_)
        ));
        assert!(matches!(
            fixture
                .activated
                .repository
                .observe_active_neighborhood_at(&other_key, observed_at + 2_000)
                .unwrap(),
            ActiveNeighborhoodObserveAck::Current(ActiveNeighborhoodSnapshot {
                state: ActiveNeighborhoodState::Promoted,
                authorizing: true,
                ..
            })
        ));
        let cooloff_until = i64::try_from(cooloff.cooloff_until_unix_ms.unwrap()).unwrap();
        assert!(matches!(
            fixture
                .activated
                .repository
                .observe_active_neighborhood_at(&key, cooloff_until - 1)
                .unwrap(),
            ActiveNeighborhoodObserveAck::Current(ActiveNeighborhoodSnapshot {
                state: ActiveNeighborhoodState::Cooloff,
                restorable: false,
                ..
            })
        ));
        assert!(matches!(
            fixture
                .activated
                .repository
                .observe_active_neighborhood_at(&key, cooloff_until)
                .unwrap(),
            ActiveNeighborhoodObserveAck::Current(ActiveNeighborhoodSnapshot {
                state: ActiveNeighborhoodState::Cooloff,
                restorable: true,
                ..
            })
        ));
        let restoration = ActiveNeighborhoodPromotion {
            active_neighborhood_state_event_id: Uuid::now_v7(),
            key: key.clone(),
            stable_reason: "fresh_restoration_support".into(),
        };
        assert!(matches!(
            fixture
                .activated
                .repository
                .promote_active_neighborhood_at(&restoration, cooloff_until)
                .unwrap(),
            ActiveNeighborhoodMutationAck::Applied(ActiveNeighborhoodSnapshot {
                state: ActiveNeighborhoodState::Restored,
                authorizing: true,
                ..
            })
        ));
    }

    #[test]
    fn generation_supersession_drains_then_terminalizes_open_roots() {
        let mut fixture = super::super::active::tests::fixture_with_max_canary_roots(128);
        let receipt = match fixture
            .activated
            .repository
            .admit_active_root(&fixture.admission)
            .unwrap()
        {
            ActiveAdmissionAck::Applied(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        let identity = fixture.activated.identity.clone();
        let prepared = crate::ledger::repository::inspection::operator::prepare_learning_reset(
            crate::inspection::LearningResetRequestV1 {
                mutation_id: Uuid::now_v7(),
                scope: crate::inspection::LearningResetScopeV1::Pool {
                    pool_id: "pool-a".into(),
                    expected_learning_generation_id: identity.pools["pool-a"]
                        .learning_generation_id,
                },
                confirm_project_id: identity.project_id.clone(),
                actor: "operator".into(),
                reason: "supersede active experiment".into(),
            },
        )
        .unwrap();
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        let reset = crate::ledger::repository::inspection::operator::apply_operator_mutation_in_transaction(
            &transaction,
            identity.project_uuid,
            identity.process_instance_id,
            &["pool-a".to_string()],
            &prepared,
            Utc::now().timestamp_millis(),
        )
        .unwrap();
        assert_eq!(
            reset.receipt.result,
            crate::inspection::OperatorMutationResultV1::Applied
        );
        transaction.commit().unwrap();
        let draining = fixture
            .activated
            .repository
            .observe_active_authorization(fixture.admission.active_experiment_id)
            .unwrap();
        assert!(matches!(
            draining,
            ActiveAuthorizationObserveAck::Current(ActiveAuthorizationSnapshot {
                state: ActiveAuthorizationState::SupersededDraining,
                authorizing: false,
                ..
            })
        ));
        fixture
            .activated
            .repository
            .terminalize_active_root(&super::super::active::tests::terminal_for(
                &fixture.admission,
                receipt,
                ActiveRootClosure::Orphaned,
                false,
                None,
            ))
            .unwrap();
        let terminal = fixture
            .activated
            .repository
            .observe_active_authorization(fixture.admission.active_experiment_id)
            .unwrap();
        assert!(matches!(
            terminal,
            ActiveAuthorizationObserveAck::Current(ActiveAuthorizationSnapshot {
                state: ActiveAuthorizationState::Superseded,
                authorizing: false,
                ..
            })
        ));
        let reason = fixture
            .activated
            .repository
            .connection
            .query_row(
                "SELECT terminal_reason FROM active_experiment_state_events
                 WHERE active_experiment_id = ?1 ORDER BY event_seq DESC LIMIT 1",
                [fixture.admission.active_experiment_id.to_string()],
                |row| row.get::<_, Option<String>>(0),
            )
            .unwrap();
        assert_eq!(reason.as_deref(), Some("superseded"));
    }

    #[test]
    fn cohort_rotation_drains_then_terminalizes_open_roots() {
        let mut fixture = super::super::active::tests::fixture_with_max_canary_roots(128);
        let receipt = match fixture
            .activated
            .repository
            .admit_active_root(&fixture.admission)
            .unwrap()
        {
            ActiveAdmissionAck::Applied(receipt) => receipt,
            acknowledgement => panic!("unexpected acknowledgement: {acknowledgement:?}"),
        };
        let identity = fixture.activated.identity.clone();
        let prepared = crate::ledger::repository::inspection::operator::prepare_cohort_rotation(
            crate::inspection::CohortRotationRequestV1 {
                mutation_id: Uuid::now_v7(),
                expected_cohort_generation_id: identity.cohort_generation_id,
                confirm_project_id: identity.project_id.clone(),
                actor: "operator".into(),
                reason: "supersede active cohort".into(),
            },
        )
        .unwrap();
        let transaction = fixture
            .activated
            .repository
            .connection_mut()
            .transaction()
            .unwrap();
        let rotation = crate::ledger::repository::inspection::operator::apply_operator_mutation_in_transaction(
            &transaction,
            identity.project_uuid,
            identity.process_instance_id,
            &["pool-a".to_string()],
            &prepared,
            Utc::now().timestamp_millis(),
        )
        .unwrap();
        assert_eq!(
            rotation.receipt.result,
            crate::inspection::OperatorMutationResultV1::Applied
        );
        transaction.commit().unwrap();

        assert!(matches!(
            fixture
                .activated
                .repository
                .observe_active_authorization(fixture.admission.active_experiment_id)
                .unwrap(),
            ActiveAuthorizationObserveAck::Current(ActiveAuthorizationSnapshot {
                state: ActiveAuthorizationState::SupersededDraining,
                authorizing: false,
                ..
            })
        ));
        fixture
            .activated
            .repository
            .terminalize_active_root(&super::super::active::tests::terminal_for(
                &fixture.admission,
                receipt,
                ActiveRootClosure::Orphaned,
                false,
                None,
            ))
            .unwrap();
        assert!(matches!(
            fixture
                .activated
                .repository
                .observe_active_authorization(fixture.admission.active_experiment_id)
                .unwrap(),
            ActiveAuthorizationObserveAck::Current(ActiveAuthorizationSnapshot {
                state: ActiveAuthorizationState::Superseded,
                authorizing: false,
                ..
            })
        ));
    }
}
