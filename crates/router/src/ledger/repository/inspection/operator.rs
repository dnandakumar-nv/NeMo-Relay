// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transactional generation authority and durable operator mutation receipts.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Instant;

use chrono::Utc;
use rusqlite::{
    Connection, ErrorCode, OptionalExtension, Transaction, TransactionBehavior, params,
};
use serde_json::json;
use unicode_normalization::UnicodeNormalization;
use uuid::{Uuid, Variant};
use zeroize::Zeroizing;

use super::super::{
    append_cohort_state_event, append_learning_generation, hash_json, load_or_initialize_cohort,
    load_or_initialize_learning, load_verified_cohort_generation,
};
use crate::canonical_json::canonical_sha256;
use crate::config::PROJECT_ID_MAX_BYTES;
use crate::control::{CONTROL_ACTOR_MAX_BYTES, CONTROL_REASON_MAX_BYTES, ControlTransactionFence};
use crate::fingerprint::sha256_hex;
use crate::inspection::{
    CohortRotationRequestV1, InspectionError, LearningResetRequestV1, LearningResetScopeV1,
    OperatorHistoryEntryV1, OperatorHistoryKindV1, OperatorMutationReceiptV1,
    OperatorMutationResultV1,
};
use crate::ledger::model::{
    COHORT_ASSIGNMENT_ALGORITHM_V1, COHORT_SALT_BYTES, CohortSalt, LedgerError, LedgerErrorClass,
};

const COHORT_SCOPE_KEY: &str = "cohort";
const OPERATOR_REQUEST_SCHEMA_V1: &str = "nemo.relay.router.operator-mutation-request@1";
const OPERATOR_RECEIPT_SCHEMA_V1: &str = "nemo.relay.router.operator-mutation-receipt@1";
const OPERATOR_MUTATION_CLOCK_SKEW_MS: i128 = 300_000;
const OPERATOR_RECEIPT_GRAPH_MAX_ROWS: usize = 100_000;

/// Exact current learning and protected cohort authority from one transaction.
pub(crate) struct CurrentGenerationAuthority {
    pub(crate) learning_generation_ids: BTreeMap<String, Uuid>,
    pub(crate) cohort_generation_id: Uuid,
    pub(in crate::ledger) cohort_salt: CohortSalt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OperatorMutationKind {
    ResetPool,
    ResetAll,
    RotateCohort,
}

impl OperatorMutationKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::ResetPool => "reset_pool",
            Self::ResetAll => "reset_all",
            Self::RotateCohort => "rotate_cohort",
        }
    }

    fn parse(value: &str) -> Result<Self, InspectionError> {
        match value {
            "reset_pool" => Ok(Self::ResetPool),
            "reset_all" => Ok(Self::ResetAll),
            "rotate_cohort" => Ok(Self::RotateCohort),
            _ => Err(InspectionError::IntegrityError),
        }
    }

    const fn generation_kind(self) -> GenerationKind {
        match self {
            Self::ResetPool | Self::ResetAll => GenerationKind::Learning,
            Self::RotateCohort => GenerationKind::Cohort,
        }
    }
}

/// Fully validated semantic mutation with its stable request hash.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct PreparedOperatorMutation {
    mutation_id: Uuid,
    kind: OperatorMutationKind,
    scope_pool_id: Option<String>,
    expected_generations: BTreeMap<String, Uuid>,
    confirm_project_id: String,
    actor: String,
    reason: String,
    canonical_request_hash: String,
}

impl PreparedOperatorMutation {
    pub(crate) const fn mutation_id(&self) -> Uuid {
        self.mutation_id
    }

    pub(crate) const fn kind(&self) -> OperatorMutationKind {
        self.kind
    }

    pub(crate) fn actor(&self) -> &str {
        &self.actor
    }

    pub(crate) fn reason(&self) -> &str {
        &self.reason
    }
}

/// Committed semantic result. `replayed` distinguishes a first write from exact replay.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OperatorMutationAck {
    pub(crate) receipt: OperatorMutationReceiptV1,
    pub(crate) replayed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum OperatorMutationTransactionAck {
    Completed(OperatorMutationAck),
    TransactionNotStarted,
}

impl super::super::LedgerRepository {
    pub(crate) fn apply_operator_mutation_with_start_check<
        G: super::super::TransactionStartGuard,
    >(
        &mut self,
        prepared: &PreparedOperatorMutation,
        fence: &Arc<ControlTransactionFence>,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<OperatorMutationTransactionAck, InspectionError> {
        let database_path = self.database_path.clone();
        super::super::enforce_sidecar_permissions(&database_path)
            .map_err(|_| InspectionError::StorageUnavailable)?;
        let Some(start_guard) = start_check() else {
            fence.abort();
            return Ok(OperatorMutationTransactionAck::TransactionNotStarted);
        };
        let now = Instant::now();
        let Some(remaining) = fence.start_deadline().checked_duration_since(now) else {
            fence.expire();
            return Ok(OperatorMutationTransactionAck::TransactionNotStarted);
        };
        self.connection
            .busy_timeout(remaining)
            .map_err(|_| InspectionError::StorageUnavailable)?;
        let active_pool_ids = self.active_pool_ids.iter().cloned().collect::<Vec<_>>();
        let outcome = execute_operator_transaction(
            &mut self.connection,
            &database_path,
            self.project_uuid,
            self.process_instance_id,
            &active_pool_ids,
            prepared,
            fence,
            start_guard,
        );
        self.connection
            .busy_timeout(super::super::BUSY_TIMEOUT)
            .map_err(|_| InspectionError::StorageUnavailable)?;
        outcome
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_operator_transaction<G: super::super::TransactionStartGuard>(
    connection: &mut Connection,
    database_path: &std::path::Path,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    active_pool_ids: &[String],
    prepared: &PreparedOperatorMutation,
    fence: &Arc<ControlTransactionFence>,
    start_guard: G,
) -> Result<OperatorMutationTransactionAck, InspectionError> {
    let transaction = match connection.transaction_with_behavior(TransactionBehavior::Immediate) {
        Ok(transaction) => transaction,
        Err(error) => {
            if !start_guard.permits_transaction() {
                fence.abort();
                return Ok(OperatorMutationTransactionAck::TransactionNotStarted);
            }
            if matches!(
                error.sqlite_error_code(),
                Some(ErrorCode::DatabaseBusy | ErrorCode::DatabaseLocked)
            ) {
                fence.expire();
                return Ok(OperatorMutationTransactionAck::TransactionNotStarted);
            }
            return Err(InspectionError::StorageUnavailable);
        }
    };
    if !start_guard.permits_transaction() {
        drop(transaction);
        fence.abort();
        return Ok(OperatorMutationTransactionAck::TransactionNotStarted);
    }
    if !fence.try_start() {
        drop(transaction);
        return Ok(OperatorMutationTransactionAck::TransactionNotStarted);
    }
    drop(start_guard);

    let created_at_unix_ms = Utc::now().timestamp_millis();
    if created_at_unix_ms < 0 {
        return Err(InspectionError::StorageUnavailable);
    }
    let acknowledgement = apply_operator_mutation_in_transaction(
        &transaction,
        project_uuid,
        process_instance_id,
        active_pool_ids,
        prepared,
        created_at_unix_ms,
    )?;
    super::super::enforce_sidecar_permissions(database_path)
        .map_err(|_| InspectionError::StorageUnavailable)?;
    transaction
        .commit()
        .map_err(|_| InspectionError::StorageUnavailable)?;
    Ok(OperatorMutationTransactionAck::Completed(acknowledgement))
}

#[derive(Debug)]
struct StoredReceipt {
    operator_ordinal: i64,
    mutation_id: String,
    project_uuid: String,
    mutation_kind: String,
    scope_pool_id: Option<String>,
    canonical_request_hash: String,
    result: String,
    confirm_project_id: String,
    expected_generation_count: i64,
    prior_generation_count: i64,
    resulting_generation_count: i64,
    superseded_generation_count: i64,
    actor: String,
    reason: String,
    process_instance_id: String,
    created_at_unix_ms: i64,
    record_hash: String,
    canonical_payload_hash: String,
}

#[derive(Debug)]
struct StoredEdge {
    role: String,
    ordinal: i64,
    scope_key: String,
    generation_kind: String,
    pool_id: Option<String>,
    learning_generation_id: Option<String>,
    cohort_generation_id: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GenerationKind {
    Learning,
    Cohort,
}

impl GenerationKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Learning => "learning",
            Self::Cohort => "cohort",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum EdgeRole {
    Expected,
    Prior,
    Resulting,
    Superseded,
}

impl EdgeRole {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Expected => "expected",
            Self::Prior => "prior",
            Self::Resulting => "resulting",
            Self::Superseded => "superseded",
        }
    }

    fn parse(value: &str) -> Result<Self, InspectionError> {
        match value {
            "expected" => Ok(Self::Expected),
            "prior" => Ok(Self::Prior),
            "resulting" => Ok(Self::Resulting),
            "superseded" => Ok(Self::Superseded),
            _ => Err(InspectionError::IntegrityError),
        }
    }
}

#[derive(Debug)]
struct VerifiedStoredReceipt {
    prepared: PreparedOperatorMutation,
    receipt: OperatorMutationReceiptV1,
}

pub(crate) fn load_verified_operator_history_entry(
    connection: &Connection,
    project_uuid: Uuid,
    mutation_id: Uuid,
) -> Result<OperatorHistoryEntryV1, InspectionError> {
    let VerifiedStoredReceipt { prepared, receipt } =
        load_verified_receipt(connection, project_uuid, mutation_id)?
            .ok_or(InspectionError::IntegrityError)?;
    let kind = match prepared.kind {
        OperatorMutationKind::ResetPool => OperatorHistoryKindV1::ResetPool,
        OperatorMutationKind::ResetAll => OperatorHistoryKindV1::ResetAll,
        OperatorMutationKind::RotateCohort => OperatorHistoryKindV1::RotateCohort,
    };
    let superseded_ids = match receipt.result {
        OperatorMutationResultV1::Applied => receipt.prior_generations.values().copied().collect(),
        OperatorMutationResultV1::Conflict => Vec::new(),
    };
    Ok(OperatorHistoryEntryV1 {
        audit_id: receipt.mutation_id,
        kind,
        result: result_name(receipt.result).into(),
        control_generation: None,
        scope: None,
        prior_value: None,
        new_value: None,
        prior_generations: receipt.prior_generations,
        new_generations: receipt.resulting_generations,
        superseded_ids,
        actor: prepared.actor,
        reason: prepared.reason,
        created_at_unix_ms: receipt.created_at_unix_ms,
    })
}

/// Validate and hash one learning-reset request without storage access.
pub(crate) fn prepare_learning_reset(
    request: LearningResetRequestV1,
) -> Result<PreparedOperatorMutation, InspectionError> {
    validate_common_request(
        request.mutation_id,
        &request.confirm_project_id,
        &request.actor,
        &request.reason,
    )?;
    let (kind, scope_pool_id, expected_generations) = match request.scope {
        LearningResetScopeV1::Pool {
            pool_id,
            expected_learning_generation_id,
        } => {
            if !valid_stable_id(&pool_id, 128) || !is_uuid_v7(expected_learning_generation_id) {
                return Err(InspectionError::InvalidArgument);
            }
            (
                OperatorMutationKind::ResetPool,
                Some(pool_id.clone()),
                BTreeMap::from([(pool_id, expected_learning_generation_id)]),
            )
        }
        LearningResetScopeV1::All {
            expected_learning_generation_ids,
        } => {
            validate_expected_generations(&expected_learning_generation_ids, false)?;
            (
                OperatorMutationKind::ResetAll,
                None,
                expected_learning_generation_ids,
            )
        }
    };
    prepare_operator_mutation(
        request.mutation_id,
        kind,
        scope_pool_id,
        expected_generations,
        request.confirm_project_id,
        request.actor,
        request.reason,
    )
}

/// Validate and hash one cohort-rotation request without storage access.
pub(crate) fn prepare_cohort_rotation(
    request: CohortRotationRequestV1,
) -> Result<PreparedOperatorMutation, InspectionError> {
    validate_common_request(
        request.mutation_id,
        &request.confirm_project_id,
        &request.actor,
        &request.reason,
    )?;
    if !is_uuid_v7(request.expected_cohort_generation_id) {
        return Err(InspectionError::InvalidArgument);
    }
    prepare_operator_mutation(
        request.mutation_id,
        OperatorMutationKind::RotateCohort,
        None,
        BTreeMap::from([(
            COHORT_SCOPE_KEY.to_string(),
            request.expected_cohort_generation_id,
        )]),
        request.confirm_project_id,
        request.actor,
        request.reason,
    )
}

/// Load all current generation pointers and verify their complete stored records.
pub(crate) fn load_current_generation_authority(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    pool_ids: &[String],
) -> Result<CurrentGenerationAuthority, LedgerError> {
    let lexical_pool_ids = pool_ids.iter().cloned().collect::<BTreeSet<_>>();
    if lexical_pool_ids.len() != pool_ids.len() {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let learning_generation_ids = lexical_pool_ids
        .into_iter()
        .map(|pool_id| {
            load_or_initialize_learning(transaction, project_uuid, &pool_id, 0, false)
                .map(|generation_id| (pool_id, generation_id))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let cohort_generation_id = load_or_initialize_cohort(transaction, project_uuid, 0, false)?;
    let (verified_cohort_generation_id, cohort_salt) = load_verified_cohort_generation(
        transaction,
        project_uuid,
        &cohort_generation_id.to_string(),
    )?;
    if verified_cohort_generation_id != cohort_generation_id {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(CurrentGenerationAuthority {
        learning_generation_ids,
        cohort_generation_id,
        cohort_salt,
    })
}

/// Apply or exactly replay one prepared reset/rotation inside the caller's transaction.
pub(crate) fn apply_operator_mutation_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    active_pool_ids: &[String],
    prepared: &PreparedOperatorMutation,
    created_at_unix_ms: i64,
) -> Result<OperatorMutationAck, InspectionError> {
    apply_operator_mutation_with_capacity(
        transaction,
        project_uuid,
        process_instance_id,
        active_pool_ids,
        prepared,
        created_at_unix_ms,
        OPERATOR_RECEIPT_GRAPH_MAX_ROWS,
    )
}

#[allow(clippy::too_many_arguments)]
fn apply_operator_mutation_with_capacity(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    active_pool_ids: &[String],
    prepared: &PreparedOperatorMutation,
    created_at_unix_ms: i64,
    capacity: usize,
) -> Result<OperatorMutationAck, InspectionError> {
    if let Some(stored) = load_verified_receipt(transaction, project_uuid, prepared.mutation_id)? {
        if stored.prepared != *prepared {
            return Err(InspectionError::InvalidArgument);
        }
        return Ok(OperatorMutationAck {
            receipt: stored.receipt,
            replayed: true,
        });
    }
    if created_at_unix_ms < 0 {
        return Err(InspectionError::StorageUnavailable);
    }
    validate_uuid_time(prepared.mutation_id, created_at_unix_ms)?;
    let stored_project_id = load_verified_project_id(transaction, project_uuid)?;
    if prepared.confirm_project_id != stored_project_id {
        return Err(InspectionError::InvalidArgument);
    }
    match super::super::process::originating_process_is_live(
        transaction,
        project_uuid,
        process_instance_id,
    ) {
        Ok(true) => {}
        Ok(false) => return Err(InspectionError::StorageUnavailable),
        Err(_) => return Err(InspectionError::IntegrityError),
    }

    let pool_ids = active_pool_ids.iter().cloned().collect::<BTreeSet<_>>();
    if pool_ids.len() != active_pool_ids.len()
        || pool_ids
            .iter()
            .any(|pool_id| !valid_stable_id(pool_id, 128))
    {
        return Err(InspectionError::IntegrityError);
    }
    let lexical_pool_ids = pool_ids.into_iter().collect::<Vec<_>>();
    let current = load_current_generation_authority(transaction, project_uuid, &lexical_pool_ids)
        .map_err(|_| InspectionError::IntegrityError)?;
    let prior_generations = relevant_prior_generations(prepared, &current)?;
    let result = if prepared.expected_generations == prior_generations {
        OperatorMutationResultV1::Applied
    } else {
        OperatorMutationResultV1::Conflict
    };
    let superseded_count = if result == OperatorMutationResultV1::Applied {
        prior_generations.len()
    } else {
        0
    };
    let added_rows = 1_usize
        .checked_add(prepared.expected_generations.len())
        .and_then(|value| value.checked_add(prior_generations.len()))
        .and_then(|value| value.checked_add(prior_generations.len()))
        .and_then(|value| value.checked_add(superseded_count))
        .ok_or(InspectionError::CapacityExhausted)?;
    enforce_operator_capacity(transaction, project_uuid, added_rows, capacity)?;

    let (resulting_generations, superseded_generations) = match result {
        OperatorMutationResultV1::Conflict => (prior_generations.clone(), BTreeMap::new()),
        OperatorMutationResultV1::Applied => {
            let resulting = append_resulting_generations(
                transaction,
                project_uuid,
                prepared,
                &prior_generations,
                created_at_unix_ms,
            )?;
            (resulting, prior_generations.clone())
        }
    };
    let receipt = insert_operator_receipt(
        transaction,
        project_uuid,
        process_instance_id,
        prepared,
        result,
        &prior_generations,
        &resulting_generations,
        &superseded_generations,
        created_at_unix_ms,
    )?;
    Ok(OperatorMutationAck {
        receipt,
        replayed: false,
    })
}

fn prepare_operator_mutation(
    mutation_id: Uuid,
    kind: OperatorMutationKind,
    scope_pool_id: Option<String>,
    expected_generations: BTreeMap<String, Uuid>,
    confirm_project_id: String,
    actor: String,
    reason: String,
) -> Result<PreparedOperatorMutation, InspectionError> {
    let canonical_request_hash = operator_request_hash(
        kind,
        scope_pool_id.as_deref(),
        &expected_generations,
        &confirm_project_id,
        &actor,
        &reason,
    )?;
    Ok(PreparedOperatorMutation {
        mutation_id,
        kind,
        scope_pool_id,
        expected_generations,
        confirm_project_id,
        actor,
        reason,
        canonical_request_hash,
    })
}

fn validate_common_request(
    mutation_id: Uuid,
    confirm_project_id: &str,
    actor: &str,
    reason: &str,
) -> Result<(), InspectionError> {
    if !is_uuid_v7(mutation_id)
        || !valid_stable_id(confirm_project_id, PROJECT_ID_MAX_BYTES)
        || !valid_operator_text(actor, CONTROL_ACTOR_MAX_BYTES)
        || !valid_operator_text(reason, CONTROL_REASON_MAX_BYTES)
    {
        return Err(InspectionError::InvalidArgument);
    }
    Ok(())
}

fn validate_expected_generations(
    generations: &BTreeMap<String, Uuid>,
    cohort: bool,
) -> Result<(), InspectionError> {
    if generations.is_empty()
        || generations.iter().any(|(scope, generation_id)| {
            (cohort && scope != COHORT_SCOPE_KEY)
                || (!cohort && !valid_stable_id(scope, 128))
                || !is_uuid_v7(*generation_id)
        })
    {
        return Err(InspectionError::InvalidArgument);
    }
    Ok(())
}

fn valid_stable_id(value: &str, max_bytes: usize) -> bool {
    if value.is_empty()
        || value.len() > max_bytes
        || value.chars().any(char::is_control)
        || !value.nfc().eq(value.chars())
    {
        return false;
    }
    let mut characters = value.chars();
    characters.next().is_some_and(char::is_alphanumeric)
        && characters.all(|character| {
            character.is_alphanumeric() || matches!(character, '_' | '-' | '.' | ':')
        })
}

fn valid_operator_text(value: &str, max_bytes: usize) -> bool {
    !value.trim().is_empty()
        && value.len() <= max_bytes
        && value.chars().all(|character| !character.is_control())
}

fn is_uuid_v7(value: Uuid) -> bool {
    value.get_variant() == Variant::RFC4122 && value.get_version_num() == 7
}

fn operator_request_hash(
    kind: OperatorMutationKind,
    scope_pool_id: Option<&str>,
    expected_generations: &BTreeMap<String, Uuid>,
    confirm_project_id: &str,
    actor: &str,
    reason: &str,
) -> Result<String, InspectionError> {
    canonical_sha256(&json!({
        "schema": OPERATOR_REQUEST_SCHEMA_V1,
        "mutation_kind": kind.as_str(),
        "scope_pool_id": scope_pool_id,
        "expected_generations": expected_generations,
        "confirm_project_id": confirm_project_id,
        "actor": actor,
        "reason": reason,
    }))
    .map_err(|_| InspectionError::InvalidArgument)
}

fn validate_uuid_time(mutation_id: Uuid, created_at_unix_ms: i64) -> Result<(), InspectionError> {
    let timestamp = mutation_id
        .get_timestamp()
        .ok_or(InspectionError::InvalidArgument)?;
    let (seconds, subsec_nanos) = timestamp.to_unix();
    let mutation_millis = u128::from(seconds)
        .checked_mul(1_000)
        .and_then(|value| value.checked_add(u128::from(subsec_nanos / 1_000_000)))
        .and_then(|value| i128::try_from(value).ok())
        .ok_or(InspectionError::InvalidArgument)?;
    let difference = mutation_millis - i128::from(created_at_unix_ms);
    if (-OPERATOR_MUTATION_CLOCK_SKEW_MS..=OPERATOR_MUTATION_CLOCK_SKEW_MS).contains(&difference) {
        Ok(())
    } else {
        Err(InspectionError::MutationExpired)
    }
}

fn load_verified_project_id(
    connection: &Connection,
    expected_project_uuid: Uuid,
) -> Result<String, InspectionError> {
    let stored = connection
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
        .map_err(|_| InspectionError::StorageUnavailable)?
        .ok_or(InspectionError::IntegrityError)?;
    let (project_uuid, project_id, created_at_unix_ms, application_version, stored_hash) = stored;
    let parsed_project_uuid = parse_uuid_v7(&project_uuid)?;
    let expected_hash = canonical_sha256(&json!({
        "project_uuid": project_uuid,
        "project_id": project_id,
        "created_at_unix_ms": created_at_unix_ms,
        "application_version": application_version,
    }))
    .map_err(|_| InspectionError::IntegrityError)?;
    if parsed_project_uuid != expected_project_uuid
        || created_at_unix_ms < 0
        || !valid_stable_id(&project_id, PROJECT_ID_MAX_BYTES)
        || stored_hash != expected_hash
    {
        return Err(InspectionError::IntegrityError);
    }
    Ok(project_id)
}

fn relevant_prior_generations(
    prepared: &PreparedOperatorMutation,
    current: &CurrentGenerationAuthority,
) -> Result<BTreeMap<String, Uuid>, InspectionError> {
    match prepared.kind {
        OperatorMutationKind::ResetPool => {
            let pool_id = prepared
                .scope_pool_id
                .as_ref()
                .ok_or(InspectionError::IntegrityError)?;
            let generation_id = current
                .learning_generation_ids
                .get(pool_id)
                .copied()
                .ok_or(InspectionError::InvalidArgument)?;
            Ok(BTreeMap::from([(pool_id.clone(), generation_id)]))
        }
        OperatorMutationKind::ResetAll => {
            if prepared
                .expected_generations
                .keys()
                .ne(current.learning_generation_ids.keys())
            {
                return Err(InspectionError::InvalidArgument);
            }
            Ok(current.learning_generation_ids.clone())
        }
        OperatorMutationKind::RotateCohort => Ok(BTreeMap::from([(
            COHORT_SCOPE_KEY.to_string(),
            current.cohort_generation_id,
        )])),
    }
}

fn enforce_operator_capacity(
    connection: &Connection,
    project_uuid: Uuid,
    added_rows: usize,
    maximum_rows: usize,
) -> Result<(), InspectionError> {
    let live_rows = connection
        .query_row(
            "SELECT
                (SELECT count(*) FROM operator_mutation_receipts
                 WHERE project_uuid = ?1)
              + (SELECT count(*) FROM operator_mutation_generation_edges
                 WHERE project_uuid = ?1)",
            [project_uuid.to_string()],
            |row| row.get::<_, i64>(0),
        )
        .map_err(|_| InspectionError::StorageUnavailable)?;
    let live_rows = usize::try_from(live_rows).map_err(|_| InspectionError::IntegrityError)?;
    if added_rows == 0
        || maximum_rows == 0
        || live_rows > maximum_rows
        || live_rows
            .checked_add(added_rows)
            .is_none_or(|next| next > maximum_rows)
    {
        return Err(InspectionError::CapacityExhausted);
    }
    Ok(())
}

fn append_resulting_generations(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    prepared: &PreparedOperatorMutation,
    prior_generations: &BTreeMap<String, Uuid>,
    created_at_unix_ms: i64,
) -> Result<BTreeMap<String, Uuid>, InspectionError> {
    match prepared.kind {
        OperatorMutationKind::ResetPool | OperatorMutationKind::ResetAll => prior_generations
            .keys()
            .map(|pool_id| {
                append_learning_generation(
                    transaction,
                    project_uuid,
                    pool_id,
                    &prepared.actor,
                    &prepared.reason,
                    created_at_unix_ms,
                )
                .map(|generation_id| (pool_id.clone(), generation_id))
                .map_err(|_| InspectionError::StorageUnavailable)
            })
            .collect(),
        OperatorMutationKind::RotateCohort => {
            let generation_id = append_cohort_generation(
                transaction,
                project_uuid,
                &prepared.actor,
                &prepared.reason,
                created_at_unix_ms,
            )?;
            Ok(BTreeMap::from([(
                COHORT_SCOPE_KEY.to_string(),
                generation_id,
            )]))
        }
    }
}

fn append_cohort_generation(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    actor: &str,
    reason: &str,
    created_at_unix_ms: i64,
) -> Result<Uuid, InspectionError> {
    let mut bytes = Zeroizing::new([0_u8; COHORT_SALT_BYTES]);
    getrandom::fill(bytes.as_mut()).map_err(|_| InspectionError::StorageUnavailable)?;
    let salt = CohortSalt::new(bytes);
    let generation_id = Uuid::now_v7();
    let fingerprint = sha256_hex(salt.as_bytes());
    let payload_hash = hash_json(&json!({
        "cohort_generation_id": generation_id,
        "project_uuid": project_uuid,
        "assignment_algorithm": COHORT_ASSIGNMENT_ALGORITHM_V1,
        "salt_fingerprint_sha256": fingerprint,
        "actor": actor,
        "reason": reason,
        "created_at_unix_ms": created_at_unix_ms,
    }))
    .map_err(|_| InspectionError::StorageUnavailable)?;
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
                actor,
                reason,
                created_at_unix_ms,
                payload_hash,
            ],
        )
        .map_err(|_| InspectionError::StorageUnavailable)?;
    append_cohort_state_event(
        transaction,
        project_uuid,
        generation_id,
        actor,
        reason,
        created_at_unix_ms,
    )
    .map_err(|_| InspectionError::StorageUnavailable)?;
    Ok(generation_id)
}

#[allow(clippy::too_many_arguments)]
fn insert_operator_receipt(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    prepared: &PreparedOperatorMutation,
    result: OperatorMutationResultV1,
    prior_generations: &BTreeMap<String, Uuid>,
    resulting_generations: &BTreeMap<String, Uuid>,
    superseded_generations: &BTreeMap<String, Uuid>,
    created_at_unix_ms: i64,
) -> Result<OperatorMutationReceiptV1, InspectionError> {
    verify_receipt_semantics(
        prepared,
        result,
        prior_generations,
        resulting_generations,
        superseded_generations,
    )?;
    for (generation_kind, generations) in [
        (
            prepared.kind.generation_kind(),
            &prepared.expected_generations,
        ),
        (prepared.kind.generation_kind(), prior_generations),
        (prepared.kind.generation_kind(), resulting_generations),
        (prepared.kind.generation_kind(), superseded_generations),
    ] {
        for (scope_key, generation_id) in generations {
            verify_generation_reference(
                transaction,
                project_uuid,
                generation_kind,
                scope_key,
                *generation_id,
            )?;
        }
    }
    let operator_ordinal = next_operator_ordinal(transaction, project_uuid)?;
    let record_hash = operator_receipt_hash(
        operator_ordinal,
        project_uuid,
        process_instance_id,
        prepared,
        result,
        prior_generations,
        resulting_generations,
        superseded_generations,
        created_at_unix_ms,
    )?;
    let inserted = transaction
        .execute(
            "INSERT INTO operator_mutation_receipts (
                operator_ordinal, mutation_id, project_uuid, mutation_kind,
                scope_pool_id, canonical_request_hash, result, confirm_project_id,
                expected_generation_count, prior_generation_count,
                resulting_generation_count, superseded_generation_count,
                actor, reason, process_instance_id, created_at_unix_ms,
                record_hash, canonical_payload_hash
             ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
                ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?17
             )",
            params![
                i64::try_from(operator_ordinal).map_err(|_| InspectionError::CapacityExhausted)?,
                prepared.mutation_id.to_string(),
                project_uuid.to_string(),
                prepared.kind.as_str(),
                prepared.scope_pool_id,
                prepared.canonical_request_hash,
                result_name(result),
                prepared.confirm_project_id,
                i64::try_from(prepared.expected_generations.len())
                    .map_err(|_| InspectionError::CapacityExhausted)?,
                i64::try_from(prior_generations.len())
                    .map_err(|_| InspectionError::CapacityExhausted)?,
                i64::try_from(resulting_generations.len())
                    .map_err(|_| InspectionError::CapacityExhausted)?,
                i64::try_from(superseded_generations.len())
                    .map_err(|_| InspectionError::CapacityExhausted)?,
                prepared.actor,
                prepared.reason,
                process_instance_id.to_string(),
                created_at_unix_ms,
                record_hash,
            ],
        )
        .map_err(|_| InspectionError::StorageUnavailable)?;
    if inserted != 1 {
        return Err(InspectionError::IntegrityError);
    }
    for (role, generations) in [
        (EdgeRole::Expected, &prepared.expected_generations),
        (EdgeRole::Prior, prior_generations),
        (EdgeRole::Resulting, resulting_generations),
        (EdgeRole::Superseded, superseded_generations),
    ] {
        insert_generation_edges(
            transaction,
            project_uuid,
            prepared.mutation_id,
            prepared.kind.generation_kind(),
            role,
            generations,
        )?;
    }
    let history_inserted = transaction
        .execute(
            "INSERT INTO operator_history_entries (
                project_uuid, audit_id, entry_kind, control_mutation_id,
                operator_mutation_id, created_at_unix_ms
             ) VALUES (?1, ?2, 'operator', NULL, ?2, ?3)",
            params![
                project_uuid.to_string(),
                prepared.mutation_id.to_string(),
                created_at_unix_ms,
            ],
        )
        .map_err(|_| InspectionError::StorageUnavailable)?;
    if history_inserted != 1 {
        return Err(InspectionError::IntegrityError);
    }
    load_verified_receipt(transaction, project_uuid, prepared.mutation_id)?
        .map(|stored| stored.receipt)
        .ok_or(InspectionError::IntegrityError)
}

fn next_operator_ordinal(
    connection: &Connection,
    project_uuid: Uuid,
) -> Result<u64, InspectionError> {
    let maximum = connection
        .query_row(
            "SELECT max(operator_ordinal) FROM operator_mutation_receipts
             WHERE project_uuid = ?1",
            [project_uuid.to_string()],
            |row| row.get::<_, Option<i64>>(0),
        )
        .map_err(|_| InspectionError::StorageUnavailable)?
        .unwrap_or(0);
    u64::try_from(maximum)
        .ok()
        .and_then(|value| value.checked_add(1))
        .filter(|value| *value <= i64::MAX as u64)
        .ok_or(InspectionError::CapacityExhausted)
}

fn insert_generation_edges(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    mutation_id: Uuid,
    generation_kind: GenerationKind,
    role: EdgeRole,
    generations: &BTreeMap<String, Uuid>,
) -> Result<(), InspectionError> {
    for (ordinal, (scope_key, generation_id)) in generations.iter().enumerate() {
        let (pool_id, learning_generation_id, cohort_generation_id) = match generation_kind {
            GenerationKind::Learning => (
                Some(scope_key.as_str()),
                Some(generation_id.to_string()),
                None,
            ),
            GenerationKind::Cohort => (None, None, Some(generation_id.to_string())),
        };
        let inserted = transaction
            .execute(
                "INSERT INTO operator_mutation_generation_edges (
                    mutation_id, project_uuid, edge_role, edge_ordinal,
                    scope_key, generation_kind, pool_id,
                    learning_generation_id, cohort_generation_id
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                params![
                    mutation_id.to_string(),
                    project_uuid.to_string(),
                    role.as_str(),
                    i64::try_from(ordinal).map_err(|_| InspectionError::CapacityExhausted)?,
                    scope_key,
                    generation_kind.as_str(),
                    pool_id,
                    learning_generation_id,
                    cohort_generation_id,
                ],
            )
            .map_err(|_| InspectionError::StorageUnavailable)?;
        if inserted != 1 {
            return Err(InspectionError::IntegrityError);
        }
    }
    Ok(())
}

fn verify_receipt_semantics(
    prepared: &PreparedOperatorMutation,
    result: OperatorMutationResultV1,
    prior_generations: &BTreeMap<String, Uuid>,
    resulting_generations: &BTreeMap<String, Uuid>,
    superseded_generations: &BTreeMap<String, Uuid>,
) -> Result<(), InspectionError> {
    validate_prepared_shape(prepared)?;
    if prepared
        .expected_generations
        .keys()
        .ne(prior_generations.keys())
        || prior_generations.keys().ne(resulting_generations.keys())
    {
        return Err(InspectionError::IntegrityError);
    }
    match result {
        OperatorMutationResultV1::Applied => {
            if prepared.expected_generations != *prior_generations
                || superseded_generations != prior_generations
                || prior_generations.iter().any(|(scope, prior)| {
                    resulting_generations
                        .get(scope)
                        .is_none_or(|resulting| resulting == prior)
                })
            {
                return Err(InspectionError::IntegrityError);
            }
        }
        OperatorMutationResultV1::Conflict => {
            if prepared.expected_generations == *prior_generations
                || resulting_generations != prior_generations
                || !superseded_generations.is_empty()
            {
                return Err(InspectionError::IntegrityError);
            }
        }
    }
    Ok(())
}

fn validate_prepared_shape(prepared: &PreparedOperatorMutation) -> Result<(), InspectionError> {
    match prepared.kind {
        OperatorMutationKind::ResetPool => {
            let Some(pool_id) = prepared.scope_pool_id.as_deref() else {
                return Err(InspectionError::IntegrityError);
            };
            if prepared.expected_generations.len() != 1
                || !prepared.expected_generations.contains_key(pool_id)
            {
                return Err(InspectionError::IntegrityError);
            }
        }
        OperatorMutationKind::ResetAll => {
            if prepared.scope_pool_id.is_some() || prepared.expected_generations.is_empty() {
                return Err(InspectionError::IntegrityError);
            }
        }
        OperatorMutationKind::RotateCohort => {
            if prepared.scope_pool_id.is_some()
                || prepared.expected_generations.len() != 1
                || !prepared.expected_generations.contains_key(COHORT_SCOPE_KEY)
            {
                return Err(InspectionError::IntegrityError);
            }
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn operator_receipt_hash(
    operator_ordinal: u64,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    prepared: &PreparedOperatorMutation,
    result: OperatorMutationResultV1,
    prior_generations: &BTreeMap<String, Uuid>,
    resulting_generations: &BTreeMap<String, Uuid>,
    superseded_generations: &BTreeMap<String, Uuid>,
    created_at_unix_ms: i64,
) -> Result<String, InspectionError> {
    canonical_sha256(&json!({
        "schema": OPERATOR_RECEIPT_SCHEMA_V1,
        "operator_ordinal": operator_ordinal,
        "mutation_id": prepared.mutation_id,
        "project_uuid": project_uuid,
        "mutation_kind": prepared.kind.as_str(),
        "scope_pool_id": prepared.scope_pool_id,
        "canonical_request_hash": prepared.canonical_request_hash,
        "result": result_name(result),
        "confirm_project_id": prepared.confirm_project_id,
        "expected_generations": prepared.expected_generations,
        "prior_generations": prior_generations,
        "resulting_generations": resulting_generations,
        "superseded_ids": superseded_generations.values().collect::<Vec<_>>(),
        "actor": prepared.actor,
        "reason": prepared.reason,
        "process_instance_id": process_instance_id,
        "created_at_unix_ms": created_at_unix_ms,
    }))
    .map_err(|_| InspectionError::IntegrityError)
}

fn load_verified_receipt(
    connection: &Connection,
    expected_project_uuid: Uuid,
    mutation_id: Uuid,
) -> Result<Option<VerifiedStoredReceipt>, InspectionError> {
    let stored = connection
        .query_row(
            "SELECT operator_ordinal, mutation_id, project_uuid, mutation_kind,
                    scope_pool_id, canonical_request_hash, result, confirm_project_id,
                    expected_generation_count, prior_generation_count,
                    resulting_generation_count, superseded_generation_count,
                    actor, reason, process_instance_id, created_at_unix_ms,
                    record_hash, canonical_payload_hash
             FROM operator_mutation_receipts WHERE mutation_id = ?1",
            [mutation_id.to_string()],
            |row| {
                Ok(StoredReceipt {
                    operator_ordinal: row.get(0)?,
                    mutation_id: row.get(1)?,
                    project_uuid: row.get(2)?,
                    mutation_kind: row.get(3)?,
                    scope_pool_id: row.get(4)?,
                    canonical_request_hash: row.get(5)?,
                    result: row.get(6)?,
                    confirm_project_id: row.get(7)?,
                    expected_generation_count: row.get(8)?,
                    prior_generation_count: row.get(9)?,
                    resulting_generation_count: row.get(10)?,
                    superseded_generation_count: row.get(11)?,
                    actor: row.get(12)?,
                    reason: row.get(13)?,
                    process_instance_id: row.get(14)?,
                    created_at_unix_ms: row.get(15)?,
                    record_hash: row.get(16)?,
                    canonical_payload_hash: row.get(17)?,
                })
            },
        )
        .optional()
        .map_err(|_| InspectionError::StorageUnavailable)?;
    stored
        .map(|stored| verify_stored_receipt(connection, expected_project_uuid, stored))
        .transpose()
}

fn verify_stored_receipt(
    connection: &Connection,
    expected_project_uuid: Uuid,
    stored: StoredReceipt,
) -> Result<VerifiedStoredReceipt, InspectionError> {
    let operator_ordinal = positive_u64(stored.operator_ordinal)?;
    let mutation_id = parse_uuid_v7(&stored.mutation_id)?;
    let project_uuid = parse_uuid_v7(&stored.project_uuid)?;
    let process_instance_id = parse_uuid_v7(&stored.process_instance_id)?;
    let kind = OperatorMutationKind::parse(&stored.mutation_kind)?;
    let result = parse_result(&stored.result)?;
    if project_uuid != expected_project_uuid
        || stored.created_at_unix_ms < 0
        || !valid_sha256(&stored.canonical_request_hash)
        || !valid_sha256(&stored.record_hash)
        || stored.canonical_payload_hash != stored.record_hash
    {
        return Err(InspectionError::IntegrityError);
    }
    super::super::process::originating_process_is_live(
        connection,
        project_uuid,
        process_instance_id,
    )
    .map_err(|_| InspectionError::IntegrityError)?;

    let edges = load_stored_edges(connection, mutation_id)?;
    let expected = collect_role_edges(
        connection,
        project_uuid,
        kind.generation_kind(),
        EdgeRole::Expected,
        &edges,
    )?;
    let prior = collect_role_edges(
        connection,
        project_uuid,
        kind.generation_kind(),
        EdgeRole::Prior,
        &edges,
    )?;
    let resulting = collect_role_edges(
        connection,
        project_uuid,
        kind.generation_kind(),
        EdgeRole::Resulting,
        &edges,
    )?;
    let superseded = collect_role_edges(
        connection,
        project_uuid,
        kind.generation_kind(),
        EdgeRole::Superseded,
        &edges,
    )?;
    if checked_count(stored.expected_generation_count)? != expected.len()
        || checked_count(stored.prior_generation_count)? != prior.len()
        || checked_count(stored.resulting_generation_count)? != resulting.len()
        || checked_count(stored.superseded_generation_count)? != superseded.len()
        || edges.len() != expected.len() + prior.len() + resulting.len() + superseded.len()
    {
        return Err(InspectionError::IntegrityError);
    }
    let prepared = prepare_operator_mutation(
        mutation_id,
        kind,
        stored.scope_pool_id.clone(),
        expected,
        stored.confirm_project_id.clone(),
        stored.actor.clone(),
        stored.reason.clone(),
    )
    .map_err(|_| InspectionError::IntegrityError)?;
    if prepared.canonical_request_hash != stored.canonical_request_hash {
        return Err(InspectionError::IntegrityError);
    }
    verify_receipt_semantics(&prepared, result, &prior, &resulting, &superseded)?;
    let expected_record_hash = operator_receipt_hash(
        operator_ordinal,
        project_uuid,
        process_instance_id,
        &prepared,
        result,
        &prior,
        &resulting,
        &superseded,
        stored.created_at_unix_ms,
    )?;
    if expected_record_hash != stored.record_hash {
        return Err(InspectionError::IntegrityError);
    }
    verify_operator_history_index_row(
        connection,
        project_uuid,
        mutation_id,
        stored.created_at_unix_ms,
    )?;
    Ok(VerifiedStoredReceipt {
        prepared,
        receipt: OperatorMutationReceiptV1 {
            mutation_id,
            result,
            prior_generations: prior,
            resulting_generations: resulting,
            created_at_unix_ms: u64::try_from(stored.created_at_unix_ms)
                .map_err(|_| InspectionError::IntegrityError)?,
            record_hash: stored.record_hash,
        },
    })
}

fn verify_operator_history_index_row(
    connection: &Connection,
    project_uuid: Uuid,
    mutation_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), InspectionError> {
    let stored = connection
        .query_row(
            "SELECT project_uuid, audit_id, entry_kind, control_mutation_id,
                    operator_mutation_id, created_at_unix_ms
             FROM operator_history_entries WHERE operator_mutation_id = ?1",
            [mutation_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                    row.get::<_, i64>(5)?,
                ))
            },
        )
        .optional()
        .map_err(|_| InspectionError::StorageUnavailable)?
        .ok_or(InspectionError::IntegrityError)?;
    let mutation_id = mutation_id.to_string();
    if stored
        != (
            project_uuid.to_string(),
            mutation_id.clone(),
            "operator".to_string(),
            None,
            Some(mutation_id),
            created_at_unix_ms,
        )
    {
        return Err(InspectionError::IntegrityError);
    }
    Ok(())
}

fn load_stored_edges(
    connection: &Connection,
    mutation_id: Uuid,
) -> Result<Vec<StoredEdge>, InspectionError> {
    let mut statement = connection
        .prepare(
            "SELECT edge_role, edge_ordinal, scope_key, generation_kind,
                    pool_id, learning_generation_id, cohort_generation_id
             FROM operator_mutation_generation_edges
             WHERE mutation_id = ?1
             ORDER BY edge_role, edge_ordinal",
        )
        .map_err(|_| InspectionError::StorageUnavailable)?;
    statement
        .query_map([mutation_id.to_string()], |row| {
            Ok(StoredEdge {
                role: row.get(0)?,
                ordinal: row.get(1)?,
                scope_key: row.get(2)?,
                generation_kind: row.get(3)?,
                pool_id: row.get(4)?,
                learning_generation_id: row.get(5)?,
                cohort_generation_id: row.get(6)?,
            })
        })
        .map_err(|_| InspectionError::StorageUnavailable)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(|_| InspectionError::StorageUnavailable)
}

fn collect_role_edges(
    connection: &Connection,
    project_uuid: Uuid,
    expected_kind: GenerationKind,
    role: EdgeRole,
    edges: &[StoredEdge],
) -> Result<BTreeMap<String, Uuid>, InspectionError> {
    let selected = edges
        .iter()
        .filter(|edge| EdgeRole::parse(&edge.role) == Ok(role))
        .collect::<Vec<_>>();
    let mut generations = BTreeMap::new();
    let mut previous_scope_key: Option<&str> = None;
    for (expected_ordinal, edge) in selected.into_iter().enumerate() {
        if usize::try_from(edge.ordinal).ok() != Some(expected_ordinal)
            || edge.generation_kind != expected_kind.as_str()
            || generations.contains_key(&edge.scope_key)
            || previous_scope_key.is_some_and(|previous| previous >= edge.scope_key.as_str())
        {
            return Err(InspectionError::IntegrityError);
        }
        let generation_id = match expected_kind {
            GenerationKind::Learning => {
                if edge.pool_id.as_deref() != Some(edge.scope_key.as_str())
                    || edge.cohort_generation_id.is_some()
                {
                    return Err(InspectionError::IntegrityError);
                }
                parse_uuid_v7(
                    edge.learning_generation_id
                        .as_deref()
                        .ok_or(InspectionError::IntegrityError)?,
                )?
            }
            GenerationKind::Cohort => {
                if edge.scope_key != COHORT_SCOPE_KEY
                    || edge.pool_id.is_some()
                    || edge.learning_generation_id.is_some()
                {
                    return Err(InspectionError::IntegrityError);
                }
                parse_uuid_v7(
                    edge.cohort_generation_id
                        .as_deref()
                        .ok_or(InspectionError::IntegrityError)?,
                )?
            }
        };
        verify_generation_reference(
            connection,
            project_uuid,
            expected_kind,
            &edge.scope_key,
            generation_id,
        )?;
        previous_scope_key = Some(&edge.scope_key);
        generations.insert(edge.scope_key.clone(), generation_id);
    }
    Ok(generations)
}

fn verify_generation_reference(
    connection: &Connection,
    project_uuid: Uuid,
    generation_kind: GenerationKind,
    scope_key: &str,
    generation_id: Uuid,
) -> Result<(), InspectionError> {
    match generation_kind {
        GenerationKind::Learning => {
            verify_learning_generation(connection, project_uuid, scope_key, generation_id)
        }
        GenerationKind::Cohort => {
            if scope_key != COHORT_SCOPE_KEY {
                return Err(InspectionError::IntegrityError);
            }
            verify_cohort_generation(connection, project_uuid, generation_id)
        }
    }
}

fn verify_learning_generation(
    connection: &Connection,
    project_uuid: Uuid,
    pool_id: &str,
    generation_id: Uuid,
) -> Result<(), InspectionError> {
    let stored = connection
        .query_row(
            "SELECT g.learning_generation_id, g.actor, g.reason,
                    g.created_at_unix_ms, g.canonical_payload_hash,
                    s.learning_state_event_id, s.state, s.actor, s.reason,
                    s.created_at_unix_ms, s.canonical_payload_hash
             FROM learning_generations AS g
             JOIN learning_generation_state_events AS s
               ON s.project_uuid = g.project_uuid
              AND s.pool_id = g.pool_id
              AND s.learning_generation_id = g.learning_generation_id
             WHERE g.project_uuid = ?1 AND g.pool_id = ?2
               AND g.learning_generation_id = ?3",
            params![project_uuid.to_string(), pool_id, generation_id.to_string()],
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
                    row.get::<_, String>(8)?,
                    row.get::<_, i64>(9)?,
                    row.get::<_, String>(10)?,
                ))
            },
        )
        .optional()
        .map_err(|_| InspectionError::StorageUnavailable)?
        .ok_or(InspectionError::IntegrityError)?;
    let (
        stored_generation_id,
        actor,
        reason,
        created_at_unix_ms,
        stored_hash,
        state_event_id,
        state,
        state_actor,
        state_reason,
        state_created_at_unix_ms,
        state_stored_hash,
    ) = stored;
    let parsed_generation_id = parse_uuid_v7(&stored_generation_id)?;
    let state_event_id = parse_uuid_v7(&state_event_id)?;
    let expected_hash = canonical_sha256(&json!({
        "learning_generation_id": stored_generation_id,
        "project_uuid": project_uuid,
        "pool_id": pool_id,
        "actor": actor,
        "reason": reason,
        "created_at_unix_ms": created_at_unix_ms,
    }))
    .map_err(|_| InspectionError::IntegrityError)?;
    let expected_state_hash = canonical_sha256(&json!({
        "learning_state_event_id": state_event_id,
        "project_uuid": project_uuid,
        "pool_id": pool_id,
        "learning_generation_id": parsed_generation_id,
        "state": "current",
        "actor": state_actor,
        "reason": state_reason,
        "created_at_unix_ms": state_created_at_unix_ms,
    }))
    .map_err(|_| InspectionError::IntegrityError)?;
    if parsed_generation_id != generation_id
        || state != "current"
        || created_at_unix_ms < 0
        || state_created_at_unix_ms < 0
        || stored_hash != expected_hash
        || state_stored_hash != expected_state_hash
    {
        return Err(InspectionError::IntegrityError);
    }
    Ok(())
}

fn verify_cohort_generation(
    connection: &Connection,
    project_uuid: Uuid,
    generation_id: Uuid,
) -> Result<(), InspectionError> {
    let (verified_generation_id, _salt) =
        load_verified_cohort_generation(connection, project_uuid, &generation_id.to_string())
            .map_err(|_| InspectionError::IntegrityError)?;
    if verified_generation_id != generation_id {
        return Err(InspectionError::IntegrityError);
    }
    let stored = connection
        .query_row(
            "SELECT cohort_state_event_id, state, actor, reason,
                    created_at_unix_ms, canonical_payload_hash
             FROM cohort_generation_state_events
             WHERE project_uuid = ?1 AND cohort_generation_id = ?2",
            params![project_uuid.to_string(), generation_id.to_string()],
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
        .map_err(|_| InspectionError::StorageUnavailable)?
        .ok_or(InspectionError::IntegrityError)?;
    let (state_event_id, state, actor, reason, created_at_unix_ms, stored_hash) = stored;
    let state_event_id = parse_uuid_v7(&state_event_id)?;
    let expected_hash = canonical_sha256(&json!({
        "cohort_state_event_id": state_event_id,
        "project_uuid": project_uuid,
        "cohort_generation_id": generation_id,
        "state": "current",
        "actor": actor,
        "reason": reason,
        "created_at_unix_ms": created_at_unix_ms,
    }))
    .map_err(|_| InspectionError::IntegrityError)?;
    if state != "current" || created_at_unix_ms < 0 || stored_hash != expected_hash {
        return Err(InspectionError::IntegrityError);
    }
    Ok(())
}

fn result_name(result: OperatorMutationResultV1) -> &'static str {
    match result {
        OperatorMutationResultV1::Applied => "applied",
        OperatorMutationResultV1::Conflict => "conflict",
    }
}

fn parse_result(value: &str) -> Result<OperatorMutationResultV1, InspectionError> {
    match value {
        "applied" => Ok(OperatorMutationResultV1::Applied),
        "conflict" => Ok(OperatorMutationResultV1::Conflict),
        _ => Err(InspectionError::IntegrityError),
    }
}

fn parse_uuid_v7(value: &str) -> Result<Uuid, InspectionError> {
    let parsed = Uuid::parse_str(value).map_err(|_| InspectionError::IntegrityError)?;
    if !is_uuid_v7(parsed) || parsed.to_string() != value {
        return Err(InspectionError::IntegrityError);
    }
    Ok(parsed)
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn positive_u64(value: i64) -> Result<u64, InspectionError> {
    u64::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or(InspectionError::IntegrityError)
}

fn checked_count(value: i64) -> Result<usize, InspectionError> {
    usize::try_from(value).map_err(|_| InspectionError::IntegrityError)
}

#[cfg(test)]
mod tests {
    use chrono::Utc;
    use rusqlite::params;
    use tempfile::TempDir;

    use super::*;
    use crate::ledger::repository::{ActivatedLedger, ready_evaluated_active_runtime_fixture};

    #[derive(Clone)]
    struct TestAuthority {
        project_uuid: Uuid,
        process_instance_id: Uuid,
        project_id: String,
        pool_ids: Vec<String>,
        learning_generation_ids: BTreeMap<String, Uuid>,
        cohort_generation_id: Uuid,
    }

    fn fixture() -> (TempDir, ActivatedLedger, TestAuthority) {
        let (temporary, _config, activated, _space, _vector, _evidence, _evaluation) =
            ready_evaluated_active_runtime_fixture();
        let authority = TestAuthority {
            project_uuid: activated.identity.project_uuid,
            process_instance_id: activated.identity.process_instance_id,
            project_id: activated.identity.project_id.clone(),
            pool_ids: activated.identity.pools.keys().cloned().collect(),
            learning_generation_ids: activated
                .identity
                .pools
                .iter()
                .map(|(pool_id, pool)| (pool_id.clone(), pool.learning_generation_id))
                .collect(),
            cohort_generation_id: activated.identity.cohort_generation_id,
        };
        (temporary, activated, authority)
    }

    fn pool_reset(
        authority: &TestAuthority,
        mutation_id: Uuid,
        expected_generation_id: Uuid,
        reason: &str,
    ) -> PreparedOperatorMutation {
        prepare_learning_reset(LearningResetRequestV1 {
            mutation_id,
            scope: LearningResetScopeV1::Pool {
                pool_id: authority.pool_ids[0].clone(),
                expected_learning_generation_id: expected_generation_id,
            },
            confirm_project_id: authority.project_id.clone(),
            actor: "operator".to_string(),
            reason: reason.to_string(),
        })
        .unwrap()
    }

    fn all_reset(
        authority: &TestAuthority,
        mutation_id: Uuid,
        expected_generations: BTreeMap<String, Uuid>,
    ) -> PreparedOperatorMutation {
        prepare_learning_reset(LearningResetRequestV1 {
            mutation_id,
            scope: LearningResetScopeV1::All {
                expected_learning_generation_ids: expected_generations,
            },
            confirm_project_id: authority.project_id.clone(),
            actor: "operator".to_string(),
            reason: "reset all".to_string(),
        })
        .unwrap()
    }

    fn rotation(
        authority: &TestAuthority,
        mutation_id: Uuid,
        expected_generation_id: Uuid,
    ) -> PreparedOperatorMutation {
        prepare_cohort_rotation(CohortRotationRequestV1 {
            mutation_id,
            expected_cohort_generation_id: expected_generation_id,
            confirm_project_id: authority.project_id.clone(),
            actor: "operator".to_string(),
            reason: "rotate cohort".to_string(),
        })
        .unwrap()
    }

    fn apply_committed(
        activated: &mut ActivatedLedger,
        authority: &TestAuthority,
        prepared: &PreparedOperatorMutation,
        created_at_unix_ms: i64,
    ) -> Result<OperatorMutationAck, InspectionError> {
        apply_committed_with_capacity(
            activated,
            authority,
            prepared,
            created_at_unix_ms,
            OPERATOR_RECEIPT_GRAPH_MAX_ROWS,
        )
    }

    fn apply_committed_with_capacity(
        activated: &mut ActivatedLedger,
        authority: &TestAuthority,
        prepared: &PreparedOperatorMutation,
        created_at_unix_ms: i64,
        capacity: usize,
    ) -> Result<OperatorMutationAck, InspectionError> {
        let transaction = activated.repository.connection_mut().transaction().unwrap();
        let result = apply_operator_mutation_with_capacity(
            &transaction,
            authority.project_uuid,
            authority.process_instance_id,
            &authority.pool_ids,
            prepared,
            created_at_unix_ms,
            capacity,
        );
        match result {
            Ok(acknowledgement) => {
                transaction.commit().unwrap();
                Ok(acknowledgement)
            }
            Err(error) => {
                drop(transaction);
                Err(error)
            }
        }
    }

    fn current_authority(
        activated: &mut ActivatedLedger,
        authority: &TestAuthority,
    ) -> (BTreeMap<String, Uuid>, Uuid) {
        let transaction = activated.repository.connection_mut().transaction().unwrap();
        let current = load_current_generation_authority(
            &transaction,
            authority.project_uuid,
            &authority.pool_ids,
        )
        .unwrap();
        let result = (
            current.learning_generation_ids,
            current.cohort_generation_id,
        );
        transaction.commit().unwrap();
        result
    }

    fn table_count(activated: &ActivatedLedger, table: &str) -> i64 {
        assert!(
            table
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
        );
        activated
            .repository
            .connection
            .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                row.get(0)
            })
            .unwrap()
    }

    fn uuid_v7_at(unix_ms: u64, discriminator: u64) -> Uuid {
        assert!(unix_ms < (1_u64 << 48));
        let timestamp = unix_ms.to_be_bytes();
        let random = discriminator.to_be_bytes();
        let mut bytes = [0_u8; 16];
        bytes[..6].copy_from_slice(&timestamp[2..]);
        bytes[6] = 0x70 | (random[0] & 0x0f);
        bytes[7] = random[1];
        bytes[8] = 0x80 | (random[2] & 0x3f);
        bytes[9..].copy_from_slice(&random[1..]);
        Uuid::from_bytes(bytes)
    }

    #[test]
    fn request_preparation_is_strict_and_hashes_semantics_without_the_id() {
        let expected = Uuid::parse_str("018f0000-0000-7000-8000-000000000001").unwrap();
        let request = |mutation_id, actor: &str| LearningResetRequestV1 {
            mutation_id,
            scope: LearningResetScopeV1::Pool {
                pool_id: "pool-a".to_string(),
                expected_learning_generation_id: expected,
            },
            confirm_project_id: "project-a".to_string(),
            actor: actor.to_string(),
            reason: "reset evidence".to_string(),
        };
        let first = prepare_learning_reset(request(
            Uuid::parse_str("018f0000-0000-7000-8000-000000000002").unwrap(),
            "operator",
        ))
        .unwrap();
        let second = prepare_learning_reset(request(
            Uuid::parse_str("018f0000-0000-7000-8000-000000000003").unwrap(),
            "operator",
        ))
        .unwrap();
        assert_eq!(first.canonical_request_hash, second.canonical_request_hash);
        assert_ne!(first.mutation_id, second.mutation_id);
        assert!(valid_sha256(&first.canonical_request_hash));

        assert_eq!(
            prepare_learning_reset(request(Uuid::new_v4(), "operator")),
            Err(InspectionError::InvalidArgument)
        );
        assert_eq!(
            prepare_learning_reset(request(Uuid::now_v7(), "operator\nname")),
            Err(InspectionError::InvalidArgument)
        );
        assert_eq!(
            prepare_learning_reset(LearningResetRequestV1 {
                mutation_id: Uuid::now_v7(),
                scope: LearningResetScopeV1::All {
                    expected_learning_generation_ids: BTreeMap::new(),
                },
                confirm_project_id: "project-a".to_string(),
                actor: "operator".to_string(),
                reason: "reset evidence".to_string(),
            }),
            Err(InspectionError::InvalidArgument)
        );
    }

    #[test]
    fn pool_reset_applies_once_and_exact_replay_ignores_the_first_use_window() {
        let (_temporary, mut activated, authority) = fixture();
        let now = Utc::now().timestamp_millis();
        let prior = authority.learning_generation_ids[&authority.pool_ids[0]];
        let prepared = pool_reset(&authority, Uuid::now_v7(), prior, "reset pool");

        let applied = apply_committed(&mut activated, &authority, &prepared, now).unwrap();
        assert_eq!(applied.receipt.result, OperatorMutationResultV1::Applied);
        assert!(!applied.replayed);
        assert_eq!(
            applied.receipt.prior_generations,
            BTreeMap::from([(authority.pool_ids[0].clone(), prior,)])
        );
        let resulting = applied.receipt.resulting_generations[&authority.pool_ids[0]];
        assert_ne!(resulting, prior);
        assert_eq!(table_count(&activated, "operator_mutation_receipts"), 1);
        assert_eq!(
            table_count(&activated, "operator_mutation_generation_edges"),
            4
        );

        let replayed =
            apply_committed(&mut activated, &authority, &prepared, now + 600_001).unwrap();
        assert!(replayed.replayed);
        assert_eq!(replayed.receipt, applied.receipt);
        assert_eq!(table_count(&activated, "operator_mutation_receipts"), 1);

        let changed = pool_reset(&authority, prepared.mutation_id(), prior, "changed payload");
        assert_eq!(
            apply_committed(&mut activated, &authority, &changed, now),
            Err(InspectionError::InvalidArgument)
        );
        let (learning, cohort) = current_authority(&mut activated, &authority);
        assert_eq!(learning[&authority.pool_ids[0]], resulting);
        assert_eq!(cohort, authority.cohort_generation_id);
    }

    #[test]
    fn stale_pool_and_all_resets_write_conflict_receipts_without_advancing() {
        let (_temporary, mut activated, authority) = fixture();
        let now = Utc::now().timestamp_millis();
        let old = authority.learning_generation_ids[&authority.pool_ids[0]];
        let first = pool_reset(&authority, Uuid::now_v7(), old, "advance first");
        let first_ack = apply_committed(&mut activated, &authority, &first, now).unwrap();
        let current = first_ack.receipt.resulting_generations[&authority.pool_ids[0]];
        let generation_rows = table_count(&activated, "learning_generations");

        let stale = pool_reset(&authority, Uuid::now_v7(), old, "stale reset");
        let conflict = apply_committed(&mut activated, &authority, &stale, now).unwrap();
        assert_eq!(conflict.receipt.result, OperatorMutationResultV1::Conflict);
        assert_eq!(
            conflict.receipt.prior_generations,
            BTreeMap::from([(authority.pool_ids[0].clone(), current)])
        );
        assert_eq!(
            conflict.receipt.resulting_generations,
            conflict.receipt.prior_generations
        );
        assert_eq!(
            table_count(&activated, "learning_generations"),
            generation_rows
        );
        assert_eq!(table_count(&activated, "operator_mutation_receipts"), 2);
        assert_eq!(
            table_count(&activated, "operator_mutation_generation_edges"),
            7
        );

        let stale_all = all_reset(
            &authority,
            Uuid::now_v7(),
            authority.learning_generation_ids.clone(),
        );
        let all_conflict = apply_committed(&mut activated, &authority, &stale_all, now).unwrap();
        assert_eq!(
            all_conflict.receipt.result,
            OperatorMutationResultV1::Conflict
        );
        assert_eq!(
            table_count(&activated, "learning_generations"),
            generation_rows
        );
    }

    #[test]
    fn multi_pool_all_reset_is_atomic_and_pool_reset_preserves_other_pools() {
        let temporary = tempfile::tempdir().unwrap();
        let path = crate::ledger::repository::tests::database_path(&temporary);
        let mut config = crate::ledger::repository::tests::config(&path, "operator-multi-pool");
        let mut second = config.pools[0].clone();
        second.id = "pool-b".into();
        second.anchor_models = vec!["anchor-b".into()];
        for candidate in &mut second.candidates {
            candidate.id.push_str("-b");
            candidate.model.push_str("-b");
        }
        config.pools.push(second);
        let mut activated = crate::ledger::repository::LedgerRepository::activate(&config).unwrap();
        let identity = activated.identity.clone();
        let pool_ids = identity.pools.keys().cloned().collect::<Vec<_>>();
        let initial = identity
            .pools
            .iter()
            .map(|(pool_id, pool)| (pool_id.clone(), pool.learning_generation_id))
            .collect::<BTreeMap<_, _>>();
        let prepared = prepare_learning_reset(LearningResetRequestV1 {
            mutation_id: Uuid::now_v7(),
            scope: LearningResetScopeV1::All {
                expected_learning_generation_ids: initial.clone(),
            },
            confirm_project_id: identity.project_id.clone(),
            actor: "operator".into(),
            reason: "reset both pools".into(),
        })
        .unwrap();
        let transaction = activated.repository.connection_mut().transaction().unwrap();
        let all = apply_operator_mutation_in_transaction(
            &transaction,
            identity.project_uuid,
            identity.process_instance_id,
            &pool_ids,
            &prepared,
            Utc::now().timestamp_millis(),
        )
        .unwrap();
        transaction.commit().unwrap();
        assert_eq!(all.receipt.result, OperatorMutationResultV1::Applied);
        assert_eq!(all.receipt.resulting_generations.len(), 2);
        assert!(initial.iter().all(|(pool_id, generation_id)| {
            all.receipt.resulting_generations[pool_id] != *generation_id
        }));

        let pool_b_after_all = all.receipt.resulting_generations["pool-b"];
        let targeted = prepare_learning_reset(LearningResetRequestV1 {
            mutation_id: Uuid::now_v7(),
            scope: LearningResetScopeV1::Pool {
                pool_id: "pool-a".into(),
                expected_learning_generation_id: all.receipt.resulting_generations["pool-a"],
            },
            confirm_project_id: identity.project_id,
            actor: "operator".into(),
            reason: "reset only pool a".into(),
        })
        .unwrap();
        let transaction = activated.repository.connection_mut().transaction().unwrap();
        let pool = apply_operator_mutation_in_transaction(
            &transaction,
            identity.project_uuid,
            identity.process_instance_id,
            &pool_ids,
            &targeted,
            Utc::now().timestamp_millis(),
        )
        .unwrap();
        transaction.commit().unwrap();
        assert_eq!(pool.receipt.result, OperatorMutationResultV1::Applied);

        let transaction = activated.repository.connection_mut().transaction().unwrap();
        let current =
            load_current_generation_authority(&transaction, identity.project_uuid, &pool_ids)
                .unwrap();
        assert_eq!(current.learning_generation_ids["pool-b"], pool_b_after_all);
        assert_eq!(
            current.learning_generation_ids["pool-a"],
            pool.receipt.resulting_generations["pool-a"]
        );
        transaction.commit().unwrap();
    }

    #[test]
    fn concurrent_active_admission_and_reset_have_one_generation_winner() {
        let fixture = crate::ledger::repository::active::tests::fixture_with_max_canary_roots(128);
        let identity = fixture.activated.identity.clone();
        let admission = fixture.admission.clone();
        let config = fixture.config.clone();
        let database_path = config.database_path.clone();
        let mut admission_repository = fixture.activated.repository;
        let mut reset_repository = crate::ledger::repository::LedgerRepository::activate(&config)
            .unwrap()
            .repository;
        let reset_process_id = reset_repository.process_instance_id;
        let prepared = prepare_learning_reset(LearningResetRequestV1 {
            mutation_id: Uuid::now_v7(),
            scope: LearningResetScopeV1::Pool {
                pool_id: "pool-a".into(),
                expected_learning_generation_id: identity.pools["pool-a"].learning_generation_id,
            },
            confirm_project_id: identity.project_id,
            actor: "operator".into(),
            reason: "race active admission".into(),
        })
        .unwrap();
        let barrier = std::sync::Arc::new(std::sync::Barrier::new(3));
        let admission_barrier = barrier.clone();
        let admission_thread = std::thread::spawn(move || {
            admission_barrier.wait();
            admission_repository.admit_active_root(&admission).unwrap()
        });
        let reset_barrier = barrier.clone();
        let reset_thread = std::thread::spawn(move || {
            let fence = Arc::new(ControlTransactionFence::new(
                Instant::now() + std::time::Duration::from_secs(5),
            ));
            reset_barrier.wait();
            reset_repository
                .apply_operator_mutation_with_start_check(&prepared, &fence, || Some(()))
                .unwrap()
        });
        barrier.wait();
        let admission_ack = admission_thread.join().unwrap();
        let reset_ack = reset_thread.join().unwrap();
        assert!(matches!(
            reset_ack,
            OperatorMutationTransactionAck::Completed(OperatorMutationAck {
                receipt: OperatorMutationReceiptV1 {
                    result: OperatorMutationResultV1::Applied,
                    ..
                },
                ..
            })
        ));
        assert!(matches!(
            admission_ack,
            crate::ledger::repository::active::ActiveAdmissionAck::Applied(_)
                | crate::ledger::repository::active::ActiveAdmissionAck::AuthorityChanged
        ));

        let connection = rusqlite::Connection::open(database_path).unwrap();
        let root_count = connection
            .query_row("SELECT count(*) FROM active_root_windows", [], |row| {
                row.get::<_, i64>(0)
            })
            .unwrap();
        assert_eq!(
            root_count,
            i64::from(matches!(
                admission_ack,
                crate::ledger::repository::active::ActiveAdmissionAck::Applied(_)
            ))
        );
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM operator_mutation_receipts
                     WHERE process_instance_id = ?1 AND result = 'applied'",
                    [reset_process_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
    }

    #[test]
    fn all_reset_and_cohort_rotation_advance_only_their_generation_authority() {
        let (_temporary, mut activated, authority) = fixture();
        let now = Utc::now().timestamp_millis();
        let reset = all_reset(
            &authority,
            Uuid::now_v7(),
            authority.learning_generation_ids.clone(),
        );
        let reset_ack = apply_committed(&mut activated, &authority, &reset, now).unwrap();
        assert_eq!(reset_ack.receipt.result, OperatorMutationResultV1::Applied);
        let (after_reset_learning, after_reset_cohort) =
            current_authority(&mut activated, &authority);
        assert_eq!(
            after_reset_learning,
            reset_ack.receipt.resulting_generations
        );
        assert_eq!(after_reset_cohort, authority.cohort_generation_id);

        let rotate = rotation(&authority, Uuid::now_v7(), authority.cohort_generation_id);
        let rotation_ack = apply_committed(&mut activated, &authority, &rotate, now).unwrap();
        assert_eq!(
            rotation_ack.receipt.result,
            OperatorMutationResultV1::Applied
        );
        let new_cohort = rotation_ack.receipt.resulting_generations[COHORT_SCOPE_KEY];
        assert_ne!(new_cohort, authority.cohort_generation_id);
        let (after_rotation_learning, after_rotation_cohort) =
            current_authority(&mut activated, &authority);
        assert_eq!(after_rotation_learning, after_reset_learning);
        assert_eq!(after_rotation_cohort, new_cohort);

        let stale_rotate = rotation(&authority, Uuid::now_v7(), authority.cohort_generation_id);
        let conflict = apply_committed(&mut activated, &authority, &stale_rotate, now).unwrap();
        assert_eq!(conflict.receipt.result, OperatorMutationResultV1::Conflict);
        assert_eq!(
            conflict.receipt.prior_generations[COHORT_SCOPE_KEY],
            new_cohort
        );
        assert_eq!(table_count(&activated, "cohort_generations"), 2);
    }

    #[test]
    fn first_use_time_and_capacity_refusals_write_nothing() {
        let (_temporary, mut activated, authority) = fixture();
        let now = Utc::now().timestamp_millis();
        let prior = authority.learning_generation_ids[&authority.pool_ids[0]];
        let expired_id = uuid_v7_at(u64::try_from(now - 300_001).unwrap(), 1);
        let expired = pool_reset(&authority, expired_id, prior, "expired");
        assert_eq!(
            apply_committed(&mut activated, &authority, &expired, now),
            Err(InspectionError::MutationExpired)
        );
        assert_eq!(table_count(&activated, "operator_mutation_receipts"), 0);

        let capacity = pool_reset(&authority, Uuid::now_v7(), prior, "capacity");
        let learning_rows = table_count(&activated, "learning_generations");
        assert_eq!(
            apply_committed_with_capacity(&mut activated, &authority, &capacity, now, 4),
            Err(InspectionError::CapacityExhausted)
        );
        assert_eq!(table_count(&activated, "operator_mutation_receipts"), 0);
        assert_eq!(
            table_count(&activated, "learning_generations"),
            learning_rows
        );
        let (learning, cohort) = current_authority(&mut activated, &authority);
        assert_eq!(learning, authority.learning_generation_ids);
        assert_eq!(cohort, authority.cohort_generation_id);
    }

    #[test]
    fn late_receipt_failure_rolls_back_generation_and_header_rows() {
        let (_temporary, mut activated, authority) = fixture();
        let now = Utc::now().timestamp_millis();
        let prior = authority.learning_generation_ids[&authority.pool_ids[0]];
        let prepared = pool_reset(&authority, Uuid::now_v7(), prior, "late failure");
        activated
            .repository
            .connection
            .execute_batch(
                "CREATE TRIGGER test_operator_edge_abort
                 BEFORE INSERT ON operator_mutation_generation_edges
                 BEGIN
                    SELECT RAISE(ABORT, 'injected edge failure');
                 END;",
            )
            .unwrap();
        let learning_rows = table_count(&activated, "learning_generations");
        assert_eq!(
            apply_committed(&mut activated, &authority, &prepared, now),
            Err(InspectionError::StorageUnavailable)
        );
        assert_eq!(table_count(&activated, "operator_mutation_receipts"), 0);
        assert_eq!(
            table_count(&activated, "operator_mutation_generation_edges"),
            0
        );
        assert_eq!(
            table_count(&activated, "learning_generations"),
            learning_rows
        );
        activated
            .repository
            .connection
            .execute_batch("DROP TRIGGER test_operator_edge_abort;")
            .unwrap();
    }

    #[test]
    fn replay_rejects_header_edge_and_generation_corruption() {
        let (_temporary, mut activated, authority) = fixture();
        let now = Utc::now().timestamp_millis();
        let prior = authority.learning_generation_ids[&authority.pool_ids[0]];
        let prepared = pool_reset(&authority, Uuid::now_v7(), prior, "corruption");
        let applied = apply_committed(&mut activated, &authority, &prepared, now).unwrap();
        let resulting = applied.receipt.resulting_generations[&authority.pool_ids[0]];
        let original_record_hash = applied.receipt.record_hash.clone();
        let original_generation_hash: String = activated
            .repository
            .connection
            .query_row(
                "SELECT canonical_payload_hash FROM learning_generations
                 WHERE learning_generation_id = ?1",
                [prior.to_string()],
                |row| row.get(0),
            )
            .unwrap();

        let transaction = activated.repository.connection_mut().transaction().unwrap();
        let assert_corrupt = |transaction: &Transaction<'_>| {
            assert_eq!(
                apply_operator_mutation_with_capacity(
                    transaction,
                    authority.project_uuid,
                    authority.process_instance_id,
                    &authority.pool_ids,
                    &prepared,
                    now,
                    OPERATOR_RECEIPT_GRAPH_MAX_ROWS,
                ),
                Err(InspectionError::IntegrityError)
            );
        };

        transaction
            .execute(
                "UPDATE operator_mutation_receipts
                 SET record_hash = ?1, canonical_payload_hash = ?1
                 WHERE mutation_id = ?2",
                params!["0".repeat(64), prepared.mutation_id().to_string()],
            )
            .unwrap();
        assert_corrupt(&transaction);
        transaction
            .execute(
                "UPDATE operator_mutation_receipts
                 SET record_hash = ?1, canonical_payload_hash = ?1
                 WHERE mutation_id = ?2",
                params![original_record_hash, prepared.mutation_id().to_string()],
            )
            .unwrap();

        transaction
            .execute(
                "UPDATE operator_mutation_generation_edges SET edge_ordinal = 1
                 WHERE mutation_id = ?1 AND edge_role = 'expected'",
                [prepared.mutation_id().to_string()],
            )
            .unwrap();
        assert_corrupt(&transaction);
        transaction
            .execute(
                "UPDATE operator_mutation_generation_edges SET edge_ordinal = 0
                 WHERE mutation_id = ?1 AND edge_role = 'expected'",
                [prepared.mutation_id().to_string()],
            )
            .unwrap();

        transaction
            .execute(
                "UPDATE operator_mutation_generation_edges
                 SET learning_generation_id = ?1
                 WHERE mutation_id = ?2 AND edge_role = 'expected'",
                params![resulting.to_string(), prepared.mutation_id().to_string()],
            )
            .unwrap();
        assert_corrupt(&transaction);
        transaction
            .execute(
                "UPDATE operator_mutation_generation_edges
                 SET learning_generation_id = ?1
                 WHERE mutation_id = ?2 AND edge_role = 'expected'",
                params![prior.to_string(), prepared.mutation_id().to_string()],
            )
            .unwrap();

        transaction
            .execute(
                "UPDATE learning_generations SET canonical_payload_hash = ?1
                 WHERE learning_generation_id = ?2",
                params!["0".repeat(64), prior.to_string()],
            )
            .unwrap();
        assert_corrupt(&transaction);
        transaction
            .execute(
                "UPDATE learning_generations SET canonical_payload_hash = ?1
                 WHERE learning_generation_id = ?2",
                params![original_generation_hash, prior.to_string()],
            )
            .unwrap();

        assert!(
            apply_operator_mutation_with_capacity(
                &transaction,
                authority.project_uuid,
                authority.process_instance_id,
                &authority.pool_ids,
                &prepared,
                now,
                OPERATOR_RECEIPT_GRAPH_MAX_ROWS,
            )
            .unwrap()
            .replayed
        );
        transaction.commit().unwrap();
    }
}
