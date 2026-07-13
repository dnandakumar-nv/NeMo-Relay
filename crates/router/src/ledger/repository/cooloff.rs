// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transactional, project-wide dependency cool-off state.

use std::collections::BTreeSet;

use nemo_relay::api::llm::LlmApiFamily;
use rusqlite::{Connection, OptionalExtension, TransactionBehavior, params};
use serde_json::{Value as Json, json};
use uuid::Uuid;

use super::process::{append_integrity_health, originating_process_is_live};
use super::shadow::{verified_shadow_dependency_context, verified_shadow_judge_policy};
use super::{LedgerRepository, TransactionStartGuard, map_fs_error, map_sqlite_error};
use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::ledger::fs::enforce_sidecar_permissions;
use crate::ledger::model::{LedgerError, LedgerErrorClass};

const INTEGRITY_CONFLICT_CLASS: &str = "router.ledger.integrity_conflict";

/// Canonical candidate-provider dependency identity.
///
/// This type deliberately omits `Debug` so transport identity is not copied into
/// an error or diagnostic accidentally.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct CandidateDependencyIdentity(PreparedDependencyIdentity);

/// Canonical judge-provider dependency identity.
///
/// Candidate identity is deliberately absent so one judge dependency is shared
/// across candidates that use the same reviewed judge contract.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct JudgeDependencyIdentity(PreparedDependencyIdentity);

#[derive(Clone, PartialEq, Eq)]
struct PreparedDependencyIdentity {
    key_kind: DependencyKeyKind,
    canonical_identity_json: String,
    dependency_key_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DependencyKeyKind {
    Candidate,
    Judge,
}

impl DependencyKeyKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::Judge => "judge",
        }
    }

    fn from_str(value: &str) -> Option<Self> {
        match value {
            "candidate" => Some(Self::Candidate),
            "judge" => Some(Self::Judge),
            _ => None,
        }
    }
}

impl CandidateDependencyIdentity {
    /// Build the exact version-1 candidate dependency key.
    pub(crate) fn new(
        api_family: LlmApiFamily,
        transport_identity: impl Into<String>,
        candidate_model: impl Into<String>,
        candidate_model_revision: impl Into<String>,
    ) -> Result<Self, LedgerError> {
        let transport_identity = transport_identity.into();
        let candidate_model = candidate_model.into();
        let candidate_model_revision = candidate_model_revision.into();
        validate_bounded_text(&transport_identity, 256)?;
        validate_bounded_text(&candidate_model, 512)?;
        validate_bounded_text(&candidate_model_revision, 128)?;
        Self::prepare(json!({
            "api_family": api_family,
            "transport_identity": transport_identity,
            "candidate_model": candidate_model,
            "candidate_model_revision": candidate_model_revision,
        }))
    }

    fn prepare(value: Json) -> Result<Self, LedgerError> {
        Ok(Self(prepare_identity(DependencyKeyKind::Candidate, value)?))
    }

    /// Return the lowercase SHA-256 dependency key.
    pub(crate) fn dependency_key_id(&self) -> &str {
        &self.0.dependency_key_id
    }
}

impl JudgeDependencyIdentity {
    /// Build the exact version-1 judge dependency key.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        api_family: LlmApiFamily,
        transport_identity: impl Into<String>,
        judge_model: impl Into<String>,
        judge_model_revision: impl Into<String>,
        prompt_version: impl Into<String>,
        prompt_sha256: impl Into<String>,
        rubric_version: impl Into<String>,
        rubric_sha256: impl Into<String>,
        output_schema_version: u32,
        output_schema_sha256: impl Into<String>,
    ) -> Result<Self, LedgerError> {
        let transport_identity = transport_identity.into();
        let judge_model = judge_model.into();
        let judge_model_revision = judge_model_revision.into();
        let prompt_version = prompt_version.into();
        let prompt_sha256 = prompt_sha256.into();
        let rubric_version = rubric_version.into();
        let rubric_sha256 = rubric_sha256.into();
        let output_schema_sha256 = output_schema_sha256.into();
        validate_bounded_text(&transport_identity, 256)?;
        validate_bounded_text(&judge_model, 512)?;
        validate_bounded_text(&judge_model_revision, 128)?;
        validate_bounded_text(&prompt_version, 128)?;
        validate_sha256(&prompt_sha256)?;
        validate_bounded_text(&rubric_version, 128)?;
        validate_sha256(&rubric_sha256)?;
        if output_schema_version != 1 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        validate_sha256(&output_schema_sha256)?;
        Self::prepare(json!({
            "api_family": api_family,
            "transport_identity": transport_identity,
            "judge_model": judge_model,
            "judge_model_revision": judge_model_revision,
            "prompt_version": prompt_version,
            "prompt_sha256": prompt_sha256,
            "rubric_version": rubric_version,
            "rubric_sha256": rubric_sha256,
            "output_schema_version": output_schema_version,
            "output_schema_sha256": output_schema_sha256,
        }))
    }

    fn prepare(value: Json) -> Result<Self, LedgerError> {
        Ok(Self(prepare_identity(DependencyKeyKind::Judge, value)?))
    }

    /// Return the lowercase SHA-256 dependency key.
    pub(crate) fn dependency_key_id(&self) -> &str {
        &self.0.dependency_key_id
    }
}

fn prepare_identity(
    key_kind: DependencyKeyKind,
    value: Json,
) -> Result<PreparedDependencyIdentity, LedgerError> {
    let canonical_identity_json = canonical_json(&value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let dependency_key_id = canonical_sha256(&value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    Ok(PreparedDependencyIdentity {
        key_kind,
        canonical_identity_json,
        dependency_key_id,
    })
}

/// Immutable operation claimed against one dependency key.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DependencyOperation {
    pub(crate) dependency_operation_id: Uuid,
    pub(crate) dependency_state_event_id: Uuid,
    pub(crate) anchor_id: Uuid,
    pub(crate) shadow_attempt_id: Uuid,
    pub(crate) base_cooloff_seconds: i64,
    pub(crate) max_cooloff_seconds: i64,
    pub(crate) created_at_unix_ms: i64,
}

impl DependencyOperation {
    /// Construct one claim with all idempotency material frozen before enqueueing.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        dependency_operation_id: Uuid,
        dependency_state_event_id: Uuid,
        anchor_id: Uuid,
        shadow_attempt_id: Uuid,
        base_cooloff_seconds: u64,
        max_cooloff_seconds: u64,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(dependency_operation_id)?;
        validate_uuid_v7(dependency_state_event_id)?;
        validate_uuid_v7(anchor_id)?;
        validate_uuid_v7(shadow_attempt_id)?;
        let base_cooloff_seconds = i64::try_from(base_cooloff_seconds)
            .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let max_cooloff_seconds = i64::try_from(max_cooloff_seconds)
            .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if created_at_unix_ms < 0
            || base_cooloff_seconds <= 0
            || max_cooloff_seconds < base_cooloff_seconds
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            dependency_operation_id,
            dependency_state_event_id,
            anchor_id,
            shadow_attempt_id,
            base_cooloff_seconds,
            max_cooloff_seconds,
            created_at_unix_ms,
        })
    }
}

/// Validated stable provider failure class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DependencyFailureClass(String);

impl DependencyFailureClass {
    /// Validate the stable class before it reaches a writer command.
    pub(crate) fn new(value: impl Into<String>) -> Result<Self, LedgerError> {
        let value = value.into();
        if !matches!(
            value.as_str(),
            "router.provider.transport"
                | "router.provider.authentication"
                | "router.provider.rate_limited"
                | "router.provider.timeout"
                | "router.provider.canceled"
                | "router.provider.unreadable_response"
                | "router.provider.truncated_response"
                | "router.provider.ambiguous_decode"
                | "router.provider.evidence_bound_exceeded"
                | "router.provider.middleware_interference"
                | "router.judge.output_invalid"
        ) {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self(value))
    }

    /// Borrow the validated stable class.
    pub(crate) fn as_str(&self) -> &str {
        &self.0
    }
}

/// Frozen completion of one admitted dependency operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DependencyCompletion {
    pub(crate) dependency_operation_id: Uuid,
    pub(crate) dependency_state_event_id: Uuid,
    pub(crate) created_at_unix_ms: i64,
    outcome: DependencyCompletionOutcome,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum DependencyCompletionOutcome {
    Success,
    Failure(DependencyFailureClass),
}

impl DependencyCompletion {
    /// Freeze a successful dependency completion.
    pub(crate) fn success(
        dependency_operation_id: Uuid,
        dependency_state_event_id: Uuid,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        Self::new(
            dependency_operation_id,
            dependency_state_event_id,
            created_at_unix_ms,
            DependencyCompletionOutcome::Success,
        )
    }

    /// Freeze an operational dependency failure.
    pub(crate) fn failure(
        dependency_operation_id: Uuid,
        dependency_state_event_id: Uuid,
        created_at_unix_ms: i64,
        failure_class: DependencyFailureClass,
    ) -> Result<Self, LedgerError> {
        Self::new(
            dependency_operation_id,
            dependency_state_event_id,
            created_at_unix_ms,
            DependencyCompletionOutcome::Failure(failure_class),
        )
    }

    fn new(
        dependency_operation_id: Uuid,
        dependency_state_event_id: Uuid,
        created_at_unix_ms: i64,
        outcome: DependencyCompletionOutcome,
    ) -> Result<Self, LedgerError> {
        validate_uuid_v7(dependency_operation_id)?;
        validate_uuid_v7(dependency_state_event_id)?;
        if created_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            dependency_operation_id,
            dependency_state_event_id,
            created_at_unix_ms,
            outcome,
        })
    }
}

/// Durable dependency transition returned by a command acknowledgement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DependencyTransition {
    Admitted,
    SkippedCooloff,
    Success,
    Failure,
    OrphanedInFlight,
}

impl DependencyTransition {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Admitted => "admitted",
            Self::SkippedCooloff => "skipped_cooloff",
            Self::Success => "success",
            Self::Failure => "failure",
            Self::OrphanedInFlight => "orphaned_in_flight",
        }
    }
}

/// Secret-free durable state returned to the scheduler/evaluator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DependencyStateSnapshot {
    pub(crate) dependency_key_id: String,
    pub(crate) dependency_operation_id: Uuid,
    pub(crate) transition: DependencyTransition,
    pub(crate) consecutive_failures: i64,
    pub(crate) cooloff_until_unix_ms: Option<i64>,
    pub(crate) failure_class: Option<String>,
}

/// Exhaustive stable acknowledgement for a dependency writer command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DependencyCommandAck {
    Applied(DependencyStateSnapshot),
    AlreadyApplied(DependencyStateSnapshot),
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

enum DomainWrite {
    Applied(DependencyStateSnapshot),
    AlreadyApplied(DependencyStateSnapshot),
    Conflict(ConflictContext),
}

#[derive(Clone)]
struct ConflictContext {
    anchor_id: Option<Uuid>,
    dependency_key_id: Option<String>,
}

#[derive(Debug)]
struct StoredOperation {
    dependency_operation_id: String,
    dependency_key_id: String,
    project_uuid: String,
    process_instance_id: String,
    anchor_id: String,
    shadow_attempt_id: String,
    base_cooloff_seconds: i64,
    max_cooloff_seconds: i64,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

struct StoredDependencyKey {
    project_uuid: String,
    key_kind: String,
    canonical_identity_json: String,
    canonical_payload_hash: String,
}

#[derive(Debug, Clone)]
struct StoredState {
    state_event_id: String,
    dependency_key_id: String,
    dependency_operation_id: Option<String>,
    anchor_id: Option<String>,
    transition: DependencyTransition,
    consecutive_failures: i64,
    cooloff_until_unix_ms: Option<i64>,
    failure_class: Option<String>,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

struct VerifiedStateChain {
    latest: Option<StoredState>,
}

impl LedgerRepository {
    /// Atomically admit or cool-off-skip one candidate dependency operation.
    pub(crate) fn claim_candidate_dependency(
        &mut self,
        identity: &CandidateDependencyIdentity,
        operation: &DependencyOperation,
    ) -> Result<DependencyCommandAck, LedgerError> {
        self.claim_dependency(&identity.0, operation, || Some(()))
    }

    /// Claim a candidate dependency after a final caller-owned start check.
    pub(crate) fn claim_candidate_dependency_with_start_check<G: TransactionStartGuard>(
        &mut self,
        identity: &CandidateDependencyIdentity,
        operation: &DependencyOperation,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<DependencyCommandAck, LedgerError> {
        self.claim_dependency(&identity.0, operation, start_check)
    }

    /// Atomically admit or cool-off-skip one judge dependency operation.
    pub(crate) fn claim_judge_dependency(
        &mut self,
        identity: &JudgeDependencyIdentity,
        operation: &DependencyOperation,
    ) -> Result<DependencyCommandAck, LedgerError> {
        self.claim_dependency(&identity.0, operation, || Some(()))
    }

    /// Claim a judge dependency after a final caller-owned start check.
    pub(crate) fn claim_judge_dependency_with_start_check<G: TransactionStartGuard>(
        &mut self,
        identity: &JudgeDependencyIdentity,
        operation: &DependencyOperation,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<DependencyCommandAck, LedgerError> {
        self.claim_dependency(&identity.0, operation, start_check)
    }

    fn claim_dependency<G: TransactionStartGuard>(
        &mut self,
        identity: &PreparedDependencyIdentity,
        operation: &DependencyOperation,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<DependencyCommandAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(DependencyCommandAck::TransactionNotStarted);
        };
        let mut transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(DependencyCommandAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(DependencyCommandAck::TransactionNotStarted);
        }
        drop(start_guard);
        if !originating_process_is_live(&transaction, project_uuid, process_instance_id)? {
            return Ok(DependencyCommandAck::OriginatingProcessNotLive);
        }

        let domain_write = {
            let mut savepoint = transaction.savepoint().map_err(database_error)?;
            let result = claim_in_savepoint(
                &savepoint,
                project_uuid,
                process_instance_id,
                identity,
                operation,
            )?;
            match result {
                DomainWrite::Conflict(_) => {
                    savepoint.rollback().map_err(database_error)?;
                    savepoint.commit().map_err(database_error)?;
                }
                DomainWrite::Applied(_) | DomainWrite::AlreadyApplied(_) => {
                    savepoint.commit().map_err(database_error)?;
                }
            }
            result
        };
        let acknowledgement = finish_domain_write(
            &transaction,
            project_uuid,
            process_instance_id,
            operation.dependency_state_event_id,
            operation.created_at_unix_ms,
            domain_write,
        )?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }

    /// Complete one admitted operation, ordered by committed event sequence.
    pub(crate) fn complete_dependency(
        &mut self,
        completion: &DependencyCompletion,
    ) -> Result<DependencyCommandAck, LedgerError> {
        self.complete_dependency_with_start_check(completion, || Some(()))
    }

    /// Complete a dependency after a final caller-owned start check.
    pub(crate) fn complete_dependency_with_start_check<G: TransactionStartGuard>(
        &mut self,
        completion: &DependencyCompletion,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<DependencyCommandAck, LedgerError> {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(DependencyCommandAck::TransactionNotStarted);
        };
        let mut transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(DependencyCommandAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(DependencyCommandAck::TransactionNotStarted);
        }
        drop(start_guard);
        if !originating_process_is_live(&transaction, project_uuid, process_instance_id)? {
            return Ok(DependencyCommandAck::OriginatingProcessNotLive);
        }

        let domain_write = {
            let mut savepoint = transaction.savepoint().map_err(database_error)?;
            let result =
                complete_in_savepoint(&savepoint, project_uuid, process_instance_id, completion)?;
            match result {
                DomainWrite::Conflict(_) => {
                    savepoint.rollback().map_err(database_error)?;
                    savepoint.commit().map_err(database_error)?;
                }
                DomainWrite::Applied(_) | DomainWrite::AlreadyApplied(_) => {
                    savepoint.commit().map_err(database_error)?;
                }
            }
            result
        };
        let acknowledgement = finish_domain_write(
            &transaction,
            project_uuid,
            process_instance_id,
            completion.dependency_state_event_id,
            completion.created_at_unix_ms,
            domain_write,
        )?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(acknowledgement)
    }
}

fn finish_domain_write(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    health_event_id: Uuid,
    created_at_unix_ms: i64,
    domain_write: DomainWrite,
) -> Result<DependencyCommandAck, LedgerError> {
    match domain_write {
        DomainWrite::Applied(snapshot) => Ok(DependencyCommandAck::Applied(snapshot)),
        DomainWrite::AlreadyApplied(snapshot) => Ok(DependencyCommandAck::AlreadyApplied(snapshot)),
        DomainWrite::Conflict(context) => {
            append_integrity_health(
                connection,
                health_event_id,
                project_uuid,
                process_instance_id,
                context.anchor_id,
                context.dependency_key_id.as_deref(),
                created_at_unix_ms,
            )?;
            Ok(DependencyCommandAck::Conflict)
        }
    }
}

fn claim_in_savepoint(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    identity: &PreparedDependencyIdentity,
    operation: &DependencyOperation,
) -> Result<DomainWrite, LedgerError> {
    let conflict = || {
        DomainWrite::Conflict(ConflictContext {
            anchor_id: Some(operation.anchor_id),
            dependency_key_id: Some(identity.dependency_key_id.clone()),
        })
    };
    if !dependency_identity_matches_shadow(
        connection,
        project_uuid,
        process_instance_id,
        identity,
        operation,
    )? {
        return Ok(conflict());
    }
    if !insert_or_verify_dependency_key(connection, project_uuid, identity)? {
        return Ok(conflict());
    }
    let operation_hash = operation_hash(project_uuid, process_instance_id, identity, operation)?;
    let operation_inserted = connection
        .execute(
            "INSERT INTO dependency_operations (
                dependency_operation_id, dependency_key_id, project_uuid,
                process_instance_id, anchor_id, shadow_attempt_id,
                base_cooloff_seconds, max_cooloff_seconds,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(dependency_operation_id) DO NOTHING",
            params![
                operation.dependency_operation_id.to_string(),
                identity.dependency_key_id,
                project_uuid.to_string(),
                process_instance_id.to_string(),
                operation.anchor_id.to_string(),
                operation.shadow_attempt_id.to_string(),
                operation.base_cooloff_seconds,
                operation.max_cooloff_seconds,
                operation.created_at_unix_ms,
                operation_hash,
            ],
        )
        .map_err(database_error)?;
    if operation_inserted == 0 {
        let stored = load_operation(connection, operation.dependency_operation_id)?;
        if stored.as_ref().is_none_or(|stored| {
            !operation_matches(
                stored,
                project_uuid,
                process_instance_id,
                identity,
                operation,
                &operation_hash,
            )
        }) {
            return Ok(conflict());
        }
        if verify_state_chain(connection, project_uuid, identity)?.is_none() {
            return Ok(conflict());
        }
        let Some(state) = load_claim_state(connection, operation.dependency_operation_id)? else {
            return Ok(conflict());
        };
        if !claim_state_matches(&state, identity, operation)? {
            return Ok(conflict());
        }
        return Ok(DomainWrite::AlreadyApplied(snapshot_from_state(&state)?));
    }

    let Some(chain) = verify_state_chain(connection, project_uuid, identity)? else {
        return Ok(conflict());
    };
    let latest = chain.latest;
    if latest
        .as_ref()
        .is_some_and(|latest| operation.created_at_unix_ms < latest.created_at_unix_ms)
    {
        return Ok(conflict());
    }
    let (transition, consecutive_failures, cooloff_until_unix_ms) = match latest {
        Some(state)
            if state
                .cooloff_until_unix_ms
                .is_some_and(|deadline| deadline > operation.created_at_unix_ms) =>
        {
            (
                DependencyTransition::SkippedCooloff,
                state.consecutive_failures,
                state.cooloff_until_unix_ms,
            )
        }
        Some(state) => (
            DependencyTransition::Admitted,
            state.consecutive_failures,
            state.cooloff_until_unix_ms,
        ),
        None => (DependencyTransition::Admitted, 0, None),
    };
    if transition == DependencyTransition::SkippedCooloff && consecutive_failures <= 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let state_hash = state_hash(
        operation.dependency_state_event_id,
        &identity.dependency_key_id,
        Some(operation.dependency_operation_id),
        Some(operation.anchor_id),
        transition,
        consecutive_failures,
        cooloff_until_unix_ms,
        None,
        operation.created_at_unix_ms,
    )?;
    let inserted = insert_state(
        connection,
        operation.dependency_state_event_id,
        &identity.dependency_key_id,
        operation.dependency_operation_id,
        operation.anchor_id,
        transition,
        consecutive_failures,
        cooloff_until_unix_ms,
        None,
        operation.created_at_unix_ms,
        &state_hash,
    )?;
    if !inserted {
        return Ok(conflict());
    }
    Ok(DomainWrite::Applied(DependencyStateSnapshot {
        dependency_key_id: identity.dependency_key_id.clone(),
        dependency_operation_id: operation.dependency_operation_id,
        transition,
        consecutive_failures,
        cooloff_until_unix_ms,
        failure_class: None,
    }))
}

fn complete_in_savepoint(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    completion: &DependencyCompletion,
) -> Result<DomainWrite, LedgerError> {
    let Some(operation) = load_operation(connection, completion.dependency_operation_id)? else {
        return Ok(DomainWrite::Conflict(ConflictContext {
            anchor_id: None,
            dependency_key_id: None,
        }));
    };
    let context = ConflictContext {
        anchor_id: Uuid::parse_str(&operation.anchor_id).ok(),
        dependency_key_id: Some(operation.dependency_key_id.clone()),
    };
    if operation.project_uuid != project_uuid.to_string()
        || operation.process_instance_id != process_instance_id.to_string()
        || !stored_operation_hash_is_valid(&operation, completion.dependency_operation_id)?
    {
        return Ok(DomainWrite::Conflict(context));
    }
    let Some(identity) =
        load_verified_dependency_key(connection, project_uuid, &operation.dependency_key_id)?
    else {
        return Ok(DomainWrite::Conflict(context));
    };
    let shadow_operation = DependencyOperation {
        dependency_operation_id: completion.dependency_operation_id,
        dependency_state_event_id: completion.dependency_state_event_id,
        anchor_id: Uuid::parse_str(&operation.anchor_id)
            .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
        shadow_attempt_id: Uuid::parse_str(&operation.shadow_attempt_id)
            .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
        base_cooloff_seconds: operation.base_cooloff_seconds,
        max_cooloff_seconds: operation.max_cooloff_seconds,
        created_at_unix_ms: operation.created_at_unix_ms,
    };
    if !dependency_identity_matches_shadow(
        connection,
        project_uuid,
        process_instance_id,
        &identity,
        &shadow_operation,
    )? {
        return Ok(DomainWrite::Conflict(context));
    }
    let Some(chain) = verify_state_chain(connection, project_uuid, &identity)? else {
        return Ok(DomainWrite::Conflict(context));
    };
    let claim_states = load_claim_states(connection, completion.dependency_operation_id)?;
    if claim_states.len() != 1
        || claim_states[0].transition != DependencyTransition::Admitted
        || !stored_state_is_canonical(&claim_states[0])
        || !state_belongs_to_operation(&claim_states[0], &operation)
    {
        return Ok(DomainWrite::Conflict(context));
    }
    if completion.created_at_unix_ms < claim_states[0].created_at_unix_ms {
        return Ok(DomainWrite::Conflict(context));
    }
    if let Some(state) = load_terminal_state(connection, completion.dependency_operation_id)? {
        if stored_state_is_canonical(&state)
            && state_belongs_to_operation(&state, &operation)
            && completion_matches_state(completion, &state)?
        {
            return Ok(DomainWrite::AlreadyApplied(snapshot_from_state(&state)?));
        }
        return Ok(DomainWrite::Conflict(context));
    }

    let Some(latest) = chain.latest else {
        return Ok(DomainWrite::Conflict(context));
    };
    if completion.created_at_unix_ms < latest.created_at_unix_ms {
        return Ok(DomainWrite::Conflict(context));
    }
    let (transition, consecutive_failures, cooloff_until_unix_ms, failure_class) =
        match &completion.outcome {
            DependencyCompletionOutcome::Success => (DependencyTransition::Success, 0, None, None),
            DependencyCompletionOutcome::Failure(failure_class) => {
                let consecutive_failures = latest.consecutive_failures.saturating_add(1).max(1);
                let cooloff_seconds = saturated_cooloff_seconds(
                    operation.base_cooloff_seconds,
                    operation.max_cooloff_seconds,
                    consecutive_failures,
                );
                let cooloff_millis = cooloff_seconds.checked_mul(1000).unwrap_or(i64::MAX);
                let deadline = completion.created_at_unix_ms.saturating_add(cooloff_millis);
                (
                    DependencyTransition::Failure,
                    consecutive_failures,
                    Some(deadline),
                    Some(failure_class.as_str()),
                )
            }
        };
    let state_hash = state_hash(
        completion.dependency_state_event_id,
        &operation.dependency_key_id,
        Some(completion.dependency_operation_id),
        Uuid::parse_str(&operation.anchor_id).ok(),
        transition,
        consecutive_failures,
        cooloff_until_unix_ms,
        failure_class,
        completion.created_at_unix_ms,
    )?;
    let anchor_id = Uuid::parse_str(&operation.anchor_id)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if !insert_state(
        connection,
        completion.dependency_state_event_id,
        &operation.dependency_key_id,
        completion.dependency_operation_id,
        anchor_id,
        transition,
        consecutive_failures,
        cooloff_until_unix_ms,
        failure_class,
        completion.created_at_unix_ms,
        &state_hash,
    )? {
        return Ok(DomainWrite::Conflict(context));
    }
    Ok(DomainWrite::Applied(DependencyStateSnapshot {
        dependency_key_id: operation.dependency_key_id,
        dependency_operation_id: completion.dependency_operation_id,
        transition,
        consecutive_failures,
        cooloff_until_unix_ms,
        failure_class: failure_class.map(str::to_string),
    }))
}

fn dependency_identity_matches_shadow(
    connection: &Connection,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    identity: &PreparedDependencyIdentity,
    operation: &DependencyOperation,
) -> Result<bool, LedgerError> {
    let Some(shadow) = verified_shadow_dependency_context(connection, operation.shadow_attempt_id)?
    else {
        return Ok(false);
    };
    if shadow.anchor_id != operation.anchor_id
        || shadow.project_uuid != project_uuid
        || shadow.process_instance_id != process_instance_id
    {
        return Ok(false);
    }
    let Some(judge_policy) = verified_shadow_judge_policy(connection, operation.shadow_attempt_id)?
    else {
        return Ok(false);
    };
    if operation.base_cooloff_seconds
        != i64::try_from(judge_policy.config.base_cooloff_seconds).unwrap_or(i64::MAX)
        || operation.max_cooloff_seconds
            != i64::try_from(judge_policy.config.max_cooloff_seconds).unwrap_or(i64::MAX)
    {
        return Ok(false);
    }

    let expected = match identity.key_kind {
        DependencyKeyKind::Candidate => json!({
            "api_family": shadow.api_family,
            "transport_identity": shadow.transport_identity,
            "candidate_model": shadow.candidate_model,
            "candidate_model_revision": shadow.candidate_model_revision,
        }),
        DependencyKeyKind::Judge => json!({
            "api_family": shadow.api_family,
            "transport_identity": shadow.transport_identity,
            "judge_model": judge_policy.config.model,
            "judge_model_revision": judge_policy.config.model_revision,
            "prompt_version": judge_policy.config.prompt_version,
            "prompt_sha256": judge_policy.prompt_sha256,
            "rubric_version": judge_policy.config.rubric_version,
            "rubric_sha256": judge_policy.rubric_sha256,
            "output_schema_version": judge_policy.config.output_schema_version,
            "output_schema_sha256": judge_policy.output_schema_sha256,
        }),
    };
    let expected_json = canonical_json(&expected)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    let expected_id = canonical_sha256(&expected)
        .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
    Ok(identity.canonical_identity_json == expected_json
        && identity.dependency_key_id == expected_id)
}

fn load_verified_dependency_key(
    connection: &Connection,
    project_uuid: Uuid,
    dependency_key_id: &str,
) -> Result<Option<PreparedDependencyIdentity>, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT project_uuid, key_kind, canonical_identity_json,
                    canonical_payload_hash
             FROM dependency_keys WHERE dependency_key_id = ?1",
            params![dependency_key_id],
            |row| {
                Ok(StoredDependencyKey {
                    project_uuid: row.get(0)?,
                    key_kind: row.get(1)?,
                    canonical_identity_json: row.get(2)?,
                    canonical_payload_hash: row.get(3)?,
                })
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(stored) = stored else {
        return Ok(None);
    };
    let Some(key_kind) = DependencyKeyKind::from_str(&stored.key_kind) else {
        return Ok(None);
    };
    let Ok(value): Result<Json, _> = serde_json::from_str(&stored.canonical_identity_json) else {
        return Ok(None);
    };
    if stored.project_uuid != project_uuid.to_string()
        || canonical_json(&value).ok().as_deref() != Some(stored.canonical_identity_json.as_str())
        || canonical_sha256(&value).ok().as_deref() != Some(dependency_key_id)
        || stored.canonical_payload_hash != dependency_key_id
    {
        return Ok(None);
    }
    Ok(Some(PreparedDependencyIdentity {
        key_kind,
        canonical_identity_json: stored.canonical_identity_json,
        dependency_key_id: dependency_key_id.to_string(),
    }))
}

fn insert_or_verify_dependency_key(
    connection: &Connection,
    project_uuid: Uuid,
    identity: &PreparedDependencyIdentity,
) -> Result<bool, LedgerError> {
    let inserted = connection
        .execute(
            "INSERT INTO dependency_keys (
                dependency_key_id, project_uuid, key_kind,
                canonical_identity_json, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?1)
             ON CONFLICT(dependency_key_id) DO NOTHING",
            params![
                identity.dependency_key_id,
                project_uuid.to_string(),
                identity.key_kind.as_str(),
                identity.canonical_identity_json,
            ],
        )
        .map_err(database_error)?;
    if inserted == 1 {
        return Ok(true);
    }
    let stored = connection
        .query_row(
            "SELECT project_uuid, key_kind, canonical_identity_json, canonical_payload_hash
             FROM dependency_keys WHERE dependency_key_id = ?1",
            params![identity.dependency_key_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    Ok(stored
        == Some((
            project_uuid.to_string(),
            identity.key_kind.as_str().to_string(),
            identity.canonical_identity_json.clone(),
            identity.dependency_key_id.clone(),
        )))
}

fn operation_hash(
    project_uuid: Uuid,
    process_instance_id: Uuid,
    identity: &PreparedDependencyIdentity,
    operation: &DependencyOperation,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "dependency_operation_id": operation.dependency_operation_id,
        "dependency_key_id": identity.dependency_key_id,
        "project_uuid": project_uuid,
        "process_instance_id": process_instance_id,
        "anchor_id": operation.anchor_id,
        "shadow_attempt_id": operation.shadow_attempt_id,
        "base_cooloff_seconds": operation.base_cooloff_seconds,
        "max_cooloff_seconds": operation.max_cooloff_seconds,
        "created_at_unix_ms": operation.created_at_unix_ms,
    }))
}

fn stored_operation_hash_is_valid(
    operation: &StoredOperation,
    dependency_operation_id: Uuid,
) -> Result<bool, LedgerError> {
    let expected = hash_json(&json!({
        "dependency_operation_id": dependency_operation_id,
        "dependency_key_id": operation.dependency_key_id,
        "project_uuid": operation.project_uuid,
        "process_instance_id": operation.process_instance_id,
        "anchor_id": operation.anchor_id,
        "shadow_attempt_id": operation.shadow_attempt_id,
        "base_cooloff_seconds": operation.base_cooloff_seconds,
        "max_cooloff_seconds": operation.max_cooloff_seconds,
        "created_at_unix_ms": operation.created_at_unix_ms,
    }))?;
    Ok(
        operation.dependency_operation_id == dependency_operation_id.to_string()
            && expected == operation.canonical_payload_hash,
    )
}

fn operation_matches(
    stored: &StoredOperation,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    identity: &PreparedDependencyIdentity,
    operation: &DependencyOperation,
    operation_hash: &str,
) -> bool {
    stored.dependency_operation_id == operation.dependency_operation_id.to_string()
        && stored.dependency_key_id == identity.dependency_key_id
        && stored.project_uuid == project_uuid.to_string()
        && stored.process_instance_id == process_instance_id.to_string()
        && stored.anchor_id == operation.anchor_id.to_string()
        && stored.shadow_attempt_id == operation.shadow_attempt_id.to_string()
        && stored.base_cooloff_seconds == operation.base_cooloff_seconds
        && stored.max_cooloff_seconds == operation.max_cooloff_seconds
        && stored.created_at_unix_ms == operation.created_at_unix_ms
        && stored.canonical_payload_hash == operation_hash
}

fn load_operation(
    connection: &Connection,
    dependency_operation_id: Uuid,
) -> Result<Option<StoredOperation>, LedgerError> {
    connection
        .query_row(
            "SELECT dependency_operation_id, dependency_key_id, project_uuid, process_instance_id,
                    anchor_id, shadow_attempt_id, base_cooloff_seconds,
                    max_cooloff_seconds, created_at_unix_ms, canonical_payload_hash
             FROM dependency_operations WHERE dependency_operation_id = ?1",
            params![dependency_operation_id.to_string()],
            |row| {
                Ok(StoredOperation {
                    dependency_operation_id: row.get(0)?,
                    dependency_key_id: row.get(1)?,
                    project_uuid: row.get(2)?,
                    process_instance_id: row.get(3)?,
                    anchor_id: row.get(4)?,
                    shadow_attempt_id: row.get(5)?,
                    base_cooloff_seconds: row.get(6)?,
                    max_cooloff_seconds: row.get(7)?,
                    created_at_unix_ms: row.get(8)?,
                    canonical_payload_hash: row.get(9)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

#[allow(clippy::too_many_arguments)]
fn insert_state(
    connection: &Connection,
    event_id: Uuid,
    dependency_key_id: &str,
    operation_id: Uuid,
    anchor_id: Uuid,
    transition: DependencyTransition,
    consecutive_failures: i64,
    cooloff_until_unix_ms: Option<i64>,
    failure_class: Option<&str>,
    created_at_unix_ms: i64,
    canonical_payload_hash: &str,
) -> Result<bool, LedgerError> {
    connection
        .execute(
            "INSERT INTO dependency_state_events (
                dependency_state_event_id, dependency_key_id,
                dependency_operation_id, anchor_id, state,
                consecutive_failures, cooloff_until_unix_ms, failure_class,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)
             ON CONFLICT(dependency_state_event_id) DO NOTHING",
            params![
                event_id.to_string(),
                dependency_key_id,
                operation_id.to_string(),
                anchor_id.to_string(),
                transition.as_str(),
                consecutive_failures,
                cooloff_until_unix_ms,
                failure_class,
                created_at_unix_ms,
                canonical_payload_hash,
            ],
        )
        .map(|rows| rows == 1)
        .map_err(database_error)
}

#[allow(clippy::too_many_arguments)]
fn state_hash(
    event_id: Uuid,
    dependency_key_id: &str,
    operation_id: Option<Uuid>,
    anchor_id: Option<Uuid>,
    transition: DependencyTransition,
    consecutive_failures: i64,
    cooloff_until_unix_ms: Option<i64>,
    failure_class: Option<&str>,
    created_at_unix_ms: i64,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "dependency_state_event_id": event_id,
        "dependency_key_id": dependency_key_id,
        "dependency_operation_id": operation_id,
        "anchor_id": anchor_id,
        "state": transition.as_str(),
        "consecutive_failures": consecutive_failures,
        "cooloff_until_unix_ms": cooloff_until_unix_ms,
        "failure_class": failure_class,
        "created_at_unix_ms": created_at_unix_ms,
    }))
}

fn load_claim_state(
    connection: &Connection,
    operation_id: Uuid,
) -> Result<Option<StoredState>, LedgerError> {
    let states = load_claim_states(connection, operation_id)?;
    if states.len() > 1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(states.into_iter().next())
}

fn load_claim_states(
    connection: &Connection,
    operation_id: Uuid,
) -> Result<Vec<StoredState>, LedgerError> {
    load_states(
        connection,
        "SELECT dependency_state_event_id, dependency_key_id,
                dependency_operation_id, anchor_id, state, consecutive_failures,
                cooloff_until_unix_ms, failure_class, created_at_unix_ms,
                canonical_payload_hash
         FROM dependency_state_events
         WHERE dependency_operation_id = ?1 AND state IN ('admitted', 'skipped_cooloff')
         ORDER BY event_seq",
        operation_id,
    )
}

fn load_terminal_state(
    connection: &Connection,
    operation_id: Uuid,
) -> Result<Option<StoredState>, LedgerError> {
    let states = load_states(
        connection,
        "SELECT dependency_state_event_id, dependency_key_id,
                dependency_operation_id, anchor_id, state, consecutive_failures,
                cooloff_until_unix_ms, failure_class, created_at_unix_ms,
                canonical_payload_hash
         FROM dependency_state_events
         WHERE dependency_operation_id = ?1
           AND state IN ('success', 'failure', 'orphaned_in_flight')
         ORDER BY event_seq",
        operation_id,
    )?;
    if states.len() > 1 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(states.into_iter().next())
}

fn load_states(
    connection: &Connection,
    sql: &str,
    operation_id: Uuid,
) -> Result<Vec<StoredState>, LedgerError> {
    let mut statement = connection.prepare(sql).map_err(database_error)?;
    statement
        .query_map(params![operation_id.to_string()], stored_state_from_row)
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)
}

fn verify_state_chain(
    connection: &Connection,
    project_uuid: Uuid,
    identity: &PreparedDependencyIdentity,
) -> Result<Option<VerifiedStateChain>, LedgerError> {
    if load_verified_dependency_key(connection, project_uuid, &identity.dependency_key_id)?.as_ref()
        != Some(identity)
    {
        return Ok(None);
    }
    let mut statement = connection
        .prepare(
            "SELECT dependency_state_event_id, dependency_key_id,
                    dependency_operation_id, anchor_id, state, consecutive_failures,
                    cooloff_until_unix_ms, failure_class, created_at_unix_ms,
                    canonical_payload_hash
             FROM dependency_state_events
             WHERE dependency_key_id = ?1 ORDER BY event_seq",
        )
        .map_err(database_error)?;
    let states = statement
        .query_map(params![identity.dependency_key_id], stored_state_from_row)
        .map_err(database_error)?
        .collect::<Result<Vec<_>, _>>()
        .map_err(database_error)?;
    let mut latest: Option<StoredState> = None;
    let mut claimed_operations = BTreeSet::new();
    let mut admitted_operations = BTreeSet::new();
    let checkpoint_index = states
        .iter()
        .rposition(|state| state.dependency_operation_id.is_none() && state.anchor_id.is_none());

    if let Some(checkpoint_index) = checkpoint_index {
        let checkpoint = &states[checkpoint_index];
        if !stored_state_is_canonical(checkpoint) || !unlinked_baseline_is_valid(checkpoint) {
            return Ok(None);
        }
        let mut latest_prefix_time = 0;
        for state in &states[..checkpoint_index] {
            if !stored_state_is_canonical(state) {
                return Ok(None);
            }
            latest_prefix_time = latest_prefix_time.max(state.created_at_unix_ms);
            if state.dependency_operation_id.is_none() && state.anchor_id.is_none() {
                if !unlinked_baseline_is_valid(state) {
                    return Ok(None);
                }
                continue;
            }
            let Some(operation) =
                verified_operation_for_state(connection, project_uuid, identity, state)?
            else {
                return Ok(None);
            };
            match state.transition {
                DependencyTransition::Admitted => {
                    if !claimed_operations.insert(operation.dependency_operation_id.clone())
                        || !admitted_operations.insert(operation.dependency_operation_id.clone())
                    {
                        return Ok(None);
                    }
                }
                DependencyTransition::SkippedCooloff => {
                    if !claimed_operations.insert(operation.dependency_operation_id) {
                        return Ok(None);
                    }
                }
                DependencyTransition::Success
                | DependencyTransition::Failure
                | DependencyTransition::OrphanedInFlight => {
                    if !admitted_operations.remove(&operation.dependency_operation_id)
                        || state.created_at_unix_ms < operation.created_at_unix_ms
                    {
                        return Ok(None);
                    }
                }
            }
        }
        if checkpoint.created_at_unix_ms < latest_prefix_time {
            return Ok(None);
        }
        latest = Some(checkpoint.clone());
    }

    for state in states
        .into_iter()
        .skip(checkpoint_index.map_or(0, |index| index + 1))
    {
        if !stored_state_is_canonical(&state) {
            return Ok(None);
        }
        if state.dependency_operation_id.is_none() && state.anchor_id.is_none() {
            if !unlinked_baseline_is_valid(&state)
                || latest
                    .as_ref()
                    .is_some_and(|previous| state.created_at_unix_ms < previous.created_at_unix_ms)
            {
                return Ok(None);
            }
            latest = Some(state);
            continue;
        }
        let Some(operation) =
            verified_operation_for_state(connection, project_uuid, identity, &state)?
        else {
            return Ok(None);
        };
        match state.transition {
            DependencyTransition::Admitted | DependencyTransition::SkippedCooloff => {
                if !claim_transition_matches(&state, &operation, latest.as_ref())
                    || !claimed_operations.insert(operation.dependency_operation_id.clone())
                {
                    return Ok(None);
                }
                if state.transition == DependencyTransition::Admitted {
                    admitted_operations.insert(operation.dependency_operation_id.clone());
                }
            }
            DependencyTransition::Success
            | DependencyTransition::Failure
            | DependencyTransition::OrphanedInFlight => {
                if !admitted_operations.remove(&operation.dependency_operation_id)
                    || !terminal_transition_matches(&state, &operation, latest.as_ref())
                {
                    return Ok(None);
                }
            }
        }
        latest = Some(state);
    }
    Ok(Some(VerifiedStateChain { latest }))
}

fn unlinked_baseline_is_valid(state: &StoredState) -> bool {
    if state.dependency_operation_id.is_some() || state.anchor_id.is_some() {
        return false;
    }
    match state.transition {
        DependencyTransition::Success => {
            state.consecutive_failures == 0
                && state.cooloff_until_unix_ms.is_none()
                && state.failure_class.is_none()
        }
        DependencyTransition::Failure => {
            state.consecutive_failures > 0
                && state.cooloff_until_unix_ms.is_some()
                && state
                    .failure_class
                    .as_deref()
                    .is_some_and(|value| DependencyFailureClass::new(value).is_ok())
        }
        DependencyTransition::SkippedCooloff => {
            state.consecutive_failures > 0
                && state.cooloff_until_unix_ms.is_some()
                && state.failure_class.is_none()
        }
        DependencyTransition::OrphanedInFlight => {
            state.failure_class.is_none()
                && ((state.consecutive_failures == 0 && state.cooloff_until_unix_ms.is_none())
                    || (state.consecutive_failures > 0 && state.cooloff_until_unix_ms.is_some()))
        }
        DependencyTransition::Admitted => false,
    }
}

/// Checkpoint every dependency chain touched by an anchor retention cascade.
pub(super) fn carry_forward_latest_dependency_states_in_transaction(
    connection: &Connection,
    project_uuid: Uuid,
    selected_anchor_ids: &BTreeSet<String>,
    created_at_unix_ms: i64,
) -> Result<BTreeSet<String>, LedgerError> {
    if created_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let mut dependency_key_ids = BTreeSet::new();
    for anchor_id in selected_anchor_ids {
        if parse_canonical_uuid_v7(anchor_id).is_none() {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let mut statement = connection
            .prepare(
                "SELECT DISTINCT dependency_key_id
                 FROM dependency_state_events
                 WHERE anchor_id = ?1
                 ORDER BY dependency_key_id",
            )
            .map_err(database_error)?;
        let keys = statement
            .query_map(params![anchor_id], |row| row.get::<_, String>(0))
            .map_err(database_error)?
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(database_error)?;
        dependency_key_ids.extend(keys);
    }

    let mut checkpointed_dependency_key_ids = BTreeSet::new();
    for dependency_key_id in dependency_key_ids {
        let identity = load_verified_dependency_key(connection, project_uuid, &dependency_key_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let chain = verify_state_chain(connection, project_uuid, &identity)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let latest = chain
            .latest
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if latest.dependency_operation_id.is_none() && latest.anchor_id.is_none() {
            checkpointed_dependency_key_ids.insert(dependency_key_id);
            continue;
        }
        if latest.dependency_operation_id.is_none() || latest.anchor_id.is_none() {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }

        let (checkpoint_transition, checkpoint_failure_class) = match latest.transition {
            DependencyTransition::Admitted => match (
                latest.consecutive_failures,
                latest.cooloff_until_unix_ms,
                latest.failure_class.as_deref(),
            ) {
                (0, None, None) => (DependencyTransition::Success, None),
                (failures, Some(_), None) if failures > 0 => {
                    (DependencyTransition::SkippedCooloff, None)
                }
                _ => return Err(LedgerErrorClass::IdentityInvariant.into()),
            },
            transition => (transition, latest.failure_class.as_deref()),
        };

        let state_event_id = Uuid::now_v7();
        let baseline_created_at_unix_ms = created_at_unix_ms.max(latest.created_at_unix_ms);
        let canonical_payload_hash = state_hash(
            state_event_id,
            &dependency_key_id,
            None,
            None,
            checkpoint_transition,
            latest.consecutive_failures,
            latest.cooloff_until_unix_ms,
            checkpoint_failure_class,
            baseline_created_at_unix_ms,
        )?;
        let inserted = connection
            .execute(
                "INSERT INTO dependency_state_events (
                    dependency_state_event_id, dependency_key_id,
                    dependency_operation_id, anchor_id, state,
                    consecutive_failures, cooloff_until_unix_ms, failure_class,
                    created_at_unix_ms, canonical_payload_hash
                 ) VALUES (?1, ?2, NULL, NULL, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    state_event_id.to_string(),
                    dependency_key_id,
                    checkpoint_transition.as_str(),
                    latest.consecutive_failures,
                    latest.cooloff_until_unix_ms,
                    checkpoint_failure_class,
                    baseline_created_at_unix_ms,
                    canonical_payload_hash,
                ],
            )
            .map_err(database_error)?;
        if inserted != 1 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }

        let carried = verify_state_chain(connection, project_uuid, &identity)?
            .and_then(|chain| chain.latest)
            .filter(|state| {
                state.state_event_id == state_event_id.to_string()
                    && state.dependency_operation_id.is_none()
                    && state.anchor_id.is_none()
                    && unlinked_baseline_is_valid(state)
            })
            .is_some();
        if !carried {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        checkpointed_dependency_key_ids.insert(dependency_key_id);
    }
    Ok(checkpointed_dependency_key_ids)
}

/// Verify that retention left every carried cool-off baseline intact and canonical.
pub(super) fn verify_retained_dependency_states_in_transaction(
    connection: &Connection,
    project_uuid: Uuid,
    carried_dependency_key_ids: &BTreeSet<String>,
) -> Result<(), LedgerError> {
    for dependency_key_id in carried_dependency_key_ids {
        let identity = load_verified_dependency_key(connection, project_uuid, dependency_key_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let latest = verify_state_chain(connection, project_uuid, &identity)?
            .and_then(|chain| chain.latest)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if latest.dependency_operation_id.is_some()
            || latest.anchor_id.is_some()
            || !unlinked_baseline_is_valid(&latest)
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
    }
    Ok(())
}

fn verified_operation_for_state(
    connection: &Connection,
    project_uuid: Uuid,
    identity: &PreparedDependencyIdentity,
    state: &StoredState,
) -> Result<Option<StoredOperation>, LedgerError> {
    let Some(operation_id) = state
        .dependency_operation_id
        .as_deref()
        .and_then(parse_canonical_uuid_v7)
    else {
        return Ok(None);
    };
    let Some(state_event_id) = parse_canonical_uuid_v7(&state.state_event_id) else {
        return Ok(None);
    };
    let Some(operation) = load_operation(connection, operation_id)? else {
        return Ok(None);
    };
    let Some(operation_process_id) = parse_canonical_uuid_v7(&operation.process_instance_id) else {
        return Ok(None);
    };
    let Some(anchor_id) = parse_canonical_uuid_v7(&operation.anchor_id) else {
        return Ok(None);
    };
    let Some(shadow_attempt_id) = parse_canonical_uuid_v7(&operation.shadow_attempt_id) else {
        return Ok(None);
    };
    if operation.project_uuid != project_uuid.to_string()
        || operation.dependency_key_id != identity.dependency_key_id
        || state.dependency_key_id != identity.dependency_key_id
        || state.dependency_operation_id.as_deref()
            != Some(operation.dependency_operation_id.as_str())
        || state.anchor_id.as_deref() != Some(operation.anchor_id.as_str())
        || operation.base_cooloff_seconds <= 0
        || operation.max_cooloff_seconds < operation.base_cooloff_seconds
        || operation.created_at_unix_ms < 0
        || !stored_operation_hash_is_valid(&operation, operation_id)?
    {
        return Ok(None);
    }
    let replay_operation = DependencyOperation {
        dependency_operation_id: operation_id,
        dependency_state_event_id: state_event_id,
        anchor_id,
        shadow_attempt_id,
        base_cooloff_seconds: operation.base_cooloff_seconds,
        max_cooloff_seconds: operation.max_cooloff_seconds,
        created_at_unix_ms: operation.created_at_unix_ms,
    };
    if !dependency_identity_matches_shadow(
        connection,
        project_uuid,
        operation_process_id,
        identity,
        &replay_operation,
    )? {
        return Ok(None);
    }
    Ok(Some(operation))
}

fn claim_transition_matches(
    state: &StoredState,
    operation: &StoredOperation,
    previous: Option<&StoredState>,
) -> bool {
    if previous.is_some_and(|previous| operation.created_at_unix_ms < previous.created_at_unix_ms) {
        return false;
    }
    let (expected_transition, expected_failures, expected_deadline) = match previous {
        Some(previous)
            if previous
                .cooloff_until_unix_ms
                .is_some_and(|deadline| deadline > operation.created_at_unix_ms) =>
        {
            (
                DependencyTransition::SkippedCooloff,
                previous.consecutive_failures,
                previous.cooloff_until_unix_ms,
            )
        }
        Some(previous) => (
            DependencyTransition::Admitted,
            previous.consecutive_failures,
            previous.cooloff_until_unix_ms,
        ),
        None => (DependencyTransition::Admitted, 0, None),
    };
    state.transition == expected_transition
        && state.consecutive_failures == expected_failures
        && state.cooloff_until_unix_ms == expected_deadline
        && state.failure_class.is_none()
        && state.created_at_unix_ms == operation.created_at_unix_ms
}

fn terminal_transition_matches(
    state: &StoredState,
    operation: &StoredOperation,
    previous: Option<&StoredState>,
) -> bool {
    let Some(previous) = previous else {
        return false;
    };
    if state.created_at_unix_ms < previous.created_at_unix_ms {
        return false;
    }
    match state.transition {
        DependencyTransition::Success => {
            state.consecutive_failures == 0
                && state.cooloff_until_unix_ms.is_none()
                && state.failure_class.is_none()
        }
        DependencyTransition::Failure => {
            let expected_failures = previous.consecutive_failures.saturating_add(1).max(1);
            let cooloff_seconds = saturated_cooloff_seconds(
                operation.base_cooloff_seconds,
                operation.max_cooloff_seconds,
                expected_failures,
            );
            let expected_deadline = state
                .created_at_unix_ms
                .saturating_add(cooloff_seconds.checked_mul(1000).unwrap_or(i64::MAX));
            state.consecutive_failures == expected_failures
                && state.cooloff_until_unix_ms == Some(expected_deadline)
                && state
                    .failure_class
                    .as_deref()
                    .is_some_and(|value| DependencyFailureClass::new(value).is_ok())
        }
        DependencyTransition::OrphanedInFlight => {
            state.consecutive_failures == previous.consecutive_failures
                && state.cooloff_until_unix_ms == previous.cooloff_until_unix_ms
                && state.failure_class.is_none()
        }
        DependencyTransition::Admitted | DependencyTransition::SkippedCooloff => false,
    }
}

fn parse_canonical_uuid_v7(value: &str) -> Option<Uuid> {
    let parsed = Uuid::parse_str(value).ok()?;
    (parsed.to_string() == value && validate_uuid_v7(parsed).is_ok()).then_some(parsed)
}

fn stored_state_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredState> {
    let state = row.get::<_, String>(4)?;
    let transition = match state.as_str() {
        "admitted" => DependencyTransition::Admitted,
        "skipped_cooloff" => DependencyTransition::SkippedCooloff,
        "success" => DependencyTransition::Success,
        "failure" => DependencyTransition::Failure,
        "orphaned_in_flight" => DependencyTransition::OrphanedInFlight,
        _ => return Err(rusqlite::Error::InvalidQuery),
    };
    Ok(StoredState {
        state_event_id: row.get(0)?,
        dependency_key_id: row.get(1)?,
        dependency_operation_id: row.get(2)?,
        anchor_id: row.get(3)?,
        transition,
        consecutive_failures: row.get(5)?,
        cooloff_until_unix_ms: row.get(6)?,
        failure_class: row.get(7)?,
        created_at_unix_ms: row.get(8)?,
        canonical_payload_hash: row.get(9)?,
    })
}

/// Neutrally terminalize admitted dependency operations owned by a fenced process.
pub(super) fn orphan_inflight_dependency_operations_in_transaction(
    connection: &Connection,
    project_uuid: Uuid,
    dead_process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<usize, LedgerError> {
    if created_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let mut statement = connection
        .prepare(
            "SELECT o.dependency_operation_id
             FROM dependency_operations AS o
             WHERE o.project_uuid = ?1 AND o.process_instance_id = ?2
               AND EXISTS (
                    SELECT 1 FROM dependency_state_events AS s
                    WHERE s.dependency_operation_id = o.dependency_operation_id
                      AND s.state = 'admitted'
               )
               AND NOT EXISTS (
                    SELECT 1 FROM dependency_state_events AS s
                    WHERE s.dependency_operation_id = o.dependency_operation_id
                      AND s.state <> 'admitted'
               )
             ORDER BY o.dependency_operation_id",
        )
        .map_err(database_error)?;
    let operation_ids = statement
        .query_map(
            params![
                project_uuid.to_string(),
                dead_process_instance_id.to_string()
            ],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);

    let mut applied = 0usize;
    for operation_id in operation_ids {
        let operation_id = parse_canonical_uuid_v7(&operation_id)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let operation = load_operation(connection, operation_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if operation.project_uuid != project_uuid.to_string()
            || operation.process_instance_id != dead_process_instance_id.to_string()
            || !stored_operation_hash_is_valid(&operation, operation_id)?
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let identity =
            load_verified_dependency_key(connection, project_uuid, &operation.dependency_key_id)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let claim = load_claim_state(connection, operation_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if claim.transition != DependencyTransition::Admitted
            || !stored_state_is_canonical(&claim)
            || !state_belongs_to_operation(&claim, &operation)
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let chain = verify_state_chain(connection, project_uuid, &identity)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let latest = chain
            .latest
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if created_at_unix_ms < claim.created_at_unix_ms
            || created_at_unix_ms < latest.created_at_unix_ms
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let anchor_id = parse_canonical_uuid_v7(&operation.anchor_id)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let event_id = Uuid::now_v7();
        let payload_hash = state_hash(
            event_id,
            &operation.dependency_key_id,
            Some(operation_id),
            Some(anchor_id),
            DependencyTransition::OrphanedInFlight,
            latest.consecutive_failures,
            latest.cooloff_until_unix_ms,
            None,
            created_at_unix_ms,
        )?;
        if !insert_state(
            connection,
            event_id,
            &operation.dependency_key_id,
            operation_id,
            anchor_id,
            DependencyTransition::OrphanedInFlight,
            latest.consecutive_failures,
            latest.cooloff_until_unix_ms,
            None,
            created_at_unix_ms,
            &payload_hash,
        )? {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        applied = applied
            .checked_add(1)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    }
    validate_reconciled_dependency_operations(connection, project_uuid, dead_process_instance_id)?;
    Ok(applied)
}

fn validate_reconciled_dependency_operations(
    connection: &Connection,
    project_uuid: Uuid,
    dead_process_instance_id: Uuid,
) -> Result<(), LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT dependency_operation_id FROM dependency_operations
             WHERE project_uuid = ?1 AND process_instance_id = ?2
             ORDER BY dependency_operation_id",
        )
        .map_err(database_error)?;
    let operation_ids = statement
        .query_map(
            params![
                project_uuid.to_string(),
                dead_process_instance_id.to_string()
            ],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);

    for operation_id in operation_ids {
        let operation_id = parse_canonical_uuid_v7(&operation_id)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let operation = load_operation(connection, operation_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if operation.project_uuid != project_uuid.to_string()
            || operation.process_instance_id != dead_process_instance_id.to_string()
            || !stored_operation_hash_is_valid(&operation, operation_id)?
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let identity =
            load_verified_dependency_key(connection, project_uuid, &operation.dependency_key_id)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let claim = load_claim_state(connection, operation_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if !stored_state_is_canonical(&claim)
            || !state_belongs_to_operation(&claim, &operation)
            || verified_operation_for_state(connection, project_uuid, &identity, &claim)?
                .as_ref()
                .is_none_or(|verified| {
                    verified.dependency_operation_id != operation.dependency_operation_id
                })
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let terminal = load_terminal_state(connection, operation_id)?;
        match (claim.transition, terminal.as_ref()) {
            (DependencyTransition::SkippedCooloff, None) => {}
            (DependencyTransition::Admitted, Some(terminal))
                if matches!(
                    terminal.transition,
                    DependencyTransition::Success
                        | DependencyTransition::Failure
                        | DependencyTransition::OrphanedInFlight
                ) && stored_state_is_canonical(terminal)
                    && terminal.created_at_unix_ms >= claim.created_at_unix_ms
                    && state_belongs_to_operation(terminal, &operation)
                    && verified_operation_for_state(
                        connection,
                        project_uuid,
                        &identity,
                        terminal,
                    )?
                    .as_ref()
                    .is_some_and(|verified| {
                        verified.dependency_operation_id == operation.dependency_operation_id
                    }) => {}
            _ => return Err(LedgerErrorClass::IdentityInvariant.into()),
        }
        let expected_state_count = if terminal.is_some() { 2 } else { 1 };
        let state_count: i64 = connection
            .query_row(
                "SELECT count(*) FROM dependency_state_events
                 WHERE dependency_operation_id = ?1",
                params![operation_id.to_string()],
                |row| row.get(0),
            )
            .map_err(database_error)?;
        if state_count != expected_state_count
            || verify_state_chain(connection, project_uuid, &identity)?.is_none()
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
    }
    Ok(())
}

fn stored_state_is_canonical(state: &StoredState) -> bool {
    let Some(event_id) = parse_canonical_uuid_v7(&state.state_event_id) else {
        return false;
    };
    let operation_id = match state.dependency_operation_id.as_deref() {
        Some(value) => match parse_canonical_uuid_v7(value) {
            Some(value) => Some(value),
            None => return false,
        },
        None => None,
    };
    let anchor_id = match state.anchor_id.as_deref() {
        Some(value) => match parse_canonical_uuid_v7(value) {
            Some(value) => Some(value),
            None => return false,
        },
        None => None,
    };
    state_hash(
        event_id,
        &state.dependency_key_id,
        operation_id,
        anchor_id,
        state.transition,
        state.consecutive_failures,
        state.cooloff_until_unix_ms,
        state.failure_class.as_deref(),
        state.created_at_unix_ms,
    )
    .is_ok_and(|expected| expected == state.canonical_payload_hash)
}

fn state_belongs_to_operation(state: &StoredState, operation: &StoredOperation) -> bool {
    state.dependency_key_id == operation.dependency_key_id
        && state.dependency_operation_id.as_deref()
            == Some(operation.dependency_operation_id.as_str())
        && state.anchor_id.as_deref() == Some(operation.anchor_id.as_str())
        && (state.transition == DependencyTransition::Failure || state.failure_class.is_none())
}

fn claim_state_matches(
    state: &StoredState,
    identity: &PreparedDependencyIdentity,
    operation: &DependencyOperation,
) -> Result<bool, LedgerError> {
    let expected_hash = state_hash(
        operation.dependency_state_event_id,
        &identity.dependency_key_id,
        Some(operation.dependency_operation_id),
        Some(operation.anchor_id),
        state.transition,
        state.consecutive_failures,
        state.cooloff_until_unix_ms,
        None,
        operation.created_at_unix_ms,
    )?;
    Ok(
        state.state_event_id == operation.dependency_state_event_id.to_string()
            && state.dependency_key_id == identity.dependency_key_id
            && state.dependency_operation_id.as_deref()
                == Some(operation.dependency_operation_id.to_string().as_str())
            && state.anchor_id.as_deref() == Some(operation.anchor_id.to_string().as_str())
            && matches!(
                state.transition,
                DependencyTransition::Admitted | DependencyTransition::SkippedCooloff
            )
            && state.failure_class.is_none()
            && state.created_at_unix_ms == operation.created_at_unix_ms
            && state.canonical_payload_hash == expected_hash,
    )
}

fn completion_matches_state(
    completion: &DependencyCompletion,
    state: &StoredState,
) -> Result<bool, LedgerError> {
    let expected_transition = match completion.outcome {
        DependencyCompletionOutcome::Success => DependencyTransition::Success,
        DependencyCompletionOutcome::Failure(_) => DependencyTransition::Failure,
    };
    let expected_failure_class = match &completion.outcome {
        DependencyCompletionOutcome::Success => None,
        DependencyCompletionOutcome::Failure(failure_class) => Some(failure_class.as_str()),
    };
    let operation_id = state
        .dependency_operation_id
        .as_deref()
        .and_then(|value| Uuid::parse_str(value).ok());
    let anchor_id = state
        .anchor_id
        .as_deref()
        .and_then(|value| Uuid::parse_str(value).ok());
    let expected_hash = state_hash(
        completion.dependency_state_event_id,
        &state.dependency_key_id,
        operation_id,
        anchor_id,
        state.transition,
        state.consecutive_failures,
        state.cooloff_until_unix_ms,
        state.failure_class.as_deref(),
        completion.created_at_unix_ms,
    )?;
    Ok(
        state.state_event_id == completion.dependency_state_event_id.to_string()
            && operation_id == Some(completion.dependency_operation_id)
            && state.transition == expected_transition
            && state.failure_class.as_deref() == expected_failure_class
            && state.created_at_unix_ms == completion.created_at_unix_ms
            && state.canonical_payload_hash == expected_hash,
    )
}

fn snapshot_from_state(state: &StoredState) -> Result<DependencyStateSnapshot, LedgerError> {
    let operation_id = state
        .dependency_operation_id
        .as_deref()
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))
        .and_then(|value| {
            Uuid::parse_str(value)
                .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))
        })?;
    Ok(DependencyStateSnapshot {
        dependency_key_id: state.dependency_key_id.clone(),
        dependency_operation_id: operation_id,
        transition: state.transition,
        consecutive_failures: state.consecutive_failures,
        cooloff_until_unix_ms: state.cooloff_until_unix_ms,
        failure_class: state.failure_class.clone(),
    })
}

fn saturated_cooloff_seconds(base_seconds: i64, max_seconds: i64, count: i64) -> i64 {
    let exponent = u32::try_from(count.saturating_sub(1)).unwrap_or(u32::MAX);
    let scaled = if exponent >= i64::BITS - 1 {
        i64::MAX
    } else {
        base_seconds
            .checked_mul(1_i64 << exponent)
            .unwrap_or(i64::MAX)
    };
    scaled.min(max_seconds)
}

fn validate_bounded_text(value: &str, max_bytes: usize) -> Result<(), LedgerError> {
    if value.is_empty() || value.len() > max_bytes || value.chars().any(char::is_control) {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn validate_sha256(value: &str) -> Result<(), LedgerError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn validate_uuid_v7(value: Uuid) -> Result<(), LedgerError> {
    if value.get_version_num() != 7 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
}

fn hash_json(value: &Json) -> Result<String, LedgerError> {
    canonical_sha256(value).map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn database_error(error: rusqlite::Error) -> LedgerError {
    map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use chrono::{TimeZone, Utc};
    use nemo_relay::api::llm::LlmApiFamily;
    use nemo_relay::api::runtime::{LLM_REPLAY_CONTRACT_VERSION, LlmReplayCapability};
    use rusqlite::{Connection, params};
    use tempfile::tempdir;

    use super::*;
    use crate::ledger::model::LedgerRuntimeIdentity;
    use crate::ledger::repository::anchors::{
        AnchorCommandAck, FrozenPendingAnchorV1, FrozenTerminalAnchorV1,
    };
    use crate::ledger::repository::retention::{RetentionAck, RetentionRequest};
    use crate::ledger::repository::shadow::{
        ReservedShadowAttempt, SampleBatchReservation, SampleBatchTerminalEvent,
        SampleBatchTerminalState, ShadowAttemptStarted, ShadowCommandAck,
        ShadowOperationalFailureClass, ShadowTerminalClass, ShadowTerminalRecord,
        ShadowVectorSourceV1,
    };
    use crate::ledger::repository::tests::{
        assert_reconciliation_noop_twice, config, database_path, recovery_snapshot,
    };
    use crate::projection::{
        REQUEST_PROJECTION_SCHEMA_V1, ROUTER_SANITIZER_VERSION, RouterRequestProjectionV1,
        SanitizedAnnotatedLlmRequest,
    };
    use crate::trajectory::test_fixtures::pending_window;
    use crate::trajectory::{
        CANDIDATE_FACT_SCHEMA_V1, PendingTrajectoryWindow, PersistedCandidateCapabilitiesV1,
        PersistedCandidateFactV1, PersistedTrajectoryTerminalV1, ReplayCapabilityFactsV1,
        TrajectoryTrigger,
    };

    const FIXTURE_CREATED_AT: i64 = 1_700_000_001_000;

    #[derive(Clone, Copy)]
    struct FakeClock {
        now_unix_ms: i64,
    }

    impl FakeClock {
        fn at(now_unix_ms: i64) -> Self {
            Self { now_unix_ms }
        }

        fn set(&mut self, now_unix_ms: i64) {
            self.now_unix_ms = now_unix_ms;
        }

        fn operation(
            self,
            target: Target,
            base_cooloff_seconds: u64,
            max_cooloff_seconds: u64,
        ) -> DependencyOperation {
            DependencyOperation::new(
                Uuid::now_v7(),
                Uuid::now_v7(),
                target.anchor_id,
                target.shadow_attempt_id,
                base_cooloff_seconds,
                max_cooloff_seconds,
                self.now_unix_ms,
            )
            .unwrap()
        }

        fn success(self, operation: &DependencyOperation) -> DependencyCompletion {
            DependencyCompletion::success(
                operation.dependency_operation_id,
                Uuid::now_v7(),
                self.now_unix_ms,
            )
            .unwrap()
        }

        fn failure(self, operation: &DependencyOperation) -> DependencyCompletion {
            DependencyCompletion::failure(
                operation.dependency_operation_id,
                Uuid::now_v7(),
                self.now_unix_ms,
                DependencyFailureClass::new("router.provider.timeout").unwrap(),
            )
            .unwrap()
        }
    }

    #[derive(Clone, Copy)]
    struct Target {
        anchor_id: Uuid,
        shadow_attempt_id: Uuid,
    }

    fn candidate_identity(_suffix: &str) -> CandidateDependencyIdentity {
        CandidateDependencyIdentity::new(
            LlmApiFamily::OpenAIChatCompletions,
            "transport-shared",
            "candidate-model-a",
            "2026-06-01",
        )
        .unwrap()
    }

    fn judge_identity(repository: &LedgerRepository) -> JudgeDependencyIdentity {
        let policy_json: String = repository
            .connection
            .query_row(
                "SELECT canonical_policy_json FROM policy_versions WHERE pool_id = 'pool-a'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        let policy: Json = serde_json::from_str(&policy_json).unwrap();
        let config = policy.pointer("/pool/judge/config").unwrap();
        let derived = policy.pointer("/pool/judge/derived_policy").unwrap();
        JudgeDependencyIdentity::new(
            LlmApiFamily::OpenAIChatCompletions,
            "transport-shared",
            config["model"].as_str().unwrap(),
            config["model_revision"].as_str().unwrap(),
            config["prompt_version"].as_str().unwrap(),
            derived["prompt_template_sha256"].as_str().unwrap(),
            config["rubric_version"].as_str().unwrap(),
            derived["rubric_template_sha256"].as_str().unwrap(),
            1,
            derived["output_schema_sha256"].as_str().unwrap(),
        )
        .unwrap()
    }

    fn activate(path: &Path, project_id: &str) -> super::super::ActivatedLedger {
        activate_with_cooloff(path, project_id, 2, 8)
    }

    fn activate_with_cooloff(
        path: &Path,
        project_id: &str,
        base_cooloff_seconds: u64,
        max_cooloff_seconds: u64,
    ) -> super::super::ActivatedLedger {
        let mut config = config(path, project_id);
        config.pools[0].judge.base_cooloff_seconds = base_cooloff_seconds;
        config.pools[0].judge.max_cooloff_seconds = max_cooloff_seconds;
        LedgerRepository::activate(&config).unwrap()
    }

    fn seed_target(
        repository: &mut LedgerRepository,
        identity: &LedgerRuntimeIdentity,
        _suffix: &str,
    ) -> Target {
        let anchor_id = Uuid::now_v7();
        let pool = identity.pools.get("pool-a").unwrap();
        let pending = pending_anchor(identity, anchor_id);
        let frozen_pending = FrozenPendingAnchorV1::new(
            &pending,
            Uuid::now_v7(),
            Uuid::now_v7(),
            FIXTURE_CREATED_AT,
        )
        .unwrap();
        assert!(matches!(
            repository.record_pending_anchor(&frozen_pending).unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
        let terminal = PersistedTrajectoryTerminalV1::closed(
            pending,
            Vec::new(),
            1,
            TrajectoryTrigger::ProgressReached,
            Utc.timestamp_millis_opt(FIXTURE_CREATED_AT + 1).unwrap(),
            Vec::new(),
        );
        let frozen_terminal = FrozenTerminalAnchorV1::new(
            &terminal,
            Uuid::now_v7(),
            Uuid::now_v7(),
            FIXTURE_CREATED_AT + 1,
        )
        .unwrap();
        assert!(matches!(
            repository.record_terminal_anchor(&frozen_terminal).unwrap(),
            AnchorCommandAck::Applied { .. }
        ));

        let evaluator_version = policy_evaluator_version(repository);
        let attempt = ReservedShadowAttempt::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            "candidate-a",
            "candidate-model-a",
            "2026-06-01",
            0,
            LlmApiFamily::OpenAIChatCompletions,
            "transport-shared",
            "anchor-a",
            "2026-07-01",
            "1".repeat(64),
            evaluator_version,
            "3".repeat(64),
            "4".repeat(64),
            true,
            candidate_request_projection(),
            FIXTURE_CREATED_AT + 2,
        )
        .unwrap();
        let reservation = SampleBatchReservation::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            anchor_id,
            identity.config_generation_id.clone(),
            pool.policy_version_id.clone(),
            pool.learning_generation_id,
            "pool-a",
            vec![attempt.clone()],
            FIXTURE_CREATED_AT + 2,
        )
        .unwrap();
        assert_eq!(
            repository.reserve_sample_batch(&reservation).unwrap(),
            ShadowCommandAck::Applied
        );
        Target {
            anchor_id,
            shadow_attempt_id: attempt.shadow_attempt_id,
        }
    }

    fn pending_anchor(
        identity: &LedgerRuntimeIdentity,
        anchor_id: Uuid,
    ) -> PendingTrajectoryWindow {
        let pool = identity.pools.get("pool-a").unwrap();
        let mut pending = pending_window(anchor_id);
        pending.anchor_call_uuid = Uuid::now_v7();
        pending.root_uuid = Uuid::now_v7();
        pending.owner_uuid = Uuid::now_v7();
        pending.pool_id = "pool-a".into();
        pending.anchor_model_revision = "2026-07-01".into();
        pending.process_instance_id = identity.process_instance_id;
        pending.project_uuid = identity.project_uuid;
        pending.project_id = identity.project_id.clone();
        pending.config_generation_id = identity.config_generation_id.clone();
        pending.policy_version_id = pool.policy_version_id.clone();
        pending.learning_generation_id = pool.learning_generation_id;
        pending.request_projection = request_projection();
        pending.routing_context_projection.tenant_policy_hash = "3".repeat(64);
        pending.routing_context_projection.agent_policy_hash = "4".repeat(64);
        pending.replay_capability_facts =
            ReplayCapabilityFactsV1::from_capability(&LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: LlmApiFamily::OpenAIChatCompletions,
                transport_identity: "transport-shared".to_string(),
            })
            .unwrap();
        pending.candidate_facts = vec![PersistedCandidateFactV1 {
            schema: CANDIDATE_FACT_SCHEMA_V1.to_string(),
            candidate_id: "candidate-a".to_string(),
            model: "candidate-model-a".to_string(),
            model_revision: "2026-06-01".to_string(),
            cost_rank: 0,
            capabilities: PersistedCandidateCapabilitiesV1 {
                tools: true,
                multimodal_input: false,
                structured_output: false,
                reasoning_controls: false,
            },
            decoding_fingerprint: "1".repeat(64),
        }];
        let mut response_value = serde_json::to_value(&pending.normalized_anchor_response).unwrap();
        response_value
            .as_object_mut()
            .unwrap()
            .remove("semantic_response_fingerprint");
        pending
            .normalized_anchor_response
            .semantic_response_fingerprint = canonical_sha256(&response_value).unwrap();
        pending
    }

    fn request_projection() -> RouterRequestProjectionV1 {
        let mut projection = RouterRequestProjectionV1 {
            schema: REQUEST_PROJECTION_SCHEMA_V1.to_string(),
            family: LlmApiFamily::OpenAIChatCompletions,
            normalized_request: SanitizedAnnotatedLlmRequest {
                messages: Vec::new(),
                model: Some("anchor-a".to_string()),
                params: None,
                tools: None,
                tool_choice: None,
                response_format: None,
                truncation: None,
                reasoning: None,
                service_tier: None,
                parallel_tool_calls: None,
                max_output_tokens: None,
                max_tool_calls: None,
                top_logprobs: None,
            },
            ordered_instructions: Vec::new(),
            response_format: None,
            response_schema_fingerprint: None,
            required_capabilities: Vec::new(),
            sanitizer_version: ROUTER_SANITIZER_VERSION,
            semantic_request_fingerprint: String::new(),
        };
        refresh_request_fingerprint(&mut projection);
        projection
    }

    fn candidate_request_projection() -> RouterRequestProjectionV1 {
        let mut projection = request_projection();
        projection.normalized_request.model = Some("candidate-model-a".to_string());
        refresh_request_fingerprint(&mut projection);
        projection
    }

    fn refresh_request_fingerprint(projection: &mut RouterRequestProjectionV1) {
        projection.semantic_request_fingerprint.clear();
        let mut value = serde_json::to_value(&*projection).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("semantic_request_fingerprint");
        projection.semantic_request_fingerprint = canonical_sha256(&value).unwrap();
    }

    fn policy_evaluator_version(repository: &LedgerRepository) -> String {
        let policy_json: String = repository
            .connection
            .query_row(
                "SELECT canonical_policy_json FROM policy_versions WHERE pool_id = 'pool-a'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        serde_json::from_str::<Json>(&policy_json)
            .unwrap()
            .pointer("/pool/judge/policy_sha256")
            .and_then(Json::as_str)
            .unwrap()
            .to_string()
    }

    fn snapshot(ack: DependencyCommandAck) -> DependencyStateSnapshot {
        match ack {
            DependencyCommandAck::Applied(snapshot)
            | DependencyCommandAck::AlreadyApplied(snapshot) => snapshot,
            other => panic!("expected applied dependency acknowledgement, got {other:?}"),
        }
    }

    fn applied(ack: DependencyCommandAck) -> DependencyStateSnapshot {
        match ack {
            DependencyCommandAck::Applied(snapshot) => snapshot,
            other => panic!("expected newly applied dependency acknowledgement, got {other:?}"),
        }
    }

    #[test]
    fn startup_neutrally_orphans_an_admitted_dependency_operation() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut config = config(&path, "cooloff-startup-orphan");
        config.pools[0].judge.base_cooloff_seconds = 2;
        config.pools[0].judge.max_cooloff_seconds = 8;
        let mut origin = LedgerRepository::activate_at(&config, FIXTURE_CREATED_AT).unwrap();
        let origin_identity = origin.identity.clone();
        let target = seed_target(&mut origin.repository, &origin_identity, "startup-orphan");
        let dependency = candidate_identity("startup-orphan");
        let failed_operation = FakeClock::at(FIXTURE_CREATED_AT + 3).operation(target, 2, 8);
        assert_eq!(
            applied(
                origin
                    .repository
                    .claim_candidate_dependency(&dependency, &failed_operation)
                    .unwrap()
            )
            .transition,
            DependencyTransition::Admitted
        );
        let failure = FakeClock::at(FIXTURE_CREATED_AT + 4).failure(&failed_operation);
        let failed = applied(origin.repository.complete_dependency(&failure).unwrap());
        assert_eq!(failed.consecutive_failures, 1);
        let inherited_cooloff = failed.cooloff_until_unix_ms;
        let operation = FakeClock::at(FIXTURE_CREATED_AT + 3_000).operation(target, 2, 8);
        let admitted = applied(
            origin
                .repository
                .claim_candidate_dependency(&dependency, &operation)
                .unwrap(),
        );
        assert_eq!(admitted.transition, DependencyTransition::Admitted);
        assert_eq!(admitted.consecutive_failures, 1);
        assert_eq!(admitted.cooloff_until_unix_ms, inherited_cooloff);
        drop(origin);

        let mut recovered = LedgerRepository::activate_at(
            &config,
            FIXTURE_CREATED_AT + super::super::PROCESS_HEARTBEAT_MILLIS,
        )
        .unwrap();
        let terminal = load_terminal_state(
            &recovered.repository.connection,
            operation.dependency_operation_id,
        )
        .unwrap()
        .unwrap();
        assert_eq!(terminal.transition, DependencyTransition::OrphanedInFlight);
        assert_eq!(terminal.consecutive_failures, 1);
        assert_eq!(terminal.cooloff_until_unix_ms, inherited_cooloff);
        assert_eq!(terminal.failure_class, None);
        assert_reconciliation_noop_twice(
            &mut recovered.repository,
            FIXTURE_CREATED_AT + super::super::PROCESS_HEARTBEAT_MILLIS,
        );
    }

    #[test]
    fn startup_rejects_future_dependency_claim_and_rolls_back_recovery() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut config = config(&path, "cooloff-future-recovery");
        config.pools[0].judge.base_cooloff_seconds = 2;
        config.pools[0].judge.max_cooloff_seconds = 8;
        let mut origin = LedgerRepository::activate_at(&config, FIXTURE_CREATED_AT).unwrap();
        let origin_identity = origin.identity.clone();
        let target = seed_target(&mut origin.repository, &origin_identity, "future-recovery");
        let dependency = candidate_identity("future-recovery");
        let operation = FakeClock::at(FIXTURE_CREATED_AT + 100_000).operation(target, 2, 8);
        applied(
            origin
                .repository
                .claim_candidate_dependency(&dependency, &operation)
                .unwrap(),
        );
        let before = recovery_snapshot(&origin.repository.connection);
        drop(origin);

        let error = match LedgerRepository::activate_at(
            &config,
            FIXTURE_CREATED_AT + super::super::PROCESS_HEARTBEAT_MILLIS,
        ) {
            Ok(_) => panic!("future dependency claim must reject recovery"),
            Err(error) => error,
        };
        assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);
        let connection = Connection::open(&path).unwrap();
        assert_eq!(recovery_snapshot(&connection), before);
    }

    #[test]
    fn dependency_completion_cannot_precede_its_admitted_claim() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "cooloff-completion-chronology");
        let identity = activated.identity.clone();
        let target = seed_target(
            &mut activated.repository,
            &identity,
            "completion-chronology",
        );
        let dependency = candidate_identity("completion-chronology");
        let operation = FakeClock::at(100).operation(target, 2, 8);
        applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &operation)
                .unwrap(),
        );
        let completion = FakeClock::at(99).success(&operation);
        assert_eq!(
            activated
                .repository
                .complete_dependency(&completion)
                .unwrap(),
            DependencyCommandAck::Conflict
        );
        assert!(
            load_terminal_state(
                &activated.repository.connection,
                operation.dependency_operation_id,
            )
            .unwrap()
            .is_none()
        );
    }

    #[test]
    fn candidate_and_judge_identity_hashes_are_exact_and_isolated() {
        let candidate = CandidateDependencyIdentity::new(
            LlmApiFamily::OpenAIChatCompletions,
            "transport-a",
            "candidate-model",
            "rev-1",
        )
        .unwrap();
        assert_eq!(
            candidate.0.canonical_identity_json,
            r#"{"api_family":"openai_chat_completions","candidate_model":"candidate-model","candidate_model_revision":"rev-1","transport_identity":"transport-a"}"#
        );
        assert_eq!(
            candidate.dependency_key_id(),
            "f238a37d93ee60e65318f5da08168568a2e114b9ed996970e2b6b26e33cfc8aa"
        );

        let judge = JudgeDependencyIdentity::new(
            LlmApiFamily::OpenAIChatCompletions,
            "transport-a",
            "judge-model",
            "judge-rev",
            "prompt-v1",
            "1".repeat(64),
            "rubric-v1",
            "2".repeat(64),
            1,
            "3".repeat(64),
        )
        .unwrap();
        assert_eq!(
            judge.dependency_key_id(),
            "412af2dc358a8a44ed6c07951f3adef353180f06bb83e5b69f12ec1325e552f0"
        );
        assert!(!judge.0.canonical_identity_json.contains("candidate"));
        assert_ne!(candidate.dependency_key_id(), judge.dependency_key_id());
    }

    #[test]
    fn dependency_failure_classes_are_closed_and_exhaustive() {
        for value in [
            "router.provider.transport",
            "router.provider.authentication",
            "router.provider.rate_limited",
            "router.provider.timeout",
            "router.provider.canceled",
            "router.provider.unreadable_response",
            "router.provider.truncated_response",
            "router.provider.ambiguous_decode",
            "router.provider.evidence_bound_exceeded",
            "router.provider.middleware_interference",
            "router.judge.output_invalid",
        ] {
            assert_eq!(DependencyFailureClass::new(value).unwrap().as_str(), value);
        }
        assert!(DependencyFailureClass::new("router.provider.invented").is_err());
    }

    #[test]
    fn retention_carries_all_four_terminal_dependency_states() {
        for expected_transition in [
            DependencyTransition::Success,
            DependencyTransition::Failure,
            DependencyTransition::SkippedCooloff,
            DependencyTransition::OrphanedInFlight,
        ] {
            let temporary = tempdir().unwrap();
            let path = database_path(&temporary);
            let mut activated = activate(&path, &format!("carry-{}", expected_transition.as_str()));
            let identity = activated.identity.clone();
            let dependency = candidate_identity(expected_transition.as_str());

            if expected_transition == DependencyTransition::SkippedCooloff {
                let prior_target =
                    seed_target(&mut activated.repository, &identity, "prior-failure");
                let prior = FakeClock::at(0).operation(prior_target, 2, 8);
                applied(
                    activated
                        .repository
                        .claim_candidate_dependency(&dependency, &prior)
                        .unwrap(),
                );
                applied(
                    activated
                        .repository
                        .complete_dependency(&FakeClock::at(1).failure(&prior))
                        .unwrap(),
                );
            }

            let target = seed_target(
                &mut activated.repository,
                &identity,
                expected_transition.as_str(),
            );
            let operation = FakeClock::at(2).operation(target, 2, 8);
            let claim = applied(
                activated
                    .repository
                    .claim_candidate_dependency(&dependency, &operation)
                    .unwrap(),
            );
            let terminal = match expected_transition {
                DependencyTransition::Success => applied(
                    activated
                        .repository
                        .complete_dependency(&FakeClock::at(3).success(&operation))
                        .unwrap(),
                ),
                DependencyTransition::Failure => applied(
                    activated
                        .repository
                        .complete_dependency(&FakeClock::at(3).failure(&operation))
                        .unwrap(),
                ),
                DependencyTransition::SkippedCooloff => claim,
                DependencyTransition::OrphanedInFlight => {
                    assert_eq!(claim.transition, DependencyTransition::Admitted);
                    let state_event_id = Uuid::now_v7();
                    let payload_hash = state_hash(
                        state_event_id,
                        dependency.dependency_key_id(),
                        Some(operation.dependency_operation_id),
                        Some(target.anchor_id),
                        DependencyTransition::OrphanedInFlight,
                        claim.consecutive_failures,
                        claim.cooloff_until_unix_ms,
                        None,
                        3,
                    )
                    .unwrap();
                    assert!(
                        insert_state(
                            &activated.repository.connection,
                            state_event_id,
                            dependency.dependency_key_id(),
                            operation.dependency_operation_id,
                            target.anchor_id,
                            DependencyTransition::OrphanedInFlight,
                            claim.consecutive_failures,
                            claim.cooloff_until_unix_ms,
                            None,
                            3,
                            &payload_hash,
                        )
                        .unwrap()
                    );
                    DependencyStateSnapshot {
                        dependency_key_id: dependency.dependency_key_id().to_string(),
                        dependency_operation_id: operation.dependency_operation_id,
                        transition: DependencyTransition::OrphanedInFlight,
                        consecutive_failures: claim.consecutive_failures,
                        cooloff_until_unix_ms: claim.cooloff_until_unix_ms,
                        failure_class: None,
                    }
                }
                DependencyTransition::Admitted => unreachable!(),
            };
            assert_eq!(terminal.transition, expected_transition);

            let selected = BTreeSet::from([target.anchor_id.to_string()]);
            let carried = carry_forward_latest_dependency_states_in_transaction(
                &activated.repository.connection,
                identity.project_uuid,
                &selected,
                4,
            )
            .unwrap();
            assert_eq!(
                carried,
                BTreeSet::from([dependency.dependency_key_id().to_string()])
            );
            activated
                .repository
                .connection
                .execute(
                    "DELETE FROM anchors WHERE anchor_id = ?1",
                    params![target.anchor_id.to_string()],
                )
                .unwrap();
            verify_retained_dependency_states_in_transaction(
                &activated.repository.connection,
                identity.project_uuid,
                &carried,
            )
            .unwrap();
            let baseline = activated
                .repository
                .connection
                .query_row(
                    "SELECT dependency_operation_id, anchor_id, state,
                            consecutive_failures, cooloff_until_unix_ms, failure_class
                     FROM dependency_state_events
                     WHERE dependency_key_id = ?1
                     ORDER BY event_seq DESC LIMIT 1",
                    params![dependency.dependency_key_id()],
                    |row| {
                        Ok((
                            row.get::<_, Option<String>>(0)?,
                            row.get::<_, Option<String>>(1)?,
                            row.get::<_, String>(2)?,
                            row.get::<_, i64>(3)?,
                            row.get::<_, Option<i64>>(4)?,
                            row.get::<_, Option<String>>(5)?,
                        ))
                    },
                )
                .unwrap();
            assert_eq!(baseline.0, None);
            assert_eq!(baseline.1, None);
            assert_eq!(baseline.2, expected_transition.as_str());
            assert_eq!(baseline.3, terminal.consecutive_failures);
            assert_eq!(baseline.4, terminal.cooloff_until_unix_ms);
            assert_eq!(baseline.5, terminal.failure_class);
        }
    }

    #[test]
    fn retention_checkpoints_a_deleted_prefix_before_a_retained_admitted_operation() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "retention-shared-prefix");
        let identity = activated.identity.clone();
        let dependency = candidate_identity("shared-prefix");

        let selected_target = seed_target(&mut activated.repository, &identity, "selected-prefix");
        let selected_operation = FakeClock::at(0).operation(selected_target, 2, 8);
        applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &selected_operation)
                .unwrap(),
        );
        let failed = applied(
            activated
                .repository
                .complete_dependency(&FakeClock::at(1).failure(&selected_operation))
                .unwrap(),
        );
        assert_eq!(failed.transition, DependencyTransition::Failure);

        let retained_target = seed_target(&mut activated.repository, &identity, "retained-suffix");
        let retained_operation = FakeClock::at(3_000).operation(retained_target, 2, 8);
        let admitted = applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &retained_operation)
                .unwrap(),
        );
        assert_eq!(admitted.transition, DependencyTransition::Admitted);
        assert_eq!(admitted.consecutive_failures, failed.consecutive_failures);
        assert_eq!(admitted.cooloff_until_unix_ms, failed.cooloff_until_unix_ms);

        let selected = BTreeSet::from([selected_target.anchor_id.to_string()]);
        let touched = carry_forward_latest_dependency_states_in_transaction(
            &activated.repository.connection,
            identity.project_uuid,
            &selected,
            3_000,
        )
        .unwrap();
        assert_eq!(
            touched,
            BTreeSet::from([dependency.dependency_key_id().to_string()])
        );
        activated
            .repository
            .connection
            .execute(
                "DELETE FROM anchors WHERE anchor_id = ?1",
                params![selected_target.anchor_id.to_string()],
            )
            .unwrap();
        verify_retained_dependency_states_in_transaction(
            &activated.repository.connection,
            identity.project_uuid,
            &touched,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM dependency_operations WHERE anchor_id = ?1",
                    params![retained_target.anchor_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );

        let completion = FakeClock::at(3_001).success(&retained_operation);
        let completed = applied(
            activated
                .repository
                .complete_dependency(&completion)
                .unwrap(),
        );
        assert_eq!(completed.transition, DependencyTransition::Success);
        assert_eq!(completed.consecutive_failures, 0);
    }

    #[test]
    fn retention_checkpoint_preserves_an_older_outstanding_operation_after_newer_reset() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "retention-overlapping-operations");
        let identity = activated.identity.clone();
        let dependency = candidate_identity("overlap-reset");

        let selected_target = seed_target(&mut activated.repository, &identity, "selected-a1");
        let first_operation = FakeClock::at(0).operation(selected_target, 2, 8);
        applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &first_operation)
                .unwrap(),
        );
        let first_failure = applied(
            activated
                .repository
                .complete_dependency(&FakeClock::at(1).failure(&first_operation))
                .unwrap(),
        );
        assert_eq!(first_failure.consecutive_failures, 1);

        let second_target = seed_target(&mut activated.repository, &identity, "retained-a2");
        let second_operation = FakeClock::at(3_000).operation(second_target, 2, 8);
        let second_claim = applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &second_operation)
                .unwrap(),
        );
        assert_eq!(second_claim.transition, DependencyTransition::Admitted);
        assert_eq!(second_claim.consecutive_failures, 1);

        let third_target = seed_target(&mut activated.repository, &identity, "retained-a3");
        let third_operation = FakeClock::at(3_001).operation(third_target, 2, 8);
        let third_claim = applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &third_operation)
                .unwrap(),
        );
        assert_eq!(third_claim.transition, DependencyTransition::Admitted);
        assert_eq!(third_claim.consecutive_failures, 1);
        let third_success = applied(
            activated
                .repository
                .complete_dependency(&FakeClock::at(3_002).success(&third_operation))
                .unwrap(),
        );
        assert_eq!(third_success.consecutive_failures, 0);

        let selected = BTreeSet::from([selected_target.anchor_id.to_string()]);
        let touched = carry_forward_latest_dependency_states_in_transaction(
            &activated.repository.connection,
            identity.project_uuid,
            &selected,
            3_003,
        )
        .unwrap();
        activated
            .repository
            .connection
            .execute(
                "DELETE FROM anchors WHERE anchor_id = ?1",
                params![selected_target.anchor_id.to_string()],
            )
            .unwrap();
        verify_retained_dependency_states_in_transaction(
            &activated.repository.connection,
            identity.project_uuid,
            &touched,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM dependency_operations
                     WHERE anchor_id IN (?1, ?2)",
                    params![
                        second_target.anchor_id.to_string(),
                        third_target.anchor_id.to_string()
                    ],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            2
        );

        let second_failure = applied(
            activated
                .repository
                .complete_dependency(&FakeClock::at(3_004).failure(&second_operation))
                .unwrap(),
        );
        assert_eq!(second_failure.transition, DependencyTransition::Failure);
        assert_eq!(second_failure.consecutive_failures, 1);
        assert_eq!(second_failure.cooloff_until_unix_ms, Some(5_004));
    }

    #[test]
    fn successive_retention_checkpoints_reset_failure_to_success_or_new_failure() {
        for second_outcome in [DependencyTransition::Success, DependencyTransition::Failure] {
            let temporary = tempdir().unwrap();
            let path = database_path(&temporary);
            let mut activated = activate(
                &path,
                &format!("checkpoint-reset-{}", second_outcome.as_str()),
            );
            let identity = activated.identity.clone();
            let dependency = candidate_identity(second_outcome.as_str());

            let first_target =
                seed_target(&mut activated.repository, &identity, "checkpoint-first");
            let first_operation = FakeClock::at(0).operation(first_target, 2, 8);
            applied(
                activated
                    .repository
                    .claim_candidate_dependency(&dependency, &first_operation)
                    .unwrap(),
            );
            let first_failure = applied(
                activated
                    .repository
                    .complete_dependency(&FakeClock::at(1).failure(&first_operation))
                    .unwrap(),
            );
            let first_selected = BTreeSet::from([first_target.anchor_id.to_string()]);
            let first_checkpoint = carry_forward_latest_dependency_states_in_transaction(
                &activated.repository.connection,
                identity.project_uuid,
                &first_selected,
                2,
            )
            .unwrap();
            activated
                .repository
                .connection
                .execute(
                    "DELETE FROM anchors WHERE anchor_id = ?1",
                    params![first_target.anchor_id.to_string()],
                )
                .unwrap();
            verify_retained_dependency_states_in_transaction(
                &activated.repository.connection,
                identity.project_uuid,
                &first_checkpoint,
            )
            .unwrap();

            let second_target =
                seed_target(&mut activated.repository, &identity, "checkpoint-second");
            let second_operation = FakeClock::at(3_000).operation(second_target, 2, 8);
            let second_claim = applied(
                activated
                    .repository
                    .claim_candidate_dependency(&dependency, &second_operation)
                    .unwrap(),
            );
            assert_eq!(second_claim.transition, DependencyTransition::Admitted);
            assert_eq!(
                second_claim.consecutive_failures,
                first_failure.consecutive_failures
            );
            let second_terminal = match second_outcome {
                DependencyTransition::Success => applied(
                    activated
                        .repository
                        .complete_dependency(&FakeClock::at(3_001).success(&second_operation))
                        .unwrap(),
                ),
                DependencyTransition::Failure => applied(
                    activated
                        .repository
                        .complete_dependency(&FakeClock::at(3_001).failure(&second_operation))
                        .unwrap(),
                ),
                _ => unreachable!(),
            };
            let second_selected = BTreeSet::from([second_target.anchor_id.to_string()]);
            let second_checkpoint = carry_forward_latest_dependency_states_in_transaction(
                &activated.repository.connection,
                identity.project_uuid,
                &second_selected,
                3_002,
            )
            .unwrap();
            activated
                .repository
                .connection
                .execute(
                    "DELETE FROM anchors WHERE anchor_id = ?1",
                    params![second_target.anchor_id.to_string()],
                )
                .unwrap();
            verify_retained_dependency_states_in_transaction(
                &activated.repository.connection,
                identity.project_uuid,
                &second_checkpoint,
            )
            .unwrap();

            let checkpoints = activated
                .repository
                .connection
                .prepare(
                    "SELECT state, consecutive_failures, cooloff_until_unix_ms
                     FROM dependency_state_events
                     WHERE dependency_key_id = ?1
                     ORDER BY event_seq",
                )
                .unwrap()
                .query_map(params![dependency.dependency_key_id()], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, i64>(1)?,
                        row.get::<_, Option<i64>>(2)?,
                    ))
                })
                .unwrap()
                .collect::<rusqlite::Result<Vec<_>>>()
                .unwrap();
            assert_eq!(checkpoints.len(), 2);
            assert_eq!(checkpoints[0].0, "failure");
            assert_eq!(checkpoints[1].0, second_outcome.as_str());
            assert_eq!(checkpoints[1].1, second_terminal.consecutive_failures);
            assert_eq!(checkpoints[1].2, second_terminal.cooloff_until_unix_ms);
        }
    }

    #[test]
    fn retention_carries_latest_dependency_state_across_anchor_deletion() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut config = config(&path, "cooloff-retention");
        config.max_evidence_records = 1;
        config.pools[0].judge.base_cooloff_seconds = 2;
        config.pools[0].judge.max_cooloff_seconds = 8;
        let mut activated = LedgerRepository::activate_at(&config, FIXTURE_CREATED_AT).unwrap();
        let identity = activated.identity.clone();
        let target = seed_target(&mut activated.repository, &identity, "retained");
        let dependency = candidate_identity("retained");
        let operation = FakeClock::at(FIXTURE_CREATED_AT + 3).operation(target, 2, 8);
        assert_eq!(
            applied(
                activated
                    .repository
                    .claim_candidate_dependency(&dependency, &operation)
                    .unwrap()
            )
            .transition,
            DependencyTransition::Admitted
        );
        let failure = FakeClock::at(FIXTURE_CREATED_AT + 4).failure(&operation);
        let failed = applied(activated.repository.complete_dependency(&failure).unwrap());
        assert_eq!(failed.transition, DependencyTransition::Failure);
        assert_eq!(failed.consecutive_failures, 1);
        assert_eq!(
            activated
                .repository
                .start_shadow_attempt(
                    ShadowAttemptStarted::new(
                        target.shadow_attempt_id,
                        Uuid::now_v7(),
                        Uuid::now_v7(),
                        FIXTURE_CREATED_AT + 3,
                    )
                    .unwrap(),
                )
                .unwrap(),
            ShadowCommandAck::Applied
        );
        let sample_batch_id = activated
            .repository
            .connection
            .query_row(
                "SELECT sample_batch_id FROM shadow_attempts WHERE shadow_attempt_id = ?1",
                params![target.shadow_attempt_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let terminal_at = FIXTURE_CREATED_AT + 5;
        let terminal = ShadowTerminalRecord::new(
            Uuid::now_v7(),
            target.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            ShadowTerminalClass::OperationalFailure,
            None,
            None,
            None,
            Some(ShadowOperationalFailureClass::new("router.provider.timeout").unwrap()),
            Some(1),
            None,
            None,
            ShadowVectorSourceV1::Canonicalizable {
                query_inputs: Box::new(candidate_request_projection()),
            },
            Some(
                SampleBatchTerminalEvent::new(
                    Uuid::parse_str(&sample_batch_id).unwrap(),
                    Uuid::now_v7(),
                    SampleBatchTerminalState::Closed,
                    None,
                    terminal_at,
                )
                .unwrap(),
            ),
            terminal_at,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_shadow_terminal(&terminal)
                .unwrap(),
            ShadowCommandAck::Applied
        );

        let retention =
            RetentionRequest::new(Uuid::now_v7(), Uuid::now_v7(), FIXTURE_CREATED_AT + 6).unwrap();
        let RetentionAck::Applied { summary, .. } =
            activated.repository.run_retention(&retention).unwrap()
        else {
            panic!("retention should remove the fully terminal anchor");
        };
        assert_eq!(summary.selected_count, 1);
        assert_eq!(
            activated
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM dependency_operations WHERE anchor_id = ?1",
                    params![target.anchor_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );
        let carried = activated
            .repository
            .connection
            .query_row(
                "SELECT dependency_operation_id, anchor_id, state,
                        consecutive_failures, cooloff_until_unix_ms
                 FROM dependency_state_events WHERE dependency_key_id = ?1",
                params![dependency.dependency_key_id()],
                |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, String>(2)?,
                        row.get::<_, i64>(3)?,
                        row.get::<_, Option<i64>>(4)?,
                    ))
                },
            )
            .unwrap();
        assert_eq!(
            carried,
            (
                None,
                None,
                "failure".to_string(),
                failed.consecutive_failures,
                failed.cooloff_until_unix_ms,
            )
        );

        let next_target = seed_target(&mut activated.repository, &identity, "next");
        let next_operation = FakeClock::at(FIXTURE_CREATED_AT + 7).operation(next_target, 2, 8);
        let next = applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &next_operation)
                .unwrap(),
        );
        assert_eq!(next.transition, DependencyTransition::SkippedCooloff);
        assert_eq!(next.consecutive_failures, failed.consecutive_failures);
        assert_eq!(next.cooloff_until_unix_ms, failed.cooloff_until_unix_ms);
    }

    #[test]
    fn schema_rejects_admitted_and_skipped_claims_for_one_operation() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "cooloff-exclusive-claim");
        let identity = activated.identity.clone();
        let target = seed_target(&mut activated.repository, &identity, "exclusive-claim");
        let dependency = candidate_identity("exclusive-claim");
        let operation = FakeClock::at(100).operation(target, 2, 8);
        let admitted = applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &operation)
                .unwrap(),
        );
        assert_eq!(admitted.transition, DependencyTransition::Admitted);

        let forged_event_id = Uuid::now_v7();
        let forged_hash = state_hash(
            forged_event_id,
            dependency.dependency_key_id(),
            Some(operation.dependency_operation_id),
            Some(target.anchor_id),
            DependencyTransition::SkippedCooloff,
            1,
            Some(2_100),
            None,
            101,
        )
        .unwrap();
        let inserted = activated.repository.connection.execute(
            "INSERT INTO dependency_state_events (
                dependency_state_event_id, dependency_key_id,
                dependency_operation_id, anchor_id, state,
                consecutive_failures, cooloff_until_unix_ms, failure_class,
                created_at_unix_ms, canonical_payload_hash
             ) VALUES (?1, ?2, ?3, ?4, 'skipped_cooloff', 1, 2100, NULL, 101, ?5)",
            params![
                forged_event_id.to_string(),
                dependency.dependency_key_id(),
                operation.dependency_operation_id.to_string(),
                target.anchor_id.to_string(),
                forged_hash,
            ],
        );
        assert!(inserted.is_err());
        let claim_count: i64 = activated
            .repository
            .connection
            .query_row(
                "SELECT count(*) FROM dependency_state_events
                 WHERE dependency_operation_id = ?1
                   AND state IN ('admitted', 'skipped_cooloff')",
                [operation.dependency_operation_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(claim_count, 1);
    }

    #[test]
    fn first_failure_expiry_boundary_and_two_connection_visibility_are_exact() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut first = activate(&path, "cooloff-shared");
        let mut second = activate(&path, "cooloff-shared");
        let first_identity = first.identity.clone();
        let second_identity = second.identity.clone();
        let first_target = seed_target(&mut first.repository, &first_identity, "first");
        let second_target = seed_target(&mut second.repository, &second_identity, "second");
        let dependency = candidate_identity("shared");

        let operation = FakeClock::at(1_000).operation(first_target, 2, 8);
        assert_eq!(
            applied(
                first
                    .repository
                    .claim_candidate_dependency(&dependency, &operation)
                    .unwrap()
            )
            .transition,
            DependencyTransition::Admitted
        );
        let failure = FakeClock::at(1_100).failure(&operation);
        let failed = applied(first.repository.complete_dependency(&failure).unwrap());
        assert_eq!(failed.consecutive_failures, 1);
        assert_eq!(failed.cooloff_until_unix_ms, Some(3_100));

        let before_boundary = FakeClock::at(3_099).operation(second_target, 2, 8);
        assert_eq!(
            applied(
                second
                    .repository
                    .claim_candidate_dependency(&dependency, &before_boundary)
                    .unwrap()
            )
            .transition,
            DependencyTransition::SkippedCooloff
        );
        let at_boundary = FakeClock::at(3_100).operation(second_target, 2, 8);
        let boundary = applied(
            second
                .repository
                .claim_candidate_dependency(&dependency, &at_boundary)
                .unwrap(),
        );
        assert_eq!(boundary.transition, DependencyTransition::Admitted);
        assert_eq!(boundary.consecutive_failures, 1);
    }

    #[test]
    fn exponential_math_saturates_caps_and_success_resets_the_sequence() {
        assert_eq!(saturated_cooloff_seconds(2, 300, 1), 2);
        assert_eq!(saturated_cooloff_seconds(2, 300, 2), 4);
        assert_eq!(saturated_cooloff_seconds(2, 5, 3), 5);
        assert_eq!(saturated_cooloff_seconds(2, 300, i64::MAX), 300);
        assert_eq!(
            saturated_cooloff_seconds(i64::MAX / 2, i64::MAX, 4),
            i64::MAX
        );

        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate_with_cooloff(&path, "cooloff-cap-reset", 2, 5);
        let identity = activated.identity.clone();
        let target = seed_target(&mut activated.repository, &identity, "cap");
        let dependency = candidate_identity("cap");
        let mut clock = FakeClock::at(0);

        let first = clock.operation(target, 2, 5);
        applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &first)
                .unwrap(),
        );
        let first_failure = clock.failure(&first);
        assert_eq!(
            applied(
                activated
                    .repository
                    .complete_dependency(&first_failure)
                    .unwrap()
            )
            .cooloff_until_unix_ms,
            Some(2_000)
        );

        clock.set(2_000);
        let second = clock.operation(target, 2, 5);
        applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &second)
                .unwrap(),
        );
        let second_failure = clock.failure(&second);
        assert_eq!(
            applied(
                activated
                    .repository
                    .complete_dependency(&second_failure)
                    .unwrap()
            )
            .cooloff_until_unix_ms,
            Some(6_000)
        );

        clock.set(6_000);
        let third = clock.operation(target, 2, 5);
        applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &third)
                .unwrap(),
        );
        let third_failure = clock.failure(&third);
        assert_eq!(
            applied(
                activated
                    .repository
                    .complete_dependency(&third_failure)
                    .unwrap()
            )
            .cooloff_until_unix_ms,
            Some(11_000)
        );

        clock.set(11_000);
        let reset = clock.operation(target, 2, 5);
        applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &reset)
                .unwrap(),
        );
        let reset_success = clock.success(&reset);
        let reset_state = applied(
            activated
                .repository
                .complete_dependency(&reset_success)
                .unwrap(),
        );
        assert_eq!(reset_state.consecutive_failures, 0);
        assert_eq!(reset_state.cooloff_until_unix_ms, None);

        clock.set(11_001);
        let after_reset = clock.operation(target, 2, 5);
        applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &after_reset)
                .unwrap(),
        );
        let after_reset_failure = clock.failure(&after_reset);
        assert_eq!(
            applied(
                activated
                    .repository
                    .complete_dependency(&after_reset_failure)
                    .unwrap()
            )
            .cooloff_until_unix_ms,
            Some(13_001)
        );
    }

    #[test]
    fn copied_and_rehashed_latest_state_forgeries_cannot_steer_admission() {
        for rehash in [false, true] {
            let temporary = tempdir().unwrap();
            let path = database_path(&temporary);
            let project_id = if rehash {
                "cooloff-rehashed-state"
            } else {
                "cooloff-copied-hash-state"
            };
            let mut activated = activate(&path, project_id);
            let identity = activated.identity.clone();
            let target = seed_target(&mut activated.repository, &identity, project_id);
            let dependency = candidate_identity(project_id);
            let operation = FakeClock::at(0).operation(target, 2, 8);
            applied(
                activated
                    .repository
                    .claim_candidate_dependency(&dependency, &operation)
                    .unwrap(),
            );
            let failure = FakeClock::at(100).failure(&operation);
            let failed = applied(activated.repository.complete_dependency(&failure).unwrap());
            assert_eq!(failed.consecutive_failures, 1);
            assert_eq!(failed.cooloff_until_unix_ms, Some(2_100));

            let forged_hash = rehash.then(|| {
                state_hash(
                    failure.dependency_state_event_id,
                    dependency.dependency_key_id(),
                    Some(operation.dependency_operation_id),
                    Some(target.anchor_id),
                    DependencyTransition::Failure,
                    2,
                    Some(4_100),
                    Some("router.provider.timeout"),
                    failure.created_at_unix_ms,
                )
                .unwrap()
            });
            activated
                .repository
                .connection
                .execute(
                    "UPDATE dependency_state_events
                     SET consecutive_failures = 2, cooloff_until_unix_ms = 4100,
                         canonical_payload_hash = coalesce(?1, canonical_payload_hash)
                     WHERE dependency_state_event_id = ?2",
                    params![forged_hash, failure.dependency_state_event_id.to_string()],
                )
                .unwrap();

            let next = FakeClock::at(3_000).operation(target, 2, 8);
            assert_eq!(
                activated
                    .repository
                    .claim_candidate_dependency(&dependency, &next)
                    .unwrap(),
                DependencyCommandAck::Conflict
            );
            let stored: i64 = activated
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM dependency_operations
                     WHERE dependency_operation_id = ?1",
                    [next.dependency_operation_id.to_string()],
                    |row| row.get(0),
                )
                .unwrap();
            assert_eq!(stored, 0);
        }
    }

    #[test]
    fn rehashed_claim_timestamp_must_still_match_its_operation() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "cooloff-rehashed-claim-time");
        let identity = activated.identity.clone();
        let target = seed_target(&mut activated.repository, &identity, "claim-time");
        let dependency = candidate_identity("claim-time");
        let operation = FakeClock::at(100).operation(target, 2, 8);
        applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &operation)
                .unwrap(),
        );
        let forged_hash = state_hash(
            operation.dependency_state_event_id,
            dependency.dependency_key_id(),
            Some(operation.dependency_operation_id),
            Some(target.anchor_id),
            DependencyTransition::Admitted,
            0,
            None,
            None,
            101,
        )
        .unwrap();
        activated
            .repository
            .connection
            .execute(
                "UPDATE dependency_state_events
                 SET created_at_unix_ms = 101, canonical_payload_hash = ?1
                 WHERE dependency_state_event_id = ?2",
                params![forged_hash, operation.dependency_state_event_id.to_string()],
            )
            .unwrap();

        let next = FakeClock::at(102).operation(target, 2, 8);
        assert_eq!(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &next)
                .unwrap(),
            DependencyCommandAck::Conflict
        );
    }

    #[test]
    fn rehashed_unknown_failure_class_cannot_steer_admission() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "cooloff-rehashed-class");
        let identity = activated.identity.clone();
        let target = seed_target(&mut activated.repository, &identity, "rehashed-class");
        let dependency = candidate_identity("rehashed-class");
        let operation = FakeClock::at(0).operation(target, 2, 8);
        applied(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &operation)
                .unwrap(),
        );
        let failure = FakeClock::at(100).failure(&operation);
        applied(activated.repository.complete_dependency(&failure).unwrap());

        let invented_class = "router.provider.invented";
        let forged_hash = state_hash(
            failure.dependency_state_event_id,
            dependency.dependency_key_id(),
            Some(operation.dependency_operation_id),
            Some(target.anchor_id),
            DependencyTransition::Failure,
            1,
            Some(2_100),
            Some(invented_class),
            100,
        )
        .unwrap();
        activated
            .repository
            .connection
            .execute(
                "UPDATE dependency_state_events
                 SET failure_class = ?1, canonical_payload_hash = ?2
                 WHERE dependency_state_event_id = ?3",
                params![
                    invented_class,
                    forged_hash,
                    failure.dependency_state_event_id.to_string(),
                ],
            )
            .unwrap();

        let next = FakeClock::at(3_000).operation(target, 2, 8);
        assert_eq!(
            activated
                .repository
                .claim_candidate_dependency(&dependency, &next)
                .unwrap(),
            DependencyCommandAck::Conflict
        );
    }

    #[test]
    fn exact_duplicates_are_stable_and_opposite_completion_records_one_health_fault() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "cooloff-idempotency");
        let identity = activated.identity.clone();
        let target = seed_target(&mut activated.repository, &identity, "idempotency");
        let dependency = candidate_identity("idempotency");
        let operation = FakeClock::at(100).operation(target, 2, 8);
        let first = activated
            .repository
            .claim_candidate_dependency(&dependency, &operation)
            .unwrap();
        assert!(matches!(first, DependencyCommandAck::Applied(_)));
        let duplicate = activated
            .repository
            .claim_candidate_dependency(&dependency, &operation)
            .unwrap();
        assert!(matches!(duplicate, DependencyCommandAck::AlreadyApplied(_)));

        let completion = FakeClock::at(200).success(&operation);
        assert!(matches!(
            activated
                .repository
                .complete_dependency(&completion)
                .unwrap(),
            DependencyCommandAck::Applied(_)
        ));
        assert!(matches!(
            activated
                .repository
                .complete_dependency(&completion)
                .unwrap(),
            DependencyCommandAck::AlreadyApplied(_)
        ));

        let opposite = FakeClock::at(201).failure(&operation);
        assert_eq!(
            activated.repository.complete_dependency(&opposite).unwrap(),
            DependencyCommandAck::Conflict
        );
        assert_eq!(
            activated.repository.complete_dependency(&opposite).unwrap(),
            DependencyCommandAck::Conflict
        );
        let health_count: i64 = activated
            .repository
            .connection
            .query_row(
                "SELECT count(*) FROM health_events WHERE stable_class = ?1",
                params![INTEGRITY_CONFLICT_CLASS],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(health_count, 1);
        let terminal_count: i64 = activated
            .repository
            .connection
            .query_row(
                "SELECT count(*) FROM dependency_state_events
                 WHERE dependency_operation_id = ?1 AND state IN ('success', 'failure')",
                params![operation.dependency_operation_id.to_string()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(terminal_count, 1);
    }

    #[test]
    fn first_use_key_rollback_still_commits_null_resolved_conflict_health() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "cooloff-first-use-conflict");
        let identity = activated.identity.clone();
        let target = seed_target(&mut activated.repository, &identity, "first-use-conflict");
        let existing_dependency = candidate_identity("existing");
        let rolled_back_dependency = judge_identity(&activated.repository);
        let operation = FakeClock::at(100).operation(target, 2, 8);

        assert!(matches!(
            activated
                .repository
                .claim_candidate_dependency(&existing_dependency, &operation)
                .unwrap(),
            DependencyCommandAck::Applied(_)
        ));
        assert_eq!(
            activated
                .repository
                .claim_judge_dependency(&rolled_back_dependency, &operation)
                .unwrap(),
            DependencyCommandAck::Conflict
        );
        assert_eq!(
            activated
                .repository
                .claim_judge_dependency(&rolled_back_dependency, &operation)
                .unwrap(),
            DependencyCommandAck::Conflict
        );

        let key_count: i64 = activated
            .repository
            .connection
            .query_row(
                "SELECT count(*) FROM dependency_keys WHERE dependency_key_id = ?1",
                [rolled_back_dependency.dependency_key_id()],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(key_count, 0);
        let conflict_refs: (Option<String>, Option<String>) = activated
            .repository
            .connection
            .query_row(
                "SELECT anchor_id, dependency_key_id FROM health_events
                 WHERE health_event_id = ?1",
                [operation.dependency_state_event_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(conflict_refs, (Some(target.anchor_id.to_string()), None));

        let later_operation = FakeClock::at(101).operation(target, 2, 8);
        assert!(matches!(
            activated
                .repository
                .claim_judge_dependency(&rolled_back_dependency, &later_operation)
                .unwrap(),
            DependencyCommandAck::Applied(_)
        ));
        assert_eq!(
            activated
                .repository
                .claim_judge_dependency(&rolled_back_dependency, &operation)
                .unwrap(),
            DependencyCommandAck::Conflict
        );

        activated
            .repository
            .connection
            .execute(
                "UPDATE health_events SET severity = 'warning'
                 WHERE health_event_id = ?1",
                [operation.dependency_state_event_id.to_string()],
            )
            .unwrap();
        let error = activated
            .repository
            .claim_judge_dependency(&rolled_back_dependency, &operation)
            .unwrap_err();
        assert_eq!(error.class(), LedgerErrorClass::IdentityInvariant);
    }

    #[test]
    fn candidate_and_judge_cooloff_are_independent() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate_with_cooloff(&path, "cooloff-key-isolation", 10, 100);
        let identity = activated.identity.clone();
        let target = seed_target(&mut activated.repository, &identity, "isolation");
        let candidate = candidate_identity("same-transport");
        let judge = judge_identity(&activated.repository);

        let candidate_operation = FakeClock::at(0).operation(target, 10, 100);
        applied(
            activated
                .repository
                .claim_candidate_dependency(&candidate, &candidate_operation)
                .unwrap(),
        );
        let candidate_failure = FakeClock::at(1).failure(&candidate_operation);
        applied(
            activated
                .repository
                .complete_dependency(&candidate_failure)
                .unwrap(),
        );
        let candidate_retry = FakeClock::at(2).operation(target, 10, 100);
        assert_eq!(
            applied(
                activated
                    .repository
                    .claim_candidate_dependency(&candidate, &candidate_retry)
                    .unwrap()
            )
            .transition,
            DependencyTransition::SkippedCooloff
        );

        let judge_operation = FakeClock::at(2).operation(target, 10, 100);
        assert_eq!(
            applied(
                activated
                    .repository
                    .claim_judge_dependency(&judge, &judge_operation)
                    .unwrap()
            )
            .transition,
            DependencyTransition::Admitted
        );
        assert_ne!(candidate.dependency_key_id(), judge.dependency_key_id());
    }

    #[test]
    fn overlapping_completions_follow_commit_order() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate_with_cooloff(&path, "cooloff-overlap", 10, 100);
        let identity = activated.identity.clone();
        let target = seed_target(&mut activated.repository, &identity, "overlap");

        let failure_then_success = candidate_identity("failure-success");
        let first = FakeClock::at(0).operation(target, 10, 100);
        let second = FakeClock::at(1).operation(target, 10, 100);
        applied(
            activated
                .repository
                .claim_candidate_dependency(&failure_then_success, &first)
                .unwrap(),
        );
        applied(
            activated
                .repository
                .claim_candidate_dependency(&failure_then_success, &second)
                .unwrap(),
        );
        let first_failure = FakeClock::at(10).failure(&first);
        applied(
            activated
                .repository
                .complete_dependency(&first_failure)
                .unwrap(),
        );
        let second_success = FakeClock::at(11).success(&second);
        let reset = applied(
            activated
                .repository
                .complete_dependency(&second_success)
                .unwrap(),
        );
        assert_eq!(reset.consecutive_failures, 0);
        let after_reset = FakeClock::at(12).operation(target, 10, 100);
        assert_eq!(
            applied(
                activated
                    .repository
                    .claim_candidate_dependency(&failure_then_success, &after_reset)
                    .unwrap()
            )
            .transition,
            DependencyTransition::Admitted
        );

        let success_then_failure = candidate_identity("success-failure");
        let third = FakeClock::at(20).operation(target, 10, 100);
        let fourth = FakeClock::at(21).operation(target, 10, 100);
        applied(
            activated
                .repository
                .claim_candidate_dependency(&success_then_failure, &third)
                .unwrap(),
        );
        applied(
            activated
                .repository
                .claim_candidate_dependency(&success_then_failure, &fourth)
                .unwrap(),
        );
        let third_success = FakeClock::at(30).success(&third);
        applied(
            activated
                .repository
                .complete_dependency(&third_success)
                .unwrap(),
        );
        let fourth_failure = FakeClock::at(31).failure(&fourth);
        let failed = applied(
            activated
                .repository
                .complete_dependency(&fourth_failure)
                .unwrap(),
        );
        assert_eq!(failed.consecutive_failures, 1);
        assert_eq!(failed.cooloff_until_unix_ms, Some(10_031));
    }

    #[test]
    fn terminalized_origin_cannot_claim_even_an_exact_duplicate() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut origin = activate(&path, "cooloff-liveness");
        let reconciler = activate(&path, "cooloff-liveness");
        let origin_identity = origin.identity.clone();
        let target = seed_target(&mut origin.repository, &origin_identity, "stale");
        let dependency = candidate_identity("stale");
        let operation = FakeClock::at(100).operation(target, 2, 8);
        applied(
            origin
                .repository
                .claim_candidate_dependency(&dependency, &operation)
                .unwrap(),
        );

        reconciler
            .repository
            .connection
            .execute(
                "INSERT INTO process_instance_state_events (
                    process_state_event_id, process_instance_id, state,
                    subject_process_instance_id, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, 'reconciled', ?3, 101, ?4)",
                params![
                    Uuid::now_v7().to_string(),
                    reconciler.identity.process_instance_id.to_string(),
                    origin.identity.process_instance_id.to_string(),
                    "a".repeat(64),
                ],
            )
            .unwrap();
        assert_eq!(
            origin
                .repository
                .claim_candidate_dependency(&dependency, &operation)
                .unwrap(),
            DependencyCommandAck::OriginatingProcessNotLive
        );
    }

    #[test]
    fn stable_failure_and_durable_numeric_inputs_are_validated() {
        assert!(DependencyFailureClass::new("router.provider.timeout").is_ok());
        assert!(DependencyFailureClass::new("Provider Timeout").is_err());
        assert!(DependencyFailureClass::new("x".repeat(129)).is_err());
        let target = Target {
            anchor_id: Uuid::now_v7(),
            shadow_attempt_id: Uuid::now_v7(),
        };
        assert!(
            DependencyOperation::new(
                Uuid::now_v7(),
                Uuid::now_v7(),
                target.anchor_id,
                target.shadow_attempt_id,
                0,
                1,
                0,
            )
            .is_err()
        );
        assert!(
            DependencyOperation::new(
                Uuid::now_v7(),
                Uuid::now_v7(),
                target.anchor_id,
                target.shadow_attempt_id,
                2,
                1,
                0,
            )
            .is_err()
        );
    }

    #[test]
    fn final_start_check_refuses_before_opening_a_transaction() {
        let temporary = tempdir().unwrap();
        let path = database_path(&temporary);
        let mut activated = activate(&path, "cooloff-start-check");
        let identity = activated.identity.clone();
        let target = seed_target(&mut activated.repository, &identity, "start-check");
        let dependency = candidate_identity("start-check");
        let operation = FakeClock::at(100).operation(target, 2, 8);
        assert_eq!(
            activated
                .repository
                .claim_candidate_dependency_with_start_check(
                    &dependency,
                    &operation,
                    || None::<()>,
                )
                .unwrap(),
            DependencyCommandAck::TransactionNotStarted
        );
        let operation_count: i64 = activated
            .repository
            .connection
            .query_row("SELECT count(*) FROM dependency_operations", [], |row| {
                row.get(0)
            })
            .unwrap();
        assert_eq!(operation_count, 0);
    }

    #[test]
    fn snapshot_helper_accepts_both_idempotent_success_variants() {
        let value = DependencyStateSnapshot {
            dependency_key_id: "a".repeat(64),
            dependency_operation_id: Uuid::now_v7(),
            transition: DependencyTransition::Success,
            consecutive_failures: 0,
            cooloff_until_unix_ms: None,
            failure_class: None,
        };
        assert_eq!(
            snapshot(DependencyCommandAck::AlreadyApplied(value.clone())),
            value
        );
    }
}
