// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Query-local Active promotion, invalidation, cooloff, and restoration facts.

use super::*;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveNeighborhoodKey {
    pub(crate) active_experiment_id: Uuid,
    pub(crate) canonical_query_hash: String,
    pub(crate) sorted_neighbor_hash: String,
    pub(crate) config_generation_id: String,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) cohort_generation_id: Uuid,
    pub(crate) active_authorization_state_event_id: Uuid,
}

impl ActiveNeighborhoodKey {
    pub(crate) fn identity_hash(&self) -> Result<String, LedgerError> {
        validate_neighborhood_key_shape(self)?;
        hash_json(json!({
            "shape": "active_neighborhood_identity_v1",
            "active_experiment_id": self.active_experiment_id,
            "canonical_query_hash": self.canonical_query_hash,
            "sorted_neighbor_hash": self.sorted_neighbor_hash,
            "config_generation_id": self.config_generation_id,
            "learning_generation_id": self.learning_generation_id,
            "cohort_generation_id": self.cohort_generation_id,
            "active_authorization_state_event_id": self.active_authorization_state_event_id,
        }))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveNeighborhoodState {
    Promoted,
    Invalidated,
    Cooloff,
    Restored,
}

impl ActiveNeighborhoodState {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Promoted => "promoted",
            Self::Invalidated => "invalidated",
            Self::Cooloff => "cooloff",
            Self::Restored => "restored",
        }
    }

    fn parse(value: &str) -> Result<Self, LedgerError> {
        match value {
            "promoted" => Ok(Self::Promoted),
            "invalidated" => Ok(Self::Invalidated),
            "cooloff" => Ok(Self::Cooloff),
            "restored" => Ok(Self::Restored),
            _ => Err(corrupt()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveNeighborhoodPromotion {
    pub(crate) active_neighborhood_state_event_id: Uuid,
    pub(crate) key: ActiveNeighborhoodKey,
    pub(crate) stable_reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveNeighborhoodInvalidation {
    pub(crate) invalidated_state_event_id: Uuid,
    pub(crate) cooloff_state_event_id: Uuid,
    pub(crate) key: ActiveNeighborhoodKey,
    pub(crate) cause_active_dispatch_id: Option<Uuid>,
    pub(crate) cause_outcome_id: Option<Uuid>,
    pub(crate) stable_reason: String,
    pub(crate) cooloff_duration_seconds: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveNeighborhoodSnapshot {
    pub(crate) active_experiment_id: Uuid,
    pub(crate) active_neighborhood_state_event_id: Uuid,
    pub(crate) neighborhood_identity_hash: String,
    pub(crate) state: ActiveNeighborhoodState,
    pub(crate) cooloff_until_unix_ms: Option<u64>,
    pub(crate) active_authorization_state_event_id: Uuid,
    pub(crate) authorizing: bool,
    pub(crate) restorable: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ActiveNeighborhoodMutationAck {
    Applied(ActiveNeighborhoodSnapshot),
    AlreadyApplied(ActiveNeighborhoodSnapshot),
    AlreadyCurrent(ActiveNeighborhoodSnapshot),
    CoolingOff(ActiveNeighborhoodSnapshot),
    AuthorityChanged,
    Conflict,
    TransactionNotStarted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ActiveNeighborhoodObserveAck {
    Current(ActiveNeighborhoodSnapshot),
    Absent {
        neighborhood_identity_hash: String,
        promotion_authorized: bool,
    },
    Unavailable,
    TransactionNotStarted,
}

impl LedgerRepository {
    pub(crate) fn promote_active_neighborhood(
        &mut self,
        promotion: &ActiveNeighborhoodPromotion,
    ) -> Result<ActiveNeighborhoodMutationAck, LedgerError> {
        self.promote_active_neighborhood_at(promotion, now_unix_ms()?)
    }

    pub(crate) fn promote_active_neighborhood_at(
        &mut self,
        promotion: &ActiveNeighborhoodPromotion,
        observed_at_unix_ms: i64,
    ) -> Result<ActiveNeighborhoodMutationAck, LedgerError> {
        self.promote_active_neighborhood_at_with_start_check(promotion, observed_at_unix_ms, || {
            Some(())
        })
    }

    pub(crate) fn promote_active_neighborhood_with_start_check<G: TransactionStartGuard>(
        &mut self,
        promotion: &ActiveNeighborhoodPromotion,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveNeighborhoodMutationAck, LedgerError> {
        self.promote_active_neighborhood_at_with_start_check(promotion, now_unix_ms()?, start_check)
    }

    fn promote_active_neighborhood_at_with_start_check<G: TransactionStartGuard>(
        &mut self,
        promotion: &ActiveNeighborhoodPromotion,
        observed_at_unix_ms: i64,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveNeighborhoodMutationAck, LedgerError> {
        validate_promotion(promotion)?;
        if observed_at_unix_ms < 0 {
            return Err(invariant());
        }
        let database_path = self.database_path.clone();
        super::super::enforce_sidecar_permissions(&database_path)
            .map_err(super::super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveNeighborhoodMutationAck::TransactionNotStarted);
        };
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| {
                super::super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
            })?;
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ActiveNeighborhoodMutationAck::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = promote_active_neighborhood_in_transaction(
            &transaction,
            self.project_uuid,
            self.process_instance_id,
            promotion,
            observed_at_unix_ms,
        )?;
        super::super::enforce_sidecar_permissions(&database_path)
            .map_err(super::super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }

    pub(crate) fn invalidate_active_neighborhood(
        &mut self,
        invalidation: &ActiveNeighborhoodInvalidation,
    ) -> Result<ActiveNeighborhoodMutationAck, LedgerError> {
        self.invalidate_active_neighborhood_at(invalidation, now_unix_ms()?)
    }

    pub(crate) fn invalidate_active_neighborhood_at(
        &mut self,
        invalidation: &ActiveNeighborhoodInvalidation,
        observed_at_unix_ms: i64,
    ) -> Result<ActiveNeighborhoodMutationAck, LedgerError> {
        self.invalidate_active_neighborhood_at_with_start_check(
            invalidation,
            observed_at_unix_ms,
            || Some(()),
        )
    }

    pub(crate) fn invalidate_active_neighborhood_with_start_check<G: TransactionStartGuard>(
        &mut self,
        invalidation: &ActiveNeighborhoodInvalidation,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveNeighborhoodMutationAck, LedgerError> {
        self.invalidate_active_neighborhood_at_with_start_check(
            invalidation,
            now_unix_ms()?,
            start_check,
        )
    }

    fn invalidate_active_neighborhood_at_with_start_check<G: TransactionStartGuard>(
        &mut self,
        invalidation: &ActiveNeighborhoodInvalidation,
        observed_at_unix_ms: i64,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveNeighborhoodMutationAck, LedgerError> {
        validate_invalidation(invalidation)?;
        if observed_at_unix_ms < 0 {
            return Err(invariant());
        }
        let database_path = self.database_path.clone();
        super::super::enforce_sidecar_permissions(&database_path)
            .map_err(super::super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveNeighborhoodMutationAck::TransactionNotStarted);
        };
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| {
                super::super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
            })?;
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ActiveNeighborhoodMutationAck::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = invalidate_active_neighborhood_in_transaction(
            &transaction,
            self.project_uuid,
            self.process_instance_id,
            invalidation,
            observed_at_unix_ms,
        )?;
        super::super::enforce_sidecar_permissions(&database_path)
            .map_err(super::super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }

    pub(crate) fn observe_active_neighborhood(
        &mut self,
        key: &ActiveNeighborhoodKey,
    ) -> Result<ActiveNeighborhoodObserveAck, LedgerError> {
        self.observe_active_neighborhood_at(key, now_unix_ms()?)
    }

    pub(crate) fn observe_active_neighborhood_at(
        &mut self,
        key: &ActiveNeighborhoodKey,
        observed_at_unix_ms: i64,
    ) -> Result<ActiveNeighborhoodObserveAck, LedgerError> {
        self.observe_active_neighborhood_with_start_check(key, observed_at_unix_ms, || Some(()))
    }

    pub(crate) fn observe_active_neighborhood_with_start_check<G: TransactionStartGuard>(
        &mut self,
        key: &ActiveNeighborhoodKey,
        observed_at_unix_ms: i64,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveNeighborhoodObserveAck, LedgerError> {
        validate_neighborhood_key_shape(key)?;
        if observed_at_unix_ms < 0 {
            return Err(invariant());
        }
        let database_path = self.database_path.clone();
        super::super::enforce_sidecar_permissions(&database_path)
            .map_err(super::super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveNeighborhoodObserveAck::TransactionNotStarted);
        };
        let transaction = self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|error| {
                super::super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
            })?;
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(ActiveNeighborhoodObserveAck::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = observe_active_neighborhood_in_transaction(
            &transaction,
            self.project_uuid,
            self.process_instance_id,
            key,
            observed_at_unix_ms,
        )?;
        super::super::enforce_sidecar_permissions(&database_path)
            .map_err(super::super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }
}

fn promote_active_neighborhood_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    promotion: &ActiveNeighborhoodPromotion,
    observed_at_unix_ms: i64,
) -> Result<ActiveNeighborhoodMutationAck, LedgerError> {
    validate_promotion(promotion)?;
    let identity_hash = promotion.key.identity_hash()?;
    if let Some(existing) =
        load_neighborhood_event_by_id(transaction, promotion.active_neighborhood_state_event_id)?
    {
        return Ok(
            if event_matches_promotion(&existing, promotion, &identity_hash) {
                ActiveNeighborhoodMutationAck::AlreadyApplied(snapshot_from_event(
                    &existing, true, false,
                )?)
            } else {
                ActiveNeighborhoodMutationAck::Conflict
            },
        );
    }
    if !neighborhood_key_exists(transaction, project_uuid, &promotion.key)? {
        return Ok(ActiveNeighborhoodMutationAck::AuthorityChanged);
    }
    let authorization = observe_active_authorization_in_transaction(
        transaction,
        project_uuid,
        process_instance_id,
        promotion.key.active_experiment_id,
        observed_at_unix_ms,
    )?;
    let authorization_current = match authorization {
        ActiveAuthorizationObserveAck::Current(snapshot) => {
            authorization_supports_neighborhood(transaction, &promotion.key, snapshot)?
        }
        _ => false,
    };
    if !authorization_current {
        return Ok(ActiveNeighborhoodMutationAck::AuthorityChanged);
    }
    let latest = latest_neighborhood_event(
        transaction,
        promotion.key.active_experiment_id,
        &identity_hash,
    )?;
    let (state, predecessor) = match latest {
        None => (ActiveNeighborhoodState::Promoted, None),
        Some(ref event)
            if matches!(
                event.state,
                ActiveNeighborhoodState::Promoted | ActiveNeighborhoodState::Restored
            ) =>
        {
            return Ok(ActiveNeighborhoodMutationAck::AlreadyCurrent(
                snapshot_from_event(event, true, false)?,
            ));
        }
        Some(ref event) if event.state == ActiveNeighborhoodState::Cooloff => {
            let cooloff_until = event.cooloff_until_unix_ms.ok_or_else(corrupt)?;
            if observed_at_unix_ms < cooloff_until {
                return Ok(ActiveNeighborhoodMutationAck::CoolingOff(
                    snapshot_from_event(event, false, false)?,
                ));
            }
            (ActiveNeighborhoodState::Restored, Some(event.event_id))
        }
        Some(_) => return Err(corrupt()),
    };
    insert_neighborhood_event(
        transaction,
        promotion.active_neighborhood_state_event_id,
        &promotion.key,
        &identity_hash,
        predecessor,
        state,
        None,
        None,
        None,
        &promotion.stable_reason,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    let event =
        load_neighborhood_event_by_id(transaction, promotion.active_neighborhood_state_event_id)?
            .ok_or_else(corrupt)?;
    Ok(ActiveNeighborhoodMutationAck::Applied(snapshot_from_event(
        &event, true, false,
    )?))
}

pub(crate) fn invalidate_active_neighborhood_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    invalidation: &ActiveNeighborhoodInvalidation,
    observed_at_unix_ms: i64,
) -> Result<ActiveNeighborhoodMutationAck, LedgerError> {
    validate_invalidation(invalidation)?;
    let identity_hash = invalidation.key.identity_hash()?;
    let replay_invalidated =
        load_neighborhood_event_by_id(transaction, invalidation.invalidated_state_event_id)?;
    let replay_cooloff =
        load_neighborhood_event_by_id(transaction, invalidation.cooloff_state_event_id)?;
    if replay_invalidated.is_some() || replay_cooloff.is_some() {
        return replay_invalidation(
            invalidation,
            &identity_hash,
            replay_invalidated.as_ref(),
            replay_cooloff.as_ref(),
        );
    }
    if !neighborhood_key_exists(transaction, project_uuid, &invalidation.key)? {
        return Ok(ActiveNeighborhoodMutationAck::AuthorityChanged);
    }
    let configured_cooloff = configured_cooloff_seconds(
        transaction,
        project_uuid,
        invalidation.key.active_experiment_id,
    )?;
    if configured_cooloff != invalidation.cooloff_duration_seconds {
        return Ok(ActiveNeighborhoodMutationAck::Conflict);
    }
    if !invalidation_causes_exist(transaction, invalidation)? {
        return Ok(ActiveNeighborhoodMutationAck::Conflict);
    }
    let latest = latest_neighborhood_event(
        transaction,
        invalidation.key.active_experiment_id,
        &identity_hash,
    )?;
    let Some(latest) = latest else {
        return Ok(ActiveNeighborhoodMutationAck::Conflict);
    };
    if latest.state == ActiveNeighborhoodState::Cooloff {
        return Ok(ActiveNeighborhoodMutationAck::AlreadyCurrent(
            snapshot_from_event(&latest, false, false)?,
        ));
    }
    if !matches!(
        latest.state,
        ActiveNeighborhoodState::Promoted | ActiveNeighborhoodState::Restored
    ) {
        return Err(corrupt());
    }
    insert_neighborhood_event(
        transaction,
        invalidation.invalidated_state_event_id,
        &invalidation.key,
        &identity_hash,
        Some(latest.event_id),
        ActiveNeighborhoodState::Invalidated,
        None,
        invalidation.cause_active_dispatch_id,
        invalidation.cause_outcome_id,
        &invalidation.stable_reason,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    let duration_millis = i64::from(invalidation.cooloff_duration_seconds)
        .checked_mul(1_000)
        .ok_or_else(invariant)?;
    let cooloff_until_unix_ms = observed_at_unix_ms
        .checked_add(duration_millis)
        .ok_or_else(invariant)?;
    insert_neighborhood_event(
        transaction,
        invalidation.cooloff_state_event_id,
        &invalidation.key,
        &identity_hash,
        Some(invalidation.invalidated_state_event_id),
        ActiveNeighborhoodState::Cooloff,
        Some(cooloff_until_unix_ms),
        invalidation.cause_active_dispatch_id,
        invalidation.cause_outcome_id,
        &invalidation.stable_reason,
        process_instance_id,
        observed_at_unix_ms,
    )?;
    let event = load_neighborhood_event_by_id(transaction, invalidation.cooloff_state_event_id)?
        .ok_or_else(corrupt)?;
    Ok(ActiveNeighborhoodMutationAck::Applied(snapshot_from_event(
        &event, false, false,
    )?))
}

fn observe_active_neighborhood_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    key: &ActiveNeighborhoodKey,
    observed_at_unix_ms: i64,
) -> Result<ActiveNeighborhoodObserveAck, LedgerError> {
    validate_neighborhood_key_shape(key)?;
    if !neighborhood_key_exists(transaction, project_uuid, key)? {
        return Ok(ActiveNeighborhoodObserveAck::Unavailable);
    }
    let identity_hash = key.identity_hash()?;
    let authorization = observe_active_authorization_in_transaction(
        transaction,
        project_uuid,
        process_instance_id,
        key.active_experiment_id,
        observed_at_unix_ms,
    )?;
    let authorization_current = match authorization {
        ActiveAuthorizationObserveAck::Current(snapshot) => {
            authorization_supports_neighborhood(transaction, key, snapshot)?
        }
        _ => false,
    };
    let Some(event) =
        latest_neighborhood_event(transaction, key.active_experiment_id, &identity_hash)?
    else {
        return Ok(ActiveNeighborhoodObserveAck::Absent {
            neighborhood_identity_hash: identity_hash,
            promotion_authorized: authorization_current,
        });
    };
    let restorable = event.state == ActiveNeighborhoodState::Cooloff
        && event
            .cooloff_until_unix_ms
            .is_some_and(|until| observed_at_unix_ms >= until)
        && authorization_current;
    let authorizing = matches!(
        event.state,
        ActiveNeighborhoodState::Promoted | ActiveNeighborhoodState::Restored
    ) && authorization_current;
    Ok(ActiveNeighborhoodObserveAck::Current(snapshot_from_event(
        &event,
        authorizing,
        restorable,
    )?))
}

fn validate_promotion(promotion: &ActiveNeighborhoodPromotion) -> Result<(), LedgerError> {
    validate_neighborhood_key_shape(&promotion.key)?;
    if !is_uuid_v7(promotion.active_neighborhood_state_event_id)
        || !valid_stable_reason(&promotion.stable_reason)
    {
        return Err(invariant());
    }
    Ok(())
}

fn authorization_supports_neighborhood(
    connection: &Connection,
    key: &ActiveNeighborhoodKey,
    snapshot: ActiveAuthorizationSnapshot,
) -> Result<bool, LedgerError> {
    if snapshot.authorization_state_event_id != key.active_authorization_state_event_id {
        return Ok(false);
    }
    if snapshot.authorizing {
        return Ok(true);
    }
    if snapshot.state != ActiveAuthorizationState::Collecting {
        return Ok(false);
    }
    let event_control_generation = connection
        .query_row(
            "SELECT control_generation FROM active_authorization_state_events
             WHERE active_authorization_state_event_id = ?1
               AND active_experiment_id = ?2",
            params![
                key.active_authorization_state_event_id.to_string(),
                key.active_experiment_id.to_string(),
            ],
            |row| row.get::<_, i64>(0),
        )
        .optional()
        .map_err(database_error)?
        .and_then(|value| u64::try_from(value).ok());
    Ok(event_control_generation == Some(snapshot.control_generation))
}

fn validate_invalidation(invalidation: &ActiveNeighborhoodInvalidation) -> Result<(), LedgerError> {
    validate_neighborhood_key_shape(&invalidation.key)?;
    if !is_uuid_v7(invalidation.invalidated_state_event_id)
        || !is_uuid_v7(invalidation.cooloff_state_event_id)
        || invalidation.invalidated_state_event_id == invalidation.cooloff_state_event_id
        || invalidation
            .cause_active_dispatch_id
            .is_some_and(|value| !is_uuid_v7(value))
        || invalidation
            .cause_outcome_id
            .is_some_and(|value| !is_uuid_v7(value))
        || (invalidation.cause_active_dispatch_id.is_none()
            && invalidation.cause_outcome_id.is_none())
        || !valid_stable_reason(&invalidation.stable_reason)
        || invalidation.cooloff_duration_seconds == 0
    {
        return Err(invariant());
    }
    Ok(())
}

fn validate_neighborhood_key_shape(key: &ActiveNeighborhoodKey) -> Result<(), LedgerError> {
    if !is_uuid_v7(key.active_experiment_id)
        || !is_hash(&key.canonical_query_hash)
        || !is_hash(&key.sorted_neighbor_hash)
        || !is_hash(&key.config_generation_id)
        || !is_uuid_v7(key.learning_generation_id)
        || !is_uuid_v7(key.cohort_generation_id)
        || !is_uuid_v7(key.active_authorization_state_event_id)
    {
        return Err(invariant());
    }
    Ok(())
}

fn neighborhood_key_exists(
    connection: &Connection,
    project_uuid: Uuid,
    key: &ActiveNeighborhoodKey,
) -> Result<bool, LedgerError> {
    let Some(policy) = load_experiment_policy(connection, project_uuid, key.active_experiment_id)?
    else {
        return Ok(false);
    };
    if policy.create.config_generation_id != key.config_generation_id
        || policy.create.learning_generation_id != key.learning_generation_id
        || policy.create.cohort_generation_id != key.cohort_generation_id
    {
        return Ok(false);
    }
    let authorization_matches: bool = connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM active_authorization_state_events
                WHERE active_authorization_state_event_id = ?1
                  AND active_experiment_id = ?2
             )",
            params![
                key.active_authorization_state_event_id.to_string(),
                key.active_experiment_id.to_string(),
            ],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    if !authorization_matches {
        return Ok(false);
    }
    let query = connection
        .query_row(
            "SELECT canonical_query_json, canonical_payload_hash
             FROM canonical_routing_queries WHERE canonical_query_hash = ?1",
            [&key.canonical_query_hash],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    let Some((canonical_query_json, payload_hash)) = query else {
        return Ok(false);
    };
    let value = serde_json::from_str::<Json>(&canonical_query_json).map_err(|_| corrupt())?;
    Ok(payload_hash == key.canonical_query_hash
        && canonical_sha256(&value).map_err(|_| corrupt())? == key.canonical_query_hash)
}

fn configured_cooloff_seconds(
    connection: &Connection,
    project_uuid: Uuid,
    active_experiment_id: Uuid,
) -> Result<u32, LedgerError> {
    let row = connection
        .query_row(
            "SELECT experiment.pool_id, config.canonical_config_json
             FROM active_experiments AS experiment
             JOIN config_generations AS config
               ON config.project_uuid = experiment.project_uuid
              AND config.config_generation_id = experiment.config_generation_id
             WHERE experiment.project_uuid = ?1 AND experiment.active_experiment_id = ?2",
            params![project_uuid.to_string(), active_experiment_id.to_string()],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(corrupt)?;
    let config = serde_json::from_str::<Json>(&row.1).map_err(|_| corrupt())?;
    let pools = config
        .get("pools")
        .and_then(Json::as_array)
        .ok_or_else(corrupt)?;
    let mut matches = pools
        .iter()
        .filter(|pool| pool.get("id").and_then(Json::as_str) == Some(row.0.as_str()));
    let pool = matches.next().ok_or_else(corrupt)?;
    if matches.next().is_some() {
        return Err(corrupt());
    }
    pool.pointer("/outcome/protected_policy/relearning_cooloff_seconds")
        .and_then(Json::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(corrupt)
}

fn invalidation_causes_exist(
    connection: &Connection,
    invalidation: &ActiveNeighborhoodInvalidation,
) -> Result<bool, LedgerError> {
    if let Some(dispatch_id) = invalidation.cause_active_dispatch_id {
        let exists: bool = connection
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM active_dispatches
                    WHERE active_dispatch_id = ?1 AND active_experiment_id = ?2
                 )",
                params![
                    dispatch_id.to_string(),
                    invalidation.key.active_experiment_id.to_string(),
                ],
                |row| row.get(0),
            )
            .map_err(database_error)?;
        if !exists {
            return Ok(false);
        }
    }
    if let Some(outcome_id) = invalidation.cause_outcome_id {
        let exists: bool = connection
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM outcomes
                    WHERE outcome_id = ?1 AND active_experiment_id = ?2
                 )",
                params![
                    outcome_id.to_string(),
                    invalidation.key.active_experiment_id.to_string(),
                ],
                |row| row.get(0),
            )
            .map_err(database_error)?;
        if !exists {
            return Ok(false);
        }
    }
    Ok(true)
}

#[derive(Debug, Clone)]
struct VerifiedNeighborhoodEvent {
    event_id: Uuid,
    active_experiment_id: Uuid,
    neighborhood_identity_hash: String,
    canonical_query_hash: String,
    sorted_neighbor_hash: String,
    config_generation_id: String,
    learning_generation_id: Uuid,
    cohort_generation_id: Uuid,
    active_authorization_state_event_id: Uuid,
    predecessor_id: Option<Uuid>,
    state: ActiveNeighborhoodState,
    cooloff_until_unix_ms: Option<i64>,
    cause_active_dispatch_id: Option<Uuid>,
    cause_outcome_id: Option<Uuid>,
    stable_reason: String,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
}

fn latest_neighborhood_event(
    connection: &Connection,
    active_experiment_id: Uuid,
    identity_hash: &str,
) -> Result<Option<VerifiedNeighborhoodEvent>, LedgerError> {
    let event_id = connection
        .query_row(
            "SELECT active_neighborhood_state_event_id
             FROM active_neighborhood_state_events
             WHERE active_experiment_id = ?1 AND neighborhood_identity_hash = ?2
             ORDER BY event_seq DESC LIMIT 1",
            params![active_experiment_id.to_string(), identity_hash],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    event_id
        .map(|value| parse_uuid_v7(&value))
        .transpose()?
        .map(|event_id| load_neighborhood_event_by_id(connection, event_id))
        .transpose()
        .map(Option::flatten)
}

fn load_neighborhood_event_by_id(
    connection: &Connection,
    event_id: Uuid,
) -> Result<Option<VerifiedNeighborhoodEvent>, LedgerError> {
    let row = connection
        .query_row(
            "SELECT active_experiment_id, neighborhood_identity_hash,
                    canonical_query_hash, sorted_neighbor_hash,
                    config_generation_id, learning_generation_id,
                    cohort_generation_id, active_authorization_state_event_id,
                    predecessor_neighborhood_state_event_id, state,
                    cooloff_until_unix_ms, cause_active_dispatch_id,
                    cause_outcome_id, stable_reason, process_instance_id,
                    created_at_unix_ms, canonical_payload_hash
             FROM active_neighborhood_state_events
             WHERE active_neighborhood_state_event_id = ?1",
            [event_id.to_string()],
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
                    row.get::<_, Option<String>>(8)?,
                    row.get::<_, String>(9)?,
                    row.get::<_, Option<i64>>(10)?,
                    row.get::<_, Option<String>>(11)?,
                    row.get::<_, Option<String>>(12)?,
                    row.get::<_, String>(13)?,
                    row.get::<_, String>(14)?,
                    row.get::<_, i64>(15)?,
                    row.get::<_, String>(16)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some(row) = row else {
        return Ok(None);
    };
    let event = VerifiedNeighborhoodEvent {
        event_id,
        active_experiment_id: parse_uuid_v7(&row.0)?,
        neighborhood_identity_hash: row.1,
        canonical_query_hash: row.2,
        sorted_neighbor_hash: row.3,
        config_generation_id: row.4,
        learning_generation_id: parse_uuid_v7(&row.5)?,
        cohort_generation_id: parse_uuid_v7(&row.6)?,
        active_authorization_state_event_id: parse_uuid_v7(&row.7)?,
        predecessor_id: row.8.as_deref().map(parse_uuid_v7).transpose()?,
        state: ActiveNeighborhoodState::parse(&row.9)?,
        cooloff_until_unix_ms: row.10,
        cause_active_dispatch_id: row.11.as_deref().map(parse_uuid_v7).transpose()?,
        cause_outcome_id: row.12.as_deref().map(parse_uuid_v7).transpose()?,
        stable_reason: row.13,
        process_instance_id: parse_uuid_v7(&row.14)?,
        created_at_unix_ms: row.15,
    };
    if !is_hash(&event.neighborhood_identity_hash)
        || !is_hash(&event.canonical_query_hash)
        || !is_hash(&event.sorted_neighbor_hash)
        || !is_hash(&event.config_generation_id)
        || !valid_stable_reason(&event.stable_reason)
        || event.created_at_unix_ms < 0
        || (event.state == ActiveNeighborhoodState::Cooloff)
            != event.cooloff_until_unix_ms.is_some()
        || event
            .cooloff_until_unix_ms
            .is_some_and(|until| until <= event.created_at_unix_ms)
        || (matches!(
            event.state,
            ActiveNeighborhoodState::Promoted | ActiveNeighborhoodState::Restored
        ) && (event.cause_active_dispatch_id.is_some() || event.cause_outcome_id.is_some()))
    {
        return Err(corrupt());
    }
    let expected_hash = neighborhood_event_payload_hash(&event)?;
    if expected_hash != row.16 {
        return Err(corrupt());
    }
    Ok(Some(event))
}

#[allow(clippy::too_many_arguments)]
fn insert_neighborhood_event(
    connection: &Connection,
    event_id: Uuid,
    key: &ActiveNeighborhoodKey,
    identity_hash: &str,
    predecessor_id: Option<Uuid>,
    state: ActiveNeighborhoodState,
    cooloff_until_unix_ms: Option<i64>,
    cause_active_dispatch_id: Option<Uuid>,
    cause_outcome_id: Option<Uuid>,
    stable_reason: &str,
    process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let event = VerifiedNeighborhoodEvent {
        event_id,
        active_experiment_id: key.active_experiment_id,
        neighborhood_identity_hash: identity_hash.to_string(),
        canonical_query_hash: key.canonical_query_hash.clone(),
        sorted_neighbor_hash: key.sorted_neighbor_hash.clone(),
        config_generation_id: key.config_generation_id.clone(),
        learning_generation_id: key.learning_generation_id,
        cohort_generation_id: key.cohort_generation_id,
        active_authorization_state_event_id: key.active_authorization_state_event_id,
        predecessor_id,
        state,
        cooloff_until_unix_ms,
        cause_active_dispatch_id,
        cause_outcome_id,
        stable_reason: stable_reason.to_string(),
        process_instance_id,
        created_at_unix_ms,
    };
    let payload_hash = neighborhood_event_payload_hash(&event)?;
    execute_one(
        connection,
        "INSERT INTO active_neighborhood_state_events (
            active_neighborhood_state_event_id, active_experiment_id,
            neighborhood_identity_hash, canonical_query_hash, sorted_neighbor_hash,
            config_generation_id, learning_generation_id, cohort_generation_id,
            active_authorization_state_event_id,
            predecessor_neighborhood_state_event_id, state,
            cooloff_until_unix_ms, cause_active_dispatch_id, cause_outcome_id,
            stable_reason, process_instance_id, created_at_unix_ms,
            canonical_payload_hash
         ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9,
            ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18
         )",
        params![
            event.event_id.to_string(),
            event.active_experiment_id.to_string(),
            event.neighborhood_identity_hash,
            event.canonical_query_hash,
            event.sorted_neighbor_hash,
            event.config_generation_id,
            event.learning_generation_id.to_string(),
            event.cohort_generation_id.to_string(),
            event.active_authorization_state_event_id.to_string(),
            event.predecessor_id.map(|value| value.to_string()),
            event.state.as_str(),
            event.cooloff_until_unix_ms,
            event
                .cause_active_dispatch_id
                .map(|value| value.to_string()),
            event.cause_outcome_id.map(|value| value.to_string()),
            event.stable_reason,
            event.process_instance_id.to_string(),
            event.created_at_unix_ms,
            payload_hash,
        ],
    )
}

fn neighborhood_event_payload_hash(
    event: &VerifiedNeighborhoodEvent,
) -> Result<String, LedgerError> {
    hash_json(json!({
        "shape": "active_neighborhood_state_v1",
        "event_id": event.event_id,
        "active_experiment_id": event.active_experiment_id,
        "neighborhood_identity_hash": event.neighborhood_identity_hash,
        "canonical_query_hash": event.canonical_query_hash,
        "sorted_neighbor_hash": event.sorted_neighbor_hash,
        "config_generation_id": event.config_generation_id,
        "learning_generation_id": event.learning_generation_id,
        "cohort_generation_id": event.cohort_generation_id,
        "active_authorization_state_event_id": event.active_authorization_state_event_id,
        "predecessor_id": event.predecessor_id,
        "state": event.state.as_str(),
        "cooloff_until_unix_ms": event.cooloff_until_unix_ms.map(|value| value.to_string()),
        "cause_active_dispatch_id": event.cause_active_dispatch_id,
        "cause_outcome_id": event.cause_outcome_id,
        "stable_reason": event.stable_reason,
        "process_instance_id": event.process_instance_id,
        "created_at_unix_ms": event.created_at_unix_ms.to_string(),
    }))
}

fn event_matches_key(
    event: &VerifiedNeighborhoodEvent,
    key: &ActiveNeighborhoodKey,
    identity_hash: &str,
) -> bool {
    event.active_experiment_id == key.active_experiment_id
        && event.neighborhood_identity_hash == identity_hash
        && event.canonical_query_hash == key.canonical_query_hash
        && event.sorted_neighbor_hash == key.sorted_neighbor_hash
        && event.config_generation_id == key.config_generation_id
        && event.learning_generation_id == key.learning_generation_id
        && event.cohort_generation_id == key.cohort_generation_id
        && event.active_authorization_state_event_id == key.active_authorization_state_event_id
}

fn event_matches_promotion(
    event: &VerifiedNeighborhoodEvent,
    promotion: &ActiveNeighborhoodPromotion,
    identity_hash: &str,
) -> bool {
    event_matches_key(event, &promotion.key, identity_hash)
        && matches!(
            event.state,
            ActiveNeighborhoodState::Promoted | ActiveNeighborhoodState::Restored
        )
        && event.cooloff_until_unix_ms.is_none()
        && event.cause_active_dispatch_id.is_none()
        && event.cause_outcome_id.is_none()
        && event.stable_reason == promotion.stable_reason
}

fn replay_invalidation(
    invalidation: &ActiveNeighborhoodInvalidation,
    identity_hash: &str,
    invalidated: Option<&VerifiedNeighborhoodEvent>,
    cooloff: Option<&VerifiedNeighborhoodEvent>,
) -> Result<ActiveNeighborhoodMutationAck, LedgerError> {
    let (Some(invalidated), Some(cooloff)) = (invalidated, cooloff) else {
        return Ok(ActiveNeighborhoodMutationAck::Conflict);
    };
    let duration_millis = i64::from(invalidation.cooloff_duration_seconds)
        .checked_mul(1_000)
        .ok_or_else(invariant)?;
    let exact = event_matches_key(invalidated, &invalidation.key, identity_hash)
        && event_matches_key(cooloff, &invalidation.key, identity_hash)
        && invalidated.state == ActiveNeighborhoodState::Invalidated
        && invalidated.cooloff_until_unix_ms.is_none()
        && cooloff.state == ActiveNeighborhoodState::Cooloff
        && cooloff.predecessor_id == Some(invalidated.event_id)
        && cooloff.created_at_unix_ms == invalidated.created_at_unix_ms
        && cooloff.cooloff_until_unix_ms == cooloff.created_at_unix_ms.checked_add(duration_millis)
        && invalidated.cause_active_dispatch_id == invalidation.cause_active_dispatch_id
        && cooloff.cause_active_dispatch_id == invalidation.cause_active_dispatch_id
        && invalidated.cause_outcome_id == invalidation.cause_outcome_id
        && cooloff.cause_outcome_id == invalidation.cause_outcome_id
        && invalidated.stable_reason == invalidation.stable_reason
        && cooloff.stable_reason == invalidation.stable_reason;
    Ok(if exact {
        ActiveNeighborhoodMutationAck::AlreadyApplied(snapshot_from_event(cooloff, false, false)?)
    } else {
        ActiveNeighborhoodMutationAck::Conflict
    })
}

fn snapshot_from_event(
    event: &VerifiedNeighborhoodEvent,
    authorizing: bool,
    restorable: bool,
) -> Result<ActiveNeighborhoodSnapshot, LedgerError> {
    Ok(ActiveNeighborhoodSnapshot {
        active_experiment_id: event.active_experiment_id,
        active_neighborhood_state_event_id: event.event_id,
        neighborhood_identity_hash: event.neighborhood_identity_hash.clone(),
        state: event.state,
        cooloff_until_unix_ms: event
            .cooloff_until_unix_ms
            .map(u64::try_from)
            .transpose()
            .map_err(|_| corrupt())?,
        active_authorization_state_event_id: event.active_authorization_state_event_id,
        authorizing,
        restorable,
    })
}
