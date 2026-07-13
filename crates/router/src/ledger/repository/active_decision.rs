// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Atomic shape-2 decision, randomized assignment, and dispatch admission.

use std::sync::Arc;

use rusqlite::{Connection, OptionalExtension, Transaction, TransactionBehavior, params};
use serde_json::{Value as Json, json};
use uuid::{Uuid, Variant};

use super::active::{
    ActiveAdmissionAck, ActiveAdmissionReceipt, ActiveAssignmentArm, ActiveRootAdmission,
    admit_active_root_in_transaction,
};
use super::decision::{
    DecisionAuditAck, active_decision_evidence_matches, insert_active_decision_audit_in_transaction,
};
use super::{LedgerRepository, TransactionStartGuard};
use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::decision_audit::{ACTIVE_DECISION_SHAPE_VERSION_V2, DecisionAuditV1};
use crate::ledger::model::{LedgerError, LedgerErrorClass};

const ACTIVE_GATE_AUDIT_BYTES_MAX: usize = 1024 * 1024;
const ACTIVE_FALLBACK_REASON_BYTES_MAX: usize = 128;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActivePlannedRouteV2 {
    Candidate,
    AnchorControl,
    AnchorHoldout,
    AnchorForced,
    AnchorPaused,
    AnchorFallback,
}

impl ActivePlannedRouteV2 {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Candidate => "candidate",
            Self::AnchorControl => "anchor_control",
            Self::AnchorHoldout => "anchor_holdout",
            Self::AnchorForced => "anchor_forced",
            Self::AnchorPaused => "anchor_paused",
            Self::AnchorFallback => "anchor_fallback",
        }
    }

    const fn expected_arm(self) -> ActiveAssignmentArm {
        match self {
            Self::Candidate => ActiveAssignmentArm::CandidateTreatment,
            Self::AnchorControl => ActiveAssignmentArm::AnchorControl,
            Self::AnchorHoldout => ActiveAssignmentArm::AnchorHoldout,
            Self::AnchorForced | Self::AnchorPaused | Self::AnchorFallback => {
                ActiveAssignmentArm::NonLearning
            }
        }
    }

    fn parse(value: &str) -> Result<Self, LedgerError> {
        match value {
            "candidate" => Ok(Self::Candidate),
            "anchor_control" => Ok(Self::AnchorControl),
            "anchor_holdout" => Ok(Self::AnchorHoldout),
            "anchor_forced" => Ok(Self::AnchorForced),
            "anchor_paused" => Ok(Self::AnchorPaused),
            "anchor_fallback" => Ok(Self::AnchorFallback),
            _ => Err(corrupt()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveDecisionFactsV2 {
    pub(crate) active_neighborhood_state_event_id: Uuid,
    pub(crate) active_outcome_look_id: Option<Uuid>,
    pub(crate) planned_route: ActivePlannedRouteV2,
    pub(crate) control_generation: u64,
    pub(crate) promotion_lower_bound_bits: u64,
    pub(crate) retention_lower_bound_bits: u64,
    pub(crate) configured_holdout_probability_bits: u64,
    pub(crate) configured_canary_probability_bits: u64,
    pub(crate) actual_outcome_noninferiority_lower_bits: Option<u64>,
    pub(crate) actual_outcome_noninferiority_upper_bits: Option<u64>,
    pub(crate) anchor_shadow_lower_bound_bits: Option<u64>,
    pub(crate) fallback_reason: Option<String>,
    pub(crate) fresh_gate_audit_json: String,
    pub(crate) fresh_gate_audit_hash: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct VerifiedActiveDecisionFactsV2 {
    pub(crate) decision_id: Uuid,
    pub(crate) active_experiment_id: Option<Uuid>,
    pub(crate) active_assignment_id: Option<Uuid>,
    pub(crate) planned_route: ActivePlannedRouteV2,
    pub(crate) assignment_arm: ActiveAssignmentArm,
    pub(crate) control_generation: u64,
    pub(crate) promotion_lower_bound: f64,
    pub(crate) retention_lower_bound: f64,
    pub(crate) configured_holdout_probability: f64,
    pub(crate) configured_canary_probability: f64,
    pub(crate) effective_arm_probability: f64,
    pub(crate) conditional_selection_probability: f64,
    pub(crate) propensity: f64,
    pub(crate) fallback_reason: Option<String>,
}

#[derive(Clone)]
pub(crate) struct ActiveDecisionAdmissionV2 {
    pub(crate) audit: Arc<DecisionAuditV1>,
    pub(crate) facts: ActiveDecisionFactsV2,
    pub(crate) root: ActiveRootAdmission,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ActiveDecisionAdmissionAckV2 {
    Applied(ActiveAdmissionReceipt),
    AlreadyApplied(ActiveAdmissionReceipt),
    RootAlreadyAssigned,
    AuthorityChanged,
    OpenWindowCapacity,
    TrancheFull,
    ExperimentCapReached,
    Conflict,
    TransactionNotStarted,
}

impl LedgerRepository {
    pub(crate) fn admit_active_decision(
        &mut self,
        admission: &ActiveDecisionAdmissionV2,
    ) -> Result<ActiveDecisionAdmissionAckV2, LedgerError> {
        self.admit_active_decision_with_start_check(admission, || Some(()))
    }

    pub(crate) fn admit_active_decision_with_start_check<G: TransactionStartGuard>(
        &mut self,
        admission: &ActiveDecisionAdmissionV2,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<ActiveDecisionAdmissionAckV2, LedgerError> {
        validate_command(admission)?;
        let database_path = self.database_path.clone();
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(ActiveDecisionAdmissionAckV2::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_error) if !start_guard.permits_transaction() => {
                return Ok(ActiveDecisionAdmissionAckV2::TransactionNotStarted);
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
            return Ok(ActiveDecisionAdmissionAckV2::TransactionNotStarted);
        }
        drop(start_guard);
        let acknowledgement = admit_in_transaction(
            &transaction,
            self.project_uuid,
            self.process_instance_id,
            admission,
            now_unix_ms()?,
        )?;
        super::enforce_sidecar_permissions(&database_path).map_err(super::map_fs_error)?;
        transaction.commit().map_err(|error| {
            super::map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
        })?;
        Ok(acknowledgement)
    }
}

pub(crate) fn admit_in_transaction(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    admission: &ActiveDecisionAdmissionV2,
    admitted_at_unix_ms: i64,
) -> Result<ActiveDecisionAdmissionAckV2, LedgerError> {
    validate_command(admission)?;
    if let Some(replay) = load_exact_replay(transaction, project_uuid, admission)? {
        return Ok(replay);
    }
    if !fresh_authority_matches(
        transaction,
        project_uuid,
        process_instance_id,
        admission,
        admitted_at_unix_ms,
    )? {
        return Ok(ActiveDecisionAdmissionAckV2::AuthorityChanged);
    }

    transaction
        .execute_batch("SAVEPOINT active_decision_admission")
        .map_err(database_error)?;
    let result = apply_graph(
        transaction,
        project_uuid,
        process_instance_id,
        admission,
        admitted_at_unix_ms,
    );
    match result {
        Ok(ack @ ActiveDecisionAdmissionAckV2::Applied(_)) => {
            transaction
                .execute_batch("RELEASE active_decision_admission")
                .map_err(database_error)?;
            Ok(ack)
        }
        Ok(ack) => {
            rollback_savepoint(transaction)?;
            Ok(ack)
        }
        Err(error) => {
            let _ = rollback_savepoint(transaction);
            Err(error)
        }
    }
}

fn apply_graph(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    admission: &ActiveDecisionAdmissionV2,
    admitted_at_unix_ms: i64,
) -> Result<ActiveDecisionAdmissionAckV2, LedgerError> {
    match insert_active_decision_audit_in_transaction(transaction, &admission.audit)? {
        DecisionAuditAck::Applied => {}
        DecisionAuditAck::AlreadyApplied => return Ok(ActiveDecisionAdmissionAckV2::Conflict),
        DecisionAuditAck::Conflict => return Ok(ActiveDecisionAdmissionAckV2::Conflict),
        _ => return Err(corrupt()),
    }
    let root_ack = admit_active_root_in_transaction(
        transaction,
        project_uuid,
        process_instance_id,
        &admission.root,
        admitted_at_unix_ms,
    )?;
    let receipt = match root_ack {
        ActiveAdmissionAck::Applied(receipt) => receipt,
        acknowledgement => return Ok(map_root_ack(acknowledgement)),
    };
    insert_facts(transaction, admission)?;
    Ok(ActiveDecisionAdmissionAckV2::Applied(receipt))
}

fn rollback_savepoint(transaction: &Transaction<'_>) -> Result<(), LedgerError> {
    transaction
        .execute_batch("ROLLBACK TO active_decision_admission; RELEASE active_decision_admission;")
        .map_err(database_error)
}

fn map_root_ack(ack: ActiveAdmissionAck) -> ActiveDecisionAdmissionAckV2 {
    match ack {
        ActiveAdmissionAck::Applied(receipt) => ActiveDecisionAdmissionAckV2::Applied(receipt),
        ActiveAdmissionAck::AlreadyApplied(receipt) => {
            ActiveDecisionAdmissionAckV2::AlreadyApplied(receipt)
        }
        ActiveAdmissionAck::RootAlreadyAssigned => {
            ActiveDecisionAdmissionAckV2::RootAlreadyAssigned
        }
        ActiveAdmissionAck::AuthorityChanged => ActiveDecisionAdmissionAckV2::AuthorityChanged,
        ActiveAdmissionAck::OpenWindowCapacity => ActiveDecisionAdmissionAckV2::OpenWindowCapacity,
        ActiveAdmissionAck::TrancheFull => ActiveDecisionAdmissionAckV2::TrancheFull,
        ActiveAdmissionAck::ExperimentCapReached => {
            ActiveDecisionAdmissionAckV2::ExperimentCapReached
        }
        ActiveAdmissionAck::TransactionNotStarted => {
            ActiveDecisionAdmissionAckV2::TransactionNotStarted
        }
    }
}

fn validate_command(admission: &ActiveDecisionAdmissionV2) -> Result<(), LedgerError> {
    admission.audit.validate_frozen().map_err(|_| invariant())?;
    let parent = &admission.audit.parent;
    let facts = &admission.facts;
    let root = &admission.root;
    if parent.decision_shape_version != ACTIVE_DECISION_SHAPE_VERSION_V2
        || parent.decision_id != root.decision_id
        || parent.active_experiment_id != Some(root.active_experiment_id)
        || parent.active_authorization_state_event_id.is_none()
        || parent.cohort_generation_id != Some(root.cohort_generation_id)
        || parent.candidate_id.as_deref() != Some(root.candidate_id.as_str())
        || parent.root_key.as_deref() != Some(root.root_key.as_str())
        || parent.pool_id != root.pool_id
        || parent.config_generation_id != root.config_generation_id
        || parent.learning_generation_id != root.learning_generation_id
        || facts.planned_route.expected_arm() != root.arm
        || facts.control_generation != root.control_generation
        || !root
            .arm
            .matches_decision_reason(parent.final_reason.as_str())
        || facts.configured_holdout_probability_bits != root.configured_holdout_probability_bits
        || facts.configured_canary_probability_bits != root.configured_canary_probability_bits
        || !is_uuid_v7(facts.active_neighborhood_state_event_id)
        || facts
            .active_outcome_look_id
            .is_some_and(|value| !is_uuid_v7(value))
        || facts.fallback_reason.as_ref().is_some_and(|reason| {
            reason.is_empty()
                || reason.len() > ACTIVE_FALLBACK_REASON_BYTES_MAX
                || reason.chars().any(char::is_control)
        })
        || facts.fresh_gate_audit_json.len() > ACTIVE_GATE_AUDIT_BYTES_MAX
        || !is_hash(&facts.fresh_gate_audit_hash)
    {
        return Err(invariant());
    }
    validate_probability(facts.promotion_lower_bound_bits, true)?;
    let retention = validate_probability(facts.retention_lower_bound_bits, true)?;
    let promotion = f64::from_bits(facts.promotion_lower_bound_bits);
    if retention > promotion {
        return Err(invariant());
    }
    for bits in [
        facts.configured_holdout_probability_bits,
        facts.configured_canary_probability_bits,
    ] {
        validate_probability(bits, false)?;
    }
    for bits in [
        facts.actual_outcome_noninferiority_lower_bits,
        facts.actual_outcome_noninferiority_upper_bits,
        facts.anchor_shadow_lower_bound_bits,
    ]
    .into_iter()
    .flatten()
    {
        validate_probability(bits, true)?;
    }
    if matches!(
        (
            facts.actual_outcome_noninferiority_lower_bits,
            facts.actual_outcome_noninferiority_upper_bits,
        ),
        (Some(lower), Some(upper)) if f64::from_bits(lower) > f64::from_bits(upper)
    ) {
        return Err(invariant());
    }
    let value =
        serde_json::from_str::<Json>(&facts.fresh_gate_audit_json).map_err(|_| invariant())?;
    let canonical = canonical_json(&value).map_err(|_| invariant())?;
    if canonical != facts.fresh_gate_audit_json
        || canonical_sha256(&value).map_err(|_| invariant())? != facts.fresh_gate_audit_hash
    {
        return Err(invariant());
    }
    Ok(())
}

fn validate_probability(bits: u64, allow_zero: bool) -> Result<f64, LedgerError> {
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
struct StoredActiveDecisionFactsV2 {
    decision_id: String,
    decision_shape_version: i64,
    active_experiment_id: Option<String>,
    active_authorization_state_event_id: Option<String>,
    active_neighborhood_state_event_id: Option<String>,
    active_root_window_id: Option<String>,
    active_assignment_id: Option<String>,
    active_dispatch_id: Option<String>,
    active_outcome_look_id: Option<String>,
    planned_route: String,
    assignment_arm: String,
    control_generation: i64,
    promotion_lower_bound: f64,
    promotion_lower_bound_bits: i64,
    retention_lower_bound: f64,
    retention_lower_bound_bits: i64,
    configured_holdout_probability: f64,
    configured_holdout_probability_bits: i64,
    configured_canary_probability: f64,
    configured_canary_probability_bits: i64,
    effective_arm_probability: f64,
    effective_arm_probability_bits: i64,
    conditional_selection_probability: f64,
    conditional_selection_probability_bits: i64,
    propensity: f64,
    propensity_bits: i64,
    actual_outcome_noninferiority_lower: Option<f64>,
    actual_outcome_noninferiority_lower_bits: Option<i64>,
    actual_outcome_noninferiority_upper: Option<f64>,
    actual_outcome_noninferiority_upper_bits: Option<i64>,
    anchor_shadow_lower_bound: Option<f64>,
    anchor_shadow_lower_bound_bits: Option<i64>,
    fallback_reason: Option<String>,
    fresh_gate_audit_json: String,
    fresh_gate_audit_hash: String,
    canonical_payload_hash: String,
}

pub(crate) fn load_verified_active_decision_facts(
    connection: &Connection,
    decision_id: Uuid,
) -> Result<Option<VerifiedActiveDecisionFactsV2>, LedgerError> {
    if !is_uuid_v7(decision_id) {
        return Err(invariant());
    }
    let stored = connection
        .query_row(
            "SELECT decision_id, decision_shape_version, active_experiment_id,
                    active_authorization_state_event_id, active_neighborhood_state_event_id,
                    active_root_window_id, active_assignment_id, active_dispatch_id,
                    active_outcome_look_id, planned_route, assignment_arm, control_generation,
                    promotion_lower_bound, promotion_lower_bound_bits,
                    retention_lower_bound, retention_lower_bound_bits,
                    configured_holdout_probability, configured_holdout_probability_bits,
                    configured_canary_probability, configured_canary_probability_bits,
                    effective_arm_probability, effective_arm_probability_bits,
                    conditional_selection_probability, conditional_selection_probability_bits,
                    propensity, propensity_bits, actual_outcome_noninferiority_lower,
                    actual_outcome_noninferiority_lower_bits,
                    actual_outcome_noninferiority_upper,
                    actual_outcome_noninferiority_upper_bits,
                    anchor_shadow_lower_bound, anchor_shadow_lower_bound_bits,
                    fallback_reason, fresh_gate_audit_json, fresh_gate_audit_hash,
                    canonical_payload_hash
             FROM active_decision_facts WHERE decision_id = ?1",
            [decision_id.to_string()],
            |row| {
                Ok(StoredActiveDecisionFactsV2 {
                    decision_id: row.get(0)?,
                    decision_shape_version: row.get(1)?,
                    active_experiment_id: row.get(2)?,
                    active_authorization_state_event_id: row.get(3)?,
                    active_neighborhood_state_event_id: row.get(4)?,
                    active_root_window_id: row.get(5)?,
                    active_assignment_id: row.get(6)?,
                    active_dispatch_id: row.get(7)?,
                    active_outcome_look_id: row.get(8)?,
                    planned_route: row.get(9)?,
                    assignment_arm: row.get(10)?,
                    control_generation: row.get(11)?,
                    promotion_lower_bound: row.get(12)?,
                    promotion_lower_bound_bits: row.get(13)?,
                    retention_lower_bound: row.get(14)?,
                    retention_lower_bound_bits: row.get(15)?,
                    configured_holdout_probability: row.get(16)?,
                    configured_holdout_probability_bits: row.get(17)?,
                    configured_canary_probability: row.get(18)?,
                    configured_canary_probability_bits: row.get(19)?,
                    effective_arm_probability: row.get(20)?,
                    effective_arm_probability_bits: row.get(21)?,
                    conditional_selection_probability: row.get(22)?,
                    conditional_selection_probability_bits: row.get(23)?,
                    propensity: row.get(24)?,
                    propensity_bits: row.get(25)?,
                    actual_outcome_noninferiority_lower: row.get(26)?,
                    actual_outcome_noninferiority_lower_bits: row.get(27)?,
                    actual_outcome_noninferiority_upper: row.get(28)?,
                    actual_outcome_noninferiority_upper_bits: row.get(29)?,
                    anchor_shadow_lower_bound: row.get(30)?,
                    anchor_shadow_lower_bound_bits: row.get(31)?,
                    fallback_reason: row.get(32)?,
                    fresh_gate_audit_json: row.get(33)?,
                    fresh_gate_audit_hash: row.get(34)?,
                    canonical_payload_hash: row.get(35)?,
                })
            },
        )
        .optional()
        .map_err(database_error)?;
    stored.map(verify_stored_active_decision_facts).transpose()
}

fn verify_stored_active_decision_facts(
    stored: StoredActiveDecisionFactsV2,
) -> Result<VerifiedActiveDecisionFactsV2, LedgerError> {
    let decision_id = parse_stored_uuid(&stored.decision_id)?;
    let active_experiment_id = parse_stored_optional_uuid(stored.active_experiment_id.as_deref())?;
    let active_authorization_state_event_id =
        parse_stored_optional_uuid(stored.active_authorization_state_event_id.as_deref())?;
    let active_neighborhood_state_event_id =
        parse_stored_optional_uuid(stored.active_neighborhood_state_event_id.as_deref())?;
    let active_root_window_id =
        parse_stored_optional_uuid(stored.active_root_window_id.as_deref())?;
    let active_assignment_id = parse_stored_optional_uuid(stored.active_assignment_id.as_deref())?;
    let active_dispatch_id = parse_stored_optional_uuid(stored.active_dispatch_id.as_deref())?;
    let active_outcome_look_id =
        parse_stored_optional_uuid(stored.active_outcome_look_id.as_deref())?;
    let planned_route = ActivePlannedRouteV2::parse(&stored.planned_route)?;
    let assignment_arm = match stored.assignment_arm.as_str() {
        "candidate_treatment" => ActiveAssignmentArm::CandidateTreatment,
        "anchor_control" => ActiveAssignmentArm::AnchorControl,
        "anchor_holdout" => ActiveAssignmentArm::AnchorHoldout,
        "non_learning" => ActiveAssignmentArm::NonLearning,
        _ => return Err(corrupt()),
    };
    let control_generation = u64::try_from(stored.control_generation).map_err(|_| corrupt())?;

    let promotion_bits = stored.promotion_lower_bound_bits as u64;
    let retention_bits = stored.retention_lower_bound_bits as u64;
    let holdout_bits = stored.configured_holdout_probability_bits as u64;
    let canary_bits = stored.configured_canary_probability_bits as u64;
    let effective_bits = stored.effective_arm_probability_bits as u64;
    let conditional_bits = stored.conditional_selection_probability_bits as u64;
    let propensity_bits = stored.propensity_bits as u64;
    let actual_lower_bits = stored
        .actual_outcome_noninferiority_lower_bits
        .map(|value| value as u64);
    let actual_upper_bits = stored
        .actual_outcome_noninferiority_upper_bits
        .map(|value| value as u64);
    let anchor_lower_bits = stored
        .anchor_shadow_lower_bound_bits
        .map(|value| value as u64);

    let float_pairs = [
        (Some(stored.promotion_lower_bound), Some(promotion_bits)),
        (Some(stored.retention_lower_bound), Some(retention_bits)),
        (
            Some(stored.configured_holdout_probability),
            Some(holdout_bits),
        ),
        (
            Some(stored.configured_canary_probability),
            Some(canary_bits),
        ),
        (Some(stored.effective_arm_probability), Some(effective_bits)),
        (
            Some(stored.conditional_selection_probability),
            Some(conditional_bits),
        ),
        (Some(stored.propensity), Some(propensity_bits)),
        (
            stored.actual_outcome_noninferiority_lower,
            actual_lower_bits,
        ),
        (
            stored.actual_outcome_noninferiority_upper,
            actual_upper_bits,
        ),
        (stored.anchor_shadow_lower_bound, anchor_lower_bits),
    ];
    if stored.decision_shape_version != i64::from(ACTIVE_DECISION_SHAPE_VERSION_V2)
        || float_pairs
            .into_iter()
            .any(|(value, bits)| value.map(f64::to_bits) != bits)
        || planned_route.expected_arm() != assignment_arm
        || (planned_route == ActivePlannedRouteV2::Candidate) != active_dispatch_id.is_some()
        || stored.fallback_reason.as_ref().is_some_and(|reason| {
            reason.is_empty()
                || reason.len() > ACTIVE_FALLBACK_REASON_BYTES_MAX
                || reason.chars().any(char::is_control)
        })
        || !is_hash(&stored.fresh_gate_audit_hash)
        || !is_hash(&stored.canonical_payload_hash)
    {
        return Err(corrupt());
    }

    let promotion = validate_probability(promotion_bits, true).map_err(|_| corrupt())?;
    let retention = validate_probability(retention_bits, true).map_err(|_| corrupt())?;
    let holdout = validate_probability(holdout_bits, true).map_err(|_| corrupt())?;
    let canary = validate_probability(canary_bits, true).map_err(|_| corrupt())?;
    let effective = validate_probability(effective_bits, false).map_err(|_| corrupt())?;
    let conditional = validate_probability(conditional_bits, false).map_err(|_| corrupt())?;
    let propensity = validate_probability(propensity_bits, false).map_err(|_| corrupt())?;
    for bits in [actual_lower_bits, actual_upper_bits, anchor_lower_bits]
        .into_iter()
        .flatten()
    {
        validate_probability(bits, true).map_err(|_| corrupt())?;
    }
    if retention > promotion
        || holdout + canary > 1.0
        || (effective * conditional).to_bits() != propensity_bits
        || (assignment_arm == ActiveAssignmentArm::NonLearning
            && (effective != 1.0 || conditional != 1.0 || propensity != 1.0))
        || matches!((actual_lower_bits, actual_upper_bits), (Some(lower), Some(upper)) if f64::from_bits(lower) > f64::from_bits(upper))
    {
        return Err(corrupt());
    }

    let audit_value =
        serde_json::from_str::<Json>(&stored.fresh_gate_audit_json).map_err(|_| corrupt())?;
    if canonical_json(&audit_value).map_err(|_| corrupt())? != stored.fresh_gate_audit_json
        || canonical_sha256(&audit_value).map_err(|_| corrupt())? != stored.fresh_gate_audit_hash
    {
        return Err(corrupt());
    }
    let expected_hash = canonical_sha256(&json!({
        "schema": "nemo.relay.router.active-decision-facts@2",
        "decision_id": decision_id,
        "decision_shape_version": ACTIVE_DECISION_SHAPE_VERSION_V2,
        "active_experiment_id": active_experiment_id,
        "active_authorization_state_event_id": active_authorization_state_event_id,
        "active_neighborhood_state_event_id": active_neighborhood_state_event_id,
        "active_root_window_id": active_root_window_id,
        "active_assignment_id": active_assignment_id,
        "active_dispatch_id": active_dispatch_id,
        "active_outcome_look_id": active_outcome_look_id,
        "planned_route": planned_route.as_str(),
        "assignment_arm": &stored.assignment_arm,
        "control_generation": control_generation.to_string(),
        "promotion_lower_bound_bits": format!("{promotion_bits:016x}"),
        "retention_lower_bound_bits": format!("{retention_bits:016x}"),
        "configured_holdout_probability_bits": format!("{holdout_bits:016x}"),
        "configured_canary_probability_bits": format!("{canary_bits:016x}"),
        "effective_arm_probability_bits": format!("{effective_bits:016x}"),
        "conditional_selection_probability_bits": format!("{conditional_bits:016x}"),
        "propensity_bits": format!("{propensity_bits:016x}"),
        "actual_outcome_noninferiority_lower_bits": optional_bits(actual_lower_bits),
        "actual_outcome_noninferiority_upper_bits": optional_bits(actual_upper_bits),
        "anchor_shadow_lower_bound_bits": optional_bits(anchor_lower_bits),
        "fallback_reason": &stored.fallback_reason,
        "fresh_gate_audit_json": &stored.fresh_gate_audit_json,
        "fresh_gate_audit_hash": &stored.fresh_gate_audit_hash,
    }))
    .map_err(|_| corrupt())?;
    if expected_hash != stored.canonical_payload_hash {
        return Err(corrupt());
    }

    Ok(VerifiedActiveDecisionFactsV2 {
        decision_id,
        active_experiment_id,
        active_assignment_id,
        planned_route,
        assignment_arm,
        control_generation,
        promotion_lower_bound: promotion,
        retention_lower_bound: retention,
        configured_holdout_probability: holdout,
        configured_canary_probability: canary,
        effective_arm_probability: effective,
        conditional_selection_probability: conditional,
        propensity,
        fallback_reason: stored.fallback_reason,
    })
}

fn parse_stored_uuid(value: &str) -> Result<Uuid, LedgerError> {
    let value = Uuid::parse_str(value).map_err(|_| corrupt())?;
    if is_uuid_v7(value) {
        Ok(value)
    } else {
        Err(corrupt())
    }
}

fn parse_stored_optional_uuid(value: Option<&str>) -> Result<Option<Uuid>, LedgerError> {
    value.map(parse_stored_uuid).transpose()
}

fn fresh_authority_matches(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    admission: &ActiveDecisionAdmissionV2,
    admitted_at_unix_ms: i64,
) -> Result<bool, LedgerError> {
    let parent = &admission.audit.parent;
    let facts = &admission.facts;
    if parent.project_uuid != project_uuid
        || parent.process_instance_id != process_instance_id
        || !experiment_facts_match(transaction, admission)?
        || !active_decision_evidence_matches(transaction, &admission.audit)?
        || !effective_control_allows(transaction, project_uuid, &parent.pool_id, facts)?
    {
        return Ok(false);
    }
    let authorization = transaction
        .query_row(
            "SELECT active_authorization_state_event_id, state, valid_until_unix_ms,
                    active_outcome_look_id
             FROM active_authorization_state_events
             WHERE active_experiment_id = ?1 ORDER BY event_seq DESC LIMIT 1",
            [admission.root.active_experiment_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                    row.get::<_, Option<String>>(3)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((authorization_id, state, valid_until, outcome_look_id)) = authorization else {
        return Ok(false);
    };
    if parent
        .active_authorization_state_event_id
        .is_none_or(|value| value.to_string() != authorization_id)
        || facts.active_outcome_look_id.map(|value| value.to_string()) != outcome_look_id
        || !matches!(state.as_str(), "collecting" | "passed")
        || (state == "passed" && valid_until.is_none_or(|until| admitted_at_unix_ms >= until))
        || (state == "collecting" && valid_until.is_some())
    {
        return Ok(false);
    }
    if !outcome_interval_matches(transaction, admission)? {
        return Ok(false);
    }
    neighborhood_matches(transaction, admission)
}

fn experiment_facts_match(
    connection: &Connection,
    admission: &ActiveDecisionAdmissionV2,
) -> Result<bool, LedgerError> {
    let parent = &admission.audit.parent;
    let facts = &admission.facts;
    let winner = admission
        .audit
        .summaries
        .iter()
        .find(|summary| parent.candidate_id.as_deref() == Some(summary.candidate_id.as_str()))
        .ok_or_else(corrupt)?;
    let stored = connection
        .query_row(
            "SELECT partition_hash, policy_version_id, vector_space_id,
                    control_generation, promotion_lower_bound_bits,
                    retention_lower_bound_bits, holdout_probability_bits,
                    active_canary_fraction_bits
             FROM active_experiments WHERE active_experiment_id = ?1",
            [admission.root.active_experiment_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, i64>(5)?,
                    row.get::<_, i64>(6)?,
                    row.get::<_, i64>(7)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    Ok(matches!(
        stored,
        Some((partition, policy, vector, control, promotion, retention, holdout, canary))
            if partition == winner.partition_hash
                && policy == parent.policy_version_id
                && vector == parent.vector_space_id
                && u64::try_from(control).ok() == Some(facts.control_generation)
                && promotion as u64 == facts.promotion_lower_bound_bits
                && retention as u64 == facts.retention_lower_bound_bits
                && holdout as u64 == facts.configured_holdout_probability_bits
                && canary as u64 == facts.configured_canary_probability_bits
    ))
}

fn outcome_interval_matches(
    connection: &Connection,
    admission: &ActiveDecisionAdmissionV2,
) -> Result<bool, LedgerError> {
    let facts = &admission.facts;
    let Some(look_id) = facts.active_outcome_look_id else {
        return Ok(facts.actual_outcome_noninferiority_lower_bits.is_none()
            && facts.actual_outcome_noninferiority_upper_bits.is_none());
    };
    let stored = connection
        .query_row(
            "SELECT audit.noninferiority_lower_bits, audit.noninferiority_upper_bits
             FROM active_outcome_looks AS look
             JOIN active_outcome_look_audits AS audit
               ON audit.active_outcome_look_id = look.active_outcome_look_id
             WHERE look.active_outcome_look_id = ?1
               AND look.active_experiment_id = ?2",
            params![
                look_id.to_string(),
                admission.root.active_experiment_id.to_string(),
            ],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
        )
        .optional()
        .map_err(database_error)?;
    Ok(stored.is_some_and(|(lower, upper)| {
        facts.actual_outcome_noninferiority_lower_bits == Some(lower as u64)
            && facts.actual_outcome_noninferiority_upper_bits == Some(upper as u64)
    }))
}

fn effective_control_allows(
    connection: &Connection,
    project_uuid: Uuid,
    pool_id: &str,
    facts: &ActiveDecisionFactsV2,
) -> Result<bool, LedgerError> {
    let generation = connection
        .query_row(
            "SELECT max(control_generation) FROM controls WHERE project_uuid = ?1",
            [project_uuid.to_string()],
            |row| row.get::<_, Option<i64>>(0),
        )
        .map_err(database_error)?
        .and_then(|value| u64::try_from(value).ok());
    let all = latest_control(connection, project_uuid, "all", None)?;
    let pool = latest_control(connection, project_uuid, "pool", Some(pool_id))?;
    let paused = all.1 || pool.1;
    let forced = all.0 || pool.0;
    Ok(generation == Some(facts.control_generation) && !paused && !forced)
}

fn latest_control(
    connection: &Connection,
    project_uuid: Uuid,
    scope: &str,
    pool_id: Option<&str>,
) -> Result<(bool, bool), LedgerError> {
    connection
        .query_row(
            "SELECT force_anchor, paused FROM controls
             WHERE project_uuid = ?1 AND scope_kind = ?2
               AND ((?3 IS NULL AND pool_id IS NULL) OR pool_id = ?3)
             ORDER BY control_generation DESC LIMIT 1",
            params![project_uuid.to_string(), scope, pool_id],
            |row| Ok((row.get::<_, bool>(0)?, row.get::<_, bool>(1)?)),
        )
        .optional()
        .map_err(database_error)
        .map(|value| value.unwrap_or((false, false)))
}

fn neighborhood_matches(
    connection: &Connection,
    admission: &ActiveDecisionAdmissionV2,
) -> Result<bool, LedgerError> {
    let facts = &admission.facts;
    let parent = &admission.audit.parent;
    let row = connection
        .query_row(
            "SELECT neighborhood.active_experiment_id,
                    neighborhood.canonical_query_hash,
                    neighborhood.active_authorization_state_event_id,
                    neighborhood.state,
                    neighborhood.neighborhood_identity_hash,
                    neighborhood.sorted_neighbor_hash,
                    neighborhood.config_generation_id,
                    neighborhood.learning_generation_id,
                    neighborhood.cohort_generation_id,
                    (SELECT latest.active_neighborhood_state_event_id
                     FROM active_neighborhood_state_events AS latest
                     WHERE latest.active_experiment_id = neighborhood.active_experiment_id
                       AND latest.neighborhood_identity_hash = neighborhood.neighborhood_identity_hash
                     ORDER BY latest.event_seq DESC LIMIT 1)
             FROM active_neighborhood_state_events AS neighborhood
             WHERE neighborhood.active_neighborhood_state_event_id = ?1",
            [facts.active_neighborhood_state_event_id.to_string()],
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
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    Ok(matches!(
        row,
        Some((experiment, query, authorization, state, _identity, neighbor_hash,
              config, learning, cohort, latest))
            if experiment == admission.root.active_experiment_id.to_string()
                && query == parent.canonical_query_hash
                && parent.active_authorization_state_event_id
                    .is_some_and(|value| value.to_string() == authorization)
                && matches!(state.as_str(), "promoted" | "restored")
                && latest == facts.active_neighborhood_state_event_id.to_string()
                && config == parent.config_generation_id
                && learning == parent.learning_generation_id.to_string()
                && parent.cohort_generation_id
                    .is_some_and(|value| value.to_string() == cohort)
                && fresh_audit_neighbor_hash(&facts.fresh_gate_audit_json)
                    .is_some_and(|value| value == neighbor_hash)
    ))
}

fn fresh_audit_neighbor_hash(document: &str) -> Option<String> {
    serde_json::from_str::<Json>(document)
        .ok()?
        .get("winner")?
        .get("sorted_neighbor_hash")?
        .as_str()
        .map(str::to_string)
}

fn insert_facts(
    transaction: &Transaction<'_>,
    admission: &ActiveDecisionAdmissionV2,
) -> Result<(), LedgerError> {
    let facts = &admission.facts;
    let root = &admission.root;
    let parent = &admission.audit.parent;
    let dispatch_id = root
        .dispatch
        .as_ref()
        .map(|dispatch| dispatch.active_dispatch_id);
    let payload_hash = facts_payload_hash(admission)?;
    let inserted = transaction
        .execute(
            "INSERT INTO active_decision_facts (
                decision_id, decision_shape_version, active_experiment_id,
                active_authorization_state_event_id, active_neighborhood_state_event_id,
                active_root_window_id, active_assignment_id, active_dispatch_id,
                active_outcome_look_id, planned_route, assignment_arm, control_generation,
                promotion_lower_bound, promotion_lower_bound_bits,
                retention_lower_bound, retention_lower_bound_bits,
                configured_holdout_probability, configured_holdout_probability_bits,
                configured_canary_probability, configured_canary_probability_bits,
                effective_arm_probability, effective_arm_probability_bits,
                conditional_selection_probability, conditional_selection_probability_bits,
                propensity, propensity_bits,
                actual_outcome_noninferiority_lower,
                actual_outcome_noninferiority_lower_bits,
                actual_outcome_noninferiority_upper,
                actual_outcome_noninferiority_upper_bits,
                anchor_shadow_lower_bound, anchor_shadow_lower_bound_bits,
                fallback_reason, fresh_gate_audit_json, fresh_gate_audit_hash,
                canonical_payload_hash
             ) VALUES (
                ?1, 2, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11,
                ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21,
                ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31,
                ?32, ?33, ?34, ?35
             )",
            params![
                parent.decision_id.to_string(),
                root.active_experiment_id.to_string(),
                parent
                    .active_authorization_state_event_id
                    .map(|value| value.to_string()),
                facts.active_neighborhood_state_event_id.to_string(),
                root.active_root_window_id.to_string(),
                root.active_assignment_id.to_string(),
                dispatch_id.map(|value| value.to_string()),
                facts.active_outcome_look_id.map(|value| value.to_string()),
                facts.planned_route.as_str(),
                root.arm.as_str(),
                i64::try_from(facts.control_generation).map_err(|_| invariant())?,
                f64::from_bits(facts.promotion_lower_bound_bits),
                facts.promotion_lower_bound_bits as i64,
                f64::from_bits(facts.retention_lower_bound_bits),
                facts.retention_lower_bound_bits as i64,
                f64::from_bits(facts.configured_holdout_probability_bits),
                facts.configured_holdout_probability_bits as i64,
                f64::from_bits(facts.configured_canary_probability_bits),
                facts.configured_canary_probability_bits as i64,
                f64::from_bits(root.effective_arm_probability_bits),
                root.effective_arm_probability_bits as i64,
                f64::from_bits(root.conditional_selection_probability_bits),
                root.conditional_selection_probability_bits as i64,
                f64::from_bits(root.propensity_bits),
                root.propensity_bits as i64,
                optional_float(facts.actual_outcome_noninferiority_lower_bits),
                facts
                    .actual_outcome_noninferiority_lower_bits
                    .map(|value| value as i64),
                optional_float(facts.actual_outcome_noninferiority_upper_bits),
                facts
                    .actual_outcome_noninferiority_upper_bits
                    .map(|value| value as i64),
                optional_float(facts.anchor_shadow_lower_bound_bits),
                facts
                    .anchor_shadow_lower_bound_bits
                    .map(|value| value as i64),
                facts.fallback_reason,
                facts.fresh_gate_audit_json,
                facts.fresh_gate_audit_hash,
                payload_hash,
            ],
        )
        .map_err(database_error)?;
    if inserted != 1 {
        return Err(corrupt());
    }
    Ok(())
}

fn load_exact_replay(
    transaction: &Transaction<'_>,
    project_uuid: Uuid,
    admission: &ActiveDecisionAdmissionV2,
) -> Result<Option<ActiveDecisionAdmissionAckV2>, LedgerError> {
    let decision_exists: bool = transaction
        .query_row(
            "SELECT EXISTS(SELECT 1 FROM decisions WHERE decision_id = ?1)",
            [admission.audit.parent.decision_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    if !decision_exists {
        return Ok(None);
    }
    if insert_active_decision_audit_in_transaction(transaction, &admission.audit)?
        != DecisionAuditAck::AlreadyApplied
        || !stored_facts_match(transaction, admission)?
    {
        return Ok(Some(ActiveDecisionAdmissionAckV2::Conflict));
    }
    let root_ack = admit_active_root_in_transaction(
        transaction,
        project_uuid,
        admission.audit.parent.process_instance_id,
        &admission.root,
        admission.audit.parent.created_at_unix_ms,
    )?;
    Ok(Some(match root_ack {
        ActiveAdmissionAck::AlreadyApplied(receipt) => {
            ActiveDecisionAdmissionAckV2::AlreadyApplied(receipt)
        }
        _ => ActiveDecisionAdmissionAckV2::Conflict,
    }))
}

fn stored_facts_match(
    connection: &Connection,
    admission: &ActiveDecisionAdmissionV2,
) -> Result<bool, LedgerError> {
    let stored = connection
        .query_row(
            "SELECT canonical_payload_hash FROM active_decision_facts WHERE decision_id = ?1",
            [admission.audit.parent.decision_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?;
    Ok(stored.as_deref() == Some(facts_payload_hash(admission)?.as_str()))
}

fn facts_payload_hash(admission: &ActiveDecisionAdmissionV2) -> Result<String, LedgerError> {
    let parent = &admission.audit.parent;
    let facts = &admission.facts;
    let root = &admission.root;
    canonical_sha256(&json!({
        "schema": "nemo.relay.router.active-decision-facts@2",
        "decision_id": parent.decision_id,
        "decision_shape_version": ACTIVE_DECISION_SHAPE_VERSION_V2,
        "active_experiment_id": root.active_experiment_id,
        "active_authorization_state_event_id": parent.active_authorization_state_event_id,
        "active_neighborhood_state_event_id": facts.active_neighborhood_state_event_id,
        "active_root_window_id": root.active_root_window_id,
        "active_assignment_id": root.active_assignment_id,
        "active_dispatch_id": root.dispatch.as_ref().map(|value| value.active_dispatch_id),
        "active_outcome_look_id": facts.active_outcome_look_id,
        "planned_route": facts.planned_route.as_str(),
        "assignment_arm": root.arm.as_str(),
        "control_generation": facts.control_generation.to_string(),
        "promotion_lower_bound_bits": format!("{:016x}", facts.promotion_lower_bound_bits),
        "retention_lower_bound_bits": format!("{:016x}", facts.retention_lower_bound_bits),
        "configured_holdout_probability_bits": format!("{:016x}", facts.configured_holdout_probability_bits),
        "configured_canary_probability_bits": format!("{:016x}", facts.configured_canary_probability_bits),
        "effective_arm_probability_bits": format!("{:016x}", root.effective_arm_probability_bits),
        "conditional_selection_probability_bits": format!("{:016x}", root.conditional_selection_probability_bits),
        "propensity_bits": format!("{:016x}", root.propensity_bits),
        "actual_outcome_noninferiority_lower_bits": optional_bits(facts.actual_outcome_noninferiority_lower_bits),
        "actual_outcome_noninferiority_upper_bits": optional_bits(facts.actual_outcome_noninferiority_upper_bits),
        "anchor_shadow_lower_bound_bits": optional_bits(facts.anchor_shadow_lower_bound_bits),
        "fallback_reason": facts.fallback_reason,
        "fresh_gate_audit_json": facts.fresh_gate_audit_json,
        "fresh_gate_audit_hash": facts.fresh_gate_audit_hash,
    }))
    .map_err(|_| invariant())
}

fn optional_float(bits: Option<u64>) -> Option<f64> {
    bits.map(f64::from_bits)
}

fn optional_bits(bits: Option<u64>) -> Json {
    bits.map_or(Json::Null, |value| Json::String(format!("{value:016x}")))
}

fn is_uuid_v7(value: Uuid) -> bool {
    value.get_version_num() == 7 && value.get_variant() == Variant::RFC4122
}

fn is_hash(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn now_unix_ms() -> Result<i64, LedgerError> {
    Ok(chrono::Utc::now().timestamp_millis().max(0))
}

fn invariant() -> LedgerError {
    LedgerErrorClass::IdentityInvariant.into()
}

fn corrupt() -> LedgerError {
    LedgerErrorClass::CorruptDatabase.into()
}

fn database_error(_: rusqlite::Error) -> LedgerError {
    LedgerErrorClass::DatabaseOperationFailed.into()
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::time::{Duration, Instant};

    use nemo_relay_types::api::llm::LlmApiFamily;

    use super::*;
    use crate::active_planner::{
        ActiveCandidateExperimentAuthorityV2, ActiveGateResolverV2, evaluate_active_query_until,
    };
    use crate::candidate_set::build_candidate_set_v1;
    use crate::canonical_query::{CanonicalRoutingQueryArtifactV1, CanonicalRoutingQueryV1};
    use crate::confidence::ConfidencePolicyV1;
    use crate::decision_audit::{ActiveDecisionParentBindingV2, DecisionFinalReasonV1};
    use crate::ledger::cohort::{CohortAssignmentRequest, RandomizedCohort};
    use crate::ledger::read_pool::LedgerReadPool;
    use crate::ledger::repository::active::{ActiveAssignmentArm, ActiveDispatchAdmission};
    use crate::ledger::repository::active_learning::{
        ActiveExperimentCreate, ActiveExperimentCreateAck,
    };
    use crate::ledger::repository::ready_evaluated_active_runtime_fixture;
    use crate::ledger::repository::vector_registry::FrozenMappingKey;
    use crate::ledger::writer::LedgerWriterOwner;
    use crate::live_embedding::{LiveEmbeddingResult, PreparedLiveQueryV1};
    use crate::recommendation::{
        RecommendationDecisionIdentityV1, RecommendationPreflightV1, RecommendationQueryV1,
        prepare_recommendation_v1,
    };
    use crate::routing_partition::{
        RoutingPartitionArtifactV1, RoutingPartitionBaseV1, RoutingPartitionV1,
        build_routing_partition_base_v1,
    };
    use crate::sqlite_vector_store::SqliteVecStore;
    use crate::trajectory::{
        CANDIDATE_FACT_SCHEMA_V1, PersistedCandidateCapabilitiesV1, PersistedCandidateFactV1,
    };

    #[tokio::test]
    async fn real_evidence_atomic_admission_and_exact_retry_commit_one_complete_graph() {
        let (_temporary, config, mut activated, vector_space_id, query_vector, evidence_id, _) =
            ready_evaluated_active_runtime_fixture();
        let identity = activated.identity.clone();
        let pool_identity = identity.pools.get("pool-a").unwrap();
        let candidate = &config.pools[0].candidates[0];
        let active_policy = config.pools[0]
            .learning
            .as_ref()
            .and_then(|learning| learning.complete_active_policy())
            .unwrap();
        let outcome: crate::config::OutcomeConfig = serde_json::from_value(Json::Object(
            config.pools[0].outcome.clone().into_iter().collect(),
        ))
        .unwrap();
        let (
            canonical_query_hash,
            canonical_query_json,
            _partition_id,
            partition_hash,
            canonical_partition_json,
            evaluation_created_at,
        ) = activated
            .repository
            .connection
            .query_row(
                "SELECT link.canonical_query_hash, query.canonical_query_json,
                        link.partition_id, partition.partition_hash,
                        partition.canonical_partition_json, evaluation.created_at_unix_ms
                 FROM evidence_vector_links AS link
                 JOIN canonical_routing_queries AS query
                   ON query.canonical_query_hash = link.canonical_query_hash
                 JOIN routing_partitions AS partition
                   ON partition.partition_id = link.partition_id
                 JOIN evaluations AS evaluation
                   ON evaluation.evaluation_id = link.evaluation_id
                 WHERE link.evidence_vector_link_id = ?1",
                [evidence_id.to_string()],
                |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, i64>(2)?,
                        row.get::<_, String>(3)?,
                        row.get::<_, String>(4)?,
                        row.get::<_, i64>(5)?,
                    ))
                },
            )
            .unwrap();
        let outcome_policy_hash = activated
            .repository
            .connection
            .query_row(
                "SELECT outcome_policy_hash FROM outcome_policy_versions
                 WHERE project_uuid = ?1 AND pool_id = 'pool-a'",
                [identity.project_uuid.to_string()],
                |row| row.get::<_, String>(0),
            )
            .unwrap();
        let create = ActiveExperimentCreate::from_config(
            &config,
            &identity,
            "pool-a",
            candidate.id.clone(),
            partition_hash.clone(),
            outcome_policy_hash.clone(),
            0,
        )
        .unwrap();
        let experiment = match activated
            .repository
            .create_active_experiment(&create)
            .unwrap()
        {
            ActiveExperimentCreateAck::Applied(receipt) => receipt,
            acknowledgement => panic!("unexpected experiment ack: {acknowledgement:?}"),
        };

        let (assignment, owner_relation_hash) = loop {
            let raw_root = Uuid::now_v7();
            let assignment = activated
                .cohort_assignment
                .assign(CohortAssignmentRequest::new(
                    raw_root,
                    &identity.config_generation_id,
                    "pool-a",
                    &candidate.id,
                    active_policy.holdout_probability,
                    active_policy.active_canary_fraction,
                ))
                .unwrap();
            if assignment.cohort() == RandomizedCohort::ActiveCanary {
                let owner_relation_hash = activated
                    .cohort_assignment
                    .protect_pinned_owner(assignment.root_key(), Uuid::now_v7())
                    .to_hex();
                break (assignment, owner_relation_hash);
            }
        };

        let partition: RoutingPartitionV1 =
            serde_json::from_str(&canonical_partition_json).unwrap();
        let partition_artifact = RoutingPartitionArtifactV1 {
            partition: partition.clone(),
            canonical_json: canonical_partition_json,
            partition_hash,
        };
        let base = build_routing_partition_base_v1(&RoutingPartitionBaseV1 {
            tenant_policy_hash: partition.tenant_policy_hash.clone(),
            agent_policy_hash: partition.agent_policy_hash.clone(),
            policy_version_id: partition.policy_version_id.clone(),
            learning_generation_id: partition.learning_generation_id,
            api_family: partition.api_family,
            transport_identity: partition.transport_identity.clone(),
            anchor_model: partition.anchor_model.clone(),
            anchor_revision: partition.anchor_revision.clone(),
            evaluator_version: partition.evaluator_version.clone(),
            vector_space_id: partition.vector_space_id.clone(),
        })
        .unwrap();
        let candidate_fact = PersistedCandidateFactV1 {
            schema: CANDIDATE_FACT_SCHEMA_V1.to_string(),
            candidate_id: candidate.id.clone(),
            model: candidate.model.clone(),
            model_revision: candidate.model_revision.clone(),
            cost_rank: candidate.cost_rank,
            capabilities: PersistedCandidateCapabilitiesV1 {
                tools: candidate.capabilities.tools,
                multimodal_input: candidate.capabilities.multimodal_input,
                structured_output: candidate.capabilities.structured_output,
                reasoning_controls: candidate.capabilities.reasoning_controls,
            },
            decoding_fingerprint: partition.decoding_fingerprint.clone(),
        };
        build_candidate_set_v1(std::slice::from_ref(&candidate_fact)).unwrap();
        let prepared_query = CanonicalRoutingQueryArtifactV1 {
            query: serde_json::from_str::<CanonicalRoutingQueryV1>(&canonical_query_json).unwrap(),
            canonical_bytes: canonical_query_json.as_bytes().to_vec(),
            canonical_query_hash: canonical_query_hash.clone(),
        };
        let mapping = FrozenMappingKey::new(
            identity.project_uuid,
            identity.config_generation_id.clone(),
            "pool-a",
            pool_identity.policy_version_id.clone(),
        )
        .unwrap();
        let live_query = PreparedLiveQueryV1::from_test_parts(
            mapping,
            vector_space_id.clone(),
            prepared_query,
            Duration::from_secs(5),
        );
        let query = RecommendationQueryV1::from_prepared(live_query).unwrap();
        let preflight = RecommendationPreflightV1::from_safe_facts(
            LlmApiFamily::OpenAIChatCompletions,
            partition.transport_identity.clone(),
            vec![candidate_fact],
        )
        .unwrap();
        let admitted_at = Instant::now();
        let deadline = admitted_at + Duration::from_secs(10);
        let prepared = prepare_recommendation_v1(
            RecommendationDecisionIdentityV1 {
                decision_id: Uuid::now_v7(),
                process_instance_id: identity.process_instance_id,
                primary_call_uuid: Uuid::now_v7(),
                as_of_unix_ms: evaluation_created_at + 1,
                created_at_unix_ms: evaluation_created_at + 1,
                admitted_at,
            },
            preflight,
            query,
            ConfidencePolicyV1::new(
                active_policy.recommend.top_k,
                active_policy.recommend.radius,
                active_policy.recommend.min_points,
                active_policy.recommend.min_independent_roots,
                active_policy.recommend.min_effective_samples,
                active_policy.recommend.min_coverage,
                f64::from(u32::try_from(outcome.anchor_shadow_half_life_seconds).unwrap()),
                active_policy.recommend.prior_success,
                active_policy.recommend.prior_failure,
                active_policy.recommend.familywise_credible_level,
                active_policy.recommend.promotion_lower_bound,
                config.pools[0].judge.judge_confidence_floor,
            )
            .unwrap(),
            base,
            vec![partition_artifact],
            deadline,
        )
        .unwrap();

        let readers = LedgerReadPool::open(Path::new(&config.database_path)).unwrap();
        let (owner, writer) = LedgerWriterOwner::start(activated.repository, 64).unwrap();
        let store = SqliteVecStore::new(writer.clone(), readers.clone());
        let resolver = ActiveGateResolverV2::new(
            writer.clone(),
            vec![ActiveCandidateExperimentAuthorityV2 {
                candidate_id: candidate.id.clone(),
                active_experiment_id: experiment.active_experiment_id,
            }],
            identity.config_generation_id.clone(),
            pool_identity.learning_generation_id,
            identity.cohort_generation_id,
            canonical_query_hash,
            active_policy.recommend.promotion_lower_bound,
            active_policy.retention_lower_bound,
            chrono::Utc::now().timestamp_millis().max(0),
            deadline,
        )
        .unwrap();
        let mut plan = evaluate_active_query_until(
            &store,
            prepared,
            LiveEmbeddingResult::Ready(query_vector),
            resolver,
        )
        .await
        .unwrap();
        assert_eq!(plan.winner_candidate_id(), Some(candidate.id.as_str()));
        let winner = plan.authorize_winner_neighborhood_until().await.unwrap();
        let audited = plan
            .into_audited_query(
                ActiveDecisionParentBindingV2 {
                    cohort_generation_id: identity.cohort_generation_id,
                    active_experiment_id: Some(winner.active_experiment_id),
                    active_authorization_state_event_id: Some(
                        winner.active_authorization_state_event_id,
                    ),
                    root_key: assignment.root_key().to_hex(),
                },
                DecisionFinalReasonV1::ActiveCandidate,
            )
            .unwrap();
        let holdout_threshold = u64::try_from(assignment.holdout_threshold_numerator())
            .unwrap()
            .to_be_bytes();
        let canary_threshold = u64::try_from(assignment.canary_threshold_numerator())
            .unwrap()
            .to_be_bytes();
        let propensity = assignment.propensity();
        let now = chrono::Utc::now().timestamp_millis().max(0);
        let command = ActiveDecisionAdmissionV2 {
            audit: Arc::clone(&audited.audit),
            facts: ActiveDecisionFactsV2 {
                active_neighborhood_state_event_id: winner.active_neighborhood_state_event_id,
                active_outcome_look_id: winner.active_outcome_look_id,
                planned_route: ActivePlannedRouteV2::Candidate,
                control_generation: 0,
                promotion_lower_bound_bits: active_policy.recommend.promotion_lower_bound.to_bits(),
                retention_lower_bound_bits: active_policy.retention_lower_bound.to_bits(),
                configured_holdout_probability_bits: assignment
                    .configured_holdout_probability_bits(),
                configured_canary_probability_bits: assignment
                    .configured_active_canary_fraction_bits(),
                actual_outcome_noninferiority_lower_bits: None,
                actual_outcome_noninferiority_upper_bits: None,
                anchor_shadow_lower_bound_bits: audited.anchor_shadow_lower_bound_bits,
                fallback_reason: None,
                fresh_gate_audit_json: audited.fresh_gate_audit_json,
                fresh_gate_audit_hash: audited.fresh_gate_audit_hash,
            },
            root: ActiveRootAdmission {
                active_root_window_id: Uuid::now_v7(),
                active_experiment_id: winner.active_experiment_id,
                active_assignment_id: Uuid::now_v7(),
                decision_id: audited.audit.parent.decision_id,
                dispatch: Some(ActiveDispatchAdmission {
                    active_dispatch_id: Uuid::now_v7(),
                    request_identity_hash: "ab".repeat(32),
                }),
                pool_id: "pool-a".into(),
                candidate_id: candidate.id.clone(),
                root_key: assignment.root_key().to_hex(),
                owner_relation_hash,
                config_generation_id: identity.config_generation_id.clone(),
                learning_generation_id: pool_identity.learning_generation_id,
                cohort_generation_id: identity.cohort_generation_id,
                outcome_policy_hash,
                opened_after_ingest_seq: 7,
                attribution_deadline_unix_ms: u64::try_from(now + 600_000).unwrap(),
                tranche_ordinal: 1,
                arm: ActiveAssignmentArm::CandidateTreatment,
                cohort_threshold_numerator: holdout_threshold,
                selection_threshold_numerator: canary_threshold,
                configured_holdout_probability_bits: assignment
                    .configured_holdout_probability_bits(),
                configured_canary_probability_bits: assignment
                    .configured_active_canary_fraction_bits(),
                effective_arm_probability_bits: propensity.effective_arm_probability_bits(),
                conditional_selection_probability_bits: propensity
                    .conditional_selection_probability_bits(),
                propensity_bits: propensity.propensity_bits(),
                control_generation: 0,
            },
        };

        let first_writer = writer.clone();
        let first_command = command.clone();
        let second_writer = writer.clone();
        let second_command = command.clone();
        let (first, second) = tokio::join!(
            first_writer.admit_active_decision_until(first_command, deadline),
            second_writer.admit_active_decision_until(second_command, deadline),
        );
        let acknowledgements = [first.unwrap(), second.unwrap()];
        assert_eq!(
            acknowledgements
                .iter()
                .filter(|acknowledgement| matches!(
                    acknowledgement,
                    ActiveDecisionAdmissionAckV2::Applied(ActiveAdmissionReceipt {
                        total_ordinal: 1,
                        ..
                    })
                ))
                .count(),
            1
        );
        assert_eq!(
            acknowledgements
                .iter()
                .filter(|acknowledgement| matches!(
                    acknowledgement,
                    ActiveDecisionAdmissionAckV2::AlreadyApplied(ActiveAdmissionReceipt {
                        total_ordinal: 1,
                        ..
                    })
                ))
                .count(),
            1
        );
        let decision_id = command.audit.parent.decision_id;
        let root_for_outcome = command.root.clone();
        let receipt = match writer
            .admit_active_decision_until(command, deadline)
            .await
            .unwrap()
        {
            ActiveDecisionAdmissionAckV2::AlreadyApplied(
                receipt @ ActiveAdmissionReceipt {
                    total_ordinal: 1, ..
                },
            ) => receipt,
            acknowledgement => panic!("unexpected replay acknowledgement: {acknowledgement:?}"),
        };
        assert!(matches!(
            writer
                .append_active_signals_until(
                    crate::ledger::repository::active::ActiveSignalBatch {
                        active_root_window_id: root_for_outcome.active_root_window_id,
                        signals: vec![crate::ledger::repository::active::tests::protected_signal(
                            1,
                            crate::ledger::repository::active::ActiveSignalDisposition::Success,
                            receipt.admitted_at_unix_ms + 1,
                            "{}".into(),
                        )],
                    },
                    deadline,
                )
                .await
                .unwrap(),
            crate::ledger::repository::active::ActiveSignalBatchAck::Applied { .. }
        ));
        let dispatch = root_for_outcome.dispatch.as_ref().unwrap();
        assert_eq!(
            writer
                .record_active_dispatch_terminal_until(
                    crate::ledger::repository::active::ActiveDispatchTerminal {
                        active_dispatch_terminal_event_id: Uuid::now_v7(),
                        active_dispatch_id: dispatch.active_dispatch_id,
                        active_assignment_id: root_for_outcome.active_assignment_id,
                        terminal_state: crate::ledger::repository::active::ActiveDispatchTerminalState::Completed,
                        stable_error_class: None,
                        provider_receipt_hash: None,
                        handed_off_at_unix_ms: Some(receipt.admitted_at_unix_ms),
                    },
                    deadline,
                )
                .await
                .unwrap(),
            crate::ledger::repository::active::ActiveDispatchTerminalAck::Applied
        );
        let terminal = crate::ledger::repository::active::tests::terminal_for(
            &root_for_outcome,
            receipt,
            crate::ledger::repository::active::ActiveRootClosure::OwnerEnd,
            true,
            Some(crate::ledger::repository::active::ActiveRepresentativeStatus::Completed),
        );
        assert_eq!(
            writer
                .terminalize_active_root_until(terminal, deadline)
                .await
                .unwrap(),
            crate::ledger::repository::active::ActiveRootTerminalAck::Applied
        );
        let inspection = rusqlite::Connection::open(&config.database_path).unwrap();
        let verified = load_verified_active_decision_facts(&inspection, decision_id)
            .unwrap()
            .unwrap();
        assert_eq!(verified.decision_id, decision_id);
        assert_eq!(verified.planned_route, ActivePlannedRouteV2::Candidate);
        assert_eq!(
            verified.assignment_arm,
            ActiveAssignmentArm::CandidateTreatment
        );
        assert_eq!(verified.propensity.to_bits(), propensity.propensity_bits());
        let exposure = crate::ledger::repository::inspection::load_decision_exposure(
            &inspection,
            &config,
            decision_id,
        )
        .unwrap()
        .unwrap();
        assert_eq!(exposure.decision_id, decision_id);
        assert_eq!(
            exposure.active.as_ref().unwrap().assignment_arm,
            "candidate_treatment"
        );
        assert_eq!(
            exposure.outcome.as_ref().unwrap().label.as_deref(),
            Some("success")
        );
        let overview = crate::ledger::repository::inspection::load_overview_snapshot(
            &inspection,
            &config,
            0,
            i64::MAX as u64,
        )
        .unwrap();
        assert_eq!(overview.decisions.total, 1);
        assert_eq!(overview.decisions.active, 1);
        assert_eq!(overview.decisions.candidate_served, 1);
        assert_eq!(overview.exposures.candidate_treatment, 1);
        assert_eq!(overview.outcomes.candidate_treatment.success, 1);
        for table in [
            "decisions",
            "active_decision_facts",
            "active_root_windows",
            "active_assignments",
            "active_dispatches",
        ] {
            let count = inspection
                .query_row(&format!("SELECT count(*) FROM {table}"), [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap();
            assert_eq!(count, 1, "{table}");
        }
        inspection
            .execute(
                "UPDATE active_decision_facts
                 SET canonical_payload_hash =
                   '0000000000000000000000000000000000000000000000000000000000000000'
                 WHERE decision_id = ?1",
                [decision_id.to_string()],
            )
            .unwrap();
        assert!(load_verified_active_decision_facts(&inspection, decision_id).is_err());
        assert!(
            crate::ledger::repository::inspection::load_decision_exposure(
                &inspection,
                &config,
                decision_id,
            )
            .is_err()
        );
        assert!(
            crate::ledger::repository::inspection::load_overview_snapshot(
                &inspection,
                &config,
                0,
                i64::MAX as u64,
            )
            .is_err()
        );
        owner.abort();
        readers.abort();
    }
}
