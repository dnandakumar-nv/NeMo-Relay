// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical Judge-attempt and quality-evaluation repository operations.

use std::collections::BTreeMap;

use rusqlite::{Connection, OptionalExtension, TransactionBehavior, named_params, params};
use serde_json::{Value as Json, json};
use uuid::Uuid;

use super::process::{append_integrity_health, originating_process_is_live};
use super::shadow::{
    VerifiedShadowJudgePolicy, canonical_shadow_attempt_is_unterminalized,
    verified_shadow_attempt_context, verified_shadow_judge_input_context,
    verified_shadow_judge_policy,
};
use super::{LedgerRepository, TransactionStartGuard, map_fs_error, map_sqlite_error};
use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::config::{
    JUDGE_MAX_RATIONALE_BYTES, JUDGE_OUTPUT_SCHEMA_SHA256_V1, JUDGE_PROMPT_TEMPLATE_SHA256_V1,
    JUDGE_RUBRIC_TEMPLATE_SHA256_V1, JudgeConfig,
};
use crate::fingerprint::sha256_hex;
use crate::judge::{
    DeterministicHardFailureV1, JudgeAttemptOutcomeV1, JudgeBinaryLabelV1, JudgeEvaluationSourceV1,
    JudgeEvaluationV1, JudgeHardFailureV1, JudgeLabelV1, PairwiseJudgeInputV1,
    PairwiseJudgeResultV1, validate_pairwise_judge_output,
};
use crate::ledger::fs::enforce_sidecar_permissions;
use crate::ledger::model::{LedgerError, LedgerErrorClass};
use crate::trajectory::RouterResponseProjectionV1;

/// Immutable Judge provider invocation and its initial state event.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct JudgeAttemptStart {
    pub(crate) judge_attempt_id: Uuid,
    pub(crate) shadow_attempt_id: Uuid,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) evaluator_version: String,
    pub(crate) judge_model: String,
    pub(crate) judge_model_revision: String,
    pub(crate) prompt_version: String,
    pub(crate) prompt_sha256: String,
    pub(crate) rubric_version: String,
    pub(crate) rubric_sha256: String,
    pub(crate) output_schema_version: u32,
    pub(crate) output_schema_sha256: String,
    pub(crate) judge_input_sha256: String,
    pub(crate) candidate_response_json: String,
    pub(crate) candidate_response_fingerprint: String,
    judge_config_sha256: String,
    pub(crate) attempt_ordinal: u8,
    pub(crate) judge_attempt_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    pub(crate) created_at_unix_ms: i64,
}

impl JudgeAttemptStart {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        judge_attempt_id: Uuid,
        shadow_attempt_id: Uuid,
        learning_generation_id: Uuid,
        evaluator_version: impl Into<String>,
        judge: &JudgeConfig,
        judge_input: &PairwiseJudgeInputV1,
        attempt_ordinal: u8,
        judge_attempt_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        for id in [
            judge_attempt_id,
            shadow_attempt_id,
            learning_generation_id,
            judge_attempt_state_event_id,
            conflict_health_event_id,
        ] {
            validate_uuid_v7(id)?;
        }
        if judge_attempt_id == judge_attempt_state_event_id
            || judge_attempt_id == conflict_health_event_id
            || judge_attempt_state_event_id == conflict_health_event_id
            || attempt_ordinal > 1
            || created_at_unix_ms < 0
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let evaluator_version = evaluator_version.into();
        validate_sha256(&evaluator_version)?;
        validate_bounded_text(&judge.model, 512)?;
        validate_bounded_text(&judge.model_revision, 128)?;
        validate_bounded_text(&judge.prompt_version, 128)?;
        validate_bounded_text(&judge.rubric_version, 128)?;
        if judge.output_schema_version != 1 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let judge_config_sha256 = judge
            .contract_sha256()
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let judge_input_sha256 = judge_input
            .canonical_sha256()
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let candidate_response = judge_input.candidate_response();
        let mut candidate_response_value = serde_json::to_value(candidate_response)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let candidate_response_json = canonical_json(&candidate_response_value)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let candidate_response_fingerprint =
            candidate_response.semantic_response_fingerprint.clone();
        validate_sha256(&candidate_response_fingerprint)?;
        candidate_response_value
            .as_object_mut()
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?
            .remove("semantic_response_fingerprint");
        if canonical_sha256(&candidate_response_value)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?
            != candidate_response_fingerprint
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self {
            judge_attempt_id,
            shadow_attempt_id,
            learning_generation_id,
            evaluator_version,
            judge_model: judge.model.clone(),
            judge_model_revision: judge.model_revision.clone(),
            prompt_version: judge.prompt_version.clone(),
            prompt_sha256: JUDGE_PROMPT_TEMPLATE_SHA256_V1.to_string(),
            rubric_version: judge.rubric_version.clone(),
            rubric_sha256: JUDGE_RUBRIC_TEMPLATE_SHA256_V1.to_string(),
            output_schema_version: judge.output_schema_version,
            output_schema_sha256: JUDGE_OUTPUT_SCHEMA_SHA256_V1.to_string(),
            judge_input_sha256,
            candidate_response_json,
            candidate_response_fingerprint,
            judge_config_sha256,
            attempt_ordinal,
            judge_attempt_state_event_id,
            conflict_health_event_id,
            created_at_unix_ms,
        })
    }
}

/// Terminal Judge invocation outcome. Safe JSON is canonicalized before enqueueing.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct JudgeAttemptTerminal {
    pub(crate) judge_attempt_id: Uuid,
    pub(crate) judge_attempt_state_event_id: Uuid,
    pub(crate) conflict_health_event_id: Uuid,
    state: JudgeTerminalState,
    judge_config_sha256: Option<String>,
    pub(crate) created_at_unix_ms: i64,
    pub(crate) evaluation: Option<EvaluationRecord>,
}

#[derive(Clone, PartialEq, Eq)]
enum JudgeTerminalState {
    Valid {
        safe_output_json: String,
        raw_output_sha256: String,
        raw_output_bytes: i64,
    },
    Invalid {
        safe_output_json: String,
        raw_output_sha256: String,
        raw_output_bytes: i64,
    },
    TransportFailure {
        stable_error_class: String,
    },
    CanceledShutdown,
}

/// Closed stable classes for provider-side Judge failures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JudgeTransportFailureClass {
    Transport,
    Authentication,
    RateLimited,
    Timeout,
    ProviderCanceled,
    UnreadableResponse,
    TruncatedResponse,
    AmbiguousDecode,
    EvidenceBoundExceeded,
    MiddlewareInterference,
}

impl JudgeTransportFailureClass {
    const fn as_str(self) -> &'static str {
        match self {
            Self::Transport => "router.provider.transport",
            Self::Authentication => "router.provider.authentication",
            Self::RateLimited => "router.provider.rate_limited",
            Self::Timeout => "router.provider.timeout",
            Self::ProviderCanceled => "router.provider.canceled",
            Self::UnreadableResponse => "router.provider.unreadable_response",
            Self::TruncatedResponse => "router.provider.truncated_response",
            Self::AmbiguousDecode => "router.provider.ambiguous_decode",
            Self::EvidenceBoundExceeded => "router.provider.evidence_bound_exceeded",
            Self::MiddlewareInterference => "router.provider.middleware_interference",
        }
    }

    fn from_str(value: &str) -> Option<Self> {
        match value {
            "router.provider.transport" => Some(Self::Transport),
            "router.provider.authentication" => Some(Self::Authentication),
            "router.provider.rate_limited" => Some(Self::RateLimited),
            "router.provider.timeout" => Some(Self::Timeout),
            "router.provider.canceled" => Some(Self::ProviderCanceled),
            "router.provider.unreadable_response" => Some(Self::UnreadableResponse),
            "router.provider.truncated_response" => Some(Self::TruncatedResponse),
            "router.provider.ambiguous_decode" => Some(Self::AmbiguousDecode),
            "router.provider.evidence_bound_exceeded" => Some(Self::EvidenceBoundExceeded),
            "router.provider.middleware_interference" => Some(Self::MiddlewareInterference),
            _ => None,
        }
    }
}

impl JudgeAttemptTerminal {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn valid(
        judge_attempt_id: Uuid,
        judge_attempt_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        shadow_attempt_id: Uuid,
        evaluator_version: impl Into<String>,
        raw_output: &str,
        judge: &JudgeConfig,
        evaluation_id: Uuid,
        evaluation_conflict_health_event_id: Uuid,
        is_partial: bool,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        let JudgeAttemptOutcomeV1::Valid(result) =
            validate_pairwise_judge_output(raw_output, judge.max_rationale_bytes)
        else {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        };
        let safe_output = serde_json::to_value(&result)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let evaluation = EvaluationRecord::from_judge_result(
            evaluation_id,
            shadow_attempt_id,
            evaluator_version,
            evaluation_conflict_health_event_id,
            &result,
            judge,
            is_partial,
            created_at_unix_ms,
        )?;
        Self::validated_output(
            judge_attempt_id,
            judge_attempt_state_event_id,
            conflict_health_event_id,
            true,
            &safe_output,
            judge,
            sha256_hex(raw_output.as_bytes()),
            u64::try_from(raw_output.len())
                .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?,
            Some(evaluation),
            created_at_unix_ms,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn invalid(
        judge_attempt_id: Uuid,
        judge_attempt_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        raw_output: &str,
        judge: &JudgeConfig,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        let JudgeAttemptOutcomeV1::Invalid(invalid) =
            validate_pairwise_judge_output(raw_output, judge.max_rationale_bytes)
        else {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        };
        if invalid.output().sha256() != sha256_hex(raw_output.as_bytes())
            || invalid.output().byte_count()
                != u64::try_from(raw_output.len())
                    .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let safe_output = serde_json::to_value(&invalid)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        Self::validated_output(
            judge_attempt_id,
            judge_attempt_state_event_id,
            conflict_health_event_id,
            false,
            &safe_output,
            judge,
            invalid.output().sha256().to_string(),
            invalid.output().byte_count(),
            None,
            created_at_unix_ms,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn validated_output(
        judge_attempt_id: Uuid,
        judge_attempt_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        valid: bool,
        safe_output: &Json,
        judge: &JudgeConfig,
        raw_output_sha256: String,
        raw_output_bytes: u64,
        evaluation: Option<EvaluationRecord>,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_terminal_ids(
            judge_attempt_id,
            judge_attempt_state_event_id,
            conflict_health_event_id,
            created_at_unix_ms,
        )?;
        validate_sha256(&raw_output_sha256)?;
        let raw_output_bytes = i64::try_from(raw_output_bytes)
            .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let safe_output_json = canonical_json(safe_output)
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let raw_output_bound = judge
            .max_rationale_bytes
            .checked_add(8 * 1024)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let evidence_bound = raw_output_bound
            .checked_mul(6)
            .and_then(|value| value.checked_add(8 * 1024))
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if judge.max_rationale_bytes == 0
            || judge.max_rationale_bytes > JUDGE_MAX_RATIONALE_BYTES
            || safe_output_json.len() > evidence_bound
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let state = if valid {
            if evaluation.is_none() {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
            JudgeTerminalState::Valid {
                safe_output_json,
                raw_output_sha256,
                raw_output_bytes,
            }
        } else {
            if evaluation.is_some() {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
            JudgeTerminalState::Invalid {
                safe_output_json,
                raw_output_sha256,
                raw_output_bytes,
            }
        };
        let judge_config_sha256 = judge
            .contract_sha256()
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        Ok(Self {
            judge_attempt_id,
            judge_attempt_state_event_id,
            conflict_health_event_id,
            state,
            judge_config_sha256: Some(judge_config_sha256),
            created_at_unix_ms,
            evaluation,
        })
    }

    pub(crate) fn transport_failure(
        judge_attempt_id: Uuid,
        judge_attempt_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        stable_error_class: JudgeTransportFailureClass,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_terminal_ids(
            judge_attempt_id,
            judge_attempt_state_event_id,
            conflict_health_event_id,
            created_at_unix_ms,
        )?;
        Ok(Self {
            judge_attempt_id,
            judge_attempt_state_event_id,
            conflict_health_event_id,
            state: JudgeTerminalState::TransportFailure {
                stable_error_class: stable_error_class.as_str().to_string(),
            },
            judge_config_sha256: None,
            created_at_unix_ms,
            evaluation: None,
        })
    }

    pub(crate) fn canceled_shutdown(
        judge_attempt_id: Uuid,
        judge_attempt_state_event_id: Uuid,
        conflict_health_event_id: Uuid,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        validate_terminal_ids(
            judge_attempt_id,
            judge_attempt_state_event_id,
            conflict_health_event_id,
            created_at_unix_ms,
        )?;
        Ok(Self {
            judge_attempt_id,
            judge_attempt_state_event_id,
            conflict_health_event_id,
            state: JudgeTerminalState::CanceledShutdown,
            judge_config_sha256: None,
            created_at_unix_ms,
            evaluation: None,
        })
    }
}

fn judge_attempt_key_exists(
    connection: &Connection,
    start: &JudgeAttemptStart,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM judge_attempts
                WHERE judge_attempt_id = ?1
                   OR (shadow_attempt_id = ?2 AND evaluator_version = ?3
                       AND attempt_ordinal = ?4)
             )",
            params![
                start.judge_attempt_id.to_string(),
                start.shadow_attempt_id.to_string(),
                start.evaluator_version,
                start.attempt_ordinal,
            ],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn judge_attempt_matches(
    connection: &Connection,
    process_instance_id: Uuid,
    start: &JudgeAttemptStart,
    expected_hash: &str,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM judge_attempts
                WHERE judge_attempt_id = ?1 AND shadow_attempt_id = ?2
                  AND process_instance_id = ?3 AND learning_generation_id = ?4
                  AND evaluator_version = ?5 AND judge_model = ?6
                  AND judge_model_revision = ?7 AND prompt_version = ?8
                  AND prompt_sha256 = ?9 AND rubric_version = ?10
                  AND rubric_sha256 = ?11 AND output_schema_version = ?12
                  AND output_schema_sha256 = ?13 AND judge_input_sha256 = ?14
                  AND candidate_response_json = ?15
                  AND candidate_response_fingerprint = ?16
                  AND attempt_ordinal = ?17 AND created_at_unix_ms = ?18
                  AND canonical_payload_hash = ?19
             )",
            params![
                start.judge_attempt_id.to_string(),
                start.shadow_attempt_id.to_string(),
                process_instance_id.to_string(),
                start.learning_generation_id.to_string(),
                start.evaluator_version,
                start.judge_model,
                start.judge_model_revision,
                start.prompt_version,
                start.prompt_sha256,
                start.rubric_version,
                start.rubric_sha256,
                start.output_schema_version,
                start.output_schema_sha256,
                start.judge_input_sha256,
                start.candidate_response_json,
                start.candidate_response_fingerprint,
                start.attempt_ordinal,
                start.created_at_unix_ms,
                expected_hash,
            ],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn judge_start_state_key_exists(
    connection: &Connection,
    start: &JudgeAttemptStart,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM judge_attempt_state_events
                WHERE judge_attempt_state_event_id = ?1
                   OR (judge_attempt_id = ?2 AND state = 'started')
             )",
            params![
                start.judge_attempt_state_event_id.to_string(),
                start.judge_attempt_id.to_string(),
            ],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn judge_start_state_matches(
    connection: &Connection,
    process_instance_id: Uuid,
    start: &JudgeAttemptStart,
    expected_hash: &str,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM judge_attempt_state_events
                WHERE judge_attempt_state_event_id = ?1
                  AND judge_attempt_id = ?2 AND process_instance_id = ?3
                  AND dead_process_instance_id IS NULL AND state = 'started'
                  AND parse_result IS NULL AND safe_output_json IS NULL
                  AND raw_output_sha256 IS NULL AND raw_output_bytes IS NULL
                  AND stable_error_class IS NULL AND created_at_unix_ms = ?4
                  AND canonical_payload_hash = ?5
             )",
            params![
                start.judge_attempt_state_event_id.to_string(),
                start.judge_attempt_id.to_string(),
                process_instance_id.to_string(),
                start.created_at_unix_ms,
                expected_hash,
            ],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn judge_terminal_state_key_exists(
    connection: &Connection,
    terminal: &JudgeAttemptTerminal,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM judge_attempt_state_events
                WHERE judge_attempt_state_event_id = ?1
                   OR (judge_attempt_id = ?2 AND state <> 'started')
             )",
            params![
                terminal.judge_attempt_state_event_id.to_string(),
                terminal.judge_attempt_id.to_string(),
            ],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn judge_terminal_state_matches(
    connection: &Connection,
    process_instance_id: Uuid,
    terminal: &JudgeAttemptTerminal,
    fields: &PreparedTerminalFields<'_>,
    expected_hash: &str,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM judge_attempt_state_events
                WHERE judge_attempt_state_event_id = :event_id
                  AND judge_attempt_id = :attempt_id
                  AND process_instance_id = :process_id
                  AND dead_process_instance_id IS NULL
                  AND state = :state AND parse_result IS :parse_result
                  AND safe_output_json IS :safe_output
                  AND raw_output_sha256 IS :raw_hash
                  AND raw_output_bytes IS :raw_bytes
                  AND stable_error_class IS :error_class
                  AND created_at_unix_ms = :created_at
                  AND canonical_payload_hash = :payload_hash
             )",
            named_params! {
                ":event_id": terminal.judge_attempt_state_event_id.to_string(),
                ":attempt_id": terminal.judge_attempt_id.to_string(),
                ":process_id": process_instance_id.to_string(),
                ":state": fields.state,
                ":parse_result": fields.parse_result,
                ":safe_output": fields.safe_output_json,
                ":raw_hash": fields.raw_output_sha256,
                ":raw_bytes": fields.raw_output_bytes,
                ":error_class": fields.stable_error_class,
                ":created_at": terminal.created_at_unix_ms,
                ":payload_hash": expected_hash,
            },
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn insert_judge_terminal_state(
    connection: &Connection,
    process_instance_id: Uuid,
    terminal: &JudgeAttemptTerminal,
    fields: &PreparedTerminalFields<'_>,
    state_hash: &str,
) -> Result<(), LedgerError> {
    connection
        .execute(
            "INSERT INTO judge_attempt_state_events (
                judge_attempt_state_event_id, judge_attempt_id,
                process_instance_id, dead_process_instance_id, state,
                parse_result, safe_output_json, raw_output_sha256,
                raw_output_bytes, stable_error_class, created_at_unix_ms,
                canonical_payload_hash
             ) VALUES (
                :event_id, :attempt_id, :process_id, NULL, :state,
                :parse_result, :safe_output, :raw_hash, :raw_bytes,
                :error_class, :created_at, :payload_hash
             )",
            named_params! {
                ":event_id": terminal.judge_attempt_state_event_id.to_string(),
                ":attempt_id": terminal.judge_attempt_id.to_string(),
                ":process_id": process_instance_id.to_string(),
                ":state": fields.state,
                ":parse_result": fields.parse_result,
                ":safe_output": fields.safe_output_json,
                ":raw_hash": fields.raw_output_sha256,
                ":raw_bytes": fields.raw_output_bytes,
                ":error_class": fields.stable_error_class,
                ":created_at": terminal.created_at_unix_ms,
                ":payload_hash": state_hash,
            },
        )
        .map_err(database_error)?;
    Ok(())
}

fn evaluation_key_exists(
    connection: &Connection,
    evaluation: &EvaluationRecord,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM evaluations
                WHERE evaluation_id = ?1
                   OR (shadow_attempt_id = ?2 AND evaluator_version = ?3)
             )",
            params![
                evaluation.evaluation_id.to_string(),
                evaluation.shadow_attempt_id.to_string(),
                evaluation.evaluator_version,
            ],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn evaluation_for_identity_exists(
    connection: &Connection,
    shadow_attempt_id: Uuid,
    evaluator_version: &str,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM evaluations
                WHERE shadow_attempt_id = ?1 AND evaluator_version = ?2
             )",
            params![shadow_attempt_id.to_string(), evaluator_version],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn evaluation_id_for_identity(
    connection: &Connection,
    shadow_attempt_id: Uuid,
    evaluator_version: &str,
) -> Result<Option<Uuid>, LedgerError> {
    connection
        .query_row(
            "SELECT evaluation_id FROM evaluations
             WHERE shadow_attempt_id = ?1 AND evaluator_version = ?2",
            params![shadow_attempt_id.to_string(), evaluator_version],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .map(|value| parse_uuid(&value))
        .transpose()
}

fn judge_attempt_for_identity_exists(
    connection: &Connection,
    shadow_attempt_id: Uuid,
    evaluator_version: &str,
) -> Result<bool, LedgerError> {
    connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM judge_attempts
                WHERE shadow_attempt_id = ?1 AND evaluator_version = ?2
             )",
            params![shadow_attempt_id.to_string(), evaluator_version],
            |row| row.get(0),
        )
        .map_err(database_error)
}

fn policy_config_matches_hash(
    connection: &Connection,
    shadow_attempt_id: Uuid,
    expected_hash: &str,
) -> Result<bool, LedgerError> {
    let Some(policy) = verified_shadow_judge_policy(connection, shadow_attempt_id)? else {
        return Ok(false);
    };
    Ok(policy.config.contract_sha256().ok().as_deref() == Some(expected_hash))
}

fn terminal_judge_config_matches_policy(
    connection: &Connection,
    terminal: &JudgeAttemptTerminal,
    parent: &JudgeAttemptParent,
) -> Result<bool, LedgerError> {
    match &terminal.state {
        JudgeTerminalState::Valid { .. } | JudgeTerminalState::Invalid { .. } => {
            let Some(policy) = verified_shadow_judge_policy(connection, parent.shadow_attempt_id)?
            else {
                return Ok(false);
            };
            let Some(expected) = policy.config.contract_sha256().ok() else {
                return Ok(false);
            };
            Ok(
                terminal.judge_config_sha256.as_deref() == Some(expected.as_str())
                    && terminal
                        .evaluation
                        .as_ref()
                        .is_none_or(|evaluation| evaluation.judge_config_sha256 == expected),
            )
        }
        JudgeTerminalState::TransportFailure { .. } | JudgeTerminalState::CanceledShutdown => {
            Ok(terminal.judge_config_sha256.is_none())
        }
    }
}

fn evaluation_matches(
    connection: &Connection,
    evaluation: &EvaluationRecord,
) -> Result<bool, LedgerError> {
    let hash = evaluation_hash(evaluation)?;
    let contract = evaluation.judge_contract.as_ref();
    let value = &evaluation.evaluation;
    let response_equivalence = score_value(value.response_equivalence);
    let response_equivalence_bits = score_bits(value.response_equivalence);
    let trajectory_equivalence = score_value(value.trajectory_equivalence);
    let trajectory_equivalence_bits = score_bits(value.trajectory_equivalence);
    let judge_confidence = score_value(value.judge_confidence);
    let judge_confidence_bits = score_bits(value.judge_confidence);
    let response_weight = score_value(value.response_weight);
    let response_weight_bits = score_bits(value.response_weight);
    let trajectory_weight = score_value(value.trajectory_weight);
    let trajectory_weight_bits = score_bits(value.trajectory_weight);
    let aggregate_score = score_value(value.aggregate_score);
    let aggregate_score_bits = score_bits(value.aggregate_score);
    let exact: bool = connection
        .query_row(
            "SELECT EXISTS(
                SELECT 1 FROM evaluations
                WHERE evaluation_id = :evaluation_id
                  AND shadow_attempt_id = :shadow_attempt_id
                  AND evaluator_version = :evaluator_version
                  AND source = :source
                  AND judge_model IS :judge_model
                  AND judge_model_revision IS :judge_revision
                  AND prompt_version IS :prompt_version
                  AND prompt_sha256 IS :prompt_sha
                  AND rubric_version IS :rubric_version
                  AND rubric_sha256 IS :rubric_sha
                  AND output_schema_version IS :schema_version
                  AND output_schema_sha256 IS :schema_sha
                  AND response_equivalence IS :response_score
                  AND response_equivalence_bits IS :response_bits
                  AND trajectory_equivalence IS :trajectory_score
                  AND trajectory_equivalence_bits IS :trajectory_bits
                  AND judge_confidence IS :confidence
                  AND judge_confidence_bits IS :confidence_bits
                  AND response_weight IS :response_weight
                  AND response_weight_bits IS :response_weight_bits
                  AND trajectory_weight IS :trajectory_weight
                  AND trajectory_weight_bits IS :trajectory_weight_bits
                  AND aggregate_score IS :aggregate
                  AND aggregate_score_bits IS :aggregate_bits
                  AND label = :label AND binary_label IS :binary_label
                  AND rationale IS :rationale AND is_partial = :is_partial
                  AND promotion_eligible = :promotion_eligible
                  AND created_at_unix_ms = :created_at
                  AND canonical_payload_hash = :payload_hash
             )",
            named_params! {
                ":evaluation_id": evaluation.evaluation_id.to_string(),
                ":shadow_attempt_id": evaluation.shadow_attempt_id.to_string(),
                ":evaluator_version": evaluation.evaluator_version,
                ":source": source_str(value.source),
                ":judge_model": contract.map(|value| value.judge_model.as_str()),
                ":judge_revision": contract.map(|value| value.judge_model_revision.as_str()),
                ":prompt_version": contract.map(|value| value.prompt_version.as_str()),
                ":prompt_sha": contract.map(|value| value.prompt_sha256.as_str()),
                ":rubric_version": contract.map(|value| value.rubric_version.as_str()),
                ":rubric_sha": contract.map(|value| value.rubric_sha256.as_str()),
                ":schema_version": contract.map(|value| value.output_schema_version),
                ":schema_sha": contract.map(|value| value.output_schema_sha256.as_str()),
                ":response_score": response_equivalence,
                ":response_bits": response_equivalence_bits,
                ":trajectory_score": trajectory_equivalence,
                ":trajectory_bits": trajectory_equivalence_bits,
                ":confidence": judge_confidence,
                ":confidence_bits": judge_confidence_bits,
                ":response_weight": response_weight,
                ":response_weight_bits": response_weight_bits,
                ":trajectory_weight": trajectory_weight,
                ":trajectory_weight_bits": trajectory_weight_bits,
                ":aggregate": aggregate_score,
                ":aggregate_bits": aggregate_score_bits,
                ":label": label_str(value.label),
                ":binary_label": value.binary_label.map(binary_label_str),
                ":rationale": value.rationale.as_deref(),
                ":is_partial": i64::from(value.is_partial),
                ":promotion_eligible": i64::from(value.promotion_eligible),
                ":created_at": evaluation.created_at_unix_ms,
                ":payload_hash": hash,
            },
            |row| row.get(0),
        )
        .map_err(database_error)?;
    if !exact {
        return Ok(false);
    }
    hard_failures_match(connection, evaluation)
}

fn insert_evaluation(
    connection: &Connection,
    evaluation: &EvaluationRecord,
) -> Result<(), LedgerError> {
    let hash = evaluation_hash(evaluation)?;
    let contract = evaluation.judge_contract.as_ref();
    let value = &evaluation.evaluation;
    connection
        .execute(
            "INSERT INTO evaluations (
                evaluation_id, shadow_attempt_id, evaluator_version, source,
                judge_model, judge_model_revision, prompt_version, prompt_sha256,
                rubric_version, rubric_sha256, output_schema_version,
                output_schema_sha256, response_equivalence,
                response_equivalence_bits, trajectory_equivalence,
                trajectory_equivalence_bits, judge_confidence,
                judge_confidence_bits, response_weight, response_weight_bits,
                trajectory_weight, trajectory_weight_bits, aggregate_score,
                aggregate_score_bits, label, binary_label, rationale, is_partial,
                promotion_eligible, created_at_unix_ms, canonical_payload_hash
             ) VALUES (
                :evaluation_id, :shadow_attempt_id, :evaluator_version, :source,
                :judge_model, :judge_revision, :prompt_version, :prompt_sha,
                :rubric_version, :rubric_sha, :schema_version, :schema_sha,
                :response_score, :response_bits, :trajectory_score,
                :trajectory_bits, :confidence, :confidence_bits,
                :response_weight, :response_weight_bits, :trajectory_weight,
                :trajectory_weight_bits, :aggregate, :aggregate_bits, :label,
                :binary_label, :rationale, :is_partial, :promotion_eligible,
                :created_at, :payload_hash
             )",
            named_params! {
                ":evaluation_id": evaluation.evaluation_id.to_string(),
                ":shadow_attempt_id": evaluation.shadow_attempt_id.to_string(),
                ":evaluator_version": evaluation.evaluator_version,
                ":source": source_str(value.source),
                ":judge_model": contract.map(|value| value.judge_model.as_str()),
                ":judge_revision": contract.map(|value| value.judge_model_revision.as_str()),
                ":prompt_version": contract.map(|value| value.prompt_version.as_str()),
                ":prompt_sha": contract.map(|value| value.prompt_sha256.as_str()),
                ":rubric_version": contract.map(|value| value.rubric_version.as_str()),
                ":rubric_sha": contract.map(|value| value.rubric_sha256.as_str()),
                ":schema_version": contract.map(|value| value.output_schema_version),
                ":schema_sha": contract.map(|value| value.output_schema_sha256.as_str()),
                ":response_score": score_value(value.response_equivalence),
                ":response_bits": score_bits(value.response_equivalence),
                ":trajectory_score": score_value(value.trajectory_equivalence),
                ":trajectory_bits": score_bits(value.trajectory_equivalence),
                ":confidence": score_value(value.judge_confidence),
                ":confidence_bits": score_bits(value.judge_confidence),
                ":response_weight": score_value(value.response_weight),
                ":response_weight_bits": score_bits(value.response_weight),
                ":trajectory_weight": score_value(value.trajectory_weight),
                ":trajectory_weight_bits": score_bits(value.trajectory_weight),
                ":aggregate": score_value(value.aggregate_score),
                ":aggregate_bits": score_bits(value.aggregate_score),
                ":label": label_str(value.label),
                ":binary_label": value.binary_label.map(binary_label_str),
                ":rationale": value.rationale.as_deref(),
                ":is_partial": i64::from(value.is_partial),
                ":promotion_eligible": i64::from(value.promotion_eligible),
                ":created_at": evaluation.created_at_unix_ms,
                ":payload_hash": hash,
            },
        )
        .map_err(database_error)?;
    for failure in &value.hard_failures {
        let failure_hash = hard_failure_hash(evaluation.evaluation_id, *failure)?;
        connection
            .execute(
                "INSERT INTO evaluation_hard_failures (
                    evaluation_id, hard_failure, canonical_ordinal,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4)",
                params![
                    evaluation.evaluation_id.to_string(),
                    failure.value,
                    failure.ordinal,
                    failure_hash,
                ],
            )
            .map_err(database_error)?;
    }
    Ok(())
}

fn hard_failures_match(
    connection: &Connection,
    evaluation: &EvaluationRecord,
) -> Result<bool, LedgerError> {
    let count: i64 = connection
        .query_row(
            "SELECT COUNT(*) FROM evaluation_hard_failures WHERE evaluation_id = ?1",
            params![evaluation.evaluation_id.to_string()],
            |row| row.get(0),
        )
        .map_err(database_error)?;
    if usize::try_from(count).ok() != Some(evaluation.evaluation.hard_failures.len()) {
        return Ok(false);
    }
    for failure in &evaluation.evaluation.hard_failures {
        let hash = hard_failure_hash(evaluation.evaluation_id, *failure)?;
        let exact: bool = connection
            .query_row(
                "SELECT EXISTS(
                    SELECT 1 FROM evaluation_hard_failures
                    WHERE evaluation_id = ?1 AND hard_failure = ?2
                      AND canonical_ordinal = ?3 AND canonical_payload_hash = ?4
                 )",
                params![
                    evaluation.evaluation_id.to_string(),
                    failure.value,
                    failure.ordinal,
                    hash,
                ],
                |row| row.get(0),
            )
            .map_err(database_error)?;
        if !exact {
            return Ok(false);
        }
    }
    Ok(true)
}

fn load_shadow_anchor(
    connection: &Connection,
    shadow_attempt_id: Uuid,
) -> Result<Option<Uuid>, LedgerError> {
    Ok(
        verified_shadow_attempt_context(connection, shadow_attempt_id)?
            .map(|value| value.anchor_id),
    )
}

fn append_judge_conflict_health(
    connection: &Connection,
    health_event_id: Uuid,
    project_uuid: Uuid,
    process_instance_id: Uuid,
    shadow_attempt_id: Option<Uuid>,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    let anchor_id = shadow_attempt_id
        .map(|attempt_id| load_shadow_anchor(connection, attempt_id))
        .transpose()?
        .flatten();
    append_integrity_health(
        connection,
        health_event_id,
        project_uuid,
        process_instance_id,
        anchor_id,
        None,
        created_at_unix_ms,
    )
}

fn judge_attempt_hash(
    process_instance_id: Uuid,
    start: &JudgeAttemptStart,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "judge_attempt_id": start.judge_attempt_id,
        "shadow_attempt_id": start.shadow_attempt_id,
        "process_instance_id": process_instance_id,
        "learning_generation_id": start.learning_generation_id,
        "evaluator_version": start.evaluator_version,
        "judge_model": start.judge_model,
        "judge_model_revision": start.judge_model_revision,
        "prompt_version": start.prompt_version,
        "prompt_sha256": start.prompt_sha256,
        "rubric_version": start.rubric_version,
        "rubric_sha256": start.rubric_sha256,
        "output_schema_version": start.output_schema_version,
        "output_schema_sha256": start.output_schema_sha256,
        "judge_input_sha256": start.judge_input_sha256,
        "candidate_response_json": start.candidate_response_json,
        "candidate_response_fingerprint": start.candidate_response_fingerprint,
        "attempt_ordinal": start.attempt_ordinal,
        "created_at_unix_ms": start.created_at_unix_ms,
    }))
}

fn judge_start_state_hash(
    process_instance_id: Uuid,
    start: &JudgeAttemptStart,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "judge_attempt_state_event_id": start.judge_attempt_state_event_id,
        "judge_attempt_id": start.judge_attempt_id,
        "process_instance_id": process_instance_id,
        "dead_process_instance_id": Json::Null,
        "state": "started",
        "parse_result": Json::Null,
        "safe_output_json": Json::Null,
        "raw_output_sha256": Json::Null,
        "raw_output_bytes": Json::Null,
        "stable_error_class": Json::Null,
        "created_at_unix_ms": start.created_at_unix_ms,
    }))
}

fn judge_terminal_state_hash(
    process_instance_id: Uuid,
    terminal: &JudgeAttemptTerminal,
    fields: &PreparedTerminalFields<'_>,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "judge_attempt_state_event_id": terminal.judge_attempt_state_event_id,
        "judge_attempt_id": terminal.judge_attempt_id,
        "process_instance_id": process_instance_id,
        "dead_process_instance_id": Json::Null,
        "state": fields.state,
        "parse_result": fields.parse_result,
        "safe_output_json": fields.safe_output_json,
        "raw_output_sha256": fields.raw_output_sha256,
        "raw_output_bytes": fields.raw_output_bytes,
        "stable_error_class": fields.stable_error_class,
        "created_at_unix_ms": terminal.created_at_unix_ms,
    }))
}

fn evaluation_hash(evaluation: &EvaluationRecord) -> Result<String, LedgerError> {
    let value = &evaluation.evaluation;
    let contract = evaluation.judge_contract.as_ref();
    hash_json(&json!({
        "evaluation_id": evaluation.evaluation_id,
        "shadow_attempt_id": evaluation.shadow_attempt_id,
        "evaluator_version": evaluation.evaluator_version,
        "source": source_str(value.source),
        "judge_model": contract.map(|value| value.judge_model.as_str()),
        "judge_model_revision": contract.map(|value| value.judge_model_revision.as_str()),
        "prompt_version": contract.map(|value| value.prompt_version.as_str()),
        "prompt_sha256": contract.map(|value| value.prompt_sha256.as_str()),
        "rubric_version": contract.map(|value| value.rubric_version.as_str()),
        "rubric_sha256": contract.map(|value| value.rubric_sha256.as_str()),
        "output_schema_version": contract.map(|value| value.output_schema_version),
        "output_schema_sha256": contract.map(|value| value.output_schema_sha256.as_str()),
        "response_equivalence": score_value(value.response_equivalence),
        "response_equivalence_bits": score_bits_identity(value.response_equivalence),
        "trajectory_equivalence": score_value(value.trajectory_equivalence),
        "trajectory_equivalence_bits": score_bits_identity(value.trajectory_equivalence),
        "judge_confidence": score_value(value.judge_confidence),
        "judge_confidence_bits": score_bits_identity(value.judge_confidence),
        "response_weight": score_value(value.response_weight),
        "response_weight_bits": score_bits_identity(value.response_weight),
        "trajectory_weight": score_value(value.trajectory_weight),
        "trajectory_weight_bits": score_bits_identity(value.trajectory_weight),
        "aggregate_score": score_value(value.aggregate_score),
        "aggregate_score_bits": score_bits_identity(value.aggregate_score),
        "label": label_str(value.label),
        "binary_label": value.binary_label.map(binary_label_str),
        "hard_failures": value.hard_failures.iter().map(|value| value.value).collect::<Vec<_>>(),
        "rationale": value.rationale,
        "is_partial": value.is_partial,
        "promotion_eligible": value.promotion_eligible,
        "created_at_unix_ms": evaluation.created_at_unix_ms,
    }))
}

fn hard_failure_hash(
    evaluation_id: Uuid,
    failure: PreparedHardFailure,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "evaluation_id": evaluation_id,
        "hard_failure": failure.value,
        "canonical_ordinal": failure.ordinal,
    }))
}

fn score_value(value: Option<PreparedScore>) -> Option<f64> {
    value.map(PreparedScore::value)
}

fn score_bits(value: Option<PreparedScore>) -> Option<i64> {
    value.map(PreparedScore::sqlite_bits)
}

fn score_bits_identity(value: Option<PreparedScore>) -> Option<String> {
    value.map(|value| format!("{:016x}", value.bits))
}

const fn source_str(value: JudgeEvaluationSourceV1) -> &'static str {
    match value {
        JudgeEvaluationSourceV1::DeterministicValidator => "deterministic_validator",
        JudgeEvaluationSourceV1::Judge => "judge",
    }
}

const fn label_str(value: JudgeLabelV1) -> &'static str {
    match value {
        JudgeLabelV1::Pass => "pass",
        JudgeLabelV1::Fail => "fail",
        JudgeLabelV1::Ambiguous => "ambiguous",
    }
}

const fn binary_label_str(value: JudgeBinaryLabelV1) -> &'static str {
    match value {
        JudgeBinaryLabelV1::Pass => "pass",
        JudgeBinaryLabelV1::Fail => "fail",
    }
}

impl LedgerRepository {
    pub(crate) fn record_judge_attempt_start(
        &mut self,
        start: &JudgeAttemptStart,
    ) -> Result<JudgeRecordAck, LedgerError> {
        self.record_judge_attempt_start_with_start_check(start, || Some(()))
    }

    pub(crate) fn record_judge_attempt_start_with_start_check<G>(
        &mut self,
        start: &JudgeAttemptStart,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<JudgeRecordAck, LedgerError>
    where
        G: TransactionStartGuard,
    {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(JudgeRecordAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_) if !start_guard.permits_transaction() => {
                return Ok(JudgeRecordAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(JudgeRecordAck::TransactionNotStarted);
        }
        drop(start_guard);
        if !originating_process_is_live(&transaction, project_uuid, process_instance_id)? {
            return Ok(JudgeRecordAck::OriginatingProcessNotLive);
        }

        let shadow = verified_shadow_attempt_context(&transaction, start.shadow_attempt_id)?;
        let shadow_is_valid = shadow.as_ref().is_some_and(|shadow| {
            shadow.process_instance_id == process_instance_id
                && shadow.learning_generation_id == start.learning_generation_id
                && shadow.evaluator_version == start.evaluator_version
                && start.created_at_unix_ms >= shadow.started_at_unix_ms
        });
        let judge_policy_is_valid = policy_config_matches_hash(
            &transaction,
            start.shadow_attempt_id,
            &start.judge_config_sha256,
        )?;
        let judge_input_context =
            serde_json::from_str::<RouterResponseProjectionV1>(&start.candidate_response_json)
                .ok()
                .map(|candidate_response| {
                    verified_shadow_judge_input_context(
                        &transaction,
                        start.shadow_attempt_id,
                        &candidate_response,
                    )
                })
                .transpose()?
                .flatten();
        let judge_input_is_valid = judge_input_context.as_ref().is_some_and(|context| {
            context.judge_input_sha256 == start.judge_input_sha256
                && context.candidate_response_json == start.candidate_response_json
                && context.candidate_response_fingerprint == start.candidate_response_fingerprint
        });
        let repair_is_valid = start.attempt_ordinal == 0
            || repair_predecessor_allows_start(&transaction, start, process_instance_id)?;
        if !shadow_is_valid || !judge_policy_is_valid || !judge_input_is_valid || !repair_is_valid {
            append_judge_conflict_health(
                &transaction,
                start.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                Some(start.shadow_attempt_id),
                start.created_at_unix_ms,
            )?;
            transaction.commit().map_err(database_error)?;
            return Ok(JudgeRecordAck::Conflict);
        }

        let attempt_hash = judge_attempt_hash(process_instance_id, start)?;
        let state_hash = judge_start_state_hash(process_instance_id, start)?;
        let attempt_exists = judge_attempt_key_exists(&transaction, start)?;
        let state_exists = judge_start_state_key_exists(&transaction, start)?;
        if attempt_exists || state_exists {
            let exact =
                judge_attempt_matches(&transaction, process_instance_id, start, &attempt_hash)?
                    && judge_start_state_matches(
                        &transaction,
                        process_instance_id,
                        start,
                        &state_hash,
                    )?;
            if exact {
                transaction.commit().map_err(database_error)?;
                return Ok(JudgeRecordAck::AlreadyApplied);
            }
            append_judge_conflict_health(
                &transaction,
                start.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                Some(start.shadow_attempt_id),
                start.created_at_unix_ms,
            )?;
            transaction.commit().map_err(database_error)?;
            return Ok(JudgeRecordAck::Conflict);
        }
        if evaluation_for_identity_exists(
            &transaction,
            start.shadow_attempt_id,
            &start.evaluator_version,
        )? {
            append_judge_conflict_health(
                &transaction,
                start.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                Some(start.shadow_attempt_id),
                start.created_at_unix_ms,
            )?;
            transaction.commit().map_err(database_error)?;
            return Ok(JudgeRecordAck::Conflict);
        }
        if !canonical_shadow_attempt_is_unterminalized(&transaction, start.shadow_attempt_id)? {
            append_judge_conflict_health(
                &transaction,
                start.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                Some(start.shadow_attempt_id),
                start.created_at_unix_ms,
            )?;
            transaction.commit().map_err(database_error)?;
            return Ok(JudgeRecordAck::Conflict);
        }

        transaction
            .execute(
                "INSERT INTO judge_attempts (
                    judge_attempt_id, shadow_attempt_id, process_instance_id,
                    learning_generation_id, evaluator_version, judge_model,
                    judge_model_revision, prompt_version, prompt_sha256,
                    rubric_version, rubric_sha256, output_schema_version,
                    output_schema_sha256, judge_input_sha256,
                    candidate_response_json, candidate_response_fingerprint,
                    attempt_ordinal, created_at_unix_ms, canonical_payload_hash
                 ) VALUES (
                    ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8,
                    ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16,
                    ?17, ?18, ?19
                 )",
                params![
                    start.judge_attempt_id.to_string(),
                    start.shadow_attempt_id.to_string(),
                    process_instance_id.to_string(),
                    start.learning_generation_id.to_string(),
                    start.evaluator_version,
                    start.judge_model,
                    start.judge_model_revision,
                    start.prompt_version,
                    start.prompt_sha256,
                    start.rubric_version,
                    start.rubric_sha256,
                    start.output_schema_version,
                    start.output_schema_sha256,
                    start.judge_input_sha256,
                    start.candidate_response_json,
                    start.candidate_response_fingerprint,
                    start.attempt_ordinal,
                    start.created_at_unix_ms,
                    attempt_hash,
                ],
            )
            .map_err(database_error)?;
        transaction
            .execute(
                "INSERT INTO judge_attempt_state_events (
                    judge_attempt_state_event_id, judge_attempt_id,
                    process_instance_id, dead_process_instance_id, state,
                    parse_result, safe_output_json, raw_output_sha256,
                    raw_output_bytes, stable_error_class, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, NULL, 'started', NULL, NULL, NULL,
                           NULL, NULL, ?4, ?5)",
                params![
                    start.judge_attempt_state_event_id.to_string(),
                    start.judge_attempt_id.to_string(),
                    process_instance_id.to_string(),
                    start.created_at_unix_ms,
                    state_hash,
                ],
            )
            .map_err(database_error)?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(JudgeRecordAck::Applied)
    }

    pub(crate) fn record_judge_attempt_terminal(
        &mut self,
        terminal: &JudgeAttemptTerminal,
    ) -> Result<JudgeRecordAck, LedgerError> {
        self.record_judge_attempt_terminal_with_start_check(terminal, || Some(()))
    }

    pub(crate) fn record_judge_attempt_terminal_with_start_check<G>(
        &mut self,
        terminal: &JudgeAttemptTerminal,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<JudgeRecordAck, LedgerError>
    where
        G: TransactionStartGuard,
    {
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(JudgeRecordAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_) if !start_guard.permits_transaction() => {
                return Ok(JudgeRecordAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(JudgeRecordAck::TransactionNotStarted);
        }
        drop(start_guard);
        if !originating_process_is_live(&transaction, project_uuid, process_instance_id)? {
            return Ok(JudgeRecordAck::OriginatingProcessNotLive);
        }

        let Some(parent) = load_judge_attempt_parent(&transaction, terminal.judge_attempt_id)?
        else {
            append_judge_conflict_health(
                &transaction,
                terminal.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                None,
                terminal.created_at_unix_ms,
            )?;
            transaction.commit().map_err(database_error)?;
            return Ok(JudgeRecordAck::Conflict);
        };
        let shadow = verified_shadow_attempt_context(&transaction, parent.shadow_attempt_id)?;
        let shadow_matches = shadow.as_ref().is_some_and(|shadow| {
            shadow.process_instance_id == process_instance_id
                && shadow.learning_generation_id.to_string() == parent.learning_generation_id
                && shadow.evaluator_version == parent.evaluator_version
                && terminal
                    .evaluation
                    .as_ref()
                    .is_none_or(|evaluation| evaluation.evaluation.is_partial == shadow.is_partial)
        });
        if parent.process_instance_id != process_instance_id.to_string()
            || !shadow_matches
            || !judge_attempt_parent_is_canonical(&transaction, &parent)?
            || terminal.created_at_unix_ms < parent.created_at_unix_ms
            || !terminal_evaluation_matches_parent(terminal, &parent)
            || !terminal_judge_config_matches_policy(&transaction, terminal, &parent)?
        {
            append_judge_conflict_health(
                &transaction,
                terminal.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                Some(parent.shadow_attempt_id),
                terminal.created_at_unix_ms,
            )?;
            transaction.commit().map_err(database_error)?;
            return Ok(JudgeRecordAck::Conflict);
        }

        let fields = terminal.fields();
        let state_hash = judge_terminal_state_hash(process_instance_id, terminal, &fields)?;
        let state_exists = judge_terminal_state_key_exists(&transaction, terminal)?;
        let evaluation_exists = evaluation_for_identity_exists(
            &transaction,
            parent.shadow_attempt_id,
            &parent.evaluator_version,
        )?;
        if state_exists || evaluation_exists {
            let exact_state = judge_terminal_state_matches(
                &transaction,
                process_instance_id,
                terminal,
                &fields,
                &state_hash,
            )?;
            let exact_evaluation = match terminal.evaluation.as_ref() {
                Some(evaluation) => evaluation_matches(&transaction, evaluation)?,
                None if !evaluation_exists => true,
                // An exact retry of the initial invalid terminal remains exact
                // only after its canonical repair produced the evaluation.
                None if parent.attempt_ordinal == 0 && fields.state == "invalid" => {
                    match evaluation_id_for_identity(
                        &transaction,
                        parent.shadow_attempt_id,
                        &parent.evaluator_version,
                    )? {
                        Some(evaluation_id) => verified_judge_evaluation_provenance(
                            &transaction,
                            evaluation_id,
                            parent.shadow_attempt_id,
                        )?,
                        None => false,
                    }
                }
                None => false,
            };
            if exact_state && exact_evaluation {
                transaction.commit().map_err(database_error)?;
                return Ok(JudgeRecordAck::AlreadyApplied);
            }
            append_judge_conflict_health(
                &transaction,
                terminal.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                Some(parent.shadow_attempt_id),
                terminal.created_at_unix_ms,
            )?;
            transaction.commit().map_err(database_error)?;
            return Ok(JudgeRecordAck::Conflict);
        }
        if !canonical_shadow_attempt_is_unterminalized(&transaction, parent.shadow_attempt_id)? {
            append_judge_conflict_health(
                &transaction,
                terminal.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                Some(parent.shadow_attempt_id),
                terminal.created_at_unix_ms,
            )?;
            transaction.commit().map_err(database_error)?;
            return Ok(JudgeRecordAck::Conflict);
        }

        insert_judge_terminal_state(
            &transaction,
            process_instance_id,
            terminal,
            &fields,
            &state_hash,
        )?;
        if let Some(evaluation) = terminal.evaluation.as_ref() {
            insert_evaluation(&transaction, evaluation)?;
        }
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(JudgeRecordAck::Applied)
    }

    pub(crate) fn record_evaluation(
        &mut self,
        evaluation: &EvaluationRecord,
    ) -> Result<JudgeRecordAck, LedgerError> {
        self.record_evaluation_with_start_check(evaluation, || Some(()))
    }

    pub(crate) fn record_evaluation_with_start_check<G>(
        &mut self,
        evaluation: &EvaluationRecord,
        start_check: impl FnOnce() -> Option<G>,
    ) -> Result<JudgeRecordAck, LedgerError>
    where
        G: TransactionStartGuard,
    {
        if evaluation.evaluation.source != JudgeEvaluationSourceV1::DeterministicValidator {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let project_uuid = self.project_uuid;
        let process_instance_id = self.process_instance_id;
        let database_path = self.database_path.clone();
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        let Some(start_guard) = start_check() else {
            return Ok(JudgeRecordAck::TransactionNotStarted);
        };
        let transaction = match self
            .connection
            .transaction_with_behavior(TransactionBehavior::Immediate)
        {
            Ok(transaction) => transaction,
            Err(_) if !start_guard.permits_transaction() => {
                return Ok(JudgeRecordAck::TransactionNotStarted);
            }
            Err(error) => return Err(database_error(error)),
        };
        if !start_guard.permits_transaction() {
            drop(transaction);
            return Ok(JudgeRecordAck::TransactionNotStarted);
        }
        drop(start_guard);
        if !originating_process_is_live(&transaction, project_uuid, process_instance_id)? {
            return Ok(JudgeRecordAck::OriginatingProcessNotLive);
        }
        let context = verified_shadow_attempt_context(&transaction, evaluation.shadow_attempt_id)?;
        let anchor_id = context.as_ref().map(|value| value.anchor_id);
        let judge_policy_is_valid = policy_config_matches_hash(
            &transaction,
            evaluation.shadow_attempt_id,
            &evaluation.judge_config_sha256,
        )?;
        if context.as_ref().is_none_or(|context| {
            context.process_instance_id != process_instance_id
                || context.evaluator_version != evaluation.evaluator_version
                || context.is_partial != evaluation.evaluation.is_partial
                || evaluation.created_at_unix_ms < context.started_at_unix_ms
        }) || !judge_policy_is_valid
        {
            append_integrity_health(
                &transaction,
                evaluation.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                anchor_id,
                None,
                evaluation.created_at_unix_ms,
            )?;
            transaction.commit().map_err(database_error)?;
            return Ok(JudgeRecordAck::Conflict);
        }
        if judge_attempt_for_identity_exists(
            &transaction,
            evaluation.shadow_attempt_id,
            &evaluation.evaluator_version,
        )? {
            append_integrity_health(
                &transaction,
                evaluation.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                anchor_id,
                None,
                evaluation.created_at_unix_ms,
            )?;
            transaction.commit().map_err(database_error)?;
            return Ok(JudgeRecordAck::Conflict);
        }
        if evaluation_key_exists(&transaction, evaluation)? {
            let acknowledgement = if evaluation_matches(&transaction, evaluation)? {
                JudgeRecordAck::AlreadyApplied
            } else {
                append_integrity_health(
                    &transaction,
                    evaluation.conflict_health_event_id,
                    project_uuid,
                    process_instance_id,
                    anchor_id,
                    None,
                    evaluation.created_at_unix_ms,
                )?;
                JudgeRecordAck::Conflict
            };
            transaction.commit().map_err(database_error)?;
            return Ok(acknowledgement);
        }
        if !canonical_shadow_attempt_is_unterminalized(&transaction, evaluation.shadow_attempt_id)?
        {
            append_integrity_health(
                &transaction,
                evaluation.conflict_health_event_id,
                project_uuid,
                process_instance_id,
                anchor_id,
                None,
                evaluation.created_at_unix_ms,
            )?;
            transaction.commit().map_err(database_error)?;
            return Ok(JudgeRecordAck::Conflict);
        }
        insert_evaluation(&transaction, evaluation)?;
        enforce_sidecar_permissions(&database_path).map_err(map_fs_error)?;
        transaction.commit().map_err(database_error)?;
        Ok(JudgeRecordAck::Applied)
    }
}

struct JudgeAttemptParent {
    judge_attempt_id: Uuid,
    shadow_attempt_id: Uuid,
    process_instance_id: String,
    learning_generation_id: String,
    evaluator_version: String,
    judge_model: String,
    judge_model_revision: String,
    prompt_version: String,
    prompt_sha256: String,
    rubric_version: String,
    rubric_sha256: String,
    output_schema_version: u32,
    output_schema_sha256: String,
    judge_input_sha256: String,
    candidate_response_json: String,
    candidate_response_fingerprint: String,
    attempt_ordinal: u8,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

fn load_judge_attempt_parent(
    connection: &Connection,
    judge_attempt_id: Uuid,
) -> Result<Option<JudgeAttemptParent>, LedgerError> {
    connection
        .query_row(
            "SELECT judge_attempt_id, shadow_attempt_id, process_instance_id,
                    learning_generation_id, evaluator_version,
                    judge_model, judge_model_revision, prompt_version,
                    prompt_sha256, rubric_version, rubric_sha256,
                    output_schema_version, output_schema_sha256,
                    judge_input_sha256, candidate_response_json,
                    candidate_response_fingerprint, attempt_ordinal,
                    created_at_unix_ms, canonical_payload_hash
             FROM judge_attempts WHERE judge_attempt_id = ?1",
            params![judge_attempt_id.to_string()],
            |row| {
                let judge_attempt_id: String = row.get(0)?;
                let shadow_attempt_id: String = row.get(1)?;
                Ok(JudgeAttemptParent {
                    judge_attempt_id: parse_sql_uuid(0, &judge_attempt_id)?,
                    shadow_attempt_id: parse_sql_uuid(1, &shadow_attempt_id)?,
                    process_instance_id: row.get(2)?,
                    learning_generation_id: row.get(3)?,
                    evaluator_version: row.get(4)?,
                    judge_model: row.get(5)?,
                    judge_model_revision: row.get(6)?,
                    prompt_version: row.get(7)?,
                    prompt_sha256: row.get(8)?,
                    rubric_version: row.get(9)?,
                    rubric_sha256: row.get(10)?,
                    output_schema_version: row.get(11)?,
                    output_schema_sha256: row.get(12)?,
                    judge_input_sha256: row.get(13)?,
                    candidate_response_json: row.get(14)?,
                    candidate_response_fingerprint: row.get(15)?,
                    attempt_ordinal: row.get(16)?,
                    created_at_unix_ms: row.get(17)?,
                    canonical_payload_hash: row.get(18)?,
                })
            },
        )
        .optional()
        .map_err(database_error)
}

struct StoredJudgeState {
    event_id: Uuid,
    process_instance_id: Uuid,
    dead_process_instance_id: Option<Uuid>,
    state: String,
    parse_result: Option<String>,
    safe_output_json: Option<String>,
    raw_output_sha256: Option<String>,
    raw_output_bytes: Option<i64>,
    stable_error_class: Option<String>,
    created_at_unix_ms: i64,
    canonical_payload_hash: String,
}

fn load_judge_state(
    connection: &Connection,
    judge_attempt_id: Uuid,
    terminal: bool,
) -> Result<Option<StoredJudgeState>, LedgerError> {
    let predicate = if terminal {
        "state <> 'started'"
    } else {
        "state = 'started'"
    };
    let sql = format!(
        "SELECT judge_attempt_state_event_id, process_instance_id,
                dead_process_instance_id, state, parse_result, safe_output_json,
                raw_output_sha256, raw_output_bytes, stable_error_class,
                created_at_unix_ms, canonical_payload_hash
         FROM judge_attempt_state_events
         WHERE judge_attempt_id = ?1 AND {predicate}"
    );
    connection
        .query_row(&sql, params![judge_attempt_id.to_string()], |row| {
            let event_id: String = row.get(0)?;
            let process_instance_id: String = row.get(1)?;
            let dead_process_instance_id: Option<String> = row.get(2)?;
            Ok(StoredJudgeState {
                event_id: parse_sql_uuid(0, &event_id)?,
                process_instance_id: parse_sql_uuid(1, &process_instance_id)?,
                dead_process_instance_id: dead_process_instance_id
                    .as_deref()
                    .map(|value| parse_sql_uuid(2, value))
                    .transpose()?,
                state: row.get(3)?,
                parse_result: row.get(4)?,
                safe_output_json: row.get(5)?,
                raw_output_sha256: row.get(6)?,
                raw_output_bytes: row.get(7)?,
                stable_error_class: row.get(8)?,
                created_at_unix_ms: row.get(9)?,
                canonical_payload_hash: row.get(10)?,
            })
        })
        .optional()
        .map_err(database_error)
}

fn judge_attempt_parent_is_canonical(
    connection: &Connection,
    parent: &JudgeAttemptParent,
) -> Result<bool, LedgerError> {
    let Some(shadow) = verified_shadow_attempt_context(connection, parent.shadow_attempt_id)?
    else {
        return Ok(false);
    };
    if parent.created_at_unix_ms < shadow.started_at_unix_ms {
        return Ok(false);
    }
    let expected_hash = hash_json(&json!({
        "judge_attempt_id": parent.judge_attempt_id,
        "shadow_attempt_id": parent.shadow_attempt_id,
        "process_instance_id": parse_uuid(&parent.process_instance_id)?,
        "learning_generation_id": parse_uuid(&parent.learning_generation_id)?,
        "evaluator_version": parent.evaluator_version,
        "judge_model": parent.judge_model,
        "judge_model_revision": parent.judge_model_revision,
        "prompt_version": parent.prompt_version,
        "prompt_sha256": parent.prompt_sha256,
        "rubric_version": parent.rubric_version,
        "rubric_sha256": parent.rubric_sha256,
        "output_schema_version": parent.output_schema_version,
        "output_schema_sha256": parent.output_schema_sha256,
        "judge_input_sha256": parent.judge_input_sha256,
        "candidate_response_json": parent.candidate_response_json,
        "candidate_response_fingerprint": parent.candidate_response_fingerprint,
        "attempt_ordinal": parent.attempt_ordinal,
        "created_at_unix_ms": parent.created_at_unix_ms,
    }))?;
    if expected_hash != parent.canonical_payload_hash {
        return Ok(false);
    }
    let Ok(candidate_response) =
        serde_json::from_str::<RouterResponseProjectionV1>(&parent.candidate_response_json)
    else {
        return Ok(false);
    };
    let Some(input_context) = verified_shadow_judge_input_context(
        connection,
        parent.shadow_attempt_id,
        &candidate_response,
    )?
    else {
        return Ok(false);
    };
    if input_context.judge_input_sha256 != parent.judge_input_sha256
        || input_context.candidate_response_json != parent.candidate_response_json
        || input_context.candidate_response_fingerprint != parent.candidate_response_fingerprint
    {
        return Ok(false);
    }
    let Some(started) = load_judge_state(connection, parent.judge_attempt_id, false)? else {
        return Ok(false);
    };
    Ok(
        started.process_instance_id.to_string() == parent.process_instance_id
            && started.dead_process_instance_id.is_none()
            && started.state == "started"
            && started.parse_result.is_none()
            && started.safe_output_json.is_none()
            && started.raw_output_sha256.is_none()
            && started.raw_output_bytes.is_none()
            && started.stable_error_class.is_none()
            && started.created_at_unix_ms == parent.created_at_unix_ms
            && stored_judge_state_hash(parent.judge_attempt_id, &started)?
                == started.canonical_payload_hash,
    )
}

fn repair_predecessor_allows_start(
    connection: &Connection,
    repair: &JudgeAttemptStart,
    process_instance_id: Uuid,
) -> Result<bool, LedgerError> {
    let initial_id = connection
        .query_row(
            "SELECT judge_attempt_id FROM judge_attempts
             WHERE shadow_attempt_id = ?1 AND evaluator_version = ?2
               AND attempt_ordinal = 0",
            params![
                repair.shadow_attempt_id.to_string(),
                repair.evaluator_version,
            ],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .map(|value| parse_uuid(&value))
        .transpose()?;
    let Some(initial_id) = initial_id else {
        return Ok(false);
    };
    let Some(initial) = load_judge_attempt_parent(connection, initial_id)? else {
        return Ok(false);
    };
    if initial.process_instance_id != process_instance_id.to_string()
        || initial.learning_generation_id != repair.learning_generation_id.to_string()
        || initial.shadow_attempt_id != repair.shadow_attempt_id
        || initial.evaluator_version != repair.evaluator_version
        || initial.judge_model != repair.judge_model
        || initial.judge_model_revision != repair.judge_model_revision
        || initial.prompt_version != repair.prompt_version
        || initial.prompt_sha256 != repair.prompt_sha256
        || initial.rubric_version != repair.rubric_version
        || initial.rubric_sha256 != repair.rubric_sha256
        || initial.output_schema_version != repair.output_schema_version
        || initial.output_schema_sha256 != repair.output_schema_sha256
        || initial.judge_input_sha256 != repair.judge_input_sha256
        || initial.candidate_response_json != repair.candidate_response_json
        || initial.candidate_response_fingerprint != repair.candidate_response_fingerprint
        || initial.attempt_ordinal != 0
        || !judge_attempt_parent_is_canonical(connection, &initial)?
    {
        return Ok(false);
    }
    let Some(terminal) = load_judge_state(connection, initial_id, true)? else {
        return Ok(false);
    };
    let Some(policy) = verified_shadow_judge_policy(connection, repair.shadow_attempt_id)? else {
        return Ok(false);
    };
    Ok(terminal.process_instance_id == process_instance_id
        && terminal.dead_process_instance_id.is_none()
        && terminal.state == "invalid"
        && terminal.parse_result.as_deref() == Some("invalid")
        && terminal.safe_output_json.is_some()
        && terminal.raw_output_sha256.is_some()
        && terminal.raw_output_bytes.is_some()
        && terminal.stable_error_class.is_none()
        && terminal.created_at_unix_ms <= repair.created_at_unix_ms
        && stored_judge_state_hash(initial_id, &terminal)? == terminal.canonical_payload_hash
        && canonical_invalid_terminal_evidence(&terminal, policy.config.max_rationale_bytes)?)
}

fn stored_judge_state_hash(
    judge_attempt_id: Uuid,
    state: &StoredJudgeState,
) -> Result<String, LedgerError> {
    hash_json(&json!({
        "judge_attempt_state_event_id": state.event_id,
        "judge_attempt_id": judge_attempt_id,
        "process_instance_id": state.process_instance_id,
        "dead_process_instance_id": state.dead_process_instance_id,
        "state": state.state,
        "parse_result": state.parse_result,
        "safe_output_json": state.safe_output_json,
        "raw_output_sha256": state.raw_output_sha256,
        "raw_output_bytes": state.raw_output_bytes,
        "stable_error_class": state.stable_error_class,
        "created_at_unix_ms": state.created_at_unix_ms,
    }))
}

/// Terminalize every started Judge invocation owned by one fenced process.
pub(super) fn orphan_inflight_judge_attempts_in_transaction(
    connection: &Connection,
    project_uuid: Uuid,
    reconciler_process_instance_id: Uuid,
    dead_process_instance_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<usize, LedgerError> {
    if reconciler_process_instance_id == dead_process_instance_id || created_at_unix_ms < 0 {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    let mut statement = connection
        .prepare(
            "SELECT j.judge_attempt_id
             FROM judge_attempts AS j
             WHERE j.process_instance_id = ?1
               AND EXISTS (
                    SELECT 1 FROM judge_attempt_state_events AS s
                    WHERE s.judge_attempt_id = j.judge_attempt_id AND s.state = 'started'
               )
               AND NOT EXISTS (
                    SELECT 1 FROM judge_attempt_state_events AS s
                    WHERE s.judge_attempt_id = j.judge_attempt_id AND s.state <> 'started'
               )
             ORDER BY j.shadow_attempt_id, j.attempt_ordinal, j.judge_attempt_id",
        )
        .map_err(database_error)?;
    let attempt_ids = statement
        .query_map(params![dead_process_instance_id.to_string()], |row| {
            row.get::<_, String>(0)
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);

    let mut applied = 0usize;
    for attempt_id in attempt_ids {
        let attempt_id = parse_uuid(&attempt_id)?;
        let parent = load_judge_attempt_parent(connection, attempt_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if parent.process_instance_id != dead_process_instance_id.to_string()
            || !judge_attempt_parent_is_canonical(connection, &parent)?
            || created_at_unix_ms < parent.created_at_unix_ms
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let shadow_project: Option<String> = connection
            .query_row(
                "SELECT project_uuid FROM shadow_attempts WHERE shadow_attempt_id = ?1",
                params![parent.shadow_attempt_id.to_string()],
                |row| row.get(0),
            )
            .optional()
            .map_err(database_error)?;
        if shadow_project.as_deref() != Some(project_uuid.to_string().as_str()) {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }

        let event_id = Uuid::now_v7();
        let payload_hash = hash_json(&json!({
            "judge_attempt_state_event_id": event_id,
            "judge_attempt_id": attempt_id,
            "process_instance_id": reconciler_process_instance_id,
            "dead_process_instance_id": dead_process_instance_id,
            "state": "orphaned_in_flight",
            "parse_result": Json::Null,
            "safe_output_json": Json::Null,
            "raw_output_sha256": Json::Null,
            "raw_output_bytes": Json::Null,
            "stable_error_class": Json::Null,
            "created_at_unix_ms": created_at_unix_ms,
        }))?;
        connection
            .execute(
                "INSERT INTO judge_attempt_state_events (
                    judge_attempt_state_event_id, judge_attempt_id,
                    process_instance_id, dead_process_instance_id, state,
                    parse_result, safe_output_json, raw_output_sha256,
                    raw_output_bytes, stable_error_class, created_at_unix_ms,
                    canonical_payload_hash
                 ) VALUES (?1, ?2, ?3, ?4, 'orphaned_in_flight',
                           NULL, NULL, NULL, NULL, NULL, ?5, ?6)",
                params![
                    event_id.to_string(),
                    attempt_id.to_string(),
                    reconciler_process_instance_id.to_string(),
                    dead_process_instance_id.to_string(),
                    created_at_unix_ms,
                    payload_hash,
                ],
            )
            .map_err(database_error)?;
        applied = applied
            .checked_add(1)
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    }
    validate_reconciled_judge_graph(connection, project_uuid, dead_process_instance_id)?;
    Ok(applied)
}

fn validate_reconciled_judge_graph(
    connection: &Connection,
    project_uuid: Uuid,
    dead_process_instance_id: Uuid,
) -> Result<(), LedgerError> {
    let mut statement = connection
        .prepare(
            "SELECT judge_attempt_id FROM judge_attempts
             WHERE process_instance_id = ?1
             ORDER BY shadow_attempt_id, evaluator_version, attempt_ordinal",
        )
        .map_err(database_error)?;
    let attempt_ids = statement
        .query_map(params![dead_process_instance_id.to_string()], |row| {
            row.get::<_, String>(0)
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);

    let mut groups: BTreeMap<(Uuid, String), Vec<JudgeAttemptParent>> = BTreeMap::new();
    for attempt_id in attempt_ids {
        let attempt_id = parse_uuid(&attempt_id)?;
        let parent = load_judge_attempt_parent(connection, attempt_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if parent.process_instance_id != dead_process_instance_id.to_string()
            || !judge_attempt_parent_is_canonical(connection, &parent)?
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let shadow_project = connection
            .query_row(
                "SELECT project_uuid FROM shadow_attempts WHERE shadow_attempt_id = ?1",
                params![parent.shadow_attempt_id.to_string()],
                |row| row.get::<_, String>(0),
            )
            .optional()
            .map_err(database_error)?;
        if shadow_project.as_deref() != Some(project_uuid.to_string().as_str()) {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let terminal = load_judge_state(connection, attempt_id, true)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if !canonical_reconciled_judge_terminal(
            connection,
            project_uuid,
            dead_process_instance_id,
            &parent,
            &terminal,
        )? {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        groups
            .entry((parent.shadow_attempt_id, parent.evaluator_version.clone()))
            .or_default()
            .push(parent);
    }

    for ((shadow_attempt_id, evaluator_version), parents) in &groups {
        if parents.is_empty()
            || parents.len() > 2
            || parents[0].attempt_ordinal != 0
            || (parents.len() == 2
                && (parents[1].attempt_ordinal != 1
                    || !judge_attempt_contracts_match(&parents[0], &parents[1])))
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        if parents.len() == 2 {
            let initial_terminal = load_judge_state(connection, parents[0].judge_attempt_id, true)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
            let policy = verified_shadow_judge_policy(connection, *shadow_attempt_id)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
            if !canonical_parse_terminal(&parents[0], &initial_terminal, "invalid")?
                || !canonical_invalid_terminal_evidence(
                    &initial_terminal,
                    policy.config.max_rationale_bytes,
                )?
                || parents[1].created_at_unix_ms < initial_terminal.created_at_unix_ms
            {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }

        let final_parent = parents
            .last()
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let final_terminal = load_judge_state(connection, final_parent.judge_attempt_id, true)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let evaluation_id =
            evaluation_id_for_identity(connection, *shadow_attempt_id, evaluator_version)?;
        match (final_terminal.state.as_str(), evaluation_id) {
            ("valid", Some(evaluation_id))
                if verified_judge_evaluation_provenance(
                    connection,
                    evaluation_id,
                    *shadow_attempt_id,
                )? => {}
            ("valid", _) | (_, Some(_)) => {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
            (_, None) => {}
        }
    }

    let mut statement = connection
        .prepare(
            "SELECT e.evaluation_id, e.shadow_attempt_id
             FROM evaluations AS e
             JOIN shadow_attempts AS s ON s.shadow_attempt_id = e.shadow_attempt_id
             WHERE s.process_instance_id = ?1 AND e.source = 'judge'
             ORDER BY e.evaluation_id",
        )
        .map_err(database_error)?;
    let evaluations = statement
        .query_map(params![dead_process_instance_id.to_string()], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    for (evaluation_id, shadow_attempt_id) in evaluations {
        let evaluation_id = parse_uuid(&evaluation_id)?;
        let shadow_attempt_id = parse_uuid(&shadow_attempt_id)?;
        if !verified_judge_evaluation_provenance(connection, evaluation_id, shadow_attempt_id)? {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
    }
    Ok(())
}

/// Verify every Judge attempt and evaluation owned by one terminal Shadow attempt.
pub(super) fn verify_terminal_judge_graph_for_retention(
    connection: &Connection,
    project_uuid: Uuid,
    shadow_attempt_id: Uuid,
) -> Result<(), LedgerError> {
    let shadow = verified_shadow_attempt_context(connection, shadow_attempt_id)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    let stored_project_uuid = connection
        .query_row(
            "SELECT project_uuid FROM shadow_attempts WHERE shadow_attempt_id = ?1",
            params![shadow_attempt_id.to_string()],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .map_err(database_error)?
        .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    if stored_project_uuid != project_uuid.to_string() {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }

    let mut statement = connection
        .prepare(
            "SELECT judge_attempt_id FROM judge_attempts
             WHERE shadow_attempt_id = ?1
             ORDER BY evaluator_version, attempt_ordinal, judge_attempt_id",
        )
        .map_err(database_error)?;
    let attempt_ids = statement
        .query_map(params![shadow_attempt_id.to_string()], |row| {
            row.get::<_, String>(0)
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);

    let mut groups: BTreeMap<String, Vec<JudgeAttemptParent>> = BTreeMap::new();
    for attempt_id in attempt_ids {
        let attempt_id = parse_uuid(&attempt_id)?;
        let parent = load_judge_attempt_parent(connection, attempt_id)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if parent.shadow_attempt_id != shadow_attempt_id
            || parent.process_instance_id != shadow.process_instance_id.to_string()
            || !judge_attempt_parent_is_canonical(connection, &parent)?
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let terminal = load_judge_state(connection, attempt_id, true)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        if !canonical_reconciled_judge_terminal(
            connection,
            project_uuid,
            shadow.process_instance_id,
            &parent,
            &terminal,
        )? {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        groups
            .entry(parent.evaluator_version.clone())
            .or_default()
            .push(parent);
    }

    for (evaluator_version, parents) in &groups {
        if parents.is_empty()
            || parents.len() > 2
            || parents[0].attempt_ordinal != 0
            || (parents.len() == 2
                && (parents[1].attempt_ordinal != 1
                    || !judge_attempt_contracts_match(&parents[0], &parents[1])))
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        if parents.len() == 2 {
            let initial_terminal = load_judge_state(connection, parents[0].judge_attempt_id, true)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
            let policy = verified_shadow_judge_policy(connection, shadow_attempt_id)?
                .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
            if !canonical_parse_terminal(&parents[0], &initial_terminal, "invalid")?
                || !canonical_invalid_terminal_evidence(
                    &initial_terminal,
                    policy.config.max_rationale_bytes,
                )?
                || parents[1].created_at_unix_ms < initial_terminal.created_at_unix_ms
            {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
        }

        let final_parent = parents
            .last()
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let final_terminal = load_judge_state(connection, final_parent.judge_attempt_id, true)?
            .ok_or_else(|| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
        let evaluation_id =
            evaluation_id_for_identity(connection, shadow_attempt_id, evaluator_version)?;
        match (final_terminal.state.as_str(), evaluation_id) {
            ("valid", Some(evaluation_id))
                if verified_judge_evaluation_provenance(
                    connection,
                    evaluation_id,
                    shadow_attempt_id,
                )? => {}
            ("valid", _) | (_, Some(_)) => {
                return Err(LedgerErrorClass::IdentityInvariant.into());
            }
            (_, None) => {}
        }
    }

    let mut statement = connection
        .prepare(
            "SELECT evaluation_id FROM evaluations
             WHERE shadow_attempt_id = ?1 AND source = 'judge'
             ORDER BY evaluation_id",
        )
        .map_err(database_error)?;
    let evaluation_ids = statement
        .query_map(params![shadow_attempt_id.to_string()], |row| {
            row.get::<_, String>(0)
        })
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    drop(statement);
    for evaluation_id in evaluation_ids {
        if !verified_judge_evaluation_provenance(
            connection,
            parse_uuid(&evaluation_id)?,
            shadow_attempt_id,
        )? {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
    }
    Ok(())
}

fn canonical_reconciled_judge_terminal(
    connection: &Connection,
    project_uuid: Uuid,
    dead_process_instance_id: Uuid,
    parent: &JudgeAttemptParent,
    terminal: &StoredJudgeState,
) -> Result<bool, LedgerError> {
    if terminal.created_at_unix_ms < parent.created_at_unix_ms
        || stored_judge_state_hash(parent.judge_attempt_id, terminal)?
            != terminal.canonical_payload_hash
    {
        return Ok(false);
    }
    match terminal.state.as_str() {
        "orphaned_in_flight" => {
            let actor_is_same_project: bool = connection
                .query_row(
                    "SELECT EXISTS(
                        SELECT 1 FROM process_instances
                        WHERE process_instance_id = ?1 AND project_uuid = ?2
                     )",
                    params![
                        terminal.process_instance_id.to_string(),
                        project_uuid.to_string()
                    ],
                    |row| row.get(0),
                )
                .map_err(database_error)?;
            Ok(terminal.process_instance_id != dead_process_instance_id
                && actor_is_same_project
                && terminal.dead_process_instance_id == Some(dead_process_instance_id)
                && terminal.parse_result.is_none()
                && terminal.safe_output_json.is_none()
                && terminal.raw_output_sha256.is_none()
                && terminal.raw_output_bytes.is_none()
                && terminal.stable_error_class.is_none())
        }
        "valid" => canonical_parse_terminal(parent, terminal, "valid"),
        "invalid" => {
            let Some(policy) = verified_shadow_judge_policy(connection, parent.shadow_attempt_id)?
            else {
                return Ok(false);
            };
            Ok(canonical_parse_terminal(parent, terminal, "invalid")?
                && canonical_invalid_terminal_evidence(
                    terminal,
                    policy.config.max_rationale_bytes,
                )?)
        }
        "transport_failure" => Ok(terminal.process_instance_id == dead_process_instance_id
            && terminal.dead_process_instance_id.is_none()
            && terminal.parse_result.as_deref() == Some("operational")
            && terminal.safe_output_json.is_none()
            && terminal.raw_output_sha256.is_none()
            && terminal.raw_output_bytes.is_none()
            && terminal
                .stable_error_class
                .as_deref()
                .and_then(JudgeTransportFailureClass::from_str)
                .is_some()),
        "canceled_shutdown" => Ok(terminal.process_instance_id == dead_process_instance_id
            && terminal.dead_process_instance_id.is_none()
            && terminal.parse_result.is_none()
            && terminal.safe_output_json.is_none()
            && terminal.raw_output_sha256.is_none()
            && terminal.raw_output_bytes.is_none()
            && terminal.stable_error_class.is_none()),
        _ => Ok(false),
    }
}

pub(super) fn verified_judge_evaluation_provenance(
    connection: &Connection,
    evaluation_id: Uuid,
    shadow_attempt_id: Uuid,
) -> Result<bool, LedgerError> {
    let evaluation = connection
        .query_row(
            "SELECT evaluator_version, source, is_partial, created_at_unix_ms
             FROM evaluations
             WHERE evaluation_id = ?1 AND shadow_attempt_id = ?2",
            params![evaluation_id.to_string(), shadow_attempt_id.to_string()],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )
        .optional()
        .map_err(database_error)?;
    let Some((evaluator_version, source, is_partial, created_at_unix_ms)) = evaluation else {
        return Ok(false);
    };
    let Some(policy) = verified_shadow_judge_policy(connection, shadow_attempt_id)? else {
        return Ok(false);
    };
    let Some(shadow) = verified_shadow_attempt_context(connection, shadow_attempt_id)? else {
        return Ok(false);
    };
    if source != "judge"
        || !matches!(is_partial, 0 | 1)
        || evaluator_version != policy.evaluator_version
        || shadow.is_partial != (is_partial != 0)
        || created_at_unix_ms < shadow.started_at_unix_ms
    {
        return Ok(false);
    }

    let mut statement = connection
        .prepare(
            "SELECT judge_attempt_id FROM judge_attempts
             WHERE shadow_attempt_id = ?1 AND evaluator_version = ?2
             ORDER BY attempt_ordinal",
        )
        .map_err(database_error)?;
    let attempt_ids = statement
        .query_map(
            params![shadow_attempt_id.to_string(), evaluator_version],
            |row| row.get::<_, String>(0),
        )
        .map_err(database_error)?
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(database_error)?;
    if !(1..=2).contains(&attempt_ids.len()) {
        return Ok(false);
    }
    let mut parents = Vec::with_capacity(attempt_ids.len());
    for attempt_id in attempt_ids {
        let Ok(attempt_id) = parse_uuid(&attempt_id) else {
            return Ok(false);
        };
        let Some(parent) = load_judge_attempt_parent(connection, attempt_id)? else {
            return Ok(false);
        };
        if parent.shadow_attempt_id != shadow_attempt_id
            || parent.evaluator_version != evaluator_version
            || !judge_attempt_parent_is_canonical(connection, &parent)?
            || !judge_parent_matches_policy(&parent, &policy)
        {
            return Ok(false);
        }
        parents.push(parent);
    }
    if parents[0].attempt_ordinal != 0
        || (parents.len() == 2
            && (parents[1].attempt_ordinal != 1
                || !judge_attempt_contracts_match(&parents[0], &parents[1])))
    {
        return Ok(false);
    }
    if parents.len() == 2 {
        let Some(initial_terminal) =
            load_judge_state(connection, parents[0].judge_attempt_id, true)?
        else {
            return Ok(false);
        };
        if !canonical_parse_terminal(&parents[0], &initial_terminal, "invalid")? {
            return Ok(false);
        }
        if !canonical_invalid_terminal_evidence(
            &initial_terminal,
            policy.config.max_rationale_bytes,
        )? {
            return Ok(false);
        }
    }
    let final_parent = parents.last().expect("attempt list is nonempty");
    let Some(final_terminal) = load_judge_state(connection, final_parent.judge_attempt_id, true)?
    else {
        return Ok(false);
    };
    if !canonical_parse_terminal(final_parent, &final_terminal, "valid")?
        || final_terminal.created_at_unix_ms != created_at_unix_ms
    {
        return Ok(false);
    }
    let Some(safe_output_json) = final_terminal.safe_output_json.as_deref() else {
        return Ok(false);
    };
    let JudgeAttemptOutcomeV1::Valid(result) =
        validate_pairwise_judge_output(safe_output_json, policy.config.max_rationale_bytes)
    else {
        return Ok(false);
    };
    let Ok(result_value) = serde_json::to_value(&result) else {
        return Ok(false);
    };
    if canonical_json(&result_value).ok().as_deref() != Some(safe_output_json) {
        return Ok(false);
    }
    let Ok(reconstructed) = EvaluationRecord::from_judge_result(
        evaluation_id,
        shadow_attempt_id,
        evaluator_version,
        final_parent.judge_attempt_id,
        &result,
        &policy.config,
        is_partial != 0,
        created_at_unix_ms,
    ) else {
        return Ok(false);
    };
    evaluation_matches(connection, &reconstructed)
}

fn judge_parent_matches_policy(
    parent: &JudgeAttemptParent,
    policy: &VerifiedShadowJudgePolicy,
) -> bool {
    parent.judge_model == policy.config.model
        && parent.judge_model_revision == policy.config.model_revision
        && parent.prompt_version == policy.config.prompt_version
        && parent.prompt_sha256 == policy.prompt_sha256
        && parent.rubric_version == policy.config.rubric_version
        && parent.rubric_sha256 == policy.rubric_sha256
        && parent.output_schema_version == policy.config.output_schema_version
        && parent.output_schema_sha256 == policy.output_schema_sha256
}

fn judge_attempt_contracts_match(left: &JudgeAttemptParent, right: &JudgeAttemptParent) -> bool {
    left.process_instance_id == right.process_instance_id
        && left.learning_generation_id == right.learning_generation_id
        && left.shadow_attempt_id == right.shadow_attempt_id
        && left.evaluator_version == right.evaluator_version
        && left.judge_model == right.judge_model
        && left.judge_model_revision == right.judge_model_revision
        && left.prompt_version == right.prompt_version
        && left.prompt_sha256 == right.prompt_sha256
        && left.rubric_version == right.rubric_version
        && left.rubric_sha256 == right.rubric_sha256
        && left.output_schema_version == right.output_schema_version
        && left.output_schema_sha256 == right.output_schema_sha256
        && left.judge_input_sha256 == right.judge_input_sha256
        && left.candidate_response_json == right.candidate_response_json
        && left.candidate_response_fingerprint == right.candidate_response_fingerprint
}

fn canonical_parse_terminal(
    parent: &JudgeAttemptParent,
    terminal: &StoredJudgeState,
    expected_state: &str,
) -> Result<bool, LedgerError> {
    if terminal.created_at_unix_ms < parent.created_at_unix_ms
        || terminal.process_instance_id.to_string() != parent.process_instance_id
        || terminal.dead_process_instance_id.is_some()
        || terminal.state != expected_state
        || terminal.parse_result.as_deref() != Some(expected_state)
        || terminal
            .raw_output_sha256
            .as_deref()
            .is_none_or(|value| validate_sha256(value).is_err())
        || terminal.raw_output_bytes.is_none_or(|value| value < 0)
        || terminal.stable_error_class.is_some()
        || stored_judge_state_hash(parent.judge_attempt_id, terminal)?
            != terminal.canonical_payload_hash
    {
        return Ok(false);
    }
    let Some(safe_output_json) = terminal.safe_output_json.as_deref() else {
        return Ok(false);
    };
    let Ok(value): Result<Json, _> = serde_json::from_str(safe_output_json) else {
        return Ok(false);
    };
    Ok(canonical_json(&value).ok().as_deref() == Some(safe_output_json))
}

fn canonical_invalid_terminal_evidence(
    terminal: &StoredJudgeState,
    max_rationale_bytes: usize,
) -> Result<bool, LedgerError> {
    let (Some(safe_output_json), Some(raw_output_sha256), Some(raw_output_bytes)) = (
        terminal.safe_output_json.as_deref(),
        terminal.raw_output_sha256.as_deref(),
        terminal.raw_output_bytes,
    ) else {
        return Ok(false);
    };
    let Ok(raw_output_bytes) = u64::try_from(raw_output_bytes) else {
        return Ok(false);
    };
    let Ok(value): Result<Json, _> = serde_json::from_str(safe_output_json) else {
        return Ok(false);
    };
    if canonical_json(&value).ok().as_deref() != Some(safe_output_json) {
        return Ok(false);
    }
    let Some(object) = value.as_object().filter(|object| object.len() == 2) else {
        return Ok(false);
    };
    let Some(output) = object
        .get("output")
        .and_then(Json::as_object)
        .filter(|output| output.len() == 4)
    else {
        return Ok(false);
    };
    let Some(issues) = object.get("issues").and_then(Json::as_array) else {
        return Ok(false);
    };
    let issue_values = issues.iter().map(Json::as_str).collect::<Option<Vec<_>>>();
    let Some(issue_values) = issue_values else {
        return Ok(false);
    };
    if issue_values.is_empty()
        || issue_values.len() > 16
        || issue_values.windows(2).any(|pair| pair[0] >= pair[1])
        || issue_values
            .iter()
            .any(|issue| !stable_judge_validation_issue(issue))
    {
        return Ok(false);
    }
    let embedded_hash = output.get("sha256").and_then(Json::as_str);
    let embedded_bytes = output.get("byte_count").and_then(Json::as_u64);
    if embedded_hash != Some(raw_output_sha256)
        || embedded_bytes != Some(raw_output_bytes)
        || validate_sha256(raw_output_sha256).is_err()
    {
        return Ok(false);
    }
    let Some(raw_output_bound) = max_rationale_bytes.checked_add(8 * 1024) else {
        return Ok(false);
    };
    match output.get("retention").and_then(Json::as_str) {
        Some("retained") => {
            let Some(raw) = output.get("output").and_then(Json::as_str) else {
                return Ok(false);
            };
            if output.contains_key("marker")
                || u64::try_from(raw.len()).ok() != Some(raw_output_bytes)
                || sha256_hex(raw.as_bytes()) != raw_output_sha256
            {
                return Ok(false);
            }
            let JudgeAttemptOutcomeV1::Invalid(revalidated) =
                validate_pairwise_judge_output(raw, max_rationale_bytes)
            else {
                return Ok(false);
            };
            let revalidated = serde_json::to_value(&revalidated)
                .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
            Ok(canonical_json(&revalidated).ok().as_deref() == Some(safe_output_json))
        }
        Some("redacted") => {
            if output.contains_key("output") {
                return Ok(false);
            }
            match output.get("marker").and_then(Json::as_str) {
                Some("oversized") => Ok(usize::try_from(raw_output_bytes)
                    .is_ok_and(|bytes| bytes > raw_output_bound)
                    && issue_values == ["output.too_large"]),
                Some("credential_bearing") => Ok(usize::try_from(raw_output_bytes)
                    .is_ok_and(|bytes| bytes <= raw_output_bound)
                    && issue_values.contains(&"output.credential_bearing")),
                _ => Ok(false),
            }
        }
        _ => Ok(false),
    }
}

fn stable_judge_validation_issue(issue: &str) -> bool {
    const FIXED: [&str; 12] = [
        "field.hard_failures.items.too_many",
        "field.hard_failures.type.array_required",
        "field.rationale.text.control_free_required",
        "field.rationale.text.nfc_required",
        "field.rationale.text.nonempty_required",
        "field.rationale.text.too_large",
        "field.rationale.type.string_required",
        "issues.truncated",
        "output.credential_bearing",
        "output.invalid_json",
        "output.too_large",
        "output.type.object_required",
    ];
    if issue.is_empty() || issue.len() > 256 || FIXED.contains(&issue) {
        return !issue.is_empty() && issue.len() <= 256;
    }
    for field in [
        "response_equivalence",
        "trajectory_equivalence",
        "judge_confidence",
        "hard_failures",
        "rationale",
    ] {
        if ["missing", "duplicate"]
            .into_iter()
            .any(|suffix| issue == format!("field.{field}.{suffix}"))
        {
            return true;
        }
    }
    for field in [
        "response_equivalence",
        "trajectory_equivalence",
        "judge_confidence",
    ] {
        if [
            "type.number_required",
            "number.finite_required",
            "number.range_0_1_required",
        ]
        .into_iter()
        .any(|suffix| issue == format!("field.{field}.{suffix}"))
        {
            return true;
        }
    }
    if ["field.unknown.sha256.", "field.unknown_duplicate.sha256."]
        .into_iter()
        .any(|prefix| issue.strip_prefix(prefix).is_some_and(is_sha256))
        || issue.strip_prefix("issue.sha256.").is_some_and(is_sha256)
    {
        return true;
    }
    let Some(indexed) = issue.strip_prefix("field.hard_failures.item.") else {
        return false;
    };
    let Some((index, suffix)) = indexed.split_once('.') else {
        return false;
    };
    index.len() == 5
        && index.bytes().all(|byte| byte.is_ascii_digit())
        && matches!(
            suffix,
            "type.string_required" | "value.unknown" | "value.duplicate"
        )
}

fn is_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || matches!(byte, b'a'..=b'f'))
}

fn terminal_evaluation_matches_parent(
    terminal: &JudgeAttemptTerminal,
    parent: &JudgeAttemptParent,
) -> bool {
    match (&terminal.state, terminal.evaluation.as_ref()) {
        (JudgeTerminalState::Valid { .. }, Some(evaluation)) => {
            evaluation.shadow_attempt_id == parent.shadow_attempt_id
                && evaluation.evaluator_version == parent.evaluator_version
                && evaluation.created_at_unix_ms == terminal.created_at_unix_ms
                && evaluation.evaluation.source == JudgeEvaluationSourceV1::Judge
                && evaluation.judge_contract.as_ref().is_some_and(|contract| {
                    contract.judge_model == parent.judge_model
                        && contract.judge_model_revision == parent.judge_model_revision
                        && contract.prompt_version == parent.prompt_version
                        && contract.prompt_sha256 == parent.prompt_sha256
                        && contract.rubric_version == parent.rubric_version
                        && contract.rubric_sha256 == parent.rubric_sha256
                        && contract.output_schema_version == parent.output_schema_version
                        && contract.output_schema_sha256 == parent.output_schema_sha256
                })
        }
        (JudgeTerminalState::Valid { .. }, None) => false,
        (
            JudgeTerminalState::Invalid { .. }
            | JudgeTerminalState::TransportFailure { .. }
            | JudgeTerminalState::CanceledShutdown,
            None,
        ) => true,
        (
            JudgeTerminalState::Invalid { .. }
            | JudgeTerminalState::TransportFailure { .. }
            | JudgeTerminalState::CanceledShutdown,
            Some(_),
        ) => false,
    }
}

/// Frozen final quality record with exact Judge contract identity when applicable.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct EvaluationRecord {
    pub(crate) evaluation_id: Uuid,
    pub(crate) shadow_attempt_id: Uuid,
    pub(crate) evaluator_version: String,
    pub(crate) conflict_health_event_id: Uuid,
    evaluation: PreparedEvaluation,
    judge_contract: Option<PreparedJudgeContract>,
    judge_config_sha256: String,
    pub(crate) created_at_unix_ms: i64,
}

#[derive(Clone, PartialEq, Eq)]
struct PreparedJudgeContract {
    judge_model: String,
    judge_model_revision: String,
    prompt_version: String,
    prompt_sha256: String,
    rubric_version: String,
    rubric_sha256: String,
    output_schema_version: u32,
    output_schema_sha256: String,
}

#[derive(Clone, PartialEq, Eq)]
struct PreparedEvaluation {
    source: JudgeEvaluationSourceV1,
    response_equivalence: Option<PreparedScore>,
    trajectory_equivalence: Option<PreparedScore>,
    judge_confidence: Option<PreparedScore>,
    response_weight: Option<PreparedScore>,
    trajectory_weight: Option<PreparedScore>,
    aggregate_score: Option<PreparedScore>,
    label: JudgeLabelV1,
    binary_label: Option<JudgeBinaryLabelV1>,
    hard_failures: Vec<PreparedHardFailure>,
    rationale: Option<String>,
    is_partial: bool,
    promotion_eligible: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PreparedScore {
    bits: u64,
}

impl PreparedScore {
    fn component(value: f64, bits: u64) -> Result<Self, LedgerError> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) || value.to_bits() != bits {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self { bits })
    }

    fn aggregate(value: f64, bits: u64) -> Result<Self, LedgerError> {
        if !value.is_finite() || !(-0.0..=1.000_000_001).contains(&value) || value.to_bits() != bits
        {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        Ok(Self { bits })
    }

    fn value(self) -> f64 {
        f64::from_bits(self.bits)
    }

    fn sqlite_bits(self) -> i64 {
        self.bits as i64
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PreparedHardFailure {
    value: &'static str,
    ordinal: i64,
}

impl EvaluationRecord {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn deterministic(
        evaluation_id: Uuid,
        shadow_attempt_id: Uuid,
        evaluator_version: impl Into<String>,
        conflict_health_event_id: Uuid,
        hard_failure: DeterministicHardFailureV1,
        is_partial: bool,
        judge: &JudgeConfig,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        let evaluation = JudgeEvaluationV1::deterministic_hard_failure(hard_failure, is_partial);
        Self::new_checked(
            evaluation_id,
            shadow_attempt_id,
            evaluator_version,
            conflict_health_event_id,
            &evaluation,
            judge,
            created_at_unix_ms,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_judge_result(
        evaluation_id: Uuid,
        shadow_attempt_id: Uuid,
        evaluator_version: impl Into<String>,
        conflict_health_event_id: Uuid,
        result: &PairwiseJudgeResultV1,
        judge: &JudgeConfig,
        is_partial: bool,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        let evaluation = JudgeEvaluationV1::from_judge_result(judge, result, is_partial);
        Self::new_checked(
            evaluation_id,
            shadow_attempt_id,
            evaluator_version,
            conflict_health_event_id,
            &evaluation,
            judge,
            created_at_unix_ms,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_checked(
        evaluation_id: Uuid,
        shadow_attempt_id: Uuid,
        evaluator_version: impl Into<String>,
        conflict_health_event_id: Uuid,
        evaluation: &JudgeEvaluationV1,
        judge: &JudgeConfig,
        created_at_unix_ms: i64,
    ) -> Result<Self, LedgerError> {
        for id in [evaluation_id, shadow_attempt_id, conflict_health_event_id] {
            validate_uuid_v7(id)?;
        }
        if evaluation_id == conflict_health_event_id || created_at_unix_ms < 0 {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let evaluator_version = evaluator_version.into();
        validate_sha256(&evaluator_version)?;
        if !evaluation.is_consistent_with_config(judge) {
            return Err(LedgerErrorClass::IdentityInvariant.into());
        }
        let prepared = prepare_evaluation(evaluation)?;
        let judge_config_sha256 = judge
            .contract_sha256()
            .map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))?;
        let judge_contract = match evaluation.source {
            JudgeEvaluationSourceV1::DeterministicValidator => None,
            JudgeEvaluationSourceV1::Judge => Some(PreparedJudgeContract {
                judge_model: judge.model.clone(),
                judge_model_revision: judge.model_revision.clone(),
                prompt_version: judge.prompt_version.clone(),
                prompt_sha256: JUDGE_PROMPT_TEMPLATE_SHA256_V1.to_string(),
                rubric_version: judge.rubric_version.clone(),
                rubric_sha256: JUDGE_RUBRIC_TEMPLATE_SHA256_V1.to_string(),
                output_schema_version: judge.output_schema_version,
                output_schema_sha256: JUDGE_OUTPUT_SCHEMA_SHA256_V1.to_string(),
            }),
        };
        Ok(Self {
            evaluation_id,
            shadow_attempt_id,
            evaluator_version,
            conflict_health_event_id,
            evaluation: prepared,
            judge_contract,
            judge_config_sha256,
            created_at_unix_ms,
        })
    }
}

/// Exhaustive acknowledgement for Judge/evaluation writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JudgeRecordAck {
    Applied,
    AlreadyApplied,
    Conflict,
    OriginatingProcessNotLive,
    TransactionNotStarted,
}

struct PreparedTerminalFields<'a> {
    state: &'static str,
    parse_result: Option<&'static str>,
    safe_output_json: Option<&'a str>,
    raw_output_sha256: Option<&'a str>,
    raw_output_bytes: Option<i64>,
    stable_error_class: Option<&'a str>,
}

impl JudgeAttemptTerminal {
    fn fields(&self) -> PreparedTerminalFields<'_> {
        match &self.state {
            JudgeTerminalState::Valid {
                safe_output_json,
                raw_output_sha256,
                raw_output_bytes,
            } => PreparedTerminalFields {
                state: "valid",
                parse_result: Some("valid"),
                safe_output_json: Some(safe_output_json),
                raw_output_sha256: Some(raw_output_sha256),
                raw_output_bytes: Some(*raw_output_bytes),
                stable_error_class: None,
            },
            JudgeTerminalState::Invalid {
                safe_output_json,
                raw_output_sha256,
                raw_output_bytes,
            } => PreparedTerminalFields {
                state: "invalid",
                parse_result: Some("invalid"),
                safe_output_json: Some(safe_output_json),
                raw_output_sha256: Some(raw_output_sha256),
                raw_output_bytes: Some(*raw_output_bytes),
                stable_error_class: None,
            },
            JudgeTerminalState::TransportFailure { stable_error_class } => PreparedTerminalFields {
                state: "transport_failure",
                parse_result: Some("operational"),
                safe_output_json: None,
                raw_output_sha256: None,
                raw_output_bytes: None,
                stable_error_class: Some(stable_error_class),
            },
            JudgeTerminalState::CanceledShutdown => PreparedTerminalFields {
                state: "canceled_shutdown",
                parse_result: None,
                safe_output_json: None,
                raw_output_sha256: None,
                raw_output_bytes: None,
                stable_error_class: None,
            },
        }
    }
}

fn prepare_evaluation(value: &JudgeEvaluationV1) -> Result<PreparedEvaluation, LedgerError> {
    let component = |value: Option<crate::judge::ScoredValueV1>| {
        value
            .map(|value| PreparedScore::component(value.value, value.bits))
            .transpose()
    };
    let aggregate = value
        .aggregate
        .map(|value| PreparedScore::aggregate(value.value, value.bits))
        .transpose()?;
    let hard_failures = value
        .hard_failures
        .iter()
        .copied()
        .map(prepare_hard_failure)
        .collect::<Vec<_>>();
    if hard_failures
        .windows(2)
        .any(|pair| pair[0].ordinal >= pair[1].ordinal)
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(PreparedEvaluation {
        source: value.source,
        response_equivalence: component(value.response_equivalence)?,
        trajectory_equivalence: component(value.trajectory_equivalence)?,
        judge_confidence: component(value.judge_confidence)?,
        response_weight: component(value.response_weight)?,
        trajectory_weight: component(value.trajectory_weight)?,
        aggregate_score: aggregate,
        label: value.label,
        binary_label: value.binary_label,
        hard_failures,
        rationale: value.rationale.clone(),
        is_partial: value.is_partial,
        promotion_eligible: value.promotion_eligible,
    })
}

fn prepare_hard_failure(value: JudgeHardFailureV1) -> PreparedHardFailure {
    match value {
        JudgeHardFailureV1::ToolContract => PreparedHardFailure {
            value: "tool_contract",
            ordinal: 0,
        },
        JudgeHardFailureV1::ResponseSchema => PreparedHardFailure {
            value: "response_schema",
            ordinal: 1,
        },
        JudgeHardFailureV1::Safety => PreparedHardFailure {
            value: "safety",
            ordinal: 2,
        },
        JudgeHardFailureV1::MalformedCandidate => PreparedHardFailure {
            value: "malformed_candidate",
            ordinal: 3,
        },
    }
}

fn validate_terminal_ids(
    judge_attempt_id: Uuid,
    judge_attempt_state_event_id: Uuid,
    conflict_health_event_id: Uuid,
    created_at_unix_ms: i64,
) -> Result<(), LedgerError> {
    for id in [
        judge_attempt_id,
        judge_attempt_state_event_id,
        conflict_health_event_id,
    ] {
        validate_uuid_v7(id)?;
    }
    if judge_attempt_id == judge_attempt_state_event_id
        || judge_attempt_id == conflict_health_event_id
        || judge_attempt_state_event_id == conflict_health_event_id
        || created_at_unix_ms < 0
    {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(())
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

fn parse_uuid(value: &str) -> Result<Uuid, LedgerError> {
    let parsed = Uuid::parse_str(value)
        .map_err(|_| LedgerError::new(LedgerErrorClass::IdentityInvariant))?;
    validate_uuid_v7(parsed)?;
    if parsed.to_string() != value {
        return Err(LedgerErrorClass::IdentityInvariant.into());
    }
    Ok(parsed)
}

fn parse_sql_uuid(index: usize, value: &str) -> rusqlite::Result<Uuid> {
    let parsed = Uuid::parse_str(value).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(
            index,
            rusqlite::types::Type::Text,
            Box::new(error),
        )
    })?;
    if parsed.get_version_num() != 7 || parsed.to_string() != value {
        return Err(rusqlite::Error::FromSqlConversionFailure(
            index,
            rusqlite::types::Type::Text,
            "UUID is not a canonical version 7 identifier".into(),
        ));
    }
    Ok(parsed)
}

fn hash_json(value: &Json) -> Result<String, LedgerError> {
    canonical_sha256(value).map_err(|_| LedgerError::new(LedgerErrorClass::CanonicalizationFailed))
}

fn database_error(error: rusqlite::Error) -> LedgerError {
    map_sqlite_error(&error, LedgerErrorClass::DatabaseOperationFailed)
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Path;

    use chrono::{TimeZone, Utc};
    use nemo_relay::api::llm::LlmApiFamily;
    use nemo_relay::api::runtime::{LLM_REPLAY_CONTRACT_VERSION, LlmReplayCapability};
    use rusqlite::{Connection, params};
    use serde_json::json;
    use tempfile::tempdir;
    use uuid::Uuid;

    use super::{
        EvaluationRecord, JudgeAttemptStart, JudgeAttemptTerminal, JudgeRecordAck,
        JudgeTransportFailureClass,
    };
    use crate::canonical_json::{canonical_json, canonical_sha256};
    use crate::config::{JudgeConfig, RouterConfig};
    use crate::judge::{
        DeterministicHardFailureV1, JudgeHorizonV1, JudgePolicyIdentityV1, PairwiseJudgeInputV1,
    };
    use crate::ledger::model::LedgerRuntimeIdentity;
    use crate::ledger::repository::anchors::{
        AnchorCommandAck, FrozenPendingAnchorV1, FrozenTerminalAnchorV1,
    };
    use crate::ledger::repository::shadow::{
        ReservedShadowAttempt, SampleBatchReservation, ShadowAttemptStarted, ShadowCommandAck,
    };
    use crate::ledger::repository::tests::{assert_reconciliation_noop_twice, recovery_snapshot};
    use crate::ledger::repository::{ActivatedLedger, LedgerRepository};
    use crate::projection::{
        REQUEST_PROJECTION_SCHEMA_V1, ROUTER_SANITIZER_VERSION, RouterRequestProjectionV1,
        SanitizedAnnotatedLlmRequest,
    };
    use crate::trajectory::test_fixtures::pending_window;
    use crate::trajectory::{
        CANDIDATE_FACT_SCHEMA_V1, PersistedCandidateCapabilitiesV1, PersistedCandidateFactV1,
        PersistedTrajectoryTerminalV1, ReplayCapabilityFactsV1, RouterResponseProjectionV1,
        TrajectoryTrigger,
    };

    fn config(path: &Path) -> RouterConfig {
        serde_json::from_value(json!({
            "version": 1,
            "mode": "shadow",
            "project_id": "judge-repository-project",
            "database_path": path.to_string_lossy(),
            "retention_days": 30,
            "max_evidence_records": 1000,
            "pools": [{
                "id": "pool-a",
                "api_family": "openai_chat_completions",
                "anchor_models": ["anchor-a"],
                "anchor_revision": "2026-07-01",
                "sampling_probability": 0.25,
                "max_candidates_per_sample": 2,
                "concurrency": {"shadow": 2, "judge": 1, "max_pending": 2},
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

    fn activate() -> (tempfile::TempDir, RouterConfig, ActivatedLedger) {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let config = config(&temporary.path().join("ledger.db"));
        let activated = LedgerRepository::activate(&config).unwrap();
        (temporary, config, activated)
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
        let mut value = serde_json::to_value(&projection).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("semantic_request_fingerprint");
        projection.semantic_request_fingerprint = canonical_sha256(&value).unwrap();
        projection
    }

    fn candidate_request_projection() -> RouterRequestProjectionV1 {
        let mut projection = request_projection();
        projection.normalized_request.model = Some("candidate-model-a".to_string());
        projection.semantic_request_fingerprint.clear();
        let mut value = serde_json::to_value(&projection).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("semantic_request_fingerprint");
        projection.semantic_request_fingerprint = canonical_sha256(&value).unwrap();
        projection
    }

    fn response_projection(model: &str) -> RouterResponseProjectionV1 {
        let mut projection = pending_window(Uuid::now_v7()).normalized_anchor_response;
        projection.model = Some(model.to_string());
        projection.semantic_response_fingerprint.clear();
        let mut value = serde_json::to_value(&projection).unwrap();
        value
            .as_object_mut()
            .unwrap()
            .remove("semantic_response_fingerprint");
        projection.semantic_response_fingerprint = canonical_sha256(&value).unwrap();
        projection
    }

    fn judge_input(config: &RouterConfig) -> PairwiseJudgeInputV1 {
        PairwiseJudgeInputV1::new(
            &request_projection(),
            &response_projection("anchor-a"),
            &response_projection("candidate-model-a"),
            &[],
            JudgeHorizonV1::new(1, 1, TrajectoryTrigger::ProgressReached, false).unwrap(),
            JudgePolicyIdentityV1::from_config(judge(config)).unwrap(),
        )
        .unwrap()
    }

    fn seed_shadow_attempt(
        repository: &mut LedgerRepository,
        identity: &LedgerRuntimeIdentity,
        config: &RouterConfig,
        _suffix: &str,
    ) -> Uuid {
        let pool = identity.pools.get("pool-a").unwrap();
        let anchor_id = Uuid::now_v7();
        let mut pending = pending_window(anchor_id);
        pending.anchor_call_uuid = Uuid::now_v7();
        pending.root_uuid = Uuid::now_v7();
        pending.owner_uuid = Uuid::now_v7();
        pending.pool_id = "pool-a".to_string();
        pending.anchor_model_revision = "2026-07-01".to_string();
        pending.process_instance_id = identity.process_instance_id;
        pending.project_uuid = identity.project_uuid;
        pending.project_id = identity.project_id.clone();
        pending.config_generation_id = identity.config_generation_id.clone();
        pending.policy_version_id = pool.policy_version_id.clone();
        pending.learning_generation_id = pool.learning_generation_id;
        pending.request_projection = request_projection();
        pending.routing_context_projection.tenant_policy_hash = "2".repeat(64);
        pending.routing_context_projection.agent_policy_hash = "3".repeat(64);
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
        pending.normalized_anchor_response = response_projection("anchor-a");
        pending.opened_at = Utc.timestamp_millis_opt(0).unwrap();
        pending.deadline_at = Utc.timestamp_millis_opt(10).unwrap();
        let frozen_pending =
            FrozenPendingAnchorV1::new(&pending, Uuid::now_v7(), Uuid::now_v7(), 0).unwrap();
        assert!(matches!(
            repository.record_pending_anchor(&frozen_pending).unwrap(),
            AnchorCommandAck::Applied { .. }
        ));
        let terminal = PersistedTrajectoryTerminalV1::closed(
            pending,
            Vec::new(),
            1,
            TrajectoryTrigger::ProgressReached,
            Utc.timestamp_millis_opt(10).unwrap(),
            Vec::new(),
        );
        let frozen_terminal =
            FrozenTerminalAnchorV1::new(&terminal, Uuid::now_v7(), Uuid::now_v7(), 10).unwrap();
        assert!(matches!(
            repository.record_terminal_anchor(&frozen_terminal).unwrap(),
            AnchorCommandAck::Applied { .. }
        ));

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
            evaluator_version(config),
            "2".repeat(64),
            "3".repeat(64),
            true,
            candidate_request_projection(),
            10,
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
            10,
        )
        .unwrap();
        assert_eq!(
            repository.reserve_sample_batch(&reservation).unwrap(),
            ShadowCommandAck::Applied
        );
        let started = ShadowAttemptStarted::new(
            attempt.shadow_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            11,
        )
        .unwrap();
        assert_eq!(
            repository.start_shadow_attempt(started).unwrap(),
            ShadowCommandAck::Applied
        );
        attempt.shadow_attempt_id
    }

    fn judge(config: &RouterConfig) -> &JudgeConfig {
        &config.pools[0].judge
    }

    fn evaluator_version(config: &RouterConfig) -> String {
        judge(config).evaluator_version().unwrap()
    }

    fn start_record(
        identity: &LedgerRuntimeIdentity,
        config: &RouterConfig,
        shadow_attempt_id: Uuid,
        ordinal: u8,
    ) -> JudgeAttemptStart {
        JudgeAttemptStart::new(
            Uuid::now_v7(),
            shadow_attempt_id,
            identity.pools["pool-a"].learning_generation_id,
            evaluator_version(config),
            judge(config),
            &judge_input(config),
            ordinal,
            Uuid::now_v7(),
            Uuid::now_v7(),
            20 + 3 * i64::from(ordinal),
        )
        .unwrap()
    }

    #[test]
    fn startup_recovers_initial_and_repair_judge_crash_boundaries() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let config = config(&temporary.path().join("ledger.db"));
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();

        let started_shadow = seed_shadow_attempt(
            &mut origin.repository,
            &origin_identity,
            &config,
            "startup-started",
        );
        let started_initial = start_record(&origin_identity, &config, started_shadow, 0);
        assert_eq!(
            origin
                .repository
                .record_judge_attempt_start(&started_initial)
                .unwrap(),
            JudgeRecordAck::Applied
        );

        let invalid_shadow = seed_shadow_attempt(
            &mut origin.repository,
            &origin_identity,
            &config,
            "startup-invalid",
        );
        let invalid_initial = start_record(&origin_identity, &config, invalid_shadow, 0);
        origin
            .repository
            .record_judge_attempt_start(&invalid_initial)
            .unwrap();
        origin
            .repository
            .record_judge_attempt_terminal(
                &JudgeAttemptTerminal::invalid(
                    invalid_initial.judge_attempt_id,
                    Uuid::now_v7(),
                    Uuid::now_v7(),
                    "{}",
                    judge(&config),
                    22,
                )
                .unwrap(),
            )
            .unwrap();

        let repair_shadow = seed_shadow_attempt(
            &mut origin.repository,
            &origin_identity,
            &config,
            "startup-repair",
        );
        let repair_initial = start_record(&origin_identity, &config, repair_shadow, 0);
        origin
            .repository
            .record_judge_attempt_start(&repair_initial)
            .unwrap();
        origin
            .repository
            .record_judge_attempt_terminal(
                &JudgeAttemptTerminal::invalid(
                    repair_initial.judge_attempt_id,
                    Uuid::now_v7(),
                    Uuid::now_v7(),
                    "{}",
                    judge(&config),
                    22,
                )
                .unwrap(),
            )
            .unwrap();
        let repair = start_record(&origin_identity, &config, repair_shadow, 1);
        assert_eq!(
            origin
                .repository
                .record_judge_attempt_start(&repair)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        drop(origin);

        let mut recovered = LedgerRepository::activate_at(&config, 31_000).unwrap();
        let states = recovered
            .repository
            .connection
            .prepare(
                "SELECT j.judge_attempt_id, j.attempt_ordinal, s.state
                 FROM judge_attempts AS j
                 JOIN judge_attempt_state_events AS s
                   ON s.judge_attempt_id = j.judge_attempt_id AND s.state <> 'started'
                 ORDER BY j.shadow_attempt_id, j.attempt_ordinal",
            )
            .unwrap()
            .query_map([], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .unwrap()
            .collect::<rusqlite::Result<Vec<_>>>()
            .unwrap();
        assert_eq!(states.len(), 4);
        assert_eq!(
            states
                .iter()
                .filter(|(_, _, state)| state == "orphaned_in_flight")
                .count(),
            2
        );
        assert_eq!(
            states
                .iter()
                .filter(|(_, _, state)| state == "invalid")
                .count(),
            2
        );
        assert_reconciliation_noop_twice(&mut recovered.repository, 31_000);
    }

    #[test]
    fn startup_preserves_valid_judge_evaluation_before_shadow_terminal() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let config = config(&temporary.path().join("ledger.db"));
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut origin.repository,
            &origin_identity,
            &config,
            "startup-valid-before-shadow",
        );
        let start = start_record(&origin_identity, &config, shadow_attempt_id, 0);
        origin
            .repository
            .record_judge_attempt_start(&start)
            .unwrap();
        let terminal = JudgeAttemptTerminal::valid(
            start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            shadow_attempt_id,
            evaluator_version(&config),
            &judge_raw(false),
            judge(&config),
            Uuid::now_v7(),
            Uuid::now_v7(),
            false,
            22,
        )
        .unwrap();
        let evaluation_id = terminal.evaluation.as_ref().unwrap().evaluation_id;
        assert_eq!(
            origin
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        let evaluation_before: (String, String, Option<String>, i64) = origin
            .repository
            .connection
            .query_row(
                "SELECT e.canonical_payload_hash, e.label, e.binary_label,
                        (SELECT count(*) FROM evaluation_hard_failures AS f
                         WHERE f.evaluation_id = e.evaluation_id)
                 FROM evaluations AS e WHERE e.evaluation_id = ?1",
                params![evaluation_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        drop(origin);

        let mut recovered = LedgerRepository::activate_at(&config, 31_000).unwrap();
        assert!(
            super::verified_judge_evaluation_provenance(
                &recovered.repository.connection,
                evaluation_id,
                shadow_attempt_id,
            )
            .unwrap()
        );
        let evaluation_after: (String, String, Option<String>, i64) = recovered
            .repository
            .connection
            .query_row(
                "SELECT e.canonical_payload_hash, e.label, e.binary_label,
                        (SELECT count(*) FROM evaluation_hard_failures AS f
                         WHERE f.evaluation_id = e.evaluation_id)
                 FROM evaluations AS e WHERE e.evaluation_id = ?1",
                params![evaluation_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
            )
            .unwrap();
        assert_eq!(evaluation_after, evaluation_before);
        assert_eq!(
            recovered
                .repository
                .connection
                .query_row(
                    "SELECT count(*) FROM evaluations WHERE shadow_attempt_id = ?1",
                    params![shadow_attempt_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );
        let shadow_result: (String, Option<String>, String) = recovered
            .repository
            .connection
            .query_row(
                "SELECT r.terminal_class, r.evaluation_id, s.dead_process_instance_id
                 FROM shadow_results AS r
                 JOIN shadow_attempt_state_events AS s
                   ON s.shadow_attempt_id = r.shadow_attempt_id
                  AND s.state = 'orphaned_in_flight'
                 WHERE r.shadow_attempt_id = ?1",
                params![shadow_attempt_id.to_string()],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            shadow_result,
            (
                "orphaned_in_flight".to_string(),
                None,
                origin_identity.process_instance_id.to_string()
            )
        );
        assert_reconciliation_noop_twice(&mut recovered.repository, 31_000);
    }

    #[test]
    fn startup_rejects_future_judge_start_and_rolls_back_recovery() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger.db");
        let config = config(&path);
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut origin.repository,
            &origin_identity,
            &config,
            "future-start",
        );
        let mut start = start_record(&origin_identity, &config, shadow_attempt_id, 0);
        start.created_at_unix_ms = 100_000;
        origin
            .repository
            .record_judge_attempt_start(&start)
            .unwrap();
        let before = recovery_snapshot(&origin.repository.connection);
        drop(origin);

        let error = match LedgerRepository::activate_at(&config, 31_000) {
            Ok(_) => panic!("future Judge start must reject recovery"),
            Err(error) => error,
        };
        assert_eq!(
            error.class(),
            crate::ledger::model::LedgerErrorClass::IdentityInvariant
        );
        let connection = Connection::open(&path).unwrap();
        assert_eq!(recovery_snapshot(&connection), before);
    }

    #[test]
    fn startup_rolls_back_earlier_orphan_when_later_valid_evaluation_is_missing() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger.db");
        let config = config(&path);
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let earlier_shadow = seed_shadow_attempt(
            &mut origin.repository,
            &origin_identity,
            &config,
            "rollback-earlier",
        );
        let earlier_start = start_record(&origin_identity, &config, earlier_shadow, 0);
        origin
            .repository
            .record_judge_attempt_start(&earlier_start)
            .unwrap();

        let valid_shadow = seed_shadow_attempt(
            &mut origin.repository,
            &origin_identity,
            &config,
            "rollback-valid",
        );
        let valid_start = start_record(&origin_identity, &config, valid_shadow, 0);
        origin
            .repository
            .record_judge_attempt_start(&valid_start)
            .unwrap();
        let valid = JudgeAttemptTerminal::valid(
            valid_start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            valid_shadow,
            evaluator_version(&config),
            &judge_raw(false),
            judge(&config),
            Uuid::now_v7(),
            Uuid::now_v7(),
            false,
            22,
        )
        .unwrap();
        let evaluation_id = valid.evaluation.as_ref().unwrap().evaluation_id;
        origin
            .repository
            .record_judge_attempt_terminal(&valid)
            .unwrap();
        origin
            .repository
            .connection
            .execute(
                "DELETE FROM evaluations WHERE evaluation_id = ?1",
                params![evaluation_id.to_string()],
            )
            .unwrap();
        let before = recovery_snapshot(&origin.repository.connection);
        drop(origin);

        let error = match LedgerRepository::activate_at(&config, 31_000) {
            Ok(_) => panic!("missing valid Judge evaluation must reject recovery"),
            Err(error) => error,
        };
        assert_eq!(
            error.class(),
            crate::ledger::model::LedgerErrorClass::IdentityInvariant
        );
        let connection = Connection::open(&path).unwrap();
        assert_eq!(recovery_snapshot(&connection), before);
    }

    #[test]
    fn startup_rejects_rehashed_unknown_judge_transport_class() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let path = temporary.path().join("ledger.db");
        let config = config(&path);
        let mut origin = LedgerRepository::activate_at(&config, 1_000).unwrap();
        let origin_identity = origin.identity.clone();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut origin.repository,
            &origin_identity,
            &config,
            "unknown-transport",
        );
        let start = start_record(&origin_identity, &config, shadow_attempt_id, 0);
        origin
            .repository
            .record_judge_attempt_start(&start)
            .unwrap();
        let terminal = JudgeAttemptTerminal::transport_failure(
            start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            JudgeTransportFailureClass::Timeout,
            22,
        )
        .unwrap();
        origin
            .repository
            .record_judge_attempt_terminal(&terminal)
            .unwrap();
        let mut stored =
            super::load_judge_state(&origin.repository.connection, start.judge_attempt_id, true)
                .unwrap()
                .unwrap();
        stored.stable_error_class = Some("router.provider.fabricated".to_string());
        stored.canonical_payload_hash =
            super::stored_judge_state_hash(start.judge_attempt_id, &stored).unwrap();
        origin
            .repository
            .connection
            .execute(
                "UPDATE judge_attempt_state_events
             SET stable_error_class = ?1, canonical_payload_hash = ?2
             WHERE judge_attempt_id = ?3 AND state = 'transport_failure'",
                params![
                    stored.stable_error_class,
                    stored.canonical_payload_hash,
                    start.judge_attempt_id.to_string()
                ],
            )
            .unwrap();
        let before = recovery_snapshot(&origin.repository.connection);
        drop(origin);

        let error = match LedgerRepository::activate_at(&config, 31_000) {
            Ok(_) => panic!("unknown Judge transport class must reject recovery"),
            Err(error) => error,
        };
        assert_eq!(
            error.class(),
            crate::ledger::model::LedgerErrorClass::IdentityInvariant
        );
        let connection = Connection::open(&path).unwrap();
        assert_eq!(recovery_snapshot(&connection), before);
    }

    fn judge_raw(hard_failure: bool) -> String {
        judge_raw_with_scores(0.9, 0.9, hard_failure)
    }

    fn judge_raw_with_scores(
        response_equivalence: f64,
        trajectory_equivalence: f64,
        hard_failure: bool,
    ) -> String {
        json!({
            "response_equivalence": response_equivalence,
            "trajectory_equivalence": trajectory_equivalence,
            "judge_confidence": 0.9,
            "hard_failures": if hard_failure { vec!["safety"] } else { Vec::<&str>::new() },
            "rationale": "bounded rationale",
        })
        .to_string()
    }

    #[test]
    fn start_is_exactly_idempotent_and_alternate_identity_degrades_health() {
        let (_temporary, config, mut activated) = activate();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "start",
        );
        let start = start_record(&activated.identity, &config, shadow_attempt_id, 0);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&start)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&start)
                .unwrap(),
            JudgeRecordAck::AlreadyApplied
        );
        let alternate = start_record(&activated.identity, &config, shadow_attempt_id, 0);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&alternate)
                .unwrap(),
            JudgeRecordAck::Conflict
        );
        let counts: (i64, i64, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT COUNT(*) FROM judge_attempts),
                    (SELECT COUNT(*) FROM judge_attempt_state_events WHERE state = 'started'),
                    (SELECT COUNT(*) FROM health_events
                     WHERE stable_class = 'router.ledger.integrity_conflict')",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(counts, (1, 1, 1));
    }

    #[test]
    fn start_rejects_a_judge_input_hash_not_reconstructed_from_shadow_evidence() {
        let (_temporary, config, mut activated) = activate();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "input-binding",
        );
        let mut start = start_record(&activated.identity, &config, shadow_attempt_id, 0);
        start.judge_input_sha256 = "f".repeat(64);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&start)
                .unwrap(),
            JudgeRecordAck::Conflict
        );
        let attempt_count: i64 = activated
            .repository
            .connection
            .query_row("SELECT COUNT(*) FROM judge_attempts", [], |row| row.get(0))
            .unwrap();
        assert_eq!(attempt_count, 0);
    }

    #[test]
    fn valid_terminal_atomically_persists_evaluation_and_hard_failures() {
        let (_temporary, config, mut activated) = activate();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "valid",
        );
        let start = start_record(&activated.identity, &config, shadow_attempt_id, 0);
        activated
            .repository
            .record_judge_attempt_start(&start)
            .unwrap();
        let raw_output = judge_raw(true);
        let terminal = JudgeAttemptTerminal::valid(
            start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            shadow_attempt_id,
            evaluator_version(&config),
            &raw_output,
            judge(&config),
            Uuid::now_v7(),
            Uuid::now_v7(),
            false,
            21,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::AlreadyApplied
        );
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&start)
                .unwrap(),
            JudgeRecordAck::AlreadyApplied
        );
        let counts: (i64, i64, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT COUNT(*) FROM judge_attempt_state_events WHERE state = 'valid'),
                    (SELECT COUNT(*) FROM evaluations),
                    (SELECT COUNT(*) FROM evaluation_hard_failures)",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(counts, (1, 1, 1));
        assert!(
            super::verified_judge_evaluation_provenance(
                &activated.repository.connection,
                terminal.evaluation.as_ref().unwrap().evaluation_id,
                shadow_attempt_id,
            )
            .unwrap()
        );
        let stored_output: (String, String, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT safe_output_json, raw_output_sha256, raw_output_bytes
                 FROM judge_attempt_state_events WHERE state = 'valid'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(stored_output.1, super::sha256_hex(raw_output.as_bytes()));
        assert_eq!(stored_output.2, raw_output.len() as i64);
        assert!(stored_output.0.contains("response_equivalence"));
        assert!(!stored_output.0.contains("judge-output"));

        let mismatch = JudgeAttemptTerminal::transport_failure(
            start.judge_attempt_id,
            terminal.judge_attempt_state_event_id,
            Uuid::now_v7(),
            JudgeTransportFailureClass::Timeout,
            21,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&mismatch)
                .unwrap(),
            JudgeRecordAck::Conflict
        );
    }

    #[test]
    fn repair_requires_one_matching_invalid_initial_attempt() {
        let (_temporary, config, mut activated) = activate();

        let missing_shadow = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "missing-initial",
        );
        let missing = start_record(&activated.identity, &config, missing_shadow, 1);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&missing)
                .unwrap(),
            JudgeRecordAck::Conflict
        );

        let valid_shadow = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "valid-repair",
        );
        let valid_initial = start_record(&activated.identity, &config, valid_shadow, 0);
        activated
            .repository
            .record_judge_attempt_start(&valid_initial)
            .unwrap();
        let valid_terminal = JudgeAttemptTerminal::valid(
            valid_initial.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            valid_shadow,
            evaluator_version(&config),
            &judge_raw(false),
            judge(&config),
            Uuid::now_v7(),
            Uuid::now_v7(),
            false,
            22,
        )
        .unwrap();
        activated
            .repository
            .record_judge_attempt_terminal(&valid_terminal)
            .unwrap();
        let after_valid = start_record(&activated.identity, &config, valid_shadow, 1);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&after_valid)
                .unwrap(),
            JudgeRecordAck::Conflict
        );

        let transport_shadow = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "transport-repair",
        );
        let transport_initial = start_record(&activated.identity, &config, transport_shadow, 0);
        activated
            .repository
            .record_judge_attempt_start(&transport_initial)
            .unwrap();
        let transport_terminal = JudgeAttemptTerminal::transport_failure(
            transport_initial.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            JudgeTransportFailureClass::Timeout,
            22,
        )
        .unwrap();
        activated
            .repository
            .record_judge_attempt_terminal(&transport_terminal)
            .unwrap();
        let after_transport = start_record(&activated.identity, &config, transport_shadow, 1);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&after_transport)
                .unwrap(),
            JudgeRecordAck::Conflict
        );

        let canceled_shadow = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "canceled-repair",
        );
        let canceled_initial = start_record(&activated.identity, &config, canceled_shadow, 0);
        activated
            .repository
            .record_judge_attempt_start(&canceled_initial)
            .unwrap();
        let canceled_terminal = JudgeAttemptTerminal::canceled_shutdown(
            canceled_initial.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            22,
        )
        .unwrap();
        activated
            .repository
            .record_judge_attempt_terminal(&canceled_terminal)
            .unwrap();
        let after_canceled = start_record(&activated.identity, &config, canceled_shadow, 1);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&after_canceled)
                .unwrap(),
            JudgeRecordAck::Conflict
        );

        let invalid_shadow = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "invalid-repair",
        );
        let invalid_initial = start_record(&activated.identity, &config, invalid_shadow, 0);
        activated
            .repository
            .record_judge_attempt_start(&invalid_initial)
            .unwrap();
        let invalid_terminal = JudgeAttemptTerminal::invalid(
            invalid_initial.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            "{}",
            judge(&config),
            22,
        )
        .unwrap();
        activated
            .repository
            .record_judge_attempt_terminal(&invalid_terminal)
            .unwrap();
        let matching_repair = start_record(&activated.identity, &config, invalid_shadow, 1);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&matching_repair)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        let repaired_terminal = JudgeAttemptTerminal::valid(
            matching_repair.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            invalid_shadow,
            evaluator_version(&config),
            &judge_raw(false),
            judge(&config),
            Uuid::now_v7(),
            Uuid::now_v7(),
            false,
            23,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&repaired_terminal)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&invalid_terminal)
                .unwrap(),
            JudgeRecordAck::AlreadyApplied
        );

        let different_shadow = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "different-contract",
        );
        let different_initial = start_record(&activated.identity, &config, different_shadow, 0);
        activated
            .repository
            .record_judge_attempt_start(&different_initial)
            .unwrap();
        let different_terminal = JudgeAttemptTerminal::invalid(
            different_initial.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            "{}",
            judge(&config),
            22,
        )
        .unwrap();
        activated
            .repository
            .record_judge_attempt_terminal(&different_terminal)
            .unwrap();
        let mut changed_judge = judge(&config).clone();
        changed_judge.model = "different-judge-model".to_string();
        let different_repair = JudgeAttemptStart::new(
            Uuid::now_v7(),
            different_shadow,
            activated.identity.pools["pool-a"].learning_generation_id,
            evaluator_version(&config),
            &changed_judge,
            &judge_input(&config),
            1,
            Uuid::now_v7(),
            Uuid::now_v7(),
            23,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&different_repair)
                .unwrap(),
            JudgeRecordAck::Conflict
        );
    }

    #[test]
    fn repair_rejects_rehashed_invalid_evidence_with_mismatched_raw_identity() {
        let (_temporary, config, mut activated) = activate();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "invalid-evidence-corruption",
        );
        let initial = start_record(&activated.identity, &config, shadow_attempt_id, 0);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&initial)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        let terminal = JudgeAttemptTerminal::invalid(
            initial.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            "{}",
            judge(&config),
            22,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::Applied
        );

        let mut safe_output: serde_json::Value = activated
            .repository
            .connection
            .query_row(
                "SELECT safe_output_json FROM judge_attempt_state_events
                 WHERE judge_attempt_id = ?1 AND state = 'invalid'",
                params![initial.judge_attempt_id.to_string()],
                |row| {
                    let value: String = row.get(0)?;
                    Ok(serde_json::from_str(&value).unwrap())
                },
            )
            .unwrap();
        safe_output["output"]["sha256"] = json!("f".repeat(64));
        let safe_output = canonical_json(&safe_output).unwrap();
        activated
            .repository
            .connection
            .execute(
                "UPDATE judge_attempt_state_events SET safe_output_json = ?1
                 WHERE judge_attempt_id = ?2 AND state = 'invalid'",
                params![safe_output, initial.judge_attempt_id.to_string()],
            )
            .unwrap();
        let stored = super::load_judge_state(
            &activated.repository.connection,
            initial.judge_attempt_id,
            true,
        )
        .unwrap()
        .unwrap();
        let rehashed = super::stored_judge_state_hash(initial.judge_attempt_id, &stored).unwrap();
        activated
            .repository
            .connection
            .execute(
                "UPDATE judge_attempt_state_events SET canonical_payload_hash = ?1
                 WHERE judge_attempt_id = ?2 AND state = 'invalid'",
                params![rehashed, initial.judge_attempt_id.to_string()],
            )
            .unwrap();

        let repair = start_record(&activated.identity, &config, shadow_attempt_id, 1);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&repair)
                .unwrap(),
            JudgeRecordAck::Conflict
        );
    }

    #[test]
    fn judge_parent_and_shadow_prerequisites_are_verified_before_writes() {
        let (_temporary, config, mut activated) = activate();
        let missing = start_record(&activated.identity, &config, Uuid::now_v7(), 0);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&missing)
                .unwrap(),
            JudgeRecordAck::Conflict
        );

        let missing_shadow_start = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "missing-shadow-start",
        );
        activated
            .repository
            .connection
            .execute(
                "DELETE FROM shadow_attempt_state_events
                 WHERE shadow_attempt_id = ?1 AND state = 'started'",
                params![missing_shadow_start.to_string()],
            )
            .unwrap();
        let start_without_shadow_start =
            start_record(&activated.identity, &config, missing_shadow_start, 0);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&start_without_shadow_start)
                .unwrap(),
            JudgeRecordAck::Conflict
        );

        let shadow_attempt_id = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "corrupt-parent",
        );
        let start = start_record(&activated.identity, &config, shadow_attempt_id, 0);
        activated
            .repository
            .record_judge_attempt_start(&start)
            .unwrap();
        activated
            .repository
            .connection
            .execute(
                "UPDATE judge_attempts SET canonical_payload_hash = ?1
                 WHERE judge_attempt_id = ?2",
                params!["f".repeat(64), start.judge_attempt_id.to_string()],
            )
            .unwrap();
        let terminal = JudgeAttemptTerminal::transport_failure(
            start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            JudgeTransportFailureClass::Timeout,
            24,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::Conflict
        );
        let terminal_rows: i64 = activated
            .repository
            .connection
            .query_row(
                "SELECT COUNT(*) FROM judge_attempt_state_events WHERE state <> 'started'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(terminal_rows, 0);

        let corrupt_started_shadow = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "corrupt-judge-start",
        );
        let corrupt_started = start_record(&activated.identity, &config, corrupt_started_shadow, 0);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&corrupt_started)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        activated
            .repository
            .connection
            .execute(
                "UPDATE judge_attempt_state_events SET canonical_payload_hash = ?1
                 WHERE judge_attempt_id = ?2 AND state = 'started'",
                params!["0".repeat(64), corrupt_started.judge_attempt_id.to_string()],
            )
            .unwrap();
        let terminal = JudgeAttemptTerminal::transport_failure(
            corrupt_started.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            JudgeTransportFailureClass::Timeout,
            25,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::Conflict
        );
    }

    #[test]
    fn deterministic_evaluation_is_exact_and_has_no_judge_contract() {
        let (_temporary, config, mut activated) = activate();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "deterministic",
        );
        let evaluation = EvaluationRecord::deterministic(
            Uuid::now_v7(),
            shadow_attempt_id,
            evaluator_version(&config),
            Uuid::now_v7(),
            DeterministicHardFailureV1::ResponseSchema,
            false,
            judge(&config),
            20,
        )
        .unwrap();
        assert_eq!(
            activated.repository.record_evaluation(&evaluation).unwrap(),
            JudgeRecordAck::Applied
        );
        assert_eq!(
            activated.repository.record_evaluation(&evaluation).unwrap(),
            JudgeRecordAck::AlreadyApplied
        );
        let stored: (String, Option<String>, String) = activated
            .repository
            .connection
            .query_row(
                "SELECT e.source, e.judge_model, h.hard_failure
                 FROM evaluations AS e
                 JOIN evaluation_hard_failures AS h USING (evaluation_id)",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert_eq!(
            stored,
            (
                "deterministic_validator".to_string(),
                None,
                "response_schema".to_string(),
            )
        );
    }

    #[test]
    fn normal_judge_writes_reject_inverted_shadow_and_attempt_timestamps() {
        let (_temporary, config, mut activated) = activate();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "chronology",
        );
        let early_evaluation = EvaluationRecord::deterministic(
            Uuid::now_v7(),
            shadow_attempt_id,
            evaluator_version(&config),
            Uuid::now_v7(),
            DeterministicHardFailureV1::ResponseSchema,
            false,
            judge(&config),
            10,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_evaluation(&early_evaluation)
                .unwrap(),
            JudgeRecordAck::Conflict
        );

        let mut early_start = start_record(&activated.identity, &config, shadow_attempt_id, 0);
        early_start.created_at_unix_ms = 10;
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&early_start)
                .unwrap(),
            JudgeRecordAck::Conflict
        );
        let start = start_record(&activated.identity, &config, shadow_attempt_id, 0);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&start)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        let early_terminal = JudgeAttemptTerminal::transport_failure(
            start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            JudgeTransportFailureClass::Timeout,
            start.created_at_unix_ms - 1,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&early_terminal)
                .unwrap(),
            JudgeRecordAck::Conflict
        );
        let counts: (i64, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT count(*) FROM evaluations WHERE shadow_attempt_id = ?1),
                    (SELECT count(*) FROM judge_attempt_state_events
                     WHERE judge_attempt_id = ?2 AND state <> 'started')",
                params![
                    shadow_attempt_id.to_string(),
                    start.judge_attempt_id.to_string()
                ],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(counts, (0, 0));
    }

    #[test]
    fn deterministic_and_judge_evaluation_paths_are_mutually_exclusive() {
        let (_temporary, config, mut activated) = activate();
        let deterministic_shadow = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "deterministic-first",
        );
        let deterministic = EvaluationRecord::deterministic(
            Uuid::now_v7(),
            deterministic_shadow,
            evaluator_version(&config),
            Uuid::now_v7(),
            DeterministicHardFailureV1::ResponseSchema,
            false,
            judge(&config),
            20,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_evaluation(&deterministic)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        let forbidden_start = start_record(&activated.identity, &config, deterministic_shadow, 0);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&forbidden_start)
                .unwrap(),
            JudgeRecordAck::Conflict
        );

        let judge_shadow = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "judge-first",
        );
        let start = start_record(&activated.identity, &config, judge_shadow, 0);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&start)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        let forbidden_deterministic = EvaluationRecord::deterministic(
            Uuid::now_v7(),
            judge_shadow,
            evaluator_version(&config),
            Uuid::now_v7(),
            DeterministicHardFailureV1::MalformedCandidate,
            false,
            judge(&config),
            21,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_evaluation(&forbidden_deterministic)
                .unwrap(),
            JudgeRecordAck::Conflict
        );

        super::insert_evaluation(&activated.repository.connection, &forbidden_deterministic)
            .unwrap();
        let terminal = JudgeAttemptTerminal::transport_failure(
            start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            JudgeTransportFailureClass::Timeout,
            22,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::Conflict
        );
        let terminal_count: i64 = activated
            .repository
            .connection
            .query_row(
                "SELECT COUNT(*) FROM judge_attempt_state_events WHERE state <> 'started'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(terminal_count, 0);
    }

    #[test]
    fn exact_transport_retry_rejects_an_injected_evaluation() {
        let (_temporary, config, mut activated) = activate();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "transport-retry-evaluation",
        );
        let start = start_record(&activated.identity, &config, shadow_attempt_id, 0);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&start)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        let terminal = JudgeAttemptTerminal::transport_failure(
            start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            JudgeTransportFailureClass::Timeout,
            22,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        let injected = EvaluationRecord::deterministic(
            Uuid::now_v7(),
            shadow_attempt_id,
            evaluator_version(&config),
            Uuid::now_v7(),
            DeterministicHardFailureV1::MalformedCandidate,
            false,
            judge(&config),
            23,
        )
        .unwrap();
        super::insert_evaluation(&activated.repository.connection, &injected).unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::Conflict
        );
    }

    #[test]
    fn signed_zero_scores_persist_with_exact_bits_and_retry() {
        let (_temporary, config, mut activated) = activate();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "negative-zero",
        );
        let start = start_record(&activated.identity, &config, shadow_attempt_id, 0);
        activated
            .repository
            .record_judge_attempt_start(&start)
            .unwrap();
        let raw = judge_raw_with_scores(-0.0, -0.0, false);
        let terminal = JudgeAttemptTerminal::valid(
            start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            shadow_attempt_id,
            evaluator_version(&config),
            &raw,
            judge(&config),
            Uuid::now_v7(),
            Uuid::now_v7(),
            false,
            21,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::AlreadyApplied
        );
        let bits: (i64, i64, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT response_equivalence_bits, trajectory_equivalence_bits,
                        aggregate_score_bits FROM evaluations",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        let negative_zero_bits = (-0.0_f64).to_bits() as i64;
        assert_eq!(
            bits,
            (negative_zero_bits, negative_zero_bits, negative_zero_bits)
        );
    }

    #[test]
    fn aggregate_above_one_within_weight_tolerance_persists_exactly() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        }
        let mut config = config(&temporary.path().join("ledger.db"));
        config.pools[0].judge.response_weight = 0.500_000_000_4;
        config.pools[0].judge.trajectory_weight = 0.500_000_000_4;
        let mut activated = LedgerRepository::activate(&config).unwrap();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "aggregate-tolerance",
        );
        let start = start_record(&activated.identity, &config, shadow_attempt_id, 0);
        activated
            .repository
            .record_judge_attempt_start(&start)
            .unwrap();
        let tolerant_judge = judge(&config);
        let expected = tolerant_judge.response_weight + tolerant_judge.trajectory_weight;
        assert!(expected > 1.0 && expected <= 1.000_000_001);
        let terminal = JudgeAttemptTerminal::valid(
            start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            shadow_attempt_id,
            evaluator_version(&config),
            &judge_raw_with_scores(1.0, 1.0, false),
            tolerant_judge,
            Uuid::now_v7(),
            Uuid::now_v7(),
            false,
            21,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::AlreadyApplied
        );
        let stored: (f64, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT aggregate_score, aggregate_score_bits FROM evaluations",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(stored.0.to_bits(), expected.to_bits());
        assert_eq!(stored.1, expected.to_bits() as i64);
    }

    #[test]
    fn judge_writes_reject_scoring_policy_drift_from_the_shadow_attempt() {
        let (_temporary, config, mut activated) = activate();
        let mut drifted_judge = judge(&config).clone();
        drifted_judge.response_weight = 0.500_000_000_4;
        drifted_judge.trajectory_weight = 0.500_000_000_4;
        drifted_judge.pass_threshold = 0.25;
        drifted_judge.max_rationale_bytes = 8_192;

        let start_shadow = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "drifted-start",
        );
        let drifted_start = JudgeAttemptStart::new(
            Uuid::now_v7(),
            start_shadow,
            activated.identity.pools["pool-a"].learning_generation_id,
            evaluator_version(&config),
            &drifted_judge,
            &judge_input(&config),
            0,
            Uuid::now_v7(),
            Uuid::now_v7(),
            20,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&drifted_start)
                .unwrap(),
            JudgeRecordAck::Conflict
        );

        let terminal_shadow = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "drifted-terminal",
        );
        let start = start_record(&activated.identity, &config, terminal_shadow, 0);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start(&start)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        let drifted_terminal = JudgeAttemptTerminal::valid(
            start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            terminal_shadow,
            evaluator_version(&config),
            &judge_raw(false),
            &drifted_judge,
            Uuid::now_v7(),
            Uuid::now_v7(),
            false,
            21,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&drifted_terminal)
                .unwrap(),
            JudgeRecordAck::Conflict
        );

        let deterministic_shadow = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "drifted-deterministic",
        );
        let drifted_evaluation = EvaluationRecord::deterministic(
            Uuid::now_v7(),
            deterministic_shadow,
            evaluator_version(&config),
            Uuid::now_v7(),
            DeterministicHardFailureV1::ToolContract,
            false,
            &drifted_judge,
            21,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_evaluation(&drifted_evaluation)
                .unwrap(),
            JudgeRecordAck::Conflict
        );
        let counts: (i64, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT
                    (SELECT COUNT(*) FROM judge_attempt_state_events WHERE state <> 'started'),
                    (SELECT COUNT(*) FROM evaluations)",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .unwrap();
        assert_eq!(counts, (0, 0));
    }

    #[test]
    fn invalid_credential_output_persists_only_redacted_bounded_evidence() {
        let (temporary, config, mut activated) = activate();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "redacted-invalid",
        );
        let start = start_record(&activated.identity, &config, shadow_attempt_id, 0);
        activated
            .repository
            .record_judge_attempt_start(&start)
            .unwrap();
        let secret = "sk-live-must-never-reach-the-ledger";
        let raw = format!(
            r#"{{"authorization":"Bearer {secret}","padding":"{}"}}"#,
            "x".repeat(judge(&config).max_rationale_bytes + 9_000)
        );
        let valid_error = JudgeAttemptTerminal::valid(
            start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            shadow_attempt_id,
            evaluator_version(&config),
            &raw,
            judge(&config),
            Uuid::now_v7(),
            Uuid::now_v7(),
            false,
            21,
        )
        .err()
        .expect("credential-bearing output must not produce a valid terminal");
        assert!(!format!("{valid_error:?}").contains(secret));
        let terminal = JudgeAttemptTerminal::invalid(
            start.judge_attempt_id,
            Uuid::now_v7(),
            Uuid::now_v7(),
            &raw,
            judge(&config),
            21,
        )
        .unwrap();
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_terminal(&terminal)
                .unwrap(),
            JudgeRecordAck::Applied
        );
        let stored: (String, String, i64) = activated
            .repository
            .connection
            .query_row(
                "SELECT safe_output_json, raw_output_sha256, raw_output_bytes
                 FROM judge_attempt_state_events WHERE state = 'invalid'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
            )
            .unwrap();
        assert!(!stored.0.contains(secret));
        assert!(!stored.0.contains("Bearer"));
        assert_eq!(stored.1, super::sha256_hex(raw.as_bytes()));
        assert_eq!(stored.2, raw.len() as i64);
        for entry in fs::read_dir(temporary.path()).unwrap() {
            let path = entry.unwrap().path();
            if path.is_file() {
                let bytes = fs::read(path).unwrap();
                assert!(
                    !bytes
                        .windows(secret.len())
                        .any(|window| window == secret.as_bytes())
                );
            }
        }
    }

    #[test]
    fn middleware_interference_has_a_closed_judge_transport_class() {
        let terminal = JudgeAttemptTerminal::transport_failure(
            Uuid::now_v7(),
            Uuid::now_v7(),
            Uuid::now_v7(),
            JudgeTransportFailureClass::MiddlewareInterference,
            1,
        )
        .unwrap();
        assert_eq!(
            terminal.fields().stable_error_class,
            Some("router.provider.middleware_interference")
        );
    }

    #[test]
    fn schema_rejects_judge_terminal_states_with_incoherent_evidence() {
        let (_temporary, config, mut activated) = activate();
        let cases = [
            ("valid", Some("valid"), Some("{}"), None, None, None),
            (
                "invalid",
                Some("invalid"),
                None,
                Some("a".repeat(64)),
                Some(1_i64),
                None,
            ),
            (
                "transport_failure",
                Some("operational"),
                None,
                Some("b".repeat(64)),
                Some(1_i64),
                Some("router.provider.timeout"),
            ),
        ];
        for (index, (state, parse_result, safe_output, raw_hash, raw_bytes, error_class)) in
            cases.into_iter().enumerate()
        {
            let shadow_attempt_id = seed_shadow_attempt(
                &mut activated.repository,
                &activated.identity,
                &config,
                &format!("bad-state-{index}"),
            );
            let start = start_record(&activated.identity, &config, shadow_attempt_id, 0);
            activated
                .repository
                .record_judge_attempt_start(&start)
                .unwrap();
            let error = activated
                .repository
                .connection
                .execute(
                    "INSERT INTO judge_attempt_state_events (
                        judge_attempt_state_event_id, judge_attempt_id,
                        process_instance_id, state, parse_result, safe_output_json,
                        raw_output_sha256, raw_output_bytes, stable_error_class,
                        created_at_unix_ms, canonical_payload_hash
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 30, ?10)",
                    params![
                        Uuid::now_v7().to_string(),
                        start.judge_attempt_id.to_string(),
                        activated.identity.process_instance_id.to_string(),
                        state,
                        parse_result,
                        safe_output,
                        raw_hash,
                        raw_bytes,
                        error_class,
                        "c".repeat(64),
                    ],
                )
                .unwrap_err();
            assert_eq!(
                error.sqlite_error_code(),
                Some(rusqlite::ErrorCode::ConstraintViolation)
            );
        }
    }

    #[test]
    fn transaction_start_refusal_writes_nothing() {
        let (_temporary, config, mut activated) = activate();
        let shadow_attempt_id = seed_shadow_attempt(
            &mut activated.repository,
            &activated.identity,
            &config,
            "refused",
        );
        let start = start_record(&activated.identity, &config, shadow_attempt_id, 0);
        assert_eq!(
            activated
                .repository
                .record_judge_attempt_start_with_start_check(&start, || None::<()>)
                .unwrap(),
            JudgeRecordAck::TransactionNotStarted
        );
        let count: i64 = activated
            .repository
            .connection
            .query_row("SELECT COUNT(*) FROM judge_attempts", [], |row| row.get(0))
            .unwrap();
        assert_eq!(count, 0);
    }
}
