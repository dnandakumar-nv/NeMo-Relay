// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure pairwise judge contracts, validation, repair, and evaluation math.

#![allow(dead_code)] // Task 4 consumes these pure contracts when it builds provider requests.

mod evaluation;
mod output;

// Task 4 consumes this curated surface from sibling adapter and validator modules.
#[allow(unused_imports)]
pub(crate) use evaluation::{
    DeterministicHardFailureV1, JudgeBinaryLabelV1, JudgeEvaluationSourceV1, JudgeEvaluationV1,
    JudgeLabelV1, ScoredValueV1,
};
#[allow(unused_imports)]
pub(crate) use output::{
    InvalidJudgeOutputMarkerV1, JudgeAttemptOrdinalV1, JudgeAttemptOutcomeV1,
    JudgeAttemptProgressV1, JudgeAttemptResolutionErrorV1, JudgeAttemptV1, JudgeFinalStateV1,
    JudgeHardFailureV1, JudgeOutputOperationalFailureV1, JudgeRepairEvidenceV1,
    JudgeRepairPendingV1, JudgeRepairRejectionV1, PairwiseJudgeResultV1,
    resolve_judge_attempt_progress, validate_pairwise_judge_output,
};

use std::sync::Arc;

use nemo_relay::api::llm::LlmApiFamily;
use serde::Serialize;
use serde_json::Value as Json;

use crate::config::{
    JUDGE_OUTPUT_SCHEMA_SHA256_V1, JUDGE_OUTPUT_SCHEMA_V1, JUDGE_OUTPUT_SCHEMA_VERSION,
    JUDGE_PROMPT_TEMPLATE_SHA256_V1, JUDGE_PROMPT_TEMPLATE_V1, JUDGE_PROMPT_VERSION_V1,
    JUDGE_RUBRIC_TEMPLATE_SHA256_V1, JUDGE_RUBRIC_TEMPLATE_V1, JUDGE_RUBRIC_VERSION_V1,
    JudgeConfig,
};
use crate::fingerprint::{canonical_serialize_bytes, sha256_hex};
use crate::preflight::has_sensitive_value_shape;
use crate::projection::{RouterRequestProjectionV1, is_sensitive_projection_key};
use crate::trajectory::{CapturedTrajectoryEvent, RouterResponseProjectionV1, TrajectoryTrigger};

pub(crate) const PAIRWISE_JUDGE_INPUT_SCHEMA_V1: &str = "nemo.relay.router.pairwise-judge-input@1";
pub(crate) const PAIRWISE_JUDGE_RESULT_SCHEMA_V1: &str =
    "nemo.relay.router.pairwise-judge-result@1";
pub(crate) const MAX_PAIRWISE_JUDGE_INPUT_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JudgeRegistryArtifactV1 {
    Prompt,
    Rubric,
    OutputSchema,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JudgeRegistryErrorV1 {
    UnsupportedConfigVersion,
    UnsupportedPromptVersion,
    UnsupportedRubricVersion,
    UnsupportedOutputSchemaVersion,
    InvalidUtf8(JudgeRegistryArtifactV1),
    HashMismatch(JudgeRegistryArtifactV1),
}

#[derive(Clone)]
pub(crate) struct JudgeRegistryV1 {
    prompt: &'static str,
    rubric: &'static str,
    output_schema: &'static str,
    policy: JudgePolicyIdentityV1,
}

impl JudgeRegistryV1 {
    pub(crate) fn load(config: &JudgeConfig) -> Result<Self, JudgeRegistryErrorV1> {
        if config.version != 1 {
            return Err(JudgeRegistryErrorV1::UnsupportedConfigVersion);
        }
        if config.prompt_version != JUDGE_PROMPT_VERSION_V1 {
            return Err(JudgeRegistryErrorV1::UnsupportedPromptVersion);
        }
        if config.rubric_version != JUDGE_RUBRIC_VERSION_V1 {
            return Err(JudgeRegistryErrorV1::UnsupportedRubricVersion);
        }
        if config.output_schema_version != JUDGE_OUTPUT_SCHEMA_VERSION {
            return Err(JudgeRegistryErrorV1::UnsupportedOutputSchemaVersion);
        }

        verify_registry_hash(
            JudgeRegistryArtifactV1::Prompt,
            JUDGE_PROMPT_TEMPLATE_V1,
            JUDGE_PROMPT_TEMPLATE_SHA256_V1,
        )?;
        verify_registry_hash(
            JudgeRegistryArtifactV1::Rubric,
            JUDGE_RUBRIC_TEMPLATE_V1,
            JUDGE_RUBRIC_TEMPLATE_SHA256_V1,
        )?;
        verify_registry_hash(
            JudgeRegistryArtifactV1::OutputSchema,
            JUDGE_OUTPUT_SCHEMA_V1,
            JUDGE_OUTPUT_SCHEMA_SHA256_V1,
        )?;

        let policy = JudgePolicyIdentityV1 {
            prompt_version: config.prompt_version.clone(),
            prompt_template_sha256: JUDGE_PROMPT_TEMPLATE_SHA256_V1.to_string(),
            rubric_version: config.rubric_version.clone(),
            rubric_template_sha256: JUDGE_RUBRIC_TEMPLATE_SHA256_V1.to_string(),
            output_schema_id: PAIRWISE_JUDGE_RESULT_SCHEMA_V1.to_string(),
            output_schema_version: config.output_schema_version,
            output_schema_sha256: JUDGE_OUTPUT_SCHEMA_SHA256_V1.to_string(),
        };
        Ok(Self {
            prompt: registry_text(JudgeRegistryArtifactV1::Prompt, JUDGE_PROMPT_TEMPLATE_V1)?,
            rubric: registry_text(JudgeRegistryArtifactV1::Rubric, JUDGE_RUBRIC_TEMPLATE_V1)?,
            output_schema: registry_text(
                JudgeRegistryArtifactV1::OutputSchema,
                JUDGE_OUTPUT_SCHEMA_V1,
            )?,
            policy,
        })
    }

    pub(crate) fn prompt(&self) -> &'static str {
        self.prompt
    }

    pub(crate) fn rubric(&self) -> &'static str {
        self.rubric
    }

    pub(crate) fn output_schema(&self) -> &'static str {
        self.output_schema
    }

    fn policy_identity(&self) -> JudgePolicyIdentityV1 {
        self.policy.clone()
    }
}

fn verify_registry_hash(
    artifact: JudgeRegistryArtifactV1,
    bytes: &[u8],
    expected: &str,
) -> Result<(), JudgeRegistryErrorV1> {
    if sha256_hex(bytes) == expected {
        Ok(())
    } else {
        Err(JudgeRegistryErrorV1::HashMismatch(artifact))
    }
}

fn registry_text(
    artifact: JudgeRegistryArtifactV1,
    bytes: &'static [u8],
) -> Result<&'static str, JudgeRegistryErrorV1> {
    std::str::from_utf8(bytes).map_err(|_| JudgeRegistryErrorV1::InvalidUtf8(artifact))
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct JudgePolicyIdentityV1 {
    prompt_version: String,
    prompt_template_sha256: String,
    rubric_version: String,
    rubric_template_sha256: String,
    output_schema_id: String,
    output_schema_version: u32,
    output_schema_sha256: String,
}

impl JudgePolicyIdentityV1 {
    pub(crate) fn from_config(config: &JudgeConfig) -> Result<Self, JudgeRegistryErrorV1> {
        JudgeRegistryV1::load(config).map(|registry| registry.policy_identity())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct JudgeHorizonV1 {
    requested_progress: u64,
    observed_progress: u64,
    trigger: TrajectoryTrigger,
    is_partial: bool,
}

impl JudgeHorizonV1 {
    pub(crate) fn new(
        requested_progress: usize,
        observed_progress: usize,
        trigger: TrajectoryTrigger,
        is_partial: bool,
    ) -> Result<Self, JudgeInputErrorV1> {
        let requested_progress =
            u64::try_from(requested_progress).map_err(|_| JudgeInputErrorV1::InvalidHorizon)?;
        let observed_progress =
            u64::try_from(observed_progress).map_err(|_| JudgeInputErrorV1::InvalidHorizon)?;
        if requested_progress == 0 {
            return Err(JudgeInputErrorV1::InvalidHorizon);
        }
        let valid_terminal = match trigger {
            TrajectoryTrigger::ProgressReached => {
                !is_partial && observed_progress == requested_progress
            }
            TrajectoryTrigger::OwnerTerminated
            | TrajectoryTrigger::DeadlineElapsed
            | TrajectoryTrigger::Shutdown => is_partial && observed_progress < requested_progress,
        };
        if !valid_terminal {
            return Err(JudgeInputErrorV1::InvalidHorizon);
        }
        Ok(Self {
            requested_progress,
            observed_progress,
            trigger,
            is_partial,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum JudgeInputErrorV1 {
    InvalidHorizon,
    EventsOutOfOrder,
    SensitiveValue,
    CanonicalizationFailed,
    TooLarge { actual: usize, maximum: usize },
}

#[derive(Clone, PartialEq, Serialize)]
pub(crate) struct PairwiseJudgeInputV1 {
    schema: String,
    request: RouterRequestProjectionV1,
    anchor_response: RouterResponseProjectionV1,
    candidate_response: RouterResponseProjectionV1,
    trajectory: Vec<CapturedTrajectoryEvent>,
    horizon: JudgeHorizonV1,
    policy: JudgePolicyIdentityV1,
}

impl PairwiseJudgeInputV1 {
    pub(crate) fn new(
        request: &RouterRequestProjectionV1,
        anchor_response: &RouterResponseProjectionV1,
        candidate_response: &RouterResponseProjectionV1,
        events: &[Arc<CapturedTrajectoryEvent>],
        horizon: JudgeHorizonV1,
        policy: JudgePolicyIdentityV1,
    ) -> Result<Self, JudgeInputErrorV1> {
        if events
            .windows(2)
            .any(|pair| pair[0].ingest_seq >= pair[1].ingest_seq)
        {
            return Err(JudgeInputErrorV1::EventsOutOfOrder);
        }

        let input = Self {
            schema: PAIRWISE_JUDGE_INPUT_SCHEMA_V1.to_string(),
            request: request.clone(),
            anchor_response: anchor_response.clone(),
            candidate_response: candidate_response.clone(),
            trajectory: events.iter().map(|event| event.as_ref().clone()).collect(),
            horizon,
            policy,
        };
        let value =
            serde_json::to_value(&input).map_err(|_| JudgeInputErrorV1::CanonicalizationFailed)?;
        if json_contains_sensitive_value(&value) {
            return Err(JudgeInputErrorV1::SensitiveValue);
        }
        let bytes = canonical_serialize_bytes(&input)
            .map_err(|_| JudgeInputErrorV1::CanonicalizationFailed)?;
        if bytes.len() > MAX_PAIRWISE_JUDGE_INPUT_BYTES {
            return Err(JudgeInputErrorV1::TooLarge {
                actual: bytes.len(),
                maximum: MAX_PAIRWISE_JUDGE_INPUT_BYTES,
            });
        }
        Ok(input)
    }

    pub(crate) fn canonical_bytes(&self) -> Result<Vec<u8>, JudgeInputErrorV1> {
        canonical_serialize_bytes(self).map_err(|_| JudgeInputErrorV1::CanonicalizationFailed)
    }

    /// Return the authoritative family captured in the sealed request projection.
    pub(crate) const fn family(&self) -> LlmApiFamily {
        self.request.family
    }

    pub(crate) fn canonical_sha256(&self) -> Result<String, JudgeInputErrorV1> {
        self.canonical_bytes().map(|bytes| sha256_hex(&bytes))
    }

    /// Return the candidate response sealed into this exact Judge input.
    pub(crate) const fn candidate_response(&self) -> &RouterResponseProjectionV1 {
        &self.candidate_response
    }
}

fn json_contains_sensitive_value(value: &Json) -> bool {
    match value {
        Json::String(value) => contains_sensitive_free_text(value),
        Json::Array(values) => values.iter().any(json_contains_sensitive_value),
        Json::Object(values) => values.iter().any(|(key, value)| {
            is_sensitive_projection_key(key)
                || contains_sensitive_free_text(key)
                || json_contains_sensitive_value(value)
        }),
        Json::Null | Json::Bool(_) | Json::Number(_) => false,
    }
}

pub(crate) fn contains_sensitive_free_text(value: &str) -> bool {
    if value.len() > MAX_PAIRWISE_JUDGE_INPUT_BYTES {
        return true;
    }
    let lowercase = value.to_ascii_lowercase();
    const ASSIGNMENT_MARKERS: [&str; 8] = [
        "authorization:",
        "api_key=",
        "api-key=",
        "apikey=",
        "password=",
        "passwd=",
        "secret=",
        "token=",
    ];
    if ASSIGNMENT_MARKERS
        .iter()
        .any(|marker| lowercase.contains(marker))
        || (lowercase.contains("-----begin ") && lowercase.contains("private key-----"))
        || contains_sensitive_labeled_sequence(value)
    {
        return true;
    }

    let mut pending_label = None;
    for candidate in value
        .split(credential_candidate_delimiter)
        .map(|candidate| candidate.trim_matches(free_text_delimiter))
        .filter(|candidate| !candidate.is_empty())
    {
        if let Some(label) = pending_label.take()
            && labeled_material_is_sensitive(label, candidate)
        {
            return true;
        }
        if has_sensitive_value_shape_or_prefix(candidate)
            || contains_credential_bearing_url(candidate)
            || labeled_candidate_is_sensitive(candidate)
        {
            return true;
        }
        pending_label = standalone_credential_label(candidate);
    }
    false
}

fn contains_sensitive_labeled_sequence(value: &str) -> bool {
    let mut cursor = 0;
    while cursor < value.len() {
        let Some(character) = value[cursor..].chars().next() else {
            break;
        };
        if !is_credential_label_character(character) {
            cursor += character.len_utf8();
            continue;
        }

        let label_start = cursor;
        while cursor < value.len() {
            let Some(character) = value[cursor..].chars().next() else {
                break;
            };
            if !is_credential_label_character(character) {
                break;
            }
            cursor += character.len_utf8();
        }
        let label = &value[label_start..cursor];
        let (next, had_whitespace) = skip_credential_whitespace(value, cursor);
        cursor = next;
        let had_separator = value[cursor..]
            .chars()
            .next()
            .is_some_and(|character| matches!(character, '=' | ':'));
        if had_separator {
            cursor += 1;
            cursor = skip_credential_whitespace(value, cursor).0;
        }

        let bearer_or_basic =
            label.eq_ignore_ascii_case("bearer") || label.eq_ignore_ascii_case("basic");
        if !(had_separator || bearer_or_basic && had_whitespace) {
            continue;
        }
        let material_start = cursor;
        while cursor < value.len() {
            let Some(character) = value[cursor..].chars().next() else {
                break;
            };
            if is_credential_material_delimiter(character) {
                break;
            }
            cursor += character.len_utf8();
        }
        let material = value[material_start..cursor].trim_matches(free_text_delimiter);
        if material.is_empty() {
            continue;
        }

        if label.eq_ignore_ascii_case("bearer") && looks_like_bearer_material(material)
            || label.eq_ignore_ascii_case("basic") && looks_like_basic_material(material)
            || matches_sensitive_assignment_label(label)
            || matches!(
                label.to_ascii_lowercase().as_str(),
                "credential" | "credentials" | "jwt" | "key"
            ) && (has_sensitive_value_shape_or_prefix(material)
                || contains_credential_bearing_url(material))
        {
            return true;
        }
    }
    false
}

fn skip_credential_whitespace(value: &str, mut cursor: usize) -> (usize, bool) {
    let start = cursor;
    while cursor < value.len() {
        let Some(character) = value[cursor..].chars().next() else {
            break;
        };
        if !character.is_whitespace() {
            break;
        }
        cursor += character.len_utf8();
    }
    (cursor, cursor != start)
}

fn is_credential_label_character(character: char) -> bool {
    character.is_ascii_alphanumeric() || matches!(character, '_' | '-')
}

fn is_credential_material_delimiter(character: char) -> bool {
    character.is_whitespace()
        || matches!(
            character,
            '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | ';'
        )
}

fn has_sensitive_value_shape_or_prefix(value: &str) -> bool {
    if has_sensitive_value_shape(value) {
        return true;
    }
    const AWS_ACCESS_KEY_ID_BYTES: usize = 20;
    value.len() > AWS_ACCESS_KEY_ID_BYTES
        && value.as_bytes()[..AWS_ACCESS_KEY_ID_BYTES]
            .iter()
            .all(u8::is_ascii)
        && has_sensitive_value_shape(&value[..AWS_ACCESS_KEY_ID_BYTES])
}

#[derive(Clone, Copy)]
enum CredentialLabel {
    Bearer,
    Basic,
}

fn credential_candidate_delimiter(character: char) -> bool {
    character.is_whitespace()
        || matches!(
            character,
            '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | ';'
        )
}

fn free_text_delimiter(character: char) -> bool {
    matches!(
        character,
        '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '<' | '>' | ',' | ';' | '.' | '!' | '?'
    )
}

fn labeled_candidate_is_sensitive(candidate: &str) -> bool {
    let Some(separator) = candidate.find(['=', ':']) else {
        return false;
    };
    let label = &candidate[..separator];
    let material = candidate[separator + 1..].trim_matches(free_text_delimiter);
    if material.is_empty() {
        return false;
    }
    if has_sensitive_value_shape_or_prefix(material) || contains_credential_bearing_url(material) {
        return true;
    }
    if label.eq_ignore_ascii_case("bearer") {
        return looks_like_bearer_material(material);
    }
    if label.eq_ignore_ascii_case("basic") {
        return looks_like_basic_material(material);
    }
    matches_sensitive_assignment_label(label)
}

fn matches_sensitive_assignment_label(label: &str) -> bool {
    matches!(
        label.to_ascii_lowercase().as_str(),
        "authorization"
            | "api_key"
            | "api-key"
            | "apikey"
            | "password"
            | "passwd"
            | "secret"
            | "token"
    )
}

fn standalone_credential_label(candidate: &str) -> Option<CredentialLabel> {
    let label = candidate.trim_end_matches(':');
    if label.eq_ignore_ascii_case("bearer") {
        Some(CredentialLabel::Bearer)
    } else if label.eq_ignore_ascii_case("basic") {
        Some(CredentialLabel::Basic)
    } else {
        None
    }
}

fn labeled_material_is_sensitive(label: CredentialLabel, material: &str) -> bool {
    match label {
        CredentialLabel::Bearer => looks_like_bearer_material(material),
        CredentialLabel::Basic => looks_like_basic_material(material),
    }
}

fn contains_credential_bearing_url(candidate: &str) -> bool {
    let lowercase = candidate.to_ascii_lowercase();
    ["https://", "http://"].into_iter().any(|scheme| {
        lowercase
            .find(scheme)
            .is_some_and(|index| has_sensitive_value_shape(&candidate[index..]))
    })
}

fn looks_like_bearer_material(value: &str) -> bool {
    value.len() >= 20 && value.bytes().all(|byte| byte.is_ascii_graphic())
}

fn looks_like_basic_material(value: &str) -> bool {
    value.len() >= 16
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'+' | b'/' | b'='))
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use chrono::Utc;
    use nemo_relay::api::llm::LlmApiFamily;
    use serde_json::Value as Json;
    use uuid::Uuid;

    use super::*;
    use crate::config::JudgeConfig;
    use crate::projection::{
        REQUEST_PROJECTION_SCHEMA_V1, ROUTER_SANITIZER_VERSION, RouterRequestProjectionV1,
        SanitizedAnnotatedLlmRequest, SanitizedMessage, SanitizedMessageContent,
    };
    use crate::trajectory::{
        CAPTURED_EVENT_SCHEMA_V1, CapturedCodecAnnotationsV1, CapturedEventKind,
        RESPONSE_PROJECTION_SCHEMA_V1, TRAJECTORY_SANITIZER_VERSION,
    };

    fn judge_config() -> JudgeConfig {
        JudgeConfig {
            version: 1,
            model: "judge-model".to_string(),
            model_revision: "judge-r1".to_string(),
            prompt_version: JUDGE_PROMPT_VERSION_V1.to_string(),
            rubric_version: JUDGE_RUBRIC_VERSION_V1.to_string(),
            output_schema_version: JUDGE_OUTPUT_SCHEMA_VERSION,
            temperature: None,
            response_weight: 0.5,
            trajectory_weight: 0.5,
            response_floor: 0.8,
            trajectory_floor: 0.8,
            judge_confidence_floor: 0.7,
            pass_threshold: 0.85,
            max_rationale_bytes: 4_096,
            base_cooloff_seconds: 10,
            max_cooloff_seconds: 300,
            unknown_fields: BTreeMap::new(),
        }
    }

    fn request_projection(text: &str) -> RouterRequestProjectionV1 {
        RouterRequestProjectionV1 {
            schema: REQUEST_PROJECTION_SCHEMA_V1.to_string(),
            family: LlmApiFamily::OpenAIChatCompletions,
            normalized_request: SanitizedAnnotatedLlmRequest {
                messages: vec![SanitizedMessage::User {
                    content: SanitizedMessageContent::Text(text.to_string()),
                    name: None,
                }],
                model: Some("anchor-model".to_string()),
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
            required_capabilities: vec!["tools".to_string()],
            sanitizer_version: ROUTER_SANITIZER_VERSION,
            semantic_request_fingerprint: "request-fingerprint".to_string(),
        }
    }

    fn response_projection(text: &str) -> RouterResponseProjectionV1 {
        RouterResponseProjectionV1 {
            schema: RESPONSE_PROJECTION_SCHEMA_V1.to_string(),
            sanitizer_version: TRAJECTORY_SANITIZER_VERSION,
            id: None,
            model: Some("response-model".to_string()),
            message: Some(SanitizedMessageContent::Text(text.to_string())),
            tool_calls: None,
            finish_reason: None,
            usage: None,
            semantic_response_fingerprint: "response-fingerprint".to_string(),
        }
    }

    fn event(ingest_seq: u64) -> Arc<CapturedTrajectoryEvent> {
        Arc::new(CapturedTrajectoryEvent {
            schema: CAPTURED_EVENT_SCHEMA_V1.to_string(),
            ingest_seq,
            event_uuid: Uuid::from_u128(u128::from(ingest_seq) + 10),
            parent_uuid: None,
            kind: CapturedEventKind::Mark,
            scope_phase: None,
            category: None,
            call_role: None,
            timestamp: Utc::now(),
            name: "checkpoint".to_string(),
            data: None,
            metadata: None,
            data_schema: None,
            scope_type: None,
            safe_scope_attributes: Vec::new(),
            codec_annotations: CapturedCodecAnnotationsV1::default(),
            canonical_payload_hash: "0".repeat(64),
            canonical_size_bytes: 1,
        })
    }

    fn complete_horizon() -> JudgeHorizonV1 {
        JudgeHorizonV1::new(1, 1, TrajectoryTrigger::ProgressReached, false).unwrap()
    }

    fn policy() -> JudgePolicyIdentityV1 {
        JudgePolicyIdentityV1::from_config(&judge_config()).unwrap()
    }

    fn input_with_text(text: &str) -> Result<PairwiseJudgeInputV1, JudgeInputErrorV1> {
        PairwiseJudgeInputV1::new(
            &request_projection(text),
            &response_projection("anchor response"),
            &response_projection("candidate response"),
            &[],
            complete_horizon(),
            policy(),
        )
    }

    #[test]
    fn registry_verifies_exact_templates_hashes_and_schema() {
        let config = judge_config();
        let registry = JudgeRegistryV1::load(&config).unwrap();
        assert_eq!(registry.prompt().as_bytes(), JUDGE_PROMPT_TEMPLATE_V1);
        assert_eq!(registry.rubric().as_bytes(), JUDGE_RUBRIC_TEMPLATE_V1);
        assert_eq!(registry.output_schema().as_bytes(), JUDGE_OUTPUT_SCHEMA_V1);
        assert_eq!(
            sha256_hex(registry.prompt().as_bytes()),
            JUDGE_PROMPT_TEMPLATE_SHA256_V1
        );
        assert_eq!(
            sha256_hex(registry.rubric().as_bytes()),
            JUDGE_RUBRIC_TEMPLATE_SHA256_V1
        );
        assert_eq!(
            sha256_hex(registry.output_schema().as_bytes()),
            JUDGE_OUTPUT_SCHEMA_SHA256_V1
        );

        let schema: Json = serde_json::from_str(registry.output_schema()).unwrap();
        assert_eq!(schema["$id"], PAIRWISE_JUDGE_RESULT_SCHEMA_V1);
        assert_eq!(schema["additionalProperties"], false);
        assert_eq!(schema["properties"].as_object().unwrap().len(), 5);
        assert_eq!(schema["required"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn registry_rejects_every_unregistered_identity() {
        let mut config = judge_config();
        config.version = 2;
        assert!(matches!(
            JudgeRegistryV1::load(&config),
            Err(JudgeRegistryErrorV1::UnsupportedConfigVersion)
        ));

        let mut config = judge_config();
        config.prompt_version = "pairwise-equivalence-v2".to_string();
        assert!(matches!(
            JudgeRegistryV1::load(&config),
            Err(JudgeRegistryErrorV1::UnsupportedPromptVersion)
        ));

        let mut config = judge_config();
        config.rubric_version = "response-only-v1".to_string();
        assert!(matches!(
            JudgeRegistryV1::load(&config),
            Err(JudgeRegistryErrorV1::UnsupportedRubricVersion)
        ));

        let mut config = judge_config();
        config.output_schema_version = 2;
        assert!(matches!(
            JudgeRegistryV1::load(&config),
            Err(JudgeRegistryErrorV1::UnsupportedOutputSchemaVersion)
        ));
    }

    #[test]
    fn input_serializes_only_the_explicit_safe_contract() {
        let request = request_projection("safe task");
        let anchor = response_projection("anchor response");
        let candidate = response_projection("candidate response");
        let input = PairwiseJudgeInputV1::new(
            &request,
            &anchor,
            &candidate,
            &[],
            complete_horizon(),
            policy(),
        )
        .unwrap();
        let first = input.canonical_bytes().unwrap();
        let second = input.canonical_bytes().unwrap();
        assert_eq!(first, second);
        assert_eq!(input.family(), LlmApiFamily::OpenAIChatCompletions);
        assert_eq!(input.canonical_sha256().unwrap(), sha256_hex(&first));
        assert!(first.len() <= MAX_PAIRWISE_JUDGE_INPUT_BYTES);

        let value: Json = serde_json::from_slice(&first).unwrap();
        let keys = value.as_object().unwrap().keys().collect::<Vec<_>>();
        assert_eq!(
            keys,
            [
                "anchor_response",
                "candidate_response",
                "horizon",
                "policy",
                "request",
                "schema",
                "trajectory",
            ]
        );
        let serialized = String::from_utf8(first).unwrap();
        for forbidden in [
            "anchor_id",
            "candidate_id",
            "config_generation",
            "learning_generation",
            "pool_id",
            "project_id",
            "replay",
            "routing_context",
            "transport",
        ] {
            assert!(!serialized.contains(forbidden), "{forbidden}");
        }
    }

    #[test]
    fn horizon_requires_exact_terminal_progress_semantics() {
        assert!(JudgeHorizonV1::new(3, 3, TrajectoryTrigger::ProgressReached, false).is_ok());
        for trigger in [
            TrajectoryTrigger::OwnerTerminated,
            TrajectoryTrigger::DeadlineElapsed,
            TrajectoryTrigger::Shutdown,
        ] {
            assert!(JudgeHorizonV1::new(3, 2, trigger, true).is_ok());
            assert_eq!(
                JudgeHorizonV1::new(3, 3, trigger, true),
                Err(JudgeInputErrorV1::InvalidHorizon)
            );
        }
        for invalid in [
            JudgeHorizonV1::new(0, 0, TrajectoryTrigger::ProgressReached, false),
            JudgeHorizonV1::new(3, 2, TrajectoryTrigger::ProgressReached, false),
            JudgeHorizonV1::new(3, 3, TrajectoryTrigger::ProgressReached, true),
            JudgeHorizonV1::new(3, 2, TrajectoryTrigger::DeadlineElapsed, false),
        ] {
            assert_eq!(invalid, Err(JudgeInputErrorV1::InvalidHorizon));
        }
    }

    #[test]
    fn event_order_is_strict_and_canonical() {
        let request = request_projection("safe task");
        let anchor = response_projection("anchor response");
        let candidate = response_projection("candidate response");
        let build = |events: &[Arc<CapturedTrajectoryEvent>]| {
            PairwiseJudgeInputV1::new(
                &request,
                &anchor,
                &candidate,
                events,
                complete_horizon(),
                policy(),
            )
        };
        assert!(build(&[event(1), event(2)]).is_ok());
        assert!(matches!(
            build(&[event(2), event(1)]),
            Err(JudgeInputErrorV1::EventsOutOfOrder)
        ));
        assert!(matches!(
            build(&[event(1), event(1)]),
            Err(JudgeInputErrorV1::EventsOutOfOrder)
        ));
    }

    #[test]
    fn canonical_input_enforces_the_exact_sixteen_mib_boundary() {
        let base = input_with_text("").unwrap();
        let base_size = base.canonical_bytes().unwrap().len();
        let payload_bytes = MAX_PAIRWISE_JUDGE_INPUT_BYTES - base_size;

        let exact = input_with_text(&"x".repeat(payload_bytes)).unwrap();
        assert_eq!(
            exact.canonical_bytes().unwrap().len(),
            MAX_PAIRWISE_JUDGE_INPUT_BYTES
        );
        assert!(matches!(
            input_with_text(&"x".repeat(payload_bytes + 1)),
            Err(JudgeInputErrorV1::TooLarge {
                actual,
                maximum: MAX_PAIRWISE_JUDGE_INPUT_BYTES,
            }) if actual == MAX_PAIRWISE_JUDGE_INPUT_BYTES + 1
        ));
    }

    #[test]
    fn free_text_detector_finds_embedded_credentials_without_prose_false_positives() {
        for sensitive in [
            "prefix sk-12345678901234567890 suffix",
            "JWT eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature123.",
            "key AKIA1234567890ABCDEF,",
            "credential=sk-12345678901234567890",
            "opaque=sk-12345678901234567890",
            "JWT=eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature123",
            "key=AKIA1234567890ABCDEF",
            "-----BEGIN PRIVATE KEY----- material -----END PRIVATE KEY-----",
            "visit https://user:pass@example.test/path",
            "prefixhttps://user:pass@example.test/path",
            "api_key=not-safe",
            "api_key = not-safe",
            "api_key\u{00a0}=\u{2003}not-safe",
            r"key=AKIA1234567890ABCDEF\q",
            "Bearer abcDEF0123456789xyz-_",
            "Bearer abcdefghijklmnopqrstuv",
            "Bearer:abcdefghijklmnopqrstuvwxyz",
            "Bearer : abcdefghijklmnopqrstuv",
            "Bearer\u{00a0}:\u{2003}abcdefghijklmnopqrstuv",
            "Basic dXNlcjpwYXNzd29yZA==",
            "Basic abcdefghijklmnop",
            "Basic:abcdefghijklmnop",
            "Basic : abcdefghijklmnop",
        ] {
            assert!(contains_sensitive_free_text(sensitive), "{sensitive}");
            assert!(matches!(
                input_with_text(sensitive),
                Err(JudgeInputErrorV1::SensitiveValue)
            ));
        }

        for near_miss in [
            "a basic explanation",
            "the bearer of responsibility",
            "token budget",
            "sketch a token budget",
            "use the authorization policy",
            "a secretariat meeting",
            "https://example.test/safe",
            "AKIA is only a four-letter example",
        ] {
            assert!(!contains_sensitive_free_text(near_miss), "{near_miss}");
            assert!(input_with_text(near_miss).is_ok(), "{near_miss}");
        }
    }

    #[test]
    fn recursive_scan_rejects_candidate_and_event_values() {
        let request = request_projection("safe task");
        let anchor = response_projection("anchor response");
        let candidate = response_projection("Bearer abcDEF0123456789xyz-_");
        assert!(matches!(
            PairwiseJudgeInputV1::new(
                &request,
                &anchor,
                &candidate,
                &[],
                complete_horizon(),
                policy(),
            ),
            Err(JudgeInputErrorV1::SensitiveValue)
        ));

        let mut sensitive_event = event(1).as_ref().clone();
        sensitive_event.data = Some(serde_json::json!({
            "safe_key": "github_pat_12345678901234567890"
        }));
        assert!(matches!(
            PairwiseJudgeInputV1::new(
                &request,
                &anchor,
                &response_projection("candidate response"),
                &[Arc::new(sensitive_event)],
                complete_horizon(),
                policy(),
            ),
            Err(JudgeInputErrorV1::SensitiveValue)
        ));

        let mut sensitive_key_event = event(1).as_ref().clone();
        sensitive_key_event.data = Some(serde_json::json!({
            "api_key": "otherwise safe"
        }));
        assert!(matches!(
            PairwiseJudgeInputV1::new(
                &request,
                &anchor,
                &response_projection("candidate response"),
                &[Arc::new(sensitive_key_event)],
                complete_horizon(),
                policy(),
            ),
            Err(JudgeInputErrorV1::SensitiveValue)
        ));
    }
}
