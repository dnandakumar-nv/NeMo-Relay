// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Lossless provider-family request envelopes and model-only rewriting.

use std::fmt;

use nemo_relay::api::llm::{LlmApiFamily, LlmRequest};
use nemo_relay::codec::anthropic::AnthropicMessagesCodec;
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay::codec::openai_responses::OpenAIResponsesCodec;
use nemo_relay::codec::request::{
    AnnotatedLlmRequest, GenerationParams, Message, MessageContent, StructuredResponseFormat,
    StructuredResponseFormatKind,
};
use nemo_relay::codec::traits::LlmCodec;
use serde::Serialize;

use crate::config::JudgeConfig;
use crate::eligibility::IneligibilityReason;
use crate::fingerprint::{
    canonical_serialize_bytes, fingerprint_serializable_bounded, validate_json_resource_bounds,
};
use crate::judge::{JudgeRegistryV1, JudgeRepairEvidenceV1, PairwiseJudgeInputV1};
use crate::projection::{
    REQUEST_JSON_MAX_DEPTH, REQUEST_JSON_MAX_VALUES, REQUEST_PROJECTION_MAX_BYTES,
};

const RESPONSES_UNPARSED_INPUT_KEY: &str = "_openai_responses_unparsed_input_items";
const ANTHROPIC_UNSUPPORTED_SYSTEM_KEY: &str = "_anthropic_messages_unsupported_system";
#[allow(dead_code)] // The evaluator scheduler consumes this Task 4 contract in Task 7.
const JUDGE_RESPONSE_FORMAT_NAME_V1: &str = "pairwise_judge_result_v1";
#[allow(dead_code)] // The evaluator scheduler consumes this Task 4 contract in Task 7.
const JUDGE_REPAIR_INSTRUCTION_V1: &str = "The previous judge output was invalid. Return only a corrected version-1 structured judge result.";

/// Instruction role retained in provider wire order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RouterInstructionRole {
    /// A system instruction.
    System,
    /// A developer instruction.
    Developer,
}

/// One text instruction retained with its normalized wire ordinal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RouterInstructionFact {
    /// Zero-based position in the normalized provider instruction/message order.
    pub(crate) wire_ordinal: usize,
    /// Original instruction role.
    pub(crate) role: RouterInstructionRole,
    /// Exact instruction text.
    pub(crate) content: String,
    /// Optional provider message name.
    pub(crate) name: Option<String>,
}

/// Memory-only lossless request view retained while replay remains live.
#[derive(Clone, PartialEq)]
pub(crate) struct RouterRequestEnvelope {
    /// Authoritative managed-call family.
    pub(crate) family: LlmApiFamily,
    /// Original request, including host-owned headers and provider extras.
    pub(crate) original_request: LlmRequest,
    /// Provider-neutral request decoded by the matching Core codec.
    pub(crate) normalized_request: AnnotatedLlmRequest,
    /// System and Developer instructions in provider wire order.
    pub(crate) ordered_instructions: Vec<RouterInstructionFact>,
    /// Semantic structured response contract, when supported.
    pub(crate) response_format: Option<StructuredResponseFormat>,
    /// RFC 8785 SHA-256 of the complete response JSON Schema.
    pub(crate) response_schema_fingerprint: Option<String>,
}

impl fmt::Debug for RouterRequestEnvelope {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RouterRequestEnvelope")
            .field("family", &self.family)
            .field("original_request", &"<redacted>")
            .field("model", &self.normalized_request.model)
            .field("message_count", &self.normalized_request.messages.len())
            .field("instruction_count", &self.ordered_instructions.len())
            .field("has_response_format", &self.response_format.is_some())
            .finish()
    }
}

/// Built-in family adapter that delegates all wire transforms to Core codecs.
#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct FamilyAdapter;

impl FamilyAdapter {
    /// Decode one request and prove a lossless baseline codec round trip.
    pub(crate) fn decode(
        &self,
        family: LlmApiFamily,
        request: &LlmRequest,
    ) -> Result<RouterRequestEnvelope, IneligibilityReason> {
        let normalized = decode_request(family, request)?;
        let envelope = build_envelope(family, request.clone(), normalized)?;
        let encoded = encode_request(family, &envelope.normalized_request, request)?;
        if encoded != *request {
            return Err(IneligibilityReason::CodecRoundTrip);
        }
        let round_trip =
            build_envelope(family, encoded.clone(), decode_request(family, &encoded)?)?;
        if !envelope_semantics_equal(&envelope, &round_trip) {
            return Err(IneligibilityReason::CodecRoundTrip);
        }
        Ok(envelope)
    }

    /// Re-encode a cloned request after changing only its normalized model.
    pub(crate) fn with_model(
        &self,
        envelope: &RouterRequestEnvelope,
        model: &str,
    ) -> Result<LlmRequest, IneligibilityReason> {
        let mut normalized = envelope.normalized_request.clone();
        normalized.model = Some(model.to_string());
        let candidate = encode_request(envelope.family, &normalized, &envelope.original_request)
            .map_err(|_| IneligibilityReason::CandidateRewrite)?;

        if candidate.headers != envelope.original_request.headers {
            return Err(IneligibilityReason::CandidateRewrite);
        }
        let mut expected_content = envelope.original_request.content.clone();
        let expected_object = expected_content
            .as_object_mut()
            .ok_or(IneligibilityReason::CandidateRewrite)?;
        expected_object.insert("model".into(), serde_json::Value::String(model.to_string()));
        if candidate.content != expected_content {
            return Err(IneligibilityReason::CandidateRewrite);
        }

        let candidate_envelope = self.decode(envelope.family, &candidate)?;
        if candidate_envelope.normalized_request != normalized
            || candidate_envelope.ordered_instructions != envelope.ordered_instructions
            || candidate_envelope.response_format != envelope.response_format
            || candidate_envelope.response_schema_fingerprint
                != envelope.response_schema_fingerprint
        {
            return Err(IneligibilityReason::CandidateRewrite);
        }
        Ok(candidate)
    }

    /// Build a fresh stateless judge request through the matching Core codec.
    #[allow(dead_code)] // The evaluator scheduler consumes this Task 4 contract in Task 7.
    pub(crate) fn build_judge_request(
        &self,
        family: LlmApiFamily,
        config: &JudgeConfig,
        registry: &JudgeRegistryV1,
        input: &PairwiseJudgeInputV1,
    ) -> Result<LlmRequest, IneligibilityReason> {
        if family != input.family() {
            return Err(IneligibilityReason::ReplayFamilyMismatch);
        }
        let user_message = input
            .canonical_bytes()
            .map_err(|_| IneligibilityReason::RuntimeFailure)
            .and_then(|bytes| {
                String::from_utf8(bytes).map_err(|_| IneligibilityReason::RuntimeFailure)
            })?;
        encode_judge_request(family, config, registry, user_message)
    }

    /// Build the sole repair request with the original input and safe invalid-output evidence.
    #[allow(dead_code)] // The evaluator scheduler consumes this Task 4 contract in Task 7.
    pub(crate) fn build_judge_repair_request(
        &self,
        family: LlmApiFamily,
        config: &JudgeConfig,
        registry: &JudgeRegistryV1,
        input: &PairwiseJudgeInputV1,
        repair: &JudgeRepairEvidenceV1,
    ) -> Result<LlmRequest, IneligibilityReason> {
        if family != input.family() {
            return Err(IneligibilityReason::ReplayFamilyMismatch);
        }
        let payload = JudgeRepairUserPayloadV1 {
            instruction: JUDGE_REPAIR_INSTRUCTION_V1,
            pairwise_input: input,
            repair_evidence: repair,
        };
        let user_message = canonical_serialize_bytes(&payload)
            .map_err(|_| IneligibilityReason::RuntimeFailure)
            .and_then(|bytes| {
                String::from_utf8(bytes).map_err(|_| IneligibilityReason::RuntimeFailure)
            })?;
        encode_judge_request(family, config, registry, user_message)
    }
}

#[derive(Serialize)]
#[allow(dead_code)] // The evaluator scheduler consumes this Task 4 contract in Task 7.
struct JudgeRepairUserPayloadV1<'a> {
    instruction: &'static str,
    pairwise_input: &'a PairwiseJudgeInputV1,
    repair_evidence: &'a JudgeRepairEvidenceV1,
}

#[allow(dead_code)] // The evaluator scheduler consumes this Task 4 contract in Task 7.
fn encode_judge_request(
    family: LlmApiFamily,
    config: &JudgeConfig,
    registry: &JudgeRegistryV1,
    user_message: String,
) -> Result<LlmRequest, IneligibilityReason> {
    let token_limit = config
        .output_token_limit()
        .and_then(|value| u64::try_from(value).ok())
        .ok_or(IneligibilityReason::RuntimeFailure)?;
    let output_schema = serde_json::from_str(registry.output_schema())
        .map_err(|_| IneligibilityReason::RuntimeFailure)?;
    let (format_name, strict) = match family {
        LlmApiFamily::OpenAIChatCompletions | LlmApiFamily::OpenAIResponses => {
            (Some(JUDGE_RESPONSE_FORMAT_NAME_V1.to_string()), Some(true))
        }
        LlmApiFamily::AnthropicMessages => (None, None),
    };

    let mut system_message =
        String::with_capacity(registry.prompt().len() + registry.rubric().len());
    system_message.push_str(registry.prompt());
    system_message.push_str(registry.rubric());
    let normalized = AnnotatedLlmRequest {
        messages: vec![
            Message::System {
                content: MessageContent::Text(system_message),
                name: None,
            },
            Message::User {
                content: MessageContent::Text(user_message),
                name: None,
            },
        ],
        model: Some(config.model.clone()),
        params: Some(GenerationParams {
            temperature: Some(config.temperature.unwrap_or(0.0)),
            max_tokens: Some(token_limit),
            top_p: None,
            stop: None,
        }),
        tools: None,
        tool_choice: None,
        response_format: Some(StructuredResponseFormat {
            kind: StructuredResponseFormatKind::JsonSchema,
            name: format_name,
            schema: Some(output_schema),
            strict,
            extra: serde_json::Map::new(),
        }),
        store: None,
        previous_response_id: None,
        truncation: None,
        reasoning: None,
        include: None,
        user: None,
        metadata: None,
        service_tier: None,
        parallel_tool_calls: None,
        max_output_tokens: None,
        max_tool_calls: None,
        top_logprobs: None,
        stream: None,
        extra: serde_json::Map::new(),
    };
    let empty = LlmRequest {
        headers: serde_json::Map::new(),
        content: serde_json::Value::Object(serde_json::Map::new()),
    };
    let request = encode_request(family, &normalized, &empty)
        .map_err(|_| IneligibilityReason::RuntimeFailure)?;
    if request.headers.is_empty() {
        Ok(request)
    } else {
        Err(IneligibilityReason::RuntimeFailure)
    }
}

fn decode_request(
    family: LlmApiFamily,
    request: &LlmRequest,
) -> Result<AnnotatedLlmRequest, IneligibilityReason> {
    let result = match family {
        LlmApiFamily::OpenAIChatCompletions => OpenAIChatCodec.decode(request),
        LlmApiFamily::OpenAIResponses => OpenAIResponsesCodec.decode(request),
        LlmApiFamily::AnthropicMessages => AnthropicMessagesCodec.decode(request),
    };
    result.map_err(|_| IneligibilityReason::CodecDecode)
}

fn encode_request(
    family: LlmApiFamily,
    normalized: &AnnotatedLlmRequest,
    original: &LlmRequest,
) -> Result<LlmRequest, IneligibilityReason> {
    let result = match family {
        LlmApiFamily::OpenAIChatCompletions => OpenAIChatCodec.encode(normalized, original),
        LlmApiFamily::OpenAIResponses => OpenAIResponsesCodec.encode(normalized, original),
        LlmApiFamily::AnthropicMessages => AnthropicMessagesCodec.encode(normalized, original),
    };
    result.map_err(|_| IneligibilityReason::CodecRoundTrip)
}

fn build_envelope(
    family: LlmApiFamily,
    original_request: LlmRequest,
    normalized_request: AnnotatedLlmRequest,
) -> Result<RouterRequestEnvelope, IneligibilityReason> {
    reject_generic_response_format(family, &normalized_request)?;
    if normalized_request
        .extra
        .contains_key(RESPONSES_UNPARSED_INPUT_KEY)
        || (family == LlmApiFamily::AnthropicMessages
            && normalized_request
                .extra
                .contains_key(ANTHROPIC_UNSUPPORTED_SYSTEM_KEY))
    {
        return Err(IneligibilityReason::UnsupportedInstructions);
    }

    let ordered_instructions = collect_instruction_facts(&normalized_request.messages)?;
    let response_format = normalized_request.response_format.clone();
    let response_schema_fingerprint = response_format
        .as_ref()
        .and_then(|format| format.schema.as_ref())
        .map(|schema| {
            validate_json_resource_bounds(
                schema,
                REQUEST_PROJECTION_MAX_BYTES,
                REQUEST_JSON_MAX_VALUES,
                REQUEST_JSON_MAX_DEPTH,
            )?;
            fingerprint_serializable_bounded(schema, REQUEST_PROJECTION_MAX_BYTES).map_err(|_| ())
        })
        .transpose()
        .map_err(|_| IneligibilityReason::UnsupportedResponseFormat)?;

    Ok(RouterRequestEnvelope {
        family,
        original_request,
        normalized_request,
        ordered_instructions,
        response_format,
        response_schema_fingerprint,
    })
}

fn collect_instruction_facts(
    messages: &[Message],
) -> Result<Vec<RouterInstructionFact>, IneligibilityReason> {
    messages
        .iter()
        .enumerate()
        .filter_map(|(wire_ordinal, message)| {
            let (role, content, name) = match message {
                Message::System { content, name } => (RouterInstructionRole::System, content, name),
                Message::Developer { content, name } => {
                    (RouterInstructionRole::Developer, content, name)
                }
                _ => return None,
            };
            Some(match content {
                MessageContent::Text(content) => Ok(RouterInstructionFact {
                    wire_ordinal,
                    role,
                    content: content.clone(),
                    name: name.clone(),
                }),
                MessageContent::Parts(_) => Err(IneligibilityReason::UnsupportedInstructions),
            })
        })
        .collect()
}

fn reject_generic_response_format(
    family: LlmApiFamily,
    normalized: &AnnotatedLlmRequest,
) -> Result<(), IneligibilityReason> {
    if normalized.response_format.is_some() {
        return Ok(());
    }
    let native_key = match family {
        LlmApiFamily::OpenAIChatCompletions => "response_format",
        LlmApiFamily::OpenAIResponses => "text",
        LlmApiFamily::AnthropicMessages => "output_config",
    };
    let has_unknown_format = match (family, normalized.extra.get(native_key)) {
        (LlmApiFamily::OpenAIChatCompletions, Some(_)) => true,
        (LlmApiFamily::OpenAIResponses, Some(value)) => value.get("format").is_some(),
        (LlmApiFamily::AnthropicMessages, Some(value)) => value.get("format").is_some(),
        _ => false,
    };
    if has_unknown_format {
        Err(IneligibilityReason::UnsupportedResponseFormat)
    } else {
        Ok(())
    }
}

fn envelope_semantics_equal(left: &RouterRequestEnvelope, right: &RouterRequestEnvelope) -> bool {
    left.family == right.family
        && left.normalized_request == right.normalized_request
        && left.ordered_instructions == right.ordered_instructions
        && left.response_format == right.response_format
        && left.response_schema_fingerprint == right.response_schema_fingerprint
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use nemo_relay::api::llm::{LlmApiFamily, LlmRequest};
    use serde_json::{Map, Value as Json, json};

    use super::{
        FamilyAdapter, JUDGE_REPAIR_INSTRUCTION_V1, JUDGE_RESPONSE_FORMAT_NAME_V1,
        RouterRequestEnvelope,
    };
    use crate::config::{
        JUDGE_OUTPUT_SCHEMA_VERSION, JUDGE_PROMPT_VERSION_V1, JUDGE_RUBRIC_VERSION_V1, JudgeConfig,
    };
    use crate::eligibility::IneligibilityReason;
    use crate::judge::{
        JudgeAttemptV1, JudgeHorizonV1, JudgePolicyIdentityV1, JudgeRegistryV1,
        JudgeRepairEvidenceV1, PairwiseJudgeInputV1, resolve_judge_attempt_progress,
        validate_pairwise_judge_output,
    };
    use crate::projection::{
        REQUEST_PROJECTION_SCHEMA_V1, ROUTER_SANITIZER_VERSION, RouterRequestProjectionV1,
        SanitizedAnnotatedLlmRequest, SanitizedMessage, SanitizedMessageContent,
    };
    use crate::trajectory::{
        RESPONSE_PROJECTION_SCHEMA_V1, RouterResponseProjectionV1, TRAJECTORY_SANITIZER_VERSION,
        TrajectoryTrigger,
    };

    macro_rules! assert_not_impl {
        ($type:ty: $trait:path) => {
            const _: fn() = || {
                trait AmbiguousIfImpl<Marker> {
                    fn marker() {}
                }
                impl<T: ?Sized> AmbiguousIfImpl<()> for T {}
                struct ImplementsTrait;
                impl<T: ?Sized + $trait> AmbiguousIfImpl<ImplementsTrait> for T {}
                let _ = <$type as AmbiguousIfImpl<_>>::marker;
            };
        };
    }

    assert_not_impl!(RouterRequestEnvelope: serde::Serialize);

    fn request(content: Json) -> LlmRequest {
        LlmRequest {
            headers: Map::from_iter([(
                "authorization".to_string(),
                Json::String("Bearer must-not-appear".to_string()),
            )]),
            content,
        }
    }

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

    fn request_projection(family: LlmApiFamily) -> RouterRequestProjectionV1 {
        RouterRequestProjectionV1 {
            schema: REQUEST_PROJECTION_SCHEMA_V1.to_string(),
            family,
            normalized_request: SanitizedAnnotatedLlmRequest {
                messages: vec![SanitizedMessage::User {
                    content: SanitizedMessageContent::Text("safe task".to_string()),
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
            required_capabilities: Vec::new(),
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
            semantic_response_fingerprint: format!("{text}-fingerprint"),
        }
    }

    fn judge_input(family: LlmApiFamily, config: &JudgeConfig) -> PairwiseJudgeInputV1 {
        PairwiseJudgeInputV1::new(
            &request_projection(family),
            &response_projection("anchor"),
            &response_projection("candidate"),
            &[],
            JudgeHorizonV1::new(1, 1, TrajectoryTrigger::ProgressReached, false).unwrap(),
            JudgePolicyIdentityV1::from_config(config).unwrap(),
        )
        .unwrap()
    }

    fn repair_evidence(raw: &str, max_rationale_bytes: usize) -> JudgeRepairEvidenceV1 {
        let initial =
            JudgeAttemptV1::initial(validate_pairwise_judge_output(raw, max_rationale_bytes));
        let (evidence, _pending) = resolve_judge_attempt_progress(initial)
            .unwrap()
            .into_repair()
            .unwrap();
        evidence
    }

    fn system_message(registry: &JudgeRegistryV1) -> String {
        format!("{}{}", registry.prompt(), registry.rubric())
    }

    fn golden_judge_content(
        family: LlmApiFamily,
        system: &str,
        user: &str,
        schema: &Json,
        token_limit: u64,
    ) -> Json {
        match family {
            LlmApiFamily::OpenAIChatCompletions => json!({
                "model": "judge-model",
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "temperature": 0.0,
                "max_tokens": token_limit,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": JUDGE_RESPONSE_FORMAT_NAME_V1,
                        "schema": schema,
                        "strict": true
                    }
                }
            }),
            LlmApiFamily::OpenAIResponses => json!({
                "model": "judge-model",
                "input": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                "temperature": 0.0,
                "max_output_tokens": token_limit,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": JUDGE_RESPONSE_FORMAT_NAME_V1,
                        "schema": schema,
                        "strict": true
                    }
                }
            }),
            LlmApiFamily::AnthropicMessages => json!({
                "model": "judge-model",
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "temperature": 0.0,
                "max_tokens": token_limit,
                "output_config": {
                    "format": {
                        "type": "json_schema",
                        "schema": schema
                    }
                }
            }),
        }
    }

    fn encoded_user_message(family: LlmApiFamily, content: &Json) -> &str {
        match family {
            LlmApiFamily::OpenAIChatCompletions => {
                content["messages"][1]["content"].as_str().unwrap()
            }
            LlmApiFamily::OpenAIResponses => content["input"][1]["content"].as_str().unwrap(),
            LlmApiFamily::AnthropicMessages => content["messages"][0]["content"].as_str().unwrap(),
        }
    }

    #[test]
    fn chat_model_rewrite_preserves_instructions_format_and_unknowns() {
        let original = request(json!({
            "model": "anchor-chat",
            "messages": [
                {"role": "system", "content": "system-secret"},
                {"role": "developer", "content": "developer-secret", "name": "policy"},
                {"role": "user", "content": "task-secret"}
            ],
            "response_format": {
                "type": "json_schema",
                "trace": "outer",
                "json_schema": {
                    "name": "answer",
                    "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}},
                    "strict": true,
                    "description": "inner"
                }
            },
            "vendor_extension": {"keep": true}
        }));
        let adapter = FamilyAdapter;
        let envelope = adapter
            .decode(LlmApiFamily::OpenAIChatCompletions, &original)
            .unwrap();
        assert_eq!(envelope.ordered_instructions.len(), 2);
        assert_eq!(
            envelope
                .response_schema_fingerprint
                .as_deref()
                .map(str::len),
            Some(64)
        );

        let candidate = adapter.with_model(&envelope, "candidate-chat").unwrap();
        let mut expected = original.clone();
        expected.content["model"] = json!("candidate-chat");
        assert_eq!(candidate, expected);

        let debug = format!("{envelope:?}");
        assert!(debug.contains("<redacted>"));
        for secret in [
            "must-not-appear",
            "system-secret",
            "developer-secret",
            "task-secret",
        ] {
            assert!(!debug.contains(secret));
        }
    }

    #[test]
    fn responses_model_rewrite_preserves_family_and_input_instructions() {
        let original = request(json!({
            "model": "anchor-responses",
            "instructions": "family-system",
            "input": [
                {"role": "system", "content": "input-system"},
                {"role": "developer", "content": "input-developer"},
                {"role": "user", "content": "task"}
            ],
            "text": {
                "verbosity": "low",
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "schema": {"type": "object"},
                    "strict": false,
                    "vendor": "keep"
                }
            },
            "reasoning": {"effort": "low"},
            "vendor_extension": 7
        }));
        let adapter = FamilyAdapter;
        let envelope = adapter
            .decode(LlmApiFamily::OpenAIResponses, &original)
            .unwrap();
        assert_eq!(envelope.ordered_instructions.len(), 3);
        let candidate = adapter
            .with_model(&envelope, "candidate-responses")
            .unwrap();
        let mut expected = original.clone();
        expected.content["model"] = json!("candidate-responses");
        assert_eq!(candidate, expected);
    }

    #[test]
    fn responses_string_input_remains_a_string_for_model_rewrite() {
        let original = request(json!({
            "model": "anchor-responses",
            "instructions": "system",
            "input": "plain task"
        }));
        let adapter = FamilyAdapter;
        let envelope = adapter
            .decode(LlmApiFamily::OpenAIResponses, &original)
            .unwrap();
        let candidate = adapter
            .with_model(&envelope, "candidate-responses")
            .unwrap();
        assert_eq!(candidate.content["input"], json!("plain task"));
        assert_eq!(candidate.content["model"], json!("candidate-responses"));
    }

    #[test]
    fn anthropic_model_rewrite_preserves_system_format_and_unknowns() {
        let original = request(json!({
            "model": "anchor-anthropic",
            "system": "system instruction",
            "messages": [{"role": "user", "content": "task"}],
            "max_tokens": 128,
            "output_config": {
                "effort": "low",
                "format": {
                    "type": "json_schema",
                    "schema": {"type": "object", "required": ["answer"]},
                    "vendor": "keep"
                }
            },
            "vendor_extension": {"keep": true}
        }));
        let adapter = FamilyAdapter;
        let envelope = adapter
            .decode(LlmApiFamily::AnthropicMessages, &original)
            .unwrap();
        assert_eq!(envelope.ordered_instructions.len(), 1);
        let candidate = adapter
            .with_model(&envelope, "candidate-anthropic")
            .unwrap();
        let mut expected = original.clone();
        expected.content["model"] = json!("candidate-anthropic");
        assert_eq!(candidate, expected);
    }

    #[test]
    fn anthropic_unsupported_system_blocks_are_ineligible() {
        for system in [
            json!([
                {"type": "text", "text": "first"},
                {"type": "text", "text": "second"}
            ]),
            json!([
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "abcd"
                    }
                },
                {"type": "text", "text": "instruction"}
            ]),
        ] {
            let original = request(json!({
                "model": "anchor-anthropic",
                "system": system,
                "messages": [{"role": "user", "content": "task"}],
                "max_tokens": 128
            }));
            assert_eq!(
                FamilyAdapter.decode(LlmApiFamily::AnthropicMessages, &original),
                Err(IneligibilityReason::UnsupportedInstructions)
            );
        }
    }

    #[test]
    fn unsupported_response_format_is_ineligible_without_rewrite() {
        let original = request(json!({
            "model": "anchor",
            "messages": [{"role": "user", "content": "task"}],
            "response_format": {"type": "text"}
        }));
        assert_eq!(
            FamilyAdapter.decode(LlmApiFamily::OpenAIChatCompletions, &original),
            Err(IneligibilityReason::UnsupportedResponseFormat)
        );
    }

    #[test]
    fn instruction_parts_are_ineligible_instead_of_merged() {
        let original = request(json!({
            "model": "anchor",
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": "one"}]},
                {"role": "user", "content": "task"}
            ]
        }));
        assert_eq!(
            FamilyAdapter.decode(LlmApiFamily::OpenAIChatCompletions, &original),
            Err(IneligibilityReason::UnsupportedInstructions)
        );
    }

    #[test]
    fn initial_judge_requests_match_exact_family_goldens() {
        let config = judge_config();
        let registry = JudgeRegistryV1::load(&config).unwrap();
        let schema: Json = serde_json::from_str(registry.output_schema()).unwrap();
        let system = system_message(&registry);
        let token_limit = u64::try_from(config.output_token_limit().unwrap()).unwrap();
        assert_eq!(token_limit, 1_536);

        for family in [
            LlmApiFamily::OpenAIChatCompletions,
            LlmApiFamily::OpenAIResponses,
            LlmApiFamily::AnthropicMessages,
        ] {
            let input = judge_input(family, &config);
            let before = input.canonical_bytes().unwrap();
            let user = String::from_utf8(before.clone()).unwrap();
            let first = FamilyAdapter
                .build_judge_request(family, &config, &registry, &input)
                .unwrap();
            let second = FamilyAdapter
                .build_judge_request(family, &config, &registry, &input)
                .unwrap();

            assert!(first.headers.is_empty());
            assert_eq!(first, second, "fresh {family:?} builds must be stable");
            assert_eq!(
                first.content,
                golden_judge_content(family, &system, &user, &schema, token_limit),
                "{family:?} initial golden changed"
            );
            assert_eq!(input.canonical_bytes().unwrap(), before);
            assert_eq!(encoded_user_message(family, &first.content), user);

            let object = first.content.as_object().unwrap();
            for forbidden in [
                "tools",
                "tool_choice",
                "stream",
                "store",
                "previous_response_id",
                "truncation",
                "reasoning",
                "include",
                "user",
                "metadata",
                "service_tier",
                "parallel_tool_calls",
                "max_tool_calls",
                "top_logprobs",
            ] {
                assert!(
                    !object.contains_key(forbidden),
                    "{family:?} leaked {forbidden}"
                );
            }
        }
    }

    #[test]
    fn configured_judge_temperature_is_encoded_for_every_family() {
        let mut config = judge_config();
        config.temperature = Some(1.0);
        let registry = JudgeRegistryV1::load(&config).unwrap();

        for family in [
            LlmApiFamily::OpenAIChatCompletions,
            LlmApiFamily::OpenAIResponses,
            LlmApiFamily::AnthropicMessages,
        ] {
            let input = judge_input(family, &config);
            let request = FamilyAdapter
                .build_judge_request(family, &config, &registry, &input)
                .unwrap();
            assert_eq!(request.content["temperature"], json!(1.0));
        }
    }

    #[test]
    fn repair_judge_requests_match_exact_family_goldens() {
        let config = judge_config();
        let registry = JudgeRegistryV1::load(&config).unwrap();
        let schema: Json = serde_json::from_str(registry.output_schema()).unwrap();
        let system = system_message(&registry);
        let token_limit = u64::try_from(config.output_token_limit().unwrap()).unwrap();
        let invalid_output = "bounded invalid judge output";

        for family in [
            LlmApiFamily::OpenAIChatCompletions,
            LlmApiFamily::OpenAIResponses,
            LlmApiFamily::AnthropicMessages,
        ] {
            let input = judge_input(family, &config);
            let before = input.canonical_bytes().unwrap();
            let repair = repair_evidence(invalid_output, config.max_rationale_bytes);
            let expected_payload = json!({
                "instruction": JUDGE_REPAIR_INSTRUCTION_V1,
                "pairwise_input": serde_json::to_value(&input).unwrap(),
                "repair_evidence": serde_json::to_value(&repair).unwrap(),
            });
            let expected_user = String::from_utf8(
                crate::fingerprint::canonical_serialize_bytes(&expected_payload).unwrap(),
            )
            .unwrap();
            let request = FamilyAdapter
                .build_judge_repair_request(family, &config, &registry, &input, &repair)
                .unwrap();

            assert!(request.headers.is_empty());
            assert_eq!(
                request.content,
                golden_judge_content(family, &system, &expected_user, &schema, token_limit),
                "{family:?} repair golden changed"
            );
            assert_eq!(input.canonical_bytes().unwrap(), before);
            let actual_user = encoded_user_message(family, &request.content);
            assert_eq!(actual_user, expected_user);
            let payload: Json = serde_json::from_str(actual_user).unwrap();
            assert_eq!(payload["instruction"], JUDGE_REPAIR_INSTRUCTION_V1);
            assert_eq!(
                payload["pairwise_input"],
                serde_json::to_value(&input).unwrap()
            );
            assert_eq!(
                payload["repair_evidence"]["output_schema"],
                registry.output_schema()
            );
            assert_eq!(
                payload["repair_evidence"]["invalid_output"]["output"],
                invalid_output
            );
            assert_eq!(
                payload["repair_evidence"]["validation_issues"],
                json!(["output.invalid_json"])
            );
        }
    }

    #[test]
    fn repair_requests_never_expose_credential_bearing_output() {
        let config = judge_config();
        let registry = JudgeRegistryV1::load(&config).unwrap();
        let credential = "Bearer abcDEF0123456789xyz-_";
        let raw = json!({
            "response_equivalence": 0.9,
            "trajectory_equivalence": 0.8,
            "judge_confidence": 0.95,
            "hard_failures": [],
            "rationale": credential,
        })
        .to_string();
        let repair = repair_evidence(&raw, config.max_rationale_bytes);

        for family in [
            LlmApiFamily::OpenAIChatCompletions,
            LlmApiFamily::OpenAIResponses,
            LlmApiFamily::AnthropicMessages,
        ] {
            let input = judge_input(family, &config);
            let request = FamilyAdapter
                .build_judge_repair_request(family, &config, &registry, &input, &repair)
                .unwrap();
            let encoded = serde_json::to_string(&request.content).unwrap();
            assert!(
                !encoded.contains(credential),
                "{family:?} exposed credential output"
            );
            let payload: Json =
                serde_json::from_str(encoded_user_message(family, &request.content)).unwrap();
            assert_eq!(
                payload["repair_evidence"]["invalid_output"]["marker"],
                "credential_bearing"
            );
            assert!(
                payload["repair_evidence"]["invalid_output"]
                    .get("output")
                    .is_none()
            );
        }
    }

    #[test]
    fn judge_request_rejects_an_unrepresentable_token_limit() {
        let mut config = judge_config();
        let registry = JudgeRegistryV1::load(&config).unwrap();
        let input = judge_input(LlmApiFamily::OpenAIChatCompletions, &config);
        config.max_rationale_bytes = usize::MAX;
        assert_eq!(
            FamilyAdapter.build_judge_request(
                LlmApiFamily::OpenAIChatCompletions,
                &config,
                &registry,
                &input,
            ),
            Err(IneligibilityReason::RuntimeFailure)
        );
    }

    #[test]
    fn judge_builders_reject_every_authoritative_family_mismatch_first() {
        let config = judge_config();
        let registry = JudgeRegistryV1::load(&config).unwrap();
        let repair = repair_evidence("invalid judge output", config.max_rationale_bytes);
        let families = [
            LlmApiFamily::OpenAIChatCompletions,
            LlmApiFamily::OpenAIResponses,
            LlmApiFamily::AnthropicMessages,
        ];

        for authoritative in families {
            let input = judge_input(authoritative, &config);
            assert_eq!(input.family(), authoritative);
            let mut poisoned_config = config.clone();
            poisoned_config.max_rationale_bytes = usize::MAX;
            for supplied in families
                .into_iter()
                .filter(|supplied| *supplied != authoritative)
            {
                assert_eq!(
                    FamilyAdapter.build_judge_request(
                        supplied,
                        &poisoned_config,
                        &registry,
                        &input,
                    ),
                    Err(IneligibilityReason::ReplayFamilyMismatch),
                    "initial {authoritative:?} input accepted as {supplied:?}"
                );
                assert_eq!(
                    FamilyAdapter.build_judge_repair_request(
                        supplied,
                        &poisoned_config,
                        &registry,
                        &input,
                        &repair,
                    ),
                    Err(IneligibilityReason::ReplayFamilyMismatch),
                    "repair {authoritative:?} input accepted as {supplied:?}"
                );
            }
        }
    }
}
