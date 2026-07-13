// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure canonical routing-query construction from sanitized frozen projections.

use nemo_relay::api::llm::LlmApiFamily;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;

use crate::config::CanonicalizerConfig;
use crate::eligibility::IneligibilityReason;
use crate::embedding_identity::CANONICAL_ROUTING_QUERY_SCHEMA_V1;
use crate::fingerprint::{canonical_serialize_bytes, fingerprint_serializable, sha256_hex};
use crate::projection::{
    REQUEST_PROJECTION_SCHEMA_V1, ROUTER_SANITIZER_VERSION, ROUTING_CONTEXT_SCHEMA_V1,
    RouterRequestProjectionV1, RouterRoutingContextProjectionV1, SanitizedContentPart,
    SanitizedMessage, SanitizedMessageContent, SanitizedRouterInstructionRole,
    validate_request_projection,
};

/// One normalized instruction role in an embedding query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum CanonicalInstructionRoleV1 {
    System,
    Developer,
}

/// One normalized System or Developer instruction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CanonicalInstructionV1 {
    pub(crate) role: CanonicalInstructionRoleV1,
    pub(crate) text: String,
}

/// The selected current user task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CanonicalTaskV1 {
    pub(crate) text: String,
}

/// One normalized bounded-context role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum CanonicalContextRoleV1 {
    User,
    Assistant,
    Tool,
}

/// One whole normalized context entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CanonicalContextEntryV1 {
    pub(crate) role: CanonicalContextRoleV1,
    pub(crate) text: String,
}

/// One explicitly configured scalar position feature.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CanonicalPositionFeatureV1 {
    pub(crate) name: String,
    pub(crate) value: Json,
}

/// Version-1 provider-neutral embedding input.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct CanonicalRoutingQueryV1 {
    pub(crate) schema: String,
    pub(crate) instructions: Vec<CanonicalInstructionV1>,
    pub(crate) current_task: CanonicalTaskV1,
    pub(crate) bounded_context: Vec<CanonicalContextEntryV1>,
    pub(crate) tool_schema_fingerprint: String,
    pub(crate) response_schema_fingerprint: Option<String>,
    pub(crate) required_capabilities: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub(crate) position_features: Option<Vec<CanonicalPositionFeatureV1>>,
}

/// Canonical query plus the exact provider input bytes and their digest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CanonicalRoutingQueryArtifactV1 {
    pub(crate) query: CanonicalRoutingQueryV1,
    pub(crate) canonical_bytes: Vec<u8>,
    pub(crate) canonical_query_hash: String,
}

/// Construct the exact version-1 embedding query from frozen safe facts.
pub(crate) fn build_canonical_routing_query(
    request: &RouterRequestProjectionV1,
    routing: &RouterRoutingContextProjectionV1,
    canonicalizer: &CanonicalizerConfig,
) -> Result<CanonicalRoutingQueryArtifactV1, IneligibilityReason> {
    validate_projection(request, routing)?;

    let instructions = canonical_instructions(request, canonicalizer)?;
    let normalized_messages = request
        .normalized_request
        .messages
        .iter()
        .map(normalize_message)
        .collect::<Result<Vec<_>, _>>()?;
    let task_index = normalized_messages
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, message)| {
            matches!(message.role, NormalizedMessageRole::User)
                .then_some(message.text.as_str())
                .filter(|text| !text.is_empty())
                .map(|_| index)
        })
        .ok_or(IneligibilityReason::CanonicalMissingTask)?;
    let task_text = normalized_messages[task_index].text.clone();
    if task_text.len() > canonicalizer.max_task_bytes {
        return Err(IneligibilityReason::CanonicalTaskBound);
    }

    let candidates = normalized_messages
        .into_iter()
        .enumerate()
        .filter_map(|(index, message)| {
            if index == task_index || message.text.is_empty() {
                return None;
            }
            let role = match message.role {
                NormalizedMessageRole::User => CanonicalContextRoleV1::User,
                NormalizedMessageRole::Assistant => CanonicalContextRoleV1::Assistant,
                NormalizedMessageRole::Tool => CanonicalContextRoleV1::Tool,
                NormalizedMessageRole::Instruction => return None,
            };
            Some(CanonicalContextEntryV1 {
                role,
                text: message.text,
            })
        })
        .collect::<Vec<_>>();
    let bounded_context = select_context(candidates, canonicalizer)?;

    let empty_tools = Vec::new();
    let tools = request
        .normalized_request
        .tools
        .as_ref()
        .unwrap_or(&empty_tools);
    let tool_schema_fingerprint =
        fingerprint_serializable(tools).map_err(|_| IneligibilityReason::CanonicalSerialization)?;
    let response_schema_fingerprint = validate_optional_sha256(
        request.response_schema_fingerprint.as_deref(),
        IneligibilityReason::CanonicalSchemaFingerprint,
    )?;
    validate_capabilities(&request.required_capabilities)?;
    let position_features = canonical_position_features(routing, canonicalizer)?;

    let query = CanonicalRoutingQueryV1 {
        schema: CANONICAL_ROUTING_QUERY_SCHEMA_V1.to_string(),
        instructions,
        current_task: CanonicalTaskV1 { text: task_text },
        bounded_context,
        tool_schema_fingerprint,
        response_schema_fingerprint,
        required_capabilities: request.required_capabilities.clone(),
        position_features,
    };
    let canonical_bytes = canonical_serialize_bytes(&query)
        .map_err(|_| IneligibilityReason::CanonicalSerialization)?;
    let canonical_query_hash = sha256_hex(&canonical_bytes);
    Ok(CanonicalRoutingQueryArtifactV1 {
        query,
        canonical_bytes,
        canonical_query_hash,
    })
}

fn validate_projection(
    request: &RouterRequestProjectionV1,
    routing: &RouterRoutingContextProjectionV1,
) -> Result<(), IneligibilityReason> {
    if request.schema != REQUEST_PROJECTION_SCHEMA_V1
        || request.sanitizer_version != ROUTER_SANITIZER_VERSION
        || routing.schema != ROUTING_CONTEXT_SCHEMA_V1
        || !supported_family(request.family)
    {
        return Err(IneligibilityReason::CanonicalProjection);
    }
    validate_request_projection(request).map_err(|_| IneligibilityReason::CanonicalProjection)?;
    for hash in [&routing.tenant_policy_hash, &routing.agent_policy_hash] {
        validate_sha256(hash, IneligibilityReason::CanonicalProjection)?;
    }
    Ok(())
}

const fn supported_family(family: LlmApiFamily) -> bool {
    match family {
        LlmApiFamily::OpenAIChatCompletions
        | LlmApiFamily::OpenAIResponses
        | LlmApiFamily::AnthropicMessages => true,
    }
}

fn canonical_instructions(
    request: &RouterRequestProjectionV1,
    canonicalizer: &CanonicalizerConfig,
) -> Result<Vec<CanonicalInstructionV1>, IneligibilityReason> {
    let mut total_bytes = 0usize;
    request
        .ordered_instructions
        .iter()
        .map(|instruction| {
            let text = normalize_text(&instruction.content);
            total_bytes = total_bytes
                .checked_add(text.len())
                .ok_or(IneligibilityReason::CanonicalInstructionBound)?;
            if total_bytes > canonicalizer.max_instruction_bytes {
                return Err(IneligibilityReason::CanonicalInstructionBound);
            }
            let role = match instruction.role {
                SanitizedRouterInstructionRole::System => CanonicalInstructionRoleV1::System,
                SanitizedRouterInstructionRole::Developer => CanonicalInstructionRoleV1::Developer,
            };
            Ok(CanonicalInstructionV1 { role, text })
        })
        .collect()
}

#[derive(Debug)]
struct NormalizedMessage {
    role: NormalizedMessageRole,
    text: String,
}

#[derive(Debug, Clone, Copy)]
enum NormalizedMessageRole {
    Instruction,
    User,
    Assistant,
    Tool,
}

fn normalize_message(message: &SanitizedMessage) -> Result<NormalizedMessage, IneligibilityReason> {
    let (role, content) = match message {
        SanitizedMessage::System { content, .. } | SanitizedMessage::Developer { content, .. } => {
            (NormalizedMessageRole::Instruction, Some(content))
        }
        SanitizedMessage::User { content, .. } => (NormalizedMessageRole::User, Some(content)),
        SanitizedMessage::Assistant { content, .. } => {
            (NormalizedMessageRole::Assistant, content.as_ref())
        }
        SanitizedMessage::Tool { content, .. } => (NormalizedMessageRole::Tool, Some(content)),
    };
    let text = content
        .map(normalize_content)
        .transpose()?
        .unwrap_or_default();
    Ok(NormalizedMessage { role, text })
}

fn normalize_content(content: &SanitizedMessageContent) -> Result<String, IneligibilityReason> {
    match content {
        SanitizedMessageContent::Text(text) => Ok(normalize_text(text)),
        SanitizedMessageContent::Parts(parts) => {
            let mut text_parts = Vec::new();
            for part in parts {
                match part {
                    SanitizedContentPart::Text { text } => {
                        let text = normalize_text(text);
                        if !text.is_empty() {
                            text_parts.push(text);
                        }
                    }
                    SanitizedContentPart::ImageUrl { .. } => {
                        return Err(IneligibilityReason::CanonicalMultimodal);
                    }
                }
            }
            Ok(text_parts.join("\n"))
        }
    }
}

fn normalize_text(value: &str) -> String {
    let mut line_normalized = String::with_capacity(value.len());
    let mut characters = value.chars().peekable();
    while let Some(character) = characters.next() {
        if character == '\r' {
            if characters.peek() == Some(&'\n') {
                characters.next();
            }
            line_normalized.push('\n');
        } else {
            line_normalized.push(character);
        }
    }

    let mut collapsed = String::with_capacity(line_normalized.len());
    let mut in_horizontal_whitespace = false;
    for character in line_normalized.chars() {
        if character == '\n' {
            collapsed.push(character);
            in_horizontal_whitespace = false;
        } else if is_horizontal_whitespace(character) {
            if !in_horizontal_whitespace {
                collapsed.push(' ');
                in_horizontal_whitespace = true;
            }
        } else {
            collapsed.push(character);
            in_horizontal_whitespace = false;
        }
    }
    collapsed.trim_matches(char::is_whitespace).to_string()
}

fn is_horizontal_whitespace(character: char) -> bool {
    matches!(
        character,
        '\u{0009}' | '\u{0020}' | '\u{00a0}' | '\u{1680}' | '\u{2000}'
            ..='\u{200a}' | '\u{202f}' | '\u{205f}' | '\u{3000}'
    )
}

fn select_context(
    candidates: Vec<CanonicalContextEntryV1>,
    canonicalizer: &CanonicalizerConfig,
) -> Result<Vec<CanonicalContextEntryV1>, IneligibilityReason> {
    let mut selected = Vec::new();
    let mut selected_bytes = 0usize;
    for candidate in candidates.into_iter().rev() {
        if selected.len() == canonicalizer.max_context_messages {
            break;
        }
        let next_bytes = selected_bytes
            .checked_add(candidate.text.len())
            .ok_or(IneligibilityReason::CanonicalContextBound)?;
        if next_bytes > canonicalizer.max_context_bytes {
            break;
        }
        selected.push(candidate);
        selected_bytes = next_bytes;
    }
    selected.reverse();
    Ok(selected)
}

fn validate_capabilities(capabilities: &[String]) -> Result<(), IneligibilityReason> {
    if capabilities
        .windows(2)
        .any(|pair| pair[0].as_str() >= pair[1].as_str())
        || capabilities.iter().any(|capability| {
            !matches!(
                capability.as_str(),
                "multimodal_input" | "reasoning_controls" | "structured_output" | "tools"
            )
        })
    {
        return Err(IneligibilityReason::CanonicalCapabilities);
    }
    Ok(())
}

fn canonical_position_features(
    routing: &RouterRoutingContextProjectionV1,
    canonicalizer: &CanonicalizerConfig,
) -> Result<Option<Vec<CanonicalPositionFeatureV1>>, IneligibilityReason> {
    let mut configured = canonicalizer.position_features.clone();
    configured.sort();
    configured.dedup();
    let projected = routing
        .position_features
        .keys()
        .cloned()
        .collect::<Vec<_>>();
    if configured != projected {
        return Err(IneligibilityReason::CanonicalPositionFeatures);
    }
    for (name, value) in &routing.position_features {
        let Some(turn_index) = value.as_str().and_then(|value| value.parse::<u64>().ok()) else {
            return Err(IneligibilityReason::CanonicalPositionFeatures);
        };
        if name != "turn_index" || turn_index.to_string() != value.as_str().unwrap_or_default() {
            return Err(IneligibilityReason::CanonicalPositionFeatures);
        }
    }
    let bytes = canonical_serialize_bytes(&routing.position_features)
        .map_err(|_| IneligibilityReason::CanonicalPositionFeatures)?;
    if bytes.len() > canonicalizer.max_position_features_bytes {
        return Err(IneligibilityReason::CanonicalPositionFeatures);
    }
    if routing.position_features.is_empty() {
        return Ok(None);
    }
    Ok(Some(
        routing
            .position_features
            .iter()
            .map(|(name, value)| CanonicalPositionFeatureV1 {
                name: name.clone(),
                value: value.clone(),
            })
            .collect(),
    ))
}

fn validate_optional_sha256(
    value: Option<&str>,
    error: IneligibilityReason,
) -> Result<Option<String>, IneligibilityReason> {
    value
        .map(|value| {
            validate_sha256(value, error)?;
            Ok(value.to_string())
        })
        .transpose()
}

fn validate_sha256(value: &str, error: IneligibilityReason) -> Result<(), IneligibilityReason> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(error);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use nemo_relay::api::llm::LlmApiFamily;
    use serde_json::{Value as Json, json};

    use super::{build_canonical_routing_query, normalize_text};
    use crate::config::CanonicalizerConfig;
    use crate::eligibility::IneligibilityReason;
    use crate::fingerprint::{canonical_serialize_bytes, sha256_hex};
    use crate::projection::{
        REQUEST_PROJECTION_SCHEMA_V1, ROUTER_SANITIZER_VERSION, ROUTING_CONTEXT_SCHEMA_V1,
        RouterRequestProjectionV1, RouterRoutingContextProjectionV1, SanitizedAnnotatedLlmRequest,
        SanitizedContentPart, SanitizedMessage, SanitizedMessageContent,
        SanitizedRouterInstructionFact, SanitizedRouterInstructionRole, SanitizedToolCall,
        SanitizedToolDefinition, projection_semantic_bytes,
    };

    fn canonicalizer() -> CanonicalizerConfig {
        CanonicalizerConfig {
            position_features: vec!["turn_index".to_string()],
            ..CanonicalizerConfig::default()
        }
    }

    fn routing(turn_index: u64) -> RouterRoutingContextProjectionV1 {
        routing_raw(json!(turn_index.to_string()))
    }

    fn routing_raw(turn_index: Json) -> RouterRoutingContextProjectionV1 {
        RouterRoutingContextProjectionV1 {
            schema: ROUTING_CONTEXT_SCHEMA_V1.to_string(),
            tenant_policy_hash: "1".repeat(64),
            agent_policy_hash: "2".repeat(64),
            position_features: BTreeMap::from([("turn_index".to_string(), turn_index)]),
        }
    }

    fn request(family: LlmApiFamily) -> RouterRequestProjectionV1 {
        let mut projection = RouterRequestProjectionV1 {
            schema: REQUEST_PROJECTION_SCHEMA_V1.to_string(),
            family,
            normalized_request: SanitizedAnnotatedLlmRequest {
                messages: vec![
                    SanitizedMessage::System {
                        content: SanitizedMessageContent::Text(
                            "ignored instruction copy".to_string(),
                        ),
                        name: Some("raw-system-name".to_string()),
                    },
                    SanitizedMessage::User {
                        content: SanitizedMessageContent::Parts(vec![
                            SanitizedContentPart::Text {
                                text: " older\t context ".to_string(),
                            },
                            SanitizedContentPart::Text {
                                text: "\r\ncontinued".to_string(),
                            },
                        ]),
                        name: Some("raw-user-name".to_string()),
                    },
                    SanitizedMessage::Assistant {
                        content: Some(SanitizedMessageContent::Text(
                            " assistant \t context ".to_string(),
                        )),
                        tool_calls: None,
                        name: None,
                    },
                    SanitizedMessage::User {
                        content: SanitizedMessageContent::Parts(vec![
                            SanitizedContentPart::Text {
                                text: " \r\nLatest\t task\r ".to_string(),
                            },
                            SanitizedContentPart::Text {
                                text: "  ".to_string(),
                            },
                        ]),
                        name: None,
                    },
                    SanitizedMessage::Tool {
                        content: SanitizedMessageContent::Text(" tool\t result ".to_string()),
                        tool_call_id: "raw-tool-call-id".to_string(),
                    },
                ],
                model: Some("anchor-model".to_string()),
                params: None,
                tools: Some(vec![SanitizedToolDefinition {
                    tool_type: "function".to_string(),
                    name: "lookup".to_string(),
                    description: Some("Look up data".to_string()),
                    parameters: Some(json!({
                        "type": "object",
                        "properties": {"q": {"type": "string"}},
                    })),
                    strict: Some(true),
                }]),
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
            ordered_instructions: vec![
                SanitizedRouterInstructionFact {
                    wire_ordinal: 9,
                    role: SanitizedRouterInstructionRole::Developer,
                    content: " Dev\t instruction ".to_string(),
                    name: Some("raw-developer-name".to_string()),
                },
                SanitizedRouterInstructionFact {
                    wire_ordinal: 1,
                    role: SanitizedRouterInstructionRole::System,
                    content: "Sys\r\n instruction".to_string(),
                    name: None,
                },
            ],
            response_format: None,
            response_schema_fingerprint: Some("3".repeat(64)),
            required_capabilities: vec!["tools".to_string()],
            sanitizer_version: ROUTER_SANITIZER_VERSION,
            semantic_request_fingerprint: String::new(),
        };
        refresh(&mut projection);
        projection
    }

    fn refresh(projection: &mut RouterRequestProjectionV1) {
        projection.semantic_request_fingerprint =
            sha256_hex(&projection_semantic_bytes(projection).unwrap());
    }

    #[test]
    fn three_families_share_exact_normalized_model_free_bytes() {
        let canonicalizer = canonicalizer();
        let routing = routing(7);
        let mut artifacts = [
            LlmApiFamily::OpenAIChatCompletions,
            LlmApiFamily::OpenAIResponses,
            LlmApiFamily::AnthropicMessages,
        ]
        .map(|family| {
            build_canonical_routing_query(&request(family), &routing, &canonicalizer).unwrap()
        });
        assert!(artifacts.windows(2).all(|pair| pair[0] == pair[1]));

        let artifact = &mut artifacts[0];
        assert_eq!(artifact.query.instructions[0].text, "Dev instruction");
        assert_eq!(artifact.query.instructions[1].text, "Sys\n instruction");
        assert_eq!(artifact.query.current_task.text, "Latest task");
        assert_eq!(
            artifact
                .query
                .bounded_context
                .iter()
                .map(|entry| entry.text.as_str())
                .collect::<Vec<_>>(),
            vec![
                "older context\ncontinued",
                "assistant context",
                "tool result"
            ]
        );
        let encoded = String::from_utf8(artifact.canonical_bytes.clone()).unwrap();
        let expected = concat!(
            r#"{"bounded_context":[{"role":"user","text":"older context\ncontinued"},"#,
            r#"{"role":"assistant","text":"assistant context"},{"role":"tool","text":"tool result"}],"#,
            r#""current_task":{"text":"Latest task"},"instructions":[{"role":"developer","text":"Dev instruction"},"#,
            r#"{"role":"system","text":"Sys\n instruction"}],"position_features":[{"name":"turn_index","value":"7"}],"#,
            r#""required_capabilities":["tools"],"response_schema_fingerprint":"3333333333333333333333333333333333333333333333333333333333333333","#,
            r#""schema":"nemo.relay.router.routing-query@1","tool_schema_fingerprint":"4edc36f14889081ceb3fdab37d5dca9fadd34cbb0729b33f59d9f5e7202c5b3b"}"#,
        );
        assert_eq!(encoded, expected);
        for omitted in [
            "anchor-model",
            "openai_",
            "anthropic_",
            "raw-system-name",
            "raw-developer-name",
            "raw-user-name",
            "raw-tool-call-id",
            "wire_ordinal",
        ] {
            assert!(!encoded.contains(omitted), "leaked {omitted}: {encoded}");
        }
        assert_eq!(
            artifact.canonical_query_hash,
            sha256_hex(encoded.as_bytes())
        );
        assert_eq!(
            artifact.canonical_query_hash,
            "5559fb64761d935e257c67351acd6a09634d9cb0ee0cba3ff2ba015b15831093"
        );

        let mut model_change = request(LlmApiFamily::OpenAIChatCompletions);
        model_change.normalized_request.model = Some("other-model".to_string());
        refresh(&mut model_change);
        let changed =
            build_canonical_routing_query(&model_change, &routing, &canonicalizer).unwrap();
        assert_eq!(changed.canonical_bytes, artifact.canonical_bytes);
    }

    #[test]
    fn normalization_preserves_newlines_and_normalizes_parts_independently() {
        assert_eq!(normalize_text(" \t a\r\n\r b\u{a0} c \n\n "), "a\n\n b c");
        assert_eq!(normalize_text("a \t \n \t b"), "a \n b");
        assert_eq!(
            normalize_text("a\u{000b}b\u{000c}c\u{0085}d\u{2028}e\u{2029}f"),
            "a\u{000b}b\u{000c}c\u{0085}d\u{2028}e\u{2029}f"
        );

        let mut request = request(LlmApiFamily::OpenAIChatCompletions);
        request.normalized_request.messages[3] = SanitizedMessage::User {
            content: SanitizedMessageContent::Parts(vec![
                SanitizedContentPart::Text {
                    text: " first ".to_string(),
                },
                SanitizedContentPart::Text {
                    text: " \t ".to_string(),
                },
                SanitizedContentPart::Text {
                    text: " second ".to_string(),
                },
            ]),
            name: None,
        };
        refresh(&mut request);
        let artifact =
            build_canonical_routing_query(&request, &routing(1), &canonicalizer()).unwrap();
        assert_eq!(artifact.query.current_task.text, "first\nsecond");
    }

    #[test]
    fn missing_task_and_any_multimodal_part_are_distinct() {
        let canonicalizer = canonicalizer();
        let routing = routing(1);
        let mut no_task = request(LlmApiFamily::OpenAIChatCompletions);
        no_task
            .normalized_request
            .messages
            .retain(|message| !matches!(message, SanitizedMessage::User { .. }));
        refresh(&mut no_task);
        assert_eq!(
            build_canonical_routing_query(&no_task, &routing, &canonicalizer).unwrap_err(),
            IneligibilityReason::CanonicalMissingTask
        );

        for index in [0usize, 1, 2, 3, 4] {
            let mut multimodal = request(LlmApiFamily::OpenAIChatCompletions);
            let content = SanitizedMessageContent::Parts(vec![SanitizedContentPart::ImageUrl {
                url: "https://example.invalid/image".to_string(),
                detail: None,
            }]);
            match &mut multimodal.normalized_request.messages[index] {
                SanitizedMessage::System {
                    content: target, ..
                }
                | SanitizedMessage::Developer {
                    content: target, ..
                }
                | SanitizedMessage::User {
                    content: target, ..
                }
                | SanitizedMessage::Tool {
                    content: target, ..
                } => *target = content,
                SanitizedMessage::Assistant {
                    content: target, ..
                } => *target = Some(content),
            }
            refresh(&mut multimodal);
            assert_eq!(
                build_canonical_routing_query(&multimodal, &routing, &canonicalizer).unwrap_err(),
                IneligibilityReason::CanonicalMultimodal
            );
        }

        let mut developer_multimodal = request(LlmApiFamily::OpenAIChatCompletions);
        developer_multimodal
            .normalized_request
            .messages
            .push(SanitizedMessage::Developer {
                content: SanitizedMessageContent::Parts(vec![SanitizedContentPart::ImageUrl {
                    url: "https://example.invalid/developer-image".to_string(),
                    detail: None,
                }]),
                name: None,
            });
        refresh(&mut developer_multimodal);
        assert_eq!(
            build_canonical_routing_query(&developer_multimodal, &routing, &canonicalizer)
                .unwrap_err(),
            IneligibilityReason::CanonicalMultimodal
        );
    }

    #[test]
    fn assistant_tool_call_structures_are_omitted_from_the_query() {
        let canonicalizer = canonicalizer();
        let routing = routing(1);
        let base = request(LlmApiFamily::OpenAIChatCompletions);
        let baseline = build_canonical_routing_query(&base, &routing, &canonicalizer)
            .unwrap()
            .canonical_bytes;

        let mut with_tool_call = base;
        let SanitizedMessage::Assistant { tool_calls, .. } =
            &mut with_tool_call.normalized_request.messages[2]
        else {
            panic!("fixture must contain an assistant message");
        };
        *tool_calls = Some(vec![SanitizedToolCall {
            id: "provider-call-id".to_string(),
            call_type: "function".to_string(),
            name: "lookup".to_string(),
            arguments: r#"{"q":"secret-free"}"#.to_string(),
        }]);
        refresh(&mut with_tool_call);

        assert_eq!(
            build_canonical_routing_query(&with_tool_call, &routing, &canonicalizer)
                .unwrap()
                .canonical_bytes,
            baseline
        );
    }

    #[test]
    fn instruction_task_and_context_boundaries_are_exact() {
        let request = request(LlmApiFamily::OpenAIChatCompletions);
        let routing = routing(1);
        let mut limits = canonicalizer();
        let instruction_bytes = "Dev instruction".len() + "Sys\n instruction".len();
        limits.max_instruction_bytes = instruction_bytes;
        assert!(build_canonical_routing_query(&request, &routing, &limits).is_ok());
        limits.max_instruction_bytes -= 1;
        assert_eq!(
            build_canonical_routing_query(&request, &routing, &limits).unwrap_err(),
            IneligibilityReason::CanonicalInstructionBound
        );

        limits = canonicalizer();
        limits.max_task_bytes = "Latest task".len();
        assert!(build_canonical_routing_query(&request, &routing, &limits).is_ok());
        limits.max_task_bytes -= 1;
        assert_eq!(
            build_canonical_routing_query(&request, &routing, &limits).unwrap_err(),
            IneligibilityReason::CanonicalTaskBound
        );

        limits = canonicalizer();
        limits.max_context_messages = 3;
        limits.max_context_bytes =
            "older context\ncontinued".len() + "assistant context".len() + "tool result".len();
        let all = build_canonical_routing_query(&request, &routing, &limits).unwrap();
        assert_eq!(all.query.bounded_context.len(), 3);
        limits.max_context_bytes -= 1;
        let stopped = build_canonical_routing_query(&request, &routing, &limits).unwrap();
        assert_eq!(stopped.query.bounded_context.len(), 2);
        assert_eq!(stopped.query.bounded_context[0].text, "assistant context");
        assert_eq!(stopped.query.bounded_context[1].text, "tool result");

        limits.max_context_bytes = usize::MAX;
        limits.max_context_messages = 2;
        let count_stopped = build_canonical_routing_query(&request, &routing, &limits).unwrap();
        assert_eq!(count_stopped.query.bounded_context.len(), 2);
    }

    #[test]
    fn first_oversized_recent_context_is_not_skipped() {
        let mut request = request(LlmApiFamily::OpenAIChatCompletions);
        request
            .normalized_request
            .messages
            .push(SanitizedMessage::Assistant {
                content: Some(SanitizedMessageContent::Text("oversized".to_string())),
                tool_calls: None,
                name: None,
            });
        request
            .normalized_request
            .messages
            .push(SanitizedMessage::Assistant {
                content: Some(SanitizedMessageContent::Text("older".to_string())),
                tool_calls: None,
                name: None,
            });
        refresh(&mut request);
        let mut limits = canonicalizer();
        limits.max_context_bytes = "older".len();
        let artifact = build_canonical_routing_query(&request, &routing(1), &limits).unwrap();
        assert_eq!(artifact.query.bounded_context.last().unwrap().text, "older");

        request.normalized_request.messages.swap(5, 6);
        refresh(&mut request);
        let artifact = build_canonical_routing_query(&request, &routing(1), &limits).unwrap();
        assert!(artifact.query.bounded_context.is_empty());
    }

    #[test]
    fn tool_fingerprint_covers_order_schema_and_optional_strictness() {
        let canonicalizer = canonicalizer();
        let routing = routing(1);
        let base = request(LlmApiFamily::OpenAIChatCompletions);
        let baseline = build_canonical_routing_query(&base, &routing, &canonicalizer)
            .unwrap()
            .query
            .tool_schema_fingerprint;

        let mut fingerprints = Vec::new();
        for strict in [None, Some(false), Some(true)] {
            let mut changed = base.clone();
            changed.normalized_request.tools.as_mut().unwrap()[0].strict = strict;
            refresh(&mut changed);
            fingerprints.push(
                build_canonical_routing_query(&changed, &routing, &canonicalizer)
                    .unwrap()
                    .query
                    .tool_schema_fingerprint,
            );
        }
        fingerprints.sort();
        fingerprints.dedup();
        assert_eq!(fingerprints.len(), 3);
        assert!(fingerprints.contains(&baseline));

        let mut schema_change = base.clone();
        schema_change.normalized_request.tools.as_mut().unwrap()[0].parameters =
            Some(json!({"type": "string"}));
        refresh(&mut schema_change);
        let changed = build_canonical_routing_query(&schema_change, &routing, &canonicalizer)
            .unwrap()
            .query
            .tool_schema_fingerprint;
        assert_ne!(changed, baseline);

        for changed_tool in [
            SanitizedToolDefinition {
                tool_type: "other".to_string(),
                ..base.normalized_request.tools.as_ref().unwrap()[0].clone()
            },
            SanitizedToolDefinition {
                name: "other".to_string(),
                ..base.normalized_request.tools.as_ref().unwrap()[0].clone()
            },
            SanitizedToolDefinition {
                description: Some("Other description".to_string()),
                ..base.normalized_request.tools.as_ref().unwrap()[0].clone()
            },
        ] {
            let mut changed = base.clone();
            changed.normalized_request.tools = Some(vec![changed_tool]);
            refresh(&mut changed);
            assert_ne!(
                build_canonical_routing_query(&changed, &routing, &canonicalizer)
                    .unwrap()
                    .query
                    .tool_schema_fingerprint,
                baseline
            );
        }

        let mut order_change = base.clone();
        let mut second = order_change.normalized_request.tools.as_ref().unwrap()[0].clone();
        second.name = "second".to_string();
        order_change
            .normalized_request
            .tools
            .as_mut()
            .unwrap()
            .push(second);
        refresh(&mut order_change);
        let in_order = build_canonical_routing_query(&order_change, &routing, &canonicalizer)
            .unwrap()
            .query
            .tool_schema_fingerprint;
        order_change
            .normalized_request
            .tools
            .as_mut()
            .unwrap()
            .reverse();
        refresh(&mut order_change);
        let reversed = build_canonical_routing_query(&order_change, &routing, &canonicalizer)
            .unwrap()
            .query
            .tool_schema_fingerprint;
        assert_ne!(in_order, reversed);

        let mut absent = base.clone();
        absent.normalized_request.tools = None;
        refresh(&mut absent);
        let mut empty = base;
        empty.normalized_request.tools = Some(Vec::new());
        refresh(&mut empty);
        assert_eq!(
            build_canonical_routing_query(&absent, &routing, &canonicalizer)
                .unwrap()
                .query
                .tool_schema_fingerprint,
            build_canonical_routing_query(&empty, &routing, &canonicalizer)
                .unwrap()
                .query
                .tool_schema_fingerprint
        );
    }

    #[test]
    fn projection_fingerprint_schema_response_and_capabilities_are_revalidated() {
        let canonicalizer = canonicalizer();
        let routing = routing(1);
        let base = request(LlmApiFamily::OpenAIChatCompletions);

        let mut corrupt = base.clone();
        corrupt.schema = "unknown".to_string();
        assert_eq!(
            build_canonical_routing_query(&corrupt, &routing, &canonicalizer).unwrap_err(),
            IneligibilityReason::CanonicalProjection
        );
        corrupt = base.clone();
        corrupt.semantic_request_fingerprint = "0".repeat(64);
        assert_eq!(
            build_canonical_routing_query(&corrupt, &routing, &canonicalizer).unwrap_err(),
            IneligibilityReason::CanonicalProjection
        );
        let mut oversized = base.clone();
        let message = oversized.normalized_request.messages[1].clone();
        oversized.normalized_request.messages =
            vec![message; crate::projection::REQUEST_PROJECTION_MAX_MESSAGES + 1];
        refresh(&mut oversized);
        assert_eq!(
            build_canonical_routing_query(&oversized, &routing, &canonicalizer).unwrap_err(),
            IneligibilityReason::CanonicalProjection
        );

        let mut bad_response = base.clone();
        bad_response.response_schema_fingerprint = Some("A".repeat(64));
        refresh(&mut bad_response);
        assert_eq!(
            build_canonical_routing_query(&bad_response, &routing, &canonicalizer).unwrap_err(),
            IneligibilityReason::CanonicalSchemaFingerprint
        );
        let mut no_response = base.clone();
        no_response.response_schema_fingerprint = None;
        refresh(&mut no_response);
        assert!(
            build_canonical_routing_query(&no_response, &routing, &canonicalizer)
                .unwrap()
                .query
                .response_schema_fingerprint
                .is_none()
        );

        for capabilities in [
            vec!["tools".to_string(), "reasoning_controls".to_string()],
            vec!["tools".to_string(), "tools".to_string()],
            vec!["unknown".to_string()],
        ] {
            let mut bad = base.clone();
            bad.required_capabilities = capabilities;
            refresh(&mut bad);
            assert_eq!(
                build_canonical_routing_query(&bad, &routing, &canonicalizer).unwrap_err(),
                IneligibilityReason::CanonicalCapabilities
            );
        }
    }

    #[test]
    fn turn_index_and_position_feature_boundaries_are_exact() {
        let request = request(LlmApiFamily::OpenAIChatCompletions);
        let mut limits = canonicalizer();
        for value in [0, u64::MAX] {
            let routing = routing(value);
            assert!(build_canonical_routing_query(&request, &routing, &limits).is_ok());
        }
        let adjacent = [1_u64 << 53, (1_u64 << 53) + 1].map(|value| {
            build_canonical_routing_query(&request, &routing(value), &limits).unwrap()
        });
        assert_ne!(adjacent[0].canonical_bytes, adjacent[1].canonical_bytes);
        assert_ne!(
            adjacent[0].canonical_query_hash,
            adjacent[1].canonical_query_hash
        );
        for value in [
            json!(0),
            json!(u64::MAX),
            json!(-1),
            json!(1.5),
            json!(true),
            Json::Null,
            json!("01"),
            json!("18446744073709551616"),
            json!([]),
            json!({}),
        ] {
            assert_eq!(
                build_canonical_routing_query(&request, &routing_raw(value), &limits).unwrap_err(),
                IneligibilityReason::CanonicalPositionFeatures
            );
        }

        let routing_projection = routing(7);
        let encoded = canonical_serialize_bytes(&routing_projection.position_features).unwrap();
        limits.max_position_features_bytes = encoded.len();
        assert!(build_canonical_routing_query(&request, &routing_projection, &limits).is_ok());
        limits.max_position_features_bytes -= 1;
        assert_eq!(
            build_canonical_routing_query(&request, &routing_projection, &limits).unwrap_err(),
            IneligibilityReason::CanonicalPositionFeatures
        );

        limits = canonicalizer();
        let mut missing = routing_projection.clone();
        missing.position_features.clear();
        assert_eq!(
            build_canonical_routing_query(&request, &missing, &limits).unwrap_err(),
            IneligibilityReason::CanonicalPositionFeatures
        );
        let mut unexpected_config = limits;
        unexpected_config.position_features.clear();
        assert_eq!(
            build_canonical_routing_query(&request, &routing_projection, &unexpected_config)
                .unwrap_err(),
            IneligibilityReason::CanonicalPositionFeatures
        );

        let mut no_positions = canonicalizer();
        no_positions.position_features.clear();
        let mut no_position_routing = routing(7);
        no_position_routing.position_features.clear();
        let no_position_artifact =
            build_canonical_routing_query(&request, &no_position_routing, &no_positions).unwrap();
        assert!(no_position_artifact.query.position_features.is_none());
        assert!(
            !String::from_utf8(no_position_artifact.canonical_bytes)
                .unwrap()
                .contains("position_features")
        );
    }
}
