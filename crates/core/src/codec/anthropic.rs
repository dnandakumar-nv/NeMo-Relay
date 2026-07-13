// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in codec for the Anthropic Messages API.
//!
//! Implements [`LlmCodec`] (request decode/encode) and [`LlmResponseCodec`]
//! (response decode) for the Anthropic Messages API format.
//!
//! # Anthropic-specific patterns handled
//!
//! - **Content blocks**: Heterogeneous array of `text`, `tool_use`, `thinking`,
//!   `redacted_thinking`, `mcp_tool_use`, `server_tool_use` blocks
//! - **Top-level system**: System prompt is a top-level field, not inside messages
//! - **stop_reason**: Maps to [`FinishReason`] (not `finish_reason`)
//! - **Tool definitions**: Uses `input_schema` instead of `parameters`
//! - **Tool choice**: `{"type":"auto"}` / `{"type":"any"}` / `{"type":"tool","name":"..."}`
//! - **Cache tokens**: `cache_read_input_tokens` / `cache_creation_input_tokens`

use serde::Deserialize;

use crate::api::llm::LlmRequest;
use crate::error::{FlowError, Result};
use crate::json::Json;

use super::request::{
    AnnotatedLlmRequest, FunctionDefinition, GenerationParams, Message, MessageContent,
    StructuredResponseFormat, StructuredResponseFormatKind, ToolChoice, ToolChoiceFunction,
    ToolChoiceFunctionName, ToolDefinition,
};
use super::resolve::{ProviderSurface, ProviderSurfaceDescriptor};
use super::response::{
    AnnotatedLlmResponse, ApiSpecificResponse, FinishReason, RawUsageCost, ResponseToolCall, Usage,
    estimate_cost_for_provider, infer_model_provider, provider_reported_cost,
};
use super::traits::{LlmCodec, LlmResponseCodec};

// ---------------------------------------------------------------------------
// Public codec struct
// ---------------------------------------------------------------------------

/// Built-in codec for the Anthropic Messages API.
pub struct AnthropicMessagesCodec;

pub(crate) const PROVIDER_SURFACE: ProviderSurfaceDescriptor = ProviderSurfaceDescriptor {
    surface: ProviderSurface::AnthropicMessages,
    detect_request: |obj, hint| {
        // A system-less Anthropic request is shape-identical to OpenAI Chat;
        // a recognized Anthropic provider hint disambiguates it.
        let hinted_anthropic = hint.is_some_and(|hint_value| {
            hint_value == "anthropic" || hint_value == "anthropic.messages"
        });
        obj.contains_key("system") || (hinted_anthropic && obj.contains_key("messages"))
    },
    detect_response: |obj| {
        obj.get("type").and_then(Json::as_str) == Some("message")
            && obj.get("content").is_some_and(Json::is_array)
    },
    decode_request: |request| AnthropicMessagesCodec.decode(request),
    decode_response: |raw| AnthropicMessagesCodec.decode_response(raw),
    codec_name: "anthropic_messages",
    request_codec: || std::sync::Arc::new(AnthropicMessagesCodec),
    response_codec: || std::sync::Arc::new(AnthropicMessagesCodec),
    streaming_codec: || Box::new(AnthropicMessagesStreamingCodec::new()),
};

// ---------------------------------------------------------------------------
// Private intermediate serde structs for response decode
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct RawAnthropicResponse {
    id: Option<String>,
    #[serde(rename = "type")]
    object_type: Option<String>,
    role: Option<String>,
    model: Option<String>,
    content: Option<Vec<Json>>,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
    service_tier: Option<String>,
    container: Option<Json>,
    usage: Option<RawAnthropicUsage>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Deserialize)]
struct RawAnthropicUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    cache_read_input_tokens: Option<u64>,
    cache_creation_input_tokens: Option<u64>,
    #[serde(rename = "cost_usd")]
    provider_cost: Option<f64>,
    cost: Option<RawUsageCost>,
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Map Anthropic `stop_reason` string to normalized [`FinishReason`].
fn map_anthropic_stop_reason(reason: &str) -> FinishReason {
    match reason {
        "end_turn" => FinishReason::Complete,
        "max_tokens" => FinishReason::Length,
        "tool_use" => FinishReason::ToolUse,
        other => FinishReason::Unknown(other.to_string()),
    }
}

/// Helper to construct a [`Json`] number from an `f64`.
fn json_f64(v: f64) -> Json {
    serde_json::Number::from_f64(v)
        .map(Json::Number)
        .unwrap_or(Json::Null)
}

/// Keys that are modeled in [`AnnotatedLlmRequest`] and should NOT go into `extra`.
const MODELED_REQUEST_KEYS: &[&str] = &[
    "system",
    "messages",
    "model",
    "max_tokens",
    "temperature",
    "top_p",
    "stop_sequences",
    "tools",
    "tool_choice",
    "output_config",
    "metadata",
    "service_tier",
];

const NATIVE_WRAPPER_KEY: &str = "native_wrapper";
const NATIVE_FORMAT_KEY: &str = "native_format";
const UNSUPPORTED_SYSTEM_KEY: &str = "_anthropic_messages_unsupported_system";

fn response_format_error(message: impl Into<String>) -> FlowError {
    FlowError::Internal(format!(
        "Anthropic Messages output_config.format decode: {}",
        message.into()
    ))
}

fn structured_extra(
    native_wrapper: serde_json::Map<String, Json>,
    native_format: serde_json::Map<String, Json>,
) -> serde_json::Map<String, Json> {
    let mut extra = serde_json::Map::new();
    if !native_wrapper.is_empty() {
        extra.insert(NATIVE_WRAPPER_KEY.into(), Json::Object(native_wrapper));
    }
    if !native_format.is_empty() {
        extra.insert(NATIVE_FORMAT_KEY.into(), Json::Object(native_format));
    }
    extra
}

fn decode_anthropic_response_format(
    output_config: Option<&Json>,
) -> Result<Option<StructuredResponseFormat>> {
    let Some(output_config) = output_config.and_then(Json::as_object) else {
        return Ok(None);
    };
    let Some(format) = output_config.get("format").and_then(Json::as_object) else {
        return Ok(None);
    };
    let Some(kind) = format.get("type").and_then(Json::as_str) else {
        return Ok(None);
    };
    if kind != "json_schema" {
        return Ok(None);
    }

    let schema = format
        .get("schema")
        .filter(|schema| !schema.is_null())
        .cloned()
        .ok_or_else(|| response_format_error("schema is required"))?;
    let native_wrapper = output_config
        .iter()
        .filter(|(key, _)| key.as_str() != "format")
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    let native_format = format
        .iter()
        .filter(|(key, _)| !matches!(key.as_str(), "type" | "schema"))
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    Ok(Some(StructuredResponseFormat {
        kind: StructuredResponseFormatKind::JsonSchema,
        name: None,
        schema: Some(schema),
        strict: None,
        extra: structured_extra(native_wrapper, native_format),
    }))
}

fn native_extra_map(
    format: &StructuredResponseFormat,
    key: &str,
) -> Result<serde_json::Map<String, Json>> {
    match format.extra.get(key) {
        Some(Json::Object(extra)) => Ok(extra.clone()),
        Some(_) => Err(FlowError::Internal(format!(
            "Anthropic Messages output_config.format encode: {key} must be an object"
        ))),
        None => Ok(serde_json::Map::new()),
    }
}

fn validate_native_extra_keys(format: &StructuredResponseFormat) -> Result<()> {
    if let Some(key) = format
        .extra
        .keys()
        .find(|key| !matches!(key.as_str(), NATIVE_WRAPPER_KEY | NATIVE_FORMAT_KEY))
    {
        return Err(FlowError::Internal(format!(
            "Anthropic Messages output_config.format encode: unsupported extra key {key}"
        )));
    }
    Ok(())
}

fn insert_format_field(
    obj: &mut serde_json::Map<String, Json>,
    key: &str,
    value: Json,
) -> Result<()> {
    if obj.contains_key(key) {
        return Err(FlowError::Internal(format!(
            "Anthropic Messages output_config.format encode: native extra conflicts with {key}"
        )));
    }
    obj.insert(key.into(), value);
    Ok(())
}

fn encode_anthropic_response_format(format: &StructuredResponseFormat) -> Result<Json> {
    validate_native_extra_keys(format)?;
    if format.kind != StructuredResponseFormatKind::JsonSchema {
        return Err(FlowError::Internal(
            "Anthropic Messages output_config.format encode: json_object is unsupported".into(),
        ));
    }
    if format.name.is_some() || format.strict.is_some() {
        return Err(FlowError::Internal(
            "Anthropic Messages output_config.format encode: name and strict are unsupported"
                .into(),
        ));
    }
    let schema = format
        .schema
        .as_ref()
        .filter(|schema| !schema.is_null())
        .ok_or_else(|| {
            FlowError::Internal(
                "Anthropic Messages output_config.format encode: schema is required".into(),
            )
        })?;

    let mut output_config = native_extra_map(format, NATIVE_WRAPPER_KEY)?;
    let mut descriptor = native_extra_map(format, NATIVE_FORMAT_KEY)?;
    insert_format_field(&mut descriptor, "type", Json::String("json_schema".into()))?;
    insert_format_field(&mut descriptor, "schema", schema.clone())?;
    insert_format_field(&mut output_config, "format", Json::Object(descriptor))?;
    Ok(Json::Object(output_config))
}

/// Decode the Anthropic `tool_choice` JSON value into a normalized [`ToolChoice`].
///
/// Anthropic format:
/// - `{"type": "auto"}` -> `ToolChoice::Auto`
/// - `{"type": "any"}` -> `ToolChoice::Required`
/// - `{"type": "none"}` -> `ToolChoice::None`
/// - `{"type": "tool", "name": "X"}` -> `ToolChoice::Specific`
fn decode_anthropic_tool_choice(val: &Json) -> Option<ToolChoice> {
    let obj = val.as_object()?;
    let tc_type = obj.get("type")?.as_str()?;
    match tc_type {
        "auto" => Some(ToolChoice::Auto),
        "any" => Some(ToolChoice::Required),
        "none" => Some(ToolChoice::None),
        "tool" => {
            let name = obj.get("name")?.as_str()?.to_string();
            Some(ToolChoice::Specific(ToolChoiceFunction {
                choice_type: "function".into(),
                function: ToolChoiceFunctionName { name },
            }))
        }
        _ => None,
    }
}

/// Extract Anthropic `disable_parallel_tool_use` from tool_choice and map
/// to normalized `parallel_tool_calls` semantics.
fn decode_parallel_tool_calls(val: &Json) -> Option<bool> {
    let obj = val.as_object()?;
    obj.get("disable_parallel_tool_use")
        .and_then(|v| v.as_bool())
        .map(|disabled| !disabled)
}

/// Encode a normalized [`ToolChoice`] back into Anthropic JSON format.
fn encode_anthropic_tool_choice(tc: &ToolChoice) -> Json {
    match tc {
        ToolChoice::Auto => serde_json::json!({"type": "auto"}),
        ToolChoice::Required => serde_json::json!({"type": "any"}),
        ToolChoice::None => serde_json::json!({"type": "none"}),
        ToolChoice::Specific(func) => {
            serde_json::json!({"type": "tool", "name": func.function.name})
        }
    }
}

fn encode_tool_choice_with_parallel_hint(
    tc: &ToolChoice,
    parallel_tool_calls: Option<bool>,
) -> Json {
    let mut value = encode_anthropic_tool_choice(tc);
    if let (Some(parallel), Some(obj)) = (parallel_tool_calls, value.as_object_mut()) {
        obj.insert("disable_parallel_tool_use".into(), Json::Bool(!parallel));
    }
    value
}

/// Extract the system prompt from an Anthropic top-level `system` field.
///
/// Handles both string and array-of-content-blocks formats.
fn extract_system_message(system_val: &Json) -> Option<Message> {
    if let Some(s) = system_val.as_str() {
        Some(Message::System {
            content: MessageContent::Text(s.to_string()),
            name: None,
        })
    } else if let Some(arr) = system_val.as_array() {
        // Array of content blocks -- extract text from each "text" block.
        let texts: Vec<&str> = arr
            .iter()
            .filter_map(|block| {
                let block_type = block.get("type")?.as_str()?;
                if block_type == "text" {
                    block.get("text")?.as_str()
                } else {
                    None
                }
            })
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(Message::System {
                content: MessageContent::Text(texts.join("\n")),
                name: None,
            })
        }
    } else {
        None
    }
}

fn is_supported_system_representation(system: &Json) -> bool {
    match system {
        Json::String(_) => true,
        Json::Array(blocks) => matches!(
            blocks.as_slice(),
            [block]
                if block.get("type").and_then(Json::as_str) == Some("text")
                    && block.get("text").is_some_and(Json::is_string)
        ),
        _ => false,
    }
}

/// Extract system text from a [`Message::System`] for encoding back to top-level.
fn extract_system_text(msg: &Message) -> Option<String> {
    match msg {
        Message::System {
            content: MessageContent::Text(s),
            ..
        } => Some(s.clone()),
        Message::System {
            content: MessageContent::Parts(parts),
            ..
        } => {
            let texts: Vec<&str> = parts
                .iter()
                .filter_map(|p| match p {
                    super::request::ContentPart::Text { text } => Some(text.as_str()),
                    super::request::ContentPart::ImageUrl { .. } => None,
                })
                .collect();
            if texts.is_empty() {
                None
            } else {
                Some(texts.join("\n"))
            }
        }
        _ => None,
    }
}

fn split_system_and_messages(messages: &[Message]) -> Result<(Option<&Message>, Vec<&Message>)> {
    let mut system_message = None;
    let mut non_system_messages = Vec::new();

    for (index, msg) in messages.iter().enumerate() {
        match msg {
            Message::System { .. } if index == 0 && system_message.is_none() => {
                system_message = Some(msg);
            }
            Message::System { .. } => {
                return Err(FlowError::Internal(
                    "Anthropic Messages encode: system instructions must be first and unique"
                        .into(),
                ));
            }
            Message::Developer { .. } => {
                return Err(FlowError::Internal(
                    "Anthropic Messages encode: developer messages are unsupported".into(),
                ));
            }
            _ => non_system_messages.push(msg),
        }
    }

    Ok((system_message, non_system_messages))
}

fn original_messages_match(obj: &serde_json::Map<String, Json>, messages: &[&Message]) -> bool {
    match obj.get("messages") {
        Some(original) => {
            serde_json::from_value::<Vec<Message>>(original.clone()).is_ok_and(|decoded| {
                decoded.len() == messages.len()
                    && decoded
                        .iter()
                        .zip(messages)
                        .all(|(decoded, message)| decoded == *message)
            })
        }
        None => messages.is_empty(),
    }
}

fn encode_anthropic_system(
    obj: &mut serde_json::Map<String, Json>,
    system_message: Option<&Message>,
) -> Result<()> {
    let original_system = obj.get("system").cloned();
    let original_message = original_system.as_ref().and_then(extract_system_message);

    match system_message {
        Some(message) if original_message.as_ref() == Some(message) => {
            // Keep the original string or content-block array, including cache controls.
        }
        Some(message) => {
            let text = extract_system_text(message).ok_or_else(|| {
                FlowError::Internal(
                    "Anthropic Messages encode: system content must contain only text".into(),
                )
            })?;
            obj.insert("system".into(), Json::String(text));
        }
        None if original_system.is_some() && original_message.is_none() => {
            // Preserve an unrecognized native system value rather than silently dropping it.
        }
        None => {
            obj.remove("system");
        }
    }
    Ok(())
}

fn insert_serialized<T: serde::Serialize>(
    obj: &mut serde_json::Map<String, Json>,
    key: &str,
    value: &T,
    context: &str,
) -> Result<()> {
    let json = serde_json::to_value(value)
        .map_err(|e| FlowError::Internal(format!("Anthropic Messages {context} encode: {e}")))?;
    obj.insert(key.into(), json);
    Ok(())
}

fn overlay_generation_params(obj: &mut serde_json::Map<String, Json>, params: &GenerationParams) {
    if let Some(temp) = params.temperature {
        obj.insert("temperature".into(), json_f64(temp));
    }
    if let Some(top_p) = params.top_p {
        obj.insert("top_p".into(), json_f64(top_p));
    }
    if let Some(max_tokens) = params.max_tokens {
        obj.insert("max_tokens".into(), Json::from(max_tokens));
    }
}

fn encode_anthropic_tools(tools: &[ToolDefinition]) -> Result<Vec<Json>> {
    tools
        .iter()
        .map(|td| {
            if td.function.strict.is_some() {
                return Err(FlowError::Internal(
                    "Anthropic Messages tools encode: strict is unsupported".into(),
                ));
            }
            let mut tool = serde_json::Map::new();
            tool.insert("name".into(), Json::String(td.function.name.clone()));
            if let Some(ref desc) = td.function.description {
                tool.insert("description".into(), Json::String(desc.clone()));
            }
            if let Some(ref params) = td.function.parameters {
                tool.insert("input_schema".into(), params.clone());
            }
            Ok(Json::Object(tool))
        })
        .collect()
}

fn decode_anthropic_tools(value: Option<&Json>) -> Result<Option<Vec<ToolDefinition>>> {
    let Some(tools) = value.and_then(Json::as_array) else {
        return Ok(None);
    };

    let mut definitions = Vec::new();
    for tool in tools {
        if tool.get("strict").is_some() {
            return Err(FlowError::Internal(
                "Anthropic Messages tools decode: strict is unsupported".into(),
            ));
        }
        let Some(name) = tool.get("name").and_then(Json::as_str) else {
            continue;
        };
        let description = tool
            .get("description")
            .and_then(Json::as_str)
            .map(String::from);
        let parameters = tool.get("input_schema").cloned();
        definitions.push(ToolDefinition {
            tool_type: "function".into(),
            function: FunctionDefinition {
                name: name.to_string(),
                description,
                parameters,
                strict: None,
            },
        });
    }

    Ok((!definitions.is_empty()).then_some(definitions))
}

fn anthropic_text_message(content_blocks: Option<&[Json]>) -> Option<MessageContent> {
    let text_parts: Vec<&str> = content_blocks
        .map(|blocks| blocks.iter().filter_map(anthropic_text_block).collect())
        .unwrap_or_default();

    (!text_parts.is_empty()).then(|| MessageContent::Text(text_parts.join("\n")))
}

fn anthropic_text_block(block: &Json) -> Option<&str> {
    if block.get("type")?.as_str()? != "text" {
        return None;
    }
    block.get("text")?.as_str()
}

fn anthropic_tool_calls(content_blocks: Option<&[Json]>) -> Option<Vec<ResponseToolCall>> {
    let tool_calls: Vec<ResponseToolCall> = content_blocks
        .map(|blocks| {
            blocks
                .iter()
                .filter_map(anthropic_tool_call_block)
                .collect()
        })
        .unwrap_or_default();

    (!tool_calls.is_empty()).then_some(tool_calls)
}

fn anthropic_tool_call_block(block: &Json) -> Option<ResponseToolCall> {
    if block.get("type")?.as_str()? != "tool_use" {
        return None;
    }
    Some(ResponseToolCall {
        id: block.get("id")?.as_str()?.to_string(),
        name: block.get("name")?.as_str()?.to_string(),
        // CRITICAL: input is already parsed JSON -- clone directly.
        arguments: block.get("input")?.clone(),
    })
}

fn anthropic_usage(
    raw_usage: Option<RawAnthropicUsage>,
    model_for_pricing: Option<&str>,
) -> Option<Usage> {
    let model_provider = infer_model_provider("anthropic", model_for_pricing);
    raw_usage.map(|u| {
        let prompt = u.input_tokens;
        let completion = u.output_tokens;
        let mut usage = Usage {
            prompt_tokens: prompt,
            completion_tokens: completion,
            // Anthropic does not supply total_tokens; compute it.
            total_tokens: match (prompt, completion) {
                (Some(p), Some(c)) => Some(p + c),
                _ => None,
            },
            cache_read_tokens: u.cache_read_input_tokens,
            cache_write_tokens: u.cache_creation_input_tokens,
            cost: provider_reported_cost(u.provider_cost, u.cost),
        };
        if usage.cost.is_none() {
            usage.cost = model_for_pricing.and_then(|model| {
                estimate_cost_for_provider(model_provider.as_deref(), model, &usage)
            });
        }
        usage
    })
}

// ---------------------------------------------------------------------------
// LlmResponseCodec implementation
// ---------------------------------------------------------------------------

impl LlmResponseCodec for AnthropicMessagesCodec {
    fn decode_response(&self, response: &Json) -> Result<AnnotatedLlmResponse> {
        let raw: RawAnthropicResponse = serde_json::from_value(response.clone())
            .map_err(|e| FlowError::Internal(format!("Anthropic Messages response decode: {e}")))?;

        let content_blocks = raw.content.as_deref();
        let message = anthropic_text_message(content_blocks);
        // Extract tool_use blocks (only "tool_use" type, NOT mcp_tool_use or server_tool_use).
        let tool_calls = anthropic_tool_calls(content_blocks);

        // Map stop_reason to FinishReason.
        let finish_reason = raw.stop_reason.as_deref().map(map_anthropic_stop_reason);

        // Map usage.
        let usage = anthropic_usage(raw.usage, raw.model.as_deref());

        // Build API-specific fields: all content blocks + stop_sequence.
        let api_specific_content_blocks = raw.content.clone();
        let api_specific = Some(ApiSpecificResponse::AnthropicMessages {
            object_type: raw.object_type,
            role: raw.role,
            stop_reason: raw.stop_reason,
            stop_sequence: raw.stop_sequence,
            service_tier: raw.service_tier,
            container: raw.container,
            content_blocks: api_specific_content_blocks,
        });

        Ok(AnnotatedLlmResponse {
            id: raw.id,
            model: raw.model,
            message,
            tool_calls,
            finish_reason,
            usage,
            optimization_summary: None,
            api_specific,
            extra: raw.extra,
        })
    }
}

// ---------------------------------------------------------------------------
// LlmCodec implementation
// ---------------------------------------------------------------------------

impl LlmCodec for AnthropicMessagesCodec {
    fn decode(&self, request: &LlmRequest) -> Result<AnnotatedLlmRequest> {
        let obj = request
            .content
            .as_object()
            .ok_or_else(|| FlowError::Internal("request content is not an object".into()))?;

        // Extract system from top-level field.
        let system = obj.get("system");
        let system_msg = system.and_then(extract_system_message);
        let unsupported_system =
            system.is_some_and(|value| !is_supported_system_representation(value));

        // Extract messages (default to empty vec if absent).
        let mut messages: Vec<Message> = obj
            .get("messages")
            .map(|value| serde_json::from_value(value.clone()))
            .transpose()
            .map_err(|error| {
                FlowError::Internal(format!("Anthropic Messages messages decode: {error}"))
            })?
            .unwrap_or_default();
        if messages
            .iter()
            .any(|message| matches!(message, Message::System { .. } | Message::Developer { .. }))
        {
            return Err(FlowError::Internal(
                "Anthropic Messages messages decode: system and developer roles are unsupported"
                    .into(),
            ));
        }

        // Prepend system message if present.
        if let Some(sys) = system_msg {
            messages.insert(0, sys);
        }

        // Extract model.
        let model = obj.get("model").and_then(|v| v.as_str()).map(String::from);

        let response_format = decode_anthropic_response_format(obj.get("output_config"))?;

        // Extract generation params.
        let temperature = obj.get("temperature").and_then(|v| v.as_f64());
        let top_p = obj.get("top_p").and_then(|v| v.as_f64());
        let max_tokens = obj.get("max_tokens").and_then(|v| v.as_u64());
        // Anthropic uses stop_sequences (not stop).
        let stop = obj
            .get("stop_sequences")
            .and_then(|v| serde_json::from_value::<Vec<String>>(v.clone()).ok());

        let params =
            if temperature.is_some() || max_tokens.is_some() || top_p.is_some() || stop.is_some() {
                Some(GenerationParams {
                    temperature,
                    max_tokens,
                    top_p,
                    stop,
                })
            } else {
                None
            };

        // Extract tools: Anthropic uses flat structure (name, description, input_schema).
        // Normalize to ToolDefinition { type: "function", function: { name, description, parameters } }.
        let tools = decode_anthropic_tools(obj.get("tools"))?;

        // Extract tool_choice: Anthropic format.
        let tool_choice = obj
            .get("tool_choice")
            .and_then(decode_anthropic_tool_choice);
        let parallel_tool_calls = obj.get("tool_choice").and_then(decode_parallel_tool_calls);

        // Collect extra fields (keys not in MODELED_REQUEST_KEYS).
        let mut extra: serde_json::Map<String, Json> = obj
            .iter()
            .filter(|(k, _)| {
                !MODELED_REQUEST_KEYS.contains(&k.as_str())
                    || (k.as_str() == "output_config" && response_format.is_none())
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        if unsupported_system {
            extra.insert(UNSUPPORTED_SYSTEM_KEY.into(), Json::Bool(true));
        }

        Ok(AnnotatedLlmRequest {
            messages,
            model,
            params,
            tools,
            tool_choice,
            response_format,
            store: None,
            previous_response_id: None,
            truncation: None,
            reasoning: None,
            include: None,
            user: None,
            metadata: obj.get("metadata").cloned(),
            service_tier: obj
                .get("service_tier")
                .and_then(|v| v.as_str())
                .map(String::from),
            parallel_tool_calls,
            max_output_tokens: None,
            max_tool_calls: None,
            top_logprobs: None,
            stream: None,
            extra,
        })
    }

    fn encode(&self, annotated: &AnnotatedLlmRequest, original: &LlmRequest) -> Result<LlmRequest> {
        let mut content = original.content.clone();
        let obj = content
            .as_object_mut()
            .ok_or_else(|| FlowError::Internal("original content is not an object".into()))?;

        let (system_message, non_system_messages) = split_system_and_messages(&annotated.messages)?;
        encode_anthropic_system(obj, system_message)?;

        // Keep native message extensions when only another request field changed.
        if !original_messages_match(obj, &non_system_messages) {
            insert_serialized(obj, "messages", &non_system_messages, "messages")?;
        }

        // Overlay model if present.
        if let Some(ref model) = annotated.model {
            obj.insert("model".into(), Json::String(model.clone()));
        }

        // Overlay generation params.
        if let Some(ref params) = annotated.params {
            overlay_generation_params(obj, params);
            // Write stop_sequences (Anthropic key name, not "stop").
            if let Some(ref stop) = params.stop {
                insert_serialized(obj, "stop_sequences", stop, "stop_sequences")?;
            }
        }

        // Overlay tools in Anthropic format: { name, description, input_schema }.
        // Denormalize from ToolDefinition (drop type/function wrapper, rename parameters -> input_schema).
        if let Some(ref tools) = annotated.tools {
            let anthropic_tools = encode_anthropic_tools(tools)?;
            insert_serialized(obj, "tools", &anthropic_tools, "tools")?;
        }

        // Overlay tool_choice in Anthropic format.
        if let Some(ref tool_choice) = annotated.tool_choice {
            obj.insert(
                "tool_choice".into(),
                encode_tool_choice_with_parallel_hint(tool_choice, annotated.parallel_tool_calls),
            );
        }

        if let Some(ref metadata) = annotated.metadata {
            obj.insert("metadata".into(), metadata.clone());
        }
        if let Some(ref service_tier) = annotated.service_tier {
            obj.insert("service_tier".into(), Json::String(service_tier.clone()));
        }

        if annotated.response_format.is_some() && annotated.extra.contains_key("output_config") {
            return Err(FlowError::Internal(
                "Anthropic Messages output_config.format encode: typed and generic representations conflict"
                    .into(),
            ));
        }
        if let Some(response_format) = &annotated.response_format {
            obj.insert(
                "output_config".into(),
                encode_anthropic_response_format(response_format)?,
            );
        } else if !annotated.extra.contains_key("output_config") {
            obj.remove("output_config");
        }

        // Merge extra fields back.
        for (k, v) in &annotated.extra {
            if k != UNSUPPORTED_SYSTEM_KEY {
                obj.insert(k.clone(), v.clone());
            }
        }

        Ok(LlmRequest {
            headers: original.headers.clone(),
            content,
        })
    }
}

// ---------------------------------------------------------------------------
// Streaming codec
// ---------------------------------------------------------------------------

/// Streaming counterpart to [`AnthropicMessagesCodec`].
///
/// Replays the Anthropic Messages SSE event sequence into the same JSON shape Anthropic returns
/// for a non-streaming request (`{id, type, role, model, content, stop_reason, stop_sequence,
/// usage}`). Once finalized, the assembled JSON can be fed back through
/// [`AnthropicMessagesCodec::decode_response`] to produce an
/// [`AnnotatedLlmResponse`] — meaning streaming and
/// non-streaming Anthropic requests converge on the same observability output.
///
/// Internal state lives behind `Arc<Mutex<...>>` so the `&self`-produced collector and finalizer
/// closures share access. Each instance is single-use because [`LlmFinalizerFn`] consumes the
/// finalize step.
///
/// [`LlmFinalizerFn`]: crate::api::runtime::LlmFinalizerFn
pub struct AnthropicMessagesStreamingCodec {
    state: std::sync::Arc<std::sync::Mutex<AnthropicMessagesStreamingState>>,
}

impl AnthropicMessagesStreamingCodec {
    /// Creates a fresh streaming codec with empty accumulator state.
    pub fn new() -> Self {
        Self {
            state: std::sync::Arc::new(std::sync::Mutex::new(
                AnthropicMessagesStreamingState::default(),
            )),
        }
    }
}

impl Default for AnthropicMessagesStreamingCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl super::streaming::StreamingCodec for AnthropicMessagesStreamingCodec {
    fn collector(&self) -> crate::api::runtime::LlmCollectorFn {
        let state = std::sync::Arc::clone(&self.state);
        Box::new(move |event: Json| -> Result<()> {
            let mut guard = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            guard.observe(&event);
            Ok(())
        })
    }

    fn finalizer(&self) -> crate::api::runtime::LlmFinalizerFn {
        let state = std::sync::Arc::clone(&self.state);
        Box::new(move || -> Json {
            let mut guard = state
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            // Move state out so finalize can consume it; the codec is single-use, so leaving a
            // default behind is intentional and never observed by another caller.
            std::mem::take(&mut *guard).finalize()
        })
    }
}

#[derive(Debug, Default)]
struct AnthropicMessagesStreamingState {
    id: Option<String>,
    type_: Option<String>,
    role: Option<String>,
    model: Option<String>,
    /// Latest usage snapshot. `message_start` carries an initial value (input tokens, zero output
    /// so far); `message_delta` updates it cumulatively. Last write wins.
    usage: Option<Json>,
    stop_reason: Option<String>,
    /// Stored as raw `Json` to preserve `null` (Anthropic's wire shape) versus omitted.
    stop_sequence: Option<Json>,
    /// Indexed by the SSE event's `index` field. `None` slots accommodate sparse indices though
    /// Anthropic emits them in order today.
    blocks: Vec<Option<StreamingBlock>>,
}

#[derive(Debug, Default, Clone)]
struct StreamingBlock {
    /// The `content_block` JSON captured at `content_block_start`. Deltas mutate fields directly
    /// for blocks Anthropic delivers incrementally (text, tool_use input, citations); other block
    /// types (server_tool_use results) ship complete at start and pass through unchanged.
    skeleton: serde_json::Map<String, Json>,
    text: String,
    has_text: bool,
    partial_json: String,
    has_partial_json: bool,
    citations: Vec<Json>,
    has_citations: bool,
}

impl AnthropicMessagesStreamingState {
    fn observe(&mut self, event: &Json) {
        let event_type = event.get("type").and_then(Json::as_str).unwrap_or("");
        match event_type {
            "message_start" => self.observe_message_start(event),
            "content_block_start" => self.observe_content_block_start(event),
            "content_block_delta" => self.observe_content_block_delta(event),
            "message_delta" => self.observe_message_delta(event),
            // content_block_stop, message_stop, ping, and any unknown event type carry no
            // accumulator-relevant payload. Unknown types are ignored rather than erroring so a
            // future Anthropic event addition does not break observability.
            _ => {}
        }
    }

    fn observe_message_start(&mut self, event: &Json) {
        let Some(message) = event.get("message") else {
            return;
        };
        if let Some(id) = message.get("id").and_then(Json::as_str) {
            self.id = Some(id.to_string());
        }
        if let Some(model) = message.get("model").and_then(Json::as_str) {
            self.model = Some(model.to_string());
        }
        if let Some(role) = message.get("role").and_then(Json::as_str) {
            self.role = Some(role.to_string());
        }
        if let Some(t) = message.get("type").and_then(Json::as_str) {
            self.type_ = Some(t.to_string());
        }
        if let Some(usage) = message.get("usage") {
            self.usage = Some(usage.clone());
        }
    }

    fn observe_content_block_start(&mut self, event: &Json) {
        let Some(index) = event.get("index").and_then(Json::as_u64) else {
            return;
        };
        let Some(content_block) = event.get("content_block") else {
            return;
        };
        let skeleton = match content_block {
            Json::Object(map) => map.clone(),
            _ => return,
        };
        let index = index as usize;
        while self.blocks.len() <= index {
            self.blocks.push(None);
        }
        self.blocks[index] = Some(StreamingBlock {
            skeleton,
            ..StreamingBlock::default()
        });
    }

    fn observe_content_block_delta(&mut self, event: &Json) {
        let Some(index) = event.get("index").and_then(Json::as_u64) else {
            return;
        };
        let index = index as usize;
        let Some(delta) = event.get("delta") else {
            return;
        };
        let delta_type = delta.get("type").and_then(Json::as_str).unwrap_or("");
        let Some(slot) = self.blocks.get_mut(index) else {
            return;
        };
        let Some(block) = slot.as_mut() else { return };
        match delta_type {
            "text_delta" => {
                if let Some(text) = delta.get("text").and_then(Json::as_str) {
                    block.text.push_str(text);
                    block.has_text = true;
                }
            }
            "input_json_delta" => {
                if let Some(partial) = delta.get("partial_json").and_then(Json::as_str) {
                    block.partial_json.push_str(partial);
                    block.has_partial_json = true;
                }
            }
            "citations_delta" => {
                if let Some(citation) = delta.get("citation") {
                    block.citations.push(citation.clone());
                    block.has_citations = true;
                }
            }
            // thinking_delta, signature_delta, and any future delta types fall through; the block
            // skeleton retains whatever shape was set at content_block_start.
            _ => {}
        }
    }

    fn observe_message_delta(&mut self, event: &Json) {
        if let Some(delta) = event.get("delta") {
            if let Some(reason) = delta.get("stop_reason").and_then(Json::as_str) {
                self.stop_reason = Some(reason.to_string());
            }
            if let Some(seq) = delta.get("stop_sequence") {
                self.stop_sequence = Some(seq.clone());
            }
        }
        if let Some(usage) = event.get("usage") {
            self.usage = Some(usage.clone());
        }
    }

    fn finalize(self) -> Json {
        let mut output = serde_json::Map::new();
        if let Some(id) = self.id {
            output.insert("id".to_string(), Json::String(id));
        }
        if let Some(t) = self.type_ {
            output.insert("type".to_string(), Json::String(t));
        }
        if let Some(role) = self.role {
            output.insert("role".to_string(), Json::String(role));
        }
        if let Some(model) = self.model {
            output.insert("model".to_string(), Json::String(model));
        }
        let content: Vec<Json> = self
            .blocks
            .into_iter()
            .filter_map(|block| block.map(StreamingBlock::finalize))
            .collect();
        output.insert("content".to_string(), Json::Array(content));
        if let Some(reason) = self.stop_reason {
            output.insert("stop_reason".to_string(), Json::String(reason));
        }
        if let Some(seq) = self.stop_sequence {
            output.insert("stop_sequence".to_string(), seq);
        }
        if let Some(usage) = self.usage {
            output.insert("usage".to_string(), usage);
        }
        Json::Object(output)
    }
}

impl StreamingBlock {
    fn finalize(mut self) -> Json {
        if self.has_text {
            self.skeleton
                .insert("text".to_string(), Json::String(self.text));
        }
        if self.has_partial_json {
            // Concatenated `partial_json` fragments are expected to parse as a JSON object — that's
            // the assembled tool input. If parsing fails (Anthropic emits malformed deltas, stream
            // truncated mid-block), surface the raw concatenation so observability still captures
            // something rather than dropping the call.
            let parsed = match serde_json::from_str::<Json>(&self.partial_json) {
                Ok(value) => value,
                Err(_) => Json::String(self.partial_json),
            };
            self.skeleton.insert("input".to_string(), parsed);
        }
        if self.has_citations {
            self.skeleton
                .insert("citations".to_string(), Json::Array(self.citations));
        }
        Json::Object(self.skeleton)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "../../tests/unit/codec/anthropic_tests.rs"]
mod tests;
