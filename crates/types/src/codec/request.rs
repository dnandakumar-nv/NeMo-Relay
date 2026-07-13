// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! LLM request codec types and trait.
//!
//! This module defines the [`AnnotatedLlmRequest`] type system for structured
//! LLM request representation and the [`crate::codec::traits::LlmCodec`] trait
//! for bidirectional translation between opaque [`crate::api::llm::LlmRequest`]
//! payloads and typed form.

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Json;

// ---------------------------------------------------------------------------
// AnnotatedLlmRequest type hierarchy
// ---------------------------------------------------------------------------

/// Structured view of an LLM request, produced by a Codec from opaque
/// [`LlmRequest`](crate::api::llm::LlmRequest) content.
///
/// The `extra` field captures any provider-specific keys not modeled by the
/// known fields, ensuring lossless round-trip through `decode`/`encode`.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct AnnotatedLlmRequest {
    /// Parsed conversation messages.
    pub messages: Vec<Message>,
    /// Model identifier (e.g., `"gpt-4"`, `"claude-sonnet-4-20250514"`).
    pub model: Option<String>,
    /// Common generation parameters, normalized.
    pub params: Option<GenerationParams>,
    /// Tool definitions (function schemas) available to the model.
    pub tools: Option<Vec<ToolDefinition>>,
    /// Tool choice control.
    pub tool_choice: Option<ToolChoice>,
    /// Structured response format requested from the provider.
    pub response_format: Option<StructuredResponseFormat>,
    /// OpenAI Responses: whether to persist response state server-side.
    pub store: Option<bool>,
    /// OpenAI Responses: prior response to continue from.
    pub previous_response_id: Option<String>,
    /// OpenAI Responses: context truncation behavior.
    pub truncation: Option<Json>,
    /// OpenAI Responses: reasoning configuration object.
    pub reasoning: Option<Json>,
    /// OpenAI Responses: include filter for additional output/state items.
    pub include: Option<Json>,
    /// OpenAI user identifier.
    pub user: Option<String>,
    /// OpenAI metadata map/object.
    pub metadata: Option<Json>,
    /// OpenAI service tier preference.
    pub service_tier: Option<String>,
    /// OpenAI tool parallelism toggle.
    pub parallel_tool_calls: Option<bool>,
    /// OpenAI Responses max output token limit.
    pub max_output_tokens: Option<u64>,
    /// OpenAI Responses max tool calls.
    pub max_tool_calls: Option<u64>,
    /// OpenAI logprob fanout count.
    pub top_logprobs: Option<u64>,
    /// OpenAI streaming toggle.
    pub stream: Option<bool>,
    /// Extensible key-value pairs for unmodeled provider-specific fields.
    /// Merged back into the request body during encode via `serde(flatten)`.
    #[cfg_attr(feature = "schema", schemars(flatten))]
    pub extra: serde_json::Map<String, Json>,
}

#[derive(Serialize)]
struct AnnotatedLlmRequestRef<'a> {
    messages: &'a [Message],
    #[serde(skip_serializing_if = "Option::is_none")]
    model: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<&'a GenerationParams>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<&'a Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<&'a ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<&'a StructuredResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    store: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    truncation: Option<&'a Json>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<&'a Json>,
    #[serde(skip_serializing_if = "Option::is_none")]
    include: Option<&'a Json>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<&'a Json>,
    #[serde(skip_serializing_if = "Option::is_none")]
    service_tier: Option<&'a String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    parallel_tool_calls: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_output_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tool_calls: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_logprobs: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(flatten)]
    extra: &'a serde_json::Map<String, Json>,
}

impl Serialize for AnnotatedLlmRequest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        if self.response_format.is_some() && self.extra.contains_key("response_format") {
            return Err(serde::ser::Error::custom(
                "typed and generic response_format representations conflict",
            ));
        }

        AnnotatedLlmRequestRef {
            messages: &self.messages,
            model: self.model.as_ref(),
            params: self.params.as_ref(),
            tools: self.tools.as_ref(),
            tool_choice: self.tool_choice.as_ref(),
            response_format: self.response_format.as_ref(),
            store: self.store,
            previous_response_id: self.previous_response_id.as_ref(),
            truncation: self.truncation.as_ref(),
            reasoning: self.reasoning.as_ref(),
            include: self.include.as_ref(),
            user: self.user.as_ref(),
            metadata: self.metadata.as_ref(),
            service_tier: self.service_tier.as_ref(),
            parallel_tool_calls: self.parallel_tool_calls,
            max_output_tokens: self.max_output_tokens,
            max_tool_calls: self.max_tool_calls,
            top_logprobs: self.top_logprobs,
            stream: self.stream,
            extra: &self.extra,
        }
        .serialize(serializer)
    }
}

#[derive(Deserialize)]
struct AnnotatedLlmRequestWire {
    messages: Vec<Message>,
    model: Option<String>,
    params: Option<GenerationParams>,
    tools: Option<Vec<ToolDefinition>>,
    tool_choice: Option<ToolChoice>,
    #[serde(default)]
    response_format: ResponseFormatWireField,
    store: Option<bool>,
    previous_response_id: Option<String>,
    truncation: Option<Json>,
    reasoning: Option<Json>,
    include: Option<Json>,
    user: Option<String>,
    metadata: Option<Json>,
    service_tier: Option<String>,
    parallel_tool_calls: Option<bool>,
    max_output_tokens: Option<u64>,
    max_tool_calls: Option<u64>,
    top_logprobs: Option<u64>,
    stream: Option<bool>,
    #[serde(flatten)]
    extra: serde_json::Map<String, Json>,
}

#[derive(Default)]
enum ResponseFormatWireField {
    #[default]
    Missing,
    Present(Json),
}

impl<'de> Deserialize<'de> for ResponseFormatWireField {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Json::deserialize(deserializer).map(Self::Present)
    }
}

impl<'de> Deserialize<'de> for AnnotatedLlmRequest {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut wire = AnnotatedLlmRequestWire::deserialize(deserializer)?;
        let response_format = match wire.response_format {
            ResponseFormatWireField::Missing => None,
            ResponseFormatWireField::Present(value)
                if matches!(
                    value.get("kind").and_then(Json::as_str),
                    Some("json_object" | "json_schema")
                ) =>
            {
                Some(serde_json::from_value(value).map_err(serde::de::Error::custom)?)
            }
            ResponseFormatWireField::Present(value) => {
                wire.extra.insert("response_format".to_string(), value);
                None
            }
        };

        Ok(Self {
            messages: wire.messages,
            model: wire.model,
            params: wire.params,
            tools: wire.tools,
            tool_choice: wire.tool_choice,
            response_format,
            store: wire.store,
            previous_response_id: wire.previous_response_id,
            truncation: wire.truncation,
            reasoning: wire.reasoning,
            include: wire.include,
            user: wire.user,
            metadata: wire.metadata,
            service_tier: wire.service_tier,
            parallel_tool_calls: wire.parallel_tool_calls,
            max_output_tokens: wire.max_output_tokens,
            max_tool_calls: wire.max_tool_calls,
            top_logprobs: wire.top_logprobs,
            stream: wire.stream,
            extra: wire.extra,
        })
    }
}

/// A single message in a conversation, tagged by role.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "role", rename_all = "lowercase")]
pub enum Message {
    /// A system instruction message.
    System {
        /// The message content.
        content: MessageContent,
        /// Optional sender name.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// A developer instruction message.
    Developer {
        /// The message content.
        content: MessageContent,
        /// Optional sender name.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// A user message.
    User {
        /// The message content.
        content: MessageContent,
        /// Optional sender name.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// An assistant response, optionally containing tool calls.
    Assistant {
        /// The message content (optional — may be absent when tool calls are present).
        #[serde(skip_serializing_if = "Option::is_none")]
        content: Option<MessageContent>,
        /// Tool calls requested by the assistant.
        #[serde(skip_serializing_if = "Option::is_none")]
        tool_calls: Option<Vec<ToolCall>>,
        /// Optional sender name.
        #[serde(skip_serializing_if = "Option::is_none")]
        name: Option<String>,
    },
    /// A tool result message.
    Tool {
        /// The tool execution result.
        content: MessageContent,
        /// The ID of the tool call this result corresponds to.
        tool_call_id: String,
    },
}

/// Normalized structured response format kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum StructuredResponseFormatKind {
    /// Request an arbitrary JSON object.
    JsonObject,
    /// Request output that conforms to a JSON Schema.
    JsonSchema,
}

/// Provider-neutral structured response format.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
pub struct StructuredResponseFormat {
    /// Normalized response format kind.
    pub kind: StructuredResponseFormatKind,
    /// Optional provider format name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Optional JSON Schema for structured output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schema: Option<Json>,
    /// Optional strict schema-conformance flag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
    /// Lossless native wrapper and format metadata.
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub extra: serde_json::Map<String, Json>,
}

/// Message content: either a plain string or multimodal parts array.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum MessageContent {
    /// Plain text content.
    Text(String),
    /// Multimodal content parts.
    Parts(Vec<ContentPart>),
}

/// A single content part within a multimodal message.
///
/// v1 supports text only. Future versions may add `ImageUrl`, `Audio`, etc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    /// A text content part.
    Text {
        /// The text content.
        text: String,
    },
    /// An image URL content part.
    ImageUrl {
        /// Image URL payload.
        image_url: OpenAiImageUrl,
    },
}

/// OpenAI image URL payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct OpenAiImageUrl {
    /// URL for the image.
    pub url: String,
    /// Optional provider-specific detail hint.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// A tool call requested by the assistant.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ToolCall {
    /// Unique identifier for this tool call.
    pub id: String,
    /// The type of tool call (typically `"function"`).
    #[serde(rename = "type")]
    pub call_type: String,
    /// The function to call.
    pub function: FunctionCall,
}

/// A function call within a tool call.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct FunctionCall {
    /// The name of the function to call.
    pub name: String,
    /// The function arguments as a JSON string (per OpenAI convention).
    pub arguments: String,
}

/// A tool definition (function schema) available to the model.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ToolDefinition {
    /// The type of tool (typically `"function"`).
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function definition.
    pub function: FunctionDefinition,
}

/// A function definition within a tool definition.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct FunctionDefinition {
    /// The name of the function.
    pub name: String,
    /// A description of what the function does.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// The JSON Schema for the function parameters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Json>,
    /// Whether the provider should enforce strict parameter-schema validation.
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        deserialize_with = "deserialize_optional_non_null_bool"
    )]
    #[cfg_attr(feature = "schema", schemars(with = "bool"))]
    pub strict: Option<bool>,
}

/// Deserialize a boolean field that may be omitted but may not be JSON null.
#[doc(hidden)]
pub fn deserialize_optional_non_null_bool<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: Deserializer<'de>,
{
    bool::deserialize(deserializer).map(Some)
}

/// Tool choice control: how the model should use available tools.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum ToolChoice {
    /// Let the model decide whether to call a tool.
    Auto,
    /// Do not call any tools.
    None,
    /// The model must call at least one tool.
    Required,
    /// Force a specific function by name.
    #[serde(untagged)]
    Specific(ToolChoiceFunction),
}

/// A specific tool choice that forces a named function.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ToolChoiceFunction {
    /// The type (typically `"function"`).
    #[serde(rename = "type")]
    pub choice_type: String,
    /// The function to call.
    pub function: ToolChoiceFunctionName,
}

/// The name component of a specific tool choice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ToolChoiceFunctionName {
    /// The function name.
    pub name: String,
}

/// Normalized generation parameters across providers.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct GenerationParams {
    /// Sampling temperature.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Maximum number of tokens to generate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    /// Nucleus sampling probability.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    /// Stop sequences.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<Vec<String>>,
}

// ---------------------------------------------------------------------------
// Helper methods
// ---------------------------------------------------------------------------

impl AnnotatedLlmRequest {
    /// Extract the text content of the first system message, if any.
    ///
    /// For [`MessageContent::Text`], returns the string directly.
    /// For [`MessageContent::Parts`], returns the text of the first
    /// [`ContentPart::Text`] part.
    pub fn system_prompt(&self) -> Option<&str> {
        self.messages.iter().find_map(|m| match m {
            Message::System { content, .. } => match content {
                MessageContent::Text(s) => Some(s.as_str()),
                MessageContent::Parts(parts) => parts.iter().find_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                }),
            },
            _ => None,
        })
    }

    /// Get the text content of the last user message, if any.
    ///
    /// Searches messages in reverse order and returns the first user
    /// message found. For [`MessageContent::Parts`], returns the text of
    /// the first [`ContentPart::Text`] part.
    pub fn last_user_message(&self) -> Option<&str> {
        self.messages.iter().rev().find_map(|m| match m {
            Message::User { content, .. } => match content {
                MessageContent::Text(s) => Some(s.as_str()),
                MessageContent::Parts(parts) => parts.iter().find_map(|p| match p {
                    ContentPart::Text { text } => Some(text.as_str()),
                    ContentPart::ImageUrl { .. } => None,
                }),
            },
            _ => None,
        })
    }

    /// Check if any assistant message in the conversation contains tool calls.
    ///
    /// Returns `true` if at least one [`Message::Assistant`] variant has a
    /// non-empty `tool_calls` field.
    pub fn has_tool_calls(&self) -> bool {
        self.messages.iter().any(|m| {
            matches!(
                m,
                Message::Assistant { tool_calls: Some(calls), .. } if !calls.is_empty()
            )
        })
    }
}
