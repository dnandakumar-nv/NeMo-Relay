// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit bounded request and routing-context projection contracts.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::sync::Arc;

use nemo_relay::api::llm::{LlmApiFamily, LlmExecutionContextSnapshot, LlmRequest};
use nemo_relay::codec::request::{
    AnnotatedLlmRequest, ContentPart, FunctionDefinition, GenerationParams, Message,
    MessageContent, StructuredResponseFormat, StructuredResponseFormatKind, ToolCall, ToolChoice,
    ToolDefinition,
};
use serde::de::{DeserializeSeed, IgnoredAny, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::Value as Json;
use url::Url;

use crate::adapter::{RouterInstructionFact, RouterInstructionRole, RouterRequestEnvelope};
use crate::config::{CanonicalizerConfig, PoolSelectorConfig};
use crate::eligibility::IneligibilityReason;
use crate::fingerprint::{
    BoundedFingerprintError, canonical_json_bytes, canonical_serialize_bytes, fingerprint_json,
    fingerprint_serializable_bounded, validate_canonical_json_domain,
    validate_json_resource_bounds, validate_serializable_size_bound,
};
use crate::preflight::has_sensitive_value_shape;

/// Versioned serialized request projection schema.
pub const REQUEST_PROJECTION_SCHEMA_V1: &str = "nemo.relay.router.request-projection@1";
/// Versioned serialized routing-context projection schema.
pub const ROUTING_CONTEXT_SCHEMA_V1: &str = "nemo.relay.router.routing-context@1";
/// Version of the deterministic Router request sanitizer.
pub const ROUTER_SANITIZER_VERSION: u32 = 1;
/// Maximum RFC 8785 bytes admitted for complete request projection semantics.
pub(crate) const REQUEST_PROJECTION_MAX_BYTES: usize = 16 * 1024 * 1024;
/// Maximum messages admitted to one request projection.
pub(crate) const REQUEST_PROJECTION_MAX_MESSAGES: usize = 4_096;
/// Maximum JSON values traversed before any Router request clone or parse.
pub(crate) const REQUEST_JSON_MAX_VALUES: usize = 65_536;
/// Maximum JSON nesting traversed before any Router request clone or parse.
pub(crate) const REQUEST_JSON_MAX_DEPTH: usize = 128;

/// Reject raw request graphs that would amplify substantially during codec decode.
pub(crate) fn validate_raw_request_resource_bounds(
    request: &LlmRequest,
) -> Result<(), IneligibilityReason> {
    let content = validate_json_resource_bounds(
        &request.content,
        REQUEST_PROJECTION_MAX_BYTES,
        REQUEST_JSON_MAX_VALUES,
        REQUEST_JSON_MAX_DEPTH,
    )
    .map_err(|_| IneligibilityReason::ProjectionFailed)?;
    let mut remaining_values = REQUEST_JSON_MAX_VALUES
        .checked_sub(content.values)
        .ok_or(IneligibilityReason::ProjectionFailed)?;
    if request.headers.len() > remaining_values {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    for value in request.headers.values() {
        let usage = validate_json_resource_bounds(
            value,
            REQUEST_PROJECTION_MAX_BYTES,
            remaining_values,
            REQUEST_JSON_MAX_DEPTH,
        )
        .map_err(|_| IneligibilityReason::ProjectionFailed)?;
        remaining_values = remaining_values
            .checked_sub(usage.values)
            .ok_or(IneligibilityReason::ProjectionFailed)?;
    }
    validate_serializable_size_bound(request, REQUEST_PROJECTION_MAX_BYTES)
        .map_err(|_| IneligibilityReason::ProjectionFailed)
}

/// Durable, allowlisted normalized request fields.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SanitizedAnnotatedLlmRequest {
    /// Ordered sanitized conversation messages.
    pub messages: Vec<SanitizedMessage>,
    /// Original anchor model identifier.
    pub model: Option<String>,
    /// Common generation parameters.
    pub params: Option<SanitizedGenerationParams>,
    /// Sanitized tool contracts.
    pub tools: Option<Vec<SanitizedToolDefinition>>,
    /// Tool-selection control.
    pub tool_choice: Option<SanitizedToolChoice>,
    /// Structured response contract without native extras.
    pub response_format: Option<SanitizedStructuredResponseFormat>,
    /// Provider truncation control.
    pub truncation: Option<Json>,
    /// Provider reasoning control.
    pub reasoning: Option<Json>,
    /// Service-tier preference.
    pub service_tier: Option<String>,
    /// Whether tools may execute in parallel.
    pub parallel_tool_calls: Option<bool>,
    /// Explicit maximum output tokens.
    pub max_output_tokens: Option<u64>,
    /// Explicit maximum tool calls.
    pub max_tool_calls: Option<u64>,
    /// Requested log-probability fanout.
    pub top_logprobs: Option<u64>,
}

/// Router-owned sanitized message union.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "role", rename_all = "lowercase", deny_unknown_fields)]
pub enum SanitizedMessage {
    /// System instruction.
    System {
        /// Sanitized content.
        content: SanitizedMessageContent,
        /// Optional sender name.
        name: Option<String>,
    },
    /// Developer instruction.
    Developer {
        /// Sanitized content.
        content: SanitizedMessageContent,
        /// Optional sender name.
        name: Option<String>,
    },
    /// User task or context.
    User {
        /// Sanitized content.
        content: SanitizedMessageContent,
        /// Optional sender name.
        name: Option<String>,
    },
    /// Assistant response context.
    Assistant {
        /// Optional sanitized content.
        content: Option<SanitizedMessageContent>,
        /// Optional sanitized tool calls.
        tool_calls: Option<Vec<SanitizedToolCall>>,
        /// Optional sender name.
        name: Option<String>,
    },
    /// Tool result context.
    Tool {
        /// Sanitized tool result content.
        content: SanitizedMessageContent,
        /// Correlated tool-call identifier.
        tool_call_id: String,
    },
}

/// Sanitized text or multimodal message content.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SanitizedMessageContent {
    /// Plain text content.
    Text(String),
    /// Ordered sanitized content parts.
    Parts(Vec<SanitizedContentPart>),
}

/// One sanitized multimodal content part.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum SanitizedContentPart {
    /// Plain text part.
    Text {
        /// Part text.
        text: String,
    },
    /// Image URL with credentials, query, and fragment removed.
    ImageUrl {
        /// Sanitized image URL.
        url: String,
        /// Optional safe provider detail hint.
        detail: Option<String>,
    },
}

/// Sanitized assistant tool invocation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SanitizedToolCall {
    /// Provider tool-call ID.
    pub id: String,
    /// Provider call type.
    pub call_type: String,
    /// Function name.
    pub name: String,
    /// Sanitized JSON argument string.
    pub arguments: String,
}

/// Sanitized tool definition with an explicit schema field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SanitizedToolDefinition {
    /// Tool type.
    pub tool_type: String,
    /// Function name.
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
    /// Sanitized JSON Schema.
    pub parameters: Option<Json>,
    /// Optional strict schema-conformance flag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

/// Sanitized tool-choice representation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum SanitizedToolChoice {
    /// Let the provider choose.
    Auto,
    /// Disable tool selection.
    None,
    /// Require a tool call.
    Required,
    /// Force one function.
    Specific {
        /// Native choice type.
        choice_type: String,
        /// Required function name.
        function_name: String,
    },
}

/// Sanitized common generation controls.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SanitizedGenerationParams {
    /// Sampling temperature.
    pub temperature: Option<f64>,
    /// Maximum generated tokens.
    pub max_tokens: Option<u64>,
    /// Nucleus sampling probability.
    pub top_p: Option<f64>,
    /// Stop sequences.
    pub stop: Option<Vec<String>>,
}

/// Sanitized structured response contract without native wrapper extras.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SanitizedStructuredResponseFormat {
    /// Normalized format kind.
    pub kind: StructuredResponseFormatKind,
    /// Optional format name.
    pub name: Option<String>,
    /// Optional sanitized response JSON Schema.
    pub schema: Option<Json>,
    /// Optional strict-conformance flag.
    pub strict: Option<bool>,
}

/// Sanitized instruction fact retained separately for canonicalization.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SanitizedRouterInstructionFact {
    /// Original normalized wire ordinal.
    pub wire_ordinal: usize,
    /// Original System or Developer role.
    pub role: SanitizedRouterInstructionRole,
    /// Exact bounded instruction text.
    pub content: String,
    /// Optional bounded message name.
    pub name: Option<String>,
}

/// Instruction role retained in a sanitized projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SanitizedRouterInstructionRole {
    /// System instruction.
    System,
    /// Developer instruction.
    Developer,
}

/// Version-1 serializable request projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterRequestProjectionV1 {
    /// Constant schema identifier.
    pub schema: String,
    /// Authoritative API family.
    pub family: LlmApiFamily,
    /// Explicit sanitized semantic request.
    pub normalized_request: SanitizedAnnotatedLlmRequest,
    /// Ordered sanitized instruction facts.
    pub ordered_instructions: Vec<SanitizedRouterInstructionFact>,
    /// Sanitized structured response format.
    pub response_format: Option<SanitizedStructuredResponseFormat>,
    /// Complete response-schema fingerprint.
    pub response_schema_fingerprint: Option<String>,
    /// Sorted unique required capability names.
    pub required_capabilities: Vec<String>,
    /// Sanitizer contract version.
    pub sanitizer_version: u32,
    /// SHA-256 over RFC 8785 projection semantics excluding this field.
    pub semantic_request_fingerprint: String,
}

#[derive(Serialize)]
struct RouterRequestProjectionSemantics<'a> {
    schema: &'a str,
    family: LlmApiFamily,
    normalized_request: &'a SanitizedAnnotatedLlmRequest,
    ordered_instructions: &'a [SanitizedRouterInstructionFact],
    response_format: &'a Option<SanitizedStructuredResponseFormat>,
    response_schema_fingerprint: &'a Option<String>,
    required_capabilities: &'a [String],
    sanitizer_version: u32,
}

impl<'a> From<&'a RouterRequestProjectionV1> for RouterRequestProjectionSemantics<'a> {
    fn from(projection: &'a RouterRequestProjectionV1) -> Self {
        Self {
            schema: &projection.schema,
            family: projection.family,
            normalized_request: &projection.normalized_request,
            ordered_instructions: &projection.ordered_instructions,
            response_format: &projection.response_format,
            response_schema_fingerprint: &projection.response_schema_fingerprint,
            required_capabilities: &projection.required_capabilities,
            sanitizer_version: projection.sanitizer_version,
        }
    }
}

/// Version-1 serializable routing-only context projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RouterRoutingContextProjectionV1 {
    /// Constant schema identifier.
    pub schema: String,
    /// Versioned tenant-policy partition hash.
    pub tenant_policy_hash: String,
    /// Versioned agent-policy partition hash.
    pub agent_policy_hash: String,
    /// Registered bounded scalar position features.
    pub position_features: BTreeMap<String, Json>,
}

/// Build the only serializable request representation accepted by Router.
pub(crate) fn project_request(
    envelope: &RouterRequestEnvelope,
    _limits: &CanonicalizerConfig,
) -> Result<RouterRequestProjectionV1, IneligibilityReason> {
    if envelope.normalized_request.previous_response_id.is_some()
        || envelope.normalized_request.store == Some(true)
        || envelope.normalized_request.stream == Some(true)
    {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    reject_capability_bearing_extras(envelope)?;
    if envelope.normalized_request.messages.len() > REQUEST_PROJECTION_MAX_MESSAGES {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    validate_serializable_size_bound(&envelope.normalized_request, REQUEST_PROJECTION_MAX_BYTES)
        .map_err(|_| IneligibilityReason::ProjectionFailed)?;
    envelope
        .ordered_instructions
        .iter()
        .try_fold(0usize, |total, instruction| {
            total
                .checked_add(instruction.content.len())
                .and_then(|total| {
                    total.checked_add(instruction.name.as_deref().map_or(0, str::len))
                })
                .filter(|total| *total <= REQUEST_PROJECTION_MAX_BYTES)
                .ok_or(IneligibilityReason::ProjectionFailed)
        })?;

    let normalized_request = sanitize_annotated_request(&envelope.normalized_request)?;

    let response_format = envelope
        .response_format
        .as_ref()
        .map(sanitize_response_format)
        .transpose()?;
    let ordered_instructions = envelope
        .ordered_instructions
        .iter()
        .map(sanitize_instruction)
        .collect();
    let required_capabilities = required_capabilities(&normalized_request);
    let mut projection = RouterRequestProjectionV1 {
        schema: REQUEST_PROJECTION_SCHEMA_V1.to_string(),
        family: envelope.family,
        normalized_request,
        ordered_instructions,
        response_format,
        response_schema_fingerprint: envelope.response_schema_fingerprint.clone(),
        required_capabilities,
        sanitizer_version: ROUTER_SANITIZER_VERSION,
        semantic_request_fingerprint: String::new(),
    };
    projection.semantic_request_fingerprint = projection_semantic_fingerprint(&projection)?;
    Ok(projection)
}

/// Derive the canonical candidate request projection from a validated anchor projection.
pub(crate) fn candidate_request_projection(
    anchor: &RouterRequestProjectionV1,
    candidate_model: &str,
) -> Result<RouterRequestProjectionV1, ()> {
    validate_request_projection(anchor).map_err(|_| ())?;
    let mut candidate = anchor.clone();
    candidate.normalized_request.model = Some(candidate_model.to_string());
    candidate.semantic_request_fingerprint =
        projection_semantic_fingerprint(&candidate).map_err(|_| ())?;
    Ok(candidate)
}

/// Freeze the policy-only portion of the V2 context for delayed Router work.
pub(crate) fn project_routing_context(
    context: &LlmExecutionContextSnapshot,
    selector: &PoolSelectorConfig,
    limits: &CanonicalizerConfig,
) -> Result<RouterRoutingContextProjectionV1, IneligibilityReason> {
    let tenant_policy_hash = policy_hash(
        "tenant",
        selector.tenant_ids.as_deref(),
        context.tenant_id.as_deref(),
    )?;
    let agent_policy_hash = policy_hash(
        "agent",
        selector.agent_ids.as_deref(),
        context.agent_id.as_deref(),
    )?;

    let mut position_features = BTreeMap::new();
    for name in &limits.position_features {
        let value = context
            .sanitized_metadata
            .get(name)
            .ok_or(IneligibilityReason::ProjectionFailed)?;
        let Some(turn_index) = value.as_u64().filter(|_| name == "turn_index") else {
            return Err(IneligibilityReason::ProjectionFailed);
        };
        // RFC 8785 uses IEEE-754 number serialization. Decimal strings retain
        // the full accepted u64 domain without collisions above 2^53.
        position_features.insert(name.clone(), Json::String(turn_index.to_string()));
    }
    let encoded = canonical_serialize_bytes(&position_features)
        .map_err(|_| IneligibilityReason::ProjectionFailed)?;
    if encoded.len() > limits.max_position_features_bytes {
        return Err(IneligibilityReason::ProjectionFailed);
    }

    Ok(RouterRoutingContextProjectionV1 {
        schema: ROUTING_CONTEXT_SCHEMA_V1.to_string(),
        tenant_policy_hash,
        agent_policy_hash,
        position_features,
    })
}

fn policy_hash(
    dimension: &str,
    configured: Option<&[String]>,
    actual: Option<&str>,
) -> Result<String, IneligibilityReason> {
    let value = if let Some(configured) = configured {
        let actual = actual.ok_or(IneligibilityReason::ProjectionFailed)?;
        if !configured.iter().any(|candidate| candidate == actual) {
            return Err(IneligibilityReason::ProjectionFailed);
        }
        let mut configured = configured.to_vec();
        configured.sort();
        serde_json::json!({
            "schema": format!("nemo.relay.router.{dimension}-policy@1"),
            "selector": configured,
            "identity": actual,
        })
    } else {
        serde_json::json!({
            "schema": format!("nemo.relay.router.{dimension}-policy@1"),
            "selector": "pool_shared_v1",
        })
    };
    fingerprint_json(&value).map_err(|_| IneligibilityReason::ProjectionFailed)
}

fn sanitize_instruction(fact: &RouterInstructionFact) -> SanitizedRouterInstructionFact {
    SanitizedRouterInstructionFact {
        wire_ordinal: fact.wire_ordinal,
        role: match fact.role {
            RouterInstructionRole::System => SanitizedRouterInstructionRole::System,
            RouterInstructionRole::Developer => SanitizedRouterInstructionRole::Developer,
        },
        content: fact.content.clone(),
        name: fact.name.clone(),
    }
}

/// Build the explicit safe subset of one normalized request annotation.
///
/// Event projection reuses this allowlist so runtime codec annotations cannot
/// smuggle provider extras or transport state into trajectory evidence.
pub(crate) fn sanitize_annotated_request(
    request: &AnnotatedLlmRequest,
) -> Result<SanitizedAnnotatedLlmRequest, IneligibilityReason> {
    let secret_names = request
        .tools
        .as_deref()
        .map(collect_secret_names)
        .transpose()?
        .unwrap_or_default();
    let tools = request.tools.as_deref().map(sanitize_tools).transpose()?;
    let messages = request
        .messages
        .iter()
        .map(|message| sanitize_message(message, &secret_names))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(SanitizedAnnotatedLlmRequest {
        messages,
        model: request.model.clone(),
        params: request.params.as_ref().map(sanitize_generation_params),
        tools,
        tool_choice: request.tool_choice.as_ref().map(sanitize_tool_choice),
        response_format: request
            .response_format
            .as_ref()
            .map(sanitize_response_format)
            .transpose()?,
        truncation: request.truncation.as_ref().map(sanitize_json).transpose()?,
        reasoning: request.reasoning.as_ref().map(sanitize_json).transpose()?,
        service_tier: request.service_tier.clone(),
        parallel_tool_calls: request.parallel_tool_calls,
        max_output_tokens: request.max_output_tokens,
        max_tool_calls: request.max_tool_calls,
        top_logprobs: request.top_logprobs,
    })
}

fn sanitize_generation_params(params: &GenerationParams) -> SanitizedGenerationParams {
    SanitizedGenerationParams {
        temperature: params.temperature,
        max_tokens: params.max_tokens,
        top_p: params.top_p,
        stop: params.stop.clone(),
    }
}

fn sanitize_response_format(
    format: &StructuredResponseFormat,
) -> Result<SanitizedStructuredResponseFormat, IneligibilityReason> {
    Ok(SanitizedStructuredResponseFormat {
        kind: format.kind,
        name: format.name.clone(),
        schema: format
            .schema
            .as_ref()
            .map(sanitize_tool_schema)
            .transpose()?,
        strict: format.strict,
    })
}

fn sanitize_tool_choice(choice: &ToolChoice) -> SanitizedToolChoice {
    match choice {
        ToolChoice::Auto => SanitizedToolChoice::Auto,
        ToolChoice::None => SanitizedToolChoice::None,
        ToolChoice::Required => SanitizedToolChoice::Required,
        ToolChoice::Specific(specific) => SanitizedToolChoice::Specific {
            choice_type: specific.choice_type.clone(),
            function_name: specific.function.name.clone(),
        },
    }
}

fn sanitize_tools(
    tools: &[ToolDefinition],
) -> Result<Vec<SanitizedToolDefinition>, IneligibilityReason> {
    tools
        .iter()
        .map(|tool| sanitize_tool(&tool.tool_type, &tool.function))
        .collect()
}

fn sanitize_tool(
    tool_type: &str,
    function: &FunctionDefinition,
) -> Result<SanitizedToolDefinition, IneligibilityReason> {
    let parameters = function
        .parameters
        .as_ref()
        .map(sanitize_tool_schema)
        .transpose()?;
    Ok(SanitizedToolDefinition {
        tool_type: tool_type.to_string(),
        name: function.name.clone(),
        description: function.description.clone(),
        parameters,
        strict: function.strict,
    })
}

fn sanitize_tool_schema(schema: &Json) -> Result<Json, IneligibilityReason> {
    validate_schema_sanitizer_input(schema)?;
    validate_canonical_json_domain(schema).map_err(|_| IneligibilityReason::ProjectionFailed)?;
    let sensitive_paths = sensitive_schema_paths(schema)?;
    let mut schema = schema.clone();
    if !sanitize_schema_node(&mut schema, &sensitive_paths, "#") {
        schema = Json::Object(Default::default());
    }
    canonical_json_bytes(&schema).map_err(|_| IneligibilityReason::ProjectionFailed)?;
    Ok(schema)
}

const SCHEMA_MAP_CONTAINERS: [&str; 5] = [
    "properties",
    "$defs",
    "definitions",
    "patternProperties",
    "dependentSchemas",
];

const SCHEMA_REFERENCE_KEYWORDS: [&str; 3] = ["$ref", "$dynamicRef", "$recursiveRef"];
const SCHEMA_ANCHOR_KEYWORDS: [&str; 2] = ["$anchor", "$dynamicAnchor"];
const SCHEMA_ARRAY_KEYWORDS: [&str; 4] = ["allOf", "anyOf", "oneOf", "prefixItems"];
const SCHEMA_SINGLE_KEYWORDS: [&str; 12] = [
    "additionalItems",
    "additionalProperties",
    "contains",
    "contentSchema",
    "else",
    "if",
    "not",
    "propertyNames",
    "then",
    "unevaluatedItems",
    "unevaluatedProperties",
    "items",
];
const MAX_SCHEMA_SANITIZER_BYTES: usize = 16 * 1024 * 1024;
const MAX_SCHEMA_SANITIZER_VALUES: usize = 65_536;
const MAX_SCHEMA_SANITIZER_DEPTH: usize = 128;
const MAX_SCHEMA_SANITIZER_GRAPH_EDGES: usize = 4 * MAX_SCHEMA_SANITIZER_VALUES;
const MAX_SCHEMA_SANITIZER_GRAPH_WORK: usize = 8 * MAX_SCHEMA_SANITIZER_VALUES;
const MAX_SCHEMA_IDENTIFIER_BYTES: usize = 4 * 1024;
const MAX_SCHEMA_REFERENCE_BYTES: usize = 16 * 1024;
const MAX_SCHEMA_REFERENCE_SEGMENTS: usize = 1_024;
const MAX_SCHEMA_RESOURCE_URI_BYTES: usize = 16 * 1024;
const SCHEMA_SANITIZER_BASE_URI: &str = "https://nemo-relay.invalid/router-schema/root";

#[derive(Clone)]
struct SchemaResourceScope {
    base: Arc<Url>,
    resource: Arc<str>,
    pointer_roots: Vec<(Arc<str>, Arc<str>)>,
}

struct SchemaGraphNode {
    path: String,
    base: Arc<Url>,
    references: Vec<String>,
    initially_sensitive: bool,
}

#[derive(Clone, Copy)]
struct SchemaVisitOwnership<'a> {
    parent: Option<usize>,
    named_owner: Option<usize>,
    map_entry_name: Option<&'a str>,
    is_root: bool,
}

impl SchemaVisitOwnership<'_> {
    fn root() -> Self {
        Self {
            parent: None,
            named_owner: None,
            map_entry_name: None,
            is_root: true,
        }
    }

    fn child(
        parent: usize,
        named_owner: Option<usize>,
        map_entry_name: Option<&str>,
    ) -> SchemaVisitOwnership<'_> {
        SchemaVisitOwnership {
            parent: Some(parent),
            named_owner,
            map_entry_name,
            is_root: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum SchemaReferenceFragment {
    Pointer(String),
    Anchor(String),
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct SchemaReferenceTarget {
    resource: String,
    fragment: SchemaReferenceFragment,
}

struct SchemaGraphBuilder {
    nodes: Vec<SchemaGraphNode>,
    dependents: Vec<Vec<usize>>,
    pointer_targets: BTreeMap<Arc<str>, BTreeMap<String, usize>>,
    anchor_targets: BTreeMap<Arc<str>, BTreeMap<String, usize>>,
    resource_roots: BTreeMap<Arc<str>, Arc<str>>,
    edge_count: usize,
    work_count: usize,
}

impl SchemaGraphBuilder {
    fn new() -> Self {
        Self {
            nodes: Vec::new(),
            dependents: Vec::new(),
            pointer_targets: BTreeMap::new(),
            anchor_targets: BTreeMap::new(),
            resource_roots: BTreeMap::new(),
            edge_count: 0,
            work_count: 0,
        }
    }

    fn visit_schema(
        &mut self,
        schema: &Json,
        path: &str,
        inherited_scope: &SchemaResourceScope,
        ownership: SchemaVisitOwnership<'_>,
    ) -> Result<(), IneligibilityReason> {
        let SchemaVisitOwnership {
            parent,
            named_owner,
            map_entry_name,
            is_root,
        } = ownership;
        let Json::Object(object) = schema else {
            return Ok(());
        };
        let (scope, identifier_anchor) =
            self.resolve_scope(object, path, inherited_scope, is_root)?;
        let mut references = Vec::new();
        for reference in SCHEMA_REFERENCE_KEYWORDS
            .iter()
            .filter_map(|keyword| object.get(*keyword).and_then(Json::as_str))
        {
            let segments = validate_schema_reference_shape(reference)?;
            self.charge_work(reference.len().saturating_add(segments))?;
            references.push(reference.to_string());
        }
        let initially_sensitive = is_secret_schema(schema)
            || map_entry_name.is_some_and(is_sensitive_projection_key)
            || schema_has_sensitive_reference_identity(schema)
            || references
                .iter()
                .any(|reference| reference_identity_is_sensitive(reference));
        let node_id = self.nodes.len();
        let current_named_owner = if map_entry_name.is_some() {
            Some(node_id)
        } else {
            named_owner
        };
        let has_references = !references.is_empty();
        self.nodes.push(SchemaGraphNode {
            path: path.to_string(),
            base: scope.base.clone(),
            references,
            initially_sensitive,
        });
        self.dependents.push(Vec::new());
        if let Some(parent) = parent {
            self.add_edge(parent, node_id)?;
        }
        if has_references
            && let Some(owner) = current_named_owner
            && owner != node_id
        {
            self.add_edge(node_id, owner)?;
        }

        for (resource, root_path) in &scope.pointer_roots {
            self.charge_work(1)?;
            let Some(pointer) = schema_pointer_for_path(path, root_path) else {
                return Err(IneligibilityReason::ProjectionFailed);
            };
            if self
                .pointer_targets
                .entry(resource.clone())
                .or_default()
                .insert(pointer, node_id)
                .is_some()
            {
                return Err(IneligibilityReason::ProjectionFailed);
            }
        }
        for keyword in SCHEMA_ANCHOR_KEYWORDS {
            if let Some(anchor) = object.get(keyword).and_then(Json::as_str) {
                self.register_anchor(&scope.resource, anchor, node_id)?;
            }
        }
        if let Some(anchor) = identifier_anchor {
            self.register_anchor(&scope.resource, &anchor, node_id)?;
        }

        for container in SCHEMA_MAP_CONTAINERS {
            if let Some(Json::Object(schemas)) = object.get(container) {
                let container_path = json_pointer_child(path, container);
                for (name, schema) in schemas {
                    self.visit_schema(
                        schema,
                        &json_pointer_child(&container_path, name),
                        &scope,
                        SchemaVisitOwnership::child(node_id, current_named_owner, Some(name)),
                    )?;
                }
            }
        }
        if let Some(Json::Object(dependencies)) = object.get("dependencies") {
            let container_path = json_pointer_child(path, "dependencies");
            for (name, dependency) in dependencies {
                if dependency.is_object() {
                    self.visit_schema(
                        dependency,
                        &json_pointer_child(&container_path, name),
                        &scope,
                        SchemaVisitOwnership::child(node_id, current_named_owner, Some(name)),
                    )?;
                }
            }
        }
        for keyword in SCHEMA_ARRAY_KEYWORDS {
            if let Some(Json::Array(schemas)) = object.get(keyword) {
                let keyword_path = json_pointer_child(path, keyword);
                for (index, schema) in schemas.iter().enumerate() {
                    self.visit_schema(
                        schema,
                        &format!("{keyword_path}/{index}"),
                        &scope,
                        SchemaVisitOwnership::child(node_id, current_named_owner, None),
                    )?;
                }
            }
        }
        for keyword in SCHEMA_SINGLE_KEYWORDS {
            if let Some(schema) = object.get(keyword) {
                match schema {
                    Json::Array(schemas) if keyword == "items" => {
                        let keyword_path = json_pointer_child(path, keyword);
                        for (index, schema) in schemas.iter().enumerate() {
                            self.visit_schema(
                                schema,
                                &format!("{keyword_path}/{index}"),
                                &scope,
                                SchemaVisitOwnership::child(node_id, current_named_owner, None),
                            )?;
                        }
                    }
                    Json::Object(_) => self.visit_schema(
                        schema,
                        &json_pointer_child(path, keyword),
                        &scope,
                        SchemaVisitOwnership::child(node_id, current_named_owner, None),
                    )?,
                    Json::Null
                    | Json::Bool(_)
                    | Json::Number(_)
                    | Json::String(_)
                    | Json::Array(_) => {}
                }
            }
        }
        Ok(())
    }

    fn resolve_scope(
        &mut self,
        object: &serde_json::Map<String, Json>,
        path: &str,
        inherited: &SchemaResourceScope,
        is_root: bool,
    ) -> Result<(SchemaResourceScope, Option<String>), IneligibilityReason> {
        let Some(identifier) = object.get("$id").and_then(Json::as_str) else {
            if is_root {
                self.register_resource_root(&inherited.resource, path)?;
            }
            return Ok((inherited.clone(), None));
        };
        let segments = validate_schema_identifier_shape(identifier)?;
        self.charge_work(identifier.len().saturating_add(segments))?;
        let resolved = inherited
            .base
            .join(identifier)
            .map_err(|_| IneligibilityReason::ProjectionFailed)?;
        if resolved.as_str().len() > MAX_SCHEMA_RESOURCE_URI_BYTES {
            return Err(IneligibilityReason::ProjectionFailed);
        }
        let resource = Arc::<str>::from(schema_resource_uri(&resolved));
        let identifier_anchor = resolved
            .fragment()
            .filter(|fragment| !fragment.is_empty())
            .map(percent_decode_fragment)
            .transpose()
            .map_err(|()| IneligibilityReason::ProjectionFailed)?;
        if identifier_anchor
            .as_deref()
            .is_some_and(|fragment| fragment.starts_with('/'))
        {
            return Err(IneligibilityReason::ProjectionFailed);
        }

        let mut scope = inherited.clone();
        scope.base = Arc::new(resolved);
        if is_root {
            scope.resource = resource.clone();
            scope.pointer_roots = vec![(resource.clone(), Arc::<str>::from(path))];
            self.register_resource_root(&resource, path)?;
        } else if resource.as_ref() != inherited.resource.as_ref() {
            self.register_resource_root(&resource, path)?;
            scope.resource = resource.clone();
            scope
                .pointer_roots
                .push((resource.clone(), Arc::<str>::from(path)));
        } else if !identifier.is_empty() && !identifier.starts_with('#') {
            return Err(IneligibilityReason::ProjectionFailed);
        }
        Ok((scope, identifier_anchor))
    }

    fn register_resource_root(
        &mut self,
        resource: &Arc<str>,
        path: &str,
    ) -> Result<(), IneligibilityReason> {
        if let Some(existing) = self.resource_roots.get(resource.as_ref()) {
            if existing.as_ref() != path {
                return Err(IneligibilityReason::ProjectionFailed);
            }
        } else {
            self.resource_roots
                .insert(resource.clone(), Arc::<str>::from(path));
        }
        Ok(())
    }

    fn register_anchor(
        &mut self,
        resource: &Arc<str>,
        anchor: &str,
        node_id: usize,
    ) -> Result<(), IneligibilityReason> {
        self.charge_work(anchor.len().saturating_add(1))?;
        if anchor.is_empty()
            || anchor.len() > MAX_SCHEMA_IDENTIFIER_BYTES
            || anchor.chars().any(char::is_control)
        {
            return Err(IneligibilityReason::ProjectionFailed);
        }
        if self
            .anchor_targets
            .entry(resource.clone())
            .or_default()
            .insert(anchor.to_string(), node_id)
            .is_some()
        {
            return Err(IneligibilityReason::ProjectionFailed);
        }
        Ok(())
    }

    fn add_edge(&mut self, source: usize, dependent: usize) -> Result<(), IneligibilityReason> {
        self.charge_work(1)?;
        self.edge_count = self
            .edge_count
            .checked_add(1)
            .ok_or(IneligibilityReason::ProjectionFailed)?;
        if self.edge_count > MAX_SCHEMA_SANITIZER_GRAPH_EDGES {
            return Err(IneligibilityReason::ProjectionFailed);
        }
        self.dependents[source].push(dependent);
        Ok(())
    }

    fn charge_work(&mut self, units: usize) -> Result<(), IneligibilityReason> {
        self.work_count = self
            .work_count
            .checked_add(units)
            .ok_or(IneligibilityReason::ProjectionFailed)?;
        if self.work_count > MAX_SCHEMA_SANITIZER_GRAPH_WORK {
            return Err(IneligibilityReason::ProjectionFailed);
        }
        Ok(())
    }

    fn finish(mut self) -> Result<BTreeSet<String>, IneligibilityReason> {
        for source in 0..self.nodes.len() {
            let base = self.nodes[source].base.clone();
            let references = self.nodes[source].references.clone();
            for reference in references {
                self.charge_work(1)?;
                let target = match resolve_schema_reference(&base, &reference) {
                    Ok(target) => target,
                    Err(()) => {
                        self.nodes[source].initially_sensitive = true;
                        continue;
                    }
                };
                let target_node = match target.fragment {
                    SchemaReferenceFragment::Pointer(pointer) => {
                        self.find_pointer_target(&target.resource, &pointer)
                    }
                    SchemaReferenceFragment::Anchor(anchor) => self
                        .anchor_targets
                        .get(target.resource.as_str())
                        .and_then(|anchors| anchors.get(anchor.as_str()))
                        .copied(),
                };
                if let Some(target_node) = target_node {
                    self.add_edge(target_node, source)?;
                }
            }
        }

        let mut sensitive = vec![false; self.nodes.len()];
        let mut work = VecDeque::new();
        for (node_id, node) in self.nodes.iter().enumerate() {
            if node.initially_sensitive {
                sensitive[node_id] = true;
                work.push_back(node_id);
            }
        }
        while let Some(node_id) = work.pop_front() {
            for &dependent in &self.dependents[node_id] {
                if !sensitive[dependent] {
                    sensitive[dependent] = true;
                    work.push_back(dependent);
                }
            }
        }
        Ok(self
            .nodes
            .into_iter()
            .zip(sensitive)
            .filter_map(|(node, sensitive)| sensitive.then_some(node.path))
            .collect())
    }

    fn find_pointer_target(&self, resource: &str, pointer: &str) -> Option<usize> {
        let pointers = self.pointer_targets.get(resource)?;
        let mut candidate = pointer;
        loop {
            if let Some(target) = pointers.get(candidate) {
                return Some(*target);
            }
            let boundary = candidate.rfind('/')?;
            candidate = &candidate[..boundary];
        }
    }
}

fn sensitive_schema_paths(root: &Json) -> Result<BTreeSet<String>, IneligibilityReason> {
    let base =
        Url::parse(SCHEMA_SANITIZER_BASE_URI).map_err(|_| IneligibilityReason::ProjectionFailed)?;
    let resource = Arc::<str>::from(schema_resource_uri(&base));
    let scope = SchemaResourceScope {
        base: Arc::new(base),
        resource: resource.clone(),
        pointer_roots: vec![(resource, Arc::<str>::from("#"))],
    };
    let mut graph = SchemaGraphBuilder::new();
    graph.visit_schema(root, "#", &scope, SchemaVisitOwnership::root())?;
    graph.finish()
}

fn schema_resource_uri(url: &Url) -> String {
    let mut resource = url.clone();
    resource.set_fragment(None);
    resource.to_string()
}

fn schema_pointer_for_path(path: &str, resource_root: &str) -> Option<String> {
    if path == resource_root {
        return Some("#".to_string());
    }
    path.strip_prefix(resource_root)
        .filter(|suffix| suffix.starts_with('/'))
        .map(|suffix| format!("#{suffix}"))
}

fn resolve_schema_reference(base: &Url, reference: &str) -> Result<SchemaReferenceTarget, ()> {
    let resolved = base.join(reference).map_err(|_| ())?;
    if resolved.as_str().len() > MAX_SCHEMA_RESOURCE_URI_BYTES + MAX_SCHEMA_REFERENCE_BYTES {
        return Err(());
    }
    let resource = schema_resource_uri(&resolved);
    let fragment = percent_decode_fragment(resolved.fragment().unwrap_or_default())?;
    let fragment = if fragment.is_empty() {
        SchemaReferenceFragment::Pointer("#".to_string())
    } else if let Some(pointer) = fragment.strip_prefix('/') {
        SchemaReferenceFragment::Pointer(normalize_json_pointer(pointer)?)
    } else {
        SchemaReferenceFragment::Anchor(fragment)
    };
    Ok(SchemaReferenceTarget { resource, fragment })
}

fn validate_schema_identifier_shape(identifier: &str) -> Result<usize, IneligibilityReason> {
    if identifier.len() > MAX_SCHEMA_IDENTIFIER_BYTES {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    bounded_reference_segments(identifier).map_err(|()| IneligibilityReason::ProjectionFailed)
}

fn validate_schema_reference_shape(reference: &str) -> Result<usize, IneligibilityReason> {
    if reference.len() > MAX_SCHEMA_REFERENCE_BYTES {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    bounded_reference_segments(reference).map_err(|()| IneligibilityReason::ProjectionFailed)
}

fn bounded_reference_segments(value: &str) -> Result<usize, ()> {
    let segments = value
        .bytes()
        .filter(|byte| *byte == b'/')
        .count()
        .checked_add(1)
        .ok_or(())?;
    (segments <= MAX_SCHEMA_REFERENCE_SEGMENTS)
        .then_some(segments)
        .ok_or(())
}

fn normalize_json_pointer(pointer: &str) -> Result<String, ()> {
    let segments = bounded_reference_segments(pointer)?;
    let mut normalized = String::with_capacity(pointer.len().saturating_add(2));
    normalized.push('#');
    for segment in pointer.split('/').take(segments) {
        validate_json_pointer_segment(segment)?;
        normalized.push('/');
        normalized.push_str(segment);
    }
    Ok(normalized)
}

fn validate_json_pointer_segment(segment: &str) -> Result<(), ()> {
    let mut characters = segment.chars();
    while let Some(character) = characters.next() {
        if character == '~' && !matches!(characters.next(), Some('0' | '1')) {
            return Err(());
        }
    }
    Ok(())
}

fn validate_schema_sanitizer_input(schema: &Json) -> Result<(), IneligibilityReason> {
    let mut stack = vec![(schema, 0usize)];
    let mut values = 0usize;
    let mut bytes = 0usize;
    while let Some((value, depth)) = stack.pop() {
        values = values
            .checked_add(1)
            .ok_or(IneligibilityReason::ProjectionFailed)?;
        if values > MAX_SCHEMA_SANITIZER_VALUES || depth > MAX_SCHEMA_SANITIZER_DEPTH {
            return Err(IneligibilityReason::ProjectionFailed);
        }
        let additional = match value {
            Json::Null => 4,
            Json::Bool(true) => 4,
            Json::Bool(false) => 5,
            Json::Number(number) => number.to_string().len(),
            Json::String(value) => encoded_json_string_bytes(value)?,
            Json::Array(items) => {
                stack.extend(items.iter().map(|item| (item, depth + 1)));
                2usize
                    .checked_add(items.len().saturating_sub(1))
                    .ok_or(IneligibilityReason::ProjectionFailed)?
            }
            Json::Object(object) => {
                let mut object_bytes = 2usize
                    .checked_add(object.len().saturating_sub(1))
                    .ok_or(IneligibilityReason::ProjectionFailed)?;
                for (key, value) in object {
                    object_bytes = object_bytes
                        .checked_add(encoded_json_string_bytes(key)?)
                        .and_then(|value| value.checked_add(1))
                        .ok_or(IneligibilityReason::ProjectionFailed)?;
                    stack.push((value, depth + 1));
                }
                object_bytes
            }
        };
        bytes = bytes
            .checked_add(additional)
            .ok_or(IneligibilityReason::ProjectionFailed)?;
        if bytes > MAX_SCHEMA_SANITIZER_BYTES {
            return Err(IneligibilityReason::ProjectionFailed);
        }
    }
    Ok(())
}

fn encoded_json_string_bytes(value: &str) -> Result<usize, IneligibilityReason> {
    let mut bytes = 2usize;
    for character in value.chars() {
        let encoded = match character {
            '"' | '\\' | '\u{0008}' | '\u{000c}' | '\n' | '\r' | '\t' => 2,
            '\u{0000}'..='\u{001f}' => 6,
            _ => character.len_utf8(),
        };
        bytes = bytes
            .checked_add(encoded)
            .ok_or(IneligibilityReason::ProjectionFailed)?;
    }
    Ok(bytes)
}

fn schema_has_sensitive_reference_identity(value: &Json) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    SCHEMA_ANCHOR_KEYWORDS.iter().any(|keyword| {
        object
            .get(*keyword)
            .and_then(Json::as_str)
            .is_some_and(schema_identifier_is_sensitive)
    }) || SCHEMA_REFERENCE_KEYWORDS.iter().any(|keyword| {
        object
            .get(*keyword)
            .and_then(Json::as_str)
            .is_some_and(reference_identity_is_sensitive)
    }) || object
        .get("$id")
        .and_then(Json::as_str)
        .is_some_and(reference_identity_is_sensitive)
}

fn reference_identity_is_sensitive(reference: &str) -> bool {
    if reference.len() > MAX_SCHEMA_REFERENCE_BYTES || has_sensitive_value_shape(reference) {
        return true;
    }
    match normalize_local_reference(reference) {
        Ok(Some(reference)) => normalized_local_reference_is_sensitive(&reference),
        Ok(None) => reference
            .split(['#', '/', '?', '&', '=', ':', '@'])
            .any(schema_identifier_is_sensitive),
        Err(()) => true,
    }
}

fn normalized_local_reference_is_sensitive(reference: &str) -> bool {
    if reference == "#" {
        return false;
    }
    if let Some(pointer) = reference.strip_prefix("#/") {
        return pointer
            .split('/')
            .any(|segment| match decode_json_pointer_segment(segment) {
                Ok(segment) => schema_identifier_is_sensitive(&segment),
                Err(()) => true,
            });
    }
    reference
        .strip_prefix('#')
        .is_none_or(schema_identifier_is_sensitive)
}

fn schema_identifier_is_sensitive(identifier: &str) -> bool {
    is_sensitive_projection_key(identifier) || has_sensitive_value_shape(identifier)
}

fn normalize_local_reference(reference: &str) -> Result<Option<String>, ()> {
    let Some(fragment) = reference.strip_prefix('#') else {
        return Ok(None);
    };
    let fragment = percent_decode_fragment(fragment)?;
    if fragment.is_empty() {
        return Ok(Some("#".to_string()));
    }
    if let Some(pointer) = fragment.strip_prefix('/') {
        return normalize_json_pointer(pointer).map(Some);
    }
    if fragment.chars().any(char::is_control) {
        return Err(());
    }
    Ok(Some(format!("#{fragment}")))
}

fn percent_decode_fragment(fragment: &str) -> Result<String, ()> {
    let bytes = fragment.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = bytes.get(index + 1).and_then(|byte| hex_value(*byte));
            let low = bytes.get(index + 2).and_then(|byte| hex_value(*byte));
            let (Some(high), Some(low)) = (high, low) else {
                return Err(());
            };
            decoded.push((high << 4) | low);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).map_err(|_| ())
}

fn hex_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn decode_json_pointer_segment(segment: &str) -> Result<String, ()> {
    let mut decoded = String::with_capacity(segment.len());
    let mut characters = segment.chars();
    while let Some(character) = characters.next() {
        if character != '~' {
            decoded.push(character);
            continue;
        }
        match characters.next() {
            Some('0') => decoded.push('~'),
            Some('1') => decoded.push('/'),
            _ => return Err(()),
        }
    }
    Ok(decoded)
}

fn sanitize_schema_node(value: &mut Json, sensitive: &BTreeSet<String>, path: &str) -> bool {
    if sensitive.contains(path)
        || is_secret_schema(value)
        || schema_has_sensitive_reference_identity(value)
    {
        return false;
    }

    match value {
        Json::Array(values) => {
            let mut index = 0;
            values.retain_mut(|value| {
                let retain = sanitize_schema_node(value, sensitive, &format!("{path}/{index}"));
                index += 1;
                retain
            });
        }
        Json::Object(object) => {
            let mut removed_properties = BTreeSet::new();
            for container in SCHEMA_MAP_CONTAINERS {
                let container_path = json_pointer_child(path, container);
                let removed = if let Some(Json::Object(schemas)) = object.get_mut(container) {
                    let mut removed = schemas
                        .keys()
                        .filter(|name| {
                            sensitive.contains(&json_pointer_child(&container_path, name))
                                || schema_identifier_is_sensitive(name)
                                || (container == "dependentSchemas"
                                    && removed_properties.contains(*name))
                        })
                        .cloned()
                        .collect::<BTreeSet<_>>();
                    for name in &removed {
                        schemas.remove(name);
                    }
                    let rejected = schemas
                        .iter_mut()
                        .filter_map(|(name, schema)| {
                            (!sanitize_schema_node(
                                schema,
                                sensitive,
                                &json_pointer_child(&container_path, name),
                            ))
                            .then(|| name.clone())
                        })
                        .collect::<Vec<_>>();
                    for name in rejected {
                        schemas.remove(&name);
                        removed.insert(name);
                    }
                    removed
                } else {
                    BTreeSet::new()
                };
                if container == "properties" {
                    removed_properties.extend(removed);
                }
            }
            prune_schema_property_names(object, &removed_properties);
            let rejected = object
                .iter_mut()
                .filter_map(|(key, value)| {
                    (!SCHEMA_MAP_CONTAINERS.contains(&key.as_str())
                        && !sanitize_schema_node(value, sensitive, &json_pointer_child(path, key)))
                    .then(|| key.clone())
                })
                .collect::<Vec<_>>();
            for key in rejected {
                object.remove(&key);
            }
        }
        Json::Null | Json::Bool(_) | Json::Number(_) | Json::String(_) => {}
    }
    true
}

fn prune_schema_property_names(
    object: &mut serde_json::Map<String, Json>,
    removed_properties: &BTreeSet<String>,
) {
    if let Some(Json::Array(required)) = object.get_mut("required") {
        prune_schema_name_array(required, removed_properties);
    }
    for keyword in ["dependentRequired", "dependencies"] {
        let Some(Json::Object(dependencies)) = object.get_mut(keyword) else {
            continue;
        };
        dependencies.retain(|name, dependency| {
            if !schema_property_name_is_safe(name, removed_properties) {
                return false;
            }
            if let Json::Array(names) = dependency {
                prune_schema_name_array(names, removed_properties);
                return !names.is_empty();
            }
            true
        });
    }
    let remove_property_names = object
        .get_mut("propertyNames")
        .and_then(Json::as_object_mut)
        .is_some_and(|property_names| {
            if let Some(Json::Array(names)) = property_names.get_mut("enum") {
                prune_schema_name_array(names, removed_properties);
                if names.is_empty() {
                    return true;
                }
            }
            property_names
                .get("const")
                .and_then(Json::as_str)
                .is_some_and(|name| !schema_property_name_is_safe(name, removed_properties))
                || property_names
                    .get("pattern")
                    .and_then(Json::as_str)
                    .is_some_and(schema_identifier_is_sensitive)
        });
    if remove_property_names {
        object.remove("propertyNames");
    }
}

fn prune_schema_name_array(names: &mut Vec<Json>, removed_properties: &BTreeSet<String>) {
    names.retain(|name| {
        name.as_str()
            .is_none_or(|name| schema_property_name_is_safe(name, removed_properties))
    });
}

fn schema_property_name_is_safe(name: &str, removed_properties: &BTreeSet<String>) -> bool {
    !removed_properties.contains(name) && !schema_identifier_is_sensitive(name)
}

fn json_pointer_child(path: &str, segment: &str) -> String {
    let segment = segment.replace('~', "~0").replace('/', "~1");
    format!("{path}/{segment}")
}

fn is_secret_schema(value: &Json) -> bool {
    let Some(object) = value.as_object() else {
        return false;
    };
    object.get("writeOnly").and_then(Json::as_bool) == Some(true)
        || object.get("x-secret").and_then(Json::as_bool) == Some(true)
        || object.get("x-sensitive").and_then(Json::as_bool) == Some(true)
        || object
            .get("format")
            .and_then(Json::as_str)
            .is_some_and(|format| {
                matches!(
                    format.to_ascii_lowercase().as_str(),
                    "password" | "secret" | "credential"
                )
            })
}

fn collect_secret_names(
    tools: &[ToolDefinition],
) -> Result<BTreeMap<String, BTreeSet<String>>, IneligibilityReason> {
    let mut names_by_tool = BTreeMap::new();
    for tool in tools {
        let mut names = BTreeSet::new();
        if let Some(schema) = tool.function.parameters.as_ref() {
            validate_schema_sanitizer_input(schema)?;
            collect_declared_secret_names(schema, &mut names);
        }
        if !names.is_empty() {
            names_by_tool
                .entry(tool.function.name.clone())
                .or_insert_with(BTreeSet::new)
                .extend(names);
        }
    }
    Ok(names_by_tool)
}

fn collect_declared_secret_names(schema: &Json, names: &mut BTreeSet<String>) {
    let mut stack = vec![schema];
    while let Some(value) = stack.pop() {
        match value {
            Json::Array(values) => stack.extend(values),
            Json::Object(object) => {
                if let Some(Json::Object(properties)) = object.get("properties") {
                    for (name, schema) in properties {
                        if is_secret_schema(schema) || is_sensitive_projection_key(name) {
                            names.insert(name.clone());
                        } else {
                            stack.push(schema);
                        }
                    }
                }
                for (key, value) in object {
                    if key != "properties" {
                        stack.push(value);
                    }
                }
            }
            _ => {}
        }
    }
}

fn sanitize_message(
    message: &Message,
    secret_names: &BTreeMap<String, BTreeSet<String>>,
) -> Result<SanitizedMessage, IneligibilityReason> {
    Ok(match message {
        Message::System { content, name } => SanitizedMessage::System {
            content: sanitize_content(content, false)?,
            name: name.clone(),
        },
        Message::Developer { content, name } => SanitizedMessage::Developer {
            content: sanitize_content(content, false)?,
            name: name.clone(),
        },
        Message::User { content, name } => SanitizedMessage::User {
            content: sanitize_content(content, false)?,
            name: name.clone(),
        },
        Message::Assistant {
            content,
            tool_calls,
            name,
        } => SanitizedMessage::Assistant {
            content: content
                .as_ref()
                .map(|content| sanitize_content(content, false))
                .transpose()?,
            tool_calls: tool_calls
                .as_deref()
                .map(|calls| sanitize_tool_calls(calls, secret_names))
                .transpose()?,
            name: name.clone(),
        },
        Message::Tool {
            content,
            tool_call_id,
        } => SanitizedMessage::Tool {
            content: sanitize_content(content, true)?,
            tool_call_id: tool_call_id.clone(),
        },
    })
}

fn sanitize_tool_calls(
    calls: &[ToolCall],
    secret_names: &BTreeMap<String, BTreeSet<String>>,
) -> Result<Vec<SanitizedToolCall>, IneligibilityReason> {
    calls
        .iter()
        .map(|call| {
            let names = secret_names
                .get(&call.function.name)
                .cloned()
                .unwrap_or_default();
            let arguments = sanitize_argument_string(&call.function.arguments, &names)?;
            Ok(SanitizedToolCall {
                id: call.id.clone(),
                call_type: call.call_type.clone(),
                name: call.function.name.clone(),
                arguments,
            })
        })
        .collect()
}

fn sanitize_argument_string(
    arguments: &str,
    secret_names: &BTreeSet<String>,
) -> Result<String, IneligibilityReason> {
    if arguments.len() > REQUEST_PROJECTION_MAX_BYTES {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    validate_json_text_resource_bounds(arguments)
        .map_err(|_| IneligibilityReason::ProjectionFailed)?;
    match serde_json::from_str::<Json>(arguments) {
        Ok(mut value) => {
            prune_credentials(&mut value, secret_names);
            let bytes =
                canonical_json_bytes(&value).map_err(|_| IneligibilityReason::ProjectionFailed)?;
            String::from_utf8(bytes).map_err(|_| IneligibilityReason::ProjectionFailed)
        }
        Err(_) => Err(IneligibilityReason::ProjectionFailed),
    }
}

pub(crate) fn sanitize_content(
    content: &MessageContent,
    sanitize_json_text: bool,
) -> Result<SanitizedMessageContent, IneligibilityReason> {
    Ok(match content {
        MessageContent::Text(text) if sanitize_json_text => {
            SanitizedMessageContent::Text(sanitize_possible_json_text(text)?)
        }
        MessageContent::Text(text) => {
            if text.len() > REQUEST_PROJECTION_MAX_BYTES {
                return Err(IneligibilityReason::ProjectionFailed);
            }
            SanitizedMessageContent::Text(text.clone())
        }
        MessageContent::Parts(parts) => SanitizedMessageContent::Parts(
            parts
                .iter()
                .map(sanitize_content_part)
                .collect::<Result<Vec<_>, _>>()?,
        ),
    })
}

fn sanitize_possible_json_text(text: &str) -> Result<String, IneligibilityReason> {
    if text.len() > REQUEST_PROJECTION_MAX_BYTES {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    match validate_json_text_resource_bounds(text) {
        Ok(()) => {}
        Err(JsonTextValidationError::InvalidSyntax) => return Ok(text.to_string()),
        Err(JsonTextValidationError::ResourceLimit) => {
            return Err(IneligibilityReason::ProjectionFailed);
        }
    }
    let Ok(mut value) = serde_json::from_str::<Json>(text) else {
        return Ok(text.to_string());
    };
    prune_credentials(&mut value, &BTreeSet::new());
    let bytes = canonical_json_bytes(&value).map_err(|_| IneligibilityReason::ProjectionFailed)?;
    String::from_utf8(bytes).map_err(|_| IneligibilityReason::ProjectionFailed)
}

struct JsonTextBudget {
    remaining_values: Cell<usize>,
}

impl JsonTextBudget {
    fn consume<E: serde::de::Error>(&self) -> Result<(), E> {
        let remaining = self
            .remaining_values
            .get()
            .checked_sub(1)
            .ok_or_else(|| E::custom("JSON value limit exceeded"))?;
        self.remaining_values.set(remaining);
        Ok(())
    }
}

#[derive(Clone, Copy)]
struct JsonTextSeed<'a> {
    budget: &'a JsonTextBudget,
    depth: usize,
}

impl<'de> DeserializeSeed<'de> for JsonTextSeed<'_> {
    type Value = ();

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        if self.depth > REQUEST_JSON_MAX_DEPTH {
            return Err(serde::de::Error::custom("JSON depth limit exceeded"));
        }
        self.budget.consume::<D::Error>()?;
        deserializer.deserialize_any(JsonTextVisitor { seed: self })
    }
}

struct JsonTextVisitor<'a> {
    seed: JsonTextSeed<'a>,
}

impl<'de> Visitor<'de> for JsonTextVisitor<'_> {
    type Value = ();

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("bounded JSON")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<(), E> {
        Ok(())
    }

    fn visit_i64<E>(self, _value: i64) -> Result<(), E> {
        Ok(())
    }

    fn visit_u64<E>(self, _value: u64) -> Result<(), E> {
        Ok(())
    }

    fn visit_f64<E>(self, _value: f64) -> Result<(), E> {
        Ok(())
    }

    fn visit_str<E>(self, _value: &str) -> Result<(), E> {
        Ok(())
    }

    fn visit_none<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_unit<E>(self) -> Result<(), E> {
        Ok(())
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<(), A::Error>
    where
        A: SeqAccess<'de>,
    {
        let child = JsonTextSeed {
            budget: self.seed.budget,
            depth: self.seed.depth + 1,
        };
        while sequence.next_element_seed(child)?.is_some() {}
        Ok(())
    }

    fn visit_map<A>(self, mut map: A) -> Result<(), A::Error>
    where
        A: MapAccess<'de>,
    {
        let child = JsonTextSeed {
            budget: self.seed.budget,
            depth: self.seed.depth + 1,
        };
        while map.next_key::<IgnoredAny>()?.is_some() {
            map.next_value_seed(child)?;
        }
        Ok(())
    }
}

enum JsonTextValidationError {
    InvalidSyntax,
    ResourceLimit,
}

fn validate_json_text_resource_bounds(text: &str) -> Result<(), JsonTextValidationError> {
    let budget = JsonTextBudget {
        remaining_values: Cell::new(REQUEST_JSON_MAX_VALUES),
    };
    let mut deserializer = serde_json::Deserializer::from_str(text);
    JsonTextSeed {
        budget: &budget,
        depth: 0,
    }
    .deserialize(&mut deserializer)
    .and_then(|()| deserializer.end())
    .map_err(|error| {
        if matches!(error.classify(), serde_json::error::Category::Data)
            || error.to_string().contains("recursion limit exceeded")
        {
            JsonTextValidationError::ResourceLimit
        } else {
            JsonTextValidationError::InvalidSyntax
        }
    })
}

fn sanitize_content_part(part: &ContentPart) -> Result<SanitizedContentPart, IneligibilityReason> {
    match part {
        ContentPart::Text { text } => {
            if text.len() > REQUEST_PROJECTION_MAX_BYTES {
                return Err(IneligibilityReason::ProjectionFailed);
            }
            Ok(SanitizedContentPart::Text { text: text.clone() })
        }
        ContentPart::ImageUrl { image_url } => {
            if image_url.url.len() > REQUEST_PROJECTION_MAX_BYTES {
                return Err(IneligibilityReason::ProjectionFailed);
            }
            let mut url =
                Url::parse(&image_url.url).map_err(|_| IneligibilityReason::ProjectionFailed)?;
            if !matches!(url.scheme(), "http" | "https") {
                return Err(IneligibilityReason::ProjectionFailed);
            }
            url.set_username("")
                .map_err(|_| IneligibilityReason::ProjectionFailed)?;
            url.set_password(None)
                .map_err(|_| IneligibilityReason::ProjectionFailed)?;
            url.set_query(None);
            url.set_fragment(None);
            let detail = match image_url.detail.as_deref() {
                None => None,
                Some(detail @ ("auto" | "low" | "high")) => Some(detail.to_string()),
                Some(_) => return Err(IneligibilityReason::ProjectionFailed),
            };
            Ok(SanitizedContentPart::ImageUrl {
                url: url.to_string(),
                detail,
            })
        }
    }
}

pub(crate) fn sanitize_json(value: &Json) -> Result<Json, IneligibilityReason> {
    validate_json_resource_bounds(
        value,
        REQUEST_PROJECTION_MAX_BYTES,
        REQUEST_JSON_MAX_VALUES,
        REQUEST_JSON_MAX_DEPTH,
    )
    .map_err(|_| IneligibilityReason::ProjectionFailed)?;
    validate_canonical_json_domain(value).map_err(|_| IneligibilityReason::ProjectionFailed)?;
    let mut value = value.clone();
    prune_credentials(&mut value, &BTreeSet::new());
    canonical_json_bytes(&value).map_err(|_| IneligibilityReason::ProjectionFailed)?;
    Ok(value)
}

fn prune_credentials(value: &mut Json, secret_names: &BTreeSet<String>) {
    match value {
        Json::Array(values) => {
            for value in values {
                prune_credentials(value, secret_names);
            }
        }
        Json::Object(object) => {
            object
                .retain(|key, _| !secret_names.contains(key) && !is_sensitive_projection_key(key));
            for value in object.values_mut() {
                prune_credentials(value, secret_names);
            }
        }
        _ => {}
    }
}

pub(crate) fn is_sensitive_projection_key(key: &str) -> bool {
    let normalized = key
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect::<String>();
    if normalized == "maxtoken" {
        return false;
    }
    if matches!(
        normalized.as_str(),
        "header"
            | "headers"
            | "httpheader"
            | "httpheaders"
            | "requestheader"
            | "requestheaders"
            | "responseheader"
            | "responseheaders"
            | "transportheader"
            | "transportheaders"
            | "cookiecontainer"
            | "cookiejar"
    ) {
        return true;
    }
    if matches!(
        normalized.as_str(),
        "authorization"
            | "auth"
            | "cookie"
            | "cookies"
            | "setcookie"
            | "apikey"
            | "accesskey"
            | "secretkey"
            | "privatekey"
            | "accesstoken"
            | "refreshtoken"
            | "sessiontoken"
            | "authtoken"
            | "bearertoken"
            | "idtoken"
            | "securitytoken"
            | "password"
            | "passwd"
            | "passphrase"
            | "clientsecret"
            | "clientpassword"
            | "credential"
            | "credentials"
            | "secret"
            | "token"
    ) {
        return true;
    }

    // General secret/password/credential/token suffixes are credential-shaped.
    // `max_token` is the one supported semantic singular-token key; plural and
    // count forms do not end in `token`. Authorization, auth, and cookie also
    // stay qualified so ordinary author/authentication/cookie-policy fields are
    // retained.
    const GENERAL_SUFFIXES: &[&str] = &[
        "secret",
        "token",
        "password",
        "passwd",
        "passphrase",
        "credential",
        "credentials",
    ];
    const QUALIFIED_SUFFIXES: &[&str] = &[
        "apikey",
        "apitoken",
        "accesskey",
        "secretkey",
        "privatekey",
        "accesstoken",
        "refreshtoken",
        "sessiontoken",
        "authtoken",
        "bearertoken",
        "clienttoken",
        "consumertoken",
        "idtoken",
        "oauthtoken",
        "securitytoken",
        "signingtoken",
        "clientsecret",
        "clientpassword",
        "authorization",
        "auth",
        "cookie",
    ];
    const QUALIFIED_PREFIXES: &[&str] = &[
        "authorizationheader",
        "authorizationvalue",
        "authorizationcode",
        "authorizationcredential",
        "authorizationcredentials",
        "authheader",
        "authvalue",
        "authcode",
        "authcredential",
        "authcredentials",
        "cookieheader",
        "cookievalue",
        "cookiejar",
        "setcookieheader",
        "setcookievalue",
    ];
    const CREDENTIAL_VALUE_ROOTS: &[&str] = &[
        "apikey",
        "accesskey",
        "secretkey",
        "privatekey",
        "accesstoken",
        "refreshtoken",
        "sessiontoken",
        "authtoken",
        "bearertoken",
        "clienttoken",
        "consumertoken",
        "idtoken",
        "oauthtoken",
        "securitytoken",
        "signingtoken",
        "clientsecret",
        "clientpassword",
        "password",
        "passwd",
        "passphrase",
        "credential",
        "credentials",
        "secret",
        "token",
    ];
    const CREDENTIAL_VALUE_QUALIFIERS: &[&str] = &[
        "value", "data", "material", "pem", "bytes", "string", "text",
    ];
    GENERAL_SUFFIXES
        .iter()
        .chain(QUALIFIED_SUFFIXES)
        .any(|suffix| normalized.len() > suffix.len() && normalized.ends_with(suffix))
        || QUALIFIED_PREFIXES
            .iter()
            .any(|prefix| normalized.starts_with(prefix))
        || CREDENTIAL_VALUE_ROOTS.iter().any(|root| {
            normalized
                .strip_prefix(root)
                .is_some_and(|qualifier| CREDENTIAL_VALUE_QUALIFIERS.contains(&qualifier))
        })
}

fn reject_capability_bearing_extras(
    envelope: &RouterRequestEnvelope,
) -> Result<(), IneligibilityReason> {
    if envelope.family == LlmApiFamily::OpenAIResponses
        && envelope
            .normalized_request
            .include
            .as_ref()
            .is_some_and(|include| !include.is_null())
    {
        return Err(IneligibilityReason::UnsupportedProviderRepresentation);
    }

    let unsupported_keys: &[&str] = match envelope.family {
        LlmApiFamily::OpenAIChatCompletions => &[
            "audio",
            "function_call",
            "functions",
            "modalities",
            "prediction",
            "reasoning_effort",
            "verbosity",
            "web_search_options",
        ],
        LlmApiFamily::OpenAIResponses => &["background", "conversation", "prompt"],
        LlmApiFamily::AnthropicMessages => &[
            "container",
            "context_management",
            "mcp_servers",
            "reasoning_effort",
            "stream",
            "thinking",
        ],
    };
    if unsupported_keys
        .iter()
        .any(|key| envelope.normalized_request.extra.contains_key(*key))
    {
        return Err(IneligibilityReason::UnsupportedProviderRepresentation);
    }

    let unsupported_wrapper = match envelope.family {
        LlmApiFamily::OpenAIResponses => {
            has_generic_or_native_wrapper_field(envelope, "text", "verbosity")
        }
        LlmApiFamily::AnthropicMessages => {
            has_generic_or_native_wrapper_field(envelope, "output_config", "effort")
        }
        LlmApiFamily::OpenAIChatCompletions => false,
    };
    if unsupported_wrapper {
        return Err(IneligibilityReason::UnsupportedProviderRepresentation);
    }
    Ok(())
}

fn has_generic_or_native_wrapper_field(
    envelope: &RouterRequestEnvelope,
    generic_container: &str,
    field: &str,
) -> bool {
    envelope
        .normalized_request
        .extra
        .get(generic_container)
        .and_then(Json::as_object)
        .is_some_and(|container| container.contains_key(field))
        || envelope
            .response_format
            .as_ref()
            .and_then(|format| format.extra.get("native_wrapper"))
            .and_then(Json::as_object)
            .is_some_and(|wrapper| wrapper.contains_key(field))
}

fn required_capabilities(request: &SanitizedAnnotatedLlmRequest) -> Vec<String> {
    let mut capabilities = BTreeSet::new();
    if request.tools.as_ref().is_some_and(|tools| !tools.is_empty())
        || request.messages.iter().any(|message| {
            matches!(message, SanitizedMessage::Tool { .. })
                || matches!(message, SanitizedMessage::Assistant { tool_calls: Some(calls), .. } if !calls.is_empty())
        })
    {
        capabilities.insert("tools".to_string());
    }
    if request.messages.iter().any(message_has_image) {
        capabilities.insert("multimodal_input".to_string());
    }
    if request.response_format.is_some() {
        capabilities.insert("structured_output".to_string());
    }
    if request.reasoning.is_some() {
        capabilities.insert("reasoning_controls".to_string());
    }
    capabilities.into_iter().collect()
}

fn message_has_image(message: &SanitizedMessage) -> bool {
    let content = match message {
        SanitizedMessage::System { content, .. }
        | SanitizedMessage::Developer { content, .. }
        | SanitizedMessage::User { content, .. }
        | SanitizedMessage::Tool { content, .. } => Some(content),
        SanitizedMessage::Assistant { content, .. } => content.as_ref(),
    };
    matches!(content, Some(SanitizedMessageContent::Parts(parts)) if parts.iter().any(|part| matches!(part, SanitizedContentPart::ImageUrl { .. })))
}

/// Return the explicit output-token bound represented by a sanitized request.
pub(crate) fn output_token_limit(request: &SanitizedAnnotatedLlmRequest) -> Option<u64> {
    request
        .max_output_tokens
        .or_else(|| request.params.as_ref().and_then(|params| params.max_tokens))
}

/// Return the conservative canonical input byte budget used by preflight.
pub(crate) fn canonical_request_budget(
    request: &SanitizedAnnotatedLlmRequest,
) -> Result<u64, IneligibilityReason> {
    let bytes =
        canonical_serialize_bytes(request).map_err(|_| IneligibilityReason::ProjectionFailed)?;
    u64::try_from(bytes.len()).map_err(|_| IneligibilityReason::ProjectionFailed)
}

pub(crate) fn projection_semantic_fingerprint(
    projection: &RouterRequestProjectionV1,
) -> Result<String, IneligibilityReason> {
    if projection.normalized_request.messages.len() > REQUEST_PROJECTION_MAX_MESSAGES {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    fingerprint_serializable_bounded(
        &RouterRequestProjectionSemantics::from(projection),
        REQUEST_PROJECTION_MAX_BYTES,
    )
    .map_err(|error| match error {
        BoundedFingerprintError::BoundExceeded | BoundedFingerprintError::Serialization => {
            IneligibilityReason::ProjectionFailed
        }
    })
}

pub(crate) fn validate_request_projection(
    projection: &RouterRequestProjectionV1,
) -> Result<(), IneligibilityReason> {
    if projection.schema != REQUEST_PROJECTION_SCHEMA_V1
        || projection.sanitizer_version != ROUTER_SANITIZER_VERSION
        || projection.semantic_request_fingerprint != projection_semantic_fingerprint(projection)?
    {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    Ok(())
}

/// Revalidate a deserialized request-file projection against the sanitizer's output contract.
pub(crate) fn validate_external_request_projection(
    projection: &RouterRequestProjectionV1,
) -> Result<(), IneligibilityReason> {
    validate_request_projection(projection)?;
    if projection.required_capabilities != required_capabilities(&projection.normalized_request)
        || projection.response_format != projection.normalized_request.response_format
        || !external_instruction_facts_match(projection)
        || !external_request_is_sanitized(&projection.normalized_request)?
    {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    let has_response_schema = projection
        .response_format
        .as_ref()
        .and_then(|format| format.schema.as_ref())
        .is_some();
    if has_response_schema != projection.response_schema_fingerprint.is_some() {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    let value =
        serde_json::to_value(projection).map_err(|_| IneligibilityReason::ProjectionFailed)?;
    if contains_external_control_material(&value) {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    Ok(())
}

fn external_instruction_facts_match(projection: &RouterRequestProjectionV1) -> bool {
    let expected = projection
        .normalized_request
        .messages
        .iter()
        .enumerate()
        .filter_map(|(wire_ordinal, message)| {
            let (role, content, name) = match message {
                SanitizedMessage::System { content, name } => {
                    (SanitizedRouterInstructionRole::System, content, name)
                }
                SanitizedMessage::Developer { content, name } => {
                    (SanitizedRouterInstructionRole::Developer, content, name)
                }
                SanitizedMessage::User { .. }
                | SanitizedMessage::Assistant { .. }
                | SanitizedMessage::Tool { .. } => return None,
            };
            let SanitizedMessageContent::Text(content) = content else {
                return Some(None);
            };
            Some(Some((
                wire_ordinal,
                role,
                content.as_str(),
                name.as_deref(),
            )))
        })
        .collect::<Option<Vec<_>>>();
    let Some(expected) = expected else {
        return false;
    };
    expected.len() == projection.ordered_instructions.len()
        && expected.iter().zip(&projection.ordered_instructions).all(
            |((wire_ordinal, role, content, name), actual)| {
                actual.wire_ordinal == *wire_ordinal
                    && actual.role == *role
                    && actual.content == *content
                    && actual.name.as_deref() == *name
            },
        )
}

fn external_request_is_sanitized(
    request: &SanitizedAnnotatedLlmRequest,
) -> Result<bool, IneligibilityReason> {
    for tool in request.tools.as_deref().unwrap_or_default() {
        if let Some(parameters) = &tool.parameters
            && sanitize_tool_schema(parameters)? != *parameters
        {
            return Ok(false);
        }
    }
    if let Some(format) = &request.response_format
        && let Some(schema) = &format.schema
        && sanitize_tool_schema(schema)? != *schema
    {
        return Ok(false);
    }
    for value in [request.truncation.as_ref(), request.reasoning.as_ref()]
        .into_iter()
        .flatten()
    {
        if sanitize_json(value)? != *value {
            return Ok(false);
        }
    }
    for message in &request.messages {
        let (content, tool_calls, sanitize_json_text) = match message {
            SanitizedMessage::System { content, .. }
            | SanitizedMessage::Developer { content, .. }
            | SanitizedMessage::User { content, .. } => (Some(content), None, false),
            SanitizedMessage::Assistant {
                content,
                tool_calls,
                ..
            } => (content.as_ref(), tool_calls.as_deref(), false),
            SanitizedMessage::Tool { content, .. } => (Some(content), None, true),
        };
        if let Some(content) = content
            && !external_content_is_sanitized(content, sanitize_json_text)?
        {
            return Ok(false);
        }
        for call in tool_calls.unwrap_or_default() {
            if sanitize_argument_string(&call.arguments, &BTreeSet::new())? != call.arguments {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn external_content_is_sanitized(
    content: &SanitizedMessageContent,
    sanitize_json_text: bool,
) -> Result<bool, IneligibilityReason> {
    match content {
        SanitizedMessageContent::Text(text) => {
            Ok(!sanitize_json_text || sanitize_possible_json_text(text)? == *text)
        }
        SanitizedMessageContent::Parts(parts) => {
            for part in parts {
                if let SanitizedContentPart::ImageUrl { url, detail } = part {
                    let parsed =
                        Url::parse(url).map_err(|_| IneligibilityReason::ProjectionFailed)?;
                    if !matches!(parsed.scheme(), "http" | "https")
                        || !parsed.username().is_empty()
                        || parsed.password().is_some()
                        || parsed.query().is_some()
                        || parsed.fragment().is_some()
                        || parsed.to_string() != *url
                        || detail
                            .as_deref()
                            .is_some_and(|detail| !matches!(detail, "auto" | "low" | "high"))
                    {
                        return Ok(false);
                    }
                }
            }
            Ok(true)
        }
    }
}

fn contains_external_control_material(value: &Json) -> bool {
    match value {
        Json::Object(object) => object.iter().any(|(key, value)| {
            let normalized = key
                .chars()
                .filter(|character| character.is_ascii_alphanumeric())
                .flat_map(char::to_lowercase)
                .collect::<String>();
            is_sensitive_projection_key(key)
                || matches!(
                    normalized.as_str(),
                    "header"
                        | "headers"
                        | "httpheaders"
                        | "requestheaders"
                        | "responseheaders"
                        | "transportheaders"
                        | "metadata"
                        | "sanitizedmetadata"
                        | "tenantid"
                        | "agentid"
                        | "userid"
                        | "rootuuid"
                        | "calluuid"
                        | "parentuuid"
                        | "trajectoryowneruuid"
                )
                || contains_external_control_material(value)
        }),
        Json::Array(values) => values.iter().any(contains_external_control_material),
        Json::String(value) => crate::preflight::contains_sensitive_control_material(value),
        Json::Null | Json::Bool(_) | Json::Number(_) => false,
    }
}

#[cfg(test)]
pub(crate) fn projection_semantic_bytes(
    projection: &RouterRequestProjectionV1,
) -> Result<Vec<u8>, IneligibilityReason> {
    canonical_serialize_bytes(&RouterRequestProjectionSemantics::from(projection))
        .map_err(|_| IneligibilityReason::ProjectionFailed)
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet};

    use nemo_relay::api::llm::{
        LlmApiFamily, LlmAttributes, LlmCallRole, LlmExecutionContextSnapshot, LlmRequest,
        LlmTrajectoryScopeSnapshot,
    };
    use nemo_relay::api::scope::ScopeType;
    use serde_json::{Map, Value as Json, json};
    use uuid::Uuid;

    use super::{
        REQUEST_JSON_MAX_DEPTH, REQUEST_JSON_MAX_VALUES, REQUEST_PROJECTION_MAX_BYTES,
        REQUEST_PROJECTION_MAX_MESSAGES, RouterRoutingContextProjectionV1, SanitizedMessage,
        candidate_request_projection, project_request, project_routing_context,
        projection_semantic_bytes, sanitize_argument_string, sanitize_json,
        sanitize_possible_json_text, sanitize_tool_schema, validate_external_request_projection,
        validate_raw_request_resource_bounds, validate_request_projection,
    };
    use crate::adapter::FamilyAdapter;
    use crate::config::{CanonicalizerConfig, PoolSelectorConfig};
    use crate::eligibility::IneligibilityReason;
    use crate::fingerprint::sha256_hex;

    fn context(tenant: Option<&str>, agent: Option<&str>) -> LlmExecutionContextSnapshot {
        LlmExecutionContextSnapshot {
            call_uuid: Uuid::new_v4(),
            root_uuid: Uuid::new_v4(),
            parent_uuid: Uuid::new_v4(),
            trajectory_owner_uuid: Uuid::new_v4(),
            trajectory_owner_path: vec![LlmTrajectoryScopeSnapshot {
                uuid: Uuid::new_v4(),
                name: "must-not-enter-policy".to_string(),
                scope_type: ScopeType::Agent,
            }],
            api_family: LlmApiFamily::OpenAIChatCompletions,
            call_role: LlmCallRole::Primary,
            attributes: LlmAttributes::empty(),
            tenant_id: tenant.map(str::to_string),
            agent_id: agent.map(str::to_string),
            sanitized_metadata: BTreeMap::new(),
        }
    }

    fn tool_request() -> LlmRequest {
        LlmRequest {
            headers: Map::from_iter([
                ("authorization".to_string(), json!("Bearer header-secret")),
                ("cookie".to_string(), json!("session=cookie-secret")),
            ]),
            content: json!({
                "model": "anchor",
                "messages": [
                    {"role": "system", "content": "policy"},
                    {"role": "user", "content": "call the tool"},
                    {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"api_key\":\"argument-secret\",\"query\":\"safe\"}"
                            }
                        }]
                    }
                ],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "lookup data",
                        "strict": true,
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "api_key": {"type": "string", "x-secret": true},
                                "query": {"type": "string"}
                            },
                            "required": ["api_key", "query"]
                        }
                    }
                }],
                "vendor_secret": "provider-extra-secret",
                "metadata": {"authorization": "metadata-secret"},
                "user": "raw-user-id"
            }),
        }
    }

    fn structured_response_request(secret: &str) -> LlmRequest {
        LlmRequest {
            headers: Map::new(),
            content: json!({
                "model": "anchor",
                "messages": [{"role": "user", "content": "return an answer"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": true,
                        "schema": {
                            "type": "object",
                            "$defs": {
                                "opaque_secret": {
                                    "type": "string",
                                    "writeOnly": true,
                                    "description": format!("opaque description {secret}"),
                                    "default": format!("opaque default {secret}"),
                                    "examples": [format!("opaque example {secret}")]
                                },
                                "safe_definition": {"type": "string"}
                            },
                            "allOf": [
                                {
                                    "$ref": "#/$defs/opaque_secret",
                                    "description": format!("all-of description {secret}")
                                },
                                {"type": "object"}
                            ],
                            "anyOf": [
                                {
                                    "$ref": "#/$defs/opaque_secret",
                                    "default": format!("any-of default {secret}")
                                },
                                {"type": "object"}
                            ],
                            "definitions": {
                                "legacy_secret": {
                                    "type": "string",
                                    "x-secret": true,
                                    "description": format!("legacy description {secret}")
                                },
                                "legacy_safe": {"type": "number"}
                            },
                            "patternProperties": {
                                "^credential$": {
                                    "type": "string",
                                    "description": format!("pattern description {secret}")
                                },
                                "^safe_[a-z]+$": {"type": "string"}
                            },
                            "dependentSchemas": {
                                "api_key": {
                                    "type": "object",
                                    "description": format!("dependent description {secret}")
                                },
                                "safe_trigger": {"required": ["answer"]}
                            },
                            "properties": {
                                "answer": {"type": "string"},
                                "opaque_alias": {"$ref": "#/$defs/opaque_secret"},
                                "write_only_field": {
                                    "type": "string",
                                    "writeOnly": true,
                                    "description": format!("write-only description {secret}"),
                                    "default": format!("write-only default {secret}"),
                                    "examples": [format!("write-only example {secret}")]
                                },
                                "declared_field": {
                                    "type": "string",
                                    "x-secret": true,
                                    "description": format!("declared description {secret}"),
                                    "default": format!("declared default {secret}"),
                                    "examples": [format!("declared example {secret}")]
                                },
                                "sensitive_field": {
                                    "type": "string",
                                    "x-sensitive": true,
                                    "description": format!("sensitive description {secret}"),
                                    "default": format!("sensitive default {secret}"),
                                    "examples": [format!("sensitive example {secret}")]
                                },
                                "Client-Secret": {
                                    "type": "string",
                                    "description": format!("key description {secret}"),
                                    "default": format!("key default {secret}"),
                                    "examples": [format!("key example {secret}")]
                                }
                            },
                            "required": [
                                "answer",
                                "opaque_alias",
                                "write_only_field",
                                "declared_field",
                                "sensitive_field",
                                "Client-Secret"
                            ]
                        }
                    }
                }
            }),
        }
    }

    #[test]
    fn projection_is_explicit_bounded_and_secret_free() {
        let envelope = FamilyAdapter
            .decode(LlmApiFamily::OpenAIChatCompletions, &tool_request())
            .unwrap();
        let projection = project_request(&envelope, &CanonicalizerConfig::default()).unwrap();
        let serialized = serde_json::to_string(&projection).unwrap();
        for forbidden in [
            "header-secret",
            "cookie-secret",
            "argument-secret",
            "provider-extra-secret",
            "metadata-secret",
            "raw-user-id",
            "api_key",
        ] {
            assert!(!serialized.contains(forbidden), "leaked {forbidden}");
        }
        assert!(serialized.contains("query"));
        assert_eq!(projection.required_capabilities, vec!["tools"]);
        assert_eq!(validate_external_request_projection(&projection), Ok(()));

        let semantics = projection_semantic_bytes(&projection).unwrap();
        assert_eq!(
            projection.semantic_request_fingerprint,
            sha256_hex(&semantics)
        );
    }

    #[test]
    fn response_schema_projection_prunes_secrets_but_fingerprints_the_full_schema() {
        let first_secret = "schema-secret-a";
        let second_secret = "schema-secret-b";
        let first_envelope = FamilyAdapter
            .decode(
                LlmApiFamily::OpenAIChatCompletions,
                &structured_response_request(first_secret),
            )
            .unwrap();
        let second_envelope = FamilyAdapter
            .decode(
                LlmApiFamily::OpenAIChatCompletions,
                &structured_response_request(second_secret),
            )
            .unwrap();

        let first = project_request(&first_envelope, &CanonicalizerConfig::default()).unwrap();
        let second = project_request(&second_envelope, &CanonicalizerConfig::default()).unwrap();
        assert_eq!(validate_external_request_projection(&first), Ok(()));
        assert_eq!(validate_external_request_projection(&second), Ok(()));

        for format in [
            first.normalized_request.response_format.as_ref().unwrap(),
            first.response_format.as_ref().unwrap(),
        ] {
            let schema = format.schema.as_ref().unwrap();
            assert_eq!(schema["required"], json!(["answer"]));
            let properties = schema["properties"].as_object().unwrap();
            assert_eq!(properties.len(), 1);
            assert!(properties.contains_key("answer"));
            assert_eq!(
                schema["$defs"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .collect::<Vec<_>>(),
                ["safe_definition"]
            );
            assert_eq!(schema["allOf"], json!([{"type": "object"}]));
            assert_eq!(schema["anyOf"], json!([{"type": "object"}]));
            assert_eq!(
                schema["definitions"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .collect::<Vec<_>>(),
                ["legacy_safe"]
            );
            assert_eq!(
                schema["patternProperties"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .collect::<Vec<_>>(),
                ["^safe_[a-z]+$"]
            );
            assert_eq!(
                schema["dependentSchemas"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .collect::<Vec<_>>(),
                ["safe_trigger"]
            );
            let serialized = serde_json::to_string(format).unwrap();
            for forbidden in [
                first_secret,
                "writeOnly",
                "x-secret",
                "x-sensitive",
                "description",
                "default",
                "examples",
                "write_only_field",
                "declared_field",
                "sensitive_field",
                "Client-Secret",
                "opaque_alias",
                "opaque_secret",
                "legacy_secret",
                "^credential$",
                "api_key",
            ] {
                assert!(!serialized.contains(forbidden), "leaked {forbidden}");
            }
        }

        assert_eq!(
            first.normalized_request.response_format,
            first.response_format
        );
        assert_eq!(
            first.normalized_request.response_format,
            second.normalized_request.response_format
        );
        assert_ne!(
            first.response_schema_fingerprint,
            second.response_schema_fingerprint
        );
        assert_eq!(
            first.response_schema_fingerprint,
            first_envelope.response_schema_fingerprint
        );
        assert_eq!(
            second.response_schema_fingerprint,
            second_envelope.response_schema_fingerprint
        );

        for original_format in [
            first_envelope
                .normalized_request
                .response_format
                .as_ref()
                .unwrap(),
            first_envelope.response_format.as_ref().unwrap(),
        ] {
            let original = serde_json::to_string(original_format.schema.as_ref().unwrap()).unwrap();
            assert!(original.contains(first_secret));
            assert!(original.contains("writeOnly"));
            assert!(original.contains("x-secret"));
            assert!(original.contains("x-sensitive"));
            assert!(original.contains("Client-Secret"));
            assert!(original.contains("opaque_alias"));
            assert!(original.contains("opaque_secret"));
            assert!(original.contains("legacy_secret"));
            assert!(original.contains("^credential$"));
            assert!(original.contains("api_key"));
        }
    }

    #[test]
    fn response_schema_projection_drops_root_ref_to_pruned_definition() {
        let schema = json!({
            "$defs": {
                "opaque_secret": {
                    "type": "string",
                    "writeOnly": true,
                    "description": "root-ref-secret"
                }
            },
            "$ref": "#/$defs/opaque_secret"
        });

        assert_eq!(sanitize_tool_schema(&schema).unwrap(), json!({}));
    }

    #[test]
    fn response_schema_projection_prunes_sensitive_local_reference_identities() {
        let credential_anchor = "sk-12345678901234567890";
        let schema = json!({
            "$defs": {
                "anchored_target": {
                    "$anchor": "privateAnchor",
                    "type": "string",
                    "writeOnly": true,
                    "properties": {
                        "innocent_child": {"type": "string"}
                    }
                },
                "dynamic_target": {
                    "$dynamicAnchor": "privateDynamic",
                    "type": "string",
                    "x-sensitive": true
                },
                "opaque_pointer": {
                    "type": "string",
                    "writeOnly": true
                },
                "credential_anchor_target": {
                    "$anchor": credential_anchor,
                    "type": "string"
                },
                "safe_target": {
                    "$anchor": "safeAnchor",
                    "type": "string"
                }
            },
            "type": "object",
            "properties": {
                "via_anchor": {"$ref": "#privateAnchor"},
                "via_sensitive_descendant": {
                    "$ref": "#/$defs/anchored_target/properties/innocent_child"
                },
                "via_dynamic": {"$dynamicRef": "#privateDynamic"},
                "via_percent_pointer": {"$ref": "#/%24defs/opaque_pointer"},
                "via_credential_anchor": {"$ref": format!("#{credential_anchor}")},
                "safe_alias": {"$ref": "#safeAnchor"}
            },
            "required": [
                "via_anchor",
                "via_sensitive_descendant",
                "via_dynamic",
                "via_percent_pointer",
                "via_credential_anchor",
                "safe_alias"
            ],
            "allOf": [
                {"$ref": "#privateAnchor"},
                {"$ref": "#/$defs/anchored_target/properties/innocent_child"},
                {"$dynamicRef": "#privateDynamic"},
                {"$ref": "#/%24defs/opaque_pointer"},
                {"$ref": "#safeAnchor"}
            ],
            "anyOf": [
                {
                    "allOf": [
                        {"$ref": "#privateAnchor"},
                        {"type": "object"}
                    ]
                },
                {"$ref": "#safeAnchor"}
            ]
        });

        let sanitized = sanitize_tool_schema(&schema).unwrap();
        assert_eq!(
            sanitized["$defs"]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            ["safe_target"]
        );
        assert_eq!(
            sanitized["properties"]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            ["safe_alias"]
        );
        assert_eq!(sanitized["required"], json!(["safe_alias"]));
        assert_eq!(sanitized["allOf"], json!([{"$ref": "#safeAnchor"}]));
        assert_eq!(
            sanitized["anyOf"],
            json!([
                {"allOf": [{"type": "object"}]},
                {"$ref": "#safeAnchor"}
            ])
        );
        let serialized = serde_json::to_string(&sanitized).unwrap();
        for forbidden in [
            credential_anchor,
            "privateAnchor",
            "privateDynamic",
            "opaque_pointer",
            "via_anchor",
            "via_sensitive_descendant",
            "via_dynamic",
            "via_percent_pointer",
            "via_credential_anchor",
        ] {
            assert!(!serialized.contains(forbidden), "leaked {forbidden}");
        }
        assert!(serialized.contains("safeAnchor"));
        assert!(serialized.contains("safe_alias"));
    }

    #[test]
    fn response_schema_projection_prunes_sensitive_root_refs_and_keeps_safe_local_refs() {
        let sensitive_definitions = json!({
            "anchored_target": {
                "$anchor": "privateAnchor",
                "$dynamicAnchor": "privateDynamic",
                "type": "string",
                "writeOnly": true
            },
            "opaque_pointer": {
                "type": "string",
                "x-secret": true
            }
        });
        for (keyword, reference) in [
            ("$ref", "#privateAnchor"),
            ("$dynamicRef", "#privateDynamic"),
            ("$ref", "#/%24defs/opaque_pointer"),
            ("$recursiveRef", "#privateAnchor"),
        ] {
            let mut schema = serde_json::Map::new();
            schema.insert("$defs".to_string(), sensitive_definitions.clone());
            schema.insert(keyword.to_string(), json!(reference));
            assert_eq!(
                sanitize_tool_schema(&Json::Object(schema)).unwrap(),
                json!({}),
                "root {keyword} {reference} should be pruned"
            );
        }

        let safe = json!({
            "$defs": {"safe": {"type": "string"}},
            "$ref": "#/%24defs/safe"
        });
        assert_eq!(sanitize_tool_schema(&safe).unwrap(), safe);

        let credential_anchor = json!({
            "$anchor": "sk-12345678901234567890",
            "type": "string"
        });
        assert_eq!(sanitize_tool_schema(&credential_anchor).unwrap(), json!({}));
    }

    #[test]
    fn response_schema_projection_propagates_long_alias_chains_with_bounded_work() {
        const ALIASES: usize = 2_048;

        let mut definitions = serde_json::Map::new();
        definitions.insert(
            "terminal".to_string(),
            json!({"type": "string", "writeOnly": true}),
        );
        for index in (0..ALIASES).rev() {
            let target = if index + 1 == ALIASES {
                "terminal".to_string()
            } else {
                format!("alias_{:04}", index + 1)
            };
            definitions.insert(
                format!("alias_{index:04}"),
                json!({"$ref": format!("#/$defs/{target}")}),
            );
        }
        definitions.insert("safe".to_string(), json!({"type": "string"}));
        let schema = json!({
            "$defs": definitions,
            "type": "object",
            "properties": {
                "unsafe_alias": {"$ref": "#/$defs/alias_0000"},
                "safe_alias": {"$ref": "#/$defs/safe"}
            }
        });

        let sanitized = sanitize_tool_schema(&schema).unwrap();
        assert_eq!(
            sanitized["$defs"]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            ["safe"]
        );
        assert_eq!(
            sanitized["properties"]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            ["safe_alias"]
        );
    }

    #[test]
    fn response_schema_projection_scopes_duplicate_anchors_by_resource() {
        let schema = json!({
            "$id": "https://schemas.example.test/root",
            "$defs": {
                "root_safe": {
                    "$anchor": "sharedAnchor",
                    "type": "string"
                },
                "embedded": {
                    "$id": "embedded",
                    "$defs": {
                        "embedded_secret": {
                            "$anchor": "sharedAnchor",
                            "type": "string",
                            "writeOnly": true
                        }
                    },
                    "type": "object",
                    "properties": {
                        "embedded_alias": {"$ref": "#sharedAnchor"}
                    }
                }
            },
            "type": "object",
            "properties": {
                "root_alias": {"$ref": "#sharedAnchor"},
                "embedded_alias_from_root": {"$ref": "embedded#sharedAnchor"}
            }
        });

        let sanitized = sanitize_tool_schema(&schema).unwrap();
        assert!(sanitized["properties"]["root_alias"].is_object());
        assert!(
            sanitized["properties"]
                .as_object()
                .unwrap()
                .get("embedded_alias_from_root")
                .is_none()
        );
        assert!(sanitized["$defs"]["root_safe"].is_object());
        assert!(
            sanitized["$defs"]["embedded"]["properties"]
                .as_object()
                .unwrap()
                .get("embedded_alias")
                .is_none()
        );
        assert!(
            sanitized["$defs"]["embedded"]["$defs"]
                .as_object()
                .unwrap()
                .get("embedded_secret")
                .is_none()
        );
    }

    #[test]
    fn response_schema_projection_prunes_sensitive_property_name_keywords() {
        let schema = json!({
            "type": "object",
            "properties": {
                "safe": {"type": "string"},
                "api_key": {"type": "string"}
            },
            "required": ["safe", "api_key", "client_secret"],
            "propertyNames": {
                "enum": ["safe", "api_key", "client_secret"]
            },
            "dependentRequired": {
                "safe": ["other", "api_key", "client_secret"],
                "api_key": ["safe"],
                "missing_trigger": ["client_secret"]
            },
            "dependencies": {
                "safe": ["other", "client_secret"],
                "client_secret": ["safe"],
                "api_key": {"required": ["safe"]}
            },
            "dependentSchemas": {
                "safe": {"required": ["other"]},
                "api_key": {"required": ["safe"]}
            }
        });

        let sanitized = sanitize_tool_schema(&schema).unwrap();
        assert_eq!(sanitized["required"], json!(["safe"]));
        assert_eq!(sanitized["propertyNames"], json!({"enum": ["safe"]}));
        assert_eq!(sanitized["dependentRequired"], json!({"safe": ["other"]}));
        assert_eq!(sanitized["dependencies"], json!({"safe": ["other"]}));
        assert_eq!(
            sanitized["dependentSchemas"],
            json!({"safe": {"required": ["other"]}})
        );
        let serialized = serde_json::to_string(&sanitized).unwrap();
        assert!(!serialized.contains("api_key"));
        assert!(!serialized.contains("client_secret"));
    }

    #[test]
    fn response_schema_projection_rejects_depth_before_recursive_sanitization() {
        let mut schema = json!({"type": "string"});
        for _ in 0..=super::MAX_SCHEMA_SANITIZER_DEPTH {
            schema = json!({"allOf": [schema]});
        }
        assert_eq!(
            sanitize_tool_schema(&schema),
            Err(IneligibilityReason::ProjectionFailed)
        );
    }

    #[test]
    fn request_projection_rejects_deep_tool_schema_before_secret_name_collection() {
        let mut schema = json!({"type": "string"});
        for _ in 0..=super::MAX_SCHEMA_SANITIZER_DEPTH {
            schema = json!({"allOf": [schema]});
        }
        let envelope = FamilyAdapter
            .decode(
                LlmApiFamily::OpenAIChatCompletions,
                &LlmRequest {
                    headers: Map::new(),
                    content: json!({
                        "model": "anchor",
                        "messages": [{"role": "user", "content": "task"}],
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "parameters": schema
                            }
                        }]
                    }),
                },
            )
            .unwrap();

        let result = std::panic::catch_unwind(|| {
            project_request(&envelope, &CanonicalizerConfig::default())
        });
        assert!(matches!(
            result,
            Ok(Err(IneligibilityReason::ProjectionFailed))
        ));
    }

    #[test]
    fn response_schema_projection_prunes_annotated_nested_reference_alias_as_a_unit() {
        let schema = json!({
            "$defs": {
                "secret_target": {
                    "type": "string",
                    "writeOnly": true
                },
                "annotated_alias": {
                    "description": "alias annotation must not survive",
                    "allOf": [
                        {"$ref": "#/$defs/secret_target"},
                        {"type": "string"}
                    ]
                }
            },
            "type": "object",
            "properties": {
                "alias_user": {"$ref": "#/$defs/annotated_alias"},
                "ordinary_secret_child": {
                    "type": "string",
                    "writeOnly": true
                },
                "safe": {"type": "string"}
            },
            "required": ["alias_user", "ordinary_secret_child", "safe"]
        });

        let sanitized = sanitize_tool_schema(&schema).unwrap();
        assert!(sanitized.is_object());
        assert_eq!(
            sanitized["properties"]
                .as_object()
                .unwrap()
                .keys()
                .collect::<Vec<_>>(),
            ["safe"]
        );
        assert_eq!(sanitized["required"], json!(["safe"]));
        assert!(
            sanitized["$defs"]
                .as_object()
                .unwrap()
                .get("annotated_alias")
                .is_none()
        );
        let serialized = serde_json::to_string(&sanitized).unwrap();
        assert!(!serialized.contains("alias annotation must not survive"));
        assert!(!serialized.contains("ordinary_secret_child"));
    }

    #[test]
    fn response_schema_projection_bounds_long_pointer_normalization() {
        let at_limit_segments = super::MAX_SCHEMA_REFERENCE_SEGMENTS - 1;
        let at_limit = format!("#/{}", vec!["x"; at_limit_segments].join("/"));
        let safe = json!({"$ref": at_limit});
        assert_eq!(sanitize_tool_schema(&safe).unwrap(), safe);

        let over_limit = format!(
            "#/{}",
            vec!["x"; super::MAX_SCHEMA_REFERENCE_SEGMENTS].join("/")
        );
        assert_eq!(
            sanitize_tool_schema(&json!({"$ref": over_limit})),
            Err(IneligibilityReason::ProjectionFailed)
        );
    }

    #[test]
    fn response_schema_projection_rejects_large_identifier_amplification() {
        let oversized_id = format!(
            "https://schemas.example.test/{}",
            "x".repeat(super::MAX_SCHEMA_IDENTIFIER_BYTES + 1)
        );
        assert_eq!(
            sanitize_tool_schema(&json!({"$id": oversized_id, "type": "string"})),
            Err(IneligibilityReason::ProjectionFailed)
        );
    }

    #[test]
    fn capability_bearing_generic_and_wrapper_fields_are_ineligible() {
        let fixtures = [
            (
                LlmApiFamily::OpenAIChatCompletions,
                json!({
                    "model": "anchor",
                    "messages": [{"role": "user", "content": "task"}],
                    "reasoning_effort": "high"
                }),
            ),
            (
                LlmApiFamily::OpenAIChatCompletions,
                json!({
                    "model": "anchor",
                    "messages": [{"role": "user", "content": "task"}],
                    "modalities": ["audio"],
                    "audio": {"format": "wav", "voice": "alloy"}
                }),
            ),
            (
                LlmApiFamily::OpenAIChatCompletions,
                json!({
                    "model": "anchor",
                    "messages": [{"role": "user", "content": "task"}],
                    "prediction": {"type": "content", "content": "expected"}
                }),
            ),
            (
                LlmApiFamily::OpenAIResponses,
                json!({
                    "model": "anchor",
                    "input": "task",
                    "background": true
                }),
            ),
            (
                LlmApiFamily::OpenAIResponses,
                json!({
                    "model": "anchor",
                    "input": "task",
                    "include": ["reasoning.encrypted_content"]
                }),
            ),
            (
                LlmApiFamily::OpenAIResponses,
                json!({
                    "model": "anchor",
                    "input": "task",
                    "text": {"verbosity": "high"}
                }),
            ),
            (
                LlmApiFamily::OpenAIResponses,
                json!({
                    "model": "anchor",
                    "input": "task",
                    "text": {
                        "verbosity": "high",
                        "format": {"type": "json_object"}
                    }
                }),
            ),
            (
                LlmApiFamily::AnthropicMessages,
                json!({
                    "model": "anchor",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "task"}],
                    "thinking": {"type": "enabled", "budget_tokens": 8}
                }),
            ),
            (
                LlmApiFamily::AnthropicMessages,
                json!({
                    "model": "anchor",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "task"}],
                    "reasoning_effort": "high"
                }),
            ),
            (
                LlmApiFamily::AnthropicMessages,
                json!({
                    "model": "anchor",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "task"}],
                    "stream": true
                }),
            ),
            (
                LlmApiFamily::AnthropicMessages,
                json!({
                    "model": "anchor",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "task"}],
                    "output_config": {"effort": "high"}
                }),
            ),
            (
                LlmApiFamily::AnthropicMessages,
                json!({
                    "model": "anchor",
                    "max_tokens": 16,
                    "messages": [{"role": "user", "content": "task"}],
                    "output_config": {
                        "effort": "high",
                        "format": {
                            "type": "json_schema",
                            "schema": {"type": "object"}
                        }
                    }
                }),
            ),
        ];
        for (family, content) in fixtures {
            let envelope = FamilyAdapter
                .decode(
                    family,
                    &LlmRequest {
                        headers: Map::new(),
                        content,
                    },
                )
                .unwrap();
            assert_eq!(
                project_request(&envelope, &CanonicalizerConfig::default()),
                Err(IneligibilityReason::UnsupportedProviderRepresentation)
            );
        }
    }

    #[test]
    fn duplicate_tool_names_union_declared_secret_fields() {
        let request = LlmRequest {
            headers: Map::new(),
            content: json!({
                "model": "anchor",
                "messages": [
                    {"role": "user", "content": "task"},
                    {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"first_secret\":\"one\",\"second_secret\":\"two\",\"safe\":true}"
                            }
                        }]
                    }
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"first_secret": {"type": "string", "x-secret": true}}
                            }
                        }
                    },
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {"second_secret": {"type": "string", "writeOnly": true}}
                            }
                        }
                    }
                ]
            }),
        };
        let envelope = FamilyAdapter
            .decode(LlmApiFamily::OpenAIChatCompletions, &request)
            .unwrap();
        let projection = project_request(&envelope, &CanonicalizerConfig::default()).unwrap();
        let serialized = serde_json::to_string(&projection).unwrap();
        assert!(!serialized.contains("first_secret"));
        assert!(!serialized.contains("second_secret"));
        assert!(!serialized.contains("\"one\""));
        assert!(!serialized.contains("\"two\""));
        assert!(serialized.contains("safe"));
    }

    #[test]
    fn unparseable_tool_arguments_fail_closed_without_schema_secrets() {
        let request = LlmRequest {
            headers: Map::new(),
            content: json!({
                "model": "anchor",
                "messages": [
                    {"role": "user", "content": "task"},
                    {
                        "role": "assistant",
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "arguments": "{\"api_secret\":\"must-not-cross-boundary\""
                            }
                        }]
                    }
                ]
            }),
        };
        let envelope = FamilyAdapter
            .decode(LlmApiFamily::OpenAIChatCompletions, &request)
            .unwrap();
        assert_eq!(
            project_request(&envelope, &CanonicalizerConfig::default()),
            Err(IneligibilityReason::ProjectionFailed)
        );
    }

    #[test]
    fn credential_classifier_prunes_nested_case_and_separator_variants() {
        let forbidden = [
            ("Client-Secret", "client-secret-value"),
            ("api_secret", "api-secret-value"),
            ("consumer.secret", "consumer-secret-value"),
            ("signing-secret", "signing-secret-value"),
            ("private_key", "private-key-value"),
            ("session.token", "session-token-value"),
            ("AUTH_TOKEN", "auth-token-value"),
            ("bearer-token", "bearer-token-value"),
            ("api_token", "api-token-value"),
            ("client.token", "client-token-value"),
            ("service_token", "service-token-value"),
            ("proxy.token", "proxy-token-value"),
            ("client_password", "client-password-value"),
            ("proxy_authorization", "proxy-authorization-value"),
            ("authorization_header", "authorization-header-value"),
            ("proxy_auth", "proxy-auth-value"),
            ("session_cookie", "session-cookie-value"),
            ("request.cookie", "request-cookie-value"),
            ("api_key_value", "api-key-qualified-value"),
            ("Api-Key-Value", "api-key-camel-qualified-value"),
            ("apiKeyValue", "api-key-no-separator-qualified-value"),
            ("secret_value", "secret-underscore-qualified-value"),
            ("secret.value", "secret-qualified-value"),
            ("Secret-Value", "secret-case-qualified-value"),
            ("secretValue", "secret-no-separator-qualified-value"),
            ("private_key_pem", "private-key-qualified-value"),
            ("PRIVATE.KEY.PEM", "private-key-case-qualified-value"),
            ("privateKeyPem", "private-key-no-separator-qualified-value"),
            ("access_token_value", "access-token-qualified-value"),
            ("Access-Token-Value", "access-token-case-qualified-value"),
            (
                "accessTokenValue",
                "access-token-no-separator-qualified-value",
            ),
        ];
        let object = Json::Object(
            forbidden
                .iter()
                .map(|(key, value)| ((*key).to_string(), json!(value)))
                .chain([
                    ("safe".to_string(), json!(true)),
                    ("max_token".to_string(), json!(8)),
                    ("token_count".to_string(), json!(3)),
                    ("public_key".to_string(), json!("not-secret-material")),
                    ("secretary".to_string(), json!("role")),
                    ("author_name".to_string(), json!("Ada")),
                    ("authentication_method".to_string(), json!("webauthn")),
                    ("authorization_status".to_string(), json!("pending")),
                    ("cookie_policy".to_string(), json!("strict")),
                    ("cookie_cutter".to_string(), json!("tool")),
                ])
                .collect(),
        );

        let sanitized_control = sanitize_json(&json!({"nested": object.clone()})).unwrap();
        let sanitized_schema = sanitize_tool_schema(&json!({
            "type": "object",
            "properties": object.clone(),
            "required": forbidden.iter().map(|(key, _)| *key).collect::<Vec<_>>()
        }))
        .unwrap();
        let sanitized_arguments = sanitize_argument_string(
            &serde_json::to_string(&object).unwrap(),
            &std::collections::BTreeSet::new(),
        )
        .unwrap();
        let sanitized_tool_result =
            sanitize_possible_json_text(&serde_json::to_string(&object).unwrap()).unwrap();

        for serialized in [
            serde_json::to_string(&sanitized_control).unwrap(),
            serde_json::to_string(&sanitized_schema).unwrap(),
            sanitized_arguments,
            sanitized_tool_result,
        ] {
            for (_, value) in forbidden {
                assert!(!serialized.contains(value), "leaked {value}");
            }
            assert!(serialized.contains("safe"));
            assert!(serialized.contains("max_token"));
            assert!(serialized.contains("token_count"));
            assert!(serialized.contains("public_key"));
            assert!(serialized.contains("secretary"));
            assert!(serialized.contains("author_name"));
            assert!(serialized.contains("authentication_method"));
            assert!(serialized.contains("authorization_status"));
            assert!(serialized.contains("cookie_policy"));
            assert!(serialized.contains("cookie_cutter"));
        }
    }

    #[test]
    fn multimodal_projection_strips_url_credentials_query_and_fragment() {
        let request = LlmRequest {
            headers: Map::new(),
            content: json!({
                "model": "anchor",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": "https://user:pass@example.test/image.png?token=secret#fragment",
                            "detail": "low"
                        }
                    }]
                }]
            }),
        };
        let envelope = FamilyAdapter
            .decode(LlmApiFamily::OpenAIChatCompletions, &request)
            .unwrap();
        let projection = project_request(&envelope, &CanonicalizerConfig::default()).unwrap();
        let serialized = serde_json::to_string(&projection).unwrap();
        assert!(serialized.contains("https://example.test/image.png"));
        for forbidden in [
            "user:pass",
            "user@",
            "pass@",
            "token=secret",
            "?token",
            "#fragment",
        ] {
            assert!(!serialized.contains(forbidden));
        }
        assert_eq!(projection.required_capabilities, vec!["multimodal_input"]);
    }

    #[test]
    fn multimodal_detail_is_strictly_allowlisted() {
        let request_with_detail = |detail: &str| LlmRequest {
            headers: Map::new(),
            content: json!({
                "model": "anchor",
                "messages": [{
                    "role": "user",
                    "content": [{
                        "type": "image_url",
                        "image_url": {
                            "url": "https://example.test/image.png",
                            "detail": detail
                        }
                    }]
                }]
            }),
        };

        for detail in ["auto", "low", "high"] {
            let envelope = FamilyAdapter
                .decode(
                    LlmApiFamily::OpenAIChatCompletions,
                    &request_with_detail(detail),
                )
                .unwrap();
            let projection = project_request(&envelope, &CanonicalizerConfig::default()).unwrap();
            let serialized = serde_json::to_string(&projection).unwrap();
            assert!(serialized.contains(&format!(r#""detail":"{detail}""#)));
        }

        let envelope = FamilyAdapter
            .decode(
                LlmApiFamily::OpenAIChatCompletions,
                &request_with_detail("Bearer must-not-cross-boundary"),
            )
            .unwrap();
        assert_eq!(
            project_request(&envelope, &CanonicalizerConfig::default()),
            Err(IneligibilityReason::ProjectionFailed)
        );
    }

    #[test]
    fn shared_policy_hashes_ignore_raw_ids_but_explicit_hashes_do_not() {
        let limits = CanonicalizerConfig::default();
        let shared = PoolSelectorConfig::default();
        let first = project_routing_context(
            &context(Some("tenant-a"), Some("agent-a")),
            &shared,
            &limits,
        )
        .unwrap();
        let second = project_routing_context(
            &context(Some("tenant-b"), Some("agent-b")),
            &shared,
            &limits,
        )
        .unwrap();
        assert_eq!(first.tenant_policy_hash, second.tenant_policy_hash);
        assert_eq!(first.agent_policy_hash, second.agent_policy_hash);
        assert_eq!(
            first.tenant_policy_hash,
            "cf79444190c703c67bc68a5313a3f523d371cbfff3d5bf976b4a66a8a9c581f8"
        );

        let explicit = PoolSelectorConfig {
            tenant_ids: Some(vec!["tenant-a".to_string(), "tenant-b".to_string()]),
            agent_ids: None,
            owner_scope_types: None,
            metadata_equals: BTreeMap::new(),
            scope_path_patterns: None,
            unknown_fields: BTreeMap::new(),
        };
        let first =
            project_routing_context(&context(Some("tenant-a"), None), &explicit, &limits).unwrap();
        let second =
            project_routing_context(&context(Some("tenant-b"), None), &explicit, &limits).unwrap();
        assert_ne!(first.tenant_policy_hash, second.tenant_policy_hash);
    }

    #[test]
    fn pool_selection_bounds_do_not_reject_safe_projection_history() {
        let envelope = FamilyAdapter
            .decode(LlmApiFamily::OpenAIChatCompletions, &tool_request())
            .unwrap();
        let limits = CanonicalizerConfig {
            max_instruction_bytes: 1,
            max_task_bytes: 1,
            max_context_messages: 1,
            max_context_bytes: 1,
            ..CanonicalizerConfig::default()
        };
        assert!(project_request(&envelope, &limits).is_ok());
    }

    #[test]
    fn projection_message_ceiling_accepts_exact_and_rejects_one_over() {
        let mut envelope = FamilyAdapter
            .decode(
                LlmApiFamily::OpenAIChatCompletions,
                &LlmRequest {
                    headers: Map::new(),
                    content: json!({
                        "model": "anchor",
                        "messages": [{"role": "user", "content": "x"}],
                    }),
                },
            )
            .unwrap();
        let message = envelope.normalized_request.messages[0].clone();
        envelope.normalized_request.messages =
            vec![message.clone(); REQUEST_PROJECTION_MAX_MESSAGES];
        assert!(project_request(&envelope, &CanonicalizerConfig::default()).is_ok());
        envelope.normalized_request.messages.push(message);
        assert_eq!(
            project_request(&envelope, &CanonicalizerConfig::default()),
            Err(IneligibilityReason::ProjectionFailed)
        );
    }

    #[test]
    fn projection_byte_ceiling_accepts_exact_and_rejects_one_over() {
        fn envelope(text: &str) -> crate::adapter::RouterRequestEnvelope {
            FamilyAdapter
                .decode(
                    LlmApiFamily::OpenAIChatCompletions,
                    &LlmRequest {
                        headers: Map::new(),
                        content: json!({
                            "model": "anchor",
                            "messages": [{"role": "user", "content": text}],
                        }),
                    },
                )
                .unwrap()
        }

        let empty = project_request(&envelope(""), &CanonicalizerConfig::default()).unwrap();
        let fixed_bytes = projection_semantic_bytes(&empty).unwrap().len();
        let exact_text = "a".repeat(REQUEST_PROJECTION_MAX_BYTES - fixed_bytes);
        let exact =
            project_request(&envelope(&exact_text), &CanonicalizerConfig::default()).unwrap();
        assert_eq!(
            projection_semantic_bytes(&exact).unwrap().len(),
            REQUEST_PROJECTION_MAX_BYTES
        );
        let one_over = format!("{exact_text}a");
        assert_eq!(
            project_request(&envelope(&one_over), &CanonicalizerConfig::default()),
            Err(IneligibilityReason::ProjectionFailed)
        );

        let mut self_consistent_oversized = exact;
        let SanitizedMessage::User { content, .. } =
            &mut self_consistent_oversized.normalized_request.messages[0]
        else {
            panic!("fixture must contain a user message");
        };
        let super::SanitizedMessageContent::Text(text) = content else {
            panic!("fixture must contain text content");
        };
        text.push('a');
        self_consistent_oversized.semantic_request_fingerprint =
            sha256_hex(&projection_semantic_bytes(&self_consistent_oversized).unwrap());
        assert_eq!(
            validate_request_projection(&self_consistent_oversized),
            Err(IneligibilityReason::ProjectionFailed)
        );
    }

    #[test]
    fn projection_rejects_integer_json_outside_the_injective_jcs_domain() {
        let maximum_safe = json!({"const": (1_u64 << 53) - 1});
        assert_eq!(sanitize_tool_schema(&maximum_safe).unwrap(), maximum_safe);

        for unsafe_value in [1_u64 << 53, (1_u64 << 53) + 1, u64::MAX] {
            let schema = json!({"const": unsafe_value});
            assert_eq!(
                sanitize_tool_schema(&schema),
                Err(IneligibilityReason::ProjectionFailed)
            );
            assert_eq!(
                sanitize_json(&schema),
                Err(IneligibilityReason::ProjectionFailed)
            );
        }
    }

    #[test]
    fn raw_and_text_json_resource_limits_precede_clone_and_parse() {
        let exact_values = vec![Json::Null; REQUEST_JSON_MAX_VALUES - 1];
        assert!(
            validate_raw_request_resource_bounds(&LlmRequest {
                headers: Map::new(),
                content: Json::Array(exact_values),
            })
            .is_ok()
        );
        let too_many_values = vec![Json::Null; REQUEST_JSON_MAX_VALUES];
        assert_eq!(
            validate_raw_request_resource_bounds(&LlmRequest {
                headers: Map::new(),
                content: Json::Array(too_many_values),
            }),
            Err(IneligibilityReason::ProjectionFailed)
        );

        let mut too_deep = Json::Null;
        for _ in 0..=REQUEST_JSON_MAX_DEPTH {
            too_deep = Json::Array(vec![too_deep]);
        }
        assert_eq!(
            sanitize_json(&too_deep),
            Err(IneligibilityReason::ProjectionFailed)
        );

        let mut text_bomb = String::from("[");
        text_bomb.push_str(&"null,".repeat(REQUEST_JSON_MAX_VALUES));
        text_bomb.push_str("null]");
        assert_eq!(
            sanitize_argument_string(&text_bomb, &BTreeSet::new()),
            Err(IneligibilityReason::ProjectionFailed)
        );
        assert_eq!(
            sanitize_possible_json_text(&text_bomb),
            Err(IneligibilityReason::ProjectionFailed)
        );
        assert_eq!(
            sanitize_possible_json_text("not valid JSON").unwrap(),
            "not valid JSON"
        );
    }

    #[test]
    fn excluded_provider_extras_do_not_enter_projection_numeric_identity() {
        let base = LlmRequest {
            headers: Map::new(),
            content: json!({
                "model": "anchor",
                "messages": [{"role": "user", "content": "hello"}],
            }),
        };
        let mut with_extra = base.clone();
        with_extra.content["vendor_counter"] = json!(1_u64 << 53);
        assert!(validate_raw_request_resource_bounds(&with_extra).is_ok());

        let base = FamilyAdapter
            .decode(LlmApiFamily::OpenAIChatCompletions, &base)
            .unwrap();
        let with_extra = FamilyAdapter
            .decode(LlmApiFamily::OpenAIChatCompletions, &with_extra)
            .unwrap();
        assert_eq!(
            project_request(&base, &CanonicalizerConfig::default()).unwrap(),
            project_request(&with_extra, &CanonicalizerConfig::default()).unwrap()
        );
    }

    #[test]
    fn candidate_model_rewrite_preserves_the_projection_byte_ceiling() {
        fn envelope(model: &str, text: &str) -> crate::adapter::RouterRequestEnvelope {
            FamilyAdapter
                .decode(
                    LlmApiFamily::OpenAIChatCompletions,
                    &LlmRequest {
                        headers: Map::new(),
                        content: json!({
                            "model": model,
                            "messages": [{"role": "user", "content": text}],
                        }),
                    },
                )
                .unwrap()
        }

        let empty = project_request(&envelope("a", ""), &CanonicalizerConfig::default()).unwrap();
        let fixed_bytes = projection_semantic_bytes(&empty).unwrap().len();
        let exact_text = "a".repeat(REQUEST_PROJECTION_MAX_BYTES - fixed_bytes);
        let exact =
            project_request(&envelope("a", &exact_text), &CanonicalizerConfig::default()).unwrap();
        let same_length = candidate_request_projection(&exact, "b").unwrap();
        assert_eq!(
            projection_semantic_bytes(&same_length).unwrap().len(),
            REQUEST_PROJECTION_MAX_BYTES
        );
        assert!(candidate_request_projection(&exact, "bb").is_err());
    }

    #[test]
    fn serialized_projection_has_no_flattened_provider_extra_map() {
        let envelope = FamilyAdapter
            .decode(LlmApiFamily::OpenAIChatCompletions, &tool_request())
            .unwrap();
        let projection = project_request(&envelope, &CanonicalizerConfig::default()).unwrap();
        let Json::Object(request) = serde_json::to_value(&projection.normalized_request).unwrap()
        else {
            panic!("sanitized request should serialize as an object")
        };
        assert!(!request.contains_key("extra"));
        assert!(!request.contains_key("vendor_secret"));
        assert!(matches!(
            projection.normalized_request.messages.first(),
            Some(SanitizedMessage::System { .. })
        ));
        assert_eq!(
            projection.normalized_request.tools.as_ref().unwrap()[0].strict,
            Some(true)
        );
    }

    #[test]
    fn routing_projection_contains_only_hashes_and_registered_scalars() {
        let mut context = context(Some("sensitive-tenant-123"), Some("sensitive-agent-456"));
        context
            .sanitized_metadata
            .insert("turn_index".to_string(), json!(7));
        context
            .sanitized_metadata
            .insert("ignored".to_string(), json!("secret"));
        let limits = CanonicalizerConfig {
            position_features: vec!["turn_index".to_string()],
            ..CanonicalizerConfig::default()
        };
        let projected: RouterRoutingContextProjectionV1 =
            project_routing_context(&context, &PoolSelectorConfig::default(), &limits).unwrap();
        assert_eq!(projected.position_features["turn_index"], json!("7"));
        let serialized = serde_json::to_string(&projected).unwrap();
        assert!(serialized.contains("turn_index"));
        assert!(!serialized.contains("ignored"));
        assert!(!serialized.contains("sensitive-tenant-123"));
        assert!(!serialized.contains("sensitive-agent-456"));
        assert!(!serialized.contains("must-not-enter-policy"));

        context
            .sanitized_metadata
            .insert("turn_index".to_string(), json!(u64::MAX));
        let maximum =
            project_routing_context(&context, &PoolSelectorConfig::default(), &limits).unwrap();
        assert_eq!(
            maximum.position_features["turn_index"],
            json!(u64::MAX.to_string())
        );

        for invalid in [
            json!(-1),
            json!(1.5),
            json!(true),
            Json::Null,
            json!("7"),
            json!([]),
            json!({}),
        ] {
            context
                .sanitized_metadata
                .insert("turn_index".to_string(), invalid);
            assert_eq!(
                project_routing_context(&context, &PoolSelectorConfig::default(), &limits),
                Err(IneligibilityReason::ProjectionFailed)
            );
        }
    }
}
