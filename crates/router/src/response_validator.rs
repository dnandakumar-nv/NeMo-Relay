// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Offline response-contract compilation and strict candidate validation.

#![allow(dead_code)] // The Spec 05 scheduler consumes the validation outcome in a later task.

use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use jsonschema::{Draft, Retrieve, Uri, Validator};
use nemo_relay::api::llm::LlmApiFamily;
use nemo_relay::codec::anthropic::AnthropicMessagesCodec;
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay::codec::openai_responses::OpenAIResponsesCodec;
use nemo_relay::codec::request::{
    AnnotatedLlmRequest, MessageContent, StructuredResponseFormatKind, ToolChoice,
};
use nemo_relay::codec::response::{AnnotatedLlmResponse, FinishReason, ResponseToolCall};
use nemo_relay::codec::traits::LlmResponseCodec;
use serde_json::{Map, Value as Json};

use crate::eligibility::IneligibilityReason;
use crate::fingerprint::{fingerprint_json, fingerprint_serializable};
use crate::judge::{DeterministicHardFailureV1, JudgeEvaluationV1, contains_sensitive_free_text};
use crate::preflight::contains_sensitive_control_material;
use crate::trajectory::{RouterResponseProjectionV1, project_anchor_response};

const DRAFT_7_URI: &str = "http://json-schema.org/draft-07/schema";
const DRAFT_7_HTTPS_URI: &str = "https://json-schema.org/draft-07/schema";
const DRAFT_2020_12_URI: &str = "https://json-schema.org/draft/2020-12/schema";
const DRAFT_2020_12_HTTP_URI: &str = "http://json-schema.org/draft/2020-12/schema";
const DRAFT_7_REFERENCE_KEYWORDS: [&str; 1] = ["$ref"];
const DRAFT_2020_12_REFERENCE_KEYWORDS: [&str; 2] = ["$ref", "$dynamicRef"];
const MAX_VALIDATION_DEPTH: usize = 64;
const MAX_VALIDATION_NODES: usize = 65_536;
const MAX_VALIDATION_STRING_BYTES: usize = 1024 * 1024;
const TOOL_MAP_ENTRY_BUDGET_BYTES: usize = 64;

/// Full, unsanitized contracts compiled once before any candidate replay starts.
///
/// This type deliberately has no serialization or debug representation. It is
/// retained only through the memory-only [`crate::preflight::EligibleCandidate`].
pub(crate) struct CandidateResponseContractsV1 {
    structured_output: Option<StructuredOutputContractV1>,
    tools: BTreeMap<String, ToolContractV1>,
    tool_choice: Option<ToolChoice>,
    max_tool_calls: Option<u64>,
    parallel_tool_calls: Option<bool>,
    max_validation_bytes: usize,
    tool_contract_fingerprint: Option<String>,
    response_contract_fingerprint: Option<String>,
}

impl CandidateResponseContractsV1 {
    pub(crate) fn tool_contract_fingerprint(&self) -> Option<&str> {
        self.tool_contract_fingerprint.as_deref()
    }

    pub(crate) fn response_contract_fingerprint(&self) -> Option<&str> {
        self.response_contract_fingerprint.as_deref()
    }
}

struct ToolContractV1 {
    argument_schema: Option<Arc<CompiledJsonSchemaV1>>,
}

enum StructuredOutputContractV1 {
    JsonObject,
    JsonSchema(Arc<CompiledJsonSchemaV1>),
}

struct CompiledJsonSchemaV1 {
    // Retain the authoritative full schema, not its persisted safe projection.
    full_schema: Json,
    validator: Validator,
    validation_cost: usize,
}

/// A readable candidate response either advances with safe evidence, produces a
/// deterministic quality failure, or remains unlabeled for an operational cause.
pub(crate) enum CandidateValidationOutcomeV1 {
    Valid {
        response: RouterResponseProjectionV1,
    },
    DeterministicFailure {
        hard_failure: DeterministicHardFailureV1,
        evaluation: JudgeEvaluationV1,
        response: Option<Box<RouterResponseProjectionV1>>,
    },
    OperationalFailure(CandidateValidationOperationalFailureV1),
}

/// Failures for which Router cannot retain enough safe response evidence to label.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CandidateValidationOperationalFailureV1 {
    ProjectionBound,
    Truncated,
    Cancelled,
    ProviderFailure,
    AmbiguousTerminal,
    UnsafeProjection,
}

/// Operational failures while extracting one readable Judge assistant output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum JudgeResponseExtractionFailureV1 {
    EvidenceBound,
    Truncated,
    ProviderCanceled,
    ProviderFailure,
    Unreadable,
    AmbiguousDecode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum SchemaDraftV1 {
    Draft7,
    Draft202012,
}

impl SchemaDraftV1 {
    const fn validator_draft(self) -> Draft {
        match self {
            Self::Draft7 => Draft::Draft7,
            Self::Draft202012 => Draft::Draft202012,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeterministicShapeFailureV1 {
    Malformed,
    ToolContract,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum StrictValidationFailureV1 {
    Deterministic(DeterministicShapeFailureV1),
    Operational(CandidateValidationOperationalFailureV1),
}

impl From<DeterministicShapeFailureV1> for StrictValidationFailureV1 {
    fn from(value: DeterministicShapeFailureV1) -> Self {
        Self::Deterministic(value)
    }
}

enum DecodeFailureV1 {
    Invalid,
    Panicked,
}

struct StrictFamilyResponseV1<'a> {
    tool_calls: Vec<ResponseToolCall>,
    text_fragments: Vec<&'a str>,
}

enum FamilyShapeOutcomeV1<'a> {
    Complete(StrictFamilyResponseV1<'a>),
    Operational(CandidateValidationOperationalFailureV1),
}

#[derive(Debug, Clone, Copy)]
struct ValidationLimitsV1 {
    max_bytes: usize,
    max_depth: usize,
    max_nodes: usize,
    max_string_bytes: usize,
    max_work: usize,
}

#[derive(Debug, Clone, Copy)]
struct JsonAdmissionV1 {
    bytes: usize,
    nodes: usize,
    work: usize,
}

struct SchemaCompilationBudgetV1 {
    occurrence_bytes: usize,
    occurrence_nodes: usize,
    occurrence_work: usize,
    retained_bytes: usize,
    retained_nodes: usize,
    retained_work: usize,
}

type CompiledSchemaCacheV1 = BTreeMap<String, Vec<Arc<CompiledJsonSchemaV1>>>;

struct ValidationWorkBudgetV1 {
    remaining: usize,
}

#[derive(Debug)]
struct OfflineReferenceError;

impl fmt::Display for OfflineReferenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("external JSON Schema retrieval is disabled")
    }
}

impl Error for OfflineReferenceError {}

#[derive(Debug, Clone, Copy)]
struct OfflineRetriever;

impl Retrieve for OfflineRetriever {
    fn retrieve(&self, _uri: &Uri<String>) -> Result<Json, Box<dyn Error + Send + Sync>> {
        Err(Box::new(OfflineReferenceError))
    }
}

impl ValidationLimitsV1 {
    fn for_bytes(max_bytes: usize) -> Self {
        Self {
            max_bytes,
            max_depth: MAX_VALIDATION_DEPTH,
            max_nodes: max_bytes.min(MAX_VALIDATION_NODES),
            max_string_bytes: max_bytes.min(MAX_VALIDATION_STRING_BYTES),
            max_work: max_bytes.saturating_mul(4),
        }
    }
}

impl ValidationWorkBudgetV1 {
    const fn new(max_work: usize) -> Self {
        Self {
            remaining: max_work,
        }
    }

    fn charge(
        &mut self,
        schema_cost: usize,
        instance: JsonAdmissionV1,
    ) -> Result<(), CandidateValidationOperationalFailureV1> {
        let cost = schema_cost
            .checked_mul(instance.work.max(1))
            .ok_or(CandidateValidationOperationalFailureV1::ProjectionBound)?;
        self.remaining = self
            .remaining
            .checked_sub(cost)
            .ok_or(CandidateValidationOperationalFailureV1::ProjectionBound)?;
        Ok(())
    }
}

impl SchemaCompilationBudgetV1 {
    const fn new(limits: ValidationLimitsV1) -> Self {
        Self {
            occurrence_bytes: limits.max_bytes,
            occurrence_nodes: limits.max_nodes,
            occurrence_work: limits.max_work,
            retained_bytes: limits.max_bytes,
            retained_nodes: limits.max_nodes,
            retained_work: limits.max_work,
        }
    }

    fn occurrence_limits(&self, limits: ValidationLimitsV1) -> ValidationLimitsV1 {
        ValidationLimitsV1 {
            max_bytes: limits.max_bytes.min(self.occurrence_bytes),
            max_depth: limits.max_depth,
            max_nodes: limits.max_nodes.min(self.occurrence_nodes),
            max_string_bytes: limits
                .max_string_bytes
                .min(self.occurrence_bytes)
                .min(self.occurrence_work),
            max_work: limits.max_work.min(self.occurrence_work),
        }
    }

    fn charge_tool_entry(&mut self, name: &str) -> Result<(), IneligibilityReason> {
        let bytes = name
            .len()
            .checked_add(TOOL_MAP_ENTRY_BUDGET_BYTES)
            .ok_or(IneligibilityReason::InvalidToolContract)?;
        let work = name
            .len()
            .checked_add(1)
            .ok_or(IneligibilityReason::InvalidToolContract)?;
        self.charge_occurrence(JsonAdmissionV1 {
            bytes,
            nodes: 1,
            work,
        })
        .map_err(|_| IneligibilityReason::InvalidToolContract)
    }

    fn charge_occurrence(&mut self, admission: JsonAdmissionV1) -> Result<(), IneligibilityReason> {
        self.occurrence_bytes = self
            .occurrence_bytes
            .checked_sub(admission.bytes)
            .ok_or(IneligibilityReason::InvalidContractSchema)?;
        self.occurrence_nodes = self
            .occurrence_nodes
            .checked_sub(admission.nodes)
            .ok_or(IneligibilityReason::InvalidContractSchema)?;
        self.occurrence_work = self
            .occurrence_work
            .checked_sub(admission.work)
            .ok_or(IneligibilityReason::InvalidContractSchema)?;
        Ok(())
    }

    fn charge_retained(&mut self, admission: JsonAdmissionV1) -> Result<(), IneligibilityReason> {
        self.retained_bytes = self
            .retained_bytes
            .checked_sub(admission.bytes)
            .ok_or(IneligibilityReason::InvalidContractSchema)?;
        self.retained_nodes = self
            .retained_nodes
            .checked_sub(admission.nodes)
            .ok_or(IneligibilityReason::InvalidContractSchema)?;
        self.retained_work = self
            .retained_work
            .checked_sub(admission.work)
            .ok_or(IneligibilityReason::InvalidContractSchema)?;
        Ok(())
    }
}

fn admit_json(
    value: &Json,
    limits: ValidationLimitsV1,
) -> Result<JsonAdmissionV1, CandidateValidationOperationalFailureV1> {
    if limits.max_bytes == 0 || limits.max_nodes == 0 {
        return Err(CandidateValidationOperationalFailureV1::ProjectionBound);
    }

    let mut pending = vec![(value, 1_usize)];
    let mut bytes = 0_usize;
    let mut nodes = 0_usize;
    let mut work = 0_usize;
    while let Some((value, depth)) = pending.pop() {
        if depth > limits.max_depth {
            return Err(CandidateValidationOperationalFailureV1::ProjectionBound);
        }
        nodes = charge_bound(nodes, 1, limits.max_nodes)?;
        work = charge_bound(work, 1, limits.max_work)?;
        match value {
            Json::Null => bytes = charge_bound(bytes, 4, limits.max_bytes)?,
            Json::Bool(true) => bytes = charge_bound(bytes, 4, limits.max_bytes)?,
            Json::Bool(false) => bytes = charge_bound(bytes, 5, limits.max_bytes)?,
            Json::Number(number) => {
                bytes = charge_bound(bytes, number.to_string().len(), limits.max_bytes)?;
            }
            Json::String(value) => {
                if value.len() > limits.max_string_bytes {
                    return Err(CandidateValidationOperationalFailureV1::ProjectionBound);
                }
                bytes = charge_bound(
                    bytes,
                    encoded_json_string_len(value, limits.max_bytes)?,
                    limits.max_bytes,
                )?;
                work = charge_bound(work, value.len(), limits.max_work)?;
            }
            Json::Array(values) => {
                bytes = charge_bound(bytes, 2, limits.max_bytes)?;
                bytes = charge_bound(bytes, values.len().saturating_sub(1), limits.max_bytes)?;
                if values.len()
                    > limits
                        .max_nodes
                        .saturating_sub(nodes)
                        .saturating_sub(pending.len())
                {
                    return Err(CandidateValidationOperationalFailureV1::ProjectionBound);
                }
                let child_depth = depth
                    .checked_add(1)
                    .ok_or(CandidateValidationOperationalFailureV1::ProjectionBound)?;
                pending.extend(values.iter().map(|value| (value, child_depth)));
            }
            Json::Object(object) => {
                bytes = charge_bound(bytes, 2, limits.max_bytes)?;
                bytes = charge_bound(bytes, object.len().saturating_sub(1), limits.max_bytes)?;
                bytes = charge_bound(bytes, object.len(), limits.max_bytes)?;
                if object.len()
                    > limits
                        .max_nodes
                        .saturating_sub(nodes)
                        .saturating_sub(pending.len())
                {
                    return Err(CandidateValidationOperationalFailureV1::ProjectionBound);
                }
                let child_depth = depth
                    .checked_add(1)
                    .ok_or(CandidateValidationOperationalFailureV1::ProjectionBound)?;
                for (key, value) in object {
                    if key.len() > limits.max_string_bytes {
                        return Err(CandidateValidationOperationalFailureV1::ProjectionBound);
                    }
                    bytes = charge_bound(
                        bytes,
                        encoded_json_string_len(key, limits.max_bytes)?,
                        limits.max_bytes,
                    )?;
                    work = charge_bound(work, key.len(), limits.max_work)?;
                    pending.push((value, child_depth));
                }
            }
        }
    }
    Ok(JsonAdmissionV1 { bytes, nodes, work })
}

/// Check the shared provider-response byte, depth, node, and string bounds.
pub(crate) fn managed_response_is_bounded(value: &Json, max_bytes: usize) -> bool {
    admit_json(value, ValidationLimitsV1::for_bytes(max_bytes)).is_ok()
}

fn charge_bound(
    current: usize,
    additional: usize,
    maximum: usize,
) -> Result<usize, CandidateValidationOperationalFailureV1> {
    current
        .checked_add(additional)
        .filter(|total| *total <= maximum)
        .ok_or(CandidateValidationOperationalFailureV1::ProjectionBound)
}

fn encoded_json_string_len(
    value: &str,
    maximum: usize,
) -> Result<usize, CandidateValidationOperationalFailureV1> {
    let mut bytes = 2_usize;
    for character in value.chars() {
        let encoded = match character {
            '"' | '\\' | '\u{0008}' | '\u{000c}' | '\n' | '\r' | '\t' => 2,
            '\u{0000}'..='\u{001f}' => 6,
            _ => character.len_utf8(),
        };
        bytes = charge_bound(bytes, encoded, maximum)?;
    }
    Ok(bytes)
}

fn admit_json_text(
    text: &str,
    limits: ValidationLimitsV1,
) -> Result<(), CandidateValidationOperationalFailureV1> {
    if text.len() > limits.max_bytes || text.len() > limits.max_work {
        return Err(CandidateValidationOperationalFailureV1::ProjectionBound);
    }
    let mut depth = 0_usize;
    let mut nodes = 1_usize;
    let mut in_string = false;
    let mut escaped = false;
    let mut string_bytes = 0_usize;
    for byte in text.bytes() {
        if in_string {
            if escaped {
                escaped = false;
                string_bytes = charge_bound(string_bytes, 1, limits.max_string_bytes)?;
            } else if byte == b'\\' {
                escaped = true;
            } else if byte == b'"' {
                in_string = false;
            } else {
                string_bytes = charge_bound(string_bytes, 1, limits.max_string_bytes)?;
            }
            continue;
        }
        match byte {
            b'"' => {
                in_string = true;
                string_bytes = 0;
                nodes = charge_bound(nodes, 1, limits.max_nodes)?;
            }
            b'{' | b'[' => {
                depth = charge_bound(depth, 1, limits.max_depth)?;
                nodes = charge_bound(nodes, 1, limits.max_nodes)?;
            }
            b'}' | b']' => depth = depth.saturating_sub(1),
            b',' => nodes = charge_bound(nodes, 1, limits.max_nodes)?,
            _ => {}
        }
    }
    Ok(())
}

/// Compile all request-owned schemas without network or filesystem resolution.
pub(crate) fn compile_candidate_response_contracts(
    request: &AnnotatedLlmRequest,
    max_validation_bytes: usize,
) -> Result<CandidateResponseContractsV1, IneligibilityReason> {
    let validation_limits = ValidationLimitsV1::for_bytes(max_validation_bytes);
    let mut compilation_budget = SchemaCompilationBudgetV1::new(validation_limits);
    let mut compiled_schemas = CompiledSchemaCacheV1::new();
    let mut tools = BTreeMap::new();
    for tool in request.tools.as_deref().unwrap_or_default() {
        if tool.tool_type != "function" || tool.function.name.is_empty() {
            return Err(IneligibilityReason::InvalidToolContract);
        }
        compilation_budget.charge_tool_entry(&tool.function.name)?;
        let argument_schema = tool
            .function
            .parameters
            .as_ref()
            .map(|schema| {
                compile_json_schema(
                    schema,
                    validation_limits,
                    &mut compilation_budget,
                    &mut compiled_schemas,
                )
            })
            .transpose()?;
        if tools
            .insert(
                tool.function.name.clone(),
                ToolContractV1 { argument_schema },
            )
            .is_some()
        {
            return Err(IneligibilityReason::InvalidToolContract);
        }
    }

    let requires_tool = matches!(
        request.tool_choice.as_ref(),
        Some(ToolChoice::Required | ToolChoice::Specific(_))
    );
    match request.tool_choice.as_ref() {
        Some(ToolChoice::Required) if tools.is_empty() => {
            return Err(IneligibilityReason::InvalidToolContract);
        }
        Some(ToolChoice::Specific(choice))
            if choice.choice_type != "function"
                || choice.function.name.is_empty()
                || !tools.contains_key(&choice.function.name) =>
        {
            return Err(IneligibilityReason::InvalidToolContract);
        }
        _ => {}
    }
    if requires_tool && request.max_tool_calls == Some(0) {
        return Err(IneligibilityReason::InvalidToolContract);
    }

    let structured_output = match request.response_format.as_ref() {
        Some(format) => Some(match format.kind {
            StructuredResponseFormatKind::JsonObject => {
                if format.schema.is_some() {
                    return Err(IneligibilityReason::InvalidContractSchema);
                } else {
                    StructuredOutputContractV1::JsonObject
                }
            }
            StructuredResponseFormatKind::JsonSchema => {
                let schema = format
                    .schema
                    .as_ref()
                    .ok_or(IneligibilityReason::InvalidContractSchema)?;
                StructuredOutputContractV1::JsonSchema(compile_json_schema(
                    schema,
                    validation_limits,
                    &mut compilation_budget,
                    &mut compiled_schemas,
                )?)
            }
        }),
        None => None,
    };

    let tool_contract_fingerprint = request
        .tools
        .as_ref()
        .map(fingerprint_serializable)
        .transpose()
        .map_err(|_| IneligibilityReason::InvalidToolContract)?;
    let response_contract_fingerprint = request
        .response_format
        .as_ref()
        .map(fingerprint_serializable)
        .transpose()
        .map_err(|_| IneligibilityReason::InvalidContractSchema)?;

    Ok(CandidateResponseContractsV1 {
        structured_output,
        tools,
        tool_choice: request.tool_choice.clone(),
        max_tool_calls: request.max_tool_calls,
        parallel_tool_calls: request.parallel_tool_calls,
        max_validation_bytes,
        tool_contract_fingerprint,
        response_contract_fingerprint,
    })
}

fn compile_json_schema(
    schema: &Json,
    limits: ValidationLimitsV1,
    compilation_budget: &mut SchemaCompilationBudgetV1,
    compiled_schemas: &mut CompiledSchemaCacheV1,
) -> Result<Arc<CompiledJsonSchemaV1>, IneligibilityReason> {
    let admission = admit_json(schema, compilation_budget.occurrence_limits(limits))
        .map_err(|_| IneligibilityReason::InvalidContractSchema)?;
    compilation_budget.charge_occurrence(admission)?;
    let fingerprint =
        fingerprint_json(schema).map_err(|_| IneligibilityReason::InvalidContractSchema)?;
    if let Some(compiled) = compiled_schemas.get(&fingerprint).and_then(|schemas| {
        schemas
            .iter()
            .find(|compiled| compiled.full_schema == *schema)
    }) {
        return Ok(Arc::clone(compiled));
    }
    compilation_budget.charge_retained(admission)?;
    let draft = schema_draft(schema)?;
    reject_nonlocal_references(schema, draft, admission.nodes)?;
    let meta_result = catch_unwind(AssertUnwindSafe(|| match draft {
        SchemaDraftV1::Draft7 => jsonschema::draft7::meta::validate(schema),
        SchemaDraftV1::Draft202012 => jsonschema::draft202012::meta::validate(schema),
    }))
    .map_err(|_| IneligibilityReason::InvalidContractSchema)?;
    meta_result.map_err(|_| IneligibilityReason::InvalidContractSchema)?;
    let validator = catch_unwind(AssertUnwindSafe(|| {
        jsonschema::options()
            .with_draft(draft.validator_draft())
            .with_retriever(OfflineRetriever)
            .build(schema)
    }))
    .map_err(|_| IneligibilityReason::InvalidContractSchema)?
    .map_err(|_| IneligibilityReason::InvalidContractSchema)?;
    let compiled = Arc::new(CompiledJsonSchemaV1 {
        full_schema: schema.clone(),
        validator,
        validation_cost: admission.nodes.max(1),
    });
    compiled_schemas
        .entry(fingerprint)
        .or_default()
        .push(Arc::clone(&compiled));
    Ok(compiled)
}

fn schema_draft(schema: &Json) -> Result<SchemaDraftV1, IneligibilityReason> {
    let object = schema
        .as_object()
        .ok_or(IneligibilityReason::InvalidContractSchema)?;
    let dialect = object
        .get("$schema")
        .ok_or(IneligibilityReason::UnsupportedContractSchema)?;
    parse_schema_draft(dialect)
}

fn parse_schema_draft(dialect: &Json) -> Result<SchemaDraftV1, IneligibilityReason> {
    let dialect = dialect
        .as_str()
        .ok_or(IneligibilityReason::InvalidContractSchema)?;
    match dialect.trim_end_matches('#') {
        DRAFT_7_URI | DRAFT_7_HTTPS_URI => Ok(SchemaDraftV1::Draft7),
        DRAFT_2020_12_URI | DRAFT_2020_12_HTTP_URI => Ok(SchemaDraftV1::Draft202012),
        _ => Err(IneligibilityReason::UnsupportedContractSchema),
    }
}

fn reject_nonlocal_references(
    schema: &Json,
    root_draft: SchemaDraftV1,
    max_visits: usize,
) -> Result<(), IneligibilityReason> {
    let mut pending = vec![(schema, root_draft, schema, root_draft)];
    let mut visited = BTreeSet::new();
    while let Some((schema, inherited_draft, inherited_resource, inherited_resource_draft)) =
        pending.pop()
    {
        let Some(object) = schema.as_object() else {
            continue;
        };
        let draft = object
            .get("$schema")
            .map(parse_schema_draft)
            .transpose()?
            .unwrap_or(inherited_draft);
        let (resource, resource_draft) = match object.get("$id") {
            Some(Json::String(identifier))
                if !identifier.is_empty() && !identifier.starts_with('#') =>
            {
                (schema, draft)
            }
            Some(Json::String(_)) | None => (inherited_resource, inherited_resource_draft),
            Some(_) => return Err(IneligibilityReason::InvalidContractSchema),
        };
        let visit_key = (
            schema as *const Json as usize,
            draft,
            resource as *const Json as usize,
        );
        if !visited.insert(visit_key) {
            continue;
        }
        if visited.len() > max_visits {
            return Err(IneligibilityReason::InvalidContractSchema);
        }
        let reference_keywords = match draft {
            SchemaDraftV1::Draft7 => DRAFT_7_REFERENCE_KEYWORDS.as_slice(),
            SchemaDraftV1::Draft202012 => DRAFT_2020_12_REFERENCE_KEYWORDS.as_slice(),
        };
        for keyword in reference_keywords {
            if let Some(reference) = object.get(*keyword) {
                let reference = reference
                    .as_str()
                    .ok_or(IneligibilityReason::InvalidContractSchema)?;
                if !reference.starts_with('#') {
                    return Err(IneligibilityReason::NonLocalContractReference);
                }
                if let Some(target) = local_pointer_target(resource, reference)? {
                    pending.push((target, resource_draft, resource, resource_draft));
                }
            }
        }
        for child in draft.validator_draft().subresources_of(schema) {
            pending.push((child, draft, resource, resource_draft));
        }
    }
    Ok(())
}

fn local_pointer_target<'a>(
    resource: &'a Json,
    reference: &str,
) -> Result<Option<&'a Json>, IneligibilityReason> {
    let fragment = reference
        .strip_prefix('#')
        .ok_or(IneligibilityReason::NonLocalContractReference)?;
    let fragment = percent_decode_local_fragment(fragment)?;
    if fragment.is_empty() {
        return Ok(Some(resource));
    }
    if !fragment.starts_with('/') {
        return Ok(None);
    }
    resource
        .pointer(&fragment)
        .map(Some)
        .ok_or(IneligibilityReason::InvalidContractSchema)
}

fn percent_decode_local_fragment(fragment: &str) -> Result<String, IneligibilityReason> {
    let bytes = fragment.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'%' {
            let high = bytes
                .get(index + 1)
                .and_then(|byte| hexadecimal_value(*byte));
            let low = bytes
                .get(index + 2)
                .and_then(|byte| hexadecimal_value(*byte));
            let (Some(high), Some(low)) = (high, low) else {
                return Err(IneligibilityReason::InvalidContractSchema);
            };
            decoded.push((high << 4) | low);
            index += 3;
        } else {
            decoded.push(bytes[index]);
            index += 1;
        }
    }
    String::from_utf8(decoded).map_err(|_| IneligibilityReason::InvalidContractSchema)
}

const fn hexadecimal_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

/// Extract one complete Judge assistant output through the strict family codec boundary.
pub(crate) fn extract_judge_assistant_text(
    family: LlmApiFamily,
    response: &Json,
    max_outer_response_bytes: usize,
) -> Result<String, JudgeResponseExtractionFailureV1> {
    let limits = ValidationLimitsV1::for_bytes(max_outer_response_bytes);
    admit_json(response, limits).map_err(map_judge_extraction_failure)?;

    let strict = match validate_family_shape(family, response, limits) {
        Ok(FamilyShapeOutcomeV1::Complete(strict)) => strict,
        Ok(FamilyShapeOutcomeV1::Operational(failure))
        | Err(StrictValidationFailureV1::Operational(failure)) => {
            return Err(map_judge_extraction_failure(failure));
        }
        Err(StrictValidationFailureV1::Deterministic(_)) => {
            return Err(JudgeResponseExtractionFailureV1::Unreadable);
        }
    };

    let decoded = match decode_response_safely(family, response) {
        Ok(decoded) => decoded,
        Err(DecodeFailureV1::Invalid) => {
            return Err(JudgeResponseExtractionFailureV1::Unreadable);
        }
        Err(DecodeFailureV1::Panicked) => {
            return Err(JudgeResponseExtractionFailureV1::AmbiguousDecode);
        }
    };
    let decoded_tool_calls = decoded.tool_calls.as_deref().unwrap_or_default();
    if decoded_tool_calls != strict.tool_calls
        || !decoded_text_matches(&strict.text_fragments, &decoded)
    {
        return Err(JudgeResponseExtractionFailureV1::AmbiguousDecode);
    }
    if !strict.tool_calls.is_empty() {
        return Err(JudgeResponseExtractionFailureV1::Unreadable);
    }

    match decoded.message {
        Some(MessageContent::Text(text)) => Ok(text),
        Some(MessageContent::Parts(_)) | None => Err(JudgeResponseExtractionFailureV1::Unreadable),
    }
}

fn map_judge_extraction_failure(
    failure: CandidateValidationOperationalFailureV1,
) -> JudgeResponseExtractionFailureV1 {
    match failure {
        CandidateValidationOperationalFailureV1::ProjectionBound => {
            JudgeResponseExtractionFailureV1::EvidenceBound
        }
        CandidateValidationOperationalFailureV1::Truncated => {
            JudgeResponseExtractionFailureV1::Truncated
        }
        CandidateValidationOperationalFailureV1::Cancelled => {
            JudgeResponseExtractionFailureV1::ProviderCanceled
        }
        CandidateValidationOperationalFailureV1::ProviderFailure => {
            JudgeResponseExtractionFailureV1::ProviderFailure
        }
        CandidateValidationOperationalFailureV1::AmbiguousTerminal
        | CandidateValidationOperationalFailureV1::UnsafeProjection => {
            JudgeResponseExtractionFailureV1::AmbiguousDecode
        }
    }
}

/// Validate one complete readable provider payload without starting transport.
pub(crate) fn validate_candidate_response(
    family: LlmApiFamily,
    response: &Json,
    contracts: &CandidateResponseContractsV1,
    max_projection_bytes: usize,
    is_partial: bool,
) -> CandidateValidationOutcomeV1 {
    let limits =
        ValidationLimitsV1::for_bytes(max_projection_bytes.min(contracts.max_validation_bytes));
    if let Err(failure) = admit_json(response, limits) {
        return CandidateValidationOutcomeV1::OperationalFailure(failure);
    }

    let strict = match validate_family_shape(family, response, limits) {
        Ok(FamilyShapeOutcomeV1::Complete(strict)) => strict,
        Ok(FamilyShapeOutcomeV1::Operational(failure)) => {
            return CandidateValidationOutcomeV1::OperationalFailure(failure);
        }
        Err(StrictValidationFailureV1::Operational(failure)) => {
            return CandidateValidationOutcomeV1::OperationalFailure(failure);
        }
        Err(StrictValidationFailureV1::Deterministic(failure)) => {
            let hard_failure = match failure {
                DeterministicShapeFailureV1::Malformed => {
                    DeterministicHardFailureV1::MalformedCandidate
                }
                DeterministicShapeFailureV1::ToolContract => {
                    DeterministicHardFailureV1::ToolContract
                }
            };
            return deterministic_after_best_effort_projection(
                family,
                response,
                hard_failure,
                limits,
                max_projection_bytes,
                is_partial,
            );
        }
    };

    if !projection_control_fields_are_safe(response, &strict) {
        return CandidateValidationOutcomeV1::OperationalFailure(
            CandidateValidationOperationalFailureV1::UnsafeProjection,
        );
    }

    let decoded = match decode_response_safely(family, response) {
        Ok(decoded) => decoded,
        Err(DecodeFailureV1::Invalid) => {
            return deterministic_failure(
                DeterministicHardFailureV1::MalformedCandidate,
                is_partial,
                None,
            );
        }
        Err(DecodeFailureV1::Panicked) => {
            return CandidateValidationOutcomeV1::OperationalFailure(
                CandidateValidationOperationalFailureV1::AmbiguousTerminal,
            );
        }
    };

    let decoded_tool_calls = decoded.tool_calls.as_deref().unwrap_or_default();
    let mut validation_work = ValidationWorkBudgetV1::new(limits.max_work);
    let hard_failure = if decoded_tool_calls != strict.tool_calls
        || !decoded_text_matches(&strict.text_fragments, &decoded)
    {
        Some(DeterministicHardFailureV1::MalformedCandidate)
    } else {
        match tool_contract_is_valid(contracts, decoded_tool_calls, limits, &mut validation_work) {
            Ok(false) => Some(DeterministicHardFailureV1::ToolContract),
            Err(failure) => {
                return CandidateValidationOutcomeV1::OperationalFailure(failure);
            }
            Ok(true) => match structured_output_is_valid(
                contracts,
                &strict.text_fragments,
                limits,
                &mut validation_work,
            ) {
                Ok(false) => Some(DeterministicHardFailureV1::ResponseSchema),
                Ok(true) => None,
                Err(failure) => {
                    return CandidateValidationOutcomeV1::OperationalFailure(failure);
                }
            },
        }
    };

    let projection = match project_response_safely(family, response, max_projection_bytes) {
        Ok(projection) => projection,
        Err(failure) => {
            return CandidateValidationOutcomeV1::OperationalFailure(failure);
        }
    };

    match hard_failure {
        Some(hard_failure) => deterministic_failure(hard_failure, is_partial, Some(projection)),
        None => CandidateValidationOutcomeV1::Valid {
            response: projection,
        },
    }
}

fn deterministic_after_best_effort_projection(
    family: LlmApiFamily,
    response: &Json,
    hard_failure: DeterministicHardFailureV1,
    limits: ValidationLimitsV1,
    max_projection_bytes: usize,
    is_partial: bool,
) -> CandidateValidationOutcomeV1 {
    match decode_response_safely(family, response) {
        Ok(_) => {}
        Err(DecodeFailureV1::Invalid) => {
            return deterministic_failure(
                DeterministicHardFailureV1::MalformedCandidate,
                is_partial,
                None,
            );
        }
        Err(DecodeFailureV1::Panicked) => {
            return CandidateValidationOutcomeV1::OperationalFailure(
                CandidateValidationOperationalFailureV1::AmbiguousTerminal,
            );
        }
    }
    match raw_tool_arguments_are_projection_safe(family, response, limits) {
        Ok(true) => {}
        Ok(false) => return deterministic_failure(hard_failure, is_partial, None),
        Err(failure) => return CandidateValidationOutcomeV1::OperationalFailure(failure),
    }
    // A strict-shape tool failure can be caused by undecodable argument text. Core preserves that
    // text as a JSON string, so projecting it would bypass the structured credential-key
    // sanitizer. Keep the deterministic label, but retain no response evidence for this path.
    if hard_failure == DeterministicHardFailureV1::ToolContract {
        return deterministic_failure(hard_failure, is_partial, None);
    }
    match project_response_safely(family, response, max_projection_bytes) {
        Ok(projection) => deterministic_failure(hard_failure, is_partial, Some(projection)),
        Err(failure) => CandidateValidationOutcomeV1::OperationalFailure(failure),
    }
}

fn raw_tool_arguments_are_projection_safe(
    family: LlmApiFamily,
    response: &Json,
    limits: ValidationLimitsV1,
) -> Result<bool, CandidateValidationOperationalFailureV1> {
    let Some(root) = response.as_object() else {
        return Ok(true);
    };
    match family {
        LlmApiFamily::OpenAIChatCompletions => {
            for choice in root
                .get("choices")
                .and_then(Json::as_array)
                .into_iter()
                .flatten()
            {
                let calls = choice
                    .get("message")
                    .and_then(Json::as_object)
                    .and_then(|message| message.get("tool_calls"))
                    .and_then(Json::as_array);
                for call in calls.into_iter().flatten() {
                    let Some(arguments) = call
                        .get("function")
                        .and_then(Json::as_object)
                        .and_then(|function| function.get("arguments"))
                        .and_then(Json::as_str)
                    else {
                        continue;
                    };
                    if !tool_argument_text_is_projection_safe(arguments, limits)? {
                        return Ok(false);
                    }
                }
            }
        }
        LlmApiFamily::OpenAIResponses => {
            for item in root
                .get("output")
                .and_then(Json::as_array)
                .into_iter()
                .flatten()
            {
                if item.get("type").and_then(Json::as_str) != Some("function_call") {
                    continue;
                }
                let Some(arguments) = item.get("arguments").and_then(Json::as_str) else {
                    continue;
                };
                if !tool_argument_text_is_projection_safe(arguments, limits)? {
                    return Ok(false);
                }
            }
        }
        LlmApiFamily::AnthropicMessages => {}
    }
    Ok(true)
}

fn tool_argument_text_is_projection_safe(
    arguments: &str,
    limits: ValidationLimitsV1,
) -> Result<bool, CandidateValidationOperationalFailureV1> {
    admit_json_text(arguments, limits)?;
    Ok(serde_json::from_str::<Json>(arguments).is_ok())
}

fn deterministic_failure(
    hard_failure: DeterministicHardFailureV1,
    is_partial: bool,
    response: Option<RouterResponseProjectionV1>,
) -> CandidateValidationOutcomeV1 {
    CandidateValidationOutcomeV1::DeterministicFailure {
        hard_failure,
        evaluation: JudgeEvaluationV1::deterministic_hard_failure(hard_failure, is_partial),
        response: response.map(Box::new),
    }
}

fn decode_response_safely(
    family: LlmApiFamily,
    response: &Json,
) -> Result<AnnotatedLlmResponse, DecodeFailureV1> {
    catch_unwind(AssertUnwindSafe(|| match family {
        LlmApiFamily::OpenAIChatCompletions => OpenAIChatCodec.decode_response(response),
        LlmApiFamily::OpenAIResponses => OpenAIResponsesCodec.decode_response(response),
        LlmApiFamily::AnthropicMessages => AnthropicMessagesCodec.decode_response(response),
    }))
    .map_err(|_| DecodeFailureV1::Panicked)?
    .map_err(|_| DecodeFailureV1::Invalid)
}

fn project_response_safely(
    family: LlmApiFamily,
    response: &Json,
    max_projection_bytes: usize,
) -> Result<RouterResponseProjectionV1, CandidateValidationOperationalFailureV1> {
    let projection = catch_unwind(AssertUnwindSafe(|| {
        project_anchor_response(family, response, max_projection_bytes)
    }))
    .map_err(|_| CandidateValidationOperationalFailureV1::AmbiguousTerminal)?
    .map_err(|_| CandidateValidationOperationalFailureV1::ProjectionBound)?;
    if response_projection_controls_are_safe(&projection) {
        Ok(projection)
    } else {
        Err(CandidateValidationOperationalFailureV1::UnsafeProjection)
    }
}

fn response_projection_controls_are_safe(projection: &RouterResponseProjectionV1) -> bool {
    projection
        .id
        .iter()
        .chain(projection.model.iter())
        .map(String::as_str)
        .chain(
            projection
                .finish_reason
                .iter()
                .filter_map(|reason| match reason {
                    FinishReason::Unknown(reason) => Some(reason.as_str()),
                    _ => None,
                }),
        )
        .chain(
            projection
                .tool_calls
                .iter()
                .flatten()
                .flat_map(|call| [call.id.as_str(), call.name.as_str()]),
        )
        .all(response_control_value_is_safe)
}

fn projection_control_fields_are_safe(
    response: &Json,
    strict: &StrictFamilyResponseV1<'_>,
) -> bool {
    let Some(root) = response.as_object() else {
        return false;
    };
    ["id", "model"]
        .into_iter()
        .filter_map(|key| root.get(key).and_then(Json::as_str))
        .chain(
            strict
                .tool_calls
                .iter()
                .flat_map(|call| [call.id.as_str(), call.name.as_str()]),
        )
        .all(response_control_value_is_safe)
}

fn response_control_value_is_safe(value: &str) -> bool {
    !contains_sensitive_free_text(value) && !contains_sensitive_control_material(value)
}

fn decoded_text_matches(raw_fragments: &[&str], decoded: &AnnotatedLlmResponse) -> bool {
    match (raw_fragments, decoded.message.as_ref()) {
        ([], None) => true,
        ([raw], Some(MessageContent::Text(decoded))) => *raw == decoded,
        (raw, Some(MessageContent::Text(decoded))) => raw.join("\n") == *decoded,
        _ => false,
    }
}

fn tool_contract_is_valid(
    contracts: &CandidateResponseContractsV1,
    calls: &[ResponseToolCall],
    limits: ValidationLimitsV1,
    validation_work: &mut ValidationWorkBudgetV1,
) -> Result<bool, CandidateValidationOperationalFailureV1> {
    if contracts
        .max_tool_calls
        .is_some_and(|maximum| u64::try_from(calls.len()).map_or(true, |count| count > maximum))
        || (contracts.parallel_tool_calls == Some(false) && calls.len() > 1)
    {
        return Ok(false);
    }

    match contracts.tool_choice.as_ref() {
        Some(ToolChoice::None) if !calls.is_empty() => return Ok(false),
        Some(ToolChoice::Required) if calls.is_empty() => return Ok(false),
        Some(ToolChoice::Specific(choice))
            if calls.is_empty() || calls.iter().any(|call| call.name != choice.function.name) =>
        {
            return Ok(false);
        }
        _ => {}
    }

    for call in calls {
        let Some(tool) = contracts.tools.get(&call.name) else {
            return Ok(false);
        };
        let instance = admit_json(&call.arguments, limits)?;
        if let Some(schema) = tool.argument_schema.as_ref() {
            validation_work.charge(schema.validation_cost, instance)?;
            if !schema_is_valid_safely(schema, &call.arguments)? {
                return Ok(false);
            }
        }
    }
    Ok(true)
}

fn structured_output_is_valid(
    contracts: &CandidateResponseContractsV1,
    text_fragments: &[&str],
    limits: ValidationLimitsV1,
    validation_work: &mut ValidationWorkBudgetV1,
) -> Result<bool, CandidateValidationOperationalFailureV1> {
    let Some(contract) = contracts.structured_output.as_ref() else {
        return Ok(true);
    };
    let [text] = text_fragments else {
        return Ok(false);
    };
    admit_json_text(text, limits)?;
    let Ok(value) = serde_json::from_str::<Json>(text) else {
        return Ok(false);
    };
    let instance = admit_json(&value, limits)?;
    match contract {
        StructuredOutputContractV1::JsonObject => Ok(value.is_object()),
        StructuredOutputContractV1::JsonSchema(schema) => {
            validation_work.charge(schema.validation_cost, instance)?;
            schema_is_valid_safely(schema, &value)
        }
    }
}

fn schema_is_valid_safely(
    schema: &CompiledJsonSchemaV1,
    instance: &Json,
) -> Result<bool, CandidateValidationOperationalFailureV1> {
    catch_unwind(AssertUnwindSafe(|| schema.validator.is_valid(instance)))
        .map_err(|_| CandidateValidationOperationalFailureV1::AmbiguousTerminal)
}

fn validate_family_shape(
    family: LlmApiFamily,
    response: &Json,
    limits: ValidationLimitsV1,
) -> Result<FamilyShapeOutcomeV1<'_>, StrictValidationFailureV1> {
    match family {
        LlmApiFamily::OpenAIChatCompletions => validate_chat_shape(response, limits),
        LlmApiFamily::OpenAIResponses => validate_responses_shape(response, limits),
        LlmApiFamily::AnthropicMessages => validate_anthropic_shape(response, limits),
    }
}

fn validate_chat_shape(
    response: &Json,
    limits: ValidationLimitsV1,
) -> Result<FamilyShapeOutcomeV1<'_>, StrictValidationFailureV1> {
    let root = required_object(response)?;
    required_nonempty_string(root, "id")?;
    require_exact_string(root, "object", "chat.completion")?;
    required_u64(root, "created")?;
    required_nonempty_string(root, "model")?;
    let choices = required_array(root, "choices")?;
    if choices.len() != 1 {
        return Err(DeterministicShapeFailureV1::Malformed.into());
    }
    let choice = required_object(&choices[0])?;
    required_u64(choice, "index")?;
    let message = required_object_field(choice, "message")?;
    require_exact_string(message, "role", "assistant")?;
    match message.get("content") {
        Some(Json::String(_) | Json::Null) => {}
        _ => return Err(DeterministicShapeFailureV1::Malformed.into()),
    }
    match message.get("refusal") {
        None | Some(Json::Null) => {}
        Some(Json::String(_)) => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                CandidateValidationOperationalFailureV1::ProviderFailure,
            ));
        }
        Some(_) => return Err(DeterministicShapeFailureV1::Malformed.into()),
    }
    let finish_reason = match choice.get("finish_reason") {
        Some(Json::String(reason)) => reason.as_str(),
        Some(Json::Null) => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                CandidateValidationOperationalFailureV1::AmbiguousTerminal,
            ));
        }
        _ => return Err(DeterministicShapeFailureV1::Malformed.into()),
    };
    match finish_reason {
        "length" => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                CandidateValidationOperationalFailureV1::Truncated,
            ));
        }
        "content_filter" => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                CandidateValidationOperationalFailureV1::ProviderFailure,
            ));
        }
        "stop" | "tool_calls" => {}
        _ => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                CandidateValidationOperationalFailureV1::AmbiguousTerminal,
            ));
        }
    }

    let mut tool_calls = Vec::new();
    if let Some(raw_calls) = optional_array(message, "tool_calls")? {
        for raw_call in raw_calls {
            let raw_call = required_object(raw_call)?;
            let id = required_nonempty_string(raw_call, "id")?;
            require_exact_string(raw_call, "type", "function")?;
            let function = required_object_field(raw_call, "function")?;
            let name = required_nonempty_string(function, "name")?;
            let arguments = required_string(function, "arguments")?;
            let arguments = parse_tool_arguments(arguments, limits)?;
            tool_calls.push(ResponseToolCall {
                id: id.to_string(),
                name: name.to_string(),
                arguments,
            });
        }
    }
    let text_fragments = message
        .get("content")
        .and_then(Json::as_str)
        .into_iter()
        .collect::<Vec<_>>();
    if (finish_reason == "stop" && (!tool_calls.is_empty() || text_fragments.is_empty()))
        || (finish_reason == "tool_calls" && tool_calls.is_empty())
    {
        return Err(DeterministicShapeFailureV1::Malformed.into());
    }
    Ok(FamilyShapeOutcomeV1::Complete(StrictFamilyResponseV1 {
        tool_calls,
        text_fragments,
    }))
}

fn validate_responses_shape(
    response: &Json,
    limits: ValidationLimitsV1,
) -> Result<FamilyShapeOutcomeV1<'_>, StrictValidationFailureV1> {
    let root = required_object(response)?;
    required_nonempty_string(root, "id")?;
    require_exact_string(root, "object", "response")?;
    required_number(root, "created_at")?;
    required_nonempty_string(root, "model")?;
    let output = required_array(root, "output")?;
    let status = match root.get("status") {
        Some(Json::String(status)) => status.as_str(),
        Some(Json::Null) => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                CandidateValidationOperationalFailureV1::AmbiguousTerminal,
            ));
        }
        _ => return Err(DeterministicShapeFailureV1::Malformed.into()),
    };
    match status {
        "incomplete" => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                responses_incomplete_failure(root),
            ));
        }
        "failed" => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                CandidateValidationOperationalFailureV1::ProviderFailure,
            ));
        }
        "cancelled" => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                CandidateValidationOperationalFailureV1::Cancelled,
            ));
        }
        "completed" => {}
        _ => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                CandidateValidationOperationalFailureV1::AmbiguousTerminal,
            ));
        }
    }
    if root.get("error").is_some_and(|value| !value.is_null()) {
        return Ok(FamilyShapeOutcomeV1::Operational(
            CandidateValidationOperationalFailureV1::ProviderFailure,
        ));
    }
    if root
        .get("incomplete_details")
        .is_some_and(|value| !value.is_null())
    {
        return Ok(FamilyShapeOutcomeV1::Operational(
            CandidateValidationOperationalFailureV1::AmbiguousTerminal,
        ));
    }

    let mut tool_calls = Vec::new();
    let mut text_fragments = Vec::new();
    let mut has_refusal = false;
    for item in output {
        let item = required_object(item)?;
        match required_string(item, "type")? {
            "reasoning" => {
                required_nonempty_string(item, "id")?;
                required_array(item, "summary")?;
                if let Some(status) = item.get("status")
                    && !status.is_null()
                {
                    classify_completed_status_value(status)?;
                }
            }
            "message" => {
                has_refusal |= validate_responses_message(item, &mut text_fragments)?;
            }
            "output_text" => {
                text_fragments.push(validate_responses_text(item)?);
            }
            "function_call" => {
                required_nonempty_string(item, "id")?;
                let id = required_nonempty_string(item, "call_id")?;
                let name = required_nonempty_string(item, "name")?;
                let arguments = required_string(item, "arguments")?;
                require_completed_status(item, "status")?;
                let arguments = parse_tool_arguments(arguments, limits)?;
                tool_calls.push(ResponseToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments,
                });
            }
            _ => return Err(DeterministicShapeFailureV1::Malformed.into()),
        }
    }
    if has_refusal {
        return Ok(FamilyShapeOutcomeV1::Operational(
            CandidateValidationOperationalFailureV1::ProviderFailure,
        ));
    }
    if let Some(output_text) = root.get("output_text") {
        let output_text = output_text
            .as_str()
            .ok_or(DeterministicShapeFailureV1::Malformed)?;
        if !text_fragments.is_empty() {
            return Err(DeterministicShapeFailureV1::Malformed.into());
        }
        text_fragments.push(output_text);
    }
    Ok(FamilyShapeOutcomeV1::Complete(StrictFamilyResponseV1 {
        tool_calls,
        text_fragments,
    }))
}

fn validate_responses_message<'a>(
    message: &'a Map<String, Json>,
    text_fragments: &mut Vec<&'a str>,
) -> Result<bool, StrictValidationFailureV1> {
    required_nonempty_string(message, "id")?;
    require_exact_string(message, "role", "assistant")?;
    require_completed_status(message, "status")?;
    let mut has_refusal = false;
    for block in required_array(message, "content")? {
        let block = required_object(block)?;
        match required_string(block, "type")? {
            "output_text" => text_fragments.push(validate_responses_text(block)?),
            "refusal" => {
                required_string(block, "refusal")?;
                has_refusal = true;
            }
            _ => return Err(DeterministicShapeFailureV1::Malformed.into()),
        }
    }
    Ok(has_refusal)
}

fn validate_responses_text(block: &Map<String, Json>) -> Result<&str, StrictValidationFailureV1> {
    let text = required_string(block, "text")?;
    if let Some(annotations) = block.get("annotations")
        && !annotations.is_array()
    {
        return Err(DeterministicShapeFailureV1::Malformed.into());
    }
    Ok(text)
}

fn responses_incomplete_failure(
    root: &Map<String, Json>,
) -> CandidateValidationOperationalFailureV1 {
    match root
        .get("incomplete_details")
        .and_then(Json::as_object)
        .and_then(|details| details.get("reason"))
        .and_then(Json::as_str)
    {
        Some("max_output_tokens") => CandidateValidationOperationalFailureV1::Truncated,
        Some("content_filter") => CandidateValidationOperationalFailureV1::ProviderFailure,
        _ => CandidateValidationOperationalFailureV1::AmbiguousTerminal,
    }
}

fn require_completed_status(
    object: &Map<String, Json>,
    key: &str,
) -> Result<(), StrictValidationFailureV1> {
    let status = object
        .get(key)
        .ok_or(DeterministicShapeFailureV1::Malformed)?;
    classify_completed_status_value(status)
}

fn classify_completed_status_value(status: &Json) -> Result<(), StrictValidationFailureV1> {
    match status {
        Json::String(status) => match status.as_str() {
            "completed" => Ok(()),
            "incomplete" => Err(StrictValidationFailureV1::Operational(
                CandidateValidationOperationalFailureV1::Truncated,
            )),
            "failed" => Err(StrictValidationFailureV1::Operational(
                CandidateValidationOperationalFailureV1::ProviderFailure,
            )),
            "cancelled" => Err(StrictValidationFailureV1::Operational(
                CandidateValidationOperationalFailureV1::Cancelled,
            )),
            _ => Err(StrictValidationFailureV1::Operational(
                CandidateValidationOperationalFailureV1::AmbiguousTerminal,
            )),
        },
        Json::Null => Err(StrictValidationFailureV1::Operational(
            CandidateValidationOperationalFailureV1::AmbiguousTerminal,
        )),
        _ => Err(DeterministicShapeFailureV1::Malformed.into()),
    }
}

fn validate_anthropic_shape(
    response: &Json,
    limits: ValidationLimitsV1,
) -> Result<FamilyShapeOutcomeV1<'_>, StrictValidationFailureV1> {
    let root = required_object(response)?;
    required_nonempty_string(root, "id")?;
    require_exact_string(root, "type", "message")?;
    require_exact_string(root, "role", "assistant")?;
    required_nonempty_string(root, "model")?;
    let content = required_array(root, "content")?;
    let usage = required_object_field(root, "usage")?;
    let input_tokens = required_u64(usage, "input_tokens")?;
    let output_tokens = required_u64(usage, "output_tokens")?;
    if input_tokens.checked_add(output_tokens).is_none() {
        return Ok(FamilyShapeOutcomeV1::Operational(
            CandidateValidationOperationalFailureV1::AmbiguousTerminal,
        ));
    }
    let stop_reason = match root.get("stop_reason") {
        Some(Json::String(reason)) => reason.as_str(),
        Some(Json::Null) => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                CandidateValidationOperationalFailureV1::AmbiguousTerminal,
            ));
        }
        _ => return Err(DeterministicShapeFailureV1::Malformed.into()),
    };
    match stop_reason {
        "max_tokens" => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                CandidateValidationOperationalFailureV1::Truncated,
            ));
        }
        "refusal" => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                CandidateValidationOperationalFailureV1::ProviderFailure,
            ));
        }
        "end_turn" | "stop_sequence" | "tool_use" => {}
        _ => {
            return Ok(FamilyShapeOutcomeV1::Operational(
                CandidateValidationOperationalFailureV1::AmbiguousTerminal,
            ));
        }
    }
    match (stop_reason, root.get("stop_sequence")) {
        ("stop_sequence", Some(Json::String(_))) | ("end_turn" | "tool_use", Some(Json::Null)) => {}
        _ => return Err(DeterministicShapeFailureV1::Malformed.into()),
    }

    let mut tool_calls = Vec::new();
    let mut text_fragments = Vec::new();
    for block in content {
        let block = required_object(block)?;
        match required_string(block, "type")? {
            "text" => {
                text_fragments.push(required_string(block, "text")?);
            }
            "thinking" => {
                required_string(block, "thinking")?;
                match block.get("signature") {
                    Some(Json::String(_)) | Some(Json::Null) => {}
                    _ => return Err(DeterministicShapeFailureV1::Malformed.into()),
                }
            }
            "redacted_thinking" => {
                required_string(block, "data")?;
            }
            "tool_use" => {
                let id = required_nonempty_string(block, "id")?;
                let name = required_nonempty_string(block, "name")?;
                let arguments = block
                    .get("input")
                    .ok_or(DeterministicShapeFailureV1::Malformed)?;
                admit_json(arguments, limits).map_err(StrictValidationFailureV1::Operational)?;
                let arguments = arguments.clone();
                tool_calls.push(ResponseToolCall {
                    id: id.to_string(),
                    name: name.to_string(),
                    arguments,
                });
            }
            "mcp_tool_use" | "server_tool_use" => {
                return Err(DeterministicShapeFailureV1::ToolContract.into());
            }
            _ => return Err(DeterministicShapeFailureV1::Malformed.into()),
        }
    }
    if (matches!(stop_reason, "end_turn" | "stop_sequence") && !tool_calls.is_empty())
        || (stop_reason == "tool_use" && tool_calls.is_empty())
    {
        return Err(DeterministicShapeFailureV1::Malformed.into());
    }
    Ok(FamilyShapeOutcomeV1::Complete(StrictFamilyResponseV1 {
        tool_calls,
        text_fragments,
    }))
}

fn parse_tool_arguments(
    arguments: &str,
    limits: ValidationLimitsV1,
) -> Result<Json, StrictValidationFailureV1> {
    admit_json_text(arguments, limits).map_err(StrictValidationFailureV1::Operational)?;
    let arguments = serde_json::from_str(arguments).map_err(|_| {
        StrictValidationFailureV1::Deterministic(DeterministicShapeFailureV1::ToolContract)
    })?;
    admit_json(&arguments, limits).map_err(StrictValidationFailureV1::Operational)?;
    Ok(arguments)
}

fn required_object(value: &Json) -> Result<&Map<String, Json>, DeterministicShapeFailureV1> {
    value
        .as_object()
        .ok_or(DeterministicShapeFailureV1::Malformed)
}

fn required_object_field<'a>(
    object: &'a Map<String, Json>,
    key: &str,
) -> Result<&'a Map<String, Json>, DeterministicShapeFailureV1> {
    object
        .get(key)
        .and_then(Json::as_object)
        .ok_or(DeterministicShapeFailureV1::Malformed)
}

fn required_array<'a>(
    object: &'a Map<String, Json>,
    key: &str,
) -> Result<&'a [Json], DeterministicShapeFailureV1> {
    object
        .get(key)
        .and_then(Json::as_array)
        .map(Vec::as_slice)
        .ok_or(DeterministicShapeFailureV1::Malformed)
}

fn optional_array<'a>(
    object: &'a Map<String, Json>,
    key: &str,
) -> Result<Option<&'a [Json]>, DeterministicShapeFailureV1> {
    object
        .get(key)
        .map(|value| {
            value
                .as_array()
                .map(Vec::as_slice)
                .ok_or(DeterministicShapeFailureV1::Malformed)
        })
        .transpose()
}

fn required_string<'a>(
    object: &'a Map<String, Json>,
    key: &str,
) -> Result<&'a str, DeterministicShapeFailureV1> {
    object
        .get(key)
        .and_then(Json::as_str)
        .ok_or(DeterministicShapeFailureV1::Malformed)
}

fn required_nonempty_string<'a>(
    object: &'a Map<String, Json>,
    key: &str,
) -> Result<&'a str, DeterministicShapeFailureV1> {
    required_string(object, key).and_then(|value| {
        if value.is_empty() {
            Err(DeterministicShapeFailureV1::Malformed)
        } else {
            Ok(value)
        }
    })
}

fn require_exact_string(
    object: &Map<String, Json>,
    key: &str,
    expected: &str,
) -> Result<(), DeterministicShapeFailureV1> {
    if required_string(object, key)? == expected {
        Ok(())
    } else {
        Err(DeterministicShapeFailureV1::Malformed)
    }
}

fn required_u64(object: &Map<String, Json>, key: &str) -> Result<u64, DeterministicShapeFailureV1> {
    object
        .get(key)
        .and_then(Json::as_u64)
        .ok_or(DeterministicShapeFailureV1::Malformed)
}

fn required_number(
    object: &Map<String, Json>,
    key: &str,
) -> Result<(), DeterministicShapeFailureV1> {
    if object.get(key).is_some_and(Json::is_number) {
        Ok(())
    } else {
        Err(DeterministicShapeFailureV1::Malformed)
    }
}

#[cfg(test)]
impl CandidateResponseContractsV1 {
    pub(crate) fn empty_for_test() -> Self {
        Self {
            structured_output: None,
            tools: BTreeMap::new(),
            tool_choice: None,
            max_tool_calls: None,
            parallel_tool_calls: None,
            max_validation_bytes: usize::MAX,
            tool_contract_fingerprint: None,
            response_contract_fingerprint: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use nemo_relay::api::llm::{LlmApiFamily, LlmRequest};
    use nemo_relay::codec::openai_chat::OpenAIChatCodec;
    use nemo_relay::codec::request::AnnotatedLlmRequest;
    use nemo_relay::codec::traits::LlmCodec;
    use serde_json::{Map, Value as Json, json};

    use super::{
        CandidateResponseContractsV1, CandidateValidationOperationalFailureV1,
        CandidateValidationOutcomeV1, JudgeResponseExtractionFailureV1, StructuredOutputContractV1,
        compile_candidate_response_contracts, extract_judge_assistant_text,
        validate_candidate_response,
    };
    use crate::eligibility::IneligibilityReason;
    use crate::judge::{
        DeterministicHardFailureV1, InvalidJudgeOutputMarkerV1, JudgeAttemptOutcomeV1,
        JudgeBinaryLabelV1, JudgeEvaluationSourceV1, JudgeLabelV1, validate_pairwise_judge_output,
    };
    use crate::projection::sanitize_annotated_request;
    use crate::trajectory::RouterResponseProjectionV1;

    const DRAFT_7: &str = "http://json-schema.org/draft-07/schema#";
    const DRAFT_2020_12: &str = "https://json-schema.org/draft/2020-12/schema";
    const MAX_PROJECTION_BYTES: usize = 64 * 1024;

    fn decode_request(content: Json) -> AnnotatedLlmRequest {
        OpenAIChatCodec
            .decode(&LlmRequest {
                headers: Map::new(),
                content,
            })
            .unwrap()
    }

    fn structured_contracts() -> CandidateResponseContractsV1 {
        compile_candidate_response_contracts(
            &decode_request(json!({
                "model": "anchor",
                "messages": [{"role": "user", "content": "task"}],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": true,
                        "schema": {
                            "$schema": DRAFT_2020_12,
                            "type": "object",
                            "properties": {"answer": {"type": "string"}},
                            "required": ["answer"],
                            "additionalProperties": false
                        }
                    }
                }
            })),
            MAX_PROJECTION_BYTES,
        )
        .unwrap()
    }

    fn tool_contracts(choice: &str) -> CandidateResponseContractsV1 {
        compile_candidate_response_contracts(
            &decode_request(json!({
                "model": "anchor",
                "messages": [{"role": "user", "content": "task"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "parameters": {
                            "$schema": DRAFT_7,
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "api_key": {"type": "string"}
                            },
                            "required": ["query"],
                            "additionalProperties": false
                        }
                    }
                }],
                "tool_choice": choice
            })),
            MAX_PROJECTION_BYTES,
        )
        .unwrap()
    }

    fn text_response(family: LlmApiFamily, text: &str) -> Json {
        match family {
            LlmApiFamily::OpenAIChatCompletions => json!({
                "id": "chatcmpl-candidate",
                "object": "chat.completion",
                "created": 1,
                "model": "candidate",
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            }),
            LlmApiFamily::OpenAIResponses => json!({
                "id": "resp-candidate",
                "object": "response",
                "created_at": 1.0,
                "status": "completed",
                "model": "candidate",
                "output": [{
                    "id": "msg-candidate",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": text, "annotations": []}]
                }],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
            }),
            LlmApiFamily::AnthropicMessages => json!({
                "id": "msg-candidate",
                "type": "message",
                "role": "assistant",
                "model": "candidate",
                "content": [{"type": "text", "text": text}],
                "stop_reason": "end_turn",
                "stop_sequence": null,
                "usage": {"input_tokens": 3, "output_tokens": 2}
            }),
        }
    }

    fn tool_response(family: LlmApiFamily, name: &str, arguments: Json) -> Json {
        let arguments_text = serde_json::to_string(&arguments).unwrap();
        match family {
            LlmApiFamily::OpenAIChatCompletions => json!({
                "id": "chatcmpl-tool",
                "object": "chat.completion",
                "created": 1,
                "model": "candidate",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": null,
                        "tool_calls": [{
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments_text}
                        }]
                    },
                    "finish_reason": "tool_calls"
                }],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
            }),
            LlmApiFamily::OpenAIResponses => json!({
                "id": "resp-tool",
                "object": "response",
                "created_at": 1.0,
                "status": "completed",
                "model": "candidate",
                "output": [{
                    "id": "fc-1",
                    "type": "function_call",
                    "call_id": "call-1",
                    "name": name,
                    "arguments": arguments_text,
                    "status": "completed"
                }],
                "usage": {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5}
            }),
            LlmApiFamily::AnthropicMessages => json!({
                "id": "msg-tool",
                "type": "message",
                "role": "assistant",
                "model": "candidate",
                "content": [{
                    "type": "tool_use",
                    "id": "call-1",
                    "name": name,
                    "input": arguments
                }],
                "stop_reason": "tool_use",
                "stop_sequence": null,
                "usage": {"input_tokens": 3, "output_tokens": 2}
            }),
        }
    }

    fn all_families() -> [LlmApiFamily; 3] {
        [
            LlmApiFamily::OpenAIChatCompletions,
            LlmApiFamily::OpenAIResponses,
            LlmApiFamily::AnthropicMessages,
        ]
    }

    fn expect_valid(outcome: CandidateValidationOutcomeV1) -> RouterResponseProjectionV1 {
        match outcome {
            CandidateValidationOutcomeV1::Valid { response } => response,
            CandidateValidationOutcomeV1::DeterministicFailure { .. }
            | CandidateValidationOutcomeV1::OperationalFailure(_) => {
                panic!("expected valid candidate response")
            }
        }
    }

    fn expect_deterministic(
        outcome: CandidateValidationOutcomeV1,
        expected: DeterministicHardFailureV1,
    ) -> Option<RouterResponseProjectionV1> {
        match outcome {
            CandidateValidationOutcomeV1::DeterministicFailure {
                hard_failure,
                evaluation,
                response,
            } => {
                assert_eq!(hard_failure, expected);
                assert_eq!(
                    evaluation.source,
                    JudgeEvaluationSourceV1::DeterministicValidator
                );
                assert_eq!(evaluation.label, JudgeLabelV1::Fail);
                assert_eq!(evaluation.binary_label, Some(JudgeBinaryLabelV1::Fail));
                assert_eq!(evaluation.hard_failures.len(), 1);
                assert!(evaluation.rationale.is_none());
                response.map(|response| *response)
            }
            CandidateValidationOutcomeV1::Valid { .. }
            | CandidateValidationOutcomeV1::OperationalFailure(_) => {
                panic!("expected deterministic candidate failure")
            }
        }
    }

    fn expect_operational(
        outcome: CandidateValidationOutcomeV1,
        expected: CandidateValidationOperationalFailureV1,
    ) {
        match outcome {
            CandidateValidationOutcomeV1::OperationalFailure(actual) => {
                assert_eq!(actual, expected);
            }
            CandidateValidationOutcomeV1::Valid { .. }
            | CandidateValidationOutcomeV1::DeterministicFailure { .. } => {
                panic!("expected operational candidate failure")
            }
        }
    }

    #[test]
    fn judge_text_extraction_is_strict_and_exact_for_every_family() {
        let raw = r#"{"response_equivalence":0.9,"trajectory_equivalence":0.8,"judge_confidence":0.95,"hard_failures":[],"rationale":"equivalent"}"#;
        for family in all_families() {
            assert_eq!(
                extract_judge_assistant_text(
                    family,
                    &text_response(family, raw),
                    MAX_PROJECTION_BYTES,
                )
                .unwrap(),
                raw
            );
        }

        let mut multiple_choices = text_response(LlmApiFamily::OpenAIChatCompletions, raw);
        let second = multiple_choices["choices"][0].clone();
        multiple_choices["choices"]
            .as_array_mut()
            .unwrap()
            .push(second);
        assert_eq!(
            extract_judge_assistant_text(
                LlmApiFamily::OpenAIChatCompletions,
                &multiple_choices,
                MAX_PROJECTION_BYTES,
            ),
            Err(JudgeResponseExtractionFailureV1::Unreadable)
        );
    }

    #[test]
    fn judge_text_extraction_uses_codec_defined_multi_block_joining() {
        let mut responses = text_response(LlmApiFamily::OpenAIResponses, "unused");
        responses["output"][0]["content"] = json!([
            {"type": "output_text", "text": "first", "annotations": []},
            {"type": "output_text", "text": "second", "annotations": []}
        ]);
        assert_eq!(
            extract_judge_assistant_text(
                LlmApiFamily::OpenAIResponses,
                &responses,
                MAX_PROJECTION_BYTES,
            )
            .unwrap(),
            "first\nsecond"
        );

        let mut anthropic = text_response(LlmApiFamily::AnthropicMessages, "unused");
        anthropic["content"] = json!([
            {"type": "text", "text": "first"},
            {"type": "thinking", "thinking": "private", "signature": "signature"},
            {"type": "text", "text": "second"}
        ]);
        assert_eq!(
            extract_judge_assistant_text(
                LlmApiFamily::AnthropicMessages,
                &anthropic,
                MAX_PROJECTION_BYTES,
            )
            .unwrap(),
            "first\nsecond"
        );

        anthropic["content"][1]["signature"] = Json::Null;
        assert_eq!(
            extract_judge_assistant_text(
                LlmApiFamily::AnthropicMessages,
                &anthropic,
                MAX_PROJECTION_BYTES,
            )
            .unwrap(),
            "first\nsecond"
        );
    }

    #[test]
    fn judge_text_extraction_rejects_tools_and_missing_assistant_text() {
        for family in all_families() {
            assert_eq!(
                extract_judge_assistant_text(
                    family,
                    &tool_response(family, "lookup", json!({"query": "safe"})),
                    MAX_PROJECTION_BYTES,
                ),
                Err(JudgeResponseExtractionFailureV1::Unreadable)
            );
        }

        let mut reasoning_only = text_response(LlmApiFamily::OpenAIResponses, "unused");
        reasoning_only["output"] = json!([{
            "type": "reasoning",
            "id": "reasoning-1",
            "summary": [],
            "status": "completed"
        }]);
        assert_eq!(
            extract_judge_assistant_text(
                LlmApiFamily::OpenAIResponses,
                &reasoning_only,
                MAX_PROJECTION_BYTES,
            ),
            Err(JudgeResponseExtractionFailureV1::Unreadable)
        );

        let mut thinking_only = text_response(LlmApiFamily::AnthropicMessages, "unused");
        thinking_only["content"] = json!([{
            "type": "thinking",
            "thinking": "private",
            "signature": "signature"
        }]);
        assert_eq!(
            extract_judge_assistant_text(
                LlmApiFamily::AnthropicMessages,
                &thinking_only,
                MAX_PROJECTION_BYTES,
            ),
            Err(JudgeResponseExtractionFailureV1::Unreadable)
        );
    }

    #[test]
    fn judge_text_extraction_preserves_operational_terminal_classes() {
        let mut chat = text_response(LlmApiFamily::OpenAIChatCompletions, "partial");
        chat["choices"][0]["finish_reason"] = json!("length");
        assert_eq!(
            extract_judge_assistant_text(
                LlmApiFamily::OpenAIChatCompletions,
                &chat,
                MAX_PROJECTION_BYTES,
            ),
            Err(JudgeResponseExtractionFailureV1::Truncated)
        );
        chat["choices"][0]["finish_reason"] = json!("future_terminal");
        assert_eq!(
            extract_judge_assistant_text(
                LlmApiFamily::OpenAIChatCompletions,
                &chat,
                MAX_PROJECTION_BYTES,
            ),
            Err(JudgeResponseExtractionFailureV1::AmbiguousDecode)
        );

        let mut responses = text_response(LlmApiFamily::OpenAIResponses, "partial");
        responses["status"] = json!("cancelled");
        assert_eq!(
            extract_judge_assistant_text(
                LlmApiFamily::OpenAIResponses,
                &responses,
                MAX_PROJECTION_BYTES,
            ),
            Err(JudgeResponseExtractionFailureV1::ProviderCanceled)
        );
        responses["status"] = json!("failed");
        assert_eq!(
            extract_judge_assistant_text(
                LlmApiFamily::OpenAIResponses,
                &responses,
                MAX_PROJECTION_BYTES,
            ),
            Err(JudgeResponseExtractionFailureV1::ProviderFailure)
        );

        let mut anthropic = text_response(LlmApiFamily::AnthropicMessages, "partial");
        anthropic["stop_reason"] = json!("max_tokens");
        assert_eq!(
            extract_judge_assistant_text(
                LlmApiFamily::AnthropicMessages,
                &anthropic,
                MAX_PROJECTION_BYTES,
            ),
            Err(JudgeResponseExtractionFailureV1::Truncated)
        );
    }

    #[test]
    fn judge_text_extraction_bounds_outer_response_before_shape_and_codec_work() {
        for family in all_families() {
            assert_eq!(
                extract_judge_assistant_text(
                    family,
                    &text_response(family, &"x".repeat(8_192)),
                    128,
                ),
                Err(JudgeResponseExtractionFailureV1::EvidenceBound)
            );
        }

        let mut deep = Json::Null;
        for _ in 0..=64 {
            deep = json!({"next": deep});
        }
        let mut response = text_response(LlmApiFamily::OpenAIChatCompletions, "answer");
        response["deep_extra"] = deep;
        assert_eq!(
            extract_judge_assistant_text(
                LlmApiFamily::OpenAIChatCompletions,
                &response,
                MAX_PROJECTION_BYTES,
            ),
            Err(JudgeResponseExtractionFailureV1::EvidenceBound)
        );
    }

    #[test]
    fn judge_output_bound_remains_authoritative_after_text_extraction() {
        let max_rationale_bytes = 1;
        let exact = "x".repeat(max_rationale_bytes + 8 * 1024);
        let oversized = format!("{exact}x");

        for family in all_families() {
            let extracted_exact = extract_judge_assistant_text(
                family,
                &text_response(family, &exact),
                MAX_PROJECTION_BYTES,
            )
            .unwrap();
            assert_eq!(extracted_exact, exact);
            let JudgeAttemptOutcomeV1::Invalid(exact_invalid) =
                validate_pairwise_judge_output(&extracted_exact, max_rationale_bytes)
            else {
                panic!("exact-bound output must remain readable invalid Judge output");
            };
            assert_eq!(
                exact_invalid.output().retained_output(),
                Some(exact.as_str())
            );

            let extracted_oversized = extract_judge_assistant_text(
                family,
                &text_response(family, &oversized),
                MAX_PROJECTION_BYTES,
            )
            .unwrap();
            assert_eq!(extracted_oversized, oversized);
            let JudgeAttemptOutcomeV1::Invalid(oversized_invalid) =
                validate_pairwise_judge_output(&extracted_oversized, max_rationale_bytes)
            else {
                panic!("oversized output must remain readable invalid Judge output");
            };
            assert_eq!(
                oversized_invalid.output().marker(),
                Some(InvalidJudgeOutputMarkerV1::Oversized)
            );
            assert_eq!(oversized_invalid.output().retained_output(), None);
        }
    }

    #[test]
    fn supported_drafts_compile_and_retain_full_unsanitized_schemas() {
        for draft in [DRAFT_7, DRAFT_2020_12] {
            let schema = json!({
                "$schema": draft,
                "$defs": {"secret_definition": {"type": "string"}},
                "type": "object",
                "properties": {"api_key": {"$ref": "#/$defs/secret_definition"}}
            });
            let contracts = compile_candidate_response_contracts(
                &decode_request(json!({
                    "model": "anchor",
                    "messages": [],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {"name": "result", "schema": schema}
                    }
                })),
                MAX_PROJECTION_BYTES,
            )
            .unwrap();
            let Some(StructuredOutputContractV1::JsonSchema(compiled)) =
                contracts.structured_output.as_ref()
            else {
                panic!("expected compiled response schema");
            };
            assert_eq!(compiled.full_schema, schema);
        }
    }

    #[test]
    fn authoritative_contract_fingerprints_distinguish_pruned_schema_content() {
        let request = |tool_secret: &str, response_secret: &str| {
            decode_request(json!({
                "model": "anchor",
                "messages": [{"role": "user", "content": "task"}],
                "tools": [{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "Look up a record",
                        "parameters": {
                            "$schema": DRAFT_2020_12,
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"},
                                "api_key": {"type": "string", "const": tool_secret}
                            },
                            "required": ["query"],
                            "additionalProperties": false
                        }
                    }
                }],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "answer",
                        "strict": true,
                        "schema": {
                            "$schema": DRAFT_2020_12,
                            "type": "object",
                            "properties": {
                                "answer": {"type": "string"},
                                "api_key": {"type": "string", "const": response_secret}
                            },
                            "required": ["answer"],
                            "additionalProperties": false
                        }
                    }
                }
            }))
        };
        let baseline_request = request("tool-secret-a", "response-secret-a");
        let changed_tool_request = request("tool-secret-b", "response-secret-a");
        let changed_response_request = request("tool-secret-a", "response-secret-b");

        let baseline_safe = sanitize_annotated_request(&baseline_request).unwrap();
        let changed_tool_safe = sanitize_annotated_request(&changed_tool_request).unwrap();
        let changed_response_safe = sanitize_annotated_request(&changed_response_request).unwrap();
        assert_eq!(baseline_safe.tools, changed_tool_safe.tools);
        assert_eq!(
            baseline_safe.response_format,
            changed_response_safe.response_format
        );

        let baseline =
            compile_candidate_response_contracts(&baseline_request, MAX_PROJECTION_BYTES).unwrap();
        let changed_tool =
            compile_candidate_response_contracts(&changed_tool_request, MAX_PROJECTION_BYTES)
                .unwrap();
        let changed_response =
            compile_candidate_response_contracts(&changed_response_request, MAX_PROJECTION_BYTES)
                .unwrap();

        assert_ne!(
            baseline.tool_contract_fingerprint(),
            changed_tool.tool_contract_fingerprint()
        );
        assert_eq!(
            baseline.response_contract_fingerprint(),
            changed_tool.response_contract_fingerprint()
        );
        assert_eq!(
            baseline.tool_contract_fingerprint(),
            changed_response.tool_contract_fingerprint()
        );
        assert_ne!(
            baseline.response_contract_fingerprint(),
            changed_response.response_contract_fingerprint()
        );
        assert_eq!(baseline.tool_contract_fingerprint().unwrap().len(), 64);
        assert_eq!(baseline.response_contract_fingerprint().unwrap().len(), 64);
    }

    #[test]
    fn unsupported_invalid_and_nonlocal_contracts_are_rejected_offline() {
        let cases = [
            (
                json!({"$schema": "https://json-schema.org/draft/2019-09/schema", "type": "object"}),
                IneligibilityReason::UnsupportedContractSchema,
            ),
            (
                json!({"$schema": DRAFT_2020_12, "type": "object", "required": "answer"}),
                IneligibilityReason::InvalidContractSchema,
            ),
            (
                json!({"$schema": DRAFT_2020_12, "$ref": "https://example.invalid/schema"}),
                IneligibilityReason::NonLocalContractReference,
            ),
            (
                json!({"$schema": DRAFT_2020_12, "$ref": "relative.json#/$defs/value"}),
                IneligibilityReason::NonLocalContractReference,
            ),
            (
                json!({
                    "$schema": DRAFT_2020_12,
                    "properties": {"answer": {"$ref": "relative.json#/$defs/value"}}
                }),
                IneligibilityReason::NonLocalContractReference,
            ),
        ];
        for (schema, expected) in cases {
            let request = decode_request(json!({
                "model": "anchor",
                "messages": [],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "result", "schema": schema}
                }
            }));
            assert_eq!(
                compile_candidate_response_contracts(&request, MAX_PROJECTION_BYTES).err(),
                Some(expected)
            );
        }
    }

    #[test]
    fn reference_scanning_uses_each_effective_schema_dialect() {
        let accepted_ignored_keywords = [
            json!({
                "$schema": DRAFT_2020_12,
                "type": "array",
                "additionalItems": {"$ref": "https://example.invalid/ignored-2020"}
            }),
            json!({
                "$schema": DRAFT_7,
                "type": "array",
                "prefixItems": [{"$ref": "https://example.invalid/ignored-draft7"}]
            }),
        ];
        for schema in accepted_ignored_keywords {
            let request = decode_request(json!({
                "model": "anchor",
                "messages": [],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "result", "schema": schema}
                }
            }));
            assert!(compile_candidate_response_contracts(&request, MAX_PROJECTION_BYTES).is_ok());
        }

        let active_nonlocal_keywords = [
            json!({
                "$schema": DRAFT_7,
                "type": "array",
                "additionalItems": {"$ref": "https://example.invalid/active-draft7"}
            }),
            json!({
                "$schema": DRAFT_2020_12,
                "type": "array",
                "prefixItems": [{"$ref": "https://example.invalid/active-2020"}]
            }),
        ];
        for schema in active_nonlocal_keywords {
            let request = decode_request(json!({
                "model": "anchor",
                "messages": [],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "result", "schema": schema}
                }
            }));
            assert_eq!(
                compile_candidate_response_contracts(&request, MAX_PROJECTION_BYTES).err(),
                Some(IneligibilityReason::NonLocalContractReference)
            );
        }
    }

    #[test]
    fn nested_schema_resources_accept_only_the_two_supported_dialects() {
        let schema = json!({
            "$schema": DRAFT_2020_12,
            "$defs": {
                "legacy": {
                    "$schema": "https://json-schema.org/draft/2019-09/schema",
                    "$recursiveRef": "#"
                }
            },
            "type": "object"
        });
        let request = decode_request(json!({
            "model": "anchor",
            "messages": [],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema}
            }
        }));
        assert_eq!(
            compile_candidate_response_contracts(&request, MAX_PROJECTION_BYTES).err(),
            Some(IneligibilityReason::UnsupportedContractSchema)
        );
    }

    #[test]
    fn draft7_local_ref_targets_enforce_dialects_without_activating_unknown_keywords() {
        for schema in [
            json!({
                "$schema": DRAFT_7,
                "$ref": "#/definitions/result",
                "definitions": {
                    "result": {
                        "$schema": "https://json-schema.org/draft/2019-09/schema",
                        "$recursiveRef": "#"
                    }
                }
            }),
            json!({
                "$schema": DRAFT_7,
                "$ref": "#/%74arget",
                "target": {
                    "$schema": "https://json-schema.org/draft/2019-09/schema",
                    "$recursiveRef": "#"
                }
            }),
        ] {
            let request = decode_request(json!({
                "model": "anchor",
                "messages": [],
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "result", "schema": schema}
                }
            }));
            assert_eq!(
                compile_candidate_response_contracts(&request, MAX_PROJECTION_BYTES).err(),
                Some(IneligibilityReason::UnsupportedContractSchema)
            );
        }

        let ignored_cross_draft_keyword = json!({
            "$schema": DRAFT_7,
            "$ref": "#/definitions/result",
            "definitions": {"result": {"type": "object"}},
            "prefixItems": [{
                "$schema": "https://json-schema.org/draft/2019-09/schema",
                "$ref": "https://example.invalid/ignored-draft7-keyword"
            }]
        });
        let request = decode_request(json!({
            "model": "anchor",
            "messages": [],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": ignored_cross_draft_keyword}
            }
        }));
        assert!(compile_candidate_response_contracts(&request, MAX_PROJECTION_BYTES).is_ok());

        let recursive = json!({
            "$schema": DRAFT_7,
            "$ref": "#/definitions/node",
            "definitions": {"node": {"$ref": "#/definitions/node"}}
        });
        let draft = super::schema_draft(&recursive).unwrap();
        assert!(
            super::reject_nonlocal_references(&recursive, draft, super::MAX_VALIDATION_NODES)
                .is_ok()
        );
    }

    #[test]
    fn schema_compile_budget_is_aggregate_and_identical_tools_share_one_arc() {
        let schema = |description: String| {
            json!({
                "$schema": DRAFT_2020_12,
                "description": description,
                "type": "object"
            })
        };
        let unique_tools = (0..3)
            .map(|index| {
                json!({
                    "type": "function",
                    "function": {
                        "name": format!("tool_{index}"),
                        "parameters": schema(format!("{index}{}", "x".repeat(850)))
                    }
                })
            })
            .collect::<Vec<_>>();
        let unique_request = decode_request(json!({
            "model": "anchor",
            "messages": [],
            "tools": unique_tools
        }));
        assert_eq!(
            compile_candidate_response_contracts(&unique_request, 2 * 1024).err(),
            Some(IneligibilityReason::InvalidToolContract)
        );

        let shared_schema = schema("x".repeat(850));
        let duplicate_tools = (0..3)
            .map(|index| {
                json!({
                    "type": "function",
                    "function": {
                        "name": format!("tool_{index}"),
                        "parameters": shared_schema.clone()
                    }
                })
            })
            .collect::<Vec<_>>();
        let duplicate_request = decode_request(json!({
            "model": "anchor",
            "messages": [],
            "tools": duplicate_tools
        }));
        assert_eq!(
            compile_candidate_response_contracts(&duplicate_request, 2 * 1024).err(),
            Some(IneligibilityReason::InvalidToolContract),
            "cache hits must still consume the aggregate occurrence-work budget"
        );

        let contracts = compile_candidate_response_contracts(&duplicate_request, 8 * 1024).unwrap();
        let first = contracts.tools["tool_0"].argument_schema.as_ref().unwrap();
        let second = contracts.tools["tool_1"].argument_schema.as_ref().unwrap();
        let third = contracts.tools["tool_2"].argument_schema.as_ref().unwrap();
        assert!(Arc::ptr_eq(first, second));
        assert!(Arc::ptr_eq(second, third));

        let schema_less_tools = (0..8)
            .map(|index| {
                json!({
                    "type": "function",
                    "function": {"name": format!("schema_less_{index}")}
                })
            })
            .collect::<Vec<_>>();
        let schema_less_request = decode_request(json!({
            "model": "anchor",
            "messages": [],
            "tools": schema_less_tools
        }));
        assert_eq!(
            compile_candidate_response_contracts(&schema_less_request, 256).err(),
            Some(IneligibilityReason::InvalidToolContract),
            "tool-map growth must consume the aggregate occurrence budget"
        );
    }

    #[test]
    fn reference_like_instance_literals_do_not_trigger_schema_retrieval_policy() {
        let literal = json!({"$ref": "https://example.invalid/literal-not-a-schema"});
        let schema = json!({
            "$schema": DRAFT_2020_12,
            "type": "object",
            "properties": {"value": {"const": literal}}
        });
        let request = decode_request(json!({
            "model": "anchor",
            "messages": [],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema}
            }
        }));
        let contracts =
            compile_candidate_response_contracts(&request, MAX_PROJECTION_BYTES).unwrap();
        let Some(StructuredOutputContractV1::JsonSchema(compiled)) =
            contracts.structured_output.as_ref()
        else {
            panic!("expected compiled response schema");
        };
        assert!(compiled.validator.is_valid(&json!({"value": literal})));
    }

    #[test]
    fn three_family_structured_response_goldens_succeed() {
        let contracts = structured_contracts();
        for family in all_families() {
            let projection = expect_valid(validate_candidate_response(
                family,
                &text_response(family, r#"{"answer":"ok"}"#),
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ));
            assert_eq!(projection.model.as_deref(), Some("candidate"));
            assert_eq!(projection.semantic_response_fingerprint.len(), 64);
        }
    }

    #[test]
    fn three_family_malformed_shapes_are_deterministic() {
        let contracts = CandidateResponseContractsV1::empty_for_test();
        for family in all_families() {
            let mut response = text_response(family, "answer");
            let root = response.as_object_mut().unwrap();
            match family {
                LlmApiFamily::OpenAIChatCompletions => {
                    root.insert("object".into(), json!("response"));
                }
                LlmApiFamily::OpenAIResponses => {
                    root.insert("object".into(), json!("chat.completion"));
                }
                LlmApiFamily::AnthropicMessages => {
                    root.insert("type".into(), json!("error"));
                }
            }
            let projection = expect_deterministic(
                validate_candidate_response(
                    family,
                    &response,
                    &contracts,
                    MAX_PROJECTION_BYTES,
                    false,
                ),
                DeterministicHardFailureV1::MalformedCandidate,
            );
            assert!(projection.is_some());
        }
    }

    #[test]
    fn three_family_response_schema_failures_are_deterministic() {
        let contracts = structured_contracts();
        for family in all_families() {
            let projection = expect_deterministic(
                validate_candidate_response(
                    family,
                    &text_response(family, r#"{"answer":7}"#),
                    &contracts,
                    MAX_PROJECTION_BYTES,
                    true,
                ),
                DeterministicHardFailureV1::ResponseSchema,
            );
            assert!(projection.is_some());
        }
    }

    #[test]
    fn three_family_tool_contract_failures_are_deterministic() {
        let contracts = tool_contracts("required");
        for family in all_families() {
            for response in [
                tool_response(family, "undeclared", json!({"query": "safe"})),
                tool_response(family, "lookup", json!({"query": 7})),
            ] {
                let projection = expect_deterministic(
                    validate_candidate_response(
                        family,
                        &response,
                        &contracts,
                        MAX_PROJECTION_BYTES,
                        false,
                    ),
                    DeterministicHardFailureV1::ToolContract,
                );
                assert!(projection.is_some());
            }
        }
    }

    #[test]
    fn required_tool_presence_and_none_choice_are_enforced() {
        let required = tool_contracts("required");
        let none = tool_contracts("none");
        for family in all_families() {
            expect_deterministic(
                validate_candidate_response(
                    family,
                    &text_response(family, "no tool"),
                    &required,
                    MAX_PROJECTION_BYTES,
                    false,
                ),
                DeterministicHardFailureV1::ToolContract,
            );
            expect_deterministic(
                validate_candidate_response(
                    family,
                    &tool_response(family, "lookup", json!({"query": "safe"})),
                    &none,
                    MAX_PROJECTION_BYTES,
                    false,
                ),
                DeterministicHardFailureV1::ToolContract,
            );
        }
    }

    #[test]
    fn three_family_safe_projections_remove_provider_and_argument_credentials() {
        let contracts = tool_contracts("auto");
        for family in all_families() {
            let mut response = tool_response(
                family,
                "lookup",
                json!({
                    "query": "safe",
                    "api_key": "nvapi-012345678901234567890123456789"
                }),
            );
            response.as_object_mut().unwrap().insert(
                "authorization".into(),
                json!("Bearer must-not-persist-0123456789"),
            );
            let projection = expect_valid(validate_candidate_response(
                family,
                &response,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ));
            let serialized = serde_json::to_string(&projection).unwrap();
            assert!(serialized.contains("safe"));
            assert!(!serialized.contains("api_key"));
            assert!(!serialized.contains("nvapi-"));
            assert!(!serialized.contains("authorization"));
            assert!(!serialized.contains("must-not-persist"));
        }
    }

    #[test]
    fn projection_bounds_are_operational_and_unlabeled() {
        let contracts = CandidateResponseContractsV1::empty_for_test();
        for family in all_families() {
            let outcome = validate_candidate_response(
                family,
                &text_response(family, &"x".repeat(8_192)),
                &contracts,
                128,
                false,
            );
            assert!(matches!(
                outcome,
                CandidateValidationOutcomeV1::OperationalFailure(
                    CandidateValidationOperationalFailureV1::ProjectionBound
                )
            ));
        }
    }

    #[test]
    fn malformed_tool_argument_json_is_a_tool_failure() {
        let contracts = tool_contracts("required");
        let mut response = tool_response(
            LlmApiFamily::OpenAIChatCompletions,
            "lookup",
            json!({"query": "safe"}),
        );
        response["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"] =
            json!(r#"{"api_key":"must-not-persist""#);
        let projection = expect_deterministic(
            validate_candidate_response(
                LlmApiFamily::OpenAIChatCompletions,
                &response,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            DeterministicHardFailureV1::ToolContract,
        );
        assert!(
            projection.is_none(),
            "undecodable tool arguments cannot become safe persisted evidence"
        );

        response["object"] = json!("wrong-family");
        let projection = expect_deterministic(
            validate_candidate_response(
                LlmApiFamily::OpenAIChatCompletions,
                &response,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            DeterministicHardFailureV1::MalformedCandidate,
        );
        assert!(
            projection.is_none(),
            "an earlier family error cannot expose undecodable tool arguments"
        );
    }

    #[test]
    fn non_success_terminal_signals_are_typed_and_unlabeled() {
        let contracts = CandidateResponseContractsV1::empty_for_test();

        let mut chat = text_response(LlmApiFamily::OpenAIChatCompletions, "partial");
        chat["choices"][0]["finish_reason"] = json!("length");
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::OpenAIChatCompletions,
                &chat,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::Truncated,
        );
        chat["choices"][0]["finish_reason"] = json!("content_filter");
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::OpenAIChatCompletions,
                &chat,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::ProviderFailure,
        );
        chat["choices"][0]["finish_reason"] = json!("future_terminal");
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::OpenAIChatCompletions,
                &chat,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::AmbiguousTerminal,
        );
        chat["choices"][0]["finish_reason"] = json!("stop");
        chat["choices"][0]["message"]["refusal"] = json!("policy refusal");
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::OpenAIChatCompletions,
                &chat,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::ProviderFailure,
        );

        let mut responses = text_response(LlmApiFamily::OpenAIResponses, "partial");
        responses["status"] = json!("incomplete");
        responses["incomplete_details"] = json!({"reason": "max_output_tokens"});
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::OpenAIResponses,
                &responses,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::Truncated,
        );
        responses["incomplete_details"] = json!({"reason": "content_filter"});
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::OpenAIResponses,
                &responses,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::ProviderFailure,
        );
        responses["incomplete_details"] = Json::Null;
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::OpenAIResponses,
                &responses,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::AmbiguousTerminal,
        );
        for (status, expected) in [
            (
                "failed",
                CandidateValidationOperationalFailureV1::ProviderFailure,
            ),
            (
                "cancelled",
                CandidateValidationOperationalFailureV1::Cancelled,
            ),
            (
                "future_terminal",
                CandidateValidationOperationalFailureV1::AmbiguousTerminal,
            ),
        ] {
            responses["status"] = json!(status);
            expect_operational(
                validate_candidate_response(
                    LlmApiFamily::OpenAIResponses,
                    &responses,
                    &contracts,
                    MAX_PROJECTION_BYTES,
                    false,
                ),
                expected,
            );
        }

        let mut anthropic = text_response(LlmApiFamily::AnthropicMessages, "partial");
        anthropic["stop_reason"] = json!("max_tokens");
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::AnthropicMessages,
                &anthropic,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::Truncated,
        );
        anthropic["stop_reason"] = Json::Null;
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::AnthropicMessages,
                &anthropic,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::AmbiguousTerminal,
        );
        anthropic["stop_reason"] = json!("future_terminal");
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::AnthropicMessages,
                &anthropic,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::AmbiguousTerminal,
        );
    }

    #[test]
    fn responses_child_terminal_signals_remain_operational() {
        let contracts = CandidateResponseContractsV1::empty_for_test();
        for (status, expected) in [
            (
                "incomplete",
                CandidateValidationOperationalFailureV1::Truncated,
            ),
            (
                "failed",
                CandidateValidationOperationalFailureV1::ProviderFailure,
            ),
            (
                "cancelled",
                CandidateValidationOperationalFailureV1::Cancelled,
            ),
            (
                "future_terminal",
                CandidateValidationOperationalFailureV1::AmbiguousTerminal,
            ),
        ] {
            let mut message = text_response(LlmApiFamily::OpenAIResponses, "answer");
            message["output"][0]["status"] = json!(status);
            expect_operational(
                validate_candidate_response(
                    LlmApiFamily::OpenAIResponses,
                    &message,
                    &contracts,
                    MAX_PROJECTION_BYTES,
                    false,
                ),
                expected,
            );

            let mut function = tool_response(
                LlmApiFamily::OpenAIResponses,
                "lookup",
                json!({"query": "safe"}),
            );
            function["output"][0]["status"] = json!(status);
            expect_operational(
                validate_candidate_response(
                    LlmApiFamily::OpenAIResponses,
                    &function,
                    &contracts,
                    MAX_PROJECTION_BYTES,
                    false,
                ),
                expected,
            );
        }
    }

    #[test]
    fn complete_terminal_content_and_tool_correlations_are_enforced() {
        let contracts = CandidateResponseContractsV1::empty_for_test();

        let mut chat = tool_response(
            LlmApiFamily::OpenAIChatCompletions,
            "lookup",
            json!({"query": "safe"}),
        );
        chat["choices"][0]["finish_reason"] = json!("stop");
        expect_deterministic(
            validate_candidate_response(
                LlmApiFamily::OpenAIChatCompletions,
                &chat,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            DeterministicHardFailureV1::MalformedCandidate,
        );

        let mut responses = text_response(LlmApiFamily::OpenAIResponses, "answer");
        responses["incomplete_details"] = json!({"reason": "max_output_tokens"});
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::OpenAIResponses,
                &responses,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::AmbiguousTerminal,
        );

        let mut anthropic = tool_response(
            LlmApiFamily::AnthropicMessages,
            "lookup",
            json!({"query": "safe"}),
        );
        anthropic["stop_reason"] = json!("end_turn");
        expect_deterministic(
            validate_candidate_response(
                LlmApiFamily::AnthropicMessages,
                &anthropic,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            DeterministicHardFailureV1::MalformedCandidate,
        );
    }

    #[test]
    fn structured_output_requires_one_raw_text_fragment() {
        let contracts = structured_contracts();

        let mut responses = text_response(LlmApiFamily::OpenAIResponses, r#"{"answer":"ok"}"#);
        responses["output"][0]["content"] = json!([
            {"type": "output_text", "text": "{\"answer\":", "annotations": []},
            {"type": "output_text", "text": "\"ok\"}", "annotations": []}
        ]);
        expect_deterministic(
            validate_candidate_response(
                LlmApiFamily::OpenAIResponses,
                &responses,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            DeterministicHardFailureV1::ResponseSchema,
        );

        let mut anthropic = text_response(LlmApiFamily::AnthropicMessages, r#"{"answer":"ok"}"#);
        anthropic["content"] = json!([
            {"type": "text", "text": "{\"answer\":"},
            {"type": "text", "text": "\"ok\"}"}
        ]);
        expect_deterministic(
            validate_candidate_response(
                LlmApiFamily::AnthropicMessages,
                &anthropic,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            DeterministicHardFailureV1::ResponseSchema,
        );

        let mut chat = text_response(LlmApiFamily::OpenAIChatCompletions, r#"{"answer":"ok"}"#);
        let duplicate = chat["choices"][0].clone();
        chat["choices"].as_array_mut().unwrap().push(duplicate);
        expect_deterministic(
            validate_candidate_response(
                LlmApiFamily::OpenAIChatCompletions,
                &chat,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            DeterministicHardFailureV1::MalformedCandidate,
        );
    }

    #[test]
    fn raw_response_bounds_precede_shape_decode_and_contract_work() {
        let contracts = structured_contracts();
        let huge = "x".repeat(2 * 1024 * 1024);
        for family in all_families() {
            let mut response = text_response(family, &huge);
            let root = response.as_object_mut().unwrap();
            match family {
                LlmApiFamily::OpenAIChatCompletions => {
                    root.insert("object".into(), json!("wrong"));
                }
                LlmApiFamily::OpenAIResponses => {
                    root.insert("object".into(), json!("wrong"));
                }
                LlmApiFamily::AnthropicMessages => {
                    root.insert("type".into(), json!("wrong"));
                }
            }
            expect_operational(
                validate_candidate_response(
                    family,
                    &response,
                    &contracts,
                    MAX_PROJECTION_BYTES,
                    false,
                ),
                CandidateValidationOperationalFailureV1::ProjectionBound,
            );
        }

        let mut deep = Json::Null;
        for _ in 0..=64 {
            deep = json!({"next": deep});
        }
        let mut response = text_response(LlmApiFamily::OpenAIChatCompletions, "answer");
        response["deep_extra"] = deep;
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::OpenAIChatCompletions,
                &response,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::ProjectionBound,
        );

        let nested_json_text = format!("{}0{}", "[".repeat(65), "]".repeat(65));
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::OpenAIChatCompletions,
                &text_response(LlmApiFamily::OpenAIChatCompletions, &nested_json_text),
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::ProjectionBound,
        );
    }

    #[test]
    fn schema_size_and_validation_work_are_bounded_before_validation() {
        let oversized_schema = json!({
            "$schema": DRAFT_2020_12,
            "$ref": "https://example.invalid/must-not-be-resolved",
            "description": "x".repeat(128 * 1024)
        });
        let oversized_request = decode_request(json!({
            "model": "anchor",
            "messages": [],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": oversized_schema}
            }
        }));
        assert_eq!(
            compile_candidate_response_contracts(&oversized_request, 4 * 1024).err(),
            Some(IneligibilityReason::InvalidContractSchema)
        );

        let properties = (0..150)
            .map(|index| (format!("field_{index}"), json!({"type": "string"})))
            .collect::<Map<String, Json>>();
        let schema = json!({
            "$schema": DRAFT_2020_12,
            "type": "object",
            "properties": properties
        });
        let request = decode_request(json!({
            "model": "anchor",
            "messages": [],
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "result", "schema": schema}
            }
        }));
        let contracts = compile_candidate_response_contracts(&request, 16 * 1024).unwrap();
        let instance = (0..150)
            .map(|index| (format!("field_{index}"), json!("value")))
            .collect::<Map<String, Json>>();
        let text = serde_json::to_string(&instance).unwrap();
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::OpenAIChatCompletions,
                &text_response(LlmApiFamily::OpenAIChatCompletions, &text),
                &contracts,
                16 * 1024,
                false,
            ),
            CandidateValidationOperationalFailureV1::ProjectionBound,
        );
    }

    #[test]
    fn anthropic_usage_overflow_is_caught_before_core_decode() {
        let contracts = CandidateResponseContractsV1::empty_for_test();
        let mut response = text_response(LlmApiFamily::AnthropicMessages, "answer");
        response["usage"]["input_tokens"] = json!(u64::MAX);
        response["usage"]["output_tokens"] = json!(1);
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            validate_candidate_response(
                LlmApiFamily::AnthropicMessages,
                &response,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            )
        }))
        .expect("validator must not unwind on provider usage overflow");
        expect_operational(
            outcome,
            CandidateValidationOperationalFailureV1::AmbiguousTerminal,
        );
    }

    #[test]
    fn impossible_tool_constraints_are_rejected_but_loose_maxima_are_valid() {
        let mut request = decode_request(json!({
            "model": "anchor",
            "messages": [],
            "tools": [{
                "type": "function",
                "function": {"name": "lookup"}
            }],
            "tool_choice": "required"
        }));
        request.max_tool_calls = Some(0);
        assert_eq!(
            compile_candidate_response_contracts(&request, MAX_PROJECTION_BYTES).err(),
            Some(IneligibilityReason::InvalidToolContract)
        );

        request.max_tool_calls = Some(2);
        request.parallel_tool_calls = Some(false);
        assert!(
            compile_candidate_response_contracts(&request, MAX_PROJECTION_BYTES).is_ok(),
            "a loose maximum remains satisfiable by one non-parallel call"
        );
    }

    #[test]
    fn credential_shaped_projection_controls_never_return_valid_evidence() {
        let contracts = CandidateResponseContractsV1::empty_for_test();
        let secret = "nvapi-abcdefghijklmnopqrstuvwxyz0123456789";
        for family in all_families() {
            for key in ["id", "model"] {
                for value in [
                    secret.to_string(),
                    format!("{key}:{secret}"),
                    format!("chatcmpl-{secret}"),
                    format!("org/{secret}"),
                    "call-Bearer:abcdefghijklmnopqrstuv".to_string(),
                    "chatcmpl-token:not-safe".to_string(),
                    "gateway:eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature123:v1".to_string(),
                    "gateway-Basic:QUJDREVGR0hJSktMTU5PUA==:v1".to_string(),
                    "call|token:not-safe".to_string(),
                    "call@Bearer:abcdefghijklmnopqrstuv".to_string(),
                    "call\u{2022}Basic:QUJDREVGR0hJSktMTU5PUA==:v1".to_string(),
                    "call-Bearer : abcdefghijklmnopqrstuv".to_string(),
                    "call-Bearer:abcdefghijklmnopqrstuv suffix".to_string(),
                    "call-Bearer:abcdefghijklmnopqrstuv\u{2022}v1".to_string(),
                    "call-Basic:\"QUJDREVGR0hJSktMTU5PUA==\"".to_string(),
                    "call-Bearer:\u{2022}abcdefghijklmnopqrstuv".to_string(),
                ] {
                    let mut response = text_response(family, "answer");
                    response[key] = json!(value);
                    expect_operational(
                        validate_candidate_response(
                            family,
                            &response,
                            &contracts,
                            MAX_PROJECTION_BYTES,
                            false,
                        ),
                        CandidateValidationOperationalFailureV1::UnsafeProjection,
                    );
                }
            }

            let mut response = text_response(family, "answer");
            match family {
                LlmApiFamily::OpenAIChatCompletions => {
                    response["choices"][0]["finish_reason"] = json!(secret);
                }
                LlmApiFamily::OpenAIResponses => response["status"] = json!(secret),
                LlmApiFamily::AnthropicMessages => response["stop_reason"] = json!(secret),
            }
            expect_operational(
                validate_candidate_response(
                    family,
                    &response,
                    &contracts,
                    MAX_PROJECTION_BYTES,
                    false,
                ),
                CandidateValidationOperationalFailureV1::AmbiguousTerminal,
            );

            for (field, value) in [
                ("id", "call-Basic:\"QUJDREVGR0hJSktMTU5PUA==\"".to_string()),
                (
                    "name",
                    "call-Bearer:\u{2022}abcdefghijklmnopqrstuv".to_string(),
                ),
            ] {
                let mut response = tool_response(family, "lookup", json!({"query": "safe"}));
                match (family, field) {
                    (LlmApiFamily::OpenAIChatCompletions, "id") => {
                        response["choices"][0]["message"]["tool_calls"][0]["id"] = json!(value);
                    }
                    (LlmApiFamily::OpenAIChatCompletions, "name") => {
                        response["choices"][0]["message"]["tool_calls"][0]["function"]["name"] =
                            json!(value);
                    }
                    (LlmApiFamily::OpenAIResponses, "id") => {
                        response["output"][0]["call_id"] = json!(value);
                    }
                    (LlmApiFamily::OpenAIResponses, "name") => {
                        response["output"][0]["name"] = json!(value);
                    }
                    (LlmApiFamily::AnthropicMessages, "id") => {
                        response["content"][0]["id"] = json!(value);
                    }
                    (LlmApiFamily::AnthropicMessages, "name") => {
                        response["content"][0]["name"] = json!(value);
                    }
                    _ => unreachable!(),
                }
                expect_operational(
                    validate_candidate_response(
                        family,
                        &response,
                        &tool_contracts("required"),
                        MAX_PROJECTION_BYTES,
                        false,
                    ),
                    CandidateValidationOperationalFailureV1::UnsafeProjection,
                );
            }
        }

        let mut malformed = text_response(LlmApiFamily::OpenAIChatCompletions, "answer");
        malformed["id"] = json!("call-Basic:\"QUJDREVGR0hJSktMTU5PUA==\"");
        malformed["object"] = json!("wrong-family");
        expect_operational(
            validate_candidate_response(
                LlmApiFamily::OpenAIChatCompletions,
                &malformed,
                &contracts,
                MAX_PROJECTION_BYTES,
                false,
            ),
            CandidateValidationOperationalFailureV1::UnsafeProjection,
        );
    }
}
