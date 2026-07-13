// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Conservative call and candidate preflight for shadow routing.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use nemo_relay::api::llm::{
    LlmApiFamily, LlmAttributes, LlmCallRole, LlmExecutionContextSnapshot, LlmRequest,
};
use nemo_relay::api::runtime::{
    LLM_REPLAY_CONTRACT_VERSION, LlmReplayCapability, LlmReplayTransport,
};

use crate::adapter::{FamilyAdapter, RouterRequestEnvelope};
use crate::config::{CandidateCapabilities, CandidateConfig, PoolConfig};
use crate::eligibility::IneligibilityReason;
use crate::projection::{
    RouterRequestProjectionV1, RouterRoutingContextProjectionV1, canonical_request_budget,
    output_token_limit, project_request, project_routing_context,
    validate_raw_request_resource_bounds,
};
use crate::response_validator::{
    CandidateResponseContractsV1, compile_candidate_response_contracts,
};
use crate::trajectory::PersistedCandidateFactV1;

const MAX_TRANSPORT_IDENTITY_BYTES: usize = 256;
const MIN_KNOWN_SECRET_BYTES: usize = 20;
const KNOWN_SECRET_PREFIXES: [&str; 13] = [
    "nvapi-",
    "sk-",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "github_pat_",
    "hf_",
    "xoxb-",
    "xoxp-",
    "xoxa-",
    "aiza",
];

type CandidatePreflightResult = Result<
    (
        Vec<EligibleCandidate>,
        Vec<PersistedCandidateFactV1>,
        Vec<CandidateRejection>,
    ),
    IneligibilityReason,
>;

/// One candidate that passed every independent preflight check.
pub(crate) struct EligibleCandidate {
    /// Validated candidate configuration.
    pub(crate) config: CandidateConfig,
    /// Lossless request clone whose only wire change is the model.
    pub(crate) request: LlmRequest,
    /// Full unsanitized response and tool contracts, compiled once and memory-only.
    // Read by the Spec 05 scheduler once candidate replay lands.
    #[allow(dead_code)]
    pub(crate) response_contracts: Arc<CandidateResponseContractsV1>,
}

impl EligibleCandidate {
    pub(crate) fn new(
        config: CandidateConfig,
        request: LlmRequest,
        response_contracts: Arc<CandidateResponseContractsV1>,
    ) -> Self {
        Self {
            config,
            request,
            response_contracts,
        }
    }
}

/// Stable, non-secret rejection information for one candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CandidateRejection {
    /// Stable configured candidate identifier.
    pub(crate) candidate_id: String,
    /// Stable reason that excluded this candidate.
    pub(crate) reason: IneligibilityReason,
}

/// Fully preflighted request state retained only in memory.
pub(crate) struct PreflightOutcome {
    /// Exact replay capability validated for this call.
    pub(crate) replay_capability: LlmReplayCapability,
    /// Lossless provider-family envelope.
    // Retained for the Spec 04 coordinator command without re-decoding the request.
    #[allow(dead_code)]
    pub(crate) envelope: RouterRequestEnvelope,
    /// Sanitized, serializable semantic request projection.
    // Retained for Spec 04 evidence construction after the anchor completes.
    #[allow(dead_code)]
    pub(crate) request_projection: RouterRequestProjectionV1,
    /// Sanitized, policy-only routing projection.
    // Retained for Spec 04 evidence construction after the anchor completes.
    #[allow(dead_code)]
    pub(crate) routing_projection: RouterRoutingContextProjectionV1,
    /// Eligible candidates in normative cheapest-first order, after truncation.
    pub(crate) candidates: Vec<EligibleCandidate>,
    /// Number of eligible candidates before truncation.
    pub(crate) eligible_candidate_count: usize,
    /// Complete safe candidate facts before Shadow-only truncation.
    #[allow(dead_code)]
    // Spec 07 Recommend consumes the full set while Shadow keeps its prefix.
    pub(crate) eligible_candidate_facts: Vec<PersistedCandidateFactV1>,
    /// Independently rejected candidates and their stable reasons.
    // Retained for Spec 04 bounded eligibility diagnostics.
    #[allow(dead_code)]
    pub(crate) rejected_candidates: Vec<CandidateRejection>,
}

/// Validate an eligible call and independently preflight its pool candidates.
pub(crate) fn preflight(
    context: &LlmExecutionContextSnapshot,
    request: &LlmRequest,
    replay: Option<&Arc<dyn LlmReplayTransport>>,
    pool: &PoolConfig,
    adapter: &FamilyAdapter,
) -> Result<PreflightOutcome, IneligibilityReason> {
    validate_call_context(context, pool)?;
    let replay_capability = validate_replay(context, replay)?;
    validate_raw_request_resource_bounds(request)?;

    let envelope = adapter.decode(context.api_family, request)?;
    if envelope
        .normalized_request
        .model
        .as_deref()
        .is_none_or(str::is_empty)
    {
        return Err(IneligibilityReason::MissingModel);
    }
    if !pool
        .anchor_models
        .iter()
        .any(|model| envelope.normalized_request.model.as_deref() == Some(model.as_str()))
    {
        return Err(IneligibilityReason::NoMatchingPool);
    }
    if envelope.normalized_request.stream == Some(true) {
        return Err(IneligibilityReason::Streaming);
    }
    if context.api_family == LlmApiFamily::OpenAIResponses {
        if envelope.normalized_request.previous_response_id.is_some() {
            return Err(IneligibilityReason::ResponsesContinuation);
        }
        if envelope.normalized_request.store == Some(true) {
            return Err(IneligibilityReason::ResponsesStore);
        }
    }

    let request_projection = project_request(&envelope, &pool.canonicalizer)?;
    let routing_projection = project_routing_context(context, &pool.selector, &pool.canonicalizer)?;
    let response_contracts = Arc::new(compile_candidate_response_contracts(
        &envelope.normalized_request,
        pool.lookahead.max_bytes_per_window,
    )?);
    let (mut candidates, mut eligible_candidate_facts, rejected_candidates) = preflight_candidates(
        pool,
        adapter,
        &envelope,
        &request_projection,
        &response_contracts,
    )?;
    if eligible_candidate_facts.is_empty() {
        return Err(IneligibilityReason::NoCandidates);
    }
    let eligible_candidate_count = eligible_candidate_facts.len();
    candidates.shrink_to_fit();
    eligible_candidate_facts.shrink_to_fit();

    Ok(PreflightOutcome {
        replay_capability,
        envelope,
        request_projection,
        routing_projection,
        candidates,
        eligible_candidate_count,
        eligible_candidate_facts,
        rejected_candidates,
    })
}

fn validate_call_context(
    context: &LlmExecutionContextSnapshot,
    pool: &PoolConfig,
) -> Result<(), IneligibilityReason> {
    if context.call_role != LlmCallRole::Primary {
        return Err(IneligibilityReason::NonPrimary);
    }
    if context.attributes.contains(LlmAttributes::STREAMING) {
        return Err(IneligibilityReason::Streaming);
    }
    if context.attributes.contains(LlmAttributes::STATEFUL) {
        return Err(IneligibilityReason::Stateful);
    }
    if context.api_family != pool.api_family {
        return Err(IneligibilityReason::ReplayFamilyMismatch);
    }
    Ok(())
}

fn validate_replay(
    context: &LlmExecutionContextSnapshot,
    replay: Option<&Arc<dyn LlmReplayTransport>>,
) -> Result<LlmReplayCapability, IneligibilityReason> {
    let replay = replay.ok_or(IneligibilityReason::MissingReplay)?;
    let capability = catch_unwind(AssertUnwindSafe(|| replay.capability().clone()))
        .map_err(|_| IneligibilityReason::RuntimeFailure)?;
    validate_replay_capability(context, &capability)?;
    Ok(capability)
}

fn validate_replay_capability(
    context: &LlmExecutionContextSnapshot,
    capability: &LlmReplayCapability,
) -> Result<(), IneligibilityReason> {
    if capability.contract_version != LLM_REPLAY_CONTRACT_VERSION {
        return Err(IneligibilityReason::ReplayVersionMismatch);
    }
    if capability.api_family != context.api_family {
        return Err(IneligibilityReason::ReplayFamilyMismatch);
    }
    if !valid_transport_identity(&capability.transport_identity) {
        return Err(IneligibilityReason::ReplayIdentityInvalid);
    }
    Ok(())
}

fn preflight_candidates(
    pool: &PoolConfig,
    adapter: &FamilyAdapter,
    envelope: &RouterRequestEnvelope,
    projection: &RouterRequestProjectionV1,
    response_contracts: &Arc<CandidateResponseContractsV1>,
) -> CandidatePreflightResult {
    let mut eligible = Vec::new();
    let mut facts = Vec::new();
    let mut rejected = Vec::new();
    let mut ordered = pool.candidates.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| {
        left.cost_rank
            .cmp(&right.cost_rank)
            .then_with(|| left.id.cmp(&right.id))
    });
    for candidate in ordered {
        match preflight_candidate(candidate, adapter, envelope, projection) {
            Ok(request) => {
                let config = runtime_candidate_config(candidate);
                facts.push(PersistedCandidateFactV1::from_config(
                    &config,
                    response_contracts,
                    projection,
                )?);
                if eligible.len() < pool.max_candidates_per_sample {
                    eligible.push(EligibleCandidate::new(
                        config,
                        request,
                        Arc::clone(response_contracts),
                    ));
                }
            }
            Err(reason) => rejected.push(CandidateRejection {
                candidate_id: candidate.id.clone(),
                reason,
            }),
        }
    }
    Ok((eligible, facts, rejected))
}

fn runtime_candidate_config(candidate: &CandidateConfig) -> CandidateConfig {
    CandidateConfig {
        id: candidate.id.clone(),
        model: candidate.model.clone(),
        model_revision: candidate.model_revision.clone(),
        cost_rank: candidate.cost_rank,
        max_context_tokens: candidate.max_context_tokens,
        capabilities: CandidateCapabilities {
            tools: candidate.capabilities.tools,
            multimodal_input: candidate.capabilities.multimodal_input,
            structured_output: candidate.capabilities.structured_output,
            reasoning_controls: candidate.capabilities.reasoning_controls,
            unknown_fields: Default::default(),
        },
        unknown_fields: Default::default(),
    }
}

fn preflight_candidate(
    candidate: &CandidateConfig,
    adapter: &FamilyAdapter,
    envelope: &RouterRequestEnvelope,
    projection: &RouterRequestProjectionV1,
) -> Result<LlmRequest, IneligibilityReason> {
    if !capabilities_cover(&candidate.capabilities, &projection.required_capabilities) {
        return Err(IneligibilityReason::CandidateCapability);
    }

    if let Some(max_context_tokens) = candidate.max_context_tokens {
        let output_tokens = output_token_limit(&projection.normalized_request)
            .ok_or(IneligibilityReason::ContextLimitUnprovable)?;
        let input_budget = canonical_request_budget(&projection.normalized_request)?;
        let total_budget = input_budget
            .checked_add(output_tokens)
            .ok_or(IneligibilityReason::ContextLimitExceeded)?;
        if total_budget > max_context_tokens {
            return Err(IneligibilityReason::ContextLimitExceeded);
        }
    }

    adapter.with_model(envelope, &candidate.model)
}

fn capabilities_cover(capabilities: &CandidateCapabilities, required: &[String]) -> bool {
    required.iter().all(|required| match required.as_str() {
        "tools" => capabilities.tools,
        "multimodal_input" => capabilities.multimodal_input,
        "structured_output" => capabilities.structured_output,
        "reasoning_controls" => capabilities.reasoning_controls,
        _ => false,
    })
}

fn valid_transport_identity(identity: &str) -> bool {
    if identity.is_empty()
        || identity.len() > MAX_TRANSPORT_IDENTITY_BYTES
        || identity.bytes().any(|byte| !byte.is_ascii_graphic())
    {
        return false;
    }

    !contains_sensitive_control_material(identity)
}

pub(crate) fn has_sensitive_value_shape(value: &str) -> bool {
    let lowercase = value.to_ascii_lowercase();
    const CREDENTIAL_MARKERS: [&str; 10] = [
        "authorization:",
        "bearer ",
        "basic ",
        "api_key=",
        "api-key=",
        "apikey=",
        "password=",
        "passwd=",
        "secret=",
        "token=",
    ];
    let private_key =
        lowercase.starts_with("-----begin ") && lowercase.contains("private key-----");
    CREDENTIAL_MARKERS
        .iter()
        .any(|marker| lowercase.contains(marker))
        || has_known_secret_prefix(&lowercase)
        || looks_like_jwt(value)
        || looks_like_aws_access_key_id(value)
        || private_key
        || credential_bearing_url(&lowercase)
}

fn has_known_secret_prefix(value: &str) -> bool {
    value.len() >= MIN_KNOWN_SECRET_BYTES
        && KNOWN_SECRET_PREFIXES
            .iter()
            .any(|prefix| value.starts_with(prefix))
}

/// Reject credential material embedded in structural identifiers and control fields.
pub(crate) fn contains_sensitive_control_material(value: &str) -> bool {
    if has_sensitive_value_shape(value) {
        return true;
    }

    let lowercase = value.to_ascii_lowercase();
    let known_prefix = KNOWN_SECRET_PREFIXES.iter().any(|prefix| {
        lowercase
            .match_indices(prefix)
            .any(|(offset, _)| lowercase.len().saturating_sub(offset) >= MIN_KNOWN_SECRET_BYTES)
    });
    let aws_access_key = value
        .as_bytes()
        .windows(20)
        .any(|candidate| std::str::from_utf8(candidate).is_ok_and(looks_like_aws_access_key_id));
    let jwt = contains_embedded_jwt(value);
    let credential_url = ["https://", "http://"].iter().any(|scheme| {
        lowercase
            .match_indices(scheme)
            .any(|(offset, _)| credential_bearing_url(&lowercase[offset..]))
    });
    let private_key = lowercase.contains("-----begin ") && lowercase.contains("private key-----");

    known_prefix
        || aws_access_key
        || jwt
        || credential_url
        || private_key
        || contains_wrapped_credential_assignment(value, &lowercase)
}

fn contains_wrapped_credential_assignment(value: &str, lowercase: &str) -> bool {
    value.char_indices().any(|(offset, separator)| {
        if !matches!(separator, ':' | '=') {
            return false;
        }
        let label = lowercase[..offset].trim_end();
        let material = value[offset + separator.len_utf8()..].trim_start();
        if material.is_empty() {
            return false;
        }

        const ASSIGNMENT_LABELS: [&str; 8] = [
            "authorization",
            "api_key",
            "api-key",
            "apikey",
            "password",
            "passwd",
            "secret",
            "token",
        ];
        ASSIGNMENT_LABELS
            .iter()
            .any(|suffix| control_label_has_suffix(label, suffix))
            || control_label_has_suffix(label, "bearer")
            || control_label_has_suffix(label, "basic")
    })
}

fn contains_embedded_jwt(value: &str) -> bool {
    value.match_indices("eyJ").any(|(offset, _)| {
        let candidate = &value[offset..];
        let candidate_bytes = candidate
            .bytes()
            .take_while(|byte| is_base64url_byte(*byte) || *byte == b'.')
            .count();
        looks_like_jwt(&candidate[..candidate_bytes])
    })
}

fn control_label_has_suffix(label: &str, suffix: &str) -> bool {
    label.ends_with(suffix)
}

fn looks_like_jwt(value: &str) -> bool {
    let mut segments = value.split('.');
    let Some(header) = segments.next() else {
        return false;
    };
    let Some(payload) = segments.next() else {
        return false;
    };
    let Some(signature) = segments.next() else {
        return false;
    };
    segments.next().is_none()
        && header.starts_with("eyJ")
        && [header, payload, signature]
            .iter()
            .all(|segment| !segment.is_empty() && segment.bytes().all(is_base64url_byte))
}

fn is_base64url_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_')
}

fn looks_like_aws_access_key_id(value: &str) -> bool {
    const PREFIXES: [&str; 8] = [
        "AKIA", "ASIA", "AIDA", "AROA", "AIPA", "ANPA", "ANVA", "ASCA",
    ];
    value.len() == 20
        && PREFIXES.iter().any(|prefix| value.starts_with(prefix))
        && value
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit())
}

fn credential_bearing_url(value: &str) -> bool {
    let Some(authority) = value
        .strip_prefix("https://")
        .or_else(|| value.strip_prefix("http://"))
        .and_then(|remainder| remainder.split('/').next())
    else {
        return false;
    };
    authority.contains('@')
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use nemo_relay::api::llm::{
        LlmApiFamily, LlmAttributes, LlmCallRole, LlmExecutionContextSnapshot, LlmRequest,
        LlmTrajectoryScopeSnapshot,
    };
    use nemo_relay::api::runtime::{
        LLM_REPLAY_CONTRACT_VERSION, LlmReplayCall, LlmReplayCapability, LlmReplayTransport,
    };
    use nemo_relay::api::scope::ScopeType;
    use serde_json::json;
    use uuid::Uuid;

    use super::{capabilities_cover, preflight, preflight_candidate};
    use crate::adapter::FamilyAdapter;
    use crate::config::{
        CandidateCapabilities, CandidateConfig, CanonicalizerConfig, ConcurrencyConfig,
        JUDGE_PROMPT_VERSION_V1, JUDGE_RUBRIC_VERSION_V1, JudgeConfig, LookaheadConfig, PoolConfig,
        PoolSelectorConfig,
    };
    use crate::eligibility::IneligibilityReason;
    use crate::projection::{canonical_request_budget, output_token_limit, project_request};

    struct TestReplay {
        capability: LlmReplayCapability,
    }

    impl LlmReplayTransport for TestReplay {
        fn capability(&self) -> &LlmReplayCapability {
            &self.capability
        }

        fn start(&self, _request: LlmRequest) -> nemo_relay::error::Result<LlmReplayCall> {
            unreachable!("preflight must not start replay")
        }
    }

    fn context(family: LlmApiFamily) -> LlmExecutionContextSnapshot {
        let root_uuid = Uuid::nil();
        LlmExecutionContextSnapshot {
            call_uuid: Uuid::from_u128(1),
            root_uuid,
            parent_uuid: root_uuid,
            trajectory_owner_uuid: root_uuid,
            trajectory_owner_path: vec![LlmTrajectoryScopeSnapshot {
                uuid: root_uuid,
                name: "root".into(),
                scope_type: ScopeType::Agent,
            }],
            api_family: family,
            call_role: LlmCallRole::Primary,
            attributes: LlmAttributes::empty(),
            tenant_id: None,
            agent_id: None,
            sanitized_metadata: BTreeMap::new(),
        }
    }

    fn replay(family: LlmApiFamily, version: u32, identity: &str) -> Arc<dyn LlmReplayTransport> {
        Arc::new(TestReplay {
            capability: LlmReplayCapability {
                contract_version: version,
                api_family: family,
                transport_identity: identity.into(),
            },
        })
    }

    fn chat_request(with_tools: bool, output_tokens: Option<u64>) -> LlmRequest {
        let mut content = json!({
            "model": "anchor",
            "messages": [{"role": "user", "content": "hello"}],
        });
        if let Some(output_tokens) = output_tokens {
            content
                .as_object_mut()
                .unwrap()
                .insert("max_tokens".into(), output_tokens.into());
        }
        if with_tools {
            content.as_object_mut().unwrap().insert(
                "tools".into(),
                json!([{
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "lookup a value",
                        "parameters": {
                            "$schema": "https://json-schema.org/draft/2020-12/schema",
                            "type": "object",
                            "properties": {"key": {"type": "string"}},
                            "required": ["key"]
                        }
                    }
                }]),
            );
        }
        LlmRequest {
            headers: serde_json::Map::new(),
            content,
        }
    }

    fn responses_request(field: &str) -> LlmRequest {
        let mut content = json!({
            "model": "anchor",
            "input": "hello",
            "max_output_tokens": 16,
        });
        match field {
            "previous_response_id" => {
                content
                    .as_object_mut()
                    .unwrap()
                    .insert(field.into(), "response-1".into());
            }
            "store" => {
                content
                    .as_object_mut()
                    .unwrap()
                    .insert(field.into(), true.into());
            }
            _ => unreachable!(),
        }
        LlmRequest {
            headers: serde_json::Map::new(),
            content,
        }
    }

    fn candidate(
        id: &str,
        model: &str,
        cost_rank: u32,
        capabilities: CandidateCapabilities,
    ) -> CandidateConfig {
        CandidateConfig {
            id: id.into(),
            model: model.into(),
            model_revision: "revision-1".into(),
            cost_rank,
            max_context_tokens: None,
            capabilities,
            unknown_fields: BTreeMap::new(),
        }
    }

    fn pool(family: LlmApiFamily, candidates: Vec<CandidateConfig>) -> PoolConfig {
        PoolConfig {
            id: "pool".into(),
            api_family: family,
            anchor_models: vec!["anchor".into()],
            anchor_revision: "revision-1".into(),
            sampling_probability: 1.0,
            max_candidates_per_sample: candidates.len().max(1),
            selector: PoolSelectorConfig::default(),
            lookahead: LookaheadConfig::default(),
            concurrency: ConcurrencyConfig {
                shadow: 1,
                judge: 1,
                max_pending: 1,
                unknown_fields: BTreeMap::new(),
            },
            candidates,
            canonicalizer: CanonicalizerConfig::default(),
            judge: judge_config(),
            learning: None,
            outcome: BTreeMap::new(),
            unknown_fields: BTreeMap::new(),
        }
    }

    fn judge_config() -> JudgeConfig {
        JudgeConfig {
            version: 1,
            model: "judge-model".into(),
            model_revision: "judge-r1".into(),
            prompt_version: JUDGE_PROMPT_VERSION_V1.into(),
            rubric_version: JUDGE_RUBRIC_VERSION_V1.into(),
            output_schema_version: 1,
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

    fn all_capabilities() -> CandidateCapabilities {
        CandidateCapabilities {
            tools: true,
            multimodal_input: true,
            structured_output: true,
            reasoning_controls: true,
            unknown_fields: BTreeMap::new(),
        }
    }

    #[test]
    fn replay_and_call_guards_are_stable() {
        let adapter = FamilyAdapter;
        let request = chat_request(false, Some(16));
        let pool = pool(
            LlmApiFamily::OpenAIChatCompletions,
            vec![candidate("candidate", "small", 1, all_capabilities())],
        );
        let valid = replay(
            LlmApiFamily::OpenAIChatCompletions,
            LLM_REPLAY_CONTRACT_VERSION,
            "gateway-A",
        );

        let mut call = context(LlmApiFamily::OpenAIChatCompletions);
        call.call_role = LlmCallRole::Shadow;
        assert_eq!(
            preflight(&call, &request, Some(&valid), &pool, &adapter)
                .err()
                .unwrap(),
            IneligibilityReason::NonPrimary
        );
        call.call_role = LlmCallRole::Primary;
        call.attributes = LlmAttributes::STREAMING;
        assert_eq!(
            preflight(&call, &request, Some(&valid), &pool, &adapter)
                .err()
                .unwrap(),
            IneligibilityReason::Streaming
        );
        call.attributes = LlmAttributes::STATEFUL;
        assert_eq!(
            preflight(&call, &request, Some(&valid), &pool, &adapter)
                .err()
                .unwrap(),
            IneligibilityReason::Stateful
        );
        call.attributes = LlmAttributes::empty();
        assert_eq!(
            preflight(&call, &request, None, &pool, &adapter)
                .err()
                .unwrap(),
            IneligibilityReason::MissingReplay
        );

        let wrong_version = replay(LlmApiFamily::OpenAIChatCompletions, 99, "gateway-A");
        assert_eq!(
            preflight(&call, &request, Some(&wrong_version), &pool, &adapter)
                .err()
                .unwrap(),
            IneligibilityReason::ReplayVersionMismatch
        );
        let wrong_family = replay(
            LlmApiFamily::AnthropicMessages,
            LLM_REPLAY_CONTRACT_VERSION,
            "gateway-A",
        );
        assert_eq!(
            preflight(&call, &request, Some(&wrong_family), &pool, &adapter)
                .err()
                .unwrap(),
            IneligibilityReason::ReplayFamilyMismatch
        );
        let secret_identity = replay(
            LlmApiFamily::OpenAIChatCompletions,
            LLM_REPLAY_CONTRACT_VERSION,
            "nvapi-abcdefghijklmnopqrstuvwxyz0123456789",
        );
        assert_eq!(
            preflight(&call, &request, Some(&secret_identity), &pool, &adapter)
                .err()
                .unwrap(),
            IneligibilityReason::ReplayIdentityInvalid
        );
        for identity in [
            "gateway:nvapi-abcdefghijklmnopqrstuvwxyz0123456789",
            "Bearer:abcdefghijklmnopqrstuv",
            "gateway-Bearer:abcdefghijklmnopqrstuv",
            "gateway-token:not-safe",
            "gateway:eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.signature123:v1",
            "gateway-Basic:QUJDREVGR0hJSktMTU5PUA==:v1",
            "gateway|token:not-safe",
            "gateway@Bearer:abcdefghijklmnopqrstuv",
            "gateway#Basic:QUJDREVGR0hJSktMTU5PUA==:v1",
            "call-Bearer:abcdefghijklmnopqrstuv suffix",
            "call-Bearer:abcdefghijklmnopqrstuv\u{2022}v1",
            "call-Basic:\"QUJDREVGR0hJSktMTU5PUA==\"",
            "call-Bearer:\u{2022}abcdefghijklmnopqrstuv",
        ] {
            let wrapped_secret_identity = replay(
                LlmApiFamily::OpenAIChatCompletions,
                LLM_REPLAY_CONTRACT_VERSION,
                identity,
            );
            assert_eq!(
                preflight(
                    &call,
                    &request,
                    Some(&wrapped_secret_identity),
                    &pool,
                    &adapter
                )
                .err()
                .unwrap(),
                IneligibilityReason::ReplayIdentityInvalid
            );
        }

        let mut unmatched = request.clone();
        unmatched.content["model"] = json!("different-anchor");
        assert_eq!(
            preflight(&call, &unmatched, Some(&valid), &pool, &adapter)
                .err()
                .unwrap(),
            IneligibilityReason::NoMatchingPool
        );
    }

    #[test]
    fn capability_truth_table_is_exact() {
        let names = [
            "tools",
            "multimodal_input",
            "structured_output",
            "reasoning_controls",
        ];
        for required_mask in 0_u8..16 {
            let required = names
                .iter()
                .enumerate()
                .filter(|(index, _)| required_mask & (1 << index) != 0)
                .map(|(_, name)| (*name).to_string())
                .collect::<Vec<_>>();
            for capability_mask in 0_u8..16 {
                let capabilities = CandidateCapabilities {
                    tools: capability_mask & 1 != 0,
                    multimodal_input: capability_mask & 2 != 0,
                    structured_output: capability_mask & 4 != 0,
                    reasoning_controls: capability_mask & 8 != 0,
                    unknown_fields: BTreeMap::new(),
                };
                assert_eq!(
                    capabilities_cover(&capabilities, &required),
                    required_mask & !capability_mask == 0,
                    "required={required_mask:04b} capabilities={capability_mask:04b}"
                );
            }
        }
    }

    #[test]
    fn responses_continuation_and_store_have_distinct_reasons() {
        let adapter = FamilyAdapter;
        let call = context(LlmApiFamily::OpenAIResponses);
        let pool = pool(
            LlmApiFamily::OpenAIResponses,
            vec![candidate("candidate", "small", 1, all_capabilities())],
        );
        let valid = replay(
            LlmApiFamily::OpenAIResponses,
            LLM_REPLAY_CONTRACT_VERSION,
            "gateway-A",
        );
        assert_eq!(
            preflight(
                &call,
                &responses_request("previous_response_id"),
                Some(&valid),
                &pool,
                &adapter,
            )
            .err()
            .unwrap(),
            IneligibilityReason::ResponsesContinuation
        );
        assert_eq!(
            preflight(
                &call,
                &responses_request("store"),
                Some(&valid),
                &pool,
                &adapter,
            )
            .err()
            .unwrap(),
            IneligibilityReason::ResponsesStore
        );
    }

    #[test]
    fn capability_bearing_generic_and_wrapper_fields_fail_full_preflight() {
        let fixtures = [
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
                    "include": ["reasoning.encrypted_content"]
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
                            "schema": {
                                "$schema": "https://json-schema.org/draft/2020-12/schema",
                                "type": "object"
                            }
                        }
                    }
                }),
            ),
        ];

        for (family, content) in fixtures {
            let request = LlmRequest {
                headers: serde_json::Map::new(),
                content,
            };
            let call = context(family);
            let valid = replay(family, LLM_REPLAY_CONTRACT_VERSION, "gateway-A");
            let pool = pool(
                family,
                vec![candidate("candidate", "small", 1, all_capabilities())],
            );
            assert_eq!(
                preflight(&call, &request, Some(&valid), &pool, &FamilyAdapter)
                    .err()
                    .unwrap(),
                IneligibilityReason::UnsupportedProviderRepresentation,
                "family={family:?} request={:?}",
                request.content
            );
        }
    }

    #[test]
    fn malformed_tool_arguments_fail_full_preflight() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let request = LlmRequest {
            headers: serde_json::Map::new(),
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
        let valid = replay(family, LLM_REPLAY_CONTRACT_VERSION, "gateway-A");
        let pool = pool(
            family,
            vec![candidate("candidate", "small", 1, all_capabilities())],
        );
        assert_eq!(
            preflight(
                &context(family),
                &request,
                Some(&valid),
                &pool,
                &FamilyAdapter,
            )
            .err()
            .unwrap(),
            IneligibilityReason::ProjectionFailed
        );
    }

    #[test]
    fn context_budget_requires_an_explicit_output_and_checks_boundaries() {
        let adapter = FamilyAdapter;
        let request = chat_request(false, Some(32));
        let envelope = adapter
            .decode(LlmApiFamily::OpenAIChatCompletions, &request)
            .unwrap();
        let projection = project_request(&envelope, &CanonicalizerConfig::default()).unwrap();
        let input = canonical_request_budget(&projection.normalized_request).unwrap();
        let output = output_token_limit(&projection.normalized_request).unwrap();
        let exact = input + output;
        let mut config = candidate("candidate", "small", 1, all_capabilities());

        config.max_context_tokens = Some(exact);
        assert!(preflight_candidate(&config, &adapter, &envelope, &projection).is_ok());
        config.max_context_tokens = Some(exact - 1);
        assert_eq!(
            preflight_candidate(&config, &adapter, &envelope, &projection),
            Err(IneligibilityReason::ContextLimitExceeded)
        );

        let unbounded_request = chat_request(false, None);
        let unbounded_envelope = adapter
            .decode(LlmApiFamily::OpenAIChatCompletions, &unbounded_request)
            .unwrap();
        let unbounded_projection =
            project_request(&unbounded_envelope, &CanonicalizerConfig::default()).unwrap();
        config.max_context_tokens = Some(u64::MAX);
        assert_eq!(
            preflight_candidate(
                &config,
                &adapter,
                &unbounded_envelope,
                &unbounded_projection,
            ),
            Err(IneligibilityReason::ContextLimitUnprovable)
        );

        let overflow_request = chat_request(false, Some(u64::MAX));
        let overflow_envelope = adapter
            .decode(LlmApiFamily::OpenAIChatCompletions, &overflow_request)
            .unwrap();
        assert_eq!(
            project_request(&overflow_envelope, &CanonicalizerConfig::default()),
            Err(IneligibilityReason::ProjectionFailed)
        );
    }

    #[test]
    fn candidates_are_filtered_then_sorted_and_truncated() {
        let adapter = FamilyAdapter;
        let request = chat_request(true, Some(16));
        let call = context(LlmApiFamily::OpenAIChatCompletions);
        let valid = replay(
            LlmApiFamily::OpenAIChatCompletions,
            LLM_REPLAY_CONTRACT_VERSION,
            "gateway-A",
        );
        let with_tools = CandidateCapabilities {
            tools: true,
            ..CandidateCapabilities::default()
        };
        let mut pool = pool(
            LlmApiFamily::OpenAIChatCompletions,
            vec![
                candidate("z", "model-z", 1, with_tools.clone()),
                candidate("rejected", "model-r", 0, CandidateCapabilities::default()),
                candidate("b", "model-b", 2, with_tools.clone()),
                candidate("a", "model-a", 1, with_tools),
            ],
        );
        pool.max_candidates_per_sample = 2;
        pool.candidates[0]
            .unknown_fields
            .insert("ignored".into(), json!({"large": "value"}));
        pool.candidates[0]
            .capabilities
            .unknown_fields
            .insert("ignored".into(), json!([1, 2, 3]));

        let outcome = preflight(&call, &request, Some(&valid), &pool, &adapter).unwrap();
        assert_eq!(outcome.eligible_candidate_count, 3);
        assert_eq!(
            outcome
                .eligible_candidate_facts
                .iter()
                .map(|candidate| candidate.candidate_id.as_str())
                .collect::<Vec<_>>(),
            ["a", "z", "b"]
        );
        assert_eq!(outcome.candidates.capacity(), outcome.candidates.len());
        assert!(Arc::ptr_eq(
            &outcome.candidates[0].response_contracts,
            &outcome.candidates[1].response_contracts
        ));
        assert!(
            outcome
                .candidates
                .iter()
                .all(|candidate| candidate.config.unknown_fields.is_empty()
                    && candidate.config.capabilities.unknown_fields.is_empty())
        );
        assert_eq!(
            outcome
                .candidates
                .iter()
                .map(|candidate| candidate.config.id.as_str())
                .collect::<Vec<_>>(),
            ["a", "z"]
        );
        assert_eq!(outcome.rejected_candidates.len(), 1);
        assert_eq!(
            outcome.rejected_candidates[0].reason,
            IneligibilityReason::CandidateCapability
        );
        assert_eq!(
            outcome.candidates[0].request.content["model"],
            json!("model-a")
        );
    }

    #[test]
    fn invalid_and_nonlocal_contracts_decline_before_replay() {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let valid = replay(family, LLM_REPLAY_CONTRACT_VERSION, "gateway-A");
        let pool = pool(
            family,
            vec![candidate("candidate", "small", 1, all_capabilities())],
        );
        let cases = [
            (
                json!({
                    "$schema": "https://json-schema.org/draft/2019-09/schema",
                    "type": "object"
                }),
                IneligibilityReason::UnsupportedContractSchema,
            ),
            (
                json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "type": "object",
                    "required": "not-an-array"
                }),
                IneligibilityReason::InvalidContractSchema,
            ),
            (
                json!({
                    "$schema": "https://json-schema.org/draft/2020-12/schema",
                    "$ref": "https://example.invalid/contract.json"
                }),
                IneligibilityReason::NonLocalContractReference,
            ),
        ];

        for (schema, expected) in cases {
            let mut request = chat_request(false, Some(16));
            request.content.as_object_mut().unwrap().insert(
                "response_format".into(),
                json!({
                    "type": "json_schema",
                    "json_schema": {"name": "answer", "schema": schema}
                }),
            );
            assert_eq!(
                preflight(
                    &context(family),
                    &request,
                    Some(&valid),
                    &pool,
                    &FamilyAdapter,
                )
                .err(),
                Some(expected)
            );
        }
    }
}
