// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Private safe trajectory evidence and memory-only handoff contracts.

use std::collections::BTreeMap;
use std::sync::Arc;

use chrono::{DateTime, Timelike, Utc};
use nemo_relay::api::event::{DataSchema, Event, ScopeCategory};
use nemo_relay::api::llm::{LlmApiFamily, LlmCallRole, LlmTrajectoryScopeSnapshot};
use nemo_relay::api::runtime::{
    LLM_REPLAY_CONTRACT_VERSION, LlmReplayCapability, LlmReplayTransport,
};
use nemo_relay::api::scope::ScopeType;
use nemo_relay::codec::anthropic::AnthropicMessagesCodec;
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay::codec::openai_responses::OpenAIResponsesCodec;
use nemo_relay::codec::request::{AnnotatedLlmRequest, Message, MessageContent, ToolChoice};
use nemo_relay::codec::response::{AnnotatedLlmResponse, FinishReason};
use nemo_relay::codec::traits::LlmResponseCodec;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use uuid::Uuid;

use crate::adapter::RouterRequestEnvelope;
use crate::config::CandidateConfig;
use crate::eligibility::IneligibilityReason;
use crate::fingerprint::{canonical_serialize_bytes, sha256_hex};
use crate::preflight::{EligibleCandidate, contains_sensitive_control_material};
use crate::projection::{
    RouterRequestProjectionV1, RouterRoutingContextProjectionV1, SanitizedMessageContent,
};
use crate::response_validator::CandidateResponseContractsV1;

pub(crate) const RESPONSE_PROJECTION_SCHEMA_V1: &str = "nemo.relay.router.response-projection@1";
pub(crate) const CAPTURED_EVENT_SCHEMA_V1: &str = "nemo.relay.router.captured-event@1";
pub(crate) const PENDING_TRAJECTORY_SCHEMA_V1: &str = "nemo.relay.router.pending-trajectory@1";
pub(crate) const TERMINAL_TRAJECTORY_SCHEMA_V1: &str = "nemo.relay.router.terminal-trajectory@1";
pub(crate) const REPLAY_CAPABILITY_SCHEMA_V1: &str = "nemo.relay.router.replay-capability@1";
pub(crate) const CANDIDATE_FACT_SCHEMA_V1: &str = "nemo.relay.router.candidate-fact@1";
pub(crate) const TRAJECTORY_SANITIZER_VERSION: u32 = 1;
pub(crate) const MAX_TRAJECTORY_DIAGNOSTICS: usize = 8;
pub(crate) const MAX_TRAJECTORY_DIAGNOSTIC_BYTES: usize = 256;

/// Normalize persisted window clocks before any canonical aggregate is hashed.
pub(crate) fn truncate_utc_to_milliseconds(value: DateTime<Utc>) -> DateTime<Utc> {
    value
        .with_nanosecond(value.timestamp_subsec_millis() * 1_000_000)
        .expect("a millisecond-aligned nanosecond is always valid")
}

/// Safe normalized assistant content and tool evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RouterResponseProjectionV1 {
    pub(crate) schema: String,
    pub(crate) sanitizer_version: u32,
    pub(crate) id: Option<String>,
    pub(crate) model: Option<String>,
    pub(crate) message: Option<SanitizedMessageContent>,
    pub(crate) tool_calls: Option<Vec<SanitizedResponseToolCallV1>>,
    pub(crate) finish_reason: Option<FinishReason>,
    pub(crate) usage: Option<SanitizedResponseUsageV1>,
    pub(crate) semantic_response_fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct SanitizedResponseToolCallV1 {
    pub(crate) id: String,
    pub(crate) name: String,
    pub(crate) arguments: Json,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct SanitizedResponseUsageV1 {
    pub(crate) prompt_tokens: Option<u64>,
    pub(crate) completion_tokens: Option<u64>,
    pub(crate) total_tokens: Option<u64>,
    pub(crate) cache_read_tokens: Option<u64>,
    pub(crate) cache_write_tokens: Option<u64>,
}

/// Safe facts proving that a memory-only replay transport was validated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct ReplayCapabilityFactsV1 {
    pub(crate) schema: String,
    pub(crate) contract_version: u32,
    pub(crate) api_family: LlmApiFamily,
    pub(crate) transport_identity: String,
    pub(crate) non_resumable: bool,
    pub(crate) capability_fingerprint: String,
}

/// Candidate capability flags that can affect output semantics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PersistedCandidateCapabilitiesV1 {
    pub(crate) tools: bool,
    pub(crate) multimodal_input: bool,
    pub(crate) structured_output: bool,
    pub(crate) reasoning_controls: bool,
}

/// Safe candidate configuration without the replay request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PersistedCandidateFactV1 {
    pub(crate) schema: String,
    pub(crate) candidate_id: String,
    pub(crate) model: String,
    pub(crate) model_revision: String,
    pub(crate) cost_rank: u32,
    pub(crate) capabilities: PersistedCandidateCapabilitiesV1,
    pub(crate) decoding_fingerprint: String,
}

/// Frozen owner-path element retained with safe persisted identity.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TrajectoryOwnerScopeV1 {
    pub(crate) uuid: Uuid,
    pub(crate) name: String,
    pub(crate) scope_type: ScopeType,
}

/// Process/project identity injected into all trajectory facts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TrajectoryIdentity {
    pub(crate) process_instance_id: Uuid,
    pub(crate) project_uuid: Uuid,
    pub(crate) project_id: String,
    pub(crate) policy_version_ids: BTreeMap<String, String>,
    pub(crate) learning_generation_ids: BTreeMap<String, Uuid>,
}

impl TrajectoryIdentity {
    pub(crate) fn for_pools<'a>(
        project_id: Option<&str>,
        pool_ids: impl IntoIterator<Item = &'a str>,
    ) -> Self {
        let project_uuid = Uuid::now_v7();
        let pool_ids = pool_ids.into_iter().collect::<Vec<_>>();
        Self {
            process_instance_id: Uuid::now_v7(),
            project_uuid,
            project_id: project_id
                .map(str::to_string)
                .unwrap_or_else(|| project_uuid.to_string()),
            policy_version_ids: pool_ids
                .iter()
                .map(|pool_id| ((*pool_id).to_string(), ephemeral_policy_version_id(pool_id)))
                .collect(),
            learning_generation_ids: pool_ids
                .iter()
                .map(|pool_id| (pool_id.to_string(), Uuid::now_v7()))
                .collect(),
        }
    }

    pub(crate) fn policy_version_id(&self, pool_id: &str) -> Option<&str> {
        self.policy_version_ids.get(pool_id).map(String::as_str)
    }

    pub(crate) fn learning_generation_id(&self, pool_id: &str) -> Option<Uuid> {
        self.learning_generation_ids.get(pool_id).copied()
    }
}

fn ephemeral_policy_version_id(pool_id: &str) -> String {
    const DOMAIN: &[u8] = b"nemo.relay.router.ephemeral-policy@1\0";
    let mut material = Vec::with_capacity(DOMAIN.len().saturating_add(pool_id.len()));
    material.extend_from_slice(DOMAIN);
    material.extend_from_slice(pool_id.as_bytes());
    sha256_hex(&material)
}

/// Stable kind used for event deduplication and actor matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CapturedEventKind {
    Scope,
    Mark,
}

/// Allowlisted normalized codec fields attached to one event.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub(crate) struct CapturedCodecAnnotationsV1 {
    pub(crate) model_name: Option<String>,
    pub(crate) tool_call_id: Option<String>,
    pub(crate) subtype: Option<String>,
    pub(crate) normalized_request: Option<crate::projection::SanitizedAnnotatedLlmRequest>,
    pub(crate) normalized_response: Option<RouterResponseProjectionV1>,
}

/// One bounded, sanitized event shared by overlapping trajectory windows.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct CapturedTrajectoryEvent {
    pub(crate) schema: String,
    pub(crate) ingest_seq: u64,
    pub(crate) event_uuid: Uuid,
    pub(crate) parent_uuid: Option<Uuid>,
    pub(crate) kind: CapturedEventKind,
    pub(crate) scope_phase: Option<ScopeCategory>,
    pub(crate) category: Option<String>,
    pub(crate) call_role: Option<LlmCallRole>,
    pub(crate) timestamp: DateTime<Utc>,
    pub(crate) name: String,
    pub(crate) data: Option<Json>,
    pub(crate) metadata: Option<Json>,
    pub(crate) data_schema: Option<DataSchema>,
    pub(crate) scope_type: Option<ScopeType>,
    pub(crate) safe_scope_attributes: Vec<String>,
    pub(crate) codec_annotations: CapturedCodecAnnotationsV1,
    pub(crate) canonical_payload_hash: String,
    pub(crate) canonical_size_bytes: usize,
}

/// Structural evidence that an event could not fit the shared projection bound.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct OversizedTrajectoryEvent {
    pub(crate) ingest_seq: u64,
    pub(crate) event_uuid: Uuid,
    pub(crate) parent_uuid: Option<Uuid>,
    pub(crate) kind: CapturedEventKind,
    pub(crate) scope_phase: Option<ScopeCategory>,
    pub(crate) category: Option<String>,
    pub(crate) call_role: Option<LlmCallRole>,
    pub(crate) scope_type: Option<ScopeType>,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum ProjectedTrajectoryEvent {
    Captured(Arc<CapturedTrajectoryEvent>),
    Oversized(OversizedTrajectoryEvent),
}

/// Why an evaluable window closed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TrajectoryTrigger {
    ProgressReached,
    OwnerTerminated,
    DeadlineElapsed,
    Shutdown,
}

/// Stable rejection reasons for non-evaluable windows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum TrajectoryRejectionReason {
    EventLoss,
    Overflow,
    ContradictoryOwnership,
    CanceledBeforeAnchorEnd,
    RejectedDeliveryBarrier,
}

/// Exhaustive terminal state persisted by the acknowledged sink.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub(crate) enum TrajectoryTerminalStateV1 {
    Closed { trigger: TrajectoryTrigger },
    Rejected { reason: TrajectoryRejectionReason },
    OrphanedNonResumable,
}

/// Bounded stable-code diagnostic; callers may not pass runtime-owned text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct TrajectoryDiagnosticV1 {
    pub(crate) code: String,
    pub(crate) message: String,
}

/// Safe pre-accept payload. It contains no executable replay authority.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PendingTrajectoryWindow {
    pub(crate) schema: String,
    pub(crate) anchor_id: Uuid,
    pub(crate) anchor_call_uuid: Uuid,
    pub(crate) root_uuid: Uuid,
    pub(crate) owner_uuid: Uuid,
    pub(crate) owner_path: Vec<TrajectoryOwnerScopeV1>,
    pub(crate) pool_id: String,
    pub(crate) anchor_model_revision: String,
    pub(crate) process_instance_id: Uuid,
    pub(crate) project_uuid: Uuid,
    pub(crate) project_id: String,
    pub(crate) config_generation_id: String,
    pub(crate) policy_version_id: String,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) request_projection: RouterRequestProjectionV1,
    pub(crate) routing_context_projection: RouterRoutingContextProjectionV1,
    pub(crate) normalized_anchor_response: RouterResponseProjectionV1,
    pub(crate) replay_capability_facts: ReplayCapabilityFactsV1,
    pub(crate) candidate_facts: Vec<PersistedCandidateFactV1>,
    pub(crate) requested_progress: usize,
    pub(crate) opened_at: DateTime<Utc>,
    pub(crate) deadline_at: DateTime<Utc>,
}

/// Safe acknowledged terminal payload.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct PersistedTrajectoryTerminalV1 {
    pub(crate) schema: String,
    pub(crate) pending: PendingTrajectoryWindow,
    pub(crate) state: TrajectoryTerminalStateV1,
    pub(crate) events: Vec<CapturedTrajectoryEvent>,
    pub(crate) observed_progress: usize,
    pub(crate) is_partial: bool,
    pub(crate) promotion_eligible: bool,
    pub(crate) closed_at: DateTime<Utc>,
    pub(crate) diagnostics: Vec<TrajectoryDiagnosticV1>,
}

/// Memory-only accepted anchor state retained until terminal acknowledgement.
pub(crate) struct TrajectoryWindowSeed {
    pub(crate) pending: PendingTrajectoryWindow,
    pub(crate) request_envelope: RouterRequestEnvelope,
    pub(crate) replay_transport: Arc<dyn LlmReplayTransport>,
    pub(crate) eligible_candidates: Vec<EligibleCandidate>,
}

/// Memory-only evaluable handoff. This deliberately has no serde or Debug impl.
#[allow(dead_code)] // Spec 05 consumes the authority fields after this rollout gate.
pub(crate) struct ClosedTrajectoryWindow {
    pub(crate) pending: PendingTrajectoryWindow,
    pub(crate) request_envelope: RouterRequestEnvelope,
    pub(crate) replay_transport: Arc<dyn LlmReplayTransport>,
    pub(crate) eligible_candidates: Vec<EligibleCandidate>,
    pub(crate) events: Vec<Arc<CapturedTrajectoryEvent>>,
    pub(crate) observed_progress: usize,
    pub(crate) trigger: TrajectoryTrigger,
    pub(crate) is_partial: bool,
    pub(crate) promotion_eligible: bool,
    pub(crate) closed_at: DateTime<Utc>,
}

/// Memory-only rejected handoff with no replay authorization.
#[allow(dead_code)] // Kept as the typed rejection boundary for the next sink rollout.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct RejectedTrajectoryWindow {
    pub(crate) pending: PendingTrajectoryWindow,
    pub(crate) reason: TrajectoryRejectionReason,
    pub(crate) events: Vec<Arc<CapturedTrajectoryEvent>>,
    pub(crate) observed_progress: usize,
    pub(crate) closed_at: DateTime<Utc>,
    pub(crate) diagnostics: Vec<TrajectoryDiagnosticV1>,
}

impl PendingTrajectoryWindow {
    #[allow(dead_code)]
    pub(crate) fn anchor_id(&self) -> Uuid {
        self.anchor_id
    }

    pub(crate) fn canonical_bytes(&self) -> Result<Vec<u8>, ()> {
        canonical_serialize_bytes(self)
    }

    pub(crate) fn payload_hash(&self) -> Result<String, ()> {
        self.canonical_bytes().map(|bytes| sha256_hex(&bytes))
    }
}

impl PersistedTrajectoryTerminalV1 {
    #[allow(dead_code)]
    pub(crate) fn anchor_id(&self) -> Uuid {
        self.pending.anchor_id
    }

    pub(crate) fn canonical_bytes(&self) -> Result<Vec<u8>, ()> {
        canonical_serialize_bytes(self)
    }

    pub(crate) fn payload_hash(&self) -> Result<String, ()> {
        self.canonical_bytes().map(|bytes| sha256_hex(&bytes))
    }
}

impl TrajectoryWindowSeed {
    pub(crate) fn new(
        pending: PendingTrajectoryWindow,
        request_envelope: RouterRequestEnvelope,
        replay_transport: Arc<dyn LlmReplayTransport>,
        eligible_candidates: Vec<EligibleCandidate>,
    ) -> Self {
        Self {
            pending,
            request_envelope,
            replay_transport,
            eligible_candidates,
        }
    }

    pub(crate) fn pending_projection(&self) -> PendingTrajectoryWindow {
        self.pending.clone()
    }

    #[allow(dead_code)]
    pub(crate) fn anchor_id(&self) -> Uuid {
        self.pending.anchor_id
    }

    pub(crate) fn into_closed(
        self,
        events: Vec<Arc<CapturedTrajectoryEvent>>,
        observed_progress: usize,
        trigger: TrajectoryTrigger,
        closed_at: DateTime<Utc>,
    ) -> ClosedTrajectoryWindow {
        ClosedTrajectoryWindow {
            pending: self.pending,
            request_envelope: self.request_envelope,
            replay_transport: self.replay_transport,
            eligible_candidates: self.eligible_candidates,
            events,
            observed_progress,
            trigger,
            is_partial: trigger != TrajectoryTrigger::ProgressReached,
            promotion_eligible: false,
            closed_at,
        }
    }

    #[allow(dead_code)]
    pub(crate) fn into_rejected(
        self,
        reason: TrajectoryRejectionReason,
        events: Vec<Arc<CapturedTrajectoryEvent>>,
        observed_progress: usize,
        closed_at: DateTime<Utc>,
        diagnostics: Vec<TrajectoryDiagnosticV1>,
    ) -> RejectedTrajectoryWindow {
        RejectedTrajectoryWindow {
            pending: self.pending,
            reason,
            events,
            observed_progress,
            closed_at,
            diagnostics: bounded_diagnostics(diagnostics),
        }
    }
}

impl ClosedTrajectoryWindow {
    pub(crate) fn anchor_id(&self) -> Uuid {
        self.pending.anchor_id
    }

    #[allow(dead_code)]
    pub(crate) fn terminal(&self) -> PersistedTrajectoryTerminalV1 {
        PersistedTrajectoryTerminalV1::closed(
            self.pending.clone(),
            self.events
                .iter()
                .map(|event| event.as_ref().clone())
                .collect(),
            self.observed_progress,
            self.trigger,
            self.closed_at,
            Vec::new(),
        )
    }
}

impl RejectedTrajectoryWindow {
    #[allow(dead_code)]
    pub(crate) fn terminal(&self) -> PersistedTrajectoryTerminalV1 {
        PersistedTrajectoryTerminalV1::rejected(
            self.pending.clone(),
            self.events
                .iter()
                .map(|event| event.as_ref().clone())
                .collect(),
            self.observed_progress,
            self.reason,
            self.closed_at,
            self.diagnostics.clone(),
        )
    }
}

impl PersistedTrajectoryTerminalV1 {
    pub(crate) fn closed(
        pending: PendingTrajectoryWindow,
        events: Vec<CapturedTrajectoryEvent>,
        observed_progress: usize,
        trigger: TrajectoryTrigger,
        closed_at: DateTime<Utc>,
        diagnostics: Vec<TrajectoryDiagnosticV1>,
    ) -> Self {
        Self {
            schema: TERMINAL_TRAJECTORY_SCHEMA_V1.to_string(),
            pending,
            state: TrajectoryTerminalStateV1::Closed { trigger },
            events,
            observed_progress,
            is_partial: trigger != TrajectoryTrigger::ProgressReached,
            promotion_eligible: false,
            closed_at,
            diagnostics: bounded_diagnostics(diagnostics),
        }
    }

    pub(crate) fn rejected(
        pending: PendingTrajectoryWindow,
        events: Vec<CapturedTrajectoryEvent>,
        observed_progress: usize,
        reason: TrajectoryRejectionReason,
        closed_at: DateTime<Utc>,
        diagnostics: Vec<TrajectoryDiagnosticV1>,
    ) -> Self {
        Self {
            schema: TERMINAL_TRAJECTORY_SCHEMA_V1.to_string(),
            pending,
            state: TrajectoryTerminalStateV1::Rejected { reason },
            events,
            observed_progress,
            is_partial: true,
            promotion_eligible: false,
            closed_at,
            diagnostics: bounded_diagnostics(diagnostics),
        }
    }
}

impl TrajectoryDiagnosticV1 {
    #[allow(dead_code)]
    pub(crate) fn stable(code: &'static str, message: &'static str) -> Self {
        Self {
            code: bounded_diagnostic_text(code),
            message: bounded_diagnostic_text(message),
        }
    }
}

#[allow(dead_code)]
fn bounded_diagnostic_text(value: &str) -> String {
    let end = value.floor_char_boundary(value.len().min(MAX_TRAJECTORY_DIAGNOSTIC_BYTES));
    value[..end].to_string()
}

fn bounded_diagnostics(diagnostics: Vec<TrajectoryDiagnosticV1>) -> Vec<TrajectoryDiagnosticV1> {
    diagnostics
        .into_iter()
        .take(MAX_TRAJECTORY_DIAGNOSTICS)
        .map(|mut diagnostic| {
            diagnostic.code.truncate(
                diagnostic.code.floor_char_boundary(
                    diagnostic.code.len().min(MAX_TRAJECTORY_DIAGNOSTIC_BYTES),
                ),
            );
            diagnostic.message.truncate(
                diagnostic.message.floor_char_boundary(
                    diagnostic
                        .message
                        .len()
                        .min(MAX_TRAJECTORY_DIAGNOSTIC_BYTES),
                ),
            );
            diagnostic
        })
        .collect()
}

/// Resource bounds applied before a subscriber retains a projected event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EventProjectionLimits {
    pub(crate) max_depth: usize,
    pub(crate) max_nodes: usize,
    pub(crate) max_string_bytes: usize,
    pub(crate) max_output_bytes: usize,
}

impl EventProjectionLimits {
    pub(crate) fn for_window_bytes(max_output_bytes: usize) -> Self {
        Self {
            max_depth: 64,
            max_nodes: max_output_bytes.clamp(1, 65_536),
            max_string_bytes: max_output_bytes.min(1024 * 1024),
            max_output_bytes,
        }
    }
}

/// Decode a raw anchor response with the authoritative family codec and retain
/// only bounded provider-neutral evidence.
pub(crate) fn project_anchor_response(
    family: LlmApiFamily,
    response: &Json,
    max_output_bytes: usize,
) -> Result<RouterResponseProjectionV1, IneligibilityReason> {
    let annotated = match family {
        LlmApiFamily::OpenAIChatCompletions => OpenAIChatCodec.decode_response(response),
        LlmApiFamily::OpenAIResponses => OpenAIResponsesCodec.decode_response(response),
        LlmApiFamily::AnthropicMessages => AnthropicMessagesCodec.decode_response(response),
    }
    .map_err(|_| IneligibilityReason::ProjectionFailed)?;
    project_annotated_response(
        &annotated,
        EventProjectionLimits::for_window_bytes(max_output_bytes),
    )
}

fn project_annotated_response(
    response: &AnnotatedLlmResponse,
    limits: EventProjectionLimits,
) -> Result<RouterResponseProjectionV1, IneligibilityReason> {
    check_optional_string(response.id.as_deref(), limits)?;
    check_optional_string(response.model.as_deref(), limits)?;
    let mut retained_bytes = response.id.as_ref().map_or(0, String::len);
    retained_bytes = retained_bytes
        .checked_add(response.model.as_ref().map_or(0, String::len))
        .ok_or(IneligibilityReason::ProjectionFailed)?;
    if let Some(FinishReason::Unknown(reason)) = response.finish_reason.as_ref() {
        check_string(reason, limits)?;
    }

    let message = response
        .message
        .as_ref()
        .map(|content| {
            precheck_message_content(content, limits)?;
            crate::projection::sanitize_content(content, false)
        })
        .transpose()?;
    if let Some(message) = &message {
        retained_bytes = retained_bytes
            .checked_add(
                canonical_serialize_bytes(message)
                    .map_err(|_| IneligibilityReason::ProjectionFailed)?
                    .len(),
            )
            .ok_or(IneligibilityReason::ProjectionFailed)?;
    }
    let tool_calls = response
        .tool_calls
        .as_ref()
        .map(|calls| {
            if calls.len() > limits.max_nodes {
                return Err(IneligibilityReason::ProjectionFailed);
            }
            calls
                .iter()
                .map(|call| {
                    check_string(&call.id, limits)?;
                    check_string(&call.name, limits)?;
                    let arguments = sanitize_bounded_json(&call.arguments, limits)
                        .map_err(|_| IneligibilityReason::ProjectionFailed)?;
                    retained_bytes = retained_bytes
                        .checked_add(call.id.len())
                        .and_then(|bytes| bytes.checked_add(call.name.len()))
                        .and_then(|bytes| {
                            canonical_serialize_bytes(&arguments)
                                .ok()
                                .and_then(|arguments| bytes.checked_add(arguments.len()))
                        })
                        .ok_or(IneligibilityReason::ProjectionFailed)?;
                    if retained_bytes > limits.max_output_bytes {
                        return Err(IneligibilityReason::ProjectionFailed);
                    }
                    Ok(SanitizedResponseToolCallV1 {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        arguments,
                    })
                })
                .collect::<Result<Vec<_>, IneligibilityReason>>()
        })
        .transpose()?;
    let usage = response
        .usage
        .as_ref()
        .map(|usage| SanitizedResponseUsageV1 {
            prompt_tokens: usage.prompt_tokens,
            completion_tokens: usage.completion_tokens,
            total_tokens: usage.total_tokens,
            cache_read_tokens: usage.cache_read_tokens,
            cache_write_tokens: usage.cache_write_tokens,
        });
    let mut projection = RouterResponseProjectionV1 {
        schema: RESPONSE_PROJECTION_SCHEMA_V1.to_string(),
        sanitizer_version: TRAJECTORY_SANITIZER_VERSION,
        id: response.id.clone(),
        model: response.model.clone(),
        message,
        tool_calls,
        finish_reason: response.finish_reason.clone(),
        usage,
        semantic_response_fingerprint: String::new(),
    };
    projection.semantic_response_fingerprint =
        fingerprint_without_fields(&projection, &["semantic_response_fingerprint"])
            .map_err(|_| IneligibilityReason::ProjectionFailed)?;
    let bytes = canonical_serialize_bytes(&projection)
        .map_err(|_| IneligibilityReason::ProjectionFailed)?;
    if bytes.len() > limits.max_output_bytes {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    Ok(projection)
}

impl ReplayCapabilityFactsV1 {
    pub(crate) fn from_capability(
        capability: &LlmReplayCapability,
    ) -> Result<Self, IneligibilityReason> {
        if capability.contract_version != LLM_REPLAY_CONTRACT_VERSION {
            return Err(IneligibilityReason::ReplayVersionMismatch);
        }
        if capability.transport_identity.is_empty()
            || capability.transport_identity.len() > 256
            || capability
                .transport_identity
                .bytes()
                .any(|byte| !byte.is_ascii_graphic())
            || contains_sensitive_control_material(&capability.transport_identity)
        {
            return Err(IneligibilityReason::ReplayIdentityInvalid);
        }
        let mut facts = Self {
            schema: REPLAY_CAPABILITY_SCHEMA_V1.to_string(),
            contract_version: capability.contract_version,
            api_family: capability.api_family,
            transport_identity: capability.transport_identity.clone(),
            non_resumable: true,
            capability_fingerprint: String::new(),
        };
        facts.capability_fingerprint =
            fingerprint_without_fields(&facts, &["capability_fingerprint"])
                .map_err(|_| IneligibilityReason::ProjectionFailed)?;
        Ok(facts)
    }
}

impl PersistedCandidateFactV1 {
    pub(crate) fn from_eligible(
        candidate: &EligibleCandidate,
        request: &RouterRequestProjectionV1,
    ) -> Result<Self, IneligibilityReason> {
        Self::from_config(&candidate.config, &candidate.response_contracts, request)
    }

    pub(crate) fn from_config(
        config: &CandidateConfig,
        response_contracts: &CandidateResponseContractsV1,
        request: &RouterRequestProjectionV1,
    ) -> Result<Self, IneligibilityReason> {
        let capabilities = PersistedCandidateCapabilitiesV1 {
            tools: config.capabilities.tools,
            multimodal_input: config.capabilities.multimodal_input,
            structured_output: config.capabilities.structured_output,
            reasoning_controls: config.capabilities.reasoning_controls,
        };
        let tool_contract_fingerprint = response_contracts.tool_contract_fingerprint();
        let response_contract_fingerprint = response_contracts.response_contract_fingerprint();
        let decoding_semantics = serde_json::json!({
            "schema": "nemo.relay.router.decoding-partition@1",
            "generation_controls": {
                "params": request.normalized_request.params,
                "tool_choice": request.normalized_request.tool_choice,
                "truncation": request.normalized_request.truncation,
                "reasoning": request.normalized_request.reasoning,
                "service_tier": request.normalized_request.service_tier,
                "parallel_tool_calls": request.normalized_request.parallel_tool_calls,
                "max_output_tokens": request.normalized_request.max_output_tokens,
                "max_tool_calls": request.normalized_request.max_tool_calls,
                "top_logprobs": request.normalized_request.top_logprobs,
            },
            "tool_contract_fingerprint": tool_contract_fingerprint,
            "response_contract_fingerprint": response_contract_fingerprint,
            "capabilities": capabilities,
        });
        let decoding_fingerprint = crate::fingerprint::fingerprint_json(&decoding_semantics)
            .map_err(|_| IneligibilityReason::ProjectionFailed)?;
        Ok(Self {
            schema: CANDIDATE_FACT_SCHEMA_V1.to_string(),
            candidate_id: config.id.clone(),
            model: config.model.clone(),
            model_revision: config.model_revision.clone(),
            cost_rank: config.cost_rank,
            capabilities,
            decoding_fingerprint,
        })
    }
}

pub(crate) fn project_candidate_facts(
    candidates: &[EligibleCandidate],
    request: &RouterRequestProjectionV1,
) -> Result<Vec<PersistedCandidateFactV1>, IneligibilityReason> {
    candidates
        .iter()
        .map(|candidate| PersistedCandidateFactV1::from_eligible(candidate, request))
        .collect()
}

pub(crate) fn project_owner_path(
    path: &[LlmTrajectoryScopeSnapshot],
    max_output_bytes: usize,
) -> Result<Vec<TrajectoryOwnerScopeV1>, IneligibilityReason> {
    let limits = EventProjectionLimits::for_window_bytes(max_output_bytes);
    if path.is_empty() || path.len() > limits.max_depth {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    let mut projected_allocation = path
        .len()
        .checked_mul(std::mem::size_of::<TrajectoryOwnerScopeV1>())
        .ok_or(IneligibilityReason::ProjectionFailed)?;
    for scope in path {
        check_string(&scope.name, limits)?;
        projected_allocation = projected_allocation
            .checked_add(scope.name.len())
            .ok_or(IneligibilityReason::ProjectionFailed)?;
        if projected_allocation > max_output_bytes {
            return Err(IneligibilityReason::ProjectionFailed);
        }
    }
    let projected = path
        .iter()
        .map(|scope| {
            Ok(TrajectoryOwnerScopeV1 {
                uuid: scope.uuid,
                name: scope.name.clone(),
                scope_type: scope.scope_type,
            })
        })
        .collect::<Result<Vec<_>, IneligibilityReason>>()?;
    if canonical_serialize_bytes(&projected)
        .map_err(|_| IneligibilityReason::ProjectionFailed)?
        .len()
        > max_output_bytes
    {
        return Err(IneligibilityReason::ProjectionFailed);
    }
    Ok(projected)
}

/// Build one bounded event or a small structural oversize observation.
pub(crate) fn project_captured_event(
    ingest_seq: u64,
    event: &Event,
    limits: EventProjectionLimits,
) -> ProjectedTrajectoryEvent {
    match try_project_captured_event(ingest_seq, event, limits) {
        Ok(event) => ProjectedTrajectoryEvent::Captured(Arc::new(event)),
        Err(()) => ProjectedTrajectoryEvent::Oversized(oversized_marker(ingest_seq, event)),
    }
}

fn try_project_captured_event(
    ingest_seq: u64,
    event: &Event,
    limits: EventProjectionLimits,
) -> Result<CapturedTrajectoryEvent, ()> {
    check_string(event.name(), limits).map_err(|_| ())?;
    let category = event.category().map(|value| value.as_str().to_string());
    if let Some(category) = category.as_deref() {
        check_string(category, limits).map_err(|_| ())?;
    }
    // Raw LLM data is a provider wire object and may contain native extras or
    // response side channels. Its explicit codec annotation is the only LLM
    // payload allowed across this boundary.
    let data = if event.category().map(|value| value.as_str()) == Some("llm") {
        None
    } else {
        event
            .data()
            .map(|value| sanitize_bounded_json(value, limits))
            .transpose()?
    };
    let metadata = event
        .metadata()
        .map(|value| sanitize_bounded_json(value, limits))
        .transpose()?;
    let data_schema = event
        .data_schema()
        .map(|schema| {
            check_string(&schema.name, limits).map_err(|_| ())?;
            check_string(&schema.version, limits).map_err(|_| ())?;
            Ok::<_, ()>(schema.clone())
        })
        .transpose()?;
    let safe_scope_attributes = event
        .attributes()
        .unwrap_or_default()
        .iter()
        .map(|attribute| {
            check_string(attribute, limits).map_err(|_| ())?;
            Ok(attribute.clone())
        })
        .collect::<Result<Vec<_>, ()>>()?;
    let codec_annotations = project_event_codec_annotations(event, limits)?;
    let mut projected = CapturedTrajectoryEvent {
        schema: CAPTURED_EVENT_SCHEMA_V1.to_string(),
        ingest_seq,
        event_uuid: event.uuid(),
        parent_uuid: event.parent_uuid(),
        kind: event_kind(event),
        scope_phase: event.scope_category(),
        category,
        call_role: event.llm_call_role(),
        timestamp: *event.timestamp(),
        name: event.name().to_string(),
        data,
        metadata,
        data_schema,
        scope_type: event.scope_type(),
        safe_scope_attributes,
        codec_annotations,
        canonical_payload_hash: String::new(),
        canonical_size_bytes: 0,
    };
    projected.canonical_payload_hash = fingerprint_without_fields(
        &projected,
        &[
            "ingest_seq",
            "canonical_payload_hash",
            "canonical_size_bytes",
        ],
    )?;
    projected.canonical_size_bytes = canonical_size_fixed_point(&mut projected)?;
    if projected.canonical_size_bytes > limits.max_output_bytes {
        return Err(());
    }
    Ok(projected)
}

fn project_event_codec_annotations(
    event: &Event,
    limits: EventProjectionLimits,
) -> Result<CapturedCodecAnnotationsV1, ()> {
    let Some(profile) = event.category_profile() else {
        return Ok(CapturedCodecAnnotationsV1::default());
    };
    for value in [
        profile.model_name.as_deref(),
        profile.tool_call_id.as_deref(),
        profile.subtype.as_deref(),
    ] {
        check_optional_string(value, limits).map_err(|_| ())?;
    }
    let normalized_request = profile
        .annotated_request
        .as_ref()
        .map(|request| {
            precheck_annotated_request(request.as_ref(), limits).map_err(|_| ())?;
            let sanitized =
                crate::projection::sanitize_annotated_request(request.as_ref()).map_err(|_| ())?;
            let value = serde_json::to_value(&sanitized).map_err(|_| ())?;
            let bounded = sanitize_bounded_json(&value, limits)?;
            serde_json::from_value(bounded).map_err(|_| ())
        })
        .transpose()?;
    let normalized_response = profile
        .annotated_response
        .as_ref()
        .map(|response| project_annotated_response(response.as_ref(), limits).map_err(|_| ()))
        .transpose()?;
    Ok(CapturedCodecAnnotationsV1 {
        model_name: profile.model_name.clone(),
        tool_call_id: profile.tool_call_id.clone(),
        subtype: profile.subtype.clone(),
        normalized_request,
        normalized_response,
    })
}

impl CapturedTrajectoryEvent {
    pub(crate) fn is_internal_router_event(&self) -> bool {
        is_internal_router_event_fields(self.scope_type, self.category.as_deref(), self.call_role)
    }

    pub(crate) fn hook_event_name(&self) -> Option<&str> {
        self.metadata
            .as_ref()?
            .as_object()?
            .get("hook_event_name")?
            .as_str()
    }

    pub(crate) fn is_subagent_handoff(&self) -> bool {
        self.kind == CapturedEventKind::Scope
            && self.scope_phase == Some(ScopeCategory::Start)
            && self.scope_type == Some(ScopeType::Agent)
            && self.category.as_deref() == Some("agent")
            && self
                .metadata
                .as_ref()
                .and_then(Json::as_object)
                .and_then(|metadata| metadata.get("nemo_relay_scope_role"))
                .and_then(Json::as_str)
                == Some("subagent")
    }
}

impl OversizedTrajectoryEvent {
    pub(crate) fn is_internal_router_event(&self) -> bool {
        is_internal_router_event_fields(self.scope_type, self.category.as_deref(), self.call_role)
    }
}

impl ProjectedTrajectoryEvent {
    pub(crate) fn is_internal_router_event(&self) -> bool {
        match self {
            Self::Captured(event) => event.is_internal_router_event(),
            Self::Oversized(event) => event.is_internal_router_event(),
        }
    }
}

fn is_internal_router_event_fields(
    scope_type: Option<ScopeType>,
    category: Option<&str>,
    call_role: Option<LlmCallRole>,
) -> bool {
    matches!(scope_type, Some(ScopeType::Evaluator | ScopeType::Embedder))
        || (category == Some("llm") && call_role != Some(LlmCallRole::Primary))
}

fn oversized_marker(ingest_seq: u64, event: &Event) -> OversizedTrajectoryEvent {
    let category = event.category().and_then(|category| {
        (category.as_str().len() <= 256).then(|| category.as_str().to_string())
    });
    OversizedTrajectoryEvent {
        ingest_seq,
        event_uuid: event.uuid(),
        parent_uuid: event.parent_uuid(),
        kind: event_kind(event),
        scope_phase: event.scope_category(),
        category,
        call_role: event.llm_call_role(),
        scope_type: event.scope_type(),
    }
}

fn event_kind(event: &Event) -> CapturedEventKind {
    match event {
        Event::Scope(_) => CapturedEventKind::Scope,
        Event::Mark(_) => CapturedEventKind::Mark,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ProjectionBoundError {
    Depth,
    Nodes,
    StringBytes,
    OutputBytes,
}

struct BoundedJsonProjector {
    limits: EventProjectionLimits,
    nodes: usize,
    retained_bytes: usize,
}

impl BoundedJsonProjector {
    fn project(&mut self, value: &Json, depth: usize) -> Result<Json, ProjectionBoundError> {
        if depth > self.limits.max_depth {
            return Err(ProjectionBoundError::Depth);
        }
        self.nodes = self
            .nodes
            .checked_add(1)
            .ok_or(ProjectionBoundError::Nodes)?;
        if self.nodes > self.limits.max_nodes {
            return Err(ProjectionBoundError::Nodes);
        }
        Ok(match value {
            Json::Null => Json::Null,
            Json::Bool(value) => Json::Bool(*value),
            Json::Number(value) => Json::Number(value.clone()),
            Json::String(value) => {
                self.retain_string(value)?;
                Json::String(value.clone())
            }
            Json::Array(values) => Json::Array(
                values
                    .iter()
                    .map(|value| self.project(value, depth + 1))
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Json::Object(values) => {
                let mut projected = serde_json::Map::new();
                for (key, value) in values {
                    self.nodes = self
                        .nodes
                        .checked_add(1)
                        .ok_or(ProjectionBoundError::Nodes)?;
                    if self.nodes > self.limits.max_nodes {
                        return Err(ProjectionBoundError::Nodes);
                    }
                    if key.len() > self.limits.max_string_bytes {
                        return Err(ProjectionBoundError::StringBytes);
                    }
                    if crate::projection::is_sensitive_projection_key(key) {
                        continue;
                    }
                    self.retain_string(key)?;
                    projected.insert(key.clone(), self.project(value, depth + 1)?);
                }
                Json::Object(projected)
            }
        })
    }

    fn retain_string(&mut self, value: &str) -> Result<(), ProjectionBoundError> {
        if value.len() > self.limits.max_string_bytes {
            return Err(ProjectionBoundError::StringBytes);
        }
        self.retained_bytes = self
            .retained_bytes
            .checked_add(value.len())
            .ok_or(ProjectionBoundError::OutputBytes)?;
        if self.retained_bytes > self.limits.max_output_bytes {
            return Err(ProjectionBoundError::OutputBytes);
        }
        Ok(())
    }
}

fn sanitize_bounded_json(value: &Json, limits: EventProjectionLimits) -> Result<Json, ()> {
    if limits.max_output_bytes == 0 || limits.max_nodes == 0 || limits.max_string_bytes == 0 {
        return Err(());
    }
    let mut projector = BoundedJsonProjector {
        limits,
        nodes: 0,
        retained_bytes: 0,
    };
    let projected = projector.project(value, 0).map_err(|_| ())?;
    if crate::fingerprint::canonical_json_bytes(&projected)
        .map_err(|_| ())?
        .len()
        > limits.max_output_bytes
    {
        return Err(());
    }
    Ok(projected)
}

fn precheck_message_content(
    content: &MessageContent,
    limits: EventProjectionLimits,
) -> Result<(), IneligibilityReason> {
    let mut total = 0usize;
    let mut nodes = 0usize;
    match content {
        MessageContent::Text(text) => {
            check_string(text, limits)?;
            total = text.len();
            nodes = 1;
            if total > limits.max_output_bytes || nodes > limits.max_nodes {
                return Err(IneligibilityReason::ProjectionFailed);
            }
            Ok(())
        }
        MessageContent::Parts(parts) => {
            for part in parts {
                nodes = nodes
                    .checked_add(1)
                    .ok_or(IneligibilityReason::ProjectionFailed)?;
                if nodes > limits.max_nodes {
                    return Err(IneligibilityReason::ProjectionFailed);
                }
                match part {
                    nemo_relay::codec::request::ContentPart::Text { text } => {
                        check_string(text, limits)?;
                        total = total
                            .checked_add(text.len())
                            .ok_or(IneligibilityReason::ProjectionFailed)?;
                    }
                    nemo_relay::codec::request::ContentPart::ImageUrl { image_url } => {
                        check_string(&image_url.url, limits)?;
                        check_optional_string(image_url.detail.as_deref(), limits)?;
                        total = total
                            .checked_add(image_url.url.len())
                            .and_then(|total| {
                                total.checked_add(image_url.detail.as_ref().map_or(0, String::len))
                            })
                            .ok_or(IneligibilityReason::ProjectionFailed)?;
                    }
                }
                if total > limits.max_output_bytes {
                    return Err(IneligibilityReason::ProjectionFailed);
                }
            }
            Ok(())
        }
    }
}

fn precheck_annotated_request(
    request: &AnnotatedLlmRequest,
    limits: EventProjectionLimits,
) -> Result<(), IneligibilityReason> {
    let mut budget = ProjectionBudget::new(limits);
    budget.optional_string(request.model.as_deref())?;
    for message in &request.messages {
        budget.node()?;
        match message {
            Message::System { content, name }
            | Message::Developer { content, name }
            | Message::User { content, name } => {
                budget.message_content(content)?;
                budget.optional_string(name.as_deref())?;
            }
            Message::Assistant {
                content,
                tool_calls,
                name,
            } => {
                if let Some(content) = content {
                    budget.message_content(content)?;
                }
                budget.optional_string(name.as_deref())?;
                if let Some(tool_calls) = tool_calls {
                    for call in tool_calls {
                        budget.node()?;
                        budget.string(&call.id)?;
                        budget.string(&call.call_type)?;
                        budget.string(&call.function.name)?;
                        budget.string(&call.function.arguments)?;
                    }
                }
            }
            Message::Tool {
                content,
                tool_call_id,
            } => {
                budget.message_content(content)?;
                budget.string(tool_call_id)?;
            }
        }
    }
    if let Some(params) = &request.params
        && let Some(stop) = &params.stop
    {
        for value in stop {
            budget.string(value)?;
        }
    }
    if let Some(tools) = &request.tools {
        for tool in tools {
            budget.node()?;
            budget.string(&tool.tool_type)?;
            budget.string(&tool.function.name)?;
            budget.optional_string(tool.function.description.as_deref())?;
            if let Some(parameters) = &tool.function.parameters {
                budget.json(parameters, 0)?;
            }
        }
    }
    if let Some(ToolChoice::Specific(choice)) = &request.tool_choice {
        budget.string(&choice.choice_type)?;
        budget.string(&choice.function.name)?;
    }
    if let Some(format) = &request.response_format {
        budget.optional_string(format.name.as_deref())?;
        if let Some(schema) = &format.schema {
            budget.json(schema, 0)?;
        }
    }
    if let Some(value) = &request.truncation {
        budget.json(value, 0)?;
    }
    if let Some(value) = &request.reasoning {
        budget.json(value, 0)?;
    }
    budget.optional_string(request.service_tier.as_deref())?;
    Ok(())
}

struct ProjectionBudget {
    limits: EventProjectionLimits,
    nodes: usize,
    string_bytes: usize,
}

impl ProjectionBudget {
    fn new(limits: EventProjectionLimits) -> Self {
        Self {
            limits,
            nodes: 0,
            string_bytes: 0,
        }
    }

    fn node(&mut self) -> Result<(), IneligibilityReason> {
        self.nodes = self
            .nodes
            .checked_add(1)
            .ok_or(IneligibilityReason::ProjectionFailed)?;
        if self.nodes > self.limits.max_nodes {
            return Err(IneligibilityReason::ProjectionFailed);
        }
        Ok(())
    }

    fn string(&mut self, value: &str) -> Result<(), IneligibilityReason> {
        self.node()?;
        check_string(value, self.limits)?;
        self.string_bytes = self
            .string_bytes
            .checked_add(value.len())
            .ok_or(IneligibilityReason::ProjectionFailed)?;
        if self.string_bytes > self.limits.max_output_bytes {
            return Err(IneligibilityReason::ProjectionFailed);
        }
        Ok(())
    }

    fn optional_string(&mut self, value: Option<&str>) -> Result<(), IneligibilityReason> {
        value.map_or(Ok(()), |value| self.string(value))
    }

    fn message_content(&mut self, content: &MessageContent) -> Result<(), IneligibilityReason> {
        match content {
            MessageContent::Text(text) => self.string(text),
            MessageContent::Parts(parts) => {
                for part in parts {
                    self.node()?;
                    match part {
                        nemo_relay::codec::request::ContentPart::Text { text } => {
                            self.string(text)?;
                        }
                        nemo_relay::codec::request::ContentPart::ImageUrl { image_url } => {
                            self.string(&image_url.url)?;
                            self.optional_string(image_url.detail.as_deref())?;
                        }
                    }
                }
                Ok(())
            }
        }
    }

    fn json(&mut self, value: &Json, depth: usize) -> Result<(), IneligibilityReason> {
        if depth > self.limits.max_depth {
            return Err(IneligibilityReason::ProjectionFailed);
        }
        self.node()?;
        match value {
            Json::String(value) => self.string(value),
            Json::Array(values) => {
                for value in values {
                    self.json(value, depth + 1)?;
                }
                Ok(())
            }
            Json::Object(values) => {
                for (key, value) in values {
                    self.string(key)?;
                    if !crate::projection::is_sensitive_projection_key(key) {
                        self.json(value, depth + 1)?;
                    }
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

fn check_optional_string(
    value: Option<&str>,
    limits: EventProjectionLimits,
) -> Result<(), IneligibilityReason> {
    value.map_or(Ok(()), |value| check_string(value, limits))
}

fn check_string(value: &str, limits: EventProjectionLimits) -> Result<(), IneligibilityReason> {
    if value.len() > limits.max_string_bytes || value.len() > limits.max_output_bytes {
        Err(IneligibilityReason::ProjectionFailed)
    } else {
        Ok(())
    }
}

fn fingerprint_without_fields<T: Serialize>(value: &T, fields: &[&str]) -> Result<String, ()> {
    let mut value = serde_json::to_value(value).map_err(|_| ())?;
    let object = value.as_object_mut().ok_or(())?;
    for field in fields {
        object.remove(*field);
    }
    crate::fingerprint::fingerprint_json(&value)
}

fn canonical_size_fixed_point(event: &mut CapturedTrajectoryEvent) -> Result<usize, ()> {
    for _ in 0..4 {
        let size = canonical_serialize_bytes(event)?.len();
        if size == event.canonical_size_bytes {
            return Ok(size);
        }
        event.canonical_size_bytes = size;
    }
    let size = canonical_serialize_bytes(event)?.len();
    event.canonical_size_bytes = size;
    Ok(size)
}

#[cfg(test)]
pub(crate) mod test_fixtures {
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use chrono::{TimeZone, Utc};
    use nemo_relay::api::llm::{LlmApiFamily, LlmRequest};
    use nemo_relay::api::runtime::{
        LLM_REPLAY_CONTRACT_VERSION, LlmReplayCall, LlmReplayCapability, LlmReplayTransport,
    };
    use nemo_relay::codec::request::AnnotatedLlmRequest;
    use serde_json::{Map, json};
    use uuid::Uuid;

    use super::{
        ClosedTrajectoryWindow, PENDING_TRAJECTORY_SCHEMA_V1, PendingTrajectoryWindow,
        PersistedTrajectoryTerminalV1, RESPONSE_PROJECTION_SCHEMA_V1, ReplayCapabilityFactsV1,
        RouterResponseProjectionV1, TERMINAL_TRAJECTORY_SCHEMA_V1, TRAJECTORY_SANITIZER_VERSION,
        TrajectoryTerminalStateV1, TrajectoryTrigger,
    };
    use crate::adapter::RouterRequestEnvelope;
    use crate::projection::{
        REQUEST_PROJECTION_SCHEMA_V1, ROUTER_SANITIZER_VERSION, ROUTING_CONTEXT_SCHEMA_V1,
        RouterRequestProjectionV1, RouterRoutingContextProjectionV1, SanitizedAnnotatedLlmRequest,
    };

    struct CountingReplay {
        capability: LlmReplayCapability,
        starts: Arc<AtomicUsize>,
    }

    impl LlmReplayTransport for CountingReplay {
        fn capability(&self) -> &LlmReplayCapability {
            &self.capability
        }

        fn start(&self, _request: LlmRequest) -> nemo_relay::error::Result<LlmReplayCall> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            Ok(LlmReplayCall::new(async { Ok(json!({"ok": true})) }, || {}))
        }
    }

    pub(crate) fn pending_window(anchor_id: Uuid) -> PendingTrajectoryWindow {
        let family = LlmApiFamily::OpenAIChatCompletions;
        let replay_capability_facts =
            ReplayCapabilityFactsV1::from_capability(&LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: family,
                transport_identity: "fixture".to_string(),
            })
            .expect("fixture replay capability should be valid");
        PendingTrajectoryWindow {
            schema: PENDING_TRAJECTORY_SCHEMA_V1.to_string(),
            anchor_id,
            anchor_call_uuid: Uuid::from_u128(2),
            root_uuid: Uuid::from_u128(3),
            owner_uuid: Uuid::from_u128(4),
            owner_path: Vec::new(),
            pool_id: "pool".to_string(),
            anchor_model_revision: "anchor-r1".to_string(),
            process_instance_id: Uuid::from_u128(5),
            project_uuid: Uuid::from_u128(6),
            project_id: "project".to_string(),
            config_generation_id: "config-generation".to_string(),
            policy_version_id: "policy-version".to_string(),
            learning_generation_id: Uuid::from_u128(7),
            request_projection: empty_request_projection(family),
            routing_context_projection: RouterRoutingContextProjectionV1 {
                schema: ROUTING_CONTEXT_SCHEMA_V1.to_string(),
                tenant_policy_hash: "tenant-policy".to_string(),
                agent_policy_hash: "agent-policy".to_string(),
                position_features: BTreeMap::new(),
            },
            normalized_anchor_response: RouterResponseProjectionV1 {
                schema: RESPONSE_PROJECTION_SCHEMA_V1.to_string(),
                sanitizer_version: TRAJECTORY_SANITIZER_VERSION,
                id: Some("response".to_string()),
                model: Some("anchor".to_string()),
                message: None,
                tool_calls: None,
                finish_reason: None,
                usage: None,
                semantic_response_fingerprint: "response-fingerprint".to_string(),
            },
            replay_capability_facts,
            candidate_facts: Vec::new(),
            requested_progress: 1,
            opened_at: Utc.timestamp_opt(1_700_000_000, 0).unwrap(),
            deadline_at: Utc.timestamp_opt(1_700_000_300, 0).unwrap(),
        }
    }

    pub(crate) fn terminal_window(anchor_id: Uuid) -> PersistedTrajectoryTerminalV1 {
        PersistedTrajectoryTerminalV1 {
            schema: TERMINAL_TRAJECTORY_SCHEMA_V1.to_string(),
            pending: pending_window(anchor_id),
            state: TrajectoryTerminalStateV1::Closed {
                trigger: TrajectoryTrigger::ProgressReached,
            },
            events: Vec::new(),
            observed_progress: 1,
            is_partial: false,
            promotion_eligible: false,
            closed_at: Utc.timestamp_opt(1_700_000_010, 0).unwrap(),
            diagnostics: Vec::new(),
        }
    }

    pub(crate) fn closed_window(
        anchor_id: Uuid,
    ) -> (Arc<ClosedTrajectoryWindow>, Arc<AtomicUsize>) {
        let starts = Arc::new(AtomicUsize::new(0));
        let replay_transport: Arc<dyn LlmReplayTransport> = Arc::new(CountingReplay {
            capability: LlmReplayCapability {
                contract_version: LLM_REPLAY_CONTRACT_VERSION,
                api_family: LlmApiFamily::OpenAIChatCompletions,
                transport_identity: "fixture".to_string(),
            },
            starts: starts.clone(),
        });
        let request_envelope = RouterRequestEnvelope {
            family: LlmApiFamily::OpenAIChatCompletions,
            original_request: LlmRequest {
                headers: Map::new(),
                content: json!({"model": "anchor", "messages": []}),
            },
            normalized_request: AnnotatedLlmRequest {
                messages: Vec::new(),
                model: Some("anchor".to_string()),
                params: None,
                tools: None,
                tool_choice: None,
                response_format: None,
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
                extra: Map::new(),
            },
            ordered_instructions: Vec::new(),
            response_format: None,
            response_schema_fingerprint: None,
        };
        (
            Arc::new(ClosedTrajectoryWindow {
                pending: pending_window(anchor_id),
                request_envelope,
                replay_transport,
                eligible_candidates: Vec::new(),
                events: Vec::new(),
                observed_progress: 1,
                trigger: TrajectoryTrigger::ProgressReached,
                is_partial: false,
                promotion_eligible: false,
                closed_at: Utc.timestamp_opt(1_700_000_010, 0).unwrap(),
            }),
            starts,
        )
    }

    fn empty_request_projection(family: LlmApiFamily) -> RouterRequestProjectionV1 {
        RouterRequestProjectionV1 {
            schema: REQUEST_PROJECTION_SCHEMA_V1.to_string(),
            family,
            normalized_request: SanitizedAnnotatedLlmRequest {
                messages: Vec::new(),
                model: Some("anchor".to_string()),
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
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use chrono::{DateTime, Timelike, Utc};
    use nemo_relay::api::event::{
        BaseEvent, CategoryProfile, Event, EventCategory, MarkEvent, ScopeCategory, ScopeEvent,
    };
    use nemo_relay::api::llm::{LlmApiFamily, LlmCallRole, LlmRequest, LlmTrajectoryScopeSnapshot};
    use nemo_relay::api::scope::ScopeType;
    use nemo_relay::codec::openai_chat::OpenAIChatCodec;
    use nemo_relay::codec::traits::LlmCodec;
    use serde_json::{Map, json};
    use uuid::{Uuid, Version};

    use super::{
        EventProjectionLimits, MAX_TRAJECTORY_DIAGNOSTIC_BYTES, MAX_TRAJECTORY_DIAGNOSTICS,
        PersistedCandidateFactV1, PersistedTrajectoryTerminalV1, ProjectedTrajectoryEvent,
        ReplayCapabilityFactsV1, TrajectoryDiagnosticV1, TrajectoryIdentity,
        TrajectoryRejectionReason, project_anchor_response, project_captured_event,
        project_owner_path, truncate_utc_to_milliseconds,
    };
    use crate::config::{CandidateCapabilities, CandidateConfig};
    use crate::eligibility::IneligibilityReason;
    use crate::preflight::EligibleCandidate;
    use crate::response_validator::compile_candidate_response_contracts;

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

    assert_not_impl!(super::ClosedTrajectoryWindow: serde::Serialize);
    assert_not_impl!(super::TrajectoryWindowSeed: serde::Serialize);
    assert_not_impl!(Arc<dyn nemo_relay::api::runtime::LlmReplayTransport>: serde::Serialize);

    #[test]
    fn persisted_window_clock_is_normalized_before_hashing() {
        let value = DateTime::<Utc>::from_timestamp(1_700_000_000, 123_456_789).unwrap();
        let normalized = truncate_utc_to_milliseconds(value);
        assert_eq!(normalized.timestamp_millis(), value.timestamp_millis());
        assert_eq!(normalized.nanosecond(), 123_000_000);
    }

    #[test]
    fn owner_path_projection_bounds_depth_and_aggregate_allocation_before_cloning() {
        let path = (0..64)
            .map(|index| LlmTrajectoryScopeSnapshot {
                uuid: Uuid::from_u128(index + 1),
                name: "scope".into(),
                scope_type: ScopeType::Function,
            })
            .collect::<Vec<_>>();
        assert_eq!(project_owner_path(&path, 16 * 1024).unwrap().len(), 64);

        let mut too_deep = path.clone();
        too_deep.push(LlmTrajectoryScopeSnapshot {
            uuid: Uuid::from_u128(65),
            name: "scope".into(),
            scope_type: ScopeType::Function,
        });
        assert_eq!(
            project_owner_path(&too_deep, 16 * 1024),
            Err(crate::eligibility::IneligibilityReason::ProjectionFailed)
        );

        let aggregate = vec![
            LlmTrajectoryScopeSnapshot {
                uuid: Uuid::from_u128(1),
                name: "a".repeat(500),
                scope_type: ScopeType::Agent,
            },
            LlmTrajectoryScopeSnapshot {
                uuid: Uuid::from_u128(2),
                name: "b".repeat(500),
                scope_type: ScopeType::Function,
            },
        ];
        assert_eq!(
            project_owner_path(&aggregate, 1024),
            Err(crate::eligibility::IneligibilityReason::ProjectionFailed)
        );
    }

    #[test]
    fn ephemeral_identity_is_uuid_v7_and_pool_scoped() {
        let generated = TrajectoryIdentity::for_pools(None, ["alpha", "beta"]);
        assert_eq!(generated.project_id, generated.project_uuid.to_string());
        assert_eq!(
            generated.process_instance_id.get_version(),
            Some(Version::SortRand)
        );
        assert_eq!(
            generated.project_uuid.get_version(),
            Some(Version::SortRand)
        );
        assert_eq!(generated.policy_version_ids.len(), 2);
        assert_eq!(
            generated.policy_version_id("alpha"),
            Some("e4ded49f128cbb64216b2569a681c758c43de644c50eac6ab17902c9c403a656")
        );
        assert_ne!(
            generated.policy_version_id("alpha"),
            generated.policy_version_id("beta")
        );
        assert_eq!(generated.learning_generation_ids.len(), 2);
        assert_eq!(
            generated
                .learning_generation_id("alpha")
                .unwrap()
                .get_version(),
            Some(Version::SortRand)
        );
        assert_ne!(
            generated.learning_generation_id("alpha"),
            generated.learning_generation_id("beta")
        );

        let regenerated = TrajectoryIdentity::for_pools(None, ["beta", "alpha"]);
        assert_eq!(
            generated.policy_version_id("alpha"),
            regenerated.policy_version_id("alpha")
        );
        assert_eq!(
            generated.policy_version_id("beta"),
            regenerated.policy_version_id("beta")
        );

        let configured = TrajectoryIdentity::for_pools(Some("deployment"), ["alpha"]);
        assert_eq!(configured.project_id, "deployment");
        assert_ne!(configured.project_uuid, generated.project_uuid);
    }

    #[test]
    fn persisted_terminal_caps_diagnostic_count() {
        let pending = super::test_fixtures::pending_window(Uuid::from_u128(1));
        let closed_at = pending.deadline_at;
        let diagnostics = (0..MAX_TRAJECTORY_DIAGNOSTICS + 2)
            .map(|index| TrajectoryDiagnosticV1 {
                code: format!("code-{index}"),
                message: format!("message-{index}"),
            })
            .collect();

        let terminal = PersistedTrajectoryTerminalV1::rejected(
            pending,
            Vec::new(),
            0,
            TrajectoryRejectionReason::Overflow,
            closed_at,
            diagnostics,
        );

        assert_eq!(terminal.diagnostics.len(), MAX_TRAJECTORY_DIAGNOSTICS);
        assert_eq!(terminal.diagnostics.last().unwrap().code, "code-7");
    }

    #[test]
    fn persisted_terminal_caps_diagnostic_fields_on_utf8_boundaries() {
        let pending = super::test_fixtures::pending_window(Uuid::from_u128(1));
        let closed_at = pending.deadline_at;
        let terminal = PersistedTrajectoryTerminalV1::rejected(
            pending,
            Vec::new(),
            0,
            TrajectoryRejectionReason::Overflow,
            closed_at,
            vec![TrajectoryDiagnosticV1 {
                code: format!("a{}", "é".repeat(MAX_TRAJECTORY_DIAGNOSTIC_BYTES)),
                message: "m".repeat(MAX_TRAJECTORY_DIAGNOSTIC_BYTES + 1),
            }],
        );

        let diagnostic = &terminal.diagnostics[0];
        assert_eq!(diagnostic.code.len(), MAX_TRAJECTORY_DIAGNOSTIC_BYTES - 1);
        assert!(diagnostic.code.is_char_boundary(diagnostic.code.len()));
        assert_eq!(diagnostic.message.len(), MAX_TRAJECTORY_DIAGNOSTIC_BYTES);
    }

    #[test]
    fn all_authoritative_response_codecs_produce_safe_stable_projections() {
        let fixtures = [
            (
                LlmApiFamily::OpenAIChatCompletions,
                json!({
                    "id": "chatcmpl-1",
                    "model": "chat-anchor",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "chat answer"},
                        "finish_reason": "stop"
                    }],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6},
                    "headers": {"authorization": "Bearer must-not-serialize"},
                    "vendor_extra": "must-not-serialize"
                }),
                "26482f814bfef75dcb7ed6449f48e379a4c374ca4f8a30210cf9546f619adeaa",
            ),
            (
                LlmApiFamily::OpenAIResponses,
                json!({
                    "id": "resp-1",
                    "object": "response",
                    "status": "completed",
                    "model": "responses-anchor",
                    "output": [{
                        "type": "message",
                        "id": "message-1",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": "responses answer", "annotations": []}]
                    }],
                    "usage": {"input_tokens": 5, "output_tokens": 3, "total_tokens": 8},
                    "response_headers": {"set-cookie": "must-not-serialize"}
                }),
                "ab9c9237a716eafc511347ca4bf57d6a54aadf2ff8519846344fd68433509533",
            ),
            (
                LlmApiFamily::AnthropicMessages,
                json!({
                    "id": "msg-1",
                    "type": "message",
                    "role": "assistant",
                    "model": "anthropic-anchor",
                    "content": [{"type": "text", "text": "anthropic answer"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": null,
                    "usage": {"input_tokens": 6, "output_tokens": 4},
                    "api_key": "must-not-serialize"
                }),
                "d55e74cd06f3370930418d055c3e55960ca3b593cb4a80b8c4a19b54d2eaa10c",
            ),
        ];

        for (family, response, expected_fingerprint) in fixtures {
            let first = project_anchor_response(family, &response, 64 * 1024).unwrap();
            let second = project_anchor_response(family, &response, 64 * 1024).unwrap();
            assert_eq!(first, second);
            assert_eq!(first.semantic_response_fingerprint, expected_fingerprint);
            let serialized = serde_json::to_string(&first).unwrap();
            assert!(!serialized.contains("must-not-serialize"));
            assert!(!serialized.contains("api_specific"));
            assert!(!serialized.contains("extra"));
        }
    }

    #[test]
    fn response_tool_arguments_are_sanitized_before_fingerprinting() {
        let response = json!({
            "id": "chatcmpl-tool",
            "model": "chat-anchor",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "arguments": "{\"api_key\":\"must-not-serialize\",\"query\":\"safe\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let projection =
            project_anchor_response(LlmApiFamily::OpenAIChatCompletions, &response, 64 * 1024)
                .unwrap();
        let serialized = serde_json::to_string(&projection).unwrap();
        assert!(!serialized.contains("api_key"));
        assert!(!serialized.contains("must-not-serialize"));
        assert!(serialized.contains("query"));
        assert!(serialized.contains("safe"));
    }

    #[test]
    fn replay_and_decoding_fingerprints_exclude_executable_and_model_state() {
        for transport_identity in [
            "gateway:nvapi-abcdefghijklmnopqrstuvwxyz0123456789",
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
            let wrapped_secret = ReplayCapabilityFactsV1::from_capability(
                &nemo_relay::api::runtime::LlmReplayCapability {
                    contract_version: nemo_relay::api::runtime::LLM_REPLAY_CONTRACT_VERSION,
                    api_family: LlmApiFamily::OpenAIChatCompletions,
                    transport_identity: transport_identity.to_string(),
                },
            );
            assert_eq!(
                wrapped_secret.err(),
                Some(IneligibilityReason::ReplayIdentityInvalid)
            );
        }

        let replay = ReplayCapabilityFactsV1::from_capability(
            &nemo_relay::api::runtime::LlmReplayCapability {
                contract_version: nemo_relay::api::runtime::LLM_REPLAY_CONTRACT_VERSION,
                api_family: LlmApiFamily::OpenAIChatCompletions,
                transport_identity: "host-policy".to_string(),
            },
        )
        .unwrap();
        let replay_json = serde_json::to_string(&replay).unwrap();
        assert_eq!(replay.capability_fingerprint.len(), 64);
        for forbidden in ["endpoint", "authorization", "headers", "callback"] {
            assert!(!replay_json.contains(forbidden));
        }

        let request = super::test_fixtures::pending_window(Uuid::from_u128(50)).request_projection;
        let candidate = |model: &str| {
            EligibleCandidate::new(
                CandidateConfig {
                    id: "candidate".to_string(),
                    model: model.to_string(),
                    model_revision: "revision".to_string(),
                    cost_rank: 1,
                    max_context_tokens: None,
                    capabilities: CandidateCapabilities {
                        tools: true,
                        multimodal_input: false,
                        structured_output: false,
                        reasoning_controls: false,
                        unknown_fields: BTreeMap::new(),
                    },
                    unknown_fields: BTreeMap::new(),
                },
                LlmRequest {
                    headers: Map::from_iter([(
                        "authorization".to_string(),
                        json!("Bearer must-not-serialize"),
                    )]),
                    content: json!({"model": model, "messages": []}),
                },
                Arc::new(crate::response_validator::CandidateResponseContractsV1::empty_for_test()),
            )
        };
        let first =
            PersistedCandidateFactV1::from_eligible(&candidate("cheap-a"), &request).unwrap();
        let second =
            PersistedCandidateFactV1::from_eligible(&candidate("cheap-b"), &request).unwrap();
        assert_eq!(first.decoding_fingerprint, second.decoding_fingerprint);
        let serialized = serde_json::to_string(&first).unwrap();
        assert!(!serialized.contains("must-not-serialize"));
        assert!(!serialized.contains("authorization"));
    }

    #[test]
    fn decoding_fingerprint_uses_full_contracts_without_persisting_them() {
        const DRAFT: &str = "https://json-schema.org/draft/2020-12/schema";
        const MAX_VALIDATION_BYTES: usize = 64 * 1024;

        let contracts = |tool_secret: &str, response_secret: &str| {
            let request = OpenAIChatCodec
                .decode(&LlmRequest {
                    headers: Map::new(),
                    content: json!({
                        "model": "anchor",
                        "messages": [{"role": "user", "content": "task"}],
                        "tools": [{
                            "type": "function",
                            "function": {
                                "name": "lookup",
                                "parameters": {
                                    "$schema": DRAFT,
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
                                    "$schema": DRAFT,
                                    "type": "object",
                                    "properties": {
                                        "answer": {"type": "string"},
                                        "api_key": {
                                            "type": "string",
                                            "const": response_secret
                                        }
                                    },
                                    "required": ["answer"],
                                    "additionalProperties": false
                                }
                            }
                        }
                    }),
                })
                .unwrap();
            Arc::new(compile_candidate_response_contracts(&request, MAX_VALIDATION_BYTES).unwrap())
        };
        let candidate = |response_contracts| {
            EligibleCandidate::new(
                CandidateConfig {
                    id: "candidate".to_string(),
                    model: "candidate-model".to_string(),
                    model_revision: "revision".to_string(),
                    cost_rank: 1,
                    max_context_tokens: None,
                    capabilities: CandidateCapabilities {
                        tools: true,
                        multimodal_input: false,
                        structured_output: true,
                        reasoning_controls: false,
                        unknown_fields: BTreeMap::new(),
                    },
                    unknown_fields: BTreeMap::new(),
                },
                LlmRequest {
                    headers: Map::from_iter([(
                        "authorization".to_string(),
                        json!("Bearer transport-secret"),
                    )]),
                    content: json!({"model": "candidate-model", "messages": []}),
                },
                response_contracts,
            )
        };
        let request = super::test_fixtures::pending_window(Uuid::from_u128(51)).request_projection;
        let baseline = PersistedCandidateFactV1::from_eligible(
            &candidate(contracts("tool-secret-a", "response-secret-a")),
            &request,
        )
        .unwrap();
        let changed_tool = PersistedCandidateFactV1::from_eligible(
            &candidate(contracts("tool-secret-b", "response-secret-a")),
            &request,
        )
        .unwrap();
        let changed_response = PersistedCandidateFactV1::from_eligible(
            &candidate(contracts("tool-secret-a", "response-secret-b")),
            &request,
        )
        .unwrap();

        assert_ne!(
            baseline.decoding_fingerprint,
            changed_tool.decoding_fingerprint
        );
        assert_ne!(
            baseline.decoding_fingerprint,
            changed_response.decoding_fingerprint
        );
        let serialized =
            serde_json::to_string(&[baseline, changed_tool, changed_response]).unwrap();
        for forbidden in [
            "tool-secret-a",
            "tool-secret-b",
            "response-secret-a",
            "response-secret-b",
            "transport-secret",
            "api_key",
            "properties",
        ] {
            assert!(!serialized.contains(forbidden));
        }
    }

    #[test]
    fn event_projection_removes_nested_credentials_and_transport_containers() {
        let event = Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .name("compaction")
                .uuid(Uuid::from_u128(10))
                .parent_uuid(Uuid::from_u128(9))
                .data(json!({
                    "safe": true,
                    "nested": {"Client-Secret": "must-not-serialize"},
                    "request_headers": {"x-api-key": "must-not-serialize"}
                }))
                .metadata(json!({
                    "hook_event_name": "precompact",
                    "cookie_jar": {"session": "must-not-serialize"}
                }))
                .build(),
            None,
            Some(CategoryProfile::default()),
        ));
        let ProjectedTrajectoryEvent::Captured(projected) = project_captured_event(
            7,
            &event,
            EventProjectionLimits::for_window_bytes(64 * 1024),
        ) else {
            panic!("bounded event should be retained")
        };
        let serialized = serde_json::to_string(projected.as_ref()).unwrap();
        assert!(!serialized.contains("must-not-serialize"));
        assert!(!serialized.contains("request_headers"));
        assert!(!serialized.contains("cookie_jar"));
        assert!(serialized.contains("safe"));
        assert_eq!(projected.hook_event_name(), Some("precompact"));
        assert_eq!(projected.canonical_payload_hash.len(), 64);
        assert_eq!(
            projected.canonical_size_bytes,
            crate::fingerprint::canonical_serialize_bytes(projected.as_ref())
                .unwrap()
                .len()
        );

        let ProjectedTrajectoryEvent::Captured(second) = project_captured_event(
            8,
            &event,
            EventProjectionLimits::for_window_bytes(64 * 1024),
        ) else {
            panic!("bounded event should be retained")
        };
        assert_eq!(
            projected.canonical_payload_hash,
            second.canonical_payload_hash
        );
    }

    #[test]
    fn llm_wire_payload_and_provider_extras_never_enter_captured_data() {
        let event = Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .name("provider.call")
                .uuid(Uuid::from_u128(20))
                .parent_uuid(Uuid::from_u128(19))
                .data(json!({
                    "headers": {"authorization": "must-not-serialize"},
                    "content": {"vendor_response_blob": "must-not-serialize"}
                }))
                .build(),
            ScopeCategory::End,
            Vec::new(),
            EventCategory::llm(),
            Some(CategoryProfile::default()),
        ));
        let ProjectedTrajectoryEvent::Captured(projected) = project_captured_event(
            9,
            &event,
            EventProjectionLimits::for_window_bytes(64 * 1024),
        ) else {
            panic!("bounded event should be retained")
        };
        assert!(projected.data.is_none());
        let serialized = serde_json::to_string(projected.as_ref()).unwrap();
        assert!(!serialized.contains("must-not-serialize"));
        assert!(!serialized.contains("vendor_response_blob"));
    }

    #[test]
    fn event_projection_returns_structural_marker_on_every_bound() {
        let event = Event::Mark(MarkEvent::new(
            BaseEvent::builder()
                .name("large")
                .uuid(Uuid::from_u128(10))
                .parent_uuid(Uuid::from_u128(9))
                .data(json!({"nested": {"value": "too large"}}))
                .build(),
            None,
            None,
        ));
        let limits = [
            EventProjectionLimits {
                max_depth: 1,
                max_nodes: 64,
                max_string_bytes: 64,
                max_output_bytes: 1024,
            },
            EventProjectionLimits {
                max_depth: 64,
                max_nodes: 2,
                max_string_bytes: 64,
                max_output_bytes: 1024,
            },
            EventProjectionLimits {
                max_depth: 64,
                max_nodes: 64,
                max_string_bytes: 4,
                max_output_bytes: 1024,
            },
            EventProjectionLimits {
                max_depth: 64,
                max_nodes: 64,
                max_string_bytes: 64,
                max_output_bytes: 16,
            },
        ];
        for limits in limits {
            let ProjectedTrajectoryEvent::Oversized(marker) =
                project_captured_event(11, &event, limits)
            else {
                panic!("event must fail closed to an oversize marker")
            };
            assert_eq!(marker.ingest_seq, 11);
            assert_eq!(marker.event_uuid, Uuid::from_u128(10));
            assert_eq!(marker.parent_uuid, Some(Uuid::from_u128(9)));
            assert_eq!(marker.scope_type, None);
        }
    }

    #[test]
    fn oversized_marker_preserves_evaluator_isolation_fields() {
        let event = Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .name("router-evaluator")
                .uuid(Uuid::from_u128(30))
                .parent_uuid(Uuid::from_u128(29))
                .build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::from(ScopeType::Evaluator),
            None,
        ));
        let ProjectedTrajectoryEvent::Oversized(marker) = project_captured_event(
            12,
            &event,
            EventProjectionLimits {
                max_depth: 64,
                max_nodes: 64,
                max_string_bytes: 4,
                max_output_bytes: 1024,
            },
        ) else {
            panic!("evaluator event must exceed the injected string bound")
        };
        assert_eq!(marker.scope_type, Some(ScopeType::Evaluator));
        assert!(marker.is_internal_router_event());

        let embedder = Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .name("router-embedder")
                .uuid(Uuid::from_u128(31))
                .parent_uuid(Uuid::from_u128(29))
                .build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::from(ScopeType::Embedder),
            None,
        ));
        let ProjectedTrajectoryEvent::Oversized(marker) = project_captured_event(
            13,
            &embedder,
            EventProjectionLimits {
                max_depth: 64,
                max_nodes: 64,
                max_string_bytes: 4,
                max_output_bytes: 1024,
            },
        ) else {
            panic!("embedder event must exceed the injected string bound")
        };
        assert_eq!(marker.scope_type, Some(ScopeType::Embedder));
        assert!(marker.is_internal_router_event());

        let mut llm_profile = CategoryProfile::default();
        llm_profile.set_llm_call_role(LlmCallRole::Judge);
        let llm = Event::Scope(ScopeEvent::new(
            BaseEvent::builder()
                .name("router-judge")
                .uuid(Uuid::from_u128(32))
                .parent_uuid(Uuid::from_u128(31))
                .build(),
            ScopeCategory::Start,
            Vec::new(),
            EventCategory::llm(),
            Some(llm_profile),
        ));
        let ProjectedTrajectoryEvent::Oversized(marker) = project_captured_event(
            13,
            &llm,
            EventProjectionLimits {
                max_depth: 64,
                max_nodes: 64,
                max_string_bytes: 4,
                max_output_bytes: 1024,
            },
        ) else {
            panic!("judge event must exceed the injected string bound")
        };
        assert_eq!(marker.scope_type, Some(ScopeType::Llm));
        assert_eq!(marker.call_role, Some(LlmCallRole::Judge));
        assert!(marker.is_internal_router_event());
    }

    #[test]
    fn persisted_payloads_round_trip_and_memory_only_window_does_not_start_replay() {
        let anchor_id = Uuid::from_u128(1);
        let pending = super::test_fixtures::pending_window(anchor_id);
        let terminal = super::test_fixtures::terminal_window(anchor_id);
        let pending_round_trip: super::PendingTrajectoryWindow =
            serde_json::from_slice(&pending.canonical_bytes().unwrap()).unwrap();
        let terminal_round_trip: super::PersistedTrajectoryTerminalV1 =
            serde_json::from_slice(&terminal.canonical_bytes().unwrap()).unwrap();
        assert_eq!(pending_round_trip, pending);
        assert_eq!(terminal_round_trip, terminal);
        assert_eq!(pending_round_trip.policy_version_id, "policy-version");
        assert_eq!(pending.payload_hash().unwrap().len(), 64);
        assert_eq!(terminal.payload_hash().unwrap().len(), 64);

        let (closed, starts) = super::test_fixtures::closed_window(anchor_id);
        assert_eq!(closed.anchor_id(), anchor_id);
        assert_eq!(starts.load(std::sync::atomic::Ordering::SeqCst), 0);
        drop(Arc::clone(&closed));
        assert_eq!(starts.load(std::sync::atomic::Ordering::SeqCst), 0);
    }
}
