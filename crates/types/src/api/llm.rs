// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared LLM data types.

use std::collections::BTreeMap;

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::Json;
use crate::api::event::PendingMarkSpec;
use crate::api::scope::ScopeType;
use crate::codec::optimization::LlmOptimizationContribution;
use crate::codec::request::AnnotatedLlmRequest;

/// Provider API family used by a managed LLM call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LlmApiFamily {
    /// OpenAI Chat Completions request and response shapes.
    #[serde(rename = "openai_chat_completions")]
    OpenAIChatCompletions,
    /// OpenAI Responses request and response shapes.
    #[serde(rename = "openai_responses")]
    OpenAIResponses,
    /// Anthropic Messages request and response shapes.
    #[serde(rename = "anthropic_messages")]
    AnthropicMessages,
}

/// Execution role of a managed LLM call.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmCallRole {
    /// A production agent call that can contribute to its trajectory.
    Primary,
    /// A delayed counterfactual call used for model comparison.
    Shadow,
    /// An evaluator call used to judge a candidate response.
    Judge,
}

impl LlmCallRole {
    /// Return the stable lowercase value used in ATOF category profiles.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Primary => "primary",
            Self::Shadow => "shadow",
            Self::Judge => "judge",
        }
    }
}

/// Frozen identifying information for one scope in an LLM trajectory path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmTrajectoryScopeSnapshot {
    /// Stable UUID of the captured scope.
    pub uuid: Uuid,
    /// Human-readable scope name captured with the path.
    pub name: String,
    /// Semantic scope category captured with the path.
    pub scope_type: ScopeType,
}

/// Serializable, router-neutral context frozen for one managed LLM call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LlmExecutionContextSnapshot {
    /// Stable UUID of the physical managed LLM call.
    pub call_uuid: Uuid,
    /// UUID of the implicit root on the captured scope stack.
    pub root_uuid: Uuid,
    /// UUID of the call's immediate active parent scope.
    pub parent_uuid: Uuid,
    /// UUID of the deepest active Agent scope that owns the trajectory.
    pub trajectory_owner_uuid: Uuid,
    /// Captured owner-to-parent path, inclusive and in ancestry order.
    pub trajectory_owner_path: Vec<LlmTrajectoryScopeSnapshot>,
    /// Explicit provider API family for the call.
    pub api_family: LlmApiFamily,
    /// Explicit execution role for the call.
    pub call_role: LlmCallRole,
    /// LLM behavior attributes frozen for the call.
    pub attributes: LlmAttributes,
    /// Optional normalized, stable tenant routing identity.
    pub tenant_id: Option<String>,
    /// Optional normalized, stable agent routing identity.
    pub agent_id: Option<String>,
    /// Caller-supplied non-secret routing metadata safe for middleware.
    pub sanitized_metadata: BTreeMap<String, Json>,
}

bitflags! {
    /// Bitflags that modify LLM-call behavior and observability.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct LlmAttributes: u32 {
        /// Marks the request as stateful from the runtime's perspective.
        const STATEFUL = 0b01;
        /// Marks the request as streaming.
        const STREAMING = 0b10;
    }
}

/// JSON-shaped LLM request payload passed through the runtime.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmRequest {
    /// Provider-specific request headers.
    pub headers: serde_json::Map<String, Json>,
    /// Provider-specific request body.
    pub content: Json,
}

/// Result of an LLM request intercept that can schedule lifecycle marks.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LlmRequestInterceptOutcome {
    /// Rewritten provider request when no request codec is active.
    ///
    /// With a request codec, callbacks may rewrite `headers`, but `content`
    /// is read-only and provider-body changes must be made through
    /// [`Self::annotated_request`].
    pub request: LlmRequest,
    /// Optional normalized request annotation to carry forward.
    ///
    /// This is required and authoritative for provider content when a request
    /// codec is active. It remains optional when no request codec is active.
    #[serde(default)]
    pub annotated_request: Option<AnnotatedLlmRequest>,
    /// Ordered marks to emit after Relay creates and starts the LLM scope.
    #[serde(default)]
    pub pending_marks: Vec<PendingMarkSpec>,
    /// Ordered plugin-neutral optimization evidence for this LLM call.
    #[serde(default)]
    pub optimization_contributions: Vec<LlmOptimizationContribution>,
}

impl LlmRequestInterceptOutcome {
    /// Create an outcome without pending marks.
    pub fn new(request: LlmRequest, annotated_request: Option<AnnotatedLlmRequest>) -> Self {
        Self {
            request,
            annotated_request,
            pending_marks: Vec::new(),
            optimization_contributions: Vec::new(),
        }
    }

    /// Append one pending mark while preserving interceptor order.
    #[must_use]
    pub fn with_pending_mark(mut self, mark: PendingMarkSpec) -> Self {
        self.pending_marks.push(mark);
        self
    }

    /// Append one optimization contribution while preserving interceptor order.
    #[must_use]
    pub fn with_optimization_contribution(
        mut self,
        contribution: LlmOptimizationContribution,
    ) -> Self {
        self.optimization_contributions.push(contribution);
        self
    }
}

impl From<LlmRequest> for LlmRequestInterceptOutcome {
    fn from(request: LlmRequest) -> Self {
        Self::new(request, None)
    }
}

impl From<(LlmRequest, AnnotatedLlmRequest)> for LlmRequestInterceptOutcome {
    fn from((request, annotated_request): (LlmRequest, AnnotatedLlmRequest)) -> Self {
        Self::new(request, Some(annotated_request))
    }
}

impl From<(LlmRequest, Option<AnnotatedLlmRequest>)> for LlmRequestInterceptOutcome {
    fn from((request, annotated_request): (LlmRequest, Option<AnnotatedLlmRequest>)) -> Self {
        Self::new(request, annotated_request)
    }
}
