# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Type declarations for Shadow Router configuration helpers."""

from dataclasses import dataclass
from typing import Literal, TypeAlias, TypedDict

from nemo_relay import Json, JsonObject, UnsupportedBehavior
from nemo_relay.llm import LlmApiFamily as LlmApiFamily
from nemo_relay.llm import LlmCallRole as LlmCallRole
from nemo_relay.llm import LlmExecutionContext as LlmExecutionContext
from nemo_relay.llm import LlmReplayDescriptor as LlmReplayDescriptor
from nemo_relay.llm import LlmReplayFactory as LlmReplayFactory

RouterMode: TypeAlias = Literal["off", "shadow", "recommend", "active"]
OutcomeMatcherEventKind: TypeAlias = Literal["scope_end", "mark"]
OutcomeTerminalStatus: TypeAlias = Literal["ok", "error", "unset"]
OutcomeDisposition: TypeAlias = Literal["success", "failure", "ignore"]

class ConfigDiagnostic(TypedDict, total=False):
    """One Router configuration validation diagnostic."""

    level: Literal["warning", "error"]
    code: str
    message: str
    component: str
    field: str

class ConfigReport(TypedDict):
    """Validation report for Router configuration."""

    diagnostics: list[ConfigDiagnostic]

class _SerializableConfig:
    def to_dict(self) -> JsonObject: ...

@dataclass(slots=True)
class ConfigPolicy(_SerializableConfig):
    """Policy for unsupported Router configuration."""

    unknown_component: UnsupportedBehavior = ...
    unknown_field: UnsupportedBehavior = ...
    unsupported_value: UnsupportedBehavior = ...

@dataclass(slots=True)
class EmbedderConfig(_SerializableConfig):
    """One OpenAI-compatible embedding provider profile."""

    id: str
    base_url: str
    model: str
    provider_revision: str
    dimensions: int
    timeout_ms: int
    api_key_env: str | None = ...
    max_in_flight: int = ...
    batch_size: int = ...

@dataclass(slots=True)
class LearningConfig(_SerializableConfig):
    """Versioned embedding association or complete routing policy."""

    embedder: str
    version: int = ...
    top_k: int | None = ...
    radius: float | None = ...
    min_points: int | None = ...
    min_independent_roots: int | None = ...
    min_effective_samples: float | None = ...
    min_coverage: float | None = ...
    time_decay_half_life_seconds: float | None = ...
    prior_success: float | None = ...
    prior_failure: float | None = ...
    familywise_credible_level: float | None = ...
    promotion_lower_bound: float | None = ...
    retention_lower_bound: float | None = ...
    holdout_probability: float | None = ...
    active_canary_fraction: float | None = ...

@dataclass(slots=True)
class OutcomeMatcher(_SerializableConfig):
    """One exact sanitized event matcher for actual outcomes."""

    event_kind: OutcomeMatcherEventKind
    category: str
    name: str
    terminal_status: OutcomeTerminalStatus
    metadata_equals: dict[str, Json] = ...

@dataclass(slots=True)
class OutcomeConfig(_SerializableConfig):
    """Complete version-1 actual-outcome and Active-look policy."""

    success_matchers: list[OutcomeMatcher | JsonObject]
    failure_matchers: list[OutcomeMatcher | JsonObject]
    completion_disposition: OutcomeDisposition
    error_disposition: OutcomeDisposition
    tool_failure_disposition: OutcomeDisposition
    end_of_run_disposition: OutcomeDisposition
    max_attribution_seconds: int
    actual_outcome_half_life_seconds: int
    anchor_shadow_half_life_seconds: int
    relearning_cooloff_seconds: int
    min_treatment_roots: int
    min_control_roots: int
    min_treatment_effective_weight: float
    min_control_effective_weight: float
    noninferiority_margin: float
    noninferiority_probability: float
    rollback_probability: float
    outcome_evaluation_batch_size: int
    max_canary_roots: int
    authorization_ttl_seconds: int
    version: int = ...

@dataclass(slots=True)
class PoolSelectorConfig(_SerializableConfig):
    """Predicates matched against frozen V2 call facts."""

    tenant_ids: list[str] | None = ...
    agent_ids: list[str] | None = ...
    owner_scope_types: list[str] | None = ...
    metadata_equals: dict[str, Json] = ...
    scope_path_patterns: list[str] | None = ...

@dataclass(slots=True)
class LookaheadConfig(_SerializableConfig):
    """Future-local trajectory window limits."""

    primary_llm_completions: int = ...
    deadline_seconds: int = ...
    lifecycle_presets: list[str] = ...
    max_events_per_window: int = ...
    max_bytes_per_window: int = ...

@dataclass(slots=True)
class ConcurrencyConfig(_SerializableConfig):
    """Independent per-pool provider and pending-work limits."""

    shadow: int
    judge: int
    max_pending: int = ...

@dataclass(slots=True)
class CandidateCapabilities(_SerializableConfig):
    """Capabilities explicitly supported by a candidate model."""

    tools: bool = ...
    multimodal_input: bool = ...
    structured_output: bool = ...
    reasoning_controls: bool = ...

@dataclass(slots=True)
class CandidateConfig(_SerializableConfig):
    """One cheaper candidate model in a routing pool."""

    id: str
    model: str
    model_revision: str
    cost_rank: int
    max_context_tokens: int | None = ...
    capabilities: CandidateCapabilities | JsonObject = ...

@dataclass(slots=True)
class CanonicalizerConfig(_SerializableConfig):
    """Limits for deterministic semantic request canonicalization."""

    version: int = ...
    max_instruction_bytes: int = ...
    max_task_bytes: int = ...
    max_context_messages: int = ...
    max_context_bytes: int = ...
    max_position_features_bytes: int = ...
    position_features: list[str] = ...

@dataclass(slots=True)
class JudgeConfig(_SerializableConfig):
    """Required versioned pairwise judge policy for one pool."""

    version: int
    model: str
    model_revision: str
    prompt_version: str
    rubric_version: str
    output_schema_version: int
    response_weight: float
    trajectory_weight: float
    response_floor: float
    trajectory_floor: float
    judge_confidence_floor: float
    pass_threshold: float
    max_rationale_bytes: int
    base_cooloff_seconds: int
    max_cooloff_seconds: int
    temperature: float | None = ...

@dataclass(slots=True)
class PoolConfig(_SerializableConfig):
    """One deterministic Router pool."""

    id: str
    api_family: LlmApiFamily
    anchor_models: list[str]
    anchor_revision: str
    sampling_probability: float
    max_candidates_per_sample: int
    concurrency: ConcurrencyConfig | JsonObject
    candidates: list[CandidateConfig | JsonObject]
    judge: JudgeConfig | JsonObject
    selector: PoolSelectorConfig | JsonObject = ...
    lookahead: LookaheadConfig | JsonObject = ...
    canonicalizer: CanonicalizerConfig | JsonObject = ...
    learning: LearningConfig | JsonObject | None = ...
    outcome: OutcomeConfig | JsonObject = ...

@dataclass(slots=True)
class RouterConfig(_SerializableConfig):
    """Canonical version-1 Router component configuration."""

    version: int = ...
    mode: RouterMode = ...
    project_id: str | None = ...
    database_path: str = ...
    retention_days: int = ...
    max_evidence_records: int = ...
    allow_remote_embedding_egress: bool = ...
    embedders: list[EmbedderConfig | JsonObject] = ...
    pools: list[PoolConfig | JsonObject] = ...
    policy: ConfigPolicy | JsonObject = ...

ROUTER_PLUGIN_KIND: Literal["router"]

@dataclass(slots=True)
class ComponentSpec(_SerializableConfig):
    """Top-level Router plugin component wrapper."""

    config: RouterConfig | JsonObject
    enabled: bool = ...
    def to_dict(self) -> JsonObject: ...

def validate_config(config: RouterConfig | JsonObject) -> ConfigReport:
    """Validate one Router config document without activating it."""
    ...

__all__: list[str]
