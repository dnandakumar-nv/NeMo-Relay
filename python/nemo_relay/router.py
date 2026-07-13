# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shadow Router plugin configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Literal, Protocol, TypedDict, cast

from nemo_relay import Json, JsonObject, UnsupportedBehavior
from nemo_relay import plugin as plugin_module
from nemo_relay.llm import (
    LlmApiFamily,
    LlmCallRole,
    LlmExecutionContext,
    LlmReplayDescriptor,
    LlmReplayFactory,
)

RouterMode = Literal["off", "shadow", "recommend", "active"]
OutcomeMatcherEventKind = Literal["scope_end", "mark"]
OutcomeTerminalStatus = Literal["ok", "error", "unset"]
OutcomeDisposition = Literal["success", "failure", "ignore"]


class _ConfigDiagnosticRequired(TypedDict):
    level: Literal["warning", "error"]
    code: str
    message: str


class ConfigDiagnostic(_ConfigDiagnosticRequired, total=False):
    """One Router configuration validation diagnostic."""

    component: str
    field: str


class ConfigReport(TypedDict):
    """Validation report for Router configuration."""

    diagnostics: list[ConfigDiagnostic]


class _SupportsToDict(Protocol):
    def to_dict(self) -> JsonObject: ...


def _normalize(value: object) -> Json:
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field_info.name: _normalize(field_value)
            for field_info in fields(value)
            if (field_value := getattr(value, field_info.name)) is not None
        }
    if hasattr(value, "to_dict"):
        return cast(_SupportsToDict, value).to_dict()
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {cast(str, key): _normalize(item) for key, item in value.items() if item is not None}
    return cast(Json, value)


def _normalize_object(value: object) -> JsonObject:
    return cast(JsonObject, _normalize(value))


class _SerializableConfig:
    def to_dict(self) -> JsonObject:
        """Serialize this helper to the canonical Router JSON object shape."""
        return _normalize_object(self)


@dataclass(slots=True)
class ConfigPolicy(_SerializableConfig):
    """Policy for unsupported Router configuration."""

    unknown_component: UnsupportedBehavior = "warn"
    unknown_field: UnsupportedBehavior = "warn"
    unsupported_value: UnsupportedBehavior = "error"


@dataclass(slots=True)
class EmbedderConfig(_SerializableConfig):
    """One OpenAI-compatible embedding provider profile."""

    id: str
    base_url: str
    model: str
    provider_revision: str
    dimensions: int
    timeout_ms: int
    api_key_env: str | None = None
    max_in_flight: int = 4
    batch_size: int = 16


@dataclass(slots=True)
class LearningConfig(_SerializableConfig):
    """Versioned embedding association or complete routing policy."""

    embedder: str
    version: int = 1
    top_k: int | None = None
    radius: float | None = None
    min_points: int | None = None
    min_independent_roots: int | None = None
    min_effective_samples: float | None = None
    min_coverage: float | None = None
    time_decay_half_life_seconds: float | None = None
    prior_success: float | None = None
    prior_failure: float | None = None
    familywise_credible_level: float | None = None
    promotion_lower_bound: float | None = None
    retention_lower_bound: float | None = None
    holdout_probability: float | None = None
    active_canary_fraction: float | None = None


@dataclass(slots=True)
class OutcomeMatcher(_SerializableConfig):
    """One exact sanitized event matcher for actual outcomes."""

    event_kind: OutcomeMatcherEventKind
    category: str
    name: str
    terminal_status: OutcomeTerminalStatus
    metadata_equals: dict[str, Json] = field(default_factory=dict)


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
    version: int = 1


@dataclass(slots=True)
class PoolSelectorConfig(_SerializableConfig):
    """Predicates matched against frozen V2 call facts."""

    tenant_ids: list[str] | None = None
    agent_ids: list[str] | None = None
    owner_scope_types: list[str] | None = None
    metadata_equals: dict[str, Json] = field(default_factory=dict)
    scope_path_patterns: list[str] | None = None


@dataclass(slots=True)
class LookaheadConfig(_SerializableConfig):
    """Future-local trajectory window limits."""

    primary_llm_completions: int = 3
    deadline_seconds: int = 300
    lifecycle_presets: list[str] = field(default_factory=list)
    max_events_per_window: int = 512
    max_bytes_per_window: int = 4 * 1024 * 1024


@dataclass(slots=True)
class ConcurrencyConfig(_SerializableConfig):
    """Independent per-pool provider and pending-work limits."""

    shadow: int
    judge: int
    max_pending: int = 32


@dataclass(slots=True)
class CandidateCapabilities(_SerializableConfig):
    """Capabilities explicitly supported by a candidate model."""

    tools: bool = False
    multimodal_input: bool = False
    structured_output: bool = False
    reasoning_controls: bool = False


@dataclass(slots=True)
class CandidateConfig(_SerializableConfig):
    """One cheaper candidate model in a routing pool."""

    id: str
    model: str
    model_revision: str
    cost_rank: int
    max_context_tokens: int | None = None
    capabilities: CandidateCapabilities | JsonObject = field(default_factory=CandidateCapabilities)


@dataclass(slots=True)
class CanonicalizerConfig(_SerializableConfig):
    """Limits for deterministic semantic request canonicalization."""

    version: int = 1
    max_instruction_bytes: int = 32_768
    max_task_bytes: int = 16_384
    max_context_messages: int = 8
    max_context_bytes: int = 32_768
    max_position_features_bytes: int = 4_096
    position_features: list[str] = field(default_factory=list)


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
    temperature: float | None = None


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
    selector: PoolSelectorConfig | JsonObject = field(default_factory=PoolSelectorConfig)
    lookahead: LookaheadConfig | JsonObject = field(default_factory=LookaheadConfig)
    canonicalizer: CanonicalizerConfig | JsonObject = field(default_factory=CanonicalizerConfig)
    learning: LearningConfig | JsonObject | None = None
    outcome: OutcomeConfig | JsonObject = field(default_factory=dict)


@dataclass(slots=True)
class RouterConfig(_SerializableConfig):
    """Canonical version-1 Router component configuration."""

    version: int = 1
    mode: RouterMode = "off"
    project_id: str | None = None
    database_path: str = ".nemo-relay/router/router.db"
    retention_days: int = 30
    max_evidence_records: int = 100_000
    allow_remote_embedding_egress: bool = False
    embedders: list[EmbedderConfig | JsonObject] = field(default_factory=list)
    pools: list[PoolConfig | JsonObject] = field(default_factory=list)
    policy: ConfigPolicy | JsonObject = field(default_factory=ConfigPolicy)


ROUTER_PLUGIN_KIND = "router"


@dataclass(slots=True)
class ComponentSpec(_SerializableConfig):
    """Top-level Router plugin component wrapper."""

    config: RouterConfig | JsonObject
    enabled: bool = True

    def to_dict(self) -> JsonObject:
        """Serialize this component to the standard plugin component shape."""
        return {
            "kind": ROUTER_PLUGIN_KIND,
            "enabled": self.enabled,
            "config": _normalize_object(self.config),
        }


def validate_config(config: RouterConfig | JsonObject) -> ConfigReport:
    """Validate one Router config document without activating it."""
    return plugin_module.validate(plugin_module.PluginConfig(components=[ComponentSpec(config)]))


__all__ = [
    "CandidateCapabilities",
    "CandidateConfig",
    "CanonicalizerConfig",
    "ComponentSpec",
    "ConcurrencyConfig",
    "ConfigDiagnostic",
    "ConfigPolicy",
    "ConfigReport",
    "EmbedderConfig",
    "JudgeConfig",
    "LearningConfig",
    "LlmApiFamily",
    "LlmCallRole",
    "LlmExecutionContext",
    "LlmReplayDescriptor",
    "LlmReplayFactory",
    "LookaheadConfig",
    "OutcomeConfig",
    "OutcomeDisposition",
    "OutcomeMatcher",
    "OutcomeMatcherEventKind",
    "OutcomeTerminalStatus",
    "PoolConfig",
    "PoolSelectorConfig",
    "ROUTER_PLUGIN_KIND",
    "RouterConfig",
    "RouterMode",
    "validate_config",
]
