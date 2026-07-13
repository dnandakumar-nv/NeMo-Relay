// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import type { Json, LlmApiFamily } from './index.js';
import type { ConfigReport } from './plugin.js';

export type { LlmApiFamily, LlmCallRole, LlmExecutionContext, LlmReplayDescriptor, LlmReplayFactory } from './index.js';

export type RouterMode = 'off' | 'shadow' | 'recommend' | 'active';
export type OutcomeMatcherEventKind = 'scope_end' | 'mark';
export type OutcomeTerminalStatus = 'ok' | 'error' | 'unset';
export type OutcomeDisposition = 'success' | 'failure' | 'ignore';

export interface ConfigPolicy {
  unknown_component?: 'ignore' | 'warn' | 'error' | string;
  unknown_field?: 'ignore' | 'warn' | 'error' | string;
  unsupported_value?: 'ignore' | 'warn' | 'error' | string;
}

export interface EmbedderConfig {
  id: string;
  base_url: string;
  model: string;
  provider_revision: string;
  dimensions: number;
  timeout_ms: number;
  api_key_env?: string;
  max_in_flight?: number;
  batch_size?: number;
}

export interface LearningConfig {
  version?: number;
  embedder: string;
  top_k?: number;
  radius?: number;
  min_points?: number;
  min_independent_roots?: number;
  min_effective_samples?: number;
  min_coverage?: number;
  time_decay_half_life_seconds?: number;
  prior_success?: number;
  prior_failure?: number;
  familywise_credible_level?: number;
  promotion_lower_bound?: number;
  retention_lower_bound?: number;
  holdout_probability?: number;
  active_canary_fraction?: number;
}

export interface OutcomeMatcher {
  event_kind: OutcomeMatcherEventKind;
  category: string;
  name: string;
  terminal_status: OutcomeTerminalStatus;
  metadata_equals?: Record<string, Json>;
}

export interface OutcomeConfig {
  version?: number;
  success_matchers: OutcomeMatcher[];
  failure_matchers: OutcomeMatcher[];
  completion_disposition: OutcomeDisposition;
  error_disposition: OutcomeDisposition;
  tool_failure_disposition: OutcomeDisposition;
  end_of_run_disposition: OutcomeDisposition;
  max_attribution_seconds: number;
  actual_outcome_half_life_seconds: number;
  anchor_shadow_half_life_seconds: number;
  relearning_cooloff_seconds: number;
  min_treatment_roots: number;
  min_control_roots: number;
  min_treatment_effective_weight: number;
  min_control_effective_weight: number;
  noninferiority_margin: number;
  noninferiority_probability: number;
  rollback_probability: number;
  outcome_evaluation_batch_size: number;
  max_canary_roots: number;
  authorization_ttl_seconds: number;
}

export interface PoolSelectorConfig {
  tenant_ids?: string[];
  agent_ids?: string[];
  owner_scope_types?: string[];
  metadata_equals?: Record<string, Json>;
  scope_path_patterns?: string[];
}

export interface LookaheadConfig {
  primary_llm_completions?: number;
  deadline_seconds?: number;
  lifecycle_presets?: string[];
  max_events_per_window?: number;
  max_bytes_per_window?: number;
}

export interface ConcurrencyConfig {
  shadow: number;
  judge: number;
  max_pending?: number;
}

export interface CandidateCapabilities {
  tools?: boolean;
  multimodal_input?: boolean;
  structured_output?: boolean;
  reasoning_controls?: boolean;
}

export interface CandidateConfig {
  id: string;
  model: string;
  model_revision: string;
  cost_rank: number;
  max_context_tokens?: number;
  capabilities?: CandidateCapabilities;
}

export interface CanonicalizerConfig {
  version?: number;
  max_instruction_bytes?: number;
  max_task_bytes?: number;
  max_context_messages?: number;
  max_context_bytes?: number;
  max_position_features_bytes?: number;
  position_features?: string[];
}

export interface JudgeConfig {
  version: number;
  model: string;
  model_revision: string;
  prompt_version: string;
  rubric_version: string;
  output_schema_version: number;
  response_weight: number;
  trajectory_weight: number;
  response_floor: number;
  trajectory_floor: number;
  judge_confidence_floor: number;
  pass_threshold: number;
  max_rationale_bytes: number;
  base_cooloff_seconds: number;
  max_cooloff_seconds: number;
  temperature?: number;
}

export interface PoolConfig {
  id: string;
  api_family: LlmApiFamily;
  anchor_models: string[];
  anchor_revision: string;
  sampling_probability: number;
  max_candidates_per_sample: number;
  selector?: PoolSelectorConfig;
  lookahead?: LookaheadConfig;
  concurrency: ConcurrencyConfig;
  candidates: CandidateConfig[];
  canonicalizer?: CanonicalizerConfig;
  judge: JudgeConfig;
  learning?: LearningConfig;
  outcome?: OutcomeConfig | Record<string, never>;
}

export interface Config {
  version?: number;
  mode?: RouterMode;
  project_id?: string;
  database_path?: string;
  retention_days?: number;
  max_evidence_records?: number;
  allow_remote_embedding_egress?: boolean;
  embedders?: EmbedderConfig[];
  pools?: PoolConfig[];
  policy?: ConfigPolicy;
}

export interface ComponentSpec {
  kind: 'router';
  enabled?: boolean;
  config: Config;
}

/** Top-level plugin kind used by the built-in Router component. */
export declare const ROUTER_PLUGIN_KIND: 'router';
/** Create a Router component config with Rust defaults applied. */
export declare function defaultConfig(config?: Config): Config;
/** Create unsupported-configuration policy with Rust defaults applied. */
export declare function configPolicy(config?: ConfigPolicy): ConfigPolicy;
/** Create an embedding provider profile with Rust defaults applied. */
export declare function embedderConfig(config: EmbedderConfig): EmbedderConfig;
/** Create a learning association or policy with version 1 applied. */
export declare function learningConfig(config: LearningConfig): LearningConfig;
/** Create one actual-outcome matcher. */
export declare function outcomeMatcher(config: OutcomeMatcher): OutcomeMatcher;
/** Create a complete Active outcome policy with version 1 applied. */
export declare function outcomeConfig(config: OutcomeConfig): OutcomeConfig;
/** Create frozen-context selector predicates. */
export declare function selectorConfig(config?: PoolSelectorConfig): PoolSelectorConfig;
/** Create future-local lookahead limits with Rust defaults applied. */
export declare function lookaheadConfig(config?: LookaheadConfig): LookaheadConfig;
/** Create independent per-pool concurrency limits. */
export declare function concurrencyConfig(config: ConcurrencyConfig): ConcurrencyConfig;
/** Create candidate capability declarations with conservative defaults. */
export declare function candidateCapabilities(config?: CandidateCapabilities): CandidateCapabilities;
/** Create one candidate model config. */
export declare function candidateConfig(config: CandidateConfig): CandidateConfig;
/** Create canonical request-projection limits with Rust defaults applied. */
export declare function canonicalizerConfig(config?: CanonicalizerConfig): CanonicalizerConfig;
/** Create a required versioned judge policy. */
export declare function judgeConfig(config: JudgeConfig): JudgeConfig;
/** Create one deterministic routing pool. */
export declare function poolConfig(config: PoolConfig): PoolConfig;
/** Wrap Router config as a standard plugin component. */
export declare function ComponentSpec(
  config: Config,
  options?: { enabled?: boolean },
): import('./plugin.js').ComponentSpec;
/** Validate Router config through the native plugin registry without activation. */
export declare function validateConfig(config: Config): ConfigReport;
