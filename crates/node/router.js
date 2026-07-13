// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

'use strict';

const plugin = require('./plugin.js');

const ROUTER_PLUGIN_KIND = 'router';

function compact(value) {
  if (Array.isArray(value)) {
    return value.map(compact);
  }
  if (value !== null && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([, item]) => item !== undefined)
        .map(([key, item]) => [key, compact(item)]),
    );
  }
  return value;
}

/**
 * Create unsupported-configuration policy with Rust defaults applied.
 *
 * @param {object} [config={}] - Partial policy settings.
 * @returns {object} A normalized Router config policy.
 */
function configPolicy(config = {}) {
  return compact({
    unknown_component: 'warn',
    unknown_field: 'warn',
    unsupported_value: 'error',
    ...config,
  });
}

/**
 * Create a Router component config with Rust defaults applied.
 *
 * @param {object} [config={}] - Partial Router settings.
 * @returns {object} A normalized version-1 Router config.
 */
function defaultConfig(config = {}) {
  return compact({
    version: 1,
    mode: 'off',
    database_path: '.nemo-relay/router/router.db',
    retention_days: 30,
    max_evidence_records: 100000,
    allow_remote_embedding_egress: false,
    embedders: [],
    pools: [],
    policy: configPolicy(),
    ...config,
  });
}

/**
 * Create an embedding provider profile with Rust defaults applied.
 *
 * @param {object} config - Embedding profile fields.
 * @returns {object} A normalized embedding profile.
 */
function embedderConfig(config) {
  return compact({
    max_in_flight: 4,
    batch_size: 16,
    ...config,
  });
}

/**
 * Create a learning association or policy with version 1 applied.
 *
 * @param {object} config - Learning association or complete policy fields.
 * @returns {object} A normalized learning config.
 */
function learningConfig(config) {
  return compact({
    version: 1,
    ...config,
  });
}

/**
 * Create one actual-outcome matcher.
 *
 * @param {object} config - Exact sanitized event matcher fields.
 * @returns {object} A normalized outcome matcher.
 */
function outcomeMatcher(config) {
  return compact({
    metadata_equals: {},
    ...config,
  });
}

/**
 * Create a complete Active outcome policy with version 1 applied.
 *
 * @param {object} config - Complete outcome policy fields.
 * @returns {object} A normalized outcome config.
 */
function outcomeConfig(config) {
  return compact({
    version: 1,
    ...config,
  });
}

/**
 * Create frozen-context selector predicates.
 *
 * @param {object} [config={}] - Partial selector predicates.
 * @returns {object} A normalized selector config.
 */
function selectorConfig(config = {}) {
  return compact({
    metadata_equals: {},
    ...config,
  });
}

/**
 * Create future-local lookahead limits with Rust defaults applied.
 *
 * @param {object} [config={}] - Partial lookahead limits.
 * @returns {object} A normalized lookahead config.
 */
function lookaheadConfig(config = {}) {
  return compact({
    primary_llm_completions: 3,
    deadline_seconds: 300,
    lifecycle_presets: [],
    max_events_per_window: 512,
    max_bytes_per_window: 4 * 1024 * 1024,
    ...config,
  });
}

/**
 * Create independent per-pool concurrency limits.
 *
 * @param {object} config - Required Shadow and Judge limits.
 * @returns {object} A normalized concurrency config.
 */
function concurrencyConfig(config) {
  return compact({
    max_pending: 32,
    ...config,
  });
}

/**
 * Create candidate capability declarations with conservative defaults.
 *
 * @param {object} [config={}] - Supported candidate capabilities.
 * @returns {object} A normalized candidate capability config.
 */
function candidateCapabilities(config = {}) {
  return compact({
    tools: false,
    multimodal_input: false,
    structured_output: false,
    reasoning_controls: false,
    ...config,
  });
}

/**
 * Create one candidate model config.
 *
 * @param {object} config - Candidate identity, model, revision, and rank.
 * @returns {object} A normalized candidate config.
 */
function candidateConfig(config) {
  const capabilities = candidateCapabilities(config.capabilities ?? {});
  return compact({
    ...config,
    capabilities,
  });
}

/**
 * Create canonical request-projection limits with Rust defaults applied.
 *
 * @param {object} [config={}] - Partial canonicalizer limits.
 * @returns {object} A normalized canonicalizer config.
 */
function canonicalizerConfig(config = {}) {
  return compact({
    version: 1,
    max_instruction_bytes: 32768,
    max_task_bytes: 16384,
    max_context_messages: 8,
    max_context_bytes: 32768,
    max_position_features_bytes: 4096,
    position_features: [],
    ...config,
  });
}

/**
 * Create a required versioned judge policy.
 *
 * @param {object} config - Complete pairwise judge policy fields.
 * @returns {object} A normalized judge config.
 */
function judgeConfig(config) {
  return compact({ ...config });
}

/**
 * Create one deterministic routing pool.
 *
 * @param {object} config - Required pool identity and routing fields.
 * @returns {object} A normalized pool config.
 */
function poolConfig(config) {
  return compact({
    ...config,
    selector: selectorConfig(config.selector ?? {}),
    lookahead: lookaheadConfig(config.lookahead ?? {}),
    concurrency: concurrencyConfig(config.concurrency),
    candidates: config.candidates.map(candidateConfig),
    canonicalizer: canonicalizerConfig(config.canonicalizer ?? {}),
    judge: judgeConfig(config.judge),
    outcome: config.outcome === undefined ? {} : compact(config.outcome),
  });
}

/**
 * Wrap Router config as a standard plugin component.
 *
 * @param {object} config - Router component configuration document.
 * @param {{ enabled?: boolean }} [options={}] - Optional component flags.
 * @returns {object} A plugin component spec for Router.
 */
function ComponentSpec(config, { enabled = true } = {}) {
  return plugin.ComponentSpec(ROUTER_PLUGIN_KIND, compact(config), { enabled });
}

/**
 * Validate Router config through the native plugin registry without activation.
 *
 * @param {object} config - Router component configuration document.
 * @returns {object} The native structured validation report.
 */
function validateConfig(config) {
  return plugin.validate({
    version: 1,
    components: [ComponentSpec(config)],
  });
}

module.exports = {
  ROUTER_PLUGIN_KIND,
  defaultConfig,
  configPolicy,
  embedderConfig,
  learningConfig,
  outcomeMatcher,
  outcomeConfig,
  selectorConfig,
  lookaheadConfig,
  concurrencyConfig,
  candidateCapabilities,
  candidateConfig,
  canonicalizerConfig,
  judgeConfig,
  poolConfig,
  ComponentSpec,
  validateConfig,
};
