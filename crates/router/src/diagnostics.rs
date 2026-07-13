// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable Router configuration diagnostics.

use std::collections::BTreeSet;

use nemo_relay::plugin::{ConfigDiagnostic, ConfigPolicy, DiagnosticLevel, UnsupportedBehavior};
use serde_json::{Map, Value as Json};

use crate::config::RouterConfig;

/// Stable diagnostic code for a malformed component document.
pub const INVALID_PLUGIN_CONFIG: &str = "router.invalid_plugin_config";
/// Stable diagnostic code for a config schema version that is not supported.
pub const UNSUPPORTED_CONFIG_VERSION: &str = "router.unsupported_config_version";
/// Stable diagnostic code for a mode that cannot activate at this rollout gate.
pub const UNSUPPORTED_MODE: &str = "router.unsupported_mode";
/// Stable diagnostic code for an unrecognized field.
pub const UNKNOWN_FIELD: &str = "router.unknown_field";
/// Stable diagnostic code for a duplicate durable identity or rank.
pub const DUPLICATE_ID: &str = "router.duplicate_id";
/// Stable diagnostic code for a numeric or collection bound violation.
pub const INVALID_RANGE: &str = "router.invalid_range";
/// Stable diagnostic code for an unknown or unsupported registry reference.
pub const INVALID_REFERENCE: &str = "router.invalid_reference";
/// Stable diagnostic code for selector domains that can match the same call.
pub const OVERLAPPING_POOL: &str = "router.overlapping_pool";
/// Stable diagnostic code for an unsafe or structurally invalid database path.
pub const INVALID_PATH: &str = "router.invalid_path";
/// Stable diagnostic code for an embedding endpoint that violates egress policy.
pub const UNSAFE_EMBEDDER_ENDPOINT: &str = "router.unsafe_embedder_endpoint";
/// Stable diagnostic code for a future-owned extension that is not active yet.
pub const UNSUPPORTED_EXTENSION: &str = "router.unsupported_extension";

/// Result of parsing and validating one Router plugin component document.
#[derive(Debug, Clone)]
pub struct RouterConfigValidation {
    /// Parsed typed configuration, or `None` when deserialization failed.
    pub config: Option<RouterConfig>,
    /// Deterministically ordered validation diagnostics.
    pub diagnostics: Vec<ConfigDiagnostic>,
    /// Canonical generation identity when the document has no error diagnostics.
    pub config_generation_id: Option<String>,
}

impl RouterConfigValidation {
    /// Returns `true` when at least one validation diagnostic is an error.
    pub fn has_errors(&self) -> bool {
        self.diagnostics
            .iter()
            .any(|diagnostic| diagnostic.level == DiagnosticLevel::Error)
    }
}

/// Parses and validates a component-local Router configuration object.
///
/// Validation is side-effect free. It does not read environment values, inspect
/// or create filesystem paths, resolve endpoint host names, start tasks, or
/// register runtime behavior.
pub fn validate_router_config(plugin_config: &Map<String, Json>) -> RouterConfigValidation {
    let mut shape_diagnostics = validate_raw_judge_sections(plugin_config);
    shape_diagnostics.extend(validate_raw_learning_sections(plugin_config));
    shape_diagnostics.extend(validate_raw_outcome_sections(plugin_config));
    let config = match serde_json::from_value::<RouterConfig>(Json::Object(plugin_config.clone())) {
        Ok(config) => config,
        Err(err) => {
            shape_diagnostics.push(error(
                INVALID_PLUGIN_CONFIG,
                None,
                format!("invalid Router plugin configuration: {err}"),
            ));
            return RouterConfigValidation {
                config: None,
                diagnostics: shape_diagnostics,
                config_generation_id: None,
            };
        }
    };

    let raw_unknown_fields = shape_diagnostics
        .iter()
        .filter(|diagnostic| diagnostic.code == UNKNOWN_FIELD)
        .filter_map(|diagnostic| diagnostic.field.clone())
        .collect::<BTreeSet<_>>();
    let mut typed_diagnostics = config.validate_diagnostics();
    typed_diagnostics.retain(|diagnostic| {
        diagnostic.code != UNKNOWN_FIELD
            || diagnostic
                .field
                .as_ref()
                .is_none_or(|field| !raw_unknown_fields.contains(field))
    });
    let mut diagnostics = shape_diagnostics;
    diagnostics.extend(typed_diagnostics);
    validate_policy_unknown_fields(plugin_config, &config.policy, &mut diagnostics);
    let has_errors = diagnostics
        .iter()
        .any(|diagnostic| diagnostic.level == DiagnosticLevel::Error);
    let config_generation_id = if has_errors {
        None
    } else {
        match config.generation_id() {
            Ok(generation_id) => Some(generation_id),
            Err(message) => {
                diagnostics.push(error(INVALID_PLUGIN_CONFIG, None, message));
                None
            }
        }
    };

    RouterConfigValidation {
        config: Some(config),
        diagnostics,
        config_generation_id,
    }
}

fn validate_raw_learning_sections(plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();
    let Some(pools) = plugin_config.get("pools").and_then(Json::as_array) else {
        return diagnostics;
    };

    for (pool_index, pool) in pools.iter().enumerate() {
        let Some(pool) = pool.as_object() else {
            continue;
        };
        let Some(learning) = pool.get("learning") else {
            continue;
        };
        let prefix = format!("pools[{pool_index}].learning");
        let Some(learning) = learning.as_object() else {
            diagnostics.push(error(
                INVALID_PLUGIN_CONFIG,
                Some(prefix),
                "learning must be an object",
            ));
            continue;
        };
        if learning.is_empty() {
            continue;
        }

        for (field, kind) in RAW_LEARNING_BASE_FIELDS {
            validate_required_raw_learning_field(learning, field, kind, &prefix, &mut diagnostics);
        }

        let has_statistical_fields = RAW_LEARNING_STATISTICAL_FIELDS
            .iter()
            .any(|(field, _)| learning.contains_key(*field));
        let has_active_fields = RAW_LEARNING_ACTIVE_FIELDS
            .iter()
            .any(|(field, _)| learning.contains_key(*field));
        if has_statistical_fields || has_active_fields {
            for (field, kind) in RAW_LEARNING_STATISTICAL_FIELDS {
                validate_required_raw_learning_field(
                    learning,
                    field,
                    kind,
                    &prefix,
                    &mut diagnostics,
                );
            }
        }
        if has_active_fields {
            for (field, kind) in RAW_LEARNING_ACTIVE_FIELDS {
                validate_required_raw_learning_field(
                    learning,
                    field,
                    kind,
                    &prefix,
                    &mut diagnostics,
                );
            }
        }

        let mut unknown = learning
            .keys()
            .filter(|field| {
                !RAW_LEARNING_BASE_FIELDS
                    .iter()
                    .chain(RAW_LEARNING_STATISTICAL_FIELDS.iter())
                    .chain(RAW_LEARNING_ACTIVE_FIELDS.iter())
                    .any(|(known, _)| known == &field.as_str())
            })
            .collect::<Vec<_>>();
        unknown.sort();
        for field in unknown {
            diagnostics.push(error(
                UNKNOWN_FIELD,
                Some(format!("{prefix}.{field}")),
                format!("field '{field}' is not recognized for 'learning'"),
            ));
        }
    }

    diagnostics
}

fn validate_raw_outcome_sections(plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();
    let Some(pools) = plugin_config.get("pools").and_then(Json::as_array) else {
        return diagnostics;
    };

    for (pool_index, pool) in pools.iter().enumerate() {
        let Some(pool) = pool.as_object() else {
            continue;
        };
        let Some(outcome) = pool.get("outcome") else {
            continue;
        };
        let prefix = format!("pools[{pool_index}].outcome");
        let Some(outcome) = outcome.as_object() else {
            diagnostics.push(error(
                INVALID_PLUGIN_CONFIG,
                Some(prefix),
                "outcome must be an object",
            ));
            continue;
        };
        if outcome.is_empty() {
            continue;
        }

        validate_required_raw_fields(
            outcome,
            &RAW_OUTCOME_FIELDS,
            &prefix,
            "outcome",
            &mut diagnostics,
        );
        validate_raw_unknown_fields(
            outcome,
            &RAW_OUTCOME_FIELDS,
            &prefix,
            "outcome",
            &mut diagnostics,
        );
        for field in [
            "completion_disposition",
            "error_disposition",
            "tool_failure_disposition",
            "end_of_run_disposition",
        ] {
            validate_raw_enum(
                outcome,
                field,
                &["success", "failure", "ignore"],
                &prefix,
                &mut diagnostics,
            );
        }

        for list_name in ["success_matchers", "failure_matchers"] {
            let Some(matchers) = outcome.get(list_name).and_then(Json::as_array) else {
                continue;
            };
            for (matcher_index, matcher) in matchers.iter().enumerate() {
                let matcher_prefix = format!("{prefix}.{list_name}[{matcher_index}]");
                let Some(matcher) = matcher.as_object() else {
                    diagnostics.push(error(
                        INVALID_PLUGIN_CONFIG,
                        Some(matcher_prefix),
                        "outcome matcher must be an object",
                    ));
                    continue;
                };
                validate_required_raw_fields(
                    matcher,
                    &RAW_OUTCOME_MATCHER_FIELDS,
                    &matcher_prefix,
                    "outcome matcher",
                    &mut diagnostics,
                );
                validate_raw_unknown_fields(
                    matcher,
                    &RAW_OUTCOME_MATCHER_FIELDS,
                    &matcher_prefix,
                    "outcome matcher",
                    &mut diagnostics,
                );
                validate_raw_enum(
                    matcher,
                    "event_kind",
                    &["scope_end", "mark"],
                    &matcher_prefix,
                    &mut diagnostics,
                );
                validate_raw_enum(
                    matcher,
                    "terminal_status",
                    &["ok", "error", "unset"],
                    &matcher_prefix,
                    &mut diagnostics,
                );
            }
        }
    }

    diagnostics
}

fn validate_raw_enum(
    object: &Map<String, Json>,
    field: &str,
    allowed: &[&str],
    prefix: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let Some(value) = object.get(field).and_then(Json::as_str) else {
        return;
    };
    if !allowed.contains(&value) {
        diagnostics.push(error(
            INVALID_REFERENCE,
            Some(format!("{prefix}.{field}")),
            format!("value '{value}' is not supported for '{field}'"),
        ));
    }
}

fn validate_required_raw_fields(
    object: &Map<String, Json>,
    fields: &[(&str, RawFieldKind)],
    prefix: &str,
    section: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    for (field, kind) in fields {
        let location = format!("{prefix}.{field}");
        match object.get(*field) {
            None => diagnostics.push(error(
                INVALID_PLUGIN_CONFIG,
                Some(location),
                format!("required {section} field '{field}' is missing"),
            )),
            Some(value) if !raw_field_matches(value, *kind) => diagnostics.push(error(
                INVALID_PLUGIN_CONFIG,
                Some(location),
                format!("{section} field '{field}' has the wrong JSON type"),
            )),
            Some(_) => {}
        }
    }
}

fn validate_raw_unknown_fields(
    object: &Map<String, Json>,
    fields: &[(&str, RawFieldKind)],
    prefix: &str,
    section: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let mut unknown = object
        .keys()
        .filter(|field| !fields.iter().any(|(known, _)| known == &field.as_str()))
        .collect::<Vec<_>>();
    unknown.sort();
    for field in unknown {
        diagnostics.push(error(
            UNKNOWN_FIELD,
            Some(format!("{prefix}.{field}")),
            format!("field '{field}' is not recognized for '{section}'"),
        ));
    }
}

#[derive(Clone, Copy)]
enum RawFieldKind {
    String,
    U32,
    U64,
    Usize,
    Number,
    Array,
    Object,
}

const RAW_LEARNING_BASE_FIELDS: [(&str, RawFieldKind); 2] = [
    ("version", RawFieldKind::U32),
    ("embedder", RawFieldKind::String),
];

const RAW_LEARNING_STATISTICAL_FIELDS: [(&str, RawFieldKind); 11] = [
    ("top_k", RawFieldKind::Usize),
    ("radius", RawFieldKind::Number),
    ("min_points", RawFieldKind::Usize),
    ("min_independent_roots", RawFieldKind::Usize),
    ("min_effective_samples", RawFieldKind::Number),
    ("min_coverage", RawFieldKind::Number),
    ("time_decay_half_life_seconds", RawFieldKind::Number),
    ("prior_success", RawFieldKind::Number),
    ("prior_failure", RawFieldKind::Number),
    ("familywise_credible_level", RawFieldKind::Number),
    ("promotion_lower_bound", RawFieldKind::Number),
];

const RAW_LEARNING_ACTIVE_FIELDS: [(&str, RawFieldKind); 3] = [
    ("retention_lower_bound", RawFieldKind::Number),
    ("holdout_probability", RawFieldKind::Number),
    ("active_canary_fraction", RawFieldKind::Number),
];

const RAW_OUTCOME_FIELDS: [(&str, RawFieldKind); 21] = [
    ("version", RawFieldKind::U32),
    ("success_matchers", RawFieldKind::Array),
    ("failure_matchers", RawFieldKind::Array),
    ("completion_disposition", RawFieldKind::String),
    ("error_disposition", RawFieldKind::String),
    ("tool_failure_disposition", RawFieldKind::String),
    ("end_of_run_disposition", RawFieldKind::String),
    ("max_attribution_seconds", RawFieldKind::U64),
    ("actual_outcome_half_life_seconds", RawFieldKind::U64),
    ("anchor_shadow_half_life_seconds", RawFieldKind::U64),
    ("relearning_cooloff_seconds", RawFieldKind::U64),
    ("min_treatment_roots", RawFieldKind::U64),
    ("min_control_roots", RawFieldKind::U64),
    ("min_treatment_effective_weight", RawFieldKind::Number),
    ("min_control_effective_weight", RawFieldKind::Number),
    ("noninferiority_margin", RawFieldKind::Number),
    ("noninferiority_probability", RawFieldKind::Number),
    ("rollback_probability", RawFieldKind::Number),
    ("outcome_evaluation_batch_size", RawFieldKind::U64),
    ("max_canary_roots", RawFieldKind::U64),
    ("authorization_ttl_seconds", RawFieldKind::U64),
];

const RAW_OUTCOME_MATCHER_FIELDS: [(&str, RawFieldKind); 5] = [
    ("event_kind", RawFieldKind::String),
    ("category", RawFieldKind::String),
    ("name", RawFieldKind::String),
    ("terminal_status", RawFieldKind::String),
    ("metadata_equals", RawFieldKind::Object),
];

const RAW_JUDGE_FIELDS: [(&str, RawFieldKind); 16] = [
    ("version", RawFieldKind::U32),
    ("model", RawFieldKind::String),
    ("model_revision", RawFieldKind::String),
    ("prompt_version", RawFieldKind::String),
    ("rubric_version", RawFieldKind::String),
    ("output_schema_version", RawFieldKind::U32),
    ("temperature", RawFieldKind::Number),
    ("response_weight", RawFieldKind::Number),
    ("trajectory_weight", RawFieldKind::Number),
    ("response_floor", RawFieldKind::Number),
    ("trajectory_floor", RawFieldKind::Number),
    ("judge_confidence_floor", RawFieldKind::Number),
    ("pass_threshold", RawFieldKind::Number),
    ("max_rationale_bytes", RawFieldKind::Usize),
    ("base_cooloff_seconds", RawFieldKind::U64),
    ("max_cooloff_seconds", RawFieldKind::U64),
];

fn validate_required_raw_learning_field(
    learning: &Map<String, Json>,
    field: &str,
    kind: RawFieldKind,
    prefix: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let location = format!("{prefix}.{field}");
    match learning.get(field) {
        None => diagnostics.push(error(
            INVALID_PLUGIN_CONFIG,
            Some(location),
            format!("required learning field '{field}' is missing"),
        )),
        Some(value) if !raw_field_matches(value, kind) => diagnostics.push(error(
            INVALID_PLUGIN_CONFIG,
            Some(location),
            format!("learning field '{field}' has the wrong JSON type"),
        )),
        Some(_) => {}
    }
}

fn validate_raw_judge_sections(plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
    let mut diagnostics = Vec::new();
    let Some(pools) = plugin_config.get("pools").and_then(Json::as_array) else {
        return diagnostics;
    };

    for (pool_index, pool) in pools.iter().enumerate() {
        let Some(pool) = pool.as_object() else {
            continue;
        };
        let prefix = format!("pools[{pool_index}].judge");
        let Some(judge) = pool.get("judge") else {
            diagnostics.push(error(
                INVALID_PLUGIN_CONFIG,
                Some(prefix),
                "required judge section is missing",
            ));
            continue;
        };
        let Some(judge) = judge.as_object() else {
            diagnostics.push(error(
                INVALID_PLUGIN_CONFIG,
                Some(prefix),
                "judge must be an object",
            ));
            continue;
        };

        for (field, kind) in RAW_JUDGE_FIELDS {
            let location = format!("{prefix}.{field}");
            match judge.get(field) {
                None if field == "temperature" => {}
                None => {
                    diagnostics.push(error(
                        INVALID_PLUGIN_CONFIG,
                        Some(location),
                        format!("required judge field '{field}' is missing"),
                    ));
                }
                Some(value) if !raw_field_matches(value, kind) => {
                    diagnostics.push(error(
                        INVALID_PLUGIN_CONFIG,
                        Some(location),
                        format!("judge field '{field}' has the wrong JSON type"),
                    ));
                }
                Some(_) => {}
            }
        }

        let mut unknown = judge
            .keys()
            .filter(|field| {
                !RAW_JUDGE_FIELDS
                    .iter()
                    .any(|(known, _)| known == &field.as_str())
            })
            .collect::<Vec<_>>();
        unknown.sort();
        for field in unknown {
            diagnostics.push(error(
                UNKNOWN_FIELD,
                Some(format!("{prefix}.{field}")),
                format!("field '{field}' is not recognized for 'judge'"),
            ));
        }
    }

    diagnostics
}

fn raw_field_matches(value: &Json, kind: RawFieldKind) -> bool {
    match kind {
        RawFieldKind::String => value.is_string(),
        RawFieldKind::U32 => value
            .as_u64()
            .is_some_and(|number| u32::try_from(number).is_ok()),
        RawFieldKind::U64 => value.as_u64().is_some(),
        RawFieldKind::Usize => value
            .as_u64()
            .is_some_and(|number| usize::try_from(number).is_ok()),
        RawFieldKind::Number => value.as_f64().is_some(),
        RawFieldKind::Array => value.is_array(),
        RawFieldKind::Object => value.is_object(),
    }
}

fn validate_policy_unknown_fields(
    plugin_config: &Map<String, Json>,
    policy: &ConfigPolicy,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let Some(policy_object) = plugin_config.get("policy").and_then(Json::as_object) else {
        return;
    };
    let mut fields = policy_object.keys().collect::<Vec<_>>();
    fields.sort();
    for field in fields {
        if !["unknown_component", "unknown_field", "unsupported_value"].contains(&field.as_str()) {
            push_policy_diagnostic(
                diagnostics,
                policy.unknown_field,
                UNKNOWN_FIELD,
                Some(format!("policy.{field}")),
                format!("field '{field}' is not recognized for 'policy'"),
            );
        }
    }
}

pub(crate) fn error(
    code: &str,
    field: Option<String>,
    message: impl Into<String>,
) -> ConfigDiagnostic {
    ConfigDiagnostic {
        level: DiagnosticLevel::Error,
        code: code.to_string(),
        component: Some("router".to_string()),
        field,
        message: message.into(),
    }
}

pub(crate) fn push_policy_diagnostic(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    behavior: UnsupportedBehavior,
    code: &str,
    field: Option<String>,
    message: impl Into<String>,
) {
    let level = match behavior {
        UnsupportedBehavior::Ignore => return,
        UnsupportedBehavior::Warn => DiagnosticLevel::Warning,
        UnsupportedBehavior::Error => DiagnosticLevel::Error,
    };
    diagnostics.push(ConfigDiagnostic {
        level,
        code: code.to_string(),
        component: Some("router".to_string()),
        field,
        message: message.into(),
    });
}
