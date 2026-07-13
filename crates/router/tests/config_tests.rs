// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router configuration validation and generation tests.

use std::collections::BTreeMap;

use nemo_relay::config_editor::{EditorConfig, EditorFieldKind};
use nemo_relay::plugin::{DiagnosticLevel, UnsupportedBehavior};
use nemo_relay_router::config::{
    CONCURRENCY_MAX_PERMITS, EMBEDDER_BATCH_SIZE_MAX, EMBEDDER_MAX_IN_FLIGHT,
    EMBEDDER_TIMEOUT_MS_MAX, EMBEDDING_DIMENSIONS_MAX, EVIDENCE_RECORDS_MAX, ID_MAX_BYTES,
    JUDGE_MAX_RATIONALE_BYTES, JUDGE_PROMPT_VERSION_V1, JUDGE_RUBRIC_VERSION_V1,
    LEARNING_CANDIDATE_NEIGHBOR_PRODUCT_MAX, LEARNING_CANDIDATES_MAX, LEARNING_TOP_K_MAX,
    MODEL_ID_MAX_BYTES, OUTCOME_DURATION_SECONDS_MAX, OUTCOME_EVALUATION_BATCH_SIZE_MAX,
    OUTCOME_LABEL_MAX_BYTES, OUTCOME_MATCHER_TEXT_MAX_BYTES, OUTCOME_MATCHERS_MAX,
    OUTCOME_MAX_CANARY_ROOTS_MAX, OUTCOME_MAX_LOOKS, OUTCOME_METADATA_EQUALS_MAX,
    PATH_PATTERN_MAX_BYTES, PROJECT_ID_MAX_BYTES, REVISION_MAX_BYTES, SCHEDULER_MAX_SLOTS,
    SELECTOR_IDENTITY_MAX_BYTES,
};
use nemo_relay_router::diagnostics::{
    DUPLICATE_ID, INVALID_PATH, INVALID_PLUGIN_CONFIG, INVALID_RANGE, INVALID_REFERENCE,
    OVERLAPPING_POOL, UNKNOWN_FIELD, UNSAFE_EMBEDDER_ENDPOINT, UNSUPPORTED_CONFIG_VERSION,
    UNSUPPORTED_MODE,
};
use nemo_relay_router::{
    JUDGE_OUTPUT_SCHEMA_SHA256_V1, JUDGE_PROMPT_TEMPLATE_SHA256_V1,
    JUDGE_RUBRIC_TEMPLATE_SHA256_V1, JudgeConfig, LearningConfig, OutcomeConfig,
    OutcomeDisposition, OutcomeMatcher, OutcomeMatcherEventKind, OutcomeTerminalStatus, PoolConfig,
    RouterConfig, RouterMode, validate_router_config,
};
use serde_json::{Map, Value as Json, json};
use uuid::Uuid;

fn object(value: Json) -> Map<String, Json> {
    value.as_object().unwrap().clone()
}

fn valid_document() -> Json {
    json!({
        "version": 1,
        "mode": "shadow",
        "project_id": "deployment-1",
        "database_path": ".nemo-relay/router/router.db",
        "retention_days": 30,
        "max_evidence_records": 100000,
        "allow_remote_embedding_egress": false,
        "embedders": [{
            "id": "embedding-main",
            "base_url": "http://127.0.0.1:8080/v1",
            "model": "nvidia/nv-embed-v1",
            "provider_revision": "2026-07-01",
            "dimensions": 1024,
            "api_key_env": "ROUTER_EMBEDDING_API_KEY",
            "timeout_ms": 10000
        }],
        "pools": [{
            "id": "chat-default",
            "api_family": "openai_chat_completions",
            "anchor_models": ["anchor-large"],
            "anchor_revision": "2026-07-01",
            "sampling_probability": 0.25,
            "max_candidates_per_sample": 1,
            "selector": {
                "tenant_ids": ["tenant-a"],
                "agent_ids": ["agent-a"],
                "owner_scope_types": ["agent"],
                "metadata_equals": {"region": "us", "tier": 2},
                "scope_path_patterns": ["agent/**"]
            },
            "concurrency": {"shadow": 2, "judge": 1},
            "canonicalizer": {},
            "judge": {
                "version": 1,
                "model": "judge-model",
                "model_revision": "2026-07-01",
                "prompt_version": "pairwise-equivalence-v1",
                "rubric_version": "response-trajectory-equivalence-v1",
                "output_schema_version": 1,
                "response_weight": 0.5,
                "trajectory_weight": 0.5,
                "response_floor": 0.8,
                "trajectory_floor": 0.8,
                "judge_confidence_floor": 0.7,
                "pass_threshold": 0.85,
                "max_rationale_bytes": 4096,
                "base_cooloff_seconds": 10,
                "max_cooloff_seconds": 300
            },
            "candidates": [{
                "id": "small",
                "model": "candidate-small",
                "model_revision": "2026-06-15",
                "cost_rank": 0,
                "max_context_tokens": 32768,
                "capabilities": {"tools": true}
            }]
        }]
    })
}

fn add_second_candidate(pool: &mut Json) {
    let mut candidate = pool["candidates"][0].clone();
    candidate["id"] = json!("tiny");
    candidate["model"] = json!("candidate-tiny");
    candidate["model_revision"] = json!("2026-06-16");
    candidate["cost_rank"] = json!(1);
    pool["candidates"].as_array_mut().unwrap().push(candidate);
    pool["max_candidates_per_sample"] = json!(2);
}

fn enable_learning(document: &mut Json, embedder: &str) {
    document["pools"][0]["learning"] = json!({"version": 1, "embedder": embedder});
}

fn complete_learning(embedder: &str) -> Json {
    json!({
        "version": 1,
        "embedder": embedder,
        "top_k": 8,
        "radius": 0.25,
        "min_points": 3,
        "min_independent_roots": 2,
        "min_effective_samples": 1.5,
        "min_coverage": 0.8,
        "time_decay_half_life_seconds": 3600.0,
        "prior_success": 1.0,
        "prior_failure": 1.0,
        "familywise_credible_level": 0.95,
        "promotion_lower_bound": 0.9
    })
}

fn enable_complete_learning(document: &mut Json, embedder: &str) {
    document["pools"][0]["learning"] = complete_learning(embedder);
}

fn complete_active_learning(embedder: &str) -> Json {
    let mut learning = complete_learning(embedder);
    learning["retention_lower_bound"] = json!(0.8);
    learning["holdout_probability"] = json!(0.2);
    learning["active_canary_fraction"] = json!(0.4);
    learning
}

fn valid_outcome() -> Json {
    json!({
        "version": 1,
        "success_matchers": [{
            "event_kind": "scope_end",
            "category": "agent",
            "name": "completed",
            "terminal_status": "ok",
            "metadata_equals": {"outcome.label": "passed"}
        }],
        "failure_matchers": [{
            "event_kind": "scope_end",
            "category": "agent",
            "name": "completed",
            "terminal_status": "error",
            "metadata_equals": {"outcome.label": "failed"}
        }],
        "completion_disposition": "success",
        "error_disposition": "failure",
        "tool_failure_disposition": "failure",
        "end_of_run_disposition": "ignore",
        "max_attribution_seconds": 600,
        "actual_outcome_half_life_seconds": 1800,
        "anchor_shadow_half_life_seconds": 1800,
        "relearning_cooloff_seconds": 300,
        "min_treatment_roots": 32,
        "min_control_roots": 32,
        "min_treatment_effective_weight": 16.0,
        "min_control_effective_weight": 16.0,
        "noninferiority_margin": 0.1,
        "noninferiority_probability": 0.99,
        "rollback_probability": 0.95,
        "outcome_evaluation_batch_size": 64,
        "max_canary_roots": 64,
        "authorization_ttl_seconds": 600
    })
}

fn enable_active(document: &mut Json) {
    document["mode"] = json!("active");
    document["pools"][0]["learning"] = complete_active_learning("embedding-main");
    document["pools"][0]["outcome"] = valid_outcome();
}

fn set_candidate_count(document: &mut Json, count: usize) {
    let template = document["pools"][0]["candidates"][0].clone();
    document["pools"][0]["candidates"] = Json::Array(
        (0..count)
            .map(|index| {
                let mut candidate = template.clone();
                candidate["id"] = json!(format!("candidate-{index}"));
                candidate["model"] = json!(format!("candidate-model-{index}"));
                candidate["model_revision"] = json!(format!("revision-{index}"));
                candidate["cost_rank"] = json!(index);
                candidate
            })
            .collect(),
    );
}

fn add_disjoint_pool(document: &mut Json) {
    let mut pool = document["pools"][0].clone();
    pool["id"] = json!("responses-default");
    pool["api_family"] = json!("openai_responses");
    pool["anchor_models"] = json!(["responses-anchor"]);
    for (index, candidate) in pool["candidates"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        candidate["id"] = json!(format!("responses-{index}"));
        candidate["model"] = json!(format!("responses-candidate-{index}"));
    }
    document["pools"].as_array_mut().unwrap().push(pool);
}

fn report(value: Json) -> nemo_relay_router::RouterConfigValidation {
    validate_router_config(&object(value))
}

fn has_diagnostic(
    report: &nemo_relay_router::RouterConfigValidation,
    code: &str,
    field: Option<&str>,
) -> bool {
    report
        .diagnostics
        .iter()
        .any(|diagnostic| diagnostic.code == code && diagnostic.field.as_deref() == field)
}

#[test]
fn root_and_nested_defaults_are_exact() {
    let empty = validate_router_config(&Map::new());
    assert!(!empty.has_errors(), "{:?}", empty.diagnostics);
    let config = empty.config.unwrap();
    assert_eq!(config.version, 1);
    assert_eq!(config.mode, RouterMode::Off);
    assert_eq!(config.project_id, None);
    assert_eq!(config.database_path, ".nemo-relay/router/router.db");
    assert_eq!(config.retention_days, 30);
    assert_eq!(config.max_evidence_records, 100_000);
    assert!(!config.allow_remote_embedding_egress);
    assert!(config.embedders.is_empty());
    assert!(config.pools.is_empty());
    assert_eq!(config.policy.unknown_field, UnsupportedBehavior::Warn);

    let validated = report(valid_document());
    assert!(!validated.has_errors(), "{:?}", validated.diagnostics);
    let config = validated.config.unwrap();
    assert_eq!(config.embedders[0].max_in_flight, 4);
    assert_eq!(config.embedders[0].batch_size, 16);
    let pool = &config.pools[0];
    assert_eq!(pool.lookahead.primary_llm_completions, 3);
    assert_eq!(pool.lookahead.deadline_seconds, 300);
    assert!(pool.lookahead.lifecycle_presets.is_empty());
    assert_eq!(pool.lookahead.max_events_per_window, 512);
    assert_eq!(pool.lookahead.max_bytes_per_window, 4 * 1024 * 1024);
    assert_eq!(pool.concurrency.max_pending, 32);
    assert_eq!(pool.canonicalizer.version, 1);
    assert_eq!(pool.canonicalizer.max_instruction_bytes, 32_768);
    assert_eq!(pool.canonicalizer.max_task_bytes, 16_384);
    assert_eq!(pool.canonicalizer.max_context_messages, 8);
    assert_eq!(pool.canonicalizer.max_context_bytes, 32_768);
    assert_eq!(pool.canonicalizer.max_position_features_bytes, 4_096);
    assert!(pool.canonicalizer.position_features.is_empty());
    assert_eq!(pool.judge.version, 1);
    assert_eq!(pool.judge.prompt_version, JUDGE_PROMPT_VERSION_V1);
    assert_eq!(pool.judge.rubric_version, JUDGE_RUBRIC_VERSION_V1);
    assert_eq!(pool.judge.output_token_limit(), Some(1_536));
    assert!(pool.learning.is_none());
    assert!(pool.candidates[0].capabilities.tools);
    assert!(!pool.candidates[0].capabilities.multimodal_input);
    assert!(!pool.candidates[0].capabilities.structured_output);
    assert!(!pool.candidates[0].capabilities.reasoning_controls);
}

#[test]
fn lifecycle_presets_accept_only_registered_unique_exact_names() {
    for presets in [
        json!(["handoff"]),
        json!(["compaction"]),
        json!(["handoff", "compaction"]),
        json!(["compaction", "handoff"]),
    ] {
        let mut document = valid_document();
        document["pools"][0]["lookahead"] = json!({"lifecycle_presets": presets});
        let result = report(document);
        assert!(!result.has_errors(), "{:?}", result.diagnostics);
    }

    let mut document = valid_document();
    document["pools"][0]["lookahead"] = json!({
        "lifecycle_presets": ["unknown", "Handoff", "", " ", "COMPACTION"]
    });
    let result = report(document);
    for (preset, index) in [
        ("unknown", 0),
        ("Handoff", 1),
        ("", 2),
        (" ", 3),
        ("COMPACTION", 4),
    ] {
        let field = format!("pools[0].lookahead.lifecycle_presets[{index}]");
        assert!(
            has_diagnostic(&result, INVALID_REFERENCE, Some(&field)),
            "{preset:?}: {:?}",
            result.diagnostics
        );
    }
    assert!(result.config_generation_id.is_none());

    for presets in [
        json!(["handoff", "handoff"]),
        json!(["compaction", "compaction"]),
    ] {
        let mut document = valid_document();
        document["pools"][0]["lookahead"] = json!({"lifecycle_presets": presets});
        let result = report(document);
        assert!(has_diagnostic(
            &result,
            DUPLICATE_ID,
            Some("pools[0].lookahead.lifecycle_presets[1]")
        ));
        assert!(result.config_generation_id.is_none());
    }
}

#[test]
fn lifecycle_preset_generation_is_order_independent_and_membership_sensitive() {
    let generation = |presets: Json| {
        let mut document = valid_document();
        document["pools"][0]["lookahead"] = json!({"lifecycle_presets": presets});
        let result = report(document);
        assert!(!result.has_errors(), "{:?}", result.diagnostics);
        result.config_generation_id.unwrap()
    };

    let empty = generation(json!([]));
    let handoff = generation(json!(["handoff"]));
    let compaction = generation(json!(["compaction"]));
    let both = generation(json!(["handoff", "compaction"]));
    let reversed = generation(json!(["compaction", "handoff"]));

    assert_eq!(both, reversed);
    assert_ne!(empty, handoff);
    assert_ne!(handoff, compaction);
    assert_ne!(handoff, both);
    assert_ne!(compaction, both);
}

#[test]
fn all_mode_values_enforce_the_exact_learning_and_outcome_shapes() {
    let shapes = [
        None,
        Some(json!({})),
        Some(json!({"version": 1, "embedder": "embedding-main"})),
        Some(complete_learning("embedding-main")),
        Some(complete_active_learning("embedding-main")),
    ];
    for mode in ["off", "shadow"] {
        for learning in &shapes {
            let mut document = valid_document();
            document["mode"] = json!(mode);
            if let Some(learning) = learning {
                document["pools"][0]["learning"] = learning.clone();
            }
            let result = report(document);
            assert!(
                !result.has_errors(),
                "{mode} {learning:?}: {:?}",
                result.diagnostics
            );
        }
    }

    for learning in [
        complete_learning("embedding-main"),
        complete_active_learning("embedding-main"),
    ] {
        let mut document = valid_document();
        document["mode"] = json!("recommend");
        document["pools"][0]["learning"] = learning;
        let result = report(document);
        assert!(!result.has_errors(), "{:?}", result.diagnostics);
    }
    for learning in &shapes[..3] {
        let mut document = valid_document();
        document["mode"] = json!("recommend");
        if let Some(learning) = learning {
            document["pools"][0]["learning"] = learning.clone();
        }
        let result = report(document);
        assert!(has_diagnostic(
            &result,
            UNSUPPORTED_MODE,
            Some("pools[0].learning")
        ));
    }

    let mut active = valid_document();
    enable_active(&mut active);
    let active = report(active);
    assert!(!active.has_errors(), "{:?}", active.diagnostics);

    let mut missing_learning = valid_document();
    missing_learning["mode"] = json!("active");
    missing_learning["pools"][0]["outcome"] = valid_outcome();
    let missing_learning = report(missing_learning);
    assert!(has_diagnostic(
        &missing_learning,
        UNSUPPORTED_MODE,
        Some("pools[0].learning")
    ));

    let mut missing_outcome = valid_document();
    missing_outcome["mode"] = json!("active");
    missing_outcome["pools"][0]["learning"] = complete_active_learning("embedding-main");
    let missing_outcome = report(missing_outcome);
    assert!(has_diagnostic(
        &missing_outcome,
        UNSUPPORTED_MODE,
        Some("pools[0].outcome")
    ));

    let malformed = report(json!({"mode": "unknown"}));
    assert!(has_diagnostic(&malformed, INVALID_PLUGIN_CONFIG, None));
}

#[test]
fn unknown_fields_follow_policy_at_every_supported_level() {
    let warning = report(json!({"z_unknown": true, "a_unknown": true}));
    assert_eq!(warning.diagnostics.len(), 2);
    assert_eq!(warning.diagnostics[0].field.as_deref(), Some("a_unknown"));
    assert_eq!(warning.diagnostics[1].field.as_deref(), Some("z_unknown"));
    assert!(
        warning
            .diagnostics
            .iter()
            .all(|diagnostic| diagnostic.level == DiagnosticLevel::Warning)
    );

    let ignored = report(json!({
        "ignored": 1,
        "policy": {"unknown_field": "ignore", "policy_extra": 1}
    }));
    assert!(ignored.diagnostics.is_empty());

    let errors = report(json!({
        "root_extra": 1,
        "policy": {"unknown_field": "error", "policy_extra": 1}
    }));
    assert!(has_diagnostic(&errors, UNKNOWN_FIELD, Some("root_extra")));
    assert!(has_diagnostic(
        &errors,
        UNKNOWN_FIELD,
        Some("policy.policy_extra")
    ));
    assert!(errors.has_errors());

    let mut nested = valid_document();
    nested["pools"][0]["candidates"][0]["capabilities"]["extra"] = json!(true);
    let nested = report(nested);
    assert!(has_diagnostic(
        &nested,
        UNKNOWN_FIELD,
        Some("pools[0].candidates[0].capabilities.extra")
    ));
}

#[test]
fn judge_shape_errors_are_required_indexed_and_policy_independent() {
    let mut missing_section = valid_document();
    missing_section["pools"][0]
        .as_object_mut()
        .unwrap()
        .remove("judge");
    let missing_section = report(missing_section);
    assert!(missing_section.config.is_none());
    assert!(has_diagnostic(
        &missing_section,
        INVALID_PLUGIN_CONFIG,
        Some("pools[0].judge")
    ));

    for field in [
        "version",
        "model",
        "model_revision",
        "prompt_version",
        "rubric_version",
        "output_schema_version",
        "response_weight",
        "trajectory_weight",
        "response_floor",
        "trajectory_floor",
        "judge_confidence_floor",
        "pass_threshold",
        "max_rationale_bytes",
        "base_cooloff_seconds",
        "max_cooloff_seconds",
    ] {
        let mut missing_field = valid_document();
        missing_field["pools"][0]["judge"]
            .as_object_mut()
            .unwrap()
            .remove(field);
        let missing_field = report(missing_field);
        assert!(has_diagnostic(
            &missing_field,
            INVALID_PLUGIN_CONFIG,
            Some(format!("pools[0].judge.{field}").as_str())
        ));
    }

    let mut wrong_type = valid_document();
    wrong_type["pools"][0]["judge"]["response_weight"] = json!("0.5");
    let wrong_type = report(wrong_type);
    assert!(has_diagnostic(
        &wrong_type,
        INVALID_PLUGIN_CONFIG,
        Some("pools[0].judge.response_weight")
    ));

    let mut wrong_temperature = valid_document();
    wrong_temperature["pools"][0]["judge"]["temperature"] = json!("1.0");
    let wrong_temperature = report(wrong_temperature);
    assert!(has_diagnostic(
        &wrong_temperature,
        INVALID_PLUGIN_CONFIG,
        Some("pools[0].judge.temperature")
    ));

    for behavior in ["ignore", "warn", "error"] {
        let mut unknown = valid_document();
        unknown["policy"] = json!({"unknown_field": behavior});
        unknown["pools"][0]["judge"]["pass_treshold"] = json!(0.5);
        let unknown = report(unknown);
        let diagnostic = unknown
            .diagnostics
            .iter()
            .find(|diagnostic| {
                diagnostic.code == UNKNOWN_FIELD
                    && diagnostic.field.as_deref() == Some("pools[0].judge.pass_treshold")
            })
            .unwrap();
        assert_eq!(diagnostic.level, DiagnosticLevel::Error, "{behavior}");
        assert_eq!(
            unknown
                .diagnostics
                .iter()
                .filter(|diagnostic| {
                    diagnostic.code == UNKNOWN_FIELD
                        && diagnostic.field.as_deref() == Some("pools[0].judge.pass_treshold")
                })
                .count(),
            1,
            "{behavior}"
        );
    }

    let mut unrelated_parse_failure = valid_document();
    unrelated_parse_failure["pools"][0]["judge"]["pass_treshold"] = json!(0.5);
    unrelated_parse_failure["retention_days"] = json!("thirty");
    let unrelated_parse_failure = report(unrelated_parse_failure);
    assert!(unrelated_parse_failure.config.is_none());
    assert!(has_diagnostic(
        &unrelated_parse_failure,
        UNKNOWN_FIELD,
        Some("pools[0].judge.pass_treshold")
    ));

    let mut off = valid_document();
    off["mode"] = json!("off");
    off["pools"][0]["judge"]["prompt_version"] = json!("unregistered");
    let off = report(off);
    assert!(has_diagnostic(
        &off,
        INVALID_REFERENCE,
        Some("pools[0].judge.prompt_version")
    ));
}

#[test]
fn learning_disabled_minimal_and_complete_shapes_have_exact_semantics() {
    let omitted = report(valid_document());
    assert!(!omitted.has_errors(), "{:?}", omitted.diagnostics);
    let omitted_generation = omitted.config_generation_id.clone().unwrap();
    let omitted_config = omitted.config.unwrap();
    assert!(omitted_config.pools[0].learning.is_none());
    assert_eq!(
        serde_json::to_value(&omitted_config).unwrap()["pools"][0]["learning"],
        json!({})
    );

    let mut empty_document = valid_document();
    empty_document["pools"][0]["learning"] = json!({});
    let empty = report(empty_document);
    assert!(!empty.has_errors(), "{:?}", empty.diagnostics);
    assert!(empty.config.as_ref().unwrap().pools[0].learning.is_none());
    assert_eq!(
        empty.config_generation_id.as_deref(),
        Some(omitted_generation.as_str())
    );

    let mut enabled_document = valid_document();
    enable_learning(&mut enabled_document, "embedding-main");
    let enabled = report(enabled_document);
    assert!(!enabled.has_errors(), "{:?}", enabled.diagnostics);
    let learning = enabled.config.as_ref().unwrap().pools[0]
        .learning
        .as_ref()
        .unwrap();
    assert_eq!(learning.version, 1);
    assert_eq!(learning.embedder, "embedding-main");
    assert_ne!(
        enabled.config_generation_id.as_deref(),
        Some(omitted_generation.as_str())
    );

    let mut complete_document = valid_document();
    enable_complete_learning(&mut complete_document, "embedding-main");
    let complete = report(complete_document);
    assert!(!complete.has_errors(), "{:?}", complete.diagnostics);
    let learning = complete.config.as_ref().unwrap().pools[0]
        .learning
        .as_ref()
        .unwrap();
    assert_eq!(learning.top_k, Some(8));
    assert_eq!(learning.radius, Some(0.25));
    assert_eq!(learning.min_points, Some(3));
    assert_eq!(learning.min_independent_roots, Some(2));
    assert_eq!(learning.min_effective_samples, Some(1.5));
    assert_eq!(learning.min_coverage, Some(0.8));
    assert_eq!(learning.time_decay_half_life_seconds, Some(3600.0));
    assert_eq!(learning.prior_success, Some(1.0));
    assert_eq!(learning.prior_failure, Some(1.0));
    assert_eq!(learning.familywise_credible_level, Some(0.95));
    assert_eq!(learning.promotion_lower_bound, Some(0.9));
    assert_ne!(complete.config_generation_id, enabled.config_generation_id);
}

#[test]
fn legacy_off_shadow_and_recommend_config_generations_are_fixed() {
    let generation = |document: Json| {
        let result = report(document);
        assert!(!result.has_errors(), "{:?}", result.diagnostics);
        result.config_generation_id.unwrap()
    };

    let mut off = valid_document();
    off["mode"] = json!("off");
    let mut recommend = valid_document();
    recommend["mode"] = json!("recommend");
    enable_complete_learning(&mut recommend, "embedding-main");

    assert_eq!(
        [
            generation(off),
            generation(valid_document()),
            generation(recommend),
        ],
        [
            "72c9b42fa716d5c9d9dbaa7ba65d6646f4460bb0a2313019e38abaf5565fdf64",
            "19c5ae95950293c28ea77ac24686e4e3952bdeec2979ecb4f9dac318e46e0b43",
            "6245558244f918596020e638ec18c65c25b24d16c4275b7690a360a6d0007884",
        ]
    );
}

#[test]
fn complete_active_learning_is_the_fourth_exact_shape() {
    let mut recommend_document = valid_document();
    enable_complete_learning(&mut recommend_document, "embedding-main");
    let recommend = report(recommend_document);
    assert!(!recommend.has_errors(), "{:?}", recommend.diagnostics);
    let recommend_json = serde_json::to_value(recommend.config.as_ref().unwrap()).unwrap();
    for field in [
        "retention_lower_bound",
        "holdout_probability",
        "active_canary_fraction",
    ] {
        assert!(recommend_json["pools"][0]["learning"].get(field).is_none());
    }

    let mut active_document = valid_document();
    active_document["pools"][0]["learning"] = complete_active_learning("embedding-main");
    let active = report(active_document);
    assert!(!active.has_errors(), "{:?}", active.diagnostics);
    let learning = active.config.as_ref().unwrap().pools[0]
        .learning
        .as_ref()
        .unwrap();
    assert_eq!(learning.retention_lower_bound, Some(0.8));
    assert_eq!(learning.holdout_probability, Some(0.2));
    assert_eq!(learning.active_canary_fraction, Some(0.4));
    assert_ne!(active.config_generation_id, recommend.config_generation_id);

    let serialized = serde_json::to_value(active.config.unwrap()).unwrap();
    assert_eq!(
        serialized["pools"][0]["learning"],
        complete_active_learning("embedding-main")
    );

    for missing in [
        "retention_lower_bound",
        "holdout_probability",
        "active_canary_fraction",
    ] {
        let mut learning = complete_active_learning("embedding-main");
        learning.as_object_mut().unwrap().remove(missing);
        let mut document = valid_document();
        document["pools"][0]["learning"] = learning;
        let result = report(document);
        assert!(
            has_diagnostic(
                &result,
                INVALID_PLUGIN_CONFIG,
                Some(&format!("pools[0].learning.{missing}"))
            ),
            "{missing}: {:?}",
            result.diagnostics
        );
    }
}

#[test]
fn pool_outcome_raw_map_and_empty_generation_remain_compatible() {
    fn assert_raw_map(_: &BTreeMap<String, Json>) {}

    let omitted = report(valid_document());
    assert!(!omitted.has_errors(), "{:?}", omitted.diagnostics);
    let omitted_generation = omitted.config_generation_id.clone();
    let omitted_config = omitted.config.as_ref().unwrap();
    assert_raw_map(&omitted_config.pools[0].outcome);
    assert!(omitted_config.pools[0].outcome.is_empty());
    assert_eq!(
        serde_json::to_value(omitted_config).unwrap()["pools"][0]["outcome"],
        json!({})
    );

    let mut explicit_empty = valid_document();
    explicit_empty["pools"][0]["outcome"] = json!({});
    let explicit_empty = report(explicit_empty);
    assert!(
        !explicit_empty.has_errors(),
        "{:?}",
        explicit_empty.diagnostics
    );
    assert_eq!(explicit_empty.config_generation_id, omitted_generation);

    let mut complete = valid_document();
    complete["pools"][0]["outcome"] = valid_outcome();
    let complete = report(complete);
    assert!(!complete.has_errors(), "{:?}", complete.diagnostics);
    let raw = &complete.config.as_ref().unwrap().pools[0].outcome;
    assert_eq!(
        Json::Object(raw.clone().into_iter().collect()),
        valid_outcome()
    );
    let typed: OutcomeConfig = serde_json::from_value(valid_outcome()).unwrap();
    let _: &OutcomeMatcher = &typed.success_matchers[0];
    assert_eq!(
        typed.success_matchers[0].event_kind,
        OutcomeMatcherEventKind::ScopeEnd
    );
    assert_eq!(
        typed.success_matchers[0].terminal_status,
        OutcomeTerminalStatus::Ok
    );
    assert_eq!(typed.completion_disposition, OutcomeDisposition::Success);
    assert_eq!(typed.error_disposition, OutcomeDisposition::Failure);
    assert_ne!(complete.config_generation_id, omitted_generation);
}

#[test]
fn outcome_shape_errors_are_exact_indexed_and_policy_independent() {
    let required = [
        "version",
        "success_matchers",
        "failure_matchers",
        "completion_disposition",
        "error_disposition",
        "tool_failure_disposition",
        "end_of_run_disposition",
        "max_attribution_seconds",
        "actual_outcome_half_life_seconds",
        "anchor_shadow_half_life_seconds",
        "relearning_cooloff_seconds",
        "min_treatment_roots",
        "min_control_roots",
        "min_treatment_effective_weight",
        "min_control_effective_weight",
        "noninferiority_margin",
        "noninferiority_probability",
        "rollback_probability",
        "outcome_evaluation_batch_size",
        "max_canary_roots",
        "authorization_ttl_seconds",
    ];
    for missing in required {
        let mut outcome = valid_outcome();
        outcome.as_object_mut().unwrap().remove(missing);
        let mut document = valid_document();
        document["pools"][0]["outcome"] = outcome;
        let result = report(document);
        assert!(
            has_diagnostic(
                &result,
                INVALID_PLUGIN_CONFIG,
                Some(&format!("pools[0].outcome.{missing}"))
            ),
            "{missing}: {:?}",
            result.diagnostics
        );
    }

    for (pointer, code, field) in [
        (
            "/pools/0/outcome/completion_disposition",
            INVALID_REFERENCE,
            "pools[0].outcome.completion_disposition",
        ),
        (
            "/pools/0/outcome/success_matchers/0/event_kind",
            INVALID_REFERENCE,
            "pools[0].outcome.success_matchers[0].event_kind",
        ),
        (
            "/pools/0/outcome/failure_matchers/0/terminal_status",
            INVALID_REFERENCE,
            "pools[0].outcome.failure_matchers[0].terminal_status",
        ),
    ] {
        let mut document = valid_document();
        document["pools"][0]["outcome"] = valid_outcome();
        *document.pointer_mut(pointer).unwrap() = json!("unknown");
        let result = report(document);
        assert!(
            has_diagnostic(&result, code, Some(field)),
            "{pointer}: {:?}",
            result.diagnostics
        );
    }

    for behavior in ["ignore", "warn", "error"] {
        let mut document = valid_document();
        document["policy"] = json!({"unknown_field": behavior});
        document["pools"][0]["outcome"] = valid_outcome();
        document["pools"][0]["outcome"]["extra"] = json!(true);
        document["pools"][0]["outcome"]["success_matchers"][0]["extra"] = json!(true);
        let result = report(document);
        for field in [
            "pools[0].outcome.extra",
            "pools[0].outcome.success_matchers[0].extra",
        ] {
            let diagnostic = result
                .diagnostics
                .iter()
                .find(|diagnostic| {
                    diagnostic.code == UNKNOWN_FIELD && diagnostic.field.as_deref() == Some(field)
                })
                .unwrap_or_else(|| panic!("{behavior} missing {field}: {:?}", result.diagnostics));
            assert_eq!(diagnostic.level, DiagnosticLevel::Error);
        }
    }
}

#[test]
fn outcome_matchers_enforce_safe_exact_scalars_and_global_uniqueness() {
    let mut all_scalars = valid_outcome();
    all_scalars["success_matchers"][0]["metadata_equals"] = json!({
        "error.type": null,
        "outcome": true,
        "outcome.label": "passed",
        "outcome.success": 1,
        "result": 9_007_199_254_740_991.0,
        "status": -7,
        "success": false
    });
    let mut document = valid_document();
    document["pools"][0]["outcome"] = all_scalars;
    let result = report(document);
    assert!(!result.has_errors(), "{:?}", result.diagnostics);

    for (key, value) in [
        ("otel.status_code", json!("ok")),
        ("secret", json!("passed")),
        ("status", json!("")),
        ("status", json!("Upper")),
        ("status", json!("non_ascii_é")),
        ("status", json!(["nested"])),
        ("status", json!({"nested": true})),
        ("status", json!(-0.0)),
        ("status", json!(9_007_199_254_740_992_u64)),
        ("status", json!(9_007_199_254_740_992.0_f64)),
        ("status", json!(-9_007_199_254_740_992.0_f64)),
    ] {
        let mut document = valid_document();
        document["pools"][0]["outcome"] = valid_outcome();
        document["pools"][0]["outcome"]["success_matchers"][0]["metadata_equals"] =
            json!({key: value});
        let result = report(document);
        assert!(result.has_errors(), "{key}: {value:?}");
    }

    let mut mark = valid_document();
    mark["pools"][0]["outcome"] = valid_outcome();
    mark["pools"][0]["outcome"]["success_matchers"][0]["event_kind"] = json!("mark");
    let mark = report(mark);
    assert!(has_diagnostic(
        &mark,
        INVALID_RANGE,
        Some("pools[0].outcome.success_matchers[0].terminal_status")
    ));

    let mut duplicate = valid_document();
    duplicate["pools"][0]["outcome"] = valid_outcome();
    duplicate["pools"][0]["outcome"]["failure_matchers"][0] =
        duplicate["pools"][0]["outcome"]["success_matchers"][0].clone();
    let duplicate = report(duplicate);
    assert!(has_diagnostic(
        &duplicate,
        DUPLICATE_ID,
        Some("pools[0].outcome.failure_matchers[0]")
    ));
}

#[test]
fn active_learning_probabilities_and_hysteresis_are_exactly_bounded() {
    let mut boundary = valid_document();
    let mut learning = complete_active_learning("embedding-main");
    learning["retention_lower_bound"] = json!(0.0);
    learning["holdout_probability"] = json!(0.25);
    learning["active_canary_fraction"] = json!(0.25);
    boundary["pools"][0]["learning"] = learning;
    let boundary = report(boundary);
    assert!(!boundary.has_errors(), "{:?}", boundary.diagnostics);

    for (field, value) in [
        ("retention_lower_bound", json!(-0.1)),
        ("retention_lower_bound", json!(0.9)),
        ("holdout_probability", json!(0.0)),
        ("holdout_probability", json!(0.250_000_1)),
        ("active_canary_fraction", json!(0.0)),
        ("active_canary_fraction", json!(0.8)),
        ("active_canary_fraction", json!(f64::from_bits(1))),
    ] {
        let mut document = valid_document();
        document["pools"][0]["learning"] = complete_active_learning("embedding-main");
        document["pools"][0]["learning"][field] = value;
        let result = report(document);
        assert!(
            has_diagnostic(
                &result,
                INVALID_RANGE,
                Some(&format!("pools[0].learning.{field}"))
            ),
            "{field}: {:?}",
            result.diagnostics
        );
    }

    let mut typed = report(valid_document()).config.unwrap();
    typed.pools[0].learning =
        Some(serde_json::from_value(complete_active_learning("embedding-main")).unwrap());
    typed.pools[0]
        .learning
        .as_mut()
        .unwrap()
        .holdout_probability = Some(f64::NAN);
    assert!(typed.validate().iter().any(|diagnostic| {
        diagnostic.code == INVALID_RANGE
            && diagnostic.field.as_deref() == Some("pools[0].learning.holdout_probability")
    }));
}

#[test]
fn outcome_numeric_and_cross_field_bounds_fail_closed() {
    let cases = [
        (
            "/pools/0/outcome/min_treatment_roots",
            json!(31),
            "min_treatment_roots",
        ),
        (
            "/pools/0/outcome/min_control_roots",
            json!(31),
            "min_control_roots",
        ),
        (
            "/pools/0/outcome/min_treatment_effective_weight",
            json!(15.9),
            "min_treatment_effective_weight",
        ),
        (
            "/pools/0/outcome/min_control_effective_weight",
            json!(65.0),
            "min_control_effective_weight",
        ),
        (
            "/pools/0/outcome/noninferiority_margin",
            json!(-0.0),
            "noninferiority_margin",
        ),
        (
            "/pools/0/outcome/noninferiority_margin",
            json!(0.251),
            "noninferiority_margin",
        ),
        (
            "/pools/0/outcome/noninferiority_probability",
            json!(0.989),
            "noninferiority_probability",
        ),
        (
            "/pools/0/outcome/rollback_probability",
            json!(0.949),
            "rollback_probability",
        ),
        (
            "/pools/0/outcome/outcome_evaluation_batch_size",
            json!(63),
            "outcome_evaluation_batch_size",
        ),
        (
            "/pools/0/outcome/outcome_evaluation_batch_size",
            json!(OUTCOME_EVALUATION_BATCH_SIZE_MAX + 1),
            "outcome_evaluation_batch_size",
        ),
        (
            "/pools/0/outcome/max_canary_roots",
            json!(0),
            "max_canary_roots",
        ),
        (
            "/pools/0/outcome/max_canary_roots",
            json!(OUTCOME_MAX_CANARY_ROOTS_MAX + 1),
            "max_canary_roots",
        ),
        (
            "/pools/0/outcome/max_attribution_seconds",
            json!(0),
            "max_attribution_seconds",
        ),
        (
            "/pools/0/outcome/relearning_cooloff_seconds",
            json!(OUTCOME_DURATION_SECONDS_MAX + 1),
            "relearning_cooloff_seconds",
        ),
        (
            "/pools/0/outcome/authorization_ttl_seconds",
            json!(1801),
            "authorization_ttl_seconds",
        ),
        (
            "/pools/0/outcome/actual_outcome_half_life_seconds",
            json!(3600),
            "actual_outcome_half_life_seconds",
        ),
    ];
    for (pointer, value, field) in cases {
        let mut document = valid_document();
        enable_active(&mut document);
        *document.pointer_mut(pointer).unwrap() = value;
        let result = report(document);
        assert!(
            has_diagnostic(
                &result,
                INVALID_RANGE,
                Some(&format!("pools[0].outcome.{field}"))
            ),
            "{pointer}: {:?}",
            result.diagnostics
        );
    }

    let mut too_many_looks = valid_document();
    enable_active(&mut too_many_looks);
    too_many_looks["pools"][0]["outcome"]["max_canary_roots"] = json!((OUTCOME_MAX_LOOKS + 1) * 64);
    let too_many_looks = report(too_many_looks);
    assert!(has_diagnostic(
        &too_many_looks,
        INVALID_RANGE,
        Some("pools[0].outcome.max_canary_roots")
    ));

    let mut rounded_threshold = valid_document();
    enable_active(&mut rounded_threshold);
    rounded_threshold["pools"][0]["outcome"]["max_canary_roots"] = json!(128);
    rounded_threshold["pools"][0]["outcome"]["noninferiority_probability"] =
        json!(f64::from_bits(1.0_f64.to_bits() - 1));
    let rounded_threshold = report(rounded_threshold);
    assert!(has_diagnostic(
        &rounded_threshold,
        INVALID_RANGE,
        Some("pools[0].outcome.noninferiority_probability")
    ));

    for (canary, field) in [(0.2, "min_treatment_roots"), (0.6, "min_control_roots")] {
        let mut underpowered = valid_document();
        enable_active(&mut underpowered);
        underpowered["pools"][0]["learning"]["active_canary_fraction"] = json!(canary);
        let underpowered = report(underpowered);
        assert!(
            has_diagnostic(
                &underpowered,
                INVALID_RANGE,
                Some(&format!("pools[0].outcome.{field}"))
            ),
            "{canary}: {:?}",
            underpowered.diagnostics
        );
    }
}

#[test]
fn outcome_matcher_and_list_resource_boundaries_are_enforced() {
    let mut exact = valid_document();
    exact["pools"][0]["outcome"] = valid_outcome();
    exact["pools"][0]["outcome"]["success_matchers"] = Json::Array(
        (0..OUTCOME_MATCHERS_MAX)
            .map(|index| {
                let mut matcher = valid_outcome()["success_matchers"][0].clone();
                matcher["name"] = json!(format!("completed-{index}"));
                matcher
            })
            .collect(),
    );
    let exact = report(exact);
    assert!(!exact.has_errors(), "{:?}", exact.diagnostics);

    for count in [0, OUTCOME_MATCHERS_MAX + 1] {
        let mut document = valid_document();
        document["pools"][0]["outcome"] = valid_outcome();
        document["pools"][0]["outcome"]["success_matchers"] = Json::Array(
            (0..count)
                .map(|index| {
                    let mut matcher = valid_outcome()["success_matchers"][0].clone();
                    matcher["name"] = json!(format!("completed-{index}"));
                    matcher
                })
                .collect(),
        );
        let result = report(document);
        assert!(has_diagnostic(
            &result,
            INVALID_RANGE,
            Some("pools[0].outcome.success_matchers")
        ));
    }

    for (field, length, valid) in [
        ("category", OUTCOME_MATCHER_TEXT_MAX_BYTES, true),
        ("name", OUTCOME_MATCHER_TEXT_MAX_BYTES, true),
        ("category", OUTCOME_MATCHER_TEXT_MAX_BYTES + 1, false),
        ("name", OUTCOME_MATCHER_TEXT_MAX_BYTES + 1, false),
    ] {
        let mut document = valid_document();
        document["pools"][0]["outcome"] = valid_outcome();
        document["pools"][0]["outcome"]["success_matchers"][0][field] = json!("x".repeat(length));
        let result = report(document);
        assert_eq!(
            !result.has_errors(),
            valid,
            "{field} {length}: {:?}",
            result.diagnostics
        );
    }

    for (length, valid) in [
        (OUTCOME_LABEL_MAX_BYTES, true),
        (OUTCOME_LABEL_MAX_BYTES + 1, false),
    ] {
        let mut document = valid_document();
        document["pools"][0]["outcome"] = valid_outcome();
        document["pools"][0]["outcome"]["success_matchers"][0]["metadata_equals"] =
            json!({"status": "x".repeat(length)});
        let result = report(document);
        assert_eq!(
            !result.has_errors(),
            valid,
            "label {length}: {:?}",
            result.diagnostics
        );
    }

    let mut clauses = valid_document();
    clauses["pools"][0]["outcome"] = valid_outcome();
    clauses["pools"][0]["outcome"]["success_matchers"][0]["metadata_equals"] = Json::Object(
        (0..=OUTCOME_METADATA_EQUALS_MAX)
            .map(|index| (format!("unknown-{index}"), json!(true)))
            .collect(),
    );
    let clauses = report(clauses);
    assert!(has_diagnostic(
        &clauses,
        INVALID_RANGE,
        Some("pools[0].outcome.success_matchers[0].metadata_equals")
    ));
}

#[test]
fn protected_outcome_identity_is_order_independent_and_semantic() {
    let generation = |outcome: Json| {
        let mut document = valid_document();
        document["pools"][0]["outcome"] = outcome;
        let result = report(document);
        assert!(!result.has_errors(), "{:?}", result.diagnostics);
        result.config_generation_id.unwrap()
    };

    let mut first = valid_outcome();
    let mut second_matcher = first["success_matchers"][0].clone();
    second_matcher["name"] = json!("completed-later");
    second_matcher["metadata_equals"] = json!({"status": 1, "result": true});
    first["success_matchers"]
        .as_array_mut()
        .unwrap()
        .push(second_matcher);
    let mut reversed = first.clone();
    reversed["success_matchers"]
        .as_array_mut()
        .unwrap()
        .reverse();
    assert_eq!(generation(first.clone()), generation(reversed));

    let mut integer = first.clone();
    integer["success_matchers"][0]["metadata_equals"] = json!({"status": 1});
    let mut float = integer.clone();
    float["success_matchers"][0]["metadata_equals"] = json!({"status": 1.0});
    let integer_generation = generation(integer.clone());
    assert_eq!(integer_generation, generation(float));

    let mut changed = integer;
    changed["success_matchers"][0]["metadata_equals"] =
        json!({"status": f64::from_bits(1.0_f64.to_bits() + 1)});
    assert_ne!(integer_generation, generation(changed));
}

#[test]
fn learning_shape_errors_are_indexed_and_policy_independent() {
    let cases = [
        (json!(null), INVALID_PLUGIN_CONFIG, "pools[0].learning"),
        (json!([]), INVALID_PLUGIN_CONFIG, "pools[0].learning"),
        (
            json!("disabled"),
            INVALID_PLUGIN_CONFIG,
            "pools[0].learning",
        ),
        (
            json!({"version": 1}),
            INVALID_PLUGIN_CONFIG,
            "pools[0].learning.embedder",
        ),
        (
            json!({"embedder": "embedding-main"}),
            INVALID_PLUGIN_CONFIG,
            "pools[0].learning.version",
        ),
        (
            json!({"version": "1", "embedder": "embedding-main"}),
            INVALID_PLUGIN_CONFIG,
            "pools[0].learning.version",
        ),
        (
            json!({"version": 1, "embedder": 7}),
            INVALID_PLUGIN_CONFIG,
            "pools[0].learning.embedder",
        ),
        (
            json!({"version": 1, "embedder": "embedding-main", "extra": true}),
            UNKNOWN_FIELD,
            "pools[0].learning.extra",
        ),
    ];

    for behavior in ["ignore", "warn", "error"] {
        for (learning, code, field) in &cases {
            let mut document = valid_document();
            document["policy"] = json!({"unknown_field": behavior});
            document["pools"][0]["learning"] = learning.clone();
            let result = report(document);
            let diagnostic = result
                .diagnostics
                .iter()
                .find(|diagnostic| {
                    diagnostic.code == *code && diagnostic.field.as_deref() == Some(*field)
                })
                .unwrap_or_else(|| {
                    panic!(
                        "{behavior} missing {code} at {field}: {:?}",
                        result.diagnostics
                    )
                });
            assert_eq!(
                diagnostic.level,
                DiagnosticLevel::Error,
                "{behavior}: {field}"
            );
        }
    }

    let statistical_fields = [
        "top_k",
        "radius",
        "min_points",
        "min_independent_roots",
        "min_effective_samples",
        "min_coverage",
        "time_decay_half_life_seconds",
        "prior_success",
        "prior_failure",
        "familywise_credible_level",
        "promotion_lower_bound",
    ];
    for missing in statistical_fields {
        let mut learning = complete_learning("embedding-main");
        learning.as_object_mut().unwrap().remove(missing);
        let mut document = valid_document();
        document["pools"][0]["learning"] = learning;
        let result = report(document);
        assert!(
            has_diagnostic(
                &result,
                INVALID_PLUGIN_CONFIG,
                Some(&format!("pools[0].learning.{missing}"))
            ),
            "{missing}: {:?}",
            result.diagnostics
        );
    }

    for (field, invalid) in [
        ("top_k", json!(1.5)),
        ("radius", json!("0.5")),
        ("min_points", json!(-1)),
        ("min_independent_roots", json!(true)),
        ("min_effective_samples", json!(null)),
        ("min_coverage", json!([])),
        ("time_decay_half_life_seconds", json!({})),
        ("prior_success", json!("1")),
        ("prior_failure", json!(false)),
        ("familywise_credible_level", json!("0.95")),
        ("promotion_lower_bound", json!(null)),
    ] {
        let mut document = valid_document();
        enable_complete_learning(&mut document, "embedding-main");
        document["pools"][0]["learning"][field] = invalid;
        let result = report(document);
        assert!(
            has_diagnostic(
                &result,
                INVALID_PLUGIN_CONFIG,
                Some(&format!("pools[0].learning.{field}"))
            ),
            "{field}: {:?}",
            result.diagnostics
        );
    }
}

#[test]
fn learning_version_and_embedder_reference_are_exact() {
    let mut unsupported_version = valid_document();
    unsupported_version["pools"][0]["learning"] =
        json!({"version": 2, "embedder": "embedding-main"});
    let unsupported_version = report(unsupported_version);
    assert!(has_diagnostic(
        &unsupported_version,
        UNSUPPORTED_CONFIG_VERSION,
        Some("pools[0].learning.version")
    ));
    assert!(unsupported_version.config_generation_id.is_none());

    let mut unknown_embedder = valid_document();
    enable_learning(&mut unknown_embedder, "embedding-missing");
    let unknown_embedder = report(unknown_embedder);
    assert!(has_diagnostic(
        &unknown_embedder,
        INVALID_REFERENCE,
        Some("pools[0].learning.embedder")
    ));
    assert!(unknown_embedder.config_generation_id.is_none());
}

#[test]
fn complete_learning_numeric_bounds_are_exact() {
    let invalid = [
        ("top_k", json!(0)),
        ("top_k", json!(LEARNING_TOP_K_MAX + 1)),
        ("radius", json!(0.0)),
        ("radius", json!(-0.1)),
        ("radius", json!(2.000_001)),
        ("min_points", json!(0)),
        ("min_points", json!(9)),
        ("min_independent_roots", json!(0)),
        ("min_independent_roots", json!(9)),
        ("min_effective_samples", json!(0.0)),
        ("min_effective_samples", json!(8.5)),
        ("min_coverage", json!(-0.1)),
        ("min_coverage", json!(1.1)),
        ("time_decay_half_life_seconds", json!(0.0)),
        ("prior_success", json!(0.0)),
        ("prior_failure", json!(-1.0)),
        ("familywise_credible_level", json!(0.5)),
        ("familywise_credible_level", json!(1.0)),
        ("promotion_lower_bound", json!(-0.1)),
        ("promotion_lower_bound", json!(1.1)),
    ];
    for (field, value) in invalid {
        let mut document = valid_document();
        enable_complete_learning(&mut document, "embedding-main");
        document["pools"][0]["learning"][field] = value;
        let result = report(document);
        assert!(
            has_diagnostic(
                &result,
                INVALID_RANGE,
                Some(&format!("pools[0].learning.{field}"))
            ),
            "{field}: {:?}",
            result.diagnostics
        );
    }

    let mut boundaries = valid_document();
    enable_complete_learning(&mut boundaries, "embedding-main");
    let learning = &mut boundaries["pools"][0]["learning"];
    learning["top_k"] = json!(LEARNING_TOP_K_MAX);
    learning["radius"] = json!(2.0);
    learning["min_points"] = json!(LEARNING_TOP_K_MAX);
    learning["min_independent_roots"] = json!(LEARNING_TOP_K_MAX);
    learning["min_effective_samples"] = json!(LEARNING_TOP_K_MAX as f64);
    learning["min_coverage"] = json!(0.0);
    learning["promotion_lower_bound"] = json!(1.0);
    learning["familywise_credible_level"] = json!(f64::from_bits(1.0_f64.to_bits() - 1));
    let boundaries = report(boundaries);
    assert!(!boundaries.has_errors(), "{:?}", boundaries.diagnostics);
}

#[test]
fn complete_learning_rejects_nonfinite_values_and_accepts_positive_subnormals() {
    let float_fields = [
        "radius",
        "min_effective_samples",
        "min_coverage",
        "time_decay_half_life_seconds",
        "prior_success",
        "prior_failure",
        "familywise_credible_level",
        "promotion_lower_bound",
    ];
    for field in float_fields {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut document = valid_document();
            enable_complete_learning(&mut document, "embedding-main");
            let result = report(document);
            assert!(!result.has_errors(), "{:?}", result.diagnostics);
            let mut config = result.config.unwrap();
            let learning = config.pools[0].learning.as_mut().unwrap();
            set_learning_float(learning, field, value);
            assert!(config.validate().iter().any(|diagnostic| {
                diagnostic.code == INVALID_RANGE
                    && diagnostic.field.as_deref() == Some(&format!("pools[0].learning.{field}"))
            }));
        }
    }

    let mut document = valid_document();
    enable_complete_learning(&mut document, "embedding-main");
    let mut config = report(document).config.unwrap();
    let learning = config.pools[0].learning.as_mut().unwrap();
    let subnormal = f64::from_bits(1);
    learning.radius = Some(subnormal);
    learning.min_effective_samples = Some(subnormal);
    learning.min_coverage = Some(subnormal);
    learning.time_decay_half_life_seconds = Some(subnormal);
    learning.prior_success = Some(subnormal);
    learning.prior_failure = Some(subnormal);
    learning.promotion_lower_bound = Some(subnormal);
    assert!(config.validate().is_empty(), "{:?}", config.validate());
}

fn set_learning_float(learning: &mut LearningConfig, field: &str, value: f64) {
    match field {
        "radius" => learning.radius = Some(value),
        "min_effective_samples" => learning.min_effective_samples = Some(value),
        "min_coverage" => learning.min_coverage = Some(value),
        "time_decay_half_life_seconds" => learning.time_decay_half_life_seconds = Some(value),
        "prior_success" => learning.prior_success = Some(value),
        "prior_failure" => learning.prior_failure = Some(value),
        "familywise_credible_level" => learning.familywise_credible_level = Some(value),
        "promotion_lower_bound" => learning.promotion_lower_bound = Some(value),
        _ => panic!("unknown learning float field {field}"),
    }
}

#[test]
fn complete_learning_candidate_and_product_limits_are_checked() {
    let mut exact_candidates = valid_document();
    set_candidate_count(&mut exact_candidates, LEARNING_CANDIDATES_MAX);
    enable_complete_learning(&mut exact_candidates, "embedding-main");
    exact_candidates["pools"][0]["learning"]["top_k"] = json!(1);
    exact_candidates["pools"][0]["learning"]["min_points"] = json!(1);
    exact_candidates["pools"][0]["learning"]["min_independent_roots"] = json!(1);
    exact_candidates["pools"][0]["learning"]["min_effective_samples"] = json!(1.0);
    let exact_candidates = report(exact_candidates);
    assert!(
        !exact_candidates.has_errors(),
        "{:?}",
        exact_candidates.diagnostics
    );

    let mut too_many = valid_document();
    set_candidate_count(&mut too_many, LEARNING_CANDIDATES_MAX + 1);
    enable_complete_learning(&mut too_many, "embedding-main");
    too_many["pools"][0]["learning"]["top_k"] = json!(1);
    too_many["pools"][0]["learning"]["min_points"] = json!(1);
    too_many["pools"][0]["learning"]["min_independent_roots"] = json!(1);
    too_many["pools"][0]["learning"]["min_effective_samples"] = json!(1.0);
    let too_many = report(too_many);
    assert!(has_diagnostic(
        &too_many,
        INVALID_RANGE,
        Some("pools[0].candidates")
    ));

    let mut product = valid_document();
    set_candidate_count(&mut product, 2);
    enable_complete_learning(&mut product, "embedding-main");
    product["pools"][0]["learning"]["top_k"] =
        json!(LEARNING_CANDIDATE_NEIGHBOR_PRODUCT_MAX / 2 + 1);
    let product = report(product);
    assert!(has_diagnostic(
        &product,
        INVALID_RANGE,
        Some("pools[0].learning.top_k")
    ));

    let mut overflow_document = valid_document();
    set_candidate_count(&mut overflow_document, 2);
    enable_complete_learning(&mut overflow_document, "embedding-main");
    let mut config = report(overflow_document).config.unwrap();
    config.pools[0].learning.as_mut().unwrap().top_k = Some(usize::MAX);
    assert!(config.validate().iter().any(|diagnostic| {
        diagnostic.code == INVALID_RANGE
            && diagnostic.field.as_deref() == Some("pools[0].learning.top_k")
            && diagnostic.message.contains("overflows usize")
    }));
}

#[test]
fn canonicalizer_accepts_only_unique_registered_position_features() {
    let mut accepted = valid_document();
    accepted["pools"][0]["canonicalizer"]["position_features"] = json!(["turn_index"]);
    let accepted = report(accepted);
    assert!(!accepted.has_errors(), "{:?}", accepted.diagnostics);

    let mut duplicate = valid_document();
    duplicate["pools"][0]["canonicalizer"]["position_features"] =
        json!(["turn_index", "turn_index"]);
    let duplicate = report(duplicate);
    assert!(has_diagnostic(
        &duplicate,
        DUPLICATE_ID,
        Some("pools[0].canonicalizer.position_features[1]")
    ));

    let mut unknown = valid_document();
    unknown["pools"][0]["canonicalizer"]["position_features"] = json!(["message_index"]);
    let unknown = report(unknown);
    assert!(has_diagnostic(
        &unknown,
        INVALID_REFERENCE,
        Some("pools[0].canonicalizer.position_features[0]")
    ));
}

#[test]
fn judge_versions_references_and_numeric_contract_are_exact() {
    let cases: Vec<(&str, Json, &str, &str)> = vec![
        (
            "/pools/0/judge/version",
            json!(2),
            UNSUPPORTED_CONFIG_VERSION,
            "pools[0].judge.version",
        ),
        (
            "/pools/0/judge/output_schema_version",
            json!(2),
            UNSUPPORTED_CONFIG_VERSION,
            "pools[0].judge.output_schema_version",
        ),
        (
            "/pools/0/judge/model",
            json!("bad model"),
            INVALID_REFERENCE,
            "pools[0].judge.model",
        ),
        (
            "/pools/0/judge/model_revision",
            json!("latest"),
            INVALID_REFERENCE,
            "pools[0].judge.model_revision",
        ),
        (
            "/pools/0/judge/prompt_version",
            json!("pairwise-equivalence-v2"),
            INVALID_REFERENCE,
            "pools[0].judge.prompt_version",
        ),
        (
            "/pools/0/judge/rubric_version",
            json!("response-only-v1"),
            INVALID_REFERENCE,
            "pools[0].judge.rubric_version",
        ),
        (
            "/pools/0/judge/response_weight",
            json!(-0.1),
            INVALID_RANGE,
            "pools[0].judge.response_weight",
        ),
        (
            "/pools/0/judge/response_weight",
            json!(1.000_000_000_1),
            INVALID_RANGE,
            "pools[0].judge.response_weight",
        ),
        (
            "/pools/0/judge/response_floor",
            json!(1.1),
            INVALID_RANGE,
            "pools[0].judge.response_floor",
        ),
        (
            "/pools/0/judge/trajectory_floor",
            json!(-0.1),
            INVALID_RANGE,
            "pools[0].judge.trajectory_floor",
        ),
        (
            "/pools/0/judge/judge_confidence_floor",
            json!(1.1),
            INVALID_RANGE,
            "pools[0].judge.judge_confidence_floor",
        ),
        (
            "/pools/0/judge/pass_threshold",
            json!(-0.1),
            INVALID_RANGE,
            "pools[0].judge.pass_threshold",
        ),
        (
            "/pools/0/judge/max_rationale_bytes",
            json!(0),
            INVALID_RANGE,
            "pools[0].judge.max_rationale_bytes",
        ),
        (
            "/pools/0/judge/max_rationale_bytes",
            json!(JUDGE_MAX_RATIONALE_BYTES + 1),
            INVALID_RANGE,
            "pools[0].judge.max_rationale_bytes",
        ),
        (
            "/pools/0/judge/base_cooloff_seconds",
            json!(0),
            INVALID_RANGE,
            "pools[0].judge.base_cooloff_seconds",
        ),
        (
            "/pools/0/judge/max_cooloff_seconds",
            json!(0),
            INVALID_RANGE,
            "pools[0].judge.max_cooloff_seconds",
        ),
    ];
    for (pointer, value, code, field) in cases {
        let mut document = valid_document();
        *document.pointer_mut(pointer).unwrap() = value;
        let result = report(document);
        assert!(
            has_diagnostic(&result, code, Some(field)),
            "missing {code} at {field}: {:?}",
            result.diagnostics
        );
    }

    let mut temperature = valid_document();
    temperature["pools"][0]["judge"]["temperature"] = json!(1.1);
    let temperature = report(temperature);
    assert!(has_diagnostic(
        &temperature,
        INVALID_RANGE,
        Some("pools[0].judge.temperature")
    ));

    let mut wrong_sum = valid_document();
    wrong_sum["pools"][0]["judge"]["response_weight"] = json!(0.6);
    let wrong_sum = report(wrong_sum);
    assert!(has_diagnostic(
        &wrong_sum,
        INVALID_RANGE,
        Some("pools[0].judge.trajectory_weight")
    ));

    let mut tolerance = valid_document();
    tolerance["pools"][0]["judge"]["response_weight"] = json!(0.500_000_000_5);
    let tolerance = report(tolerance);
    assert!(!tolerance.has_errors(), "{:?}", tolerance.diagnostics);

    let mut cooloff = valid_document();
    cooloff["pools"][0]["judge"]["base_cooloff_seconds"] = json!(301);
    let cooloff = report(cooloff);
    assert!(has_diagnostic(
        &cooloff,
        INVALID_RANGE,
        Some("pools[0].judge.max_cooloff_seconds")
    ));
}

#[test]
fn judge_nonfinite_values_and_output_token_arithmetic_are_checked() {
    let mut config = report(valid_document()).config.unwrap();
    config.pools[0].judge.response_weight = f64::NAN;
    assert!(config.validate().iter().any(|diagnostic| {
        diagnostic.code == INVALID_RANGE
            && diagnostic.field.as_deref() == Some("pools[0].judge.response_weight")
    }));

    let mut config = report(valid_document()).config.unwrap();
    config.pools[0].judge.pass_threshold = f64::INFINITY;
    assert!(config.validate().iter().any(|diagnostic| {
        diagnostic.code == INVALID_RANGE
            && diagnostic.field.as_deref() == Some("pools[0].judge.pass_threshold")
    }));

    let mut config = report(valid_document()).config.unwrap();
    config.pools[0].judge.temperature = Some(f64::NAN);
    assert!(config.validate().iter().any(|diagnostic| {
        diagnostic.code == INVALID_RANGE
            && diagnostic.field.as_deref() == Some("pools[0].judge.temperature")
    }));

    let mut judge = report(valid_document())
        .config
        .unwrap()
        .pools
        .remove(0)
        .judge;
    judge.max_rationale_bytes = JUDGE_MAX_RATIONALE_BYTES;
    assert_eq!(judge.output_token_limit(), Some(4_608));
    judge.max_rationale_bytes = usize::MAX;
    assert_eq!(judge.output_token_limit(), None);
}

#[test]
fn judge_template_hashes_are_explicit_golden_contracts() {
    assert_eq!(
        JUDGE_PROMPT_TEMPLATE_SHA256_V1,
        "51a96ca6611ab63441084dbf371714fa1fb9305438976ae41b3f2f18bdab4eab"
    );
    assert_eq!(
        JUDGE_RUBRIC_TEMPLATE_SHA256_V1,
        "99d3cf4fd51086b57d9aa40690cc77b2cda1fc6926e2f4e76f76095037ef842f"
    );
    assert_eq!(
        JUDGE_OUTPUT_SCHEMA_SHA256_V1,
        "bcb012992c8805b6ea4d61f77f169434be5b43435934e0a8ee43881df1aed4a2"
    );

    let generated = report(valid_document());
    assert!(!generated.has_errors(), "{:?}", generated.diagnostics);
    assert!(generated.config_generation_id.is_some());
}

#[test]
fn stable_diagnostic_codes_and_indexed_locations_cover_invalid_classes() {
    let cases: Vec<(Json, &str, Option<&str>)> = vec![
        (
            json!({"version": 2}),
            UNSUPPORTED_CONFIG_VERSION,
            Some("version"),
        ),
        (
            json!({"retention_days": 0}),
            INVALID_RANGE,
            Some("retention_days"),
        ),
        (
            json!({"database_path": "../router.db"}),
            INVALID_PATH,
            Some("database_path"),
        ),
        (
            json!({"embedders": [{
                "id": "x", "base_url": "http://example.com", "model": "m",
                "provider_revision": "r1", "dimensions": 1, "timeout_ms": 1
            }]}),
            UNSAFE_EMBEDDER_ENDPOINT,
            Some("embedders[0].base_url"),
        ),
    ];
    for (value, code, field) in cases {
        let result = report(value);
        assert!(
            has_diagnostic(&result, code, field),
            "missing {code} at {field:?}: {:?}",
            result.diagnostics
        );
    }

    let mut duplicate = valid_document();
    duplicate["pools"][0]["candidates"] = json!([
        {
            "id": "same", "model": "candidate-a", "model_revision": "r1",
            "cost_rank": 0
        },
        {
            "id": "same", "model": "candidate-b", "model_revision": "r2",
            "cost_rank": 1
        }
    ]);
    duplicate["pools"][0]["max_candidates_per_sample"] = json!(2);
    let duplicate = report(duplicate);
    assert!(has_diagnostic(
        &duplicate,
        DUPLICATE_ID,
        Some("pools[0].candidates[1].id")
    ));

    let mut invalid_reference = valid_document();
    invalid_reference["pools"][0]["anchor_revision"] = json!("latest");
    let invalid_reference = report(invalid_reference);
    assert!(has_diagnostic(
        &invalid_reference,
        INVALID_REFERENCE,
        Some("pools[0].anchor_revision")
    ));

    let mut extension = valid_document();
    extension["pools"][0]["outcome"] = json!({"version": 1});
    let extension = report(extension);
    assert!(has_diagnostic(
        &extension,
        INVALID_PLUGIN_CONFIG,
        Some("pools[0].outcome.success_matchers")
    ));
}

#[test]
fn every_positive_bound_and_probability_is_checked() {
    let mutations: Vec<(&str, Json, &str)> = vec![
        ("/retention_days", json!(0), "retention_days"),
        ("/max_evidence_records", json!(0), "max_evidence_records"),
        (
            "/embedders/0/dimensions",
            json!(0),
            "embedders[0].dimensions",
        ),
        (
            "/embedders/0/timeout_ms",
            json!(0),
            "embedders[0].timeout_ms",
        ),
        (
            "/embedders/0/max_in_flight",
            json!(0),
            "embedders[0].max_in_flight",
        ),
        (
            "/embedders/0/batch_size",
            json!(0),
            "embedders[0].batch_size",
        ),
        (
            "/pools/0/sampling_probability",
            json!(1.1),
            "pools[0].sampling_probability",
        ),
        (
            "/pools/0/max_candidates_per_sample",
            json!(0),
            "pools[0].max_candidates_per_sample",
        ),
        (
            "/pools/0/concurrency/shadow",
            json!(0),
            "pools[0].concurrency.shadow",
        ),
        (
            "/pools/0/concurrency/judge",
            json!(0),
            "pools[0].concurrency.judge",
        ),
        (
            "/pools/0/concurrency/max_pending",
            json!(0),
            "pools[0].concurrency.max_pending",
        ),
        (
            "/pools/0/candidates/0/max_context_tokens",
            json!(0),
            "pools[0].candidates[0].max_context_tokens",
        ),
        (
            "/pools/0/canonicalizer/max_instruction_bytes",
            json!(0),
            "pools[0].canonicalizer.max_instruction_bytes",
        ),
        (
            "/pools/0/canonicalizer/max_task_bytes",
            json!(0),
            "pools[0].canonicalizer.max_task_bytes",
        ),
        (
            "/pools/0/canonicalizer/max_context_messages",
            json!(0),
            "pools[0].canonicalizer.max_context_messages",
        ),
        (
            "/pools/0/canonicalizer/max_context_bytes",
            json!(0),
            "pools[0].canonicalizer.max_context_bytes",
        ),
        (
            "/pools/0/canonicalizer/max_position_features_bytes",
            json!(0),
            "pools[0].canonicalizer.max_position_features_bytes",
        ),
    ];
    for (pointer, value, field) in mutations {
        let mut document = valid_document();
        if document.pointer(pointer).is_none() {
            let (parent, leaf) = pointer.rsplit_once('/').unwrap();
            document.pointer_mut(parent).unwrap()[leaf] = value;
        } else {
            *document.pointer_mut(pointer).unwrap() = value;
        }
        let result = report(document);
        assert!(
            has_diagnostic(&result, INVALID_RANGE, Some(field)),
            "missing invalid range for {field}: {:?}",
            result.diagnostics
        );
    }
}

#[test]
fn evidence_and_embedder_upper_bounds_accept_exact_maxima_and_reject_max_plus_one() {
    let mut boundary = valid_document();
    boundary["max_evidence_records"] = json!(EVIDENCE_RECORDS_MAX);
    boundary["embedders"][0]["dimensions"] = json!(EMBEDDING_DIMENSIONS_MAX);
    boundary["embedders"][0]["timeout_ms"] = json!(EMBEDDER_TIMEOUT_MS_MAX);
    boundary["embedders"][0]["max_in_flight"] = json!(EMBEDDER_MAX_IN_FLIGHT);
    boundary["embedders"][0]["batch_size"] = json!(EMBEDDER_BATCH_SIZE_MAX);
    let boundary_result = report(boundary.clone());
    assert!(
        !boundary_result.has_errors(),
        "{:?}",
        boundary_result.diagnostics
    );

    let mutations = [
        (
            "/max_evidence_records",
            json!(EVIDENCE_RECORDS_MAX + 1),
            "max_evidence_records",
        ),
        (
            "/embedders/0/dimensions",
            json!(EMBEDDING_DIMENSIONS_MAX + 1),
            "embedders[0].dimensions",
        ),
        (
            "/embedders/0/timeout_ms",
            json!(EMBEDDER_TIMEOUT_MS_MAX + 1),
            "embedders[0].timeout_ms",
        ),
        (
            "/embedders/0/max_in_flight",
            json!(EMBEDDER_MAX_IN_FLIGHT + 1),
            "embedders[0].max_in_flight",
        ),
        (
            "/embedders/0/batch_size",
            json!(EMBEDDER_BATCH_SIZE_MAX + 1),
            "embedders[0].batch_size",
        ),
    ];
    for (pointer, value, field) in mutations {
        let mut oversized = boundary.clone();
        *oversized.pointer_mut(pointer).unwrap() = value;
        let result = report(oversized);
        assert!(
            has_diagnostic(&result, INVALID_RANGE, Some(field)),
            "{field}: {:?}",
            result.diagnostics
        );
    }
}

#[test]
fn semaphore_backed_concurrency_is_bounded_before_activation() {
    let mut boundary = valid_document();
    for field in ["shadow", "judge"] {
        boundary["pools"][0]["concurrency"][field] = json!(CONCURRENCY_MAX_PERMITS);
    }
    boundary["pools"][0]["concurrency"]["max_pending"] = json!(SCHEDULER_MAX_SLOTS);
    let report = validate_router_config(boundary.as_object().unwrap());
    assert!(!report.has_errors(), "{:?}", report.diagnostics);

    for field in ["shadow", "judge"] {
        let mut oversized = valid_document();
        oversized["pools"][0]["concurrency"][field] = json!(CONCURRENCY_MAX_PERMITS + 1);
        let report = validate_router_config(oversized.as_object().unwrap());
        assert!(report.diagnostics.iter().any(|diagnostic| {
            diagnostic.code == INVALID_RANGE
                && diagnostic.field.as_deref()
                    == Some(format!("pools[0].concurrency.{field}").as_str())
        }));
    }

    let mut oversized = valid_document();
    oversized["pools"][0]["concurrency"]["max_pending"] = json!(SCHEDULER_MAX_SLOTS + 1);
    let report = validate_router_config(oversized.as_object().unwrap());
    assert!(has_diagnostic(
        &report,
        INVALID_RANGE,
        Some("pools[0].concurrency.max_pending")
    ));
}

#[test]
fn derived_scheduler_slots_use_checked_per_pool_and_aggregate_bounds() {
    let mut product_boundary = valid_document();
    add_second_candidate(&mut product_boundary["pools"][0]);
    product_boundary["pools"][0]["concurrency"]["max_pending"] = json!(SCHEDULER_MAX_SLOTS / 2);
    let boundary = report(product_boundary.clone());
    assert!(!boundary.has_errors(), "{:?}", boundary.diagnostics);

    product_boundary["pools"][0]["concurrency"]["max_pending"] = json!(SCHEDULER_MAX_SLOTS / 2 + 1);
    let oversized_product = report(product_boundary);
    assert!(has_diagnostic(
        &oversized_product,
        INVALID_RANGE,
        Some("pools[0].max_candidates_per_sample")
    ));

    let mut aggregate_boundary = valid_document();
    aggregate_boundary["pools"][0]["concurrency"]["max_pending"] = json!(SCHEDULER_MAX_SLOTS / 2);
    add_disjoint_pool(&mut aggregate_boundary);
    let boundary = report(aggregate_boundary.clone());
    assert!(!boundary.has_errors(), "{:?}", boundary.diagnostics);

    for pool in aggregate_boundary["pools"].as_array_mut().unwrap() {
        pool["concurrency"]["max_pending"] = json!(SCHEDULER_MAX_SLOTS / 2 + 1);
    }
    let aggregate_over = report(aggregate_boundary);
    assert!(aggregate_over.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == INVALID_RANGE
            && diagnostic.field.as_deref() == Some("pools")
            && diagnostic.message.contains("aggregate batch slots")
    }));

    let mut candidate_aggregate = valid_document();
    add_second_candidate(&mut candidate_aggregate["pools"][0]);
    candidate_aggregate["pools"][0]["concurrency"]["max_pending"] = json!(16_385);
    add_disjoint_pool(&mut candidate_aggregate);
    let candidate_aggregate = report(candidate_aggregate);
    assert!(candidate_aggregate.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == INVALID_RANGE
            && diagnostic.field.as_deref() == Some("pools")
            && diagnostic.message.contains("aggregate candidate slots")
    }));
    assert!(!candidate_aggregate.diagnostics.iter().any(|diagnostic| {
        diagnostic.field.as_deref() == Some("pools")
            && diagnostic.message.contains("aggregate batch slots")
    }));

    let mut product_overflow = valid_document();
    add_second_candidate(&mut product_overflow["pools"][0]);
    product_overflow["pools"][0]["concurrency"]["max_pending"] = json!(usize::MAX);
    let product_overflow = report(product_overflow);
    assert!(product_overflow.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == INVALID_RANGE
            && diagnostic.field.as_deref() == Some("pools[0].max_candidates_per_sample")
            && diagnostic.message.contains("overflow")
    }));

    let mut sum_overflow = valid_document();
    sum_overflow["pools"][0]["concurrency"]["max_pending"] = json!(usize::MAX);
    add_disjoint_pool(&mut sum_overflow);
    let sum_overflow = report(sum_overflow);
    assert!(sum_overflow.diagnostics.iter().any(|diagnostic| {
        diagnostic.code == INVALID_RANGE
            && diagnostic.field.as_deref() == Some("pools")
            && diagnostic.message.contains("overflow usize")
    }));
}

#[test]
fn exact_utf8_byte_boundaries_and_nfc_are_enforced() {
    let mut document = valid_document();
    document["project_id"] = json!("a".repeat(PROJECT_ID_MAX_BYTES));
    document["embedders"][0]["id"] = json!("a".repeat(ID_MAX_BYTES));
    document["embedders"][0]["model"] = json!("a".repeat(MODEL_ID_MAX_BYTES));
    document["embedders"][0]["provider_revision"] = json!("a".repeat(REVISION_MAX_BYTES));
    document["pools"][0]["selector"]["tenant_ids"] =
        json!(["a".repeat(SELECTOR_IDENTITY_MAX_BYTES)]);
    let exact_pattern = std::iter::repeat_n("*", 508)
        .chain(std::iter::once("function"))
        .collect::<Vec<_>>()
        .join("/");
    assert_eq!(exact_pattern.len(), PATH_PATTERN_MAX_BYTES);
    document["pools"][0]["selector"]["scope_path_patterns"] = json!([exact_pattern]);
    let exact = report(document.clone());
    assert!(!exact.has_errors(), "{:?}", exact.diagnostics);

    document["project_id"] = json!("a".repeat(PROJECT_ID_MAX_BYTES + 1));
    document["embedders"][0]["id"] = json!("a".repeat(ID_MAX_BYTES + 1));
    document["embedders"][0]["model"] = json!("a".repeat(MODEL_ID_MAX_BYTES + 1));
    document["embedders"][0]["provider_revision"] = json!("a".repeat(REVISION_MAX_BYTES + 1));
    document["pools"][0]["selector"]["tenant_ids"] =
        json!(["a".repeat(SELECTOR_IDENTITY_MAX_BYTES + 1)]);
    document["pools"][0]["selector"]["scope_path_patterns"] =
        json!(["*".repeat(PATH_PATTERN_MAX_BYTES + 1)]);
    let over = report(document);
    for field in [
        "project_id",
        "embedders[0].id",
        "embedders[0].model",
        "embedders[0].provider_revision",
        "pools[0].selector.tenant_ids[0]",
    ] {
        assert!(has_diagnostic(&over, INVALID_RANGE, Some(field)), "{field}");
    }
    assert!(has_diagnostic(
        &over,
        INVALID_RANGE,
        Some("pools[0].selector.scope_path_patterns[0]")
    ));

    let mut non_nfc = valid_document();
    non_nfc["project_id"] = json!("e\u{301}");
    let non_nfc = report(non_nfc);
    assert!(has_diagnostic(
        &non_nfc,
        INVALID_REFERENCE,
        Some("project_id")
    ));
}

#[test]
fn database_path_validation_is_lexical_and_side_effect_free() {
    for path in [
        "",
        "/",
        "C:",
        "C:\\",
        "../router.db",
        "dir/./router.db",
        "file:router.db",
        "sqlite:router.db",
        "sqlite://router.db",
        "\\\\server\\share\\router.db",
        "dir/",
        "bad\0path.db",
        "NUL",
        "NUL.db",
        "con.",
        "aux ",
        "cache/PRN/router.db",
        "cache/CLOCK$.db",
        "console/CONIN$.log",
        "console/conout$",
        r"C:\temp\LPT1.db",
        "C:/temp/com9.sqlite",
        "archive/COM\u{b9}.log",
        "archive\\LPT\u{b3} ",
        "cache/router.db:secret",
        "cache/router.db.",
        "cache/router.db ",
        "cache/bad<name>.db",
        "cache/bad>name.db",
        "cache/bad\"name.db",
        "cache/bad|name.db",
        "cache/bad?name.db",
        "cache/bad*name.db",
        r"C:\temp\router.db:secret",
        r"C:\temp.\router.db",
        r"C:\temp \router.db",
    ] {
        let result = report(json!({"database_path": path}));
        assert!(
            has_diagnostic(&result, INVALID_PATH, Some("database_path")),
            "{path:?}: {:?}",
            result.diagnostics
        );
    }

    for prefix in ["COM", "LPT"] {
        for number in 1..=9 {
            let path = format!("cache/{prefix}{number}.db");
            let result = report(json!({"database_path": path}));
            assert!(
                has_diagnostic(&result, INVALID_PATH, Some("database_path")),
                "{path:?}: {:?}",
                result.diagnostics
            );
        }
    }

    for path in [
        "null.db",
        "console.db",
        "auxiliary/router.db",
        "com0.db",
        "com10.db",
        "com1_port.db",
        "lpt0.db",
        "lpt10.db",
        "clock.db",
        "clock$work/router.db",
        r"C:\temp\LPT10.db",
    ] {
        let result = report(json!({"database_path": path}));
        assert!(
            !result.has_errors(),
            "safe near-miss {path:?}: {:?}",
            result.diagnostics
        );
    }

    let parent = std::env::temp_dir().join(format!("nemo-router-{}", Uuid::new_v4()));
    let database = parent.join("nested/router.db");
    assert!(!parent.exists());
    let mut document = valid_document();
    document["database_path"] = json!(database.to_string_lossy());
    document["embedders"][0]["api_key_env"] = json!("CERTAINLY_NOT_A_REAL_SECRET_VARIABLE");
    enable_learning(&mut document, "embedding-main");
    let result = report(document);
    assert!(!result.has_errors(), "{:?}", result.diagnostics);
    assert!(!parent.exists(), "validation created filesystem state");
}

#[test]
fn embedder_environment_name_is_syntax_checked_but_never_resolved() {
    let mut exact = valid_document();
    exact["embedders"][0]["api_key_env"] = json!(format!("A{}", "B".repeat(ID_MAX_BYTES - 1)));
    enable_learning(&mut exact, "embedding-main");
    let exact = report(exact);
    assert!(!exact.has_errors(), "{:?}", exact.diagnostics);

    for (value, code) in [
        ("1INVALID".to_string(), INVALID_REFERENCE),
        ("INVALID-NAME".to_string(), INVALID_REFERENCE),
        (format!("A{}", "B".repeat(ID_MAX_BYTES)), INVALID_RANGE),
    ] {
        let mut document = valid_document();
        document["embedders"][0]["api_key_env"] = json!(value);
        let result = report(document);
        assert!(has_diagnostic(
            &result,
            code,
            Some("embedders[0].api_key_env")
        ));
    }
}

#[test]
fn embedder_endpoint_egress_classes_are_exact() {
    for endpoint in [
        "http://localhost:8080/v1",
        "http://127.0.0.2:8080/v1",
        "http://[::1]:8080/v1",
        "https://localhost/v1",
    ] {
        let mut document = valid_document();
        document["embedders"][0]["base_url"] = json!(endpoint);
        let result = report(document);
        assert!(!result.has_errors(), "{endpoint}: {:?}", result.diagnostics);
    }

    for endpoint in ["https://api.example.com/v1", "https://10.0.0.1/v1"] {
        let mut remote_https = valid_document();
        remote_https["allow_remote_embedding_egress"] = json!(true);
        remote_https["embedders"][0]["base_url"] = json!(endpoint);
        let result = report(remote_https);
        assert!(!result.has_errors(), "{endpoint}: {:?}", result.diagnostics);
    }

    let mut remote_http_with_opt_in = valid_document();
    remote_http_with_opt_in["allow_remote_embedding_egress"] = json!(true);
    remote_http_with_opt_in["embedders"][0]["base_url"] = json!("http://api.example.com/v1");
    let remote_http_with_opt_in = report(remote_http_with_opt_in);
    assert!(has_diagnostic(
        &remote_http_with_opt_in,
        UNSAFE_EMBEDDER_ENDPOINT,
        Some("embedders[0].base_url")
    ));

    for endpoint in [
        "http://api.example.com/v1",
        "https://api.example.com/v1",
        "http://localhost.example.com/v1",
        "http://127.0.0.1.example.com/v1",
        "ftp://localhost/model",
        "https://user:secret@localhost/v1",
        "http://@localhost/v1",
        "http:\\@localhost\\v1",
        "http:/\\@localhost/v1",
        "http:////@localhost/v1",
        "https://localhost/v1?token=secret",
        "https://localhost/v1#fragment",
        "http://localhost:8080/v1/embeddings",
        "http://localhost:8080/v1/embeddings/",
        "http://localhost:8080/v1/%65mbeddings",
        "http://localhost:8080/v1%2Fnested",
        "http://localhost:8080/v1%5Cnested",
        "not-a-url",
    ] {
        let mut document = valid_document();
        document["embedders"][0]["base_url"] = json!(endpoint);
        let result = report(document);
        assert!(
            has_diagnostic(
                &result,
                UNSAFE_EMBEDDER_ENDPOINT,
                Some("embedders[0].base_url")
            ),
            "{endpoint}: {:?}",
            result.diagnostics
        );
    }
}

#[test]
fn overlapping_pool_validation_is_pairwise_and_conservative() {
    let mut overlap = valid_document();
    let second = overlap["pools"][0].clone();
    overlap["pools"].as_array_mut().unwrap().push(second);
    overlap["pools"][1]["id"] = json!("chat-second");
    let result = report(overlap);
    assert!(has_diagnostic(
        &result,
        OVERLAPPING_POOL,
        Some("pools[1].selector")
    ));

    for dimension in ["family", "anchor", "tenant", "metadata", "path"] {
        let mut disjoint = valid_document();
        let second = disjoint["pools"][0].clone();
        disjoint["pools"].as_array_mut().unwrap().push(second);
        disjoint["pools"][1]["id"] = json!("chat-second");
        disjoint["pools"][1]["candidates"][0]["id"] = json!("other");
        match dimension {
            "family" => {
                disjoint["pools"][1]["api_family"] = json!("openai_responses");
            }
            "anchor" => disjoint["pools"][1]["anchor_models"] = json!(["other-anchor"]),
            "tenant" => disjoint["pools"][1]["selector"]["tenant_ids"] = json!(["tenant-b"]),
            "metadata" => {
                disjoint["pools"][1]["selector"]["metadata_equals"]["region"] = json!("eu")
            }
            "path" => {
                disjoint["pools"][0]["selector"]["scope_path_patterns"] = json!(["agent/tool"]);
                disjoint["pools"][1]["selector"]["scope_path_patterns"] = json!(["agent/llm"]);
            }
            _ => unreachable!(),
        }
        let result = report(disjoint);
        assert!(
            !has_diagnostic(&result, OVERLAPPING_POOL, Some("pools[1].selector")),
            "{dimension}: {:?}",
            result.diagnostics
        );
    }
}

#[test]
fn generation_is_order_independent_and_behavior_sensitive() {
    let mut document = valid_document();
    document["pools"][0]["concurrency"]["max_pending"] = json!(32);
    document["pools"][0]["anchor_models"] = json!(["anchor-large", "anchor-medium"]);
    document["pools"][0]["selector"]["tenant_ids"] = json!(["tenant-b", "tenant-a"]);
    document["pools"][0]["candidates"] = json!([
        {
            "id": "small", "model": "candidate-small", "model_revision": "r1",
            "cost_rank": 1, "max_context_tokens": 32768
        },
        {
            "id": "tiny", "model": "candidate-tiny", "model_revision": "r2",
            "cost_rank": 0, "max_context_tokens": 16384
        }
    ]);
    document["pools"][0]["max_candidates_per_sample"] = json!(2);
    let mut second = document["pools"][0].clone();
    second["id"] = json!("responses-default");
    second["api_family"] = json!("openai_responses");
    second["anchor_models"] = json!(["responses-anchor"]);
    second["candidates"][0]["id"] = json!("responses-small");
    second["candidates"][0]["model"] = json!("responses-candidate-small");
    second["candidates"][1]["id"] = json!("responses-tiny");
    second["candidates"][1]["model"] = json!("responses-candidate-tiny");
    document["pools"].as_array_mut().unwrap().push(second);

    let baseline = report(document.clone());
    assert!(!baseline.has_errors(), "{:?}", baseline.diagnostics);
    let baseline_id = baseline.config_generation_id.unwrap();
    assert_eq!(baseline_id.len(), 64);
    assert!(
        baseline_id
            .chars()
            .all(|character| character.is_ascii_hexdigit() && !character.is_ascii_uppercase())
    );

    let mut reordered = document.clone();
    reordered["embedders"].as_array_mut().unwrap().reverse();
    reordered["pools"].as_array_mut().unwrap().reverse();
    for pool in reordered["pools"].as_array_mut().unwrap() {
        pool["anchor_models"].as_array_mut().unwrap().reverse();
        pool["selector"]["tenant_ids"]
            .as_array_mut()
            .unwrap()
            .reverse();
        pool["candidates"].as_array_mut().unwrap().reverse();
    }
    reordered["ignored_unknown"] = json!({"does_not": "hash"});
    let reordered = report(reordered);
    assert!(!reordered.has_errors(), "{:?}", reordered.diagnostics);
    assert_eq!(
        reordered.config_generation_id.as_deref(),
        Some(baseline_id.as_str())
    );

    let changes: Vec<(&str, Json)> = vec![
        ("/pools/0/anchor_models/0", json!("changed-anchor")),
        ("/pools/0/anchor_revision", json!("changed-revision")),
        ("/pools/0/candidates/0/model", json!("changed-candidate")),
        (
            "/pools/0/candidates/0/model_revision",
            json!("changed-revision"),
        ),
        ("/pools/0/candidates/0/cost_rank", json!(7)),
        ("/pools/0/sampling_probability", json!(0.5)),
        ("/pools/0/concurrency/shadow", json!(7)),
        ("/pools/0/concurrency/max_pending", json!(16)),
        ("/pools/0/max_candidates_per_sample", json!(1)),
        ("/pools/0/judge/model", json!("changed-judge")),
        (
            "/pools/0/judge/model_revision",
            json!("changed-judge-revision"),
        ),
        ("/pools/0/judge/response_weight", json!(0.6)),
        ("/pools/0/judge/trajectory_weight", json!(0.6)),
        ("/pools/0/judge/response_floor", json!(0.75)),
        ("/pools/0/judge/trajectory_floor", json!(0.75)),
        ("/pools/0/judge/judge_confidence_floor", json!(0.65)),
        ("/pools/0/judge/pass_threshold", json!(0.8)),
        ("/pools/0/judge/max_rationale_bytes", json!(8_192)),
        ("/pools/0/judge/base_cooloff_seconds", json!(20)),
        ("/pools/0/judge/max_cooloff_seconds", json!(600)),
    ];
    for (pointer, value) in changes {
        let mut changed = document.clone();
        *changed.pointer_mut(pointer).unwrap() = value;
        if pointer.ends_with("cost_rank") {
            changed["pools"][0]["candidates"][1]["cost_rank"] = json!(8);
        }
        if pointer.ends_with("response_weight") {
            changed["pools"][0]["judge"]["trajectory_weight"] = json!(0.4);
        }
        if pointer.ends_with("trajectory_weight") {
            changed["pools"][0]["judge"]["response_weight"] = json!(0.4);
        }
        let changed = report(changed);
        assert!(
            !changed.has_errors(),
            "{pointer}: {:?}",
            changed.diagnostics
        );
        assert_ne!(
            changed.config_generation_id.as_deref(),
            Some(baseline_id.as_str()),
            "{pointer}"
        );
    }

    let mut explicit_default_temperature = document.clone();
    explicit_default_temperature["pools"][0]["judge"]["temperature"] = json!(0.0);
    let explicit_default_temperature = report(explicit_default_temperature);
    assert!(!explicit_default_temperature.has_errors());
    assert_eq!(
        explicit_default_temperature.config_generation_id.as_deref(),
        Some(baseline_id.as_str())
    );

    let mut changed_temperature = document;
    changed_temperature["pools"][0]["judge"]["temperature"] = json!(1.0);
    let changed_temperature = report(changed_temperature);
    assert!(!changed_temperature.has_errors());
    assert_ne!(
        changed_temperature.config_generation_id.as_deref(),
        Some(baseline_id.as_str())
    );
}

#[test]
fn generation_normalizes_endpoints_and_hashes_learning_associations() {
    let generation_for_endpoint = |endpoint: &str| {
        let mut document = valid_document();
        document["embedders"][0]["base_url"] = json!(endpoint);
        enable_learning(&mut document, "embedding-main");
        let result = report(document);
        assert!(!result.has_errors(), "{endpoint}: {:?}", result.diagnostics);
        result.config_generation_id.unwrap()
    };

    let equivalent = [
        "http://localhost/v1",
        "http://LOCALHOST:80/v1/",
        "http://localhost:80/v1///",
    ]
    .map(generation_for_endpoint);
    assert!(equivalent.windows(2).all(|pair| pair[0] == pair[1]));
    assert_ne!(
        equivalent[0],
        generation_for_endpoint("http://localhost/v2")
    );

    let mut main_association = valid_document();
    let mut alternate = main_association["embedders"][0].clone();
    alternate["id"] = json!("embedding-alternate");
    main_association["embedders"]
        .as_array_mut()
        .unwrap()
        .push(alternate);
    enable_learning(&mut main_association, "embedding-main");
    let main = report(main_association.clone());
    assert!(!main.has_errors(), "{:?}", main.diagnostics);

    enable_learning(&mut main_association, "embedding-alternate");
    let alternate = report(main_association);
    assert!(!alternate.has_errors(), "{:?}", alternate.diagnostics);
    assert_ne!(main.config_generation_id, alternate.config_generation_id);
}

#[test]
fn generation_normalizes_equivalent_floats_but_preserves_one_ulp_changes() {
    let generation_for = |field: &str, value: Json| {
        let mut document = valid_document();
        document["pools"][0]["judge"][field] = value;
        let result = report(document);
        assert!(!result.has_errors(), "{field}: {:?}", result.diagnostics);
        result.config_generation_id.unwrap()
    };

    let equivalent = ["0", "0.0", "-0.0", "0e10"]
        .into_iter()
        .map(|literal| {
            generation_for(
                "response_floor",
                serde_json::from_str::<Json>(literal).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    assert!(equivalent.windows(2).all(|pair| pair[0] == pair[1]));

    let threshold = 0.85_f64;
    let threshold_id = generation_for("pass_threshold", json!(threshold));
    let next_threshold = f64::from_bits(threshold.to_bits() + 1);
    assert_ne!(
        threshold_id,
        generation_for("pass_threshold", json!(next_threshold))
    );

    let weight = 0.5_f64;
    let weight_id = generation_for("response_weight", json!(weight));
    let next_weight = f64::from_bits(weight.to_bits() + 1);
    assert_ne!(
        weight_id,
        generation_for("response_weight", json!(next_weight))
    );
}

#[test]
fn editor_metadata_uses_json_controls_for_typed_arrays() {
    let schema = RouterConfig::editor_schema();
    assert_eq!(
        schema.field("embedders").unwrap().kind,
        EditorFieldKind::Json
    );
    assert_eq!(schema.field("pools").unwrap().kind, EditorFieldKind::Json);
    assert_eq!(
        schema.field("mode").unwrap().enum_values,
        ["off", "shadow", "recommend", "active"]
    );
    assert!(schema.field("unknown_fields").is_none());

    let pool_schema = PoolConfig::editor_schema();
    let learning_field = pool_schema.field("learning").unwrap();
    assert_eq!(learning_field.kind, EditorFieldKind::Json);
    assert!(learning_field.optional);
    let judge_field = pool_schema.field("judge").unwrap();
    assert_eq!(judge_field.kind, EditorFieldKind::Section);
    assert!(!judge_field.optional);
    assert!(judge_field.default_value().is_none());
    assert!(std::ptr::eq(
        judge_field.schema().unwrap(),
        JudgeConfig::editor_schema()
    ));

    let judge_schema = JudgeConfig::editor_schema();
    let expected_fields = [
        "version",
        "model",
        "model_revision",
        "prompt_version",
        "rubric_version",
        "output_schema_version",
        "temperature",
        "response_weight",
        "trajectory_weight",
        "response_floor",
        "trajectory_floor",
        "judge_confidence_floor",
        "pass_threshold",
        "max_rationale_bytes",
        "base_cooloff_seconds",
        "max_cooloff_seconds",
    ];
    assert_eq!(
        judge_schema
            .fields
            .iter()
            .map(|field| field.name)
            .collect::<Vec<_>>(),
        expected_fields
    );
    let expected_kinds = [
        EditorFieldKind::Integer,
        EditorFieldKind::String,
        EditorFieldKind::String,
        EditorFieldKind::Enum,
        EditorFieldKind::Enum,
        EditorFieldKind::Integer,
        EditorFieldKind::Float,
        EditorFieldKind::Float,
        EditorFieldKind::Float,
        EditorFieldKind::Float,
        EditorFieldKind::Float,
        EditorFieldKind::Float,
        EditorFieldKind::Float,
        EditorFieldKind::Integer,
        EditorFieldKind::Integer,
        EditorFieldKind::Integer,
    ];
    assert_eq!(
        judge_schema
            .fields
            .iter()
            .map(|field| field.kind)
            .collect::<Vec<_>>(),
        expected_kinds
    );
    assert!(judge_schema.fields.iter().all(|field| {
        (field.name == "temperature") == field.optional
            && field.default_value().is_none()
            && field.name != "unknown_fields"
    }));
    assert_eq!(
        judge_schema.field("prompt_version").unwrap().enum_values,
        [JUDGE_PROMPT_VERSION_V1]
    );
    assert_eq!(
        judge_schema.field("rubric_version").unwrap().enum_values,
        [JUDGE_RUBRIC_VERSION_V1]
    );
}

#[cfg(feature = "schema")]
#[test]
fn optional_schema_feature_describes_router_config() {
    let schema = schemars::schema_for!(RouterConfig);
    let value = serde_json::to_value(schema).unwrap();
    assert_eq!(value["title"], "RouterConfig");

    let judge = &value["definitions"]["JudgeConfig"];
    let mut properties = judge["properties"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    properties.sort_unstable();
    let mut expected = vec![
        "version",
        "model",
        "model_revision",
        "prompt_version",
        "rubric_version",
        "output_schema_version",
        "temperature",
        "response_weight",
        "trajectory_weight",
        "response_floor",
        "trajectory_floor",
        "judge_confidence_floor",
        "pass_threshold",
        "max_rationale_bytes",
        "base_cooloff_seconds",
        "max_cooloff_seconds",
    ];
    expected.sort_unstable();
    assert_eq!(properties, expected);

    let mut required = judge["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|field| field.as_str().unwrap())
        .collect::<Vec<_>>();
    required.sort_unstable();
    let mut expected_required = expected
        .iter()
        .copied()
        .filter(|field| *field != "temperature")
        .collect::<Vec<_>>();
    expected_required.sort_unstable();
    assert_eq!(required, expected_required);
    assert_eq!(judge["additionalProperties"], json!(false));

    let pool_required = value["definitions"]["PoolConfig"]["required"]
        .as_array()
        .unwrap();
    assert!(pool_required.iter().any(|field| field == "judge"));
    assert!(!pool_required.iter().any(|field| field == "learning"));
    assert_eq!(
        value["definitions"]["PoolConfig"]["properties"]["judge"]["$ref"],
        json!("#/definitions/JudgeConfig"),
        "{}",
        serde_json::to_string_pretty(&value["definitions"]["PoolConfig"]["properties"]["judge"])
            .unwrap()
    );

    let learning = &value["definitions"]["PoolConfig"]["properties"]["learning"];
    let variants = learning["oneOf"].as_array().unwrap();
    assert_eq!(variants.len(), 4);
    let empty = variants
        .iter()
        .find(|variant| variant["maxProperties"] == json!(0))
        .unwrap();
    assert_eq!(empty["type"], "object");
    assert_eq!(empty["additionalProperties"], json!(false));

    let minimal = variants
        .iter()
        .find(|variant| {
            variant["properties"]
                .as_object()
                .is_some_and(|fields| fields.len() == 2)
        })
        .unwrap();
    let mut properties = minimal["properties"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    properties.sort_unstable();
    assert_eq!(properties, ["embedder", "version"]);
    let mut required = minimal["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|field| field.as_str().unwrap())
        .collect::<Vec<_>>();
    required.sort_unstable();
    assert_eq!(required, ["embedder", "version"]);
    assert_eq!(minimal["additionalProperties"], json!(false));

    let complete_recommend = variants
        .iter()
        .find(|variant| {
            variant["properties"]
                .as_object()
                .is_some_and(|fields| fields.len() == 13)
        })
        .unwrap();
    let mut properties = complete_recommend["properties"]
        .as_object()
        .unwrap()
        .keys()
        .map(String::as_str)
        .collect::<Vec<_>>();
    properties.sort_unstable();
    let mut expected = vec![
        "version",
        "embedder",
        "top_k",
        "radius",
        "min_points",
        "min_independent_roots",
        "min_effective_samples",
        "min_coverage",
        "time_decay_half_life_seconds",
        "prior_success",
        "prior_failure",
        "familywise_credible_level",
        "promotion_lower_bound",
    ];
    expected.sort_unstable();
    assert_eq!(properties, expected);
    let mut required = complete_recommend["required"]
        .as_array()
        .unwrap()
        .iter()
        .map(|field| field.as_str().unwrap())
        .collect::<Vec<_>>();
    required.sort_unstable();
    assert_eq!(required, expected);
    assert_eq!(complete_recommend["additionalProperties"], json!(false));
    for field in expected
        .iter()
        .copied()
        .filter(|field| !matches!(*field, "version" | "embedder"))
    {
        let field_type = &complete_recommend["properties"][field]["type"];
        assert!(
            field_type != "null"
                && field_type
                    .as_array()
                    .is_none_or(|types| types.iter().all(|value| value != "null")),
            "{field}: {}",
            serde_json::to_string_pretty(&complete_recommend["properties"][field]).unwrap()
        );
    }

    let complete_active = variants
        .iter()
        .find(|variant| {
            variant["properties"]
                .as_object()
                .is_some_and(|fields| fields.len() == 16)
        })
        .unwrap();
    let active_properties = complete_active["properties"].as_object().unwrap();
    assert!(active_properties.contains_key("retention_lower_bound"));
    assert!(active_properties.contains_key("holdout_probability"));
    assert!(active_properties.contains_key("active_canary_fraction"));
    assert_eq!(complete_active["required"].as_array().unwrap().len(), 16);
    assert_eq!(complete_active["additionalProperties"], json!(false));

    let outcome = &value["definitions"]["PoolConfig"]["properties"]["outcome"];
    let outcome_variants = outcome["oneOf"].as_array().unwrap();
    assert_eq!(outcome_variants.len(), 2);
    let empty_outcome = outcome_variants
        .iter()
        .find(|variant| variant["maxProperties"] == json!(0))
        .unwrap();
    assert_eq!(empty_outcome["additionalProperties"], json!(false));
    let complete_outcome = outcome_variants
        .iter()
        .find(|variant| {
            variant["properties"]
                .as_object()
                .is_some_and(|fields| fields.len() == 21)
        })
        .unwrap();
    assert_eq!(complete_outcome["required"].as_array().unwrap().len(), 21);
    assert_eq!(complete_outcome["additionalProperties"], json!(false));
    assert_eq!(
        complete_outcome["properties"]["success_matchers"]["minItems"],
        json!(1)
    );
    assert_eq!(
        complete_outcome["properties"]["success_matchers"]["maxItems"],
        json!(OUTCOME_MATCHERS_MAX)
    );

    let matcher = &value["definitions"]["OutcomeMatcher"];
    assert_eq!(matcher["required"].as_array().unwrap().len(), 5);
    assert_eq!(matcher["additionalProperties"], json!(false));
    assert_eq!(
        matcher["properties"]["metadata_equals"]["maxProperties"],
        json!(OUTCOME_METADATA_EQUALS_MAX)
    );
}

#[test]
fn raw_object_order_does_not_change_diagnostic_order() {
    let first = object(json!({"z": 1, "a": 2}));
    let second = BTreeMap::from([("a".to_string(), json!(2)), ("z".to_string(), json!(1))])
        .into_iter()
        .collect::<Map<_, _>>();
    let first = validate_router_config(&first);
    let second = validate_router_config(&second);
    let signature = |report: &nemo_relay_router::RouterConfigValidation| {
        report
            .diagnostics
            .iter()
            .map(|diagnostic| (diagnostic.code.clone(), diagnostic.field.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(signature(&first), signature(&second));
}
