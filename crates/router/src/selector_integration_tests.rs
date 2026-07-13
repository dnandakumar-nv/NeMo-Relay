// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deterministic selector compiler and pool matcher tests.

use std::collections::BTreeMap;

use nemo_relay_types::api::llm::{
    LlmApiFamily, LlmAttributes, LlmCallRole, LlmExecutionContextSnapshot,
    LlmTrajectoryScopeSnapshot,
};
use nemo_relay_types::api::scope::ScopeType;
use serde_json::{Map, Value as Json, json};
use uuid::Uuid;

use crate::matcher::PoolMatcher;
use crate::selector::{CompiledSelector, ScopePathPattern, selector_domains_overlap};
use crate::{PoolSelectorConfig, RouterConfig, validate_router_config};

fn context(
    tenant_id: Option<&str>,
    agent_id: Option<&str>,
    path: &[ScopeType],
    metadata: BTreeMap<String, Json>,
) -> LlmExecutionContextSnapshot {
    LlmExecutionContextSnapshot {
        call_uuid: Uuid::from_u128(1),
        root_uuid: Uuid::from_u128(2),
        parent_uuid: Uuid::from_u128(3),
        trajectory_owner_uuid: Uuid::from_u128(4),
        trajectory_owner_path: path
            .iter()
            .enumerate()
            .map(|(index, scope_type)| LlmTrajectoryScopeSnapshot {
                uuid: Uuid::from_u128(10 + index as u128),
                name: format!("display-name-{index}"),
                scope_type: *scope_type,
            })
            .collect(),
        api_family: LlmApiFamily::OpenAIChatCompletions,
        call_role: LlmCallRole::Primary,
        attributes: LlmAttributes::empty(),
        tenant_id: tenant_id.map(str::to_string),
        agent_id: agent_id.map(str::to_string),
        sanitized_metadata: metadata,
    }
}

fn metadata(region: Option<&str>) -> BTreeMap<String, Json> {
    region
        .map(|region| BTreeMap::from([("region".to_string(), json!(region))]))
        .unwrap_or_default()
}

fn selector_from(value: Json) -> PoolSelectorConfig {
    serde_json::from_value(value).unwrap()
}

fn judge_config() -> Json {
    json!({
        "version": 1,
        "model": "judge-model",
        "model_revision": "judge-r1",
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
    })
}

fn pool(id: &str, tenant: &str) -> Json {
    json!({
        "id": id,
        "api_family": "openai_chat_completions",
        "anchor_models": ["anchor"],
        "anchor_revision": "r1",
        "sampling_probability": 1.0,
        "max_candidates_per_sample": 1,
        "selector": {"tenant_ids": [tenant]},
        "concurrency": {"shadow": 1, "judge": 1},
        "judge": judge_config(),
        "candidates": [{
            "id": "candidate",
            "model": format!("{id}-model"),
            "model_revision": "r1",
            "cost_rank": 0
        }]
    })
}

fn router_config(pools: Vec<Json>) -> RouterConfig {
    let value = json!({"mode": "shadow", "pools": pools});
    let report = validate_router_config(value.as_object().unwrap());
    assert!(!report.has_errors(), "{:?}", report.diagnostics);
    report.config.unwrap()
}

#[test]
fn restricted_path_grammar_has_golden_matches() {
    let valid = [
        ("agent", vec![ScopeType::Agent], true),
        ("*", vec![ScopeType::Tool], true),
        ("**", vec![ScopeType::Agent, ScopeType::Tool], true),
        ("agent/**", vec![ScopeType::Agent], true),
        (
            "agent/**",
            vec![ScopeType::Agent, ScopeType::Function, ScopeType::Llm],
            true,
        ),
        (
            "agent/*/llm",
            vec![ScopeType::Agent, ScopeType::Function, ScopeType::Llm],
            true,
        ),
        ("agent/*/llm", vec![ScopeType::Agent, ScopeType::Llm], false),
        ("agent/tool", vec![ScopeType::Agent, ScopeType::Llm], false),
    ];
    for (pattern, path, expected) in valid {
        let compiled = ScopePathPattern::parse(pattern).unwrap();
        assert_eq!(compiled.as_str(), pattern);
        assert_eq!(compiled.matches(&path), expected, "{pattern}");
    }

    for invalid in [
        "",
        "/agent",
        "agent/",
        "agent//tool",
        "agent/**/tool",
        "agent/t*",
        "Agent",
        "agent/not-a-scope",
    ] {
        assert!(ScopePathPattern::parse(invalid).is_err(), "{invalid}");
    }
}

#[test]
fn selector_matches_only_frozen_stable_facts() {
    let selector = selector_from(json!({
        "tenant_ids": ["tenant-a", "tenant-b"],
        "agent_ids": ["agent-a"],
        "owner_scope_types": ["agent"],
        "metadata_equals": {"enabled": true, "rank": 1.0, "region": "us"},
        "scope_path_patterns": ["agent/*/llm"]
    }));
    let compiled = CompiledSelector::compile(&selector).unwrap();
    let mut facts = BTreeMap::from([
        ("enabled".to_string(), json!(true)),
        ("rank".to_string(), json!(1)),
        ("region".to_string(), json!("us")),
    ]);
    let mut matching = context(
        Some("tenant-a"),
        Some("agent-a"),
        &[ScopeType::Agent, ScopeType::Function, ScopeType::Llm],
        facts.clone(),
    );
    assert!(compiled.matches(&matching));

    for snapshot in &mut matching.trajectory_owner_path {
        snapshot.name = "a-completely-different-display-name".to_string();
        snapshot.uuid = Uuid::new_v4();
    }
    assert!(
        compiled.matches(&matching),
        "names and UUIDs must not be read"
    );

    assert!(!compiled.matches(&context(
        None,
        Some("agent-a"),
        &[ScopeType::Agent, ScopeType::Function, ScopeType::Llm],
        facts.clone(),
    )));
    assert!(!compiled.matches(&context(
        Some("tenant-a"),
        Some("agent-b"),
        &[ScopeType::Agent, ScopeType::Function, ScopeType::Llm],
        facts.clone(),
    )));
    assert!(!compiled.matches(&context(
        Some("tenant-a"),
        Some("agent-a"),
        &[ScopeType::Tool, ScopeType::Function, ScopeType::Llm],
        facts.clone(),
    )));
    facts.insert("region".to_string(), json!("eu"));
    assert!(!compiled.matches(&context(
        Some("tenant-a"),
        Some("agent-a"),
        &[ScopeType::Agent, ScopeType::Function, ScopeType::Llm],
        facts,
    )));
}

#[test]
fn missing_identity_lists_are_wildcards_and_present_lists_require_present_exact_ids() {
    let wildcard = CompiledSelector::compile(&PoolSelectorConfig::default()).unwrap();
    assert!(wildcard.matches(&context(None, None, &[ScopeType::Agent], BTreeMap::new())));
    assert!(wildcard.matches(&context(
        Some("any-tenant"),
        Some("any-agent"),
        &[ScopeType::Agent],
        BTreeMap::new()
    )));

    let explicit = CompiledSelector::compile(&selector_from(json!({
        "tenant_ids": ["tenant-a"],
        "agent_ids": ["agent-a"]
    })))
    .unwrap();
    assert!(!explicit.matches(&context(
        None,
        Some("agent-a"),
        &[ScopeType::Agent],
        BTreeMap::new()
    )));
    assert!(!explicit.matches(&context(
        Some("tenant-a"),
        None,
        &[ScopeType::Agent],
        BTreeMap::new()
    )));
    assert!(explicit.matches(&context(
        Some("tenant-a"),
        Some("agent-a"),
        &[ScopeType::Agent],
        BTreeMap::new()
    )));
}

#[test]
fn overlap_checker_intersects_every_selector_dimension() {
    let wildcard = PoolSelectorConfig::default();
    let tenant_a = selector_from(json!({"tenant_ids": ["a"]}));
    let tenant_b = selector_from(json!({"tenant_ids": ["b"]}));
    assert!(selector_domains_overlap(&wildcard, &tenant_a));
    assert!(!selector_domains_overlap(&tenant_a, &tenant_b));

    let agent_a = selector_from(json!({"agent_ids": ["a"]}));
    let agent_b = selector_from(json!({"agent_ids": ["b"]}));
    assert!(!selector_domains_overlap(&agent_a, &agent_b));

    let owner_agent = selector_from(json!({"owner_scope_types": ["agent"]}));
    let owner_tool = selector_from(json!({"owner_scope_types": ["tool"]}));
    assert!(!selector_domains_overlap(&owner_agent, &owner_tool));

    let metadata_one = selector_from(json!({"metadata_equals": {"rank": 1}}));
    let metadata_one_float = selector_from(json!({"metadata_equals": {"rank": 1.0}}));
    let metadata_two = selector_from(json!({"metadata_equals": {"rank": 2}}));
    assert!(selector_domains_overlap(&metadata_one, &metadata_one_float));
    assert!(!selector_domains_overlap(&metadata_one, &metadata_two));

    let path_suffix = selector_from(json!({"scope_path_patterns": ["agent/**"]}));
    let path_tool = selector_from(json!({"scope_path_patterns": ["agent/tool"]}));
    let path_other = selector_from(json!({"scope_path_patterns": ["tool/**"]}));
    assert!(selector_domains_overlap(&path_suffix, &path_tool));
    assert!(!selector_domains_overlap(&path_suffix, &path_other));
}

#[test]
fn finite_selector_universe_proves_overlap_equivalent_to_a_common_witness() {
    let tenant_dimensions = [None, Some("tenant-a"), Some("tenant-b")];
    let agent_dimensions = [None, Some("agent-a"), Some("agent-b")];
    let owner_dimensions = [None, Some(ScopeType::Agent), Some(ScopeType::Tool)];
    let metadata_dimensions = [None, Some("us"), Some("eu")];
    let path_dimensions = [None, Some("agent/**"), Some("tool/**"), Some("agent/tool")];

    let mut selectors = Vec::new();
    for tenant in tenant_dimensions {
        for agent in agent_dimensions {
            for owner in owner_dimensions {
                for region in metadata_dimensions {
                    for path in path_dimensions {
                        selectors.push(PoolSelectorConfig {
                            tenant_ids: tenant.map(|value| vec![value.to_string()]),
                            agent_ids: agent.map(|value| vec![value.to_string()]),
                            owner_scope_types: owner.map(|value| vec![value]),
                            metadata_equals: region
                                .map(|value| BTreeMap::from([("region".to_string(), json!(value))]))
                                .unwrap_or_default(),
                            scope_path_patterns: path.map(|value| vec![value.to_string()]),
                            unknown_fields: BTreeMap::new(),
                        });
                    }
                }
            }
        }
    }

    let paths = [
        vec![ScopeType::Agent],
        vec![ScopeType::Tool],
        vec![ScopeType::Agent, ScopeType::Tool],
        vec![ScopeType::Agent, ScopeType::Llm],
        vec![ScopeType::Tool, ScopeType::Llm],
    ];
    let mut contexts = Vec::new();
    for tenant in [None, Some("tenant-a"), Some("tenant-b")] {
        for agent in [None, Some("agent-a"), Some("agent-b")] {
            for path in &paths {
                for region in [None, Some("us"), Some("eu")] {
                    contexts.push(context(tenant, agent, path, metadata(region)));
                }
            }
        }
    }

    let compiled = selectors
        .iter()
        .map(|selector| CompiledSelector::compile(selector).unwrap())
        .collect::<Vec<_>>();
    for left in 0..selectors.len() {
        for right in left..selectors.len() {
            let predicted = selector_domains_overlap(&selectors[left], &selectors[right]);
            let witnessed = contexts
                .iter()
                .any(|context| compiled[left].matches(context) && compiled[right].matches(context));
            assert_eq!(
                predicted, witnessed,
                "selector pair {left}/{right}: {:?} versus {:?}",
                selectors[left], selectors[right]
            );
        }
    }
}

#[test]
fn matcher_is_deterministic_across_pool_order_and_ignores_display_identity() {
    let config = router_config(vec![pool("pool-b", "tenant-b"), pool("pool-a", "tenant-a")]);
    let reversed = router_config(vec![pool("pool-a", "tenant-a"), pool("pool-b", "tenant-b")]);
    let matcher = PoolMatcher::compile(&config).unwrap();
    let reversed_matcher = PoolMatcher::compile(&reversed).unwrap();
    let mut facts = context(
        Some("tenant-a"),
        None,
        &[ScopeType::Agent, ScopeType::Function],
        BTreeMap::new(),
    );
    assert_eq!(
        matcher.match_pool("anchor", &facts).matched_pool_id,
        Some("pool-a")
    );
    assert_eq!(
        reversed_matcher
            .match_pool("anchor", &facts)
            .matched_pool_id,
        Some("pool-a")
    );
    for snapshot in &mut facts.trajectory_owner_path {
        snapshot.name = "different".to_string();
        snapshot.uuid = Uuid::new_v4();
    }
    assert_eq!(
        matcher.match_pool("anchor", &facts).matched_pool_id,
        Some("pool-a")
    );
}

#[test]
fn matcher_bypasses_nonprimary_streaming_stateful_family_and_model_mismatches() {
    let matcher = PoolMatcher::compile(&router_config(vec![pool("pool-a", "tenant-a")])).unwrap();
    let mut facts = context(Some("tenant-a"), None, &[ScopeType::Agent], BTreeMap::new());
    assert!(
        matcher
            .match_pool("anchor", &facts)
            .matched_pool_id
            .is_some()
    );

    facts.call_role = LlmCallRole::Shadow;
    assert!(
        matcher
            .match_pool("anchor", &facts)
            .matched_pool_id
            .is_none()
    );
    facts.call_role = LlmCallRole::Judge;
    assert!(
        matcher
            .match_pool("anchor", &facts)
            .matched_pool_id
            .is_none()
    );
    facts.call_role = LlmCallRole::Primary;
    facts.attributes = LlmAttributes::STREAMING;
    assert!(
        matcher
            .match_pool("anchor", &facts)
            .matched_pool_id
            .is_none()
    );
    facts.attributes = LlmAttributes::STATEFUL;
    assert!(
        matcher
            .match_pool("anchor", &facts)
            .matched_pool_id
            .is_none()
    );
    facts.attributes = LlmAttributes::empty();
    facts.api_family = LlmApiFamily::OpenAIResponses;
    assert!(
        matcher
            .match_pool("anchor", &facts)
            .matched_pool_id
            .is_none()
    );
    facts.api_family = LlmApiFamily::OpenAIChatCompletions;
    assert!(
        matcher
            .match_pool("different-model", &facts)
            .matched_pool_id
            .is_none()
    );
}

#[test]
fn invalid_pattern_shapes_report_exact_indexed_locations() {
    let invalid_patterns = [
        "",
        "/agent",
        "agent/",
        "agent//tool",
        "agent/**/tool",
        "agent/t*",
        "Agent",
    ];
    for pattern in invalid_patterns {
        let mut value = json!({"mode": "shadow", "pools": [pool("pool-a", "tenant-a")]});
        value["pools"][0]["selector"]["scope_path_patterns"] = json!([pattern]);
        let report = validate_router_config(value.as_object().unwrap());
        assert!(
            report.diagnostics.iter().any(|diagnostic| {
                diagnostic.code == "router.invalid_reference"
                    && diagnostic.field.as_deref()
                        == Some("pools[0].selector.scope_path_patterns[0]")
            }),
            "{pattern:?}: {:?}",
            report.diagnostics
        );
    }
}

#[test]
fn malformed_selector_values_fail_before_matcher_construction() {
    let malformed = json!({
        "mode": "shadow",
        "pools": [{
            "id": "pool-a",
            "api_family": "openai_chat_completions",
            "anchor_models": ["anchor"],
            "anchor_revision": "r1",
            "sampling_probability": 1.0,
            "max_candidates_per_sample": 1,
            "selector": {
                "tenant_ids": [],
                "owner_scope_types": [],
                "metadata_equals": {"object": {"secret": true}}
            },
            "concurrency": {"shadow": 1, "judge": 1},
            "judge": judge_config(),
            "candidates": [{
                "id": "candidate", "model": "candidate-model",
                "model_revision": "r1", "cost_rank": 0
            }]
        }]
    });
    let report = validate_router_config(malformed.as_object().unwrap());
    assert!(report.has_errors());
    assert_eq!(report.config_generation_id, None);
    let config = report.config.unwrap();
    assert!(PoolMatcher::compile(&config).is_err());
}

#[test]
fn map_helper_preserves_json_object_contract() {
    let map: Map<String, Json> = json!({"a": 1}).as_object().unwrap().clone();
    assert_eq!(map.get("a"), Some(&json!(1)));
}
