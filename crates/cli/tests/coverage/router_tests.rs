// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{self, Cursor, Write};

use clap::Parser;
use nemo_relay::plugin::{ConfigPolicy, PluginComponentSpec, PluginConfig};
use serde_json::json;

use super::*;
use crate::config::{Cli, Command};

fn router_config(path: &Path, project_id: &str) -> RouterConfig {
    serde_json::from_value(json!({
        "version": 1,
        "mode": "shadow",
        "project_id": project_id,
        "database_path": path.to_string_lossy(),
        "retention_days": 30,
        "max_evidence_records": 1000,
        "pools": [{
            "id": "pool-a",
            "api_family": "openai_chat_completions",
            "anchor_models": ["anchor-a"],
            "anchor_revision": "2026-07-01",
            "sampling_probability": 0.25,
            "max_candidates_per_sample": 1,
            "concurrency": {"shadow": 2, "judge": 1},
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
                "id": "candidate-a",
                "model": "candidate-model-a",
                "model_revision": "2026-06-01",
                "cost_rank": 0,
                "capabilities": {}
            }]
        }]
    }))
    .unwrap()
}

fn plugin_config(components: Vec<PluginComponentSpec>) -> PluginConfig {
    PluginConfig {
        version: 1,
        components,
        policy: ConfigPolicy::default(),
    }
}

fn write_command_config(temporary: &tempfile::TempDir, plugin: &PluginConfig) -> ServerArgs {
    let config_path = temporary.path().join("config.toml");
    std::fs::write(&config_path, "").unwrap();
    std::fs::write(
        temporary.path().join("plugins.toml"),
        toml::to_string_pretty(plugin).unwrap(),
    )
    .unwrap();
    ServerArgs {
        config: Some(config_path),
        ..ServerArgs::default()
    }
}

fn parse_router(arguments: &[&str]) -> RouterCommand {
    let parsed = Cli::try_parse_from(arguments).unwrap();
    let Some(Command::Router(command)) = parsed.command else {
        panic!("expected Router command");
    };
    command
}

async fn run_router_json(arguments: &[&str], server: &ServerArgs) -> (ExitCode, Value) {
    let mut output = Vec::new();
    let mut errors = Vec::new();
    let exit = run_with_io(parse_router(arguments), server, &mut output, &mut errors)
        .await
        .unwrap();
    assert!(errors.is_empty());
    (exit, serde_json::from_slice(&output).unwrap())
}

#[test]
fn router_clap_tree_is_exact_and_excludes_unimplemented_or_freeform_leaves() {
    for arguments in [
        vec!["nemo-relay", "router", "dashboard"],
        vec!["nemo-relay", "router", "status", "--json"],
        vec!["nemo-relay", "router", "pools", "--limit", "1"],
        vec![
            "nemo-relay",
            "router",
            "evidence",
            "list",
            "--pool",
            "pool-a",
        ],
        vec![
            "nemo-relay",
            "router",
            "evidence",
            "show",
            "018f47b8-0000-7000-8000-000000000001",
        ],
        vec![
            "nemo-relay",
            "router",
            "evidence",
            "export",
            "evidence.jsonl",
            "--format",
            "jsonl",
        ],
        vec![
            "nemo-relay",
            "router",
            "neighborhood",
            "inspect",
            "--evidence-id",
            "018f47b8-0000-7000-8000-000000000001",
        ],
        vec![
            "nemo-relay",
            "router",
            "neighborhood",
            "inspect",
            "--query-hash",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "--partition-file",
            "partition.json",
        ],
        vec![
            "nemo-relay",
            "router",
            "neighborhood",
            "inspect",
            "--request-file",
            "-",
        ],
        vec![
            "nemo-relay",
            "router",
            "decisions",
            "tail",
            "--follow",
            "--cursor",
            "opaque",
            "--json",
        ],
        vec!["nemo-relay", "router", "pause", "--reason", "maintenance"],
        vec![
            "nemo-relay",
            "router",
            "resume",
            "--reason",
            "ready",
            "--actor",
            "operator-a",
            "--pool",
            "pool-a",
            "--json",
        ],
        vec![
            "nemo-relay",
            "router",
            "force-anchor",
            "set",
            "--reason",
            "incident",
        ],
        vec![
            "nemo-relay",
            "router",
            "force-anchor",
            "clear",
            "--reason",
            "recovered",
            "--pool",
            "pool-a",
        ],
        vec![
            "nemo-relay",
            "router",
            "reset",
            "--pool",
            "pool-a",
            "--reason",
            "new-corpus",
            "--confirm",
            "project-a",
        ],
        vec![
            "nemo-relay",
            "router",
            "reset",
            "--all",
            "--reason",
            "new-corpus",
            "--confirm",
            "project-a",
            "--json",
        ],
        vec![
            "nemo-relay",
            "router",
            "cohort",
            "rotate",
            "--reason",
            "new-cohort",
            "--confirm",
            "project-a",
        ],
    ] {
        assert!(Cli::try_parse_from(arguments).is_ok());
    }

    assert!(
        Cli::try_parse_from([
            "nemo-relay",
            "router",
            "neighborhood",
            "inspect",
            "--query",
            "raw prompt",
        ])
        .is_err()
    );
    assert!(
        Cli::try_parse_from([
            "nemo-relay",
            "router",
            "neighborhood",
            "inspect",
            "--query-hash",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        ])
        .is_err()
    );
    assert!(
        Cli::try_parse_from([
            "nemo-relay",
            "router",
            "neighborhood",
            "inspect",
            "--evidence-id",
            "018f47b8-0000-7000-8000-000000000001",
            "--request-file",
            "request.json",
        ])
        .is_err()
    );
    assert!(Cli::try_parse_from(["nemo-relay", "router", "pools", "--limit", "501",]).is_err());
    assert!(
        Cli::try_parse_from([
            "nemo-relay",
            "router",
            "reset",
            "--reason",
            "missing-scope",
            "--confirm",
            "project-a",
        ])
        .is_err()
    );
    assert!(
        Cli::try_parse_from([
            "nemo-relay",
            "router",
            "reset",
            "--all",
            "--pool",
            "pool-a",
            "--reason",
            "two-scopes",
            "--confirm",
            "project-a",
        ])
        .is_err()
    );
    assert!(
        Cli::try_parse_from([
            "nemo-relay",
            "router",
            "pause",
            "--reason",
            "maintenance",
            "--mutation-id",
            "018f47b8-0000-7000-8000-000000000001",
        ])
        .is_err()
    );
}

#[test]
fn bounded_json_decoder_accepts_exact_input_and_rejects_invalid_or_oversized_data() {
    let value: Value = decode_bounded_json(Cursor::new(br#"{"value":1}"#)).unwrap();
    assert_eq!(value, json!({"value": 1}));
    assert_eq!(
        decode_bounded_json::<Value>(Cursor::new(b"{"))
            .unwrap_err()
            .code(),
        "invalid_json_input"
    );
    assert_eq!(
        decode_bounded_json::<Value>(Cursor::new(vec![b' '; INSPECTION_INPUT_MAX_BYTES + 1]))
            .unwrap_err()
            .code(),
        "input_too_large"
    );
}

#[test]
fn json_and_text_renderers_are_deterministic_and_schema_versioned() {
    let report = StatusReportV1 {
        schema: "nemo_relay.router.status.v1".into(),
        project_id: None,
        configured_mode: nemo_relay_router::RouterMode::Shadow,
        effective_mode: nemo_relay_router::inspection::EffectiveRouterModeV1::Unavailable,
        config_generation_id: None,
        cohort_generation_id: None,
        controls: None,
        database: nemo_relay_router::inspection::DatabaseStatusV1 {
            state: nemo_relay_router::inspection::InspectionDatabaseStateV1::Missing,
            application_id: None,
            schema_version: None,
            supported_schema_version: 7,
        },
        queues: Default::default(),
        leases: Default::default(),
        vector_index: Default::default(),
        freshness: Default::default(),
        health: Default::default(),
        snapshot_time_unix_ms: 123,
    };
    let mut text = Vec::new();
    write_status_text(&mut text, &report).unwrap();
    assert_eq!(
        String::from_utf8(text).unwrap(),
        "Project: unavailable\nConfigured mode: shadow\nEffective mode: unavailable\nDatabase: missing\nSchema: unavailable/7\nQueues: 0/0 pending\nHealth: clear\nSnapshot: 123\n"
    );

    let mut output = Vec::new();
    write_success_json(&mut output, "router status", &report).unwrap();
    let envelope: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(envelope["schema_version"], 1);
    assert_eq!(envelope["ok"], true);
    assert_eq!(envelope["command"], "router status");
    assert_eq!(envelope["data"]["snapshot_time_unix_ms"], 123);
}

#[test]
fn broken_output_is_normal_and_partial_files_are_removed() {
    struct BrokenWriter;

    impl Write for BrokenWriter {
        fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }
    }

    assert!(matches!(
        write_bytes(&mut BrokenWriter, b"value"),
        Err(RouterFailure::OutputClosed)
    ));

    let temporary = tempfile::tempdir().unwrap();
    let path = temporary.path().join("partial.jsonl");
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let mut partial = PartialOutput::new(path.clone(), file);
    partial.writer().write_all(b"partial").unwrap();
    drop(partial);
    assert!(!path.exists());
}

#[test]
fn inspection_failures_use_the_locked_exit_and_error_codes() {
    for error in [
        InspectionError::InvalidArgument,
        InspectionError::InvalidCursor,
        InspectionError::NotFound,
        InspectionError::Forbidden,
        InspectionError::Conflict,
        InspectionError::EgressDenied,
    ] {
        let failure = RouterFailure::Inspection(error);
        assert_eq!(failure.exit_code(), ExitCode::from(2));
        assert_eq!(failure.code(), error.code());
    }
    for error in [
        InspectionError::Busy,
        InspectionError::StorageUnavailable,
        InspectionError::MigrationRequired,
        InspectionError::CapacityExhausted,
        InspectionError::IntegrityError,
    ] {
        let failure = RouterFailure::Inspection(error);
        assert_eq!(failure.exit_code(), ExitCode::FAILURE);
        assert_eq!(failure.code(), error.code());
    }
}

#[test]
fn command_json_detection_matches_only_structured_output_commands() {
    let parsed = Cli::try_parse_from(["nemo-relay", "router", "status", "--json"]).unwrap();
    let Some(Command::Router(command)) = parsed.command else {
        panic!("expected Router command");
    };
    assert!(command_uses_json(&command));
    assert_eq!(command_name(&command), "router status");

    let parsed = Cli::try_parse_from(["nemo-relay", "router", "evidence", "export", "-"]).unwrap();
    let Some(Command::Router(command)) = parsed.command else {
        panic!("expected Router command");
    };
    assert!(!command_uses_json(&command));
    assert_eq!(command_name(&command), "router evidence export");
}

#[test]
fn every_neighborhood_lookup_form_builds_from_typed_bounded_input() {
    let temporary = tempfile::tempdir().unwrap();
    let partition: RoutingPartitionV1 = serde_json::from_value(json!({
        "tenant_policy_hash": "1".repeat(64),
        "agent_policy_hash": "2".repeat(64),
        "policy_version_id": "3".repeat(64),
        "learning_generation_id": "018f47b8-0000-7000-8000-000000000001",
        "api_family": "openai_chat_completions",
        "transport_identity": "transport-v1",
        "anchor_model": "anchor-a",
        "anchor_revision": "2026-07-01",
        "candidate_id": "candidate-a",
        "candidate_model": "candidate-model-a",
        "candidate_model_revision": "2026-06-01",
        "decoding_fingerprint": "4".repeat(64),
        "evaluator_version": "5".repeat(64),
        "vector_space_id": "6".repeat(64)
    }))
    .unwrap();
    let partition_path = temporary.path().join("partition.json");
    std::fs::write(&partition_path, serde_json::to_vec(&partition).unwrap()).unwrap();
    let query = neighborhood_lookup(RouterNeighborhoodInspectCommand {
        evidence_id: None,
        query_hash: Some("a".repeat(64)),
        partition_file: Some(partition_path),
        request_file: None,
        json: false,
    })
    .unwrap();
    assert!(matches!(query, NeighborhoodLookupV1::QueryHash { .. }));

    let request: RoutingInspectionInputV1 = serde_json::from_value(json!({
        "schema": "nemo.relay.router.inspection-input@1",
        "pool_id": "pool-a",
        "partition": partition,
        "request": {
            "schema": "nemo.relay.router.request-projection@1",
            "family": "openai_chat_completions",
            "normalized_request": {
                "messages": [{"role": "user", "content": "hello", "name": null}],
                "model": "anchor-a",
                "params": null,
                "tools": null,
                "tool_choice": null,
                "response_format": null,
                "truncation": null,
                "reasoning": null,
                "service_tier": null,
                "parallel_tool_calls": null,
                "max_output_tokens": null,
                "max_tool_calls": null,
                "top_logprobs": null
            },
            "ordered_instructions": [],
            "response_format": null,
            "response_schema_fingerprint": null,
            "required_capabilities": [],
            "sanitizer_version": 1,
            "semantic_request_fingerprint": "7".repeat(64)
        },
        "routing_context": {
            "schema": "nemo.relay.router.routing-context@1",
            "tenant_policy_hash": "1".repeat(64),
            "agent_policy_hash": "2".repeat(64),
            "position_features": {}
        }
    }))
    .unwrap();
    let request_path = temporary.path().join("request.json");
    std::fs::write(&request_path, serde_json::to_vec(&request).unwrap()).unwrap();
    let request = neighborhood_lookup(RouterNeighborhoodInspectCommand {
        evidence_id: None,
        query_hash: None,
        partition_file: None,
        request_file: Some(request_path),
        json: true,
    })
    .unwrap();
    assert!(matches!(request, NeighborhoodLookupV1::Request { .. }));

    let evidence = neighborhood_lookup(RouterNeighborhoodInspectCommand {
        evidence_id: Some("018f47b8-0000-7000-8000-000000000001".parse().unwrap()),
        query_hash: None,
        partition_file: None,
        request_file: None,
        json: false,
    })
    .unwrap();
    assert!(matches!(evidence, NeighborhoodLookupV1::Evidence { .. }));
}

#[test]
fn decision_follow_jsonl_carries_the_resume_cursor() {
    let decision = DecisionSummaryV1 {
        decision_id: "018f47b8-0000-7000-8000-000000000001".parse().unwrap(),
        pool_id: "pool-a".into(),
        mode: "recommend".into(),
        candidate_id: None,
        recommended_model: "anchor-a".into(),
        served_model: "anchor-a".into(),
        final_reason: "no_partition".into(),
        canonical_query_hash: "a".repeat(64),
        cohort_generation_id: None,
        created_at_unix_ms: 10,
    };
    let page = Page {
        items: vec![decision],
        next: Some("resume-cursor".into()),
        snapshot_time_unix_ms: 11,
        content_policy: nemo_relay_router::inspection::ContentPolicy::Redacted,
    };
    let mut output = Vec::new();
    write_decision_jsonl(&mut output, &page, None).unwrap();
    let record: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(record["schema_version"], 1);
    assert_eq!(record["cursor"], "resume-cursor");
    assert_eq!(
        record["decision"]["decision_id"],
        "018f47b8-0000-7000-8000-000000000001"
    );
}

#[test]
fn mutation_context_derives_or_overrides_actor_and_rejects_invalid_text() {
    let command = parse_router(&["nemo-relay", "router", "pause", "--reason", "maintenance"]);
    let context = prepare_mutation_context(&command.command).unwrap().unwrap();
    assert_eq!(
        context.actor,
        crate::principal::authenticated_principal().unwrap()
    );

    let command = parse_router(&[
        "nemo-relay",
        "router",
        "pause",
        "--reason",
        "maintenance",
        "--actor",
        "operator-a",
    ]);
    assert_eq!(
        prepare_mutation_context(&command.command)
            .unwrap()
            .unwrap()
            .actor,
        "operator-a"
    );

    let command = parse_router(&["nemo-relay", "router", "pause", "--reason", ""]);
    assert_eq!(
        prepare_mutation_context(&command.command)
            .unwrap_err()
            .code(),
        "invalid_reason"
    );
    let oversized = "a".repeat(CONTROL_ACTOR_MAX_BYTES + 1);
    let command = parse_router(&[
        "nemo-relay",
        "router",
        "pause",
        "--reason",
        "maintenance",
        "--actor",
        oversized.as_str(),
    ]);
    assert_eq!(
        prepare_mutation_context(&command.command)
            .unwrap_err()
            .code(),
        "invalid_actor"
    );
}

#[tokio::test]
async fn runner_reports_missing_disabled_and_multiple_router_components_as_refusals() {
    for (plugin, expected_code) in [
        (plugin_config(Vec::new()), "router_not_configured"),
        (
            plugin_config(vec![PluginComponentSpec {
                enabled: false,
                ..nemo_relay_router::ComponentSpec::new(router_config(
                    Path::new("disabled.db"),
                    "disabled-router",
                ))
                .into()
            }]),
            "router_disabled",
        ),
        (
            plugin_config(vec![
                nemo_relay_router::ComponentSpec::new(router_config(
                    Path::new("first.db"),
                    "first-router",
                ))
                .into(),
                nemo_relay_router::ComponentSpec::new(router_config(
                    Path::new("second.db"),
                    "second-router",
                ))
                .into(),
            ]),
            "invalid_configuration",
        ),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let server = write_command_config(&temporary, &plugin);
        let mut output = Vec::new();
        let mut errors = Vec::new();
        let exit = run_with_io(
            parse_router(&["nemo-relay", "router", "status", "--json"]),
            &server,
            &mut output,
            &mut errors,
        )
        .await
        .unwrap();
        assert_eq!(exit, ExitCode::from(2));
        assert!(errors.is_empty());
        let envelope: Value = serde_json::from_slice(&output).unwrap();
        assert_eq!(envelope["ok"], false);
        assert_eq!(envelope["error"]["code"], expected_code);
    }
}

#[tokio::test]
async fn dashboard_uses_the_same_exact_router_component_resolution() {
    for (plugin, expected_code) in [
        (plugin_config(Vec::new()), "router_not_configured"),
        (
            plugin_config(vec![PluginComponentSpec {
                enabled: false,
                ..nemo_relay_router::ComponentSpec::new(router_config(
                    Path::new("disabled.db"),
                    "disabled-router",
                ))
                .into()
            }]),
            "router_disabled",
        ),
        (
            plugin_config(vec![
                nemo_relay_router::ComponentSpec::new(router_config(
                    Path::new("first.db"),
                    "first-router",
                ))
                .into(),
                nemo_relay_router::ComponentSpec::new(router_config(
                    Path::new("second.db"),
                    "second-router",
                ))
                .into(),
            ]),
            "invalid_configuration",
        ),
    ] {
        let temporary = tempfile::tempdir().unwrap();
        let server = write_command_config(&temporary, &plugin);
        let mut output = Vec::new();
        let mut errors = Vec::new();
        let exit = run_with_io(
            parse_router(&["nemo-relay", "router", "dashboard", "--no-open"]),
            &server,
            &mut output,
            &mut errors,
        )
        .await
        .unwrap();
        assert_eq!(exit, ExitCode::from(2));
        assert!(output.is_empty());
        let errors = String::from_utf8(errors).unwrap();
        assert!(errors.contains(expected_code), "{errors}");
        assert!(errors.len() < 1_024);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn runner_reads_a_real_ledger_and_handles_export_and_stable_failures() {
    let _plugins = crate::test_support::PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let temporary = tempfile::tempdir().unwrap();
    let database = temporary.path().join("router.db");
    let component =
        nemo_relay_router::ComponentSpec::new(router_config(&database, "router-cli-reads"));
    let plugin = plugin_config(vec![component.into()]);
    nemo_relay_router::register_router_component().unwrap();
    let report = nemo_relay::plugin::initialize_plugins_exact(plugin.clone())
        .await
        .unwrap();
    assert!(!report.has_errors());
    nemo_relay::plugin::clear_plugin_configuration_async(Duration::from_secs(10))
        .await
        .unwrap();
    assert!(database.exists());
    let server = write_command_config(&temporary, &plugin);

    let mut output = Vec::new();
    let mut errors = Vec::new();
    assert_eq!(
        run_with_io(
            parse_router(&["nemo-relay", "router", "status", "--json"]),
            &server,
            &mut output,
            &mut errors,
        )
        .await
        .unwrap(),
        ExitCode::SUCCESS
    );
    let status: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(status["data"]["project_id"], "router-cli-reads");
    assert_eq!(status["data"]["database"]["state"], "current");
    assert!(errors.is_empty());

    output.clear();
    assert_eq!(
        run_with_io(
            parse_router(&["nemo-relay", "router", "pools", "--limit", "1", "--json",]),
            &server,
            &mut output,
            &mut errors,
        )
        .await
        .unwrap(),
        ExitCode::SUCCESS
    );
    let pools: Value = serde_json::from_slice(&output).unwrap();
    assert_eq!(pools["data"]["items"][0]["id"], "pool-a");

    output.clear();
    assert_eq!(
        run_with_io(
            parse_router(&[
                "nemo-relay",
                "router",
                "evidence",
                "show",
                "018f47b8-0000-7000-8000-000000000001",
                "--json",
            ]),
            &server,
            &mut output,
            &mut errors,
        )
        .await
        .unwrap(),
        ExitCode::from(2)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&output).unwrap()["error"]["code"],
        "not_found"
    );

    output.clear();
    assert_eq!(
        run_with_io(
            parse_router(&[
                "nemo-relay",
                "router",
                "decisions",
                "tail",
                "--cursor",
                "invalid",
                "--json",
            ]),
            &server,
            &mut output,
            &mut errors,
        )
        .await
        .unwrap(),
        ExitCode::from(2)
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&output).unwrap()["error"]["code"],
        "invalid_cursor"
    );

    let export = temporary.path().join("evidence.csv");
    let export_arg = export.to_string_lossy().to_string();
    output.clear();
    assert_eq!(
        run_with_io(
            parse_router(&[
                "nemo-relay",
                "router",
                "evidence",
                "export",
                export_arg.as_str(),
                "--format",
                "csv",
            ]),
            &server,
            &mut output,
            &mut errors,
        )
        .await
        .unwrap(),
        ExitCode::SUCCESS
    );
    let header = std::fs::read_to_string(&export).unwrap();
    assert!(header.starts_with("evidence_id,pool_id,candidate_id"));

    errors.clear();
    assert_eq!(
        run_with_io(
            parse_router(&[
                "nemo-relay",
                "router",
                "evidence",
                "export",
                export_arg.as_str(),
            ]),
            &server,
            &mut output,
            &mut errors,
        )
        .await
        .unwrap(),
        ExitCode::from(2)
    );
    assert!(String::from_utf8_lossy(&errors).contains("already exists"));
    assert_eq!(std::fs::read_to_string(&export).unwrap(), header);

    errors.clear();
    assert_eq!(
        run_with_io(
            parse_router(&[
                "nemo-relay",
                "router",
                "evidence",
                "export",
                export_arg.as_str(),
                "--force",
            ]),
            &server,
            &mut output,
            &mut errors,
        )
        .await
        .unwrap(),
        ExitCode::SUCCESS
    );
    assert!(std::fs::read(&export).unwrap().is_empty());

    struct BrokenWriter;

    impl Write for BrokenWriter {
        fn write(&mut self, _buffer: &[u8]) -> io::Result<usize> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }

        fn flush(&mut self) -> io::Result<()> {
            Err(io::Error::from(io::ErrorKind::BrokenPipe))
        }
    }

    assert_eq!(
        run_with_io(
            parse_router(&[
                "nemo-relay",
                "router",
                "evidence",
                "export",
                "-",
                "--format",
                "csv",
            ]),
            &server,
            &mut BrokenWriter,
            &mut errors,
        )
        .await
        .unwrap(),
        ExitCode::SUCCESS
    );

    let failed_export = temporary.path().join("failed.jsonl");
    let failed_export_arg = failed_export.to_string_lossy().to_string();
    errors.clear();
    assert_eq!(
        run_with_io(
            parse_router(&[
                "nemo-relay",
                "router",
                "evidence",
                "export",
                failed_export_arg.as_str(),
                "--terminal-class",
                "invalid",
            ]),
            &server,
            &mut output,
            &mut errors,
        )
        .await
        .unwrap(),
        ExitCode::from(2)
    );
    assert!(!failed_export.exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn mutation_commands_preserve_control_independence_and_generation_scope() {
    let _plugins = crate::test_support::PLUGIN_CONFIG_TEST_LOCK.lock().await;
    let temporary = tempfile::tempdir().unwrap();
    let database = temporary.path().join("router.db");
    let mut router = router_config(&database, "router-cli-mutations");
    router.pools[0].selector.tenant_ids = Some(vec!["tenant-a".into()]);
    let mut second_pool = router.pools[0].clone();
    second_pool.id = "pool-b".into();
    second_pool.selector.tenant_ids = Some(vec!["tenant-b".into()]);
    router.pools.push(second_pool);
    let component = nemo_relay_router::ComponentSpec::new(router.clone());
    let plugin = plugin_config(vec![component.into()]);
    nemo_relay_router::register_router_component().unwrap();
    let report = nemo_relay::plugin::initialize_plugins_exact(plugin.clone())
        .await
        .unwrap();
    assert!(!report.has_errors());
    nemo_relay::plugin::clear_plugin_configuration_async(Duration::from_secs(10))
        .await
        .unwrap();
    let server = write_command_config(&temporary, &plugin);
    let default_actor = crate::principal::authenticated_principal().unwrap();

    let reader = InspectionService::open(router.clone(), InspectionServiceOptions::default())
        .await
        .unwrap();
    let initial_pools = reader
        .list_pools(PageRequest {
            limit: 10,
            after: None,
        })
        .await
        .unwrap();
    let initial_pool_b = initial_pools
        .items
        .iter()
        .find(|pool| pool.id == "pool-b")
        .unwrap()
        .learning_generation_id;
    reader.close().await.unwrap();

    let (exit, paused) = run_router_json(
        &[
            "nemo-relay",
            "router",
            "pause",
            "--reason",
            "maintenance",
            "--json",
        ],
        &server,
    )
    .await;
    assert_eq!(exit, ExitCode::SUCCESS);
    let pause_audit: uuid::Uuid = paused["data"]["mutation_id"]
        .as_str()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(pause_audit.get_version_num(), 7);
    assert_eq!(paused["data"]["snapshot"]["all"]["paused"], true);
    assert_eq!(paused["data"]["snapshot"]["all"]["force_anchor"], false);

    let (exit, forced) = run_router_json(
        &[
            "nemo-relay",
            "router",
            "force-anchor",
            "set",
            "--pool",
            "pool-a",
            "--reason",
            "incident",
            "--actor",
            "operator-a",
            "--json",
        ],
        &server,
    )
    .await;
    assert_eq!(exit, ExitCode::SUCCESS);
    assert_eq!(
        forced["data"]["snapshot"]["pools"]["pool-a"]["effective"]["paused"],
        true
    );
    assert_eq!(
        forced["data"]["snapshot"]["pools"]["pool-a"]["effective"]["force_anchor"],
        true
    );

    let (exit, resumed) = run_router_json(
        &[
            "nemo-relay",
            "router",
            "resume",
            "--reason",
            "ready",
            "--json",
        ],
        &server,
    )
    .await;
    assert_eq!(exit, ExitCode::SUCCESS);
    assert_eq!(resumed["data"]["snapshot"]["all"]["paused"], false);
    assert_eq!(
        resumed["data"]["snapshot"]["pools"]["pool-a"]["effective"]["force_anchor"],
        true
    );

    let (exit, cleared) = run_router_json(
        &[
            "nemo-relay",
            "router",
            "force-anchor",
            "clear",
            "--pool",
            "pool-a",
            "--reason",
            "recovered",
            "--json",
        ],
        &server,
    )
    .await;
    assert_eq!(exit, ExitCode::SUCCESS);
    assert_eq!(
        cleared["data"]["snapshot"]["pools"]["pool-a"]["effective"]["force_anchor"],
        false
    );

    let (exit, mismatch) = run_router_json(
        &[
            "nemo-relay",
            "router",
            "reset",
            "--pool",
            "pool-a",
            "--reason",
            "new-corpus",
            "--confirm",
            "wrong-project",
            "--json",
        ],
        &server,
    )
    .await;
    assert_eq!(exit, ExitCode::from(2));
    assert_eq!(mismatch["error"]["code"], "invalid_argument");

    let (exit, pool_reset) = run_router_json(
        &[
            "nemo-relay",
            "router",
            "reset",
            "--pool",
            "pool-a",
            "--reason",
            "new-corpus",
            "--confirm",
            "router-cli-mutations",
            "--json",
        ],
        &server,
    )
    .await;
    assert_eq!(exit, ExitCode::SUCCESS);
    assert_eq!(pool_reset["data"]["result"], "applied");
    assert!(pool_reset["data"]["resulting_generations"]["pool-a"].is_string());

    let reader = InspectionService::open(router.clone(), InspectionServiceOptions::default())
        .await
        .unwrap();
    let after_pool_reset = reader
        .list_pools(PageRequest {
            limit: 10,
            after: None,
        })
        .await
        .unwrap();
    assert_eq!(
        after_pool_reset
            .items
            .iter()
            .find(|pool| pool.id == "pool-b")
            .unwrap()
            .learning_generation_id,
        initial_pool_b
    );
    reader.close().await.unwrap();

    let (exit, all_reset) = run_router_json(
        &[
            "nemo-relay",
            "router",
            "reset",
            "--all",
            "--reason",
            "new-corpus-all",
            "--confirm",
            "router-cli-mutations",
            "--json",
        ],
        &server,
    )
    .await;
    assert_eq!(exit, ExitCode::SUCCESS);
    assert_eq!(
        all_reset["data"]["resulting_generations"]
            .as_object()
            .unwrap()
            .len(),
        2
    );

    let (exit, rotation) = run_router_json(
        &[
            "nemo-relay",
            "router",
            "cohort",
            "rotate",
            "--reason",
            "new-cohort",
            "--confirm",
            "router-cli-mutations",
            "--json",
        ],
        &server,
    )
    .await;
    assert_eq!(exit, ExitCode::SUCCESS);
    assert_eq!(rotation["data"]["result"], "applied");
    assert!(rotation["data"]["resulting_generations"]["cohort"].is_string());

    let reader = InspectionService::open(router, InspectionServiceOptions::default())
        .await
        .unwrap();
    let history = reader
        .list_controls(PageRequest {
            limit: 20,
            after: None,
        })
        .await
        .unwrap();
    assert!(history.items.iter().any(|entry| {
        entry.audit_id == pause_audit
            && entry.actor == default_actor
            && entry.reason == "maintenance"
    }));
    assert!(
        history
            .items
            .iter()
            .any(|entry| entry.actor == "operator-a" && entry.reason == "incident")
    );
    assert!(
        history
            .items
            .iter()
            .any(|entry| entry.reason == "new-corpus")
    );
    assert!(
        history
            .items
            .iter()
            .any(|entry| entry.reason == "new-cohort")
    );
    reader.close().await.unwrap();
}
