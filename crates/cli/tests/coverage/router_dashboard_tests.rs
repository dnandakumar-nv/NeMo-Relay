// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use axum::http::header::CONTENT_TYPE;
use axum::http::header::{AUTHORIZATION, COOKIE, HOST, ORIGIN};
use clap::Parser as _;
use cookie::Cookie;
use http_body_util::BodyExt as _;
use nemo_relay::plugin::{ConfigPolicy, PluginComponentSpec, PluginConfig};
use nemo_relay_router::RouterConfig;
use serde_json::json;
use tokio::sync::oneshot;
use tower::ServiceExt as _;

use super::*;
use crate::config::{Cli, Command, RouterCommand, RouterSubcommand};

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

fn dashboard(arguments: &[&str]) -> RouterDashboardCommand {
    let parsed = Cli::try_parse_from(arguments).unwrap();
    let Some(Command::Router(RouterCommand {
        command: RouterSubcommand::Dashboard(command),
    })) = parsed.command
    else {
        panic!("expected dashboard command");
    };
    command
}

fn command_config(temporary: &tempfile::TempDir) -> ServerArgs {
    let config_path = temporary.path().join("config.toml");
    std::fs::write(&config_path, "").unwrap();
    let component = PluginComponentSpec::from(nemo_relay_router::ComponentSpec::new(
        router_config(&temporary.path().join("missing.db"), "dashboard-tests"),
    ));
    let plugin = PluginConfig {
        version: 1,
        components: vec![component],
        policy: ConfigPolicy::default(),
    };
    std::fs::write(
        temporary.path().join("plugins.toml"),
        toml::to_string_pretty(&plugin).unwrap(),
    )
    .unwrap();
    ServerArgs {
        config: Some(config_path),
        ..ServerArgs::default()
    }
}

async fn exercise_token_host(
    token_path: &Path,
    shutdown: oneshot::Sender<()>,
    accept_invalid_tls: bool,
) -> String {
    let launch_url = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Ok(value) = std::fs::read_to_string(token_path) {
                break value.trim().to_owned();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let url = reqwest::Url::parse(&launch_url).unwrap();
    let origin = url.origin().ascii_serialization();
    let nonce = url.fragment().unwrap().strip_prefix("bootstrap=").unwrap();
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .danger_accept_invalid_certs(accept_invalid_tls)
        .build()
        .unwrap();
    let bootstrap = client
        .post(format!("{origin}/auth/bootstrap"))
        .header(ORIGIN, &origin)
        .header(AUTHORIZATION, format!("Bootstrap {nonce}"))
        .body(Vec::new())
        .send()
        .await
        .unwrap();
    assert_eq!(bootstrap.status(), StatusCode::NO_CONTENT);
    let set_cookie = bootstrap
        .headers()
        .get(SET_COOKIE)
        .unwrap()
        .to_str()
        .unwrap()
        .to_owned();
    assert!(!token_path.exists());
    let status = client
        .get(format!("{origin}/api/router/v1/status"))
        .header(COOKIE, set_cookie.split(';').next().unwrap())
        .send()
        .await
        .unwrap();
    assert_eq!(status.status(), StatusCode::OK);
    shutdown.send(()).unwrap();
    set_cookie
}

#[test]
fn dashboard_clap_defaults_ranges_and_pairs_are_exact() {
    let command = dashboard(&["nemo-relay", "router", "dashboard"]);
    assert_eq!(command.bind, "127.0.0.1:0".parse().unwrap());
    assert_eq!(command.idle_timeout, 900);
    assert!(!command.allow_remote);
    assert!(!command.full_content);
    assert!(!command.no_open);
    assert!(command.token_file.is_none());
    assert!(command.tls_cert.is_none());
    assert!(command.tls_key.is_none());

    for arguments in [
        vec!["nemo-relay", "router", "dashboard", "--idle-timeout", "59"],
        vec![
            "nemo-relay",
            "router",
            "dashboard",
            "--idle-timeout",
            "86401",
        ],
        vec![
            "nemo-relay",
            "router",
            "dashboard",
            "--tls-cert",
            "cert.pem",
        ],
        vec!["nemo-relay", "router", "dashboard", "--tls-key", "key.pem"],
    ] {
        assert!(Cli::try_parse_from(arguments).is_err());
    }
}

#[test]
fn dashboard_validation_rejects_unspecified_and_incomplete_remote_consent() {
    let unspecified = dashboard(&["nemo-relay", "router", "dashboard", "--bind", "0.0.0.0:0"]);
    assert_eq!(
        ValidatedDashboardOptions::validate(unspecified)
            .unwrap_err()
            .code(),
        "unspecified_dashboard_bind"
    );
    let remote = dashboard(&[
        "nemo-relay",
        "router",
        "dashboard",
        "--bind",
        "192.0.2.10:4040",
    ]);
    assert_eq!(
        ValidatedDashboardOptions::validate(remote)
            .unwrap_err()
            .code(),
        "remote_dashboard_not_allowed"
    );
    let remote = dashboard(&[
        "nemo-relay",
        "router",
        "dashboard",
        "--bind",
        "192.0.2.10:4040",
        "--allow-remote",
    ]);
    assert_eq!(
        ValidatedDashboardOptions::validate(remote)
            .unwrap_err()
            .code(),
        "remote_dashboard_token_file_required"
    );
}

#[test]
fn canonical_origins_cover_ipv4_ipv6_and_default_ports() {
    assert_eq!(
        canonical_origin("http", "127.0.0.1:4040".parse().unwrap()),
        ("http://127.0.0.1:4040".into(), "127.0.0.1:4040".into())
    );
    assert_eq!(
        canonical_origin("https", "[::1]:443".parse().unwrap()),
        ("https://[::1]".into(), "[::1]".into())
    );
}

#[test]
fn dashboard_service_capabilities_are_fixed_read_only() {
    let redacted = dashboard_service_options(false);
    assert_eq!(redacted.content_policy, ContentPolicy::Redacted);
    assert!(!redacted.allow_operations);
    assert!(!redacted.allow_request_embedding);
    let full = dashboard_service_options(true);
    assert_eq!(full.content_policy, ContentPolicy::Full);
    assert!(!full.allow_operations);
    assert!(!full.allow_request_embedding);
}

#[tokio::test]
async fn embedded_dashboard_assets_are_exact_and_receive_host_headers() {
    let temporary = tempfile::tempdir().unwrap();
    let service = InspectionService::open(
        router_config(&temporary.path().join("missing.db"), "dashboard-assets"),
        InspectionServiceOptions::default(),
    )
    .await
    .unwrap();
    let activity = ActivityTracker::new();
    let (auth, _secret) = DashboardAuth::new(
        "http://127.0.0.1:4040".into(),
        "127.0.0.1:4040".into(),
        false,
        activity.clone(),
    )
    .unwrap();
    let app = dashboard_router(service.clone(), auth, activity, None).unwrap();

    for (path, content_type, expected) in [
        ("/", "text/html; charset=utf-8", INDEX_HTML),
        ("/app.js", "text/javascript; charset=utf-8", APP_JS),
        ("/styles.css", "text/css; charset=utf-8", STYLES_CSS),
    ] {
        let response = app
            .clone()
            .oneshot(Request::builder().uri(path).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response.headers().get(CONTENT_TYPE),
            Some(&HeaderValue::from_static(content_type))
        );
        assert_eq!(
            response.headers().get(CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store"))
        );
        assert_eq!(
            response.headers().get(CONTENT_SECURITY_POLICY),
            Some(&HeaderValue::from_static(CSP))
        );
        assert_eq!(
            response.headers().get(REFERRER_POLICY),
            Some(&HeaderValue::from_static("no-referrer"))
        );
        assert_eq!(
            response.headers().get(X_CONTENT_TYPE_OPTIONS),
            Some(&HeaderValue::from_static("nosniff"))
        );
        assert_eq!(
            response.headers().get(X_FRAME_OPTIONS),
            Some(&HeaderValue::from_static("DENY"))
        );
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(body.as_ref(), expected);
    }

    service.close().await.unwrap();
}

#[test]
fn embedded_dashboard_client_has_a_static_memory_only_bootstrap_boundary() {
    let html = std::str::from_utf8(INDEX_HTML).unwrap();
    let javascript = std::str::from_utf8(APP_JS).unwrap();
    let css = std::str::from_utf8(STYLES_CSS).unwrap();

    let fragment_read = javascript.find("window.location.hash").unwrap();
    let fragment_clear = javascript.find("window.history.replaceState").unwrap();
    let first_dom_access = javascript.find("document.").unwrap();
    let first_request = javascript.find("window.fetch").unwrap();
    assert!(fragment_read < fragment_clear);
    assert!(fragment_clear < first_dom_access);
    assert!(fragment_clear < first_request);
    assert_eq!(javascript.matches("/auth/bootstrap").count(), 1);

    for forbidden in [
        "localStorage",
        "sessionStorage",
        "indexedDB",
        "innerHTML",
        "outerHTML",
        "insertAdjacentHTML",
        "sourceMappingURL",
        "/api/router/v1/operations",
    ] {
        assert!(!javascript.contains(forbidden), "unexpected {forbidden}");
    }
    assert!(javascript.contains("textContent"));
    assert!(javascript.contains("const PREVIEW_LIMIT = 256"));
    assert!(javascript.contains("const CONTENT_LIMIT = 16 * 1024"));

    assert_eq!(html.matches("<script").count(), 1);
    assert!(html.contains("<script src=\"/app.js\" defer></script>"));
    assert!(!html.contains("<style"));
    assert!(!html.contains("http://") && !html.contains("https://"));
    assert!(!css.contains("http://") && !css.contains("https://"));
    let javascript_without_svg_namespace = javascript.replace("http://www.w3.org/2000/svg", "");
    assert!(!javascript_without_svg_namespace.contains("http://"));
    assert!(!javascript_without_svg_namespace.contains("https://"));
}

#[tokio::test]
async fn authenticated_router_bootstraps_once_and_applies_host_headers() {
    let temporary = tempfile::tempdir().unwrap();
    let service = InspectionService::open(
        router_config(&temporary.path().join("missing.db"), "dashboard-router"),
        InspectionServiceOptions::default(),
    )
    .await
    .unwrap();
    let activity = ActivityTracker::new();
    let (auth, secret) = DashboardAuth::new(
        "http://127.0.0.1:4040".into(),
        "127.0.0.1:4040".into(),
        false,
        activity.clone(),
    )
    .unwrap();
    let app = dashboard_router(service.clone(), auth, activity, None).unwrap();

    let root = app
        .clone()
        .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(root.status(), StatusCode::OK);
    assert_eq!(
        root.headers().get(CACHE_CONTROL),
        Some(&HeaderValue::from_static("no-store"))
    );
    assert_eq!(
        root.headers().get(CONTENT_TYPE),
        Some(&HeaderValue::from_static("text/html; charset=utf-8"))
    );

    let launch_url = secret.launch_url("http://127.0.0.1:4040");
    let nonce = launch_url.split("#bootstrap=").nth(1).unwrap();
    let bootstrap_request = || {
        Request::builder()
            .method("POST")
            .uri("/auth/bootstrap")
            .header(HOST, "127.0.0.1:4040")
            .header(ORIGIN, "http://127.0.0.1:4040")
            .header(AUTHORIZATION, format!("Bootstrap {nonce}"))
            .body(Body::empty())
            .unwrap()
    };
    let nonempty = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/auth/bootstrap")
                .header(HOST, "127.0.0.1:4040")
                .header(ORIGIN, "http://127.0.0.1:4040")
                .header(AUTHORIZATION, format!("Bootstrap {nonce}"))
                .body(Body::from("x"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(nonempty.status(), StatusCode::BAD_REQUEST);
    let bootstrapped = app.clone().oneshot(bootstrap_request()).await.unwrap();
    assert_eq!(bootstrapped.status(), StatusCode::NO_CONTENT);
    let set_cookie = bootstrapped.headers().get(SET_COOKIE).unwrap().clone();
    let parsed = Cookie::parse(set_cookie.to_str().unwrap()).unwrap();
    let request_cookie = format!("{}={}", parsed.name(), parsed.value());
    assert_eq!(
        app.clone()
            .oneshot(bootstrap_request())
            .await
            .unwrap()
            .status(),
        StatusCode::UNAUTHORIZED
    );

    let status = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/api/router/v1/status")
                .header(HOST, "127.0.0.1:4040")
                .header(COOKIE, &request_cookie)
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(status.status(), StatusCode::OK);
    assert!(!status.headers().contains_key(WWW_AUTHENTICATE));
    let body = status.into_body().collect().await.unwrap().to_bytes();
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&body).unwrap()["project_id"],
        "dashboard-router"
    );

    let operation = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/api/router/v1/operations/control")
                .header(HOST, "127.0.0.1:4040")
                .header(ORIGIN, "http://127.0.0.1:4040")
                .header(COOKIE, request_cookie)
                .header(CONTENT_TYPE, "application/json")
                .body(Body::from("{}"))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(operation.status(), StatusCode::NOT_FOUND);
    service.close().await.unwrap();
}

#[tokio::test]
async fn host_validation_precedes_config_and_filesystem_work() {
    let command = dashboard(&[
        "nemo-relay",
        "router",
        "dashboard",
        "--bind",
        "0.0.0.0:0",
        "--no-open",
    ]);
    let server = ServerArgs {
        config: Some("definitely-missing-dashboard-config.toml".into()),
        ..ServerArgs::default()
    };
    let mut output = Vec::new();
    let mut errors = Vec::new();
    let error = run_inner(
        command,
        &server,
        &mut output,
        &mut errors,
        |_| panic!("validation failure must not launch a browser"),
        std::future::pending(),
    )
    .await
    .unwrap_err();
    assert_eq!(error.code(), "unspecified_dashboard_bind");
    assert!(output.is_empty());
    assert!(errors.is_empty());
}

#[tokio::test]
async fn idle_shutdown_waits_for_open_stream_and_resets_after_disconnect() {
    let activity = ActivityTracker::new();
    let guard = activity.open_stream();
    assert!(
        tokio::time::timeout(
            Duration::from_millis(40),
            idle_shutdown(activity.clone(), Duration::from_millis(20))
        )
        .await
        .is_err()
    );
    drop(guard);
    tokio::time::timeout(
        Duration::from_millis(100),
        idle_shutdown(activity, Duration::from_millis(20)),
    )
    .await
    .unwrap();
}

#[test]
fn headless_and_browser_failure_output_never_duplicates_token_file_credentials() {
    let base = ValidatedDashboardOptions::validate(dashboard(&[
        "nemo-relay",
        "router",
        "dashboard",
        "--no-open",
    ]))
    .unwrap();
    let mut output = Vec::new();
    let mut errors = Vec::new();
    emit_or_launch(
        &base,
        Zeroizing::new("http://127.0.0.1/#bootstrap=secret".into()),
        false,
        &mut output,
        &mut errors,
        |_| panic!("no-open must not launch a browser"),
    )
    .unwrap();
    assert_eq!(
        String::from_utf8(output).unwrap(),
        "http://127.0.0.1/#bootstrap=secret\n"
    );
    assert!(errors.is_empty());

    let mut output = Vec::new();
    emit_or_launch(
        &base,
        Zeroizing::new("http://127.0.0.1/#bootstrap=secret".into()),
        true,
        &mut output,
        &mut errors,
        |_| panic!("no-open must not launch a browser"),
    )
    .unwrap();
    assert!(output.is_empty());

    let browser_options =
        ValidatedDashboardOptions::validate(dashboard(&["nemo-relay", "router", "dashboard"]))
            .unwrap();
    let mut output = Vec::new();
    let mut errors = Vec::new();
    emit_or_launch(
        &browser_options,
        Zeroizing::new("http://127.0.0.1/#bootstrap=fallback".into()),
        false,
        &mut output,
        &mut errors,
        |_| Err("browser unavailable".into()),
    )
    .unwrap();
    assert_eq!(
        String::from_utf8(output).unwrap(),
        "http://127.0.0.1/#bootstrap=fallback\n"
    );
    assert!(errors.is_empty());

    let mut output = Vec::new();
    emit_or_launch(
        &browser_options,
        Zeroizing::new("http://127.0.0.1/#bootstrap=suppressed".into()),
        true,
        &mut output,
        &mut errors,
        |_| Err("browser unavailable".into()),
    )
    .unwrap();
    assert!(output.is_empty());
    let errors = String::from_utf8(errors).unwrap();
    assert!(errors.contains("token file"));
    assert!(!errors.contains("suppressed"));

    let mut output = Vec::new();
    let mut errors = Vec::new();
    let mut opened = false;
    emit_or_launch(
        &browser_options,
        Zeroizing::new("http://127.0.0.1/#bootstrap=opened".into()),
        false,
        &mut output,
        &mut errors,
        |_| {
            opened = true;
            Ok(())
        },
    )
    .unwrap();
    assert!(opened);
    assert!(output.is_empty());
    assert!(errors.is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn port_zero_host_uses_token_file_bootstraps_and_shuts_down_cleanly() {
    let temporary = tempfile::tempdir().unwrap();
    let token_path = temporary.path().join("dashboard.token");
    let server = command_config(&temporary);
    let command = dashboard(&[
        "nemo-relay",
        "router",
        "dashboard",
        "--no-open",
        "--token-file",
        token_path.to_str().unwrap(),
    ]);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let mut output = Vec::new();
    let mut errors = Vec::new();
    let host = run_inner(
        command,
        &server,
        &mut output,
        &mut errors,
        |_| panic!("no-open must not launch a browser"),
        async move {
            let _ = shutdown_rx.await;
        },
    );
    let exercise = async {
        let launch_url = tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                if let Ok(value) = std::fs::read_to_string(&token_path) {
                    break value.trim().to_owned();
                }
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        let url = reqwest::Url::parse(&launch_url).unwrap();
        let origin = format!(
            "{}://{}",
            url.scheme(),
            url.host_str().unwrap().to_string()
                + &url
                    .port()
                    .map(|port| format!(":{port}"))
                    .unwrap_or_default()
        );
        let nonce = url.fragment().unwrap().strip_prefix("bootstrap=").unwrap();
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        let bootstrap = client
            .post(format!("{origin}/auth/bootstrap"))
            .header(ORIGIN, &origin)
            .header(AUTHORIZATION, format!("Bootstrap {nonce}"))
            .body(Vec::new())
            .send()
            .await
            .unwrap();
        assert_eq!(bootstrap.status(), StatusCode::NO_CONTENT);
        let set_cookie = bootstrap.headers().get(SET_COOKIE).unwrap().clone();
        assert!(!token_path.exists());
        let status = client
            .get(format!("{origin}/api/router/v1/status"))
            .header(
                COOKIE,
                set_cookie.to_str().unwrap().split(';').next().unwrap(),
            )
            .send()
            .await
            .unwrap();
        assert_eq!(status.status(), StatusCode::OK);
        shutdown_tx.send(()).unwrap();
    };
    let (result, ()) = tokio::join!(host, exercise);
    result.unwrap();
    assert!(output.is_empty());
    assert!(errors.is_empty());
    assert!(!token_path.exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn loopback_tls_host_serves_https_and_sets_a_secure_session_cookie() {
    let temporary = tempfile::tempdir().unwrap();
    let rcgen::CertifiedKey { cert, signing_key } =
        rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let cert_path = temporary.path().join("cert.pem");
    let key_path = temporary.path().join("key.pem");
    std::fs::write(&cert_path, cert.pem()).unwrap();
    std::fs::write(&key_path, signing_key.serialize_pem()).unwrap();
    let token_path = temporary.path().join("dashboard-tls.token");
    let server = command_config(&temporary);
    let command = dashboard(&[
        "nemo-relay",
        "router",
        "dashboard",
        "--no-open",
        "--token-file",
        token_path.to_str().unwrap(),
        "--tls-cert",
        cert_path.to_str().unwrap(),
        "--tls-key",
        key_path.to_str().unwrap(),
    ]);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let mut output = Vec::new();
    let mut errors = Vec::new();
    let host = run_inner(
        command,
        &server,
        &mut output,
        &mut errors,
        |_| panic!("no-open must not launch a browser"),
        async move {
            let _ = shutdown_rx.await;
        },
    );
    let exercise = exercise_token_host(&token_path, shutdown_tx, true);
    let (result, set_cookie) = tokio::join!(host, exercise);
    result.unwrap();
    assert!(set_cookie.contains("Secure"));
    assert!(output.is_empty());
    assert!(errors.is_empty());
    assert!(!token_path.exists());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ipv6_loopback_port_zero_host_uses_the_bracketed_origin() {
    let temporary = tempfile::tempdir().unwrap();
    let token_path = temporary.path().join("dashboard-v6.token");
    let server = command_config(&temporary);
    let command = dashboard(&[
        "nemo-relay",
        "router",
        "dashboard",
        "--bind",
        "[::1]:0",
        "--no-open",
        "--token-file",
        token_path.to_str().unwrap(),
    ]);
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let mut output = Vec::new();
    let mut errors = Vec::new();
    let host = run_inner(
        command,
        &server,
        &mut output,
        &mut errors,
        |_| panic!("no-open must not launch a browser"),
        async move {
            let _ = shutdown_rx.await;
        },
    );
    let exercise = exercise_token_host(&token_path, shutdown_tx, false);
    let (result, set_cookie) = tokio::join!(host, exercise);
    result.unwrap();
    assert!(!set_cookie.contains("Secure"));
    assert!(output.is_empty());
    assert!(errors.is_empty());
}

#[tokio::test]
async fn bind_failure_is_reported_and_does_not_launch_or_emit_a_credential() {
    let temporary = tempfile::tempdir().unwrap();
    let occupied = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = occupied.local_addr().unwrap().to_string();
    let server = command_config(&temporary);
    let command = dashboard(&[
        "nemo-relay",
        "router",
        "dashboard",
        "--bind",
        &address,
        "--no-open",
    ]);
    let mut output = Vec::new();
    let mut errors = Vec::new();
    let error = run_inner(
        command,
        &server,
        &mut output,
        &mut errors,
        |_| panic!("bind failure must not launch a browser"),
        std::future::pending(),
    )
    .await
    .unwrap_err();
    assert_eq!(error.code(), "dashboard_bind_failed");
    assert!(output.is_empty());
    assert!(errors.is_empty());
    drop(occupied);
}
