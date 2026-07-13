// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dedicated local host for the read-only Router dashboard.

mod auth;
mod token_file;

use std::future::Future;
use std::io::{self, Write};
use std::net::{IpAddr, SocketAddr, TcpListener};
use std::pin::Pin;
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_stream::stream;
use axum::body::{Body, to_bytes};
use axum::extract::State;
use axum::http::header::{
    CACHE_CONTROL, CONTENT_SECURITY_POLICY, CONTENT_TYPE, REFERRER_POLICY, SET_COOKIE,
    WWW_AUTHENTICATE, X_CONTENT_TYPE_OPTIONS, X_FRAME_OPTIONS,
};
use axum::http::{HeaderMap, HeaderValue, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use axum_server::Handle;
use axum_server::tls_rustls::RustlsConfig;
use futures_util::StreamExt as _;
use nemo_relay_router::inspection::{
    ContentPolicy, InspectionHttpMode, InspectionService, InspectionServiceOptions,
    inspection_http_router,
};
use serde::Serialize;
use zeroize::Zeroizing;

use self::auth::{AuthError, DashboardAuth};
use self::token_file::{TokenFileError, TransientTokenFile};
use crate::config::{RouterDashboardCommand, ServerArgs};
use crate::error::CliError;
use crate::router::{RouterFailure, resolve_router_config};

const HEADER_COUNT_MAX: usize = 64;
const HEADER_BYTES_MAX: usize = 32 * 1024;
const SERVER_SHUTDOWN_GRACE: Duration = Duration::from_secs(5);
const CSP: &str = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'";
const INDEX_HTML: &[u8] = include_bytes!("../assets/router-dashboard/index.html");
const APP_JS: &[u8] = include_bytes!("../assets/router-dashboard/app.js");
const STYLES_CSS: &[u8] = include_bytes!("../assets/router-dashboard/styles.css");

#[derive(Clone)]
pub(super) struct ActivityTracker {
    inner: Arc<ActivityInner>,
}

struct ActivityInner {
    last: Mutex<Instant>,
    streams: AtomicUsize,
}

impl ActivityTracker {
    pub(super) fn new() -> Self {
        Self {
            inner: Arc::new(ActivityInner {
                last: Mutex::new(Instant::now()),
                streams: AtomicUsize::new(0),
            }),
        }
    }

    pub(super) fn touch(&self) {
        if let Ok(mut last) = self.inner.last.lock() {
            *last = Instant::now();
        }
    }

    pub(super) fn elapsed(&self) -> Duration {
        self.inner
            .last
            .lock()
            .map(|last| last.elapsed())
            .unwrap_or(Duration::MAX)
    }

    fn has_open_stream(&self) -> bool {
        self.inner.streams.load(Ordering::Acquire) != 0
    }

    fn open_stream(&self) -> ActiveStreamGuard {
        self.inner.streams.fetch_add(1, Ordering::AcqRel);
        self.touch();
        ActiveStreamGuard {
            activity: self.clone(),
        }
    }
}

struct ActiveStreamGuard {
    activity: ActivityTracker,
}

impl Drop for ActiveStreamGuard {
    fn drop(&mut self) {
        self.activity.inner.streams.fetch_sub(1, Ordering::AcqRel);
        self.activity.touch();
    }
}

#[derive(Clone)]
struct DashboardAppState {
    auth: DashboardAuth,
    token_file: Option<Arc<TransientTokenFile>>,
}

#[derive(Debug, Clone)]
struct ValidatedDashboardOptions {
    bind: SocketAddr,
    full_content: bool,
    token_file: Option<std::path::PathBuf>,
    tls_cert: Option<std::path::PathBuf>,
    tls_key: Option<std::path::PathBuf>,
    no_open: bool,
    idle_timeout: Duration,
}

impl ValidatedDashboardOptions {
    fn validate(command: RouterDashboardCommand) -> Result<Self, DashboardFailure> {
        if command.bind.ip().is_unspecified() {
            return Err(DashboardFailure::input(
                "unspecified_dashboard_bind",
                "dashboard bind must be one concrete address, not an unspecified address",
            ));
        }
        if !(60..=86_400).contains(&command.idle_timeout) {
            return Err(DashboardFailure::input(
                "invalid_idle_timeout",
                "dashboard idle timeout must be between 60 and 86400 seconds",
            ));
        }
        if command.tls_cert.is_some() != command.tls_key.is_some() {
            return Err(DashboardFailure::input(
                "incomplete_tls_configuration",
                "dashboard TLS requires both --tls-cert and --tls-key",
            ));
        }
        if let Some(path) = command.token_file.as_deref() {
            TransientTokenFile::preflight(path).map_err(DashboardFailure::Token)?;
        }
        if !command.bind.ip().is_loopback() {
            if !command.allow_remote {
                return Err(DashboardFailure::input(
                    "remote_dashboard_not_allowed",
                    "a non-loopback dashboard bind requires --allow-remote",
                ));
            }
            if command.token_file.is_none() {
                return Err(DashboardFailure::input(
                    "remote_dashboard_token_file_required",
                    "a non-loopback dashboard bind requires --token-file",
                ));
            }
            if command.tls_cert.is_none() {
                return Err(DashboardFailure::input(
                    "remote_dashboard_tls_required",
                    "a non-loopback dashboard bind requires --tls-cert and --tls-key",
                ));
            }
        }
        Ok(Self {
            bind: command.bind,
            full_content: command.full_content,
            token_file: command.token_file,
            tls_cert: command.tls_cert,
            tls_key: command.tls_key,
            no_open: command.no_open,
            idle_timeout: Duration::from_secs(command.idle_timeout),
        })
    }

    async fn tls_config(&self) -> Result<Option<RustlsConfig>, DashboardFailure> {
        match (&self.tls_cert, &self.tls_key) {
            (Some(cert), Some(key)) => {
                let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
                RustlsConfig::from_pem_file(cert, key)
                    .await
                    .map(|config| {
                        let mut server = (*config.get_inner()).clone();
                        server.alpn_protocols = vec![b"http/1.1".to_vec()];
                        Some(RustlsConfig::from_config(Arc::new(server)))
                    })
                    .map_err(|error| {
                        DashboardFailure::input(
                            "invalid_tls_configuration",
                            format!("dashboard TLS certificate or key is invalid: {error}"),
                        )
                    })
            }
            (None, None) => Ok(None),
            _ => Err(DashboardFailure::input(
                "incomplete_tls_configuration",
                "dashboard TLS requires both --tls-cert and --tls-key",
            )),
        }
    }
}

pub(crate) async fn run_with_io(
    command: RouterDashboardCommand,
    server: &ServerArgs,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
) -> Result<ExitCode, CliError> {
    let result = run_inner(
        command,
        server,
        stdout,
        stderr,
        |url| webbrowser::open(url).map_err(|error| error.to_string()),
        shutdown_signal(),
    )
    .await;
    match result {
        Ok(()) => Ok(ExitCode::SUCCESS),
        Err(error) => {
            writeln!(
                stderr,
                "router dashboard failed [{}]: {error}",
                error.code()
            )?;
            Ok(error.exit_code())
        }
    }
}

async fn run_inner<F, S>(
    command: RouterDashboardCommand,
    server: &ServerArgs,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
    browser: F,
    external_shutdown: S,
) -> Result<(), DashboardFailure>
where
    F: FnOnce(&str) -> Result<(), String>,
    S: Future<Output = ()>,
{
    let options = ValidatedDashboardOptions::validate(command)?;
    let tls = options.tls_config().await?;
    let config = resolve_router_config(server).map_err(DashboardFailure::Router)?;
    let service = InspectionService::open(config, dashboard_service_options(options.full_content))
        .await
        .map_err(DashboardFailure::Inspection)?;
    let result = serve_dashboard(
        options,
        tls,
        service.clone(),
        stdout,
        stderr,
        browser,
        external_shutdown,
    )
    .await;
    if result.is_err() {
        let _ = service.close().await;
    }
    result
}

fn dashboard_service_options(full_content: bool) -> InspectionServiceOptions {
    InspectionServiceOptions {
        content_policy: if full_content {
            ContentPolicy::Full
        } else {
            ContentPolicy::Redacted
        },
        allow_request_embedding: false,
        allow_operations: false,
        ..InspectionServiceOptions::default()
    }
}

#[allow(clippy::too_many_arguments)]
async fn serve_dashboard<F, S>(
    options: ValidatedDashboardOptions,
    tls: Option<RustlsConfig>,
    service: InspectionService,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
    browser: F,
    external_shutdown: S,
) -> Result<(), DashboardFailure>
where
    F: FnOnce(&str) -> Result<(), String>,
    S: Future<Output = ()>,
{
    let listener = TcpListener::bind(options.bind).map_err(DashboardFailure::Bind)?;
    listener
        .set_nonblocking(true)
        .map_err(DashboardFailure::Bind)?;
    let local = listener.local_addr().map_err(DashboardFailure::Bind)?;
    let scheme = if tls.is_some() { "https" } else { "http" };
    let (origin, host) = canonical_origin(scheme, local);
    let activity = ActivityTracker::new();
    let (auth, bootstrap) =
        DashboardAuth::new(origin.clone(), host, tls.is_some(), activity.clone())
            .map_err(|_| DashboardFailure::Auth)?;
    let launch_url = bootstrap.launch_url(&origin);
    let token_file = options
        .token_file
        .as_ref()
        .map(|path| TransientTokenFile::create(path.clone(), launch_url.as_str()))
        .transpose()
        .map_err(DashboardFailure::Token)?
        .map(Arc::new);
    let app = dashboard_router(
        service.clone(),
        auth.clone(),
        activity.clone(),
        token_file.clone(),
    )?;
    let handle = Handle::new();
    let mut serving: Pin<Box<dyn Future<Output = io::Result<()>> + Send>> = match tls {
        Some(tls) => Box::pin(
            axum_server::tls_rustls::from_tcp_rustls(listener, tls)
                .map_err(DashboardFailure::Bind)?
                .http1_only()
                .handle(handle.clone())
                .serve(app.into_make_service()),
        ),
        None => Box::pin(
            axum_server::from_tcp(listener)
                .map_err(DashboardFailure::Bind)?
                .http1_only()
                .handle(handle.clone())
                .serve(app.into_make_service()),
        ),
    };

    emit_or_launch(
        &options,
        launch_url,
        token_file.is_some(),
        stdout,
        stderr,
        browser,
    )?;

    let idle = idle_shutdown(activity, options.idle_timeout);
    tokio::pin!(idle);
    tokio::pin!(external_shutdown);
    let serve_ended = tokio::select! {
        result = &mut serving => Some(result),
        () = &mut idle => None,
        () = &mut external_shutdown => None,
    };
    auth.invalidate();
    handle.graceful_shutdown(Some(SERVER_SHUTDOWN_GRACE));
    let close = service.close().await;
    let unexpected_stop = serve_ended.is_some();
    let serve: io::Result<()> = match serve_ended {
        Some(result) => result,
        None => match tokio::time::timeout(SERVER_SHUTDOWN_GRACE, &mut serving).await {
            Ok(result) => result,
            Err(_) => {
                handle.shutdown();
                serving.await
            }
        },
    };
    if let Some(token_file) = token_file {
        let _ = token_file.cleanup();
    }
    close.map_err(DashboardFailure::Inspection)?;
    serve.map_err(DashboardFailure::Serve)?;
    if unexpected_stop {
        return Err(DashboardFailure::runtime(
            "dashboard_server_stopped",
            "dashboard listener stopped before shutdown",
        ));
    }
    Ok(())
}

fn dashboard_router(
    service: InspectionService,
    auth: DashboardAuth,
    activity: ActivityTracker,
    token_file: Option<Arc<TransientTokenFile>>,
) -> Result<Router, DashboardFailure> {
    let inspection = inspection_http_router(
        service,
        Arc::new(auth.clone()),
        InspectionHttpMode::ReadOnly,
    )
    .map_err(DashboardFailure::Inspection)?;
    let state = DashboardAppState { auth, token_file };
    Ok(Router::new()
        .route("/", get(index))
        .route("/app.js", get(app_js))
        .route("/styles.css", get(styles_css))
        .route("/auth/bootstrap", post(bootstrap))
        .with_state(state)
        .merge(inspection)
        .fallback(not_found)
        .layer(middleware::from_fn_with_state(activity, dashboard_boundary)))
}

async fn index() -> Response {
    static_asset(INDEX_HTML, "text/html; charset=utf-8")
}

async fn app_js() -> Response {
    static_asset(APP_JS, "text/javascript; charset=utf-8")
}

async fn styles_css() -> Response {
    static_asset(STYLES_CSS, "text/css; charset=utf-8")
}

fn static_asset(bytes: &'static [u8], content_type: &'static str) -> Response {
    let mut response = Body::from(bytes).into_response();
    response
        .headers_mut()
        .insert(CONTENT_TYPE, HeaderValue::from_static(content_type));
    response
}

async fn bootstrap(State(state): State<DashboardAppState>, request: Request<Body>) -> Response {
    if !headers_are_bounded(request.headers()) {
        return dashboard_error(StatusCode::BAD_REQUEST, "invalid_request");
    }
    let headers = request.headers().clone();
    let body = match to_bytes(request.into_body(), 1).await {
        Ok(body) if body.is_empty() => body,
        Ok(_) | Err(_) => return dashboard_error(StatusCode::BAD_REQUEST, "nonempty_bootstrap"),
    };
    drop(body);
    match state.auth.bootstrap(&headers) {
        Ok(cookie) => {
            if let Some(token_file) = state.token_file {
                let _ = token_file.cleanup();
            }
            let mut response = StatusCode::NO_CONTENT.into_response();
            response.headers_mut().insert(SET_COOKIE, cookie);
            response
        }
        Err(AuthError::Unauthorized) => dashboard_error(StatusCode::UNAUTHORIZED, "unauthorized"),
        Err(AuthError::Forbidden) => dashboard_error(StatusCode::FORBIDDEN, "forbidden"),
        Err(AuthError::Invalid | AuthError::Entropy) => {
            dashboard_error(StatusCode::INTERNAL_SERVER_ERROR, "internal_error")
        }
    }
}

async fn not_found() -> Response {
    dashboard_error(StatusCode::NOT_FOUND, "not_found")
}

async fn dashboard_boundary(
    State(activity): State<ActivityTracker>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if !headers_are_bounded(request.headers()) {
        return secure_headers(dashboard_error(StatusCode::BAD_REQUEST, "invalid_request"));
    }
    let stream_request = request.uri().path() == "/api/router/v1/decisions/stream";
    let response = next.run(request).await;
    let mut response = if stream_request && response.status().is_success() {
        let (parts, body) = response.into_parts();
        let source = body.into_data_stream();
        let guard = activity.open_stream();
        let guarded = stream! {
            let _guard = guard;
            futures_util::pin_mut!(source);
            while let Some(item) = source.next().await {
                yield item;
            }
        };
        Response::from_parts(parts, Body::from_stream(guarded))
    } else {
        response
    };
    response.headers_mut().remove(WWW_AUTHENTICATE);
    secure_headers(response)
}

fn secure_headers(mut response: Response) -> Response {
    let headers = response.headers_mut();
    headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
    headers.insert(CONTENT_SECURITY_POLICY, HeaderValue::from_static(CSP));
    headers.insert(REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
    headers.insert(X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    headers.insert(X_FRAME_OPTIONS, HeaderValue::from_static("DENY"));
    response
}

fn headers_are_bounded(headers: &HeaderMap) -> bool {
    if headers.len() > HEADER_COUNT_MAX {
        return false;
    }
    headers
        .iter()
        .try_fold(0_usize, |total, (name, value)| {
            total
                .checked_add(name.as_str().len())?
                .checked_add(value.as_bytes().len())
        })
        .is_some_and(|total| total <= HEADER_BYTES_MAX)
}

#[derive(Serialize)]
struct DashboardErrorEnvelope {
    error: DashboardErrorBody,
}

#[derive(Serialize)]
struct DashboardErrorBody {
    code: &'static str,
}

fn dashboard_error(status: StatusCode, code: &'static str) -> Response {
    (
        status,
        Json(DashboardErrorEnvelope {
            error: DashboardErrorBody { code },
        }),
    )
        .into_response()
}

fn emit_or_launch<F>(
    options: &ValidatedDashboardOptions,
    launch_url: Zeroizing<String>,
    has_token_file: bool,
    stdout: &mut dyn Write,
    stderr: &mut dyn Write,
    browser: F,
) -> Result<(), DashboardFailure>
where
    F: FnOnce(&str) -> Result<(), String>,
{
    if options.no_open {
        if !has_token_file {
            writeln!(stdout, "{}", launch_url.as_str()).map_err(DashboardFailure::Output)?;
            stdout.flush().map_err(DashboardFailure::Output)?;
        }
        return Ok(());
    }
    if browser(launch_url.as_str()).is_err() {
        if has_token_file {
            writeln!(
                stderr,
                "dashboard browser launch failed; use the configured token file"
            )
            .map_err(DashboardFailure::Output)?;
        } else {
            writeln!(stdout, "{}", launch_url.as_str()).map_err(DashboardFailure::Output)?;
            stdout.flush().map_err(DashboardFailure::Output)?;
        }
    }
    Ok(())
}

fn canonical_origin(scheme: &str, address: SocketAddr) -> (String, String) {
    let host = match address.ip() {
        IpAddr::V4(ip) => ip.to_string(),
        IpAddr::V6(ip) => format!("[{ip}]"),
    };
    let default_port = matches!((scheme, address.port()), ("http", 80) | ("https", 443));
    let authority = if default_port {
        host
    } else {
        format!("{host}:{}", address.port())
    };
    (format!("{scheme}://{authority}"), authority)
}

async fn idle_shutdown(activity: ActivityTracker, timeout: Duration) {
    let tick = timeout
        .min(Duration::from_secs(1))
        .max(Duration::from_millis(10));
    loop {
        tokio::time::sleep(tick).await;
        if !activity.has_open_stream() && activity.elapsed() >= timeout {
            return;
        }
    }
}

async fn shutdown_signal() {
    #[cfg(unix)]
    {
        let mut terminate =
            tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
                .expect("installing SIGTERM handler should succeed");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = terminate.recv() => {}
        }
    }

    #[cfg(windows)]
    {
        let mut ctrl_shutdown = tokio::signal::windows::ctrl_shutdown()
            .expect("installing Windows shutdown handler should succeed");
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = ctrl_shutdown.recv() => {}
        }
    }

    #[cfg(not(any(unix, windows)))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[derive(Debug)]
enum DashboardFailure {
    Input { code: &'static str, message: String },
    Router(RouterFailure),
    Inspection(nemo_relay_router::inspection::InspectionError),
    Token(TokenFileError),
    Auth,
    Bind(io::Error),
    Serve(io::Error),
    Output(io::Error),
    Runtime { code: &'static str, message: String },
}

impl DashboardFailure {
    fn input(code: &'static str, message: impl Into<String>) -> Self {
        Self::Input {
            code,
            message: message.into(),
        }
    }

    fn runtime(code: &'static str, message: impl Into<String>) -> Self {
        Self::Runtime {
            code,
            message: message.into(),
        }
    }

    fn code(&self) -> &'static str {
        match self {
            Self::Input { code, .. } | Self::Runtime { code, .. } => code,
            Self::Router(error) => error.code(),
            Self::Inspection(error) => error.code(),
            Self::Token(error) => error.code(),
            Self::Auth => "dashboard_authority_failed",
            Self::Bind(_) => "dashboard_bind_failed",
            Self::Serve(_) => "dashboard_server_failed",
            Self::Output(_) => "dashboard_output_failed",
        }
    }

    fn exit_code(&self) -> ExitCode {
        match self {
            Self::Input { .. } | Self::Token(_) => ExitCode::from(2),
            Self::Router(error) => error.exit_code(),
            Self::Inspection(_)
            | Self::Auth
            | Self::Bind(_)
            | Self::Serve(_)
            | Self::Output(_)
            | Self::Runtime { .. } => ExitCode::FAILURE,
        }
    }
}

impl std::fmt::Display for DashboardFailure {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input { message, .. } | Self::Runtime { message, .. } => {
                formatter.write_str(message)
            }
            Self::Router(error) => error.fmt(formatter),
            Self::Inspection(error) => {
                write!(formatter, "Router inspection failed: {}", error.code())
            }
            Self::Token(error) => error.fmt(formatter),
            Self::Auth => formatter.write_str("dashboard credential generation failed"),
            Self::Bind(error) => write!(formatter, "cannot bind dashboard listener: {error}"),
            Self::Serve(error) => write!(formatter, "dashboard listener failed: {error}"),
            Self::Output(error) => error.fmt(formatter),
        }
    }
}

#[cfg(test)]
#[path = "../tests/coverage/router_dashboard_tests.rs"]
mod tests;
