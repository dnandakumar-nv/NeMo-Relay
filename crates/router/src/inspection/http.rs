// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Optional bounded Axum adapter for the typed inspection service.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::fmt;
use std::io;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::body::Body;
use axum::extract::rejection::{JsonRejection, QueryRejection};
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::header::{
    AUTHORIZATION, CACHE_CONTROL, CONTENT_TYPE, ETAG, IF_NONE_MATCH, WWW_AUTHENTICATE,
};
use axum::http::{HeaderMap, HeaderValue, Method, Request, StatusCode};
use axum::middleware::{self, Next};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use futures_util::stream;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, watch};
use uuid::Uuid;
use zeroize::Zeroizing;

use super::{
    CohortRotationRequestV1, ContentPolicy, DecisionFilterV1, EvidenceExportRequestV1,
    EvidenceFilterV1, INSPECTION_INPUT_MAX_BYTES, INSPECTION_PAGE_LIMIT_DEFAULT,
    InspectionControlRequestV1, InspectionError, InspectionService, LearningResetRequestV1,
    NeighborhoodLookupV1, OutcomeFilterV1, PageRequest,
};

const OPERATIONS_PREFIX: &str = "/api/router/v1/operations";
const HTTP_HEADER_COUNT_MAX: usize = 64;
const HTTP_HEADER_BYTES_MAX: usize = 32 * 1024;
const HTTP_BEARER_TOKEN_BYTES_MAX: usize = 4 * 1024;
const HTTP_CONCURRENCY_MAX: usize = 32;
const HTTP_STREAM_CONCURRENCY_MAX: usize = 8;
const CONTENT_POLICY_HEADER: &str = "x-nemo-relay-content-policy";
const LAST_EVENT_ID_HEADER: &str = "last-event-id";
const DECISION_STREAM_POLL_INTERVAL: Duration = Duration::from_secs(1);
const DECISION_STREAM_HEARTBEAT_INTERVAL: Duration = Duration::from_secs(15);

/// Route composition selected by the host, independent of request input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InspectionHttpMode {
    /// Mount only typed read and export routes.
    ReadOnly,
    /// Mount reads plus explicitly operations-capable mutation routes.
    Operations,
}

/// Bounded request metadata presented to a host authentication guard.
pub struct InspectionHttpAuthContext<'a> {
    /// HTTP method.
    pub method: &'a Method,
    /// Exact versioned request path.
    pub path: &'a str,
    /// Parsed request headers within the adapter's aggregate limit.
    pub headers: &'a HeaderMap,
    /// Whether the request targets the operations namespace.
    pub operations: bool,
}

/// Synchronous host authentication and authorization guard.
pub trait InspectionHttpAuthGuard: Send + Sync + 'static {
    /// Authorize one already-bounded request before any handler work starts.
    fn authorize(&self, context: &InspectionHttpAuthContext<'_>) -> Result<(), InspectionError>;
}

/// Generic exact bearer-token guard with constant-time digest comparison.
#[derive(Clone)]
pub struct BearerTokenAuthGuard {
    expected_sha256: Arc<Zeroizing<[u8; 32]>>,
}

impl BearerTokenAuthGuard {
    /// Freeze one nonblank visible-ASCII bearer token.
    pub fn new(token: impl AsRef<str>) -> Result<Self, InspectionError> {
        let token = token.as_ref().as_bytes();
        if token.is_empty()
            || token.len() > HTTP_BEARER_TOKEN_BYTES_MAX
            || !token.iter().all(|byte| (0x21..=0x7e).contains(byte))
        {
            return Err(InspectionError::InvalidArgument);
        }
        let mut expected = Zeroizing::new([0_u8; 32]);
        expected.copy_from_slice(&Sha256::digest(token));
        Ok(Self {
            expected_sha256: Arc::new(expected),
        })
    }
}

impl InspectionHttpAuthGuard for BearerTokenAuthGuard {
    fn authorize(&self, context: &InspectionHttpAuthContext<'_>) -> Result<(), InspectionError> {
        let mut values = context.headers.get_all(AUTHORIZATION).iter();
        let value = values.next().ok_or(InspectionError::Unauthorized)?;
        if values.next().is_some() {
            return Err(InspectionError::Unauthorized);
        }
        let value = value.to_str().map_err(|_| InspectionError::Unauthorized)?;
        let token = value
            .strip_prefix("Bearer ")
            .filter(|token| {
                !token.is_empty()
                    && token.len() <= HTTP_BEARER_TOKEN_BYTES_MAX
                    && token.bytes().all(|byte| (0x21..=0x7e).contains(&byte))
            })
            .ok_or(InspectionError::Unauthorized)?;
        let actual = Sha256::digest(token.as_bytes());
        if bool::from((**self.expected_sha256).ct_eq(actual.as_ref())) {
            Ok(())
        } else {
            Err(InspectionError::Unauthorized)
        }
    }
}

impl fmt::Debug for BearerTokenAuthGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BearerTokenAuthGuard")
            .finish_non_exhaustive()
    }
}

/// Build the complete version-1 inspection router under `/api/router/v1`.
pub fn inspection_http_router(
    service: InspectionService,
    auth: Arc<dyn InspectionHttpAuthGuard>,
    mode: InspectionHttpMode,
) -> Result<Router, InspectionError> {
    if mode == InspectionHttpMode::Operations && !service.operations_enabled() {
        return Err(InspectionError::Forbidden);
    }
    let state = HttpState {
        timeout: Duration::from_millis(service.request_timeout_ms()),
        service,
        auth,
        concurrency: Arc::new(Semaphore::new(HTTP_CONCURRENCY_MAX)),
        stream_concurrency: Arc::new(Semaphore::new(HTTP_STREAM_CONCURRENCY_MAX)),
    };
    let mut routes = Router::<HttpState>::new()
        .route("/api/router/v1/status", get(status))
        .route("/api/router/v1/overview", get(overview))
        .route("/api/router/v1/pools", get(pools))
        .route("/api/router/v1/pools/{pool_id}", get(pool_detail))
        .route("/api/router/v1/evidence", get(evidence))
        .route("/api/router/v1/evidence/export", post(evidence_export))
        .route(
            "/api/router/v1/evidence/{evidence_id}",
            get(evidence_detail),
        )
        .route("/api/router/v1/neighborhood", post(neighborhood))
        .route("/api/router/v1/decisions", get(decisions))
        .route("/api/router/v1/decisions/tail", get(decision_tail))
        .route("/api/router/v1/decisions/stream", get(decision_stream))
        .route(
            "/api/router/v1/decisions/{decision_id}/exposure",
            get(decision_exposure),
        )
        .route(
            "/api/router/v1/decisions/{decision_id}",
            get(decision_detail),
        )
        .route("/api/router/v1/outcomes", get(outcomes))
        .route("/api/router/v1/controls", get(controls))
        .route("/api/router/v1/health", get(health))
        .route("/api/router/v1/migrations", get(migrations));
    if mode == InspectionHttpMode::Operations {
        routes = routes
            .route("/api/router/v1/operations/control", post(apply_control))
            .route("/api/router/v1/operations/reset", post(reset))
            .route(
                "/api/router/v1/operations/cohort/rotate",
                post(rotate_cohort),
            );
    }
    Ok(routes
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(INSPECTION_INPUT_MAX_BYTES))
        .layer(middleware::from_fn_with_state(
            state.clone(),
            enforce_request_boundary,
        ))
        .with_state(state))
}

#[derive(Clone)]
struct HttpState {
    service: InspectionService,
    auth: Arc<dyn InspectionHttpAuthGuard>,
    timeout: Duration,
    concurrency: Arc<Semaphore>,
    stream_concurrency: Arc<Semaphore>,
}

async fn enforce_request_boundary(
    State(state): State<HttpState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let deadline = match Instant::now().checked_add(state.timeout) {
        Some(deadline) => deadline,
        None => return HttpFailure::new(InspectionError::Busy).into_response(),
    };
    if !headers_are_bounded(request.headers()) {
        return HttpFailure::new(InspectionError::InvalidArgument).into_response();
    }
    if let Err(error) = require_content_policy(request.headers(), state.service.content_policy()) {
        return HttpFailure::new(error).into_response();
    }
    let path = request.uri().path();
    let context = InspectionHttpAuthContext {
        method: request.method(),
        path,
        headers: request.headers(),
        operations: path.starts_with(OPERATIONS_PREFIX),
    };
    if let Err(error) = state.auth.authorize(&context) {
        let error = match error {
            InspectionError::Unauthorized | InspectionError::Forbidden => error,
            _ => InspectionError::Forbidden,
        };
        return HttpFailure::new(error).into_response();
    }
    if Instant::now() >= deadline {
        return HttpFailure::new(InspectionError::Busy).into_response();
    }
    let permit = match tokio::time::timeout_at(
        tokio::time::Instant::from_std(deadline),
        state.concurrency.clone().acquire_owned(),
    )
    .await
    {
        Ok(Ok(permit)) => permit,
        Ok(Err(_)) => return HttpFailure::new(InspectionError::StorageUnavailable).into_response(),
        Err(_) => return HttpFailure::new(InspectionError::Busy).into_response(),
    };
    let response =
        tokio::time::timeout_at(tokio::time::Instant::from_std(deadline), next.run(request)).await;
    drop(permit);
    match response {
        Ok(mut response) => {
            add_no_store(response.headers_mut());
            response
        }
        Err(_) => HttpFailure::new(InspectionError::Busy).into_response(),
    }
}

fn headers_are_bounded(headers: &HeaderMap) -> bool {
    if headers.len() > HTTP_HEADER_COUNT_MAX {
        return false;
    }
    headers
        .iter()
        .try_fold(0_usize, |total, (name, value)| {
            total
                .checked_add(name.as_str().len())?
                .checked_add(value.as_bytes().len())
        })
        .is_some_and(|total| total <= HTTP_HEADER_BYTES_MAX)
}

fn require_content_policy(
    headers: &HeaderMap,
    granted: ContentPolicy,
) -> Result<(), InspectionError> {
    let mut values = headers.get_all(CONTENT_POLICY_HEADER).iter();
    let Some(value) = values.next() else {
        return Ok(());
    };
    if values.next().is_some() {
        return Err(InspectionError::InvalidArgument);
    }
    let required = match value.to_str().ok() {
        Some("redacted") => ContentPolicy::Redacted,
        Some("full") => ContentPolicy::Full,
        _ => return Err(InspectionError::InvalidArgument),
    };
    if required == granted {
        Ok(())
    } else {
        Err(InspectionError::Forbidden)
    }
}

async fn status(State(state): State<HttpState>) -> HttpResult<impl IntoResponse> {
    Ok(Json(state.service.status().await?))
}

async fn overview(State(state): State<HttpState>, headers: HeaderMap) -> HttpResult<Response> {
    let report = state.service.overview().await?;
    etagged_json(&headers, report, "overview", SemanticEtagShape::Overview)
}

async fn pools(
    State(state): State<HttpState>,
    headers: HeaderMap,
    query: Result<Query<PageQuery>, QueryRejection>,
) -> HttpResult<Response> {
    let page = state.service.list_pools(parse_query(query)?.page()).await?;
    etagged_json(&headers, page, "pools", SemanticEtagShape::Page)
}

async fn pool_detail(
    State(state): State<HttpState>,
    Path(pool_id): Path<String>,
) -> HttpResult<impl IntoResponse> {
    Ok(Json(state.service.get_pool(&pool_id).await?))
}

async fn controls(
    State(state): State<HttpState>,
    headers: HeaderMap,
    query: Result<Query<PageQuery>, QueryRejection>,
) -> HttpResult<Response> {
    let page = state
        .service
        .list_controls(parse_query(query)?.page())
        .await?;
    etagged_json(&headers, page, "controls", SemanticEtagShape::Page)
}

async fn health(
    State(state): State<HttpState>,
    headers: HeaderMap,
    query: Result<Query<PageQuery>, QueryRejection>,
) -> HttpResult<Response> {
    let page = state
        .service
        .list_health(parse_query(query)?.page())
        .await?;
    etagged_json(&headers, page, "health", SemanticEtagShape::Page)
}

async fn migrations(
    State(state): State<HttpState>,
    headers: HeaderMap,
    query: Result<Query<PageQuery>, QueryRejection>,
) -> HttpResult<Response> {
    let page = state
        .service
        .list_migrations(parse_query(query)?.page())
        .await?;
    etagged_json(&headers, page, "migrations", SemanticEtagShape::Page)
}

async fn evidence(
    State(state): State<HttpState>,
    query: Result<Query<EvidenceQuery>, QueryRejection>,
) -> HttpResult<impl IntoResponse> {
    let query = parse_query(query)?;
    Ok(Json(
        state
            .service
            .list_evidence(query.filter(), query.page())
            .await?,
    ))
}

async fn evidence_detail(
    State(state): State<HttpState>,
    Path(evidence_id): Path<String>,
) -> HttpResult<impl IntoResponse> {
    Ok(Json(
        state
            .service
            .get_evidence(parse_uuid(&evidence_id)?)
            .await?,
    ))
}

async fn decisions(
    State(state): State<HttpState>,
    query: Result<Query<DecisionQuery>, QueryRejection>,
) -> HttpResult<impl IntoResponse> {
    let query = parse_query(query)?;
    Ok(Json(
        state
            .service
            .list_decisions(query.filter(), query.page())
            .await?,
    ))
}

async fn decision_tail(
    State(state): State<HttpState>,
    query: Result<Query<DecisionQuery>, QueryRejection>,
) -> HttpResult<impl IntoResponse> {
    let query = parse_query(query)?;
    Ok(Json(
        state
            .service
            .tail_decisions(query.filter(), query.page())
            .await?,
    ))
}

async fn decision_stream(
    State(state): State<HttpState>,
    headers: HeaderMap,
    query: Result<Query<DecisionQuery>, QueryRejection>,
) -> HttpResult<Response> {
    let query = parse_query(query)?;
    let filter = query.filter();
    let mut page = query.page();
    if let Some(last_event_id) = parse_last_event_id(&headers)? {
        page.after = Some(last_event_id);
    }
    filter.validate()?;
    page.validate()?;

    let permit = state
        .stream_concurrency
        .clone()
        .try_acquire_owned()
        .map_err(|_| HttpFailure::new(InspectionError::Busy))?;
    let initial = state
        .service
        .tail_decisions(filter.clone(), page.clone())
        .await?;
    let stream_state = DecisionStreamState::new(state.service, filter, page, initial, permit)?;
    let events = stream::unfold(stream_state, next_decision_stream_event);
    Ok(Sse::new(events)
        .keep_alive(
            KeepAlive::new()
                .interval(DECISION_STREAM_HEARTBEAT_INTERVAL)
                .text("heartbeat"),
        )
        .into_response())
}

async fn decision_detail(
    State(state): State<HttpState>,
    Path(decision_id): Path<String>,
) -> HttpResult<impl IntoResponse> {
    Ok(Json(
        state
            .service
            .get_decision(parse_uuid(&decision_id)?)
            .await?,
    ))
}

async fn decision_exposure(
    State(state): State<HttpState>,
    Path(decision_id): Path<String>,
) -> HttpResult<impl IntoResponse> {
    Ok(Json(
        state
            .service
            .get_decision_exposure(parse_uuid(&decision_id)?)
            .await?,
    ))
}

async fn outcomes(
    State(state): State<HttpState>,
    query: Result<Query<OutcomeQuery>, QueryRejection>,
) -> HttpResult<impl IntoResponse> {
    let query = parse_query(query)?;
    Ok(Json(
        state
            .service
            .list_outcomes(query.filter(), query.page())
            .await?,
    ))
}

async fn neighborhood(
    State(state): State<HttpState>,
    body: Result<Json<NeighborhoodLookupV1>, JsonRejection>,
) -> HttpResult<impl IntoResponse> {
    Ok(Json(
        state
            .service
            .inspect_neighborhood(parse_json(body)?)
            .await?,
    ))
}

async fn evidence_export(
    State(state): State<HttpState>,
    body: Result<Json<EvidenceExportRequestV1>, JsonRejection>,
) -> HttpResult<Response> {
    let export = state.service.export_evidence(parse_json(body)?).await?;
    let content_type = export.content_type();
    let body = Body::from_stream(stream::unfold(export, |mut export| async move {
        export.next_chunk().await.map(|chunk| {
            let chunk = chunk
                .map(axum::body::Bytes::from)
                .map_err(|error| io::Error::other(error.code()));
            (chunk, export)
        })
    }));
    Response::builder()
        .status(StatusCode::OK)
        .header(CONTENT_TYPE, content_type)
        .body(body)
        .map_err(|_| HttpFailure::new(InspectionError::IntegrityError))
}

async fn apply_control(
    State(state): State<HttpState>,
    body: Result<Json<InspectionControlRequestV1>, JsonRejection>,
) -> HttpResult<impl IntoResponse> {
    Ok(Json(state.service.apply_control(parse_json(body)?).await?))
}

async fn reset(
    State(state): State<HttpState>,
    body: Result<Json<LearningResetRequestV1>, JsonRejection>,
) -> HttpResult<impl IntoResponse> {
    Ok(Json(state.service.reset(parse_json(body)?).await?))
}

async fn rotate_cohort(
    State(state): State<HttpState>,
    body: Result<Json<CohortRotationRequestV1>, JsonRejection>,
) -> HttpResult<impl IntoResponse> {
    Ok(Json(state.service.rotate_cohort(parse_json(body)?).await?))
}

async fn not_found() -> HttpFailure {
    HttpFailure::new(InspectionError::NotFound)
}

type HttpResult<T> = Result<T, HttpFailure>;

fn parse_query<T>(query: Result<Query<T>, QueryRejection>) -> HttpResult<T> {
    query
        .map(|Query(value)| value)
        .map_err(|_| HttpFailure::new(InspectionError::InvalidArgument))
}

fn parse_json<T>(body: Result<Json<T>, JsonRejection>) -> HttpResult<T> {
    body.map(|Json(value)| value).map_err(|rejection| {
        let mut failure = HttpFailure::new(InspectionError::InvalidArgument);
        if rejection.status() == StatusCode::PAYLOAD_TOO_LARGE {
            failure.status = StatusCode::PAYLOAD_TOO_LARGE;
        }
        failure
    })
}

fn parse_uuid(value: &str) -> HttpResult<Uuid> {
    Uuid::parse_str(value).map_err(|_| HttpFailure::new(InspectionError::InvalidArgument))
}

fn parse_last_event_id(headers: &HeaderMap) -> HttpResult<Option<String>> {
    let mut values = headers.get_all(LAST_EVENT_ID_HEADER).iter();
    let Some(value) = values.next() else {
        return Ok(None);
    };
    if values.next().is_some() {
        return Err(HttpFailure::new(InspectionError::InvalidArgument));
    }
    let value = value
        .to_str()
        .map_err(|_| HttpFailure::new(InspectionError::InvalidArgument))?;
    if value.is_empty() {
        return Ok(None);
    }
    Ok(Some(value.to_string()))
}

#[derive(Debug, Clone, Copy)]
enum SemanticEtagShape {
    Overview,
    Page,
}

fn etagged_json<T: Serialize>(
    request_headers: &HeaderMap,
    value: T,
    resource: &'static str,
    shape: SemanticEtagShape,
) -> HttpResult<Response> {
    let etag = semantic_etag(&value, resource, shape)?;
    if if_none_match(request_headers, etag.to_str().unwrap_or_default())? {
        let mut response = StatusCode::NOT_MODIFIED.into_response();
        response.headers_mut().insert(ETAG, etag);
        return Ok(response);
    }
    let mut response = Json(value).into_response();
    response.headers_mut().insert(ETAG, etag);
    Ok(response)
}

fn semantic_etag<T: Serialize>(
    value: &T,
    resource: &'static str,
    shape: SemanticEtagShape,
) -> HttpResult<HeaderValue> {
    let mut semantic = serde_json::to_value(value)
        .map_err(|_| HttpFailure::new(InspectionError::IntegrityError))?;
    let root = semantic
        .as_object_mut()
        .ok_or_else(|| HttpFailure::new(InspectionError::IntegrityError))?;
    match shape {
        SemanticEtagShape::Overview => {
            remove_semantic_field(root, "window_start_unix_ms")?;
            remove_semantic_field(root, "snapshot_time_unix_ms")?;
            let status = root
                .get_mut("status")
                .and_then(serde_json::Value::as_object_mut)
                .ok_or_else(|| HttpFailure::new(InspectionError::IntegrityError))?;
            remove_semantic_field(status, "snapshot_time_unix_ms")?;
        }
        SemanticEtagShape::Page => {
            remove_semantic_field(root, "next")?;
            remove_semantic_field(root, "snapshot_time_unix_ms")?;
        }
    }
    let bytes = serde_json::to_vec(&semantic)
        .map_err(|_| HttpFailure::new(InspectionError::IntegrityError))?;
    let mut digest = Sha256::new();
    digest.update(b"nemo.relay.router.semantic-etag@1\0");
    digest.update(resource.as_bytes());
    digest.update(b"\0");
    digest.update(bytes);
    let hash = digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    HeaderValue::from_str(&format!("W/\"{hash}\""))
        .map_err(|_| HttpFailure::new(InspectionError::IntegrityError))
}

fn remove_semantic_field(
    object: &mut serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> HttpResult<()> {
    object
        .remove(field)
        .map(|_| ())
        .ok_or_else(|| HttpFailure::new(InspectionError::IntegrityError))
}

fn if_none_match(headers: &HeaderMap, expected: &str) -> HttpResult<bool> {
    let expected =
        opaque_etag(expected).ok_or_else(|| HttpFailure::new(InspectionError::IntegrityError))?;
    for value in headers.get_all(IF_NONE_MATCH).iter() {
        let value = value
            .to_str()
            .map_err(|_| HttpFailure::new(InspectionError::InvalidArgument))?;
        for candidate in value.split(',').map(str::trim) {
            if candidate == "*" || opaque_etag(candidate) == Some(expected) {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

fn opaque_etag(value: &str) -> Option<&str> {
    let value = value.strip_prefix("W/").unwrap_or(value);
    let value = value.strip_prefix('"')?.strip_suffix('"')?;
    (!value.is_empty()
        && value
            .bytes()
            .all(|byte| byte == 0x21 || (0x23..=0x7e).contains(&byte)))
    .then_some(value)
}

#[derive(Serialize)]
struct DecisionStreamCursorV1<'a> {
    cursor: &'a str,
}

struct DecisionStreamState {
    service: InspectionService,
    filter: DecisionFilterV1,
    page: PageRequest,
    pending: VecDeque<Event>,
    cancellation: watch::Receiver<bool>,
    poll_after_pending: bool,
    stopped: bool,
    _permit: OwnedSemaphorePermit,
}

impl DecisionStreamState {
    fn new(
        service: InspectionService,
        filter: DecisionFilterV1,
        page: PageRequest,
        initial: super::Page<super::DecisionSummaryV1>,
        permit: OwnedSemaphorePermit,
    ) -> Result<Self, HttpFailure> {
        let cancellation = service.cancellation_receiver();
        let mut state = Self {
            service,
            filter,
            page,
            pending: VecDeque::new(),
            cancellation,
            poll_after_pending: false,
            stopped: false,
            _permit: permit,
        };
        state.queue_page(initial)?;
        Ok(state)
    }

    fn queue_page(
        &mut self,
        page: super::Page<super::DecisionSummaryV1>,
    ) -> Result<(), HttpFailure> {
        let prior_cursor = self.page.after.clone();
        let item_count = page.items.len();
        for decision in page.items {
            let event = Event::default()
                .event("decision")
                .json_data(decision)
                .map_err(|_| HttpFailure::new(InspectionError::IntegrityError))?;
            self.pending.push_back(event);
        }
        if page.next != prior_cursor
            && let Some(cursor) = page.next.as_deref()
        {
            let event = Event::default()
                .event("cursor")
                .id(cursor)
                .json_data(DecisionStreamCursorV1 { cursor })
                .map_err(|_| HttpFailure::new(InspectionError::IntegrityError))?;
            self.pending.push_back(event);
        }
        self.page.after = page.next;
        self.poll_after_pending = item_count < usize::from(self.page.limit);
        Ok(())
    }

    fn queue_error(&mut self, error: InspectionError) {
        let event = Event::default().event("error").json_data(ErrorEnvelope {
            error: ErrorBody { code: error.code() },
        });
        if let Ok(event) = event {
            self.pending.push_back(event);
        }
        self.stopped = true;
    }
}

async fn next_decision_stream_event(
    mut state: DecisionStreamState,
) -> Option<(Result<Event, Infallible>, DecisionStreamState)> {
    loop {
        if let Some(event) = state.pending.pop_front() {
            return Some((Ok(event), state));
        }
        if state.stopped || *state.cancellation.borrow() {
            return None;
        }
        if state.poll_after_pending {
            tokio::select! {
                () = tokio::time::sleep(DECISION_STREAM_POLL_INTERVAL) => {}
                changed = state.cancellation.changed() => {
                    if changed.is_err() || *state.cancellation.borrow() {
                        return None;
                    }
                }
            }
        }
        state.poll_after_pending = false;
        match state
            .service
            .tail_decisions(state.filter.clone(), state.page.clone())
            .await
        {
            Ok(page) => {
                if state.queue_page(page).is_err() {
                    state.queue_error(InspectionError::IntegrityError);
                }
            }
            Err(error) => state.queue_error(error),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct PageQuery {
    limit: Option<u16>,
    after: Option<String>,
}

impl PageQuery {
    fn page(&self) -> PageRequest {
        PageRequest {
            limit: self.limit.unwrap_or(INSPECTION_PAGE_LIMIT_DEFAULT),
            after: self.after.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct EvidenceQuery {
    limit: Option<u16>,
    after: Option<String>,
    pool_id: Option<String>,
    candidate_id: Option<String>,
    terminal_class: Option<String>,
    quality_label: Option<String>,
    learning_generation_id: Option<Uuid>,
}

impl EvidenceQuery {
    fn page(&self) -> PageRequest {
        PageRequest {
            limit: self.limit.unwrap_or(INSPECTION_PAGE_LIMIT_DEFAULT),
            after: self.after.clone(),
        }
    }

    fn filter(&self) -> EvidenceFilterV1 {
        EvidenceFilterV1 {
            pool_id: self.pool_id.clone(),
            candidate_id: self.candidate_id.clone(),
            terminal_class: self.terminal_class.clone(),
            quality_label: self.quality_label.clone(),
            learning_generation_id: self.learning_generation_id,
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct DecisionQuery {
    limit: Option<u16>,
    after: Option<String>,
    pool_id: Option<String>,
    mode: Option<String>,
    candidate_id: Option<String>,
    final_reason: Option<String>,
}

impl DecisionQuery {
    fn page(&self) -> PageRequest {
        PageRequest {
            limit: self.limit.unwrap_or(INSPECTION_PAGE_LIMIT_DEFAULT),
            after: self.after.clone(),
        }
    }

    fn filter(&self) -> DecisionFilterV1 {
        DecisionFilterV1 {
            pool_id: self.pool_id.clone(),
            mode: self.mode.clone(),
            candidate_id: self.candidate_id.clone(),
            final_reason: self.final_reason.clone(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct OutcomeQuery {
    limit: Option<u16>,
    after: Option<String>,
    pool_id: Option<String>,
    arm: Option<String>,
    label: Option<String>,
    attribution_status: Option<String>,
}

impl OutcomeQuery {
    fn page(&self) -> PageRequest {
        PageRequest {
            limit: self.limit.unwrap_or(INSPECTION_PAGE_LIMIT_DEFAULT),
            after: self.after.clone(),
        }
    }

    fn filter(&self) -> OutcomeFilterV1 {
        OutcomeFilterV1 {
            pool_id: self.pool_id.clone(),
            arm: self.arm.clone(),
            label: self.label.clone(),
            attribution_status: self.attribution_status.clone(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct HttpFailure {
    error: InspectionError,
    status: StatusCode,
}

impl HttpFailure {
    fn new(error: InspectionError) -> Self {
        Self {
            error,
            status: status_for_error(error),
        }
    }
}

impl From<InspectionError> for HttpFailure {
    fn from(error: InspectionError) -> Self {
        Self::new(error)
    }
}

#[derive(Serialize)]
struct ErrorEnvelope {
    error: ErrorBody,
}

#[derive(Serialize)]
struct ErrorBody {
    code: &'static str,
}

impl IntoResponse for HttpFailure {
    fn into_response(self) -> Response {
        let mut response = (
            self.status,
            Json(ErrorEnvelope {
                error: ErrorBody {
                    code: self.error.code(),
                },
            }),
        )
            .into_response();
        if self.error == InspectionError::Unauthorized {
            response
                .headers_mut()
                .insert(WWW_AUTHENTICATE, HeaderValue::from_static("Bearer"));
        }
        add_no_store(response.headers_mut());
        response
    }
}

fn status_for_error(error: InspectionError) -> StatusCode {
    match error {
        InspectionError::InvalidArgument | InspectionError::InvalidCursor => {
            StatusCode::BAD_REQUEST
        }
        InspectionError::NotFound => StatusCode::NOT_FOUND,
        InspectionError::Unauthorized => StatusCode::UNAUTHORIZED,
        InspectionError::Forbidden | InspectionError::EgressDenied => StatusCode::FORBIDDEN,
        InspectionError::Conflict | InspectionError::MigrationRequired => StatusCode::CONFLICT,
        InspectionError::Busy | InspectionError::StorageUnavailable => {
            StatusCode::SERVICE_UNAVAILABLE
        }
        InspectionError::MutationExpired => StatusCode::GONE,
        InspectionError::CapacityExhausted => StatusCode::INSUFFICIENT_STORAGE,
        InspectionError::IntegrityError => StatusCode::INTERNAL_SERVER_ERROR,
        InspectionError::IncompatibleApi => StatusCode::UPGRADE_REQUIRED,
    }
}

fn add_no_store(headers: &mut HeaderMap) {
    headers.insert(CACHE_CONTROL, HeaderValue::from_static("no-store"));
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::thread;

    use axum::http::header::HeaderName;
    use http_body_util::BodyExt as _;
    use serde_json::{Value, json};
    use tempfile::{TempDir, tempdir};
    use tower::ServiceExt as _;

    use super::*;
    use crate::config::RouterConfig;
    use crate::control::{ControlOperation, ControlScope, RouterControlError};
    use crate::inspection::{
        DecisionExposureV1, DecisionSummaryV1, InspectionServiceOptions, OverviewReportV1, Page,
        PoolDetailV1,
    };
    use crate::ledger::repository::decision::{DecisionAuditAck, tests as decision_test_fixtures};
    use crate::ledger::repository::process::{LedgerHealthEvent, LedgerHealthSeverity};
    use crate::ledger::repository::{LedgerRepository, ready_evaluated_runtime_fixture};

    const TEST_TOKEN: &str = "inspection-test-token";

    fn config(path: &Path, project_id: &str) -> RouterConfig {
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

    async fn missing_service(
        options: InspectionServiceOptions,
        project_id: &str,
    ) -> (TempDir, InspectionService) {
        let temporary = tempdir().unwrap();
        let service = InspectionService::open(
            config(&temporary.path().join("missing.db"), project_id),
            options,
        )
        .await
        .unwrap();
        (temporary, service)
    }

    fn bearer_guard() -> Arc<dyn InspectionHttpAuthGuard> {
        Arc::new(BearerTokenAuthGuard::new(TEST_TOKEN).unwrap())
    }

    fn request(
        method: Method,
        uri: &str,
        token: Option<&str>,
        body: impl Into<Body>,
        has_json_body: bool,
    ) -> Request<Body> {
        let mut builder = Request::builder().method(method).uri(uri);
        if let Some(token) = token {
            builder = builder.header(AUTHORIZATION, format!("Bearer {token}"));
        }
        if has_json_body {
            builder = builder.header(CONTENT_TYPE, "application/json");
        }
        builder.body(body.into()).unwrap()
    }

    fn get(uri: &str, token: Option<&str>) -> Request<Body> {
        request(Method::GET, uri, token, Body::empty(), false)
    }

    fn conditional_get(uri: &str, token: Option<&str>, etag: &HeaderValue) -> Request<Body> {
        let mut request = get(uri, token);
        request.headers_mut().insert(IF_NONE_MATCH, etag.clone());
        request
    }

    fn post_json(uri: &str, token: Option<&str>, value: &impl Serialize) -> Request<Body> {
        request(
            Method::POST,
            uri,
            token,
            serde_json::to_vec(value).unwrap(),
            true,
        )
    }

    async fn response_json(response: Response) -> Value {
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    fn assert_no_store(response: &Response) {
        assert_eq!(
            response.headers().get(CACHE_CONTROL),
            Some(&HeaderValue::from_static("no-store"))
        );
    }

    async fn assert_error(response: Response, status: StatusCode, code: &str) {
        assert_eq!(response.status(), status);
        assert_no_store(&response);
        assert_eq!(response_json(response).await["error"]["code"], code);
    }

    async fn sse_json_frame(body: &mut Body, expected_event: &str) -> Value {
        let frame = tokio::time::timeout(Duration::from_secs(2), body.frame())
            .await
            .expect("SSE frame should arrive before the test deadline")
            .expect("SSE stream should remain open")
            .expect("SSE frame should be valid");
        let data = std::str::from_utf8(frame.data_ref().expect("SSE should emit data frames"))
            .expect("SSE should be UTF-8");
        assert!(
            data.lines()
                .any(|line| line == format!("event: {expected_event}")),
            "unexpected SSE frame: {data:?}"
        );
        let payload = data
            .lines()
            .find_map(|line| line.strip_prefix("data: "))
            .expect("SSE event should contain JSON data");
        serde_json::from_str(payload).expect("SSE data should be JSON")
    }

    struct DenyHostGuard;

    impl InspectionHttpAuthGuard for DenyHostGuard {
        fn authorize(
            &self,
            context: &InspectionHttpAuthContext<'_>,
        ) -> Result<(), InspectionError> {
            assert_eq!(context.method, Method::GET);
            assert_eq!(context.path, "/api/router/v1/status");
            assert!(!context.operations);
            Err(InspectionError::Forbidden)
        }
    }

    struct SlowHostGuard;

    impl InspectionHttpAuthGuard for SlowHostGuard {
        fn authorize(
            &self,
            _context: &InspectionHttpAuthContext<'_>,
        ) -> Result<(), InspectionError> {
            thread::sleep(Duration::from_millis(10));
            Ok(())
        }
    }

    #[tokio::test]
    async fn bearer_and_host_guards_gate_every_response_without_caching() {
        assert_eq!(
            BearerTokenAuthGuard::new("").unwrap_err(),
            InspectionError::InvalidArgument
        );
        assert_eq!(
            BearerTokenAuthGuard::new("contains space").unwrap_err(),
            InspectionError::InvalidArgument
        );

        let (_temporary, service) =
            missing_service(InspectionServiceOptions::default(), "http-auth").await;
        let app = inspection_http_router(
            service.clone(),
            bearer_guard(),
            InspectionHttpMode::ReadOnly,
        )
        .unwrap();

        let missing = app
            .clone()
            .oneshot(get("/api/router/v1/status", None))
            .await
            .unwrap();
        assert_eq!(
            missing.headers().get(WWW_AUTHENTICATE),
            Some(&HeaderValue::from_static("Bearer"))
        );
        assert_error(missing, StatusCode::UNAUTHORIZED, "unauthorized").await;

        let wrong = app
            .clone()
            .oneshot(get("/api/router/v1/status", Some("wrong")))
            .await
            .unwrap();
        assert_error(wrong, StatusCode::UNAUTHORIZED, "unauthorized").await;

        let correct = app
            .oneshot(get("/api/router/v1/status", Some(TEST_TOKEN)))
            .await
            .unwrap();
        assert_eq!(correct.status(), StatusCode::OK);
        assert_no_store(&correct);

        let denied = inspection_http_router(
            service.clone(),
            Arc::new(DenyHostGuard),
            InspectionHttpMode::ReadOnly,
        )
        .unwrap()
        .oneshot(get("/api/router/v1/status", None))
        .await
        .unwrap();
        assert_error(denied, StatusCode::FORBIDDEN, "forbidden").await;
        service.close().await.unwrap();
    }

    #[tokio::test]
    async fn readonly_composition_physically_omits_operation_routes() {
        let (_temporary, service) =
            missing_service(InspectionServiceOptions::default(), "http-readonly").await;
        assert_eq!(
            inspection_http_router(
                service.clone(),
                bearer_guard(),
                InspectionHttpMode::Operations,
            )
            .err(),
            Some(InspectionError::Forbidden)
        );

        let response = inspection_http_router(
            service.clone(),
            bearer_guard(),
            InspectionHttpMode::ReadOnly,
        )
        .unwrap()
        .oneshot(post_json(
            "/api/router/v1/operations/control",
            Some(TEST_TOKEN),
            &json!({}),
        ))
        .await
        .unwrap();
        assert_error(response, StatusCode::NOT_FOUND, "not_found").await;
        service.close().await.unwrap();
    }

    #[tokio::test]
    async fn request_boundaries_reject_capability_page_header_and_body_overruns() {
        let (_temporary, service) =
            missing_service(InspectionServiceOptions::default(), "http-limits").await;
        let app = inspection_http_router(
            service.clone(),
            bearer_guard(),
            InspectionHttpMode::ReadOnly,
        )
        .unwrap();

        let mut full = get("/api/router/v1/status", Some(TEST_TOKEN));
        full.headers_mut().insert(
            HeaderName::from_static(CONTENT_POLICY_HEADER),
            HeaderValue::from_static("full"),
        );
        assert_error(
            app.clone().oneshot(full).await.unwrap(),
            StatusCode::FORBIDDEN,
            "forbidden",
        )
        .await;

        assert_error(
            app.clone()
                .oneshot(get("/api/router/v1/pools?limit=501", Some(TEST_TOKEN)))
                .await
                .unwrap(),
            StatusCode::BAD_REQUEST,
            "invalid_argument",
        )
        .await;

        let mut too_many_headers = get("/api/router/v1/status", Some(TEST_TOKEN));
        for index in 0..HTTP_HEADER_COUNT_MAX {
            let name = HeaderName::from_bytes(format!("x-bound-{index}").as_bytes()).unwrap();
            too_many_headers
                .headers_mut()
                .insert(name, HeaderValue::from_static("1"));
        }
        assert_error(
            app.clone().oneshot(too_many_headers).await.unwrap(),
            StatusCode::BAD_REQUEST,
            "invalid_argument",
        )
        .await;

        let mut oversized_header = get("/api/router/v1/status", Some(TEST_TOKEN));
        oversized_header.headers_mut().insert(
            HeaderName::from_static("x-oversized"),
            HeaderValue::from_bytes(&vec![b'a'; HTTP_HEADER_BYTES_MAX]).unwrap(),
        );
        assert_error(
            app.clone().oneshot(oversized_header).await.unwrap(),
            StatusCode::BAD_REQUEST,
            "invalid_argument",
        )
        .await;

        let oversized_body = request(
            Method::POST,
            "/api/router/v1/neighborhood",
            Some(TEST_TOKEN),
            vec![b' '; INSPECTION_INPUT_MAX_BYTES + 1],
            true,
        );
        assert_error(
            app.oneshot(oversized_body).await.unwrap(),
            StatusCode::PAYLOAD_TOO_LARGE,
            "invalid_argument",
        )
        .await;
        service.close().await.unwrap();
    }

    #[tokio::test]
    async fn dashboard_read_routes_preserve_strict_contracts_and_static_precedence() {
        let (_temporary, config, mut activated) = decision_test_fixtures::activate(0.0, 10);
        let audit = decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        let decision_id = audit.parent.decision_id;
        assert_eq!(
            activated
                .repository
                .record_decision_audit(&audit, 10, Uuid::now_v7())
                .unwrap(),
            DecisionAuditAck::Applied
        );
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        let app = inspection_http_router(
            service.clone(),
            bearer_guard(),
            InspectionHttpMode::ReadOnly,
        )
        .unwrap();

        let overview = app
            .clone()
            .oneshot(get("/api/router/v1/overview", Some(TEST_TOKEN)))
            .await
            .unwrap();
        assert_eq!(overview.status(), StatusCode::OK);
        assert!(serde_json::from_value::<OverviewReportV1>(response_json(overview).await).is_ok());

        let pool = app
            .clone()
            .oneshot(get("/api/router/v1/pools/pool-a", Some(TEST_TOKEN)))
            .await
            .unwrap();
        assert_eq!(pool.status(), StatusCode::OK);
        let pool: PoolDetailV1 = serde_json::from_value(response_json(pool).await).unwrap();
        assert_eq!(pool.summary.id, "pool-a");

        let exposure = app
            .clone()
            .oneshot(get(
                &format!("/api/router/v1/decisions/{decision_id}/exposure"),
                Some(TEST_TOKEN),
            ))
            .await
            .unwrap();
        assert_eq!(exposure.status(), StatusCode::OK);
        let exposure: DecisionExposureV1 =
            serde_json::from_value(response_json(exposure).await).unwrap();
        assert_eq!(exposure.decision_id, decision_id);
        assert!(exposure.active.is_none());

        let tail = app
            .clone()
            .oneshot(get(
                "/api/router/v1/decisions/tail?limit=1",
                Some(TEST_TOKEN),
            ))
            .await
            .unwrap();
        assert_eq!(tail.status(), StatusCode::OK);
        let tail: Page<DecisionSummaryV1> =
            serde_json::from_value(response_json(tail).await).unwrap();
        assert_eq!(tail.items[0].decision_id, decision_id);
        assert!(tail.next.is_some());

        assert_error(
            app.oneshot(get(
                "/api/router/v1/decisions/stream?unknown=true",
                Some(TEST_TOKEN),
            ))
            .await
            .unwrap(),
            StatusCode::BAD_REQUEST,
            "invalid_argument",
        )
        .await;
        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn semantic_etags_ignore_volatile_fields_change_with_records_and_follow_auth() {
        let (_temporary, config, mut activated) = decision_test_fixtures::activate(0.0, 10);
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        let app = inspection_http_router(
            service.clone(),
            bearer_guard(),
            InspectionHttpMode::ReadOnly,
        )
        .unwrap();

        for uri in [
            "/api/router/v1/overview",
            "/api/router/v1/pools",
            "/api/router/v1/controls",
            "/api/router/v1/health",
            "/api/router/v1/migrations",
        ] {
            let first = app
                .clone()
                .oneshot(get(uri, Some(TEST_TOKEN)))
                .await
                .unwrap();
            assert_eq!(first.status(), StatusCode::OK, "{uri}");
            assert!(first.headers().contains_key(ETAG), "{uri}");
        }

        let first_overview = app
            .clone()
            .oneshot(get("/api/router/v1/overview", Some(TEST_TOKEN)))
            .await
            .unwrap();
        let overview_etag = first_overview.headers().get(ETAG).unwrap().clone();
        let first_overview_json = response_json(first_overview).await;
        tokio::time::sleep(Duration::from_millis(2)).await;
        let second_overview = app
            .clone()
            .oneshot(conditional_get(
                "/api/router/v1/overview",
                Some(TEST_TOKEN),
                &overview_etag,
            ))
            .await
            .unwrap();
        assert_eq!(second_overview.status(), StatusCode::NOT_MODIFIED);
        assert_eq!(second_overview.headers().get(ETAG), Some(&overview_etag));
        assert!(
            second_overview
                .into_body()
                .collect()
                .await
                .unwrap()
                .to_bytes()
                .is_empty()
        );
        let fresh_overview = app
            .clone()
            .oneshot(get("/api/router/v1/overview", Some(TEST_TOKEN)))
            .await
            .unwrap();
        let fresh_overview_json = response_json(fresh_overview).await;
        assert_ne!(
            first_overview_json["snapshot_time_unix_ms"],
            fresh_overview_json["snapshot_time_unix_ms"]
        );

        assert_error(
            app.clone()
                .oneshot(conditional_get(
                    "/api/router/v1/overview",
                    None,
                    &overview_etag,
                ))
                .await
                .unwrap(),
            StatusCode::UNAUTHORIZED,
            "unauthorized",
        )
        .await;

        let first_health = app
            .clone()
            .oneshot(get("/api/router/v1/health", Some(TEST_TOKEN)))
            .await
            .unwrap();
        let health_etag = first_health.headers().get(ETAG).unwrap().clone();
        let health_event = LedgerHealthEvent::new(
            Uuid::now_v7(),
            Uuid::now_v7(),
            None,
            None,
            "router.dependency.cooloff_active",
            LedgerHealthSeverity::Warning,
            1_000,
        )
        .unwrap();
        activated
            .repository
            .append_health_event(&health_event)
            .unwrap();
        let changed = app
            .oneshot(conditional_get(
                "/api/router/v1/health",
                Some(TEST_TOKEN),
                &health_etag,
            ))
            .await
            .unwrap();
        assert_eq!(changed.status(), StatusCode::OK);
        assert_ne!(changed.headers().get(ETAG), Some(&health_etag));

        service.close().await.unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn decision_tail_and_stream_resume_in_order_without_replaying_a_checkpoint() {
        let (_temporary, config, mut activated) = decision_test_fixtures::activate(0.0, 10);
        let first = decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        let second =
            decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        for audit in [&first, &second] {
            assert_eq!(
                activated
                    .repository
                    .record_decision_audit(audit, 10, Uuid::now_v7())
                    .unwrap(),
                DecisionAuditAck::Applied
            );
        }
        let service = InspectionService::open(config.clone(), InspectionServiceOptions::default())
            .await
            .unwrap();
        let app = inspection_http_router(
            service.clone(),
            bearer_guard(),
            InspectionHttpMode::ReadOnly,
        )
        .unwrap();

        let initial = app
            .clone()
            .oneshot(get(
                "/api/router/v1/decisions/tail?limit=1",
                Some(TEST_TOKEN),
            ))
            .await
            .unwrap();
        let initial: Page<DecisionSummaryV1> =
            serde_json::from_value(response_json(initial).await).unwrap();
        assert_eq!(initial.items[0].decision_id, second.parent.decision_id);
        let cursor = initial.next.unwrap();

        let third = decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        let fourth =
            decision_test_fixtures::no_partition_audit(&activated, &config, Uuid::now_v7());
        for audit in [&third, &fourth] {
            assert_eq!(
                activated
                    .repository
                    .record_decision_audit(audit, 10, Uuid::now_v7())
                    .unwrap(),
                DecisionAuditAck::Applied
            );
        }

        let resumed = app
            .clone()
            .oneshot(get(
                &format!("/api/router/v1/decisions/tail?limit=1&after={cursor}"),
                Some(TEST_TOKEN),
            ))
            .await
            .unwrap();
        let resumed: Page<DecisionSummaryV1> =
            serde_json::from_value(response_json(resumed).await).unwrap();
        assert_eq!(resumed.items[0].decision_id, third.parent.decision_id);

        let mut stream_request = get("/api/router/v1/decisions/stream?limit=2", Some(TEST_TOKEN));
        stream_request.headers_mut().insert(
            HeaderName::from_static(LAST_EVENT_ID_HEADER),
            HeaderValue::from_str(&cursor).unwrap(),
        );
        let stream = app.oneshot(stream_request).await.unwrap();
        assert_eq!(stream.status(), StatusCode::OK);
        assert_eq!(
            stream.headers().get(CONTENT_TYPE),
            Some(&HeaderValue::from_static("text/event-stream"))
        );
        assert_no_store(&stream);
        let mut body = stream.into_body();
        let third_event = sse_json_frame(&mut body, "decision").await;
        let fourth_event = sse_json_frame(&mut body, "decision").await;
        let cursor_event = sse_json_frame(&mut body, "cursor").await;
        assert_eq!(
            third_event["decision_id"],
            third.parent.decision_id.to_string()
        );
        assert_eq!(
            fourth_event["decision_id"],
            fourth.parent.decision_id.to_string()
        );
        assert!(
            cursor_event["cursor"]
                .as_str()
                .is_some_and(|value| value != cursor)
        );
        assert!(
            tokio::time::timeout(Duration::from_millis(100), body.frame())
                .await
                .is_err(),
            "a completed cursor checkpoint must not replay decisions"
        );

        service.close().await.unwrap();
        assert!(
            tokio::time::timeout(Duration::from_secs(2), body.frame())
                .await
                .unwrap()
                .is_none(),
            "service closure should close the event stream"
        );
        drop(body);
        drop(activated);
    }

    #[tokio::test]
    async fn decision_stream_slots_are_bounded_and_released_on_disconnect() {
        let (_temporary, config, activated) = decision_test_fixtures::activate(0.0, 10);
        let service = InspectionService::open(config, InspectionServiceOptions::default())
            .await
            .unwrap();
        let app = inspection_http_router(
            service.clone(),
            bearer_guard(),
            InspectionHttpMode::ReadOnly,
        )
        .unwrap();
        let mut streams = Vec::new();
        for _ in 0..HTTP_STREAM_CONCURRENCY_MAX {
            let response = app
                .clone()
                .oneshot(get(
                    "/api/router/v1/decisions/stream?limit=1",
                    Some(TEST_TOKEN),
                ))
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
            streams.push(response);
        }

        assert_error(
            app.clone()
                .oneshot(get(
                    "/api/router/v1/decisions/stream?limit=1",
                    Some(TEST_TOKEN),
                ))
                .await
                .unwrap(),
            StatusCode::SERVICE_UNAVAILABLE,
            "busy",
        )
        .await;

        drop(streams.pop());
        let replacement = app
            .oneshot(get(
                "/api/router/v1/decisions/stream?limit=1",
                Some(TEST_TOKEN),
            ))
            .await
            .unwrap();
        assert_eq!(replacement.status(), StatusCode::OK);
        drop(replacement);
        drop(streams);
        tokio::time::timeout(Duration::from_secs(2), service.close())
            .await
            .unwrap()
            .unwrap();
        drop(activated);
    }

    #[tokio::test]
    async fn one_absolute_request_deadline_includes_the_host_guard() {
        let (_temporary, service) = missing_service(
            InspectionServiceOptions {
                request_timeout_ms: 1,
                ..InspectionServiceOptions::default()
            },
            "http-timeout",
        )
        .await;
        let response = inspection_http_router(
            service.clone(),
            Arc::new(SlowHostGuard),
            InspectionHttpMode::ReadOnly,
        )
        .unwrap()
        .oneshot(get("/api/router/v1/status", None))
        .await
        .unwrap();
        assert_error(response, StatusCode::SERVICE_UNAVAILABLE, "busy").await;
        service.close().await.unwrap();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn operations_routes_apply_and_report_compare_and_swap_conflicts() {
        let _control = crate::control::CONTROL_PUBLICATION_TEST_MUTEX.lock().await;
        let temporary = tempdir().unwrap();
        let path = temporary.path().join("router.db");
        let config = config(&path, "http-operations");
        let mut activated = LedgerRepository::activate(&config).unwrap();
        activated.repository.stop_abandoned_process();
        drop(activated);

        let service = InspectionService::open(
            config,
            InspectionServiceOptions {
                allow_operations: true,
                request_timeout_ms: 10_000,
                ..InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        let app = inspection_http_router(
            service.clone(),
            bearer_guard(),
            InspectionHttpMode::Operations,
        )
        .unwrap();

        let applied = app
            .clone()
            .oneshot(post_json(
                "/api/router/v1/operations/control",
                Some(TEST_TOKEN),
                &InspectionControlRequestV1 {
                    mutation_id: Uuid::now_v7(),
                    scope: ControlScope::All,
                    operation: ControlOperation::SetPaused { value: true },
                    expected_control_generation: 0,
                    actor: "http-operator".into(),
                    reason: "pause through HTTP".into(),
                },
            ))
            .await
            .unwrap();
        assert_eq!(applied.status(), StatusCode::OK);
        assert_no_store(&applied);
        assert_eq!(response_json(applied).await["control_generation"], 1);

        let conflict = app
            .oneshot(post_json(
                "/api/router/v1/operations/control",
                Some(TEST_TOKEN),
                &InspectionControlRequestV1 {
                    mutation_id: Uuid::now_v7(),
                    scope: ControlScope::All,
                    operation: ControlOperation::SetPaused { value: false },
                    expected_control_generation: 0,
                    actor: "http-operator".into(),
                    reason: "stale clear through HTTP".into(),
                },
            ))
            .await
            .unwrap();
        assert_error(conflict, StatusCode::CONFLICT, "conflict").await;
        service.close().await.unwrap();
        drop(service);
        assert_eq!(
            crate::control::router_control_service().unwrap_err(),
            RouterControlError::Unavailable
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn export_body_preserves_chunk_bounds_and_disconnect_cancels_production() {
        let (
            _temporary,
            config,
            activated,
            _vector_space_id,
            _vector_root,
            _evidence_id,
            _evaluation_id,
        ) = ready_evaluated_runtime_fixture();
        let service = InspectionService::open(
            config,
            InspectionServiceOptions {
                export_chunk_bytes: 4 * 1024,
                ..InspectionServiceOptions::default()
            },
        )
        .await
        .unwrap();
        let app = inspection_http_router(
            service.clone(),
            bearer_guard(),
            InspectionHttpMode::ReadOnly,
        )
        .unwrap();

        let response = app
            .clone()
            .oneshot(post_json(
                "/api/router/v1/evidence/export",
                Some(TEST_TOKEN),
                &EvidenceExportRequestV1::default(),
            ))
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert_no_store(&response);
        assert_eq!(
            response.headers().get(CONTENT_TYPE),
            Some(&HeaderValue::from_static("application/x-ndjson"))
        );
        let mut body = response.into_body();
        let frame = body.frame().await.unwrap().unwrap();
        let data = frame.data_ref().unwrap();
        assert!(!data.is_empty());
        assert!(data.len() <= 4 * 1024);
        drop(frame);
        drop(body);

        let disconnected = app
            .oneshot(post_json(
                "/api/router/v1/evidence/export",
                Some(TEST_TOKEN),
                &json!({"format": "csv"}),
            ))
            .await
            .unwrap();
        assert_eq!(disconnected.status(), StatusCode::OK);
        drop(disconnected);

        tokio::time::timeout(Duration::from_secs(2), service.close())
            .await
            .unwrap()
            .unwrap();
        drop(activated);
    }

    #[test]
    fn stable_error_status_mapping_covers_every_service_error() {
        for (error, status) in [
            (InspectionError::InvalidArgument, StatusCode::BAD_REQUEST),
            (InspectionError::InvalidCursor, StatusCode::BAD_REQUEST),
            (InspectionError::NotFound, StatusCode::NOT_FOUND),
            (InspectionError::Unauthorized, StatusCode::UNAUTHORIZED),
            (InspectionError::Forbidden, StatusCode::FORBIDDEN),
            (InspectionError::Conflict, StatusCode::CONFLICT),
            (InspectionError::Busy, StatusCode::SERVICE_UNAVAILABLE),
            (InspectionError::MutationExpired, StatusCode::GONE),
            (
                InspectionError::CapacityExhausted,
                StatusCode::INSUFFICIENT_STORAGE,
            ),
            (
                InspectionError::StorageUnavailable,
                StatusCode::SERVICE_UNAVAILABLE,
            ),
            (InspectionError::MigrationRequired, StatusCode::CONFLICT),
            (InspectionError::EgressDenied, StatusCode::FORBIDDEN),
            (
                InspectionError::IntegrityError,
                StatusCode::INTERNAL_SERVER_ERROR,
            ),
            (
                InspectionError::IncompatibleApi,
                StatusCode::UPGRADE_REQUIRED,
            ),
        ] {
            assert_eq!(status_for_error(error), status, "{}", error.code());
        }
    }
}
