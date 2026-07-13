// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Managed Shadow and Judge replay execution under isolated Evaluator scopes.

use std::collections::{BTreeMap, btree_map::Entry};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use nemo_relay::api::llm::{
    LlmApiFamily, LlmCallExecuteV2Params, LlmCallRole, LlmRequest, llm_call_execute_v2,
};
use nemo_relay::api::runtime::{
    LlmExecutionNextFn, LlmReplayCall, LlmReplayCancellationHandle, LlmReplayTransport,
    TASK_SCOPE_STACK, create_scope_stack,
};
use nemo_relay::api::scope::{
    PopScopeParams, PushScopeParams, ScopeHandle, ScopeType, pop_scope, push_scope,
};
use nemo_relay::codec::anthropic::AnthropicMessagesCodec;
use nemo_relay::codec::openai_chat::OpenAIChatCodec;
use nemo_relay::codec::openai_responses::OpenAIResponsesCodec;
use nemo_relay::codec::traits::{LlmCodec, LlmResponseCodec};
use nemo_relay::error::{FlowError, Result as FlowResult};
use nemo_relay::json::Json;
use tokio::sync::Notify;
use uuid::Uuid;

use crate::fingerprint::{BoundedFingerprintError, fingerprint_serializable_bounded};
use crate::response_validator::managed_response_is_bounded;

pub(crate) const EVALUATOR_SCOPE_NAME: &str = "nemo_relay.router.evaluator";
pub(crate) const SHADOW_CALL_NAME: &str = "nemo_relay.router.shadow";
pub(crate) const JUDGE_CALL_NAME: &str = "nemo_relay.router.judge";

const CANCELED_ERROR: &str = "Router replay canceled during shutdown";
const CAPACITY_ERROR: &str = "Router replay cancellation capacity exhausted";
const MIDDLEWARE_ERROR: &str = "Router managed replay middleware interference";
const RESPONSE_BOUND_ERROR: &str = "Router replay response exceeded its evidence bound";
const RUNTIME_ERROR: &str = "Router managed replay runtime failure";
const TRANSPORT_ERROR: &str = "Router replay transport failed";

/// Runtime-wide authority for starting new background provider work.
#[derive(Clone)]
pub(crate) struct BackgroundStartGate {
    open: Arc<AtomicBool>,
}

impl BackgroundStartGate {
    pub(crate) fn new() -> Self {
        Self {
            open: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Close the gate permanently. Returns whether this call performed the close.
    pub(crate) fn close(&self) -> bool {
        self.open.swap(false, Ordering::AcqRel)
    }

    pub(crate) fn is_open(&self) -> bool {
        self.open.load(Ordering::Acquire)
    }

    /// Linearize permission immediately before invoking the synchronous host start.
    fn start(
        &self,
        transport: &dyn LlmReplayTransport,
        request: LlmRequest,
    ) -> Result<LlmReplayCall, ReplayStartFailure> {
        if !self.open.load(Ordering::Acquire) {
            return Err(ReplayStartFailure::Closed);
        }
        match catch_unwind(AssertUnwindSafe(|| transport.start(request))) {
            Ok(Ok(call)) => Ok(call),
            Ok(Err(_)) | Err(_) => Err(ReplayStartFailure::Transport),
        }
    }
}

impl Default for BackgroundStartGate {
    fn default() -> Self {
        Self::new()
    }
}

enum ReplayStartFailure {
    Closed,
    Transport,
}

/// Monotonic cancellation signal shared by evaluator operations.
#[derive(Clone, Default)]
pub(crate) struct EvaluatorCancellation {
    inner: Arc<EvaluatorCancellationInner>,
}

#[derive(Default)]
struct EvaluatorCancellationInner {
    canceled: AtomicBool,
    notify: Notify,
}

impl EvaluatorCancellation {
    pub(crate) fn cancel(&self) -> bool {
        let first = !self.inner.canceled.swap(true, Ordering::AcqRel);
        if first {
            self.inner.notify.notify_waiters();
        }
        first
    }

    pub(crate) fn is_canceled(&self) -> bool {
        self.inner.canceled.load(Ordering::Acquire)
    }

    async fn canceled(&self) {
        loop {
            if self.is_canceled() {
                return;
            }
            let notified = self.inner.notify.notified();
            if self.is_canceled() {
                return;
            }
            notified.await;
        }
    }
}

/// Invalid capacity supplied for the active replay registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveReplayRegistryBuildError;

/// Exhaustive failure to register one already-started replay call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ActiveReplayRegistrationError {
    Closed,
    Full,
    Duplicate,
}

/// Bounded synchronous cancellation authority for currently active replay calls.
#[derive(Clone)]
pub(crate) struct ActiveReplayRegistry {
    inner: Arc<ActiveReplayRegistryInner>,
}

struct ActiveReplayRegistryInner {
    capacity: usize,
    state: Mutex<ActiveReplayRegistryState>,
}

#[derive(Default)]
struct ActiveReplayRegistryState {
    closed: bool,
    handles: BTreeMap<Uuid, LlmReplayCancellationHandle>,
}

impl ActiveReplayRegistry {
    pub(crate) fn new(capacity: usize) -> Result<Self, ActiveReplayRegistryBuildError> {
        if capacity == 0 {
            return Err(ActiveReplayRegistryBuildError);
        }
        Ok(Self {
            inner: Arc::new(ActiveReplayRegistryInner {
                capacity,
                state: Mutex::new(ActiveReplayRegistryState::default()),
            }),
        })
    }

    fn register(
        &self,
        invocation_id: Uuid,
        handle: LlmReplayCancellationHandle,
    ) -> Result<ActiveReplayRegistration, ActiveReplayRegistrationError> {
        let mut state = lock_unpoisoned(&self.inner.state);
        if state.closed {
            return Err(ActiveReplayRegistrationError::Closed);
        }
        if state.handles.len() >= self.inner.capacity {
            return Err(ActiveReplayRegistrationError::Full);
        }
        match state.handles.entry(invocation_id) {
            Entry::Vacant(entry) => {
                entry.insert(handle);
            }
            Entry::Occupied(_) => return Err(ActiveReplayRegistrationError::Duplicate),
        }
        Ok(ActiveReplayRegistration {
            invocation_id,
            registry: self.clone(),
        })
    }

    /// Close registration and synchronously invoke every registered cancellation handle.
    pub(crate) fn close_and_cancel_all(&self) -> usize {
        let handles = {
            let mut state = lock_unpoisoned(&self.inner.state);
            state.closed = true;
            std::mem::take(&mut state.handles)
                .into_values()
                .collect::<Vec<_>>()
        };
        handles
            .into_iter()
            .filter(LlmReplayCancellationHandle::cancel)
            .count()
    }

    #[cfg(test)]
    fn active_count(&self) -> usize {
        lock_unpoisoned(&self.inner.state).handles.len()
    }
}

struct ActiveReplayRegistration {
    invocation_id: Uuid,
    registry: ActiveReplayRegistry,
}

impl Drop for ActiveReplayRegistration {
    fn drop(&mut self) {
        lock_unpoisoned(&self.registry.inner.state)
            .handles
            .remove(&self.invocation_id);
    }
}

/// Stable, non-secret linkage shared by the Evaluator scope and managed call.
#[derive(Clone, PartialEq, Eq)]
pub(crate) struct EvaluatorLinkage {
    pub(crate) anchor_uuid: Uuid,
    pub(crate) anchor_id: Uuid,
    pub(crate) pool_id: String,
    pub(crate) candidate_id: String,
    pub(crate) config_generation_id: String,
    pub(crate) learning_generation_id: Uuid,
}

impl EvaluatorLinkage {
    fn sanitized_metadata(&self) -> BTreeMap<String, Json> {
        BTreeMap::from([
            (
                "anchor_uuid".to_string(),
                Json::String(self.anchor_uuid.to_string()),
            ),
            (
                "anchor_id".to_string(),
                Json::String(self.anchor_id.to_string()),
            ),
            ("pool_id".to_string(), Json::String(self.pool_id.clone())),
            (
                "candidate_id".to_string(),
                Json::String(self.candidate_id.clone()),
            ),
            (
                "config_generation_id".to_string(),
                Json::String(self.config_generation_id.clone()),
            ),
            (
                "learning_generation_id".to_string(),
                Json::String(self.learning_generation_id.to_string()),
            ),
        ])
    }

    fn event_metadata(&self) -> Json {
        Json::Object(self.sanitized_metadata().into_iter().collect())
    }
}

/// Inputs for one independent physical managed replay call.
pub(crate) struct ManagedReplayParams {
    pub(crate) invocation_id: Uuid,
    pub(crate) linkage: EvaluatorLinkage,
    pub(crate) api_family: LlmApiFamily,
    pub(crate) request: LlmRequest,
    pub(crate) transport: Arc<dyn LlmReplayTransport>,
    pub(crate) start_gate: BackgroundStartGate,
    pub(crate) active_replays: ActiveReplayRegistry,
    pub(crate) cancellation: EvaluatorCancellation,
    pub(crate) max_response_bytes: usize,
}

/// Stable evaluator classification without provider or middleware error text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ManagedReplayFailure {
    Transport,
    CanceledShutdown,
    MiddlewareInterference,
    EvidenceBound,
    CancellationCapacity,
    Runtime,
}

/// Complete managed replay result used by later validation and persistence stages.
pub(crate) enum ManagedReplayOutcome {
    Completed(Json),
    OperationalFailure(ManagedReplayFailure),
}

pub(crate) async fn execute_shadow_replay(params: ManagedReplayParams) -> ManagedReplayOutcome {
    execute_managed_replay(LlmCallRole::Shadow, SHADOW_CALL_NAME, params).await
}

pub(crate) async fn execute_judge_replay(params: ManagedReplayParams) -> ManagedReplayOutcome {
    execute_managed_replay(LlmCallRole::Judge, JUDGE_CALL_NAME, params).await
}

async fn execute_managed_replay(
    role: LlmCallRole,
    call_name: &'static str,
    params: ManagedReplayParams,
) -> ManagedReplayOutcome {
    TASK_SCOPE_STACK
        .scope(create_scope_stack(), async move {
            let scope = match push_scope(
                PushScopeParams::builder()
                    .name(EVALUATOR_SCOPE_NAME)
                    .scope_type(ScopeType::Evaluator)
                    .metadata(params.linkage.event_metadata())
                    .build(),
            ) {
                Ok(scope) => scope,
                Err(_) => {
                    return ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::Runtime);
                }
            };

            let outcome = execute_under_evaluator_scope(role, call_name, &scope, params).await;
            if pop_scope(PopScopeParams::builder().handle_uuid(&scope.uuid).build()).is_err() {
                return ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::Runtime);
            }
            outcome
        })
        .await
}

async fn execute_under_evaluator_scope(
    role: LlmCallRole,
    call_name: &'static str,
    scope: &ScopeHandle,
    params: ManagedReplayParams,
) -> ManagedReplayOutcome {
    let ManagedReplayParams {
        invocation_id,
        linkage,
        api_family,
        request,
        transport,
        start_gate,
        active_replays,
        cancellation,
        max_response_bytes,
    } = params;
    let observation = Arc::new(CallbackObservation::default());
    let callback_observation = observation.clone();
    let expected_request = request.clone();
    let callback: LlmExecutionNextFn = Arc::new(move |actual_request| {
        let invocation = callback_observation
            .invocations
            .fetch_add(1, Ordering::AcqRel);
        if invocation != 0 {
            callback_observation
                .interference
                .store(true, Ordering::Release);
            return fixed_error(MIDDLEWARE_ERROR);
        }
        if actual_request != expected_request {
            callback_observation
                .interference
                .store(true, Ordering::Release);
            callback_observation.set_terminal(CallbackTerminal::MiddlewareInterference);
            return fixed_error(MIDDLEWARE_ERROR);
        }

        let observation = callback_observation.clone();
        let transport = transport.clone();
        let start_gate = start_gate.clone();
        let active_replays = active_replays.clone();
        let cancellation = cancellation.clone();
        Box::pin(async move {
            if cancellation.is_canceled() {
                observation.set_terminal(CallbackTerminal::CanceledShutdown);
                return Err(FlowError::Internal(CANCELED_ERROR.to_string()));
            }
            let mut call = match start_gate.start(transport.as_ref(), actual_request) {
                Ok(call) => {
                    observation.starts.fetch_add(1, Ordering::AcqRel);
                    call
                }
                Err(ReplayStartFailure::Closed) => {
                    observation.set_terminal(CallbackTerminal::CanceledShutdown);
                    return Err(FlowError::Internal(CANCELED_ERROR.to_string()));
                }
                Err(ReplayStartFailure::Transport) => {
                    observation.starts.fetch_add(1, Ordering::AcqRel);
                    observation.set_terminal(CallbackTerminal::Transport);
                    return Err(FlowError::Internal(TRANSPORT_ERROR.to_string()));
                }
            };
            let cancellation_handle = call.cancellation_handle();
            let registration = match active_replays
                .register(invocation_id, cancellation_handle.clone())
            {
                Ok(registration) => registration,
                Err(ActiveReplayRegistrationError::Closed) => {
                    cancellation_handle.cancel();
                    drop(call);
                    observation.set_terminal(CallbackTerminal::CanceledShutdown);
                    return Err(FlowError::Internal(CANCELED_ERROR.to_string()));
                }
                Err(
                    ActiveReplayRegistrationError::Full | ActiveReplayRegistrationError::Duplicate,
                ) => {
                    cancellation_handle.cancel();
                    drop(call);
                    observation.set_terminal(CallbackTerminal::CancellationCapacity);
                    return Err(FlowError::Internal(CAPACITY_ERROR.to_string()));
                }
            };

            let replay_result = if cancellation.is_canceled() {
                None
            } else {
                tokio::select! {
                    biased;
                    () = cancellation.canceled() => None,
                    result = &mut call => Some(result),
                }
            };
            let result = match replay_result {
                Some(result) => result,
                None => {
                    cancellation_handle.cancel();
                    drop(call);
                    drop(registration);
                    observation.set_terminal(CallbackTerminal::CanceledShutdown);
                    return Err(FlowError::Internal(CANCELED_ERROR.to_string()));
                }
            };
            drop(call);
            drop(registration);
            match result {
                Ok(response) if !managed_response_is_bounded(&response, max_response_bytes) => {
                    observation.set_terminal(CallbackTerminal::EvidenceBound);
                    Err(FlowError::Internal(RESPONSE_BOUND_ERROR.to_string()))
                }
                Ok(response) => {
                    match fingerprint_serializable_bounded(&response, max_response_bytes) {
                        Ok(hash) => {
                            observation.set_terminal(CallbackTerminal::Completed(hash));
                            Ok(response)
                        }
                        Err(BoundedFingerprintError::BoundExceeded) => {
                            observation.set_terminal(CallbackTerminal::EvidenceBound);
                            Err(FlowError::Internal(RESPONSE_BOUND_ERROR.to_string()))
                        }
                        Err(BoundedFingerprintError::Serialization) => {
                            observation.set_terminal(CallbackTerminal::Runtime);
                            Err(FlowError::Internal(RUNTIME_ERROR.to_string()))
                        }
                    }
                }
                Err(_) => {
                    observation.set_terminal(CallbackTerminal::Transport);
                    Err(FlowError::Internal(TRANSPORT_ERROR.to_string()))
                }
            }
        })
    });

    let metadata = linkage.sanitized_metadata();
    let event_metadata = linkage.event_metadata();
    let model_name = request
        .content
        .get("model")
        .and_then(Json::as_str)
        .map(str::to_string);
    let (codec, response_codec) = codecs(api_family);
    let managed = llm_call_execute_v2(
        LlmCallExecuteV2Params::builder()
            .name(call_name)
            .request(request)
            .func(callback)
            .api_family(api_family)
            .call_role(role)
            .sanitized_metadata(metadata)
            .parent(scope.clone())
            .metadata(event_metadata)
            .model_name_opt(model_name)
            .codec(codec)
            .response_codec(response_codec)
            .build(),
    )
    .await;

    classify_managed_result(managed, &observation, max_response_bytes)
}

#[derive(Default)]
struct CallbackObservation {
    invocations: AtomicUsize,
    starts: AtomicUsize,
    interference: AtomicBool,
    terminal: Mutex<Option<CallbackTerminal>>,
}

enum CallbackTerminal {
    Completed(String),
    Transport,
    CanceledShutdown,
    MiddlewareInterference,
    EvidenceBound,
    CancellationCapacity,
    Runtime,
}

impl CallbackObservation {
    fn set_terminal(&self, terminal: CallbackTerminal) {
        let mut current = lock_unpoisoned(&self.terminal);
        if current.is_none() {
            *current = Some(terminal);
        }
    }
}

fn classify_managed_result(
    managed: FlowResult<Json>,
    observation: &CallbackObservation,
    max_response_bytes: usize,
) -> ManagedReplayOutcome {
    if observation.invocations.load(Ordering::Acquire) != 1
        || observation.starts.load(Ordering::Acquire) > 1
        || observation.interference.load(Ordering::Acquire)
    {
        return ManagedReplayOutcome::OperationalFailure(
            ManagedReplayFailure::MiddlewareInterference,
        );
    }

    let terminal = lock_unpoisoned(&observation.terminal);
    match (managed, terminal.as_ref()) {
        (Ok(response), Some(CallbackTerminal::Completed(raw_hash))) => {
            if !managed_response_is_bounded(&response, max_response_bytes) {
                return ManagedReplayOutcome::OperationalFailure(
                    ManagedReplayFailure::MiddlewareInterference,
                );
            }
            match fingerprint_serializable_bounded(&response, max_response_bytes) {
                Ok(managed_hash) if managed_hash == *raw_hash => {
                    ManagedReplayOutcome::Completed(response)
                }
                _ => ManagedReplayOutcome::OperationalFailure(
                    ManagedReplayFailure::MiddlewareInterference,
                ),
            }
        }
        (Err(error), Some(CallbackTerminal::Transport))
            if fixed_internal_error(&error, TRANSPORT_ERROR) =>
        {
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::Transport)
        }
        (Err(error), Some(CallbackTerminal::CanceledShutdown))
            if fixed_internal_error(&error, CANCELED_ERROR) =>
        {
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::CanceledShutdown)
        }
        (Err(error), Some(CallbackTerminal::EvidenceBound))
            if fixed_internal_error(&error, RESPONSE_BOUND_ERROR) =>
        {
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::EvidenceBound)
        }
        (Err(error), Some(CallbackTerminal::CancellationCapacity))
            if fixed_internal_error(&error, CAPACITY_ERROR) =>
        {
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::CancellationCapacity)
        }
        (Err(error), Some(CallbackTerminal::Runtime))
            if fixed_internal_error(&error, RUNTIME_ERROR) =>
        {
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::Runtime)
        }
        (Err(error), Some(CallbackTerminal::MiddlewareInterference))
            if fixed_internal_error(&error, MIDDLEWARE_ERROR) =>
        {
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::MiddlewareInterference)
        }
        _ => ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::MiddlewareInterference),
    }
}

fn fixed_error(
    message: &'static str,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = FlowResult<Json>> + Send>> {
    Box::pin(async move { Err(FlowError::Internal(message.to_string())) })
}

fn fixed_internal_error(error: &FlowError, expected: &str) -> bool {
    matches!(error, FlowError::Internal(message) if message == expected)
}

fn codecs(api_family: LlmApiFamily) -> (Arc<dyn LlmCodec>, Arc<dyn LlmResponseCodec>) {
    match api_family {
        LlmApiFamily::OpenAIChatCompletions => {
            (Arc::new(OpenAIChatCodec), Arc::new(OpenAIChatCodec))
        }
        LlmApiFamily::OpenAIResponses => (
            Arc::new(OpenAIResponsesCodec),
            Arc::new(OpenAIResponsesCodec),
        ),
        LlmApiFamily::AnthropicMessages => (
            Arc::new(AnthropicMessagesCodec),
            Arc::new(AnthropicMessagesCodec),
        ),
    }
}

fn lock_unpoisoned<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(test)]
mod tests {
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use nemo_relay::api::event::{Event, ScopeCategory};
    use nemo_relay::api::llm::{LlmExecutionContextSnapshot, LlmRequestInterceptOutcome};
    use nemo_relay::api::registry::{
        deregister_llm_execution_intercept, deregister_llm_request_intercept,
        register_llm_execution_intercept, register_llm_execution_intercept_v2,
        register_llm_request_intercept,
    };
    use nemo_relay::api::runtime::{
        LLM_REPLAY_CONTRACT_VERSION, LlmReplayCapability, NemoRelayContextState, global_context,
        task_scope_top,
    };
    use nemo_relay::api::subscriber::{
        deregister_subscriber, flush_subscribers, register_subscriber,
    };
    use serde_json::{Map, json};

    use super::*;

    #[derive(Clone)]
    enum TestReplayBehavior {
        Immediate(Json),
        Pending,
        Error,
    }

    struct TestReplayTransport {
        capability: LlmReplayCapability,
        behavior: TestReplayBehavior,
        starts: Arc<AtomicUsize>,
        cancellations: Arc<AtomicUsize>,
        requests: Arc<Mutex<Vec<LlmRequest>>>,
    }

    impl TestReplayTransport {
        fn new(family: LlmApiFamily, behavior: TestReplayBehavior) -> Arc<Self> {
            Arc::new(Self {
                capability: LlmReplayCapability {
                    contract_version: LLM_REPLAY_CONTRACT_VERSION,
                    api_family: family,
                    transport_identity: "evaluator-test".to_string(),
                },
                behavior,
                starts: Arc::new(AtomicUsize::new(0)),
                cancellations: Arc::new(AtomicUsize::new(0)),
                requests: Arc::new(Mutex::new(Vec::new())),
            })
        }
    }

    impl LlmReplayTransport for TestReplayTransport {
        fn capability(&self) -> &LlmReplayCapability {
            &self.capability
        }

        fn start(&self, request: LlmRequest) -> FlowResult<LlmReplayCall> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            lock_unpoisoned(&self.requests).push(request);
            let behavior = self.behavior.clone();
            let cancellations = self.cancellations.clone();
            Ok(LlmReplayCall::new(
                async move {
                    match behavior {
                        TestReplayBehavior::Immediate(response) => Ok(response),
                        TestReplayBehavior::Pending => std::future::pending().await,
                        TestReplayBehavior::Error => {
                            Err(FlowError::Internal("provider detail".to_string()))
                        }
                    }
                },
                move || {
                    cancellations.fetch_add(1, Ordering::SeqCst);
                },
            ))
        }
    }

    struct BlockingStartTransport {
        capability: LlmReplayCapability,
        entered: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
        release: Arc<Barrier>,
        behavior: TestReplayBehavior,
        starts: AtomicUsize,
        cancellations: Arc<AtomicUsize>,
    }

    impl LlmReplayTransport for BlockingStartTransport {
        fn capability(&self) -> &LlmReplayCapability {
            &self.capability
        }

        fn start(&self, _request: LlmRequest) -> FlowResult<LlmReplayCall> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            if let Some(entered) = lock_unpoisoned(&self.entered).take() {
                let _ = entered.send(());
            }
            self.release.wait();
            let behavior = self.behavior.clone();
            let cancellations = self.cancellations.clone();
            Ok(LlmReplayCall::new(
                async move {
                    match behavior {
                        TestReplayBehavior::Immediate(response) => Ok(response),
                        TestReplayBehavior::Pending => std::future::pending().await,
                        TestReplayBehavior::Error => {
                            Err(FlowError::Internal("provider detail".to_string()))
                        }
                    }
                },
                move || {
                    cancellations.fetch_add(1, Ordering::SeqCst);
                },
            ))
        }
    }

    fn blocking_transport(
        behavior: TestReplayBehavior,
    ) -> (
        Arc<BlockingStartTransport>,
        tokio::sync::oneshot::Receiver<()>,
        Arc<Barrier>,
    ) {
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let release = Arc::new(Barrier::new(2));
        (
            Arc::new(BlockingStartTransport {
                capability: LlmReplayCapability {
                    contract_version: LLM_REPLAY_CONTRACT_VERSION,
                    api_family: LlmApiFamily::OpenAIChatCompletions,
                    transport_identity: "blocking-evaluator-test".to_string(),
                },
                entered: Mutex::new(Some(entered_tx)),
                release: release.clone(),
                behavior,
                starts: AtomicUsize::new(0),
                cancellations: Arc::new(AtomicUsize::new(0)),
            }),
            entered_rx,
            release,
        )
    }

    fn reset_runtime() {
        *global_context().write().unwrap() = NemoRelayContextState::new();
    }

    fn linkage() -> EvaluatorLinkage {
        EvaluatorLinkage {
            anchor_uuid: Uuid::now_v7(),
            anchor_id: Uuid::now_v7(),
            pool_id: "pool-a".to_string(),
            candidate_id: "candidate-a".to_string(),
            config_generation_id: "a".repeat(64),
            learning_generation_id: Uuid::now_v7(),
        }
    }

    fn chat_request() -> LlmRequest {
        LlmRequest {
            headers: Map::new(),
            content: json!({
                "model": "candidate-model",
                "messages": [{"role": "user", "content": "hello"}]
            }),
        }
    }

    fn chat_response(text: &str) -> Json {
        json!({
            "id": "chatcmpl-evaluator",
            "object": "chat.completion",
            "created": 1,
            "model": "candidate-model",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}
        })
    }

    fn family_fixture(family: LlmApiFamily) -> (LlmRequest, Json) {
        match family {
            LlmApiFamily::OpenAIChatCompletions => (chat_request(), chat_response("answer")),
            LlmApiFamily::OpenAIResponses => (
                LlmRequest {
                    headers: Map::new(),
                    content: json!({
                        "model": "candidate-model",
                        "input": [{"role": "user", "content": "hello"}]
                    }),
                },
                json!({
                    "id": "resp-evaluator",
                    "object": "response",
                    "created_at": 1,
                    "model": "candidate-model",
                    "status": "completed",
                    "error": null,
                    "incomplete_details": null,
                    "output": [{
                        "type": "message",
                        "id": "msg-evaluator",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": "answer",
                            "annotations": []
                        }]
                    }],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}
                }),
            ),
            LlmApiFamily::AnthropicMessages => (
                LlmRequest {
                    headers: Map::new(),
                    content: json!({
                        "model": "candidate-model",
                        "max_tokens": 32,
                        "messages": [{"role": "user", "content": "hello"}]
                    }),
                },
                json!({
                    "id": "msg-evaluator",
                    "type": "message",
                    "role": "assistant",
                    "model": "candidate-model",
                    "content": [{"type": "text", "text": "answer"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": null,
                    "usage": {"input_tokens": 1, "output_tokens": 1}
                }),
            ),
        }
    }

    fn params(
        family: LlmApiFamily,
        request: LlmRequest,
        transport: Arc<dyn LlmReplayTransport>,
        start_gate: BackgroundStartGate,
        active_replays: ActiveReplayRegistry,
        cancellation: EvaluatorCancellation,
    ) -> ManagedReplayParams {
        ManagedReplayParams {
            invocation_id: Uuid::now_v7(),
            linkage: linkage(),
            api_family: family,
            request,
            transport,
            start_gate,
            active_replays,
            cancellation,
            max_response_bytes: 64 * 1024,
        }
    }

    async fn wait_until(mut condition: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(2), async {
            while !condition() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("test condition did not converge");
    }

    #[test]
    fn registry_is_bounded_duplicate_safe_and_cancels_outside_registration_lifetime() {
        let registry = ActiveReplayRegistry::new(1).unwrap();
        let first_cancellations = Arc::new(AtomicUsize::new(0));
        let first_probe = first_cancellations.clone();
        let first = LlmReplayCall::new(std::future::pending(), move || {
            first_probe.fetch_add(1, Ordering::SeqCst);
        });
        let first_id = Uuid::now_v7();
        let registration = registry
            .register(first_id, first.cancellation_handle())
            .unwrap();
        assert_eq!(registry.active_count(), 1);

        let duplicate = LlmReplayCall::new(std::future::pending(), || {});
        assert!(matches!(
            registry.register(first_id, duplicate.cancellation_handle()),
            Err(ActiveReplayRegistrationError::Full)
        ));
        drop(duplicate);

        assert_eq!(registry.close_and_cancel_all(), 1);
        assert_eq!(first_cancellations.load(Ordering::SeqCst), 1);
        assert_eq!(registry.active_count(), 0);
        drop(first);
        drop(registration);
        assert_eq!(first_cancellations.load(Ordering::SeqCst), 1);

        let late = LlmReplayCall::new(std::future::pending(), || {});
        assert!(matches!(
            registry.register(Uuid::now_v7(), late.cancellation_handle()),
            Err(ActiveReplayRegistrationError::Closed)
        ));
    }

    #[tokio::test]
    async fn all_families_execute_frozen_request_once_for_shadow_and_judge() {
        let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_runtime();
        for (index, family) in [
            LlmApiFamily::OpenAIChatCompletions,
            LlmApiFamily::OpenAIResponses,
            LlmApiFamily::AnthropicMessages,
        ]
        .into_iter()
        .enumerate()
        {
            let (request, response) = family_fixture(family);
            let transport =
                TestReplayTransport::new(family, TestReplayBehavior::Immediate(response.clone()));
            let params = params(
                family,
                request.clone(),
                transport.clone(),
                BackgroundStartGate::new(),
                ActiveReplayRegistry::new(2).unwrap(),
                EvaluatorCancellation::default(),
            );
            let outcome = if index % 2 == 0 {
                execute_shadow_replay(params).await
            } else {
                execute_judge_replay(params).await
            };
            assert!(
                matches!(outcome, ManagedReplayOutcome::Completed(actual) if actual == response)
            );
            assert_eq!(transport.starts.load(Ordering::SeqCst), 1);
            assert_eq!(lock_unpoisoned(&transport.requests).as_slice(), &[request]);
            assert_eq!(transport.cancellations.load(Ordering::SeqCst), 0);
        }
    }

    #[tokio::test]
    async fn isolated_scope_and_call_have_fixed_names_roles_and_linkage() {
        let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_runtime();
        let events = Arc::new(Mutex::new(Vec::<Event>::new()));
        let event_sink = events.clone();
        register_subscriber(
            "router-evaluator-isolation-events",
            Arc::new(move |event| lock_unpoisoned(&event_sink).push(event.clone())),
        )
        .unwrap();
        let contexts = Arc::new(Mutex::new(Vec::<Arc<LlmExecutionContextSnapshot>>::new()));
        let context_sink = contexts.clone();
        register_llm_execution_intercept_v2(
            "router-evaluator-isolation-context",
            0,
            Arc::new(move |_, context, request, _, next| {
                lock_unpoisoned(&context_sink).push(context);
                Box::pin(async move { next(request).await })
            }),
        )
        .unwrap();

        let outer_stack = create_scope_stack();
        let (outer_uuid, outcome) = TASK_SCOPE_STACK
            .scope(outer_stack, async {
                let outer = push_scope(
                    PushScopeParams::builder()
                        .name("primary-agent")
                        .scope_type(ScopeType::Agent)
                        .build(),
                )
                .unwrap();
                let transport = TestReplayTransport::new(
                    LlmApiFamily::OpenAIChatCompletions,
                    TestReplayBehavior::Immediate(chat_response("answer")),
                );
                let outcome = execute_shadow_replay(params(
                    LlmApiFamily::OpenAIChatCompletions,
                    chat_request(),
                    transport,
                    BackgroundStartGate::new(),
                    ActiveReplayRegistry::new(1).unwrap(),
                    EvaluatorCancellation::default(),
                ))
                .await;
                assert_eq!(task_scope_top().uuid, outer.uuid);
                let outer_uuid = outer.uuid;
                pop_scope(PopScopeParams::builder().handle_uuid(&outer.uuid).build()).unwrap();
                (outer_uuid, outcome)
            })
            .await;
        assert!(matches!(outcome, ManagedReplayOutcome::Completed(_)));
        flush_subscribers().unwrap();

        let captured = lock_unpoisoned(&contexts);
        assert_eq!(captured.len(), 1);
        let context = &captured[0];
        assert_eq!(context.call_role, LlmCallRole::Shadow);
        assert_eq!(context.trajectory_owner_path.len(), 2);
        assert_eq!(
            context.trajectory_owner_path.last().unwrap().scope_type,
            ScopeType::Evaluator
        );
        assert!(
            !context
                .trajectory_owner_path
                .iter()
                .any(|scope| scope.uuid == outer_uuid)
        );
        assert_eq!(context.sanitized_metadata.len(), 6);
        for key in [
            "anchor_uuid",
            "anchor_id",
            "pool_id",
            "candidate_id",
            "config_generation_id",
            "learning_generation_id",
        ] {
            assert!(context.sanitized_metadata.contains_key(key));
        }
        drop(captured);

        let captured_events = lock_unpoisoned(&events);
        let evaluator_events = captured_events
            .iter()
            .filter(|event| event.name() == EVALUATOR_SCOPE_NAME)
            .collect::<Vec<_>>();
        assert_eq!(evaluator_events.len(), 2);
        assert_eq!(
            evaluator_events[0].scope_category(),
            Some(ScopeCategory::Start)
        );
        assert_eq!(
            evaluator_events[1].scope_category(),
            Some(ScopeCategory::End)
        );
        let shadow_events = captured_events
            .iter()
            .filter(|event| event.name() == SHADOW_CALL_NAME)
            .collect::<Vec<_>>();
        assert_eq!(shadow_events.len(), 2);
        assert!(
            shadow_events
                .iter()
                .all(|event| event.llm_call_role() == Some(LlmCallRole::Shadow))
        );
        drop(captured_events);

        deregister_llm_execution_intercept("router-evaluator-isolation-context").unwrap();
        deregister_subscriber("router-evaluator-isolation-events").unwrap();
    }

    #[tokio::test]
    async fn request_mutation_and_response_replacement_are_middleware_interference() {
        let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_runtime();
        register_llm_request_intercept(
            "router-evaluator-mutate-request",
            0,
            false,
            Arc::new(|_, mut request, annotated| {
                request.headers.insert("x-mutated".to_string(), json!(true));
                Ok(LlmRequestInterceptOutcome::new(request, annotated))
            }),
        )
        .unwrap();
        let transport = TestReplayTransport::new(
            LlmApiFamily::OpenAIChatCompletions,
            TestReplayBehavior::Immediate(chat_response("raw")),
        );
        let outcome = execute_shadow_replay(params(
            LlmApiFamily::OpenAIChatCompletions,
            chat_request(),
            transport.clone(),
            BackgroundStartGate::new(),
            ActiveReplayRegistry::new(1).unwrap(),
            EvaluatorCancellation::default(),
        ))
        .await;
        assert!(matches!(
            outcome,
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::MiddlewareInterference)
        ));
        assert_eq!(transport.starts.load(Ordering::SeqCst), 0);
        deregister_llm_request_intercept("router-evaluator-mutate-request").unwrap();

        register_llm_execution_intercept_v2(
            "router-evaluator-replace-response",
            0,
            Arc::new(|_, _, request, _, next| {
                Box::pin(async move {
                    next(request).await?;
                    Ok(chat_response("replacement"))
                })
            }),
        )
        .unwrap();
        let outcome = execute_shadow_replay(params(
            LlmApiFamily::OpenAIChatCompletions,
            chat_request(),
            transport.clone(),
            BackgroundStartGate::new(),
            ActiveReplayRegistry::new(1).unwrap(),
            EvaluatorCancellation::default(),
        ))
        .await;
        assert!(matches!(
            outcome,
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::MiddlewareInterference)
        ));
        assert_eq!(transport.starts.load(Ordering::SeqCst), 1);
        deregister_llm_execution_intercept("router-evaluator-replace-response").unwrap();
    }

    #[tokio::test]
    async fn oversized_provider_response_is_rejected_before_managed_hashing() {
        let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_runtime();
        let transport = TestReplayTransport::new(
            LlmApiFamily::OpenAIChatCompletions,
            TestReplayBehavior::Immediate(chat_response(&"x".repeat(8_192))),
        );
        let mut replay = params(
            LlmApiFamily::OpenAIChatCompletions,
            chat_request(),
            transport.clone(),
            BackgroundStartGate::new(),
            ActiveReplayRegistry::new(1).unwrap(),
            EvaluatorCancellation::default(),
        );
        replay.max_response_bytes = 128;

        let outcome = execute_shadow_replay(replay).await;
        assert!(matches!(
            outcome,
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::EvidenceBound)
        ));
        assert_eq!(transport.starts.load(Ordering::SeqCst), 1);
        assert_eq!(transport.cancellations.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn oversized_middleware_replacement_remains_interference() {
        let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_runtime();
        register_llm_execution_intercept_v2(
            "router-evaluator-oversized-replacement",
            0,
            Arc::new(|_, _, request, _, next| {
                Box::pin(async move {
                    next(request).await?;
                    Ok(chat_response(&"x".repeat(8_192)))
                })
            }),
        )
        .unwrap();
        let transport = TestReplayTransport::new(
            LlmApiFamily::OpenAIChatCompletions,
            TestReplayBehavior::Immediate(chat_response("raw")),
        );
        let mut replay = params(
            LlmApiFamily::OpenAIChatCompletions,
            chat_request(),
            transport.clone(),
            BackgroundStartGate::new(),
            ActiveReplayRegistry::new(1).unwrap(),
            EvaluatorCancellation::default(),
        );
        replay.max_response_bytes = 1_024;

        let outcome = execute_shadow_replay(replay).await;
        assert!(matches!(
            outcome,
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::MiddlewareInterference)
        ));
        assert_eq!(transport.starts.load(Ordering::SeqCst), 1);
        assert_eq!(transport.cancellations.load(Ordering::SeqCst), 0);
        deregister_llm_execution_intercept("router-evaluator-oversized-replacement").unwrap();
    }

    #[tokio::test]
    async fn middleware_bypass_never_starts_replay() {
        let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_runtime();
        register_llm_execution_intercept_v2(
            "router-evaluator-bypass",
            0,
            Arc::new(|_, _, _, _, _| Box::pin(async { Ok(chat_response("bypass")) })),
        )
        .unwrap();
        let transport = TestReplayTransport::new(
            LlmApiFamily::OpenAIChatCompletions,
            TestReplayBehavior::Immediate(chat_response("raw")),
        );
        let outcome = execute_shadow_replay(params(
            LlmApiFamily::OpenAIChatCompletions,
            chat_request(),
            transport.clone(),
            BackgroundStartGate::new(),
            ActiveReplayRegistry::new(1).unwrap(),
            EvaluatorCancellation::default(),
        ))
        .await;
        assert!(matches!(
            outcome,
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::MiddlewareInterference)
        ));
        assert_eq!(transport.starts.load(Ordering::SeqCst), 0);
        deregister_llm_execution_intercept("router-evaluator-bypass").unwrap();
    }

    #[tokio::test]
    async fn legacy_multiple_next_calls_start_transport_only_once() {
        let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_runtime();
        register_llm_execution_intercept(
            "router-evaluator-multiple-next",
            0,
            Arc::new(|_, request, next| {
                Box::pin(async move {
                    let first = next(request.clone()).await;
                    let _second = next(request).await;
                    first
                })
            }),
        )
        .unwrap();
        let transport = TestReplayTransport::new(
            LlmApiFamily::OpenAIChatCompletions,
            TestReplayBehavior::Immediate(chat_response("raw")),
        );
        let outcome = execute_shadow_replay(params(
            LlmApiFamily::OpenAIChatCompletions,
            chat_request(),
            transport.clone(),
            BackgroundStartGate::new(),
            ActiveReplayRegistry::new(1).unwrap(),
            EvaluatorCancellation::default(),
        ))
        .await;
        assert!(matches!(
            outcome,
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::MiddlewareInterference)
        ));
        assert_eq!(transport.starts.load(Ordering::SeqCst), 1);
        deregister_llm_execution_intercept("router-evaluator-multiple-next").unwrap();
    }

    #[tokio::test]
    async fn graceful_cancel_drops_call_after_registration_and_emits_end_events() {
        let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_runtime();
        let events = Arc::new(Mutex::new(Vec::<Event>::new()));
        let event_sink = events.clone();
        register_subscriber(
            "router-evaluator-graceful-events",
            Arc::new(move |event| lock_unpoisoned(&event_sink).push(event.clone())),
        )
        .unwrap();
        let transport = TestReplayTransport::new(
            LlmApiFamily::OpenAIChatCompletions,
            TestReplayBehavior::Pending,
        );
        let registry = ActiveReplayRegistry::new(1).unwrap();
        let cancellation = EvaluatorCancellation::default();
        let task = tokio::spawn(execute_shadow_replay(params(
            LlmApiFamily::OpenAIChatCompletions,
            chat_request(),
            transport.clone(),
            BackgroundStartGate::new(),
            registry.clone(),
            cancellation.clone(),
        )));
        wait_until(|| registry.active_count() == 1).await;
        assert_eq!(registry.close_and_cancel_all(), 1);
        assert!(cancellation.cancel());
        let outcome = task.await.unwrap();
        assert!(matches!(
            outcome,
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::CanceledShutdown)
        ));
        assert_eq!(transport.cancellations.load(Ordering::SeqCst), 1);
        assert_eq!(registry.active_count(), 0);
        flush_subscribers().unwrap();
        let captured = lock_unpoisoned(&events);
        assert_eq!(
            captured
                .iter()
                .filter(|event| event.name() == SHADOW_CALL_NAME)
                .count(),
            2
        );
        assert_eq!(
            captured
                .iter()
                .filter(|event| event.name() == EVALUATOR_SCOPE_NAME)
                .count(),
            2
        );
        drop(captured);
        deregister_subscriber("router-evaluator-graceful-events").unwrap();
    }

    #[tokio::test]
    async fn forced_cancel_invokes_registered_host_handle_before_task_abort() {
        let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_runtime();
        let transport = TestReplayTransport::new(
            LlmApiFamily::OpenAIChatCompletions,
            TestReplayBehavior::Pending,
        );
        let registry = ActiveReplayRegistry::new(1).unwrap();
        let task = tokio::spawn(execute_judge_replay(params(
            LlmApiFamily::OpenAIChatCompletions,
            chat_request(),
            transport.clone(),
            BackgroundStartGate::new(),
            registry.clone(),
            EvaluatorCancellation::default(),
        )));
        wait_until(|| registry.active_count() == 1).await;
        assert_eq!(registry.close_and_cancel_all(), 1);
        assert_eq!(transport.cancellations.load(Ordering::SeqCst), 1);
        task.abort();
        assert!(matches!(task.await, Err(error) if error.is_cancelled()));
        assert_eq!(transport.cancellations.load(Ordering::SeqCst), 1);
        assert_eq!(registry.active_count(), 0);
    }

    #[tokio::test]
    async fn closed_gate_and_transport_error_have_distinct_operational_results() {
        let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_runtime();
        let gate = BackgroundStartGate::new();
        assert!(gate.close());
        let transport = TestReplayTransport::new(
            LlmApiFamily::OpenAIChatCompletions,
            TestReplayBehavior::Pending,
        );
        let outcome = execute_shadow_replay(params(
            LlmApiFamily::OpenAIChatCompletions,
            chat_request(),
            transport.clone(),
            gate,
            ActiveReplayRegistry::new(1).unwrap(),
            EvaluatorCancellation::default(),
        ))
        .await;
        assert!(matches!(
            outcome,
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::CanceledShutdown)
        ));
        assert_eq!(transport.starts.load(Ordering::SeqCst), 0);

        let failing = TestReplayTransport::new(
            LlmApiFamily::OpenAIChatCompletions,
            TestReplayBehavior::Error,
        );
        let outcome = execute_shadow_replay(params(
            LlmApiFamily::OpenAIChatCompletions,
            chat_request(),
            failing.clone(),
            BackgroundStartGate::new(),
            ActiveReplayRegistry::new(1).unwrap(),
            EvaluatorCancellation::default(),
        ))
        .await;
        assert!(matches!(
            outcome,
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::Transport)
        ));
        assert_eq!(failing.starts.load(Ordering::SeqCst), 1);
        assert_eq!(failing.cancellations.load(Ordering::SeqCst), 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn gate_close_racing_synchronous_start_obeys_load_linearization() {
        let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_runtime();
        let gate = BackgroundStartGate::new();
        let (transport, entered, release) =
            blocking_transport(TestReplayBehavior::Immediate(chat_response("linearized")));
        let first = tokio::spawn(execute_shadow_replay(params(
            LlmApiFamily::OpenAIChatCompletions,
            chat_request(),
            transport.clone(),
            gate.clone(),
            ActiveReplayRegistry::new(1).unwrap(),
            EvaluatorCancellation::default(),
        )));
        entered.await.unwrap();
        assert!(gate.close());
        release.wait();
        assert!(matches!(
            first.await.unwrap(),
            ManagedReplayOutcome::Completed(response)
                if response == chat_response("linearized")
        ));
        assert_eq!(transport.starts.load(Ordering::SeqCst), 1);

        let second = execute_shadow_replay(params(
            LlmApiFamily::OpenAIChatCompletions,
            chat_request(),
            transport.clone(),
            gate,
            ActiveReplayRegistry::new(1).unwrap(),
            EvaluatorCancellation::default(),
        ))
        .await;
        assert!(matches!(
            second,
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::CanceledShutdown)
        ));
        assert_eq!(transport.starts.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn registry_close_during_start_cancels_on_registration_without_hanging() {
        let _guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_runtime();
        let (transport, entered, release) = blocking_transport(TestReplayBehavior::Pending);
        let registry = ActiveReplayRegistry::new(1).unwrap();
        let task = tokio::spawn(execute_shadow_replay(params(
            LlmApiFamily::OpenAIChatCompletions,
            chat_request(),
            transport.clone(),
            BackgroundStartGate::new(),
            registry.clone(),
            EvaluatorCancellation::default(),
        )));
        entered.await.unwrap();
        assert_eq!(registry.close_and_cancel_all(), 0);
        release.wait();
        let outcome = tokio::time::timeout(Duration::from_secs(2), task)
            .await
            .expect("closed registry left a replay task pending")
            .unwrap();
        assert!(matches!(
            outcome,
            ManagedReplayOutcome::OperationalFailure(ManagedReplayFailure::CanceledShutdown)
        ));
        assert_eq!(transport.starts.load(Ordering::SeqCst), 1);
        assert_eq!(transport.cancellations.load(Ordering::SeqCst), 1);
        assert_eq!(registry.active_count(), 0);
    }
}
