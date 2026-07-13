// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests for plugin in the NeMo Relay core crate.

use super::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Barrier, Mutex, OnceLock};
use std::time::Duration;

use serde_json::json;

use crate::api::llm::{
    LlmApiFamily, LlmCallExecuteV2Params, LlmCallRole, LlmRequest, LlmRequestInterceptOutcome,
    llm_call_execute_v2, llm_conditional_execution, llm_request_intercepts,
};
use crate::api::runtime::{
    LLM_REPLAY_CONTRACT_VERSION, LlmReplayCall, LlmReplayCapability, LlmReplayTransport,
    NemoRelayContextState, ScopeStackHandle, TASK_SCOPE_STACK, create_scope_stack, global_context,
    set_thread_scope_stack, with_scope_stack,
};
use crate::api::scope::{
    EmitMarkEventParams, PushScopeParams, ScopeHandle, ScopeType, event as emit_scope_event,
    push_scope,
};
use crate::api::tool::tool_conditional_execution;
use crate::error::FlowError;

struct TestPlugin;

struct SingletonPlugin;
struct RecordingPlugin;
struct ReplacementPlugin;
struct RestoreFailPlugin;
struct RestoreBreakPlugin;
struct PartialFailPlugin;
struct VanishingPlugin;
struct ActivationCancelPlugin;
struct ActivationCommitRacePlugin {
    barrier: Arc<Barrier>,
    candidate_rollbacks: Arc<AtomicUsize>,
}

#[derive(Clone, Copy)]
enum TestDrainBehavior {
    Success,
    Error,
    Panic,
    Pending,
}

fn shutdown_test_registration(
    name: &'static str,
    order: Arc<Mutex<Vec<String>>>,
    drain_behavior: TestDrainBehavior,
) -> PluginRegistration {
    let deregister_order = Arc::clone(&order);
    let stop_order = Arc::clone(&order);
    let drain_order = Arc::clone(&order);
    let abort_order = Arc::clone(&order);
    PluginRegistration::with_shutdown(
        "test",
        name,
        Box::new(move || {
            deregister_order
                .lock()
                .unwrap()
                .push(format!("deregister:{name}"));
            Ok(())
        }),
        Box::new(move || {
            stop_order.lock().unwrap().push(format!("stop:{name}"));
            Ok(())
        }),
        Box::new(move |_deadline| {
            let order = Arc::clone(&drain_order);
            Box::pin(async move {
                order.lock().unwrap().push(format!("drain:{name}"));
                match drain_behavior {
                    TestDrainBehavior::Success => Ok(()),
                    TestDrainBehavior::Error => {
                        Err(PluginError::Internal(format!("{name} drain error")))
                    }
                    TestDrainBehavior::Panic => panic!("{name} drain panic"),
                    TestDrainBehavior::Pending => std::future::pending::<Result<()>>().await,
                }
            })
        }),
        Box::new(move || {
            abort_order.lock().unwrap().push(format!("abort:{name}"));
            Ok(())
        }),
    )
}

#[derive(Default)]
struct InternalReplayShutdownState {
    intake_open: bool,
    after_return: bool,
    managed_calls: usize,
    replay_starts: usize,
    managed_calls_after_return: usize,
    replay_starts_after_return: usize,
    completed_workers: usize,
    workers: Vec<std::thread::JoinHandle<()>>,
}

struct InternalReplayShutdownInner {
    state: Mutex<InternalReplayShutdownState>,
    scope_stack: ScopeStackHandle,
    evaluator: ScopeHandle,
    anchor_uuid: uuid::Uuid,
    replay_release: AtomicBool,
    replay_aborted: AtomicBool,
    replay_release_notify: tokio::sync::Notify,
    provider_start_release: AtomicBool,
    provider_start_notify: tokio::sync::Notify,
    worker_completion_notify: tokio::sync::Notify,
    replay_started: SyncSender<()>,
    provider_paused: SyncSender<()>,
    intake_stopped: SyncSender<()>,
}

struct InternalReplayShutdownTransport {
    capability: LlmReplayCapability,
    inner: Arc<InternalReplayShutdownInner>,
}

impl LlmReplayTransport for InternalReplayShutdownTransport {
    fn capability(&self) -> &LlmReplayCapability {
        &self.capability
    }

    fn start(&self, request: LlmRequest) -> crate::error::Result<LlmReplayCall> {
        {
            let mut state = self.inner.state.lock().unwrap();
            if !state.intake_open {
                return Err(FlowError::Internal(
                    "internal replay intake is closed".to_string(),
                ));
            }
            state.replay_starts += 1;
            if state.after_return {
                state.replay_starts_after_return += 1;
            }
        }
        let _ = self.inner.replay_started.try_send(());
        let inner = Arc::clone(&self.inner);
        Ok(LlmReplayCall::new(
            async move {
                while !inner.replay_release.load(Ordering::SeqCst) {
                    inner.replay_release_notify.notified().await;
                }
                if inner.replay_aborted.load(Ordering::SeqCst) {
                    Err(FlowError::Internal("internal replay aborted".to_string()))
                } else {
                    Ok(request.content)
                }
            },
            || {},
        ))
    }
}

struct InternalReplayShutdownHarness {
    inner: Arc<InternalReplayShutdownInner>,
    replay_started: Receiver<()>,
    provider_paused: Receiver<()>,
    intake_stopped: Receiver<()>,
}

impl InternalReplayShutdownHarness {
    fn schedule(&self, role: LlmCallRole) -> bool {
        schedule_internal_replay_call(&self.inner, role, false)
    }

    fn schedule_paused_before_replay_start(&self, role: LlmCallRole) -> bool {
        schedule_internal_replay_call(&self.inner, role, true)
    }

    fn wait_for_first_replay_start(&self) {
        self.replay_started
            .recv_timeout(Duration::from_secs(2))
            .expect("managed internal replay did not start");
        let state = self.inner.state.lock().unwrap();
        assert_eq!(state.managed_calls, 1);
        assert_eq!(state.replay_starts, 1);
    }

    fn wait_for_second_provider_pause(&self) {
        self.provider_paused
            .recv_timeout(Duration::from_secs(2))
            .expect("second managed internal provider did not pause before replay start");
        let state = self.inner.state.lock().unwrap();
        assert_eq!(state.managed_calls, 2);
        assert_eq!(state.replay_starts, 1);
    }

    fn wait_for_intake_stop(&self) {
        self.intake_stopped
            .recv_timeout(Duration::from_secs(2))
            .expect("plugin teardown did not stop internal replay intake");
    }

    fn complete_replay(&self) {
        release_internal_replay(&self.inner, false);
    }

    fn poised_contender(
        &self,
        role: LlmCallRole,
    ) -> (SyncSender<()>, std::thread::JoinHandle<bool>) {
        let (release, wait) = sync_channel(1);
        let inner = Arc::clone(&self.inner);
        let contender = std::thread::spawn(move || {
            wait.recv().expect("contender release sender dropped");
            schedule_internal_replay_call(&inner, role, false)
        });
        (release, contender)
    }

    fn assert_closed_at_return(
        &self,
        release_contender: SyncSender<()>,
        contender: std::thread::JoinHandle<bool>,
    ) {
        {
            let mut state = self.inner.state.lock().unwrap();
            state.after_return = true;
            assert!(!state.intake_open);
            assert_eq!(state.managed_calls, 2);
            assert_eq!(state.replay_starts, 1);
            assert_eq!(state.completed_workers, 2);
            assert!(state.workers.is_empty());
        }
        release_contender.send(()).unwrap();
        assert!(!contender.join().unwrap());
        let state = self.inner.state.lock().unwrap();
        assert_eq!(state.managed_calls, 2);
        assert_eq!(state.replay_starts, 1);
        assert_eq!(state.managed_calls_after_return, 0);
        assert_eq!(state.replay_starts_after_return, 0);
    }
}

fn schedule_internal_replay_call(
    inner: &Arc<InternalReplayShutdownInner>,
    role: LlmCallRole,
    pause_before_replay_start: bool,
) -> bool {
    let mut state = inner.state.lock().unwrap();
    if !state.intake_open {
        return false;
    }
    state.managed_calls += 1;
    if state.after_return {
        state.managed_calls_after_return += 1;
    }

    let worker_inner = Arc::clone(inner);
    state.workers.push(std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let call_inner = Arc::clone(&worker_inner);
        runtime.block_on(
            TASK_SCOPE_STACK.scope(worker_inner.scope_stack.clone(), async move {
                let transport = Arc::new(InternalReplayShutdownTransport {
                    capability: LlmReplayCapability {
                        contract_version: LLM_REPLAY_CONTRACT_VERSION,
                        api_family: LlmApiFamily::OpenAIResponses,
                        transport_identity: "plugin-shutdown-race".to_string(),
                    },
                    inner: Arc::clone(&call_inner),
                });
                let replay = Arc::clone(&transport);
                let provider_inner = Arc::clone(&call_inner);
                let params = LlmCallExecuteV2Params::builder()
                    .name("plugin-shutdown-internal-replay")
                    .request(LlmRequest {
                        headers: Map::new(),
                        content: json!({"candidate": true}),
                    })
                    .func(Arc::new(move |request| {
                        let replay = Arc::clone(&replay);
                        let inner = Arc::clone(&provider_inner);
                        Box::pin(async move {
                            if pause_before_replay_start {
                                let _ = inner.provider_paused.try_send(());
                                while !inner.provider_start_release.load(Ordering::SeqCst) {
                                    inner.provider_start_notify.notified().await;
                                }
                            }
                            replay.start(request)?.await
                        })
                    }))
                    .api_family(LlmApiFamily::OpenAIResponses)
                    .call_role(role)
                    .sanitized_metadata(BTreeMap::from([(
                        "anchor_uuid".to_string(),
                        json!(call_inner.anchor_uuid.to_string()),
                    )]))
                    .parent(call_inner.evaluator.clone())
                    .build();
                let _ = llm_call_execute_v2(params).await;
            }),
        );
        {
            let mut state = worker_inner.state.lock().unwrap();
            state.completed_workers += 1;
        }
        worker_inner.worker_completion_notify.notify_one();
    }));
    true
}

fn close_internal_replay_intake(inner: &InternalReplayShutdownInner) {
    inner.state.lock().unwrap().intake_open = false;
    inner.provider_start_release.store(true, Ordering::SeqCst);
    inner.provider_start_notify.notify_one();
    let _ = inner.intake_stopped.try_send(());
}

fn release_internal_replay(inner: &InternalReplayShutdownInner, aborted: bool) {
    if aborted {
        inner.replay_aborted.store(true, Ordering::SeqCst);
    }
    inner.replay_release.store(true, Ordering::SeqCst);
    inner.replay_release_notify.notify_one();
}

async fn drain_internal_replay_workers(inner: Arc<InternalReplayShutdownInner>) -> Result<()> {
    loop {
        let completed = {
            let state = inner.state.lock().unwrap();
            state.completed_workers == state.managed_calls
        };
        if completed {
            return Ok(());
        }
        inner.worker_completion_notify.notified().await;
    }
}

fn join_internal_replay_workers(inner: &InternalReplayShutdownInner) -> Result<()> {
    if !inner.replay_release.load(Ordering::SeqCst) {
        release_internal_replay(inner, true);
    }
    let workers = std::mem::take(&mut inner.state.lock().unwrap().workers);
    for worker in workers {
        worker.join().map_err(|_| {
            PluginError::Internal("managed internal replay worker panicked".to_string())
        })?;
    }
    Ok(())
}

fn install_internal_replay_shutdown_race(role: LlmCallRole) -> InternalReplayShutdownHarness {
    let scope_stack = create_scope_stack();
    let evaluator = with_scope_stack(scope_stack.clone(), || {
        push_scope(
            PushScopeParams::builder()
                .name("plugin-shutdown-evaluator")
                .scope_type(ScopeType::Evaluator)
                .build(),
        )
        .unwrap()
    });
    let (replay_started_sender, replay_started) = sync_channel(1);
    let (provider_paused_sender, provider_paused) = sync_channel(1);
    let (intake_stopped_sender, intake_stopped) = sync_channel(1);
    let inner = Arc::new(InternalReplayShutdownInner {
        state: Mutex::new(InternalReplayShutdownState {
            intake_open: true,
            ..InternalReplayShutdownState::default()
        }),
        scope_stack,
        evaluator,
        anchor_uuid: uuid::Uuid::now_v7(),
        replay_release: AtomicBool::new(false),
        replay_aborted: AtomicBool::new(false),
        replay_release_notify: tokio::sync::Notify::new(),
        provider_start_release: AtomicBool::new(false),
        provider_start_notify: tokio::sync::Notify::new(),
        worker_completion_notify: tokio::sync::Notify::new(),
        replay_started: replay_started_sender,
        provider_paused: provider_paused_sender,
        intake_stopped: intake_stopped_sender,
    });

    let deregister_inner = Arc::clone(&inner);
    let stop_inner = Arc::clone(&inner);
    let drain_inner = Arc::clone(&inner);
    let abort_inner = Arc::clone(&inner);
    install_active_registrations(vec![PluginRegistration::with_shutdown(
        "test",
        "managed-internal-replay",
        Box::new(move || join_internal_replay_workers(&deregister_inner)),
        Box::new(move || {
            close_internal_replay_intake(&stop_inner);
            Ok(())
        }),
        Box::new(move |_deadline| {
            let inner = Arc::clone(&drain_inner);
            Box::pin(async move { drain_internal_replay_workers(inner).await })
        }),
        Box::new(move || {
            close_internal_replay_intake(&abort_inner);
            release_internal_replay(&abort_inner, true);
            Ok(())
        }),
    )]);

    let harness = InternalReplayShutdownHarness {
        inner,
        replay_started,
        provider_paused,
        intake_stopped,
    };
    assert!(harness.schedule(role));
    harness.wait_for_first_replay_start();
    harness
}

static RECORDED_NAMES: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
static PARTIAL_FAIL_ROLLBACKS: AtomicUsize = AtomicUsize::new(0);
static RESTORE_FAIL_REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);
static RESTORE_BREAK_REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);
static REPLACEMENT_REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);
static ACTIVATION_CANCEL_REGISTRATIONS: AtomicUsize = AtomicUsize::new(0);
static ACTIVATION_CANCEL_ROLLBACKS: AtomicUsize = AtomicUsize::new(0);
static ACTIVATION_CANCEL_PENDING_ROLLBACKS: AtomicUsize = AtomicUsize::new(0);

fn recorded_names() -> &'static Mutex<Vec<String>> {
    RECORDED_NAMES.get_or_init(|| Mutex::new(Vec::new()))
}

fn lock_runtime_owner() -> std::sync::MutexGuard<'static, ()> {
    crate::shared_runtime::runtime_owner_test_mutex()
        .lock()
        .unwrap_or_else(|err| err.into_inner())
}

fn expect_registration_failed(result: Result<()>, message_fragment: &str) {
    match result {
        Err(PluginError::RegistrationFailed(message)) => {
            assert!(message.contains(message_fragment), "{message}");
        }
        Err(other) => panic!("unexpected registration failure: {other}"),
        Ok(_) => panic!("expected registration to fail"),
    }
}

fn set_conflicting_runtime_owner_for_tests() {
    unsafe {
        std::env::set_var(
            "NEMO_RELAY_RUNTIME_OWNER",
            format!(
                "pid={};binding=python;version={}",
                std::process::id(),
                env!("CARGO_PKG_VERSION")
            ),
        )
    };
}

impl Plugin for TestPlugin {
    fn plugin_kind(&self) -> &str {
        "test.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![ConfigDiagnostic {
            level: DiagnosticLevel::Warning,
            code: "test.warning".into(),
            component: Some("test.plugin".into()),
            field: None,
            message: "validated".into(),
        }]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            ctx.register_llm_request_intercept(
                "intercept",
                1,
                false,
                Arc::new(|_name, mut request, annotated| {
                    request.headers.insert("x-plugin".into(), json!(true));
                    Ok(LlmRequestInterceptOutcome::new(request, annotated))
                }),
            )
        })
    }
}

impl Plugin for SingletonPlugin {
    fn plugin_kind(&self) -> &str {
        "singleton.plugin"
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        _ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

impl Plugin for RecordingPlugin {
    fn plugin_kind(&self) -> &str {
        "recording.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let qualified = ctx.qualify_name("subscriber");
        recorded_names().lock().unwrap().push(qualified.clone());
        Box::pin(async move {
            ctx.add_registration(PluginRegistration::new(
                "plugin",
                qualified,
                Box::new(|| Ok(())),
            ));
            Ok(())
        })
    }
}

impl Plugin for ReplacementPlugin {
    fn plugin_kind(&self) -> &str {
        "replacement.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![ConfigDiagnostic {
            level: DiagnosticLevel::Warning,
            code: "replacement.warning".into(),
            component: Some("replacement.plugin".into()),
            field: None,
            message: "replacement validated".into(),
        }]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            REPLACEMENT_REGISTRATIONS.fetch_add(1, Ordering::SeqCst);
            ctx.add_registration(PluginRegistration::new(
                "plugin",
                ctx.qualify_name("replacement"),
                Box::new(|| Ok(())),
            ));
            Ok(())
        })
    }
}

impl Plugin for RestoreFailPlugin {
    fn plugin_kind(&self) -> &str {
        "restore.fail.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            RESTORE_FAIL_REGISTRATIONS.fetch_add(1, Ordering::SeqCst);
            ctx.add_registration(PluginRegistration::new(
                "plugin",
                ctx.qualify_name("restore-fail"),
                Box::new(|| Ok(())),
            ));
            Err(PluginError::RegistrationFailed(
                "restore.fail.plugin refused to initialize".into(),
            ))
        })
    }
}

impl Plugin for RestoreBreakPlugin {
    fn plugin_kind(&self) -> &str {
        "restore.break.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            if RESTORE_BREAK_REGISTRATIONS.fetch_add(1, Ordering::SeqCst) == 0 {
                ctx.add_registration(PluginRegistration::new(
                    "plugin",
                    ctx.qualify_name("restore-break"),
                    Box::new(|| Ok(())),
                ));
                Ok(())
            } else {
                Err(PluginError::RegistrationFailed(
                    "restore.break.plugin refused to restore".into(),
                ))
            }
        })
    }
}

impl Plugin for PartialFailPlugin {
    fn plugin_kind(&self) -> &str {
        "partial.fail.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            ctx.add_registration(PluginRegistration::new(
                "plugin",
                ctx.qualify_name("partial-fail"),
                Box::new(|| {
                    PARTIAL_FAIL_ROLLBACKS.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }),
            ));
            Err(PluginError::RegistrationFailed(
                "partial.fail.plugin refused to finish initialization".into(),
            ))
        })
    }
}

impl Plugin for VanishingPlugin {
    fn plugin_kind(&self) -> &str {
        "vanishing.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        let _ = deregister_plugin("vanishing.plugin");
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        _ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async { Ok(()) })
    }
}

impl Plugin for ActivationCancelPlugin {
    fn plugin_kind(&self) -> &str {
        "activation.cancel.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            let registration = ACTIVATION_CANCEL_REGISTRATIONS.fetch_add(1, Ordering::SeqCst);
            ctx.add_registration(
                PluginRegistration::new(
                    "plugin",
                    ctx.qualify_name("cancellation-test"),
                    Box::new(|| {
                        ACTIVATION_CANCEL_ROLLBACKS.fetch_add(1, Ordering::SeqCst);
                        Ok(())
                    }),
                )
                .with_activation_rollback(Box::new(|| {
                    ACTIVATION_CANCEL_PENDING_ROLLBACKS.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                })),
            );
            if registration == 1 {
                std::future::pending::<()>().await;
            }
            Ok(())
        })
    }
}

impl Plugin for ActivationCommitRacePlugin {
    fn plugin_kind(&self) -> &str {
        "activation.commit.race.plugin"
    }

    fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        vec![]
    }

    fn register<'a>(
        &'a self,
        _plugin_config: &Map<String, Json>,
        ctx: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let barrier = self.barrier.clone();
        let candidate_rollbacks = self.candidate_rollbacks.clone();
        Box::pin(async move {
            ctx.add_registration(PluginRegistration::new(
                "plugin",
                ctx.qualify_name("candidate"),
                Box::new(move || {
                    candidate_rollbacks.fetch_add(1, Ordering::SeqCst);
                    Ok(())
                }),
            ));
            barrier.wait();
            barrier.wait();
            Ok(())
        })
    }
}

fn reset_global() {
    crate::shared_runtime::reset_runtime_owner_for_tests();
    FAILED_PLUGIN_DEREGISTRATIONS
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clear();
    let ctx = global_context();
    let mut state = ctx.write().unwrap();
    *state = NemoRelayContextState::new();
    clear_plugin_configuration().unwrap();
    recorded_names().lock().unwrap().clear();
    PARTIAL_FAIL_ROLLBACKS.store(0, Ordering::SeqCst);
    RESTORE_FAIL_REGISTRATIONS.store(0, Ordering::SeqCst);
    RESTORE_BREAK_REGISTRATIONS.store(0, Ordering::SeqCst);
    REPLACEMENT_REGISTRATIONS.store(0, Ordering::SeqCst);
    ACTIVATION_CANCEL_REGISTRATIONS.store(0, Ordering::SeqCst);
    ACTIVATION_CANCEL_ROLLBACKS.store(0, Ordering::SeqCst);
    ACTIVATION_CANCEL_PENDING_ROLLBACKS.store(0, Ordering::SeqCst);
    let _ = deregister_plugin("test.plugin");
    let _ = deregister_plugin("singleton.plugin");
    let _ = deregister_plugin("recording.plugin");
    let _ = deregister_plugin("replacement.plugin");
    let _ = deregister_plugin("restore.fail.plugin");
    let _ = deregister_plugin("restore.break.plugin");
    let _ = deregister_plugin("partial.fail.plugin");
    let _ = deregister_plugin("vanishing.plugin");
    let _ = deregister_plugin("activation.cancel.plugin");
}

fn install_active_registrations(registrations: Vec<PluginRegistration>) {
    *ACTIVE_PLUGIN_CONFIGURATION.lock().unwrap() = Some(ActivePluginConfiguration {
        config: PluginConfig::default(),
        report: ConfigReport::default(),
        registrations,
    });
}

#[test]
fn test_layer_config_overlay_wins() {
    // The overlay is the higher-precedence layer: it overrides shared component fields, deep-merges
    // nested config objects, replaces arrays, appends overlay-only kinds, preserves base-only kinds,
    // replaces top-level scalars, and recursively merges top-level objects (policy).
    let base = json!({
        "version": 1,
        "components": [
            {
                "kind": "alpha",
                "enabled": true,
                "config": { "keep": "base", "override": "base", "nested": {"a": 1, "b": 2}, "list": [1, 2, 3] }
            },
            { "kind": "base_only", "enabled": true, "config": {} }
        ],
        "policy": { "unknown_component": "warn", "unknown_field": "warn" }
    });
    let overlay = json!({
        "version": 2,
        "components": [
            {
                "kind": "alpha",
                "enabled": false,
                "config": { "override": "overlay", "added": true, "nested": {"b": 20, "c": 30}, "list": [9] }
            },
            { "kind": "overlay_only", "enabled": true, "config": {} }
        ],
        "policy": { "unknown_component": "error" }
    });

    let mut merged = base;
    layer_config(&mut merged, overlay);
    let components = merged["components"].as_array().unwrap();

    // Ordering: base components first (in base order), then overlay-only components appended.
    let kinds: Vec<&str> = components
        .iter()
        .map(|component| component["kind"].as_str().unwrap())
        .collect();
    assert_eq!(kinds, ["alpha", "base_only", "overlay_only"]);

    let alpha = &components[0];
    assert_eq!(alpha["enabled"], json!(false), "overlay enabled wins");
    assert_eq!(
        alpha["config"]["keep"],
        json!("base"),
        "base-only key preserved"
    );
    assert_eq!(
        alpha["config"]["override"],
        json!("overlay"),
        "overlay scalar wins"
    );
    assert_eq!(alpha["config"]["added"], json!(true), "overlay key added");
    assert_eq!(
        alpha["config"]["nested"],
        json!({"a": 1, "b": 20, "c": 30}),
        "nested objects merge recursively"
    );
    assert_eq!(
        alpha["config"]["list"],
        json!([9]),
        "arrays are replaced, not merged"
    );

    // Base-only component is preserved.
    assert_eq!(components[1]["kind"], json!("base_only"));

    // Top-level scalars are replaced by the overlay; objects (policy) merge recursively.
    assert_eq!(merged["version"], json!(2));
    assert_eq!(merged["policy"]["unknown_component"], json!("error"));
    assert_eq!(
        merged["policy"]["unknown_field"],
        json!("warn"),
        "base-only policy field preserved"
    );
}

#[test]
fn test_layer_config_preserves_multi_instance_kinds() {
    // A kind used more than once (multi-instance plugins) must not collapse into the first slot.
    let base = json!({ "components": [ { "kind": "multi", "config": { "n": 0 } } ] });
    let overlay = json!({
        "components": [
            { "kind": "multi", "config": { "n": 1 } },
            { "kind": "multi", "config": { "tag": "second" } }
        ]
    });

    let mut merged = base;
    layer_config(&mut merged, overlay);
    let components = merged["components"].as_array().unwrap();

    // First overlay instance pairs with the base instance; the second is appended, not dropped.
    assert_eq!(components.len(), 2);
    assert!(
        components
            .iter()
            .all(|component| component["kind"] == json!("multi"))
    );
    assert_eq!(components[0]["config"]["n"], json!(1));
    assert_eq!(components[1]["config"]["tag"], json!("second"));
}

#[test]
fn test_config_report_has_errors() {
    let report = ConfigReport {
        diagnostics: vec![ConfigDiagnostic {
            level: DiagnosticLevel::Error,
            code: "x".into(),
            component: None,
            field: None,
            message: "boom".into(),
        }],
    };
    assert!(report.has_errors());
}

#[test]
fn test_register_and_deregister_plugin() {
    let _guard = lock_runtime_owner();
    reset_global();
    let plugin: Arc<dyn Plugin> = Arc::new(TestPlugin);
    let foreign_instance: Arc<dyn Plugin> = Arc::new(TestPlugin);
    assert!(register_plugin(Arc::clone(&plugin)).is_ok());
    match register_plugin(Arc::new(TestPlugin)) {
        Err(PluginError::RegistrationFailed(message)) => {
            assert!(message.contains("already registered"));
        }
        Err(other) => panic!("unexpected duplicate-registration error: {other}"),
        Ok(_) => panic!("expected duplicate registration to fail"),
    }
    assert!(list_plugin_kinds().contains(&"test.plugin".to_string()));
    let registered = lookup_plugin("test.plugin").expect("test plugin should be registered");
    assert!(Arc::ptr_eq(&registered, &plugin));
    assert!(!deregister_plugin_if(&foreign_instance));
    assert!(deregister_plugin_if(&plugin));
    assert!(!deregister_plugin_if(&plugin));
    assert!(!deregister_plugin("missing.plugin"));
    assert!(clear_plugin_configuration().is_ok());
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_plugin_registration_context_registers_and_rolls_back() {
    let _guard = lock_runtime_owner();
    reset_global();

    let mut ctx = PluginRegistrationContext::new();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(TestPlugin.register(&Map::new(), &mut ctx))
        .unwrap();

    let request = llm_request_intercepts(
        "model",
        LlmRequest {
            headers: Map::new(),
            content: json!({"messages": []}),
        },
    )
    .unwrap();
    assert_eq!(request.request.headers.get("x-plugin"), Some(&json!(true)));

    let mut registrations = ctx.into_registrations();
    rollback_registrations(&mut registrations);

    let request = llm_request_intercepts(
        "model",
        LlmRequest {
            headers: Map::new(),
            content: json!({"messages": []}),
        },
    )
    .unwrap();
    assert_eq!(request.request.headers.get("x-plugin"), None);
    reset_global();
}

#[test]
fn test_initialize_plugins_registers_and_clears_components() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(TestPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let report = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("test.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();
    assert!(!report.has_errors());
    assert!(active_plugin_report().is_some());

    let request = llm_request_intercepts(
        "model",
        LlmRequest {
            headers: Map::new(),
            content: json!({"messages": []}),
        },
    )
    .unwrap();
    assert_eq!(request.request.headers.get("x-plugin"), Some(&json!(true)));

    clear_plugin_configuration().unwrap();
    let request = llm_request_intercepts(
        "model",
        LlmRequest {
            headers: Map::new(),
            content: json!({"messages": []}),
        },
    )
    .unwrap();
    assert_eq!(request.request.headers.get("x-plugin"), None);
    reset_global();
}

#[test]
fn test_validate_plugin_config_honors_policy_and_duplicate_singletons() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(SingletonPlugin)).unwrap();

    let report = validate_plugin_config(&PluginConfig {
        components: vec![
            PluginComponentSpec::new("singleton.plugin"),
            PluginComponentSpec::new("singleton.plugin"),
            PluginComponentSpec::new("missing.plugin"),
        ],
        policy: ConfigPolicy {
            unknown_component: UnsupportedBehavior::Warn,
            unknown_field: UnsupportedBehavior::Ignore,
            unsupported_value: UnsupportedBehavior::Error,
        },
        ..PluginConfig::default()
    });

    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "plugin.duplicate_component")
    );
    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "plugin.unknown_component"
                && diag.level == DiagnosticLevel::Warning)
    );

    let ignored = validate_plugin_config(&PluginConfig {
        components: vec![PluginComponentSpec::new("still.missing")],
        policy: ConfigPolicy {
            unknown_component: UnsupportedBehavior::Ignore,
            ..PluginConfig::default().policy
        },
        ..PluginConfig::default()
    });
    assert!(ignored.diagnostics.is_empty());

    reset_global();
}

#[test]
fn test_plugin_config_defaults_debug_and_invalid_config_messages() {
    let _guard = lock_runtime_owner();
    reset_global();

    let config: PluginConfig = serde_json::from_value(json!({})).unwrap();
    assert_eq!(config.version, 1);
    assert!(config.components.is_empty());
    assert_eq!(config.policy.unknown_component, UnsupportedBehavior::Warn);
    assert_eq!(config.policy.unknown_field, UnsupportedBehavior::Warn);
    assert_eq!(config.policy.unsupported_value, UnsupportedBehavior::Error);

    let component: PluginComponentSpec =
        serde_json::from_value(json!({"kind": "demo.plugin"})).unwrap();
    assert_eq!(component.kind, "demo.plugin");
    assert!(component.enabled);
    assert!(component.config.is_empty());

    let registration = PluginRegistration::new("plugin", "demo::registration", Box::new(|| Ok(())));
    let debug = format!("{registration:?}");
    assert!(debug.contains("PluginRegistration"));
    assert!(debug.contains("demo::registration"));
    assert!(debug.contains("plugin"));

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let error = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            version: 2,
            components: vec![PluginComponentSpec::new("missing.plugin")],
            policy: ConfigPolicy {
                unknown_component: UnsupportedBehavior::Error,
                ..PluginConfig::default().policy
            },
        }))
        .unwrap_err();

    match error {
        PluginError::InvalidConfig(message) => {
            assert!(message.contains("plugin config version 2 is unsupported"));
            assert!(message.contains("plugin component kind 'missing.plugin' is unsupported"));
            assert!(message.contains(";"));
        }
        other => panic!("unexpected invalid config error: {other}"),
    }

    reset_global();
}

#[test]
fn test_plugin_helper_defaults_and_policy_diagnostics() {
    let _guard = lock_runtime_owner();
    reset_global();

    assert_eq!(default_warn(), UnsupportedBehavior::Warn);
    assert_eq!(default_error(), UnsupportedBehavior::Error);
    assert_eq!(default_plugin_config_version(), 1);
    assert!(default_enabled());
    assert_eq!(UnsupportedBehavior::default(), UnsupportedBehavior::Warn);

    let mut diagnostics = Vec::new();
    push_policy_diag(
        &mut diagnostics,
        UnsupportedBehavior::Ignore,
        "ignored.code",
        None,
        None,
        "ignored".into(),
    );
    assert!(diagnostics.is_empty());

    push_policy_diag(
        &mut diagnostics,
        UnsupportedBehavior::Warn,
        "warn.code",
        Some("warn.plugin".into()),
        Some("field".into()),
        "warn".into(),
    );
    push_policy_diag(
        &mut diagnostics,
        UnsupportedBehavior::Error,
        "error.code",
        Some("error.plugin".into()),
        None,
        "error".into(),
    );

    assert_eq!(diagnostics.len(), 2);
    assert_eq!(diagnostics[0].level, DiagnosticLevel::Warning);
    assert_eq!(diagnostics[0].component.as_deref(), Some("warn.plugin"));
    assert_eq!(diagnostics[0].field.as_deref(), Some("field"));
    assert_eq!(diagnostics[1].level, DiagnosticLevel::Error);
    assert_eq!(join_error_messages(&ConfigReport { diagnostics }), "error");

    reset_global();
}

#[test]
fn test_plugin_component_helpers_and_serialization_error_variant() {
    let _guard = lock_runtime_owner();
    reset_global();

    let config = PluginConfig {
        components: vec![
            PluginComponentSpec::new("alpha.plugin"),
            PluginComponentSpec::new("beta.plugin"),
            PluginComponentSpec::new("alpha.plugin"),
        ],
        ..PluginConfig::default()
    };

    let totals = plugin_component_totals(&config);
    assert_eq!(totals.get("alpha.plugin"), Some(&2));
    assert_eq!(totals.get("beta.plugin"), Some(&1));
    assert_eq!(
        component_namespace("alpha.plugin", 1, totals["alpha.plugin"]),
        "__nemo_relay_plugin__alpha.plugin__1__"
    );
    assert_eq!(
        component_namespace("beta.plugin", 1, totals["beta.plugin"]),
        "__nemo_relay_plugin__beta.plugin__"
    );

    let parse_error = serde_json::from_str::<PluginConfig>("{").unwrap_err();
    let wrapped: PluginError = parse_error.into();
    match wrapped {
        PluginError::Serialization(message) => {
            assert!(!message.to_string().is_empty());
        }
        other => panic!("unexpected conversion result: {other}"),
    }

    reset_global();
}

#[test]
fn test_registration_context_namespace_and_manual_registration_helpers() {
    let mut ctx = PluginRegistrationContext::with_namespace("demo::");
    assert_eq!(ctx.qualify_name("subscriber"), "demo::subscriber");

    ctx.add_registration(PluginRegistration::new(
        "plugin",
        "demo::manual".to_string(),
        Box::new(|| Ok(())),
    ));
    ctx.extend_registrations(vec![PluginRegistration::new(
        "plugin",
        "demo::extra".to_string(),
        Box::new(|| Ok(())),
    )]);

    let names = ctx
        .into_registrations()
        .into_iter()
        .map(|registration| registration.name)
        .collect::<Vec<_>>();
    assert_eq!(names, vec!["demo::manual", "demo::extra"]);
}

#[test]
fn test_plugin_registration_context_covers_all_registration_helpers() {
    let _guard = lock_runtime_owner();
    reset_global();

    let mut ctx = PluginRegistrationContext::with_namespace("demo::");
    ctx.register_subscriber("subscriber", Arc::new(|_event| {}))
        .unwrap();
    ctx.register_tool_request_intercept("tool-request", 1, false, Arc::new(|_name, args| Ok(args)))
        .unwrap();
    ctx.register_tool_execution_intercept(
        "tool-exec",
        1,
        Arc::new(|_name, args, _next| Box::pin(async move { Ok(args.into()) })),
    )
    .unwrap();
    ctx.register_llm_request_intercept(
        "llm-request",
        1,
        false,
        Arc::new(|_name, request, annotated| {
            Ok(LlmRequestInterceptOutcome::new(request, annotated))
        }),
    )
    .unwrap();
    ctx.register_llm_execution_intercept(
        "llm-exec",
        1,
        Arc::new(|_name, request, _next| Box::pin(async move { Ok(request.content) })),
    )
    .unwrap();
    ctx.register_llm_execution_intercept_v2(
        "llm-exec-v2",
        1,
        Arc::new(|_name, _context, request, _replay, _next| {
            Box::pin(async move { Ok(request.content) })
        }),
    )
    .unwrap();
    ctx.register_llm_stream_execution_intercept(
        "llm-stream",
        1,
        Arc::new(|_name, request, _next| {
            Box::pin(async move {
                Ok(Box::pin(tokio_stream::iter(vec![Ok(request.content)]))
                    as Pin<
                        Box<dyn tokio_stream::Stream<Item = crate::error::Result<Json>> + Send>,
                    >)
            })
        }),
    )
    .unwrap();

    let mut registrations = ctx.into_registrations();
    let names = registrations
        .iter()
        .map(|registration| registration.name.clone())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "demo::subscriber",
            "demo::tool-request",
            "demo::tool-exec",
            "demo::llm-request",
            "demo::llm-exec",
            "demo::llm-exec-v2",
            "demo::llm-stream",
        ]
    );

    rollback_registrations(&mut registrations);
    assert!(registrations.is_empty());
    reset_global();
}

#[test]
fn test_rollback_registrations_runs_in_reverse_and_ignores_failures() {
    let _guard = lock_runtime_owner();
    reset_global();
    let mut registrations = vec![];
    let call_order = Arc::new(Mutex::new(Vec::new()));
    let second_attempts = Arc::new(AtomicUsize::new(0));

    let first_order = Arc::clone(&call_order);
    registrations.push(PluginRegistration::new(
        "plugin",
        "first",
        Box::new(move || {
            first_order.lock().unwrap().push("first");
            Ok(())
        }),
    ));

    let second_order = Arc::clone(&call_order);
    let attempts = Arc::clone(&second_attempts);
    registrations.push(PluginRegistration::new(
        "plugin",
        "second",
        Box::new(move || {
            second_order.lock().unwrap().push("second");
            if attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(PluginError::RegistrationFailed(
                    "expected rollback failure".into(),
                ))
            } else {
                Ok(())
            }
        }),
    ));

    rollback_registrations(&mut registrations);

    assert!(registrations.is_empty());
    assert_eq!(*call_order.lock().unwrap(), vec!["second", "first"]);
    retry_failed_plugin_deregistrations().unwrap();
    assert_eq!(
        *call_order.lock().unwrap(),
        vec!["second", "first", "second"]
    );
    reset_global();
}

#[test]
fn test_pending_activation_rollback_hook_runs_before_shutdown() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let rollback_order = Arc::clone(&order);
    let mut registrations = vec![
        shutdown_test_registration(
            "pending-activation",
            Arc::clone(&order),
            TestDrainBehavior::Success,
        )
        .with_activation_rollback(Box::new(move || {
            rollback_order
                .lock()
                .unwrap()
                .push("activation-rollback:pending-activation".to_string());
            Ok(())
        })),
    ];

    rollback_registrations(&mut registrations);

    assert_eq!(
        *order.lock().unwrap(),
        [
            "activation-rollback:pending-activation",
            "stop:pending-activation",
            "abort:pending-activation",
            "deregister:pending-activation",
        ]
    );
}

#[test]
fn test_activation_commit_consumes_pending_rollback_hook() {
    let _guard = lock_runtime_owner();
    reset_global();
    let rollback_calls = Arc::new(AtomicUsize::new(0));
    let captured = Arc::clone(&rollback_calls);
    let registration = PluginRegistration::new("test", "committed", Box::new(|| Ok(())))
        .with_activation_rollback(Box::new(move || {
            captured.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }));

    store_active_plugin_configuration(
        PluginConfig::default(),
        ConfigReport::default(),
        vec![registration],
    )
    .unwrap();
    clear_plugin_configuration().unwrap();

    assert_eq!(rollback_calls.load(Ordering::SeqCst), 0);
    reset_global();
}

#[test]
fn test_dropping_unconsumed_registration_context_rolls_back() {
    let rollbacks = Arc::new(AtomicUsize::new(0));
    let captured = Arc::clone(&rollbacks);
    {
        let mut context = PluginRegistrationContext::new();
        context.add_registration(PluginRegistration::new(
            "test",
            "drop-owned",
            Box::new(move || {
                captured.fetch_add(1, Ordering::SeqCst);
                Ok(())
            }),
        ));
    }
    assert_eq!(rollbacks.load(Ordering::SeqCst), 1);
}

#[test]
fn test_immediate_teardown_stops_then_aborts_and_deregisters_in_reverse() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut registrations = vec![
        shutdown_test_registration("first", Arc::clone(&order), TestDrainBehavior::Success),
        shutdown_test_registration("second", Arc::clone(&order), TestDrainBehavior::Success),
    ];

    teardown_registrations_immediate(&mut registrations, false).unwrap();

    assert!(registrations.is_empty());
    assert_eq!(
        *order.lock().unwrap(),
        [
            "stop:first",
            "stop:second",
            "abort:second",
            "abort:first",
            "deregister:second",
            "deregister:first",
        ]
    );
}

#[test]
fn test_immediate_teardown_contains_panics_and_aggregates_all_hook_errors() {
    let mut registrations = vec![PluginRegistration::with_shutdown(
        "test",
        "failing",
        Box::new(|| panic!("deregister panic")),
        Box::new(|| panic!("stop panic")),
        Box::new(|_deadline| Box::pin(async { Ok(()) })),
        Box::new(|| Err(PluginError::Internal("abort error".into()))),
    )];

    let error = teardown_registrations_immediate(&mut registrations, false).unwrap_err();

    assert_eq!(registrations.len(), 1);
    let message = error.to_string();
    assert!(message.contains("stop_intake failed: hook panicked"));
    assert!(message.contains("abort failed: internal error: abort error"));
    assert!(message.contains("deregister failed: hook panicked"));
}

#[test]
fn test_async_teardown_drains_and_deregisters_in_reverse_without_abort() {
    let _guard = lock_runtime_owner();
    reset_global();
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut registrations = vec![
        shutdown_test_registration("first", Arc::clone(&order), TestDrainBehavior::Success),
        shutdown_test_registration("second", Arc::clone(&order), TestDrainBehavior::Success),
    ];
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime
        .block_on(teardown_registrations_async(
            &mut registrations,
            Instant::now() + Duration::from_secs(1),
        ))
        .unwrap();

    assert!(registrations.is_empty());
    assert_eq!(
        *order.lock().unwrap(),
        [
            "stop:first",
            "stop:second",
            "drain:second",
            "drain:first",
            "deregister:second",
            "deregister:first",
        ]
    );
    reset_global();
}

#[test]
fn test_async_teardown_aggregates_drain_errors_and_aborts_only_unfinished() {
    let _guard = lock_runtime_owner();
    reset_global();
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut registrations = vec![
        shutdown_test_registration("success", Arc::clone(&order), TestDrainBehavior::Success),
        shutdown_test_registration("panic", Arc::clone(&order), TestDrainBehavior::Panic),
        shutdown_test_registration("error", Arc::clone(&order), TestDrainBehavior::Error),
    ];
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let error = runtime
        .block_on(teardown_registrations_async(
            &mut registrations,
            Instant::now() + Duration::from_secs(1),
        ))
        .unwrap_err();

    assert!(registrations.is_empty());
    let message = error.to_string();
    assert!(message.contains("error' drain failed: internal error: error drain error"));
    assert!(message.contains("panic' drain failed: future panicked"));
    assert_eq!(
        *order.lock().unwrap(),
        [
            "stop:success",
            "stop:panic",
            "stop:error",
            "drain:error",
            "drain:panic",
            "drain:success",
            "abort:error",
            "abort:panic",
            "deregister:error",
            "deregister:panic",
            "deregister:success",
        ]
    );
    reset_global();
}

#[test]
fn test_async_teardown_timeout_aborts_timed_out_and_unattempted_registrations() {
    let _guard = lock_runtime_owner();
    reset_global();
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut registrations = vec![
        shutdown_test_registration(
            "unattempted",
            Arc::clone(&order),
            TestDrainBehavior::Success,
        ),
        shutdown_test_registration("pending", Arc::clone(&order), TestDrainBehavior::Pending),
    ];
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let error = runtime
        .block_on(teardown_registrations_async(
            &mut registrations,
            Instant::now() + Duration::from_millis(20),
        ))
        .unwrap_err();

    assert!(error.to_string().contains("shared deadline expired"));
    assert!(registrations.is_empty());
    assert_eq!(
        *order.lock().unwrap(),
        [
            "stop:unattempted",
            "stop:pending",
            "drain:pending",
            "abort:pending",
            "abort:unattempted",
            "deregister:pending",
            "deregister:unattempted",
        ]
    );
    reset_global();
}

#[test]
fn test_async_clear_timeout_aborts_before_deregister_and_leaves_state_empty() {
    let _guard = lock_runtime_owner();
    reset_global();
    let order = Arc::new(Mutex::new(Vec::new()));
    install_active_registrations(vec![shutdown_test_registration(
        "pending",
        Arc::clone(&order),
        TestDrainBehavior::Pending,
    )]);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let error = runtime
        .block_on(clear_plugin_configuration_async(Duration::from_millis(20)))
        .unwrap_err();

    assert!(error.to_string().contains("shared deadline expired"));
    assert_eq!(
        *order.lock().unwrap(),
        [
            "stop:pending",
            "drain:pending",
            "abort:pending",
            "deregister:pending",
        ]
    );
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_failed_deregistration_is_retained_and_retried_before_activation() {
    let _guard = lock_runtime_owner();
    reset_global();
    let attempts = Arc::new(AtomicUsize::new(0));
    let captured_attempts = Arc::clone(&attempts);
    install_active_registrations(vec![PluginRegistration::new(
        "test",
        "retry-cleanup",
        Box::new(move || {
            if captured_attempts.fetch_add(1, Ordering::SeqCst) == 0 {
                Err(PluginError::Internal("transient cleanup failure".into()))
            } else {
                Ok(())
            }
        }),
    )]);

    let error = clear_plugin_configuration().unwrap_err();
    assert!(error.to_string().contains("transient cleanup failure"));
    assert_eq!(attempts.load(Ordering::SeqCst), 1);
    assert!(active_plugin_report().is_none());
    assert_eq!(FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().len(), 1);

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(initialize_plugins_exact(PluginConfig::default()))
        .unwrap();
    assert_eq!(attempts.load(Ordering::SeqCst), 2);
    assert!(FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().is_empty());
    assert!(active_plugin_report().is_some());
    clear_plugin_configuration().unwrap();
    reset_global();
}

#[test]
fn test_unresolved_retained_cleanup_does_not_prevent_sync_clear() {
    let _guard = lock_runtime_owner();
    reset_global();
    let retained_attempts = Arc::new(AtomicUsize::new(0));
    let captured_retained_attempts = Arc::clone(&retained_attempts);
    FAILED_PLUGIN_DEREGISTRATIONS
        .lock()
        .unwrap()
        .push(PluginRegistration::new(
            "test",
            "unresolved",
            Box::new(move || {
                captured_retained_attempts.fetch_add(1, Ordering::SeqCst);
                Err(PluginError::Internal("still unresolved".into()))
            }),
        ));
    let active_cleanups = Arc::new(AtomicUsize::new(0));
    let captured = Arc::clone(&active_cleanups);
    install_active_registrations(vec![PluginRegistration::new(
        "test",
        "active",
        Box::new(move || {
            captured.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }),
    )]);

    clear_plugin_configuration().unwrap();
    assert_eq!(active_cleanups.load(Ordering::SeqCst), 1);
    assert_eq!(retained_attempts.load(Ordering::SeqCst), 0);
    assert!(active_plugin_report().is_none());
    assert_eq!(FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().len(), 1);
    reset_global();
}

#[test]
fn test_unresolved_retained_cleanup_does_not_prevent_async_clear() {
    let _guard = lock_runtime_owner();
    reset_global();
    let retained_attempts = Arc::new(AtomicUsize::new(0));
    let captured_retained_attempts = Arc::clone(&retained_attempts);
    FAILED_PLUGIN_DEREGISTRATIONS
        .lock()
        .unwrap()
        .push(PluginRegistration::new(
            "test",
            "unresolved",
            Box::new(move || {
                captured_retained_attempts.fetch_add(1, Ordering::SeqCst);
                Err(PluginError::Internal("still unresolved".into()))
            }),
        ));
    let active_cleanups = Arc::new(AtomicUsize::new(0));
    let captured = Arc::clone(&active_cleanups);
    install_active_registrations(vec![PluginRegistration::new(
        "test",
        "active",
        Box::new(move || {
            captured.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }),
    )]);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime
        .block_on(clear_plugin_configuration_async(Duration::from_secs(1)))
        .unwrap();
    assert_eq!(active_cleanups.load(Ordering::SeqCst), 1);
    assert_eq!(retained_attempts.load(Ordering::SeqCst), 0);
    assert!(active_plugin_report().is_none());
    assert_eq!(FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().len(), 1);
    reset_global();
}

#[test]
fn test_dropping_async_clear_aborts_and_deregisters_taken_state() {
    let _guard = lock_runtime_owner();
    reset_global();
    let order = Arc::new(Mutex::new(Vec::new()));
    install_active_registrations(vec![shutdown_test_registration(
        "pending",
        Arc::clone(&order),
        TestDrainBehavior::Pending,
    )]);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime.block_on(async {
        let mut clear = Box::pin(clear_plugin_configuration_async(Duration::from_secs(30)));
        tokio::select! {
            result = &mut clear => panic!("pending drain unexpectedly completed: {result:?}"),
            _ = tokio::time::sleep(Duration::from_millis(10)) => {}
        }
        drop(clear);
    });

    assert_eq!(
        *order.lock().unwrap(),
        [
            "stop:pending",
            "drain:pending",
            "abort:pending",
            "deregister:pending",
        ]
    );
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_sync_clear_returns_with_no_new_managed_internal_replay_starts() {
    let _guard = lock_runtime_owner();
    reset_global();
    let harness = install_internal_replay_shutdown_race(LlmCallRole::Shadow);
    assert!(harness.schedule_paused_before_replay_start(LlmCallRole::Judge));
    harness.wait_for_second_provider_pause();
    let (release_contender, contender) = harness.poised_contender(LlmCallRole::Shadow);

    clear_plugin_configuration().unwrap();

    harness.assert_closed_at_return(release_contender, contender);
    assert!(harness.inner.replay_aborted.load(Ordering::SeqCst));
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_async_clear_returns_with_no_new_managed_internal_replay_starts() {
    let _guard = lock_runtime_owner();
    reset_global();
    let harness = install_internal_replay_shutdown_race(LlmCallRole::Judge);
    assert!(harness.schedule_paused_before_replay_start(LlmCallRole::Shadow));
    harness.wait_for_second_provider_pause();
    let (release_contender, contender) = harness.poised_contender(LlmCallRole::Judge);
    let clear = std::thread::spawn(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(clear_plugin_configuration_async(Duration::from_secs(1)))
    });

    harness.wait_for_intake_stop();
    harness.complete_replay();
    clear.join().unwrap().unwrap();

    harness.assert_closed_at_return(release_contender, contender);
    assert!(!harness.inner.replay_aborted.load(Ordering::SeqCst));
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_replacement_returns_with_no_new_managed_internal_replay_starts() {
    let _guard = lock_runtime_owner();
    reset_global();
    let harness = install_internal_replay_shutdown_race(LlmCallRole::Shadow);
    assert!(harness.schedule_paused_before_replay_start(LlmCallRole::Judge));
    harness.wait_for_second_provider_pause();
    let (release_contender, contender) = harness.poised_contender(LlmCallRole::Shadow);
    let replace = std::thread::spawn(|| {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
            .block_on(initialize_plugins_exact_with_options(
                PluginConfig::default(),
                PluginInitializationOptions {
                    shutdown_timeout: Duration::from_secs(1),
                },
            ))
    });

    harness.wait_for_intake_stop();
    harness.complete_replay();
    replace.join().unwrap().unwrap();

    harness.assert_closed_at_return(release_contender, contender);
    assert!(!harness.inner.replay_aborted.load(Ordering::SeqCst));
    assert!(active_plugin_report().is_some());
    clear_plugin_configuration().unwrap();
    reset_global();
}

#[test]
fn test_rollback_aborts_hook_bearing_partial_registrations() {
    let order = Arc::new(Mutex::new(Vec::new()));
    let mut registrations = vec![shutdown_test_registration(
        "partial",
        Arc::clone(&order),
        TestDrainBehavior::Success,
    )];

    rollback_registrations(&mut registrations);

    assert_eq!(
        *order.lock().unwrap(),
        ["stop:partial", "abort:partial", "deregister:partial"]
    );
}

#[test]
fn test_clear_stops_intake_before_flushing_queued_subscriber_callbacks() {
    let _guard = lock_runtime_owner();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let intake_open = Arc::new(AtomicBool::new(true));
    let callback_started = Arc::new(AtomicBool::new(false));
    let callback_saw_closed = Arc::new(AtomicBool::new(false));
    let release_callback = Arc::new(Barrier::new(2));
    let mut ctx = PluginRegistrationContext::new();
    let callback_open = Arc::clone(&intake_open);
    let callback_started_flag = Arc::clone(&callback_started);
    let callback_closed_flag = Arc::clone(&callback_saw_closed);
    let callback_barrier = Arc::clone(&release_callback);
    ctx.register_subscriber(
        "shutdown-order",
        Arc::new(move |_event| {
            callback_started_flag.store(true, Ordering::SeqCst);
            callback_barrier.wait();
            callback_closed_flag.store(!callback_open.load(Ordering::SeqCst), Ordering::SeqCst);
        }),
    )
    .unwrap();
    let stop_open = Arc::clone(&intake_open);
    ctx.add_registration(PluginRegistration::with_shutdown(
        "test",
        "shutdown-resource",
        Box::new(|| Ok(())),
        Box::new(move || {
            stop_open.store(false, Ordering::SeqCst);
            Ok(())
        }),
        Box::new(|_deadline| Box::pin(async { Ok(()) })),
        Box::new(|| Ok(())),
    ));
    install_active_registrations(ctx.into_registrations());

    emit_scope_event(
        EmitMarkEventParams::builder()
            .name("queued-before-clear")
            .build(),
    )
    .unwrap();
    let wait_deadline = Instant::now() + Duration::from_secs(1);
    while !callback_started.load(Ordering::SeqCst) && Instant::now() < wait_deadline {
        std::thread::yield_now();
    }
    assert!(callback_started.load(Ordering::SeqCst));

    let clear_thread = std::thread::spawn(clear_plugin_configuration);
    let wait_deadline = Instant::now() + Duration::from_secs(1);
    while intake_open.load(Ordering::SeqCst) && Instant::now() < wait_deadline {
        std::thread::yield_now();
    }
    assert!(!intake_open.load(Ordering::SeqCst));
    release_callback.wait();

    clear_thread.join().unwrap().unwrap();
    assert!(callback_saw_closed.load(Ordering::SeqCst));
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_dispatcher_thread_clear_returns_conflict_without_taking_active_state() {
    let _guard = lock_runtime_owner();
    reset_global();
    set_thread_scope_stack(create_scope_stack());

    let callback_result = Arc::new(Mutex::new(None));
    let result_slot = Arc::clone(&callback_result);
    let mut ctx = PluginRegistrationContext::new();
    ctx.register_subscriber(
        "reentrant-clear",
        Arc::new(move |_event| {
            *result_slot.lock().unwrap() = Some(clear_plugin_configuration());
        }),
    )
    .unwrap();
    install_active_registrations(ctx.into_registrations());

    emit_scope_event(
        EmitMarkEventParams::builder()
            .name("reentrant-clear")
            .build(),
    )
    .unwrap();
    crate::api::runtime::flush_subscribers().unwrap();

    match callback_result.lock().unwrap().take().unwrap() {
        Err(PluginError::Conflict(message)) => {
            assert!(message.contains("subscriber callback"));
        }
        other => panic!("unexpected dispatcher clear result: {other:?}"),
    }
    assert!(active_plugin_report().is_some());
    clear_plugin_configuration().unwrap();
    reset_global();
}

#[test]
fn test_sync_clear_returns_conflict_while_an_async_transition_owns_the_gate() {
    let _guard = lock_runtime_owner();
    reset_global();
    install_active_registrations(Vec::new());
    let transition = PLUGIN_CONFIGURATION_TRANSITION.try_lock().unwrap();

    match clear_plugin_configuration() {
        Err(PluginError::Conflict(message)) => {
            assert!(message.contains("transition is already in progress"));
        }
        other => panic!("unexpected concurrent clear result: {other:?}"),
    }
    assert!(active_plugin_report().is_some());

    drop(transition);
    clear_plugin_configuration().unwrap();
    reset_global();
}

#[test]
fn test_replacement_drains_previous_configuration_before_storing_new_state() {
    let _guard = lock_runtime_owner();
    reset_global();
    let order = Arc::new(Mutex::new(Vec::new()));
    install_active_registrations(vec![shutdown_test_registration(
        "previous",
        Arc::clone(&order),
        TestDrainBehavior::Success,
    )]);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime
        .block_on(initialize_plugins_exact_with_options(
            PluginConfig::default(),
            PluginInitializationOptions {
                shutdown_timeout: Duration::from_secs(1),
            },
        ))
        .unwrap();

    assert_eq!(
        *order.lock().unwrap(),
        ["stop:previous", "drain:previous", "deregister:previous"]
    );
    assert!(active_plugin_report().is_some());
    clear_plugin_configuration().unwrap();
    reset_global();
}

#[test]
fn test_replacement_teardown_failure_leaves_configuration_cleared() {
    let _guard = lock_runtime_owner();
    reset_global();
    let order = Arc::new(Mutex::new(Vec::new()));
    install_active_registrations(vec![shutdown_test_registration(
        "previous",
        Arc::clone(&order),
        TestDrainBehavior::Error,
    )]);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let error = runtime
        .block_on(initialize_plugins_exact(PluginConfig::default()))
        .unwrap_err();

    assert!(error.to_string().contains("previous' drain failed"));
    assert_eq!(
        *order.lock().unwrap(),
        [
            "stop:previous",
            "drain:previous",
            "abort:previous",
            "deregister:previous",
        ]
    );
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_initialize_plugins_restores_previous_configuration_after_failed_replacement() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(RecordingPlugin)).unwrap();
    register_plugin(Arc::new(RestoreFailPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("recording.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();

    let err = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("restore.fail.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap_err();
    match err {
        PluginError::RegistrationFailed(message) => {
            assert!(message.contains("restore.fail.plugin refused to initialize"));
        }
        other => panic!("unexpected replacement failure: {other}"),
    }

    assert_eq!(RESTORE_FAIL_REGISTRATIONS.load(Ordering::SeqCst), 1);
    let restored_report = active_plugin_report().expect("previous config should be restored");
    assert!(restored_report.diagnostics.is_empty());
    let names = recorded_names().lock().unwrap().clone();
    assert_eq!(
        names,
        vec![
            "__nemo_relay_plugin__recording.plugin__subscriber",
            "__nemo_relay_plugin__recording.plugin__subscriber",
        ]
    );
    reset_global();
}

#[test]
fn test_initialize_plugins_rolls_back_partial_component_registration_on_failure() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(PartialFailPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let err = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("partial.fail.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap_err();

    match err {
        PluginError::RegistrationFailed(message) => {
            assert!(message.contains("partial.fail.plugin refused to finish initialization"));
        }
        other => panic!("unexpected partial registration failure: {other}"),
    }

    assert_eq!(PARTIAL_FAIL_ROLLBACKS.load(Ordering::SeqCst), 1);
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_cancelling_activation_rolls_back_completed_and_in_progress_components() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(ActivationCancelPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let mut initialize = Box::pin(initialize_plugins_exact(PluginConfig {
            components: vec![
                PluginComponentSpec::new("activation.cancel.plugin"),
                PluginComponentSpec::new("activation.cancel.plugin"),
            ],
            ..PluginConfig::default()
        }));

        tokio::select! {
            result = &mut initialize => panic!("pending activation unexpectedly completed: {result:?}"),
            _ = tokio::time::sleep(Duration::from_millis(10)) => {}
        }
        assert_eq!(
            ACTIVATION_CANCEL_REGISTRATIONS.load(Ordering::SeqCst),
            2
        );
        drop(initialize);
    });

    assert_eq!(ACTIVATION_CANCEL_ROLLBACKS.load(Ordering::SeqCst), 2);
    assert_eq!(
        ACTIVATION_CANCEL_PENDING_ROLLBACKS.load(Ordering::SeqCst),
        2
    );
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_initialize_plugins_skips_disabled_components_and_namespaces_multiple_instances() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(RecordingPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![
                PluginComponentSpec::new("recording.plugin"),
                PluginComponentSpec {
                    enabled: false,
                    ..PluginComponentSpec::new("recording.plugin")
                },
                PluginComponentSpec::new("recording.plugin"),
            ],
            ..PluginConfig::default()
        }))
        .unwrap();

    let names = recorded_names().lock().unwrap().clone();
    assert_eq!(
        names,
        vec![
            "__nemo_relay_plugin__recording.plugin__1__subscriber",
            "__nemo_relay_plugin__recording.plugin__2__subscriber",
        ]
    );
    reset_global();
}

#[test]
fn test_initialize_plugins_reports_missing_component_during_activation() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(VanishingPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let error = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("vanishing.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap_err();

    match error {
        PluginError::NotFound(message) => {
            assert!(message.contains("vanishing.plugin"));
            assert!(active_plugin_report().is_none());
        }
        other => panic!("unexpected activation failure: {other}"),
    }

    reset_global();
}

#[test]
fn test_plugin_registration_context_supports_guardrail_helpers() {
    let _guard = lock_runtime_owner();
    reset_global();

    let mut ctx = PluginRegistrationContext::with_namespace("plugin::");
    ctx.register_mark_sanitize_guardrail("mark_sanitize", 1, Arc::new(|_, fields| fields))
        .unwrap();
    ctx.register_scope_sanitize_start_guardrail(
        "scope_sanitize_start",
        1,
        Arc::new(|_, fields| fields),
    )
    .unwrap();
    ctx.register_scope_sanitize_end_guardrail(
        "scope_sanitize_end",
        1,
        Arc::new(|_, fields| fields),
    )
    .unwrap();
    ctx.register_tool_sanitize_request_guardrail(
        "tool_sanitize_request",
        1,
        Arc::new(|_, args| args),
    )
    .unwrap();
    ctx.register_tool_sanitize_response_guardrail(
        "tool_sanitize_response",
        1,
        Arc::new(|_, response| response),
    )
    .unwrap();
    ctx.register_tool_conditional_execution_guardrail(
        "tool_conditional",
        1,
        Arc::new(|name, _args| Ok((name == "blocked-tool").then(|| "blocked tool".to_string()))),
    )
    .unwrap();
    ctx.register_llm_sanitize_request_guardrail(
        "llm_sanitize_request",
        1,
        Arc::new(|request| request),
    )
    .unwrap();
    ctx.register_llm_sanitize_response_guardrail(
        "llm_sanitize_response",
        1,
        Arc::new(|response| response),
    )
    .unwrap();
    ctx.register_llm_conditional_execution_guardrail(
        "llm_conditional",
        1,
        Arc::new(|request| {
            Ok((request.headers.get("blocked") == Some(&json!(true)))
                .then(|| "blocked llm".to_string()))
        }),
    )
    .unwrap();

    match tool_conditional_execution("blocked-tool", &json!({})) {
        Err(FlowError::GuardrailRejected(message)) => assert_eq!(message, "blocked tool"),
        other => panic!("expected tool guardrail rejection, got {other:?}"),
    }

    match llm_conditional_execution(&LlmRequest {
        headers: Map::from_iter([(String::from("blocked"), json!(true))]),
        content: json!({"messages": []}),
    }) {
        Err(FlowError::GuardrailRejected(message)) => assert_eq!(message, "blocked llm"),
        other => panic!("expected llm guardrail rejection, got {other:?}"),
    }

    let mut registrations = ctx.into_registrations();
    rollback_registrations(&mut registrations);

    assert!(tool_conditional_execution("blocked-tool", &json!({})).is_ok());
    assert!(
        llm_conditional_execution(&LlmRequest {
            headers: Map::from_iter([(String::from("blocked"), json!(true))]),
            content: json!({"messages": []}),
        })
        .is_ok()
    );

    reset_global();
}

#[test]
fn test_plugin_registration_context_maps_duplicate_registration_errors() {
    let _guard = lock_runtime_owner();
    reset_global();

    let mut ctx = PluginRegistrationContext::with_namespace("duplicate::");
    ctx.register_mark_sanitize_guardrail("mark", 1, Arc::new(|_, fields| fields))
        .unwrap();
    expect_registration_failed(
        ctx.register_mark_sanitize_guardrail("mark", 1, Arc::new(|_, fields| fields)),
        "mark sanitizer:",
    );
    ctx.register_scope_sanitize_start_guardrail("scope-start", 1, Arc::new(|_, fields| fields))
        .unwrap();
    expect_registration_failed(
        ctx.register_scope_sanitize_start_guardrail("scope-start", 1, Arc::new(|_, fields| fields)),
        "scope-start sanitizer:",
    );
    ctx.register_scope_sanitize_end_guardrail("scope-end", 1, Arc::new(|_, fields| fields))
        .unwrap();
    expect_registration_failed(
        ctx.register_scope_sanitize_end_guardrail("scope-end", 1, Arc::new(|_, fields| fields)),
        "scope-end sanitizer:",
    );
    ctx.register_llm_request_intercept(
        "llm-request",
        1,
        false,
        Arc::new(|_name, request, annotated| {
            Ok(LlmRequestInterceptOutcome::new(request, annotated))
        }),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_llm_request_intercept(
            "llm-request",
            1,
            false,
            Arc::new(|_name, request, annotated| {
                Ok(LlmRequestInterceptOutcome::new(request, annotated))
            }),
        ),
        "llm request intercept:",
    );

    ctx.register_tool_sanitize_request_guardrail(
        "tool-sanitize-request",
        1,
        Arc::new(|_, args| args),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_tool_sanitize_request_guardrail(
            "tool-sanitize-request",
            1,
            Arc::new(|_, args| args),
        ),
        "tool sanitize request guardrail:",
    );

    ctx.register_tool_sanitize_response_guardrail(
        "tool-sanitize-response",
        1,
        Arc::new(|_, response| response),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_tool_sanitize_response_guardrail(
            "tool-sanitize-response",
            1,
            Arc::new(|_, response| response),
        ),
        "tool sanitize response guardrail:",
    );

    ctx.register_tool_conditional_execution_guardrail(
        "tool-conditional",
        1,
        Arc::new(|_, _| Ok(None)),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_tool_conditional_execution_guardrail(
            "tool-conditional",
            1,
            Arc::new(|_, _| Ok(None)),
        ),
        "tool conditional execution guardrail:",
    );

    ctx.register_llm_sanitize_request_guardrail(
        "llm-sanitize-request",
        1,
        Arc::new(|request| request),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_llm_sanitize_request_guardrail(
            "llm-sanitize-request",
            1,
            Arc::new(|request| request),
        ),
        "llm sanitize request guardrail:",
    );

    ctx.register_llm_sanitize_response_guardrail(
        "llm-sanitize-response",
        1,
        Arc::new(|response| response),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_llm_sanitize_response_guardrail(
            "llm-sanitize-response",
            1,
            Arc::new(|response| response),
        ),
        "llm sanitize response guardrail:",
    );

    ctx.register_llm_conditional_execution_guardrail("llm-conditional", 1, Arc::new(|_| Ok(None)))
        .unwrap();
    expect_registration_failed(
        ctx.register_llm_conditional_execution_guardrail(
            "llm-conditional",
            1,
            Arc::new(|_| Ok(None)),
        ),
        "llm conditional execution guardrail:",
    );

    ctx.register_llm_execution_intercept(
        "llm-exec",
        1,
        Arc::new(|_name, request, _next| Box::pin(async move { Ok(request.content) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_llm_execution_intercept(
            "llm-exec",
            1,
            Arc::new(|_name, request, _next| Box::pin(async move { Ok(request.content) })),
        ),
        "llm execution intercept:",
    );
    expect_registration_failed(
        ctx.register_llm_execution_intercept_v2(
            "llm-exec",
            1,
            Arc::new(|_name, _context, request, _replay, _next| {
                Box::pin(async move { Ok(request.content) })
            }),
        ),
        "llm V2 execution intercept:",
    );

    ctx.register_llm_stream_execution_intercept(
        "llm-stream",
        1,
        Arc::new(|_name, request, _next| {
            Box::pin(async move {
                Ok(Box::pin(tokio_stream::iter(vec![Ok(request.content)]))
                    as Pin<
                        Box<dyn tokio_stream::Stream<Item = crate::error::Result<Json>> + Send>,
                    >)
            })
        }),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_llm_stream_execution_intercept(
            "llm-stream",
            1,
            Arc::new(|_name, request, _next| {
                Box::pin(async move {
                    Ok(Box::pin(tokio_stream::iter(vec![Ok(request.content)]))
                        as Pin<
                            Box<dyn tokio_stream::Stream<Item = crate::error::Result<Json>> + Send>,
                        >)
                })
            }),
        ),
        "llm stream execution intercept:",
    );

    ctx.register_tool_request_intercept("tool-request", 1, false, Arc::new(|_name, args| Ok(args)))
        .unwrap();
    expect_registration_failed(
        ctx.register_tool_request_intercept(
            "tool-request",
            1,
            false,
            Arc::new(|_name, args| Ok(args)),
        ),
        "tool request intercept:",
    );

    ctx.register_tool_execution_intercept(
        "tool-exec",
        1,
        Arc::new(|_name, args, _next| Box::pin(async move { Ok(args.into()) })),
    )
    .unwrap();
    expect_registration_failed(
        ctx.register_tool_execution_intercept(
            "tool-exec",
            1,
            Arc::new(|_name, args, _next| Box::pin(async move { Ok(args.into()) })),
        ),
        "tool execution intercept:",
    );

    let mut registrations = ctx.into_registrations();
    rollback_registrations(&mut registrations);
    reset_global();
}

#[test]
fn test_plugin_registration_context_maps_deregistration_errors() {
    let _guard = lock_runtime_owner();
    reset_global();

    let mut ctx = PluginRegistrationContext::with_namespace("teardown::");
    ctx.register_mark_sanitize_guardrail("mark-sanitize", 1, Arc::new(|_, fields| fields))
        .unwrap();
    ctx.register_scope_sanitize_start_guardrail(
        "scope-sanitize-start",
        1,
        Arc::new(|_, fields| fields),
    )
    .unwrap();
    ctx.register_scope_sanitize_end_guardrail(
        "scope-sanitize-end",
        1,
        Arc::new(|_, fields| fields),
    )
    .unwrap();
    ctx.register_subscriber("subscriber", Arc::new(|_event| {}))
        .unwrap();
    ctx.register_llm_request_intercept(
        "llm-request",
        1,
        false,
        Arc::new(|_name, request, annotated| {
            Ok(LlmRequestInterceptOutcome::new(request, annotated))
        }),
    )
    .unwrap();
    ctx.register_tool_sanitize_request_guardrail(
        "tool-sanitize-request",
        1,
        Arc::new(|_, args| args),
    )
    .unwrap();
    ctx.register_tool_sanitize_response_guardrail(
        "tool-sanitize-response",
        1,
        Arc::new(|_, response| response),
    )
    .unwrap();
    ctx.register_tool_conditional_execution_guardrail(
        "tool-conditional",
        1,
        Arc::new(|_, _| Ok(None)),
    )
    .unwrap();
    ctx.register_llm_sanitize_request_guardrail(
        "llm-sanitize-request",
        1,
        Arc::new(|request| request),
    )
    .unwrap();
    ctx.register_llm_sanitize_response_guardrail(
        "llm-sanitize-response",
        1,
        Arc::new(|response| response),
    )
    .unwrap();
    ctx.register_llm_conditional_execution_guardrail("llm-conditional", 1, Arc::new(|_| Ok(None)))
        .unwrap();
    ctx.register_llm_execution_intercept(
        "llm-exec",
        1,
        Arc::new(|_name, request, _next| Box::pin(async move { Ok(request.content) })),
    )
    .unwrap();
    ctx.register_llm_execution_intercept_v2(
        "llm-exec-v2",
        1,
        Arc::new(|_name, _context, request, _replay, _next| {
            Box::pin(async move { Ok(request.content) })
        }),
    )
    .unwrap();
    ctx.register_llm_stream_execution_intercept(
        "llm-stream",
        1,
        Arc::new(|_name, request, _next| {
            Box::pin(async move {
                Ok(Box::pin(tokio_stream::iter(vec![Ok(request.content)]))
                    as Pin<
                        Box<dyn tokio_stream::Stream<Item = crate::error::Result<Json>> + Send>,
                    >)
            })
        }),
    )
    .unwrap();
    ctx.register_tool_request_intercept("tool-request", 1, false, Arc::new(|_name, args| Ok(args)))
        .unwrap();
    ctx.register_tool_execution_intercept(
        "tool-exec",
        1,
        Arc::new(|_name, args, _next| Box::pin(async move { Ok(args.into()) })),
    )
    .unwrap();

    let mut registrations = ctx.into_registrations();
    let expected_messages = [
        "mark sanitizer deregistration failed:",
        "scope-start sanitizer deregistration failed:",
        "scope-end sanitizer deregistration failed:",
        "subscriber deregistration failed:",
        "llm request intercept deregistration failed:",
        "tool sanitize request guardrail deregistration failed:",
        "tool sanitize response guardrail deregistration failed:",
        "tool conditional execution guardrail deregistration failed:",
        "llm sanitize request guardrail deregistration failed:",
        "llm sanitize response guardrail deregistration failed:",
        "llm conditional execution guardrail deregistration failed:",
        "llm execution intercept deregistration failed:",
        "llm V2 execution intercept deregistration failed:",
        "llm stream execution intercept deregistration failed:",
        "tool request intercept deregistration failed:",
        "tool execution intercept deregistration failed:",
    ];

    set_conflicting_runtime_owner_for_tests();
    for (registration, expected) in registrations.iter_mut().zip(expected_messages) {
        match (registration.deregister)() {
            Err(PluginError::RegistrationFailed(message)) => {
                assert!(message.contains(expected), "{message}");
            }
            Err(other) => panic!("unexpected deregistration failure: {other}"),
            Ok(()) => panic!("expected deregistration to fail"),
        }
    }

    reset_global();
}

#[test]
fn test_initialize_plugins_replaces_previous_configuration_on_success() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(RecordingPlugin)).unwrap();
    register_plugin(Arc::new(ReplacementPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("recording.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();

    let report = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("replacement.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();

    assert!(
        report
            .diagnostics
            .iter()
            .any(|diag| diag.code == "replacement.warning")
    );
    assert_eq!(active_plugin_report().unwrap().diagnostics.len(), 1);
    assert_eq!(REPLACEMENT_REGISTRATIONS.load(Ordering::SeqCst), 1);

    reset_global();
}

#[test]
fn test_initialize_plugins_reports_failed_restore_when_previous_configuration_cannot_be_restored() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(RestoreBreakPlugin)).unwrap();
    register_plugin(Arc::new(RestoreFailPlugin)).unwrap();

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("restore.break.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();

    let error = runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("restore.fail.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap_err();

    match error {
        PluginError::RegistrationFailed(message) => {
            assert!(message.contains("restore.fail.plugin refused to initialize"));
            assert!(message.contains("previous plugin configuration could not be restored"));
            assert!(message.contains("restore.break.plugin refused to restore"));
        }
        other => panic!("unexpected failed-restore error: {other}"),
    }

    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_load_plugin_config_files_merges_files_by_precedence() {
    let dir = tempfile::tempdir().unwrap();
    let lower = dir.path().join("lower.toml");
    let higher = dir.path().join("higher.toml");
    std::fs::write(
        &lower,
        "version = 1\n\
         [[components]]\n\
         kind = \"observability\"\n\
         enabled = false\n\
         [components.config]\n\
         output_directory = \"/var/log\"\n\
         mode = \"append\"\n",
    )
    .unwrap();
    std::fs::write(
        &higher,
        "[[components]]\n\
         kind = \"observability\"\n\
         [components.config]\n\
         mode = \"overwrite\"\n\
         [[components]]\n\
         kind = \"adaptive\"\n",
    )
    .unwrap();

    let (merged, sources) = load_plugin_config_files([lower.clone(), higher.clone()])
        .unwrap()
        .expect("a file exists");
    assert_eq!(sources, vec![lower, higher]);

    let components = merged["components"].as_array().unwrap();
    let observability = &components[0];
    assert_eq!(observability["kind"], json!("observability"));
    assert_eq!(
        observability["enabled"],
        json!(false),
        "lower-file enabled is inherited (higher omits it)"
    );
    assert_eq!(
        observability["config"]["output_directory"],
        json!("/var/log"),
        "lower-only config key is inherited"
    );
    assert_eq!(
        observability["config"]["mode"],
        json!("overwrite"),
        "higher file overrides the shared config key"
    );
    assert_eq!(
        components[1]["kind"],
        json!("adaptive"),
        "higher-only component kind is appended"
    );
}

#[test]
fn test_layer_config_applies_typed_overlay_defaults_over_file_base() {
    // The code-vs-file path `initialize_plugins` takes: a typed `PluginConfig` is layered
    // over the discovered file base. Its serde defaults (`version`/`policy`/`enabled`)
    // override the file, the free-form `config` body merges, and an undeclared component
    // kind is inherited from the file.
    let file_base = json!({
        "version": 2,
        "components": [
            {
                "kind": "observability",
                "enabled": false,
                "config": { "output_directory": "/var/log", "mode": "append" }
            },
            { "kind": "adaptive", "config": { "ttl": 60 } }
        ],
        "policy": {
            "unknown_component": "error",
            "unknown_field": "warn",
            "unsupported_value": "error"
        }
    });
    let code = PluginConfig {
        components: vec![PluginComponentSpec {
            config: Map::from_iter([(String::from("mode"), json!("overwrite"))]),
            ..PluginComponentSpec::new("observability")
        }],
        ..PluginConfig::default()
    };

    let mut merged = file_base;
    layer_config(&mut merged, serde_json::to_value(code).unwrap());
    let typed: PluginConfig = serde_json::from_value(merged).unwrap();

    // Typed defaults override the file base.
    assert_eq!(typed.version, 1, "typed default version overrides the file");
    assert_eq!(
        typed.policy.unknown_component,
        UnsupportedBehavior::Warn,
        "typed default policy overrides the file"
    );
    let observability = &typed.components[0];
    assert_eq!(observability.kind, "observability");
    assert!(
        observability.enabled,
        "typed default enabled=true overrides the file's false"
    );
    // The component config body merges: code's `mode` wins, the file's `output_directory`
    // is inherited.
    assert_eq!(observability.config["mode"], json!("overwrite"));
    assert_eq!(observability.config["output_directory"], json!("/var/log"));
    // A kind the code config does not declare is inherited from the file.
    assert_eq!(typed.components[1].kind, "adaptive");
}

struct DrainWithPanickingDrop {
    name: &'static str,
    order: Arc<Mutex<Vec<String>>>,
    completes: bool,
}

impl Future for DrainWithPanickingDrop {
    type Output = Result<()>;

    fn poll(
        self: Pin<&mut Self>,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Self::Output> {
        if self.completes {
            std::task::Poll::Ready(Ok(()))
        } else {
            std::task::Poll::Pending
        }
    }
}

impl Drop for DrainWithPanickingDrop {
    fn drop(&mut self) {
        {
            self.order
                .lock()
                .unwrap()
                .push(format!("drop:{}", self.name));
        }
        panic!("{} drain future drop panic", self.name);
    }
}

fn panicking_drop_shutdown_registration(
    name: &'static str,
    order: Arc<Mutex<Vec<String>>>,
    completes: bool,
) -> PluginRegistration {
    let stop_order = order.clone();
    let drain_order = order.clone();
    let drain_future_order = order.clone();
    let abort_order = order.clone();
    let deregister_order = order;
    PluginRegistration::with_shutdown(
        "test",
        name,
        Box::new(move || {
            deregister_order
                .lock()
                .unwrap()
                .push(format!("deregister:{name}"));
            Ok(())
        }),
        Box::new(move || {
            stop_order.lock().unwrap().push(format!("stop:{name}"));
            Ok(())
        }),
        Box::new(move |_deadline| {
            drain_order.lock().unwrap().push(format!("drain:{name}"));
            Box::pin(DrainWithPanickingDrop {
                name,
                order: drain_future_order.clone(),
                completes,
            })
        }),
        Box::new(move || {
            abort_order.lock().unwrap().push(format!("abort:{name}"));
            Ok(())
        }),
    )
}

#[test]
fn test_async_clear_contains_drain_future_drop_panic_on_timeout() {
    let _guard = lock_runtime_owner();
    reset_global();
    let order = Arc::new(Mutex::new(Vec::new()));
    install_active_registrations(vec![panicking_drop_shutdown_registration(
        "timeout",
        order.clone(),
        false,
    )]);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        runtime.block_on(clear_plugin_configuration_async(Duration::from_millis(20)))
    }));

    let error = result
        .expect("drain future drop panic must be contained")
        .unwrap_err();
    assert!(error.to_string().contains("shared deadline expired"));
    assert_eq!(
        *order.lock().unwrap(),
        [
            "stop:timeout",
            "drain:timeout",
            "drop:timeout",
            "abort:timeout",
            "deregister:timeout",
        ]
    );
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_async_clear_contains_drain_future_drop_panic_on_cancellation() {
    let _guard = lock_runtime_owner();
    reset_global();
    let order = Arc::new(Mutex::new(Vec::new()));
    install_active_registrations(vec![panicking_drop_shutdown_registration(
        "cancelled",
        order.clone(),
        false,
    )]);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        runtime.block_on(async {
            let mut clear = Box::pin(clear_plugin_configuration_async(Duration::from_secs(30)));
            tokio::select! {
                result = &mut clear => panic!("pending drain unexpectedly completed: {result:?}"),
                _ = tokio::time::sleep(Duration::from_millis(10)) => {}
            }
            drop(clear);
        });
    }));

    result.expect("drain future drop panic must be contained");
    assert_eq!(
        *order.lock().unwrap(),
        [
            "stop:cancelled",
            "drain:cancelled",
            "drop:cancelled",
            "abort:cancelled",
            "deregister:cancelled",
        ]
    );
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_async_clear_reports_completed_drain_future_drop_panic() {
    let _guard = lock_runtime_owner();
    reset_global();
    let order = Arc::new(Mutex::new(Vec::new()));
    install_active_registrations(vec![panicking_drop_shutdown_registration(
        "completed",
        order.clone(),
        true,
    )]);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        runtime.block_on(clear_plugin_configuration_async(Duration::from_secs(1)))
    }));

    let error = result
        .expect("completed drain future drop panic must be contained")
        .unwrap_err();
    assert!(error.to_string().contains("future panicked"));
    assert_eq!(
        *order.lock().unwrap(),
        [
            "stop:completed",
            "drain:completed",
            "drop:completed",
            "abort:completed",
            "deregister:completed",
        ]
    );
    assert!(active_plugin_report().is_none());
    reset_global();
}

#[test]
fn test_async_deregistration_quarantine_releases_unused_abort_hook() {
    let _guard = lock_runtime_owner();
    reset_global();
    let owner = Arc::new(());
    let owner_weak = Arc::downgrade(&owner);
    let abort_owner = owner.clone();
    drop(owner);
    install_active_registrations(vec![PluginRegistration::with_shutdown(
        "test",
        "async-quarantine",
        Box::new(|| Err(PluginError::Internal("deregister failed".into()))),
        Box::new(|| Ok(())),
        Box::new(|_deadline| Box::pin(async { Ok(()) })),
        Box::new(move || {
            let _ = &abort_owner;
            Ok(())
        }),
    )]);
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let error = runtime
        .block_on(clear_plugin_configuration_async(Duration::from_secs(1)))
        .unwrap_err();

    assert!(error.to_string().contains("deregister failed"));
    assert_eq!(FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().len(), 1);
    assert!(
        owner_weak.upgrade().is_none(),
        "quarantine retained an abort hook after successful drain"
    );
    reset_global();
}

#[test]
fn test_sync_deregistration_quarantine_releases_unused_drain_hook() {
    let _guard = lock_runtime_owner();
    reset_global();
    let owner = Arc::new(());
    let owner_weak = Arc::downgrade(&owner);
    let drain_owner = owner.clone();
    drop(owner);
    install_active_registrations(vec![PluginRegistration::with_shutdown(
        "test",
        "sync-quarantine",
        Box::new(|| Err(PluginError::Internal("deregister failed".into()))),
        Box::new(|| Ok(())),
        Box::new(move |_deadline| {
            let owner = drain_owner.clone();
            Box::pin(async move {
                let _ = owner;
                Ok(())
            })
        }),
        Box::new(|| Ok(())),
    )]);

    let error = clear_plugin_configuration().unwrap_err();

    assert!(error.to_string().contains("deregister failed"));
    assert_eq!(FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().len(), 1);
    assert!(
        owner_weak.upgrade().is_none(),
        "quarantine retained a drain hook after synchronous abort"
    );
    reset_global();
}

#[test]
fn test_activation_commit_rejects_quarantine_added_during_registration() {
    let _guard = lock_runtime_owner();
    reset_global();
    let barrier = Arc::new(Barrier::new(2));
    let candidate_rollbacks = Arc::new(AtomicUsize::new(0));
    register_plugin(Arc::new(ActivationCommitRacePlugin {
        barrier: barrier.clone(),
        candidate_rollbacks: candidate_rollbacks.clone(),
    }))
    .unwrap();

    let initializer = std::thread::spawn(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("activation.commit.race.plugin")],
            ..PluginConfig::default()
        }))
    });

    barrier.wait();
    let failed_cleanup_attempts = Arc::new(AtomicUsize::new(0));
    let attempts = failed_cleanup_attempts.clone();
    let mut concurrent_cleanup = vec![PluginRegistration::new(
        "test",
        "concurrent-failed-cleanup",
        Box::new(move || {
            attempts.fetch_add(1, Ordering::SeqCst);
            Err(PluginError::Internal("unresolved cleanup".to_string()))
        }),
    )];
    rollback_registrations(&mut concurrent_cleanup);
    assert!(concurrent_cleanup.is_empty());
    assert_eq!(failed_cleanup_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().len(), 1);
    barrier.wait();

    let error = initializer.join().unwrap().unwrap_err();
    assert!(error.to_string().contains("failed deregistration cleanup"));
    assert_eq!(candidate_rollbacks.load(Ordering::SeqCst), 1);
    assert!(active_plugin_report().is_none());
    assert_eq!(FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().len(), 1);

    FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().clear();
    assert!(deregister_plugin("activation.commit.race.plugin"));
    reset_global();
}

#[test]
fn test_replacement_commit_rejection_attempts_previous_configuration_restore() {
    let _guard = lock_runtime_owner();
    reset_global();
    register_plugin(Arc::new(RecordingPlugin)).unwrap();
    let barrier = Arc::new(Barrier::new(2));
    let candidate_rollbacks = Arc::new(AtomicUsize::new(0));
    register_plugin(Arc::new(ActivationCommitRacePlugin {
        barrier: barrier.clone(),
        candidate_rollbacks: candidate_rollbacks.clone(),
    }))
    .unwrap();
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime
        .block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("recording.plugin")],
            ..PluginConfig::default()
        }))
        .unwrap();
    assert_eq!(recorded_names().lock().unwrap().len(), 1);

    let initializer = std::thread::spawn(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(initialize_plugins_exact(PluginConfig {
            components: vec![PluginComponentSpec::new("activation.commit.race.plugin")],
            ..PluginConfig::default()
        }))
    });

    barrier.wait();
    let mut concurrent_cleanup = vec![PluginRegistration::new(
        "test",
        "replacement-concurrent-failed-cleanup",
        Box::new(|| Err(PluginError::Internal("unresolved cleanup".to_string()))),
    )];
    rollback_registrations(&mut concurrent_cleanup);
    barrier.wait();

    let error = initializer.join().unwrap().unwrap_err();
    let message = error.to_string();
    assert!(message.contains("plugin activation blocked by failed deregistration cleanup"));
    assert!(message.contains("previous plugin configuration could not be restored"));
    assert_eq!(candidate_rollbacks.load(Ordering::SeqCst), 1);
    assert_eq!(
        recorded_names().lock().unwrap().len(),
        2,
        "previous plugin registration was not retried"
    );
    assert!(active_plugin_report().is_none());
    assert_eq!(FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().len(), 1);

    FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().clear();
    assert!(deregister_plugin("activation.commit.race.plugin"));
    assert!(deregister_plugin("recording.plugin"));
    reset_global();
}

#[test]
fn test_activation_commit_quarantine_lock_poison_rolls_back_without_deadlock() {
    let _guard = lock_runtime_owner();
    reset_global();
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _quarantine = FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap();
        panic!("poison quarantine lock");
    }));
    assert!(FAILED_PLUGIN_DEREGISTRATIONS.is_poisoned());
    let deregistration_attempts = Arc::new(AtomicUsize::new(0));
    let attempts = deregistration_attempts.clone();

    let error = store_active_plugin_configuration(
        PluginConfig::default(),
        ConfigReport::default(),
        vec![PluginRegistration::new(
            "test",
            "quarantine-poison-candidate",
            Box::new(move || {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err(PluginError::Internal(
                    "candidate cleanup failed".to_string(),
                ))
            }),
        )],
    )
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("failed plugin deregistration lock poisoned")
    );
    assert_eq!(deregistration_attempts.load(Ordering::SeqCst), 1);
    assert!(active_plugin_report().is_none());
    assert_eq!(
        FAILED_PLUGIN_DEREGISTRATIONS
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .len(),
        1
    );

    FAILED_PLUGIN_DEREGISTRATIONS.clear_poison();
    FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().clear();
    reset_global();
}

#[test]
fn test_activation_commit_active_lock_poison_releases_quarantine_before_rollback() {
    let _guard = lock_runtime_owner();
    reset_global();
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let _active = ACTIVE_PLUGIN_CONFIGURATION.lock().unwrap();
        panic!("poison active configuration lock");
    }));
    assert!(ACTIVE_PLUGIN_CONFIGURATION.is_poisoned());
    let deregistration_attempts = Arc::new(AtomicUsize::new(0));
    let attempts = deregistration_attempts.clone();

    let error = store_active_plugin_configuration(
        PluginConfig::default(),
        ConfigReport::default(),
        vec![PluginRegistration::new(
            "test",
            "active-poison-candidate",
            Box::new(move || {
                attempts.fetch_add(1, Ordering::SeqCst);
                Err(PluginError::Internal(
                    "candidate cleanup failed".to_string(),
                ))
            }),
        )],
    )
    .unwrap_err();

    assert!(
        error
            .to_string()
            .contains("active plugin configuration lock poisoned")
    );
    assert_eq!(deregistration_attempts.load(Ordering::SeqCst), 1);
    assert_eq!(FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().len(), 1);

    ACTIVE_PLUGIN_CONFIGURATION.clear_poison();
    *ACTIVE_PLUGIN_CONFIGURATION.lock().unwrap() = None;
    FAILED_PLUGIN_DEREGISTRATIONS.lock().unwrap().clear();
    reset_global();
}
