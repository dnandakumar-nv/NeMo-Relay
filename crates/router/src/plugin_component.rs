// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Explicit Core plugin registration for the Router runtime.

use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, LazyLock};

use nemo_relay::plugin::{
    ConfigDiagnostic, DiagnosticLevel, Plugin, PluginComponentSpec, PluginError,
    PluginRegistrationContext, Result, deregister_plugin_if, lookup_plugin, register_plugin,
};
use serde_json::{Map, Value as Json};

use crate::config::{RouterConfig, RouterMode};
use crate::diagnostics::validate_router_config;
use crate::ledger::repository::{ActivatedLedger, LedgerRepository};
use crate::runtime::RouterRuntime;

type LedgerActivationResult = std::result::Result<PendingLedgerActivation, String>;

struct PendingLedgerActivation {
    activated: Option<ActivatedLedger>,
}

impl PendingLedgerActivation {
    fn new(activated: ActivatedLedger) -> Self {
        Self {
            activated: Some(activated),
        }
    }

    #[cfg_attr(test, allow(dead_code))]
    fn into_activated(mut self) -> ActivatedLedger {
        self.activated
            .take()
            .expect("pending ledger activation must be armed")
    }
}

impl Drop for PendingLedgerActivation {
    fn drop(&mut self) {
        if let Some(activated) = self.activated.as_mut() {
            activated.repository.stop_abandoned_process();
        }
    }
}

fn deliver_ledger_activation(
    sender: tokio::sync::oneshot::Sender<LedgerActivationResult>,
    result: LedgerActivationResult,
) {
    let _ = sender.send(result);
}

#[cfg(test)]
static LAST_TEST_RUNTIME: LazyLock<std::sync::Mutex<Option<std::sync::Weak<RouterRuntime>>>> =
    LazyLock::new(|| std::sync::Mutex::new(None));

#[cfg(test)]
type TestRuntimeStarter =
    Arc<dyn Fn(RouterConfig) -> std::result::Result<Arc<RouterRuntime>, String> + Send + Sync>;

#[cfg(test)]
static TEST_RUNTIME_STARTER: LazyLock<std::sync::Mutex<Option<TestRuntimeStarter>>> =
    LazyLock::new(|| std::sync::Mutex::new(None));

async fn start_router_runtime(
    config: RouterConfig,
) -> std::result::Result<Arc<RouterRuntime>, String> {
    #[cfg(test)]
    {
        if let Some(starter) = TEST_RUNTIME_STARTER.lock().unwrap().clone() {
            return starter(config);
        }
        RouterRuntime::start(config)
    }

    #[cfg(not(test))]
    {
        let activation_config = config.clone();
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        let _activation_task = tokio::task::spawn_blocking(move || {
            let result = LedgerRepository::activate(&activation_config)
                .map(PendingLedgerActivation::new)
                .map_err(|error| format!("{} ({})", error, error.code()));
            deliver_ledger_activation(result_tx, result);
        });
        let pending = result_rx
            .await
            .map_err(|_| "Router ledger activation task failed".to_string())??;
        let activated = pending.into_activated();
        RouterRuntime::start_with_activated_ledger(config, activated)
    }
}

async fn register_started_runtime(
    runtime: Arc<RouterRuntime>,
    context: &mut PluginRegistrationContext,
) -> Result<()> {
    // Lifecycle ownership precedes observation and execution so reverse
    // teardown removes the intercept, then subscriber, then runtime.
    let lifecycle_name = context.qualify_name("runtime");
    context.add_registration(runtime.lifecycle_registration(lifecycle_name)?);
    if let Err(error) = context.register_subscriber("events", runtime.subscriber_callback()) {
        return Err(rollback_registration_failure(&runtime, error).await);
    }
    if let Err(error) =
        context.register_llm_execution_intercept_v2("execution", 0, runtime.execution_callback())
    {
        return Err(rollback_registration_failure(&runtime, error).await);
    }
    Ok(())
}

async fn rollback_registration_failure(
    runtime: &RouterRuntime,
    registration_error: PluginError,
) -> PluginError {
    match runtime.rollback_activation().await {
        Ok(()) => registration_error,
        Err(rollback_error) => PluginError::Internal(format!(
            "{registration_error}; Router activation rollback failed: {rollback_error}"
        )),
    }
}

/// Core plugin kind for an explicitly registered Router component.
pub const ROUTER_PLUGIN_KIND: &str = "router";

/// One configured Router component for use in a Core plugin document.
#[derive(Debug, Clone)]
pub struct ComponentSpec {
    /// Whether the component should be activated after validation.
    pub enabled: bool,
    /// Complete versioned Router configuration.
    pub config: RouterConfig,
}

impl ComponentSpec {
    /// Creates an enabled Router component.
    pub fn new(config: RouterConfig) -> Self {
        Self {
            enabled: true,
            config,
        }
    }
}

impl From<ComponentSpec> for PluginComponentSpec {
    fn from(component: ComponentSpec) -> Self {
        let Json::Object(config) = serde_json::to_value(component.config)
            .expect("Router configuration must serialize to an object")
        else {
            unreachable!("Router configuration must serialize to an object");
        };
        Self {
            kind: ROUTER_PLUGIN_KIND.to_string(),
            enabled: component.enabled,
            config,
        }
    }
}

struct RouterPlugin;

impl Plugin for RouterPlugin {
    fn plugin_kind(&self) -> &str {
        ROUTER_PLUGIN_KIND
    }

    fn allows_multiple_components(&self) -> bool {
        false
    }

    fn validate(&self, plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
        validate_router_config(plugin_config).diagnostics
    }

    fn register<'a>(
        &'a self,
        plugin_config: &Map<String, Json>,
        context: &'a mut PluginRegistrationContext,
    ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        let plugin_config = plugin_config.clone();
        Box::pin(async move {
            let validation = validate_router_config(&plugin_config);
            if validation.has_errors() {
                let codes = validation
                    .diagnostics
                    .iter()
                    .filter(|diagnostic| diagnostic.level == DiagnosticLevel::Error)
                    .map(|diagnostic| diagnostic.code.as_str())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(PluginError::InvalidConfig(format!(
                    "Router component validation failed: {codes}"
                )));
            }
            let config = validation.config.ok_or_else(|| {
                PluginError::InvalidConfig(
                    "Router component validation produced no configuration".into(),
                )
            })?;
            // Spec 08 Task 10 replaces this fence when Active dispatch is wired end to end.
            if config.mode == RouterMode::Active {
                return Err(PluginError::InvalidConfig(
                    "Router Active mode is configured but runtime dispatch is not implemented"
                        .into(),
                ));
            }
            if config.mode == RouterMode::Off {
                return Ok(());
            }
            let runtime = start_router_runtime(config)
                .await
                .map_err(PluginError::Internal)?;
            #[cfg(test)]
            {
                *LAST_TEST_RUNTIME.lock().unwrap() = Some(Arc::downgrade(&runtime));
            }

            register_started_runtime(runtime, context).await
        })
    }
}

fn router_plugin() -> Arc<dyn Plugin> {
    static ROUTER_PLUGIN: LazyLock<Arc<dyn Plugin>> =
        LazyLock::new(|| Arc::new(RouterPlugin) as Arc<dyn Plugin>);
    ROUTER_PLUGIN.clone()
}

/// Registers the exact Router implementation in the Core plugin registry.
///
/// Calling this function repeatedly is idempotent only while the registry
/// still contains this crate's pointer-identical static plugin instance. A
/// foreign implementation using the `router` kind remains an error.
pub fn register_router_component() -> Result<()> {
    let plugin = router_plugin();
    match register_plugin(plugin.clone()) {
        Ok(()) => Ok(()),
        Err(error) => match lookup_plugin(ROUTER_PLUGIN_KIND) {
            Some(registered) if Arc::ptr_eq(&registered, &plugin) => Ok(()),
            _ => Err(error),
        },
    }
}

/// Deregisters this crate's exact Router plugin implementation.
///
/// Active Router component resources remain owned by the current Core plugin
/// configuration until that configuration is cleared or replaced. A foreign
/// implementation using the same kind is never removed.
pub fn deregister_router_component() -> bool {
    deregister_plugin_if(&router_plugin())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Barrier, Mutex as StdMutex, Weak};
    use std::time::{Duration, Instant};

    use nemo_relay::api::llm::{
        LlmApiFamily, LlmCallExecuteV2Params, LlmCallRole, LlmRequest, llm_call_execute_v2,
    };
    use nemo_relay::api::registry::{
        deregister_llm_execution_intercept, register_llm_execution_intercept_v2,
    };
    use nemo_relay::api::runtime::{
        LLM_REPLAY_CONTRACT_VERSION, LlmExecutionNextFn, LlmReplayCall, LlmReplayCapability,
        LlmReplayFactory, LlmReplayTransport, NemoRelayContextState, TASK_SCOPE_STACK,
        create_scope_stack, global_context,
    };
    use nemo_relay::api::scope::{
        EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeHandle, ScopeType,
        event as emit_scope_event, pop_scope, push_scope,
    };
    use nemo_relay::api::subscriber::{
        deregister_subscriber, flush_subscribers, register_subscriber,
    };
    use nemo_relay::error::{FlowError, Result as FlowResult};
    use nemo_relay::plugin::{
        PluginConfig, PluginInitializationOptions, active_plugin_report,
        clear_plugin_configuration, clear_plugin_configuration_async, deregister_plugin_if,
        initialize_plugins_exact, initialize_plugins_exact_with_options, rollback_registrations,
        validate_plugin_config,
    };
    use serde_json::json;
    use tempfile::tempdir;
    use tokio::sync::oneshot;
    use uuid::Uuid;

    use super::*;
    use crate::sink::{InMemoryTrajectoryDelivery, InMemoryTrajectorySink, SinkOperation};
    use crate::trajectory::{TrajectoryTerminalStateV1, TrajectoryTrigger};

    use crate::control::CONTROL_PUBLICATION_TEST_MUTEX as PLUGIN_TEST_MUTEX;

    fn configured_router() -> PluginConfig {
        PluginConfig {
            components: vec![ComponentSpec::new(live_router_config()).into()],
            ..PluginConfig::default()
        }
    }

    fn live_router_config() -> RouterConfig {
        let value = json!({
            "mode": "shadow",
            "project_id": "plugin-lifecycle-test",
            "max_evidence_records": 16,
            "pools": [{
                "id": "plugin-live",
                "api_family": "openai_chat_completions",
                "anchor_models": ["anchor"],
                "anchor_revision": "anchor-r1",
                "sampling_probability": 1.0,
                "max_candidates_per_sample": 1,
                "lookahead": {
                    "primary_llm_completions": 1,
                    "deadline_seconds": 30,
                    "max_events_per_window": 32,
                    "max_bytes_per_window": 65536
                },
                "concurrency": {"shadow": 1, "judge": 1, "max_pending": 4},
                "judge": {
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
                },
                "candidates": [{
                    "id": "candidate",
                    "model": "candidate-model",
                    "model_revision": "candidate-r1",
                    "cost_rank": 0
                }]
            }]
        });
        let report = validate_router_config(value.as_object().unwrap());
        assert!(!report.has_errors(), "{:?}", report.diagnostics);
        report.config.unwrap()
    }

    fn active_router_config() -> RouterConfig {
        let mut value = serde_json::to_value(live_router_config()).unwrap();
        value["mode"] = json!("active");
        value["max_evidence_records"] = json!(128);
        value["embedders"] = json!([{
            "id": "active-embedder",
            "base_url": "http://127.0.0.1:8080/v1",
            "model": "embedder-model",
            "provider_revision": "embedder-r1",
            "dimensions": 16,
            "timeout_ms": 1_000
        }]);
        value["pools"][0]["learning"] = json!({
            "version": 1,
            "embedder": "active-embedder",
            "top_k": 1,
            "radius": 1.0,
            "min_points": 1,
            "min_independent_roots": 1,
            "min_effective_samples": 1.0,
            "min_coverage": 0.0,
            "time_decay_half_life_seconds": 3_600.0,
            "prior_success": 1.0,
            "prior_failure": 1.0,
            "familywise_credible_level": 0.95,
            "promotion_lower_bound": 0.9,
            "retention_lower_bound": 0.8,
            "holdout_probability": 0.2,
            "active_canary_fraction": 0.4
        });
        value["pools"][0]["outcome"] = json!({
            "version": 1,
            "success_matchers": [{
                "event_kind": "scope_end", "category": "agent", "name": "completed",
                "terminal_status": "ok", "metadata_equals": {}
            }],
            "failure_matchers": [{
                "event_kind": "scope_end", "category": "agent", "name": "completed",
                "terminal_status": "error", "metadata_equals": {}
            }],
            "completion_disposition": "success",
            "error_disposition": "failure",
            "tool_failure_disposition": "failure",
            "end_of_run_disposition": "ignore",
            "max_attribution_seconds": 600,
            "actual_outcome_half_life_seconds": 1_800,
            "anchor_shadow_half_life_seconds": 1_800,
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
        });
        let report = validate_router_config(value.as_object().unwrap());
        assert!(!report.has_errors(), "{:?}", report.diagnostics);
        report.config.unwrap()
    }

    fn configured_live_router() -> PluginConfig {
        PluginConfig {
            components: vec![ComponentSpec::new(live_router_config()).into()],
            ..PluginConfig::default()
        }
    }

    #[derive(Clone)]
    struct RuntimeCapture {
        sink: Arc<InMemoryTrajectorySink>,
        delivery: Arc<InMemoryTrajectoryDelivery>,
        runtime: Weak<RouterRuntime>,
        state: Weak<()>,
    }

    type RuntimeCaptures = Arc<StdMutex<Vec<RuntimeCapture>>>;

    #[derive(Clone)]
    struct RealLedgerRuntimeCapture {
        process_instance_id: Uuid,
        runtime: Weak<RouterRuntime>,
        state: Weak<()>,
    }

    type RealLedgerRuntimeCaptures = Arc<StdMutex<Vec<RealLedgerRuntimeCapture>>>;

    fn configured_real_router(database_path: &std::path::Path, project_id: &str) -> PluginConfig {
        let mut config = live_router_config();
        config.project_id = Some(project_id.into());
        config.database_path = database_path.to_string_lossy().into_owned();
        PluginConfig {
            components: vec![ComponentSpec::new(config).into()],
            ..PluginConfig::default()
        }
    }

    fn start_captured_real_ledger_runtime(
        config: RouterConfig,
        captures: &RealLedgerRuntimeCaptures,
    ) -> std::result::Result<Arc<RouterRuntime>, String> {
        let activated = LedgerRepository::activate(&config)
            .map_err(|error| format!("{} ({})", error, error.code()))?;
        let process_instance_id = activated.identity.process_instance_id;
        let runtime = RouterRuntime::start_with_activated_ledger(config, activated)?;
        captures.lock().unwrap().push(RealLedgerRuntimeCapture {
            process_instance_id,
            runtime: Arc::downgrade(&runtime),
            state: runtime.test_state_probe(),
        });
        Ok(runtime)
    }

    fn install_runtime_capture() -> RuntimeCaptures {
        let captures = Arc::new(StdMutex::new(Vec::new()));
        let captured = captures.clone();
        *TEST_RUNTIME_STARTER.lock().unwrap() = Some(Arc::new(move |config| {
            let sink = Arc::new(InMemoryTrajectorySink::new(16));
            let delivery = Arc::new(InMemoryTrajectoryDelivery::new(16));
            let runtime =
                RouterRuntime::start_with_test_storage(config, sink.clone(), delivery.clone())?;
            let state = runtime.test_state_probe();
            captured.lock().unwrap().push(RuntimeCapture {
                sink,
                delivery,
                runtime: Arc::downgrade(&runtime),
                state,
            });
            Ok(runtime)
        }));
        captures
    }

    fn runtime_capture(captures: &RuntimeCaptures, index: usize) -> RuntimeCapture {
        captures.lock().unwrap()[index].clone()
    }

    struct CountingReplay {
        capability: LlmReplayCapability,
        starts: Arc<AtomicUsize>,
    }

    impl LlmReplayTransport for CountingReplay {
        fn capability(&self) -> &LlmReplayCapability {
            &self.capability
        }

        fn start(&self, _request: LlmRequest) -> FlowResult<LlmReplayCall> {
            self.starts.fetch_add(1, Ordering::SeqCst);
            Err(FlowError::Internal(
                "Spec 04 lifecycle tests must never start replay".into(),
            ))
        }
    }

    struct CountingReplayFactory {
        replay: Arc<CountingReplay>,
    }

    impl LlmReplayFactory for CountingReplayFactory {
        fn build(
            &self,
            _context: &nemo_relay::api::llm::LlmExecutionContextSnapshot,
        ) -> FlowResult<Arc<dyn LlmReplayTransport>> {
            Ok(self.replay.clone())
        }
    }

    fn counting_replay() -> (Arc<CountingReplay>, Arc<AtomicUsize>) {
        let starts = Arc::new(AtomicUsize::new(0));
        (
            Arc::new(CountingReplay {
                capability: LlmReplayCapability {
                    contract_version: LLM_REPLAY_CONTRACT_VERSION,
                    api_family: LlmApiFamily::OpenAIChatCompletions,
                    transport_identity: "plugin-lifecycle-test".into(),
                },
                starts: starts.clone(),
            }),
            starts,
        )
    }

    fn live_request() -> LlmRequest {
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({
                "model": "anchor",
                "messages": [{"role": "user", "content": "hello"}]
            }),
        }
    }

    fn live_response() -> Json {
        json!({
            "id": "chatcmpl-plugin-lifecycle",
            "model": "anchor",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "answer"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3}
        })
    }

    fn immediate_next(response: Json, owner: Arc<()>) -> LlmExecutionNextFn {
        Arc::new(move |_request| {
            let response = response.clone();
            let owner = owner.clone();
            Box::pin(async move {
                drop(owner);
                Ok(response)
            })
        })
    }

    async fn execute_live_anchor(
        parent: ScopeHandle,
        replay: Arc<CountingReplay>,
        next: LlmExecutionNextFn,
    ) -> FlowResult<Json> {
        let replay_factory: Arc<dyn LlmReplayFactory> = Arc::new(CountingReplayFactory { replay });
        llm_call_execute_v2(
            LlmCallExecuteV2Params::builder()
                .name("plugin-live-anchor")
                .request(live_request())
                .func(next)
                .api_family(LlmApiFamily::OpenAIChatCompletions)
                .call_role(LlmCallRole::Primary)
                .sanitized_metadata(BTreeMap::new())
                .parent(parent)
                .replay_factory(replay_factory)
                .build(),
        )
        .await
    }

    async fn wait_until(mut predicate: impl FnMut() -> bool) {
        tokio::time::timeout(Duration::from_secs(10), async {
            while !predicate() {
                tokio::task::yield_now().await;
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .expect("lifecycle state did not converge");
    }

    fn install_dispatcher_blocker() -> (Arc<AtomicBool>, Arc<Barrier>) {
        let started = Arc::new(AtomicBool::new(false));
        let release = Arc::new(Barrier::new(2));
        let callback_started = started.clone();
        let callback_release = release.clone();
        register_subscriber(
            "router-plugin-lifecycle-blocker",
            Arc::new(move |_| {
                if !callback_started.swap(true, Ordering::SeqCst) {
                    callback_release.wait();
                }
            }),
        )
        .unwrap();
        emit_scope_event(
            EmitMarkEventParams::builder()
                .name("router-plugin-lifecycle-blocker")
                .build(),
        )
        .unwrap();
        (started, release)
    }

    fn last_test_runtime() -> std::sync::Weak<RouterRuntime> {
        LAST_TEST_RUNTIME
            .lock()
            .unwrap()
            .as_ref()
            .expect("Router registration should record its runtime")
            .clone()
    }

    fn reset_core_router_test_state() {
        let _ = clear_plugin_configuration();
        let _ = deregister_router_component();
        *global_context().write().unwrap() = NemoRelayContextState::new();
        *LAST_TEST_RUNTIME.lock().unwrap() = None;
        *TEST_RUNTIME_STARTER.lock().unwrap() = None;
    }

    struct ForeignRouter;

    struct FailAfterRouter;

    impl Plugin for ForeignRouter {
        fn plugin_kind(&self) -> &str {
            ROUTER_PLUGIN_KIND
        }

        fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
            Vec::new()
        }

        fn register<'a>(
            &'a self,
            _plugin_config: &Map<String, Json>,
            _context: &'a mut PluginRegistrationContext,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async { Ok(()) })
        }
    }

    impl Plugin for FailAfterRouter {
        fn plugin_kind(&self) -> &str {
            "router.test.fail_after"
        }

        fn validate(&self, _plugin_config: &Map<String, Json>) -> Vec<ConfigDiagnostic> {
            Vec::new()
        }

        fn register<'a>(
            &'a self,
            _plugin_config: &Map<String, Json>,
            _context: &'a mut PluginRegistrationContext,
        ) -> Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
            Box::pin(async {
                Err(PluginError::RegistrationFailed(
                    "intentional failure after Router registration".into(),
                ))
            })
        }
    }

    #[tokio::test(flavor = "current_thread")]
    async fn exact_static_registration_is_idempotent() {
        let _guard = PLUGIN_TEST_MUTEX.lock().await;
        let _ = deregister_router_component();
        register_router_component().unwrap();
        register_router_component().unwrap();
        let registered = lookup_plugin(ROUTER_PLUGIN_KIND).unwrap();
        assert!(Arc::ptr_eq(&registered, &router_plugin()));
        assert!(deregister_router_component());
        assert!(!deregister_router_component());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn foreign_collision_fails_and_survives_router_deregistration() {
        let _guard = PLUGIN_TEST_MUTEX.lock().await;
        let _ = deregister_router_component();
        let foreign: Arc<dyn Plugin> = Arc::new(ForeignRouter);
        register_plugin(foreign.clone()).unwrap();
        assert!(register_router_component().is_err());
        assert!(!deregister_router_component());
        let registered = lookup_plugin(ROUTER_PLUGIN_KIND).unwrap();
        assert!(Arc::ptr_eq(&registered, &foreign));
        assert!(deregister_plugin_if(&foreign));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn plugin_is_singleton_and_validation_is_side_effect_free() {
        let _guard = PLUGIN_TEST_MUTEX.lock().await;
        let _ = deregister_router_component();
        let plugin = router_plugin();
        assert!(!plugin.allows_multiple_components());
        let config = serde_json::to_value(RouterConfig::default()).unwrap();
        let diagnostics = plugin.validate(config.as_object().unwrap());
        assert!(
            diagnostics
                .iter()
                .all(|diagnostic| diagnostic.level != DiagnosticLevel::Error)
        );
        assert!(lookup_plugin(ROUTER_PLUGIN_KIND).is_none());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn valid_active_config_remains_fenced_before_runtime_dispatch_lands() {
        let _guard = PLUGIN_TEST_MUTEX.lock().await;
        let started = Arc::new(AtomicBool::new(false));
        let started_in_runtime = started.clone();
        *TEST_RUNTIME_STARTER.lock().unwrap() = Some(Arc::new(move |_| {
            started_in_runtime.store(true, Ordering::SeqCst);
            Err("Active runtime starter must remain fenced".into())
        }));

        let plugin = router_plugin();
        let config = serde_json::to_value(active_router_config()).unwrap();
        let diagnostics = plugin.validate(config.as_object().unwrap());
        assert!(
            diagnostics
                .iter()
                .all(|diagnostic| diagnostic.level != DiagnosticLevel::Error),
            "{diagnostics:?}"
        );
        let mut context = PluginRegistrationContext::with_namespace("router-active-fence:");
        let error = plugin
            .register(config.as_object().unwrap(), &mut context)
            .await
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("runtime dispatch is not implemented")
        );
        assert!(!started.load(Ordering::SeqCst));
        assert!(context.into_registrations().is_empty());
        *TEST_RUNTIME_STARTER.lock().unwrap() = None;
    }

    #[tokio::test(flavor = "current_thread")]
    async fn duplicate_router_components_are_rejected_by_core_validation() {
        let _guard = PLUGIN_TEST_MUTEX.lock().await;
        let _ = deregister_router_component();
        register_router_component().unwrap();
        let component: PluginComponentSpec = ComponentSpec::new(RouterConfig::default()).into();
        let report = validate_plugin_config(&PluginConfig {
            components: vec![component.clone(), component],
            ..PluginConfig::default()
        });
        assert!(report.has_errors());
        assert!(
            report
                .diagnostics
                .iter()
                .any(|diagnostic| diagnostic.code == "plugin.duplicate_component")
        );
        assert!(deregister_router_component());
    }

    #[tokio::test(flavor = "current_thread")]
    async fn core_commit_opens_provider_admission_and_teardown_closes_it() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        register_router_component().unwrap();

        initialize_plugins_exact(configured_router()).await.unwrap();
        let runtime = last_test_runtime().upgrade().unwrap();
        let gate = runtime.test_provider_admission();
        assert_eq!(
            gate.phase(),
            crate::provider_admission::ProviderAdmissionPhase::Open
        );
        let starts = AtomicUsize::new(0);
        assert_eq!(
            gate.start_owned(|| starts.fetch_add(1, Ordering::AcqRel)),
            Ok(0)
        );

        clear_plugin_configuration().unwrap();

        assert_eq!(
            gate.phase(),
            crate::provider_admission::ProviderAdmissionPhase::Closed
        );
        assert_eq!(
            gate.start_owned(|| starts.fetch_add(1, Ordering::AcqRel)),
            Err(crate::provider_admission::ProviderStartRefusal::Closed)
        );
        assert_eq!(starts.load(Ordering::Acquire), 1);
        assert!(active_plugin_report().is_none());
        assert!(deregister_router_component());
        reset_core_router_test_state();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn registration_owns_lifecycle_before_subscriber_and_v2_intercept() {
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        let plugin = router_plugin();
        let config = serde_json::to_value(live_router_config()).unwrap();
        let mut context = PluginRegistrationContext::with_namespace("router-test:");
        plugin
            .register(config.as_object().unwrap(), &mut context)
            .await
            .unwrap();
        let mut registrations = context.into_registrations();
        assert_eq!(registrations.len(), 3);
        assert_eq!(registrations[0].kind, ROUTER_PLUGIN_KIND);
        assert_eq!(registrations[0].name, "router-test:runtime");
        assert_eq!(registrations[1].name, "router-test:events");
        assert_eq!(registrations[2].name, "router-test:execution");
        rollback_registrations(&mut registrations);
        assert!(registrations.is_empty());
        *global_context().write().unwrap() = NemoRelayContextState::new();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn activation_failure_precedes_every_registration() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        *TEST_RUNTIME_STARTER.lock().unwrap() = Some(Arc::new(|_| {
            Err("Router ledger activation failed before registration".into())
        }));

        let plugin = router_plugin();
        let config = serde_json::to_value(live_router_config()).unwrap();
        let mut context = PluginRegistrationContext::with_namespace("router-activation-failure:");
        assert!(
            plugin
                .register(config.as_object().unwrap(), &mut context)
                .await
                .is_err()
        );
        assert!(context.into_registrations().is_empty());

        *TEST_RUNTIME_STARTER.lock().unwrap() = None;
        *global_context().write().unwrap() = NemoRelayContextState::new();
    }

    #[test]
    fn abandoned_blocking_activation_terminalizes_its_process_identity() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let mut config = live_router_config();
        config.project_id = Some("abandoned-activation-test".into());
        config.database_path = database_path.to_string_lossy().into_owned();
        let activated = LedgerRepository::activate(&config).unwrap();
        let process_instance_id = activated.identity.process_instance_id;
        let (sender, receiver) = tokio::sync::oneshot::channel();
        drop(receiver);

        deliver_ledger_activation(sender, Ok(PendingLedgerActivation::new(activated)));

        let connection = rusqlite::Connection::open(database_path).unwrap();
        let stopped = connection
            .query_row(
                "SELECT count(*) FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'stopped'",
                [process_instance_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(stopped, 1);
    }

    #[test]
    fn received_then_canceled_activation_terminalizes_its_process_identity() {
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let mut config = live_router_config();
        config.project_id = Some("received-canceled-activation-test".into());
        config.database_path = database_path.to_string_lossy().into_owned();
        let activated = LedgerRepository::activate(&config).unwrap();
        let process_instance_id = activated.identity.process_instance_id;
        let (sender, receiver) = tokio::sync::oneshot::channel();

        deliver_ledger_activation(sender, Ok(PendingLedgerActivation::new(activated)));
        drop(receiver);

        let connection = rusqlite::Connection::open(database_path).unwrap();
        let stopped = connection
            .query_row(
                "SELECT count(*) FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'stopped'",
                [process_instance_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(stopped, 1);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn off_mode_has_no_runtime_registration_or_database_side_effect() {
        const CHILD_ENV: &str = "NEMO_RELAY_ROUTER_OFF_MODE_CHILD";
        if std::env::var_os(CHILD_ENV).is_none() {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    "plugin_component::tests::off_mode_has_no_runtime_registration_or_database_side_effect",
                    "--nocapture",
                ])
                .env(CHILD_ENV, "1")
                .status()
                .unwrap();
            assert!(status.success());
            return;
        }

        assert_eq!(crate::sqlite_vec_extension::registration_status(), None);
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        *TEST_RUNTIME_STARTER.lock().unwrap() = None;
        let temporary = tempdir().unwrap();
        let database_path = temporary.path().join("must-not-exist/router.db");
        let mut config = live_router_config();
        config.mode = RouterMode::Off;
        config.database_path = database_path.to_string_lossy().into_owned();
        config.embedders.push(crate::config::EmbedderConfig {
            id: "off-embedder".into(),
            base_url: "http://127.0.0.1:9/v1".into(),
            model: "off-embedding-model".into(),
            provider_revision: "off-r1".into(),
            dimensions: 8,
            api_key_env: Some("CERTAINLY_NOT_A_REAL_OFF_SECRET".into()),
            timeout_ms: 1_000,
            max_in_flight: 1,
            batch_size: 1,
            unknown_fields: BTreeMap::new(),
        });
        config.pools[0].learning = Some(crate::config::LearningConfig::minimal("off-embedder"));

        let plugin = router_plugin();
        let config = serde_json::to_value(config).unwrap();
        let mut context = PluginRegistrationContext::with_namespace("router-off:");
        plugin
            .register(config.as_object().unwrap(), &mut context)
            .await
            .unwrap();
        assert!(!database_path.exists());
        assert!(context.into_registrations().is_empty());
        assert!(!database_path.exists());
        assert_eq!(crate::sqlite_vec_extension::registration_status(), None);
        *global_context().write().unwrap() = NemoRelayContextState::new();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn intercept_collision_rolls_back_the_started_runtime() {
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        register_llm_execution_intercept_v2(
            "router-collision:execution",
            0,
            Arc::new(|_, _, request, _, next| next(request)),
        )
        .unwrap();

        let plugin = router_plugin();
        let config = serde_json::to_value(live_router_config()).unwrap();
        let mut context = PluginRegistrationContext::with_namespace("router-collision:");
        assert!(
            plugin
                .register(config.as_object().unwrap(), &mut context)
                .await
                .is_err()
        );
        let mut registrations = context.into_registrations();
        assert_eq!(registrations.len(), 2);
        assert_eq!(registrations[0].name, "router-collision:runtime");
        assert_eq!(registrations[1].name, "router-collision:events");
        rollback_registrations(&mut registrations);
        assert!(registrations.is_empty());

        assert!(deregister_llm_execution_intercept("router-collision:execution").unwrap());
        *global_context().write().unwrap() = NemoRelayContextState::new();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn real_ledger_intercept_collision_stops_the_activated_process() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        register_llm_execution_intercept_v2(
            "router-real-collision:execution",
            0,
            Arc::new(|_, _, request, _, next| next(request)),
        )
        .unwrap();

        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let mut config = live_router_config();
        config.project_id = Some("real-registration-collision-test".into());
        config.database_path = database_path.to_string_lossy().into_owned();
        let activated = LedgerRepository::activate(&config).unwrap();
        let process_instance_id = activated.identity.process_instance_id;
        let runtime = RouterRuntime::start_with_activated_ledger(config, activated).unwrap();
        let mut context = PluginRegistrationContext::with_namespace("router-real-collision:");

        assert!(
            register_started_runtime(runtime, &mut context)
                .await
                .is_err()
        );
        let mut registrations = context.into_registrations();
        assert_eq!(registrations.len(), 2);
        rollback_registrations(&mut registrations);
        assert!(registrations.is_empty());

        let connection = rusqlite::Connection::open(database_path).unwrap();
        let stopped = connection
            .query_row(
                "SELECT count(*) FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'stopped'",
                [process_instance_id.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert_eq!(stopped, 1);

        assert!(deregister_llm_execution_intercept("router-real-collision:execution").unwrap());
        *global_context().write().unwrap() = NemoRelayContextState::new();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn later_component_failure_stops_router_process_during_core_rollback() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let mut config = live_router_config();
        config.project_id = Some("later-component-rollback-test".into());
        config.database_path = database_path.to_string_lossy().into_owned();
        config.embedders.push(crate::config::EmbedderConfig {
            id: "rollback-embedder".into(),
            base_url: "http://127.0.0.1:9/v1".into(),
            model: "rollback-embedding-model".into(),
            provider_revision: "rollback-r1".into(),
            dimensions: 8,
            api_key_env: None,
            timeout_ms: 1_000,
            max_in_flight: 1,
            batch_size: 1,
            unknown_fields: BTreeMap::new(),
        });
        config.pools[0].learning =
            Some(crate::config::LearningConfig::minimal("rollback-embedder"));
        let process_instance_id = Arc::new(StdMutex::new(None));
        let captured_process_instance_id = process_instance_id.clone();
        let provider_gate = Arc::new(StdMutex::new(None));
        let captured_provider_gate = provider_gate.clone();
        let provider_watcher = Arc::new(StdMutex::new(None));
        let captured_provider_watcher = provider_watcher.clone();
        let provider_starts = Arc::new(AtomicUsize::new(0));
        let captured_provider_starts = provider_starts.clone();
        *TEST_RUNTIME_STARTER.lock().unwrap() = Some(Arc::new(move |config| {
            let activated =
                LedgerRepository::activate(&config).map_err(|error| error.to_string())?;
            *captured_process_instance_id.lock().unwrap() =
                Some(activated.identity.process_instance_id);
            let runtime = RouterRuntime::start_with_activated_ledger(config, activated)?;
            let gate = runtime.test_provider_admission();
            *captured_provider_gate.lock().unwrap() = Some(gate.clone());
            let starts = captured_provider_starts.clone();
            *captured_provider_watcher.lock().unwrap() = Some(tokio::spawn(async move {
                let phase = gate
                    .wait_for_change(crate::provider_admission::ProviderAdmissionPhase::Pending)
                    .await;
                if phase == crate::provider_admission::ProviderAdmissionPhase::Open {
                    let _ = gate.start_owned(|| starts.fetch_add(1, Ordering::AcqRel));
                }
                phase
            }));
            Ok(runtime)
        }));

        register_router_component().unwrap();
        let failing: Arc<dyn Plugin> = Arc::new(FailAfterRouter);
        register_plugin(failing.clone()).unwrap();
        let result = initialize_plugins_exact(PluginConfig {
            components: vec![
                ComponentSpec::new(config).into(),
                PluginComponentSpec::new("router.test.fail_after"),
            ],
            ..PluginConfig::default()
        })
        .await;
        assert!(result.is_err());
        assert!(active_plugin_report().is_none());

        let watcher = provider_watcher
            .lock()
            .unwrap()
            .take()
            .expect("provider phase watcher must start");
        let observed_phase = watcher.await.unwrap();
        assert_eq!(
            observed_phase,
            crate::provider_admission::ProviderAdmissionPhase::Closed
        );
        let gate = provider_gate
            .lock()
            .unwrap()
            .clone()
            .expect("Router startup must publish its provider gate");
        assert_eq!(
            gate.phase(),
            crate::provider_admission::ProviderAdmissionPhase::Closed
        );
        assert_eq!(
            gate.start_owned(|| provider_starts.fetch_add(1, Ordering::AcqRel)),
            Err(crate::provider_admission::ProviderStartRefusal::Closed)
        );
        assert_eq!(provider_starts.load(Ordering::Acquire), 0);

        let process_instance_id = process_instance_id
            .lock()
            .unwrap()
            .expect("Router activation must record its process identity");
        assert!(deregister_plugin_if(&failing));
        assert!(deregister_router_component());
        *TEST_RUNTIME_STARTER.lock().unwrap() = None;
        *global_context().write().unwrap() = NemoRelayContextState::new();

        let connection = rusqlite::Connection::open(database_path).unwrap();
        wait_until(|| {
            connection
                .query_row(
                    "SELECT count(*) FROM process_instance_state_events
                     WHERE process_instance_id = ?1 AND state = 'stopped'",
                    [process_instance_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap()
                == 1
        })
        .await;
        assert_eq!(
            connection
                .query_row("SELECT count(*) FROM vector_index_manifest", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap(),
            0
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn subscriber_collision_leaves_only_lifecycle_for_rollback() {
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        *global_context().write().unwrap() = NemoRelayContextState::new();
        register_subscriber("router-subscriber-collision:events", Arc::new(|_| {})).unwrap();

        let plugin = router_plugin();
        let config = serde_json::to_value(live_router_config()).unwrap();
        let mut context = PluginRegistrationContext::with_namespace("router-subscriber-collision:");
        assert!(
            plugin
                .register(config.as_object().unwrap(), &mut context)
                .await
                .is_err()
        );
        let mut registrations = context.into_registrations();
        assert_eq!(registrations.len(), 1);
        assert_eq!(registrations[0].name, "router-subscriber-collision:runtime");
        rollback_registrations(&mut registrations);
        assert!(registrations.is_empty());

        assert!(deregister_subscriber("router-subscriber-collision:events").unwrap());
        *global_context().write().unwrap() = NemoRelayContextState::new();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn real_core_sync_clear_aborts_live_window_and_releases_authority() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        let captures = install_runtime_capture();
        register_router_component().unwrap();
        initialize_plugins_exact(configured_live_router())
            .await
            .unwrap();
        let capture = runtime_capture(&captures, 0);
        let runtime_lifetime = capture.runtime.clone();
        let state_lifetime = capture.state.clone();

        TASK_SCOPE_STACK
            .scope(create_scope_stack(), async {
                let agent = push_scope(
                    PushScopeParams::builder()
                        .name("plugin-sync-clear-agent")
                        .scope_type(ScopeType::Agent)
                        .build(),
                )
                .unwrap();
                let (replay, replay_starts) = counting_replay();
                let replay_lifetime = Arc::downgrade(&replay);
                execute_live_anchor(
                    agent.clone(),
                    replay.clone(),
                    immediate_next(live_response(), Arc::new(())),
                )
                .await
                .unwrap();
                flush_subscribers().unwrap();
                wait_until(|| capture.sink.unresolved_pending().len() == 1).await;
                drop(replay);

                clear_plugin_configuration().unwrap();
                wait_until(|| state_lifetime.upgrade().is_none()).await;

                assert!(runtime_lifetime.upgrade().is_none());
                assert!(replay_lifetime.upgrade().is_none());
                assert_eq!(replay_starts.load(Ordering::SeqCst), 0);
                assert_eq!(capture.sink.unresolved_pending().len(), 1);
                assert!(capture.sink.terminal_payloads().is_empty());
                assert!(capture.delivery.delivered_anchor_ids().is_empty());
                pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
            })
            .await;

        assert!(active_plugin_report().is_none());
        assert!(deregister_router_component());
        reset_core_router_test_state();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn real_core_async_clear_drains_and_releases_the_router_runtime() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        register_router_component().unwrap();
        initialize_plugins_exact(configured_router()).await.unwrap();
        let runtime = last_test_runtime();

        clear_plugin_configuration_async(Duration::from_secs(1))
            .await
            .unwrap();

        assert!(active_plugin_report().is_none());
        assert!(runtime.upgrade().is_none());
        assert!(deregister_router_component());
        *global_context().write().unwrap() = NemoRelayContextState::new();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn real_core_replacement_releases_the_previous_router_runtime() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        register_router_component().unwrap();
        initialize_plugins_exact(configured_router()).await.unwrap();
        let previous = last_test_runtime();

        initialize_plugins_exact(configured_router()).await.unwrap();
        let replacement = last_test_runtime();

        assert!(previous.upgrade().is_none());
        assert!(replacement.upgrade().is_some());
        clear_plugin_configuration_async(Duration::from_secs(1))
            .await
            .unwrap();
        assert!(replacement.upgrade().is_none());
        assert!(deregister_router_component());
        *global_context().write().unwrap() = NemoRelayContextState::new();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_ledger_replacement_stops_old_process_before_successor_start() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let activations = Arc::new(StdMutex::new(Vec::new()));
        let captured_activations = activations.clone();
        *TEST_RUNTIME_STARTER.lock().unwrap() = Some(Arc::new(move |config| {
            let activated = LedgerRepository::activate(&config)
                .map_err(|error| format!("{} ({})", error, error.code()))?;
            let process_instance_id = activated.identity.process_instance_id;
            let runtime = RouterRuntime::start_with_activated_ledger(config, activated)?;
            captured_activations
                .lock()
                .unwrap()
                .push((process_instance_id, Arc::downgrade(&runtime)));
            Ok(runtime)
        }));
        register_router_component().unwrap();
        let mut router_config = live_router_config();
        router_config.project_id = Some("real-ledger-replacement-test".into());
        router_config.database_path = database_path.to_string_lossy().into_owned();
        let configured = || PluginConfig {
            components: vec![ComponentSpec::new(router_config.clone()).into()],
            ..PluginConfig::default()
        };

        initialize_plugins_exact(configured()).await.unwrap();
        let (old_process, old_runtime) = activations.lock().unwrap()[0].clone();
        initialize_plugins_exact(configured()).await.unwrap();
        let (successor_process, successor_runtime) = activations.lock().unwrap()[1].clone();

        assert!(old_runtime.upgrade().is_none());
        assert!(successor_runtime.upgrade().is_some());
        let connection = rusqlite::Connection::open(&database_path).unwrap();
        let old_stop_seq = connection
            .query_row(
                "SELECT event_seq FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'stopped'",
                [old_process.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        let successor_start_seq = connection
            .query_row(
                "SELECT event_seq FROM process_instance_state_events
                 WHERE process_instance_id = ?1 AND state = 'started'",
                [successor_process.to_string()],
                |row| row.get::<_, i64>(0),
            )
            .unwrap();
        assert!(old_stop_seq < successor_start_seq);

        clear_plugin_configuration_async(Duration::from_secs(5))
            .await
            .unwrap();
        assert!(successor_runtime.upgrade().is_none());
        assert!(deregister_router_component());
        reset_core_router_test_state();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_ledger_replacement_timeout_opens_no_successor() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let captures = Arc::new(StdMutex::new(Vec::new()));
        let captured = captures.clone();
        *TEST_RUNTIME_STARTER.lock().unwrap() = Some(Arc::new(move |config| {
            start_captured_real_ledger_runtime(config, &captured)
        }));
        register_router_component().unwrap();
        let config = configured_real_router(&database_path, "replacement-timeout-test");
        initialize_plugins_exact(config.clone()).await.unwrap();
        let old = captures.lock().unwrap()[0].clone();
        let database_lock = rusqlite::Connection::open(&database_path).unwrap();
        database_lock.execute_batch("BEGIN IMMEDIATE").unwrap();

        let error = initialize_plugins_exact_with_options(
            config,
            PluginInitializationOptions {
                shutdown_timeout: Duration::from_millis(25),
            },
        )
        .await
        .unwrap_err();

        assert!(
            error.to_string().contains("deadline"),
            "unexpected replacement error: {error}"
        );
        assert!(active_plugin_report().is_none());
        assert_eq!(captures.lock().unwrap().len(), 1);
        assert!(old.runtime.upgrade().is_none());
        database_lock.execute_batch("ROLLBACK").unwrap();
        wait_until(|| old.state.upgrade().is_none()).await;
        let connection = rusqlite::Connection::open(&database_path).unwrap();
        let process_counts = connection
            .query_row(
                "SELECT
                    (SELECT count(*) FROM process_instances),
                    (SELECT count(*) FROM process_instance_state_events
                     WHERE process_instance_id = ?1 AND state = 'stopped')",
                [old.process_instance_id.to_string()],
                |row| Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?)),
            )
            .unwrap();
        assert_eq!(process_counts, (1, 0));

        assert!(deregister_router_component());
        reset_core_router_test_state();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_ledger_replacement_drain_error_opens_no_successor() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let captures = Arc::new(StdMutex::new(Vec::new()));
        let captured = captures.clone();
        *TEST_RUNTIME_STARTER.lock().unwrap() = Some(Arc::new(move |config| {
            start_captured_real_ledger_runtime(config, &captured)
        }));
        register_router_component().unwrap();
        let config = configured_real_router(&database_path, "replacement-drain-error-test");
        initialize_plugins_exact(config.clone()).await.unwrap();
        let old = captures.lock().unwrap()[0].clone();
        let runtime = old.runtime.upgrade().unwrap();
        runtime.test_cancel_retention_worker();
        wait_until(|| runtime.test_retention_worker_finished()).await;
        drop(runtime);

        let error = initialize_plugins_exact(config).await.unwrap_err();

        assert!(
            error.to_string().contains("retention task failed"),
            "unexpected replacement error: {error}"
        );
        assert!(active_plugin_report().is_none());
        assert_eq!(captures.lock().unwrap().len(), 1);
        assert!(old.runtime.upgrade().is_none());
        wait_until(|| old.state.upgrade().is_none()).await;
        let connection = rusqlite::Connection::open(&database_path).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM process_instance_state_events
                     WHERE process_instance_id = ?1 AND state = 'stopped'",
                    [old.process_instance_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            0
        );

        assert!(deregister_router_component());
        reset_core_router_test_state();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_successor_restores_fresh_real_ledger_runtime_and_transport() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let captures = Arc::new(StdMutex::new(Vec::new()));
        let captured = captures.clone();
        let activation_attempt = Arc::new(AtomicUsize::new(0));
        let captured_attempt = activation_attempt.clone();
        *TEST_RUNTIME_STARTER.lock().unwrap() = Some(Arc::new(move |config| {
            let attempt = captured_attempt.fetch_add(1, Ordering::SeqCst);
            if attempt == 1 {
                return Err("intentional successor activation failure".into());
            }
            start_captured_real_ledger_runtime(config, &captured)
        }));
        register_router_component().unwrap();
        let config = configured_real_router(&database_path, "replacement-restoration-test");
        initialize_plugins_exact(config.clone()).await.unwrap();
        let old = captures.lock().unwrap()[0].clone();

        TASK_SCOPE_STACK
            .scope(create_scope_stack(), async {
                let agent = push_scope(
                    PushScopeParams::builder()
                        .name("failed-successor-restoration-agent")
                        .scope_type(ScopeType::Agent)
                        .build(),
                )
                .unwrap();
                let (old_replay, old_starts) = counting_replay();
                let old_transport = Arc::downgrade(&old_replay);
                execute_live_anchor(
                    agent.clone(),
                    old_replay.clone(),
                    immediate_next(live_response(), Arc::new(())),
                )
                .await
                .unwrap();
                flush_subscribers().unwrap();
                wait_until(|| {
                    rusqlite::Connection::open(&database_path)
                        .unwrap()
                        .query_row("SELECT count(*) FROM anchors", [], |row| {
                            row.get::<_, i64>(0)
                        })
                        .unwrap()
                        >= 1
                })
                .await;
                drop(old_replay);
                assert!(old_transport.upgrade().is_some());

                let error = initialize_plugins_exact(config.clone()).await.unwrap_err();
                assert!(
                    error
                        .to_string()
                        .contains("intentional successor activation failure")
                );
                assert!(active_plugin_report().is_some());
                assert_eq!(activation_attempt.load(Ordering::SeqCst), 3);
                let restored = captures.lock().unwrap()[1].clone();
                assert_ne!(old.process_instance_id, restored.process_instance_id);
                assert!(old.runtime.upgrade().is_none());
                wait_until(|| old.state.upgrade().is_none()).await;
                assert!(old_transport.upgrade().is_none());
                assert_eq!(old_starts.load(Ordering::SeqCst), 0);
                assert!(restored.runtime.upgrade().is_some());

                let connection = rusqlite::Connection::open(&database_path).unwrap();
                let old_stop_seq = connection
                    .query_row(
                        "SELECT event_seq FROM process_instance_state_events
                         WHERE process_instance_id = ?1 AND state = 'stopped'",
                        [old.process_instance_id.to_string()],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap();
                let restored_start_seq = connection
                    .query_row(
                        "SELECT event_seq FROM process_instance_state_events
                         WHERE process_instance_id = ?1 AND state = 'started'",
                        [restored.process_instance_id.to_string()],
                        |row| row.get::<_, i64>(0),
                    )
                    .unwrap();
                assert!(old_stop_seq < restored_start_seq);
                drop(connection);

                let (restored_replay, restored_starts) = counting_replay();
                let restored_transport = Arc::downgrade(&restored_replay);
                execute_live_anchor(
                    agent.clone(),
                    restored_replay.clone(),
                    immediate_next(live_response(), Arc::new(())),
                )
                .await
                .unwrap();
                flush_subscribers().unwrap();
                wait_until(|| {
                    rusqlite::Connection::open(&database_path)
                        .unwrap()
                        .query_row("SELECT count(*) FROM anchors", [], |row| {
                            row.get::<_, i64>(0)
                        })
                        .unwrap()
                        >= 2
                })
                .await;
                drop(restored_replay);
                assert!(restored_transport.upgrade().is_some());

                clear_plugin_configuration_async(Duration::from_secs(5))
                    .await
                    .unwrap();
                wait_until(|| restored.state.upgrade().is_none()).await;
                assert!(restored.runtime.upgrade().is_none());
                assert!(restored_transport.upgrade().is_none());
                assert_eq!(restored_starts.load(Ordering::SeqCst), 0);
                pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
            })
            .await;

        assert!(deregister_router_component());
        reset_core_router_test_state();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn failed_successor_and_restoration_surface_both_errors() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        let temporary = tempdir().unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;

            std::fs::set_permissions(temporary.path(), std::fs::Permissions::from_mode(0o700))
                .unwrap();
        }
        let database_path = temporary.path().join("ledger/router.db");
        let captures = Arc::new(StdMutex::new(Vec::new()));
        let captured = captures.clone();
        let activation_attempt = Arc::new(AtomicUsize::new(0));
        let captured_attempt = activation_attempt.clone();
        *TEST_RUNTIME_STARTER.lock().unwrap() =
            Some(Arc::new(move |config| {
                match captured_attempt.fetch_add(1, Ordering::SeqCst) {
                    0 => start_captured_real_ledger_runtime(config, &captured),
                    1 => Err("intentional successor activation failure".into()),
                    _ => Err("intentional restoration activation failure".into()),
                }
            }));
        register_router_component().unwrap();
        let config = configured_real_router(&database_path, "replacement-restore-failure-test");
        initialize_plugins_exact(config.clone()).await.unwrap();
        let old = captures.lock().unwrap()[0].clone();

        let error = initialize_plugins_exact(config).await.unwrap_err();

        let message = error.to_string();
        assert!(message.contains("intentional successor activation failure"));
        assert!(message.contains("previous plugin configuration could not be restored"));
        assert!(message.contains("intentional restoration activation failure"));
        assert!(active_plugin_report().is_none());
        assert_eq!(activation_attempt.load(Ordering::SeqCst), 3);
        assert_eq!(captures.lock().unwrap().len(), 1);
        assert!(old.runtime.upgrade().is_none());
        wait_until(|| old.state.upgrade().is_none()).await;
        let connection = rusqlite::Connection::open(&database_path).unwrap();
        assert_eq!(
            connection
                .query_row(
                    "SELECT count(*) FROM process_instance_state_events
                     WHERE process_instance_id = ?1 AND state = 'stopped'",
                    [old.process_instance_id.to_string()],
                    |row| row.get::<_, i64>(0),
                )
                .unwrap(),
            1
        );

        assert!(deregister_router_component());
        reset_core_router_test_state();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn real_core_expired_drain_aborts_and_releases_the_router_runtime() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        register_router_component().unwrap();
        initialize_plugins_exact(configured_router()).await.unwrap();
        let runtime = last_test_runtime();

        let error = clear_plugin_configuration_async(Duration::ZERO)
            .await
            .unwrap_err();

        assert!(error.to_string().contains("shared deadline expired"));
        assert!(active_plugin_report().is_none());
        assert!(runtime.upgrade().is_none());
        assert!(deregister_router_component());
        *global_context().write().unwrap() = NemoRelayContextState::new();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_core_async_clear_flushes_queued_events_and_awaits_terminal_ack() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();

        TASK_SCOPE_STACK
            .scope(create_scope_stack(), async {
                let (blocker_started, release_blocker) = install_dispatcher_blocker();
                wait_until(|| blocker_started.load(Ordering::SeqCst)).await;

                let captures = install_runtime_capture();
                register_router_component().unwrap();
                initialize_plugins_exact(configured_live_router())
                    .await
                    .unwrap();
                let capture = runtime_capture(&captures, 0);
                capture
                    .sink
                    .set_delay(SinkOperation::Pending, Duration::from_millis(150));
                capture
                    .sink
                    .set_delay(SinkOperation::Terminal, Duration::from_millis(75));
                let runtime = capture.runtime.upgrade().unwrap();
                let agent = push_scope(
                    PushScopeParams::builder()
                        .name("plugin-live-agent")
                        .scope_type(ScopeType::Agent)
                        .build(),
                )
                .unwrap();

                let (replay, replay_starts) = counting_replay();
                let replay_lifetime = Arc::downgrade(&replay);
                let next_owner = Arc::new(());
                let next_lifetime = Arc::downgrade(&next_owner);
                let raw_response = live_response();
                let result = execute_live_anchor(
                    agent.clone(),
                    replay.clone(),
                    immediate_next(raw_response.clone(), next_owner.clone()),
                )
                .await
                .unwrap();
                assert_eq!(result, raw_response);
                emit_scope_event(
                    EmitMarkEventParams::builder()
                        .name("queued-after-anchor-boundary")
                        .parent(&agent)
                        .build(),
                )
                .unwrap();
                drop(next_owner);
                assert!(next_lifetime.upgrade().is_none());
                drop(replay);
                assert!(replay_lifetime.upgrade().is_some());
                wait_until(|| capture.sink.operation_attempts(SinkOperation::Pending) == 1).await;
                assert!(capture.sink.unresolved_pending().is_empty());

                let runtime_lifetime = capture.runtime.clone();
                let state_lifetime = capture.state.clone();
                let clear = tokio::spawn(clear_plugin_configuration_async(Duration::from_secs(2)));
                wait_until(|| !runtime.test_accepting()).await;
                assert!(!clear.is_finished());
                drop(runtime);

                let released_at = Instant::now();
                release_blocker.wait();
                clear.await.unwrap().unwrap();
                assert!(released_at.elapsed() >= Duration::from_millis(50));

                wait_until(|| state_lifetime.upgrade().is_none()).await;
                assert!(runtime_lifetime.upgrade().is_none());
                assert!(replay_lifetime.upgrade().is_none());
                assert_eq!(replay_starts.load(Ordering::SeqCst), 0);
                assert!(capture.sink.unresolved_pending().is_empty());
                let terminals = capture.sink.terminal_payloads();
                assert_eq!(terminals.len(), 1);
                assert_eq!(
                    terminals[0].state,
                    TrajectoryTerminalStateV1::Closed {
                        trigger: TrajectoryTrigger::Shutdown,
                    }
                );
                assert_eq!(
                    capture.delivery.delivered_anchor_ids(),
                    vec![terminals[0].pending.anchor_id]
                );
                assert!(
                    terminals[0]
                        .events
                        .iter()
                        .all(|event| event.event_uuid != terminals[0].pending.anchor_call_uuid)
                );
                assert!(terminals[0].events.iter().any(|event| {
                    event.name == "queued-after-anchor-boundary"
                        && event.parent_uuid == Some(agent.uuid)
                        && event.scope_phase.is_none()
                }));

                assert!(deregister_subscriber("router-plugin-lifecycle-blocker").unwrap());
                pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
            })
            .await;

        assert!(active_plugin_report().is_none());
        assert!(deregister_router_component());
        reset_core_router_test_state();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_core_clear_linearizes_paused_post_next_and_releases_authority() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        let captures = install_runtime_capture();
        register_router_component().unwrap();
        initialize_plugins_exact(configured_live_router())
            .await
            .unwrap();
        let capture = runtime_capture(&captures, 0);
        let runtime = capture.runtime.upgrade().unwrap();
        let runtime_lifetime = capture.runtime.clone();
        let state_lifetime = capture.state.clone();
        let scope_stack = create_scope_stack();

        TASK_SCOPE_STACK
            .scope(scope_stack.clone(), async {
                let agent = push_scope(
                    PushScopeParams::builder()
                        .name("plugin-paused-next-agent")
                        .scope_type(ScopeType::Agent)
                        .build(),
                )
                .unwrap();
                let (replay, replay_starts) = counting_replay();
                let replay_lifetime = Arc::downgrade(&replay);
                let (entered_tx, entered_rx) = oneshot::channel();
                let (release_tx, release_rx) = oneshot::channel();
                let entered = Arc::new(StdMutex::new(Some(entered_tx)));
                let release = Arc::new(StdMutex::new(Some(release_rx)));
                let next_owner = Arc::new(());
                let next_lifetime = Arc::downgrade(&next_owner);
                let raw_response = live_response();
                let next: LlmExecutionNextFn = Arc::new({
                    let entered = entered.clone();
                    let release = release.clone();
                    let next_owner = next_owner.clone();
                    let raw_response = raw_response.clone();
                    move |_| {
                        let entered = entered.lock().unwrap().take().unwrap();
                        let release = release.lock().unwrap().take().unwrap();
                        let next_owner = next_owner.clone();
                        let raw_response = raw_response.clone();
                        Box::pin(async move {
                            let _ = entered.send(());
                            let _ = release.await;
                            drop(next_owner);
                            Ok(raw_response)
                        })
                    }
                });
                drop(next_owner);

                let call = tokio::spawn(TASK_SCOPE_STACK.scope(scope_stack.clone(), {
                    let agent = agent.clone();
                    async move { execute_live_anchor(agent, replay, next).await }
                }));
                entered_rx.await.unwrap();

                let clear = tokio::spawn(clear_plugin_configuration_async(Duration::from_secs(2)));
                wait_until(|| !runtime.test_accepting()).await;
                drop(runtime);
                clear.await.unwrap().unwrap();
                assert!(active_plugin_report().is_none());
                assert!(runtime_lifetime.upgrade().is_some());
                assert!(state_lifetime.upgrade().is_some());

                release_tx.send(()).unwrap();
                assert_eq!(call.await.unwrap().unwrap(), raw_response);
                wait_until(|| state_lifetime.upgrade().is_none()).await;
                assert!(runtime_lifetime.upgrade().is_none());
                assert!(replay_lifetime.upgrade().is_none());
                assert!(next_lifetime.upgrade().is_none());
                assert_eq!(replay_starts.load(Ordering::SeqCst), 0);
                assert!(capture.sink.unresolved_pending().is_empty());
                assert!(capture.sink.terminal_payloads().is_empty());
                assert!(capture.delivery.delivered_anchor_ids().is_empty());

                pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
            })
            .await;

        assert!(deregister_router_component());
        reset_core_router_test_state();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_core_replacement_drains_live_window_before_installing_successor() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();

        TASK_SCOPE_STACK
            .scope(create_scope_stack(), async {
                let (blocker_started, release_blocker) = install_dispatcher_blocker();
                wait_until(|| blocker_started.load(Ordering::SeqCst)).await;
                let captures = install_runtime_capture();
                register_router_component().unwrap();
                initialize_plugins_exact(configured_live_router())
                    .await
                    .unwrap();
                let previous = runtime_capture(&captures, 0);
                previous
                    .sink
                    .set_delay(SinkOperation::Terminal, Duration::from_millis(50));
                let previous_runtime = previous.runtime.upgrade().unwrap();
                let previous_state = previous.state.clone();
                let agent = push_scope(
                    PushScopeParams::builder()
                        .name("plugin-replacement-agent")
                        .scope_type(ScopeType::Agent)
                        .build(),
                )
                .unwrap();
                let (replay, replay_starts) = counting_replay();
                let replay_lifetime = Arc::downgrade(&replay);
                execute_live_anchor(
                    agent.clone(),
                    replay.clone(),
                    immediate_next(live_response(), Arc::new(())),
                )
                .await
                .unwrap();
                drop(replay);

                let replacement = tokio::spawn(initialize_plugins_exact(configured_live_router()));
                wait_until(|| !previous_runtime.test_accepting()).await;
                assert!(!replacement.is_finished());
                drop(previous_runtime);
                release_blocker.wait();
                replacement.await.unwrap().unwrap();

                wait_until(|| previous_state.upgrade().is_none()).await;
                assert!(previous.runtime.upgrade().is_none());
                assert!(replay_lifetime.upgrade().is_none());
                assert_eq!(replay_starts.load(Ordering::SeqCst), 0);
                assert!(previous.sink.unresolved_pending().is_empty());
                let terminals = previous.sink.terminal_payloads();
                assert_eq!(terminals.len(), 1);
                assert_eq!(
                    terminals[0].state,
                    TrajectoryTerminalStateV1::Closed {
                        trigger: TrajectoryTrigger::Shutdown,
                    }
                );
                assert_eq!(
                    previous.delivery.delivered_anchor_ids(),
                    vec![terminals[0].pending.anchor_id]
                );

                let successor = runtime_capture(&captures, 1);
                assert!(successor.runtime.upgrade().is_some());
                clear_plugin_configuration_async(Duration::from_secs(2))
                    .await
                    .unwrap();
                wait_until(|| successor.state.upgrade().is_none()).await;
                assert!(successor.runtime.upgrade().is_none());

                assert!(deregister_subscriber("router-plugin-lifecycle-blocker").unwrap());
                pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
            })
            .await;

        assert!(deregister_router_component());
        reset_core_router_test_state();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_core_timed_out_live_drain_aborts_and_releases_runtime_state() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        let captures = install_runtime_capture();
        register_router_component().unwrap();
        initialize_plugins_exact(configured_live_router())
            .await
            .unwrap();
        let capture = runtime_capture(&captures, 0);
        let runtime_lifetime = capture.runtime.clone();
        let state_lifetime = capture.state.clone();

        TASK_SCOPE_STACK
            .scope(create_scope_stack(), async {
                let agent = push_scope(
                    PushScopeParams::builder()
                        .name("plugin-timeout-agent")
                        .scope_type(ScopeType::Agent)
                        .build(),
                )
                .unwrap();
                let (replay, replay_starts) = counting_replay();
                let replay_lifetime = Arc::downgrade(&replay);
                execute_live_anchor(
                    agent.clone(),
                    replay.clone(),
                    immediate_next(live_response(), Arc::new(())),
                )
                .await
                .unwrap();
                flush_subscribers().unwrap();
                wait_until(|| capture.sink.unresolved_pending().len() == 1).await;
                capture
                    .sink
                    .set_delay(SinkOperation::Terminal, Duration::from_secs(30));
                drop(replay);

                let error = clear_plugin_configuration_async(Duration::from_millis(20))
                    .await
                    .unwrap_err();
                assert!(
                    error.to_string().contains("deadline"),
                    "unexpected clear error: {error}"
                );
                wait_until(|| state_lifetime.upgrade().is_none()).await;
                assert!(runtime_lifetime.upgrade().is_none());
                assert!(replay_lifetime.upgrade().is_none());
                assert_eq!(replay_starts.load(Ordering::SeqCst), 0);
                assert_eq!(capture.sink.unresolved_pending().len(), 1);
                assert!(capture.sink.terminal_payloads().is_empty());
                assert!(capture.delivery.delivered_anchor_ids().is_empty());

                pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
            })
            .await;

        assert!(active_plugin_report().is_none());
        assert!(deregister_router_component());
        reset_core_router_test_state();
    }

    #[tokio::test(flavor = "current_thread")]
    async fn dropping_live_router_registration_context_releases_all_authority() {
        let _plugin_guard = PLUGIN_TEST_MUTEX.lock().await;
        let _global_guard = crate::TEST_GLOBAL_CONTEXT_MUTEX.lock().await;
        reset_core_router_test_state();
        let captures = install_runtime_capture();
        let plugin = router_plugin();
        let config = serde_json::to_value(live_router_config()).unwrap();
        let mut context = PluginRegistrationContext::with_namespace("router-drop:");
        plugin
            .register(config.as_object().unwrap(), &mut context)
            .await
            .unwrap();
        let capture = runtime_capture(&captures, 0);
        let runtime_lifetime = capture.runtime.clone();
        let state_lifetime = capture.state.clone();

        TASK_SCOPE_STACK
            .scope(create_scope_stack(), async {
                let agent = push_scope(
                    PushScopeParams::builder()
                        .name("plugin-context-drop-agent")
                        .scope_type(ScopeType::Agent)
                        .build(),
                )
                .unwrap();
                let (replay, replay_starts) = counting_replay();
                let replay_lifetime = Arc::downgrade(&replay);
                execute_live_anchor(
                    agent.clone(),
                    replay.clone(),
                    immediate_next(live_response(), Arc::new(())),
                )
                .await
                .unwrap();
                flush_subscribers().unwrap();
                wait_until(|| capture.sink.unresolved_pending().len() == 1).await;
                drop(replay);

                drop(context);
                wait_until(|| state_lifetime.upgrade().is_none()).await;

                assert!(runtime_lifetime.upgrade().is_none());
                assert!(replay_lifetime.upgrade().is_none());
                assert_eq!(replay_starts.load(Ordering::SeqCst), 0);
                assert_eq!(capture.sink.unresolved_pending().len(), 1);
                assert!(capture.sink.terminal_payloads().is_empty());
                assert!(capture.delivery.delivered_anchor_ids().is_empty());
                pop_scope(PopScopeParams::builder().handle_uuid(&agent.uuid).build()).unwrap();
            })
            .await;

        reset_core_router_test_state();
    }
}
