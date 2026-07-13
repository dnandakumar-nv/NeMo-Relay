// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Node.js replay-factory and replay-transport bridge.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use napi::bindgen_prelude::{FromNapiValue, ToNapiValue};
use napi::threadsafe_function::{
    ThreadSafeCallContext, ThreadsafeFunction, ThreadsafeFunctionCallMode,
};
use napi::{Env, JsFunction, JsObject, JsUnknown, NapiRaw, NapiValue};
use serde_json::{Value as Json, json};

use nemo_relay::api::llm::{
    LlmApiFamily, LlmAttributes, LlmCallRole, LlmExecutionContextSnapshot, LlmRequest,
    LlmTrajectoryScopeSnapshot,
};
use nemo_relay::api::runtime::{
    LlmReplayCall, LlmReplayCapability, LlmReplayFactory, LlmReplayTransport,
};
use nemo_relay::api::scope::ScopeType as CoreScopeType;
use nemo_relay::error::{FlowError, Result as FlowResult};

const FACTORY_FAILED: &str = "Node replay factory failed";
const INVALID_DESCRIPTOR: &str = "Node replay factory returned an invalid descriptor";

fn json_to_unknown(env: &Env, value: Json) -> napi::Result<JsUnknown> {
    // SAFETY: `env` is live for the duration of the threadsafe-function
    // callback, and the returned handle is consumed before that callback ends.
    let raw = unsafe { <Json as ToNapiValue>::to_napi_value(env.raw(), value) }?;
    // SAFETY: `raw` was created in this environment immediately above.
    Ok(unsafe { JsUnknown::from_raw_unchecked(env.raw(), raw) })
}

fn function_to_unknown(env: &Env, value: &JsFunction) -> JsUnknown {
    // SAFETY: `value` belongs to `env` and remains live while the wrapper is
    // constructed.
    unsafe { JsUnknown::from_raw_unchecked(env.raw(), value.raw()) }
}

fn object_to_unknown(env: &Env, value: &JsObject) -> JsUnknown {
    // SAFETY: `value` belongs to `env` and remains live while the wrapper is
    // constructed.
    unsafe { JsUnknown::from_raw_unchecked(env.raw(), value.raw()) }
}

fn undefined_to_unknown(env: &Env) -> napi::Result<JsUnknown> {
    let value = env.get_undefined()?;
    // SAFETY: `value` belongs to `env` and is used immediately.
    Ok(unsafe { JsUnknown::from_raw_unchecked(env.raw(), value.raw()) })
}

fn rejection_message(
    string_result: napi::Result<String>,
    object_message_result: Option<napi::Result<String>>,
) -> String {
    if let Ok(value) = string_result {
        value
    } else if let Some(message_result) = object_message_result {
        message_result.unwrap_or_else(|_| "unknown replay error".to_string())
    } else {
        "unknown replay error".to_string()
    }
}

fn parse_api_family(value: &str) -> FlowResult<LlmApiFamily> {
    serde_json::from_value(Json::String(value.to_string())).map_err(|_| {
        FlowError::InvalidArgument("Node replay descriptor has an invalid apiFamily".to_string())
    })
}

fn create_factory_wrapper(env: &Env, factory: &JsFunction) -> napi::Result<JsFunction> {
    let wrapper_factory: JsFunction = env.run_script(
        r#"((factory) => {
  const deepFreeze = (value) => {
    if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
      Object.freeze(value)
      for (const child of Object.values(value)) deepFreeze(child)
    }
    return value
  }
  return function __nemo_relay_replay_factory(error, context) {
    if (error != null) return { ok: false }
    try {
      const descriptor = factory(deepFreeze(context))
      if (descriptor == null || typeof descriptor !== 'object') return { ok: false }
      if (typeof descriptor.then === 'function') {
        // The factory contract is synchronous. Consume a rejected async result
        // so invalid factories remain fail-open under strict rejection policy.
        void Promise.resolve(descriptor).catch(() => {})
        return { ok: false }
      }
      const keys = Object.keys(descriptor).sort()
      const expected = ['apiFamily', 'contractVersion', 'replay', 'transportIdentity']
      if (keys.length !== expected.length || keys.some((key, index) => key !== expected[index])) {
        return { ok: false }
      }
      if (!Number.isInteger(descriptor.contractVersion) ||
          descriptor.contractVersion < 0 || descriptor.contractVersion > 0xffffffff) {
        return { ok: false }
      }
      if (typeof descriptor.apiFamily !== 'string' ||
          typeof descriptor.transportIdentity !== 'string' ||
          typeof descriptor.replay !== 'function') {
        return { ok: false }
      }
      return {
        ok: true,
        descriptor: Object.freeze(descriptor),
      }
    } catch (_) {
      return { ok: false }
    }
  }
})"#,
    )?;
    let wrapper = wrapper_factory.call(None, &[function_to_unknown(env, factory)])?;
    // SAFETY: The script above always returns a function.
    Ok(unsafe { wrapper.cast::<JsFunction>() })
}

fn create_replay_wrapper(env: &Env, descriptor: &JsObject) -> napi::Result<JsFunction> {
    let wrapper_factory: JsFunction = env.run_script(
        r#"((descriptor) => {
  const active = new Map()
  const deepFreeze = (value) => {
    if (value !== null && typeof value === 'object' && !Object.isFrozen(value)) {
      Object.freeze(value)
      for (const child of Object.values(value)) deepFreeze(child)
    }
    return value
  }
  const isJsonValue = (value, seen = new Set()) => {
    if (value === null) return true
    const kind = typeof value
    if (kind === 'string' || kind === 'boolean') return true
    if (kind === 'number') return Number.isFinite(value)
    if (kind !== 'object' || seen.has(value)) return false

    const prototype = Object.getPrototypeOf(value)
    if (!Array.isArray(value) && prototype !== Object.prototype && prototype !== null) {
      return false
    }
    seen.add(value)
    try {
      return Array.isArray(value)
        ? value.every((child) => isJsonValue(child, seen))
        : Object.keys(value).every((key) => isJsonValue(value[key], seen))
    } finally {
      seen.delete(value)
    }
  }
  return function __nemo_relay_replay(error, operation, resolve, reject) {
    if (error != null) {
      if (typeof reject === 'function') reject(error)
      return
    }
    const { kind, id } = operation
    if (kind === 'cancel') {
      const cancel = active.get(id)
      // `start` is queued before its LlmReplayCall can be returned and
      // dropped. A missing entry is therefore already terminal.
      if (cancel === undefined) return
      active.delete(id)
      try { cancel() } catch (_) {}
      return
    }
    if (kind !== 'start') {
      reject(new Error('invalid replay operation'))
      return
    }

    let invocation
    try {
      invocation = descriptor.replay(deepFreeze(operation.request))
      if (invocation == null || typeof invocation !== 'object') throw new Error()
      const keys = Object.keys(invocation).sort()
      if (keys.length !== 2 || keys[0] !== 'cancel' || keys[1] !== 'result') throw new Error()
      if (typeof invocation.cancel !== 'function' ||
          invocation.result == null || typeof invocation.result.then !== 'function') {
        throw new Error()
      }
    } catch (_) {
      reject(new Error('Node replay callable returned an invalid invocation'))
      return
    }

    active.set(id, invocation.cancel)
    Promise.resolve(invocation.result).then(
      (value) => {
        active.delete(id)
        try {
          if (!isJsonValue(value)) throw new Error()
          resolve(value)
        } catch (_) {
          reject(new Error('Node replay result must be valid JSON'))
        }
      },
      (reason) => {
        active.delete(id)
        reject(reason)
      },
    )
  }
})"#,
    )?;
    let wrapper = wrapper_factory.call(None, &[object_to_unknown(env, descriptor)])?;
    // SAFETY: The script above always returns a function.
    Ok(unsafe { wrapper.cast::<JsFunction>() })
}

#[derive(Clone)]
struct ReplayCompletion {
    sender: Arc<Mutex<Option<tokio::sync::oneshot::Sender<FlowResult<Json>>>>>,
}

impl ReplayCompletion {
    fn new(sender: tokio::sync::oneshot::Sender<FlowResult<Json>>) -> Self {
        Self {
            sender: Arc::new(Mutex::new(Some(sender))),
        }
    }

    fn send(&self, value: FlowResult<Json>) {
        if let Some(sender) = self.sender.lock().unwrap().take() {
            let _ = sender.send(value);
        }
    }
}

enum ReplayOperation {
    Start {
        id: String,
        request: Json,
        completion: ReplayCompletion,
    },
    Cancel {
        id: String,
    },
}

fn replay_completion_unknowns(
    env: &Env,
    completion: Option<ReplayCompletion>,
) -> napi::Result<(JsUnknown, JsUnknown)> {
    let Some(completion) = completion else {
        return Ok((undefined_to_unknown(env)?, undefined_to_unknown(env)?));
    };

    let resolve_completion = completion.clone();
    let resolve = env.create_function_from_closure("__nemo_relay_replay_resolve", move |ctx| {
        match ctx.get::<Json>(0) {
            Ok(value) => resolve_completion.send(Ok(value)),
            Err(_) => resolve_completion.send(Err(FlowError::InvalidArgument(
                "Node replay result must be valid JSON".to_string(),
            ))),
        }
        ctx.env.get_undefined()
    })?;
    let reject = env.create_function_from_closure("__nemo_relay_replay_reject", move |ctx| {
        let message = rejection_message(
            ctx.get::<String>(0),
            ctx.get::<JsObject>(0)
                .ok()
                .map(|value| value.get_named_property::<String>("message")),
        );
        completion.send(Err(FlowError::Internal(message)));
        ctx.env.get_undefined()
    })?;
    Ok((
        function_to_unknown(env, &resolve),
        function_to_unknown(env, &reject),
    ))
}

struct NodeReplayTransport {
    capability: LlmReplayCapability,
    replay: ThreadsafeFunction<ReplayOperation>,
    next_id: AtomicU64,
}

impl NodeReplayTransport {
    fn new(env: &Env, descriptor: JsObject, capability: LlmReplayCapability) -> napi::Result<Self> {
        let wrapper = create_replay_wrapper(env, &descriptor)?;
        let mut replay = env.create_threadsafe_function(
            &wrapper,
            0,
            |ctx: ThreadSafeCallContext<ReplayOperation>| {
                let (operation, completion) = match ctx.value {
                    ReplayOperation::Start {
                        id,
                        request,
                        completion,
                    } => (
                        json!({ "kind": "start", "id": id, "request": request }),
                        Some(completion),
                    ),
                    ReplayOperation::Cancel { id } => (json!({ "kind": "cancel", "id": id }), None),
                };
                let (resolve, reject) = replay_completion_unknowns(&ctx.env, completion)?;
                Ok(vec![json_to_unknown(&ctx.env, operation)?, resolve, reject])
            },
        )?;
        replay.unref(env)?;
        Ok(Self {
            capability,
            replay,
            next_id: AtomicU64::new(1),
        })
    }
}

impl LlmReplayTransport for NodeReplayTransport {
    fn capability(&self) -> &LlmReplayCapability {
        &self.capability
    }

    fn start(&self, request: LlmRequest) -> FlowResult<LlmReplayCall> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed).to_string();
        let request = serde_json::to_value(request)
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let status = self.replay.call(
            Ok(ReplayOperation::Start {
                id: id.clone(),
                request,
                completion: ReplayCompletion::new(sender),
            }),
            ThreadsafeFunctionCallMode::NonBlocking,
        );
        if status != napi::Status::Ok {
            return Err(FlowError::Internal(format!(
                "failed to queue Node replay call: {status:?}",
            )));
        }

        let replay = self.replay.clone();
        Ok(LlmReplayCall::new(
            async move {
                receiver
                    .await
                    .map_err(|_| FlowError::Internal("Node replay completion closed".to_string()))?
            },
            move || {
                let _ = replay.call(
                    Ok(ReplayOperation::Cancel { id }),
                    ThreadsafeFunctionCallMode::NonBlocking,
                );
            },
        ))
    }
}

struct FactoryCallbackResult(FlowResult<Arc<dyn LlmReplayTransport>>);

impl FromNapiValue for FactoryCallbackResult {
    unsafe fn from_napi_value(
        raw_env: napi::sys::napi_env,
        value: napi::sys::napi_value,
    ) -> napi::Result<Self> {
        let env = Env::from(raw_env);
        // SAFETY: napi-rs calls this implementation with the live return value
        // produced by `create_factory_wrapper` in `raw_env`.
        let result = unsafe { JsObject::from_raw_unchecked(raw_env, value) };
        let parsed = (|| -> napi::Result<Arc<dyn LlmReplayTransport>> {
            if !result.get_named_property::<bool>("ok")? {
                return Err(napi::Error::from_reason(FACTORY_FAILED));
            }
            let descriptor = result.get_named_property::<JsObject>("descriptor")?;
            let contract_version = descriptor.get_named_property::<u32>("contractVersion")?;
            let api_family =
                parse_api_family(&descriptor.get_named_property::<String>("apiFamily")?)
                    .map_err(|_| napi::Error::from_reason(INVALID_DESCRIPTOR))?;
            let transport_identity =
                descriptor.get_named_property::<String>("transportIdentity")?;
            descriptor.get_named_property::<JsFunction>("replay")?;
            let capability = LlmReplayCapability {
                contract_version,
                api_family,
                transport_identity,
            };
            NodeReplayTransport::new(&env, descriptor, capability)
                .map(|transport| Arc::new(transport) as Arc<dyn LlmReplayTransport>)
        })()
        .map_err(|_| FlowError::InvalidArgument(INVALID_DESCRIPTOR.to_string()));
        Ok(Self(parsed))
    }
}

/// A long-lived synchronous JavaScript replay factory.
pub(crate) struct NodeReplayFactory {
    factory: ThreadsafeFunction<Json>,
}

impl NodeReplayFactory {
    /// Capture a JavaScript factory on its owning runtime.
    pub(crate) fn new(env: &Env, factory: &JsFunction) -> napi::Result<Self> {
        let wrapper = create_factory_wrapper(env, factory)?;
        let mut factory =
            env.create_threadsafe_function(&wrapper, 0, |ctx: ThreadSafeCallContext<Json>| {
                Ok(vec![json_to_unknown(&ctx.env, ctx.value)?])
            })?;
        factory.unref(env)?;
        Ok(Self { factory })
    }
}

impl LlmReplayFactory for NodeReplayFactory {
    fn build(
        &self,
        context: &LlmExecutionContextSnapshot,
    ) -> FlowResult<Arc<dyn LlmReplayTransport>> {
        let api_family = serde_json::to_value(context.api_family)
            .map_err(|error| FlowError::Internal(error.to_string()))?;
        let trajectory_owner_path = context
            .trajectory_owner_path
            .iter()
            .map(|scope| {
                json!({
                    "uuid": scope.uuid.to_string(),
                    "name": scope.name,
                    "scopeType": scope.scope_type.as_str(),
                })
            })
            .collect::<Vec<_>>();
        let context = json!({
            "callUuid": context.call_uuid.to_string(),
            "rootUuid": context.root_uuid.to_string(),
            "parentUuid": context.parent_uuid.to_string(),
            "trajectoryOwnerUuid": context.trajectory_owner_uuid.to_string(),
            "trajectoryOwnerPath": trajectory_owner_path,
            "apiFamily": api_family,
            "callRole": context.call_role.as_str(),
            "attributes": context.attributes.bits(),
            "tenantId": context.tenant_id,
            "agentId": context.agent_id,
            "sanitizedMetadata": context.sanitized_metadata,
        });
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        let status = self.factory.call_with_return_value(
            Ok(context),
            ThreadsafeFunctionCallMode::NonBlocking,
            move |result: FactoryCallbackResult| {
                let _ = sender.send(result.0);
                Ok(())
            },
        );
        if status != napi::Status::Ok {
            return Err(FlowError::Internal(format!(
                "failed to queue Node replay factory: {status:?}",
            )));
        }
        receiver
            .recv()
            .map_err(|_| FlowError::Internal("Node replay factory completion closed".to_string()))?
    }
}

/// Exercise the opaque Node transport from binding tests without exporting a
/// public V2-intercept registration API.
pub(crate) async fn exercise_replay_factory(
    factory: Arc<NodeReplayFactory>,
    api_family: LlmApiFamily,
    requests: Vec<LlmRequest>,
    cancel_first: bool,
) -> FlowResult<Vec<Json>> {
    let root_uuid = uuid::Uuid::now_v7();
    let context = LlmExecutionContextSnapshot {
        call_uuid: uuid::Uuid::now_v7(),
        root_uuid,
        parent_uuid: root_uuid,
        trajectory_owner_uuid: root_uuid,
        trajectory_owner_path: vec![LlmTrajectoryScopeSnapshot {
            uuid: root_uuid,
            name: "node-replay-test".to_string(),
            scope_type: CoreScopeType::Agent,
        }],
        api_family,
        call_role: LlmCallRole::Primary,
        attributes: LlmAttributes::empty(),
        tenant_id: None,
        agent_id: None,
        sanitized_metadata: BTreeMap::new(),
    };
    let transport = factory.build(&context)?;
    drop(factory);

    if cancel_first {
        let mut calls = requests
            .into_iter()
            .map(|request| transport.start(request))
            .collect::<FlowResult<Vec<_>>>()?;
        if calls.is_empty() {
            return Err(FlowError::InvalidArgument(
                "cancelFirst requires at least one request".to_string(),
            ));
        }
        let cancelled_call = calls.remove(0);
        let cancellation = cancelled_call.cancellation_handle();
        let repeated_cancellation = cancellation.clone();
        if !cancellation.cancel() || repeated_cancellation.cancel() {
            return Err(FlowError::Internal(
                "replay cancellation handle was not exact-once".to_string(),
            ));
        }
        drop(cancelled_call);
        if cancellation.cancel() {
            return Err(FlowError::Internal(
                "dropping a canceled replay rearmed cancellation".to_string(),
            ));
        }
        let tasks = calls.into_iter().map(tokio::spawn).collect::<Vec<_>>();
        let mut results = Vec::with_capacity(tasks.len());
        for task in tasks {
            results.push(
                task.await
                    .map_err(|error| FlowError::Internal(error.to_string()))??,
            );
        }
        return Ok(results);
    }

    let mut requests = requests.into_iter();
    let mut results = Vec::with_capacity(requests.len());
    if let Some(request) = requests.next() {
        results.push(transport.start(request)?.await?);
    }
    let tasks = requests
        .map(|request| transport.start(request).map(tokio::spawn))
        .collect::<FlowResult<Vec<_>>>()?;
    for task in tasks {
        results.push(
            task.await
                .map_err(|error| FlowError::Internal(error.to_string()))??,
        );
    }
    Ok(results)
}
