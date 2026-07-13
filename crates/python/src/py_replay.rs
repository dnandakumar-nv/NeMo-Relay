// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python-owned replay transports for V2 managed LLM calls.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};

use nemo_relay::api::llm::{LlmApiFamily, LlmExecutionContextSnapshot, LlmRequest};
use nemo_relay::api::runtime::{
    LlmReplayCall, LlmReplayCapability, LlmReplayFactory, LlmReplayTransport,
};
use nemo_relay::error::{FlowError, Result as FlowResult};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyMappingProxy, PyTuple};
use pyo3_async_runtimes::TaskLocals;
use tokio::sync::oneshot;

use crate::convert::{json_to_py, py_to_json};
use crate::py_types::PyLLMRequest;

/// Convert an optional Python factory into the Core replay-factory contract.
pub(crate) fn wrap_py_replay_factory(
    py: Python<'_>,
    factory: Option<&Bound<'_, PyAny>>,
) -> PyResult<Option<Arc<dyn LlmReplayFactory>>> {
    let Some(factory) = factory.filter(|factory| !factory.is_none()) else {
        return Ok(None);
    };
    if !factory.is_callable() {
        return Err(pyo3::exceptions::PyTypeError::new_err(
            "replay_factory must be callable",
        ));
    }

    let locals = TaskLocals::with_running_loop(py)?.copy_context(py)?;
    Ok(Some(Arc::new(PyReplayFactory {
        factory: factory.clone().unbind(),
        locals,
    })))
}

struct PyReplayFactory {
    factory: Py<PyAny>,
    locals: TaskLocals,
}

impl LlmReplayFactory for PyReplayFactory {
    fn build(
        &self,
        context: &LlmExecutionContextSnapshot,
    ) -> FlowResult<Arc<dyn LlmReplayTransport>> {
        Python::attach(|py| {
            let context = serde_json::to_value(context).map_err(|_| {
                FlowError::Internal("failed to serialize replay context".to_string())
            })?;
            let context = frozen_json_to_py(py, &context)
                .map_err(|_| FlowError::Internal("failed to convert replay context".to_string()))?;
            let descriptor = self
                .factory
                .bind(py)
                .call1((context,))
                .map_err(|_| FlowError::Internal("Python replay factory raised".to_string()))?;

            if descriptor
                .hasattr(pyo3::intern!(py, "__await__"))
                .unwrap_or(false)
            {
                if descriptor
                    .hasattr(pyo3::intern!(py, "cancel"))
                    .unwrap_or(false)
                {
                    descriptor.call_method0(pyo3::intern!(py, "cancel")).ok();
                } else if descriptor
                    .hasattr(pyo3::intern!(py, "close"))
                    .unwrap_or(false)
                {
                    descriptor.call_method0(pyo3::intern!(py, "close")).ok();
                }
                return Err(FlowError::InvalidArgument(
                    "Python replay factory must return synchronously".to_string(),
                ));
            }
            let descriptor = descriptor.cast::<PyDict>().map_err(|_| {
                FlowError::InvalidArgument(
                    "Python replay factory must return a descriptor mapping".to_string(),
                )
            })?;

            let contract_version = required_item(descriptor, "contract_version")?
                .extract::<u32>()
                .map_err(|_| invalid_descriptor("contract_version must be an integer"))?;
            let family = required_item(descriptor, "api_family")?
                .extract::<String>()
                .map_err(|_| invalid_descriptor("api_family must be a string"))?;
            let api_family =
                serde_json::from_value::<LlmApiFamily>(serde_json::Value::String(family))
                    .map_err(|_| invalid_descriptor("api_family is unsupported"))?;
            let transport_identity = required_item(descriptor, "transport_identity")?
                .extract::<String>()
                .map_err(|_| invalid_descriptor("transport_identity must be a string"))?;
            let replay = required_item(descriptor, "replay")?;
            if !replay.is_callable() {
                return Err(invalid_descriptor("replay must be callable"));
            }

            Ok(Arc::new(PyReplayTransport {
                capability: LlmReplayCapability {
                    contract_version,
                    api_family,
                    transport_identity,
                },
                _descriptor: descriptor.clone().unbind().into_any(),
                replay: Arc::new(replay.clone().unbind()),
                locals: self.locals.clone(),
            }) as Arc<dyn LlmReplayTransport>)
        })
    }
}

fn frozen_json_to_py(py: Python<'_>, value: &serde_json::Value) -> PyResult<Py<PyAny>> {
    match value {
        serde_json::Value::Array(values) => {
            let values = values
                .iter()
                .map(|value| frozen_json_to_py(py, value))
                .collect::<PyResult<Vec<_>>>()?;
            Ok(PyTuple::new(py, values)?.unbind().into_any())
        }
        serde_json::Value::Object(values) => {
            let dict = PyDict::new(py);
            for (key, value) in values {
                dict.set_item(key, frozen_json_to_py(py, value)?)?;
            }
            Ok(PyMappingProxy::new(py, dict.as_mapping())
                .unbind()
                .into_any())
        }
        scalar => json_to_py(py, scalar),
    }
}

fn required_item<'py>(descriptor: &Bound<'py, PyDict>, key: &str) -> FlowResult<Bound<'py, PyAny>> {
    descriptor
        .get_item(key)
        .map_err(|_| invalid_descriptor("descriptor lookup failed"))?
        .ok_or_else(|| invalid_descriptor(&format!("missing {key}")))
}

fn invalid_descriptor(message: &str) -> FlowError {
    FlowError::InvalidArgument(format!("invalid Python replay descriptor: {message}"))
}

struct PyReplayTransport {
    capability: LlmReplayCapability,
    // Retain the complete descriptor because hosts may attach private state to it.
    _descriptor: Py<PyAny>,
    replay: Arc<Py<PyAny>>,
    locals: TaskLocals,
}

impl LlmReplayTransport for PyReplayTransport {
    fn capability(&self) -> &LlmReplayCapability {
        &self.capability
    }

    fn start(&self, request: LlmRequest) -> FlowResult<LlmReplayCall> {
        let (tx, rx) = oneshot::channel();
        let state = Arc::new(ReplayTaskState::default());
        let start_state = Arc::clone(&state);
        let start_replay = Arc::clone(&self.replay);
        let start_locals = self.locals.clone();
        pyo3_async_runtimes::tokio::get_runtime().spawn_blocking(move || {
            Python::attach(|py| {
                // Dropping the callback closes its result sender if scheduling
                // fails, so the Rust future reports an operational error.
                let _ = Py::new(
                    py,
                    ReplayStart {
                        replay: start_replay,
                        request: Some(request),
                        result_tx: Some(tx),
                        state: start_state,
                    },
                )
                .and_then(|callback| call_soon_threadsafe(&start_locals, py, callback.into_any()));
            });
        });

        let result = async move {
            let value = rx
                .await
                .map_err(|_| {
                    FlowError::Internal("Python replay task ended without a result".to_string())
                })?
                .map_err(replay_py_error)?;
            Python::attach(|py| py_to_json(value.bind(py))).map_err(replay_py_error)
        };

        let cancel_locals = self.locals.clone();
        let cancel = move || {
            request_cancel(&state);
            pyo3_async_runtimes::tokio::get_runtime().spawn_blocking(move || {
                Python::attach(|py| {
                    let scheduled = Py::new(
                        py,
                        ReplayCancel {
                            state: Arc::clone(&state),
                            invoked: false,
                        },
                    )
                    .and_then(|callback| {
                        call_soon_threadsafe(&cancel_locals, py, callback.into_any())
                    });
                    if scheduled.is_err() {
                        cancel_task_without_loop(py, &state);
                    }
                });
            });
        };
        Ok(LlmReplayCall::new(result, cancel))
    }
}

#[derive(Default)]
struct ReplayTaskState {
    cancel_requested: AtomicBool,
    task: Mutex<ReplayTaskSlot>,
}

#[derive(Default)]
struct ReplayTaskSlot {
    task: Option<Py<PyAny>>,
    cancel_sent: bool,
}

#[pyclass]
struct ReplayStart {
    replay: Arc<Py<PyAny>>,
    request: Option<LlmRequest>,
    result_tx: Option<oneshot::Sender<PyResult<Py<PyAny>>>>,
    state: Arc<ReplayTaskState>,
}

#[pymethods]
impl ReplayStart {
    fn __call__(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.state.cancel_requested.load(Ordering::Acquire) {
            self.result_tx.take();
            self.request.take();
            return Ok(());
        }

        let Some(request) = self.request.take() else {
            return Ok(());
        };
        let request = Py::new(py, PyLLMRequest { inner: request })?;
        let awaitable = match self.replay.bind(py).call1((request,)) {
            Ok(awaitable) => awaitable,
            Err(error) => {
                send_result(&mut self.result_tx, Err(error));
                return Ok(());
            }
        };
        if !awaitable
            .hasattr(pyo3::intern!(py, "__await__"))
            .unwrap_or(false)
        {
            send_result(
                &mut self.result_tx,
                Err(pyo3::exceptions::PyTypeError::new_err(
                    "replay callable must return an awaitable",
                )),
            );
            return Ok(());
        }

        let task = match py
            .import("asyncio")?
            .getattr(pyo3::intern!(py, "ensure_future"))?
            .call1((awaitable,))
        {
            Ok(task) => task,
            Err(error) => {
                send_result(&mut self.result_tx, Err(error));
                return Ok(());
            }
        };
        let completion = match Py::new(
            py,
            ReplayComplete {
                result_tx: self.result_tx.take(),
                state: Arc::downgrade(&self.state),
            },
        ) {
            Ok(completion) => completion,
            Err(error) => {
                cancel_untracked_task(py, &task);
                send_result(&mut self.result_tx, Err(error));
                return Ok(());
            }
        };
        if task
            .call_method1(pyo3::intern!(py, "add_done_callback"), (completion,))
            .is_err()
        {
            cancel_untracked_task(py, &task);
            return Ok(());
        }

        let cancel_now = {
            let mut state = self.state.task.lock().map_err(|_| {
                pyo3::exceptions::PyRuntimeError::new_err("replay task state lock poisoned")
            })?;
            state.task = Some(task.clone().unbind());
            self.state.cancel_requested.load(Ordering::Acquire) && !state.cancel_sent
        };
        if cancel_now {
            cancel_task_or_release(py, &self.state);
        }
        Ok(())
    }
}

#[pyclass]
struct ReplayComplete {
    result_tx: Option<oneshot::Sender<PyResult<Py<PyAny>>>>,
    state: Weak<ReplayTaskState>,
}

#[pymethods]
impl ReplayComplete {
    fn __call__(&mut self, task: &Bound<'_, PyAny>) -> PyResult<()> {
        let result = task.call_method0(pyo3::intern!(task.py(), "result"));
        send_result(&mut self.result_tx, result.map(Bound::unbind));
        if let Some(state) = self.state.upgrade()
            && let Ok(mut state) = state.task.lock()
        {
            state.task.take();
        }
        Ok(())
    }
}

#[pyclass]
struct ReplayCancel {
    state: Arc<ReplayTaskState>,
    invoked: bool,
}

#[pymethods]
impl ReplayCancel {
    fn __call__(&mut self, py: Python<'_>) -> PyResult<()> {
        self.invoked = true;
        cancel_task_or_release(py, &self.state);
        Ok(())
    }
}

impl Drop for ReplayCancel {
    fn drop(&mut self) {
        if !self.invoked {
            Python::try_attach(|py| cancel_task_without_loop(py, &self.state));
        }
    }
}

fn request_cancel(state: &ReplayTaskState) {
    state.cancel_requested.store(true, Ordering::Release);
}

fn cancel_task(py: Python<'_>, state: &ReplayTaskState) -> PyResult<()> {
    let task = {
        let mut state = state.task.lock().map_err(|_| {
            pyo3::exceptions::PyRuntimeError::new_err("replay task state lock poisoned")
        })?;
        if state.cancel_sent {
            return Ok(());
        }
        let Some(task) = state.task.as_ref().map(|task| task.clone_ref(py)) else {
            return Ok(());
        };
        state.cancel_sent = true;
        task
    };
    task.bind(py).call_method0(pyo3::intern!(py, "cancel"))?;
    Ok(())
}

fn cancel_task_or_release(py: Python<'_>, state: &ReplayTaskState) {
    if cancel_task(py, state).is_err() {
        release_stalled_task(py, state);
    }
}

fn cancel_untracked_task(py: Python<'_>, task: &Bound<'_, PyAny>) {
    task.call_method0(pyo3::intern!(py, "cancel")).ok();
    finalize_stalled_task(py, task);
}

fn cancel_task_without_loop(py: Python<'_>, state: &ReplayTaskState) {
    let task = {
        let Ok(mut state) = state.task.lock() else {
            return;
        };
        if state.cancel_sent {
            return;
        }
        let Some(task) = state.task.take() else {
            return;
        };
        state.cancel_sent = true;
        task
    };

    let task = task.bind(py);
    let cancel_failed = task.call_method0(pyo3::intern!(py, "cancel")).is_err();
    let loop_closed = task
        .call_method0(pyo3::intern!(py, "get_loop"))
        .and_then(|event_loop| event_loop.call_method0(pyo3::intern!(py, "is_closed")))
        .and_then(|closed| closed.extract::<bool>())
        .unwrap_or(true);
    if cancel_failed || loop_closed {
        finalize_stalled_task(py, task);
    }
}

fn release_stalled_task(py: Python<'_>, state: &ReplayTaskState) {
    let task = state
        .task
        .lock()
        .ok()
        .and_then(|mut state| state.task.take());
    if let Some(task) = task {
        finalize_stalled_task(py, task.bind(py));
    }
}

fn finalize_stalled_task(py: Python<'_>, task: &Bound<'_, PyAny>) {
    if let Ok(coroutine) = task.call_method0(pyo3::intern!(py, "get_coro")) {
        coroutine.call_method0(pyo3::intern!(py, "close")).ok();
    }
    task.setattr(pyo3::intern!(py, "_log_destroy_pending"), false)
        .ok();
}

fn send_result(
    sender: &mut Option<oneshot::Sender<PyResult<Py<PyAny>>>>,
    result: PyResult<Py<PyAny>>,
) {
    if let Some(sender) = sender.take() {
        sender.send(result).ok();
    }
}

fn call_soon_threadsafe(locals: &TaskLocals, py: Python<'_>, callback: Py<PyAny>) -> PyResult<()> {
    let kwargs = PyDict::new(py);
    kwargs.set_item(pyo3::intern!(py, "context"), locals.context(py))?;
    locals.event_loop(py).call_method(
        pyo3::intern!(py, "call_soon_threadsafe"),
        (callback,),
        Some(&kwargs),
    )?;
    Ok(())
}

fn replay_py_error(error: PyErr) -> FlowError {
    FlowError::Internal(format!("Python replay error: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::ffi::CString;
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use nemo_relay::api::llm::{LlmAttributes, LlmCallRole, LlmTrajectoryScopeSnapshot};
    use nemo_relay::api::scope::ScopeType;
    use pyo3::types::PyModule;
    use serde_json::json;
    use uuid::Uuid;

    const API_FAMILIES: [LlmApiFamily; 3] = [
        LlmApiFamily::OpenAIChatCompletions,
        LlmApiFamily::OpenAIResponses,
        LlmApiFamily::AnthropicMessages,
    ];

    fn load_module<'py>(py: Python<'py>, code: &str) -> Bound<'py, PyModule> {
        PyModule::from_code(
            py,
            &CString::new(code).unwrap(),
            &CString::new("py_replay_tests.py").unwrap(),
            &CString::new("py_replay_tests").unwrap(),
        )
        .unwrap()
    }

    fn with_event_loop<T>(py: Python<'_>, f: impl FnOnce(Bound<'_, PyAny>) -> T) -> T {
        let asyncio = py.import("asyncio").unwrap();
        let event_loop = asyncio.call_method0("new_event_loop").unwrap();
        asyncio
            .call_method1("set_event_loop", (&event_loop,))
            .unwrap();
        let result = f(event_loop.clone().into_any());
        asyncio
            .call_method1("set_event_loop", (py.None(),))
            .unwrap();
        event_loop.call_method0("close").unwrap();
        result
    }

    fn context(api_family: LlmApiFamily) -> LlmExecutionContextSnapshot {
        let root_uuid = Uuid::now_v7();
        LlmExecutionContextSnapshot {
            call_uuid: Uuid::now_v7(),
            root_uuid,
            parent_uuid: root_uuid,
            trajectory_owner_uuid: root_uuid,
            trajectory_owner_path: vec![LlmTrajectoryScopeSnapshot {
                uuid: root_uuid,
                name: "root".to_string(),
                scope_type: ScopeType::Agent,
            }],
            api_family,
            call_role: LlmCallRole::Primary,
            attributes: LlmAttributes::empty(),
            tenant_id: Some("tenant-a".to_string()),
            agent_id: Some("agent-a".to_string()),
            sanitized_metadata: BTreeMap::from([
                ("region".to_string(), json!("us")),
                ("nested".to_string(), json!({"items": [{"value": 1}]})),
            ]),
        }
    }

    fn request(id: &str) -> LlmRequest {
        LlmRequest {
            headers: serde_json::Map::new(),
            content: json!({"id": id}),
        }
    }

    fn build_transport(
        py: Python<'_>,
        event_loop: &Bound<'_, PyAny>,
        factory: Py<PyAny>,
        api_family: LlmApiFamily,
    ) -> Arc<dyn LlmReplayTransport> {
        let factory = PyReplayFactory {
            factory,
            locals: TaskLocals::new(event_loop.clone())
                .copy_context(py)
                .unwrap(),
        };
        factory.build(&context(api_family)).unwrap()
    }

    #[test]
    fn descriptor_errors_are_classified_as_invalid_arguments() {
        assert!(matches!(
            invalid_descriptor("missing replay"),
            FlowError::InvalidArgument(_)
        ));
    }

    #[test]
    fn replay_contract_version_matches_core() {
        assert_eq!(nemo_relay::api::runtime::LLM_REPLAY_CONTRACT_VERSION, 1);
    }

    #[test]
    fn replay_factory_context_is_recursively_frozen() {
        let _python = crate::test_support::init_python_test();
        Python::attach(|py| {
            let module = load_module(
                py,
                r#"
from types import MappingProxyType

observed = []

def factory(context):
    assert isinstance(context, MappingProxyType)
    assert context["call_role"] == "primary"
    assert isinstance(context["trajectory_owner_path"], tuple)
    assert isinstance(context["trajectory_owner_path"][0], MappingProxyType)
    metadata = context["sanitized_metadata"]
    assert isinstance(metadata, MappingProxyType)
    assert isinstance(metadata["nested"], MappingProxyType)
    assert isinstance(metadata["nested"]["items"], tuple)
    assert isinstance(metadata["nested"]["items"][0], MappingProxyType)

    for target, key, value in (
        (context, "tenant_id", "changed"),
        (metadata["nested"]["items"][0], "value", 2),
    ):
        try:
            target[key] = value
        except TypeError:
            pass
        else:
            raise AssertionError("frozen mapping accepted mutation")
    observed.append(context["api_family"])

    async def replay(request):
        return request.content

    return {
        "contract_version": 1,
        "api_family": context["api_family"],
        "transport_identity": "python-frozen-context-test",
        "replay": replay,
    }
"#,
            );
            with_event_loop(py, |event_loop| {
                for api_family in API_FAMILIES {
                    let transport = build_transport(
                        py,
                        &event_loop,
                        module.getattr("factory").unwrap().unbind(),
                        api_family,
                    );
                    assert_eq!(transport.capability().api_family, api_family);
                    drop(transport);
                }
                assert_eq!(
                    module
                        .getattr("observed")
                        .unwrap()
                        .extract::<Vec<String>>()
                        .unwrap(),
                    vec![
                        "openai_chat_completions",
                        "openai_responses",
                        "anthropic_messages",
                    ]
                );
            });
        });
    }

    #[test]
    fn replay_transport_outlives_factory_and_supports_sequential_concurrent_calls() {
        let _python = crate::test_support::init_python_test();
        Python::attach(|py| {
            for api_family in API_FAMILIES {
                let module = load_module(
                    py,
                    r#"
import asyncio

def factory(context):
    call_count = {"value": 0}

    async def replay(request):
        await asyncio.sleep(0)
        call_count["value"] += 1
        content = request.content
        request_id = content["id"]
        content["id"] = "mutated-copy"
        return {"id": request.content["id"], "call": call_count["value"]}

    return {
        "contract_version": 1,
        "api_family": context["api_family"],
        "transport_identity": "python-concurrent-test",
        "replay": replay,
        "private_state": call_count,
    }
"#,
                );
                with_event_loop(py, |event_loop| {
                    let transport = build_transport(
                        py,
                        &event_loop,
                        module.getattr("factory").unwrap().unbind(),
                        api_family,
                    );
                    assert_eq!(transport.capability().api_family, api_family);
                    let driver = Arc::clone(&transport);

                    let _runtime = tokio::runtime::Runtime::new().unwrap();
                    pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                        let first = driver.start(request("first")).unwrap().await.unwrap();
                        assert_eq!(first, json!({"id": "first", "call": 1}));

                        let second = driver.start(request("second")).unwrap();
                        let third = driver.start(request("third")).unwrap();
                        let (second, third) = tokio::join!(second, third);
                        let second = second.unwrap();
                        let third = third.unwrap();
                        assert_eq!(second["id"], "second");
                        assert_eq!(third["id"], "third");
                        assert_eq!(
                            [second["call"].as_u64(), third["call"].as_u64()]
                                .into_iter()
                                .collect::<std::collections::BTreeSet<_>>(),
                            std::collections::BTreeSet::from([Some(2), Some(3)])
                        );
                        Ok(())
                    })
                    .unwrap();
                    drop(transport);
                });
            }
        });
    }

    #[test]
    fn descriptor_is_retained_for_transport_lifetime() {
        let _python = crate::test_support::init_python_test();
        Python::attach(|py| {
            for api_family in API_FAMILIES {
                let module = load_module(
                    py,
                    r#"
released = []

class Descriptor(dict):
    def __del__(self):
        released.append("released")

def factory(context):
    async def replay(request):
        return request.content

    return Descriptor(
        contract_version=1,
        api_family=context["api_family"],
        transport_identity="python-lifetime-test",
        replay=replay,
    )
"#,
                );
                with_event_loop(py, |event_loop| {
                    let transport = build_transport(
                        py,
                        &event_loop,
                        module.getattr("factory").unwrap().unbind(),
                        api_family,
                    );
                    assert_eq!(transport.capability().api_family, api_family);
                    py.import("gc").unwrap().call_method0("collect").unwrap();
                    assert_eq!(module.getattr("released").unwrap().len().unwrap(), 0);

                    drop(transport);
                    py.import("gc").unwrap().call_method0("collect").unwrap();
                    assert_eq!(module.getattr("released").unwrap().len().unwrap(), 1);
                });
            }
        });
    }

    #[test]
    fn cancellation_handle_cancels_only_its_python_task_once() {
        let _python = crate::test_support::init_python_test();
        Python::attach(|py| {
            for api_family in API_FAMILIES {
                let module = load_module(
                    py,
                    r#"
import asyncio

started = []
cancelled = []
release = asyncio.Event()

async def replay(request):
    request_id = request.content["id"]
    started.append(request_id)
    try:
        await release.wait()
    except asyncio.CancelledError:
        cancelled.append(request_id)
        raise
    return {"id": request_id}

def factory(context):
    return {
        "contract_version": 1,
        "api_family": context["api_family"],
        "transport_identity": "python-cancel-test",
        "replay": replay,
    }
"#,
                );
                with_event_loop(py, |event_loop| {
                    let transport = build_transport(
                        py,
                        &event_loop,
                        module.getattr("factory").unwrap().unbind(),
                        api_family,
                    );
                    assert_eq!(transport.capability().api_family, api_family);
                    let cancelled_call = transport.start(request("cancelled")).unwrap();
                    let cancellation = cancelled_call.cancellation_handle();
                    let repeated_cancellation = cancellation.clone();
                    let completed_call = transport.start(request("completed")).unwrap();
                    let module: Py<PyModule> = module.clone().unbind();
                    let event_loop_ref = event_loop.clone().unbind();

                    let _runtime = tokio::runtime::Runtime::new().unwrap();
                    pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                        for _ in 0..100 {
                            let started = Python::attach(|py| {
                                module.bind(py).getattr("started").unwrap().len().unwrap()
                            });
                            if started == 2 {
                                break;
                            }
                            tokio::time::sleep(Duration::from_millis(2)).await;
                        }
                        assert!(cancellation.cancel());
                        assert!(!repeated_cancellation.cancel());

                        for _ in 0..100 {
                            let cancelled = Python::attach(|py| {
                                module.bind(py).getattr("cancelled").unwrap().len().unwrap()
                            });
                            if cancelled == 1 {
                                break;
                            }
                            tokio::time::sleep(Duration::from_millis(2)).await;
                        }
                        drop(cancelled_call);
                        assert!(!cancellation.cancel());
                        Python::attach(|py| {
                            let release = module
                                .bind(py)
                                .getattr("release")
                                .unwrap()
                                .getattr("set")
                                .unwrap();
                            event_loop_ref
                                .bind(py)
                                .call_method1("call_soon_threadsafe", (release,))
                                .unwrap();
                        });

                        let completed =
                            tokio::time::timeout(Duration::from_secs(2), completed_call)
                                .await
                                .expect("completed replay timed out")
                                .unwrap();
                        assert_eq!(completed, json!({"id": "completed"}));
                        Python::attach(|py| {
                            let cancelled: Vec<String> = module
                                .bind(py)
                                .getattr("cancelled")
                                .unwrap()
                                .extract()
                                .unwrap();
                            assert_eq!(cancelled, vec!["cancelled"]);
                        });
                        Ok(())
                    })
                    .unwrap();
                    drop(transport);
                });
            }
        });
    }

    #[test]
    fn start_and_cancel_return_while_another_thread_holds_the_gil() {
        let _python = crate::test_support::init_python_test();
        Python::attach(|py| {
            let module = load_module(
                py,
                r#"
import asyncio
import weakref

started = False
cancel_calls = 0
future_ref = None

class TrackedFuture(asyncio.Future):
    def cancel(self, *args, **kwargs):
        global cancel_calls
        cancel_calls += 1
        return super().cancel(*args, **kwargs)

def replay(request):
    global started, future_ref
    future = TrackedFuture()
    started = True
    future_ref = weakref.ref(future)
    return future

def factory(context):
    return {
        "contract_version": 1,
        "api_family": context["api_family"],
        "transport_identity": "python-nonblocking-cancel-test",
        "replay": replay,
    }
"#,
            );
            with_event_loop(py, |event_loop| {
                let transport = build_transport(
                    py,
                    &event_loop,
                    module.getattr("factory").unwrap().unbind(),
                    LlmApiFamily::OpenAIResponses,
                );
                let (start_tx, start_rx) = mpsc::sync_channel(1);
                let start_transport = Arc::clone(&transport);
                let start_thread = thread::spawn(move || {
                    start_tx
                        .send(start_transport.start(request("nonblocking")))
                        .unwrap();
                });

                let call = start_rx
                    .recv_timeout(Duration::from_secs(1))
                    .expect("replay start waited for the GIL")
                    .unwrap();
                start_thread.join().unwrap();

                let module_ref = Arc::new(module.clone().unbind());
                let started_module = Arc::clone(&module_ref);
                pyo3_async_runtimes::tokio::run_until_complete(
                    event_loop.clone().into_any(),
                    async move {
                        for _ in 0..100 {
                            let started = Python::attach(|py| {
                                started_module
                                    .bind(py)
                                    .getattr("started")
                                    .unwrap()
                                    .extract::<bool>()
                                    .unwrap()
                            });
                            if started {
                                return Ok(());
                            }
                            tokio::time::sleep(Duration::from_millis(2)).await;
                        }
                        panic!("replay future was not started");
                    },
                )
                .unwrap();

                let cancellation = call.cancellation_handle();
                let repeated_cancellation = cancellation.clone();
                let (cancel_tx, cancel_rx) = mpsc::sync_channel(1);
                let cancel_thread = thread::spawn(move || {
                    cancel_tx
                        .send((cancellation.cancel(), repeated_cancellation.cancel()))
                        .unwrap();
                });
                assert_eq!(
                    cancel_rx
                        .recv_timeout(Duration::from_secs(1))
                        .expect("replay cancellation waited for the GIL"),
                    (true, false)
                );
                cancel_thread.join().unwrap();

                let canceled_module = Arc::clone(&module_ref);
                pyo3_async_runtimes::tokio::run_until_complete(
                    event_loop.clone().into_any(),
                    async move {
                        for _ in 0..100 {
                            let cancel_calls = Python::attach(|py| {
                                canceled_module
                                    .bind(py)
                                    .getattr("cancel_calls")
                                    .unwrap()
                                    .extract::<usize>()
                                    .unwrap()
                            });
                            if cancel_calls == 1 {
                                return Ok(());
                            }
                            tokio::time::sleep(Duration::from_millis(2)).await;
                        }
                        panic!("replay cancellation was not dispatched");
                    },
                )
                .unwrap();

                drop(call);
                drop(transport);
                py.import("gc").unwrap().call_method0("collect").unwrap();
                assert_eq!(
                    module_ref
                        .bind(py)
                        .getattr("cancel_calls")
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    1
                );
                assert!(
                    module_ref
                        .bind(py)
                        .getattr("future_ref")
                        .unwrap()
                        .call0()
                        .unwrap()
                        .is_none()
                );
            });
        });
    }

    #[test]
    fn cancellation_request_does_not_wait_for_the_task_state_lock() {
        let state = Arc::new(ReplayTaskState::default());
        let task = state.task.lock().unwrap();
        let request_state = Arc::clone(&state);
        let (requested_tx, requested_rx) = mpsc::sync_channel(1);
        let request_thread = thread::spawn(move || {
            request_cancel(&request_state);
            requested_tx.send(()).unwrap();
        });

        requested_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("cancellation request waited for the task state lock");
        assert!(state.cancel_requested.load(Ordering::Acquire));
        drop(task);
        request_thread.join().unwrap();
    }

    #[test]
    fn cancellation_error_releases_stalled_python_future() {
        let _python = crate::test_support::init_python_test();
        Python::attach(|py| {
            let module = load_module(
                py,
                r#"
import asyncio
import weakref

cancel_calls = 0
future_ref = None

class RaisingFuture(asyncio.Future):
    def cancel(self, *args, **kwargs):
        global cancel_calls
        cancel_calls += 1
        raise RuntimeError("cancel failed")

def make_future():
    global future_ref
    future = RaisingFuture()
    future_ref = weakref.ref(future)
    return future
"#,
            );
            with_event_loop(py, |_event_loop| {
                let future = module.getattr("make_future").unwrap().call0().unwrap();
                let state = Arc::new(ReplayTaskState {
                    cancel_requested: AtomicBool::new(true),
                    task: Mutex::new(ReplayTaskSlot {
                        task: Some(future.clone().unbind()),
                        cancel_sent: false,
                    }),
                });

                cancel_task_or_release(py, &state);

                assert_eq!(
                    module
                        .getattr("cancel_calls")
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    1
                );
                assert!(state.task.lock().unwrap().task.is_none());
                drop(future);
                py.import("gc").unwrap().call_method0("collect").unwrap();
                assert!(
                    module
                        .getattr("future_ref")
                        .unwrap()
                        .call0()
                        .unwrap()
                        .is_none()
                );
            });
        });
    }

    #[test]
    fn callback_registration_error_cancels_and_releases_python_future() {
        let _python = crate::test_support::init_python_test();
        Python::attach(|py| {
            let module = load_module(
                py,
                r#"
import asyncio
import weakref

cancel_calls = 0
future_ref = None

class RejectCallbackFuture(asyncio.Future):
    def add_done_callback(self, callback, *, context=None):
        raise RuntimeError("callback registration failed")

    def cancel(self, *args, **kwargs):
        global cancel_calls
        cancel_calls += 1
        return super().cancel(*args, **kwargs)

def replay(request):
    global future_ref
    future = RejectCallbackFuture()
    future_ref = weakref.ref(future)
    return future

def factory(context):
    return {
        "contract_version": 1,
        "api_family": "openai_responses",
        "transport_identity": "python-callback-registration-test",
        "replay": replay,
    }
"#,
            );
            with_event_loop(py, |event_loop| {
                let transport = build_transport(
                    py,
                    &event_loop,
                    module.getattr("factory").unwrap().unbind(),
                    LlmApiFamily::OpenAIResponses,
                );
                let call = transport.start(request("callback-registration")).unwrap();
                let module_ref: Py<PyModule> = module.clone().unbind();

                let _runtime = tokio::runtime::Runtime::new().unwrap();
                pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                    assert!(call.await.is_err());
                    Ok(())
                })
                .unwrap();
                drop(transport);
                py.import("gc").unwrap().call_method0("collect").unwrap();

                assert_eq!(
                    module_ref
                        .bind(py)
                        .getattr("cancel_calls")
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    1
                );
                assert!(
                    module_ref
                        .bind(py)
                        .getattr("future_ref")
                        .unwrap()
                        .call0()
                        .unwrap()
                        .is_none()
                );
            });
        });
    }

    #[test]
    fn dropping_replay_as_loop_closes_or_after_close_is_exact_and_leak_free() {
        let _python = crate::test_support::init_python_test();
        Python::attach(|py| {
            for close_before_drop in [false, true] {
                let module = load_module(
                    py,
                    r#"
import asyncio
import weakref

cancel_calls = 0
released = 0
future_ref = None

def mark_released():
    global released
    released += 1

class TrackedFuture(asyncio.Future):
    def cancel(self, *args, **kwargs):
        global cancel_calls
        cancel_calls += 1
        return super().cancel(*args, **kwargs)

def replay(request):
    global future_ref
    future = TrackedFuture()
    future_ref = weakref.ref(future)
    weakref.finalize(future, mark_released)
    return future

def factory(context):
    return {
        "contract_version": 1,
        "api_family": "openai_responses",
        "transport_identity": "python-closed-loop-test",
        "replay": replay,
    }
"#,
                );
                let asyncio = py.import("asyncio").unwrap();
                let event_loop = asyncio.call_method0("new_event_loop").unwrap();
                asyncio
                    .call_method1("set_event_loop", (&event_loop,))
                    .unwrap();
                let transport = build_transport(
                    py,
                    &event_loop,
                    module.getattr("factory").unwrap().unbind(),
                    LlmApiFamily::OpenAIResponses,
                );
                let call = transport.start(request("closed-loop")).unwrap();
                let module_ref: Py<PyModule> = module.clone().unbind();

                let _runtime = tokio::runtime::Runtime::new().unwrap();
                pyo3_async_runtimes::tokio::run_until_complete(
                    event_loop.clone().into_any(),
                    async move {
                        for _ in 0..100 {
                            let created = Python::attach(|py| {
                                !module_ref.bind(py).getattr("future_ref").unwrap().is_none()
                            });
                            if created {
                                return Ok(());
                            }
                            tokio::time::sleep(Duration::from_millis(2)).await;
                        }
                        panic!("replay future was not created");
                    },
                )
                .unwrap();

                asyncio
                    .call_method1("set_event_loop", (py.None(),))
                    .unwrap();
                if close_before_drop {
                    event_loop.call_method0("close").unwrap();
                    drop(call);
                } else {
                    drop(call);
                    event_loop.call_method0("close").unwrap();
                }
                drop(transport);
                for _ in 0..100 {
                    py.detach(|| thread::sleep(Duration::from_millis(2)));
                    py.import("gc").unwrap().call_method0("collect").unwrap();
                    let cancel_calls = module
                        .getattr("cancel_calls")
                        .unwrap()
                        .extract::<usize>()
                        .unwrap();
                    let released = module
                        .getattr("released")
                        .unwrap()
                        .extract::<usize>()
                        .unwrap();
                    if cancel_calls == 1
                        && released == 1
                        && module
                            .getattr("future_ref")
                            .unwrap()
                            .call0()
                            .unwrap()
                            .is_none()
                    {
                        break;
                    }
                }

                assert_eq!(
                    module
                        .getattr("cancel_calls")
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    1
                );
                assert!(
                    module
                        .getattr("future_ref")
                        .unwrap()
                        .call0()
                        .unwrap()
                        .is_none()
                );
                assert_eq!(
                    module
                        .getattr("released")
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    1
                );
            }
        });
    }

    #[test]
    fn replay_exception_maps_to_flow_error() {
        let _python = crate::test_support::init_python_test();
        Python::attach(|py| {
            for api_family in API_FAMILIES {
                let module = load_module(
                    py,
                    r#"
async def replay(request):
    raise RuntimeError("replay boom")

def factory(context):
    return {
        "contract_version": 1,
        "api_family": context["api_family"],
        "transport_identity": "python-error-test",
        "replay": replay,
    }
"#,
                );
                with_event_loop(py, |event_loop| {
                    let transport = build_transport(
                        py,
                        &event_loop,
                        module.getattr("factory").unwrap().unbind(),
                        api_family,
                    );
                    assert_eq!(transport.capability().api_family, api_family);
                    let call = transport.start(request("error")).unwrap();
                    let _runtime = tokio::runtime::Runtime::new().unwrap();
                    pyo3_async_runtimes::tokio::run_until_complete(event_loop, async move {
                        assert!(call.await.unwrap_err().to_string().contains("replay boom"));
                        Ok(())
                    })
                    .unwrap();
                    drop(transport);
                });
            }
        });
    }
}
