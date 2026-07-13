# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for NeMo Relay LLM lifecycle, guardrails, intercepts, and streaming."""

import inspect
from types import MappingProxyType
from typing import Awaitable, Callable, NoReturn, cast

import pytest

from nemo_relay import (
    LLMAttributes,
    LLMHandle,
    LLMRequest,
    LLMRequestInterceptOutcome,
    PendingMarkSpec,
    ScopeEvent,
    ScopeType,
    guardrails,
    intercepts,
    llm,
    scope,
    subscribers,
)

V2_API_FAMILIES: tuple[llm.LlmApiFamily, ...] = (
    "openai_chat_completions",
    "openai_responses",
    "anthropic_messages",
)


def make_request():
    return LLMRequest({}, {"messages": [], "model": "test-model"})


def raise_runtime_error(message: str) -> NoReturn:
    raise RuntimeError(message)


def _llm_event(events, name: str, scope_category: str) -> ScopeEvent:
    return next(
        event
        for event in events
        if event.name == name
        and isinstance(event, ScopeEvent)
        and event.category == "llm"
        and event.scope_category == scope_category
    )


class TestLLM:
    def test_call_and_call_end(self):
        request = make_request()
        handle = llm.call("my_llm", request)
        assert isinstance(handle, LLMHandle)
        assert handle.name == "my_llm"
        llm.call_end(handle, {"response": "ok"})

    def test_call_with_attributes(self):
        request = make_request()
        attrs = LLMAttributes(LLMAttributes.STREAMING)
        handle = llm.call("streaming_llm", request, attributes=attrs)
        llm.call_end(handle, {})

    def test_call_with_data_metadata(self):
        request = make_request()
        handle = llm.call(
            "llm_dm",
            request,
            data={"custom": "data"},
            metadata={"trace": "xyz"},
        )
        llm.call_end(handle, {"result": "ok"}, data={"end": True})

    def test_call_with_parent(self):
        parent = scope.push("llm_parent", ScopeType.Agent)
        request = make_request()
        handle = llm.call("child_llm", request, handle=parent)
        assert handle.parent_uuid == parent.uuid
        llm.call_end(handle, {})
        scope.pop(parent)


class TestLLMAsync:
    async def test_execute_basic(self):
        # LLM execute receives an LLMRequest object
        def func(request):
            return {"model": request.content["model"]}

        request = make_request()
        result = await llm.execute("exec_llm", request, func)
        assert result["model"] == "test-model"

    async def test_execute_with_sync_func(self):
        def func(request):
            return {"echoed_messages": request.content["messages"]}

        request = make_request()
        result = await llm.execute("sync_llm", request, func)
        assert result["echoed_messages"] == []

    async def test_execute_async_func(self):
        """llm.execute should accept async functions."""

        async def func(request):
            return {"model": request.content["model"], "async": True}

        request = make_request()
        result = await llm.execute("async_exec_llm", request, func)
        assert result["model"] == "test-model"
        assert result["async"] is True

    async def test_execute_async_func_with_messages(self):
        async def func(request):
            return {"messages": request.content["messages"]}

        request = make_request()
        result = await llm.execute("async_method_llm", request, func)
        assert result["messages"] == []

    @pytest.mark.parametrize("api_family", V2_API_FAMILIES)
    async def test_execute_v2_captures_explicit_context_and_builds_factory_once(self, api_family: llm.LlmApiFamily):
        contexts: list[llm.LlmExecutionContext] = []

        async def replay(request):
            return {"replayed_model": request.content["model"]}

        def replay_factory(context: llm.LlmExecutionContext):
            contexts.append(context)
            return {
                "contract_version": 1,
                "api_family": api_family,
                "transport_identity": "python-test-transport",
                "replay": replay,
            }

        result = await llm.execute_v2(
            "v2_llm",
            make_request(),
            lambda request: {"model": request.content["model"]},
            api_family=api_family,
            call_role="primary",
            sanitized_metadata={
                "region": "us-west",
                "nested": {"items": [{"value": 1}]},
            },
            tenant_id="tenant-a",
            agent_id="agent-a",
            replay_factory=replay_factory,
        )

        assert result == {"model": "test-model"}
        assert len(contexts) == 1
        context = contexts[0]
        assert isinstance(context, MappingProxyType)
        assert context["api_family"] == api_family
        assert context["call_role"] == "primary"
        assert context["tenant_id"] == "tenant-a"
        assert context["agent_id"] == "agent-a"

        metadata = cast(dict[str, object], context["sanitized_metadata"])
        assert isinstance(metadata, MappingProxyType)
        assert metadata["region"] == "us-west"
        nested = cast(dict[str, object], metadata["nested"])
        items = cast(tuple[object, ...], nested["items"])
        item = cast(dict[str, object], items[0])
        assert isinstance(nested, MappingProxyType)
        assert isinstance(items, tuple)
        assert isinstance(item, MappingProxyType)
        with pytest.raises(TypeError):
            cast(dict[str, object], context)["tenant_id"] = "changed"
        with pytest.raises(TypeError):
            item["value"] = 2
        with pytest.raises(AttributeError):
            cast(list[object], items).append({"value": 2})

        path = cast(tuple[object, ...], context["trajectory_owner_path"])
        assert path
        assert isinstance(path, tuple)
        assert isinstance(path[0], MappingProxyType)

    @pytest.mark.parametrize("api_family", V2_API_FAMILIES)
    @pytest.mark.parametrize("failure", ["factory_error", "unsupported_contract", "family_mismatch"])
    async def test_execute_v2_factory_errors_fail_open(self, api_family: llm.LlmApiFamily, failure: str):
        async def replay(request):
            return request.content

        def replay_factory(context):
            if failure == "factory_error":
                raise_runtime_error("factory secret")
            mismatched_families: dict[llm.LlmApiFamily, llm.LlmApiFamily] = {
                "openai_chat_completions": "openai_responses",
                "openai_responses": "anthropic_messages",
                "anthropic_messages": "openai_chat_completions",
            }
            return {
                "contract_version": 99 if failure == "unsupported_contract" else 1,
                "api_family": (mismatched_families[api_family] if failure == "family_mismatch" else api_family),
                "transport_identity": "python-test-transport",
                "replay": replay,
            }

        result = await llm.execute_v2(
            "v2_fail_open",
            make_request(),
            lambda request: {"anchor": request.content["model"]},
            api_family=api_family,
            call_role="primary",
            sanitized_metadata={},
            replay_factory=cast(llm.LlmReplayFactory, replay_factory),
        )
        assert result == {"anchor": "test-model"}

    async def test_execute_v2_rejects_async_factory_without_failing_anchor(self):
        async def replay_factory(context):
            return context

        result = await llm.execute_v2(
            "v2_async_factory",
            make_request(),
            lambda request: {"anchor": True},
            api_family="openai_responses",
            call_role="primary",
            sanitized_metadata={},
            replay_factory=cast(llm.LlmReplayFactory, replay_factory),
        )
        assert result == {"anchor": True}

    def test_execute_v2_rejects_unknown_family_role_and_metadata_shape(self):
        def provider(request):
            return {"unexpected": True}

        with pytest.raises(ValueError, match="unsupported api_family"):
            llm.execute_v2(
                "bad_family",
                make_request(),
                provider,
                api_family=cast(llm.LlmApiFamily, "unknown"),
                call_role="primary",
                sanitized_metadata={},
            )
        with pytest.raises(ValueError, match="unsupported call_role"):
            llm.execute_v2(
                "bad_role",
                make_request(),
                provider,
                api_family="openai_responses",
                call_role=cast(llm.LlmCallRole, "unknown"),
                sanitized_metadata={},
            )
        with pytest.raises(TypeError, match="sanitized_metadata must be a mapping"):
            llm.execute_v2(
                "bad_metadata",
                make_request(),
                provider,
                api_family="openai_responses",
                call_role="primary",
                sanitized_metadata=cast(dict[str, str], []),
            )

    @pytest.mark.parametrize(
        ("identity", "value"),
        [("tenant_id", ""), ("agent_id", "secret=not-routing-data")],
    )
    async def test_execute_v2_rejects_invalid_identity_before_provider(self, identity, value):
        provider_called = False

        def provider(request):
            nonlocal provider_called
            provider_called = True
            return {"unexpected": True}

        execute_v2 = cast(Callable[..., Awaitable[object]], llm.execute_v2)
        with pytest.raises(RuntimeError, match=identity):
            await execute_v2(
                "bad_identity",
                make_request(),
                provider,
                api_family="openai_responses",
                call_role="primary",
                sanitized_metadata={},
                **{identity: value},
            )
        assert not provider_called

    @pytest.mark.parametrize("required", ["api_family", "call_role", "sanitized_metadata"])
    def test_execute_v2_requires_explicit_routing_fields(self, required):
        execute_v2 = cast(Callable[..., object], llm.execute_v2)
        kwargs = {
            "api_family": "openai_responses",
            "call_role": "primary",
            "sanitized_metadata": {},
        }
        del kwargs[required]

        with pytest.raises(TypeError, match=required):
            execute_v2("missing_context", make_request(), lambda request: {}, **kwargs)

    def test_execute_v1_signature_remains_unchanged(self):
        assert str(inspect.signature(llm.execute)) == (
            "(name: 'str', request: 'LLMRequest', func, *, handle=None, "
            "attributes=None, data=None, metadata=None, model_name: 'str | None' = None, "
            "codec: 'LlmCodec | None' = None, response_codec: 'LlmResponseCodec | None' = None)"
        )


class TestLLMGuardrails:
    def test_sanitize_request_guardrail(self):
        def sanitizer(request):
            # request is an LLMRequest object; must return a new LLMRequest
            headers = request.headers
            headers["X-Sanitized"] = "true"
            return LLMRequest(headers, request.content)

        guardrails.register_llm_sanitize_request("py_llm_san_req", 1, sanitizer)
        guardrails.deregister_llm_sanitize_request("py_llm_san_req")

    def test_sanitize_response_guardrail(self):
        def sanitizer(response):
            # response is a plain dict
            response["cleaned"] = True
            return response

        guardrails.register_llm_sanitize_response("py_llm_san_resp", 1, sanitizer)
        guardrails.deregister_llm_sanitize_response("py_llm_san_resp")

    def test_conditional_execution_guardrail(self):
        def checker(request):
            return None

        guardrails.register_llm_conditional_execution("py_llm_cond", 1, checker)
        guardrails.deregister_llm_conditional_execution("py_llm_cond")

    def test_conditional_execution_direct(self):
        guardrails.register_llm_conditional_execution("py_llm_cond_direct", 1, lambda request: "blocked directly")
        with pytest.raises(RuntimeError, match="guardrail rejected"):
            llm.conditional_execution(make_request())
        guardrails.deregister_llm_conditional_execution("py_llm_cond_direct")

    def test_duplicate_raises(self):
        guardrails.register_llm_sanitize_request("py_llm_dup", 1, lambda r: r)
        with pytest.raises(RuntimeError):
            guardrails.register_llm_sanitize_request("py_llm_dup", 1, lambda r: r)
        guardrails.deregister_llm_sanitize_request("py_llm_dup")

    def test_sanitize_request_callable_error_falls_back_to_original_input(self):
        events = []
        subscribers.register("py_llm_sanitize_req_sub", lambda event: events.append(event))
        guardrails.register_llm_sanitize_request(
            "py_llm_sanitize_req_fail",
            1,
            lambda request: raise_runtime_error("boom"),
        )
        try:
            request = make_request()
            handle = llm.call("llm_sanitize_req_fail", request)
            llm.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_llm_sanitize_request("py_llm_sanitize_req_fail")
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_llm_sanitize_req_sub")

        start = _llm_event(events, "llm_sanitize_req_fail", "start")
        request = make_request()
        assert start.data == {"headers": request.headers, "content": request.content}

    def test_sanitize_request_invalid_return_falls_back_to_original_input(self):
        events = []
        subscribers.register("py_llm_sanitize_req_bad_sub", lambda event: events.append(event))
        guardrails.register_llm_sanitize_request(
            "py_llm_sanitize_req_bad",
            1,
            cast(guardrails.LlmSanitizeRequestGuardrail, lambda request: object()),
        )
        try:
            request = make_request()
            handle = llm.call("llm_sanitize_req_bad", request)
            llm.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_llm_sanitize_request("py_llm_sanitize_req_bad")
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_llm_sanitize_req_bad_sub")

        start = _llm_event(events, "llm_sanitize_req_bad", "start")
        request = make_request()
        assert start.data == {"headers": request.headers, "content": request.content}

    def test_sanitize_response_callable_error_falls_back_to_original_output(self):
        events = []
        subscribers.register("py_llm_sanitize_resp_sub", lambda event: events.append(event))
        guardrails.register_llm_sanitize_response(
            "py_llm_sanitize_resp_fail",
            1,
            lambda response: raise_runtime_error("boom"),
        )
        try:
            handle = llm.call("llm_sanitize_resp_fail", make_request())
            llm.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_llm_sanitize_response("py_llm_sanitize_resp_fail")
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_llm_sanitize_resp_sub")

        end = _llm_event(events, "llm_sanitize_resp_fail", "end")
        assert end.data == {"ok": True}

    def test_sanitize_response_invalid_return_falls_back_to_original_output(self):
        events = []
        subscribers.register("py_llm_sanitize_resp_bad_sub", lambda event: events.append(event))
        guardrails.register_llm_sanitize_response(
            "py_llm_sanitize_resp_bad",
            1,
            cast(guardrails.LlmSanitizeResponseGuardrail, lambda response: object()),
        )
        try:
            handle = llm.call("llm_sanitize_resp_bad", make_request())
            llm.call_end(handle, {"ok": True})
        finally:
            guardrails.deregister_llm_sanitize_response("py_llm_sanitize_resp_bad")
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_llm_sanitize_resp_bad_sub")

        end = _llm_event(events, "llm_sanitize_resp_bad", "end")
        assert end.data == {"ok": True}

    def test_deregister_nonexistent(self):
        assert not guardrails.deregister_llm_sanitize_request("nope")
        assert not guardrails.deregister_llm_sanitize_response("nope")
        assert not guardrails.deregister_llm_conditional_execution("nope")

    def test_conditional_execution_invalid_return_type_raises(self):
        guardrails.register_llm_conditional_execution(
            "py_llm_cond_bad_type",
            1,
            cast(guardrails.LlmConditionalExecutionGuardrail, lambda request: 123),
        )
        try:
            with pytest.raises(RuntimeError, match="expected str or None"):
                llm.conditional_execution(make_request())
        finally:
            guardrails.deregister_llm_conditional_execution("py_llm_cond_bad_type")

    def test_conditional_execution_callable_error_raises(self):
        guardrails.register_llm_conditional_execution(
            "py_llm_cond_error",
            1,
            lambda request: raise_runtime_error("boom"),
        )
        try:
            with pytest.raises(RuntimeError, match="callable failed"):
                llm.conditional_execution(make_request())
        finally:
            guardrails.deregister_llm_conditional_execution("py_llm_cond_error")


class TestLLMGuardrailsAsync:
    async def test_conditional_blocks_execution(self):
        guardrails.register_llm_conditional_execution("py_llm_blocker", 1, lambda req: "LLM blocked")

        def func(request):
            return {"should": "not reach"}

        request = make_request()
        with pytest.raises(RuntimeError, match="guardrail rejected"):
            await llm.execute("blocked_llm", request, func)

        guardrails.deregister_llm_conditional_execution("py_llm_blocker")


class TestLLMIntercepts:
    def test_request_intercept(self):
        # Request intercepts now operate on LLMRequest
        intercepts.register_llm_request(
            "py_llm_req",
            1,
            False,
            lambda name, request, annotated: LLMRequestInterceptOutcome(request, annotated),
        )
        assert intercepts.deregister_llm_request("py_llm_req")

    def test_request_intercepts_direct(self):
        pending_mark = PendingMarkSpec("request.direct", data={"source": "python"})

        def intercept_fn(name, request, annotated):
            content = request.content
            content["direct"] = True
            return LLMRequestInterceptOutcome(
                LLMRequest(request.headers, content),
                annotated,
                [pending_mark],
            )

        intercepts.register_llm_request("py_llm_req_direct", 1, False, intercept_fn)
        transformed = llm.request_intercepts("direct_llm", make_request())
        intercepts.deregister_llm_request("py_llm_req_direct")

        assert transformed.request.content["direct"] is True
        assert len(transformed.pending_marks) == 1
        assert transformed.pending_marks[0].name == pending_mark.name
        assert transformed.pending_marks[0].data == pending_mark.data

    def test_request_intercept_raises_on_exception(self):
        intercepts.register_llm_request(
            "py_llm_req_raise",
            1,
            False,
            lambda name, request, annotated: raise_runtime_error("boom"),
        )
        try:
            with pytest.raises(RuntimeError, match="callable failed"):
                llm.request_intercepts("raise_llm", make_request())
        finally:
            intercepts.deregister_llm_request("py_llm_req_raise")

    def test_request_intercept_raises_on_invalid_return(self):
        intercepts.register_llm_request("py_llm_req_bad_return", 1, False, lambda name, request, annotated: object())  # type: ignore[arg-type] # ty: ignore[invalid-argument-type]
        try:
            with pytest.raises(RuntimeError, match="must return LLMRequestInterceptOutcome"):
                llm.request_intercepts("bad_return_llm", make_request())
        finally:
            intercepts.deregister_llm_request("py_llm_req_bad_return")

    def test_execution_intercept(self):
        # Execution intercepts now take LLMRequest
        intercepts.register_llm_execution(
            "py_llm_exec",
            1,
            lambda name, request, next: {"intercepted": True},
        )
        assert intercepts.deregister_llm_execution("py_llm_exec")

    def test_stream_execution_intercept(self):
        def stream_fn(request, next):
            async def gen():
                yield {"token": "test"}

            return gen()

        intercepts.register_llm_stream_execution(
            "py_llm_sexec",
            1,
            stream_fn,
        )
        assert intercepts.deregister_llm_stream_execution("py_llm_sexec")

    def test_deregister_nonexistent(self):
        assert not intercepts.deregister_llm_request("nope")
        assert not intercepts.deregister_llm_execution("nope")
        assert not intercepts.deregister_llm_stream_execution("nope")
        assert not intercepts.deregister_tool_request("nope")
        assert not intercepts.deregister_tool_execution("nope")


class TestLLMInterceptsAsync:
    async def test_request_intercept_modifies(self):
        def intercept_fn(name, request, annotated):
            # Request intercepts now operate on LLMRequest
            content = request.content
            content["intercepted"] = True
            return LLMRequestInterceptOutcome(LLMRequest(request.headers, content), annotated)

        intercepts.register_llm_request("py_llm_req_mod", 1, False, intercept_fn)

        def func(request):
            return {"saw_intercepted": request.content.get("intercepted", False)}

        request = make_request()
        result = await llm.execute("int_llm", request, func)
        assert result["saw_intercepted"] is True

        intercepts.deregister_llm_request("py_llm_req_mod")

    async def test_execution_intercept_replaces(self):
        intercepts.register_llm_execution(
            "py_llm_exec_rep",
            1,
            lambda name, request, next: {"from_intercept": True},
        )

        def original_func(request):
            return {"from_original": True}

        request = make_request()
        result = await llm.execute("exec_llm", request, original_func)
        assert result["from_intercept"] is True
        assert "from_original" not in result

        intercepts.deregister_llm_execution("py_llm_exec_rep")

    async def test_execution_intercept_can_await_next(self):
        async def middleware(name, request, next):
            updated = LLMRequest(request.headers, {**request.content, "model": "via-next"})
            result = await next(updated)
            result["from_intercept"] = True
            return result

        intercepts.register_llm_execution("py_llm_exec_next", 1, middleware)

        def original_func(request):
            return {"model": request.content["model"]}

        try:
            result = await llm.execute("exec_llm_next", make_request(), original_func)
            assert result == {"model": "via-next", "from_intercept": True}
        finally:
            intercepts.deregister_llm_execution("py_llm_exec_next")

    async def test_stream_execution_intercept_can_await_next(self):
        def middleware(request, next):
            async def gen():
                updated = LLMRequest(request.headers, {**request.content, "prefix": "wrapped"})
                stream = await next(updated)
                async for chunk in stream:
                    yield {"token": f"{updated.content['prefix']}:{chunk['token']}"}

            return gen()

        def stream_func(request):
            async def gen():
                yield {"token": request.content["model"]}
                yield {"token": "done"}

            return gen()

        intercepts.register_llm_stream_execution("py_llm_stream_next", 1, middleware)
        try:
            stream = await llm.stream_execute(
                "stream_next_llm", make_request(), stream_func, lambda chunk: None, lambda: {}
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)

            assert chunks == [{"token": "wrapped:test-model"}, {"token": "wrapped:done"}]
        finally:
            intercepts.deregister_llm_stream_execution("py_llm_stream_next")

    async def test_stream_execution_intercept_async_function_is_supported(self):
        def middleware(request, next):
            updated = LLMRequest(request.headers, {**request.content, "prefix": "async"})

            async def gen():
                upstream = await next(updated)
                async for chunk in upstream:
                    yield {"token": f"{updated.content['prefix']}:{chunk['token']}"}

            return gen()

        def stream_func(request):
            async def gen():
                yield {"token": request.content["model"]}
                yield {"token": "done"}

            return gen()

        intercepts.register_llm_stream_execution("py_llm_stream_async", 1, middleware)
        try:
            stream = await llm.stream_execute(
                "stream_async_llm", make_request(), stream_func, lambda chunk: None, lambda: {}
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)

            assert chunks == [{"token": "async:test-model"}, {"token": "async:done"}]
        finally:
            intercepts.deregister_llm_stream_execution("py_llm_stream_async")


class TestLLMStreaming:
    async def test_stream_execute(self):
        # Stream functions now take LLMRequest and return async iterator of Json
        def stream_func(request):
            async def gen():
                yield {"token": "hello"}
                yield {"token": "world"}

            return gen()

        collected = []

        def collector(chunk):
            collected.append(chunk)

        def finalizer():
            return {"chunks": collected}

        request = make_request()
        stream = await llm.stream_execute("stream_llm", request, stream_func, collector, finalizer)
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)

        assert len(chunks) >= 2
        # Collector should have received all chunks
        assert len(collected) == len(chunks)

    async def test_stream_execute_propagates_generator_error(self):
        def stream_func(request):
            async def gen():
                yield {"token": "hello"}
                raise RuntimeError("stream boom")

            return gen()

        stream = await llm.stream_execute(
            "stream_error_llm", make_request(), stream_func, lambda chunk: None, lambda: {}
        )
        assert await anext(stream) == {"token": "hello"}
        with pytest.raises(RuntimeError, match="stream boom"):
            await anext(stream)

    async def test_stream_execute_rejects_invalid_iterator(self):
        stream = await llm.stream_execute(
            "stream_invalid_iter_llm", make_request(), lambda request: object(), lambda chunk: None, lambda: {}
        )
        with pytest.raises(RuntimeError, match="__anext__"):
            await anext(stream)

    async def test_stream_execute_handles_iterator_that_stops_in___anext__(self):
        stream = await llm.stream_execute(
            "stream_direct_stop_llm",
            make_request(),
            lambda request: _ImmediateStopAsyncIter(),
            lambda chunk: None,
            lambda: {},
        )
        chunks = []
        async for chunk in stream:
            chunks.append(chunk)
        assert chunks == []

    async def test_stream_execute_propagates_direct___anext__error(self):
        stream = await llm.stream_execute(
            "stream_direct_error_llm",
            make_request(),
            lambda request: _BrokenAsyncIter(),
            lambda chunk: None,
            lambda: {},
        )
        with pytest.raises(RuntimeError, match="direct __anext__ boom"):
            await anext(stream)

    async def test_stream_execution_intercept_rejects_invalid_iterator(self):
        intercepts.register_llm_stream_execution(
            "py_llm_stream_bad_iter",
            1,
            cast(intercepts.LlmStreamExecutionIntercept, lambda request, next: object()),
        )
        try:
            stream = await llm.stream_execute(
                "stream_intercept_invalid_iter_llm",
                make_request(),
                lambda request: _single_chunk_stream(),
                lambda chunk: None,
                lambda: {},
            )
            with pytest.raises(RuntimeError, match="__anext__"):
                await anext(stream)
        finally:
            intercepts.deregister_llm_stream_execution("py_llm_stream_bad_iter")

    async def test_stream_execution_intercept_handles_iterator_that_stops_in___anext__(self):
        intercepts.register_llm_stream_execution(
            "py_llm_stream_direct_stop",
            1,
            lambda request, next: _ImmediateStopAsyncIter(),
        )
        try:
            stream = await llm.stream_execute(
                "stream_intercept_direct_stop_llm",
                make_request(),
                lambda request: _single_chunk_stream(),
                lambda chunk: None,
                lambda: {},
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            assert chunks == []
        finally:
            intercepts.deregister_llm_stream_execution("py_llm_stream_direct_stop")

    async def test_stream_execution_intercept_propagates_direct___anext__error(self):
        intercepts.register_llm_stream_execution(
            "py_llm_stream_direct_error",
            1,
            lambda request, next: _BrokenAsyncIter(),
        )
        try:
            stream = await llm.stream_execute(
                "stream_intercept_direct_error_llm",
                make_request(),
                lambda request: _single_chunk_stream(),
                lambda chunk: None,
                lambda: {},
            )
            with pytest.raises(RuntimeError, match="direct __anext__ boom"):
                await anext(stream)
        finally:
            intercepts.deregister_llm_stream_execution("py_llm_stream_direct_error")

    async def test_stream_execute_collector_failure_raises(self):
        def stream_func(request):
            async def gen():
                yield {"token": "hello"}

            return gen()

        stream = await llm.stream_execute(
            "stream_collector_fail_llm",
            make_request(),
            stream_func,
            lambda chunk: raise_runtime_error("collector boom"),
            lambda: {},
        )
        with pytest.raises(RuntimeError, match="collector boom"):
            await anext(stream)

    async def test_stream_execute_finalizer_failure_records_null_output(self):
        events = []
        subscribers.register("py_llm_finalizer_fail_sub", lambda event: events.append(event))

        def stream_func(request):
            async def gen():
                yield {"token": "hello"}

            return gen()

        try:
            stream = await llm.stream_execute(
                "stream_finalizer_fail_llm",
                make_request(),
                stream_func,
                lambda chunk: None,
                lambda: object(),
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            assert chunks == [{"token": "hello"}]
        finally:
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_llm_finalizer_fail_sub")

        end = _llm_event(events, "stream_finalizer_fail_llm", "end")
        assert end.data is None

    async def test_stream_execute_finalizer_callable_error_records_null_output(self):
        events = []
        subscribers.register("py_llm_finalizer_callable_fail_sub", lambda event: events.append(event))

        def stream_func(request):
            async def gen():
                yield {"token": "hello"}

            return gen()

        try:
            stream = await llm.stream_execute(
                "stream_finalizer_callable_fail_llm",
                make_request(),
                stream_func,
                lambda chunk: None,
                lambda: raise_runtime_error("finalizer boom"),
            )
            chunks = []
            async for chunk in stream:
                chunks.append(chunk)
            assert chunks == [{"token": "hello"}]
        finally:
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_llm_finalizer_callable_fail_sub")

        end = _llm_event(events, "stream_finalizer_callable_fail_llm", "end")
        assert end.data is None

    async def test_subscriber_exception_does_not_break_streaming(self):
        seen = []
        subscribers.register("py_llm_bad_sub", lambda event: raise_runtime_error("subscriber boom"))
        subscribers.register("py_llm_good_sub", lambda event: seen.append(event.kind))
        try:
            handle = llm.call("llm_subscriber_error", make_request())
            llm.call_end(handle, {"ok": True})
        finally:
            try:
                subscribers.flush()
            finally:
                subscribers.deregister("py_llm_bad_sub")
                subscribers.deregister("py_llm_good_sub")

        assert seen == ["scope", "scope"]


async def _single_chunk_stream():
    yield {"token": "downstream"}


class _ImmediateStopAsyncIter:
    def __aiter__(self):
        return self

    def __anext__(self):
        raise StopAsyncIteration


class _BrokenAsyncIter:
    def __aiter__(self):
        return self

    def __anext__(self):
        raise RuntimeError("direct __anext__ boom")
