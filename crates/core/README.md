<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![License](https://img.shields.io/github/license/NVIDIA/NeMo-Relay)](https://github.com/NVIDIA/NeMo-Relay/blob/main/LICENSE)
[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/NVIDIA/NeMo-Relay/)
[![Release](https://img.shields.io/github/v/release/NVIDIA/NeMo-Relay?color=green)](https://github.com/NVIDIA/NeMo-Relay/releases)
[![Codecov](https://codecov.io/gh/NVIDIA/NeMo-Relay/branch/main/graph/badge.svg)](https://app.codecov.io/gh/NVIDIA/NeMo-Relay)
[![PyPI](https://img.shields.io/pypi/v/nemo-relay?color=4B8BBE&logo=pypi)](https://pypi.org/project/nemo-relay/)
[![npm node](https://img.shields.io/npm/v/nemo-relay-node?label=nemo-relay-node&color=CC3534&logo=npm)](https://www.npmjs.com/package/nemo-relay-node)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay?label=nemo-relay&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-adaptive?label=nemo-relay-adaptive&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-adaptive)
[![Crates.io](https://img.shields.io/crates/v/nemo-relay-cli?label=nemo-relay-cli&color=B7410E&logo=rust)](https://crates.io/crates/nemo-relay-cli)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/NVIDIA/NeMo-Relay)

# NeMo Relay

`nemo-relay` is the core Rust SDK for NeMo Relay, a portable execution
runtime for agent systems. Use it when a Rust application, framework adapter,
or service needs one consistent way to scope, control, and observe tool and LLM
calls.

Rust is the source of truth for NeMo Relay runtime behavior. The Python and
Node.js bindings mirror the semantics exposed by this crate.

## Why Use It?

- **Own Rust execution context**: Hierarchical scopes preserve parent-child
  relationships across tools, LLM calls, middleware, subscribers, and events.
- **Put policy around real calls**: Guardrails and intercepts can block work,
  sanitize observability payloads, rewrite requests, or wrap execution.
- **Emit one lifecycle stream**: Subscribers can consume canonical runtime
  events in-process or export them to Agent Trajectory Interchange Format
  (ATIF), OpenTelemetry, and OpenInference.
- **Integrate without changing orchestration**: Wrap framework and provider
  callbacks while leaving scheduling, retries, memory, and result handling in
  the owning application.

## What You Get

- **Managed tool and LLM execution**: Run call boundaries through consistent
  lifecycle helpers and middleware ordering.
- **Scope-local runtime behavior**: Attach middleware and subscribers to the
  scope that owns them and clean them up when that scope closes.
- **Plugin primitives**: Register reusable runtime behavior configured from
  one shared plugin system.
- **Built-in observability plugin**: Configure first-party Agent Trajectory
  Observability Format (ATOF), Agent Trajectory Interchange Format (ATIF),
  OpenTelemetry, and OpenInference exporters from the core crate.
- **Codec and typed helpers**: Normalize provider requests and responses for
  framework integrations.
- **Binding source of truth**: Use the runtime semantics mirrored by the
  Python and Node.js bindings.

## Provider-Neutral Request Annotations

The built-in request codecs expose `Message::Developer` without relabeling it
as a system or user message. They also normalize structured output into
`AnnotatedLlmRequest::response_format` with `json_object` and `json_schema`
kinds. Unknown fields remain lossless in `native_wrapper` and `native_format`
objects on the normalized format.

The native mappings are:

| Provider surface | Developer role | Structured-output location |
|---|---|---|
| OpenAI Chat Completions | Native `messages[].role = "developer"` | `response_format` |
| OpenAI Responses | Native developer input items; top-level `instructions` remains the leading System fact | `text.format` |
| Anthropic Messages | Unsupported and rejected rather than relabeled | `output_config.format` (`json_schema` only) |

Supported native formats are removed from the request's generic `extra` map
while typed. Unknown format kinds remain generic and round-trip unchanged.
Malformed supported formats and typed/generic conflicts return codec errors.
Encoding merges over the original provider request and preserves unrelated
native fields; it never patches provider JSON by guessing a model path.

## V2 LLM Context and Call Roles

The core crate re-exports `LlmApiFamily`, `LlmCallRole`,
`LlmTrajectoryScopeSnapshot`, and `LlmExecutionContextSnapshot` from
`nemo-relay-types`. The additive V2 context freezes one coherent scope-stack
view for a physical managed LLM call. Its call UUID remains distinct even when
concurrent calls have identical requests.

Trajectory ownership belongs to the deepest active explicit Agent scope at or
above the call's immediate parent, with the implicit root as the fallback. The
snapshot owns the owner-to-parent path and caller-supplied sanitized metadata;
it does not retain provider payloads, credentials, callbacks, continuations, or
runtime handles.

Present tenant and agent identities are validated before capture. Core rejects
empty, oversized, non-NFC, control-bearing, or credential-shaped values,
including authorization assignments, credential-bearing URLs, private-key
headers, JWTs, AWS access key IDs, and common provider token prefixes.

Core-created LLM start and end events carry the same call UUID and
`category_profile.call_role`. Existing managed-call entry points remain
Primary by default. Shadow and Judge roles describe execution purpose; they do
not authorize replay. Embedder scopes are not LLM calls and carry neither an
LLM API family nor an LLM call role.

The V2 contracts do not change these existing boundaries:

| Boundary | Compatibility Behavior |
|---|---|
| Rust V1 `LlmExecutionFn` and `LlmExecutionNextFn` | Signatures are unchanged and receive no V2 snapshot. A continuation remains the next execution-chain call, not a delayed replay handle. |
| Raw C FFI | No function parameter or ABI structure gains a V2 context field. |
| Native plugin ABI v1 | The ABI version and host/plugin tables remain unchanged. |
| Worker `grpc-v1` | The protobuf and worker protocol gain no V2 context field. |

V1 calls remain valid and observable as Primary calls. Without an explicit V2
family and snapshot, they are not V2 replay authority. Consumers of raw event
JSON can observe the additive ATOF role field without a change to these typed
interfaces.

### V2 Managed Calls and Replay

Use `llm_call_execute_v2` only when the host must give a context-aware
execution intercept a frozen call context or a repeatable replay transport.
`LlmCallExecuteV2Params` requires an explicit `LlmApiFamily`, `LlmCallRole`,
and owned `sanitized_metadata`. Core does not infer these values from request
JSON.

V1 and V2 non-streaming execution intercepts share one registry, namespace,
priority order, and scope-visibility model. A V1 managed call invokes only V1
entries. A V2 managed call invokes V2 entries and adapts visible V1 entries in
the same deterministic order.

The execution continuation and replay transport have different authority and
lifetimes:

| Value | Lifetime and Use |
|---|---|
| `LlmExecutionNextFn` | Represents the remaining chain for the current physical call. A V2 intercept can invoke it at most once while that intercept future is active. Do not retain it for delayed work. |
| `Arc<LlmExecutionContextSnapshot>` | Immutable, owned call context. An intercept can retain it after the anchor chain returns. |
| `Arc<dyn LlmReplayTransport>` | Optional host-owned transport for delayed, repeated non-streaming calls. Each `start` creates an independent `LlmReplayCall`. |
| `LlmReplayCall` | Future for one replay invocation. Dropping an incomplete call invokes the shared cancellation hook if it is still armed. The hook runs at most once and does not cancel sibling calls. |
| `LlmReplayCancellationHandle` | Cloneable exact-once cancellation authority returned by `LlmReplayCall::cancellation_handle()`. The first call to `cancel()` that claims the hook returns `true`; later calls return `false`. Completing the call disarms all handle clones. |

`LlmReplayTransport::start` must return promptly without waiting for host I/O,
an interpreter lock, or an event loop. It must dispatch potentially blocking
work asynchronously through the returned call. The call's cancellation hook
has the same nonblocking contract: signal or dispatch cancellation promptly
without waiting for host I/O, an interpreter lock, or task completion.

For an otherwise eligible call, Core calls a supplied `LlmReplayFactory` once
after freezing context. Replay is eligible only for a `Primary` call without
`STREAMING` or `STATEFUL` when the factory returns contract version `1`, the
exact call API family, and a valid non-secret `transport_identity`. A factory
or capability failure does not fail the anchor call. The V2 intercept receives
`None` for the transport and Core emits a `nemo_relay.replay_ineligible` mark
with the call UUID and one stable
reason code: `internal_role`, `streaming`, `stateful`, `factory_error`,
`factory_panic`, `capability_panic`, `unsupported_contract`, `family_mismatch`,
or `invalid_transport_identity`. The mark never includes the factory error text.
Omitting a replay factory simply makes replay unavailable and does not emit an
ineligibility mark.

Keep endpoint, authentication, TLS, proxy, and transport policy inside the
opaque factory and transport implementation. Do not put those values in
`transport_identity`, sanitized metadata, or diagnostics. The replay contract
exposes only immutable capability fields and `start(LlmRequest)`; it has no
endpoint or authentication mutator and no access to anchor response status,
headers, streaming writers, or other client-side channels.

`Shadow` and `Judge` calls are independent physical managed calls, not reuse of an
anchor continuation. Core requires their immediate parent to be an active
`Evaluator` scope and requires sanitized metadata to contain a valid string
`anchor_uuid`. Internal calls use no replay factory, which prevents recursive
replay authority. Their provider callback starts one retained transport call
inside the managed call so middleware and ATOF events include the replay
latency and error.

Run the in-memory example from the repository root:

```bash
cargo run -p nemo-relay --example v2_replay
```

The example registers a V2 intercept, serves the anchor through `next`, and
then uses only the retained replay transport for work that completes after the
managed anchor call returns.

### Router Boundary

The published `nemo-relay-router` crate consumes this V2 context and replay
contract. Router is not a Core built-in: a custom Rust host must link that crate
and explicitly register its plugin component. The CLI, Python, and Node.js host
packages do so automatically and expose configuration through their existing
package surfaces. Core still has no Router dependency. The generic C FFI and Go
binding can recognize Router configuration but expose no V2 replay factory,
typed Active controls, or inspection APIs.

## Plugin Shutdown Ownership

Registrations created with `PluginRegistration::new` have no stop, drain, or
abort hooks and retain their existing deregistration callback. Deregistration
failures are reported and retained for retry before the next activation. A Rust
component that owns background tasks, queues, replay transports, or durable work
can add one resource registration with `PluginRegistration::with_shutdown`. Its
hooks have distinct responsibilities:

- `stop_intake` is synchronous and nonblocking. It closes every scheduling
  path before queued subscriber callbacks are flushed.
- `drain(deadline)` is asynchronous. It closes queues, resolves accepted work,
  and releases drained resources by the supplied absolute deadline.
- `abort` is synchronous and nonblocking. It cancels tasks and transports and
  releases resources that cannot drain.

Make all three hooks idempotent. `stop_intake` and `abort` must establish their
closed or inert state before returning, even when they report a secondary
cleanup error. The deregistration closure has the same inert-before-error
contract. Every scheduling site must check the intake state before it creates
background work, and replay providers must check again immediately before
`LlmReplayTransport::start`.

`clear_plugin_configuration()` provides immediate teardown: stop intake, flush
subscribers, abort in reverse registration order, and deregister in reverse
order. `clear_plugin_configuration_async(timeout)` stops intake, flushes
subscribers, and drains in reverse order under one deadline. A failed or timed
out drain causes Core to abort unfinished registrations before reverse
deregistration. Both APIs leave the active configuration empty even when they
return an aggregate teardown error. Core retains a failed deregistration
callback for retry, blocks later activation until retained cleanup succeeds,
and does not re-enter that callback during a later clear. Clear still removes
the current active configuration while retained cleanup waits for activation.

Async configuration replacement uses the same drain sequence with a 30-second
default. Use `initialize_plugins_with_options` and
`PluginInitializationOptions` to override that deadline. If old teardown
fails, Core leaves the configuration empty and does not activate the
replacement. If teardown succeeds but new activation fails, Core attempts to
restore the previous config.

Use synchronous clear only when immediate cancellation is acceptable. A
durable component must reconcile interrupted accepted work on its next
activation. Do not clear or replace plugin configuration from a subscriber
callback; Core returns `PluginError::Conflict` because the dispatcher cannot
wait for its own flush barrier.

## Installation

Install the published crate in a Rust application:

```bash
cargo add nemo-relay serde_json
```

To add adaptive runtime behavior, install the companion crate too:

```bash
cargo add nemo-relay-adaptive
```

When consuming a local checkout, use path dependencies:

```toml
[dependencies]
nemo-relay = { path = "../NeMo-Relay/crates/core" }
nemo-relay-adaptive = { path = "../NeMo-Relay/crates/adaptive" }
serde_json = "1"
```

## Getting Started

The smallest useful workflow is to create a scope, emit a mark event, and close
the scope:

```rust
use nemo_relay::api::scope::{
    self, EmitMarkEventParams, PopScopeParams, PushScopeParams, ScopeAttributes, ScopeType,
};
use serde_json::json;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let handle = scope::push_scope(
        PushScopeParams::builder()
            .name("demo-agent")
            .scope_type(ScopeType::Agent)
            .attributes(ScopeAttributes::empty())
            .data(json!({"binding": "rust"}))
            .build(),
    )?;

    scope::event(
        EmitMarkEventParams::builder()
            .name("initialized")
            .parent(&handle)
            .data(json!({"ok": true}))
            .build(),
    )?;

    scope::pop_scope(PopScopeParams::builder().handle_uuid(&handle.uuid).build())?;
    Ok(())
}
```

## Documentation

NeMo Relay Documentation: https://docs.nvidia.com/nemo/relay
