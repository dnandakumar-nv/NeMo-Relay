---
name: nemo-relay-debug-runtime-integration
description: Debug application-side NeMo Relay integration issues such as load failures, inactive scopes, missing events, or adaptive/plugin wiring problems
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---


# Debug Runtime Integration

Use this skill when NeMo Relay is present in the application but something is not
working.

## First Checks

- Can the binding or native artifact load?
- Is there an active scope when the failing call runs?
- Is the work happening on the expected scope stack?
- Is the subscriber/exporter/plugin configuration actually active?
- Did the app choose the right public API layer: managed execute vs manual
  lifecycle vs typed wrappers vs adaptive/plugins?

## Common Failure Classes

- Python native extension missing
- Go dynamic library not on the loader path
- Node native addon not built or not loading
- Execute call outside a scope
- Missing events because registration never happened
- Concurrency causing the wrong scope stack to be active
- Adaptive component never initialized or config validation ignored

## Embedded Troubleshooting Matrix

- **Rust build failure**: run the narrowest core build first, then expand to the
  affected binding or workspace command.
- **Python import failure**: rebuild the virtual environment and native
  extension with `uv sync`, then run a small Python test or import check from
  the same environment as the application.
- **Node.js addon failure**: reinstall and rebuild the native addon from the
  Node binding package before debugging application code.
- **Go loader failure**: build the release FFI shared library and point both the
  linker and runtime loader at the release output directory; use the macOS
  dynamic-library path variable (`DYLD_LIBRARY_PATH`) on macOS.
- **Scope stack empty**: the work is outside an active scope or crossed a thread,
  task, goroutine, or worker boundary without the intended stack.
- **Work leaks across requests**: separate requests are sharing one scope stack;
  create a fresh stack per independent request or agent.
- **Middleware missing or ordered incorrectly**: check global vs scope-local
  registration, active scope ancestry, names, and priority values.
- **Subscriber missing events**: register before the events are emitted; for
  scope-local subscribers, ensure the current scope is the owner or descendant.
- **Event fields missing**: managed helpers populate semantic fields; manual
  lifecycle calls require explicit params for input, output, model names, and
  tool call IDs.
- **ATIF empty or mixed**: register before work starts, use one exporter per run
  or clear between runs, and separate concurrent agents by root scope.
- **Provider payload conversion failure**: convert non-JSON provider objects,
  SDK handles, callbacks, streams, or class instances with explicit codecs.
- **Plugin validation failure**: validate config independently from runtime
  registration and check required fields, value types, defaults, and config
  source.
- **Replay transport missing in a V2 intercept**: inspect the
  `nemo_relay.replay_ineligible` mark reason. If there is no mark, confirm that
  the call supplied a replay factory. Check the `Primary` role, ensure the call
  is not marked `STREAMING` or `STATEFUL`, and check contract version `1`, the
  exact API family, and the non-secret transport identity. Factory failures
  intentionally do not expose their error text or fail the anchor call.
- **Shadow or Judge context rejected**: create the internal managed call under
  an immediate `Evaluator` parent and put a valid string `anchor_uuid` in
  sanitized metadata. Do not attach another replay factory.
- **Plugin clear returns a conflict**: do not clear or replace configuration
  from a subscriber callback. If a synchronous clear races another transition,
  retry from the host lifecycle path after that transition completes.
- **Background work starts after clear**: verify that the component closes
  intake before queue submission and checks again immediately before each
  internal managed call or replay start.
- **Router CLI cannot open inspection**: run `nemo-relay router status --json`
  first. Confirm the merged plugin config contains exactly one enabled Router,
  then inspect the reported database state. Do not migrate or repair the
  private ledger with direct SQL.
- **Router mutation returns conflict or busy**: inspect current status and the
  durable operator history before issuing another mutation. The CLI does not
  retry generation conflicts, and only one standalone mutation process is
  admitted per database.
- **Router neighborhood request cannot call an embedder**: the CLI is
  provider-free. Use a Rust inspection service whose host explicitly enabled
  request embedding and whose Router config permits the endpoint, or use an
  evidence/query-hash lookup.
- **Adaptive behavior unchanged**: confirm instrumentation emits events, the
  adaptive component is enabled, policy allows the behavior, and the call path
  reaches the configured component.
- **OpenTelemetry or OpenInference export failure**: confirm `http_binary` vs
  `grpc`, endpoint, headers, target support, and whether a native gRPC exporter
  has an active Tokio runtime.
- **Callback succeeded but no lifecycle events appear**: confirm the integration
  uses managed execute helpers or balanced manual start/end APIs, not only the
  underlying business callback.

## Related Skills

- `nemo-relay-start`
- `nemo-relay-use-context-isolation`
- `nemo-relay-tune-adaptive-config`
- `nemo-relay-build-plugin`
- `nemo-relay-operate-router`
