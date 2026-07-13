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

`nemo-relay-node` is the NeMo Relay package for Node.js applications. It gives
JavaScript and TypeScript code access to the same execution scopes, middleware,
plugins, lifecycle events, and observability model used by the Rust runtime.

The package is implemented as a napi-rs native extension, but Node.js users
should install it from npm rather than depend on the Rust crate directly.

## Why Use It?

- **Own execution context in Node.js**: Group agent, tool, and LLM work into
  one scope tree from JavaScript or TypeScript.
- **Put policy around callbacks**: Register guardrails and intercepts for
  request rewriting, blocking, sanitization, and execution wrapping.
- **Emit one lifecycle stream**: Send runtime events to in-process
  subscribers, Agent Trajectory Interchange Format (ATIF), OpenTelemetry, or
  OpenInference workflows.
- **Use package entry points by need**: Import the main runtime surface plus
  typed, plugin, adaptive, and observability helpers from npm.

## What You Get

- **npm package for Node.js**: A Node.js 24 or newer package backed by a
  napi-rs native extension.
- **Managed tool and LLM execution**: Helpers that emit lifecycle events and
  run middleware in a consistent order.
- **Middleware APIs**: Guardrails and intercepts for tool and LLM boundaries.
- **Observability exporters**: Subscriber and exporter support for common
  runtime telemetry flows.
- **Additional entry points**: `nemo-relay-node/typed`,
  `nemo-relay-node/plugin`, `nemo-relay-node/router`,
  `nemo-relay-node/adaptive`, and `nemo-relay-node/observability`.

## Installation

Install the npm package in a Node.js 24 or newer project:

```bash
npm install nemo-relay-node
```

## Getting Started

Register a subscriber and emit a mark inside a scope:

```js
const {
  ScopeType,
  deregisterSubscriber,
  event,
  flushSubscribers,
  registerSubscriber,
  withScope,
} = require("nemo-relay-node");

async function main() {
  registerSubscriber("printer", (runtimeEvent) => {
    console.log(`${runtimeEvent.kind} ${runtimeEvent.name}`);
    console.log(JSON.stringify(runtimeEvent));
  });

  await withScope("demo-agent", ScopeType.Agent, async (handle) => {
    event("initialized", handle, { binding: "node" }, null);
  });

  flushSubscribers();
  deregisterSubscriber("printer");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

The main runtime API is exported from `nemo-relay-node`. Additional entry points
are available at `nemo-relay-node/typed`, `nemo-relay-node/plugin`,
`nemo-relay-node/router`, `nemo-relay-node/adaptive`, and
`nemo-relay-node/observability`.

## Context-Aware LLM Calls

Use `llmCallExecuteV2` when middleware needs an explicit API family, call role,
and a frozen routing context. The optional replay factory is separate from the
provider callback. It returns a transport descriptor whose replay callable can
outlive the anchor call; it never receives the managed `next` continuation. The
binding freezes and retains the original descriptor until the replay transport
is released. Host-private state can use non-enumerable or symbol-keyed fields;
additional enumerable fields invalidate the descriptor.

```js
const { llmCallExecuteV2 } = require("nemo-relay-node");

const request = {
  headers: {},
  content: { model: "model-name", input: "hello" },
};
async function callProvider(request, { signal } = {}) {
  if (signal?.aborted) throw new Error("cancelled");
  return { model: request.content.model, output: "hello" };
}

const replayFactory = (context) => ({
  contractVersion: 1,
  apiFamily: context.apiFamily,
  transportIdentity: "prod-openai-us",
  replay: (request) => {
    const controller = new AbortController();
    return {
      result: callProvider(request, { signal: controller.signal }),
      cancel: () => controller.abort(),
    };
  },
});

async function main() {
  const response = await llmCallExecuteV2(
    "openai.responses",
    request,
    callProvider,
    "openai_responses",
    "primary",
    { region: "us" },
    null,
    null,
    null,
    null,
    "model-name",
    "tenant-a",
    "agent-a",
    replayFactory,
  );
  console.log(response);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

Family and role are closed string enums and are never inferred from the request.
The factory must return synchronously. Each replay invocation returns its own
`Promise` and idempotent cancellation callback, so dropping one pending replay
does not affect siblings. Factory or capability validation failures make replay
unavailable while leaving the anchor provider result unchanged. Streaming,
stateful, Shadow, and Judge calls are replay-ineligible.

Plugin configuration has two teardown modes. `clearPluginConfiguration()` and
`plugin.clear()` stop intake and abort immediately. Use
`clearPluginConfigurationAsync(timeoutMillis)` or
`plugin.clearAsync(timeoutMillis)` to flush subscriber callbacks and drain
component resources under one shared deadline before deregistration.

## Bundled Router

Loading `nemo-relay-node` registers the native Router component. Typed
configuration builders ship in the same npm package under
`nemo-relay-node/router`; there is no separate Router npm package.

```javascript
const plugin = require('nemo-relay-node/plugin');
const router = require('nemo-relay-node/router');

async function main() {
  const config = router.defaultConfig({ mode: 'off' });
  if (router.validateConfig(config).diagnostics.length !== 0) {
    throw new Error('invalid Router configuration');
  }
  await plugin.initialize({
    version: 1,
    components: [router.ComponentSpec(config)],
  });
  await plugin.clearAsync(30000);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
```

The builders apply Rust defaults, omit `undefined` values, and emit canonical
snake-case plugin fields. The native package includes the same V2 replay bridge
and pinned sqlite-vec capability as the CLI. Node.js can activate all Router
modes through the generic plugin lifecycle, but Active controls and decision
inspection are not exposed as Node.js APIs.

## Documentation

NeMo Relay Documentation: https://docs.nvidia.com/nemo/relay
