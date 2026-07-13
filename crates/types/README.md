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

# NeMo Relay Types

`nemo-relay-types` provides the shared serializable data model for NeMo Relay.
Use it when a Rust integration, plugin SDK, or protocol implementation needs
the same event, scope, tool, LLM, codec, and plugin-diagnostic types as Relay.

This crate intentionally contains no runtime registries, dynamic loading,
exporters, or process-global state. Applications normally depend on
`nemo-relay`; this crate is the lower-level contract shared by the runtime and
authoring SDKs.

## Why Use It?

- **Keep wire shapes consistent**: Exchange Relay DTOs without duplicating
  serializable event, request, response, and diagnostic models.
- **Share one JSON representation**: Use `Json`, an alias for
  `serde_json::Value`, across Relay-facing payloads.
- **Build SDKs without the runtime**: Depend on data contracts without pulling
  in runtime behavior or global state.

## What You Get

- **`api` module**: Event, scope, tool, and LLM DTOs and attributes.
- **`codec` module**: Normalized LLM request and response annotations.
- **`plugin` module**: Structured plugin configuration diagnostics.
- **Optional `schema` feature**: `schemars` implementations for supported
  serializable types.

## Normalized LLM Request Contracts

`codec::request::Message` keeps `system` and `developer` as distinct
instruction roles. `AnnotatedLlmRequest::response_format` represents the two
provider-neutral structured-output modes: `json_object` and `json_schema`.
Schema formats can carry a name, the complete JSON Schema, and an optional
strictness flag.

`StructuredResponseFormat::extra` reserves two object-valued entries for a
lossless provider round trip:

- `native_wrapper` contains unmodeled fields from the native object that owns
  the format descriptor.
- `native_format` contains unmodeled fields from the native format descriptor.

An absent typed format preserves the legacy serialized request shape. A native
`response_format` value that does not use a normalized `kind` remains in the
flattened request `extra` map. Serializing both a typed format and a generic
flattened `response_format` is rejected as ambiguous.

## LLM Execution Context Contracts

The `api::llm` module defines the router-neutral V2 context contracts:

- `LlmApiFamily` identifies OpenAI Chat Completions, OpenAI Responses, or
  Anthropic Messages with the stable wire values `openai_chat_completions`,
  `openai_responses`, and `anthropic_messages`.
- `LlmCallRole` identifies `primary`, `shadow`, or `judge` execution.
- `LlmTrajectoryScopeSnapshot` freezes one scope's UUID, name, and type.
- `LlmExecutionContextSnapshot` combines physical call identity, root and
  parent identity, trajectory ownership, provider family, call role,
  attributes, optional routing identities, and sanitized metadata.

Present tenant and agent identities must be Unicode NFC, nonempty, no more
than 256 UTF-8 bytes, free of control characters and credential syntax, and
are preserved and compared case-sensitively. Credential syntax includes
authorization labels and assignments, user information in URLs, private-key
headers, JWTs, AWS access key IDs, and common provider token prefixes such as
`nvapi-`, `sk-`, `ghp_`, `github_pat_`, `hf_`, and `xoxb-` when the value has a
credential-like length.

The V2 ownership contract uses the deepest active explicit Agent scope at or
above the call's parent as `trajectory_owner_uuid`. It falls back to the
implicit root and records the owner-to-parent path in ancestry order. The snapshot
owns its data and remains stable after scopes close or asynchronous work yields.
It never contains a request, response, header, endpoint, credential, callback,
continuation, or runtime handle.

Embedding work is not an LLM call. It uses `ScopeType::Embedder` and does not
fabricate an `LlmApiFamily` or `LlmCallRole`.

These DTOs are additive. Existing Rust V1 execution callbacks do not receive a
snapshot, and the raw C FFI, native plugin ABI v1, and worker `grpc-v1` protocol
gain no typed context fields.

## Installation

Add the crate when implementing a Relay-adjacent SDK, protocol, or integration:

```bash
cargo add nemo-relay-types
```

Enable JSON Schema support when needed:

```bash
cargo add nemo-relay-types --features schema
```

## Getting Started

Use the shared `Json` type for a Relay-compatible payload:

```rust
use nemo_relay_types::Json;
use serde_json::json;

let payload: Json = json!({"source": "my-integration"});
assert_eq!(payload["source"], "my-integration");
```

## Documentation

- [NeMo Relay documentation](https://docs.nvidia.com/nemo/relay)
- [NeMo Relay Rust crate](https://github.com/NVIDIA/NeMo-Relay/blob/main/crates/core/README.md)
