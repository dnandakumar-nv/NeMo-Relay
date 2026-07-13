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

`nemo-relay-cli` installs the NeMo Relay CLI, the `nemo-relay` binary for local
coding-agent observability. It can configure supported coding-agent hooks, run
agents through an ephemeral gateway, and diagnose local agent and exporter
readiness.

The CLI is a Rust package in this repository, but most users should interact
with the installed `nemo-relay` command rather than link against the crate.

## Why Use It?

- **Observe existing coding agents**: Run Claude Code, Codex, or Hermes
  Agent through a local NeMo Relay gateway without changing the agent
  itself.
- **Configure hooks interactively**: Use the setup wizard to write project or
  user config and install the hook files needed by supported agents.
- **Export local sessions**: Write ATIF trajectory files, ATOF event JSONL
  streams, or OpenInference spans from one shared config model.
- **Diagnose setup readiness**: Check config layers, `plugins.toml` discovery,
  agent binaries, persistent host-plugin installs, hook status, observability
  outputs, and shell completions with `nemo-relay doctor`.

## What You Get

- **`nemo-relay` binary**: The executable installed by the `nemo-relay-cli`
  Cargo package.
- **First-run setup**: Bare `nemo-relay` launches setup when no config exists,
  then runs doctor once config is present.
- **Agent shortcuts**: `nemo-relay claude`, `nemo-relay codex`, and
  `nemo-relay hermes` start observed agent runs.
- **Config-driven launch**: `nemo-relay run` resolves config, environment, and
  CLI overrides for deterministic non-interactive use.
- **Hook forwarding server**: A local gateway accepts agent hook events and
  provider-shaped OpenAI or Anthropic requests.
- **Bundled Router host**: The CLI registers Router before plugin validation.
  Buffered Responses, Chat Completions, and Anthropic Messages calls use the V2
  managed path with frozen host transport policy and independent delayed
  replay. Streaming, model-list, and token-count routes remain
  replay-ineligible. A buffered non-success is a managed-call error while the
  client still receives its exact status, allowed headers, and bounded body.
  Graceful gateway shutdown drains plugin resources under a bounded deadline.

## Installation Options

Cargo:

```bash
cargo install nemo-relay-cli
```

Unix curl:

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.sh | sh
```

Windows PowerShell:

```powershell
irm https://raw.githubusercontent.com/NVIDIA/NeMo-Relay/main/install.ps1 | iex
```

For version pinning, custom installation directories, verification,
troubleshooting, and CLI usage, refer to the
[NeMo Relay installation guide](https://docs.nvidia.com/nemo/relay/getting-started/installation).

After installation, verify the binary with:

```bash
nemo-relay --version
```

## Getting Started

Run the first-time setup wizard:

```bash
nemo-relay
```

After setup, inspect local readiness:

```bash
nemo-relay doctor
```

Run a supported agent through the gateway:

```bash
nemo-relay codex
nemo-relay claude -- "summarize this repository"
```

Use `run --dry-run` to inspect resolved config without spawning the agent:

```bash
nemo-relay run --agent codex --dry-run
```

## Configuration

Project config lives at `./.nemo-relay/config.toml`; user config lives at
`~/.config/nemo-relay/config.toml` or `$XDG_CONFIG_HOME/nemo-relay/config.toml`.
The project layer overrides system config, and the user layer overrides the
project layer.

General options are configured through the top-level config. Edit the config with:

```bash
nemo-relay config
```

Observability exporters are configured through the plugin config. Edit the user
plugin config with:

```bash
nemo-relay plugins edit
```

The top-level editor menu contains one entry per supported built-in, including
Router, followed by the dynamic plugin references in the selected physical
`plugins.toml`. Router fields use the crate's typed editor schema and preserve
unknown configuration during edits. Dynamic plugins with a manifest-declared
JSON Schema provide structured field controls. Other dynamic plugins use a raw
JSON object editor.

`nemo-relay doctor` reports Router registration, the bundled SQLite and
sqlite-vec versions, a temporary `vec0` insert/query probe, V2 route support,
configured mode and queue capacity, database directory and schema readiness,
embedding egress, and Active control/outcome-policy readiness. Doctor reads the
same discovered `plugins.toml` configuration as gateway startup and does not
call a model provider or migrate the Router ledger.

## Router Inspection and Control

An enabled Router component can be inspected without starting the gateway:

```bash
nemo-relay router status --json
nemo-relay router pools --limit 100 --json
nemo-relay router evidence list --pool support --json
nemo-relay router decisions tail --follow --json
```

Launch the embedded read-only dashboard on an available loopback port:

```bash
nemo-relay router dashboard
```

For a headless session, use `--no-open`. Add `--token-file PATH` to place the
one-time launch URL in a new owner-only file instead of printing it. Remote
binding requires a concrete address, `--allow-remote`, a token file, and both
`--tls-cert` and `--tls-key`. The dashboard uses redacted inspection content by
default; `--full-content` enables bounded secret-filtered content for the
process session.

Export verified redacted evidence to standard output or a new file:

```bash
nemo-relay router evidence export - --format jsonl
nemo-relay router evidence export evidence.csv --format csv
```

Operator commands require a bounded reason. Reset and cohort rotation also
require the exact configured project ID:

```bash
nemo-relay router pause --reason "maintenance"
nemo-relay router force-anchor set --pool support --reason "provider incident"
nemo-relay router reset --pool support --confirm my-project --reason "new corpus"
nemo-relay router cohort rotate --confirm my-project --reason "new cohort"
```

The CLI derives the audit actor from the OS principal unless `--actor` is
provided. It submits one generation compare-and-swap and does not retry a
conflict. Refer to [Use Router CLI and
HTTP](https://docs.nvidia.com/nemo/relay/router/cli-and-http) for the complete
tree, filters, JSON schemas, file behavior, and exit codes. Refer to [Use the
Local Router Dashboard](https://docs.nvidia.com/nemo/relay/router/dashboard)
for session lifetime, views, remote TLS setup, and troubleshooting.

The canonical plugin file is `plugins.toml`; user config lives at
`~/.config/nemo-relay/plugins.toml` or
`$XDG_CONFIG_HOME/nemo-relay/plugins.toml`. Project config lives at
`.nemo-relay/plugins.toml`.

Minimal ATIF example:

```toml
version = 1

[[components]]
kind = "observability"
enabled = true

[components.config.atif]
enabled = true
output_directory = "./atif"
```

## Documentation

NeMo Relay Documentation: https://docs.nvidia.com/nemo/relay
