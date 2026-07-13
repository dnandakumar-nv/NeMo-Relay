---
name: nemo-relay-operate-router
description: Inspect, export, pause, force-anchor, reset, or rotate a configured NeMo Relay Router through the CLI, Rust inspection service, or optional authenticated Rust HTTP adapter
author: NVIDIA Corporation and Affiliates
license: Apache-2.0
---

# Operate NeMo Relay Router

Use this skill when a user needs to inspect a configured Router ledger, export
evidence, diagnose a recommendation, or apply an operator control.

## Choose a Surface

- Use `nemo-relay router` for local operation and scripts.
- Use `nemo_relay_router::inspection::InspectionService` in a Rust host.
- Use `inspection_http_router` with the Router crate's `http` feature when a
  Rust host must expose authenticated routes.
- Do not query SQLite tables directly.
- Python and Node.js Router modules configure and activate Router but do not
  expose inspection or operator methods. Raw FFI and Go remain V2
  replay-ineligible and expose no typed Router inspection/control surface.

## Inspect First

Start with one current snapshot:

```bash
nemo-relay router status --json
nemo-relay router pools --limit 100 --json
```

Use the exact read needed for the question:

- Evidence: `router evidence list`, `show`, or `export`
- Neighborhood: `router neighborhood inspect` by evidence ID, query hash plus
  partition file, or sanitized request file
- Decisions: `router decisions tail`, optionally with `--follow`

Pages use `--limit 1..500` and opaque `--cursor` values. Preserve the same
command and filters when resuming a cursor. The CLI request-file form is
provider-free and cannot spend an embedding call on a cache miss.

## Apply Controls

Every mutation needs a bounded nonblank `--reason`. The CLI uses the OS
principal unless `--actor` is supplied.

```bash
nemo-relay router pause --reason "maintenance"
nemo-relay router resume --reason "maintenance complete"
nemo-relay router force-anchor set --pool POOL --reason "provider incident"
nemo-relay router force-anchor clear --pool POOL --reason "provider recovered"
nemo-relay router reset --pool POOL --confirm PROJECT_ID --reason "new corpus"
nemo-relay router reset --all --confirm PROJECT_ID --reason "project reset"
nemo-relay router cohort rotate --confirm PROJECT_ID --reason "new cohort"
```

Pause and force-anchor are independent. Resume does not clear force-anchor.
Pool reset preserves other pools. All-pool reset is atomic. Cohort rotation
does not expose or accept cohort salt.

Each command reads one generation, creates one UUIDv7 mutation ID, submits one
compare-and-swap request, and never retries a conflict. If a command is
interrupted or reports contention, inspect status and use the Rust or HTTP
surface to inspect operator history before running another mutation. Only one
standalone CLI mutation process is admitted for a database at a time; another
process can return `busy`.

## Handle Output

- Add `--json` for a schema-version-1 success or failure envelope.
- `router decisions tail --follow --json` emits JSON Lines with a resume cursor.
- Evidence export supports `jsonl` and `csv`; `-` streams to standard output.
- A file export refuses an existing target unless `--force` is present.
- Exit `2` means a request/configuration refusal or conflict. Exit `1` means an
  operational, storage, migration, capacity, I/O, or integrity failure.

## HTTP Host Rules

Always construct the service with fixed content and operation capabilities and
always provide an `InspectionHttpAuthGuard`. `ReadOnly` mode contains no
mutation routes. `Operations` mode requires `allow_operations = true`. A
request can require the host policy through `x-nemo-relay-content-policy`, but
it cannot elevate from `redacted` to `full`.

## Related Skills

- `nemo-relay-debug-runtime-integration`
- `nemo-relay-tune-performance`
- `nemo-relay-setup-observability`
