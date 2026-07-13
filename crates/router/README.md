<!--
SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# NeMo Relay Router

`nemo-relay-router` is the published Rust shadow-evaluation, recommendation,
and bounded Active-canary component for NeMo Relay. It depends on
`nemo-relay`; the Core runtime does not depend on Router.

> **Important:** `shadow` and `recommend` always serve the anchor response.
> Recommend records a decision but performs zero replay execution and zero
> model rewrite. `active` can substitute only a durably authorized randomized
> canary root; control and holdout roots still serve the anchor. Judge labels
> are counterfactual proxy evidence and never replace randomized outcomes.

The `nemo-relay` CLI, `nemo-relay` Python wheel, and `nemo-relay-node` npm
package link and register Router before plugin validation and initialization.
Python and Node.js expose typed configuration helpers from their existing main
packages; there is no separate Router wheel or npm package. A custom Rust host
must still depend on this crate and register it explicitly.

The experimental C FFI and Go binding recognize Router through the generic
JSON plugin lifecycle, but they expose no Router-specific symbols, typed
helpers, or V2 replay bridge. They are not Router execution surfaces.

## Register Router in a Rust Host

Register Router before plugin validation or initialization. Registration is
idempotent for this crate's static implementation and rejects a different
plugin that already uses kind `router`.

```rust
use std::time::Duration;

use nemo_relay::plugin::{
    PluginConfig, clear_plugin_configuration_async, initialize_plugins,
};
use nemo_relay_router::{
    ComponentSpec, RouterConfig, RouterMode, deregister_router_component,
    register_router_component,
};

register_router_component()?;

let mut router = RouterConfig::default();
router.mode = RouterMode::Off;

let mut plugins = PluginConfig::default();
plugins.components.push(ComponentSpec::new(router).into());
initialize_plugins(plugins).await?;

clear_plugin_configuration_async(Duration::from_secs(30)).await?;
assert!(deregister_router_component());
```

Initialization requires an active Tokio runtime. Deregistering the plugin kind
does not tear down an active component. Clear or replace the Core plugin
configuration first.

`initialize_plugins` discovers `plugins.toml` files in system, project, and
user order, then applies code configuration. Components pair by `kind`.
Nested tables merge recursively; arrays and scalar values are replaced by the
higher-precedence layer. A custom host can register Router and pass
`PluginConfig::default()` to activate a Router component defined only in a
discovered file.

## Configure the Runtime

Configuration version `1` supports `off`, `shadow`, `recommend`, and `active`.

Root defaults are:

| Field | Default |
|---|---|
| `version` | `1` |
| `mode` | `off` |
| `project_id` | omitted |
| `database_path` | `.nemo-relay/router/router.db` |
| `retention_days` | `30` |
| `max_evidence_records` | `100000` |
| `allow_remote_embedding_egress` | `false` |
| `embedders` | `[]` |
| `pools` | `[]` |

Each configured pool requires a unique ID, one supported API family, anchor
models and a pinned revision, sampling and future-local limits, independent
Shadow and Judge concurrency, at least one candidate, and a complete
`JudgeConfig`. The Judge policy has no defaults and requires these fields:

- Contract `version`, `output_schema_version`, model, and pinned model
  revision.
- Exact `pairwise-equivalence-v1` prompt and
  `response-trajectory-equivalence-v1` rubric identifiers.
- Response and trajectory weights, component floors, confidence floor, and
  pass threshold.
- Rationale byte limit and base and maximum dependency cool-off durations.

Weights must be finite and nonnegative and sum to `1.0` within `1e-9`. Floors,
confidence, and threshold are finite values in `[0, 1]`.
`max_rationale_bytes` is in `[1, 16384]`; its provider output-token limit is
`512 + ceil(max_rationale_bytes / 4)`. Cool-off durations are positive and the
maximum is not less than the base. Judge unknown fields are always errors.

Every pool must include this policy even in `off` mode. `off` validates all
supplied fields but opens no ledger, starts no task, and registers no Router
runtime behavior. `shadow`, `recommend`, and `active` secure and migrate the
ledger, reconcile expired process work, start owned workers, and then register
runtime behavior.

A pool can set `learning = { version = 1, embedder = "profile-id" }` to activate
canonical query, embedding, and vector-materialization work. A complete
learning policy adds all confidence fields. Shadow accepts minimal or complete
learning for evidence warm-up; Recommend requires a complete policy for every
pool. A partial statistical policy is invalid. Omitted or empty `learning`
disables vectorization for that pool. Only referenced profiles read a
configured credential environment variable or create provider clients.

Active requires three additional learning fields:
`retention_lower_bound`, `holdout_probability`, and
`active_canary_fraction`. It also requires a complete nonempty version-1
`outcome` policy for every pool. Holdout is in `(0, 0.25]`, the canary fraction
is in `(0, 1)`, their sum is less than `1`, and the retention bound is
nonnegative and lower than `promotion_lower_bound`. Refer to
[Active Canary and Outcomes](https://docs.nvidia.com/nemo/relay/router/active-canary-and-outcomes)
for the complete configuration and rollout contract.

Per-pool candidate capacity is
`max_pending * max_candidates_per_sample`. The sum of batch slots and the sum
of candidate slots across pools must each be at most `65536`. The bounded
writer capacity is derived from total candidate slots. Validation uses checked
arithmetic.

Recommend retains the complete capability-eligible candidate set rather than
the Shadow-only `max_candidates_per_sample` prefix. A complete policy permits
at most 64 candidates, requires `candidate_count * top_k <= 4095`, and has a
32 MiB decision-command limit. Four process-wide slots bound concurrent
recommendation computation and retained audit delivery.

For the complete TOML example and field rules, refer to
[Router Configuration](https://docs.nvidia.com/nemo/relay/router/configuration).

## Understand Eligibility and Execution

Router supports OpenAI Chat Completions, OpenAI Responses, and Anthropic
Messages through Core's V2 codec and replay contracts. An anchor must be a
non-streaming, non-stateful Primary call with an exact pool/family/model match,
a current replay capability, a safe bounded transport identity, and at least
one compatible candidate. Responses continuations and requests with
`store = true` are ineligible.

Pool selectors use only frozen V2 context. Candidate capabilities must cover
the request's tools, multimodal input, structured output, and reasoning
controls. Eligible candidates sort by `(cost_rank, candidate_id)` before
Shadow truncation. Candidate requests are codec-built, model-only rewrites.
Router proves that headers and all non-model request semantics remain
unchanged.

Router invokes the anchor continuation exactly once with the original request
and returns its raw value or error unchanged. A successful sampled call can
open a bounded future-local window. After that window and its reserved attempts
are durable, candidates run independently under the Shadow semaphore. Each
deterministically valid candidate response can then use the separate Judge
semaphore.

In `recommend`, Router inspects replay capability only for exact partition
identity, then drops executable replay authority before query construction. It
performs cache-first embedding and strict candidate-partition search, applies
root, coverage, decay, effective-sample, and candidate-corrected lower-bound
gates, and constructs one immutable audit. It schedules no future-local window,
candidate replay, or Judge call. It submits the immutable audit before the
unchanged anchor continuation. Every failure still invokes that continuation
exactly once.

In `active`, Router reruns the complete fresh recommendation gate and verifies
current experiment, outcome-look, control, generation, tranche, cap, and
query-local neighborhood authority. It deterministically assigns each
independent root to candidate treatment, anchor control, or anchor holdout.
Before a candidate substitution, one transaction commits the decision, root
window, assignment, propensity, and dispatch. The current managed continuation
then runs exactly once with the model-only rewrite. A provider error is returned
without an anchor retry. A malformed successful response is returned unchanged
but records a policy failure and starts query-local cool-off.

Active outcomes use bounded protected signals plus the representative Primary
result. Candidate transport failure, codec failure, policy failure, and
attributable Tool failure invalidate only the exact experiment/query/neighbor
identity. Operator pause disables new Active and Shadow work; force-anchor
disables substitution while retaining eligible Shadow work and maintenance.

Each real candidate or Judge transport start occurs inside one managed V2 call
under an Evaluator scope with role `Shadow` or `Judge`. These internal calls do
not recursively sample, advance Primary trajectories, enter default ATIF
exports, or train Adaptive behavior.

For detailed selection, projection, and lifecycle behavior, refer to
[Current Runtime Behavior](https://docs.nvidia.com/nemo/relay/router/current-behavior)
and
[Future-Local Trajectories](https://docs.nvidia.com/nemo/relay/router/future-local-trajectories).

Refer to
[Router Recommendation Confidence](https://docs.nvidia.com/nemo/relay/router/recommendation-confidence)
for warm-up, bounds, numeric semantics, decision reasons, audit retry,
retention, and privacy.

## Interpret Evidence

Complete, readable response-schema, tool-contract, and malformed-family
violations are deterministic quality failures. They bypass the Judge and create
a `Fail` record. Transport, authentication, rate-limit, timeout, cancellation,
unreadable, truncated, ambiguous-decode, evidence-bound, middleware, and
invalid-Judge-output failures are operational. They create no quality label.

A Judge returns response equivalence, trajectory equivalence, confidence, hard
failures, and a bounded rationale. Invalid structured output gets at most one
separate repair attempt. Hard failures take precedence. Low confidence produces
`Ambiguous` with no binary label. Otherwise, both component floors and the
weighted aggregate threshold must pass.

A partial future-local window can retain a label for inspection but is never
promotion eligible. Recommend can use full promotion-eligible `Pass` or `Fail`
evidence from the exact partition. Judge evidence must independently meet the
current confidence floor. Router collapses labeled evidence to one observation
per trajectory root and keeps unlabeled terminal roots in the coverage
denominator. It still does not serve the recorded recommendation.
Operational dependency failures apply exponential project-wide cool-off,
scoped independently to the candidate or Judge dependency. Success resets the
failure count.

Refer to
[Shadow Evaluation and Evidence](https://docs.nvidia.com/nemo/relay/router/shadow-evaluation-and-evidence)
for the exact scoring and evidence contract.

## Build Embedding and Vector Evidence

For an associated pool, every canonicalizable terminal attempt and applicable
vector space has its own evidence link and materialization state. Canonical
System and Developer instructions, the latest nonempty User task, bounded
textual context, tool and response-schema fingerprints, required capabilities,
and configured `turn_index` become one RFC 8785 embedding input. Models, API
family, headers, credentials, replay state, arbitrary metadata, and generation
controls are not part of that query.

The authoritative SQLite cache shares one embedding per vector space and query
hash. Durable fenced jobs provide cross-process single flight, bounded retries,
and permanent quarantine for invalid provider output. `sqlite-vec` 0.1.9 is a
strictly partitioned derived index; Router rebuilds immutable generations from
relational vectors without calling the provider.

Cache-backed materialization is claimed only under a verified active index.
Index outage leaves it pending without attempt growth; repair resumes it. After
64 claim attempts, the next eligible claim writes matching nonfatal
`failed_index` job and link terminals instead of retrying indefinitely. A live
lease is preserved, an expired owner is orphaned first, and verified histories
are bounded to 130 materialization events and 66 link events per job.

Provider discovery is restricted to spaces mapped by the running config
generation. Historical jobs remain inert until exact mapping authority returns.
Embedding, materialization, and rebuild claims carry monotonic start deadlines
through the writer queue, preventing a delayed writer from mutating a claim
after its operation has expired.

The synchronous internal vector-store trait and memory store are a deterministic
parity oracle. Recommend production vector operations use an asynchronous
facade over the ledger's sole writer and bounded read pool. The facade resolves
the exact partition, active generation, KNN rows, and full relational evidence
in one read transaction and does not implement the synchronous trait. Version
1 also makes retained source mandatory: missing source selected for backfill is
corruption with no partial write, while the schema's `source_expired` token
remains reserved for a future explicit retention mode.

Every embedding call sends the canonical instruction, task, and context content
to the configured endpoint, including loopback services. Nonloopback endpoints
require HTTPS and `allow_remote_embedding_egress = true`. Treat both the
endpoint and the owner-only ledger as authorized recipients of sensitive
application content.

Refer to
[Embeddings and Vector Store](https://docs.nvidia.com/nemo/relay/router/embeddings-and-vector-store)
for profile association, exact query and partition fields, endpoint policy,
cache deadlines, retries, rebuild, retention, observability, and packaging
limits.

## Operate the Ledger

In `shadow`, `recommend`, or `active` mode, `database_path` is a private SQLite ledger.
Relative paths resolve from the host working directory. On Unix, Router
enforces `0700` on the immediate database directory and `0600` on the database
and sidecars. It rejects symlinks, nonregular files, unsafe ownership, and
unsupported security boundaries. On macOS, Router normalizes the operating
system's fixed `/var` and `/tmp` aliases to their `/private` paths before secure
descriptor traversal; arbitrary symlinks remain invalid.

Migrations are ordered and checksum verified. A checksum mismatch, unsupported
future schema, configured project mismatch, or malformed durable invariant
refuses activation before middleware registration. Router uses WAL, foreign
keys, a fixed five-second busy timeout, one bounded writer, and append-only
domain transitions. SQLite is authoritative for vector source data; sqlite-vec
generations are verified, rebuildable search indexes.

A five-second heartbeat renews a 30-second process lease. Startup atomically
terminalizes nonresumable work owned by stopped or expired processes without
reconstructing replay. Retention runs hourly and on capacity pressure. It
selects only fully terminal anchors, uses inclusive age and count limits,
deletes at most 1,000 anchors per transaction, preserves dependency cool-off
facts and referenced shared vector assets, and appends an immutable summary
even for an empty selection.

Decision retention uses an independent age and count policy. It deletes only
complete verified aggregates under cumulative limits of 1,000 parents, 4,159
summary-plus-neighbor rows, and 32 MiB per pass. A source referenced by a
decision is deindexed and marked retiring before dependent decisions and the
source anchor are removed across bounded passes.

Active decisions are excluded from ordinary decision retention. Terminal
experiments use marker-first graph retirement with per-transaction limits of
1,000 root/outcome parents, 4,095 child facts, and 32 MiB. Chained receipts and
bounded checkpoints preserve cleanup history while root windows, outcomes,
look members, authorization history, decisions, and the experiment are removed
in foreign-key-safe order.

Graceful clear closes recommendation and Active foreground admission first,
waits admitted foreground calls, performs a second subscriber flush, and
terminalizes unresolved Active roots as shutdown-orphaned. It then drains the
coordinator, outcome actor, pending recommendation audits, live embedding,
background work, retention, heartbeat, process stop, readers, and writer under
Core's one deadline. It does not drain the durable backlog or wait through
unrelated retry delays. Synchronous abort fences writer acceptance before
canceling pending delivery, closes every start gate, and interrupts owned
resources without joining or writing process stop; startup reconciliation
handles incomplete durable facts conservatively.

The ledger sanitizer excludes request headers, replay transport state,
environment values, auth headers, cookies, and bearer material. It bounds and
sanitizes normalized payloads, provider failure classes, and Judge rationales.
It cannot guarantee that ordinary free-form prompt or response text, retained
event data or metadata values, scope attributes, event names, or owner-path
scope names contain no secret. Canonical query text remains while referenced
by embedding or evidence state, so the ledger remains sensitive local data.

Refer to
[Ledger Migrations and Recovery](https://docs.nvidia.com/nemo/relay/router/ledger-migrations-and-recovery)
for maintainer guidance. Direct SQL is not a supported inspection interface.

## Inspection and Operator APIs

Open an existing ledger with
`nemo_relay_router::inspection::InspectionService`. Host-fixed
`InspectionServiceOptions` control redacted versus full content, provider-backed
request inspection, operation capability, request deadlines, and export chunk
size. Defaults are redacted, provider-free, and read-only.

The service exposes status, 24-hour overview, pool summary/detail, evidence,
decision/exposure, outcome, operator-history, health, migration, neighborhood,
and streaming JSONL/CSV export reads. An operations-capable service also
exposes generation-checked pause, force-anchor, learning reset, and cohort
rotation. It never migrates or activates Router.

Refer to
[Inspect and Control Router](https://docs.nvidia.com/nemo/relay/router/inspection-and-control)
for Rust construction, DTO schemas, cursors, filters, evidence fields, content
policy, mutations, and errors. Refer to
[Use Router CLI and HTTP](https://docs.nvidia.com/nemo/relay/router/cli-and-http)
for exact CLI syntax and the optional authenticated Axum adapter. Refer to
[Use the Local Router Dashboard](https://docs.nvidia.com/nemo/relay/router/dashboard)
for the embedded read-only CLI dashboard.

## Package and Operations Boundary

Published CLI, wheel, and npm artifacts include the pinned sqlite-vec native
extension and verify it with a temporary `vec0` create, insert, and nearest-row
query before release upload. Bundled hosts also use bounded asynchronous plugin
clear so Router can drain owned work during graceful shutdown.

Rust, the bundled CLI, and the optional Rust HTTP adapter expose supported
inspection and operator surfaces. Python and Node.js expose typed configuration
helpers but no inspection or operator methods. The raw FFI and Go APIs remain
V2 replay-ineligible and expose no typed Router inspection or control surface.
The CLI dashboard consumes the Rust inspection API and adds no binding surface.
