// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

use nemo_relay_types::api::llm::LlmApiFamily;
use nemo_relay_types::api::scope::ScopeType;
use serde::{Deserialize, Serialize};
use serde_json::Value as Json;
use unicode_normalization::UnicodeNormalization;
use uuid::Uuid;

use crate::RoutingPartitionV1;
use crate::config::RouterMode;
use crate::control::{ControlOperation, ControlScope, RouterControlState};
use crate::projection::{RouterRequestProjectionV1, RouterRoutingContextProjectionV1};

/// Inspection HTTP and CLI API major version.
pub const INSPECTION_API_VERSION_V1: u32 = 1;
/// Schema identifier for request-file neighborhood input.
pub const ROUTING_INSPECTION_INPUT_SCHEMA_V1: &str = "nemo.relay.router.inspection-input@1";
/// Schema identifier for status reports.
pub const STATUS_REPORT_SCHEMA_V1: &str = "nemo.relay.router.status@1";
/// Schema identifier for fixed-window overview reports.
pub const OVERVIEW_REPORT_SCHEMA_V1: &str = "nemo.relay.router.overview@1";
/// Schema identifier for one configured pool detail report.
pub const POOL_DETAIL_SCHEMA_V1: &str = "nemo.relay.router.pool-detail@1";
/// Schema identifier for one decision exposure report.
pub const DECISION_EXPOSURE_SCHEMA_V1: &str = "nemo.relay.router.decision-exposure@1";
/// Schema identifier for neighborhood reports.
pub const NEIGHBORHOOD_REPORT_SCHEMA_V1: &str = "nemo.relay.router.neighborhood-report@1";
/// Schema identifier for one JSON Lines evidence export record.
pub const EVIDENCE_EXPORT_RECORD_SCHEMA_V1: &str = "nemo.relay.router.evidence-export-record@1";
/// Default number of items returned by one list request.
pub const INSPECTION_PAGE_LIMIT_DEFAULT: u16 = 100;
/// Maximum number of items returned by one list request.
pub const INSPECTION_PAGE_LIMIT_MAX: u16 = 500;
/// Maximum opaque cursor bytes accepted from a caller.
pub const INSPECTION_CURSOR_MAX_BYTES: usize = 4 * 1024;
/// Maximum request or partition file bytes accepted by the CLI and HTTP adapter.
pub const INSPECTION_INPUT_MAX_BYTES: usize = 16 * 1024 * 1024;
/// Maximum Unicode scalar values in a redacted preview.
pub const INSPECTION_REDACTED_PREVIEW_MAX_CHARS: usize = 256;
/// Default bounded export chunk size.
pub const INSPECTION_EXPORT_CHUNK_BYTES_DEFAULT: usize = 64 * 1024;
/// Minimum bounded export chunk size.
pub const INSPECTION_EXPORT_CHUNK_BYTES_MIN: usize = 4 * 1024;
/// Maximum bounded export chunk size.
pub const INSPECTION_EXPORT_CHUNK_BYTES_MAX: usize = 1024 * 1024;
/// Default completion budget for one inspection request.
pub const INSPECTION_REQUEST_TIMEOUT_MS_DEFAULT: u64 = 5_000;
/// Maximum completion budget accepted from a host.
pub const INSPECTION_REQUEST_TIMEOUT_MS_MAX: u64 = 30_000;
/// Fixed recent window used by the version-1 overview report.
pub const OVERVIEW_WINDOW_MS_V1: u64 = 24 * 60 * 60 * 1_000;
/// Maximum points admitted to diagnostic projection.
pub const DIAGNOSTIC_PROJECTION_POINTS_MAX: usize = 512;
/// Maximum symmetric matrix cells admitted to diagnostic projection.
pub const DIAGNOSTIC_PROJECTION_MATRIX_CELLS_MAX: usize = 262_144;
/// Maximum checked multiply-add operations admitted to diagnostic projection.
pub const DIAGNOSTIC_PROJECTION_MULTIPLY_ADDS_MAX: u64 = 64 * 1024 * 1024;

/// Stable failure returned by inspection, export, HTTP, and CLI adapters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InspectionError {
    /// A request shape, value, bound, identifier, or file is invalid.
    InvalidArgument,
    /// An opaque cursor is malformed, altered, stale, or used with another query.
    InvalidCursor,
    /// The requested immutable object does not exist.
    NotFound,
    /// An HTTP request did not satisfy its host authentication guard.
    Unauthorized,
    /// The host did not grant the requested content or operation capability.
    Forbidden,
    /// A compare-and-swap generation no longer matches current authority.
    Conflict,
    /// The bounded queue, read, or transaction start deadline elapsed.
    Busy,
    /// An idempotency key is outside the durable replay horizon.
    MutationExpired,
    /// A bounded durable history cannot accept another mutation.
    CapacityExhausted,
    /// The Router ledger is missing, unreadable, read-only for a mutation, or unavailable.
    StorageUnavailable,
    /// The ledger requires a migration before this operation can run.
    MigrationRequired,
    /// Explicit request inspection would violate configured embedding egress policy.
    EgressDenied,
    /// Stored authority or an immutable record failed deterministic verification.
    IntegrityError,
    /// The caller and service disagree on the inspection API major.
    IncompatibleApi,
}

impl InspectionError {
    /// Return the stable machine-readable error code.
    pub const fn code(self) -> &'static str {
        match self {
            Self::InvalidArgument => "invalid_argument",
            Self::InvalidCursor => "invalid_cursor",
            Self::NotFound => "not_found",
            Self::Unauthorized => "unauthorized",
            Self::Forbidden => "forbidden",
            Self::Conflict => "conflict",
            Self::Busy => "busy",
            Self::MutationExpired => "mutation_expired",
            Self::CapacityExhausted => "capacity_exhausted",
            Self::StorageUnavailable => "storage_unavailable",
            Self::MigrationRequired => "migration_required",
            Self::EgressDenied => "egress_denied",
            Self::IntegrityError => "integrity_error",
            Self::IncompatibleApi => "incompatible_api",
        }
    }
}

impl fmt::Display for InspectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.code())
    }
}

impl Error for InspectionError {}

/// Maximum content detail granted by the host that constructs a service.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentPolicy {
    /// Return hashes, lengths, typed metadata, and bounded secret-filtered previews.
    #[default]
    Redacted,
    /// Return larger allowed content while still removing secrets.
    Full,
}

/// Host-fixed inspection service limits and provider capability.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InspectionServiceOptions {
    /// Maximum content detail this service may return.
    pub content_policy: ContentPolicy,
    /// Whether request-file inspection may call one configured embedder.
    pub allow_request_embedding: bool,
    /// Whether this host may create or attach bounded mutation authority.
    pub allow_operations: bool,
    /// Completion deadline for one bounded service request.
    pub request_timeout_ms: u64,
    /// Maximum bytes emitted by one incremental export chunk.
    pub export_chunk_bytes: usize,
}

impl Default for InspectionServiceOptions {
    fn default() -> Self {
        Self {
            content_policy: ContentPolicy::Redacted,
            allow_request_embedding: false,
            allow_operations: false,
            request_timeout_ms: INSPECTION_REQUEST_TIMEOUT_MS_DEFAULT,
            export_chunk_bytes: INSPECTION_EXPORT_CHUNK_BYTES_DEFAULT,
        }
    }
}

impl InspectionServiceOptions {
    /// Validate all host-fixed bounds before opening a ledger.
    pub fn validate(&self) -> Result<(), InspectionError> {
        if !(1..=INSPECTION_REQUEST_TIMEOUT_MS_MAX).contains(&self.request_timeout_ms)
            || !(INSPECTION_EXPORT_CHUNK_BYTES_MIN..=INSPECTION_EXPORT_CHUNK_BYTES_MAX)
                .contains(&self.export_chunk_bytes)
        {
            return Err(InspectionError::InvalidArgument);
        }
        Ok(())
    }
}

/// Bounded request for one stable page.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PageRequest {
    /// Requested item count in `1..=500`.
    pub limit: u16,
    /// Opaque cursor returned by a previous request with the same filters.
    pub after: Option<String>,
}

impl Default for PageRequest {
    fn default() -> Self {
        Self {
            limit: INSPECTION_PAGE_LIMIT_DEFAULT,
            after: None,
        }
    }
}

impl PageRequest {
    /// Validate the page bound and cursor resource limit.
    pub fn validate(&self) -> Result<(), InspectionError> {
        if !(1..=INSPECTION_PAGE_LIMIT_MAX).contains(&self.limit)
            || self.after.as_ref().is_some_and(|cursor| {
                cursor.is_empty() || cursor.len() > INSPECTION_CURSOR_MAX_BYTES
            })
        {
            return Err(InspectionError::InvalidArgument);
        }
        Ok(())
    }
}

/// Stable page of typed inspection records.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Page<T> {
    /// Records in the endpoint's declared deterministic order.
    pub items: Vec<T>,
    /// Cursor for the next page, or `None` at the snapshot boundary.
    pub next: Option<String>,
    /// UTC millisecond time captured for the first page.
    pub snapshot_time_unix_ms: u64,
    /// Effective host-fixed content policy.
    pub content_policy: ContentPolicy,
}

/// Router database state exposed without path or SQL details.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InspectionDatabaseStateV1 {
    /// No database exists at the configured path.
    Missing,
    /// The configured file is an empty SQLite database.
    Empty,
    /// The database matches the current complete schema.
    Current,
    /// A known migration is required.
    UpgradeRequired,
    /// The database was created by a newer Router build.
    Newer,
    /// The file is not a valid Router ledger at its claimed version.
    Incompatible,
    /// The path or database metadata cannot be read.
    Unreadable,
}

/// Effective runtime behavior after durable controls are applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EffectiveRouterModeV1 {
    /// Router is configured off.
    Off,
    /// Router observes Shadow evidence only.
    Shadow,
    /// Router computes recommendations while serving the anchor.
    Recommend,
    /// Router may serve a durably authorized candidate.
    Active,
    /// Operator pause prevents substitution and new background work.
    Paused,
    /// Operator force-anchor prevents substitution while learning may continue.
    ForceAnchor,
    /// Durable authority is not readable, so serving must fail to the anchor.
    Unavailable,
}

/// Secret-free database and migration summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatabaseStatusV1 {
    /// Stable database classification.
    pub state: InspectionDatabaseStateV1,
    /// Observed SQLite application ID when readable.
    pub application_id: Option<i64>,
    /// Observed schema version when readable.
    pub schema_version: Option<i64>,
    /// Highest schema version supported by this Router build.
    pub supported_schema_version: i64,
}

/// Current all-scope and effective control summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ControlStatusV1 {
    /// Gap-free global control generation.
    pub control_generation: u64,
    /// Current global control values.
    pub all: RouterControlState,
    /// Effective values after optional pool-local state.
    pub effective: RouterControlState,
}

/// Aggregate bounded queue status.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueueStatusV1 {
    /// Current queued commands or tasks.
    pub pending: u64,
    /// Configured or derived capacity.
    pub capacity: u64,
    /// Number of work items rejected for pressure in the current snapshot.
    pub rejected: u64,
}

/// Aggregate lease status.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LeaseStatusV1 {
    /// Currently live leases.
    pub active: u64,
    /// Expired leases awaiting or completing reconciliation.
    pub expired: u64,
}

/// Aggregate vector-index status.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VectorIndexStatusV1 {
    /// Number of active vector spaces.
    pub active_spaces: u64,
    /// Number of rebuilding vector spaces.
    pub rebuilding_spaces: u64,
    /// Number of degraded vector spaces.
    pub degraded_spaces: u64,
    /// Active vector spaces whose applied source sequence trails current evidence.
    pub stale_spaces: u64,
}

/// Latest durable data timestamps in one consistent inspection snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FreshnessStatusV1 {
    /// UTC creation time of the newest evidence record.
    pub latest_evidence_unix_ms: Option<u64>,
    /// UTC creation time of the newest persisted decision.
    pub latest_decision_unix_ms: Option<u64>,
    /// UTC creation time of the newest randomized outcome.
    pub latest_outcome_unix_ms: Option<u64>,
}

/// Aggregate health status without raw diagnostic content.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HealthSummaryV1 {
    /// Stable worst health class.
    pub worst: String,
    /// Number of currently degraded subjects.
    pub degraded: u64,
    /// UTC time of the newest verified health event.
    pub latest_event_unix_ms: Option<u64>,
}

impl Default for HealthSummaryV1 {
    fn default() -> Self {
        Self {
            worst: "clear".into(),
            degraded: 0,
            latest_event_unix_ms: None,
        }
    }
}

/// Top-level version-1 Router status report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StatusReportV1 {
    /// Constant [`STATUS_REPORT_SCHEMA_V1`].
    pub schema: String,
    /// Non-secret project confirmation ID from configuration.
    pub project_id: Option<String>,
    /// Configured Router mode.
    pub configured_mode: RouterMode,
    /// Effective behavior after control and authority checks.
    pub effective_mode: EffectiveRouterModeV1,
    /// Current canonical config generation hash.
    pub config_generation_id: Option<String>,
    /// Current cohort generation without protected salt.
    pub cohort_generation_id: Option<Uuid>,
    /// Current global/pool control summary when readable.
    pub controls: Option<ControlStatusV1>,
    /// Database and migration classification.
    pub database: DatabaseStatusV1,
    /// Aggregate queue status.
    pub queues: QueueStatusV1,
    /// Aggregate lease status.
    pub leases: LeaseStatusV1,
    /// Aggregate vector-index status.
    pub vector_index: VectorIndexStatusV1,
    /// Latest durable evidence, decision, and outcome timestamps.
    pub freshness: FreshnessStatusV1,
    /// Aggregate health status.
    pub health: HealthSummaryV1,
    /// UTC time at which this report was captured.
    pub snapshot_time_unix_ms: u64,
}

/// Fixed-window routing-decision counts for one overview snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverviewDecisionCountsV1 {
    /// All persisted decisions inside the declared window.
    pub total: u64,
    /// Recommend-mode decisions inside the window.
    pub recommend: u64,
    /// Active-mode decisions inside the window.
    pub active: u64,
    /// Active decisions whose persisted planned route served a candidate.
    pub candidate_served: u64,
    /// Decisions that served an anchor.
    pub anchor_served: u64,
    /// Active decisions whose persisted planned route was an anchor fallback.
    pub fallback: u64,
    /// Counts by stable persisted final reason.
    pub final_reasons: BTreeMap<String, u64>,
}

/// Fixed-window randomized assignment counts by durable arm.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverviewExposureCountsV1 {
    /// Candidate-treatment assignments.
    pub candidate_treatment: u64,
    /// Randomized anchor-control assignments.
    pub anchor_control: u64,
    /// Randomized anchor-holdout assignments.
    pub anchor_holdout: u64,
    /// Deterministic non-learning assignments.
    pub non_learning: u64,
}

/// Fixed-window label counts for one durable Active arm.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverviewArmOutcomeCountsV1 {
    /// All outcomes for this arm inside the window.
    pub total: u64,
    /// Outcomes with a success label.
    pub success: u64,
    /// Outcomes with a failure label.
    pub failure: u64,
    /// Outcomes that remain unlabeled.
    pub unlabeled: u64,
}

/// Fixed-window outcome counts grouped by durable Active arm.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverviewOutcomeCountsV1 {
    /// Candidate-treatment outcomes.
    pub candidate_treatment: OverviewArmOutcomeCountsV1,
    /// Randomized anchor-control outcomes.
    pub anchor_control: OverviewArmOutcomeCountsV1,
    /// Randomized anchor-holdout outcomes.
    pub anchor_holdout: OverviewArmOutcomeCountsV1,
    /// Deterministic non-learning outcomes.
    pub non_learning: OverviewArmOutcomeCountsV1,
}

/// One authoritative current-state and fixed-window operational overview.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OverviewReportV1 {
    /// Constant [`OVERVIEW_REPORT_SCHEMA_V1`].
    pub schema: String,
    /// Inclusive start of the fixed 24-hour window.
    pub window_start_unix_ms: u64,
    /// Inclusive snapshot boundary for every count and embedded status field.
    pub snapshot_time_unix_ms: u64,
    /// Current Router status captured under the same authority snapshot.
    pub status: StatusReportV1,
    /// Decision counts inside the declared window.
    pub decisions: OverviewDecisionCountsV1,
    /// Randomized assignment counts inside the declared window.
    pub exposures: OverviewExposureCountsV1,
    /// Outcome counts inside the declared window.
    pub outcomes: OverviewOutcomeCountsV1,
}

/// One configured candidate in a pool summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateSummaryV1 {
    /// Candidate ID.
    pub id: String,
    /// Candidate model ID.
    pub model: String,
    /// Pinned model revision.
    pub model_revision: String,
    /// Ascending cost rank.
    pub cost_rank: u32,
}

/// Current support counters for one pool.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolSupportV1 {
    /// Ready evidence rows in the current learning generation.
    pub ready_evidence: u64,
    /// Distinct current independent roots.
    pub independent_roots: u64,
    /// Current labeled evidence coverage when available.
    pub coverage: Option<f64>,
}

/// One deterministic pool summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolSummaryV1 {
    /// Configured pool ID.
    pub id: String,
    /// Supported LLM API family.
    pub api_family: LlmApiFamily,
    /// Configured anchor model IDs.
    pub anchor_models: Vec<String>,
    /// Pinned anchor revision.
    pub anchor_revision: String,
    /// Candidates in deterministic configured cost order.
    pub candidates: Vec<CandidateSummaryV1>,
    /// Exact current pool policy hash.
    pub policy_version_id: String,
    /// Exact current learning generation.
    pub learning_generation_id: Uuid,
    /// Optional canary fraction for Active policy.
    pub active_canary_fraction: Option<f64>,
    /// Optional holdout probability for Active policy.
    pub holdout_probability: Option<f64>,
    /// Current verified support summary.
    pub support: PoolSupportV1,
    /// Current global and effective pool controls when configured.
    pub controls: Option<ControlStatusV1>,
    /// Current aggregate health summary.
    pub health: HealthSummaryV1,
}

/// Stable selector predicates displayed for one configured pool.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolSelectorDetailV1 {
    /// Exact normalized tenant IDs, or a wildcard when absent.
    pub tenant_ids: Option<Vec<String>>,
    /// Exact normalized agent IDs, or a wildcard when absent.
    pub agent_ids: Option<Vec<String>>,
    /// Exact owner scope types, or a wildcard when absent.
    pub owner_scope_types: Option<Vec<ScopeType>>,
    /// Conjunctive exact scalar metadata comparisons.
    pub metadata_equals: BTreeMap<String, Json>,
    /// Restricted owner-to-parent scope path patterns, or a wildcard when absent.
    pub scope_path_patterns: Option<Vec<String>>,
}

/// Stable bounded concurrency settings for one configured pool.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolConcurrencyDetailV1 {
    /// Maximum concurrent Shadow provider calls.
    pub shadow: u64,
    /// Maximum concurrent Judge provider calls.
    pub judge: u64,
    /// Maximum independently pending sampled intents or accepted anchors.
    pub max_pending: u64,
}

/// Exact optional version-1 learning policy fields for one configured pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolLearningPolicyV1 {
    /// Learning policy shape version.
    pub version: u32,
    /// Stable embedding profile ID.
    pub embedder: String,
    /// Maximum nearest neighbors per candidate.
    pub top_k: Option<u64>,
    /// Inclusive cosine-distance radius.
    pub radius: Option<f64>,
    /// Minimum query-local labeled points.
    pub min_points: Option<u64>,
    /// Minimum query-local independent roots.
    pub min_independent_roots: Option<u64>,
    /// Minimum query-local Kish effective sample size.
    pub min_effective_samples: Option<f64>,
    /// Minimum query-local labeled-root coverage.
    pub min_coverage: Option<f64>,
    /// Evidence time-decay half-life in seconds.
    pub time_decay_half_life_seconds: Option<f64>,
    /// Positive Beta prior success shape.
    pub prior_success: Option<f64>,
    /// Positive Beta prior failure shape.
    pub prior_failure: Option<f64>,
    /// Familywise credible level before candidate correction.
    pub familywise_credible_level: Option<f64>,
    /// Promotion lower-bound threshold.
    pub promotion_lower_bound: Option<f64>,
    /// Active retention lower-bound threshold.
    pub retention_lower_bound: Option<f64>,
    /// Unconditional Active holdout probability.
    pub holdout_probability: Option<f64>,
    /// Unconditional Active candidate-canary probability.
    pub active_canary_fraction: Option<f64>,
}

/// Necessary global support comparisons; query-local evaluation remains authoritative.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolSupportPrerequisitesV1 {
    /// Whether total ready evidence reaches the configured point minimum, when configured.
    pub ready_evidence: Option<bool>,
    /// Whether total independent roots reach the configured root minimum, when configured.
    pub independent_roots: Option<bool>,
    /// Whether global labeled coverage reaches the configured minimum, when available.
    pub coverage: Option<bool>,
    /// Always true for a complete statistical policy; global counts are not eligibility.
    pub query_local_evaluation_required: bool,
}

/// Complete stable inspection detail for one configured pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolDetailV1 {
    /// Constant [`POOL_DETAIL_SCHEMA_V1`].
    pub schema: String,
    /// Existing stable pool summary.
    pub summary: PoolSummaryV1,
    /// Probability of proposing a Shadow sample.
    pub sampling_probability: f64,
    /// Maximum eligible candidates proposed by one sample.
    pub max_candidates_per_sample: u64,
    /// Frozen-context selector predicates.
    pub selector: PoolSelectorDetailV1,
    /// Bounded concurrency settings.
    pub concurrency: PoolConcurrencyDetailV1,
    /// Optional association/statistical/Active learning policy.
    pub learning: Option<PoolLearningPolicyV1>,
    /// Necessary global prerequisites, never a query-local eligibility claim.
    pub support_prerequisites: PoolSupportPrerequisitesV1,
}

/// Common secret-filtered content field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InspectionContentV1 {
    /// SHA-256 of the canonical allowed source content.
    pub sha256: String,
    /// Canonical source byte length before preview truncation.
    pub byte_length: u64,
    /// Redacted bounded preview when available.
    pub preview: Option<String>,
    /// Full secret-filtered content only under host-granted full policy.
    pub value: Option<Json>,
}

/// Evidence list filters bound into the page cursor.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvidenceFilterV1 {
    /// Optional exact pool ID.
    pub pool_id: Option<String>,
    /// Optional exact candidate ID.
    pub candidate_id: Option<String>,
    /// Optional exact terminal class.
    pub terminal_class: Option<String>,
    /// Optional exact `pass` or `fail` quality label.
    pub quality_label: Option<String>,
    /// Optional exact learning generation.
    pub learning_generation_id: Option<Uuid>,
}

impl EvidenceFilterV1 {
    /// Validate exact filter values before cursor decoding or storage work.
    pub fn validate(&self) -> Result<(), InspectionError> {
        if !valid_optional_id(self.pool_id.as_deref())
            || !valid_optional_id(self.candidate_id.as_deref())
            || self.terminal_class.as_deref().is_some_and(|value| {
                !matches!(
                    value,
                    "completed"
                        | "deterministic_failure"
                        | "operational_failure"
                        | "skipped_cooloff"
                        | "canceled_shutdown"
                        | "orphaned_before_schedule"
                        | "orphaned_in_flight"
                )
            })
            || self
                .quality_label
                .as_deref()
                .is_some_and(|value| !matches!(value, "pass" | "fail"))
            || self.learning_generation_id.is_some_and(|value| {
                value.get_variant() != uuid::Variant::RFC4122 || value.get_version_num() != 7
            })
        {
            return Err(InspectionError::InvalidArgument);
        }
        Ok(())
    }
}

/// One immutable evidence summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceSummaryV1 {
    /// Immutable evidence vector-link UUID.
    pub evidence_id: Uuid,
    /// Pool ID resolved from the strict partition.
    pub pool_id: String,
    /// Candidate ID resolved from the strict partition.
    pub candidate_id: String,
    /// Current evidence terminal class.
    pub terminal_class: String,
    /// Optional binary quality label.
    pub quality_label: Option<String>,
    /// Canonical routing query hash.
    pub canonical_query_hash: String,
    /// Exact learning generation recorded by this evidence.
    pub learning_generation_id: Uuid,
    /// Latest vector-link state.
    pub vector_state: String,
    /// Writer-owned UTC creation time.
    pub created_at_unix_ms: u64,
    /// Secret-filtered query content summary.
    pub content: InspectionContentV1,
}

/// Complete bounded evidence detail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceDetailV1 {
    /// List-level summary.
    pub summary: EvidenceSummaryV1,
    /// Exact strict partition.
    pub partition: RoutingPartitionV1,
    /// Anchor UUID.
    pub anchor_id: Uuid,
    /// Shadow attempt UUID.
    pub shadow_attempt_id: Uuid,
    /// Optional evaluation UUID.
    pub evaluation_id: Option<Uuid>,
    /// Immutable evidence payload hash.
    pub record_hash: String,
}

/// Decision list filters bound into the page or follow cursor.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DecisionFilterV1 {
    /// Optional exact pool ID.
    pub pool_id: Option<String>,
    /// Optional exact mode.
    pub mode: Option<String>,
    /// Optional exact candidate ID.
    pub candidate_id: Option<String>,
    /// Optional exact final reason.
    pub final_reason: Option<String>,
}

impl DecisionFilterV1 {
    /// Validate exact filter values before cursor decoding or storage work.
    pub fn validate(&self) -> Result<(), InspectionError> {
        if !valid_optional_id(self.pool_id.as_deref())
            || !valid_optional_id(self.candidate_id.as_deref())
            || self
                .mode
                .as_deref()
                .is_some_and(|value| !matches!(value, "recommend" | "active"))
            || !valid_optional_reason(self.final_reason.as_deref())
        {
            return Err(InspectionError::InvalidArgument);
        }
        Ok(())
    }
}

/// One immutable decision summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSummaryV1 {
    /// Decision UUID.
    pub decision_id: Uuid,
    /// Pool ID.
    pub pool_id: String,
    /// `recommend` or `active`.
    pub mode: String,
    /// Candidate chosen by confidence, if any.
    pub candidate_id: Option<String>,
    /// Recommended model.
    pub recommended_model: String,
    /// Model actually served.
    pub served_model: String,
    /// Stable final reason.
    pub final_reason: String,
    /// Canonical query hash.
    pub canonical_query_hash: String,
    /// Optional cohort generation for Active decisions.
    pub cohort_generation_id: Option<Uuid>,
    /// Writer-owned UTC creation time.
    pub created_at_unix_ms: u64,
}

/// One decision candidate confidence summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionCandidateSummaryV1 {
    /// Candidate ID.
    pub candidate_id: String,
    /// Deterministic candidate rank.
    pub rank_ordinal: u32,
    /// Optional partition hash when evaluated.
    pub partition_hash: Option<String>,
    /// Returned neighbor count.
    pub neighbor_count: u32,
    /// Optional effective sample size.
    pub effective_sample_size: Option<f64>,
    /// Optional posterior lower bound.
    pub lower_bound: Option<f64>,
    /// Stable terminal confidence reason.
    pub reason: String,
}

/// One persisted decision neighbor audit row.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionNeighborV1 {
    /// Global neighbor order in this decision.
    pub neighbor_ordinal: u32,
    /// Candidate ID.
    pub candidate_id: String,
    /// Evidence vector-link UUID.
    pub evidence_id: Uuid,
    /// Cosine distance.
    pub distance: f64,
    /// Optional final confidence weight.
    pub final_weight: Option<f64>,
    /// Optional binary quality label.
    pub binary_label: Option<String>,
    /// Stable inclusion/exclusion reason.
    pub exclusion_reason: String,
}

/// Complete bounded decision detail.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionDetailV1 {
    /// List-level decision summary.
    pub summary: DecisionSummaryV1,
    /// Candidate summaries in rank order.
    pub candidates: Vec<DecisionCandidateSummaryV1>,
    /// Neighbor audit in global order.
    pub neighbors: Vec<DecisionNeighborV1>,
    /// Immutable parent payload hash.
    pub record_hash: String,
}

/// Outcome list filters bound into the page cursor.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OutcomeFilterV1 {
    /// Optional exact pool ID.
    pub pool_id: Option<String>,
    /// Optional exact arm.
    pub arm: Option<String>,
    /// Optional exact label.
    pub label: Option<String>,
    /// Optional exact attribution status.
    pub attribution_status: Option<String>,
}

impl OutcomeFilterV1 {
    /// Validate exact filter values before cursor decoding or storage work.
    pub fn validate(&self) -> Result<(), InspectionError> {
        if !valid_optional_id(self.pool_id.as_deref())
            || self.arm.as_deref().is_some_and(|value| {
                !matches!(
                    value,
                    "candidate_treatment" | "anchor_control" | "anchor_holdout" | "non_learning"
                )
            })
            || self
                .label
                .as_deref()
                .is_some_and(|value| !matches!(value, "success" | "failure"))
            || self.attribution_status.as_deref().is_some_and(|value| {
                !matches!(
                    value,
                    "eligible_treatment"
                        | "eligible_control"
                        | "monitoring_only"
                        | "unattributed"
                        | "orphaned"
                        | "ambiguous_exposure"
                )
            })
        {
            return Err(InspectionError::InvalidArgument);
        }
        Ok(())
    }
}

/// One immutable randomized Active outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutcomeSummaryV1 {
    /// Outcome UUID.
    pub outcome_id: Uuid,
    /// Active experiment UUID.
    pub active_experiment_id: Uuid,
    /// Optional representative decision UUID.
    pub decision_id: Option<Uuid>,
    /// Randomized arm.
    pub arm: String,
    /// Optional success/failure label.
    pub label: Option<String>,
    /// Stable attribution status.
    pub attribution_status: String,
    /// Bounded latency in milliseconds.
    pub latency_ms: f64,
    /// Writer-owned UTC creation time.
    pub created_at_unix_ms: u64,
}

/// Persisted Active assignment and planned-route facts for one decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActiveDecisionExposureV1 {
    /// Active experiment UUID when this decision entered an experiment.
    pub active_experiment_id: Option<Uuid>,
    /// Active assignment UUID when this decision admitted a root.
    pub active_assignment_id: Option<Uuid>,
    /// Stable planned route such as `candidate` or `anchor_fallback`.
    pub planned_route: String,
    /// Stable durable assignment arm.
    pub assignment_arm: String,
    /// Control generation captured by the Active decision.
    pub control_generation: u64,
    /// Promotion threshold captured by the decision.
    pub promotion_lower_bound: f64,
    /// Retention threshold captured by the decision.
    pub retention_lower_bound: f64,
    /// Configured unconditional holdout probability.
    pub configured_holdout_probability: f64,
    /// Configured unconditional candidate-canary probability.
    pub configured_canary_probability: f64,
    /// Effective probability of the randomized assignment arm.
    pub effective_arm_probability: f64,
    /// Conditional probability of the selected candidate within its arm.
    pub conditional_selection_probability: f64,
    /// Complete persisted assignment propensity.
    pub propensity: f64,
    /// Stable fallback reason when the planned route fell back to an anchor.
    pub fallback_reason: Option<String>,
}

/// Stable exposure and representative outcome facts for one persisted decision.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionExposureV1 {
    /// Constant [`DECISION_EXPOSURE_SCHEMA_V1`].
    pub schema: String,
    /// Immutable decision UUID.
    pub decision_id: Uuid,
    /// Persisted Active facts, absent for Recommend decisions.
    pub active: Option<ActiveDecisionExposureV1>,
    /// Representative outcome directly linked to this decision, when committed.
    pub outcome: Option<OutcomeSummaryV1>,
}

/// Kind of immutable operator history entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorHistoryKindV1 {
    /// Pause or resume mutation.
    Pause,
    /// Force-anchor set or clear mutation.
    ForceAnchor,
    /// One-pool learning reset.
    ResetPool,
    /// Atomic all-pool learning reset.
    ResetAll,
    /// Cohort salt-generation rotation.
    RotateCohort,
}

/// One immutable control or generation operation record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorHistoryEntryV1 {
    /// Caller-generated mutation or audit UUID.
    pub audit_id: Uuid,
    /// Operation kind.
    pub kind: OperatorHistoryKindV1,
    /// Stable result such as `applied`, `no_op`, or `conflict`.
    pub result: String,
    /// Resulting global control generation for a control mutation.
    pub control_generation: Option<u64>,
    /// Optional control scope.
    pub scope: Option<ControlScope>,
    /// Optional prior boolean value.
    pub prior_value: Option<bool>,
    /// Optional resulting boolean value.
    pub new_value: Option<bool>,
    /// Prior generation IDs by lexical scope key.
    pub prior_generations: BTreeMap<String, Uuid>,
    /// Resulting generation IDs by lexical scope key.
    pub new_generations: BTreeMap<String, Uuid>,
    /// Superseded IDs retained for inspection.
    pub superseded_ids: Vec<Uuid>,
    /// Bounded actor.
    pub actor: String,
    /// Bounded reason.
    pub reason: String,
    /// Writer-owned UTC creation time.
    pub created_at_unix_ms: u64,
}

/// One stable health event summary.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HealthEventV1 {
    /// Health event UUID.
    pub health_event_id: Uuid,
    /// Stable subject kind.
    pub subject_kind: String,
    /// Non-secret subject identity hash or bounded ID.
    pub subject_id: Option<String>,
    /// Stable severity.
    pub severity: String,
    /// Stable reason code.
    pub reason: String,
    /// Writer-owned UTC creation time.
    pub created_at_unix_ms: u64,
}

/// One verified immutable schema migration receipt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MigrationSummaryV1 {
    /// Ordered schema version.
    pub version: i64,
    /// Immutable migration name.
    pub name: String,
    /// Expected migration SQL hash.
    pub sha256: String,
    /// UTC application time when present.
    pub applied_at_unix_ms: Option<u64>,
    /// Whether the stored receipt matches this build.
    pub verified: bool,
}

/// Exact version-1 input for request-based neighborhood inspection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoutingInspectionInputV1 {
    /// Constant [`ROUTING_INSPECTION_INPUT_SCHEMA_V1`].
    pub schema: String,
    /// Configured pool ID.
    pub pool_id: String,
    /// Exact strict routing partition.
    pub partition: RoutingPartitionV1,
    /// Sanitized request projection.
    pub request: RouterRequestProjectionV1,
    /// Sanitized routing-context projection.
    pub routing_context: RouterRoutingContextProjectionV1,
}

/// Exactly one neighborhood lookup source.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum NeighborhoodLookupV1 {
    /// Resolve the persisted query and partition for one evidence row.
    Evidence {
        /// Immutable evidence vector-link UUID.
        evidence_id: Uuid,
    },
    /// Resolve one persisted query under an exact strict partition.
    QueryHash {
        /// Canonical routing query hash.
        canonical_query_hash: String,
        /// Exact strict partition; a hash alone is insufficient.
        partition: Box<RoutingPartitionV1>,
    },
    /// Canonicalize and optionally embed one versioned sanitized request file.
    Request {
        /// Complete versioned inspection input.
        input: Box<RoutingInspectionInputV1>,
    },
}

impl NeighborhoodLookupV1 {
    /// Validate lookup identity and exact public partition shape before storage work.
    pub fn validate(&self) -> Result<(), InspectionError> {
        let valid = match self {
            Self::Evidence { evidence_id } => {
                evidence_id.get_variant() == uuid::Variant::RFC4122
                    && evidence_id.get_version_num() == 7
            }
            Self::QueryHash {
                canonical_query_hash,
                partition,
            } => {
                valid_sha256(canonical_query_hash)
                    && crate::routing_partition::artifact_from_routing_partition_v1(partition)
                        .is_ok()
            }
            Self::Request { input } => {
                input.schema == ROUTING_INSPECTION_INPUT_SCHEMA_V1
                    && valid_optional_id(Some(&input.pool_id))
                    && crate::routing_partition::artifact_from_routing_partition_v1(
                        &input.partition,
                    )
                    .is_ok()
            }
        };
        if valid {
            Ok(())
        } else {
            Err(InspectionError::InvalidArgument)
        }
    }
}

/// Point kind in a diagnostic projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DiagnosticPointKindV1 {
    /// Inspected query vector.
    Query,
    /// One returned evidence vector.
    Evidence,
}

/// One deterministic two-dimensional diagnostic point.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticPointV1 {
    /// Stable query or evidence record ID.
    pub record_id: String,
    /// Query or evidence classification.
    pub kind: DiagnosticPointKindV1,
    /// First diagnostic component coordinate.
    pub x: f64,
    /// Second diagnostic component coordinate.
    pub y: f64,
}

/// Optional deterministic query-local diagnostic projection.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DiagnosticProjectionV1 {
    /// Constant `pca_2`.
    pub algorithm: String,
    /// Constant algorithm version 1.
    pub algorithm_version: u32,
    /// Always true; coordinates have no routing authority.
    pub diagnostic_only: bool,
    /// Exact vector-space identity.
    pub vector_space_id: String,
    /// Query then evidence points in stable record order.
    pub points: Vec<DiagnosticPointV1>,
    /// Explained variance ratios for the two components.
    pub explained_variance_ratio: [f64; 2],
}

/// One query-local neighbor and its confidence facts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeighborhoodNeighborV1 {
    /// Evidence vector-link UUID.
    pub evidence_id: Uuid,
    /// Stable ordinal after distance/ID sorting.
    pub ordinal: u32,
    /// Cosine distance.
    pub distance: f64,
    /// Optional age at the report snapshot.
    pub age_seconds: Option<f64>,
    /// Optional similarity weight.
    pub similarity_weight: Option<f64>,
    /// Optional time-decay weight.
    pub time_weight: Option<f64>,
    /// Optional final confidence weight.
    pub final_weight: Option<f64>,
    /// Optional binary quality label.
    pub binary_label: Option<String>,
    /// Stable root-collapse inclusion reason.
    pub inclusion: String,
}

/// Confidence support reported for one neighborhood.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeighborhoodSupportV1 {
    /// Returned neighbor count.
    pub returned_neighbors: u32,
    /// Neighbors inside the configured radius.
    pub within_radius: u32,
    /// Distinct attempted roots.
    pub attempted_roots: u32,
    /// Selected labeled roots.
    pub selected_roots: u32,
    /// Optional labeled coverage.
    pub coverage: Option<f64>,
    /// Optional Kish effective sample size.
    pub effective_sample_size: Option<f64>,
}

/// Named confidence gates for one neighborhood.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeighborhoodGatesV1 {
    /// Whether an exact partition exists.
    pub partition: Option<bool>,
    /// Whether point support passes.
    pub points: Option<bool>,
    /// Whether independent-root support passes.
    pub roots: Option<bool>,
    /// Whether coverage passes.
    pub coverage: Option<bool>,
    /// Whether weight math passes.
    pub weight_math: Option<bool>,
    /// Whether effective-sample support passes.
    pub effective_samples: Option<bool>,
    /// Whether posterior inversion passes.
    pub beta_quantile: Option<bool>,
    /// Whether the lower-bound threshold passes.
    pub lower_bound: Option<bool>,
}

/// Recommendation and fallback result for inspected evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeighborhoodRecommendationV1 {
    /// Candidate ID when the neighborhood passes every gate.
    pub candidate_id: Option<String>,
    /// Stable recommendation/fallback reason.
    pub reason: String,
    /// Whether serving must use the anchor.
    pub anchor_fallback: bool,
}

/// Complete version-1 neighborhood inspection report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NeighborhoodReportV1 {
    /// Constant [`NEIGHBORHOOD_REPORT_SCHEMA_V1`].
    pub schema: String,
    /// Exact configured pool.
    pub pool_id: String,
    /// Canonical routing query hash.
    pub canonical_query_hash: String,
    /// Exact strict partition.
    pub partition: RoutingPartitionV1,
    /// Ordered query-local neighbors.
    pub neighbors: Vec<NeighborhoodNeighborV1>,
    /// Support and coverage facts.
    pub support: NeighborhoodSupportV1,
    /// Optional posterior credible lower bound.
    pub credible_lower_bound: Option<f64>,
    /// Named confidence gate results.
    pub gates: NeighborhoodGatesV1,
    /// Recommendation or anchor fallback.
    pub recommendation: NeighborhoodRecommendationV1,
    /// Optional deterministic diagnostic-only projection.
    pub projection: Option<DiagnosticProjectionV1>,
    /// UTC report snapshot time.
    pub snapshot_time_unix_ms: u64,
}

/// Streaming evidence export format.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceExportFormatV1 {
    /// One versioned JSON object per line.
    #[default]
    Jsonl,
    /// Stable scalar summary columns.
    Csv,
}

/// Request for a bounded streaming evidence export.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EvidenceExportRequestV1 {
    /// Evidence filters.
    pub filter: EvidenceFilterV1,
    /// Streaming format.
    pub format: EvidenceExportFormatV1,
}

impl EvidenceExportRequestV1 {
    /// Validate export filters before starting a producer task.
    pub fn validate(&self) -> Result<(), InspectionError> {
        self.filter.validate()
    }
}

/// One versioned JSON Lines evidence export record.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceExportRecordV1 {
    /// Constant [`EVIDENCE_EXPORT_RECORD_SCHEMA_V1`].
    pub schema: String,
    /// Complete verified, secret-filtered evidence detail.
    pub evidence: EvidenceDetailV1,
}

/// Scope and expected generations for one atomic learning reset.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LearningResetScopeV1 {
    /// Reset one configured pool.
    Pool {
        /// Exact pool ID.
        pool_id: String,
        /// Current learning generation observed by the caller.
        expected_learning_generation_id: Uuid,
    },
    /// Reset every configured pool in one transaction.
    All {
        /// Exact lexical map observed by the caller.
        expected_learning_generation_ids: BTreeMap<String, Uuid>,
    },
}

/// Idempotent compare-and-swap learning reset request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearningResetRequestV1 {
    /// Caller-generated UUIDv7 idempotency key and audit ID.
    pub mutation_id: Uuid,
    /// Pool or all-pool reset authority.
    pub scope: LearningResetScopeV1,
    /// Exact non-secret project confirmation ID.
    pub confirm_project_id: String,
    /// Bounded nonblank operator identity.
    pub actor: String,
    /// Bounded nonblank reason.
    pub reason: String,
}

/// Idempotent compare-and-swap cohort rotation request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CohortRotationRequestV1 {
    /// Caller-generated UUIDv7 idempotency key and audit ID.
    pub mutation_id: Uuid,
    /// Current cohort generation observed by the caller.
    pub expected_cohort_generation_id: Uuid,
    /// Exact non-secret project confirmation ID.
    pub confirm_project_id: String,
    /// Bounded nonblank operator identity.
    pub actor: String,
    /// Bounded nonblank reason.
    pub reason: String,
}

/// Stable result class for reset and cohort mutation receipts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OperatorMutationResultV1 {
    /// The requested generations advanced atomically.
    Applied,
    /// One or more expected generations were stale.
    Conflict,
}

/// Durable result of one learning reset or cohort rotation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperatorMutationReceiptV1 {
    /// Caller-generated audit ID.
    pub mutation_id: Uuid,
    /// Applied or conflict.
    pub result: OperatorMutationResultV1,
    /// Prior generation IDs by lexical scope key.
    pub prior_generations: BTreeMap<String, Uuid>,
    /// Resulting generation IDs by lexical scope key.
    pub resulting_generations: BTreeMap<String, Uuid>,
    /// Writer-owned UTC receipt time.
    pub created_at_unix_ms: u64,
    /// Immutable receipt hash.
    pub record_hash: String,
}

/// Version-1 wrapper for an existing pause/force-anchor control request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InspectionControlRequestV1 {
    /// Caller-generated UUIDv7 idempotency key.
    pub mutation_id: Uuid,
    /// All or one-pool control scope.
    pub scope: ControlScope,
    /// Exact pause or force-anchor field update.
    pub operation: ControlOperation,
    /// Latest global control generation observed by the caller.
    pub expected_control_generation: u64,
    /// Bounded nonblank operator identity.
    pub actor: String,
    /// Bounded nonblank reason.
    pub reason: String,
}

fn valid_optional_id(value: Option<&str>) -> bool {
    value.is_none_or(|value| {
        if value.is_empty()
            || value.len() > 128
            || value.chars().any(char::is_control)
            || !value.nfc().eq(value.chars())
        {
            return false;
        }
        let mut characters = value.chars();
        characters.next().is_some_and(char::is_alphanumeric)
            && characters.all(|character| {
                character.is_alphanumeric() || matches!(character, '_' | '-' | '.' | ':')
            })
    })
}

pub(crate) fn validate_inspection_id(value: &str) -> Result<(), InspectionError> {
    if valid_optional_id(Some(value)) {
        Ok(())
    } else {
        Err(InspectionError::InvalidArgument)
    }
}

fn valid_optional_reason(value: Option<&str>) -> bool {
    value.is_none_or(|value| {
        !value.is_empty()
            && value.len() <= 128
            && value.bytes().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'_' | b'.')
            })
    })
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use serde_json::json;

    use super::*;

    fn uuid7(value: u128) -> Uuid {
        let mut bytes = value.to_be_bytes();
        bytes[6] = (bytes[6] & 0x0f) | 0x70;
        bytes[8] = (bytes[8] & 0x3f) | 0x80;
        Uuid::from_bytes(bytes)
    }

    fn hash(character: char) -> String {
        std::iter::repeat_n(character, 64).collect()
    }

    fn partition() -> RoutingPartitionV1 {
        RoutingPartitionV1 {
            tenant_policy_hash: hash('1'),
            agent_policy_hash: hash('2'),
            policy_version_id: hash('3'),
            learning_generation_id: uuid7(1),
            api_family: LlmApiFamily::OpenAIChatCompletions,
            transport_identity: "transport-v1".into(),
            anchor_model: "anchor-model".into(),
            anchor_revision: "anchor-r1".into(),
            candidate_id: "candidate-a".into(),
            candidate_model: "candidate-model".into(),
            candidate_model_revision: "candidate-r1".into(),
            decoding_fingerprint: hash('4'),
            evaluator_version: hash('5'),
            vector_space_id: hash('6'),
        }
    }

    #[test]
    fn page_and_service_options_enforce_declared_bounds() {
        assert_eq!(PageRequest::default().limit, 100);
        assert_eq!(PageRequest::default().validate(), Ok(()));
        for limit in [0, INSPECTION_PAGE_LIMIT_MAX + 1] {
            assert_eq!(
                PageRequest { limit, after: None }.validate(),
                Err(InspectionError::InvalidArgument)
            );
        }
        assert_eq!(
            PageRequest {
                limit: 1,
                after: Some("x".repeat(INSPECTION_CURSOR_MAX_BYTES + 1)),
            }
            .validate(),
            Err(InspectionError::InvalidArgument)
        );

        let mut options = InspectionServiceOptions::default();
        assert!(!options.allow_operations);
        assert_eq!(options.validate(), Ok(()));
        options.request_timeout_ms = 0;
        assert_eq!(options.validate(), Err(InspectionError::InvalidArgument));
        options.request_timeout_ms = INSPECTION_REQUEST_TIMEOUT_MS_DEFAULT;
        options.export_chunk_bytes = INSPECTION_EXPORT_CHUNK_BYTES_MAX + 1;
        assert_eq!(options.validate(), Err(InspectionError::InvalidArgument));
    }

    #[test]
    fn inspection_error_codes_are_stable_and_unique() {
        let errors = [
            InspectionError::InvalidArgument,
            InspectionError::InvalidCursor,
            InspectionError::NotFound,
            InspectionError::Unauthorized,
            InspectionError::Forbidden,
            InspectionError::Conflict,
            InspectionError::Busy,
            InspectionError::MutationExpired,
            InspectionError::CapacityExhausted,
            InspectionError::StorageUnavailable,
            InspectionError::MigrationRequired,
            InspectionError::EgressDenied,
            InspectionError::IntegrityError,
            InspectionError::IncompatibleApi,
        ];
        let codes = errors
            .iter()
            .map(|error| error.code())
            .collect::<BTreeSet<_>>();
        assert_eq!(codes.len(), errors.len());
        for error in errors {
            assert_eq!(error.to_string(), error.code());
            assert_eq!(serde_json::to_value(error).unwrap(), json!(error.code()));
        }
    }

    #[test]
    fn lookup_union_is_exactly_tagged_and_boxes_large_partition() {
        let lookup = NeighborhoodLookupV1::QueryHash {
            canonical_query_hash: hash('a'),
            partition: Box::new(partition()),
        };
        let value = serde_json::to_value(&lookup).unwrap();
        assert_eq!(value["kind"], "query_hash");
        assert_eq!(value["canonical_query_hash"], hash('a'));
        assert_eq!(
            serde_json::from_value::<NeighborhoodLookupV1>(value.clone()).unwrap(),
            lookup
        );

        let mut with_extra = value;
        with_extra
            .as_object_mut()
            .unwrap()
            .insert("evidence_id".into(), json!(uuid7(9)));
        assert!(serde_json::from_value::<NeighborhoodLookupV1>(with_extra).is_err());
        assert!(
            serde_json::from_value::<NeighborhoodLookupV1>(json!({
                "canonical_query_hash": hash('a'),
                "partition": partition(),
            }))
            .is_err()
        );
    }

    #[test]
    fn neighborhood_lookup_validation_rejects_invalid_ids_hashes_and_partitions() {
        assert_eq!(
            NeighborhoodLookupV1::Evidence {
                evidence_id: uuid7(10),
            }
            .validate(),
            Ok(())
        );
        assert_eq!(
            NeighborhoodLookupV1::Evidence {
                evidence_id: Uuid::nil(),
            }
            .validate(),
            Err(InspectionError::InvalidArgument)
        );
        assert_eq!(
            NeighborhoodLookupV1::QueryHash {
                canonical_query_hash: hash('a'),
                partition: Box::new(partition()),
            }
            .validate(),
            Ok(())
        );
        assert_eq!(
            NeighborhoodLookupV1::QueryHash {
                canonical_query_hash: hash('A'),
                partition: Box::new(partition()),
            }
            .validate(),
            Err(InspectionError::InvalidArgument)
        );
        let mut invalid_partition = partition();
        invalid_partition.learning_generation_id = Uuid::nil();
        assert_eq!(
            NeighborhoodLookupV1::QueryHash {
                canonical_query_hash: hash('a'),
                partition: Box::new(invalid_partition),
            }
            .validate(),
            Err(InspectionError::InvalidArgument)
        );
    }

    #[test]
    fn read_and_export_filters_reject_invalid_values_before_storage() {
        assert_eq!(EvidenceFilterV1::default().validate(), Ok(()));
        assert_eq!(DecisionFilterV1::default().validate(), Ok(()));
        assert_eq!(OutcomeFilterV1::default().validate(), Ok(()));
        assert_eq!(EvidenceExportRequestV1::default().validate(), Ok(()));

        for terminal_class in [
            "completed",
            "deterministic_failure",
            "operational_failure",
            "skipped_cooloff",
            "canceled_shutdown",
            "orphaned_before_schedule",
            "orphaned_in_flight",
        ] {
            assert_eq!(
                EvidenceFilterV1 {
                    terminal_class: Some(terminal_class.into()),
                    ..EvidenceFilterV1::default()
                }
                .validate(),
                Ok(())
            );
        }
        for arm in [
            "candidate_treatment",
            "anchor_control",
            "anchor_holdout",
            "non_learning",
        ] {
            assert_eq!(
                OutcomeFilterV1 {
                    arm: Some(arm.into()),
                    ..OutcomeFilterV1::default()
                }
                .validate(),
                Ok(())
            );
        }

        let invalid_evidence = [
            EvidenceFilterV1 {
                pool_id: Some(String::new()),
                ..EvidenceFilterV1::default()
            },
            EvidenceFilterV1 {
                candidate_id: Some("bad/id".into()),
                ..EvidenceFilterV1::default()
            },
            EvidenceFilterV1 {
                terminal_class: Some("unknown".into()),
                ..EvidenceFilterV1::default()
            },
            EvidenceFilterV1 {
                quality_label: Some("maybe".into()),
                ..EvidenceFilterV1::default()
            },
            EvidenceFilterV1 {
                learning_generation_id: Some(Uuid::nil()),
                ..EvidenceFilterV1::default()
            },
        ];
        for filter in invalid_evidence {
            assert_eq!(filter.validate(), Err(InspectionError::InvalidArgument));
            assert_eq!(
                EvidenceExportRequestV1 {
                    filter,
                    format: EvidenceExportFormatV1::Jsonl,
                }
                .validate(),
                Err(InspectionError::InvalidArgument)
            );
        }

        for filter in [
            DecisionFilterV1 {
                mode: Some("shadow".into()),
                ..DecisionFilterV1::default()
            },
            DecisionFilterV1 {
                final_reason: Some("Bad-Reason".into()),
                ..DecisionFilterV1::default()
            },
        ] {
            assert_eq!(filter.validate(), Err(InspectionError::InvalidArgument));
        }
        for filter in [
            OutcomeFilterV1 {
                arm: Some("experimental".into()),
                ..OutcomeFilterV1::default()
            },
            OutcomeFilterV1 {
                label: Some("unknown".into()),
                ..OutcomeFilterV1::default()
            },
            OutcomeFilterV1 {
                attribution_status: Some("eligible".into()),
                ..OutcomeFilterV1::default()
            },
        ] {
            assert_eq!(filter.validate(), Err(InspectionError::InvalidArgument));
        }
        assert!(serde_json::from_value::<EvidenceFilterV1>(json!({ "unexpected": true })).is_err());
    }
}
