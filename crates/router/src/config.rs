// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Canonical version-1 Router configuration.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path};

use nemo_relay::api::runtime::LLM_REPLAY_CONTRACT_VERSION;
use nemo_relay::plugin::{ConfigDiagnostic, ConfigPolicy};
use nemo_relay_types::Json;
use nemo_relay_types::api::llm::LlmApiFamily;
use nemo_relay_types::api::scope::ScopeType;
use serde::de::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::json;
use tokio::sync::Semaphore;
use unicode_normalization::UnicodeNormalization;

use crate::canonical_json::{canonical_json, canonical_sha256};
use crate::diagnostics::{
    DUPLICATE_ID, INVALID_PATH, INVALID_PLUGIN_CONFIG, INVALID_RANGE, INVALID_REFERENCE,
    OVERLAPPING_POOL, UNKNOWN_FIELD, UNSAFE_EMBEDDER_ENDPOINT, UNSUPPORTED_CONFIG_VERSION,
    UNSUPPORTED_MODE, error, push_policy_diagnostic,
};
use crate::embedding_identity::{
    canonicalizer_version, embedder_profile_version, normalize_embedder_endpoint,
    propose_vector_space_registry,
};
use crate::fingerprint::sha256_hex;
use crate::projection::{
    REQUEST_PROJECTION_SCHEMA_V1, ROUTER_SANITIZER_VERSION, ROUTING_CONTEXT_SCHEMA_V1,
};
use crate::selector::{selectors_overlap, validate_selector};
use crate::trajectory::{
    CANDIDATE_FACT_SCHEMA_V1, PENDING_TRAJECTORY_SCHEMA_V1, REPLAY_CAPABILITY_SCHEMA_V1,
    RESPONSE_PROJECTION_SCHEMA_V1, TRAJECTORY_SANITIZER_VERSION,
};

/// Maximum UTF-8 byte length of a durable Router identifier.
pub const ID_MAX_BYTES: usize = 128;
/// Maximum UTF-8 byte length of a candidate identifier.
pub const CANDIDATE_ID_MAX_BYTES: usize = ID_MAX_BYTES;
/// Maximum UTF-8 byte length of an optional project identifier.
pub const PROJECT_ID_MAX_BYTES: usize = 128;
/// Maximum UTF-8 byte length of a provider model or endpoint identifier.
pub const MODEL_ID_MAX_BYTES: usize = 512;
/// Maximum UTF-8 byte length of an operator-pinned revision.
pub const REVISION_MAX_BYTES: usize = 128;
/// Maximum UTF-8 byte length of a normalized selector identity.
pub const SELECTOR_IDENTITY_MAX_BYTES: usize = 256;
/// Maximum UTF-8 byte length of one scope-path pattern.
pub const PATH_PATTERN_MAX_BYTES: usize = 1024;
/// Maximum supported value for each semaphore-backed concurrency field.
pub const CONCURRENCY_MAX_PERMITS: usize = Semaphore::MAX_PERMITS;
/// Version-1 canonical pairwise judge prompt identifier.
pub const JUDGE_PROMPT_VERSION_V1: &str = "pairwise-equivalence-v1";
/// Version-1 canonical response-and-trajectory rubric identifier.
pub const JUDGE_RUBRIC_VERSION_V1: &str = "response-trajectory-equivalence-v1";
/// SHA-256 of the canonical version-1 pairwise prompt template bytes.
pub const JUDGE_PROMPT_TEMPLATE_SHA256_V1: &str =
    "51a96ca6611ab63441084dbf371714fa1fb9305438976ae41b3f2f18bdab4eab";
/// SHA-256 of the canonical version-1 response-and-trajectory rubric bytes.
pub const JUDGE_RUBRIC_TEMPLATE_SHA256_V1: &str =
    "99d3cf4fd51086b57d9aa40690cc77b2cda1fc6926e2f4e76f76095037ef842f";
/// SHA-256 of the canonical version-1 structured judge output schema bytes.
pub const JUDGE_OUTPUT_SCHEMA_SHA256_V1: &str =
    "bcb012992c8805b6ea4d61f77f169434be5b43435934e0a8ee43881df1aed4a2";
/// Maximum UTF-8 bytes permitted in a version-1 judge rationale.
pub const JUDGE_MAX_RATIONALE_BYTES: usize = 16_384;
/// Maximum configured batch or candidate scheduler slots.
pub const SCHEDULER_MAX_SLOTS: usize = 65_536;
/// Maximum retained evidence rows accepted by version 1.
pub const EVIDENCE_RECORDS_MAX: u64 = 1_000_000;
/// Maximum embedding dimensions accepted by version 1.
pub const EMBEDDING_DIMENSIONS_MAX: u32 = 8_192;
/// Maximum embedding provider timeout accepted by version 1.
pub const EMBEDDER_TIMEOUT_MS_MAX: u64 = 55_000;
/// Maximum in-flight requests for one embedding profile.
pub const EMBEDDER_MAX_IN_FLIGHT: usize = 256;
/// Maximum inputs in one embedding provider request.
pub const EMBEDDER_BATCH_SIZE_MAX: usize = 128;
/// Maximum canonical JSON bytes in one embedding provider request.
pub const EMBEDDER_REQUEST_BYTES_MAX: usize = 32 * 1024 * 1024;
/// Maximum derived response bytes accepted by the profile validator.
pub const EMBEDDER_RESPONSE_BYTES_MAX: usize = 64 * 1024 * 1024;
/// Maximum nearest neighbors considered per candidate by learning policy version 1.
pub const LEARNING_TOP_K_MAX: usize = 4_095;
/// Maximum configured candidates in a complete learning policy.
pub const LEARNING_CANDIDATES_MAX: usize = 64;
/// Maximum candidate-neighbor rows admitted by one recommendation.
pub const LEARNING_CANDIDATE_NEIGHBOR_PRODUCT_MAX: usize = 4_095;
/// Maximum matchers in one success or failure outcome list.
pub const OUTCOME_MATCHERS_MAX: usize = 64;
/// Maximum exact metadata clauses in one outcome matcher.
pub const OUTCOME_METADATA_EQUALS_MAX: usize = 16;
/// Maximum UTF-8 byte length of an outcome category or name.
pub const OUTCOME_MATCHER_TEXT_MAX_BYTES: usize = 256;
/// Maximum byte length of one non-secret outcome string label.
pub const OUTCOME_LABEL_MAX_BYTES: usize = 64;
/// Maximum canonical bytes in one protected outcome-policy identity.
pub const OUTCOME_POLICY_IDENTITY_MAX_BYTES: usize = 1024 * 1024;
/// Maximum duration accepted by the version-1 outcome policy.
pub const OUTCOME_DURATION_SECONDS_MAX: u64 = 31_536_000;
/// Maximum randomized non-holdout roots in one version-1 experiment.
pub const OUTCOME_MAX_CANARY_ROOTS_MAX: u64 = 65_536;
/// Maximum non-holdout roots in one version-1 evaluation batch.
pub const OUTCOME_EVALUATION_BATCH_SIZE_MAX: u64 = 2_048;
/// Maximum fixed looks in one version-1 experiment.
pub const OUTCOME_MAX_LOOKS: u64 = 256;
const EMBEDDER_RESPONSE_BASE_BYTES: usize = 1024 * 1024;
const EMBEDDER_RESPONSE_COMPONENT_BYTES_MAX: usize = 32;
pub(crate) const EMBEDDER_AGGREGATE_WORK_ITEMS_MAX: usize = 65_536;
const WRITER_MIN_COMMAND_CAPACITY: usize = 256;
const WRITER_MAX_COMMAND_CAPACITY: usize = 65_536;

const ROUTER_CONFIG_VERSION: u32 = 1;
const CANONICALIZER_VERSION: u32 = 1;
const JUDGE_CONFIG_VERSION: u32 = 1;
const LEARNING_CONFIG_VERSION: u32 = 1;
const OUTCOME_CONFIG_VERSION: u32 = 1;
pub(crate) const JUDGE_OUTPUT_SCHEMA_VERSION: u32 = 1;
const JUDGE_WEIGHT_SUM_TOLERANCE: f64 = 1e-9;
const JUDGE_OUTPUT_TOKEN_BASE: usize = 512;

pub(crate) const LEARNING_LINEAR_RADIUS_ID_V1: &str = "linear_radius_v1";
pub(crate) const LEARNING_ROOT_SELECTOR_ID_V1: &str = "min_distance_newest_evaluation_lex_id_v1";
pub(crate) const LEARNING_COVERAGE_ID_V1: &str = "labeled_roots_over_attempted_roots_v1";
pub(crate) const LEARNING_TIME_DECAY_ID_V1: &str = "exp2_half_life_v1";
pub(crate) const LEARNING_FUTURE_SKEW_ID_V1: &str = "future_skew_300000ms_v1";
pub(crate) const LEARNING_SUMMATION_ID_V1: &str = "neumaier_f64_no_fma_v1";
pub(crate) const LEARNING_EFFECTIVE_SAMPLE_ID_V1: &str = "kish_effective_sample_v1";
pub(crate) const LEARNING_BETA_APPROXIMATION_ID_V1: &str = "beta_effective_sample_v1";
pub(crate) const LEARNING_CANDIDATE_CORRECTION_ID_V1: &str = "bonferroni_v1";
pub(crate) const LEARNING_BETA_INVERSE_CDF_ID_V1: &str = env!("STATRS_BETA_IMPLEMENTATION_ID");
pub(crate) const LEARNING_CANDIDATE_SET_ID_V1: &str = "candidate_set_v1";
pub(crate) const ACTIVE_COHORT_ASSIGNMENT_ID_V1: &str = "cohort_assignment_v1";
pub(crate) const ACTIVE_COHORT_HMAC_ID_V1: &str = "hmac-sha256-v1";
pub(crate) const ACTIVE_THRESHOLD_ID_V1: &str = "binary64_floor_u64_threshold_v1";
pub(crate) const OUTCOME_REDUCER_ID_V1: &str = "root_outcome_reducer_v1";
pub(crate) const OUTCOME_DECAY_ID_V1: &str = env!("LIBM_EXP2_IMPLEMENTATION_ID");
pub(crate) const OUTCOME_QUANTILE_ID_V1: &str = "monotone_beta_quantile_bounds_isotonic_1e-11_v1";
pub(crate) const OUTCOME_BETA_CDF_ID_V1: &str = env!("STATRS_BETA_CDF_IMPLEMENTATION_ID");
pub(crate) const OUTCOME_POSTERIOR_RESOLUTION_ID_V1: &str = "posterior_resolution_4096_4e-9_v1";
pub(crate) const OUTCOME_BONFERRONI_ID_V1: &str = "bonferroni_max_looks_v1";
pub(crate) const OUTCOME_ATTRIBUTION_GATE_ID_V1: &str = "outcome_attribution_gate_v1";
pub(crate) const OUTCOME_NONINFERIORITY_MATH_ID_V1: &str = "noninferiority_math_v1";

const OUTCOME_METADATA_KEY_HASH_DOMAIN_V1: &[u8] = b"nemo-relay-router/outcome-metadata-key/v1\0";
const OUTCOME_EXPECTED_SCALAR_HASH_DOMAIN_V1: &[u8] =
    b"nemo-relay-router/outcome-expected-scalar/v1\0";
#[allow(dead_code)] // Consumed by the sanitized signal projection added in Task 4.
pub(crate) const OUTCOME_OBSERVED_SCALAR_HASH_DOMAIN_V1: &[u8] =
    b"nemo-relay-router/outcome-observed-scalar/v1\0";
const OUTCOME_MATCHER_HASH_DOMAIN_V1: &[u8] = b"nemo-relay-router/outcome-matcher/v1\0";
const OUTCOME_POLICY_HASH_DOMAIN_V1: &[u8] = b"nemo-relay-router/outcome-policy/v1\0";
const OUTCOME_POSTERIOR_RESOLUTION_V1: f64 =
    crate::active_math::ACTIVE_MATH_POSTERIOR_RESOLUTION_V1;
const MAX_INTEROPERABLE_JSON_INTEGER: u64 = (1_u64 << 53) - 1;

// These immutable bytes are the Task 2 identity boundary. Task 3 consumes the
// same bytes when it builds the full prompt registry and request DTOs.
pub(crate) const JUDGE_PROMPT_TEMPLATE_V1: &[u8] = b"Compare one candidate response with the served anchor response using the supplied future-local trajectory and contracts. Return only the version-1 structured judge result.\n";
pub(crate) const JUDGE_RUBRIC_TEMPLATE_V1: &[u8] = b"Score response equivalence, trajectory equivalence, and judge confidence in the inclusive range [0,1]. Apply declared hard failures before numeric gates.\n";
pub(crate) const JUDGE_OUTPUT_SCHEMA_V1: &[u8] = br#"{"$id":"nemo.relay.router.pairwise-judge-result@1","$schema":"https://json-schema.org/draft/2020-12/schema","additionalProperties":false,"properties":{"hard_failures":{"items":{"enum":["tool_contract","response_schema","safety","malformed_candidate"],"type":"string"},"maxItems":4,"type":"array"},"judge_confidence":{"maximum":1,"minimum":0,"type":"number"},"rationale":{"maxLength":16384,"minLength":1,"type":"string"},"response_equivalence":{"maximum":1,"minimum":0,"type":"number"},"trajectory_equivalence":{"maximum":1,"minimum":0,"type":"number"}},"required":["response_equivalence","trajectory_equivalence","judge_confidence","hard_failures","rationale"],"title":"PairwiseJudgeResultV1","type":"object"}"#;

/// Router operating mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "lowercase")]
pub enum RouterMode {
    /// Disable all matching and routing work.
    #[default]
    Off,
    /// Evaluate eligible candidates without changing the served anchor result.
    Shadow,
    /// Produce recommendations without substituting the anchor result.
    Recommend,
    /// Permit controlled candidate substitution.
    Active,
}

/// Canonical version-1 Router component configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct RouterConfig {
    /// Configuration schema version. Only version `1` is supported.
    #[serde(default = "default_router_version")]
    pub version: u32,
    /// Router operating mode.
    #[serde(default)]
    pub mode: RouterMode,
    /// Optional stable deployment identifier.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub project_id: Option<String>,
    /// Local SQLite database path reserved for the persistence rollout.
    #[serde(default = "default_database_path")]
    pub database_path: String,
    /// Evidence retention period in days.
    #[serde(default = "default_retention_days")]
    pub retention_days: u32,
    /// Maximum number of retained evidence records.
    #[serde(default = "default_max_evidence_records")]
    pub max_evidence_records: u64,
    /// Whether remote HTTPS embedding endpoints are allowed.
    #[serde(default)]
    pub allow_remote_embedding_egress: bool,
    /// Named embedding provider profiles.
    #[serde(default)]
    pub embedders: Vec<EmbedderConfig>,
    /// Ordered routing-pool definitions.
    #[serde(default)]
    pub pools: Vec<PoolConfig>,
    /// Policy for unsupported configuration fields.
    #[serde(default)]
    pub policy: ConfigPolicy,
    /// Unrecognized root fields retained for policy-aware validation.
    #[doc(hidden)]
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown_fields: BTreeMap<String, Json>,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self {
            version: default_router_version(),
            mode: RouterMode::Off,
            project_id: None,
            database_path: default_database_path(),
            retention_days: default_retention_days(),
            max_evidence_records: default_max_evidence_records(),
            allow_remote_embedding_egress: false,
            embedders: Vec::new(),
            pools: Vec::new(),
            policy: ConfigPolicy::default(),
            unknown_fields: BTreeMap::new(),
        }
    }
}

impl PartialEq for RouterConfig {
    fn eq(&self, other: &Self) -> bool {
        self.version == other.version
            && self.mode == other.mode
            && self.project_id == other.project_id
            && self.database_path == other.database_path
            && self.retention_days == other.retention_days
            && self.max_evidence_records == other.max_evidence_records
            && self.allow_remote_embedding_egress == other.allow_remote_embedding_egress
            && self.embedders == other.embedders
            && self.pools == other.pools
            && self.policy.unknown_component == other.policy.unknown_component
            && self.policy.unknown_field == other.policy.unknown_field
            && self.policy.unsupported_value == other.policy.unsupported_value
            && self.unknown_fields == other.unknown_fields
    }
}

impl RouterConfig {
    /// Validates this typed configuration without external side effects.
    pub fn validate(&self) -> Vec<ConfigDiagnostic> {
        self.validate_diagnostics()
    }

    /// Returns the canonical generation identity for a valid configuration.
    ///
    /// The hash is independent of ID-keyed collection order and selector-set
    /// order. Unknown fields and external credential values are excluded.
    pub fn config_generation_id(&self) -> Result<String, Vec<ConfigDiagnostic>> {
        let diagnostics = self.validate_diagnostics();
        if diagnostics
            .iter()
            .any(|diagnostic| diagnostic.level == nemo_relay::plugin::DiagnosticLevel::Error)
        {
            return Err(diagnostics);
        }
        self.generation_id().map_err(|message| {
            vec![error(
                crate::diagnostics::INVALID_PLUGIN_CONFIG,
                None,
                message,
            )]
        })
    }

    pub(crate) fn validate_diagnostics(&self) -> Vec<ConfigDiagnostic> {
        let mut diagnostics = Vec::new();
        validate_unknown_fields(&mut diagnostics, &self.policy, &self.unknown_fields, "");

        if self.version != ROUTER_CONFIG_VERSION {
            diagnostics.push(error(
                UNSUPPORTED_CONFIG_VERSION,
                Some("version".to_string()),
                format!("Router config version {} is not supported", self.version),
            ));
        }
        if let Some(project_id) = &self.project_id {
            validate_stable_id(
                project_id,
                PROJECT_ID_MAX_BYTES,
                "project_id",
                &mut diagnostics,
            );
        }
        validate_database_path(&self.database_path, &mut diagnostics);
        validate_positive_u64(
            u64::from(self.retention_days),
            "retention_days",
            &mut diagnostics,
        );
        validate_positive_u64(
            self.max_evidence_records,
            "max_evidence_records",
            &mut diagnostics,
        );
        validate_max_u64(
            self.max_evidence_records,
            EVIDENCE_RECORDS_MAX,
            "max_evidence_records",
            &mut diagnostics,
        );

        let mut embedder_ids = BTreeMap::new();
        for (index, embedder) in self.embedders.iter().enumerate() {
            let prefix = format!("embedders[{index}]");
            validate_unknown_fields(
                &mut diagnostics,
                &self.policy,
                &embedder.unknown_fields,
                &prefix,
            );
            validate_stable_id(
                &embedder.id,
                ID_MAX_BYTES,
                &format!("{prefix}.id"),
                &mut diagnostics,
            );
            if let Some(previous) = embedder_ids.insert(embedder.id.as_str(), index) {
                diagnostics.push(error(
                    DUPLICATE_ID,
                    Some(format!("{prefix}.id")),
                    format!(
                        "embedder ID '{}' duplicates embedders[{previous}].id",
                        embedder.id
                    ),
                ));
            }
            validate_provider_identifier(
                &embedder.model,
                &format!("{prefix}.model"),
                &mut diagnostics,
            );
            validate_revision(
                &embedder.provider_revision,
                &format!("{prefix}.provider_revision"),
                &mut diagnostics,
            );
            validate_positive_u64(
                u64::from(embedder.dimensions),
                &format!("{prefix}.dimensions"),
                &mut diagnostics,
            );
            validate_max_u64(
                u64::from(embedder.dimensions),
                u64::from(EMBEDDING_DIMENSIONS_MAX),
                &format!("{prefix}.dimensions"),
                &mut diagnostics,
            );
            validate_positive_u64(
                embedder.timeout_ms,
                &format!("{prefix}.timeout_ms"),
                &mut diagnostics,
            );
            validate_max_u64(
                embedder.timeout_ms,
                EMBEDDER_TIMEOUT_MS_MAX,
                &format!("{prefix}.timeout_ms"),
                &mut diagnostics,
            );
            validate_positive_u64(
                embedder.max_in_flight as u64,
                &format!("{prefix}.max_in_flight"),
                &mut diagnostics,
            );
            validate_max_u64(
                embedder.max_in_flight as u64,
                EMBEDDER_MAX_IN_FLIGHT as u64,
                &format!("{prefix}.max_in_flight"),
                &mut diagnostics,
            );
            validate_positive_u64(
                embedder.batch_size as u64,
                &format!("{prefix}.batch_size"),
                &mut diagnostics,
            );
            validate_max_u64(
                embedder.batch_size as u64,
                EMBEDDER_BATCH_SIZE_MAX as u64,
                &format!("{prefix}.batch_size"),
                &mut diagnostics,
            );
            validate_embedder_response_bound(embedder, &prefix, &mut diagnostics);
            if let Some(variable) = &embedder.api_key_env {
                validate_environment_variable(
                    variable,
                    &format!("{prefix}.api_key_env"),
                    &mut diagnostics,
                );
            }
            validate_embedder_endpoint(
                &embedder.base_url,
                self.allow_remote_embedding_egress,
                &format!("{prefix}.base_url"),
                &mut diagnostics,
            );
        }

        let mut pool_ids = BTreeMap::new();
        for (index, pool) in self.pools.iter().enumerate() {
            let prefix = format!("pools[{index}]");
            validate_pool(
                pool,
                index,
                &self.policy,
                self.retention_days,
                self.max_evidence_records,
                &mut diagnostics,
            );
            if self.mode == RouterMode::Recommend
                && pool
                    .learning
                    .as_ref()
                    .and_then(LearningConfig::complete_policy)
                    .is_none()
            {
                diagnostics.push(error(
                    UNSUPPORTED_MODE,
                    Some(format!("{prefix}.learning")),
                    "recommend mode requires a complete version-1 learning policy for every pool",
                ));
            }
            if self.mode == RouterMode::Active {
                if pool
                    .learning
                    .as_ref()
                    .and_then(LearningConfig::complete_active_policy)
                    .is_none()
                {
                    diagnostics.push(error(
                        UNSUPPORTED_MODE,
                        Some(format!("{prefix}.learning")),
                        "active mode requires a complete version-1 Active learning policy for every pool",
                    ));
                }
                if pool.outcome.is_empty() {
                    diagnostics.push(error(
                        UNSUPPORTED_MODE,
                        Some(format!("{prefix}.outcome")),
                        "active mode requires a complete version-1 outcome policy for every pool",
                    ));
                }
            }
            if let Some(learning) = &pool.learning {
                validate_stable_id(
                    &learning.embedder,
                    ID_MAX_BYTES,
                    &format!("{prefix}.learning.embedder"),
                    &mut diagnostics,
                );
                if !embedder_ids.contains_key(learning.embedder.as_str()) {
                    diagnostics.push(error(
                        INVALID_REFERENCE,
                        Some(format!("{prefix}.learning.embedder")),
                        format!(
                            "learning embedder '{}' does not reference a configured profile",
                            learning.embedder
                        ),
                    ));
                }
            }
            if let Some(previous) = pool_ids.insert(pool.id.as_str(), index) {
                diagnostics.push(error(
                    DUPLICATE_ID,
                    Some(format!("{prefix}.id")),
                    format!("pool ID '{}' duplicates pools[{previous}].id", pool.id),
                ));
            }
        }
        validate_scheduler_slots(&self.pools, &mut diagnostics);

        for left_index in 0..self.pools.len() {
            for right_index in (left_index + 1)..self.pools.len() {
                let left = &self.pools[left_index];
                let right = &self.pools[right_index];
                if left.api_family == right.api_family
                    && left
                        .anchor_models
                        .iter()
                        .any(|model| right.anchor_models.contains(model))
                    && selectors_overlap(&left.selector, &right.selector)
                {
                    diagnostics.push(error(
                        OVERLAPPING_POOL,
                        Some(format!("pools[{right_index}].selector")),
                        format!(
                            "pool '{}' overlaps pools[{left_index}] ('{}')",
                            right.id, left.id
                        ),
                    ));
                }
            }
        }

        if !diagnostics
            .iter()
            .any(|diagnostic| diagnostic.level == nemo_relay::plugin::DiagnosticLevel::Error)
            && let Err(message) = propose_vector_space_registry(self)
        {
            diagnostics.push(error(INVALID_RANGE, Some("embedders".to_string()), message));
        }

        diagnostics
    }

    pub(crate) fn generation_id(&self) -> Result<String, String> {
        canonical_sha256(&self.generation_value()?)
    }

    pub(crate) fn generation_value(&self) -> Result<Json, String> {
        let mut embedders = self.embedders.iter().collect::<Vec<_>>();
        embedders.sort_by(|left, right| left.id.cmp(&right.id));
        let mut pools = self.pools.iter().collect::<Vec<_>>();
        pools.sort_by(|left, right| left.id.cmp(&right.id));
        let pool_values = pools
            .into_iter()
            .map(pool_generation_value)
            .collect::<Result<Vec<_>, _>>()?;
        let scheduler_policy = scheduler_policy_generation_value(&self.pools)?;

        let embedder_values = embedders
            .into_iter()
            .map(|config| embedder_generation_value(config, self.allow_remote_embedding_egress))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(json!({
            "schema": "nemo.relay.router.config-generation@1",
            "version": self.version,
            "mode": self.mode,
            "project_id": self.project_id,
            "database_path_sha256": protected_string_sha256(
                "router-database-path-v1",
                &self.database_path,
            ),
            "retention_days": self.retention_days,
            "max_evidence_records": self.max_evidence_records,
            "allow_remote_embedding_egress": self.allow_remote_embedding_egress,
            "embedders": embedder_values,
            "pools": pool_values,
            "derived_scheduler_policy": scheduler_policy,
        }))
    }

    pub(crate) fn policy_generation_values(&self) -> Result<BTreeMap<String, Json>, String> {
        self.pools
            .iter()
            .map(|pool| {
                let pool_id = pool.id.clone();
                let value = json!({
                    "schema": "nemo.relay.router.policy@1",
                    "pool": pool_generation_value(pool)?,
                    "contracts": {
                        "llm_replay_contract_version": LLM_REPLAY_CONTRACT_VERSION,
                        "request_projection_schema": REQUEST_PROJECTION_SCHEMA_V1,
                        "routing_context_schema": ROUTING_CONTEXT_SCHEMA_V1,
                        "response_projection_schema": RESPONSE_PROJECTION_SCHEMA_V1,
                        "replay_capability_schema": REPLAY_CAPABILITY_SCHEMA_V1,
                        "candidate_fact_schema": CANDIDATE_FACT_SCHEMA_V1,
                        "pending_trajectory_schema": PENDING_TRAJECTORY_SCHEMA_V1,
                        "request_sanitizer_version": ROUTER_SANITIZER_VERSION,
                        "trajectory_sanitizer_version": TRAJECTORY_SANITIZER_VERSION,
                    },
                });
                Ok((pool_id, value))
            })
            .collect()
    }

    pub(crate) fn writer_command_capacity(&self) -> Result<usize, String> {
        let total_candidate_slots = self.pools.iter().try_fold(0usize, |total, pool| {
            let slots = pool_scheduler_slots(pool).ok_or_else(|| {
                format!(
                    "derived candidate slots overflow for Router pool '{}'",
                    pool.id
                )
            })?;
            total
                .checked_add(slots.candidates)
                .ok_or_else(|| "aggregate Router candidate slots overflow usize".to_string())
        })?;
        Ok(total_candidate_slots
            .checked_mul(4)
            .and_then(|scaled| scaled.checked_add(WRITER_MIN_COMMAND_CAPACITY))
            .ok_or_else(|| "Router writer command capacity overflow usize".to_string())?
            .clamp(WRITER_MIN_COMMAND_CAPACITY, WRITER_MAX_COMMAND_CAPACITY))
    }

    pub(crate) fn active_replay_capacity(&self) -> Result<usize, String> {
        let capacity = self.pools.iter().try_fold(0usize, |total, pool| {
            let slots = pool_scheduler_slots(pool).ok_or_else(|| {
                format!(
                    "derived candidate slots overflow for Router pool '{}'",
                    pool.id
                )
            })?;
            let provider_slots = pool
                .concurrency
                .shadow
                .saturating_add(pool.concurrency.judge);
            total
                .checked_add(slots.candidates.min(provider_slots))
                .ok_or_else(|| "aggregate Router active replay capacity overflow usize".to_string())
        })?;
        if capacity == 0 || capacity > SCHEDULER_MAX_SLOTS {
            return Err(format!(
                "aggregate Router active replay capacity {capacity} is outside 1..={SCHEDULER_MAX_SLOTS}"
            ));
        }
        Ok(capacity)
    }
}

/// One embedding provider profile.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct EmbedderConfig {
    /// Stable profile identifier.
    pub id: String,
    /// Provider base URL.
    pub base_url: String,
    /// Provider model identifier.
    pub model: String,
    /// Operator-pinned provider revision.
    pub provider_revision: String,
    /// Expected embedding vector dimensions.
    pub dimensions: u32,
    /// Optional environment-variable name containing an API key.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    /// Request timeout in milliseconds.
    pub timeout_ms: u64,
    /// Maximum provider calls in flight.
    #[serde(default = "default_embedder_max_in_flight")]
    pub max_in_flight: usize,
    /// Maximum inputs per provider batch.
    #[serde(default = "default_embedder_batch_size")]
    pub batch_size: usize,
    /// Unrecognized profile fields retained for policy-aware validation.
    #[doc(hidden)]
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown_fields: BTreeMap<String, Json>,
}

/// Version-1 association or complete policy for one routing pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
pub struct LearningConfig {
    /// Learning association contract version. Only version `1` is supported.
    pub version: u32,
    /// Stable ID of one configured embedding profile.
    pub embedder: String,
    /// Maximum nearest neighbors retrieved per candidate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_k: Option<usize>,
    /// Inclusive distance radius applied after top-K retrieval.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<f64>,
    /// Minimum labeled points inside the configured radius.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_points: Option<usize>,
    /// Minimum independent labeled roots inside the radius.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_independent_roots: Option<usize>,
    /// Minimum Kish effective sample size.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_effective_samples: Option<f64>,
    /// Minimum labeled-root coverage.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min_coverage: Option<f64>,
    /// Half-life in seconds for evidence time decay.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_decay_half_life_seconds: Option<f64>,
    /// Positive Beta prior success shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_success: Option<f64>,
    /// Positive Beta prior failure shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prior_failure: Option<f64>,
    /// Familywise credible level before candidate correction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub familywise_credible_level: Option<f64>,
    /// Minimum one-sided lower bound required for promotion.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub promotion_lower_bound: Option<f64>,
    /// Lower bound retained by an already-promoted Active neighborhood.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retention_lower_bound: Option<f64>,
    /// Unconditional probability of assigning an Active root to holdout.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub holdout_probability: Option<f64>,
    /// Unconditional probability of assigning an Active root to treatment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_canary_fraction: Option<f64>,
}

/// Validated nonoptional view of a complete version-1 learning policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CompleteLearningPolicyV1<'a> {
    pub(crate) version: u32,
    pub(crate) embedder: &'a str,
    pub(crate) top_k: usize,
    pub(crate) radius: f64,
    pub(crate) min_points: usize,
    pub(crate) min_independent_roots: usize,
    pub(crate) min_effective_samples: f64,
    pub(crate) min_coverage: f64,
    pub(crate) time_decay_half_life_seconds: f64,
    pub(crate) prior_success: f64,
    pub(crate) prior_failure: f64,
    pub(crate) familywise_credible_level: f64,
    pub(crate) promotion_lower_bound: f64,
}

/// Validated nonoptional view of a complete Active learning policy.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CompleteActiveLearningPolicyV1<'a> {
    pub(crate) recommend: CompleteLearningPolicyV1<'a>,
    pub(crate) retention_lower_bound: f64,
    pub(crate) holdout_probability: f64,
    pub(crate) active_canary_fraction: f64,
}

impl LearningConfig {
    #[cfg(test)]
    pub(crate) fn minimal(embedder: impl Into<String>) -> Self {
        Self {
            version: LEARNING_CONFIG_VERSION,
            embedder: embedder.into(),
            top_k: None,
            radius: None,
            min_points: None,
            min_independent_roots: None,
            min_effective_samples: None,
            min_coverage: None,
            time_decay_half_life_seconds: None,
            prior_success: None,
            prior_failure: None,
            familywise_credible_level: None,
            promotion_lower_bound: None,
            retention_lower_bound: None,
            holdout_probability: None,
            active_canary_fraction: None,
        }
    }

    pub(crate) fn complete_policy(&self) -> Option<CompleteLearningPolicyV1<'_>> {
        Some(CompleteLearningPolicyV1 {
            version: self.version,
            embedder: &self.embedder,
            top_k: self.top_k?,
            radius: self.radius?,
            min_points: self.min_points?,
            min_independent_roots: self.min_independent_roots?,
            min_effective_samples: self.min_effective_samples?,
            min_coverage: self.min_coverage?,
            time_decay_half_life_seconds: self.time_decay_half_life_seconds?,
            prior_success: self.prior_success?,
            prior_failure: self.prior_failure?,
            familywise_credible_level: self.familywise_credible_level?,
            promotion_lower_bound: self.promotion_lower_bound?,
        })
    }

    pub(crate) fn complete_active_policy(&self) -> Option<CompleteActiveLearningPolicyV1<'_>> {
        Some(CompleteActiveLearningPolicyV1 {
            recommend: self.complete_policy()?,
            retention_lower_bound: self.retention_lower_bound?,
            holdout_probability: self.holdout_probability?,
            active_canary_fraction: self.active_canary_fraction?,
        })
    }

    fn has_statistical_fields(&self) -> bool {
        self.top_k.is_some()
            || self.radius.is_some()
            || self.min_points.is_some()
            || self.min_independent_roots.is_some()
            || self.min_effective_samples.is_some()
            || self.min_coverage.is_some()
            || self.time_decay_half_life_seconds.is_some()
            || self.prior_success.is_some()
            || self.prior_failure.is_some()
            || self.familywise_credible_level.is_some()
            || self.promotion_lower_bound.is_some()
    }

    fn has_active_fields(&self) -> bool {
        self.retention_lower_bound.is_some()
            || self.holdout_probability.is_some()
            || self.active_canary_fraction.is_some()
    }
}

fn deserialize_learning_config<'de, D>(deserializer: D) -> Result<Option<LearningConfig>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Json::deserialize(deserializer)?;
    let object = value
        .as_object()
        .ok_or_else(|| D::Error::custom("learning must be an object"))?;
    if object.is_empty() {
        return Ok(None);
    }
    serde_json::from_value(value)
        .map(Some)
        .map_err(D::Error::custom)
}

fn serialize_learning_config<S>(
    config: &Option<LearningConfig>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    match config {
        Some(config) => config.serialize(serializer),
        None => Json::Object(Default::default()).serialize(serializer),
    }
}

#[cfg(feature = "schema")]
fn learning_config_field_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    use schemars::schema::{InstanceType, ObjectValidation, Schema, SchemaObject, SingleOrVec};

    let mut empty = SchemaObject {
        instance_type: Some(SingleOrVec::Single(Box::new(InstanceType::Object))),
        ..SchemaObject::default()
    };
    empty.object = Some(Box::new(ObjectValidation {
        max_properties: Some(0),
        additional_properties: Some(Box::new(Schema::Bool(false))),
        ..ObjectValidation::default()
    }));

    let Schema::Object(mut active) =
        <LearningConfig as schemars::JsonSchema>::json_schema(generator)
    else {
        unreachable!("LearningConfig must generate an object schema");
    };
    let mut minimal = active.clone();
    let minimal_object = minimal
        .object
        .as_mut()
        .expect("LearningConfig schema must have object validation");
    minimal_object
        .properties
        .retain(|field, _| matches!(field.as_str(), "version" | "embedder"));
    minimal_object
        .required
        .retain(|field| matches!(field.as_str(), "version" | "embedder"));

    let mut recommend = active.clone();
    let recommend_object = recommend
        .object
        .as_mut()
        .expect("LearningConfig schema must have object validation");
    recommend_object.properties.retain(|field, _| {
        !matches!(
            field.as_str(),
            "retention_lower_bound" | "holdout_probability" | "active_canary_fraction"
        )
    });
    for (field, schema) in &mut recommend_object.properties {
        if !matches!(field.as_str(), "version" | "embedder") {
            remove_schema_null_type(schema);
        }
    }
    recommend_object.required = recommend_object.properties.keys().cloned().collect();

    let active_object = active
        .object
        .as_mut()
        .expect("LearningConfig schema must have object validation");
    for (field, schema) in &mut active_object.properties {
        if !matches!(field.as_str(), "version" | "embedder") {
            remove_schema_null_type(schema);
        }
    }
    active_object.required = active_object.properties.keys().cloned().collect();

    let mut union = SchemaObject::default();
    union.subschemas().one_of = Some(vec![
        empty.into(),
        minimal.into(),
        recommend.into(),
        active.into(),
    ]);
    union.into()
}

#[cfg(feature = "schema")]
fn remove_schema_null_type(schema: &mut schemars::schema::Schema) {
    use schemars::schema::{InstanceType, Schema, SingleOrVec};

    let Schema::Object(schema) = schema else {
        return;
    };
    if let Some(SingleOrVec::Vec(types)) = &mut schema.instance_type {
        types.retain(|instance| *instance != InstanceType::Null);
    }
}

/// Event shape matched by a version-1 outcome matcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum OutcomeMatcherEventKind {
    /// A Scope event whose phase is End.
    ScopeEnd,
    /// A Mark event.
    Mark,
}

/// Sanitized terminal status required by a version-1 outcome matcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum OutcomeTerminalStatus {
    /// Exact OpenTelemetry status `OK`.
    Ok,
    /// Exact OpenTelemetry status `ERROR`.
    Error,
    /// No OpenTelemetry status is present.
    Unset,
}

/// Terminal disposition contributed by an outcome signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum OutcomeDisposition {
    /// Contribute a successful outcome signal.
    Success,
    /// Contribute a failed outcome signal.
    Failure,
    /// Do not contribute a Bernoulli signal.
    Ignore,
}

/// One exact version-1 outcome event matcher.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
pub struct OutcomeMatcher {
    /// Event shape to match.
    pub event_kind: OutcomeMatcherEventKind,
    /// Exact sanitized event category.
    pub category: String,
    /// Exact sanitized event name.
    pub name: String,
    /// Required terminal status.
    pub terminal_status: OutcomeTerminalStatus,
    /// Exact allowlisted top-level metadata clauses.
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "outcome_metadata_equals_schema")
    )]
    pub metadata_equals: BTreeMap<String, Json>,
}

/// Complete version-1 actual-outcome policy for one routing pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(deny_unknown_fields)]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
pub struct OutcomeConfig {
    /// Outcome policy version. Only version `1` is supported.
    pub version: u32,
    /// Disjunctive matchers that contribute success signals.
    #[cfg_attr(feature = "schema", schemars(length(min = 1, max = 64)))]
    pub success_matchers: Vec<OutcomeMatcher>,
    /// Disjunctive matchers that contribute failure signals.
    #[cfg_attr(feature = "schema", schemars(length(min = 1, max = 64)))]
    pub failure_matchers: Vec<OutcomeMatcher>,
    /// Disposition of a successful representative Primary completion.
    pub completion_disposition: OutcomeDisposition,
    /// Disposition of a representative Primary error. Must be failure.
    pub error_disposition: OutcomeDisposition,
    /// Disposition of an external terminal Tool error. Must be failure.
    pub tool_failure_disposition: OutcomeDisposition,
    /// Disposition of the exact trajectory-owner end.
    pub end_of_run_disposition: OutcomeDisposition,
    /// Maximum attribution interval.
    pub max_attribution_seconds: u64,
    /// Half-life for randomized actual outcomes.
    pub actual_outcome_half_life_seconds: u64,
    /// Half-life for fresh anchor-shadow evidence used by Active.
    pub anchor_shadow_half_life_seconds: u64,
    /// Cooloff after a query-local Active failure.
    pub relearning_cooloff_seconds: u64,
    /// Minimum raw treatment roots required by one look.
    pub min_treatment_roots: u64,
    /// Minimum raw control roots required by one look.
    pub min_control_roots: u64,
    /// Minimum decayed treatment weight required by one look.
    pub min_treatment_effective_weight: f64,
    /// Minimum decayed control weight required by one look.
    pub min_control_effective_weight: f64,
    /// Accepted treatment loss relative to control.
    pub noninferiority_margin: f64,
    /// Required posterior noninferiority probability.
    pub noninferiority_probability: f64,
    /// Required posterior rollback probability.
    pub rollback_probability: f64,
    /// Non-holdout admission batch size.
    pub outcome_evaluation_batch_size: u64,
    /// Lifetime non-holdout Active cap.
    pub max_canary_roots: u64,
    /// Lifetime of one passed actual-outcome authorization.
    pub authorization_ttl_seconds: u64,
}

#[cfg(feature = "schema")]
fn outcome_metadata_equals_schema(
    _generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    let scalar = json!({
        "oneOf": [
            {"type": "null"},
            {"type": "boolean"},
            {"type": "number"},
            {"type": "string", "pattern": "^[a-z0-9_.-]{1,64}$"}
        ]
    });
    serde_json::from_value(json!({
        "type": "object",
        "maxProperties": OUTCOME_METADATA_EQUALS_MAX,
        "additionalProperties": false,
        "properties": {
            "error.type": scalar,
            "outcome": scalar,
            "outcome.label": scalar,
            "outcome.success": scalar,
            "result": scalar,
            "status": scalar,
            "success": scalar,
        }
    }))
    .expect("static outcome metadata schema must be valid")
}

#[cfg(feature = "schema")]
fn outcome_config_field_schema(
    generator: &mut schemars::r#gen::SchemaGenerator,
) -> schemars::schema::Schema {
    use schemars::schema::{InstanceType, ObjectValidation, Schema, SchemaObject, SingleOrVec};

    let mut empty = SchemaObject {
        instance_type: Some(SingleOrVec::Single(Box::new(InstanceType::Object))),
        ..SchemaObject::default()
    };
    empty.object = Some(Box::new(ObjectValidation {
        max_properties: Some(0),
        additional_properties: Some(Box::new(Schema::Bool(false))),
        ..ObjectValidation::default()
    }));

    let complete = <OutcomeConfig as schemars::JsonSchema>::json_schema(generator);
    let mut union = SchemaObject::default();
    union.subschemas().one_of = Some(vec![empty.into(), complete]);
    union.into()
}

/// One deterministic routing pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PoolConfig {
    /// Stable pool identifier.
    pub id: String,
    /// Authoritative provider API family.
    #[cfg_attr(feature = "schema", schemars(with = "String"))]
    pub api_family: LlmApiFamily,
    /// Anchor model identifiers served by this pool.
    pub anchor_models: Vec<String>,
    /// Operator-pinned anchor evidence revision.
    pub anchor_revision: String,
    /// Probability of proposing a shadow sample, in the inclusive range `[0, 1]`.
    pub sampling_probability: f64,
    /// Maximum eligible candidates proposed by one sample.
    pub max_candidates_per_sample: usize,
    /// Frozen-context selector predicates.
    #[serde(default)]
    pub selector: PoolSelectorConfig,
    /// Future-local lookahead limits.
    #[serde(default)]
    pub lookahead: LookaheadConfig,
    /// Independent shadow, judge, and pending concurrency limits.
    pub concurrency: ConcurrencyConfig,
    /// Candidate model definitions.
    pub candidates: Vec<CandidateConfig>,
    /// Semantic request canonicalization limits.
    #[serde(default)]
    pub canonicalizer: CanonicalizerConfig,
    /// Required versioned judge policy for this pool.
    #[cfg_attr(feature = "schema", schemars(description = ""))]
    pub judge: JudgeConfig,
    /// Optional versioned embedding association.
    #[serde(
        default,
        deserialize_with = "deserialize_learning_config",
        serialize_with = "serialize_learning_config"
    )]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "learning_config_field_schema")
    )]
    pub learning: Option<LearningConfig>,
    /// Outcome extension boundary reserved for the outcome rollout.
    #[serde(default)]
    #[cfg_attr(
        feature = "schema",
        schemars(schema_with = "outcome_config_field_schema")
    )]
    pub outcome: BTreeMap<String, Json>,
    /// Unrecognized pool fields retained for policy-aware validation.
    #[doc(hidden)]
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown_fields: BTreeMap<String, Json>,
}

/// Required version-1 judge policy for one routing pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[cfg_attr(feature = "schema", schemars(deny_unknown_fields))]
pub struct JudgeConfig {
    /// Judge configuration contract version. Only version `1` is supported.
    pub version: u32,
    /// Provider model identifier used for judge calls.
    pub model: String,
    /// Operator-pinned judge model revision.
    pub model_revision: String,
    /// Immutable prompt template identifier.
    pub prompt_version: String,
    /// Immutable scoring rubric identifier.
    pub rubric_version: String,
    /// Structured judge output schema version. Only version `1` is supported.
    pub output_schema_version: u32,
    /// Fixed provider sampling temperature. Omitted configurations retain `0.0`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Weight applied to response equivalence.
    pub response_weight: f64,
    /// Weight applied to trajectory equivalence.
    pub trajectory_weight: f64,
    /// Minimum response-equivalence score required to pass.
    pub response_floor: f64,
    /// Minimum trajectory-equivalence score required to pass.
    pub trajectory_floor: f64,
    /// Minimum judge confidence required to produce a quality label.
    pub judge_confidence_floor: f64,
    /// Minimum weighted aggregate score required to pass.
    pub pass_threshold: f64,
    /// Maximum UTF-8 bytes retained for the judge rationale.
    pub max_rationale_bytes: usize,
    /// Initial dependency cool-off duration in seconds.
    pub base_cooloff_seconds: u64,
    /// Maximum dependency cool-off duration in seconds.
    pub max_cooloff_seconds: u64,
    /// Unrecognized judge fields retained for strict indexed validation.
    #[doc(hidden)]
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    #[cfg_attr(feature = "schema", schemars(skip))]
    pub unknown_fields: BTreeMap<String, Json>,
}

impl JudgeConfig {
    /// Returns the version-1 provider output-token bound using checked arithmetic.
    pub fn output_token_limit(&self) -> Option<usize> {
        self.max_rationale_bytes
            .checked_add(3)?
            .checked_div(4)?
            .checked_add(JUDGE_OUTPUT_TOKEN_BASE)
    }

    pub(crate) fn contract_sha256(&self) -> Result<String, String> {
        canonical_sha256(&judge_config_generation_value(self))
    }

    pub(crate) fn evaluator_version(&self) -> Result<String, String> {
        judge_generation_value(self)?
            .get("policy_sha256")
            .and_then(Json::as_str)
            .map(str::to_owned)
            .ok_or_else(|| "judge policy generation omitted policy_sha256".to_string())
    }
}

/// Selector predicates evaluated only against frozen V2 call facts.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PoolSelectorConfig {
    /// Exact normalized tenant identities, or a wildcard when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tenant_ids: Option<Vec<String>>,
    /// Exact normalized agent identities, or a wildcard when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub agent_ids: Option<Vec<String>>,
    /// Exact owner scope types, or a wildcard when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[cfg_attr(feature = "schema", schemars(with = "Option<Vec<String>>"))]
    pub owner_scope_types: Option<Vec<ScopeType>>,
    /// Conjunctive exact scalar comparisons against sanitized metadata.
    #[serde(default)]
    pub metadata_equals: BTreeMap<String, Json>,
    /// Restricted owner-to-parent scope-type path patterns, or a wildcard when omitted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_path_patterns: Option<Vec<String>>,
    /// Unrecognized selector fields retained for policy-aware validation.
    #[doc(hidden)]
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown_fields: BTreeMap<String, Json>,
}

/// Future-local lookahead limits used by a pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct LookaheadConfig {
    /// Number of later Primary completions included in a window.
    #[serde(default = "default_primary_llm_completions")]
    pub primary_llm_completions: usize,
    /// Maximum window age in seconds.
    #[serde(default = "default_deadline_seconds")]
    pub deadline_seconds: u64,
    /// Registered lifecycle presets applied to the window.
    #[serde(default)]
    pub lifecycle_presets: Vec<String>,
    /// Maximum events retained in a window.
    #[serde(default = "default_max_events_per_window")]
    pub max_events_per_window: usize,
    /// Maximum canonical event bytes retained in a window.
    #[serde(default = "default_max_bytes_per_window")]
    pub max_bytes_per_window: usize,
    /// Unrecognized lookahead fields retained for policy-aware validation.
    #[doc(hidden)]
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown_fields: BTreeMap<String, Json>,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self {
            primary_llm_completions: default_primary_llm_completions(),
            deadline_seconds: default_deadline_seconds(),
            lifecycle_presets: Vec::new(),
            max_events_per_window: default_max_events_per_window(),
            max_bytes_per_window: default_max_bytes_per_window(),
            unknown_fields: BTreeMap::new(),
        }
    }
}

/// Independent per-pool concurrency limits.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct ConcurrencyConfig {
    /// Maximum concurrent shadow provider calls.
    pub shadow: usize,
    /// Maximum concurrent judge provider calls.
    pub judge: usize,
    /// Per-pool bound used independently for sampled intents and accepted anchors.
    #[serde(default = "default_max_pending")]
    pub max_pending: usize,
    /// Unrecognized concurrency fields retained for policy-aware validation.
    #[doc(hidden)]
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown_fields: BTreeMap<String, Json>,
}

/// One cheaper candidate model in a pool.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CandidateConfig {
    /// Stable candidate identifier, durable together with the pool ID.
    pub id: String,
    /// Provider model identifier.
    pub model: String,
    /// Operator-pinned candidate evidence revision.
    pub model_revision: String,
    /// Normative candidate order within the pool.
    pub cost_rank: u32,
    /// Optional provider context-token limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_context_tokens: Option<u64>,
    /// Explicit model capabilities used by conservative preflight.
    #[serde(default)]
    pub capabilities: CandidateCapabilities,
    /// Unrecognized candidate fields retained for policy-aware validation.
    #[doc(hidden)]
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown_fields: BTreeMap<String, Json>,
}

/// Capabilities explicitly declared for one candidate model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CandidateCapabilities {
    /// Whether the candidate accepts tool definitions and tool messages.
    #[serde(default)]
    pub tools: bool,
    /// Whether the candidate accepts multimodal input.
    #[serde(default)]
    pub multimodal_input: bool,
    /// Whether the candidate supports structured response formats.
    #[serde(default)]
    pub structured_output: bool,
    /// Whether the candidate supports reasoning controls.
    #[serde(default)]
    pub reasoning_controls: bool,
    /// Unrecognized capability fields retained for policy-aware validation.
    #[doc(hidden)]
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown_fields: BTreeMap<String, Json>,
}

/// Limits for deterministic semantic request canonicalization.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CanonicalizerConfig {
    /// Canonicalizer contract version. Only version `1` is supported.
    #[serde(default = "default_canonicalizer_version")]
    pub version: u32,
    /// Maximum UTF-8 bytes of instruction content.
    #[serde(default = "default_max_instruction_bytes")]
    pub max_instruction_bytes: usize,
    /// Maximum UTF-8 bytes of user-task content.
    #[serde(default = "default_max_task_bytes")]
    pub max_task_bytes: usize,
    /// Maximum number of normalized context messages.
    #[serde(default = "default_max_context_messages")]
    pub max_context_messages: usize,
    /// Maximum canonical semantic-context bytes.
    #[serde(default = "default_max_context_bytes")]
    pub max_context_bytes: usize,
    /// Maximum canonical bytes of registered position features.
    #[serde(default = "default_max_position_features_bytes")]
    pub max_position_features_bytes: usize,
    /// Registered scalar position features copied into routing context.
    #[serde(default)]
    pub position_features: Vec<String>,
    /// Unrecognized canonicalizer fields retained for policy-aware validation.
    #[doc(hidden)]
    #[serde(flatten, default, skip_serializing_if = "BTreeMap::is_empty")]
    pub unknown_fields: BTreeMap<String, Json>,
}

impl Default for CanonicalizerConfig {
    fn default() -> Self {
        Self {
            version: default_canonicalizer_version(),
            max_instruction_bytes: default_max_instruction_bytes(),
            max_task_bytes: default_max_task_bytes(),
            max_context_messages: default_max_context_messages(),
            max_context_bytes: default_max_context_bytes(),
            max_position_features_bytes: default_max_position_features_bytes(),
            position_features: Vec::new(),
            unknown_fields: BTreeMap::new(),
        }
    }
}

fn validate_pool(
    pool: &PoolConfig,
    index: usize,
    policy: &ConfigPolicy,
    retention_days: u32,
    max_evidence_records: u64,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let prefix = format!("pools[{index}]");
    validate_unknown_fields(diagnostics, policy, &pool.unknown_fields, &prefix);
    validate_stable_id(&pool.id, ID_MAX_BYTES, &format!("{prefix}.id"), diagnostics);
    if pool.anchor_models.is_empty() {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.anchor_models")),
            "anchor_models must contain at least one model",
        ));
    }
    let mut anchor_models = BTreeMap::new();
    for (model_index, model) in pool.anchor_models.iter().enumerate() {
        let field = format!("{prefix}.anchor_models[{model_index}]");
        validate_provider_identifier(model, &field, diagnostics);
        if let Some(previous) = anchor_models.insert(model.as_str(), model_index) {
            diagnostics.push(error(
                DUPLICATE_ID,
                Some(field),
                format!("anchor model '{model}' duplicates anchor_models[{previous}]"),
            ));
        }
    }
    validate_revision(
        &pool.anchor_revision,
        &format!("{prefix}.anchor_revision"),
        diagnostics,
    );
    if !pool.sampling_probability.is_finite() || !(0.0..=1.0).contains(&pool.sampling_probability) {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.sampling_probability")),
            "sampling_probability must be finite and in the inclusive range [0, 1]",
        ));
    }
    validate_positive_u64(
        pool.max_candidates_per_sample as u64,
        &format!("{prefix}.max_candidates_per_sample"),
        diagnostics,
    );

    validate_selector(&pool.selector, &prefix, policy, diagnostics);
    validate_lookahead(&pool.lookahead, &prefix, policy, diagnostics);
    validate_concurrency(&pool.concurrency, &prefix, policy, diagnostics);
    validate_canonicalizer(&pool.canonicalizer, &prefix, policy, diagnostics);
    validate_judge(&pool.judge, &prefix, diagnostics);
    if let Some(learning) = &pool.learning {
        validate_learning(learning, pool, &prefix, diagnostics);
    }
    validate_outcome(
        &pool.outcome,
        pool.learning.as_ref(),
        retention_days,
        max_evidence_records,
        &prefix,
        diagnostics,
    );

    if pool.candidates.is_empty() {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.candidates")),
            "candidates must contain at least one model",
        ));
    }
    if pool.max_candidates_per_sample > pool.candidates.len() {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.max_candidates_per_sample")),
            "max_candidates_per_sample cannot exceed the candidate count",
        ));
    }
    let mut candidate_ids = BTreeMap::new();
    let mut candidate_models = BTreeMap::new();
    let mut cost_ranks = BTreeMap::new();
    for (candidate_index, candidate) in pool.candidates.iter().enumerate() {
        let candidate_prefix = format!("{prefix}.candidates[{candidate_index}]");
        validate_unknown_fields(
            diagnostics,
            policy,
            &candidate.unknown_fields,
            &candidate_prefix,
        );
        validate_unknown_fields(
            diagnostics,
            policy,
            &candidate.capabilities.unknown_fields,
            &format!("{candidate_prefix}.capabilities"),
        );
        validate_stable_id(
            &candidate.id,
            CANDIDATE_ID_MAX_BYTES,
            &format!("{candidate_prefix}.id"),
            diagnostics,
        );
        if let Some(previous) = candidate_ids.insert(candidate.id.as_str(), candidate_index) {
            diagnostics.push(error(
                DUPLICATE_ID,
                Some(format!("{candidate_prefix}.id")),
                format!(
                    "candidate ID '{}' duplicates candidates[{previous}].id",
                    candidate.id
                ),
            ));
        }
        validate_provider_identifier(
            &candidate.model,
            &format!("{candidate_prefix}.model"),
            diagnostics,
        );
        if let Some(previous) = candidate_models.insert(candidate.model.as_str(), candidate_index) {
            diagnostics.push(error(
                DUPLICATE_ID,
                Some(format!("{candidate_prefix}.model")),
                format!(
                    "candidate model '{}' duplicates candidates[{previous}].model",
                    candidate.model
                ),
            ));
        }
        if anchor_models.contains_key(candidate.model.as_str()) {
            diagnostics.push(error(
                DUPLICATE_ID,
                Some(format!("{candidate_prefix}.model")),
                format!(
                    "candidate model '{}' is also an anchor model",
                    candidate.model
                ),
            ));
        }
        validate_revision(
            &candidate.model_revision,
            &format!("{candidate_prefix}.model_revision"),
            diagnostics,
        );
        if let Some(previous) = cost_ranks.insert(candidate.cost_rank, candidate_index) {
            diagnostics.push(error(
                DUPLICATE_ID,
                Some(format!("{candidate_prefix}.cost_rank")),
                format!(
                    "cost rank {} duplicates candidates[{previous}].cost_rank",
                    candidate.cost_rank
                ),
            ));
        }
        if candidate.max_context_tokens == Some(0) {
            diagnostics.push(error(
                INVALID_RANGE,
                Some(format!("{candidate_prefix}.max_context_tokens")),
                "max_context_tokens must be positive when configured",
            ));
        }
    }
}

fn validate_learning(
    learning: &LearningConfig,
    pool: &PoolConfig,
    pool_prefix: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let prefix = format!("{pool_prefix}.learning");
    if learning.version != LEARNING_CONFIG_VERSION {
        diagnostics.push(error(
            UNSUPPORTED_CONFIG_VERSION,
            Some(format!("{prefix}.version")),
            format!(
                "learning configuration version {} is not supported",
                learning.version
            ),
        ));
    }

    let Some(policy) = learning.complete_policy() else {
        if learning.has_statistical_fields() || learning.has_active_fields() {
            diagnostics.push(error(
                INVALID_PLUGIN_CONFIG,
                Some(prefix),
                "learning statistical fields must be all present before Active fields are configured",
            ));
        }
        return;
    };

    if !(1..=LEARNING_TOP_K_MAX).contains(&policy.top_k) {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.top_k")),
            format!("top_k must be in the inclusive range [1, {LEARNING_TOP_K_MAX}]"),
        ));
    }
    if policy.min_points == 0 || policy.min_points > policy.top_k {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.min_points")),
            "min_points must be positive and no greater than top_k",
        ));
    }
    if policy.min_independent_roots == 0 || policy.min_independent_roots > policy.top_k {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.min_independent_roots")),
            "min_independent_roots must be positive and no greater than top_k",
        ));
    }
    if !policy.radius.is_finite() || policy.radius <= 0.0 || policy.radius > 2.0 {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.radius")),
            "radius must be finite and in the range (0, 2]",
        ));
    }
    let top_k_f64 = u32::try_from(policy.top_k).map_or(f64::INFINITY, f64::from);
    if !policy.min_effective_samples.is_finite()
        || policy.min_effective_samples <= 0.0
        || policy.min_effective_samples > top_k_f64
    {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.min_effective_samples")),
            "min_effective_samples must be finite, positive, and no greater than top_k",
        ));
    }
    validate_learning_unit_interval(
        policy.min_coverage,
        &format!("{prefix}.min_coverage"),
        diagnostics,
    );
    validate_learning_positive_finite(
        policy.time_decay_half_life_seconds,
        &format!("{prefix}.time_decay_half_life_seconds"),
        diagnostics,
    );
    validate_learning_positive_finite(
        policy.prior_success,
        &format!("{prefix}.prior_success"),
        diagnostics,
    );
    validate_learning_positive_finite(
        policy.prior_failure,
        &format!("{prefix}.prior_failure"),
        diagnostics,
    );
    if !policy.familywise_credible_level.is_finite()
        || policy.familywise_credible_level <= 0.5
        || policy.familywise_credible_level >= 1.0
    {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.familywise_credible_level")),
            "familywise_credible_level must be finite and in the range (0.5, 1)",
        ));
    }
    validate_learning_unit_interval(
        policy.promotion_lower_bound,
        &format!("{prefix}.promotion_lower_bound"),
        diagnostics,
    );

    if learning.has_active_fields() {
        let Some(active) = learning.complete_active_policy() else {
            diagnostics.push(error(
                INVALID_PLUGIN_CONFIG,
                Some(prefix.clone()),
                "Active learning fields must be either all absent or all present",
            ));
            return;
        };
        validate_active_learning(active, &prefix, diagnostics);
    }

    if pool.candidates.len() > LEARNING_CANDIDATES_MAX {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{pool_prefix}.candidates")),
            format!(
                "a complete learning policy supports at most {LEARNING_CANDIDATES_MAX} candidates"
            ),
        ));
    }
    match pool.candidates.len().checked_mul(policy.top_k) {
        Some(product) if product <= LEARNING_CANDIDATE_NEIGHBOR_PRODUCT_MAX => {}
        Some(product) => diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.top_k")),
            format!(
                "configured candidate count multiplied by top_k ({product}) exceeds {LEARNING_CANDIDATE_NEIGHBOR_PRODUCT_MAX}"
            ),
        )),
        None => diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.top_k")),
            "configured candidate count multiplied by top_k overflows usize",
        )),
    }
}

fn validate_active_learning(
    policy: CompleteActiveLearningPolicyV1<'_>,
    prefix: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if !policy.retention_lower_bound.is_finite()
        || policy.retention_lower_bound < 0.0
        || policy.retention_lower_bound >= policy.recommend.promotion_lower_bound
    {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.retention_lower_bound")),
            "retention_lower_bound must be finite, nonnegative, and lower than promotion_lower_bound",
        ));
    }
    if !policy.holdout_probability.is_finite()
        || policy.holdout_probability <= 0.0
        || policy.holdout_probability > 0.25
    {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.holdout_probability")),
            "holdout_probability must be finite and in the range (0, 0.25]",
        ));
    }
    if !policy.active_canary_fraction.is_finite()
        || policy.active_canary_fraction <= 0.0
        || policy.active_canary_fraction >= 1.0
    {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.active_canary_fraction")),
            "active_canary_fraction must be finite and in the range (0, 1)",
        ));
    }
    if policy.holdout_probability.is_finite()
        && policy.active_canary_fraction.is_finite()
        && policy.holdout_probability + policy.active_canary_fraction >= 1.0
    {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.active_canary_fraction")),
            "holdout_probability plus active_canary_fraction must be strictly less than 1",
        ));
    }

    if exact_probability_threshold(policy.holdout_probability).is_none() {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.holdout_probability")),
            "holdout_probability must produce an integer threshold in [1, 2^64 - 1]",
        ));
    }
    let conditional_canary = policy.active_canary_fraction / (1.0 - policy.holdout_probability);
    if exact_probability_threshold(conditional_canary).is_none() {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.active_canary_fraction")),
            "conditional Active probability must produce an integer threshold in [1, 2^64 - 1]",
        ));
    }
}

pub(crate) fn exact_probability_threshold(probability: f64) -> Option<u128> {
    if !probability.is_finite() || probability <= 0.0 || probability >= 1.0 {
        return None;
    }
    let bits = probability.to_bits();
    let exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    let (significand, binary_exponent) = if exponent == 0 {
        (u128::from(fraction), -1022 - 52)
    } else {
        (u128::from((1_u64 << 52) | fraction), exponent - 1023 - 52)
    };
    let shift = binary_exponent + 64;
    let threshold = if shift >= 0 {
        significand.checked_shl(u32::try_from(shift).ok()?)?
    } else {
        significand.checked_shr(shift.unsigned_abs())?
    };
    (threshold > 0 && threshold < (1_u128 << 64)).then_some(threshold)
}

fn validate_outcome(
    raw: &BTreeMap<String, Json>,
    learning: Option<&LearningConfig>,
    retention_days: u32,
    max_evidence_records: u64,
    pool_prefix: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if raw.is_empty() {
        return;
    }
    let prefix = format!("{pool_prefix}.outcome");
    let value = Json::Object(raw.clone().into_iter().collect());
    let config = match serde_json::from_value::<OutcomeConfig>(value) {
        Ok(config) => config,
        Err(parse_error) => {
            diagnostics.push(error(
                INVALID_PLUGIN_CONFIG,
                Some(prefix),
                format!("invalid outcome policy: {parse_error}"),
            ));
            return;
        }
    };
    let first_diagnostic = diagnostics.len();

    if config.version != OUTCOME_CONFIG_VERSION {
        diagnostics.push(error(
            UNSUPPORTED_CONFIG_VERSION,
            Some(format!("{prefix}.version")),
            format!(
                "outcome configuration version {} is not supported",
                config.version
            ),
        ));
    }

    let mut matcher_hashes = BTreeMap::new();
    validate_outcome_matcher_list(
        &config.success_matchers,
        "success_matchers",
        &prefix,
        &mut matcher_hashes,
        diagnostics,
    );
    validate_outcome_matcher_list(
        &config.failure_matchers,
        "failure_matchers",
        &prefix,
        &mut matcher_hashes,
        diagnostics,
    );

    if config.error_disposition != OutcomeDisposition::Failure {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.error_disposition")),
            "error_disposition must be 'failure'",
        ));
    }
    if config.tool_failure_disposition != OutcomeDisposition::Failure {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.tool_failure_disposition")),
            "tool_failure_disposition must be 'failure'",
        ));
    }

    for (field, duration) in [
        ("max_attribution_seconds", config.max_attribution_seconds),
        (
            "actual_outcome_half_life_seconds",
            config.actual_outcome_half_life_seconds,
        ),
        (
            "anchor_shadow_half_life_seconds",
            config.anchor_shadow_half_life_seconds,
        ),
        (
            "relearning_cooloff_seconds",
            config.relearning_cooloff_seconds,
        ),
        (
            "authorization_ttl_seconds",
            config.authorization_ttl_seconds,
        ),
    ] {
        if !(1..=OUTCOME_DURATION_SECONDS_MAX).contains(&duration)
            || duration.checked_mul(1_000).is_none()
        {
            diagnostics.push(error(
                INVALID_RANGE,
                Some(format!("{prefix}.{field}")),
                format!(
                    "{field} must be in the inclusive range [1, {OUTCOME_DURATION_SECONDS_MAX}] and convert to milliseconds"
                ),
            ));
        }
    }
    let retention_seconds = u64::from(retention_days).checked_mul(86_400);
    if retention_seconds.is_none_or(|retention| config.max_attribution_seconds > retention) {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.max_attribution_seconds")),
            "max_attribution_seconds cannot exceed the configured retention horizon",
        ));
    }
    if config.authorization_ttl_seconds > config.actual_outcome_half_life_seconds {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.authorization_ttl_seconds")),
            "authorization_ttl_seconds cannot exceed actual_outcome_half_life_seconds",
        ));
    }
    if let Some(recommend) = learning.and_then(LearningConfig::complete_policy) {
        for (field, half_life) in [
            (
                "actual_outcome_half_life_seconds",
                config.actual_outcome_half_life_seconds,
            ),
            (
                "anchor_shadow_half_life_seconds",
                config.anchor_shadow_half_life_seconds,
            ),
        ] {
            if half_life as f64 >= recommend.time_decay_half_life_seconds {
                diagnostics.push(error(
                    INVALID_RANGE,
                    Some(format!("{prefix}.{field}")),
                    format!("{field} must be shorter than time_decay_half_life_seconds"),
                ));
            }
        }
    }

    for (field, minimum) in [
        ("min_treatment_roots", config.min_treatment_roots),
        ("min_control_roots", config.min_control_roots),
    ] {
        if minimum < 32 {
            diagnostics.push(error(
                INVALID_RANGE,
                Some(format!("{prefix}.{field}")),
                format!("{field} must be at least 32"),
            ));
        }
    }
    validate_effective_weight(
        config.min_treatment_effective_weight,
        config.min_treatment_roots,
        config.max_canary_roots,
        &format!("{prefix}.min_treatment_effective_weight"),
        diagnostics,
    );
    validate_effective_weight(
        config.min_control_effective_weight,
        config.min_control_roots,
        config.max_canary_roots,
        &format!("{prefix}.min_control_effective_weight"),
        diagnostics,
    );

    if !config.noninferiority_margin.is_finite()
        || config.noninferiority_margin < 0.0
        || config.noninferiority_margin > 0.25
        || (config.noninferiority_margin == 0.0 && config.noninferiority_margin.is_sign_negative())
    {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.noninferiority_margin")),
            "noninferiority_margin must be finite, not negative zero, and in [0, 0.25]",
        ));
    }
    validate_probability_range(
        config.noninferiority_probability,
        0.99,
        &format!("{prefix}.noninferiority_probability"),
        diagnostics,
    );
    validate_probability_range(
        config.rollback_probability,
        0.95,
        &format!("{prefix}.rollback_probability"),
        diagnostics,
    );

    if !(64..=OUTCOME_EVALUATION_BATCH_SIZE_MAX).contains(&config.outcome_evaluation_batch_size) {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.outcome_evaluation_batch_size")),
            format!(
                "outcome_evaluation_batch_size must be in [64, {OUTCOME_EVALUATION_BATCH_SIZE_MAX}]"
            ),
        ));
    }
    if config.max_canary_roots == 0 || config.max_canary_roots > OUTCOME_MAX_CANARY_ROOTS_MAX {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.max_canary_roots")),
            format!("max_canary_roots must be in [1, {OUTCOME_MAX_CANARY_ROOTS_MAX}]"),
        ));
    }
    let minimum_sum = config
        .min_treatment_roots
        .checked_add(config.min_control_roots);
    if minimum_sum.is_none_or(|sum| config.max_canary_roots < sum) {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.max_canary_roots")),
            "max_canary_roots must be at least the checked sum of both raw-root minima",
        ));
    }
    if minimum_sum.is_none_or(|sum| config.outcome_evaluation_batch_size < sum) {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.outcome_evaluation_batch_size")),
            "outcome_evaluation_batch_size must be at least the checked sum of both raw-root minima",
        ));
    }
    if config.outcome_evaluation_batch_size == 0
        || config.max_canary_roots % config.outcome_evaluation_batch_size != 0
    {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.max_canary_roots")),
            "max_canary_roots must be an exact multiple of outcome_evaluation_batch_size",
        ));
    }

    let max_looks = config
        .max_canary_roots
        .checked_div(config.outcome_evaluation_batch_size);
    if max_looks.is_none_or(|looks| looks == 0 || looks > OUTCOME_MAX_LOOKS) {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.max_canary_roots")),
            format!("the configured policy must produce 1..={OUTCOME_MAX_LOOKS} looks"),
        ));
    }
    let total_cap = config.max_canary_roots.checked_mul(2);
    if total_cap.is_none_or(|cap| cap > max_evidence_records) {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.max_canary_roots")),
            "twice max_canary_roots must fit max_evidence_records",
        ));
    }

    if let Some(looks) = max_looks.filter(|looks| *looks > 0) {
        validate_prior_only_separation(
            config.noninferiority_margin,
            config.noninferiority_probability,
            looks,
            &prefix,
            diagnostics,
        );
    }
    if let Some(active) = learning.and_then(LearningConfig::complete_active_policy) {
        validate_arm_power(&config, active, &prefix, diagnostics);
    }

    if diagnostics.len() == first_diagnostic {
        match outcome_policy_identity(&config) {
            Ok(identity) => {
                if identity.canonical_size_bytes > OUTCOME_POLICY_IDENTITY_MAX_BYTES {
                    diagnostics.push(error(
                        INVALID_RANGE,
                        Some(prefix),
                        format!(
                            "canonical protected outcome policy exceeds {OUTCOME_POLICY_IDENTITY_MAX_BYTES} bytes"
                        ),
                    ));
                }
            }
            Err(message) => diagnostics.push(error(INVALID_PLUGIN_CONFIG, Some(prefix), message)),
        }
    }
}

fn validate_outcome_matcher_list(
    matchers: &[OutcomeMatcher],
    list_name: &str,
    prefix: &str,
    seen_hashes: &mut BTreeMap<String, String>,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if matchers.is_empty() || matchers.len() > OUTCOME_MATCHERS_MAX {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.{list_name}")),
            format!("{list_name} must contain 1..={OUTCOME_MATCHERS_MAX} matchers"),
        ));
    }
    for (index, matcher) in matchers.iter().enumerate() {
        let matcher_prefix = format!("{prefix}.{list_name}[{index}]");
        let first_diagnostic = diagnostics.len();
        for (field, value) in [("category", &matcher.category), ("name", &matcher.name)] {
            if value.is_empty() || value.len() > OUTCOME_MATCHER_TEXT_MAX_BYTES {
                diagnostics.push(error(
                    INVALID_RANGE,
                    Some(format!("{matcher_prefix}.{field}")),
                    format!(
                        "{field} must contain 1..={OUTCOME_MATCHER_TEXT_MAX_BYTES} UTF-8 bytes"
                    ),
                ));
            }
        }
        if matcher.event_kind == OutcomeMatcherEventKind::Mark
            && matcher.terminal_status != OutcomeTerminalStatus::Unset
        {
            diagnostics.push(error(
                INVALID_RANGE,
                Some(format!("{matcher_prefix}.terminal_status")),
                "a mark matcher must use terminal_status 'unset'",
            ));
        }
        if matcher.metadata_equals.len() > OUTCOME_METADATA_EQUALS_MAX {
            diagnostics.push(error(
                INVALID_RANGE,
                Some(format!("{matcher_prefix}.metadata_equals")),
                format!(
                    "metadata_equals must contain at most {OUTCOME_METADATA_EQUALS_MAX} clauses"
                ),
            ));
        }
        for (key, value) in &matcher.metadata_equals {
            let field = format!("{matcher_prefix}.metadata_equals.{key}");
            if !matches!(
                key.as_str(),
                "error.type"
                    | "outcome"
                    | "outcome.label"
                    | "outcome.success"
                    | "result"
                    | "status"
                    | "success"
            ) {
                diagnostics.push(error(
                    INVALID_REFERENCE,
                    Some(field),
                    format!("metadata key '{key}' is not allowlisted for outcome matching"),
                ));
                continue;
            }
            if let Err(message) = validate_outcome_scalar(value) {
                diagnostics.push(error(INVALID_RANGE, Some(field), message));
            }
        }

        if diagnostics.len() == first_diagnostic {
            match protected_outcome_matcher(matcher) {
                Ok(protected) => {
                    if let Some(previous) =
                        seen_hashes.insert(protected.matcher_hash.clone(), matcher_prefix.clone())
                    {
                        diagnostics.push(error(
                            DUPLICATE_ID,
                            Some(matcher_prefix),
                            format!("outcome matcher duplicates '{previous}'"),
                        ));
                    }
                }
                Err(message) => {
                    diagnostics.push(error(INVALID_PLUGIN_CONFIG, Some(matcher_prefix), message))
                }
            }
        }
    }
}

fn validate_outcome_scalar(value: &Json) -> Result<(), String> {
    match value {
        Json::Null | Json::Bool(_) => Ok(()),
        Json::Number(number) => {
            let value = number
                .as_f64()
                .ok_or_else(|| "numeric matcher value must be finite binary64".to_string())?;
            if value == 0.0 && value.is_sign_negative() {
                return Err("numeric matcher value cannot be negative zero".to_string());
            }
            if value.fract() == 0.0 && value.abs() > MAX_INTEROPERABLE_JSON_INTEGER as f64 {
                return Err(
                    "integral matcher value exceeds the interoperable JSON range".to_string(),
                );
            }
            Ok(())
        }
        Json::String(label)
            if !label.is_empty()
                && label.len() <= OUTCOME_LABEL_MAX_BYTES
                && label.bytes().all(|byte| {
                    byte.is_ascii_lowercase() || byte.is_ascii_digit() || b"_.-".contains(&byte)
                }) =>
        {
            Ok(())
        }
        Json::String(_) => Err(format!(
            "string matcher value must match [a-z0-9_.-]{{1,{OUTCOME_LABEL_MAX_BYTES}}}"
        )),
        Json::Array(_) | Json::Object(_) => {
            Err("matcher value must be null, boolean, finite number, or safe label".to_string())
        }
    }
}

fn validate_effective_weight(
    weight: f64,
    raw_minimum: u64,
    max_canary_roots: u64,
    field: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let minimum = raw_minimum as f64 / 2.0;
    if !weight.is_finite() || weight < minimum || weight > max_canary_roots as f64 {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(field.to_string()),
            "effective-weight floor must be finite, at least half its raw-root minimum, and no greater than max_canary_roots",
        ));
    }
}

fn validate_probability_range(
    probability: f64,
    minimum: f64,
    field: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if !probability.is_finite() || probability < minimum || probability >= 1.0 {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(field.to_string()),
            format!("value must be finite and in [{minimum}, 1)"),
        ));
    }
}

fn validate_prior_only_separation(
    margin: f64,
    probability: f64,
    max_looks: u64,
    prefix: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let per_look = u32::try_from(max_looks)
        .ok()
        .and_then(|looks| crate::active_math::bonferroni_threshold_v1(probability, looks));
    let Some(per_look) = per_look else {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.noninferiority_probability")),
            "the binary64 per-look noninferiority threshold must remain finite and below 1",
        ));
        return;
    };
    let prior_only = crate::active_math::prior_only_noninferiority_v1(margin);
    if prior_only.is_none_or(|prior_only| per_look <= prior_only + OUTCOME_POSTERIOR_RESOLUTION_V1)
    {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.noninferiority_probability")),
            "the per-look threshold must exceed prior-only noninferiority by the posterior resolution",
        ));
    }
}

fn validate_arm_power(
    outcome: &OutcomeConfig,
    active: CompleteActiveLearningPolicyV1<'_>,
    prefix: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let conditional = active.active_canary_fraction / (1.0 - active.holdout_probability);
    let Some(canary_threshold) = exact_probability_threshold(conditional) else {
        return;
    };
    let scale = 1_u128 << 64;
    let max_roots = u128::from(outcome.max_canary_roots);
    if max_roots * canary_threshold < u128::from(outcome.min_treatment_roots) * scale {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.min_treatment_roots")),
            "the treatment arm is underpowered at max_canary_roots",
        ));
    }
    if max_roots * (scale - canary_threshold) < u128::from(outcome.min_control_roots) * scale {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.min_control_roots")),
            "the control arm is underpowered at max_canary_roots",
        ));
    }
}

fn validate_learning_positive_finite(
    value: f64,
    field: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if !value.is_finite() || value <= 0.0 {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(field.to_string()),
            "value must be finite and positive",
        ));
    }
}

fn validate_learning_unit_interval(
    value: f64,
    field: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(field.to_string()),
            "value must be finite and in the inclusive range [0, 1]",
        ));
    }
}

fn validate_judge(
    config: &JudgeConfig,
    pool_prefix: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let prefix = format!("{pool_prefix}.judge");
    for field in config.unknown_fields.keys() {
        diagnostics.push(error(
            UNKNOWN_FIELD,
            Some(format!("{prefix}.{field}")),
            format!("field '{field}' is not recognized for 'judge'"),
        ));
    }

    if config.version != JUDGE_CONFIG_VERSION {
        diagnostics.push(error(
            UNSUPPORTED_CONFIG_VERSION,
            Some(format!("{prefix}.version")),
            format!("judge config version {} is not supported", config.version),
        ));
    }
    validate_provider_identifier(&config.model, &format!("{prefix}.model"), diagnostics);
    validate_revision(
        &config.model_revision,
        &format!("{prefix}.model_revision"),
        diagnostics,
    );
    validate_template_version(
        &config.prompt_version,
        JUDGE_PROMPT_VERSION_V1,
        "prompt_version",
        &prefix,
        diagnostics,
    );
    validate_template_version(
        &config.rubric_version,
        JUDGE_RUBRIC_VERSION_V1,
        "rubric_version",
        &prefix,
        diagnostics,
    );
    if config.output_schema_version != JUDGE_OUTPUT_SCHEMA_VERSION {
        diagnostics.push(error(
            UNSUPPORTED_CONFIG_VERSION,
            Some(format!("{prefix}.output_schema_version")),
            format!(
                "judge output schema version {} is not supported",
                config.output_schema_version
            ),
        ));
    }
    if let Some(temperature) = config.temperature {
        validate_unit_interval(temperature, &format!("{prefix}.temperature"), diagnostics);
    }

    let response_weight_valid = validate_weight(
        config.response_weight,
        &format!("{prefix}.response_weight"),
        diagnostics,
    );
    let trajectory_weight_valid = validate_weight(
        config.trajectory_weight,
        &format!("{prefix}.trajectory_weight"),
        diagnostics,
    );
    if response_weight_valid && trajectory_weight_valid {
        let sum = config.response_weight + config.trajectory_weight;
        if !sum.is_finite() || (sum - 1.0).abs() > JUDGE_WEIGHT_SUM_TOLERANCE {
            diagnostics.push(error(
                INVALID_RANGE,
                Some(format!("{prefix}.trajectory_weight")),
                format!(
                    "response_weight and trajectory_weight must sum to 1.0 within {JUDGE_WEIGHT_SUM_TOLERANCE}"
                ),
            ));
        }
    }

    for (name, value) in [
        ("response_floor", config.response_floor),
        ("trajectory_floor", config.trajectory_floor),
        ("judge_confidence_floor", config.judge_confidence_floor),
        ("pass_threshold", config.pass_threshold),
    ] {
        validate_unit_interval(value, &format!("{prefix}.{name}"), diagnostics);
    }

    if config.max_rationale_bytes == 0 || config.max_rationale_bytes > JUDGE_MAX_RATIONALE_BYTES {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.max_rationale_bytes")),
            format!(
                "max_rationale_bytes must be in the inclusive range [1, {JUDGE_MAX_RATIONALE_BYTES}]"
            ),
        ));
    }
    if config.output_token_limit().is_none() {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.max_rationale_bytes")),
            "max_rationale_bytes cannot be represented by the output-token formula",
        ));
    }
    validate_positive_u64(
        config.base_cooloff_seconds,
        &format!("{prefix}.base_cooloff_seconds"),
        diagnostics,
    );
    validate_positive_u64(
        config.max_cooloff_seconds,
        &format!("{prefix}.max_cooloff_seconds"),
        diagnostics,
    );
    if config.max_cooloff_seconds < config.base_cooloff_seconds {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.max_cooloff_seconds")),
            "max_cooloff_seconds must be greater than or equal to base_cooloff_seconds",
        ));
    }
}

fn validate_template_version(
    value: &str,
    expected: &str,
    name: &str,
    prefix: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let field = format!("{prefix}.{name}");
    if value != expected {
        diagnostics.push(error(
            INVALID_REFERENCE,
            Some(field),
            format!("unsupported {name} '{value}'; expected '{expected}'"),
        ));
    }
}

fn validate_weight(value: f64, field: &str, diagnostics: &mut Vec<ConfigDiagnostic>) -> bool {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(field.to_string()),
            "weight must be finite and in the inclusive range [0, 1]",
        ));
        false
    } else {
        true
    }
}

fn validate_unit_interval(value: f64, field: &str, diagnostics: &mut Vec<ConfigDiagnostic>) {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(field.to_string()),
            "value must be finite and in the inclusive range [0, 1]",
        ));
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SchedulerSlots {
    pub(crate) batch: usize,
    pub(crate) candidates: usize,
}

pub(crate) fn pool_scheduler_slots(pool: &PoolConfig) -> Option<SchedulerSlots> {
    Some(SchedulerSlots {
        batch: pool.concurrency.max_pending,
        candidates: pool
            .concurrency
            .max_pending
            .checked_mul(pool.max_candidates_per_sample)?,
    })
}

fn scheduler_slot_limit() -> usize {
    SCHEDULER_MAX_SLOTS.min(CONCURRENCY_MAX_PERMITS)
}

fn validate_scheduler_slots(pools: &[PoolConfig], diagnostics: &mut Vec<ConfigDiagnostic>) {
    let limit = scheduler_slot_limit();
    let mut total_batch = Some(0usize);
    let mut total_candidates = Some(0usize);

    for (index, pool) in pools.iter().enumerate() {
        let batch = pool.concurrency.max_pending;
        if batch > limit {
            diagnostics.push(error(
                INVALID_RANGE,
                Some(format!("pools[{index}].concurrency.max_pending")),
                format!("derived batch slots must be at most {limit}"),
            ));
        }
        total_batch = total_batch.and_then(|total| total.checked_add(batch));

        match pool_scheduler_slots(pool) {
            Some(slots) => {
                if slots.candidates > limit {
                    diagnostics.push(error(
                        INVALID_RANGE,
                        Some(format!("pools[{index}].max_candidates_per_sample")),
                        format!("derived candidate slots must be at most {limit}"),
                    ));
                }
                total_candidates =
                    total_candidates.and_then(|total| total.checked_add(slots.candidates));
            }
            None => {
                diagnostics.push(error(
                    INVALID_RANGE,
                    Some(format!("pools[{index}].max_candidates_per_sample")),
                    "derived candidate slots overflow usize",
                ));
                total_candidates = None;
            }
        }
    }

    validate_scheduler_total(total_batch, "batch", limit, diagnostics);
    validate_scheduler_total(total_candidates, "candidate", limit, diagnostics);
}

fn validate_scheduler_total(
    total: Option<usize>,
    name: &str,
    limit: usize,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    match total {
        Some(total) if total <= limit => {}
        Some(total) => diagnostics.push(error(
            INVALID_RANGE,
            Some("pools".to_string()),
            format!("aggregate {name} slots {total} exceed the limit {limit}"),
        )),
        None => diagnostics.push(error(
            INVALID_RANGE,
            Some("pools".to_string()),
            format!("aggregate {name} slots overflow usize"),
        )),
    }
}

fn validate_lookahead(
    config: &LookaheadConfig,
    pool_prefix: &str,
    policy: &ConfigPolicy,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let prefix = format!("{pool_prefix}.lookahead");
    validate_unknown_fields(diagnostics, policy, &config.unknown_fields, &prefix);
    for (name, value) in [
        (
            "primary_llm_completions",
            config.primary_llm_completions as u64,
        ),
        ("deadline_seconds", config.deadline_seconds),
        ("max_events_per_window", config.max_events_per_window as u64),
        ("max_bytes_per_window", config.max_bytes_per_window as u64),
    ] {
        validate_positive_u64(value, &format!("{prefix}.{name}"), diagnostics);
    }
    let mut lifecycle_presets = BTreeMap::new();
    for (preset_index, preset) in config.lifecycle_presets.iter().enumerate() {
        let field = format!("{prefix}.lifecycle_presets[{preset_index}]");
        if !matches!(preset.as_str(), "handoff" | "compaction") {
            diagnostics.push(error(
                INVALID_REFERENCE,
                Some(field.clone()),
                format!(
                    "lifecycle preset '{preset}' is not registered; expected 'handoff' or 'compaction'"
                ),
            ));
        }
        if let Some(previous) = lifecycle_presets.insert(preset.as_str(), preset_index) {
            diagnostics.push(error(
                DUPLICATE_ID,
                Some(field),
                format!("lifecycle preset '{preset}' duplicates lifecycle_presets[{previous}]"),
            ));
        }
    }
}

fn validate_concurrency(
    config: &ConcurrencyConfig,
    pool_prefix: &str,
    policy: &ConfigPolicy,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let prefix = format!("{pool_prefix}.concurrency");
    validate_unknown_fields(diagnostics, policy, &config.unknown_fields, &prefix);
    for (name, value) in [
        ("shadow", config.shadow),
        ("judge", config.judge),
        ("max_pending", config.max_pending),
    ] {
        let field = format!("{prefix}.{name}");
        validate_positive_u64(value as u64, &field, diagnostics);
        if value > CONCURRENCY_MAX_PERMITS {
            diagnostics.push(error(
                INVALID_RANGE,
                Some(field),
                format!("value must be at most {CONCURRENCY_MAX_PERMITS}"),
            ));
        }
    }
}

fn validate_canonicalizer(
    config: &CanonicalizerConfig,
    pool_prefix: &str,
    policy: &ConfigPolicy,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let prefix = format!("{pool_prefix}.canonicalizer");
    validate_unknown_fields(diagnostics, policy, &config.unknown_fields, &prefix);
    if config.version != CANONICALIZER_VERSION {
        diagnostics.push(error(
            UNSUPPORTED_CONFIG_VERSION,
            Some(format!("{prefix}.version")),
            format!("canonicalizer version {} is not supported", config.version),
        ));
    }
    for (name, value) in [
        ("max_instruction_bytes", config.max_instruction_bytes),
        ("max_task_bytes", config.max_task_bytes),
        ("max_context_messages", config.max_context_messages),
        ("max_context_bytes", config.max_context_bytes),
        (
            "max_position_features_bytes",
            config.max_position_features_bytes,
        ),
    ] {
        validate_positive_u64(value as u64, &format!("{prefix}.{name}"), diagnostics);
    }
    let mut features = BTreeMap::new();
    for (feature_index, feature) in config.position_features.iter().enumerate() {
        let field = format!("{prefix}.position_features[{feature_index}]");
        if let Some(previous) = features.insert(feature.as_str(), feature_index) {
            diagnostics.push(error(
                DUPLICATE_ID,
                Some(field.clone()),
                format!("position feature '{feature}' duplicates position_features[{previous}]"),
            ));
        }
        if feature != "turn_index" {
            diagnostics.push(error(
                INVALID_REFERENCE,
                Some(field),
                format!("position feature '{feature}' is not registered"),
            ));
        }
    }
}

pub(crate) fn validate_unknown_fields(
    diagnostics: &mut Vec<ConfigDiagnostic>,
    policy: &ConfigPolicy,
    fields: &BTreeMap<String, Json>,
    prefix: &str,
) {
    for field in fields.keys() {
        let location = if prefix.is_empty() {
            field.clone()
        } else {
            format!("{prefix}.{field}")
        };
        push_policy_diagnostic(
            diagnostics,
            policy.unknown_field,
            UNKNOWN_FIELD,
            Some(location),
            format!("field '{field}' is not recognized"),
        );
    }
}

fn validate_stable_id(
    value: &str,
    max_bytes: usize,
    field: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if !validate_normalized_string(value, max_bytes, field, diagnostics) {
        return;
    }
    let mut characters = value.chars();
    let valid_first = characters.next().is_some_and(char::is_alphanumeric);
    let valid_rest = characters
        .all(|character| character.is_alphanumeric() || matches!(character, '_' | '-' | '.' | ':'));
    if !valid_first || !valid_rest {
        diagnostics.push(error(
            INVALID_REFERENCE,
            Some(field.to_string()),
            "identifier must start with a letter or number and contain only letters, numbers, '.', '_', '-', or ':'",
        ));
    }
}

fn validate_provider_identifier(value: &str, field: &str, diagnostics: &mut Vec<ConfigDiagnostic>) {
    if !validate_normalized_string(value, MODEL_ID_MAX_BYTES, field, diagnostics) {
        return;
    }
    let mut characters = value.chars();
    let valid_first = characters.next().is_some_and(char::is_alphanumeric);
    let valid_rest = characters.all(|character| {
        character.is_alphanumeric() || matches!(character, '/' | ':' | '.' | '_' | '-')
    });
    if !valid_first || !valid_rest {
        diagnostics.push(error(
            INVALID_REFERENCE,
            Some(field.to_string()),
            "provider identifier contains an unsupported character",
        ));
    }
}

fn validate_revision(value: &str, field: &str, diagnostics: &mut Vec<ConfigDiagnostic>) {
    validate_stable_id(value, REVISION_MAX_BYTES, field, diagnostics);
    if matches!(
        value.to_ascii_lowercase().as_str(),
        "latest" | "current" | "unversioned" | "*"
    ) {
        diagnostics.push(error(
            INVALID_REFERENCE,
            Some(field.to_string()),
            "revision must be pinned and cannot use an unversioned alias",
        ));
    }
}

pub(crate) fn validate_selector_identity(
    value: &str,
    field: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let _ = validate_normalized_string(value, SELECTOR_IDENTITY_MAX_BYTES, field, diagnostics);
}

pub(crate) fn validate_normalized_string(
    value: &str,
    max_bytes: usize,
    field: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) -> bool {
    let mut valid = true;
    if value.trim().is_empty() {
        diagnostics.push(error(
            INVALID_REFERENCE,
            Some(field.to_string()),
            "value must not be blank",
        ));
        valid = false;
    }
    if value.len() > max_bytes {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(field.to_string()),
            format!("value exceeds the {max_bytes}-byte UTF-8 limit"),
        ));
        valid = false;
    }
    if value.chars().any(char::is_control) {
        diagnostics.push(error(
            INVALID_REFERENCE,
            Some(field.to_string()),
            "value must not contain control characters",
        ));
        valid = false;
    }
    if !value.nfc().eq(value.chars()) {
        diagnostics.push(error(
            INVALID_REFERENCE,
            Some(field.to_string()),
            "value must use NFC Unicode normalization",
        ));
        valid = false;
    }
    valid
}

fn validate_environment_variable(
    value: &str,
    field: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if value.len() > ID_MAX_BYTES {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(field.to_string()),
            format!("environment variable name exceeds the {ID_MAX_BYTES}-byte limit"),
        ));
        return;
    }
    let mut characters = value.chars();
    let valid_first = characters
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_alphabetic());
    let valid_rest =
        characters.all(|character| character == '_' || character.is_ascii_alphanumeric());
    if !valid_first || !valid_rest {
        diagnostics.push(error(
            INVALID_REFERENCE,
            Some(field.to_string()),
            "api_key_env must be a valid environment-variable name",
        ));
    }
}

fn validate_positive_u64(value: u64, field: &str, diagnostics: &mut Vec<ConfigDiagnostic>) {
    if value == 0 {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(field.to_string()),
            "value must be positive",
        ));
    }
}

fn validate_max_u64(
    value: u64,
    maximum: u64,
    field: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if value > maximum {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(field.to_string()),
            format!("value must be at most {maximum}"),
        ));
    }
}

fn validate_embedder_response_bound(
    config: &EmbedderConfig,
    prefix: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    let response_bound = embedder_response_bytes_bound(config.batch_size, config.dimensions);
    if response_bound.is_none_or(|bytes| bytes > EMBEDDER_RESPONSE_BYTES_MAX) {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(format!("{prefix}.batch_size")),
            format!(
                "derived embedding response bound must not exceed {EMBEDDER_RESPONSE_BYTES_MAX} bytes"
            ),
        ));
    }
}

pub(crate) fn embedder_response_bytes_bound(batch_size: usize, dimensions: u32) -> Option<usize> {
    batch_size
        .checked_mul(usize::try_from(dimensions).ok()?)
        .and_then(|components| components.checked_mul(EMBEDDER_RESPONSE_COMPONENT_BYTES_MAX))
        .and_then(|bytes| bytes.checked_add(EMBEDDER_RESPONSE_BASE_BYTES))
}

fn validate_database_path(value: &str, diagnostics: &mut Vec<ConfigDiagnostic>) {
    let invalid = value.is_empty()
        || value.chars().any(char::is_control)
        || looks_like_uri(value)
        || value.starts_with("//")
        || value.starts_with("\\\\")
        || value.ends_with('/')
        || value.ends_with('\\')
        || is_root_only_path(value)
        || has_unsafe_windows_path_component(value)
        || Path::new(value)
            .components()
            .any(|component| matches!(component, Component::CurDir | Component::ParentDir))
        || value.split(['/', '\\']).any(|component| {
            matches!(component, "." | "..") || is_windows_reserved_device_component(component)
        });

    if invalid {
        diagnostics.push(error(
            INVALID_PATH,
            Some("database_path".to_string()),
            "database_path must be a local file path without URI, device, control, or traversal components",
        ));
    }
}

fn has_unsafe_windows_path_component(value: &str) -> bool {
    value
        .split(['/', '\\'])
        .enumerate()
        .any(|(index, component)| {
            if component.is_empty() {
                return false;
            }
            let drive_prefix = index == 0
                && component.len() == 2
                && component.as_bytes()[0].is_ascii_alphabetic()
                && component.as_bytes()[1] == b':';
            (!drive_prefix && component.contains(['<', '>', ':', '"', '|', '?', '*']))
                || component.ends_with([' ', '.'])
        })
}

fn is_windows_reserved_device_component(component: &str) -> bool {
    let basename = component
        .split(['.', ':'])
        .next()
        .unwrap_or_default()
        .trim_end_matches([' ', '.'])
        .to_ascii_uppercase();

    if matches!(
        basename.as_str(),
        "CON" | "PRN" | "AUX" | "NUL" | "CLOCK$" | "CONIN$" | "CONOUT$"
    ) {
        return true;
    }

    ["COM", "LPT"].into_iter().any(|prefix| {
        basename.strip_prefix(prefix).is_some_and(|suffix| {
            matches!(
                suffix,
                "1" | "2"
                    | "3"
                    | "4"
                    | "5"
                    | "6"
                    | "7"
                    | "8"
                    | "9"
                    | "\u{b9}"
                    | "\u{b2}"
                    | "\u{b3}"
            )
        })
    })
}

fn is_root_only_path(value: &str) -> bool {
    value == "/"
        || value == "\\"
        || (value.len() == 2
            && value.as_bytes()[0].is_ascii_alphabetic()
            && value.as_bytes()[1] == b':')
        || (value.len() == 3
            && value.as_bytes()[0].is_ascii_alphabetic()
            && value.as_bytes()[1] == b':'
            && matches!(value.as_bytes()[2], b'/' | b'\\'))
}

fn looks_like_uri(value: &str) -> bool {
    let Some((scheme, remainder)) = value.split_once(':') else {
        return false;
    };
    let windows_drive = scheme.len() == 1
        && scheme.as_bytes()[0].is_ascii_alphabetic()
        && (remainder.starts_with('/') || remainder.starts_with('\\'));
    !windows_drive
        && !scheme.is_empty()
        && scheme.chars().enumerate().all(|(index, character)| {
            character.is_ascii_alphabetic()
                || index > 0 && (character.is_ascii_digit() || matches!(character, '+' | '-' | '.'))
        })
}

fn validate_embedder_endpoint(
    value: &str,
    allow_remote_https: bool,
    field: &str,
    diagnostics: &mut Vec<ConfigDiagnostic>,
) {
    if value.len() > MODEL_ID_MAX_BYTES {
        diagnostics.push(error(
            INVALID_RANGE,
            Some(field.to_string()),
            format!("embedding endpoint exceeds the {MODEL_ID_MAX_BYTES}-byte limit"),
        ));
        return;
    }
    if let Err(message) = normalize_embedder_endpoint(value, allow_remote_https) {
        diagnostics.push(error(
            UNSAFE_EMBEDDER_ENDPOINT,
            Some(field.to_string()),
            message,
        ));
    }
}

fn embedder_generation_value(
    config: &EmbedderConfig,
    allow_remote_https: bool,
) -> Result<Json, String> {
    let profile = embedder_profile_version(config, allow_remote_https)?;
    serde_json::from_str(&profile.canonical_identity_json)
        .map_err(|error| format!("failed to decode canonical embedder profile identity: {error}"))
}

#[derive(Debug, Clone)]
struct ProtectedOutcomeMatcherV1 {
    matcher_hash: String,
    protected_value: Json,
}

#[derive(Debug, Clone)]
struct OutcomePolicyIdentityV1 {
    protected_value: Json,
    outcome_policy_hash: String,
    canonical_size_bytes: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ProtectedOutcomePolicyVersionV1 {
    pub(crate) outcome_policy_hash: String,
    pub(crate) canonical_policy_json: String,
}

pub(crate) fn protected_outcome_policy_version_v1(
    raw: &BTreeMap<String, Json>,
) -> Result<Option<ProtectedOutcomePolicyVersionV1>, String> {
    if raw.is_empty() {
        return Ok(None);
    }
    let config =
        serde_json::from_value::<OutcomeConfig>(Json::Object(raw.clone().into_iter().collect()))
            .map_err(|error| format!("cannot compile invalid outcome policy: {error}"))?;
    let identity = outcome_policy_identity(&config)?;
    Ok(Some(ProtectedOutcomePolicyVersionV1 {
        outcome_policy_hash: identity.outcome_policy_hash,
        canonical_policy_json: canonical_json(&identity.protected_value)?,
    }))
}

fn protected_outcome_matcher(
    matcher: &OutcomeMatcher,
) -> Result<ProtectedOutcomeMatcherV1, String> {
    let metadata_equals = matcher
        .metadata_equals
        .iter()
        .map(|(key, expected)| {
            validate_outcome_scalar(expected)?;
            Ok(json!({
                "key_sha256": length_prefixed_sha256(
                    OUTCOME_METADATA_KEY_HASH_DOMAIN_V1,
                    key.as_bytes(),
                )?,
                "expected_scalar_sha256": length_prefixed_sha256(
                    OUTCOME_EXPECTED_SCALAR_HASH_DOMAIN_V1,
                    canonical_json(expected)?.as_bytes(),
                )?,
            }))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let protected_value = json!({
        "event_kind": matcher.event_kind,
        "category": matcher.category,
        "name": matcher.name,
        "terminal_status": matcher.terminal_status,
        "metadata_equals": metadata_equals,
    });
    let canonical = canonical_json(&protected_value)?;
    let matcher_hash =
        length_prefixed_sha256(OUTCOME_MATCHER_HASH_DOMAIN_V1, canonical.as_bytes())?;
    Ok(ProtectedOutcomeMatcherV1 {
        matcher_hash,
        protected_value,
    })
}

/// Return the protected version-1 identity of one already validated matcher.
pub(crate) fn outcome_matcher_hash_v1(matcher: &OutcomeMatcher) -> Result<String, String> {
    protected_outcome_matcher(matcher).map(|protected| protected.matcher_hash)
}

/// Canonicalize one safe outcome scalar with the exact config identity rules.
pub(crate) fn canonical_outcome_scalar_v1(value: &Json) -> Result<String, String> {
    validate_outcome_scalar(value)?;
    canonical_json(value)
}

/// Hash one allowlisted metadata key with the pinned version-1 domain.
pub(crate) fn outcome_metadata_key_hash_v1(key: &str) -> Result<String, String> {
    length_prefixed_sha256(OUTCOME_METADATA_KEY_HASH_DOMAIN_V1, key.as_bytes())
}

/// Hash one observed safe scalar with the pinned version-1 domain.
pub(crate) fn outcome_observed_scalar_hash_v1(value: &Json) -> Result<String, String> {
    let canonical = canonical_outcome_scalar_v1(value)?;
    length_prefixed_sha256(OUTCOME_OBSERVED_SCALAR_HASH_DOMAIN_V1, canonical.as_bytes())
}

fn outcome_policy_identity(config: &OutcomeConfig) -> Result<OutcomePolicyIdentityV1, String> {
    let mut seen = BTreeSet::new();
    let mut compile_list = |matchers: &[OutcomeMatcher]| {
        let mut protected = matchers
            .iter()
            .map(protected_outcome_matcher)
            .collect::<Result<Vec<_>, _>>()?;
        protected.sort_by(|left, right| left.matcher_hash.cmp(&right.matcher_hash));
        for matcher in &protected {
            if !seen.insert(matcher.matcher_hash.clone()) {
                return Err("duplicate protected outcome matcher hash".to_string());
            }
        }
        Ok::<Vec<Json>, String>(
            protected
                .into_iter()
                .map(|matcher| {
                    json!({
                        "matcher_hash": matcher.matcher_hash,
                        "matcher": matcher.protected_value,
                    })
                })
                .collect(),
        )
    };
    let success_matchers = compile_list(&config.success_matchers)?;
    let failure_matchers = compile_list(&config.failure_matchers)?;
    let protected_value = json!({
        "schema": "nemo.relay.router.outcome-policy@1",
        "version": config.version,
        "success_matchers": success_matchers,
        "failure_matchers": failure_matchers,
        "completion_disposition": config.completion_disposition,
        "error_disposition": config.error_disposition,
        "tool_failure_disposition": config.tool_failure_disposition,
        "end_of_run_disposition": config.end_of_run_disposition,
        "max_attribution_seconds": config.max_attribution_seconds,
        "actual_outcome_half_life_seconds": config.actual_outcome_half_life_seconds,
        "anchor_shadow_half_life_seconds": config.anchor_shadow_half_life_seconds,
        "relearning_cooloff_seconds": config.relearning_cooloff_seconds,
        "min_treatment_roots": config.min_treatment_roots,
        "min_control_roots": config.min_control_roots,
        "min_treatment_effective_weight": config.min_treatment_effective_weight,
        "min_control_effective_weight": config.min_control_effective_weight,
        "noninferiority_margin": config.noninferiority_margin,
        "noninferiority_probability": config.noninferiority_probability,
        "rollback_probability": config.rollback_probability,
        "outcome_evaluation_batch_size": config.outcome_evaluation_batch_size,
        "max_canary_roots": config.max_canary_roots,
        "authorization_ttl_seconds": config.authorization_ttl_seconds,
        "algorithms": {
            "root_outcome_reducer": OUTCOME_REDUCER_ID_V1,
            "time_decay": OUTCOME_DECAY_ID_V1,
            "future_skew": LEARNING_FUTURE_SKEW_ID_V1,
            "summation": LEARNING_SUMMATION_ID_V1,
            "beta_quantile_bounds": OUTCOME_QUANTILE_ID_V1,
            "beta_cdf": OUTCOME_BETA_CDF_ID_V1,
            "posterior_resolution": OUTCOME_POSTERIOR_RESOLUTION_ID_V1,
            "candidate_correction": OUTCOME_BONFERRONI_ID_V1,
            "attribution_gate": OUTCOME_ATTRIBUTION_GATE_ID_V1,
            "noninferiority_math": OUTCOME_NONINFERIORITY_MATH_ID_V1,
        },
    });
    let canonical = canonical_json(&protected_value)?;
    let outcome_policy_hash =
        length_prefixed_sha256(OUTCOME_POLICY_HASH_DOMAIN_V1, canonical.as_bytes())?;
    Ok(OutcomePolicyIdentityV1 {
        protected_value,
        outcome_policy_hash,
        canonical_size_bytes: canonical.len(),
    })
}

fn outcome_generation_value(raw: &BTreeMap<String, Json>) -> Result<Json, String> {
    if raw.is_empty() {
        return Ok(json!({}));
    }
    let config =
        serde_json::from_value::<OutcomeConfig>(Json::Object(raw.clone().into_iter().collect()))
            .map_err(|error| {
                format!("cannot generate identity for invalid outcome policy: {error}")
            })?;
    let identity = outcome_policy_identity(&config)?;
    if identity.canonical_size_bytes > OUTCOME_POLICY_IDENTITY_MAX_BYTES {
        return Err(format!(
            "canonical protected outcome policy exceeds {OUTCOME_POLICY_IDENTITY_MAX_BYTES} bytes"
        ));
    }
    Ok(json!({
        "protected_policy": identity.protected_value,
        "outcome_policy_hash": identity.outcome_policy_hash,
    }))
}

fn length_prefixed_sha256(domain: &[u8], value: &[u8]) -> Result<String, String> {
    let length = u32::try_from(value.len())
        .map_err(|_| "protected outcome identity component exceeds u32 bytes".to_string())?;
    let mut preimage = Vec::with_capacity(domain.len() + 4 + value.len());
    preimage.extend_from_slice(domain);
    preimage.extend_from_slice(&length.to_be_bytes());
    preimage.extend_from_slice(value);
    Ok(sha256_hex(&preimage))
}

fn pool_generation_value(config: &PoolConfig) -> Result<Json, String> {
    let mut anchor_models = config.anchor_models.clone();
    anchor_models.sort();
    let mut candidates = config.candidates.iter().collect::<Vec<_>>();
    candidates.sort_by(|left, right| left.id.cmp(&right.id));
    let slots = pool_scheduler_slots(config).ok_or_else(|| {
        format!(
            "derived candidate slots overflow for Router pool '{}'",
            config.id
        )
    })?;
    let canonicalizer = canonicalizer_generation_value(&config.canonicalizer)?;
    let learning = config
        .learning
        .as_ref()
        .map_or_else(|| Ok(json!({})), learning_generation_value)?;
    let outcome = outcome_generation_value(&config.outcome)?;
    Ok(json!({
        "id": config.id,
        "api_family": config.api_family,
        "anchor_models": anchor_models,
        "anchor_revision": config.anchor_revision,
        "sampling_probability": config.sampling_probability,
        "max_candidates_per_sample": config.max_candidates_per_sample,
        "selector": selector_generation_value(&config.selector)?,
        "lookahead": lookahead_generation_value(&config.lookahead),
        "concurrency": {
            "shadow": config.concurrency.shadow.to_string(),
            "judge": config.concurrency.judge.to_string(),
            "max_pending": config.concurrency.max_pending,
        },
        "derived_scheduler_policy": {
            "batch_slots": slots.batch,
            "candidate_slots": slots.candidates,
            "slot_limit": SCHEDULER_MAX_SLOTS,
        },
        "candidates": candidates.into_iter().map(candidate_generation_value).collect::<Vec<_>>(),
        "canonicalizer": canonicalizer,
        "judge": judge_generation_value(&config.judge)?,
        "learning": learning,
        "outcome": outcome,
    }))
}

fn learning_generation_value(config: &LearningConfig) -> Result<Json, String> {
    if !config.has_statistical_fields() && !config.has_active_fields() {
        return Ok(json!({
            "version": config.version,
            "embedder": config.embedder,
        }));
    }
    let policy = config
        .complete_policy()
        .ok_or_else(|| "cannot generate identity for a partial learning policy".to_string())?;
    let mut value = json!({
        "version": policy.version,
        "embedder": policy.embedder,
        "top_k": policy.top_k,
        "radius": policy.radius,
        "min_points": policy.min_points,
        "min_independent_roots": policy.min_independent_roots,
        "min_effective_samples": policy.min_effective_samples,
        "min_coverage": policy.min_coverage,
        "time_decay_half_life_seconds": policy.time_decay_half_life_seconds,
        "prior_success": policy.prior_success,
        "prior_failure": policy.prior_failure,
        "familywise_credible_level": policy.familywise_credible_level,
        "promotion_lower_bound": policy.promotion_lower_bound,
        "algorithms": {
            "similarity_kernel": LEARNING_LINEAR_RADIUS_ID_V1,
            "root_selector": LEARNING_ROOT_SELECTOR_ID_V1,
            "coverage": LEARNING_COVERAGE_ID_V1,
            "time_decay": LEARNING_TIME_DECAY_ID_V1,
            "future_skew": LEARNING_FUTURE_SKEW_ID_V1,
            "summation": LEARNING_SUMMATION_ID_V1,
            "effective_sample": LEARNING_EFFECTIVE_SAMPLE_ID_V1,
            "beta_approximation": LEARNING_BETA_APPROXIMATION_ID_V1,
            "candidate_correction": LEARNING_CANDIDATE_CORRECTION_ID_V1,
            "beta_inverse_cdf": LEARNING_BETA_INVERSE_CDF_ID_V1,
            "candidate_set": LEARNING_CANDIDATE_SET_ID_V1,
        },
    });
    if config.has_active_fields() {
        let active = config.complete_active_policy().ok_or_else(|| {
            "cannot generate identity for a partial Active learning policy".to_string()
        })?;
        let object = value
            .as_object_mut()
            .expect("learning generation value must be an object");
        object.insert(
            "retention_lower_bound".into(),
            json!(active.retention_lower_bound),
        );
        object.insert(
            "holdout_probability".into(),
            json!(active.holdout_probability),
        );
        object.insert(
            "active_canary_fraction".into(),
            json!(active.active_canary_fraction),
        );
        let algorithms = object
            .get_mut("algorithms")
            .and_then(Json::as_object_mut)
            .expect("learning algorithms must be an object");
        for (name, identifier) in [
            ("cohort_assignment", ACTIVE_COHORT_ASSIGNMENT_ID_V1),
            ("cohort_hmac", ACTIVE_COHORT_HMAC_ID_V1),
            ("probability_threshold", ACTIVE_THRESHOLD_ID_V1),
            ("root_outcome_reducer", OUTCOME_REDUCER_ID_V1),
            ("actual_outcome_decay", OUTCOME_DECAY_ID_V1),
            ("outcome_future_skew", LEARNING_FUTURE_SKEW_ID_V1),
            ("outcome_summation", LEARNING_SUMMATION_ID_V1),
            ("beta_quantile_bounds", OUTCOME_QUANTILE_ID_V1),
            ("beta_cdf", OUTCOME_BETA_CDF_ID_V1),
            ("posterior_resolution", OUTCOME_POSTERIOR_RESOLUTION_ID_V1),
            ("max_look_correction", OUTCOME_BONFERRONI_ID_V1),
            ("outcome_attribution_gate", OUTCOME_ATTRIBUTION_GATE_ID_V1),
            ("noninferiority_math", OUTCOME_NONINFERIORITY_MATH_ID_V1),
        ] {
            algorithms.insert(name.into(), json!(identifier));
        }
    }
    Ok(value)
}

fn judge_generation_value(config: &JudgeConfig) -> Result<Json, String> {
    let normalized_config = judge_config_generation_value(config);
    let config_sha256 = canonical_sha256(&normalized_config)?;
    let prompt_template_sha256 = sha256_hex(JUDGE_PROMPT_TEMPLATE_V1);
    let rubric_template_sha256 = sha256_hex(JUDGE_RUBRIC_TEMPLATE_V1);
    let output_schema_sha256 = sha256_hex(JUDGE_OUTPUT_SCHEMA_V1);
    if prompt_template_sha256 != JUDGE_PROMPT_TEMPLATE_SHA256_V1 {
        return Err("compiled judge prompt hash does not match its contract".to_string());
    }
    if rubric_template_sha256 != JUDGE_RUBRIC_TEMPLATE_SHA256_V1 {
        return Err("compiled judge rubric hash does not match its contract".to_string());
    }
    if output_schema_sha256 != JUDGE_OUTPUT_SCHEMA_SHA256_V1 {
        return Err("compiled judge output schema hash does not match its contract".to_string());
    }
    let output_token_limit = config.output_token_limit().ok_or_else(|| {
        "judge max_rationale_bytes cannot be represented by the output-token formula".to_string()
    })?;
    let derived_policy = json!({
        "output_token_limit": output_token_limit,
        "prompt_template_sha256": prompt_template_sha256,
        "rubric_template_sha256": rubric_template_sha256,
        "output_schema_sha256": output_schema_sha256,
        "request_sanitizer_version": ROUTER_SANITIZER_VERSION,
        "trajectory_sanitizer_version": TRAJECTORY_SANITIZER_VERSION,
    });
    let policy_sha256 = canonical_sha256(&json!({
        "config_sha256": config_sha256,
        "derived_policy": derived_policy,
    }))?;

    Ok(json!({
        "config": normalized_config,
        "config_sha256": config_sha256,
        "derived_policy": derived_policy,
        "policy_sha256": policy_sha256,
    }))
}

fn judge_config_generation_value(config: &JudgeConfig) -> Json {
    let mut value = json!({
        "version": config.version,
        "model": config.model,
        "model_revision": config.model_revision,
        "prompt_version": config.prompt_version,
        "rubric_version": config.rubric_version,
        "output_schema_version": config.output_schema_version,
        "response_weight": config.response_weight,
        "trajectory_weight": config.trajectory_weight,
        "response_floor": config.response_floor,
        "trajectory_floor": config.trajectory_floor,
        "judge_confidence_floor": config.judge_confidence_floor,
        "pass_threshold": config.pass_threshold,
        "max_rationale_bytes": config.max_rationale_bytes,
        "base_cooloff_seconds": config.base_cooloff_seconds,
        "max_cooloff_seconds": config.max_cooloff_seconds,
    });
    if let Some(temperature) = config.temperature.filter(|value| *value != 0.0) {
        value
            .as_object_mut()
            .expect("judge generation value must be an object")
            .insert("temperature".to_string(), json!(temperature));
    }
    value
}

fn scheduler_policy_generation_value(pools: &[PoolConfig]) -> Result<Json, String> {
    let mut total_batch_slots = 0usize;
    let mut total_candidate_slots = 0usize;
    for pool in pools {
        let slots = pool_scheduler_slots(pool).ok_or_else(|| {
            format!(
                "derived candidate slots overflow for Router pool '{}'",
                pool.id
            )
        })?;
        total_batch_slots = total_batch_slots
            .checked_add(slots.batch)
            .ok_or_else(|| "aggregate Router batch slots overflow usize".to_string())?;
        total_candidate_slots = total_candidate_slots
            .checked_add(slots.candidates)
            .ok_or_else(|| "aggregate Router candidate slots overflow usize".to_string())?;
    }
    let policy = json!({
        "slot_limit": SCHEDULER_MAX_SLOTS,
        "total_batch_slots": total_batch_slots,
        "total_candidate_slots": total_candidate_slots,
    });
    let policy_sha256 = canonical_sha256(&policy)?;
    Ok(json!({
        "policy": policy,
        "policy_sha256": policy_sha256,
    }))
}

fn selector_generation_value(config: &PoolSelectorConfig) -> Result<Json, String> {
    Ok(json!({
        "tenant_selector": protected_selector_set(
            config.tenant_ids.as_ref(),
            "router-tenant-selector-v1",
        )?,
        "agent_selector": protected_selector_set(
            config.agent_ids.as_ref(),
            "router-agent-selector-v1",
        )?,
        "owner_scope_types": config.owner_scope_types.as_ref().map(|values| {
            let mut sorted = values.iter().map(|value| value.as_str()).collect::<Vec<_>>();
            sorted.sort();
            sorted.dedup();
            sorted
        }),
        "metadata_equals_sha256": canonical_sha256(&json!({
            "schema": "nemo.relay.router.selector-metadata@1",
            "metadata_equals": config.metadata_equals,
        }))?,
        "scope_path_patterns": sorted_unique_strings(config.scope_path_patterns.as_ref()),
    }))
}

fn protected_selector_set(values: Option<&Vec<String>>, domain: &str) -> Result<Json, String> {
    let Some(values) = values else {
        return Ok(json!({"kind": "wildcard"}));
    };
    let values = values
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();
    let count = values.len();
    let sha256 = canonical_sha256(&json!({
        "schema": domain,
        "values": values,
    }))?;
    Ok(json!({
        "kind": "explicit",
        "count": count,
        "sha256": sha256,
    }))
}

fn protected_string_sha256(domain: &str, value: &str) -> String {
    let mut preimage = Vec::with_capacity(domain.len() + value.len() + 1);
    preimage.extend_from_slice(domain.as_bytes());
    preimage.push(0);
    preimage.extend_from_slice(value.as_bytes());
    sha256_hex(&preimage)
}

fn sorted_unique_strings(values: Option<&Vec<String>>) -> Option<Vec<&str>> {
    values.map(|values| {
        values
            .iter()
            .map(String::as_str)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    })
}

fn lookahead_generation_value(config: &LookaheadConfig) -> Json {
    let mut presets = config.lifecycle_presets.clone();
    presets.sort();
    presets.dedup();
    json!({
        "primary_llm_completions": config.primary_llm_completions,
        "deadline_seconds": config.deadline_seconds,
        "lifecycle_presets": presets,
        "max_events_per_window": config.max_events_per_window,
        "max_bytes_per_window": config.max_bytes_per_window,
    })
}

fn candidate_generation_value(config: &CandidateConfig) -> Json {
    json!({
        "id": config.id,
        "model": config.model,
        "model_revision": config.model_revision,
        "cost_rank": config.cost_rank,
        "max_context_tokens": config.max_context_tokens,
        "capabilities": {
            "tools": config.capabilities.tools,
            "multimodal_input": config.capabilities.multimodal_input,
            "structured_output": config.capabilities.structured_output,
            "reasoning_controls": config.capabilities.reasoning_controls,
        },
    })
}

fn canonicalizer_generation_value(config: &CanonicalizerConfig) -> Result<Json, String> {
    let version = canonicalizer_version(config)?;
    let identity: Json = serde_json::from_str(&version.canonical_identity_json)
        .map_err(|error| format!("failed to decode canonicalizer identity: {error}"))?;
    Ok(json!({
        "canonicalizer_version_id": version.canonicalizer_version_id,
        "identity": identity,
    }))
}

const fn default_router_version() -> u32 {
    ROUTER_CONFIG_VERSION
}

fn default_database_path() -> String {
    ".nemo-relay/router/router.db".to_string()
}

const fn default_retention_days() -> u32 {
    30
}

const fn default_max_evidence_records() -> u64 {
    100_000
}

const fn default_embedder_max_in_flight() -> usize {
    4
}

const fn default_embedder_batch_size() -> usize {
    16
}

const fn default_primary_llm_completions() -> usize {
    3
}

const fn default_deadline_seconds() -> u64 {
    300
}

const fn default_max_events_per_window() -> usize {
    512
}

const fn default_max_bytes_per_window() -> usize {
    4 * 1024 * 1024
}

const fn default_max_pending() -> usize {
    32
}

const fn default_canonicalizer_version() -> u32 {
    CANONICALIZER_VERSION
}

const fn default_max_instruction_bytes() -> usize {
    32_768
}

const fn default_max_task_bytes() -> usize {
    16_384
}

const fn default_max_context_messages() -> usize {
    8
}

const fn default_max_context_bytes() -> usize {
    32_768
}

const fn default_max_position_features_bytes() -> usize {
    4_096
}

nemo_relay::editor_config! {
    impl RouterConfig {
        version => { label: "version", kind: Integer },
        mode => { label: "mode", kind: Enum, values: ["off", "shadow", "recommend", "active"] },
        project_id => { label: "project_id", kind: String, optional: true },
        database_path => { label: "database_path", kind: String },
        retention_days => { label: "retention_days", kind: Integer },
        max_evidence_records => { label: "max_evidence_records", kind: Integer },
        allow_remote_embedding_egress => { label: "allow_remote_embedding_egress", kind: Boolean },
        embedders => { label: "embedders", kind: Json },
        pools => { label: "pools", kind: Json },
        policy => { label: "policy", kind: Section, nested: ConfigPolicy, default: ConfigPolicy }
    }
}

nemo_relay::editor_config! {
    impl EmbedderConfig {
        id => { label: "id", kind: String },
        base_url => { label: "base_url", kind: String },
        model => { label: "model", kind: String },
        provider_revision => { label: "provider_revision", kind: String },
        dimensions => { label: "dimensions", kind: Integer },
        api_key_env => { label: "api_key_env", kind: String, optional: true },
        timeout_ms => { label: "timeout_ms", kind: Integer },
        max_in_flight => { label: "max_in_flight", kind: Integer },
        batch_size => { label: "batch_size", kind: Integer }
    }
}

nemo_relay::editor_config! {
    impl PoolConfig {
        id => { label: "id", kind: String },
        api_family => { label: "api_family", kind: Enum, values: ["openai_chat_completions", "openai_responses", "anthropic_messages"] },
        anchor_models => { label: "anchor_models", kind: Json },
        anchor_revision => { label: "anchor_revision", kind: String },
        sampling_probability => { label: "sampling_probability", kind: Float },
        max_candidates_per_sample => { label: "max_candidates_per_sample", kind: Integer },
        selector => { label: "selector", kind: Section, nested: PoolSelectorConfig, default: PoolSelectorConfig },
        lookahead => { label: "lookahead", kind: Section, nested: LookaheadConfig, default: LookaheadConfig },
        concurrency => { label: "concurrency", kind: Section, nested: ConcurrencyConfig },
        candidates => { label: "candidates", kind: Json },
        canonicalizer => { label: "canonicalizer", kind: Section, nested: CanonicalizerConfig, default: CanonicalizerConfig },
        judge => { label: "judge", kind: Section, nested: JudgeConfig },
        learning => { label: "learning", kind: Json, optional: true },
        outcome => { label: "outcome", kind: Json }
    }
}

nemo_relay::editor_config! {
    impl JudgeConfig {
        version => { label: "version", kind: Integer },
        model => { label: "model", kind: String },
        model_revision => { label: "model_revision", kind: String },
        prompt_version => { label: "prompt_version", kind: Enum, values: ["pairwise-equivalence-v1"] },
        rubric_version => { label: "rubric_version", kind: Enum, values: ["response-trajectory-equivalence-v1"] },
        output_schema_version => { label: "output_schema_version", kind: Integer },
        temperature => { label: "temperature", kind: Float, optional: true },
        response_weight => { label: "response_weight", kind: Float },
        trajectory_weight => { label: "trajectory_weight", kind: Float },
        response_floor => { label: "response_floor", kind: Float },
        trajectory_floor => { label: "trajectory_floor", kind: Float },
        judge_confidence_floor => { label: "judge_confidence_floor", kind: Float },
        pass_threshold => { label: "pass_threshold", kind: Float },
        max_rationale_bytes => { label: "max_rationale_bytes", kind: Integer },
        base_cooloff_seconds => { label: "base_cooloff_seconds", kind: Integer },
        max_cooloff_seconds => { label: "max_cooloff_seconds", kind: Integer }
    }
}

nemo_relay::editor_config! {
    impl PoolSelectorConfig {
        tenant_ids => { label: "tenant_ids", kind: Json, optional: true },
        agent_ids => { label: "agent_ids", kind: Json, optional: true },
        owner_scope_types => { label: "owner_scope_types", kind: Json, optional: true },
        metadata_equals => { label: "metadata_equals", kind: Json },
        scope_path_patterns => { label: "scope_path_patterns", kind: Json, optional: true }
    }
}

nemo_relay::editor_config! {
    impl LookaheadConfig {
        primary_llm_completions => { label: "primary_llm_completions", kind: Integer },
        deadline_seconds => { label: "deadline_seconds", kind: Integer },
        lifecycle_presets => { label: "lifecycle_presets", kind: Json },
        max_events_per_window => { label: "max_events_per_window", kind: Integer },
        max_bytes_per_window => { label: "max_bytes_per_window", kind: Integer }
    }
}

nemo_relay::editor_config! {
    impl ConcurrencyConfig {
        shadow => { label: "shadow", kind: Integer },
        judge => { label: "judge", kind: Integer },
        max_pending => { label: "max_pending", kind: Integer }
    }
}

nemo_relay::editor_config! {
    impl CandidateConfig {
        id => { label: "id", kind: String },
        model => { label: "model", kind: String },
        model_revision => { label: "model_revision", kind: String },
        cost_rank => { label: "cost_rank", kind: Integer },
        max_context_tokens => { label: "max_context_tokens", kind: Integer, optional: true },
        capabilities => { label: "capabilities", kind: Section, nested: CandidateCapabilities, default: CandidateCapabilities }
    }
}

nemo_relay::editor_config! {
    impl CandidateCapabilities {
        tools => { label: "tools", kind: Boolean },
        multimodal_input => { label: "multimodal_input", kind: Boolean },
        structured_output => { label: "structured_output", kind: Boolean },
        reasoning_controls => { label: "reasoning_controls", kind: Boolean }
    }
}

nemo_relay::editor_config! {
    impl CanonicalizerConfig {
        version => { label: "version", kind: Integer },
        max_instruction_bytes => { label: "max_instruction_bytes", kind: Integer },
        max_task_bytes => { label: "max_task_bytes", kind: Integer },
        max_context_messages => { label: "max_context_messages", kind: Integer },
        max_context_bytes => { label: "max_context_bytes", kind: Integer },
        max_position_features_bytes => { label: "max_position_features_bytes", kind: Integer },
        position_features => { label: "position_features", kind: Json }
    }
}

#[cfg(test)]
mod learning_policy_tests {
    use super::*;

    fn complete_learning() -> LearningConfig {
        LearningConfig {
            version: 1,
            embedder: "embedding-main".into(),
            top_k: Some(8),
            radius: Some(0.25),
            min_points: Some(3),
            min_independent_roots: Some(2),
            min_effective_samples: Some(1.5),
            min_coverage: Some(0.8),
            time_decay_half_life_seconds: Some(3600.0),
            prior_success: Some(1.0),
            prior_failure: Some(1.0),
            familywise_credible_level: Some(0.95),
            promotion_lower_bound: Some(0.9),
            retention_lower_bound: None,
            holdout_probability: None,
            active_canary_fraction: None,
        }
    }

    fn complete_active_learning() -> LearningConfig {
        LearningConfig {
            retention_lower_bound: Some(0.8),
            holdout_probability: Some(0.2),
            active_canary_fraction: Some(0.4),
            ..complete_learning()
        }
    }

    fn complete_outcome() -> OutcomeConfig {
        serde_json::from_value(json!({
            "version": 1,
            "success_matchers": [{
                "event_kind": "scope_end",
                "category": "agent",
                "name": "completed",
                "terminal_status": "ok",
                "metadata_equals": {"outcome.label": "passed"}
            }],
            "failure_matchers": [{
                "event_kind": "scope_end",
                "category": "agent",
                "name": "completed",
                "terminal_status": "error",
                "metadata_equals": {"outcome.label": "failed"}
            }],
            "completion_disposition": "success",
            "error_disposition": "failure",
            "tool_failure_disposition": "failure",
            "end_of_run_disposition": "ignore",
            "max_attribution_seconds": 600,
            "actual_outcome_half_life_seconds": 1800,
            "anchor_shadow_half_life_seconds": 1800,
            "relearning_cooloff_seconds": 300,
            "min_treatment_roots": 32,
            "min_control_roots": 32,
            "min_treatment_effective_weight": 16.0,
            "min_control_effective_weight": 16.0,
            "noninferiority_margin": 0.1,
            "noninferiority_probability": 0.99,
            "rollback_probability": 0.95,
            "outcome_evaluation_batch_size": 64,
            "max_canary_roots": 64,
            "authorization_ttl_seconds": 600
        }))
        .unwrap()
    }

    #[test]
    fn learning_identity_preserves_minimal_shape_and_versions_every_complete_field() {
        let minimal = LearningConfig::minimal("embedding-main");
        assert_eq!(
            learning_generation_value(&minimal).unwrap(),
            json!({"version": 1, "embedder": "embedding-main"})
        );

        let complete = complete_learning();
        let baseline = learning_generation_value(&complete).unwrap();
        assert_eq!(
            baseline,
            json!({
                "version": 1,
                "embedder": "embedding-main",
                "top_k": 8,
                "radius": 0.25,
                "min_points": 3,
                "min_independent_roots": 2,
                "min_effective_samples": 1.5,
                "min_coverage": 0.8,
                "time_decay_half_life_seconds": 3600.0,
                "prior_success": 1.0,
                "prior_failure": 1.0,
                "familywise_credible_level": 0.95,
                "promotion_lower_bound": 0.9,
                "algorithms": {
                    "similarity_kernel": "linear_radius_v1",
                    "root_selector": "min_distance_newest_evaluation_lex_id_v1",
                    "coverage": "labeled_roots_over_attempted_roots_v1",
                    "time_decay": "exp2_half_life_v1",
                    "future_skew": "future_skew_300000ms_v1",
                    "summation": "neumaier_f64_no_fma_v1",
                    "effective_sample": "kish_effective_sample_v1",
                    "beta_approximation": "beta_effective_sample_v1",
                    "candidate_correction": "bonferroni_v1",
                    "beta_inverse_cdf": "statrs_0_18_0_v1",
                    "candidate_set": "candidate_set_v1",
                },
            })
        );
        assert_eq!(
            baseline["algorithms"],
            json!({
                "similarity_kernel": "linear_radius_v1",
                "root_selector": "min_distance_newest_evaluation_lex_id_v1",
                "coverage": "labeled_roots_over_attempted_roots_v1",
                "time_decay": "exp2_half_life_v1",
                "future_skew": "future_skew_300000ms_v1",
                "summation": "neumaier_f64_no_fma_v1",
                "effective_sample": "kish_effective_sample_v1",
                "beta_approximation": "beta_effective_sample_v1",
                "candidate_correction": "bonferroni_v1",
                "beta_inverse_cdf": "statrs_0_18_0_v1",
                "candidate_set": "candidate_set_v1",
            })
        );
        assert_eq!(LEARNING_BETA_INVERSE_CDF_ID_V1, "statrs_0_18_0_v1");

        let baseline_id = canonical_sha256(&baseline).unwrap();
        assert_eq!(
            baseline_id,
            "d6fdf2521a70760bf78c1200fe31f5b43c325233e374dec4da8d091ef30d9457"
        );
        let mut variants = Vec::new();
        let mut changed = complete.clone();
        changed.top_k = Some(9);
        variants.push(changed);
        let mut changed = complete.clone();
        changed.radius = Some(0.5);
        variants.push(changed);
        let mut changed = complete.clone();
        changed.min_points = Some(4);
        variants.push(changed);
        let mut changed = complete.clone();
        changed.min_independent_roots = Some(3);
        variants.push(changed);
        let mut changed = complete.clone();
        changed.min_effective_samples = Some(2.0);
        variants.push(changed);
        let mut changed = complete.clone();
        changed.min_coverage = Some(0.75);
        variants.push(changed);
        let mut changed = complete.clone();
        changed.time_decay_half_life_seconds = Some(7200.0);
        variants.push(changed);
        let mut changed = complete.clone();
        changed.prior_success = Some(2.0);
        variants.push(changed);
        let mut changed = complete.clone();
        changed.prior_failure = Some(2.0);
        variants.push(changed);
        let mut changed = complete.clone();
        changed.familywise_credible_level = Some(0.99);
        variants.push(changed);
        let mut changed = complete;
        changed.promotion_lower_bound = Some(0.85);
        variants.push(changed);

        assert!(variants.into_iter().all(|variant| {
            canonical_sha256(&learning_generation_value(&variant).unwrap()).unwrap() != baseline_id
        }));
    }

    #[test]
    fn active_learning_identity_versions_every_new_field_and_algorithm() {
        let active = complete_active_learning();
        let baseline = learning_generation_value(&active).unwrap();
        assert_eq!(baseline["retention_lower_bound"], json!(0.8));
        assert_eq!(baseline["holdout_probability"], json!(0.2));
        assert_eq!(baseline["active_canary_fraction"], json!(0.4));
        for (field, identifier) in [
            ("cohort_assignment", ACTIVE_COHORT_ASSIGNMENT_ID_V1),
            ("cohort_hmac", ACTIVE_COHORT_HMAC_ID_V1),
            ("probability_threshold", ACTIVE_THRESHOLD_ID_V1),
            ("root_outcome_reducer", OUTCOME_REDUCER_ID_V1),
            ("actual_outcome_decay", OUTCOME_DECAY_ID_V1),
            ("outcome_future_skew", LEARNING_FUTURE_SKEW_ID_V1),
            ("outcome_summation", LEARNING_SUMMATION_ID_V1),
            ("beta_quantile_bounds", OUTCOME_QUANTILE_ID_V1),
            ("beta_cdf", OUTCOME_BETA_CDF_ID_V1),
            ("posterior_resolution", OUTCOME_POSTERIOR_RESOLUTION_ID_V1),
            ("max_look_correction", OUTCOME_BONFERRONI_ID_V1),
            ("outcome_attribution_gate", OUTCOME_ATTRIBUTION_GATE_ID_V1),
            ("noninferiority_math", OUTCOME_NONINFERIORITY_MATH_ID_V1),
        ] {
            assert_eq!(baseline["algorithms"][field], json!(identifier));
        }

        let baseline_hash = canonical_sha256(&baseline).unwrap();
        let mut variants = Vec::new();
        let mut changed = active.clone();
        changed.retention_lower_bound = Some(0.7);
        variants.push(changed);
        let mut changed = active.clone();
        changed.holdout_probability = Some(0.1);
        variants.push(changed);
        let mut changed = active;
        changed.active_canary_fraction = Some(0.3);
        variants.push(changed);
        assert!(variants.into_iter().all(|variant| {
            canonical_sha256(&learning_generation_value(&variant).unwrap()).unwrap()
                != baseline_hash
        }));
    }

    #[test]
    fn exact_probability_threshold_uses_binary64_bits_without_float_roundtrip() {
        assert_eq!(exact_probability_threshold(0.5), Some(1_u128 << 63));
        assert_eq!(exact_probability_threshold(0.25), Some(1_u128 << 62));
        assert_eq!(exact_probability_threshold(2.0_f64.powi(-64)), Some(1));
        assert_eq!(
            exact_probability_threshold(f64::from_bits(1.0_f64.to_bits() - 1)),
            Some((1_u128 << 64) - 2_048)
        );
        for probability in [0.0, 1.0, f64::NAN, f64::INFINITY, f64::from_bits(1)] {
            assert_eq!(exact_probability_threshold(probability), None);
        }
    }

    #[test]
    fn protected_outcome_identity_has_pinned_domains_and_no_expected_scalars() {
        let outcome = complete_outcome();
        let identity = outcome_policy_identity(&outcome).unwrap();
        let serialized = canonical_json(&identity.protected_value).unwrap();
        assert!(!serialized.contains("passed"));
        assert!(!serialized.contains("failed"));
        assert!(!serialized.contains("outcome.label"));
        assert_eq!(identity.outcome_policy_hash.len(), 64);
        assert!(
            identity
                .outcome_policy_hash
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        );
        assert_eq!(
            identity.outcome_policy_hash,
            "7325f50ba50bc031402da88e536f537f86fa16ded6bc5c8291d8614606f9e6ad"
        );

        let integer = length_prefixed_sha256(
            OUTCOME_EXPECTED_SCALAR_HASH_DOMAIN_V1,
            canonical_json(&json!(1)).unwrap().as_bytes(),
        )
        .unwrap();
        let float = length_prefixed_sha256(
            OUTCOME_EXPECTED_SCALAR_HASH_DOMAIN_V1,
            canonical_json(&json!(1.0)).unwrap().as_bytes(),
        )
        .unwrap();
        assert_eq!(integer, float);
        assert_eq!(
            OUTCOME_OBSERVED_SCALAR_HASH_DOMAIN_V1,
            b"nemo-relay-router/outcome-observed-scalar/v1\0"
        );
    }

    #[test]
    fn protected_outcome_identity_versions_every_semantic_field() {
        let baseline = complete_outcome();
        let baseline_hash = outcome_policy_identity(&baseline)
            .unwrap()
            .outcome_policy_hash;
        let mut variants = Vec::new();

        macro_rules! variant {
            ($body:expr) => {{
                let mut changed = baseline.clone();
                $body(&mut changed);
                variants.push(changed);
            }};
        }

        variant!(|value: &mut OutcomeConfig| value.version = 2);
        variant!(|value: &mut OutcomeConfig| {
            value.success_matchers[0].event_kind = OutcomeMatcherEventKind::Mark
        });
        variant!(|value: &mut OutcomeConfig| value.success_matchers[0]
            .category
            .push_str("-changed"));
        variant!(|value: &mut OutcomeConfig| value.success_matchers[0].name.push_str("-changed"));
        variant!(|value: &mut OutcomeConfig| {
            value.success_matchers[0].terminal_status = OutcomeTerminalStatus::Unset
        });
        variant!(|value: &mut OutcomeConfig| {
            let expected = value.success_matchers[0]
                .metadata_equals
                .remove("outcome.label")
                .unwrap();
            value.success_matchers[0]
                .metadata_equals
                .insert("status".into(), expected);
        });
        variant!(|value: &mut OutcomeConfig| {
            *value.success_matchers[0]
                .metadata_equals
                .get_mut("outcome.label")
                .unwrap() = json!("accepted")
        });
        variant!(|value: &mut OutcomeConfig| {
            std::mem::swap(&mut value.success_matchers, &mut value.failure_matchers)
        });
        variant!(
            |value: &mut OutcomeConfig| value.completion_disposition = OutcomeDisposition::Ignore
        );
        variant!(|value: &mut OutcomeConfig| value.error_disposition = OutcomeDisposition::Ignore);
        variant!(
            |value: &mut OutcomeConfig| value.tool_failure_disposition = OutcomeDisposition::Ignore
        );
        variant!(
            |value: &mut OutcomeConfig| value.end_of_run_disposition = OutcomeDisposition::Failure
        );
        variant!(|value: &mut OutcomeConfig| value.max_attribution_seconds += 1);
        variant!(|value: &mut OutcomeConfig| value.actual_outcome_half_life_seconds += 1);
        variant!(|value: &mut OutcomeConfig| value.anchor_shadow_half_life_seconds += 1);
        variant!(|value: &mut OutcomeConfig| value.relearning_cooloff_seconds += 1);
        variant!(|value: &mut OutcomeConfig| value.min_treatment_roots += 1);
        variant!(|value: &mut OutcomeConfig| value.min_control_roots += 1);
        variant!(|value: &mut OutcomeConfig| value.min_treatment_effective_weight += 1.0);
        variant!(|value: &mut OutcomeConfig| value.min_control_effective_weight += 1.0);
        variant!(|value: &mut OutcomeConfig| value.noninferiority_margin += 0.01);
        variant!(|value: &mut OutcomeConfig| value.noninferiority_probability = 0.995);
        variant!(|value: &mut OutcomeConfig| value.rollback_probability = 0.96);
        variant!(|value: &mut OutcomeConfig| value.outcome_evaluation_batch_size = 128);
        variant!(|value: &mut OutcomeConfig| value.max_canary_roots = 128);
        variant!(|value: &mut OutcomeConfig| value.authorization_ttl_seconds -= 1);

        assert!(variants.into_iter().all(|variant| {
            outcome_policy_identity(&variant)
                .unwrap()
                .outcome_policy_hash
                != baseline_hash
        }));
    }

    #[test]
    fn prior_only_and_nonfinite_guards_are_independently_exercised() {
        let mut diagnostics = Vec::new();
        validate_prior_only_separation(0.25, 0.70, 1, "outcome", &mut diagnostics);
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.field.as_deref() == Some("outcome.noninferiority_probability")
        }));

        let mut diagnostics = Vec::new();
        validate_effective_weight(f64::NAN, 32, 64, "weight", &mut diagnostics);
        validate_probability_range(f64::INFINITY, 0.99, "probability", &mut diagnostics);
        assert_eq!(diagnostics.len(), 2);
    }
}
