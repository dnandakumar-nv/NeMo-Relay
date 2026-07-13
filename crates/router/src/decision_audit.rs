// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Immutable, repository-independent Spec 07 decision audit graphs.

use std::collections::{BTreeMap, BTreeSet};

use serde::Serialize;
use serde_json::{Value as Json, json};
use unicode_normalization::UnicodeNormalization;
use uuid::{Uuid, Variant};

use crate::candidate_set::{CandidateSetMemberInputV1, build_candidate_set_from_members_v1};
use crate::canonical_json::canonical_json;
use crate::canonical_query::{CanonicalRoutingQueryArtifactV1, CanonicalRoutingQueryV1};
use crate::confidence::beta_inverse_cdf_v1;
use crate::config::{CANDIDATE_ID_MAX_BYTES, ID_MAX_BYTES, MODEL_ID_MAX_BYTES, REVISION_MAX_BYTES};
use crate::embedding_identity::CANONICAL_ROUTING_QUERY_SCHEMA_V1;
use crate::fingerprint::{canonical_json_bytes, sha256_hex};
use crate::routing_partition::{RoutingPartitionArtifactV1, RoutingPartitionBaseV1};

pub(crate) const DECISION_SHAPE_VERSION_V1: u32 = 1;
pub(crate) const DECISION_ALGORITHM_VERSION_V1: u32 = 1;
pub(crate) const ACTIVE_DECISION_SHAPE_VERSION_V2: u32 = 2;
pub(crate) const ACTIVE_DECISION_ALGORITHM_VERSION_V2: u32 = 2;
pub(crate) const DECISION_AUDIT_BYTES_MAX: usize = 32 * 1024 * 1024;
pub(crate) const DECISION_SUMMARY_COUNT_MAX: usize = 64;
pub(crate) const DECISION_NEIGHBOR_COUNT_MAX: usize = 4_095;

const DECISION_PARENT_HASH_SCHEMA_V1: &str = "nemo.relay.router.decision-parent-hash@1";
const DECISION_SUMMARY_HASH_SCHEMA_V1: &str = "nemo.relay.router.decision-summary-hash@1";
const DECISION_NEIGHBOR_HASH_SCHEMA_V1: &str = "nemo.relay.router.decision-neighbor-hash@1";
const DECISION_SUMMARY_AGGREGATE_SCHEMA_V1: &str = "nemo.relay.router.decision-summary-aggregate@1";
const DECISION_NEIGHBOR_AGGREGATE_SCHEMA_V1: &str =
    "nemo.relay.router.decision-neighbor-aggregate@1";
const DECISION_CANDIDATE_NEIGHBOR_AGGREGATE_SCHEMA_V1: &str =
    "nemo.relay.router.decision-candidate-neighbor-aggregate@1";
const PARTITION_TRANSPORT_IDENTITY_MAX_BYTES: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AuditF64V1(u64);

impl AuditF64V1 {
    pub(crate) fn new(value: f64) -> Result<Self, DecisionAuditError> {
        Self::from_bits(value.to_bits())
    }

    pub(crate) fn from_bits(bits: u64) -> Result<Self, DecisionAuditError> {
        f64::from_bits(bits)
            .is_finite()
            .then_some(Self(bits))
            .ok_or(DecisionAuditError::InvalidFloat)
    }

    pub(crate) fn from_stored(value: f64, bits: u64) -> Result<Self, DecisionAuditError> {
        let audited = Self::from_bits(bits)?;
        (audited.value().to_bits() == value.to_bits())
            .then_some(audited)
            .ok_or(DecisionAuditError::FloatBitsMismatch)
    }

    pub(crate) fn value(self) -> f64 {
        f64::from_bits(self.0)
    }

    pub(crate) fn bits(self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct AuditF32V1(u32);

impl AuditF32V1 {
    pub(crate) fn new(value: f32) -> Result<Self, DecisionAuditError> {
        Self::from_bits(value.to_bits())
    }

    pub(crate) fn from_bits(bits: u32) -> Result<Self, DecisionAuditError> {
        f32::from_bits(bits)
            .is_finite()
            .then_some(Self(bits))
            .ok_or(DecisionAuditError::InvalidFloat)
    }

    pub(crate) fn from_stored(value: f64, bits: u32) -> Result<Self, DecisionAuditError> {
        let audited = Self::from_bits(bits)?;
        (f64::from(audited.value()).to_bits() == value.to_bits())
            .then_some(audited)
            .ok_or(DecisionAuditError::FloatBitsMismatch)
    }

    pub(crate) fn value(self) -> f32 {
        f32::from_bits(self.0)
    }

    pub(crate) fn bits(self) -> u32 {
        self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DecisionFinalReasonV1 {
    EmbeddingUnavailable,
    VectorUnhealthy,
    VersionMismatch,
    NoPartition,
    SparsePoints,
    InsufficientRoots,
    LowCoverage,
    InsufficientEffectiveSamples,
    LowerBoundBelowThreshold,
    InvalidEvidenceTime,
    NumericError,
    NoCandidatePassed,
    RecommendObserveOnly,
    ActiveCandidate,
    ActiveAnchorControl,
    ActiveAnchorHoldout,
    ActiveForceAnchor,
    ActivePaused,
    ActiveIneligible,
    ActiveExhausted,
    ActiveStorageFallback,
    ActiveAuthorizationStale,
    ActiveCapReached,
}

impl DecisionFinalReasonV1 {
    pub(crate) const fn is_active(self) -> bool {
        matches!(
            self,
            Self::ActiveCandidate
                | Self::ActiveAnchorControl
                | Self::ActiveAnchorHoldout
                | Self::ActiveForceAnchor
                | Self::ActivePaused
                | Self::ActiveIneligible
                | Self::ActiveExhausted
                | Self::ActiveStorageFallback
                | Self::ActiveAuthorizationStale
                | Self::ActiveCapReached
        )
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::EmbeddingUnavailable => "embedding_unavailable",
            Self::VectorUnhealthy => "vector_unhealthy",
            Self::VersionMismatch => "version_mismatch",
            Self::NoPartition => "no_partition",
            Self::SparsePoints => "sparse_points",
            Self::InsufficientRoots => "insufficient_roots",
            Self::LowCoverage => "low_coverage",
            Self::InsufficientEffectiveSamples => "insufficient_effective_samples",
            Self::LowerBoundBelowThreshold => "lower_bound_below_threshold",
            Self::InvalidEvidenceTime => "invalid_evidence_time",
            Self::NumericError => "numeric_error",
            Self::NoCandidatePassed => "no_candidate_passed",
            Self::RecommendObserveOnly => "recommend_observe_only",
            Self::ActiveCandidate => "active_candidate",
            Self::ActiveAnchorControl => "active_anchor_control",
            Self::ActiveAnchorHoldout => "active_anchor_holdout",
            Self::ActiveForceAnchor => "active_force_anchor",
            Self::ActivePaused => "active_paused",
            Self::ActiveIneligible => "active_ineligible",
            Self::ActiveExhausted => "active_exhausted",
            Self::ActiveStorageFallback => "active_storage_fallback",
            Self::ActiveAuthorizationStale => "active_authorization_stale",
            Self::ActiveCapReached => "active_cap_reached",
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self, DecisionAuditError> {
        match value {
            "embedding_unavailable" => Ok(Self::EmbeddingUnavailable),
            "vector_unhealthy" => Ok(Self::VectorUnhealthy),
            "version_mismatch" => Ok(Self::VersionMismatch),
            "no_partition" => Ok(Self::NoPartition),
            "sparse_points" => Ok(Self::SparsePoints),
            "insufficient_roots" => Ok(Self::InsufficientRoots),
            "low_coverage" => Ok(Self::LowCoverage),
            "insufficient_effective_samples" => Ok(Self::InsufficientEffectiveSamples),
            "lower_bound_below_threshold" => Ok(Self::LowerBoundBelowThreshold),
            "invalid_evidence_time" => Ok(Self::InvalidEvidenceTime),
            "numeric_error" => Ok(Self::NumericError),
            "no_candidate_passed" => Ok(Self::NoCandidatePassed),
            "recommend_observe_only" => Ok(Self::RecommendObserveOnly),
            "active_candidate" => Ok(Self::ActiveCandidate),
            "active_anchor_control" => Ok(Self::ActiveAnchorControl),
            "active_anchor_holdout" => Ok(Self::ActiveAnchorHoldout),
            "active_force_anchor" => Ok(Self::ActiveForceAnchor),
            "active_paused" => Ok(Self::ActivePaused),
            "active_ineligible" => Ok(Self::ActiveIneligible),
            "active_exhausted" => Ok(Self::ActiveExhausted),
            "active_storage_fallback" => Ok(Self::ActiveStorageFallback),
            "active_authorization_stale" => Ok(Self::ActiveAuthorizationStale),
            "active_cap_reached" => Ok(Self::ActiveCapReached),
            _ => Err(DecisionAuditError::InvalidState),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DecisionCandidateReasonV1 {
    NoPartition,
    SparsePoints,
    InsufficientRoots,
    LowCoverage,
    InsufficientEffectiveSamples,
    LowerBoundBelowThreshold,
    InvalidEvidenceTime,
    NumericError,
    Passed,
    NotEvaluatedAfterWinner,
    NotEvaluatedAfterFallback,
}

impl DecisionCandidateReasonV1 {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::NoPartition => "no_partition",
            Self::SparsePoints => "sparse_points",
            Self::InsufficientRoots => "insufficient_roots",
            Self::LowCoverage => "low_coverage",
            Self::InsufficientEffectiveSamples => "insufficient_effective_samples",
            Self::LowerBoundBelowThreshold => "lower_bound_below_threshold",
            Self::InvalidEvidenceTime => "invalid_evidence_time",
            Self::NumericError => "numeric_error",
            Self::Passed => "passed",
            Self::NotEvaluatedAfterWinner => "not_evaluated_after_winner",
            Self::NotEvaluatedAfterFallback => "not_evaluated_after_fallback",
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self, DecisionAuditError> {
        match value {
            "no_partition" => Ok(Self::NoPartition),
            "sparse_points" => Ok(Self::SparsePoints),
            "insufficient_roots" => Ok(Self::InsufficientRoots),
            "low_coverage" => Ok(Self::LowCoverage),
            "insufficient_effective_samples" => Ok(Self::InsufficientEffectiveSamples),
            "lower_bound_below_threshold" => Ok(Self::LowerBoundBelowThreshold),
            "invalid_evidence_time" => Ok(Self::InvalidEvidenceTime),
            "numeric_error" => Ok(Self::NumericError),
            "passed" => Ok(Self::Passed),
            "not_evaluated_after_winner" => Ok(Self::NotEvaluatedAfterWinner),
            "not_evaluated_after_fallback" => Ok(Self::NotEvaluatedAfterFallback),
            _ => Err(DecisionAuditError::InvalidState),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DecisionBinaryLabelV1 {
    Pass,
    Fail,
}

impl DecisionBinaryLabelV1 {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self, DecisionAuditError> {
        match value {
            "pass" => Ok(Self::Pass),
            "fail" => Ok(Self::Fail),
            _ => Err(DecisionAuditError::InvalidState),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DecisionNeighborExclusionReasonV1 {
    OutsideRadius,
    IneligibleQuality,
    DuplicateRoot,
    Included,
}

impl DecisionNeighborExclusionReasonV1 {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::OutsideRadius => "outside_radius",
            Self::IneligibleQuality => "ineligible_quality",
            Self::DuplicateRoot => "duplicate_root",
            Self::Included => "included",
        }
    }

    pub(crate) fn parse(value: &str) -> Result<Self, DecisionAuditError> {
        match value {
            "outside_radius" => Ok(Self::OutsideRadius),
            "ineligible_quality" => Ok(Self::IneligibleQuality),
            "duplicate_root" => Ok(Self::DuplicateRoot),
            "included" => Ok(Self::Included),
            _ => Err(DecisionAuditError::InvalidState),
        }
    }
}

#[derive(PartialEq, Eq)]
pub(crate) struct PreparedCanonicalQueryV1 {
    artifact: CanonicalRoutingQueryArtifactV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecisionParentInputV1 {
    pub(crate) decision_id: Uuid,
    pub(crate) project_uuid: Uuid,
    pub(crate) process_instance_id: Uuid,
    pub(crate) config_generation_id: String,
    pub(crate) policy_version_id: String,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) pool_id: String,
    pub(crate) candidate_id: Option<String>,
    pub(crate) primary_call_uuid: Uuid,
    pub(crate) canonical_query_hash: String,
    pub(crate) partition_base_json: String,
    pub(crate) partition_base_hash: String,
    pub(crate) vector_space_id: String,
    pub(crate) candidate_set_hash: String,
    pub(crate) recommended_model: String,
    pub(crate) recommended_model_revision: String,
    pub(crate) served_model: String,
    pub(crate) served_model_revision: String,
    pub(crate) as_of_unix_ms: i64,
    pub(crate) decision_latency_ms: u64,
    pub(crate) final_reason: DecisionFinalReasonV1,
    pub(crate) created_at_unix_ms: i64,
}

/// Active-only immutable authority bound into a shape-2 decision parent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ActiveDecisionParentBindingV2 {
    pub(crate) cohort_generation_id: Uuid,
    pub(crate) active_experiment_id: Option<Uuid>,
    pub(crate) active_authorization_state_event_id: Option<Uuid>,
    pub(crate) root_key: String,
}

enum DecisionParentBinding {
    Recommend,
    Active(ActiveDecisionParentBindingV2),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecisionCandidateSummaryInputV1 {
    pub(crate) candidate_id: String,
    pub(crate) rank_ordinal: usize,
    pub(crate) candidate_model: String,
    pub(crate) candidate_model_revision: String,
    pub(crate) cost_rank: u32,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) vector_space_id: String,
    pub(crate) partition_id: Option<i64>,
    pub(crate) decoding_fingerprint: String,
    pub(crate) top_k: usize,
    pub(crate) radius: AuditF64V1,
    pub(crate) min_points: usize,
    pub(crate) min_independent_roots: usize,
    pub(crate) min_effective_samples: AuditF64V1,
    pub(crate) min_coverage: AuditF64V1,
    pub(crate) time_decay_half_life_seconds: AuditF64V1,
    pub(crate) prior_success: AuditF64V1,
    pub(crate) prior_failure: AuditF64V1,
    pub(crate) familywise_credible_level: AuditF64V1,
    pub(crate) candidate_alpha: AuditF64V1,
    pub(crate) promotion_lower_bound: AuditF64V1,
    pub(crate) returned_neighbor_count: usize,
    pub(crate) within_radius_count: usize,
    pub(crate) labeled_point_count: usize,
    pub(crate) attempted_root_count: usize,
    pub(crate) labeled_root_count: usize,
    pub(crate) selected_root_count: usize,
    pub(crate) coverage: Option<AuditF64V1>,
    pub(crate) sum_weight: Option<AuditF64V1>,
    pub(crate) sum_weighted_label: Option<AuditF64V1>,
    pub(crate) sum_squared_weight: Option<AuditF64V1>,
    pub(crate) p_hat: Option<AuditF64V1>,
    pub(crate) effective_sample_size: Option<AuditF64V1>,
    pub(crate) beta_alpha: Option<AuditF64V1>,
    pub(crate) beta_beta: Option<AuditF64V1>,
    pub(crate) lower_bound: Option<AuditF64V1>,
    pub(crate) partition_gate_passed: Option<bool>,
    pub(crate) points_gate_passed: Option<bool>,
    pub(crate) roots_gate_passed: Option<bool>,
    pub(crate) coverage_gate_passed: Option<bool>,
    pub(crate) weight_gate_passed: Option<bool>,
    pub(crate) effective_samples_gate_passed: Option<bool>,
    pub(crate) beta_quantile_gate_passed: Option<bool>,
    pub(crate) lower_bound_gate_passed: Option<bool>,
    pub(crate) terminal_reason: DecisionCandidateReasonV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecisionNeighborInputV1 {
    pub(crate) neighbor_ordinal: usize,
    pub(crate) candidate_id: String,
    pub(crate) candidate_neighbor_ordinal: usize,
    pub(crate) evidence_vector_link_id: Uuid,
    pub(crate) shadow_attempt_id: Uuid,
    pub(crate) anchor_id: Uuid,
    pub(crate) evaluation_id: Option<Uuid>,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) distance: AuditF32V1,
    pub(crate) age_millis: Option<u64>,
    pub(crate) similarity_weight: Option<AuditF64V1>,
    pub(crate) time_weight: Option<AuditF64V1>,
    pub(crate) final_weight: Option<AuditF64V1>,
    pub(crate) binary_label: Option<DecisionBinaryLabelV1>,
    pub(crate) selected_for_root: bool,
    pub(crate) root_group_ordinal: usize,
    pub(crate) exclusion_reason: DecisionNeighborExclusionReasonV1,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecisionCandidateInputV1 {
    pub(crate) summary: DecisionCandidateSummaryInputV1,
    pub(crate) partition_artifact: RoutingPartitionArtifactV1,
    pub(crate) neighbors: Vec<DecisionNeighborInputV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecisionRecordV1 {
    pub(crate) decision_id: Uuid,
    pub(crate) decision_shape_version: u32,
    pub(crate) algorithm_version: u32,
    pub(crate) project_uuid: Uuid,
    pub(crate) process_instance_id: Uuid,
    pub(crate) config_generation_id: String,
    pub(crate) policy_version_id: String,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) cohort_generation_id: Option<Uuid>,
    pub(crate) active_experiment_id: Option<Uuid>,
    pub(crate) active_authorization_state_event_id: Option<Uuid>,
    pub(crate) pool_id: String,
    pub(crate) candidate_id: Option<String>,
    pub(crate) root_key: Option<String>,
    pub(crate) primary_call_uuid: Uuid,
    pub(crate) canonical_query_hash: String,
    pub(crate) partition_base_json: String,
    pub(crate) partition_base_hash: String,
    pub(crate) vector_space_id: String,
    pub(crate) candidate_set_hash: String,
    pub(crate) candidate_count: usize,
    pub(crate) recommended_model: String,
    pub(crate) recommended_model_revision: String,
    pub(crate) served_model: String,
    pub(crate) served_model_revision: String,
    pub(crate) as_of_unix_ms: i64,
    pub(crate) decision_latency_ms: u64,
    pub(crate) final_reason: DecisionFinalReasonV1,
    pub(crate) summary_count: usize,
    pub(crate) summary_aggregate_hash: String,
    pub(crate) neighbor_count: usize,
    pub(crate) neighbor_aggregate_hash: String,
    pub(crate) aggregate_size_bytes: usize,
    pub(crate) created_at_unix_ms: i64,
    pub(crate) canonical_payload_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecisionCandidateSummaryV1 {
    pub(crate) decision_id: Uuid,
    pub(crate) candidate_id: String,
    pub(crate) rank_ordinal: usize,
    pub(crate) candidate_model: String,
    pub(crate) candidate_model_revision: String,
    pub(crate) cost_rank: u32,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) vector_space_id: String,
    pub(crate) partition_hash: String,
    pub(crate) partition_id: Option<i64>,
    pub(crate) decoding_fingerprint: String,
    pub(crate) top_k: usize,
    pub(crate) radius: AuditF64V1,
    pub(crate) min_points: usize,
    pub(crate) min_independent_roots: usize,
    pub(crate) min_effective_samples: AuditF64V1,
    pub(crate) min_coverage: AuditF64V1,
    pub(crate) time_decay_half_life_seconds: AuditF64V1,
    pub(crate) prior_success: AuditF64V1,
    pub(crate) prior_failure: AuditF64V1,
    pub(crate) familywise_credible_level: AuditF64V1,
    pub(crate) candidate_alpha: AuditF64V1,
    pub(crate) promotion_lower_bound: AuditF64V1,
    pub(crate) returned_neighbor_count: usize,
    pub(crate) within_radius_count: usize,
    pub(crate) labeled_point_count: usize,
    pub(crate) attempted_root_count: usize,
    pub(crate) labeled_root_count: usize,
    pub(crate) selected_root_count: usize,
    pub(crate) coverage: Option<AuditF64V1>,
    pub(crate) sum_weight: Option<AuditF64V1>,
    pub(crate) sum_weighted_label: Option<AuditF64V1>,
    pub(crate) sum_squared_weight: Option<AuditF64V1>,
    pub(crate) p_hat: Option<AuditF64V1>,
    pub(crate) effective_sample_size: Option<AuditF64V1>,
    pub(crate) beta_alpha: Option<AuditF64V1>,
    pub(crate) beta_beta: Option<AuditF64V1>,
    pub(crate) lower_bound: Option<AuditF64V1>,
    pub(crate) partition_gate_passed: Option<bool>,
    pub(crate) points_gate_passed: Option<bool>,
    pub(crate) roots_gate_passed: Option<bool>,
    pub(crate) coverage_gate_passed: Option<bool>,
    pub(crate) weight_gate_passed: Option<bool>,
    pub(crate) effective_samples_gate_passed: Option<bool>,
    pub(crate) beta_quantile_gate_passed: Option<bool>,
    pub(crate) lower_bound_gate_passed: Option<bool>,
    pub(crate) terminal_reason: DecisionCandidateReasonV1,
    pub(crate) neighbor_count: usize,
    pub(crate) neighbor_aggregate_hash: String,
    pub(crate) canonical_payload_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct DecisionNeighborV1 {
    pub(crate) decision_id: Uuid,
    pub(crate) neighbor_ordinal: usize,
    pub(crate) candidate_id: String,
    pub(crate) candidate_neighbor_ordinal: usize,
    pub(crate) evidence_vector_link_id: Uuid,
    pub(crate) shadow_attempt_id: Uuid,
    pub(crate) anchor_id: Uuid,
    pub(crate) evaluation_id: Option<Uuid>,
    pub(crate) learning_generation_id: Uuid,
    pub(crate) distance: AuditF32V1,
    pub(crate) age_seconds: Option<AuditF64V1>,
    pub(crate) similarity_weight: Option<AuditF64V1>,
    pub(crate) time_weight: Option<AuditF64V1>,
    pub(crate) final_weight: Option<AuditF64V1>,
    pub(crate) binary_label: Option<DecisionBinaryLabelV1>,
    pub(crate) selected_for_root: bool,
    pub(crate) root_group_ordinal: usize,
    pub(crate) exclusion_reason: DecisionNeighborExclusionReasonV1,
    pub(crate) canonical_payload_hash: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StoredDecisionGraphV1 {
    pub(crate) parent: DecisionRecordV1,
    pub(crate) summaries: Vec<DecisionCandidateSummaryV1>,
    pub(crate) neighbors: Vec<DecisionNeighborV1>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct VerifiedStoredDecisionGraphV1 {
    pub(crate) graph: StoredDecisionGraphV1,
}

#[derive(PartialEq, Eq)]
pub(crate) struct DecisionAuditV1 {
    pub(crate) parent: DecisionRecordV1,
    pub(crate) summaries: Vec<DecisionCandidateSummaryV1>,
    pub(crate) neighbors: Vec<DecisionNeighborV1>,
    pub(crate) prepared_query: PreparedCanonicalQueryV1,
    pub(crate) partition_artifacts: Vec<RoutingPartitionArtifactV1>,
    pub(crate) command_size_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DecisionAuditError {
    InvalidFloat,
    FloatBitsMismatch,
    InvalidIdentity,
    InvalidText,
    InvalidCanonicalDocument,
    InvalidOrdering,
    InvalidCounts,
    InvalidState,
    DuplicateIdentity,
    HashMismatch,
    SizeOverflow,
    SizeLimitExceeded,
}

impl PreparedCanonicalQueryV1 {
    pub(crate) fn new(
        canonical_query_hash: impl Into<String>,
        canonical_query_json: impl Into<String>,
    ) -> Result<Self, DecisionAuditError> {
        let canonical_query_hash = canonical_query_hash.into();
        let canonical_query_json = canonical_query_json.into();
        let query: CanonicalRoutingQueryV1 = serde_json::from_str(&canonical_query_json)
            .map_err(|_| DecisionAuditError::InvalidCanonicalDocument)?;
        let artifact = CanonicalRoutingQueryArtifactV1 {
            query,
            canonical_bytes: canonical_query_json.into_bytes(),
            canonical_query_hash,
        };
        Self::from_artifact(artifact)
    }

    pub(crate) fn from_artifact(
        artifact: CanonicalRoutingQueryArtifactV1,
    ) -> Result<Self, DecisionAuditError> {
        if artifact.query.schema != CANONICAL_ROUTING_QUERY_SCHEMA_V1 {
            return Err(DecisionAuditError::InvalidCanonicalDocument);
        }
        crate::ledger::repository::vector_catalog::validate_query_artifact(&artifact)
            .map_err(|_| DecisionAuditError::InvalidCanonicalDocument)?;
        let prepared = Self { artifact };
        validate_canonical_document(
            prepared.canonical_query_hash(),
            prepared.canonical_query_json(),
        )?;
        Ok(prepared)
    }

    pub(crate) fn canonical_query_hash(&self) -> &str {
        &self.artifact.canonical_query_hash
    }

    pub(crate) fn canonical_query_json(&self) -> &str {
        std::str::from_utf8(&self.artifact.canonical_bytes)
            .expect("validated canonical routing query is UTF-8")
    }

    pub(crate) fn canonical_query_bytes_len(&self) -> usize {
        self.artifact.canonical_bytes.len()
    }

    pub(crate) fn artifact(&self) -> &CanonicalRoutingQueryArtifactV1 {
        &self.artifact
    }
}

impl DecisionAuditV1 {
    pub(crate) fn new(
        parent: DecisionParentInputV1,
        candidates: Vec<DecisionCandidateInputV1>,
        prepared_query: PreparedCanonicalQueryV1,
    ) -> Result<Self, DecisionAuditError> {
        Self::new_with_binding(
            parent,
            DecisionParentBinding::Recommend,
            candidates,
            prepared_query,
        )
    }

    pub(crate) fn new_active(
        parent: DecisionParentInputV1,
        binding: ActiveDecisionParentBindingV2,
        candidates: Vec<DecisionCandidateInputV1>,
        prepared_query: PreparedCanonicalQueryV1,
    ) -> Result<Self, DecisionAuditError> {
        Self::new_with_binding(
            parent,
            DecisionParentBinding::Active(binding),
            candidates,
            prepared_query,
        )
    }

    fn new_with_binding(
        parent: DecisionParentInputV1,
        binding: DecisionParentBinding,
        candidates: Vec<DecisionCandidateInputV1>,
        prepared_query: PreparedCanonicalQueryV1,
    ) -> Result<Self, DecisionAuditError> {
        validate_parent_input(&parent, &binding, candidates.len(), &prepared_query)?;
        let partition_base: RoutingPartitionBaseV1 =
            serde_json::from_str(&parent.partition_base_json)
                .map_err(|_| DecisionAuditError::InvalidCanonicalDocument)?;

        let mut summaries = Vec::with_capacity(candidates.len());
        let mut neighbors = Vec::new();
        let mut partition_artifacts = Vec::with_capacity(candidates.len());
        for (rank_ordinal, candidate) in candidates.into_iter().enumerate() {
            validate_partition_artifact(
                &parent,
                &partition_base,
                &candidate.summary,
                &candidate.partition_artifact,
            )?;
            if candidate.summary.rank_ordinal != rank_ordinal {
                return Err(DecisionAuditError::InvalidOrdering);
            }
            let candidate_id = candidate.summary.candidate_id.clone();
            let mut candidate_neighbors = Vec::with_capacity(candidate.neighbors.len());
            for neighbor in candidate.neighbors {
                let expected_neighbor_ordinal = neighbors
                    .len()
                    .checked_add(candidate_neighbors.len())
                    .ok_or(DecisionAuditError::SizeOverflow)?;
                candidate_neighbors.push(build_neighbor(
                    parent.decision_id,
                    &parent.learning_generation_id,
                    &candidate_id,
                    expected_neighbor_ordinal,
                    neighbor,
                )?);
            }
            let summary = build_summary(
                parent.decision_id,
                &parent,
                candidate.summary,
                &candidate.partition_artifact.partition_hash,
                &candidate_neighbors,
            )?;
            neighbors.extend(candidate_neighbors);
            summaries.push(summary);
            partition_artifacts.push(candidate.partition_artifact);
        }

        let mut graph = StoredDecisionGraphV1 {
            parent: build_parent(parent, binding, summaries.len(), neighbors.len()),
            summaries,
            neighbors,
        };
        normalize_graph(&mut graph)?;
        let command_size_bytes = checked_command_size(
            graph.parent.aggregate_size_bytes,
            prepared_query.canonical_query_bytes_len(),
            partition_artifacts
                .iter()
                .map(|artifact| artifact.canonical_json.len()),
        )?;
        Ok(Self {
            parent: graph.parent,
            summaries: graph.summaries,
            neighbors: graph.neighbors,
            prepared_query,
            partition_artifacts,
            command_size_bytes,
        })
    }

    pub(crate) fn stored_graph(&self) -> StoredDecisionGraphV1 {
        StoredDecisionGraphV1 {
            parent: self.parent.clone(),
            summaries: self.summaries.clone(),
            neighbors: self.neighbors.clone(),
        }
    }

    pub(crate) fn persisted_eq(
        &self,
        stored: StoredDecisionGraphV1,
    ) -> Result<bool, DecisionAuditError> {
        let verified = stored.verify()?.graph;
        Ok(verified.parent == self.parent
            && verified.summaries == self.summaries
            && verified.neighbors == self.neighbors)
    }

    pub(crate) fn validate_frozen(&self) -> Result<(), DecisionAuditError> {
        validate_graph_parts(&self.parent, &self.summaries, &self.neighbors)?;
        validate_prepared_query(&self.prepared_query, &self.parent.canonical_query_hash)?;
        validate_frozen_partition_artifacts(
            &self.parent,
            &self.summaries,
            &self.partition_artifacts,
        )?;
        let command_size_bytes = checked_command_size(
            self.parent.aggregate_size_bytes,
            self.prepared_query.canonical_query_bytes_len(),
            self.partition_artifacts
                .iter()
                .map(|artifact| artifact.canonical_json.len()),
        )?;
        if command_size_bytes != self.command_size_bytes {
            return Err(DecisionAuditError::InvalidCounts);
        }
        Ok(())
    }
}

fn checked_command_size(
    graph_size_bytes: usize,
    query_size_bytes: usize,
    partition_sizes: impl IntoIterator<Item = usize>,
) -> Result<usize, DecisionAuditError> {
    let total = partition_sizes.into_iter().try_fold(
        graph_size_bytes
            .checked_add(query_size_bytes)
            .ok_or(DecisionAuditError::SizeOverflow)?,
        |total, size| {
            total
                .checked_add(size)
                .ok_or(DecisionAuditError::SizeOverflow)
        },
    )?;
    if total > DECISION_AUDIT_BYTES_MAX {
        return Err(DecisionAuditError::SizeLimitExceeded);
    }
    Ok(total)
}

impl StoredDecisionGraphV1 {
    pub(crate) fn verify(self) -> Result<VerifiedStoredDecisionGraphV1, DecisionAuditError> {
        validate_stored_graph(&self)?;
        Ok(VerifiedStoredDecisionGraphV1 { graph: self })
    }
}

fn validate_stored_graph(graph: &StoredDecisionGraphV1) -> Result<(), DecisionAuditError> {
    validate_graph_parts(&graph.parent, &graph.summaries, &graph.neighbors)
}

fn validate_graph_parts(
    parent: &DecisionRecordV1,
    summaries: &[DecisionCandidateSummaryV1],
    neighbors: &[DecisionNeighborV1],
) -> Result<(), DecisionAuditError> {
    validate_parent_record(parent)?;
    if !(1..=DECISION_SUMMARY_COUNT_MAX).contains(&summaries.len())
        || neighbors.len() > DECISION_NEIGHBOR_COUNT_MAX
        || parent.candidate_count != summaries.len()
        || parent.summary_count != summaries.len()
        || parent.neighbor_count != neighbors.len()
    {
        return Err(DecisionAuditError::InvalidCounts);
    }

    let mut candidate_ids = BTreeSet::new();
    let mut cost_ranks = BTreeSet::new();
    let mut members = Vec::with_capacity(summaries.len());
    let mut previous: Option<(u32, &str)> = None;
    for (rank, summary) in summaries.iter().enumerate() {
        if summary.rank_ordinal != rank
            || !candidate_ids.insert(summary.candidate_id.as_str())
            || !cost_ranks.insert(summary.cost_rank)
            || previous
                .is_some_and(|value| (summary.cost_rank, summary.candidate_id.as_str()) <= value)
        {
            return Err(DecisionAuditError::InvalidOrdering);
        }
        previous = Some((summary.cost_rank, summary.candidate_id.as_str()));
        members.push(CandidateSetMemberInputV1 {
            candidate_id: summary.candidate_id.clone(),
            model: summary.candidate_model.clone(),
            model_revision: summary.candidate_model_revision.clone(),
            cost_rank: summary.cost_rank,
        });
    }
    let candidate_set = build_candidate_set_from_members_v1(&members)
        .map_err(|_| DecisionAuditError::InvalidOrdering)?;
    if candidate_set.candidate_count != summaries.len()
        || candidate_set.candidate_set_hash != parent.candidate_set_hash
    {
        return Err(DecisionAuditError::HashMismatch);
    }
    validate_cross_candidate_policy(parent, summaries)?;

    let mut evidence_ids = BTreeSet::new();
    let mut root_ordinals = BTreeSet::new();
    let mut next_root_ordinal = 0_usize;
    for (ordinal, neighbor) in neighbors.iter().enumerate() {
        validate_neighbor(neighbor)?;
        if neighbor.decision_id != parent.decision_id
            || neighbor.learning_generation_id != parent.learning_generation_id
            || neighbor.neighbor_ordinal != ordinal
            || !evidence_ids.insert(neighbor.evidence_vector_link_id)
        {
            return Err(DecisionAuditError::DuplicateIdentity);
        }
        if root_ordinals.insert(neighbor.root_group_ordinal) {
            if neighbor.root_group_ordinal != next_root_ordinal {
                return Err(DecisionAuditError::InvalidOrdering);
            }
            next_root_ordinal = next_root_ordinal
                .checked_add(1)
                .ok_or(DecisionAuditError::SizeOverflow)?;
        }
    }

    let mut offset = 0_usize;
    for summary in summaries {
        let end = offset
            .checked_add(summary.returned_neighbor_count)
            .ok_or(DecisionAuditError::SizeOverflow)?;
        let candidate_neighbors = neighbors
            .get(offset..end)
            .ok_or(DecisionAuditError::InvalidCounts)?;
        if candidate_neighbors
            .iter()
            .any(|neighbor| neighbor.candidate_id != summary.candidate_id)
        {
            return Err(DecisionAuditError::InvalidOrdering);
        }
        validate_summary_common(
            parent.decision_id,
            parent.learning_generation_id,
            &parent.vector_space_id,
            summary,
            candidate_neighbors,
        )?;
        replay_summary(summary, candidate_neighbors, summaries.len())?;
        offset = end;
    }
    if offset != neighbors.len() {
        return Err(DecisionAuditError::InvalidCounts);
    }
    validate_decision_progression(parent, summaries)?;

    let mut graph_bytes = 0_usize;
    for neighbor in neighbors {
        let (expected_hash, size) = hash_payload(neighbor_payload(neighbor))?;
        if expected_hash != neighbor.canonical_payload_hash {
            return Err(DecisionAuditError::HashMismatch);
        }
        graph_bytes = graph_bytes
            .checked_add(size)
            .ok_or(DecisionAuditError::SizeOverflow)?;
    }
    offset = 0;
    for summary in summaries {
        let end = offset
            .checked_add(summary.neighbor_count)
            .ok_or(DecisionAuditError::SizeOverflow)?;
        let candidate_neighbors = neighbors
            .get(offset..end)
            .ok_or(DecisionAuditError::InvalidCounts)?;
        let hashes = candidate_neighbors
            .iter()
            .map(|neighbor| neighbor.canonical_payload_hash.as_str())
            .collect::<Vec<_>>();
        let expected_aggregate = aggregate_hash(
            DECISION_CANDIDATE_NEIGHBOR_AGGREGATE_SCHEMA_V1,
            Some(&summary.candidate_id),
            &hashes,
        )?;
        if expected_aggregate != summary.neighbor_aggregate_hash {
            return Err(DecisionAuditError::HashMismatch);
        }
        let (expected_hash, size) = hash_payload(summary_payload(summary))?;
        if expected_hash != summary.canonical_payload_hash {
            return Err(DecisionAuditError::HashMismatch);
        }
        graph_bytes = graph_bytes
            .checked_add(size)
            .ok_or(DecisionAuditError::SizeOverflow)?;
        offset = end;
    }

    let summary_hashes = summaries
        .iter()
        .map(|summary| summary.canonical_payload_hash.as_str())
        .collect::<Vec<_>>();
    if aggregate_hash(DECISION_SUMMARY_AGGREGATE_SCHEMA_V1, None, &summary_hashes)?
        != parent.summary_aggregate_hash
    {
        return Err(DecisionAuditError::HashMismatch);
    }
    let neighbor_hashes = neighbors
        .iter()
        .map(|neighbor| neighbor.canonical_payload_hash.as_str())
        .collect::<Vec<_>>();
    if aggregate_hash(
        DECISION_NEIGHBOR_AGGREGATE_SCHEMA_V1,
        None,
        &neighbor_hashes,
    )? != parent.neighbor_aggregate_hash
    {
        return Err(DecisionAuditError::HashMismatch);
    }

    let (aggregate_size_bytes, parent_hash) = expected_parent_size_and_hash(parent, graph_bytes)?;
    if aggregate_size_bytes != parent.aggregate_size_bytes
        || parent_hash != parent.canonical_payload_hash
    {
        return Err(DecisionAuditError::HashMismatch);
    }
    Ok(())
}

fn expected_parent_size_and_hash(
    parent: &DecisionRecordV1,
    child_size_bytes: usize,
) -> Result<(usize, String), DecisionAuditError> {
    let mut total = 1_usize;
    for _ in 0..32 {
        let (_, parent_size) = hash_payload(parent_payload_with_size(parent, total))?;
        let next = child_size_bytes
            .checked_add(parent_size)
            .ok_or(DecisionAuditError::SizeOverflow)?;
        if next > DECISION_AUDIT_BYTES_MAX {
            return Err(DecisionAuditError::SizeLimitExceeded);
        }
        if next == total {
            let hash = hash_payload(parent_payload_with_size(parent, total))?.0;
            return Ok((total, hash));
        }
        total = next;
    }
    Err(DecisionAuditError::SizeOverflow)
}

fn validate_prepared_query(
    prepared: &PreparedCanonicalQueryV1,
    expected_hash: &str,
) -> Result<(), DecisionAuditError> {
    if prepared.canonical_query_hash() != expected_hash
        || prepared.artifact.query.schema != CANONICAL_ROUTING_QUERY_SCHEMA_V1
        || prepared.artifact.canonical_bytes.is_empty()
        || prepared.artifact.canonical_bytes.len() > DECISION_AUDIT_BYTES_MAX
        || sha256_hex(&prepared.artifact.canonical_bytes) != expected_hash
        || std::str::from_utf8(&prepared.artifact.canonical_bytes).is_err()
    {
        return Err(DecisionAuditError::InvalidState);
    }
    Ok(())
}

fn validate_frozen_partition_artifacts(
    parent: &DecisionRecordV1,
    summaries: &[DecisionCandidateSummaryV1],
    artifacts: &[RoutingPartitionArtifactV1],
) -> Result<(), DecisionAuditError> {
    if artifacts.len() != summaries.len() {
        return Err(DecisionAuditError::InvalidCounts);
    }
    let base: RoutingPartitionBaseV1 = serde_json::from_str(&parent.partition_base_json)
        .map_err(|_| DecisionAuditError::InvalidCanonicalDocument)?;
    validate_partition_base(&base)?;
    for (summary, artifact) in summaries.iter().zip(artifacts) {
        validate_canonical_document(&artifact.partition_hash, &artifact.canonical_json)?;
        let value = serde_json::to_value(&artifact.partition)
            .map_err(|_| DecisionAuditError::InvalidCanonicalDocument)?;
        if canonical_json(&value).map_err(|_| DecisionAuditError::InvalidCanonicalDocument)?
            != artifact.canonical_json
            || artifact.partition_hash != summary.partition_hash
            || artifact.partition.tenant_policy_hash != base.tenant_policy_hash
            || artifact.partition.agent_policy_hash != base.agent_policy_hash
            || artifact.partition.policy_version_id != base.policy_version_id
            || artifact.partition.learning_generation_id != base.learning_generation_id
            || artifact.partition.api_family != base.api_family
            || artifact.partition.transport_identity != base.transport_identity
            || artifact.partition.anchor_model != base.anchor_model
            || artifact.partition.anchor_revision != base.anchor_revision
            || artifact.partition.evaluator_version != base.evaluator_version
            || artifact.partition.vector_space_id != base.vector_space_id
            || artifact.partition.candidate_id != summary.candidate_id
            || artifact.partition.candidate_model != summary.candidate_model
            || artifact.partition.candidate_model_revision != summary.candidate_model_revision
            || artifact.partition.decoding_fingerprint != summary.decoding_fingerprint
        {
            return Err(DecisionAuditError::InvalidState);
        }
    }
    Ok(())
}

fn normalize_graph(graph: &mut StoredDecisionGraphV1) -> Result<(), DecisionAuditError> {
    validate_parent_record(&graph.parent)?;
    if !(1..=DECISION_SUMMARY_COUNT_MAX).contains(&graph.summaries.len())
        || graph.neighbors.len() > DECISION_NEIGHBOR_COUNT_MAX
    {
        return Err(DecisionAuditError::InvalidCounts);
    }

    let mut candidate_ids = BTreeSet::new();
    let mut cost_ranks = BTreeSet::new();
    let mut members = Vec::with_capacity(graph.summaries.len());
    let mut previous: Option<(u32, &str)> = None;
    for (rank, summary) in graph.summaries.iter().enumerate() {
        if summary.rank_ordinal != rank
            || !candidate_ids.insert(summary.candidate_id.as_str())
            || !cost_ranks.insert(summary.cost_rank)
            || previous
                .is_some_and(|value| (summary.cost_rank, summary.candidate_id.as_str()) <= value)
        {
            return Err(DecisionAuditError::InvalidOrdering);
        }
        previous = Some((summary.cost_rank, summary.candidate_id.as_str()));
        members.push(CandidateSetMemberInputV1 {
            candidate_id: summary.candidate_id.clone(),
            model: summary.candidate_model.clone(),
            model_revision: summary.candidate_model_revision.clone(),
            cost_rank: summary.cost_rank,
        });
    }
    let candidate_set = build_candidate_set_from_members_v1(&members)
        .map_err(|_| DecisionAuditError::InvalidOrdering)?;
    if candidate_set.candidate_count != graph.summaries.len()
        || candidate_set.candidate_set_hash != graph.parent.candidate_set_hash
    {
        return Err(DecisionAuditError::HashMismatch);
    }
    validate_cross_candidate_policy(&graph.parent, &graph.summaries)?;

    let mut evidence_ids = BTreeSet::new();
    let mut root_ordinals = BTreeSet::new();
    let mut next_root_ordinal = 0_usize;
    for (ordinal, neighbor) in graph.neighbors.iter().enumerate() {
        validate_neighbor(neighbor)?;
        if neighbor.decision_id != graph.parent.decision_id
            || neighbor.learning_generation_id != graph.parent.learning_generation_id
            || neighbor.neighbor_ordinal != ordinal
            || !evidence_ids.insert(neighbor.evidence_vector_link_id)
        {
            return Err(DecisionAuditError::DuplicateIdentity);
        }
        if root_ordinals.insert(neighbor.root_group_ordinal) {
            if neighbor.root_group_ordinal != next_root_ordinal {
                return Err(DecisionAuditError::InvalidOrdering);
            }
            next_root_ordinal = next_root_ordinal
                .checked_add(1)
                .ok_or(DecisionAuditError::SizeOverflow)?;
        }
    }

    let mut ranges = Vec::with_capacity(graph.summaries.len());
    let mut offset = 0_usize;
    let candidate_count = graph.summaries.len();
    for summary in &mut graph.summaries {
        let end = offset
            .checked_add(summary.returned_neighbor_count)
            .ok_or(DecisionAuditError::SizeOverflow)?;
        let candidate_neighbors = graph
            .neighbors
            .get(offset..end)
            .ok_or(DecisionAuditError::InvalidCounts)?;
        if candidate_neighbors
            .iter()
            .any(|neighbor| neighbor.candidate_id != summary.candidate_id)
        {
            return Err(DecisionAuditError::InvalidOrdering);
        }
        summary.neighbor_count = candidate_neighbors.len();
        validate_summary_common(
            graph.parent.decision_id,
            graph.parent.learning_generation_id,
            &graph.parent.vector_space_id,
            summary,
            candidate_neighbors,
        )?;
        replay_summary(summary, candidate_neighbors, candidate_count)?;
        ranges.push(offset..end);
        offset = end;
    }
    if offset != graph.neighbors.len() {
        return Err(DecisionAuditError::InvalidCounts);
    }
    validate_decision_progression(&graph.parent, &graph.summaries)?;

    let mut graph_bytes = 0_usize;
    for neighbor in &mut graph.neighbors {
        let (hash, size) = hash_payload(neighbor_payload(neighbor))?;
        neighbor.canonical_payload_hash = hash;
        graph_bytes = graph_bytes
            .checked_add(size)
            .ok_or(DecisionAuditError::SizeOverflow)?;
    }
    for (summary, range) in graph.summaries.iter_mut().zip(ranges) {
        let hashes = graph.neighbors[range]
            .iter()
            .map(|neighbor| neighbor.canonical_payload_hash.as_str())
            .collect::<Vec<_>>();
        summary.neighbor_count = hashes.len();
        summary.neighbor_aggregate_hash = aggregate_hash(
            DECISION_CANDIDATE_NEIGHBOR_AGGREGATE_SCHEMA_V1,
            Some(&summary.candidate_id),
            &hashes,
        )?;
        let (hash, size) = hash_payload(summary_payload(summary))?;
        summary.canonical_payload_hash = hash;
        graph_bytes = graph_bytes
            .checked_add(size)
            .ok_or(DecisionAuditError::SizeOverflow)?;
    }

    graph.parent.candidate_count = graph.summaries.len();
    graph.parent.summary_count = graph.summaries.len();
    graph.parent.neighbor_count = graph.neighbors.len();
    let summary_hashes = graph
        .summaries
        .iter()
        .map(|summary| summary.canonical_payload_hash.as_str())
        .collect::<Vec<_>>();
    graph.parent.summary_aggregate_hash =
        aggregate_hash(DECISION_SUMMARY_AGGREGATE_SCHEMA_V1, None, &summary_hashes)?;
    let neighbor_hashes = graph
        .neighbors
        .iter()
        .map(|neighbor| neighbor.canonical_payload_hash.as_str())
        .collect::<Vec<_>>();
    graph.parent.neighbor_aggregate_hash = aggregate_hash(
        DECISION_NEIGHBOR_AGGREGATE_SCHEMA_V1,
        None,
        &neighbor_hashes,
    )?;

    let mut total = 1_usize;
    let mut converged = false;
    for _ in 0..32 {
        graph.parent.aggregate_size_bytes = total;
        let (_, parent_size) = hash_payload(parent_payload(&graph.parent))?;
        let next = graph_bytes
            .checked_add(parent_size)
            .ok_or(DecisionAuditError::SizeOverflow)?;
        if next > DECISION_AUDIT_BYTES_MAX {
            return Err(DecisionAuditError::SizeLimitExceeded);
        }
        if next == total {
            converged = true;
            break;
        }
        total = next;
    }
    if !converged {
        return Err(DecisionAuditError::SizeOverflow);
    }
    graph.parent.aggregate_size_bytes = total;
    graph.parent.canonical_payload_hash = hash_payload(parent_payload(&graph.parent))?.0;
    Ok(())
}

fn validate_parent_record(parent: &DecisionRecordV1) -> Result<(), DecisionAuditError> {
    let mode_valid = match (parent.decision_shape_version, parent.algorithm_version) {
        (DECISION_SHAPE_VERSION_V1, DECISION_ALGORITHM_VERSION_V1) => {
            !parent.final_reason.is_active()
                && parent.cohort_generation_id.is_none()
                && parent.active_experiment_id.is_none()
                && parent.active_authorization_state_event_id.is_none()
                && parent.root_key.is_none()
        }
        (ACTIVE_DECISION_SHAPE_VERSION_V2, ACTIVE_DECISION_ALGORITHM_VERSION_V2) => {
            parent.final_reason.is_active()
                && parent.cohort_generation_id.is_some_and(is_uuid_v7)
                && parent
                    .root_key
                    .as_deref()
                    .is_some_and(|root_key| validate_hash(root_key).is_ok())
                && parent.active_experiment_id.is_none_or(is_uuid_v7)
                && parent
                    .active_authorization_state_event_id
                    .is_none_or(is_uuid_v7)
                && parent.active_experiment_id.is_some()
                    == parent.active_authorization_state_event_id.is_some()
                && active_record_route_is_coherent(parent)
        }
        _ => false,
    };
    if !mode_valid
        || ![
            parent.decision_id,
            parent.project_uuid,
            parent.process_instance_id,
            parent.learning_generation_id,
            parent.primary_call_uuid,
        ]
        .into_iter()
        .all(is_uuid_v7)
        || parent.as_of_unix_ms < 0
        || parent.created_at_unix_ms < 0
    {
        return Err(DecisionAuditError::InvalidIdentity);
    }
    for hash in [
        &parent.config_generation_id,
        &parent.policy_version_id,
        &parent.canonical_query_hash,
        &parent.partition_base_hash,
        &parent.vector_space_id,
        &parent.candidate_set_hash,
    ] {
        validate_hash(hash)?;
    }
    validate_text(&parent.pool_id, ID_MAX_BYTES)?;
    if let Some(candidate_id) = &parent.candidate_id {
        validate_text(candidate_id, CANDIDATE_ID_MAX_BYTES)?;
    }
    validate_text(&parent.recommended_model, MODEL_ID_MAX_BYTES)?;
    validate_text(&parent.recommended_model_revision, REVISION_MAX_BYTES)?;
    validate_text(&parent.served_model, MODEL_ID_MAX_BYTES)?;
    validate_text(&parent.served_model_revision, REVISION_MAX_BYTES)?;
    validate_canonical_document(&parent.partition_base_hash, &parent.partition_base_json)?;
    let base: RoutingPartitionBaseV1 = serde_json::from_str(&parent.partition_base_json)
        .map_err(|_| DecisionAuditError::InvalidCanonicalDocument)?;
    validate_partition_base(&base)?;
    if base.policy_version_id != parent.policy_version_id
        || base.learning_generation_id != parent.learning_generation_id
        || base.vector_space_id != parent.vector_space_id
    {
        return Err(DecisionAuditError::InvalidState);
    }
    Ok(())
}

pub(crate) fn verify_decision_parent_record(
    parent: &DecisionRecordV1,
) -> Result<(), DecisionAuditError> {
    validate_parent_record(parent)?;
    let expected_hash = hash_payload(parent_payload(parent))?.0;
    if expected_hash != parent.canonical_payload_hash {
        return Err(DecisionAuditError::HashMismatch);
    }
    Ok(())
}

fn active_record_route_is_coherent(parent: &DecisionRecordV1) -> bool {
    match parent.final_reason {
        DecisionFinalReasonV1::ActiveCandidate => {
            parent.candidate_id.is_some()
                && parent.recommended_model == parent.served_model
                && parent.recommended_model_revision == parent.served_model_revision
        }
        DecisionFinalReasonV1::ActiveAnchorControl | DecisionFinalReasonV1::ActiveAnchorHoldout => {
            parent.candidate_id.is_some()
        }
        reason if reason.is_active() => {
            parent.candidate_id.is_some()
                || (parent.recommended_model == parent.served_model
                    && parent.recommended_model_revision == parent.served_model_revision)
        }
        _ => false,
    }
}

fn validate_cross_candidate_policy(
    parent: &DecisionRecordV1,
    summaries: &[DecisionCandidateSummaryV1],
) -> Result<(), DecisionAuditError> {
    let first = summaries.first().ok_or(DecisionAuditError::InvalidCounts)?;
    for summary in &summaries[1..] {
        if summary.top_k != first.top_k
            || summary.radius != first.radius
            || summary.min_points != first.min_points
            || summary.min_independent_roots != first.min_independent_roots
            || summary.min_effective_samples != first.min_effective_samples
            || summary.min_coverage != first.min_coverage
            || summary.time_decay_half_life_seconds != first.time_decay_half_life_seconds
            || summary.prior_success != first.prior_success
            || summary.prior_failure != first.prior_failure
            || summary.familywise_credible_level != first.familywise_credible_level
            || summary.candidate_alpha != first.candidate_alpha
            || (parent.decision_shape_version == DECISION_SHAPE_VERSION_V1
                && summary.promotion_lower_bound != first.promotion_lower_bound)
        {
            return Err(DecisionAuditError::InvalidState);
        }
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn replay_summary(
    summary: &DecisionCandidateSummaryV1,
    neighbors: &[DecisionNeighborV1],
    candidate_count: usize,
) -> Result<(), DecisionAuditError> {
    let expected_alpha = (1.0 - summary.familywise_credible_level.value()) / candidate_count as f64;
    require_close(summary.candidate_alpha.value(), expected_alpha)?;

    if matches!(
        summary.terminal_reason,
        DecisionCandidateReasonV1::NoPartition
            | DecisionCandidateReasonV1::NotEvaluatedAfterWinner
            | DecisionCandidateReasonV1::NotEvaluatedAfterFallback
    ) {
        return require_no_computed_neighbors(neighbors);
    }

    let mut labeled_roots = BTreeMap::<usize, usize>::new();
    for neighbor in neighbors {
        let inside_radius = f64::from(neighbor.distance.value()) <= summary.radius.value();
        if inside_radius
            != (neighbor.exclusion_reason != DecisionNeighborExclusionReasonV1::OutsideRadius)
        {
            return Err(DecisionAuditError::InvalidState);
        }
        if matches!(
            neighbor.exclusion_reason,
            DecisionNeighborExclusionReasonV1::DuplicateRoot
                | DecisionNeighborExclusionReasonV1::Included
        ) {
            if neighbor.evaluation_id.is_none() || neighbor.binary_label.is_none() {
                return Err(DecisionAuditError::InvalidState);
            }
            let included = labeled_roots
                .entry(neighbor.root_group_ordinal)
                .or_default();
            if neighbor.exclusion_reason == DecisionNeighborExclusionReasonV1::Included {
                *included = included
                    .checked_add(1)
                    .ok_or(DecisionAuditError::SizeOverflow)?;
            }
        }
    }
    if labeled_roots.len() != summary.labeled_root_count
        || summary.selected_root_count != summary.labeled_root_count
        || labeled_roots.values().any(|included| *included != 1)
    {
        return Err(DecisionAuditError::InvalidCounts);
    }

    let points_passed = summary.labeled_point_count >= summary.min_points;
    require_gate(summary.points_gate_passed, points_passed)?;
    if !points_passed {
        require_reason(summary, DecisionCandidateReasonV1::SparsePoints)?;
        return require_no_computed_neighbors(neighbors);
    }
    let roots_passed = summary.labeled_root_count >= summary.min_independent_roots;
    require_gate(summary.roots_gate_passed, roots_passed)?;
    if !roots_passed {
        require_reason(summary, DecisionCandidateReasonV1::InsufficientRoots)?;
        return require_no_computed_neighbors(neighbors);
    }
    let coverage = summary
        .coverage
        .ok_or(DecisionAuditError::InvalidState)?
        .value();
    let coverage_passed = coverage >= summary.min_coverage.value();
    require_gate(summary.coverage_gate_passed, coverage_passed)?;
    if !coverage_passed {
        require_reason(summary, DecisionCandidateReasonV1::LowCoverage)?;
        return require_no_computed_neighbors(neighbors);
    }

    if summary.weight_gate_passed == Some(false) {
        if !matches!(
            summary.terminal_reason,
            DecisionCandidateReasonV1::InvalidEvidenceTime
                | DecisionCandidateReasonV1::NumericError
        ) {
            return Err(DecisionAuditError::InvalidState);
        }
        return require_no_computed_neighbors(neighbors);
    }
    if summary.weight_gate_passed != Some(true) {
        return Err(DecisionAuditError::InvalidState);
    }

    let computed = replay_weights(summary, neighbors)?;
    require_close(
        summary
            .sum_weight
            .ok_or(DecisionAuditError::InvalidState)?
            .value(),
        computed.sum_weight,
    )?;
    require_close(
        summary
            .sum_weighted_label
            .ok_or(DecisionAuditError::InvalidState)?
            .value(),
        computed.sum_weighted_label,
    )?;
    require_close(
        summary
            .sum_squared_weight
            .ok_or(DecisionAuditError::InvalidState)?
            .value(),
        computed.sum_squared_weight,
    )?;
    require_close(
        summary
            .p_hat
            .ok_or(DecisionAuditError::InvalidState)?
            .value(),
        computed.p_hat,
    )?;
    require_close(
        summary
            .effective_sample_size
            .ok_or(DecisionAuditError::InvalidState)?
            .value(),
        computed.effective_sample_size,
    )?;

    let effective_samples_passed =
        computed.effective_sample_size >= summary.min_effective_samples.value();
    require_gate(
        summary.effective_samples_gate_passed,
        effective_samples_passed,
    )?;
    if !effective_samples_passed {
        return require_reason(
            summary,
            DecisionCandidateReasonV1::InsufficientEffectiveSamples,
        );
    }

    let beta_alpha =
        summary.prior_success.value() + computed.p_hat * computed.effective_sample_size;
    let beta_beta =
        summary.prior_failure.value() + (1.0 - computed.p_hat) * computed.effective_sample_size;
    if !beta_alpha.is_finite() || !beta_beta.is_finite() {
        if summary.beta_alpha.is_some()
            || summary.beta_beta.is_some()
            || summary.beta_quantile_gate_passed != Some(false)
            || summary.lower_bound.is_some()
        {
            return Err(DecisionAuditError::InvalidState);
        }
        return require_reason(summary, DecisionCandidateReasonV1::NumericError);
    }
    require_close(
        summary
            .beta_alpha
            .ok_or(DecisionAuditError::InvalidState)?
            .value(),
        beta_alpha,
    )?;
    require_close(
        summary
            .beta_beta
            .ok_or(DecisionAuditError::InvalidState)?
            .value(),
        beta_beta,
    )?;
    let lower_bound =
        match beta_inverse_cdf_v1(summary.candidate_alpha.value(), beta_alpha, beta_beta) {
            Ok(lower_bound) => lower_bound,
            Err(_) => {
                if summary.beta_quantile_gate_passed != Some(false) || summary.lower_bound.is_some()
                {
                    return Err(DecisionAuditError::InvalidState);
                }
                return require_reason(summary, DecisionCandidateReasonV1::NumericError);
            }
        };
    if summary.beta_quantile_gate_passed != Some(true) {
        return Err(DecisionAuditError::InvalidState);
    }
    require_close(
        summary
            .lower_bound
            .ok_or(DecisionAuditError::InvalidState)?
            .value(),
        lower_bound,
    )?;
    let lower_bound_passed = lower_bound >= summary.promotion_lower_bound.value();
    require_gate(summary.lower_bound_gate_passed, lower_bound_passed)?;
    require_reason(
        summary,
        if lower_bound_passed {
            DecisionCandidateReasonV1::Passed
        } else {
            DecisionCandidateReasonV1::LowerBoundBelowThreshold
        },
    )
}

struct ReplayedWeightsV1 {
    sum_weight: f64,
    sum_weighted_label: f64,
    sum_squared_weight: f64,
    p_hat: f64,
    effective_sample_size: f64,
}

fn replay_weights(
    summary: &DecisionCandidateSummaryV1,
    neighbors: &[DecisionNeighborV1],
) -> Result<ReplayedWeightsV1, DecisionAuditError> {
    let mut selected = Vec::with_capacity(summary.selected_root_count);
    for neighbor in neighbors {
        let is_included = neighbor.exclusion_reason == DecisionNeighborExclusionReasonV1::Included;
        let phases = [
            neighbor.age_seconds,
            neighbor.similarity_weight,
            neighbor.time_weight,
            neighbor.final_weight,
        ];
        if is_included {
            if neighbor.evaluation_id.is_none()
                || neighbor.binary_label.is_none()
                || phases.iter().any(Option::is_none)
            {
                return Err(DecisionAuditError::InvalidState);
            }
            let age_seconds = neighbor
                .age_seconds
                .ok_or(DecisionAuditError::InvalidState)?
                .value();
            let similarity_weight =
                (1.0 - f64::from(neighbor.distance.value()) / summary.radius.value()).max(0.0);
            let time_weight = (-age_seconds / summary.time_decay_half_life_seconds.value()).exp2();
            let final_weight = similarity_weight * time_weight;
            if !similarity_weight.is_finite()
                || !time_weight.is_finite()
                || !final_weight.is_finite()
            {
                return Err(DecisionAuditError::InvalidState);
            }
            require_close(
                neighbor
                    .similarity_weight
                    .ok_or(DecisionAuditError::InvalidState)?
                    .value(),
                similarity_weight,
            )?;
            require_close(
                neighbor
                    .time_weight
                    .ok_or(DecisionAuditError::InvalidState)?
                    .value(),
                time_weight,
            )?;
            require_close(
                neighbor
                    .final_weight
                    .ok_or(DecisionAuditError::InvalidState)?
                    .value(),
                final_weight,
            )?;
            selected.push((
                neighbor
                    .evaluation_id
                    .ok_or(DecisionAuditError::InvalidState)?,
                final_weight,
                neighbor
                    .binary_label
                    .ok_or(DecisionAuditError::InvalidState)?,
            ));
        } else if phases.iter().any(Option::is_some) {
            return Err(DecisionAuditError::InvalidState);
        }
    }
    selected.sort_by_key(|item| item.0);
    let mut sum_weight = ReplaySum::default();
    let mut sum_weighted_label = ReplaySum::default();
    let mut sum_squared_weight = ReplaySum::default();
    for (_, weight, label) in selected {
        sum_weight.add(weight)?;
        sum_weighted_label.add(
            weight
                * if label == DecisionBinaryLabelV1::Pass {
                    1.0
                } else {
                    0.0
                },
        )?;
        sum_squared_weight.add(weight * weight)?;
    }
    let sum_weight = sum_weight.total()?;
    let sum_weighted_label = sum_weighted_label.total()?;
    let sum_squared_weight = sum_squared_weight.total()?;
    if sum_weight <= 0.0 || sum_squared_weight <= 0.0 {
        return Err(DecisionAuditError::InvalidState);
    }
    let p_hat = sum_weighted_label / sum_weight;
    let effective_sample_size = (sum_weight * sum_weight) / sum_squared_weight;
    if !p_hat.is_finite()
        || !(0.0..=1.0).contains(&p_hat)
        || !effective_sample_size.is_finite()
        || effective_sample_size <= 0.0
    {
        return Err(DecisionAuditError::InvalidState);
    }
    Ok(ReplayedWeightsV1 {
        sum_weight,
        sum_weighted_label,
        sum_squared_weight,
        p_hat,
        effective_sample_size,
    })
}

#[derive(Default)]
struct ReplaySum {
    sum: f64,
    correction: f64,
}

impl ReplaySum {
    fn add(&mut self, value: f64) -> Result<(), DecisionAuditError> {
        let next = self.sum + value;
        let adjustment = if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        let correction = self.correction + adjustment;
        if !next.is_finite() || !adjustment.is_finite() || !correction.is_finite() {
            return Err(DecisionAuditError::InvalidState);
        }
        self.sum = next;
        self.correction = correction;
        Ok(())
    }

    fn total(self) -> Result<f64, DecisionAuditError> {
        let total = self.sum + self.correction;
        total
            .is_finite()
            .then_some(total)
            .ok_or(DecisionAuditError::InvalidState)
    }
}

fn require_gate(value: Option<bool>, expected: bool) -> Result<(), DecisionAuditError> {
    (value == Some(expected))
        .then_some(())
        .ok_or(DecisionAuditError::InvalidState)
}

fn require_reason(
    summary: &DecisionCandidateSummaryV1,
    expected: DecisionCandidateReasonV1,
) -> Result<(), DecisionAuditError> {
    (summary.terminal_reason == expected)
        .then_some(())
        .ok_or(DecisionAuditError::InvalidState)
}

fn require_close(actual: f64, expected: f64) -> Result<(), DecisionAuditError> {
    (actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= 1.0e-12)
        .then_some(())
        .ok_or(DecisionAuditError::InvalidState)
}

fn require_no_computed_neighbors(
    neighbors: &[DecisionNeighborV1],
) -> Result<(), DecisionAuditError> {
    neighbors
        .iter()
        .all(|neighbor| {
            neighbor.age_seconds.is_none()
                && neighbor.similarity_weight.is_none()
                && neighbor.time_weight.is_none()
                && neighbor.final_weight.is_none()
        })
        .then_some(())
        .ok_or(DecisionAuditError::InvalidState)
}

fn validate_decision_progression(
    parent: &DecisionRecordV1,
    summaries: &[DecisionCandidateSummaryV1],
) -> Result<(), DecisionAuditError> {
    if parent.decision_shape_version == ACTIVE_DECISION_SHAPE_VERSION_V2 {
        return validate_active_decision_progression(parent, summaries);
    }
    let is_evaluated_failure = |reason: DecisionCandidateReasonV1| {
        !matches!(
            reason,
            DecisionCandidateReasonV1::Passed
                | DecisionCandidateReasonV1::InvalidEvidenceTime
                | DecisionCandidateReasonV1::NumericError
                | DecisionCandidateReasonV1::NotEvaluatedAfterWinner
                | DecisionCandidateReasonV1::NotEvaluatedAfterFallback
        )
    };
    match parent.final_reason {
        DecisionFinalReasonV1::RecommendObserveOnly => {
            let winners = summaries
                .iter()
                .enumerate()
                .filter(|(_, summary)| summary.terminal_reason == DecisionCandidateReasonV1::Passed)
                .collect::<Vec<_>>();
            let [(winner_index, winner)] = winners.as_slice() else {
                return Err(DecisionAuditError::InvalidState);
            };
            if parent.candidate_id.as_deref() != Some(winner.candidate_id.as_str())
                || parent.recommended_model != winner.candidate_model
                || parent.recommended_model_revision != winner.candidate_model_revision
                || summaries[..*winner_index]
                    .iter()
                    .any(|summary| !is_evaluated_failure(summary.terminal_reason))
                || summaries[*winner_index + 1..].iter().any(|summary| {
                    summary.terminal_reason != DecisionCandidateReasonV1::NotEvaluatedAfterWinner
                })
            {
                return Err(DecisionAuditError::InvalidState);
            }
        }
        DecisionFinalReasonV1::EmbeddingUnavailable => {
            if summaries.iter().any(|summary| {
                summary.terminal_reason != DecisionCandidateReasonV1::NotEvaluatedAfterFallback
            }) {
                return Err(DecisionAuditError::InvalidState);
            }
        }
        DecisionFinalReasonV1::VectorUnhealthy | DecisionFinalReasonV1::VersionMismatch => {
            let fallback_index = summaries
                .iter()
                .position(|summary| {
                    summary.terminal_reason == DecisionCandidateReasonV1::NotEvaluatedAfterFallback
                })
                .ok_or(DecisionAuditError::InvalidState)?;
            if summaries[..fallback_index]
                .iter()
                .any(|summary| !is_evaluated_failure(summary.terminal_reason))
                || summaries[fallback_index..].iter().any(|summary| {
                    summary.terminal_reason != DecisionCandidateReasonV1::NotEvaluatedAfterFallback
                })
            {
                return Err(DecisionAuditError::InvalidState);
            }
        }
        DecisionFinalReasonV1::InvalidEvidenceTime | DecisionFinalReasonV1::NumericError => {
            let terminal = if parent.final_reason == DecisionFinalReasonV1::InvalidEvidenceTime {
                DecisionCandidateReasonV1::InvalidEvidenceTime
            } else {
                DecisionCandidateReasonV1::NumericError
            };
            let matching = summaries
                .iter()
                .enumerate()
                .filter(|(_, summary)| summary.terminal_reason == terminal)
                .collect::<Vec<_>>();
            let [(fallback_index, _)] = matching.as_slice() else {
                return Err(DecisionAuditError::InvalidState);
            };
            if summaries[..*fallback_index]
                .iter()
                .any(|summary| !is_evaluated_failure(summary.terminal_reason))
                || summaries[*fallback_index + 1..].iter().any(|summary| {
                    summary.terminal_reason != DecisionCandidateReasonV1::NotEvaluatedAfterFallback
                })
            {
                return Err(DecisionAuditError::InvalidState);
            }
        }
        DecisionFinalReasonV1::NoCandidatePassed => {
            if summaries
                .iter()
                .any(|summary| !is_evaluated_failure(summary.terminal_reason))
            {
                return Err(DecisionAuditError::InvalidState);
            }
        }
        final_reason => {
            if summaries
                .iter()
                .any(|summary| !is_evaluated_failure(summary.terminal_reason))
                || !summaries.iter().any(|summary| {
                    candidate_reason_matches_final(summary.terminal_reason, final_reason)
                })
            {
                return Err(DecisionAuditError::InvalidState);
            }
        }
    }
    Ok(())
}

fn validate_active_decision_progression(
    parent: &DecisionRecordV1,
    summaries: &[DecisionCandidateSummaryV1],
) -> Result<(), DecisionAuditError> {
    if !parent.final_reason.is_active() {
        return Err(DecisionAuditError::InvalidState);
    }
    if let Some(candidate_id) = parent.candidate_id.as_deref() {
        let winner_index = summaries
            .iter()
            .position(|summary| summary.candidate_id == candidate_id)
            .ok_or(DecisionAuditError::InvalidState)?;
        let winner = &summaries[winner_index];
        if winner.terminal_reason != DecisionCandidateReasonV1::Passed
            || parent.recommended_model != winner.candidate_model
            || parent.recommended_model_revision != winner.candidate_model_revision
            || summaries[..winner_index].iter().any(|summary| {
                matches!(
                    summary.terminal_reason,
                    DecisionCandidateReasonV1::NotEvaluatedAfterWinner
                        | DecisionCandidateReasonV1::NotEvaluatedAfterFallback
                        | DecisionCandidateReasonV1::InvalidEvidenceTime
                        | DecisionCandidateReasonV1::NumericError
                )
            })
            || summaries[winner_index + 1..].iter().any(|summary| {
                summary.terminal_reason != DecisionCandidateReasonV1::NotEvaluatedAfterWinner
            })
        {
            return Err(DecisionAuditError::InvalidState);
        }
        return Ok(());
    }

    if summaries.iter().any(|summary| {
        summary.terminal_reason == DecisionCandidateReasonV1::NotEvaluatedAfterWinner
    }) {
        return Err(DecisionAuditError::InvalidState);
    }
    let fallback_index = summaries.iter().position(|summary| {
        matches!(
            summary.terminal_reason,
            DecisionCandidateReasonV1::NotEvaluatedAfterFallback
                | DecisionCandidateReasonV1::InvalidEvidenceTime
                | DecisionCandidateReasonV1::NumericError
        )
    });
    if fallback_index.is_some_and(|index| {
        summaries[index + 1..].iter().any(|summary| {
            summary.terminal_reason != DecisionCandidateReasonV1::NotEvaluatedAfterFallback
        })
    }) {
        return Err(DecisionAuditError::InvalidState);
    }
    Ok(())
}

fn candidate_reason_matches_final(
    candidate: DecisionCandidateReasonV1,
    final_reason: DecisionFinalReasonV1,
) -> bool {
    matches!(
        (candidate, final_reason),
        (
            DecisionCandidateReasonV1::NoPartition,
            DecisionFinalReasonV1::NoPartition
        ) | (
            DecisionCandidateReasonV1::SparsePoints,
            DecisionFinalReasonV1::SparsePoints
        ) | (
            DecisionCandidateReasonV1::InsufficientRoots,
            DecisionFinalReasonV1::InsufficientRoots
        ) | (
            DecisionCandidateReasonV1::LowCoverage,
            DecisionFinalReasonV1::LowCoverage
        ) | (
            DecisionCandidateReasonV1::InsufficientEffectiveSamples,
            DecisionFinalReasonV1::InsufficientEffectiveSamples
        ) | (
            DecisionCandidateReasonV1::LowerBoundBelowThreshold,
            DecisionFinalReasonV1::LowerBoundBelowThreshold
        )
    )
}

fn hash_payload(payload: Json) -> Result<(String, usize), DecisionAuditError> {
    let bytes =
        canonical_json_bytes(&payload).map_err(|_| DecisionAuditError::InvalidCanonicalDocument)?;
    Ok((sha256_hex(&bytes), bytes.len()))
}

fn aggregate_hash(
    schema: &str,
    candidate_id: Option<&str>,
    hashes: &[&str],
) -> Result<String, DecisionAuditError> {
    for hash in hashes {
        validate_hash(hash)?;
    }
    let payload = if let Some(candidate_id) = candidate_id {
        json!({
            "schema": schema,
            "candidate_id": candidate_id,
            "count": hashes.len(),
            "hashes": hashes,
        })
    } else {
        json!({
            "schema": schema,
            "count": hashes.len(),
            "hashes": hashes,
        })
    };
    Ok(hash_payload(payload)?.0)
}

fn parent_payload(parent: &DecisionRecordV1) -> Json {
    parent_payload_with_size(parent, parent.aggregate_size_bytes)
}

fn parent_payload_with_size(parent: &DecisionRecordV1, aggregate_size_bytes: usize) -> Json {
    let mode = if parent.decision_shape_version == ACTIVE_DECISION_SHAPE_VERSION_V2 {
        "active"
    } else {
        "recommend"
    };
    json!({
        "schema": DECISION_PARENT_HASH_SCHEMA_V1,
        "decision_id": parent.decision_id.to_string(),
        "decision_shape_version": parent.decision_shape_version,
        "algorithm_version": parent.algorithm_version,
        "project_uuid": parent.project_uuid.to_string(),
        "process_instance_id": parent.process_instance_id.to_string(),
        "config_generation_id": parent.config_generation_id,
        "policy_version_id": parent.policy_version_id,
        "learning_generation_id": parent.learning_generation_id.to_string(),
        "cohort_generation_id": option_uuid(parent.cohort_generation_id),
        "active_experiment_id": option_uuid(parent.active_experiment_id),
        "active_authorization_state_event_id": option_uuid(
            parent.active_authorization_state_event_id,
        ),
        "pool_id": parent.pool_id,
        "candidate_id": parent.candidate_id,
        "root_key": parent.root_key,
        "primary_call_uuid": parent.primary_call_uuid.to_string(),
        "mode": mode,
        "canonical_query_hash": parent.canonical_query_hash,
        "partition_base_json": parent.partition_base_json,
        "partition_base_hash": parent.partition_base_hash,
        "vector_space_id": parent.vector_space_id,
        "candidate_set_hash": parent.candidate_set_hash,
        "candidate_count": parent.candidate_count,
        "recommended_model": parent.recommended_model,
        "recommended_model_revision": parent.recommended_model_revision,
        "served_model": parent.served_model,
        "served_model_revision": parent.served_model_revision,
        "as_of_unix_ms": parent.as_of_unix_ms.to_string(),
        "decision_latency_ms": parent.decision_latency_ms.to_string(),
        "final_reason": parent.final_reason.as_str(),
        "summary_count": parent.summary_count,
        "summary_aggregate_hash": parent.summary_aggregate_hash,
        "neighbor_count": parent.neighbor_count,
        "neighbor_aggregate_hash": parent.neighbor_aggregate_hash,
        "aggregate_size_bytes": aggregate_size_bytes,
        "created_at_unix_ms": parent.created_at_unix_ms.to_string(),
    })
}

fn summary_payload(summary: &DecisionCandidateSummaryV1) -> Json {
    merge_json_objects([
        json!({
            "schema": DECISION_SUMMARY_HASH_SCHEMA_V1,
            "decision_id": summary.decision_id.to_string(),
            "candidate_id": summary.candidate_id,
            "rank_ordinal": summary.rank_ordinal,
            "candidate_model": summary.candidate_model,
            "candidate_model_revision": summary.candidate_model_revision,
            "cost_rank": summary.cost_rank,
            "learning_generation_id": summary.learning_generation_id.to_string(),
            "vector_space_id": summary.vector_space_id,
            "partition_hash": summary.partition_hash,
            "partition_id": option_i64(summary.partition_id),
            "decoding_fingerprint": summary.decoding_fingerprint,
            "top_k": summary.top_k,
            "radius_bits": f64_bits(summary.radius),
            "min_points": summary.min_points,
            "min_independent_roots": summary.min_independent_roots,
            "min_effective_samples_bits": f64_bits(summary.min_effective_samples),
        }),
        json!({
            "min_coverage_bits": f64_bits(summary.min_coverage),
            "time_decay_half_life_seconds_bits": f64_bits(
                summary.time_decay_half_life_seconds,
            ),
            "prior_success_bits": f64_bits(summary.prior_success),
            "prior_failure_bits": f64_bits(summary.prior_failure),
            "familywise_credible_level_bits": f64_bits(summary.familywise_credible_level),
            "candidate_alpha_bits": f64_bits(summary.candidate_alpha),
            "promotion_lower_bound_bits": f64_bits(summary.promotion_lower_bound),
            "returned_neighbor_count": summary.returned_neighbor_count,
            "within_radius_count": summary.within_radius_count,
            "labeled_point_count": summary.labeled_point_count,
            "attempted_root_count": summary.attempted_root_count,
            "labeled_root_count": summary.labeled_root_count,
            "selected_root_count": summary.selected_root_count,
            "coverage_bits": option_f64_bits(summary.coverage),
            "sum_weight_bits": option_f64_bits(summary.sum_weight),
            "sum_weighted_label_bits": option_f64_bits(summary.sum_weighted_label),
            "sum_squared_weight_bits": option_f64_bits(summary.sum_squared_weight),
        }),
        json!({
            "p_hat_bits": option_f64_bits(summary.p_hat),
            "effective_sample_size_bits": option_f64_bits(summary.effective_sample_size),
            "beta_alpha_bits": option_f64_bits(summary.beta_alpha),
            "beta_beta_bits": option_f64_bits(summary.beta_beta),
            "lower_bound_bits": option_f64_bits(summary.lower_bound),
            "partition_gate_passed": summary.partition_gate_passed,
            "points_gate_passed": summary.points_gate_passed,
            "roots_gate_passed": summary.roots_gate_passed,
            "coverage_gate_passed": summary.coverage_gate_passed,
            "weight_gate_passed": summary.weight_gate_passed,
            "effective_samples_gate_passed": summary.effective_samples_gate_passed,
            "beta_quantile_gate_passed": summary.beta_quantile_gate_passed,
            "lower_bound_gate_passed": summary.lower_bound_gate_passed,
            "terminal_reason": summary.terminal_reason.as_str(),
            "neighbor_count": summary.neighbor_count,
            "neighbor_aggregate_hash": summary.neighbor_aggregate_hash,
        }),
    ])
}

fn neighbor_payload(neighbor: &DecisionNeighborV1) -> Json {
    json!({
        "schema": DECISION_NEIGHBOR_HASH_SCHEMA_V1,
        "decision_id": neighbor.decision_id.to_string(),
        "neighbor_ordinal": neighbor.neighbor_ordinal,
        "candidate_id": neighbor.candidate_id,
        "candidate_neighbor_ordinal": neighbor.candidate_neighbor_ordinal,
        "evidence_vector_link_id": neighbor.evidence_vector_link_id.to_string(),
        "shadow_attempt_id": neighbor.shadow_attempt_id.to_string(),
        "anchor_id": neighbor.anchor_id.to_string(),
        "evaluation_id": option_uuid(neighbor.evaluation_id),
        "learning_generation_id": neighbor.learning_generation_id.to_string(),
        "distance_f32_bits": f32_bits(neighbor.distance),
        "age_seconds_bits": option_f64_bits(neighbor.age_seconds),
        "similarity_weight_bits": option_f64_bits(neighbor.similarity_weight),
        "time_weight_bits": option_f64_bits(neighbor.time_weight),
        "final_weight_bits": option_f64_bits(neighbor.final_weight),
        "binary_label": neighbor.binary_label.map(DecisionBinaryLabelV1::as_str),
        "selected_for_root": neighbor.selected_for_root,
        "root_group_ordinal": neighbor.root_group_ordinal,
        "exclusion_reason": neighbor.exclusion_reason.as_str(),
    })
}

fn f64_bits(value: AuditF64V1) -> String {
    format!("{:016x}", value.bits())
}

fn option_f64_bits(value: Option<AuditF64V1>) -> Json {
    value.map_or(Json::Null, |value| Json::String(f64_bits(value)))
}

fn f32_bits(value: AuditF32V1) -> String {
    format!("{:08x}", value.bits())
}

fn option_uuid(value: Option<Uuid>) -> Json {
    value.map_or(Json::Null, |value| Json::String(value.to_string()))
}

fn option_i64(value: Option<i64>) -> Json {
    value.map_or(Json::Null, |value| Json::String(value.to_string()))
}

fn merge_json_objects<const N: usize>(parts: [Json; N]) -> Json {
    let mut merged = serde_json::Map::new();
    for part in parts {
        let Json::Object(part) = part else {
            unreachable!("decision hash payload fragments are objects");
        };
        merged.extend(part);
    }
    Json::Object(merged)
}

fn build_parent(
    input: DecisionParentInputV1,
    binding: DecisionParentBinding,
    summary_count: usize,
    neighbor_count: usize,
) -> DecisionRecordV1 {
    let (
        decision_shape_version,
        algorithm_version,
        cohort_generation_id,
        active_experiment_id,
        active_authorization_state_event_id,
        root_key,
    ) = match binding {
        DecisionParentBinding::Recommend => (
            DECISION_SHAPE_VERSION_V1,
            DECISION_ALGORITHM_VERSION_V1,
            None,
            None,
            None,
            None,
        ),
        DecisionParentBinding::Active(binding) => (
            ACTIVE_DECISION_SHAPE_VERSION_V2,
            ACTIVE_DECISION_ALGORITHM_VERSION_V2,
            Some(binding.cohort_generation_id),
            binding.active_experiment_id,
            binding.active_authorization_state_event_id,
            Some(binding.root_key),
        ),
    };
    DecisionRecordV1 {
        decision_id: input.decision_id,
        decision_shape_version,
        algorithm_version,
        project_uuid: input.project_uuid,
        process_instance_id: input.process_instance_id,
        config_generation_id: input.config_generation_id,
        policy_version_id: input.policy_version_id,
        learning_generation_id: input.learning_generation_id,
        cohort_generation_id,
        active_experiment_id,
        active_authorization_state_event_id,
        pool_id: input.pool_id,
        candidate_id: input.candidate_id,
        root_key,
        primary_call_uuid: input.primary_call_uuid,
        canonical_query_hash: input.canonical_query_hash,
        partition_base_json: input.partition_base_json,
        partition_base_hash: input.partition_base_hash,
        vector_space_id: input.vector_space_id,
        candidate_set_hash: input.candidate_set_hash,
        candidate_count: summary_count,
        recommended_model: input.recommended_model,
        recommended_model_revision: input.recommended_model_revision,
        served_model: input.served_model,
        served_model_revision: input.served_model_revision,
        as_of_unix_ms: input.as_of_unix_ms,
        decision_latency_ms: input.decision_latency_ms,
        final_reason: input.final_reason,
        summary_count,
        summary_aggregate_hash: String::new(),
        neighbor_count,
        neighbor_aggregate_hash: String::new(),
        aggregate_size_bytes: 1,
        created_at_unix_ms: input.created_at_unix_ms,
        canonical_payload_hash: String::new(),
    }
}

fn build_summary(
    decision_id: Uuid,
    parent: &DecisionParentInputV1,
    input: DecisionCandidateSummaryInputV1,
    partition_hash: &str,
    neighbors: &[DecisionNeighborV1],
) -> Result<DecisionCandidateSummaryV1, DecisionAuditError> {
    let summary = DecisionCandidateSummaryV1 {
        decision_id,
        candidate_id: input.candidate_id,
        rank_ordinal: input.rank_ordinal,
        candidate_model: input.candidate_model,
        candidate_model_revision: input.candidate_model_revision,
        cost_rank: input.cost_rank,
        learning_generation_id: input.learning_generation_id,
        vector_space_id: input.vector_space_id,
        partition_hash: partition_hash.to_string(),
        partition_id: input.partition_id,
        decoding_fingerprint: input.decoding_fingerprint,
        top_k: input.top_k,
        radius: input.radius,
        min_points: input.min_points,
        min_independent_roots: input.min_independent_roots,
        min_effective_samples: input.min_effective_samples,
        min_coverage: input.min_coverage,
        time_decay_half_life_seconds: input.time_decay_half_life_seconds,
        prior_success: input.prior_success,
        prior_failure: input.prior_failure,
        familywise_credible_level: input.familywise_credible_level,
        candidate_alpha: input.candidate_alpha,
        promotion_lower_bound: input.promotion_lower_bound,
        returned_neighbor_count: input.returned_neighbor_count,
        within_radius_count: input.within_radius_count,
        labeled_point_count: input.labeled_point_count,
        attempted_root_count: input.attempted_root_count,
        labeled_root_count: input.labeled_root_count,
        selected_root_count: input.selected_root_count,
        coverage: input.coverage,
        sum_weight: input.sum_weight,
        sum_weighted_label: input.sum_weighted_label,
        sum_squared_weight: input.sum_squared_weight,
        p_hat: input.p_hat,
        effective_sample_size: input.effective_sample_size,
        beta_alpha: input.beta_alpha,
        beta_beta: input.beta_beta,
        lower_bound: input.lower_bound,
        partition_gate_passed: input.partition_gate_passed,
        points_gate_passed: input.points_gate_passed,
        roots_gate_passed: input.roots_gate_passed,
        coverage_gate_passed: input.coverage_gate_passed,
        weight_gate_passed: input.weight_gate_passed,
        effective_samples_gate_passed: input.effective_samples_gate_passed,
        beta_quantile_gate_passed: input.beta_quantile_gate_passed,
        lower_bound_gate_passed: input.lower_bound_gate_passed,
        terminal_reason: input.terminal_reason,
        neighbor_count: neighbors.len(),
        neighbor_aggregate_hash: String::new(),
        canonical_payload_hash: String::new(),
    };
    validate_summary(parent, &summary, neighbors)?;
    Ok(summary)
}

#[allow(clippy::cast_precision_loss)]
fn build_neighbor(
    decision_id: Uuid,
    learning_generation_id: &Uuid,
    candidate_id: &str,
    expected_neighbor_ordinal: usize,
    input: DecisionNeighborInputV1,
) -> Result<DecisionNeighborV1, DecisionAuditError> {
    if input.neighbor_ordinal != expected_neighbor_ordinal
        || input.candidate_id != candidate_id
        || &input.learning_generation_id != learning_generation_id
    {
        return Err(DecisionAuditError::InvalidOrdering);
    }
    let age_seconds = input
        .age_millis
        .map(|age_millis| AuditF64V1::new(age_millis as f64 / 1000.0))
        .transpose()?;
    let neighbor = DecisionNeighborV1 {
        decision_id,
        neighbor_ordinal: input.neighbor_ordinal,
        candidate_id: input.candidate_id,
        candidate_neighbor_ordinal: input.candidate_neighbor_ordinal,
        evidence_vector_link_id: input.evidence_vector_link_id,
        shadow_attempt_id: input.shadow_attempt_id,
        anchor_id: input.anchor_id,
        evaluation_id: input.evaluation_id,
        learning_generation_id: input.learning_generation_id,
        distance: input.distance,
        age_seconds,
        similarity_weight: input.similarity_weight,
        time_weight: input.time_weight,
        final_weight: input.final_weight,
        binary_label: input.binary_label,
        selected_for_root: input.selected_for_root,
        root_group_ordinal: input.root_group_ordinal,
        exclusion_reason: input.exclusion_reason,
        canonical_payload_hash: String::new(),
    };
    validate_neighbor(&neighbor)?;
    Ok(neighbor)
}

fn validate_parent_input(
    parent: &DecisionParentInputV1,
    binding: &DecisionParentBinding,
    candidate_count: usize,
    prepared_query: &PreparedCanonicalQueryV1,
) -> Result<(), DecisionAuditError> {
    if !(1..=DECISION_SUMMARY_COUNT_MAX).contains(&candidate_count)
        || ![
            parent.decision_id,
            parent.project_uuid,
            parent.process_instance_id,
            parent.learning_generation_id,
            parent.primary_call_uuid,
        ]
        .into_iter()
        .all(is_uuid_v7)
        || parent.as_of_unix_ms < 0
        || parent.created_at_unix_ms < 0
    {
        return Err(DecisionAuditError::InvalidIdentity);
    }
    for hash in [
        &parent.config_generation_id,
        &parent.policy_version_id,
        &parent.canonical_query_hash,
        &parent.partition_base_hash,
        &parent.vector_space_id,
        &parent.candidate_set_hash,
    ] {
        validate_hash(hash)?;
    }
    validate_text(&parent.pool_id, ID_MAX_BYTES)?;
    if let Some(candidate_id) = &parent.candidate_id {
        validate_text(candidate_id, CANDIDATE_ID_MAX_BYTES)?;
    }
    validate_text(&parent.recommended_model, MODEL_ID_MAX_BYTES)?;
    validate_text(&parent.recommended_model_revision, REVISION_MAX_BYTES)?;
    validate_text(&parent.served_model, MODEL_ID_MAX_BYTES)?;
    validate_text(&parent.served_model_revision, REVISION_MAX_BYTES)?;
    validate_canonical_document(&parent.partition_base_hash, &parent.partition_base_json)?;
    if parent.canonical_query_hash != prepared_query.canonical_query_hash() {
        return Err(DecisionAuditError::InvalidState);
    }
    let winner_coherent = match binding {
        DecisionParentBinding::Recommend => match parent.final_reason {
            DecisionFinalReasonV1::RecommendObserveOnly => parent.candidate_id.is_some(),
            reason if !reason.is_active() => {
                parent.candidate_id.is_none()
                    && parent.recommended_model == parent.served_model
                    && parent.recommended_model_revision == parent.served_model_revision
            }
            _ => false,
        },
        DecisionParentBinding::Active(binding) => {
            parent.final_reason.is_active()
                && is_uuid_v7(binding.cohort_generation_id)
                && binding.active_experiment_id.is_none_or(is_uuid_v7)
                && binding
                    .active_authorization_state_event_id
                    .is_none_or(is_uuid_v7)
                && binding.active_experiment_id.is_some()
                    == binding.active_authorization_state_event_id.is_some()
                && validate_hash(&binding.root_key).is_ok()
                && active_parent_route_is_coherent(parent)
        }
    };
    if !winner_coherent {
        return Err(DecisionAuditError::InvalidState);
    }
    Ok(())
}

fn active_parent_route_is_coherent(parent: &DecisionParentInputV1) -> bool {
    match parent.final_reason {
        DecisionFinalReasonV1::ActiveCandidate => {
            parent.candidate_id.is_some()
                && parent.recommended_model == parent.served_model
                && parent.recommended_model_revision == parent.served_model_revision
        }
        DecisionFinalReasonV1::ActiveAnchorControl | DecisionFinalReasonV1::ActiveAnchorHoldout => {
            parent.candidate_id.is_some()
        }
        reason if reason.is_active() => {
            parent.candidate_id.is_some()
                || (parent.recommended_model == parent.served_model
                    && parent.recommended_model_revision == parent.served_model_revision)
        }
        _ => false,
    }
}

fn validate_partition_artifact(
    parent: &DecisionParentInputV1,
    base: &RoutingPartitionBaseV1,
    summary: &DecisionCandidateSummaryInputV1,
    artifact: &RoutingPartitionArtifactV1,
) -> Result<(), DecisionAuditError> {
    validate_partition_base(base)?;
    validate_canonical_document(&artifact.partition_hash, &artifact.canonical_json)?;
    let partition = &artifact.partition;
    let partition_value = serde_json::to_value(partition)
        .map_err(|_| DecisionAuditError::InvalidCanonicalDocument)?;
    if canonical_json(&partition_value).map_err(|_| DecisionAuditError::InvalidCanonicalDocument)?
        != artifact.canonical_json
    {
        return Err(DecisionAuditError::InvalidCanonicalDocument);
    }
    if base.policy_version_id != parent.policy_version_id
        || base.learning_generation_id != parent.learning_generation_id
        || base.vector_space_id != parent.vector_space_id
        || partition.tenant_policy_hash != base.tenant_policy_hash
        || partition.agent_policy_hash != base.agent_policy_hash
        || partition.policy_version_id != base.policy_version_id
        || partition.learning_generation_id != base.learning_generation_id
        || partition.api_family != base.api_family
        || partition.transport_identity != base.transport_identity
        || partition.anchor_model != base.anchor_model
        || partition.anchor_revision != base.anchor_revision
        || partition.evaluator_version != base.evaluator_version
        || partition.vector_space_id != base.vector_space_id
        || partition.candidate_id != summary.candidate_id
        || partition.candidate_model != summary.candidate_model
        || partition.candidate_model_revision != summary.candidate_model_revision
        || partition.decoding_fingerprint != summary.decoding_fingerprint
    {
        return Err(DecisionAuditError::InvalidState);
    }
    Ok(())
}

fn validate_partition_base(base: &RoutingPartitionBaseV1) -> Result<(), DecisionAuditError> {
    for hash in [
        &base.tenant_policy_hash,
        &base.agent_policy_hash,
        &base.policy_version_id,
        &base.evaluator_version,
        &base.vector_space_id,
    ] {
        validate_hash(hash)?;
    }
    if !is_uuid_v7(base.learning_generation_id) {
        return Err(DecisionAuditError::InvalidIdentity);
    }
    validate_text(
        &base.transport_identity,
        PARTITION_TRANSPORT_IDENTITY_MAX_BYTES,
    )?;
    validate_text(&base.anchor_model, MODEL_ID_MAX_BYTES)?;
    validate_text(&base.anchor_revision, REVISION_MAX_BYTES)
}

fn validate_summary(
    parent: &DecisionParentInputV1,
    summary: &DecisionCandidateSummaryV1,
    neighbors: &[DecisionNeighborV1],
) -> Result<(), DecisionAuditError> {
    validate_summary_common(
        parent.decision_id,
        parent.learning_generation_id,
        &parent.vector_space_id,
        summary,
        neighbors,
    )
}

#[allow(clippy::cast_precision_loss)]
fn validate_summary_common(
    decision_id: Uuid,
    learning_generation_id: Uuid,
    vector_space_id: &str,
    summary: &DecisionCandidateSummaryV1,
    neighbors: &[DecisionNeighborV1],
) -> Result<(), DecisionAuditError> {
    if summary.decision_id != decision_id
        || summary.learning_generation_id != learning_generation_id
        || summary.vector_space_id != vector_space_id
        || summary.rank_ordinal >= DECISION_SUMMARY_COUNT_MAX
        || summary.partition_id.is_some_and(|value| value <= 0)
    {
        return Err(DecisionAuditError::InvalidIdentity);
    }
    validate_text(&summary.candidate_id, CANDIDATE_ID_MAX_BYTES)?;
    validate_text(&summary.candidate_model, MODEL_ID_MAX_BYTES)?;
    validate_text(&summary.candidate_model_revision, REVISION_MAX_BYTES)?;
    validate_hash(&summary.partition_hash)?;
    validate_hash(&summary.decoding_fingerprint)?;
    let configuration_valid = (1..=DECISION_NEIGHBOR_COUNT_MAX).contains(&summary.top_k)
        && 0.0 < summary.radius.value()
        && summary.radius.value() <= 2.0
        && (1..=summary.top_k).contains(&summary.min_points)
        && (1..=summary.top_k).contains(&summary.min_independent_roots)
        && 0.0 < summary.min_effective_samples.value()
        && summary.min_effective_samples.value() <= summary.top_k as f64
        && in_unit_interval(summary.min_coverage)
        && summary.time_decay_half_life_seconds.value() > 0.0
        && summary.prior_success.value() > 0.0
        && summary.prior_failure.value() > 0.0
        && 0.5 < summary.familywise_credible_level.value()
        && summary.familywise_credible_level.value() < 1.0
        && 0.0 < summary.candidate_alpha.value()
        && summary.candidate_alpha.value() < 0.5
        && in_unit_interval(summary.promotion_lower_bound);
    if !configuration_valid {
        return Err(DecisionAuditError::InvalidState);
    }
    if neighbors.len() > summary.top_k
        || summary.returned_neighbor_count != neighbors.len()
        || summary.neighbor_count != neighbors.len()
    {
        return Err(DecisionAuditError::InvalidCounts);
    }
    for (index, neighbor) in neighbors.iter().enumerate() {
        if neighbor.candidate_id != summary.candidate_id
            || neighbor.candidate_neighbor_ordinal != index
        {
            return Err(DecisionAuditError::InvalidOrdering);
        }
        validate_neighbor(neighbor)?;
    }
    for pair in neighbors.windows(2) {
        let distance_order = if pair[0].distance.value() < pair[1].distance.value() {
            std::cmp::Ordering::Less
        } else if pair[0].distance.value() > pair[1].distance.value() {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        };
        if distance_order
            .then_with(|| {
                pair[0]
                    .evidence_vector_link_id
                    .cmp(&pair[1].evidence_vector_link_id)
            })
            .is_gt()
        {
            return Err(DecisionAuditError::InvalidOrdering);
        }
    }

    let within = neighbors
        .iter()
        .filter(|neighbor| {
            neighbor.exclusion_reason != DecisionNeighborExclusionReasonV1::OutsideRadius
        })
        .collect::<Vec<_>>();
    let labeled = within
        .iter()
        .copied()
        .filter(|neighbor| {
            matches!(
                neighbor.exclusion_reason,
                DecisionNeighborExclusionReasonV1::DuplicateRoot
                    | DecisionNeighborExclusionReasonV1::Included
            )
        })
        .collect::<Vec<_>>();
    let attempted_roots = within
        .iter()
        .map(|neighbor| neighbor.root_group_ordinal)
        .collect::<BTreeSet<_>>();
    let labeled_roots = labeled
        .iter()
        .map(|neighbor| neighbor.root_group_ordinal)
        .collect::<BTreeSet<_>>();
    let selected = neighbors
        .iter()
        .filter(|neighbor| neighbor.selected_for_root)
        .count();
    if summary.within_radius_count != within.len()
        || summary.labeled_point_count != labeled.len()
        || summary.attempted_root_count != attempted_roots.len()
        || summary.labeled_root_count != labeled_roots.len()
        || summary.selected_root_count != selected
        || summary.labeled_root_count > summary.labeled_point_count
        || summary.labeled_root_count > summary.attempted_root_count
    {
        return Err(DecisionAuditError::InvalidCounts);
    }
    validate_summary_state(summary)?;
    Ok(())
}

fn validate_neighbor(neighbor: &DecisionNeighborV1) -> Result<(), DecisionAuditError> {
    if ![
        neighbor.decision_id,
        neighbor.evidence_vector_link_id,
        neighbor.shadow_attempt_id,
        neighbor.anchor_id,
        neighbor.learning_generation_id,
    ]
    .into_iter()
    .all(is_uuid_v7)
        || neighbor
            .evaluation_id
            .is_some_and(|value| !is_uuid_v7(value))
        || neighbor.neighbor_ordinal >= DECISION_NEIGHBOR_COUNT_MAX
        || neighbor.candidate_neighbor_ordinal >= DECISION_NEIGHBOR_COUNT_MAX
        || neighbor.root_group_ordinal >= DECISION_NEIGHBOR_COUNT_MAX
        || !(0.0..=2.0).contains(&neighbor.distance.value())
    {
        return Err(DecisionAuditError::InvalidIdentity);
    }
    validate_text(&neighbor.candidate_id, CANDIDATE_ID_MAX_BYTES)?;
    if neighbor.binary_label.is_some() && neighbor.evaluation_id.is_none() {
        return Err(DecisionAuditError::InvalidState);
    }
    if neighbor.selected_for_root
        != (neighbor.exclusion_reason == DecisionNeighborExclusionReasonV1::Included)
    {
        return Err(DecisionAuditError::InvalidState);
    }
    let computed_presence = [
        neighbor.age_seconds.is_some(),
        neighbor.similarity_weight.is_some(),
        neighbor.time_weight.is_some(),
        neighbor.final_weight.is_some(),
    ];
    if computed_presence.iter().any(|present| *present)
        && !computed_presence.iter().all(|present| *present)
    {
        return Err(DecisionAuditError::InvalidState);
    }
    if neighbor
        .age_seconds
        .is_some_and(|value| value.value() < 0.0)
        || [
            neighbor.similarity_weight,
            neighbor.time_weight,
            neighbor.final_weight,
        ]
        .into_iter()
        .flatten()
        .any(|value| !in_unit_interval(value))
    {
        return Err(DecisionAuditError::InvalidState);
    }
    Ok(())
}

#[allow(clippy::cast_precision_loss)]
fn validate_summary_state(summary: &DecisionCandidateSummaryV1) -> Result<(), DecisionAuditError> {
    type Gates = [Option<bool>; 8];
    const N: Option<bool> = None;
    const P: Option<bool> = Some(true);
    const F: Option<bool> = Some(false);
    let gates: Gates = [
        summary.partition_gate_passed,
        summary.points_gate_passed,
        summary.roots_gate_passed,
        summary.coverage_gate_passed,
        summary.weight_gate_passed,
        summary.effective_samples_gate_passed,
        summary.beta_quantile_gate_passed,
        summary.lower_bound_gate_passed,
    ];
    let state_matches = match summary.terminal_reason {
        DecisionCandidateReasonV1::NoPartition => gates == [F, N, N, N, N, N, N, N],
        DecisionCandidateReasonV1::SparsePoints => gates == [P, F, N, N, N, N, N, N],
        DecisionCandidateReasonV1::InsufficientRoots => gates == [P, P, F, N, N, N, N, N],
        DecisionCandidateReasonV1::LowCoverage => gates == [P, P, P, F, N, N, N, N],
        DecisionCandidateReasonV1::InvalidEvidenceTime => gates == [P, P, P, P, F, N, N, N],
        DecisionCandidateReasonV1::NumericError => {
            gates == [P, P, P, P, F, N, N, N] || gates == [P, P, P, P, P, P, F, N]
        }
        DecisionCandidateReasonV1::InsufficientEffectiveSamples => {
            gates == [P, P, P, P, P, F, N, N]
        }
        DecisionCandidateReasonV1::LowerBoundBelowThreshold => gates == [P, P, P, P, P, P, P, F],
        DecisionCandidateReasonV1::Passed => gates == [P, P, P, P, P, P, P, P],
        DecisionCandidateReasonV1::NotEvaluatedAfterWinner
        | DecisionCandidateReasonV1::NotEvaluatedAfterFallback => gates == [N, N, N, N, N, N, N, N],
    };
    if !state_matches {
        return Err(DecisionAuditError::InvalidState);
    }

    let no_partition = summary.terminal_reason == DecisionCandidateReasonV1::NoPartition;
    let not_evaluated = matches!(
        summary.terminal_reason,
        DecisionCandidateReasonV1::NotEvaluatedAfterWinner
            | DecisionCandidateReasonV1::NotEvaluatedAfterFallback
    );
    if (no_partition || not_evaluated) != summary.partition_id.is_none() {
        return Err(DecisionAuditError::InvalidState);
    }
    if no_partition || not_evaluated {
        if summary.returned_neighbor_count != 0
            || summary.within_radius_count != 0
            || summary.labeled_point_count != 0
            || summary.attempted_root_count != 0
            || summary.labeled_root_count != 0
            || summary.selected_root_count != 0
            || summary.coverage.is_some()
        {
            return Err(DecisionAuditError::InvalidState);
        }
    } else {
        let expected_coverage = if summary.attempted_root_count == 0 {
            0.0
        } else {
            summary.labeled_root_count as f64 / summary.attempted_root_count as f64
        };
        require_close(
            summary
                .coverage
                .ok_or(DecisionAuditError::InvalidState)?
                .value(),
            expected_coverage,
        )?;
    }

    let weight_metrics = [
        summary.sum_weight,
        summary.sum_weighted_label,
        summary.sum_squared_weight,
        summary.p_hat,
        summary.effective_sample_size,
    ];
    let weights_present = weight_metrics.iter().all(Option::is_some);
    if weight_metrics.iter().any(Option::is_some) != weights_present
        || weights_present != (summary.weight_gate_passed == Some(true))
    {
        return Err(DecisionAuditError::InvalidState);
    }
    if summary.sum_weight.is_some_and(|value| value.value() < 0.0)
        || summary
            .sum_weighted_label
            .is_some_and(|value| value.value() < 0.0)
        || summary
            .sum_squared_weight
            .is_some_and(|value| value.value() < 0.0)
        || summary.p_hat.is_some_and(|value| !in_unit_interval(value))
        || summary
            .effective_sample_size
            .is_some_and(|value| value.value() < 0.0)
        || summary
            .sum_weighted_label
            .zip(summary.sum_weight)
            .is_some_and(|(weighted, total)| weighted.value() > total.value())
    {
        return Err(DecisionAuditError::InvalidState);
    }

    let beta_present = summary.beta_alpha.is_some() && summary.beta_beta.is_some();
    if summary.beta_alpha.is_some() != summary.beta_beta.is_some()
        || (summary.beta_quantile_gate_passed.is_none() && beta_present)
        || (summary.beta_quantile_gate_passed == Some(true) && !beta_present)
        || summary.beta_alpha.is_some_and(|value| value.value() <= 0.0)
        || summary.beta_beta.is_some_and(|value| value.value() <= 0.0)
        || summary.lower_bound.is_some() != (summary.beta_quantile_gate_passed == Some(true))
        || summary
            .lower_bound
            .is_some_and(|value| !in_unit_interval(value))
    {
        return Err(DecisionAuditError::InvalidState);
    }
    Ok(())
}

fn validate_hash(value: &str) -> Result<(), DecisionAuditError> {
    (value.len() == 64
        && value
            .as_bytes()
            .iter()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(byte)))
    .then_some(())
    .ok_or(DecisionAuditError::InvalidIdentity)
}

fn validate_text(value: &str, maximum_bytes: usize) -> Result<(), DecisionAuditError> {
    (!value.trim().is_empty()
        && value.len() <= maximum_bytes
        && !value.chars().any(char::is_control)
        && value.nfc().eq(value.chars()))
    .then_some(())
    .ok_or(DecisionAuditError::InvalidText)
}

fn validate_canonical_document(hash: &str, document: &str) -> Result<(), DecisionAuditError> {
    validate_hash(hash)?;
    let parsed: Json =
        serde_json::from_str(document).map_err(|_| DecisionAuditError::InvalidCanonicalDocument)?;
    let canonical =
        canonical_json(&parsed).map_err(|_| DecisionAuditError::InvalidCanonicalDocument)?;
    if canonical != document || sha256_hex(document.as_bytes()) != hash {
        return Err(DecisionAuditError::InvalidCanonicalDocument);
    }
    Ok(())
}

fn is_uuid_v7(value: Uuid) -> bool {
    value.get_variant() == Variant::RFC4122 && value.get_version_num() == 7
}

fn in_unit_interval(value: AuditF64V1) -> bool {
    (0.0..=1.0).contains(&value.value())
}

#[cfg(test)]
mod tests {
    use nemo_relay_types::api::llm::LlmApiFamily;

    use super::*;
    use crate::canonical_query::CanonicalTaskV1;
    use crate::fingerprint::canonical_serialize_bytes;
    use crate::routing_partition::{
        RoutingPartitionInputV1, build_routing_partition_base_v1,
        build_routing_partition_from_input_v1,
    };

    fn hash(byte: char) -> String {
        byte.to_string().repeat(64)
    }

    fn uuid(suffix: u16) -> Uuid {
        Uuid::parse_str(&format!("018f1e66-0000-7000-8000-{suffix:012x}")).unwrap()
    }

    fn prepared_query() -> PreparedCanonicalQueryV1 {
        let query = CanonicalRoutingQueryV1 {
            schema: CANONICAL_ROUTING_QUERY_SCHEMA_V1.to_string(),
            instructions: Vec::new(),
            current_task: CanonicalTaskV1 {
                text: "route this request".to_string(),
            },
            bounded_context: Vec::new(),
            tool_schema_fingerprint: hash('1'),
            response_schema_fingerprint: None,
            required_capabilities: Vec::new(),
            position_features: None,
        };
        let canonical_bytes = canonical_serialize_bytes(&query).unwrap();
        let canonical_query_hash = sha256_hex(&canonical_bytes);
        PreparedCanonicalQueryV1::from_artifact(CanonicalRoutingQueryArtifactV1 {
            query,
            canonical_bytes,
            canonical_query_hash,
        })
        .unwrap()
    }

    fn valid_audit() -> DecisionAuditV1 {
        valid_audit_with_duplicate_neighbor(false)
    }

    fn valid_audit_with_duplicate_neighbor(include_duplicate: bool) -> DecisionAuditV1 {
        let learning_generation_id = uuid(4);
        let base = RoutingPartitionBaseV1 {
            tenant_policy_hash: hash('a'),
            agent_policy_hash: hash('b'),
            policy_version_id: hash('c'),
            learning_generation_id,
            api_family: LlmApiFamily::OpenAIChatCompletions,
            transport_identity: "transport-v1".to_string(),
            anchor_model: "anchor".to_string(),
            anchor_revision: "anchor-r1".to_string(),
            evaluator_version: hash('e'),
            vector_space_id: hash('6'),
        };
        let base_artifact = build_routing_partition_base_v1(&base).unwrap();
        let decoding_fingerprint = hash('d');
        let partition = build_routing_partition_from_input_v1(&RoutingPartitionInputV1 {
            base,
            candidate_id: "candidate-a".to_string(),
            candidate_model: "model-a".to_string(),
            candidate_model_revision: "r1".to_string(),
            decoding_fingerprint: decoding_fingerprint.clone(),
        })
        .unwrap();
        let candidate_set = build_candidate_set_from_members_v1(&[CandidateSetMemberInputV1 {
            candidate_id: "candidate-a".to_string(),
            model: "model-a".to_string(),
            model_revision: "r1".to_string(),
            cost_rank: 1,
        }])
        .unwrap();
        let prepared_query = prepared_query();
        let canonical_query_hash = prepared_query.canonical_query_hash().to_string();
        let familywise = 0.95;
        let candidate_alpha = 1.0 - familywise;
        let beta_alpha = 2.0;
        let beta_beta = 1.0;
        let lower_bound = beta_inverse_cdf_v1(candidate_alpha, beta_alpha, beta_beta).unwrap();
        let parent = DecisionParentInputV1 {
            decision_id: uuid(1),
            project_uuid: uuid(2),
            process_instance_id: uuid(3),
            config_generation_id: hash('7'),
            policy_version_id: hash('c'),
            learning_generation_id,
            pool_id: "pool-a".to_string(),
            candidate_id: Some("candidate-a".to_string()),
            primary_call_uuid: uuid(5),
            canonical_query_hash,
            partition_base_json: base_artifact.canonical_json,
            partition_base_hash: base_artifact.partition_base_hash,
            vector_space_id: hash('6'),
            candidate_set_hash: candidate_set.candidate_set_hash,
            recommended_model: "model-a".to_string(),
            recommended_model_revision: "r1".to_string(),
            served_model: "anchor".to_string(),
            served_model_revision: "anchor-r1".to_string(),
            as_of_unix_ms: 1_000,
            decision_latency_ms: 2,
            final_reason: DecisionFinalReasonV1::RecommendObserveOnly,
            created_at_unix_ms: 1_002,
        };
        let neighbor_count = if include_duplicate { 2 } else { 1 };
        let summary = DecisionCandidateSummaryInputV1 {
            candidate_id: "candidate-a".to_string(),
            rank_ordinal: 0,
            candidate_model: "model-a".to_string(),
            candidate_model_revision: "r1".to_string(),
            cost_rank: 1,
            learning_generation_id,
            vector_space_id: hash('6'),
            partition_id: Some(1),
            decoding_fingerprint,
            top_k: neighbor_count,
            radius: AuditF64V1::new(1.0).unwrap(),
            min_points: 1,
            min_independent_roots: 1,
            min_effective_samples: AuditF64V1::new(1.0).unwrap(),
            min_coverage: AuditF64V1::new(1.0).unwrap(),
            time_decay_half_life_seconds: AuditF64V1::new(3_600.0).unwrap(),
            prior_success: AuditF64V1::new(1.0).unwrap(),
            prior_failure: AuditF64V1::new(1.0).unwrap(),
            familywise_credible_level: AuditF64V1::new(familywise).unwrap(),
            candidate_alpha: AuditF64V1::new(candidate_alpha).unwrap(),
            promotion_lower_bound: AuditF64V1::new(0.2).unwrap(),
            returned_neighbor_count: neighbor_count,
            within_radius_count: neighbor_count,
            labeled_point_count: neighbor_count,
            attempted_root_count: 1,
            labeled_root_count: 1,
            selected_root_count: 1,
            coverage: Some(AuditF64V1::new(1.0).unwrap()),
            sum_weight: Some(AuditF64V1::new(1.0).unwrap()),
            sum_weighted_label: Some(AuditF64V1::new(1.0).unwrap()),
            sum_squared_weight: Some(AuditF64V1::new(1.0).unwrap()),
            p_hat: Some(AuditF64V1::new(1.0).unwrap()),
            effective_sample_size: Some(AuditF64V1::new(1.0).unwrap()),
            beta_alpha: Some(AuditF64V1::new(beta_alpha).unwrap()),
            beta_beta: Some(AuditF64V1::new(beta_beta).unwrap()),
            lower_bound: Some(AuditF64V1::new(lower_bound).unwrap()),
            partition_gate_passed: Some(true),
            points_gate_passed: Some(true),
            roots_gate_passed: Some(true),
            coverage_gate_passed: Some(true),
            weight_gate_passed: Some(true),
            effective_samples_gate_passed: Some(true),
            beta_quantile_gate_passed: Some(true),
            lower_bound_gate_passed: Some(true),
            terminal_reason: DecisionCandidateReasonV1::Passed,
        };
        let neighbor = DecisionNeighborInputV1 {
            neighbor_ordinal: 0,
            candidate_id: "candidate-a".to_string(),
            candidate_neighbor_ordinal: 0,
            evidence_vector_link_id: uuid(6),
            shadow_attempt_id: uuid(7),
            anchor_id: uuid(8),
            evaluation_id: Some(uuid(9)),
            learning_generation_id,
            distance: AuditF32V1::new(0.0).unwrap(),
            age_millis: Some(0),
            similarity_weight: Some(AuditF64V1::new(1.0).unwrap()),
            time_weight: Some(AuditF64V1::new(1.0).unwrap()),
            final_weight: Some(AuditF64V1::new(1.0).unwrap()),
            binary_label: Some(DecisionBinaryLabelV1::Pass),
            selected_for_root: true,
            root_group_ordinal: 0,
            exclusion_reason: DecisionNeighborExclusionReasonV1::Included,
        };
        let mut neighbors = vec![neighbor];
        if include_duplicate {
            neighbors.push(DecisionNeighborInputV1 {
                neighbor_ordinal: 1,
                candidate_id: "candidate-a".to_string(),
                candidate_neighbor_ordinal: 1,
                evidence_vector_link_id: uuid(10),
                shadow_attempt_id: uuid(11),
                anchor_id: uuid(12),
                evaluation_id: Some(uuid(13)),
                learning_generation_id,
                distance: AuditF32V1::new(0.0).unwrap(),
                age_millis: None,
                similarity_weight: None,
                time_weight: None,
                final_weight: None,
                binary_label: Some(DecisionBinaryLabelV1::Pass),
                selected_for_root: false,
                root_group_ordinal: 0,
                exclusion_reason: DecisionNeighborExclusionReasonV1::DuplicateRoot,
            });
        }
        DecisionAuditV1::new(
            parent,
            vec![DecisionCandidateInputV1 {
                summary,
                partition_artifact: partition,
                neighbors,
            }],
            prepared_query,
        )
        .unwrap()
    }

    #[test]
    fn constructor_accepts_multiple_neighbors_for_one_candidate() {
        let audit = valid_audit_with_duplicate_neighbor(true);
        assert_eq!(audit.parent.neighbor_count, 2);
        assert_eq!(
            audit
                .neighbors
                .iter()
                .map(|neighbor| neighbor.neighbor_ordinal)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        audit.validate_frozen().unwrap();
    }

    #[test]
    fn constructor_hashes_exact_graph_and_stored_replay_matches() {
        let audit = valid_audit();
        assert!(audit.validate_frozen().is_ok());
        assert_eq!(audit.parent.summary_count, 1);
        assert_eq!(audit.parent.neighbor_count, 1);
        assert!(audit.parent.aggregate_size_bytes > 0);
        assert!(audit.command_size_bytes > audit.parent.aggregate_size_bytes);
        validate_hash(&audit.parent.canonical_payload_hash).unwrap();
        validate_hash(&audit.summaries[0].canonical_payload_hash).unwrap();
        validate_hash(&audit.neighbors[0].canonical_payload_hash).unwrap();
        let exact_graph_bytes = hash_payload(parent_payload(&audit.parent)).unwrap().1
            + audit
                .summaries
                .iter()
                .map(|summary| hash_payload(summary_payload(summary)).unwrap().1)
                .sum::<usize>()
            + audit
                .neighbors
                .iter()
                .map(|neighbor| hash_payload(neighbor_payload(neighbor)).unwrap().1)
                .sum::<usize>();
        assert_eq!(audit.parent.aggregate_size_bytes, exact_graph_bytes);

        let stored = audit.stored_graph();
        assert!(audit.persisted_eq(stored).unwrap());
    }

    #[test]
    fn command_size_accepts_exact_limit_and_rejects_one_byte_over() {
        assert_eq!(
            checked_command_size(DECISION_AUDIT_BYTES_MAX - 2, 1, std::iter::once(1),),
            Ok(DECISION_AUDIT_BYTES_MAX)
        );
        assert_eq!(
            checked_command_size(DECISION_AUDIT_BYTES_MAX - 2, 1, std::iter::once(2),),
            Err(DecisionAuditError::SizeLimitExceeded)
        );
        assert_eq!(
            checked_command_size(usize::MAX, 1, std::iter::empty()),
            Err(DecisionAuditError::SizeOverflow)
        );
    }

    #[test]
    fn stored_replay_rejects_hash_membership_and_root_selection_tampering() {
        let audit = valid_audit();
        let mut hash_tamper = audit.stored_graph();
        hash_tamper.neighbors[0].canonical_payload_hash = hash('0');
        assert_eq!(hash_tamper.verify(), Err(DecisionAuditError::HashMismatch));

        let mut membership_tamper = audit.stored_graph();
        membership_tamper.parent.candidate_set_hash = hash('0');
        assert_eq!(
            membership_tamper.verify(),
            Err(DecisionAuditError::HashMismatch)
        );

        let mut root_tamper = audit.stored_graph();
        root_tamper.neighbors[0].selected_for_root = false;
        root_tamper.neighbors[0].exclusion_reason =
            DecisionNeighborExclusionReasonV1::DuplicateRoot;
        root_tamper.summaries[0].selected_root_count = 0;
        assert_eq!(root_tamper.verify(), Err(DecisionAuditError::InvalidCounts));

        let mut masked_summary = audit.summaries[0].clone();
        masked_summary.top_k = 3;
        masked_summary.returned_neighbor_count = 3;
        masked_summary.neighbor_count = 3;
        masked_summary.within_radius_count = 3;
        masked_summary.labeled_point_count = 3;
        masked_summary.attempted_root_count = 2;
        masked_summary.labeled_root_count = 2;
        masked_summary.selected_root_count = 2;
        let first = audit.neighbors[0].clone();
        let mut second = first.clone();
        second.candidate_neighbor_ordinal = 1;
        second.neighbor_ordinal = 1;
        second.evidence_vector_link_id = uuid(10);
        second.evaluation_id = Some(uuid(11));
        let mut third = first.clone();
        third.candidate_neighbor_ordinal = 2;
        third.neighbor_ordinal = 2;
        third.evidence_vector_link_id = uuid(12);
        third.evaluation_id = Some(uuid(13));
        third.root_group_ordinal = 1;
        third.selected_for_root = false;
        third.exclusion_reason = DecisionNeighborExclusionReasonV1::DuplicateRoot;
        assert_eq!(
            replay_summary(&masked_summary, &[first, second, third], 1),
            Err(DecisionAuditError::InvalidCounts)
        );
    }

    #[test]
    fn fallback_progression_preserves_an_ordinary_evaluated_prefix() {
        let audit = valid_audit();
        let mut ordinary = audit.summaries[0].clone();
        ordinary.terminal_reason = DecisionCandidateReasonV1::SparsePoints;
        let mut fallback = ordinary.clone();
        fallback.candidate_id = "candidate-b".to_string();
        fallback.terminal_reason = DecisionCandidateReasonV1::NotEvaluatedAfterFallback;
        let summaries = vec![ordinary, fallback];
        let mut parent = audit.parent.clone();
        parent.candidate_id = None;
        parent.recommended_model = parent.served_model.clone();
        parent.recommended_model_revision = parent.served_model_revision.clone();
        for reason in [
            DecisionFinalReasonV1::VectorUnhealthy,
            DecisionFinalReasonV1::VersionMismatch,
        ] {
            parent.final_reason = reason;
            assert!(validate_decision_progression(&parent, &summaries).is_ok());
        }
        parent.final_reason = DecisionFinalReasonV1::EmbeddingUnavailable;
        assert_eq!(
            validate_decision_progression(&parent, &summaries),
            Err(DecisionAuditError::InvalidState)
        );
    }

    #[test]
    fn replay_accepts_only_reproduced_uncertified_beta_numeric_failure() {
        let audit = valid_audit();
        let mut summary = audit.summaries[0].clone();
        let extreme = AuditF64V1::new(1.0e308).unwrap();
        summary.prior_success = extreme;
        summary.prior_failure = extreme;
        summary.beta_alpha = Some(extreme);
        summary.beta_beta = Some(extreme);
        summary.lower_bound = None;
        summary.beta_quantile_gate_passed = Some(false);
        summary.lower_bound_gate_passed = None;
        summary.terminal_reason = DecisionCandidateReasonV1::NumericError;
        assert!(
            beta_inverse_cdf_v1(
                summary.candidate_alpha.value(),
                extreme.value(),
                extreme.value(),
            )
            .is_err()
        );
        assert!(replay_summary(&summary, &audit.neighbors, 1).is_ok());

        summary.beta_quantile_gate_passed = Some(true);
        assert_eq!(
            replay_summary(&summary, &audit.neighbors, 1),
            Err(DecisionAuditError::InvalidState)
        );
    }

    #[test]
    fn pre_weight_terminal_requires_every_computed_neighbor_phase_to_be_null() {
        let audit = valid_audit();
        let mut summary = audit.summaries[0].clone();
        summary.top_k = 2;
        summary.min_points = 2;
        summary.sum_weight = None;
        summary.sum_weighted_label = None;
        summary.sum_squared_weight = None;
        summary.p_hat = None;
        summary.effective_sample_size = None;
        summary.beta_alpha = None;
        summary.beta_beta = None;
        summary.lower_bound = None;
        summary.points_gate_passed = Some(false);
        summary.roots_gate_passed = None;
        summary.coverage_gate_passed = None;
        summary.weight_gate_passed = None;
        summary.effective_samples_gate_passed = None;
        summary.beta_quantile_gate_passed = None;
        summary.lower_bound_gate_passed = None;
        summary.terminal_reason = DecisionCandidateReasonV1::SparsePoints;
        let mut neighbors = audit.neighbors.clone();
        neighbors[0].age_seconds = None;
        neighbors[0].similarity_weight = None;
        neighbors[0].time_weight = None;
        neighbors[0].final_weight = None;
        assert!(replay_summary(&summary, &neighbors, 1).is_ok());

        neighbors[0].age_seconds = Some(AuditF64V1::new(0.0).unwrap());
        neighbors[0].similarity_weight = Some(AuditF64V1::new(1.0).unwrap());
        neighbors[0].time_weight = Some(AuditF64V1::new(1.0).unwrap());
        neighbors[0].final_weight = Some(AuditF64V1::new(1.0).unwrap());
        assert_eq!(
            replay_summary(&summary, &neighbors, 1),
            Err(DecisionAuditError::InvalidState)
        );
    }

    #[test]
    fn float_hash_encoding_is_fixed_width_hex_and_preserves_signed_zero() {
        let high_bit = AuditF64V1::new(-1.5).unwrap();
        let negative_zero = AuditF64V1::new(-0.0).unwrap();
        let high_bit_f32 = AuditF32V1::new(-1.5).unwrap();
        let negative_zero_f32 = AuditF32V1::new(-0.0).unwrap();
        assert_eq!(f64_bits(high_bit), "bff8000000000000");
        assert_eq!(f64_bits(negative_zero), "8000000000000000");
        assert_eq!(f32_bits(high_bit_f32), "bfc00000");
        assert_eq!(f32_bits(negative_zero_f32), "80000000");
        assert_ne!(
            f64_bits(negative_zero),
            f64_bits(AuditF64V1::new(0.0).unwrap())
        );
        let bytes = canonical_json_bytes(&json!({
            "high": f64_bits(high_bit),
            "zero": f64_bits(negative_zero),
        }))
        .unwrap();
        let encoded = std::str::from_utf8(&bytes).unwrap();
        assert!(encoded.contains("\"bff8000000000000\""));
        assert!(encoded.contains("\"8000000000000000\""));
        assert_eq!(
            AuditF64V1::from_stored(0.0, negative_zero.bits()),
            Err(DecisionAuditError::FloatBitsMismatch)
        );
        assert_eq!(
            AuditF64V1::new(f64::INFINITY),
            Err(DecisionAuditError::InvalidFloat)
        );
    }

    #[test]
    fn prepared_query_requires_exact_canonical_schema_and_bytes() {
        let prepared = prepared_query();
        let hash = prepared.canonical_query_hash().to_string();
        let pretty: Json = serde_json::from_str(prepared.canonical_query_json()).unwrap();
        assert!(matches!(
            PreparedCanonicalQueryV1::new(hash, serde_json::to_string_pretty(&pretty).unwrap()),
            Err(DecisionAuditError::InvalidCanonicalDocument)
        ));

        let mut wrong_schema = pretty;
        wrong_schema["schema"] = Json::String("wrong".to_string());
        let bytes = canonical_json_bytes(&wrong_schema).unwrap();
        assert!(matches!(
            PreparedCanonicalQueryV1::new(sha256_hex(&bytes), String::from_utf8(bytes).unwrap()),
            Err(DecisionAuditError::InvalidCanonicalDocument)
        ));
    }
}
