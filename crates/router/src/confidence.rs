// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure version-1 density, confidence, and cheapest-first selection.

#![allow(dead_code)] // Tasks 7 and 8 wire the pure engine into the Router runtime.

use std::collections::{BTreeMap, BTreeSet};
use std::panic::{AssertUnwindSafe, catch_unwind};

use statrs::distribution::{Beta, ContinuousCDF};
use uuid::{Uuid, Variant};

const TOP_K_MAX: usize = 4_095;
const CANDIDATE_COUNT_MAX: usize = 64;
const CANDIDATE_ID_MAX_BYTES: usize = 128;
const FUTURE_SKEW_MAX_MILLIS: i64 = 300_000;
const BETA_SHAPE_CERTIFICATION_MAX_V1: f64 = 1_048_576.0;

/// Stable validation failures for policy and evidence supplied to the pure engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfidenceInputError {
    InvalidPolicy,
    InvalidCandidateCount,
    CandidateProductExceeded,
    InvalidAsOf,
    InvalidCandidate,
    DuplicateCandidate,
    DuplicateCostRank,
    TooManyNeighbors,
    InvalidNeighbor,
    DuplicateEvidence,
    UnexpectedCandidateOrder,
    UnexpectedEvidenceState,
    IncompleteCandidateSet,
}

/// Stable failure from the contained third-party Beta quantile boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BetaQuantileErrorV1 {
    InvalidOrUncertified,
}

/// Complete validated statistical policy used without depending on config types.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConfidencePolicyV1 {
    top_k: usize,
    radius: f64,
    min_points: usize,
    min_independent_roots: usize,
    min_effective_samples: f64,
    min_coverage: f64,
    time_decay_half_life_seconds: f64,
    prior_success: f64,
    prior_failure: f64,
    familywise_credible_level: f64,
    promotion_lower_bound: f64,
    judge_confidence_floor: f64,
}

impl ConfidencePolicyV1 {
    /// Validate every version-1 field before any evidence can reach the math.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        top_k: usize,
        radius: f64,
        min_points: usize,
        min_independent_roots: usize,
        min_effective_samples: f64,
        min_coverage: f64,
        time_decay_half_life_seconds: f64,
        prior_success: f64,
        prior_failure: f64,
        familywise_credible_level: f64,
        promotion_lower_bound: f64,
        judge_confidence_floor: f64,
    ) -> Result<Self, ConfidenceInputError> {
        let valid = (1..=TOP_K_MAX).contains(&top_k)
            && radius.is_finite()
            && (0.0..=2.0).contains(&radius)
            && radius > 0.0
            && (1..=top_k).contains(&min_points)
            && (1..=top_k).contains(&min_independent_roots)
            && min_effective_samples.is_finite()
            && min_effective_samples > 0.0
            && min_effective_samples <= top_k as f64
            && finite_unit_interval(min_coverage)
            && time_decay_half_life_seconds.is_finite()
            && time_decay_half_life_seconds > 0.0
            && prior_success.is_finite()
            && prior_success > 0.0
            && prior_failure.is_finite()
            && prior_failure > 0.0
            && familywise_credible_level.is_finite()
            && familywise_credible_level > 0.5
            && familywise_credible_level < 1.0
            && finite_unit_interval(promotion_lower_bound)
            && finite_unit_interval(judge_confidence_floor);
        if !valid {
            return Err(ConfidenceInputError::InvalidPolicy);
        }
        Ok(Self {
            top_k,
            radius,
            min_points,
            min_independent_roots,
            min_effective_samples,
            min_coverage,
            time_decay_half_life_seconds,
            prior_success,
            prior_failure,
            familywise_credible_level,
            promotion_lower_bound,
            judge_confidence_floor,
        })
    }

    pub(crate) const fn top_k(&self) -> usize {
        self.top_k
    }

    pub(crate) const fn radius(&self) -> f64 {
        self.radius
    }

    pub(crate) const fn min_points(&self) -> usize {
        self.min_points
    }

    pub(crate) const fn min_independent_roots(&self) -> usize {
        self.min_independent_roots
    }

    pub(crate) const fn min_effective_samples(&self) -> f64 {
        self.min_effective_samples
    }

    pub(crate) const fn min_coverage(&self) -> f64 {
        self.min_coverage
    }

    pub(crate) const fn time_decay_half_life_seconds(&self) -> f64 {
        self.time_decay_half_life_seconds
    }

    pub(crate) const fn prior_success(&self) -> f64 {
        self.prior_success
    }

    pub(crate) const fn prior_failure(&self) -> f64 {
        self.prior_failure
    }

    pub(crate) const fn familywise_credible_level(&self) -> f64 {
        self.familywise_credible_level
    }

    pub(crate) const fn promotion_lower_bound(&self) -> f64 {
        self.promotion_lower_bound
    }
}

fn finite_unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

/// Terminal evidence classes all retained in the raw coverage denominator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfidenceTerminalClassV1 {
    Completed,
    DeterministicFailure,
    OperationalFailure,
    SkippedCooloff,
    CanceledShutdown,
    OrphanedBeforeSchedule,
    OrphanedInFlight,
}

/// Origin of an optional final quality evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfidenceEvaluationSourceV1 {
    DeterministicValidator,
    Judge,
}

/// Binary labels that may enter the confidence calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfidenceBinaryLabelV1 {
    Pass,
    Fail,
}

impl ConfidenceBinaryLabelV1 {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Pass => "pass",
            Self::Fail => "fail",
        }
    }
}

/// Validated evaluation facts projected with one neighbor.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConfidenceEvaluationInputV1 {
    evaluation_id: Uuid,
    source: ConfidenceEvaluationSourceV1,
    binary_label: Option<ConfidenceBinaryLabelV1>,
    judge_confidence: Option<f64>,
    promotion_eligible: bool,
    created_at_unix_ms: i64,
}

impl ConfidenceEvaluationInputV1 {
    /// Build an evaluation from authoritative confidence bits.
    pub(crate) fn new(
        evaluation_id: Uuid,
        source: ConfidenceEvaluationSourceV1,
        binary_label: Option<ConfidenceBinaryLabelV1>,
        judge_confidence_bits: Option<u64>,
        promotion_eligible: bool,
        created_at_unix_ms: i64,
    ) -> Result<Self, ConfidenceInputError> {
        let judge_confidence = judge_confidence_bits.map(f64::from_bits);
        if !is_uuid_v7(evaluation_id)
            || created_at_unix_ms < 0
            || judge_confidence.is_some_and(|value| !finite_unit_interval(value))
            || (promotion_eligible && binary_label.is_none())
        {
            return Err(ConfidenceInputError::InvalidNeighbor);
        }
        let coherent = match source {
            ConfidenceEvaluationSourceV1::Judge => judge_confidence.is_some(),
            ConfidenceEvaluationSourceV1::DeterministicValidator => {
                judge_confidence.is_none() && binary_label == Some(ConfidenceBinaryLabelV1::Fail)
            }
        };
        if !coherent {
            return Err(ConfidenceInputError::InvalidNeighbor);
        }
        Ok(Self {
            evaluation_id,
            source,
            binary_label,
            judge_confidence,
            promotion_eligible,
            created_at_unix_ms,
        })
    }
}

/// Complete repository-independent input for one top-K evidence row.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConfidenceNeighborInputV1 {
    evidence_vector_link_id: Uuid,
    shadow_attempt_id: Uuid,
    anchor_id: Uuid,
    root_uuid: Uuid,
    terminal_class: ConfidenceTerminalClassV1,
    distance_bits: u32,
    evaluation: Option<ConfidenceEvaluationInputV1>,
}

impl ConfidenceNeighborInputV1 {
    /// Validate identity, terminal/evaluation coherence, and authoritative f32 distance bits.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        evidence_vector_link_id: Uuid,
        shadow_attempt_id: Uuid,
        anchor_id: Uuid,
        root_uuid: Uuid,
        terminal_class: ConfidenceTerminalClassV1,
        distance_bits: u32,
        evaluation: Option<ConfidenceEvaluationInputV1>,
    ) -> Result<Self, ConfidenceInputError> {
        let distance = f32::from_bits(distance_bits);
        let identities_valid = [
            evidence_vector_link_id,
            shadow_attempt_id,
            anchor_id,
            root_uuid,
        ]
        .into_iter()
        .all(is_uuid_v7);
        let terminal_coherent = evaluation.as_ref().is_none_or(|evaluation| {
            matches!(
                (terminal_class, evaluation.source),
                (
                    ConfidenceTerminalClassV1::Completed,
                    ConfidenceEvaluationSourceV1::Judge
                ) | (
                    ConfidenceTerminalClassV1::DeterministicFailure,
                    ConfidenceEvaluationSourceV1::DeterministicValidator
                )
            )
        });
        if !identities_valid
            || !distance.is_finite()
            || !(0.0..=2.0).contains(&distance)
            || !terminal_coherent
        {
            return Err(ConfidenceInputError::InvalidNeighbor);
        }
        Ok(Self {
            evidence_vector_link_id,
            shadow_attempt_id,
            anchor_id,
            root_uuid,
            terminal_class,
            distance_bits,
            evaluation,
        })
    }

    fn distance(&self) -> f32 {
        f32::from_bits(self.distance_bits)
    }
}

/// Exact partition outcome supplied for one capability-eligible candidate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum CandidateEvidenceV1 {
    NoPartition,
    Neighbors(Vec<ConfidenceNeighborInputV1>),
}

/// One candidate and its exact top-K evidence, independent of config and storage types.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CandidateConfidenceInputV1 {
    candidate_id: String,
    cost_rank: u32,
    evidence: CandidateEvidenceV1,
}

impl CandidateConfidenceInputV1 {
    pub(crate) fn new(
        candidate_id: impl Into<String>,
        cost_rank: u32,
        evidence: CandidateEvidenceV1,
    ) -> Result<Self, ConfidenceInputError> {
        let candidate_id = candidate_id.into();
        validate_candidate_id(&candidate_id)?;
        Ok(Self {
            candidate_id,
            cost_rank,
            evidence,
        })
    }
}

fn validate_candidate_id(candidate_id: &str) -> Result<(), ConfidenceInputError> {
    if candidate_id.is_empty()
        || candidate_id.len() > CANDIDATE_ID_MAX_BYTES
        || candidate_id.chars().any(char::is_control)
    {
        return Err(ConfidenceInputError::InvalidCandidate);
    }
    Ok(())
}

/// Why one retained top-K row did or did not enter the root-collapsed math.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NeighborExclusionReasonV1 {
    OutsideRadius,
    IneligibleQuality,
    DuplicateRoot,
    Included,
}

impl NeighborExclusionReasonV1 {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::OutsideRadius => "outside_radius",
            Self::IneligibleQuality => "ineligible_quality",
            Self::DuplicateRoot => "duplicate_root",
            Self::Included => "included",
        }
    }
}

/// One audit-ready neighbor without duplicating the raw independent-root UUID.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct AuditedNeighborV1 {
    pub(crate) candidate_ordinal: usize,
    pub(crate) evidence_vector_link_id: Uuid,
    pub(crate) shadow_attempt_id: Uuid,
    pub(crate) anchor_id: Uuid,
    pub(crate) evaluation_id: Option<Uuid>,
    pub(crate) distance: f32,
    pub(crate) distance_bits: u32,
    pub(crate) age_millis: Option<u64>,
    pub(crate) similarity_weight: Option<f64>,
    pub(crate) time_weight: Option<f64>,
    pub(crate) final_weight: Option<f64>,
    pub(crate) binary_label: Option<ConfidenceBinaryLabelV1>,
    pub(crate) selected_root: bool,
    pub(crate) root_group_ordinal: usize,
    pub(crate) exclusion_reason: NeighborExclusionReasonV1,
}

/// Progressive gate state retained in a candidate summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfidenceGateResultV1 {
    NotEvaluated,
    Passed,
    Failed,
}

/// Exact version-1 gate precedence for one candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CandidateGateResultsV1 {
    pub(crate) partition: ConfidenceGateResultV1,
    pub(crate) points: ConfidenceGateResultV1,
    pub(crate) roots: ConfidenceGateResultV1,
    pub(crate) coverage: ConfidenceGateResultV1,
    pub(crate) weight_math: ConfidenceGateResultV1,
    pub(crate) effective_samples: ConfidenceGateResultV1,
    pub(crate) beta_quantile: ConfidenceGateResultV1,
    pub(crate) lower_bound: ConfidenceGateResultV1,
}

impl CandidateGateResultsV1 {
    fn pending() -> Self {
        Self {
            partition: ConfidenceGateResultV1::NotEvaluated,
            points: ConfidenceGateResultV1::NotEvaluated,
            roots: ConfidenceGateResultV1::NotEvaluated,
            coverage: ConfidenceGateResultV1::NotEvaluated,
            weight_math: ConfidenceGateResultV1::NotEvaluated,
            effective_samples: ConfidenceGateResultV1::NotEvaluated,
            beta_quantile: ConfidenceGateResultV1::NotEvaluated,
            lower_bound: ConfidenceGateResultV1::NotEvaluated,
        }
    }
}

/// Stable terminal reason for one candidate summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CandidateConfidenceReasonV1 {
    NoPartition,
    SparsePoints,
    InsufficientRoots,
    LowCoverage,
    InvalidEvidenceTime,
    NumericError,
    InsufficientEffectiveSamples,
    LowerBoundBelowThreshold,
    Passed,
    NotEvaluatedAfterWinner,
    NotEvaluatedAfterFallback,
}

impl CandidateConfidenceReasonV1 {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::NoPartition => "no_partition",
            Self::SparsePoints => "sparse_points",
            Self::InsufficientRoots => "insufficient_roots",
            Self::LowCoverage => "low_coverage",
            Self::InvalidEvidenceTime => "invalid_evidence_time",
            Self::NumericError => "numeric_error",
            Self::InsufficientEffectiveSamples => "insufficient_effective_samples",
            Self::LowerBoundBelowThreshold => "lower_bound_below_threshold",
            Self::Passed => "passed",
            Self::NotEvaluatedAfterWinner => "not_evaluated_after_winner",
            Self::NotEvaluatedAfterFallback => "not_evaluated_after_fallback",
        }
    }
}

/// Counts and deterministic statistics for one candidate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CandidateConfidenceSummaryV1 {
    pub(crate) candidate_id: String,
    pub(crate) cost_rank: u32,
    pub(crate) top_k_points: usize,
    pub(crate) raw_points: usize,
    pub(crate) labeled_points: usize,
    pub(crate) attempted_roots: usize,
    pub(crate) labeled_roots: usize,
    pub(crate) coverage: Option<f64>,
    pub(crate) sum_weight: Option<f64>,
    pub(crate) sum_weighted_pass: Option<f64>,
    pub(crate) sum_weight_squared: Option<f64>,
    pub(crate) p_hat: Option<f64>,
    pub(crate) n_eff: Option<f64>,
    pub(crate) beta_alpha: Option<f64>,
    pub(crate) beta_beta: Option<f64>,
    pub(crate) candidate_alpha: Option<f64>,
    pub(crate) lower_bound: Option<f64>,
    pub(crate) gates: CandidateGateResultsV1,
    pub(crate) reason: CandidateConfidenceReasonV1,
    pub(crate) neighbors: Vec<AuditedNeighborV1>,
}

impl CandidateConfidenceSummaryV1 {
    fn unevaluated(
        candidate_id: String,
        cost_rank: u32,
        reason: CandidateConfidenceReasonV1,
    ) -> Self {
        Self {
            candidate_id,
            cost_rank,
            top_k_points: 0,
            raw_points: 0,
            labeled_points: 0,
            attempted_roots: 0,
            labeled_roots: 0,
            coverage: None,
            sum_weight: None,
            sum_weighted_pass: None,
            sum_weight_squared: None,
            p_hat: None,
            n_eff: None,
            beta_alpha: None,
            beta_beta: None,
            candidate_alpha: None,
            lower_bound: None,
            gates: CandidateGateResultsV1::pending(),
            reason,
            neighbors: Vec::new(),
        }
    }
}

/// Decision-level result of the pure cheapest-first engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfidenceDecisionReasonV1 {
    EmbeddingUnavailable,
    VectorUnhealthy,
    VersionMismatch,
    RecommendObserveOnly,
    NoCandidatePassed,
    InvalidEvidenceTime,
    NumericError,
}

/// Complete pure result, including one summary for every exact live candidate.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ConfidenceDecisionV1 {
    pub(crate) recommended_candidate_id: Option<String>,
    pub(crate) reason: ConfidenceDecisionReasonV1,
    pub(crate) summaries: Vec<CandidateConfidenceSummaryV1>,
}

/// Incremental pure engine used to avoid searching candidates after a stop condition.
pub(crate) struct ConfidenceEngineV1<'a> {
    policy: &'a ConfidencePolicyV1,
    exact_candidate_count: usize,
    as_of_unix_ms: i64,
    candidate_alpha: f64,
    candidate_ids: BTreeSet<String>,
    cost_ranks: BTreeSet<u32>,
    evidence_ids: BTreeSet<Uuid>,
    last_candidate: Option<(u32, String)>,
    root_ordinals: BTreeMap<Uuid, usize>,
    next_root_ordinal: usize,
    winner: Option<String>,
    fallback: Option<ConfidenceDecisionReasonV1>,
    summaries: Vec<CandidateConfidenceSummaryV1>,
}

impl<'a> ConfidenceEngineV1<'a> {
    pub(crate) fn new(
        policy: &'a ConfidencePolicyV1,
        exact_candidate_count: usize,
        as_of_unix_ms: i64,
    ) -> Result<Self, ConfidenceInputError> {
        if !(1..=CANDIDATE_COUNT_MAX).contains(&exact_candidate_count) {
            return Err(ConfidenceInputError::InvalidCandidateCount);
        }
        if exact_candidate_count
            .checked_mul(policy.top_k)
            .is_none_or(|rows| rows > TOP_K_MAX)
        {
            return Err(ConfidenceInputError::CandidateProductExceeded);
        }
        if as_of_unix_ms < 0 {
            return Err(ConfidenceInputError::InvalidAsOf);
        }
        let candidate_alpha =
            (1.0 - policy.familywise_credible_level) / exact_candidate_count as f64;
        if !candidate_alpha.is_finite() || candidate_alpha <= 0.0 || candidate_alpha >= 1.0 {
            return Err(ConfidenceInputError::InvalidPolicy);
        }
        Ok(Self {
            policy,
            exact_candidate_count,
            as_of_unix_ms,
            candidate_alpha,
            candidate_ids: BTreeSet::new(),
            cost_ranks: BTreeSet::new(),
            evidence_ids: BTreeSet::new(),
            last_candidate: None,
            root_ordinals: BTreeMap::new(),
            next_root_ordinal: 0,
            winner: None,
            fallback: None,
            summaries: Vec::with_capacity(exact_candidate_count),
        })
    }

    /// Whether the next candidate needs an exact partition search.
    pub(crate) fn needs_evidence(&self) -> bool {
        self.winner.is_none() && self.fallback.is_none()
    }

    /// Stop candidate evidence reads after an embedding, vector, or version fallback.
    pub(crate) fn stop_with_external_fallback(
        &mut self,
        reason: ConfidenceDecisionReasonV1,
    ) -> Result<(), ConfidenceInputError> {
        if !self.needs_evidence()
            || !matches!(
                reason,
                ConfidenceDecisionReasonV1::EmbeddingUnavailable
                    | ConfidenceDecisionReasonV1::VectorUnhealthy
                    | ConfidenceDecisionReasonV1::VersionMismatch
            )
        {
            return Err(ConfidenceInputError::UnexpectedEvidenceState);
        }
        self.fallback = Some(reason);
        Ok(())
    }

    /// Evaluate the next sorted candidate while the engine remains open.
    pub(crate) fn evaluate_candidate(
        &mut self,
        candidate: CandidateConfidenceInputV1,
    ) -> Result<CandidateConfidenceReasonV1, ConfidenceInputError> {
        self.evaluate_candidate_with_lower_bound(candidate, self.policy.promotion_lower_bound)
    }

    /// Evaluate with a caller-selected promotion or retention threshold.
    pub(crate) fn evaluate_candidate_with_lower_bound(
        &mut self,
        candidate: CandidateConfidenceInputV1,
        lower_bound_threshold: f64,
    ) -> Result<CandidateConfidenceReasonV1, ConfidenceInputError> {
        self.evaluate_candidate_with_gate(candidate, lower_bound_threshold, true)
    }

    /// Evaluate fresh evidence while allowing external Active authority to veto a winner.
    pub(crate) fn evaluate_candidate_with_gate(
        &mut self,
        candidate: CandidateConfidenceInputV1,
        lower_bound_threshold: f64,
        candidate_authorized: bool,
    ) -> Result<CandidateConfidenceReasonV1, ConfidenceInputError> {
        if !self.needs_evidence() {
            return Err(ConfidenceInputError::UnexpectedEvidenceState);
        }
        if !lower_bound_threshold.is_finite()
            || !(0.0..=self.policy.promotion_lower_bound).contains(&lower_bound_threshold)
        {
            return Err(ConfidenceInputError::InvalidPolicy);
        }
        self.accept_candidate(&candidate.candidate_id, candidate.cost_rank)?;
        self.accept_evidence(&candidate.evidence)?;
        let summary = evaluate_candidate(
            self.policy,
            lower_bound_threshold,
            self.candidate_alpha,
            self.as_of_unix_ms,
            candidate,
            &mut self.root_ordinals,
            &mut self.next_root_ordinal,
        );
        match summary.reason {
            CandidateConfidenceReasonV1::Passed if candidate_authorized => {
                self.winner = Some(summary.candidate_id.clone());
            }
            CandidateConfidenceReasonV1::InvalidEvidenceTime => {
                self.fallback = Some(ConfidenceDecisionReasonV1::InvalidEvidenceTime);
            }
            CandidateConfidenceReasonV1::NumericError => {
                self.fallback = Some(ConfidenceDecisionReasonV1::NumericError);
            }
            _ => {}
        }
        let reason = summary.reason;
        self.summaries.push(summary);
        Ok(reason)
    }

    /// Record one later candidate without accepting or inspecting evidence.
    pub(crate) fn record_unevaluated(
        &mut self,
        candidate_id: impl Into<String>,
        cost_rank: u32,
    ) -> Result<(), ConfidenceInputError> {
        if self.needs_evidence() {
            return Err(ConfidenceInputError::UnexpectedEvidenceState);
        }
        let candidate_id = candidate_id.into();
        validate_candidate_id(&candidate_id)?;
        self.accept_candidate(&candidate_id, cost_rank)?;
        let reason = if self.winner.is_some() {
            CandidateConfidenceReasonV1::NotEvaluatedAfterWinner
        } else {
            CandidateConfidenceReasonV1::NotEvaluatedAfterFallback
        };
        self.summaries
            .push(CandidateConfidenceSummaryV1::unevaluated(
                candidate_id,
                cost_rank,
                reason,
            ));
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<ConfidenceDecisionV1, ConfidenceInputError> {
        if self.summaries.len() != self.exact_candidate_count {
            return Err(ConfidenceInputError::IncompleteCandidateSet);
        }
        let reason = if self.winner.is_some() {
            ConfidenceDecisionReasonV1::RecommendObserveOnly
        } else {
            self.fallback
                .unwrap_or(ConfidenceDecisionReasonV1::NoCandidatePassed)
        };
        Ok(ConfidenceDecisionV1 {
            recommended_candidate_id: self.winner,
            reason,
            summaries: self.summaries,
        })
    }

    fn accept_candidate(
        &mut self,
        candidate_id: &str,
        cost_rank: u32,
    ) -> Result<(), ConfidenceInputError> {
        if self.summaries.len() >= self.exact_candidate_count {
            return Err(ConfidenceInputError::InvalidCandidateCount);
        }
        if self
            .last_candidate
            .as_ref()
            .is_some_and(|last| (cost_rank, candidate_id) <= (last.0, last.1.as_str()))
        {
            return Err(ConfidenceInputError::UnexpectedCandidateOrder);
        }
        if !self.candidate_ids.insert(candidate_id.to_string()) {
            return Err(ConfidenceInputError::DuplicateCandidate);
        }
        if !self.cost_ranks.insert(cost_rank) {
            return Err(ConfidenceInputError::DuplicateCostRank);
        }
        self.last_candidate = Some((cost_rank, candidate_id.to_string()));
        Ok(())
    }

    fn accept_evidence(
        &mut self,
        evidence: &CandidateEvidenceV1,
    ) -> Result<(), ConfidenceInputError> {
        let CandidateEvidenceV1::Neighbors(neighbors) = evidence else {
            return Ok(());
        };
        if neighbors.len() > self.policy.top_k {
            return Err(ConfidenceInputError::TooManyNeighbors);
        }
        for neighbor in neighbors {
            if !self.evidence_ids.insert(neighbor.evidence_vector_link_id) {
                return Err(ConfidenceInputError::DuplicateEvidence);
            }
        }
        Ok(())
    }
}

/// Evaluate exact candidate facts in normative cheapest-first order.
pub(crate) fn evaluate_confidence_v1(
    policy: &ConfidencePolicyV1,
    exact_candidate_count: usize,
    as_of_unix_ms: i64,
    mut candidates: Vec<CandidateConfidenceInputV1>,
) -> Result<ConfidenceDecisionV1, ConfidenceInputError> {
    if candidates.len() != exact_candidate_count {
        return Err(ConfidenceInputError::InvalidCandidateCount);
    }
    validate_candidate_set(policy, &candidates)?;
    candidates.sort_by(|left, right| {
        left.cost_rank
            .cmp(&right.cost_rank)
            .then_with(|| left.candidate_id.cmp(&right.candidate_id))
    });

    let mut engine = ConfidenceEngineV1::new(policy, exact_candidate_count, as_of_unix_ms)?;
    for candidate in candidates {
        if engine.needs_evidence() {
            engine.evaluate_candidate(candidate)?;
        } else {
            engine.record_unevaluated(candidate.candidate_id, candidate.cost_rank)?;
        }
    }
    engine.finish()
}

/// Evaluate one exact candidate while preserving the configured familywise count.
pub(crate) fn evaluate_single_candidate_v1(
    policy: &ConfidencePolicyV1,
    exact_candidate_count: usize,
    as_of_unix_ms: i64,
    candidate: CandidateConfidenceInputV1,
) -> Result<CandidateConfidenceSummaryV1, ConfidenceInputError> {
    let mut engine = ConfidenceEngineV1::new(policy, exact_candidate_count, as_of_unix_ms)?;
    engine.evaluate_candidate(candidate)?;
    engine
        .summaries
        .pop()
        .ok_or(ConfidenceInputError::IncompleteCandidateSet)
}

fn validate_candidate_set(
    policy: &ConfidencePolicyV1,
    candidates: &[CandidateConfidenceInputV1],
) -> Result<(), ConfidenceInputError> {
    let mut candidate_ids = BTreeSet::new();
    let mut cost_ranks = BTreeSet::new();
    let mut evidence_ids = BTreeSet::new();
    for candidate in candidates {
        if !candidate_ids.insert(candidate.candidate_id.as_str()) {
            return Err(ConfidenceInputError::DuplicateCandidate);
        }
        if !cost_ranks.insert(candidate.cost_rank) {
            return Err(ConfidenceInputError::DuplicateCostRank);
        }
        let CandidateEvidenceV1::Neighbors(neighbors) = &candidate.evidence else {
            continue;
        };
        if neighbors.len() > policy.top_k {
            return Err(ConfidenceInputError::TooManyNeighbors);
        }
        for neighbor in neighbors {
            if !evidence_ids.insert(neighbor.evidence_vector_link_id) {
                return Err(ConfidenceInputError::DuplicateEvidence);
            }
        }
    }
    Ok(())
}

fn evaluate_candidate(
    policy: &ConfidencePolicyV1,
    lower_bound_threshold: f64,
    candidate_alpha: f64,
    as_of_unix_ms: i64,
    candidate: CandidateConfidenceInputV1,
    root_ordinals: &mut BTreeMap<Uuid, usize>,
    next_root_ordinal: &mut usize,
) -> CandidateConfidenceSummaryV1 {
    let CandidateConfidenceInputV1 {
        candidate_id,
        cost_rank,
        evidence,
    } = candidate;
    let CandidateEvidenceV1::Neighbors(mut inputs) = evidence else {
        let mut summary = CandidateConfidenceSummaryV1::unevaluated(
            candidate_id,
            cost_rank,
            CandidateConfidenceReasonV1::NoPartition,
        );
        summary.gates.partition = ConfidenceGateResultV1::Failed;
        return summary;
    };

    inputs.sort_by(|left, right| {
        compare_distance(left.distance(), right.distance()).then_with(|| {
            left.evidence_vector_link_id
                .cmp(&right.evidence_vector_link_id)
        })
    });

    let mut working = Vec::with_capacity(inputs.len());
    for (candidate_ordinal, input) in inputs.into_iter().enumerate() {
        let root_group_ordinal = *root_ordinals.entry(input.root_uuid).or_insert_with(|| {
            let assigned = *next_root_ordinal;
            *next_root_ordinal = next_root_ordinal.saturating_add(1);
            assigned
        });
        working.push(WorkingNeighbor {
            candidate_ordinal,
            root_group_ordinal,
            input,
            exclusion_reason: NeighborExclusionReasonV1::OutsideRadius,
            selected_root: false,
            computed: None,
        });
    }

    let raw_indices = working
        .iter()
        .enumerate()
        .filter_map(|(index, neighbor)| {
            (f64::from(neighbor.input.distance()) <= policy.radius).then_some(index)
        })
        .collect::<Vec<_>>();
    let attempted_roots = raw_indices
        .iter()
        .map(|index| working[*index].input.root_uuid)
        .collect::<BTreeSet<_>>();
    let labeled_indices = raw_indices
        .iter()
        .copied()
        .filter(|index| eligible_evaluation(&working[*index].input, policy))
        .collect::<Vec<_>>();
    let labeled_roots = labeled_indices
        .iter()
        .map(|index| working[*index].input.root_uuid)
        .collect::<BTreeSet<_>>();
    let coverage = if attempted_roots.is_empty() {
        0.0
    } else {
        labeled_roots.len() as f64 / attempted_roots.len() as f64
    };

    let mut selected_by_root = BTreeMap::<Uuid, usize>::new();
    for index in &labeled_indices {
        let root_uuid = working[*index].input.root_uuid;
        match selected_by_root.get(&root_uuid).copied() {
            Some(current) if !preferred_root_observation(&working[*index], &working[current]) => {}
            _ => {
                selected_by_root.insert(root_uuid, *index);
            }
        }
    }
    let selected_indices = selected_by_root.values().copied().collect::<BTreeSet<_>>();
    for (index, neighbor) in working.iter_mut().enumerate() {
        let inside_radius = f64::from(neighbor.input.distance()) <= policy.radius;
        neighbor.exclusion_reason = if !inside_radius {
            NeighborExclusionReasonV1::OutsideRadius
        } else if !eligible_evaluation(&neighbor.input, policy) {
            NeighborExclusionReasonV1::IneligibleQuality
        } else if selected_indices.contains(&index) {
            neighbor.selected_root = true;
            NeighborExclusionReasonV1::Included
        } else {
            NeighborExclusionReasonV1::DuplicateRoot
        };
    }

    let mut gates = CandidateGateResultsV1::pending();
    gates.partition = ConfidenceGateResultV1::Passed;
    let mut summary = CandidateConfidenceSummaryV1 {
        candidate_id,
        cost_rank,
        top_k_points: working.len(),
        raw_points: raw_indices.len(),
        labeled_points: labeled_indices.len(),
        attempted_roots: attempted_roots.len(),
        labeled_roots: labeled_roots.len(),
        coverage: Some(coverage),
        sum_weight: None,
        sum_weighted_pass: None,
        sum_weight_squared: None,
        p_hat: None,
        n_eff: None,
        beta_alpha: None,
        beta_beta: None,
        candidate_alpha: Some(candidate_alpha),
        lower_bound: None,
        gates,
        reason: CandidateConfidenceReasonV1::SparsePoints,
        neighbors: Vec::new(),
    };

    if labeled_indices.len() < policy.min_points {
        summary.gates.points = ConfidenceGateResultV1::Failed;
        summary.neighbors = finish_neighbors(working);
        return summary;
    }
    summary.gates.points = ConfidenceGateResultV1::Passed;
    if labeled_roots.len() < policy.min_independent_roots {
        summary.gates.roots = ConfidenceGateResultV1::Failed;
        summary.reason = CandidateConfidenceReasonV1::InsufficientRoots;
        summary.neighbors = finish_neighbors(working);
        return summary;
    }
    summary.gates.roots = ConfidenceGateResultV1::Passed;
    if coverage < policy.min_coverage {
        summary.gates.coverage = ConfidenceGateResultV1::Failed;
        summary.reason = CandidateConfidenceReasonV1::LowCoverage;
        summary.neighbors = finish_neighbors(working);
        return summary;
    }
    summary.gates.coverage = ConfidenceGateResultV1::Passed;

    let mut selected_numeric_order = selected_indices.into_iter().collect::<Vec<_>>();
    selected_numeric_order.sort_by_key(|index| {
        working[*index]
            .input
            .evaluation
            .as_ref()
            .expect("selected observations have evaluations")
            .evaluation_id
    });
    let computed = match compute_weights(policy, as_of_unix_ms, &working, &selected_numeric_order) {
        Ok(computed) => computed,
        Err(NumericFailure::InvalidEvidenceTime) => {
            summary.gates.weight_math = ConfidenceGateResultV1::Failed;
            summary.reason = CandidateConfidenceReasonV1::InvalidEvidenceTime;
            summary.neighbors = finish_neighbors(working);
            return summary;
        }
        Err(NumericFailure::Numeric) => {
            summary.gates.weight_math = ConfidenceGateResultV1::Failed;
            summary.reason = CandidateConfidenceReasonV1::NumericError;
            summary.neighbors = finish_neighbors(working);
            return summary;
        }
    };
    for item in &computed.items {
        working[item.index].computed = Some(item.values);
    }
    summary.gates.weight_math = ConfidenceGateResultV1::Passed;
    summary.sum_weight = Some(computed.sum_weight);
    summary.sum_weighted_pass = Some(computed.sum_weighted_pass);
    summary.sum_weight_squared = Some(computed.sum_weight_squared);
    summary.p_hat = Some(computed.p_hat);
    summary.n_eff = Some(computed.n_eff);
    if computed.n_eff < policy.min_effective_samples {
        summary.gates.effective_samples = ConfidenceGateResultV1::Failed;
        summary.reason = CandidateConfidenceReasonV1::InsufficientEffectiveSamples;
        summary.neighbors = finish_neighbors(working);
        return summary;
    }
    summary.gates.effective_samples = ConfidenceGateResultV1::Passed;

    let Some((beta_alpha, beta_beta)) = checked_beta_shapes(
        policy.prior_success,
        policy.prior_failure,
        computed.p_hat,
        computed.n_eff,
    ) else {
        summary.gates.beta_quantile = ConfidenceGateResultV1::Failed;
        summary.reason = CandidateConfidenceReasonV1::NumericError;
        summary.neighbors = finish_neighbors(working);
        return summary;
    };
    summary.beta_alpha = Some(beta_alpha);
    summary.beta_beta = Some(beta_beta);
    let lower_bound = match beta_inverse_cdf_v1(candidate_alpha, beta_alpha, beta_beta) {
        Ok(value) => value,
        Err(BetaQuantileErrorV1::InvalidOrUncertified) => {
            summary.gates.beta_quantile = ConfidenceGateResultV1::Failed;
            summary.reason = CandidateConfidenceReasonV1::NumericError;
            summary.neighbors = finish_neighbors(working);
            return summary;
        }
    };
    summary.gates.beta_quantile = ConfidenceGateResultV1::Passed;
    summary.lower_bound = Some(lower_bound);
    if lower_bound < lower_bound_threshold {
        summary.gates.lower_bound = ConfidenceGateResultV1::Failed;
        summary.reason = CandidateConfidenceReasonV1::LowerBoundBelowThreshold;
    } else {
        summary.gates.lower_bound = ConfidenceGateResultV1::Passed;
        summary.reason = CandidateConfidenceReasonV1::Passed;
    }
    summary.neighbors = finish_neighbors(working);
    summary
}

fn checked_beta_shapes(
    prior_success: f64,
    prior_failure: f64,
    p_hat: f64,
    n_eff: f64,
) -> Option<(f64, f64)> {
    let beta_alpha = prior_success + p_hat * n_eff;
    let beta_beta = prior_failure + (1.0 - p_hat) * n_eff;
    (beta_alpha.is_finite() && beta_alpha > 0.0 && beta_beta.is_finite() && beta_beta > 0.0)
        .then_some((beta_alpha, beta_beta))
}

fn eligible_evaluation(neighbor: &ConfidenceNeighborInputV1, policy: &ConfidencePolicyV1) -> bool {
    let Some(evaluation) = neighbor.evaluation.as_ref() else {
        return false;
    };
    if !evaluation.promotion_eligible || evaluation.binary_label.is_none() {
        return false;
    }
    match evaluation.source {
        ConfidenceEvaluationSourceV1::DeterministicValidator => {
            evaluation.binary_label == Some(ConfidenceBinaryLabelV1::Fail)
                && evaluation.judge_confidence.is_none()
        }
        ConfidenceEvaluationSourceV1::Judge => evaluation
            .judge_confidence
            .is_some_and(|confidence| confidence >= policy.judge_confidence_floor),
    }
}

fn preferred_root_observation(candidate: &WorkingNeighbor, current: &WorkingNeighbor) -> bool {
    let distance_order = compare_distance(candidate.input.distance(), current.input.distance());
    if !distance_order.is_eq() {
        return distance_order.is_lt();
    }
    let candidate_evaluation = candidate
        .input
        .evaluation
        .as_ref()
        .expect("eligible observations have evaluations");
    let current_evaluation = current
        .input
        .evaluation
        .as_ref()
        .expect("eligible observations have evaluations");
    candidate_evaluation
        .created_at_unix_ms
        .cmp(&current_evaluation.created_at_unix_ms)
        .reverse()
        .then_with(|| {
            candidate_evaluation
                .evaluation_id
                .cmp(&current_evaluation.evaluation_id)
        })
        .is_lt()
}

fn compare_distance(left: f32, right: f32) -> std::cmp::Ordering {
    if left < right {
        std::cmp::Ordering::Less
    } else if left > right {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}

#[derive(Debug, Clone)]
struct WorkingNeighbor {
    candidate_ordinal: usize,
    root_group_ordinal: usize,
    input: ConfidenceNeighborInputV1,
    exclusion_reason: NeighborExclusionReasonV1,
    selected_root: bool,
    computed: Option<ComputedNeighborValues>,
}

#[derive(Debug, Clone, Copy)]
struct ComputedNeighborValues {
    age_millis: u64,
    similarity_weight: f64,
    time_weight: f64,
    final_weight: f64,
}

fn finish_neighbors(working: Vec<WorkingNeighbor>) -> Vec<AuditedNeighborV1> {
    working
        .into_iter()
        .map(|neighbor| {
            let evaluation = neighbor.input.evaluation.as_ref();
            AuditedNeighborV1 {
                candidate_ordinal: neighbor.candidate_ordinal,
                evidence_vector_link_id: neighbor.input.evidence_vector_link_id,
                shadow_attempt_id: neighbor.input.shadow_attempt_id,
                anchor_id: neighbor.input.anchor_id,
                evaluation_id: evaluation.map(|value| value.evaluation_id),
                distance: neighbor.input.distance(),
                distance_bits: neighbor.input.distance_bits,
                age_millis: neighbor.computed.map(|value| value.age_millis),
                similarity_weight: neighbor.computed.map(|value| value.similarity_weight),
                time_weight: neighbor.computed.map(|value| value.time_weight),
                final_weight: neighbor.computed.map(|value| value.final_weight),
                binary_label: evaluation.and_then(|value| value.binary_label),
                selected_root: neighbor.selected_root,
                root_group_ordinal: neighbor.root_group_ordinal,
                exclusion_reason: neighbor.exclusion_reason,
            }
        })
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NumericFailure {
    InvalidEvidenceTime,
    Numeric,
}

struct ComputedWeightItem {
    index: usize,
    values: ComputedNeighborValues,
}

struct ComputedWeights {
    items: Vec<ComputedWeightItem>,
    sum_weight: f64,
    sum_weighted_pass: f64,
    sum_weight_squared: f64,
    p_hat: f64,
    n_eff: f64,
}

fn compute_weights(
    policy: &ConfidencePolicyV1,
    as_of_unix_ms: i64,
    working: &[WorkingNeighbor],
    selected_indices: &[usize],
) -> Result<ComputedWeights, NumericFailure> {
    let mut items = Vec::with_capacity(selected_indices.len());
    let mut sum_weight = NeumaierSum::default();
    let mut sum_weighted_pass = NeumaierSum::default();
    let mut sum_weight_squared = NeumaierSum::default();

    for index in selected_indices {
        let neighbor = &working[*index].input;
        let evaluation = neighbor
            .evaluation
            .as_ref()
            .ok_or(NumericFailure::Numeric)?;
        let age_millis = evidence_age_millis(as_of_unix_ms, evaluation.created_at_unix_ms)?;
        let age_seconds = age_millis as f64 / 1_000.0;
        let normalized_distance = f64::from(neighbor.distance()) / policy.radius;
        let similarity_weight = (1.0 - normalized_distance).max(0.0);
        let decay_exponent = -age_seconds / policy.time_decay_half_life_seconds;
        let time_weight = decay_exponent.exp2();
        let final_weight = similarity_weight * time_weight;
        if !age_seconds.is_finite()
            || !normalized_distance.is_finite()
            || !similarity_weight.is_finite()
            || !decay_exponent.is_finite()
            || !time_weight.is_finite()
            || !final_weight.is_finite()
        {
            return Err(NumericFailure::Numeric);
        }
        let y = match evaluation.binary_label {
            Some(ConfidenceBinaryLabelV1::Pass) => 1.0,
            Some(ConfidenceBinaryLabelV1::Fail) => 0.0,
            None => return Err(NumericFailure::Numeric),
        };
        let weighted_pass = final_weight * y;
        let weight_squared = final_weight * final_weight;
        sum_weight.add(final_weight)?;
        sum_weighted_pass.add(weighted_pass)?;
        sum_weight_squared.add(weight_squared)?;
        items.push(ComputedWeightItem {
            index: *index,
            values: ComputedNeighborValues {
                age_millis,
                similarity_weight,
                time_weight,
                final_weight,
            },
        });
    }

    let sum_weight = sum_weight.total()?;
    let sum_weighted_pass = sum_weighted_pass.total()?;
    let sum_weight_squared = sum_weight_squared.total()?;
    if sum_weight <= 0.0 || sum_weight_squared <= 0.0 {
        return Err(NumericFailure::Numeric);
    }
    let p_hat = sum_weighted_pass / sum_weight;
    let weight_sum_squared = sum_weight * sum_weight;
    let n_eff = weight_sum_squared / sum_weight_squared;
    if !p_hat.is_finite()
        || !(0.0..=1.0).contains(&p_hat)
        || !weight_sum_squared.is_finite()
        || !n_eff.is_finite()
        || n_eff <= 0.0
    {
        return Err(NumericFailure::Numeric);
    }
    Ok(ComputedWeights {
        items,
        sum_weight,
        sum_weighted_pass,
        sum_weight_squared,
        p_hat,
        n_eff,
    })
}

fn evidence_age_millis(as_of_unix_ms: i64, created_at_unix_ms: i64) -> Result<u64, NumericFailure> {
    if created_at_unix_ms > as_of_unix_ms {
        let future = created_at_unix_ms
            .checked_sub(as_of_unix_ms)
            .ok_or(NumericFailure::InvalidEvidenceTime)?;
        if future > FUTURE_SKEW_MAX_MILLIS {
            return Err(NumericFailure::InvalidEvidenceTime);
        }
        return Ok(0);
    }
    let age = as_of_unix_ms
        .checked_sub(created_at_unix_ms)
        .ok_or(NumericFailure::InvalidEvidenceTime)?;
    u64::try_from(age).map_err(|_| NumericFailure::InvalidEvidenceTime)
}

#[derive(Default)]
struct NeumaierSum {
    sum: f64,
    correction: f64,
}

impl NeumaierSum {
    fn add(&mut self, value: f64) -> Result<(), NumericFailure> {
        if !value.is_finite() {
            return Err(NumericFailure::Numeric);
        }
        let next = self.sum + value;
        if !next.is_finite() {
            return Err(NumericFailure::Numeric);
        }
        let adjustment = if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        let correction = self.correction + adjustment;
        if !adjustment.is_finite() || !correction.is_finite() {
            return Err(NumericFailure::Numeric);
        }
        self.sum = next;
        self.correction = correction;
        Ok(())
    }

    fn total(self) -> Result<f64, NumericFailure> {
        let total = self.sum + self.correction;
        total
            .is_finite()
            .then_some(total)
            .ok_or(NumericFailure::Numeric)
    }
}

/// Contained version-1 statrs Beta inverse-CDF boundary.
pub(crate) fn beta_inverse_cdf_v1(
    probability: f64,
    alpha: f64,
    beta: f64,
) -> Result<f64, BetaQuantileErrorV1> {
    if !probability.is_finite()
        || probability <= 0.0
        || probability >= 1.0
        || !alpha.is_finite()
        || alpha <= 0.0
        || alpha > BETA_SHAPE_CERTIFICATION_MAX_V1
        || !beta.is_finite()
        || beta <= 0.0
        || beta > BETA_SHAPE_CERTIFICATION_MAX_V1
    {
        return Err(BetaQuantileErrorV1::InvalidOrUncertified);
    }
    // statrs 0.18 AS109 can fail to converge for highly asymmetric extreme
    // shapes. The fixed ceiling keeps every accepted call in the certified V1 domain.
    catch_unwind(AssertUnwindSafe(|| {
        let distribution =
            Beta::new(alpha, beta).map_err(|_| BetaQuantileErrorV1::InvalidOrUncertified)?;
        let quantile = distribution.inverse_cdf(probability);
        if quantile.is_finite() && quantile > 0.0 && quantile < 1.0 {
            Ok(quantile)
        } else {
            Err(BetaQuantileErrorV1::InvalidOrUncertified)
        }
    }))
    .map_err(|_| BetaQuantileErrorV1::InvalidOrUncertified)?
}

fn is_uuid_v7(value: Uuid) -> bool {
    value.get_version_num() == 7 && value.get_variant() == Variant::RFC4122
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde::Deserialize;

    use super::*;

    const AS_OF: i64 = 1_800_000_000_000;

    fn id(sequence: u64) -> Uuid {
        Uuid::from_u128(0x018f_0000_0000_7000_8000_0000_0000_0000_u128 + u128::from(sequence))
    }

    fn policy() -> ConfidencePolicyV1 {
        ConfidencePolicyV1::new(32, 1.0, 1, 1, 1.0, 0.0, 3_600.0, 1.0, 1.0, 0.95, 0.0, 0.7).unwrap()
    }

    fn evaluation(
        sequence: u64,
        source: ConfidenceEvaluationSourceV1,
        label: Option<ConfidenceBinaryLabelV1>,
        confidence: Option<f64>,
        promotion_eligible: bool,
        created_at_unix_ms: i64,
    ) -> ConfidenceEvaluationInputV1 {
        ConfidenceEvaluationInputV1::new(
            id(sequence),
            source,
            label,
            confidence.map(f64::to_bits),
            promotion_eligible,
            created_at_unix_ms,
        )
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn neighbor(
        sequence: u64,
        root_sequence: u64,
        distance: f32,
        terminal: ConfidenceTerminalClassV1,
        evaluation: Option<ConfidenceEvaluationInputV1>,
    ) -> ConfidenceNeighborInputV1 {
        ConfidenceNeighborInputV1::new(
            id(1_000 + sequence),
            id(2_000 + sequence),
            id(3_000 + sequence),
            id(4_000 + root_sequence),
            terminal,
            distance.to_bits(),
            evaluation,
        )
        .unwrap()
    }

    fn candidate(
        candidate_id: &str,
        cost_rank: u32,
        neighbors: Vec<ConfidenceNeighborInputV1>,
    ) -> CandidateConfidenceInputV1 {
        CandidateConfidenceInputV1::new(
            candidate_id,
            cost_rank,
            CandidateEvidenceV1::Neighbors(neighbors),
        )
        .unwrap()
    }

    #[test]
    fn single_candidate_evaluation_matches_full_familywise_summary() {
        let inspected = candidate(
            "candidate-b",
            1,
            vec![neighbor(
                1,
                1,
                0.1,
                ConfidenceTerminalClassV1::Completed,
                Some(evaluation(
                    1,
                    ConfidenceEvaluationSourceV1::Judge,
                    Some(ConfidenceBinaryLabelV1::Pass),
                    Some(0.99),
                    true,
                    AS_OF,
                )),
            )],
        );
        let single = evaluate_single_candidate_v1(&policy(), 3, AS_OF, inspected.clone()).unwrap();
        let full = evaluate_confidence_v1(
            &policy(),
            3,
            AS_OF,
            vec![
                CandidateConfidenceInputV1::new("candidate-a", 0, CandidateEvidenceV1::NoPartition)
                    .unwrap(),
                inspected,
                CandidateConfidenceInputV1::new("candidate-c", 2, CandidateEvidenceV1::NoPartition)
                    .unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(single, full.summaries[1]);
    }

    #[test]
    fn policy_rejects_every_invalid_numeric_class() {
        let valid = [
            32.0, 1.0, 1.0, 1.0, 1.0, 0.0, 3_600.0, 1.0, 1.0, 0.95, 0.0, 0.7,
        ];
        for (index, replacement) in [
            (1, f64::NAN),
            (1, 0.0),
            (1, 2.1),
            (4, 0.0),
            (4, 33.0),
            (5, -0.1),
            (5, 1.1),
            (6, 0.0),
            (7, 0.0),
            (8, f64::INFINITY),
            (9, 0.5),
            (9, 1.0),
            (10, -0.1),
            (11, 1.1),
        ] {
            let mut fields = valid;
            fields[index] = replacement;
            assert_eq!(
                ConfidencePolicyV1::new(
                    fields[0] as usize,
                    fields[1],
                    fields[2] as usize,
                    fields[3] as usize,
                    fields[4],
                    fields[5],
                    fields[6],
                    fields[7],
                    fields[8],
                    fields[9],
                    fields[10],
                    fields[11],
                ),
                Err(ConfidenceInputError::InvalidPolicy)
            );
        }
        assert_eq!(
            ConfidencePolicyV1::new(0, 1.0, 1, 1, 1.0, 0.0, 1.0, 1.0, 1.0, 0.95, 0.0, 0.7),
            Err(ConfidenceInputError::InvalidPolicy)
        );
        assert_eq!(
            ConfidencePolicyV1::new(4_096, 1.0, 1, 1, 1.0, 0.0, 1.0, 1.0, 1.0, 0.95, 0.0, 0.7),
            Err(ConfidenceInputError::InvalidPolicy)
        );

        let boundary =
            ConfidencePolicyV1::new(65, 1.0, 1, 1, 1.0, 0.0, 1.0, 1.0, 1.0, 0.95, 0.0, 0.7)
                .unwrap();
        assert!(ConfidenceEngineV1::new(&boundary, 63, AS_OF).is_ok());
        let over = ConfidencePolicyV1::new(64, 1.0, 1, 1, 1.0, 0.0, 1.0, 1.0, 1.0, 0.95, 0.0, 0.7)
            .unwrap();
        assert!(matches!(
            ConfidenceEngineV1::new(&over, 64, AS_OF),
            Err(ConfidenceInputError::CandidateProductExceeded)
        ));
    }

    #[test]
    fn every_terminal_is_raw_but_only_eligible_quality_is_labeled() {
        let terminals = [
            ConfidenceTerminalClassV1::Completed,
            ConfidenceTerminalClassV1::DeterministicFailure,
            ConfidenceTerminalClassV1::OperationalFailure,
            ConfidenceTerminalClassV1::SkippedCooloff,
            ConfidenceTerminalClassV1::CanceledShutdown,
            ConfidenceTerminalClassV1::OrphanedBeforeSchedule,
            ConfidenceTerminalClassV1::OrphanedInFlight,
        ];
        let mut neighbors = terminals
            .into_iter()
            .enumerate()
            .map(|(index, terminal)| {
                let evaluation = match terminal {
                    ConfidenceTerminalClassV1::Completed if index == 0 => Some(evaluation(
                        10,
                        ConfidenceEvaluationSourceV1::Judge,
                        Some(ConfidenceBinaryLabelV1::Pass),
                        Some(0.9),
                        true,
                        AS_OF,
                    )),
                    ConfidenceTerminalClassV1::DeterministicFailure => Some(evaluation(
                        11,
                        ConfidenceEvaluationSourceV1::DeterministicValidator,
                        Some(ConfidenceBinaryLabelV1::Fail),
                        None,
                        true,
                        AS_OF,
                    )),
                    _ => None,
                };
                neighbor(index as u64, index as u64, 0.1, terminal, evaluation)
            })
            .collect::<Vec<_>>();
        neighbors.push(neighbor(
            20,
            20,
            0.1,
            ConfidenceTerminalClassV1::Completed,
            Some(evaluation(
                20,
                ConfidenceEvaluationSourceV1::Judge,
                None,
                Some(0.99),
                false,
                AS_OF,
            )),
        ));
        neighbors.push(neighbor(
            21,
            21,
            0.1,
            ConfidenceTerminalClassV1::Completed,
            Some(evaluation(
                21,
                ConfidenceEvaluationSourceV1::Judge,
                Some(ConfidenceBinaryLabelV1::Pass),
                Some(0.69),
                true,
                AS_OF,
            )),
        ));
        neighbors.push(neighbor(
            22,
            22,
            0.1,
            ConfidenceTerminalClassV1::Completed,
            Some(evaluation(
                22,
                ConfidenceEvaluationSourceV1::Judge,
                Some(ConfidenceBinaryLabelV1::Pass),
                Some(0.99),
                false,
                AS_OF,
            )),
        ));

        let result =
            evaluate_confidence_v1(&policy(), 1, AS_OF, vec![candidate("c", 1, neighbors)])
                .unwrap();
        let summary = &result.summaries[0];
        assert_eq!(summary.raw_points, 10);
        assert_eq!(summary.attempted_roots, 10);
        assert_eq!(summary.labeled_points, 2);
        assert_eq!(summary.labeled_roots, 2);
        assert_eq!(summary.coverage, Some(0.2));
        assert_eq!(
            summary
                .neighbors
                .iter()
                .filter(|neighbor| neighbor.selected_root)
                .count(),
            2
        );
    }

    #[test]
    fn gate_precedence_stops_at_the_first_failed_gate() {
        let pass = |sequence, root, distance| {
            neighbor(
                sequence,
                root,
                distance,
                ConfidenceTerminalClassV1::Completed,
                Some(evaluation(
                    100 + sequence,
                    ConfidenceEvaluationSourceV1::Judge,
                    Some(ConfidenceBinaryLabelV1::Pass),
                    Some(0.99),
                    true,
                    AS_OF,
                )),
            )
        };

        let mut points_policy = policy();
        points_policy.min_points = 2;
        let points = evaluate_confidence_v1(
            &points_policy,
            1,
            AS_OF,
            vec![candidate("points", 1, vec![pass(1, 1, 0.1)])],
        )
        .unwrap();
        assert_eq!(
            points.summaries[0].reason,
            CandidateConfidenceReasonV1::SparsePoints
        );
        assert_eq!(
            points.summaries[0].gates.points,
            ConfidenceGateResultV1::Failed
        );
        assert_eq!(
            points.summaries[0].gates.roots,
            ConfidenceGateResultV1::NotEvaluated
        );

        let mut roots_policy = points_policy.clone();
        roots_policy.min_independent_roots = 2;
        let roots = evaluate_confidence_v1(
            &roots_policy,
            1,
            AS_OF,
            vec![candidate(
                "roots",
                1,
                vec![pass(2, 1, 0.1), pass(3, 1, 0.2)],
            )],
        )
        .unwrap();
        assert_eq!(
            roots.summaries[0].reason,
            CandidateConfidenceReasonV1::InsufficientRoots
        );
        assert_eq!(
            roots.summaries[0].gates.points,
            ConfidenceGateResultV1::Passed
        );
        assert_eq!(
            roots.summaries[0].gates.roots,
            ConfidenceGateResultV1::Failed
        );

        let mut coverage_policy = roots_policy.clone();
        coverage_policy.min_coverage = 0.75;
        let coverage = evaluate_confidence_v1(
            &coverage_policy,
            1,
            AS_OF,
            vec![candidate(
                "coverage",
                1,
                vec![
                    pass(4, 1, 0.1),
                    pass(5, 2, 0.2),
                    neighbor(
                        6,
                        3,
                        0.1,
                        ConfidenceTerminalClassV1::OperationalFailure,
                        None,
                    ),
                    neighbor(7, 4, 0.1, ConfidenceTerminalClassV1::CanceledShutdown, None),
                ],
            )],
        )
        .unwrap();
        assert_eq!(
            coverage.summaries[0].reason,
            CandidateConfidenceReasonV1::LowCoverage
        );
        assert_eq!(
            coverage.summaries[0].gates.roots,
            ConfidenceGateResultV1::Passed
        );
        assert_eq!(
            coverage.summaries[0].gates.coverage,
            ConfidenceGateResultV1::Failed
        );

        let mut effective_policy = roots_policy.clone();
        effective_policy.min_effective_samples = 1.9;
        let effective = evaluate_confidence_v1(
            &effective_policy,
            1,
            AS_OF,
            vec![candidate(
                "effective",
                1,
                vec![pass(8, 1, 0.01), pass(9, 2, 0.99)],
            )],
        )
        .unwrap();
        assert_eq!(
            effective.summaries[0].reason,
            CandidateConfidenceReasonV1::InsufficientEffectiveSamples
        );
        assert_eq!(
            effective.summaries[0].gates.weight_math,
            ConfidenceGateResultV1::Passed
        );
        assert_eq!(
            effective.summaries[0].gates.effective_samples,
            ConfidenceGateResultV1::Failed
        );

        let mut lower_policy = roots_policy;
        lower_policy.promotion_lower_bound = 1.0;
        let lower = evaluate_confidence_v1(
            &lower_policy,
            1,
            AS_OF,
            vec![candidate(
                "lower",
                1,
                vec![pass(10, 1, 0.1), pass(11, 2, 0.1)],
            )],
        )
        .unwrap();
        assert_eq!(
            lower.summaries[0].reason,
            CandidateConfidenceReasonV1::LowerBoundBelowThreshold
        );
        assert_eq!(
            lower.summaries[0].gates.beta_quantile,
            ConfidenceGateResultV1::Passed
        );
        assert_eq!(
            lower.summaries[0].gates.lower_bound,
            ConfidenceGateResultV1::Failed
        );
    }

    #[test]
    fn radius_then_label_independent_root_collapse_uses_exact_ties() {
        let root = 1;
        let older_pass = neighbor(
            1,
            root,
            0.25,
            ConfidenceTerminalClassV1::Completed,
            Some(evaluation(
                50,
                ConfidenceEvaluationSourceV1::Judge,
                Some(ConfidenceBinaryLabelV1::Pass),
                Some(0.9),
                true,
                AS_OF - 1,
            )),
        );
        let newer_fail = neighbor(
            2,
            root,
            0.25,
            ConfidenceTerminalClassV1::Completed,
            Some(evaluation(
                51,
                ConfidenceEvaluationSourceV1::Judge,
                Some(ConfidenceBinaryLabelV1::Fail),
                Some(0.9),
                true,
                AS_OF,
            )),
        );
        let lexicographically_first_fail = neighbor(
            5,
            root,
            0.25,
            ConfidenceTerminalClassV1::Completed,
            Some(evaluation(
                49,
                ConfidenceEvaluationSourceV1::Judge,
                Some(ConfidenceBinaryLabelV1::Fail),
                Some(0.9),
                true,
                AS_OF,
            )),
        );
        let exact_radius = neighbor(
            3,
            2,
            1.0,
            ConfidenceTerminalClassV1::Completed,
            Some(evaluation(
                52,
                ConfidenceEvaluationSourceV1::Judge,
                Some(ConfidenceBinaryLabelV1::Pass),
                Some(0.9),
                true,
                AS_OF,
            )),
        );
        let outside = neighbor(
            4,
            3,
            f32::from_bits(1.0_f32.to_bits() + 1),
            ConfidenceTerminalClassV1::Completed,
            Some(evaluation(
                53,
                ConfidenceEvaluationSourceV1::Judge,
                Some(ConfidenceBinaryLabelV1::Pass),
                Some(0.9),
                true,
                AS_OF,
            )),
        );
        let result = evaluate_confidence_v1(
            &policy(),
            1,
            AS_OF,
            vec![candidate(
                "c",
                1,
                vec![
                    outside,
                    older_pass,
                    exact_radius,
                    newer_fail,
                    lexicographically_first_fail,
                ],
            )],
        )
        .unwrap();
        let summary = &result.summaries[0];
        assert_eq!(summary.raw_points, 4);
        assert_eq!(summary.labeled_points, 4);
        let selected = summary
            .neighbors
            .iter()
            .find(|neighbor| neighbor.evaluation_id == Some(id(49)))
            .unwrap();
        assert!(selected.selected_root);
        assert_eq!(selected.binary_label, Some(ConfidenceBinaryLabelV1::Fail));
        assert_eq!(
            summary
                .neighbors
                .iter()
                .find(|neighbor| neighbor.evaluation_id == Some(id(50)))
                .unwrap()
                .exclusion_reason,
            NeighborExclusionReasonV1::DuplicateRoot
        );
        assert_eq!(
            summary.neighbors.last().unwrap().exclusion_reason,
            NeighborExclusionReasonV1::OutsideRadius
        );
    }

    #[test]
    fn signed_zero_distances_tie_before_timestamp_selection() {
        let older_negative_zero = neighbor(
            1,
            1,
            -0.0,
            ConfidenceTerminalClassV1::Completed,
            Some(evaluation(
                1,
                ConfidenceEvaluationSourceV1::Judge,
                Some(ConfidenceBinaryLabelV1::Fail),
                Some(0.99),
                true,
                AS_OF - 1,
            )),
        );
        let newer_positive_zero = neighbor(
            2,
            1,
            0.0,
            ConfidenceTerminalClassV1::Completed,
            Some(evaluation(
                2,
                ConfidenceEvaluationSourceV1::Judge,
                Some(ConfidenceBinaryLabelV1::Pass),
                Some(0.99),
                true,
                AS_OF,
            )),
        );
        let result = evaluate_confidence_v1(
            &policy(),
            1,
            AS_OF,
            vec![candidate(
                "signed-zero",
                1,
                vec![older_negative_zero, newer_positive_zero],
            )],
        )
        .unwrap();
        let selected = result.summaries[0]
            .neighbors
            .iter()
            .find(|neighbor| neighbor.selected_root)
            .unwrap();
        assert_eq!(selected.evaluation_id, Some(id(2)));
        assert_eq!(selected.distance_bits, 0.0_f32.to_bits());
    }

    #[test]
    fn future_skew_boundary_clamps_then_rejects_globally() {
        let within = candidate(
            "within",
            1,
            vec![neighbor(
                1,
                1,
                0.1,
                ConfidenceTerminalClassV1::Completed,
                Some(evaluation(
                    1,
                    ConfidenceEvaluationSourceV1::Judge,
                    Some(ConfidenceBinaryLabelV1::Pass),
                    Some(0.9),
                    true,
                    AS_OF + FUTURE_SKEW_MAX_MILLIS,
                )),
            )],
        );
        let result = evaluate_confidence_v1(&policy(), 1, AS_OF, vec![within]).unwrap();
        assert_eq!(result.summaries[0].neighbors[0].age_millis, Some(0));

        let invalid = candidate(
            "invalid",
            1,
            vec![neighbor(
                2,
                2,
                0.1,
                ConfidenceTerminalClassV1::Completed,
                Some(evaluation(
                    2,
                    ConfidenceEvaluationSourceV1::Judge,
                    Some(ConfidenceBinaryLabelV1::Pass),
                    Some(0.9),
                    true,
                    AS_OF + FUTURE_SKEW_MAX_MILLIS + 1,
                )),
            )],
        );
        let later =
            CandidateConfidenceInputV1::new("later", 2, CandidateEvidenceV1::NoPartition).unwrap();
        let result = evaluate_confidence_v1(&policy(), 2, AS_OF, vec![later, invalid]).unwrap();
        assert_eq!(
            result.reason,
            ConfidenceDecisionReasonV1::InvalidEvidenceTime
        );
        assert_eq!(
            result.summaries[0].reason,
            CandidateConfidenceReasonV1::InvalidEvidenceTime
        );
        assert_eq!(
            result.summaries[1].reason,
            CandidateConfidenceReasonV1::NotEvaluatedAfterFallback
        );
    }

    #[test]
    fn zero_similarity_weight_is_a_global_numeric_fallback() {
        let input = candidate(
            "zero",
            1,
            vec![neighbor(
                1,
                1,
                1.0,
                ConfidenceTerminalClassV1::Completed,
                Some(evaluation(
                    1,
                    ConfidenceEvaluationSourceV1::Judge,
                    Some(ConfidenceBinaryLabelV1::Pass),
                    Some(0.9),
                    true,
                    AS_OF,
                )),
            )],
        );
        let result = evaluate_confidence_v1(&policy(), 1, AS_OF, vec![input]).unwrap();
        assert_eq!(result.reason, ConfidenceDecisionReasonV1::NumericError);
        assert_eq!(
            result.summaries[0].reason,
            CandidateConfidenceReasonV1::NumericError
        );
        assert_eq!(
            result.summaries[0].gates.weight_math,
            ConfidenceGateResultV1::Failed
        );
    }

    #[test]
    fn cheapest_first_selection_and_global_root_ordinals_ignore_insertion_order() {
        let passing = |candidate_id: &str, cost_rank, base| {
            candidate(
                candidate_id,
                cost_rank,
                (0..4)
                    .map(|offset| {
                        neighbor(
                            base + offset,
                            offset,
                            0.1,
                            ConfidenceTerminalClassV1::Completed,
                            Some(evaluation(
                                100 + base + offset,
                                ConfidenceEvaluationSourceV1::Judge,
                                Some(ConfidenceBinaryLabelV1::Pass),
                                Some(0.99),
                                true,
                                AS_OF,
                            )),
                        )
                    })
                    .collect(),
            )
        };
        let mut strict = policy();
        strict.promotion_lower_bound = 0.2;
        let first = evaluate_confidence_v1(
            &strict,
            3,
            AS_OF,
            vec![
                passing("expensive", 20, 20),
                passing("cheap", 1, 1),
                passing("later", 30, 30),
            ],
        )
        .unwrap();
        let second = evaluate_confidence_v1(
            &strict,
            3,
            AS_OF,
            vec![
                passing("later", 30, 30),
                passing("cheap", 1, 1),
                passing("expensive", 20, 20),
            ],
        )
        .unwrap();
        assert_eq!(first, second);
        assert_eq!(first.recommended_candidate_id.as_deref(), Some("cheap"));
        assert_eq!(
            first.summaries[0].reason,
            CandidateConfidenceReasonV1::Passed
        );
        assert!(
            first.summaries[0]
                .neighbors
                .iter()
                .map(|neighbor| neighbor.root_group_ordinal)
                .eq(0..4)
        );
        assert!(first.summaries[1..].iter().all(|summary| {
            summary.reason == CandidateConfidenceReasonV1::NotEvaluatedAfterWinner
                && summary.neighbors.is_empty()
        }));
    }

    #[test]
    fn incremental_engine_stops_evidence_and_requires_every_summary() {
        let make_first = |sequence| {
            candidate(
                "first",
                1,
                vec![neighbor(
                    sequence,
                    sequence,
                    0.0,
                    ConfidenceTerminalClassV1::Completed,
                    Some(evaluation(
                        sequence,
                        ConfidenceEvaluationSourceV1::Judge,
                        Some(ConfidenceBinaryLabelV1::Pass),
                        Some(0.99),
                        true,
                        AS_OF,
                    )),
                )],
            )
        };

        let engine_policy = policy();
        let mut incomplete = ConfidenceEngineV1::new(&engine_policy, 2, AS_OF).unwrap();
        assert_eq!(
            incomplete.evaluate_candidate(make_first(1)),
            Ok(CandidateConfidenceReasonV1::Passed)
        );
        assert!(!incomplete.needs_evidence());
        assert!(matches!(
            incomplete.finish(),
            Err(ConfidenceInputError::IncompleteCandidateSet)
        ));

        let mut engine = ConfidenceEngineV1::new(&engine_policy, 2, AS_OF).unwrap();
        engine.evaluate_candidate(make_first(2)).unwrap();
        assert_eq!(
            engine.evaluate_candidate(
                CandidateConfidenceInputV1::new("later", 2, CandidateEvidenceV1::NoPartition)
                    .unwrap()
            ),
            Err(ConfidenceInputError::UnexpectedEvidenceState)
        );
        engine.record_unevaluated("later", 2).unwrap();
        let result = engine.finish().unwrap();
        assert_eq!(
            result.summaries[1].reason,
            CandidateConfidenceReasonV1::NotEvaluatedAfterWinner
        );
    }

    #[test]
    fn root_group_ordinals_are_global_across_evaluated_candidates() {
        let labeled = |sequence, root, label, distance| {
            neighbor(
                sequence,
                root,
                distance,
                ConfidenceTerminalClassV1::Completed,
                Some(evaluation(
                    700 + sequence,
                    ConfidenceEvaluationSourceV1::Judge,
                    Some(label),
                    Some(0.99),
                    true,
                    AS_OF,
                )),
            )
        };
        let mut selection_policy = policy();
        selection_policy.min_points = 2;
        selection_policy.min_independent_roots = 2;
        selection_policy.promotion_lower_bound = 0.3;
        let result = evaluate_confidence_v1(
            &selection_policy,
            2,
            AS_OF,
            vec![
                candidate(
                    "first",
                    1,
                    vec![
                        labeled(1, 1, ConfidenceBinaryLabelV1::Fail, 0.1),
                        labeled(2, 2, ConfidenceBinaryLabelV1::Fail, 0.2),
                    ],
                ),
                candidate(
                    "second",
                    2,
                    vec![
                        labeled(10, 1, ConfidenceBinaryLabelV1::Pass, 0.05),
                        labeled(11, 3, ConfidenceBinaryLabelV1::Pass, 0.1),
                        labeled(12, 4, ConfidenceBinaryLabelV1::Pass, 0.2),
                        labeled(13, 5, ConfidenceBinaryLabelV1::Pass, 0.3),
                    ],
                ),
            ],
        )
        .unwrap();
        assert_eq!(
            result.summaries[0].reason,
            CandidateConfidenceReasonV1::LowerBoundBelowThreshold
        );
        assert_eq!(
            result.summaries[1].reason,
            CandidateConfidenceReasonV1::Passed
        );
        let first_ordinal = result.summaries[0]
            .neighbors
            .iter()
            .find(|neighbor| neighbor.evidence_vector_link_id == id(1_001))
            .unwrap()
            .root_group_ordinal;
        let second_ordinal = result.summaries[1]
            .neighbors
            .iter()
            .find(|neighbor| neighbor.evidence_vector_link_id == id(1_010))
            .unwrap()
            .root_group_ordinal;
        assert_eq!(first_ordinal, second_ordinal);
        assert_eq!(first_ordinal, 0);
    }

    #[test]
    fn candidate_count_bonferroni_is_more_conservative() {
        let evidence = || {
            (0..4)
                .map(|offset| {
                    neighbor(
                        50 + offset,
                        offset,
                        0.1,
                        ConfidenceTerminalClassV1::Completed,
                        Some(evaluation(
                            500 + offset,
                            ConfidenceEvaluationSourceV1::Judge,
                            Some(ConfidenceBinaryLabelV1::Pass),
                            Some(0.99),
                            true,
                            AS_OF,
                        )),
                    )
                })
                .collect()
        };
        let one = evaluate_confidence_v1(&policy(), 1, AS_OF, vec![candidate("c", 1, evidence())])
            .unwrap();
        let two = evaluate_confidence_v1(
            &policy(),
            2,
            AS_OF,
            vec![
                candidate("c", 1, evidence()),
                CandidateConfidenceInputV1::new("none", 2, CandidateEvidenceV1::NoPartition)
                    .unwrap(),
            ],
        )
        .unwrap();
        assert!(two.summaries[0].lower_bound.unwrap() < one.summaries[0].lower_bound.unwrap());
    }

    #[test]
    fn malformed_inputs_and_beta_boundaries_fail_closed() {
        assert!(beta_inverse_cdf_v1(0.0, 1.0, 1.0).is_err());
        assert!(beta_inverse_cdf_v1(0.5, f64::NAN, 1.0).is_err());
        assert!(beta_inverse_cdf_v1(0.5, 1.0, 0.0).is_err());
        assert!(beta_inverse_cdf_v1(0.05, BETA_SHAPE_CERTIFICATION_MAX_V1, 1.0).is_ok());
        let above_ceiling = f64::from_bits(
            BETA_SHAPE_CERTIFICATION_MAX_V1
                .to_bits()
                .checked_add(1)
                .unwrap(),
        );
        assert!(beta_inverse_cdf_v1(0.05, above_ceiling, 1.0).is_err());
        assert_eq!(checked_beta_shapes(f64::MAX, 1.0, 1.0, f64::MAX), None);
        assert_eq!(
            ConfidenceEvaluationInputV1::new(
                id(1),
                ConfidenceEvaluationSourceV1::DeterministicValidator,
                Some(ConfidenceBinaryLabelV1::Pass),
                None,
                true,
                AS_OF,
            ),
            Err(ConfidenceInputError::InvalidNeighbor)
        );
        assert_eq!(
            ConfidenceNeighborInputV1::new(
                id(1),
                id(2),
                id(3),
                id(4),
                ConfidenceTerminalClassV1::Completed,
                f32::NAN.to_bits(),
                None,
            ),
            Err(ConfidenceInputError::InvalidNeighbor)
        );
    }

    #[test]
    fn finite_uncertified_beta_shapes_remain_auditable() {
        let extreme = ConfidencePolicyV1::new(
            1,
            1.0,
            1,
            1,
            1.0,
            0.0,
            3_600.0,
            BETA_SHAPE_CERTIFICATION_MAX_V1,
            1.0,
            0.95,
            0.0,
            0.7,
        )
        .unwrap();
        let input = candidate(
            "extreme",
            1,
            vec![neighbor(
                1,
                1,
                0.1,
                ConfidenceTerminalClassV1::Completed,
                Some(evaluation(
                    1,
                    ConfidenceEvaluationSourceV1::Judge,
                    Some(ConfidenceBinaryLabelV1::Pass),
                    Some(0.99),
                    true,
                    AS_OF,
                )),
            )],
        );

        let result = evaluate_confidence_v1(&extreme, 1, AS_OF, vec![input]).unwrap();
        let summary = &result.summaries[0];
        assert_eq!(result.reason, ConfidenceDecisionReasonV1::NumericError);
        assert_eq!(summary.reason, CandidateConfidenceReasonV1::NumericError);
        assert_eq!(
            summary.beta_alpha,
            Some(BETA_SHAPE_CERTIFICATION_MAX_V1 + 1.0)
        );
        assert_eq!(summary.beta_beta, Some(1.0));
        assert_eq!(summary.lower_bound, None);
        assert_eq!(summary.gates.beta_quantile, ConfidenceGateResultV1::Failed);
    }

    #[derive(Deserialize)]
    struct GoldenFixture {
        schema: String,
        algorithm_ids: Vec<String>,
        statrs_version: String,
        generator: GoldenGenerator,
        quantiles: Vec<GoldenQuantile>,
        decision: GoldenDecision,
    }

    #[derive(Deserialize)]
    struct GoldenGenerator {
        name: String,
        version: String,
        precision_decimal_digits: u32,
        script_sha256: String,
    }

    #[derive(Deserialize)]
    struct GoldenQuantile {
        name: String,
        probability: String,
        alpha: String,
        beta: String,
        expected: String,
    }

    #[derive(Deserialize)]
    struct GoldenDecision {
        policy: GoldenPolicy,
        as_of_unix_ms: i64,
        candidates: Vec<GoldenCandidate>,
        expected_winner: String,
        expected_reasons: Vec<String>,
        expected_lower_bounds: Vec<Option<String>>,
    }

    #[derive(Deserialize)]
    struct GoldenPolicy {
        top_k: usize,
        radius: String,
        min_points: usize,
        min_independent_roots: usize,
        min_effective_samples: String,
        min_coverage: String,
        half_life_seconds: String,
        prior_success: String,
        prior_failure: String,
        familywise_credible_level: String,
        promotion_lower_bound: String,
        judge_confidence_floor: String,
    }

    #[derive(Deserialize)]
    struct GoldenCandidate {
        candidate_id: String,
        cost_rank: u32,
        partition_present: bool,
        neighbors: Vec<GoldenNeighbor>,
    }

    #[derive(Deserialize)]
    struct GoldenNeighbor {
        evidence_sequence: u64,
        attempt_sequence: u64,
        anchor_sequence: u64,
        root_sequence: u64,
        terminal: String,
        distance_bits: String,
        evaluation: Option<GoldenEvaluation>,
    }

    #[derive(Deserialize)]
    struct GoldenEvaluation {
        evaluation_sequence: u64,
        source: String,
        label: Option<String>,
        judge_confidence_bits: Option<String>,
        promotion_eligible: bool,
        created_at_unix_ms: i64,
    }

    #[test]
    fn independent_mpmath_fixture_matches_quantiles_and_complete_decision() {
        let fixture_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/spec07_confidence_v1.json");
        let fixture: GoldenFixture =
            serde_json::from_slice(&std::fs::read(fixture_path).unwrap()).unwrap();
        assert_eq!(fixture.schema, "nemo.relay.router.confidence-goldens@1");
        assert_eq!(fixture.statrs_version, "0.18.0");
        assert_eq!(
            fixture.algorithm_ids,
            [
                "linear_radius_v1",
                "min_distance_newest_evaluation_lex_id_v1",
                "labeled_roots_over_attempted_roots_v1",
                "exp2_half_life_v1",
                "future_skew_300000ms_v1",
                "neumaier_f64_no_fma_v1",
                "kish_effective_sample_v1",
                "beta_effective_sample_v1",
                "bonferroni_v1",
                "statrs_0_18_0_v1",
            ]
        );
        assert_eq!(fixture.generator.name, "mpmath");
        assert_eq!(fixture.generator.version, "1.3.0");
        assert!(fixture.generator.precision_decimal_digits >= 80);
        assert_eq!(fixture.generator.script_sha256.len(), 64);
        let script_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../scripts/dev/generate_spec07_confidence_goldens.py");
        assert_eq!(
            fixture.generator.script_sha256,
            crate::fingerprint::sha256_hex(&std::fs::read(script_path).unwrap())
        );

        for case in fixture.quantiles {
            let probability = parse_f64(&case.probability);
            let alpha = parse_f64(&case.alpha);
            let beta = parse_f64(&case.beta);
            let expected = parse_f64(&case.expected);
            let actual = beta_inverse_cdf_v1(probability, alpha, beta)
                .unwrap_or_else(|_| panic!("golden quantile failed for {}", case.name));
            assert_tail_close(&case.name, actual, expected);
        }

        let golden_policy = fixture.decision.policy;
        let policy = ConfidencePolicyV1::new(
            golden_policy.top_k,
            parse_f64(&golden_policy.radius),
            golden_policy.min_points,
            golden_policy.min_independent_roots,
            parse_f64(&golden_policy.min_effective_samples),
            parse_f64(&golden_policy.min_coverage),
            parse_f64(&golden_policy.half_life_seconds),
            parse_f64(&golden_policy.prior_success),
            parse_f64(&golden_policy.prior_failure),
            parse_f64(&golden_policy.familywise_credible_level),
            parse_f64(&golden_policy.promotion_lower_bound),
            parse_f64(&golden_policy.judge_confidence_floor),
        )
        .unwrap();
        let candidates = fixture
            .decision
            .candidates
            .into_iter()
            .map(golden_candidate)
            .collect::<Vec<_>>();
        let result = evaluate_confidence_v1(
            &policy,
            candidates.len(),
            fixture.decision.as_of_unix_ms,
            candidates,
        )
        .unwrap();
        assert_eq!(
            result.recommended_candidate_id.as_deref(),
            Some(fixture.decision.expected_winner.as_str())
        );
        let reasons = result
            .summaries
            .iter()
            .map(|summary| reason_name(summary.reason))
            .collect::<Vec<_>>();
        assert_eq!(reasons, fixture.decision.expected_reasons);
        for (index, expected) in fixture.decision.expected_lower_bounds.iter().enumerate() {
            match (result.summaries[index].lower_bound, expected) {
                (Some(actual), Some(expected)) => {
                    assert_tail_close("decision lower bound", actual, parse_f64(expected));
                }
                (None, None) => {}
                values => panic!("lower-bound presence mismatch: {values:?}"),
            }
        }
    }

    fn golden_candidate(candidate: GoldenCandidate) -> CandidateConfidenceInputV1 {
        let evidence = if candidate.partition_present {
            CandidateEvidenceV1::Neighbors(
                candidate
                    .neighbors
                    .into_iter()
                    .map(|neighbor| {
                        let evaluation = neighbor.evaluation.map(|evaluation| {
                            ConfidenceEvaluationInputV1::new(
                                id(evaluation.evaluation_sequence),
                                match evaluation.source.as_str() {
                                    "judge" => ConfidenceEvaluationSourceV1::Judge,
                                    "deterministic_validator" => {
                                        ConfidenceEvaluationSourceV1::DeterministicValidator
                                    }
                                    value => panic!("unknown source {value}"),
                                },
                                evaluation.label.as_deref().map(|label| match label {
                                    "pass" => ConfidenceBinaryLabelV1::Pass,
                                    "fail" => ConfidenceBinaryLabelV1::Fail,
                                    value => panic!("unknown label {value}"),
                                }),
                                evaluation
                                    .judge_confidence_bits
                                    .as_deref()
                                    .map(parse_hex_u64),
                                evaluation.promotion_eligible,
                                evaluation.created_at_unix_ms,
                            )
                            .unwrap()
                        });
                        ConfidenceNeighborInputV1::new(
                            id(neighbor.evidence_sequence),
                            id(neighbor.attempt_sequence),
                            id(neighbor.anchor_sequence),
                            id(neighbor.root_sequence),
                            match neighbor.terminal.as_str() {
                                "completed" => ConfidenceTerminalClassV1::Completed,
                                "deterministic_failure" => {
                                    ConfidenceTerminalClassV1::DeterministicFailure
                                }
                                "operational_failure" => {
                                    ConfidenceTerminalClassV1::OperationalFailure
                                }
                                value => panic!("unknown terminal {value}"),
                            },
                            parse_hex_u32(&neighbor.distance_bits),
                            evaluation,
                        )
                        .unwrap()
                    })
                    .collect(),
            )
        } else {
            CandidateEvidenceV1::NoPartition
        };
        CandidateConfidenceInputV1::new(candidate.candidate_id, candidate.cost_rank, evidence)
            .unwrap()
    }

    fn reason_name(reason: CandidateConfidenceReasonV1) -> &'static str {
        match reason {
            CandidateConfidenceReasonV1::NoPartition => "no_partition",
            CandidateConfidenceReasonV1::SparsePoints => "sparse_points",
            CandidateConfidenceReasonV1::InsufficientRoots => "insufficient_roots",
            CandidateConfidenceReasonV1::LowCoverage => "low_coverage",
            CandidateConfidenceReasonV1::InvalidEvidenceTime => "invalid_evidence_time",
            CandidateConfidenceReasonV1::NumericError => "numeric_error",
            CandidateConfidenceReasonV1::InsufficientEffectiveSamples => {
                "insufficient_effective_samples"
            }
            CandidateConfidenceReasonV1::LowerBoundBelowThreshold => "lower_bound_below_threshold",
            CandidateConfidenceReasonV1::Passed => "passed",
            CandidateConfidenceReasonV1::NotEvaluatedAfterWinner => "not_evaluated_after_winner",
            CandidateConfidenceReasonV1::NotEvaluatedAfterFallback => {
                "not_evaluated_after_fallback"
            }
        }
    }

    fn parse_f64(value: &str) -> f64 {
        value.parse().unwrap()
    }

    fn parse_hex_u64(value: &str) -> u64 {
        u64::from_str_radix(value.trim_start_matches("0x"), 16).unwrap()
    }

    fn parse_hex_u32(value: &str) -> u32 {
        u32::from_str_radix(value.trim_start_matches("0x"), 16).unwrap()
    }

    fn assert_tail_close(name: &str, actual: f64, expected: f64) {
        assert!(
            actual > 0.0 && actual < 1.0,
            "{name}: boundary result {actual}"
        );
        let absolute_error = (actual - expected).abs();
        let tail_scale = expected.min(1.0 - expected);
        let relative_tail_error = absolute_error / tail_scale;
        assert!(
            absolute_error <= 1.0e-12 && relative_tail_error <= 1.0e-8,
            "{name}: actual={actual:.17e} expected={expected:.17e} abs={absolute_error:.3e} tail_rel={relative_tail_error:.3e}"
        );
    }
}
