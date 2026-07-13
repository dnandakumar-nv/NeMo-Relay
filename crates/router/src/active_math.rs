// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Bounded version-1 actual-outcome decay and noninferiority math.

use std::panic::{AssertUnwindSafe, catch_unwind};

use serde::{Serialize, Serializer};
use sha2::{Digest, Sha256};
use statrs::distribution::{Beta, ContinuousCDF};

pub(crate) const ACTIVE_MATH_SHAPE_VERSION_V1: u32 = 1;
pub(crate) const ACTIVE_MATH_GRID_PANELS_V1: usize = 4_096;
pub(crate) const ACTIVE_MATH_CDF_GUARD_V1: f64 = 1e-9;
pub(crate) const ACTIVE_MATH_MONOTONIC_REPAIR_MAX_V1: f64 = 1e-11;
pub(crate) const ACTIVE_MATH_POSTERIOR_RESOLUTION_V1: f64 =
    1.0 / ACTIVE_MATH_GRID_PANELS_V1 as f64 + 4e-9;
pub(crate) const ACTIVE_MATH_QUANTILE_STEPS_PER_ENDPOINT_V1: usize = 64;
pub(crate) const ACTIVE_MATH_RAW_BETA_CDF_CALLS_MAX_V1: u32 = 2_200_000;
pub(crate) const ACTIVE_MATH_MEMORY_BYTES_MAX_V1: usize = 64 * 1024 * 1024;
pub(crate) const ACTIVE_MATH_ROOTS_MAX_V1: usize = 65_536;
pub(crate) const ACTIVE_MATH_FUTURE_SKEW_MS_V1: i64 = 300_000;
pub(crate) const ACTIVE_MATH_DURATION_SECONDS_MAX_V1: u32 = 31_536_000;
pub(crate) const ACTIVE_MATH_BUILD_ID_V1: &str = env!("ACTIVE_MATH_BUILD_ID");
const ACTIVE_MATH_INPUT_DOMAIN_V1: &[u8] = b"nemo-relay-router/active-math-input/v1\0";
const ACTIVE_MATH_ALGORITHM_DOMAIN_V1: &[u8] = b"nemo-relay-router/active-math-algorithm/v1\0";
pub(crate) const ACTIVE_MATH_CROSS_PLATFORM_ENDPOINT_TOLERANCE_V1: f64 = 2e-9;
const ACTIVE_MATH_ALGORITHM_FIELDS_V1: &[&str] = &[
    "root_outcome_reducer_v1",
    "libm_0_2_16_exp2_force_soft_v1",
    "future_skew_300000ms_v1",
    "neumaier_f64_no_fma_v1",
    "monotone_beta_quantile_bounds_isotonic_1e-11_v1",
    "statrs_beta_cdf_0_18_0_v1",
    "posterior_resolution_4096_4e-9_v1",
    "bonferroni_max_looks_v1",
    "outcome_attribution_gate_v1",
    "noninferiority_math_v1",
];

/// A finite binary64 value retained by exact bits and serialized as fixed-width hex.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ActiveAuditF64V1(u64);

impl ActiveAuditF64V1 {
    pub(crate) fn new(value: f64) -> Result<Self, ActiveMathErrorV1> {
        value
            .is_finite()
            .then_some(Self(value.to_bits()))
            .ok_or(ActiveMathErrorV1::Nonfinite)
    }

    pub(crate) fn from_bits(bits: u64) -> Result<Self, ActiveMathErrorV1> {
        Self::new(f64::from_bits(bits))
    }

    pub(crate) const fn bits(self) -> u64 {
        self.0
    }

    pub(crate) fn value(self) -> f64 {
        f64::from_bits(self.0)
    }
}

impl Serialize for ActiveAuditF64V1 {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&format!("{:016x}", self.0))
    }
}

/// Stable fail-closed outcomes from the pure numeric boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ActiveMathErrorV1 {
    InvalidInput,
    InvalidOrder,
    FutureSkew,
    Nonfinite,
    NumericFailure,
    InvalidShape,
    CdfFailure,
    SymmetryDisagreement,
    NonmonotoneCdf,
    QuantileUnresolved,
    CdfBudgetExceeded,
    DisjointOrientations,
    PosteriorResolutionExceeded,
    Canceled,
}

/// One randomized arm admitted to actual-outcome learning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ActiveOutcomeArmV1 {
    Treatment,
    Control,
}

/// One eligible Bernoulli outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum ActiveOutcomeLabelV1 {
    Success,
    Failure,
}

/// One frozen cumulative look member in immutable pre-treatment order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ActiveLookMemberV1 {
    pub(crate) cap_ordinal: u64,
    pub(crate) admission_unix_ms: i64,
    pub(crate) arm: ActiveOutcomeArmV1,
    pub(crate) label: Option<ActiveOutcomeLabelV1>,
}

/// Validated policy inputs for one completed actual-outcome look.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct ActiveLookPolicyV1 {
    actual_outcome_half_life_seconds: u32,
    min_treatment_roots: u32,
    min_control_roots: u32,
    min_treatment_effective_weight: f64,
    min_control_effective_weight: f64,
    noninferiority_margin: f64,
    noninferiority_probability: f64,
    rollback_probability: f64,
    max_looks: u32,
    per_look_noninferiority_threshold: f64,
}

impl ActiveLookPolicyV1 {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        actual_outcome_half_life_seconds: u32,
        min_treatment_roots: u32,
        min_control_roots: u32,
        min_treatment_effective_weight: f64,
        min_control_effective_weight: f64,
        noninferiority_margin: f64,
        noninferiority_probability: f64,
        rollback_probability: f64,
        max_looks: u32,
    ) -> Result<Self, ActiveMathErrorV1> {
        let per_look_noninferiority_threshold =
            bonferroni_threshold_v1(noninferiority_probability, max_looks)
                .ok_or(ActiveMathErrorV1::InvalidInput)?;
        let valid = (1..=ACTIVE_MATH_DURATION_SECONDS_MAX_V1)
            .contains(&actual_outcome_half_life_seconds)
            && min_treatment_roots >= 32
            && min_control_roots >= 32
            && usize::try_from(min_treatment_roots)
                .is_ok_and(|value| value <= ACTIVE_MATH_ROOTS_MAX_V1)
            && usize::try_from(min_control_roots)
                .is_ok_and(|value| value <= ACTIVE_MATH_ROOTS_MAX_V1)
            && positive_finite(min_treatment_effective_weight)
            && positive_finite(min_control_effective_weight)
            && min_treatment_effective_weight >= f64::from(min_treatment_roots) / 2.0
            && min_control_effective_weight >= f64::from(min_control_roots) / 2.0
            && min_treatment_effective_weight <= ACTIVE_MATH_ROOTS_MAX_V1 as f64
            && min_control_effective_weight <= ACTIVE_MATH_ROOTS_MAX_V1 as f64
            && noninferiority_margin.is_finite()
            && (0.0..=0.25).contains(&noninferiority_margin)
            && !(noninferiority_margin == 0.0 && noninferiority_margin.is_sign_negative())
            && noninferiority_probability.is_finite()
            && (0.99..1.0).contains(&noninferiority_probability)
            && rollback_probability.is_finite()
            && (0.95..1.0).contains(&rollback_probability)
            && (1..=256).contains(&max_looks)
            && prior_only_noninferiority_v1(noninferiority_margin).is_some_and(|prior_only| {
                per_look_noninferiority_threshold > prior_only + ACTIVE_MATH_POSTERIOR_RESOLUTION_V1
            });
        if !valid {
            return Err(ActiveMathErrorV1::InvalidInput);
        }
        Ok(Self {
            actual_outcome_half_life_seconds,
            min_treatment_roots,
            min_control_roots,
            min_treatment_effective_weight,
            min_control_effective_weight,
            noninferiority_margin,
            noninferiority_probability,
            rollback_probability,
            max_looks,
            per_look_noninferiority_threshold,
        })
    }

    pub(crate) const fn actual_outcome_half_life_seconds(self) -> u32 {
        self.actual_outcome_half_life_seconds
    }

    pub(crate) const fn per_look_noninferiority_threshold(self) -> f64 {
        self.per_look_noninferiority_threshold
    }

    fn audit(self) -> Result<ActiveLookPolicyAuditV1, ActiveMathErrorV1> {
        Ok(ActiveLookPolicyAuditV1 {
            actual_outcome_half_life_seconds: self.actual_outcome_half_life_seconds,
            min_treatment_roots: self.min_treatment_roots,
            min_control_roots: self.min_control_roots,
            min_treatment_effective_weight: ActiveAuditF64V1::new(
                self.min_treatment_effective_weight,
            )?,
            min_control_effective_weight: ActiveAuditF64V1::new(self.min_control_effective_weight)?,
            noninferiority_margin: ActiveAuditF64V1::new(self.noninferiority_margin)?,
            noninferiority_probability: ActiveAuditF64V1::new(self.noninferiority_probability)?,
            per_look_noninferiority_threshold: ActiveAuditF64V1::new(
                self.per_look_noninferiority_threshold,
            )?,
            rollback_probability: ActiveAuditF64V1::new(self.rollback_probability)?,
            max_looks: self.max_looks,
        })
    }
}

/// Written-order sequential correction shared with config validation.
pub(crate) fn bonferroni_threshold_v1(probability: f64, max_looks: u32) -> Option<f64> {
    if !probability.is_finite() || !(0.0..1.0).contains(&probability) || max_looks == 0 {
        return None;
    }
    let complement = 1.0 - probability;
    let per_look_complement = complement / f64::from(max_looks);
    let threshold = 1.0 - per_look_complement;
    (threshold.is_finite() && threshold < 1.0).then_some(threshold)
}

/// Written-order prior-only noninferiority for the fixed independent uniforms.
pub(crate) fn prior_only_noninferiority_v1(margin: f64) -> Option<f64> {
    if !margin.is_finite()
        || !(0.0..=0.25).contains(&margin)
        || (margin == 0.0 && margin.is_sign_negative())
    {
        return None;
    }
    let one_minus_margin = 1.0 - margin;
    let squared = one_minus_margin * one_minus_margin;
    let probability = 1.0 - squared / 2.0;
    probability.is_finite().then_some(probability)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct ActiveLookPolicyAuditV1 {
    pub(crate) actual_outcome_half_life_seconds: u32,
    pub(crate) min_treatment_roots: u32,
    pub(crate) min_control_roots: u32,
    pub(crate) min_treatment_effective_weight: ActiveAuditF64V1,
    pub(crate) min_control_effective_weight: ActiveAuditF64V1,
    pub(crate) noninferiority_margin: ActiveAuditF64V1,
    pub(crate) noninferiority_probability: ActiveAuditF64V1,
    pub(crate) per_look_noninferiority_threshold: ActiveAuditF64V1,
    pub(crate) rollback_probability: ActiveAuditF64V1,
    pub(crate) max_looks: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct ActiveArmAuditV1 {
    pub(crate) denominator: u32,
    pub(crate) labeled: u32,
    pub(crate) successes: u32,
    pub(crate) failures: u32,
    pub(crate) success_weight: ActiveAuditF64V1,
    pub(crate) failure_weight: ActiveAuditF64V1,
    pub(crate) effective_weight: ActiveAuditF64V1,
    pub(crate) beta_alpha: ActiveAuditF64V1,
    pub(crate) beta_beta: ActiveAuditF64V1,
    pub(crate) label_rate: ActiveAuditF64V1,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct PosteriorSufficientStatisticsV1 {
    pub(crate) treatment_success_weight_bits: u64,
    pub(crate) treatment_failure_weight_bits: u64,
    pub(crate) control_success_weight_bits: u64,
    pub(crate) control_failure_weight_bits: u64,
    pub(crate) noninferiority_margin_bits: u64,
}

impl PosteriorSufficientStatisticsV1 {
    fn from_arms(treatment: ActiveArmAuditV1, control: ActiveArmAuditV1, margin: f64) -> Self {
        Self {
            treatment_success_weight_bits: treatment.success_weight.bits(),
            treatment_failure_weight_bits: treatment.failure_weight.bits(),
            control_success_weight_bits: control.success_weight.bits(),
            control_failure_weight_bits: control.failure_weight.bits(),
            noninferiority_margin_bits: margin.to_bits(),
        }
    }

    pub(crate) fn shapes(self) -> Result<(f64, f64, f64, f64, f64), ActiveMathErrorV1> {
        let treatment_success = finite_nonnegative(self.treatment_success_weight_bits)?;
        let treatment_failure = finite_nonnegative(self.treatment_failure_weight_bits)?;
        let control_success = finite_nonnegative(self.control_success_weight_bits)?;
        let control_failure = finite_nonnegative(self.control_failure_weight_bits)?;
        let margin = f64::from_bits(self.noninferiority_margin_bits);
        if !margin.is_finite()
            || !(0.0..=0.25).contains(&margin)
            || (margin == 0.0 && margin.is_sign_negative())
        {
            return Err(ActiveMathErrorV1::InvalidInput);
        }
        Ok((
            checked_shape(treatment_success)?,
            checked_shape(treatment_failure)?,
            checked_shape(control_success)?,
            checked_shape(control_failure)?,
            margin,
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct ProbabilityIntervalAuditV1 {
    pub(crate) lower: ActiveAuditF64V1,
    pub(crate) upper: ActiveAuditF64V1,
}

impl ProbabilityIntervalAuditV1 {
    pub(crate) fn new(lower: f64, upper: f64) -> Result<Self, ActiveMathErrorV1> {
        if !valid_probability(lower) || !valid_probability(upper) || lower > upper {
            return Err(ActiveMathErrorV1::NumericFailure);
        }
        Ok(Self {
            lower: ActiveAuditF64V1::new(lower)?,
            upper: ActiveAuditF64V1::new(upper)?,
        })
    }

    pub(crate) fn lower(self) -> f64 {
        self.lower.value()
    }

    pub(crate) fn upper(self) -> f64 {
        self.upper.value()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub(crate) struct ActiveLookGateAuditV1 {
    pub(crate) treatment_raw_roots: bool,
    pub(crate) control_raw_roots: bool,
    pub(crate) treatment_effective_weight: bool,
    pub(crate) control_effective_weight: bool,
    pub(crate) treatment_attribution: bool,
    pub(crate) control_attribution: bool,
    pub(crate) differential_attribution: bool,
    pub(crate) noninferiority: bool,
    pub(crate) rollback: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LookProducedStateV1 {
    Collecting,
    Passed,
    Rollback,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[allow(clippy::enum_variant_names)] // Persisted reason names intentionally share the contract prefix.
pub(crate) enum LookSkipReasonV1 {
    InsufficientTreatmentRoots,
    InsufficientControlRoots,
    InsufficientBothArms,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct SkippedLookAuditV1 {
    pub(crate) shape_version: u32,
    pub(crate) active_math_build_id: String,
    pub(crate) active_math_algorithm_id_sha256: String,
    pub(crate) input_identity_sha256: String,
    pub(crate) as_of_unix_ms: i64,
    pub(crate) treatment_denominator: u32,
    pub(crate) treatment_labeled: u32,
    pub(crate) control_denominator: u32,
    pub(crate) control_labeled: u32,
    pub(crate) reason: LookSkipReasonV1,
    pub(crate) raw_beta_cdf_calls: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct PosteriorAuditV1 {
    pub(crate) noninferiority_g: ProbabilityIntervalAuditV1,
    pub(crate) noninferiority_h: ProbabilityIntervalAuditV1,
    pub(crate) noninferiority: ProbabilityIntervalAuditV1,
    pub(crate) raw_beta_cdf_calls: u32,
    pub(crate) operational_cdf_queries: u32,
    pub(crate) max_symmetry_disagreement: ActiveAuditF64V1,
    pub(crate) max_monotonic_repair: ActiveAuditF64V1,
    pub(crate) max_quantile_width: ActiveAuditF64V1,
    pub(crate) cdf_transcript_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct ActiveLookAuditV1 {
    pub(crate) shape_version: u32,
    pub(crate) active_math_build_id: String,
    pub(crate) active_math_algorithm_id_sha256: String,
    pub(crate) input_identity_sha256: String,
    pub(crate) as_of_unix_ms: i64,
    pub(crate) policy: ActiveLookPolicyAuditV1,
    pub(crate) treatment: ActiveArmAuditV1,
    pub(crate) control: ActiveArmAuditV1,
    pub(crate) noninferiority_g: ProbabilityIntervalAuditV1,
    pub(crate) noninferiority_h: ProbabilityIntervalAuditV1,
    pub(crate) noninferiority: ProbabilityIntervalAuditV1,
    pub(crate) rollback_lower: ActiveAuditF64V1,
    pub(crate) gates: ActiveLookGateAuditV1,
    pub(crate) state: LookProducedStateV1,
    pub(crate) raw_beta_cdf_calls: u32,
    pub(crate) operational_cdf_queries: u32,
    pub(crate) max_symmetry_disagreement: ActiveAuditF64V1,
    pub(crate) max_monotonic_repair: ActiveAuditF64V1,
    pub(crate) max_quantile_width: ActiveAuditF64V1,
    pub(crate) cdf_transcript_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "result")]
pub(crate) enum ActiveLookEvaluationV1 {
    Skipped(SkippedLookAuditV1),
    Completed(Box<ActiveLookAuditV1>),
}

/// Evaluate one frozen whole-tranche cumulative look.
pub(crate) fn evaluate_active_look_v1(
    policy: ActiveLookPolicyV1,
    as_of_unix_ms: i64,
    members: &[ActiveLookMemberV1],
) -> Result<ActiveLookEvaluationV1, ActiveMathErrorV1> {
    evaluate_active_look_with_control_v1(policy, as_of_unix_ms, members, |_| true)
}

/// Evaluate while allowing the owner to cancel between dyadic grid points.
pub(crate) fn evaluate_active_look_with_control_v1<Control>(
    policy: ActiveLookPolicyV1,
    as_of_unix_ms: i64,
    members: &[ActiveLookMemberV1],
    mut control: Control,
) -> Result<ActiveLookEvaluationV1, ActiveMathErrorV1>
where
    Control: FnMut(usize) -> bool,
{
    let (treatment, control_arm) = aggregate_arms_v1(
        policy.actual_outcome_half_life_seconds,
        as_of_unix_ms,
        members,
    )?;
    let input_identity_sha256 = active_look_input_identity_v1(policy, as_of_unix_ms, members)?;
    let treatment_raw = treatment.labeled >= policy.min_treatment_roots;
    let control_raw = control_arm.labeled >= policy.min_control_roots;
    if !treatment_raw || !control_raw {
        let reason = match (treatment_raw, control_raw) {
            (false, false) => LookSkipReasonV1::InsufficientBothArms,
            (false, true) => LookSkipReasonV1::InsufficientTreatmentRoots,
            (true, false) => LookSkipReasonV1::InsufficientControlRoots,
            (true, true) => unreachable!("the successful case returned above"),
        };
        return Ok(ActiveLookEvaluationV1::Skipped(SkippedLookAuditV1 {
            shape_version: ACTIVE_MATH_SHAPE_VERSION_V1,
            active_math_build_id: ACTIVE_MATH_BUILD_ID_V1.to_string(),
            active_math_algorithm_id_sha256: active_math_algorithm_identity_v1(),
            input_identity_sha256,
            as_of_unix_ms,
            treatment_denominator: treatment.denominator,
            treatment_labeled: treatment.labeled,
            control_denominator: control_arm.denominator,
            control_labeled: control_arm.labeled,
            reason,
            raw_beta_cdf_calls: 0,
        }));
    }

    let statistics = PosteriorSufficientStatisticsV1::from_arms(
        treatment,
        control_arm,
        policy.noninferiority_margin,
    );
    let posterior = evaluate_posterior_with_control_v1(statistics, &mut control)?;
    let rollback_lower_value = 1.0 - posterior.noninferiority.upper();
    let rollback_lower = ActiveAuditF64V1::new(rollback_lower_value)?;
    let gates = reduce_gates_v1(
        policy,
        treatment,
        control_arm,
        &posterior,
        rollback_lower_value,
    )?;
    let state = reduce_predicates_v1(gates.rollback, gates.noninferiority);
    Ok(ActiveLookEvaluationV1::Completed(Box::new(
        ActiveLookAuditV1 {
            shape_version: ACTIVE_MATH_SHAPE_VERSION_V1,
            active_math_build_id: ACTIVE_MATH_BUILD_ID_V1.to_string(),
            active_math_algorithm_id_sha256: active_math_algorithm_identity_v1(),
            input_identity_sha256,
            as_of_unix_ms,
            policy: policy.audit()?,
            treatment,
            control: control_arm,
            noninferiority_g: posterior.noninferiority_g,
            noninferiority_h: posterior.noninferiority_h,
            noninferiority: posterior.noninferiority,
            rollback_lower,
            gates,
            state,
            raw_beta_cdf_calls: posterior.raw_beta_cdf_calls,
            operational_cdf_queries: posterior.operational_cdf_queries,
            max_symmetry_disagreement: posterior.max_symmetry_disagreement,
            max_monotonic_repair: posterior.max_monotonic_repair,
            max_quantile_width: posterior.max_quantile_width,
            cdf_transcript_sha256: posterior.cdf_transcript_sha256,
        },
    )))
}

pub(crate) fn active_look_input_identity_v1(
    policy: ActiveLookPolicyV1,
    as_of_unix_ms: i64,
    members: &[ActiveLookMemberV1],
) -> Result<String, ActiveMathErrorV1> {
    let member_count = u32::try_from(members.len()).map_err(|_| ActiveMathErrorV1::InvalidInput)?;
    let mut digest = Sha256::new();
    digest.update(ACTIVE_MATH_INPUT_DOMAIN_V1);
    digest.update(policy.actual_outcome_half_life_seconds.to_be_bytes());
    digest.update(policy.min_treatment_roots.to_be_bytes());
    digest.update(policy.min_control_roots.to_be_bytes());
    digest.update(
        policy
            .min_treatment_effective_weight
            .to_bits()
            .to_be_bytes(),
    );
    digest.update(policy.min_control_effective_weight.to_bits().to_be_bytes());
    digest.update(policy.noninferiority_margin.to_bits().to_be_bytes());
    digest.update(policy.noninferiority_probability.to_bits().to_be_bytes());
    digest.update(policy.rollback_probability.to_bits().to_be_bytes());
    digest.update(policy.max_looks.to_be_bytes());
    digest.update(
        policy
            .per_look_noninferiority_threshold
            .to_bits()
            .to_be_bytes(),
    );
    digest.update(as_of_unix_ms.to_be_bytes());
    digest.update(member_count.to_be_bytes());
    for member in members {
        digest.update(member.cap_ordinal.to_be_bytes());
        digest.update(member.admission_unix_ms.to_be_bytes());
        digest.update([match member.arm {
            ActiveOutcomeArmV1::Treatment => 0,
            ActiveOutcomeArmV1::Control => 1,
        }]);
        digest.update([match member.label {
            None => 0,
            Some(ActiveOutcomeLabelV1::Success) => 1,
            Some(ActiveOutcomeLabelV1::Failure) => 2,
        }]);
    }
    Ok(hex_digest(digest.finalize()))
}

pub(crate) fn active_math_algorithm_identity_v1() -> String {
    let mut digest = Sha256::new();
    digest.update(ACTIVE_MATH_ALGORITHM_DOMAIN_V1);
    for field in ACTIVE_MATH_ALGORITHM_FIELDS_V1 {
        let length = u32::try_from(field.len()).expect("fixed algorithm identity field fits u32");
        digest.update(length.to_be_bytes());
        digest.update(field.as_bytes());
    }
    hex_digest(digest.finalize())
}

/// Verify a persisted result against a replay under the exact build contract.
pub(crate) fn active_look_replay_matches_v1(
    persisted: &ActiveLookAuditV1,
    replayed: &ActiveLookAuditV1,
) -> bool {
    if persisted.active_math_build_id == replayed.active_math_build_id {
        return persisted == replayed;
    }
    persisted.shape_version == replayed.shape_version
        && persisted.active_math_algorithm_id_sha256 == replayed.active_math_algorithm_id_sha256
        && persisted.input_identity_sha256 == replayed.input_identity_sha256
        && persisted.state == replayed.state
        && endpoint_within_cross_platform_tolerance(
            persisted.noninferiority.lower(),
            replayed.noninferiority.lower(),
        )
        && endpoint_within_cross_platform_tolerance(
            persisted.noninferiority.upper(),
            replayed.noninferiority.upper(),
        )
}

fn endpoint_within_cross_platform_tolerance(left: f64, right: f64) -> bool {
    let difference = (left - right).abs();
    difference.is_finite() && difference <= ACTIVE_MATH_CROSS_PLATFORM_ENDPOINT_TOLERANCE_V1
}

/// Evaluate only the production posterior for an exact simulation-cache key.
pub(crate) fn evaluate_posterior_v1(
    statistics: PosteriorSufficientStatisticsV1,
) -> Result<PosteriorAuditV1, ActiveMathErrorV1> {
    evaluate_posterior_with_control_v1(statistics, &mut |_| true)
}

fn evaluate_posterior_with_control_v1<Control>(
    statistics: PosteriorSufficientStatisticsV1,
    control: &mut Control,
) -> Result<PosteriorAuditV1, ActiveMathErrorV1>
where
    Control: FnMut(usize) -> bool,
{
    let (treatment_alpha, treatment_beta, control_alpha, control_beta, margin) =
        statistics.shapes()?;
    let treatment = BetaShapeV1::new(treatment_alpha, treatment_beta)?;
    let control_shape = BetaShapeV1::new(control_alpha, control_beta)?;
    let mut audit = CdfAuditStateV1::new();
    let first = integrand_point_v1(0, &treatment, &control_shape, margin, &mut audit)?;
    let mut previous = first;
    let mut g_lower_sum = NeumaierSumV1::default();
    let mut g_upper_sum = NeumaierSumV1::default();
    let mut h_lower_sum = NeumaierSumV1::default();
    let mut h_upper_sum = NeumaierSumV1::default();

    for index in 1..=ACTIVE_MATH_GRID_PANELS_V1 {
        if !control(index) {
            return Err(ActiveMathErrorV1::Canceled);
        }
        let current = integrand_point_v1(index, &treatment, &control_shape, margin, &mut audit)?;
        if current.treatment_quantile.lower < previous.treatment_quantile.lower
            || current.treatment_quantile.upper < previous.treatment_quantile.upper
            || current.control_quantile.lower < previous.control_quantile.lower
            || current.control_quantile.upper < previous.control_quantile.upper
            || current.g.lower() < previous.g.lower()
            || current.g.upper() < previous.g.upper()
            || current.h.lower() > previous.h.lower()
            || current.h.upper() > previous.h.upper()
        {
            return Err(ActiveMathErrorV1::NonmonotoneCdf);
        }
        g_lower_sum.add(previous.g.lower())?;
        g_upper_sum.add(current.g.upper())?;
        h_lower_sum.add(current.h.lower())?;
        h_upper_sum.add(previous.h.upper())?;
        previous = current;
    }

    let divisor = ACTIVE_MATH_GRID_PANELS_V1 as f64;
    let g = ProbabilityIntervalAuditV1::new(
        g_lower_sum.total()? / divisor,
        g_upper_sum.total()? / divisor,
    )?;
    let h = ProbabilityIntervalAuditV1::new(
        h_lower_sum.total()? / divisor,
        h_upper_sum.total()? / divisor,
    )?;
    let lower = g.lower().max(h.lower());
    let upper = g.upper().min(h.upper());
    if lower > upper {
        return Err(ActiveMathErrorV1::DisjointOrientations);
    }
    let interval = ProbabilityIntervalAuditV1::new(lower, upper)?;
    let width = upper - lower;
    if !width.is_finite() || width > ACTIVE_MATH_POSTERIOR_RESOLUTION_V1 {
        return Err(ActiveMathErrorV1::PosteriorResolutionExceeded);
    }
    audit.finish(g, h, interval)
}

fn aggregate_arms_v1(
    half_life_seconds: u32,
    as_of_unix_ms: i64,
    members: &[ActiveLookMemberV1],
) -> Result<(ActiveArmAuditV1, ActiveArmAuditV1), ActiveMathErrorV1> {
    if as_of_unix_ms < 0 || members.is_empty() || members.len() > ACTIVE_MATH_ROOTS_MAX_V1 {
        return Err(ActiveMathErrorV1::InvalidInput);
    }
    let mut previous_ordinal = None;
    let mut treatment = ArmAccumulatorV1::default();
    let mut control = ArmAccumulatorV1::default();
    for member in members {
        if member.cap_ordinal == 0
            || previous_ordinal.is_some_and(|previous| member.cap_ordinal <= previous)
            || member.admission_unix_ms < 0
        {
            return Err(ActiveMathErrorV1::InvalidOrder);
        }
        previous_ordinal = Some(member.cap_ordinal);
        let accumulator = match member.arm {
            ActiveOutcomeArmV1::Treatment => &mut treatment,
            ActiveOutcomeArmV1::Control => &mut control,
        };
        accumulator.denominator = accumulator
            .denominator
            .checked_add(1)
            .ok_or(ActiveMathErrorV1::NumericFailure)?;
        if let Some(label) = member.label {
            let weight =
                decay_weight_v1(as_of_unix_ms, member.admission_unix_ms, half_life_seconds)?;
            accumulator.add(label, weight)?;
        }
    }
    Ok((treatment.finish()?, control.finish()?))
}

/// Pinned pure-software decay for one writer-owned admission time.
pub(crate) fn decay_weight_v1(
    as_of_unix_ms: i64,
    admission_unix_ms: i64,
    half_life_seconds: u32,
) -> Result<f64, ActiveMathErrorV1> {
    if as_of_unix_ms < 0
        || admission_unix_ms < 0
        || !(1..=ACTIVE_MATH_DURATION_SECONDS_MAX_V1).contains(&half_life_seconds)
    {
        return Err(ActiveMathErrorV1::InvalidInput);
    }
    let delta_ms = as_of_unix_ms
        .checked_sub(admission_unix_ms)
        .ok_or(ActiveMathErrorV1::FutureSkew)?;
    let nonnegative_ms = if delta_ms < 0 {
        if delta_ms < -ACTIVE_MATH_FUTURE_SKEW_MS_V1 {
            return Err(ActiveMathErrorV1::FutureSkew);
        }
        0_u64
    } else {
        u64::try_from(delta_ms).map_err(|_| ActiveMathErrorV1::FutureSkew)?
    };
    let age_seconds = nonnegative_ms as f64 / 1_000.0;
    let normalized_age = age_seconds / f64::from(half_life_seconds);
    let exponent = -normalized_age;
    let weight = libm::exp2(exponent);
    if !age_seconds.is_finite()
        || !normalized_age.is_finite()
        || !exponent.is_finite()
        || !weight.is_finite()
        || weight.is_sign_negative()
        || weight > 1.0
    {
        return Err(ActiveMathErrorV1::NumericFailure);
    }
    Ok(weight)
}

#[derive(Default)]
struct ArmAccumulatorV1 {
    denominator: u32,
    labeled: u32,
    successes: u32,
    failures: u32,
    success_weight: NeumaierSumV1,
    failure_weight: NeumaierSumV1,
}

impl ArmAccumulatorV1 {
    fn add(&mut self, label: ActiveOutcomeLabelV1, weight: f64) -> Result<(), ActiveMathErrorV1> {
        self.labeled = self
            .labeled
            .checked_add(1)
            .ok_or(ActiveMathErrorV1::NumericFailure)?;
        match label {
            ActiveOutcomeLabelV1::Success => {
                self.successes = self
                    .successes
                    .checked_add(1)
                    .ok_or(ActiveMathErrorV1::NumericFailure)?;
                self.success_weight.add(weight)
            }
            ActiveOutcomeLabelV1::Failure => {
                self.failures = self
                    .failures
                    .checked_add(1)
                    .ok_or(ActiveMathErrorV1::NumericFailure)?;
                self.failure_weight.add(weight)
            }
        }
    }

    fn finish(self) -> Result<ActiveArmAuditV1, ActiveMathErrorV1> {
        let success_weight = self.success_weight.total()?;
        let failure_weight = self.failure_weight.total()?;
        let effective_weight = success_weight + failure_weight;
        let beta_alpha = checked_shape(success_weight)?;
        let beta_beta = checked_shape(failure_weight)?;
        if self.labeled != self.successes + self.failures
            || self.labeled > self.denominator
            || !effective_weight.is_finite()
            || effective_weight.is_sign_negative()
        {
            return Err(ActiveMathErrorV1::NumericFailure);
        }
        let label_rate = if self.denominator == 0 {
            0.0
        } else {
            f64::from(self.labeled) / f64::from(self.denominator)
        };
        Ok(ActiveArmAuditV1 {
            denominator: self.denominator,
            labeled: self.labeled,
            successes: self.successes,
            failures: self.failures,
            success_weight: ActiveAuditF64V1::new(success_weight)?,
            failure_weight: ActiveAuditF64V1::new(failure_weight)?,
            effective_weight: ActiveAuditF64V1::new(effective_weight)?,
            beta_alpha: ActiveAuditF64V1::new(beta_alpha)?,
            beta_beta: ActiveAuditF64V1::new(beta_beta)?,
            label_rate: ActiveAuditF64V1::new(label_rate)?,
        })
    }
}

#[derive(Default)]
struct NeumaierSumV1 {
    sum: f64,
    correction: f64,
}

impl NeumaierSumV1 {
    fn add(&mut self, value: f64) -> Result<(), ActiveMathErrorV1> {
        if !value.is_finite() {
            return Err(ActiveMathErrorV1::Nonfinite);
        }
        let next = self.sum + value;
        if !next.is_finite() {
            return Err(ActiveMathErrorV1::NumericFailure);
        }
        let adjustment = if self.sum.abs() >= value.abs() {
            (self.sum - next) + value
        } else {
            (value - next) + self.sum
        };
        let correction = self.correction + adjustment;
        if !adjustment.is_finite() || !correction.is_finite() {
            return Err(ActiveMathErrorV1::NumericFailure);
        }
        self.sum = next;
        self.correction = correction;
        Ok(())
    }

    fn total(self) -> Result<f64, ActiveMathErrorV1> {
        let total = self.sum + self.correction;
        total
            .is_finite()
            .then_some(total)
            .ok_or(ActiveMathErrorV1::NumericFailure)
    }
}

struct BetaShapeV1 {
    direct: Beta,
    symmetric: Beta,
}

impl BetaShapeV1 {
    fn new(alpha: f64, beta: f64) -> Result<Self, ActiveMathErrorV1> {
        if !positive_finite(alpha)
            || !positive_finite(beta)
            || alpha > 65_537.0
            || beta > 65_537.0
            || alpha + beta > 65_538.0
        {
            return Err(ActiveMathErrorV1::InvalidShape);
        }
        let direct = Beta::new(alpha, beta).map_err(|_| ActiveMathErrorV1::InvalidShape)?;
        let symmetric = Beta::new(beta, alpha).map_err(|_| ActiveMathErrorV1::InvalidShape)?;
        Ok(Self { direct, symmetric })
    }
}

#[derive(Debug, Clone, Copy)]
struct CdfEnvelopeV1 {
    x: f64,
    direct: f64,
    symmetric: f64,
    midpoint: f64,
    lower: f64,
    upper: f64,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy)]
enum CdfRoleV1 {
    TreatmentBelow = 0x01,
    TreatmentAbove = 0x02,
    ControlBelow = 0x03,
    ControlAbove = 0x04,
    GLower = 0x11,
    GUpper = 0x12,
    HLower = 0x13,
    HUpper = 0x14,
}

#[derive(Debug, Clone, Copy)]
struct CdfSampleV1 {
    x: f64,
    lower: f64,
    upper: f64,
}

struct CdfAuditStateV1 {
    raw_calls: u32,
    operational_queries: u32,
    max_symmetry_disagreement: f64,
    max_monotonic_repair: f64,
    max_quantile_width: f64,
    transcript: Sha256,
}

impl CdfAuditStateV1 {
    fn new() -> Self {
        let mut transcript = Sha256::new();
        transcript.update(b"nemo-relay-router/active-cdf-transcript/v1\0");
        Self {
            raw_calls: 0,
            operational_queries: 0,
            max_symmetry_disagreement: 0.0,
            max_monotonic_repair: 0.0,
            max_quantile_width: 0.0,
            transcript,
        }
    }

    fn raw_cdf(&mut self, distribution: &Beta, x: f64) -> Result<f64, ActiveMathErrorV1> {
        self.raw_calls = self
            .raw_calls
            .checked_add(1)
            .ok_or(ActiveMathErrorV1::CdfBudgetExceeded)?;
        if self.raw_calls > ACTIVE_MATH_RAW_BETA_CDF_CALLS_MAX_V1 {
            return Err(ActiveMathErrorV1::CdfBudgetExceeded);
        }
        let value = catch_unwind(AssertUnwindSafe(|| distribution.cdf(x)))
            .map_err(|_| ActiveMathErrorV1::CdfFailure)?;
        value
            .is_finite()
            .then_some(value)
            .ok_or(ActiveMathErrorV1::CdfFailure)
    }

    fn envelope(
        &mut self,
        shape: &BetaShapeV1,
        x: f64,
    ) -> Result<CdfEnvelopeV1, ActiveMathErrorV1> {
        if !valid_probability(x) {
            return Err(ActiveMathErrorV1::InvalidInput);
        }
        let envelope = if x == 0.0 || x == 1.0 {
            CdfEnvelopeV1 {
                x,
                direct: x,
                symmetric: x,
                midpoint: x,
                lower: x,
                upper: x,
            }
        } else {
            let direct = self.raw_cdf(&shape.direct, x)?.clamp(0.0, 1.0);
            let reflected_x = 1.0 - x;
            let reflected = self.raw_cdf(&shape.symmetric, reflected_x)?;
            let symmetric = (1.0 - reflected).clamp(0.0, 1.0);
            let disagreement = (direct - symmetric).abs();
            if !disagreement.is_finite() {
                return Err(ActiveMathErrorV1::Nonfinite);
            }
            self.max_symmetry_disagreement = self.max_symmetry_disagreement.max(disagreement);
            let direct_lower = (direct - ACTIVE_MATH_CDF_GUARD_V1).max(0.0);
            let direct_upper = (direct + ACTIVE_MATH_CDF_GUARD_V1).min(1.0);
            let symmetric_lower = (symmetric - ACTIVE_MATH_CDF_GUARD_V1).max(0.0);
            let symmetric_upper = (symmetric + ACTIVE_MATH_CDF_GUARD_V1).min(1.0);
            let lower = direct_lower.max(symmetric_lower);
            let upper = direct_upper.min(symmetric_upper);
            let difference = symmetric - direct;
            let midpoint = direct + difference / 2.0;
            if !valid_probability(midpoint) || lower > upper || midpoint < lower || midpoint > upper
            {
                return Err(ActiveMathErrorV1::SymmetryDisagreement);
            }
            CdfEnvelopeV1 {
                x,
                direct,
                symmetric,
                midpoint,
                lower,
                upper,
            }
        };
        Ok(envelope)
    }

    fn record(
        &mut self,
        role: CdfRoleV1,
        index: usize,
        step: u8,
        predicate: u8,
        envelope: CdfEnvelopeV1,
    ) -> Result<(), ActiveMathErrorV1> {
        if !matches!(predicate, 0..=2) || (step != u8::MAX && step >= 64) {
            return Err(ActiveMathErrorV1::NumericFailure);
        }
        let index = u32::try_from(index).map_err(|_| ActiveMathErrorV1::NumericFailure)?;
        self.operational_queries = self
            .operational_queries
            .checked_add(1)
            .ok_or(ActiveMathErrorV1::NumericFailure)?;
        self.transcript.update([role as u8]);
        self.transcript.update(index.to_be_bytes());
        self.transcript.update([step]);
        self.transcript.update([predicate]);
        for value in [
            envelope.x,
            envelope.direct,
            envelope.symmetric,
            envelope.midpoint,
            envelope.lower,
            envelope.upper,
        ] {
            self.transcript.update(value.to_bits().to_be_bytes());
        }
        Ok(())
    }

    fn finish(
        self,
        g: ProbabilityIntervalAuditV1,
        h: ProbabilityIntervalAuditV1,
        interval: ProbabilityIntervalAuditV1,
    ) -> Result<PosteriorAuditV1, ActiveMathErrorV1> {
        debug_assert!(self.raw_calls <= ACTIVE_MATH_RAW_BETA_CDF_CALLS_MAX_V1);
        let cdf_transcript_sha256 = hex_digest(self.transcript.finalize());
        Ok(PosteriorAuditV1 {
            noninferiority_g: g,
            noninferiority_h: h,
            noninferiority: interval,
            raw_beta_cdf_calls: self.raw_calls,
            operational_cdf_queries: self.operational_queries,
            max_symmetry_disagreement: ActiveAuditF64V1::new(self.max_symmetry_disagreement)?,
            max_monotonic_repair: ActiveAuditF64V1::new(self.max_monotonic_repair)?,
            max_quantile_width: ActiveAuditF64V1::new(self.max_quantile_width)?,
            cdf_transcript_sha256,
        })
    }
}

#[derive(Debug, Clone, Copy)]
struct QuantileBoundsV1 {
    lower: f64,
    upper: f64,
}

fn quantile_bounds_v1(
    shape: &BetaShapeV1,
    probability: f64,
    below_role: CdfRoleV1,
    above_role: CdfRoleV1,
    index: usize,
    audit: &mut CdfAuditStateV1,
) -> Result<QuantileBoundsV1, ActiveMathErrorV1> {
    if !valid_probability(probability) {
        return Err(ActiveMathErrorV1::InvalidInput);
    }
    if probability == 0.0 || probability == 1.0 {
        let envelope = audit.envelope(shape, probability)?;
        audit.record(below_role, index, u8::MAX, 0, envelope)?;
        audit.record(above_role, index, u8::MAX, 0, envelope)?;
        return Ok(QuantileBoundsV1 {
            lower: probability,
            upper: probability,
        });
    }

    let mut samples = Vec::with_capacity(ACTIVE_MATH_QUANTILE_STEPS_PER_ENDPOINT_V1 * 2);
    let mut proven_below = 0.0;
    let mut not_proven_below = 1.0;
    for step in 0..ACTIVE_MATH_QUANTILE_STEPS_PER_ENDPOINT_V1 {
        let midpoint = binary64_midpoint(proven_below, not_proven_below);
        if midpoint == proven_below || midpoint == not_proven_below {
            break;
        }
        let envelope = audit.envelope(shape, midpoint)?;
        samples.push(CdfSampleV1 {
            x: midpoint,
            lower: envelope.lower,
            upper: envelope.upper,
        });
        let advances = envelope.upper < probability;
        audit.record(
            below_role,
            index,
            u8::try_from(step).map_err(|_| ActiveMathErrorV1::NumericFailure)?,
            if advances { 1 } else { 2 },
            envelope,
        )?;
        if advances {
            proven_below = midpoint;
        } else {
            not_proven_below = midpoint;
        }
    }

    let mut not_proven_above = 0.0;
    let mut proven_above = 1.0;
    for step in 0..ACTIVE_MATH_QUANTILE_STEPS_PER_ENDPOINT_V1 {
        let midpoint = binary64_midpoint(not_proven_above, proven_above);
        if midpoint == not_proven_above || midpoint == proven_above {
            break;
        }
        let envelope = audit.envelope(shape, midpoint)?;
        samples.push(CdfSampleV1 {
            x: midpoint,
            lower: envelope.lower,
            upper: envelope.upper,
        });
        let advances = envelope.lower > probability;
        audit.record(
            above_role,
            index,
            u8::try_from(step).map_err(|_| ActiveMathErrorV1::NumericFailure)?,
            if advances { 1 } else { 2 },
            envelope,
        )?;
        if advances {
            proven_above = midpoint;
        } else {
            not_proven_above = midpoint;
        }
    }
    let monotonic_repair = normalize_monotone_samples_v1(&mut samples)?;
    audit.max_monotonic_repair = audit.max_monotonic_repair.max(monotonic_repair);
    proven_below = samples
        .iter()
        .filter(|sample| sample.upper < probability)
        .map(|sample| sample.x)
        .max_by(f64::total_cmp)
        .unwrap_or(0.0);
    proven_above = samples
        .iter()
        .filter(|sample| sample.lower > probability)
        .map(|sample| sample.x)
        .min_by(f64::total_cmp)
        .unwrap_or(1.0);
    let width = proven_above - proven_below;
    if proven_below > proven_above
        || !width.is_finite()
        || width.is_sign_negative()
        || width > ACTIVE_MATH_POSTERIOR_RESOLUTION_V1
    {
        return Err(ActiveMathErrorV1::QuantileUnresolved);
    }
    audit.max_quantile_width = audit.max_quantile_width.max(width);
    Ok(QuantileBoundsV1 {
        lower: proven_below,
        upper: proven_above,
    })
}

fn normalize_monotone_samples_v1(samples: &mut [CdfSampleV1]) -> Result<f64, ActiveMathErrorV1> {
    samples.sort_by(|left, right| left.x.total_cmp(&right.x));
    for pair in samples.windows(2) {
        let [left, right] = pair else {
            unreachable!("windows of two always contain two elements")
        };
        if left.x == right.x
            && (left.lower.to_bits() != right.lower.to_bits()
                || left.upper.to_bits() != right.upper.to_bits())
        {
            return Err(ActiveMathErrorV1::NonmonotoneCdf);
        }
    }
    let mut maximum_repair = 0.0_f64;
    let mut next_lower = 1.0_f64;
    for sample in samples.iter_mut().rev() {
        if sample.lower > next_lower {
            maximum_repair = maximum_repair.max(sample.lower - next_lower);
            sample.lower = next_lower;
        }
        next_lower = sample.lower;
    }
    let mut previous_upper = 0.0_f64;
    for sample in samples.iter_mut() {
        if sample.upper < previous_upper {
            maximum_repair = maximum_repair.max(previous_upper - sample.upper);
            sample.upper = previous_upper;
        }
        previous_upper = sample.upper;
        if sample.lower > sample.upper {
            return Err(ActiveMathErrorV1::NonmonotoneCdf);
        }
    }
    if !maximum_repair.is_finite() || maximum_repair > ACTIVE_MATH_MONOTONIC_REPAIR_MAX_V1 {
        return Err(ActiveMathErrorV1::NonmonotoneCdf);
    }
    Ok(maximum_repair)
}

fn binary64_midpoint(lower: f64, upper: f64) -> f64 {
    let difference = upper - lower;
    lower + difference / 2.0
}

#[derive(Debug, Clone, Copy)]
struct IntegrandPointV1 {
    treatment_quantile: QuantileBoundsV1,
    control_quantile: QuantileBoundsV1,
    g: ProbabilityIntervalAuditV1,
    h: ProbabilityIntervalAuditV1,
}

fn integrand_point_v1(
    index: usize,
    treatment: &BetaShapeV1,
    control: &BetaShapeV1,
    margin: f64,
    audit: &mut CdfAuditStateV1,
) -> Result<IntegrandPointV1, ActiveMathErrorV1> {
    let probability = index as f64 / ACTIVE_MATH_GRID_PANELS_V1 as f64;
    let treatment_quantile = quantile_bounds_v1(
        treatment,
        probability,
        CdfRoleV1::TreatmentBelow,
        CdfRoleV1::TreatmentAbove,
        index,
        audit,
    )?;
    let g_lower_x = (treatment_quantile.lower + margin).min(1.0);
    let g_upper_x = (treatment_quantile.upper + margin).min(1.0);
    let g_lower_envelope = audit.envelope(control, g_lower_x)?;
    audit.record(CdfRoleV1::GLower, index, u8::MAX, 0, g_lower_envelope)?;
    let g_upper_envelope = audit.envelope(control, g_upper_x)?;
    audit.record(CdfRoleV1::GUpper, index, u8::MAX, 0, g_upper_envelope)?;
    let g = ProbabilityIntervalAuditV1::new(g_lower_envelope.lower, g_upper_envelope.upper)?;

    let control_quantile = quantile_bounds_v1(
        control,
        probability,
        CdfRoleV1::ControlBelow,
        CdfRoleV1::ControlAbove,
        index,
        audit,
    )?;
    let h_upper_x = (control_quantile.lower - margin).max(0.0);
    let h_lower_x = (control_quantile.upper - margin).max(0.0);
    let h_lower_envelope = audit.envelope(treatment, h_lower_x)?;
    audit.record(CdfRoleV1::HLower, index, u8::MAX, 0, h_lower_envelope)?;
    let h_upper_envelope = audit.envelope(treatment, h_upper_x)?;
    audit.record(CdfRoleV1::HUpper, index, u8::MAX, 0, h_upper_envelope)?;
    let h = ProbabilityIntervalAuditV1::new(
        1.0 - h_lower_envelope.upper,
        1.0 - h_upper_envelope.lower,
    )?;
    if !g_lower_envelope.midpoint.is_finite()
        || !g_upper_envelope.midpoint.is_finite()
        || !h_lower_envelope.midpoint.is_finite()
        || !h_upper_envelope.midpoint.is_finite()
    {
        return Err(ActiveMathErrorV1::Nonfinite);
    }
    Ok(IntegrandPointV1 {
        treatment_quantile,
        control_quantile,
        g,
        h,
    })
}

pub(crate) fn reduce_gates_v1(
    policy: ActiveLookPolicyV1,
    treatment: ActiveArmAuditV1,
    control: ActiveArmAuditV1,
    posterior: &PosteriorAuditV1,
    rollback_lower: f64,
) -> Result<ActiveLookGateAuditV1, ActiveMathErrorV1> {
    let treatment_attribution = attribution_rate_gate_v1(treatment.labeled, treatment.denominator)?;
    let control_attribution = attribution_rate_gate_v1(control.labeled, control.denominator)?;
    let differential_attribution = differential_attribution_gate_v1(
        treatment.labeled,
        treatment.denominator,
        control.labeled,
        control.denominator,
    )?;
    let treatment_effective_weight =
        treatment.effective_weight.value() >= policy.min_treatment_effective_weight;
    let control_effective_weight =
        control.effective_weight.value() >= policy.min_control_effective_weight;
    let pass_support = treatment_effective_weight
        && control_effective_weight
        && treatment_attribution
        && control_attribution
        && differential_attribution;
    Ok(ActiveLookGateAuditV1 {
        treatment_raw_roots: treatment.labeled >= policy.min_treatment_roots,
        control_raw_roots: control.labeled >= policy.min_control_roots,
        treatment_effective_weight,
        control_effective_weight,
        treatment_attribution,
        control_attribution,
        differential_attribution,
        noninferiority: pass_support
            && posterior.noninferiority.lower() >= policy.per_look_noninferiority_threshold,
        rollback: rollback_lower >= policy.rollback_probability,
    })
}

pub(crate) fn attribution_rate_gate_v1(
    labeled: u32,
    denominator: u32,
) -> Result<bool, ActiveMathErrorV1> {
    if denominator == 0 || labeled > denominator {
        return Err(ActiveMathErrorV1::InvalidInput);
    }
    let labeled_scaled = u128::from(labeled)
        .checked_mul(5)
        .ok_or(ActiveMathErrorV1::NumericFailure)?;
    let denominator_scaled = u128::from(denominator)
        .checked_mul(4)
        .ok_or(ActiveMathErrorV1::NumericFailure)?;
    Ok(labeled_scaled >= denominator_scaled)
}

pub(crate) fn differential_attribution_gate_v1(
    treatment_labeled: u32,
    treatment_denominator: u32,
    control_labeled: u32,
    control_denominator: u32,
) -> Result<bool, ActiveMathErrorV1> {
    if treatment_denominator == 0
        || control_denominator == 0
        || treatment_labeled > treatment_denominator
        || control_labeled > control_denominator
    {
        return Err(ActiveMathErrorV1::InvalidInput);
    }
    let treatment_cross = u128::from(treatment_labeled)
        .checked_mul(u128::from(control_denominator))
        .ok_or(ActiveMathErrorV1::NumericFailure)?;
    let control_cross = u128::from(control_labeled)
        .checked_mul(u128::from(treatment_denominator))
        .ok_or(ActiveMathErrorV1::NumericFailure)?;
    let difference = treatment_cross.abs_diff(control_cross);
    let denominator_product = u128::from(treatment_denominator)
        .checked_mul(u128::from(control_denominator))
        .ok_or(ActiveMathErrorV1::NumericFailure)?;
    let scaled_difference = difference
        .checked_mul(10)
        .ok_or(ActiveMathErrorV1::NumericFailure)?;
    Ok(scaled_difference <= denominator_product)
}

pub(crate) fn reduce_predicates_v1(rollback: bool, pass: bool) -> LookProducedStateV1 {
    if rollback {
        LookProducedStateV1::Rollback
    } else if pass {
        LookProducedStateV1::Passed
    } else {
        LookProducedStateV1::Collecting
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum FinalCapStateV1 {
    PendingDrain,
    PendingTargets,
    ClosedPassed,
    Exhausted,
}

pub(crate) fn reduce_final_cap_state_v1(
    unresolved_assignments: u32,
    all_targets_processed: bool,
    current_authorizing_pass: bool,
) -> FinalCapStateV1 {
    if unresolved_assignments > 0 {
        FinalCapStateV1::PendingDrain
    } else if !all_targets_processed {
        FinalCapStateV1::PendingTargets
    } else if current_authorizing_pass {
        FinalCapStateV1::ClosedPassed
    } else {
        FinalCapStateV1::Exhausted
    }
}

pub(crate) fn authorization_expired_v1(
    passed_as_of_unix_ms: i64,
    authorization_ttl_seconds: u32,
    observed_at_unix_ms: i64,
) -> Result<bool, ActiveMathErrorV1> {
    if passed_as_of_unix_ms < 0
        || observed_at_unix_ms < 0
        || !(1..=ACTIVE_MATH_DURATION_SECONDS_MAX_V1).contains(&authorization_ttl_seconds)
    {
        return Err(ActiveMathErrorV1::InvalidInput);
    }
    let ttl_ms = i64::from(authorization_ttl_seconds)
        .checked_mul(1_000)
        .ok_or(ActiveMathErrorV1::NumericFailure)?;
    let expires_at = passed_as_of_unix_ms
        .checked_add(ttl_ms)
        .ok_or(ActiveMathErrorV1::NumericFailure)?;
    Ok(observed_at_unix_ms >= expires_at)
}

fn checked_shape(weight: f64) -> Result<f64, ActiveMathErrorV1> {
    if !weight.is_finite() || weight.is_sign_negative() || weight > 65_536.0 {
        return Err(ActiveMathErrorV1::InvalidShape);
    }
    let shape = 1.0 + weight;
    (shape.is_finite() && shape > 0.0 && shape <= 65_537.0)
        .then_some(shape)
        .ok_or(ActiveMathErrorV1::InvalidShape)
}

fn finite_nonnegative(bits: u64) -> Result<f64, ActiveMathErrorV1> {
    let value = f64::from_bits(bits);
    if !value.is_finite() || value.is_sign_negative() || value > 65_536.0 {
        return Err(ActiveMathErrorV1::InvalidInput);
    }
    Ok(value)
}

fn positive_finite(value: f64) -> bool {
    value.is_finite() && value > 0.0
}

fn valid_probability(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value) && !value.is_sign_negative()
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
pub(crate) fn active_cdf_envelope_probe_v1(
    alpha: f64,
    beta: f64,
    x: f64,
) -> Result<([u64; 6], u32), ActiveMathErrorV1> {
    let shape = BetaShapeV1::new(alpha, beta)?;
    let mut audit = CdfAuditStateV1::new();
    let envelope = audit.envelope(&shape, x)?;
    Ok((
        [
            envelope.x.to_bits(),
            envelope.direct.to_bits(),
            envelope.symmetric.to_bits(),
            envelope.midpoint.to_bits(),
            envelope.lower.to_bits(),
            envelope.upper.to_bits(),
        ],
        audit.raw_calls,
    ))
}

#[cfg(test)]
pub(crate) struct ActiveQuantileProbeV1 {
    pub(crate) lower_bits: u64,
    pub(crate) upper_bits: u64,
    pub(crate) raw_calls: u32,
    pub(crate) operational_queries: u32,
    pub(crate) max_monotonic_repair_bits: u64,
    pub(crate) max_quantile_width_bits: u64,
    pub(crate) transcript_sha256: String,
}

#[cfg(test)]
pub(crate) fn active_quantile_probe_v1(
    alpha: f64,
    beta: f64,
    probability: f64,
) -> Result<ActiveQuantileProbeV1, ActiveMathErrorV1> {
    let shape = BetaShapeV1::new(alpha, beta)?;
    let mut audit = CdfAuditStateV1::new();
    let bounds = quantile_bounds_v1(
        &shape,
        probability,
        CdfRoleV1::TreatmentBelow,
        CdfRoleV1::TreatmentAbove,
        0,
        &mut audit,
    )?;
    Ok(ActiveQuantileProbeV1 {
        lower_bits: bounds.lower.to_bits(),
        upper_bits: bounds.upper.to_bits(),
        raw_calls: audit.raw_calls,
        operational_queries: audit.operational_queries,
        max_monotonic_repair_bits: audit.max_monotonic_repair.to_bits(),
        max_quantile_width_bits: audit.max_quantile_width.to_bits(),
        transcript_sha256: hex_digest(audit.transcript.finalize()),
    })
}

const _: () = assert!(std::mem::size_of::<ActiveLookAuditV1>() < ACTIVE_MATH_MEMORY_BYTES_MAX_V1);

#[cfg(test)]
mod internal_tests {
    use super::*;

    #[test]
    fn predicate_precedence_is_rollback_then_pass_then_collecting() {
        assert_eq!(
            reduce_predicates_v1(true, true),
            LookProducedStateV1::Rollback
        );
        assert_eq!(
            reduce_predicates_v1(false, true),
            LookProducedStateV1::Passed
        );
        assert_eq!(
            reduce_predicates_v1(false, false),
            LookProducedStateV1::Collecting
        );
    }
}
