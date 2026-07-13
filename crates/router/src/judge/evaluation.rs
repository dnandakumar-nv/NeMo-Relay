// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pure pairwise-judge evaluation math and durable score representation.

use serde::{Deserialize, Deserializer, Serialize};
use unicode_normalization::UnicodeNormalization;

use super::{
    contains_sensitive_free_text,
    output::{JudgeHardFailureV1, PairwiseJudgeResultV1},
};
use crate::config::JudgeConfig;

/// Stable origin of one quality evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum JudgeEvaluationSourceV1 {
    DeterministicValidator,
    Judge,
}

/// Hard failures that the deterministic candidate validator may produce.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DeterministicHardFailureV1 {
    ToolContract,
    ResponseSchema,
    MalformedCandidate,
}

impl From<DeterministicHardFailureV1> for JudgeHardFailureV1 {
    fn from(value: DeterministicHardFailureV1) -> Self {
        match value {
            DeterministicHardFailureV1::ToolContract => Self::ToolContract,
            DeterministicHardFailureV1::ResponseSchema => Self::ResponseSchema,
            DeterministicHardFailureV1::MalformedCandidate => Self::MalformedCandidate,
        }
    }
}

/// Complete quality label, including the non-binary ambiguous state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum JudgeLabelV1 {
    Pass,
    Fail,
    Ambiguous,
}

impl JudgeLabelV1 {
    pub(crate) const fn binary_label(self) -> Option<JudgeBinaryLabelV1> {
        match self {
            Self::Pass => Some(JudgeBinaryLabelV1::Pass),
            Self::Fail => Some(JudgeBinaryLabelV1::Fail),
            Self::Ambiguous => None,
        }
    }
}

/// Binary label retained only for pass and fail evaluations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum JudgeBinaryLabelV1 {
    Pass,
    Fail,
}

/// One round-trippable floating-point value and its exact IEEE-754 bits.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct ScoredValueV1 {
    pub(crate) value: f64,
    pub(crate) bits: u64,
}

impl<'de> Deserialize<'de> for ScoredValueV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct WireValue {
            value: f64,
            bits: u64,
        }

        let value = WireValue::deserialize(deserializer)?;
        if value.value.to_bits() != value.bits {
            return Err(serde::de::Error::custom(
                "floating-point value does not match its persisted bits",
            ));
        }
        Ok(Self {
            value: value.value,
            bits: value.bits,
        })
    }
}

impl ScoredValueV1 {
    pub(crate) fn new(value: f64) -> Self {
        Self {
            value,
            bits: value.to_bits(),
        }
    }

    pub(crate) fn has_matching_bits(self) -> bool {
        self.value.to_bits() == self.bits
    }
}

/// Durable evaluation produced only when a quality label exists.
///
/// Invalid judge output and operational attempts deliberately produce no value
/// of this type.
#[derive(Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct JudgeEvaluationV1 {
    pub(crate) source: JudgeEvaluationSourceV1,
    pub(crate) response_equivalence: Option<ScoredValueV1>,
    pub(crate) trajectory_equivalence: Option<ScoredValueV1>,
    pub(crate) judge_confidence: Option<ScoredValueV1>,
    pub(crate) response_weight: Option<ScoredValueV1>,
    pub(crate) trajectory_weight: Option<ScoredValueV1>,
    pub(crate) aggregate: Option<ScoredValueV1>,
    pub(crate) label: JudgeLabelV1,
    pub(crate) binary_label: Option<JudgeBinaryLabelV1>,
    pub(crate) hard_failures: Vec<JudgeHardFailureV1>,
    pub(crate) rationale: Option<String>,
    pub(crate) is_partial: bool,
    pub(crate) promotion_eligible: bool,
}

impl JudgeEvaluationV1 {
    pub(crate) fn from_judge_result(
        config: &JudgeConfig,
        result: &PairwiseJudgeResultV1,
        is_partial: bool,
    ) -> Self {
        let response_equivalence = result.response_equivalence();
        let trajectory_equivalence = result.trajectory_equivalence();
        let judge_confidence = result.judge_confidence();
        let hard_failures = result.hard_failures();
        let rationale = result.rationale();
        let aggregate = exact_weighted_aggregate(
            config.response_weight,
            response_equivalence,
            config.trajectory_weight,
            trajectory_equivalence,
        );
        let label = classify(
            config,
            response_equivalence,
            trajectory_equivalence,
            judge_confidence,
            aggregate,
            hard_failures,
        );
        let binary_label = label.binary_label();

        Self {
            source: JudgeEvaluationSourceV1::Judge,
            response_equivalence: Some(ScoredValueV1::new(response_equivalence)),
            trajectory_equivalence: Some(ScoredValueV1::new(trajectory_equivalence)),
            judge_confidence: Some(ScoredValueV1::new(judge_confidence)),
            response_weight: Some(ScoredValueV1::new(config.response_weight)),
            trajectory_weight: Some(ScoredValueV1::new(config.trajectory_weight)),
            aggregate: Some(ScoredValueV1::new(aggregate)),
            label,
            binary_label,
            hard_failures: hard_failures.to_vec(),
            rationale: Some(rationale.to_string()),
            is_partial,
            promotion_eligible: !is_partial && binary_label.is_some(),
        }
    }

    pub(crate) fn deterministic_hard_failure(
        hard_failure: DeterministicHardFailureV1,
        is_partial: bool,
    ) -> Self {
        Self {
            source: JudgeEvaluationSourceV1::DeterministicValidator,
            response_equivalence: None,
            trajectory_equivalence: None,
            judge_confidence: None,
            response_weight: None,
            trajectory_weight: None,
            aggregate: None,
            label: JudgeLabelV1::Fail,
            binary_label: Some(JudgeBinaryLabelV1::Fail),
            hard_failures: vec![hard_failure.into()],
            rationale: None,
            is_partial,
            promotion_eligible: !is_partial,
        }
    }

    /// Recomputes the aggregate from the persisted score and weight values.
    pub(crate) fn recalculated_aggregate(&self) -> Option<ScoredValueV1> {
        let response_equivalence = self.response_equivalence?;
        let trajectory_equivalence = self.trajectory_equivalence?;
        let response_weight = self.response_weight?;
        let trajectory_weight = self.trajectory_weight?;
        Some(ScoredValueV1::new(exact_weighted_aggregate(
            response_weight.value,
            response_equivalence.value,
            trajectory_weight.value,
            trajectory_equivalence.value,
        )))
    }

    /// Verifies the persisted numeric, classification, and eligibility facts.
    pub(crate) fn is_consistent_with_config(&self, config: &JudgeConfig) -> bool {
        let scored_values = [
            self.response_equivalence,
            self.trajectory_equivalence,
            self.judge_confidence,
            self.response_weight,
            self.trajectory_weight,
            self.aggregate,
        ];
        if scored_values
            .into_iter()
            .flatten()
            .any(|value| !value.has_matching_bits())
            || !hard_failures_are_canonical(&self.hard_failures)
            || self.binary_label != self.label.binary_label()
            || self.promotion_eligible != (!self.is_partial && self.binary_label.is_some())
        {
            return false;
        }

        match self.source {
            JudgeEvaluationSourceV1::DeterministicValidator => {
                self.response_equivalence.is_none()
                    && self.trajectory_equivalence.is_none()
                    && self.judge_confidence.is_none()
                    && self.response_weight.is_none()
                    && self.trajectory_weight.is_none()
                    && self.aggregate.is_none()
                    && self.rationale.is_none()
                    && self.label == JudgeLabelV1::Fail
                    && self.hard_failures.len() == 1
                    && self.hard_failures.iter().all(is_deterministic_hard_failure)
            }
            JudgeEvaluationSourceV1::Judge => {
                let (
                    Some(response_equivalence),
                    Some(trajectory_equivalence),
                    Some(judge_confidence),
                    Some(response_weight),
                    Some(trajectory_weight),
                    Some(aggregate),
                    Some(rationale),
                ) = (
                    self.response_equivalence,
                    self.trajectory_equivalence,
                    self.judge_confidence,
                    self.response_weight,
                    self.trajectory_weight,
                    self.aggregate,
                    self.rationale.as_ref(),
                )
                else {
                    return false;
                };
                if ![
                    response_equivalence.value,
                    trajectory_equivalence.value,
                    judge_confidence.value,
                    response_weight.value,
                    trajectory_weight.value,
                ]
                .into_iter()
                .all(is_unit_interval)
                    || !aggregate.value.is_finite()
                    || response_weight.bits != config.response_weight.to_bits()
                    || trajectory_weight.bits != config.trajectory_weight.to_bits()
                    || !rationale_is_valid(rationale, config.max_rationale_bytes)
                {
                    return false;
                }
                let recalculated = exact_weighted_aggregate(
                    response_weight.value,
                    response_equivalence.value,
                    trajectory_weight.value,
                    trajectory_equivalence.value,
                );
                aggregate.bits == recalculated.to_bits()
                    && self.label
                        == classify(
                            config,
                            response_equivalence.value,
                            trajectory_equivalence.value,
                            judge_confidence.value,
                            recalculated,
                            &self.hard_failures,
                        )
            }
        }
    }
}

fn classify(
    config: &JudgeConfig,
    response_equivalence: f64,
    trajectory_equivalence: f64,
    judge_confidence: f64,
    aggregate: f64,
    hard_failures: &[JudgeHardFailureV1],
) -> JudgeLabelV1 {
    if !hard_failures.is_empty() {
        JudgeLabelV1::Fail
    } else if judge_confidence < config.judge_confidence_floor {
        JudgeLabelV1::Ambiguous
    } else if response_equivalence >= config.response_floor
        && trajectory_equivalence >= config.trajectory_floor
        && aggregate >= config.pass_threshold
    {
        JudgeLabelV1::Pass
    } else {
        JudgeLabelV1::Fail
    }
}

fn is_deterministic_hard_failure(failure: &JudgeHardFailureV1) -> bool {
    matches!(
        failure,
        JudgeHardFailureV1::ToolContract
            | JudgeHardFailureV1::ResponseSchema
            | JudgeHardFailureV1::MalformedCandidate
    )
}

fn hard_failures_are_canonical(failures: &[JudgeHardFailureV1]) -> bool {
    failures
        .windows(2)
        .all(|pair| hard_failure_rank(&pair[0]) < hard_failure_rank(&pair[1]))
}

fn hard_failure_rank(failure: &JudgeHardFailureV1) -> u8 {
    match failure {
        JudgeHardFailureV1::ToolContract => 0,
        JudgeHardFailureV1::ResponseSchema => 1,
        JudgeHardFailureV1::Safety => 2,
        JudgeHardFailureV1::MalformedCandidate => 3,
    }
}

fn is_unit_interval(value: f64) -> bool {
    value.is_finite() && (0.0..=1.0).contains(&value)
}

fn rationale_is_valid(value: &str, max_bytes: usize) -> bool {
    !value.is_empty()
        && value.len() <= max_bytes
        && value.nfc().eq(value.chars())
        && !value.chars().any(char::is_control)
        && !contains_sensitive_free_text(value)
}

fn exact_weighted_aggregate(
    response_weight: f64,
    response_score: f64,
    trajectory_weight: f64,
    trajectory_score: f64,
) -> f64 {
    let response_component = response_weight * response_score;
    let trajectory_component = trajectory_weight * trajectory_score;
    response_component + trajectory_component
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::{
        DeterministicHardFailureV1, JudgeBinaryLabelV1, JudgeEvaluationSourceV1, JudgeEvaluationV1,
        JudgeLabelV1, ScoredValueV1, exact_weighted_aggregate,
    };
    use crate::config::{JUDGE_PROMPT_VERSION_V1, JUDGE_RUBRIC_VERSION_V1, JudgeConfig};
    use crate::judge::output::{
        JudgeAttemptOutcomeV1, JudgeHardFailureV1, PairwiseJudgeResultV1,
        validate_pairwise_judge_output,
    };

    fn config() -> JudgeConfig {
        JudgeConfig {
            version: 1,
            model: "judge-model".to_string(),
            model_revision: "judge-revision".to_string(),
            prompt_version: JUDGE_PROMPT_VERSION_V1.to_string(),
            rubric_version: JUDGE_RUBRIC_VERSION_V1.to_string(),
            output_schema_version: 1,
            temperature: None,
            response_weight: 0.6,
            trajectory_weight: 0.4,
            response_floor: 0.8,
            trajectory_floor: 0.75,
            judge_confidence_floor: 0.7,
            pass_threshold: 0.8,
            max_rationale_bytes: 4_096,
            base_cooloff_seconds: 10,
            max_cooloff_seconds: 300,
            unknown_fields: BTreeMap::new(),
        }
    }

    fn result(
        response_equivalence: f64,
        trajectory_equivalence: f64,
        judge_confidence: f64,
        hard_failures: Vec<JudgeHardFailureV1>,
    ) -> PairwiseJudgeResultV1 {
        let raw = serde_json::json!({
            "response_equivalence": response_equivalence,
            "trajectory_equivalence": trajectory_equivalence,
            "judge_confidence": judge_confidence,
            "hard_failures": hard_failures,
            "rationale": "bounded rationale",
        })
        .to_string();
        match validate_pairwise_judge_output(&raw, 4_096) {
            JudgeAttemptOutcomeV1::Valid(result) => result,
            JudgeAttemptOutcomeV1::Invalid(_) | JudgeAttemptOutcomeV1::Operational(_) => {
                panic!("evaluation fixture should be valid")
            }
        }
    }

    #[test]
    fn evaluation_precedence_and_numeric_gates_are_table_driven() {
        struct Case {
            name: &'static str,
            result: PairwiseJudgeResultV1,
            expected: JudgeLabelV1,
        }

        let below_confidence = f64::from_bits(0.7_f64.to_bits() - 1);
        let below_response_floor = f64::from_bits(0.8_f64.to_bits() - 1);
        let below_trajectory_floor = f64::from_bits(0.75_f64.to_bits() - 1);
        let cases = [
            Case {
                name: "pass",
                result: result(0.9, 0.9, 0.7, Vec::new()),
                expected: JudgeLabelV1::Pass,
            },
            Case {
                name: "hard failure precedes confidence",
                result: result(1.0, 1.0, below_confidence, vec![JudgeHardFailureV1::Safety]),
                expected: JudgeLabelV1::Fail,
            },
            Case {
                name: "confidence precedes numeric gates",
                result: result(0.0, 0.0, below_confidence, Vec::new()),
                expected: JudgeLabelV1::Ambiguous,
            },
            Case {
                name: "response component floor",
                result: result(below_response_floor, 1.0, 1.0, Vec::new()),
                expected: JudgeLabelV1::Fail,
            },
            Case {
                name: "trajectory component floor",
                result: result(1.0, below_trajectory_floor, 1.0, Vec::new()),
                expected: JudgeLabelV1::Fail,
            },
            Case {
                name: "aggregate threshold",
                result: result(0.8, 0.75, 1.0, Vec::new()),
                expected: JudgeLabelV1::Fail,
            },
        ];

        for case in cases {
            let evaluation = JudgeEvaluationV1::from_judge_result(&config(), &case.result, false);
            assert_eq!(evaluation.label, case.expected, "{}", case.name);
            assert_eq!(
                evaluation.binary_label,
                case.expected.binary_label(),
                "{}",
                case.name
            );
            assert_eq!(
                evaluation.promotion_eligible,
                case.expected != JudgeLabelV1::Ambiguous,
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn exact_boundaries_pass_and_one_ulp_below_or_above_fails() {
        let mut config = config();
        let exact = result(
            config.response_floor,
            config.trajectory_floor,
            config.judge_confidence_floor,
            Vec::new(),
        );
        let exact_aggregate = exact_weighted_aggregate(
            config.response_weight,
            exact.response_equivalence(),
            config.trajectory_weight,
            exact.trajectory_equivalence(),
        );
        config.pass_threshold = exact_aggregate;
        assert_eq!(
            JudgeEvaluationV1::from_judge_result(&config, &exact, false).label,
            JudgeLabelV1::Pass
        );

        let below_response = result(
            f64::from_bits(exact.response_equivalence().to_bits() - 1),
            exact.trajectory_equivalence(),
            exact.judge_confidence(),
            Vec::new(),
        );
        assert_eq!(
            JudgeEvaluationV1::from_judge_result(&config, &below_response, false).label,
            JudgeLabelV1::Fail
        );

        let below_confidence = result(
            exact.response_equivalence(),
            exact.trajectory_equivalence(),
            f64::from_bits(exact.judge_confidence().to_bits() - 1),
            Vec::new(),
        );
        assert_eq!(
            JudgeEvaluationV1::from_judge_result(&config, &below_confidence, false).label,
            JudgeLabelV1::Ambiguous
        );

        config.pass_threshold = f64::from_bits(exact_aggregate.to_bits() + 1);
        assert_eq!(
            JudgeEvaluationV1::from_judge_result(&config, &exact, false).label,
            JudgeLabelV1::Fail
        );
    }

    #[test]
    fn aggregate_uses_two_multiplies_then_add_and_recalculates_exact_bits() {
        let mut config = config();
        config.response_weight = 0.345_000_515_994_419_3;
        config.trajectory_weight = 0.654_999_484_005_580_7;
        config.response_floor = 0.0;
        config.trajectory_floor = 0.0;
        config.judge_confidence_floor = 0.0;
        config.pass_threshold = 0.0;
        let result = result(
            0.752_709_198_581_346_9,
            0.795_745_269_919_544,
            1.0,
            Vec::new(),
        );

        let evaluation = JudgeEvaluationV1::from_judge_result(&config, &result, false);
        let aggregate = evaluation.aggregate.unwrap();
        assert_eq!(aggregate.bits, 0x3fe8_fd1d_63ba_da68);
        assert!(aggregate.has_matching_bits());
        assert_eq!(evaluation.recalculated_aggregate(), Some(aggregate));
        assert!(evaluation.is_consistent_with_config(&config));

        let encoded = serde_json::to_string(&evaluation).unwrap();
        let decoded: JudgeEvaluationV1 = serde_json::from_str(&encoded).unwrap();
        assert!(decoded == evaluation);
        assert_eq!(decoded.aggregate.unwrap().bits, aggregate.bits);

        let mut inconsistent = evaluation.clone();
        inconsistent.aggregate.as_mut().unwrap().bits ^= 1;
        assert!(!inconsistent.is_consistent_with_config(&config));
    }

    #[test]
    fn tolerance_bound_weights_can_produce_a_consistent_aggregate_above_one() {
        let mut config = config();
        config.response_weight = 0.5;
        config.trajectory_weight = 0.500_000_000_9;
        config.response_floor = 0.0;
        config.trajectory_floor = 0.0;
        config.judge_confidence_floor = 0.0;
        config.pass_threshold = 1.0;
        assert!((config.response_weight + config.trajectory_weight - 1.0).abs() <= 1e-9);

        let evaluation = JudgeEvaluationV1::from_judge_result(
            &config,
            &result(1.0, 1.0, 1.0, Vec::new()),
            false,
        );
        assert!(evaluation.aggregate.unwrap().value > 1.0);
        assert!(evaluation.is_consistent_with_config(&config));
    }

    #[test]
    fn binary_and_promotion_labels_exclude_ambiguous_and_partial_results() {
        let pass = JudgeEvaluationV1::from_judge_result(
            &config(),
            &result(1.0, 1.0, 1.0, Vec::new()),
            false,
        );
        assert_eq!(pass.binary_label, Some(JudgeBinaryLabelV1::Pass));
        assert!(pass.promotion_eligible);

        let partial_pass = JudgeEvaluationV1::from_judge_result(
            &config(),
            &result(1.0, 1.0, 1.0, Vec::new()),
            true,
        );
        assert_eq!(partial_pass.binary_label, Some(JudgeBinaryLabelV1::Pass));
        assert!(!partial_pass.promotion_eligible);

        let ambiguous = JudgeEvaluationV1::from_judge_result(
            &config(),
            &result(1.0, 1.0, 0.0, Vec::new()),
            false,
        );
        assert_eq!(ambiguous.binary_label, None);
        assert!(!ambiguous.promotion_eligible);

        assert_eq!(
            serde_json::to_string(&JudgeEvaluationSourceV1::DeterministicValidator).unwrap(),
            "\"deterministic_validator\""
        );
        assert_eq!(
            serde_json::to_string(&JudgeLabelV1::Ambiguous).unwrap(),
            "\"ambiguous\""
        );
    }

    #[test]
    fn deterministic_failure_has_no_synthetic_scores() {
        let full = JudgeEvaluationV1::deterministic_hard_failure(
            DeterministicHardFailureV1::MalformedCandidate,
            false,
        );
        assert_eq!(full.source, JudgeEvaluationSourceV1::DeterministicValidator);
        assert_eq!(full.label, JudgeLabelV1::Fail);
        assert_eq!(full.binary_label, Some(JudgeBinaryLabelV1::Fail));
        assert!(full.promotion_eligible);
        assert!(full.response_equivalence.is_none());
        assert!(full.trajectory_equivalence.is_none());
        assert!(full.judge_confidence.is_none());
        assert!(full.response_weight.is_none());
        assert!(full.trajectory_weight.is_none());
        assert!(full.aggregate.is_none());
        assert!(full.rationale.is_none());
        assert_eq!(full.recalculated_aggregate(), None);
        assert!(full.is_consistent_with_config(&config()));

        let partial = JudgeEvaluationV1::deterministic_hard_failure(
            DeterministicHardFailureV1::MalformedCandidate,
            true,
        );
        assert!(!partial.promotion_eligible);

        assert!(
            serde_json::from_str::<DeterministicHardFailureV1>("\"safety\"").is_err(),
            "safety is judge-owned and must not be representable as deterministic"
        );

        let score = ScoredValueV1::new(-0.0);
        let encoded = serde_json::to_string(&score).unwrap();
        let decoded: ScoredValueV1 = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded.bits, (-0.0_f64).to_bits());
        assert!(decoded.has_matching_bits());
        assert!(serde_json::from_str::<ScoredValueV1>(r#"{"value":0.5,"bits":0}"#).is_err());
    }

    #[test]
    fn consistency_rejects_invalid_numbers_and_noncanonical_hard_failures() {
        let config = config();
        let evaluation = JudgeEvaluationV1::from_judge_result(
            &config,
            &result(1.0, 1.0, 1.0, Vec::new()),
            false,
        );
        assert!(evaluation.is_consistent_with_config(&config));

        for invalid in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
            let mut corrupted = evaluation.clone();
            corrupted.response_equivalence = Some(ScoredValueV1::new(invalid));
            assert!(!corrupted.is_consistent_with_config(&config));
        }

        let mut duplicate = evaluation.clone();
        duplicate.hard_failures = vec![
            JudgeHardFailureV1::ToolContract,
            JudgeHardFailureV1::ToolContract,
        ];
        duplicate.label = JudgeLabelV1::Fail;
        duplicate.binary_label = Some(JudgeBinaryLabelV1::Fail);
        assert!(!duplicate.is_consistent_with_config(&config));

        let mut out_of_order = evaluation;
        out_of_order.hard_failures = vec![
            JudgeHardFailureV1::Safety,
            JudgeHardFailureV1::ResponseSchema,
        ];
        out_of_order.label = JudgeLabelV1::Fail;
        out_of_order.binary_label = Some(JudgeBinaryLabelV1::Fail);
        assert!(!out_of_order.is_consistent_with_config(&config));
    }

    #[test]
    fn consistency_rejects_invalid_durable_rationales() {
        let config = config();
        let evaluation = JudgeEvaluationV1::from_judge_result(
            &config,
            &result(1.0, 1.0, 1.0, Vec::new()),
            false,
        );

        for invalid in [
            String::new(),
            "x".repeat(config.max_rationale_bytes + 1),
            "e\u{301}".to_string(),
            "line\nbreak".to_string(),
            "Bearer abcDEF0123456789xyz-_".to_string(),
        ] {
            let mut corrupted = evaluation.clone();
            corrupted.rationale = Some(invalid);
            assert!(!corrupted.is_consistent_with_config(&config));
        }

        let mut exact_utf8 = evaluation.clone();
        exact_utf8.rationale = Some("éé".to_string());
        let mut exact_config = config.clone();
        exact_config.max_rationale_bytes = 4;
        assert!(exact_utf8.is_consistent_with_config(&exact_config));
        exact_config.max_rationale_bytes = 3;
        assert!(!exact_utf8.is_consistent_with_config(&exact_config));

        let mut whitespace = evaluation;
        whitespace.rationale = Some("   ".to_string());
        assert!(whitespace.is_consistent_with_config(&config));
    }
}
