// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::active_math::*;
use serde::Deserialize;
use serde::de::{MapAccess, SeqAccess, Visitor};
use sha2::{Digest, Sha256};

fn valid_policy() -> ActiveLookPolicyV1 {
    ActiveLookPolicyV1::new(3_600, 32, 32, 32.0, 32.0, 0.1, 0.99, 0.95, 2)
        .expect("the normative fixed-simulation policy is valid")
}

fn skipped_audit(evaluation: ActiveLookEvaluationV1) -> SkippedLookAuditV1 {
    match evaluation {
        ActiveLookEvaluationV1::Skipped(audit) => audit,
        ActiveLookEvaluationV1::Completed(_) => panic!("expected raw-root short circuit"),
    }
}

fn balanced_members(count_per_arm: u64) -> Vec<ActiveLookMemberV1> {
    let mut members = Vec::with_capacity((count_per_arm * 2) as usize);
    for index in 0..count_per_arm {
        members.push(ActiveLookMemberV1 {
            cap_ordinal: index + 1,
            admission_unix_ms: 1_000_000,
            arm: ActiveOutcomeArmV1::Treatment,
            label: Some(if index % 2 == 0 {
                ActiveOutcomeLabelV1::Success
            } else {
                ActiveOutcomeLabelV1::Failure
            }),
        });
    }
    for index in 0..count_per_arm {
        members.push(ActiveLookMemberV1 {
            cap_ordinal: count_per_arm + index + 1,
            admission_unix_ms: 1_000_000,
            arm: ActiveOutcomeArmV1::Control,
            label: Some(if index % 2 == 0 {
                ActiveOutcomeLabelV1::Success
            } else {
                ActiveOutcomeLabelV1::Failure
            }),
        });
    }
    members
}

fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn hex_bits(bits: u64) -> String {
    format!("{bits:016x}")
}

fn interval_distance(value: f64, lower: f64, upper: f64) -> f64 {
    if value < lower {
        lower - value
    } else if value > upper {
        value - upper
    } else {
        0.0
    }
}

fn fixture_cases_by_id(
    fixture: &serde_json::Value,
    family: &str,
) -> std::collections::BTreeMap<String, serde_json::Value> {
    fixture[family]
        .as_array()
        .expect("fixture family must be an array")
        .iter()
        .map(|case| {
            (
                case["id"]
                    .as_str()
                    .expect("case id must be a string")
                    .to_string(),
                case.clone(),
            )
        })
        .collect()
}

struct StrictJson;

impl<'de> Deserialize<'de> for StrictJson {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictJsonVisitor)
    }
}

struct StrictJsonVisitor;

impl<'de> Visitor<'de> for StrictJsonVisitor {
    type Value = StrictJson;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("JSON without duplicate object keys")
    }

    fn visit_bool<E>(self, _value: bool) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }

    fn visit_i64<E>(self, _value: i64) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }

    fn visit_u64<E>(self, _value: u64) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }

    fn visit_f64<E>(self, _value: f64) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }

    fn visit_str<E>(self, _value: &str) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictJson)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        while sequence.next_element::<StrictJson>()?.is_some() {}
        Ok(StrictJson)
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut keys = std::collections::BTreeSet::new();
        while let Some(key) = map.next_key::<String>()? {
            if !keys.insert(key.clone()) {
                return Err(serde::de::Error::custom(format!(
                    "duplicate JSON object key: {key}"
                )));
            }
            map.next_value::<StrictJson>()?;
        }
        Ok(StrictJson)
    }
}

fn validate_strict_json(bytes: &[u8]) -> Result<(), serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_slice(bytes);
    StrictJson::deserialize(&mut deserializer)?;
    deserializer.end()
}

#[test]
fn active_math_build_and_algorithm_identities_are_pinned() {
    assert_eq!(ACTIVE_MATH_BUILD_ID_V1.len(), 64);
    assert!(
        ACTIVE_MATH_BUILD_ID_V1
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    );
    assert_eq!(
        active_math_algorithm_identity_v1(),
        "cb4915056f4b70a76c2d5f0b9db9f9ba029afbc4e4bdf81ba7fe073f68cd4203"
    );
    assert_eq!(
        super::config::OUTCOME_REDUCER_ID_V1,
        "root_outcome_reducer_v1"
    );
    assert_eq!(
        super::config::OUTCOME_DECAY_ID_V1,
        "libm_0_2_16_exp2_force_soft_v1"
    );
    assert_eq!(
        super::config::LEARNING_FUTURE_SKEW_ID_V1,
        "future_skew_300000ms_v1"
    );
    assert_eq!(
        super::config::LEARNING_SUMMATION_ID_V1,
        "neumaier_f64_no_fma_v1"
    );
    assert_eq!(
        super::config::OUTCOME_QUANTILE_ID_V1,
        "monotone_beta_quantile_bounds_isotonic_1e-11_v1"
    );
    assert_eq!(
        super::config::OUTCOME_BETA_CDF_ID_V1,
        "statrs_beta_cdf_0_18_0_v1"
    );
    assert_eq!(
        super::config::OUTCOME_POSTERIOR_RESOLUTION_ID_V1,
        "posterior_resolution_4096_4e-9_v1"
    );
    assert_eq!(
        super::config::OUTCOME_BONFERRONI_ID_V1,
        "bonferroni_max_looks_v1"
    );
    assert_eq!(
        super::config::OUTCOME_ATTRIBUTION_GATE_ID_V1,
        "outcome_attribution_gate_v1"
    );
    assert_eq!(
        super::config::OUTCOME_NONINFERIORITY_MATH_ID_V1,
        "noninferiority_math_v1"
    );
}

#[test]
fn simulation_manifest_is_strict_and_self_consistent() {
    let crate_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture_directory = crate_directory.join("tests/fixtures");
    let manifest_bytes =
        std::fs::read(fixture_directory.join("spec08_active_math_simulation_manifest_v2.json"))
            .unwrap();
    let corpus_bytes =
        std::fs::read(fixture_directory.join("spec08_active_math_corpus_v1.json")).unwrap();
    let reference_bytes =
        std::fs::read(fixture_directory.join("spec08_active_math_reference_v1.json")).unwrap();
    let operational_bytes =
        std::fs::read(fixture_directory.join("spec08_active_math_operational_v1.json")).unwrap();
    let report_bytes =
        std::fs::read(fixture_directory.join("spec08_active_math_simulation_report_v2.json"))
            .unwrap();
    let failure_bytes =
        std::fs::read(fixture_directory.join("spec08_active_math_simulation_failure_v1.json"))
            .unwrap();
    let script_bytes = std::fs::read(
        crate_directory
            .join("../..")
            .join("scripts/dev/generate_spec08_active_math_goldens.py"),
    )
    .unwrap();
    validate_strict_json(&manifest_bytes).unwrap();
    validate_strict_json(&corpus_bytes).unwrap();
    validate_strict_json(&reference_bytes).unwrap();
    validate_strict_json(&operational_bytes).unwrap();
    validate_strict_json(&report_bytes).unwrap();
    validate_strict_json(&failure_bytes).unwrap();
    assert!(validate_strict_json(br#"{"key":1,"key":2}"#).is_err());

    let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).unwrap();
    assert_eq!(
        manifest["schema"],
        "nemo.relay.router.active-math-simulation-manifest@2"
    );
    assert_eq!(manifest["normative"], true);
    assert_eq!(manifest["generated"], false);
    assert_eq!(manifest["bounds_locked_before_execution"], true);
    let reference: serde_json::Value = serde_json::from_slice(&reference_bytes).unwrap();
    let operational: serde_json::Value = serde_json::from_slice(&operational_bytes).unwrap();
    let report: serde_json::Value = serde_json::from_slice(&report_bytes).unwrap();
    let failure: serde_json::Value = serde_json::from_slice(&failure_bytes).unwrap();
    assert_eq!(manifest["supersedes"]["sha256"], failure["manifest_sha256"]);
    assert_eq!(failure["status"], "failed_acceptance");
    assert_eq!(failure["bound_changed_after_result"], false);
    assert_eq!(
        reference["generator"]["script_sha256"],
        sha256_hex(&script_bytes)
    );
    assert_eq!(
        reference["generator"]["corpus_sha256"],
        sha256_hex(&corpus_bytes)
    );
    assert_eq!(
        reference["generator"]["simulation_manifest_sha256"],
        sha256_hex(&manifest_bytes)
    );
    assert_eq!(
        operational["active_math_algorithm_id_sha256"],
        active_math_algorithm_identity_v1()
    );
    assert_eq!(operational["corpus_sha256"], sha256_hex(&corpus_bytes));
    assert_eq!(
        operational["reference_sha256"],
        sha256_hex(&reference_bytes)
    );
    assert_eq!(
        operational["simulation_manifest_sha256"],
        sha256_hex(&manifest_bytes)
    );
    assert_eq!(
        report["schema"],
        "nemo.relay.router.active-math-simulation-report@2"
    );
    assert_eq!(report["manifest_sha256"], sha256_hex(&manifest_bytes));
    assert_eq!(report["corpus_sha256"], sha256_hex(&corpus_bytes));
    assert_eq!(report["reference_sha256"], sha256_hex(&reference_bytes));
    assert_eq!(report["generator_sha256"], sha256_hex(&script_bytes));
    assert_eq!(
        report["active_math_algorithm_id_sha256"],
        active_math_algorithm_identity_v1()
    );
    assert_eq!(report["profile"], "release");
    assert_eq!(report["trial_count"], 125_000);
    assert_eq!(report["all_acceptance_bounds_passed"], true);
    assert!(
        report["scenarios"]
            .as_object()
            .unwrap()
            .values()
            .all(|scenario| scenario["accepted"] == true
                || scenario["acceptance"]["accepted"] == true)
    );
    assert!(
        report["deterministic_cases"]
            .as_array()
            .unwrap()
            .iter()
            .all(|case| case["accepted"] == true)
    );
    let algorithms: Vec<_> = manifest["production_algorithms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    assert_eq!(
        algorithms,
        [
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
        ]
    );
    assert_eq!(
        sha256_hex(manifest["prng"]["source_utf8"].as_str().unwrap().as_bytes()),
        manifest["prng"]["source_sha256"].as_str().unwrap()
    );
    assert_eq!(
        manifest["shared_policy"]["posterior_resolution_bits"],
        hex_bits(ACTIVE_MATH_POSTERIOR_RESOLUTION_V1.to_bits())
    );
    assert_eq!(
        manifest["shared_policy"]["per_look_noninferiority_threshold_bits"],
        hex_bits(valid_policy().per_look_noninferiority_threshold().to_bits())
    );

    for probability in manifest["probabilities"].as_object().unwrap().values() {
        let bits = u64::from_str_radix(probability["bits"].as_str().unwrap(), 16).unwrap();
        let threshold = super::config::exact_probability_threshold(f64::from_bits(bits)).unwrap();
        assert_eq!(probability["threshold_hex"], format!("{threshold:016x}"));
    }

    let scenarios = manifest["scenarios"].as_array().unwrap();
    let mut expected_seed = 0_u64;
    let mut total_trials = 0_u64;
    let mut scenario_ids = std::collections::BTreeSet::new();
    for scenario in scenarios {
        let id = scenario["id"].as_str().unwrap();
        assert!(scenario_ids.insert(id));
        let start = scenario["seed_start"].as_u64().unwrap();
        let end = scenario["seed_end_inclusive"].as_u64().unwrap();
        let trial_count = scenario["trial_count"].as_u64().unwrap();
        assert_eq!(start, expected_seed);
        assert_eq!(end - start + 1, trial_count);
        expected_seed = end + 1;
        total_trials += trial_count;
    }
    assert_eq!(scenario_ids.len(), 9);
    assert_eq!(expected_seed, 125_000);
    assert_eq!(total_trials, 125_000);
    assert_eq!(manifest["allocation"]["scenario_count"], scenarios.len());
    assert_eq!(manifest["allocation"]["total_trial_count"], total_trials);
    assert_eq!(manifest["allocation"]["first_seed"], 0);
    assert_eq!(manifest["allocation"]["last_seed"], expected_seed - 1);
    assert_eq!(
        manifest["simulation_cache"]["parallel_exact_key_plan"]["worker_process_count"],
        16
    );
    assert_eq!(
        manifest["simulation_cache"]["parallel_exact_key_plan"]["hard_wall_seconds_max"],
        3_600
    );
}

#[test]
fn audit_float_serialization_is_exact_and_rejects_nonfinite_values() {
    let value = ActiveAuditF64V1::new(0.1).unwrap();
    assert_eq!(value.bits(), 0x3fb9_9999_9999_999a);
    assert_eq!(
        serde_json::to_string(&value).unwrap(),
        "\"3fb999999999999a\""
    );
    assert_eq!(
        ActiveAuditF64V1::from_bits(f64::NAN.to_bits()),
        Err(ActiveMathErrorV1::Nonfinite)
    );
    assert_eq!(
        ActiveAuditF64V1::new(f64::INFINITY),
        Err(ActiveMathErrorV1::Nonfinite)
    );
}

#[test]
fn policy_helpers_and_validation_use_written_order_bits() {
    assert_eq!(
        bonferroni_threshold_v1(0.99, 2).unwrap().to_bits(),
        0x3fef_d70a_3d70_a3d7
    );
    assert_eq!(
        prior_only_noninferiority_v1(0.1).unwrap().to_bits(),
        0x3fe3_0a3d_70a3_d70a
    );
    assert_eq!(prior_only_noninferiority_v1(0.0), Some(0.5));
    assert_eq!(prior_only_noninferiority_v1(-0.0), None);
    assert_eq!(prior_only_noninferiority_v1(0.250_000_000_1), None);
    assert_eq!(bonferroni_threshold_v1(1.0, 2), None);
    assert_eq!(bonferroni_threshold_v1(0.99, 0), None);

    let policy = valid_policy();
    assert_eq!(policy.actual_outcome_half_life_seconds(), 3_600);
    assert_eq!(
        policy.per_look_noninferiority_threshold().to_bits(),
        0x3fef_d70a_3d70_a3d7
    );
    for invalid in [
        ActiveLookPolicyV1::new(0, 32, 32, 32.0, 32.0, 0.1, 0.99, 0.95, 2),
        ActiveLookPolicyV1::new(3_600, 31, 32, 32.0, 32.0, 0.1, 0.99, 0.95, 2),
        ActiveLookPolicyV1::new(3_600, 32, 32, 15.999, 32.0, 0.1, 0.99, 0.95, 2),
        ActiveLookPolicyV1::new(3_600, 32, 32, 65_537.0, 32.0, 0.1, 0.99, 0.95, 2),
        ActiveLookPolicyV1::new(3_600, 32, 32, 32.0, 32.0, -0.0, 0.99, 0.95, 2),
        ActiveLookPolicyV1::new(3_600, 32, 32, 32.0, 32.0, 0.1, 0.989, 0.95, 2),
        ActiveLookPolicyV1::new(3_600, 32, 32, 32.0, 32.0, 0.1, 0.99, 0.949, 2),
        ActiveLookPolicyV1::new(3_600, 32, 32, 32.0, 32.0, 0.1, 0.99, 0.95, 0),
        ActiveLookPolicyV1::new(3_600, 32, 32, 32.0, 32.0, 0.1, 0.99, 0.95, 257),
    ] {
        assert_eq!(invalid, Err(ActiveMathErrorV1::InvalidInput));
    }
}

#[test]
fn decay_is_pure_software_with_inclusive_future_skew_and_underflow() {
    assert_eq!(
        decay_weight_v1(0, 0, 3_600).unwrap().to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        decay_weight_v1(3_600_000, 0, 3_600).unwrap().to_bits(),
        0.5_f64.to_bits()
    );
    assert_eq!(
        decay_weight_v1(7_200_000, 0, 3_600).unwrap().to_bits(),
        0.25_f64.to_bits()
    );
    assert_eq!(
        decay_weight_v1(1_000_000, 1_300_000, 3_600)
            .unwrap()
            .to_bits(),
        1.0_f64.to_bits()
    );
    assert_eq!(
        decay_weight_v1(1_000_000, 1_300_001, 3_600),
        Err(ActiveMathErrorV1::FutureSkew)
    );
    assert_eq!(
        decay_weight_v1(31_536_000_000_000, 0, 1).unwrap().to_bits(),
        0.0_f64.to_bits()
    );
}

#[test]
fn raw_root_short_circuit_keeps_denominators_and_identity_exact() {
    let members = vec![
        ActiveLookMemberV1 {
            cap_ordinal: 1,
            admission_unix_ms: 1_000_000,
            arm: ActiveOutcomeArmV1::Treatment,
            label: Some(ActiveOutcomeLabelV1::Success),
        },
        ActiveLookMemberV1 {
            cap_ordinal: 2,
            admission_unix_ms: 1_000_000,
            arm: ActiveOutcomeArmV1::Treatment,
            label: None,
        },
        ActiveLookMemberV1 {
            cap_ordinal: 3,
            admission_unix_ms: 1_000_000,
            arm: ActiveOutcomeArmV1::Control,
            label: Some(ActiveOutcomeLabelV1::Failure),
        },
    ];
    let audit =
        skipped_audit(evaluate_active_look_v1(valid_policy(), 1_000_000, &members).unwrap());
    assert_eq!(audit.treatment_denominator, 2);
    assert_eq!(audit.treatment_labeled, 1);
    assert_eq!(audit.control_denominator, 1);
    assert_eq!(audit.control_labeled, 1);
    assert_eq!(audit.reason, LookSkipReasonV1::InsufficientBothArms);
    assert_eq!(audit.raw_beta_cdf_calls, 0);
    assert_eq!(
        audit.active_math_algorithm_id_sha256,
        active_math_algorithm_identity_v1()
    );
    assert_eq!(audit.input_identity_sha256.len(), 64);

    let repeated =
        skipped_audit(evaluate_active_look_v1(valid_policy(), 1_000_000, &members).unwrap());
    assert_eq!(audit, repeated);
    let mut changed = members.clone();
    changed[1].label = Some(ActiveOutcomeLabelV1::Failure);
    let changed =
        skipped_audit(evaluate_active_look_v1(valid_policy(), 1_000_000, &changed).unwrap());
    assert_ne!(audit.input_identity_sha256, changed.input_identity_sha256);

    let mut invalid_order = members;
    invalid_order[1].cap_ordinal = 1;
    assert_eq!(
        evaluate_active_look_v1(valid_policy(), 1_000_000, &invalid_order),
        Err(ActiveMathErrorV1::InvalidOrder)
    );
    assert_eq!(
        evaluate_active_look_v1(valid_policy(), 1_000_000, &[]),
        Err(ActiveMathErrorV1::InvalidInput)
    );
}

#[test]
fn posterior_input_validation_and_cancellation_fail_closed() {
    let invalid = PosteriorSufficientStatisticsV1 {
        treatment_success_weight_bits: (-0.0_f64).to_bits(),
        treatment_failure_weight_bits: 0.0_f64.to_bits(),
        control_success_weight_bits: 0.0_f64.to_bits(),
        control_failure_weight_bits: 0.0_f64.to_bits(),
        noninferiority_margin_bits: 0.1_f64.to_bits(),
    };
    assert_eq!(
        evaluate_posterior_v1(invalid),
        Err(ActiveMathErrorV1::InvalidInput)
    );

    let mut checkpoints = Vec::new();
    let error = evaluate_active_look_with_control_v1(
        valid_policy(),
        1_000_000,
        &balanced_members(32),
        |index| {
            checkpoints.push(index);
            false
        },
    )
    .unwrap_err();
    assert_eq!(error, ActiveMathErrorV1::Canceled);
    assert_eq!(checkpoints, [1]);
}

#[test]
fn uniform_posterior_contains_analytic_probability_and_has_exact_work_counts() {
    let posterior = evaluate_posterior_v1(PosteriorSufficientStatisticsV1 {
        treatment_success_weight_bits: 0.0_f64.to_bits(),
        treatment_failure_weight_bits: 0.0_f64.to_bits(),
        control_success_weight_bits: 0.0_f64.to_bits(),
        control_failure_weight_bits: 0.0_f64.to_bits(),
        noninferiority_margin_bits: 0.0_f64.to_bits(),
    })
    .unwrap();
    assert_eq!(
        posterior.noninferiority_g.lower.bits(),
        0x3fdf_fdff_fdda_60de
    );
    assert_eq!(
        posterior.noninferiority_g.upper.bits(),
        0x3fe0_0100_0112_cf91
    );
    assert_eq!(posterior.noninferiority_h, posterior.noninferiority_g);
    assert_eq!(posterior.noninferiority, posterior.noninferiority_g);
    assert!(posterior.noninferiority.lower() <= 0.5);
    assert!(posterior.noninferiority.upper() >= 0.5);
    assert_eq!(posterior.raw_beta_cdf_calls, 1_801_748);
    assert_eq!(posterior.operational_cdf_queries, 900_890);
    assert!(posterior.raw_beta_cdf_calls <= ACTIVE_MATH_RAW_BETA_CDF_CALLS_MAX_V1);
    assert_eq!(
        posterior.max_symmetry_disagreement.bits(),
        0x3c90_0000_0000_0000
    );
    assert_eq!(posterior.max_quantile_width.bits(), 0x3e21_2e0c_0000_0000);
    assert_eq!(
        posterior.cdf_transcript_sha256,
        "ffc9597c97d793550ca35e72b109bc7c215fc63a4885a7252f413550b05aef31"
    );
}

#[test]
fn completed_look_replay_requires_exact_same_build_or_bounded_cross_build_result() {
    let evaluation = evaluate_active_look_v1(valid_policy(), 1_000_000, &balanced_members(32))
        .expect("balanced complete look must evaluate");
    let audit = match evaluation {
        ActiveLookEvaluationV1::Completed(audit) => audit,
        ActiveLookEvaluationV1::Skipped(_) => panic!("balanced look meets both raw minima"),
    };
    assert_eq!(audit.treatment.denominator, 32);
    assert_eq!(audit.control.denominator, 32);
    assert!(audit.gates.treatment_raw_roots);
    assert!(audit.gates.control_raw_roots);
    assert!(active_look_replay_matches_v1(&audit, &audit));

    let mut changed_transcript = audit.clone();
    changed_transcript.cdf_transcript_sha256 = "0".repeat(64);
    assert!(!active_look_replay_matches_v1(&audit, &changed_transcript));

    let mut cross_build = audit.clone();
    cross_build.active_math_build_id = "0".repeat(64);
    cross_build.noninferiority = ProbabilityIntervalAuditV1::new(
        audit.noninferiority.lower() + 1e-9,
        audit.noninferiority.upper() + 1e-9,
    )
    .unwrap();
    assert!(active_look_replay_matches_v1(&audit, &cross_build));

    let mut outside_tolerance = cross_build.clone();
    outside_tolerance.noninferiority = ProbabilityIntervalAuditV1::new(
        audit.noninferiority.lower(),
        audit.noninferiority.upper() + 3e-9,
    )
    .unwrap();
    assert!(!active_look_replay_matches_v1(&audit, &outside_tolerance));

    let mut wrong_algorithm = cross_build.clone();
    wrong_algorithm.active_math_algorithm_id_sha256 = "f".repeat(64);
    assert!(!active_look_replay_matches_v1(&audit, &wrong_algorithm));
    let mut wrong_input = cross_build;
    wrong_input.input_identity_sha256 = "f".repeat(64);
    assert!(!active_look_replay_matches_v1(&audit, &wrong_input));
}

#[test]
#[ignore = "runs the bounded uncached production engine over the full numeric corpus"]
fn production_engine_matches_checked_in_numeric_corpus() {
    let crate_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture_directory = crate_directory.join("tests/fixtures");
    let corpus_path = fixture_directory.join("spec08_active_math_corpus_v1.json");
    let reference_path = fixture_directory.join("spec08_active_math_reference_v1.json");
    let manifest_path = fixture_directory.join("spec08_active_math_simulation_manifest_v2.json");
    let operational_path = fixture_directory.join("spec08_active_math_operational_v1.json");
    let script_path = crate_directory
        .join("../..")
        .join("scripts/dev/generate_spec08_active_math_goldens.py");
    let corpus_bytes = std::fs::read(&corpus_path).unwrap();
    let reference_bytes = std::fs::read(&reference_path).unwrap();
    let manifest_bytes = std::fs::read(&manifest_path).unwrap();
    let script_bytes = std::fs::read(&script_path).unwrap();
    let corpus: serde_json::Value = serde_json::from_slice(&corpus_bytes).unwrap();
    let reference: serde_json::Value = serde_json::from_slice(&reference_bytes).unwrap();

    assert_eq!(
        reference["generator"]["corpus_sha256"].as_str(),
        Some(sha256_hex(&corpus_bytes).as_str())
    );
    assert_eq!(
        reference["generator"]["simulation_manifest_sha256"].as_str(),
        Some(sha256_hex(&manifest_bytes).as_str())
    );
    assert_eq!(
        reference["generator"]["script_sha256"].as_str(),
        Some(sha256_hex(&script_bytes).as_str())
    );

    let cdf_references = fixture_cases_by_id(&reference, "cdf_cases");
    let mut cdf_results = Vec::new();
    for case in corpus["cdf_cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let alpha = case["alpha"].as_str().unwrap().parse::<f64>().unwrap();
        let beta = case["beta"].as_str().unwrap().parse::<f64>().unwrap();
        let x = case["x"].as_str().unwrap().parse::<f64>().unwrap();
        let (bits, raw_calls) = active_cdf_envelope_probe_v1(alpha, beta, x).unwrap();
        let reference_value = cdf_references[id]["reference"]
            .as_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        let distance = interval_distance(
            reference_value,
            f64::from_bits(bits[4]),
            f64::from_bits(bits[5]),
        );
        assert!(
            distance <= ACTIVE_MATH_CDF_GUARD_V1,
            "CDF corpus miss for {id}"
        );
        cdf_results.push(serde_json::json!({
            "id": id,
            "x_bits": hex_bits(bits[0]),
            "direct_bits": hex_bits(bits[1]),
            "symmetric_bits": hex_bits(bits[2]),
            "midpoint_bits": hex_bits(bits[3]),
            "lower_bits": hex_bits(bits[4]),
            "upper_bits": hex_bits(bits[5]),
            "raw_beta_cdf_calls": raw_calls,
            "reference_distance_bits": hex_bits(distance.to_bits()),
        }));
    }

    let quantile_references = fixture_cases_by_id(&reference, "quantile_cases");
    let mut quantile_results = Vec::new();
    for case in corpus["quantile_cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let alpha = case["alpha"].as_str().unwrap().parse::<f64>().unwrap();
        let beta = case["beta"].as_str().unwrap().parse::<f64>().unwrap();
        let probability = case["u"].as_str().unwrap().parse::<f64>().unwrap();
        let probe = active_quantile_probe_v1(alpha, beta, probability)
            .unwrap_or_else(|error| panic!("quantile corpus case {id} failed: {error:?}"));
        let reference_value = quantile_references[id]["reference"]
            .as_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        let distance = interval_distance(
            reference_value,
            f64::from_bits(probe.lower_bits),
            f64::from_bits(probe.upper_bits),
        );
        assert_eq!(distance, 0.0, "quantile corpus miss for {id}");
        quantile_results.push(serde_json::json!({
            "id": id,
            "lower_bits": hex_bits(probe.lower_bits),
            "upper_bits": hex_bits(probe.upper_bits),
            "raw_beta_cdf_calls": probe.raw_calls,
            "operational_cdf_queries": probe.operational_queries,
            "max_monotonic_repair_bits": hex_bits(probe.max_monotonic_repair_bits),
            "max_quantile_width_bits": hex_bits(probe.max_quantile_width_bits),
            "cdf_transcript_sha256": probe.transcript_sha256,
            "reference_distance_bits": hex_bits(distance.to_bits()),
        }));
    }

    let posterior_references = fixture_cases_by_id(&reference, "posterior_cases");
    let mut posterior_results = Vec::new();
    for case in corpus["posterior_cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        let treatment_alpha = case["treatment_alpha"]
            .as_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        let treatment_beta = case["treatment_beta"]
            .as_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        let control_alpha = case["control_alpha"]
            .as_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        let control_beta = case["control_beta"]
            .as_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        let margin = case["margin"].as_str().unwrap().parse::<f64>().unwrap();
        let statistics = PosteriorSufficientStatisticsV1 {
            treatment_success_weight_bits: (treatment_alpha - 1.0).to_bits(),
            treatment_failure_weight_bits: (treatment_beta - 1.0).to_bits(),
            control_success_weight_bits: (control_alpha - 1.0).to_bits(),
            control_failure_weight_bits: (control_beta - 1.0).to_bits(),
            noninferiority_margin_bits: margin.to_bits(),
        };
        let audit = evaluate_posterior_v1(statistics).unwrap();
        let reference_value = posterior_references[id]["reference"]
            .as_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        let distance = interval_distance(
            reference_value,
            audit.noninferiority.lower(),
            audit.noninferiority.upper(),
        );
        assert!(
            distance <= ACTIVE_MATH_CROSS_PLATFORM_ENDPOINT_TOLERANCE_V1,
            "posterior corpus miss for {id}: {distance}"
        );
        posterior_results.push(serde_json::json!({
            "id": id,
            "statistics": {
                "treatment_success_weight_bits": hex_bits(statistics.treatment_success_weight_bits),
                "treatment_failure_weight_bits": hex_bits(statistics.treatment_failure_weight_bits),
                "control_success_weight_bits": hex_bits(statistics.control_success_weight_bits),
                "control_failure_weight_bits": hex_bits(statistics.control_failure_weight_bits),
                "noninferiority_margin_bits": hex_bits(statistics.noninferiority_margin_bits),
            },
            "audit": audit,
            "reference_distance_bits": hex_bits(distance.to_bits()),
        }));
    }

    let operational = serde_json::json!({
        "schema": "nemo.relay.router.active-math-operational@1",
        "scope": "declared finite corpus on the pinned production engine",
        "continuous_domain_certification": false,
        "active_math_algorithm_id_sha256": active_math_algorithm_identity_v1(),
        "corpus_sha256": sha256_hex(&corpus_bytes),
        "reference_sha256": sha256_hex(&reference_bytes),
        "simulation_manifest_sha256": sha256_hex(&manifest_bytes),
        "cdf_cases": cdf_results,
        "quantile_cases": quantile_results,
        "posterior_cases": posterior_results,
    });
    let mut encoded = serde_json::to_vec_pretty(&operational).unwrap();
    encoded.push(b'\n');
    if std::env::var_os("NEMO_RELAY_UPDATE_ACTIVE_MATH_OPERATIONAL").is_some() {
        std::fs::write(&operational_path, encoded).unwrap();
    } else {
        assert_eq!(std::fs::read(&operational_path).unwrap(), encoded);
    }
}

#[test]
fn fixed_gate_arithmetic_is_inclusive_and_rejects_bad_denominators() {
    assert!(attribution_rate_gate_v1(4, 5).unwrap());
    assert!(!attribution_rate_gate_v1(3, 5).unwrap());
    assert!(differential_attribution_gate_v1(9, 10, 8, 10).unwrap());
    assert!(!differential_attribution_gate_v1(10, 10, 8, 10).unwrap());
    assert_eq!(
        attribution_rate_gate_v1(0, 0),
        Err(ActiveMathErrorV1::InvalidInput)
    );
    assert_eq!(
        attribution_rate_gate_v1(6, 5),
        Err(ActiveMathErrorV1::InvalidInput)
    );
    assert_eq!(
        differential_attribution_gate_v1(0, 0, 1, 1),
        Err(ActiveMathErrorV1::InvalidInput)
    );
}

#[test]
fn final_cap_and_authorization_expiry_boundaries_are_inclusive() {
    assert_eq!(
        reduce_final_cap_state_v1(1, true, true),
        FinalCapStateV1::PendingDrain
    );
    assert_eq!(
        reduce_final_cap_state_v1(0, false, true),
        FinalCapStateV1::PendingTargets
    );
    assert_eq!(
        reduce_final_cap_state_v1(0, true, true),
        FinalCapStateV1::ClosedPassed
    );
    assert_eq!(
        reduce_final_cap_state_v1(0, true, false),
        FinalCapStateV1::Exhausted
    );
    assert!(!authorization_expired_v1(1_000, 1, 1_999).unwrap());
    assert!(authorization_expired_v1(1_000, 1, 2_000).unwrap());
    assert_eq!(
        authorization_expired_v1(-1, 1, 2_000),
        Err(ActiveMathErrorV1::InvalidInput)
    );
    assert_eq!(
        authorization_expired_v1(i64::MAX, 1, i64::MAX),
        Err(ActiveMathErrorV1::NumericFailure)
    );
}
