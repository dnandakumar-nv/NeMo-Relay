// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use statrs::distribution::{Beta, ContinuousCDF};

use super::active_math::*;

const FIXTURE_DIRECTORY: &str = "tests/fixtures";
const MANIFEST_FILE: &str = "spec08_active_math_simulation_manifest_v2.json";
const CORPUS_FILE: &str = "spec08_active_math_corpus_v1.json";
const REFERENCE_FILE: &str = "spec08_active_math_reference_v1.json";
const REPORT_FILE: &str = "spec08_active_math_simulation_report_v2.json";
const WORKER_ENV: &str = "NEMO_RELAY_ACTIVE_SIM_WORKER";
const WORK_DIRECTORY_ENV: &str = "NEMO_RELAY_ACTIVE_SIM_WORK_DIRECTORY";
const UPDATE_REPORT_ENV: &str = "NEMO_RELAY_UPDATE_ACTIVE_MATH_SIMULATION";
const WORKER_COUNT: usize = 16;
const SEED_SHARD_COUNT: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct SufficientKey {
    treatment_success_weight_bits: u64,
    treatment_failure_weight_bits: u64,
    control_success_weight_bits: u64,
    control_failure_weight_bits: u64,
    noninferiority_margin_bits: u64,
}

impl SufficientKey {
    fn statistics(self) -> PosteriorSufficientStatisticsV1 {
        PosteriorSufficientStatisticsV1 {
            treatment_success_weight_bits: self.treatment_success_weight_bits,
            treatment_failure_weight_bits: self.treatment_failure_weight_bits,
            control_success_weight_bits: self.control_success_weight_bits,
            control_failure_weight_bits: self.control_failure_weight_bits,
            noninferiority_margin_bits: self.noninferiority_margin_bits,
        }
    }

    fn to_json(self) -> Value {
        json!([
            hex_bits(self.treatment_success_weight_bits),
            hex_bits(self.treatment_failure_weight_bits),
            hex_bits(self.control_success_weight_bits),
            hex_bits(self.control_failure_weight_bits),
            hex_bits(self.noninferiority_margin_bits),
        ])
    }

    fn from_json(value: &Value) -> Self {
        let fields = value.as_array().expect("sufficient key must be an array");
        assert_eq!(fields.len(), 5);
        Self {
            treatment_success_weight_bits: parse_bits(&fields[0]),
            treatment_failure_weight_bits: parse_bits(&fields[1]),
            control_success_weight_bits: parse_bits(&fields[2]),
            control_failure_weight_bits: parse_bits(&fields[3]),
            noninferiority_margin_bits: parse_bits(&fields[4]),
        }
    }

    fn update_digest(self, digest: &mut Sha256) {
        digest.update(self.treatment_success_weight_bits.to_be_bytes());
        digest.update(self.treatment_failure_weight_bits.to_be_bytes());
        digest.update(self.control_success_weight_bits.to_be_bytes());
        digest.update(self.control_failure_weight_bits.to_be_bytes());
        digest.update(self.noninferiority_margin_bits.to_be_bytes());
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Arm {
    Treatment,
    Control,
}

#[derive(Debug, Clone, Copy)]
struct GeneratedRow {
    arm: Arm,
    admission_unix_ms: i64,
    label: Option<ActiveOutcomeLabelV1>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct ArmSummary {
    denominator: u32,
    labeled: u32,
    successes: u32,
    failures: u32,
    success_weight: f64,
    failure_weight: f64,
}

impl ArmSummary {
    fn push(
        &mut self,
        label: Option<ActiveOutcomeLabelV1>,
        weight: f64,
    ) -> Result<(), ActiveMathErrorV1> {
        self.denominator += 1;
        if let Some(label) = label {
            self.labeled += 1;
            match label {
                ActiveOutcomeLabelV1::Success => {
                    self.successes += 1;
                    self.success_weight += weight;
                }
                ActiveOutcomeLabelV1::Failure => {
                    self.failures += 1;
                    self.failure_weight += weight;
                }
            }
        }
        if !self.success_weight.is_finite() || !self.failure_weight.is_finite() {
            return Err(ActiveMathErrorV1::NumericFailure);
        }
        Ok(())
    }

    fn audit(self) -> ActiveArmAuditV1 {
        let effective_weight = self.success_weight + self.failure_weight;
        ActiveArmAuditV1 {
            denominator: self.denominator,
            labeled: self.labeled,
            successes: self.successes,
            failures: self.failures,
            success_weight: ActiveAuditF64V1::new(self.success_weight).unwrap(),
            failure_weight: ActiveAuditF64V1::new(self.failure_weight).unwrap(),
            effective_weight: ActiveAuditF64V1::new(effective_weight).unwrap(),
            beta_alpha: ActiveAuditF64V1::new(1.0 + self.success_weight).unwrap(),
            beta_beta: ActiveAuditF64V1::new(1.0 + self.failure_weight).unwrap(),
            label_rate: ActiveAuditF64V1::new(
                f64::from(self.labeled) / f64::from(self.denominator),
            )
            .unwrap(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LookInput {
    treatment: ArmSummary,
    control: ArmSummary,
    key: SufficientKey,
}

#[derive(Debug, Clone, PartialEq)]
struct GeneratedTrial {
    looks: [LookInput; 2],
}

#[derive(Debug, Clone, Copy, Default)]
struct AssignmentCounts {
    holdout: u64,
    treatment: u64,
    control: u64,
}

struct PhaseOne {
    keys: Vec<SufficientKey>,
    shard_hashes: Vec<String>,
    assignment: AssignmentCounts,
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

fn shuffled_arms(rng: &mut SplitMix64) -> [Arm; 128] {
    let mut arms = [Arm::Treatment; 128];
    arms[64..].fill(Arm::Control);
    for index in (1..128).rev() {
        let bound = (index + 1) as u128;
        let full_range = 1_u128 << 64;
        let limit = full_range - full_range % bound;
        let draw = loop {
            let draw = u128::from(rng.next_u64());
            if draw < limit {
                break draw;
            }
        };
        arms.swap(index, usize::try_from(draw % bound).unwrap());
    }
    arms
}

fn probability_thresholds(manifest: &Value) -> BTreeMap<String, u128> {
    manifest["probabilities"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, probability)| {
            (
                name.clone(),
                u128::from_str_radix(probability["threshold_hex"].as_str().unwrap(), 16).unwrap(),
            )
        })
        .collect()
}

fn scenario_half_life_seconds(manifest: &Value, scenario: &Value) -> u32 {
    scenario
        .get("policy_overrides")
        .and_then(|value| value.get("actual_outcome_half_life_seconds"))
        .and_then(Value::as_u64)
        .unwrap_or_else(|| {
            manifest["shared_policy"]["actual_outcome_half_life_seconds"]
                .as_u64()
                .unwrap()
        })
        .try_into()
        .unwrap()
}

fn scenario_admission_time(scenario: &Value, tranche: usize) -> i64 {
    let timing = &scenario["timing"];
    if let Some(first) = timing.get("first_admission_unix_ms") {
        let spacing = timing["tranche_spacing_ms"].as_i64().unwrap();
        first.as_i64().unwrap() + spacing * i64::try_from(tranche).unwrap()
    } else {
        timing["admission_unix_ms"].as_i64().unwrap()
    }
}

fn outcome_probability_name(scenario: &Value, arm: Arm, tranche: usize) -> Option<&str> {
    let arm_name = match arm {
        Arm::Treatment => "treatment",
        Arm::Control => "control",
    };
    let by_tranche = format!("{arm_name}_success_probability_by_tranche");
    if let Some(probabilities) = scenario.get(&by_tranche) {
        return probabilities[tranche].as_str();
    }
    scenario
        .get(format!("{arm_name}_success_probability"))
        .and_then(Value::as_str)
}

fn generate_trial(
    manifest: &Value,
    scenario: &Value,
    thresholds: &BTreeMap<String, u128>,
    seed: u64,
) -> GeneratedTrial {
    let mut rng = SplitMix64::new(seed);
    let mut rows = Vec::with_capacity(256);
    let half_life_seconds = scenario_half_life_seconds(manifest, scenario);
    let margin_bits = u64::from_str_radix(
        manifest["shared_policy"]["noninferiority_margin_bits"]
            .as_str()
            .unwrap(),
        16,
    )
    .unwrap();
    let mut looks = Vec::with_capacity(2);

    for tranche in 0..2 {
        let arms = shuffled_arms(&mut rng);
        let admission_unix_ms = scenario_admission_time(scenario, tranche);
        let mut treatment_ordinal = 0_u32;
        let mut control_ordinal = 0_u32;
        let mut generated = Vec::with_capacity(128);
        for arm in arms {
            let arm_ordinal = match arm {
                Arm::Treatment => {
                    let ordinal = treatment_ordinal;
                    treatment_ordinal += 1;
                    ordinal
                }
                Arm::Control => {
                    let ordinal = control_ordinal;
                    control_ordinal += 1;
                    ordinal
                }
            };
            generated.push((
                arm,
                arm_ordinal,
                rng.next_u64(),
                rng.next_u64(),
                rng.next_u64(),
            ));
        }
        assert_eq!(treatment_ordinal, 64);
        assert_eq!(control_ordinal, 64);

        let attrition = scenario.get("attrition").unwrap();
        let mut selected = [true; 128];
        if let Some(attrition) = attrition.as_object() {
            selected.fill(false);
            for arm in [Arm::Treatment, Arm::Control] {
                let quota_name = match arm {
                    Arm::Treatment => "treatment_labeled_per_64",
                    Arm::Control => "control_labeled_per_64",
                };
                let quota = usize::try_from(attrition[quota_name].as_u64().unwrap()).unwrap();
                let mut candidates: Vec<_> = generated
                    .iter()
                    .enumerate()
                    .filter(|(_, row)| row.0 == arm)
                    .map(|(index, row)| (row.3, row.1, index))
                    .collect();
                candidates.sort_unstable();
                for (_, _, index) in candidates.into_iter().take(quota) {
                    selected[index] = true;
                }
            }
        }

        let alternating = scenario.get("outcomes").and_then(Value::as_str)
            == Some("alternating_labeled_arm_ordinal_start_success_v1");
        let mut treatment_labeled_ordinal = 0_u32;
        let mut control_labeled_ordinal = 0_u32;
        for (index, (arm, arm_ordinal, outcome_draw, _attribution_draw, _latency_draw)) in
            generated.into_iter().enumerate()
        {
            let label = if !selected[index] {
                None
            } else if alternating || attrition.is_object() {
                let labeled_ordinal = match arm {
                    Arm::Treatment => {
                        let ordinal = treatment_labeled_ordinal;
                        treatment_labeled_ordinal += 1;
                        ordinal
                    }
                    Arm::Control => {
                        let ordinal = control_labeled_ordinal;
                        control_labeled_ordinal += 1;
                        ordinal
                    }
                };
                let _ = arm_ordinal;
                Some(if labeled_ordinal % 2 == 0 {
                    ActiveOutcomeLabelV1::Success
                } else {
                    ActiveOutcomeLabelV1::Failure
                })
            } else {
                let probability_name = outcome_probability_name(scenario, arm, tranche)
                    .expect("probability scenario must name each arm probability");
                Some(if u128::from(outcome_draw) < thresholds[probability_name] {
                    ActiveOutcomeLabelV1::Success
                } else {
                    ActiveOutcomeLabelV1::Failure
                })
            };
            rows.push(GeneratedRow {
                arm,
                admission_unix_ms,
                label,
            });
        }

        let as_of_unix_ms = scenario_admission_time(scenario, tranche);
        let mut treatment = ArmSummary::default();
        let mut control = ArmSummary::default();
        for row in &rows {
            let weight = if row.admission_unix_ms == as_of_unix_ms {
                1.0
            } else {
                decay_weight_v1(as_of_unix_ms, row.admission_unix_ms, half_life_seconds).unwrap()
            };
            match row.arm {
                Arm::Treatment => treatment.push(row.label, weight).unwrap(),
                Arm::Control => control.push(row.label, weight).unwrap(),
            }
        }
        let key = SufficientKey {
            treatment_success_weight_bits: treatment.success_weight.to_bits(),
            treatment_failure_weight_bits: treatment.failure_weight.to_bits(),
            control_success_weight_bits: control.success_weight.to_bits(),
            control_failure_weight_bits: control.failure_weight.to_bits(),
            noninferiority_margin_bits: margin_bits,
        };
        looks.push(LookInput {
            treatment,
            control,
            key,
        });
    }
    GeneratedTrial {
        looks: looks.try_into().unwrap(),
    }
}

fn generate_assignment_counts(scenario: &Value) -> AssignmentCounts {
    let assignment = &scenario["assignment"];
    let holdout_threshold =
        u128::from_str_radix(assignment["holdout_threshold_hex"].as_str().unwrap(), 16).unwrap();
    let treatment_threshold = u128::from_str_radix(
        assignment["conditional_treatment_threshold_hex"]
            .as_str()
            .unwrap(),
        16,
    )
    .unwrap();
    let roots_per_trial = assignment["roots_per_trial"].as_u64().unwrap();
    let mut counts = AssignmentCounts::default();
    for seed in
        scenario["seed_start"].as_u64().unwrap()..=scenario["seed_end_inclusive"].as_u64().unwrap()
    {
        let mut rng = SplitMix64::new(seed);
        for _ in 0..roots_per_trial {
            let holdout_draw = u128::from(rng.next_u64());
            let treatment_draw = u128::from(rng.next_u64());
            if holdout_draw < holdout_threshold {
                counts.holdout += 1;
            } else if treatment_draw < treatment_threshold {
                counts.treatment += 1;
            } else {
                counts.control += 1;
            }
        }
    }
    counts
}

fn update_transition_digest(
    digest: &mut Sha256,
    scenario_id: &str,
    seed: u64,
    look_index: usize,
    look: LookInput,
) {
    digest.update(u32::try_from(scenario_id.len()).unwrap().to_be_bytes());
    digest.update(scenario_id.as_bytes());
    digest.update(seed.to_be_bytes());
    digest.update(u32::try_from(look_index).unwrap().to_be_bytes());
    look.key.update_digest(digest);
    for arm in [look.treatment, look.control] {
        digest.update(arm.denominator.to_be_bytes());
        digest.update(arm.labeled.to_be_bytes());
        digest.update(arm.successes.to_be_bytes());
        digest.update(arm.failures.to_be_bytes());
    }
}

fn fixed_look(treatment: ArmSummary, control: ArmSummary, margin_bits: u64) -> LookInput {
    LookInput {
        treatment,
        control,
        key: SufficientKey {
            treatment_success_weight_bits: treatment.success_weight.to_bits(),
            treatment_failure_weight_bits: treatment.failure_weight.to_bits(),
            control_success_weight_bits: control.success_weight.to_bits(),
            control_failure_weight_bits: control.failure_weight.to_bits(),
            noninferiority_margin_bits: margin_bits,
        },
    }
}

fn deterministic_looks(manifest: &Value) -> Vec<LookInput> {
    let margin_bits = u64::from_str_radix(
        manifest["shared_policy"]["noninferiority_margin_bits"]
            .as_str()
            .unwrap(),
        16,
    )
    .unwrap();
    vec![
        fixed_look(
            ArmSummary {
                denominator: 64,
                labeled: 64,
                successes: 32,
                failures: 32,
                success_weight: 2.0,
                failure_weight: 2.0,
            },
            ArmSummary {
                denominator: 64,
                labeled: 64,
                successes: 32,
                failures: 32,
                success_weight: 2.0,
                failure_weight: 2.0,
            },
            margin_bits,
        ),
        fixed_look(
            ArmSummary {
                denominator: 128,
                labeled: 128,
                successes: 64,
                failures: 64,
                success_weight: 32.0,
                failure_weight: 64.0,
            },
            ArmSummary {
                denominator: 128,
                labeled: 128,
                successes: 64,
                failures: 64,
                success_weight: 64.0,
                failure_weight: 32.0,
            },
            margin_bits,
        ),
        fixed_look(
            ArmSummary {
                denominator: 64,
                labeled: 64,
                successes: 32,
                failures: 32,
                success_weight: 32.0,
                failure_weight: 32.0,
            },
            ArmSummary {
                denominator: 64,
                labeled: 64,
                successes: 32,
                failures: 32,
                success_weight: 32.0,
                failure_weight: 32.0,
            },
            margin_bits,
        ),
    ]
}

fn phase_one(manifest: &Value) -> PhaseOne {
    let thresholds = probability_thresholds(manifest);
    let scenarios = manifest["scenarios"].as_array().unwrap();
    let assignment_scenario = scenarios
        .iter()
        .find(|scenario| scenario["id"] == "assignment_calibration")
        .unwrap();
    let assignment = generate_assignment_counts(assignment_scenario);
    let mut keys = BTreeSet::new();
    let mut shard_digests: [Sha256; SEED_SHARD_COUNT] = std::array::from_fn(|_| Sha256::new());
    for scenario in scenarios {
        let scenario_id = scenario["id"].as_str().unwrap();
        if scenario_id == "assignment_calibration" {
            continue;
        }
        for seed in scenario["seed_start"].as_u64().unwrap()
            ..=scenario["seed_end_inclusive"].as_u64().unwrap()
        {
            let trial = generate_trial(manifest, scenario, &thresholds, seed);
            let shard = usize::try_from(seed % SEED_SHARD_COUNT as u64).unwrap();
            for (look_index, look) in trial.looks.into_iter().enumerate() {
                keys.insert(look.key);
                update_transition_digest(
                    &mut shard_digests[shard],
                    scenario_id,
                    seed,
                    look_index,
                    look,
                );
            }
        }
    }
    for look in deterministic_looks(manifest) {
        keys.insert(look.key);
    }
    PhaseOne {
        keys: keys.into_iter().collect(),
        shard_hashes: shard_digests
            .into_iter()
            .map(|digest| hex_digest(digest.finalize()))
            .collect(),
        assignment,
    }
}

fn parse_bits(value: &Value) -> u64 {
    u64::from_str_radix(value.as_str().unwrap(), 16).unwrap()
}

fn hex_bits(bits: u64) -> String {
    format!("{bits:016x}")
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn posterior_from_json(value: &Value) -> PosteriorAuditV1 {
    fn interval(value: &Value) -> ProbabilityIntervalAuditV1 {
        ProbabilityIntervalAuditV1::new(
            f64::from_bits(parse_bits(&value["lower"])),
            f64::from_bits(parse_bits(&value["upper"])),
        )
        .unwrap()
    }

    PosteriorAuditV1 {
        noninferiority_g: interval(&value["noninferiority_g"]),
        noninferiority_h: interval(&value["noninferiority_h"]),
        noninferiority: interval(&value["noninferiority"]),
        raw_beta_cdf_calls: value["raw_beta_cdf_calls"]
            .as_u64()
            .unwrap()
            .try_into()
            .unwrap(),
        operational_cdf_queries: value["operational_cdf_queries"]
            .as_u64()
            .unwrap()
            .try_into()
            .unwrap(),
        max_symmetry_disagreement: ActiveAuditF64V1::from_bits(parse_bits(
            &value["max_symmetry_disagreement"],
        ))
        .unwrap(),
        max_monotonic_repair: ActiveAuditF64V1::from_bits(parse_bits(
            &value["max_monotonic_repair"],
        ))
        .unwrap(),
        max_quantile_width: ActiveAuditF64V1::from_bits(parse_bits(&value["max_quantile_width"]))
            .unwrap(),
        cdf_transcript_sha256: value["cdf_transcript_sha256"].as_str().unwrap().to_string(),
    }
}

fn run_worker() {
    let worker: usize = std::env::var(WORKER_ENV).unwrap().parse().unwrap();
    assert!(worker < WORKER_COUNT);
    let work_directory = std::path::PathBuf::from(std::env::var_os(WORK_DIRECTORY_ENV).unwrap());
    let keys: Value =
        serde_json::from_slice(&std::fs::read(work_directory.join("keys.json")).unwrap()).unwrap();
    let mut records = Vec::new();
    for (ordinal, value) in keys.as_array().unwrap().iter().enumerate() {
        if ordinal % WORKER_COUNT != worker {
            continue;
        }
        let key = SufficientKey::from_json(value);
        let audit = evaluate_posterior_v1(key.statistics()).unwrap_or_else(|error| {
            panic!("worker {worker} failed key {ordinal} {key:?}: {error:?}")
        });
        assert!(audit.raw_beta_cdf_calls <= ACTIVE_MATH_RAW_BETA_CDF_CALLS_MAX_V1);
        records.push(json!({
            "ordinal": ordinal,
            "key": key.to_json(),
            "audit": audit,
        }));
    }
    let mut encoded = serde_json::to_vec(&records).unwrap();
    encoded.push(b'\n');
    std::fs::write(
        work_directory.join(format!("worker-{worker:02}.json")),
        encoded,
    )
    .unwrap();
}

fn worker_cache_path(keys: &[SufficientKey]) -> std::path::PathBuf {
    let mut digest = Sha256::new();
    digest.update(b"nemo-relay-router/active-simulation-key-set/v1\0");
    for key in keys {
        key.update_digest(&mut digest);
    }
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("target/spec08-active-math-cache")
        .join(format!(
            "{}-{}.json",
            ACTIVE_MATH_BUILD_ID_V1,
            hex_digest(digest.finalize())
        ))
}

fn decode_worker_records(
    records: &Value,
    keys: &[SufficientKey],
) -> BTreeMap<SufficientKey, PosteriorAuditV1> {
    let mut audits = BTreeMap::new();
    for record in records.as_array().unwrap() {
        let ordinal = record["ordinal"].as_u64().unwrap() as usize;
        let key = SufficientKey::from_json(&record["key"]);
        assert_eq!(keys[ordinal], key);
        let audit = posterior_from_json(&record["audit"]);
        assert!(audits.insert(key, audit).is_none());
    }
    audits
}

fn evaluate_keys_in_workers(keys: &[SufficientKey]) -> BTreeMap<SufficientKey, PosteriorAuditV1> {
    let cache_path = worker_cache_path(keys);
    if let Ok(encoded) = std::fs::read(&cache_path) {
        let records: Value = serde_json::from_slice(&encoded).unwrap();
        let audits = decode_worker_records(&records, keys);
        assert_eq!(audits.len(), keys.len());
        return audits;
    }
    let desired: BTreeSet<_> = keys.iter().copied().collect();
    let mut audits = BTreeMap::new();
    if let Ok(entries) = std::fs::read_dir(cache_path.parent().unwrap()) {
        for entry in entries.flatten() {
            let Some(name) = entry.file_name().to_str().map(str::to_string) else {
                continue;
            };
            if !name.starts_with(ACTIVE_MATH_BUILD_ID_V1) || !name.ends_with(".json") {
                continue;
            }
            let records: Value =
                serde_json::from_slice(&std::fs::read(entry.path()).unwrap()).unwrap();
            for record in records.as_array().unwrap() {
                let key = SufficientKey::from_json(&record["key"]);
                if !desired.contains(&key) {
                    continue;
                }
                let audit = posterior_from_json(&record["audit"]);
                if let Some(existing) = audits.insert(key, audit.clone()) {
                    assert_eq!(existing, audit);
                }
            }
        }
    }
    let missing_keys: Vec<_> = keys
        .iter()
        .copied()
        .filter(|key| !audits.contains_key(key))
        .collect();
    if missing_keys.is_empty() {
        assert_eq!(audits.len(), keys.len());
        return audits;
    }
    let work_directory = tempfile::tempdir().unwrap();
    let key_values: Vec<_> = missing_keys.iter().map(|key| key.to_json()).collect();
    std::fs::write(
        work_directory.path().join("keys.json"),
        serde_json::to_vec(&key_values).unwrap(),
    )
    .unwrap();

    let preflight_start = Instant::now();
    let preflight = evaluate_posterior_v1(missing_keys[0].statistics()).unwrap();
    assert!(preflight_start.elapsed() <= Duration::from_secs(2));
    assert!(preflight.raw_beta_cdf_calls <= ACTIVE_MATH_RAW_BETA_CDF_CALLS_MAX_V1);

    let executable = std::env::current_exe().unwrap();
    let test_name = "active_math_simulation_tests::fixed_manifest_simulation_is_deterministic";
    let started = Instant::now();
    let mut children: Vec<_> = (0..WORKER_COUNT)
        .map(|worker| {
            Command::new(&executable)
                .arg(test_name)
                .arg("--exact")
                .arg("--ignored")
                .env(WORKER_ENV, worker.to_string())
                .env(WORK_DIRECTORY_ENV, work_directory.path())
                .env("RUST_TEST_THREADS", "1")
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .spawn()
                .unwrap()
        })
        .collect();
    let mut complete = [false; WORKER_COUNT];
    while complete.iter().any(|done| !done) {
        if started.elapsed() > Duration::from_secs(3_600) {
            for child in &mut children {
                let _ = child.kill();
                let _ = child.wait();
            }
            panic!("fixed simulation worker wall exceeded 3,600 seconds");
        }
        let mut failure = None;
        for (worker, child) in children.iter_mut().enumerate() {
            if complete[worker] {
                continue;
            }
            if let Some(status) = child.try_wait().unwrap() {
                if !status.success() {
                    failure = Some(format!("simulation worker {worker} failed: {status}"));
                    break;
                }
                complete[worker] = true;
            }
        }
        if let Some(failure) = failure {
            for (worker, child) in children.iter_mut().enumerate() {
                if !complete[worker] {
                    let _ = child.kill();
                    let _ = child.wait();
                }
            }
            panic!("{failure}");
        }
        if complete.iter().any(|done| !done) {
            std::thread::sleep(Duration::from_millis(100));
        }
    }

    let mut combined_records = Vec::new();
    for worker in 0..WORKER_COUNT {
        let records: Value = serde_json::from_slice(
            &std::fs::read(
                work_directory
                    .path()
                    .join(format!("worker-{worker:02}.json")),
            )
            .unwrap(),
        )
        .unwrap();
        for record in records.as_array().unwrap() {
            let ordinal = record["ordinal"].as_u64().unwrap() as usize;
            assert_eq!(ordinal % WORKER_COUNT, worker);
            combined_records.push(record.clone());
        }
    }
    combined_records.sort_by_key(|record| record["ordinal"].as_u64().unwrap());
    let new_records = Value::Array(combined_records);
    let new_audits = decode_worker_records(&new_records, &missing_keys);
    for (key, audit) in new_audits {
        assert!(audits.insert(key, audit).is_none());
    }
    assert_eq!(audits.len(), keys.len());
    let records = Value::Array(
        keys.iter()
            .enumerate()
            .map(|(ordinal, key)| {
                json!({
                    "ordinal": ordinal,
                    "key": key.to_json(),
                    "audit": audits[key],
                })
            })
            .collect(),
    );
    std::fs::create_dir_all(cache_path.parent().unwrap()).unwrap();
    let temporary_path = cache_path.with_extension(format!("tmp-{}", std::process::id()));
    std::fs::write(&temporary_path, serde_json::to_vec(&records).unwrap()).unwrap();
    std::fs::rename(temporary_path, cache_path).unwrap();
    audits
}

fn production_audit_sha256(
    keys: &[SufficientKey],
    audits: &BTreeMap<SufficientKey, PosteriorAuditV1>,
) -> String {
    let mut aggregate = Sha256::new();
    aggregate.update(b"nemo-relay-router/active-simulation-audits/v1\0");
    for key in keys {
        key.update_digest(&mut aggregate);
        aggregate.update(serde_json::to_vec(&audits[key]).unwrap());
    }
    hex_digest(aggregate.finalize())
}

fn simulation_policy(half_life_seconds: u32) -> ActiveLookPolicyV1 {
    ActiveLookPolicyV1::new(half_life_seconds, 32, 32, 32.0, 32.0, 0.1, 0.99, 0.95, 2).unwrap()
}

fn reduce_look(
    policy: ActiveLookPolicyV1,
    look: LookInput,
    posterior: &PosteriorAuditV1,
) -> (ActiveLookGateAuditV1, LookProducedStateV1) {
    let rollback_lower = 1.0 - posterior.noninferiority.upper();
    let gates = reduce_gates_v1(
        policy,
        look.treatment.audit(),
        look.control.audit(),
        posterior,
        rollback_lower,
    )
    .unwrap();
    let state = reduce_predicates_v1(gates.rollback, gates.noninferiority);
    (gates, state)
}

#[derive(Debug, Default)]
struct ScenarioMetrics {
    trials: u64,
    passed: u64,
    rollback: u64,
    collecting: u64,
    final_attribution_gate_pass: u64,
    pass_with_failed_attribution_gate: u64,
    paired_state_mismatch: u64,
    paired_sufficient_statistic_mismatch: u64,
}

fn fraction(numerator: u64, denominator: u64) -> f64 {
    numerator as f64 / denominator as f64
}

fn clopper_pearson_upper(successes: u64, trials: u64, confidence: f64) -> f64 {
    if successes == trials {
        1.0
    } else {
        Beta::new((successes + 1) as f64, (trials - successes) as f64)
            .unwrap()
            .inverse_cdf(confidence)
    }
}

fn clopper_pearson_interval(successes: u64, trials: u64, confidence: f64) -> (f64, f64) {
    let tail = (1.0 - confidence) / 2.0;
    let lower = if successes == 0 {
        0.0
    } else {
        Beta::new(successes as f64, (trials - successes + 1) as f64)
            .unwrap()
            .inverse_cdf(tail)
    };
    let upper = if successes == trials {
        1.0
    } else {
        Beta::new((successes + 1) as f64, (trials - successes) as f64)
            .unwrap()
            .inverse_cdf(1.0 - tail)
    };
    (lower, upper)
}

fn metric_value(count: u64, trials: u64) -> Value {
    json!({
        "count": count,
        "fraction_bits": hex_bits(fraction(count, trials).to_bits()),
    })
}

fn assignment_report(manifest: &Value, counts: AssignmentCounts) -> Value {
    let scenario = manifest["scenarios"]
        .as_array()
        .unwrap()
        .iter()
        .find(|scenario| scenario["id"] == "assignment_calibration")
        .unwrap();
    let assignment = &scenario["assignment"];
    let trials = counts.holdout + counts.treatment + counts.control;
    let full_range = (1_u128 << 64) as f64;
    let holdout_threshold =
        u128::from_str_radix(assignment["holdout_threshold_hex"].as_str().unwrap(), 16).unwrap();
    let treatment_threshold = u128::from_str_radix(
        assignment["conditional_treatment_threshold_hex"]
            .as_str()
            .unwrap(),
        16,
    )
    .unwrap();
    let holdout_probability = holdout_threshold as f64 / full_range;
    let conditional_treatment = treatment_threshold as f64 / full_range;
    let treatment_probability = (1.0 - holdout_probability) * conditional_treatment;
    let control_probability = (1.0 - holdout_probability) * (1.0 - conditional_treatment);
    let maximum_error = scenario["acceptance"]["absolute_arm_frequency_error_max"]
        .as_str()
        .unwrap()
        .parse::<f64>()
        .unwrap();
    let confidence = scenario["acceptance"]["clopper_pearson_confidence"]
        .as_str()
        .unwrap()
        .parse::<f64>()
        .unwrap();
    let arms = [
        ("holdout", counts.holdout, holdout_probability),
        ("treatment", counts.treatment, treatment_probability),
        ("control", counts.control, control_probability),
    ];
    let mut arm_reports = serde_json::Map::new();
    for (name, count, expected) in arms {
        let observed = fraction(count, trials);
        let interval = clopper_pearson_interval(count, trials, confidence);
        assert!((observed - expected).abs() <= maximum_error);
        assert!(interval.0 <= expected && expected <= interval.1);
        arm_reports.insert(
            name.to_string(),
            json!({
                "count": count,
                "observed_fraction_bits": hex_bits(observed.to_bits()),
                "expected_discrete_fraction_bits": hex_bits(expected.to_bits()),
                "clopper_pearson_lower_bits": hex_bits(interval.0.to_bits()),
                "clopper_pearson_upper_bits": hex_bits(interval.1.to_bits()),
            }),
        );
    }
    json!({
        "trial_count": scenario["trial_count"],
        "root_count": trials,
        "arms": arm_reports,
        "accepted": true,
    })
}

fn scenario_acceptance(scenario: &Value, metrics: &ScenarioMetrics) -> Value {
    let id = scenario["id"].as_str().unwrap();
    let acceptance = &scenario["acceptance"];
    match id {
        "null_boundary" => {
            let confidence = acceptance["false_pass_clopper_pearson_confidence"]
                .as_str()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            let upper = clopper_pearson_upper(metrics.passed, metrics.trials, confidence);
            let maximum = acceptance["false_pass_upper_max"]
                .as_str()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            json!({
                "false_pass_clopper_pearson_upper_bits": hex_bits(upper.to_bits()),
                "maximum_bits": hex_bits(maximum.to_bits()),
                "accepted": upper <= maximum,
            })
        }
        "degraded" | "drift" => {
            let minimum = acceptance["rollback_by_cap_fraction_min"]
                .as_str()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            let observed = fraction(metrics.rollback, metrics.trials);
            json!({
                "rollback_fraction_bits": hex_bits(observed.to_bits()),
                "minimum_bits": hex_bits(minimum.to_bits()),
                "accepted": observed >= minimum,
            })
        }
        "equal" => {
            let pass = fraction(metrics.passed, metrics.trials);
            let rollback = fraction(metrics.rollback, metrics.trials);
            let minimum = acceptance["pass_by_cap_fraction_min"]
                .as_str()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            let maximum = acceptance["pass_by_cap_fraction_max"]
                .as_str()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            let rollback_maximum = acceptance["rollback_by_cap_fraction_max"]
                .as_str()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            json!({
                "pass_fraction_bits": hex_bits(pass.to_bits()),
                "pass_minimum_bits": hex_bits(minimum.to_bits()),
                "pass_maximum_bits": hex_bits(maximum.to_bits()),
                "rollback_fraction_bits": hex_bits(rollback.to_bits()),
                "rollback_maximum_bits": hex_bits(rollback_maximum.to_bits()),
                "accepted": (minimum..=maximum).contains(&pass)
                    && rollback <= rollback_maximum,
            })
        }
        "improved" => {
            let pass_minimum = acceptance["pass_by_cap_fraction_min"]
                .as_str()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            let rollback_maximum = acceptance["rollback_by_cap_fraction_max"]
                .as_str()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            let pass = fraction(metrics.passed, metrics.trials);
            let rollback = fraction(metrics.rollback, metrics.trials);
            json!({
                "pass_fraction_bits": hex_bits(pass.to_bits()),
                "pass_minimum_bits": hex_bits(pass_minimum.to_bits()),
                "rollback_fraction_bits": hex_bits(rollback.to_bits()),
                "rollback_maximum_bits": hex_bits(rollback_maximum.to_bits()),
                "accepted": pass >= pass_minimum && rollback <= rollback_maximum,
            })
        }
        "arm_latency" => {
            json!({
                "accepted": metrics.paired_state_mismatch == 0
                    && metrics.paired_sufficient_statistic_mismatch == 0,
            })
        }
        "equal_attrition" => {
            let minimum = acceptance["final_attribution_gate_pass_fraction_min"]
                .as_str()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            let observed = fraction(metrics.final_attribution_gate_pass, metrics.trials);
            json!({
                "gate_pass_fraction_bits": hex_bits(observed.to_bits()),
                "minimum_bits": hex_bits(minimum.to_bits()),
                "accepted": observed >= minimum
                    && metrics.pass_with_failed_attribution_gate == 0,
            })
        }
        "differential_attrition" => {
            let minimum = acceptance["final_attribution_gate_rejection_fraction_min"]
                .as_str()
                .unwrap()
                .parse::<f64>()
                .unwrap();
            let rejected = metrics.trials - metrics.final_attribution_gate_pass;
            let observed = fraction(rejected, metrics.trials);
            json!({
                "gate_rejection_fraction_bits": hex_bits(observed.to_bits()),
                "minimum_bits": hex_bits(minimum.to_bits()),
                "accepted": observed >= minimum
                    && metrics.pass_with_failed_attribution_gate == 0,
            })
        }
        _ => panic!("unknown simulation scenario {id}"),
    }
}

fn replay_scenarios(
    manifest: &Value,
    phase_one: &PhaseOne,
    audits: &BTreeMap<SufficientKey, PosteriorAuditV1>,
) -> (Value, Vec<String>) {
    let thresholds = probability_thresholds(manifest);
    let mut reports = serde_json::Map::new();
    let mut failures = Vec::new();
    reports.insert(
        "assignment_calibration".to_string(),
        assignment_report(manifest, phase_one.assignment),
    );
    let mut shard_digests: [Sha256; SEED_SHARD_COUNT] = std::array::from_fn(|_| Sha256::new());
    for scenario in manifest["scenarios"].as_array().unwrap() {
        let id = scenario["id"].as_str().unwrap();
        if id == "assignment_calibration" {
            continue;
        }
        let policy = simulation_policy(scenario_half_life_seconds(manifest, scenario));
        let mut metrics = ScenarioMetrics::default();
        for seed in scenario["seed_start"].as_u64().unwrap()
            ..=scenario["seed_end_inclusive"].as_u64().unwrap()
        {
            metrics.trials += 1;
            let trial = generate_trial(manifest, scenario, &thresholds, seed);
            let shard = usize::try_from(seed % SEED_SHARD_COUNT as u64).unwrap();
            let mut final_state = LookProducedStateV1::Collecting;
            let mut rollback = false;
            let mut final_attribution = false;
            for (look_index, look) in trial.looks.into_iter().enumerate() {
                update_transition_digest(&mut shard_digests[shard], id, seed, look_index, look);
                let (gates, state) = reduce_look(policy, look, &audits[&look.key]);
                let attribution = gates.treatment_attribution
                    && gates.control_attribution
                    && gates.differential_attribution;
                if state == LookProducedStateV1::Passed && !attribution {
                    metrics.pass_with_failed_attribution_gate += 1;
                }
                final_attribution = attribution;
                final_state = state;
                if state == LookProducedStateV1::Rollback {
                    rollback = true;
                }
            }
            if final_attribution {
                metrics.final_attribution_gate_pass += 1;
            }
            if rollback {
                metrics.rollback += 1;
            } else {
                match final_state {
                    LookProducedStateV1::Passed => metrics.passed += 1,
                    LookProducedStateV1::Collecting => metrics.collecting += 1,
                    LookProducedStateV1::Rollback => unreachable!(),
                }
            }
        }
        assert_eq!(metrics.trials, scenario["trial_count"].as_u64().unwrap());
        let acceptance = scenario_acceptance(scenario, &metrics);
        if acceptance["accepted"] != true {
            failures.push(id.to_string());
        }
        reports.insert(
            id.to_string(),
            json!({
                "trial_count": metrics.trials,
                "passed_by_cap": metric_value(metrics.passed, metrics.trials),
                "rollback_by_cap": metric_value(metrics.rollback, metrics.trials),
                "collecting_at_cap": metric_value(metrics.collecting, metrics.trials),
                "final_attribution_gate_pass": metric_value(
                    metrics.final_attribution_gate_pass,
                    metrics.trials,
                ),
                "pass_with_failed_attribution_gate_count": metrics.pass_with_failed_attribution_gate,
                "paired_state_mismatch_count": metrics.paired_state_mismatch,
                "paired_sufficient_statistic_mismatch_count":
                    metrics.paired_sufficient_statistic_mismatch,
                "acceptance": acceptance,
            }),
        );
    }
    let reports = Value::Object(reports);
    if !failures.is_empty() {
        eprintln!(
            "fixed simulation scenario metrics:\n{}",
            serde_json::to_string_pretty(&reports).unwrap()
        );
        panic!(
            "fixed simulation acceptance failed: {}",
            failures.join(", ")
        );
    }
    (
        reports,
        shard_digests
            .into_iter()
            .map(|digest| hex_digest(digest.finalize()))
            .collect(),
    )
}

fn deterministic_report(
    manifest: &Value,
    audits: &BTreeMap<SufficientKey, PosteriorAuditV1>,
) -> Value {
    let looks = deterministic_looks(manifest);
    let (floor_gates, floor_state) =
        reduce_look(simulation_policy(3_600), looks[0], &audits[&looks[0].key]);
    assert_eq!(floor_state, LookProducedStateV1::Collecting);
    assert!(!floor_gates.treatment_effective_weight);
    assert!(!floor_gates.control_effective_weight);

    let (_, drift_state) = reduce_look(simulation_policy(600), looks[1], &audits[&looks[1].key]);
    assert_eq!(drift_state, LookProducedStateV1::Rollback);
    assert_eq!(
        looks[1].treatment.success_weight.to_bits(),
        0x4040_0000_0000_0000
    );
    assert_eq!(
        looks[1].treatment.failure_weight.to_bits(),
        0x4050_0000_0000_0000
    );
    assert_eq!(
        looks[1].control.success_weight.to_bits(),
        0x4050_0000_0000_0000
    );
    assert_eq!(
        looks[1].control.failure_weight.to_bits(),
        0x4040_0000_0000_0000
    );

    let fresh_recovery = &audits[&looks[2].key];
    let recovered_recomputation = evaluate_posterior_v1(looks[2].key.statistics()).unwrap();
    assert_eq!(fresh_recovery, &recovered_recomputation);
    let (_, recovery_state) = reduce_look(simulation_policy(3_600), looks[2], fresh_recovery);
    assert_eq!(recovery_state, LookProducedStateV1::Collecting);

    assert!(authorization_expired_v1(1_800_000_000_000, 3_600, 1_800_003_600_000).unwrap());
    assert_eq!(
        reduce_final_cap_state_v1(0, true, false),
        FinalCapStateV1::Exhausted
    );
    assert_eq!(
        reduce_final_cap_state_v1(0, true, true),
        FinalCapStateV1::ClosedPassed
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
        reduce_predicates_v1(true, true),
        LookProducedStateV1::Rollback
    );

    json!([
        {"id": "effective_weight_floor", "derived_state": "collecting", "accepted": true},
        {
            "id": "nonzero_dyadic_decay_drift",
            "derived_state": "rollback",
            "treatment_success_weight_bits": hex_bits(looks[1].treatment.success_weight.to_bits()),
            "treatment_failure_weight_bits": hex_bits(looks[1].treatment.failure_weight.to_bits()),
            "control_success_weight_bits": hex_bits(looks[1].control.success_weight.to_bits()),
            "control_failure_weight_bits": hex_bits(looks[1].control.failure_weight.to_bits()),
            "accepted": true,
        },
        {"id": "authorization_expiry_exact_boundary", "derived_state": "expired", "accepted": true},
        {"id": "final_cap_without_current_pass", "derived_state": "exhausted", "accepted": true},
        {"id": "final_cap_with_current_pass", "derived_state": "closed_passed", "accepted": true},
        {"id": "future_skew_inclusive", "derived_age_ms": 0, "accepted": true},
        {"id": "future_skew_exceeded", "derived_error": "future_skew", "accepted": true},
        {"id": "rollback_precedes_pass", "derived_state": "rollback", "accepted": true},
        {
            "id": "expired_claim_recovery",
            "derived_state": "collecting",
            "recomputed_audit_matches_fresh": true,
            "recomputed_cdf_call_count": recovered_recomputation.raw_beta_cdf_calls,
            "ownership_commit_fence_validation": "deferred_to_task_8_runtime_claim_integration",
            "accepted": true,
        },
    ])
}

fn sha256_file(path: &std::path::Path) -> String {
    hex_digest(Sha256::digest(std::fs::read(path).unwrap()))
}

fn run_simulation_once(
    cached_audits: Option<&BTreeMap<SufficientKey, PosteriorAuditV1>>,
) -> (Vec<u8>, Option<BTreeMap<SufficientKey, PosteriorAuditV1>>) {
    let crate_directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let fixture_directory = crate_directory.join(FIXTURE_DIRECTORY);
    let manifest_path = fixture_directory.join(MANIFEST_FILE);
    let corpus_path = fixture_directory.join(CORPUS_FILE);
    let reference_path = fixture_directory.join(REFERENCE_FILE);
    let generator_path = crate_directory
        .join("../..")
        .join("scripts/dev/generate_spec08_active_math_goldens.py");
    let manifest: Value = serde_json::from_slice(&std::fs::read(&manifest_path).unwrap()).unwrap();
    let reference: Value =
        serde_json::from_slice(&std::fs::read(&reference_path).unwrap()).unwrap();
    let phase_one = phase_one(&manifest);
    assert_eq!(phase_one.shard_hashes.len(), SEED_SHARD_COUNT);
    let expected_maximum = manifest["simulation_cache"]["expected_fixed_manifest_key_count_max"]
        .as_u64()
        .unwrap() as usize;
    let hard_maximum = manifest["simulation_cache"]["hard_distinct_key_upper_bound"]
        .as_u64()
        .unwrap() as usize;
    assert!(phase_one.keys.len() <= expected_maximum);
    assert!(phase_one.keys.len() <= hard_maximum);
    if let Some(cached_audits) = cached_audits {
        assert_eq!(
            cached_audits.keys().copied().collect::<Vec<_>>(),
            phase_one.keys
        );
    }
    let evaluated_audits = if cached_audits.is_none() {
        Some(evaluate_keys_in_workers(&phase_one.keys))
    } else {
        None
    };
    let audits = cached_audits
        .or(evaluated_audits.as_ref())
        .expect("one exact posterior cache must exist");
    let production_audit_sha256 = production_audit_sha256(&phase_one.keys, audits);
    let (scenarios, replay_shard_hashes) = replay_scenarios(&manifest, &phase_one, audits);
    assert_eq!(replay_shard_hashes, phase_one.shard_hashes);
    let deterministic_cases = deterministic_report(&manifest, audits);
    let raw_beta_cdf_calls: u64 = audits
        .values()
        .map(|audit| u64::from(audit.raw_beta_cdf_calls))
        .sum();
    let operational_cdf_queries: u64 = audits
        .values()
        .map(|audit| u64::from(audit.operational_cdf_queries))
        .sum();
    assert_eq!(
        reference["generator"]["simulation_manifest_sha256"],
        sha256_file(&manifest_path)
    );
    assert_eq!(
        reference["generator"]["script_sha256"],
        sha256_file(&generator_path)
    );
    let report = json!({
        "schema": "nemo.relay.router.active-math-simulation-report@2",
        "manifest_sha256": sha256_file(&manifest_path),
        "corpus_sha256": sha256_file(&corpus_path),
        "reference_sha256": sha256_file(&reference_path),
        "generator_sha256": sha256_file(&generator_path),
        "active_math_build_id": ACTIVE_MATH_BUILD_ID_V1,
        "active_math_algorithm_id_sha256": active_math_algorithm_identity_v1(),
        "profile": "release",
        "trial_count": manifest["allocation"]["total_trial_count"],
        "seed_shard_count": SEED_SHARD_COUNT,
        "worker_process_count": WORKER_COUNT,
        "distinct_sufficient_statistic_key_count": phase_one.keys.len(),
        "raw_beta_cdf_calls": raw_beta_cdf_calls,
        "operational_cdf_queries": operational_cdf_queries,
        "production_audit_sha256": production_audit_sha256,
        "phase_one_transition_shard_sha256": phase_one.shard_hashes,
        "scenarios": scenarios,
        "deterministic_cases": deterministic_cases,
        "all_acceptance_bounds_passed": true,
    });
    let mut encoded = serde_json::to_vec_pretty(&report).unwrap();
    encoded.push(b'\n');
    (encoded, evaluated_audits)
}

#[test]
#[ignore = "runs 125,000 fixed trials and the exact production posterior in 16 processes"]
fn fixed_manifest_simulation_is_deterministic() {
    if std::env::var_os(WORKER_ENV).is_some() {
        run_worker();
        return;
    }
    assert!(
        std::env::current_exe()
            .unwrap()
            .components()
            .any(|component| component.as_os_str() == "release"),
        "fixed simulation must run in release mode"
    );
    let (first, audits) = run_simulation_once(None);
    let (second, _) = run_simulation_once(audits.as_ref());
    assert_eq!(first, second);
    let report_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_DIRECTORY)
        .join(REPORT_FILE);
    if std::env::var_os(UPDATE_REPORT_ENV).is_some() {
        std::fs::write(report_path, first).unwrap();
    } else {
        assert_eq!(std::fs::read(report_path).unwrap(), first);
    }
}

#[test]
#[ignore = "generates all fixed trial keys without running posterior workers"]
fn fixed_manifest_phase_one_is_deterministic_and_bounded() {
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_DIRECTORY)
        .join(MANIFEST_FILE);
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest_path).unwrap()).unwrap();
    let first = phase_one(&manifest);
    let second = phase_one(&manifest);
    assert_eq!(first.keys, second.keys);
    assert_eq!(first.shard_hashes, second.shard_hashes);
    assert_eq!(first.assignment.holdout, second.assignment.holdout);
    assert_eq!(first.assignment.treatment, second.assignment.treatment);
    assert_eq!(first.assignment.control, second.assignment.control);
    eprintln!("fixed manifest distinct exact keys: {}", first.keys.len());
    assert!(first.keys.len() <= 8_192);
}

#[test]
#[ignore = "evaluates every fixed exact key once through 16 production workers"]
fn fixed_manifest_all_keys_are_production_evaluable() {
    assert!(
        std::env::current_exe()
            .unwrap()
            .components()
            .any(|component| component.as_os_str() == "release"),
        "fixed simulation workers must run in release mode"
    );
    let manifest_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_DIRECTORY)
        .join(MANIFEST_FILE);
    let manifest: Value = serde_json::from_slice(&std::fs::read(manifest_path).unwrap()).unwrap();
    let phase = phase_one(&manifest);
    let audits = evaluate_keys_in_workers(&phase.keys);
    assert_eq!(audits.len(), phase.keys.len());
    assert_eq!(production_audit_sha256(&phase.keys, &audits).len(), 64);
}
