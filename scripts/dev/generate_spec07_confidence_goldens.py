# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate independent high-precision Spec 07 confidence fixtures.

Run with:
    uv run --with mpmath==1.3.0 python scripts/dev/generate_spec07_confidence_goldens.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import mpmath as mp  # ty: ignore[unresolved-import]

SCHEMA = "nemo.relay.router.confidence-goldens@1"
PRECISION_DECIMAL_DIGITS = 100
EXPECTED_MPMATH_VERSION = "1.3.0"
ALGORITHM_IDS = [
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


def decimal(value: mp.mpf) -> str:
    return mp.nstr(value, 90, strip_zeros=False)


def beta_inverse_cdf(probability: mp.mpf, alpha: mp.mpf, beta: mp.mpf) -> mp.mpf:
    """Invert the regularized incomplete Beta by monotone high-precision bisection."""
    if not (0 < probability < 1 and alpha > 0 and beta > 0):
        raise ValueError("Beta inverse-CDF inputs must be interior and positive")
    lower = mp.mpf("0")
    upper = mp.mpf("1")
    tolerance = mp.power(10, -(PRECISION_DECIMAL_DIGITS - 10))
    for _ in range(2_048):
        midpoint = (lower + upper) / 2
        cumulative = mp.betainc(alpha, beta, 0, midpoint, regularized=True)
        if cumulative < probability:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance:
            break
    return (lower + upper) / 2


def f32_bits(value: float) -> str:
    bits = struct.unpack(">I", struct.pack(">f", value))[0]
    return f"0x{bits:08x}"


def f64_bits(value: float) -> str:
    bits = struct.unpack(">Q", struct.pack(">d", value))[0]
    return f"0x{bits:016x}"


def f32_from_bits(bits: str) -> mp.mpf:
    raw = int(bits, 16)
    sign = -1 if raw >> 31 else 1
    exponent = (raw >> 23) & 0xFF
    significand = raw & 0x7FFFFF
    if exponent == 0:
        return mp.mpf(sign) * mp.mpf(significand) * mp.power(2, -149)
    if exponent == 0xFF:
        raise ValueError("fixture distances must be finite")
    return mp.mpf(sign) * mp.mpf((1 << 23) | significand) * mp.power(2, exponent - 127 - 23)


def evaluation(
    sequence: int,
    label: str | None,
    created_at_unix_ms: int,
    *,
    confidence: float = 0.99,
    promotion_eligible: bool = True,
) -> dict[str, Any]:
    return {
        "evaluation_sequence": sequence,
        "source": "judge",
        "label": label,
        "judge_confidence_bits": f64_bits(confidence),
        "promotion_eligible": promotion_eligible,
        "created_at_unix_ms": created_at_unix_ms,
    }


def neighbor(
    sequence: int,
    root_sequence: int,
    distance: float,
    evaluation_value: dict[str, Any] | None,
    *,
    terminal: str = "completed",
) -> dict[str, Any]:
    return {
        "evidence_sequence": 1_000 + sequence,
        "attempt_sequence": 2_000 + sequence,
        "anchor_sequence": 3_000 + sequence,
        "root_sequence": 4_000 + root_sequence,
        "terminal": terminal,
        "distance_bits": f32_bits(distance),
        "evaluation": evaluation_value,
    }


def fixture_candidates(as_of: int) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "candidate-a",
            "cost_rank": 1,
            "partition_present": True,
            "neighbors": [
                neighbor(1, 1, 0.10, evaluation(101, "pass", as_of - 1_000)),
                neighbor(2, 1, 0.10, evaluation(102, "fail", as_of)),
                neighbor(3, 2, 0.30, evaluation(103, "pass", as_of - 1_800_000)),
                neighbor(4, 3, 0.20, None, terminal="operational_failure"),
                neighbor(5, 4, 0.40, evaluation(105, None, as_of, promotion_eligible=False)),
                neighbor(6, 5, 0.90, evaluation(106, "pass", as_of)),
            ],
        },
        {
            "candidate_id": "candidate-b",
            "cost_rank": 2,
            "partition_present": True,
            "neighbors": [
                neighbor(11, 11, 0.05, evaluation(111, "pass", as_of)),
                neighbor(12, 12, 0.15, evaluation(112, "pass", as_of - 600_000)),
                neighbor(13, 13, 0.25, evaluation(113, "pass", as_of - 1_800_000)),
                neighbor(14, 14, 0.35, evaluation(114, "pass", as_of + 300_000)),
                neighbor(
                    15,
                    15,
                    0.20,
                    evaluation(115, "pass", as_of, confidence=0.69),
                ),
            ],
        },
        {
            "candidate_id": "candidate-c",
            "cost_rank": 3,
            "partition_present": False,
            "neighbors": [],
        },
    ]


def eligible(evaluation_value: dict[str, Any] | None, judge_floor: mp.mpf) -> bool:
    if evaluation_value is None:
        return False
    if not evaluation_value["promotion_eligible"] or evaluation_value["label"] not in {
        "pass",
        "fail",
    }:
        return False
    confidence_bits = int(evaluation_value["judge_confidence_bits"], 16)
    confidence = struct.unpack(">d", struct.pack(">Q", confidence_bits))[0]
    return mp.mpf(confidence) >= judge_floor


def evaluate_candidate(
    policy: dict[str, Any],
    candidate_count: int,
    as_of: int,
    candidate: dict[str, Any],
) -> tuple[str, mp.mpf | None]:
    if not candidate["partition_present"]:
        return "no_partition", None
    radius = mp.mpf(policy["radius"])
    judge_floor = mp.mpf(policy["judge_confidence_floor"])
    inside = [row for row in candidate["neighbors"] if f32_from_bits(row["distance_bits"]) <= radius]
    labeled = [row for row in inside if eligible(row["evaluation"], judge_floor)]
    attempted_roots = {row["root_sequence"] for row in inside}
    labeled_roots = {row["root_sequence"] for row in labeled}
    coverage = mp.mpf(len(labeled_roots)) / len(attempted_roots) if attempted_roots else mp.mpf("0")
    if len(labeled) < policy["min_points"]:
        return "sparse_points", None
    if len(labeled_roots) < policy["min_independent_roots"]:
        return "insufficient_roots", None
    if coverage < mp.mpf(policy["min_coverage"]):
        return "low_coverage", None

    selected: dict[int, dict[str, Any]] = {}
    for row in labeled:
        evaluation_value = row["evaluation"]
        key = (
            f32_from_bits(row["distance_bits"]),
            -evaluation_value["created_at_unix_ms"],
            evaluation_value["evaluation_sequence"],
        )
        current = selected.get(row["root_sequence"])
        if current is None:
            selected[row["root_sequence"]] = row
            continue
        current_evaluation = current["evaluation"]
        current_key = (
            f32_from_bits(current["distance_bits"]),
            -current_evaluation["created_at_unix_ms"],
            current_evaluation["evaluation_sequence"],
        )
        if key < current_key:
            selected[row["root_sequence"]] = row

    weights: list[mp.mpf] = []
    weighted_passes: list[mp.mpf] = []
    for row in sorted(selected.values(), key=lambda value: value["evaluation"]["evaluation_sequence"]):
        evaluation_value = row["evaluation"]
        future = evaluation_value["created_at_unix_ms"] - as_of
        if future > 300_000:
            return "invalid_evidence_time", None
        age_millis = max(0, as_of - evaluation_value["created_at_unix_ms"])
        similarity = max(mp.mpf("0"), mp.mpf("1") - f32_from_bits(row["distance_bits"]) / radius)
        time_weight = mp.power(
            2,
            -(mp.mpf(age_millis) / 1000) / mp.mpf(policy["half_life_seconds"]),
        )
        weight = similarity * time_weight
        weights.append(weight)
        weighted_passes.append(weight if evaluation_value["label"] == "pass" else mp.mpf("0"))
    sum_weight = mp.fsum(weights)
    sum_weight_squared = mp.fsum([weight * weight for weight in weights])
    if sum_weight == 0 or sum_weight_squared == 0:
        return "numeric_error", None
    p_hat = mp.fsum(weighted_passes) / sum_weight
    n_eff = sum_weight * sum_weight / sum_weight_squared
    if n_eff < mp.mpf(policy["min_effective_samples"]):
        return "insufficient_effective_samples", None
    alpha = mp.mpf(policy["prior_success"]) + p_hat * n_eff
    beta = mp.mpf(policy["prior_failure"]) + (1 - p_hat) * n_eff
    probability = (1 - mp.mpf(policy["familywise_credible_level"])) / candidate_count
    lower_bound = beta_inverse_cdf(probability, alpha, beta)
    if lower_bound < mp.mpf(policy["promotion_lower_bound"]):
        return "lower_bound_below_threshold", lower_bound
    return "passed", lower_bound


def complete_decision(policy: dict[str, Any], as_of: int, candidates: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(candidates, key=lambda value: (value["cost_rank"], value["candidate_id"]))
    winner: str | None = None
    fallback = False
    reasons: list[str] = []
    lower_bounds: list[str | None] = []
    for candidate in ordered:
        if winner is not None:
            reasons.append("not_evaluated_after_winner")
            lower_bounds.append(None)
            continue
        if fallback:
            reasons.append("not_evaluated_after_fallback")
            lower_bounds.append(None)
            continue
        reason, lower_bound = evaluate_candidate(policy, len(ordered), as_of, candidate)
        reasons.append(reason)
        lower_bounds.append(None if lower_bound is None else decimal(lower_bound))
        if reason == "passed":
            winner = candidate["candidate_id"]
        elif reason in {"invalid_evidence_time", "numeric_error"}:
            fallback = True
    if winner is None:
        raise RuntimeError("the independent fixture must exercise a winner")
    return {
        "policy": policy,
        "as_of_unix_ms": as_of,
        "candidates": candidates,
        "expected_winner": winner,
        "expected_reasons": reasons,
        "expected_lower_bounds": lower_bounds,
    }


def build_fixture(script_path: Path) -> dict[str, Any]:
    as_of = 1_800_000_000_000
    policy = {
        "top_k": 16,
        "radius": "0.8",
        "min_points": 2,
        "min_independent_roots": 2,
        "min_effective_samples": "1.5",
        "min_coverage": "0.5",
        "half_life_seconds": "3600",
        "prior_success": "1",
        "prior_failure": "1",
        "familywise_credible_level": "0.95",
        "promotion_lower_bound": "0.35",
        "judge_confidence_floor": "0.7",
    }
    quantile_inputs = [
        ("central", "0.025", "6", "2"),
        ("lower_tail", "0.000001", "2.5", "12"),
        ("asymmetric_near_one", "0.01", "40", "0.5"),
        ("small_shape_tail", "0.0001", "0.75", "8"),
        ("bonferroni_m64", "0.00015625", "5", "1"),
    ]
    quantiles = []
    for name, probability, alpha, beta in quantile_inputs:
        expected = beta_inverse_cdf(mp.mpf(probability), mp.mpf(alpha), mp.mpf(beta))
        quantiles.append(
            {
                "name": name,
                "probability": probability,
                "alpha": alpha,
                "beta": beta,
                "expected": decimal(expected),
            }
        )
    return {
        "schema": SCHEMA,
        "algorithm_ids": ALGORITHM_IDS,
        "statrs_version": "0.18.0",
        "generator": {
            "name": "mpmath",
            "version": mp.__version__,
            "precision_decimal_digits": PRECISION_DECIMAL_DIGITS,
            "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        },
        "quantiles": quantiles,
        "decision": complete_decision(policy, as_of, fixture_candidates(as_of)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    repository_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "crates/router/tests/fixtures/spec07_confidence_v1.json",
    )
    args = parser.parse_args()
    if mp.__version__ != EXPECTED_MPMATH_VERSION:
        raise RuntimeError(f"expected mpmath {EXPECTED_MPMATH_VERSION}, found {mp.__version__}")
    mp.mp.dps = PRECISION_DECIMAL_DIGITS
    fixture = build_fixture(Path(__file__).resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
