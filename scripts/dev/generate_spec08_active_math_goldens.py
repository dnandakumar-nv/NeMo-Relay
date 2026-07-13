# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate independent high-precision Spec 08 active-math references.

Run with:

    uv run --with mpmath==1.3.0 python \
      scripts/dev/generate_spec08_active_math_goldens.py

The generator evaluates the declared finite corpus only. It does not certify the
continuous Beta-shape or margin domain and is not a production dependency.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import mpmath as mp  # ty: ignore[unresolved-import]

SCHEMA = "nemo.relay.router.active-math-reference@1"
EXPECTED_MPMATH_VERSION = "1.3.0"
PRECISION_DECIMAL_DIGITS = 100
VERIFICATION_DECIMAL_DIGITS = 140
QUANTILE_STEPS = 512
POSTERIOR_WORKER_PROCESSES = 4
mp.mp.dps = VERIFICATION_DECIMAL_DIGITS
INTEGRATION_ERROR_GUARD = mp.mpf("1e-60")
INTEGRATION_RELATIVE_ERROR_GUARD = mp.mpf("1e-40")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def load_json_strict(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_pairs)


def decimal(value: mp.mpf) -> str:
    if not mp.isfinite(value):
        raise ValueError("reference values must be finite")
    return mp.nstr(value, n=PRECISION_DECIMAL_DIGITS, strip_zeros=False)


def beta_continued_fraction(alpha: mp.mpf, beta: mp.mpf, x: mp.mpf) -> mp.mpf:
    """Evaluate the incomplete-beta continued fraction with arbitrary precision."""
    total = alpha + beta
    alpha_plus_one = alpha + 1
    alpha_minus_one = alpha - 1
    tiny = mp.power(2, -4 * mp.mp.prec)
    tolerance = 32 * mp.eps
    c = mp.mpf(1)
    d = 1 - total * x / alpha_plus_one
    if abs(d) < tiny:
        d = tiny
    d = 1 / d
    result = d
    for iteration in range(1, 200_001):
        doubled = 2 * iteration
        coefficient = iteration * (beta - iteration) * x / ((alpha_minus_one + doubled) * (alpha + doubled))
        d = 1 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        result *= d * c

        coefficient = -(
            (alpha + iteration) * (total + iteration) * x / ((alpha + doubled) * (alpha_plus_one + doubled))
        )
        d = 1 + coefficient * d
        if abs(d) < tiny:
            d = tiny
        c = 1 + coefficient / c
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        delta = d * c
        result *= delta
        if abs(delta - 1) <= tolerance:
            return result
    raise RuntimeError("incomplete-beta continued fraction did not converge")


def beta_cdf(alpha: mp.mpf, beta: mp.mpf, x: mp.mpf) -> mp.mpf:
    if x <= 0:
        return mp.mpf("0")
    if x >= 1:
        return mp.mpf("1")
    log_factor = (
        mp.loggamma(alpha + beta) - mp.loggamma(alpha) - mp.loggamma(beta) + alpha * mp.log(x) + beta * mp.log1p(-x)
    )
    factor = mp.exp(log_factor)
    if x < (alpha + 1) / (alpha + beta + 2):
        return factor * beta_continued_fraction(alpha, beta, x) / alpha
    complement = factor * beta_continued_fraction(beta, alpha, 1 - x) / beta
    return 1 - complement


def beta_survival(alpha: mp.mpf, beta: mp.mpf, x: mp.mpf) -> mp.mpf:
    if x <= 0:
        return mp.mpf("1")
    if x >= 1:
        return mp.mpf("0")
    return beta_cdf(beta, alpha, 1 - x)


def beta_quantile(alpha: mp.mpf, beta: mp.mpf, probability: mp.mpf) -> mp.mpf:
    if not (alpha > 0 and beta > 0 and 0 < probability < 1):
        raise ValueError("quantile inputs must be positive and interior")
    lower = mp.mpf("0")
    upper = mp.mpf("1")
    for _ in range(QUANTILE_STEPS):
        midpoint = lower + (upper - lower) / 2
        if beta_cdf(alpha, beta, midpoint) < probability:
            lower = midpoint
        else:
            upper = midpoint
    return lower + (upper - lower) / 2


def beta_log_pdf(alpha: mp.mpf, beta: mp.mpf, log_normalizer: mp.mpf, x: mp.mpf) -> mp.mpf:
    if x <= 0 or x >= 1:
        return mp.ninf
    return (alpha - 1) * mp.log(x) + (beta - 1) * mp.log1p(-x) - log_normalizer


def concave_peak(log_integrand: Any) -> mp.mpf:
    """Locate the peak of a log-concave one-dimensional integrand."""
    left = mp.mpf("0")
    right = mp.mpf("1")
    for _ in range(256):
        span = (right - left) / 3
        first = left + span
        second = right - span
        if log_integrand(first) < log_integrand(second):
            left = first
        else:
            right = second
    return left + (right - left) / 2


def peak_breakpoints(peak: mp.mpf, total_shape: mp.mpf) -> set[mp.mpf]:
    points = {peak}
    scale = 1 / mp.sqrt(total_shape)
    for exponent in range(-8, 6):
        offset = mp.power(2, exponent) * scale
        points.add(peak - offset)
        points.add(peak + offset)
    return points


def shape_breakpoints(alpha: mp.mpf, beta: mp.mpf) -> set[mp.mpf]:
    points: set[mp.mpf] = set()
    total = alpha + beta
    mean = alpha / total
    variance = alpha * beta / (total * total * (total + 1))
    sigma = mp.sqrt(variance)
    points.add(mean)
    if alpha > 1 and beta > 1:
        points.add((alpha - 1) / (total - 2))
    for scale in (1, 2, 4, 8, 16):
        points.add(max(mp.mpf("0"), mean - scale * sigma))
        points.add(min(mp.mpf("1"), mean + scale * sigma))
    return points


def integration_breakpoints(
    primary_alpha: mp.mpf,
    primary_beta: mp.mpf,
    secondary_alpha: mp.mpf,
    secondary_beta: mp.mpf,
    secondary_shift: mp.mpf,
    clamp_boundary: mp.mpf,
    extra_points: set[mp.mpf],
) -> list[mp.mpf]:
    points = {mp.mpf("0"), mp.mpf("1"), clamp_boundary}
    points.update(shape_breakpoints(primary_alpha, primary_beta))
    points.update(point + secondary_shift for point in shape_breakpoints(secondary_alpha, secondary_beta))
    points.update(extra_points)
    return sorted(point for point in points if 0 <= point <= 1)


def integrate_with_error(integrand: Any, points: list[mp.mpf]) -> tuple[mp.mpf, mp.mpf]:
    values: list[mp.mpf] = []
    errors: list[mp.mpf] = []
    for left, right in zip(points, points[1:]):
        if left == right:
            continue
        value, error = mp.quad(integrand, [left, right], error=True, maxdegree=12)
        values.append(value)
        errors.append(error)
    return mp.fsum(values), mp.fsum(errors)


def noninferiority_reference(
    treatment_alpha: mp.mpf,
    treatment_beta: mp.mpf,
    control_alpha: mp.mpf,
    control_beta: mp.mpf,
    margin: mp.mpf,
) -> dict[str, mp.mpf]:
    if not 0 <= margin <= mp.mpf("0.25"):
        raise ValueError("margin is outside the declared domain")
    exact = None
    if margin == 0 and treatment_alpha == control_alpha and treatment_beta == control_beta:
        exact = mp.mpf("0.5")
    elif treatment_alpha == 1 and treatment_beta == 1 and control_alpha == 1 and control_beta == 1:
        exact = 1 - (1 - margin) * (1 - margin) / 2
    if exact is not None:
        zero = mp.mpf("0")
        return {
            "reference": exact,
            "orientation_g": exact,
            "orientation_h": exact,
            "orientation_error": zero,
            "precision_repeat_error": zero,
            "quadrature_reported_error": zero,
            "quadrature_reported_relative_error": zero,
            "relative_error": zero,
        }

    def evaluate(dps: int) -> tuple[mp.mpf, mp.mpf, mp.mpf, mp.mpf]:
        with mp.workdps(dps):
            treatment_log_normalizer = (
                mp.loggamma(treatment_alpha)
                + mp.loggamma(treatment_beta)
                - mp.loggamma(treatment_alpha + treatment_beta)
            )
            control_log_normalizer = (
                mp.loggamma(control_alpha) + mp.loggamma(control_beta) - mp.loggamma(control_alpha + control_beta)
            )

            def g_log_integrand(theta_t: mp.mpf) -> mp.mpf:
                control_limit = min(mp.mpf("1"), theta_t + margin)
                cdf = beta_cdf(control_alpha, control_beta, control_limit)
                if cdf <= 0:
                    return mp.ninf
                return beta_log_pdf(
                    treatment_alpha,
                    treatment_beta,
                    treatment_log_normalizer,
                    theta_t,
                ) + mp.log(cdf)

            def h_log_integrand(theta_c: mp.mpf) -> mp.mpf:
                treatment_limit = max(mp.mpf("0"), theta_c - margin)
                survival = beta_survival(treatment_alpha, treatment_beta, treatment_limit)
                if survival <= 0:
                    return mp.ninf
                return beta_log_pdf(
                    control_alpha,
                    control_beta,
                    control_log_normalizer,
                    theta_c,
                ) + mp.log(survival)

            total_shape = treatment_alpha + treatment_beta + control_alpha + control_beta
            g_peak = concave_peak(g_log_integrand)
            h_peak = concave_peak(h_log_integrand)
            g_peak_log = g_log_integrand(g_peak)
            h_peak_log = h_log_integrand(h_peak)

            def scaled_g(theta_t: mp.mpf) -> mp.mpf:
                value = g_log_integrand(theta_t)
                return mp.mpf("0") if value == mp.ninf else mp.exp(value - g_peak_log)

            def scaled_h(theta_c: mp.mpf) -> mp.mpf:
                value = h_log_integrand(theta_c)
                return mp.mpf("0") if value == mp.ninf else mp.exp(value - h_peak_log)

            g_points = integration_breakpoints(
                treatment_alpha,
                treatment_beta,
                control_alpha,
                control_beta,
                -margin,
                1 - margin,
                peak_breakpoints(g_peak, total_shape),
            )
            h_points = integration_breakpoints(
                control_alpha,
                control_beta,
                treatment_alpha,
                treatment_beta,
                margin,
                margin,
                peak_breakpoints(h_peak, total_shape),
            )
            scaled_g_integral, scaled_g_error = integrate_with_error(scaled_g, g_points)
            scaled_h_integral, scaled_h_error = integrate_with_error(scaled_h, h_points)
            g_scale = mp.exp(g_peak_log)
            h_scale = mp.exp(h_peak_log)
            g = scaled_g_integral * g_scale
            h = scaled_h_integral * h_scale
            g_error = scaled_g_error * g_scale
            h_error = scaled_h_error * h_scale
            return g, h, g_error, h_error

    low_g, low_h, low_g_error, low_h_error = evaluate(PRECISION_DECIMAL_DIGITS)
    high_g, high_h, high_g_error, high_h_error = evaluate(VERIFICATION_DECIMAL_DIGITS)
    orientation_error = abs(high_g - high_h)
    repeat_error = max(abs(high_g - low_g), abs(high_h - low_h))
    reported_error = max(low_g_error, low_h_error, high_g_error, high_h_error)
    absolute_error = max(orientation_error, repeat_error)
    scale = max(abs(high_g), abs(high_h))
    relative_error = absolute_error / scale if scale > 0 else absolute_error
    reported_relative_error = reported_error / scale if scale > 0 else reported_error
    if absolute_error > INTEGRATION_ERROR_GUARD or relative_error > INTEGRATION_RELATIVE_ERROR_GUARD:
        raise RuntimeError(
            "posterior reference failed its independent precision/orientation guard: "
            f"orientation={orientation_error}, repeat={repeat_error}, "
            f"relative={relative_error}, quad={reported_error}"
        )
    return {
        "reference": (high_g + high_h) / 2,
        "orientation_g": high_g,
        "orientation_h": high_h,
        "orientation_error": orientation_error,
        "precision_repeat_error": repeat_error,
        "quadrature_reported_error": reported_error,
        "quadrature_reported_relative_error": reported_relative_error,
        "relative_error": relative_error,
    }


def f64_from_hex(bits: str) -> float:
    return struct.unpack(">d", bytes.fromhex(bits))[0]


def f64_hex(value: float) -> str:
    return struct.pack(">d", value).hex()


def decay_reference(case: dict[str, Any]) -> dict[str, Any]:
    delta_ms = int(case["delta_ms"])
    half_life = int(case["half_life_seconds"])
    if delta_ms < -300_000:
        return {**case, "expected_error": "future_skew"}
    age_ms = max(0, delta_ms)
    exponent = -(mp.mpf(age_ms) / 1000) / half_life
    reference = mp.power(2, exponent)
    result: dict[str, Any] = {
        **case,
        "age_ms": age_ms,
        "reference": decimal(reference),
    }
    if exponent == mp.floor(exponent) and exponent >= -1074:
        result["exact_binary64_bits"] = f64_hex(float(mp.power(2, exponent)))
        result["expected_production_class"] = "finite_positive"
    elif reference <= mp.power(2, -1075):
        result["expected_production_class"] = "positive_zero"
        result["exact_binary64_bits"] = "0000000000000000"
    else:
        result["expected_production_class"] = "finite_positive"
    return result


def state_boundary_reference(case: dict[str, Any]) -> dict[str, Any]:
    if case.get("synthetic_precedence_only"):
        state = "rollback" if case["rollback_predicate"] else ("passed" if case["pass_predicate"] else "collecting")
        if state != case["expected_state"]:
            raise ValueError(f"invalid synthetic state boundary {case['id']}")
        return {**case, "derived_state": state}
    lower = f64_from_hex(case["noninferiority_lower_bits"])
    upper = f64_from_hex(case["noninferiority_upper_bits"])
    per_look = f64_from_hex(case["per_look_threshold_bits"])
    rollback_threshold = f64_from_hex(case.get("rollback_threshold_bits", "3fee666666666666"))
    rollback_lower = 1.0 - upper
    rollback = rollback_lower >= rollback_threshold
    passed = case["all_promotion_gates_pass"] and lower >= per_look
    state = "rollback" if rollback else ("passed" if passed else "collecting")
    rollback_bits = f64_hex(rollback_lower)
    if rollback_bits != case["expected_rollback_lower_bits"] or state != case["expected_state"]:
        raise ValueError(f"invalid state boundary {case['id']}")
    return {
        **case,
        "derived_rollback_lower_bits": rollback_bits,
        "derived_state": state,
    }


def posterior_case_reference(case: dict[str, Any]) -> dict[str, Any]:
    mp.mp.dps = VERIFICATION_DECIMAL_DIGITS
    reference = noninferiority_reference(
        mp.mpf(case["treatment_alpha"]),
        mp.mpf(case["treatment_beta"]),
        mp.mpf(case["control_alpha"]),
        mp.mpf(case["control_beta"]),
        mp.mpf(case["margin"]),
    )
    return {
        **case,
        **{name: decimal(value) for name, value in reference.items()},
    }


def reference_fixture(corpus_path: Path, simulation_manifest_path: Path, script_path: Path) -> dict[str, Any]:
    corpus = load_json_strict(corpus_path)
    simulation_manifest = load_json_strict(simulation_manifest_path)
    if simulation_manifest.get("normative") is not True:
        raise ValueError("simulation manifest must be normative")
    cdf_cases = []
    for case in corpus["cdf_cases"]:
        value = beta_cdf(mp.mpf(case["alpha"]), mp.mpf(case["beta"]), mp.mpf(case["x"]))
        cdf_cases.append({**case, "reference": decimal(value)})

    quantile_cases = []
    for case in corpus["quantile_cases"]:
        value = beta_quantile(mp.mpf(case["alpha"]), mp.mpf(case["beta"]), mp.mpf(case["u"]))
        quantile_cases.append({**case, "reference": decimal(value)})

    with concurrent.futures.ProcessPoolExecutor(max_workers=POSTERIOR_WORKER_PROCESSES) as executor:
        posterior_cases = list(executor.map(posterior_case_reference, corpus["posterior_cases"]))

    decay_cases = [decay_reference(case) for case in corpus["decay_cases"]]
    state_boundary_cases = [state_boundary_reference(case) for case in corpus["state_boundary_cases"]]

    return {
        "schema": SCHEMA,
        "scope": "declared finite adversarial corpus only",
        "continuous_domain_certification": False,
        "generator": {
            "name": "mpmath",
            "version": mp.__version__,
            "precision_decimal_digits": PRECISION_DECIMAL_DIGITS,
            "verification_decimal_digits": VERIFICATION_DECIMAL_DIGITS,
            "quantile_bisection_steps": QUANTILE_STEPS,
            "posterior_worker_processes": POSTERIOR_WORKER_PROCESSES,
            "integration_error_guard": decimal(INTEGRATION_ERROR_GUARD),
            "integration_relative_error_guard": decimal(INTEGRATION_RELATIVE_ERROR_GUARD),
            "script_sha256": sha256(script_path),
            "corpus_sha256": sha256(corpus_path),
            "simulation_manifest_sha256": sha256(simulation_manifest_path),
        },
        "cdf_cases": cdf_cases,
        "quantile_cases": quantile_cases,
        "posterior_cases": posterior_cases,
        "decay_cases": decay_cases,
        "state_boundary_cases": state_boundary_cases,
    }


def encoded_fixture(corpus_path: Path, simulation_manifest_path: Path, script_path: Path) -> bytes:
    fixture = reference_fixture(corpus_path, simulation_manifest_path, script_path)
    return (json.dumps(fixture, indent=2) + "\n").encode("utf-8")


def main() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--simulation-manifest",
        type=Path,
        default=repository_root / "crates/router/tests/fixtures/spec08_active_math_simulation_manifest_v2.json",
    )
    parser.add_argument(
        "--corpus",
        type=Path,
        default=repository_root / "crates/router/tests/fixtures/spec08_active_math_corpus_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "crates/router/tests/fixtures/spec08_active_math_reference_v1.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if mp.__version__ != EXPECTED_MPMATH_VERSION:
        raise RuntimeError(f"expected mpmath {EXPECTED_MPMATH_VERSION}, found {mp.__version__}")
    mp.mp.dps = PRECISION_DECIMAL_DIGITS
    encoded = encoded_fixture(args.corpus, args.simulation_manifest, Path(__file__).resolve())
    if args.check:
        if not args.output.exists() or args.output.read_bytes() != encoded:
            raise SystemExit(f"fixture is stale: {args.output}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded)


if __name__ == "__main__":
    main()
