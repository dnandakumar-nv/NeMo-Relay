# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clean-artifact smoke test for a Router-bearing NeMo Relay CLI binary."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def _plugin_toml(database_path: Path) -> str:
    database = json.dumps(str(database_path))
    return f"""version = 1

[[components]]
kind = "router"
enabled = true

[components.config]
version = 1
mode = "shadow"
project_id = "cli-package-smoke"
database_path = {database}

[[components.config.pools]]
id = "cli-package-pool"
api_family = "openai_chat_completions"
anchor_models = ["anchor-model"]
anchor_revision = "package-smoke-v1"
sampling_probability = 1.0
max_candidates_per_sample = 1

[components.config.pools.selector]
tenant_ids = ["tenant-a"]

[components.config.pools.concurrency]
shadow = 1
judge = 1
max_pending = 2

[components.config.pools.judge]
version = 1
model = "judge-model"
model_revision = "package-smoke-v1"
prompt_version = "pairwise-equivalence-v1"
rubric_version = "response-trajectory-equivalence-v1"
output_schema_version = 1
response_weight = 0.5
trajectory_weight = 0.5
response_floor = 0.8
trajectory_floor = 0.8
judge_confidence_floor = 0.7
pass_threshold = 0.85
max_rationale_bytes = 4096
base_cooloff_seconds = 10
max_cooloff_seconds = 300

[[components.config.pools.candidates]]
id = "candidate"
model = "candidate-model"
model_revision = "package-smoke-v1"
cost_rank = 0

[[components.config.pools]]
id = "cli-package-pool-b"
api_family = "openai_chat_completions"
anchor_models = ["anchor-model"]
anchor_revision = "package-smoke-v1"
sampling_probability = 1.0
max_candidates_per_sample = 1

[components.config.pools.selector]
tenant_ids = ["tenant-b"]

[components.config.pools.concurrency]
shadow = 1
judge = 1
max_pending = 2

[components.config.pools.judge]
version = 1
model = "judge-model"
model_revision = "package-smoke-v1"
prompt_version = "pairwise-equivalence-v1"
rubric_version = "response-trajectory-equivalence-v1"
output_schema_version = 1
response_weight = 0.5
trajectory_weight = 0.5
response_floor = 0.8
trajectory_floor = 0.8
judge_confidence_floor = 0.7
pass_threshold = 0.85
max_rationale_bytes = 4096
base_cooloff_seconds = 10
max_cooloff_seconds = 300

[[components.config.pools.candidates]]
id = "candidate-b"
model = "candidate-model-b"
model_revision = "package-smoke-v1"
cost_rank = 0
"""


def _partition() -> dict[str, object]:
    return {
        "tenant_policy_hash": "1" * 64,
        "agent_policy_hash": "2" * 64,
        "policy_version_id": "3" * 64,
        "learning_generation_id": "018f47b8-0000-7000-8000-000000000001",
        "api_family": "openai_chat_completions",
        "transport_identity": "transport-v1",
        "anchor_model": "anchor-model",
        "anchor_revision": "package-smoke-v1",
        "candidate_id": "candidate",
        "candidate_model": "candidate-model",
        "candidate_model_revision": "package-smoke-v1",
        "decoding_fingerprint": "4" * 64,
        "evaluator_version": "5" * 64,
        "vector_space_id": "6" * 64,
    }


def _inspection_request() -> dict[str, object]:
    return {
        "schema": "nemo.relay.router.inspection-input@1",
        "pool_id": "cli-package-pool",
        "partition": _partition(),
        "request": {
            "schema": "nemo.relay.router.request-projection@1",
            "family": "openai_chat_completions",
            "normalized_request": {
                "messages": [{"role": "user", "content": "hello", "name": None}],
                "model": "anchor-model",
                "params": None,
                "tools": None,
                "tool_choice": None,
                "response_format": None,
                "truncation": None,
                "reasoning": None,
                "service_tier": None,
                "parallel_tool_calls": None,
                "max_output_tokens": None,
                "max_tool_calls": None,
                "top_logprobs": None,
            },
            "ordered_instructions": [],
            "response_format": None,
            "response_schema_fingerprint": None,
            "required_capabilities": [],
            "sanitizer_version": 1,
            "semantic_request_fingerprint": "7" * 64,
        },
        "routing_context": {
            "schema": "nemo.relay.router.routing-context@1",
            "tenant_policy_hash": "1" * 64,
            "agent_policy_hash": "2" * 64,
            "position_features": {},
        },
    }


def _expect_lookup_refusal(result: subprocess.CompletedProcess[str], lookup: str) -> None:
    if result.returncode != 2 or result.stderr:
        raise RuntimeError(f"CLI Router {lookup} stdin probe returned {result.returncode}: {result.stderr}")
    report = json.loads(result.stdout)
    if report.get("error", {}).get("code") not in {"invalid_argument", "not_found"}:
        raise RuntimeError(f"CLI Router {lookup} stdin probe failed: {report}")


def _run_json(
    binary: Path,
    root: Path,
    environment: dict[str, str],
    arguments: list[str],
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    result = subprocess.run(
        [str(binary), *arguments],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.stderr:
        raise RuntimeError(f"CLI Router {' '.join(arguments)} wrote stderr: {result.stderr}")
    try:
        report = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"CLI Router {' '.join(arguments)} returned invalid JSON: {result.stdout}") from error
    return result, report


def _expect_success(
    result: subprocess.CompletedProcess[str],
    report: dict[str, Any],
    command: str,
) -> dict[str, Any]:
    if result.returncode != 0 or report.get("schema_version") != 1 or not report.get("ok"):
        raise RuntimeError(f"CLI Router {command} failed: {result.returncode}: {report}")
    data = report.get("data")
    if not isinstance(data, dict):
        raise RuntimeError(f"CLI Router {command} omitted its data object: {report}")
    return data


def _expect_uuid_v7(value: object, label: str) -> None:
    try:
        identifier = uuid.UUID(str(value))
    except ValueError as error:
        raise RuntimeError(f"CLI Router {label} is not a UUID: {value}") from error
    if identifier.version != 7:
        raise RuntimeError(f"CLI Router {label} is not UUIDv7: {value}")


def _run_concurrent_json(
    binary: Path,
    root: Path,
    environment: dict[str, str],
    commands: list[list[str]],
) -> list[tuple[int, dict[str, Any]]]:
    children: list[subprocess.Popen[str]] = []
    try:
        for command in commands:
            children.append(
                subprocess.Popen(
                    [str(binary), *command],
                    cwd=root,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            )
        results = []
        for command, child in zip(commands, children, strict=True):
            stdout, stderr = child.communicate(timeout=120)
            if stderr:
                raise RuntimeError(f"concurrent CLI Router {' '.join(command)} wrote stderr: {stderr}")
            try:
                report = json.loads(stdout)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"concurrent CLI Router {' '.join(command)} returned {child.returncode} with invalid JSON: {stdout}"
                ) from error
            results.append((child.returncode, report))
        return results
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=10)


def _pool_generations(data: dict[str, Any]) -> dict[str, str]:
    items = data.get("items")
    if not isinstance(items, list):
        raise RuntimeError(f"CLI Router pools omitted items: {data}")
    return {str(item["id"]): str(item["learning_generation_id"]) for item in items if isinstance(item, dict)}


def _run_mutation_smoke(binary: Path, root: Path, environment: dict[str, str]) -> None:
    result, report = _run_json(
        binary,
        root,
        environment,
        ["router", "pause", "--reason", "package-pause", "--json"],
    )
    paused = _expect_success(result, report, "pause")
    _expect_uuid_v7(paused.get("mutation_id"), "pause mutation ID")
    if paused.get("snapshot", {}).get("all") != {"force_anchor": False, "paused": True}:
        raise RuntimeError(f"CLI Router pause changed the wrong control fields: {paused}")

    result, report = _run_json(
        binary,
        root,
        environment,
        [
            "router",
            "force-anchor",
            "set",
            "--pool",
            "cli-package-pool",
            "--reason",
            "package-force",
            "--actor",
            "package-operator",
            "--json",
        ],
    )
    forced = _expect_success(result, report, "force-anchor set")
    effective = forced.get("snapshot", {}).get("pools", {}).get("cli-package-pool", {}).get("effective")
    if effective != {"force_anchor": True, "paused": True}:
        raise RuntimeError(f"CLI Router force-anchor changed the wrong fields: {forced}")

    result, report = _run_json(
        binary,
        root,
        environment,
        ["router", "resume", "--reason", "package-resume", "--json"],
    )
    resumed = _expect_success(result, report, "resume")
    if resumed.get("snapshot", {}).get("all", {}).get("paused") is not False:
        raise RuntimeError(f"CLI Router resume did not clear pause: {resumed}")
    if (
        resumed.get("snapshot", {})
        .get("pools", {})
        .get("cli-package-pool", {})
        .get("effective", {})
        .get("force_anchor")
        is not True
    ):
        raise RuntimeError(f"CLI Router resume cleared force-anchor: {resumed}")

    result, report = _run_json(
        binary,
        root,
        environment,
        [
            "router",
            "force-anchor",
            "clear",
            "--pool",
            "cli-package-pool",
            "--reason",
            "package-clear",
            "--json",
        ],
    )
    cleared = _expect_success(result, report, "force-anchor clear")
    effective = cleared.get("snapshot", {}).get("pools", {}).get("cli-package-pool", {}).get("effective")
    if effective != {"force_anchor": False, "paused": False}:
        raise RuntimeError(f"CLI Router force-anchor clear changed the wrong fields: {cleared}")

    result, refused = _run_json(
        binary,
        root,
        environment,
        [
            "router",
            "reset",
            "--pool",
            "cli-package-pool",
            "--confirm",
            "wrong-project",
            "--reason",
            "package-refusal",
            "--json",
        ],
    )
    if result.returncode != 2 or refused.get("error", {}).get("code") != "invalid_argument":
        raise RuntimeError(f"CLI Router confirmation refusal failed: {result.returncode}: {refused}")
    if str(root) in result.stdout:
        raise RuntimeError("CLI Router confirmation refusal exposed its database path")

    result, report = _run_json(
        binary,
        root,
        environment,
        [
            "router",
            "reset",
            "--pool",
            "cli-package-pool",
            "--confirm",
            "cli-package-smoke",
            "--reason",
            "package-pool-reset",
            "--json",
        ],
    )
    pool_reset = _expect_success(result, report, "pool reset")
    if set(pool_reset.get("resulting_generations", {})) != {"cli-package-pool"}:
        raise RuntimeError(f"CLI Router pool reset changed the wrong scope: {pool_reset}")

    result, report = _run_json(
        binary,
        root,
        environment,
        [
            "router",
            "cohort",
            "rotate",
            "--confirm",
            "cli-package-smoke",
            "--reason",
            "package-cohort",
            "--json",
        ],
    )
    rotation = _expect_success(result, report, "cohort rotate")
    _expect_uuid_v7(rotation.get("mutation_id"), "cohort mutation ID")
    if set(rotation.get("resulting_generations", {})) != {"cohort"}:
        raise RuntimeError(f"CLI Router cohort rotation returned wrong generations: {rotation}")

    result, report = _run_json(binary, root, environment, ["router", "status", "--json"])
    before_status = _expect_success(result, report, "pre-concurrency status")
    before_control_generation = before_status.get("controls", {}).get("control_generation")
    if not isinstance(before_control_generation, int):
        raise RuntimeError(f"CLI Router status omitted control generation: {before_status}")
    control_results = _run_concurrent_json(
        binary,
        root,
        environment,
        [
            [
                "router",
                "pause",
                "--reason",
                f"concurrent-pause-{index}",
                "--json",
            ]
            for index in range(2)
        ],
    )
    if not any(returncode == 0 and report.get("ok") for returncode, report in control_results):
        raise RuntimeError(f"concurrent CLI Router pause had no winner: {control_results}")
    for returncode, outcome in control_results:
        if returncode == 0 and outcome.get("ok"):
            continue
        code = outcome.get("error", {}).get("code")
        if code not in {"busy", "conflict", "storage_unavailable"}:
            raise RuntimeError(f"concurrent CLI Router pause failed unexpectedly: {control_results}")
        expected_exit = 2 if code == "conflict" else 1
        if returncode != expected_exit:
            raise RuntimeError(f"concurrent CLI Router pause used the wrong exit: {outcome}")
    result, report = _run_json(binary, root, environment, ["router", "status", "--json"])
    after_status = _expect_success(result, report, "post-concurrency status")
    if after_status.get("controls", {}).get("all", {}).get("paused") is not True:
        raise RuntimeError(f"concurrent CLI Router pause was not visible: {after_status}")
    if after_status.get("controls", {}).get("control_generation") != before_control_generation + 1:
        raise RuntimeError(f"concurrent CLI Router pause advanced more than once: {before_status}: {after_status}")

    result, report = _run_json(
        binary,
        root,
        environment,
        ["router", "resume", "--reason", "post-concurrency-resume", "--json"],
    )
    _expect_success(result, report, "post-concurrency resume")

    reset_results = _run_concurrent_json(
        binary,
        root,
        environment,
        [
            [
                "router",
                "reset",
                "--all",
                "--confirm",
                "cli-package-smoke",
                "--reason",
                f"concurrent-reset-{index}",
                "--json",
            ]
            for index in range(2)
        ],
    )
    reset_generations = []
    for returncode, outcome in reset_results:
        if returncode == 0 and outcome.get("ok"):
            generations = outcome.get("data", {}).get("resulting_generations", {})
            if set(generations) != {"cli-package-pool", "cli-package-pool-b"}:
                raise RuntimeError(f"concurrent CLI Router reset was partial: {outcome}")
            reset_generations.append({str(key): str(value) for key, value in generations.items()})
            continue
        code = outcome.get("error", {}).get("code")
        if code not in {"busy", "conflict", "storage_unavailable"}:
            raise RuntimeError(f"concurrent CLI Router reset failed unexpectedly: {outcome}")
        expected_exit = 2 if code == "conflict" else 1
        if returncode != expected_exit:
            raise RuntimeError(f"concurrent CLI Router reset used the wrong exit: {outcome}")
    if not reset_generations:
        raise RuntimeError(f"concurrent CLI Router reset had no winner: {reset_results}")
    result, report = _run_json(
        binary,
        root,
        environment,
        ["router", "pools", "--limit", "10", "--json"],
    )
    final_pools = _pool_generations(_expect_success(result, report, "post-reset pools"))
    if final_pools not in reset_generations:
        raise RuntimeError(f"concurrent CLI Router reset exposed a partial generation set: {final_pools}")


def _run_dashboard_smoke(
    binary: Path,
    root: Path,
    environment: dict[str, str],
    database_path: Path,
) -> None:
    token_path = root / "dashboard.url"
    creation_flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    process = subprocess.Popen(
        [
            str(binary),
            "router",
            "dashboard",
            "--no-open",
            "--token-file",
            str(token_path),
            "--idle-timeout",
            "60",
        ],
        cwd=root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creation_flags,
    )
    before = hashlib.sha256(database_path.read_bytes()).hexdigest()
    output = ""
    errors = ""
    try:
        deadline = time.monotonic() + 15
        launch_url = ""
        while time.monotonic() < deadline:
            if process.poll() is not None:
                output, errors = process.communicate()
                raise RuntimeError(
                    f"CLI Router dashboard exited before launch: {process.returncode}: {output}: {errors}"
                )
            try:
                launch_url = token_path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                launch_url = ""
            if launch_url:
                break
            time.sleep(0.025)
        if not launch_url:
            raise RuntimeError("CLI Router dashboard did not create its token file")

        parsed = urllib.parse.urlsplit(launch_url)
        parameters = urllib.parse.parse_qs(parsed.fragment, strict_parsing=True)
        nonce = parameters.get("bootstrap", [None])
        if len(nonce) != 1 or nonce[0] is None:
            raise RuntimeError("CLI Router dashboard launch URL had no bootstrap value")
        origin = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        bootstrap = urllib.request.Request(
            f"{origin}/auth/bootstrap",
            data=b"",
            headers={
                "Authorization": f"Bootstrap {nonce[0]}",
                "Origin": origin,
            },
            method="POST",
        )
        with opener.open(bootstrap, timeout=10) as response:
            if response.status != 204:
                raise RuntimeError(f"CLI Router dashboard bootstrap returned {response.status}")
            set_cookie = response.headers.get("Set-Cookie", "")
        if not set_cookie or "HttpOnly" not in set_cookie or "SameSite=Strict" not in set_cookie:
            raise RuntimeError("CLI Router dashboard bootstrap omitted session cookie flags")
        if token_path.exists():
            raise RuntimeError("CLI Router dashboard did not remove its used token file")

        for route, content_type, marker in [
            ("/", "text/html; charset=utf-8", b"NeMo Relay Router"),
            ("/app.js", "text/javascript; charset=utf-8", b"window.history.replaceState"),
            ("/styles.css", "text/css; charset=utf-8", b".view-tabs"),
        ]:
            with opener.open(f"{origin}{route}", timeout=10) as response:
                body = response.read()
                if response.headers.get("Content-Type") != content_type or marker not in body:
                    raise RuntimeError(f"CLI Router dashboard asset was invalid: {route}")

        status_request = urllib.request.Request(
            f"{origin}/api/router/v1/status",
            headers={"Cookie": set_cookie.split(";", 1)[0]},
        )
        with opener.open(status_request, timeout=10) as response:
            status = json.load(response)
        if (
            status.get("schema") != "nemo.relay.router.status@1"
            or status.get("project_id") != "cli-package-smoke"
            or status.get("database", {}).get("state") != "current"
        ):
            raise RuntimeError(f"CLI Router dashboard status failed: {status}")

        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
        output, errors = process.communicate(timeout=15)
        if process.returncode != 0:
            raise RuntimeError(f"CLI Router dashboard shutdown failed: {process.returncode}: {output}: {errors}")
        if token_path.exists():
            raise RuntimeError("CLI Router dashboard left its token file after shutdown")
        after = hashlib.sha256(database_path.read_bytes()).hexdigest()
        if after != before:
            raise RuntimeError("CLI Router dashboard changed the Router database")
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path)
    args = parser.parse_args()
    binary = args.binary.resolve()
    if not binary.is_file():
        raise SystemExit(f"CLI binary does not exist: {binary}")

    with tempfile.TemporaryDirectory(prefix="nemo-relay-router-cli-package-") as temporary:
        root = Path(temporary)
        database_path = root / "router.sqlite3"
        plugin_config = root / "plugins.toml"
        plugin_config.write_text(_plugin_toml(database_path), encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "HOME": str(root),
                "XDG_CONFIG_HOME": str(root / "xdg"),
                "APPDATA": str(root / "appdata"),
                "LOCALAPPDATA": str(root / "localappdata"),
                "NEMO_RELAY_PLUGIN_CONFIG_PATH": str(plugin_config),
            }
        )
        for key in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "INFERENCE_API_KEY"]:
            environment.pop(key, None)

        subprocess.run(
            [str(binary), "router-package-probe"],
            cwd=root,
            env=environment,
            check=True,
            timeout=120,
        )
        if not database_path.is_file():
            raise RuntimeError("CLI package probe did not initialize the Router ledger")
        status = subprocess.run(
            [str(binary), "router", "status", "--json"],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        status_report = json.loads(status.stdout)
        if (
            status_report.get("schema_version") != 1
            or not status_report.get("ok")
            or status_report.get("data", {}).get("project_id") != "cli-package-smoke"
            or status_report.get("data", {}).get("database", {}).get("state") != "current"
        ):
            raise RuntimeError(f"CLI Router status failed: {status_report}")
        pools = subprocess.run(
            [str(binary), "router", "pools", "--limit", "1", "--json"],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        pool_report = json.loads(pools.stdout)
        items = pool_report.get("data", {}).get("items", [])
        if (
            pool_report.get("schema_version") != 1
            or not pool_report.get("ok")
            or len(items) != 1
            or items[0].get("id") != "cli-package-pool"
        ):
            raise RuntimeError(f"CLI Router pool read failed: {pool_report}")
        export_path = root / "evidence.jsonl"
        export = subprocess.run(
            [str(binary), "router", "evidence", "export", str(export_path)],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if export.returncode != 0 or export.stderr or not export_path.is_file():
            raise RuntimeError(
                f"CLI Router evidence export failed: {export.returncode}: {export.stdout}: {export.stderr}"
            )
        if export_path.read_bytes():
            raise RuntimeError("CLI Router empty-ledger JSONL export was not empty")
        partition_lookup = subprocess.run(
            [
                str(binary),
                "router",
                "neighborhood",
                "inspect",
                "--query-hash",
                "a" * 64,
                "--partition-file",
                "-",
                "--json",
            ],
            cwd=root,
            env=environment,
            input=json.dumps(_partition()),
            capture_output=True,
            text=True,
            timeout=120,
        )
        _expect_lookup_refusal(partition_lookup, "partition")
        request_lookup = subprocess.run(
            [
                str(binary),
                "router",
                "neighborhood",
                "inspect",
                "--request-file",
                "-",
                "--json",
            ],
            cwd=root,
            env=environment,
            input=json.dumps(_inspection_request()),
            capture_output=True,
            text=True,
            timeout=120,
        )
        _expect_lookup_refusal(request_lookup, "request")
        _run_mutation_smoke(binary, root, environment)
        doctor = subprocess.run(
            [str(binary), "doctor", "--json"],
            cwd=root,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
        report = json.loads(doctor.stdout)
        checks = {check["name"]: check for check in report["observability"]}
        for name in [
            "Router registration",
            "Router native vector",
            "Router V2 bridge",
            "Router database schema",
        ]:
            if checks.get(name, {}).get("status") != "pass":
                raise RuntimeError(f"CLI doctor check failed for {name}: {checks.get(name)}")
        if "temporary vec0 insert/query passed" not in checks["Router native vector"]["details"]:
            raise RuntimeError("CLI doctor did not execute the native vector probe")
        if "current" not in checks["Router database schema"]["details"]:
            raise RuntimeError("CLI doctor did not observe the activated current schema")
        _run_dashboard_smoke(binary, root, environment, database_path)


if __name__ == "__main__":
    main()
