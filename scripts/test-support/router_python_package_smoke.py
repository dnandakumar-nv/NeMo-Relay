# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Clean-install smoke test for a Router-bearing NeMo Relay Python wheel."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import tempfile
import venv
from pathlib import Path


def _judge_config(router):
    return router.JudgeConfig(
        version=1,
        model="judge-model",
        model_revision="package-smoke-v1",
        prompt_version="pairwise-equivalence-v1",
        rubric_version="response-trajectory-equivalence-v1",
        output_schema_version=1,
        response_weight=0.5,
        trajectory_weight=0.5,
        response_floor=0.8,
        trajectory_floor=0.8,
        judge_confidence_floor=0.7,
        pass_threshold=0.85,
        max_rationale_bytes=4096,
        base_cooloff_seconds=10,
        max_cooloff_seconds=300,
    )


async def _installed_smoke(source_root: Path) -> None:
    import nemo_relay
    from nemo_relay import _native, plugin, router

    installed_path = Path(nemo_relay.__file__).resolve()
    if installed_path.is_relative_to(source_root):
        raise RuntimeError(f"wheel resolved from source tree: {installed_path}")
    sqlite_version, vector_version = _native._router_native_vector_probe()
    if not sqlite_version or vector_version != "v0.1.9":
        raise RuntimeError(f"unexpected native versions: {sqlite_version}, {vector_version}")

    database_path = Path.cwd() / "router.sqlite3"
    config = router.RouterConfig(
        mode="shadow",
        project_id="python-package-smoke",
        database_path=str(database_path),
        pools=[
            router.PoolConfig(
                id="python-package-pool",
                api_family="openai_chat_completions",
                anchor_models=["anchor-model"],
                anchor_revision="package-smoke-v1",
                sampling_probability=1.0,
                max_candidates_per_sample=1,
                concurrency=router.ConcurrencyConfig(shadow=1, judge=1, max_pending=2),
                candidates=[
                    router.CandidateConfig(
                        id="candidate",
                        model="candidate-model",
                        model_revision="package-smoke-v1",
                        cost_rank=0,
                    )
                ],
                judge=_judge_config(router),
            )
        ],
    )
    if router.ROUTER_PLUGIN_KIND not in plugin.list_kinds():
        raise RuntimeError("Router component is not registered in the installed wheel")
    if router.validate_config(config)["diagnostics"]:
        raise RuntimeError("installed Router helper config did not validate")
    try:
        report = await plugin.initialize(plugin.PluginConfig(components=[router.ComponentSpec(config)]))
        if report["diagnostics"]:
            raise RuntimeError(f"unexpected activation diagnostics: {report['diagnostics']}")
        if not database_path.is_file():
            raise RuntimeError("installed wheel did not initialize the Router ledger")
    finally:
        await plugin.clear_async(5000.0)


def _venv_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel", nargs="?", type=Path)
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()

    if args.installed:
        if args.source_root is None:
            raise SystemExit("--source-root is required with --installed")
        asyncio.run(_installed_smoke(args.source_root.resolve()))
        return
    if args.wheel is None:
        raise SystemExit("wheel path is required")

    source_root = Path(__file__).resolve().parents[2]
    wheel = args.wheel.resolve()
    if not wheel.is_file():
        raise SystemExit(f"wheel does not exist: {wheel}")
    with tempfile.TemporaryDirectory(prefix="nemo-relay-router-wheel-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        venv.EnvBuilder(with_pip=True, clear=True).create(environment)
        python = _venv_python(environment)
        clean_env = os.environ.copy()
        clean_env.pop("PYTHONPATH", None)
        clean_env["PYTHONNOUSERSITE"] = "1"
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                str(wheel),
            ],
            cwd=root,
            env=clean_env,
            check=True,
            timeout=300,
        )
        subprocess.run(
            [
                str(python),
                str(Path(__file__).resolve()),
                "--installed",
                "--source-root",
                str(source_root),
            ],
            cwd=root,
            env=clean_env,
            check=True,
            timeout=120,
        )


if __name__ == "__main__":
    main()
