# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for bundled Router registration and typed configuration helpers."""

from __future__ import annotations

from typing import Any, cast

import pytest

from nemo_relay import _native, plugin, router


def judge_config() -> router.JudgeConfig:
    """Return a complete deterministic judge policy for tests."""
    return router.JudgeConfig(
        version=1,
        model="judge-model",
        model_revision="2026-07-11",
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
        temperature=1.0,
    )


def shadow_config(database_path: str) -> router.RouterConfig:
    """Return a complete provider-free Shadow configuration for tests."""
    return router.RouterConfig(
        mode="shadow",
        project_id="python-router-test",
        database_path=database_path,
        embedders=[
            router.EmbedderConfig(
                id="embedding-main",
                base_url="https://127.0.0.1:9443/v1",
                model="embedding-model",
                provider_revision="2026-07-11",
                dimensions=3,
                timeout_ms=1000,
            )
        ],
        pools=[
            router.PoolConfig(
                id="python-pool",
                api_family="openai_chat_completions",
                anchor_models=["anchor-model"],
                anchor_revision="2026-07-11",
                sampling_probability=1.0,
                max_candidates_per_sample=1,
                concurrency=router.ConcurrencyConfig(shadow=1, judge=1, max_pending=2),
                candidates=[
                    router.CandidateConfig(
                        id="candidate",
                        model="candidate-model",
                        model_revision="2026-07-11",
                        cost_rank=0,
                        capabilities=router.CandidateCapabilities(tools=True),
                    )
                ],
                judge=judge_config(),
                selector=router.PoolSelectorConfig(tenant_ids=["tenant-a"]),
                learning=router.LearningConfig(embedder="embedding-main"),
            )
        ],
    )


def test_router_defaults_and_nested_serialization_match_canonical_shape(tmp_path):
    config = shadow_config(str(tmp_path / "router.sqlite3"))
    serialized = cast(dict[str, Any], config.to_dict())

    assert router.RouterConfig().to_dict() == {
        "version": 1,
        "mode": "off",
        "database_path": ".nemo-relay/router/router.db",
        "retention_days": 30,
        "max_evidence_records": 100_000,
        "allow_remote_embedding_egress": False,
        "embedders": [],
        "pools": [],
        "policy": {
            "unknown_component": "warn",
            "unknown_field": "warn",
            "unsupported_value": "error",
        },
    }
    assert serialized["embedders"][0]["max_in_flight"] == 4
    assert serialized["embedders"][0]["batch_size"] == 16
    assert serialized["pools"][0]["selector"] == {
        "tenant_ids": ["tenant-a"],
        "metadata_equals": {},
    }
    assert serialized["pools"][0]["lookahead"]["deadline_seconds"] == 300
    assert serialized["pools"][0]["canonicalizer"]["max_context_messages"] == 8
    assert serialized["pools"][0]["judge"]["temperature"] == 1.0
    assert serialized["pools"][0]["candidates"][0]["capabilities"]["tools"] is True
    assert serialized["pools"][0]["learning"] == {
        "embedder": "embedding-main",
        "version": 1,
    }
    assert serialized["pools"][0]["outcome"] == {}
    assert "project_id" not in router.RouterConfig().to_dict()

    component = router.ComponentSpec(config, enabled=False).to_dict()
    assert component["kind"] == "router"
    assert component["enabled"] is False
    assert component["config"] == serialized


def test_router_reexports_existing_v2_replay_types():
    from nemo_relay import llm

    assert router.LlmExecutionContext is llm.LlmExecutionContext
    assert router.LlmReplayDescriptor is llm.LlmReplayDescriptor
    assert router.LlmReplayFactory is llm.LlmReplayFactory


def test_python_native_artifact_passes_real_vector_probe():
    sqlite_version, vector_version = _native._router_native_vector_probe()

    assert sqlite_version
    assert vector_version == "v0.1.9"


@pytest.mark.asyncio
async def test_router_is_registered_validates_and_activates_shadow_without_provider_io(tmp_path):
    config = shadow_config(str(tmp_path / "router.sqlite3"))
    assert router.ROUTER_PLUGIN_KIND in plugin.list_kinds()
    assert router.validate_config(config)["diagnostics"] == []

    try:
        report = await plugin.initialize(plugin.PluginConfig(components=[router.ComponentSpec(config)]))
        assert report["diagnostics"] == []
        assert (tmp_path / "router.sqlite3").is_file()
    finally:
        await plugin.clear_async(5.0)


def test_router_validation_uses_native_mode_specific_diagnostics(tmp_path):
    config = shadow_config(str(tmp_path / "router.sqlite3"))
    config.mode = "active"
    diagnostics = router.validate_config(config)["diagnostics"]

    assert any(
        diagnostic["level"] == "error" and "outcome" in diagnostic.get("field", "") for diagnostic in diagnostics
    )
