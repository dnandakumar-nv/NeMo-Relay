# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for bounded Python plugin shutdown."""

import pytest

from nemo_relay import plugin


async def test_clear_async_clears_active_configuration():
    await plugin.initialize(plugin.PluginConfig())

    await plugin.clear_async(timeout=1.0)

    assert plugin.report() is None


@pytest.mark.parametrize("timeout", [-1.0, float("inf"), float("nan"), 1e300])
async def test_clear_async_rejects_invalid_timeout(timeout):
    with pytest.raises(ValueError, match="finite non-negative"):
        await plugin.clear_async(timeout)
