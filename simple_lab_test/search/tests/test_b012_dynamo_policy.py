from __future__ import annotations

import pytest
import torch._dynamo.config as dynamo_config

from paper.scripts.run_with_b012_dynamo_policy import configure_dynamo


def test_configure_dynamo_applies_bounded_shape_specialization(monkeypatch) -> None:
    monkeypatch.setattr(dynamo_config, "recompile_limit", 8)
    monkeypatch.setattr(dynamo_config, "accumulated_recompile_limit", 256)

    policy = configure_dynamo(recompile_limit=64, accumulated_limit=512)

    assert policy == {
        "recompile_limit": 64,
        "accumulated_recompile_limit": 512,
    }
    assert dynamo_config.recompile_limit == 64
    assert dynamo_config.accumulated_recompile_limit == 512


@pytest.mark.parametrize(
    ("recompile_limit", "accumulated_limit"),
    ((0, 512), (64, 63)),
)
def test_configure_dynamo_rejects_invalid_limits(
    recompile_limit: int,
    accumulated_limit: int,
) -> None:
    with pytest.raises(ValueError):
        configure_dynamo(
            recompile_limit=recompile_limit,
            accumulated_limit=accumulated_limit,
        )

