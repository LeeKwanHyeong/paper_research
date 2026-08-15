from __future__ import annotations

import math

import pytest

from paper.scripts.calibrate_count_aware_tail_lambda import calibrate_lambda


def test_calibration_targets_gradient_ratio() -> None:
    raw, frozen = calibrate_lambda(
        2.0,
        0.5,
        target_ratio=0.1,
        minimum=1e-4,
        maximum=100.0,
    )

    assert math.isclose(raw, 0.4)
    assert math.isclose(frozen, 0.4)
    assert math.isclose(frozen * 0.5 / 2.0, 0.1)


def test_calibration_clips_only_to_frozen_bounds() -> None:
    _, low = calibrate_lambda(
        1.0,
        1.0e9,
        target_ratio=0.1,
        minimum=1e-4,
        maximum=100.0,
    )
    _, high = calibrate_lambda(
        1.0,
        1.0e-9,
        target_ratio=0.1,
        minimum=1e-4,
        maximum=100.0,
    )

    assert low == 1e-4
    assert high == 100.0


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
def test_calibration_rejects_invalid_gradient_norm(value: float) -> None:
    with pytest.raises(ValueError):
        calibrate_lambda(
            1.0,
            value,
            target_ratio=0.1,
            minimum=1e-4,
            maximum=100.0,
        )
