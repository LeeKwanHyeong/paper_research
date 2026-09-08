from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paper.scripts.run_hard_lmm_bounded_qk_error_diagnostic import (
    PROJECT_ROOT,
    history_features,
    quantity_metrics,
    validate_contract,
    validate_replay,
)


CONTRACT = PROJECT_ROOT / "paper/contracts/hard_lmm_bounded_qk_error_diagnostic_v1.json"


def test_frozen_contract_rejects_training_and_held_out_access() -> None:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    validate_contract(payload)

    payload["scope"]["training"] = True
    with pytest.raises(ValueError, match="Forbidden operation enabled: training"):
        validate_contract(payload)


def test_history_features_never_need_a_target_or_padding() -> None:
    quantities = np.asarray([2.0, 8.0, 26.0], dtype=np.float64)
    durations = np.asarray([1.0, 3.0, 7.0], dtype=np.float64)
    observed = history_features(quantities, durations)
    logs = np.log1p(quantities)

    assert observed["last_qty"] == 26.0
    assert observed["previous_qty"] == 8.0
    assert observed["history_mean_log_qty"] == pytest.approx(float(logs.mean()))
    assert observed["recent3_mean_log_qty"] == pytest.approx(float(logs.mean()))
    assert observed["last_log_qty_change"] == pytest.approx(float(logs[-1] - logs[-2]))

    # A hypothetical target and left padding are intentionally absent from the
    # function contract; changing either outside this prefix cannot alter it.
    repeated = history_features(quantities.copy(), durations.copy())
    assert observed == repeated


def test_one_event_history_has_defined_recent_statistics() -> None:
    observed = history_features(np.asarray([5.0]), np.asarray([2.0]))
    assert np.isnan(observed["previous_qty"])
    assert np.isnan(observed["last_log_qty_change"])
    assert observed["recent3_minus_history_mean_log_qty"] == pytest.approx(0.0)


def test_raw_and_log_metrics_use_the_same_point_prediction() -> None:
    true = np.asarray([0.0, 4.0, 12.0])
    pred = np.asarray([1.0, 3.0, 9.0])
    observed = quantity_metrics(true, pred)
    error = pred - true
    assert observed["count"] == 3
    assert observed["qty_mae"] == pytest.approx(float(np.abs(error).mean()))
    assert observed["qty_rmse"] == pytest.approx(float(np.sqrt(np.square(error).mean())))
    assert observed["qty_bias"] == pytest.approx(float(error.mean()))
    assert observed["log_qty_mse"] == pytest.approx(
        float(np.square(np.log1p(pred) - np.log1p(true)).mean())
    )


def test_replay_is_fail_closed() -> None:
    expected = {
        "count": 2,
        "qty_mae": 1.0,
        "qty_rmse": 1.0,
        "qty_bias": -1.0,
        "log_qty_mse": 0.1,
    }
    validate_replay(expected, expected, tolerance=1e-5, role="B")
    drifted = dict(expected, qty_rmse=1.001)
    with pytest.raises(ValueError, match="Replay metric drift"):
        validate_replay(drifted, expected, tolerance=1e-5, role="B")
