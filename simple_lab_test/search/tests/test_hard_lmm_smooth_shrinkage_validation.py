import copy

import polars as pl
import pytest

from paper.scripts.hard_lmm_frozen_probe import predict, summarize, write_event_deltas
from paper.scripts.run_hard_lmm_smooth_shrinkage import fit_gate, new_gate
from paper.scripts.validate_hard_lmm_smooth_shrinkage import (
    check_history, independent_gate_decision, reconcile_events,
)
from simple_lab_test.search.tests.test_hard_lmm_smooth_shrinkage import POLICY, cache


def test_history_rejects_wrong_selection_and_early_stopping():
    _, _, result = fit_gate("smooth_shrinkage", cache(), cache(), POLICY, lambda row: None)
    assert check_history(result, POLICY)["epoch"] == result["best_epoch"]
    wrong = copy.deepcopy(result)
    wrong["best_epoch"] = 0
    with pytest.raises(AssertionError):
        check_history(wrong, POLICY)
    short = copy.deepcopy(result)
    short["history"] = short["history"][:-1]
    short["completed_epochs"] -= 1
    short["best_epoch"] -= 1
    with pytest.raises(AssertionError):
        check_history(short, POLICY)


def test_event_reaggregation_rejects_metric_and_key_corruption(tmp_path):
    data = cache(2.)
    model = new_gate(3, "smooth_shrinkage")
    replay = predict(model, data)
    boundaries = [1., 2., 3., 4.]
    scopes = summarize(data, *replay, boundaries)
    path = tmp_path / "events.parquet"
    write_event_deltas(path, data, *replay)
    events = pl.read_parquet(path)
    assert reconcile_events(events, data, replay, scopes, boundaries) < 1e-12
    wrong = copy.deepcopy(scopes)
    wrong["overall"]["candidate"]["qty_mae"] += 1
    with pytest.raises(AssertionError):
        reconcile_events(events, data, replay, wrong, boundaries)
    wrong_events = events.with_columns((pl.col("target_index") + 1).alias("target_index"))
    with pytest.raises(AssertionError):
        reconcile_events(wrong_events, data, replay, scopes, boundaries)


def test_independent_gate_decision_rejects_relaxed_acceptance():
    scopes = {name: {"baseline": {"qty_mae": 10., "qty_rmse": 10., "time_nll": 2.},
        "candidate": {"qty_mae": 9.6, "qty_rmse": 10., "time_nll": 2.}}
        for name in ("overall", "body_le_p95", "gt_p99")}
    with pytest.raises(AssertionError):
        independent_gate_decision({"scopes": scopes, "best_epoch": 1,
            "decision": {"status": "exploratory_pass"}})
