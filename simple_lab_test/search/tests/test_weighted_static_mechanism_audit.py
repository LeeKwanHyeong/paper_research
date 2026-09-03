"""Checks for the post-stop diagnostic, not a new training configuration."""

import json

import pytest
import torch
from torch.nn import functional as F

from paper.scripts.diagnose_hard_lmm_weighted_static import cosine, describe, history_audit


def test_history_keeps_official_selector_distinct_from_posthoc(tmp_path):
    rows = [
        {"epoch": 1, "val_time_nll": 1., "val_log_qty_mse": .2,
         "val_joint_objective": 1.2, "val_qty_mae": 5., "val_qty_rmse": 8.},
        {"epoch": 2, "val_time_nll": 1.2, "val_log_qty_mse": .1,
         "val_joint_objective": 1.3, "val_qty_mae": 3., "val_qty_rmse": 4.},
    ]
    (tmp_path / "history.json").write_text(json.dumps({"history": rows}))
    (tmp_path / "summary.json").write_text(json.dumps({"best_epoch": 1}))
    audit = history_audit(tmp_path)
    assert audit["views"]["official_joint"]["epoch"] == 1
    assert audit["views"]["posthoc_min_mae"]["epoch"] == 2
    assert audit["views"]["posthoc_min_mae"]["delta_joint_time_component"] == pytest.approx(.2)
    assert audit["views"]["posthoc_min_mae"]["delta_joint_quantity_component"] == pytest.approx(-.1)
    assert audit["posthoc_views_are_not_new_official_results"]
    (tmp_path / "summary.json").write_text(json.dumps({"best_epoch": 2}))
    with pytest.raises(AssertionError):
        history_audit(tmp_path)


def test_near_equal_cosine_scores_produce_near_uniform_weights():
    weights = torch.tensor([[.51, .50, .49, .48]]).softmax(-1)
    assert weights.max() < .254
    assert weights.min() > .246


def test_additive_memory_is_multiplicative_for_current_quantity_head():
    z = torch.linspace(-8, 8, 100, dtype=torch.float64)
    shift = torch.linspace(-.2, .2, 100, dtype=torch.float64)
    actual = F.softplus(z + shift).expm1() / F.softplus(z).expm1() - 1
    torch.testing.assert_close(actual, shift.expm1(), rtol=1e-10, atol=1e-10)


def test_gradient_cosine_zero_is_not_reported_as_no_conflict():
    assert cosine(torch.tensor([1., 0.]), torch.tensor([-1., 0.])) == -1
    assert cosine(torch.zeros(2), torch.ones(2)) is None


def test_nonfinite_measurements_are_rejected():
    with pytest.raises(AssertionError):
        describe(torch.tensor([float("nan")]))
