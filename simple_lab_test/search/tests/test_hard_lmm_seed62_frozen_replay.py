import ast
from pathlib import Path

import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.hard_lmm_frozen_probe import extract_features
from paper.scripts.run_hard_lmm_seed62_frozen_replay import (
    frozen, indices_for, official_metrics, replay_assessment, traced_outputs,
)
from simple_lab_test.search.common.runner import canonical_state_dict_sha256
from paper.scripts.compare_hard_lmm_seed62_frozen_replay import paired_difference


def test_only_train_and_validation_indices():
    assert len(indices_for(range(393824), "train")) == 256
    assert torch.equal(indices_for(range(86285), "validation"), torch.arange(86285))
    with pytest.raises(ValueError):
        indices_for(range(4), "test")


def test_replay_reports_failure_without_raising_or_relaxing_tolerance():
    keys = ("qty_mae", "qty_rmse", "time_nll", "joint_objective")
    observed = dict.fromkeys(keys, 1.0)
    reference = {f"best_val_{k}": 1.0 for k in keys}
    assert replay_assessment(observed, reference, 1e-5)["all_pass"]
    observed["qty_rmse"] += 2.55e-5
    result = replay_assessment(observed, reference, 1e-5)
    assert not result["all_pass"] and result["tolerance"] == 1e-5
    assert result["absolute_differences"]["qty_rmse"] > 1e-5


def test_native_metrics_keep_raw_prediction_and_double_aggregation():
    events = {"official_prediction": torch.tensor([2., 7.]), "quantity": torch.tensor([1., 3.]),
        "official_time_nll": torch.tensor([-2., -4.]), "official_log_loss": torch.tensor([.1, .3]),
        "official_joint_loss": torch.tensor([-1.9, -3.7])}
    result = official_metrics(events)
    assert result["qty_mae"] == 2.5
    assert result["qty_rmse"] == pytest.approx((17/2)**.5)
    assert result["time_nll"] == -3


def test_tracing_preserves_outputs_weights_and_target_masking():
    torch.manual_seed(62)
    model, _ = build_count_aware_model("titantpp", hidden_dim=64, train_log_mean=1.,
        max_seq_len=16, quantity_variant="count_only_log_regression")
    with torch.no_grad():
        model.quantity_head.weight.normal_(std=.1)
    model.requires_grad_(False).eval()
    digest = canonical_state_dict_sha256(model.state_dict())
    dts, q, mask = torch.ones(2, 8), torch.rand(2, 8)*10, torch.ones(2, 8, dtype=torch.bool)
    native = target_outputs(model, dts, mask, q, lambda_log_qty=1)
    probe = extract_features(model, dts, mask, q)
    a, b = traced_outputs(model,dts,mask,q,"official"), traced_outputs(model,dts,mask,q,"probe")
    assert torch.equal(a["prediction"], native["pred_qty"])
    assert torch.equal(b["z"], probe["z"])
    assert a["prototype_indices"].shape == (2,4)
    changed=q.clone()
    changed[:,-1] += 10000
    assert torch.equal(traced_outputs(model,dts,mask,changed,"probe")["z"], b["z"])
    frozen(model,digest)
    model.requires_grad_(True)
    with pytest.raises(AssertionError):
        frozen(model,digest)


def test_no_training_operations_in_inference_runner():
    path=Path(__file__).resolve().parents[3]/"paper/scripts/run_hard_lmm_seed62_frozen_replay.py"
    tree=ast.parse(path.read_text())
    calls=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
    assert not {"backward", "step", "zero_grad"}.intersection(calls)


def test_comparison_distinguishes_prototype_order_from_membership():
    a={k: torch.ones(2) for k in ("target_index","series_index","context_end","quantity","history_length")}
    a.update(official_prediction=torch.tensor([1.,2.]),official_time_nll=torch.ones(2),
        official_prototype_indices=torch.tensor([[0,1,2,3],[4,5,6,7]]))
    b={k:v.clone() for k,v in a.items()}
    b["official_prototype_indices"][0]=torch.tensor([3,2,1,0])
    b["official_prototype_indices"][1,0]=8
    result=paired_difference(a,b,"official")
    assert result["prototype_order_different_events"]==2
    assert result["prototype_set_different_events"]==1
    b["quantity"][0]+=1
    with pytest.raises(ValueError,match="alignment"):
        paired_difference(a,b,"official")
