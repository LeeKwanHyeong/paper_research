import math

import pytest
import torch

from paper.scripts.compare_count_aware_lognormal_k1_screening import evaluate_gate
from paper.scripts.run_count_aware_tpp_backbone_control import (
    BACKBONES,
    LOGNORMAL_VARIANT,
    VARIANT,
    build_model,
    normalize_quantity_variants,
    target_outputs,
)


def build_pair(backbone: str = "rmtpp"):
    kwargs = {
        "hidden_dim": 16,
        "train_log_mean": 1.5,
        "train_log_std": 0.7,
        "max_seq_len": 8,
    }
    torch.manual_seed(17)
    control, _ = build_model(backbone, quantity_variant=VARIANT, **kwargs)
    torch.manual_seed(17)
    candidate, _ = build_model(backbone, quantity_variant=LOGNORMAL_VARIANT, **kwargs)
    return control, candidate


def sample_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dts = torch.tensor([[0.0, 0.0, 1.0, 2.0], [0.0, 1.0, 1.0, 3.0]])
    mask = torch.tensor([[False, False, True, True], [False, True, True, True]])
    quantities = torch.tensor([[0.0, 0.0, 2.0, 5.0], [0.0, 3.0, 4.0, 6.0]])
    return dts, mask, quantities


def test_variant_aliases_are_explicit_and_unique() -> None:
    assert normalize_quantity_variants("log_mse,lognormal_k1") == (
        VARIANT,
        LOGNORMAL_VARIANT,
    )
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_quantity_variants(f"log_mse,{VARIANT}")


@pytest.mark.parametrize("backbone", BACKBONES)
def test_k1_forward_is_finite_positive_and_uses_one_point_prediction(backbone: str) -> None:
    _, model = build_pair(backbone)
    model.eval()
    dts, mask, quantities = sample_batch()

    outputs = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)

    for key, value in outputs.items():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all(), key
    assert torch.all(outputs["quantity_scale"] > 1e-3)
    assert torch.all(outputs["pred_qty"] >= 0.0)
    assert torch.allclose(
        outputs["joint_loss"],
        outputs["time_loss"] + outputs["quantity_train_loss"],
    )


def test_k1_adds_only_scale_head_at_common_initialization() -> None:
    control, candidate = build_pair()
    control_state = control.state_dict()
    candidate_state = candidate.state_dict()

    assert set(candidate_state) - set(control_state) == {
        "quantity_scale_head.weight",
        "quantity_scale_head.bias",
    }
    for name, value in control_state.items():
        assert torch.equal(value, candidate_state[name]), name


def test_k1_quantity_loss_routes_gradients_to_location_scale_and_hidden() -> None:
    _, model = build_pair()
    hidden = torch.randn(5, model.hidden_dim, requires_grad=True)
    target = torch.tensor([0.0, 1.0, 2.0, 10.0, 200.0])

    quantity = model.quantity_outputs(hidden, target)
    quantity["train_loss"].mean().backward()

    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert model.quantity_head.weight.grad is not None
    assert torch.isfinite(model.quantity_head.weight.grad).all()
    assert torch.count_nonzero(model.quantity_head.weight.grad) > 0
    assert model.quantity_scale_head.weight.grad is not None
    assert torch.isfinite(model.quantity_scale_head.weight.grad).all()
    assert torch.count_nonzero(model.quantity_scale_head.weight.grad) > 0
    assert model.v_t.weight.grad is None


def test_gate_uses_frozen_thresholds() -> None:
    control = {
        "qty_mae": 1.0,
        "qty_rmse": 2.0,
        "p99_qty_mae": 10.0,
        "time_nll": -3.5,
        "quantity_scale_mean": 0.0,
    }
    passing = {
        "qty_mae": 0.95,
        "qty_rmse": 2.04,
        "p99_qty_mae": 10.2,
        "time_nll": -3.49,
        "quantity_scale_mean": 0.5,
    }
    failing = dict(passing, qty_mae=0.951)

    passed = evaluate_gate(control, passing)
    failed = evaluate_gate(control, failing)

    assert passed["status"] == "pass"
    assert all(passed["checks"].values())
    assert failed["status"] == "fail"
    assert not failed["checks"]["overall_mae_improvement_at_least_5pct"]
    assert math.isclose(passed["deltas"]["time_nll_absolute_regression"], 0.01)
