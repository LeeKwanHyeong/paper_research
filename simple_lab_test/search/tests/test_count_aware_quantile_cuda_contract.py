"""Device-level identity and active-gradient checks used by the 5090 controller."""

from __future__ import annotations

import os

import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    LOG_MSE_VARIANT,
    QUANTILE_ADAPTIVE_VARIANT,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


BOUNDARIES = (10.0, 20.0, 30.0, 40.0)
WEIGHTS = (0.8, 0.8, 1.2, 1.6, 2.4)


def execution_device() -> torch.device:
    requested = os.environ.get("COUNT_AWARE_QUANTILE_TEST_DEVICE", "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("5090 CUDA contract requested CUDA but PyTorch cannot use it")
    return torch.device(requested)


def model(variant: str, *, strength: float) -> torch.nn.Module:
    kwargs = {}
    if variant == QUANTILE_ADAPTIVE_VARIANT:
        kwargs = {
            "quantile_adaptive_strength": strength,
            "quantile_adaptive_boundaries": BOUNDARIES,
            "quantile_adaptive_weights": WEIGHTS,
        }
    return build_count_aware_model(
        "titantpp",
        hidden_dim=16,
        train_log_mean=1.0,
        train_log_std=1.0,
        max_seq_len=8,
        quantity_variant=variant,
        time_intercept_limit=300.0,
        **kwargs,
    )[0].to(execution_device())


def batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = execution_device()
    dts = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0], [0.0, 2.0, 1.0, 4.0]],
        dtype=torch.float32,
        device=device,
    )
    mask = torch.ones_like(dts, dtype=torch.bool)
    quantities = torch.tensor(
        [[1.0, 2.0, 3.0, 35.0], [2.0, 5.0, 7.0, 50.0]],
        dtype=torch.float32,
        device=device,
    )
    return dts, mask, quantities


def test_device_zero_strength_preserves_exact_output_and_gradient() -> None:
    torch.manual_seed(20260906)
    control = model(LOG_MSE_VARIANT, strength=0.0).eval()
    torch.manual_seed(20260906)
    candidate = model(QUANTILE_ADAPTIVE_VARIANT, strength=0.0).eval()
    for name, tensor in control.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[name]), name
    dts, mask, quantities = batch()
    before = target_outputs(control, dts, mask, quantities, lambda_log_qty=1.0)
    after = target_outputs(candidate, dts, mask, quantities, lambda_log_qty=1.0)
    for name in before:
        assert torch.equal(before[name], after[name]), name
    before["joint_loss"].mean().backward()
    after["joint_loss"].mean().backward()
    for name, parameter in control.named_parameters():
        other = dict(candidate.named_parameters())[name]
        if parameter.grad is None:
            assert other.grad is None
        else:
            assert other.grad is not None and torch.equal(parameter.grad, other.grad), name


def test_device_active_loss_changes_gradient_and_stays_finite() -> None:
    torch.manual_seed(17)
    control = model(LOG_MSE_VARIANT, strength=0.0).eval()
    torch.manual_seed(17)
    candidate = model(QUANTILE_ADAPTIVE_VARIANT, strength=1.0).eval()
    dts, mask, quantities = batch()
    before = target_outputs(control, dts, mask, quantities, lambda_log_qty=1.0)
    after = target_outputs(candidate, dts, mask, quantities, lambda_log_qty=1.0)
    assert torch.equal(before["pred_qty"], after["pred_qty"])
    assert not torch.equal(before["quantity_train_loss"], after["quantity_train_loss"])
    before["joint_loss"].mean().backward()
    after["joint_loss"].mean().backward()
    assert all(torch.isfinite(value).all() for value in after.values())
    assert any(
        left.grad is not None
        and right.grad is not None
        and not torch.equal(left.grad, right.grad)
        for left, right in zip(control.parameters(), candidate.parameters())
    )
