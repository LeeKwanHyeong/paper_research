from __future__ import annotations

import pytest
import torch

from paper.scripts.run_count_aware_tpp_backbone_control import (
    BACKBONES,
    TAIL_HEAD_ONLY_VARIANT,
    TAIL_SHARED_VARIANT,
    VARIANT,
    build_model,
    normalize_quantity_variants,
    target_outputs,
)


def build_variant(
    variant: str,
    *,
    backbone: str = "rmtpp",
    lambda_tail: float = 1.0,
):
    torch.manual_seed(29)
    model, _ = build_model(
        backbone,
        hidden_dim=16,
        train_log_mean=1.5,
        max_seq_len=8,
        quantity_variant=variant,
        lambda_tail=lambda_tail,
        tail_threshold=46.0,
        tail_normalization_scale=46.0,
        tail_clip_cap=187.0,
        tail_huber_delta=1.0,
    )
    return model


def sample_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dts = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 3.0]])
    mask = torch.ones_like(dts, dtype=torch.bool)
    quantities = torch.tensor([[2.0, 5.0, 47.0], [3.0, 7.0, 477.0]])
    return dts, mask, quantities


def test_tail_variant_aliases_are_explicit() -> None:
    assert normalize_quantity_variants("log_mse,tail_shared,tail_head_only") == (
        VARIANT,
        TAIL_SHARED_VARIANT,
        TAIL_HEAD_ONLY_VARIANT,
    )


@pytest.mark.parametrize("variant", [TAIL_SHARED_VARIANT, TAIL_HEAD_ONLY_VARIANT])
def test_tail_variants_add_no_parameters(variant: str) -> None:
    control = build_variant(VARIANT)
    candidate = build_variant(variant)

    assert set(control.state_dict()) == set(candidate.state_dict())
    for name, tensor in control.state_dict().items():
        assert torch.equal(tensor, candidate.state_dict()[name]), name


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("variant", [TAIL_SHARED_VARIANT, TAIL_HEAD_ONLY_VARIANT])
def test_lambda_zero_matches_log_mse_forward_loss_and_gradients(
    backbone: str,
    variant: str,
) -> None:
    control = build_variant(VARIANT, backbone=backbone, lambda_tail=0.0).eval()
    candidate = build_variant(variant, backbone=backbone, lambda_tail=0.0).eval()
    dts, mask, quantities = sample_batch()

    control_output = target_outputs(
        control,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    candidate_output = target_outputs(
        candidate,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    for key in ("joint_loss", "time_loss", "log_qty_loss", "pred_qty"):
        assert torch.equal(control_output[key], candidate_output[key]), key

    control_output["joint_loss"].mean().backward()
    candidate_output["joint_loss"].mean().backward()
    for (control_name, control_parameter), (candidate_name, candidate_parameter) in zip(
        control.named_parameters(),
        candidate.named_parameters(),
    ):
        assert control_name == candidate_name
        if control_parameter.grad is None:
            assert candidate_parameter.grad is None
        else:
            assert torch.equal(control_parameter.grad, candidate_parameter.grad), control_name


def test_shared_tail_loss_updates_hidden_and_quantity_head_only() -> None:
    model = build_variant(TAIL_SHARED_VARIANT)
    torch.nn.init.constant_(model.quantity_head.weight, 0.05)
    hidden = torch.randn(4, model.hidden_dim, requires_grad=True)
    target = torch.tensor([47.0, 60.0, 187.0, 477.0])

    outputs = model.quantity_outputs(hidden, target)
    outputs["tail_aux_loss"].mean().backward()

    assert hidden.grad is not None and torch.count_nonzero(hidden.grad) > 0
    assert model.quantity_head.weight.grad is not None
    assert torch.count_nonzero(model.quantity_head.weight.grad) > 0
    assert model.v_t.weight.grad is None


def test_head_only_tail_loss_stops_hidden_gradient() -> None:
    model = build_variant(TAIL_HEAD_ONLY_VARIANT)
    torch.nn.init.constant_(model.quantity_head.weight, 0.05)
    hidden = torch.randn(4, model.hidden_dim, requires_grad=True)
    target = torch.tensor([47.0, 60.0, 187.0, 477.0])

    outputs = model.quantity_outputs(hidden, target)
    outputs["tail_aux_loss"].mean().backward()

    assert hidden.grad is None or torch.count_nonzero(hidden.grad) == 0
    assert model.quantity_head.weight.grad is not None
    assert torch.count_nonzero(model.quantity_head.weight.grad) > 0
    assert model.v_t.weight.grad is None


def test_tail_loss_is_zero_for_body_and_finite_for_extreme_quantity() -> None:
    model = build_variant(TAIL_HEAD_ONLY_VARIANT)
    hidden = torch.randn(4, model.hidden_dim, requires_grad=True)
    target = torch.tensor([0.0, 46.0, 47.0, 1.0e12])

    outputs = model.quantity_outputs(hidden, target)
    outputs["train_loss"].mean().backward()

    assert torch.equal(outputs["tail_indicator"], torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert torch.count_nonzero(outputs["tail_aux_loss"][:2]) == 0
    for key, value in outputs.items():
        assert torch.isfinite(value).all(), key
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
