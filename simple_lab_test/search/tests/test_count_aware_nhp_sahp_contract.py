from __future__ import annotations

import pytest
import torch

from models.TPPs.NeuralHawkesTPP import CountAwareNHP as ModelCountAwareNHP
from models.TPPs.SelfAttentiveHawkesTPP import (
    CountAwareSAHP as ModelCountAwareSAHP,
)
from paper.scripts.run_count_aware_tpp_backbone_control import (
    BACKBONES,
    BACKBONE_LABELS,
    CountAwareNHP,
    CountAwareSAHP,
    build_model,
    target_outputs,
)


ADAPTED_BACKBONES = ("nhp", "sahp")


def test_script_reexports_models_from_the_model_package() -> None:
    assert CountAwareNHP is ModelCountAwareNHP
    assert CountAwareSAHP is ModelCountAwareSAHP


def build_adapted(backbone: str):
    torch.manual_seed(101)
    model, metadata = build_model(
        backbone,
        hidden_dim=16,
        train_log_mean=1.5,
        max_seq_len=8,
    )
    return model, metadata


def sample_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dts = torch.tensor([
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 2.0, 4.0, 0.0],
    ])
    mask = torch.tensor([
        [True, True, True, True],
        [True, True, True, False],
    ])
    quantities = torch.tensor([
        [2.0, 5.0, 9.0, 20.0],
        [1.0, 4.0, 12.0, 0.0],
    ])
    return dts, mask, quantities


def test_adapted_backbones_are_registered_with_explicit_labels() -> None:
    assert BACKBONES[-2:] == ADAPTED_BACKBONES
    assert BACKBONE_LABELS["nhp"] == "Adapted NHP"
    assert BACKBONE_LABELS["sahp"] == "Adapted SAHP"


def test_adapted_backbone_classes_and_metadata_are_explicit() -> None:
    nhp, nhp_metadata = build_adapted("nhp")
    sahp, sahp_metadata = build_adapted("sahp")

    assert isinstance(nhp, CountAwareNHP)
    assert isinstance(sahp, CountAwareSAHP)
    assert nhp_metadata["encoder_type"] == "continuous_time_lstm"
    assert sahp_metadata["encoder_type"] == (
        "causal_self_attention_with_continuous_decay"
    )
    assert nhp_metadata["shared_time_head"] is True
    assert sahp_metadata["shared_time_head"] is True


@pytest.mark.parametrize("backbone", ADAPTED_BACKBONES)
def test_adapted_forward_backward_is_finite_and_mark_free(backbone: str) -> None:
    model, _ = build_adapted(backbone)
    dts, mask, quantities = sample_batch()

    outputs = target_outputs(
        model,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    outputs["joint_loss"].mean().backward()

    assert not hasattr(model, "mark_head")
    for key, value in outputs.items():
        if torch.is_tensor(value):
            assert torch.isfinite(value).all(), key
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("backbone", ADAPTED_BACKBONES)
def test_target_quantity_is_hidden_from_adapted_encoder(backbone: str) -> None:
    model, _ = build_adapted(backbone)
    model.eval()
    dts, mask, quantities = sample_batch()
    changed = quantities.clone()
    changed[0, -1] = 2000.0
    changed[1, 2] = 3000.0

    with torch.no_grad():
        original = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
        mutated = target_outputs(model, dts, mask, changed, lambda_log_qty=1.0)

    assert torch.allclose(original["pred_qty"], mutated["pred_qty"], atol=1e-7)
    assert torch.allclose(original["time_loss"], mutated["time_loss"], atol=1e-7)


@pytest.mark.parametrize("backbone", ADAPTED_BACKBONES)
def test_masked_padding_values_do_not_change_valid_outputs(backbone: str) -> None:
    model, _ = build_adapted(backbone)
    model.eval()
    dts = torch.tensor([[0.0, 2.0, 0.0, 0.0]])
    quantities = torch.tensor([[3.0, 8.0, 0.0, 0.0]])
    mask = torch.tensor([[True, True, False, False]])
    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    changed_dts[:, 2:] = 1_000_000.0
    changed_quantities[:, 2:] = 1_000_000.0

    with torch.no_grad():
        original = model.encode(dts, quantities, mask)
        mutated = model.encode(changed_dts, changed_quantities, mask)

    assert torch.allclose(original[:, :2], mutated[:, :2], atol=1e-6)
    assert torch.count_nonzero(original[:, 2:]) == 0
    assert torch.count_nonzero(mutated[:, 2:]) == 0


def test_nhp_masked_step_does_not_update_recurrent_state() -> None:
    model, _ = build_adapted("nhp")
    model.eval()
    dts = torch.tensor([[0.0, 10.0, 3.0]])
    quantities = torch.tensor([[2.0, 99.0, 5.0]])
    mask = torch.tensor([[True, False, True]])
    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    changed_dts[:, 1] = 1_000_000.0
    changed_quantities[:, 1] = 1_000_000.0

    with torch.no_grad():
        original = model.encode(dts, quantities, mask)
        mutated = model.encode(changed_dts, changed_quantities, mask)

    assert torch.allclose(original[:, 2], mutated[:, 2], atol=1e-7)


def test_sahp_attention_does_not_read_future_events() -> None:
    model, _ = build_adapted("sahp")
    model.eval()
    dts = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    quantities = torch.tensor([[2.0, 4.0, 8.0, 16.0]])
    mask = torch.ones_like(dts, dtype=torch.bool)
    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    changed_dts[:, -1] = 1_000_000.0
    changed_quantities[:, -1] = 1_000_000.0

    with torch.no_grad():
        original = model.encode(dts, quantities, mask)
        mutated = model.encode(changed_dts, changed_quantities, mask)

    assert torch.allclose(original[:, :-1], mutated[:, :-1], atol=1e-6)


@pytest.mark.parametrize("backbone", ADAPTED_BACKBONES)
def test_extreme_intervals_and_quantities_remain_finite(backbone: str) -> None:
    model, _ = build_adapted(backbone)
    model.eval()
    dts = torch.tensor([[0.0, 1e4, 1e6]])
    mask = torch.ones_like(dts, dtype=torch.bool)
    quantities = torch.tensor([[1.0, 1e6, 1e8]])

    outputs = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.0)
    outputs["joint_loss"].mean().backward()

    assert all(
        torch.isfinite(value).all()
        for value in outputs.values()
        if torch.is_tensor(value)
    )
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
