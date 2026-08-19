from __future__ import annotations

import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE,
    TITAN_MEMORY_MODE_PERSISTENT_ONLY,
    TITAN_MEMORY_MODE_PERSISTENT_SURPRISE_GATED,
    TITAN_MEMORY_MODE_STATIC_HARD,
    TITAN_QUANTITY_GRADIENT_ADAPTER_ONLY,
    TITAN_QUANTITY_GRADIENT_SHARED,
)
from paper.scripts.count_aware_tpp_backbone.constants import (
    BACKBONE_LABELS,
    SUPPORTED_BACKBONES,
    TITAN_PERSISTENT_MEMORY_BACKBONES,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


def build_titan(
    backbone: str,
    *,
    seed: int = 41,
) -> tuple[CountAwareTitanTPP, dict[str, object]]:
    torch.manual_seed(seed)
    model, metadata = build_count_aware_model(
        backbone,
        hidden_dim=16,
        train_log_mean=1.5,
        max_seq_len=8,
    )
    assert isinstance(model, CountAwareTitanTPP)
    return model, metadata


def sample_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dts = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 2.0, 4.0, 0.0],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
        ]
    )
    quantities = torch.tensor(
        [
            [2.0, 5.0, 9.0, 20.0],
            [1.0, 4.0, 12.0, 0.0],
        ]
    )
    return dts, mask, quantities


def open_memory_and_quantity_path(model: CountAwareTitanTPP) -> None:
    assert model.surprise_memory is not None
    model.surprise_memory.residual_scale.data.fill_(0.5)
    model.quantity_head.weight.data.fill_(0.05)


def has_nonzero_gradient(parameters: list[torch.nn.Parameter]) -> bool:
    return any(
        parameter.grad is not None
        and bool(torch.count_nonzero(parameter.grad).item())
        for parameter in parameters
    )


def assert_no_gradient(parameters: list[torch.nn.Parameter]) -> None:
    assert all(
        parameter.grad is None or not bool(torch.count_nonzero(parameter.grad).item())
        for parameter in parameters
    )


def test_persistent_matched_variants_are_explicit_opt_in_backbones() -> None:
    assert TITAN_PERSISTENT_MEMORY_BACKBONES == (
        "titantpp_persistent_only",
        "titantpp_persistent_surprise_memory",
        "titantpp_dual_memory_shared",
        "titantpp_dual_memory_adapter_only",
    )
    assert all(name in SUPPORTED_BACKBONES for name in TITAN_PERSISTENT_MEMORY_BACKBONES)
    assert BACKBONE_LABELS["titantpp_persistent_only"] == (
        "TitanTPP Persistent Only"
    )


@pytest.mark.parametrize(
    ("backbone", "memory_mode", "gradient_mode", "has_hard", "has_surprise"),
    [
        (
            "titantpp_persistent_only",
            TITAN_MEMORY_MODE_PERSISTENT_ONLY,
            TITAN_QUANTITY_GRADIENT_SHARED,
            False,
            False,
        ),
        (
            "titantpp",
            TITAN_MEMORY_MODE_STATIC_HARD,
            TITAN_QUANTITY_GRADIENT_SHARED,
            True,
            False,
        ),
        (
            "titantpp_persistent_surprise_memory",
            TITAN_MEMORY_MODE_PERSISTENT_SURPRISE_GATED,
            TITAN_QUANTITY_GRADIENT_SHARED,
            False,
            True,
        ),
        (
            "titantpp_dual_memory_shared",
            TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE,
            TITAN_QUANTITY_GRADIENT_SHARED,
            True,
            True,
        ),
        (
            "titantpp_dual_memory_adapter_only",
            TITAN_MEMORY_MODE_DUAL_HARD_SURPRISE,
            TITAN_QUANTITY_GRADIENT_ADAPTER_ONLY,
            True,
            True,
        ),
    ],
)
def test_all_matched_variants_keep_persistent_tokens(
    backbone: str,
    memory_mode: str,
    gradient_mode: str,
    has_hard: bool,
    has_surprise: bool,
) -> None:
    model, metadata = build_titan(backbone)

    assert model.memory_mode == memory_mode
    assert model.quantity_memory_gradient_mode == gradient_mode
    assert (model.lmm is not None) is has_hard
    assert (model.surprise_memory is not None) is has_surprise
    assert metadata["persistent_mem_size"] == 16
    assert all(
        layer.attn.persistent_mem is not None
        and layer.attn.persistent_mem.shape[1] == 16
        for layer in model.encoder.layers
    )


def test_zero_gate_persistent_surprise_matches_persistent_only() -> None:
    control, _ = build_titan("titantpp_persistent_only")
    candidate, _ = build_titan("titantpp_persistent_surprise_memory")
    control.eval()
    candidate.eval()
    dts, mask, quantities = sample_batch()

    with torch.no_grad():
        control_state = control.encode(dts, quantities, mask)
        candidate_state = candidate.encode(dts, quantities, mask)

    assert torch.equal(candidate_state, control_state)


@pytest.mark.parametrize(
    "backbone",
    ["titantpp_dual_memory_shared", "titantpp_dual_memory_adapter_only"],
)
def test_zero_gate_dual_states_match_hard_lmm(backbone: str) -> None:
    control, _ = build_titan("titantpp")
    candidate, _ = build_titan(backbone)
    control.eval()
    candidate.eval()
    dts, mask, quantities = sample_batch()

    with torch.no_grad():
        hard_state = control.encode(dts, quantities, mask)
        time_state, quantity_state = candidate.encode_task_states(
            dts,
            quantities,
            mask,
        )

    assert torch.equal(time_state, hard_state)
    assert torch.equal(quantity_state, hard_state)


@pytest.mark.parametrize(
    "backbone",
    ["titantpp_dual_memory_shared", "titantpp_dual_memory_adapter_only"],
)
def test_open_dual_memory_changes_only_quantity_forward_route(backbone: str) -> None:
    control, _ = build_titan("titantpp")
    candidate, _ = build_titan(backbone)
    control.eval()
    candidate.eval()
    open_memory_and_quantity_path(candidate)
    dts, mask, quantities = sample_batch()

    with torch.no_grad():
        hard_state = control.encode(dts, quantities, mask)
        time_state, quantity_state = candidate.encode_task_states(
            dts,
            quantities,
            mask,
        )

    assert torch.equal(time_state, hard_state)
    assert not torch.equal(quantity_state[0, 1:], hard_state[0, 1:])
    assert torch.count_nonzero(quantity_state[1, -1]) == 0


def test_adapter_only_quantity_loss_does_not_update_encoder_or_hard_lmm() -> None:
    model, _ = build_titan("titantpp_dual_memory_adapter_only")
    open_memory_and_quantity_path(model)
    dts, mask, quantities = sample_batch()

    outputs = target_outputs(
        model,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    outputs["quantity_train_loss"].mean().backward()

    assert_no_gradient(list(model.encoder.parameters()))
    assert model.lmm is not None
    assert_no_gradient(list(model.lmm.parameters()))
    assert model.surprise_memory is not None
    assert has_nonzero_gradient(list(model.surprise_memory.parameters()))
    assert has_nonzero_gradient(list(model.quantity_head.parameters()))
    assert_no_gradient(list(model.v_t.parameters()) + [model.b_t, model.w_raw])


def test_shared_quantity_loss_updates_hard_and_surprise_routes() -> None:
    model, _ = build_titan("titantpp_dual_memory_shared")
    open_memory_and_quantity_path(model)
    dts, mask, quantities = sample_batch()

    outputs = target_outputs(
        model,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    outputs["quantity_train_loss"].mean().backward()

    assert has_nonzero_gradient(list(model.encoder.parameters()))
    assert model.lmm is not None
    assert has_nonzero_gradient(list(model.lmm.parameters()))
    assert model.surprise_memory is not None
    assert has_nonzero_gradient(list(model.surprise_memory.parameters()))
    assert_no_gradient(list(model.v_t.parameters()) + [model.b_t, model.w_raw])


@pytest.mark.parametrize(
    "backbone",
    ["titantpp_dual_memory_shared", "titantpp_dual_memory_adapter_only"],
)
def test_time_loss_updates_hard_route_but_not_surprise_adapter(backbone: str) -> None:
    model, _ = build_titan(backbone)
    open_memory_and_quantity_path(model)
    dts, mask, quantities = sample_batch()

    outputs = target_outputs(
        model,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    outputs["time_loss"].mean().backward()

    assert has_nonzero_gradient(list(model.encoder.parameters()))
    assert model.lmm is not None
    assert has_nonzero_gradient(list(model.lmm.parameters()))
    assert model.surprise_memory is not None
    assert_no_gradient(list(model.surprise_memory.parameters()))
    assert has_nonzero_gradient(list(model.v_t.parameters()) + [model.b_t, model.w_raw])
    assert_no_gradient(list(model.quantity_head.parameters()))


@pytest.mark.parametrize("backbone", TITAN_PERSISTENT_MEMORY_BACKBONES)
def test_persistent_memory_variants_have_finite_forward_and_gradients(
    backbone: str,
) -> None:
    model, _ = build_titan(backbone)
    if model.surprise_memory is not None:
        open_memory_and_quantity_path(model)
    dts, mask, quantities = sample_batch()

    outputs = target_outputs(
        model,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
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


def test_adapter_only_is_rejected_outside_dual_memory() -> None:
    with pytest.raises(ValueError, match="valid only for dual memory"):
        CountAwareTitanTPP(
            hidden_dim=16,
            train_log_mean=1.5,
            max_seq_len=8,
            memory_mode=TITAN_MEMORY_MODE_PERSISTENT_ONLY,
            quantity_memory_gradient_mode=TITAN_QUANTITY_GRADIENT_ADAPTER_ONLY,
        )


@pytest.mark.parametrize(
    "backbone",
    ["titantpp_dual_memory_shared", "titantpp_dual_memory_adapter_only"],
)
def test_dual_route_prediction_does_not_observe_target_quantity(
    backbone: str,
) -> None:
    model, _ = build_titan(backbone)
    model.eval()
    open_memory_and_quantity_path(model)
    dts, mask, quantities = sample_batch()
    changed_quantities = quantities.clone()
    changed_quantities[0, 3] = 1_000_000.0
    changed_quantities[1, 2] = 1_000_000.0

    with torch.no_grad():
        original = target_outputs(
            model,
            dts,
            mask,
            quantities,
            lambda_log_qty=1.0,
        )
        changed = target_outputs(
            model,
            dts,
            mask,
            changed_quantities,
            lambda_log_qty=1.0,
        )

    assert torch.equal(original["pred_qty"], changed["pred_qty"])
    assert torch.equal(original["time_loss"], changed["time_loss"])
