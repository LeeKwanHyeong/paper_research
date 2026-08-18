from __future__ import annotations

import torch

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    TITAN_MEMORY_MODE_NONE,
    TITAN_MEMORY_MODE_STATIC_HARD,
    TITAN_MEMORY_MODE_STATIC_SOFT_GATED,
)
from models.Titan.common.memory import GatedSoftMemory
from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


def build_titan(backbone: str) -> tuple[CountAwareTitanTPP, dict[str, object]]:
    torch.manual_seed(41)
    model, metadata = build_count_aware_model(
        backbone,
        hidden_dim=16,
        train_log_mean=1.5,
        max_seq_len=8,
    )
    assert isinstance(model, CountAwareTitanTPP)
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


def test_no_memory_backbone_removes_both_static_memory_paths() -> None:
    control, control_metadata = build_titan("titantpp")
    candidate, candidate_metadata = build_titan("titantpp_no_memory")

    assert control.memory_mode == TITAN_MEMORY_MODE_STATIC_HARD
    assert candidate.memory_mode == TITAN_MEMORY_MODE_NONE
    assert control.lmm is not None
    assert candidate.lmm is None
    assert all(
        layer.attn.persistent_mem is None
        for layer in candidate.encoder.layers
    )
    assert control_metadata["candidate_name"] == "count_titan_small_lmm"
    assert candidate_metadata["candidate_name"] == "count_titan_no_memory"
    assert candidate_metadata["persistent_mem_size"] == 0
    assert candidate_metadata["lmm_mem_size"] == 0
    assert sum(parameter.numel() for parameter in candidate.parameters()) < sum(
        parameter.numel() for parameter in control.parameters()
    )


def test_no_memory_forward_backward_is_finite_and_causal() -> None:
    model, _ = build_titan("titantpp_no_memory")
    model.eval()
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

    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    changed_dts[0, -1] = 1_000_000.0
    changed_quantities[0, -1] = 1_000_000.0
    with torch.no_grad():
        original = model.encode(dts, quantities, mask)
        changed = model.encode(changed_dts, changed_quantities, mask)

    assert torch.allclose(original[0, :-1], changed[0, :-1], atol=1e-6)
    assert torch.count_nonzero(original[1, -1]) == 0
    assert torch.count_nonzero(changed[1, -1]) == 0


def test_gated_soft_memory_starts_as_exact_no_memory_residual() -> None:
    control, _ = build_titan("titantpp_no_memory")
    candidate, metadata = build_titan("titantpp_gated_soft_memory")
    control.eval()
    candidate.eval()
    dts, mask, quantities = sample_batch()

    assert candidate.memory_mode == TITAN_MEMORY_MODE_STATIC_SOFT_GATED
    assert candidate.lmm is None
    assert isinstance(candidate.soft_memory, GatedSoftMemory)
    assert metadata["candidate_name"] == "count_titan_gated_soft_memory"
    assert metadata["memory_residual_gate_init"] == 0.0
    assert all(
        layer.attn.persistent_mem is None
        for layer in candidate.encoder.layers
    )

    with torch.no_grad():
        control_hidden = control.encode(dts, quantities, mask)
        candidate_hidden = candidate.encode(dts, quantities, mask)

    assert torch.equal(control_hidden, candidate_hidden)


def test_gated_soft_memory_uses_dense_normalized_retrieval() -> None:
    candidate, _ = build_titan("titantpp_gated_soft_memory")
    assert candidate.soft_memory is not None
    encoded = torch.randn(2, 5, candidate.hidden_dim)

    _, weights = candidate.soft_memory.retrieve(encoded)

    assert weights.shape == (2, 5, candidate.soft_memory.mem_size)
    assert torch.all(weights > 0.0)
    assert torch.allclose(
        weights.sum(dim=-1),
        torch.ones(2, 5),
        atol=1e-6,
    )


def test_open_gated_soft_memory_routes_finite_gradients_to_memory() -> None:
    candidate, _ = build_titan("titantpp_gated_soft_memory")
    assert candidate.soft_memory is not None
    candidate.soft_memory.residual_scale.data.fill_(0.5)
    dts, mask, quantities = sample_batch()

    loss = target_outputs(
        candidate,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )["joint_loss"].mean()
    loss.backward()

    memory_parameters = (
        candidate.soft_memory.memory_keys,
        candidate.soft_memory.memory_values,
        candidate.soft_memory.query_proj.weight,
        candidate.soft_memory.output_proj.weight,
    )
    assert all(parameter.grad is not None for parameter in memory_parameters)
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in memory_parameters
        if parameter.grad is not None
    )
    assert all(
        torch.count_nonzero(parameter.grad) > 0
        for parameter in memory_parameters
        if parameter.grad is not None
    )
