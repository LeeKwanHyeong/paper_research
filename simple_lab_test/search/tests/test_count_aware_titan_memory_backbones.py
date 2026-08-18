from __future__ import annotations

import torch

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    TITAN_MEMORY_MODE_NONE,
    TITAN_MEMORY_MODE_STATIC_HARD,
)
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
