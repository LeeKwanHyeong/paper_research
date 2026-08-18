from __future__ import annotations

import torch

from models.TPPs.CountAwareTPP import (
    CountAwareTitanTPP,
    TITAN_MEMORY_MODE_NONE,
    TITAN_MEMORY_MODE_STATIC_HARD,
    TITAN_MEMORY_MODE_STATIC_SOFT_GATED,
    TITAN_MEMORY_MODE_SURPRISE_GATED,
)
from models.Titan.common.memory import GatedSoftMemory, SurpriseGatedMemory
from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts.count_aware_tpp_backbone.core import target_outputs
from paper.scripts.count_aware_tpp_backbone.constants import (
    BACKBONES,
    BACKBONE_LABELS,
    SUPPORTED_BACKBONES,
    TITAN_MEMORY_BACKBONES,
)


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


def test_memory_variants_are_opt_in_supported_backbones() -> None:
    assert TITAN_MEMORY_BACKBONES == (
        "titantpp_no_memory",
        "titantpp_gated_soft_memory",
        "titantpp_surprise_memory",
    )
    assert all(name not in BACKBONES for name in TITAN_MEMORY_BACKBONES)
    assert SUPPORTED_BACKBONES == (*BACKBONES, *TITAN_MEMORY_BACKBONES)
    assert BACKBONE_LABELS["titantpp_no_memory"] == "TitanTPP No Memory"
    assert BACKBONE_LABELS["titantpp_gated_soft_memory"] == (
        "TitanTPP Gated Soft Memory"
    )
    assert BACKBONE_LABELS["titantpp_surprise_memory"] == (
        "TitanTPP Surprise Memory"
    )


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


def test_surprise_memory_starts_as_exact_no_memory_residual() -> None:
    control, _ = build_titan("titantpp_no_memory")
    candidate, metadata = build_titan("titantpp_surprise_memory")
    control.eval()
    candidate.eval()
    dts, mask, quantities = sample_batch()

    assert candidate.memory_mode == TITAN_MEMORY_MODE_SURPRISE_GATED
    assert isinstance(candidate.surprise_memory, SurpriseGatedMemory)
    assert candidate.lmm is None
    assert candidate.soft_memory is None
    assert metadata["candidate_name"] == "count_titan_surprise_memory"
    assert metadata["surprise_memory_rank"] == 16
    assert metadata["surprise_chunk_size"] == 32
    assert metadata["surprise_state_scope"] == "independent_input_sequence"

    with torch.no_grad():
        control_hidden = control.encode(dts, quantities, mask)
        candidate_hidden = candidate.encode(dts, quantities, mask)

    assert torch.equal(control_hidden, candidate_hidden)


def test_surprise_memory_is_prefix_causal_and_resets_between_calls() -> None:
    candidate, _ = build_titan("titantpp_surprise_memory")
    candidate.eval()
    assert candidate.surprise_memory is not None
    candidate.surprise_memory.residual_scale.data.fill_(0.5)
    dts, mask, quantities = sample_batch()
    changed_dts = dts.clone()
    changed_quantities = quantities.clone()
    changed_dts[0, -1] = 1_000_000.0
    changed_quantities[0, -1] = 1_000_000.0

    with torch.no_grad():
        original = candidate.encode(dts, quantities, mask)
        repeated = candidate.encode(dts, quantities, mask)
        changed = candidate.encode(changed_dts, changed_quantities, mask)

    assert torch.equal(original, repeated)
    assert torch.allclose(original[0, :-1], changed[0, :-1], atol=1e-6)
    assert torch.count_nonzero(original[1, -1]) == 0
    assert torch.count_nonzero(changed[1, -1]) == 0


def test_surprise_memory_padding_does_not_update_fast_weight_state() -> None:
    memory = SurpriseGatedMemory(
        d_model=8,
        memory_rank=4,
        chunk_size=2,
        dropout=0.0,
    ).eval()
    memory.residual_scale.data.fill_(0.5)
    encoded = torch.randn(1, 4, 8)
    mask = torch.tensor([[True, False, True, True]])
    changed = encoded.clone()
    changed[:, 1] = 1_000_000.0

    with torch.no_grad():
        original, original_diagnostics = memory.process(encoded, mask=mask)
        mutated, mutated_diagnostics = memory.process(changed, mask=mask)

    assert torch.equal(original, mutated)
    for key in original_diagnostics:
        assert torch.equal(original_diagnostics[key], mutated_diagnostics[key])
        assert original_diagnostics[key][0, 1] == 0.0


def test_surprise_chunking_changes_gradient_horizon_not_forward_values() -> None:
    short = SurpriseGatedMemory(
        d_model=8,
        memory_rank=4,
        chunk_size=2,
        dropout=0.0,
    ).eval()
    long = SurpriseGatedMemory(
        d_model=8,
        memory_rank=4,
        chunk_size=8,
        dropout=0.0,
    ).eval()
    long.load_state_dict(short.state_dict())
    short.residual_scale.data.fill_(0.5)
    long.residual_scale.data.fill_(0.5)
    encoded = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6, dtype=torch.bool)

    with torch.no_grad():
        short_output, short_diagnostics = short.process(encoded, mask=mask)
        long_output, long_diagnostics = long.process(encoded, mask=mask)

    assert torch.equal(short_output, long_output)
    for key in short_diagnostics:
        assert torch.equal(short_diagnostics[key], long_diagnostics[key])


def test_open_surprise_memory_routes_finite_gradients_through_updates() -> None:
    candidate, _ = build_titan("titantpp_surprise_memory")
    assert candidate.surprise_memory is not None
    candidate.surprise_memory.residual_scale.data.fill_(0.5)
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
        candidate.surprise_memory.query_proj.weight,
        candidate.surprise_memory.key_proj.weight,
        candidate.surprise_memory.value_proj.weight,
        candidate.surprise_memory.output_proj.weight,
        candidate.surprise_memory.update_rate_proj.bias,
        candidate.surprise_memory.momentum_logit,
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
