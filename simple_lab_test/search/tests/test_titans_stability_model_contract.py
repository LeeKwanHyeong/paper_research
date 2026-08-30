"""Causal/state/checkpoint contracts with the inner stability option enabled."""

import copy

import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.Titan.common.titans_mac_optimized import apply_titantpp_mac_semantic_optimization
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


def build():
    torch.manual_seed(31)
    return build_count_aware_model("titantpp_titans_mac", hidden_dim=8,
                                  train_log_mean=1.5, max_seq_len=64,
                                  titans_memory_gradient_clip=1.0)


def test_stability_is_explicit_and_rejected_for_other_backbones():
    model, metadata = build()
    assert metadata["titans_memory_gradient_clip"] == 1.0
    assert metadata["candidate_name"] == "count_titan_mac_inner_grad_clipped"
    assert model.titans_mac_encoder.neural_memory.gradient_max_norm == 1.0
    optimized = apply_titantpp_mac_semantic_optimization(copy.deepcopy(model))
    assert optimized.titans_mac_encoder.neural_memory.gradient_max_norm == 1.0
    with pytest.raises(ValueError, match="requires titantpp"):
        build_count_aware_model("thp", hidden_dim=8, train_log_mean=1.5,
                               max_seq_len=64, titans_memory_gradient_clip=1.0)


def test_multisegment_causality_masking_series_reset_and_checkpoint(tmp_path):
    model, _ = build()
    model.eval()
    dts = torch.ones(2, 33)
    quantities = torch.arange(1, 34).float().repeat(2, 1)
    mask = torch.ones_like(dts, dtype=torch.bool)
    mask[1, 18:] = False
    write_mask = mask.clone()
    write_mask[0, 32] = False
    write_mask[1, 17] = False
    with torch.no_grad():
        encoded, state, _ = model.encode_with_memory_state(
            dts, quantities, mask, memory_write_mask=write_mask,
            series_ids=torch.tensor([1, 2]))
        changed = quantities.clone()
        changed[0, 32] = 1e20
        changed[1, 17:] = 1e20
        mutated, mutated_state, _ = model.encode_with_memory_state(
            dts, changed, mask, memory_write_mask=write_mask,
            series_ids=torch.tensor([1, 2]))
        assert torch.equal(encoded[0, :32], mutated[0, :32])
        assert torch.equal(encoded[1, :17], mutated[1, :17])
        for left, right in zip(state.memory_tensors(), mutated_state.memory_tensors()):
            assert torch.equal(left, right)
        reset, reset_state, _ = model.encode_with_memory_state(
            dts[:, :17], quantities[:, :17], mask[:, :17], state=state,
            series_ids=torch.tensor([3, 4]))
        fresh, fresh_state, _ = model.encode_with_memory_state(
            dts[:, :17], quantities[:, :17], mask[:, :17],
            series_ids=torch.tensor([3, 4]))
        assert torch.equal(reset, fresh)
        for left, right in zip(reset_state.memory_tensors(), fresh_state.memory_tensors()):
            assert torch.equal(left, right)

    path = tmp_path / "checkpoint.pt"
    torch.save(model.state_dict(), path)
    restored, _ = build()
    restored.load_state_dict(torch.load(path, weights_only=True), strict=True)
    restored.eval()
    with torch.no_grad():
        before = target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)
        after = target_outputs(restored, dts, mask, quantities, lambda_log_qty=1.)
        for key in before:
            assert torch.isfinite(before[key]).all()
            assert torch.equal(before[key], after[key])
