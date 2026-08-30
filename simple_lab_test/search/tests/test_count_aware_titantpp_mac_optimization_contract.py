from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import torch

from models.TPPs.CountAwareFactory import build_count_aware_model
from models.TPPs.CountAwareTPP import CountAwareTitanTPP
from models.Titan.common.titans_mac import (
    TitansMemoryState,
    TitansNeuralMemory,
    _scan_titans_write_sequence,
)
from models.Titan.common.titans_mac_optimized import (
    OptimizedTitansMACEncoder,
    OptimizedTitansNeuralMemory,
    _scan_titans_write_sequence_state_only,
    apply_titantpp_mac_semantic_optimization,
)
from paper.scripts.count_aware_titantpp_mac_runtime import (
    build_count_aware_titantpp_mac_primary,
)
from paper.scripts.count_aware_tpp_backbone.core import target_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def build_b1(*, optimized: bool) -> CountAwareTitanTPP:
    torch.manual_seed(20260830)
    kwargs = {
        "hidden_dim": 8,
        "train_log_mean": 1.5,
        "train_log_std": 0.7,
        "max_seq_len": 32,
    }
    if optimized:
        model, _ = build_count_aware_titantpp_mac_primary(**kwargs)
    else:
        model, _ = build_count_aware_model("titantpp_titans_mac", **kwargs)
    assert isinstance(model, CountAwareTitanTPP)
    return model.eval()


def sample_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dts = torch.tensor(
        [[0.0, 1.0, 2.0, 4.0, 8.0], [0.0, 2.0, 3.0, 5.0, 0.0]]
    )
    quantities = torch.tensor(
        [[1.0, 2.0, 5.0, 13.0, 34.0], [2.0, 3.0, 8.0, 21.0, 0.0]]
    )
    mask = torch.tensor(
        [[True, True, True, True, True], [True, True, True, True, False]]
    )
    return dts, quantities, mask


def state_tensors(state: TitansMemoryState) -> tuple[torch.Tensor, ...]:
    return (*state.memory_tensors(), *state.momentum_tensors())


def test_semantic_optimization_contract_freezes_allowed_changes_and_shapes() -> None:
    contract = json.loads(
        (
            PROJECT_ROOT
            / "paper/contracts/count_aware_titantpp_mac_semantic_optimization_v1.json"
        ).read_text(encoding="utf-8")
    )

    assert contract["parent_contract"] == "count_aware_titantpp_mac_primary_v1"
    assert "neural_memory_equations" in contract["forbidden_changes"]
    assert "segment_size" in contract["forbidden_changes"]
    assert "checkpoint_parameter_keys" in contract["forbidden_changes"]
    shapes = contract["frozen_runtime_shapes"]
    assert shapes == {
        "compiled_scan_batch_size": 128,
        "compiled_scan_chunk_size": 16,
        "mac_segment_size": 16,
    }
    assert contract["runtime_acceptance"][
        "b1_b0_steady_training_step_ratio_target_maximum"
    ] == 3.0
    for relative_path, expected_digest in contract["frozen_base_sha256"].items():
        # This is a historical freeze, not a ban on later opt-in stability
        # variants. Check its immutable revision, never replace its digests.
        result = subprocess.run(
            ["git", "show", f"08e59880cd61cbd27cec40aa04636452b87bebfc:{relative_path}"],
            cwd=PROJECT_ROOT, capture_output=True,
        )
        if result.returncode:
            pytest.skip("Historical source object unavailable in exported snapshot")
        actual_digest = hashlib.sha256(result.stdout).hexdigest()
        assert actual_digest == expected_digest


def test_state_only_recurrence_matches_frozen_diagnostic_recurrence() -> None:
    torch.manual_seed(7)
    memory = TitansNeuralMemory(4, hidden_expansion=2).double()
    state = memory.initial_state(
        3,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    inputs = torch.randn(3, 5, 4, dtype=torch.float64)
    keys, values, theta, eta, alpha = memory._project_write(inputs)
    write_mask = torch.tensor(
        [
            [True, True, False, True, True],
            [True, False, False, True, True],
            [False, True, True, True, False],
        ]
    )
    arguments = (
        *state.memory_tensors(),
        *state.momentum_tensors(),
        keys,
        values,
        theta.squeeze(-1),
        eta.squeeze(-1),
        alpha.squeeze(-1),
        write_mask,
    )

    diagnostic = _scan_titans_write_sequence(*arguments)
    state_only = _scan_titans_write_sequence_state_only(*arguments)

    for expected, actual in zip(diagnostic[:8], state_only, strict=True):
        assert torch.allclose(actual, expected, atol=1e-12, rtol=1e-12)


def test_adapter_preserves_rng_state_parameter_keys_and_values() -> None:
    frozen = build_b1(optimized=False)
    before = {
        name: value.detach().clone() for name, value in frozen.state_dict().items()
    }
    rng_before = torch.random.get_rng_state().clone()

    apply_titantpp_mac_semantic_optimization(frozen)

    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert isinstance(frozen.titans_mac_encoder, OptimizedTitansMACEncoder)
    assert isinstance(
        frozen.titans_mac_encoder.neural_memory,
        OptimizedTitansNeuralMemory,
    )
    assert frozen.state_dict().keys() == before.keys()
    for name, value in frozen.state_dict().items():
        assert torch.equal(value, before[name])


def test_primary_runtime_adds_forward_metadata_without_factory_drift() -> None:
    model, metadata = build_count_aware_titantpp_mac_primary(
        hidden_dim=8,
        train_log_mean=1.5,
        max_seq_len=16,
    )

    assert isinstance(model.titans_mac_encoder, OptimizedTitansMACEncoder)
    assert metadata["candidate_name"] == "count_titan_faithful_titans_mac"
    assert metadata["backbone_contract_id"] == "B1"
    assert metadata["paper_model_name"] == "Count-aware TitanTPP-MAC"
    assert metadata["model_positioning"] == "primary_candidate"
    assert metadata["checkpoint_parameter_keys_changed"] is False

    with pytest.raises(ValueError, match="lambda_tail=0"):
        build_count_aware_titantpp_mac_primary(
            hidden_dim=8,
            train_log_mean=1.5,
            max_seq_len=16,
            lambda_tail=0.1,
        )


def test_standard_forward_and_gradients_match_frozen_b1() -> None:
    reference = build_b1(optimized=False)
    optimized = copy.deepcopy(reference)
    apply_titantpp_mac_semantic_optimization(optimized)
    dts, quantities, mask = sample_batch()

    reference_outputs = target_outputs(
        reference,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    optimized_outputs = target_outputs(
        optimized,
        dts,
        mask,
        quantities,
        lambda_log_qty=1.0,
    )
    reference_outputs["joint_loss"].mean().backward()
    optimized_outputs["joint_loss"].mean().backward()

    for name in (
        "joint_loss",
        "time_loss",
        "pred_qty",
        "quantity_train_loss",
    ):
        assert torch.allclose(
            optimized_outputs[name],
            reference_outputs[name],
            atol=1e-6,
            rtol=1e-6,
        )
    reference_gradients = {
        name: parameter.grad
        for name, parameter in reference.named_parameters()
        if parameter.grad is not None
    }
    optimized_gradients = {
        name: parameter.grad
        for name, parameter in optimized.named_parameters()
        if parameter.grad is not None
    }
    assert optimized_gradients.keys() == reference_gradients.keys()
    for name in reference_gradients:
        assert torch.allclose(
            optimized_gradients[name],
            reference_gradients[name],
            atol=1e-6,
            rtol=1e-6,
        )


def test_explicit_state_api_keeps_diagnostics_chunk_equivalence_and_series_reset() -> None:
    model = build_b1(optimized=True)
    dts, quantities, mask = sample_batch()
    series_ids = torch.tensor([11, 22])
    with torch.no_grad():
        token_encoded, token_state, diagnostics = model.encode_with_memory_state(
            dts,
            quantities,
            mask,
            series_ids=series_ids,
            write_chunk_size=1,
        )
        chunk_encoded, chunk_state, _ = model.encode_with_memory_state(
            dts,
            quantities,
            mask,
            series_ids=series_ids,
            write_chunk_size=16,
        )
    assert set(diagnostics) == {
        "associative_loss",
        "update_rate",
        "momentum_rate",
        "forgetting_rate",
        "write_applied",
    }
    assert diagnostics["write_applied"].shape == mask.shape
    assert torch.allclose(token_encoded, chunk_encoded, atol=1e-6, rtol=1e-6)
    for left, right in zip(
        state_tensors(token_state),
        state_tensors(chunk_state),
        strict=True,
    ):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-6)

    next_dts = dts[:, :3]
    next_quantities = quantities[:, :3]
    next_mask = mask[:, :3]
    changed_ids = torch.tensor([33, 44])
    with torch.no_grad():
        reset_encoded, reset_state, _ = model.encode_with_memory_state(
            next_dts,
            next_quantities,
            next_mask,
            state=token_state,
            series_ids=changed_ids,
        )
        fresh_encoded, fresh_state, _ = model.encode_with_memory_state(
            next_dts,
            next_quantities,
            next_mask,
            series_ids=changed_ids,
        )
    assert torch.allclose(reset_encoded, fresh_encoded, atol=1e-6, rtol=1e-6)
    for left, right in zip(
        state_tensors(reset_state),
        state_tensors(fresh_state),
        strict=True,
    ):
        assert torch.allclose(left, right, atol=1e-6, rtol=1e-6)


def test_padding_and_extreme_input_forward_backward_are_finite() -> None:
    tensor = torch.arange(6).reshape(2, 3)
    padded = OptimizedTitansNeuralMemory._pad_dimension(
        tensor,
        dimension=0,
        target_size=4,
    )
    assert torch.equal(padded[:2], tensor)
    assert torch.equal(padded[2:], torch.zeros(2, 3, dtype=tensor.dtype))

    model = build_b1(optimized=True).train()
    dts = torch.tensor([[0.0, 1e-6, 1e2, 1e4]])
    quantities = torch.tensor([[0.0, 1.0, 1e3, 1e6]])
    mask = torch.ones_like(dts, dtype=torch.bool)
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
