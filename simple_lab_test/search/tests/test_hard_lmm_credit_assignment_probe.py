from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[3]
HELPER_PATH = ROOT / "paper/scripts/hard_lmm_credit_assignment_probe.py"
RUNNER_PATH = ROOT / "paper/scripts/run_hard_lmm_credit_assignment_probe.py"
CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_credit_assignment_probe_v1.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


probe = load_module("hard_lmm_credit_assignment_probe_test", HELPER_PATH)
runner = load_module("run_hard_lmm_credit_assignment_probe_test", RUNNER_PATH)


def test_contract_is_train_only_and_pins_three_frozen_b_checkpoints() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    runner.validate_contract(contract)
    assert contract["authorization"]["validation_rows"] is False
    assert contract["authorization"]["held_out_test"] is False
    assert contract["diagnostic_scope"]["model_parameter_updates"] is False
    assert contract["diagnostic_scope"]["batch_size"] == 128
    assert len(contract["datasets"]) == 3
    assert all(row["checkpoint_state_sha256"] for row in contract["datasets"])


def test_cyclic_derangement_is_fixed_and_complete() -> None:
    permutation = probe.cyclic_derangement(8)
    assert torch.equal(permutation, torch.tensor([7, 0, 1, 2, 3, 4, 5, 6]))
    assert torch.equal(permutation.sort().values, torch.arange(8))
    assert not bool(torch.eq(permutation, torch.arange(8)).any())
    with pytest.raises(ValueError):
        probe.cyclic_derangement(1)


def test_soft_read_trains_query_and_key_but_detaches_values() -> None:
    torch.manual_seed(11)
    query = torch.randn(5, 4, requires_grad=True)
    key = torch.randn(7, 4, requires_grad=True)
    values = torch.randn(7, 4, requires_grad=True)
    read, scores, weights = probe.cosine_soft_read(
        query,
        key,
        value_memory=values,
        temperature=1.0,
    )
    query_gradient, key_gradient, value_gradient = torch.autograd.grad(
        read.square().sum(),
        (query, key, values),
        allow_unused=True,
    )
    assert read.shape == (5, 4)
    assert scores.shape == weights.shape == (5, 7)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(5))
    assert float(query_gradient.norm()) > 0.0
    assert float(key_gradient.norm()) > 0.0
    assert value_gradient is None


def test_zero_forward_branch_is_bitwise_identical_and_keeps_gradient() -> None:
    torch.manual_seed(17)
    hard = torch.randn(6, 5, requires_grad=True)
    soft = torch.randn(6, 5, requires_grad=True)
    fused = probe.forward_identical_hidden(hard, soft)
    assert torch.equal(fused, hard)
    hard_gradient, soft_gradient = torch.autograd.grad(
        fused.square().sum(),
        (hard, soft),
    )
    expected = 2.0 * hard.detach()
    assert torch.equal(hard_gradient, expected)
    assert torch.equal(soft_gradient, expected)


def test_hard_top4_has_no_query_or_key_addressing_gradient() -> None:
    torch.manual_seed(23)
    query = torch.randn(5, 4, requires_grad=True)
    key = torch.randn(9, 4, requires_grad=True)
    detached_values = key.detach().clone()
    scores = F.normalize(query, dim=-1) @ F.normalize(key, dim=-1).T
    indices = torch.topk(scores, 4, dim=-1).indices
    hard_read = detached_values[indices].mean(dim=1)
    objective = hard_read.square().sum() + 0.0 * (query.sum() + key.sum())
    query_gradient, key_gradient = torch.autograd.grad(objective, (query, key))
    assert torch.count_nonzero(query_gradient) == 0
    assert torch.count_nonzero(key_gradient) == 0


def test_quantity_credit_signal_matches_autograd() -> None:
    torch.manual_seed(29)
    hidden = torch.randn(7, 5, requires_grad=True)
    weight = torch.randn(1, 5)
    bias = torch.randn(1)
    target = torch.rand(7) * 30.0
    observed = probe.quantity_credit_signal(hidden, weight, bias, target)
    autograd_credit = torch.autograd.grad(observed["log_mse"].sum(), hidden)[0]
    assert torch.allclose(observed["hidden_credit"], autograd_credit)
    assert torch.equal(observed["raw_prediction"], torch.expm1(observed["location"]))


def test_directional_alignment_uses_descent_sign() -> None:
    aligned = probe.directional_alignment(
        torch.tensor([1.0, -2.0]),
        torch.tensor([3.0, -4.0]),
    )
    opposed = probe.directional_alignment(
        torch.tensor([1.0, -2.0]),
        torch.tensor([-3.0, 4.0]),
    )
    assert aligned["dot"] == 11.0
    assert aligned["improvement_direction"] is True
    assert aligned["predicted_first_order_change_for_negative_source_step"] == -11.0
    assert opposed["improvement_direction"] is False


def _gate_fixture(*, normal_dot: float = 2.0, normal_cosine: float = 0.5):
    metric = {"dot": normal_dot, "cosine": normal_cosine}
    sham = {"dot": 1.0, "cosine": 0.1}
    return {
        "structural_audit": {
            "normal_forward_bitwise_identity": True,
            "shuffled_forward_bitwise_identity": True,
            "normal_prediction_bitwise_identity": True,
            "shuffled_prediction_bitwise_identity": True,
            "hard_query_addressing_gradient_max_abs": 0.0,
            "hard_key_addressing_gradient_max_abs": 0.0,
            "normal_query_addressing_gradient_l2": 1.0,
            "normal_key_addressing_gradient_l2": 1.0,
        },
        "cross_fold": {
            direction: {
                "normal": {
                    "log_mse": metric,
                    "raw_squared_error": metric,
                    "body_mae": metric,
                },
                "shuffled": {
                    "log_mse": sham,
                    "raw_squared_error": sham,
                    "body_mae": sham,
                },
                "normal_clipped": {
                    "log_mse": metric,
                    "raw_squared_error": metric,
                    "body_mae": metric,
                },
                "shuffled_clipped": {
                    "log_mse": sham,
                    "raw_squared_error": sham,
                    "body_mae": sham,
                },
            }
            for direction in ("fold0_to_fold1", "fold1_to_fold0")
        },
    }


def test_gate_requires_both_folds_body_and_shuffled_controls() -> None:
    passing = probe.evaluate_dataset_gate(_gate_fixture())
    assert passing["passed"] is True

    wrong_direction = _gate_fixture(normal_dot=-0.01)
    assert probe.evaluate_dataset_gate(wrong_direction)["passed"] is False

    not_better_than_sham = _gate_fixture(normal_cosine=0.05)
    assert probe.evaluate_dataset_gate(not_better_than_sham)["passed"] is False


def test_global_clip_difference_matches_full_parameter_vector() -> None:
    parameters = [torch.nn.Parameter(torch.zeros(2)), torch.nn.Parameter(torch.zeros(1))]
    hard = [torch.tensor([3.0, 4.0]), torch.tensor([0.0])]
    extra = [torch.tensor([0.0, 0.0]), torch.tensor([12.0])]
    difference = runner.clipped_difference(hard, extra, parameters, max_norm=1.0)
    hard_vector = torch.cat(hard)
    candidate_vector = torch.cat([hard[0] + extra[0], hard[1] + extra[1]])
    expected = (
        candidate_vector / (candidate_vector.norm() + 1e-6)
        - hard_vector / (hard_vector.norm() + 1e-6)
    )
    assert torch.allclose(torch.cat(difference), expected)
