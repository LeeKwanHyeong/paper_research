"""Cost proof safety and CUDA learning-path coverage for causal-QKV screening."""

from __future__ import annotations

import math

import pytest
import torch

from paper.scripts.profile_hard_lmm_causal_qkv import (
    BASELINE,
    BATCH_SIZE,
    CANDIDATE,
    HIDDEN_DIM,
    KERNEL_KEYS,
    LENGTHS,
    MEASURED_STEPS,
    REPEATS,
    WARMUP_STEPS,
    kernel_gradient_audit,
    measurement_order,
    positive_ratio,
    run_profile,
    run_worker,
    summarize_cost,
    synthetic_batch,
    tensor_tree_finite,
)


def proof_rows(step_ratio: float = 1.4, peak_ratio: float = 1.2) -> list[dict]:
    rows = []
    for length in LENGTHS:
        for repeat in range(REPEATS):
            for name in (BASELINE, CANDIDATE):
                step = 0.01 * (step_ratio if name == CANDIDATE else 1.0)
                rows.append({
                    "status": "passed", "sequence_length": length, "repeat": repeat,
                    "backbone": name, "batch_size": BATCH_SIZE, "hidden_dim": HIDDEN_DIM,
                    "warmup_steps": WARMUP_STEPS, "measured_steps": MEASURED_STEPS,
                    "step_seconds": [step] * MEASURED_STEPS, "median_step_seconds": step,
                    "peak_allocated_bytes": 1000 * (peak_ratio if name == CANDIDATE else 1.0),
                    "finite_model_optimizer": True,
                    "learning_path": {"status": "passed"} if name == CANDIDATE else None,
                })
    return rows


@pytest.mark.parametrize("numerator,denominator", [
    (0.0, 1.0), (1.0, 0.0), (-1.0, 1.0), (1.0, -1.0),
    (math.nan, 1.0), (1.0, math.nan), (math.inf, 1.0), (1.0, math.inf),
])
def test_cost_ratio_rejects_missing_or_invalid_measurements(numerator, denominator) -> None:
    with pytest.raises(ValueError, match="finite, strictly positive"):
        positive_ratio(numerator, denominator)


def test_order_alternates_and_rejects_out_of_contract_repeats() -> None:
    assert [measurement_order(index)[0] for index in range(REPEATS)] == [BASELINE, CANDIDATE, BASELINE]
    for invalid in (-1, REPEATS, True):
        with pytest.raises(ValueError):
            measurement_order(invalid)


def test_profile_requires_complete_nonduplicated_paired_grid() -> None:
    rows = proof_rows()
    assert summarize_cost(rows)["status"] == "passed"
    for invalid in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError, match="complete paired grid"):
            summarize_cost(invalid)


def test_each_length_must_pass_both_cost_limits() -> None:
    assert summarize_cost(proof_rows(1.5, 1.25))["status"] == "passed"
    assert summarize_cost(proof_rows(1.50001, 1.0))["status"] == "failed"
    rows = proof_rows(1.0, 1.0)
    for row in rows:
        if row["sequence_length"] == 256 and row["backbone"] == CANDIDATE:
            row["peak_allocated_bytes"] = 1251
    summary = summarize_cost(rows)
    assert summary["status"] == "failed"
    assert summary["lengths"][-1]["peak_passed"] is False


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("warmup_steps", 4), ("measured_steps", 14),
    ("finite_model_optimizer", False), ("median_step_seconds", 0.0141),
    ("learning_path", {"status": "failed"}),
])
def test_a_cost_ratio_cannot_hide_failed_execution_proof(field, value) -> None:
    rows = proof_rows()
    rows[1][field] = value
    with pytest.raises(ValueError):
        summarize_cost(rows)


def test_nested_optimizer_nonfinite_tensor_is_rejected() -> None:
    optimizer = {"state": {0: {"exp_avg_sq": torch.tensor([1.0, math.nan])}}}
    assert tensor_tree_finite(optimizer) is False
    assert tensor_tree_finite({"state": {0: {"step": torch.tensor(20.0)}}}) is True


def test_fixed_synthetic_batch_has_observed_prefix_and_target() -> None:
    first = synthetic_batch(8, torch.device("cpu"))
    torch.manual_seed(990)
    second = synthetic_batch(8, torch.device("cpu"))
    assert all(torch.equal(left, right) for left, right in zip(first, second))
    dts, quantities, mask = first
    assert dts.shape == quantities.shape == mask.shape == (128, 8)
    assert mask.dtype == torch.bool and bool(mask.all())
    assert bool((dts > 0).all()) and bool((quantities >= 0).all())
    with pytest.raises(ValueError, match="frozen profile grid"):
        synthetic_batch(3, "cpu")


def test_lag_learning_check_rejects_a_live_current_but_dead_past_kernel() -> None:
    parameters = {}
    for name in KERNEL_KEYS:
        parameter = torch.nn.Parameter(torch.zeros(3, HIDDEN_DIM))
        parameter.grad = torch.ones_like(parameter)
        parameters[name] = parameter

    class ParameterCollection:
        def named_parameters(self):
            return parameters.items()

    model = ParameterCollection()
    assert set(kernel_gradient_audit(model)) == set(KERNEL_KEYS)
    parameters[KERNEL_KEYS[1]].grad[2].zero_()
    with pytest.raises(ValueError, match="lag1/lag2 gradient must all be nonzero"):
        kernel_gradient_audit(model)


def test_existing_artifact_is_preserved(tmp_path) -> None:
    output = tmp_path / "profile.json"
    output.write_text('{"status":"previous_proof"}')
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_profile(output)
    assert output.read_text() == '{"status":"previous_proof"}'


def test_cpu_cannot_be_reported_as_cuda_cost_proof(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        run_worker(BASELINE, 8, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA learning-path audit")
def test_cuda_new_lags_train_and_change_selected_final_state() -> None:
    result = run_worker(CANDIDATE, 8, 0)
    assert result["status"] == "passed"
    assert result["learning_path"]["status"] == "passed"
    assert result["learning_path"]["state_digest_restored"] is True
    assert result["learning_path"]["zero_kernel_counterfactual_final_state_max_abs_delta"] > 0.0
    assert result["learning_path"]["zero_kernel_counterfactual_quantity_max_abs_delta"] > 0.0
