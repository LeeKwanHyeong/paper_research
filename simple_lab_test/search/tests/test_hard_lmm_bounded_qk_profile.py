"""CPU contracts for the three-model RTX 5090 profile; never CPU cost claims."""

from __future__ import annotations

import copy
import math

import pytest
import torch

from paper.scripts import profile_hard_lmm_bounded_qk as profile


def proof_rows(step_ratio=1.4, peak_ratio=1.2):
    result = []
    for length in profile.LENGTHS:
        for repeat in range(profile.REPEATS):
            for backbone in profile.BACKBONES:
                factor = {profile.BASELINE: 1., profile.FULL: 1.2, profile.CANDIDATE: step_ratio}[backbone]
                memory_factor = {profile.BASELINE: 1., profile.FULL: 1.1, profile.CANDIDATE: peak_ratio}[backbone]
                learning = {"status": "passed"}
                if backbone == profile.CANDIDATE:
                    learning["bounded_gain"] = {"status": "passed"}
                result.append({
                    "status": "passed", "profile_contract": profile.PROFILE_CONTRACT,
                    "backbone": backbone, "sequence_length": length, "repeat": repeat,
                    "batch_size": profile.BATCH_SIZE, "hidden_dim": profile.HIDDEN_DIM,
                    "warmup_steps": profile.WARMUP_STEPS, "measured_steps": profile.MEASURED_STEPS,
                    "seed": 42 + repeat, "tf32_matmul": False, "amp_enabled": False,
                    "measurement_dtype": "float32", "device_name": "NVIDIA GeForce RTX 5090",
                    "torch_version": "synthetic_fixture", "cuda_version": "synthetic_fixture",
                    "step_seconds": [.01 * factor] * profile.MEASURED_STEPS,
                    "median_step_seconds": .01 * factor,
                    "peak_allocated_bytes": 1000 * memory_factor,
                    "finite_model_optimizer": True, "gradient_audit": {"status": "passed"},
                    "learning_path": learning,
                    "parameter_count": 1000 if backbone == profile.BASELINE else 1576,
                })
    return result


def test_repeat_order_places_each_model_in_each_position():
    orders = [profile.measurement_order(repeat) for repeat in range(profile.REPEATS)]
    assert orders[0] == profile.BACKBONES
    for position in range(3):
        assert {order[position] for order in orders} == set(profile.BACKBONES)
    for invalid in (-1, 3, True):
        with pytest.raises(ValueError):
            profile.measurement_order(invalid)


def test_complete_grid_and_separate_full_ratios():
    rows = proof_rows()
    summary = profile.summarize_cost(rows)
    assert summary["status"] == "passed"
    assert summary["gate_reference"] == profile.BASELINE
    for length in summary["lengths"]:
        assert math.isclose(length["candidate_over_baseline_step"], 1.4)
        assert math.isclose(length["candidate_over_full_step"], 1.4 / 1.2)
        assert math.isclose(length["full_over_baseline_step"], 1.2)
    for invalid in (rows[:-1], rows + [rows[0]], rows[:-1] + [rows[0]]):
        with pytest.raises(ValueError, match="complete three-model grid"):
            profile.summarize_cost(invalid)


def test_each_length_must_pass_both_candidate_over_b_limits():
    assert profile.summarize_cost(proof_rows(1.5, 1.25))["status"] == "passed"
    assert profile.summarize_cost(proof_rows(1.50001, 1.25))["status"] == "failed"
    rows = proof_rows(1., 1.)
    for row in rows:
        if row["sequence_length"] == 256 and row["backbone"] == profile.CANDIDATE:
            row["peak_allocated_bytes"] = 1251
    result = profile.summarize_cost(rows)
    assert result["status"] == "failed"
    assert result["lengths"][-1]["peak_passed"] is False


@pytest.mark.parametrize("field,value", [
    ("status", "failed"), ("profile_contract", "old_profile"),
    ("warmup_steps", 4), ("measured_steps", 14), ("seed", 999),
    ("amp_enabled", True), ("device_name", "NVIDIA GeForce RTX 5080"),
    ("finite_model_optimizer", False), ("median_step_seconds", 0.0),
    ("peak_allocated_bytes", float("nan")), ("gradient_audit", {"status": "failed"}),
    ("learning_path", {"status": "failed"}), ("parameter_count", 1577),
])
def test_failed_runtime_measurement_or_learning_cannot_qualify(field, value):
    rows = proof_rows()
    rows[0][field] = value
    with pytest.raises(ValueError):
        profile.summarize_cost(rows)


def test_dead_bound_and_mixed_runtime_are_rejected():
    rows = proof_rows()
    rows[2]["learning_path"]["bounded_gain"] = {"status": "failed"}
    with pytest.raises(ValueError, match="active Q/K bound"):
        profile.summarize_cost(rows)
    rows = proof_rows()
    rows[0]["cuda_version"] = "another runtime"
    with pytest.raises(ValueError, match="mixes runtimes"):
        profile.summarize_cost(rows)


def test_cpu_and_wrong_gpu_cannot_qualify_even_direct_workers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA is required"):
        profile.run_worker(profile.BASELINE, 8, 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *_: "NVIDIA GeForce RTX 5080")
    with pytest.raises(RuntimeError, match="RTX 5090"):
        profile.run_worker(profile.CANDIDATE, 8, 0)


def test_existing_proof_is_not_overwritten(tmp_path):
    output = tmp_path / "cost.json"
    output.write_text('{"status":"existing"}')
    with pytest.raises(FileExistsError):
        profile.run_profile(output)
    assert output.read_text() == '{"status":"existing"}'


@pytest.mark.parametrize("backbone", profile.BACKBONES)
def test_actual_cpu_gradient_and_learning_audits_restore_every_backbone(backbone):
    """Test audit logic on CPU without invoking the CUDA-only cost worker."""
    from models.TPPs.CountAwareFactory import build_count_aware_model
    from paper.scripts.count_aware_tpp_backbone.core import target_outputs

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(42)
        model, _ = build_count_aware_model(
            backbone, hidden_dim=64, train_log_mean=2., max_seq_len=256,
            quantity_variant="count_only_log_regression", lambda_tail=0.,
        )
        initial = profile.initial_backbone_state(model)
        initial_digest = profile.state_digest(model)
        batch = profile.synthetic_batch(8, "cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        model.train()
        for index in range(2):
            dts, quantities, mask = batch
            optimizer.zero_grad(set_to_none=True)
            target_outputs(model, dts, mask, quantities, lambda_log_qty=1.)["joint_loss"].mean().backward()
            if index == 0:
                gradient = profile.gradient_audit(model, backbone)
                assert gradient["status"] == "passed"
                assert set(gradient["shared_gradient_norms"]) == set(profile.COMMON_GRADIENT_KEYS)
                assert set(gradient["kernel_gradient_norm_by_lag"]) == set(profile.KERNEL_KEYS.get(backbone, ()))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
        before = profile.state_digest(model)
        assert before != initial_digest
        optimizer_before = copy.deepcopy(optimizer.state_dict())
        audit = profile.learned_path_audit(model, backbone, batch, initial)
        assert audit["status"] == "passed"
        assert audit["initial_backbone_counterfactual"]["state_digest_restored"] is True
        assert profile.state_digest(model) == before
        assert model.training is True
        if backbone in profile.KERNEL_KEYS:
            assert audit["zero_kernel_counterfactual"]["state_digest_restored"] is True
        if backbone == profile.CANDIDATE:
            assert audit["bounded_gain"]["v_operation_bitwise_identical"] is True
        for parameter_id, state in optimizer_before["state"].items():
            for name, value in state.items():
                assert torch.equal(value, optimizer.state_dict()["state"][parameter_id][name])
    finally:
        torch.set_num_threads(previous)
