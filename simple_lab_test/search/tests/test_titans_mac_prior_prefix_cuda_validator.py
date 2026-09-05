"""CPU fixture tests for mandatory CUDA runtime/cost evidence validation."""

from __future__ import annotations

import argparse
import copy
import json
import statistics

import pytest
import torch

from paper.scripts import validate_titans_mac_prior_prefix_cuda as validator


@pytest.fixture
def contract():
    return json.loads(validator.DEFAULT_CONTRACT.read_text(encoding="utf-8"))


def benchmark_rows(cost):
    rows = []
    for history in cost["history_lengths"]:
        for model in validator.MODELS:
            seconds = {validator.T0: 1.0, validator.B1: 5.0, validator.CANDIDATE: 10.0}[model]
            peak = {validator.T0: 400, validator.B1: 1000, validator.CANDIDATE: 1500}[model]
            samples = [seconds] * cost["measured_steps"]
            rows.append({
                "backbone": model, "history_length": history,
                "sequence_length": history + 1, "batch_size": cost["batch_size"],
                "step_seconds": samples, "median_step_seconds": statistics.median(samples),
                "warmup_steps_completed": cost["warmup_steps"],
                "measured_steps_completed": cost["measured_steps"],
                "peak_allocated_bytes": peak, "finite_steps": True,
                "parameters_finite": True, "parameter_count": 100,
                "initial_parameter_sha256": "a" * 64,
                "input_sha256": "b" * 64,
                "allocated_before_model_bytes": 0, "allocated_after_cleanup_bytes": 0,
            })
    return rows


def candidate_row(rows, history=64):
    return next(row for row in rows if row["backbone"] == validator.CANDIDATE and row["history_length"] == history)


def valid_runtime(expected):
    return {
        "python": expected["expected_python"], "torch": expected["expected_torch"],
        "cuda": expected["expected_cuda"], "polars": expected["expected_polars"],
        "cuda_available": True, "device_name": "NVIDIA GeForce RTX 5090",
        "free_bytes": expected["minimum_free_mib"] * 1024**2,
    }


def test_cost_boundaries_pass_without_reclassifying_T0_cost(contract):
    result = validator.evaluate_cost_gate(benchmark_rows(contract["cost_gate"]), contract["cost_gate"])
    assert result["passed"]
    assert len(result["comparisons"]) == 3
    assert all(row["candidate_to_B1_step_ratio"] == 2.0 for row in result["comparisons"])
    assert all(row["candidate_to_B1_peak_allocated_ratio"] == 1.5 for row in result["comparisons"])
    assert all(row["candidate_to_T0_step_ratio"] == 10.0 and row["T0_ratios_are_report_only"] for row in result["comparisons"])


@pytest.mark.parametrize("failure", ["time", "memory_ratio", "absolute_memory", "different_parameters", "different_inputs"])
def test_one_history_failure_rejects_whole_gate_without_rounding(contract, failure):
    rows = benchmark_rows(contract["cost_gate"])
    candidate = candidate_row(rows)
    if failure == "time":
        candidate["step_seconds"] = [10.0 + 1e-10] * 10
        candidate["median_step_seconds"] = statistics.median(candidate["step_seconds"])
    elif failure == "memory_ratio":
        candidate["peak_allocated_bytes"] = 1501
    elif failure == "absolute_memory":
        candidate["peak_allocated_bytes"] = contract["cost_gate"]["candidate_peak_allocated_bytes_max"] + 1
    elif failure == "different_parameters":
        candidate["initial_parameter_sha256"] = "c" * 64
    else:
        candidate["input_sha256"] = "c" * 64
    result = validator.evaluate_cost_gate(rows, contract["cost_gate"])
    assert not result["passed"]
    assert not next(row for row in result["comparisons"] if row["history_length"] == 64)["passed"]


@pytest.mark.parametrize("failure", ["missing", "duplicate", "steps", "warmup", "nan", "median", "finite", "leaked_allocation", "shape"])
def test_incomplete_or_invalid_measurements_cannot_pass(contract, failure):
    rows = benchmark_rows(contract["cost_gate"])
    candidate = candidate_row(rows)
    if failure == "missing":
        rows.pop()
    elif failure == "duplicate":
        rows.append(copy.deepcopy(rows[-1]))
    elif failure == "steps":
        candidate["step_seconds"].pop()
    elif failure == "warmup":
        candidate["warmup_steps_completed"] = 2
    elif failure == "nan":
        candidate["step_seconds"][0] = float("nan")
    elif failure == "median":
        candidate["median_step_seconds"] = 1.0
    elif failure == "finite":
        candidate["parameters_finite"] = False
    elif failure == "leaked_allocation":
        candidate["allocated_after_cleanup_bytes"] = 4
    else:
        candidate["sequence_length"] = 64
    result = validator.evaluate_cost_gate(rows, contract["cost_gate"])
    assert not result["passed"] and result["errors"]


def test_runtime_requires_every_exact_version_cuda_and_5090(contract):
    expected = contract["runtime"]
    observed = valid_runtime(expected)
    assert all(validator.runtime_version_checks(expected, observed).values())
    for key, wrong in (("python", "3.12.12"), ("torch", "2.11.0"), ("cuda", "12.8"), ("polars", "1.39.2"), ("cuda_available", False), ("device_name", "RTX 5080"), ("free_bytes", observed["free_bytes"] - 1)):
        assert not all(validator.runtime_version_checks(expected, {**observed, key: wrong}).values())


def test_cpu_failure_writes_proof_and_refuses_overwrite(monkeypatch, tmp_path):
    output = tmp_path / "cuda_validation.json"
    monkeypatch.setattr(validator, "observe_runtime", lambda: {
        "python": "CPU fixture", "torch": "CPU fixture", "cuda": None,
        "polars": "CPU fixture", "cuda_available": False,
    })
    monkeypatch.setattr("sys.argv", ["validator", "--output", str(output), "--source-revision", "a" * 40])
    assert validator.main() == 1
    report = json.loads(output.read_text())
    assert report["status"] == "FAIL" and not report["cost_gate_pass"]
    assert report["cuda_available"] is False and report["device"] == "cuda"
    assert report["held_out_test_evaluated"] is False
    assert report["benchmark_rows"] == [] and report["error"]
    before = output.read_bytes()
    with pytest.raises(SystemExit) as caught:
        validator.main()
    assert caught.value.code == 2 and output.read_bytes() == before


def test_profile_exception_preserves_fail_proof_and_partial_matrix(monkeypatch, contract, tmp_path):
    monkeypatch.setattr(validator, "observe_runtime", lambda: valid_runtime(contract["runtime"]))
    import torch._dynamo.config as dynamo_config
    for name in ("recompile_limit", "accumulated_recompile_limit", "suppress_errors"):
        monkeypatch.setattr(dynamo_config, name, getattr(dynamo_config, name))

    def fail_profile(row, frozen):
        row["measured_steps_completed"] = 2
        raise RuntimeError("synthetic compiler failure fixture")

    monkeypatch.setattr(validator, "_profile_one", fail_profile)
    report = validator.validate(argparse.Namespace(
        source_revision="a" * 40, contract=validator.DEFAULT_CONTRACT,
        output=tmp_path / "unused.json",
    ))
    assert report["status"] == "FAIL" and not report["cost_gate_pass"]
    assert len(report["benchmark_rows"]) == 1
    assert report["benchmark_rows"][0]["measured_steps_completed"] == 2
    assert "synthetic compiler failure" in report["error"]["message"]


def test_paired_cpu_factory_initialization_and_synthetic_inputs(contract):
    torch.set_num_threads(1)
    baseline = validator._build_model(validator.B1, contract)
    candidate = validator._build_model(validator.CANDIDATE, contract)
    assert validator.parameter_digest(baseline) == validator.parameter_digest(candidate)
    assert baseline.time_intercept_limit == candidate.time_intercept_limit == 300.0
    assert candidate.titans_mac_encoder.output_read_policy == "prior_prefix"
    assert baseline.titans_mac_encoder.output_read_policy == "segment_start"
    assert candidate.titans_mac_encoder.neural_memory.gradient_max_norm == 1.0
    for history in contract["cost_gate"]["history_lengths"]:
        first = validator.make_synthetic_batch(history, 128)
        second = validator.make_synthetic_batch(history, 128)
        assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
        assert all(t.shape == (128, history + 1) for t in first)
        assert bool((first[0] > 0).all()) and bool((first[2] > 0).all())
        assert bool(first[1].all())
