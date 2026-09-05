"""CPU fixture tests for mandatory CUDA runtime/cost evidence validation."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
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
                "measurement_isolation": "fresh_subprocess",
                "worker_process_exited": True, "worker_evidence_verified": True,
                "worker_returncode": 0,
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


@pytest.mark.parametrize("failure", ["missing", "duplicate", "steps", "warmup", "nan", "median", "finite", "leaked_allocation", "shape", "no_subprocess", "worker_running", "worker_unverified", "worker_failed", "no_cleanup_telemetry"])
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
        candidate["allocated_before_model_bytes"] = 4
    elif failure == "no_subprocess":
        candidate["measurement_isolation"] = "same_process"
    elif failure == "worker_running":
        candidate["worker_process_exited"] = False
    elif failure == "worker_unverified":
        candidate["worker_evidence_verified"] = False
    elif failure == "worker_failed":
        candidate["worker_returncode"] = 1
    elif failure == "no_cleanup_telemetry":
        candidate.pop("allocated_after_cleanup_bytes")
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


def test_process_local_workspace_does_not_contaminate_a_fresh_worker(contract):
    rows = benchmark_rows(contract["cost_gate"])
    for row in rows:
        row["allocated_after_cleanup_bytes"] = 68157440
    assert validator.evaluate_cost_gate(rows, contract["cost_gate"])["passed"]


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

    def fail_profile(row, args, binding, evidence_dir):
        row["measured_steps_completed"] = 2
        raise RuntimeError("synthetic compiler failure fixture")

    monkeypatch.setattr(validator, "_profile_in_subprocess", fail_profile)
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


def fake_worker_processes(monkeypatch, contract, *, failure=None):
    commands = []

    class FakeProcess:
        def __init__(self, command, *, cwd):
            assert cwd == validator.PROJECT_ROOT
            assert command[1:3] == ["-s", "-B"]
            assert Path(command[3]) == Path(validator.__file__).resolve()
            commands.append(command)
            self.pid = os.getpid() + 100000 + len(commands)
            self.returncode = None
            self.command = command

        def wait(self):
            def argument(name):
                return self.command[self.command.index(name) + 1]

            request = json.loads(Path(argument("--worker-request")).read_text())
            assert request["parent_pid"] == os.getpid()
            assert request["source_revision"] == argument("--source-revision")
            assert request["contract_sha256"] == validator.sha256(Path(argument("--contract")))
            assert request["source_file_sha256"] == validator._source_file_hashes()
            requested = request["benchmark_row"]
            row = next(copy.deepcopy(row) for row in benchmark_rows(contract["cost_gate"])
                       if (row["backbone"], row["history_length"]) == (requested["backbone"], requested["history_length"]))
            for key in ("measurement_isolation", "worker_process_exited", "worker_evidence_verified", "worker_returncode"):
                row.pop(key)
            row["allocated_after_cleanup_bytes"] = 68157440
            result = {
                "status": "PASS", "source_revision": request["source_revision"],
                "contract_sha256": request["contract_sha256"],
                "source_file_sha256": request["source_file_sha256"],
                "worker_pid": self.pid, "worker_parent_pid": os.getpid(),
                "measurement_isolation": "fresh_subprocess", "benchmark_row": row,
                "checks": {"execution_completed": True},
                "runtime": valid_runtime(contract["runtime"]),
                "held_out_test_evaluated": False, "real_data_used": False,
                "checkpoint_loaded": False,
            }
            self.returncode = 0
            if failure == "partial":
                result["status"] = "FAIL"
                result["checks"]["execution_completed"] = False
                result["error"] = {"type": "RuntimeError", "message": "fixture compile failure"}
                row["measured_steps_completed"] = 2
                row["step_seconds"] = row["step_seconds"][:2]
                self.returncode = 1
            elif failure in ("source_file_sha256", "contract_sha256", "worker_pid"):
                result[failure] = None
            elif failure == "missing_report":
                self.returncode = -9
                return self.returncode
            validator._write_report(Path(argument("--output")), result)
            return self.returncode

    monkeypatch.setattr(validator.subprocess, "Popen", FakeProcess)
    return commands


def test_parent_dispatches_nine_new_workers_and_preserves_nonzero_cleanup(monkeypatch, contract, tmp_path):
    monkeypatch.setattr(validator, "observe_runtime", lambda: valid_runtime(contract["runtime"]))
    monkeypatch.setattr(validator, "_configure_dynamo", lambda frozen: {"suppress_errors": False})
    monkeypatch.setattr(validator, "_profile_one", lambda *args: pytest.fail("Parent must never profile a model"))
    commands = fake_worker_processes(monkeypatch, contract)
    report = validator.validate(argparse.Namespace(
        source_revision="a" * 40, contract=validator.DEFAULT_CONTRACT, output=tmp_path / "validation.json",
    ))
    assert report["status"] == "PASS" and report["cost_gate_evaluated"] and report["cost_gate_pass"]
    assert len(commands) == len(report["benchmark_rows"]) == 9
    assert len({row["worker_pid"] for row in report["benchmark_rows"]}) == 9
    assert all(row["allocated_before_model_bytes"] == 0 and row["allocated_after_cleanup_bytes"] == 68157440
               and row["worker_process_exited"] and row["worker_evidence_verified"] for row in report["benchmark_rows"])
    directory = Path(report["worker_evidence_directory"])
    assert len(list(directory.glob("*.request.json"))) == len(list(directory.glob("*.result.json"))) == 9


@pytest.mark.parametrize("failure", ["partial", "source_file_sha256", "contract_sha256", "worker_pid", "missing_report"])
def test_parent_rejects_failed_or_unbound_worker_and_preserves_partial_evidence(monkeypatch, contract, tmp_path, failure):
    monkeypatch.setattr(validator, "observe_runtime", lambda: valid_runtime(contract["runtime"]))
    monkeypatch.setattr(validator, "_configure_dynamo", lambda frozen: {"suppress_errors": False})
    commands = fake_worker_processes(monkeypatch, contract, failure=failure)
    report = validator.validate(argparse.Namespace(
        source_revision="a" * 40, contract=validator.DEFAULT_CONTRACT, output=tmp_path / "validation.json",
    ))
    assert report["status"] == "FAIL" and not report["cost_gate_evaluated"] and not report["cost_gate_pass"]
    assert len(commands) == len(report["benchmark_rows"]) == 1
    row = report["benchmark_rows"][0]
    assert row["worker_process_exited"]
    assert Path(row["worker_request_path"]).is_file()
    if failure == "partial":
        assert row["measured_steps_completed"] == len(row["step_seconds"]) == 2
        assert row["worker_error"]["message"] == "fixture compile failure"
    elif failure != "missing_report":
        assert not row["worker_evidence_verified"]


def worker_args(tmp_path):
    request_path = tmp_path / "request.json"
    validator._write_report(request_path, {
        "source_revision": "a" * 40,
        "contract_sha256": validator.sha256(validator.DEFAULT_CONTRACT),
        "source_file_sha256": validator._source_file_hashes(),
        "parent_pid": os.getppid(),
        "benchmark_row": {"backbone": validator.T0, "history_length": 16, "sequence_length": 17, "batch_size": 128},
    })
    return argparse.Namespace(
        source_revision="a" * 40, contract=validator.DEFAULT_CONTRACT,
        worker_request=request_path, output=tmp_path / "result.json",
    )


@pytest.mark.parametrize("failure", [None, "partial", "source_revision", "contract_sha256", "source_file_sha256", "parent_pid", "shape"])
def test_worker_checks_binding_before_profiling_and_retains_partial_row(monkeypatch, contract, tmp_path, failure):
    args = worker_args(tmp_path)
    calls = []
    monkeypatch.setattr(validator, "observe_runtime", lambda: valid_runtime(contract["runtime"]))
    monkeypatch.setattr(validator, "_configure_dynamo", lambda frozen: {"suppress_errors": False})

    def profile(row, frozen):
        calls.append(copy.deepcopy(row))
        row["measured_steps_completed"] = 2 if failure == "partial" else 10
        if failure == "partial":
            raise RuntimeError("worker fixture partial failure")

    monkeypatch.setattr(validator, "_profile_one", profile)
    if failure not in (None, "partial"):
        request = json.loads(args.worker_request.read_text())
        if failure == "shape":
            request["benchmark_row"]["sequence_length"] = 16
        else:
            request[failure] = None
        args.worker_request.write_text(json.dumps(request))
    report = validator.validate_worker(args)
    assert report["worker_pid"] == os.getpid()
    assert report["worker_parent_pid"] == os.getppid()
    if failure is None:
        assert report["status"] == "PASS" and len(calls) == 1
        assert report["contract_sha256"] == validator.sha256(validator.DEFAULT_CONTRACT)
        assert report["source_file_sha256"] == validator._source_file_hashes()
    else:
        assert report["status"] == "FAIL" and report["checks"]["execution_completed"] is False
        if failure == "partial":
            assert len(calls) == 1 and report["benchmark_row"]["measured_steps_completed"] == 2
        else:
            assert not calls
