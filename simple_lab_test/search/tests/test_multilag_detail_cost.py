"""Synthetic-only qualification tests: no GPU, SSH or data files are used."""
from __future__ import annotations

import json

import pytest
import torch

from paper.scripts import verify_multilag_detail_cost as verifier


def test_cpu_qualification_checks_and_cost_accounting(tmp_path):
    output = tmp_path / "receipt.json"
    initial_rng, initial_threads = torch.get_rng_state().clone(), torch.get_num_threads()
    result = verifier.run(output, device="cpu", batch_size=2, lengths=(8,),
                          warmup_steps=1, measured_steps=1, max_wall_seconds=120.)
    assert result == json.loads(output.read_text())
    assert result["status"] == "cpu_observation_complete"
    assert result["cuda_qualification"] is False and result["device"] == "cpu"
    assert result["real_data_loaded"] is False and result["held_out_evaluated"] is False
    assert result["synthetic_optimizer_updates"] == 12
    assert set(result["checks"]) == set(verifier.QUALIFICATION_CHECKS)
    assert all(result["checks"].values())
    assert torch.equal(initial_rng, torch.get_rng_state())
    assert initial_threads == torch.get_num_threads()
    row = result["costs"][0]
    assert row["length"] == 8 and row["batch_size"] == 2
    assert len(row["synthetic_batch_sha256"]) == 64
    assert set(row["measurements"]) == set(verifier.ARMS)
    for arm, count in zip(verifier.ARMS, (89859, 96003, 96003)):
        value = row["measurements"][arm]
        assert value["parameters"] == count
        assert value["peak_allocated_bytes"] is None
        assert len(value["step_seconds"]) == 1
        assert value["median_step_seconds"] > 0
    assert set(row["checks"]) == {arm + ":parameters" for arm in verifier.ARMS[1:]}
    for record in result["branch_gradient_checks"].values():
        assert len(record["activated_finite_nonzero"]) == 24
        assert all(record["activated_finite_nonzero"].values())
        assert all(value for name, value in record["initial_nonzero"].items() if ".output_projections." in name)
        assert all(not value for name, value in record["initial_nonzero"].items() if ".output_projections." not in name)


@pytest.mark.parametrize("override", [
    {"device": "mps"}, {"device": "cuda:1"}, {"batch_size": 0},
    {"batch_size": True}, {"batch_size": 129}, {"lengths": ()},
    {"lengths": (64, 64)}, {"lengths": (1,)}, {"lengths": (257,)},
    {"lengths": (True,)}, {"warmup_steps": 0}, {"measured_steps": 0},
    {"max_wall_seconds": float("nan")}, {"cost_gates": {"parameter_ratio_max": 2.}},
])
def test_invalid_options_rejected_before_model_build(override, monkeypatch):
    from models.TPPs import CountAwareFactory
    monkeypatch.setattr(CountAwareFactory, "build_count_aware_model", lambda *a, **k: pytest.fail("Invalid options reached model construction"))
    with pytest.raises(RuntimeError):
        verifier.run(**override)


def test_existing_receipt_is_immutable(tmp_path):
    output = tmp_path / "receipt.json"
    output.write_text("original")
    with pytest.raises(RuntimeError, match="overwrite"):
        verifier.run(output)
    assert output.read_text() == "original"


def test_cuda_request_cannot_fall_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_device", lambda *_: pytest.fail("Unavailable CUDA was initialized"))
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        verifier.run(device="cuda:0")


def test_external_budget_checked_before_device_initialization(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Expired budget touched CUDA"))
    def expired():
        raise ValueError("shared deadline expired")
    with pytest.raises(ValueError, match="shared deadline"):
        verifier.run(device="cuda:0", budget=expired)


def test_rng_and_thread_state_restored_after_correctness_failure(monkeypatch):
    state, threads = torch.get_rng_state().clone(), torch.get_num_threads()
    def fail(*_):
        raise RuntimeError("synthetic injected correctness failure")
    monkeypatch.setattr(verifier, "_direct_reference_check", fail)
    with pytest.raises(RuntimeError, match="injected"):
        verifier.run(batch_size=1, lengths=(2,), warmup_steps=1, measured_steps=1)
    assert torch.equal(state, torch.get_rng_state())
    assert threads == torch.get_num_threads()


def test_cuda_cost_gates_compute_from_raw_measurements():
    measurements = {arm: {"parameters": 100 if arm == verifier.ARMS[0] else 110,
        "median_step_seconds": 1. if arm == verifier.ARMS[0] else 1.5,
        "peak_allocated_bytes": 100 if arm == verifier.ARMS[0] else 125} for arm in verifier.ARMS}
    checks = verifier._cost_checks(measurements, verifier.DEFAULT_COST_GATES, device_memory=1000)
    assert len(checks) == 8 and all(checks.values())
    measurements[verifier.ARMS[2]]["median_step_seconds"] = 1.5001
    measurements[verifier.ARMS[1]]["peak_allocated_bytes"] = 126
    checks = verifier._cost_checks(measurements, verifier.DEFAULT_COST_GATES, device_memory=1000)
    assert checks[verifier.ARMS[2] + ":step"] is False
    assert checks[verifier.ARMS[1] + ":memory"] is False
