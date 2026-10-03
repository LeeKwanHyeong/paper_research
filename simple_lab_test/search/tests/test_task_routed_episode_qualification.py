"""Synthetic CPU and mocked-device qualification checks; never start CUDA or SSH."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

from paper.scripts import task_routed_episode_parallel_common as common
from paper.scripts import qualify_task_routed_episode_parallel as qualifier
from paper.scripts import verify_task_routed_episode_cost as cost


@pytest.fixture(scope="module")
def cpu_receipt():
    threads = torch.get_num_threads()
    cpu_rng = torch.get_rng_state().clone()
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark = torch.backends.cudnn.benchmark
    cudnn_deterministic = torch.backends.cudnn.deterministic
    torch.set_num_threads(1)
    try:
        # The CPU path must not even seed or read a CUDA generator.
        with pytest.MonkeyPatch.context() as patch:
            def forbidden(name):
                def call(*args, **kwargs):
                    raise AssertionError("CPU qualification touched CUDA: " + name)
                return call
            for name in ("manual_seed_all", "get_rng_state", "set_rng_state", "synchronize"):
                patch.setattr(torch.cuda, name, forbidden(name))
            receipt = cost.verify(batch_size=2)
        assert torch.equal(cpu_rng, torch.get_rng_state())
        assert torch.are_deterministic_algorithms_enabled() == deterministic
        assert torch.is_deterministic_algorithms_warn_only_enabled() == warn_only
        assert torch.backends.cudnn.benchmark == benchmark
        assert torch.backends.cudnn.deterministic == cudnn_deterministic
        return receipt
    finally:
        torch.set_num_threads(threads)


def test_cpu_qualifier_runs_real_initial_gradient_causality_and_restore_checks(cpu_receipt):
    assert cpu_receipt["status"] == "synthetic_checks_passed"
    assert cpu_receipt["device"] == "cpu"
    assert cpu_receipt["real_data_loaded"] is False
    assert cpu_receipt["held_out_evaluated"] is False
    assert cpu_receipt["synthetic_optimizer_updates"] == 48
    assert set(cpu_receipt["checks"]) == set(common.ARMS[1:])
    assert all(result == {key: True for key in cost.CHECK_KEYS}
               for result in cpu_receipt["checks"].values())
    assert all(cpu_receipt["empty_read_checks"].values())
    assert cpu_receipt["new_arm_initial_parameters_exact"] is True
    expected_routes = {}
    for arm, routing in zip(common.ARMS[1:], ("shared", "split")):
        expected_routes[arm] = {
            "routing": routing,
            "time": {"time_query": True, "quantity_query": routing == "shared",
                     "shared_key_value": True, "time_output": True, "quantity_output": False},
            "quantity": {"time_query": routing == "shared", "quantity_query": True,
                         "shared_key_value": True, "time_output": False, "quantity_output": True},
        }
    assert cpu_receipt["task_route_gradient_checks"] == expected_routes
    for row in cpu_receipt["costs"]:
        assert row["batch"] == 2 and row["updates_per_arm"] == 7
        assert row["length"] in (64, 256) and row["max_seq_len"] == row["length"]
        baseline = row["measurements"][common.ARMS[0]]
        assert baseline["parameters"] == {64: 77507, 256: 89795}[row["length"]]
        assert all(value["peak_allocated_bytes"] is None for value in row["measurements"].values())
        for name in common.ARMS[1:]:
            assert row["measurements"][name]["parameters"] == baseline["parameters"] + 3632
            assert len(row["measurements"][name]["step_seconds"]) == 5


def contract_fixture(tmp_path=None):
    host = {"root": str(tmp_path or "/unused"), "source_root": str(common.ROOT), "python": sys.executable}
    return {"hosts": {name: {**host, "alias": name} for name in ("5080", "5090")},
            "limits": copy.deepcopy(common.LIMITS), "cost_gates": copy.deepcopy(common.COST_GATES),
            "source": {"files_sha256": "1" * 64}}


def native_receipt():
    result = {"status": "synthetic_checks_passed", "device": "cuda:0",
              "synthetic_optimizer_updates": 48, "real_data_loaded": False, "held_out_evaluated": False,
              "checks": {name: {key: True for key in cost.CHECK_KEYS} for name in common.ARMS[1:]}, "costs": []}
    for length in (64, 256):
        measurements = {name: {"parameters": {64: 77507, 256: 89795}[length] + (0 if name == common.ARMS[0] else 3632),
                              "step_seconds": [1.] * 5, "median_step_seconds": 1.,
                              "peak_allocated_bytes": 100000} for name in common.ARMS}
        result["costs"].append({"batch": 128, "length": length, "max_seq_len": length,
                                "updates_per_arm": 7, "measurements": measurements})
    return result


@pytest.mark.parametrize("device", ["cuda", "cuda:1", "mps"])
def test_unpinned_devices_are_rejected_before_loading_torch_state(device):
    with pytest.raises(ValueError, match="pinned CUDA"):
        cost.verify(device)


@pytest.mark.parametrize("value", [0, 129, True, 1.5])
def test_batch_ceiling_is_enforced(value):
    with pytest.raises(ValueError, match="Synthetic batch"):
        cost.verify(batch_size=value)


@pytest.mark.parametrize("owner,started", [(None, "1000"), ("wrong", "1000"), ("parent", "nan"),
                                           ("parent", "invalid"), ("parent", "100"), ("parent", "1200")])
def test_cuda_requires_live_supervising_parent_before_any_device_access(monkeypatch, owner, started):
    if owner is None:
        monkeypatch.delenv(cost.OWNER_ENV, raising=False)
    else:
        monkeypatch.setenv(cost.OWNER_ENV, str(os.getppid()) if owner == "parent" else owner)
    monkeypatch.setenv(cost.STARTED_ENV, started)
    monkeypatch.setattr(cost.time, "time", lambda: 1100.)
    monkeypatch.setattr(cost, "_verify", lambda *a, **kw: pytest.fail("Native work was attempted"))
    with pytest.raises(ValueError, match="supervising parent|deadline"):
        cost.verify("cuda:0")


def test_cpu_cli_has_no_cuda_device_or_approval_bypass(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["cost", "--output", str(tmp_path / "receipt.json"), "--device", "cuda:0"])
    monkeypatch.setattr(cost, "verify", lambda *a, **kw: pytest.fail("Measurement was attempted"))
    with pytest.raises(SystemExit) as error:
        cost.main()
    assert error.value.code == 2


@pytest.mark.parametrize("gate", ["step", "relative_memory", "device_memory", "parameters"])
def test_cost_gate_at_bound_and_above_is_detected(gate):
    receipt, contract = native_receipt(), contract_fixture()
    runtime = {"gpu": {"total_memory_bytes": 1000000}}
    value = receipt["costs"][0]["measurements"][common.ARMS[1]]
    if gate == "step":
        value.update(step_seconds=[2.5] * 5, median_step_seconds=2.5)
    elif gate == "relative_memory":
        value["peak_allocated_bytes"] = 250000
    elif gate == "device_memory":
        runtime["gpu"]["total_memory_bytes"] = 125000
    else:
        contract["cost_gates"]["parameter_ratio_max"] = 81139 / 77507
    assert cost.cost_gates(receipt, contract, runtime)[0][gate] is True
    if gate == "step":
        value.update(step_seconds=[2.50001] * 5, median_step_seconds=2.50001)
    elif gate in ("relative_memory", "device_memory"):
        value["peak_allocated_bytes"] += 1
    else:
        contract["cost_gates"]["parameter_ratio_max"] -= .00001
    assert cost.cost_gates(receipt, contract, runtime)[0][gate] is False


@pytest.mark.parametrize("corruption", ["cpu", "updates", "batch", "shape", "positional_length", "samples", "nan", "median", "memory", "parameters"])
def test_cost_gates_reject_incomplete_or_nonfinite_evidence(corruption):
    receipt, contract = native_receipt(), contract_fixture()
    row = receipt["costs"][0]
    value = row["measurements"][common.ARMS[1]]
    if corruption == "cpu":
        receipt["device"] = "cpu"
    elif corruption == "updates":
        receipt["synthetic_optimizer_updates"] = 47
    elif corruption == "batch":
        row["batch"] = 2
    elif corruption == "shape":
        row["length"] = 256
    elif corruption == "positional_length":
        row["max_seq_len"] = 256
    elif corruption == "samples":
        value["step_seconds"].pop()
    elif corruption == "nan":
        value["step_seconds"][0] = float("nan")
    elif corruption == "median":
        value["median_step_seconds"] = 2.
    elif corruption == "memory":
        value["peak_allocated_bytes"] = None
    else:
        value["parameters"] = 0
    with pytest.raises(ValueError):
        cost.cost_gates(receipt, contract, {"gpu": {"total_memory_bytes": 1000000}})


@pytest.mark.parametrize("host", ["5080", "5090"])
def test_native_qualifier_uses_bound_host_and_rechecks_source_without_gpu(monkeypatch, host):
    contract, calls = contract_fixture(), []
    inventories = iter((set(), {os.getpid()}))
    runtime = {"gpu": {"total_memory_bytes": 1000000, "uuid": "fixture-" + host}}
    monkeypatch.setattr(qualifier.time, "time", lambda: 1100.)
    monkeypatch.setattr(common, "verify_source", lambda _: calls.append("source"))
    monkeypatch.setattr(common, "gpu_pids", lambda _: next(inventories))
    monkeypatch.setattr(common, "runtime_check", lambda _, name: calls.append(name) or runtime)
    monkeypatch.setattr(qualifier, "verify", lambda device: calls.append(device) or native_receipt())
    receipt = qualifier.qualify(contract, host, 1000.)
    assert calls == ["source", host, "cuda:0", "source"]
    assert receipt["status"] == "passed" and receipt["host"] == host
    assert receipt["started_at_unix"] == 1000. and receipt["completed_at_unix"] == 1100.
    assert receipt["source_files_sha256"] == contract["source"]["files_sha256"]
    assert all(all(row[key] for key in ("step", "relative_memory", "device_memory", "parameters"))
               for row in receipt["cost_gate_checks"])


@pytest.mark.parametrize("condition", ["initial_busy", "foreign_job", "deadline"])
def test_native_qualifier_refuses_busy_or_expired_work_without_stopping_others(monkeypatch, condition):
    contract = contract_fixture()
    moments = iter((1100., 1901. if condition == "deadline" else 1100.))
    inventories = iter(({987654} if condition == "initial_busy" else set(),
                        {987654} if condition == "foreign_job" else {os.getpid()}))
    monkeypatch.setattr(qualifier.time, "time", lambda: next(moments))
    monkeypatch.setattr(common, "verify_source", lambda _: None)
    monkeypatch.setattr(common, "gpu_pids", lambda _: next(inventories))
    monkeypatch.setattr(common, "runtime_check", lambda *a: {"gpu": {"total_memory_bytes": 1000000}})
    monkeypatch.setattr(qualifier, "verify", lambda _: native_receipt())
    with pytest.raises(ValueError, match="GPU busy|Another GPU job|deadline"):
        qualifier.qualify(contract, "5090", 1000.)


def supervisor_fixture(monkeypatch, tmp_path):
    contract = contract_fixture(tmp_path)
    approval = tmp_path / "approval.json"
    approval.write_text("{}")
    monkeypatch.setattr(sys, "argv", ["qualifier", "--host", "5090", "--contract", str(tmp_path / "contract.json"),
                                     "--approval", str(approval)])
    monkeypatch.setattr(common, "read_contract", lambda path, host: contract)
    monkeypatch.setattr(common, "verify_approval", lambda *args: None)
    monkeypatch.setattr(common, "apply_environment", lambda *args: None)
    monkeypatch.setattr(common, "gpu_pids", lambda *args: set())
    monkeypatch.setattr(qualifier.time, "time", lambda: 1000.)
    return contract


def test_supervisor_timeout_kills_only_owned_child_and_records_no_retry(monkeypatch, tmp_path):
    contract = supervisor_fixture(monkeypatch, tmp_path)
    launched, killed = [], []
    class OwnedChild:
        def wait(self, timeout):
            assert timeout == 900
            raise subprocess.TimeoutExpired("owned child", timeout)
    child = OwnedChild()
    def launch(command, **kwargs):
        launched.append((command, kwargs))
        return child
    monkeypatch.setattr(qualifier.subprocess, "Popen", launch)
    monkeypatch.setattr(qualifier, "kill_owned_process_group", lambda value: killed.append(value))
    with pytest.raises(subprocess.TimeoutExpired):
        qualifier.main()
    assert killed == [child] and len(launched) == 1
    command, kwargs = launched[0]
    assert "--child" in command and kwargs["start_new_session"] is True
    assert kwargs["env"][cost.OWNER_ENV] == str(os.getpid())
    assert kwargs["env"][cost.STARTED_ENV] == "1000.0"
    start = json.loads((tmp_path / "qualification" / "start.json").read_text())
    assert start["supervisor_pid"] == os.getpid() and start["qualification_deadline_unix"] == 1900.
    failure = json.loads((tmp_path / "qualification" / "failure.json").read_text())
    assert failure["automatic_retry"] is False and failure["status"] == "qualification_stopped"
    assert failure["contract_sha256"] == common.sha_json(contract)


def test_supervisor_preserves_an_existing_qualification_directory(monkeypatch, tmp_path):
    supervisor_fixture(monkeypatch, tmp_path)
    output = tmp_path / "qualification"
    output.mkdir()
    previous = output / "receipt.json"
    previous.write_text("existing evidence")
    monkeypatch.setattr(qualifier.subprocess, "Popen", lambda *a, **kw: pytest.fail("Duplicate qualification started"))
    with pytest.raises(FileExistsError):
        qualifier.main()
    assert previous.read_text() == "existing evidence"


def test_child_rejects_foreign_supervisor_start_receipt(monkeypatch, tmp_path):
    contract = supervisor_fixture(monkeypatch, tmp_path)
    monkeypatch.setattr(sys, "argv", [*sys.argv, "--child"])
    monkeypatch.setenv(cost.OWNER_ENV, str(os.getppid()))
    monkeypatch.setenv(cost.STARTED_ENV, "1000.0")
    output = tmp_path / "qualification"
    output.mkdir()
    (output / "start.json").write_text(json.dumps({
        "host": "5090", "contract_sha256": common.sha_json(contract), "started_at_unix": 1000.,
        "qualification_deadline_unix": 1900., "supervisor_pid": os.getppid() + 1,
    }))
    monkeypatch.setattr(qualifier, "qualify", lambda *a: pytest.fail("Foreign child reached CUDA qualifier"))
    with pytest.raises(ValueError, match="owning supervisor"):
        qualifier.main()
