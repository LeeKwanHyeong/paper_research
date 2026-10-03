"""CPU-only qualification tests; no CUDA, research inputs or SSH."""
from copy import deepcopy
import subprocess
import sys
from types import SimpleNamespace

import pytest

from paper.scripts import run_raw_aux_gradient_cuda as qualifier


def test_import_is_stdlib_only():
    source = "import sys; import paper.scripts.run_raw_aux_gradient_cuda; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules"
    result = subprocess.run([sys.executable, "-B", "-c", source], check=True, capture_output=True, text=True)
    assert result.returncode == 0


@pytest.mark.parametrize("host", ["5080", "5090"])
def test_output_is_contracted(tmp_path, host):
    contract = {"hosts": {host: {"root": str(tmp_path)}}}
    assert qualifier._output_path(contract, host, tmp_path / "qualification") == tmp_path / "qualification"
    with pytest.raises(ValueError, match="contracted"):
        qualifier._output_path(contract, host, tmp_path / "run")


def test_json_writer_exclusive_finite_bounded(tmp_path, monkeypatch):
    target = tmp_path / "evidence.json"
    qualifier.write_json(target, {"passed": True})
    with pytest.raises(FileExistsError):
        qualifier.write_json(target, {"passed": False})
    with pytest.raises(ValueError):
        qualifier.write_json(tmp_path / "nan.json", {"n": float("nan")})
    monkeypatch.setattr(qualifier, "MAX_FILE_BYTES", 10)
    with pytest.raises(ValueError, match="ceiling"):
        qualifier.write_json(tmp_path / "large.json", {"long": "long"})


@pytest.mark.parametrize("kind", ["deadline", "file", "space", "symlink", "aggregate"])
def test_budget_fails_closed(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(qualifier.time, "time", lambda: 10.)
    monkeypatch.setattr(qualifier.shutil, "disk_usage", lambda p: SimpleNamespace(free=8 * 1024**3))
    deadline = 11.
    if kind == "deadline":
        deadline = 10.
    elif kind == "space":
        monkeypatch.setattr(qualifier.shutil, "disk_usage", lambda p: SimpleNamespace(free=1))
    elif kind == "symlink":
        (tmp_path / "link").symlink_to(tmp_path / "absent")
    elif kind == "file":
        monkeypatch.setattr(qualifier, "MAX_FILE_BYTES", 4)
        (tmp_path / "file").write_bytes(b"1234")
    else:
        monkeypatch.setattr(qualifier, "MAX_OUTPUT_BYTES", qualifier.MAX_FILE_BYTES + 1)
        (tmp_path / "file").write_bytes(b"12")
    with pytest.raises(ValueError):
        qualifier.Budget(tmp_path, deadline)()


def fixture_summary():
    history = [{"epoch": epoch, "global_step": epoch * 2, "train_count": 8, "train_batches": 2,
                "validation_count": 8, "validation_batches": 2, "train_batch_order_sha256": str(epoch) * 64}
               for epoch in (1, 2)]
    return {"status": "complete", "epochs_completed": 2, "global_step": 4,
            "initial_state_sha256": "a" * 64, "last_state_sha256": "b" * 64,
            "selectors": {"quantity": "first", "time": "first"}, "history": history}


def write_fixture(output, *, mismatch=None):
    for arm in qualifier.ARMS:
        summary = fixture_summary()
        for stage in ("full", "split"):
            value = deepcopy(summary)
            if mismatch == "history" and arm == qualifier.ARMS[2] and stage == "split":
                value["history"][0]["train_batch_order_sha256"] = "changed"
            qualifier.write_json(output / arm / stage / "summary.json", value)
        proof = {"runtime": {"device": "cuda:0"}, "execution_contract_sha256": "c" * 64,
                 "source": {"files_sha256": "d" * 64}, "model_state_sha256": "e" * 64,
                 "optimizer_state_sha256": "f" * 64, "rng_state_sha256": "0" * 64}
        for stage in ("full", "resume"):
            value = deepcopy(proof)
            if mismatch in {"rng_state_sha256", "runtime"} and arm == qualifier.ARMS[1] and stage == "resume":
                value[mismatch] = "changed"
            qualifier.write_json(output / arm / (stage + "_proof.json"), value)


def test_exact_fresh_process_replay_audit(tmp_path):
    write_fixture(tmp_path)
    result = qualifier._audit_replays(tmp_path)
    assert result["passed"] and result["synthetic_optimizer_updates"] == 24
    assert result["paired_exposure"] == {"passed": True, "arms": 3, "epochs": 2, "global_steps_per_arm": 4}


@pytest.mark.parametrize("mismatch", ["history", "rng_state_sha256", "runtime"])
def test_replay_tamper_rejected(tmp_path, mismatch):
    write_fixture(tmp_path, mismatch=mismatch)
    with pytest.raises(ValueError, match="replay"):
        qualifier._audit_replays(tmp_path)


def test_fresh_output_required_before_subprocess(tmp_path, monkeypatch):
    output = tmp_path / "qualification"
    output.mkdir()
    contract = {"hosts": {"5090": {"root": str(tmp_path)}}, "limits": {"qualification_seconds": 900}}
    monkeypatch.setattr(qualifier, "_load", lambda *a: (None, contract))
    monkeypatch.setattr(qualifier.subprocess, "run", lambda *a, **k: pytest.fail("Must not spawn"))
    with pytest.raises(ValueError, match="fresh"):
        qualifier.qualify(tmp_path / "contract.json", "5090", output)


def test_timeout_records_failure_and_never_retries(tmp_path, monkeypatch):
    output = tmp_path / "qualification"
    contract = {"hosts": {"5090": {"root": str(tmp_path)}}, "limits": {"qualification_seconds": 900}}
    common = SimpleNamespace(source_manifest=lambda: {"files": {}, "files_sha256": "a" * 64})
    monkeypatch.setattr(qualifier, "_load", lambda *a: (common, contract))
    monkeypatch.setattr(qualifier.Budget, "__call__", lambda *a, **k: None)
    monkeypatch.setattr(qualifier, "_assert_idle", lambda *a: None)
    calls = []
    def failure(command, **kwargs):
        calls.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])
    monkeypatch.setattr(qualifier.subprocess, "run", failure)
    with pytest.raises(subprocess.TimeoutExpired):
        qualifier.qualify(tmp_path / "contract.json", "5090", output)
    assert len(calls) == 1
    assert not (output / "receipt.json").exists()
    assert qualifier.read_json(output / "failure.json")["automatic_retry"] is False


def test_monotonic_deadline_survives_wall_clock_rollback(tmp_path, monkeypatch):
    monkeypatch.setattr(qualifier.time, "time", lambda: 1.)
    monkeypatch.setattr(qualifier.time, "monotonic", lambda: 101.)
    with pytest.raises(ValueError, match="monotonic"):
        qualifier.Budget(tmp_path, deadline=900., monotonic_deadline=100.)()


@pytest.mark.parametrize("processes,passed", [("", True), ("GPU-other, 12\n", True), ("GPU-owned, 42\n", False)])
def test_gpu_idle_check_reads_only_and_rejects_existing_job(monkeypatch, processes, passed):
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="GPU-owned\n" if len(calls) == 1 else processes)
    monkeypatch.setattr(qualifier.subprocess, "run", run)
    if passed:
        qualifier._assert_idle({"gpu_uuid": "GPU-owned"})
    else:
        with pytest.raises(ValueError, match="busy"):
            qualifier._assert_idle({"gpu_uuid": "GPU-owned"})
    assert len(calls) == 2 and all(command[0] == "nvidia-smi" for command in calls)


@pytest.fixture
def tiny_model():
    import torch
    from paper.scripts.run_raw_aux_gradient_comparison import synthetic_inputs, objective_for_arm, load_design
    from paper.scripts.quantity_objective_comparison import QuantityStatistics
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    model, train, _ = synthetic_inputs()
    with torch.no_grad():
        model.quantity_head.weight.copy_(torch.linspace(-.02, .03, model.quantity_head.weight.numel()).reshape_as(model.quantity_head.weight))
        train.dataset.tensors[2][:, -1] = 10000.
    yield model, train.dataset.tensors, QuantityStatistics(1., load_design()["losses"]["raw_scale"]), objective_for_arm
    torch.set_num_threads(threads)


def test_actual_small_titan_active_cap_postclip_and_causality(tiny_model):
    model, tensors, statistics, objectives = tiny_model
    objective, control = objectives(qualifier.ARMS[2])
    result = qualifier._gradient_audit(model, tensors, statistics, objective, control)
    assert result["passed"] and 0 < result["s"] < 1
    assert result["weighted_raw_norm"] > 0 and result["state_and_rng_preserved"]
    assert result["optimizer_steps"] == 0 and result["postclip_gradient_passed"]
    assert qualifier._causality(model, tensors, statistics, objective)["valid_history_sensitivity"]


def test_actual_small_titan_B_path_exactly_preserved(tiny_model):
    model, tensors, statistics, objectives = tiny_model
    objective, _ = objectives(qualifier.ARMS[0])
    result = qualifier._b_preservation(model, tensors, statistics, objective)
    assert result["objective_and_gradients_bitwise_equal"] and result["optimizer_steps"] == 0


def test_gradient_check_rejects_none_dtype_and_changed_values():
    import torch
    with pytest.raises(ValueError, match="Unused"):
        qualifier._same_gradients([None], [torch.zeros(1)])
    with pytest.raises(ValueError, match="dtype"):
        qualifier._same_gradients([torch.ones(1)], [torch.ones(1, dtype=torch.float64)])
    with pytest.raises(AssertionError):
        qualifier._same_gradients([torch.ones(1)], [torch.zeros(1)])
