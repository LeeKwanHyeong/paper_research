"""Synthetic CPU tests using only the repository execution contract metadata."""
from __future__ import annotations

import copy
import io
import json
import os
import signal
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from models.TPPs.CountAwareFactory import build_count_aware_model
from paper.scripts import jqt_time_analysis as timing
from paper.scripts import run_jqt_frozen_diagnostics as runner
from paper.scripts import time_quantity_diagnostic as engine
from simple_lab_test.search.common.runner import canonical_state_dict_sha256


@pytest.fixture
def contract():
    value = runner.read_json(runner.ROOT / "paper/contracts/jqt_frozen_diagnosis_v1.json")
    value["authorization"] = "pending"
    return value


def test_metadata_cli_cannot_import_numerical_runtime_or_open_research_files(contract, tmp_path):
    path = tmp_path / "contract.json"
    runner.write_new_json(path, contract)
    code = '''
import importlib.abc, pathlib, runpy, sys
class NoRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'torch', 'numpy', 'polars', 'paper', 'models', 'data_loader'}:
            raise AssertionError('Unexpected runtime import: ' + fullname)
sys.meta_path.insert(0, NoRuntime())
original = pathlib.Path.open
def checked_open(path, *args, **kwargs):
    assert path.suffix not in {'.pt', '.parquet'}, str(path)
    return original(path, *args, **kwargs)
pathlib.Path.open = checked_open
sys.argv = [sys.argv[1], '--contract', sys.argv[2]]
runpy.run_path(sys.argv[0], run_name='__main__')
'''
    result = subprocess.run([sys.executable, "-c", code, runner.__file__, str(path)],
                            text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {"status": "metadata_valid", "authorization": "pending",
                                        "model_loaded": False, "data_read": False, "gpu_initialized": False}
    assert list(tmp_path.iterdir()) == [path]


def test_pending_execution_and_unsupervised_worker_fail_before_source_access(contract, monkeypatch):
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **kw: pytest.fail("source accessed"))
    with pytest.raises(ValueError, match="authorization is pending"):
        runner.execute(contract)
    contract["authorization"] = "approved"
    monkeypatch.delenv("JQT_DIAGNOSTIC_CONTRACT_SHA256", raising=False)
    with pytest.raises(ValueError, match="supervised"):
        runner.execute(contract)


@pytest.mark.parametrize("path,value", [
    (("held_out",), True),
    (("max_wall_seconds",), 7201),
    (("source", "revision"), "x" * 40),
    (("source", "root"), "/other/source"),
    (("datasets", 0, "quantity_boundaries"), [2, 31, 47, 188]),
    (("datasets", 0, "states", 0, "path"), "/other/checkpoint.pt"),
    (("datasets", 0, "states", 0, "expected", "global_step"), 1),
    (("datasets", 0, "training_contract", "loader", "validation_shuffle"), True),
    (("evaluator_files", "jqt_time_analysis.py"), "a" * 64),
])
def test_scope_identity_and_code_drift_fail_closed(contract, path, value):
    target = contract
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        runner.validate_contract(contract)


def test_environment_is_fixed_before_child_interpreter_and_clears_nulls(contract, monkeypatch):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "unexpected")
    monkeypatch.setenv("PYTHONHASHSEED", "123")
    env = runner.worker_environment(contract)
    assert "CUDA_DEVICE_ORDER" not in env
    assert env["PYTHONDONTWRITEBYTECODE"] == "1"
    result = subprocess.run([sys.executable, "-c", "import os; print(os.environ['PYTHONHASHSEED'])"],
                            env=env, capture_output=True, text=True, timeout=15, check=True)
    assert result.stdout.strip() == "42"
    assert os.environ["PYTHONHASHSEED"] == "123"


@pytest.mark.parametrize("selector", ["raw_quantity_rmse", "last"])
def test_safe_synthetic_checkpoint_replay_and_corruption_rejection(selector, tmp_path):
    weights = {"weight": torch.tensor([[1.0, 2.0]])}
    sha = canonical_state_dict_sha256(weights)
    ac = {"objective": "joint", "identity": {"synthetic": True}}
    state = {"arm_contract": ac, "selector": selector, "epoch": 2, "state_sha256": sha,
             "expected": {"global_step": 4, "raw_quantity_rmse": 3.0}}
    payload = {"schema_version": "time_quantity_diagnostic_cuda_v2", "model_state_dict": weights,
               "contract_sha256": runner.json_digest(ac), "global_step": 4}
    if selector == "last":
        payload.update(contract=ac, epoch=2, model_state_sha256=sha, rng={"numpy": np.random.get_state()})
    else:
        payload.update(selector=selector, applicable=True, best_epoch=2, best_value=3.0, state_sha256=sha)
    path = tmp_path / "synthetic.pt"
    torch.save(payload, path)
    loaded = runner.load_checkpoint(path)
    verified = runner.verify_checkpoint_payload(loaded, state, canonical_state_dict_sha256)
    assert torch.equal(verified["weight"], weights["weight"])
    changed = copy.deepcopy(loaded)
    changed["global_step"] += 1
    with pytest.raises(ValueError, match="step mismatch"):
        runner.verify_checkpoint_payload(changed, state, canonical_state_dict_sha256)
    changed = copy.deepcopy(loaded)
    changed["model_state_dict"]["weight"][0, 0] += 1
    with pytest.raises(ValueError, match="tensor state mismatch"):
        runner.verify_checkpoint_payload(changed, state, canonical_state_dict_sha256)
    if selector != "last":
        loaded["applicable"] = False
        with pytest.raises(ValueError, match="Selector mismatch"):
            runner.verify_checkpoint_payload(loaded, state, canonical_state_dict_sha256)


def synthetic_model_and_loader():
    torch.manual_seed(42)
    model, _ = build_count_aware_model("titantpp", hidden_dim=8, train_log_mean=1.0,
                                      max_seq_len=8, time_intercept_limit=300.0)
    dts = torch.tensor([[0., 1., 2.], [1., 1., 3.], [0., 1., 4.]])
    mask = torch.tensor([[False, True, True], [True, True, True], [False, True, True]])
    quantities = torch.tensor([[0., 2., 4.], [1., 2., 3.], [0., 7., 8.]])
    return model.eval(), DataLoader(TensorDataset(dts, mask, quantities), batch_size=2, shuffle=False)


@pytest.mark.parametrize("purpose,objective", [
    ("quantity", "joint"), ("quantity", "quantity_only"), ("time", "joint"), ("time", "time_only"),
])
def test_collect_original_outputs_preserves_state_order_and_inactive_head(purpose, objective, monkeypatch):
    model, loader = synthetic_model_and_loader()
    engine.configure_parameters(model, objective)
    before = canonical_state_dict_sha256(model.state_dict())
    with torch.no_grad():
        expected = [engine.task_outputs(model, *engine._batch_tensors(b, "cpu"), objective) for b in loader]
    if objective == "quantity_only":
        monkeypatch.setattr(model, "log_f_dt", lambda *a: pytest.fail("Inactive time head invoked"))
    if objective == "time_only":
        monkeypatch.setattr(model, "quantity_outputs", lambda *a: pytest.fail("Inactive quantity head invoked"))
    checks = []
    rows = runner.collect_rows(model, loader, engine, timing, purpose, objective, "cpu", lambda: checks.append(1))
    assert len(checks) == 2 and rows["history_length"].tolist() == [1, 2, 1]
    assert canonical_state_dict_sha256(model.state_dict()) == before
    assert all(p.grad is None for p in model.parameters())
    if purpose == "quantity":
        assert rows["true_qty"].tolist() == [4., 3., 8.]
        assert np.array_equal(rows["pred_qty"], torch.cat([r["pred_qty"] for r in expected]).numpy())
        assert np.array_equal(rows["log_qty_loss"], torch.cat([r["log_qty_loss"] for r in expected]).numpy())
        assert runner.quantity_metrics(rows)["quantity_train_loss"] == pytest.approx(rows["log_qty_loss"].astype("float64").mean())
    else:
        original = torch.cat([r["time_loss"] for r in expected]).numpy()
        assert np.array_equal(rows["original_legacy_loss"], original)
        target = {"legacy_time_loss": float(original.astype("float64").mean())}
        assert runner.verify_time_replay(rows, target)["passed"]
        assert timing.summarize_time_rows(rows)["n"] == 3
        with pytest.raises(ValueError, match="Time metric replay failed"):
            runner.verify_time_replay(rows, {"legacy_time_loss": target["legacy_time_loss"] + 1})


def test_collection_rejects_dropped_targets_and_resource_stop():
    model, loader = synthetic_model_and_loader()
    dropped = DataLoader(loader.dataset, batch_size=2, drop_last=True)
    with pytest.raises(ValueError, match="target count drift"):
        runner.collect_rows(model, dropped, engine, timing, "quantity", "joint", "cpu", lambda: None)
    def stop():
        raise ValueError("budget exhausted")
    with pytest.raises(ValueError, match="budget exhausted"):
        runner.collect_rows(model, loader, engine, timing, "quantity", "joint", "cpu", stop)


def test_failure_receipt_cannot_modify_an_existing_unowned_run(contract, tmp_path):
    contract["output_dir"] = str(tmp_path)
    runner.write_new_json(tmp_path / "started.json", {"pid": 987, "contract_sha256": runner.json_digest(contract)})
    runner.record_failure(contract, 123, "error")
    assert not (tmp_path / "failed.json").exists()
    runner.record_failure(contract, 987, "first failure")
    runner.record_failure(contract, 987, "second failure")
    assert runner.read_json(tmp_path / "failed.json")["error"] == "first failure"
    with pytest.raises(FileExistsError):
        runner.write_new_json(tmp_path / "failed.json", {})


def test_output_budget_supports_seekable_npz_and_refuses_excess_before_write(tmp_path):
    path = tmp_path / "synthetic.npz"
    runner.bounded_artifact(path, 2**20, lambda handle: np.savez_compressed(handle, values=np.arange(10)))
    with np.load(path, allow_pickle=False) as saved:
        assert saved["values"].tolist() == list(range(10))
    handle = io.BytesIO()
    limited = runner.BudgetedFile(handle, 5)
    limited.write(b"1234")
    limited.seek(0)
    limited.write(b"ab")
    limited.seek(4)
    with pytest.raises(ValueError, match="Output size limit"):
        limited.write(b"67")
    assert handle.getvalue() == b"ab34"
    terminal = tmp_path / "complete.json"
    used = path.stat().st_size
    with pytest.raises(ValueError, match="Output size limit"):
        runner.write_budgeted_json(terminal, {"status": "complete"}, used + 1, terminal=True)
    assert not terminal.exists()


def test_incomplete_terminal_json_does_not_suppress_failure_receipt(contract, tmp_path):
    contract["output_dir"] = str(tmp_path)
    runner.write_new_json(tmp_path / "started.json", {"pid": 123, "contract_sha256": runner.json_digest(contract)})
    (tmp_path / "complete.json").write_text('{"status":')
    runner.record_failure(contract, 123, "incomplete terminal write")
    assert runner.read_json(tmp_path / "failed.json")["error"] == "incomplete terminal write"
    assert (tmp_path / "complete.json").read_text() == '{"status":'


@pytest.mark.parametrize("mode", ["timeout", "sigterm", "nonzero"])
def test_supervisor_reaps_only_owned_dummy_child_and_records_failure(contract, tmp_path, monkeypatch, mode):
    """Spawn only a sleeping CPU child, never the evaluator or any GPU process."""
    contract["output_dir"] = str(tmp_path)
    contract["max_wall_seconds"] = 1
    popen = subprocess.Popen
    children = []
    old_handler = signal.getsignal(signal.SIGTERM)

    def dummy_child(*args, **kwargs):
        child = popen([sys.executable, "-c", "import time; time.sleep(30)"], **kwargs)
        children.append(child)
        runner.write_new_json(tmp_path / "started.json", {"pid": child.pid, "contract_sha256": runner.json_digest(contract)})
        if mode == "nonzero":
            child.kill()
        elif mode == "sigterm":
            wait = child.wait
            first = True
            def interrupted_wait(*a, **kw):
                nonlocal first
                if first:
                    first = False
                    os.kill(os.getpid(), signal.SIGTERM)
                return wait(*a, **kw)
            child.wait = interrupted_wait
        return child

    monkeypatch.setattr(runner.subprocess, "Popen", dummy_child)
    expected_error = {"timeout": subprocess.TimeoutExpired, "sigterm": runner.SupervisorTerminated,
                      "nonzero": ValueError}[mode]
    try:
        with pytest.raises(expected_error):
            runner.supervise(contract, tmp_path / "unused_contract.json")
        assert len(children) == 1 and children[0].poll() is not None
        assert runner.read_json(tmp_path / "failed.json")["automatic_retry"] is False
        assert signal.getsignal(signal.SIGTERM) == old_handler
    finally:
        for child in children:
            if child.poll() is None:
                child.kill()
                child.wait()
