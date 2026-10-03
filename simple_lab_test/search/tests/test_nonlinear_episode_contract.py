"""Local contract, portable receipts and isolated-bundle checks; no GPU/data."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from paper.scripts import nonlinear_episode_parallel_common as common
from paper.scripts import prepare_nonlinear_episode_execution as prepare


@pytest.fixture
def contract():
    return prepare.build_contract()


def approved_fixture(contract):
    result = prepare.approval_template(contract)
    result.update(approved=True, user_instruction="synthetic test fixture only")
    return result


def receipt_fixture(contract, alias, start):
    host = contract["hosts"][alias]
    runtime = {**copy.deepcopy(host["runtime_expected"]),
               "gpu": {"uuid": host["gpu_uuid"], "total_memory_bytes": 16 * 1024**3}}
    receipt = {"status": "passed", "host": alias, "device": "cuda:0",
        "real_data_loaded": False, "held_out_evaluated": False,
        "contract_sha256": common.sha_json(contract),
        "source_files_sha256": contract["source"]["files_sha256"],
        "runtime": runtime, "started_at_unix": start, "completed_at_unix": start + 20,
        "synthetic_optimizer_updates": 48,
        "checks": {name: {key: True for key in (
            "initial_output_objective_shared_preclip_gradient_exact", "activated_target_prediction_causal",
            "activated_Q_K_V_b_U_gradients", "saved_optimizer_rng_next_step_exact")}
            for name in common.ARMS[1:]},
        "costs": [{"batch": 128, "length": length, "updates_per_arm": 7,
            "measurements": {name: {
                "parameters": 89795 if index == 0 else 92915,
                "step_seconds": [0.1 if index == 0 else 0.12] * 5,
                "median_step_seconds": 0.1 if index == 0 else 0.12,
                "peak_allocated_bytes": (100 if index == 0 else 110) * 1024**2,
            } for index, name in enumerate(common.ARMS)}} for length in (64, 256)]}
    receipt["cost_gate_checks"] = common.verified_cost_gates(receipt, contract)
    return receipt


def bind_receipt(contract, alias, receipt):
    path = Path(contract["hosts"][alias]["root"]) / "qualification/receipt.json"
    common.write_json(path, receipt)
    return {"path": str(path), "sha256": common.sha_file(path),
            "receipt_sha256": common.sha_json(receipt), "receipt": receipt}


def qualified_fixture(contract, tmp_path):
    for alias in common.ASSIGNMENTS:
        contract["hosts"][alias]["root"] = str(tmp_path / alias)
    bindings = {alias: bind_receipt(contract, alias, receipt_fixture(contract, alias, started))
                for alias, started in (("5080", 1000.), ("5090", 1200.))}
    return approved_fixture(contract), bindings


def test_contract_is_reviewable_and_template_cannot_authorize_execution(contract, tmp_path):
    path = tmp_path / "contract.json"
    common.write_json(path, contract)
    for host in common.ASSIGNMENTS:
        assert common.read_contract(path, host) == contract
    assert contract["total_optimizer_steps"] == 6816240
    assert contract["endpoint_replays"] == 18
    assert sum(data["expected_global_steps"] for data in contract["datasets"]) * 3 == 6816240
    assert contract["cost"]["expected_training_hours"] is None
    with pytest.raises(ValueError, match="approval"):
        common.verify_approval(contract, prepare.approval_template(contract))


@pytest.mark.parametrize("change", [
    lambda c: c["hosts"]["5080"]["assigned_datasets"].append("insta_market_basket"),
    lambda c: c["hosts"]["5090"]["assigned_datasets"].append("insta_market_basket"),
    lambda c: c["hosts"]["5080"]["assigned_datasets"].reverse(),
    lambda c: c["hosts"].pop("5090"),
    lambda c: c["arms"].reverse(),
    lambda c: c.update(seed=43),
    lambda c: c.update(epochs=121),
    lambda c: c.update(total_optimizer_steps=6816241),
    lambda c: c.update(endpoint_replays=19),
    lambda c: c["limits"].update(total_seconds=172801),
    lambda c: c["limits"].update(max_concurrent_gpu_jobs_per_host=2),
    lambda c: c["policy"].update(resume=True),
    lambda c: c["policy"].update(held_out=True),
    lambda c: c["acceptance"].update(raw_rmse_ratio_strictly_less_than=1.1),
    lambda c: c["datasets"][0]["loader"].update(batch_size=64),
    lambda c: c["datasets"][1]["model"].update(time_intercept_limit=30),
    lambda c: c["hosts"]["5090"]["runtime_expected"].update(torch="foreign"),
    lambda c: c["hosts"]["5080"].update(tmux_binary="/tmp/other"),
    lambda c: c["hosts"]["5090"].update(root="/tmp/other"),
    lambda c: c.update(design_sha256="0" * 64),
    lambda c: c.update(cpu_receipt_sha256="0" * 64),
    lambda c: c.update(approval_required=False),
])
def test_contract_drift_rejected(contract, tmp_path, change):
    change(contract)
    path = tmp_path / "changed.json"
    common.write_json(path, contract)
    with pytest.raises(ValueError):
        common.read_contract(path, "5090", check_source=False)


@pytest.mark.parametrize("key,value", [
    ("schema", "successor_episode_parallel_approval_v1"), ("approved", False),
    ("hosts", ["5090"]), ("contract_sha256", "0" * 64),
    ("scope", "old_experiment"), ("user_instruction", " "),
])
def test_old_partial_or_unbound_approval_rejected(contract, key, value):
    approval = approved_fixture(contract)
    approval[key] = value
    with pytest.raises(ValueError):
        common.verify_approval(contract, approval)


def test_common_deadline_uses_first_qualification_and_cannot_be_reset(contract, tmp_path, monkeypatch):
    approval, bindings = qualified_fixture(contract, tmp_path)
    monkeypatch.setattr(common.time, "time", lambda: 1500.)
    permit = common.create_permit(contract, approval, bindings)
    assert permit["started_at_unix"] == 1000.
    assert permit["deadline_unix"] == 173800.
    for alias in common.ASSIGNMENTS:
        common.verify_permit(contract, permit, alias)
    monkeypatch.setattr(common.time, "time", lambda: 2200.)
    assert common.create_permit(contract, approval, bindings)["deadline_unix"] == permit["deadline_unix"]
    # Each host checks its own file and portable copy of the other host's receipt.
    Path(bindings["5080"]["path"]).unlink()
    common.verify_permit(contract, permit, "5090")
    permit["deadline_unix"] += 1
    with pytest.raises(ValueError, match="deadline"):
        common.verify_permit(contract, permit, "5090")


@pytest.mark.parametrize("change", [
    lambda r: r.update(status="cost_gate_failed"),
    lambda r: r.update(device="cpu"),
    lambda r: r.update(real_data_loaded=True),
    lambda r: r.update(synthetic_optimizer_updates=49),
    lambda r: r.update(completed_at_unix=1999.),
    lambda r: r.update(source_files_sha256="0" * 64),
    lambda r: r["runtime"]["gpu"].update(uuid="foreign"),
    lambda r: r["checks"][common.ARMS[1]].update(activated_Q_K_V_b_U_gradients=False),
    lambda r: r["costs"][0].update(batch=2),
    lambda r: r["costs"][0]["measurements"][common.ARMS[2]].update(parameters=91971),
    lambda r: r["costs"][0]["measurements"][common.ARMS[2]].update(median_step_seconds=0.9),
    lambda r: r["costs"][0]["measurements"][common.ARMS[2]].update(median_step_seconds=0.9, step_seconds=[0.9] * 5),
    lambda r: r["costs"][1]["measurements"][common.ARMS[1]].update(peak_allocated_bytes=15 * 1024**3),
])
def test_claimed_pass_does_not_hide_bad_native_evidence(contract, tmp_path, monkeypatch, change):
    approval, bindings = qualified_fixture(contract, tmp_path)
    monkeypatch.setattr(common.time, "time", lambda: 1500.)
    receipt = bindings["5080"]["receipt"]
    change(receipt)
    # Rehash changed bytes: rejection must check substance, not only stale hashes.
    bindings["5080"] = bind_receipt(contract, "5080", receipt)
    with pytest.raises(ValueError):
        common.create_permit(contract, approval, bindings)


def test_source_manifest_binds_new_stack_and_detects_file_drift(contract, monkeypatch):
    files = contract["source"]["files"]
    for name in (*common.ENTRYPOINTS, common.DESIGN, common.CPU_RECEIPT, common.IMPLEMENTATION_RECEIPT,
                 "models/TPPs/CountAwareTitanNonlinearEpisodeMemory.py",
                 "paper/scripts/run_nonlinear_episode_comparison.py",
                 "paper/scripts/verify_nonlinear_episode_cost.py",
                 "paper/scripts/count_aware_tpp_backbone/training.py",
                 "paper/scripts/run_observed_slot_partition.py"):
        assert name in files
    changed = dict(files)
    changed["models/TPPs/CountAwareTitanNonlinearEpisodeMemory.py"] = "0" * 64
    monkeypatch.setattr(common, "source_manifest", lambda: changed)
    with pytest.raises(ValueError, match="source changed"):
        common.verify_source(contract)


def test_bundle_runs_from_frozen_source_outside_checkout_without_data_or_cuda(contract, tmp_path):
    bundle = tmp_path / "source_bundle.tar.gz"
    prepare.write_bundle(bundle, contract)
    deployment = tmp_path / "deployment"
    with tarfile.open(bundle, "r:gz") as archive:
        archive.extractall(deployment, filter="data")
    frozen = deployment / "source"
    assert list((frozen / "sample_data").iterdir()) == []
    outside = tmp_path / "outside"
    outside.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME", "PROJECT_ROOT")}
    env.update(CUDA_VISIBLE_DEVICES="", MPLBACKEND="Agg")
    script = '''
import importlib,json,sys
from pathlib import Path
frozen, checkout, contract_path = map(Path, sys.argv[1:])
assert sys.flags.isolated and not Path.cwd().is_relative_to(checkout)
sys.path.insert(0,str(frozen))
for name in ('paper.scripts.run_nonlinear_episode_parallel',
             'paper.scripts.qualify_nonlinear_episode_parallel',
             'paper.scripts.count_aware_tpp_backbone.training',
             'paper.scripts.run_time_quantity_diagnostic'):
    importlib.import_module(name)
from paper.scripts import nonlinear_episode_parallel_common as common
for alias in ('5080','5090'): common.read_contract(contract_path, alias)
for name,module in tuple(sys.modules.items()):
    if name.split('.')[0] in {'models','paper','data_loader','simple_lab_test','utils'}:
        path=getattr(module,'__file__',None)
        if path: assert Path(path).resolve().is_relative_to(frozen),(name,path)
        for path in getattr(module,'__path__',()): assert Path(path).resolve().is_relative_to(frozen)
assert all(not Path(path).resolve().is_relative_to(checkout) for path in sys.path)
import torch
assert not torch.cuda.is_initialized()
assert list((frozen/'sample_data').iterdir()) == []
print('isolated_frozen_imports_and_source_passed')
'''
    result = subprocess.run([sys.executable, "-I", "-B", "-c", script, str(frozen), str(common.ROOT),
        str(deployment / "frozen_execution/execution_contract.json")], cwd=outside, env=env,
        capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == "isolated_frozen_imports_and_source_passed"
