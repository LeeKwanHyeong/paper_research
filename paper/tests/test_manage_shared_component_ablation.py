"""Negative contracts protect the frozen A-D matrix and historical processes."""
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from paper.scripts import manage_shared_component_ablation as m


@pytest.fixture
def contract():
    c = {"schema": m.SCHEMA, "campaign": m.NAME, "hosts": {"5080": {}, "5090": {}},
        "common_condition": {"time_input": "log1p", "loss_mode": "log_mse", "time_alpha": 1.0},
        "training": {"maximum_epochs": 300, "minimum_epochs": 40, "patience": 40,
            "batch_size": 128, "monitor": "validation_raw_quantity_rmse"},
        "source": {"files": {"safe.py": "hash"}}, "input_files": {f"inputs/{i}": "hash" for i in range(9)}}
    c["source"]["files_sha256"] = m.canonical(c["source"]["files"])
    c["jobs"] = []
    for d in m.DATASETS:
        for seed in (42, 52, 62):
            for arm in m.ARMS:
                j = {"id": f"{d}__{arm}__seed{seed}", "dataset": d, "seed": seed,
                    "arm": arm, "host": "5080" if seed == 42 else "5090"}
                if arm == "full" and seed != 42: j["reuse"] = {"verified": True}
                c["jobs"].append(j)
    return c


def test_exact_36_matrix_and_six_reused_A(contract):
    m.validate(contract)
    assert sum(bool(j.get("reuse")) for j in contract["jobs"]) == 6


@pytest.mark.parametrize("change", ["cross_host", "seed", "duplicate", "reuse_wrong_arm", "reuse_new_gpu", "batch", "empty_source", "missing_input", "path_escape"])
def test_protocol_changes_are_rejected(contract, change):
    if change == "cross_host": contract["jobs"][0]["host"] = "5090"
    if change == "seed": contract["jobs"][0]["seed"] = 72
    if change == "duplicate": contract["jobs"][-1] = deepcopy(contract["jobs"][0])
    if change == "reuse_wrong_arm": contract["jobs"][1]["reuse"] = {"verified": True}
    if change == "reuse_new_gpu": contract["jobs"][0]["reuse"] = {"verified": True}
    if change == "batch": contract["training"]["batch_size"] = 64
    if change == "empty_source":
        contract["source"]["files"] = {}; contract["source"]["files_sha256"] = m.canonical({})
    if change == "missing_input": contract["input_files"].pop("inputs/0")
    if change == "path_escape": contract["legacy_files"] = {"../original.pt": "hash"}
    with pytest.raises(RuntimeError): m.validate(contract)


def test_original_receipts_cannot_be_overwritten(tmp_path):
    p = tmp_path / "receipt.json"; m.write(p, {"original": True})
    with pytest.raises(FileExistsError): m.write(p, {"original": False})
    assert m.read(p)["original"] is True


def test_symlink_cannot_escape_evidence_root(tmp_path):
    source = tmp_path / "root"; source.mkdir(); other = tmp_path / "outside"; other.mkdir()
    (source / "link").symlink_to(other, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"): m.below(source, "link/checkpoint.pt")


def make_gate(c, root):
    jobs = {}
    for job in c["jobs"]:
        evidence = {"host": job["host"]}
        for stage in ("train", "validation"):
            for kind in ("receipt", "exit"):
                path = f"global_validation_sources/{job['id']}.{stage}.{kind}.json"
                observed = "reuse" if stage == "train" and kind == "exit" and job.get("reuse") else stage
                value = {"job_id": job["id"], "host": job["host"], "stage": observed,
                    "contract_sha256": m.canonical(c)}
                if kind == "exit": value.update(actual_os_returncode=0, timeout=False, supervisor_error=None, resource_incident=None)
                else: value.update(status="complete", validation_replay_passed=True, individual_predictions_match=True)
                m.write(root / path, value); evidence[f"{stage}_{kind}"] = {"path": path, "sha256": m.sha(root / path)}
        jobs[job["id"]] = evidence
    gate = {"status": "passed", "contract_sha256": m.canonical(c), "jobs": jobs}
    m.write(root / "global_validation_gate.json", gate)
    return gate


def test_full_raw_validation_gate_accepts_reuse_admission_without_fabricating_new_training(contract, tmp_path):
    make_gate(contract, tmp_path)
    assert len(m.verify_global_gate(contract, tmp_path)["jobs"]) == 36


@pytest.mark.parametrize("fault", ["partial", "raw_tamper", "nonzero", "incident", "summary_only", "reused_fit_mislabelled"])
def test_test_is_blocked_by_incomplete_or_changed_validation(contract, tmp_path, fault):
    gate = make_gate(contract, tmp_path)
    if fault == "partial":
        gate["jobs"].pop(next(iter(gate["jobs"])))
    else:
        entry = next(iter(gate["jobs"].values()))
        if fault == "reused_fit_mislabelled":
            entry = next(gate["jobs"][j["id"]] for j in contract["jobs"] if j.get("reuse"))
        item = entry["validation_receipt" if fault in ("raw_tamper", "summary_only") else "train_exit"]
        value = m.read(tmp_path / item["path"])
        if fault == "nonzero": value["actual_os_returncode"] = 1
        if fault == "incident": value["resource_incident"] = "foreign GPU owner"
        if fault == "summary_only": value.pop("individual_predictions_match")
        if fault == "raw_tamper": value["changed"] = True
        if fault == "reused_fit_mislabelled": value["stage"] = "train"
        path = tmp_path / item["path"]; path.write_text(__import__("json").dumps(value))
        if fault != "raw_tamper": item["sha256"] = m.sha(path)
    (tmp_path / "global_validation_gate.json").write_text(__import__("json").dumps(gate))
    with pytest.raises(RuntimeError): m.verify_global_gate(contract, tmp_path)


def test_pending_implementation_blocks_remote_work(contract, tmp_path):
    contract["hosts"]["5080"]["source_root"] = str(tmp_path)
    contract["implementation_qualification"] = {"status": "pending", "training_allowed": False}
    with patch.object(m, "verify_files"), patch.object(m, "stage_child") as child:
        with pytest.raises(RuntimeError, match="implementation gate"):
            m.dispatch(contract, "5080", "fit_validation", tmp_path / "contract.json")
        child.assert_not_called()


def test_native_gate_cannot_be_summary_only(contract, tmp_path):
    m.write(tmp_path / "qualification.complete.json", {"status": "complete", "host": "5080",
        "contract_sha256": m.canonical(contract), "actual_child_exits": []})
    with pytest.raises(RuntimeError, match="actual successful exits"):
        m.verify_native_qualification(contract, "5080", tmp_path)


def native_gate(c, root):
    jobs = [j for j in c["jobs"] if j["host"] == "5080"]
    entries = []
    for stage, batch in (("synthetic", jobs), ("capacity", jobs[:4])):
        for job in batch:
            value = {"host": "5080", "job_id": job["id"], "stage": stage,
                "contract_sha256": m.canonical(c), "actual_os_returncode": 0, "timeout": False,
                "supervisor_error": None, "resource_incident": None}
            m.write(root / "control" / (job["id"] + "." + stage + ".exit.json"), value)
            entries.append(value)
    capacities = {}
    for arm in m.ARMS:
        path = f"cost/qualification/{arm}.json"
        value = {"status": "qualified", "host": "5080", "arm": arm, "contract_sha256": m.canonical(c),
            "source_files_sha256": c["source"]["files_sha256"], "synthetic": True, "real_data_loaded": False,
            "cost_complete": False, "checkpoint_saved": False, "rows": [{"mode": mode, "batch_size": 128,
                "padded_length": 256, "optimizer_steps": int(mode == "train"), "elapsed_seconds": .1,
                "memory_baseline_bytes": {"allocated": 1, "reserved": 1},
                "memory_peak_bytes": {"allocated": 2, "reserved": 2}} for mode in ("train", "eval")]}
        m.write(root / path, value); capacities[arm] = {"path": path, "sha256": m.sha(root / path)}
    complete = {"status": "complete", "host": "5080", "contract_sha256": m.canonical(c),
        "actual_child_exits": entries, "capacity_receipts": capacities}
    m.write(root / "qualification.complete.json", complete)
    return complete


def test_complete_native_gate_is_bound_to_actual_exit_originals(contract, tmp_path):
    native_gate(contract, tmp_path)
    assert len(m.verify_native_qualification(contract, "5080", tmp_path)["capacity_receipts"]) == 4


@pytest.mark.parametrize("fault", ["empty_rows", "wrong_batch", "missing_eval", "bad_memory", "zero_time", "wrong_optimizer", "wrong_actual_exit"])
def test_native_summary_cannot_hide_absent_or_invalid_capacity(contract, tmp_path, fault):
    value = native_gate(contract, tmp_path)
    item = value["capacity_receipts"]["full"]; path = tmp_path / item["path"]; receipt = m.read(path)
    if fault == "empty_rows": receipt["rows"] = []
    if fault == "wrong_batch": receipt["rows"][0]["batch_size"] = 64
    if fault == "missing_eval": receipt["rows"][1]["mode"] = "train"
    if fault == "bad_memory": receipt["rows"][0]["memory_peak_bytes"]["reserved"] = -1
    if fault == "zero_time": receipt["rows"][0]["elapsed_seconds"] = 0
    if fault == "wrong_optimizer": receipt["rows"][0]["optimizer_steps"] = 0
    if fault == "wrong_actual_exit":
        entry = value["actual_child_exits"][0]
        raw = tmp_path / "control" / (entry["job_id"] + ".synthetic.exit.json")
        raw.write_text('{}')
    path.write_text(__import__("json").dumps(receipt)); item["sha256"] = m.sha(path)
    (tmp_path / "qualification.complete.json").write_text(__import__("json").dumps(value))
    with pytest.raises(RuntimeError): m.verify_native_qualification(contract, "5080", tmp_path)


def test_termination_refuses_reused_PID():
    process = Mock(pid=123); process.poll.return_value = None
    old = {"pid": 123, "start_ticks": 1, "pgid": 123, "cwd": "/ours", "argv": ["our_worker"]}
    live = {**old, "start_ticks": 2}
    with patch.object(m, "proc_identity", return_value=live), patch.object(m.os, "killpg") as kill:
        with pytest.raises(RuntimeError, match="unverified process"): m.terminate_owned(process, old["argv"], old)
        kill.assert_not_called()


def test_missing_external_contract_SHA_blocks_dispatch(tmp_path):
    p = tmp_path / "contract.json"; m.write(p, {})
    with pytest.raises(RuntimeError, match="pinned contract SHA"):
        m.main(["--mode", "dispatch", "--contract", str(p), "--host", "5080", "--phase", "test_cost"])
