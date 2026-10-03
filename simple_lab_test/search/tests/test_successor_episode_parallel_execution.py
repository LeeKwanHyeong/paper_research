"""CPU-only boundary checks for the fresh two-host successor comparison."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import sys

import pytest

from paper.scripts import successor_episode_common as legacy
from paper.scripts import successor_episode_parallel_common as common
from paper.scripts import run_successor_episode_comparison as comparison
from paper.scripts import run_successor_episode_parallel as runner
from paper.scripts import qualify_successor_episode_parallel as qualifier
from paper.scripts import run_observed_slot_partition as shared


LEGACY_CONTRACT = "paper/contracts/successor_episode_execution_5090_v1.json"


def contract_fixture():
    # Configuration provenance only; none of the referenced datasets are read.
    value = json.loads((common.ROOT / LEGACY_CONTRACT).read_text())
    reference = json.loads((common.ROOT / common.RUNTIME_REFERENCE).read_text())["hosts"]
    root = "/home/leekwanhyeong/workspace/paper_research_experiment_artifacts/successor_episode_seed42_parallel_20260914_v1"
    hosts = {}
    for alias, datasets in common.ASSIGNMENTS.items():
        hosts[alias] = copy.deepcopy(reference[alias])
        hosts[alias].update(
            root=root, source_root=root + "/source", output_dir=root + "/run",
            assigned_datasets=list(datasets), tmux=f"episode_seed42_parallel_20260914_{alias}",
        )
    value.update(schema=common.SCHEMA, hosts=hosts, limits=dict(common.LIMITS))
    return value


def approval_fixture(contract):
    return {
        "schema": "successor_episode_parallel_approval_v1", "approved": True,
        "hosts": ["5080", "5090"], "contract_sha256": common.sha_json(contract),
        "scope": "dual_native_synthetic_cuda_qualification_and_fresh_nine_arm_validation",
        "user_instruction": "5080/5090 병렬로 진행하자.",
    }


def synthetic_receipt(contract, host, started):
    runtime = copy.deepcopy(contract["hosts"][host]["runtime_expected"])
    runtime["gpu"] = {"uuid": contract["hosts"][host]["gpu_uuid"], "total_memory_bytes": 8 * 1024**3}
    return {
        "status": "passed", "host": host, "device": "cuda:0",
        "real_data_loaded": False, "held_out_evaluated": False,
        "contract_sha256": common.sha_json(contract),
        "source_files_sha256": contract["source"]["files_sha256"],
        "runtime": runtime, "started_at_unix": started,
        "checks": {name: {
            "initial_output_objective_shared_preclip_gradient_exact": True,
            "activated_target_prediction_causal": True,
            "activated_Q_K_U_gradients": True,
            "saved_optimizer_rng_next_step_exact": True,
        } for name in common.ARMS[1:]},
        "cost_gate_checks": [
            {"backbone": name, "length": length, "step": True, "relative_memory": True,
             "device_memory": True, "parameters": True}
            for name in common.ARMS[1:] for length in (64, 256)
        ],
        "costs": [
            {"batch": 128, "length": length, "measurements": {
                name: {"median_step_seconds": 0.1 if index == 0 else 0.12,
                       "peak_allocated_bytes": (100 if index == 0 else 110) * 1024**2,
                       "parameters": 89795 if index == 0 else 91971}
                for index, name in enumerate(common.ARMS)
            }} for length in (64, 256)
        ],
    }


def qualified_fixture(tmp_path):
    contract = contract_fixture()
    # Each synthetic local folder represents a separate host filesystem. No
    # remote path is accessed by these receipt and deadline checks.
    for host in common.ASSIGNMENTS:
        contract["hosts"][host]["root"] = str(tmp_path / host)
    bindings = {}
    for host, started in (("5090", 1200.0), ("5080", 1000.0)):
        receipt = synthetic_receipt(contract, host, started)
        path = Path(contract["hosts"][host]["root"]) / "qualification/receipt.json"
        common.write_json(path, receipt)
        bindings[host] = {"path": str(path), "sha256": common.sha_file(path),
                          "receipt_sha256": common.sha_json(receipt), "receipt": receipt}
    return contract, approval_fixture(contract), bindings


def test_contract_partitions_all_nine_arms_and_exact_optimizer_replay_budgets(tmp_path):
    contract = contract_fixture()
    path = tmp_path / "contract.json"
    common.write_json(path, contract)
    for host in ("5080", "5090"):
        assert common.read_contract(path, host, check_source=False) == contract
    assert common.validate_assignment(contract, "5090")["assigned_datasets"] == ["insta_market_basket"]
    assert common.validate_assignment(contract, "5080")["assigned_datasets"] == [
        "intermittent_frozen_5000", "yellow_trip_hourly",
    ]
    totals = {host: comparison.partition_totals(contract, host) for host in common.ASSIGNMENTS}
    assert totals["5090"] == {"arms": 3, "steps": 5600520, "replays": 6}
    assert totals["5080"] == {"arms": 6, "steps": 1215720, "replays": 12}
    assert sum(row["steps"] for row in totals.values()) == contract["total_optimizer_steps"] == 6816240
    assert sum(row["replays"] for row in totals.values()) == contract["endpoint_replays"] == 18
    for data in contract["datasets"]:
        for backbone in common.ARMS:
            args = comparison.training_args(contract, data, tmp_path / "unused", backbone,
                                            execution_role=common.EXECUTION_ROLE)
            assert args.epochs == args.min_epochs == 120
            assert args.batch_size == 128
            assert args.device == "cuda:0"
            assert args.execution_role == "fresh_successor_episode_parallel_validation"


@pytest.mark.parametrize("mutate", [
    lambda c: c["hosts"]["5090"]["assigned_datasets"].append("insta_market_basket"),
    lambda c: c["hosts"]["5080"]["assigned_datasets"].remove("yellow_trip_hourly"),
    lambda c: c["hosts"]["5080"]["assigned_datasets"].reverse(),
    lambda c: c["hosts"]["5090"]["assigned_datasets"].append("yellow_trip_hourly"),
    lambda c: c["hosts"].pop("5080"),
    lambda c: c["arms"].reverse(),
    lambda c: c["limits"].update(total_seconds=172801),
    lambda c: c["datasets"][0]["loader"].update(batch_size=64),
    lambda c: c["policy"].update(resume=True),
])
def test_contract_rejects_duplicate_missing_moved_work_and_budget_drift(tmp_path, mutate):
    contract = contract_fixture()
    mutate(contract)
    path = tmp_path / "changed.json"
    common.write_json(path, contract)
    with pytest.raises(ValueError):
        common.read_contract(path, "5090", check_source=False)


def test_parallel_source_manifest_binds_new_entrypoints_and_shared_execution(monkeypatch):
    contract = contract_fixture()
    files = common.source_manifest()
    for name in (*common.ENTRYPOINTS, "models/TPPs/CountAwareTitanSuccessorMemory.py",
                 "paper/scripts/run_successor_episode_comparison.py", "paper/scripts/run_observed_slot_partition.py"):
        assert name in files
    contract["source"].update(files=files, files_sha256=common.sha_json(files))
    common.verify_source(contract)
    changed = dict(files)
    changed["paper/scripts/run_successor_episode_parallel.py"] = "0" * 64
    monkeypatch.setattr(common, "source_manifest", lambda: changed)
    with pytest.raises(ValueError, match="source changed"):
        common.verify_source(contract)


def test_single_host_approval_cannot_authorize_the_new_parallel_scope():
    contract = contract_fixture()
    old_approval = {
        "schema": "successor_episode_approval_v1", "host": "5090", "approved": True,
        "contract_sha256": common.sha_json(contract), "user_instruction": "old single-host approval",
        "scope": "synthetic_cuda_qualification_and_fresh_nine_arm_validation",
    }
    with pytest.raises(ValueError):
        common.verify_approval(contract, old_approval)
    valid = approval_fixture(contract)
    common.verify_approval(contract, valid)
    for key, value in (("hosts", ["5090"]), ("contract_sha256", "0" * 64), ("approved", False)):
        changed = copy.deepcopy(valid)
        changed[key] = value
        with pytest.raises(ValueError):
            common.verify_approval(contract, changed)


def test_both_qualifications_share_first_start_deadline_without_reset(tmp_path, monkeypatch):
    contract, approval, bindings = qualified_fixture(tmp_path)
    monkeypatch.setattr(common.time, "time", lambda: 1500.0)
    permit = common.create_permit(contract, approval, bindings)
    assert permit["started_at_unix"] == 1000.0
    assert permit["deadline_unix"] == 173800.0
    for host in ("5080", "5090"):
        common.verify_permit(contract, permit, host)
    assert shared.fixed_start(permit, wall=lambda: 1500.0, monotonic=lambda: 20.0,
                              total_seconds=172800) == 172320.0
    monkeypatch.setattr(common.time, "time", lambda: 7200.0)
    later = common.create_permit(contract, approval, bindings)
    assert later["started_at_unix"] == permit["started_at_unix"]
    assert later["deadline_unix"] == permit["deadline_unix"]
    # The other host's physical receipt need not exist on the current host.
    Path(bindings["5080"]["path"]).unlink()
    common.verify_permit(contract, permit, "5090")


@pytest.mark.parametrize("drift", ["second_start", "extension", "expired"])
def test_permit_cannot_shift_reset_or_exceed_the_first_qualification_budget(tmp_path, monkeypatch, drift):
    contract, approval, bindings = qualified_fixture(tmp_path)
    monkeypatch.setattr(common.time, "time", lambda: 1500.0)
    permit = common.create_permit(contract, approval, bindings)
    if drift == "second_start":
        permit["started_at_unix"] = 1200.0
        permit["deadline_unix"] = 174000.0
    elif drift == "extension":
        permit["deadline_unix"] += 1.0
    else:
        monkeypatch.setattr(common.time, "time", lambda: 173800.0)
    with pytest.raises(ValueError, match="deadline"):
        common.verify_permit(contract, permit, "5090")


@pytest.mark.parametrize("corruption", ["source", "runtime", "gpu", "status", "model_check", "cost_coverage", "device"])
def test_each_host_rejects_a_failed_or_relabelled_other_host_qualification(tmp_path, monkeypatch, corruption):
    contract, approval, bindings = qualified_fixture(tmp_path)
    monkeypatch.setattr(common.time, "time", lambda: 1500.0)
    permit = common.create_permit(contract, approval, bindings)
    other = permit["qualifications"]["5080"]
    receipt = other["receipt"]
    if corruption == "source":
        receipt["source_files_sha256"] = "0" * 64
    elif corruption == "runtime":
        receipt["runtime"]["torch"] = "unqualified"
    elif corruption == "gpu":
        receipt["runtime"]["gpu"]["uuid"] = contract["hosts"]["5090"]["gpu_uuid"]
    elif corruption == "status":
        receipt["status"] = "cost_gate_failed"
    elif corruption == "model_check":
        receipt["checks"][common.ARMS[2]]["activated_Q_K_U_gradients"] = False
    elif corruption == "cost_coverage":
        receipt["cost_gate_checks"] = [copy.deepcopy(receipt["cost_gate_checks"][0]) for _ in range(4)]
    else:
        receipt["device"] = "cpu"
    other["receipt_sha256"] = common.sha_json(receipt)
    with pytest.raises(ValueError):
        common.verify_permit(contract, permit, "5090")


def test_missing_native_receipt_and_changed_local_file_are_rejected(tmp_path, monkeypatch):
    contract, approval, bindings = qualified_fixture(tmp_path)
    monkeypatch.setattr(common.time, "time", lambda: 1500.0)
    with pytest.raises(ValueError, match="Both"):
        common.create_permit(contract, approval, {"5090": bindings["5090"]})
    permit = common.create_permit(contract, approval, bindings)
    path = Path(bindings["5090"]["path"])
    path.write_text(json.dumps(bindings["5090"]["receipt"]) + "\n ")
    with pytest.raises(ValueError, match="Local"):
        common.verify_permit(contract, permit, "5090")


@pytest.mark.parametrize("host", ["5080", "5090"])
def test_parallel_cli_dispatches_the_requested_host_with_its_contract(monkeypatch, tmp_path, host):
    captured = []
    monkeypatch.setattr(sys, "argv", ["runner", "execute", "--contract", str(tmp_path / "contract.json"),
                                     "--permit", str(tmp_path / "permit.json"), "--host", host])
    monkeypatch.setattr(shared, "supervisor", lambda *args, **kwargs: captured.append((args, kwargs)))
    runner.main()
    assert len(captured) == 1
    args, kwargs = captured[0]
    assert args[2] == host
    assert kwargs["common"] is common
    assert kwargs["assignment_validator"] is common.validate_assignment


@pytest.mark.parametrize("host", ["5080", "5090"])
def test_native_qualifier_uses_each_hosts_runtime_without_cuda_in_the_test(monkeypatch, host):
    contract = contract_fixture()
    receipt = synthetic_receipt(contract, host, 1000.0)
    calls = []
    inventories = iter((set(), {os.getpid()}))
    monkeypatch.setattr(common, "gpu_pids", lambda _: next(inventories))
    monkeypatch.setattr(common, "runtime_check", lambda c, h: calls.append(h) or receipt["runtime"])
    monkeypatch.setattr(common, "verify_source", lambda c: calls.append("source_verified"))

    def synthetic_measurement(device):
        assert device == "cuda:0"
        calls.append("measurement_fixture")
        return copy.deepcopy(receipt)

    monkeypatch.setattr(qualifier, "verify", synthetic_measurement)
    actual = qualifier.qualify(contract, host, 1000.0)
    assert calls == [host, "measurement_fixture", "source_verified"]
    assert actual["status"] == "passed"
    assert actual["host"] == host
    assert actual["runtime"]["gpu"]["uuid"] == contract["hosts"][host]["gpu_uuid"]
    assert len(actual["cost_gate_checks"]) == 4
    assert all(all(row[key] for key in ("step", "relative_memory", "device_memory", "parameters"))
               for row in actual["cost_gate_checks"])


def test_parallel_production_injects_its_common_without_changing_legacy_defaults(monkeypatch, tmp_path):
    contract = contract_fixture()
    captured = []
    monkeypatch.setattr(comparison, "production", lambda *args, **kwargs: captured.append((args, kwargs)) or "delegated")
    assert runner.production(contract, {}, "5080", {}, lambda: None) == "delegated"
    assert captured[0][0][2] == "5080"
    assert captured[0][1]["execution_common"] is common
    original = legacy.read_contract(common.ROOT / LEGACY_CONTRACT, check_source=False)
    assert list(original["hosts"]) == ["5090"]
    assert comparison.partition_totals(original, "5090") == {"arms": 9, "steps": 6816240, "replays": 18}
    args = comparison.training_args(original, original["datasets"][0], tmp_path / "unused", legacy.ARMS[0])
    assert args.execution_role == "fresh_successor_episode_5090_validation"
    with pytest.raises(ValueError):
        legacy.validate_assignment(original, "5080")
    with pytest.raises(ValueError, match="schema"):
        common.read_contract(common.ROOT / LEGACY_CONTRACT, "5090", check_source=False)
