"""Schedule, gate, audit, and retry tests for raw-RMSE baseline alignment."""

from __future__ import annotations

import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from paper.scripts import control_raw_rmse_baseline_alignment as controller
from paper.scripts import run_raw_rmse_baseline_alignment_job as launcher


CONTRACT_PATH = ROOT / launcher.CONTRACT_REL


def load_contract() -> dict:
    return launcher.read_json(CONTRACT_PATH)


def receipt(job: dict, *, mae: float = 999.0, rmse: float = 999.0) -> dict:
    return {
        "status": "passed",
        "job_id": job["job_id"],
        "phase": job["phase"],
        "host_role": job["host_role"],
        "dataset": job["dataset"],
        "backbone": job["backbone"],
        "seed": 42,
        "metrics": {"overall_mae": mae, "raw_rmse": rmse},
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
    }


def completed_for(jobs: list[dict], **metrics_by_backbone) -> dict[str, dict]:
    values: dict[str, dict] = {}
    for job in jobs:
        metrics = metrics_by_backbone.get(job["backbone"], {})
        values[job["job_id"]] = receipt(
            job,
            mae=metrics.get("mae", 999.0),
            rmse=metrics.get("rmse", 999.0),
        )
    return values


def runtime_fingerprint_for(job: dict) -> dict:
    gpu_name = f"NVIDIA GeForce RTX {job['host_role']}"
    return {
        "schema_version": 1,
        "host_role": job["host_role"],
        "runtime": {
            "interpreter_executable": "/opt/test/bin/python",
            "python_version": "3.12.9",
            "python_implementation": "CPython",
            "torch_version": "2.7.0",
            "torch_cuda_version": "12.8",
            "cudnn_version": 90701,
            "numpy_version": "2.2.5",
            "polars_version": "1.30.0",
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_current_device": 0,
            "gpu_name": gpu_name,
            "gpu_compute_capability": [12, 0],
            "gpu_total_memory_bytes": 32 * 1024**3,
        },
        "launch_preflight": {
            "gpu_name": gpu_name,
            "free_vram_mib": 16000,
            "compute_processes": [],
            "gdm": "inactive",
        },
    }


def test_frozen_contract_has_six_5090_e1_and_only_seed42_full_jobs() -> None:
    contract = load_contract()
    validated = launcher.validate_contract(contract)
    e1 = contract["schedule"]["e1_jobs"]
    full = contract["schedule"]["seed42_e300_jobs"]
    assert len(validated["jobs"]) == 12
    assert len(e1) == 6
    assert {(job["dataset"], job["backbone"]) for job in e1} == {
        (dataset, backbone)
        for dataset in launcher.DATASETS
        for backbone in launcher.BACKBONES
    }
    assert {job["host_role"] for job in e1} == {"5090"}
    assert {(job["backbone"], job["host_role"]) for job in full} == {
        ("rmtpp", "5080"),
        ("thp", "5090"),
    }
    assert {job["seed"] for job in [*e1, *full]} == {42}
    assert contract["schedule"]["additional_seeds_in_immediate_stage"] == []
    assert contract["execution"]["seeds52_and62"] is False
    assert contract["execution"]["held_out_test"] is False


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value["scope"].__setitem__("seeds", [42, 52]),
        lambda value: value["scope"].__setitem__("checkpoint_monitor", "validation_joint_objective"),
        lambda value: value["held_out_lock"].__setitem__("test_metrics_allowed", True),
        lambda value: value["execution"].__setitem__("seeds52_and62", True),
        lambda value: value["schedule"]["e1_jobs"][0].__setitem__("host_role", "5080"),
        lambda value: value["schedule"]["seed42_e300_jobs"][0].__setitem__("seed", 52),
        lambda value: value["runtime_policy"]["hosts"]["5080"].__setitem__("require_gdm_inactive", True),
    ),
)
def test_contract_drift_fails_closed(mutation) -> None:
    changed = copy.deepcopy(load_contract())
    mutation(changed)
    with pytest.raises(ValueError):
        launcher.validate_contract(changed)


def test_scheduler_advances_e1_as_one_strict_prefix() -> None:
    contract = load_contract()
    e1 = contract["schedule"]["e1_jobs"]
    initial = controller.schedule_snapshot(contract, {})
    assert initial["next_job_by_host"] == {"5080": None, "5090": e1[0]["job_id"]}

    completed = completed_for(e1[:3])
    replay = controller.schedule_snapshot(contract, completed)
    assert replay["next_job_by_host"]["5090"] == e1[3]["job_id"]
    assert replay["completed_job_ids"] == [job["job_id"] for job in e1[:3]]

    out_of_order = completed_for([e1[0], e1[2]])
    with pytest.raises(ValueError, match="not a prefix"):
        controller.schedule_snapshot(contract, out_of_order)


def test_instacart_jobs_split_across_hosts_then_wait_for_both() -> None:
    contract = load_contract()
    e1 = contract["schedule"]["e1_jobs"]
    screening = contract["schedule"]["seed42_e300_jobs"][:2]
    completed = completed_for(e1)
    ready = controller.schedule_snapshot(contract, completed)
    assert ready["next_job_by_host"] == {
        "5080": screening[0]["job_id"],
        "5090": screening[1]["job_id"],
    }

    completed[screening[0]["job_id"]] = receipt(screening[0])
    waiting = controller.schedule_snapshot(contract, completed)
    assert waiting["next_job_by_host"] == {"5080": None, "5090": screening[1]["job_id"]}
    assert waiting["instacart_gate"]["status"] == "pending"


def test_gate_requires_strict_B_win_on_both_metrics_against_each_baseline() -> None:
    contract = load_contract()
    e1 = contract["schedule"]["e1_jobs"]
    screening = contract["schedule"]["seed42_e300_jobs"][:2]
    b = next(
        row["B_t0_raw_rmse_validation_metrics"]
        for row in contract["datasets"]
        if row["dataset"] == "insta_market_basket"
    )
    completed = completed_for(e1)
    completed.update(completed_for(
        screening,
        rmtpp={"mae": b["overall_mae"] + 0.2, "rmse": b["raw_rmse"] + 0.2},
        thp={"mae": b["overall_mae"] + 0.1, "rmse": b["raw_rmse"] + 0.1},
    ))
    passed = controller.schedule_snapshot(contract, completed)
    assert passed["instacart_gate"]["status"] == "passed"
    assert passed["next_job_by_host"] == {
        "5080": contract["schedule"]["seed42_e300_jobs"][2]["job_id"],
        "5090": contract["schedule"]["seed42_e300_jobs"][3]["job_id"],
    }

    equality = copy.deepcopy(completed)
    equality[screening[0]["job_id"]]["metrics"]["raw_rmse"] = b["raw_rmse"]
    stopped = controller.schedule_snapshot(contract, equality)
    assert stopped["status"] == "stopped_gate_failed"
    assert stopped["next_job_by_host"] == {"5080": None, "5090": None}
    assert not stopped["instacart_gate"]["comparisons"]["rmtpp"]["raw_rmse"]["strictly_lower"]


def test_post_gate_jobs_are_idempotently_skipped_per_host() -> None:
    contract = load_contract()
    e1 = contract["schedule"]["e1_jobs"]
    full = contract["schedule"]["seed42_e300_jobs"]
    completed = completed_for(e1)
    completed.update(completed_for(full[:2]))
    first = controller.schedule_snapshot(contract, completed)
    completed[first["next_job_by_host"]["5080"]] = receipt(full[2])
    second = controller.schedule_snapshot(contract, completed)
    assert second["next_job_by_host"]["5080"] == full[4]["job_id"]
    assert second["next_job_by_host"]["5090"] == full[3]["job_id"]

    completed.update(completed_for(full[3:]))
    final = controller.schedule_snapshot(contract, completed)
    assert final["status"] == "complete"
    assert final["next_job_by_host"] == {"5080": None, "5090": None}


def test_training_command_freezes_selector_role_and_has_no_held_out_or_extra_seed_flags(tmp_path) -> None:
    contract = load_contract()
    validated = launcher.validate_contract(contract)
    job = contract["schedule"]["e1_jobs"][0]
    row = validated["datasets"][job["dataset"]]
    command = launcher.training_command(
        python="python",
        project=tmp_path,
        training_output=tmp_path / "out",
        revision="a" * 40,
        dataset_row=row,
        job=job,
    )
    assert command[command.index("--model-role") + 1] == "raw_rmse_baseline_alignment"
    assert command[command.index("--checkpoint-monitor") + 1] == "validation_raw_quantity_rmse"
    assert command[command.index("--backbones") + 1] == "rmtpp"
    assert command[command.index("--seeds") + 1] == "42"
    assert command[command.index("--epochs") + 1] == "1"
    assert "52" not in command and "62" not in command
    flags = [argument.lower() for argument in command if argument.startswith("--")]
    assert not any("test" in argument or "held_out" in argument for argument in flags)
    assert "--max-train-batches" not in command
    assert "--max-val-batches" not in command


def test_registry_roots_and_gpu_lock_have_no_cli_override(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    revision = "a" * 40
    monkeypatch.setattr(launcher, "EXECUTION_ROOT_BASE", tmp_path / "registry")
    paths = launcher.fixed_execution_paths(revision)
    expected = tmp_path / "registry" / f"{launcher.CONTRACT_ID}_{revision}"
    assert paths["coordination_root"] == expected / "coordination"
    assert paths["artifact_roots"] == {
        "5080": expected / "artifacts" / "5080",
        "5090": expected / "artifacts" / "5090",
    }

    parsed = launcher.parse_args([
        "--source-revision", revision,
        "--host-role", "5090",
        "--job-id", "e1_01_intermittent_frozen_5000_rmtpp_seed42",
    ])
    assert not hasattr(parsed, "output_root")
    assert not hasattr(parsed, "gpu_lock_path")
    parsed_controller = controller.parse_args([
        "--source-revision", revision,
        "--host-role", "5090",
    ])
    assert not hasattr(parsed_controller, "coordination_root")
    assert not hasattr(parsed_controller, "artifact_root_5090")
    assert not hasattr(parsed_controller, "gpu_lock_5090")


def test_runtime_fingerprint_binds_versions_gpu_idle_and_gdm() -> None:
    contract = load_contract()
    job = contract["schedule"]["e1_jobs"][0]
    fingerprint = runtime_fingerprint_for(job)
    assert launcher.validate_runtime_fingerprint(
        fingerprint,
        contract=contract,
        job=job,
    ) == fingerprint

    changed = copy.deepcopy(fingerprint)
    changed["runtime"]["numpy_version"] = ""
    with pytest.raises(ValueError, match="numpy_version"):
        launcher.validate_runtime_fingerprint(changed, contract=contract, job=job)
    changed = copy.deepcopy(fingerprint)
    changed["launch_preflight"]["compute_processes"] = [123]
    with pytest.raises(ValueError, match="not idle"):
        launcher.validate_runtime_fingerprint(changed, contract=contract, job=job)


def test_runtime_fingerprint_file_digest_and_content_are_reaudited(tmp_path) -> None:
    contract = load_contract()
    job = contract["schedule"]["e1_jobs"][0]
    path = tmp_path / launcher.RUNTIME_FINGERPRINT_FILENAME
    launcher.save_json_atomic(path, runtime_fingerprint_for(job))
    payload, digest = launcher.audit_runtime_fingerprint_file(
        path,
        contract=contract,
        job=job,
    )
    assert payload == runtime_fingerprint_for(job)
    assert digest == launcher.sha256_file(path)

    changed = runtime_fingerprint_for(job)
    changed["runtime"]["torch_cuda_version"] = ""
    launcher.save_json_atomic(path, changed)
    with pytest.raises(ValueError, match="torch_cuda_version"):
        launcher.audit_runtime_fingerprint_file(path, contract=contract, job=job)


def test_selected_gate_metrics_are_recomputed_from_strata_and_bound_to_history_checkpoint() -> None:
    names = ("le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99")
    rows = [
        {"stratum": name, "count": 2, "qty_mae": 1.0, "qty_rmse": 2.0}
        for name in names
    ]
    history = [{"epoch": 1, "val_qty_rmse": 2.0, "val_qty_mae": 1.0}]
    summary = {
        "selected_metric_value": 2.0,
        "best_val_qty_rmse": 2.0,
        "best_val_qty_mae": 1.0,
    }
    checkpoint = {"selected_metric_value": 2.0}
    assert launcher.audit_selected_quantity_metrics(
        summary=summary,
        history=history,
        checkpoint=checkpoint,
        quantity_rows=rows,
        expected_count=10,
    ) == {"overall_mae": 1.0, "raw_rmse": 2.0}

    changed = copy.deepcopy(summary)
    changed["best_val_qty_mae"] = 0.5
    with pytest.raises(ValueError, match="overall MAE"):
        launcher.audit_selected_quantity_metrics(
            summary=changed,
            history=history,
            checkpoint=checkpoint,
            quantity_rows=rows,
            expected_count=10,
        )
    changed_history = copy.deepcopy(history)
    changed_history[0]["val_qty_mae"] = 0.5
    with pytest.raises(ValueError, match="Selected history MAE"):
        launcher.audit_selected_quantity_metrics(
            summary=summary,
            history=changed_history,
            checkpoint=checkpoint,
            quantity_rows=rows,
            expected_count=10,
        )


def test_mirrored_receipt_requires_exact_runtime_fingerprint_and_digest() -> None:
    job = load_contract()["schedule"]["e1_jobs"][0]
    fingerprint = runtime_fingerprint_for(job)
    fresh = {key: f"value-{key}" for key in controller.FRESH_AUDIT_RECEIPT_FIELDS}
    fresh["runtime_fingerprint"] = fingerprint
    fresh["runtime_fingerprint_sha256"] = "a" * 64
    fresh["launch_gpu_preflight"] = fingerprint["launch_preflight"]
    receipt_payload = copy.deepcopy(fresh)
    controller.require_receipt_matches_fresh_audit(
        job_id=job["job_id"],
        receipt=receipt_payload,
        fresh=fresh,
    )

    receipt_payload["runtime_fingerprint"]["runtime"]["torch_version"] = "tampered"
    with pytest.raises(ValueError, match="runtime_fingerprint"):
        controller.require_receipt_matches_fresh_audit(
            job_id=job["job_id"],
            receipt=receipt_payload,
            fresh=fresh,
        )


def test_standalone_launcher_reaudits_fixed_registry_and_cannot_bypass_full_gate(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_contract()
    full_job = contract["schedule"]["seed42_e300_jobs"][2]
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setattr(launcher, "EXECUTION_ROOT_BASE", tmp_path / "fixed-registry")
    monkeypatch.setattr(controller, "verify_source_manifest", lambda *args, **kwargs: {})
    monkeypatch.setattr(controller, "validate_b_reference_evidence", lambda *args, **kwargs: {})
    monkeypatch.setattr(controller, "verify_dataset_inputs", lambda *args, **kwargs: None)
    args = argparse.Namespace(
        project_root=project,
        contract=CONTRACT_PATH,
        source_revision="a" * 40,
        host_role=full_job["host_role"],
        job_id=full_job["job_id"],
        python_5080="/host/5080/python",
        python_5090="/host/5090/python",
        verify_only=False,
    )
    with pytest.raises(ValueError, match="Fresh audited schedule"):
        launcher.execute_job(args)
    assert not (tmp_path / "fixed-registry").exists()


def test_job_identity_is_portable_across_project_artifact_and_python_paths(tmp_path) -> None:
    contract = load_contract()
    validated = launcher.validate_contract(contract)
    job = contract["schedule"]["seed42_e300_jobs"][0]
    row = validated["datasets"][job["dataset"]]
    identities = []
    for suffix, python in (("first", "/host/a/python"), ("second", "/other/python")):
        project = tmp_path / suffix / "project"
        project.mkdir(parents=True)
        manifest = project / "source_manifest.json"
        manifest.write_text('{"same": true}\n', encoding="utf-8")
        training = tmp_path / suffix / "artifacts" / "jobs" / job["job_id"] / "training"
        command = launcher.training_command(
            python=python,
            project=project,
            training_output=training,
            revision="a" * 40,
            dataset_row=row,
            job=job,
        )
        identities.append(launcher._job_identity(
            contract_path=CONTRACT_PATH,
            manifest_path=manifest,
            revision="a" * 40,
            project=project,
            job=job,
            dataset_row=row,
            command=command,
            training_output=training,
        ))
    assert identities[0] == identities[1]
    serialized = json.dumps(identities[0], sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "/host/a/python" not in serialized
    assert identities[0]["command"][0] == "<host-python:5080>"


def test_B_gate_values_identities_and_checkpoint_hashes_are_bound_to_tracked_evidence() -> None:
    contract = load_contract()
    validated = launcher.validate_contract(contract)
    evidence = launcher.validate_b_reference_evidence(
        ROOT,
        contract=contract,
        datasets=validated["datasets"],
    )
    assert evidence["status"] == "common_raw_rmse_goal_met"

    changed = copy.deepcopy(contract)
    changed["datasets"][2]["B_t0_raw_rmse_validation_metrics"]["raw_rmse"] += 1e-12
    with pytest.raises(ValueError, match="not exactly bound"):
        launcher.validate_b_reference_evidence(
            ROOT,
            contract=changed,
            datasets={row["dataset"]: row for row in changed["datasets"]},
        )
    changed = copy.deepcopy(contract)
    changed["datasets"][2]["expected_validation_target_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="target identity drifted"):
        launcher.validate_b_reference_evidence(
            ROOT,
            contract=changed,
            datasets={row["dataset"]: row for row in changed["datasets"]},
        )
    changed = copy.deepcopy(contract)
    changed["datasets"][2]["B_t0_raw_rmse_checkpoint_state_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint state digest drifted"):
        launcher.validate_b_reference_evidence(
            ROOT,
            contract=changed,
            datasets={row["dataset"]: row for row in changed["datasets"]},
        )


def test_gpu_preflight_allows_5080_desktop_but_requires_idle_compute_and_5090_inactive_gdm(monkeypatch) -> None:
    contract = load_contract()
    replies = iter(("NVIDIA GeForce RTX 5080, 15000", "", "active"))
    monkeypatch.setattr(launcher, "command_output", lambda *args, **kwargs: next(replies))
    assert launcher.gpu_preflight(contract, "5080", backbone="rmtpp")["gdm"] == "active"

    replies = iter(("NVIDIA GeForce RTX 5080, 15000", "1234", "active"))
    monkeypatch.setattr(launcher, "command_output", lambda *args, **kwargs: next(replies))
    with pytest.raises(ValueError, match="compute processes"):
        launcher.gpu_preflight(contract, "5080", backbone="rmtpp")

    replies = iter(("NVIDIA GeForce RTX 5090, 30000", "", "active"))
    monkeypatch.setattr(launcher, "command_output", lambda *args, **kwargs: next(replies))
    with pytest.raises(ValueError, match="GDM"):
        launcher.gpu_preflight(contract, "5090", backbone="thp")

    replies = iter(("NVIDIA GeForce RTX 5080, 15000", "", "active"))
    monkeypatch.setattr(launcher, "command_output", lambda *args, **kwargs: next(replies))
    with pytest.raises(ValueError, match="not permitted"):
        launcher.gpu_preflight(contract, "5080", backbone="thp")


def test_launcher_reaudits_and_rejects_completed_job_replay(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "source_manifest.json").write_text("{}\n", encoding="utf-8")
    contract = load_contract()
    validated = launcher.validate_contract(contract)
    job = contract["schedule"]["e1_jobs"][0]
    monkeypatch.setattr(launcher, "EXECUTION_ROOT_BASE", tmp_path / "fixed-registry")
    paths = launcher.fixed_execution_paths("a" * 40)
    output = paths["artifact_roots"]["5090"]
    monkeypatch.setattr(launcher, "GPU_LOCK_PATH", tmp_path / "host-gpu.lock")
    calls = {"run": 0, "audit": 0}

    monkeypatch.setattr(launcher, "gpu_preflight", lambda *args, **kwargs: {"gpu_name": "test"})
    monkeypatch.setattr(
        launcher,
        "collect_runtime_fingerprint",
        lambda **kwargs: {"schema_version": 1, "host_role": "5090", "runtime": {}, "launch_preflight": {}},
    )

    def fresh_snapshot(_args):
        completed = {}
        receipt_path = output / "jobs" / job["job_id"] / "audit_receipt.json"
        if receipt_path.exists():
            completed[job["job_id"]] = receipt(job, mae=10.0, rmse=20.0)
        return controller.schedule_snapshot(contract, completed), validated, completed

    monkeypatch.setattr(launcher, "_fresh_schedule_snapshot", fresh_snapshot)

    def fake_run(command, *, log_path, project, env, child_slot, lock_fds):
        assert len(lock_fds) == 2
        assert len(set(lock_fds)) == 2
        calls["run"] += 1
        training = output / "jobs" / job["job_id"] / "training"
        training.mkdir(parents=True, exist_ok=True)
        launcher.save_json_atomic(training / "launch_contract.json", {"status": "complete"})

    def fake_audit(**kwargs):
        calls["audit"] += 1
        return {
            **receipt(job, mae=10.0, rmse=20.0),
            "schema_version": 1,
            "source_revision": "a" * 40,
            "checkpoint_monitor": launcher.SELECTOR,
            "best_epoch": 1,
            "completed_epochs": 1,
            "train_target_count": 1,
            "validation_target_count": 1,
            "checkpoint_state_sha256": "b" * 64,
            "artifact_sha256": {},
        }

    monkeypatch.setattr(launcher, "_run_logged", fake_run)
    monkeypatch.setattr(launcher, "audit_training_artifact", fake_audit)
    args = argparse.Namespace(
        project_root=project,
        contract=CONTRACT_PATH,
        source_revision="a" * 40,
        host_role="5090",
        job_id=job["job_id"],
        python_5080="/host/5080/python",
        python_5090="/host/5090/python",
        verify_only=False,
    )
    first = launcher.execute_job(args)
    assert first["crash_recovery"] is False
    with pytest.raises(ValueError, match="Fresh audited schedule"):
        launcher.execute_job(args)
    assert calls == {"run": 1, "audit": 1}
    assert launcher.read_json(
        output / "jobs" / job["job_id"] / "job_status.json"
    )["status"] == "complete"


def test_host_wide_gpu_lock_blocks_a_second_job_before_preflight(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "source_manifest.json").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "artifacts"
    contract = load_contract()
    job = contract["schedule"]["e1_jobs"][0]
    gpu_lock_path = tmp_path / "host-gpu.lock"
    monkeypatch.setattr(launcher, "EXECUTION_ROOT_BASE", tmp_path / "fixed-registry")
    output = launcher.fixed_execution_paths("a" * 40)["artifact_roots"]["5090"]
    monkeypatch.setattr(launcher, "GPU_LOCK_PATH", gpu_lock_path)
    validated = launcher.validate_contract(contract)
    monkeypatch.setattr(
        launcher,
        "_fresh_schedule_snapshot",
        lambda _args: (controller.schedule_snapshot(contract, {}), validated, {}),
    )
    preflight = pytest.fail
    monkeypatch.setattr(launcher, "gpu_preflight", preflight)
    args = argparse.Namespace(
        project_root=project,
        contract=CONTRACT_PATH,
        source_revision="a" * 40,
        host_role="5090",
        job_id=job["job_id"],
        python_5080="/host/5080/python",
        python_5090="/host/5090/python",
        verify_only=False,
    )
    with gpu_lock_path.open("a+", encoding="utf-8") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Host GPU is locked"):
            launcher.execute_job(args)


def test_run_logged_passes_job_and_gpu_locks_to_detached_trainer(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    class FakeChild:
        def wait(self) -> int:
            return 0

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return FakeChild()

    monkeypatch.setattr(launcher.subprocess, "Popen", fake_popen)
    child_slot = {"child": None}
    launcher._run_logged(
        ["python", "trainer.py"],
        log_path=tmp_path / "launcher.log",
        project=tmp_path,
        env={"PYTHONDONTWRITEBYTECODE": "1"},
        child_slot=child_slot,
        lock_fds=(11, 12),
    )
    assert captured["start_new_session"] is True
    assert captured["pass_fds"] == (11, 12)
    assert child_slot["child"] is None


def test_closing_parent_reference_keeps_inherited_gpu_lock_until_child_exits(
    tmp_path,
) -> None:
    lock_path = tmp_path / "gpu.lock"
    parent_lock = lock_path.open("a+", encoding="utf-8")
    fcntl.flock(parent_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import signal,time; "
                "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready', flush=True); time.sleep(30)"
            ),
        ],
        pass_fds=(parent_lock.fileno(),),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    probe = None
    try:
        assert child.stdout is not None
        assert child.stdout.readline().strip() == "ready"
        os.killpg(child.pid, signal.SIGTERM)
        launcher._close_parent_lock_reference(parent_lock)
        probe = lock_path.open("a+", encoding="utf-8")
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

        os.killpg(child.pid, signal.SIGKILL)
        child.wait(timeout=5)
        fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5)
        if probe is not None:
            probe.close()
        parent_lock.close()


def test_completed_job_allowlist_rejects_top_level_file_and_extra_seed_directory(tmp_path) -> None:
    job = load_contract()["schedule"]["e1_jobs"][0]
    job_dir = tmp_path / "job"
    for relative in launcher.expected_training_artifact_files(job):
        path = job_dir / "training" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    for name in (
        "job.lock",
        "job_identity.json",
        "job_status.json",
        "launcher.log",
        launcher.RUNTIME_FINGERPRINT_FILENAME,
    ):
        (job_dir / name).write_text("fixture\n", encoding="utf-8")
    launcher.audit_completed_job_layout(job_dir=job_dir, job=job, require_receipt=False)

    stale_status_write = job_dir / "job_status.json.tmp"
    stale_status_write.write_text("partial", encoding="utf-8")
    launcher.audit_completed_job_layout(job_dir=job_dir, job=job, require_receipt=False)
    stale_status_write.unlink()

    rogue = job_dir / "test_metrics.json"
    rogue.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside the allowlist"):
        launcher.audit_completed_job_layout(job_dir=job_dir, job=job, require_receipt=False)
    rogue.unlink()

    extra_seed = job_dir / "training/runs/rmtpp/count_only_log_regression/seed_52"
    extra_seed.mkdir()
    with pytest.raises(ValueError, match="directory layout drifted"):
        launcher.audit_completed_job_layout(job_dir=job_dir, job=job, require_receipt=False)


def test_artifact_root_and_jobs_registry_reject_unknown_entries(tmp_path) -> None:
    contract = load_contract()
    validated = launcher.validate_contract(contract)
    root_5080 = tmp_path / "5080"
    root_5080.mkdir()
    (root_5080 / "test_results.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown entry"):
        controller.audit_completed_jobs(
            project=ROOT,
            contract_path=CONTRACT_PATH,
            contract=contract,
            validated=validated,
            artifact_roots={"5080": root_5080, "5090": tmp_path / "5090"},
            revision="a" * 40,
            python_by_host={"5080": "python", "5090": "python"},
        )


def test_snapshot_rejects_future_incomplete_job_but_allows_current_one(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_contract()
    current_job, future_job = contract["schedule"]["e1_jobs"][:2]
    project = tmp_path / "project"
    project.mkdir()
    (project / "source_manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "EXECUTION_ROOT_BASE", tmp_path / "fixed-registry")
    monkeypatch.setattr(controller, "verify_source_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "validate_b_reference_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "verify_dataset_inputs", lambda *args, **kwargs: None)
    paths = launcher.fixed_execution_paths("a" * 40)
    args = argparse.Namespace(
        project_root=project,
        contract=CONTRACT_PATH,
        source_revision="a" * 40,
        python_5080="/host/5080/python",
        python_5090="/host/5090/python",
    )

    current_dir = paths["artifact_roots"]["5090"] / "jobs" / current_job["job_id"]
    current_dir.mkdir(parents=True)
    (current_dir / "job.lock").touch()
    snapshot, _, _ = controller.build_snapshot(args)
    assert snapshot["next_job_by_host"]["5090"] == current_job["job_id"]
    assert snapshot["audited_incomplete_job_ids"] == [current_job["job_id"]]

    (current_dir / "job.lock").unlink()
    current_dir.rmdir()
    future_dir = paths["artifact_roots"]["5090"] / "jobs" / future_job["job_id"]
    future_dir.mkdir()
    (future_dir / "job.lock").touch()
    with pytest.raises(ValueError, match="not a current next job"):
        controller.build_snapshot(args)


def test_current_incomplete_job_revalidates_stable_identity(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = load_contract()
    job = contract["schedule"]["e1_jobs"][0]
    project = tmp_path / "project"
    project.mkdir()
    (project / "source_manifest.json").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(launcher, "EXECUTION_ROOT_BASE", tmp_path / "fixed-registry")
    monkeypatch.setattr(controller, "verify_source_manifest", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "validate_b_reference_evidence", lambda *args, **kwargs: None)
    monkeypatch.setattr(controller, "verify_dataset_inputs", lambda *args, **kwargs: None)
    paths = launcher.fixed_execution_paths("a" * 40)
    job_dir = paths["artifact_roots"]["5090"] / "jobs" / job["job_id"]
    job_dir.mkdir(parents=True)
    (job_dir / "job.lock").touch()
    (job_dir / "job_identity.json").write_text("{}\n", encoding="utf-8")
    args = argparse.Namespace(
        project_root=project,
        contract=CONTRACT_PATH,
        source_revision="a" * 40,
        python_5080="/host/5080/python",
        python_5090="/host/5090/python",
    )
    with pytest.raises(ValueError, match="identity drifted"):
        controller.build_snapshot(args)


def test_wrapper_temporary_files_are_removed_only_after_job_lock_is_held(tmp_path) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    for relative in launcher.WRAPPER_TEMP_FILENAMES:
        (job_dir / relative).write_text("partial", encoding="utf-8")
    launcher._cleanup_wrapper_temporary_files(job_dir)
    assert all(not (job_dir / relative).exists() for relative in launcher.WRAPPER_TEMP_FILENAMES)


def test_resume_audit_binds_identity_history_current_and_best_states() -> None:
    from simple_lab_test.search.common.runner import canonical_state_dict_sha256

    contract = load_contract()
    validated = launcher.validate_contract(contract)
    job = contract["schedule"]["e1_jobs"][0]
    row = validated["datasets"][job["dataset"]]
    revision = "a" * 40
    history = [{
        "epoch": 1,
        "val_qty_rmse": 2.0,
        "val_joint_objective": 3.0,
    }]
    state = {"weight": torch.tensor([1.0])}
    state_digest = canonical_state_dict_sha256(state)
    arguments = {
        "epochs": 1,
        "batch_size": 128,
        "lr": 0.001,
        "lookback_weeks": row["lookback"],
        "max_seq_len": row["max_sequence_length"],
        "hidden_dim": 64,
        "lambda_log_qty": 1.0,
        "grad_clip": 1.0,
        "early_stopping_patience": 1,
        "min_epochs": 1,
        "lambda_tail": 0.0,
        "time_head_mode": "legacy_clamped_rmtpp",
        "time_scale": 3.0,
        "time_w_max": 10.0 / 3.0,
        "time_intercept_limit": 300.0,
        "max_train_batches": None,
        "max_val_batches": None,
        "model_role": launcher.MODEL_ROLE,
        "dataset_contract": job["dataset"],
        "max_series": None,
        "allow_partial_contract": True,
        "quantile_adaptive_strength": 0.0,
        "device": "cuda",
        "source_revision": revision,
        "data_sha256": row["data_sha256"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "execution_role": f"raw_rmse_baseline_alignment_{job['phase']}_{job['host_role']}_{job['job_id']}",
    }
    resume_identity = {
        "schema_version": 1,
        "backbone": job["backbone"],
        "variant": launcher.VARIANT,
        "seed": 42,
        "checkpoint_monitor": launcher.SELECTOR,
        "arguments": arguments,
        "quantity_contract": {},
        "interface_meta": {},
    }
    common = {
        "checkpoint_monitor": launcher.SELECTOR,
        "checkpoint_monitor_history_key": launcher.HISTORY_KEY,
        "checkpoint_selection": launcher.SELECTION,
        "source_revision": revision,
        "source_revision_history": [revision],
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "resume_identity": resume_identity,
        "initial_state_sha256": "1" * 64,
    }
    checkpoint = {
        **common,
        "best_epoch": 1,
        "selected_metric_value": 2.0,
        "model_state_dict": state,
        "model_state_sha256": state_digest,
    }
    summary = {
        **common,
        "checkpoint_state_sha256": state_digest,
    }
    resume = {
        **common,
        "checkpoint_type": "epoch_resume",
        "checkpoint_schema_version": 2,
        "epoch": 1,
        "history": history,
        "best_epoch": 1,
        "best_selection_value": 2.0,
        "model_state_dict": state,
        "model_state_sha256": state_digest,
        "best_state_dict": state,
        "best_state_sha256": state_digest,
        "optimizer_state_dict": {"state": {0: {"step": 1}}, "param_groups": [{}]},
        "rng_state": {"python": (), "numpy": (), "torch": torch.tensor([1]), "cuda": []},
        "train_loader_generator_state": torch.tensor([1], dtype=torch.uint8),
    }
    audited = launcher.audit_resume_state(
        resume=resume,
        checkpoint=checkpoint,
        summary=summary,
        history=history,
        job=job,
        dataset_row=row,
        revision=revision,
    )
    assert audited["resume_best_state_sha256"] == state_digest

    changed = copy.deepcopy(resume)
    changed["model_state_dict"]["weight"] = torch.tensor([2.0])
    with pytest.raises(ValueError, match="current state digest"):
        launcher.audit_resume_state(
            resume=changed,
            checkpoint=checkpoint,
            summary=summary,
            history=history,
            job=job,
            dataset_row=row,
            revision=revision,
        )
    changed = copy.deepcopy(resume)
    changed["source_revision_history"] = [revision, "b" * 40]
    with pytest.raises(ValueError, match="source history"):
        launcher.audit_resume_state(
            resume=changed,
            checkpoint=checkpoint,
            summary=summary,
            history=history,
            job=job,
            dataset_row=row,
            revision=revision,
        )


def test_artifact_audit_rejects_any_held_out_file_before_checkpoint_reuse(tmp_path) -> None:
    contract = load_contract()
    validated = launcher.validate_contract(contract)
    job = contract["schedule"]["e1_jobs"][0]
    row = validated["datasets"][job["dataset"]]
    output = tmp_path / "training"
    output.mkdir()
    launch = {
        "status": "complete",
        "model_role": launcher.MODEL_ROLE,
        "dataset": job["dataset"],
        "data_sha256": row["data_sha256"],
        "split_manifest_sha256": row["split_manifest_sha256"],
        "quantity_variants": [launcher.VARIANT],
        "backbones": [job["backbone"]],
        "seeds": [42],
        "expected_run_count": 1,
        "completed_run_count": 1,
        "epochs": 1,
        "batch_size": 128,
        "lr": 0.001,
        "lambda_log_qty": 1.0,
        "lambda_tail": 0.0,
        "grad_clip": 1.0,
        "lookback_weeks": row["lookback"],
        "max_seq_len": row["max_sequence_length"],
        "hidden_dim": 64,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "source_revision": "a" * 40,
        "partial_smoke": False,
        "split_rows": {"train": 1, "validation": 1},
        "validation_target_population": {
            "target_count": row["expected_validation_targets"],
            "target_identity_sha256": row["expected_validation_target_identity_sha256"],
            "target_quantity_sha256": row["expected_validation_target_quantity_sha256"],
        },
        "early_stopping": {
            "monitor": launcher.SELECTOR,
            "min_epochs": 1,
            "patience": 1,
            "comparison": "earliest_strict_finite_minimum",
            "restore": launcher.SELECTION,
        },
        "time_head": {"time_intercept_limit": 300.0},
    }
    launcher.save_json_atomic(output / "launch_contract.json", launch)
    (output / "test_metrics.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="validation-only allowlist"):
        launcher.audit_training_artifact(
            training_output=output,
            job=job,
            dataset_row=row,
            contract=contract,
            revision="a" * 40,
        )


def test_json_reader_rejects_duplicate_keys(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"status": "first", "status": "second"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON key"):
        launcher.read_json(path)
