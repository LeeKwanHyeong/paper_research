from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from paper.scripts import continue_hard_lmm_bounded_qk_campaign as recovery
from paper.scripts import run_hard_lmm_bounded_qk_campaign as campaign_policy


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(tmp_path: Path) -> dict:
    controller = Path(recovery.__file__).resolve()
    root = controller.parents[2]
    return {
        "schema_version": 1,
        "recovery_id": recovery.RECOVERY_ID,
        "status": "frozen_before_recovery_outputs",
        "diagnosis": recovery.DIAGNOSIS,
        "training": {
            "source_root": str(tmp_path / "source89"),
            "source_revision": recovery.TRAINING_REVISION,
            "python_executable": str(tmp_path / "python"),
            "runtime": {"device_type": "cuda"},
            "dependencies": {"numpy": "test"},
            "gpu_uuid": "GPU-test",
        },
        "controller": {
            "source_root": str(root),
            "source_revision": "1" * 40,
            "path": str(controller),
            "file_sha256": recovery.sha256(controller),
        },
        "original": {
            "output_root": str(tmp_path / "failed"),
            "deployment_manifest_path": str(tmp_path / "failed" / "deployment_manifest.json"),
            "deployment_manifest_sha256": recovery.DEPLOYMENT_SHA256,
            "failed_status_path": str(tmp_path / "failed" / "campaign_status.json"),
            "failed_status_sha256": recovery.FAILED_STATUS_SHA256,
            "failed_job": recovery.FAILED_JOB,
        },
        "new_output_root": str(tmp_path / "continued"),
        "nvrtc": {
            "library_directory": str(recovery.LIBRARY_DIRECTORY),
            "ld_library_path": str(recovery.LIBRARY_DIRECTORY),
            "files": [
                {"path": str(recovery.LIBRARY_DIRECTORY / name), "sha256": digest}
                for name, digest in recovery.LIBRARY_SHA256.items()
            ],
        },
        "reused_files": {name: "0" * 64 for name in recovery.expected_reused_files()},
        "failed_duration_cache": {"reuse": False},
        "held_out_test_evaluated": False,
        "additional_seeds_authorized": False,
    }


def _continuation(tmp_path: Path) -> recovery.Continuation:
    manifest = _manifest(tmp_path)
    new_root = Path(manifest["new_output_root"])
    new_root.mkdir(parents=True)
    paths = {
        "training_root": Path(manifest["training"]["source_root"]),
        "controller_root": Path(manifest["controller"]["source_root"]),
        "old_root": Path(manifest["original"]["output_root"]),
        "new_root": new_root,
        "deployment": Path(manifest["original"]["deployment_manifest_path"]),
        "failed_status": Path(manifest["original"]["failed_status_path"]),
    }
    fake_campaign = SimpleNamespace()
    reused = {"joint_e1": {dataset: {"status": "passed", "metrics": {}}
                            for dataset in recovery.DATASETS}}
    return recovery.Continuation(
        manifest_path=tmp_path / "recovery.json", manifest_sha256="a" * 64,
        manifest=manifest, paths=paths, campaign=fake_campaign, deployment={},
        contract={}, reused=reused,
    )


def test_manifest_is_incident_specific_and_child_environment_is_closed(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    paths = recovery.validate_recovery_manifest(manifest)
    assert paths["new_root"] == tmp_path / "continued"

    monkeypatch.setenv("LD_LIBRARY_PATH", "/unregistered/library")
    environment = recovery.child_environment(manifest, paths["new_root"])
    assert environment["LD_LIBRARY_PATH"] == str(recovery.LIBRARY_DIRECTORY)
    assert environment["TORCH_KERNEL_CACHE_PATH"] == str(tmp_path / "continued/torch_kernel_cache")
    assert environment["MPLCONFIGDIR"] == str(tmp_path / "continued/matplotlib")
    assert all(environment[name] == value for name, value in recovery.THREAD_ENVIRONMENT.items())

    drifted = copy.deepcopy(manifest)
    drifted["failed_duration_cache"] = {"reuse": True}
    with pytest.raises(ValueError, match="must not be reused"):
        recovery.validate_recovery_manifest(drifted)
    drifted = copy.deepcopy(manifest)
    drifted["reused_files"].pop(next(iter(drifted["reused_files"])))
    with pytest.raises(ValueError, match="population drift"):
        recovery.validate_recovery_manifest(drifted)


def test_completed_file_population_is_hash_bound_and_read_only(tmp_path):
    manifest = _manifest(tmp_path)
    root = Path(manifest["original"]["output_root"])
    for index, relative in enumerate(sorted(manifest["reused_files"])):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"completed-{index}".encode()
        path.write_bytes(content)
        manifest["reused_files"][relative] = _digest(content)
    before = {name: (root / name).read_bytes() for name in manifest["reused_files"]}
    assert recovery.verify_pinned_files(root, manifest) == manifest["reused_files"]
    assert before == {name: (root / name).read_bytes() for name in manifest["reused_files"]}

    victim = root / sorted(manifest["reused_files"])[0]
    victim.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Completed artifact drift"):
        recovery.verify_pinned_files(root, manifest)


def test_reused_e1_is_not_relaunched_and_policy_stops_at_first_quantity_failure(tmp_path, monkeypatch):
    continuation = _continuation(tmp_path)
    launched = []
    monkeypatch.setattr(continuation, "run_command", lambda *args, **kwargs: launched.append(args[0]))

    order = []

    def joint(dataset, phase):
        order.append(("joint", dataset, phase))
        if phase == "e1":
            return continuation.joint(dataset, phase)
        return {"status": "passed", "metrics": {
            "raw_rmse": 10.0, "overall_mae": 1.0, "body_mae": 1.0, "gt_p99_mae": 1.0,
        }}

    def duration(dataset, role, full_fit, audit):
        order.append(("duration", dataset, role, full_fit))
        return {"status": "passed", "proper_time_nll": 0.0}

    refs = {
        dataset: {
            "B": {"metrics": {"raw_rmse": 10.0, "overall_mae": 1.0,
                                "body_mae": 1.0, "gt_p99_mae": 1.0}},
            "FULL": {"metrics": {"raw_rmse": 10.0, "overall_mae": 1.0,
                                   "body_mae": 1.0, "gt_p99_mae": 1.0}},
            "aligned_B": {"normalized_time_nll": 0.0},
        }
        for dataset in recovery.DATASETS
    }
    contract = {
        "phases": {
            "e1": {"datasets_in_order": list(recovery.DATASETS)},
            "seed42_screening": {"datasets_in_order": [
                "insta_market_basket", "yellow_trip_hourly", "intermittent_frozen_5000"
            ]},
        },
        "datasets": refs,
    }
    result = campaign_policy.execute_phases(
        contract, joint=joint, duration=duration, record_gate=lambda *_: None,
    )
    assert result["status"] == "stopped_quantity_gate_failed"
    assert launched == []
    assert order[:3] == [("joint", dataset, "e1") for dataset in recovery.DATASETS]
    assert order[3:6] == [("duration", dataset, "FULL", False) for dataset in recovery.DATASETS]
    assert order[6] == ("joint", "insta_market_basket", "seed42_screening")
    assert len(order) == 7


def test_duration_cache_is_new_and_full_fit_uses_only_its_continuation_e1(tmp_path, monkeypatch):
    continuation = _continuation(tmp_path)
    commands = []

    def duration_manifest(**kwargs):
        return {"dataset": kwargs["dataset"], "role": kwargs["role"], "source": kwargs["source"]}

    def run_command(job_id, command, output):
        commands.append((job_id, command, output))
        output.mkdir(parents=True)
        if job_id.startswith("normalized_e1_"):
            (output / "cache").mkdir()

    continuation.campaign = SimpleNamespace(
        DURATION_RUNNER=tmp_path / "source89/paper/scripts/run_backbone_normalized_duration.py",
        duration_manifest=duration_manifest,
        audit_duration=lambda output, **kwargs: {"status": "passed", "proper_time_nll": 0.2},
    )
    continuation.contract = {
        "datasets": {"yellow_trip_hourly": {"FULL": {"source": {"checkpoint": "frozen"}}}}
    }
    monkeypatch.setattr(continuation, "run_command", run_command)
    monkeypatch.setattr(continuation, "complete", lambda *_: None)

    continuation.duration("yellow_trip_hourly", "FULL", False, None)
    continuation.duration("yellow_trip_hourly", "FULL", True, None)
    assert [row[0] for row in commands] == [
        "normalized_e1_FULL_yellow_trip_hourly", "normalized_full_FULL_yellow_trip_hourly"
    ]
    e1_cache = Path(continuation.manifest["new_output_root"]) / (
        "jobs/normalized_e1_FULL_yellow_trip_hourly/cache"
    )
    full_command = commands[1][1]
    assert full_command[full_command.index("--feature-cache-dir") + 1] == str(e1_cache)
    assert continuation.manifest["original"]["output_root"] not in " ".join(full_command)


def test_failed_attempt_cannot_be_retried_and_subprocess_runs_once(tmp_path, monkeypatch):
    manifest = _manifest(tmp_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    Path(manifest["new_output_root"]).mkdir()
    with pytest.raises(ValueError, match="automatic restart is forbidden"):
        recovery.run_recovery(manifest_path, recovery.sha256(manifest_path))

    other = _continuation(tmp_path / "other")
    monkeypatch.setattr(other, "verify_boundary", lambda: {"gpu_name": "RTX 5090"})
    calls = []

    def fail_once(*args, **kwargs):
        calls.append(args[0])
        raise subprocess.CalledProcessError(1, args[0])

    monkeypatch.setattr(recovery.subprocess, "run", fail_once)
    output = Path(other.manifest["new_output_root"]) / "jobs/failed_once"
    with pytest.raises(subprocess.CalledProcessError):
        other.run_command("failed_once", ["python", "job.py"], output)
    with pytest.raises(FileExistsError):
        other.run_command("failed_once", ["python", "job.py"], output)
    assert len(calls) == 1
