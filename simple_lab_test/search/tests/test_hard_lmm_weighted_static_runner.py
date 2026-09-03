import json
from pathlib import Path
import subprocess
import sys

import pytest

from paper.scripts import run_hard_lmm_weighted_static as runner


def test_registered_single_change_contract():
    c = runner.read(runner.CONTRACT)
    assert c["temperature"] == 1. and c["additional_parameters"] == 0
    assert c["seed"] == 42 and len(c["dataset_order"]) == 4
    assert runner.digest(runner.ROOT / c["baseline_registry"]) == c["baseline_registry_sha256"]
    assert not c["online_memory_updates"] and not c["held_out_rows_materialized"]


@pytest.mark.parametrize("phase,epochs", [("smoke", "1"), ("screening", "300")])
def test_command_does_not_resume_or_run_baselines_or_subsample(phase, epochs):
    c = runner.read(runner.CONTRACT)
    for row in runner.read(runner.ROOT / c["baseline_registry"])["datasets"]:
        cmd = runner.command(row, Path("/project"), Path("/fresh"), "a" * 40, phase)
        options = dict(zip(cmd[4:-1:2], cmd[5:-1:2]))
        assert options["--epochs"] == epochs
        assert options["--backbones"] == runner.BACKBONE
        assert options["--seeds"] == "42"
        assert options["--model-role"] == "t0_weighted_static_retrieval"
        assert options["--lambda-tail"] == "0.0"
        assert options["--time-scale"] == "3.0"
        assert "--force-rerun" not in cmd and "--max-train-batches" not in cmd
        assert "--max-val-batches" not in cmd and "--eval-test" not in cmd


def test_existing_artifact_cannot_be_entered(tmp_path, monkeypatch):
    (tmp_path / "status.json").write_text('{"status":"complete"}')
    monkeypatch.setattr(sys, "argv", ["runner", "--phase", "smoke", "--project-root", str(tmp_path),
                                     "--output-root", str(tmp_path), "--source-revision", "a" * 40])
    with pytest.raises(FileExistsError):
        runner.main()
    assert json.loads((tmp_path / "status.json").read_text())["status"] == "complete"


def test_exception_always_records_failed_without_launch(tmp_path, monkeypatch):
    output = tmp_path / "new"
    monkeypatch.setattr(sys, "argv", ["runner", "--phase", "smoke", "--project-root", str(tmp_path),
                                     "--output-root", str(output), "--source-revision", "a" * 40])
    def broken(_):
        raise ValueError("source mismatch")
    monkeypatch.setattr(runner, "verify_source", broken)
    with pytest.raises(ValueError, match="source mismatch"):
        runner.main()
    assert runner.read(output / "status.json")["status"] == "failed"
    assert not list(output.rglob("*.pt"))


@pytest.mark.parametrize("problem", ["low_memory", "gdm", "gpu_process", "xid", "journal_unavailable"])
def test_resource_preflight_fails_closed(problem, monkeypatch):
    def fake(command, **kwargs):
        text, code = "", 0
        if command[0] == "systemctl":
            text, code = ("active", 0) if problem == "gdm" else ("inactive", 3)
        elif command[0] == "nvidia-smi":
            if "--query-gpu=memory.free" in command:
                text = "100" if problem == "low_memory" else "15000"
            elif problem == "gpu_process":
                text = "1000, python, 200"
        elif command[0] == "journalctl":
            text = "NVRM: Xid 8" if problem == "xid" else ""
            code = 1 if problem == "journal_unavailable" else 0
        return subprocess.CompletedProcess(command, code, stdout=text, stderr="")
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(ValueError):
        runner.preflight(12000)


def test_body_gate_uses_count_weighted_absolute_errors_not_mean_of_bins():
    c = runner.read(runner.CONTRACT)
    reference = {"quantity_rows": [{"stratum": s, "count": n, "qty_mae": v} for s, n, v in
                                   [("le_p50", 80, 1.), ("p50_p90", 15, 4.), ("p90_p95", 5, 12.), ("gt_p99", 1, 50.)]],
                 "best_val_qty_mae": 3., "best_val_qty_rmse": 9., "best_val_time_nll": -1.}
    candidate = json.loads(json.dumps(reference))
    for r in candidate["quantity_rows"][:3]:
        r["qty_mae"] *= .9
    result = runner.compare(reference, candidate, c["per_dataset_gate"])
    assert result["baseline"]["body_mae"] == 2.
    assert result["passed"]
    candidate["best_val_qty_rmse"] *= 1.03
    assert not runner.compare(reference, candidate, c["per_dataset_gate"])["passed"]


def test_legacy_intermittent_missing_unused_std_is_explicit_not_fabricated():
    row = {"dataset": "intermittent_v2", "checkpoint_source_revision": "044add1f3de768d804d9f0269fd0013bd9658a35"}
    old = {"interface_meta": {"train_target_mean": 1.}, "variant": runner.VARIANT}
    new = {"interface_meta": {"train_target_mean": 1., "train_target_std": 2.}, "variant": runner.VARIANT}
    assert runner.validate_quantity_initialization(new, old, row) == "legacy_unrecorded_not_used_by_log_mse"
    assert "train_target_std" not in old["interface_meta"]
    for change in ({"dataset": "yellow_trip_hourly"}, {"checkpoint_source_revision": "a" * 40}):
        with pytest.raises(ValueError):
            runner.validate_quantity_initialization(new, old, row | change)
    with pytest.raises(ValueError):
        runner.validate_quantity_initialization(new | {"variant": "lognormal"}, old, row)
    with pytest.raises(ValueError):
        runner.validate_quantity_initialization(new | {"interface_meta": {"train_target_mean": 1.1}}, old, row)


def test_present_baseline_std_must_still_match():
    row = {"dataset": "yellow_trip_hourly"}
    old = {"interface_meta": {"train_target_mean": 1., "train_target_std": 2.}, "variant": runner.VARIANT}
    assert runner.validate_quantity_initialization(old, old, row) == "matched"
    with pytest.raises(ValueError):
        runner.validate_quantity_initialization(old | {"interface_meta": {"train_target_mean": 1., "train_target_std": 3.}}, old, row)
