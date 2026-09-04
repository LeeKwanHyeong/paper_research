"""Fresh screening orchestration, frozen imports, and prospective gate tests."""

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from paper.scripts import run_hard_lmm_local_time_screening as screening

smoke = screening.smoke


def registry():
    contract = smoke.read(smoke.CONTRACT)
    rows = {r["dataset"]: r for r in smoke.read(smoke.ROOT / contract["baseline"]["registry"])["datasets"]}
    return contract, rows


def test_fresh_command_only_changes_approved_budget_and_role(tmp_path):
    _, rows = registry()
    for name, row in rows.items():
        if name not in smoke.DATASETS:
            with pytest.raises(ValueError, match="Only Taxi/RAF"):
                screening.command(row, tmp_path, tmp_path / "fresh", tmp_path / "frozen")
            continue
        cmd = screening.command(row, tmp_path, tmp_path / name, tmp_path / "frozen")
        original = smoke.command(row, tmp_path, tmp_path / name, screening.TRAINING_REVISION)
        expected = list(original)
        expected[3] = str(tmp_path / "frozen/paper/scripts/run_count_aware_tpp_backbone_control.py")
        for key, value in {"epochs": "300", "min-epochs": "40", "early-stopping-patience": "40",
                           "execution-role": "hard_local_time_screening_5080"}.items():
            expected[expected.index("--" + key) + 1] = value
        assert cmd == expected
    with pytest.raises(ValueError, match="Fresh"):
        screening.command(rows[smoke.DATASETS[0]], tmp_path, tmp_path, tmp_path / "frozen")


def summary():
    return {"quantity_rows": [{"stratum": name, "count": 10, "qty_mae": 10.}
                              for name in ("le_p50", "p50_p90", "p90_p95", "gt_p99")],
            "best_val_qty_mae": 10., "best_val_qty_rmse": 20., "best_val_time_nll": 1.}


@pytest.mark.parametrize("violation", [None, "body", "rmse", "tail", "time", "missing", "empty", "zero", "nan", "duplicate"])
def test_gate_is_fixed_and_missing_evidence_never_passes(violation):
    contract, _ = registry()
    reference = summary()
    candidate = deepcopy(reference)
    for row in candidate["quantity_rows"][:3]:
        row["qty_mae"] = 9.5
    candidate["best_val_qty_rmse"] = 20.4
    candidate["quantity_rows"][-1]["qty_mae"] = 10.2
    if violation == "body":
        candidate["quantity_rows"][0]["qty_mae"] = 10.
    elif violation == "rmse":
        candidate["best_val_qty_rmse"] = 20.41
    elif violation == "tail":
        candidate["quantity_rows"][-1]["qty_mae"] = 10.21
    elif violation == "time":
        candidate["best_val_time_nll"] = 1.011
    elif violation == "missing":
        candidate["quantity_rows"].pop()
    elif violation == "empty":
        candidate["quantity_rows"][0]["count"] = 0
    elif violation == "zero":
        reference["best_val_qty_rmse"] = 0.
    elif violation == "nan":
        candidate["best_val_qty_rmse"] = float("nan")
    elif violation == "duplicate":
        candidate["quantity_rows"].append(candidate["quantity_rows"][0])
    result = screening.compare(reference, candidate, contract["performance_gate"])
    expected = "passed" if violation is None else "rejected" if violation in {"body", "rmse", "tail", "time"} else "not_evaluable"
    assert result["status"] == expected


def test_existing_artifact_is_never_modified(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    path.write_text("original")
    ready = Mock()
    monkeypatch.setattr(screening, "readiness", ready)
    with pytest.raises(FileExistsError):
        screening.execute(argparse.Namespace(output_root=tmp_path))
    assert path.read_text() == "original"
    ready.assert_not_called()


@pytest.mark.parametrize("boundary", ["readiness", "preflight", "training", "audit", None])
def test_failures_stop_without_retry_and_gate_rejection_still_evaluates_both(tmp_path, monkeypatch, boundary):
    contract, rows = registry()
    specs = {r["dataset"]: r for r in contract["datasets"]}
    monkeypatch.setattr(screening, "readiness", Mock(return_value=(contract, rows, specs, {n: summary() for n in smoke.DATASETS})))
    monkeypatch.setattr(screening, "verify_training", lambda *_: None)
    monkeypatch.setattr(smoke, "verify_source", lambda *_: None)
    monkeypatch.setattr(smoke, "preflight", Mock(return_value={"free_vram_mib": 16000}))
    monkeypatch.setattr(smoke, "audit_run", Mock(return_value={"status": "passed"}))
    run = Mock()
    monkeypatch.setattr(screening, "run_logged", run)
    original_read = smoke.read
    monkeypatch.setattr(smoke, "read", lambda p: {} if Path(p).name == "source_manifest.json" else
                        summary() if Path(p).name == "summary.json" else original_read(p))
    failure = RuntimeError("synthetic failure")
    if boundary == "readiness":
        screening.readiness.side_effect = failure
    elif boundary == "preflight":
        smoke.preflight.side_effect = failure
    elif boundary == "training":
        run.side_effect = failure
    elif boundary == "audit":
        smoke.audit_run.side_effect = failure
    args = argparse.Namespace(output_root=tmp_path / "fresh", project_root=tmp_path,
                              training_root=tmp_path / "frozen", orchestration_revision="b" * 40)
    if boundary:
        with pytest.raises(RuntimeError, match="synthetic"):
            screening.execute(args)
    else:
        screening.execute(args)
    status = original_read(args.output_root / "status.json")
    assert status["status"] == ("failed" if boundary else "complete")
    assert run.call_count == {"readiness": 0, "preflight": 0, "training": 1, "audit": 1, None: 2}[boundary]
    assert len(status["completed"]) == (0 if boundary else 2)
    if not boundary:
        assert status["candidate_decision"] == "hold"
    assert not list(args.output_root.glob("**/*test_summary*"))


@pytest.mark.parametrize("exit_code", [0, 7])
def test_child_uses_only_frozen_training_imports_and_records_failure(tmp_path, exit_code):
    root = tmp_path / "frozen"
    scripts = root / "paper/scripts"
    scripts.mkdir(parents=True)
    (root / "paper/__init__.py").write_text("FROZEN_SENTINEL = 42\n")
    entry = scripts / "trainer.py"
    entry.write_text("import paper, sys\nassert paper.FROZEN_SENTINEL == 42\n"
                     "assert sys.argv[1:] == ['--epochs', '300']\n"
                     f"raise SystemExit({exit_code})\n")
    report = tmp_path / "resources.json"
    result = subprocess.run([sys.executable, "-s", str(Path(screening.__file__)), "child",
                             "--entrypoint", str(entry), "--report", str(report), "--", "--epochs", "300"],
                            capture_output=True, text=True)
    assert result.returncode == exit_code, result.stderr
    record = smoke.read(report)
    assert record["status"] == ("complete" if exit_code == 0 else "failed")
    assert record["elapsed_seconds_including_startup"] >= 0
