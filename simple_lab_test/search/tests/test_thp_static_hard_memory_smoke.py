"""e1 orchestration safety and real-format synthetic artifact verification."""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from paper.scripts import run_count_aware_tpp_backbone_control as entry
from paper.scripts import run_thp_static_hard_memory_smoke as smoke
from paper.scripts.count_aware_tpp_backbone.thp_static_contract import validate_static_memory_launch
from simple_lab_test.search.tests.test_thp_static_hard_memory_model import frozen_factory_source
from simple_lab_test.search.tests.test_thp_static_hard_memory_runner import (
    test_actual_entrypoint_filters_heldout_and_persists_identity_or_failed_status as create_synthetic_artifact,
)


@pytest.mark.parametrize("name", smoke.DATASETS)
def test_only_full_e1_commands_pass_frozen_launch_guard(monkeypatch, tmp_path, name):
    _, registry = smoke.frozen_documents()
    row = next(r for r in registry["datasets"] if r["dataset"] == name)
    cmd = smoke.command(row, tmp_path, tmp_path / "fresh", "a" * 40)
    monkeypatch.setattr(sys, "argv", ["test"] + cmd[4:])
    args = entry.parse_args()
    validate_static_memory_launch(args, (42,))
    assert args.epochs == args.min_epochs == args.early_stopping_patience == 1
    assert args.device == "cuda" and args.model_role == smoke.ROLE
    assert not args.force_rerun and args.max_series is None
    assert args.max_train_batches is args.max_val_batches is None


def test_exported_reference_factory_is_pinned(monkeypatch, tmp_path):
    reference = frozen_factory_source()
    path = tmp_path / "reference.py"
    path.write_text(reference)
    monkeypatch.setenv("THP_STATIC_MEMORY_REFERENCE_FACTORY", str(path))
    assert frozen_factory_source() == reference
    path.write_text(reference + "\n")
    with pytest.raises(ValueError, match="checksum"):
        frozen_factory_source()


@pytest.mark.parametrize("failure", [None, "preflight", "cuda", "dataset", "audit"])
def test_execution_order_failure_status_and_no_screening(monkeypatch, tmp_path, failure):
    _, registry = smoke.frozen_documents()
    monkeypatch.setattr(smoke, "verify_source", lambda revision: "manifest_hash")
    monkeypatch.setattr(smoke, "baseline", lambda *args: {})
    monkeypatch.setattr(smoke, "read", lambda path: {})
    calls = []
    def preflight(*args):
        calls.append("preflight")
        if failure == "preflight":
            raise ValueError("Insufficient VRAM")
        return {"gdm": "inactive"}
    def run(cmd, path, **kwargs):
        calls.append(path.name)
        if (failure == "cuda" and path.name.startswith("cuda")) or (
                failure == "dataset" and path.name.startswith(smoke.DATASETS[0])):
            raise subprocess.CalledProcessError(1, cmd)
    def audit(*args):
        if failure == "audit":
            raise ValueError("Contract mismatch")
        return {"status": "passed", "dataset": args[1]["dataset"]}
    monkeypatch.setattr(smoke, "preflight", preflight)
    monkeypatch.setattr(smoke, "run_logged", run)
    monkeypatch.setattr(smoke, "audit_run", audit)
    args = argparse.Namespace(project_root=tmp_path, output_root=tmp_path / "new", source_revision="a" * 40)
    if failure:
        with pytest.raises((ValueError, subprocess.CalledProcessError)):
            smoke.execute(args)
    else:
        smoke.execute(args)
    status = json.loads((args.output_root / "status.json").read_text())
    assert status["status"] == ("failed" if failure else "complete")
    assert not status["performance_training_authorized"] and not status["held_out_test_evaluated"]
    if failure:
        assert "error" in status and not status["completed"]
        assert f"{smoke.DATASETS[1]}.log" not in calls
    else:
        assert calls == ["preflight", "cuda_contract_tests.log", "preflight", f"{smoke.DATASETS[0]}.log",
                         "preflight", f"{smoke.DATASETS[1]}.log", "preflight"]
        assert len(status["completed"]) == 2
        assert status["command"][status["command"].index("--epochs") + 1] == "1"
    sentinel = (args.output_root / "status.json").read_bytes()
    with pytest.raises(FileExistsError):
        smoke.execute(args)
    assert (args.output_root / "status.json").read_bytes() == sentinel


def test_full_auditor_on_real_runner_schema_and_rejected_drift(monkeypatch, tmp_path):
    # Only synthetic CPU events; no real-data model training in local tests.
    create_synthetic_artifact(monkeypatch, tmp_path, None)
    output = tmp_path / "fresh"
    launch = smoke.read(output / "launch_contract.json")
    summary = smoke.read(output / smoke.SUMMARY)
    row = {key: launch[key] for key in ("data_sha256", "split_manifest_sha256", "quantity_contract", "history_length_contract")}
    row.update(dataset="yellow_trip_hourly", lookback=168, max_seq_len=256, train_targets=38, validation_targets=6)
    reference = copy.deepcopy(summary)
    reference["parameter_count"] = 100291
    assert smoke.audit_run(output, row, reference, "a" * 40)["status"] == "passed"
    baseline_bytes = (output / "audit.json").read_bytes()
    for change in ({"epochs": 300}, {"partial_smoke": True}, {"held_out_test_evaluated": True},
                   {"source_revision": "wrong"}, {"lambda_tail": .1}):
        smoke.save(output / "launch_contract.json", launch | change)
        with pytest.raises(ValueError):
            smoke.audit_run(output, row, reference, "a" * 40)
    smoke.save(output / "launch_contract.json", launch)
    altered = copy.deepcopy(launch)
    altered["time_head"]["time_scale"] = 1.
    smoke.save(output / "launch_contract.json", altered)
    with pytest.raises(ValueError, match="Time launch"):
        smoke.audit_run(output, row, reference, "a" * 40)
    smoke.save(output / "launch_contract.json", launch)
    (output / "test_summary.json").write_text('{}')
    with pytest.raises(ValueError, match="Held-out artifact"):
        smoke.audit_run(output, row, reference, "a" * 40)
    assert (output / "audit.json").read_bytes() == baseline_bytes


def test_changed_baseline_is_rejected_before_checkpoint_loading(monkeypatch, tmp_path):
    _, registry = smoke.frozen_documents()
    row = registry["datasets"][0]
    monkeypatch.setattr(smoke, "digest", lambda path: "wrong")
    load = Mock(side_effect=AssertionError("Do not load an unverified checkpoint"))
    monkeypatch.setattr("torch.load", load)
    with pytest.raises(ValueError, match="Baseline/input changed"):
        smoke.baseline(row, tmp_path)
    load.assert_not_called()
