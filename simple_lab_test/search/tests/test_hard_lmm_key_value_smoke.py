"""Smoke scope, failure handling, packaging and real-format synthetic audit."""

import argparse
import copy
import io
import json
import subprocess
import sys
import tarfile
from unittest.mock import Mock

import polars as pl
import pytest

from paper.scripts import run_count_aware_tpp_backbone_control as entry
from paper.scripts import run_hard_lmm_key_value_smoke as smoke
from paper.scripts import package_hard_lmm_key_value_smoke as packager


@pytest.mark.parametrize("name", smoke.DATASETS)
def test_full_e1_only_command(monkeypatch, tmp_path, name):
    spec, rows = smoke.frozen_documents()
    row = next(r for r in rows if r["dataset"] == name)
    cmd = smoke.command(row, tmp_path, tmp_path / "fresh", "a" * 40)
    monkeypatch.setattr(sys, "argv", ["test"] + cmd[4:])
    args = entry.parse_args()
    assert args.epochs == args.min_epochs == args.early_stopping_patience == 1
    assert args.device == "cuda" and args.model_role == smoke.ROLE and args.backbones == smoke.BACKBONE
    assert args.execution_role == smoke.EXECUTION_ROLE and args.seeds == "42"
    assert not args.force_rerun and args.max_series is None
    assert args.max_train_batches is args.max_val_batches is None
    assert not spec["performance_training_authorized"] and not spec["held_out_test"]
    with pytest.raises(ValueError, match="Only Taxi/Instacart"):
        smoke.command(row | {"dataset": "raf_spare_parts"}, tmp_path, tmp_path, "a" * 40)


@pytest.mark.parametrize("failure", [None, "preflight", "cuda", "dataset", "audit"])
def test_failure_status_no_retry_and_existing_output_protected(monkeypatch, tmp_path, failure):
    spec, rows = smoke.frozen_documents()
    monkeypatch.setattr(smoke, "frozen_documents", lambda: (spec, rows))
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
    before = (args.output_root / "status.json").read_bytes()
    with pytest.raises(FileExistsError):
        smoke.execute(args)
    assert (args.output_root / "status.json").read_bytes() == before


def test_packaging_root_sentinel_reference_digests_and_no_overwrite(monkeypatch, tmp_path):
    from simple_lab_test.common.pathing import resolve_project_root
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name in ("models/__init__.py", "utils/example.py", "simple_lab_test/example.ipynb"):
            data = b"# fixture\n"
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    def git(cmd, **kwargs):
        return {"rev-parse": "a" * 40, "diff": b"", "archive": stream.getvalue()}[cmd[1]]
    monkeypatch.setattr(packager.subprocess, "check_output", git)
    output = tmp_path / "source.tar.gz"
    packager.package(output)
    snapshot = tmp_path / "isolated"
    snapshot.mkdir()
    with tarfile.open(output) as archive:
        archive.extractall(snapshot, filter="data")
    assert resolve_project_root(snapshot / "models") == snapshot
    assert list((snapshot / "sample_data").iterdir()) == []
    assert not list(snapshot.rglob("*.ipynb"))
    manifest = json.loads((snapshot / "source_manifest.json").read_text())
    assert not manifest["performance_training_authorized"]
    assert len(list((snapshot / "references").rglob("*.pt"))) == 2
    for name, expected in manifest["files"].items():
        assert smoke.digest(snapshot / name) == expected
    with pytest.raises(FileExistsError):
        packager.package(output)


def test_changed_input_rejected_before_loading_checkpoint(monkeypatch, tmp_path):
    _, rows = smoke.frozen_documents()
    monkeypatch.setattr(smoke, "digest", lambda path: "wrong")
    load = Mock(side_effect=AssertionError("Do not load unverified checkpoint"))
    monkeypatch.setattr("torch.load", load)
    with pytest.raises(ValueError, match="Input changed"):
        smoke.baseline(rows[0], tmp_path)
    load.assert_not_called()


def create_synthetic_e1(monkeypatch, tmp_path):
    _, rows = smoke.frozen_documents()
    row = rows[0]
    data, split, output = tmp_path / "data.parquet", tmp_path / "split.json", tmp_path / "fresh"
    pl.DataFrame({"oper_part_no": ["a"] * 24 + ["b"] * 24,
                  "seq": list(range(1, 25)) * 2, "delta_t": [1.] * 48,
                  "demand_qty": (list(map(float, range(1, 24))) + [-999999.]) * 2,
                  "chronological_split": (["train"] * 20 + ["validation"] * 3 + ["test"]) * 2}).write_parquet(data)
    split.write_text('{}')
    cmd = smoke.command(row, tmp_path, output, "a" * 40)
    for flag, value in (("--data", str(data)), ("--split-manifest", str(split)), ("--device", "cpu")):
        cmd[cmd.index(flag) + 1] = value
    monkeypatch.setattr(sys, "argv", ["test"] + cmd[4:])
    monkeypatch.setattr(entry, "sha256_file", lambda path: row[
        "data_sha256" if path == data else "split_manifest_sha256"])
    monkeypatch.setattr(entry.pl, "read_parquet", Mock(side_effect=AssertionError("No held-out materialization")))
    entry.main()
    launch, summary = smoke.read(output / "launch_contract.json"), smoke.read(output / smoke.SUMMARY)
    reference = copy.deepcopy(summary)
    reference["parameter_count"] = row["baseline_parameters"]
    synthetic_row = row | {"train_targets": 38, "validation_targets": 6}
    return output, synthetic_row, launch, reference


def test_real_runner_synthetic_e1_and_audit_reject_drift(monkeypatch, tmp_path):
    output, synthetic_row, launch, reference = create_synthetic_e1(monkeypatch, tmp_path)
    assert smoke.audit_run(output, synthetic_row, (launch, reference), "a" * 40)["status"] == "passed"
    for change in ({"epochs": 300}, {"partial_smoke": True}, {"held_out_test_evaluated": True},
                   {"source_revision": "wrong"}, {"lambda_tail": .1}):
        smoke.save(output / "launch_contract.json", launch | change)
        with pytest.raises(ValueError):
            smoke.audit_run(output, synthetic_row, (launch, reference), "a" * 40)
    smoke.save(output / "launch_contract.json", launch)
    for change in ({"train_targets": 39}, {"validation_targets": 7}):
        with pytest.raises(ValueError, match="Partial"):
            smoke.audit_run(output, synthetic_row | change, (launch, reference), "a" * 40)
    (output / "test_summary.json").write_text('{}')
    with pytest.raises(ValueError, match="Held-out artifact"):
        smoke.audit_run(output, synthetic_row, (launch, reference), "a" * 40)
