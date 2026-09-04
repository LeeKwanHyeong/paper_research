"""Budget/early-stop and CUDA-e1 gating for the approved fresh screening."""

import argparse
import copy
import platform
import sys

import polars
import pytest
import torch

from paper.scripts import run_hard_lmm_key_value_screening as screening
from paper.scripts import run_hard_lmm_key_value_smoke as smoke
from paper.scripts import run_count_aware_tpp_backbone_control as entry
from simple_lab_test.search.tests.test_hard_lmm_key_value_smoke import create_synthetic_e1


@pytest.mark.parametrize("name", smoke.DATASETS)
def test_screening_changes_only_budget_and_execution_role(monkeypatch, tmp_path, name):
    _, rows = smoke.frozen_documents()
    row = next(r for r in rows if r["dataset"] == name)
    before = smoke.command(row, tmp_path, tmp_path / "new", "a" * 40)
    after = screening.command(row, tmp_path, tmp_path / "new", "a" * 40)
    changes = {before[i-1] for i, (a, b) in enumerate(zip(before, after)) if a != b}
    assert changes == {"--epochs", "--min-epochs", "--early-stopping-patience", "--execution-role"}
    monkeypatch.setattr(sys, "argv", ["test"] + after[4:])
    args = entry.parse_args()
    assert args.epochs == 300 and args.min_epochs == args.early_stopping_patience == 40
    assert args.backbones == smoke.BACKBONE and args.seeds == "42" and not args.force_rerun
    assert args.max_series is args.max_train_batches is args.max_val_batches is None


def history(n=41):
    return [{"epoch": i, "val_joint_objective": float(i), "train_event_count": 38, "train_all_finite": True}
            for i in range(1, n+1)]


@pytest.mark.parametrize("bad", ["premature", "continued", "best", "gap", "partial", "finite", "stop_flag"])
def test_history_rejects_invalid_completion(bad):
    h = history()
    s = dict(epochs=300, completed_epochs=41, best_epoch=1, stopped_early=True)
    if bad == "premature":
        h = history(40)
        s["completed_epochs"] = 40
    elif bad == "continued":
        h = history(42)
        s["completed_epochs"] = 42
    elif bad == "best":
        s["best_epoch"] = 2
    elif bad == "gap":
        h[1]["epoch"] = 3
    elif bad == "partial":
        h[0]["train_event_count"] = 37
    elif bad == "finite":
        h[0]["train_all_finite"] = False
    else:
        s["stopped_early"] = False
    with pytest.raises(ValueError):
        smoke.validate_history(h, s, {"train_targets": 38}, screening=True)


def test_history_accepts_first_legal_stop_and_budget_exhaustion():
    s = dict(epochs=300, completed_epochs=41, best_epoch=1, stopped_early=True)
    assert smoke.validate_history(history(), s, {"train_targets": 38}, screening=True)["epoch"] == 1
    h = history(300)
    for row in h:
        row["val_joint_objective"] *= -1
    s.update(completed_epochs=300, best_epoch=300, stopped_early=False)
    assert smoke.validate_history(h, s, {"train_targets": 38}, screening=True)["epoch"] == 300


def test_source_compatibility_includes_training_but_not_orchestration():
    files = {"models/a.py": "a", "data_loader/b.py": "b", "simple_lab_test/search/common/runner.py": "c",
             "paper/scripts/run_taxi_quantity_interface_ablation.py": "d",
             "paper/scripts/run_count_aware_tpp_backbone_control.py": "e",
             "paper/scripts/count_aware_tpp_backbone/training.py": "f",
             "paper/scripts/run_hard_lmm_key_value_screening.py": "g"}
    assert len(screening.training_files({"files": files})) == 6
    changed = files | {"models/a.py": "wrong"}
    assert screening.training_files({"files": files}) != screening.training_files({"files": changed})


def test_screening_audit_handles_best_vs_last_and_rejects_digest(monkeypatch, tmp_path):
    # Synthetic CPU e1 weights with synthetic e41 metadata exercise the auditor,
    # not a real training trajectory or performance result.
    output, row, launch, reference = create_synthetic_e1(monkeypatch, tmp_path)
    run = (output / smoke.SUMMARY).parent
    s = smoke.read(output / smoke.SUMMARY)
    old_history = smoke.read(run / "history.json")["history"][0]
    h = [dict(old_history, epoch=i, val_joint_objective=old_history["val_joint_objective"] + i - 1)
         for i in range(1, 42)]
    launch["epochs"] = 300
    launch["early_stopping"].update(min_epochs=40, patience=40)
    smoke.save(output / "launch_contract.json", launch)
    s.update(epochs=300, completed_epochs=41, stopped_early=True)
    smoke.save(output / smoke.SUMMARY, s)
    smoke.save(run / "history.json", {"history": h})
    last_path = run / "last_epoch_state.pt"
    last = torch.load(last_path, map_location="cpu", weights_only=False)
    last.update(epoch=41, history=h)
    last["model_state_dict"]["lmm.memory_keys"] += .01
    torch.save(last, last_path)
    assert smoke.audit_run(output, row, (launch, reference), "a" * 40, screening=True)["status"] == "passed"
    last["best_state_dict"]["lmm.memory_keys"] += .01
    torch.save(last, last_path)
    with pytest.raises(ValueError, match="digest"):
        smoke.audit_run(output, row, (launch, reference), "a" * 40, screening=True)


def test_incomplete_smoke_prevents_any_training_and_records_failed(monkeypatch, tmp_path):
    spec = copy.deepcopy(smoke.read(screening.CONTRACT))
    spec["runtime"] = {"python": platform.python_version(), "torch": torch.__version__,
                       "cuda": torch.version.cuda, "polars": polars.__version__}
    real_read = smoke.read
    monkeypatch.setattr(smoke, "read", lambda p: spec if p == screening.CONTRACT else (
        {"performance_training_authorized": True, "scope": spec["scope"]} if p.name == "source_manifest.json" else real_read(p)))
    monkeypatch.setattr(smoke, "verify_source", lambda *_: "sha")
    monkeypatch.setattr(smoke, "baseline", lambda *_: {})
    def fail(*_):
        raise ValueError("Smoke incomplete/changed")
    monkeypatch.setattr(screening, "verify_smoke", fail)
    monkeypatch.setattr(smoke, "run_logged", lambda *_: pytest.fail("Training must not start"))
    args = argparse.Namespace(project_root=tmp_path, output_root=tmp_path / "new", source_revision="a" * 40)
    with pytest.raises(ValueError, match="Smoke incomplete"):
        screening.execute(args)
    assert real_read(args.output_root / "status.json")["status"] == "failed"
    with pytest.raises(FileExistsError):
        screening.execute(args)
