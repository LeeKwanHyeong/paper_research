"""Synthetic-only guards for frozen reference gates; no model or data loading."""
from copy import deepcopy
import hashlib
import json

import pytest

from paper.scripts.compare_titantpp_backbone_loop import compare_rows, first_best, stats, verify_registry


def fixture():
    pop = {"count": 4, "target_identity_sha256": "a" * 64, "truth_sha256": "b" * 64, "time_sha256": "c" * 64}
    registry = {"dataset": "raf_spare_parts", "expected_population": {s: deepcopy(pop) for s in ("validation", "test")},
                "historical_cnn": {}, "external_frontier": {}, "rows": []}
    groups = {"candidate": [], "same_host": [], "capacity": []}
    for split in ("validation", "test"):
        registry["historical_cnn"][split] = {m: {"mean": 10., "sample_sd": 0.} for m in ("qty_rmse", "qty_mae")}
        registry["external_frontier"][split] = {m: {"mean": 8., "models": ["external"]} for m in ("qty_rmse", "qty_mae")}
        for seed in (42, 52, 62):
            registry["rows"].append({"model": "historical_CNN", "split": split, "seed": seed,
                                     "metrics": {"qty_rmse": 10., "qty_mae": 10.}})
            for model, value in (("candidate", 7.), ("same_host", 9.), ("capacity", 8.)):
                groups[model].append({"dataset": "raf_spare_parts", "model": model, "split": split, "seed": seed,
                    "status": "complete", "validation_selected_checkpoint_verified": True,
                    "count": 4, "population": deepcopy(pop), "selected_epoch": 4,
                    "checkpoint_sha256": str(seed)[0] * 64, "metrics": {"qty_rmse": value, "qty_mae": value},
                    "runtime": {"host": "5090", "gpu_uuid": "gpu", "python": "3.12", "torch": "2.11",
                                "cudnn": 92000, "threads": 4}})
    return registry, groups


def test_both_splits_both_metrics_pass_and_ddof_one():
    r, g = fixture()
    out = compare_rows(r, g["candidate"], g["same_host"], g["capacity"])
    assert out["user_performance_gate"] and out["same_host_control_gate"] and out["capacity_attribution_gate"]
    assert len(out["seed_differences"]) == 12
    assert stats([1., 2., 3.])["sample_sd"] == 1.


def test_tie_fails_and_single_bad_test_metric_fails():
    r, g = fixture()
    for row in g["candidate"]:
        if row["split"] == "test":
            row["metrics"]["qty_mae"] = 8.
    assert compare_rows(r, g["candidate"], g["same_host"])["user_performance_gate"] is False


def test_same_host_and_capacity_are_separate_from_user_gate():
    r, g = fixture()
    for row in g["same_host"] + g["capacity"]:
        row["metrics"] = {"qty_rmse": 6., "qty_mae": 6.}
    out = compare_rows(r, g["candidate"], g["same_host"], g["capacity"])
    assert out["user_performance_gate"] is True
    assert out["same_host_control_gate"] is False and out["capacity_attribution_gate"] is False


def test_worse_seed_is_preserved_without_adding_all_seed_gate():
    r, g = fixture()
    for row in g["candidate"]:
        row["metrics"]["qty_rmse"] = 11. if row["seed"] == 42 else 5.
    out = compare_rows(r, g["candidate"], g["same_host"])
    assert out["user_performance_gate"]
    assert any(x["seed"] == 42 and x["differences"]["historical_CNN"]["absolute_candidate_minus_reference"] > 0
               for x in out["seed_differences"])


@pytest.mark.parametrize("mutation", ["missing_seed", "duplicate_seed", "wrong_truth", "nonfinite", "different_checkpoint", "partial", "runtime", "model"])
def test_invalid_evidence_never_passes(mutation):
    r, g = fixture(); rows = g["candidate"]
    if mutation == "missing_seed": rows.pop()
    elif mutation == "duplicate_seed": rows[0]["seed"] = rows[1]["seed"]
    elif mutation == "wrong_truth": rows[0]["population"]["truth_sha256"] = "f" * 64
    elif mutation == "nonfinite": rows[0]["metrics"]["qty_mae"] = float("nan")
    elif mutation == "different_checkpoint": rows[0]["checkpoint_sha256"] = "f" * 64
    elif mutation == "partial": rows[0]["status"] = "running"
    elif mutation == "runtime": rows[0]["runtime"]["cudnn"] = 91900
    elif mutation == "model": rows[0]["model"] = "other_candidate"
    with pytest.raises(ValueError): compare_rows(r, rows, g["same_host"])


def test_first_strict_minimum_uses_first_tie_not_mae():
    h = [{"epoch": 1, "val_qty_rmse": 3., "val_qty_mae": 1.},
         {"epoch": 2, "val_qty_rmse": 2., "val_qty_mae": 4.},
         {"epoch": 3, "val_qty_rmse": 2., "val_qty_mae": 1.}]
    assert first_best(h)["epoch"] == 2
    with pytest.raises(ValueError): first_best(h[1:])


def test_source_tampering_fails_registry_verification(tmp_path):
    src = tmp_path / "source.json"; src.write_text("{}")
    reg = {"status": "verified", "dataset": "raf_spare_parts",
           "source_references": {"source.json": hashlib.sha256(src.read_bytes()).hexdigest()}}
    p = tmp_path / "registry.json"; p.write_text(json.dumps(reg))
    expected = hashlib.sha256(p.read_bytes()).hexdigest()
    assert verify_registry(p, expected, tmp_path) == reg
    src.write_text('{"changed": true}')
    with pytest.raises(ValueError): verify_registry(p, expected, tmp_path)
    with pytest.raises(ValueError): verify_registry(p, "0" * 64, tmp_path)
