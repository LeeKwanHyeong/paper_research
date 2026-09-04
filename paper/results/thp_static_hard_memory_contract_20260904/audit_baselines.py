"""Local baseline identity audit only; no model creation, forward or training."""

import ast
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
import torch
from simple_lab_test.search.common.runner import canonical_state_dict_sha256

REGISTRY = ROOT / "paper/contracts/count_aware_thp_static_hard_memory_baselines_v1.json"
BASE_REVISION = "af14bcfba59fef6f5b9f1d7c4682fba804beddaa"
DATASETS = {"yellow_trip_hourly": (38393, 8268), "raf_spare_parts": (25779, 6690)}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def finite(value):
    if isinstance(value, torch.Tensor):
        assert bool(torch.isfinite(value).all())
    elif isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, dict):
        for item in value.values():
            finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            finite(item)


def close(left, right):
    assert math.isclose(left, right, rel_tol=1e-10, abs_tol=1e-8), (left, right)


def class_ast(source, name):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


old_registry = ROOT / "paper/contracts/count_aware_hard_lmm_frozen_probe_v1.json"
reference_rows = {r["dataset"]: r for r in read(old_registry)["datasets"]}
registry = {"schema_version": 1, "registry_id": "count_aware_thp_static_hard_memory_baselines_v1",
    "created_on": "2026-09-04", "scope": "original_seed42_validation_baselines_only",
    "source_registry": str(old_registry.relative_to(ROOT)), "source_registry_sha256": digest(old_registry),
    "datasets": []}
checks = []
for name, (train_count, val_count) in DATASETS.items():
    row = reference_rows[name]
    artifact = ROOT / row["artifact_dir"]
    launch_path = artifact / "launch_contract.json"
    assert digest(launch_path) == row["contract_sha256"]
    for path_key, hash_key in (("data_path", "data_sha256"), ("split_manifest_path", "split_manifest_sha256")):
        assert digest(ROOT / row[path_key]) == row[hash_key]
    launch = read(launch_path)
    assert launch["status"] == "complete" and launch["model_role"] == "t0_common_control"
    assert launch["evaluation_scope"] == "validation_only" and not launch["held_out_test_evaluated"]
    for key, expected in {"epochs": 300, "batch_size": 128, "lr": .001, "grad_clip": 1., "hidden_dim": 64,
                          "lambda_log_qty": 1., "lookback_weeks": row["lookback"], "max_seq_len": row["max_seq_len"]}.items():
        assert launch[key] == expected, key
    assert launch["early_stopping"]["min_epochs"] == launch["early_stopping"]["patience"] == 40
    assert launch["early_stopping"]["monitor"] == "validation_joint_objective"
    for key, expected in {"mode": "legacy_clamped_rmtpp", "time_scale": 3., "time_w_max": 10/3,
                          "time_intercept_limit": 30., "time_wd_safety_limit": 40., "time_head_lr_multiplier": 1.}.items():
        assert launch["time_head"][key] == expected
    assert launch["time_head"]["train_time_statistics"]["target_count"] == train_count
    record = {k: row[k] for k in ("dataset", "contract_dataset", "artifact_dir", "data_path", "data_sha256",
                                 "split_manifest_path", "split_manifest_sha256", "lookback", "max_seq_len")}
    record.update(launch_contract_sha256=digest(launch_path), baseline_source_revision=launch["source_revision"],
                  train_targets=train_count, validation_targets=val_count,
                  quantity_contract=launch["quantity_contract"], history_length_contract=launch["history_length_contract"],
                  historical_split_rows=launch["split_rows"], baselines={})
    for path, klass in (("models/TPPs/CountAwareTPP.py", "CountAwareTHP"),
                        ("models/TPPs/TransformerHawkesTPP.py", "THPEncoderLayer")):
        old = subprocess.check_output(["git", "show", f"{launch['source_revision']}:{path}"], cwd=ROOT, text=True)
        current = subprocess.check_output(["git", "show", f"{BASE_REVISION}:{path}"], cwd=ROOT, text=True)
        assert class_ast(old, klass) == class_ast(current, klass), klass
    reference_strata = {}
    for model in ("thp", "titantpp"):
        run = artifact / "runs" / model / "count_only_log_regression/seed_42"
        paths = [run / f for f in ("summary.json", "history.json", "best_val_joint_objective_model.pt", "last_epoch_state.pt")]
        before = {str(p.relative_to(ROOT)): digest(p) for p in paths}
        summary, history = read(paths[0]), read(paths[1])["history"]
        best, last = [torch.load(p, map_location="cpu", weights_only=False) for p in paths[2:]]
        for payload in (summary, history, best, last):
            finite(payload)
        assert summary["status"] == "success" and summary["backbone"] == model and summary["seed"] == 42
        assert summary["variant"] == "count_only_log_regression" and summary["lambda_tail"] == 0.
        assert summary["source_revision"] == row["checkpoint_source_revision"]
        assert summary["source_revision_history"] == [row["checkpoint_source_revision"]]
        assert summary["evaluation_scope"] == "validation_only" and not summary["held_out_test_evaluated"]
        assert not [p for p in run.rglob("*") if p.name.startswith("test_")]
        assert [r["epoch"] for r in history] == list(range(1, len(history)+1))
        assert summary["completed_epochs"] == len(history) and summary["epochs"] == 300
        chosen = min(history, key=lambda r: r["val_joint_objective"])
        assert summary["best_epoch"] == chosen["epoch"]
        assert len(history) == 300 or len(history) - chosen["epoch"] == 40
        for length in range(40, len(history)):
            assert length - min(history[:length], key=lambda r: r["val_joint_objective"])["epoch"] < 40
        assert all(r["train_event_count"] == train_count and r["train_all_finite"] for r in history)
        for key in ("qty_mae", "qty_rmse", "time_nll", "log_qty_mse", "joint_objective"):
            close(summary["best_val_"+key], chosen["val_"+key])
        state_hash = canonical_state_dict_sha256(best["model_state_dict"])
        assert state_hash == summary["checkpoint_state_sha256"] == canonical_state_dict_sha256(last["best_state_dict"])
        assert sum(v.numel() for v in best["model_state_dict"].values()) == summary["parameter_count"]
        groups = last["optimizer_state_dict"]["param_groups"]
        assert len(groups) == 1
        for key, expected in {"lr": .001, "betas": (.9, .999), "eps": 1e-8, "weight_decay": .01, "amsgrad": False}.items():
            assert groups[0][key] == expected
        for axis in ("quantity", "history"):
            bins = summary[axis+"_rows"]
            signature = [(b["stratum"], b["stratum_label"], b["count"]) for b in bins]
            if model == "thp":
                reference_strata[axis] = signature
            else:
                assert signature == reference_strata[axis], (name, axis)
            assert sum(b["count"] for b in bins) == val_count
            close(sum(b["count"]*b["qty_mae"] for b in bins)/val_count, summary["best_val_qty_mae"])
            close(math.sqrt(sum(b["count"]*b["qty_rmse"]**2 for b in bins)/val_count), summary["best_val_qty_rmse"])
        body = [b for b in summary["quantity_rows"] if b["stratum"] in {"le_p50", "p50_p90", "p90_p95"}]
        tail = [b for b in summary["quantity_rows"] if b["stratum"] == "gt_p99"]
        assert len(body) == 3 and len(tail) == 1 and all(b["count"] > 0 for b in body+tail)
        if model == "titantpp":
            assert digest(paths[0]) == row["summary_sha256"] and digest(paths[2]) == row["checkpoint_file_sha256"]
            assert state_hash == row["checkpoint_state_sha256"]
        else:
            assert summary["parameter_count"] == 100291
            assert summary["encoder_config"]["normalize_before"] is False
            assert summary["encoder_config"]["d_inner"] == 256
        assert before == {str(p.relative_to(ROOT)): digest(p) for p in paths}
        record["baselines"][model] = {"files_sha256": before, "checkpoint_state_sha256": state_hash,
            "parameter_count": summary["parameter_count"], "completed_epochs": len(history), "best_epoch": chosen["epoch"],
            "body_mae": sum(b["count"]*b["qty_mae"] for b in body)/sum(b["count"] for b in body),
            "gt_p99_mae": tail[0]["qty_mae"],
            **{k: summary[k] for k in ("best_val_qty_mae", "best_val_qty_rmse", "best_val_time_nll", "best_val_joint_objective")}}
        checks.append({"dataset": name, "model": model, "status": "passed", "files_unchanged": True})
    thp, hard = record["baselines"]["thp"], record["baselines"]["titantpp"]
    record["prospective_absolute_gate_limits"] = {
        "body_mae_max": min(thp["body_mae"], .95*hard["body_mae"]),
        "overall_mae_max": thp["best_val_qty_mae"],
        "overall_rmse_max": min(.98*thp["best_val_qty_rmse"], 1.02*hard["best_val_qty_rmse"]),
        "gt_p99_mae_max": min(.98*thp["gt_p99_mae"], 1.02*hard["gt_p99_mae"]),
        "time_nll_max": min(thp["best_val_time_nll"], hard["best_val_time_nll"])+.01}
    registry["datasets"].append(record)

if REGISTRY.exists():
    assert read(REGISTRY) == registry, "Frozen baseline registry changed; do not overwrite"
else:
    save(REGISTRY, registry)
save(HERE / "baseline_verification.json", {"status": "passed", "audited_at": datetime.now(timezone.utc).isoformat(),
    "base_code_revision": BASE_REVISION, "runtime": {"python": platform.python_version(), "torch": torch.__version__},
    "registry_sha256": digest(REGISTRY), "baselines": checks,
    "checks": ["file/state digests", "full train/validation target counts", "strict joint selector/patience",
               "finite metrics and checkpoint/optimizer tensors", "AdamW and legacy head", "weighted strata totals",
               "THP/Hard-LMM quantity and history stratum labels and counts identical",
               "historical/current CountAwareTHP and THPEncoderLayer AST equality"],
    "scope": "local metadata/hash/CPU tensor audit; no model creation, forward, optimizer step, dataset materialization, server access or test evaluation",
    "reuse": "eligible for historical validation comparison subject to unchanged implementation/training contract and prelaunch re-audit",
    "caveats": ["Historical split_rows includes test counts; absence of test evaluation does not prove absence of test-row materialization.",
                "Candidate must filter test before materialization and match train/validation counts and strata.",
                "Single seed and historical runtime; no matched-runtime/bitwise replay or statistical superiority claim."]})
print(json.dumps({"status": "passed", "registry_sha256": digest(REGISTRY), "baselines": checks,
                  "limits": {r["dataset"]:r["prospective_absolute_gate_limits"] for r in registry["datasets"]}}, indent=2))
