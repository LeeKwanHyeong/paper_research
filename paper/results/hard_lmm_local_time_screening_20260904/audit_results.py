"""Read-only CPU artifact audit; generate evidence without inference or training."""

import ast
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import shutil
import statistics
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from paper.scripts import run_hard_lmm_local_time_smoke as audit
from paper.scripts.run_hard_lmm_local_time_screening import compare, TRAINING_REVISION
from simple_lab_test.search.common.runner import canonical_state_dict_sha256
import torch

RAW = ROOT / "search_artifacts/hard_lmm_local_time_seed42_e300_20260904"
ORCHESTRATION = "484a052cc820c111dcd514e90ee294bcba59cacb"


def inventory(root):
    return {str(p.relative_to(root)): audit.digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


def close(actual, expected):
    assert math.isclose(float(actual), float(expected), rel_tol=1e-10, abs_tol=1e-8), (actual, expected)


def tensor_finite(value):
    if isinstance(value, torch.Tensor):
        assert bool(torch.isfinite(value).all())
    elif isinstance(value, dict):
        for item in value.values():
            tensor_finite(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensor_finite(item)
    else:
        audit.finite(value)


def csv_rows(path):
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows
    for row in rows:
        assert None not in row and all(v not in (None, "") for v in row.values())
        for v in row.values():
            try:
                number = float(v)
            except ValueError:
                continue
            assert math.isfinite(number), (path, row)
    return rows


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_evidence(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    assert audit.digest(source) == audit.digest(target)


before = inventory(RAW)
status = audit.read(RAW / "status.json")
assert status["status"] == "complete" and len(status["completed"]) == 2
assert status["training_revision"] == TRAINING_REVISION
assert status["orchestration_revision"] == ORCHESTRATION
assert not status["held_out_test_evaluated"] and not status["automatic_resume"]
observation = audit.read(HERE / "final_observation_5080.json")
assert observation["status"] == status
manifest_results = {}
for kind, revision, expected in (
    ("training", TRAINING_REVISION, "54b913e67cdf7042e20754fb9a20ebf369ae7cef7965c825db6cade20b6b31bd"),
    ("orchestration", ORCHESTRATION, "b6cd107dfe2512ea98b447f4d9ae85a9fca4872c82266e5ca50c288ced0e98b1"),
):
    path = RAW / f"{kind}_source_manifest.json"
    assert audit.digest(path) == expected
    manifest = audit.read(path)
    assert manifest["source_revision"] == revision
    for name, wanted in manifest["files"].items():
        blob = subprocess.check_output(["git", "show", f"{revision}:{name}"], cwd=ROOT)
        assert hashlib.sha256(blob).hexdigest() == wanted, name
    manifest_results[kind] = {"revision": revision, "manifest_sha256": expected,
                              "committed_file_hashes_verified": len(manifest["files"])}
contract = audit.read(audit.CONTRACT)
assert audit.digest(audit.CONTRACT) == audit.CONTRACT_SHA256
assert audit.read(RAW / "frozen_contract.json") == contract
registry = ROOT / contract["baseline"]["registry"]
assert audit.digest(registry) == contract["baseline"]["registry_sha256"]
rows = {r["dataset"]: r for r in audit.read(registry)["datasets"]}
assert audit.read(RAW / "baseline_registry.json") == [rows[d] for d in audit.DATASETS]
specs = {r["dataset"]: r for r in contract["datasets"]}
for path in RAW.rglob("*.json"):
    audit.finite(audit.read(path))
assert not [name for name in before if re.search(r"(^|/)(test_|held_out|heldout)", name)]

decisions, checks, timings, summaries, histories, strata, metric_rows = {}, {}, {}, {}, {}, [], []
for dataset in audit.DATASETS:
    output = RAW / dataset
    row = rows[dataset]
    baseline_contract, reference = audit.baseline(row, ROOT)
    summary = audit.read(output / audit.SUMMARY)
    launch = audit.read(output / "launch_contract.json")
    history = audit.read((output / audit.SUMMARY).parent / "history.json")["history"]
    log = (RAW / f"{dataset}.log").read_text()
    assert "[complete]" in log and "[early-stop]" in log
    assert not re.search(r"Traceback|OutOfMemory|out of memory|NVRM: Xid|\bnan\b|\binf\b", log, re.I)
    assert [int(x) for x in re.findall(r"^\[epoch (\d+)\]", log, re.M)] == list(range(1, len(history) + 1))
    result = audit.audit_run(output, row, reference, TRAINING_REVISION, specs[dataset],
                             screening=True, write_audit=False)
    assert result == audit.read(output / "audit.json")
    assert result == next(r for r in status["completed"] if r["dataset"] == dataset)
    assert all(r["train_all_finite"] for r in history)
    selected = history[summary["best_epoch"] - 1]
    for key, value in summary.items():
        if key.startswith("best_val_"):
            close(value, selected[key.removeprefix("best_")])
    best_path = (output / audit.SUMMARY).parent / "best_val_joint_objective_model.pt"
    for path in (best_path, best_path.with_name("last_epoch_state.pt")):
        tensor_finite(torch.load(path, map_location="cpu", weights_only=False))
    original_dir = ROOT / row["artifact_dir"]
    original_run = original_dir / "runs/titantpp/count_only_log_regression/seed_42"
    original_checkpoint = torch.load(original_run / "best_val_joint_objective_model.pt",
                                     map_location="cpu", weights_only=False)
    assert canonical_state_dict_sha256(original_checkpoint["model_state_dict"]) == row["checkpoint_state_sha256"]
    for kind in ("quantity", "history"):
        expected = summary[f"{kind}_rows"]
        seed_rows = csv_rows(output / f"{kind}_seed_metrics.csv")
        aggregates = csv_rows(output / f"{kind}_summary.csv")
        assert len(expected) == len(seed_rows) == len(aggregates)
        for saved, aggregated, actual, original in zip(seed_rows, aggregates, expected, reference[f"{kind}_rows"]):
            for key, value in actual.items():
                if isinstance(value, (int, float)):
                    close(saved[key], value)
                else:
                    assert saved[key] == value
            for key, value in aggregated.items():
                if key.endswith("_mean"):
                    close(value, actual[key[:-5]])
                elif key.endswith("_std"):
                    close(value, 0.)
                elif key == "n_seeds":
                    assert int(value) == 1
                elif isinstance(actual[key], (int, float)):
                    close(value, actual[key])
                else:
                    assert value == actual[key]
            assert actual["stratum"] == original["stratum"] and actual["count"] == original["count"]
            strata.append({"dataset": dataset, "axis": kind, "stratum": actual["stratum"],
                "count": actual["count"], "label": actual["stratum_label"],
                "baseline_mae": original["qty_mae"], "candidate_mae": actual["qty_mae"],
                "baseline_rmse": original["qty_rmse"], "candidate_rmse": actual["qty_rmse"],
                "mae_relative_change": actual["qty_mae"] / original["qty_mae"] - 1})
        total = sum(r["count"] for r in expected)
        for metric in ("qty_mae", "time_nll", "joint_objective", "log_qty_mse"):
            close(sum(r["count"] * r[metric] for r in expected) / total, summary[f"best_val_{metric}"])
        close(math.sqrt(sum(r["count"] * r["qty_rmse"] ** 2 for r in expected) / total), summary["best_val_qty_rmse"])
    run_csv = csv_rows(output / "run_summaries.csv")
    assert len(run_csv) == 1
    for key, value in run_csv[0].items():
        expected = summary[key]
        if isinstance(expected, (dict, list, bool)):
            assert ast.literal_eval(value) == expected
        elif isinstance(expected, (int, float)):
            close(value, expected)
        else:
            assert value == expected
    decisions[dataset] = compare(reference, summary, contract["performance_gate"])
    assert decisions[dataset] == audit.read(RAW / "comparison.json")[dataset]
    for metric, value in decisions[dataset]["candidate"].items():
        original = decisions[dataset]["baseline"][metric]
        metric_rows.append({"dataset": dataset, "metric": metric, "baseline": original, "candidate": value,
                           "absolute_change": value - original,
                           "relative_change": value / original - 1})
    telemetry = audit.read(RAW / f"{dataset}_epoch_timing.json")
    resources = audit.read(RAW / f"{dataset}_resources.json")
    assert resources["status"] == "complete"
    assert [r["epoch"] for r in telemetry] == list(range(1, len(history) + 1))
    assert all(r["observed_seconds"] > 0 for r in telemetry)
    assert resources["elapsed_seconds_including_startup"] >= telemetry[-1]["elapsed_seconds"]
    timings[dataset] = {"completed_epochs": len(history), "best_epoch": summary["best_epoch"],
        "trainer_seconds": summary["elapsed_seconds"], **resources,
        "mean_epoch_seconds_excluding_first": statistics.mean(r["observed_seconds"] for r in telemetry[1:]),
        "recent_10_epoch_seconds_mean": statistics.mean(r["observed_seconds"] for r in telemetry[-10:]),
        "peak_allocated_mib": resources["peak_allocated_bytes"] / 1024 ** 2,
        "peak_reserved_mib": resources["peak_reserved_bytes"] / 1024 ** 2,
        "compile_cost_separately_measured": False, "matched_runtime_baseline_cost_available": False}
    checks[dataset] = {**result, "csv_and_best_history_reconciled": True,
        "weighted_strata_reconciled": True, "optimizer_tensors_finite": True,
        "performance_acceptance": decisions[dataset]["status"]}
    summaries[dataset], histories[dataset] = (reference, summary), history
    for name in ("summary.json", "history.json"):
        copy_evidence(original_run / name, HERE / "baseline_evidence" / dataset / name)
    copy_evidence(original_dir / "launch_contract.json", HERE / "baseline_evidence" / dataset / "launch_contract.json")

assert inventory(RAW) == before
assert observation["comparison"] == decisions
assert status["candidate_decision"] == ("eligible_for_separate_followup" if all(
    d["status"] == "passed" for d in decisions.values()) else "hold")
for name in before:
    if not name.endswith(".pt"):
        copy_evidence(RAW / name, HERE / "source_5080" / name)
audit.save(HERE / "artifact_inventory.json", before)
audit.save(HERE / "exact_comparison.json", decisions)
audit.save(HERE / "costs.json", timings)
write_csv(HERE / "official_metrics.csv", metric_rows)
write_csv(HERE / "scale_wise_comparison.csv", strata)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
for i, dataset in enumerate(audit.DATASETS):
    reference, summary = summaries[dataset]
    history = histories[dataset]
    label = "Taxi" if i == 0 else "RAF"
    for j, (key, ylabel) in enumerate((("joint_objective", "Validation joint objective"), ("qty_mae", "Validation quantity MAE"))):
        ax = axes[i, j]
        ax.plot([r["epoch"] for r in history], [r[f"val_{key}"] for r in history], color="#166b6b")
        ax.axhline(reference[f"best_val_{key}"], color="#9b521c", linestyle="--", label="Historical Hard-LMM selected")
        ax.axvline(summary["best_epoch"], color="#555555", linestyle=":", label=f"Joint-selected e{summary['best_epoch']}")
        ax.set(xlabel="Epoch", ylabel=ylabel, title=label)
        ax.legend(fontsize=8)
    ax = axes[i, 2]
    changes = [decisions[dataset]["relative_changes"][m] * 100 for m in ("body_mae", "qty_mae", "qty_rmse", "gt_p99_mae")]
    ax.barh(["Body MAE", "Overall MAE", "Overall RMSE", ">p99 MAE"], changes, color="#166b6b")
    ax.axvline(0, color="black", linewidth=.8)
    ax.set(xlim=(-13, 3), xlabel="Change from baseline (%)", title=f"{label}: selected checkpoint only")
    for j, value in enumerate(changes):
        inside = value < -10
        ax.text(value + .25 if inside or value >= 0 else value - .2, j, f"{value:+.3f}%",
                va="center", ha="left" if inside or value >= 0 else "right",
                color="white" if inside else "black", fontsize=9)
fig.suptitle("Local-time routing, seed 42 validation: 0/2 fixed-gate passes\nHistorical baseline; no held-out evaluation or post-hoc checkpoint replacement", fontsize=13)
fig.savefig(HERE / "validation_comparison.png", dpi=160)
plt.close(fig)
audit.save(HERE / "final_verification.json", {
    "status": "passed", "assessment": "share_with_caveats", "audited_at": datetime.now(timezone.utc).isoformat(),
    "scope": "CPU metadata/checkpoint audit only; no forward pass, optimizer step or dataset materialization",
    "runtime": {"python": platform.python_version(), "torch": torch.__version__},
    "raw_artifact": str(RAW.relative_to(ROOT)), "raw_files_unchanged": True,
    "source_integrity": manifest_results, "contract_sha256": audit.digest(audit.CONTRACT),
    "dataset_runs": checks, "candidate_decision": status["candidate_decision"],
    "held_out_test_evaluated": False, "held_out_artifacts": [], "all_json_csv_numeric_metrics_finite": True,
    "total_wall_seconds": (datetime.fromisoformat(status["completed_at"]) - datetime.fromisoformat(status["started_at"])).total_seconds(),
    "caveats": ["One seed, validation only; no statistical significance or general superiority claim.",
        "Historical CUDA baseline is not a bitwise or matched-runtime replay.",
        "Training path changes shared encoder gradients; fixed-weight quantity invariance does not imply post-training invariance.",
        "Only recorded epoch/startup/peak memory costs; no separately instrumented compile cost or matched cost ratio."]})
print(json.dumps({"audit": "passed", "decisions": decisions, "costs": timings}, indent=2))
