#!/usr/bin/env python3
"""Run one host-bound Hard-LMM candidate through e1 and gated seed42 screening."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "paper/contracts/hard_lmm_backbone_parallel_screening_v1.json"
TRAINER = ROOT / "paper/scripts/run_count_aware_tpp_backbone_control.py"
DATASETS = {
    "intermittent_frozen_5000": {
        "data": "sample_data/intermittent_v2/intermittent_frozen_5000_with_split.parquet",
        "manifest": "sample_data/intermittent_v2/intermittent_frozen_5000_split_manifest.json",
        "lookback": 520,
        "max_seq_len": 256,
        "train_targets": 393824,
        "validation_targets": 86285,
    },
    "yellow_trip_hourly": {
        "data": "sample_data/new_york_taxi/yellow_trip_hourly_with_split.parquet",
        "manifest": "sample_data/new_york_taxi/yellow_trip_hourly_split_manifest.json",
        "lookback": 168,
        "max_seq_len": 256,
        "train_targets": 38393,
        "validation_targets": 8268,
    },
    "insta_market_basket": {
        "data": "sample_data/insta_market_basket/instacart_marked_target_with_split.parquet",
        "manifest": "sample_data/insta_market_basket/instacart_marked_target_split_manifest.json",
        "lookback": 52,
        "max_seq_len": 64,
        "train_targets": 1991192,
        "validation_targets": 503733,
    },
}


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def gpu_preflight(expected_name: str) -> dict[str, Any]:
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()
    fields = [field.strip() for field in query.split(",")]
    require(len(fields) == 4, f"Unexpected nvidia-smi output: {query}")
    require(expected_name in fields[0], f"Expected {expected_name}, found {fields[0]}")
    processes = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip().splitlines()
    require(not processes, f"GPU already has compute processes: {processes}")
    require(int(fields[2]) >= 12000, f"Insufficient free VRAM: {fields[2]} MiB")
    return {
        "gpu_name": fields[0],
        "total_vram_mib": int(fields[1]),
        "free_vram_mib": int(fields[2]),
        "utilization_percent": int(fields[3]),
        "compute_processes": processes,
    }


def job_command(
    *,
    python: str,
    candidate: dict[str, Any],
    dataset: str,
    output: Path,
    source_revision: str,
    phase: dict[str, Any],
    host_role: str,
) -> list[str]:
    spec = DATASETS[dataset]
    return [
        python,
        "-s",
        str(TRAINER),
        "--data",
        str(ROOT / spec["data"]),
        "--split-manifest",
        str(ROOT / spec["manifest"]),
        "--output-dir",
        str(output),
        "--source-revision",
        source_revision,
        "--execution-role",
        f"hard_lmm_backbone_{host_role}_{candidate['backbone']}_{dataset}_{phase['epochs']}e",
        "--dataset-contract",
        dataset,
        "--model-role",
        candidate["model_role"],
        "--device",
        "cuda",
        "--epochs",
        str(phase["epochs"]),
        "--min-epochs",
        str(phase["minimum_epochs"]),
        "--early-stopping-patience",
        str(phase["patience"]),
        "--batch-size",
        "128",
        "--lr",
        "0.001",
        "--lookback-weeks",
        str(spec["lookback"]),
        "--max-seq-len",
        str(spec["max_seq_len"]),
        "--hidden-dim",
        "64",
        "--lambda-log-qty",
        "1",
        "--lambda-tail",
        "0",
        "--grad-clip",
        "1",
        "--backbones",
        candidate["backbone"],
        "--seeds",
        "42",
        "--quantity-variants",
        "log_mse",
        "--checkpoint-monitor",
        "validation_raw_quantity_rmse",
        "--quantile-adaptive-strength",
        "0",
        "--time-head-mode",
        "legacy_clamped_rmtpp",
        "--time-scale",
        "3",
        "--time-w-max",
        "3.3333333333333335",
        "--time-intercept-limit",
        "300",
        "--allow-partial-contract",
    ]


def audit_job(
    output: Path,
    *,
    candidate: dict[str, Any],
    dataset: str,
    source_revision: str,
    expected_epochs: int,
) -> dict[str, Any]:
    summary_path = (
        output
        / "runs"
        / candidate["backbone"]
        / "count_only_log_regression"
        / "seed_42"
        / "summary.json"
    )
    summary = read_json(summary_path)
    launch = read_json(output / "launch_contract.json")
    require(summary.get("status") == "success", "Training summary is incomplete")
    require(summary.get("backbone") == candidate["backbone"], "Backbone identity drift")
    require(summary.get("source_revision") == source_revision, "Source revision drift")
    require(summary.get("evaluation_scope") == "validation_only", "Scope drift")
    require(summary.get("held_out_test_evaluated") is False, "Held-out test was evaluated")
    require(summary.get("checkpoint_monitor") == "validation_raw_quantity_rmse", "Selector drift")
    require(summary.get("epochs") == expected_epochs, "Epoch budget drift")
    require(str(summary.get("training_device", "")).startswith("cuda"), "CUDA was not used")
    require(int(summary.get("cuda_peak_memory_allocated_bytes", 0)) > 0, "CUDA allocation missing")
    require(launch.get("status") == "complete", "Launch contract is incomplete")
    population = launch.get("validation_target_population", {})
    require(
        population.get("target_count") == DATASETS[dataset]["validation_targets"],
        "Validation target population drift",
    )
    history = read_json(summary_path.parent / "history.json").get("history")
    require(isinstance(history, list) and len(history) == summary["completed_epochs"], "History drift")
    require(
        all(row.get("train_event_count") == DATASETS[dataset]["train_targets"] for row in history),
        "An epoch did not process the full train population",
    )
    selected = min(history, key=lambda row: (float(row["val_qty_rmse"]), int(row["epoch"])))
    require(selected["epoch"] == summary["best_epoch"], "Earliest raw-RMSE minimum drift")
    rows = summary.get("quantity_rows")
    require(isinstance(rows, list) and len(rows) == 5, "Quantity strata missing")
    body = [row for row in rows if row["stratum"] != "gt_p99"]
    body_count = sum(int(row["count"]) for row in body)
    body_mae = sum(int(row["count"]) * float(row["qty_mae"]) for row in body) / body_count
    tail = next(row for row in rows if row["stratum"] == "gt_p99")
    metrics = {
        "raw_rmse": float(summary["best_val_qty_rmse"]),
        "overall_mae": float(summary["best_val_qty_mae"]),
        "body_mae": body_mae,
        "gt_p99_mae": float(tail["qty_mae"]),
        "clamped_time_loss": float(summary["best_val_time_nll"]),
    }
    require(all(math.isfinite(value) for value in metrics.values()), "Nonfinite metric")
    return {
        "status": "passed",
        "summary": str(summary_path),
        "summary_sha256": sha256(summary_path),
        "checkpoint_sha256": sha256(summary_path.parent / "best_val_qty_rmse_model.pt"),
        "best_epoch": int(summary["best_epoch"]),
        "completed_epochs": int(summary["completed_epochs"]),
        "metrics": metrics,
        "cuda_peak_memory_allocated_bytes": int(summary["cuda_peak_memory_allocated_bytes"]),
        "validation_target_identity_sha256": population.get("target_identity_sha256"),
        "validation_target_quantity_sha256": population.get("target_quantity_sha256"),
        "held_out_test_evaluated": False,
    }


def evaluate_gate(metrics: dict[str, float], baseline: dict[str, float]) -> dict[str, Any]:
    checks = {
        "raw_rmse": metrics["raw_rmse"] < baseline["raw_rmse"],
        "overall_mae": metrics["overall_mae"] <= baseline["overall_mae"] * 1.01,
        "body_mae": metrics["body_mae"] <= baseline["body_mae"] * 1.02,
        "gt_p99_mae": metrics["gt_p99_mae"] <= baseline["gt_p99_mae"] * 1.02,
        "clamped_time_loss": metrics["clamped_time_loss"] <= baseline["clamped_time_loss"] + 0.01,
    }
    return {"status": "passed" if all(checks.values()) else "failed", "checks": checks}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host-role", choices=("5080", "5090"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    require(len(args.source_revision) == 40, "A full source revision is required")
    contract = read_json(CONTRACT_PATH)
    candidate = contract["candidates"][args.host_role]
    baseline_path = ROOT / contract["B_validation_reference"]["source"]
    require(
        sha256(baseline_path) == contract["B_validation_reference"]["sha256"],
        "B reference evidence hash drift",
    )
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    status_path = output_root / "campaign_status.json"
    status: dict[str, Any] = {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "contract_sha256": sha256(CONTRACT_PATH),
        "source_revision": args.source_revision,
        "host_role": args.host_role,
        "candidate": candidate,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "gpu_preflight": gpu_preflight(candidate["gpu_name_contains"]),
        "jobs": {},
        "status": "running_e1",
    }
    save_json(status_path, status)
    try:
        for phase_name in ("e1", "seed42_screening"):
            phase = contract["phases"][phase_name]
            status["status"] = f"running_{phase_name}"
            save_json(status_path, status)
            for dataset in phase["datasets_in_order"]:
                job_id = f"{phase_name}_{dataset}"
                output = output_root / "jobs" / job_id
                output.mkdir(parents=True, exist_ok=True)
                command = job_command(
                    python=args.python,
                    candidate=candidate,
                    dataset=dataset,
                    output=output,
                    source_revision=args.source_revision,
                    phase=phase,
                    host_role=args.host_role,
                )
                with (output / "campaign_runner.log").open("a", encoding="utf-8") as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
                audit = audit_job(
                    output,
                    candidate=candidate,
                    dataset=dataset,
                    source_revision=args.source_revision,
                    expected_epochs=phase["epochs"],
                )
                if phase_name == "seed42_screening":
                    baseline = contract["B_validation_reference"]["datasets"][dataset]
                    audit["gate"] = evaluate_gate(audit["metrics"], baseline)
                status["jobs"][job_id] = audit
                save_json(status_path, status)
                if phase_name == "seed42_screening" and audit["gate"]["status"] == "failed":
                    status["status"] = "stopped_seed42_gate_failed"
                    status["failed_dataset"] = dataset
                    save_json(status_path, status)
                    return
        status["status"] = "seed42_all_datasets_passed"
        save_json(status_path, status)
    except BaseException as error:
        status["status"] = "failed_execution"
        status["error"] = repr(error)
        save_json(status_path, status)
        raise


if __name__ == "__main__":
    main()
