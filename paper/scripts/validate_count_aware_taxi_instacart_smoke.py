#!/usr/bin/env python3
"""Validate the Taxi and Instacart mark-free CUDA smoke artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DATASETS = {
    "yellow_trip_hourly": {"lookback": 168, "max_seq_len": 256},
    "insta_market_basket": {"lookback": 52, "max_seq_len": 64},
}
ROLES = {
    "t0_common_control": {
        "backbones": {"rmtpp", "thp", "nhp", "sahp", "titantpp"},
        "variant": "count_only_log_regression",
        "lambda_tail": 0.0,
    },
    "t1_incumbent": {
        "backbones": {"titantpp"},
        "variant": "count_only_log_mse_tail_shared",
        "lambda_tail": 0.09111380335463036,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-source-revision", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_row(row: dict[str, str]) -> bool:
    metric_names = (
        "best_val_joint_objective",
        "best_val_time_nll",
        "best_val_log_qty_mse",
        "best_val_qty_mae",
        "best_val_qty_rmse",
    )
    return all(math.isfinite(float(row[name])) for name in metric_names)


def main() -> None:
    args = parse_args()
    if len(args.expected_source_revision) != 40:
        raise ValueError("Expected source revision must be a 40-character Git SHA")
    root = args.artifact_root
    status = load_json(root / "status.json")
    cuda = load_json(root / "cuda_model_test.json")
    checks: list[dict[str, Any]] = []

    checks.append({
        "name": "launcher_status",
        "passed": status.get("status") == "complete"
        and status.get("source_revision") == args.expected_source_revision
        and status.get("execution_server") == "5080"
        and status.get("held_out_test_evaluated") is False,
    })
    cuda_results = cuda.get("results", [])
    checks.append({
        "name": "cuda_model_test",
        "passed": cuda.get("status") == "complete"
        and cuda.get("device") == "cuda"
        and cuda.get("case_count") == 6
        and len(cuda_results) == 6
        and all(row.get("finite") is True for row in cuda_results),
    })

    role_details: list[dict[str, Any]] = []
    for dataset, dataset_contract in DATASETS.items():
        for role, role_contract in ROLES.items():
            role_root = root / dataset / role
            launch = load_json(role_root / "launch_contract.json")
            summaries = load_csv(role_root / "run_summaries.csv")
            run_keys = {(row["backbone"], row["variant"], int(row["seed"])) for row in summaries}
            expected_keys = {
                (backbone, role_contract["variant"], 42)
                for backbone in role_contract["backbones"]
            }
            histories_ok = True
            for row in summaries:
                candidates = list(role_root.glob(
                    f"**/{row['backbone']}/{row['variant']}/seed_{row['seed']}/history.json"
                ))
                if len(candidates) != 1:
                    histories_ok = False
                    continue
                history = load_json(candidates[0]).get("history", [])
                histories_ok = histories_ok and len(history) == 1
                histories_ok = histories_ok and all(
                    math.isfinite(float(value))
                    for value in history[0].values()
                    if isinstance(value, (int, float))
                )
            no_test_artifacts = not any(
                "test" in path.name.lower()
                for path in role_root.rglob("*")
                if path.is_file()
            )
            passed = all(
                [
                    launch.get("status") == "complete",
                    launch.get("source_revision") == args.expected_source_revision,
                    launch.get("dataset") == dataset,
                    launch.get("model_role") == role,
                    launch.get("evaluation_scope") == "validation_only",
                    launch.get("held_out_test_evaluated") is False,
                    launch.get("partial_smoke") is True,
                    launch.get("epochs") == 1,
                    launch.get("batch_size") == 128,
                    math.isclose(float(launch.get("lr")), 0.001),
                    launch.get("max_series") == 20,
                    launch.get("seeds") == [42],
                    launch.get("quantity_variants") == [role_contract["variant"]],
                    launch.get("lookback_weeks") == dataset_contract["lookback"],
                    launch.get("max_seq_len") == dataset_contract["max_seq_len"],
                    launch.get("hidden_dim") == 64,
                    launch.get("time_head", {}).get("mode") == "legacy_clamped_rmtpp",
                    math.isclose(float(launch.get("lambda_tail")), role_contract["lambda_tail"]),
                    run_keys == expected_keys,
                    all(row.get("status") == "success" for row in summaries),
                    all(row.get("evaluation_scope") == "validation_only" for row in summaries),
                    all(row.get("held_out_test_evaluated", "").lower() == "false" for row in summaries),
                    all(finite_row(row) for row in summaries),
                    histories_ok,
                    no_test_artifacts,
                ]
            )
            role_details.append({
                "dataset": dataset,
                "role": role,
                "passed": passed,
                "run_count": len(summaries),
                "histories_ok": histories_ok,
                "held_out_test_artifacts_absent": no_test_artifacts,
                "scale_wise_metrics_expected": False,
                "scale_wise_metrics_reason": "partial validation batches",
            })
    checks.append({"name": "dataset_role_artifacts", "passed": all(row["passed"] for row in role_details)})

    passed = all(check["passed"] for check in checks)
    report = {
        "status": "pass" if passed else "fail",
        "artifact_root": str(root),
        "source_revision": args.expected_source_revision,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "checks": checks,
        "dataset_roles": role_details,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Taxi·Instacart mark-free CUDA smoke 검증",
        "",
        f"- 최종 판정: **{report['status'].upper()}**",
        "- 평가 범위: validation-only",
        "- held-out test: 사용하지 않음",
        "",
        "| Dataset | Role | Runs | 판정 |",
        "| --- | --- | ---: | --- |",
    ]
    lines.extend(
        f"| {row['dataset']} | {row['role']} | {row['run_count']} | "
        f"{'PASS' if row['passed'] else 'FAIL'} |"
        for row in role_details
    )
    (args.output_dir / "validation_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
