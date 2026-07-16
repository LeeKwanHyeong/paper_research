from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


VARIANT_CONTRACT = {
    "v2_shared_value_shared_time": {
        "value_head_mode": "shared",
        "qty_mark_gradient_mode": "coupled",
        "time_head_mode": "shared",
    },
    "v3b_mark_value_shared_time": {
        "value_head_mode": "mark_conditioned_experts",
        "qty_mark_gradient_mode": "detached",
        "time_head_mode": "shared",
    },
    "v4a_shared_value_mark_time": {
        "value_head_mode": "shared",
        "qty_mark_gradient_mode": "coupled",
        "time_head_mode": "mark_conditioned",
    },
    "v4b_mark_value_mark_time": {
        "value_head_mode": "mark_conditioned_experts",
        "qty_mark_gradient_mode": "detached",
        "time_head_mode": "mark_conditioned",
    },
}

PAIR_CONTRACT = {
    "v4a_vs_v2": ("v2_shared_value_shared_time", "v4a_shared_value_mark_time"),
    "v4b_vs_v3b": ("v3b_mark_value_shared_time", "v4b_mark_value_mark_time"),
}

THRESHOLDS = {
    "min_time_nll_improvement_pct": 0.5,
    "max_total_nll_regression_pct": 0.5,
    "max_dt_mae_regression_pct": 1.0,
    "max_marker_nll_regression_pct": 2.0,
    "min_mark_accuracy_delta_pp": -0.25,
    "max_qty_mae_regression_pct": 5.0,
}
COMPARISON_EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def one_path(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected one {label}, found {len(paths)}: {paths}")
    return paths[0]


def lower_is_better_change_pct(candidate: float, control: float) -> float:
    if control == 0.0:
        raise ValueError("Control metric cannot be zero for relative comparison.")
    return 100.0 * (candidate - control) / abs(control)


def evaluate_pair(
    control: dict[str, float],
    candidate: dict[str, float],
) -> dict[str, Any]:
    changes = {
        "time_nll_improvement_pct": -lower_is_better_change_pct(
            candidate["val_nll_time"], control["val_nll_time"]
        ),
        "total_nll_regression_pct": lower_is_better_change_pct(
            candidate["val_nll"], control["val_nll"]
        ),
        "dt_mae_regression_pct": lower_is_better_change_pct(
            candidate["dt_mae"], control["dt_mae"]
        ),
        "marker_nll_regression_pct": lower_is_better_change_pct(
            candidate["val_nll_marker"], control["val_nll_marker"]
        ),
        "mark_accuracy_delta_pp": 100.0
        * (candidate["mark_acc"] - control["mark_acc"]),
        "qty_mae_regression_pct": lower_is_better_change_pct(
            candidate["qty_mae"], control["qty_mae"]
        ),
    }
    checks = {
        "time_nll_improvement": changes["time_nll_improvement_pct"]
        >= THRESHOLDS["min_time_nll_improvement_pct"] - COMPARISON_EPS,
        "total_nll_safety": changes["total_nll_regression_pct"]
        <= THRESHOLDS["max_total_nll_regression_pct"] + COMPARISON_EPS,
        "dt_mae_safety": changes["dt_mae_regression_pct"]
        <= THRESHOLDS["max_dt_mae_regression_pct"] + COMPARISON_EPS,
        "marker_nll_safety": changes["marker_nll_regression_pct"]
        <= THRESHOLDS["max_marker_nll_regression_pct"] + COMPARISON_EPS,
        "mark_accuracy_safety": changes["mark_accuracy_delta_pp"]
        >= THRESHOLDS["min_mark_accuracy_delta_pp"] - COMPARISON_EPS,
        "qty_mae_safety": changes["qty_mae_regression_pct"]
        <= THRESHOLDS["max_qty_mae_regression_pct"] + COMPARISON_EPS,
    }
    return {
        **changes,
        "checks": checks,
        "passed": all(checks.values()),
    }


def validate_status_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if {row["variant"] for row in rows} != set(VARIANT_CONTRACT):
        raise ValueError(f"Variant status mismatch: {rows}")
    if any(int(row["exit_code"]) != 0 for row in rows):
        raise ValueError(f"Non-zero variant exit: {rows}")
    return rows


def load_variant(
    root: Path,
    variant: str,
    contract: dict[str, str],
    source_revision: str,
) -> dict[str, Any]:
    variant_root = root / "variants" / variant
    summary_path = one_path(
        list(variant_root.glob("runs/**/metrics/summary.json")),
        f"{variant} summary",
    )
    history_path = one_path(
        list(variant_root.glob("runs/**/metrics/history.json")),
        f"{variant} history",
    )
    manifest_path = one_path(
        list(variant_root.glob("runs/**/manifest/run_config.json")),
        f"{variant} manifest",
    )
    summary = read_json(summary_path)
    history = read_json(history_path).get("history", [])
    manifest = read_json(manifest_path)

    expected = {
        **contract,
        "status": "success",
        "reproducibility_mode": "strict",
        "source_revision": source_revision,
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "epochs": 50,
        "seed": 42,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"{variant} identity mismatch: {key}={summary.get(key)!r}")

    if len(history) != 50 or int(history[-1]["epoch"]) != 50:
        raise ValueError(f"{variant} did not complete exactly 50 epochs")
    best_epoch = int(summary["best_val_nll_epoch"])
    best_rows = [row for row in history if int(row["epoch"]) == best_epoch]
    if len(best_rows) != 1:
        raise ValueError(f"{variant} best epoch missing from history: {best_epoch}")

    loader_counts = manifest.get("loader_sample_counts", {})
    if loader_counts != {"train": 38393, "validation": 8268, "test": 8327}:
        raise ValueError(f"{variant} loader counts mismatch: {loader_counts}")
    if manifest.get("held_out_test_evaluation_enabled") is not False:
        raise ValueError(f"Held-out evaluation was enabled for {variant}")

    run_dir = summary_path.parents[1]
    test_artifacts = sorted(run_dir.glob("metrics/test_*"))
    if test_artifacts:
        raise ValueError(f"Held-out artifacts exist for {variant}: {test_artifacts}")
    for name in (
        "scale_wise_best_val_nll.csv",
        "validation_mark_confusion_best_val_nll.csv",
        "validation_mark_class_metrics_best_val_nll.csv",
    ):
        if not (run_dir / "metrics" / name).exists():
            raise ValueError(f"Missing validation artifact for {variant}: {name}")

    row = best_rows[0]
    metrics = {
        "val_nll": float(row["val_nll"]),
        "val_nll_marker": float(row["val_nll_marker"]),
        "val_nll_time": float(row["val_nll_time"]),
        "dt_mae": float(row["dt_mae"]),
        "mark_acc": float(row["mark_acc"]),
        "qty_mae": float(row["qty_mae"]),
        "value_mae": float(row["value_mae"]),
        "score": float(row["score"]),
    }
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"Non-finite best validation metrics for {variant}: {metrics}")

    return {
        "best_epoch": best_epoch,
        "metrics": metrics,
        "run_dir": str(run_dir),
    }


def write_pairwise_csv(path: Path, pairs: dict[str, Any]) -> None:
    rows = []
    for pair_name, payload in pairs.items():
        rows.append(
            {
                "pair": pair_name,
                "control": payload["control"],
                "candidate": payload["candidate"],
                "passed": payload["gate"]["passed"],
                **{
                    key: value
                    for key, value in payload["gate"].items()
                    if key not in {"checks", "passed"}
                },
                **{
                    f"check_{key}": value
                    for key, value in payload["gate"]["checks"].items()
                },
            }
        )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root = args.artifact_root.expanduser().resolve()
    status_rows = validate_status_table(root / "variant_status.tsv")
    variants = {
        name: load_variant(root, name, contract, args.source_revision)
        for name, contract in VARIANT_CONTRACT.items()
    }
    pairs = {}
    for pair_name, (control_name, candidate_name) in PAIR_CONTRACT.items():
        pairs[pair_name] = {
            "control": control_name,
            "candidate": candidate_name,
            "gate": evaluate_pair(
                variants[control_name]["metrics"],
                variants[candidate_name]["metrics"],
            ),
        }

    v4a_passed = bool(pairs["v4a_vs_v2"]["gate"]["passed"])
    v4b_passed = bool(pairs["v4b_vs_v3b"]["gate"]["passed"])
    if v4b_passed:
        decision = "promote_v4b_to_strict_matched_multiseed_validation"
    elif v4a_passed:
        decision = "retain_v3b_for_taxi_and_record_v4a_attribution_only"
    else:
        decision = "retain_v2_and_taxi_v3b_do_not_promote_v4"

    result = {
        "status": "COMPLETED",
        "decision": decision,
        "held_out_test_evaluated": False,
        "source_revision": args.source_revision,
        "thresholds": THRESHOLDS,
        "status_rows": status_rows,
        "variants": variants,
        "pairs": pairs,
    }
    (root / "screening_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    write_pairwise_csv(root / "pairwise_gate.csv", pairs)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
