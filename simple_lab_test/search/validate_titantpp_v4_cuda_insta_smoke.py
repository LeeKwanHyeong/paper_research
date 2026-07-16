from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


EXPECTED_VARIANTS = {
    "v4a_shared_value_mark_time": {
        "value_head_mode": "shared",
        "qty_mark_gradient_mode": "coupled",
    },
    "v4b_mark_value_mark_time": {
        "value_head_mode": "mark_conditioned_experts",
        "qty_mark_gradient_mode": "detached",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def one_path(paths: list[Path], label: str) -> Path:
    if len(paths) != 1:
        raise ValueError(f"Expected one {label}, found {len(paths)}: {paths}")
    return paths[0]


def model_test_row(root: Path, variant: str) -> dict[str, Any]:
    path = root / "cuda_model_test" / variant / "model_test_summary.json"
    payload = read_json(path)
    rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or len(rows) != 1:
        raise ValueError(f"Unexpected model-test payload: {path}")
    return rows[0]


def validate_status_table(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    expected = {
        *(f"model_test:{name}" for name in EXPECTED_VARIANTS),
        *(f"insta_smoke:{name}" for name in EXPECTED_VARIANTS),
    }
    observed = {row["stage_variant"] for row in rows}
    if observed != expected:
        raise ValueError(f"Status variants mismatch: expected={expected} observed={observed}")
    if any(int(row["exit_code"]) != 0 for row in rows):
        raise ValueError(f"Non-zero variant status: {rows}")
    return rows


def validate_variant(root: Path, variant: str, contract: dict[str, str]) -> dict[str, Any]:
    model_test = model_test_row(root, variant)
    for key, expected in {
        **contract,
        "time_head_mode": "mark_conditioned",
        "status": "success",
    }.items():
        if model_test.get(key) != expected:
            raise ValueError(
                f"Model-test identity mismatch for {variant}: {key}={model_test.get(key)!r}"
            )
    for key in ("nll", "nll_marker", "nll_time", "total_loss", "dt_hat_mean"):
        if not math.isfinite(float(model_test[key])):
            raise ValueError(f"Non-finite model-test metric for {variant}: {key}")

    smoke_root = root / "insta_smoke" / variant
    summary_path = one_path(
        list(smoke_root.glob("runs/**/metrics/summary.json")),
        f"{variant} smoke summary",
    )
    manifest_path = one_path(
        list(smoke_root.glob("runs/**/manifest/run_config.json")),
        f"{variant} run manifest",
    )
    history_path = one_path(
        list(smoke_root.glob("runs/**/metrics/history.json")),
        f"{variant} history",
    )
    summary = read_json(summary_path)
    manifest = read_json(manifest_path)
    history = read_json(history_path).get("history", [])

    expected_summary = {
        **contract,
        "time_head_mode": "mark_conditioned",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "status": "success",
        "epochs": 1,
        "seed": 42,
    }
    for key, expected in expected_summary.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"Smoke summary mismatch for {variant}: {key}={summary.get(key)!r}"
            )
    if len(history) != 1 or int(history[0]["epoch"]) != 1:
        raise ValueError(f"Smoke history is not exactly one epoch for {variant}")

    loader_counts = manifest.get("loader_sample_counts", {})
    if loader_counts != {"train": 1380, "validation": 300, "test": 300}:
        raise ValueError(f"Unexpected Instacart loader counts for {variant}: {loader_counts}")
    if manifest.get("held_out_test_evaluation_enabled") is not False:
        raise ValueError(f"Held-out evaluation was enabled for {variant}")

    run_dir = summary_path.parents[1]
    test_artifacts = sorted(run_dir.glob("metrics/test_*"))
    if test_artifacts:
        raise ValueError(f"Held-out test artifacts exist for {variant}: {test_artifacts}")

    finite_metrics = {
        "best_val_nll": float(summary["best_val_nll"]),
        "best_val_nll_time": float(history[0]["val_nll_time"]),
        "best_val_nll_marker": float(history[0]["val_nll_marker"]),
        "best_val_nll_qty_mae": float(summary["best_val_nll_qty_mae"]),
    }
    if not all(math.isfinite(value) for value in finite_metrics.values()):
        raise ValueError(f"Non-finite smoke metrics for {variant}: {finite_metrics}")

    return {
        "model_test": {
            "nll": float(model_test["nll"]),
            "nll_time": float(model_test["nll_time"]),
            "parameter_count": int(model_test["parameter_count"]),
        },
        "insta_smoke": {
            **finite_metrics,
            "run_dir": str(run_dir),
            "held_out_test_evaluated": False,
        },
    }


def main() -> None:
    args = parse_args()
    root = args.artifact_root.expanduser().resolve()
    status_rows = validate_status_table(root / "variant_status.tsv")
    variants = {
        variant: validate_variant(root, variant, contract)
        for variant, contract in EXPECTED_VARIANTS.items()
    }
    result = {
        "status": "PASS",
        "decision": "proceed_to_taxi_v4_2x2_validation_only_screening",
        "held_out_test_evaluated": False,
        "status_rows": status_rows,
        "variants": variants,
    }
    output_path = root / "integration_summary.json"
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
