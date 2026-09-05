"""Frozen launch validation and metric definitions for the one prefix-read candidate."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = ROOT / "paper/contracts/titans_mac_prior_prefix_v1.json"
BACKBONE = "titantpp_titans_mac_prior_prefix"
BASELINE = "titantpp_titans_mac"
ROLE = "titans_mac_prior_prefix_screening"
CONTRACT_ID = "titans_mac_prior_prefix_v1"


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def load_contract(path: Path = CONTRACT_PATH) -> dict[str, Any]:
    contract = json.loads(path.read_text())
    if (contract["contract_id"] != CONTRACT_ID or contract["backbone"] != BACKBONE
            or contract["baseline"] != BASELINE or contract["model_role"] != ROLE):
        raise ValueError("Prior-prefix contract identity mismatch")
    if set(contract["datasets"]) != {
        "yellow_trip_hourly", "intermittent_frozen_5000", "insta_market_basket"
    }:
        raise ValueError("Prior-prefix dataset grid mismatch")
    if (contract["architecture"]["output_read_policy"] != "prior_prefix"
            or contract["architecture"]["segment_size"] != 16
            or contract["architecture"]["additional_parameters"] != 0):
        raise ValueError("Prior-prefix architecture mismatch")
    return contract


def validate_candidate_launch(args: Any) -> None:
    """Reject partial, stale, or altered production research training arguments."""
    contract = load_contract()
    frozen = contract["training"]
    expected = {
        "backbones": BACKBONE, "model_role": ROLE,
        "hidden_dim": 64, "batch_size": 128, "lr": .001,
        "lambda_log_qty": 1., "lambda_tail": 0., "grad_clip": 1.,
        "titans_memory_gradient_clip": 1.,
        "titans_mac_execution_backend": "optimized",
        "time_head_mode": frozen["time_head"],
        "time_scale": frozen["time_scale"],
        "time_w_max": frozen["time_w_max"],
        "time_intercept_limit": frozen["time_intercept_limit"],
        "time_wd_safety_limit": frozen["time_wd_safety_limit"],
        "time_head_lr_multiplier": 1.,
        "max_series": None, "max_train_batches": None, "max_val_batches": None,
        "force_rerun": False,
    }
    for key, value in expected.items():
        if getattr(args, key, None) != value:
            raise ValueError(f"Prior-prefix launch mismatch: {key} must be {value!r}")
    if args.quantity_variants not in {"log_mse", "count_only_log_regression"}:
        raise ValueError("Prior-prefix requires one direct log-MSE variant")
    if args.seeds not in {"42", "52", "62"}:
        raise ValueError("Prior-prefix launch requires one frozen seed")
    phase = {1: (1, 1), 300: (40, 40)}.get(args.epochs)
    if phase != (args.min_epochs, args.early_stopping_patience):
        raise ValueError("Prior-prefix supports full e1 or e300/min40/pat40 only")
    if args.epochs == 1 and args.seeds != "42":
        raise ValueError("Prior-prefix e1 requires seed42")
    dataset = contract["datasets"].get(args.dataset_contract)
    if dataset is None or (args.lookback_weeks, args.max_seq_len) != (
            dataset["lookback"], dataset["max_seq_len"]):
        raise ValueError("Prior-prefix dataset/context mismatch")
    for key, attr in (("data", "data"), ("split_manifest", "split_manifest")):
        if digest(Path(getattr(args, attr))) != dataset[key + "_sha256"]:
            raise ValueError(f"Prior-prefix {key} identity mismatch")


def summary_metrics(summary: dict[str, Any]) -> dict[str, float | int]:
    """Compute body and tail from the frozen train-threshold strata, without rounding."""
    rows = summary["quantity_rows"]
    expected = {"le_p50", "p50_p90", "p90_p95", "p95_p99", "gt_p99"}
    if len(rows) != len(expected) or {r["stratum"] for r in rows} != expected:
        raise ValueError("Missing or duplicate quantity strata")
    if any(isinstance(r["count"], bool) or not isinstance(r["count"], int)
           or r["count"] < 0 for r in rows):
        raise ValueError("Invalid quantity stratum count")
    body = [r for r in rows if r["stratum"] in {"le_p50", "p50_p90", "p90_p95"}]
    tail = [r for r in rows if r["stratum"] == "gt_p99"]
    if len(body) != 3 or len(tail) != 1:
        raise ValueError("Missing or duplicate quantity strata")
    body_count = sum(r["count"] for r in body)
    if body_count <= 0 or tail[0]["count"] <= 0:
        raise ValueError("Empty body/tail cannot qualify")
    result = {
        "body_mae": sum(r["count"] * r["qty_mae"] for r in body) / body_count,
        "qty_mae": summary["best_val_qty_mae"],
        "qty_rmse": summary["best_val_qty_rmse"],
        "tail_mae": tail[0]["qty_mae"],
        "time_loss": summary["best_val_time_nll"],
        "log_qty_mse": summary["best_val_log_qty_mse"],
        "validation_targets": sum(r["count"] for r in rows),
        "body_targets": body_count, "tail_targets": tail[0]["count"],
    }
    if not all(math.isfinite(float(v)) for v in result.values()):
        raise ValueError("Nonfinite decision metric")
    return result


def metric_gates(candidate, baseline, t0, external, contract, dataset):
    """Evaluate B1 addition, original gate, common RMSE, and external claims separately."""
    if dataset not in contract["datasets"]:
        raise ValueError("Unknown comparison dataset")
    if set(external) != set(contract["performance_gates"]["external_TPP_claim"]["comparators"]):
        raise ValueError("Missing or extra external comparator")
    for metrics in (candidate, baseline, t0, *external.values()):
        if not all(math.isfinite(float(metrics[k])) for k in (
                "body_mae", "qty_mae", "qty_rmse", "tail_mae", "time_loss")):
            raise ValueError("Nonfinite comparison metric")
    gates = contract["performance_gates"]
    b1 = gates["B1_incremental"]
    t = gates["T0_legacy_gate"]
    c = gates["T0_common_RMSE"]
    comparisons = {
        "B1_incremental": {
            "body_preserved": candidate["body_mae"] <= baseline["body_mae"] * (1 + b1["all_datasets_body_mae_regression_max"]),
            "rmse_not_worse": candidate["qty_rmse"] <= baseline["qty_rmse"] * (1 + b1["all_datasets_rmse_regression_max"]),
            "tail_preserved": candidate["tail_mae"] <= baseline["tail_mae"] * (1 + b1["all_datasets_tail_mae_regression_max"]),
            "time_preserved": candidate["time_loss"] <= baseline["time_loss"] + b1["all_datasets_time_loss_increase_max"],
        },
        "T0_legacy_gate": {
            "body_improves_5pct": candidate["body_mae"] <= t0["body_mae"] * (1 - t["all_datasets_body_mae_improvement_min"]),
            "rmse_guardrail": candidate["qty_rmse"] <= t0["qty_rmse"] * (1 + t["all_datasets_rmse_regression_max"]),
            "tail_guardrail": candidate["tail_mae"] <= t0["tail_mae"] * (1 + t["all_datasets_tail_mae_regression_max"]),
            "time_guardrail": candidate["time_loss"] <= t0["time_loss"] + t["all_datasets_time_loss_increase_max"],
        },
        "T0_common_RMSE": {
            "rmse_strictly_decreases": candidate["qty_rmse"] < t0["qty_rmse"],
            "overall_mae_preserved": candidate["qty_mae"] <= t0["qty_mae"] * (1 + c["all_datasets_overall_mae_regression_max"]),
        },
    }
    if dataset == "insta_market_basket":
        comparisons["B1_incremental"]["instacart_body_improves_1pct"] = (
            candidate["body_mae"] <= baseline["body_mae"] * (1 - b1["instacart_body_mae_improvement_min"]))
    e = gates["external_TPP_claim"]
    comparisons["external_TPP_claim"] = {
        name: {
            "rmse_strictly_decreases": candidate["qty_rmse"] < ref["qty_rmse"],
            "mae_preserved": candidate["qty_mae"] <= ref["qty_mae"] * (1 + e["all_datasets_overall_mae_regression_max"]),
            "time_preserved": candidate["time_loss"] <= ref["time_loss"] + e["all_datasets_time_loss_increase_max"],
        } for name, ref in external.items()
    }
    return {
        "checks": comparisons,
        "screening_pass": all(all(comparisons[group].values()) for group in (
            "B1_incremental", "T0_legacy_gate", "T0_common_RMSE")),
        "external_TPP_claim_pass": all(all(row.values()) for row in comparisons["external_TPP_claim"].values()),
    }
