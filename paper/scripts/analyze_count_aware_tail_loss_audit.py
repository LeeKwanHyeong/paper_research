#!/usr/bin/env python3
"""Audit train-only quantity tails before adding a tail-aware loss."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import polars as pl

from paper.scripts.run_intermittent_log_backbone_control import (
    EXPECTED_DATA_SHA256,
    EXPECTED_SPLIT_MANIFEST_SHA256,
)
from paper.scripts.run_taxi_quantity_interface_ablation import sha256_file


QUANTILES = (0.90, 0.95, 0.99)
TAIL_GRADIENT_STOP_SHARE = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    return parser.parse_args()


def nearest_quantiles(quantity: np.ndarray) -> dict[str, float]:
    series = pl.Series("quantity", quantity)
    return {
        f"p{int(level * 100)}": float(series.quantile(level, interpolation="nearest"))
        for level in QUANTILES
    }


def huber_gradient(residual: np.ndarray, *, delta: float) -> np.ndarray:
    return np.where(np.abs(residual) <= delta, residual, np.sign(residual) * delta)


def audit_quantity(quantity: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    quantity = np.asarray(quantity, dtype=np.float64)
    if quantity.ndim != 1 or quantity.size == 0:
        raise ValueError("quantity must be a non-empty one-dimensional array")
    if not np.isfinite(quantity).all() or (quantity < 0.0).any():
        raise ValueError("quantity must contain finite nonnegative values")

    quantiles = nearest_quantiles(quantity)
    p90, p95, p99 = (quantiles["p90"], quantiles["p95"], quantiles["p99"])
    if not (0.0 < p90 < p95 < p99):
        raise ValueError(f"Tail quantiles must be strictly increasing: {quantiles}")

    log_target = np.log1p(quantity)
    initial_location = float(log_target.mean())
    initial_prediction = float(np.expm1(initial_location))
    log_residual = initial_location - log_target
    log_mse = np.square(log_residual)
    log_location_gradient_abs = 2.0 * np.abs(log_residual)

    threshold = p95
    normalization_scale = p95
    clip_cap = p99
    huber_delta = 1.0
    tail_mask = quantity > threshold
    clipped_target = np.minimum(quantity, clip_cap) / normalization_scale
    clipped_prediction = min(initial_prediction, clip_cap) / normalization_scale
    normalized_residual = clipped_prediction - clipped_target
    raw_huber_location_gradient_abs = np.zeros_like(quantity)
    if initial_prediction < clip_cap:
        raw_huber_location_gradient_abs[tail_mask] = np.abs(
            huber_gradient(normalized_residual[tail_mask], delta=huber_delta)
            * math.exp(initial_location)
            / normalization_scale
        )

    stratum_specs = (
        ("le_p90", f"<= p90 ({p90:g})", quantity <= p90),
        ("p90_p95", f"p90-p95 ({p90:g}, {p95:g}]", (quantity > p90) & (quantity <= p95)),
        ("p95_p99", f"p95-p99 ({p95:g}, {p99:g}]", (quantity > p95) & (quantity <= p99)),
        ("gt_p99", f"> p99 ({p99:g})", quantity > p99),
    )
    total_log_loss = float(log_mse.sum())
    total_log_gradient = float(log_location_gradient_abs.sum())
    total_raw_aux_gradient = float(raw_huber_location_gradient_abs.sum())
    rows: list[dict[str, Any]] = []
    for order, (key, label, selected) in enumerate(stratum_specs):
        count = int(selected.sum())
        rows.append({
            "stratum_order": order,
            "stratum": key,
            "stratum_label": label,
            "count": count,
            "sample_share": count / quantity.size,
            "quantity_min": float(quantity[selected].min()) if count else None,
            "quantity_max": float(quantity[selected].max()) if count else None,
            "quantity_mean": float(quantity[selected].mean()) if count else None,
            "log_mse_share": float(log_mse[selected].sum()) / total_log_loss,
            "log_location_gradient_share": (
                float(log_location_gradient_abs[selected].sum()) / total_log_gradient
            ),
            "unweighted_tail_aux_gradient_share": (
                float(raw_huber_location_gradient_abs[selected].sum())
                / total_raw_aux_gradient
                if total_raw_aux_gradient > 0.0
                else 0.0
            ),
        })

    baseline_tail_gradient_share = float(
        log_location_gradient_abs[tail_mask].sum() / total_log_gradient
    )
    gate_checks = {
        "finite": bool(
            np.isfinite(log_mse).all()
            and np.isfinite(log_location_gradient_abs).all()
            and np.isfinite(raw_huber_location_gradient_abs).all()
        ),
        "tail_support_at_least_100": int(tail_mask.sum()) >= 100,
        "p95_tail_does_not_dominate_log_mse_gradient": (
            baseline_tail_gradient_share <= TAIL_GRADIENT_STOP_SHARE
        ),
        "tail_constants_ordered": 0.0 < threshold < clip_cap,
    }
    summary = {
        "status": "continue" if all(gate_checks.values()) else "stop",
        "scope": "intermittent_train_only",
        "event_count": int(quantity.size),
        "quantity_min": float(quantity.min()),
        "quantity_max": float(quantity.max()),
        "quantity_mean": float(quantity.mean()),
        "quantity_median": float(np.median(quantity)),
        "quantiles": quantiles,
        "initial_log_location": initial_location,
        "initial_raw_prediction": initial_prediction,
        "baseline_p95_tail_sample_share": float(tail_mask.mean()),
        "baseline_p95_tail_log_mse_share": float(log_mse[tail_mask].sum() / total_log_loss),
        "baseline_p95_tail_log_gradient_share": baseline_tail_gradient_share,
        "unweighted_tail_aux_to_log_gradient_ratio": (
            total_raw_aux_gradient / total_log_gradient
        ),
        "stop_threshold": {
            "metric": "p95_tail_absolute_log_location_gradient_share",
            "maximum": TAIL_GRADIENT_STOP_SHARE,
        },
        "proposed_constants": {
            "tail_threshold": threshold,
            "normalization_scale": normalization_scale,
            "clip_cap": clip_cap,
            "huber_delta": huber_delta,
            "tail_definition": "true_quantity > train_p95",
            "loss_reduction": "mean_over_all_target_events_with_zero_body_terms",
        },
        "gate_checks": gate_checks,
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_briefing(path: Path, payload: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    summary = payload["summary"]
    constants = summary["proposed_constants"]
    lines = [
        "# Count-aware Tail-aware Loss Train-only Audit",
        "",
        f"- 판정: **{summary['status'].upper()}**",
        f"- 분석 범위: train split `{summary['event_count']:,}` events only",
        f"- 수량 p90 / p95 / p99: `{summary['quantiles']['p90']:.0f}` / "
        f"`{summary['quantiles']['p95']:.0f}` / `{summary['quantiles']['p99']:.0f}`",
        f"- p95 초과 표본 비율: `{summary['baseline_p95_tail_sample_share']:.4%}`",
        f"- p95 초과 log-MSE loss 비율: `{summary['baseline_p95_tail_log_mse_share']:.4%}`",
        f"- p95 초과 absolute log-location gradient 비율: "
        f"`{summary['baseline_p95_tail_log_gradient_share']:.4%}`",
        "",
        "## Tail 구간",
        "",
        "| 구간 | 표본 수 | 비율 | Log-MSE 비율 | Log gradient 비율 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['stratum_label']} | {row['count']:,} | "
            f"{row['sample_share']:.4%} | {row['log_mse_share']:.4%} | "
            f"{row['log_location_gradient_share']:.4%} |"
        )
    lines.extend([
        "",
        "## 중복성 판정",
        "",
        "- Q3b/Q3c는 marked TPP의 `direct_raw_qty + causal shrinkage RevIN`에 "
        "log2 Huber를 더한 실험이다. 이번 계약은 mark와 RevIN이 없는 `log1p-MSE` "
        "decoder에 capped raw Huber를 더하므로 동일 실험이 아니다.",
        "- K=1은 log-normal NLL이 기존 log-MSE를 대체하고 shared encoder를 크게 "
        "변경했다. 이번 계약은 log-MSE를 그대로 유지하고 tail term만 보조하므로 "
        "K=1 반복 실험이 아니다.",
        "",
        "## 고정 계약",
        "",
        f"- Tail: `q > {constants['tail_threshold']:.0f}` (train p95)",
        f"- Raw normalization scale: `{constants['normalization_scale']:.0f}`",
        f"- Target/prediction cap: `{constants['clip_cap']:.0f}` (train p99)",
        f"- Huber delta: `{constants['huber_delta']:.1f}`",
        "- Body sample의 보조 손실은 0이며, reduction은 전체 target event 평균이다.",
        "- `lambda_tail`은 validation을 보지 않고 별도 train-only gradient calibration으로 고정한다.",
        "",
        "## 결론",
        "",
    ])
    if summary["status"] == "continue":
        lines.append(
            "기존 log-MSE의 p95 초과 gradient가 사전 중단선 50%를 넘지 않았다. "
            "따라서 tail-aware auxiliary 구현과 train-only coefficient calibration을 진행한다."
        )
    else:
        lines.append(
            "기존 log-MSE의 tail gradient가 이미 사전 중단선을 넘었다. 보조 tail loss를 "
            "추가하지 않고 가설을 종료한다."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if len(args.source_revision) != 40:
        raise ValueError("--source-revision must be a 40-character Git SHA")
    data_sha = sha256_file(args.data)
    manifest_sha = sha256_file(args.split_manifest)
    if data_sha != EXPECTED_DATA_SHA256:
        raise ValueError(f"Unexpected data SHA-256: {data_sha}")
    if manifest_sha != EXPECTED_SPLIT_MANIFEST_SHA256:
        raise ValueError(f"Unexpected split manifest SHA-256: {manifest_sha}")

    frame = pl.read_parquet(args.data)
    required = {"demand_qty", "chronological_split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    train = frame.filter(pl.col("chronological_split") == "train")
    quantity = train["demand_qty"].to_numpy().astype(np.float64)
    summary, rows = audit_quantity(quantity)
    payload = {
        "schema_version": 1,
        "source_revision": args.source_revision,
        "data_path": str(args.data.resolve()),
        "data_sha256": data_sha,
        "split_manifest_path": str(args.split_manifest.resolve()),
        "split_manifest_sha256": manifest_sha,
        "validation_rows_read": 0,
        "test_rows_read": 0,
        "summary": summary,
        "strata": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "audit.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "tail_strata.csv", rows)
    write_briefing(args.output_dir / "briefing.md", payload, rows)
    print(json.dumps(payload["summary"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
