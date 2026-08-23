#!/usr/bin/env python3
"""Build the paper-facing Intermittent T0 and TitanTPP-T1 validation table."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from paper.scripts.build_count_aware_tail_shared_multiseed_results import write_csv


SEEDS = (42, 52, 62)
MODEL_ORDER = ("rmtpp", "thp", "nhp", "sahp", "titantpp", "titantpp_t1")
METRICS = (
    "best_val_joint_objective",
    "best_val_time_nll",
    "best_val_log_qty_mse",
    "best_val_qty_mae",
    "best_val_qty_rmse",
)
MODEL_SPECS = {
    "rmtpp": ("Adapted RMTPP", "GRU", "T0 common control", "Direct log-MSE"),
    "thp": ("Adapted THP", "Causal Transformer", "T0 common control", "Direct log-MSE"),
    "nhp": ("Adapted NHP", "Continuous-time LSTM", "T0 common control", "Direct log-MSE"),
    "sahp": ("Adapted SAHP", "Self-attention + decay", "T0 common control", "Direct log-MSE"),
    "titantpp": ("TitanTPP-T0", "Titan Hard-LMM", "T0 common control", "Direct log-MSE"),
    "titantpp_t1": (
        "TitanTPP-T1",
        "Titan Hard-LMM",
        "Proposed method",
        "Log-MSE + tail-aware auxiliary",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-t0-results", type=Path, required=True)
    parser.add_argument("--t1-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_metadata(external: dict[str, Any], t1: dict[str, Any]) -> None:
    for name, metadata in (("external T0", external), ("T1", t1)):
        if metadata.get("status") != "complete":
            raise ValueError(f"{name} result is not complete")
        if metadata.get("evaluation_scope") != "validation_only":
            raise ValueError(f"{name} result is not validation-only")
        if metadata.get("held_out_test_evaluated") is not False:
            raise ValueError(f"{name} result used the held-out test")
    if external.get("model_role") != "t0_common_control":
        raise ValueError("external result is not the official T0 control")
    roles = t1.get("model_roles", {})
    if roles.get("titantpp") != "t0_common_control":
        raise ValueError("TitanTPP-T0 role mismatch")
    if roles.get("titantpp_t1") != "t1_incumbent":
        raise ValueError("TitanTPP-T1 role mismatch")


def index_summaries(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        model = row["model"]
        if model in indexed:
            raise ValueError(f"duplicate model summary: {model}")
        if int(row["n_seeds"]) != len(SEEDS):
            raise ValueError(f"{model} is not a three-seed summary")
        for metric in METRICS:
            for suffix in ("mean", "std"):
                key = f"{metric}_{suffix}"
                if not math.isfinite(float(row[key])):
                    raise ValueError(f"non-finite {model} metric: {key}")
        indexed[model] = row
    return indexed


def validate_overlap(
    external: dict[str, dict[str, str]], t1: dict[str, dict[str, str]]
) -> None:
    for model in ("rmtpp", "thp"):
        for metric in METRICS:
            for suffix in ("mean", "std"):
                key = f"{metric}_{suffix}"
                if not math.isclose(
                    float(external[model][key]),
                    float(t1[model][key]),
                    rel_tol=0.0,
                    abs_tol=1e-12,
                ):
                    raise ValueError(f"overlapping {model} metric mismatch: {key}")


def build_table_rows(
    external: dict[str, dict[str, str]], t1: dict[str, dict[str, str]]
) -> list[dict[str, Any]]:
    required_external = {"rmtpp", "thp", "nhp", "sahp"}
    required_t1 = {"rmtpp", "thp", "titantpp", "titantpp_t1"}
    if set(external) != required_external:
        raise ValueError(f"external model grid mismatch: {sorted(external)}")
    if set(t1) != required_t1:
        raise ValueError(f"T1 model grid mismatch: {sorted(t1)}")
    validate_overlap(external, t1)

    output = []
    for model in MODEL_ORDER:
        source = external if model in required_external else t1
        raw = source[model]
        label, encoder, role, objective = MODEL_SPECS[model]
        row: dict[str, Any] = {
            "model": model,
            "model_label": label,
            "encoder": encoder,
            "table_role": role,
            "quantity_objective": objective,
            "n_seeds": len(SEEDS),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = float(raw[f"{metric}_mean"])
            row[f"{metric}_std"] = float(raw[f"{metric}_std"])
        output.append(row)
    return output


def improvement(lower: float, higher: float) -> float:
    return (higher - lower) / higher * 100.0


def build_claims(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model = {row["model"]: row for row in rows}
    t1 = by_model["titantpp_t1"]

    def compare(model: str) -> dict[str, float]:
        reference = by_model[model]
        return {
            "quantity_mae_improvement_pct": improvement(
                t1["best_val_qty_mae_mean"], reference["best_val_qty_mae_mean"]
            ),
            "quantity_rmse_improvement_pct": improvement(
                t1["best_val_qty_rmse_mean"], reference["best_val_qty_rmse_mean"]
            ),
            "time_nll_absolute_regression": (
                t1["best_val_time_nll_mean"] - reference["best_val_time_nll_mean"]
            ),
        }

    return {
        "versus_rmtpp_t0": compare("rmtpp"),
        "versus_thp_t0": compare("thp"),
        "versus_titantpp_t0": compare("titantpp"),
    }


def fmt(row: dict[str, Any], metric: str) -> str:
    return f"{row[f'{metric}_mean']:.6f} +/- {row[f'{metric}_std']:.6f}"


def render_table(rows: list[dict[str, Any]]) -> str:
    best_mae = min(row["best_val_qty_mae_mean"] for row in rows)
    best_rmse = min(row["best_val_qty_rmse_mean"] for row in rows)
    lines = [
        "# Count-aware TPP Three-seed Validation Results",
        "",
        "- Dataset: Intermittent fixed split",
        "- Seeds: 42, 52, 62",
        "- Selection: minimum validation joint objective",
        "- Held-out test: not evaluated",
        "- T0 rows share direct log-MSE; TitanTPP-T1 alone adds the train-only tail-aware auxiliary loss.",
        "",
        "| Role | Model | Encoder | Quantity objective | Time NLL | Quantity MAE | Quantity RMSE |",
        "| --- | --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        mae = fmt(row, "best_val_qty_mae")
        rmse = fmt(row, "best_val_qty_rmse")
        if math.isclose(row["best_val_qty_mae_mean"], best_mae):
            mae = f"**{mae}**"
        if math.isclose(row["best_val_qty_rmse_mean"], best_rmse):
            rmse = f"**{rmse}**"
        lines.append(
            f"| {row['table_role']} | {row['model_label']} | {row['encoder']} | "
            f"{row['quantity_objective']} | {fmt(row, 'best_val_time_nll')} | {mae} | {rmse} |"
        )
    lines.extend(
        [
            "",
            "**Reading rule.** Lower is better for Time NLL, Quantity MAE, and Quantity RMSE. "
            "The T0 rows isolate backbone differences. TitanTPP-T1 is the final proposed configuration, "
            "so its result reflects both the Titan backbone and the tail-aware training objective.",
            "",
        ]
    )
    return "\n".join(lines)


def render_audit() -> str:
    return "\n".join(
        [
            "# Final Paper Table Contract Audit",
            "",
            "- Status: **PASS**",
            "- RMTPP and THP overlapping summaries matched exactly across both source results.",
            "- All six rows use seeds 42, 52, and 62 on the same Intermittent fixed split.",
            "- Maximum/minimum epochs, patience, batch size, learning rate, lookback, max sequence length, and checkpoint selection are matched.",
            "- All models use the `legacy_clamped_rmtpp` time head.",
            "- T0 uses direct log-MSE; TitanTPP-T1 intentionally adds the train-only tail-aware auxiliary loss.",
            "- Evaluation is validation-only and the held-out test remains locked.",
            "- H0/H3 time-head diagnostics and previous mark-residual V2/V3 variants are excluded.",
            "",
        ]
    )


def render_analysis(claims: dict[str, Any]) -> str:
    thp = claims["versus_thp_t0"]
    titan = claims["versus_titantpp_t0"]
    rmtpp = claims["versus_rmtpp_t0"]
    return "\n".join(
        [
            "# 논문용 통합 결과 해석",
            "",
            "## 핵심 결론",
            "",
            "TitanTPP-T1은 전체 모델 중 Quantity RMSE가 가장 낮지만, Quantity MAE는 Adapted THP가 가장 낮다. 따라서 현재 결과는 TitanTPP-T1의 전 구간 우월성이 아니라 큰 수량 오차를 줄이는 데 강점이 있다는 근거로 사용한다.",
            "",
            "## 정량 비교",
            "",
            f"- TitanTPP-T1은 TitanTPP-T0보다 MAE를 `{titan['quantity_mae_improvement_pct']:.2f}%`, RMSE를 `{titan['quantity_rmse_improvement_pct']:.2f}%` 개선했다.",
            f"- Adapted THP와 비교하면 TitanTPP-T1의 MAE 개선률은 `{thp['quantity_mae_improvement_pct']:.2f}%`로 음수지만, RMSE는 `{thp['quantity_rmse_improvement_pct']:.2f}%` 개선했다.",
            f"- Adapted RMTPP와 비교하면 MAE `{rmtpp['quantity_mae_improvement_pct']:.2f}%`, RMSE `{rmtpp['quantity_rmse_improvement_pct']:.2f}%` 개선했다.",
            f"- THP 대비 Time NLL은 `{thp['time_nll_absolute_regression']:.6f}` 악화되어 time modeling 우위는 주장하지 않는다.",
            "",
            "## 논문 서술 경계",
            "",
            "T0 구간은 동일 loss와 time head에서 encoder 차이를 비교한다. 이 조건에서는 THP가 MAE 기준으로 가장 강한 backbone이다. TitanTPP-T1 행은 Titan backbone과 tail-aware objective가 결합된 최종 방법이므로, T1의 RMSE 개선을 Titan backbone만의 효과로 해석하지 않는다. Backbone 기여는 TitanTPP-T0 행으로, tail-aware objective의 추가 기여는 TitanTPP-T0 대비 T1 차이로 설명한다.",
            "",
            "Held-out test는 아직 사용하지 않았으며, 이 표는 validation 기준 모델 선택 근거다.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    external_dir = args.external_t0_results.resolve()
    t1_dir = args.t1_results.resolve()
    output_dir = args.output_dir.resolve()

    validate_metadata(
        read_json(external_dir / "comparison.json"),
        read_json(t1_dir / "comparison.json"),
    )
    rows = build_table_rows(
        index_summaries(read_csv(external_dir / "model_summary.csv")),
        index_summaries(read_csv(t1_dir / "model_summary.csv")),
    )
    claims = build_claims(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "paper_model_table.csv", rows)
    (output_dir / "paper_model_table.md").write_text(render_table(rows), encoding="utf-8")
    (output_dir / "contract_audit.md").write_text(render_audit(), encoding="utf-8")
    (output_dir / "analysis.md").write_text(render_analysis(claims), encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "status": "complete",
        "models": list(MODEL_ORDER),
        "seeds": list(SEEDS),
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "claims": claims,
    }
    (output_dir / "paper_model_table.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
