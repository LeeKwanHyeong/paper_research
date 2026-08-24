#!/usr/bin/env python3
"""Audit e300 dataset contracts and build the compatible primary result table."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any


SEEDS = (42, 52, 62)
MODEL_ORDER = ("rmtpp", "thp", "nhp", "sahp", "titantpp", "titantpp_t1")
MODEL_SPECS = {
    "rmtpp": ("Adapted RMTPP", "T0 backbone control", "Direct log-MSE"),
    "thp": ("Adapted THP", "T0 backbone control", "Direct log-MSE"),
    "nhp": ("Adapted NHP", "T0 backbone control", "Direct log-MSE"),
    "sahp": ("Adapted SAHP", "T0 backbone control", "Direct log-MSE"),
    "titantpp": ("TitanTPP-T0", "T0 backbone control", "Direct log-MSE"),
    "titantpp_t1": (
        "TitanTPP-T1",
        "Tail-aware objective effect",
        "Log-MSE + tail-aware auxiliary",
    ),
}
REGIONS = {
    "body_le_p95": (0, 1, 2),
    "tail_gt_p95": (3, 4),
    "extreme_tail_gt_p99": (4,),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def load_scale_rows(root: Path) -> list[dict[str, Any]]:
    sources = (
        root
        / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/quantity_seed_metrics.csv",
        root
        / "search_artifacts/count_aware_external_t0_nhp_sahp_e300_20260820/quantity_seed_metrics.csv",
        root
        / "search_artifacts/count_aware_external_t0_nhp_seed62_e300_20260821_5090/quantity_seed_metrics.csv",
        root
        / "search_artifacts/count_aware_external_t0_sahp_all_e300_20260821_5090/quantity_seed_metrics.csv",
        root
        / "search_artifacts/count_aware_tail_auxiliary_screening_e300_20260816_5080_fresh_rerun/quantity_seed_metrics.csv",
        root
        / "search_artifacts/count_aware_tail_shared_multiseed_extension_e300_20260817/quantity_seed_metrics.csv",
    )
    output: list[dict[str, Any]] = []
    for path in sources:
        for row in read_csv(path):
            backbone = row["backbone"]
            variant = row["variant"]
            model = backbone
            if backbone == "titantpp" and variant == "count_only_log_mse_tail_shared":
                model = "titantpp_t1"
            elif variant != "count_only_log_regression":
                continue
            if model not in MODEL_ORDER:
                continue
            output.append({**row, "model": model})

    unique: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in output:
        key = (row["model"], int(row["seed"]), int(row["stratum_order"]))
        if key in unique:
            previous = unique[key]
            for metric in ("count", "qty_mae", "qty_rmse"):
                if not math.isclose(
                    float(previous[metric]), float(row[metric]), rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(f"Conflicting duplicate scale row: {key}/{metric}")
        unique[key] = row

    expected = {
        (model, seed, order)
        for model in MODEL_ORDER
        for seed in SEEDS
        for order in range(5)
    }
    if set(unique) != expected:
        missing = sorted(expected - set(unique))
        extra = sorted(set(unique) - expected)
        raise ValueError(f"Scale grid mismatch: missing={missing}, extra={extra}")
    return list(unique.values())


def summarize_regions(scale_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for region, orders in REGIONS.items():
            seed_values = []
            for seed in SEEDS:
                rows = [
                    row
                    for row in scale_rows
                    if row["model"] == model
                    and int(row["seed"]) == seed
                    and int(row["stratum_order"]) in orders
                ]
                if len(rows) != len(orders):
                    raise ValueError(f"Incomplete region: {model}/{seed}/{region}")
                total = sum(int(row["count"]) for row in rows)
                mae = sum(int(row["count"]) * float(row["qty_mae"]) for row in rows) / total
                mse = sum(
                    int(row["count"]) * float(row["qty_rmse"]) ** 2 for row in rows
                ) / total
                seed_values.append((mae, math.sqrt(mse), total))
            counts = {item[2] for item in seed_values}
            if len(counts) != 1:
                raise ValueError(f"Region count mismatch: {model}/{region}")
            label, role, objective = MODEL_SPECS[model]
            output.append(
                {
                    "model": model,
                    "model_label": label,
                    "comparison_role": role,
                    "quantity_objective": objective,
                    "region": region,
                    "count": counts.pop(),
                    "n_seeds": len(SEEDS),
                    "qty_mae_mean": statistics.mean(item[0] for item in seed_values),
                    "qty_mae_std": statistics.stdev(item[0] for item in seed_values),
                    "qty_rmse_mean": statistics.mean(item[1] for item in seed_values),
                    "qty_rmse_std": statistics.stdev(item[1] for item in seed_values),
                }
            )
    return output


def build_primary_rows(root: Path, regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overall = read_csv(
        root / "paper/results/count_aware_final_model_table_20260824/paper_model_table.csv"
    )
    overall_by_model = {row["model"]: row for row in overall}
    region_by_key = {(row["model"], row["region"]): row for row in regions}
    if set(overall_by_model) != set(MODEL_ORDER):
        raise ValueError("Overall model table does not contain the official model grid")

    output = []
    for model in MODEL_ORDER:
        source = overall_by_model[model]
        label, role, objective = MODEL_SPECS[model]
        row: dict[str, Any] = {
            "dataset": "Intermittent v2",
            "model": model,
            "model_label": label,
            "comparison_role": role,
            "quantity_objective": objective,
            "n_seeds": len(SEEDS),
            "time_nll_mean": float(source["best_val_time_nll_mean"]),
            "time_nll_std": float(source["best_val_time_nll_std"]),
            "quantity_mae_mean": float(source["best_val_qty_mae_mean"]),
            "quantity_mae_std": float(source["best_val_qty_mae_std"]),
            "quantity_rmse_mean": float(source["best_val_qty_rmse_mean"]),
            "quantity_rmse_std": float(source["best_val_qty_rmse_std"]),
        }
        for region in REGIONS:
            region_row = region_by_key[(model, region)]
            row[f"{region}_mae_mean"] = region_row["qty_mae_mean"]
            row[f"{region}_mae_std"] = region_row["qty_mae_std"]
            row[f"{region}_rmse_mean"] = region_row["qty_rmse_mean"]
            row[f"{region}_rmse_std"] = region_row["qty_rmse_std"]
        output.append(row)
    return output


def build_contract_audit(root: Path) -> list[dict[str, Any]]:
    intermittent = read_json(
        root
        / "paper/results/count_aware_tpp_backbone_control_20260812/source_5080/launch_contract.json"
    )
    taxi = read_json(
        root / "paper/results/taxi_log_backbone_control_20260811/source_5090/launch_contract.json"
    )
    instacart = read_json(
        root
        / "paper/results/instacart_validation_quantiles_20260810/quantile_analysis_contract.json"
    )
    if intermittent["interface"]["quantity_mark_used"] is not False:
        raise ValueError("Intermittent reference is not mark-free")
    if taxi["interface"]["history_quantity_input"] != "log10_within_mark_residual":
        raise ValueError("Taxi legacy interface changed unexpectedly")
    instacart_paths = " ".join(item["checkpoint_path"] for item in instacart["checkpoints"])
    if "lossmode_hybrid" not in instacart_paths or "valueinput_residual" not in instacart_paths:
        raise ValueError("Instacart legacy interface changed unexpectedly")

    common = {
        "seeds": "42,52,62",
        "max_epochs": 300,
        "batch_size": 128,
        "learning_rate": 0.001,
        "evaluation_scope": "validation_only",
        "held_out_test_used": False,
    }
    return [
        {
            "dataset": "Intermittent v2",
            "artifact": "count_aware_tpp_backbone_control + TitanTPP-T1 multiseed",
            **common,
            "split": "fixed split; dataset and split SHA verified",
            "mark_free": True,
            "history_quantity_input": "log1p raw quantity",
            "time_head": "legacy_clamped_rmtpp",
            "quantity_objective": "T0 direct log-MSE; T1 adds train-only tail auxiliary",
            "checkpoint_selection": "minimum validation joint objective",
            "time_nll_available": True,
            "contract_status": "compatible",
            "table_action": "include in primary table",
        },
        {
            "dataset": "Taxi",
            "artifact": "taxi_log_backbone_control_20260811",
            **common,
            "split": "fixed split; data SHA verified within legacy run",
            "mark_free": False,
            "history_quantity_input": "log10 within-mark residual",
            "time_head": "legacy hybrid contract; official common time-head field absent",
            "quantity_objective": "log1p MSE plus retained mark prediction path",
            "checkpoint_selection": "best validation event NLL",
            "time_nll_available": "partial; RMTPP decomposition missing",
            "contract_status": "incompatible",
            "table_action": "exclude; rerun under official T0/T1 contract",
        },
        {
            "dataset": "Instacart",
            "artifact": "final_fair e300 checkpoints + validation quantiles",
            **common,
            "split": "fixed split; data SHA verified within legacy run",
            "mark_free": False,
            "history_quantity_input": "within-mark residual",
            "time_head": "legacy hybrid contract; official common time-head field absent",
            "quantity_objective": "hybrid mark/value loss",
            "checkpoint_selection": "best validation event NLL",
            "time_nll_available": False,
            "contract_status": "incompatible",
            "table_action": "exclude; rerun under official T0/T1 contract",
        },
    ]


def build_rerun_matrix() -> list[dict[str, Any]]:
    output = []
    for dataset in ("Taxi", "Instacart"):
        for model in MODEL_ORDER:
            label, role, objective = MODEL_SPECS[model]
            output.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "model_label": label,
                    "comparison_role": role,
                    "target_quantity_objective": objective,
                    "reason": (
                        "legacy marked-hybrid artifact"
                        if model in {"rmtpp", "thp", "titantpp"}
                        else "official matched e300 artifact missing"
                    ),
                    "required_contract": (
                        "mark-free; legacy_clamped_rmtpp; seeds 42/52/62; e300; "
                        "batch 128; lr 0.001; minimum validation joint objective"
                    ),
                    "held_out_test": "locked",
                    "status": "rerun_required",
                }
            )
    return output


def fmt(mean: Any, std: Any) -> str:
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def render_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Compatible e300 primary comparison",
        "",
        "Only Intermittent v2 currently satisfies the official mark-free T0/T1 contract.",
        "Body is quantity <= train p95, tail is quantity > train p95, and extreme tail is quantity > train p99.",
        "All values are validation mean +/- sample standard deviation over seeds 42, 52, and 62.",
        "",
        "| Role | Model | Time NLL | Quantity MAE | Quantity RMSE | Body MAE | Tail MAE | >p99 MAE |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['comparison_role']} | {row['model_label']} | "
            f"{fmt(row['time_nll_mean'], row['time_nll_std'])} | "
            f"{fmt(row['quantity_mae_mean'], row['quantity_mae_std'])} | "
            f"{fmt(row['quantity_rmse_mean'], row['quantity_rmse_std'])} | "
            f"{fmt(row['body_le_p95_mae_mean'], row['body_le_p95_mae_std'])} | "
            f"{fmt(row['tail_gt_p95_mae_mean'], row['tail_gt_p95_mae_std'])} | "
            f"{fmt(row['extreme_tail_gt_p99_mae_mean'], row['extreme_tail_gt_p99_mae_std'])} |"
        )
    lines.extend(
        [
            "",
            "T0 rows isolate encoder/backbone effects under direct log-MSE. TitanTPP-T1 must be compared with TitanTPP-T0 to isolate the added tail-aware objective; it is not a pure backbone row.",
            "Taxi and Instacart are excluded because their existing e300 checkpoints retain the marked-hybrid interface and use best event NLL rather than the official mark-free validation joint objective.",
            "Held-out test data were not used.",
            "",
        ]
    )
    return "\n".join(lines)


def render_analysis(audit: list[dict[str, Any]], rows: list[dict[str, Any]]) -> str:
    by_model = {row["model"]: row for row in rows}
    t0 = by_model["titantpp"]
    t1 = by_model["titantpp_t1"]
    thp = by_model["thp"]

    def improvement(candidate: float, reference: float) -> float:
        return (reference - candidate) / reference * 100.0

    return "\n".join(
        [
            "# Intermittent-Taxi-Instacart e300 계약 감사 결과",
            "",
            "## 결론",
            "",
            "세 데이터셋을 한 표에 즉시 합칠 수 없다. 현재 공식 mark-free count formulation, 공통 time head, validation joint objective 선택까지 모두 충족하는 3-seed e300 결과는 Intermittent v2뿐이다. Taxi와 Instacart의 기존 e300은 수량 회귀 결과를 포함하지만 mark 입력과 mark loss를 유지한 이전 hybrid 실험이므로 참고 결과로만 남긴다.",
            "",
            "## Intermittent 주 비교",
            "",
            f"- TitanTPP-T1은 TitanTPP-T0 대비 전체 MAE를 {improvement(t1['quantity_mae_mean'], t0['quantity_mae_mean']):.2f}%, 전체 RMSE를 {improvement(t1['quantity_rmse_mean'], t0['quantity_rmse_mean']):.2f}% 낮췄다.",
            f"- Tail(>p95) MAE는 {improvement(t1['tail_gt_p95_mae_mean'], t0['tail_gt_p95_mae_mean']):.2f}%, extreme tail(>p99) MAE는 {improvement(t1['extreme_tail_gt_p99_mae_mean'], t0['extreme_tail_gt_p99_mae_mean']):.2f}% 낮아졌다.",
            f"- Adapted THP는 전체 MAE와 body MAE가 더 낮다. TitanTPP-T1의 THP 대비 전체 MAE 차이는 {-improvement(t1['quantity_mae_mean'], thp['quantity_mae_mean']):.2f}% 악화이고, RMSE는 {improvement(t1['quantity_rmse_mean'], thp['quantity_rmse_mean']):.2f}% 개선이다.",
            "- 따라서 T0 표는 backbone 비교로, TitanTPP-T0와 T1 차이는 objective ablation으로 각각 해석한다.",
            "",
            "## 제외 및 재실행 범위",
            "",
            "- Taxi: 기존 RMTPP/THP/TitanTPP e300 전부 mark-free 계약으로 재실행해야 한다. NHP, SAHP, TitanTPP-T1도 공식 matched e300 결과가 없다.",
            "- Instacart: 기존 RMTPP/THP/TitanTPP e300 전부 marked-hybrid checkpoint이므로 재실행해야 한다. NHP, SAHP, TitanTPP-T1도 공식 matched e300 결과가 없다.",
            "- 재실행 전까지 세 데이터셋 평균이나 순위를 계산하지 않는다.",
            "- Held-out test는 계속 잠근다.",
            "",
            "## Artifact 판정",
            "",
            *[
                f"- {row['dataset']}: {row['contract_status']} ({row['table_action']})"
                for row in audit
            ],
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scale_rows = load_scale_rows(root)
    regions = summarize_regions(scale_rows)
    primary_rows = build_primary_rows(root, regions)
    audit = build_contract_audit(root)
    reruns = build_rerun_matrix()

    write_csv(output_dir / "dataset_contract_audit.csv", audit)
    write_csv(output_dir / "rerun_matrix.csv", reruns)
    write_csv(output_dir / "intermittent_region_summary.csv", regions)
    write_csv(output_dir / "primary_comparison_table.csv", primary_rows)
    (output_dir / "primary_comparison_table.md").write_text(
        render_table(primary_rows), encoding="utf-8"
    )
    (output_dir / "analysis.md").write_text(
        render_analysis(audit, primary_rows), encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "evaluation_scope": "validation_only",
        "held_out_test_evaluated": False,
        "compatible_datasets": ["Intermittent v2"],
        "rerun_required_datasets": ["Taxi", "Instacart"],
        "seeds": list(SEEDS),
        "models": list(MODEL_ORDER),
        "body_definition": "quantity <= train p95",
        "tail_definition": "quantity > train p95",
        "extreme_tail_definition": "quantity > train p99",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
