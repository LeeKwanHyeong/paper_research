#!/usr/bin/env python3
"""Build the qualified Taxi and Instacart e300 validation tables."""

from __future__ import annotations

import csv
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper" / "results" / "e300_matched_20260808"
RAW = RESULTS / "raw_remote"
TABLES = RESULTS / "tables"

DATASETS = {
    "yellow_trip_hourly": "Taxi",
    "insta_market_basket": "Instacart",
}

MODELS = {
    "rmtpp": "Adapted RMTPP",
    "thp": "Adapted THP",
    "titantpp": "TitanTPP",
}

MODEL_ORDER = ["Adapted RMTPP", "Adapted THP", "TitanTPP"]
SEEDS = {42, 52, 62}
SOURCE_REVISION = "726aa64ab0b5478646d11be36fc19dcb224d417e"
METRICS = {
    "val_nll": "best_val_nll",
    "qty_mae": "best_val_nll_qty_mae",
    "dt_mae": "best_val_nll_dt_mae",
    "mark_acc": "best_val_nll_mark_acc",
    "best_epoch": "best_val_nll_epoch",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def source_paths(dataset_id: str) -> tuple[Path, Path]:
    baseline = RAW / dataset_id / "leaderboard" / "runs.csv"
    titan = (
        RAW
        / "titantpp_e300_20260808"
        / dataset_id
        / "leaderboard"
        / "runs.csv"
    )
    return baseline, titan


def load_qualified_rows() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dataset_id, dataset_label in DATASETS.items():
        baseline_path, titan_path = source_paths(dataset_id)
        raw_rows = read_csv(baseline_path) + read_csv(titan_path)
        for row in raw_rows:
            model_name = row.get("model_name", "")
            if model_name not in MODELS:
                continue
            if row.get("status") != "success":
                raise ValueError(f"Non-success row in {dataset_id}: {row}")
            if row.get("reproducibility_mode") != "strict":
                raise ValueError(f"Non-strict row in {dataset_id}: {row}")
            if row.get("split_mode") != "fixed":
                raise ValueError(f"Non-fixed split row in {dataset_id}: {row}")
            if row.get("source_revision") != SOURCE_REVISION:
                raise ValueError(f"Source revision mismatch in {dataset_id}: {row}")
            if row.get("evaluation_scope") != "validation_only":
                raise ValueError(f"Unexpected evaluation scope in {dataset_id}: {row}")
            if row.get("held_out_test_evaluated", "").lower() != "false":
                raise ValueError(f"Held-out test flag is not locked in {dataset_id}: {row}")
            if int(row["epochs"]) != 300:
                raise ValueError(f"Unexpected epoch budget in {dataset_id}: {row}")
            if not row.get("best_val_nll_checkpoint_path"):
                raise ValueError(f"Missing best-validation checkpoint in {dataset_id}: {row}")

            output.append(
                {
                    "dataset": dataset_label,
                    "model": MODELS[model_name],
                    "seed": int(row["seed"]),
                    **{
                        output_name: float(row[source_name])
                        for output_name, source_name in METRICS.items()
                    },
                }
            )

    for dataset in DATASETS.values():
        for model in MODEL_ORDER:
            seeds = {
                int(row["seed"])
                for row in output
                if row["dataset"] == dataset and row["model"] == model
            }
            if seeds != SEEDS:
                raise ValueError(f"Seed contract failed for {dataset}/{model}: {seeds}")
    return output


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    for dataset in DATASETS.values():
        for model in MODEL_ORDER:
            group = [
                row for row in rows if row["dataset"] == dataset and row["model"] == model
            ]
            record: dict[str, object] = {"dataset": dataset, "model": model, "n": len(group)}
            for metric in METRICS:
                values = [float(row[metric]) for row in group]
                record[f"{metric}_mean"] = statistics.mean(values)
                record[f"{metric}_std"] = statistics.stdev(values)
            summary.append(record)
    return summary


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def mean_std(row: dict[str, object], metric: str, digits: int = 4) -> str:
    return (
        f"{float(row[f'{metric}_mean']):.{digits}f} "
        f"+/- {float(row[f'{metric}_std']):.{digits}f}"
    )


def write_markdown(summary: list[dict[str, object]]) -> None:
    lines = [
        "# Taxi and Instacart e300 validation summary",
        "",
        "All values are mean +/- sample standard deviation over seeds 42, 52, and 62. "
        "Lower is better except for mark accuracy.",
    ]
    for dataset in DATASETS.values():
        lines.extend(
            [
                "",
                f"## {dataset}",
                "",
                "| Model | Val NLL | Qty MAE | Delta-t MAE | Mark acc. | Best epoch |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in summary:
            if row["dataset"] != dataset:
                continue
            acc_mean = 100.0 * float(row["mark_acc_mean"])
            acc_std = 100.0 * float(row["mark_acc_std"])
            lines.append(
                "| {model} | {nll} | {qty} | {dt} | {acc:.3f}% +/- {acc_std:.3f}%p | "
                "{epoch:.1f} +/- {epoch_std:.1f} |".format(
                    model=row["model"],
                    nll=mean_std(row, "val_nll"),
                    qty=mean_std(row, "qty_mae"),
                    dt=mean_std(row, "dt_mae"),
                    acc=acc_mean,
                    acc_std=acc_std,
                    epoch=float(row["best_epoch_mean"]),
                    epoch_std=float(row["best_epoch_std"]),
                )
            )
    (TABLES / "taxi_instacart_e300_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    seed_rows = load_qualified_rows()
    summary = summarize(seed_rows)
    write_csv(TABLES / "taxi_instacart_e300_seed_results.csv", seed_rows)
    write_csv(TABLES / "taxi_instacart_e300_summary.csv", summary)
    write_markdown(summary)


if __name__ == "__main__":
    main()
