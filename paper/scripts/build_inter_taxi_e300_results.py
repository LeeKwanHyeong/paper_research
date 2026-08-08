#!/usr/bin/env python3
"""Build Intermittent/Taxi e300 comparison tables and figures for the manuscript."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mpl_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "paper" / "results" / "e300_matched_20260808"
RAW = RESULTS / "raw_remote"
OUT_TABLES = RESULTS / "tables"
OUT_FIGURES = RESULTS / "figures"

DATASET_LABELS = {
    "intermittent": "Intermittent",
    "yellow_trip_hourly": "Taxi",
}

MODEL_LABELS = {
    "rmtpp": "RMTPP-matched",
    "thp": "THP-matched",
    "titantpp": "TitanTPP",
}

MODEL_ORDER = ["RMTPP-matched", "THP-matched", "TitanTPP"]
DATASET_ORDER = ["Intermittent", "Taxi"]

STYLE = {
    "RMTPP-matched": {"color": "#5B6770", "hatch": ""},
    "THP-matched": {"color": "#C96B18", "hatch": "//"},
    "TitanTPP": {"color": "#2563EB", "hatch": ""},
}


def load_rows() -> pd.DataFrame:
    paths = [
        RAW / "intermittent" / "leaderboard" / "runs.csv",
        RAW / "yellow_trip_hourly" / "leaderboard" / "runs.csv",
        RAW / "titantpp_e300_20260808" / "intermittent" / "leaderboard" / "runs.csv",
        RAW / "titantpp_e300_20260808" / "yellow_trip_hourly" / "leaderboard" / "runs.csv",
    ]
    frames = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_csv(path))
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows["dataset_name"].isin(DATASET_LABELS)]
    rows["Dataset"] = rows["dataset_name"].map(DATASET_LABELS)
    rows["Model"] = rows["model_name"].map(MODEL_LABELS)
    rows = rows[rows["Model"].isin(MODEL_ORDER)]
    return rows


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "best_val_nll",
        "best_val_nll_qty_mae",
        "best_val_nll_dt_mae",
        "best_val_nll_mark_acc",
        "best_val_nll_epoch",
    ]
    summary = (
        rows.groupby(["Dataset", "Model"])
        .agg(
            n=("seed", "count"),
            **{f"{metric}_mean": (metric, "mean") for metric in metrics},
            **{f"{metric}_std": (metric, "std") for metric in metrics},
        )
        .reset_index()
    )
    summary["Dataset"] = pd.Categorical(summary["Dataset"], DATASET_ORDER, ordered=True)
    summary["Model"] = pd.Categorical(summary["Model"], MODEL_ORDER, ordered=True)
    return summary.sort_values(["Dataset", "Model"]).reset_index(drop=True)


def mean_std(row: pd.Series, metric: str, digits: int = 4) -> str:
    return f"{row[f'{metric}_mean']:.{digits}f} +/- {row[f'{metric}_std']:.{digits}f}"


def acc_mean_std(row: pd.Series) -> str:
    mean = row["best_val_nll_mark_acc_mean"] * 100.0
    std = row["best_val_nll_mark_acc_std"] * 100.0
    return f"{mean:.3f}% +/- {std:.3f}%p"


def write_tables(summary: pd.DataFrame) -> None:
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT_TABLES / "inter_taxi_e300_summary.csv", index=False)

    lines = [
        "# Intermittent and Taxi e300 validation summary",
        "",
        "Lower is better for Val NLL, Qty MAE, and Delta-t MAE. Higher is better for Mark acc.",
        "",
        "| Dataset | Model | n | Val NLL | Qty MAE | Delta-t MAE | Mark acc | Best epoch |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "| {dataset} | {model} | {n} | {nll} | {qty} | {dt} | {acc} | {epoch:.1f} |".format(
                dataset=row["Dataset"],
                model=row["Model"],
                n=int(row["n"]),
                nll=mean_std(row, "best_val_nll"),
                qty=mean_std(row, "best_val_nll_qty_mae"),
                dt=mean_std(row, "best_val_nll_dt_mae"),
                acc=acc_mean_std(row),
                epoch=row["best_val_nll_epoch_mean"],
            )
        )
    (OUT_TABLES / "inter_taxi_e300_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metric(summary: pd.DataFrame, metric: str, ylabel: str, title: str, filename: str) -> None:
    OUT_FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.0,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(6.4, 3.15))
    width = 0.22
    x_positions = range(len(DATASET_ORDER))
    offsets = [-width, 0, width]

    for offset, model in zip(offsets, MODEL_ORDER):
        subset = summary[summary["Model"] == model].set_index("Dataset").loc[DATASET_ORDER]
        values = subset[f"{metric}_mean"].to_numpy(dtype=float)
        errors = subset[f"{metric}_std"].to_numpy(dtype=float)
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width,
            yerr=errors,
            capsize=3,
            label=model,
            color=STYLE[model]["color"],
            hatch=STYLE[model]["hatch"],
            edgecolor="#17212B",
            linewidth=0.6,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.3f}" if metric == "best_val_nll" else f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
                color="#17212B",
            )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x_positions), DATASET_ORDER)
    ax.grid(axis="y", color="#D8DEE4", linewidth=0.8, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.16)
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    fig.tight_layout()
    fig.savefig(OUT_FIGURES / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_FIGURES / f"{filename}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    summary = summarize(rows)
    write_tables(summary)
    plot_metric(
        summary,
        "best_val_nll",
        "Validation NLL",
        "Validation likelihood on completed e300 comparisons",
        "inter_taxi_e300_validation_nll",
    )
    plot_metric(
        summary,
        "best_val_nll_qty_mae",
        "Quantity MAE",
        "Quantity reconstruction error on completed e300 comparisons",
        "inter_taxi_e300_quantity_mae",
    )


if __name__ == "__main__":
    main()
