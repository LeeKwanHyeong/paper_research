"""
Build partial artifacts for an interrupted unified TPP experiment.

The long-epoch runner writes the final leaderboard, report, and plots only
after every scheduled run finishes. When a GPU job is intentionally stopped,
we still want to preserve the completed runs and the interrupted run history
as meeting-ready artifacts. This script reconstructs those partial outputs
from:

1. leaderboard/runs.csv for successfully completed runs
2. runs/**/metrics/history.json for completed and interrupted histories
3. experiment_manifest.json for the original planned run inventory

The output is written under <base-dir>/partial_artifacts so it never overwrites
the official artifacts that would be produced by a fully completed run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


DEFAULT_REMOTE_PROJECT_ROOT = "/home/leekwanhyeong/workspace/paper_research"
PRIMARY_HISTORY_METRICS = ("train_loss", "score", "val_nll", "qty_mae", "dt_mae")


@dataclass(frozen=True)
class RunKey:
    """Stable identity for one scheduled experiment run."""

    dataset_name: str
    model_name: str
    candidate_name: str
    seed: int

    def as_tuple(self) -> tuple[str, str, str, int]:
        return (self.dataset_name, self.model_name, self.candidate_name, self.seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create partial leaderboard, report, and plots for an interrupted TPP experiment."
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        required=True,
        help="Interrupted experiment directory under search_artifacts.",
    )
    parser.add_argument(
        "--remote-project-root",
        default=DEFAULT_REMOTE_PROJECT_ROOT,
        help="Project root prefix stored in server-side run_dir paths.",
    )
    parser.add_argument(
        "--output-name",
        default="partial_artifacts",
        help="Output directory name created under --base-dir.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_project_root(start: Path | None = None) -> Path:
    """Find the local project root from the current working directory."""

    anchor = (start or Path.cwd()).resolve()
    base = anchor if anchor.is_dir() else anchor.parent
    for candidate in [base, *base.parents]:
        if all((candidate / name).exists() for name in ("models", "simple_lab_test", "search_artifacts")):
            return candidate
    raise RuntimeError("Could not locate the paper_research project root.")


def localize_path(path_value: str | Path, *, project_root: Path, remote_project_root: str) -> Path:
    """
    Convert server-side absolute paths embedded in CSV/JSON metadata to local paths.

    Run metadata was generated on the GPU server, so paths often start with
    /home/leekwanhyeong/workspace/paper_research. After rsync, the same suffix
    lives under the local project root.
    """

    raw = str(path_value)
    if raw.startswith(remote_project_root):
        return project_root / raw[len(remote_project_root) :].lstrip("/")
    path = Path(raw)
    if path.exists():
        return path
    parts = path.parts
    if "search_artifacts" in parts:
        idx = parts.index("search_artifacts")
        return project_root.joinpath(*parts[idx:])
    return path


def safe_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_finite(value: Any) -> bool:
    number = safe_float(value)
    return math.isfinite(number)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_completed_runs(base_dir: Path, *, project_root: Path, remote_project_root: str) -> list[dict[str, Any]]:
    runs_path = base_dir / "leaderboard" / "runs.csv"
    if not runs_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with runs_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row = dict(row)
            if row.get("run_dir"):
                row["run_dir"] = str(
                    localize_path(row["run_dir"], project_root=project_root, remote_project_root=remote_project_root)
                )
            row["run_state"] = "completed"
            rows.append(row)
    return rows


def load_experiment_manifest(base_dir: Path) -> dict[str, Any]:
    manifest_path = base_dir / "experiment_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing experiment manifest: {manifest_path}")
    return read_json(manifest_path)


def planned_run_rows(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Rebuild the full run grid from the experiment manifest."""

    cfg = manifest["experiment_config"]
    profile_map = manifest.get("profile_map", {})
    rows: list[dict[str, Any]] = []
    for dataset_name in cfg.get("datasets", []):
        profile = profile_map.get(dataset_name, {})
        scale_base = profile.get("scale_base")
        for seed in cfg.get("seeds", []):
            if "titantpp" in cfg.get("models", []):
                for candidate_name in cfg.get("titan_candidates", []):
                    rows.append(
                        {
                            "dataset_name": dataset_name,
                            "model_name": "titantpp",
                            "candidate_name": candidate_name,
                            "seed": int(seed),
                            "scale_base": scale_base,
                        }
                    )
            if "thp" in cfg.get("models", []):
                for candidate_name in cfg.get("thp_candidates", []):
                    rows.append(
                        {
                            "dataset_name": dataset_name,
                            "model_name": "thp",
                            "candidate_name": candidate_name,
                            "seed": int(seed),
                            "scale_base": scale_base,
                        }
                    )
            if "rmtpp" in cfg.get("models", []):
                rows.append(
                    {
                        "dataset_name": dataset_name,
                        "model_name": "rmtpp",
                        "candidate_name": "rmtpp_gru_h64",
                        "seed": int(seed),
                        "scale_base": scale_base,
                    }
                )
    return rows


def run_key_from_row(row: dict[str, Any]) -> RunKey:
    return RunKey(
        dataset_name=str(row["dataset_name"]),
        model_name=str(row["model_name"]),
        candidate_name=str(row["candidate_name"]),
        seed=int(row["seed"]),
    )


def read_run_config(run_dir: Path) -> dict[str, Any]:
    config_path = run_dir / "manifest" / "run_config.json"
    if config_path.exists():
        return read_json(config_path)
    return {}


def scan_history_runs(base_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Read every run-level history file.

    The interrupted run is not present in leaderboard/runs.csv, but it still has
    metrics/history.json and manifest/run_config.json. We use those files to
    include its partial learning curve and NaN diagnostics.
    """

    history_rows: list[dict[str, Any]] = []
    inventory_rows: list[dict[str, Any]] = []
    for history_path in sorted((base_dir / "runs").glob("**/metrics/history.json")):
        run_dir = history_path.parent.parent
        config = read_run_config(run_dir)
        run_cfg = config.get("run_config", {})
        experiment_cfg = config.get("experiment_config", {})
        data = read_json(history_path)
        history = data.get("history", [])
        if not history:
            continue

        dataset_name = str(run_cfg.get("dataset_name", "unknown"))
        model_name = str(run_cfg.get("model_name", "unknown"))
        candidate_name = str(run_cfg.get("candidate_name", "unknown"))
        seed = int(run_cfg.get("seed", 0))
        scale_base = run_cfg.get("scale_base")
        epochs_target = int(run_cfg.get("epochs", experiment_cfg.get("epochs", 0)) or 0)
        key = RunKey(dataset_name, model_name, candidate_name, seed)

        first_nan_epoch: int | None = None
        last_finite_epoch: int | None = None
        for epoch_row in history:
            epoch = int(epoch_row.get("epoch", 0))
            has_nan = any(not is_finite(epoch_row.get(metric)) for metric in PRIMARY_HISTORY_METRICS)
            if has_nan and first_nan_epoch is None:
                first_nan_epoch = epoch
            if not has_nan:
                last_finite_epoch = epoch
            history_rows.append(
                {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "candidate_name": candidate_name,
                    "titan_candidate_name": candidate_name,
                    "seed": seed,
                    "scale_base": scale_base,
                    "run_dir": str(run_dir),
                    "epoch": epoch,
                    "score": safe_float(epoch_row.get("score")),
                    "val_nll": safe_float(epoch_row.get("val_nll")),
                    "val_nll_marker": safe_float(epoch_row.get("val_nll_marker")),
                    "val_nll_time": safe_float(epoch_row.get("val_nll_time")),
                    "qty_mae": safe_float(epoch_row.get("qty_mae")),
                    "dt_mae": safe_float(epoch_row.get("dt_mae")),
                    "mark_acc": safe_float(epoch_row.get("mark_acc")),
                    "train_loss": safe_float(epoch_row.get("train_loss")),
                    "run_key": "|".join(map(str, key.as_tuple())),
                }
            )

        inventory_rows.append(
            {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "candidate_name": candidate_name,
                "seed": seed,
                "scale_base": scale_base,
                "epochs_recorded": len(history),
                "epochs_target": epochs_target,
                "last_epoch": int(history[-1].get("epoch", len(history))),
                "first_nan_epoch": first_nan_epoch,
                "last_finite_epoch": last_finite_epoch,
                "history_path": str(history_path),
                "run_dir": str(run_dir),
                "run_key": "|".join(map(str, key.as_tuple())),
            }
        )
    return history_rows, inventory_rows


def attach_run_states(
    *,
    planned_rows: list[dict[str, Any]],
    completed_rows: list[dict[str, Any]],
    history_inventory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    completed_keys = {run_key_from_row(row).as_tuple() for row in completed_rows if row.get("status") == "success"}
    history_by_key = {
        run_key_from_row(row).as_tuple(): row
        for row in history_inventory
        if row.get("dataset_name") != "unknown"
    }
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(planned_rows, start=1):
        key = run_key_from_row(row).as_tuple()
        history = history_by_key.get(key, {})
        if key in completed_keys:
            state = "completed"
        elif key in history_by_key:
            state = "partial_interrupted"
        else:
            state = "not_started"
        output.append(
            {
                "planned_order": idx,
                **row,
                "run_state": state,
                "run_key": "|".join(map(str, key)),
                "epochs_recorded": history.get("epochs_recorded", 0),
                "last_epoch": history.get("last_epoch"),
                "first_nan_epoch": history.get("first_nan_epoch"),
                "last_finite_epoch": history.get("last_finite_epoch"),
                "history_path": history.get("history_path"),
            }
        )
    return output


def aggregate_completed_summary(completed_rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not completed_rows:
        return pl.DataFrame()
    numeric_cols = [
        "best_val_nll",
        "best_val_nll_epoch",
        "best_val_nll_score",
        "best_val_nll_qty_mae",
        "best_val_nll_dt_mae",
        "best_val_nll_mark_acc",
        "best_score",
        "best_score_epoch",
        "best_score_val_nll",
        "best_score_qty_mae",
        "final_val_nll",
        "final_qty_mae",
        "final_score",
        "final_train_loss",
    ]
    rows: list[dict[str, Any]] = []
    for row in completed_rows:
        converted = dict(row)
        for col in numeric_cols:
            converted[col] = safe_float(row.get(col))
        converted["seed"] = int(row.get("seed", 0))
        rows.append(converted)
    df = pl.DataFrame(rows)
    return (
        df.filter(pl.col("status") == "success")
        .group_by(["dataset_name", "model_name", "candidate_name"])
        .agg(
            [
                pl.len().alias("run_count"),
                pl.mean("best_val_nll").alias("mean_best_val_nll"),
                pl.mean("best_val_nll_epoch").alias("mean_best_val_nll_epoch"),
                pl.mean("best_val_nll_score").alias("mean_best_val_nll_score"),
                pl.mean("best_val_nll_qty_mae").alias("mean_best_val_nll_qty_mae"),
                pl.mean("best_val_nll_dt_mae").alias("mean_best_val_nll_dt_mae"),
                pl.mean("best_val_nll_mark_acc").alias("mean_best_val_nll_mark_acc"),
                pl.mean("final_val_nll").alias("mean_final_val_nll"),
                pl.mean("final_qty_mae").alias("mean_final_qty_mae"),
                pl.mean("final_score").alias("mean_final_score"),
            ]
        )
        .sort(["dataset_name", "mean_best_val_nll"])
    )


def best_completed_by_dataset(summary_df: pl.DataFrame) -> pl.DataFrame:
    if summary_df.height == 0:
        return pl.DataFrame()
    return (
        summary_df.sort(["dataset_name", "mean_best_val_nll"])
        .group_by("dataset_name", maintain_order=True)
        .head(1)
    )


def write_df(df: pl.DataFrame, path_prefix: Path) -> None:
    if df.height == 0:
        return
    df.write_csv(path_prefix.with_suffix(".csv"))
    df.write_parquet(path_prefix.with_suffix(".parquet"))


def line_label(row: dict[str, Any]) -> str:
    label = str(row["candidate_name"])
    if row.get("run_state") == "partial_interrupted":
        label += " (partial)"
    return label


def plot_learning_curves(history_df: pl.DataFrame, inventory_df: pl.DataFrame, plots_dir: Path) -> None:
    if history_df.height == 0:
        return
    ensure_dir(plots_dir)
    inventory_state = {
        row["run_key"]: row["run_state"]
        for row in inventory_df.to_dicts()
        if "run_key" in row and "run_state" in row
    }
    history_df = history_df.with_columns(
        pl.col("run_key")
        .map_elements(lambda key: inventory_state.get(key, "unknown"), return_dtype=pl.Utf8)
        .alias("run_state")
    )
    metrics = [
        ("train_loss", "Train Loss"),
        ("val_nll", "Validation NLL"),
        ("val_nll_marker", "Validation Marker NLL"),
        ("val_nll_time", "Validation Time NLL"),
        ("qty_mae", "Validation Qty MAE"),
        ("mark_acc", "Validation Mark Acc"),
    ]
    palette = [
        "#5DA5DA",
        "#F17CB0",
        "#60BD68",
        "#B276B2",
        "#F15854",
        "#DECF3F",
        "#FAA43A",
        "#4D4D4D",
    ]
    for dataset_name in history_df["dataset_name"].unique().sort().to_list():
        dataset_df = history_df.filter(pl.col("dataset_name") == dataset_name)
        run_keys = dataset_df.select("run_key").unique().sort("run_key")["run_key"].to_list()
        labels: dict[str, str] = {}
        for key in run_keys:
            first = dataset_df.filter(pl.col("run_key") == key).row(0, named=True)
            labels[key] = line_label(first)

        fig, axes = plt.subplots(2, 3, figsize=(20, 10))
        axes_flat = axes.flatten()
        for ax, (metric, title) in zip(axes_flat, metrics):
            for idx, key in enumerate(run_keys):
                run_df = dataset_df.filter(pl.col("run_key") == key).sort("epoch")
                x = np.array(run_df["epoch"].to_list(), dtype=float)
                y = np.array(run_df[metric].to_list(), dtype=float)
                state = inventory_state.get(key, "unknown")
                linestyle = "--" if state == "partial_interrupted" else "-"
                ax.plot(
                    x,
                    y,
                    label=labels[key],
                    color=palette[idx % len(palette)],
                    linewidth=1.8,
                    linestyle=linestyle,
                )
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"{dataset_name}: partial learning curves", fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(plots_dir / f"{dataset_name}_partial_learning_curves.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_run_coverage(inventory_df: pl.DataFrame, plots_dir: Path) -> None:
    if inventory_df.height == 0:
        return
    ensure_dir(plots_dir)
    count_df = (
        inventory_df.group_by(["dataset_name", "run_state"])
        .agg(pl.len().alias("count"))
        .sort(["dataset_name", "run_state"])
    )
    datasets = count_df["dataset_name"].unique().sort().to_list()
    states = ["completed", "partial_interrupted", "not_started"]
    colors = {
        "completed": "#60BD68",
        "partial_interrupted": "#FAA43A",
        "not_started": "#F15854",
    }
    bottom = np.zeros(len(datasets))
    fig, ax = plt.subplots(figsize=(10, 5))
    for state in states:
        values = []
        for dataset_name in datasets:
            row = count_df.filter((pl.col("dataset_name") == dataset_name) & (pl.col("run_state") == state))
            values.append(int(row["count"][0]) if row.height else 0)
        ax.bar(datasets, values, bottom=bottom, label=state, color=colors[state])
        bottom += np.array(values)
    ax.set_title("Partial experiment run coverage")
    ax.set_ylabel("Run count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plots_dir / "partial_run_coverage.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def format_float(value: Any, digits: int = 6) -> str:
    number = safe_float(value)
    if not math.isfinite(number):
        return "nan"
    return f"{number:.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows:
        return "_No rows._"
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def write_report(
    *,
    report_path: Path,
    base_dir: Path,
    inventory_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    best_df: pl.DataFrame,
    history_df: pl.DataFrame,
) -> None:
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total_runs = inventory_df.height
    completed = inventory_df.filter(pl.col("run_state") == "completed").height
    partial = inventory_df.filter(pl.col("run_state") == "partial_interrupted").height
    not_started = inventory_df.filter(pl.col("run_state") == "not_started").height

    coverage_rows = (
        inventory_df.group_by(["dataset_name", "run_state"])
        .agg(pl.len().alias("count"))
        .sort(["dataset_name", "run_state"])
        .to_dicts()
    )
    partial_rows = (
        inventory_df.filter(pl.col("run_state") == "partial_interrupted")
        .select(
            [
                "planned_order",
                "dataset_name",
                "candidate_name",
                "last_epoch",
                "first_nan_epoch",
                "last_finite_epoch",
            ]
        )
        .to_dicts()
    )
    missing_rows = (
        inventory_df.filter(pl.col("run_state") == "not_started")
        .select(["planned_order", "dataset_name", "candidate_name", "seed"])
        .to_dicts()
    )
    best_rows = []
    if best_df.height > 0:
        for row in best_df.to_dicts():
            best_rows.append(
                {
                    "dataset_name": row["dataset_name"],
                    "candidate_name": row["candidate_name"],
                    "best_val_nll": format_float(row["mean_best_val_nll"]),
                    "best_epoch": format_float(row["mean_best_val_nll_epoch"], digits=1),
                    "qty_mae": format_float(row["mean_best_val_nll_qty_mae"]),
                    "mark_acc": format_float(row["mean_best_val_nll_mark_acc"]),
                }
            )

    lines = [
        "# Partial Artifact Report",
        "",
        f"Generated at: {generated_at}",
        f"Source experiment: `{base_dir}`",
        "",
        "## Scope",
        "",
        "This report was generated after the original long-epoch experiment was interrupted.",
        "It includes completed run summaries plus any available partial learning histories.",
        "It does not pretend to be the final official experiment output.",
        "",
        "## Run Coverage",
        "",
        f"- Planned runs: {total_runs}",
        f"- Completed runs: {completed}",
        f"- Partial interrupted runs: {partial}",
        f"- Not-started runs: {not_started}",
        "",
        markdown_table(coverage_rows, ["dataset_name", "run_state", "count"]),
        "",
        "## Interrupted Run Diagnostics",
        "",
        markdown_table(
            partial_rows,
            [
                "planned_order",
                "dataset_name",
                "candidate_name",
                "last_epoch",
                "first_nan_epoch",
                "last_finite_epoch",
            ],
        ),
        "",
        "## Not Started Runs",
        "",
        markdown_table(missing_rows, ["planned_order", "dataset_name", "candidate_name", "seed"]),
        "",
        "## Best Completed Runs By Dataset",
        "",
        "These rows use only completed runs from `leaderboard/runs.csv`.",
        "",
        markdown_table(
            best_rows,
            ["dataset_name", "candidate_name", "best_val_nll", "best_epoch", "qty_mae", "mark_acc"],
        ),
        "",
        "## Files",
        "",
        "- `leaderboard/partial_run_inventory.csv`: planned/completed/partial/missing run map",
        "- `leaderboard/partial_histories.csv`: epoch-level histories, including interrupted runs",
        "- `leaderboard/completed_summary.csv`: completed-run aggregate summary",
        "- `leaderboard/best_completed_by_dataset.csv`: best completed candidate per dataset",
        "- `plots/*partial_learning_curves.png`: learning curves reconstructed from history files",
        "- `plots/partial_run_coverage.png`: completed/partial/not-started coverage",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = resolve_project_root()
    base_dir = args.base_dir.expanduser().resolve()
    output_dir = ensure_dir(base_dir / args.output_name)
    leaderboard_dir = ensure_dir(output_dir / "leaderboard")
    plots_dir = ensure_dir(output_dir / "plots")

    manifest = load_experiment_manifest(base_dir)
    planned_rows = planned_run_rows(manifest)
    completed_rows = read_completed_runs(
        base_dir,
        project_root=project_root,
        remote_project_root=args.remote_project_root,
    )
    history_rows, history_inventory = scan_history_runs(base_dir)
    inventory_rows = attach_run_states(
        planned_rows=planned_rows,
        completed_rows=completed_rows,
        history_inventory=history_inventory,
    )

    inventory_df = pl.DataFrame(inventory_rows)
    history_df = pl.DataFrame(history_rows)
    completed_summary_df = aggregate_completed_summary(completed_rows)
    best_df = best_completed_by_dataset(completed_summary_df)

    write_df(inventory_df, leaderboard_dir / "partial_run_inventory")
    write_df(history_df, leaderboard_dir / "partial_histories")
    write_df(completed_summary_df, leaderboard_dir / "completed_summary")
    write_df(best_df, leaderboard_dir / "best_completed_by_dataset")

    plot_learning_curves(history_df, inventory_df, plots_dir)
    plot_run_coverage(inventory_df, plots_dir)
    write_report(
        report_path=output_dir / "partial_report.md",
        base_dir=base_dir,
        inventory_df=inventory_df,
        summary_df=completed_summary_df,
        best_df=best_df,
        history_df=history_df,
    )

    print(f"partial artifacts written to: {output_dir}")
    print(f"planned={inventory_df.height}")
    print(f"completed={inventory_df.filter(pl.col('run_state') == 'completed').height}")
    print(f"partial={inventory_df.filter(pl.col('run_state') == 'partial_interrupted').height}")
    print(f"not_started={inventory_df.filter(pl.col('run_state') == 'not_started').height}")


if __name__ == "__main__":
    main()
