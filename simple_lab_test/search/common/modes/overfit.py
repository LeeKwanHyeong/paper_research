"""
Overfitting diagnostic for RMTPP and TitanTPP.

This script is intentionally different from the paper A/B benchmark. The goal
is not to report the prettiest validation number, but to prove that each model
can actually fit the training distribution when capacity, sequence length, and
learning rate are made aggressive enough.

What it checks:
1. RMTPP capacity by RNN type (`rnn`, `gru`, `lstm`), mark embedding width, and
   hidden width
2. TitanTPP capacity by Titan preset, mark embedding width, and `max_seq_len`
3. train-vs-validation divergence using a deliberately higher learning rate
   default (`1e-3`)

The `yellow_trip_*` presets are follow-up diagnostics for the case where the
full yellow-trip run still improved at epoch 100. They keep the intermittent
results untouched by writing to separate output directories.

If a model is learning properly, at least one high-capacity configuration should
show a clear train-loss decrease. If validation NLL later rises while train loss
continues to fall, we have an explicit overfitting signal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_paper_research")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache_paper_research")

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import torch


def _configure_stdio_utf8() -> None:
    """
    Keep Korean/English logs readable on remote Linux shells.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_configure_stdio_utf8()

THIS_FILE = Path(__file__).resolve()
BOOTSTRAP_ROOT = THIS_FILE.parents[2]
if str(BOOTSTRAP_ROOT) not in sys.path:
    sys.path.insert(0, str(BOOTSTRAP_ROOT))

from simple_lab_test.common.pathing import ensure_project_root_on_path

PROJECT_ROOT = ensure_project_root_on_path(THIS_FILE)

from simple_lab_test.search.common.models import default_titan_candidates
from simple_lab_test.search.common.experiment_utils import (
    MARKED_TARGET_MAX_SEQ_LEN,
    MARKED_TARGET_TITAN_CANDIDATES,
    TitanCandidate,
    build_logger,
    ensure_dir,
    is_marked_target_kind,
    save_json,
    to_jsonable,
)
from simple_lab_test.search.common.benchmark_utils import (
    build_marked_cache,
    default_profile_map,
    find_candidate_by_name,
    make_dataset_specs,
    markdown_table_from_df,
    persist_rows,
)
from simple_lab_test.search.common.modes.long_epoch_legacy import (
    LongEpochConfig,
    LongRunConfig,
    build_error_row,
    train_one_long_run,
)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_int_list(text: str) -> tuple[int, ...]:
    """
    Parse comma-separated integers from CLI arguments.
    """
    return tuple(int(value.strip()) for value in text.split(",") if value.strip())


def parse_float_label(value: float) -> str:
    """
    Turn a learning rate into a path-safe short label.
    """
    return f"{value:.0e}".replace("-", "m").replace("+", "p")


def parse_str_list(text: str) -> tuple[str, ...]:
    """
    Parse comma-separated strings from CLI arguments.
    """
    return tuple(value.strip() for value in text.split(",") if value.strip())


def cli_arg_was_passed(option_name: str) -> bool:
    """
    Detect whether a CLI option was explicitly supplied by the user.

    Presets should provide convenient defaults, but an explicit command-line
    option should still be allowed to override the preset value.
    """
    prefix = f"{option_name}="
    return any(arg == option_name or arg.startswith(prefix) for arg in sys.argv[1:])


def set_preset_default(args: argparse.Namespace, option_name: str, attr_name: str, value: Any) -> None:
    """
    Apply a preset value only when the user did not pass that option manually.
    """
    if not cli_arg_was_passed(option_name):
        setattr(args, attr_name, value)


def apply_experiment_preset(args: argparse.Namespace) -> argparse.Namespace:
    """
    Expand high-level follow-up presets into concrete CLI settings.

    `yellow_trip_full_long` checks whether the full yellow-trip data eventually
    overfits when trained longer. `yellow_trip_subset_stress` intentionally
    reduces the number of series and sweeps sequence/model capacity more
    aggressively so overfitting should be easier to trigger.
    """
    if args.preset == "custom":
        return args

    if args.preset == "yellow_trip_full_long":
        set_preset_default(
            args,
            "--base-dir",
            "base_dir",
            str(PROJECT_ROOT / "search_artifacts" / "tpp_overfit_yellow_trip_full_long"),
        )
        set_preset_default(args, "--datasets", "datasets", "yellow_trip_hourly")
        set_preset_default(args, "--epochs", "epochs", 300)
        set_preset_default(args, "--lr", "lr", 1e-3)
        set_preset_default(args, "--max-seq-lens", "max_seq_lens", "250")
        set_preset_default(args, "--rmtpp-rnn-types", "rmtpp_rnn_types", "rnn,gru,lstm")
        set_preset_default(args, "--rmtpp-hidden-dims", "rmtpp_hidden_dims", "64,128,256")
        set_preset_default(args, "--rmtpp-mark-emb-dims", "rmtpp_mark_emb_dims", "32,64")
        set_preset_default(args, "--titan-candidates", "titan_candidates", "small_lmm,mid_lmm,mid_deep_lmm")
        set_preset_default(args, "--titan-mark-emb-dims", "titan_mark_emb_dims", "32,64")
        return args

    if args.preset == "yellow_trip_subset_stress":
        set_preset_default(
            args,
            "--base-dir",
            "base_dir",
            str(PROJECT_ROOT / "search_artifacts" / "tpp_overfit_yellow_trip_subset_stress"),
        )
        set_preset_default(args, "--datasets", "datasets", "yellow_trip_hourly")
        set_preset_default(args, "--epochs", "epochs", 300)
        set_preset_default(args, "--lr", "lr", 1e-3)
        set_preset_default(args, "--yellow-max-series", "yellow_max_series", 120)
        set_preset_default(args, "--max-seq-lens", "max_seq_lens", "64,128,250")
        set_preset_default(args, "--rmtpp-rnn-types", "rmtpp_rnn_types", "gru,lstm")
        set_preset_default(args, "--rmtpp-hidden-dims", "rmtpp_hidden_dims", "128,256")
        set_preset_default(args, "--rmtpp-mark-emb-dims", "rmtpp_mark_emb_dims", "32,64")
        set_preset_default(args, "--titan-candidates", "titan_candidates", "mid_lmm,mid_deep_lmm")
        set_preset_default(args, "--titan-mark-emb-dims", "titan_mark_emb_dims", "32,64")
        return args

    raise ValueError(f"Unsupported preset: {args.preset}")


def copy_titan_candidate(candidate: TitanCandidate, *, name: str) -> TitanCandidate:
    """
    Preserve a Titan preset while giving it a config-specific run name.
    """
    return replace(candidate, name=name)


def make_rmtpp_capacity_candidate(
    *,
    rnn_type: str,
    mark_emb_dim: int,
    hidden_dim: int,
    max_seq_len: int,
    lr: float,
) -> TitanCandidate:
    """
    RMTPP uses only `d_model` from TitanCandidate in the shared builder.

    The rest of the fields are harmless metadata, but keeping them populated
    lets us reuse the existing run/manifest/checkpoint layout without inventing
    a second config object.
    """
    name = (
        f"rmtpp_{rnn_type}"
        f"_emb{mark_emb_dim}"
        f"_h{hidden_dim}"
        f"_seq{max_seq_len}"
        f"_lr{parse_float_label(lr)}"
    )
    return TitanCandidate(
        name=name,
        d_model=int(hidden_dim),
        n_layers=1,
        n_heads=1,
        d_ff=max(int(hidden_dim) * 2, 32),
        dropout=0.0,
        contextual_mem_size=0,
        persistent_mem_size=0,
        use_lmm=False,
        mem_size=0,
        mem_topk=0,
    )


def make_titan_capacity_candidate(
    *,
    base_candidate: TitanCandidate,
    mark_emb_dim: int,
    max_seq_len: int,
    lr: float,
) -> TitanCandidate:
    """
    Attach sequence length and mark embedding width to the Titan run identity.
    """
    name = (
        f"titan_{base_candidate.name}"
        f"_emb{mark_emb_dim}"
        f"_seq{max_seq_len}"
        f"_lr{parse_float_label(lr)}"
    )
    return copy_titan_candidate(base_candidate, name=name)


def max_seq_lens_for_spec(spec, max_seq_lens: tuple[int, ...]) -> tuple[int, ...]:
    """
    Keep the marked target diagnostic compact unless the user explicitly opts out.
    """
    if is_marked_target_kind(spec.kind) and not cli_arg_was_passed("--max-seq-lens"):
        return (MARKED_TARGET_MAX_SEQ_LEN,)
    return max_seq_lens


def titan_candidate_names_for_spec(spec, candidate_names: tuple[str, ...]) -> tuple[str, ...]:
    """
    Restrict marked target Titan diagnostics to the small presets by default.
    """
    if not is_marked_target_kind(spec.kind) or cli_arg_was_passed("--titan-candidates"):
        return candidate_names

    filtered = tuple(name for name in candidate_names if name in MARKED_TARGET_TITAN_CANDIDATES)
    return filtered or ("small_lmm",)


# ---------------------------------------------------------------------------
# Aggregation and overfit diagnostics
# ---------------------------------------------------------------------------

def load_overfit_histories(run_rows: list[dict[str, Any]]) -> pl.DataFrame:
    """
    Expand per-run history files while keeping config identity columns.
    """
    rows: list[dict[str, Any]] = []
    for row in run_rows:
        if row.get("status") != "success":
            continue
        history_path = Path(str(row["run_dir"])) / "metrics" / "history.json"
        if not history_path.exists():
            continue
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f).get("history", [])
        for epoch_row in history:
            rows.append({
                "dataset_name": row["dataset_name"],
                "model_name": row["model_name"],
                "seed": int(row["seed"]),
                "config_id": row["titan_candidate_name"],
                "rmtpp_rnn_type": row.get("rmtpp_rnn_type", ""),
                "rmtpp_hidden_dim": int(row.get("rmtpp_hidden_dim", 0)),
                "rmtpp_mark_emb_dim": int(row.get("rmtpp_mark_emb_dim", 0)),
                "max_seq_len": int(row.get("max_seq_len", 0)),
                "lr": float(row.get("lr", float("nan"))),
                "epoch": int(epoch_row["epoch"]),
                "train_loss": float(epoch_row["train_loss"]),
                "score": float(epoch_row["score"]),
                "val_nll": float(epoch_row["val_nll"]),
                "val_nll_marker": float(epoch_row.get("val_nll_marker", float("nan"))),
                "val_nll_time": float(epoch_row.get("val_nll_time", float("nan"))),
                "qty_mae": float(epoch_row["qty_mae"]),
                "dt_mae": float(epoch_row["dt_mae"]),
                "mark_acc": float(epoch_row["mark_acc"]),
            })
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def build_overfit_tables(
    *,
    run_rows: list[dict[str, Any]],
    overfit_gap_threshold: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Convert run summaries into run-level and config-level diagnostic tables.
    """
    run_df = pl.DataFrame([{key: to_jsonable(value) for key, value in row.items()} for row in run_rows])
    success_df = run_df.filter(pl.col("status") == "success")
    if success_df.height == 0:
        return run_df, pl.DataFrame()

    diagnostic_df = (
        success_df.with_columns([
            (pl.col("final_val_nll") - pl.col("best_val_nll")).alias("overfit_gap_val_nll"),
            (pl.col("best_score") - pl.col("final_score")).alias("score_drop_after_best"),
            (pl.col("final_qty_mae") - pl.col("best_val_nll_qty_mae")).alias("qty_mae_drift_after_best_nll"),
            (pl.col("best_val_nll_epoch") / pl.col("epochs")).alias("best_val_nll_epoch_ratio"),
        ])
        .with_columns([
            (
                (pl.col("overfit_gap_val_nll") >= overfit_gap_threshold)
                & (pl.col("best_val_nll_epoch") < pl.col("epochs"))
            ).alias("overfit_observed"),
        ])
    )

    group_cols = [
        "dataset_name",
        "model_name",
        "titan_candidate_name",
        "max_seq_len",
        "lr",
        "rmtpp_rnn_type",
        "rmtpp_hidden_dim",
        "rmtpp_mark_emb_dim",
    ]
    summary_df = (
        diagnostic_df.group_by(group_cols)
        .agg([
            pl.len().alias("run_count"),
            pl.mean("best_val_nll").alias("mean_best_val_nll"),
            pl.std("best_val_nll").fill_null(0.0).alias("std_best_val_nll"),
            pl.mean("best_val_nll_epoch").alias("mean_best_val_nll_epoch"),
            pl.mean("final_val_nll").alias("mean_final_val_nll"),
            pl.mean("overfit_gap_val_nll").alias("mean_overfit_gap_val_nll"),
            pl.mean("overfit_observed").alias("overfit_observed_rate"),
            pl.mean("best_score").alias("mean_best_score"),
            pl.mean("final_score").alias("mean_final_score"),
            pl.mean("best_val_nll_qty_mae").alias("mean_best_nll_qty_mae"),
            pl.mean("final_qty_mae").alias("mean_final_qty_mae"),
            pl.mean("best_val_nll_mark_acc").alias("mean_best_nll_mark_acc"),
            pl.mean("final_mark_acc").alias("mean_final_mark_acc"),
            pl.mean("final_train_loss").alias("mean_final_train_loss"),
        ])
        .sort(["dataset_name", "model_name", "mean_best_val_nll"])
    )
    return diagnostic_df, summary_df


def save_overfit_curve_plots(history_df: pl.DataFrame, plots_dir: Path) -> None:
    """
    Plot train loss and decomposed validation NLL by configuration.
    """
    if history_df.height == 0:
        return

    ensure_dir(plots_dir)
    for dataset_name in history_df["dataset_name"].unique().to_list():
        for model_name in history_df["model_name"].unique().to_list():
            model_df = history_df.filter(
                (pl.col("dataset_name") == dataset_name)
                & (pl.col("model_name") == model_name)
            )
            if model_df.height == 0:
                continue

            # The split NLL curves reveal whether overfitting or continued
            # learning is driven by mark prediction or event-time likelihood.
            fig, axes = plt.subplots(2, 2, figsize=(16, 9))
            axes_flat = axes.flatten()
            for config_id in model_df["config_id"].unique().to_list():
                config_df = (
                    model_df.filter(pl.col("config_id") == config_id)
                    .group_by("epoch")
                    .agg([
                        pl.mean("train_loss").alias("train_loss"),
                        pl.mean("val_nll").alias("val_nll"),
                        pl.mean("val_nll_marker").alias("val_nll_marker"),
                        pl.mean("val_nll_time").alias("val_nll_time"),
                    ])
                    .sort("epoch")
                )
                epochs = config_df["epoch"].to_list()
                short_label = str(config_id).replace("rmtpp_", "").replace("titan_", "")
                axes_flat[0].plot(epochs, config_df["train_loss"].to_list(), label=short_label, linewidth=1.8)
                axes_flat[1].plot(epochs, config_df["val_nll"].to_list(), label=short_label, linewidth=1.8)
                axes_flat[2].plot(epochs, config_df["val_nll_marker"].to_list(), label=short_label, linewidth=1.8)
                axes_flat[3].plot(epochs, config_df["val_nll_time"].to_list(), label=short_label, linewidth=1.8)

            axes_flat[0].set_title("Train Loss")
            axes_flat[1].set_title("Validation NLL")
            axes_flat[2].set_title("Validation Marker NLL")
            axes_flat[3].set_title("Validation Time NLL")
            for ax in axes_flat:
                ax.set_xlabel("Epoch")
                ax.grid(alpha=0.25)
                ax.legend(fontsize=7)
            fig.suptitle(f"{dataset_name}: {model_name.upper()} overfit diagnostic", fontsize=14)
            fig.tight_layout(rect=(0, 0, 1, 0.94))
            fig.savefig(
                plots_dir / f"{dataset_name}_{model_name}_overfit_curves.png",
                dpi=200,
                bbox_inches="tight",
            )
            plt.close(fig)


def save_overfit_report(
    *,
    run_df: pl.DataFrame,
    summary_df: pl.DataFrame,
    output_path: Path,
) -> None:
    """
    Write a compact meeting-ready diagnostic report.
    """
    lines = [
        "# TPP Overfitting Diagnostic Report",
        "",
        "## Purpose",
        "",
        "This experiment intentionally increases learning pressure and model capacity to check whether RMTPP and TitanTPP can truly fit the data. The target pattern is clear train-loss decrease, followed by either validation plateau or validation degradation.",
        "",
        "## How To Read",
        "",
        "- `best_val_nll_epoch` shows the validation sweet spot.",
        "- `overfit_gap_val_nll = final_val_nll - best_val_nll`.",
        "- A positive overfit gap means validation got worse after its best point.",
        "- `overfit_observed=True` means the gap exceeded the configured threshold.",
        "",
        "## Config Summary",
        "",
        markdown_table_from_df(summary_df),
        "",
        "## Run-level Diagnostics",
        "",
        markdown_table_from_df(run_df),
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse overfit diagnostic settings.
    """
    parser = argparse.ArgumentParser(
        description="Run overfitting diagnostics for RMTPP/TitanTPP capacity checks."
    )
    parser.add_argument(
        "--base-dir",
        default=str(PROJECT_ROOT / "search_artifacts" / "tpp_overfit_diagnostic"),
        help="Directory where overfit diagnostic artifacts will be written.",
    )
    parser.add_argument(
        "--preset",
        default="custom",
        choices=["custom", "yellow_trip_full_long", "yellow_trip_subset_stress"],
        help=(
            "High-level diagnostic preset. Use yellow_trip_full_long for a longer "
            "full-data run, or yellow_trip_subset_stress to force overfitting on "
            "a smaller yellow-trip subset."
        ),
    )
    parser.add_argument("--datasets", default="intermittent,yellow_trip_hourly")
    parser.add_argument("--models", default="rmtpp,titantpp")
    parser.add_argument("--titan-profile", default="dataset_best", choices=["dataset_best", "overall", "score_priority"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--lookback-weeks", type=int, default=52)
    parser.add_argument("--max-seq-lens", default="250")
    parser.add_argument("--rmtpp-rnn-types", default="rnn,gru,lstm")
    parser.add_argument("--rmtpp-hidden-dims", default="64,128,256")
    parser.add_argument("--rmtpp-mark-emb-dims", default="32")
    parser.add_argument("--titan-candidates", default="small_lmm,mid_lmm,mid_deep_lmm")
    parser.add_argument("--titan-mark-emb-dims", default="32")
    parser.add_argument("--intermittent-max-series", type=int, default=None)
    parser.add_argument("--yellow-max-series", type=int, default=None)
    parser.add_argument("--overfit-gap-threshold", type=float, default=0.01)
    parser.add_argument("--force-rerun", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def main() -> None:
    """
    Execute the full overfit diagnostic grid.
    """
    args = parse_args()
    args = apply_experiment_preset(args)

    selected_dataset_names = {name.strip() for name in args.datasets.split(",") if name.strip()}
    selected_models = set(parse_str_list(args.models))
    seeds = parse_int_list(args.seeds)
    max_seq_lens = parse_int_list(args.max_seq_lens)
    rmtpp_rnn_types = parse_str_list(args.rmtpp_rnn_types)
    rmtpp_hidden_dims = parse_int_list(args.rmtpp_hidden_dims)
    rmtpp_mark_emb_dims = parse_int_list(args.rmtpp_mark_emb_dims)
    titan_candidate_names = parse_str_list(args.titan_candidates)
    titan_mark_emb_dims = parse_int_list(args.titan_mark_emb_dims)

    base_cfg = LongEpochConfig(
        base_dir=args.base_dir,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seeds=seeds,
        lookback_weeks=args.lookback_weeks,
        max_seq_len=max_seq_lens[0],
        titan_profile=args.titan_profile,
        intermittent_max_series=args.intermittent_max_series,
        yellow_max_series=args.yellow_max_series,
        eval_selections=("best_val_nll", "final"),
        force_rerun=args.force_rerun,
        stop_on_error=args.stop_on_error,
    )

    base_dir = ensure_dir(Path(base_cfg.base_dir))
    leaderboard_dir = ensure_dir(base_dir / "leaderboard")
    paper_dir = ensure_dir(base_dir / "paper_outputs")
    plots_dir = ensure_dir(paper_dir / "plots")
    logger = build_logger(base_dir / "overfit_diagnostic.log", "tpp_overfit_diagnostic")

    dataset_specs = make_dataset_specs(base_cfg, selected_dataset_names)
    if not dataset_specs:
        raise ValueError("No datasets selected for overfit diagnostic.")

    profile_map = default_profile_map(base_cfg.titan_profile)
    profile_map = {name: profile_map[name] for name in selected_dataset_names}
    all_titan_candidates = default_titan_candidates()

    save_json(
        {
            "preset": args.preset,
            "overfit_config": base_cfg,
            "selected_models": sorted(selected_models),
            "max_seq_lens": max_seq_lens,
            "marked_target_default_max_seq_len": MARKED_TARGET_MAX_SEQ_LEN,
            "marked_target_default_titan_candidates": sorted(MARKED_TARGET_TITAN_CANDIDATES),
            "rmtpp_rnn_types": rmtpp_rnn_types,
            "rmtpp_hidden_dims": rmtpp_hidden_dims,
            "rmtpp_mark_emb_dims": rmtpp_mark_emb_dims,
            "titan_candidate_names": titan_candidate_names,
            "titan_mark_emb_dims": titan_mark_emb_dims,
            "profile_map": profile_map,
        },
        base_dir / "overfit_manifest.json",
    )

    logger.info("Preparing marked datasets for overfit diagnostic")
    marked_cache = build_marked_cache(
        dataset_specs=dataset_specs,
        profile_map=profile_map,
        ab_cfg=base_cfg,
        logger=logger,
    )

    rows: list[dict[str, Any]] = []
    total_planned = 0
    for _spec in dataset_specs:
        spec_max_seq_lens = max_seq_lens_for_spec(_spec, max_seq_lens)
        spec_titan_candidate_names = titan_candidate_names_for_spec(_spec, titan_candidate_names)
        for _seed in seeds:
            for _seq_len in spec_max_seq_lens:
                if "rmtpp" in selected_models:
                    total_planned += len(rmtpp_rnn_types) * len(rmtpp_hidden_dims) * len(rmtpp_mark_emb_dims)
                if "titantpp" in selected_models:
                    total_planned += len(spec_titan_candidate_names) * len(titan_mark_emb_dims)

    completed = 0
    for spec in dataset_specs:
        profile = profile_map[spec.name]
        scale_base = float(profile["scale_base"])
        marked_df, marked_meta = marked_cache[(spec.name, scale_base)]
        spec_max_seq_lens = max_seq_lens_for_spec(spec, max_seq_lens)
        spec_titan_candidate_names = titan_candidate_names_for_spec(spec, titan_candidate_names)

        for seed in seeds:
            for max_seq_len in spec_max_seq_lens:
                if "rmtpp" in selected_models:
                    for rnn_type in rmtpp_rnn_types:
                        for mark_emb_dim in rmtpp_mark_emb_dims:
                            for hidden_dim in rmtpp_hidden_dims:
                                completed += 1
                                cfg = replace(
                                    base_cfg,
                                    max_seq_len=max_seq_len,
                                    rmtpp_rnn_type=rnn_type,
                                    rmtpp_mark_emb_dim=mark_emb_dim,
                                )
                                candidate = make_rmtpp_capacity_candidate(
                                    rnn_type=rnn_type,
                                    mark_emb_dim=mark_emb_dim,
                                    hidden_dim=hidden_dim,
                                    max_seq_len=max_seq_len,
                                    lr=args.lr,
                                )
                                run_cfg = LongRunConfig(
                                    dataset_name=spec.name,
                                    dataset_kind=spec.kind,
                                    model_name="rmtpp",
                                    seed=seed,
                                    epochs=args.epochs,
                                    scale_base=scale_base,
                                    titan_profile=args.titan_profile,
                                    titan_candidate_name=candidate.name,
                                    titan_candidate=candidate,
                                )
                                logger.info(
                                    "Overfit run %s/%s | dataset=%s model=rmtpp rnn=%s emb=%s hidden=%s seq=%s seed=%s",
                                    completed,
                                    total_planned,
                                    spec.name,
                                    rnn_type,
                                    mark_emb_dim,
                                    hidden_dim,
                                    max_seq_len,
                                    seed,
                                )
                                try:
                                    row = train_one_long_run(
                                        long_cfg=cfg,
                                        run_cfg=run_cfg,
                                        marked_df=marked_df,
                                        marked_meta=marked_meta,
                                        logger=logger,
                                    )
                                except Exception as exc:
                                    row = build_error_row(run_cfg, exc)
                                    logger.exception("RMTPP overfit run failed")
                                    if args.stop_on_error:
                                        raise
                                rows.append(row)
                                persist_rows(rows, leaderboard_dir / "overfit_runs")

                if "titantpp" in selected_models:
                    for mark_emb_dim in titan_mark_emb_dims:
                        for candidate_name in spec_titan_candidate_names:
                            completed += 1
                            base_candidate = find_candidate_by_name(all_titan_candidates, candidate_name)
                            candidate = make_titan_capacity_candidate(
                                base_candidate=base_candidate,
                                mark_emb_dim=mark_emb_dim,
                                max_seq_len=max_seq_len,
                                lr=args.lr,
                            )
                            cfg = replace(
                                base_cfg,
                                max_seq_len=max_seq_len,
                                rmtpp_mark_emb_dim=mark_emb_dim,
                            )
                            run_cfg = LongRunConfig(
                                dataset_name=spec.name,
                                dataset_kind=spec.kind,
                                model_name="titantpp",
                                seed=seed,
                                epochs=args.epochs,
                                scale_base=scale_base,
                                titan_profile=args.titan_profile,
                                titan_candidate_name=candidate.name,
                                titan_candidate=candidate,
                            )
                            logger.info(
                                "Overfit run %s/%s | dataset=%s model=titantpp candidate=%s emb=%s seq=%s seed=%s",
                                completed,
                                total_planned,
                                spec.name,
                                candidate_name,
                                mark_emb_dim,
                                max_seq_len,
                                seed,
                            )
                            try:
                                row = train_one_long_run(
                                    long_cfg=cfg,
                                    run_cfg=run_cfg,
                                    marked_df=marked_df,
                                    marked_meta=marked_meta,
                                    logger=logger,
                                )
                            except Exception as exc:
                                row = build_error_row(run_cfg, exc)
                                logger.exception("TitanTPP overfit run failed")
                                if args.stop_on_error:
                                    raise
                            rows.append(row)
                            persist_rows(rows, leaderboard_dir / "overfit_runs")

    run_df, summary_df = build_overfit_tables(
        run_rows=rows,
        overfit_gap_threshold=float(args.overfit_gap_threshold),
    )
    history_df = load_overfit_histories(rows)

    if run_df.width > 0:
        run_df.write_csv(leaderboard_dir / "overfit_runs.csv")
        run_df.write_parquet(leaderboard_dir / "overfit_runs.parquet")
    if summary_df.width > 0:
        summary_df.write_csv(leaderboard_dir / "overfit_summary.csv")
        summary_df.write_parquet(leaderboard_dir / "overfit_summary.parquet")
        summary_df.write_csv(paper_dir / "paper_table_overfit_summary.csv")
    if history_df.width > 0:
        history_df.write_csv(leaderboard_dir / "overfit_histories.csv")
        history_df.write_parquet(leaderboard_dir / "overfit_histories.parquet")

    save_overfit_curve_plots(history_df, plots_dir)
    save_overfit_report(
        run_df=run_df,
        summary_df=summary_df,
        output_path=paper_dir / "overfit_diagnostic_report.md",
    )

    logger.info("Overfit diagnostic complete. Summary rows:\n%s", summary_df)


if __name__ == "__main__":
    main()
