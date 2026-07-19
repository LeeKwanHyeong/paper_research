#!/usr/bin/env python3
"""Build the evidence figures used in the July 2026 advisor meeting report."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


INK = "#16324F"
TEAL = "#1B998B"
CORAL = "#D95D39"
AMBER = "#E9B44C"
SLATE = "#667085"
PALE = "#F7F4ED"
WHITE = "#FFFFFF"
GRID = "#D8D5CC"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "reports" / "advisor_meeting_after_2026_06_28",
    )
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "AppleGothic",
            "axes.unicode_minus": False,
            "figure.facecolor": PALE,
            "axes.facecolor": PALE,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": SLATE,
            "ytick.color": SLATE,
            "grid.color": GRID,
            "grid.alpha": 0.65,
            # Keep text as text so the figures remain small enough for Notion uploads.
            "svg.fonttype": "none",
        }
    )


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[Path]:
    paths = []
    for suffix, kwargs in (
        (".png", {"dpi": 220}),
        (".svg", {}),
    ):
        path = output_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor(), **kwargs)
        if suffix == ".svg":
            svg = path.read_text(encoding="utf-8")
            path.write_text(
                "\n".join(line.rstrip() for line in svg.splitlines()) + "\n",
                encoding="utf-8",
            )
        paths.append(path)
    plt.close(fig)
    return paths


def bool_value(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def percent_change(new: float, old: float) -> float:
    return (new - old) / old * 100.0


def relative_path(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_f1(root: Path, output_dir: Path, tables_dir: Path) -> tuple[list[Path], list[Path]]:
    rmtpp_path = (
        root
        / "search_artifacts"
        / "insta_rmtpp_lr_sensitivity_e50"
        / "rmtpp_lr_sensitivity_aggregated.csv"
    )
    titan_path = (
        root
        / "search_artifacts"
        / "insta_lr_sensitivity_e50"
        / "lr_sensitivity_aggregated_with_nll_split.csv"
    )
    rmtpp = pd.read_csv(rmtpp_path)
    titan = pd.read_csv(titan_path)

    rmtpp_table = rmtpp.assign(
        model="RMTPP",
        status=rmtpp["stable_no_nan"].map(lambda value: "stable" if bool_value(value) else "nan"),
    )
    titan_table = titan.assign(
        model="TitanTPP",
        status=titan["stable_no_nan"].map(lambda value: "stable" if bool_value(value) else "nan"),
    )
    columns = [
        "model",
        "variant",
        "candidate",
        "lr",
        "status",
        "first_nan_epoch",
        "best_val_nll_epoch",
        "best_val_nll",
        "best_score",
    ]
    diagnostic = pd.concat(
        [rmtpp_table.reindex(columns=columns), titan_table.reindex(columns=columns)],
        ignore_index=True,
    )
    table_path = tables_dir / "F1_learning_rate_diagnostic.csv"
    diagnostic.to_csv(table_path, index=False)

    lr_order = [0.001, 0.005, 0.01]
    lr_labels = ["1e-3", "5e-3", "1e-2"]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2), gridspec_kw={"width_ratios": [0.8, 1.7]})
    fig.suptitle(
        "F1. 학습률을 높이면 수렴은 빨라질 수 있지만 안정성은 보장되지 않는다",
        fontsize=17,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    fig.text(
        0.04,
        0.925,
        "Instacart, e50, seed 42 진단 실험. 원은 정상 완료, X는 최초 NaN 발생 시점이다.",
        fontsize=10.5,
        color=SLATE,
    )

    ax = axes[0]
    rmtpp_plot = rmtpp.set_index("lr").loc[lr_order].reset_index()
    bars = ax.bar(
        np.arange(len(lr_order)),
        rmtpp_plot["best_val_nll_epoch"],
        color=[INK, TEAL, AMBER],
        width=0.62,
    )
    for bar, epoch in zip(bars, rmtpp_plot["best_val_nll_epoch"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"best {int(epoch)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_title("RMTPP: 세 설정 모두 정상 완료", loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(lr_labels)), lr_labels)
    ax.set_ylabel("Best validation NLL epoch")
    ax.set_ylim(0, max(rmtpp_plot["best_val_nll_epoch"]) + 10)
    ax.grid(axis="y")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    variant_order = ["value_conditioned", "value_conditioned_hybrid"]
    candidate_order = ["small_lmm", "mid_lmm"]
    row_specs = [
        (variant, candidate)
        for variant in variant_order
        for candidate in candidate_order
    ]
    row_labels = [
        "Residual-only / small",
        "Residual-only / mid",
        "Hybrid / small",
        "Hybrid / mid",
    ]
    for y, (variant, candidate) in enumerate(row_specs):
        subset = titan[(titan["variant"] == variant) & (titan["candidate"] == candidate)]
        for x, lr in enumerate(lr_order):
            row = subset[np.isclose(subset["lr"], lr)].iloc[0]
            stable = bool_value(row["stable_no_nan"])
            if stable:
                ax.scatter(x, y, s=185, marker="o", color=TEAL, edgecolor=INK, linewidth=1.1, zorder=3)
                label = f"best {int(row['best_val_nll_epoch'])}"
            else:
                ax.scatter(x, y, s=190, marker="X", color=CORAL, edgecolor=INK, linewidth=0.8, zorder=3)
                label = f"NaN {int(row['first_nan_epoch'])}"
            ax.text(x, y + 0.28, label, ha="center", va="bottom", fontsize=8.7)
    ax.set_title("TitanTPP: candidate와 loss에 따라 NaN 발생", loc="left", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(lr_labels)), lr_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-0.55, len(row_labels) - 0.35)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.04,
        0.025,
        "해석: 5e-3은 RMTPP의 best epoch를 26에서 5로 앞당겼지만, TitanTPP에서는 같은 학습률도 설정별로 안정성이 달랐다.",
        fontsize=10.5,
        color=INK,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.83, bottom=0.14, wspace=0.3)
    return save_figure(fig, output_dir, "F1_learning_rate_stability"), [rmtpp_path, titan_path, table_path]


def build_f2(root: Path, output_dir: Path, tables_dir: Path) -> tuple[list[Path], list[Path]]:
    history_path = (
        root
        / "search_artifacts"
        / "nll_decomposition_yellow_overfit_e1000"
        / "leaderboard"
        / "overfit_histories.csv"
    )
    history = pd.read_csv(history_path)
    configs = [
        ("RMTPP", "rmtpp_gru_emb32_h128_seq250_lr1em03"),
        ("TitanTPP", "titan_mid_lmm_emb32_seq250_lr1em03"),
    ]
    summary_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 6.4), sharex=True)
    fig.suptitle(
        "F2. Total NLL 감소만 보면 marker 성능의 변화를 놓칠 수 있다",
        fontsize=17,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    fig.text(
        0.04,
        0.925,
        "Taxi e1000, seed 42, lr 1e-3. Hidden width와 mark embedding을 128/32로 맞춘 대표 진단 설정.",
        fontsize=10.5,
        color=SLATE,
    )

    line_specs = [
        ("train_loss", "Train loss", SLATE, 1.5, "--"),
        ("val_nll", "Validation total NLL", INK, 2.2, "-"),
        ("val_nll_marker", "Marker NLL", CORAL, 1.8, "-"),
        ("val_nll_time", "Time NLL", TEAL, 1.8, "-"),
    ]
    for ax, (label, config_id) in zip(axes, configs):
        frame = history[history["config_id"] == config_id].sort_values("epoch")
        if frame.empty:
            raise RuntimeError(f"Missing e1000 history for {config_id}")
        best_index = frame["val_nll"].idxmin()
        best = frame.loc[best_index]
        final = frame.iloc[-1]
        for column, line_label, color, width, style in line_specs:
            ax.plot(
                frame["epoch"],
                frame[column],
                label=line_label,
                color=color,
                linewidth=width,
                linestyle=style,
                alpha=0.95,
            )
        ax.axvline(best["epoch"], color=AMBER, linestyle=":", linewidth=2)
        ax.text(
            best["epoch"],
            ax.get_ylim()[1],
            f" best epoch {int(best['epoch'])}",
            color=AMBER,
            fontsize=9.5,
            va="top",
            ha="left",
        )
        ax.set_title(label, loc="left", fontsize=13, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        summary_rows.append(
            {
                "model": label,
                "config_id": config_id,
                "best_epoch": int(best["epoch"]),
                "best_val_nll": float(best["val_nll"]),
                "final_val_nll": float(final["val_nll"]),
                "overfit_gap": float(final["val_nll"] - best["val_nll"]),
                "best_marker_nll": float(best["val_nll_marker"]),
                "final_marker_nll": float(final["val_nll_marker"]),
                "best_time_nll": float(best["val_nll_time"]),
                "final_time_nll": float(final["val_nll_time"]),
            }
        )
    axes[0].set_ylabel("Loss / NLL")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.035))
    fig.text(
        0.04,
        0.005,
        "Time NLL이 음수로 크게 내려가 total NLL을 지배할 수 있으므로 marker NLL, mark accuracy, quantity MAE를 함께 봐야 한다.",
        fontsize=10.5,
        color=INK,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.83, bottom=0.18, wspace=0.22)
    table_path = tables_dir / "F2_e1000_selected_config_summary.csv"
    pd.DataFrame(summary_rows).to_csv(table_path, index=False)
    return save_figure(fig, output_dir, "F2_e1000_nll_decomposition"), [history_path, table_path]


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str,
    edgecolor: str = INK,
    fontsize: float = 10,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        linewidth=1.3,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        wrap=True,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float], **kwargs: object) -> None:
    defaults = {"arrowstyle": "-|>", "color": INK, "linewidth": 1.5}
    defaults.update(kwargs)
    ax.annotate("", xy=end, xytext=start, arrowprops=defaults)


def build_f3(root: Path, output_dir: Path) -> tuple[list[Path], list[Path]]:
    source_paths = [
        root / "models" / "RMTPPs" / "RMTPP.py",
        root / "models" / "RMTPPs" / "TitanTPP.py",
        root / ".agents" / "results" / "architecture" / "titantpp-model-status-baseline-registry.md",
    ]
    fig, ax = plt.subplots(figsize=(16, 7.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle(
        "F3. RMTPP의 확률적 예측 계약은 유지하고 history encoder와 quantity branch를 확장했다",
        fontsize=17,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    fig.text(
        0.04,
        0.925,
        "V3b는 V2와 같은 Titan + static LMM을 사용하며, Taxi에서 value head와 quantity gradient route만 변경한다.",
        fontsize=10.5,
        color=SLATE,
    )

    columns = [
        (0.04, "RMTPP", "Recurrent reference", "GRU history encoder", "Shared value residual", WHITE),
        (0.36, "TitanTPP V2", "Common incumbent", "Titan encoder + static LMM", "Shared value residual\n+ hybrid quantity loss", "#E9F6F3"),
        (0.68, "TitanTPP V3b", "Taxi-specific incumbent", "Titan encoder + static LMM", "Mark-conditioned value experts", "#FFF1EB"),
    ]
    for x, title, role, encoder, value_head, face in columns:
        ax.text(x + 0.13, 0.83, title, ha="center", fontsize=14, fontweight="bold")
        ax.text(x + 0.13, 0.79, role, ha="center", fontsize=9.5, color=SLATE)
        add_box(ax, x, 0.64, 0.26, 0.09, "Observed event history\nmark, delta-time, past quantity", facecolor=PALE, fontsize=9.5)
        add_box(ax, x, 0.46, 0.26, 0.10, encoder, facecolor=face, fontsize=10.5, weight="bold")
        add_box(ax, x, 0.27, 0.12, 0.10, "Mark head\nCE", facecolor=WHITE, fontsize=9.5)
        add_box(ax, x + 0.14, 0.27, 0.12, 0.10, "RMTPP time head\nNLL", facecolor=WHITE, fontsize=9.5)
        add_box(ax, x, 0.09, 0.26, 0.10, value_head, facecolor=face, fontsize=9.5)
        arrow(ax, (x + 0.13, 0.64), (x + 0.13, 0.56))
        arrow(ax, (x + 0.13, 0.46), (x + 0.06, 0.37))
        arrow(ax, (x + 0.13, 0.46), (x + 0.20, 0.37))
        arrow(ax, (x + 0.13, 0.27), (x + 0.13, 0.19))

    ax.annotate(
        "same probabilistic heads",
        xy=(0.65, 0.33),
        xytext=(0.31, 0.33),
        arrowprops={"arrowstyle": "<->", "color": TEAL, "linewidth": 2},
        ha="center",
        va="bottom",
        color=TEAL,
        fontsize=9.5,
        fontweight="bold",
    )
    ax.plot([0.75, 0.75], [0.20, 0.27], color=CORAL, linewidth=2.2, linestyle="--")
    ax.text(0.755, 0.225, "gradient stop", color=CORAL, fontsize=9.2, va="center", fontweight="bold")
    ax.text(
        0.5,
        0.015,
        "현재 입력은 관측된 사건 이력이다. 날씨, 공휴일, 프로모션 같은 외생변수는 아직 포함하지 않았다.",
        ha="center",
        fontsize=10.5,
        color=INK,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.86, bottom=0.08)
    return save_figure(fig, output_dir, "F3_model_architecture_evolution"), source_paths


def select_variant_metrics(root: Path, artifact_name: str, variant: str) -> pd.DataFrame:
    base = root / "search_artifacts" / artifact_name / "leaderboard"
    summary = pd.read_csv(base / "summary.csv")
    test = pd.read_csv(base / "test_summary.csv")
    rows = []
    for dataset, candidates in summary.groupby("dataset_name"):
        selected = candidates.loc[candidates["mean_best_val_nll"].idxmin()]
        metric = test[
            (test["dataset_name"] == dataset)
            & (test["candidate_name"] == selected["candidate_name"])
            & (test["selection"] == "best_val_nll")
        ].iloc[0]
        row = metric.to_dict()
        row["variant"] = variant
        row["selected_candidate"] = selected["candidate_name"]
        row["candidate_selection_val_nll"] = selected["mean_best_val_nll"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_f4(root: Path, output_dir: Path, tables_dir: Path) -> tuple[list[Path], list[Path]]:
    v1_name = "model_enhancement_v1_residual_e200_0705"
    v2_name = "model_enhancement_v2_hybrid_e200_0705"
    v1 = select_variant_metrics(root, v1_name, "V1")
    v2 = select_variant_metrics(root, v2_name, "V2")
    merged = v1.merge(v2, on="dataset_name", suffixes=("_v1", "_v2"))
    merged["qty_mae_change_pct"] = percent_change(
        merged["mean_test_qty_mae_v2"], merged["mean_test_qty_mae_v1"]
    )
    merged["nll_change_pct"] = percent_change(
        merged["mean_test_nll_v2"], merged["mean_test_nll_v1"]
    )
    merged["score_change"] = merged["mean_test_score_v2"] - merged["mean_test_score_v1"]
    keep = [
        "dataset_name",
        "selected_candidate_v1",
        "selected_candidate_v2",
        "mean_test_score_v1",
        "mean_test_score_v2",
        "score_change",
        "mean_test_nll_v1",
        "mean_test_nll_v2",
        "nll_change_pct",
        "mean_test_qty_mae_v1",
        "mean_test_qty_mae_v2",
        "qty_mae_change_pct",
        "mean_test_mark_acc_v1",
        "mean_test_mark_acc_v2",
    ]
    table = merged[keep].copy()
    table_path = tables_dir / "F4_v1_v2_selected_metrics.csv"
    table.to_csv(table_path, index=False)

    order = ["insta_market_basket", "intermittent", "yellow_trip_hourly"]
    labels = {"insta_market_basket": "Instacart", "intermittent": "Demand", "yellow_trip_hourly": "Taxi"}
    plot = table.set_index("dataset_name").loc[order].reset_index()
    qty_improvement = -plot["qty_mae_change_pct"].to_numpy()
    nll_change = plot["nll_change_pct"].to_numpy()
    y = np.arange(len(plot))

    fig, axes = plt.subplots(1, 2, figsize=(14.8, 6.3), gridspec_kw={"width_ratios": [1.15, 1]})
    fig.suptitle(
        "F4. V2는 V1 대비 세 데이터셋에서 quantity MAE를 낮췄다",
        fontsize=17,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    fig.text(
        0.04,
        0.925,
        "e200, seeds 42/52/62, fixed split, best validation NLL checkpoint. Candidate는 variant별 validation NLL로 선택했다.",
        fontsize=10.5,
        color=SLATE,
    )

    ax = axes[0]
    comparison_labels = [
        (
            f"{labels[dataset]}\n"
            f"{plot.loc[index, 'selected_candidate_v1']} -> "
            f"{plot.loc[index, 'selected_candidate_v2']}"
        )
        for index, dataset in enumerate(order)
    ]
    bars = ax.barh(y, qty_improvement, color=[INK, TEAL, CORAL], height=0.56)
    for index, (bar, value) in enumerate(zip(bars, qty_improvement)):
        ax.text(value + 0.45, bar.get_y() + bar.get_height() / 2, f"{value:.2f}%", va="center", fontweight="bold")
    ax.set_yticks(y, comparison_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Quantity MAE improvement (%)")
    ax.set_xlim(0, max(qty_improvement) + 5)
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax = axes[1]
    colors = [TEAL if value < 0 else CORAL for value in nll_change]
    bars = ax.barh(y, nll_change, color=colors, height=0.56)
    for index, (bar, value) in enumerate(zip(bars, nll_change)):
        if abs(value) >= 0.15:
            text_x, ha, color = value / 2, "center", WHITE
        else:
            text_x, ha, color = value + 0.05, "left", INK
        ax.text(
            text_x,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.2f}%",
            va="center",
            ha=ha,
            color=color,
            fontweight="bold",
        )
    ax.axvline(0, color=INK, linewidth=1)
    score_labels = [
        f"{labels[dataset]}\nscore {plot.loc[index, 'score_change']:+.6f}"
        for index, dataset in enumerate(order)
    ]
    ax.set_yticks(y, score_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Test NLL change (%)  [negative is better]")
    ax.grid(axis="x")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.text(
        0.04,
        0.025,
        "주의: Instacart와 Taxi는 V1/V2의 선택 candidate가 다르므로, 이 그림은 pure head ablation이 아니라 validation-selected system 비교다.",
        fontsize=10.2,
        color=INK,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.83, bottom=0.16, wspace=0.3)
    source_paths = [
        root / "search_artifacts" / v1_name / "experiment_manifest.json",
        root / "search_artifacts" / v1_name / "leaderboard" / "summary.csv",
        root / "search_artifacts" / v1_name / "leaderboard" / "test_summary.csv",
        root / "search_artifacts" / v2_name / "experiment_manifest.json",
        root / "search_artifacts" / v2_name / "leaderboard" / "summary.csv",
        root / "search_artifacts" / v2_name / "leaderboard" / "test_summary.csv",
        table_path,
    ]
    return save_figure(fig, output_dir, "F4_v1_v2_dataset_comparison"), source_paths


def build_f5(root: Path, output_dir: Path, tables_dir: Path) -> tuple[list[Path], list[Path]]:
    paths = {
        "V2": root
        / "search_artifacts"
        / "model_enhancement_v2_taxi_multiseed_e50_0710"
        / "leaderboard"
        / "test_metrics.csv",
        "V3b": root
        / "search_artifacts"
        / "model_enhancement_v3b_taxi_multiseed_e50_0710"
        / "leaderboard"
        / "test_metrics.csv",
    }
    frames = []
    for model, path in paths.items():
        frame = pd.read_csv(path)
        frame = frame[frame["selection"] == "best_val_nll"].copy()
        frame["variant"] = model
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    table_columns = [
        "variant",
        "seed",
        "score",
        "val_nll",
        "val_nll_marker",
        "val_nll_time",
        "qty_mae",
        "value_mae",
        "dt_mae",
        "mark_acc",
    ]
    table = metrics[table_columns].sort_values(["seed", "variant"])
    table_path = tables_dir / "F5_taxi_v2_v3b_seed_metrics.csv"
    table.to_csv(table_path, index=False)

    means = table.groupby("variant", as_index=True).mean(numeric_only=True)
    changes = {
        "total_nll": percent_change(means.loc["V3b", "val_nll"], means.loc["V2", "val_nll"]),
        "marker_nll": percent_change(
            means.loc["V3b", "val_nll_marker"], means.loc["V2", "val_nll_marker"]
        ),
        "time_nll": percent_change(
            means.loc["V3b", "val_nll_time"], means.loc["V2", "val_nll_time"]
        ),
        "qty_mae": percent_change(means.loc["V3b", "qty_mae"], means.loc["V2", "qty_mae"]),
        "mark_acc_pp": (means.loc["V3b", "mark_acc"] - means.loc["V2", "mark_acc"]) * 100,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14.8, 9.0))
    fig.suptitle(
        "F5. Taxi V3b의 marker와 quantity 개선은 세 seed에서 같은 방향으로 나타났다",
        fontsize=17,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    fig.text(
        0.04,
        0.943,
        "V2/V3b 모두 mid_lmm, e50, seeds 42/52/62, fixed split, best validation NLL checkpoint.",
        fontsize=10.5,
        color=SLATE,
    )
    specs = [
        ("val_nll", "Total NLL", "lower"),
        ("val_nll_marker", "Marker NLL", "lower"),
        ("qty_mae", "Quantity MAE", "lower"),
        ("mark_acc", "Mark accuracy", "higher"),
    ]
    for ax, (column, title, direction) in zip(axes.flat, specs):
        for variant, color, marker in (("V2", SLATE, "o"), ("V3b", TEAL, "s")):
            frame = table[table["variant"] == variant].sort_values("seed")
            ax.plot(
                frame["seed"].astype(str),
                frame[column],
                color=color,
                marker=marker,
                linewidth=2,
                markersize=7,
                label=f"{variant} mean {frame[column].mean():.4f}",
            )
        ax.set_title(f"{title} ({direction} is better)", loc="left", fontsize=12, fontweight="bold")
        ax.grid(axis="y")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.8, loc="best")
        ax.set_xlabel("Seed")
    fig.text(
        0.04,
        0.025,
        (
            f"Mean change: total NLL {changes['total_nll']:+.3f}%, marker NLL {changes['marker_nll']:+.3f}%, "
            f"quantity MAE {changes['qty_mae']:+.3f}%, mark accuracy {changes['mark_acc_pp']:+.3f}%p. "
            f"Time NLL은 {changes['time_nll']:+.3f}%로 소폭 악화됐다."
        ),
        fontsize=10.2,
        color=INK,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.87, bottom=0.11, hspace=0.35, wspace=0.23)
    return save_figure(fig, output_dir, "F5_taxi_v2_v3b_multiseed"), [*paths.values(), table_path]


def build_f6(root: Path, output_dir: Path) -> tuple[list[Path], list[Path]]:
    registry = root / ".agents" / "results" / "architecture" / "titantpp-model-status-baseline-registry.md"
    fig, ax = plt.subplots(figsize=(16, 8.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.suptitle(
        "F6. Gate를 통과한 모델만 데이터셋별 incumbent로 유지했다",
        fontsize=17,
        fontweight="bold",
        x=0.04,
        ha="left",
    )
    fig.text(
        0.04,
        0.93,
        "초록은 현재 비교 기준, 주황은 설계만 동결, 붉은색은 미승격 또는 종료된 가설이다.",
        fontsize=10.5,
        color=SLATE,
    )

    add_box(ax, 0.04, 0.72, 0.14, 0.10, "V1\nreference", facecolor=WHITE, fontsize=10, weight="bold")
    add_box(ax, 0.25, 0.72, 0.18, 0.10, "V2\nACTIVE BASELINE", facecolor="#DDF2EA", edgecolor=TEAL, fontsize=10, weight="bold")
    arrow(ax, (0.18, 0.77), (0.25, 0.77))

    ax.text(0.05, 0.62, "Intermittent", fontsize=12, fontweight="bold")
    add_box(ax, 0.17, 0.57, 0.16, 0.09, "V3c / V5a\nnot promoted", facecolor="#FBE3DC", edgecolor=CORAL, fontsize=9.3)
    add_box(ax, 0.39, 0.57, 0.16, 0.09, "M0 / Q0-Q3\nclosed", facecolor="#FBE3DC", edgecolor=CORAL, fontsize=9.3)
    add_box(ax, 0.61, 0.57, 0.17, 0.09, "V5b\ndesign frozen", facecolor="#FFF0CC", edgecolor=AMBER, fontsize=9.3)
    arrow(ax, (0.34, 0.72), (0.25, 0.66), color=SLATE)
    arrow(ax, (0.33, 0.615), (0.39, 0.615), color=CORAL)
    arrow(ax, (0.55, 0.615), (0.61, 0.615), color=AMBER)
    add_box(ax, 0.82, 0.57, 0.14, 0.09, "V2 retained", facecolor="#DDF2EA", edgecolor=TEAL, fontsize=9.3, weight="bold")
    arrow(ax, (0.78, 0.615), (0.82, 0.615), color=TEAL)

    ax.text(0.05, 0.43, "Taxi", fontsize=12, fontweight="bold")
    add_box(ax, 0.17, 0.38, 0.14, 0.09, "V3a\nablation", facecolor=WHITE, fontsize=9.3)
    add_box(ax, 0.37, 0.38, 0.17, 0.09, "V3b\nPROMOTED", facecolor="#DDF2EA", edgecolor=TEAL, fontsize=9.3, weight="bold")
    add_box(ax, 0.60, 0.38, 0.14, 0.09, "V4\nnot promoted", facecolor="#FBE3DC", edgecolor=CORAL, fontsize=9.3)
    add_box(ax, 0.80, 0.38, 0.16, 0.09, "V6 / V7\nclosed", facecolor="#FBE3DC", edgecolor=CORAL, fontsize=9.3)
    arrow(ax, (0.34, 0.72), (0.24, 0.47), color=SLATE)
    arrow(ax, (0.31, 0.425), (0.37, 0.425), color=TEAL)
    arrow(ax, (0.54, 0.425), (0.60, 0.425), color=CORAL)
    arrow(ax, (0.74, 0.425), (0.80, 0.425), color=CORAL)

    ax.text(0.05, 0.24, "Instacart", fontsize=12, fontweight="bold")
    add_box(ax, 0.25, 0.18, 0.18, 0.09, "V2\nretained", facecolor="#DDF2EA", edgecolor=TEAL, fontsize=9.5, weight="bold")
    add_box(ax, 0.55, 0.18, 0.26, 0.09, "Later e1 smoke tests\nintegration evidence only", facecolor=WHITE, fontsize=9.2)
    arrow(ax, (0.34, 0.72), (0.34, 0.27), color=SLATE)
    arrow(ax, (0.43, 0.225), (0.55, 0.225), color=SLATE)

    ax.text(
        0.5,
        0.06,
        "결론: 공통 기준은 V2, Taxi의 dataset-specific incumbent는 V3b다. V5b는 아직 구현 또는 성능 근거가 없다.",
        ha="center",
        fontsize=10.5,
        fontweight="bold",
    )
    fig.subplots_adjust(top=0.87, bottom=0.05)
    return save_figure(fig, output_dir, "F6_model_selection_flow"), [registry]


def write_final_comparison_contract(tables_dir: Path) -> Path:
    rows = [
        {
            "dataset": "intermittent",
            "rmtpp_original": "GRU h64, value_input=none, residual_only",
            "rmtpp_matched": "GRU h64, value_input=residual, hybrid",
            "thp_matched": "small d64, value_input=residual, hybrid",
            "titantpp_control": "V2 small_lmm",
            "titantpp_primary": "V2 small_lmm",
            "lookback": 52,
            "max_seq_len": 16,
        },
        {
            "dataset": "yellow_trip_hourly",
            "rmtpp_original": "GRU h128, value_input=none, residual_only",
            "rmtpp_matched": "GRU h128, value_input=residual, hybrid",
            "thp_matched": "base d128, value_input=residual, hybrid",
            "titantpp_control": "V2 mid_lmm",
            "titantpp_primary": "V3b mid_lmm",
            "lookback": 168,
            "max_seq_len": 256,
        },
        {
            "dataset": "insta_market_basket",
            "rmtpp_original": "GRU h64, value_input=none, residual_only",
            "rmtpp_matched": "GRU h64, value_input=residual, hybrid",
            "thp_matched": "small d64, value_input=residual, hybrid",
            "titantpp_control": "V2 small_lmm",
            "titantpp_primary": "V2 small_lmm",
            "lookback": 52,
            "max_seq_len": 64,
        },
    ]
    contract = pd.DataFrame(rows)
    contract["split"] = "fixed train/validation/test"
    contract["candidate_rule"] = "pre-declared capacity-matched identity; no test-based selection"
    contract["checkpoint_rule"] = "best validation total NLL; best_score/final diagnostic only"
    contract["epochs"] = 800
    contract["seeds"] = "42,52,62"
    contract["learning_rate"] = 0.001
    contract["batch_size"] = 128
    contract["value_head_activation"] = "identity"
    contract["train_loss_scope"] = "target_only"
    contract["reporting"] = "mean +/- std; test opened after configuration freeze"
    path = tables_dir / "final_rmtpp_titantpp_thp_comparison_contract.csv"
    contract.to_csv(path, index=False)
    return path


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    tables_dir = output_dir / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()

    outputs: list[Path] = []
    sources: list[Path] = []
    for builder in (
        lambda: build_f1(root, output_dir, tables_dir),
        lambda: build_f2(root, output_dir, tables_dir),
        lambda: build_f3(root, output_dir),
        lambda: build_f4(root, output_dir, tables_dir),
        lambda: build_f5(root, output_dir, tables_dir),
        lambda: build_f6(root, output_dir),
    ):
        built_outputs, used_sources = builder()
        outputs.extend(built_outputs)
        sources.extend(used_sources)

    contract_path = write_final_comparison_contract(tables_dir)
    outputs.append(contract_path)
    table_paths = sorted(tables_dir.glob("*.csv"))

    unique_sources = sorted({path.resolve() for path in sources if path.exists()})
    manifest = {
        "generated_on": "2026-07-19",
        "purpose": "Advisor meeting report after the 2026-06-28 feedback",
        "checkpoint_policy": "best_val_nll for headline result; best_score/final diagnostic only",
        "figures": [
            relative_path(path, root)
            for path in outputs
            if path.suffix in {".png", ".svg"}
        ],
        "tables": [
            relative_path(path, root)
            for path in table_paths
        ],
        "sources": [
            {"path": relative_path(path, root), "sha256": sha256(path)}
            for path in unique_sources
        ],
        "limitations": [
            "F1 and F2 are single-seed diagnostics, not final model comparisons.",
            "F4 uses variant-level validation-selected candidates; candidate identity changes for Instacart and Taxi.",
            "F5 is the only strict candidate-matched V2/V3b multi-seed comparison in this report.",
            "The final RMTPP/TitanTPP/THP contract is frozen, but the corresponding matched e800 experiment has not been run.",
        ],
    }
    manifest_path = output_dir / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(outputs)} figure/table files and {manifest_path}")


if __name__ == "__main__":
    main()
