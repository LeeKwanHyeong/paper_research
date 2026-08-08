#!/usr/bin/env python3
"""Build manuscript Figures F1-F3 from frozen paper contracts and datasets."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[2] / ".mpl_cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import polars as pl


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
SOURCE_DATA = FIGURES / "source_data"
DATASET_CONTRACT = PAPER / "contracts" / "datasets.json"
T1_AUDIT = PAPER / "data" / "T1_dataset_audit.json"

COLORS = {
    "ink": "#17212B",
    "muted": "#52606D",
    "grid": "#D8DEE4",
    "paper": "#FFFFFF",
    "panel": "#F7F9FB",
    "blue": "#2563EB",
    "blue_light": "#E8F0FE",
    "orange": "#C96B18",
    "orange_light": "#FCEBDD",
    "pink": "#B43B68",
    "pink_light": "#F9E6ED",
    "olive": "#647A2F",
    "olive_light": "#EEF3E4",
}

DATASET_STYLE = {
    "intermittent": ("Intermittent", COLORS["blue"], "-"),
    "yellow_trip_hourly": ("Taxi", COLORS["orange"], "--"),
    "insta_market_basket": ("Instacart", COLORS["pink"], "-."),
}


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.0,
            "axes.edgecolor": COLORS["ink"],
            "axes.labelcolor": COLORS["ink"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["muted"],
            "text.color": COLORS["ink"],
            "figure.facecolor": COLORS["paper"],
            "savefig.facecolor": COLORS["paper"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_figure(fig: Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("svg", "pdf", "png"):
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.04}
        if suffix == "png":
            kwargs["dpi"] = 300
        fig.savefig(FIGURES / f"{stem}.{suffix}", **kwargs)
    plt.close(fig)


def rounded_box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: str = COLORS["panel"],
    edgecolor: str = COLORS["grid"],
    linewidth: float = 1.0,
    radius: float = 0.012,
    zorder: int = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        transform=ax.transAxes,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["muted"],
    linewidth: float = 1.3,
    style: str = "-|>",
    linestyle: str = "-",
    zorder: int = 4,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=10,
            linewidth=linewidth,
            linestyle=linestyle,
            color=color,
            transform=ax.transAxes,
            zorder=zorder,
        )
    )


def panel_heading(ax: Axes, x: float, y: float, label: str, title: str) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10.5, fontweight="bold", va="top")
    ax.text(
        x + 0.032,
        y,
        title,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight="bold",
        va="top",
    )


def build_f1() -> None:
    fig, ax = plt.subplots(figsize=(12.2, 4.25))
    ax.set_axis_off()
    panel_x = [0.015, 0.264, 0.513, 0.762]
    width = 0.222
    y0, height = 0.08, 0.84

    for x in panel_x:
        rounded_box(ax, x, y0, width, height)
    for left, right in zip(panel_x[:-1], panel_x[1:]):
        arrow(ax, (left + width + 0.004, 0.50), (right - 0.004, 0.50))

    # (a) Sparse fixed-grid observations.
    x = panel_x[0]
    panel_heading(ax, x + 0.014, 0.88, "(a)", "Sparse demand")
    ax.text(
        x + 0.014,
        0.80,
        "Fixed-grid series contains\nlong zero-demand intervals.",
        transform=ax.transAxes,
        color=COLORS["muted"],
        va="top",
        linespacing=1.35,
    )
    chart_left, chart_bottom = x + 0.025, 0.30
    chart_w, chart_h = width - 0.05, 0.31
    ax.plot(
        [chart_left, chart_left + chart_w],
        [chart_bottom, chart_bottom],
        transform=ax.transAxes,
        color=COLORS["ink"],
        linewidth=0.9,
    )
    values = [0, 0, 0.42, 0, 0, 0, 0.88, 0, 0.30, 0]
    xs = np.linspace(chart_left + 0.006, chart_left + chart_w - 0.006, len(values))
    for index, (x_i, value) in enumerate(zip(xs, values)):
        if value > 0:
            ax.plot(
                [x_i, x_i],
                [chart_bottom, chart_bottom + chart_h * value],
                transform=ax.transAxes,
                color=COLORS["blue"],
                linewidth=3.0,
                solid_capstyle="round",
            )
            ax.scatter(
                [x_i],
                [chart_bottom + chart_h * value],
                transform=ax.transAxes,
                s=20,
                color=COLORS["blue"],
                zorder=5,
            )
        else:
            ax.scatter(
                [x_i],
                [chart_bottom],
                transform=ax.transAxes,
                s=10,
                facecolor=COLORS["paper"],
                edgecolor=COLORS["grid"],
                linewidth=0.8,
                zorder=5,
            )
        if index in (0, len(values) - 1):
            ax.text(
                x_i,
                chart_bottom - 0.045,
                f"t{index + 1}",
                transform=ax.transAxes,
                fontsize=7.5,
                ha="center",
                color=COLORS["muted"],
            )
    ax.text(
        x + width / 2,
        0.17,
        "Zero-heavy regular observations",
        transform=ax.transAxes,
        fontsize=8.2,
        ha="center",
        color=COLORS["muted"],
    )

    # (b) Positive-demand event sequence.
    x = panel_x[1]
    panel_heading(ax, x + 0.014, 0.88, "(b)", "Marked events")
    ax.text(
        x + 0.014,
        0.80,
        "Retain positive-demand events\nand their inter-event times.",
        transform=ax.transAxes,
        color=COLORS["muted"],
        va="top",
        linespacing=1.35,
    )
    timeline_y = 0.43
    timeline_left, timeline_right = x + 0.035, x + width - 0.035
    ax.plot(
        [timeline_left, timeline_right],
        [timeline_y, timeline_y],
        transform=ax.transAxes,
        color=COLORS["ink"],
        linewidth=1.0,
    )
    event_x = [timeline_left + 0.015, timeline_left + 0.073, timeline_left + 0.142]
    quantities = [5, 18, 3]
    for i, (x_i, qty) in enumerate(zip(event_x, quantities), start=1):
        ax.scatter(
            [x_i],
            [timeline_y],
            transform=ax.transAxes,
            s=58,
            color=COLORS["orange"],
            edgecolor=COLORS["paper"],
            linewidth=0.8,
            zorder=5,
        )
        ax.text(
            x_i,
            timeline_y + 0.060,
            rf"$e_{i}$" + "\n" + rf"$q={qty}$",
            transform=ax.transAxes,
            fontsize=7.6,
            ha="center",
            va="bottom",
            linespacing=1.15,
        )
    for i in range(2):
        arrow(
            ax,
            (event_x[i] + 0.006, timeline_y - 0.075),
            (event_x[i + 1] - 0.006, timeline_y - 0.075),
            color=COLORS["orange"],
            linewidth=1.0,
            style="<->",
        )
        ax.text(
            (event_x[i] + event_x[i + 1]) / 2,
            timeline_y - 0.12,
            rf"$\Delta t_{i + 2}$",
            transform=ax.transAxes,
            fontsize=8.0,
            ha="center",
            color=COLORS["orange"],
        )
    ax.text(
        x + width / 2,
        0.17,
        r"History $\mathcal{H}_i=\{(t_j,q_j)\}_{j\leq i}$",
        transform=ax.transAxes,
        fontsize=8.5,
        ha="center",
        color=COLORS["muted"],
    )

    # (c) Magnitude-factorized continuous quantity.
    x = panel_x[2]
    panel_heading(ax, x + 0.014, 0.88, "(c)", "Quantity factorization")
    ax.text(
        x + 0.014,
        0.80,
        "A categorical mark alone identifies\na range, not the exact quantity.",
        transform=ax.transAxes,
        color=COLORS["muted"],
        va="top",
        linespacing=1.35,
    )
    rounded_box(
        ax,
        x + 0.025,
        0.50,
        width - 0.05,
        0.12,
        facecolor=COLORS["orange_light"],
        edgecolor=COLORS["orange"],
    )
    ax.text(
        x + width / 2,
        0.585,
        "coarse magnitude mark",
        transform=ax.transAxes,
        fontsize=7.2,
        ha="center",
        va="center",
        color=COLORS["muted"],
    )
    ax.text(
        x + width / 2,
        0.535,
        r"$m_i=\min(\lfloor\log_b q_i\rfloor,M)$",
        transform=ax.transAxes,
        fontsize=8.4,
        ha="center",
        va="center",
    )
    rounded_box(
        ax,
        x + 0.025,
        0.34,
        width - 0.05,
        0.12,
        facecolor=COLORS["blue_light"],
        edgecolor=COLORS["blue"],
    )
    ax.text(
        x + width / 2,
        0.425,
        "continuous residual",
        transform=ax.transAxes,
        fontsize=7.2,
        ha="center",
        va="center",
        color=COLORS["muted"],
    )
    ax.text(
        x + width / 2,
        0.375,
        r"$r_i=\log_b q_i-m_i$",
        transform=ax.transAxes,
        fontsize=8.4,
        ha="center",
        va="center",
    )
    ax.text(
        x + width / 2,
        0.265,
        r"$q_i=b^{m_i+r_i}$",
        transform=ax.transAxes,
        fontsize=12.0,
        fontweight="bold",
        ha="center",
        color=COLORS["ink"],
    )
    ax.text(
        x + width / 2,
        0.17,
        r"Example: $q=18$, $b=2$: $m=4$, $r\approx0.17$",
        transform=ax.transAxes,
        fontsize=8.1,
        ha="center",
        color=COLORS["muted"],
    )

    # (d) Joint next-event targets and objective.
    x = panel_x[3]
    panel_heading(ax, x + 0.014, 0.88, "(d)", "Next-event prediction")
    ax.text(
        x + 0.014,
        0.80,
        "Predict when demand returns and\nhow much demand arrives.",
        transform=ax.transAxes,
        color=COLORS["muted"],
        va="top",
        linespacing=1.35,
    )
    target_specs = [
        (0.58, COLORS["orange_light"], COLORS["orange"], r"$p(m_{i+1}\mid\mathcal{H}_i)$", "magnitude mark"),
        (0.43, COLORS["olive_light"], COLORS["olive"], r"$f(\Delta t_{i+1}\mid\mathcal{H}_i)$", "event time"),
        (0.28, COLORS["blue_light"], COLORS["blue"], r"$\hat r_{i+1}\rightarrow\hat q_{i+1}$", "continuous quantity"),
    ]
    for y, face, edge, formula, label in target_specs:
        rounded_box(ax, x + 0.025, y, width - 0.05, 0.105, facecolor=face, edgecolor=edge)
        ax.text(
            x + 0.038,
            y + 0.075,
            label,
            transform=ax.transAxes,
            fontsize=7.1,
            va="center",
            color=COLORS["muted"],
        )
        ax.text(
            x + width / 2,
            y + 0.036,
            formula,
            transform=ax.transAxes,
            fontsize=8.7,
            va="center",
            ha="center",
        )
    ax.text(
        x + width / 2,
        0.17,
        r"$\mathcal{L}=\mathcal{L}_{mark}+\mathcal{L}_{time}+\mathcal{L}_{res}+\lambda_q\mathcal{L}_{qty}$",
        transform=ax.transAxes,
        fontsize=8.3,
        ha="center",
        color=COLORS["muted"],
    )

    save_figure(fig, "F1_problem_formulation")


def build_f2() -> None:
    fig, ax = plt.subplots(figsize=(12.6, 5.35))
    ax.set_axis_off()

    # Main path: token construction -> Titan encoder -> prediction heads.
    rounded_box(ax, 0.025, 0.55, 0.18, 0.31, facecolor=COLORS["panel"])
    ax.text(0.04, 0.82, "Observed event token", transform=ax.transAxes, fontweight="bold", fontsize=9.4)
    token_rows = [
        (0.745, COLORS["orange_light"], COLORS["orange"], r"mark embedding $E(m_i)$"),
        (0.675, COLORS["olive_light"], COLORS["olive"], r"time feature $\log(1+\Delta t_i)$"),
        (0.605, COLORS["blue_light"], COLORS["blue"], r"residual projection $P(r_i)$"),
    ]
    for y, face, edge, text_value in token_rows:
        rounded_box(ax, 0.043, y, 0.144, 0.045, facecolor=face, edgecolor=edge, radius=0.008)
        ax.text(0.115, y + 0.022, text_value, transform=ax.transAxes, fontsize=8.2, ha="center", va="center")
    ax.text(
        0.115,
        0.592,
        "Observed history only;\nthe appended target is masked",
        transform=ax.transAxes,
        fontsize=6.5,
        ha="center",
        va="top",
        color=COLORS["muted"],
        linespacing=1.2,
    )

    arrow(ax, (0.207, 0.705), (0.258, 0.705), color=COLORS["blue"])
    ax.text(0.232, 0.73, "sequence", transform=ax.transAxes, fontsize=7.3, ha="center", color=COLORS["muted"])

    rounded_box(ax, 0.26, 0.48, 0.30, 0.44, facecolor=COLORS["blue_light"], edgecolor=COLORS["blue"], linewidth=1.3)
    ax.text(0.282, 0.875, "Titan causal memory encoder", transform=ax.transAxes, fontweight="bold", fontsize=11.2)
    ax.text(
        0.282,
        0.835,
        "Causal long-range interactions within the observed window",
        transform=ax.transAxes,
        fontsize=7.2,
        color=COLORS["muted"],
    )
    for layer_y in (0.70, 0.57):
        rounded_box(ax, 0.292, layer_y, 0.235, 0.085, facecolor=COLORS["paper"], edgecolor=COLORS["blue"], radius=0.01)
        ax.text(
            0.409,
            layer_y + 0.043,
            "Pre-norm causal memory attention  +  FFN",
            transform=ax.transAxes,
            fontsize=8.6,
            ha="center",
            va="center",
        )
    ax.text(0.409, 0.675, "residual path", transform=ax.transAxes, fontsize=7.2, ha="center", color=COLORS["muted"])
    ax.text(0.409, 0.545, "2 layers in the frozen contract", transform=ax.transAxes, fontsize=7.2, ha="center", color=COLORS["muted"])

    rounded_box(ax, 0.302, 0.39, 0.215, 0.07, facecolor=COLORS["olive_light"], edgecolor=COLORS["olive"])
    ax.text(
        0.409,
        0.425,
        "Static LMM: top-k retrieval + residual",
        transform=ax.transAxes,
        fontsize=7.7,
        ha="center",
        va="center",
    )
    arrow(ax, (0.409, 0.48), (0.409, 0.462), color=COLORS["olive"], linewidth=1.0)

    arrow(ax, (0.562, 0.705), (0.615, 0.705), color=COLORS["blue"])
    ax.text(0.588, 0.73, r"$h_i$", transform=ax.transAxes, fontsize=9.5, ha="center", color=COLORS["blue"])

    rounded_box(ax, 0.62, 0.50, 0.19, 0.40, facecolor=COLORS["panel"])
    ax.text(0.64, 0.855, "Marked TPP heads", transform=ax.transAxes, fontweight="bold", fontsize=10.5)
    head_specs = [
        (0.745, COLORS["orange_light"], COLORS["orange"], "Mark head", r"$p_k=p(m_{i+1}=k\mid h_i)$"),
        (0.635, COLORS["olive_light"], COLORS["olive"], "Time head", r"$\lambda(\tau\mid h_i)$ and $f(\Delta t)$"),
        (0.525, COLORS["blue_light"], COLORS["blue"], "Residual head", r"$\hat r_{i+1,k}$"),
    ]
    for y, face, edge, title, formula in head_specs:
        rounded_box(ax, 0.64, y, 0.15, 0.078, facecolor=face, edgecolor=edge)
        ax.text(0.65, y + 0.050, title, transform=ax.transAxes, fontsize=8.3, fontweight="bold", va="center")
        ax.text(0.65, y + 0.022, formula, transform=ax.transAxes, fontsize=7.6, va="center", color=COLORS["muted"])

    # Quantity reconstruction and losses.
    arrow(ax, (0.812, 0.715), (0.855, 0.715), color=COLORS["orange"], linewidth=1.1)
    arrow(ax, (0.812, 0.565), (0.855, 0.665), color=COLORS["blue"], linewidth=1.1)
    rounded_box(ax, 0.855, 0.58, 0.12, 0.20, facecolor=COLORS["paper"], edgecolor=COLORS["ink"], linewidth=1.1)
    ax.text(0.915, 0.748, "Differentiable", transform=ax.transAxes, fontsize=8.4, fontweight="bold", ha="center")
    ax.text(0.915, 0.716, "quantity decoder", transform=ax.transAxes, fontsize=8.4, fontweight="bold", ha="center")
    ax.text(
        0.915,
        0.655,
        r"$\hat q=\sum_k p_k b^{k+\hat r_k}$",
        transform=ax.transAxes,
        fontsize=9.0,
        ha="center",
    )
    ax.text(
        0.915,
        0.608,
        "preserves continuous quantity",
        transform=ax.transAxes,
        fontsize=7.4,
        ha="center",
        color=COLORS["muted"],
    )

    rounded_box(ax, 0.62, 0.33, 0.355, 0.12, facecolor=COLORS["orange_light"], edgecolor=COLORS["orange"])
    ax.text(0.64, 0.415, "Hybrid training objective", transform=ax.transAxes, fontweight="bold", fontsize=9.2)
    ax.text(
        0.64,
        0.365,
        r"$\mathcal{L}_{CE}+\mathcal{L}_{time}+\mathcal{L}_{Huber(r)}+0.25\,\mathcal{L}_{Huber(q)}$",
        transform=ax.transAxes,
        fontsize=9.0,
    )
    arrow(ax, (0.715, 0.50), (0.715, 0.452), color=COLORS["muted"], linewidth=1.0)
    arrow(ax, (0.915, 0.58), (0.915, 0.452), color=COLORS["muted"], linewidth=1.0)

    # Baseline and Taxi-specific redesign callouts.
    rounded_box(ax, 0.025, 0.10, 0.30, 0.24, facecolor=COLORS["panel"], edgecolor=COLORS["grid"])
    ax.text(0.045, 0.300, "Matched RMTPP baseline", transform=ax.transAxes, fontweight="bold", fontsize=9.5)
    ax.text(
        0.045,
        0.255,
        "RMTPP-matched shares the quantity input, prediction\ntasks, decoder, and hybrid objective; it uses one GRU.",
        transform=ax.transAxes,
        fontsize=7.1,
        color=COLORS["muted"],
        va="top",
        linespacing=1.25,
    )
    rounded_box(ax, 0.063, 0.125, 0.23, 0.055, facecolor=COLORS["paper"], edgecolor=COLORS["ink"])
    ax.text(0.178, 0.152, "tokens  ->  GRU  ->  shared heads", transform=ax.transAxes, fontsize=7.5, ha="center", va="center")

    rounded_box(ax, 0.36, 0.10, 0.615, 0.18, facecolor=COLORS["pink_light"], edgecolor=COLORS["pink"], linewidth=1.1)
    ax.text(0.38, 0.240, "Taxi V3b specialization", transform=ax.transAxes, fontweight="bold", fontsize=9.5, color=COLORS["pink"])
    ax.text(
        0.38,
        0.195,
        r"Mark-conditioned residual experts: $\hat r_k=\hat r_{shared}+\Delta\hat r_k$",
        transform=ax.transAxes,
        fontsize=8.5,
    )
    ax.text(
        0.38,
        0.148,
        "The quantity loss reads mark probabilities through stop-gradient; mark CE remains unchanged.",
        transform=ax.transAxes,
        fontsize=8.1,
        color=COLORS["muted"],
    )
    ax.text(
        0.945,
        0.195,
        r"$\operatorname{sg}(p_k)$",
        transform=ax.transAxes,
        fontsize=9.0,
        ha="right",
        color=COLORS["pink"],
    )

    save_figure(fig, "F2_titantpp_architecture")


def discrete_survival(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("Distribution input is empty after finite-value filtering.")
    unique, counts = np.unique(finite, return_counts=True)
    survival = np.cumsum(counts[::-1], dtype=np.float64)[::-1] / counts.sum()
    return unique.astype(np.float64), survival, counts.astype(np.int64)


def downsample_curve(x: np.ndarray, y: np.ndarray, max_points: int = 900) -> tuple[np.ndarray, np.ndarray]:
    if x.size <= max_points:
        return x, y
    positions = np.unique(np.geomspace(1, x.size, max_points).astype(int) - 1)
    positions = np.unique(np.concatenate(([0], positions, [x.size - 1])))
    return x[positions], y[positions]


def collect_f3_source_data() -> tuple[list[dict], dict]:
    dataset_contract = load_json(DATASET_CONTRACT)
    audit = load_json(T1_AUDIT)
    audit_by_id = {item["dataset_id"]: item for item in audit["datasets"]}
    rows: list[dict] = []
    summaries: dict = {}

    for spec in dataset_contract["datasets"]:
        dataset_id = spec["dataset_id"]
        path = ROOT / spec["with_split_path"]
        expected = spec["expected_hashes"]["with_split"]
        observed = sha256(path)
        if observed != expected:
            raise ValueError(
                f"{dataset_id}: with-split hash mismatch: expected={expected}, observed={observed}"
            )

        frame = pl.read_parquet(path, columns=["oper_part_no", "demand_qty"])
        sequence_lengths = (
            frame.group_by("oper_part_no").len().get_column("len").cast(pl.Float64).to_numpy()
        )
        quantities = frame.get_column("demand_qty").cast(pl.Float64).to_numpy()
        audit_row = audit_by_id[dataset_id]
        if frame.height != audit_row["rows"] or sequence_lengths.size != audit_row["series"]:
            raise ValueError(f"{dataset_id}: F3 counts do not match the frozen T1 audit.")

        summaries[dataset_id] = {
            "sequence": audit_row["sequence_summary"],
            "quantity": audit_row["quantity_summary"],
            "hash": observed,
        }
        for distribution, values in (
            ("sequence_length", sequence_lengths),
            ("quantity", quantities),
        ):
            x, survival, counts = discrete_survival(values)
            total = int(counts.sum())
            for value, share, count in zip(x, survival, counts):
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "paper_name": spec["paper_name"],
                        "distribution": distribution,
                        "value": f"{value:.12g}",
                        "survival_probability": f"{share:.12g}",
                        "value_count": int(count),
                        "population_size": total,
                        "with_split_sha256": observed,
                    }
                )

    SOURCE_DATA.mkdir(parents=True, exist_ok=True)
    output = SOURCE_DATA / "F3_quantity_sequence_distributions.csv"
    fieldnames = list(rows[0].keys())
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows, summaries


def plot_survival_panel(
    ax: Axes,
    rows: Iterable[dict],
    summaries: dict,
    *,
    distribution: str,
    title: str,
    xlabel: str,
    panel_label: str,
) -> None:
    row_list = list(rows)
    for dataset_id, (paper_name, color, linestyle) in DATASET_STYLE.items():
        subset = [
            row
            for row in row_list
            if row["dataset_id"] == dataset_id and row["distribution"] == distribution
        ]
        x = np.asarray([float(row["value"]) for row in subset])
        y = np.asarray([float(row["survival_probability"]) for row in subset])
        x_plot, y_plot = downsample_curve(x, y)
        ax.step(
            x_plot,
            y_plot,
            where="post",
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=paper_name,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(bottom=5e-7, top=1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"Empirical $P(X\geq x)$")
    ax.grid(True, which="major", color=COLORS["grid"], linewidth=0.7, alpha=0.75)
    ax.grid(True, which="minor", color=COLORS["grid"], linewidth=0.35, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(title, loc="left", pad=25, fontweight="bold")
    ax.text(-0.10, 1.115, panel_label, transform=ax.transAxes, fontsize=11, fontweight="bold")

    if distribution == "sequence_length":
        summary_text = "P95 length: " + " | ".join(
            f"{DATASET_STYLE[key][0]} {int(summaries[key]['sequence']['seq_len_p95'])}"
            for key in DATASET_STYLE
        )
    else:
        summary_text = "P95 quantity: " + " | ".join(
            f"{DATASET_STYLE[key][0]} {int(summaries[key]['quantity']['p95']):,}"
            for key in DATASET_STYLE
        )
    ax.text(0.0, 1.035, summary_text, transform=ax.transAxes, fontsize=8.0, color=COLORS["muted"])


def build_f3() -> None:
    rows, summaries = collect_f3_source_data()
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.25))
    plot_survival_panel(
        axes[0],
        rows,
        summaries,
        distribution="sequence_length",
        title="Sequence length distributions",
        xlabel="Events per sequence (log scale)",
        panel_label="(a)",
    )
    plot_survival_panel(
        axes[1],
        rows,
        summaries,
        distribution="quantity",
        title="Event quantity distributions",
        xlabel="Observed quantity (log scale)",
        panel_label="(b)",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=3,
        frameon=False,
        handlelength=3.0,
        columnspacing=1.8,
    )
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.83, wspace=0.28)
    save_figure(fig, "F3_quantity_sequence_distributions")


def main() -> None:
    configure_matplotlib()
    build_f1()
    build_f2()
    build_f3()
    print("Built F1-F3 in paper/figures")


if __name__ == "__main__":
    main()
