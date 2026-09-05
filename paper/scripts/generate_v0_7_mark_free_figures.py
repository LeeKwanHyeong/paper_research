#!/usr/bin/env python3
"""Generate the v0.7 mark-free TitanTPP architecture figure.

Figure contract
---------------
Question
    How does the mark-free model map an observed time/count history to its two
    next-event predictions?
Takeaway
    Every event supplies only continuous log-time and log-count features.  The
    Titan-inspired encoder produces one shared history state for the clamped
    RMTPP time-score and direct log-count regression heads.
Surface
    Standalone paper figure: vector PDF/SVG plus a 300 dpi PNG.
Palette
    Blue and gold roots with neutral ink; labels and layout carry distinctions
    without relying on colour alone.
QA
    All labels must remain readable at two-column width, arrows must not cross
    text, and the three exports must depict the same fixed canvas.

The encoder details mirror ``CountAwareTitanTPP`` with static hard memory:
two causal memory-attention layers, 16 learned persistent tokens per layer,
and a 64-entry learned prototype bank with top-4 mean retrieval.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path
import xml.etree.ElementTree as ET


os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_v0_7_mark_free")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache_v0_7_mark_free")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "figures"
STEM = "F1_v0_7_mark_free_architecture"

COLORS = {
    "paper": "#FFFFFF",
    "panel": "#F7F9FC",
    "ink": "#172033",
    "muted": "#596579",
    "line": "#687386",
    "grid": "#CBD3DF",
    "blue": "#2563EB",
    "blue_dark": "#1746A2",
    "blue_light": "#EAF2FF",
    "blue_open": "#F5F8FF",
    "gold": "#B96B00",
    "gold_dark": "#7A4600",
    "gold_light": "#FFF1D6",
    "gold_open": "#FFFAF0",
}


def setup_style() -> None:
    """Use embedded TrueType text and deterministic publication styling."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "svg.hashsalt": "titantpp-v0.7-mark-free-architecture",
            "axes.unicode_minus": False,
            "savefig.facecolor": COLORS["paper"],
        }
    )


def box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str = "panel",
    edge: str = "grid",
    linewidth: float = 1.25,
    radius: float = 0.10,
    zorder: int = 1,
):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        facecolor=COLORS[face],
        edgecolor=COLORS[edge],
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def label(
    ax,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 11.0,
    weight: str = "normal",
    color: str = "ink",
    horizontal: str = "left",
    vertical: str = "center",
    linespacing: float = 1.15,
    zorder: int = 10,
):
    return ax.text(
        x,
        y,
        value,
        fontsize=size,
        fontweight=weight,
        color=COLORS[color],
        ha=horizontal,
        va=vertical,
        linespacing=linespacing,
        zorder=zorder,
    )


def centered(ax, x, y, width, height, value, **kwargs):
    return label(
        ax,
        x + width / 2,
        y + height / 2,
        value,
        horizontal="center",
        vertical="center",
        **kwargs,
    )


def arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = "line",
    linewidth: float = 1.75,
    mutation_scale: float = 13.0,
    connection: str = "arc3,rad=0",
    zorder: int = 6,
):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=COLORS[color],
        connectionstyle=connection,
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def routed_arrow(
    ax,
    points: list[tuple[float, float]],
    *,
    color: str = "line",
    linewidth: float = 1.0,
    mutation_scale: float = 9.0,
):
    """Draw a thin orthogonal route outside content panels."""
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.LINETO] * (len(points) - 1))
    patch = FancyArrowPatch(
        path=path,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=COLORS[color],
        capstyle="round",
        joinstyle="round",
        zorder=5,
    )
    ax.add_patch(patch)
    return patch


def draw_history(ax) -> None:
    x, y, width, height = 0.35, 1.30, 2.75, 4.55
    box(ax, x, y, width, height, face="panel", edge="grid", linewidth=1.25, radius=0.13)
    label(ax, x + 0.20, y + height - 0.36, "Observed event history", size=13.2, weight="bold")
    label(
        ax,
        x + 0.20,
        y + height - 0.72,
        r"$\mathcal{H}_i=\{(\Delta t_j,q_j)\}_{j\leq i}$",
        size=11.2,
        color="muted",
    )

    axis_y = y + 1.50
    left, right = x + 0.30, x + width - 0.25
    ax.plot([left, right], [axis_y, axis_y], color=COLORS["ink"], linewidth=1.45, zorder=2)
    arrow(ax, (right - 0.14, axis_y), (right + 0.08, axis_y), color="ink", linewidth=1.25, mutation_scale=10)

    events = (
        (left + 0.20, 0.54, r"$q_{i-3}$"),
        (left + 0.73, 0.95, r"$q_{i-2}$"),
        (left + 1.27, 0.38, r"$q_{i-1}$"),
        (left + 1.84, 1.18, r"$q_i$"),
    )
    for index, (event_x, bar_height, quantity) in enumerate(events):
        tone = "blue" if index < len(events) - 1 else "gold"
        ax.plot(
            [event_x, event_x],
            [axis_y, axis_y + bar_height],
            color=COLORS[tone],
            linewidth=4.2,
            solid_capstyle="round",
            zorder=3,
        )
        ax.scatter(
            [event_x],
            [axis_y + bar_height],
            s=66,
            facecolor=COLORS["paper"],
            edgecolor=COLORS[tone],
            linewidth=1.5,
            zorder=4,
        )
        label(
            ax,
            event_x,
            axis_y + bar_height + 0.24,
            quantity,
            size=9.4,
            color=tone,
            horizontal="center",
        )

    label(ax, events[0][0], axis_y - 0.28, r"$t_{i-3}$", size=8.8, color="muted", horizontal="center")
    label(ax, events[-1][0], axis_y - 0.28, r"$t_i$", size=8.8, color="muted", horizontal="center")
    label(ax, (events[1][0] + events[2][0]) / 2, axis_y - 0.28, r"$\cdots$", size=12, color="muted", horizontal="center")

    hidden_x = right - 0.03
    ax.plot(
        [hidden_x, hidden_x],
        [axis_y - 0.42, axis_y + 1.55],
        color=COLORS["line"],
        linewidth=1.0,
        linestyle=(0, (3, 3)),
        zorder=2,
    )
    label(ax, hidden_x - 0.05, axis_y + 1.88, "target $i+1$", size=8.5, color="muted", horizontal="right")
    label(ax, hidden_x - 0.05, axis_y + 1.62, "not observed", size=8.5, color="muted", horizontal="right")

    box(ax, x + 0.20, y + 0.30, width - 0.40, 0.72, face="blue_open", edge="blue", linewidth=1.0, radius=0.07)
    centered(
        ax,
        x + 0.20,
        y + 0.30,
        width - 0.40,
        0.72,
        "Only observed time gaps\nand counts enter the\nhistory representation",
        size=8.7,
        color="blue_dark",
    )


def draw_token(ax) -> None:
    x, y, width, height = 3.55, 2.05, 2.40, 3.15
    box(ax, x, y, width, height, face="panel", edge="grid", linewidth=1.25, radius=0.13)
    label(ax, x + width / 2, y + height - 0.36, "Continuous event token", size=10.8, weight="bold", horizontal="center")

    box(ax, x + 0.20, y + 1.75, width - 0.40, 0.58, face="blue_light", edge="blue", linewidth=1.0, radius=0.07)
    centered(
        ax,
        x + 0.20,
        y + 1.75,
        width - 0.40,
        0.58,
        r"time   $\tau_j=\log(1+\Delta t_j)$",
        size=9.3,
        color="blue_dark",
        weight="bold",
    )
    box(ax, x + 0.20, y + 1.02, width - 0.40, 0.58, face="gold_light", edge="gold", linewidth=1.0, radius=0.07)
    centered(
        ax,
        x + 0.20,
        y + 1.02,
        width - 0.40,
        0.58,
        r"count   $c_j=\log(1+q_j)$",
        size=9.3,
        color="gold_dark",
        weight="bold",
    )
    label(
        ax,
        x + width / 2,
        y + 0.48,
        r"$x_j=[\tau_j,\,c_j]\in\mathbb{R}^{2}$",
        size=12.1,
        weight="bold",
        horizontal="center",
    )


def draw_encoder(ax) -> None:
    x, y, width, height = 6.40, 0.92, 4.35, 5.38
    box(ax, x, y, width, height, face="blue_open", edge="blue", linewidth=1.55, radius=0.15)
    label(ax, x + 0.22, y + height - 0.36, "Titan-inspired history encoder", size=14.0, weight="bold", color="blue_dark")
    label(
        ax,
        x + 0.22,
        y + height - 0.72,
        "Causal encoding within the observed window",
        size=9.6,
        color="muted",
    )

    block_x, block_width = x + 1.05, width - 1.30
    stages = (
        (y + 3.62, 0.64, "Linear input projection\n+ learned position"),
        (y + 2.45, 0.82, "2 × causal memory-attention\n+ feed-forward update"),
        (y + 1.22, 0.76, "Static top-4 prototype retrieval\n+ additive fusion"),
    )
    for index, (stage_y, stage_h, value) in enumerate(stages):
        face = "paper" if index != 2 else "gold_open"
        edge = "blue" if index != 2 else "gold"
        box(ax, block_x, stage_y, block_width, stage_h, face=face, edge=edge, linewidth=1.05, radius=0.07)
        centered(
            ax,
            block_x,
            stage_y,
            block_width,
            stage_h,
            value,
            size=9.4,
            weight="bold" if index == 1 else "normal",
            color="ink" if index != 2 else "gold_dark",
        )

    arrow(
        ax,
        (block_x + block_width / 2, stages[0][0] - 0.05),
        (block_x + block_width / 2, stages[1][0] + stages[1][1] + 0.05),
        color="blue",
        linewidth=1.35,
        mutation_scale=10,
    )
    arrow(
        ax,
        (block_x + block_width / 2, stages[1][0] - 0.05),
        (block_x + block_width / 2, stages[2][0] + stages[2][1] + 0.05),
        color="blue",
        linewidth=1.35,
        mutation_scale=10,
    )

    token_y = stages[1][0] + stages[1][1] / 2
    for row in range(2):
        for col in range(3):
            circle = Circle(
                (x + 0.33 + col * 0.17, token_y + (row - 0.5) * 0.19),
                0.055,
                facecolor=COLORS["paper"],
                edgecolor=COLORS["blue"],
                linewidth=1.0,
                zorder=4,
            )
            ax.add_patch(circle)
    label(
        ax,
        x + 0.18,
        token_y + 0.68,
        "learned persistent\ntokens (16 / layer)",
        size=8.4,
        color="blue_dark",
        vertical="top",
    )
    arrow(
        ax,
        (x + 0.83, token_y),
        (block_x - 0.04, token_y),
        color="blue",
        linewidth=1.1,
        mutation_scale=9,
    )

    bank_y = stages[2][0] + stages[2][1] / 2
    for row in range(3):
        for col in range(3):
            circle = Circle(
                (x + 0.33 + col * 0.17, bank_y + (row - 1) * 0.16),
                0.050,
                facecolor=COLORS["gold_light"],
                edgecolor=COLORS["gold"],
                linewidth=0.9,
                zorder=4,
            )
            ax.add_patch(circle)
    label(
        ax,
        x + 0.18,
        bank_y + 0.73,
        "learned prototype\nbank (64 entries)",
        size=8.4,
        color="gold_dark",
        vertical="top",
    )
    arrow(
        ax,
        (x + 0.83, bank_y),
        (block_x - 0.04, bank_y),
        color="gold",
        linewidth=1.1,
        mutation_scale=9,
    )

    label(
        ax,
        x + width / 2,
        y + 0.49,
        "shared fused history representation",
        size=9.4,
        color="muted",
        horizontal="center",
    )


def draw_state_and_heads(ax) -> None:
    state_x, state_y, state_w, state_h = 11.08, 3.22, 1.18, 1.18
    box(ax, state_x, state_y, state_w, state_h, face="paper", edge="blue", linewidth=1.45, radius=0.16)
    centered(ax, state_x, state_y + 0.16, state_w, 0.56, r"$h_i$", size=18, weight="bold", color="blue_dark")
    label(
        ax,
        state_x + state_w / 2,
        state_y + 0.20,
        "shared state",
        size=8.8,
        color="muted",
        horizontal="center",
    )

    time = (12.45, 4.25, 2.80, 1.56)
    count = (12.45, 1.77, 2.80, 1.95)
    box(ax, *time, face="blue_light", edge="blue", linewidth=1.35, radius=0.11)
    label(ax, time[0] + time[2] / 2, time[1] + time[3] - 0.32, "Clamped RMTPP time head", size=10.6, weight="bold", color="blue_dark", horizontal="center")
    label(
        ax,
        time[0] + time[2] / 2,
        time[1] + 0.77,
        r"$s_{time}(\Delta t_{i+1}\mid h_i)$",
        size=12.0,
        horizontal="center",
    )
    label(
        ax,
        time[0] + time[2] / 2,
        time[1] + 0.33,
        r"$\mathcal{L}_{time}=-s_{time}(\Delta t_{i+1}\mid h_i)$",
        size=8.9,
        color="muted",
        horizontal="center",
    )

    box(ax, *count, face="gold_light", edge="gold", linewidth=1.35, radius=0.11)
    label(ax, count[0] + count[2] / 2, count[1] + count[3] - 0.32, "Direct log-count head", size=10.8, weight="bold", color="gold_dark", horizontal="center")
    label(
        ax,
        count[0] + count[2] / 2,
        count[1] + 1.18,
        r"$\hat z_{i+1}=\mathrm{softplus}(w_q^\top h_i+b_q)$",
        size=9.5,
        horizontal="center",
    )
    label(
        ax,
        count[0] + count[2] / 2,
        count[1] + 0.75,
        r"$\hat q_{i+1}=\exp(\hat z_{i+1})-1$",
        size=9.9,
        horizontal="center",
    )
    label(
        ax,
        count[0] + count[2] / 2,
        count[1] + 0.31,
        r"$\mathcal{L}_{qty}=(\hat z_{i+1}-\log(1+q_{i+1}))^2$",
        size=8.6,
        color="muted",
        horizontal="center",
    )

    arrow(ax, (state_x + state_w, state_y + state_h * 0.72), (time[0], time[1] + time[3] / 2), color="blue", linewidth=1.55)
    arrow(ax, (state_x + state_w, state_y + state_h * 0.28), (count[0], count[1] + count[3] / 2), color="gold", linewidth=1.55)

    objective = (10.98, 0.42, 4.26, 0.78)
    box(ax, *objective, face="panel", edge="line", linewidth=1.15, radius=0.09)
    centered(
        ax,
        *objective,
        r"Joint objective   $\mathcal{L}_{joint}=\mathcal{L}_{time}+\lambda_q\mathcal{L}_{qty}$",
        size=10.7,
        weight="bold",
    )
    routed_arrow(
        ax,
        [
            (time[0] + time[2] + 0.02, time[1] + time[3] / 2),
            (15.43, time[1] + time[3] / 2),
            (15.43, objective[1] + objective[3] + 0.20),
            (objective[0] + objective[2] - 0.12, objective[1] + objective[3] + 0.02),
        ],
        color="blue",
    )
    arrow(
        ax,
        (count[0] + count[2] * 0.57, count[1] - 0.04),
        (objective[0] + objective[2] * 0.72, objective[1] + objective[3] + 0.03),
        color="gold",
        linewidth=1.0,
        mutation_scale=9,
    )


def build_figure():
    setup_style()
    fig, ax = plt.subplots(figsize=(15.6, 7.3), constrained_layout=False)
    fig.patch.set_facecolor(COLORS["paper"])
    ax.set_facecolor(COLORS["paper"])
    ax.set_xlim(0, 15.6)
    ax.set_ylim(0, 7.3)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    label(ax, 0.36, 6.91, "TitanTPP: mark-free count-aware next-event prediction", size=17.0, weight="bold")
    label(
        ax,
        0.37,
        6.55,
        "Continuous time/count tokens are encoded once; the resulting history state drives both prediction tasks.",
        size=10.4,
        color="muted",
    )
    ax.plot([0.36, 15.24], [6.34, 6.34], color=COLORS["grid"], linewidth=0.9)

    draw_history(ax)
    draw_token(ax)
    draw_encoder(ax)
    draw_state_and_heads(ax)

    arrow(ax, (3.10, 3.58), (3.55, 3.58), color="line", linewidth=1.75)
    arrow(ax, (5.95, 3.58), (6.40, 3.58), color="line", linewidth=1.75)
    arrow(ax, (10.75, 3.81), (11.08, 3.81), color="blue", linewidth=1.75)

    return fig


def export(fig) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [OUT_DIR / f"{STEM}.{suffix}" for suffix in ("pdf", "png", "svg")]
    generator = "paper/scripts/generate_v0_7_mark_free_figures.py"
    fixed_time = dt.datetime(2026, 9, 5, tzinfo=dt.timezone.utc)
    fig.savefig(
        outputs[0],
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={"Creator": generator, "CreationDate": fixed_time, "ModDate": fixed_time},
    )
    fig.savefig(
        outputs[1],
        format="png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={"Software": generator},
    )
    fig.savefig(
        outputs[2],
        format="svg",
        bbox_inches="tight",
        pad_inches=0.08,
        metadata={"Creator": generator, "Date": "2026-09-05"},
    )
    svg_text = outputs[2].read_text(encoding="utf-8")
    outputs[2].write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
    )
    return outputs


def validate_exports(outputs: list[Path]) -> None:
    expected = {".pdf", ".png", ".svg"}
    if {path.suffix for path in outputs} != expected:
        raise RuntimeError("The figure export set is incomplete")
    minimum_bytes = {".pdf": 20_000, ".png": 250_000, ".svg": 30_000}
    for path in outputs:
        if not path.is_file() or path.stat().st_size < minimum_bytes[path.suffix]:
            raise RuntimeError(f"Invalid or unexpectedly small export: {path}")
    ET.parse(next(path for path in outputs if path.suffix == ".svg"))


def main() -> None:
    figure = build_figure()
    try:
        outputs = export(figure)
    finally:
        plt.close(figure)
    validate_exports(outputs)
    for output in outputs:
        print(f"wrote {output.relative_to(ROOT)} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
