from __future__ import annotations

from pathlib import Path

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig_titantpp_figure")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg_cache_titantpp_figure")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)


COLORS = {
    "ink": "#111827",
    "muted": "#526071",
    "line": "#374151",
    "slate": "#64748b",
    "blue": "#2563eb",
    "green": "#16a34a",
    "orange": "#ea580c",
    "red": "#b91c1c",
    "purple": "#7c3aed",
    "amber": "#d97706",
}


def add_panel(ax, x, y, w, h, *, fc, ec, lw=1.8, radius=0.12, zorder=1):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.025,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_arrow(
    ax,
    start,
    end,
    *,
    color="#374151",
    lw=1.9,
    rad=0.0,
    style="-|>",
    mutation_scale=13,
    zorder=5,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        shrinkA=2,
        shrinkB=2,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def add_poly_arrow(ax, points, *, color="#374151", lw=1.7, mutation_scale=12, zorder=5):
    """Draw a routed arrow through fixed waypoints without crossing boxes."""
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, lw=lw, zorder=zorder)
    add_arrow(
        ax,
        points[-2],
        points[-1],
        color=color,
        lw=lw,
        mutation_scale=mutation_scale,
        zorder=zorder,
    )


def label(
    ax,
    x,
    y,
    text,
    *,
    size=10,
    weight="normal",
    color=None,
    ha="left",
    va="center",
    zorder=10,
    rotation=0,
):
    ax.text(
        x,
        y,
        text,
        fontsize=size,
        fontweight=weight,
        color=color or COLORS["ink"],
        ha=ha,
        va=va,
        rotation=rotation,
        zorder=zorder,
    )


def center_label(ax, x, y, w, h, text, *, size=10, weight="normal", color=None):
    label(ax, x + w / 2, y + h / 2, text, size=size, weight=weight, color=color, ha="center", va="center")


def draw_event_sequence(ax):
    x0, y0, w, h = 0.55, 5.42, 8.85, 2.05
    add_panel(ax, x0, y0, w, h, fc="#fffaf0", ec=COLORS["amber"], lw=1.7, radius=0.16)
    label(ax, x0 + 0.22, y0 + h - 0.28, "Observed quantity-bearing event sequence", size=12, weight="bold")
    axis_y = y0 + 0.58
    ax.plot([x0 + 0.55, x0 + w - 0.35], [axis_y, axis_y], color=COLORS["ink"], lw=1.8, zorder=2)
    add_arrow(ax, (x0 + w - 0.55, axis_y), (x0 + w - 0.16, axis_y), color=COLORS["ink"], lw=1.6, mutation_scale=11)
    label(ax, x0 + w - 0.46, axis_y - 0.22, "time", size=8.7)

    events = [
        (1.45, 3, 0.28, "#60a5fa"),
        (2.55, 12, 0.48, "#34d399"),
        (3.75, 1, 0.18, "#93c5fd"),
        (4.98, 68, 0.68, "#f59e0b"),
        (6.55, 7, 0.36, "#a78bfa"),
        (7.95, 145, 0.82, "#ef4444"),
    ]
    xs = []
    for idx, (x, q, bar_h, color) in enumerate(events, start=1):
        xs.append(x)
        ax.plot([x, x], [axis_y, axis_y + bar_h], color=color, lw=5.2, solid_capstyle="round", zorder=3)
        ax.scatter([x], [axis_y + bar_h], s=160, color=color, edgecolors=COLORS["ink"], linewidths=0.8, zorder=4)
        label(ax, x, axis_y + bar_h + 0.18, rf"$q={q}$", size=8.8, ha="center")
        label(ax, x, axis_y - 0.22, rf"$t_{idx}$", size=8.8, ha="center")

    brace_y = y0 + 0.24
    for left, right in zip(xs[:-1], xs[1:]):
        ax.plot([left, right], [brace_y, brace_y], color=COLORS["slate"], lw=0.8)
        ax.plot([left, left], [brace_y - 0.05, brace_y + 0.05], color=COLORS["slate"], lw=0.8)
        ax.plot([right, right], [brace_y - 0.05, brace_y + 0.05], color=COLORS["slate"], lw=0.8)
        label(ax, (left + right) / 2, brace_y - 0.22, r"$\Delta t$", size=8.6, color=COLORS["slate"], ha="center")

    ax.plot([8.22, 8.22], [y0 + 0.28, y0 + h - 0.12], color="#94a3b8", lw=1.1, ls="--")
    label(ax, 8.35, y0 + h - 0.45, "next event\nis hidden", size=8.8, color=COLORS["slate"], va="top")
    label(ax, x0 + 0.22, y0 + 0.48, "observed prefix", size=8.8, color=COLORS["muted"])


def draw_tokenization(ax):
    x0, y0, w, h = 0.55, 2.72, 8.85, 1.95
    add_panel(ax, x0, y0, w, h, fc="#f8fafc", ec=COLORS["slate"], lw=1.6, radius=0.16)
    label(ax, x0 + 0.22, y0 + h - 0.28, "Tokenization for each observed event", size=12, weight="bold")

    boxes = [
        (1.25, y0 + 0.55, 1.65, 0.78, "#dbeafe", COLORS["blue"], "inter-event time", r"$\log(1+\Delta t_i)$"),
        (3.35, y0 + 0.55, 1.65, 0.78, "#dcfce7", COLORS["green"], "magnitude mark", r"$m_i=\lfloor\log_b q_i\rfloor$"),
        (5.45, y0 + 0.55, 1.65, 0.78, "#ffedd5", COLORS["orange"], "residual", r"$r_i=\log_b q_i-m_i$"),
        (7.55, y0 + 0.55, 1.65, 0.78, "#ede9fe", COLORS["purple"], "model token", r"$x_i=[e_{m_i},\,\tau_i,\,W_r r_i]$"),
    ]
    for x, y, bw, bh, fc, ec, title, formula in boxes:
        add_panel(ax, x, y, bw, bh, fc=fc, ec=ec, lw=1.2, radius=0.08)
        label(ax, x + bw / 2, y + bh - 0.2, title, size=8.7, weight="bold", ha="center")
        label(ax, x + bw / 2, y + 0.24, formula, size=8.9, ha="center")

    for left, right in zip(boxes[:-1], boxes[1:]):
        add_arrow(
            ax,
            (left[0] + left[2] + 0.02, left[1] + left[3] / 2),
            (right[0] - 0.04, right[1] + right[3] / 2),
            color=COLORS["slate"],
            lw=1.5,
            mutation_scale=10,
        )
    label(
        ax,
        x0 + 0.36,
        y0 + 0.28,
        "The quantity is represented by a coarse magnitude mark plus a continuous within-mark residual.",
        size=9.5,
        color=COLORS["muted"],
    )


def draw_encoder(ax):
    x0, y0, w, h = 9.92, 3.98, 3.35, 3.05
    add_panel(ax, x0, y0, w, h, fc="#eef2ff", ec="#4f46e5", lw=1.8, radius=0.18)
    label(ax, x0 + 0.24, y0 + h - 0.33, "TitanTPP history encoder", size=12, weight="bold")
    label(ax, x0 + 0.24, y0 + h - 0.68, "causal memory-attention over event tokens", size=9.3, color=COLORS["muted"])

    block_x, block_y, block_w, block_h = x0 + 0.46, y0 + 0.46, w - 0.92, 1.55
    add_panel(ax, block_x, block_y, block_w, block_h, fc="#ffffff", ec="#94a3b8", lw=0.9, radius=0.08)
    label(ax, block_x + block_w / 2, block_y + block_h - 0.24, r"Titan encoder block $\times L$", size=9.5, weight="bold", ha="center")

    sublayers = [
        (block_x + 0.28, block_y + 0.72, block_w - 0.56, 0.34, "memory attention"),
        (block_x + 0.28, block_y + 0.18, block_w - 0.56, 0.42, "FFN + residual\nupdate"),
    ]
    for sx, sy, sw, sh, txt in sublayers:
        add_panel(ax, sx, sy, sw, sh, fc="#f8fafc", ec="#cbd5e1", lw=0.8, radius=0.05)
        center_label(ax, sx, sy, sw, sh, txt, size=8.1)
    add_arrow(
        ax,
        (block_x + block_w / 2, sublayers[0][1] - 0.02),
        (block_x + block_w / 2, sublayers[1][1] + sublayers[1][3] + 0.02),
        color=COLORS["slate"],
        lw=1.1,
        mutation_scale=8,
    )
    add_arrow(
        ax,
        (block_x + block_w / 2, block_y + 0.04),
        (block_x + block_w / 2, block_y - 0.34),
        color=COLORS["slate"],
        lw=1.2,
        mutation_scale=9,
    )
    label(ax, x0 + 0.55, y0 + 0.13, r"history state $h_i$", size=9.5)

    add_arrow(ax, (9.2, 3.5), (9.92, 5.45), color=COLORS["line"], lw=2.0, rad=-0.07, mutation_scale=14)
    label(ax, 9.0, 4.05, "prefix tokens", size=8.6, color=COLORS["muted"], ha="right")


def draw_heads(ax):
    heads = {
        "time": (14.05, 6.02, 1.35, 0.76, "#fee2e2", COLORS["red"], "time", r"$\hat{\Delta t}_{i+1}$"),
        "mark": (14.05, 4.95, 1.35, 0.76, "#dcfce7", COLORS["green"], "mark", r"$p(m_{i+1}\mid h_i)$"),
        "residual": (14.05, 3.85, 1.35, 0.76, "#ffedd5", COLORS["orange"], "residual", r"$\hat r_{i+1,m}$"),
    }
    for x, y, w, h, fc, ec, title, formula in heads.values():
        add_panel(ax, x, y, w, h, fc=fc, ec=ec, lw=1.5, radius=0.08)
        label(ax, x + 0.13, y + h - 0.2, title, size=10.8, weight="bold")
        label(ax, x + w / 2, y + 0.25, formula, size=9.6, ha="center")

    add_arrow(ax, (13.27, 5.72), (14.05, 6.39), color=COLORS["line"], lw=1.7, mutation_scale=12)
    add_arrow(ax, (13.27, 5.44), (14.05, 5.33), color=COLORS["line"], lw=1.7, mutation_scale=12)
    add_arrow(ax, (13.27, 5.13), (14.05, 4.23), color=COLORS["line"], lw=1.7, mutation_scale=12)

    add_panel(ax, 12.05, 2.55, 3.35, 1.02, fc="#fefce8", ec="#ca8a04", lw=1.6, radius=0.1)
    label(ax, 12.32, 3.23, "Quantity reconstruction", size=11.2, weight="bold")
    label(ax, 13.73, 2.82, r"$\hat q_{i+1}=\sum_m p_m\,b^{m+\hat r_m}$", size=10.4, ha="center")

    add_panel(ax, 14.05, 7.05, 1.35, 0.46, fc="#f8fafc", ec="#94a3b8", lw=1.0, radius=0.05)
    center_label(ax, 14.05, 7.05, 1.35, 0.46, "next-event time", size=8.4, color=COLORS["muted"])
    add_arrow(ax, (14.72, 6.78), (14.72, 7.05), color=COLORS["red"], lw=1.3, mutation_scale=10)

    # Route mark and residual paths around the right side so arrows do not
    # cross through the head boxes.
    add_poly_arrow(
        ax,
        [(15.42, 5.33), (15.72, 5.33), (15.72, 3.78), (15.18, 3.78), (15.08, 3.60)],
        color=COLORS["green"],
        lw=1.65,
        mutation_scale=11,
    )
    add_poly_arrow(
        ax,
        [(15.42, 4.23), (15.58, 4.23), (15.58, 3.70), (14.56, 3.70), (14.46, 3.60)],
        color=COLORS["orange"],
        lw=1.65,
        mutation_scale=11,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": False,
        }
    )
    fig, ax = plt.subplots(figsize=(16, 8.6), dpi=220)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.0)
    ax.axis("off")

    label(ax, 0.55, 8.63, "TitanTPP schematic for quantity-bearing event prediction", size=16, weight="bold")
    label(
        ax,
        0.55,
        8.31,
        "Observed demand events are converted into tokens; the history encoder predicts the next event time and quantity.",
        size=10.2,
        color=COLORS["muted"],
    )

    draw_event_sequence(ax)
    draw_tokenization(ax)
    draw_encoder(ax)
    draw_heads(ax)

    label(
        ax,
        0.55,
        0.42,
        "Figure 1. Example-driven schematic of TitanTPP. The demand quantity is split into a magnitude mark and a residual before sequence encoding.",
        size=9.2,
        color=COLORS["ink"],
        va="bottom",
    )

    png = OUT_DIR / "F1_titantpp_event_sequence_architecture.png"
    fig.savefig(png, bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(png)


if __name__ == "__main__":
    main()
