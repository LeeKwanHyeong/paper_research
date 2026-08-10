from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 2400, 1350


def font(size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if italic:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Italic.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/Library/Fonts/Arial.ttf",
        ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


TITLE = font(42, bold=True)
SUBTITLE = font(25)
LABEL = font(28, bold=True)
BODY = font(23)
SMALL = font(20)
TINY = font(18)
ITALIC = font(21, italic=True)


def text_center(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str = "#111827",
    spacing: int = 7,
) -> None:
    x0, y0, x1, y1 = xy
    lines = text.split("\n")
    metrics = [draw.textbbox((0, 0), line, font=fnt) for line in lines]
    widths = [box[2] - box[0] for box in metrics]
    heights = [box[3] - box[1] for box in metrics]
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y0 + ((y1 - y0) - total_h) // 2
    for line, width, height in zip(lines, widths, heights):
        draw.text((x0 + ((x1 - x0) - width) // 2, y), line, font=fnt, fill=fill)
        y += height + spacing


def rounded(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    *,
    fill: str,
    outline: str,
    radius: int = 24,
    width: int = 3,
) -> None:
    draw.rounded_rectangle(xy, radius=radius, fill=fill, outline=outline, width=width)


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    color: str = "#374151",
    width: int = 4,
    head: int = 16,
) -> None:
    draw.line([start, end], fill=color, width=width)
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    p1 = (x1 - ux * head + px * head * 0.55, y1 - uy * head + py * head * 0.55)
    p2 = (x1 - ux * head - px * head * 0.55, y1 - uy * head - py * head * 0.55)
    draw.polygon([end, p1, p2], fill=color)


def dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    fill: str,
    width: int = 2,
    dash: int = 12,
    gap: int = 8,
) -> None:
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    pos = 0.0
    while pos < length:
        seg = min(dash, length - pos)
        sx, sy = x0 + ux * pos, y0 + uy * pos
        ex, ey = x0 + ux * (pos + seg), y0 + uy * (pos + seg)
        draw.line([(sx, sy), (ex, ey)], fill=fill, width=width)
        pos += dash + gap


def draw_event_sequence(draw: ImageDraw.ImageDraw) -> None:
    panel = (80, 200, 1435, 595)
    rounded(draw, panel, fill="#fffaf0", outline="#d97706", radius=30, width=3)
    draw.text((110, 228), "Observed quantity-bearing event sequence", font=LABEL, fill="#111827")
    draw.text((110, 268), "Positive-demand events arrive irregularly; each event carries a quantity.", font=BODY, fill="#4b5563")

    axis_y = 455
    draw.line([(160, axis_y), (1325, axis_y)], fill="#111827", width=3)
    arrow(draw, (1325, axis_y), (1385, axis_y), color="#111827", width=3, head=14)
    draw.text((1340, axis_y + 20), "time", font=SMALL, fill="#111827")

    events = [
        (225, 3, 2, "#60a5fa"),
        (390, 12, 4, "#34d399"),
        (570, 1, 1, "#93c5fd"),
        (760, 68, 7, "#f59e0b"),
        (1005, 7, 3, "#a78bfa"),
        (1220, 145, 9, "#ef4444"),
    ]
    for idx, (x, qty, height_unit, color) in enumerate(events, start=1):
        bar_h = 16 * height_unit
        draw.line([(x, axis_y), (x, axis_y - bar_h)], fill=color, width=11)
        draw.ellipse((x - 17, axis_y - bar_h - 17, x + 17, axis_y - bar_h + 17), fill=color, outline="#111827", width=2)
        draw.line([(x, axis_y - bar_h - 17), (x, axis_y - 6)], fill=color, width=5)
        draw.text((x - 18, axis_y + 18), f"t{idx}", font=SMALL, fill="#111827")
        draw.text((x - 30, axis_y - bar_h - 58), f"q={qty}", font=SMALL, fill="#111827")
        if idx > 1:
            prev_x = events[idx - 2][0]
            y = axis_y + 76
            draw.line([(prev_x, y), (x, y)], fill="#64748b", width=2)
            draw.line([(prev_x, y - 7), (prev_x, y + 7)], fill="#64748b", width=2)
            draw.line([(x, y - 7), (x, y + 7)], fill="#64748b", width=2)
            text_center(draw, (prev_x, y + 8, x, y + 36), "Delta t", TINY, fill="#64748b")

    draw.text((110, 535), "Observed prefix", font=SMALL, fill="#374151")
    dashed_line(draw, (1260, 225), (1260, 555), fill="#94a3b8", width=2)
    draw.text((1278, 228), "next event\nis hidden", font=SMALL, fill="#64748b")


def draw_tokenization(draw: ImageDraw.ImageDraw) -> None:
    panel = (80, 680, 1435, 1040)
    rounded(draw, panel, fill="#f8fafc", outline="#64748b", radius=30, width=3)
    draw.text((110, 708), "Tokenization of each observed event", font=LABEL, fill="#111827")

    cols = [
        (185, "inter-event time", "log(1 + Delta t_i)", "#dbeafe", "#2563eb"),
        (505, "magnitude mark", "m_i = floor(log_b q_i)", "#dcfce7", "#16a34a"),
        (825, "within-scale residual", "r_i = log_b q_i - m_i", "#ffedd5", "#ea580c"),
        (1145, "model token", "x_i = concat(time, mark, residual)", "#ede9fe", "#7c3aed"),
    ]
    for x, title, body, fill, outline in cols:
        rounded(draw, (x, 785, x + 250, 940), fill=fill, outline=outline, radius=18, width=2)
        text_center(draw, (x + 10, 802, x + 240, 842), title, SMALL, fill="#111827")
        text_center(draw, (x + 12, 860, x + 238, 925), body, TINY, fill="#1f2937")

    arrow(draw, (435, 862), (505, 862), color="#64748b", width=3, head=12)
    arrow(draw, (755, 862), (825, 862), color="#64748b", width=3, head=12)
    arrow(draw, (1075, 862), (1145, 862), color="#64748b", width=3, head=12)
    draw.text((135, 974), "The quantity is not treated as a plain category: mark and residual reconstruct the original scale.", font=BODY, fill="#4b5563")


def draw_encoder_and_heads(draw: ImageDraw.ImageDraw) -> None:
    enc = (1485, 255, 2005, 800)
    rounded(draw, enc, fill="#eef2ff", outline="#4f46e5", radius=38, width=3)
    draw.text((1540, 285), "TitanTPP history encoder", font=LABEL, fill="#111827")
    draw.text((1540, 326), "causal memory-attention over event tokens", font=BODY, fill="#4b5563")

    layer_y = [395, 500, 605]
    for i, y in enumerate(layer_y, start=1):
        rounded(draw, (1560, y, 1930, y + 70), fill="#ffffff", outline="#94a3b8", radius=16, width=2)
        label = "memory attention" if i == 1 else "feed-forward + residual"
        text_center(draw, (1595, y + 8, 1970, y + 62), label, BODY, fill="#111827")
        if i < len(layer_y):
            arrow(draw, (1745, y + 70), (1745, layer_y[i] - 3), color="#475569", width=3, head=11)

    draw.text((1588, 718), "history state h_i", font=BODY, fill="#111827")
    draw.text((1588, 748), "RMTPP-Q and THP-Q keep the same quantity targets\nbut replace this encoder family.", font=SMALL, fill="#475569")

    arrow(draw, (1365, 862), (1485, 535), color="#374151", width=5, head=18)
    draw.text((1320, 750), "prefix tokens", font=SMALL, fill="#475569")

    heads = [
        ((2110, 230, 2325, 360), "time", "next Delta t", "#fee2e2", "#b91c1c"),
        ((2110, 450, 2325, 580), "mark", "P(next mark)", "#dcfce7", "#16a34a"),
        ((2110, 670, 2325, 800), "residual", "next residual", "#ffedd5", "#ea580c"),
    ]
    for xy, title, body, fill, outline in heads:
        rounded(draw, xy, fill=fill, outline=outline, radius=20, width=3)
        draw.text((xy[0] + 22, xy[1] + 20), title, font=LABEL, fill="#111827")
        text_center(draw, (xy[0] + 10, xy[1] + 64, xy[2] - 10, xy[3] - 15), body, BODY, fill="#1f2937")

    arrow(draw, (2005, 445), (2110, 295), color="#475569", width=4, head=16)
    arrow(draw, (2005, 535), (2110, 515), color="#475569", width=4, head=16)
    arrow(draw, (2005, 625), (2110, 735), color="#475569", width=4, head=16)

    recon = (1810, 950, 2325, 1145)
    rounded(draw, recon, fill="#fefce8", outline="#ca8a04", radius=26, width=3)
    draw.text((1870, 980), "Quantity reconstruction", font=LABEL, fill="#111827")
    text_center(
        draw,
        (1870, 1030, 2295, 1118),
        "combine mark probability and residual\nq_hat = E[ base^(mark + residual) ]",
        BODY,
        fill="#1f2937",
    )
    arrow(draw, (2218, 580), (2025, 950), color="#16a34a", width=4, head=16)
    arrow(draw, (2218, 800), (2075, 950), color="#ea580c", width=4, head=16)
    arrow(draw, (2218, 360), (2218, 950), color="#b91c1c", width=3, head=13)


def main() -> None:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    draw.text((80, 58), "TitanTPP schematic for quantity-bearing event prediction", font=TITLE, fill="#111827")
    draw.text(
        (80, 114),
        "The figure starts from observed demand events, converts each event into a token, and predicts the next time and quantity.",
        font=SUBTITLE,
        fill="#4b5563",
    )

    draw_event_sequence(draw)
    draw_tokenization(draw)
    draw_encoder_and_heads(draw)

    draw.text(
        (80, 1242),
        "Figure 1. Example-driven schematic of TitanTPP. Quantities are split into a coarse magnitude mark and a continuous residual before sequence encoding.",
        font=ITALIC,
        fill="#111827",
    )

    png = OUT_DIR / "F1_titantpp_event_sequence_architecture.png"
    img.save(png)
    print(png)


if __name__ == "__main__":
    main()
