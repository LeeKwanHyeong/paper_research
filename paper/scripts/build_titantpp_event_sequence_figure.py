from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

W, H = 2200, 1080


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
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
LABEL = font(27, bold=True)
BODY = font(22)
SMALL = font(21)


def multiline_center(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fnt: ImageFont.ImageFont,
    fill: str = "#1f2937",
    spacing: int = 8,
) -> None:
    x0, y0, x1, y1 = box
    lines = text.split("\n")
    heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    total_h = sum(heights) + spacing * (len(lines) - 1)
    y = y0 + ((y1 - y0) - total_h) // 2
    for line, w, h in zip(lines, widths, heights):
        draw.text((x0 + ((x1 - x0) - w) // 2, y), line, font=fnt, fill=fill)
        y += h + spacing


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    body: str,
    fill: str,
    outline: str,
) -> None:
    draw.rounded_rectangle(xy, radius=20, fill=fill, outline=outline, width=4)
    x0, y0, x1, y1 = xy
    draw.text((x0 + 24, y0 + 20), title, font=LABEL, fill="#111827")
    multiline_center(draw, (x0 + 24, y0 + 72, x1 - 24, y1 - 18), body, BODY)


def arrow(draw: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int], color: str = "#475569") -> None:
    draw.line([start, end], fill=color, width=5)
    x0, y0 = start
    x1, y1 = end
    dx, dy = x1 - x0, y1 - y0
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    size = 18
    p1 = (x1 - ux * size + px * size * 0.55, y1 - uy * size + py * size * 0.55)
    p2 = (x1 - ux * size - px * size * 0.55, y1 - uy * size - py * size * 0.55)
    draw.polygon([end, p1, p2], fill=color)


def main() -> None:
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    draw.text((80, 62), "TitanTPP prediction from quantity-bearing event history", font=TITLE, fill="#111827")
    draw.text(
        (80, 118),
        "Observed demand events are transformed into inter-event time, magnitude mark, and residual before next-event prediction.",
        font=SUBTITLE,
        fill="#4b5563",
    )

    rounded_box(
        draw,
        (80, 260, 620, 455),
        "Observed event sequence",
        "(t1, q1)   (t2, q2)   (t3, q3)\nquantity example: 3, 12, 68",
        "#fff7ed",
        "#c2410c",
    )
    rounded_box(
        draw,
        (80, 555, 620, 755),
        "Quantity representation",
        "m = floor(log_b q)\nr = log_b q - m\nDelta t = t_i - t_{i-1}",
        "#ecfdf5",
        "#047857",
    )
    rounded_box(
        draw,
        (80, 840, 620, 990),
        "Model token",
        "mark embedding\nlog(1+Delta t)\nresidual projection",
        "#eff6ff",
        "#2563eb",
    )

    rounded_box(
        draw,
        (820, 450, 1230, 660),
        "TitanTPP encoder",
        "causal memory attention\npersistent memory\nhistory state h_i",
        "#f8fafc",
        "#475569",
    )
    draw.text((820, 725), "matched baselines keep the same targets", font=SMALL, fill="#64748b")
    draw.text((820, 756), "but replace this encoder with GRU or Transformer", font=SMALL, fill="#64748b")

    rounded_box(
        draw,
        (1480, 245, 2060, 385),
        "Time head",
        "predict next inter-event time",
        "#fef2f2",
        "#b91c1c",
    )
    rounded_box(
        draw,
        (1480, 465, 2060, 605),
        "Magnitude-mark head",
        "predict magnitude-mark probability",
        "#eef2ff",
        "#4f46e5",
    )
    rounded_box(
        draw,
        (1480, 685, 2060, 825),
        "Residual path",
        "predict within-scale residual",
        "#f0fdf4",
        "#16a34a",
    )
    rounded_box(
        draw,
        (1480, 895, 2060, 1015),
        "Quantity reconstruction",
        "combine mark probability and residual",
        "#fffbeb",
        "#b45309",
    )

    arrow(draw, (350, 455), (350, 555))
    arrow(draw, (350, 755), (350, 840))
    arrow(draw, (620, 915), (820, 565))
    arrow(draw, (1230, 540), (1480, 315))
    arrow(draw, (1230, 555), (1480, 535))
    arrow(draw, (1230, 585), (1480, 755))
    arrow(draw, (1770, 605), (1770, 685))
    arrow(draw, (1770, 825), (1770, 895))

    draw.text((1300, 302), "next event time", font=SMALL, fill="#64748b")
    draw.text((1285, 500), "next magnitude", font=SMALL, fill="#64748b")
    draw.text((1245, 692), "within-scale quantity", font=SMALL, fill="#64748b")
    draw.text((1668, 850), "mark + residual", font=SMALL, fill="#64748b")

    png = OUT_DIR / "F1_titantpp_event_sequence_architecture.png"
    img.save(png)
    print(png)


if __name__ == "__main__":
    main()
