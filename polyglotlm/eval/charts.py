"""Inline SVG charts for the report card. Colors are CSS custom properties set by the page.

Mark specs: bars <= 14px thick with a 4px rounded data end and a 2px surface gap, 2px lines,
8px markers with a 2px surface ring, hairline solid gridlines. Every mark has a <title> for
hover, and every chart ships next to a data table (the page renders the tables).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape

WIDTH = 640
FONT = 'font-family="var(--font-mono)" font-size="11"'


@dataclass(frozen=True)
class Series:
    key: str  # stable entity id (language code): colour follows the entity, never its rank
    label: str
    points: list[tuple[float, float]]
    color: str  # CSS colour, normally var(--series-N)


def nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """Round tick values covering [low, high]."""
    if high <= low:
        high = low + 1.0
    raw = (high - low) / count
    magnitude = 10 ** math.floor(math.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    ticks = [math.floor(low / step) * step]
    while ticks[-1] < high - 1e-12:  # the last tick always covers the largest value
        ticks.append(round(ticks[-1] + step, 10))
    return ticks


def log_ticks(low: float, high: float) -> list[float]:
    """1-2-5 ticks per decade between low and high (both > 0)."""
    ticks = []
    for exponent in range(math.floor(math.log10(low)), math.ceil(math.log10(high)) + 1):
        for mantissa in (1, 2, 5):
            value = mantissa * 10**exponent
            if low <= value <= high:
                ticks.append(value)
    return ticks or [low, high]


def fmt(value: float) -> str:
    if float(value).is_integer():
        return f"{int(value):,}"
    if abs(value) >= 100:
        return f"{value:,.0f}"
    return f"{value:.1f}" if abs(value) >= 10 else f"{value:.2f}"


def _bar_path(x0: float, y: float, length: float, thickness: float) -> str:
    """Horizontal bar: square at the baseline (x0), 4px rounded at the data end."""
    r = min(4.0, thickness / 2, length)
    x1 = x0 + length
    return (
        f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
        f"V{y + thickness - r:.1f} Q{x1:.1f},{y + thickness:.1f} {x1 - r:.1f},{y + thickness:.1f} "
        f"H{x0:.1f} Z"
    )


def grouped_bars(
    rows: list[tuple[str, list[float]]],
    series: list[tuple[str, str]],
    end_labels: list[str],
    title_id: str,
    desc_id: str,
    unit: str,
) -> str:
    """Horizontal grouped bars: one row per category, one bar per (label, colour) series.

    `end_labels` holds one short label per row, drawn after the row's last bar (the story,
    e.g. the saving), never a number on every bar.
    """
    label_w, right_pad, top, bar, gap, row_gap = 96, 56, 8, 14, 2, 18
    row_h = len(series) * (bar + gap) - gap
    height = top + len(rows) * (row_h + row_gap) + 24
    vmax = max((v for _, values in rows for v in values), default=1.0)
    ticks = nice_ticks(0, vmax)
    scale = (WIDTH - label_w - right_pad) / ticks[-1]
    parts = [
        f'<svg viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="{title_id}" '
        f'aria-describedby="{desc_id}" class="chart">'
    ]
    axis_y = height - 20
    for tick in ticks:
        x = label_w + tick * scale
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{axis_y}" class="grid"/>'
            f'<text x="{x:.1f}" y="{axis_y + 14}" text-anchor="middle" class="tick" {FONT}>'
            f"{fmt(tick)}</text>"
        )
    for i, (category, values) in enumerate(rows):
        y0 = top + i * (row_h + row_gap)
        parts.append(
            f'<text x="{label_w - 10}" y="{y0 + row_h / 2 + 4:.1f}" text-anchor="end" '
            f'class="label">{escape(category)}</text>'
        )
        for j, ((name, color), value) in enumerate(zip(series, values, strict=True)):
            y = y0 + j * (bar + gap)
            parts.append(
                f'<path d="{_bar_path(label_w, y, max(value * scale, 1.0), bar)}" '
                f'style="fill:{color}"><title>{escape(category)} · {escape(name)}: '
                f"{fmt(value)} {escape(unit)}</title></path>"
            )
        if end_labels[i]:
            x = label_w + max(values) * scale + 8
            parts.append(
                f'<text x="{x:.1f}" y="{y0 + row_h / 2 + 4:.1f}" class="value" {FONT}>'
                f"{escape(end_labels[i])}</text>"
            )
    parts.append(
        f'<line x1="{label_w}" y1="{top}" x2="{label_w}" y2="{axis_y}" class="axis"/></svg>'
    )
    return "".join(parts)


def line_chart(series: list[Series], title_id: str, desc_id: str, log_y: bool = True) -> str:
    """Multi-series line chart with one y-axis.

    Log scale by default (perplexity spans orders of magnitude early in training), falling back
    to linear when the values cover less than a 10x range.
    """
    left, right, top, bottom, height = 56, 16, 12, 28, 260
    xs = [x for s in series for x, _ in s.points]
    ys = [y for s in series for _, y in s.points if y > 0]
    if not xs or not ys:
        return ""
    x_lo, x_hi = min(xs), max(xs) or 1.0
    y_lo, y_hi = min(ys), max(ys)
    log_y = log_y and y_hi / y_lo >= 10  # under one decade a log axis gets a single tick
    transform = math.log10 if log_y else (lambda v: v)
    t_lo, t_hi = transform(y_lo * 0.9), transform(y_hi * 1.1)

    def px(x: float) -> float:
        return left + (x - x_lo) / ((x_hi - x_lo) or 1.0) * (WIDTH - left - right)

    def py(y: float) -> float:
        return top + (t_hi - transform(y)) / ((t_hi - t_lo) or 1.0) * (height - top - bottom)

    parts = [
        f'<svg viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="{title_id}" '
        f'aria-describedby="{desc_id}" class="chart">'
    ]
    y_ticks = log_ticks(y_lo * 0.9, y_hi * 1.1) if log_y else nice_ticks(y_lo, y_hi)
    for tick in y_ticks:
        y = py(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{WIDTH - right}" y2="{y:.1f}" class="grid"/>'
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="tick" {FONT}>'
            f"{fmt(tick)}</text>"
        )
    for tick in nice_ticks(x_lo, x_hi):
        if x_lo <= tick <= x_hi:
            parts.append(
                f'<text x="{px(tick):.1f}" y="{height - 8}" text-anchor="middle" class="tick" '
                f"{FONT}>{fmt(tick)}</text>"
            )
    parts.append(
        f'<line x1="{left}" y1="{height - bottom}" x2="{WIDTH - right}" y2="{height - bottom}" '
        'class="axis"/>'
    )
    for s in series:
        points = [(x, y) for x, y in s.points if y > 0]
        path = " ".join(
            f"{'M' if i == 0 else 'L'}{px(x):.1f},{py(y):.1f}" for i, (x, y) in enumerate(points)
        )
        parts.append(f'<path d="{path}" class="line" style="stroke:{s.color}"/>')
        for x, y in points:
            parts.append(
                f'<g class="pt"><circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="12" class="hit"/>'
                f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="4" class="dot" '
                f'style="fill:{s.color}"/><title>{escape(s.label)} · step {fmt(x)}: '
                f"{fmt(y)}</title></g>"
            )
    parts.append("</svg>")
    return "".join(parts)
