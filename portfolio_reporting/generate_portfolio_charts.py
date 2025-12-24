#!/usr/bin/env python3
"""
Generate HTML line charts from portfolio and holdings CSV exports.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


@dataclass
class Series:
    title: str
    x_labels: list[str]
    y_values: list[float | None]


def parse_timestamp(raw: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def format_timestamp(raw: str) -> str:
    parsed = parse_timestamp(raw)
    if parsed:
        return parsed.strftime("%Y-%m-%d %H:%M")
    return raw


def read_portfolio_series(path: Path) -> list[Series]:
    timestamps: list[str] = []
    total_values: list[float | None] = []
    unrealized_values: list[float | None] = []

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            timestamp = row.get("timestamp", "").strip()
            if not timestamp:
                continue
            timestamps.append(format_timestamp(timestamp))
            try:
                total_values.append(float(row.get("total_value", "")))
            except (TypeError, ValueError):
                total_values.append(None)
            try:
                unrealized_values.append(float(row.get("unrealized_pl", "")))
            except (TypeError, ValueError):
                unrealized_values.append(None)

    return [
        Series("Total Value", timestamps, total_values),
        Series("Unrealized P/L", timestamps, unrealized_values),
    ]


def read_holdings_series(path: Path) -> list[Series]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)

    if not rows:
        return []

    header = rows[0]
    if len(header) < 2:
        return []

    x_labels = [format_timestamp(value) for value in header[1:]]
    series: list[Series] = []
    for row in rows[1:]:
        if not row:
            continue
        title = row[0].strip() or "Unknown"
        values: list[float | None] = []
        for raw in row[1:]:
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(None)
        series.append(Series(title, x_labels, values))
    return series


def iter_valid(values: Iterable[float | None]) -> list[float]:
    return [value for value in values if value is not None]


def svg_line_chart(series: Series, width: int = 900, height: int = 260) -> str:
    padding_left = 72
    padding_right = 36
    padding_top = 36
    padding_bottom = 48
    plot_width = max(1, width - padding_left - padding_right)
    plot_height = max(1, height - padding_top - padding_bottom)

    valid_values = iter_valid(series.y_values)
    if not valid_values:
        return (
            f'<div class="chart"><h2>{series.title}</h2>'
            f'<div class="empty">No data available.</div></div>'
        )

    min_value = min(valid_values)
    max_value = max(valid_values)
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0

    def x_pos(index: int, count: int) -> float:
        if count <= 1:
            return padding_left + plot_width / 2
        return padding_left + (plot_width * index / (count - 1))

    def y_pos(value: float) -> float:
        return padding_top + ((max_value - value) / (max_value - min_value)) * plot_height

    points = []
    for index, value in enumerate(series.y_values):
        if value is None:
            continue
        points.append(f"{x_pos(index, len(series.y_values)):.2f},{y_pos(value):.2f}")

    polyline = " ".join(points)
    y_ticks = 4
    y_tick_labels = []
    for step in range(y_ticks + 1):
        frac = step / y_ticks
        value = min_value + (max_value - min_value) * (1 - frac)
        y = padding_top + plot_height * frac
        y_tick_labels.append((y, f"{value:,.2f}"))

    x_tick_labels = []
    if series.x_labels:
        tick_count = min(4, len(series.x_labels))
        for step in range(tick_count):
            index = round(step * (len(series.x_labels) - 1) / max(1, tick_count - 1))
            label = series.x_labels[index]
            x = x_pos(index, len(series.x_labels))
            x_tick_labels.append((x, label))

    svg_parts = [
        f'<div class="chart"><h2>{series.title}</h2>',
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" role="img">',
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />',
        f'<line x1="{padding_left}" y1="{padding_top}" '
        f'x2="{padding_left}" y2="{height - padding_bottom}" class="axis" />',
        f'<line x1="{padding_left}" y1="{height - padding_bottom}" '
        f'x2="{width - padding_right}" y2="{height - padding_bottom}" class="axis" />',
    ]

    for y, label in y_tick_labels:
        svg_parts.append(
            f'<line x1="{padding_left - 4}" y1="{y:.2f}" '
            f'x2="{padding_left}" y2="{y:.2f}" class="tick" />'
        )
        svg_parts.append(
            f'<text x="{padding_left - 8}" y="{y + 4:.2f}" '
            f'text-anchor="end" class="tick-label">{label}</text>'
        )

    for x, label in x_tick_labels:
        svg_parts.append(
            f'<line x1="{x:.2f}" y1="{height - padding_bottom}" '
            f'x2="{x:.2f}" y2="{height - padding_bottom + 4}" class="tick" />'
        )
        svg_parts.append(
            f'<text x="{x:.2f}" y="{height - padding_bottom + 18}" '
            f'text-anchor="middle" class="tick-label">{label}</text>'
        )

    svg_parts.append(f'<polyline points="{polyline}" class="line" />')
    svg_parts.append("</svg></div>")
    return "".join(svg_parts)


def render_page(title: str, series_list: list[Series]) -> str:
    charts = "\n".join(svg_line_chart(series) for series in series_list)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title}</title>
  <style>
    :root {{
      color-scheme: light;
    }}
    body {{
      font-family: "Georgia", "Times New Roman", serif;
      margin: 24px;
      background: #f8f4ef;
      color: #2b2b2b;
    }}
    h1 {{
      font-size: 28px;
      margin-bottom: 16px;
    }}
    .chart {{
      margin-bottom: 28px;
      padding: 18px 20px;
      background: #ffffff;
      border: 1px solid #e0d9cf;
      border-radius: 10px;
      box-shadow: 0 6px 20px rgba(31, 24, 16, 0.08);
    }}
    .chart h2 {{
      font-size: 20px;
      margin: 0 0 8px 0;
    }}
    svg {{
      max-width: 100%;
      height: auto;
      display: block;
    }}
    .axis {{
      stroke: #8e857a;
      stroke-width: 1;
    }}
    .tick {{
      stroke: #8e857a;
      stroke-width: 1;
    }}
    .tick-label {{
      font-size: 11px;
      fill: #6a5f51;
    }}
    .line {{
      fill: none;
      stroke: #2f5d50;
      stroke-width: 2.2;
    }}
    .empty {{
      padding: 20px 0;
      color: #6a5f51;
      font-style: italic;
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  {charts}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate HTML line charts from portfolio and holdings CSV data."
    )
    parser.add_argument(
        "--portfolio-csv",
        type=Path,
        default=Path("./stockreport/portfolio_20251223_223807.csv"),
        help="Path to portfolio CSV containing timestamp,total_value,unrealized_pl.",
    )
    parser.add_argument(
        "--holdings-csv",
        type=Path,
        default=Path("./stockreport/holdings_20251223_223807.csv"),
        help="Path to holdings CSV with ticker followed by timestamp columns.",
    )
    parser.add_argument(
        "--portfolio-html",
        type=Path,
        default=Path("./stockreport/portfolio_charts.html"),
        help="Output HTML file for portfolio charts.",
    )
    parser.add_argument(
        "--holdings-html",
        type=Path,
        default=Path("./stockreport/holdings_charts.html"),
        help="Output HTML file for holdings charts.",
    )
    args = parser.parse_args()

    portfolio_series = read_portfolio_series(args.portfolio_csv)
    holdings_series = read_holdings_series(args.holdings_csv)

    portfolio_html = render_page("Portfolio History", portfolio_series)
    holdings_html = render_page("Holdings History", holdings_series)

    args.portfolio_html.write_text(portfolio_html, encoding="utf-8")
    args.holdings_html.write_text(holdings_html, encoding="utf-8")

    print(f"Wrote {args.portfolio_html} and {args.holdings_html}")


if __name__ == "__main__":
    main()
