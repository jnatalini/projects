#!/usr/bin/env python3
"""
Parse a daily stock portfolio analysis log and export a tidy CSV
containing portfolio-level and per-holding metrics for each snapshot.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path
from datetime import datetime


TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$")


def parse_currency(value: str) -> float:
    """Convert a currency string such as '$1,234.56' into a float."""
    cleaned = value.replace("$", "").replace(",", "")
    return float(cleaned)


def parse_log_lines(lines: list[str]) -> tuple[list[dict[str, float | str]], list[dict[str, float | str]]]:
    portfolio_records: list[dict[str, float | str]] = []
    holding_records: list[dict[str, float | str]] = []
    current_rows: list[dict[str, float | str]] = []
    total_value: float | None = None
    unrealized_pl: float | None = None
    table_active = False

    for raw_line in lines:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        if not stripped:
            if table_active:
                table_active = False
            continue

        if stripped.startswith("Total value:"):
            match = re.search(r"([\d,]+\.\d+)", stripped)
            if match:
                total_value = float(match.group(1).replace(",", ""))
            continue

        if stripped.startswith("Unrealized P/L:"):
            match = re.search(r"([-]?\d[\d,]*\.\d+)", stripped)
            if match:
                unrealized_pl = float(match.group(1).replace(",", ""))
            continue

        if stripped.startswith("Holdings overview:"):
            table_active = False
            continue

        if stripped.startswith("Ticker"):
            table_active = True
            continue

        if stripped.startswith("Portfolio recommendations:"):
            table_active = False
            continue

        if TIMESTAMP_RE.match(stripped):
            timestamp = stripped
            if total_value is not None and unrealized_pl is not None:
                portfolio_records.append(
                    {
                        "timestamp": timestamp,
                        "total_value": total_value,
                        "unrealized_pl": unrealized_pl,
                    }
                )
                for row in current_rows:
                    row_copy = dict(row)
                    row_copy["timestamp"] = timestamp
                    holding_records.append(row_copy)
            current_rows = []
            total_value = None
            unrealized_pl = None
            table_active = False
            continue

        if table_active:
            parts = stripped.split()
            if len(parts) < 4:
                continue
            ticker = parts[0]
            value_token = parts[1]
            weight_token = parts[2]
            recommendation = parts[-1]
            try:
                value = parse_currency(value_token)
            except ValueError:
                continue
            current_rows.append(
                {
                    "ticker": ticker,
                    "value": value,
                    "weight": weight_token,
                    "recommendation": recommendation,
                }
            )

    return portfolio_records, holding_records


def export_csv(records: list[dict[str, float | str]], output_path: Path, fieldnames: list[str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def add_timestamp_to_filename(path: Path, timestamp_suffix: str) -> Path:
    suffix = path.suffix or ".csv"
    stem = path.stem or path.name
    return path.with_name(f"{stem}_{timestamp_suffix}{suffix}")


def pivot_holdings(holding_records: list[dict[str, float | str]]) -> tuple[list[str], list[dict[str, float | str]]]:
    timestamps: list[str] = []
    ticker_values: dict[str, dict[str, float | str]] = {}
    for record in holding_records:
        timestamp = str(record["timestamp"])
        ticker = str(record["ticker"])
        value = record.get("value", "")
        if timestamp not in timestamps:
            timestamps.append(timestamp)
        ticker_entry = ticker_values.setdefault(ticker, {})
        ticker_entry[timestamp] = value

    rows: list[dict[str, float | str]] = []
    for ticker in sorted(ticker_values.keys()):
        row: dict[str, float | str] = {"ticker": ticker}
        for timestamp in timestamps:
            row[timestamp] = ticker_values[ticker].get(timestamp, "")
        rows.append(row)

    fieldnames = ["ticker"] + timestamps
    return fieldnames, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export portfolio holdings history from a text log into CSV."
    )
    parser.add_argument(
        "input_file",
        type=Path,
        help="Path to the stock analysis log file (e.g., stocklogs.txt).",
    )
    parser.add_argument(
        "--portfolio-output",
        type=Path,
        default=Path("portfolio_totals.csv"),
        help="Destination CSV for portfolio totals. Defaults to portfolio_totals.csv",
    )
    parser.add_argument(
        "--holdings-output",
        type=Path,
        default=Path("holding_details.csv"),
        help="Destination CSV for holdings details. Defaults to holding_details.csv",
    )
    args = parser.parse_args()

    lines = args.input_file.read_text(encoding="utf-8").splitlines()
    portfolio_records, holding_records = parse_log_lines(lines)
    if not portfolio_records:
        raise SystemExit("No portfolio snapshots found in the provided log.")

    timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    portfolio_path = add_timestamp_to_filename(args.portfolio_output, timestamp_suffix)
    holdings_path = add_timestamp_to_filename(args.holdings_output, timestamp_suffix)

    export_csv(
        portfolio_records,
        portfolio_path,
        ["timestamp", "total_value", "unrealized_pl"],
    )
    holdings_fields, holdings_rows = pivot_holdings(holding_records)
    export_csv(holdings_rows, holdings_path, holdings_fields)
    print(
        f"Wrote {len(portfolio_records)} portfolio rows to {portfolio_path} "
        f"and {len(holdings_rows)} holding rows to {holdings_path}"
    )


if __name__ == "__main__":
    main()
