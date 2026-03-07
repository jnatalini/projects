#!/usr/bin/env python3
"""Rank portfolio holdings into INVEST/WATCH/AVOID recommendations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

REQUIRED_COLUMNS = [
    "ticker",
    "value",
    "weight",
    "return_1m",
    "return_1y",
    "volatility_1y",
    "max_drawdown_1y",
    "beta",
]

NUMERIC_COLUMNS = [
    "value",
    "weight",
    "return_1m",
    "return_1y",
    "volatility_1y",
    "max_drawdown_1y",
    "beta",
]

DEFAULT_WEIGHTS = {
    "return_1m": 25.0,
    "return_1y": 35.0,
    "volatility_1y": 15.0,
    "max_drawdown_1y": 15.0,
    "beta": 10.0,
}

RISK_CONFIG = {
    "conservative": {
        "return_mult": 0.85,
        "risk_mult": 1.35,
        "beta_target": 0.85,
        "beta_tolerance": 0.20,
        "invest_threshold": 22.0,
        "watch_threshold": 8.0,
        "risk_hot_threshold": 0.52,
    },
    "balanced": {
        "return_mult": 1.00,
        "risk_mult": 1.00,
        "beta_target": 1.00,
        "beta_tolerance": 0.28,
        "invest_threshold": 20.0,
        "watch_threshold": 5.0,
        "risk_hot_threshold": 0.60,
    },
    "aggressive": {
        "return_mult": 1.20,
        "risk_mult": 0.75,
        "beta_target": 1.20,
        "beta_tolerance": 0.40,
        "invest_threshold": 18.0,
        "watch_threshold": 2.0,
        "risk_hot_threshold": 0.68,
    },
}

YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (stock-recommender-script)"}


@dataclass
class MarketData:
    status: str
    current_price: Optional[float]
    price_assessment: str
    overpricing_penalty: float
    notes: str


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rank holdings into INVEST/WATCH/AVOID using CSV metrics and Yahoo "
            "Finance price context."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file.")
    parser.add_argument(
        "--output",
        help="Path to output CSV file. Defaults to ./recommendations_<input-name>.csv",
    )
    parser.add_argument(
        "--risk-profile",
        choices=sorted(RISK_CONFIG.keys()),
        default="balanced",
        help="Risk profile controls return-vs-risk emphasis.",
    )
    parser.add_argument(
        "--strict-price-check",
        action="store_true",
        help="Require Yahoo price data to consider INVEST/WATCH recommendations.",
    )
    parser.add_argument(
        "--missing-policy",
        choices=["exclude", "penalize"],
        default="penalize",
        help="How to treat missing required numeric fields.",
    )
    parser.add_argument(
        "--missing-penalty",
        type=positive_float,
        default=20.0,
        help="Penalty applied when --missing-policy=penalize.",
    )
    parser.add_argument(
        "--overprice-ma-ratio",
        type=positive_float,
        default=1.12,
        help="Price/MA threshold considered overheated.",
    )
    parser.add_argument(
        "--near-high-threshold",
        type=float,
        default=0.93,
        help="Position in 6m range above this value is near-high [0..1].",
    )
    parser.add_argument(
        "--elevated-penalty",
        type=positive_float,
        default=6.0,
        help="Penalty for elevated price assessment.",
    )
    parser.add_argument(
        "--overheated-penalty",
        type=positive_float,
        default=16.0,
        help="Penalty for overheated price assessment.",
    )
    parser.add_argument(
        "--weight-return-1m",
        type=positive_float,
        default=DEFAULT_WEIGHTS["return_1m"],
    )
    parser.add_argument(
        "--weight-return-1y",
        type=positive_float,
        default=DEFAULT_WEIGHTS["return_1y"],
    )
    parser.add_argument(
        "--weight-volatility-1y",
        type=positive_float,
        default=DEFAULT_WEIGHTS["volatility_1y"],
    )
    parser.add_argument(
        "--weight-max-drawdown-1y",
        type=positive_float,
        default=DEFAULT_WEIGHTS["max_drawdown_1y"],
    )
    parser.add_argument(
        "--weight-beta",
        type=positive_float,
        default=DEFAULT_WEIGHTS["beta"],
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Parallel Yahoo chart requests.",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print console summary after output generation.",
    )
    args = parser.parse_args()
    if not 0.0 <= args.near_high_threshold <= 1.0:
        parser.error("--near-high-threshold must be in range [0, 1].")
    if args.max_workers <= 0:
        parser.error("--max-workers must be greater than zero.")
    return args


def ensure_input_readable(path: str) -> None:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    if not os.access(path, os.R_OK):
        raise PermissionError(f"Input file is not readable: {path}")


def read_header(path: str) -> List[str]:
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
    if not header:
        raise ValueError("Input CSV is empty or missing header row.")
    return [h.strip() for h in header]


def validate_columns(header: List[str]) -> None:
    dupes = sorted({name for name in header if header.count(name) > 1})
    if dupes:
        raise ValueError(f"Input CSV has duplicated columns: {', '.join(dupes)}")
    missing = [name for name in REQUIRED_COLUMNS if name not in header]
    if missing:
        raise ValueError(f"Input CSV missing required columns: {', '.join(missing)}")
    malformed = [name for name in REQUIRED_COLUMNS if not name.strip()]
    if malformed:
        raise ValueError("Input CSV contains malformed empty column names.")


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        number = float(text)
        if math.isnan(number) or math.isinf(number):
            return None
        return number
    except ValueError:
        return None


def normalize_ticker(raw: object) -> str:
    if raw is None:
        return ""
    text = str(raw).strip().upper()
    return text.replace(" ", "")


def load_rows(path: str) -> Tuple[List[dict], List[str]]:
    rows: List[dict] = []
    issues: List[str] = []
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader, start=2):
            ticker = normalize_ticker(row.get("ticker"))
            if not ticker:
                issues.append(f"Row {idx}: missing ticker, skipped.")
                continue
            clean = dict(row)
            clean["ticker"] = ticker
            clean["_rownum"] = idx
            clean["_invalid_numeric"] = []
            clean["_missing_numeric"] = []
            for col in NUMERIC_COLUMNS:
                parsed = safe_float(row.get(col))
                clean[col] = parsed
                if parsed is None:
                    if str(row.get(col, "")).strip() == "":
                        clean["_missing_numeric"].append(col)
                    else:
                        clean["_invalid_numeric"].append(col)
            if clean["_invalid_numeric"]:
                issues.append(
                    f"Row {idx} ({ticker}): invalid numeric fields "
                    f"{', '.join(clean['_invalid_numeric'])}."
                )
            rows.append(clean)
    return rows, issues


def minmax(values: List[Optional[float]], invert: bool = False) -> List[Optional[float]]:
    valid = [v for v in values if v is not None]
    if not valid:
        return [None for _ in values]
    lo = min(valid)
    hi = max(valid)
    if math.isclose(lo, hi):
        base = [0.5 if v is not None else None for v in values]
    else:
        span = hi - lo
        base = [((v - lo) / span) if v is not None else None for v in values]
    if invert:
        return [(1.0 - v) if v is not None else None for v in base]
    return base


def request_json(url: str) -> dict:
    req = urllib.request.Request(url, headers=HTTP_HEADERS)
    with urllib.request.urlopen(req, timeout=12) as response:
        body = response.read()
    return json.loads(body)


def fetch_chart_context(ticker: str) -> Tuple[str, Optional[List[float]], str, Optional[float]]:
    url = YAHOO_CHART_URL.format(ticker=urllib.parse.quote(ticker))
    url += "?range=6mo&interval=1d"
    try:
        data = request_json(url)
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return "NOT_FOUND", None, "ticker not found in chart endpoint", None
        if err.code == 429:
            return "RATE_LIMITED", None, "rate limited by Yahoo Finance", None
        return "DATA_INCOMPLETE", None, f"http error {err.code}", None
    except urllib.error.URLError:
        return "DATA_INCOMPLETE", None, "network error", None
    except json.JSONDecodeError:
        return "DATA_INCOMPLETE", None, "invalid chart response", None

    chart = data.get("chart", {})
    if chart.get("error"):
        code = str(chart["error"].get("code", ""))
        if "Not Found" in code:
            return "NOT_FOUND", None, "ticker not found", None
        return "DATA_INCOMPLETE", None, f"chart error: {code}", None
    results = chart.get("result")
    if not results:
        return "DATA_INCOMPLETE", None, "missing chart result", None

    result = results[0]
    quotes = result.get("indicators", {}).get("quote", [])
    if not quotes:
        return "DATA_INCOMPLETE", None, "missing quote history", None
    closes = [safe_float(v) for v in quotes[0].get("close", [])]
    closes = [v for v in closes if v is not None]
    if len(closes) < 30:
        return "DATA_INCOMPLETE", None, "insufficient close history", None

    # regularMarketPrice is Yahoo's current market context from chart metadata.
    current_price = safe_float(result.get("meta", {}).get("regularMarketPrice"))
    if current_price is None:
        current_price = closes[-1]
    return "OK", closes, "", current_price


def assess_price(
    current_price: Optional[float],
    closes: Optional[List[float]],
    row_risk_penalty_component: float,
    args: argparse.Namespace,
    risk_cfg: dict,
) -> Tuple[str, float, str]:
    if current_price is None or not closes:
        return "unknown", 0.0, "price context unavailable"

    window = closes[-20:] if len(closes) >= 20 else closes
    ma20 = sum(window) / len(window)
    low = min(closes)
    high = max(closes)
    near_high = 0.5 if math.isclose(high, low) else (current_price - low) / (high - low)
    ma_ratio = current_price / ma20 if ma20 else 1.0

    risky = row_risk_penalty_component >= risk_cfg["risk_hot_threshold"]
    if ma_ratio >= args.overprice_ma_ratio or (
        near_high >= args.near_high_threshold and risky
    ):
        return "overheated", args.overheated_penalty, (
            f"price {ma_ratio:.2f}x MA20 and range position {near_high:.2f}"
        )
    if ma_ratio >= (args.overprice_ma_ratio - 0.05) or near_high >= (
        args.near_high_threshold - 0.05
    ):
        return "elevated", args.elevated_penalty, (
            f"price moderately elevated ({ma_ratio:.2f}x MA20)"
        )
    return "fair", 0.0, "price near recent trend/range"


def build_rationale(
    row: dict,
    recommendation: str,
    price_assessment: str,
    market_status: str,
    missing_cols: List[str],
) -> str:
    parts: List[str] = []
    r1m = row.get("return_1m")
    r1y = row.get("return_1y")
    vol = row.get("volatility_1y")
    dd = row.get("max_drawdown_1y")
    beta = row.get("beta")

    if r1y is not None:
        parts.append(f"1Y return {r1y * 100:.1f}%")
    if r1m is not None:
        parts.append(f"1M return {r1m * 100:.1f}%")
    if vol is not None:
        parts.append("stable volatility" if vol <= 0.2 else "high volatility")
    if dd is not None:
        parts.append("contained drawdown" if dd >= -0.15 else "deep drawdown")
    if beta is not None:
        parts.append("moderate beta" if beta <= 1.1 else "high beta")

    if missing_cols:
        parts.append(f"missing metrics: {', '.join(missing_cols)}")
    if market_status != "OK":
        parts.append(f"market data {market_status.lower()}")
    else:
        if price_assessment == "fair":
            parts.append("price looks fair")
        elif price_assessment == "elevated":
            parts.append("price looks elevated")
        elif price_assessment == "overheated":
            parts.append("price appears overheated")

    if recommendation == "AVOID" and not parts:
        parts.append("insufficient quality and risk profile")

    return "; ".join(parts[:5])


def determine_recommendation(
    score: float,
    price_assessment: str,
    market_status: str,
    missing_cols: List[str],
    args: argparse.Namespace,
    risk_cfg: dict,
) -> str:
    if missing_cols and args.missing_policy == "exclude":
        return "AVOID"
    if args.strict_price_check and market_status != "OK":
        return "AVOID"
    if price_assessment == "overheated":
        return "WATCH" if score >= risk_cfg["invest_threshold"] else "AVOID"
    if score >= risk_cfg["invest_threshold"]:
        return "INVEST"
    if score >= risk_cfg["watch_threshold"]:
        return "WATCH"
    return "AVOID"


def print_summary(results: List[dict], issues: List[str]) -> None:
    analyzed = len(results)
    recommended = sum(1 for r in results if r["recommendation"] == "INVEST")
    skipped = [line for line in issues if "skipped" in line.lower()]
    top5 = [r for r in results if r["recommendation"] != "AVOID"][:5]
    print("\nSummary")
    print(f"- tickers analyzed: {analyzed}")
    print(f"- tickers recommended (INVEST): {recommended}")
    print(f"- tickers skipped (invalid/missing ticker): {len(skipped)}")
    if top5:
        print("- top candidates:")
        for row in top5:
            print(
                f"  {row['rank']}. {row['ticker']} ({row['recommendation']}, "
                f"score={row['investment_score']:.2f})"
            )
    if issues:
        print("- notable data issues:")
        for issue in issues[:10]:
            print(f"  {issue}")


def resolve_output_path(input_path: str, output_path: Optional[str]) -> str:
    if output_path:
        return output_path
    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(os.getcwd(), f"recommendations_{base}.csv")


def main() -> int:
    args = parse_args()
    try:
        ensure_input_readable(args.input)
        header = read_header(args.input)
        validate_columns(header)
        rows, issues = load_rows(args.input)
    except Exception as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2

    if not rows:
        print("ERROR: no usable rows after input validation.", file=sys.stderr)
        return 3

    ticker_list = sorted({r["ticker"] for r in rows})
    chart_context: Dict[str, Tuple[str, Optional[List[float]], str, Optional[float]]] = {}
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(fetch_chart_context, t): t for t in ticker_list}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                chart_context[ticker] = future.result()
            except Exception as err:  # pragma: no cover
                chart_context[ticker] = (
                    "DATA_INCOMPLETE",
                    None,
                    f"unexpected error: {err}",
                    None,
                )

    risk_cfg = RISK_CONFIG[args.risk_profile]
    metric_r1m = [r.get("return_1m") for r in rows]
    metric_r1y = [r.get("return_1y") for r in rows]
    metric_vol = [r.get("volatility_1y") for r in rows]
    metric_dd = [abs(r["max_drawdown_1y"]) if r.get("max_drawdown_1y") is not None else None for r in rows]
    metric_beta_dev = [
        abs(r["beta"] - risk_cfg["beta_target"]) if r.get("beta") is not None else None
        for r in rows
    ]

    n_r1m = minmax(metric_r1m)
    n_r1y = minmax(metric_r1y)
    n_vol = minmax(metric_vol, invert=True)
    n_dd = minmax(metric_dd, invert=True)
    n_beta = minmax(metric_beta_dev, invert=True)

    # Use medians as neutral fallback when missing-policy=penalize.
    fallback = {
        "n_r1m": statistics.median([v for v in n_r1m if v is not None]) if any(v is not None for v in n_r1m) else 0.5,
        "n_r1y": statistics.median([v for v in n_r1y if v is not None]) if any(v is not None for v in n_r1y) else 0.5,
        "n_vol": statistics.median([v for v in n_vol if v is not None]) if any(v is not None for v in n_vol) else 0.5,
        "n_dd": statistics.median([v for v in n_dd if v is not None]) if any(v is not None for v in n_dd) else 0.5,
        "n_beta": statistics.median([v for v in n_beta if v is not None]) if any(v is not None for v in n_beta) else 0.5,
    }

    output: List[dict] = []
    for idx, row in enumerate(rows):
        ticker = row["ticker"]
        missing_cols = sorted(set(row["_missing_numeric"] + row["_invalid_numeric"]))
        if missing_cols and args.missing_policy == "exclude":
            base_return_score = 0.0
            risk_penalty = 0.0
            row_risk_component = 1.0
        else:
            row_n_r1m = n_r1m[idx] if n_r1m[idx] is not None else fallback["n_r1m"]
            row_n_r1y = n_r1y[idx] if n_r1y[idx] is not None else fallback["n_r1y"]
            row_n_vol = n_vol[idx] if n_vol[idx] is not None else fallback["n_vol"]
            row_n_dd = n_dd[idx] if n_dd[idx] is not None else fallback["n_dd"]
            row_n_beta = n_beta[idx] if n_beta[idx] is not None else fallback["n_beta"]

            base_return_score = risk_cfg["return_mult"] * (
                args.weight_return_1m * row_n_r1m + args.weight_return_1y * row_n_r1y
            )
            risk_penalty = risk_cfg["risk_mult"] * (
                args.weight_volatility_1y * (1 - row_n_vol)
                + args.weight_max_drawdown_1y * (1 - row_n_dd)
                + args.weight_beta * (1 - row_n_beta)
            )
            row_risk_component = (
                ((1 - row_n_vol) + (1 - row_n_dd) + (1 - row_n_beta)) / 3.0
            )

        status, closes, market_note, current_price = chart_context.get(
            ticker, ("DATA_INCOMPLETE", None, "missing chart context", None)
        )
        price_assessment, overpricing_penalty, price_note = assess_price(
            current_price, closes, row_risk_component, args, risk_cfg
        )
        if status != "OK":
            price_assessment = "unknown"
            overpricing_penalty = 0.0
        missing_penalty = args.missing_penalty if missing_cols and args.missing_policy == "penalize" else 0.0

        final_score = base_return_score - risk_penalty - overpricing_penalty - missing_penalty
        recommendation = determine_recommendation(
            final_score, price_assessment, status, missing_cols, args, risk_cfg
        )
        rationale = build_rationale(row, recommendation, price_assessment, status, missing_cols)

        if status != "OK" and market_note:
            issues.append(f"{ticker}: {market_note}")

        output.append(
            {
                "ticker": ticker,
                "current_price": current_price,
                "price_assessment": price_assessment,
                "investment_score": final_score,
                "recommendation": recommendation,
                "return_1m": row.get("return_1m"),
                "return_1y": row.get("return_1y"),
                "volatility_1y": row.get("volatility_1y"),
                "max_drawdown_1y": row.get("max_drawdown_1y"),
                "beta": row.get("beta"),
                "market_data_status": status,
                "rationale": rationale,
                "_sort_score": final_score,
                "_sort_ticker": ticker,
                "_diagnostic": f"{price_note}",
            }
        )

    if not output:
        print("ERROR: no analysis results could be produced.", file=sys.stderr)
        return 4

    output.sort(key=lambda r: (-r["_sort_score"], r["_sort_ticker"]))
    for i, row in enumerate(output, start=1):
        row["rank"] = i

    out_path = resolve_output_path(args.input, args.output)
    fieldnames = [
        "rank",
        "ticker",
        "current_price",
        "price_assessment",
        "investment_score",
        "recommendation",
        "return_1m",
        "return_1y",
        "volatility_1y",
        "max_drawdown_1y",
        "beta",
        "market_data_status",
        "rationale",
    ]
    try:
        with open(out_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in output:
                serial = {k: row.get(k) for k in fieldnames}
                writer.writerow(serial)
    except Exception as err:
        print(f"ERROR: failed to write output CSV: {err}", file=sys.stderr)
        return 5

    print(
        "Scoring assumptions: "
        f"risk_profile={args.risk_profile}, "
        f"weights=(1m:{args.weight_return_1m},1y:{args.weight_return_1y},"
        f"vol:{args.weight_volatility_1y},dd:{args.weight_max_drawdown_1y},beta:{args.weight_beta}), "
        f"missing_policy={args.missing_policy}, strict_price_check={args.strict_price_check}"
    )
    print(f"Output written to: {out_path}")
    if args.summary:
        print_summary(output, issues)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
