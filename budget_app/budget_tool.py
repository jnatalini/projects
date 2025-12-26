#!/usr/bin/env python3
import argparse
import csv
import json
import logging
import re
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional dependency
    Image = None

try:
    import pytesseract
except Exception:  # pragma: no cover - optional dependency
    pytesseract = None


MONTH_NAMES = {
    "january": "01",
    "february": "02",
    "march": "03",
    "april": "04",
    "may": "05",
    "june": "06",
    "july": "07",
    "august": "08",
    "september": "09",
    "october": "10",
    "november": "11",
    "december": "12",
}


@dataclass
class SpendingRow:
    month: str
    category: str
    amount: float


def setup_logging(log_file: str, level: str, console: bool) -> logging.Logger:
    logger = logging.getLogger("budget_tool")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger


def load_config(config_path: Optional[str]) -> Dict[str, object]:
    config = {
        "prediction_method": "enhanced",
        "rolling_window": 6,
        "trend_window": 6,
        "average_weight": 0.4,
        "trend_weight": 0.4,
        "seasonality_weight": 0.2,
        "inflation_rate": 0.03,
        "category_adjustments": {},
        "weighted_weights": None,
        "sparse_min_months": 2,
        "sparse_strategy": "last",
        "adjustment_factor": 1.0,
        "overwrite_existing": True,
        "log_level": "INFO",
        "console_log": True,
    }
    if not config_path:
        return config
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if path.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyYAML required for YAML config files. Install with: pip install pyyaml") from exc
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    else:
        data = json.loads(path.read_text(encoding="utf-8")) or {}
    for key, value in data.items():
        config[key] = value
    return config


def normalize_category(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return ""
    value = re.sub(r"^[^A-Za-z]+", "", value)
    value = re.sub(r"^(Oo|O0|0o)\s+", "", value, flags=re.IGNORECASE)
    value = " ".join(value.split())
    return value.title()


def is_noise_category(category: str) -> bool:
    normalized = normalize_category(category)
    if not normalized:
        return True
    lower = normalized.lower()
    if lower.startswith("categories"):
        return True
    if "sort by" in lower:
        return True
    if lower in {"spent", "default", "default vv"}:
        return True
    return False


def normalize_csv_values(values: Optional[List[str]]) -> List[str]:
    if not values:
        return []
    results: List[str] = []
    for value in values:
        parts = [item.strip() for item in value.split(",")]
        results.extend([item for item in parts if item])
    return results


def parse_amount(text: str) -> Optional[float]:
    cleaned = text.replace("$", "").replace(",", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_ocr_text(text: str, logger: logging.Logger) -> Dict[str, float]:
    results: Dict[str, float] = {}
    amount_pattern = re.compile(r"(-?\$?\d[\d,]*(?:\.\d{2})?)")
    noise_pattern = re.compile(
        r"\b(quick view|set a budget|your avg/?mo|avg/?mo)\b",
        re.IGNORECASE,
    )
    last_category: Optional[str] = None
    pending_categories = deque()
    category_block: List[str] = []
    spent_amounts: List[float] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        lower = line.lower()
        if "avg/mo" in lower or "avgmo" in lower:
            continue
        cleaned_line = noise_pattern.sub("", line).strip()
        if not cleaned_line:
            continue
        if "spent" in lower:
            matches = list(amount_pattern.finditer(cleaned_line))
            if matches:
                spent_match = None
                spent_index = lower.find("spent")
                if spent_index != -1:
                    for match in matches:
                        if match.start() < spent_index:
                            spent_match = match
                        else:
                            break
                amount_match = spent_match or matches[0]
                amount_text = amount_match.group(1)
                amount = parse_amount(amount_text)
                if amount is not None:
                    spent_amounts.append(amount)
        matches = list(amount_pattern.finditer(cleaned_line))
        if not matches:
            if re.search(r"[A-Za-z]", cleaned_line) and "spent" not in lower:
                normalized = normalize_category(cleaned_line)
                if normalized and not is_noise_category(normalized):
                    pending_categories.append(normalized)
                    last_category = normalized
                    category_block.append(normalized)
            continue
        spent_match = None
        lower = cleaned_line.lower()
        spent_index = lower.find("spent")
        if spent_index != -1:
            for match in matches:
                if match.start() < spent_index:
                    spent_match = match
                else:
                    break
        amount_match = spent_match or matches[0]
        amount_text = amount_match.group(1)
        if "." not in amount_text and "$" not in amount_text and "spent" not in lower:
            continue
        amount = parse_amount(amount_text)
        if amount is None:
            logger.debug("Skipping unparsable amount line: %s", line)
            continue
        category_text = cleaned_line[: amount_match.start()].strip(" -\t")
        category = normalize_category(category_text)
        if not category and pending_categories:
            category = pending_categories.popleft()
        if not category and last_category:
            category = last_category
        if not category:
            logger.debug("Skipping line without category: %s", line)
            continue
        if is_noise_category(category):
            continue
        results[category] = amount
        last_category = None
    if spent_amounts and category_block and len(results) < len(spent_amounts):
        if len(category_block) >= len(spent_amounts):
            results = {
                category: amount
                for category, amount in zip(category_block, spent_amounts)
                if not is_noise_category(category)
            }
        else:
            logger.debug(
                "OCR fallback parse found %s categories for %s spent lines.",
                len(category_block),
                len(spent_amounts),
            )
    return results


def detect_month(text: str) -> Optional[str]:
    lower = text.lower()
    for name, number in MONTH_NAMES.items():
        if name in lower:
            match = re.search(rf"{name}\s+(\d{{4}})", lower)
            if match:
                year = match.group(1)
                return f"{year}-{number}"
    match = re.search(r"(20\d{2})[-/](\d{1,2})", text)
    if match:
        year, month = match.group(1), int(match.group(2))
        if 1 <= month <= 12:
            return f"{year}-{month:02d}"
    return None


def load_ocr_cache(cache_path: Path, logger: logging.Logger) -> Dict[str, Dict[str, object]]:
    if not cache_path.exists():
        return {}
    try:
        return json.loads(cache_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load OCR cache: %s", exc)
        return {}


def save_ocr_cache(cache_path: Path, cache: Dict[str, Dict[str, object]], logger: logging.Logger) -> None:
    try:
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save OCR cache: %s", exc)


def ocr_image(image_path: Path, cache_path: Path, logger: logging.Logger) -> str:
    if Image is None or pytesseract is None:
        raise RuntimeError("Pillow and pytesseract are required for OCR.")
    stat = image_path.stat()
    cache_key = f"{image_path}|{stat.st_size}|{int(stat.st_mtime)}"
    cache = load_ocr_cache(cache_path, logger)
    if cache_key in cache:
        return cache[cache_key].get("text", "")
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    cache[cache_key] = {"text": text}
    save_ocr_cache(cache_path, cache, logger)
    return text


def load_corrections(path: Optional[str]) -> Dict[str, float]:
    if not path:
        return {}
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Corrections file not found: {path}")
    if file_path.suffix.lower() == ".csv":
        with file_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return {normalize_category(row["category"]): float(row["amount"]) for row in reader if row.get("category")}
    data = json.loads(file_path.read_text(encoding="utf-8"))
    return {normalize_category(key): float(value) for key, value in data.items()}


def interactive_edit(entries: Dict[str, float]) -> Dict[str, float]:
    print("\nExtracted categories:")
    for category, amount in entries.items():
        print(f"  {category}: ${amount:.2f}")
    choice = input("Edit entries? (y/N): ").strip().lower()
    if choice != "y":
        return entries
    updated = dict(entries)
    while True:
        category = input("Enter category to add/update (blank to finish): ").strip()
        if not category:
            break
        category = normalize_category(category)
        amount_text = input("Enter amount (blank to remove): ").strip()
        if not amount_text:
            updated.pop(category, None)
            continue
        amount = parse_amount(amount_text)
        if amount is None:
            print("Invalid amount, try again.")
            continue
        updated[category] = amount
    return updated


def load_history(history_path: Path, logger: logging.Logger) -> List[SpendingRow]:
    if not history_path.exists():
        return []
    rows = []
    with history_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                month = row.get("month", "").strip()
                category = normalize_category(row.get("category", ""))
                amount = float(row.get("amount", "0"))
                if month and category:
                    rows.append(SpendingRow(month=month, category=category, amount=amount))
            except Exception as exc:
                logger.error("Skipping invalid history row %s (%s)", row, exc)
    return rows


def save_history(history_path: Path, rows: List[SpendingRow]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["month", "category", "amount"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"month": row.month, "category": row.category, "amount": f"{row.amount:.2f}"})


def update_history(
    history: List[SpendingRow],
    month: str,
    entries: Dict[str, float],
    overwrite: bool,
) -> List[SpendingRow]:
    updated = [row for row in history if not (row.month == month and row.category in entries and overwrite)]
    for category, amount in entries.items():
        updated.append(SpendingRow(month=month, category=category, amount=amount))
    return updated


def get_months(history: List[SpendingRow]) -> List[str]:
    return sorted({row.month for row in history})


def get_budget_months(data_dir: Path) -> List[str]:
    budgets_dir = data_dir / "budgets"
    if not budgets_dir.exists():
        return []
    months = []
    for path in budgets_dir.glob("budget_*.csv"):
        match = re.match(r"budget_(\d{4}-\d{2})\.csv$", path.name)
        if match:
            months.append(match.group(1))
    return sorted(set(months))


def month_to_date(month: str) -> datetime:
    return datetime.strptime(month, "%Y-%m")


def month_index(month: str) -> int:
    date = month_to_date(month)
    return date.year * 12 + date.month


def next_month(month: str) -> str:
    date = month_to_date(month)
    year = date.year + (1 if date.month == 12 else 0)
    month_num = 1 if date.month == 12 else date.month + 1
    return f"{year}-{month_num:02d}"


def month_range(start_month: str, end_month: str) -> List[str]:
    start = month_to_date(start_month)
    end = month_to_date(end_month)
    months = []
    current = start
    while current <= end:
        months.append(f"{current.year}-{current.month:02d}")
        year = current.year + (1 if current.month == 12 else 0)
        month_num = 1 if current.month == 12 else current.month + 1
        current = datetime(year, month_num, 1)
    return months


def linear_trend_predict(values: List[float]) -> float:
    if len(values) < 2:
        return values[-1] if values else 0.0
    x_vals = list(range(len(values)))
    mean_x = sum(x_vals) / len(x_vals)
    mean_y = sum(values) / len(values)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(x_vals, values))
    denominator = sum((x - mean_x) ** 2 for x in x_vals)
    slope = numerator / denominator if denominator else 0.0
    return values[-1] + slope


def group_history(history: List[SpendingRow]) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, Dict[str, float]] = defaultdict(dict)
    for row in history:
        grouped[row.month][row.category] = row.amount
    return grouped


def last_nonzero_amount(category: str, grouped: Dict[str, Dict[str, float]], months: List[str]) -> float:
    for month in reversed(months):
        value = grouped.get(month, {}).get(category, 0.0)
        if value:
            return value
    return 0.0


def compute_basic_stats(history: List[SpendingRow]) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    by_category: Dict[str, List[float]] = defaultdict(list)
    for row in history:
        by_category[row.category].append(row.amount)
    for category, values in by_category.items():
        total = sum(values)
        avg = total / len(values)
        stats[category] = {"total": total, "average": avg}
    return stats


def predict_budget(
    history: List[SpendingRow],
    config: Dict[str, object],
    logger: logging.Logger,
) -> Tuple[str, Dict[str, float], str]:
    months = get_months(history)
    if not months:
        raise RuntimeError("No history available to forecast.")
    latest_month = months[-1]
    target_month = next_month(latest_month)
    grouped = group_history(history)
    method = config.get("prediction_method", "rolling_average")
    window = int(config.get("rolling_window", 3))
    weights = config.get("weighted_weights")
    sparse_min = int(config.get("sparse_min_months", 2))
    sparse_strategy = config.get("sparse_strategy", "last")
    adjustment = float(config.get("adjustment_factor", 1.0))

    recent_months = months[-window:]
    categories = sorted({row.category for row in history if not is_noise_category(row.category)})
    predictions: Dict[str, float] = {}

    if method == "enhanced":
        all_months = month_range(months[0], months[-1])
        trend_window = int(config.get("trend_window", window))
        average_weight = float(config.get("average_weight", 0.4))
        trend_weight = float(config.get("trend_weight", 0.4))
        seasonality_weight = float(config.get("seasonality_weight", 0.2))
        inflation_rate = float(config.get("inflation_rate", 0.03))
        category_adjustments = config.get("category_adjustments", {})
        months_ahead = 1
        inflation_factor = (1 + inflation_rate) ** (months_ahead / 12)
        has_seasonal = any(month_index(month) == month_index(target_month) - 12 for month in all_months)
        logger.info(
            "Forecast assumptions: method=enhanced, window=%s, trend_window=%s, weights(avg=%.2f, trend=%.2f, season=%.2f), inflation=%.2f%% annual, seasonality=%s, sparse_strategy=%s, adjustment_factor=%.2f, category_adjustments=%s",
            window,
            trend_window,
            average_weight,
            trend_weight,
            seasonality_weight,
            inflation_rate * 100,
            "enabled" if has_seasonal else "not available",
            sparse_strategy,
            adjustment,
            "yes" if isinstance(category_adjustments, dict) and category_adjustments else "no",
        )

        for category in categories:
            series = [grouped.get(month, {}).get(category, 0.0) for month in all_months]
            non_zero = [value for value in series if value > 0]
            if len(non_zero) < sparse_min and sparse_strategy == "last":
                base = last_nonzero_amount(category, grouped, all_months)
            else:
                recent = series[-window:] if len(series) >= window else series
                avg_recent = sum(recent) / max(len(recent), 1)
                trend_values = series[-trend_window:] if len(series) >= trend_window else series
                trend_pred = linear_trend_predict(trend_values)
                season_value = 0.0
                season_weight = seasonality_weight
                season_month_index = month_index(target_month) - 12
                for month in all_months:
                    if month_index(month) == season_month_index:
                        season_value = grouped.get(month, {}).get(category, 0.0)
                        break
                if season_value == 0.0:
                    season_weight = 0.0
                total_weight = average_weight + trend_weight + season_weight
                if total_weight <= 0:
                    avg_weight = 1.0
                    trend_weight_norm = 0.0
                    season_weight_norm = 0.0
                else:
                    avg_weight = average_weight / total_weight
                    trend_weight_norm = trend_weight / total_weight
                    season_weight_norm = season_weight / total_weight
                base = (avg_recent * avg_weight) + (trend_pred * trend_weight_norm) + (season_value * season_weight_norm)
            base = max(base, 0.0) * inflation_factor
            if isinstance(category_adjustments, dict):
                base *= float(category_adjustments.get(category, 1.0))
            predictions[category] = base

        if adjustment != 1.0:
            predictions = {category: amount * adjustment for category, amount in predictions.items()}
        logger.info("Forecast method: enhanced (window=%s, trend_window=%s, inflation=%.2f)", window, trend_window, inflation_rate)
        return target_month, predictions, "enhanced"

    logger.info(
        "Forecast assumptions: method=%s, window=%s, sparse_strategy=%s, adjustment_factor=%.2f",
        method,
        window,
        sparse_strategy,
        adjustment,
    )
    for category in categories:
        values = [grouped.get(month, {}).get(category, 0.0) for month in recent_months]
        non_zero = [value for value in values if value > 0]
        if len(non_zero) < sparse_min and sparse_strategy == "last":
            value = last_nonzero_amount(category, grouped, months)
            predictions[category] = value
            continue
        if method == "last":
            predictions[category] = last_nonzero_amount(category, grouped, months)
        elif method == "weighted":
            weights_list = weights if isinstance(weights, list) and len(weights) == len(values) else list(range(1, len(values) + 1))
            weighted_sum = sum(v * w for v, w in zip(values, weights_list))
            weight_total = sum(weights_list) if weights_list else 1
            predictions[category] = weighted_sum / weight_total
        else:
            predictions[category] = sum(values) / max(len(values), 1)

    if adjustment != 1.0:
        predictions = {category: amount * adjustment for category, amount in predictions.items()}
    logger.info("Forecast method: %s (window=%s)", method, window)
    return target_month, predictions, method


def write_budget_files(data_dir: Path, month: str, predictions: Dict[str, float], method: str) -> Tuple[Path, Path]:
    budgets_dir = data_dir / "budgets"
    budgets_dir.mkdir(parents=True, exist_ok=True)
    csv_path = budgets_dir / f"budget_{month}.csv"
    json_path = budgets_dir / f"budget_{month}.json"
    rows = [{"category": category, "predicted_amount": amount, "method": method} for category, amount in predictions.items()]
    total_budget = sum(predictions.values())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "predicted_amount", "method"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"category": row["category"], "predicted_amount": f"{row['predicted_amount']:.2f}", "method": method})
    payload = {"rows": rows, "totals": {"total_budget": total_budget}}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def load_budget_file(data_dir: Path, month: str) -> Dict[str, float]:
    budgets_dir = data_dir / "budgets"
    csv_path = budgets_dir / f"budget_{month}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Budget file not found for {month}: {csv_path}")
    budgets: Dict[str, float] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            category = normalize_category(row.get("category", ""))
            amount = float(row.get("predicted_amount", "0"))
            if category:
                budgets[category] = amount
    return budgets


def compare_budget(
    actuals: Dict[str, float],
    budgets: Dict[str, float],
) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    categories = sorted(set(actuals.keys()) | set(budgets.keys()))
    rows = []
    sum_budget = 0.0
    sum_actual = 0.0
    for category in categories:
        budget = budgets.get(category, 0.0)
        actual = actuals.get(category, 0.0)
        diff = actual - budget
        pct = (diff / budget * 100) if budget else 0.0
        rows.append(
            {
                "category": category,
                "budget": budget,
                "actual": actual,
                "difference": diff,
                "percent_difference": pct,
            }
        )
        sum_budget += budget
        sum_actual += actual
    totals = {
        "sum_budget": sum_budget,
        "sum_actual": sum_actual,
        "total_difference": sum_actual - sum_budget,
        "total_percent_difference": ((sum_actual - sum_budget) / sum_budget * 100) if sum_budget else 0.0,
    }
    return rows, totals


def write_report(data_dir: Path, month: str, rows: List[Dict[str, object]], totals: Dict[str, float]) -> Tuple[Path, Path]:
    reports_dir = data_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / f"report_{month}_vs_budget_{month}.csv"
    json_path = reports_dir / f"report_{month}_vs_budget_{month}.json"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["category", "budget", "actual", "difference", "percent_difference"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "category": row["category"],
                    "budget": f"{row['budget']:.2f}",
                    "actual": f"{row['actual']:.2f}",
                    "difference": f"{row['difference']:.2f}",
                    "percent_difference": f"{row['percent_difference']:.2f}",
                }
            )
    payload = {"rows": rows, "totals": totals}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return csv_path, json_path


def ingest(
    data_dir: Path,
    input_image: Optional[str],
    month: Optional[str],
    corrections_path: Optional[str],
    interactive: bool,
    dump_ocr: bool,
    config: Dict[str, object],
    logger: logging.Logger,
) -> Tuple[List[SpendingRow], Optional[str]]:
    history_path = data_dir / "spending_history.csv"
    history = load_history(history_path, logger)
    if not input_image:
        logger.info("No input image provided for ingest.")
        return history, month
    image_path = Path(input_image)
    ocr_cache_path = data_dir / "ocr_cache.json"
    try:
        text = ocr_image(image_path, ocr_cache_path, logger)
    except Exception as exc:
        logger.error("OCR failed: %s", exc)
        if not interactive:
            return history, month
        text = ""
    if dump_ocr:
        ocr_text_path = data_dir / "ocr_last.txt"
        ocr_text_path.write_text(text, encoding="utf-8")
        print("\nOCR raw text:")
        print(text)
        logger.info("OCR raw text saved: %s", ocr_text_path)
    entries = parse_ocr_text(text, logger)
    if entries:
        logger.info("OCR categories detected (%s):", len(entries))
        for category, amount in sorted(entries.items()):
            logger.info("  %s: $%.2f", category, amount)
        print("\nOCR categories detected:")
        for category, amount in sorted(entries.items()):
            print(f"  {category}: ${amount:.2f}")
    else:
        print("\nOCR categories detected: none")
    if not month:
        month = detect_month(text)
    if not month:
        if interactive:
            month = input("Enter month for this data (YYYY-MM): ").strip()
        else:
            logger.error("Month not provided and not detected.")
            return history, None
    entries = {normalize_category(k): v for k, v in entries.items() if not is_noise_category(k)}
    if corrections_path:
        corrections = load_corrections(corrections_path)
        entries.update(corrections)
    if interactive:
        entries = interactive_edit(entries)
    overwrite = bool(config.get("overwrite_existing", True))
    updated_history = update_history(history, month, entries, overwrite)
    save_history(history_path, updated_history)
    monthly_path = data_dir / f"{month}_spending.csv"
    save_history(monthly_path, [row for row in updated_history if row.month == month])
    logger.info("Ingested %s categories for %s", len(entries), month)
    return updated_history, month


def forecast(data_dir: Path, history: List[SpendingRow], config: Dict[str, object], logger: logging.Logger) -> Optional[str]:
    if not history:
        logger.warning("No history available to forecast.")
        return None
    target_month, predictions, method = predict_budget(history, config, logger)
    csv_path, json_path = write_budget_files(data_dir, target_month, predictions, method)
    total_budget = sum(predictions.values())
    logger.info("Forecast saved: %s (total $%.2f)", target_month, total_budget)
    logger.info("Budget files: %s, %s", csv_path, json_path)
    print(f"\nProjected total budget for {target_month}: ${total_budget:.2f}")
    return target_month


def report(
    data_dir: Path,
    history: List[SpendingRow],
    month: Optional[str],
    logger: logging.Logger,
    allow_fallback: bool = True,
) -> Optional[str]:
    if not history:
        logger.warning("No history available to report.")
        return None
    months = get_months(history)
    target_month = month or (months[-1] if months else None)
    if not target_month:
        logger.error("No month available for reporting.")
        return None
    actuals = {row.category: row.amount for row in history if row.month == target_month}
    if not actuals:
        logger.error("No actual spending data for %s", target_month)
        return None
    try:
        budgets = load_budget_file(data_dir, target_month)
    except FileNotFoundError:
        if not allow_fallback:
            logger.error("Budget file not found for %s. Run forecast for that month or adjust --report-month.", target_month)
            return None
        budget_months = get_budget_months(data_dir)
        if not budget_months:
            logger.error("No budget files available for reporting.")
            return None
        common = sorted(set(months) & set(budget_months))
        if not common:
            logger.error("No months with both budget and actuals. Available budgets: %s", ", ".join(budget_months))
            return None
        fallback_month = common[-1]
        logger.warning("Budget file not found for %s; using latest month with budget and actuals: %s", target_month, fallback_month)
        target_month = fallback_month
        actuals = {row.category: row.amount for row in history if row.month == target_month}
        budgets = load_budget_file(data_dir, target_month)
    rows, totals = compare_budget(actuals, budgets)
    csv_path, json_path = write_report(data_dir, target_month, rows, totals)
    overspend = sorted(rows, key=lambda r: r["difference"], reverse=True)[:3]
    logger.info("Report saved: %s, %s", csv_path, json_path)
    print("\nBudget performance summary")
    print(f"Month: {target_month}")
    print(f"Total budget: ${totals['sum_budget']:.2f}")
    print(f"Total actual: ${totals['sum_actual']:.2f}")
    print(f"Difference: ${totals['total_difference']:.2f} ({totals['total_percent_difference']:.2f}%)")
    print("Top overspend categories:")
    for row in overspend:
        print(f"  {row['category']}: ${row['difference']:.2f}")
    return target_month


def main() -> int:
    parser = argparse.ArgumentParser(description="Budget tool for OCR-based monthly spending and forecasts.")
    parser.add_argument("--data-dir", default="./data", help="Directory for history, budgets, and reports.")
    parser.add_argument(
        "--input-image",
        action="append",
        help="Path to a Bank of America spending screenshot (repeat or comma-separate for multiple).",
    )
    parser.add_argument(
        "--month",
        action="append",
        help="Month for the input data (YYYY-MM); repeat or comma-separate to match multiple images.",
    )
    parser.add_argument("--mode", default="all", choices=["ingest", "forecast", "report", "all"], help="Run mode.")
    parser.add_argument("--config", help="Path to JSON/YAML config file.")
    parser.add_argument("--log-file", default="./budget_tool.log", help="Log file path.")
    parser.add_argument("--log-level", default="INFO", help="Log level.")
    parser.add_argument("--no-console-log", action="store_true", help="Disable console logging.")
    parser.add_argument("--corrections", help="Corrections file (CSV or JSON).")
    parser.add_argument("--no-interactive", action="store_true", help="Disable interactive prompts.")
    parser.add_argument("--report-month", help="Month for budget vs actual report (YYYY-MM).")
    parser.add_argument("--dump-ocr", action="store_true", help="Print OCR text and save it for debugging.")

    args = parser.parse_args()
    config = load_config(args.config)
    log_level = args.log_level or config.get("log_level", "INFO")
    console_log = not args.no_console_log if args.no_console_log is not None else bool(config.get("console_log", True))
    logger = setup_logging(args.log_file, log_level, console_log)

    data_dir = Path(args.data_dir)
    logger.info("Starting run mode: %s", args.mode)

    history = load_history(data_dir / "spending_history.csv", logger)
    logger.info("Loaded %s history rows", len(history))

    input_images = normalize_csv_values(args.input_image)
    input_months = normalize_csv_values(args.month)
    month = input_months[-1] if input_months else None
    if args.mode in {"ingest", "all"}:
        if input_months and len(input_images) > 1 and len(input_months) != len(input_images):
            logger.error("Provide the same number of months as images (got %s months, %s images).", len(input_months), len(input_images))
            return 1
        if not input_images:
            history, month = ingest(
                data_dir=data_dir,
                input_image=None,
                month=month,
                corrections_path=args.corrections,
                interactive=not args.no_interactive,
                dump_ocr=args.dump_ocr,
                config=config,
                logger=logger,
            )
        else:
            for index, image_path in enumerate(input_images):
                month_arg = input_months[index] if input_months else None
                history, month = ingest(
                    data_dir=data_dir,
                    input_image=image_path,
                    month=month_arg,
                    corrections_path=args.corrections,
                    interactive=not args.no_interactive,
                    dump_ocr=args.dump_ocr,
                    config=config,
                    logger=logger,
                )

    if args.mode in {"forecast", "all"}:
        forecast(data_dir, history, config, logger)

    if args.mode in {"report", "all"}:
        report(data_dir, history, args.report_month, logger, allow_fallback=args.report_month is None)

    logger.info("Run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
