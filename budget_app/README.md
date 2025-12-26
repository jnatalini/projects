# Budget Tool (Bank of America Spending Screenshot Helper)

Single-script CLI that ingests Bank of America “Spending by Category” screenshots, builds a monthly history, forecasts next month’s budget, and generates budget-vs-actual reports. Everything runs locally.

## Requirements

- Python 3.8+
- OCR dependencies (optional but recommended):
  - `pytesseract`
  - `Pillow`
  - Tesseract OCR installed on your system
    - macOS: `brew install tesseract`
    - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
    - Windows: install Tesseract and ensure `tesseract.exe` is on PATH

Install Python packages:

```bash
pip install pytesseract pillow
```

Optional (for YAML config):

```bash
pip install pyyaml
```

## Quick Start

Ingest a screenshot and forecast the next month:

```bash
python budget_tool.py --input-image /path/to/screenshot.png --month 2025-12
```

Ingest multiple months (best for projections):

```bash
python budget_tool.py \
  --input-image /path/to/nov.png,/path/to/dec.png \
  --month 2025-11,2025-12
```

You can also repeat flags instead of comma-separating:

```bash
python budget_tool.py \
  --input-image /path/to/nov.png --month 2025-11 \
  --input-image /path/to/dec.png --month 2025-12
```

Run everything end-to-end (ingest, forecast, report):

```bash
python budget_tool.py --mode all --input-image /path/to/screenshot.png --month 2025-12
```

Generate a report vs budget for a specific month:

```bash
python budget_tool.py --mode report --report-month 2025-12
```

Disable interactive corrections:

```bash
python budget_tool.py --input-image /path/to/screenshot.png --month 2025-12 --no-interactive
```

## OCR Tips (So Every Category Is Captured)

- Capture the full category list and the "spent" amounts in the screenshot. Cropped or zoomed screenshots often drop category names.
- Zoom in enough that each category line is readable; blurry images usually reduce OCR to "Spent" only.
- If the screenshot shows both "spent" and "avg/mo", the tool uses the amount just before "spent".
- If OCR misses categories, use `--corrections` or the interactive editor to add them.

## Command-Line Options

- `--data-dir`: Storage directory (default: `./data`).
- `--input-image`: Screenshot path for ingestion (repeat or comma-separate for multiple).
- `--month`: Month for the input data (YYYY-MM); repeat or comma-separate to match multiple images.
- `--mode`: `ingest`, `forecast`, `report`, `all` (default: `all`).
- `--config`: JSON/YAML config file path.
- `--log-file`: Log file path (default: `./budget_tool.log`).
- `--log-level`: Log level (`INFO`, `DEBUG`, `WARNING`, `ERROR`).
- `--no-console-log`: Disable console output.
- `--corrections`: CSV or JSON file with category/amount overrides.
- `--no-interactive`: Skip interactive correction prompts.
- `--report-month`: Month for the budget vs actual report.

## Outputs

All outputs go under `--data-dir` (default `./data`):

- `spending_history.csv`: master history (month, category, amount).
- `YYYY-MM_spending.csv`: per-month spending snapshot.
- `budgets/budget_YYYY-MM.csv|json`: predicted budget for next month.
- `reports/report_YYYY-MM_vs_budget_YYYY-MM.csv|json`: budget vs actual report.
- `ocr_cache.json`: OCR cache to avoid reprocessing the same screenshot.
- `budget_tool.log`: run logs (location configurable).

## Corrections File

Use a corrections file to fix OCR output or add categories. JSON format:

```json
{
  "Groceries": 812.34,
  "Dining": 123.45
}
```

CSV format:

```csv
category,amount
Groceries,812.34
Dining,123.45
```

## Config File (Optional)

Example JSON:

```json
{
  "prediction_method": "enhanced",
  "rolling_window": 6,
  "trend_window": 6,
  "average_weight": 0.4,
  "trend_weight": 0.4,
  "seasonality_weight": 0.2,
  "inflation_rate": 0.03,
  "category_adjustments": {
    "Groceries": 1.05
  },
  "weighted_weights": [1, 2, 3],
  "sparse_min_months": 2,
  "sparse_strategy": "last",
  "adjustment_factor": 0.9,
  "overwrite_existing": true,
  "log_level": "INFO",
  "console_log": true
}
```

## Notes

- OCR is optional; if it fails, you can still edit results interactively or use a corrections file.
- Better forecasts require multiple months of history. With only one month, projections will match the most recent values.
- Prediction methods:
  - `last`: use the most recent month per category.
  - `rolling_average`: average of the last N months.
  - `weighted`: weighted average of the last N months.
  - `enhanced`: blends rolling average, trend, and seasonality, then applies inflation (default).
- Reports highlight total budget vs actual and top overspend categories.
  - If you do not pass `--report-month`, the report uses the latest month that has both actuals and a budget file.
