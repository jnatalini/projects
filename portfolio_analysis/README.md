# Portfolio Health CLI

Single-file command line application (`main.py`) that ingests a portfolio of tickers, fetches quotes and fundamentals from Yahoo Finance, and summarizes performance plus explainable recommendations.

## Features

- CSV or interactive portfolio input with validation (`ticker`, `quantity`, optional `avg_cost`, `asset_type`, `notes`).
- Yahoo Finance data retrieval via `yfinance` with lightweight caching.
- Metrics per holding: allocation, multi-window returns, volatility, max drawdown, beta vs benchmark, dividend yield, and ETF expense ratio when available.
- Portfolio overview: total value, unrealized P/L, weighted returns, alerts, and concentration checks.
- Rule-based recommendations for each holding with transparent reasons and confidence.
- CSV export of computed metrics and console-friendly reporting including disclaimers.

## Getting Started

```bash
cd portfolio_analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a portfolio CSV (see `sample_portfolio.csv`) and optional YAML config (`config/app_config.yml`). Paths above assume commands run from this directory.

### Run the analyzer

```bash
python main.py --portfolio sample_portfolio.csv --config config/app_config.yml --export reports/holdings.csv
```

If `--portfolio` is omitted, the CLI prompts for tickers interactively. Use `--risk-profile` to override the config (`Conservative`, `Balanced`, `Aggressive`).

Outputs are printed to the console, and exports land in the `reports/` directory. Remember the disclaimer: this tool is informational and not financial advice.

## Tests

```bash
pytest
```

Unit tests cover validation, metric calculations, and recommendation logic with mocked data.
