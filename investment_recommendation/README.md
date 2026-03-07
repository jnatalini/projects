# Stock Recommender CLI

`stock_recommender.py` is a single-file Python command-line tool that reads holdings performance data from CSV, enriches it with Yahoo Finance market context, computes a risk-adjusted investment score, and produces ranked recommendations.

## What It Does

- Validates and loads an input CSV of holdings.
- Normalizes and scores each ticker using:
  - `return_1m`, `return_1y`
  - `volatility_1y`, `max_drawdown_1y`, `beta`
- Pulls market data from Yahoo Finance:
  - current market price
  - 6-month daily close history
- Applies an overpricing check (trend/range based).
- Classifies each ticker as:
  - `INVEST`
  - `WATCH`
  - `AVOID`
- Writes a ranked output CSV.

## Requirements

- Python 3.8+ (tested with Python 3.10)
- Internet access to Yahoo Finance endpoints:
  - `query1.finance.yahoo.com`
- No external Python packages required (standard library only)

## Input CSV Requirements

The script expects a header row and at least these required columns:

- `ticker`
- `value`
- `weight`
- `return_1m`
- `return_1y`
- `volatility_1y`
- `max_drawdown_1y`
- `beta`

Optional columns are allowed and ignored by default for scoring.

### Input Validation Behavior

- Fails with a clear error if:
  - file does not exist
  - file is unreadable
  - required columns are missing
  - header contains duplicate column names
- Rows with missing ticker are skipped.
- Invalid numeric values are flagged and handled according to `--missing-policy`.

## Quick Start

Run from the project directory:

```bash
python3 stock_recommender.py \
  --input holdings_analysis.csv \
  --output recommended_stocks.csv \
  --risk-profile balanced \
  --strict-price-check \
  --summary
```

If `--output` is omitted, the script writes:

`recommendations_<input_filename_without_ext>.csv`

in the current directory.

## CLI Options

### Required

- `--input PATH`  
  Input CSV file path.

### Common

- `--output PATH`  
  Output CSV path.
- `--risk-profile {conservative,balanced,aggressive}`  
  Controls return-vs-risk emphasis.
- `--strict-price-check`  
  Requires market data status `OK` before any positive recommendation.
- `--summary`  
  Prints run summary to console.

### Missing Data Handling

- `--missing-policy {exclude,penalize}` (default: `penalize`)
  - `exclude`: rows with missing/invalid required numeric fields become `AVOID`.
  - `penalize`: apply score penalty.
- `--missing-penalty FLOAT` (default: `20.0`)  
  Penalty used when `--missing-policy penalize`.

### Price/Overheating Controls

- `--overprice-ma-ratio FLOAT` (default: `1.12`)  
  Current price / MA20 threshold for overheated signal.
- `--near-high-threshold FLOAT` (default: `0.93`)  
  Near-high position in 6-month range.
- `--elevated-penalty FLOAT` (default: `6.0`)
- `--overheated-penalty FLOAT` (default: `16.0`)

### Scoring Weights

- `--weight-return-1m FLOAT` (default: `25.0`)
- `--weight-return-1y FLOAT` (default: `35.0`)
- `--weight-volatility-1y FLOAT` (default: `15.0`)
- `--weight-max-drawdown-1y FLOAT` (default: `15.0`)
- `--weight-beta FLOAT` (default: `10.0`)

### Performance

- `--max-workers INT` (default: `8`)  
  Number of parallel Yahoo chart requests.

## Scoring Model

The script follows this structure:

`final_score = return_score - risk_penalty - overpricing_penalty - missing_penalty`

Where:

- `return_score` rewards stronger `return_1m` and `return_1y`.
- `risk_penalty` penalizes weaker risk characteristics from:
  - volatility
  - max drawdown
  - beta distance from profile target
- `overpricing_penalty` is based on Yahoo price context:
  - `fair` -> no penalty
  - `elevated` -> moderate penalty
  - `overheated` -> high penalty
- `missing_penalty` applies when required metrics are incomplete and policy is `penalize`.

### Risk Profiles

- `conservative`
  - lower return emphasis
  - stronger risk penalties
  - tighter beta target
- `balanced`
  - middle-ground defaults
- `aggressive`
  - higher return emphasis
  - lighter risk penalties
  - more tolerance for higher beta

## Recommendation Rules

- `INVEST` when score is high enough and no hard disqualifier is present.
- `WATCH` when middling score or strong score with overheating signal.
- `AVOID` when weak score, missing required metrics (with exclude policy), or strict price check fails.

If `--strict-price-check` is enabled and Yahoo data is unavailable for a ticker, it is forced to `AVOID`.

## Yahoo Finance Market Data Status

Each ticker gets a market-data status:

- `OK`
- `NOT_FOUND`
- `RATE_LIMITED`
- `DATA_INCOMPLETE`

The run continues even if some tickers fail market-data lookup.

## Output CSV Schema

The output is ranked from highest to lowest investment score and includes:

- `rank`
- `ticker`
- `current_price`
- `price_assessment`
- `investment_score`
- `recommendation`
- `return_1m`
- `return_1y`
- `volatility_1y`
- `max_drawdown_1y`
- `beta`
- `market_data_status`
- `rationale`

`rationale` is a concise plain-English explanation of main decision drivers.

## Console Summary (`--summary`)

When enabled, prints:

- number of tickers analyzed
- number recommended (`INVEST`)
- top candidates
- notable data/market lookup issues

## Example Commands

Balanced profile:

```bash
python3 stock_recommender.py --input holdings_analysis.csv --summary
```

Conservative with strict market validation:

```bash
python3 stock_recommender.py \
  --input holdings_analysis.csv \
  --risk-profile conservative \
  --strict-price-check \
  --output conservative_recommendations.csv \
  --summary
```

Aggressive with custom weights:

```bash
python3 stock_recommender.py \
  --input holdings_analysis.csv \
  --risk-profile aggressive \
  --weight-return-1m 30 \
  --weight-return-1y 40 \
  --weight-volatility-1y 10 \
  --weight-max-drawdown-1y 10 \
  --weight-beta 10 \
  --summary
```

## Exit Codes

- `0`: Success
- `2`: Invalid input or CSV schema error
- `3`: No usable rows after validation
- `4`: No analysis results produced
- `5`: Failed to write output CSV

## Troubleshooting

- `python: command not found`  
  Use `python3` instead.
- Many `DATA_INCOMPLETE` statuses  
  Check internet/DNS access to Yahoo endpoints.
- All recommendations become `AVOID` with strict mode  
  Likely due to unavailable market data plus `--strict-price-check`.
- Empty or unexpected output ranking  
  Verify required columns and numeric values in input CSV.

## Notes

- Results are deterministic for the same input data and same market data snapshot.
- Market prices and chart context change over time, so rankings can differ across runs.
