from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger("portfolio-cli")
logging.getLogger("yfinance").setLevel(logging.CRITICAL)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass
class CacheSettings:
    quote_ttl_minutes: int = 15
    fundamentals_ttl_minutes: int = 60 * 24
    cache_dir: Path = Path.home() / ".cache" / "portfolio_analyzer"


@dataclass
class Thresholds:
    high_weight_pct: float = 0.25
    watch_weight_pct: float = 0.15
    high_volatility: float = 0.35
    high_drawdown: float = 0.2
    bullish_momentum: float = 0.05
    bearish_momentum: float = -0.05
    neutral_band: float = 0.01
    high_beta: float = 1.2
    low_beta: float = 0.8
    high_expense_ratio: float = 0.0075


@dataclass
class AppConfig:
    base_currency: str = "USD"
    benchmark: str = "SPY"
    risk_profile: str = "Balanced"
    quote_history_period: str = "1y"
    thresholds: Thresholds = field(default_factory=Thresholds)
    cache: CacheSettings = field(default_factory=CacheSettings)


def load_app_config(path: Path | None) -> AppConfig:
    if path is None:
        return AppConfig()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    def merge(base: Dict, override: Dict) -> Dict:
        result = dict(base)
        for key, value in override.items():
            if isinstance(value, dict) and isinstance(base.get(key), dict):
                result[key] = merge(base[key], value)
            else:
                result[key] = value
        return result

    defaults = {
        "base_currency": "USD",
        "benchmark": "SPY",
        "risk_profile": "Balanced",
        "quote_history_period": "1y",
        "thresholds": Thresholds().__dict__,
        "cache": {
            "quote_ttl_minutes": CacheSettings().quote_ttl_minutes,
            "fundamentals_ttl_minutes": CacheSettings().fundamentals_ttl_minutes,
            "cache_dir": str(CacheSettings().cache_dir),
        },
    }
    merged = merge(defaults, raw)
    thresholds = Thresholds(**merged["thresholds"])
    cache = CacheSettings(
        quote_ttl_minutes=merged["cache"]["quote_ttl_minutes"],
        fundamentals_ttl_minutes=merged["cache"]["fundamentals_ttl_minutes"],
        cache_dir=Path(merged["cache"]["cache_dir"]),
    )
    return AppConfig(
        base_currency=merged["base_currency"],
        benchmark=merged["benchmark"],
        risk_profile=merged["risk_profile"],
        quote_history_period=merged["quote_history_period"],
        thresholds=thresholds,
        cache=cache,
    )


# -----------------------------------------------------------------------------
# Data models
# -----------------------------------------------------------------------------
class HoldingInput(BaseModel):
    ticker: str = Field(..., description="Ticker symbol, e.g., AAPL")
    quantity: float = Field(..., ge=0, description="Quantity of shares")
    avg_cost: Optional[float] = Field(None, ge=0)
    purchase_date: Optional[str] = None
    asset_type: Optional[str] = None
    notes: Optional[str] = None

    @field_validator("ticker", mode="before")
    @classmethod
    def ensure_ticker(cls, value: str) -> str:
        cleaned = str(value).strip().upper()
        if not cleaned:
            raise ValueError("Ticker cannot be empty")
        return cleaned

    @field_validator("asset_type", "notes", mode="before")
    @classmethod
    def optional_strings(cls, value):
        if value is None:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        return str(value)


class PortfolioInput(BaseModel):
    holdings: List[HoldingInput]


@dataclass
class HoldingAnalysis:
    ticker: str
    name: Optional[str] = None
    asset_type: Optional[str] = None
    last_price: Optional[float] = None
    prev_close: Optional[float] = None
    value: Optional[float] = None
    weight: Optional[float] = None
    change_1d_pct: Optional[float] = None
    returns: Dict[str, Optional[float]] = field(default_factory=dict)
    volatility_1y: Optional[float] = None
    max_drawdown_1y: Optional[float] = None
    beta: Optional[float] = None
    dividend_yield: Optional[float] = None
    expense_ratio: Optional[float] = None
    avg_cost: Optional[float] = None
    cost_basis: Optional[float] = None
    unrealized_pl: Optional[float] = None
    unrealized_pl_pct: Optional[float] = None
    recommendation_label: str = "Review"
    recommendation_reasons: List[str] = field(default_factory=list)
    recommendation_confidence: str = "Low"
    data_quality_flags: List[str] = field(default_factory=list)
    alerts: List[str] = field(default_factory=list)


@dataclass
class PortfolioRecommendation:
    label: str
    reasons: List[str]


@dataclass
class PortfolioAnalysis:
    total_value: float
    total_cost_basis: Optional[float]
    total_unrealized_pl: Optional[float]
    holdings: List[HoldingAnalysis]
    allocation: Dict[str, float]
    portfolio_returns: Dict[str, Optional[float]]
    volatility_1y: Optional[float]
    max_drawdown_1y: Optional[float]
    beta_vs_benchmark: Optional[float]
    benchmark_ticker: str
    benchmark_returns: Dict[str, Optional[float]]
    recommendations: List[PortfolioRecommendation]
    alerts: List[str]


# -----------------------------------------------------------------------------
# Data retrieval
# -----------------------------------------------------------------------------
@dataclass
class MarketData:
    ticker: str
    info: Dict
    fast_info: Dict
    history: pd.DataFrame
    dividends: pd.Series
    actions: pd.DataFrame
    calendar: Dict


class CacheEntry:
    def __init__(self, data, timestamp: float):
        self.data = data
        self.timestamp = timestamp


class CacheManager:
    def __init__(self):
        self._cache: Dict[str, CacheEntry] = {}

    def get(self, key: str, ttl_minutes: int):
        entry = self._cache.get(key)
        if not entry:
            return None
        if time.time() - entry.timestamp > ttl_minutes * 60:
            return None
        return entry.data

    def set(self, key: str, data):
        self._cache[key] = CacheEntry(data, time.time())


class YahooFinanceClient:
    def __init__(self, config: AppConfig):
        self.config = config
        self.cache = CacheManager()

    def get_market_data(self, ticker: str, period: Optional[str] = None) -> MarketData:
        period = period or self.config.quote_history_period
        cache_key = f"{ticker}:{period}"
        cached = self.cache.get(cache_key, self.config.cache.quote_ttl_minutes)
        if cached:
            return cached

        ticker_obj = yf.Ticker(ticker)
        try:
            history = ticker_obj.history(period=period, interval="1d", auto_adjust=False)
        except Exception as exc:  # pragma: no cover
            LOGGER.warning("Failed to download history for %s: %s", ticker, exc)
            history = pd.DataFrame()

        info = {}
        try:
            info = ticker_obj.info or {}
        except Exception:  # pragma: no cover
            LOGGER.debug("Ticker %s info unavailable", ticker)

        fast_info = {}
        try:
            fast_info = dict(getattr(ticker_obj, "fast_info", {}) or {})
        except Exception:
            LOGGER.debug("Ticker %s fast info unavailable", ticker)

        dividends = pd.Series(dtype=float)
        try:
            dividends = ticker_obj.dividends
        except Exception:
            LOGGER.debug("Ticker %s dividends unavailable", ticker)

        actions = pd.DataFrame()
        try:
            actions = ticker_obj.actions
        except Exception:
            LOGGER.debug("Ticker %s actions unavailable", ticker)

        calendar = {}
        try:
            calendar_df = ticker_obj.calendar
            if calendar_df is not None and not calendar_df.empty:
                calendar = calendar_df.to_dict().get("Value", {})
        except Exception:
            LOGGER.debug("Ticker %s calendar unavailable", ticker)

        market_data = MarketData(
            ticker=ticker,
            info=info,
            fast_info=fast_info,
            history=history,
            dividends=dividends,
            actions=actions,
            calendar=calendar,
        )
        self.cache.set(cache_key, market_data)
        return market_data

    def get_history(self, ticker: str, period: Optional[str] = None) -> pd.DataFrame:
        return self.get_market_data(ticker, period).history


# -----------------------------------------------------------------------------
# Analytics & recommendations
# -----------------------------------------------------------------------------
RETURN_WINDOWS = {
    "1d": 1,
    "1w": 5,
    "1m": 21,
    "3m": 63,
    "1y": 252,
}


def _price_at(history: pd.DataFrame, days_ago: int) -> Optional[float]:
    if history is None or history.empty or "Close" not in history.columns:
        return None
    history = history.dropna(subset=["Close"])
    if history.empty:
        return None
    last_date = history.index[-1]
    target = last_date - pd.Timedelta(days=days_ago)
    prior = history.loc[history.index <= target]
    if prior.empty:
        return None
    return float(prior["Close"].iloc[-1])


def compute_returns(history: pd.DataFrame) -> Dict[str, Optional[float]]:
    if history is None or history.empty or "Close" not in history.columns:
        return {label: None for label in RETURN_WINDOWS}
    closes = history["Close"].dropna()
    if closes.empty:
        return {label: None for label in RETURN_WINDOWS}
    latest = float(closes.iloc[-1])
    result: Dict[str, Optional[float]] = {}
    for label, days in RETURN_WINDOWS.items():
        price_then = _price_at(history, days)
        if price_then and price_then > 0:
            result[label] = (latest - price_then) / price_then
        else:
            result[label] = None
    return result


def annualized_volatility(history: pd.DataFrame) -> Optional[float]:
    if history is None or history.empty or "Close" not in history.columns:
        return None
    closes = history["Close"].dropna()
    if len(closes) < 3:
        return None
    returns = closes.pct_change().dropna()
    if returns.empty:
        return None
    return float(np.std(returns) * np.sqrt(252))


def max_drawdown(history: pd.DataFrame) -> Optional[float]:
    if history is None or history.empty or "Close" not in history.columns:
        return None
    closes = history["Close"].dropna()
    if closes.empty:
        return None
    running_max = closes.cummax()
    drawdowns = closes / running_max - 1
    return float(drawdowns.min())


def beta_vs_benchmark(history: pd.DataFrame, benchmark_history: pd.DataFrame) -> Optional[float]:
    if history is None or benchmark_history is None:
        return None
    if "Close" not in history.columns or "Close" not in benchmark_history.columns:
        return None
    stock_returns = history["Close"].pct_change().dropna()
    benchmark_returns = benchmark_history["Close"].pct_change().dropna()
    joined = pd.concat([stock_returns, benchmark_returns], axis=1, join="inner")
    joined.columns = ["stock", "benchmark"]
    joined = joined.dropna()
    if len(joined) < 20:
        return None
    var_bench = np.var(joined["benchmark"])
    if var_bench == 0:
        return None
    cov = np.cov(joined["stock"], joined["benchmark"])[0][1]
    return float(cov / var_bench)


def _confidence(analysis: HoldingAnalysis) -> str:
    metrics = [
        analysis.last_price,
        analysis.returns.get("1m"),
        analysis.returns.get("3m"),
        analysis.volatility_1y,
        analysis.max_drawdown_1y,
        analysis.beta,
    ]
    available = sum(metric is not None for metric in metrics)
    if available >= 5:
        return "High"
    if available >= 3:
        return "Medium"
    return "Low"


def build_recommendation(analysis: HoldingAnalysis, config: AppConfig) -> HoldingAnalysis:
    thresholds = config.thresholds
    reasons: List[str] = []
    label = "Hold"

    if analysis.last_price is None:
        analysis.recommendation_label = "Review"
        analysis.recommendation_reasons = ["Price data unavailable."]
        analysis.recommendation_confidence = "Low"
        return analysis

    momentum = analysis.returns.get("3m")
    long_term = analysis.returns.get("1y")
    weight = analysis.weight

    if analysis.change_1d_pct is not None and analysis.change_1d_pct < -0.05:
        reasons.append(f"Daily change {analysis.change_1d_pct:.1%} triggered watch threshold.")

    if weight and weight > thresholds.high_weight_pct:
        reasons.append(
            f"Position weight {weight:.1%} exceeds {thresholds.high_weight_pct:.0%} limit."
        )
        label = "Reduce"
    elif weight and weight > thresholds.watch_weight_pct:
        reasons.append(
            f"Position weight {weight:.1%} above watch level {thresholds.watch_weight_pct:.0%}."
        )

    if momentum is not None:
        if momentum >= thresholds.bullish_momentum:
            reasons.append(
                f"3M return {momentum:.1%} beats bullish threshold {thresholds.bullish_momentum:.0%}."
            )
            if label == "Hold":
                label = "Add"
        elif momentum <= thresholds.bearish_momentum:
            reasons.append(
                f"3M return {momentum:.1%} below bearish threshold {thresholds.bearish_momentum:.0%}."
            )
            label = "Watch"

    if long_term is not None:
        if long_term < 0 and label == "Hold":
            label = "Watch"
            reasons.append(f"1Y return {long_term:.1%} negative.")
        elif long_term > thresholds.bullish_momentum * 2:
            reasons.append(f"1Y return {long_term:.1%} highlights longer-term strength.")

    if analysis.volatility_1y and analysis.volatility_1y > thresholds.high_volatility:
        reasons.append(
            f"Volatility {analysis.volatility_1y:.1%} > {thresholds.high_volatility:.0%}."
        )
        if label == "Hold":
            label = "Watch"

    if analysis.max_drawdown_1y and abs(analysis.max_drawdown_1y) > thresholds.high_drawdown:
        reasons.append(
            f"Max drawdown {analysis.max_drawdown_1y:.1%} breaches {thresholds.high_drawdown:.0%}."
        )
        label = "Watch" if label in {"Hold", "Add"} else label

    if analysis.beta is not None:
        if analysis.beta > thresholds.high_beta:
            reasons.append(f"Beta {analysis.beta:.2f} > {thresholds.high_beta:.2f}, high sensitivity.")
        elif analysis.beta < thresholds.low_beta:
            reasons.append(f"Beta {analysis.beta:.2f} < {thresholds.low_beta:.2f}, defensive.")

    if (
        analysis.expense_ratio is not None
        and analysis.expense_ratio > thresholds.high_expense_ratio
    ):
        reasons.append(
            f"Expense ratio {analysis.expense_ratio:.2%} > {thresholds.high_expense_ratio:.2%}."
        )
        label = "Reduce"

    if not reasons:
        reasons.append("Signals neutral; maintaining Hold stance.")

    analysis.recommendation_label = label
    analysis.recommendation_reasons = reasons[:5]
    analysis.recommendation_confidence = _confidence(analysis)
    return analysis


# -----------------------------------------------------------------------------
# Portfolio analyzer
# -----------------------------------------------------------------------------
@dataclass
class AnalysisSettings:
    risk_profile: Optional[str] = None


class PortfolioAnalyzer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.client = YahooFinanceClient(config)

    def _aggregate(self, holdings: List[HoldingInput]) -> List[HoldingInput]:
        aggregated: Dict[str, HoldingInput] = {}
        for holding in holdings:
            if holding.ticker in aggregated:
                existing = aggregated[holding.ticker]
                total_qty = existing.quantity + holding.quantity
                avg_cost = None
                if existing.avg_cost is not None and holding.avg_cost is not None:
                    avg_cost = (
                        existing.avg_cost * existing.quantity
                        + holding.avg_cost * holding.quantity
                    ) / total_qty
                else:
                    avg_cost = existing.avg_cost or holding.avg_cost
                aggregated[holding.ticker] = HoldingInput(
                    ticker=existing.ticker,
                    quantity=total_qty,
                    avg_cost=avg_cost,
                    purchase_date=existing.purchase_date or holding.purchase_date,
                    asset_type=existing.asset_type or holding.asset_type,
                    notes=existing.notes or holding.notes,
                )
            else:
                aggregated[holding.ticker] = holding
        return list(aggregated.values())

    def analyze(self, portfolio: PortfolioInput, settings: AnalysisSettings) -> PortfolioAnalysis:
        holdings = self._aggregate(portfolio.holdings)
        benchmark_history = self.client.get_history(self.config.benchmark)

        analyses: List[HoldingAnalysis] = []
        total_value = 0.0
        total_cost = 0.0
        cost_available = True

        for holding in holdings:
            market_data = self.client.get_market_data(holding.ticker)
            history = market_data.history
            last_price = None
            prev_close = None
            data_flags: List[str] = []

            if history is not None and not history.empty:
                close_series = history["Close"].dropna()
                if not close_series.empty:
                    last_price = float(close_series.iloc[-1])
                    if len(close_series) > 1:
                        prev_close = float(close_series.iloc[-2])
            else:
                data_flags.append("Price history unavailable from Yahoo Finance.")

            if market_data.fast_info:
                last_price = market_data.fast_info.get("lastPrice") or last_price
                prev_close = market_data.fast_info.get("previousClose") or prev_close

            if last_price is None:
                data_flags.append("Latest price unavailable; ticker may be invalid or delayed.")

            returns = compute_returns(history)
            volatility = annualized_volatility(history)
            drawdown = max_drawdown(history)
            beta = beta_vs_benchmark(history, benchmark_history)

            value = last_price * holding.quantity if last_price is not None else None
            total_value += value or 0.0

            cost_basis = (
                holding.avg_cost * holding.quantity if holding.avg_cost is not None else None
            )
            if cost_basis is None:
                cost_available = False
            else:
                total_cost += cost_basis

            change_1d = None
            if last_price and prev_close:
                change_1d = (last_price - prev_close) / prev_close if prev_close else None

            dividend_yield = market_data.info.get("dividendYield")
            if dividend_yield and dividend_yield > 1:
                dividend_yield = dividend_yield / 100

            expense_ratio = market_data.info.get("annualReportExpenseRatio")
            if expense_ratio and expense_ratio > 1:
                expense_ratio = expense_ratio / 100

            name = market_data.info.get("shortName") or market_data.info.get("longName")
            asset_type = (
                holding.asset_type
                or market_data.info.get("quoteType")
                or market_data.info.get("securityType")
            )

            unrealized_pl = None
            unrealized_pct = None
            if cost_basis is not None and value is not None:
                unrealized_pl = value - cost_basis
                if cost_basis != 0:
                    unrealized_pct = unrealized_pl / cost_basis

            analysis = HoldingAnalysis(
                ticker=holding.ticker,
                name=name,
                asset_type=asset_type,
                last_price=last_price,
                prev_close=prev_close,
                value=value,
                change_1d_pct=change_1d,
                returns=returns,
                volatility_1y=volatility,
                max_drawdown_1y=drawdown,
                beta=beta,
                dividend_yield=dividend_yield,
                expense_ratio=expense_ratio,
                avg_cost=holding.avg_cost,
                cost_basis=cost_basis,
                unrealized_pl=unrealized_pl,
                unrealized_pl_pct=unrealized_pct,
                data_quality_flags=data_flags,
            )

            analyses.append(analysis)

        for holding in analyses:
            if holding.value is not None and total_value > 0:
                holding.weight = holding.value / total_value
            build_recommendation(holding, self.config)

        allocation = {
            holding.ticker: holding.weight for holding in analyses if holding.weight is not None
        }

        portfolio_returns = {}
        for window in RETURN_WINDOWS:
            weighted_sum = 0.0
            weight_total = 0.0
            for holding in analyses:
                ret = holding.returns.get(window)
                if ret is None or holding.weight is None:
                    continue
                weighted_sum += holding.weight * ret
                weight_total += holding.weight
            portfolio_returns[window] = weighted_sum if weight_total > 0 else None

        volatility = None
        vol_components = [
            (holding.weight or 0.0, holding.volatility_1y)
            for holding in analyses
            if holding.volatility_1y is not None and holding.weight is not None
        ]
        if vol_components:
            volatility = sum(weight * vol for weight, vol in vol_components)

        drawdown = None
        draw_components = [
            (holding.weight or 0.0, holding.max_drawdown_1y)
            for holding in analyses
            if holding.max_drawdown_1y is not None and holding.weight is not None
        ]
        if draw_components:
            drawdown = sum(weight * val for weight, val in draw_components)

        beta = None
        beta_components = [
            (holding.weight or 0.0, holding.beta)
            for holding in analyses
            if holding.beta is not None and holding.weight is not None
        ]
        if beta_components:
            beta = sum(weight * val for weight, val in beta_components)

        recommendations = self._portfolio_recommendations(analyses)
        benchmark_returns = compute_returns(benchmark_history)

        total_unrealized = None
        if cost_available:
            total_unrealized = total_value - total_cost

        return PortfolioAnalysis(
            total_value=total_value,
            total_cost_basis=total_cost if cost_available else None,
            total_unrealized_pl=total_unrealized,
            holdings=analyses,
            allocation=allocation,
            portfolio_returns=portfolio_returns,
            volatility_1y=volatility,
            max_drawdown_1y=drawdown,
            beta_vs_benchmark=beta,
            benchmark_ticker=self.config.benchmark,
            benchmark_returns=benchmark_returns,
            recommendations=recommendations,
            alerts=self._collect_alerts(analyses),
        )

    def _portfolio_recommendations(self, holdings: List[HoldingAnalysis]) -> List[PortfolioRecommendation]:
        recs: List[PortfolioRecommendation] = []
        if not holdings:
            return recs

        sorted_holdings = sorted(holdings, key=lambda h: h.weight or 0, reverse=True)
        top = sorted_holdings[0]
        if top.weight and top.weight > self.config.thresholds.high_weight_pct:
            recs.append(
                PortfolioRecommendation(
                    label="Concentration Risk",
                    reasons=[
                        f"{top.ticker} weight {top.weight:.1%} exceeds "
                        f"{self.config.thresholds.high_weight_pct:.0%}.",
                        "Consider trimming or rebalancing.",
                    ],
                )
            )

        losers = [
            h for h in holdings if h.unrealized_pl_pct is not None and h.unrealized_pl_pct < -0.15
        ]
        if losers:
            tickers = ", ".join(h.ticker for h in losers[:5])
            recs.append(
                PortfolioRecommendation(
                    label="Losers Watchlist",
                    reasons=[f"{tickers} below -15% unrealized return; reassess thesis."],
                )
            )
        return recs

    def _collect_alerts(self, holdings: List[HoldingAnalysis]) -> List[str]:
        alerts: List[str] = []
        for holding in holdings:
            if holding.change_1d_pct is not None and holding.change_1d_pct <= -0.07:
                alerts.append(f"{holding.ticker} dropped {holding.change_1d_pct:.1%} today.")
            if holding.max_drawdown_1y is not None and holding.max_drawdown_1y <= -0.3:
                alerts.append(f"{holding.ticker} drawdown {holding.max_drawdown_1y:.1%} over 1Y.")
        return alerts


# -----------------------------------------------------------------------------
# Reporting / CLI helpers
# -----------------------------------------------------------------------------
DISCLAIMER = (
    "This tool provides informational analysis only and does not constitute investment advice. "
    "Markets are volatile; validate before acting."
)


def _format_pct(value):
    if value is None:
        return "—"
    return f"{value:.1%}"


def _format_currency(value):
    if value is None:
        return "—"
    return f"${value:,.2f}"


def display_portfolio(analysis: PortfolioAnalysis):
    print(DISCLAIMER)
    print("=" * 80)
    print(f"Total value: {_format_currency(analysis.total_value)}")
    if analysis.total_unrealized_pl is not None and analysis.total_cost_basis:
        pct = (
            analysis.total_unrealized_pl / analysis.total_cost_basis
            if analysis.total_cost_basis
            else None
        )
        print(
            f"Unrealized P/L: {_format_currency(analysis.total_unrealized_pl)} "
            f"({_format_pct(pct)})"
        )
    print()

    rows = []
    for holding in analysis.holdings:
        rows.append(
            {
                "Ticker": holding.ticker,
                "Value": _format_currency(holding.value),
                "Weight": _format_pct(holding.weight),
                "1M": _format_pct(holding.returns.get("1m")),
                "1Y": _format_pct(holding.returns.get("1y")),
                "Vol (1Y)": _format_pct(holding.volatility_1y),
                "Drawdown (1Y)": _format_pct(holding.max_drawdown_1y),
                "Recommendation": holding.recommendation_label,
            }
        )
    if rows:
        df = pd.DataFrame(rows)
        print("Holdings overview:")
        print(df.to_string(index=False))
        print()

    if analysis.recommendations:
        print("Portfolio recommendations:")
        for rec in analysis.recommendations:
            print(f"- {rec.label}: {'; '.join(rec.reasons)}")
        print()

    flagged = [(h.ticker, h.data_quality_flags) for h in analysis.holdings if h.data_quality_flags]
    if flagged:
        print("Data quality notices:")
        for ticker, flags in flagged:
            print(f"- {ticker}: {'; '.join(flags)}")
        print()

    if analysis.alerts:
        print("Alerts:")
        for alert in analysis.alerts:
            print(f"- {alert}")
        print()


def export_holdings_csv(holdings: Iterable[HoldingAnalysis], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "ticker": h.ticker,
            "value": h.value,
            "weight": h.weight,
            "return_1m": h.returns.get("1m"),
            "return_1y": h.returns.get("1y"),
            "volatility_1y": h.volatility_1y,
            "max_drawdown_1y": h.max_drawdown_1y,
            "beta": h.beta,
            "recommendation": h.recommendation_label,
            "reasons": " | ".join(h.recommendation_reasons),
        }
        for h in holdings
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def parse_portfolio_csv(path: Path) -> PortfolioInput:
    df = pd.read_csv(path)
    holdings: List[HoldingInput] = []
    for _, row in df.iterrows():
        asset_type = row.get("asset_type")
        if pd.isna(asset_type):
            asset_type = None
        notes = row.get("notes")
        if pd.isna(notes):
            notes = None
        holdings.append(
            HoldingInput(
                ticker=str(row["ticker"]),
                quantity=float(row["quantity"]),
                avg_cost=float(row["avg_cost"]) if not pd.isna(row.get("avg_cost")) else None,
                asset_type=asset_type,
                notes=notes,
            )
        )
    if not holdings:
        raise ValueError("Portfolio file is empty.")
    return PortfolioInput(holdings=holdings)


def parse_interactive() -> PortfolioInput:
    print("Enter holdings (blank ticker to finish):")
    holdings: List[HoldingInput] = []
    while True:
        ticker = input("Ticker: ").strip()
        if not ticker:
            break
        quantity = float(input("Quantity: ").strip())
        avg_cost_str = input("Avg cost (optional): ").strip()
        avg_cost = float(avg_cost_str) if avg_cost_str else None
        holdings.append(HoldingInput(ticker=ticker, quantity=quantity, avg_cost=avg_cost))
    if not holdings:
        raise ValueError("No holdings entered.")
    return PortfolioInput(holdings=holdings)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Single-file portfolio analyzer (Yahoo Finance).")
    parser.add_argument("--portfolio", type=Path, help="Path to portfolio CSV.")
    parser.add_argument("--config", type=Path, help="Optional YAML config.")
    parser.add_argument("--export", type=Path, help="Optional CSV export path.")
    parser.add_argument("--risk-profile", choices=["Conservative", "Balanced", "Aggressive"])
    args = parser.parse_args()

    config = load_app_config(args.config) if args.config else AppConfig()
    if args.risk_profile:
        config.risk_profile = args.risk_profile

    if args.portfolio:
        portfolio_input = parse_portfolio_csv(args.portfolio)
    else:
        portfolio_input = parse_interactive()

    analyzer = PortfolioAnalyzer(config)
    settings = AnalysisSettings(risk_profile=config.risk_profile)
    analysis = analyzer.analyze(portfolio_input, settings)
    display_portfolio(analysis)

    if args.export:
        export_holdings_csv(analysis.holdings, args.export)
        print(f"Holdings metrics exported to {args.export}")


if __name__ == "__main__":
    main()
