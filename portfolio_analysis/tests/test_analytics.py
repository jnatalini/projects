import pandas as pd

from main import annualized_volatility, compute_returns, max_drawdown


def build_history(prices):
    return pd.DataFrame(
        {"Close": prices},
        index=pd.date_range("2024-01-01", periods=len(prices), freq="D"),
    )


def test_compute_returns_produces_values():
    history = build_history([100, 101, 102, 110, 120, 130])
    returns = compute_returns(history)
    assert "1m" in returns


def test_volatility_and_drawdown():
    history = build_history([100, 90, 95, 85, 88, 110, 120])
    vol = annualized_volatility(history)
    drawdown = max_drawdown(history)
    assert vol is not None
    assert drawdown < 0
