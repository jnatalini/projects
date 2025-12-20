from main import AppConfig, HoldingAnalysis, build_recommendation


def test_recommendation_generates_reasons():
    analysis = HoldingAnalysis(
        ticker="XYZ",
        last_price=100,
        prev_close=95,
        value=1000,
        weight=0.3,
        change_1d_pct=0.02,
        returns={"3m": 0.1, "1y": 0.2, "1m": 0.05, "1w": 0.02, "1d": 0.01},
        volatility_1y=0.2,
        max_drawdown_1y=-0.1,
        beta=1.1,
        dividend_yield=0.01,
        expense_ratio=None,
        avg_cost=90,
        cost_basis=900,
        unrealized_pl=100,
        unrealized_pl_pct=0.111,
    )
    config = AppConfig()
    result = build_recommendation(analysis, config)
    assert result.recommendation_label in {"Add", "Hold", "Watch", "Reduce", "Review"}
    assert result.recommendation_reasons
