import pytest

from main import HoldingInput, PortfolioInput


def test_ticker_validation():
    with pytest.raises(ValueError):
        HoldingInput(ticker=" ", quantity=1)


def test_portfolio_input_normalizes_ticker():
    holding = HoldingInput(ticker="aapl", quantity=2)
    portfolio = PortfolioInput(holdings=[holding])
    assert portfolio.holdings[0].ticker == "AAPL"
