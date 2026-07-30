"""Tests for agent/tools.py, with `load_cached_prices` monkeypatched to
return synthetic, hand-built data -- these tools should never hit the
filesystem or network in a unit test, only pandas logic that can be
checked against hand-computed expected values.
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

import finrag.agent.tools as tools_module
from finrag.agent.tools import (
    ToolError,
    compare_tickers,
    get_price_on_date,
    get_return,
    get_volatility,
)


def _prices(dates: list[str], closes: list[float]) -> pd.DataFrame:
    n = len(dates)
    return pd.DataFrame(
        {
            "Ticker": ["AAPL"] * n,
            "Date": pd.to_datetime(dates),
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": [1_000_000] * n,
        }
    )


AAPL_DATES = ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"]  # Mon-Fri, then next Mon
AAPL_CLOSES = [100.0, 102.0, 101.0, 105.0, 110.0]


@pytest.fixture
def fake_prices(monkeypatch):
    frames = {"AAPL": _prices(AAPL_DATES, AAPL_CLOSES)}

    def _fake_load(ticker, settings=None):
        if ticker not in frames:
            raise FileNotFoundError(f"no fake data for {ticker}")
        return frames[ticker]

    monkeypatch.setattr(tools_module, "load_cached_prices", _fake_load)
    return frames


def test_get_price_on_date_exact_match(fake_prices):
    result = get_price_on_date("AAPL", "2024-01-03")

    assert result["actual_trading_date"] == "2024-01-03"
    assert result["close"] == 102.0


def test_get_price_on_date_falls_back_to_prior_trading_day(fake_prices):
    # 2024-01-06 and 01-07 are a weekend with no data; should return 01-05.
    result = get_price_on_date("AAPL", "2024-01-07")

    assert result["actual_trading_date"] == "2024-01-05"
    assert result["requested_date"] == "2024-01-07"


def test_get_price_on_date_rejects_unknown_ticker():
    with pytest.raises(ToolError, match="not in the tracked ticker universe"):
        get_price_on_date("NOT_A_TICKER", "2024-01-03")


def test_get_price_on_date_rejects_bad_date_format(fake_prices):
    with pytest.raises(ToolError, match="Invalid date"):
        get_price_on_date("AAPL", "not-a-date")


def test_get_price_on_date_raises_when_no_data_before_target(fake_prices):
    with pytest.raises(ToolError, match="No price data"):
        get_price_on_date("AAPL", "2023-01-01")


def test_get_return_matches_hand_computed_value(fake_prices):
    result = get_return("AAPL", "2024-01-02", "2024-01-08")

    expected_pct = (110.0 / 100.0 - 1) * 100
    assert result["return_pct"] == pytest.approx(expected_pct, abs=0.01)
    assert result["start_date"] == "2024-01-02"
    assert result["end_date"] == "2024-01-08"


def test_get_return_rejects_start_after_end(fake_prices):
    with pytest.raises(ToolError, match="after end_date"):
        get_return("AAPL", "2024-01-08", "2024-01-02")


def test_get_volatility_matches_formula(fake_prices):
    result = get_volatility("AAPL", "2024-01-02", "2024-01-08")

    daily_returns = pd.Series(AAPL_CLOSES).pct_change().dropna()
    expected = daily_returns.std() * math.sqrt(252) * 100
    assert result["volatility_annualized_pct"] == pytest.approx(expected, abs=0.01)


def test_get_volatility_raises_with_fewer_than_two_trading_days(fake_prices):
    with pytest.raises(ToolError, match="Not enough trading days"):
        get_volatility("AAPL", "2024-01-02", "2024-01-02")


def test_compare_tickers_picks_the_best_performer(monkeypatch):
    frames = {
        "AAPL": _prices(AAPL_DATES, AAPL_CLOSES),
        "MSFT": _prices(AAPL_DATES, [200.0, 200.0, 200.0, 200.0, 202.0]),  # only +1%
    }

    def _fake_load(ticker, settings=None):
        return frames[ticker]

    monkeypatch.setattr(tools_module, "load_cached_prices", _fake_load)

    result = compare_tickers(["AAPL", "MSFT"], "2024-01-02", "2024-01-08")

    assert result["best_performer"] == "AAPL"
    assert len(result["comparisons"]) == 2


def test_compare_tickers_requires_at_least_two_tickers():
    with pytest.raises(ToolError, match="at least two tickers"):
        compare_tickers(["AAPL"], "2024-01-02", "2024-01-08")
