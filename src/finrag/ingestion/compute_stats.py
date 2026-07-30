"""Compute price-behavior statistics for a period of daily OHLCV data.

These functions are pure (no I/O, no network): given a DataFrame slice for
one ticker and one period, they return a `PeriodStats` value. That makes
them trivial to unit test with hand-built synthetic data and hand-checked
expected numbers -- this is the one module doing actual financial
arithmetic, so it's the module most worth pinning down precisely.

Downstream, `build_documents.py` turns `PeriodStats` into narrative text;
this module never touches text.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import pandas as pd

TRADING_DAYS_PER_YEAR = 252
NOTABLE_MOVE_THRESHOLD = 0.03  # single-day move of 3%+ is worth calling out
MAX_NOTABLE_MOVES = 3

Granularity = str  # "weekly" | "monthly" | "yearly"

_PERIOD_FREQ = {
    "weekly": "W-FRI",
    "monthly": "M",
    "yearly": "Y",
}


@dataclass
class NotableMove:
    move_date: date
    pct_change: float


@dataclass
class PeriodStats:
    ticker: str
    granularity: Granularity
    period_start: date
    period_end: date
    start_price: float
    end_price: float
    return_pct: float
    high: float
    low: float
    max_drawdown_pct: float
    avg_volume: float
    volatility_annualized_pct: float
    notable_moves: list[NotableMove]


def _daily_returns(close: pd.Series) -> pd.Series:
    return close.pct_change().dropna()


def _max_drawdown_pct(close: pd.Series) -> float:
    running_peak = close.cummax()
    drawdown = (close - running_peak) / running_peak
    return float(drawdown.min() * 100)


def return_pct(start_price: float, end_price: float) -> float:
    """Percentage change from `start_price` to `end_price`. Public (not
    prefixed `_`) because `agent/tools.py`'s `get_return` tool reuses this
    exact formula -- the narrative documents and the agent's deterministic
    tool answers must agree on how "return" is defined, or a user could get
    two different numbers for what looks like the same question.
    """
    return (end_price / start_price - 1) * 100


def annualized_volatility_pct(daily_returns: pd.Series) -> float:
    if len(daily_returns) < 2:
        return 0.0
    return float(daily_returns.std() * math.sqrt(TRADING_DAYS_PER_YEAR) * 100)


def _notable_moves(df: pd.DataFrame, daily_returns: pd.Series) -> list[NotableMove]:
    big_moves = daily_returns[daily_returns.abs() >= NOTABLE_MOVE_THRESHOLD]
    top = big_moves.reindex(big_moves.abs().sort_values(ascending=False).index)
    top = top.iloc[:MAX_NOTABLE_MOVES]
    dates = df.loc[top.index, "Date"]
    return [
        NotableMove(move_date=d.date(), pct_change=float(pct) * 100)
        for d, pct in zip(dates, top)
    ]


def compute_period_stats(df_period: pd.DataFrame, ticker: str, granularity: Granularity) -> PeriodStats:
    """Compute stats for one ticker over one period.

    `df_period` must have columns [Date, Open, High, Low, Close, Volume],
    sorted or unsorted (this function sorts by Date), covering exactly the
    days in that period.
    """
    if df_period.empty:
        raise ValueError(f"{ticker}: cannot compute stats for an empty period")

    df_period = df_period.sort_values("Date").reset_index(drop=True)
    close = df_period["Close"]
    daily_returns = _daily_returns(close)

    start_price = float(close.iloc[0])
    end_price = float(close.iloc[-1])

    return PeriodStats(
        ticker=ticker,
        granularity=granularity,
        period_start=df_period["Date"].iloc[0].date(),
        period_end=df_period["Date"].iloc[-1].date(),
        start_price=start_price,
        end_price=end_price,
        return_pct=return_pct(start_price, end_price),
        high=float(df_period["High"].max()),
        low=float(df_period["Low"].min()),
        max_drawdown_pct=_max_drawdown_pct(close),
        avg_volume=float(df_period["Volume"].mean()),
        volatility_annualized_pct=annualized_volatility_pct(daily_returns),
        notable_moves=_notable_moves(df_period, daily_returns),
    )


def iter_period_stats(df: pd.DataFrame, ticker: str, granularity: Granularity):
    """Split a full price history into periods (by calendar week/month/year)
    and yield PeriodStats for each one that has at least 2 trading days
    (fewer than that and return/volatility aren't meaningful).
    """
    freq = _PERIOD_FREQ[granularity]
    period_keys = df["Date"].dt.to_period(freq)
    for _, group in df.groupby(period_keys):
        if len(group) < 2:
            continue
        yield compute_period_stats(group, ticker, granularity)
