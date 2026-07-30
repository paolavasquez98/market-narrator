"""Deterministic financial calculation tools the LLM can call via function
calling, instead of doing arithmetic over retrieved narrative text.

Why tools instead of relying on retrieval for numbers: narrative documents
(build_documents.py) summarize *fixed* periods -- calendar weeks, months,
years -- computed ahead of time during ingestion. A question asking for an
arbitrary range ("AAPL's return from March 3 to June 17, 2022") will
rarely line up with any single retrieved chunk's exact boundaries, so
retrieval can only ever serve pre-computed answers. These tools instead
read the cached daily price data directly (same Parquet cache
`fetch_prices.py` already built) and compute the exact number for
whatever ticker/date range the user actually asked about.

`return_pct` and `annualized_volatility_pct` are imported from
`ingestion/compute_stats.py` rather than redefined here -- the narrative
documents and these tool answers must use the identical formula, or a
user could get two different numbers for what looks like the same
question depending on whether retrieval or a tool happened to answer it.
"""

from __future__ import annotations

from datetime import date

import pandas as pd

from finrag.config.tickers import all_tickers
from finrag.ingestion.compute_stats import annualized_volatility_pct, return_pct
from finrag.ingestion.fetch_prices import load_cached_prices


class ToolError(Exception):
    """Raised when a tool cannot fulfill a request (unknown ticker, bad
    date, no data in range). The agent orchestrator catches this and
    reports the message back to the LLM as the tool's result, instead of
    letting it crash the whole pipeline -- the model can then explain the
    limitation to the user or try different arguments.
    """


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except (ValueError, TypeError) as exc:
        raise ToolError(f"Invalid date {value!r}, expected YYYY-MM-DD") from exc


def _validate_ticker(ticker: str) -> str:
    ticker = str(ticker).upper()
    if ticker not in all_tickers():
        raise ToolError(f"{ticker!r} is not in the tracked ticker universe")
    return ticker


def _load_range(ticker: str, start: date, end: date) -> pd.DataFrame:
    if start > end:
        raise ToolError(f"start_date {start} is after end_date {end}")
    df = load_cached_prices(ticker)
    mask = (df["Date"].dt.date >= start) & (df["Date"].dt.date <= end)
    subset = df.loc[mask].sort_values("Date")
    if subset.empty:
        raise ToolError(f"No price data for {ticker} between {start} and {end}")
    return subset


def get_price_on_date(ticker: str, date_str: str) -> dict:
    """Exact OHLCV for one ticker on one date. If the market was closed
    that day (weekend/holiday), returns the most recent prior trading day
    instead of failing -- the calling LLM is told which date it actually
    got back via `actual_trading_date`.
    """
    ticker = _validate_ticker(ticker)
    target = _parse_date(date_str)
    df = load_cached_prices(ticker)
    on_or_before = df[df["Date"].dt.date <= target].sort_values("Date")
    if on_or_before.empty:
        raise ToolError(f"No price data for {ticker} on or before {target}")

    row = on_or_before.iloc[-1]
    return {
        "ticker": ticker,
        "requested_date": date_str,
        "actual_trading_date": row["Date"].date().isoformat(),
        "open": round(float(row["Open"]), 2),
        "high": round(float(row["High"]), 2),
        "low": round(float(row["Low"]), 2),
        "close": round(float(row["Close"]), 2),
        "volume": int(row["Volume"]),
    }


def get_return(ticker: str, start_date: str, end_date: str) -> dict:
    """Exact Close-to-Close percentage return between two dates. Uses the
    nearest trading days on/after `start_date` and on/before `end_date` if
    the exact dates fall on a weekend/holiday.
    """
    ticker = _validate_ticker(ticker)
    start, end = _parse_date(start_date), _parse_date(end_date)
    subset = _load_range(ticker, start, end)

    start_price = float(subset.iloc[0]["Close"])
    end_price = float(subset.iloc[-1]["Close"])

    return {
        "ticker": ticker,
        "start_date": subset.iloc[0]["Date"].date().isoformat(),
        "end_date": subset.iloc[-1]["Date"].date().isoformat(),
        "start_price": round(start_price, 2),
        "end_price": round(end_price, 2),
        "return_pct": round(return_pct(start_price, end_price), 2),
    }


def get_volatility(ticker: str, start_date: str, end_date: str) -> dict:
    """Exact annualized volatility (stdev of daily returns * sqrt(252))
    over an arbitrary date range -- same formula `compute_stats.py` uses
    for the fixed weekly/monthly/yearly narrative periods, applied here to
    whatever range the user actually asked about.
    """
    ticker = _validate_ticker(ticker)
    start, end = _parse_date(start_date), _parse_date(end_date)
    subset = _load_range(ticker, start, end)

    daily_returns = subset["Close"].pct_change().dropna()
    if len(daily_returns) < 2:
        raise ToolError(
            f"Not enough trading days for {ticker} between {start} and {end} "
            "to compute volatility"
        )

    return {
        "ticker": ticker,
        "start_date": subset.iloc[0]["Date"].date().isoformat(),
        "end_date": subset.iloc[-1]["Date"].date().isoformat(),
        "volatility_annualized_pct": round(annualized_volatility_pct(daily_returns), 2),
        "trading_days": len(subset),
    }


def compare_tickers(tickers: list[str], start_date: str, end_date: str) -> dict:
    """Side-by-side percentage return for two or more tickers over the same
    date range, plus which one performed best. Reuses `get_return` per
    ticker rather than reimplementing the comparison -- one formula, one
    place it can be wrong.
    """
    if len(tickers) < 2:
        raise ToolError("compare_tickers needs at least two tickers")

    comparisons = [get_return(t, start_date, end_date) for t in tickers]
    best = max(comparisons, key=lambda r: r["return_pct"])

    return {"comparisons": comparisons, "best_performer": best["ticker"]}
