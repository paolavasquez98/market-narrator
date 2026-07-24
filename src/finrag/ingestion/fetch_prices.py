"""Fetch and cache daily OHLCV price history for the fixed ticker universe.

Design notes
------------
Yahoo Finance calls are slow (network round-trips for 24 tickers) and we'll
re-run this pipeline many times while iterating on downstream steps
(narrative generation, embeddings). So raw price history is cached to disk
as Parquet, keyed by ticker, and re-fetching is skipped unless the cache is
missing or `--force` is passed. This keeps the ingestion pipeline idempotent:
running it twice in a row does no network I/O the second time.

We use `yf.download(..., group_by="ticker")` for a single batched call
instead of looping `Ticker(t).history()` per symbol -- one HTTP round trip
(per yfinance's internal chunking) instead of 24, and it sidesteps
`Ticker.history()`'s extra call to the `quoteSummary` endpoint for timezone
lookup, which needs a Yahoo auth "crumb" and is a common source of flakiness.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

from finrag.config.settings import Settings, get_settings
from finrag.config.tickers import all_tickers

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def cache_path(ticker: str, settings: Settings) -> Path:
    return settings.raw_data_dir / f"{ticker}.parquet"


def _is_cached(ticker: str, settings: Settings) -> bool:
    return cache_path(ticker, settings).exists()


def _write_cache(ticker: str, df: pd.DataFrame, settings: Settings) -> None:
    settings.raw_data_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path(ticker, settings))


def _read_cache(ticker: str, settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(cache_path(ticker, settings))


def _clean_single_ticker_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize one ticker's raw yfinance frame to a flat, predictable shape."""
    df = df.dropna(how="all")
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing or df.empty:
        raise ValueError(f"{ticker}: no usable price data returned (missing={missing})")

    df = df[REQUIRED_COLUMNS].copy()
    df.index.name = "Date"
    df = df.reset_index()
    df.insert(0, "Ticker", ticker)
    return df


def fetch_universe(
    settings: Settings | None = None,
    tickers: list[str] | None = None,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch (or load from cache) daily OHLCV history for every requested ticker.

    Returns a dict mapping ticker -> DataFrame with columns
    [Ticker, Date, Open, High, Low, Close, Volume]. Tickers that fail to
    download are logged and skipped rather than aborting the whole run.
    """
    settings = settings or get_settings()
    tickers = tickers or all_tickers()

    to_fetch = tickers if force else [t for t in tickers if not _is_cached(t, settings)]

    if to_fetch:
        logger.info("Downloading %d ticker(s) from Yahoo Finance: %s", len(to_fetch), to_fetch)
        raw = yf.download(
            tickers=to_fetch,
            start=str(settings.ingestion_start_date),
            end=str(settings.ingestion_end_date),
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=False,
        )

        for ticker in to_fetch:
            try:
                ticker_df = raw[ticker] if len(to_fetch) > 1 else raw
                cleaned = _clean_single_ticker_frame(ticker_df, ticker)
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping %s: %s", ticker, exc)
                continue
            _write_cache(ticker, cleaned, settings)
            logger.info("Cached %s: %d rows", ticker, len(cleaned))
    else:
        logger.info("All %d ticker(s) already cached, nothing to download", len(tickers))

    return {t: _read_cache(t, settings) for t in tickers if _is_cached(t, settings)}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if a cached file exists"
    )
    parser.add_argument(
        "--tickers", type=str, default=None, help="Comma-separated ticker override (default: full universe)"
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    tickers = args.tickers.split(",") if args.tickers else None
    results = fetch_universe(tickers=tickers, force=args.force)
    logger.info("Done. %d/%d tickers fetched successfully.", len(results), len(tickers or all_tickers()))
    failed = set(tickers or all_tickers()) - set(results)
    if failed:
        logger.warning("Failed tickers: %s", sorted(failed))


if __name__ == "__main__":
    main()
