"""Turn PeriodStats into the narrative text documents that make up the
knowledge base.

This is the crux of the whole project's retrieval design: instead of
asking an LLM to write summaries (which could hallucinate numbers) or
embedding raw OHLCV rows (numbers don't carry semantic meaning that vector
search can exploit), we deterministically template PeriodStats into
readable English. The numbers are always exactly what the pandas
computation produced; embeddings only have to capture the *meaning* of a
sentence like "AAPL fell sharply in March", not memorize digits.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

from finrag.config.settings import Settings, get_settings
from finrag.config.tickers import all_tickers, sector_for
from finrag.ingestion.compute_stats import Granularity, PeriodStats, iter_period_stats
from finrag.ingestion.fetch_prices import load_cached_prices

GRANULARITIES: tuple[Granularity, ...] = ("weekly", "monthly", "yearly")

logger = logging.getLogger(__name__)


@dataclass
class DocumentRecord:
    """Maps 1:1 to a row in the `documents` table (schema.sql), minus the
    embedding, which is added later by the ingestion orchestrator.
    """

    doc_id: str
    ticker: str
    sector: str
    granularity: Granularity
    period_start: date
    period_end: date
    content: str


def _format_notable_moves(stats: PeriodStats) -> str:
    if not stats.notable_moves:
        return "No single-day moves of 3% or more."
    parts = [f"{m.move_date.isoformat()} ({m.pct_change:+.1f}%)" for m in stats.notable_moves]
    return "Notable single-day moves: " + ", ".join(parts) + "."


def render_narrative(stats: PeriodStats, sector: str) -> str:
    """Deterministically template a PeriodStats into English. No LLM involved --
    every number here traces directly back to the pandas computation.
    """
    return (
        f"{stats.ticker} ({sector}) {stats.granularity} summary, "
        f"{stats.period_start.isoformat()} to {stats.period_end.isoformat()}. "
        f"Price moved from ${stats.start_price:.2f} to ${stats.end_price:.2f}, "
        f"a {stats.return_pct:+.2f}% change. "
        f"Period range: high ${stats.high:.2f}, low ${stats.low:.2f}. "
        f"Maximum drawdown from a running peak within the period was "
        f"{stats.max_drawdown_pct:.2f}%. "
        f"Annualized volatility (from daily returns) was "
        f"{stats.volatility_annualized_pct:.2f}%. "
        f"Average daily volume: {stats.avg_volume:,.0f} shares. "
        f"{_format_notable_moves(stats)}"
    )


def build_documents_for_ticker(
    ticker: str, settings: Settings | None = None
) -> list[DocumentRecord]:
    settings = settings or get_settings()
    sector = sector_for(ticker)
    if sector is None:
        raise ValueError(f"{ticker} is not in the configured ticker universe")

    df = load_cached_prices(ticker, settings)
    records: list[DocumentRecord] = []

    for granularity in GRANULARITIES:
        for stats in iter_period_stats(df, ticker, granularity):
            doc_id = f"{ticker}:{granularity}:{stats.period_start.isoformat()}"
            records.append(
                DocumentRecord(
                    doc_id=doc_id,
                    ticker=ticker,
                    sector=sector,
                    granularity=granularity,
                    period_start=stats.period_start,
                    period_end=stats.period_end,
                    content=render_narrative(stats, sector),
                )
            )
    return records


def build_all_documents(settings: Settings | None = None) -> list[DocumentRecord]:
    """Build narrative documents for every ticker in the universe. Tickers
    without cached price data are skipped with a log message rather than
    aborting the whole build (mirrors fetch_universe's per-ticker resilience).
    """
    settings = settings or get_settings()

    all_records: list[DocumentRecord] = []
    for ticker in all_tickers():
        try:
            all_records.extend(build_documents_for_ticker(ticker, settings))
        except FileNotFoundError as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
    return all_records
