"""Fixed ticker universe for Market Narrator.

The universe is intentionally small and fixed (not "all of S&P 500") so that:
  * ingestion runs quickly and predictably (bounded API calls to yfinance),
  * the knowledge base size stays manageable for local embedding/reranking,
  * every peer reviewer who clones the repo gets the exact same corpus.

If you need to grow the universe later, add tickers here — every downstream
module (ingestion, retrieval, tools) reads from this single source of truth.
"""

from __future__ import annotations

# Sector -> list of ticker symbols.
TICKER_UNIVERSE: dict[str, list[str]] = {
    "ETF": ["SPY", "QQQ"],
    "Technology": [
        "AAPL",
        "MSFT",
        "NVDA",
        "AMD",
        "GOOGL",
        "AMZN",
        "META",
        "TSLA",
        "NFLX",
    ],
    "Finance": ["JPM", "BAC", "V", "MA"],
    "Healthcare": ["LLY", "JNJ", "PFE"],
    "Consumer": ["KO", "PEP", "COST"],
    "Energy": ["XOM", "CVX"],
    "Industrial": ["CAT"],
    "Semiconductors": ["AVGO", "TSM"],
}


def all_tickers() -> list[str]:
    """Flat, sorted list of every ticker in the universe."""
    tickers = {t for group in TICKER_UNIVERSE.values() for t in group}
    return sorted(tickers)


def sector_for(ticker: str) -> str | None:
    """Look up which sector a ticker belongs to, or None if not in the universe."""
    for sector, tickers in TICKER_UNIVERSE.items():
        if ticker in tickers:
            return sector
    return None
