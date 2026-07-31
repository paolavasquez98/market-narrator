# Ticker universe

Market Narrator tracks a fixed set of 26 liquid, well-known tickers across 8 sectors, rather than the full market. This keeps ingestion fast and predictable, keeps the knowledge base small enough to embed/rerank locally, and — most importantly — means every reviewer who clones this repo and runs the ingestion pipeline gets an identical corpus.

The single source of truth is [`src/finrag/config/tickers.py`](../src/finrag/config/tickers.py); this file documents it for humans.

| Sector | Tickers |
|---|---|
| ETF | SPY, QQQ |
| Technology | AAPL, MSFT, NVDA, AMD, GOOGL, AMZN, META, TSLA, NFLX |
| Finance | JPM, BAC, V, MA |
| Healthcare | LLY, JNJ, PFE |
| Consumer | KO, PEP, COST |
| Energy | XOM, CVX |
| Industrial | CAT |
| Semiconductors | AVGO, TSM |

Data window: **2015-01-01 to 2026-07-24** (fixed in `Settings.ingestion_start_date` / `ingestion_end_date`), daily OHLCV via `yfinance`.

Growing the universe later just means adding entries to `TICKER_UNIVERSE` in `tickers.py` and re-running ingestion — every downstream module (document generation, retrieval, agent tools) reads from that dict.
