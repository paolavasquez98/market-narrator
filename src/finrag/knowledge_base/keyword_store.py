"""Keyword (lexical) search over the `documents` table, using Postgres's
built-in full-text search -- no separate search engine (Elasticsearch,
etc.) needed for a corpus this size.

Why we still need this alongside vector search: embeddings are good at
"similar meaning", but weak at exact tokens that matter a lot in finance
-- a ticker symbol like "NVDA", an exact date, a specific number. Keyword
search catches those directly; hybrid_search.py (this module's sibling in
retrieval/) combines both so neither weakness dominates.
"""

from __future__ import annotations

from collections.abc import Sequence

import psycopg

from finrag.knowledge_base.models import DOCUMENT_COLUMNS, SearchResult, row_to_search_result


def keyword_search(
    conn: psycopg.Connection,
    query_text: str,
    top_k: int = 10,
    tickers: Sequence[str] | None = None,
) -> list[SearchResult]:
    """Full-text search using the generated `content_tsv` column (schema.sql)
    and its GIN index. `plainto_tsquery` tokenizes and stems `query_text`
    the same way `content_tsv` was built (english config), so "volatility"
    matches "volatile" and so on. `ts_rank` scores how well a document
    matches the query terms (higher = stronger match).
    """
    where_ticker = "AND ticker = ANY(%(tickers)s)" if tickers else ""
    query = f"""
        SELECT {DOCUMENT_COLUMNS},
               ts_rank(content_tsv, plainto_tsquery('english', %(query_text)s)) AS score
        FROM documents
        WHERE content_tsv @@ plainto_tsquery('english', %(query_text)s)
        {where_ticker}
        ORDER BY score DESC
        LIMIT %(top_k)s
    """
    params = {"query_text": query_text, "top_k": top_k, "tickers": tickers}

    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [row_to_search_result(r) for r in rows]
