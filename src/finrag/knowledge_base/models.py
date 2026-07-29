"""Shared value types for anything that searches the `documents` table.

Both `vector_store.vector_search()` and `keyword_store.keyword_search()`
return the same `SearchResult` shape even though the underlying SQL and
the meaning of `score` are completely different (cosine similarity vs.
text-rank). That's deliberate: `retrieval/hybrid_search.py` fuses two
lists of results together, and fusion only works cleanly if both lists
speak the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

# Both vector_store.vector_search() and keyword_store.keyword_search() SELECT
# these columns (plus their own method-specific `score` expression) so that
# `_row_to_result` below can map either one's rows the same way.
DOCUMENT_COLUMNS = "doc_id, ticker, sector, granularity, period_start, period_end, content"


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    ticker: str
    sector: str
    granularity: str
    period_start: date
    period_end: date
    content: str
    score: float
    """Meaning of `score` depends on where this result came from:
    cosine similarity (vector search, higher is more similar), ts_rank
    (keyword search, higher is a stronger text match), or an RRF score
    (after hybrid fusion, higher is ranked better by both signals
    combined). Never compare scores across methods directly.
    """


def row_to_search_result(row: tuple[Any, ...]) -> SearchResult:
    """Map a SQL row shaped like `DOCUMENT_COLUMNS, score` to a SearchResult.
    Shared by vector_store and keyword_store so the column order is only
    encoded in one place.
    """
    return SearchResult(
        doc_id=row[0],
        ticker=row[1],
        sector=row[2],
        granularity=row[3],
        period_start=row[4],
        period_end=row[5],
        content=row[6],
        score=float(row[7]),
    )
