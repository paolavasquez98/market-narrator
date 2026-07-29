"""Integration tests for vector_search and keyword_search: real queries
against a real Postgres+pgvector instance. Skipped locally without
`docker compose up -d`; always run in CI (see tests/conftest.py and
.github/workflows/ci.yml).
"""

from __future__ import annotations

from datetime import date

import pytest

from finrag.ingestion.build_documents import DocumentRecord
from finrag.knowledge_base.keyword_store import keyword_search
from finrag.knowledge_base.vector_store import get_connection, upsert_documents

DOC_ID_A = "TESTSEARCH:weekly:2024-01-01"
DOC_ID_B = "TESTSEARCH:weekly:2024-01-08"

# 384-dim, matching the schema's VECTOR(384) column. Orthogonal one-hot
# vectors give a deterministic, easy-to-reason-about cosine similarity:
# identical direction -> similarity 1.0, orthogonal -> similarity 0.0.
_DIM = 384


def _one_hot(index: int) -> list[float]:
    vec = [0.0] * _DIM
    vec[index] = 1.0
    return vec


def _record(doc_id: str, ticker: str, content: str) -> DocumentRecord:
    return DocumentRecord(
        doc_id=doc_id,
        ticker=ticker,
        sector="Technology",
        granularity="weekly",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 1, 5),
        content=content,
    )


@pytest.fixture
def two_test_docs(settings, skip_if_db_unreachable):
    """Insert two distinguishable documents (different tickers, orthogonal
    embeddings, distinct keywords), yield, then clean up regardless of
    test outcome.
    """
    records = [
        _record(DOC_ID_A, "TESTSEARCHA", "Zzyzxquant Corp narrative for the search test."),
        _record(DOC_ID_B, "TESTSEARCHB", "A different sentence with no special keyword."),
    ]
    embeddings = [_one_hot(0), _one_hot(1)]

    with get_connection(settings) as conn:
        upsert_documents(conn, records, embeddings)

    yield

    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE doc_id = ANY(%s)", ([DOC_ID_A, DOC_ID_B],))
        conn.commit()


def test_vector_search_ranks_the_matching_direction_first(settings, two_test_docs):
    from finrag.knowledge_base.vector_store import vector_search

    with get_connection(settings) as conn:
        results = vector_search(conn, _one_hot(0), top_k=5, tickers=["TESTSEARCHA", "TESTSEARCHB"])

    assert results[0].doc_id == DOC_ID_A
    assert results[0].score == pytest.approx(1.0, abs=1e-6)


def test_vector_search_respects_ticker_filter(settings, two_test_docs):
    from finrag.knowledge_base.vector_store import vector_search

    with get_connection(settings) as conn:
        results = vector_search(conn, _one_hot(0), top_k=50, tickers=["TESTSEARCHB"])

    # Filtering to TESTSEARCHB should exclude doc A even though its
    # embedding is the closer match -- the WHERE clause runs before ranking.
    assert all(r.ticker == "TESTSEARCHB" for r in results)
    assert DOC_ID_A not in {r.doc_id for r in results}


def test_keyword_search_finds_the_distinctive_term(settings, two_test_docs):
    with get_connection(settings) as conn:
        results = keyword_search(conn, "Zzyzxquant", top_k=5)

    assert any(r.doc_id == DOC_ID_A for r in results)
    assert all(r.doc_id != DOC_ID_B for r in results)
