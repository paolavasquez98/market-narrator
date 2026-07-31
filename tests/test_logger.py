"""Integration tests for monitoring/logger.py, run against a real
Postgres instance -- same skip-if-unreachable pattern as
test_vector_store.py. `query_logs` rows are cleaned up after each test so
repeated local runs don't accumulate synthetic rows in a dev database.
"""

from __future__ import annotations

from datetime import date

import pytest

from finrag.knowledge_base.models import SearchResult
from finrag.knowledge_base.vector_store import get_connection
from finrag.monitoring.logger import log_query, record_feedback
from finrag.rag.pipeline import RagAnswer


def _fake_answer() -> RagAnswer:
    return RagAnswer(
        question="How did AAPL do in 2022?",
        answer="AAPL fell about 26% in 2022.",
        retrieved=[
            SearchResult(
                doc_id="AAPL:yearly:2022-01-03",
                ticker="AAPL",
                sector="Technology",
                granularity="yearly",
                period_start=date(2022, 1, 3),
                period_end=date(2022, 12, 30),
                content="AAPL fell 26% in 2022.",
                score=0.9,
            )
        ],
        rewritten_query="AAPL stock performance 2022",
        resolved_tickers=["AAPL"],
    )


@pytest.fixture
def logged_query_id(settings, skip_if_db_unreachable):
    with get_connection(settings) as conn:
        query_id = log_query(
            conn,
            _fake_answer(),
            retrieval_method="hybrid_rerank_rewrite",
            model="llama-3.3-70b-versatile",
            tool_calls=[{"name": "get_return", "arguments": {"ticker": "AAPL"}}],
            latency_ms=1234,
        )
    yield query_id
    with get_connection(settings) as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM query_logs WHERE id = %s", (query_id,))
        conn.commit()


def test_log_query_returns_an_id_and_stores_the_row(settings, logged_query_id):
    with get_connection(settings) as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT question, rewritten_query, extracted_tickers, retrieval_method, "
            "retrieved_doc_ids, model, answer, latency_ms, feedback "
            "FROM query_logs WHERE id = %s",
            (logged_query_id,),
        )
        row = cur.fetchone()

    assert row is not None
    question, rewritten_query, tickers, method, doc_ids, model, answer, latency_ms, feedback = row
    assert question == "How did AAPL do in 2022?"
    assert rewritten_query == "AAPL stock performance 2022"
    assert tickers == ["AAPL"]
    assert method == "hybrid_rerank_rewrite"
    assert doc_ids == ["AAPL:yearly:2022-01-03"]
    assert model == "llama-3.3-70b-versatile"
    assert answer == "AAPL fell about 26% in 2022."
    assert latency_ms == 1234
    assert feedback is None  # no feedback recorded yet


def test_record_feedback_updates_existing_row(settings, logged_query_id):
    with get_connection(settings) as conn:
        updated = record_feedback(conn, logged_query_id, 1)

    assert updated is True
    with get_connection(settings) as conn, conn.cursor() as cur:
        cur.execute("SELECT feedback FROM query_logs WHERE id = %s", (logged_query_id,))
        assert cur.fetchone()[0] == 1


def test_record_feedback_returns_false_for_unknown_id(settings, skip_if_db_unreachable):
    with get_connection(settings) as conn:
        updated = record_feedback(conn, query_id=-1, feedback=1)

    assert updated is False


def test_record_feedback_rejects_invalid_values(settings, skip_if_db_unreachable):
    with get_connection(settings) as conn, pytest.raises(ValueError, match="feedback must be"):
        record_feedback(conn, query_id=1, feedback=0)
