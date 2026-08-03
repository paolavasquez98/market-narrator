"""Tests for the FastAPI service (src/finrag/api/), using FastAPI's
TestClient with every dependency (the RAG pipeline, the DB connection,
the logger) monkeypatched. This deliberately never touches a real
database or Groq -- it verifies the API's own logic (request/response
shaping, status codes, the "logging failure shouldn't break a successful
answer" contract) independent of whether the pipeline or DB actually work,
which are already covered by test_pipeline.py, test_logger.py, and the
DB-integration tests elsewhere.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date

from fastapi.testclient import TestClient

import finrag.api.routes as routes_module
from finrag.api.main import app
from finrag.knowledge_base.models import SearchResult
from finrag.rag.pipeline import RagAnswer

client = TestClient(app)


def test_root_redirects_to_docs():
    response = client.get("/", follow_redirects=False)
    assert response.status_code in (302, 307)
    assert response.headers["location"] == "/docs"


def _fake_rag_answer(question: str, top_k: int = 5) -> RagAnswer:
    return RagAnswer(
        question=question,
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


@contextmanager
def _fake_connection_ok(settings=None):
    yield object()  # never actually used by the monkeypatched log_query/record_feedback


@contextmanager
def _fake_connection_raises(settings=None):
    raise RuntimeError("simulated DB outage")
    yield  # pragma: no cover -- unreachable, keeps this a generator function


# --- POST /ask ---


def test_ask_returns_answer_and_a_query_id(monkeypatch):
    monkeypatch.setattr(routes_module, "answer_question", _fake_rag_answer)
    monkeypatch.setattr(routes_module, "get_connection", _fake_connection_ok)
    monkeypatch.setattr(routes_module, "log_query", lambda *a, **k: 42)

    response = client.post("/ask", json={"question": "How did AAPL do in 2022?"})

    assert response.status_code == 200
    body = response.json()
    assert body["query_id"] == 42
    assert body["answer"] == "AAPL fell about 26% in 2022."
    assert body["resolved_tickers"] == ["AAPL"]
    assert body["retrieved"][0]["doc_id"] == "AAPL:yearly:2022-01-03"
    assert body["retrieved"][0]["period_start"] == "2022-01-03"
    assert isinstance(body["latency_ms"], int)


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422


def test_ask_returns_502_when_pipeline_fails(monkeypatch):
    def _raise(question, top_k=5):
        raise RuntimeError("Groq is down")

    monkeypatch.setattr(routes_module, "answer_question", _raise)

    response = client.post("/ask", json={"question": "How did AAPL do?"})

    assert response.status_code == 502


def test_ask_still_returns_the_answer_when_logging_fails(monkeypatch):
    """A DB outage during logging shouldn't turn a successful answer into
    an error response -- see routes.py's comment on why this is a
    deliberate try/except around the logging call specifically.
    """
    monkeypatch.setattr(routes_module, "answer_question", _fake_rag_answer)
    monkeypatch.setattr(routes_module, "get_connection", _fake_connection_raises)

    response = client.post("/ask", json={"question": "How did AAPL do in 2022?"})

    assert response.status_code == 200
    assert response.json()["query_id"] == -1
    assert response.json()["answer"] == "AAPL fell about 26% in 2022."


# --- POST /feedback/{query_id} ---


def test_feedback_records_successfully(monkeypatch):
    monkeypatch.setattr(routes_module, "get_connection", _fake_connection_ok)
    monkeypatch.setattr(routes_module, "record_feedback", lambda conn, query_id, feedback: True)

    response = client.post("/feedback/42", json={"feedback": 1})

    assert response.status_code == 200
    assert response.json() == {"query_id": 42, "recorded": True}


def test_feedback_returns_404_for_unknown_query_id(monkeypatch):
    monkeypatch.setattr(routes_module, "get_connection", _fake_connection_ok)
    monkeypatch.setattr(routes_module, "record_feedback", lambda conn, query_id, feedback: False)

    response = client.post("/feedback/9999", json={"feedback": -1})

    assert response.status_code == 404


def test_feedback_returns_404_for_the_unlogged_sentinel_id(monkeypatch):
    # query_id == -1 is what /ask returns when logging itself failed --
    # there's no row to attach feedback to, and this should be rejected
    # before ever touching the database.
    called = False

    def _should_not_be_called(*a, **k):
        nonlocal called
        called = True

    monkeypatch.setattr(routes_module, "get_connection", _should_not_be_called)

    response = client.post("/feedback/-1", json={"feedback": 1})

    assert response.status_code == 404
    assert called is False


def test_feedback_rejects_invalid_feedback_value():
    response = client.post("/feedback/1", json={"feedback": 0})
    assert response.status_code == 422


# --- GET /tickers ---


def test_tickers_returns_the_full_universe():
    response = client.get("/tickers")

    assert response.status_code == 200
    groups = {g["sector"]: g["tickers"] for g in response.json()["tickers"]}
    assert "AAPL" in groups["Technology"]
    assert "SPY" in groups["ETF"]


# --- GET /health ---


def test_health_ok_when_database_reachable(monkeypatch):
    class _FakeCursor:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def execute(self, *a, **k):
            pass

        def fetchone(self):
            return (1,)

    class _FakeConn:
        def cursor(self):
            return _FakeCursor()

    @contextmanager
    def _fake_connection(settings=None):
        yield _FakeConn()

    monkeypatch.setattr(routes_module, "get_connection", _fake_connection)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "database": "ok"}


def test_health_degraded_when_database_unreachable(monkeypatch):
    monkeypatch.setattr(routes_module, "get_connection", _fake_connection_raises)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "degraded", "database": "unreachable"}
