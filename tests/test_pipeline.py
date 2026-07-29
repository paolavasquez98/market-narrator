"""Tests the pipeline's wiring (call order, data flow between stages) with
every external dependency (settings, embedder, DB connection, search, LLM)
replaced by a fake -- not the real behavior of any of those pieces, which
are tested elsewhere (test_hybrid_search.py, test_llm_client.py, etc.).
"""

from datetime import date
from unittest.mock import MagicMock

import finrag.rag.pipeline as pipeline_module
from finrag.knowledge_base.models import SearchResult
from finrag.llm.base import LLMResponse


def _fake_result() -> SearchResult:
    return SearchResult(
        doc_id="AAPL:weekly:2024-01-01",
        ticker="AAPL",
        sector="Technology",
        granularity="weekly",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 1, 5),
        content="AAPL rose 3% this week.",
        score=0.9,
    )


class _FakeConnectionContext:
    def __init__(self, conn):
        self._conn = conn

    def __enter__(self):
        return self._conn

    def __exit__(self, *exc_info):
        return False


def test_answer_question_wires_embed_search_and_llm_together(monkeypatch):
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    monkeypatch.setattr(pipeline_module, "get_embedder", lambda: fake_embedder)

    monkeypatch.setattr(pipeline_module, "get_settings", lambda: "fake-settings")

    fake_conn = MagicMock()
    monkeypatch.setattr(
        pipeline_module, "get_connection", lambda settings: _FakeConnectionContext(fake_conn)
    )

    fake_result = _fake_result()
    fake_hybrid_search = MagicMock(return_value=[fake_result])
    monkeypatch.setattr(pipeline_module, "hybrid_search", fake_hybrid_search)

    fake_llm = MagicMock()
    fake_llm.complete.return_value = LLMResponse(content="AAPL rose 3% this week.", tool_calls=[])
    monkeypatch.setattr(pipeline_module, "get_llm_client", lambda settings: fake_llm)

    result = pipeline_module.answer_question("How did AAPL do this week?", top_k=3)

    # The question was embedded exactly once, with the raw question text.
    fake_embedder.embed.assert_called_once_with(["How did AAPL do this week?"])

    # hybrid_search was called with the connection, the question, the
    # embedding, and top_k -- not some subset or the wrong order.
    fake_hybrid_search.assert_called_once_with(
        fake_conn, "How did AAPL do this week?", [0.1, 0.2, 0.3], top_k=3
    )

    # The LLM saw a system message and a user message (in that order).
    call_args = fake_llm.complete.call_args
    messages = call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "How did AAPL do this week?" in messages[1]["content"]

    assert result.answer == "AAPL rose 3% this week."
    assert result.retrieved == [fake_result]
