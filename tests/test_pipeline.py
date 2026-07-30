"""Tests the pipeline's wiring (call order, data flow between stages) with
every external dependency (settings, LLM, query rewriting, embedder, DB
connection, search, reranker, agent loop) replaced by a fake -- not the
real behavior of any of those pieces, which are tested elsewhere
(test_query_rewriter.py, test_hybrid_search.py, test_reranker.py,
test_orchestrator.py, test_llm_client.py).
"""

from datetime import date
from unittest.mock import MagicMock

import finrag.rag.pipeline as pipeline_module
from finrag.knowledge_base.models import SearchResult
from finrag.retrieval.query_rewriter import QueryIntent


def _fake_result(doc_id: str) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
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


def test_answer_question_wires_every_stage_together(monkeypatch):
    monkeypatch.setattr(pipeline_module, "get_settings", lambda: "fake-settings")

    fake_llm = MagicMock()
    monkeypatch.setattr(pipeline_module, "get_llm_client", lambda settings: fake_llm)

    fake_intent = QueryIntent(rewritten_query="Apple stock performance", tickers=["AAPL"])
    fake_rewrite_query = MagicMock(return_value=fake_intent)
    monkeypatch.setattr(pipeline_module, "rewrite_query", fake_rewrite_query)

    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1, 0.2, 0.3]]
    monkeypatch.setattr(pipeline_module, "get_embedder", lambda: fake_embedder)

    fake_conn = MagicMock()
    monkeypatch.setattr(
        pipeline_module, "get_connection", lambda settings: _FakeConnectionContext(fake_conn)
    )

    candidate = _fake_result("AAPL:weekly:2024-01-01")
    fake_hybrid_search = MagicMock(return_value=[candidate])
    monkeypatch.setattr(pipeline_module, "hybrid_search", fake_hybrid_search)

    reranked = _fake_result("AAPL:monthly:2024-01-01")
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [reranked]
    monkeypatch.setattr(pipeline_module, "get_reranker", lambda: fake_reranker)

    fake_run_agent_loop = MagicMock(return_value="AAPL rose 3% this week.")
    monkeypatch.setattr(pipeline_module, "run_agent_loop", fake_run_agent_loop)

    result = pipeline_module.answer_question("How did AAPL do?", top_k=3, rerank_candidates=15)

    # Query rewriting ran first, with the raw question.
    fake_rewrite_query.assert_called_once_with(fake_llm, "How did AAPL do?")

    # The *rewritten* query was embedded, not the raw question.
    fake_embedder.embed.assert_called_once_with(["Apple stock performance"])

    # hybrid_search got the rewritten query, the embedding, the larger
    # rerank-candidate top_k, and the resolved ticker filter.
    fake_hybrid_search.assert_called_once_with(
        fake_conn, "Apple stock performance", [0.1, 0.2, 0.3], top_k=15, tickers=["AAPL"]
    )

    # The reranker narrowed the hybrid candidates down to the final top_k.
    fake_reranker.rerank.assert_called_once_with("Apple stock performance", [candidate], top_k=3)

    # The agent loop, not a bare llm.complete(), produced the final answer.
    run_agent_call = fake_run_agent_loop.call_args
    llm_arg, messages_arg = run_agent_call.args
    assert llm_arg is fake_llm
    assert messages_arg[0]["role"] == "system"
    assert messages_arg[1]["role"] == "user"
    assert "How did AAPL do?" in messages_arg[1]["content"]

    assert result.answer == "AAPL rose 3% this week."
    assert result.retrieved == [reranked]
    assert result.rewritten_query == "Apple stock performance"
    assert result.resolved_tickers == ["AAPL"]


def test_answer_question_passes_empty_ticker_list_as_none_filter(monkeypatch):
    """When query rewriting finds no ticker, hybrid_search should get
    `tickers=None` (no filter), not an empty list that could be
    mishandled downstream as "match nothing".
    """
    monkeypatch.setattr(pipeline_module, "get_settings", lambda: "fake-settings")
    monkeypatch.setattr(pipeline_module, "get_llm_client", lambda settings: MagicMock())
    monkeypatch.setattr(
        pipeline_module,
        "rewrite_query",
        lambda llm, question: QueryIntent(rewritten_query=question, tickers=[]),
    )
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1]]
    monkeypatch.setattr(pipeline_module, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr(
        pipeline_module, "get_connection", lambda settings: _FakeConnectionContext(MagicMock())
    )
    fake_hybrid_search = MagicMock(return_value=[])
    monkeypatch.setattr(pipeline_module, "hybrid_search", fake_hybrid_search)
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = []
    monkeypatch.setattr(pipeline_module, "get_reranker", lambda: fake_reranker)
    monkeypatch.setattr(pipeline_module, "run_agent_loop", lambda llm, messages: "no data")

    pipeline_module.answer_question("Summarize the market")

    assert fake_hybrid_search.call_args.kwargs["tickers"] is None
