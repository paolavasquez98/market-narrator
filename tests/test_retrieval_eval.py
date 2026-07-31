"""Dispatch tests for retrieval_eval._retrieve_doc_ids: verify each variant
name calls exactly the functions it's supposed to, with the arguments it's
supposed to use. This is the kind of copy-paste-prone if/elif chain where
a wiring bug would silently produce misleading evaluation numbers instead
of an obvious crash -- worth guarding against directly, separately from
whether the underlying search functions behave correctly (tested
elsewhere).
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

import pytest

import finrag.eval.retrieval_eval as re_module
from finrag.eval.retrieval_eval import (
    RetrievalEvalResult,
    _retrieve_doc_ids,
    best_variant,
    results_to_dataframe,
)
from finrag.knowledge_base.models import SearchResult
from finrag.retrieval.query_rewriter import QueryIntent


def _result(doc_id: str) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        ticker="AAPL",
        sector="Technology",
        granularity="weekly",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 1, 5),
        content="x",
        score=1.0,
    )


def test_keyword_variant_only_calls_keyword_search(monkeypatch):
    fake_keyword = MagicMock(return_value=[_result("A")])
    fake_vector = MagicMock()
    fake_hybrid = MagicMock()
    monkeypatch.setattr(re_module, "keyword_search", fake_keyword)
    monkeypatch.setattr(re_module, "vector_search", fake_vector)
    monkeypatch.setattr(re_module, "hybrid_search", fake_hybrid)

    doc_ids = _retrieve_doc_ids(
        "keyword", "q", conn=None, llm=None, embedder=MagicMock(), reranker=MagicMock()
    )

    assert doc_ids == ["A"]
    fake_keyword.assert_called_once()
    fake_vector.assert_not_called()
    fake_hybrid.assert_not_called()


def test_vector_variant_embeds_the_question_and_calls_vector_search(monkeypatch):
    fake_vector = MagicMock(return_value=[_result("B")])
    monkeypatch.setattr(re_module, "vector_search", fake_vector)
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1, 0.2]]

    doc_ids = _retrieve_doc_ids(
        "vector", "q", conn=None, llm=None, embedder=fake_embedder, reranker=MagicMock()
    )

    assert doc_ids == ["B"]
    fake_embedder.embed.assert_called_once_with(["q"])
    fake_vector.assert_called_once()


def test_hybrid_variant_does_not_rerank(monkeypatch):
    fake_hybrid = MagicMock(return_value=[_result("C")])
    monkeypatch.setattr(re_module, "hybrid_search", fake_hybrid)
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1]]
    fake_reranker = MagicMock()

    doc_ids = _retrieve_doc_ids(
        "hybrid", "q", conn=None, llm=None, embedder=fake_embedder, reranker=fake_reranker
    )

    assert doc_ids == ["C"]
    fake_reranker.rerank.assert_not_called()


def test_hybrid_rerank_variant_calls_hybrid_then_reranker(monkeypatch):
    fake_hybrid = MagicMock(return_value=[_result("C")])
    monkeypatch.setattr(re_module, "hybrid_search", fake_hybrid)
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1]]
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [_result("D")]

    doc_ids = _retrieve_doc_ids(
        "hybrid_rerank", "q", conn=None, llm=None, embedder=fake_embedder, reranker=fake_reranker
    )

    assert doc_ids == ["D"]
    fake_hybrid.assert_called_once()
    fake_reranker.rerank.assert_called_once()


def test_hybrid_rerank_rewrite_variant_uses_rewritten_query_and_ticker_filter(monkeypatch):
    monkeypatch.setattr(
        re_module,
        "rewrite_query",
        lambda llm, question: QueryIntent(rewritten_query="Apple stock", tickers=["AAPL"]),
    )
    fake_hybrid = MagicMock(return_value=[_result("E")])
    monkeypatch.setattr(re_module, "hybrid_search", fake_hybrid)
    fake_embedder = MagicMock()
    fake_embedder.embed.return_value = [[0.1]]
    fake_reranker = MagicMock()
    fake_reranker.rerank.return_value = [_result("F")]

    doc_ids = _retrieve_doc_ids(
        "hybrid_rerank_rewrite",
        "q",
        conn=None,
        llm=MagicMock(),
        embedder=fake_embedder,
        reranker=fake_reranker,
    )

    assert doc_ids == ["F"]
    fake_embedder.embed.assert_called_once_with(["Apple stock"])
    assert fake_hybrid.call_args.kwargs["tickers"] == ["AAPL"]


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="Unknown retrieval variant"):
        _retrieve_doc_ids("bogus", "q", conn=None, llm=None, embedder=MagicMock(), reranker=MagicMock())


def test_results_to_dataframe_preserves_order():
    results = [
        RetrievalEvalResult(variant="a", hit_rate=0.8, mrr=0.5, n_queries=10),
        RetrievalEvalResult(variant="b", hit_rate=0.9, mrr=0.7, n_queries=10),
    ]
    df = results_to_dataframe(results)
    assert list(df["variant"]) == ["a", "b"]


def test_best_variant_picks_highest_mrr():
    results = [
        RetrievalEvalResult(variant="a", hit_rate=0.9, mrr=0.5, n_queries=10),
        RetrievalEvalResult(variant="b", hit_rate=0.8, mrr=0.7, n_queries=10),
    ]
    assert best_variant(results).variant == "b"


def test_best_variant_breaks_ties_by_hit_rate():
    results = [
        RetrievalEvalResult(variant="a", hit_rate=0.7, mrr=0.5, n_queries=10),
        RetrievalEvalResult(variant="b", hit_rate=0.9, mrr=0.5, n_queries=10),
    ]
    assert best_variant(results).variant == "b"
