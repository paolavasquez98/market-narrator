"""Tests for Reranker, using a stubbed fastembed cross-encoder so no
network call or model download is required. This tests that our wrapper
correctly re-scores and re-sorts candidates -- not the cross-encoder
model's actual relevance judgments.
"""

from __future__ import annotations

import sys
import types
from datetime import date

import pytest

from finrag.knowledge_base.models import SearchResult


def _result(doc_id: str, content: str) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        ticker="AAPL",
        sector="Technology",
        granularity="weekly",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 1, 5),
        content=content,
        score=0.5,  # the RRF score from hybrid search, pre-rerank
    )


@pytest.fixture
def stub_fastembed_rerank(monkeypatch):
    """Install a fake `fastembed.rerank.cross_encoder` module before
    Reranker imports it. The fake scores documents by how many times the
    word "relevant" appears -- deterministic and easy to assert on.
    """
    fake_module = types.ModuleType("fastembed.rerank.cross_encoder")

    class FakeTextCrossEncoder:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def rerank(self, query: str, documents: list[str]):
            for doc in documents:
                yield float(doc.count("relevant"))

    fake_module.TextCrossEncoder = FakeTextCrossEncoder
    monkeypatch.setitem(sys.modules, "fastembed.rerank.cross_encoder", fake_module)
    return fake_module


def test_rerank_reorders_by_cross_encoder_score(stub_fastembed_rerank):
    from finrag.retrieval.reranker import Reranker

    reranker = Reranker()
    results = [
        _result("A", "totally unrelated text"),
        _result("B", "very relevant relevant relevant"),
        _result("C", "somewhat relevant"),
    ]

    reranked = reranker.rerank("query", results, top_k=3)

    assert [r.doc_id for r in reranked] == ["B", "C", "A"]
    assert reranked[0].score == 3.0


def test_rerank_respects_top_k(stub_fastembed_rerank):
    from finrag.retrieval.reranker import Reranker

    reranker = Reranker()
    results = [_result("A", "relevant"), _result("B", "relevant relevant")]

    reranked = reranker.rerank("query", results, top_k=1)

    assert len(reranked) == 1
    assert reranked[0].doc_id == "B"


def test_rerank_handles_empty_input(stub_fastembed_rerank):
    from finrag.retrieval.reranker import Reranker

    reranker = Reranker()
    assert reranker.rerank("query", [], top_k=5) == []
