from datetime import date

import pytest

from finrag.knowledge_base.models import SearchResult
from finrag.retrieval.hybrid_search import RRF_K, reciprocal_rank_fusion


def _result(doc_id: str) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        ticker="AAPL",
        sector="Technology",
        granularity="weekly",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 1, 5),
        content=f"content for {doc_id}",
        score=1.0,
    )


def test_docs_found_by_both_lists_rank_above_docs_found_by_one():
    vector_ranked = [_result("A"), _result("B"), _result("C")]
    keyword_ranked = [_result("B"), _result("A"), _result("D")]

    fused = reciprocal_rank_fusion(vector_ranked, keyword_ranked)
    fused_ids = [r.doc_id for r in fused]

    # A and B were each found by both lists (at ranks 1+2, in some order) so
    # both outrank C and D, which were only found by one list each.
    assert set(fused_ids[:2]) == {"A", "B"}
    assert set(fused_ids[2:]) == {"C", "D"}


def test_fused_score_matches_rrf_formula():
    fused = reciprocal_rank_fusion([_result("A")], k=RRF_K)
    assert fused[0].score == pytest.approx(1.0 / (RRF_K + 1))


def test_fusion_of_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([], []) == []


def test_fusion_deduplicates_by_doc_id():
    # Same doc_id appearing in both lists should produce exactly one fused
    # result, not two.
    fused = reciprocal_rank_fusion([_result("A"), _result("B")], [_result("A")])
    assert len(fused) == 2
    assert {r.doc_id for r in fused} == {"A", "B"}


def test_hybrid_search_combines_both_methods_and_respects_top_k(monkeypatch):
    import finrag.retrieval.hybrid_search as hs

    monkeypatch.setattr(
        hs,
        "vector_search",
        lambda conn, query_embedding, top_k, tickers=None: [_result("A"), _result("B")],
    )
    monkeypatch.setattr(
        hs,
        "keyword_search",
        lambda conn, query_text, top_k, tickers=None: [_result("B"), _result("C")],
    )

    results = hs.hybrid_search(conn=None, query_text="q", query_embedding=[0.1], top_k=2)

    assert len(results) == 2
    # "B" was found by both methods, so it should be ranked first.
    assert results[0].doc_id == "B"


def test_hybrid_search_forwards_ticker_filter_to_both_methods(monkeypatch):
    import finrag.retrieval.hybrid_search as hs

    calls = {}

    def _fake_vector_search(conn, query_embedding, top_k, tickers=None):
        calls["vector_tickers"] = tickers
        return [_result("A")]

    def _fake_keyword_search(conn, query_text, top_k, tickers=None):
        calls["keyword_tickers"] = tickers
        return [_result("A")]

    monkeypatch.setattr(hs, "vector_search", _fake_vector_search)
    monkeypatch.setattr(hs, "keyword_search", _fake_keyword_search)

    hs.hybrid_search(
        conn=None, query_text="q", query_embedding=[0.1], top_k=2, tickers=["NVDA"]
    )

    assert calls["vector_tickers"] == ["NVDA"]
    assert calls["keyword_tickers"] == ["NVDA"]
