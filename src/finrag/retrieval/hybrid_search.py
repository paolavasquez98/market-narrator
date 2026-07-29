"""Hybrid search: combine vector and keyword search using Reciprocal Rank
Fusion (RRF).

Why fusion is needed at all: vector search returns cosine similarities
(roughly 0-1), keyword search returns `ts_rank` values (an unbounded,
corpus-dependent scale). Averaging those two numbers directly would be
comparing units that don't mean the same thing -- a 0.7 cosine similarity
and a 0.7 ts_rank are not "equally good" in any meaningful sense.

RRF sidesteps that entirely by only looking at *rank position* within
each list, never the raw score:

    RRF_score(doc) = sum over each list the doc appears in of  1 / (k + rank)

A document ranked #1 by both vector and keyword search accumulates two
large terms and comes out on top; a document that only one method found
still gets credit, just less of it. `k=60` is the constant from the
original RRF paper (Cormack et al., 2009) -- large enough that rank 1 vs.
rank 2 isn't a cliff, small enough that being highly ranked still matters
much more than being buried at rank 40.
"""

from __future__ import annotations

from dataclasses import replace

import psycopg

from finrag.knowledge_base.keyword_store import keyword_search
from finrag.knowledge_base.models import SearchResult
from finrag.knowledge_base.vector_store import vector_search

RRF_K = 60


def reciprocal_rank_fusion(
    *ranked_lists: list[SearchResult], k: int = RRF_K
) -> list[SearchResult]:
    """Fuse any number of ranked result lists into one, ordered by combined
    RRF score. Pure function: no I/O, so it's fully unit-testable with
    hand-built fake result lists (see tests/test_hybrid_search.py).
    """
    rrf_scores: dict[str, float] = {}
    best_by_doc_id: dict[str, SearchResult] = {}

    for ranked_list in ranked_lists:
        for rank, result in enumerate(ranked_list, start=1):
            rrf_scores[result.doc_id] = rrf_scores.get(result.doc_id, 0.0) + 1.0 / (k + rank)
            # Keep one representative SearchResult per doc_id (content is
            # identical regardless of which list it came from).
            best_by_doc_id.setdefault(result.doc_id, result)

    fused = [
        replace(result, score=rrf_scores[doc_id]) for doc_id, result in best_by_doc_id.items()
    ]
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused


def hybrid_search(
    conn: psycopg.Connection,
    query_text: str,
    query_embedding: list[float],
    top_k: int = 10,
    candidates_per_method: int = 30,
) -> list[SearchResult]:
    """Run vector and keyword search independently, then fuse.

    `candidates_per_method` (default 30) is deliberately larger than
    `top_k`: fusion needs enough candidates from each method for RRF to
    have something to combine -- if we only fetched `top_k` from each,
    a document ranked #8 by vector search but missing from keyword
    search's top-`top_k` might unfairly look like it "wasn't found" by
    keyword search at all, when a slightly larger candidate pool would
    have included it.
    """
    vector_results = vector_search(conn, query_embedding, top_k=candidates_per_method)
    keyword_results = keyword_search(conn, query_text, top_k=candidates_per_method)

    fused = reciprocal_rank_fusion(vector_results, keyword_results)
    return fused[:top_k]
