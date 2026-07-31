"""Retrieval ranking metrics: Hit Rate and Mean Reciprocal Rank (MRR).

Both take a list of "ranks" -- one per evaluated query, each either the
1-based position of the known-relevant document within that query's
retrieved results, or `None` if it wasn't retrieved at all. Pure functions,
no I/O, so they're trivial to unit test with hand-computed expected values
(see tests/test_eval_metrics.py) -- exactly the same reasoning as Day 2's
compute_stats.py being pure.
"""

from __future__ import annotations


def rank_of(target_doc_id: str, retrieved_doc_ids: list[str]) -> int | None:
    """1-based rank of `target_doc_id` within `retrieved_doc_ids`, or None
    if it isn't present. 1-based because "rank 1" (first result) reads
    naturally; reciprocal-rank math below divides by it directly.
    """
    try:
        return retrieved_doc_ids.index(target_doc_id) + 1
    except ValueError:
        return None


def hit_rate(ranks: list[int | None]) -> float:
    """Fraction of queries where the relevant document was retrieved at
    all. Callers control "at all" vs. "in the top k" by how many results
    they passed to `rank_of` in the first place (e.g. only the top 5).
    """
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if r is not None)
    return hits / len(ranks)


def mean_reciprocal_rank(ranks: list[int | None]) -> float:
    """Average of 1/rank across queries (0 for a query where the relevant
    document wasn't found at all). Rewards finding the relevant document
    *near the top*, not just somewhere in the results -- a document found
    at rank 1 contributes 1.0, at rank 5 only 0.2, unlike Hit Rate which
    treats every found-at-all case identically.
    """
    if not ranks:
        return 0.0
    reciprocal_ranks = [(1.0 / r) if r is not None else 0.0 for r in ranks]
    return sum(reciprocal_ranks) / len(ranks)
