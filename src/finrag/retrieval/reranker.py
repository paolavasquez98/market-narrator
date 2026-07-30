"""Cross-encoder reranking: a second, more accurate relevance pass over an
already-retrieved candidate set.

Why this is a separate step from hybrid search rather than "just retrieve
better": vector search is a *bi-encoder* -- the query and each document are
embedded independently, and relevance is approximated by comparing those
two fixed vectors. That's what makes it fast enough to index (pgvector's
HNSW index) and search over the whole knowledge base. A *cross-encoder*
instead reads the query and one document *together* in a single forward
pass, letting it model interactions a bi-encoder's independent embeddings
structurally cannot capture -- but it cannot be indexed (there's no fixed
per-document vector to precompute), and is far more expensive per
comparison. The standard pattern, used here, is to let the cheap method
(hybrid search) narrow the whole knowledge base down to a small candidate
set, then spend the cross-encoder's extra accuracy only on those few
candidates.

Uses `fastembed`'s `TextCrossEncoder` for the same reason Day 2 chose
`fastembed` for embeddings: local, free, ONNX-based (no torch dependency),
deterministic.
"""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from finrag.knowledge_base.models import SearchResult

RERANK_MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL_NAME) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self._model = TextCrossEncoder(model_name=model_name)

    def rerank(self, query: str, results: list[SearchResult], top_k: int) -> list[SearchResult]:
        """Score every candidate against `query` with the cross-encoder and
        return the top `top_k`, replacing each result's `score` with the
        cross-encoder's score (the RRF score that got it retrieved is no
        longer meaningful once we have a more direct relevance signal).
        """
        if not results:
            return []

        scores = list(self._model.rerank(query, [r.content for r in results]))
        rescored = [replace(r, score=float(s)) for r, s in zip(results, scores)]
        rescored.sort(key=lambda r: r.score, reverse=True)
        return rescored[:top_k]


@lru_cache
def get_reranker() -> Reranker:
    """Process-wide cached Reranker instance -- same rationale as
    `embeddings.get_embedder()`: loading the ONNX model has a real
    one-time cost that a single request shouldn't pay repeatedly.
    """
    return Reranker()
