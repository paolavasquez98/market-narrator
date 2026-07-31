"""Compare retrieval configurations against the ground truth set, using
Hit Rate and MRR (eval/metrics.py).

Five configurations, each adding exactly one thing on top of the last:

    keyword                -> keyword_search() alone, raw question
    vector                 -> vector_search() alone, raw question
    hybrid                 -> hybrid_search() (RRF fusion), raw question
    hybrid_rerank          -> + cross-encoder reranking, raw question
    hybrid_rerank_rewrite  -> + LLM query rewriting/ticker filter
                               (this is what rag/pipeline.py actually runs)

Holding everything else fixed while adding one piece at a time is what
makes it possible to attribute a Hit Rate/MRR change to a *specific*
architectural decision (does reranking actually help? does rewriting?)
instead of only ever comparing "the whole pipeline" against nothing.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import pandas as pd
import psycopg

from finrag.eval.metrics import hit_rate, mean_reciprocal_rank, rank_of
from finrag.knowledge_base.embeddings import Embedder
from finrag.knowledge_base.keyword_store import keyword_search
from finrag.knowledge_base.vector_store import vector_search
from finrag.llm.base import LLMClient
from finrag.retrieval.hybrid_search import hybrid_search
from finrag.retrieval.query_rewriter import rewrite_query
from finrag.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

TOP_K = 5
CANDIDATE_POOL = 20

VARIANTS = ("keyword", "vector", "hybrid", "hybrid_rerank", "hybrid_rerank_rewrite")


@dataclass
class RetrievalEvalResult:
    variant: str
    hit_rate: float
    mrr: float
    n_queries: int


def _retrieve_doc_ids(
    variant: str,
    question: str,
    conn: psycopg.Connection,
    llm: LLMClient,
    embedder: Embedder,
    reranker: Reranker,
) -> list[str]:
    """Run one variant's retrieval for one question, returning just the
    ordered doc_ids (all `rank_of` needs).
    """
    if variant == "keyword":
        results = keyword_search(conn, question, top_k=TOP_K)

    elif variant == "vector":
        embedding = embedder.embed([question])[0]
        results = vector_search(conn, embedding, top_k=TOP_K)

    elif variant == "hybrid":
        embedding = embedder.embed([question])[0]
        results = hybrid_search(conn, question, embedding, top_k=TOP_K)

    elif variant == "hybrid_rerank":
        embedding = embedder.embed([question])[0]
        candidates = hybrid_search(conn, question, embedding, top_k=CANDIDATE_POOL)
        results = reranker.rerank(question, candidates, top_k=TOP_K)

    elif variant == "hybrid_rerank_rewrite":
        intent = rewrite_query(llm, question)
        embedding = embedder.embed([intent.rewritten_query])[0]
        candidates = hybrid_search(
            conn,
            intent.rewritten_query,
            embedding,
            top_k=CANDIDATE_POOL,
            tickers=intent.tickers or None,
        )
        results = reranker.rerank(intent.rewritten_query, candidates, top_k=TOP_K)

    else:
        raise ValueError(f"Unknown retrieval variant: {variant!r}")

    return [r.doc_id for r in results]


def evaluate_variant(
    variant: str,
    conn: psycopg.Connection,
    llm: LLMClient,
    embedder: Embedder,
    reranker: Reranker,
    ground_truth: pd.DataFrame,
) -> RetrievalEvalResult:
    ranks = []
    for _, row in ground_truth.iterrows():
        doc_ids = _retrieve_doc_ids(variant, row["question"], conn, llm, embedder, reranker)
        ranks.append(rank_of(row["doc_id"], doc_ids))

    return RetrievalEvalResult(
        variant=variant,
        hit_rate=hit_rate(ranks),
        mrr=mean_reciprocal_rank(ranks),
        n_queries=len(ranks),
    )


def evaluate_all_variants(
    conn: psycopg.Connection,
    llm: LLMClient,
    embedder: Embedder,
    reranker: Reranker,
    ground_truth: pd.DataFrame,
    variants: tuple[str, ...] = VARIANTS,
) -> list[RetrievalEvalResult]:
    results = []
    for variant in variants:
        logger.info("Evaluating retrieval variant: %s", variant)
        results.append(evaluate_variant(variant, conn, llm, embedder, reranker, ground_truth))
    return results


def results_to_dataframe(results: list[RetrievalEvalResult]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in results])


def best_variant(results: list[RetrievalEvalResult]) -> RetrievalEvalResult:
    """Pick the winner by MRR first (rewards ranking the relevant document
    *near the top*, which is what actually matters for the LLM's prompt --
    it only sees the top few chunks), breaking ties by Hit Rate.
    """
    return max(results, key=lambda r: (r.mrr, r.hit_rate))
