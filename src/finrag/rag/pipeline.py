"""End-to-end RAG pipeline: query rewriting -> hybrid search -> rerank ->
prompt -> agentic LLM answer (with tool-calling).

This is the full agentic RAG flow from docs/PROJECT_PLAN.md's Option C.
As of Day 4:

- Query rewriting (retrieval/query_rewriter.py) resolves company names to
  tickers in the fixed universe and cleans up the search query, fixing
  Day 3's "Nvidia retrieves AMD" failure mode.
- Hybrid search now takes a ticker filter, narrowing the whole knowledge
  base before ranking even happens.
- A cross-encoder reranks the (larger) hybrid candidate pool down to the
  final top-k passed to the LLM.
- The LLM runs inside an agent loop (agent/orchestrator.py) with four
  deterministic financial tools available, so exact numbers come from
  pandas arithmetic, not from the model estimating over retrieved text.

Still deliberately not done: retrieval-method selection (Day 5's
evaluation compares keyword-only/vector-only/hybrid and the winner is
what stays wired in here -- today it's always hybrid+rerank).

Usage:
    uv run python -m finrag.rag.pipeline "How did AAPL do in 2022?"
    uv run finrag-ask "Compare MSFT and NVDA over the last year" --top-k 8
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

from finrag.agent.orchestrator import run_agent_loop
from finrag.config.settings import get_settings
from finrag.knowledge_base.embeddings import get_embedder
from finrag.knowledge_base.models import SearchResult
from finrag.knowledge_base.vector_store import get_connection
from finrag.llm.factory import get_llm_client
from finrag.rag.prompts import SYSTEM_INSTRUCTIONS, build_prompt
from finrag.retrieval.hybrid_search import hybrid_search
from finrag.retrieval.query_rewriter import rewrite_query
from finrag.retrieval.reranker import get_reranker

logger = logging.getLogger(__name__)

# Pull more candidates from hybrid search than we ultimately want, so the
# reranker has a real pool to work with (same "cast a slightly wider net
# before narrowing" idea as hybrid_search's candidates_per_method).
DEFAULT_TOP_K = 5
DEFAULT_RERANK_CANDIDATES = 20


@dataclass
class RagAnswer:
    question: str
    answer: str
    retrieved: list[SearchResult]
    rewritten_query: str = ""
    resolved_tickers: list[str] = field(default_factory=list)


def answer_question(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    rerank_candidates: int = DEFAULT_RERANK_CANDIDATES,
) -> RagAnswer:
    """Run the full pipeline for one question.

    Two LLM calls happen before the answer is final even in the simplest
    case: one for query rewriting, one (at least) inside the agent loop.
    That's a deliberate latency/cost trade-off in exchange for the ticker
    filter's precision -- see docs/learning/day04_learning.md for the
    reasoning and the alternatives considered.
    """
    settings = get_settings()
    llm = get_llm_client(settings)

    intent = rewrite_query(llm, question)
    logger.info("Rewritten query: %r, resolved tickers: %s", intent.rewritten_query, intent.tickers)

    embedder = get_embedder()
    query_embedding = embedder.embed([intent.rewritten_query])[0]

    with get_connection(settings) as conn:
        candidates = hybrid_search(
            conn,
            intent.rewritten_query,
            query_embedding,
            top_k=rerank_candidates,
            tickers=intent.tickers or None,
        )

    reranker = get_reranker()
    retrieved = reranker.rerank(intent.rewritten_query, candidates, top_k=top_k)

    prompt = build_prompt(question, retrieved)
    logger.debug("Prompt sent to LLM:\n%s", prompt)

    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": prompt},
    ]
    answer = run_agent_loop(llm, messages)

    return RagAnswer(
        question=question,
        answer=answer,
        retrieved=retrieved,
        rewritten_query=intent.rewritten_query,
        resolved_tickers=intent.tickers,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask Market Narrator a question.")
    parser.add_argument("question", type=str, help="Natural-language question")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="Chunks kept after reranking")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    result = answer_question(args.question, top_k=args.top_k)

    print(f"\nQ: {result.question}\n")
    if result.resolved_tickers:
        print(f"(resolved tickers: {', '.join(result.resolved_tickers)})")
    print(f"A: {result.answer}\n")
    print("Retrieved:")
    for r in result.retrieved:
        print(f"  - {r.doc_id}  (score={r.score:.4f})")


if __name__ == "__main__":
    main()
