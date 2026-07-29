"""End-to-end RAG pipeline: hybrid search -> prompt -> LLM answer.

This is the "basic RAG pipeline working end-to-end via script" milestone
(Day 3 of docs/PROJECT_PLAN.md). Two things it deliberately does NOT do
yet, both scheduled for later days rather than forgotten:

- Query rewriting (Day 4): the raw question goes straight to search.
- Agentic tool-calling for exact numbers (Day 4): the LLM only ever sees
  retrieved text, never a deterministic calculation.
- Retrieval-method selection (Day 5): this pipeline always uses hybrid
  search. Day 5's evaluation compares keyword-only/vector-only/hybrid and
  the winner is what stays wired in here.

Usage:
    uv run python -m finrag.rag.pipeline "How did AAPL do in 2022?"
    uv run finrag-ask "Compare MSFT and NVDA over the last year" --top-k 8
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

from finrag.config.settings import get_settings
from finrag.knowledge_base.embeddings import get_embedder
from finrag.knowledge_base.models import SearchResult
from finrag.knowledge_base.vector_store import get_connection
from finrag.llm.factory import get_llm_client
from finrag.rag.prompts import SYSTEM_INSTRUCTIONS, build_prompt
from finrag.retrieval.hybrid_search import hybrid_search

logger = logging.getLogger(__name__)


@dataclass
class RagAnswer:
    question: str
    answer: str
    retrieved: list[SearchResult]


def answer_question(question: str, top_k: int = 5) -> RagAnswer:
    """Run the full pipeline for one question: embed -> hybrid search ->
    build prompt -> call the LLM -> return the answer plus what was
    retrieved (kept around for logging/debugging, not just the text).
    """
    settings = get_settings()
    embedder = get_embedder()
    query_embedding = embedder.embed([question])[0]

    with get_connection(settings) as conn:
        retrieved = hybrid_search(conn, question, query_embedding, top_k=top_k)

    prompt = build_prompt(question, retrieved)
    logger.debug("Prompt sent to LLM:\n%s", prompt)

    llm = get_llm_client(settings)
    response = llm.complete(
        messages=[
            {"role": "system", "content": SYSTEM_INSTRUCTIONS},
            {"role": "user", "content": prompt},
        ]
    )

    return RagAnswer(question=question, answer=response.content or "", retrieved=retrieved)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ask Market Narrator a question.")
    parser.add_argument("question", type=str, help="Natural-language question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    result = answer_question(args.question, top_k=args.top_k)

    print(f"\nQ: {result.question}\n")
    print(f"A: {result.answer}\n")
    print("Retrieved:")
    for r in result.retrieved:
        print(f"  - {r.doc_id}  (score={r.score:.4f})")


if __name__ == "__main__":
    main()
