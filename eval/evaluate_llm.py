"""Measure the impact of agentic tool-calling on answer quality: for each
question in finrag.eval.llm_eval.EVAL_QUESTIONS, generate a with-tools and
a without-tools answer from identical retrieved context, have an LLM judge
score both (position-randomized to cancel judge position bias), and print
a per-category summary. Saves a timestamped results file to eval/results/.

Requires a running Postgres with ingested documents (finrag-ingest) and a
configured Groq API key -- this makes real LLM calls (2 generations + 1
judge call per question) and cannot run in CI/sandbox without both.

Usage:
    uv run python eval/evaluate_llm.py
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from finrag.config.settings import get_settings
from finrag.eval.llm_eval import (
    evaluate_tool_calling_impact,
    results_to_dataframe,
    summarize_by_category,
)
from finrag.knowledge_base.embeddings import Embedder
from finrag.knowledge_base.vector_store import get_connection
from finrag.llm.factory import get_llm_client
from finrag.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    settings = get_settings()
    llm = get_llm_client(settings)
    embedder = Embedder()
    reranker = Reranker()

    with get_connection(settings) as conn:
        rows = evaluate_tool_calling_impact(conn, llm, embedder, reranker)

    df = results_to_dataframe(rows)
    print("\n" + df.to_string(index=False) + "\n")

    summary = summarize_by_category(df)
    print("Mean score by category (1-5, higher is better):\n")
    print(summary.to_string(index=False) + "\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"llm_eval_{timestamp}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
