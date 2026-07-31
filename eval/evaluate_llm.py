"""Measure the impact of agentic tool-calling on answer quality: for each
question in finrag.eval.llm_eval.EVAL_QUESTIONS, generate a with-tools and
a without-tools answer from identical retrieved context, have an LLM judge
score both (position-randomized to cancel judge position bias), and print
a per-category summary.

Resumable: each question's result is appended to
eval/results/llm_eval_progress.csv as soon as it's scored, and a re-run
skips any question already present in that file. This matters because
Groq's free-tier daily token budget can run out mid-evaluation
(`groq.RateLimitError`, HTTP 429) -- when that happens, this script stops
cleanly instead of crashing, and everything scored up to that point is
already saved. Run again (any time, including after the quota resets) to
pick up exactly where it left off; nothing already scored is re-paid-for
or re-judged. Pass --fresh to ignore saved progress and start over.

Requires a running Postgres with ingested documents (finrag-ingest) and a
configured Groq API key.

Usage:
    uv run python eval/evaluate_llm.py
    uv run python eval/evaluate_llm.py --fresh
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import groq
import pandas as pd

from finrag.config.settings import get_settings
from finrag.eval.llm_eval import (
    EVAL_QUESTIONS,
    LLMEvalRow,
    evaluate_tool_calling_impact,
    filter_pending_questions,
    results_to_dataframe,
    summarize_by_category,
)
from finrag.knowledge_base.embeddings import Embedder
from finrag.knowledge_base.vector_store import get_connection
from finrag.llm.factory import get_llm_client
from finrag.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PROGRESS_PATH = RESULTS_DIR / "llm_eval_progress.csv"
PROGRESS_COLUMNS = ["question", "category", "with_tools_score", "without_tools_score", "reasoning"]


def _load_progress() -> pd.DataFrame:
    if PROGRESS_PATH.exists():
        return pd.read_csv(PROGRESS_PATH)
    return pd.DataFrame(columns=PROGRESS_COLUMNS)


def _append_row(row: LLMEvalRow) -> None:
    """Write one row immediately, in append mode -- called as soon as
    evaluate_tool_calling_impact finishes judging each question, so a
    rate-limit exception raised on a later question doesn't cost this one
    its saved result.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    header = not PROGRESS_PATH.exists()
    results_to_dataframe([row]).to_csv(PROGRESS_PATH, mode="a", header=header, index=False)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh", action="store_true", help="Ignore saved progress and re-score every question"
    )
    args = parser.parse_args()

    if args.fresh and PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

    already_done = set(_load_progress()["question"])
    pending = filter_pending_questions(EVAL_QUESTIONS, already_done)

    if not pending:
        logger.info("All %d questions already scored in %s.", len(EVAL_QUESTIONS), PROGRESS_PATH)
    else:
        logger.info(
            "%d of %d questions remaining (%d already scored).",
            len(pending),
            len(EVAL_QUESTIONS),
            len(EVAL_QUESTIONS) - len(pending),
        )

        settings = get_settings()
        llm = get_llm_client(settings)
        embedder = Embedder()
        reranker = Reranker()

        try:
            with get_connection(settings) as conn:
                evaluate_tool_calling_impact(
                    conn, llm, embedder, reranker, questions=pending, on_row=_append_row
                )
        except groq.RateLimitError:
            logger.warning(
                "Hit Groq's rate limit (likely the daily token budget), stopping here. "
                "Progress so far is saved in %s -- re-run this script (any time, including "
                "after the quota resets) to continue from where it left off.",
                PROGRESS_PATH,
            )

    df = _load_progress()
    if df.empty:
        print("No results scored yet.")
        return

    print("\n" + df.to_string(index=False) + "\n")
    print("Mean score by category (1-5, higher is better):\n")
    print(summarize_by_category(df).to_string(index=False) + "\n")
    print(f"{len(df)}/{len(EVAL_QUESTIONS)} questions scored so far.")

    if len(df) == len(EVAL_QUESTIONS):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        final_path = RESULTS_DIR / f"llm_eval_{timestamp}.csv"
        df.to_csv(final_path, index=False)
        logger.info("All questions scored. Saved a final snapshot to %s", final_path)


if __name__ == "__main__":
    main()
