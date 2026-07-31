"""Build the retrieval-evaluation ground truth set: (question, doc_id)
pairs where the question is designed to be answered by that specific
document.

Why generate questions from documents (not write them by hand): with
~19,600 documents, hand-labeling is not realistic in the time available,
and it's the standard technique for this exact situation (used in the
course's own retrieval-evaluation material): ask an LLM "what question
would this document answer", then treat that document as the known
answer for Hit Rate/MRR purposes. The LLM never sees retrieval at this
stage -- it only ever reads the document being asked about, so there's no
circularity in later using retrieval to try to find that same document.

Sampling is stratified (a fixed number of yearly/monthly/weekly documents
per ticker) rather than uniform-random over all 19,600 documents, so every
ticker and every granularity is represented in the evaluation -- a
uniform-random sample over the whole corpus would be dominated by weekly
documents (there are ~10x more of them than yearly ones) and could
under-represent smaller tickers.
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import pandas as pd
import psycopg

from finrag.config.tickers import all_tickers
from finrag.llm.base import LLMClient

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
GROUND_TRUTH_PATH = PROJECT_ROOT / "eval" / "ground_truth.csv"

# Fixed seed: re-running sampling produces the identical set of documents
# every time, which is what "reproducible metrics" (see docs/PROJECT_PLAN.md)
# requires -- the ground truth set shouldn't silently change between runs.
SAMPLING_SEED = 42

DEFAULT_PER_TICKER = {"yearly": 1, "monthly": 2, "weekly": 2}

_QUESTION_PROMPT_INSTRUCTIONS = """\
You will be shown one document from a financial knowledge base describing \
a stock's price behavior over a period of time.

Write ONE realistic question a user might ask that this document fully \
answers. The question should NOT mention exact numbers from the document \
(a real user wouldn't already know the answer). Respond with ONLY the \
question, no quotes, no other text.
"""


def sample_documents(
    conn: psycopg.Connection,
    per_ticker: dict[str, int] | None = None,
    seed: int = SAMPLING_SEED,
) -> list[dict]:
    """Stratified sample of documents: `per_ticker[granularity]` documents
    per ticker per granularity, deterministically chosen via `seed`.
    """
    per_ticker = per_ticker or DEFAULT_PER_TICKER
    sampled: list[dict] = []
    rng = random.Random(seed)

    with conn.cursor() as cur:
        for ticker in all_tickers():
            for granularity, count in per_ticker.items():
                cur.execute(
                    """
                    SELECT doc_id, ticker, granularity, period_start, period_end, content
                    FROM documents
                    WHERE ticker = %s AND granularity = %s
                    ORDER BY period_start
                    """,
                    (ticker, granularity),
                )
                rows = cur.fetchall()
                if not rows:
                    logger.warning("No %s documents found for %s", granularity, ticker)
                    continue

                chosen = rng.sample(rows, k=min(count, len(rows)))
                for row in chosen:
                    sampled.append(
                        {
                            "doc_id": row[0],
                            "ticker": row[1],
                            "granularity": row[2],
                            "period_start": row[3],
                            "period_end": row[4],
                            "content": row[5],
                        }
                    )
    return sampled


def generate_question_for_document(llm: LLMClient, content: str) -> str | None:
    """Ask the LLM for one question this document would answer. Returns
    None (rather than raising) if the call fails or returns something
    unusable -- a single bad generation shouldn't abort the whole batch.
    """
    try:
        response = llm.complete(
            messages=[
                {"role": "system", "content": _QUESTION_PROMPT_INSTRUCTIONS},
                {"role": "user", "content": content},
            ]
        )
        question = (response.content or "").strip().strip('"')
        return question or None
    except Exception:
        logger.exception("Question generation failed for one document; skipping it")
        return None


def build_ground_truth(
    llm: LLMClient,
    conn: psycopg.Connection,
    per_ticker: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Sample documents, generate one question per document, and return a
    DataFrame with columns [question, doc_id, ticker, granularity,
    period_start, period_end]. Documents where question generation failed
    are dropped, not left with an empty question.
    """
    sampled = sample_documents(conn, per_ticker=per_ticker)
    logger.info("Sampled %d documents for ground-truth generation", len(sampled))

    records = []
    for doc in sampled:
        question = generate_question_for_document(llm, doc["content"])
        if question is None:
            continue
        records.append(
            {
                "question": question,
                "doc_id": doc["doc_id"],
                "ticker": doc["ticker"],
                "granularity": doc["granularity"],
                "period_start": doc["period_start"],
                "period_end": doc["period_end"],
            }
        )

    logger.info("Generated %d/%d ground-truth questions", len(records), len(sampled))
    return pd.DataFrame.from_records(records)
