"""Measure the impact of agentic tool-calling on answer quality.

Retrieval is held fixed (query rewriting -> hybrid search -> rerank, the
same path rag/pipeline.py uses) for every question; only the final
generation step varies:

    with_tools    -> agent/orchestrator.run_agent_loop() (Day 4 behavior)
    without_tools -> a single llm.complete() call, no tools (Day 3 behavior)

Holding retrieval constant isolates tool-calling as the only variable --
if we let retrieval vary too, a score difference couldn't be attributed
to tool-calling specifically versus just noisier retrieval on that run.

The question set is hand-curated (not sampled from eval/ground_truth.csv)
and split into two categories on purpose:

    numeric    -- needs an exact figure over a specific range/date
                  (the case tools exist for)
    narrative  -- answerable from retrieved context alone

Comparing scores *within* each category is what shows whether tools help
where they're supposed to (numeric) without being a wash or a regression
where they're not needed (narrative).
"""

from __future__ import annotations

import logging
import random
from dataclasses import asdict, dataclass

import pandas as pd
import psycopg

from finrag.agent.orchestrator import run_agent_loop
from finrag.eval.llm_judge import judge_answers
from finrag.knowledge_base.embeddings import Embedder
from finrag.knowledge_base.models import SearchResult
from finrag.llm.base import LLMClient
from finrag.rag.prompts import SYSTEM_INSTRUCTIONS, build_context, build_prompt
from finrag.retrieval.hybrid_search import hybrid_search
from finrag.retrieval.query_rewriter import rewrite_query
from finrag.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

TOP_K = 5
CANDIDATE_POOL = 20

# Same seed every run -> the same A/B position assignment every time (see
# _judge_without_position_bias below), matching this project's general
# "reproducible metrics" goal.
POSITION_SEED = 7

EVAL_QUESTIONS: list[dict[str, str]] = [
    {"question": "What was AAPL's exact percentage return from 2022-01-03 to 2022-06-30?", "category": "numeric"},
    {"question": "What was NVDA's closing price on 2022-03-15?", "category": "numeric"},
    {"question": "What was the annualized volatility of MSFT between 2022-01-01 and 2022-12-31?", "category": "numeric"},
    {"question": "Compare the returns of NVDA and AMD from 2022-01-01 to 2022-12-31.", "category": "numeric"},
    {"question": "What was AAPL's return between 2023-01-01 and 2023-03-31?", "category": "numeric"},
    {"question": "What was JPM's closing price on 2020-03-23?", "category": "numeric"},
    {"question": "Summarize AAPL's performance in 2022.", "category": "narrative"},
    {"question": "Describe AMD's market performance in 2021.", "category": "narrative"},
    {"question": "What was the overall trend for SPY last year?", "category": "narrative"},
    {"question": "Summarize NVDA's 2022.", "category": "narrative"},
    {"question": "How did the technology sector look based on NVDA and AAPL?", "category": "narrative"},
    {"question": "Describe COST's price behavior in 2023.", "category": "narrative"},
]


@dataclass
class LLMEvalRow:
    question: str
    category: str
    with_tools_score: int
    without_tools_score: int
    reasoning: str


def retrieve_for_question(
    conn: psycopg.Connection, llm: LLMClient, embedder: Embedder, reranker: Reranker, question: str
) -> list[SearchResult]:
    intent = rewrite_query(llm, question)
    embedding = embedder.embed([intent.rewritten_query])[0]
    candidates = hybrid_search(
        conn,
        intent.rewritten_query,
        embedding,
        top_k=CANDIDATE_POOL,
        tickers=intent.tickers or None,
    )
    return reranker.rerank(intent.rewritten_query, candidates, top_k=TOP_K)


def generate_with_tools(llm: LLMClient, question: str, retrieved: list[SearchResult]) -> str:
    prompt = build_prompt(question, retrieved)
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": prompt},
    ]
    return run_agent_loop(llm, messages)


def generate_without_tools(llm: LLMClient, question: str, retrieved: list[SearchResult]) -> str:
    prompt = build_prompt(question, retrieved)
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTIONS},
        {"role": "user", "content": prompt},
    ]
    response = llm.complete(messages)  # no tools -- Day 3-style single call
    return response.content or ""


def _judge_without_position_bias(
    llm: LLMClient,
    question: str,
    context: str,
    with_tools_answer: str,
    without_tools_answer: str,
    rng: random.Random,
) -> tuple[int, int, str] | None:
    """Randomly swap which answer is shown as "A" vs "B" before judging,
    then map the scores back -- an LLM judge can favor whichever position
    it sees first/second regardless of content (a documented judge
    artifact), and always putting the same variant in the same slot would
    let that bias masquerade as a real quality difference.
    """
    swap = rng.random() < 0.5
    answer_a = without_tools_answer if swap else with_tools_answer
    answer_b = with_tools_answer if swap else without_tools_answer

    verdict = judge_answers(llm, question, context, answer_a, answer_b)
    if verdict is None:
        return None

    with_tools_score = verdict.answer_b_score if swap else verdict.answer_a_score
    without_tools_score = verdict.answer_a_score if swap else verdict.answer_b_score
    return with_tools_score, without_tools_score, verdict.reasoning


def evaluate_tool_calling_impact(
    conn: psycopg.Connection,
    llm: LLMClient,
    embedder: Embedder,
    reranker: Reranker,
    questions: list[dict[str, str]] = EVAL_QUESTIONS,
) -> list[LLMEvalRow]:
    rng = random.Random(POSITION_SEED)
    rows: list[LLMEvalRow] = []

    for item in questions:
        question, category = item["question"], item["category"]
        logger.info("Evaluating (%s): %r", category, question)

        retrieved = retrieve_for_question(conn, llm, embedder, reranker, question)
        context = build_context(retrieved)

        with_tools_answer = generate_with_tools(llm, question, retrieved)
        without_tools_answer = generate_without_tools(llm, question, retrieved)

        judged = _judge_without_position_bias(
            llm, question, context, with_tools_answer, without_tools_answer, rng
        )
        if judged is None:
            logger.warning("Skipping %r: judge failed to produce a usable verdict", question)
            continue

        with_tools_score, without_tools_score, reasoning = judged
        rows.append(
            LLMEvalRow(
                question=question,
                category=category,
                with_tools_score=with_tools_score,
                without_tools_score=without_tools_score,
                reasoning=reasoning,
            )
        )

    return rows


def results_to_dataframe(rows: list[LLMEvalRow]) -> pd.DataFrame:
    return pd.DataFrame([asdict(r) for r in rows])


def summarize_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """Mean with/without-tools score per category -- the comparison that
    actually matters here, since an overall average could hide "tools help
    a lot on numeric questions, do nothing on narrative ones" behind a
    single unremarkable-looking number.
    """
    if df.empty:
        return df
    return (
        df.groupby("category")[["with_tools_score", "without_tools_score"]]
        .mean()
        .round(2)
        .reset_index()
    )
