"""Tests for llm_eval.py's generation/judging wiring.

These are dispatch-style tests (like test_retrieval_eval.py): they verify
generate_with_tools/generate_without_tools call exactly the right
underlying function, and that the position-bias-randomization logic maps
judge scores back to the correct with/without-tools variant regardless of
which slot ("A" or "B") each was placed in. None of this exercises a real
LLM -- see docs/learning/day05_learning.md for what still needs live
verification against Groq.
"""

from __future__ import annotations

import random
from datetime import date
from unittest.mock import MagicMock

import finrag.eval.llm_eval as llm_eval_module
from finrag.eval.llm_eval import (
    EVAL_QUESTIONS,
    LLMEvalRow,
    _judge_without_position_bias,
    generate_with_tools,
    generate_without_tools,
    results_to_dataframe,
    summarize_by_category,
)
from finrag.eval.llm_judge import JudgeVerdict
from finrag.knowledge_base.models import SearchResult


def _result() -> SearchResult:
    return SearchResult(
        doc_id="AAPL:weekly:2024-01-01",
        ticker="AAPL",
        sector="Technology",
        granularity="weekly",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 1, 5),
        content="AAPL rose this week.",
        score=1.0,
    )


def test_generate_with_tools_calls_run_agent_loop(monkeypatch):
    fake_run_agent_loop = MagicMock(return_value="tool-backed answer")
    monkeypatch.setattr(llm_eval_module, "run_agent_loop", fake_run_agent_loop)

    answer = generate_with_tools(MagicMock(), "How did AAPL do?", [_result()])

    assert answer == "tool-backed answer"
    fake_run_agent_loop.assert_called_once()


def test_generate_without_tools_calls_bare_complete_with_no_tools(monkeypatch):
    fake_llm = MagicMock()
    fake_llm.complete.return_value = MagicMock(content="context-only answer")

    answer = generate_without_tools(fake_llm, "How did AAPL do?", [_result()])

    assert answer == "context-only answer"
    fake_llm.complete.assert_called_once()
    # Only `messages` should be passed -- no `tools` kwarg at all, mirroring
    # the Day 3 single-call behavior we're comparing against.
    assert fake_llm.complete.call_args.args or fake_llm.complete.call_args.kwargs
    assert "tools" not in fake_llm.complete.call_args.kwargs


def test_judge_without_position_bias_maps_scores_back_when_not_swapped(monkeypatch):
    fake_judge = MagicMock(return_value=JudgeVerdict(answer_a_score=4, answer_b_score=2, reasoning="r"))
    monkeypatch.setattr(llm_eval_module, "judge_answers", fake_judge)
    fake_llm = MagicMock()

    rng = random.Random(0)
    # With seed 0, force the "no swap" branch by monkeypatching rng.random directly.
    rng.random = lambda: 0.9  # >= 0.5 -> swap is False

    result = _judge_without_position_bias(fake_llm, "q", "ctx", "with-tools-answer", "without-tools-answer", rng)

    assert result is not None
    with_tools_score, without_tools_score, reasoning = result
    assert with_tools_score == 4
    assert without_tools_score == 2
    assert reasoning == "r"
    # Answer A should have been the with-tools answer in the no-swap case.
    fake_judge.assert_called_once_with(fake_llm, "q", "ctx", "with-tools-answer", "without-tools-answer")


def test_judge_without_position_bias_maps_scores_back_when_swapped(monkeypatch):
    fake_judge = MagicMock(return_value=JudgeVerdict(answer_a_score=4, answer_b_score=2, reasoning="r"))
    monkeypatch.setattr(llm_eval_module, "judge_answers", fake_judge)

    rng = random.Random(0)
    rng.random = lambda: 0.1  # < 0.5 -> swap is True

    result = _judge_without_position_bias(MagicMock(), "q", "ctx", "with-tools-answer", "without-tools-answer", rng)

    assert result is not None
    with_tools_score, without_tools_score, _reasoning = result
    # Swapped: answer A was without-tools (scored 4), answer B was with-tools (scored 2).
    assert with_tools_score == 2
    assert without_tools_score == 4

    call_args = fake_judge.call_args.args
    assert call_args[3] == "without-tools-answer"  # answer_a
    assert call_args[4] == "with-tools-answer"  # answer_b


def test_judge_without_position_bias_returns_none_when_judge_fails(monkeypatch):
    monkeypatch.setattr(llm_eval_module, "judge_answers", MagicMock(return_value=None))

    rng = random.Random(0)
    result = _judge_without_position_bias(MagicMock(), "q", "ctx", "a", "b", rng)

    assert result is None


def test_eval_questions_cover_both_categories():
    categories = {item["category"] for item in EVAL_QUESTIONS}
    assert categories == {"numeric", "narrative"}
    assert len(EVAL_QUESTIONS) >= 4


def test_summarize_by_category_averages_per_category():
    rows = [
        LLMEvalRow(question="q1", category="numeric", with_tools_score=5, without_tools_score=1, reasoning=""),
        LLMEvalRow(question="q2", category="numeric", with_tools_score=3, without_tools_score=1, reasoning=""),
        LLMEvalRow(question="q3", category="narrative", with_tools_score=4, without_tools_score=4, reasoning=""),
    ]
    df = results_to_dataframe(rows)
    summary = summarize_by_category(df)

    numeric_row = summary[summary["category"] == "numeric"].iloc[0]
    assert numeric_row["with_tools_score"] == 4.0
    assert numeric_row["without_tools_score"] == 1.0

    narrative_row = summary[summary["category"] == "narrative"].iloc[0]
    assert narrative_row["with_tools_score"] == 4.0
    assert narrative_row["without_tools_score"] == 4.0
