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
    _format_tool_trace,
    _judge_without_position_bias,
    evaluate_tool_calling_impact,
    filter_pending_questions,
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


# Day 9: generate_with_tools forwards an optional tool_trace list straight
# through to run_agent_loop, so evaluate_tool_calling_impact can capture
# what the agent actually called and show it to the judge.
def test_generate_with_tools_forwards_tool_trace_to_run_agent_loop(monkeypatch):
    fake_run_agent_loop = MagicMock(return_value="tool-backed answer")
    monkeypatch.setattr(llm_eval_module, "run_agent_loop", fake_run_agent_loop)
    trace: list[dict] = []

    generate_with_tools(MagicMock(), "How did AAPL do?", [_result()], tool_trace=trace)

    assert fake_run_agent_loop.call_args.kwargs["tool_trace"] is trace


def test_format_tool_trace_renders_one_line_per_call():
    trace = [
        {
            "name": "get_return",
            "arguments": {"ticker": "AAPL", "start_date": "2022-01-01", "end_date": "2022-12-31"},
            "result": {"return_pct": 5.0},
        }
    ]
    formatted = _format_tool_trace(trace)
    assert formatted == (
        "get_return(ticker=AAPL, start_date=2022-01-01, end_date=2022-12-31) -> {'return_pct': 5.0}"
    )


def test_format_tool_trace_returns_empty_string_for_no_calls():
    assert _format_tool_trace([]) == ""


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
    fake_judge.assert_called_once_with(
        fake_llm, "q", "ctx", "with-tools-answer", "without-tools-answer", tool_results=""
    )


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


# --- Day 6: resumability ---
# eval/evaluate_llm.py needs to survive a Groq rate-limit interruption
# without losing (or re-paying for) completed work. These tests cover the
# two pieces that make that possible: filtering out already-scored
# questions before a run starts, and evaluate_tool_calling_impact
# reporting each row as soon as it's ready via `on_row` rather than only
# once the whole batch finishes.


def test_filter_pending_questions_drops_already_done():
    questions = [
        {"question": "q1", "category": "numeric"},
        {"question": "q2", "category": "narrative"},
        {"question": "q3", "category": "numeric"},
    ]
    pending = filter_pending_questions(questions, already_done={"q1", "q3"})
    assert pending == [{"question": "q2", "category": "narrative"}]


def test_filter_pending_questions_returns_everything_when_nothing_done():
    questions = [{"question": "q1", "category": "numeric"}]
    assert filter_pending_questions(questions, already_done=set()) == questions


def _wire_fake_pipeline(monkeypatch, judge_return=None):
    """Monkeypatch every dependency evaluate_tool_calling_impact calls, so
    it can run end-to-end against a fake question list with no real DB,
    embedder, reranker, or LLM.
    """
    monkeypatch.setattr(llm_eval_module, "retrieve_for_question", lambda *a, **k: [_result()])
    monkeypatch.setattr(llm_eval_module, "build_context", lambda retrieved: "ctx")

    def _fake_generate_with_tools(llm, q, retrieved, tool_trace=None):
        if tool_trace is not None:
            tool_trace.append({"name": "get_return", "arguments": {"ticker": "AAPL"}, "result": {}})
        return f"with:{q}"

    monkeypatch.setattr(llm_eval_module, "generate_with_tools", _fake_generate_with_tools)
    monkeypatch.setattr(
        llm_eval_module, "generate_without_tools", lambda llm, q, retrieved: f"without:{q}"
    )
    verdict = judge_return or JudgeVerdict(answer_a_score=5, answer_b_score=1, reasoning="r")
    monkeypatch.setattr(llm_eval_module, "judge_answers", lambda *a, **k: verdict)


def test_evaluate_tool_calling_impact_calls_on_row_for_each_completed_question(monkeypatch):
    _wire_fake_pipeline(monkeypatch)
    questions = [
        {"question": "q1", "category": "numeric"},
        {"question": "q2", "category": "narrative"},
    ]
    seen: list[LLMEvalRow] = []

    rows = evaluate_tool_calling_impact(
        conn=None, llm=MagicMock(), embedder=MagicMock(), reranker=MagicMock(),
        questions=questions, on_row=seen.append,
    )

    assert [r.question for r in seen] == ["q1", "q2"]
    assert seen == rows


def test_evaluate_tool_calling_impact_passes_captured_tool_trace_to_judge(monkeypatch):
    """End-to-end wiring check for the Day 9 fix: whatever tool calls
    generate_with_tools reports via tool_trace should reach judge_answers
    as a formatted tool_results string -- not just be captured and
    discarded.
    """
    _wire_fake_pipeline(monkeypatch)
    captured_kwargs = {}

    def _capturing_judge(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return JudgeVerdict(answer_a_score=5, answer_b_score=1, reasoning="r")

    monkeypatch.setattr(llm_eval_module, "judge_answers", _capturing_judge)

    evaluate_tool_calling_impact(
        conn=None, llm=MagicMock(), embedder=MagicMock(), reranker=MagicMock(),
        questions=[{"question": "q1", "category": "numeric"}],
    )

    assert "get_return(ticker=AAPL)" in captured_kwargs["tool_results"]


def test_evaluate_tool_calling_impact_position_bias_is_stable_regardless_of_batch_position(monkeypatch):
    """The whole point of seeding the swap decision per-question (not
    sequentially from one shared Random) is that resuming a partial run
    reproduces the same A/B assignment a single uninterrupted run would
    have made. Verify that directly: score a question alone vs. score it
    as the second of two questions, and confirm the result is identical.
    """
    _wire_fake_pipeline(monkeypatch)
    target = {"question": "How did AAPL do in 2022?", "category": "narrative"}
    other = {"question": "What was NVDA's return in 2021?", "category": "numeric"}

    [alone_row] = evaluate_tool_calling_impact(
        conn=None, llm=MagicMock(), embedder=MagicMock(), reranker=MagicMock(), questions=[target]
    )
    batch_rows = evaluate_tool_calling_impact(
        conn=None, llm=MagicMock(), embedder=MagicMock(), reranker=MagicMock(),
        questions=[other, target],
    )
    batch_row = next(r for r in batch_rows if r.question == target["question"])

    assert alone_row.with_tools_score == batch_row.with_tools_score
    assert alone_row.without_tools_score == batch_row.without_tools_score
