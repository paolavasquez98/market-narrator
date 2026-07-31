"""Tests for failure_cases.py's harness plumbing: run_failure_cases wires
each FailureCase's question into rag.pipeline.answer_question and captures
its output; render_markdown produces a readable transcript. Neither test
exercises the real pipeline (that needs a live DB + Groq key) -- they only
check the wiring and rendering logic are correct, since a bug here would
silently produce a misleading or incomplete transcript for the human
reviewer these tools exist for.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import finrag.eval.failure_cases as fc_module
from finrag.eval.failure_cases import (
    FailureCase,
    FailureCaseResult,
    filter_pending_cases,
    render_markdown,
    run_failure_cases,
)
from finrag.rag.pipeline import RagAnswer


def _fake_rag_answer(question: str) -> RagAnswer:
    return RagAnswer(
        question=question,
        answer=f"answer for: {question}",
        retrieved=[],
        rewritten_query=f"rewritten: {question}",
        resolved_tickers=["AAPL"],
    )


def test_run_failure_cases_calls_answer_question_once_per_case(monkeypatch):
    fake_answer_question = MagicMock(side_effect=lambda q: _fake_rag_answer(q))
    monkeypatch.setattr(fc_module, "answer_question", fake_answer_question)

    cases = [
        FailureCase(id="c1", category="cat1", question="q1", known_issue="issue1"),
        FailureCase(id="c2", category="cat2", question="q2", known_issue="issue2"),
    ]
    results = run_failure_cases(cases)

    assert fake_answer_question.call_count == 2
    fake_answer_question.assert_any_call("q1")
    fake_answer_question.assert_any_call("q2")

    assert [r.id for r in results] == ["c1", "c2"]
    assert results[0].answer == "answer for: q1"
    assert results[0].rewritten_query == "rewritten: q1"
    assert results[0].resolved_tickers == ["AAPL"]
    assert results[0].known_issue == "issue1"


def test_render_markdown_includes_all_case_fields(monkeypatch):
    fake_answer_question = MagicMock(side_effect=lambda q: _fake_rag_answer(q))
    monkeypatch.setattr(fc_module, "answer_question", fake_answer_question)

    cases = [FailureCase(id="my-case", category="my_category", question="my question?", known_issue="the known issue")]
    results = run_failure_cases(cases)
    transcript = render_markdown(results)

    assert "my-case" in transcript
    assert "my_category" in transcript
    assert "my question?" in transcript
    assert "the known issue" in transcript
    assert "answer for: my question?" in transcript
    assert "rewritten: my question?" in transcript


def test_render_markdown_handles_empty_results():
    transcript = render_markdown([])
    assert transcript.startswith("# Failure case transcript")


# --- Day 6: resumability ---
# eval/run_failure_cases.py needs to survive a Groq rate-limit
# interruption without losing (or re-paying for) completed cases.


def test_filter_pending_cases_drops_already_done():
    cases = [
        FailureCase(id="c1", category="cat", question="q1", known_issue="i1"),
        FailureCase(id="c2", category="cat", question="q2", known_issue="i2"),
    ]
    pending = filter_pending_cases(cases, already_done={"c1"})
    assert [c.id for c in pending] == ["c2"]


def test_filter_pending_cases_returns_everything_when_nothing_done():
    cases = [FailureCase(id="c1", category="cat", question="q1", known_issue="i1")]
    assert filter_pending_cases(cases, already_done=set()) == cases


def test_run_failure_cases_calls_on_result_for_each_completed_case(monkeypatch):
    monkeypatch.setattr(fc_module, "answer_question", lambda q: _fake_rag_answer(q))

    cases = [
        FailureCase(id="c1", category="cat", question="q1", known_issue="i1"),
        FailureCase(id="c2", category="cat", question="q2", known_issue="i2"),
    ]
    seen: list[FailureCaseResult] = []

    results = run_failure_cases(cases, on_result=seen.append)

    assert [r.id for r in seen] == ["c1", "c2"]
    assert seen == results
