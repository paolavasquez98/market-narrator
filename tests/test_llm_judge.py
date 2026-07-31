"""Tests for llm_judge.judge_answers, using a scripted LLMClient (no real
API call). These only assert our parsing/error-handling around the judge's
response -- not whether a real model's judgments are actually good (that's
a methodology question documented in docs/learning/day05_learning.md, not
something a unit test can verify).
"""

from __future__ import annotations

import json

from finrag.eval.llm_judge import judge_answers
from finrag.llm.base import LLMClient, LLMResponse


class _FakeLLM(LLMClient):
    def __init__(self, content: str | None):
        self._content = content

    def complete(self, messages, tools=None, temperature=0.0):
        return LLMResponse(content=self._content, tool_calls=[])


def test_judge_answers_parses_valid_json():
    fake_response = json.dumps(
        {"answer_a_score": 4, "answer_b_score": 2, "reasoning": "A cites the context, B invents a number."}
    )
    verdict = judge_answers(_FakeLLM(fake_response), "q", "ctx", "answer a", "answer b")

    assert verdict is not None
    assert verdict.answer_a_score == 4
    assert verdict.answer_b_score == 2
    assert verdict.reasoning == "A cites the context, B invents a number."


def test_judge_answers_returns_none_on_invalid_json():
    verdict = judge_answers(_FakeLLM("not json"), "q", "ctx", "answer a", "answer b")

    assert verdict is None


def test_judge_answers_returns_none_when_content_is_none():
    verdict = judge_answers(_FakeLLM(None), "q", "ctx", "answer a", "answer b")

    assert verdict is None


def test_judge_answers_returns_none_when_score_key_missing():
    fake_response = json.dumps({"answer_a_score": 4, "reasoning": "missing answer_b_score"})
    verdict = judge_answers(_FakeLLM(fake_response), "q", "ctx", "answer a", "answer b")

    assert verdict is None


def test_judge_answers_returns_none_when_score_is_not_coercible_to_int():
    fake_response = json.dumps({"answer_a_score": "great", "answer_b_score": 2, "reasoning": "x"})
    verdict = judge_answers(_FakeLLM(fake_response), "q", "ctx", "answer a", "answer b")

    assert verdict is None


def test_judge_answers_defaults_missing_reasoning_to_empty_string():
    fake_response = json.dumps({"answer_a_score": 5, "answer_b_score": 5})
    verdict = judge_answers(_FakeLLM(fake_response), "q", "ctx", "answer a", "answer b")

    assert verdict is not None
    assert verdict.reasoning == ""
