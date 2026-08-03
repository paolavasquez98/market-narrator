"""Tests for llm_judge.judge_answers, using a scripted LLMClient (no real
API call). These only assert our parsing/error-handling around the judge's
response -- not whether a real model's judgments are actually good (that's
a methodology question documented in docs/learning/day05_learning.md, not
something a unit test can verify).
"""

from __future__ import annotations

import json

from finrag.eval.llm_judge import _JUDGE_INSTRUCTIONS, judge_answers
from finrag.llm.base import LLMClient, LLMResponse


class _FakeLLM(LLMClient):
    def __init__(self, content: str | None):
        self._content = content
        self.calls: list[list] = []

    def complete(self, messages, tools=None, temperature=0.0):
        self.calls.append(list(messages))
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


# Day 9: the first real evaluation run showed the judge penalizing the
# with-tools pipeline for producing exact figures not present in
# CONTEXT -- exactly what agent/tools.py exists to do for date ranges
# CONTEXT was never meant to cover (see rag/prompts.py's
# SYSTEM_INSTRUCTIONS). The instructions text was fixed to stop treating
# "not in CONTEXT" as synonymous with "invented"; this guards that fix
# stays in place. Can't verify a real judge actually behaves better --
# only that the instruction text still says what it's supposed to.
def test_judge_instructions_do_not_treat_absence_from_context_as_invented():
    lowered = _JUDGE_INSTRUCTIONS.lower()
    assert "not automatically" in lowered
    assert "tool" in lowered


# Day 9, second half of the fix: pass the with-tools answer's actual tool
# calls to the judge (see eval/llm_eval.py's _format_tool_trace and
# orchestrator.run_agent_loop's tool_trace param) so it can verify a
# number directly instead of just being told to be lenient about it.
def test_judge_answers_includes_tool_results_section_when_provided():
    fake_response = json.dumps({"answer_a_score": 3, "answer_b_score": 3, "reasoning": "r"})
    llm = _FakeLLM(fake_response)

    judge_answers(
        llm, "q", "ctx", "answer a", "answer b",
        tool_results="get_return(ticker=AAPL) -> {'return_pct': 5.0}",
    )

    user_prompt = llm.calls[0][1]["content"]
    assert "TOOL RESULTS" in user_prompt
    assert "get_return(ticker=AAPL)" in user_prompt


def test_judge_answers_omits_tool_results_section_when_not_provided():
    fake_response = json.dumps({"answer_a_score": 3, "answer_b_score": 3, "reasoning": "r"})
    llm = _FakeLLM(fake_response)

    judge_answers(llm, "q", "ctx", "answer a", "answer b")

    user_prompt = llm.calls[0][1]["content"]
    assert "TOOL RESULTS" not in user_prompt
