"""Tests for query_rewriter, using a fake LLMClient (no real API call) so
these tests assert only our parsing/validation logic -- not whether a real
model actually resolves company names correctly (that's an LLM-evaluation
question for Day 5, not a unit-test question).
"""

from __future__ import annotations

import json

from finrag.llm.base import LLMClient, LLMResponse
from finrag.retrieval.query_rewriter import _build_instructions, rewrite_query


class _FakeLLM(LLMClient):
    def __init__(self, content: str | None):
        self._content = content

    def complete(self, messages, tools=None, temperature=0.0):
        return LLMResponse(content=self._content, tool_calls=[])


def test_rewrite_query_parses_valid_json():
    fake_response = json.dumps({"tickers": ["NVDA"], "rewritten_query": "Nvidia stock performance"})
    intent = rewrite_query(_FakeLLM(fake_response), "How did Nvidia do?")

    assert intent.tickers == ["NVDA"]
    assert intent.rewritten_query == "Nvidia stock performance"


def test_rewrite_query_filters_out_hallucinated_tickers():
    fake_response = json.dumps({"tickers": ["NVDA", "NOT_REAL_TICKER"], "rewritten_query": "q"})
    intent = rewrite_query(_FakeLLM(fake_response), "q")

    assert intent.tickers == ["NVDA"]


def test_rewrite_query_falls_back_on_invalid_json():
    intent = rewrite_query(_FakeLLM("this is not json"), "How did AAPL do?")

    assert intent.tickers == []
    assert intent.rewritten_query == "How did AAPL do?"


def test_rewrite_query_falls_back_when_content_is_none():
    intent = rewrite_query(_FakeLLM(None), "How did AAPL do?")

    assert intent.tickers == []
    assert intent.rewritten_query == "How did AAPL do?"


def test_rewrite_query_falls_back_when_rewritten_query_is_wrong_type():
    fake_response = json.dumps({"tickers": [], "rewritten_query": 12345})
    intent = rewrite_query(_FakeLLM(fake_response), "original question")

    assert intent.tickers == []
    assert intent.rewritten_query == "original question"


def test_rewrite_query_defaults_missing_keys_gracefully():
    intent = rewrite_query(_FakeLLM(json.dumps({})), "original question")

    assert intent.tickers == []
    assert intent.rewritten_query == "original question"


# Day 6: eval/evaluate_retrieval.py showed hybrid_rerank_rewrite scoring
# worse than hybrid_rerank -- traced to the rewrite prompt having no
# instruction to preserve exact dates, which are the main signal that
# distinguishes one period's document from another for the same ticker
# (see the module docstring's "Day 6 finding"). This just guards that the
# instruction stays in the prompt; it can't verify a real model obeys it.
def test_rewrite_instructions_tell_model_to_preserve_dates():
    instructions = _build_instructions()
    assert "preserve" in instructions.lower()
    assert "date" in instructions.lower()
