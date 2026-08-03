"""Tests for the agent tool-calling loop, using a scripted fake LLMClient
that returns pre-programmed responses in sequence -- this lets us drive
every branch (no tool call, one tool call round, a tool erroring out,
exhausting MAX_TOOL_ROUNDS) deterministically, without a real API call.
"""

from __future__ import annotations

import json

import finrag.agent.orchestrator as orch
from finrag.agent.tools import ToolError
from finrag.llm.base import LLMResponse, ToolCall


class _ScriptedLLM:
    """Returns each response in `responses`, in order, one per `.complete()`
    call. Records every call's arguments so tests can assert on them.
    """

    def __init__(self, responses: list[LLMResponse]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def complete(self, messages, tools=None, temperature=0.0):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self._responses.pop(0)


def _base_messages():
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "How did AAPL do in 2022?"},
    ]


def test_returns_immediately_when_no_tool_call_needed():
    llm = _ScriptedLLM([LLMResponse(content="AAPL rose 5% in 2022.", tool_calls=[])])

    answer = orch.run_agent_loop(llm, _base_messages())

    assert answer == "AAPL rose 5% in 2022."
    assert len(llm.calls) == 1
    assert llm.calls[0]["tools"] == orch.TOOLS


def test_executes_a_tool_call_and_returns_the_final_answer(monkeypatch):
    fake_result = {"ticker": "AAPL", "return_pct": 5.0}
    monkeypatch.setitem(orch._TOOL_FUNCTIONS, "get_return", lambda **kwargs: fake_result)

    tool_call = ToolCall(
        id="call_1",
        name="get_return",
        arguments={"ticker": "AAPL", "start_date": "2022-01-01", "end_date": "2022-12-31"},
    )
    llm = _ScriptedLLM(
        [
            LLMResponse(content=None, tool_calls=[tool_call]),
            LLMResponse(content="AAPL returned 5% in 2022.", tool_calls=[]),
        ]
    )

    answer = orch.run_agent_loop(llm, _base_messages())

    assert answer == "AAPL returned 5% in 2022."
    assert len(llm.calls) == 2

    second_call_messages = llm.calls[1]["messages"]
    assert second_call_messages[-2]["role"] == "assistant"
    assert second_call_messages[-2]["tool_calls"][0]["function"]["name"] == "get_return"
    assert second_call_messages[-1]["role"] == "tool"
    assert second_call_messages[-1]["tool_call_id"] == "call_1"
    assert json.loads(second_call_messages[-1]["content"]) == fake_result


def test_tool_error_is_reported_back_to_the_model_not_raised(monkeypatch):
    def _raise(**kwargs):
        raise ToolError("bad ticker")

    monkeypatch.setitem(orch._TOOL_FUNCTIONS, "get_return", _raise)

    tool_call = ToolCall(
        id="call_1", name="get_return", arguments={"ticker": "BAD", "start_date": "x", "end_date": "y"}
    )
    llm = _ScriptedLLM(
        [
            LLMResponse(content=None, tool_calls=[tool_call]),
            LLMResponse(content="I couldn't find that ticker.", tool_calls=[]),
        ]
    )

    answer = orch.run_agent_loop(llm, _base_messages())

    assert answer == "I couldn't find that ticker."
    tool_result_message = llm.calls[1]["messages"][-1]
    assert json.loads(tool_result_message["content"]) == {"error": "bad ticker"}


# Day 9: eval/llm_eval.py needs to see what tools the agent actually
# called, to show the LLM judge real computed values instead of asking it
# to trust an unverifiable number (see docs/learning/day09_learning.md).
# `tool_trace` is opt-in via a list argument; every other test in this
# file omits it and exercises the (unchanged) default behavior.
def test_tool_trace_captures_name_arguments_and_result_when_provided(monkeypatch):
    fake_result = {"ticker": "AAPL", "return_pct": 5.0}
    monkeypatch.setitem(orch._TOOL_FUNCTIONS, "get_return", lambda **kwargs: fake_result)

    tool_call = ToolCall(
        id="call_1",
        name="get_return",
        arguments={"ticker": "AAPL", "start_date": "2022-01-01", "end_date": "2022-12-31"},
    )
    llm = _ScriptedLLM(
        [
            LLMResponse(content=None, tool_calls=[tool_call]),
            LLMResponse(content="AAPL returned 5% in 2022.", tool_calls=[]),
        ]
    )

    tool_trace: list[dict] = []
    answer = orch.run_agent_loop(llm, _base_messages(), tool_trace=tool_trace)

    assert answer == "AAPL returned 5% in 2022."
    assert tool_trace == [
        {
            "name": "get_return",
            "arguments": {"ticker": "AAPL", "start_date": "2022-01-01", "end_date": "2022-12-31"},
            "result": fake_result,
        }
    ]


def test_tool_trace_stays_empty_when_no_tool_is_called():
    llm = _ScriptedLLM([LLMResponse(content="AAPL rose 5% in 2022.", tool_calls=[])])

    tool_trace: list[dict] = []
    orch.run_agent_loop(llm, _base_messages(), tool_trace=tool_trace)

    assert tool_trace == []


def test_execute_tool_handles_unknown_tool_name():
    result = orch._execute_tool("not_a_real_tool", {})
    assert result == {"error": "Unknown tool: not_a_real_tool"}


def test_execute_tool_handles_wrong_arguments(monkeypatch):
    monkeypatch.setitem(orch._TOOL_FUNCTIONS, "get_return", lambda ticker, start_date, end_date: {})

    result = orch._execute_tool("get_return", {"ticker": "AAPL"})  # missing required args

    assert "Invalid arguments" in result["error"]


def test_stops_after_max_rounds_and_forces_a_final_plain_answer(monkeypatch):
    monkeypatch.setitem(orch._TOOL_FUNCTIONS, "get_return", lambda **kwargs: {"ok": True})

    tool_call = ToolCall(
        id="call_1",
        name="get_return",
        arguments={"ticker": "AAPL", "start_date": "2022-01-01", "end_date": "2022-12-31"},
    )
    # The model keeps asking for tool calls every round...
    responses = [LLMResponse(content=None, tool_calls=[tool_call]) for _ in range(orch.MAX_TOOL_ROUNDS)]
    # ...until the loop gives up and asks one final time without tools.
    responses.append(LLMResponse(content="Best answer I can give.", tool_calls=[]))
    llm = _ScriptedLLM(responses)

    answer = orch.run_agent_loop(llm, _base_messages())

    assert answer == "Best answer I can give."
    assert len(llm.calls) == orch.MAX_TOOL_ROUNDS + 1
    assert llm.calls[-1]["tools"] is None
