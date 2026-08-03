"""The agent loop: give the LLM a set of deterministic tools, let it decide
whether to call any, execute the ones it asks for, feed the results back,
and return a final grounded answer.

Why `tool_choice="auto"` (the model decides) rather than always calling a
tool or never calling one: many questions ("summarize AAPL's trend last
month") are already well served by the retrieved narrative context and
don't need a tool call at all -- forcing one would add latency and a
chance for the model to invent arguments just to satisfy the requirement.
Questions asking for a precise figure over an exact range benefit from a
tool call. Letting the model choose, informed by the context already in
the prompt, is the standard "agentic RAG" pattern (course Module 1).
"""

from __future__ import annotations

import json
import logging

from finrag.agent.tools import (
    ToolError,
    compare_tickers,
    get_price_on_date,
    get_return,
    get_volatility,
)
from finrag.llm.base import LLMClient, Message

logger = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 3

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_price_on_date",
            "description": (
                "Get the exact OHLCV price for one ticker on one date. "
                "If the market was closed that day, returns the most recent prior trading day."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string", "description": "Ticker symbol, e.g. AAPL"},
                    "date_str": {"type": "string", "description": "Date as YYYY-MM-DD"},
                },
                "required": ["ticker", "date_str"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_return",
            "description": "Get the exact percentage return for one ticker between two dates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["ticker", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_volatility",
            "description": (
                "Get the exact annualized volatility (stdev of daily returns) "
                "for one ticker between two dates."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["ticker", "start_date", "end_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_tickers",
            "description": (
                "Compare the exact percentage return of two or more tickers "
                "over the same date range and report the best performer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["tickers", "start_date", "end_date"],
            },
        },
    },
]

_TOOL_FUNCTIONS = {
    "get_price_on_date": get_price_on_date,
    "get_return": get_return,
    "get_volatility": get_volatility,
    "compare_tickers": compare_tickers,
}


def _execute_tool(name: str, arguments: dict) -> dict:
    """Run one tool call and always return a JSON-able dict -- errors are
    reported back to the model as a normal tool result (`{"error": ...}`),
    not raised, so a bad ticker or malformed date doesn't crash the whole
    request. The model can then explain the limitation or try again.
    """
    func = _TOOL_FUNCTIONS.get(name)
    if func is None:
        return {"error": f"Unknown tool: {name}"}
    try:
        return func(**arguments)
    except ToolError as exc:
        return {"error": str(exc)}
    except TypeError as exc:
        return {"error": f"Invalid arguments for {name}: {exc}"}


def run_agent_loop(
    llm: LLMClient, messages: list[Message], tool_trace: list[dict] | None = None
) -> str:
    """Run up to MAX_TOOL_ROUNDS of tool calling, then return the final
    text answer. `messages` should already contain the system instructions
    and the user turn (with retrieved context) -- this function only
    appends tool-call/tool-result turns on top of what's passed in.

    `tool_trace`: optional, defaults to None -- every production call site
    (rag/pipeline.py) omits it and behavior is unchanged. When a caller
    passes a list, each executed tool call's name/arguments/result is
    appended to it, so the caller can inspect what the agent actually did
    after the fact. Added for eval/llm_eval.py, whose LLM judge previously
    had no way to see tool results and, per the Day 9 investigation
    (docs/learning/day09_learning.md), was penalizing correct tool-computed
    answers for not matching the static retrieved CONTEXT.
    """
    conversation = list(messages)

    for round_num in range(MAX_TOOL_ROUNDS):
        response = llm.complete(conversation, tools=TOOLS)

        if not response.tool_calls:
            return response.content or ""

        logger.info(
            "Round %d: model requested %d tool call(s): %s",
            round_num + 1,
            len(response.tool_calls),
            [tc.name for tc in response.tool_calls],
        )

        # Groq/OpenAI require the assistant's tool-call turn to be echoed
        # back verbatim before the tool results, so the model can match
        # results to the calls it made.
        conversation.append(
            {
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                    }
                    for tc in response.tool_calls
                ],
            }
        )

        for tool_call in response.tool_calls:
            result = _execute_tool(tool_call.name, tool_call.arguments)
            if tool_trace is not None:
                tool_trace.append(
                    {"name": tool_call.name, "arguments": tool_call.arguments, "result": result}
                )
            conversation.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    # Ran out of rounds without a plain-text answer. Ask one final time
    # without tools so the model is forced to summarize what it has.
    logger.warning("Hit MAX_TOOL_ROUNDS (%d) without a final answer", MAX_TOOL_ROUNDS)
    final = llm.complete(conversation)
    return final.content or ""
