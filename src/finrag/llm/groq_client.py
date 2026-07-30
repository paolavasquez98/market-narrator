"""Groq implementation of the LLMClient interface."""

from __future__ import annotations

import json

from groq import Groq

from finrag.llm.base import LLMClient, LLMResponse, Message, ToolCall, ToolSchema


class GroqClient(LLMClient):
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise RuntimeError("Groq API key is missing (set GROQ_API_KEY in .env)")
        self._client = Groq(api_key=api_key)
        self._model = model

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        # Only include `tools`/`tool_choice` in the request at all when tools
        # were actually provided. Groq's Python SDK (like OpenAI's) treats an
        # explicitly-passed `None` as a real value distinct from "argument
        # not given" -- omitting the parameter is what actually excludes it
        # from the request body. See docs/learning/day03_learning.md,
        # section 13, for the full investigation.
        kwargs: dict[str, object] = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0].message

        tool_calls = [
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments),
            )
            for tc in (choice.tool_calls or [])
        ]

        return LLMResponse(content=choice.content, tool_calls=tool_calls, raw=response)
