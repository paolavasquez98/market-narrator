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
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            # tool_choice="auto" if tools else None,
            tool_choice="none",
            temperature=temperature,
        )
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
