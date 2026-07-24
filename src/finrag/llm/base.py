"""Provider-agnostic LLM client interface.

Why this exists: Groq and OpenAI's chat-completion APIs are almost
identical (Groq's SDK is deliberately OpenAI-compatible), but "almost"
is the trap -- call a vendor SDK directly from the RAG pipeline and agent
orchestrator, and swapping providers later (which we need to do for the
LLM-evaluation criterion: compare multiple models, pick the best) means
hunting down every call site.

Instead, `rag/`, `agent/`, and `retrieval/` code depends only on the
`LLMClient` interface below. `GroqClient` (llm/groq_client.py) is the only
implementation today; an `OpenAIClient` can be added later as a new class
that satisfies the same interface, with zero changes anywhere else.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# A chat message in OpenAI/Groq wire format, e.g. {"role": "user", "content": "..."}.
Message = dict[str, Any]

# A tool/function schema in OpenAI/Groq wire format. Defined and owned by
# whoever calls the LLM (the agent orchestrator), not by this module --
# this module only needs to pass it through.
ToolSchema = dict[str, Any]


@dataclass
class ToolCall:
    """A single tool invocation the model asked for."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Normalized response from any provider."""

    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # underlying SDK response object, for debugging only


class LLMClient(ABC):
    """Interface every LLM provider adapter must implement."""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send a chat completion request and return a normalized response.

        Args:
            messages: Conversation so far, in OpenAI/Groq message format.
            tools: Optional tool schemas the model may call.
            temperature: Sampling temperature. Defaults to 0 for reproducible,
                gradeable answers -- this is a RAG app answering factual
                questions about price history, not a creative-writing tool.
        """
        raise NotImplementedError
