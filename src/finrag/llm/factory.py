"""Picks the configured LLM provider. This is the only place that knows
which concrete LLMClient implementation to construct -- everything else in
the codebase depends on the LLMClient interface, not on this function.
"""

from __future__ import annotations

from finrag.config.settings import Settings, get_settings
from finrag.llm.base import LLMClient


def get_llm_client(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()

    if settings.llm_provider == "groq":
        from finrag.llm.groq_client import GroqClient

        return GroqClient(api_key=settings.groq_api_key, model=settings.llm_model)

    if settings.llm_provider == "openai":
        # Intentionally not implemented yet -- see docs/PROJECT_PLAN.md, Day 5
        # (LLM evaluation: compare Groq vs. OpenAI, keep the better one).
        # Add an OpenAIClient in llm/openai_client.py satisfying LLMClient
        # and wire it in here when we get there.
        raise NotImplementedError("OpenAI provider not implemented yet")

    raise ValueError(f"Unknown LLM_PROVIDER: {settings.llm_provider!r}")
