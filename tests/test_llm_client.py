"""Tests for the Groq LLMClient adapter, using a stubbed SDK client so no
network call (and no API key) is required. This tests the *parsing*
contract -- that we correctly turn a Groq SDK response into our
provider-agnostic LLMResponse -- not the model's actual behavior.
"""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from finrag.llm.groq_client import GroqClient


def _fake_groq_response(content=None, tool_calls=None):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(choices=[choice])


@patch("finrag.llm.groq_client.Groq")
def test_complete_returns_plain_text_response(mock_groq_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_groq_response(
        content="AAPL rose 5% in 2022."
    )
    mock_groq_cls.return_value = mock_client

    client = GroqClient(api_key="fake-key", model="llama-3.3-70b-versatile")
    response = client.complete([{"role": "user", "content": "How did AAPL do in 2022?"}])

    assert response.content == "AAPL rose 5% in 2022."
    assert response.tool_calls == []


@patch("finrag.llm.groq_client.Groq")
def test_complete_parses_tool_calls(mock_groq_cls):
    fake_tool_call = SimpleNamespace(
        id="call_1",
        function=SimpleNamespace(
            name="get_return",
            arguments=json.dumps({"ticker": "AAPL", "start": "2022-01-01", "end": "2022-12-31"}),
        ),
    )
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_groq_response(
        content=None, tool_calls=[fake_tool_call]
    )
    mock_groq_cls.return_value = mock_client

    client = GroqClient(api_key="fake-key", model="llama-3.3-70b-versatile")
    response = client.complete(
        [{"role": "user", "content": "What was AAPL's return in 2022?"}],
        tools=[{"type": "function", "function": {"name": "get_return"}}],
    )

    assert response.content is None
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.name == "get_return"
    assert call.arguments == {"ticker": "AAPL", "start": "2022-01-01", "end": "2022-12-31"}
