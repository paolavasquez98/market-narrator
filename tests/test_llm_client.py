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
def test_complete_without_tools_omits_tools_and_tool_choice_entirely(mock_groq_cls):
    """Regression test for the `tool_choice=None` bug: Groq's API rejects a
    literal JSON `null` for `tool_choice` (`Only allowed string values for
    'tool_choice' are [none, auto, required]`), and the SDK only strips
    parameters that are truly *omitted* -- an explicit `None` is still sent
    as `null`. So when no tools are given, `tools`/`tool_choice` must not
    appear as keyword arguments at all, not just be set to `None`.
    """
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_groq_response(content="ok")
    mock_groq_cls.return_value = mock_client

    client = GroqClient(api_key="fake-key", model="llama-3.3-70b-versatile")
    client.complete([{"role": "user", "content": "hello"}])

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "tools" not in call_kwargs
    assert "tool_choice" not in call_kwargs


@patch("finrag.llm.groq_client.Groq")
def test_complete_with_tools_sets_tool_choice_auto(mock_groq_cls):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _fake_groq_response(content="ok")
    mock_groq_cls.return_value = mock_client

    client = GroqClient(api_key="fake-key", model="llama-3.3-70b-versatile")
    tools = [{"type": "function", "function": {"name": "get_return"}}]
    client.complete([{"role": "user", "content": "hello"}], tools=tools)

    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["tools"] == tools
    assert call_kwargs["tool_choice"] == "auto"


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
