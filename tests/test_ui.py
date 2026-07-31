"""Tests for the pure-logic parts of ui/app.py.

Streamlit scripts run inside a script-runner context that's awkward to
fully unit test (no `streamlit` package equivalent of FastAPI's
TestClient is used here) -- what's covered instead is the one thing in
this file that's genuinely just Python logic independent of Streamlit's
rendering: `_api_base_url()` reading from Settings, and `_call_api()`'s
error handling for the three failure modes a user is actually likely to
hit (API not running, API returned an error status, request timed out).
Actually clicking through the UI still needs `streamlit run ui/app.py`
against a live API -- see docs/learning for what to verify locally.
"""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ui"))
import app as ui_app


def test_api_base_url_defaults_to_localhost():
    assert ui_app._api_base_url() == "http://localhost:8000"


def test_call_api_returns_none_and_shows_error_on_connect_failure(monkeypatch):
    class _RaisingClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def request(self, *a, **k):
            raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(ui_app.httpx, "Client", _RaisingClient)
    errors = []
    monkeypatch.setattr(ui_app.st, "error", lambda msg: errors.append(msg))

    result = ui_app._call_api("GET", "/tickers")

    assert result is None
    assert len(errors) == 1
    assert "Can't reach" in errors[0]


def test_call_api_returns_none_on_timeout(monkeypatch):
    class _TimingOutClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def request(self, *a, **k):
            raise httpx.TimeoutException("timed out")

    monkeypatch.setattr(ui_app.httpx, "Client", _TimingOutClient)
    errors = []
    monkeypatch.setattr(ui_app.st, "error", lambda msg: errors.append(msg))

    result = ui_app._call_api("POST", "/ask", json={"question": "q"})

    assert result is None
    assert "timed out" in errors[0].lower()


def test_call_api_returns_response_on_success(monkeypatch):
    class _FakeResponse:
        def raise_for_status(self):
            pass

    class _OkClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def request(self, *a, **k):
            return _FakeResponse()

    monkeypatch.setattr(ui_app.httpx, "Client", _OkClient)

    result = ui_app._call_api("GET", "/tickers")

    assert isinstance(result, _FakeResponse)


def test_example_questions_is_non_empty():
    assert len(ui_app.EXAMPLE_QUESTIONS) > 0
    assert all(isinstance(q, str) and q.strip() for q in ui_app.EXAMPLE_QUESTIONS)


if __name__ == "__main__":
    pytest.main([__file__])
