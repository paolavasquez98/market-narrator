"""Tests for eval/ground_truth.py. `build_ground_truth`'s assembly/error-
handling logic is tested with a mocked `sample_documents` (no DB needed).
`sample_documents` itself needs a real Postgres to test against (real SQL,
real GROUP-BY-like stratification), so that one integration test follows
the same skip-if-unreachable pattern as the other DB-backed tests.
"""

from __future__ import annotations

from collections import Counter
from datetime import date

import finrag.eval.ground_truth as gt_module
from finrag.eval.ground_truth import build_ground_truth, sample_documents
from finrag.knowledge_base.vector_store import get_connection
from finrag.llm.base import LLMClient, LLMResponse


class _ScriptedLLM(LLMClient):
    """Returns each response in order; raises if asked for more than given."""

    def __init__(self, contents: list[str | None]):
        self._contents = list(contents)

    def complete(self, messages, tools=None, temperature=0.0):
        content = self._contents.pop(0)
        if content is _RAISE:
            raise RuntimeError("simulated API failure")
        return LLMResponse(content=content, tool_calls=[])


_RAISE = object()


def _fake_sampled_docs(n: int) -> list[dict]:
    return [
        {
            "doc_id": f"AAPL:weekly:2024-01-{i:02d}",
            "ticker": "AAPL",
            "granularity": "weekly",
            "period_start": date(2024, 1, i),
            "period_end": date(2024, 1, i + 4),
            "content": f"AAPL rose this week, document {i}.",
        }
        for i in range(1, n + 1)
    ]


def test_build_ground_truth_assembles_one_row_per_successful_generation(monkeypatch):
    monkeypatch.setattr(gt_module, "sample_documents", lambda conn, per_ticker=None: _fake_sampled_docs(2))

    llm = _ScriptedLLM(["What was AAPL's return this week?", "How did AAPL trade this week?"])
    df = build_ground_truth(llm, conn=None)

    assert len(df) == 2
    assert list(df.columns) == [
        "question",
        "doc_id",
        "ticker",
        "granularity",
        "period_start",
        "period_end",
    ]
    assert df.iloc[0]["question"] == "What was AAPL's return this week?"
    assert df.iloc[0]["doc_id"] == "AAPL:weekly:2024-01-01"


def test_build_ground_truth_drops_documents_where_generation_failed(monkeypatch):
    monkeypatch.setattr(gt_module, "sample_documents", lambda conn, per_ticker=None: _fake_sampled_docs(3))

    # Second document's generation raises; it should be skipped, not crash the batch.
    llm = _ScriptedLLM(["question one", _RAISE, "question three"])
    df = build_ground_truth(llm, conn=None)

    assert len(df) == 2
    assert set(df["doc_id"]) == {"AAPL:weekly:2024-01-01", "AAPL:weekly:2024-01-03"}


def test_build_ground_truth_drops_documents_with_empty_question(monkeypatch):
    monkeypatch.setattr(gt_module, "sample_documents", lambda conn, per_ticker=None: _fake_sampled_docs(2))

    llm = _ScriptedLLM(["", "a real question"])
    df = build_ground_truth(llm, conn=None)

    assert len(df) == 1
    assert df.iloc[0]["doc_id"] == "AAPL:weekly:2024-01-02"


# --- Integration test: real stratified SQL sampling against a real DB ---
#
# Deliberately doesn't insert its own fixture data: this checks two
# properties that must hold no matter what's actually in `documents` --
# no (ticker, granularity) group ever exceeds its requested count, and the
# same seed always produces the same sample. On a fully-ingested dev DB
# this exercises real stratification across real data; in CI (schema
# only, no ingestion -- no Yahoo Finance network access there) it exercises
# the same code against an empty table and both properties still hold
# trivially. Either way, no unit test was going to catch a real SQL bug
# in the GROUP-BY-like per-ticker/per-granularity query -- only an actual
# database can.


def test_sample_documents_respects_per_ticker_cap_and_is_reproducible(settings, skip_if_db_unreachable):
    requested = {"weekly": 2, "monthly": 2, "yearly": 1}

    with get_connection(settings) as conn:
        first = sample_documents(conn, per_ticker=requested)
        second = sample_documents(conn, per_ticker=requested)

    counts = Counter((d["ticker"], d["granularity"]) for d in first)
    for (ticker, granularity), count in counts.items():
        assert count <= requested[granularity], f"{ticker}/{granularity} returned {count}"

    assert [d["doc_id"] for d in first] == [d["doc_id"] for d in second]
