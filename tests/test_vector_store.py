"""Integration tests for vector_store, run against a real Postgres+pgvector
instance (via `docker compose up -d` locally, or the `postgres` service
container in CI -- see .github/workflows/ci.yml).

These are skipped automatically if no database is reachable, so they don't
fail the suite on a machine that hasn't started the containers -- but they
DO run in CI, where a pgvector-enabled Postgres is always spun up
alongside the schema. Local-only unit tests (compute_stats, build_documents,
embeddings, llm client) never need this and are not skipped.
"""

from __future__ import annotations

from datetime import date

import psycopg
import pytest

from finrag.config.settings import get_settings
from finrag.ingestion.build_documents import DocumentRecord
from finrag.knowledge_base.vector_store import count_documents, get_connection, upsert_documents

TEST_DOC_ID = "TESTTICKER:weekly:2024-01-01"


@pytest.fixture
def settings():
    return get_settings()


@pytest.fixture(autouse=True)
def _skip_if_db_unreachable(settings):
    try:
        conn = psycopg.connect(settings.database_url, connect_timeout=2)
        conn.close()
    except psycopg.OperationalError:
        pytest.skip(
            "Postgres not reachable at "
            f"{settings.database_url} -- run `docker compose up -d` to enable this test"
        )


@pytest.fixture
def cleanup_test_doc(settings):
    yield
    with get_connection(settings) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM documents WHERE doc_id = %s", (TEST_DOC_ID,))
        conn.commit()


def _fake_record() -> DocumentRecord:
    return DocumentRecord(
        doc_id=TEST_DOC_ID,
        ticker="TESTTICKER",
        sector="Technology",
        granularity="weekly",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 1, 5),
        content="TESTTICKER rose 5% this week.",
    )


def test_upsert_then_count_increases(settings, cleanup_test_doc):
    with get_connection(settings) as conn:
        before = count_documents(conn)
        upsert_documents(conn, [_fake_record()], [[0.1] * 384])
        after = count_documents(conn)

    assert after == before + 1


def test_upsert_is_idempotent_by_doc_id(settings, cleanup_test_doc):
    record = _fake_record()
    with get_connection(settings) as conn:
        upsert_documents(conn, [record], [[0.1] * 384])
        first_count = count_documents(conn)

        # Re-upserting the same doc_id with different content should UPDATE,
        # not insert a second row.
        updated = _fake_record()
        updated.content = "TESTTICKER actually rose 6% this week."
        upsert_documents(conn, [updated], [[0.2] * 384])
        second_count = count_documents(conn)

        with conn.cursor() as cur:
            cur.execute("SELECT content FROM documents WHERE doc_id = %s", (TEST_DOC_ID,))
            stored_content = cur.fetchone()[0]

    assert first_count == second_count
    assert stored_content == "TESTTICKER actually rose 6% this week."


def test_upsert_raises_on_length_mismatch(settings):
    with get_connection(settings) as conn, pytest.raises(ValueError, match="length mismatch"):
        upsert_documents(conn, [_fake_record()], [])
