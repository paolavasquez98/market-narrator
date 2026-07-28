"""Postgres/pgvector data access for the `documents` table.

Loading is idempotent by design: `ON CONFLICT (doc_id) DO UPDATE` means the
ingestion orchestrator can be re-run any time the stats/narrative logic
changes, without manually truncating tables first -- the same idempotent
philosophy as `fetch_prices`' on-disk caching.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from finrag.config.settings import Settings, get_settings
from finrag.ingestion.build_documents import DocumentRecord


@contextmanager
def get_connection(settings: Settings | None = None):
    """Context-managed Postgres connection with the pgvector type adapter
    registered, so Python lists can be passed directly as `vector` values.
    """
    settings = settings or get_settings()
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    try:
        yield conn
    finally:
        conn.close()


def upsert_documents(
    conn: psycopg.Connection,
    records: Sequence[DocumentRecord],
    embeddings: Sequence[list[float]],
) -> None:
    """Insert or update documents by `doc_id`. `content_tsv` (full-text
    search) updates automatically since it's a generated column.
    """
    if len(records) != len(embeddings):
        raise ValueError(
            f"records ({len(records)}) and embeddings ({len(embeddings)}) length mismatch"
        )

    with conn.cursor() as cur:
        for record, embedding in zip(records, embeddings):
            cur.execute(
                """
                INSERT INTO documents
                    (doc_id, ticker, sector, granularity, period_start, period_end, content, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (doc_id) DO UPDATE SET
                    content = EXCLUDED.content,
                    embedding = EXCLUDED.embedding,
                    period_end = EXCLUDED.period_end
                """,
                (
                    record.doc_id,
                    record.ticker,
                    record.sector,
                    record.granularity,
                    record.period_start,
                    record.period_end,
                    record.content,
                    embedding,
                ),
            )
    conn.commit()


def count_documents(conn: psycopg.Connection) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        row = cur.fetchone()
        return int(row[0]) if row else 0
