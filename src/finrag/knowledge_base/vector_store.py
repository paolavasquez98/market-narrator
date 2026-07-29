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
from finrag.knowledge_base.models import DOCUMENT_COLUMNS, SearchResult, row_to_search_result


@contextmanager
def get_connection(settings: Settings | None = None):
    """Context-managed Postgres connection with the pgvector type adapter
    registered.

    `register_vector(conn)` teaches this connection how to read `vector`
    columns back as results, and how to send `pgvector.Vector` / numpy
    array *parameters* as the `vector` wire type. It does **not** change
    how plain Python `list` parameters are sent -- see the `::vector`
    casts in `vector_search()` and `upsert_documents()` below for why
    that distinction matters.
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
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
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


def vector_search(
    conn: psycopg.Connection,
    query_embedding: list[float],
    top_k: int = 10,
    tickers: Sequence[str] | None = None,
) -> list[SearchResult]:
    """Semantic search: nearest documents to `query_embedding` by cosine
    similarity, using the HNSW index on `documents.embedding`.

    pgvector's `<=>` operator computes cosine *distance* (0 = identical,
    2 = opposite) for the `vector_cosine_ops` index we built in schema.sql,
    so we order by it ascending (closest first) and report
    `1 - distance` as `score` (higher = more similar), for consistency
    with keyword_search's "higher score is better" convention.

    `query_embedding` is cast explicitly to `::vector` in the SQL below.
    Without it, psycopg dumps a plain Python `list[float]` as a
    `double precision[]` parameter (its default array adapter -- pgvector's
    `register_vector()` only overrides dumping for `numpy.ndarray` and its
    own `Vector` wrapper, not built-in `list`), and `<=>` has no overload
    for `vector <=> double precision[]`.
    """
    where_clause = "WHERE ticker = ANY(%(tickers)s)" if tickers else ""
    query = f"""
        SELECT {DOCUMENT_COLUMNS}, 1 - (embedding <=> %(query_embedding)s::vector) AS score
        FROM documents
        {where_clause}
        ORDER BY embedding <=> %(query_embedding)s::vector
        LIMIT %(top_k)s
    """
    params = {"query_embedding": query_embedding, "top_k": top_k, "tickers": tickers}

    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()

    return [row_to_search_result(r) for r in rows]
