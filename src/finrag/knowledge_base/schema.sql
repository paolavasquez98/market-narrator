-- Market Narrator database schema.
-- This is the single source of truth for the schema. It is mounted directly
-- into the Postgres container's docker-entrypoint-initdb.d, so it runs
-- automatically the first time the `db` volume is created.

CREATE EXTENSION IF NOT EXISTS vector;

-- Narrative documents produced by the ingestion pipeline: one row per
-- (ticker, granularity, period). `embedding` is populated during ingestion
-- using fastembed (BAAI/bge-small-en-v1.5, 384 dimensions). `content_tsv`
-- is a generated column so Postgres full-text search stays in sync with
-- `content` automatically -- no separate indexing step to forget.
CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    doc_id        TEXT UNIQUE NOT NULL,       -- e.g. "AAPL:weekly:2022-03-07"
    ticker        TEXT NOT NULL,
    sector        TEXT NOT NULL,
    granularity   TEXT NOT NULL,               -- 'weekly' | 'monthly' | 'yearly'
    period_start  DATE NOT NULL,
    period_end    DATE NOT NULL,
    content       TEXT NOT NULL,
    embedding     VECTOR(384),
    content_tsv   TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS documents_tsv_idx
    ON documents USING gin (content_tsv);

CREATE INDEX IF NOT EXISTS documents_ticker_idx
    ON documents (ticker);

-- One row per user query, used for both debugging the RAG pipeline and the
-- Grafana monitoring dashboard (Day 7).
CREATE TABLE IF NOT EXISTS query_logs (
    id                BIGSERIAL PRIMARY KEY,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    question          TEXT NOT NULL,
    rewritten_query   TEXT,
    extracted_tickers TEXT[],
    retrieval_method  TEXT,                    -- 'keyword' | 'vector' | 'hybrid' | 'hybrid_rerank'
    retrieved_doc_ids TEXT[],
    tool_calls        JSONB,
    model             TEXT,
    answer            TEXT,
    latency_ms        INTEGER,
    feedback          SMALLINT                 -- 1 = thumbs up, -1 = thumbs down, NULL = none
);

CREATE INDEX IF NOT EXISTS query_logs_created_at_idx ON query_logs (created_at);
