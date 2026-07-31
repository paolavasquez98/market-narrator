"""Query/feedback logging into the `query_logs` table (schema.sql).

This is deliberately a thin, separate module from rag/pipeline.py rather
than logging being baked into `answer_question()` itself: logging is a
concern of the *interactive* surfaces (the API, eventually the CLI) that
serve real users, not of every caller of the pipeline -- the evaluation
scripts in eval/ call `answer_question()` and `run_agent_loop()` directly
thousands of times each run, and would flood `query_logs` with synthetic
traffic if logging were unconditional inside the pipeline itself. Keeping
it a separate, explicit call also means a future caller (e.g. a CLI) can
choose whether to log without touching pipeline internals.

This is what feeds the Grafana dashboard (monitoring/grafana/) -- every
column here maps to a panel: query volume over time, latency
distribution, tool-call frequency, ticker popularity, and the
thumbs-up/thumbs-down feedback ratio.
"""

from __future__ import annotations

import json

import psycopg

from finrag.rag.pipeline import RagAnswer


def log_query(
    conn: psycopg.Connection,
    answer: RagAnswer,
    *,
    retrieval_method: str,
    model: str,
    tool_calls: list[dict] | None = None,
    latency_ms: int | None = None,
) -> int:
    """Insert one row for a completed question, returning its `id` so the
    caller (the API) can hand it back to the client for a later feedback
    call -- feedback arrives in a separate request, after the answer's
    already been shown to the user, so the row has to be addressable by
    id rather than re-matched by question text.
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO query_logs
                (question, rewritten_query, extracted_tickers, retrieval_method,
                 retrieved_doc_ids, tool_calls, model, answer, latency_ms)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (
                answer.question,
                answer.rewritten_query,
                answer.resolved_tickers,
                retrieval_method,
                [r.doc_id for r in answer.retrieved],
                json.dumps(tool_calls or []),
                model,
                answer.answer,
                latency_ms,
            ),
        )
        row = cur.fetchone()
    conn.commit()
    return int(row[0])


def record_feedback(conn: psycopg.Connection, query_id: int, feedback: int) -> bool:
    """Set feedback (1 = thumbs up, -1 = thumbs down) on an existing
    query_logs row. Returns False (rather than raising) if `query_id`
    doesn't exist -- a stale/invalid id from a client shouldn't be a
    server error, just a no-op the caller can turn into a 404.
    """
    if feedback not in (1, -1):
        raise ValueError(f"feedback must be 1 or -1, got {feedback!r}")

    with conn.cursor() as cur:
        cur.execute(
            "UPDATE query_logs SET feedback = %s WHERE id = %s",
            (feedback, query_id),
        )
        updated = cur.rowcount
    conn.commit()
    return updated > 0
