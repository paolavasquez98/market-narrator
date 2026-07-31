"""API routes: ask a question, submit feedback, health check, ticker list.

Each route stays thin -- the real work (retrieval, reranking, tool
calling) already lives in rag/pipeline.py and agent/orchestrator.py; this
module's job is only to translate an HTTP request into a call to that
existing code, shape the result into api/models.py's response schemas,
and handle the one thing that's genuinely an API-layer concern: logging
each interaction via monitoring/logger.py (see that module's docstring
for why logging lives at this layer and not inside answer_question()
itself).
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from finrag.api.models import (
    AskRequest,
    AskResponse,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    RetrievedChunk,
    TickerGroup,
    TickersResponse,
)
from finrag.config.settings import get_settings
from finrag.config.tickers import TICKER_UNIVERSE
from finrag.knowledge_base.vector_store import get_connection
from finrag.monitoring.logger import log_query, record_feedback
from finrag.rag.pipeline import answer_question

logger = logging.getLogger(__name__)

router = APIRouter()

# The pipeline always runs the full rewrite -> hybrid search -> rerank
# flow today (see rag/pipeline.py's module docstring: "today it's always
# hybrid+rerank"). Logged as a constant rather than threaded through
# answer_question() since there's currently exactly one retrieval
# configuration wired into the live app -- eval/retrieval_eval.py is
# where the alternatives get compared, not this service.
RETRIEVAL_METHOD = "hybrid_rerank_rewrite"

# Sentinel query_id returned when logging itself failed -- the answer is
# still valid and returned to the client, there's just no row a later
# feedback call could attach to.
_UNLOGGED_QUERY_ID = -1


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    settings = get_settings()

    start = time.perf_counter()
    try:
        result = answer_question(request.question, top_k=request.top_k)
    except Exception as exc:
        logger.exception("answer_question failed for %r", request.question)
        raise HTTPException(
            status_code=502, detail="Failed to generate an answer. Please try again."
        ) from exc
    latency_ms = int((time.perf_counter() - start) * 1000)

    try:
        with get_connection(settings) as conn:
            query_id = log_query(
                conn,
                result,
                retrieval_method=RETRIEVAL_METHOD,
                model=settings.llm_model,
                latency_ms=latency_ms,
            )
    except Exception:
        # A logging failure shouldn't take down an otherwise-successful
        # answer -- the user still gets their response, just without a
        # query_id to attach feedback to later. Monitoring should be
        # observability, not a new way for the service to go down.
        logger.exception("Failed to log query to query_logs; continuing without a query_id")
        query_id = _UNLOGGED_QUERY_ID

    return AskResponse(
        query_id=query_id,
        question=result.question,
        answer=result.answer,
        rewritten_query=result.rewritten_query,
        resolved_tickers=result.resolved_tickers,
        retrieved=[
            RetrievedChunk(
                doc_id=r.doc_id,
                ticker=r.ticker,
                granularity=r.granularity,
                period_start=r.period_start.isoformat(),
                period_end=r.period_end.isoformat(),
                score=r.score,
            )
            for r in result.retrieved
        ],
        latency_ms=latency_ms,
    )


@router.post("/feedback/{query_id}", response_model=FeedbackResponse)
def feedback(query_id: int, request: FeedbackRequest) -> FeedbackResponse:
    if query_id == _UNLOGGED_QUERY_ID:
        raise HTTPException(status_code=404, detail="This query was never logged; no id to attach feedback to")

    settings = get_settings()
    with get_connection(settings) as conn:
        recorded = record_feedback(conn, query_id, request.feedback)

    if not recorded:
        raise HTTPException(status_code=404, detail=f"No query_log row with id={query_id}")

    return FeedbackResponse(query_id=query_id, recorded=True)


@router.get("/tickers", response_model=TickersResponse)
def tickers() -> TickersResponse:
    """Static, no DB/LLM call -- lets the UI populate a ticker reference
    list (or validate a user's typed ticker) without a round trip to
    anything but this process's own memory.
    """
    return TickersResponse(
        tickers=[
            TickerGroup(sector=sector, tickers=sorted(symbols))
            for sector, symbols in sorted(TICKER_UNIVERSE.items())
        ]
    )


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness/readiness check for container orchestration and uptime
    monitoring. Deliberately does NOT call Groq: this endpoint needs to be
    cheap and safe to call frequently, and burning LLM API quota (or
    adding a multi-second round trip) on every health probe would be a
    poor trade for a signal that mostly duplicates what `/ask` failing
    would already tell you. The one dependency worth checking directly is
    the database, since a dead DB connection is a failure mode `/ask`
    can't work around at all.
    """
    settings = get_settings()
    try:
        with get_connection(settings) as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        database_status = "ok"
    except Exception:
        logger.exception("Health check: database unreachable")
        database_status = "unreachable"

    return HealthResponse(
        status="ok" if database_status == "ok" else "degraded",
        database=database_status,
    )
