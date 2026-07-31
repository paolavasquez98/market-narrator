"""Pydantic request/response models for the FastAPI service.

Kept in their own module, separate from routes.py, for the same reason
prompts live apart from pipeline.py (rag/prompts.py) and SQL lives apart
from orchestration (knowledge_base/): this file is the API's public
contract. Anyone integrating against this service -- the Streamlit UI
today, potentially something else later -- should be able to read one
short file and see exactly what every endpoint sends and receives,
without wading through retrieval/logging logic to find it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000, description="Natural-language question")
    top_k: int = Field(default=5, ge=1, le=20, description="Chunks kept after reranking")


class RetrievedChunk(BaseModel):
    """One reranked chunk that was in the LLM's prompt context, surfaced
    to the client so the UI can show "what the answer is grounded in" --
    important for a finance assistant where trusting the answer means
    being able to see the source.
    """

    doc_id: str
    ticker: str
    granularity: str
    period_start: str
    period_end: str
    score: float


class AskResponse(BaseModel):
    query_id: int = Field(
        description="Id of the logged query_logs row; pass this to POST /feedback/{query_id}. "
        "-1 if logging itself failed (the answer is still valid, just not loggable)."
    )
    question: str
    answer: str
    rewritten_query: str
    resolved_tickers: list[str]
    retrieved: list[RetrievedChunk]
    latency_ms: int


class FeedbackRequest(BaseModel):
    # Literal[1, -1] rather than a plain int with a custom validator --
    # pydantic rejects anything else automatically and it's self-documenting
    # in the generated OpenAPI schema (the only two values ever show up in
    # /docs), which a bare "must be 1 or -1" docstring wouldn't give you.
    feedback: Literal[1, -1] = Field(description="1 = thumbs up, -1 = thumbs down")


class FeedbackResponse(BaseModel):
    query_id: int
    recorded: bool


class TickerGroup(BaseModel):
    sector: str
    tickers: list[str]


class TickersResponse(BaseModel):
    tickers: list[TickerGroup]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    database: str
