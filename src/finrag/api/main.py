"""FastAPI application entrypoint.

Run locally with:
    uv run uvicorn finrag.api.main:app --reload

Or via docker-compose (see docker-compose.yml's `api` service and
Dockerfile.api at the repo root).
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from finrag.api.routes import router

app = FastAPI(
    title="Market Narrator API",
    description=(
        "Agentic RAG over historical stock market data. Ask a natural-language "
        "question, get an answer grounded in retrieved narrative documents and, "
        "for anything requiring an exact figure, deterministic tool calls."
    ),
    version="0.1.0",
)

# Wide-open CORS is a deliberate choice, not an oversight: this API has no
# authentication and sets no cookies, and its only client is the
# Streamlit UI (a different origin/port in docker-compose) plus ad-hoc
# developer scripts -- there's no session or credential a cross-origin
# request could steal here. Revisit if auth is ever added.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
