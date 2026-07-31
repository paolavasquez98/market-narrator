# Market Narrator

An agentic RAG application for exploring historical stock market behavior. Ask questions like *"How did Apple perform in 2022?"* or *"Compare Microsoft and Nvidia over the last five years"* and get answers grounded in real historical price data, not the LLM's memory.

> **Status: work in progress.** Built for the [DataTalks.Club LLM Zoomcamp](https://github.com/DataTalksClub/llm-zoomcamp) capstone project. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full design and build roadmap, and [`docs/deployment.md`](docs/deployment.md) for how to run the full stack.

## What this is

Raw daily price data isn't itself a knowledge base — it's a table of numbers. This project turns [`yfinance`](https://github.com/ranaroussi/yfinance) historical price data into a retrievable knowledge base two ways:

1. **Narrative documents.** Deterministically generated (not LLM-written) weekly/monthly/yearly summaries of price behavior per ticker — returns, volatility, drawdowns, volume trends — indexed for hybrid (vector + keyword) search.
2. **Deterministic tools.** For questions that need an exact number (a specific return, a volatility figure, a head-to-head comparison), the LLM calls a Python function that computes it directly from the price data, instead of doing arithmetic over retrieved text.

A query-rewriting step extracts structured parameters (ticker, date range) from natural-language questions, feeding both retrieval and the tools. See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full architecture and the reasoning behind these choices.

## Ticker universe

A fixed set of 26 liquid tickers across 8 sectors — see [`docs/ticker_universe.md`](docs/ticker_universe.md) for the full list and the reasoning for keeping it bounded.

## Project layout

```
src/finrag/
├── config/          Ticker universe + typed settings (single source of truth)
├── ingestion/       yfinance fetch, stats computation, narrative doc generation
├── knowledge_base/  Postgres/pgvector schema, embedding + keyword indexing
├── retrieval/       Hybrid search, reranking, query rewriting
├── agent/           Deterministic tools + the function-calling loop
├── rag/             Prompt assembly, end-to-end pipeline
├── llm/             Provider-agnostic LLM client (Groq today, OpenAI later)
├── eval/            Retrieval eval, LLM-as-judge, failure-case harness (library code)
├── monitoring/      Query/feedback logging (writes to the query_logs table)
└── api/             FastAPI service (models, routes, app)
ui/                  Streamlit frontend (calls the API over HTTP, nothing else)
eval/                Evaluation CLI scripts, ground truth, and results
monitoring/grafana/  Grafana datasource + dashboard provisioning
tests/               Unit + integration tests
Dockerfile.api        API container image
Dockerfile.ui          UI container image
docker-compose.yml     Full local stack: db, pgadmin, api, ui, grafana
```

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.12.

```bash
uv sync --group dev
cp .env.example .env   # fill in GROQ_API_KEY
docker compose up -d   # starts Postgres+pgvector, pgadmin, the API, the UI, and Grafana
```

Fetch and cache price history, then build and load the knowledge base:

```bash
uv run python -m finrag.ingestion.fetch_prices
uv run finrag-ingest
```

Now the app is live: UI at http://localhost:8501, API docs at http://localhost:8000/docs, Grafana at http://localhost:3000 (anonymous viewer access enabled locally).

Ask a question from the command line instead, without the API/UI:

```bash
uv run finrag-ask "How did Apple perform in 2022?"
```

### Running the API/UI without Docker

Useful while iterating, since code changes don't need a rebuild:

```bash
uv run uvicorn finrag.api.main:app --reload      # http://localhost:8000
uv run streamlit run ui/app.py                    # http://localhost:8501, in another terminal
```

### Evaluation

```bash
uv run python eval/generate_ground_truth.py   # one-time, LLM-generated ground truth (cached)
uv run python eval/evaluate_retrieval.py      # Hit Rate / MRR across 5 retrieval configurations
uv run python eval/evaluate_llm.py            # LLM-as-judge: with-tools vs. without-tools (resumable)
uv run python eval/run_failure_cases.py       # known-tricky-question regression transcript (resumable)
```

See `docs/learning/day05_learning.md` and `day06_learning.md` for the methodology and findings.

### Tests and linting

```bash
uv run pytest
uv run ruff check src tests eval ui
```

## Documentation

- [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) — architecture, tech choices, roadmap, evaluation-criteria mapping
- [`docs/ticker_universe.md`](docs/ticker_universe.md) — the fixed ticker universe
- [`docs/deployment.md`](docs/deployment.md) — running the full docker-compose stack, environment variables, cloud deployment notes

## License

MIT
