# Market Narrator — Project Plan

An end-to-end RAG application for exploring historical stock market behavior, built for the DataTalks.Club LLM Zoomcamp capstone.

Deadline: **August 4, 2026**. Today: **July 23, 2026**. This document is the working plan for the project and is updated as decisions are made.

---

## 1. Problem statement

Raw OHLCV data from `yfinance` is not itself "knowledge" — it's a table of numbers. A good RAG project needs a retrievable text corpus, not a spreadsheet. So the core design problem is: **how do you build a meaningful knowledge base out of price data?**

The answer we're using: generate grounded, deterministic natural-language summaries of price behavior (weekly / monthly / yearly, per ticker) via Python — not an LLM — so the corpus is factually reliable, then let the RAG system retrieve and reason over those summaries. For questions that need an exact number (a specific return %, a volatility figure, a head-to-head comparison), we don't trust the LLM to do arithmetic over retrieved text — we give it a **tool** that computes the number deterministically from the underlying price data. This mirrors Module 1 of the course (Agentic RAG: search + function calling) and solves the "RAG over numeric data" problem properly instead of faking it.

## 2. Project ideas considered

**A. Narrative-only RAG.** Generate price-behavior summary documents per ticker/period, embed and index them, retrieve + answer. Simplest option. Weakness: any question needing precise numbers (e.g. "what was NVDA's exact return in 2023?") is at the mercy of whatever number happened to land in a retrieved chunk — not reliable, and doesn't showcase much engineering depth.

**B. Fundamentals RAG.** Use `yfinance` financial statements (income statement, balance sheet, analyst recommendations) as the corpus instead of price history. Richer text, but drifts away from the "historical market behavior" framing you actually want, and fundamentals data from `yfinance` is less consistently available across tickers than price history.

**C. Hybrid agentic RAG (recommended).** Narrative summary documents (as in A) as the vector-searchable knowledge base, **plus** a small set of deterministic tools (`get_return`, `get_volatility`, `compare_tickers`, `get_price_on_date`) that the LLM can call via function calling for anything numeric. A query-rewriting step extracts structured parameters (ticker, date range) from the user's natural-language question, which both improves retrieval and feeds the tools.

### Recommendation: C

It directly answers every example question you listed — "How did Apple perform in 2022" and "Compare MSFT vs NVDA over 5 years" become tool calls with retrieved narrative context for color; "What happened to Tesla after X event" and "Summarize SPY's trend last month" lean on retrieval. It gives you a legitimate, defensible reason to build the agentic piece (not bolted on for bonus points), and it's the same amount of *additional* work as A once the narrative generation pipeline exists — the tools are simple pandas functions, not a new subsystem. It's realistic for 11 days because nothing about it requires a heavy agent framework: a single function-calling loop with 3-4 tools is Module 1 material, not a research project.

## 3. Architecture

```
User (Streamlit UI)
      │
      ▼
FastAPI service
      │
      ├─► Query rewriter (LLM) ──► structured filters (ticker, date range) + rewritten query
      │
      ├─► Hybrid retrieval (pgvector cosine + Postgres full-text, fused with RRF)
      │        └─► Reranker (cross-encoder) on top-k
      │
      ├─► Agent tool layer (deterministic pandas/duckdb functions: return, volatility, comparison, price-on-date)
      │
      └─► LLM (Groq) composes grounded answer from retrieved context + tool outputs
      │
      ▼
Postgres (logs: query, rewritten query, retrieved doc ids + scores, tool calls, latency, answer, feedback)
      │
      ▼
Grafana dashboard (≥5 charts)
```

Ingestion is a separate offline pipeline: pull `yfinance` OHLCV for a fixed ticker universe → compute stats → generate narrative documents at multiple time granularities → embed → upsert into Postgres/pgvector.

## 4. Technology choices

| Concern | Choice | Why |
|---|---|---|
| Package/env management | `uv` (already in repo) | Fast, lockfile-based, good reproducibility story |
| Data source | `yfinance` | As specified |
| Numeric layer | pandas | Standard, no reason to add duckdb complexity for this data volume |
| Knowledge base / vector store | Postgres + `pgvector` | Explicitly covered in this year's course (Module 2), lets one database also serve keyword search (hybrid) and monitoring logs — one moving part instead of three |
| Keyword search | Postgres full-text search (`tsvector`) | Avoids standing up Elasticsearch just for hybrid search |
| Embeddings + reranking | `fastembed` (ONNX-based, no torch) | Local, free, deterministic, and avoids pulling in a multi-GB torch dependency into Docker images — matters for build time and reproducibility on a deadline |
| LLM | Groq (`llama-3.3-70b-versatile`), OpenAI `gpt-4o-mini` as a second model for evaluation/judging | Groq key already configured; fast + free tier good for heavy iteration; second provider needed anyway for the "multiple approaches evaluated" LLM-eval criterion |
| Agent / tool calling | Hand-rolled function-calling loop (Groq/OpenAI tool schema) | No LangChain/LangGraph — fewer moving parts, and you understand every line, which matters more than the framework given interview-readiness is a goal |
| API | FastAPI | Clean separation from the UI, testable, matches "production-style code" goal |
| UI | Streamlit | Fast to build, sufficient for the 2-point "interface" criterion, calls the FastAPI service rather than reimplementing logic |
| Monitoring | Postgres logging table + Grafana | Same pattern as course Module 5 / fitness-assistant reference project |
| Containerization | Docker Compose: `postgres`, `api`, `ui`, `grafana` | Everything in compose = full containerization score |
| CI | GitHub Actions: lint (ruff) + unit tests + a retrieval-eval smoke test | Cheap to add, strong "best practices" signal for reviewers |

**Two corrections to the current repo setup**, flagging before we build on top of them:

1. `pyproject.toml` currently pins `requires-python = ">=3.14"`. Python 3.14 is very new (Oct 2025) and several libraries in this stack (fastembed's onnxruntime dependency, psycopg, etc.) may not have wheels yet, which risks losing a day to install issues close to the deadline. Recommend pinning to `>=3.12,<3.13` — stable, universally supported, no functionality lost.
2. `main.py`, `ingest.py`, `rag_helper.py`, and `test.ipynb` are leftovers from earlier course homework (the FAQ dataset, `minsearch`) and aren't part of this project. Recommend removing them once the new structure is in place rather than leaving dead code in a repo reviewers will read.

## 5. Repository structure

```
llm-zoomcamp/                      (rename to something project-specific before submission, e.g. market-narrator-rag)
├── README.md
├── docs/
│   ├── PROJECT_PLAN.md            (this file)
│   ├── architecture.md
│   ├── evaluation.md
│   ├── design_decisions.md
│   └── future_work.md
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.ui
├── Makefile
├── pyproject.toml / uv.lock
├── .env.example
├── data/                          (generated corpus; raw price cache gitignored, sample docs committed)
├── src/finrag/
│   ├── config.py
│   ├── ingestion/
│   │   ├── fetch_prices.py
│   │   ├── compute_stats.py
│   │   ├── build_documents.py
│   │   └── run_ingestion.py
│   ├── knowledge_base/
│   │   ├── schema.sql
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── keyword_store.py
│   ├── retrieval/
│   │   ├── hybrid_search.py
│   │   ├── reranker.py
│   │   └── query_rewriter.py
│   ├── agent/
│   │   ├── tools.py
│   │   └── orchestrator.py
│   ├── rag/
│   │   ├── prompts.py
│   │   └── pipeline.py
│   ├── llm/client.py
│   ├── monitoring/logger.py
│   └── api/{main.py, routes.py}
├── ui/app.py
├── eval/
│   ├── generate_ground_truth.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_llm.py
│   └── results/
├── monitoring/grafana/{provisioning, dashboards}
├── tests/
└── .github/workflows/ci.yml
```

## 6. Roadmap (8 working days)

Each day maps to specific scoring criteria so nothing is built without a reason.

| Day | Date | Focus | Criteria advanced | Est. hours |
|---|---|---|---|---|
| 1 | Fri Jul 24 | Repo cleanup, new structure, `uv` deps, Docker Compose Postgres+pgvector up, ticker universe fixed, raw price fetch + caching | Reproducibility, ingestion (partial) | 3-4 |
| 2 | Mon Jul 27 | Stats computation + narrative doc generation (weekly/monthly/yearly) for full ticker universe, embeddings, load into Postgres/pgvector + FTS index | Ingestion pipeline (2 pts), knowledge base | 3-4 |
| 3 | Tue Jul 28 | Keyword-only + vector-only + hybrid (RRF) search; basic RAG pipeline (search → prompt → LLM) working end-to-end via script | Retrieval flow (2 pts) | 4 |
| 4 | Wed Jul 29 | Query rewriter, agent tool layer (return/volatility/comparison/price-on-date), reranker wired in | Best practices: hybrid search, reranking, query rewriting (3 pts) | 4-5 |
| 5 | Thu Jul 30 | Ground-truth Q&A generation; retrieval eval (Hit Rate/MRR) across keyword/vector/hybrid/hybrid+rerank, pick best; LLM-as-judge eval across ≥2 prompt/model variants, pick best | Retrieval evaluation (2 pts), LLM evaluation (2 pts) | 4-5 |
| 6 | Fri Jul 31 | FastAPI service, Streamlit UI wired to it, feedback capture (thumbs up/down) + query logging to Postgres | Interface (2 pts), monitoring (partial) | 4-5 |
| 7 | Mon Aug 3 | Grafana dashboard (≥5 charts), full docker-compose stack, GitHub Actions CI, README + architecture doc draft | Monitoring (2 pts), containerization (2 pts) | 5-6 |
| 8 | Tue Aug 4 | Buffer for bugs, finish documentation (setup guide, screenshots, demo clip, design decisions, future work), attempt cloud deployment if time allows, final checklist pass against every criterion, submit **early**, not at the deadline | Reproducibility, problem description, bonus (cloud deploy) | 4-6 |

Cloud deployment (2 bonus points) is scheduled as a stretch goal on Day 8 rather than a committed milestone — if Days 1-7 run long, it's the thing to cut, not core RAG functionality.

## 7. Evaluation criteria coverage

| Criterion | How this plan hits max score |
|---|---|
| Problem description | Clear README framing: grounded Q&A over historical price behavior, worked examples |
| Retrieval flow | Knowledge base (pgvector+FTS) + LLM, both used |
| Retrieval evaluation | 4 approaches compared (keyword, vector, hybrid, hybrid+rerank), best one selected with numbers to show it |
| LLM evaluation | ≥2 prompt/model variants compared via LLM-as-judge, best one selected |
| Interface | FastAPI + Streamlit |
| Ingestion pipeline | Fully automated Python script, one command |
| Monitoring | Feedback capture + Grafana dashboard, ≥5 charts |
| Containerization | Full docker-compose (postgres, api, ui, grafana) |
| Reproducibility | uv lockfile, pinned Python version, `.env.example`, fixed ticker universe & date range, Makefile targets |
| Best practices | Hybrid search + reranking + query rewriting, all three |
| Bonus | Cloud deployment (stretch), extra engineering (agentic tool-calling, CI) |

## 8. Open decisions to confirm before Day 1

- Final ticker universe (recommend ~20-30 liquid names across sectors + SPY/QQQ, fixed list documented in the repo)
- Repo name / whether to rename from `llm-zoomcamp`
- Whether to keep both Groq and OpenAI, or pick one to reduce API key management (recommend keeping both — needed for LLM eval criterion anyway)
