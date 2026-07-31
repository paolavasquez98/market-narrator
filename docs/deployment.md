# Deployment

How to run the full stack locally via Docker Compose, what each service needs, and what's involved in taking this to a real cloud deployment (not done yet — see the end of this doc).

## Local stack (docker-compose)

```bash
cp .env.example .env   # fill in GROQ_API_KEY
uv run python -m finrag.ingestion.fetch_prices   # populates ./data/raw (needed by the API container, see below)
docker compose up -d --build
uv run finrag-ingest   # or run it inside the api container; either way it needs to happen once
```

| Service | Image / build | Port | Depends on |
|---|---|---|---|
| `db` | `pgvector/pgvector:pg16` | 5432 | — |
| `pgadmin` | `dpage/pgadmin4` | 8085 | `db` healthy |
| `api` | `Dockerfile.api` | 8000 | `db` healthy |
| `ui` | `Dockerfile.ui` | 8501 | `api` |
| `grafana` | `grafana/grafana:11.2.0` | 3000 | `db` healthy |

Once up: UI at `localhost:8501`, interactive API docs at `localhost:8000/docs`, Grafana at `localhost:3000` (anonymous viewer access is enabled for this local/demo deployment — see the `grafana` service's comments in `docker-compose.yml` for why that's an acceptable trade-off here specifically).

### Why the `api` container needs `./data` mounted, not just Postgres

This is the one non-obvious dependency worth calling out explicitly: `agent/tools.py`'s deterministic tools (`get_return`, `get_volatility`, etc.) read cached daily price data directly from Parquet files via `fetch_prices.load_cached_prices()` — **not** from Postgres. Postgres only holds the narrative documents used for retrieval. If `./data/raw/*.parquet` isn't populated and mounted into the `api` container (`docker-compose.yml`'s `api.volumes`), retrieval and narrative answers work fine, but every tool call fails with a `FileNotFoundError` surfaced to the model as a tool error — a confusing partial failure if you don't know to look for it. Run `fetch_prices.py` on the host before `docker compose up` (or into a volume the container also writes to) so the cache exists before the API needs it.

### Why there's a named volume for the embedding/reranking models

`fastembed` downloads its ONNX models (BAAI/bge-small-en-v1.5 for embeddings, Xenova/ms-marco-MiniLM-L-6-v2 for reranking — roughly 100-200MB combined) from Hugging Face on first use, to `$FASTEMBED_CACHE_PATH`. The `api` service sets that to `/app/.cache/fastembed` and mounts a named volume (`fastembed-cache`) there, so a container restart doesn't re-download the models every time — the first `/ask` call after a fresh `docker compose up` will be slower than subsequent ones while the models download once.

### Environment variables

See `.env.example` for the full list. The two worth understanding rather than just copying:

- `DATABASE_URL` — the `.env` value (`localhost:5432`) is correct for running the API directly on the host (`uv run uvicorn ...`). Inside docker-compose, Postgres is reachable at the service name `db`, not `localhost` — `docker-compose.yml`'s `api` service overrides this explicitly via its own `environment:` block (which takes precedence over `env_file:`), so you don't need to maintain two different `.env` files.
- `API_BASE_URL` (read by `ui/app.py` via `Settings.api_base_url`) — same story: `localhost:8000` for running the UI directly on the host, `http://api:8000` (the compose service name) when containerized. Set via the `ui` service's `environment:` block.

### Health and observability

- `GET /health` — checks Postgres connectivity, deliberately does **not** call Groq (see `api/routes.py`'s docstring for why: cheap, frequent-safe, and DB is the one dependency `/ask` genuinely can't work around). Used by nothing automated yet, but is what a container orchestrator's liveness probe or an uptime monitor should point at.
- `query_logs` (schema.sql) — every `/ask` call is logged here (question, rewritten query, resolved tickers, retrieved doc ids, tool calls, latency, and later, feedback via `/feedback/{query_id}`). This is what the Grafana dashboard (`monitoring/grafana/provisioning/dashboards/market_narrator.json`) reads from — query volume over time, latency (average and p95), feedback ratio, tool-calling usage, and most-asked-about tickers.

## Cloud deployment — not done, notes for later

Cloud deployment is the bonus/stretch item in `docs/PROJECT_PLAN.md` (Day 8, explicitly scheduled as a cut-if-time-runs-out item, not a committed milestone), and hasn't been attempted — there's no live deployment to point at. If picked up later, the docker-compose setup above is the starting point; what it doesn't yet handle:

- **Secrets.** `GROQ_API_KEY` currently comes from a local `.env` file (gitignored, never committed). A real deployment needs a secrets manager (the specific choice depends on the platform) instead of an env file baked into a container image or committed anywhere.
- **Managed Postgres+pgvector.** The `db` service's data lives in a local Docker volume. A cloud deployment needs a managed Postgres instance with the `vector` extension available (e.g. a managed Postgres offering that supports pgvector, or a self-hosted instance with persistent storage and backups) — `schema.sql` is the same either way, just applied once against a real host.
- **The `data/` volume.** The price-cache dependency described above needs to exist somewhere the deployed `api` container can read from — either baked into the image at build time (simplest, but means rebuilding the image to refresh prices) or a persistent volume populated by a one-off ingestion job.
- **Ingestion as a job, not a manual step.** `finrag-ingest` and `fetch_prices.py` are currently run by hand. A real deployment would run them as a one-off job (before first traffic, and periodically to pick up new price data) rather than expecting someone to SSH in.
- **CORS.** `api/main.py` currently allows all origins — fine for a same-machine UI-to-API call locally, worth tightening to the actual UI's deployed origin once there is one.
