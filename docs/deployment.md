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

## Cloud deployment — prepared, not attempted

Cloud deployment is the bonus/stretch item in `docs/PROJECT_PLAN.md` (Day 8, explicitly scheduled as a cut-if-time-runs-out item, not a committed milestone). No live deployment exists — this development environment has no credentials for any cloud provider, and attempting to sign up for one and hand over payment/account details wasn't something to do without the project owner directly involved. What's here instead: a concrete, reviewed starting point plus an honest list of what it doesn't solve yet.

**Recommended platform: Render.** Chosen over Railway/Fly.io mainly because its managed Postgres supports the `vector` extension directly (confirmed against Render's current docs), so `schema.sql` doesn't need to change at all — the same schema that runs locally via docker-compose applies as-is. [`render.yaml`](../render.yaml) at the repo root is a Blueprint (Render's declarative multi-service config, analogous to `docker-compose.yml`) covering the `api` and `ui` services plus a managed Postgres database. **It has not been deployed or tested against a real Render account** — review it against [Render's current Blueprint spec](https://render.com/docs/blueprint-spec) before relying on it, particularly the `API_BASE_URL` cross-service reference, which is set as a literal placeholder URL rather than a verified `fromService` reference.

What a real deployment via that Blueprint (or any other platform) still needs, beyond what's in the file:

- **Secrets.** `GROQ_API_KEY` currently comes from a local `.env` file (gitignored, never committed). `render.yaml` marks it `sync: false` — set once in the platform's dashboard, never committed — but the same principle applies on any platform: a secrets manager, not an env file baked into an image.
- **Applying `schema.sql` once.** Render (like most managed Postgres offerings) doesn't run an init-script-on-creation hook the way `docker-compose.yml`'s local `db` service does (`docker-entrypoint-initdb.d`) — this is a manual one-time `psql` step against the new database, noted directly in `render.yaml`'s comments.
- **The price-cache volume needs a different approach in the cloud.** Locally, `docker-compose.yml` bind-mounts the host's `./data` directory into the `api` container. Render (and most PaaS platforms) has no equivalent for a host directory that doesn't exist in the cloud. `render.yaml`'s comments describe the correct fix: bake `data/raw/*.parquet` into the image at build time (`COPY data/raw ./data/raw` in `Dockerfile.api`) instead of mounting it — not made the Dockerfile's default because it would bloat and slow down every local build for a concern that's cloud-deployment-specific.
- **Ingestion as a one-off job, not a manual step.** `finrag-ingest` and `fetch_prices.py` are currently run by hand. A real deployment would run them once (before first traffic) and periodically (to pick up new price data) as a scheduled job, not by someone SSHing into a running container.
- **CORS.** `api/main.py` currently allows all origins — fine for a same-machine UI-to-API call with no authentication anywhere in the system; worth tightening to the deployed UI's actual origin once there is one.

If picked up later, the fastest path to a real (not just prepared) deployment: create the Render Postgres database first, apply `schema.sql` and confirm `CREATE EXTENSION vector` works, then connect the repo as a Blueprint and fill in `GROQ_API_KEY` — at that point the gaps above (data volume, ingestion job) become concrete blockers to work through one at a time rather than a wall of upfront unknowns.
