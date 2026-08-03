.PHONY: help up down logs restart ingest ask test lint fmt \
        eval-ground-truth eval-retrieval eval-llm eval-failures clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Local stack -------------------------------------------------------

up: ## Start the full stack (db, pgadmin, api, ui, grafana)
	docker compose up -d --build

down: ## Stop the stack (keeps volumes -- data, price cache, grafana state)
	docker compose down

logs: ## Tail logs from every service
	docker compose logs -f

restart: down up ## Recreate every container (e.g. after a Dockerfile change)

# --- Data / ingestion ----------------------------------------------------

ingest: ## Fetch price history (cached) and (re)build the knowledge base
	uv run python -m finrag.ingestion.fetch_prices
	uv run finrag-ingest

ask: ## Ask a question from the CLI, e.g. `make ask Q="How did AAPL do in 2022?"`
	uv run finrag-ask "$(Q)"

# --- Quality gates -------------------------------------------------------

test: ## Run the test suite (DB-dependent tests skip unless `make up` has run)
	uv run pytest

lint: ## Lint everything ruff covers in CI
	uv run ruff check src tests eval ui

fmt: ## Auto-format with ruff
	uv run ruff format src tests eval ui

# --- Evaluation (see docs/learning/day05_learning.md and day06_learning.md) ---

eval-ground-truth: ## Generate eval/ground_truth.csv (cached, skips if it exists)
	uv run python eval/generate_ground_truth.py

eval-retrieval: ## Compare keyword/vector/hybrid/hybrid_rerank/hybrid_rerank_rewrite
	uv run python eval/evaluate_retrieval.py

eval-llm: ## LLM-as-judge: with-tools vs. without-tools (resumable across Groq rate limits)
	uv run python eval/evaluate_llm.py

eval-failures: ## Re-run the known-tricky-question regression transcript (resumable)
	uv run python eval/run_failure_cases.py

# --- Cleanup ---------------------------------------------------------------

clean: ## Remove Python/tool caches (not data, not Docker volumes)
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
