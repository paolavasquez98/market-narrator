"""CLI entrypoint that runs the full ingestion pipeline end to end:

    fetch prices -> compute stats -> generate narrative docs -> embed -> load

Each step is independently idempotent (fetch_prices caches to Parquet,
vector_store upserts by doc_id), so re-running this after changing, say,
the narrative template only touches what actually changed downstream --
no re-download, just new embeddings and an upsert.

Usage:
    uv run python -m finrag.ingestion.run_ingestion
    uv run finrag-ingest --force-fetch
"""

from __future__ import annotations

import argparse
import logging

from finrag.config.settings import get_settings
from finrag.ingestion.build_documents import build_all_documents
from finrag.ingestion.fetch_prices import fetch_universe
from finrag.knowledge_base.embeddings import Embedder
from finrag.knowledge_base.vector_store import count_documents, get_connection, upsert_documents

logger = logging.getLogger(__name__)


def run(force_fetch: bool = False) -> int:
    """Run the full pipeline once. Returns the total document count in the
    knowledge base after loading (useful for tests/CLI feedback).
    """
    settings = get_settings()

    fetch_universe(settings, force=force_fetch)

    records = build_all_documents(settings)
    logger.info("Generated %d narrative documents", len(records))
    if not records:
        logger.warning("No documents generated -- did fetch_universe succeed?")
        return 0

    embedder = Embedder()
    embeddings = embedder.embed([r.content for r in records])

    with get_connection(settings) as conn:
        upsert_documents(conn, records, embeddings)
        total = count_documents(conn)

    logger.info("Knowledge base now has %d documents", total)
    return total


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-fetch",
        action="store_true",
        help="Re-download price data even if already cached",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    run(force_fetch=args.force_fetch)


if __name__ == "__main__":
    main()
