"""Build the retrieval-evaluation ground truth set and save it to
eval/ground_truth.csv.

This costs one LLM call per sampled document (130 by default: 5 per
ticker x 26 tickers), so it's cached to disk like everything else in this
project's ingestion pipeline -- re-running this script is a no-op unless
the file is missing or --force is passed.

Usage:
    uv run python eval/generate_ground_truth.py
    uv run python eval/generate_ground_truth.py --force
"""

from __future__ import annotations

import argparse
import logging

from finrag.config.settings import get_settings
from finrag.eval.ground_truth import GROUND_TRUTH_PATH, build_ground_truth
from finrag.knowledge_base.vector_store import get_connection
from finrag.llm.factory import get_llm_client

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Regenerate even if the file exists")
    args = parser.parse_args()

    if GROUND_TRUTH_PATH.exists() and not args.force:
        logger.info("%s already exists. Use --force to regenerate.", GROUND_TRUTH_PATH)
        return

    settings = get_settings()
    llm = get_llm_client(settings)

    with get_connection(settings) as conn:
        ground_truth = build_ground_truth(llm, conn)

    GROUND_TRUTH_PATH.parent.mkdir(parents=True, exist_ok=True)
    ground_truth.to_csv(GROUND_TRUTH_PATH, index=False)
    logger.info("Wrote %d ground-truth questions to %s", len(ground_truth), GROUND_TRUTH_PATH)


if __name__ == "__main__":
    main()
