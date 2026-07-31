"""Compare retrieval configurations against eval/ground_truth.csv and save
a timestamped results file to eval/results/.

Requires eval/ground_truth.csv to exist -- run
`uv run python eval/generate_ground_truth.py` first if it doesn't.

Usage:
    uv run python eval/evaluate_retrieval.py
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd

from finrag.config.settings import get_settings
from finrag.eval.ground_truth import GROUND_TRUTH_PATH
from finrag.eval.retrieval_eval import best_variant, evaluate_all_variants, results_to_dataframe
from finrag.knowledge_base.embeddings import Embedder
from finrag.knowledge_base.vector_store import get_connection
from finrag.llm.factory import get_llm_client
from finrag.retrieval.reranker import Reranker

logger = logging.getLogger(__name__)

RESULTS_DIR = GROUND_TRUTH_PATH.parent / "results"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if not GROUND_TRUTH_PATH.exists():
        raise SystemExit(
            f"{GROUND_TRUTH_PATH} not found. "
            "Run `uv run python eval/generate_ground_truth.py` first."
        )
    ground_truth = pd.read_csv(GROUND_TRUTH_PATH)
    logger.info("Loaded %d ground-truth questions", len(ground_truth))

    settings = get_settings()
    llm = get_llm_client(settings)
    embedder = Embedder()
    reranker = Reranker()

    with get_connection(settings) as conn:
        results = evaluate_all_variants(conn, llm, embedder, reranker, ground_truth)

    df = results_to_dataframe(results)
    print("\n" + df.to_string(index=False) + "\n")

    winner = best_variant(results)
    print(f"Best variant by MRR: {winner.variant} (MRR={winner.mrr:.3f}, Hit Rate={winner.hit_rate:.3f})\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"retrieval_eval_{timestamp}.csv"
    df.to_csv(out_path, index=False)
    logger.info("Saved results to %s", out_path)


if __name__ == "__main__":
    main()
