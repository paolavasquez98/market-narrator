"""Re-run the fixed set of known-tricky questions (finrag.eval.failure_cases)
against the live pipeline and save a markdown transcript to eval/results/
for manual review.

Requires a running Postgres with ingested documents and a configured Groq
API key -- like evaluate_llm.py, this makes real calls and can't run in
CI/sandbox.

Usage:
    uv run python eval/run_failure_cases.py
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

from finrag.eval.failure_cases import render_markdown, run_failure_cases

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    results = run_failure_cases()
    transcript = render_markdown(results)
    print(transcript)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"failure_cases_{timestamp}.md"
    out_path.write_text(transcript)
    logger.info("Saved transcript to %s", out_path)


if __name__ == "__main__":
    main()
