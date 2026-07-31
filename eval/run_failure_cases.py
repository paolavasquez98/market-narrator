"""Re-run the fixed set of known-tricky questions (finrag.eval.failure_cases)
against the live pipeline and save a markdown transcript to eval/results/
for manual review.

Resumable: each case's result is appended to
eval/results/failure_cases_progress.json as soon as it's captured, and a
re-run skips any case already present there. Groq's free-tier daily token
budget can run out mid-run (`groq.RateLimitError`, HTTP 429) -- when that
happens, this script stops cleanly and everything captured so far is
already saved; re-run later to pick up the remaining cases. Pass --fresh
to ignore saved progress and start over.

Requires a running Postgres with ingested documents and a configured Groq
API key.

Usage:
    uv run python eval/run_failure_cases.py
    uv run python eval/run_failure_cases.py --fresh
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import groq

from finrag.eval.failure_cases import (
    FAILURE_CASES,
    FailureCaseResult,
    filter_pending_cases,
    render_markdown,
    run_failure_cases,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PROGRESS_PATH = RESULTS_DIR / "failure_cases_progress.json"


def _load_progress() -> list[FailureCaseResult]:
    if not PROGRESS_PATH.exists():
        return []
    with PROGRESS_PATH.open() as f:
        return [FailureCaseResult(**row) for row in json.load(f)]


def _save_progress(results: list[FailureCaseResult]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)


def _append_result(existing: list[FailureCaseResult], result: FailureCaseResult) -> None:
    """Persist the full progress list after each case -- JSON (unlike CSV)
    has no natural append mode, so this rewrites the whole (small, 5-case)
    file each time. Called from a closure in main() that keeps `existing`
    up to date.
    """
    existing.append(result)
    _save_progress(existing)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fresh", action="store_true", help="Ignore saved progress and re-run every case"
    )
    args = parser.parse_args()

    if args.fresh and PROGRESS_PATH.exists():
        PROGRESS_PATH.unlink()

    existing = _load_progress()
    already_done = {r.id for r in existing}
    pending = filter_pending_cases(FAILURE_CASES, already_done)

    if not pending:
        logger.info("All %d cases already captured in %s.", len(FAILURE_CASES), PROGRESS_PATH)
    else:
        logger.info(
            "%d of %d cases remaining (%d already captured).",
            len(pending),
            len(FAILURE_CASES),
            len(FAILURE_CASES) - len(pending),
        )
        try:
            run_failure_cases(
                cases=pending, on_result=lambda result: _append_result(existing, result)
            )
        except groq.RateLimitError:
            logger.warning(
                "Hit Groq's rate limit (likely the daily token budget), stopping here. "
                "Progress so far is saved in %s -- re-run this script (any time, including "
                "after the quota resets) to continue from where it left off.",
                PROGRESS_PATH,
            )

    results = _load_progress()
    if not results:
        print("No cases captured yet.")
        return

    transcript = render_markdown(results)
    print(transcript)
    print(f"{len(results)}/{len(FAILURE_CASES)} cases captured so far.")

    if len(results) == len(FAILURE_CASES):
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_path = RESULTS_DIR / f"failure_cases_{timestamp}.md"
        out_path.write_text(transcript)
        logger.info("All cases captured. Saved a final transcript to %s", out_path)


if __name__ == "__main__":
    main()
