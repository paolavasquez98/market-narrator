"""LLM-based query rewriting: extract structured search intent from a raw
natural-language question before retrieval.

Why this is needed (motivated directly by Day 3's observed failures):
embedding-only retrieval matches "Nvidia" to semantically-adjacent-but-
wrong documents (AMD, another semiconductor maker) because "Nvidia" never
appears in any document -- the narrative documents only ever say "NVDA".
No amount of better embeddings fixes a company name that isn't the
ticker symbol; what's needed is to resolve "Nvidia" -> "NVDA" *before*
searching, and use that as a hard filter (`tickers=` on `hybrid_search`),
not just a fuzzy signal for the embedding to approximate.

We don't maintain our own company-name-to-ticker lookup table for this.
The fixed ticker universe (26 symbols) is small enough to hand the LLM
directly and let its own world knowledge do the mapping -- then validate
whatever it returns against the real universe in code, rather than
trusting the prompt alone (a hallucinated ticker would otherwise silently
turn into an always-empty filter).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from finrag.config.tickers import all_tickers
from finrag.llm.base import LLMClient

logger = logging.getLogger(__name__)

_REWRITE_INSTRUCTIONS = """\
You extract structured search parameters from a user's question about \
historical stock market data.

The ONLY tickers that exist in this system are: {tickers}

Given the user's question, respond with ONLY a JSON object (no other \
text, no markdown fences) with these exact keys:
- "tickers": a list of tickers from the list above that the question is \
  about. Use your own knowledge of company names to match them (for \
  example "Nvidia" -> "NVDA", "Apple" -> "AAPL"). Use an empty list if no \
  ticker from the list above is mentioned or clearly implied.
- "rewritten_query": the question rewritten as a clear, standalone search \
  query -- expand company names/abbreviations, drop conversational \
  filler. If the question is already clear, return it unchanged.
"""


@dataclass
class QueryIntent:
    rewritten_query: str
    tickers: list[str]


def _build_instructions() -> str:
    return _REWRITE_INSTRUCTIONS.format(tickers=", ".join(all_tickers()))


def rewrite_query(llm: LLMClient, question: str) -> QueryIntent:
    """Best-effort structured extraction. Falls back to the original
    question with no ticker filter if the LLM's response isn't valid
    JSON, or if it hallucinates tickers outside the real universe --
    retrieval should degrade to Day 3's behavior in that case, not crash.
    """
    response = llm.complete(
        messages=[
            {"role": "system", "content": _build_instructions()},
            {"role": "user", "content": question},
        ]
    )

    try:
        parsed = json.loads(response.content or "")
        valid_tickers = set(all_tickers())
        tickers = [t for t in parsed.get("tickers", []) if t in valid_tickers]
        rewritten_query = parsed.get("rewritten_query") or question
        if not isinstance(rewritten_query, str):
            raise TypeError("rewritten_query was not a string")
        return QueryIntent(rewritten_query=rewritten_query, tickers=tickers)
    except (json.JSONDecodeError, AttributeError, TypeError) as exc:
        logger.warning(
            "Query rewriting failed to parse LLM response (%s); falling back to raw question",
            exc,
        )
        return QueryIntent(rewritten_query=question, tickers=[])
