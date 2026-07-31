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

Day 6 finding: the retrieval eval (eval/evaluate_retrieval.py) showed the
`hybrid_rerank_rewrite` variant scoring *worse* (Hit Rate 0.16, MRR 0.11)
than plain `hybrid_rerank` (0.24, 0.17) -- rewriting was hurting, not
helping. Root cause: `build_documents.render_narrative` templates every
document almost identically for a given ticker, differing mainly in the
exact numbers and the exact period dates -- there is no other strong
signal to tell "AAPL week of March 3" apart from "AAPL week of March 10"
once the ticker filter has narrowed the field. The original rewrite
instructions said to "expand company names/abbreviations, drop
conversational filler" with no explicit instruction to preserve dates --
a rewrite that generalized "the week of October 21 to October 25, 2019"
down to something vaguer discarded the one piece of signal retrieval
depended on for that ticker. Fixed below by making date preservation an
explicit, loud rule. See docs/learning/day06_learning.md for the full
investigation and what to re-check once Groq quota allows a re-run.
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

CRITICAL: preserve every date, month, quarter, or year mentioned in the \
question EXACTLY as given, in the rewritten query. Documents in this \
system are distinguished from one another almost entirely by their time \
period -- two documents about the same company can be otherwise nearly \
identical in wording and differ only in which week, month, or year they \
cover. Dropping, paraphrasing, or generalizing a date (for example \
turning "the week of October 21 to October 25, 2019" into just "in 2019", \
or "May 2017" into "recently") destroys the only signal that tells the \
right period apart from every other period for that ticker. Never do this.
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
