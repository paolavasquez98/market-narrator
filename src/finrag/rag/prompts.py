"""Prompt templates for the RAG pipeline.

Kept in one place, separate from the orchestration logic in pipeline.py,
for the same reason SQL lives in one place in knowledge_base/: prompt
wording is exactly the kind of thing we'll be iterating on and A/B
comparing during LLM evaluation (Day 5) -- it should be easy to find and
easy to swap without touching pipeline control flow.
"""

from __future__ import annotations

from finrag.knowledge_base.models import SearchResult

SYSTEM_INSTRUCTIONS = """\
You are a financial market research assistant. You answer questions about \
historical stock price behavior using only the CONTEXT provided below, \
which was generated deterministically from real daily price data (not \
written by an LLM) -- every number in it is accurate as of when the \
knowledge base was built.

Rules:
- Base your answer only on the CONTEXT. Do not use outside knowledge about \
  companies, markets, or events.
- If the CONTEXT does not contain enough information to answer, say so \
  plainly instead of guessing.
- When you state a number (a return, a price, a volatility figure), it \
  should come directly from the CONTEXT.
- Be concise and specific. Prefer citing the exact tickers and periods \
  from the CONTEXT over vague language.\
"""

PROMPT_TEMPLATE = """\
QUESTION: {question}

CONTEXT:
{context}\
"""

_NO_CONTEXT_PLACEHOLDER = "(No matching documents were found in the knowledge base.)"


def build_context(results: list[SearchResult]) -> str:
    """Turn retrieved SearchResults into the CONTEXT block of the prompt.
    Each chunk is labeled with its ticker/granularity/period so the model
    (and a human reading logs later) can tell which document a fact came
    from -- useful for both grounding and debugging retrieval quality.
    """
    if not results:
        return _NO_CONTEXT_PLACEHOLDER

    blocks = []
    for r in results:
        header = f"[{r.ticker} | {r.granularity} | {r.period_start} to {r.period_end}]"
        blocks.append(f"{header}\n{r.content}")
    return "\n\n".join(blocks)


def build_prompt(question: str, results: list[SearchResult]) -> str:
    return PROMPT_TEMPLATE.format(question=question, context=build_context(results))
