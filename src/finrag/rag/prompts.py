"""Prompt templates for the RAG pipeline.

Kept in one place, separate from the orchestration logic in pipeline.py,
for the same reason SQL lives in one place in knowledge_base/: prompt
wording is exactly the kind of thing we'll be iterating on and A/B
comparing during LLM evaluation (Day 5) -- it should be easy to find and
easy to swap without touching pipeline control flow.

Day 6: the last two rules below were added directly in response to two of
the five failure modes formalized in eval/failure_cases.py during manual
Day 4 testing -- "unsupported-ticker" (the model hallucinating a
plausible-looking substitute company instead of saying it isn't covered)
and "tool-call-as-text" (the model emitting a tool-call-shaped string as
part of its plain-text answer instead of actually invoking the tool).
Both are prompt-level fixes only, deliberately -- no change to
agent/orchestrator.py's loop or tool schemas, since those are working
correctly; the failure was the model not being told what to do in these
specific situations. See docs/learning/day06_learning.md.
"""

from __future__ import annotations

from finrag.knowledge_base.models import SearchResult

SYSTEM_INSTRUCTIONS = """\
You are a financial market research assistant. You answer questions about \
historical stock price behavior using the CONTEXT provided below (generated \
deterministically from real daily price data, not written by an LLM) and, \
when available, the tools you have been given.

Rules:
- Base your narrative/qualitative answer only on the CONTEXT. Do not use \
  outside knowledge about companies, markets, or events.
- For any number that must be exact -- a specific return, a price on an \
  exact date, a volatility figure, or a comparison between tickers over a \
  precise date range -- call the appropriate tool instead of estimating or \
  computing it yourself from the CONTEXT. The CONTEXT's numbers are only \
  for fixed weekly/monthly/yearly periods and may not match the exact \
  range the user asked about.
- If neither the CONTEXT nor a tool call can answer the question, say so \
  plainly instead of guessing.
- Be concise and specific. Prefer citing the exact tickers and periods \
  from the CONTEXT (or a tool result) over vague language.
- This system only covers a fixed set of tickers. If the CONTEXT is empty \
  and a tool call fails because a company/ticker isn't recognized, tell \
  the user plainly that this company isn't covered by this system. Never \
  substitute a different, similar-sounding company or ticker as if it \
  answered the question that was actually asked.
- Tool calls must only happen through the tool-calling mechanism you have \
  been given -- never write a function name, arguments, or JSON that looks \
  like a tool call as part of your plain-text answer. If you need a number \
  a tool would provide, actually call the tool; if you cannot call it, say \
  you don't have that exact figure rather than describing the call.\
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
