"""LLM-as-a-judge: score two candidate answers to the same question, on a
1-5 scale, for faithfulness and relevance combined.

Why an LLM judge instead of exact-match against a gold answer: this isn't
a Q&A dataset with one known-correct phrasing -- there's no single right
way to write "AAPL had a rough 2022." What can be judged is whether an
answer is grounded in the context/tool results it had access to
(faithfulness) and whether it actually addresses the question asked
(relevance), which an LLM judge can assess given the same information a
human reviewer would use.

Known limitation, not hidden: the judge here is the same model
(Groq/Llama) that generated the answers being judged. A same-model judge
can exhibit self-preference bias -- favoring its own typical phrasing or
reasoning style over an equally good answer written differently. Using a
different provider (OpenAI, once wired in -- see llm/factory.py) as judge
would reduce this risk; see docs/learning/day05_learning.md for the full
discussion of why this wasn't done today.

Fixed after the first real evaluation run: `_JUDGE_INSTRUCTIONS` used to
describe CONTEXT as "available when generating both answers," and told
the judge to penalize "invented numbers" not found in it. That's
literally false for the with-tools answer -- it also has agent/tools.py
available, which computes exact figures CONTEXT was deliberately never
meant to contain (see rag/prompts.py's SYSTEM_INSTRUCTIONS: "The
CONTEXT's numbers are only for fixed weekly/monthly/yearly periods and
may not match the exact range the user asked about"). The judge was
penalizing the with-tools pipeline for doing exactly what it was
designed to do. See docs/learning/day09_learning.md for the full
investigation and the evaluation-run evidence that led to this fix.

Second, stronger fix added the same day: `judge_answers` now optionally
accepts `tool_results` -- the actual tool calls/results captured from
`agent/orchestrator.run_agent_loop` (see its `tool_trace` parameter) --
and shows them to the judge as a distinct section. Previously the judge
had to take it on faith that a number absent from CONTEXT was a
legitimate tool computation rather than an invention; now, when a tool
was actually called, the judge can directly check a candidate answer's
number against it instead of just being told to be lenient about
unverifiable numbers. eval/llm_eval.py wires this up; callers that don't
pass `tool_results` (e.g. tests using the old 5-argument call) get the
previous behavior unchanged.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from finrag.llm.base import LLMClient

logger = logging.getLogger(__name__)

_JUDGE_INSTRUCTIONS = """\
You are evaluating two candidate answers (A and B) to the same question \
about historical stock market data. You are also shown the CONTEXT that \
was available to both answers -- but one or both answers may ALSO have \
had access to deterministic tools that compute an exact price, return, \
volatility, or comparison directly from the underlying daily price data, \
for any date range the user asks about -- not just the fixed \
weekly/monthly/yearly periods summarized in CONTEXT. A specific number \
that does not appear verbatim in CONTEXT is therefore NOT automatically \
invented or unfaithful -- it may be a correct tool computation for a \
date range CONTEXT was never meant to cover. Only mark an answer \
unfaithful for a number that is implausible, internally inconsistent, \
about the wrong ticker or date range, or otherwise not a reasonable \
answer to the question -- not merely for being absent from CONTEXT. If a \
TOOL RESULTS section is shown below, it contains the exact values a tool \
actually computed for this question -- use it to directly verify any \
matching number in either answer (a match is faithful; a contradiction \
is not), rather than guessing.

Score each answer from 1 (poor) to 5 (excellent), combining:
- Faithfulness: is the answer grounded in the CONTEXT or a plausible, \
  question-appropriate computation -- not an implausible or internally \
  inconsistent number?
- Relevance: does the answer actually address the question asked?

Respond with ONLY a JSON object (no other text, no markdown fences):
{"answer_a_score": <1-5 integer>, "answer_b_score": <1-5 integer>, "reasoning": "<one sentence>"}
"""

_JUDGE_PROMPT_TEMPLATE = """\
QUESTION: {question}

CONTEXT:
{context}
{tool_results_section}
ANSWER A:
{answer_a}

ANSWER B:
{answer_b}\
"""


@dataclass
class JudgeVerdict:
    answer_a_score: int
    answer_b_score: int
    reasoning: str


def judge_answers(
    llm: LLMClient,
    question: str,
    context: str,
    answer_a: str,
    answer_b: str,
    tool_results: str = "",
) -> JudgeVerdict | None:
    """Returns None (rather than raising) if the judge's response can't be
    parsed into the expected shape -- one bad judgment shouldn't abort a
    whole evaluation run, the same graceful-degradation philosophy as
    query_rewriter.rewrite_query and eval/ground_truth.py.

    `tool_results`: optional, defaults to "" (no section shown). Pass the
    formatted output of a captured tool_trace (see
    agent.orchestrator.run_agent_loop's `tool_trace` param and
    eval.llm_eval._format_tool_trace) so the judge can verify a candidate
    answer's number directly instead of guessing whether it's a
    legitimate tool computation -- see the module docstring's "Second,
    stronger fix" note.
    """
    tool_results_section = (
        f"\nTOOL RESULTS (exact values a tool computed for this question; "
        f"used by whichever answer(s) below called one -- not necessarily "
        f"both):\n{tool_results}\n"
        if tool_results
        else ""
    )
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        tool_results_section=tool_results_section,
        answer_a=answer_a,
        answer_b=answer_b,
    )
    try:
        response = llm.complete(
            messages=[
                {"role": "system", "content": _JUDGE_INSTRUCTIONS},
                {"role": "user", "content": prompt},
            ]
        )
        parsed = json.loads(response.content or "")
        return JudgeVerdict(
            answer_a_score=int(parsed["answer_a_score"]),
            answer_b_score=int(parsed["answer_b_score"]),
            reasoning=str(parsed.get("reasoning", "")),
        )
    except Exception:
        logger.exception("Judge response could not be parsed; skipping this comparison")
        return None
