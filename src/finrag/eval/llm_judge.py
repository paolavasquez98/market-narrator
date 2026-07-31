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
was available when generating both answers.

Score each answer from 1 (poor) to 5 (excellent), combining:
- Faithfulness: does the answer only state facts supported by the \
  CONTEXT or a reasonable computation, with no invented numbers?
- Relevance: does the answer actually address the question asked?

Respond with ONLY a JSON object (no other text, no markdown fences):
{"answer_a_score": <1-5 integer>, "answer_b_score": <1-5 integer>, "reasoning": "<one sentence>"}
"""

_JUDGE_PROMPT_TEMPLATE = """\
QUESTION: {question}

CONTEXT:
{context}

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
    llm: LLMClient, question: str, context: str, answer_a: str, answer_b: str
) -> JudgeVerdict | None:
    """Returns None (rather than raising) if the judge's response can't be
    parsed into the expected shape -- one bad judgment shouldn't abort a
    whole evaluation run, the same graceful-degradation philosophy as
    query_rewriter.rewrite_query and eval/ground_truth.py.
    """
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        question=question, context=context, answer_a=answer_a, answer_b=answer_b
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
