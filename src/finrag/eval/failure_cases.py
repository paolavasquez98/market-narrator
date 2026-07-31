"""Reproducible harness for the five failure modes found during Day 4's
manual testing (ambiguous questions, unsupported companies, pronoun/
conversational context, tool misuse, and vague-question retrieval).

Why this isn't a metric like Hit Rate/MRR or the LLM judge: these failure
modes don't have a single correct answer to score against -- "how should
the model handle an ambiguous timeframe" is a product/UX judgment call,
not a fact that can be graded right/wrong. What can be made reproducible
is *running the same fixed set of known-tricky questions* against the
pipeline every time and capturing exactly what happened, so a human
reviewer can read the transcript, and -- more importantly -- diff two
transcripts from before/after a pipeline change to see whether a fix
(e.g. an updated system prompt) actually changed the behavior.

Each case's `known_issue` records the specific pathological behavior
observed in Day 4 manual testing, so a future re-run against the *same*
question is checked against the *same* concern rather than re-litigated
from scratch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from finrag.rag.pipeline import answer_question

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FailureCase:
    id: str
    category: str
    question: str
    known_issue: str


FAILURE_CASES: list[FailureCase] = [
    FailureCase(
        id="ambiguous-timeframe",
        category="ambiguous_question",
        question="How did Tesla do?",
        known_issue=(
            "No date range given. Day 4 manual testing showed the model picking an "
            "arbitrary period (2017) without flagging the ambiguity to the user."
        ),
    ),
    FailureCase(
        id="unsupported-ticker",
        category="unsupported_company",
        question="How did XYZ Corp perform in 2022?",
        known_issue=(
            "XYZ is not in the 26-ticker universe. Day 4 manual testing showed the "
            "agent hallucinating a plausible-looking substitute (a tool call for XOM) "
            "instead of reporting that the company isn't covered."
        ),
    ),
    FailureCase(
        id="pronoun-reference",
        category="conversational_context",
        question="How did Nvidia do in 2022? Now compare it to Apple.",
        known_issue=(
            "The CLI is stateless -- each call to answer_question() starts a fresh "
            "conversation, so a pronoun referring to a prior turn ('it') has nothing "
            "to resolve against within a single call. Packed into one question here "
            "so the limitation is reproducible without a multi-turn CLI."
        ),
    ),
    FailureCase(
        id="tool-call-as-text",
        category="tool_misuse",
        question="What was Apple's exact percentage return in 2022, and how would you describe its overall trend?",
        known_issue=(
            "Mixes a numeric ask (should trigger get_return) with a narrative ask "
            "(should use CONTEXT). Day 4 manual testing showed the model sometimes "
            "emitting a tool-call-shaped string as part of the plain-text answer "
            "instead of actually invoking the tool."
        ),
    ),
    FailureCase(
        id="vague-retrieval",
        category="vague_question_retrieval",
        question="How has AMD's market performance been?",
        known_issue=(
            "No date range or granularity cue. Day 4 manual testing showed hybrid "
            "search surfacing an arbitrary weekly document rather than a "
            "representative yearly/overall summary."
        ),
    ),
]


@dataclass
class FailureCaseResult:
    id: str
    category: str
    question: str
    known_issue: str
    rewritten_query: str
    resolved_tickers: list[str]
    answer: str


def run_failure_cases(cases: list[FailureCase] = FAILURE_CASES) -> list[FailureCaseResult]:
    """Run the full pipeline (rag.pipeline.answer_question, the same path
    finrag-ask uses) against each known-tricky question and capture what
    actually happened. Not scored -- see module docstring.
    """
    results = []
    for case in cases:
        logger.info("Running failure case %s: %r", case.id, case.question)
        rag_answer = answer_question(case.question)
        results.append(
            FailureCaseResult(
                id=case.id,
                category=case.category,
                question=case.question,
                known_issue=case.known_issue,
                rewritten_query=rag_answer.rewritten_query,
                resolved_tickers=rag_answer.resolved_tickers,
                answer=rag_answer.answer,
            )
        )
    return results


def render_markdown(results: list[FailureCaseResult]) -> str:
    """Render results as a markdown transcript for human review. Diffing
    this file against a previous run's transcript is the reproducibility
    story for these qualitative cases.
    """
    lines = ["# Failure case transcript", ""]
    for r in results:
        lines.append(f"## {r.id} ({r.category})")
        lines.append("")
        lines.append(f"**Question:** {r.question}")
        lines.append("")
        lines.append(f"**Known issue being probed:** {r.known_issue}")
        lines.append("")
        lines.append(
            f"**Rewritten query:** {r.rewritten_query!r}  |  **Resolved tickers:** {r.resolved_tickers}"
        )
        lines.append("")
        lines.append("**Answer:**")
        lines.append("")
        lines.append(r.answer)
        lines.append("")
        lines.append("---")
        lines.append("")
    return "\n".join(lines)
