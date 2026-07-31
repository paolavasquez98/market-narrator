from datetime import date

from finrag.knowledge_base.models import SearchResult
from finrag.rag.prompts import SYSTEM_INSTRUCTIONS, build_context, build_prompt


def _result(doc_id: str, content: str) -> SearchResult:
    return SearchResult(
        doc_id=doc_id,
        ticker="AAPL",
        sector="Technology",
        granularity="weekly",
        period_start=date(2024, 1, 1),
        period_end=date(2024, 1, 5),
        content=content,
        score=0.9,
    )


def test_build_context_labels_each_chunk_with_ticker_and_period():
    context = build_context([_result("AAPL:weekly:2024-01-01", "AAPL rose 3% this week.")])

    assert "[AAPL | weekly | 2024-01-01 to 2024-01-05]" in context
    assert "AAPL rose 3% this week." in context


def test_build_context_handles_no_results():
    context = build_context([])
    assert "No matching documents" in context


def test_build_prompt_includes_question_and_context():
    prompt = build_prompt(
        "How did AAPL do?", [_result("AAPL:weekly:2024-01-01", "AAPL rose 3% this week.")]
    )

    assert "QUESTION: How did AAPL do?" in prompt
    assert "AAPL rose 3% this week." in prompt


# --- Day 6: regression guards for the two prompt-level failure-mode fixes ---
# (see eval/failure_cases.py's "unsupported-ticker" and "tool-call-as-text"
# cases). These only check the instructions text is present, not that a
# real model obeys it -- that still needs eval/run_failure_cases.py against
# a live model.


def test_system_instructions_tell_model_not_to_substitute_unsupported_companies():
    lowered = SYSTEM_INSTRUCTIONS.lower()
    assert "isn't covered" in lowered
    assert "substitute" in lowered


def test_system_instructions_forbid_writing_tool_calls_as_plain_text():
    lowered = SYSTEM_INSTRUCTIONS.lower()
    assert "tool-calling mechanism" in lowered
