"""Streamlit frontend for Market Narrator.

Deliberately a thin HTTP client over the FastAPI service (src/finrag/api/)
and nothing more -- no retrieval, no LLM calls, no database access happen
in this file. That split (User -> Streamlit -> FastAPI -> Postgres/Groq,
per docs/PROJECT_PLAN.md's architecture diagram) means the UI can be
restarted, redeployed, or replaced independently of the RAG pipeline
itself, and it's the FastAPI service -- not this script -- that's
responsible for logging, error handling, and being the single source of
truth for "what Market Narrator can do."

Run with:
    uv run streamlit run ui/app.py
"""

from __future__ import annotations

import httpx
import streamlit as st

from finrag.config.settings import get_settings

st.set_page_config(page_title="Market Narrator", page_icon="\U0001f4c8", layout="centered")

EXAMPLE_QUESTIONS = [
    "How did AAPL perform in 2022?",
    "Compare MSFT and NVDA over the last year.",
    "What was Tesla's exact return from 2022-01-03 to 2022-06-30?",
    "Summarize AMD's 2021.",
]


def _api_base_url() -> str:
    return get_settings().api_base_url.rstrip("/")


def _call_api(method: str, path: str, **kwargs) -> httpx.Response | None:
    """Thin wrapper around httpx that turns connection/timeout errors into
    a Streamlit error message instead of an unhandled traceback -- the
    most likely failure mode for a user running the UI without the API
    (or the API without the DB/Groq) up yet, and it should be obvious to
    them what's wrong.
    """
    try:
        with httpx.Client(base_url=_api_base_url(), timeout=60.0) as client:
            response = client.request(method, path, **kwargs)
        response.raise_for_status()
        return response
    except httpx.ConnectError:
        st.error(
            f"Can't reach the Market Narrator API at {_api_base_url()}. "
            "Is it running? (`uv run uvicorn finrag.api.main:app` or `docker compose up`)"
        )
    except httpx.HTTPStatusError as exc:
        detail = exc.response.json().get("detail", exc.response.text)
        st.error(f"API error ({exc.response.status_code}): {detail}")
    except httpx.TimeoutException:
        st.error("The request timed out. The LLM/tool-calling step can take a while -- try again.")
    return None


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Covered tickers")
        response = _call_api("GET", "/tickers")
        if response is not None:
            for group in response.json()["tickers"]:
                st.caption(group["sector"])
                st.write(", ".join(group["tickers"]))

        st.divider()
        st.header("Example questions")
        for question in EXAMPLE_QUESTIONS:
            if st.button(question, use_container_width=True):
                st.session_state["question_input"] = question
                st.rerun()


def _render_answer(result: dict) -> None:
    st.markdown(f"**Q: {result['question']}**")
    if result["resolved_tickers"]:
        st.caption(f"Resolved tickers: {', '.join(result['resolved_tickers'])}")
    st.write(result["answer"])
    st.caption(f"Answered in {result['latency_ms']:,} ms")

    with st.expander(f"Retrieved context ({len(result['retrieved'])} chunks)"):
        for chunk in result["retrieved"]:
            st.markdown(
                f"`{chunk['doc_id']}` — {chunk['ticker']} ({chunk['granularity']}, "
                f"{chunk['period_start']} to {chunk['period_end']}), score={chunk['score']:.3f}"
            )

    query_id = result["query_id"]
    if query_id == -1:
        st.caption("(Feedback unavailable -- this query wasn't logged.)")
        return

    col1, col2, _ = st.columns([1, 1, 6])
    feedback_key = f"feedback_sent_{query_id}"
    already_sent = st.session_state.get(feedback_key)

    up_clicked = col1.button("\U0001f44d", key=f"up_{query_id}", disabled=bool(already_sent))
    if up_clicked and _call_api("POST", f"/feedback/{query_id}", json={"feedback": 1}) is not None:
        st.session_state[feedback_key] = "up"
        st.rerun()

    down_clicked = col2.button("\U0001f44e", key=f"down_{query_id}", disabled=bool(already_sent))
    if down_clicked and _call_api("POST", f"/feedback/{query_id}", json={"feedback": -1}) is not None:
        st.session_state[feedback_key] = "down"
        st.rerun()
    if already_sent:
        st.caption(f"Thanks for the feedback ({already_sent}).")


def main() -> None:
    st.title("\U0001f4c8 Market Narrator")
    st.caption(
        "Ask about historical stock behavior. Answers are grounded in real price "
        "data -- narrative context for trends, deterministic tool calls for exact numbers."
    )

    _render_sidebar()

    question = st.text_input(
        "Ask a question", key="question_input", placeholder="How did Apple perform in 2022?"
    )
    top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)

    if st.button("Ask", type="primary") and question.strip():
        with st.spinner("Retrieving context and generating an answer..."):
            response = _call_api("POST", "/ask", json={"question": question, "top_k": top_k})
        if response is not None:
            st.session_state["last_result"] = response.json()

    if "last_result" in st.session_state:
        st.divider()
        _render_answer(st.session_state["last_result"])


if __name__ == "__main__":
    main()
