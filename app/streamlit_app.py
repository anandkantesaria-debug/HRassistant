from __future__ import annotations

import uuid
from typing import Any

import streamlit as st
from langchain_chroma import Chroma
from langgraph.graph.state import CompiledStateGraph

from hrbot.graph import Citation, RetrieverFn, ask, build_graph
from hrbot.llm import check_llm_reachable
from hrbot.retriever import get_chunk_count, get_vectorstore, make_retriever

EXAMPLE_QUESTIONS: tuple[str, ...] = (
    "How many days of paid annual leave do full-time employees receive?",
    "How many days per week can eligible employees work remotely?",
    "What is the daily meal reimbursement limit during business travel?",
)

FRIENDLY_ERROR_NOTICE = (
    "We encountered a temporary issue while processing your request. "
    "Please try again in a moment or contact HR directly."
)


@st.cache_resource
def get_retriever_resource() -> tuple[Chroma, RetrieverFn]:
    """Create and cache the Chroma vectorstore and retriever callable once."""
    vectorstore = get_vectorstore()
    retriever_fn = make_retriever(vectorstore=vectorstore)
    return vectorstore, retriever_fn


@st.cache_resource
def get_graph_resource() -> CompiledStateGraph:
    """Create and cache the compiled LangGraph pipeline once."""
    _, retriever_fn = get_retriever_resource()
    return build_graph(retriever_fn=retriever_fn)


def _serialize_citations(citations: list[Citation | dict[str, Any]]) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for cit in citations:
        if isinstance(cit, Citation):
            serialized.append(cit.to_dict())
        elif isinstance(cit, dict):
            serialized.append(
                {
                    "n": cit.get("n", 1),
                    "source": cit.get("source", ""),
                    "page": cit.get("page", 1),
                    "snippet": cit.get("snippet", ""),
                }
            )
    return serialized


def _render_sources_expander(citations: list[dict[str, Any]]) -> None:
    with st.expander("Sources"):
        if citations:
            for cit in citations:
                st.markdown(
                    f"**[{cit['n']}] {cit['source']}** — Page {cit['page']}\n\n"
                    f"> {cit['snippet']}"
                )
        else:
            st.caption("No policy sources cited for this response.")


def _render_assistant_message(message: dict[str, Any]) -> None:
    if message.get("error"):
        st.warning(FRIENDLY_ERROR_NOTICE)
    st.markdown(str(message.get("content", "")))
    _render_sources_expander(list(message.get("citations") or []))


def _render_sidebar(chunk_count: int, llm_ok: bool) -> str | None:
    selected_question: str | None = None

    with st.sidebar:
        st.header("Example Questions")
        for idx, question in enumerate(EXAMPLE_QUESTIONS):
            if st.button(question, key=f"example_q_{idx}", use_container_width=True):
                selected_question = question

        st.divider()
        if st.button("Clear chat", key="clear_chat_btn", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()

        st.divider()
        st.subheader("System Status")
        index_status = f"Yes ({chunk_count} chunks)" if chunk_count > 0 else "No (0 chunks)"
        llm_status = "Yes" if llm_ok else "No"
        st.markdown(f"- **Index loaded:** {index_status}")
        st.markdown(f"- **LLM endpoint reachable:** {llm_status}")

    return selected_question


def main() -> None:
    st.set_page_config(page_title="HR Policy Assistant", page_icon="📘")

    st.title("HR Policy Assistant")
    st.caption("Ask questions about company HR policies and get grounded answers with page citations.")
    st.info(
        "Disclaimer: Answers come from the HR policy documents and should be confirmed "
        "with HR for important decisions."
    )

    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    vectorstore, _ = get_retriever_resource()
    graph = get_graph_resource()

    chunk_count = get_chunk_count(vectorstore=vectorstore)
    llm_ok = check_llm_reachable()

    if chunk_count == 0:
        st.warning(
            "The HR policy index is currently empty. Run ingestion before asking questions:\n\n"
            "`uv run python -m hrbot.ingest --path data/policies`"
        )

    clicked_example = _render_sidebar(chunk_count=chunk_count, llm_ok=llm_ok)

    for message in st.session_state["messages"]:
        with st.chat_message(str(message.get("role", "assistant"))):
            if message.get("role") == "user":
                st.markdown(str(message.get("content", "")))
            else:
                _render_assistant_message(message)

    typed_question = st.chat_input("Ask a question about HR policies...")
    active_question = typed_question or clicked_example

    if active_question:
        user_msg: dict[str, Any] = {"role": "user", "content": active_question}
        st.session_state["messages"].append(user_msg)
        with st.chat_message("user"):
            st.markdown(active_question)

        run_id = str(uuid.uuid4())
        with st.chat_message("assistant"):
            with st.spinner("Searching HR policies and generating answer..."):
                try:
                    result = ask(active_question, run_id=run_id, graph=graph)
                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": result.answer,
                        "citations": _serialize_citations(list(result.citations)),
                        "route": result.route,
                        "error": FRIENDLY_ERROR_NOTICE if result.error else None,
                        "run_id": run_id,
                    }
                except Exception:  # noqa: BLE001
                    assistant_msg = {
                        "role": "assistant",
                        "content": (
                            "I'm unable to answer right now due to a temporary system issue. "
                            "Please try again shortly or contact HR directly."
                        ),
                        "citations": [],
                        "route": "escalate",
                        "error": FRIENDLY_ERROR_NOTICE,
                        "run_id": run_id,
                    }

            _render_assistant_message(assistant_msg)
            st.session_state["messages"].append(assistant_msg)


if __name__ == "__main__":
    main()
