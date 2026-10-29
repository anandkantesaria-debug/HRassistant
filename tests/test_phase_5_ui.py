from __future__ import annotations

import ast
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from hrbot.graph import AskResult, Citation

APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture(autouse=True)
def _clear_streamlit_cache() -> Any:
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _make_answer_result(
    answer: str = "Full-time employees receive 20 days of paid annual leave per year [1].",
    citations: list[Citation] | None = None,
    error: str | None = None,
) -> AskResult:
    if citations is None:
        citations = [
            Citation(
                n=1,
                source="annual_leave_policy.pdf",
                page=1,
                snippet="Full-time employees accrue 20 days of paid annual leave per calendar year.",
            )
        ]
    return AskResult(
        answer=answer,
        citations=citations,
        route="escalate" if error else "answer",
        chunks=[],
        error=error,
        citations_missing=len(citations) == 0,
    )


def test_ac_5_1_title_intro_and_disclaimer() -> None:
    """AC-5.1: Page shows a title, a one-line intro, and a disclaimer."""
    with (
        patch("hrbot.retriever.get_vectorstore", return_value=MagicMock()),
        patch("hrbot.retriever.make_retriever", return_value=MagicMock()),
        patch("hrbot.graph.build_graph", return_value=MagicMock()),
        patch("hrbot.retriever.get_chunk_count", return_value=15),
        patch("hrbot.llm.check_llm_reachable", return_value=True),
    ):
        at = AppTest.from_file(APP_PATH).run()

    assert len(at.exception) == 0
    assert len(at.title) == 1
    assert "HR Policy" in at.title[0].value

    assert len(at.caption) >= 1
    assert "HR policies" in at.caption[0].value

    assert len(at.info) == 1
    disclaimer = at.info[0].value.lower()
    assert "hr policy documents" in disclaimer
    assert "confirmed with hr" in disclaimer


def test_ac_5_2_chat_input_messages_and_stateless_calls() -> None:
    """AC-5.2: Chat uses st.chat_input and st.chat_message; messages in session_state for display only; each question answered independently."""
    recorded_questions: list[str] = []

    def fake_ask(question: str, run_id: str | None = None, **kwargs: Any) -> AskResult:
        recorded_questions.append(question)
        return _make_answer_result(answer=f"Answer for: {question} [1]")

    with (
        patch("hrbot.retriever.get_vectorstore", return_value=MagicMock()),
        patch("hrbot.retriever.make_retriever", return_value=MagicMock()),
        patch("hrbot.graph.build_graph", return_value=MagicMock()),
        patch("hrbot.retriever.get_chunk_count", return_value=15),
        patch("hrbot.llm.check_llm_reachable", return_value=True),
        patch("hrbot.graph.ask", side_effect=fake_ask),
    ):
        at = AppTest.from_file(APP_PATH).run()
        assert len(at.chat_input) == 1

        at.chat_input[0].set_value("Question one?").run()
        at.chat_input[0].set_value("Question two?").run()

    assert len(at.exception) == 0
    assert recorded_questions == ["Question one?", "Question two?"]
    assert len(at.chat_message) == 4
    messages = at.session_state["messages"]
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Question one?"
    assert messages[1]["role"] == "assistant"
    assert "Answer for: Question one?" in messages[1]["content"]
    assert messages[2]["role"] == "user"
    assert messages[2]["content"] == "Question two?"
    assert messages[3]["role"] == "assistant"
    assert "Answer for: Question two?" in messages[3]["content"]


def test_ac_5_3_graph_and_retriever_cached_once() -> None:
    """AC-5.3: The graph and retriever are created once with st.cache_resource."""
    mock_get_vs = MagicMock(return_value=MagicMock())
    mock_make_ret = MagicMock(return_value=MagicMock())
    mock_build_graph = MagicMock(return_value=MagicMock())

    with (
        patch("hrbot.retriever.get_vectorstore", mock_get_vs),
        patch("hrbot.retriever.make_retriever", mock_make_ret),
        patch("hrbot.graph.build_graph", mock_build_graph),
        patch("hrbot.retriever.get_chunk_count", return_value=12),
        patch("hrbot.llm.check_llm_reachable", return_value=True),
        patch("hrbot.graph.ask", return_value=_make_answer_result()),
    ):
        at = AppTest.from_file(APP_PATH).run()
        at.chat_input[0].set_value("First question?").run()
        at.chat_input[0].set_value("Second question?").run()

    assert len(at.exception) == 0
    assert mock_get_vs.call_count == 1
    assert mock_make_ret.call_count == 1
    assert mock_build_graph.call_count == 1


def test_ac_5_4_sources_expander_lists_doc_page_snippet() -> None:
    """AC-5.4: Every answer has a 'Sources' expander listing document, page and snippet for each citation."""
    citations = [
        Citation(
            n=1,
            source="remote_work_policy.pdf",
            page=2,
            snippet="Eligible employees may work remotely up to 3 days per week.",
        ),
        Citation(
            n=2,
            source="expense_reimbursement_policy.pdf",
            page=4,
            snippet="Daily meal reimbursement is capped at $75 per day.",
        ),
    ]
    result = _make_answer_result(
        answer="Remote work is up to 3 days [1] and meals are $75 [2].",
        citations=citations,
    )

    with (
        patch("hrbot.retriever.get_vectorstore", return_value=MagicMock()),
        patch("hrbot.retriever.make_retriever", return_value=MagicMock()),
        patch("hrbot.graph.build_graph", return_value=MagicMock()),
        patch("hrbot.retriever.get_chunk_count", return_value=20),
        patch("hrbot.llm.check_llm_reachable", return_value=True),
        patch("hrbot.graph.ask", return_value=result),
    ):
        at = AppTest.from_file(APP_PATH).run()
        at.chat_input[0].set_value("Tell me about remote work and meals").run()

    assert len(at.exception) == 0
    expanders = [exp for exp in at.expander if exp.label == "Sources"]
    assert len(expanders) == 1
    expander_markdown = "\n".join(md.value for md in expanders[0].markdown)
    assert "remote_work_policy.pdf" in expander_markdown
    assert "Page 2" in expander_markdown
    assert "Eligible employees may work remotely up to 3 days per week." in expander_markdown
    assert "expense_reimbursement_policy.pdf" in expander_markdown
    assert "Page 4" in expander_markdown
    assert "Daily meal reimbursement is capped at $75 per day." in expander_markdown


def test_ac_5_5_spinner_and_friendly_error_no_stack_trace() -> None:
    """AC-5.5: A spinner shows while the answer is generated. Errors appear as friendly messages, never stack traces."""
    source_code = Path(APP_PATH).read_text(encoding="utf-8")
    assert "st.spinner(" in source_code

    # Case 1: ask() returns an AskResult with error set
    error_result = _make_answer_result(
        answer="I couldn't find a confident answer. Please contact HR.",
        citations=[],
        error="ConnectionError: internal socket traceback details",
    )
    with (
        patch("hrbot.retriever.get_vectorstore", return_value=MagicMock()),
        patch("hrbot.retriever.make_retriever", return_value=MagicMock()),
        patch("hrbot.graph.build_graph", return_value=MagicMock()),
        patch("hrbot.retriever.get_chunk_count", return_value=10),
        patch("hrbot.llm.check_llm_reachable", return_value=False),
        patch("hrbot.graph.ask", return_value=error_result),
    ):
        at = AppTest.from_file(APP_PATH).run()
        at.chat_input[0].set_value("Will this show a friendly error?").run()

    assert len(at.exception) == 0
    warnings_text = "\n".join(w.value for w in at.warning)
    assert "temporary issue" in warnings_text.lower()
    assert "internal socket traceback details" not in warnings_text
    assert "Traceback" not in warnings_text

    # Case 2: ask() raises an unexpected runtime exception
    st.cache_resource.clear()
    with (
        patch("hrbot.retriever.get_vectorstore", return_value=MagicMock()),
        patch("hrbot.retriever.make_retriever", return_value=MagicMock()),
        patch("hrbot.graph.build_graph", return_value=MagicMock()),
        patch("hrbot.retriever.get_chunk_count", return_value=10),
        patch("hrbot.llm.check_llm_reachable", return_value=False),
        patch("hrbot.graph.ask", side_effect=RuntimeError("Boom internal crash")),
    ):
        at2 = AppTest.from_file(APP_PATH).run()
        at2.chat_input[0].set_value("Trigger unexpected crash").run()

    assert len(at2.exception) == 0
    all_text = "\n".join(m.value for m in at2.markdown) + "\n" + "\n".join(w.value for w in at2.warning)
    assert "temporary" in all_text.lower()
    assert "Boom internal crash" not in all_text
    assert "Traceback" not in all_text


def test_ac_5_6_sidebar_examples_clear_chat_and_status() -> None:
    """AC-5.6: Sidebar has clickable example questions, a 'Clear chat' button, and status block (chunk count + LLM reachable yes/no)."""
    asked: list[str] = []

    def fake_ask(question: str, run_id: str | None = None, **kwargs: Any) -> AskResult:
        asked.append(question)
        return _make_answer_result(answer=f"Answered: {question} [1]")

    with (
        patch("hrbot.retriever.get_vectorstore", return_value=MagicMock()),
        patch("hrbot.retriever.make_retriever", return_value=MagicMock()),
        patch("hrbot.graph.build_graph", return_value=MagicMock()),
        patch("hrbot.retriever.get_chunk_count", return_value=42),
        patch("hrbot.llm.check_llm_reachable", return_value=True),
        patch("hrbot.graph.ask", side_effect=fake_ask),
    ):
        at = AppTest.from_file(APP_PATH).run()

        sidebar_md = "\n".join(m.value for m in at.sidebar.markdown)
        assert "42 chunks" in sidebar_md
        assert "LLM endpoint reachable:" in sidebar_md
        assert "Yes" in sidebar_md

        # Click the first example question button in the sidebar
        example_btn = at.sidebar.button(key="example_q_0")
        example_btn.click().run()

        assert len(asked) == 1
        assert len(at.session_state["messages"]) == 2
        assert len(at.chat_message) == 2

        # Click the Clear chat button in the sidebar
        clear_btn = at.sidebar.button(key="clear_chat_btn")
        clear_btn.click().run()

        assert len(at.session_state["messages"]) == 0
        assert len(at.chat_message) == 0


def test_ac_5_7_empty_index_shows_ingestion_banner() -> None:
    """AC-5.7: If the Chroma index is empty, a banner explains how to run ingestion."""
    with (
        patch("hrbot.retriever.get_vectorstore", return_value=MagicMock()),
        patch("hrbot.retriever.make_retriever", return_value=MagicMock()),
        patch("hrbot.graph.build_graph", return_value=MagicMock()),
        patch("hrbot.retriever.get_chunk_count", return_value=0),
        patch("hrbot.llm.check_llm_reachable", return_value=False),
    ):
        at = AppTest.from_file(APP_PATH).run()

    assert len(at.exception) == 0
    assert len(at.warning) >= 1
    banner_text = "\n".join(w.value for w in at.warning)
    assert "empty" in banner_text.lower()
    assert "hrbot.ingest" in banner_text
    assert "data/policies" in banner_text


def test_ac_5_8_ui_contains_no_business_logic() -> None:
    """AC-5.8: The UI file contains no business logic: it only calls hrbot.graph.ask and renders results."""
    source_code = Path(APP_PATH).read_text(encoding="utf-8")
    tree = ast.parse(source_code)

    imported_modules: set[str] = set()
    called_names: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called_names.add(node.func.attr)

    # Disallow prompt construction, PDF parsing, text splitting, regex citation parsing, or direct LLM creation in UI
    forbidden_imports = {
        "fitz",
        "pymupdf",
        "langchain_openai",
        "langchain_text_splitters",
        "sentence_transformers",
        "hrbot.prompts",
        "hrbot.ingest",
    }
    assert imported_modules.isdisjoint(forbidden_imports)
    assert "ask" in called_names

