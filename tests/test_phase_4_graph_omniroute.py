from __future__ import annotations

import socket
import uuid
from typing import Any

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tracers.run_collector import RunCollectorCallbackHandler

from hrbot.ask import main as ask_cli_main
from hrbot.config import Settings
from hrbot.graph import (
    AskResult,
    Citation,
    ask,
    build_graph,
    format_escalation_message,
)
from hrbot.prompts import SYSTEM_PROMPT, build_messages, format_context
from hrbot.retriever import RetrievedChunk


class FakeChatModel(BaseChatModel):
    """Offline fake chat model for testing generation, citations, and retries."""

    responses: list[str | Exception]
    calls: list[list[BaseMessage]]

    def __init__(self, responses: list[str | Exception]) -> None:
        super().__init__(responses=responses, calls=[])
        self.responses = list(responses)
        self.calls = []

    @property
    def _llm_type(self) -> str:
        return "fake-hrbot-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(messages)
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        item = self.responses[idx]
        if isinstance(item, Exception):
            raise item
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=str(item)))]
        )


def _make_settings(
    threshold: float = 0.68,
    model: str = "test-free-model",
    hr_email: str = "hr@example.com",
) -> Settings:
    return Settings(
        OMNIROUTE_API_KEY="fake-key",
        LLM_MODEL=model,
        RELEVANCE_THRESHOLD=threshold,
        HR_CONTACT_EMAIL=hr_email,
        LANGSMITH_TRACING=False,
        _env_file=None,
    )


def _sample_chunks(scores: tuple[float, float] = (0.82, 0.74)) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            text="Employees may carry over up to 5 unused vacation days into the next calendar year.",
            source="annual_leave.pdf",
            page=43,
            score=scores[0],
        ),
        RetrievedChunk(
            text="Educational assistance reimburses eligible tuition up to $3,000 per calendar year.",
            source="education_policy.pdf",
            page=48,
            score=scores[1],
        ),
    ]


def test_ac_4_1_build_graph_and_ask_result() -> None:
    """AC-4.1: build_graph() returns a compiled graph; ask() returns AskResult."""
    settings = _make_settings()
    chunks = _sample_chunks()
    fake_llm = FakeChatModel(["You can carry over up to 5 vacation days [1]."])

    graph = build_graph(
        settings=settings,
        llm=fake_llm,
        retriever_fn=lambda q, k: chunks,
    )
    assert hasattr(graph, "invoke")

    result = ask(
        "How many vacation days can I carry over?",
        run_id=str(uuid.uuid4()),
        settings=settings,
        graph=graph,
    )
    assert isinstance(result, AskResult)
    assert result.route == "answer"
    assert result.answer == "You can carry over up to 5 vacation days [1]."
    assert result.error is None
    assert result.chunks == chunks
    assert len(result.citations) == 1
    assert result.citations[0] == {
        "n": 1,
        "source": "annual_leave.pdf",
        "page": 43,
        "snippet": "Employees may carry over up to 5 unused vacation days into the next calendar year.",
    }


def test_ac_4_2_prompt_instructions_and_context_numbering() -> None:
    """AC-4.2: Prompt instructs grounded answers, plain admission when missing, brevity, and [n] citations."""
    chunks = _sample_chunks()
    context_str = format_context(chunks)
    assert "[1] (Source: annual_leave.pdf, Page: 43)" in context_str
    assert "[2] (Source: education_policy.pdf, Page: 48)" in context_str

    messages = build_messages("What is the tuition limit?", chunks)
    assert len(messages) == 2
    sys_text = str(messages[0].content).lower()
    user_text = str(messages[1].content)

    assert "only" in sys_text and "numbered context" in sys_text
    assert "does not contain the answer" in sys_text
    assert "short" in sys_text
    assert "[1]" in SYSTEM_PROMPT or "[n]" in user_text
    assert "What is the tuition limit?" in user_text


def test_ac_4_3_cite_parses_valid_drops_invalid_and_flags_missing() -> None:
    """AC-4.3: cite parses [n] into {n, source, page, snippet}, drops invalid n, sets citations_missing when none remain."""
    settings = _make_settings()
    chunks = _sample_chunks()

    # Case 1: valid [2] and [1], duplicate [2], and non-existent [99] and [0]
    llm_with_mixed = FakeChatModel(
        ["Tuition is $3,000 [2] and carryover is 5 days [1] (see also [99], [0], [2])."]
    )
    res_valid = ask(
        "Tell me about tuition and carryover.",
        settings=settings,
        llm=llm_with_mixed,
        retriever_fn=lambda q, k: chunks,
    )
    assert res_valid.route == "answer"
    assert res_valid.citations_missing is False
    assert [c["n"] for c in res_valid.citations] == [2, 1]
    assert isinstance(res_valid.citations[0], Citation)
    assert set(res_valid.citations[0].keys()) == {"n", "source", "page", "snippet"}
    assert res_valid.citations[0]["source"] == "education_policy.pdf"
    assert res_valid.citations[0]["page"] == 48

    # Case 2: only out-of-range markers [9] -> dropped -> citations_missing=True, route stays "answer"
    llm_no_valid = FakeChatModel(["Employees receive 5 carryover days [9]."])
    res_missing = ask(
        "How many carryover days?",
        settings=settings,
        llm=llm_no_valid,
        retriever_fn=lambda q, k: chunks,
    )
    assert res_missing.route == "answer"
    assert res_missing.citations == []
    assert res_missing.citations_missing is True


def test_ac_4_4_low_relevance_routes_to_escalate_without_llm_call() -> None:
    """AC-4.4: Low relevance or empty retrieval routes to escalate with fixed message and 0 LLM calls."""
    settings = _make_settings(threshold=0.68, hr_email="people-ops@company.test")
    low_chunks = _sample_chunks(scores=(0.55, 0.42))
    fake_llm = FakeChatModel(["Should never be called [1]."])

    res_low = ask(
        "Do we get stock options?",
        settings=settings,
        llm=fake_llm,
        retriever_fn=lambda q, k: low_chunks,
    )
    assert res_low.route == "escalate"
    assert res_low.answer == format_escalation_message("people-ops@company.test")
    assert res_low.citations == []
    assert res_low.error is None
    assert len(fake_llm.calls) == 0

    res_empty = ask(
        "Do we get stock options?",
        settings=settings,
        llm=fake_llm,
        retriever_fn=lambda q, k: [],
    )
    assert res_empty.route == "escalate"
    assert len(fake_llm.calls) == 0


def test_ac_4_5_llm_errors_retried_twice_then_escalate_no_raise() -> None:
    """AC-4.5: LLM errors retry up to 2 times with backoff, then route to escalate with error set without raising."""
    settings = _make_settings()
    chunks = _sample_chunks()

    # Subcase A: 2 transient errors (TimeoutError, 429 rate limit) followed by success on 3rd attempt
    sleep_delays_a: list[float] = []
    flaky_llm = FakeChatModel(
        [
            TimeoutError("Request timed out"),
            RuntimeError("HTTP 429 Too Many Requests"),
            "Carryover is capped at 5 days [1].",
        ]
    )
    res_recovered = ask(
        "Carryover limit?",
        settings=settings,
        llm=flaky_llm,
        retriever_fn=lambda q, k: chunks,
        backoff_fn=sleep_delays_a.append,
    )
    assert res_recovered.route == "answer"
    assert res_recovered.error is None
    assert "5 days [1]" in res_recovered.answer
    assert len(flaky_llm.calls) == 3
    assert len(sleep_delays_a) == 2
    assert sleep_delays_a[1] > sleep_delays_a[0] > 0

    # Subcase B: All 3 attempts fail (initial + 2 retries) -> escalates with error, does not raise
    sleep_delays_b: list[float] = []
    failing_llm = FakeChatModel(
        [
            ConnectionError("Connection refused"),
            TimeoutError("Timed out"),
            RuntimeError("429 rate limit exceeded"),
        ]
    )
    res_failed = ask(
        "Carryover limit?",
        settings=settings,
        llm=failing_llm,
        retriever_fn=lambda q, k: chunks,
        backoff_fn=sleep_delays_b.append,
    )
    assert res_failed.route == "escalate"
    assert res_failed.error is not None
    assert "429 rate limit exceeded" in res_failed.error
    assert res_failed.citations == []
    assert len(failing_llm.calls) == 3
    assert len(sleep_delays_b) == 2


def test_ac_4_6_langsmith_trace_config_name_tag_and_run_id() -> None:
    """AC-4.6: Each ask produces exactly one root trace named 'hr_policy_qa', tagged with model name and run_id."""
    settings = _make_settings(model="omniroute/gemini-free")
    chunks = _sample_chunks()
    fake_llm = FakeChatModel(["Answer with citation [1]."])

    collector = RunCollectorCallbackHandler()
    custom_run_id = uuid.uuid4()
    result = ask(
        "How many days can I carry over?",
        run_id=str(custom_run_id),
        settings=settings,
        llm=fake_llm,
        retriever_fn=lambda q, k: chunks,
        callbacks=[collector],
    )

    assert result.route == "answer"
    assert len(collector.traced_runs) == 1
    root_run = collector.traced_runs[0]
    assert root_run.name == "hr_policy_qa"
    assert "omniroute/gemini-free" in (root_run.tags or [])
    assert root_run.id == custom_run_id
    child_names = [child.name for child in root_run.child_runs]
    assert child_names == ["retrieve", "grade", "generate", "cite"]


def test_ac_4_7_empty_or_whitespace_question_rejected_early() -> None:
    """AC-4.7: Empty or whitespace-only questions are rejected before any retrieval or LLM call."""
    settings = _make_settings()
    retriever_calls: list[str] = []
    fake_llm = FakeChatModel(["Should not run [1]."])

    def spy_retriever(q: str, k: int) -> list[RetrievedChunk]:
        retriever_calls.append(q)
        return _sample_chunks()

    for blank in ("", "   ", "\n\t  "):
        res = ask(
            blank,
            settings=settings,
            llm=fake_llm,
            retriever_fn=spy_retriever,
        )
        assert res.route == "escalate"
        assert res.chunks == []
        assert res.citations == []
        assert res.error is not None

    assert retriever_calls == []
    assert fake_llm.calls == []


def test_ac_4_8_cli_ask_prints_answer_and_sources(capsys: pytest.CaptureFixture[str]) -> None:
    """AC-4.8: `python -m hrbot.ask "question"` prints the answer and its sources."""
    settings = _make_settings()
    chunks = _sample_chunks()
    fake_llm = FakeChatModel(
        ["Employees can carry over up to 5 unused vacation days [1]."]
    )

    exit_code = ask_cli_main(
        ["How many vacation days can I carry over?"],
        settings=settings,
        llm=fake_llm,
        retriever_fn=lambda q, k: chunks,
    )
    assert exit_code == 0
    captured = capsys.readouterr().out
    assert "Employees can carry over up to 5 unused vacation days [1]." in captured
    assert "Sources:" in captured
    assert "[1] annual_leave.pdf (page 43)" in captured


def test_ac_4_9_unit_tests_use_fake_llm_and_fake_retriever_offline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC-4.9: Pipeline runs completely offline with fake chat model and fake retriever when sockets are blocked."""

    def _block_socket(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("Network access is forbidden in unit tests")

    monkeypatch.setattr(socket, "socket", _block_socket)

    settings = _make_settings()
    chunks = _sample_chunks()
    fake_llm = FakeChatModel(["Tuition reimbursement is up to $3,000 per year [2]."])

    result = ask(
        "What is the tuition reimbursement limit?",
        settings=settings,
        llm=fake_llm,
        retriever_fn=lambda q, k: chunks,
    )
    assert result.route == "answer"
    assert len(result.citations) == 1
    assert result.citations[0].source == "education_policy.pdf"
    assert result.citations[0].page == 48
