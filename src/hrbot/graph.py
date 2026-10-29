from __future__ import annotations

import os
import re
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from hrbot.config import Settings, setup_environment
from hrbot.llm import get_llm
from hrbot.prompts import build_messages
from hrbot.retriever import RetrievedChunk, retrieve

CITATION_REGEX = re.compile(r"\[(\d+)\]")
THINK_BLOCK_REGEX = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
SNIPPET_MAX_CHARS = 240


@dataclass
class Citation:
    """A single parsed citation mapping a [n] marker to its retrieved chunk."""

    n: int
    source: str
    page: int
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "source": self.source,
            "page": self.page,
            "snippet": self.snippet,
        }

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def keys(self) -> Any:
        return self.to_dict().keys()

    def values(self) -> Any:
        return self.to_dict().values()

    def items(self) -> Any:
        return self.to_dict().items()

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return 4

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Citation):
            return self.to_dict() == other.to_dict()
        if isinstance(other, dict):
            return self.to_dict() == other
        return NotImplemented


@dataclass
class AskResult:
    """Result returned by hrbot.graph.ask()."""

    answer: str
    citations: list[Citation] = field(default_factory=list)
    route: Literal["answer", "escalate"] = "escalate"
    chunks: list[RetrievedChunk] = field(default_factory=list)
    error: str | None = None
    citations_missing: bool = False


class GraphState(TypedDict, total=False):
    question: str
    chunks: list[RetrievedChunk]
    best_score: float
    route: Literal["answer", "escalate"]
    answer: str
    citations: list[Citation]
    citations_missing: bool
    error: str | None


RetrieverFn = Callable[[str, int], list[RetrievedChunk]]


def _resolve_settings(settings: Settings | None = None) -> Settings:
    """Resolve Settings, falling back to safe defaults if env vars are missing in offline tests."""
    if settings is not None:
        return settings
    try:
        return Settings()
    except ValidationError:
        return Settings(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY") or "unused",
            LLM_MODEL=os.environ.get("LLM_MODEL") or "unused",
        )


def format_escalation_message(hr_email: str) -> str:
    """Return the fixed escalation message when policy retrieval is insufficient or unavailable."""
    return (
        "I couldn't find a confident answer to your question in the HR policy documents. "
        f"Please contact HR directly at {hr_email} for assistance."
    )


def _make_snippet(text: str, max_chars: int = SNIPPET_MAX_CHARS) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 3].rstrip() + "..."


def parse_citations(
    answer: str,
    chunks: list[RetrievedChunk],
) -> tuple[list[Citation], bool]:
    """Parse [n] markers from answer into Citation objects.

    Markers referencing a non-existent chunk index (n < 1 or n > len(chunks))
    are dropped. Duplicate markers keep their first appearance order.
    Returns (citations, citations_missing).
    """
    matches = CITATION_REGEX.findall(answer or "")
    seen: set[int] = set()
    citations: list[Citation] = []

    for match in matches:
        n = int(match)
        if n < 1 or n > len(chunks) or n in seen:
            continue
        seen.add(n)
        chunk = chunks[n - 1]
        citations.append(
            Citation(
                n=n,
                source=chunk.source,
                page=chunk.page,
                snippet=_make_snippet(chunk.text),
            )
        )

    citations_missing = len(citations) == 0
    return citations, citations_missing


def _extract_text_content(response: Any) -> str:
    """Extract plain text from a LangChain message or string response."""
    if isinstance(response, BaseMessage) or hasattr(response, "content"):
        content = response.content
    else:
        content = str(response)

    if isinstance(content, list):
        parts = [
            item.get("text", "") if isinstance(item, dict) else str(item)
            for item in content
        ]
        raw_text = "".join(parts)
    else:
        raw_text = str(content)
    return THINK_BLOCK_REGEX.sub("", raw_text).strip()


def build_graph(
    settings: Settings | None = None,
    llm: BaseChatModel | Callable[[list[BaseMessage]], Any] | None = None,
    retriever_fn: RetrieverFn | None = None,
    backoff_fn: Callable[[float], None] = time.sleep,
    max_retries: int = 2,
    initial_backoff_seconds: float = 0.5,
) -> CompiledStateGraph:
    """Build and compile the HR policy QA LangGraph pipeline."""
    resolved_settings = _resolve_settings(settings)

    def retrieve_node(state: GraphState) -> GraphState:
        question = state["question"]
        if retriever_fn is not None:
            chunks = retriever_fn(question, resolved_settings.TOP_K)
        else:
            chunks = retrieve(
                query=question,
                k=resolved_settings.TOP_K,
                settings=resolved_settings,
            )
        best_score = max((c.score for c in chunks), default=0.0)
        return {
            "chunks": chunks,
            "best_score": best_score,
        }

    def grade_node(state: GraphState) -> GraphState:
        chunks = state.get("chunks") or []
        best_score = state.get("best_score", 0.0)
        if len(chunks) > 0 and best_score >= resolved_settings.RELEVANCE_THRESHOLD:
            route: Literal["answer", "escalate"] = "answer"
        else:
            route = "escalate"
        return {"route": route}

    def route_after_grade(state: GraphState) -> Literal["generate", "escalate"]:
        if state.get("route") == "answer":
            return "generate"
        return "escalate"

    def generate_node(state: GraphState) -> GraphState:
        question = state["question"]
        chunks = state.get("chunks") or []
        messages = build_messages(question, chunks)

        model = llm if llm is not None else get_llm(resolved_settings)
        last_error: Exception | None = None

        for attempt in range(max_retries + 1):
            try:
                if hasattr(model, "invoke"):
                    response = model.invoke(messages)
                else:
                    response = model(messages)
                answer_text = _extract_text_content(response)
                return {
                    "answer": answer_text,
                    "error": None,
                }
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt < max_retries:
                    delay = initial_backoff_seconds * (2**attempt)
                    backoff_fn(delay)

        error_msg = f"{type(last_error).__name__}: {last_error}" if last_error else "LLM error"
        return {
            "route": "escalate",
            "error": error_msg,
        }

    def route_after_generate(state: GraphState) -> Literal["cite", "escalate"]:
        if state.get("error") is not None or state.get("route") == "escalate":
            return "escalate"
        return "cite"

    def cite_node(state: GraphState) -> GraphState:
        answer = state.get("answer", "")
        chunks = state.get("chunks") or []
        citations, citations_missing = parse_citations(answer, chunks)
        return {
            "route": "answer",
            "citations": citations,
            "citations_missing": citations_missing,
        }

    def escalate_node(state: GraphState) -> GraphState:
        return {
            "route": "escalate",
            "answer": format_escalation_message(resolved_settings.HR_CONTACT_EMAIL),
            "citations": [],
            "citations_missing": False,
            "error": state.get("error"),
        }

    builder = StateGraph(GraphState)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("grade", grade_node)
    builder.add_node("generate", generate_node)
    builder.add_node("cite", cite_node)
    builder.add_node("escalate", escalate_node)

    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "grade")
    builder.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "escalate": "escalate"},
    )
    builder.add_conditional_edges(
        "generate",
        route_after_generate,
        {"cite": "cite", "escalate": "escalate"},
    )
    builder.add_edge("cite", END)
    builder.add_edge("escalate", END)

    return builder.compile()


def _normalize_run_id(run_id: str | uuid.UUID | None) -> uuid.UUID | None:
    if run_id is None:
        return None
    if isinstance(run_id, uuid.UUID):
        return run_id
    try:
        return uuid.UUID(str(run_id))
    except ValueError:
        return uuid.uuid5(uuid.NAMESPACE_OID, str(run_id))


def ask(
    question: str,
    run_id: str | uuid.UUID | None = None,
    *,
    settings: Settings | None = None,
    graph: CompiledStateGraph | None = None,
    llm: BaseChatModel | Callable[[list[BaseMessage]], Any] | None = None,
    retriever_fn: RetrieverFn | None = None,
    backoff_fn: Callable[[float], None] = time.sleep,
    callbacks: list[Any] | None = None,
) -> AskResult:
    """Answer an HR policy question using the LangGraph pipeline."""
    resolved_settings = _resolve_settings(settings)

    if not isinstance(question, str) or not question.strip():
        return AskResult(
            answer="Please enter a non-empty question.",
            citations=[],
            route="escalate",
            chunks=[],
            error="Empty or whitespace-only question.",
            citations_missing=False,
        )

    setup_environment(resolved_settings)

    compiled_graph = (
        graph
        if graph is not None
        else build_graph(
            settings=resolved_settings,
            llm=llm,
            retriever_fn=retriever_fn,
            backoff_fn=backoff_fn,
        )
    )

    config: RunnableConfig = {
        "run_name": "hr_policy_qa",
        "tags": [resolved_settings.LLM_MODEL],
    }
    normalized_run_id = _normalize_run_id(run_id)
    if normalized_run_id is not None:
        config["run_id"] = normalized_run_id
    if callbacks is not None:
        config["callbacks"] = callbacks

    initial_state: GraphState = {
        "question": question.strip(),
        "chunks": [],
        "best_score": 0.0,
        "route": "escalate",
        "answer": "",
        "citations": [],
        "citations_missing": False,
        "error": None,
    }

    final_state: GraphState = compiled_graph.invoke(initial_state, config=config)

    return AskResult(
        answer=final_state.get("answer", ""),
        citations=list(final_state.get("citations") or []),
        route=final_state.get("route", "escalate"),
        chunks=list(final_state.get("chunks") or []),
        error=final_state.get("error"),
        citations_missing=bool(final_state.get("citations_missing", False)),
    )
