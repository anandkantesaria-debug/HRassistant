from __future__ import annotations

import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from hrbot.config import Settings
from hrbot.llm import get_judge_llm

SCORE_LINE_REGEX = re.compile(r"SCORE\s*:\s*(1(?:\.0+)?|0(?:\.0+)?|PASS|FAIL|YES|NO)", re.IGNORECASE)
REASON_LINE_REGEX = re.compile(r"REASON\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)
THINK_BLOCK_REGEX = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
RETRY_DELAY_REGEX = re.compile(r"retry in\s+(\d+(?:\.\d+)?)\s*s", re.IGNORECASE)

CORRECTNESS_SYSTEM_PROMPT = (
    "You are an impartial HR policy evaluation judge.\n"
    "Compare the Assistant Answer against the Reference Answer for the Employee Question.\n"
    "Rules:\n"
    "1. Score 1 if the Assistant Answer matches the Reference Answer in substance "
    "(including key numbers, conditions, or appropriate escalation when the topic is not in the policy).\n"
    "2. Score 0 if the Assistant Answer contradicts the Reference Answer, omits critical facts/numbers "
    "present in the Reference Answer, or fabricates policy rules.\n"
    "3. Respond in plain text using EXACTLY this two-line format:\n"
    "SCORE: <1 or 0>\n"
    "REASON: <one concise sentence>"
)

GROUNDEDNESS_SYSTEM_PROMPT = (
    "You are an impartial groundedness judge for an HR policy RAG system.\n"
    "Determine whether every factual policy claim in the Assistant Answer is supported by the Retrieved Context.\n"
    "Rules:\n"
    "1. Score 1 if every factual claim in the Assistant Answer is directly supported by the Retrieved Context, "
    "OR if the Assistant Answer is a standard escalation/refusal message that makes no policy claims.\n"
    "2. Score 0 if the Assistant Answer contains any unsupported claim, hallucinated number, or extrinsic assertion "
    "not present in the Retrieved Context.\n"
    "3. Respond in plain text using EXACTLY this two-line format:\n"
    "SCORE: <1 or 0>\n"
    "REASON: <one concise sentence>"
)


def _extract_dict(obj: Any, attr: str) -> dict[str, Any]:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        if attr in obj and isinstance(obj[attr], dict):
            return obj[attr]
        return obj
    val = getattr(obj, attr, None)
    return val if isinstance(val, dict) else {}


def _extract_chunk_source(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("source", ""))
    return str(getattr(chunk, "source", ""))


def _extract_chunk_page(chunk: Any) -> int | None:
    if isinstance(chunk, dict):
        val = chunk.get("page")
    else:
        val = getattr(chunk, "page", None)
    return int(val) if val is not None else None


def _extract_chunk_text(chunk: Any) -> str:
    if isinstance(chunk, dict):
        return str(chunk.get("text", ""))
    return str(getattr(chunk, "text", ""))


def _normalize_source_name(source: str | None) -> str:
    if not source:
        return ""
    return Path(str(source)).name.strip().lower()


def _extract_text_content(response: Any) -> str:
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


def parse_judge_response(text: str) -> tuple[float, str]:
    """Parse plain-text judge output (`SCORE: 1|0` and `REASON: ...`) into `(score, reason)`."""
    cleaned = (text or "").strip()
    score = 0.0

    match = SCORE_LINE_REGEX.search(cleaned)
    if match:
        token = match.group(1).strip().upper()
        if token in {"1", "1.0", "PASS", "YES"}:
            score = 1.0
        else:
            score = 0.0
    else:
        first_line = cleaned.splitlines()[0].strip() if cleaned else ""
        if first_line in {"1", "1.0"}:
            score = 1.0

    reason_match = REASON_LINE_REGEX.search(cleaned)
    if reason_match:
        reason = " ".join(reason_match.group(1).strip().split())
    else:
        reason = " ".join(cleaned.split()) or "No reason provided."

    return score, reason


def _retry_delay_from_error(exc: Exception) -> float | None:
    """Extract the server-suggested retry delay (e.g. 'Please retry in 28.5s') from a rate-limit error."""
    match = RETRY_DELAY_REGEX.search(str(exc))
    return float(match.group(1)) if match else None


def _invoke_judge(
    messages: list[BaseMessage],
    judge_llm: BaseChatModel | Callable[[list[BaseMessage]], Any] | None = None,
    settings: Settings | None = None,
    *,
    max_retries: int = 3,
    initial_backoff_seconds: float = 1.0,
    backoff_fn: Callable[[float], None] = time.sleep,
) -> tuple[float, str]:
    model = judge_llm if judge_llm is not None else get_judge_llm(settings)
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            if hasattr(model, "invoke"):
                raw = model.invoke(messages)
            else:
                raw = model(messages)
            text = _extract_text_content(raw)
            return parse_judge_response(text)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < max_retries:
                delay = initial_backoff_seconds * (2**attempt)
                server_delay = _retry_delay_from_error(exc)
                if server_delay is not None:
                    delay = max(delay, server_delay + 1.0)
                backoff_fn(delay)

    if last_error is not None:
        raise last_error
    raise RuntimeError("Judge LLM invocation failed")


def retrieval_hit(run: Any, example: Any) -> dict[str, Any]:
    """AC-7.4: Expected source appears in retrieved chunks (answerable items only)."""
    ex_outputs = _extract_dict(example, "outputs")
    answerable = bool(ex_outputs.get("answerable", True))
    if not answerable:
        return {
            "key": "retrieval_hit",
            "score": None,
            "comment": "Unanswerable item; excluded from retrieval_hit.",
        }

    expected_source = _normalize_source_name(ex_outputs.get("expected_source"))
    run_outputs = _extract_dict(run, "outputs")
    chunks = run_outputs.get("chunks") or []
    retrieved_sources = {_normalize_source_name(_extract_chunk_source(c)) for c in chunks}

    hit = bool(expected_source and expected_source in retrieved_sources)
    return {
        "key": "retrieval_hit",
        "score": 1.0 if hit else 0.0,
        "comment": (
            f"Found '{expected_source}' in retrieved chunks."
            if hit
            else f"Expected '{expected_source}', got {sorted(retrieved_sources)}."
        ),
    }


def citation_present(run: Any, example: Any) -> dict[str, Any]:
    """AC-7.4: Answer has at least one valid citation (answerable items only)."""
    ex_outputs = _extract_dict(example, "outputs")
    answerable = bool(ex_outputs.get("answerable", True))
    if not answerable:
        return {
            "key": "citation_present",
            "score": None,
            "comment": "Unanswerable item; excluded from citation_present.",
        }

    run_outputs = _extract_dict(run, "outputs")
    citations = run_outputs.get("citations") or []
    citations_missing = bool(run_outputs.get("citations_missing", False))
    has_valid_citation = len(citations) > 0 and not citations_missing

    return {
        "key": "citation_present",
        "score": 1.0 if has_valid_citation else 0.0,
        "comment": (
            f"{len(citations)} valid citation(s) present."
            if has_valid_citation
            else "No valid citations present in answer."
        ),
    }


def correct_escalation(run: Any, example: Any) -> dict[str, Any]:
    """AC-7.4: Unanswerable items route to escalate; answerable items do not."""
    ex_outputs = _extract_dict(example, "outputs")
    answerable = bool(ex_outputs.get("answerable", True))
    run_outputs = _extract_dict(run, "outputs")
    route = str(run_outputs.get("route", ""))

    if answerable:
        ok = route != "escalate" and route == "answer"
    else:
        ok = route == "escalate"

    return {
        "key": "correct_escalation",
        "score": 1.0 if ok else 0.0,
        "comment": f"answerable={answerable}, route={route!r}",
    }


def correctness(
    run: Any,
    example: Any,
    *,
    judge_llm: BaseChatModel | Callable[[list[BaseMessage]], Any] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """AC-7.4 (LLM judge): Answer matches the reference answer in substance."""
    ex_inputs = _extract_dict(example, "inputs")
    ex_outputs = _extract_dict(example, "outputs")
    run_outputs = _extract_dict(run, "outputs")

    question = str(ex_inputs.get("question", ""))
    reference_answer = str(ex_outputs.get("reference_answer", ""))
    answerable = bool(ex_outputs.get("answerable", True))
    answer = str(run_outputs.get("answer", ""))
    route = str(run_outputs.get("route", ""))

    user_prompt = (
        f"Employee Question: {question}\n"
        f"Question Answerable from Policy: {answerable}\n"
        f"Reference Answer: {reference_answer}\n"
        f"Assistant Route: {route}\n"
        f"Assistant Answer: {answer}\n\n"
        "Does the Assistant Answer match the Reference Answer in substance?\n"
        "Reply with SCORE: <1 or 0> and REASON: <explanation>."
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=CORRECTNESS_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    score, reason = _invoke_judge(messages, judge_llm=judge_llm, settings=settings)
    return {
        "key": "correctness",
        "score": score,
        "comment": reason,
    }


def groundedness(
    run: Any,
    example: Any,
    *,
    judge_llm: BaseChatModel | Callable[[list[BaseMessage]], Any] | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """AC-7.4 (LLM judge): Every claim in the answer is supported by the retrieved context."""
    ex_inputs = _extract_dict(example, "inputs")
    run_outputs = _extract_dict(run, "outputs")

    question = str(ex_inputs.get("question", ""))
    answer = str(run_outputs.get("answer", ""))
    route = str(run_outputs.get("route", ""))
    chunks = run_outputs.get("chunks") or []

    if chunks:
        formatted_chunks = "\n\n".join(
            f"[{idx}] (Source: {_extract_chunk_source(c)}, Page: {_extract_chunk_page(c)})\n{_extract_chunk_text(c)}"
            for idx, c in enumerate(chunks, start=1)
        )
    else:
        formatted_chunks = "(No retrieved context chunks.)"

    user_prompt = (
        f"Employee Question: {question}\n"
        f"Retrieved Context:\n{formatted_chunks}\n\n"
        f"Assistant Route: {route}\n"
        f"Assistant Answer: {answer}\n\n"
        "Is every factual claim in the Assistant Answer supported by the Retrieved Context "
        "(or is it a standard escalation refusal making no policy claims)?\n"
        "Reply with SCORE: <1 or 0> and REASON: <explanation>."
    )
    messages: list[BaseMessage] = [
        SystemMessage(content=GROUNDEDNESS_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]
    score, reason = _invoke_judge(messages, judge_llm=judge_llm, settings=settings)
    return {
        "key": "groundedness",
        "score": score,
        "comment": reason,
    }


def build_evaluators(
    settings: Settings | None = None,
    judge_llm: BaseChatModel | Callable[[list[BaseMessage]], Any] | None = None,
) -> list[Callable[[Any, Any], dict[str, Any]]]:
    """Build the 5 evaluators required by AC-7.4."""

    def _correctness_eval(run: Any, example: Any) -> dict[str, Any]:
        return correctness(run, example, judge_llm=judge_llm, settings=settings)

    def _groundedness_eval(run: Any, example: Any) -> dict[str, Any]:
        return groundedness(run, example, judge_llm=judge_llm, settings=settings)

    _correctness_eval.__name__ = "correctness"
    _groundedness_eval.__name__ = "groundedness"

    return [
        retrieval_hit,
        citation_present,
        correct_escalation,
        _correctness_eval,
        _groundedness_eval,
    ]

