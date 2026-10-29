import os
import urllib.request
from functools import lru_cache
from typing import Any

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from hrbot.config import Settings, setup_environment


def _resolve_settings(settings: Settings | None = None) -> Settings:
    if settings is not None:
        return settings
    try:
        return Settings()
    except ValidationError:
        return Settings(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY") or "unused",
            LLM_MODEL=os.environ.get("LLM_MODEL") or "unused",
        )


def _uses_groq_qwen3_reasoning(settings: Settings) -> bool:
    """Return True when targeting a Qwen 3 reasoning model on Groq."""
    model_lower = settings.LLM_MODEL.lower()
    base_lower = settings.GROQ_BASE_URL.lower()
    return "groq.com" in base_lower and ("qwen3" in model_lower or "qwen/qwen3" in model_lower)


def get_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at Groq."""
    if settings is None:
        settings = Settings()

    setup_environment(settings)

    kwargs: dict[str, Any] = {
        "base_url": settings.GROQ_BASE_URL,
        "api_key": settings.GROQ_API_KEY,
        "model": settings.LLM_MODEL,
        "temperature": 0.0,
        "request_timeout": settings.LLM_TIMEOUT_SECONDS,
        "streaming": False,
    }
    if _uses_groq_qwen3_reasoning(settings):
        kwargs["extra_body"] = {"reasoning_format": "hidden"}

    return ChatOpenAI(**kwargs)


@lru_cache(maxsize=4)
def _get_judge_rate_limiter(requests_per_minute: float) -> InMemoryRateLimiter:
    """Return a process-wide rate limiter shared by all judge clients with the same RPM."""
    return InMemoryRateLimiter(
        requests_per_second=requests_per_minute / 60.0,
        check_every_n_seconds=0.1,
        max_bucket_size=1,
    )


def get_judge_llm(settings: Settings | None = None) -> ChatOpenAI:
    """Return a ChatOpenAI client pointed at Google Gemini API with EVAL_JUDGE_MODEL."""
    if settings is None:
        settings = Settings()

    if not settings.EVAL_JUDGE_MODEL or not settings.EVAL_JUDGE_MODEL.strip():
        raise ValueError("EVAL_JUDGE_MODEL must be set in Settings/.env for evaluation.")
    if not settings.GOOGLE_API_KEY or not settings.GOOGLE_API_KEY.strip():
        raise ValueError("GOOGLE_API_KEY must be set in Settings/.env for evaluation.")

    setup_environment(settings)

    kwargs: dict[str, Any] = {
        "base_url": settings.GOOGLE_BASE_URL,
        "api_key": settings.GOOGLE_API_KEY,
        "model": settings.EVAL_JUDGE_MODEL,
        "temperature": 0.0,
        "request_timeout": settings.LLM_TIMEOUT_SECONDS,
        "streaming": False,
    }
    if settings.EVAL_JUDGE_REQUESTS_PER_MINUTE > 0:
        kwargs["rate_limiter"] = _get_judge_rate_limiter(settings.EVAL_JUDGE_REQUESTS_PER_MINUTE)

    return ChatOpenAI(**kwargs)


def check_llm_reachable(settings: Settings | None = None, timeout: float = 2.0) -> bool:
    """Check whether the configured Groq / OpenAI-compatible endpoint is reachable."""
    try:
        resolved = _resolve_settings(settings)
        base_url = resolved.GROQ_BASE_URL.rstrip("/")
        url = f"{base_url}/models"
        headers: dict[str, str] = {"User-Agent": "hrbot/0.1"}
        if resolved.GROQ_API_KEY:
            headers["Authorization"] = f"Bearer {resolved.GROQ_API_KEY}"
        req = urllib.request.Request(url, headers=headers, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return 200 <= int(getattr(response, "status", 200)) < 500
    except Exception:  # noqa: BLE001
        return False
