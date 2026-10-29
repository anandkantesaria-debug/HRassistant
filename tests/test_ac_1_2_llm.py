from unittest.mock import MagicMock, patch

import pytest
from langchain_openai import ChatOpenAI

from hrbot.config import Settings
from hrbot.llm import check_llm_reachable, get_judge_llm, get_llm


def test_ac_1_2_get_llm_configures_groq_qwen3_with_hidden_reasoning() -> None:
    settings = Settings(
        GROQ_BASE_URL="https://api.groq.com/openai/v1",
        GROQ_API_KEY="gsk-test-api-key",
        LLM_MODEL="qwen/qwen3.8-27b",
        LLM_TIMEOUT_SECONDS=42,
        LANGSMITH_TRACING=False,
        _env_file=None,
    )

    with patch("hrbot.llm.ChatOpenAI") as mock_chat_openai:
        llm = get_llm(settings=settings)
        mock_chat_openai.assert_called_once()
        _, kwargs = mock_chat_openai.call_args
        assert kwargs["base_url"] == "https://api.groq.com/openai/v1"
        assert kwargs["api_key"] == "gsk-test-api-key"
        assert kwargs["model"] == "qwen/qwen3.8-27b"
        assert kwargs["temperature"] == 0.0
        assert kwargs["request_timeout"] == 42
        assert kwargs["streaming"] is False
        assert kwargs["extra_body"] == {"reasoning_format": "hidden"}
        assert llm == mock_chat_openai.return_value


def test_ac_1_2_get_llm_omits_reasoning_format_for_non_qwen3_models() -> None:
    settings = Settings(
        GROQ_BASE_URL="https://api.groq.com/openai/v1",
        GROQ_API_KEY="gsk-test-api-key",
        LLM_MODEL="llama-3.3-70b-versatile",
        LANGSMITH_TRACING=False,
        _env_file=None,
    )

    with patch("hrbot.llm.ChatOpenAI") as mock_chat_openai:
        get_llm(settings=settings)
        _, kwargs = mock_chat_openai.call_args
        assert "extra_body" not in kwargs


def test_ac_1_2_get_judge_llm_configures_google_api_and_validates_keys() -> None:
    settings = Settings(
        GROQ_API_KEY="gsk-test-api-key",
        LLM_MODEL="qwen/qwen3.8-27b",
        GOOGLE_BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai/",
        GOOGLE_API_KEY="AIzaSy-test-google-key",
        EVAL_JUDGE_MODEL="gemini-2.5-flash",
        LLM_TIMEOUT_SECONDS=50,
        LANGSMITH_TRACING=False,
        _env_file=None,
    )

    with patch("hrbot.llm.ChatOpenAI") as mock_chat_openai:
        judge = get_judge_llm(settings=settings)
        mock_chat_openai.assert_called_once()
        _, kwargs = mock_chat_openai.call_args
        assert kwargs["base_url"] == "https://generativelanguage.googleapis.com/v1beta/openai/"
        assert kwargs["api_key"] == "AIzaSy-test-google-key"
        assert kwargs["model"] == "gemini-2.5-flash"
        assert kwargs["temperature"] == 0.0
        assert kwargs["request_timeout"] == 50
        assert kwargs["streaming"] is False
        assert judge == mock_chat_openai.return_value

    # Judge calls are throttled client-side by a shared rate limiter (default 12 RPM)
    limiter = kwargs["rate_limiter"]
    assert limiter.requests_per_second == pytest.approx(12.0 / 60.0)
    with patch("hrbot.llm.ChatOpenAI") as mock_chat_openai_again:
        get_judge_llm(settings=settings)
        assert mock_chat_openai_again.call_args[1]["rate_limiter"] is limiter

    unthrottled = settings.model_copy(update={"EVAL_JUDGE_REQUESTS_PER_MINUTE": 0})
    with patch("hrbot.llm.ChatOpenAI") as mock_chat_openai_off:
        get_judge_llm(settings=unthrottled)
        assert "rate_limiter" not in mock_chat_openai_off.call_args[1]

    missing_judge_model = settings.model_copy(update={"EVAL_JUDGE_MODEL": None})
    with pytest.raises(ValueError, match="EVAL_JUDGE_MODEL"):
        get_judge_llm(settings=missing_judge_model)

    missing_google_key = settings.model_copy(update={"GOOGLE_API_KEY": None})
    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        get_judge_llm(settings=missing_google_key)


def test_ac_1_2_check_llm_reachable_sends_user_agent_and_bearer_token() -> None:
    settings = Settings(
        GROQ_BASE_URL="https://api.groq.com/openai/v1",
        GROQ_API_KEY="gsk-test-api-key",
        LLM_MODEL="qwen/qwen3.8-27b",
        LANGSMITH_TRACING=False,
        _env_file=None,
    )

    fake_resp = MagicMock()
    fake_resp.status = 200
    fake_cm = MagicMock()
    fake_cm.__enter__.return_value = fake_resp
    fake_cm.__exit__.return_value = False

    with patch("urllib.request.urlopen", return_value=fake_cm) as mock_urlopen:
        assert check_llm_reachable(settings=settings) is True
        req = mock_urlopen.call_args[0][0]
        assert req.full_url == "https://api.groq.com/openai/v1/models"
        assert req.get_header("Authorization") == "Bearer gsk-test-api-key"
        assert req.get_header("User-agent") == "hrbot/0.1"


@pytest.mark.integration
def test_ac_1_2_llm_integration() -> None:
    llm = get_llm()
    assert isinstance(llm, ChatOpenAI)
    response = llm.invoke("Reply with exactly: OK")
    assert "OK" in response.content
