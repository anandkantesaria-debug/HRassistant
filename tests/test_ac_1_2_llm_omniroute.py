from unittest.mock import patch

import pytest
from langchain_openai import ChatOpenAI

from hrbot.config import Settings
from hrbot.llm import get_llm


def test_ac_1_2_get_llm_configures_chat_openai() -> None:
    settings = Settings(
        OMNIROUTE_BASE_URL="http://custom-url:20128/v1",
        OMNIROUTE_API_KEY="test-api-key",
        LLM_MODEL="test-model",
        LLM_TIMEOUT_SECONDS=42,
        _env_file=None,
    )

    with patch("hrbot.llm.ChatOpenAI") as mock_chat_openai:
        llm = get_llm(settings=settings)
        mock_chat_openai.assert_called_once()
        _, kwargs = mock_chat_openai.call_args
        assert kwargs["base_url"] == "http://custom-url:20128/v1"
        assert kwargs["api_key"] == "test-api-key"
        assert kwargs["model"] == "test-model"
        assert kwargs["temperature"] == 0.0
        assert kwargs["request_timeout"] == 42
        assert kwargs["streaming"] is False
        assert llm == mock_chat_openai.return_value


@pytest.mark.integration
def test_ac_1_2_llm_integration() -> None:
    llm = get_llm()
    assert isinstance(llm, ChatOpenAI)
    response = llm.invoke("Reply with exactly: OK")
    assert "OK" in response.content
