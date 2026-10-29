import os

import pytest

from hrbot.config import Settings, setup_environment
from hrbot.llm import get_llm


def test_ac_1_5_langsmith_tracing_env_propagation(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = Settings(
        GROQ_API_KEY="test-key",
        LLM_MODEL="test-model",
        LANGSMITH_TRACING=True,
        LANGSMITH_PROJECT="test-hr-policy-bot",
        LANGSMITH_API_KEY="test-lsv2-key",
        LANGSMITH_HIDE_INPUTS=True,
        LANGSMITH_HIDE_OUTPUTS=False,
        _env_file=None,
    )

    setup_environment(settings)

    assert os.environ.get("LANGSMITH_TRACING") == "true"
    assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    assert os.environ.get("LANGSMITH_PROJECT") == "test-hr-policy-bot"
    assert os.environ.get("LANGCHAIN_PROJECT") == "test-hr-policy-bot"
    assert os.environ.get("LANGSMITH_API_KEY") == "test-lsv2-key"
    assert os.environ.get("LANGSMITH_HIDE_INPUTS") == "true"
    assert os.environ.get("LANGSMITH_HIDE_OUTPUTS") == "false"


@pytest.mark.integration
def test_ac_1_5_langsmith_smoke_trace() -> None:
    settings = Settings()
    assert settings.LANGSMITH_TRACING is True
    llm = get_llm(settings=settings)
    res = llm.invoke("Reply with exactly: OK")
    assert res is not None
