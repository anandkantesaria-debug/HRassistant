import pytest
from pydantic import ValidationError

from hrbot.config import Settings


def test_ac_1_1_loads_all_variables_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    test_env = {
        "OMNIROUTE_BASE_URL": "http://test-server:20128/v1",
        "OMNIROUTE_API_KEY": "test-key-123",
        "LLM_MODEL": "test-model-abc",
        "EVAL_JUDGE_MODEL": "test-judge-xyz",
        "LLM_TIMEOUT_SECONDS": "45",
        "EMBEDDING_MODEL": "BAAI/bge-small-en-v1.5",
        "QUERY_INSTRUCTION": "Test instruction:",
        "CHROMA_DIR": "test/chroma",
        "COLLECTION_NAME": "test_collection",
        "CHUNK_SIZE": "800",
        "CHUNK_OVERLAP": "120",
        "TOP_K": "3",
        "RELEVANCE_THRESHOLD": "0.7",
        "LANGSMITH_TRACING": "true",
        "LANGSMITH_API_KEY": "test-ls-key",
        "LANGSMITH_PROJECT": "test-project",
        "LANGSMITH_HIDE_INPUTS": "true",
        "LANGSMITH_HIDE_OUTPUTS": "true",
        "HR_CONTACT_EMAIL": "hr-test@example.com",
    }
    for k, v in test_env.items():
        monkeypatch.setenv(k, v)

    settings = Settings(_env_file=None)
    assert settings.OMNIROUTE_BASE_URL == "http://test-server:20128/v1"
    assert settings.OMNIROUTE_API_KEY == "test-key-123"
    assert settings.LLM_MODEL == "test-model-abc"
    assert settings.EVAL_JUDGE_MODEL == "test-judge-xyz"
    assert settings.LLM_TIMEOUT_SECONDS == 45
    assert settings.EMBEDDING_MODEL == "BAAI/bge-small-en-v1.5"
    assert settings.QUERY_INSTRUCTION == "Test instruction:"
    assert settings.CHROMA_DIR == "test/chroma"
    assert settings.COLLECTION_NAME == "test_collection"
    assert settings.CHUNK_SIZE == 800
    assert settings.CHUNK_OVERLAP == 120
    assert settings.TOP_K == 3
    assert settings.RELEVANCE_THRESHOLD == 0.7
    assert settings.LANGSMITH_TRACING is True
    assert settings.LANGSMITH_API_KEY == "test-ls-key"
    assert settings.LANGSMITH_PROJECT == "test-project"
    assert settings.LANGSMITH_HIDE_INPUTS is True
    assert settings.LANGSMITH_HIDE_OUTPUTS is True
    assert settings.HR_CONTACT_EMAIL == "hr-test@example.com"


def test_ac_1_1_missing_required_variable_omniroute_api_key_names_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIROUTE_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODEL", "valid-model")

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    error_str = str(exc_info.value)
    assert "OMNIROUTE_API_KEY" in error_str


def test_ac_1_1_missing_required_variable_llm_model_names_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNIROUTE_API_KEY", "valid-key")
    monkeypatch.delenv("LLM_MODEL", raising=False)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    error_str = str(exc_info.value)
    assert "LLM_MODEL" in error_str


def test_ac_1_1_blank_required_variable_names_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNIROUTE_API_KEY", "   ")
    monkeypatch.setenv("LLM_MODEL", "valid-model")

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    error_str = str(exc_info.value)
    assert "OMNIROUTE_API_KEY" in error_str
