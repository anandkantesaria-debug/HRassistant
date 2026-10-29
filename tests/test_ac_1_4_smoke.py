from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from hrbot.smoke import smoke


def test_ac_1_4_smoke_success(capsys: pytest.CaptureFixture[str]) -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="OK")

    with patch("hrbot.smoke.get_llm", return_value=mock_llm):
        smoke()

    captured = capsys.readouterr()
    mock_llm.invoke.assert_called_once_with("Reply with exactly: OK")
    assert "Reply: OK" in captured.out
    assert "Latency:" in captured.out


def test_ac_1_4_smoke_unreachable_exits_nonzero_with_hint(
    capsys: pytest.CaptureFixture[str],
) -> None:
    mock_llm = MagicMock()
    mock_llm.invoke.side_effect = ConnectionError("Connection refused: https://api.groq.com/openai/v1")

    with patch("hrbot.smoke.get_llm", return_value=mock_llm), pytest.raises(SystemExit) as exc_info:
        smoke()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Hint: Is GROQ_API_KEY valid and reachable? Is LLM_MODEL valid?" in captured.out


@pytest.mark.integration
def test_ac_1_4_smoke_live() -> None:
    smoke()
