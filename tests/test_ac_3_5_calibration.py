from pathlib import Path

import pytest

from hrbot.retriever import RetrievedChunk
from tools.calibrate_threshold import load_calibration_questions, run_calibration


def test_ac_3_5_calibration_questions_and_threshold_script(
    capsys: pytest.CaptureFixture[str],
) -> None:
    questions_path = Path("eval/calibration_questions.jsonl")
    assert questions_path.exists()

    questions = load_calibration_questions(questions_path)
    answerable = [q for q in questions if q.get("answerable") is True]
    unanswerable = [q for q in questions if q.get("answerable") is False]

    assert len(answerable) >= 10
    assert len(unanswerable) >= 10

    answerable_set = {q["question"] for q in answerable}

    def fake_retrieve(query: str, k: int) -> list[RetrievedChunk]:
        if query in answerable_set:
            return [
                RetrievedChunk(
                    text="Matching policy passage",
                    source="Company Salaried Employee Handbook 2025.pdf",
                    page=43,
                    score=0.76,
                )
            ]
        return [
            RetrievedChunk(
                text="Unrelated policy passage",
                source="Company Salaried Employee Handbook 2025.pdf",
                page=1,
                score=0.55,
            )
        ]

    report = run_calibration(
        questions_path=questions_path,
        k=5,
        retrieve_fn=fake_retrieve,
    )
    out = capsys.readouterr().out

    assert "Answerable" in out
    assert "Unanswerable" in out
    assert "Suggested RELEVANCE_THRESHOLD=" in out
    assert 0.55 < report.suggested_threshold < 0.76

    env_example = Path(".env.example").read_text(encoding="utf-8")
    assert "RELEVANCE_THRESHOLD=0.68" in env_example
    assert "tools/calibrate_threshold.py" in env_example

