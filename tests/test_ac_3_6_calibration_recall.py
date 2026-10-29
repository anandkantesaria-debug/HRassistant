import os
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from hrbot.config import Settings
from hrbot.ingest import compute_chunk_id, index_chunks
from hrbot.retriever import retrieve
from tools.calibrate_threshold import load_calibration_questions


class TopicMatchingEmbeddings(Embeddings):
    """Offline topic embeddings mapping calibration questions to policy pages."""

    TOPIC_KEYWORDS: tuple[tuple[int, tuple[str, ...]], ...] = (
        (43, ("roll into next year", "unused days", "carry over up to 5 vacation")),
        (44, ("purchase extra weeks", "buy vacation", "10 pay periods")),
        (48, ("college tuition", "course fees", "3,000 per calendar year")),
        (47, ("recommend a friend", "referral", "jury", "court gives me")),
        (11, ("pension plan", "employer match", "registered pension plan")),
        (12, ("void check", "bank account", "five business days")),
        (9, ("non-production salaried", "37.5 hour workweeks")),
        (10, ("1.5x overtime", "40 hours", "quality technicians")),
        (46, ("short-term and long-term disability", "bereavement", "85% of your basic salary")),
        (45, ("medical, dental, and vision", "three months following their date of hire")),
    )

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        vec = [0.0] * len(self.TOPIC_KEYWORDS)
        for idx, (_page, keywords) in enumerate(self.TOPIC_KEYWORDS):
            if any(kw in lower for kw in keywords):
                vec[idx] = 1.0
        if not any(vec):
            vec[0] = 0.1
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def test_ac_3_6_answerable_calibration_questions_hit_expected_source_top_5(
    tmp_path: Path,
) -> None:
    settings = Settings(
        GROQ_API_KEY="test-key",
        LLM_MODEL="test-model",
        CHROMA_DIR=str(tmp_path / "chroma"),
        COLLECTION_NAME="hr_policies",
        _env_file=None,
    )
    emb = TopicMatchingEmbeddings()

    source_name = "Company Salaried Employee Handbook 2025.pdf"
    sample_passages = [
        (43, "Employees may carry over up to 5 vacation days into the following year."),
        (44, "Staff may elect to buy vacation time deducted over 10 pay periods."),
        (48, "Educational assistance reimburses course fees up to $3,000 per calendar year."),
        (47, "Employee referral program award and jury duty court payments policy."),
        (11, "Group Retirement Plan Registered Pension Plan required 4% and employer match."),
        (12, "Banking changes require a void check five business days before pay period end."),
        (9, "Non-production salaried employees work 37.5 hour workweeks."),
        (10, "Quality Technicians and Maintenance Planners get 1.5x overtime after 40 hours."),
        (46, "Short-term and long-term disability pays 85% of your basic salary; bereavement leave."),
        (45, "Full-time salaried employees enroll in medical, dental, and vision three months following their date of hire."),
        (1, "Distractor document text from another file."),
    ]

    docs: list[Document] = []
    for page, text in sample_passages:
        src = "other_policy.pdf" if page == 1 else source_name
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": src,
                    "page": page,
                    "chunk_index": 0,
                    "doc_title": src,
                    "chunk_id": compute_chunk_id(src, page, 0, text),
                },
            )
        )

    index_chunks(docs, settings=settings, embeddings=emb, reset=True)

    questions = load_calibration_questions("eval/calibration_questions.jsonl")
    answerable = [q for q in questions if q.get("answerable") is True]
    assert len(answerable) >= 10

    for item in answerable:
        chunks = retrieve(item["question"], k=5, settings=settings, embeddings=emb)
        assert len(chunks) > 0
        sources_in_top_5 = [c.source for c in chunks]
        pages_in_top_5 = [c.page for c in chunks]
        assert item["expected_source"] in sources_in_top_5
        assert item["expected_page"] in pages_in_top_5


@pytest.mark.integration
def test_ac_3_6_live_index_answerable_questions_hit_expected_source_top_5(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    os.environ["HF_HUB_OFFLINE"] = "1"

    questions = load_calibration_questions("eval/calibration_questions.jsonl")
    answerable = [q for q in questions if q.get("answerable") is True]

    for item in answerable:
        chunks = retrieve(item["question"], k=5)
        assert len(chunks) == 5
        assert any(c.source == item["expected_source"] for c in chunks)
        assert any(
            c.source == item["expected_source"] and c.page == item["expected_page"]
            for c in chunks
        )
