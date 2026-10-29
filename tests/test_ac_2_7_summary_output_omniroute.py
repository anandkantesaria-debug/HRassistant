from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from hrbot.config import Settings
from hrbot.ingest import ingest_directory


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.2, 0.4, 0.6, 0.8] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.2, 0.4, 0.6, 0.8]


def test_ac_2_7_prints_completion_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    policies_dir = tmp_path / "policies"
    policies_dir.mkdir()

    pdf_path = policies_dir / "sample.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=A4)
    c.drawString(72, 750, "Page 1 policy text.")
    c.showPage()
    # Blank page 2
    c.showPage()
    c.save()

    settings = Settings(
        OMNIROUTE_API_KEY="test-key",
        LLM_MODEL="test-model",
        CHROMA_DIR=str(tmp_path / "chroma"),
        COLLECTION_NAME="hr_policies",
        _env_file=None,
    )

    summary = ingest_directory(
        path=policies_dir,
        reset=False,
        settings=settings,
        embeddings=FakeEmbeddings(),
    )

    captured = capsys.readouterr().out
    assert summary.files == 1
    assert summary.pages == 2
    assert summary.chunks == 1
    assert summary.skipped_pages == 1
    assert summary.elapsed_seconds >= 0.0

    assert "files=1" in captured
    assert "pages=2" in captured
    assert "chunks=1" in captured
    assert "skipped_pages=1" in captured
    assert "elapsed=" in captured

