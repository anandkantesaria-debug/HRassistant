from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from hrbot import retrieval_debug
from hrbot.config import Settings
from hrbot.ingest import compute_chunk_id, index_chunks


class SimpleEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0, 0.0] if "jury" in t.lower() else [0.0, 1.0, 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]


def test_ac_3_4_retrieval_debug_prints_scores_sources_and_pages(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings(
        OMNIROUTE_API_KEY="test-key",
        LLM_MODEL="test-model",
        CHROMA_DIR=str(tmp_path / "chroma"),
        COLLECTION_NAME="hr_policies",
        _env_file=None,
    )
    emb = SimpleEmbeddings()

    text = "Regular full-time employees will be paid their normal salary during jury duty."
    doc = Document(
        page_content=text,
        metadata={
            "source": "Company Salaried Employee Handbook 2025.pdf",
            "page": 47,
            "chunk_index": 0,
            "doc_title": "Handbook",
            "chunk_id": compute_chunk_id(
                "Company Salaried Employee Handbook 2025.pdf", 47, 0, text
            ),
        },
    )
    index_chunks([doc], settings=settings, embeddings=emb, reset=True)

    exit_code = retrieval_debug.main(
        ["What is the jury duty policy?", "-k", "1"],
        settings=settings,
        embeddings=emb,
    )
    assert exit_code == 0

    out = capsys.readouterr().out
    assert "score=" in out
    assert "source=Company Salaried Employee Handbook 2025.pdf" in out
    assert "page=47" in out
    assert "jury duty" in out

