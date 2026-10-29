from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from hrbot.config import Settings
from hrbot.ingest import compute_chunk_id, index_chunks
from hrbot.retriever import RetrievedChunk, retrieve


class DeterministicEmbeddings(Embeddings):
    """Offline embeddings that map keywords to predictable 4D vectors."""

    def _embed(self, text: str) -> list[float]:
        lower = text.lower()
        if "vacation" in lower or "holiday" in lower:
            return [1.0, 0.0, 0.0, 0.0]
        if "pension" in lower or "retirement" in lower:
            return [0.8, 0.6, 0.0, 0.0]
        if "bereavement" in lower:
            return [0.0, 1.0, 0.0, 0.0]
        return [0.0, 0.0, 1.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _make_doc(source: str, page: int, idx: int, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": source,
            "page": page,
            "chunk_index": idx,
            "doc_title": source,
            "chunk_id": compute_chunk_id(source, page, idx, text),
        },
    )


def test_ac_3_1_retrieve_returns_retrieved_chunks_with_scores(tmp_path: Path) -> None:
    settings = Settings(
        GROQ_API_KEY="test-key",
        LLM_MODEL="test-model",
        CHROMA_DIR=str(tmp_path / "chroma"),
        COLLECTION_NAME="hr_policies",
        _env_file=None,
    )
    emb = DeterministicEmbeddings()

    docs = [
        _make_doc("handbook.pdf", 43, 0, "Employees accrue paid vacation time each month."),
        _make_doc("handbook.pdf", 11, 0, "The Group Retirement Plan includes a pension plan."),
        _make_doc("handbook.pdf", 46, 0, "Bereavement leave provides up to five days off."),
    ]
    index_chunks(docs, settings=settings, embeddings=emb, reset=True)

    results = retrieve("How does vacation accrual work?", k=5, settings=settings, embeddings=emb)

    assert len(results) == 3
    for item in results:
        assert isinstance(item, RetrievedChunk)
        assert isinstance(item.text, str) and len(item.text) > 0
        assert item.source == "handbook.pdf"
        assert isinstance(item.page, int)
        assert 0.0 <= item.score <= 1.0

    # Higher score is better: exact match on 'vacation' is ranked #1 with score ~1.0
    assert results[0].page == 43
    assert results[0].score > results[1].score > results[2].score
    assert results[0].score > 0.99

    # Respects k parameter
    top_2 = retrieve("How does vacation accrual work?", k=2, settings=settings, embeddings=emb)
    assert len(top_2) == 2

