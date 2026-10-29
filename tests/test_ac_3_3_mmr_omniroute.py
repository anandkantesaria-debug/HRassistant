import inspect
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from hrbot.config import Settings
from hrbot.ingest import compute_chunk_id, index_chunks
from hrbot.retriever import retrieve


class MMREmbeddings(Embeddings):
    """Embeddings designed to test MMR diversity vs pure similarity."""

    def _embed(self, text: str) -> list[float]:
        if "dup_1" in text:
            return [1.0, 0.0, 0.0]
        if "dup_2" in text:
            return [0.99, 0.141, 0.0]
        if "diverse_relevant" in text:
            return [0.6, 0.0, 0.8]
        # Query vector
        return [1.0, 0.0, 0.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def _doc(page: int, text: str) -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "handbook.pdf",
            "page": page,
            "chunk_index": 0,
            "doc_title": "Handbook",
            "chunk_id": compute_chunk_id("handbook.pdf", page, 0, text),
        },
    )


def test_ac_3_3_mmr_optional_default_off(tmp_path: Path) -> None:
    sig = inspect.signature(retrieve)
    assert sig.parameters["use_mmr"].default is False

    settings = Settings(
        OMNIROUTE_API_KEY="test-key",
        LLM_MODEL="test-model",
        CHROMA_DIR=str(tmp_path / "chroma"),
        COLLECTION_NAME="hr_policies",
        _env_file=None,
    )
    emb = MMREmbeddings()

    docs = [
        _doc(1, "dup_1: Vacation policy overview part A"),
        _doc(2, "dup_2: Vacation policy overview part A duplicate"),
        _doc(3, "diverse_relevant: Vacation carryover rules"),
    ]
    index_chunks(docs, settings=settings, embeddings=emb, reset=True)

    # Default (use_mmr=False) picks the top-2 most similar (pages 1 and 2)
    standard_chunks = retrieve("query", k=2, settings=settings, embeddings=emb)
    assert [c.page for c in standard_chunks] == [1, 2]

    # MMR (use_mmr=True) diversifies away from near-identical dup_2 and selects page 3
    mmr_chunks = retrieve(
        "query",
        k=2,
        use_mmr=True,
        settings=settings,
        embeddings=emb,
        lambda_mult=0.3,
    )
    assert [c.page for c in mmr_chunks] == [1, 3]
    for c in mmr_chunks:
        assert 0.0 <= c.score <= 1.0

