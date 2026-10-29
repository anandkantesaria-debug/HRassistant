from pathlib import Path
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from hrbot.config import Settings
from hrbot.ingest import compute_chunk_id, index_chunks


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3, 0.4] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.1, 0.2, 0.3, 0.4]


def test_ac_2_6_chroma_collection_cosine_and_get_embeddings(tmp_path: Path) -> None:
    chroma_dir = tmp_path / "chroma_store"
    settings = Settings(
        GROQ_API_KEY="test-key",
        LLM_MODEL="test-model",
        CHROMA_DIR=str(chroma_dir),
        COLLECTION_NAME="hr_policies",
        _env_file=None,
    )

    chunk_text = "Remote work is permitted up to 3 days per week."
    cid = compute_chunk_id("remote.pdf", 1, 0, chunk_text)
    doc = Document(
        page_content=chunk_text,
        metadata={
            "source": "remote.pdf",
            "page": 1,
            "chunk_index": 0,
            "doc_title": "Remote Work Policy",
            "chunk_id": cid,
        },
    )

    fake_emb = FakeEmbeddings()
    with patch("hrbot.ingest.get_embeddings", return_value=fake_emb) as mock_get_emb:
        vs = index_chunks([doc], settings=settings, embeddings=None, reset=False)
        mock_get_emb.assert_called_once_with(settings=settings)

    assert vs._collection.name == "hr_policies"
    assert vs._collection.metadata is not None
    assert vs._collection.metadata.get("hnsw:space") == "cosine"
    assert vs._collection.count() == 1
    assert chroma_dir.exists()

