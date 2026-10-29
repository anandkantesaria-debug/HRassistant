from pathlib import Path

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from hrbot.config import Settings
from hrbot.ingest import compute_chunk_id, index_chunks
from hrbot.retriever import format_query, retrieve


class SpyEmbeddings(Embeddings):
    def __init__(self) -> None:
        self.embedded_documents: list[str] = []
        self.embedded_queries: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded_documents.extend(texts)
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.embedded_queries.append(text)
        return [1.0, 0.0, 0.0]


def test_ac_3_2_bge_prefix_applied_to_queries_only_and_configurable(
    tmp_path: Path,
) -> None:
    default_prefix = "Represent this sentence for searching relevant passages: "
    bge_settings = Settings(
        OMNIROUTE_API_KEY="test-key",
        LLM_MODEL="test-model",
        EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
        CHROMA_DIR=str(tmp_path / "chroma"),
        COLLECTION_NAME="hr_policies",
        _env_file=None,
    )
    spy = SpyEmbeddings()

    raw_doc_text = "Regular full-time salaried employees work 37.5 hours per week."
    doc = Document(
        page_content=raw_doc_text,
        metadata={
            "source": "handbook.pdf",
            "page": 9,
            "chunk_index": 0,
            "doc_title": "Handbook",
            "chunk_id": compute_chunk_id("handbook.pdf", 9, 0, raw_doc_text),
        },
    )

    # 1. Ingestion never applies the query instruction prefix to documents
    index_chunks([doc], settings=bge_settings, embeddings=spy, reset=True)
    assert spy.embedded_documents == [raw_doc_text]
    assert not any(
        d.startswith(bge_settings.QUERY_INSTRUCTION) for d in spy.embedded_documents
    )

    # 2. Retrieval applies default BGE prefix to queries
    query = "How many hours is the standard workweek?"
    retrieve(query, k=1, settings=bge_settings, embeddings=spy)
    assert spy.embedded_queries[-1] == f"{default_prefix}{query}"

    # 3. Prefix is configurable via Settings.QUERY_INSTRUCTION
    custom_settings = Settings(
        OMNIROUTE_API_KEY="test-key",
        LLM_MODEL="test-model",
        EMBEDDING_MODEL="BAAI/bge-base-en-v1.5",
        QUERY_INSTRUCTION="Custom search prefix:",
        CHROMA_DIR=str(tmp_path / "chroma"),
        COLLECTION_NAME="hr_policies",
        _env_file=None,
    )
    retrieve(query, k=1, settings=custom_settings, embeddings=spy)
    assert spy.embedded_queries[-1] == f"Custom search prefix: {query}"

    # 4. Non-BGE models do not receive the BGE query prefix
    non_bge_settings = Settings(
        OMNIROUTE_API_KEY="test-key",
        LLM_MODEL="test-model",
        EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2",
        CHROMA_DIR=str(tmp_path / "chroma"),
        COLLECTION_NAME="hr_policies",
        _env_file=None,
    )
    assert format_query(query, settings=non_bge_settings) == query
    retrieve(query, k=1, settings=non_bge_settings, embeddings=spy)
    assert spy.embedded_queries[-1] == query

