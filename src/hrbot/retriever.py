from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from langchain_chroma import Chroma
from langchain_chroma.vectorstores import maximal_marginal_relevance
from langchain_core.embeddings import Embeddings
from pydantic import ValidationError

from hrbot.config import Settings
from hrbot.embeddings import get_embeddings


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    score: float


def _resolve_settings(settings: Settings | None = None) -> Settings:
    """Resolve Settings, providing non-empty LLM placeholders if only retrieval settings are configured."""
    if settings is not None:
        return settings
    try:
        return Settings()
    except ValidationError:
        return Settings(
            GROQ_API_KEY=os.environ.get("GROQ_API_KEY") or "unused",
            LLM_MODEL=os.environ.get("LLM_MODEL") or "unused",
        )


def format_query(query: str, settings: Settings | None = None) -> str:
    """Apply the query instruction prefix for BGE models to queries only."""
    resolved = _resolve_settings(settings)

    if "bge" in resolved.EMBEDDING_MODEL.lower() and resolved.QUERY_INSTRUCTION:
        prefix = f"{resolved.QUERY_INSTRUCTION.rstrip()} "
        if not query.startswith(prefix):
            return f"{prefix}{query}"
    return query


def get_vectorstore(
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Return the persisted Chroma vector store configured with cosine distance."""
    resolved = _resolve_settings(settings)
    if embeddings is None:
        embeddings = get_embeddings(settings=resolved)

    return Chroma(
        collection_name=resolved.COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=resolved.CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},
    )


def get_chunk_count(
    vectorstore: Chroma | None = None,
    settings: Settings | None = None,
) -> int:
    """Return the number of indexed chunks in Chroma, or 0 if unavailable."""
    try:
        store = vectorstore if vectorstore is not None else get_vectorstore(settings=settings)
        return int(store._collection.count())
    except Exception:  # noqa: BLE001
        return 0


def make_retriever(
    settings: Settings | None = None,
    vectorstore: Chroma | None = None,
    embeddings: Embeddings | None = None,
) -> Callable[[str, int], list[RetrievedChunk]]:
    """Create a reusable retriever callable backed by a single Chroma vectorstore instance."""
    resolved = _resolve_settings(settings)
    resolved_embeddings = (
        embeddings
        if embeddings is not None
        else (
            getattr(vectorstore, "_embedding_function", None)
            if vectorstore is not None
            else get_embeddings(settings=resolved)
        )
    )
    store = (
        vectorstore
        if vectorstore is not None
        else get_vectorstore(settings=resolved, embeddings=resolved_embeddings)
    )

    def _retrieve_fn(query: str, k: int = resolved.TOP_K) -> list[RetrievedChunk]:
        return retrieve(
            query=query,
            k=k,
            settings=resolved,
            embeddings=resolved_embeddings,
            vectorstore=store,
        )

    return _retrieve_fn


def _distance_to_relevance(distance: float) -> float:
    """Convert cosine distance to relevance score in [0.0, 1.0] (higher is better)."""
    return max(0.0, min(1.0, 1.0 - float(distance)))


def retrieve(
    query: str,
    k: int = 5,
    use_mmr: bool = False,
    settings: Settings | None = None,
    embeddings: Embeddings | None = None,
    vectorstore: Chroma | None = None,
    fetch_k: int | None = None,
    lambda_mult: float = 0.5,
) -> list[RetrievedChunk]:
    """Retrieve the top-k relevant chunks from Chroma with relevance scores in [0, 1]."""
    if k <= 0:
        return []

    resolved = _resolve_settings(settings)

    if embeddings is None:
        if vectorstore is not None and getattr(vectorstore, "_embedding_function", None) is not None:
            embeddings = vectorstore._embedding_function
        else:
            embeddings = get_embeddings(settings=resolved)

    if vectorstore is None:
        vectorstore = get_vectorstore(settings=resolved, embeddings=embeddings)

    count = vectorstore._collection.count()
    if count == 0:
        return []

    formatted_query = format_query(query, settings=resolved)
    query_embedding = embeddings.embed_query(formatted_query)
    n_target = min(k, count)

    if not use_mmr:
        results = vectorstore._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_target,
            include=["documents", "metadatas", "distances"],
        )
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for doc_text, meta, dist in zip(docs, metas, distances, strict=False):
            meta_dict = meta or {}
            chunks.append(
                RetrievedChunk(
                    text=str(doc_text or ""),
                    source=str(meta_dict.get("source", "")),
                    page=int(meta_dict.get("page", 1)),
                    score=_distance_to_relevance(float(dist)),
                )
            )
        return chunks

    candidate_k = min(count, fetch_k if fetch_k is not None else max(20, k * 4))
    results = vectorstore._collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_k,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]
    candidate_embeddings = (results.get("embeddings") or [[]])[0]

    if not docs or candidate_embeddings is None or len(candidate_embeddings) == 0:
        return []

    mmr_indices = maximal_marginal_relevance(
        np.array(query_embedding, dtype=np.float32),
        candidate_embeddings,
        lambda_mult=lambda_mult,
        k=n_target,
    )

    chunks = []
    for idx in mmr_indices:
        meta_dict = metas[idx] or {}
        chunks.append(
            RetrievedChunk(
                text=str(docs[idx] or ""),
                source=str(meta_dict.get("source", "")),
                page=int(meta_dict.get("page", 1)),
                score=_distance_to_relevance(float(distances[idx])),
            )
        )
    return chunks
