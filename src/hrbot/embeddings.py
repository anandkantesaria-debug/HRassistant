from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from hrbot.config import Settings


@lru_cache(maxsize=1)
def _get_cached_embeddings(model_name: str) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
    )


def get_embeddings(settings: Settings | None = None) -> HuggingFaceEmbeddings:
    """Return a cached HuggingFace embeddings object using EMBEDDING_MODEL on CPU."""
    if settings is None:
        settings = Settings()
    return _get_cached_embeddings(settings.EMBEDDING_MODEL)
