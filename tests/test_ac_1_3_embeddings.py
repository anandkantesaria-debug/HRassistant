from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from langchain_huggingface import HuggingFaceEmbeddings

from hrbot.config import Settings
from hrbot.embeddings import _get_cached_embeddings, get_embeddings


def test_ac_1_3_get_embeddings_cached_on_cpu() -> None:
    settings = Settings(
        GROQ_API_KEY="key",
        LLM_MODEL="model",
        EMBEDDING_MODEL="mock-model",
        _env_file=None,
    )

    with patch("hrbot.embeddings.HuggingFaceEmbeddings") as mock_hf:
        mock_instance = MagicMock(spec=HuggingFaceEmbeddings)
        mock_hf.return_value = mock_instance

        # Clear lru_cache for isolated test
        _get_cached_embeddings.cache_clear()

        e1 = get_embeddings(settings=settings)
        e2 = get_embeddings(settings=settings)

        assert e1 is e2
        mock_hf.assert_called_once_with(
            model_name="mock-model",
            model_kwargs={"device": "cpu"},
        )


@pytest.mark.integration
def test_ac_1_3_embeddings_dimension_and_similarity() -> None:
    settings = Settings(
        GROQ_API_KEY="key",
        LLM_MODEL="model",
        EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
        _env_file=None,
    )
    embeddings = get_embeddings(settings=settings)
    assert isinstance(embeddings, HuggingFaceEmbeddings)

    # Test dimension is 384 for bge-small
    vec_a = embeddings.embed_query("Employees are entitled to 20 days of annual leave per calendar year.")
    assert len(vec_a) == 384

    # Test similarity: two similar sentences vs unrelated sentence
    vec_similar = embeddings.embed_query("Staff members receive twenty days of paid annual vacation each year.")
    vec_unrelated = embeddings.embed_query("Photosynthesis converts sunlight, water, and carbon dioxide into oxygen.")

    def cosine_similarity(a: list[float], b: list[float]) -> float:
        va = np.array(a)
        vb = np.array(b)
        return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))

    sim_similar = cosine_similarity(vec_a, vec_similar)
    sim_unrelated = cosine_similarity(vec_a, vec_unrelated)

    assert sim_similar > sim_unrelated
