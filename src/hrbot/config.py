import os

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        str_strip_whitespace=True,
    )

    # --- Generation LLM via Groq (OpenAI-compatible) ---
    GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"
    GROQ_API_KEY: str = Field(..., min_length=1)
    LLM_MODEL: str = Field(..., min_length=1)
    LLM_TIMEOUT_SECONDS: int = 60

    # --- Evaluation Judge LLM via Google Gemini API (OpenAI-compatible) ---
    GOOGLE_BASE_URL: str = "https://api.groq.com/openai/v1"
    GOOGLE_API_KEY: str | None = None
    EVAL_JUDGE_MODEL: str | None = None

    # --- Embeddings (local, free) ---
    EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    QUERY_INSTRUCTION: str = "Represent this sentence for searching relevant passages:"

    # --- Vector store ---
    CHROMA_DIR: str = "data/chroma"
    COLLECTION_NAME: str = "hr_policies"

    # --- Ingestion / retrieval ---
    CHUNK_SIZE: int = 600
    CHUNK_OVERLAP: int = 150
    TOP_K: int = 3
    RELEVANCE_THRESHOLD: float = 0.68

    # --- LangSmith ---
    LANGSMITH_TRACING: bool = True
    LANGSMITH_API_KEY: str | None = None
    LANGSMITH_PROJECT: str = "hr-policy-bot"
    LANGSMITH_HIDE_INPUTS: bool | None = None
    LANGSMITH_HIDE_OUTPUTS: bool | None = None

    # --- App ---
    HR_CONTACT_EMAIL: str = "hr@yourcompany.com"

    # --- Evaluation ---
    EVAL_MAX_CONCURRENCY: int = 2
    # Client-side throttle for judge calls (Google free tier allows 15 RPM per model). 0 disables.
    EVAL_JUDGE_REQUESTS_PER_MINUTE: float = 12.0


def setup_environment(settings: Settings | None = None) -> None:
    """Set up process environment variables for external libraries like LangSmith."""
    if settings is None:
        settings = Settings()

    if settings.LANGSMITH_TRACING:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
        os.environ["LANGCHAIN_PROJECT"] = settings.LANGSMITH_PROJECT
        if settings.LANGSMITH_API_KEY:
            os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY
            os.environ["LANGCHAIN_API_KEY"] = settings.LANGSMITH_API_KEY
        if settings.LANGSMITH_HIDE_INPUTS is not None:
            os.environ["LANGSMITH_HIDE_INPUTS"] = str(settings.LANGSMITH_HIDE_INPUTS).lower()
        if settings.LANGSMITH_HIDE_OUTPUTS is not None:
            os.environ["LANGSMITH_HIDE_OUTPUTS"] = str(settings.LANGSMITH_HIDE_OUTPUTS).lower()
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
