"""Application configuration loaded from environment variables."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


def _required(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise ValueError(f"{name} is required in .env")
    return value


def _positive_int(name: str) -> int:
    raw_value = _required(name)
    try:
        value = int(raw_value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {raw_value!r}") from error
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


@dataclass(frozen=True)
class Settings:
    pdf_dir: Path
    chroma_db_dir: Path
    collection_name: str
    ollama_host: str
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    embed_batch_size: int
    upsert_batch_size: int
    dense_candidates: int
    bm25_candidates: int
    final_top_k: int
    rrf_k: int
    llm_temperature: float
    llm_max_tokens: int
    max_context_chars: int

    def __post_init__(self) -> None:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        if not self.collection_name.strip():
            raise ValueError("COLLECTION_NAME must not be empty")


def load_settings() -> Settings:
    """Load and validate settings, resolving paths relative to the project."""
    def path_value(name: str) -> Path:
        path = Path(_required(name))
        return path if path.is_absolute() else ROOT_DIR / path

    raw_temperature = _required("LLM_TEMPERATURE")
    try:
        temperature = float(raw_temperature)
    except ValueError as error:
        raise ValueError(
            f"LLM_TEMPERATURE must be a number, got {raw_temperature!r}"
        ) from error
    if not 0 <= temperature <= 2:
        raise ValueError("LLM_TEMPERATURE must be between 0 and 2")

    return Settings(
        pdf_dir=path_value("PDF_DIR"),
        chroma_db_dir=path_value("CHROMA_DB_DIR"),
        collection_name=_required("COLLECTION_NAME"),
        ollama_host=_required("OLLAMA_HOST"),
        embedding_model=_required("EMBEDDING_MODEL"),
        llm_model=_required("LLM_MODEL"),
        chunk_size=_positive_int("CHUNK_SIZE"),
        chunk_overlap=_positive_int("CHUNK_OVERLAP"),
        embed_batch_size=_positive_int("EMBED_BATCH_SIZE"),
        upsert_batch_size=_positive_int("UPSERT_BATCH_SIZE"),
        dense_candidates=_positive_int("DENSE_CANDIDATES"),
        bm25_candidates=_positive_int("BM25_CANDIDATES"),
        final_top_k=_positive_int("FINAL_TOP_K"),
        rrf_k=_positive_int("RRF_K"),
        llm_temperature=temperature,
        llm_max_tokens=_positive_int("LLM_MAX_TOKENS"),
        max_context_chars=_positive_int("MAX_CONTEXT_CHARS"),
    )
