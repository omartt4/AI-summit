"""Ollama embedding client."""

from collections.abc import Sequence

import ollama

from .config import Settings


def embed_texts(texts: Sequence[str], settings: Settings) -> list[list[float]]:
    if not texts:
        return []
    client = ollama.Client(host=settings.ollama_host)
    try:
        response = client.embed(model=settings.embedding_model, input=list(texts))
    finally:
        client.close()
    embeddings = response["embeddings"]
    if len(embeddings) != len(texts):
        raise RuntimeError("Ollama returned an unexpected embedding count")
    return embeddings
