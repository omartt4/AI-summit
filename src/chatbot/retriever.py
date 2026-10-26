"""Hybrid dense, BM25, and reciprocal-rank-fusion retrieval."""

from .config import Settings
from .embeddings import embed_texts
from .sparse_retriever import SparseRetriever


def rrf_fuse(ranked_lists: list[list[dict]], k: int, top_k: int) -> list[dict]:
    scores: dict[str, float] = {}
    items: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            key = item["id"]
            items[key] = item
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
    return [
        items[key]
        for key in sorted(scores, key=scores.get, reverse=True)[:top_k]
    ]


def retrieve(collection, query: str, settings: Settings, sparse: SparseRetriever) -> list[dict]:
    total = collection.count()
    if total == 0:
        return []
    dense = collection.query(
        query_embeddings=embed_texts([query], settings),
        n_results=min(settings.dense_candidates, total),
        include=["documents", "metadatas"],
    )
    dense_items = [
        {
            "id": dense["ids"][0][index],
            "text": dense["documents"][0][index],
            **dense["metadatas"][0][index],
        }
        for index in range(len(dense["ids"][0]))
    ]
    return rrf_fuse(
        [dense_items, sparse.search(query, settings.bm25_candidates)],
        settings.rrf_k,
        settings.final_top_k,
    )
