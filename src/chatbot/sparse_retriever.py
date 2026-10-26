"""BM25 retrieval over the complete Chroma collection."""

from rank_bm25 import BM25Okapi


class SparseRetriever:
    def __init__(self, documents: list[dict]):
        self.documents = documents
        self._bm25 = BM25Okapi([item["text"].lower().split() for item in documents])

    def search(self, query: str, top_k: int) -> list[dict]:
        if not self.documents:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        return [self.documents[index] for index in indices[:top_k]]
