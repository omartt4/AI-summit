"""Indexing and question-answer orchestration."""


import chromadb

from .config import Settings
from .generator import generate_answer
from .pdf_loader import Passage, discover_pdfs, load_pdf_passages
from .retriever import retrieve
from .sparse_retriever import SparseRetriever
from .vector_store import build_index


def collect_passages(settings: Settings) -> list[Passage]:
    passages: list[Passage] = []
    for pdf_path in discover_pdfs(settings.pdf_dir):
        passages.extend(load_pdf_passages(pdf_path, settings.chunk_size, settings.chunk_overlap))
    return passages


def index_pdfs(settings: Settings) -> int:
    return len(collect_passages_and_build(settings))


def collect_passages_and_build(settings: Settings) -> list[Passage]:
    passages = collect_passages(settings)
    build_index(passages, settings)
    return passages


def ask_question(question: str, settings: Settings) -> dict:
    client = chromadb.PersistentClient(path=str(settings.chroma_db_dir))
    try:
        collection = client.get_collection(settings.collection_name)
        records = collection.get(include=["documents", "metadatas"])
        documents = [
            {
                "id": records["ids"][index],
                "text": records["documents"][index],
                **records["metadatas"][index],
            }
            for index in range(len(records["ids"]))
        ]
        passages = retrieve(collection, question, settings, SparseRetriever(documents))
        return {
            "answer": generate_answer(question, passages, settings),
            "sources": passages,
        }
    finally:
        client.close()
