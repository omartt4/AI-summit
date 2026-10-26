"""ChromaDB access and failure-safe index replacement."""

import shutil
import tempfile
from pathlib import Path

import chromadb

from .config import Settings
from .embeddings import embed_texts
from .pdf_loader import Passage


def _client(path: Path):
    return chromadb.PersistentClient(path=str(path))


def collection_count(settings: Settings) -> int:
    if not settings.chroma_db_dir.exists():
        return 0
    client = None
    try:
        client = _client(settings.chroma_db_dir)
        return client.get_collection(settings.collection_name).count()
    except ValueError:
        return 0
    finally:
        if client is not None:
            client.close()


def build_index(passages: list[Passage], settings: Settings) -> Path:
    """Build and validate a replacement database, preserving the active one."""
    if not passages:
        raise ValueError("No extractable PDF passages were found")
    parent = settings.chroma_db_dir.parent
    temporary_dir = Path(tempfile.mkdtemp(prefix=".chroma-build-", dir=parent))
    client = None
    backup: Path | None = None
    try:
        client = _client(temporary_dir)
        collection = client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        for offset in range(0, len(passages), settings.upsert_batch_size):
            upsert_batch = passages[offset : offset + settings.upsert_batch_size]
            embeddings: list[list[float]] = []
            for embed_offset in range(0, len(upsert_batch), settings.embed_batch_size):
                embed_batch = upsert_batch[embed_offset : embed_offset + settings.embed_batch_size]
                embeddings.extend(embed_texts([item.text for item in embed_batch], settings))
            collection.upsert(
                ids=[item.id for item in upsert_batch],
                documents=[item.text for item in upsert_batch],
                embeddings=embeddings,
                metadatas=[
                    {
                        "source": item.source,
                        "page": item.page,
                        "document_id": item.document_id,
                    }
                    for item in upsert_batch
                ],
            )
        if collection.count() != len(passages):
            raise RuntimeError("Replacement index validation failed: count mismatch")
        client.close()
        client = None
        if settings.chroma_db_dir.exists():
            backup_candidate = parent / f".{settings.chroma_db_dir.name}-backup"
            if backup_candidate.exists():
                shutil.rmtree(backup_candidate)
            shutil.copytree(settings.chroma_db_dir, backup_candidate)
            backup = backup_candidate
            shutil.rmtree(settings.chroma_db_dir)
        shutil.copytree(temporary_dir, settings.chroma_db_dir)
        active_client = _client(settings.chroma_db_dir)
        active_count = active_client.get_collection(settings.collection_name).count()
        active_client.close()
        if active_count != len(passages):
            raise RuntimeError("Active index validation failed after replacement")
        shutil.rmtree(temporary_dir)
        if backup is not None:
            shutil.rmtree(backup)
        return settings.chroma_db_dir
    except PermissionError as error:
        if client is not None:
            client.close()
        shutil.rmtree(temporary_dir, ignore_errors=True)
        if backup is not None and backup.exists():
            shutil.rmtree(settings.chroma_db_dir, ignore_errors=True)
            shutil.copytree(backup, settings.chroma_db_dir)
        raise PermissionError(
            "Cannot replace the ChromaDB directory. Close Streamlit, the terminal "
            "chat, and any other process using chroma_db, then retry indexing."
        ) from error
    except Exception:
        if client is not None:
            client.close()
        shutil.rmtree(temporary_dir, ignore_errors=True)
        if backup is not None and backup.exists():
            shutil.rmtree(settings.chroma_db_dir, ignore_errors=True)
            shutil.copytree(backup, settings.chroma_db_dir)
        raise
