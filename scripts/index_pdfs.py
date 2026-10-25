
import os
from pathlib import Path

import chromadb
import ollama
from dotenv import load_dotenv
from pypdf import PdfReader


# Load configuration from .env in the project root
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def get_int(name, default):
    return int(os.getenv(name, str(default)))


PDF_DIR = BASE_DIR / os.getenv("PDF_DIR", "pdfs")
DB_DIR = BASE_DIR / os.getenv("DB_DIR", "chroma_db")
COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME", "pdf_documents"
)


OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST", "http://localhost:11434"
)
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "nomic-embed-text"
)

CHUNK_SIZE = get_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = get_int("CHUNK_OVERLAP", 150)
BATCH_SIZE = get_int("BATCH_SIZE", 32)

if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be positive.")

if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError(
        "CHUNK_OVERLAP must be >= 0 and less than CHUNK_SIZE."
    )

if BATCH_SIZE <= 0:
    raise ValueError("BATCH_SIZE must be positive.")

ollama_client = ollama.Client(host=OLLAMA_HOST)


def chunk_text(text):
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def generate_embeddings(texts):
    response = ollama_client.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return response["embeddings"]


def main():
    if not PDF_DIR.exists():
        print(f"PDF directory not found: {PDF_DIR}")
        return

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in {PDF_DIR}")
        return

    client = chromadb.PersistentClient(path=str(DB_DIR))

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    for pdf_path in pdf_files:
        print(f"\nProcessing: {pdf_path.name}")

        try:
            reader = PdfReader(str(pdf_path))

            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                chunks = chunk_text(text)

                for offset in range(0, len(chunks), BATCH_SIZE):
                    batch = chunks[offset:offset + BATCH_SIZE]
                    embeddings = generate_embeddings(batch)

                    ids = [
                        f"{pdf_path.name}:{page_num}:{i}"
                        for i in range(offset, offset + len(batch))
                    ]

                    metadatas = [
                        {
                            "source": pdf_path.name,
                            "page": page_num,
                        }
                        for _ in batch
                    ]

                    collection.upsert(
                        ids=ids,
                        documents=batch,
                        embeddings=embeddings,
                        metadatas=metadatas,
                    )

            print(f"Finished: {pdf_path.name}")

        except Exception as error:
            print(f"Error processing {pdf_path.name}: {error}")

    print(f"\nTotal stored chunks: {collection.count()}")


if __name__ == "__main__":
    main()
