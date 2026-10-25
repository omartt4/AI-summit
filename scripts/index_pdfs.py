
from pathlib import Path

import chromadb
import ollama
from pypdf import PdfReader

PDF_DIR = Path("data")
DB_DIR = "chroma_db"
COLLECTION_NAME = "pdf_documents"

EMBEDDING_MODEL = "nomic-embed-text"
CHUNK_SIZE = 800
OVERLAP = 150
BATCH_SIZE = 32


def chunk_text(text):
    text = " ".join(text.split())
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - OVERLAP

    return chunks


def generate_embeddings(texts):
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=texts,
    )
    return response["embeddings"]


def main():
    if not PDF_DIR.exists():
        print(f"PDF folder not found: {PDF_DIR}")
        return

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the data folder.")
        return

    client = chromadb.PersistentClient(path=DB_DIR)

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
