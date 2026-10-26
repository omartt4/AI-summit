"""PDF discovery and page-by-page text extraction."""

import logging
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from pypdf import PdfReader

from .chunking import chunk_id, chunk_text

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    source: str
    page: int
    document_id: str


def load_pdf_passages(pdf_path: Path, chunk_size: int, overlap: int) -> list[Passage]:
    """Extract passages with PyMuPDF, falling back to pypdf when needed."""
    document_id = f"{pdf_path.name}:{pdf_path.stat().st_size}:{pdf_path.stat().st_mtime_ns}"
    passages: list[Passage] = []
    try:
        with pymupdf.open(pdf_path) as reader:
            pages = [(page.number + 1, page.get_text("text")) for page in reader]
    except (pymupdf.FileDataError, OSError, RuntimeError) as error:
        LOGGER.warning("PyMuPDF could not read %s; trying pypdf: %s", pdf_path.name, error)
        reader = PdfReader(str(pdf_path))
        pages = [
            (page_number, page.extract_text() or "")
            for page_number, page in enumerate(reader.pages, start=1)
        ]

    for page_number, text in pages:
        chunks = chunk_text(text, chunk_size, overlap)
        if not chunks:
            LOGGER.warning("No extractable text in %s page %s", pdf_path.name, page_number)
        for index, chunk in enumerate(chunks):
            passages.append(
                Passage(
                    id=chunk_id(document_id, page_number, index),
                    text=chunk,
                    source=pdf_path.name,
                    page=page_number,
                    document_id=document_id,
                )
            )
    if not passages:
        LOGGER.warning("No extractable text found in %s; scanned PDFs need OCR", pdf_path)
    return passages


def discover_pdfs(pdf_dir: Path) -> list[Path]:
    return sorted(path for path in pdf_dir.glob("*.pdf") if path.is_file())
