"""Build the local PDF index."""

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from chatbot.config import load_settings
from chatbot.pipeline import index_pdfs

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = load_settings()
    settings.pdf_dir.mkdir(parents=True, exist_ok=True)
    count = index_pdfs(settings)
    print(f"Indexed {count} passages successfully.")
