from .config import load_settings


def main() -> None:
    settings = load_settings()
    print(f"Configured PDF directory: {settings.pdf_dir}")
