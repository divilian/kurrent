# Generic kurrent-agnostic functions for file control.

import hashlib
from pathlib import Path

import pymupdf

__all__ = [
    "is_pdf",
    "sha256_file",
    "normalize_path",
    "silence_mupdf_messages",
]

def is_pdf(path: str | Path) -> bool:
    path = normalize_path(path)
    if not path.is_file():
        return False
    with path.open("rb") as f:
        header = f.read(5)
    return header == b"%PDF-"


def sha256_file(path: str | Path) -> str:
    path = normalize_path(path)

    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def normalize_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def silence_mupdf_messages() -> None:
    """Suppress noisy MuPDF parser diagnostics during normal ingestion."""

    pymupdf.TOOLS.mupdf_display_errors(False)
    pymupdf.TOOLS.mupdf_display_warnings(False)
