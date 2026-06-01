"""Managed PDF storage helpers for kurrent."""

from __future__ import annotations

from pathlib import Path
import re
import shutil

from kurrent.file_utils import normalize_path, sha256_file

__all__ = [
    "safe_pdf_stem",
    "managed_pdf_filename",
    "managed_pdf_path",
    "copy_pdf_to_managed_store",
]

_FILENAME_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_REPEATED_DASH_RE = re.compile(r"-+")


def safe_pdf_stem(source_path: Path, max_length: int = 80) -> str:
    """Return a readable, filesystem-safe stem derived from a source filename."""

    stem = source_path.stem.strip()
    stem = _FILENAME_SAFE_RE.sub("-", stem)
    stem = _REPEATED_DASH_RE.sub("-", stem)
    stem = stem.strip("-._")

    if not stem:
        stem = "document"

    if len(stem) > max_length:
        stem = stem[:max_length].rstrip("-._") or "document"

    return stem


def managed_pdf_filename(source_path: Path, pdf_sha256: str) -> str:
    """Return the readable managed filename for a PDF.

    The original filename stem keeps the managed PDF directory inspectable;
    the hash suffix disambiguates same-named but different PDFs.
    """

    return f"{safe_pdf_stem(source_path)}--{pdf_sha256[:12]}.pdf"


def managed_pdf_path(
    source_path: Path,
    pdfs_dir: Path,
    pdf_sha256: str,
) -> Path:
    """Return the managed destination path for a source PDF."""

    return pdfs_dir / managed_pdf_filename(source_path, pdf_sha256)


def copy_pdf_to_managed_store(
    source_path: Path,
    pdfs_dir: Path,
    pdf_sha256: str,
) -> Path:
    """Copy a PDF into kurrent's managed PDF directory if needed.

    If the destination already exists, verify that it has the expected full
    content hash. This guards against the extremely unlikely event of a short
    hash filename collision or accidental manual tampering.
    """

    source_path = normalize_path(source_path)
    pdfs_dir = Path(pdfs_dir).expanduser().resolve()
    pdfs_dir.mkdir(parents=True, exist_ok=True)

    destination = managed_pdf_path(source_path, pdfs_dir, pdf_sha256)

    if destination.exists():
        existing_sha256 = sha256_file(destination)

        if existing_sha256 != pdf_sha256:
            raise ValueError(
                "Managed PDF filename collision or corrupted managed file: "
                f"{destination}. Expected SHA-256 {pdf_sha256}, "
                f"found {existing_sha256}."
            )

        return destination

    shutil.copy2(source_path, destination)

    copied_sha256 = sha256_file(destination)

    if copied_sha256 != pdf_sha256:
        try:
            destination.unlink()
        except OSError:
            pass

        raise ValueError(
            "Managed PDF copy failed hash verification: "
            f"{destination}. Expected SHA-256 {pdf_sha256}, "
            f"found {copied_sha256}."
        )

    return destination
