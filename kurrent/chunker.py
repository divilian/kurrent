# Functions to split text into retrievable chunks.
from __future__ import annotations

from pathlib import Path
import hashlib
from collections.abc import Sequence

import pymupdf

from kurrent.file_utils import is_pdf
from kurrent.state_store import StateStore
from kurrent.schema import Chunk, SectionSpan
from kurrent.sectioner import (
    detect_heading_candidates,
    make_section_spans_from_headings,
)

__all__ = [
    "chunk_document",
    "chunker_version",
    "make_word_aware_fixed_size_chunks",
    "make_section_aware_fixed_size_chunks",
]


def chunker_version(target_chars: int = 2000) -> str:
    """Return kurrent's current canonical chunker version."""

    return f"section-aware-fixed-char-{target_chars}-v1"


def legacy_chunker_version(target_chars: int = 2000) -> str:
    """Return the pre-section-aware chunker version string."""

    return f"word-aware-fixed-char-{target_chars}-v1"


def chunk_document(
    doc_id: str,
    store: StateStore,
    reviewed_headings: Sequence[str] | None = None,
) -> list[Chunk]:
    """Extract, section, chunk, store, and return chunks for a document.

    reviewed_headings=None means no human-reviewed heading list was supplied,
    so headings are detected automatically.

    reviewed_headings=list means use that caller-supplied heading list. An
    empty list is meaningful: it says to use no section headings, producing
    unsectioned chunks under the current section-aware chunker version.
    """

    doc = store.get_document(doc_id)

    if doc is None:
        raise ValueError(f"No such document: {doc_id}")

    existing_chunks = store.get_chunks_for_document(
        doc_id=doc.doc_id,
        chunker_version=chunker_version(),
    )

    if existing_chunks:
        return existing_chunks

    if reviewed_headings is None:
        headings = detect_heading_candidates(doc.pdf_path)
    else:
        headings = list(reviewed_headings)

    sections = make_section_spans_from_headings(
        pdf_path=doc.pdf_path,
        doc_id=doc.doc_id,
        headings=headings,
    )
    chunks = make_section_aware_fixed_size_chunks(
        sections=sections,
        doc_id=doc.doc_id,
    )

    store.insert_chunks(chunks)
    return chunks


def make_word_aware_fixed_size_chunks(
    pages: dict[int, str],
    doc_id: str,
    # Roughly 300-500 English words. Enough to preserve scholarly content - a
    # full paragraph or several related paragraphs - but small enough that
    # embeddings are fairly focused.
    target_chars: int = 2000,
) -> list[Chunk]:
    """Convert page-indexed PDF text into legacy fixed-size chunks.

    This function is retained for tests, comparison, and possible migration
    work. It no longer uses kurrent's canonical chunker_version(); its chunks
    are explicitly marked with the legacy word-aware version string.
    """

    chunks: list[Chunk] = []

    current_parts: list[str] = []
    current_start_page: int | None = None
    current_end_page: int | None = None

    def current_text() -> str:
        return " ".join(current_parts).strip()

    def emit_chunk() -> None:
        text = current_text()
        if not text:
            return

        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunker_version=legacy_chunker_version(target_chars),
                chunk_index=len(chunks),
                text=text,
                text_sha256=sha256_text(text),
                page_start=current_start_page,
                page_end=current_end_page,
            )
        )

    for page_num in sorted(pages):
        words = pages[page_num].split()

        for word in words:
            candidate_len = len(current_text()) + 1 + len(word)

            if current_parts and candidate_len > target_chars:
                emit_chunk()
                current_parts = []
                current_start_page = None
                current_end_page = None

            if current_start_page is None:
                current_start_page = page_num

            current_parts.append(word)
            current_end_page = page_num

    emit_chunk()

    return chunks


def make_section_aware_fixed_size_chunks(
    sections: Sequence[SectionSpan],
    doc_id: str,
    target_chars: int = 2000,
) -> list[Chunk]:
    """Create chunks by splitting each SectionSpan independently."""

    chunks: list[Chunk] = []

    def emit_chunk(
        text: str,
        section: SectionSpan,
    ) -> None:
        text = text.strip()

        if not text:
            return

        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunker_version=chunker_version(target_chars),
                chunk_index=len(chunks),
                text=text,
                text_sha256=sha256_text(text),
                page_start=section.page_start,
                page_end=section.page_end,
                section_index=section.section_index,
                section_number=section.section_number,
                section_title=section.section_title,
            )
        )

    for section in sections:
        current_parts: list[str] = []

        def current_text() -> str:
            return " ".join(current_parts).strip()

        for word in section.text.split():
            candidate_len = len(current_text()) + 1 + len(word)

            if current_parts and candidate_len > target_chars:
                emit_chunk(current_text(), section)
                current_parts = []

            current_parts.append(word)

        emit_chunk(current_text(), section)

    return chunks


def extract_pdf_pages(path: str | Path) -> dict[int, str]:
    """Return a dict mapping 1-based page numbers to extracted page text."""

    pages = {}

    with pymupdf.open(path) as pdf:
        # Use 1-based page numbers because that's what humans use.
        for page_num, page in enumerate(pdf, start=1):
            # Preserve empty pages.
            # (Note to later self: to get coordinates for auto-annotation, we
            # can instead use page.get_text("blocks") (or "words") here, which
            # return positional information.)
            pages[page_num] = page.get_text("text", sort=True) or ""

    return pages


def extract_pages(path: str | Path) -> dict[int, str]:
    """Extract page text from a supported file type."""

    if is_pdf(path):
        return extract_pdf_pages(path)

    raise ValueError(f"File {path} is not a valid PDF!")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


if __name__ == "__main__":

    # Smoke test.
    from tempfile import TemporaryDirectory
    from pprint import pprint

    from kurrent.ingester import ingest_pdf

    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "kurrent.db"

        with StateStore(db_path) as store:
            doc_id = ingest_pdf(
                "~/research/SSI/polarbear/CSSSA16/daviesZontine.pdf",
                store,
            )
            chunks = chunk_document(doc_id, store)

            for i, chunk in enumerate(chunks):
                print(f"Chunk {chunk.chunk_index}:")
                pprint(chunk)

                if i < len(chunks) - 1:
                    input(f"Press Enter for chunk {chunks[i + 1].chunk_index}: ")
