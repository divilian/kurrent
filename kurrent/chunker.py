# Functions to split text into retrievable chunks.
from __future__ import annotations

from pathlib import Path
import hashlib
from collections.abc import Callable, Sequence

import pymupdf

from kurrent.file_utils import is_pdf
from kurrent.state_store import StateStore
from kurrent.schema import Chunk, SectionSpan
from kurrent.sectioner import (
    detect_heading_candidates,
    detect_heading_candidates_with_context,
    make_section_spans_from_headings,
    make_section_spans_from_llm_decisions,
)

__all__ = [
    "chunk_document",
    "chunker_version",
    "make_word_aware_fixed_size_chunks",
    "make_section_aware_fixed_size_chunks",
    "make_section_spans_with_llm",
]


def chunker_version(target_chars: int = 2000) -> str:
    """Return kurrent's current canonical chunker version.

    v2 preserves line/page provenance from SectionSpan objects so individual
    chunks receive their own page ranges instead of inheriting the full parent
    section's page range.
    """

    return f"section-aware-fixed-char-{target_chars}-v2"


def legacy_chunker_version(target_chars: int = 2000) -> str:
    """Return the pre-section-aware chunker version string."""

    return f"word-aware-fixed-char-{target_chars}-v1"


def chunk_document(
    doc_id: str,
    store: StateStore,
    reviewed_headings: Sequence[str] | None = None,
    use_llm_sectioning: bool = True,
    llm_max_pages: int | None = None,
    llm_progress_total_callback: Callable[[int], None] | None = None,
    llm_progress_callback: Callable[[int], None] | None = None,
) -> list[Chunk]:
    """Extract, section, chunk, store, and return chunks for a document.

    By default, kurrent uses the LLM-assisted sectioning path. The rules-based
    path is still available by passing use_llm_sectioning=False.

    reviewed_headings=None means no human-reviewed heading list was supplied,
    so headings/sections are detected automatically according to
    use_llm_sectioning.

    reviewed_headings=list means use that caller-supplied heading list. An
    empty list is meaningful: it says to use no section headings, producing
    unsectioned chunks under the current section-aware chunker version.
    Human-reviewed string headings use the rules-based exact-heading span
    builder, because they are heading strings rather than LLM candidate IDs.

    llm_progress_total_callback and llm_progress_callback are optional hooks
    used by interactive playgrounds to display Ollama progress.
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

    if reviewed_headings is not None:
        sections = make_section_spans_from_headings(
            pdf_path=doc.pdf_path,
            doc_id=doc.doc_id,
            headings=list(reviewed_headings),
        )
    elif use_llm_sectioning:
        sections = make_section_spans_with_llm(
            pdf_path=doc.pdf_path,
            doc_id=doc.doc_id,
            max_pages=llm_max_pages,
            progress_total_callback=llm_progress_total_callback,
            progress_callback=llm_progress_callback,
        )
    else:
        headings = detect_heading_candidates(doc.pdf_path)
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


def make_section_spans_with_llm(
    pdf_path: str | Path,
    doc_id: str,
    max_pages: int | None = None,
    progress_total_callback: Callable[[int], None] | None = None,
    progress_callback: Callable[[int], None] | None = None,
) -> list[SectionSpan]:
    """Detect sections with the LLM-assisted HeadingCandidate pipeline."""

    from kurrent.llm_sectioner import select_section_headings_with_ollama

    candidates = detect_heading_candidates_with_context(
        pdf_path=pdf_path,
        max_pages=max_pages,
    )
    decisions = select_section_headings_with_ollama(
        candidates,
        progress_total_callback=progress_total_callback,
        progress_callback=progress_callback,
    )

    return make_section_spans_from_llm_decisions(
        pdf_path=pdf_path,
        doc_id=doc_id,
        candidates=candidates,
        decisions=decisions,
    )


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
    """Create chunks by splitting each SectionSpan independently.

    When SectionSpan.lines is available, chunk page_start/page_end are derived
    from the actual source lines included in each chunk. This avoids assigning
    every chunk the full page range of its parent section. Older tests or
    callers may still construct SectionSpan objects without line provenance; in
    that case, we fall back to the section-level page range.
    """

    chunks: list[Chunk] = []

    def emit_chunk(
        text: str,
        section: SectionSpan,
        page_start: int | None,
        page_end: int | None,
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
                page_start=page_start,
                page_end=page_end,
                section_index=section.section_index,
                section_number=section.section_number,
                section_title=section.section_title,
            )
        )

    for section in sections:
        current_parts: list[str] = []
        current_start_page: int | None = None
        current_end_page: int | None = None

        def current_text() -> str:
            return " ".join(current_parts).strip()

        def emit_current_chunk() -> None:
            nonlocal current_parts, current_start_page, current_end_page

            emit_chunk(
                text=current_text(),
                section=section,
                page_start=current_start_page,
                page_end=current_end_page,
            )
            current_parts = []
            current_start_page = None
            current_end_page = None

        if section.lines:
            word_page_pairs = [
                (word, section_line.page)
                for section_line in section.lines
                for word in section_line.text.split()
            ]
        else:
            word_page_pairs = [
                (word, None)
                for word in section.text.split()
            ]

        for word, page_num in word_page_pairs:
            candidate_len = len(current_text()) + 1 + len(word)

            if current_parts and candidate_len > target_chars:
                emit_current_chunk()

            if current_start_page is None:
                current_start_page = (
                    page_num
                    if page_num is not None
                    else section.page_start
                )

            current_parts.append(word)
            current_end_page = (
                page_num
                if page_num is not None
                else section.page_end
            )

        emit_current_chunk()

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
