from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pymupdf
import pytest

from kurrent.chunker import (
    chunk_document,
    chunker_version,
    extract_pdf_pages,
    make_section_aware_fixed_size_chunks,
    make_word_aware_fixed_size_chunks,
    sha256_text,
)
from kurrent.schema import Document, SectionSpan
from kurrent.state_store import StateStore
from test.factories import make_document


def write_pdf(path: Path, pages: list[str]) -> Path:
    pdf = pymupdf.open()

    for text in pages:
        page = pdf.new_page()
        page.insert_text((72, 72), text)

    pdf.save(path)
    pdf.close()

    return path


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "kurrent.db"

    with StateStore(db_path) as store:
        yield store


def test_sha256_text_is_deterministic():
    first = sha256_text("hello world")
    second = sha256_text("hello world")

    assert first == second
    assert first != sha256_text("goodbye world")


def test_chunker_version_is_section_aware_by_default():
    """Verify that kurrent's canonical chunker is now section-aware."""

    assert chunker_version() == "section-aware-fixed-char-2000-v1"


def test_extract_pdf_pages_returns_one_based_page_text(tmp_path):
    pdf_path = write_pdf(
        tmp_path / "paper.pdf",
        [
            "This is page one.",
            "This is page two.",
        ],
    )

    pages = extract_pdf_pages(pdf_path)

    assert sorted(pages) == [1, 2]
    assert "This is page one." in pages[1]
    assert "This is page two." in pages[2]


def test_make_chunks_returns_single_chunk_for_short_text():
    pages = {
        1: "This is a short document.",
    }

    chunks = make_word_aware_fixed_size_chunks(
        pages=pages,
        doc_id="doc-1",
        target_chars=2000,
    )

    assert len(chunks) == 1

    chunk = chunks[0]
    assert chunk.doc_id == "doc-1"
    assert chunk.chunker_version == "word-aware-fixed-char-2000-v1"
    assert chunk.chunk_index == 0
    assert chunk.text == "This is a short document."
    assert chunk.text_sha256 == sha256_text(chunk.text)
    assert chunk.page_start == 1
    assert chunk.page_end == 1


def test_make_chunks_splits_without_splitting_words():
    pages = {
        1: "alpha beta gamma delta",
    }

    chunks = make_word_aware_fixed_size_chunks(
        pages=pages,
        doc_id="doc-1",
        target_chars=12,
    )

    assert [chunk.text for chunk in chunks] == [
        "alpha beta",
        "gamma delta",
    ]

    assert [chunk.chunk_index for chunk in chunks] == [0, 1]
    assert all(chunk.page_start == 1 for chunk in chunks)
    assert all(chunk.page_end == 1 for chunk in chunks)


def test_make_chunks_can_span_page_boundaries():
    pages = {
        1: "alpha beta",
        2: "gamma delta",
    }

    chunks = make_word_aware_fixed_size_chunks(
        pages=pages,
        doc_id="doc-1",
        target_chars=25,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "alpha beta gamma delta"
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


def test_make_chunks_sorts_pages_before_chunking():
    pages = {
        2: "second page",
        1: "first page",
    }

    chunks = make_word_aware_fixed_size_chunks(
        pages=pages,
        doc_id="doc-1",
        target_chars=2000,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "first page second page"
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2


def test_make_chunks_ignores_empty_pages():
    pages = {
        1: "",
        2: "real text",
        3: "",
    }

    chunks = make_word_aware_fixed_size_chunks(
        pages=pages,
        doc_id="doc-1",
        target_chars=2000,
    )

    assert len(chunks) == 1
    assert chunks[0].text == "real text"
    assert chunks[0].page_start == 2
    assert chunks[0].page_end == 2


def test_make_section_aware_chunks_preserves_section_metadata():
    """Verify that section-aware chunking copies section metadata to chunks."""

    section = SectionSpan(
        doc_id="doc-1",
        section_index=2,
        section_number="3.1",
        section_title="LLM Setup",
        page_start=3,
        page_end=4,
        text="alpha beta gamma delta",
    )

    chunks = make_section_aware_fixed_size_chunks(
        sections=[section],
        doc_id="doc-1",
        target_chars=2000,
    )

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.chunker_version == "section-aware-fixed-char-2000-v1"
    assert chunk.section_index == 2
    assert chunk.section_number == "3.1"
    assert chunk.section_title == "LLM Setup"
    assert chunk.page_start == 3
    assert chunk.page_end == 4


def test_make_section_aware_chunks_does_not_cross_section_boundaries():
    """Verify that section-aware chunks split each section independently."""

    sections = [
        SectionSpan(
            doc_id="doc-1",
            section_index=0,
            section_number="1",
            section_title="Introduction",
            page_start=1,
            page_end=1,
            text="alpha beta",
        ),
        SectionSpan(
            doc_id="doc-1",
            section_index=1,
            section_number="2",
            section_title="Methods",
            page_start=2,
            page_end=2,
            text="gamma delta",
        ),
    ]

    chunks = make_section_aware_fixed_size_chunks(
        sections=sections,
        doc_id="doc-1",
        target_chars=2000,
    )

    assert [chunk.text for chunk in chunks] == [
        "alpha beta",
        "gamma delta",
    ]
    assert [chunk.section_title for chunk in chunks] == [
        "Introduction",
        "Methods",
    ]


def test_chunk_document_stores_chunks(store, tmp_path):
    pdf_path = write_pdf(
        tmp_path / "paper.pdf",
        [
            "This is page one.",
            "This is page two.",
        ],
    )

    doc = make_document(pdf_path)
    store.insert_document(doc)

    chunks = chunk_document(doc.doc_id, store)

    assert len(chunks) == 1

    stored_chunk = store.get_chunk_by_parts(
        doc_id=doc.doc_id,
        chunker_version=chunks[0].chunker_version,
        chunk_index=chunks[0].chunk_index,
    )

    assert stored_chunk == chunks[0]


def test_chunk_document_rejects_missing_document(store):
    with pytest.raises(ValueError):
        chunk_document("not-a-real-doc-id", store)
