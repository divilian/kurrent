from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import kurrent.cli as cli
from kurrent.pipeline import (
    current_semantic_index_fingerprint,
    current_text_pipeline_fingerprint,
)


@dataclass(slots=True)
class FakeDocument:
    doc_id: str
    pdf_path: Path
    storage_mode: str = "external"
    title: str | None = None
    authors: str | None = None
    year: int | None = None
    doi: str | None = None


class FakeStore:
    def __init__(self, documents, fingerprints):
        self._documents = list(documents)
        self._fingerprints = dict(fingerprints)

    def list_documents(self):
        return list(self._documents)

    def get_document_pipeline_fingerprint(self, doc_id):
        return self._fingerprints.get(doc_id)


class FakeEmbedder:
    def __init__(self, indexed_doc_ids):
        self.indexed_doc_ids = set(indexed_doc_ids)

    def has_document(self, doc_id):
        return doc_id in self.indexed_doc_ids


def test_semantic_index_fingerprint_ignores_reviewed_heading_choices():
    """Verify Chroma namespace fingerprint avoids per-document heading hashes."""

    current_a = current_semantic_index_fingerprint()
    current_b = current_semantic_index_fingerprint()

    reviewed_a = current_text_pipeline_fingerprint(
        reviewed_headings=["1 Introduction", "2 Model"],
    )
    reviewed_b = current_text_pipeline_fingerprint(
        reviewed_headings=["1 Introduction", "3 Results"],
    )

    assert current_a == current_b
    assert reviewed_a != reviewed_b
    assert "reviewed_headings" not in current_a


def test_semantic_refresh_documents_flags_stale_and_unindexed_docs():
    """Verify semantic search maintenance catches stale and missing-index docs."""

    current = current_text_pipeline_fingerprint()
    stale = current_text_pipeline_fingerprint(extractor_version="old-extractor")

    fresh_indexed = FakeDocument("fresh-indexed", Path("fresh-indexed.pdf"))
    fresh_unindexed = FakeDocument("fresh-unindexed", Path("fresh-unindexed.pdf"))
    stale_indexed = FakeDocument("stale-indexed", Path("stale-indexed.pdf"))

    store = FakeStore(
        [fresh_indexed, fresh_unindexed, stale_indexed],
        {
            fresh_indexed.doc_id: current,
            fresh_unindexed.doc_id: current,
            stale_indexed.doc_id: stale,
        },
    )
    embedder = FakeEmbedder({fresh_indexed.doc_id, stale_indexed.doc_id})

    docs = cli.semantic_refresh_documents(store, embedder)

    assert [doc.doc_id for doc in docs] == [
        fresh_unindexed.doc_id,
        stale_indexed.doc_id,
    ]


def test_prompt_refresh_semantic_documents_accepts_default_yes(monkeypatch):
    """Verify the semantic refresh prompt defaults to yes on Enter."""

    monkeypatch.setattr("builtins.input", lambda _prompt: "")

    docs = [FakeDocument("doc-1", Path("one.pdf"))]

    assert cli.prompt_refresh_semantic_documents(docs) is True


def test_prompt_refresh_semantic_documents_allows_no(monkeypatch):
    """Verify the semantic refresh prompt lets the user continue stale."""

    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    docs = [FakeDocument("doc-1", Path("one.pdf"))]

    assert cli.prompt_refresh_semantic_documents(docs) is False


def test_refresh_documents_for_semantic_search_reingests_with_existing_metadata(monkeypatch):
    """Verify semantic refresh reuses stored document metadata without review."""

    calls = []

    def fake_ingest_pdf_with_metadata(**kwargs):
        calls.append(kwargs)
        return cli.IngestOutcome(doc_id="doc-1", already_existed=True)

    monkeypatch.setattr(cli, "ingest_pdf_with_metadata", fake_ingest_pdf_with_metadata)

    document = FakeDocument(
        doc_id="doc-1",
        pdf_path=Path("paper.pdf"),
        storage_mode="external",
        title="Stored Title",
        authors="A. Author",
        year=2025,
        doi="10.example/test",
    )

    refreshed, failed = cli.refresh_documents_for_semantic_search(
        [document],
        store=object(),
        embedder=object(),
    )

    assert (refreshed, failed) == (1, 0)
    assert len(calls) == 1
    call = calls[0]
    assert call["pdf_path"] == document.pdf_path
    assert call["metadata"].title == "Stored Title"
    assert call["metadata"].authors == "A. Author"
    assert call["metadata"].year == 2025
    assert call["metadata"].doi == "10.example/test"
    assert call["metadata_was_reviewed"] is False
    assert call["reviewed_headings"] is None
    assert call["use_llm_sectioning"] is True
