from datetime import datetime, timezone
from pathlib import Path

from kurrent.pipeline import current_text_pipeline_fingerprint
from kurrent.schema import Chunk, Document
from kurrent.state_store import StateStore


def insert_test_document(store: StateStore) -> Document:
    document = Document(
        doc_id="doc-1",
        pdf_sha256="abc123",
        storage_mode="external",
        pdf_path=Path("/tmp/test.pdf"),
        ingested_at=datetime.now(timezone.utc),
        title="Test Document",
    )
    store.insert_document(document)
    return document


def test_text_pipeline_fingerprint_changes_when_extractor_changes() -> None:
    """The freshness key includes extractor version, not only chunker version."""

    old_fingerprint = current_text_pipeline_fingerprint(
        extractor_version="layout-aware-pymupdf-v1",
        chunker_algorithm_version="section-aware-fixed-char-v2",
    )
    new_fingerprint = current_text_pipeline_fingerprint(
        extractor_version="layout-aware-pymupdf-v2",
        chunker_algorithm_version="section-aware-fixed-char-v2",
    )

    assert old_fingerprint != new_fingerprint


def test_text_pipeline_fingerprint_changes_when_reviewed_headings_change() -> None:
    """Human-reviewed headings are part of the derived-text pipeline state."""

    first_fingerprint = current_text_pipeline_fingerprint(
        reviewed_headings=["I Introduction", "II Model"],
    )
    second_fingerprint = current_text_pipeline_fingerprint(
        reviewed_headings=["I Introduction", "II Minimal Model"],
    )

    assert first_fingerprint != second_fingerprint


def test_state_store_tracks_pipeline_fingerprint_independently_of_chunker(
    tmp_path,
) -> None:
    """A document with same-version chunks is stale if its pipeline differs."""

    store = StateStore(tmp_path / "kurrent.db")
    document = insert_test_document(store)

    old_fingerprint = current_text_pipeline_fingerprint(
        extractor_version="layout-aware-pymupdf-v1",
        chunker_algorithm_version="section-aware-fixed-char-v2",
    )
    new_fingerprint = current_text_pipeline_fingerprint(
        extractor_version="layout-aware-pymupdf-v2",
        chunker_algorithm_version="section-aware-fixed-char-v2",
    )

    store.insert_chunks([
        Chunk(
            doc_id=document.doc_id,
            chunker_version="section-aware-fixed-char-2000-v2",
            chunk_index=0,
            text="old extracted text",
            text_sha256="old-hash",
        )
    ])
    store.set_document_pipeline_fingerprint(document.doc_id, old_fingerprint)

    assert store.get_chunks_for_document(
        document.doc_id,
        "section-aware-fixed-char-2000-v2",
    )
    assert store.document_has_current_pipeline(document.doc_id, old_fingerprint)
    assert not store.document_has_current_pipeline(document.doc_id, new_fingerprint)

    store.close()


def test_delete_derived_artifacts_removes_chunks_and_pipeline_state(tmp_path) -> None:
    """Refresh can preserve the document while clearing stale derived artifacts."""

    store = StateStore(tmp_path / "kurrent.db")
    document = insert_test_document(store)
    fingerprint = current_text_pipeline_fingerprint()

    store.insert_chunks([
        Chunk(
            doc_id=document.doc_id,
            chunker_version="section-aware-fixed-char-2000-v2",
            chunk_index=0,
            text="derived text",
            text_sha256="hash",
        )
    ])
    store.set_document_pipeline_fingerprint(document.doc_id, fingerprint)

    store.delete_derived_artifacts_for_document(document.doc_id)

    assert store.get_document(document.doc_id) is not None
    assert store.get_chunks_for_document(
        document.doc_id,
        "section-aware-fixed-char-2000-v2",
    ) == []
    assert store.get_document_pipeline_fingerprint(document.doc_id) is None

    store.close()
