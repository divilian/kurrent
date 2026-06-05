from __future__ import annotations

from dataclasses import dataclass

from kurrent.embedder import Embedder
from kurrent.pipeline import current_semantic_index_fingerprint


@dataclass(slots=True)
class FakeChunk:
    doc_id: str = "doc-1"
    chunker_version: str = "section-aware-fixed-char-2000-v2"
    chunk_index: int = 0
    text: str = "chunk text"
    text_sha256: str = "abc123"
    page_start: int | None = 1
    page_end: int | None = 2


def test_chroma_collection_name_uses_semantic_index_fingerprint():
    """Verify Chroma collection namespace is not keyed only by chunker version."""

    index_fingerprint = current_semantic_index_fingerprint()
    collection_name = Embedder._make_collection_name(
        semantic_index_fingerprint=index_fingerprint,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    assert "semantic-index-fingerprint-v1" in collection_name
    assert "layout-aware-pymupdf-v2" in collection_name
    assert "sentence-transformers_all-MiniLM-L6-v2" in collection_name


def test_chroma_chunk_metadata_records_text_and_index_fingerprints():
    """Verify Chroma metadata keeps both document text and index fingerprints."""

    chunk = FakeChunk()
    metadata = Embedder._metadata_for_chunk(
        Embedder.__new__(Embedder),
        chunk,
        pipeline_fingerprint="text-pipeline",
        semantic_index_fingerprint="semantic-index",
    )

    assert metadata["doc_id"] == "doc-1"
    assert metadata["text_pipeline_fingerprint"] == "text-pipeline"
    assert metadata["semantic_index_fingerprint"] == "semantic-index"
