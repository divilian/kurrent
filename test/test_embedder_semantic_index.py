from __future__ import annotations

from dataclasses import dataclass

from kurrent.embedder import Embedder
from kurrent.pipeline import current_semantic_index_fingerprint
from kurrent.schema import make_chunk_id


@dataclass(slots=True)
class FakeChunk:
    doc_id: str = "doc-1"
    chunker_version: str = "section-aware-fixed-char-2000-v2"
    chunk_index: int = 0
    text: str = "chunk text"
    text_sha256: str = "abc123"
    page_start: int | None = 1
    page_end: int | None = 2
    section_number: str | None = None
    section_title: str | None = None

    @property
    def chunk_id(self) -> str:
        return make_chunk_id(
            self.doc_id,
            self.chunker_version,
            self.chunk_index,
        )


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


def test_semantic_index_fingerprint_includes_embedding_input_version():
    """Changing embedding input construction must create a new Chroma namespace."""

    index_fingerprint = current_semantic_index_fingerprint()

    assert "embedding_input=metadata-enriched-embedding-input-v1" in index_fingerprint


def test_embedding_text_for_chunk_enriches_chunk_with_document_metadata():
    """The embedding input gets metadata, but the chunk text stays clean."""
    from datetime import datetime, timezone
    from pathlib import Path

    from kurrent.schema import Document

    chunk = FakeChunk(
        text="This is the clean chunk text.",
        section_number="2",
        section_title="Personal Knowledge Bases",
    )
    document = Document(
        doc_id="doc-1",
        pdf_sha256="sha",
        storage_mode="managed",
        pdf_path=Path("/tmp/memex.pdf"),
        ingested_at=datetime.now(timezone.utc),
        title="Still Building the Memex",
        authors="Stephen Davies",
        year=2011,
        doi="10.example/memex",
    )

    embedding_text = Embedder.embedding_text_for_chunk(chunk, document)

    assert "Document title: Still Building the Memex" in embedding_text
    assert "Authors: Stephen Davies" in embedding_text
    assert "Year: 2011" in embedding_text
    assert "DOI: 10.example/memex" in embedding_text
    assert "PDF filename: memex.pdf" in embedding_text
    assert "Section: 2 Personal Knowledge Bases" in embedding_text
    assert embedding_text.endswith("\n\nThis is the clean chunk text.")
    assert chunk.text == "This is the clean chunk text."


def test_index_chunks_embeds_metadata_enriched_text_but_stores_clean_documents():
    """Chroma vectors use enriched input while Chroma documents remain clean."""
    from datetime import datetime, timezone
    from pathlib import Path

    import numpy as np

    from kurrent.schema import Document

    class FakeModel:
        def __init__(self):
            self.encoded_texts = []

        def encode(self, texts, convert_to_numpy=True):
            self.encoded_texts.append(list(texts))
            assert convert_to_numpy is True
            return np.array([[0.1, 0.2, 0.3]])

    class FakeCollection:
        def __init__(self):
            self.upsert_kwargs = None

        def upsert(self, **kwargs):
            self.upsert_kwargs = kwargs

    class FakeStore:
        def __init__(self, chunk, document):
            self.chunk = chunk
            self.document = document

        def get_chunks_for_document(self, doc_id, chunker_version):
            assert doc_id == "doc-1"
            return [self.chunk]

        def get_document(self, doc_id):
            assert doc_id == "doc-1"
            return self.document

        def get_document_pipeline_fingerprint(self, doc_id):
            assert doc_id == "doc-1"
            return "text-pipeline"

    chunk = FakeChunk(
        doc_id="doc-1",
        chunker_version="section-aware-fixed-char-2000-v2",
        chunk_index=0,
        text="Only this clean passage should be stored as the Chroma document.",
    )
    document = Document(
        doc_id="doc-1",
        pdf_sha256="sha",
        storage_mode="managed",
        pdf_path=Path("/tmp/memex.pdf"),
        ingested_at=datetime.now(timezone.utc),
        title="Still Building the Memex",
        authors="Stephen Davies",
        year=2011,
    )

    embedder = Embedder.__new__(Embedder)
    embedder.model_name = "fake-model"
    embedder.semantic_index_fingerprint = current_semantic_index_fingerprint()
    embedder.model = FakeModel()
    embedder.collection = FakeCollection()

    embedder.index_chunks("doc-1", FakeStore(chunk, document))

    encoded = embedder.model.encoded_texts[0][0]
    assert "Document title: Still Building the Memex" in encoded
    assert "Authors: Stephen Davies" in encoded
    assert "Only this clean passage" in encoded

    upsert = embedder.collection.upsert_kwargs
    assert upsert["documents"] == [
        "Only this clean passage should be stored as the Chroma document."
    ]
    assert upsert["embeddings"] == [[0.1, 0.2, 0.3]]
    assert upsert["metadatas"][0]["embedding_input_version"] == (
        "metadata-enriched-embedding-input-v1"
    )
