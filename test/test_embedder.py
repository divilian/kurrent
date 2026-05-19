from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import chromadb
import pytest

from kurrent.chunker import chunker_version
from kurrent.embedder import Embedder
from kurrent.schema import Chunk, Document
from kurrent.state_store import StateStore


class FakeModel:
    def encode(self, texts, convert_to_numpy=True):
        return FakeEmbeddings([
            [float(len(text)), float(i), 1.0]
            for i, text in enumerate(texts)
        ])


class FakeEmbeddings:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "kurrent.db"

    with StateStore(db_path) as store:
        yield store


@pytest.fixture
def embedder(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kurrent.embedder.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    return Embedder(
        chroma_path=tmp_path / "chroma",
        model_name="fake-model",
    )


def make_document(**overrides) -> Document:
    doc_id = str(uuid4())

    values = {
        "doc_id": doc_id,
        "pdf_sha256": f"fake-sha256-{doc_id}",
        "storage_mode": "external",
        "pdf_path": Path("/tmp/example.pdf"),
        "ingested_at": datetime.now(timezone.utc),
        "title": None,
        "authors": None,
        "year": None,
        "doi": None,
    }
    values.update(overrides)

    return Document(**values)


def make_chunk(
    doc_id: str,
    chunk_index: int,
    text: str,
    **overrides,
) -> Chunk:
    values = {
        "doc_id": doc_id,
        "chunker_version": chunker_version(),
        "chunk_index": chunk_index,
        "text": text,
        "text_sha256": f"fake-text-sha256-{chunk_index}",
        "page_start": chunk_index + 1,
        "page_end": chunk_index + 1,
    }
    values.update(overrides)

    return Chunk(**values)


def test_generate_embeddings_returns_lists(embedder):
    embeddings = embedder.generate_embeddings([
        "short",
        "somewhat longer",
    ])

    assert embeddings == [
        [5.0, 0.0, 1.0],
        [15.0, 1.0, 1.0],
    ]


def test_index_chunks_upserts_chunks_into_chroma(store, embedder):
    doc = make_document()
    store.insert_document(doc)

    chunks = [
        make_chunk(doc.doc_id, 0, "first chunk text"),
        make_chunk(doc.doc_id, 1, "second chunk text"),
    ]
    store.insert_chunks(chunks)

    embedder.index_chunks(doc.doc_id, store)

    results = embedder.collection.get(
        where={"doc_id": doc.doc_id},
        include=["documents", "metadatas", "embeddings"],
    )

    assert set(results["ids"]) == {chunk.chunk_id for chunk in chunks}
    assert results["documents"] == [
        "first chunk text",
        "second chunk text",
    ]

    assert len(results["embeddings"]) == 2
    assert results["embeddings"][0][:3].tolist() == [16.0, 0.0, 1.0]
    assert results["embeddings"][1][:3].tolist() == [17.0, 1.0, 1.0]

    first_metadata = results["metadatas"][0]
    assert first_metadata["doc_id"] == doc.doc_id
    assert first_metadata["chunker_version"] == chunker_version()
    assert first_metadata["chunk_index"] == 0
    assert first_metadata["text_sha256"] == "fake-text-sha256-0"
    assert first_metadata["embedding_model"] == "fake-model"
    assert first_metadata["page_start"] == 1
    assert first_metadata["page_end"] == 1


def test_index_chunks_raises_for_document_with_no_chunks(store, embedder):
    doc = make_document()
    store.insert_document(doc)

    with pytest.raises(ValueError):
        embedder.index_chunks(doc.doc_id, store)


def test_index_chunks_raises_for_missing_document(store, embedder):
    with pytest.raises(ValueError):
        embedder.index_chunks("not-a-real-doc-id", store)


def test_index_chunks_is_idempotent_for_same_chunks(store, embedder):
    doc = make_document()
    store.insert_document(doc)

    chunks = [
        make_chunk(doc.doc_id, 0, "first chunk text"),
        make_chunk(doc.doc_id, 1, "second chunk text"),
    ]
    store.insert_chunks(chunks)

    embedder.index_chunks(doc.doc_id, store)
    embedder.index_chunks(doc.doc_id, store)

    results = embedder.collection.get(
        where={"doc_id": doc.doc_id},
        include=["documents", "metadatas", "embeddings"],
    )

    assert len(results["ids"]) == 2
    assert set(results["ids"]) == {chunk.chunk_id for chunk in chunks}


def test_make_collection_name_sanitizes_model_name(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "kurrent.embedder.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    embedder = Embedder(
        chroma_path=tmp_path / "chroma",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    assert "/" not in embedder.collection_name
    assert embedder.collection_name.startswith("kurrent_chunks__")


@pytest.mark.slow
def test_generate_embeddings_with_real_sentence_transformers(tmp_path):
    embedder = Embedder(
        chroma_path=tmp_path / "chroma",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    embeddings = embedder.generate_embeddings([
        "This is a test sentence.",
        "This is another test sentence.",
    ])

    assert len(embeddings) == 2
    assert len(embeddings[0]) > 0
    assert len(embeddings[0]) == len(embeddings[1])
    assert all(isinstance(x, float) for x in embeddings[0][:10])
