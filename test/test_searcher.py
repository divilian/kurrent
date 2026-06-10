from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from kurrent.chunker import chunker_version
from kurrent.schema import (
    Chunk,
    ChunkHit,
    Document,
    DocumentHit,
    VectorChunkMatch,
)
from kurrent.searcher import Searcher
from kurrent.state_store import StateStore
from test.factories import make_chunk, make_document


class FakeEmbedder:
    """Tiny fake of the Embedder API used by Searcher."""

    def __init__(self, matches: list[VectorChunkMatch]) -> None:
        self.matches = matches
        self.query_calls: list[dict] = []

    def query_chunks(
        self,
        search_text: str,
        n_results: int = 10,
        max_distance: float | None = None,
        exclude_doc_ids: list[str] | None = None,
    ) -> list[VectorChunkMatch]:
        self.query_calls.append(
            {
                "search_text": search_text,
                "n_results": n_results,
                "max_distance": max_distance,
                "exclude_doc_ids": exclude_doc_ids,
            }
        )
        return self.matches


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "kurrent.db"

    with StateStore(db_path) as store:
        yield store


def test_semantic_chunk_search_queries_embedder_with_expected_arguments(store):
    embedder = FakeEmbedder(matches=[])
    searcher = Searcher(
        state_store=store,
        embedder=embedder,
    )

    hits = searcher.semantic_chunk_search(
        "bounded confidence models",
        n_results=7,
        max_distance=0.25,
        exclude_doc_ids=["doc-to-exclude"],
    )

    assert hits == []
    assert embedder.query_calls == [
        {
            "search_text": "bounded confidence models",
            "n_results": 40,
            "max_distance": 0.25,
            "exclude_doc_ids": ["doc-to-exclude"],
        }
    ]


def test_semantic_chunk_search_returns_enriched_chunk_hits(store):
    doc_a = make_document(
        doc_id="doc-a",
        pdf_path=Path("/tmp/doc-a.pdf"),
        title="Document A",
    )
    doc_b = make_document(
        doc_id="doc-b",
        pdf_path=Path("/tmp/doc-b.pdf"),
        title="Document B",
    )

    store.insert_document(doc_a)
    store.insert_document(doc_b)

    chunk_a = make_chunk(
        doc_id=doc_a.doc_id,
        chunk_index=0,
        text="SQLite stores durable kurrent state.",
        page_start=3,
        page_end=4,
    )
    chunk_b = make_chunk(
        doc_id=doc_b.doc_id,
        chunk_index=1,
        text="Embeddings support semantic chunk search.",
        page_start=8,
        page_end=9,
    )

    store.insert_chunks([chunk_a, chunk_b])

    embedder = FakeEmbedder(
        matches=[
            VectorChunkMatch(
                chunk_id=chunk_b.chunk_id,
                distance=0.11,
                text="stale vector-side text for B",
            ),
            VectorChunkMatch(
                chunk_id=chunk_a.chunk_id,
                distance=0.22,
                text="stale vector-side text for A",
            ),
        ]
    )
    searcher = Searcher(
        state_store=store,
        embedder=embedder,
    )

    hits = searcher.semantic_chunk_search("semantic search", n_results=2)

    assert [hit.chunk_id for hit in hits] == [chunk_b.chunk_id, chunk_a.chunk_id]
    assert [hit.distance for hit in hits] == [0.11, 0.22]
    assert [hit.text for hit in hits] == [
        "Embeddings support semantic chunk search.",
        "SQLite stores durable kurrent state.",
    ]
    assert [hit.path for hit in hits] == [
        Path("/tmp/doc-b.pdf"),
        Path("/tmp/doc-a.pdf"),
    ]
    assert [hit.title for hit in hits] == ["Document B", "Document A"]
    assert [hit.page_start for hit in hits] == [8, 3]
    assert [hit.page_end for hit in hits] == [9, 4]
    assert hits[0].score is not None
    assert "semantic" in hits[0].match_reasons


def test_semantic_chunk_search_ranks_semantic_candidates_by_similarity(store):
    doc = make_document(doc_id="doc-a")
    store.insert_document(doc)

    chunks = [
        make_chunk(doc.doc_id, 0, "First chunk."),
        make_chunk(doc.doc_id, 1, "Second chunk."),
        make_chunk(doc.doc_id, 2, "Third chunk."),
    ]
    store.insert_chunks(chunks)

    embedder = FakeEmbedder(
        matches=[
            VectorChunkMatch(chunk_id=chunks[2].chunk_id, distance=0.30),
            VectorChunkMatch(chunk_id=chunks[0].chunk_id, distance=0.10),
            VectorChunkMatch(chunk_id=chunks[1].chunk_id, distance=0.20),
        ]
    )
    searcher = Searcher(
        state_store=store,
        embedder=embedder,
    )

    hits = searcher.semantic_chunk_search("some query", n_results=3)

    assert [hit.chunk_id for hit in hits] == [
        chunks[0].chunk_id,
        chunks[1].chunk_id,
        chunks[2].chunk_id,
    ]


def test_semantic_chunk_search_uses_state_store_text_as_authoritative(store):
    doc = make_document(doc_id="doc-a")
    store.insert_document(doc)

    chunk = make_chunk(
        doc.doc_id,
        0,
        "Authoritative text from SQLite.",
    )
    store.insert_chunks([chunk])

    embedder = FakeEmbedder(
        matches=[
            VectorChunkMatch(
                chunk_id=chunk.chunk_id,
                distance=0.12,
                text="Text copied from Chroma.",
            )
        ]
    )
    searcher = Searcher(
        state_store=store,
        embedder=embedder,
    )

    hits = searcher.semantic_chunk_search("some query", n_results=1)

    assert hits[0].text == "Authoritative text from SQLite."


def test_semantic_chunk_search_raises_for_missing_chunk(store):
    embedder = FakeEmbedder(
        matches=[
            VectorChunkMatch(
                chunk_id="missing-doc:simple-v1:0",
                distance=0.12,
                text="orphan vector result",
            )
        ]
    )
    searcher = Searcher(
        state_store=store,
        embedder=embedder,
    )

    with pytest.raises(ValueError, match="not found in kurrent state"):
        searcher.semantic_chunk_search("orphan query", n_results=1)


def test_semantic_document_search_groups_chunks_by_document(store):
    doc_a = make_document(
        doc_id="doc-a",
        pdf_path=Path("/tmp/doc-a.pdf"),
        title="Document A",
        authors="Alice A.",
        year=2020,
    )
    doc_b = make_document(
        doc_id="doc-b",
        pdf_path=Path("/tmp/doc-b.pdf"),
        title="Document B",
        authors="Bob B.",
        year=2021,
    )

    store.insert_document(doc_a)
    store.insert_document(doc_b)

    chunk_a0 = make_chunk(doc_a.doc_id, 0, "Doc A first relevant chunk.")
    chunk_a1 = make_chunk(doc_a.doc_id, 1, "Doc A best relevant chunk.")
    chunk_b0 = make_chunk(doc_b.doc_id, 0, "Doc B relevant chunk.")

    store.insert_chunks([chunk_a0, chunk_a1, chunk_b0])

    embedder = FakeEmbedder(
        matches=[
            VectorChunkMatch(chunk_id=chunk_a0.chunk_id, distance=0.30),
            VectorChunkMatch(chunk_id=chunk_b0.chunk_id, distance=0.20),
            VectorChunkMatch(chunk_id=chunk_a1.chunk_id, distance=0.10),
        ]
    )
    searcher = Searcher(
        state_store=store,
        embedder=embedder,
    )

    hits = searcher.semantic_document_search(
        "some query",
        max_documents=10,
    )

    assert hits == [
        DocumentHit(
            doc_id=doc_a.doc_id,
            path=Path("/tmp/doc-a.pdf"),
            title="Document A",
            authors="Alice A.",
            year=2020,
            score=0.90,
            best_chunk_id=chunk_a1.chunk_id,
        ),
        DocumentHit(
            doc_id=doc_b.doc_id,
            path=Path("/tmp/doc-b.pdf"),
            title="Document B",
            authors="Bob B.",
            year=2021,
            score=0.80,
            best_chunk_id=chunk_b0.chunk_id,
        ),
    ]


def test_semantic_document_search_ranks_by_best_chunk_distance(store):
    doc_a = make_document(doc_id="doc-a", pdf_path=Path("/tmp/doc-a.pdf"))
    doc_b = make_document(doc_id="doc-b", pdf_path=Path("/tmp/doc-b.pdf"))
    doc_c = make_document(doc_id="doc-c", pdf_path=Path("/tmp/doc-c.pdf"))

    store.insert_document(doc_a)
    store.insert_document(doc_b)
    store.insert_document(doc_c)

    chunk_a0 = make_chunk(doc_a.doc_id, 0, "Doc A okay chunk.")
    chunk_a1 = make_chunk(doc_a.doc_id, 1, "Doc A best chunk.")
    chunk_b0 = make_chunk(doc_b.doc_id, 0, "Doc B chunk.")
    chunk_c0 = make_chunk(doc_c.doc_id, 0, "Doc C chunk.")

    store.insert_chunks([chunk_a0, chunk_a1, chunk_b0, chunk_c0])

    embedder = FakeEmbedder(
        matches=[
            VectorChunkMatch(chunk_id=chunk_c0.chunk_id, distance=0.40),
            VectorChunkMatch(chunk_id=chunk_a0.chunk_id, distance=0.30),
            VectorChunkMatch(chunk_id=chunk_b0.chunk_id, distance=0.20),
            VectorChunkMatch(chunk_id=chunk_a1.chunk_id, distance=0.10),
        ]
    )
    searcher = Searcher(
        state_store=store,
        embedder=embedder,
    )

    hits = searcher.semantic_document_search(
        "some query",
        max_documents=10,
    )

    assert [hit.doc_id for hit in hits] == [
        doc_a.doc_id,
        doc_b.doc_id,
        doc_c.doc_id,
    ]
    assert [hit.score for hit in hits] == [0.90, 0.80, 0.60]
    assert [hit.best_chunk_id for hit in hits] == [
        chunk_a1.chunk_id,
        chunk_b0.chunk_id,
        chunk_c0.chunk_id,
    ]


def test_semantic_document_search_respects_max_documents(store):
    doc_a = make_document(doc_id="doc-a", pdf_path=Path("/tmp/doc-a.pdf"))
    doc_b = make_document(doc_id="doc-b", pdf_path=Path("/tmp/doc-b.pdf"))
    doc_c = make_document(doc_id="doc-c", pdf_path=Path("/tmp/doc-c.pdf"))

    store.insert_document(doc_a)
    store.insert_document(doc_b)
    store.insert_document(doc_c)

    chunk_a = make_chunk(doc_a.doc_id, 0, "Best chunk.")
    chunk_b = make_chunk(doc_b.doc_id, 0, "Second best chunk.")
    chunk_c = make_chunk(doc_c.doc_id, 0, "Third best chunk.")

    store.insert_chunks([chunk_a, chunk_b, chunk_c])

    embedder = FakeEmbedder(
        matches=[
            VectorChunkMatch(chunk_id=chunk_a.chunk_id, distance=0.10),
            VectorChunkMatch(chunk_id=chunk_b.chunk_id, distance=0.20),
            VectorChunkMatch(chunk_id=chunk_c.chunk_id, distance=0.30),
        ]
    )
    searcher = Searcher(
        state_store=store,
        embedder=embedder,
    )

    hits = searcher.semantic_document_search(
        "some query",
        max_documents=2,
    )

    assert [hit.doc_id for hit in hits] == [
        doc_a.doc_id,
        doc_b.doc_id,
    ]


def test_semantic_document_search_passes_arguments_to_chunk_search(store):
    embedder = FakeEmbedder(matches=[])
    searcher = Searcher(
        state_store=store,
        embedder=embedder,
    )

    hits = searcher.semantic_document_search(
        "bounded confidence",
        max_documents=4,
        max_distance=0.25,
    )

    assert hits == []
    assert embedder.query_calls == [
        {
            "search_text": "bounded confidence",
            "n_results": 125,
            "max_distance": 0.25,
            "exclude_doc_ids": None,
        }
    ]


def test_semantic_document_search_returns_empty_list_for_no_chunk_hits(store):
    embedder = FakeEmbedder(matches=[])
    searcher = Searcher(
        state_store=store,
        embedder=embedder,
    )

    hits = searcher.semantic_document_search(
        "no matching documents",
        max_documents=10,
    )

    assert hits == []


def test_semantic_document_search_raises_for_missing_chunk(store):
    embedder = FakeEmbedder(
        matches=[
            VectorChunkMatch(
                chunk_id="missing-doc:simple-v1:0",
                distance=0.10,
            ),
        ]
    )
    searcher = Searcher(
        state_store=store,
        embedder=embedder,
    )

    with pytest.raises(ValueError, match="not found in kurrent state"):
        searcher.semantic_document_search(
            "orphaned vector result",
            max_documents=10,
        )


def test_semantic_chunk_search_lexically_rescues_chunk_missing_from_vector_results(store):
    doc_semantic = make_document(doc_id="doc-semantic", title="Semantic")
    doc_lexical = make_document(doc_id="doc-lexical", title="Lexical")
    store.insert_document(doc_semantic)
    store.insert_document(doc_lexical)

    semantic_chunk = make_chunk(
        doc_semantic.doc_id,
        0,
        "A generic discussion of database systems.",
    )
    lexical_chunk = make_chunk(
        doc_lexical.doc_id,
        0,
        "This article defines a personal knowledge base for scholars.",
    )
    store.insert_chunks([semantic_chunk, lexical_chunk])

    embedder = FakeEmbedder(
        matches=[VectorChunkMatch(chunk_id=semantic_chunk.chunk_id, distance=0.20)]
    )
    searcher = Searcher(state_store=store, embedder=embedder)

    hits = searcher.semantic_chunk_search("personal knowledge base", n_results=2)

    assert [hit.chunk_id for hit in hits] == [
        lexical_chunk.chunk_id,
        semantic_chunk.chunk_id,
    ]
    assert hits[0].distance is None
    assert "lexical" in hits[0].match_reasons


def test_semantic_chunk_search_metadata_rescues_document_missing_from_vector_results(store):
    doc_semantic = make_document(doc_id="doc-semantic", title="Semantic")
    doc_metadata = make_document(
        doc_id="doc-metadata",
        title="Still Building the Memex",
        authors="Stephen Davies",
        year=2011,
    )
    store.insert_document(doc_semantic)
    store.insert_document(doc_metadata)

    semantic_chunk = make_chunk(
        doc_semantic.doc_id,
        0,
        "A generic discussion of memories and databases.",
    )
    metadata_chunk = make_chunk(
        doc_metadata.doc_id,
        0,
        "This introduction motivates personal research tools.",
    )
    store.insert_chunks([semantic_chunk, metadata_chunk])

    embedder = FakeEmbedder(
        matches=[VectorChunkMatch(chunk_id=semantic_chunk.chunk_id, distance=0.20)]
    )
    searcher = Searcher(state_store=store, embedder=embedder)

    hits = searcher.semantic_chunk_search("Still Building the Memex", n_results=2)

    assert hits[0].chunk_id == metadata_chunk.chunk_id
    assert hits[0].distance is None
    assert "metadata" in hits[0].match_reasons
