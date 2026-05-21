from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from kurrent.chunker import chunker_version
from kurrent.proximity_alerter import ProximityAlerter
from kurrent.schema import Chunk, Document, ProximityAlert, VectorChunkMatch
from kurrent.state_store import StateStore


class FakeEmbedder:
    """Tiny fake of the Embedder API used by ProximityAlerter."""

    def __init__(
        self,
        matches_by_chunk_id: dict[str, list[VectorChunkMatch]],
    ) -> None:
        self.matches_by_chunk_id = matches_by_chunk_id
        self.query_calls: list[dict] = []

    def query_similar_chunks_by_chunk_id(
        self,
        chunk_id: str,
        n_results: int = 10,
        max_distance: float | None = None,
        exclude_doc_ids: list[str] | None = None,
    ) -> list[VectorChunkMatch]:
        self.query_calls.append(
            {
                "chunk_id": chunk_id,
                "n_results": n_results,
                "max_distance": max_distance,
                "exclude_doc_ids": exclude_doc_ids,
            }
        )
        return self.matches_by_chunk_id.get(chunk_id, [])


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "kurrent.db"

    with StateStore(db_path) as store:
        yield store


def make_document(**overrides) -> Document:
    doc_id = str(uuid4())

    values = {
        "doc_id": doc_id,
        "pdf_sha256": f"fake-sha256-{doc_id}",
        "storage_mode": "external",
        "pdf_path": Path(f"/tmp/{doc_id}.pdf"),
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
        "text_sha256": f"fake-text-sha256-{doc_id}-{chunk_index}",
        "page_start": chunk_index + 1,
        "page_end": chunk_index + 1,
    }
    values.update(overrides)

    return Chunk(**values)


def test_find_alerts_for_document_queries_each_source_chunk(store):
    """Verify that document-level alerting queries every source chunk.

    Also verifies that n_results, max_distance, and same-document exclusion are
    passed through to the embedder.
    """
    source_doc = make_document(doc_id="source-doc")
    target_doc = make_document(doc_id="target-doc")

    store.insert_document(source_doc)
    store.insert_document(target_doc)

    source_chunk_0 = make_chunk(source_doc.doc_id, 0, "Source chunk zero.")
    source_chunk_1 = make_chunk(source_doc.doc_id, 1, "Source chunk one.")
    target_chunk = make_chunk(target_doc.doc_id, 0, "Target chunk.")

    store.insert_chunks([source_chunk_0, source_chunk_1, target_chunk])

    embedder = FakeEmbedder(
        matches_by_chunk_id={
            source_chunk_0.chunk_id: [
                VectorChunkMatch(
                    chunk_id=target_chunk.chunk_id,
                    distance=0.12,
                ),
            ],
            source_chunk_1.chunk_id: [],
        }
    )
    alerter = ProximityAlerter(
        state_store=store,
        embedder=embedder,
    )

    alerts = alerter.find_alerts_for_document(
        source_doc.doc_id,
        n_results_per_chunk=7,
        max_distance=0.25,
    )

    assert len(alerts) == 1
    assert embedder.query_calls == [
        {
            "chunk_id": source_chunk_0.chunk_id,
            "n_results": 7,
            "max_distance": 0.25,
            "exclude_doc_ids": [source_doc.doc_id],
        },
        {
            "chunk_id": source_chunk_1.chunk_id,
            "n_results": 7,
            "max_distance": 0.25,
            "exclude_doc_ids": [source_doc.doc_id],
        },
    ]


def test_find_alerts_for_chunk_returns_enriched_proximity_alerts(store):
    """
    Verify that chunk-level alerting returns enriched ProximityAlert objects.

    The alert should use authoritative chunk text and document paths from
    kurrent state, not stale text from the vector index.
    """
    source_doc = make_document(
        doc_id="source-doc",
        pdf_path=Path("/tmp/source.pdf"),
    )
    target_doc = make_document(
        doc_id="target-doc",
        pdf_path=Path("/tmp/target.pdf"),
    )

    store.insert_document(source_doc)
    store.insert_document(target_doc)

    source_chunk = make_chunk(
        source_doc.doc_id,
        0,
        "The source text from kurrent state.",
        page_start=3,
        page_end=4,
    )
    target_chunk = make_chunk(
        target_doc.doc_id,
        2,
        "The target text from kurrent state.",
        page_start=9,
        page_end=10,
    )

    store.insert_chunks([source_chunk, target_chunk])

    embedder = FakeEmbedder(
        matches_by_chunk_id={
            source_chunk.chunk_id: [
                VectorChunkMatch(
                    chunk_id=target_chunk.chunk_id,
                    distance=0.18,
                    text="stale target text from vector store",
                ),
            ],
        }
    )
    alerter = ProximityAlerter(
        state_store=store,
        embedder=embedder,
    )

    alerts = alerter.find_alerts_for_chunk(
        source_chunk=source_chunk,
        source_path=source_doc.pdf_path,
        n_results=5,
        max_distance=0.30,
        exclude_doc_ids=[source_doc.doc_id],
    )

    assert alerts == [
        ProximityAlert(
            source_chunk_id=source_chunk.chunk_id,
            target_chunk_id=target_chunk.chunk_id,
            distance=0.18,
            source_doc_id=source_doc.doc_id,
            target_doc_id=target_doc.doc_id,
            source_text="The source text from kurrent state.",
            target_text="The target text from kurrent state.",
            source_path=Path("/tmp/source.pdf"),
            target_path=Path("/tmp/target.pdf"),
            source_page_start=3,
            source_page_end=4,
            target_page_start=9,
            target_page_end=10,
        )
    ]


def test_find_alerts_for_chunk_skips_self_match(store):
    """
    Verify that a chunk is not reported as proximate to itself.

    Chroma may return the queried chunk among the nearest neighbors, but the
    alerter should skip that self-match.
    """
    doc = make_document(doc_id="doc-a")
    target_doc = make_document(doc_id="doc-b")

    store.insert_document(doc)
    store.insert_document(target_doc)

    source_chunk = make_chunk(doc.doc_id, 0, "Source chunk.")
    target_chunk = make_chunk(target_doc.doc_id, 0, "Target chunk.")

    store.insert_chunks([source_chunk, target_chunk])

    embedder = FakeEmbedder(
        matches_by_chunk_id={
            source_chunk.chunk_id: [
                VectorChunkMatch(
                    chunk_id=source_chunk.chunk_id,
                    distance=0.00,
                ),
                VectorChunkMatch(
                    chunk_id=target_chunk.chunk_id,
                    distance=0.20,
                ),
            ],
        }
    )
    alerter = ProximityAlerter(
        state_store=store,
        embedder=embedder,
    )

    alerts = alerter.find_alerts_for_chunk(
        source_chunk=source_chunk,
        source_path=doc.pdf_path,
        n_results=2,
    )

    assert len(alerts) == 1
    assert alerts[0].target_chunk_id == target_chunk.chunk_id


def test_find_alerts_for_document_preserves_source_chunk_order(store):
    """
    Verify that document-level alerts preserve source chunk processing order.

    Matches for each source chunk should be appended in the order returned by
    the embedder before moving to the next source chunk.
    """
    source_doc = make_document(doc_id="source-doc")
    target_doc = make_document(doc_id="target-doc")

    store.insert_document(source_doc)
    store.insert_document(target_doc)

    source_chunk_0 = make_chunk(source_doc.doc_id, 0, "Source chunk zero.")
    source_chunk_1 = make_chunk(source_doc.doc_id, 1, "Source chunk one.")
    target_chunk_0 = make_chunk(target_doc.doc_id, 0, "Target chunk zero.")
    target_chunk_1 = make_chunk(target_doc.doc_id, 1, "Target chunk one.")
    target_chunk_2 = make_chunk(target_doc.doc_id, 2, "Target chunk two.")

    store.insert_chunks([
        source_chunk_0,
        source_chunk_1,
        target_chunk_0,
        target_chunk_1,
        target_chunk_2,
    ])

    embedder = FakeEmbedder(
        matches_by_chunk_id={
            source_chunk_0.chunk_id: [
                VectorChunkMatch(
                    chunk_id=target_chunk_1.chunk_id,
                    distance=0.20,
                ),
                VectorChunkMatch(
                    chunk_id=target_chunk_0.chunk_id,
                    distance=0.10,
                ),
            ],
            source_chunk_1.chunk_id: [
                VectorChunkMatch(
                    chunk_id=target_chunk_2.chunk_id,
                    distance=0.30,
                ),
            ],
        }
    )
    alerter = ProximityAlerter(
        state_store=store,
        embedder=embedder,
    )

    alerts = alerter.find_alerts_for_document(source_doc.doc_id)

    assert [alert.source_chunk_id for alert in alerts] == [
        source_chunk_0.chunk_id,
        source_chunk_0.chunk_id,
        source_chunk_1.chunk_id,
    ]
    assert [alert.target_chunk_id for alert in alerts] == [
        target_chunk_1.chunk_id,
        target_chunk_0.chunk_id,
        target_chunk_2.chunk_id,
    ]


def test_find_alerts_for_document_raises_for_document_with_no_chunks(store):
    """
    Verify that document-level alerting fails for a document with no chunks.

    A document without chunks cannot be used as a proximity-alert source.
    """
    doc = make_document(doc_id="doc-a")
    store.insert_document(doc)

    embedder = FakeEmbedder(matches_by_chunk_id={})
    alerter = ProximityAlerter(
        state_store=store,
        embedder=embedder,
    )

    with pytest.raises(ValueError, match="No chunks found for document"):
        alerter.find_alerts_for_document(doc.doc_id)


def test_find_alerts_for_document_raises_for_missing_document(store):
    """
    Verify that alerting fails cleanly for an unknown document id.

    Since no chunks can be found for the missing document, the alerter should
    raise a ValueError.
    """
    embedder = FakeEmbedder(matches_by_chunk_id={})
    alerter = ProximityAlerter(
        state_store=store,
        embedder=embedder,
    )

    with pytest.raises(ValueError, match="No chunks found for document"):
        alerter.find_alerts_for_document("missing-doc")


def test_find_alerts_for_chunk_raises_for_missing_target_chunk(store):
    """
    Verify that stale vector results are rejected.

    If the vector index returns a target chunk id that is missing from kurrent
    state, the alerter should raise a ValueError.
    """
    source_doc = make_document(doc_id="source-doc")
    store.insert_document(source_doc)

    source_chunk = make_chunk(source_doc.doc_id, 0, "Source chunk.")
    store.insert_chunks([source_chunk])

    embedder = FakeEmbedder(
        matches_by_chunk_id={
            source_chunk.chunk_id: [
                VectorChunkMatch(
                    chunk_id="missing-doc:simple-v1:0",
                    distance=0.12,
                ),
            ],
        }
    )
    alerter = ProximityAlerter(
        state_store=store,
        embedder=embedder,
    )

    with pytest.raises(ValueError, match="not found in kurrent state"):
        alerter.find_alerts_for_chunk(
            source_chunk=source_chunk,
            source_path=source_doc.pdf_path,
        )


def test_find_alerts_for_chunk_returns_empty_list_for_no_matches(store):
    """
    Verify that chunk-level alerting returns an empty list for no matches.

    This is the normal no-alerts case, not an error.
    """
    source_doc = make_document(doc_id="source-doc")
    store.insert_document(source_doc)

    source_chunk = make_chunk(source_doc.doc_id, 0, "Source chunk.")
    store.insert_chunks([source_chunk])

    embedder = FakeEmbedder(matches_by_chunk_id={})
    alerter = ProximityAlerter(
        state_store=store,
        embedder=embedder,
    )

    alerts = alerter.find_alerts_for_chunk(
        source_chunk=source_chunk,
        source_path=source_doc.pdf_path,
    )

    assert alerts == []
