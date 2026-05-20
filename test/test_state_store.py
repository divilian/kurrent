from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

import sqlite3
import pytest

from kurrent.schema import (
    Chunk,
    ConfirmedLink,
    Document,
    ProximityAlert,
)
from kurrent.state_store import StateStore


@pytest.fixture
def store(tmp_path):
    store = StateStore(tmp_path / "test.db")
    yield store
    store.close()


def make_document(**overrides) -> Document:
    doc_id = str(uuid4())
    fake_sha256 = f"fake-sha256-{doc_id}"
    values = {
        "doc_id": doc_id,
        "pdf_sha256": fake_sha256,
        "storage_mode": "external",
        "pdf_path": Path("/tmp/example.pdf"),
        "ingested_at": datetime.now(timezone.utc),
        "title": "Example Paper",
        "authors": "Ren, Kylo",
        "year": 2015,
        "doi": "10.1234/example",
    }
    values.update(overrides)
    return Document(**values)


def make_chunk(doc_id: str, **overrides) -> Chunk:
    values = {
        "doc_id": doc_id,
        "chunker_version": "word-aware-fixed-char-v1",
        "chunk_index": 0,
        "text": "This is a test chunk.",
        "text_sha256": "chunkhash123",
        "page_start": 1,
        "page_end": 1,
    }
    values.update(overrides)
    return Chunk(**values)


def make_proximity_alert(
    chunk_a: Chunk,
    chunk_b: Chunk,
    **overrides,
) -> ProximityAlert:
    values = {
        "pa_id": str(uuid4()),
        "doc_a_id": chunk_a.doc_id,
        "chunker_a_version": chunk_a.chunker_version,
        "chunk_a_index": chunk_a.chunk_index,
        "doc_b_id": chunk_b.doc_id,
        "chunker_b_version": chunk_b.chunker_version,
        "chunk_b_index": chunk_b.chunk_index,
        "score": 0.87,
        "status": "pending",
        "explanation": "There's a vergence between these two documents.",
        "created_at": datetime.now(timezone.utc),
        "decided_at": None,
    }
    values.update(overrides)
    return ProximityAlert(**values)


def make_confirmed_link(pa_id: str, **overrides) -> ConfirmedLink:
    values = {
        "cl_id": str(uuid4()),
        "pa_id": pa_id,
        "created_at": datetime.now(timezone.utc),
        "relationship_type": "same_claim",
    }
    values.update(overrides)
    return ConfirmedLink(**values)


def test_insert_and_get_document(store):
    doc = make_document()

    store.insert_document(doc)
    retrieved = store.get_document(doc.doc_id)

    assert retrieved == doc


def test_missing_document_returns_none(store):
    assert store.get_document("not-a-real-id") is None


def test_insert_and_get_chunk(store):
    doc = make_document()
    store.insert_document(doc)

    chunk = make_chunk(doc.doc_id)
    store.insert_chunks([chunk])

    retrieved = store.get_chunk_by_parts(
        doc_id=doc.doc_id,
        chunker_version=chunk.chunker_version,
        chunk_index=chunk.chunk_index,
    )

    assert retrieved == chunk


def test_missing_chunk_returns_none(store):
    retrieved = store.get_chunk_by_parts(
        doc_id="not-a-real-id",
        chunker_version="word-aware-fixed-char-v1",
        chunk_index=0,
    )

    assert retrieved is None


def test_insert_and_get_proximity_alert(store):
    doc_a = make_document()
    doc_b = make_document()
    store.insert_document(doc_a)
    store.insert_document(doc_b)

    chunk_a = make_chunk(doc_a.doc_id, chunk_index=0)
    chunk_b = make_chunk(doc_b.doc_id, chunk_index=0)
    store.insert_chunks([chunk_a, chunk_b])

    pa = make_proximity_alert(chunk_a, chunk_b)
    store.insert_proximity_alert(pa)

    retrieved = store.get_proximity_alert(pa.pa_id)

    assert retrieved == pa


def test_missing_proximity_alert_returns_none(store):
    assert store.get_proximity_alert("not-a-real-id") is None


def test_update_proximity_alert_status(store):
    doc_a = make_document()
    doc_b = make_document()
    store.insert_document(doc_a)
    store.insert_document(doc_b)

    chunk_a = make_chunk(doc_a.doc_id, chunk_index=0)
    chunk_b = make_chunk(doc_b.doc_id, chunk_index=0)
    store.insert_chunks([chunk_a, chunk_b])

    pa = make_proximity_alert(chunk_a, chunk_b)
    store.insert_proximity_alert(pa)

    decided_at = datetime.now(timezone.utc)
    store.update_proximity_alert_status(
        pa_id=pa.pa_id,
        status="confirmed",
        decided_at=decided_at,
    )

    retrieved = store.get_proximity_alert(pa.pa_id)

    assert retrieved is not None
    assert retrieved.status == "confirmed"
    assert retrieved.decided_at == decided_at


def test_insert_and_get_confirmed_link(store):
    doc_a = make_document()
    doc_b = make_document()
    store.insert_document(doc_a)
    store.insert_document(doc_b)

    chunk_a = make_chunk(doc_a.doc_id, chunk_index=0)
    chunk_b = make_chunk(doc_b.doc_id, chunk_index=0)
    store.insert_chunks([chunk_a, chunk_b])

    pa = make_proximity_alert(
        chunk_a,
        chunk_b,
        status="confirmed",
        decided_at=datetime.now(timezone.utc),
    )
    store.insert_proximity_alert(pa)

    cl = make_confirmed_link(pa.pa_id)
    store.insert_confirmed_link(cl)

    retrieved = store.get_confirmed_link(cl.cl_id)

    assert retrieved == cl


def test_missing_confirmed_link_returns_none(store):
    assert store.get_confirmed_link("not-a-real-id") is None


def test_invalid_storage_mode_rejected(store):
    doc = make_document(storage_mode="nonsense")

    with pytest.raises(sqlite3.IntegrityError):
        store.insert_document(doc)


def test_duplicate_doc_id_rejected(store):
    doc = make_document()

    store.insert_document(doc)

    with pytest.raises(sqlite3.IntegrityError):
        store.insert_document(doc)


def test_duplicate_chunk_triplet_rejected(store):
    doc = make_document()
    store.insert_document(doc)

    chunk = make_chunk(doc.doc_id)
    store.insert_chunks([chunk])

    with pytest.raises(sqlite3.IntegrityError):
        store.insert_chunks([chunk])


def test_chunk_cannot_refer_to_nonexistent_document(store):
    chunk = make_chunk(doc_id=str(uuid4()))

    with pytest.raises(sqlite3.IntegrityError):
        store.insert_chunks([chunk])


def test_proximity_alert_cannot_refer_to_nonexistent_chunks(store):
    doc_a = make_document()
    doc_b = make_document()
    store.insert_document(doc_a)
    store.insert_document(doc_b)

    chunk_a = make_chunk(doc_a.doc_id, chunk_index=0)
    chunk_b = make_chunk(doc_b.doc_id, chunk_index=0)

    pa = make_proximity_alert(chunk_a, chunk_b)

    with pytest.raises(sqlite3.IntegrityError):
        store.insert_proximity_alert(pa)


def test_invalid_proximity_alert_status_rejected(store):
    doc_a = make_document()
    doc_b = make_document()
    store.insert_document(doc_a)
    store.insert_document(doc_b)

    chunk_a = make_chunk(doc_a.doc_id, chunk_index=0)
    chunk_b = make_chunk(doc_b.doc_id, chunk_index=0)
    store.insert_chunks([chunk_a, chunk_b])

    pa = make_proximity_alert(chunk_a, chunk_b, status="nonsense")

    with pytest.raises(sqlite3.IntegrityError):
        store.insert_proximity_alert(pa)
