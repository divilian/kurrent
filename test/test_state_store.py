from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4

import sqlite3
import pytest

from kurrent.schema import (
    Chunk,
    ConfirmedLink,
    Document,
    ExtractedMetadata,
    ProximityAlertRecord,
)
from kurrent.state_store import StateStore
from test.factories import (
    make_chunk,
    make_document,
    make_proximity_alert,
    make_confirmed_link,
)


@pytest.fixture
def store(tmp_path):
    store = StateStore(tmp_path / "test.db")
    yield store
    store.close()


def test_insert_and_get_document(store):
    doc = make_document()

    store.insert_document(doc)
    retrieved = store.get_document(doc.doc_id)

    assert retrieved == doc


def test_missing_document_returns_none(store):
    assert store.get_document("not-a-real-id") is None


def test_get_or_create_document_uses_metadata_for_new_document(store):
    """Verify that metadata is used when creating a new document."""

    metadata = ExtractedMetadata(
        title="Still Building the Memex",
        authors="Stephen Davies",
        year=2011,
        doi="10.1145/1897816.1897840",
    )

    doc = store.get_or_create_document(
        Path("/tmp/memex.pdf"),
        "fake-sha256",
        metadata=metadata,
    )

    assert doc.title == "Still Building the Memex"
    assert doc.authors == "Stephen Davies"
    assert doc.year == 2011
    assert doc.doi == "10.1145/1897816.1897840"


def test_get_or_create_document_ignores_metadata_for_existing_document(store):
    """Verify that metadata does not overwrite an existing document."""

    first_metadata = ExtractedMetadata(
        title="Original Title",
        authors="Original Author",
        year=2011,
        doi="10.1/original",
    )
    replacement_metadata = ExtractedMetadata(
        title="Replacement Title",
        authors="Replacement Author",
        year=2020,
        doi="10.1/replacement",
    )

    first_doc = store.get_or_create_document(
        Path("/tmp/original.pdf"),
        "fake-sha256",
        metadata=first_metadata,
    )
    second_doc = store.get_or_create_document(
        Path("/tmp/replacement.pdf"),
        "fake-sha256",
        metadata=replacement_metadata,
    )

    assert second_doc == first_doc


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

def test_get_proximity_alert_by_chunk_ids_returns_matching_alert(store):
    doc_a = make_document(doc_id="doc-a")
    doc_b = make_document(doc_id="doc-b")

    store.insert_document(doc_a)
    store.insert_document(doc_b)

    chunk_a = make_chunk(doc_a.doc_id, 0, "Chunk A.")
    chunk_b = make_chunk(doc_b.doc_id, 0, "Chunk B.")

    store.insert_chunks([chunk_a, chunk_b])

    pa = make_proximity_alert(chunk_a, chunk_b)
    store.insert_proximity_alert(pa)

    fetched = store.get_proximity_alert_by_chunk_ids(
        chunk_a.chunk_id,
        chunk_b.chunk_id,
    )

    assert fetched == pa

def test_insert_proximity_alert_is_idempotent_for_same_record(store):
    """Verify that inserting the exact same PA record twice is harmless.

    The second insert should not create a duplicate row.
    """
    doc_a = make_document(doc_id="doc-a")
    doc_b = make_document(doc_id="doc-b")

    store.insert_document(doc_a)
    store.insert_document(doc_b)

    chunk_a = make_chunk(doc_a.doc_id, 0, "Chunk A.")
    chunk_b = make_chunk(doc_b.doc_id, 0, "Chunk B.")

    store.insert_chunks([chunk_a, chunk_b])

    pa = make_proximity_alert(
        chunk_a,
        chunk_b,
        pa_id="pa-1",
    )

    store.insert_proximity_alert(pa)
    store.insert_proximity_alert(pa)

    fetched = store.get_proximity_alert_by_chunk_ids(
        chunk_a.chunk_id,
        chunk_b.chunk_id,
    )

    assert fetched == pa

    row = store.conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM proximity_alerts
        """
    ).fetchone()

    assert row["n"] == 1


def test_insert_proximity_alert_is_idempotent_for_same_chunk_pair(store):
    """Verify that the same chunk pair cannot create multiple PA rows.

    Even if the second PA has a different pa_id, the chunk-pair uniqueness
    constraint should make insertion idempotent.
    """
    doc_a = make_document(doc_id="doc-a")
    doc_b = make_document(doc_id="doc-b")

    store.insert_document(doc_a)
    store.insert_document(doc_b)

    chunk_a = make_chunk(doc_a.doc_id, 0, "Chunk A.")
    chunk_b = make_chunk(doc_b.doc_id, 0, "Chunk B.")

    store.insert_chunks([chunk_a, chunk_b])

    first_pa = make_proximity_alert(
        chunk_a,
        chunk_b,
        pa_id="pa-1",
        explanation="First generated PA.",
    )
    duplicate_pa = make_proximity_alert(
        chunk_a,
        chunk_b,
        pa_id="pa-2",
        explanation="Duplicate generated PA.",
    )

    store.insert_proximity_alert(first_pa)
    store.insert_proximity_alert(duplicate_pa)

    fetched = store.get_proximity_alert_by_chunk_ids(
        chunk_a.chunk_id,
        chunk_b.chunk_id,
    )

    assert fetched == first_pa

    row = store.conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM proximity_alerts
        """
    ).fetchone()

    assert row["n"] == 1


def test_insert_proximity_alert_is_idempotent_for_reversed_pair(store):
    """Verify that PA chunk pairs are treated as undirected.

    Inserting A/B and then B/A should still leave only one stored PA row.
    """
    doc_a = make_document(doc_id="doc-a")
    doc_b = make_document(doc_id="doc-b")

    store.insert_document(doc_a)
    store.insert_document(doc_b)

    chunk_a = make_chunk(doc_a.doc_id, 0, "Chunk A.")
    chunk_b = make_chunk(doc_b.doc_id, 0, "Chunk B.")

    store.insert_chunks([chunk_a, chunk_b])

    forward_pa = make_proximity_alert(
        chunk_a,
        chunk_b,
        pa_id="forward-pa",
    )
    reverse_pa = make_proximity_alert(
        chunk_b,
        chunk_a,
        pa_id="reverse-pa",
    )

    store.insert_proximity_alert(forward_pa)
    store.insert_proximity_alert(reverse_pa)

    fetched = store.get_proximity_alert_by_chunk_ids(
        chunk_b.chunk_id,
        chunk_a.chunk_id,
    )

    assert fetched == forward_pa

    row = store.conn.execute(
        """
        SELECT COUNT(*) AS n
        FROM proximity_alerts
        """
    ).fetchone()

    assert row["n"] == 1
