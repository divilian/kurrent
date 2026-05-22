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


def insert_test_document(store):
    """Insert one document for StateStore metadata update tests."""

    document = Document.for_pdf(
        pdf_path=Path("/tmp/example.pdf"),
        pdf_sha256="abc123",
        metadata=ExtractedMetadata(
            title="Old Title",
            authors="Old Author",
            year=1999,
            doi="10.0000/old",
        ),
    )

    store.insert_document(document)

    return document.doc_id


def test_update_document_metadata_updates_selected_fields(store):
    """Verify that all supplied metadata fields are updated."""

    doc_id = insert_test_document(store)

    updated = store.update_document_metadata(
        doc_id,
        title="New Title",
        authors="New Author",
        year=2005,
        doi="10.1234/new",
    )

    assert updated.doc_id == doc_id
    assert updated.title == "New Title"
    assert updated.authors == "New Author"
    assert updated.year == 2005
    assert updated.doi == "10.1234/new"


def test_update_document_metadata_persists_selected_field_updates(store):
    """Verify that metadata updates are written to SQLite, not just returned."""

    doc_id = insert_test_document(store)

    store.update_document_metadata(
        doc_id,
        title="Persisted Title",
        year=2010,
    )

    reloaded = store.get_document(doc_id)

    assert reloaded is not None
    assert reloaded.title == "Persisted Title"
    assert reloaded.year == 2010


def test_update_document_metadata_preserves_none_fields(store):
    """Verify that fields passed as None are left unchanged."""

    doc_id = insert_test_document(store)

    updated = store.update_document_metadata(
        doc_id,
        title="New Title",
    )

    assert updated.title == "New Title"
    assert updated.authors == "Old Author"
    assert updated.year == 1999
    assert updated.doi == "10.0000/old"


def test_update_document_metadata_rejects_empty_update(store):
    """Verify that calling the method without fields raises ValueError."""

    doc_id = insert_test_document(store)

    with pytest.raises(ValueError, match="No metadata fields"):
        store.update_document_metadata(doc_id)


def test_update_document_metadata_rejects_unknown_document(store):
    """Verify that updating a nonexistent document raises ValueError."""

    with pytest.raises(ValueError, match="Document not found"):
        store.update_document_metadata(
            "no-such-doc",
            title="New Title",
        )

def test_search_documents_by_metadata_matches_title(store):
    """Verify that metadata search matches document titles."""

    matching_doc = Document.for_pdf(
        pdf_path=Path("/tmp/bounded-confidence.pdf"),
        pdf_sha256="sha-title-match",
        metadata=ExtractedMetadata(
            title="Bounded Confidence in Social Influence",
            authors="Alice Example",
            year=2001,
            doi="10.1234/title-match",
        ),
    )
    other_doc = Document.for_pdf(
        pdf_path=Path("/tmp/weak-ties.pdf"),
        pdf_sha256="sha-title-other",
        metadata=ExtractedMetadata(
            title="The Strength of Weak Ties",
            authors="Mark Granovetter",
            year=1973,
            doi="10.1234/weak-ties",
        ),
    )

    store.insert_document(matching_doc)
    store.insert_document(other_doc)

    results = store.search_documents_by_metadata("bounded confidence")

    assert results == [matching_doc]


def test_search_documents_by_metadata_matches_authors_year_doi_and_path(store):
    """Verify that metadata search checks authors, year, DOI, and PDF path."""

    doc = Document.for_pdf(
        pdf_path=Path("/tmp/granovetter1973.pdf"),
        pdf_sha256="sha-metadata-match",
        metadata=ExtractedMetadata(
            title="The Strength of Weak Ties",
            authors="Mark Granovetter",
            year=1973,
            doi="10.1086/225469",
        ),
    )

    store.insert_document(doc)

    assert store.search_documents_by_metadata("Granovetter") == [doc]
    assert store.search_documents_by_metadata("1973") == [doc]
    assert store.search_documents_by_metadata("10.1086") == [doc]
    assert store.search_documents_by_metadata("granovetter1973") == [doc]


def test_search_documents_by_metadata_returns_empty_list_for_blank_search(store):
    """Verify that blank metadata searches return no results."""

    doc = make_document()
    store.insert_document(doc)

    assert store.search_documents_by_metadata("") == []
    assert store.search_documents_by_metadata("   ") == []


def test_search_chunks_text_returns_matching_chunk_hits(store):
    """Verify that text search returns chunk-level hits with document metadata."""

    doc = Document.for_pdf(
        pdf_path=Path("/tmp/social-influence.pdf"),
        pdf_sha256="sha-content-match",
        metadata=ExtractedMetadata(
            title="Social Influence and Diffusion",
            authors="Alice Example",
            year=2005,
            doi="10.1234/social-influence",
        ),
    )
    store.insert_document(doc)

    matching_chunk = make_chunk(
        doc.doc_id,
        chunk_index=0,
        text="This chunk discusses bounded confidence models.",
    )
    other_chunk = make_chunk(
        doc.doc_id,
        chunk_index=1,
        text="This chunk discusses something else.",
    )

    store.insert_chunks([matching_chunk, other_chunk])

    results = store.search_chunks_by_fulltext("bounded confidence")

    assert len(results) == 1

    hit = results[0]

    assert hit.chunk_id == matching_chunk.chunk_id
    assert hit.distance is None
    assert hit.text == matching_chunk.text
    assert hit.path == doc.pdf_path
    assert hit.title == "Social Influence and Diffusion"
    assert hit.page_start == matching_chunk.page_start
    assert hit.page_end == matching_chunk.page_end


def test_search_chunks_text_returns_empty_list_for_blank_search(store):
    """Verify that blank text searches return no chunk hits."""

    doc = make_document()
    store.insert_document(doc)

    chunk = make_chunk(
        doc.doc_id,
        chunk_index=0,
        text="This chunk discusses bounded confidence models.",
    )
    store.insert_chunks([chunk])

    assert store.search_chunks_by_fulltext("") == []
    assert store.search_chunks_by_fulltext("   ") == []


def test_search_chunks_text_treats_like_wildcards_as_literal_text(store):
    """Verify that %, _, and backslash are not treated as LIKE wildcards."""

    doc = make_document()
    store.insert_document(doc)

    wildcard_chunk = make_chunk(
        doc.doc_id,
        chunk_index=0,
        text="This chunk literally contains 100% coverage.",
    )
    ordinary_chunk = make_chunk(
        doc.doc_id,
        chunk_index=1,
        text="This chunk contains 1000 coverage but no percent sign.",
    )

    store.insert_chunks([wildcard_chunk, ordinary_chunk])

    results = store.search_chunks_by_fulltext("100%")

    assert len(results) == 1
    assert results[0].chunk_id == wildcard_chunk.chunk_id
