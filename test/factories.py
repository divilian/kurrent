from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from kurrent.chunker import chunker_version
from kurrent.schema import (
    Chunk,
    Document,
    ProximityAlertRecord,
    ConfirmedLink,
)

def make_chunk(
    doc_id: str,
    chunk_index: int = 0,
    text: str = "This is a test chunk.",
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
        "section_index": None,
        "section_number": None,
        "section_title": None,
    }
    values.update(overrides)

    return Chunk(**values)

def make_document(
    pdf_path: Path | None = None,
    **overrides,
) -> Document:
    doc_id = str(uuid4())
    fake_sha256 = f"fake-sha256-{doc_id}"

    values = {
        "doc_id": doc_id,
        "pdf_sha256": fake_sha256,
        "storage_mode": "external",
        "pdf_path": pdf_path or Path("/tmp/example.pdf"),
        "ingested_at": datetime.now(timezone.utc),
        "title": "Example Paper",
        "authors": "Ren, Kylo",
        "year": 2015,
        "doi": "10.1234/example",
    }
    values.update(overrides)

    return Document(**values)

def make_proximity_alert(
    chunk_a: Chunk,
    chunk_b: Chunk,
    **overrides,
) -> ProximityAlertRecord:
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
    return ProximityAlertRecord(**values)


def make_confirmed_link(pa_id: str, **overrides) -> ConfirmedLink:
    values = {
        "cl_id": str(uuid4()),
        "pa_id": pa_id,
        "created_at": datetime.now(timezone.utc),
        "relationship_type": "same_claim",
    }
    values.update(overrides)
    return ConfirmedLink(**values)


