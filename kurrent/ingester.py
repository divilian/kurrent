# Orchestrates initial PDF registration in kurrent:
# - normalizes and validates the filesystem path
# - verifies that files are PDFs
# - computes PDF content hash
# - registers (or looks up) the document in the kurrent state store

from pathlib import Path

from kurrent.file_utils import normalize_path, is_pdf, sha256_file
from kurrent.state_store import StateStore
from kurrent.chunker import chunk_document


def ingest_pdf(path: str | Path, store: StateStore) -> str:
    """
    Ingests the PDF into kurrent, and returns the doc_id for it. This could be
    an already-existing doc_id if that PDF had been previously ingested (even
    under a different file path).

    If this is indeed a new PDF, chunk it and insert the chunks into kurrent.

    Assumptions for the moment:
    - externally managed ("external" storage mode only)
    """
    path = normalize_path(path)

    if not is_pdf(path):
        raise ValueError(f"No such PDF file {path}")

    sha256 = sha256_file(path)

    doc = store.get_or_create_document(path, sha256)
    chunk_document(doc.doc_id, store)   # idempotent

    return doc.doc_id


if __name__ == "__main__":

    # Smoke test.
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "kurrent.db"

        with StateStore(db_path) as store:
            doc_id = ingest_pdf(
                "/home/stephen/teaching/419/syllabus.pdf",
                store,
            )
            print(f"Should be the same as following: {doc_id}")
            doc2_id = ingest_pdf(
                "/home/stephen/teaching/419/syllabus.pdf",
                store,
            )
            print(f"Should be the same as preceding: {doc2_id}")
            doc3_id = ingest_pdf(
                "/home/stephen/teaching/350/syllabus.pdf",
                store,
            )
            print(f"Should be different: {doc3_id}")
