# Orchestrates initial PDF registration in kurrent:
# - normalizes and validates the filesystem path
# - verifies that files are PDFs
# - computes PDF content hash
# - registers (or looks up) the document in the kurrent state store
# - optionally discovers PDFs in a filesystem hierarchy

from pathlib import Path
from typing import Sequence

from kurrent.embedder import Embedder
from kurrent.file_utils import normalize_path, is_pdf, sha256_file
from kurrent.state_store import StateStore
from kurrent.chunker import chunk_document



def ingest_pdf(
    path: str | Path,
    store: StateStore,
    embedder: Embedder | None = None,
) -> str:
    """
    Ingests the PDF into kurrent, and returns the doc_id for it. This could be
    an already-existing doc_id if that PDF had been previously ingested (even
    under a different file path).

    If this is indeed a new PDF, chunk it and insert the chunks into kurrent.

    If a Chroma embedder is provided, also compute and store each chunk's
    embeddings in the vector store.

    Assumptions for the moment:
    - externally managed ("external" storage mode only)
    """
    path = normalize_path(path)

    if not is_pdf(path):
        raise ValueError(f"No such PDF file {path}")

    sha256 = sha256_file(path)

    doc = store.get_or_create_document(path, sha256)
    chunk_document(doc.doc_id, store)   # idempotent
    if embedder is not None:
        # This is possibly slow if this document has been previously ingested.
        # We're redoing the embedding work. Possible performance improvement.
        embedder.index_chunks(doc.doc_id, store)

    return doc.doc_id


def discover_pdfs(root_dir: str | Path) -> list[Path]:
    """Recursively discover PDF files under root_dir."""

    root = Path(root_dir).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root}")

    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    )


def ingest_pdfs_recursively(
    root_dir: str | Path,
    store: StateStore,
    embedder: Embedder | None = None,
    verbose: bool | None = True,
) -> dict[Path, str]:
    """Recursively ingest all PDF files under root_dir.

    Returns a mapping specifying which files became ingested as which doc_ids.

    If embedder is provided, each document is also indexed in Chroma.
    This function is intentionally fail-fast: any exception stops the batch.
    """

    doc_ids: dict[Path, str] = {}

    for pdf_path in discover_pdfs(root_dir):
        try:
            doc_id = ingest_pdf(pdf_path, store)

            if embedder is not None:
                embedder.index_chunks(doc_id, store)

            doc_ids[pdf_path] = doc_id
            print(f"Ingested {pdf_path.name} as {doc_id}")
        except Exception as e:
            print(f"Could not ingest {pdf_path.name}: {e}")

    return doc_ids


def print_batch_ingest_summary(results: dict[Path, str]) -> None:
    for path, doc_id in results.items():
        print(f"Created {doc_id} for {path.name}")

if __name__ == "__main__":

    # Smoke test / IPython playground.
    #
    # Run from IPython with:
    #
    #     run -m kurrent.ingester
    #
    # Then inspect:
    #
    #     results
    #     store
    #     embedder

    from pathlib import Path

    from kurrent.embedder import Embedder
    from kurrent.state_store import StateStore

    root_dir = Path("~/teaching/letters").expanduser().resolve()

    tmpdir = Path("/tmp/kurrent-batch-ingest")
    tmpdir.mkdir(parents=True, exist_ok=True)

    db_path = tmpdir / "kurrent.db"
    chroma_path = tmpdir / "chroma"

    store = StateStore(db_path)
    embedder = Embedder(chroma_path=chroma_path)

    results = ingest_pdfs_recursively(
        root_dir=root_dir,
        store=store,
        embedder=embedder,
    )

    print_batch_ingest_summary(results)
