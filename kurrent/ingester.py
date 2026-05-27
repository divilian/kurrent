# Orchestrates initial PDF registration in kurrent:
# - normalizes and validates the filesystem path
# - verifies that files are PDFs
# - computes PDF content hash
# - registers (or looks up) the document in the kurrent state store
# - chunks the document
# - optionally discovers PDFs in a filesystem hierarchy

from pathlib import Path
import time
from collections.abc import Callable
from typing import Sequence

from kurrent.embedder import Embedder
from kurrent.file_utils import (
    normalize_path,
    is_pdf,
    sha256_file,
    silence_mupdf_messages,
)
from kurrent.metadata_extractor import extract_metadata
from kurrent.state_store import StateStore
from kurrent.chunker import chunk_document


silence_mupdf_messages()

CROSSREF_REQUEST_INTERVAL_SECONDS = 1.0


def ingest_pdf(
    path: str | Path,
    store: StateStore,
    embedder: Embedder | None = None,
    doi_lookup: bool = False,
    crossref_mailto: str | None = None,
    reviewed_headings: Sequence[str] | None = None,
    use_llm_sectioning: bool = True,
    llm_progress_total_callback: Callable[[int], None] | None = None,
    llm_progress_callback: Callable[[int], None] | None = None,
) -> str:
    """Ingest a PDF into kurrent and return its kurrent ID.

    reviewed_headings=None means the ingest pipeline should detect headings
    automatically.

    reviewed_headings=list means the caller has supplied a reviewed/accepted
    heading list. An empty list is meaningful: it explicitly says to use no
    section headings.

    use_llm_sectioning controls automatic sectioning only when
    reviewed_headings is None.

    llm_progress_total_callback and llm_progress_callback are optional hooks
    used by interactive playgrounds to display Ollama progress.
    """

    path = normalize_path(path)

    if not is_pdf(path):
        raise ValueError(f"No such PDF file {path}")

    sha256 = sha256_file(path)
    metadata = extract_metadata(
        path,
        doi_lookup=doi_lookup,
        crossref_mailto=crossref_mailto,
    )

    doc = store.get_or_create_document(
        path,
        sha256,
        metadata=metadata,
    )

    chunk_document(
        doc.doc_id,
        store,
        reviewed_headings=reviewed_headings,
        use_llm_sectioning=use_llm_sectioning,
        llm_progress_total_callback=llm_progress_total_callback,
        llm_progress_callback=llm_progress_callback,
    )

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
    no_more_than: int | None = None,
    verbose: bool | None = True,
    doi_lookup: bool = False,
    crossref_mailto: str | None = None,
    use_llm_sectioning: bool = True,
) -> dict[Path, str]:
    """Recursively ingest all PDF files under root_dir.

    Returns a mapping specifying which files became ingested as which kurrent
    IDs.

    If embedder is provided, each document is also indexed in Chroma.
    This function is intentionally fail-soft: exceptions are reported and the
    batch continues.

    use_llm_sectioning controls the automatic sectioning path used for each
    PDF. Search-oriented playgrounds can pass False to avoid slow Ollama calls
    when LLM-quality sections are not the point of the playground.
    """

    doc_ids: dict[Path, str] = {}

    discovered_pdfs = discover_pdfs(root_dir)

    if no_more_than is not None:
        discovered_pdfs = discovered_pdfs[:no_more_than]

    if doi_lookup and verbose:
        print(
            "DOI lookup is enabled; additional time here is Crossref "
            "metadata lookup, not slow kurrent ingestion."
        )

    for i, pdf_path in enumerate(discovered_pdfs, start=1):
        try:
            doc_id = ingest_pdf(
                pdf_path,
                store,
                embedder=embedder,
                doi_lookup=doi_lookup,
                crossref_mailto=crossref_mailto,
                use_llm_sectioning=use_llm_sectioning,
            )

            doc_ids[pdf_path] = doc_id
            print(f"Ingested {pdf_path.name} as {doc_id}")
        except Exception as e:
            print(f"Could not ingest {pdf_path.name}: {e}")

        if doi_lookup and i < len(discovered_pdfs):
            time.sleep(CROSSREF_REQUEST_INTERVAL_SECONDS)

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
