"""Manual playground for batch ingest + semantic document search.

Run from the project root with:

    python playground/semantic_document_search_playground.py /path/to/pdf/root

Or from IPython with:

    run playground/semantic_document_search_playground.py /path/to/pdf/root
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys

from kurrent.embedder import Embedder
from kurrent.ingester import ingest_pdfs_recursively
from kurrent.schema import ChunkHit
from kurrent.searcher import Searcher
from kurrent.state_store import StateStore


DEFAULT_ROOT_DIR = Path("/home/stephen/teaching/420")
PLAYGROUND_DIR = Path("/tmp/kurrent-semantic-document-search-playground")


def existing_playground_paths(db_path: Path, chroma_path: Path) -> list[Path]:
    """Return existing playground database and Chroma paths."""

    candidates = [
        db_path,
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
        chroma_path,
    ]

    return [path for path in candidates if path.exists()]


def prepare_fresh_playground_state(db_path: Path, chroma_path: Path) -> None:
    """Delete existing playground state after confirmation."""

    existing_paths = existing_playground_paths(db_path, chroma_path)

    if not existing_paths:
        return

    print()
    print("Existing playground state found.")
    print("This playground is intended to start with fresh state each run.")
    print()
    print("Files/directories to delete:")

    for path in existing_paths:
        print(f"  {path}")

    print()

    try:
        response = input("Delete existing playground state? [Y/n] ")
    except EOFError:
        raise SystemExit(
            "Existing playground state was not deleted; aborting."
        )

    response = response.strip().lower()

    if response not in {"", "y", "yes"}:
        raise SystemExit("Cancelled; existing playground state left in place.")

    for path in existing_paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    print("Deleted existing playground state.")


def cleanup_playground_state(db_path: Path, chroma_path: Path) -> None:
    """Delete playground state on normal program exit."""

    existing_paths = existing_playground_paths(db_path, chroma_path)

    for path in existing_paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()

    if existing_paths:
        print()
        print("Deleted playground state.")


def format_chunk_section(chunk) -> str | None:
    """Return a compact section label for a chunk, if available."""

    pieces = []

    if chunk.section_number is not None:
        pieces.append(str(chunk.section_number))

    if chunk.section_title is not None:
        pieces.append(chunk.section_title)

    if not pieces:
        return None

    return " ".join(pieces)


def print_document_hit_list(hits: list[ChunkHit]) -> None:
    """Print a numbered list of document search hits."""

    if not hits:
        print("No hits.")
        return

    for i, hit in enumerate(hits, start=1):
        score = f"{hit.score:.4f}" if hit.score is not None else "n/a"

        print(f"{i}. {hit.path.name}  [score={score}]")


def print_document_hit_detail(
    hit: ChunkHit,
    index: int,
    searcher: Searcher,
    preview_chars: int = 2000,
) -> None:
    """Print the selected document hit and its best matching chunk."""

    print()
    print(f"Document hit {index}")
    print(f"doc_id: {hit.doc_id}")
    print(f"path:   {hit.path.name}")

    if hit.score is not None:
        print(f"score:  {hit.score:.4f}")

    if hit.title:
        print(f"title:  {hit.title}")

    if hit.authors:
        print(f"authors: {hit.authors}")

    if hit.year:
        print(f"year:   {hit.year}")

    if hit.best_chunk_id is None:
        print()
        print("No representative chunk recorded for this document hit.")
        print()
        return

    chunk = searcher.state_store.get_chunk(hit.best_chunk_id)

    if chunk is None:
        print()
        print(f"Representative chunk not found: {hit.best_chunk_id}")
        print()
        return

    print()
    print("Best matching chunk")
    print(f"chunk_id: {chunk.chunk_id}")

    section = format_chunk_section(chunk)

    if section is not None:
        print(f"section:  {section}")

    print(f"pages:    {chunk.page_start}–{chunk.page_end}")
    print()

    text = " ".join(chunk.text.split())

    if len(text) > preview_chars:
        text = text[:preview_chars] + " [...]"

    print(text)
    print()


def semantic_document_search_loop(
    searcher: Searcher,
    max_documents: int = 10,
    max_distance: float | None = None,
) -> None:
    """Prompt repeatedly for semantic document searches."""

    print()
    print("Semantic document search playground")
    print("Type a search expression and press Enter.")
    print("Type :q, :quit, quit, or exit to leave.")
    print()

    hits = []
    last_search_text: str | None = None

    def print_current_hit_list() -> None:
        if last_search_text is not None:
            print()
            print(f"Search: {last_search_text!r}")
            print(f"Hits:   {len(hits)}")
            print()

        print_document_hit_list(hits)

    while True:
        if hits:
            print()
            print("You can enter a document number, or your next search.")

        try:
            user_input = input("kurrent> ").strip()
        except EOFError:
            print()
            break

        if user_input in {":q", ":quit", "done", "quit", "exit"}:
            break

        if not user_input:
            continue

        if hits and user_input.isdigit():
            index = int(user_input)

            if not 1 <= index <= len(hits):
                print(f"Please enter a number from 1 to {len(hits)}.")
                continue

            print_document_hit_detail(hits[index - 1], index, searcher)
            print_current_hit_list()
            continue

        last_search_text = user_input
        hits = searcher.semantic_document_search(
            user_input,
            max_documents=max_documents,
            max_distance=max_distance,
        )

        print_current_hit_list()


if __name__ == "__main__":

    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = DEFAULT_ROOT_DIR

    PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)

    db_path = PLAYGROUND_DIR / "kurrent.db"
    chroma_path = PLAYGROUND_DIR / "chroma"
    prepare_fresh_playground_state(db_path, chroma_path)

    store = StateStore(db_path)
    embedder = Embedder(chroma_path=chroma_path)

    try:
        print(f"Ingesting PDFs under: {root_dir}")
        print(f"Database path:        {db_path}")
        print(f"Chroma path:          {chroma_path}")
        print()

        doc_ids = ingest_pdfs_recursively(
            root_dir=root_dir,
            store=store,
            embedder=embedder,
        )

        searcher = Searcher(
            state_store=store,
            embedder=embedder,
        )

        print()
        print(f"Documents ingested/indexed: {len(doc_ids)}")

        semantic_document_search_loop(
            searcher,
            max_documents=10,
            max_distance=None,
        )
    finally:
        cleanup_playground_state(db_path, chroma_path)
