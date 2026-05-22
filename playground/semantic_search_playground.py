"""Manual playground for batch ingest + semantic search.

Run from the project root with:

    python playground/semantic_search_playground.py /path/to/pdf/root

Or from IPython with:

    run playground/semantic_search_playground.py /path/to/pdf/root
"""

from __future__ import annotations

from pathlib import Path
import shutil
import sys
from textwrap import shorten

from kurrent.embedder import Embedder
from kurrent.ingester import ingest_pdfs_recursively
from kurrent.schema import DocumentHit
from kurrent.searcher import Searcher
from kurrent.state_store import StateStore


DEFAULT_ROOT_DIR = Path("/home/stephen/teaching/420")
PLAYGROUND_DIR = Path("/tmp/kurrent-semantic-search-playground")


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


def print_hit_list(hits) -> None:
    """Print only the numbered list of source PDF base names."""

    if not hits:
        print("No hits.")
        return

    for i, hit in enumerate(hits, start=1):
        if hit.path is not None:
            source_name = hit.path.name
        else:
            source_name = "(unknown PDF)"

        distance = (
            f"{hit.distance:.4f}"
            if hit.distance is not None
            else "n/a"
        )

        pages = ""
        if hit.page_start is not None or hit.page_end is not None:
            pages = f", pp. {hit.page_start}–{hit.page_end}"

        print(f"{i}. {source_name}  [distance={distance}{pages}]")


def print_hit_detail(hit, index: int, *, preview_chars: int = 2000) -> None:
    """Print the selected chunk hit in detail."""

    print()
    print(f"Hit {index}")
    print(f"chunk_id: {hit.chunk_id}")
    print(f"doc_id:   {hit.doc_id}")

    if hit.distance is not None:
        print(f"distance: {hit.distance:.4f}")

    if hit.title:
        print(f"title:    {hit.title}")

    if hit.path:
        print(f"path:     {hit.path}")

    if hit.page_start is not None or hit.page_end is not None:
        print(f"pages:    {hit.page_start}–{hit.page_end}")

    print()
    print(
        shorten(
            " ".join(hit.text.split()),
            width=preview_chars,
            placeholder=" [...]",
        )
    )
    print()


def semantic_search_loop(
    searcher: Searcher,
    n_results: int = 10,
    max_distance: float | None = None,
) -> None:
    """Prompt repeatedly for semantic searches and print top PDF names."""

    print()
    print("Semantic search playground")
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

        print_hit_list(hits)

    while True:
        if hits:
            print()
            print("You can enter a chunk number, or your next search.")

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

            print_hit_detail(hits[index - 1], index)
            print_current_hit_list()
            continue

        last_search_text = user_input
        hits = searcher.semantic_chunk_search(
            user_input,
            n_results=n_results,
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

        semantic_search_loop(
            searcher,
            n_results=10,
            max_distance=None,
        )
    finally:
        cleanup_playground_state(db_path, chroma_path)
