"""Manual playground for batch ingest + semantic document search.

Run from the project root with:

    python playground/semantic_document_search_playground.py /path/to/pdf/root

Or from IPython with:

    run playground/semantic_document_search_playground.py /path/to/pdf/root
"""

from __future__ import annotations

from pathlib import Path
import sys

from kurrent.embedder import Embedder
from kurrent.ingester import ingest_pdfs_recursively
from kurrent.searcher import Searcher
from kurrent.state_store import StateStore


DEFAULT_ROOT_DIR = Path("/home/stephen/teaching/420")
PLAYGROUND_DIR = Path("/tmp/kurrent-semantic-document-search-playground")


def print_document_hit_list(hits) -> None:
    """Print a numbered list of document search hits."""

    if not hits:
        print("No hits.")
        return

    for i, hit in enumerate(hits, start=1):
        score = f"{hit.score:.4f}" if hit.score is not None else "n/a"

        print(f"{i}. {hit.name}  [score={score}]")


def print_document_hit_detail(hit, index: int) -> None:
    """Print the selected document hit in detail."""

    print()
    print(f"Document hit {index}")
    print(f"doc_id: {hit.doc_id}")
    print(f"path:   {hit.path}")

    if hit.score is not None:
        print(f"score:  {hit.score:.4f}")

    if hit.title:
        print(f"title:  {hit.title}")

    if hit.authors:
        print(f"authors: {hit.authors}")

    if hit.year:
        print(f"year:   {hit.year}")

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

        if user_input in {":q", ":quit", "quit", "exit"}:
            break

        if not user_input:
            continue

        if hits and user_input.isdigit():
            index = int(user_input)

            if not 1 <= index <= len(hits):
                print(f"Please enter a number from 1 to {len(hits)}.")
                continue

            print_document_hit_detail(hits[index - 1], index)
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

    store = StateStore(db_path)
    embedder = Embedder(chroma_path=chroma_path)

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
