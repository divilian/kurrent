"""Manual playground for batch ingest + semantic search.

Run from the project root with:

    python playground/semantic_search_playground.py /path/to/pdf/root

Or from IPython with:

    run playground/semantic_search_playground.py /path/to/pdf/root
"""

from __future__ import annotations

from pathlib import Path
import sys

from kurrent.embedder import Embedder
from kurrent.ingester import ingest_pdfs_recursively
from kurrent.schema import DocumentHit
from kurrent.searcher import Searcher
from kurrent.sectioner import is_reference_section_chunk
from kurrent.semantic_highlighter import semantically_highlighted_excerpt
from kurrent.state_store import StateStore
from playground.common import (
    DEFAULT_ROOT_DIR,
    QUIT_COMMANDS,
    cleanup_playground_state,
    playground_dir,
    prepare_fresh_playground_state,
)


PLAYGROUND_DIR = playground_dir("semantic-search")
USE_LLM_SECTIONING = False


def print_help() -> None:
    """Print semantic search playground help."""

    print()
    print("Semantic search playground")
    print("Type a search expression and press Enter.")
    print("Type a hit number to inspect that chunk.")
    print("Type help to show this message again.")
    print(f"Type {', '.join(QUIT_COMMANDS)} to leave.")
    print()


def reference_marker(hit: DocumentHit) -> str:
    """Return a display marker for chunks that appear to be references."""

    if is_reference_section_chunk(hit.text):
        return " [reference section]"

    return ""


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


def print_hit_detail(
    hit,
    index: int,
    search_text: str,
    embedder: Embedder,
    preview_chars: int = 2000,
) -> None:
    """Print the selected chunk hit in detail."""

    print()
    print(f"Hit {index}{reference_marker(hit)}")
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
        semantically_highlighted_excerpt(
            hit.text,
            search_text,
            embedder,
            max_chars=preview_chars,
        )
    )
    print()


def semantic_search_loop(
    searcher: Searcher,
    n_results: int = 10,
    max_distance: float | None = None,
) -> None:
    """Prompt repeatedly for semantic searches and print top PDF names."""

    if searcher.embedder is None:
        raise ValueError("Semantic search playground requires an Embedder.")

    print_help()

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
            print("You can enter a chunk number, or your next search, or q.")

        try:
            user_input = input("sem-search> ").strip()
        except EOFError:
            print()
            break

        if user_input in QUIT_COMMANDS:
            break

        if not user_input:
            continue

        if user_input == "help":
            print_help()
            continue

        if hits and user_input.isdigit():
            index = int(user_input)

            if not 1 <= index <= len(hits):
                print(f"Please enter a number from 1 to {len(hits)}.")
                continue

            if last_search_text is None:
                print("No active search query is available for highlighting.")
                continue

            print_hit_detail(
                hits[index - 1],
                index,
                search_text=last_search_text,
                embedder=searcher.embedder,
            )
            print_current_hit_list()
            continue

        last_search_text = user_input
        hits = searcher.semantic_chunk_search(
            user_input,
            n_results=n_results,
            max_distance=max_distance,
            include_reference_sections=False,
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

        print(f"Ingesting and embedding PDFs from {root_dir}...")
        print("Sectioning mode:      rules-based (LLM disabled)")
        doc_ids = ingest_pdfs_recursively(
            root_dir=root_dir,
            store=store,
            embedder=embedder,
            use_llm_sectioning=USE_LLM_SECTIONING,
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
