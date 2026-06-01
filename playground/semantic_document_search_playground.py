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
from kurrent.schema import DocumentHit
from kurrent.searcher import Searcher
from kurrent.semantic_highlighter import semantically_highlighted_excerpt
from kurrent.state_store import StateStore
from playground.common import (
    DEFAULT_ROOT_DIR,
    QUIT_COMMANDS,
    cleanup_playground_state,
    playground_dir,
    prepare_fresh_playground_state,
)

PLAYGROUND_DIR = playground_dir("semantic-document-search")
USE_LLM_SECTIONING = False


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


def print_document_hit_list(hits: list[DocumentHit]) -> None:
    """Print a numbered list of document search hits."""

    if not hits:
        print("No hits.")
        return

    for i, hit in enumerate(hits, start=1):
        score = f"{hit.score:.4f}" if hit.score is not None else "n/a"

        print(f"{i}. {hit.path.name}  [score={score}]")


def print_document_hit_detail(
    hit: DocumentHit,
    index: int,
    searcher: Searcher,
    search_text: str,
    preview_chars: int = 2000,
) -> None:
    """Print the selected document hit and its best matching chunk."""

    if searcher.embedder is None:
        raise ValueError("Semantic document search playground requires an Embedder.")

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

    print(
        semantically_highlighted_excerpt(
            chunk.text,
            search_text,
            searcher.embedder,
            max_chars=preview_chars,
        )
    )
    print()


def semantic_document_search_loop(
    searcher: Searcher,
    max_documents: int = 10,
    max_distance: float | None = None,
) -> None:
    """Prompt repeatedly for semantic document searches."""

    if searcher.embedder is None:
        raise ValueError("Semantic document search playground requires an Embedder.")

    print()
    print("Semantic document search playground")
    print("Type a search expression and press Enter.")
    print(f"Type {', '.join(QUIT_COMMANDS)} to leave.")
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
            print("You can enter a doc number, or your next search, or q.")

        try:
            user_input = input("sem-doc-search> ").strip()
        except EOFError:
            print()
            break

        if user_input in QUIT_COMMANDS:
            break

        if not user_input:
            continue

        if hits and user_input.isdigit():
            index = int(user_input)

            if not 1 <= index <= len(hits):
                print(f"Please enter a number from 1 to {len(hits)}.")
                continue

            if last_search_text is None:
                print("No active search query is available for highlighting.")
                continue

            print_document_hit_detail(
                hits[index - 1],
                index,
                searcher,
                search_text=last_search_text,
            )
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

        semantic_document_search_loop(
            searcher,
            max_documents=10,
            max_distance=None,
        )
    finally:
        cleanup_playground_state(db_path, chroma_path)
