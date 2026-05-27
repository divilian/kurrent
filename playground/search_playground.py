"""Manual playground for SQLite metadata and chunk text search.

Run from the project root with:

    python playground/search_playground.py /path/to/pdf/root

Or from IPython with:

    run playground/search_playground.py /path/to/pdf/root

This playground stores temporary kurrent state under:

    /tmp/kurrent-search-playground/kurrent.db

That means it does not write to your real kurrent state. The playground starts
fresh each run and deletes its temporary state on normal exit.
"""

from __future__ import annotations

from pathlib import Path
import os
import re
import sys

from kurrent.ingester import ingest_pdfs_recursively
from kurrent.schema import ChunkHit, Document
from kurrent.sectioner import is_reference_section_chunk
from kurrent.state_store import StateStore


DEFAULT_ROOT_DIR = Path("/home/stephen/papers")
PLAYGROUND_DIR = Path("/tmp/kurrent-search-playground")
QUIT_COMMANDS = {"q", "done", "quit", "exit"}
USE_LLM_SECTIONING = False


def existing_sqlite_paths(db_path: Path) -> list[Path]:
    """Return existing SQLite database and sidecar paths."""

    candidates = [
        db_path,
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
    ]

    return [path for path in candidates if path.exists()]


def prepare_fresh_playground_database(db_path: Path) -> None:
    """Delete an existing playground database after confirmation."""

    existing_paths = existing_sqlite_paths(db_path)

    if not existing_paths:
        return

    print()
    print("Existing playground database found.")
    print("This playground is intended to start with fresh state each run.")
    print()
    print("Files to delete:")

    for path in existing_paths:
        print(f"  {path}")

    print()

    try:
        response = input("Delete existing playground database? [Y/n] ")
    except EOFError:
        raise SystemExit(
            "Existing playground database was not deleted; aborting."
        )

    response = response.strip().lower()

    if response not in {"", "y", "yes"}:
        raise SystemExit("Cancelled; existing playground database left in place.")

    for path in existing_paths:
        path.unlink()

    print("Deleted existing playground database.")


def cleanup_playground_database(db_path: Path) -> None:
    """Delete playground database files on normal program exit."""

    existing_paths = existing_sqlite_paths(db_path)

    for path in existing_paths:
        path.unlink()

    if existing_paths:
        print()
        print("Deleted playground database.")


def print_help() -> None:
    """Print search playground help."""

    print()
    print("Search playground")
    print("-----------------")
    print()
    print("Commands:")
    print("  m <text>          search document metadata")
    print("  metadata <text>   search document metadata")
    print("  c <text>          search chunk text")
    print("  content <text>    search chunk text")
    print("  <number>          show details for a numbered result")
    print("  help              show this help")
    print("  done              quit")
    print()
    print("Metadata search checks title, authors, year, DOI, and PDF path.")
    print("Content search checks stored chunk text using SQLite LIKE.")
    print("Reference-section chunk hits are visibly marked.")
    print()


def print_document_list(documents: list[Document]) -> None:
    """Print a numbered list of document results."""

    if not documents:
        print("No matching documents.")
        return

    for i, document in enumerate(documents, start=1):
        title = document.title or "(untitled)"
        year = document.year if document.year is not None else "n.d."
        authors = document.authors or "unknown author"

        print(f"{i}. {title} ({year}) — {authors}")


def print_document_detail(document: Document, index: int) -> None:
    """Print one document search result in detail."""

    print()
    print(f"Document {index}")
    print("-" * 10)
    print(f"kurrent ID: {document.doc_id}")
    print(f"title:      {document.title}")
    print(f"authors:    {document.authors}")
    print(f"year:       {document.year}")
    print(f"doi:        {document.doi}")
    print(f"path:       {document.pdf_path}")
    print()


ANSI_BOLD = "\033[1m"
ANSI_RESET = "\033[0m"


def ansi_enabled() -> bool:
    """Return whether ANSI formatting should be used for terminal output."""

    if os.environ.get("NO_COLOR") is not None:
        return False

    if os.environ.get("TERM") == "dumb":
        return False

    return sys.stdout.isatty()


def bold_matches(text: str, search_text: str | None) -> str:
    """Return text with literal case-insensitive matches bolded."""

    if not ansi_enabled():
        return text

    if search_text is None:
        return text

    search_text = search_text.strip()

    if not search_text:
        return text

    pattern = re.compile(re.escape(search_text), flags=re.IGNORECASE)

    return pattern.sub(
        lambda match: f"{ANSI_BOLD}{match.group(0)}{ANSI_RESET}",
        text,
    )


def collapse_whitespace(text: str) -> str:
    """Normalize text to a single display-friendly line."""

    return " ".join(text.split())


def matched_context_window(
    text: str,
    search_text: str | None,
    width: int = 2000,
) -> str:
    """Return a display window centered around the first search match."""

    text = collapse_whitespace(text)

    if len(text) <= width:
        return text

    if search_text is None:
        return text[:width].rstrip() + " [...]"

    search_text = search_text.strip()

    if not search_text:
        return text[:width].rstrip() + " [...]"

    match = re.search(re.escape(search_text), text, flags=re.IGNORECASE)

    if match is None:
        return text[:width].rstrip() + " [...]"

    match_center = (match.start() + match.end()) // 2
    start = max(0, match_center - width // 2)
    end = min(len(text), start + width)

    start = max(0, end - width)

    window = text[start:end].strip()

    if start > 0:
        window = "[...] " + window

    if end < len(text):
        window = window + " [...]"

    return window


def wait_for_enter() -> None:
    """Pause until the user presses Enter."""

    try:
        input("Press Enter to return to the hit list...")
    except EOFError:
        print()


def section_label(hit: ChunkHit) -> str | None:
    """Return a compact section label for a chunk hit, if available."""

    pieces = []

    if hit.section_number is not None:
        pieces.append(str(hit.section_number))

    if hit.section_title is not None:
        pieces.append(hit.section_title)

    if not pieces:
        return None

    return " ".join(pieces)


def reference_marker(hit: ChunkHit) -> str:
    """Return a visible marker for reference-section hits."""

    if is_reference_section_chunk(hit):
        return " [REFERENCE SECTION]"

    return ""


def print_chunk_hit_list(
    hits: list[ChunkHit],
    search_text: str | None = None,
) -> None:
    """Print a numbered list of chunk text hits."""

    if not hits:
        print("No matching chunks.")
        return

    for i, hit in enumerate(hits, start=1):
        source_name = hit.path.name if hit.path is not None else "(unknown PDF)"
        title = hit.title or source_name

        pages = ""
        if hit.page_start is not None or hit.page_end is not None:
            pages = f", pp. {hit.page_start}–{hit.page_end}"

        section = section_label(hit)

        preview = matched_context_window(
            hit.text,
            search_text,
            width=140,
        )
        preview = bold_matches(preview, search_text)

        print(f"{i}. {title}{reference_marker(hit)}{pages}")

        if section is not None:
            print(f"   section: {section}")

        print(f"   {source_name}")
        print(f"   {preview}")

        if i != len(hits):
            print()


def print_chunk_hit_detail(
    hit: ChunkHit,
    index: int,
    search_text: str | None = None,
    *,
    preview_chars: int = 2000,
) -> None:
    """Print one chunk text hit in detail."""

    print()
    print(f"Chunk hit {index}{reference_marker(hit)}")
    print("-" * 11)
    print(f"chunk_id: {hit.chunk_id}")

    if hit.title:
        print(f"title:    {hit.title}")

    if hit.path:
        print(f"path:     {hit.path}")

    if hit.page_start is not None or hit.page_end is not None:
        print(f"pages:    {hit.page_start}–{hit.page_end}")

    section = section_label(hit)

    if section is not None:
        print(f"section:  {section}")

    text = matched_context_window(
        hit.text,
        search_text,
        width=preview_chars,
    )

    print()
    print(bold_matches(text, search_text))
    print()


def search_loop(store: StateStore, limit: int = 10) -> None:
    """Prompt repeatedly for metadata or content searches."""

    print_help()

    last_result_type: str | None = None
    document_results: list[Document] = []
    chunk_results: list[ChunkHit] = []
    last_search_text: str | None = None

    def print_current_results() -> None:
        if last_search_text is not None:
            print()
            print(f"Search: {last_search_text!r}")

        if last_result_type == "metadata":
            print(f"Documents: {len(document_results)}")
            print()
            print_document_list(document_results)
        elif last_result_type == "content":
            print(f"Chunks: {len(chunk_results)}")
            print()
            print_chunk_hit_list(
                chunk_results,
                search_text=last_search_text,
            )

    while True:
        print()

        try:
            user_input = input("search> ").strip()
        except EOFError:
            print()
            return

        if user_input in QUIT_COMMANDS:
            return

        if not user_input:
            continue

        if user_input == "help":
            print_help()
            continue

        if (
            last_result_type == "metadata"
            and document_results
            and user_input.isdigit()
        ):
            index = int(user_input)

            if not 1 <= index <= len(document_results):
                print(f"Please enter a number from 1 to {len(document_results)}.")
                continue

            print_document_detail(document_results[index - 1], index)
            print_current_results()
            continue

        if (
            last_result_type == "content"
            and chunk_results
            and user_input.isdigit()
        ):
            index = int(user_input)

            if not 1 <= index <= len(chunk_results):
                print(f"Please enter a number from 1 to {len(chunk_results)}.")
                continue

            print_chunk_hit_detail(
                chunk_results[index - 1],
                index,
                search_text=last_search_text,
            )
            wait_for_enter()
            print_current_results()
            continue

        command, _, search_text = user_input.partition(" ")

        if command in {"m", "metadata"}:
            search_text = search_text.strip()

            if not search_text:
                print("Please provide metadata search text.")
                continue

            last_result_type = "metadata"
            last_search_text = search_text
            document_results = store.search_documents_by_metadata(
                search_text,
                limit=limit,
            )
            chunk_results = []

            print_current_results()
            continue

        if command in {"c", "content"}:
            search_text = search_text.strip()

            if not search_text:
                print("Please provide content search text.")
                continue

            last_result_type = "content"
            last_search_text = search_text
            chunk_results = store.search_chunks_by_fulltext(
                search_text,
                limit=limit,
            )
            document_results = []

            print_current_results()
            continue

        print(
            "Please enter m <text>, metadata <text>, c <text>, "
            "content <text>, help, or done."
        )


if __name__ == "__main__":

    if len(sys.argv) > 1:
        root_dir = Path(sys.argv[1])
    else:
        root_dir = DEFAULT_ROOT_DIR

    PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)

    db_path = PLAYGROUND_DIR / "kurrent.db"
    prepare_fresh_playground_database(db_path)

    store = StateStore(db_path)

    try:
        print(f"Ingesting PDFs under: {root_dir}")
        print(f"Database path:        {db_path}")
        print()

        print(f"Ingesting PDFs from {root_dir}...")
        print("Sectioning mode:      rules-based (LLM disabled)")
        doc_ids = ingest_pdfs_recursively(
            root_dir=root_dir,
            store=store,
            use_llm_sectioning=USE_LLM_SECTIONING,
        )

        print()
        print(f"Documents ingested: {len(doc_ids)}")

        search_loop(store, limit=10)
    finally:
        store.close()
        cleanup_playground_database(db_path)
