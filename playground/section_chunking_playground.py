"""Manual playground for inspecting detected sections and sectioned chunks.

Run from the project root with:

    python playground/section_chunking_playground.py /path/to/pdf/or/root

Or from IPython with:

    run playground/section_chunking_playground.py /path/to/pdf/or/root

This playground is intentionally light: it lets you choose a PDF, inspect
heading candidates, optionally remove bogus headings, ingest the PDF into
temporary kurrent state, and then display the resulting section-aware chunks.

The point is to test the wiring from reviewed headings to SectionSpan objects
to section-tagged Chunk objects, without involving Chroma or semantic search.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import sys
import textwrap

from kurrent.chunker import chunker_version
from kurrent.file_utils import is_pdf
from kurrent.ingester import ingest_pdf
from kurrent.sectioner import (
    detect_heading_candidates,
    is_reference_section_chunk,
)
from kurrent.state_store import StateStore


DEFAULT_ROOT_DIR = Path("/home/stephen/papers")
PLAYGROUND_DIR = Path("/tmp/kurrent-section-chunking-playground")
QUIT_COMMANDS = {":q", ":quit", "done", "quit", "exit"}


def existing_playground_paths(db_path: Path) -> list[Path]:
    """Return existing playground database paths."""

    candidates = [
        db_path,
        db_path.with_name(f"{db_path.name}-wal"),
        db_path.with_name(f"{db_path.name}-shm"),
    ]

    return [path for path in candidates if path.exists()]


def prepare_fresh_playground_state(db_path: Path) -> None:
    """Delete existing playground state after confirmation."""

    existing_paths = existing_playground_paths(db_path)

    if not existing_paths:
        return

    print()
    print("Existing playground state found.")
    print("This playground is intended to start with fresh state each run.")
    print()
    print("Files to delete:")

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
        path.unlink()

    print("Deleted existing playground state.")


def cleanup_playground_state(db_path: Path) -> None:
    """Delete playground state on normal program exit."""

    existing_paths = existing_playground_paths(db_path)

    for path in existing_paths:
        path.unlink()

    if existing_paths:
        print()
        print("Deleted playground state.")


def discover_pdfs(path: str | Path) -> list[Path]:
    """Return one PDF path or all PDFs recursively under a directory."""

    path = Path(path).expanduser().resolve()

    if path.is_file():
        if not is_pdf(path):
            raise ValueError(f"Not a PDF file: {path}")

        return [path]

    if not path.is_dir():
        raise FileNotFoundError(f"No such file or directory: {path}")

    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() == ".pdf"
    )


def print_pdf_list(pdf_paths: Sequence[Path]) -> None:
    """Print a numbered list of PDF basenames."""

    if not pdf_paths:
        print("No PDFs found.")
        return

    for i, pdf_path in enumerate(pdf_paths, start=1):
        print(f"{i}. {pdf_path.name}")


def print_heading_candidates(headings: Sequence[str]) -> None:
    """Print numbered heading candidates."""

    print()
    print("Detected heading candidates")
    print("---------------------------")

    if not headings:
        print("No plausible section headings found.")
        return

    for i, heading in enumerate(headings, start=1):
        print(f"{i}. {heading}")


def parse_number_list(text: str, maximum: int) -> set[int]:
    """Parse comma-separated 1-based numbers into a set."""

    selected: set[int] = set()

    for raw_part in text.split(","):
        part = raw_part.strip()

        if not part:
            continue

        try:
            number = int(part)
        except ValueError as exc:
            raise ValueError(f"Not a number: {part!r}") from exc

        if not 1 <= number <= maximum:
            raise ValueError(f"Number out of range: {number}")

        selected.add(number)

    return selected


def review_heading_candidates(headings: list[str]) -> list[str]:
    """Let the user remove bogus heading candidates."""

    print_heading_candidates(headings)

    if not headings:
        return []

    print()
    print("Enter comma-separated numbers to remove bogus headings.")
    print("Press Enter to keep all headings.")

    while True:
        raw = input("remove headings> ").strip()

        if raw in QUIT_COMMANDS:
            raise KeyboardInterrupt("Review cancelled by user.")

        if not raw:
            return headings

        try:
            to_remove = parse_number_list(raw, len(headings))
        except ValueError as exc:
            print(exc)
            continue

        accepted = [
            heading
            for i, heading in enumerate(headings, start=1)
            if i not in to_remove
        ]

        print()
        print("Accepted heading candidates:")
        if accepted:
            for heading in accepted:
                print(f"  - {heading}")
        else:
            print("  (none)")

        return accepted


def section_label(chunk) -> str:
    """Return a readable section label for a chunk."""

    pieces = []

    if chunk.section_number is not None:
        pieces.append(str(chunk.section_number))

    if chunk.section_title is not None:
        pieces.append(chunk.section_title)

    if pieces:
        return " ".join(pieces)

    if chunk.section_index is not None:
        return f"section index {chunk.section_index}"

    return "unsectioned"


def reference_marker(chunk) -> str:
    """Return a visible marker for reference-section chunks."""

    if is_reference_section_chunk(chunk):
        return " [REFERENCE SECTION]"

    return ""


def print_chunks_by_section(chunks) -> None:
    """Print stored chunks grouped by section metadata."""

    if not chunks:
        print("No chunks found.")
        return

    last_key = object()

    for chunk in chunks:
        key = (
            chunk.section_index,
            chunk.section_number,
            chunk.section_title,
        )

        if key != last_key:
            label = section_label(chunk)
            marker = reference_marker(chunk)
            print()
            print(f"Section: {label}{marker}")
            print("-" * (9 + len(label) + len(marker)))
            last_key = key

        preview = " ".join(chunk.text.split())

        if len(preview) > 300:
            preview = preview[:300] + " [...]"

        print(
            f"  chunk {chunk.chunk_index}{reference_marker(chunk)} "
            f"(pp. {chunk.page_start}–{chunk.page_end})"
        )
        print(
            textwrap.fill(
                preview,
                width=79,
                initial_indent="    ",
                subsequent_indent="    ",
            )
        )


def section_chunking_loop(pdf_paths: Sequence[Path], store: StateStore) -> None:
    """Prompt for PDFs, review headings, ingest, and show sectioned chunks."""

    print()
    print("Section chunking playground")
    print("Choose a PDF number to inspect detected headings and stored chunks.")
    print("Type list, ls, or pdfs to redisplay the numbered PDF list.")
    print("Type :q, :quit, done, quit, or exit to leave.")
    print()
    print_pdf_list(pdf_paths)

    while True:
        print()

        try:
            user_input = input("kurrent> ").strip()
        except EOFError:
            print()
            return

        if user_input in QUIT_COMMANDS:
            return

        if not user_input:
            continue

        if user_input.lower() in {"list", "ls", "pdfs"}:
            print()
            print_pdf_list(pdf_paths)
            continue

        if not user_input.isdigit():
            print("Please enter a PDF number, list, ls, pdfs, or done.")
            continue

        index = int(user_input)

        if not 1 <= index <= len(pdf_paths):
            print(f"Please enter a number from 1 to {len(pdf_paths)}.")
            continue

        pdf_path = pdf_paths[index - 1]

        print()
        print(f"PDF: {pdf_path}")
        headings = detect_heading_candidates(pdf_path)
        reviewed_headings = review_heading_candidates(headings)

        print()
        print("Ingesting PDF and creating section-aware chunks...")
        doc_id = ingest_pdf(
            pdf_path,
            store,
            reviewed_headings=reviewed_headings,
        )

        chunks = store.get_chunks_for_document(
            doc_id=doc_id,
            chunker_version=chunker_version(),
        )

        print_chunks_by_section(chunks)


if __name__ == "__main__":

    if len(sys.argv) > 1:
        root_or_pdf = Path(sys.argv[1])
    else:
        root_or_pdf = DEFAULT_ROOT_DIR

    PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)

    db_path = PLAYGROUND_DIR / "kurrent.db"
    prepare_fresh_playground_state(db_path)

    store = StateStore(db_path)

    try:
        pdf_paths = discover_pdfs(root_or_pdf)

        print(f"PDF source:     {root_or_pdf}")
        print(f"Database path:  {db_path}")
        print(f"PDFs found:     {len(pdf_paths)}")

        section_chunking_loop(pdf_paths, store)
    finally:
        store.close()
        cleanup_playground_state(db_path)
