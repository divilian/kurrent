"""Manual playground for PDF metadata extraction and ingest metadata.

Run from the project root with:

    python playground/metadata_playground.py /path/to/pdf/or/root

Or from IPython with:

    run playground/metadata_playground.py /path/to/pdf/or/root

Optionally set a Crossref polite-pool email address:

    export KURRENT_CROSSREF_MAILTO=you@example.edu

This playground stores its temporary kurrent state under:

    /tmp/kurrent-metadata-playground/kurrent.db

That means it does not write to your real kurrent state, but the playground
database may persist across runs until /tmp is cleaned or you delete it.
"""

from __future__ import annotations

from collections.abc import Sequence
import ast
import re
from pathlib import Path
import sys

from kurrent.file_utils import is_pdf
from kurrent.ingester import ingest_pdf
from kurrent.metadata_extractor import extract_metadata
from kurrent.schema import Document, ExtractedMetadata
from kurrent.state_store import StateStore


DEFAULT_ROOT_DIR = Path("/home/stephen/papers")
PLAYGROUND_DIR = Path("/tmp/kurrent-metadata-playground")
QUIT_COMMANDS = {"q", "done", "quit", "exit"}
METADATA_MODES = {
    "local":"local",
    "l":"local",
    "crossref":"crossref",
    "c":"crossref"
}


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


def normalize_filename_part(value: str) -> str:
    """Return a conservative lowercase filename-safe version of value."""

    return re.sub(r"[^a-z0-9]+", "", value.lower())


def authors_to_list(authors: object) -> list[str]:
    """Best-effort conversion of stored author data into a list of names."""

    if authors is None:
        return []

    if isinstance(authors, list):
        return [str(author).strip() for author in authors if str(author).strip()]

    if isinstance(authors, tuple):
        return [str(author).strip() for author in authors if str(author).strip()]

    author_text = str(authors).strip()

    if not author_text:
        return []

    try:
        parsed_authors = ast.literal_eval(author_text)
    except (SyntaxError, ValueError):
        parsed_authors = None

    if isinstance(parsed_authors, list):
        return [
            str(author).strip()
            for author in parsed_authors
            if str(author).strip()
        ]

    if isinstance(parsed_authors, tuple):
        return [
            str(author).strip()
            for author in parsed_authors
            if str(author).strip()
        ]

    if ";" in author_text:
        return [
            author.strip()
            for author in author_text.split(";")
            if author.strip()
        ]

    if " and " in author_text:
        return [
            author.strip()
            for author in author_text.split(" and ")
            if author.strip()
        ]

    # If the string looks like "Davis, James", keep it as one author.
    # Otherwise, comma-separated strings are probably multiple authors.
    comma_parts = [
        part.strip()
        for part in author_text.split(",")
        if part.strip()
    ]

    if len(comma_parts) == 2:
        first_part, second_part = comma_parts

        if (
            " " not in first_part
            and " " not in second_part
            and first_part.isalpha()
            and second_part.isalpha()
        ):
            return [author_text]

    if len(comma_parts) > 1:
        return comma_parts

    return [author_text]


def surname_from_author(author: str) -> str | None:
    """Best-effort extraction of a surname from one author string."""

    author = author.strip()

    if not author:
        return None

    author = re.sub(r"^\s*['\"\[]+", "", author)
    author = re.sub(r"['\"\]]+\s*$", "", author)
    author = re.sub(r"\s+", " ", author)

    if "," in author:
        surname = author.split(",", maxsplit=1)[0].strip()
    else:
        name_parts = author.split()
        surname = name_parts[-1].strip() if name_parts else ""

    surname = normalize_filename_part(surname)

    if not surname:
        return None

    return surname


def stephen_style_filename_from_parts(
    authors: object,
    year: object,
) -> str | None:
    """Return first-author-surname-plus-year filename, if possible."""

    author_list = authors_to_list(authors)

    if not author_list or year is None:
        return None

    surname = surname_from_author(author_list[0])

    if surname is None:
        return None

    clean_year = normalize_filename_part(str(year))

    if not clean_year:
        return None

    return f"{surname}{clean_year}.pdf"


def stephen_style_filename(document: Document) -> str | None:
    """Return first-author-surname-plus-year filename, if possible."""

    return stephen_style_filename_from_parts(
        authors=document.authors,
        year=document.year,
    )


def print_help(pdf_paths: Sequence[Path], metadata_mode: str) -> None:
    """Print playground help and the numbered PDF list."""

    print()
    print("Metadata playground")
    print("-------------------")
    print(f"Current mode: {metadata_mode}")
    print()
    print("Commands:")
    print("  <number>   ingest selected PDF using current metadata mode")
    print("  local      switch to local PDF metadata mode")
    print("  crossref   switch to Crossref-enhanced metadata mode")
    print("  help       show this help")
    print("  done       quit")
    print()
    print_pdf_list(pdf_paths)


def print_metadata_result(
    metadata: ExtractedMetadata,
    metadata_source: str,
) -> None:
    """Print the metadata produced by the selected extraction mode."""

    suggested_filename = stephen_style_filename_from_parts(
        authors=metadata.authors,
        year=metadata.year,
    )
    label = f"Metadata result ({metadata_source} metadata)"

    print()
    print(label)
    print("-" * len(label))
    print(f"title:              {metadata.title}")
    print(f"authors:            {metadata.authors}")
    print(f"year:               {metadata.year}")
    print(f"doi:                {metadata.doi}")
    print(f"suggested filename: {suggested_filename}")


def warn_crossref_fallback(reason: str) -> None:
    """Print a loud warning that Crossref mode fell back to local metadata."""

    print()
    print("WARNING: Crossref mode requested, but Crossref metadata was not used.")
    print(f"         {reason}")
    print("         Falling back to local PDF metadata.")


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


def extract_metadata_for_mode(
    pdf_path: Path,
    metadata_mode: str,
    crossref_mailto: str | None = None,
) -> tuple[ExtractedMetadata, str, bool]:
    """Extract metadata for the requested mode.

    Returns:
        metadata: the metadata to display
        actual_mode: "local" or "crossref"
        doi_lookup: whether ingest should attempt Crossref lookup
    """

    if metadata_mode == "local":
        metadata = extract_metadata(
            pdf_path,
            doi_lookup=False,
            crossref_mailto=crossref_mailto,
        )
        return metadata, "local", False

    local_metadata = extract_metadata(
        pdf_path,
        doi_lookup=False,
        crossref_mailto=crossref_mailto,
    )

    if local_metadata.doi is None:
        warn_crossref_fallback("No DOI was found in the PDF metadata/text.")
        return local_metadata, "local", False

    crossref_metadata = extract_metadata(
        pdf_path,
        doi_lookup=True,
        crossref_mailto=crossref_mailto,
    )

    return crossref_metadata, "crossref", True


def ingest_and_print_document(
    pdf_path: Path,
    store: StateStore,
    metadata_mode: str,
    crossref_mailto: str | None = None,
) -> None:
    """Ingest a PDF using the current mode and print that mode's metadata."""

    try:
        metadata, actual_mode, doi_lookup = extract_metadata_for_mode(
            pdf_path=pdf_path,
            metadata_mode=metadata_mode,
            crossref_mailto=crossref_mailto,
        )

        ingest_pdf(
            pdf_path,
            store,
            doi_lookup=doi_lookup,
            crossref_mailto=crossref_mailto,
        )
    except Exception as exc:
        if metadata_mode != "crossref":
            raise

        actual_mode = "local"
        warn_crossref_fallback(
            f"Crossref lookup failed with {type(exc).__name__}: {exc}"
        )

        metadata = extract_metadata(
            pdf_path,
            doi_lookup=False,
            crossref_mailto=crossref_mailto,
        )

        ingest_pdf(
            pdf_path,
            store,
            doi_lookup=False,
            crossref_mailto=crossref_mailto,
        )

    print_metadata_result(metadata, metadata_source=actual_mode)


def metadata_loop(
    pdf_paths: Sequence[Path],
    store: StateStore,
    crossref_mailto: str | None = None,
) -> None:
    """Prompt for PDFs and ingest using the selected metadata mode."""

    metadata_mode = "local"

    print()
    print("Metadata playground")
    print("Choose a PDF number to ingest using the current metadata mode.")
    print("Type local or crossref to switch modes.")
    print(f"Type {', '.join(QUIT_COMMANDS)} to leave.")

    if crossref_mailto is None:
        print()
        print(
            "No Crossref mailto address configured. DOI lookup can still run, "
            "but setting KURRENT_CROSSREF_MAILTO is more polite."
        )
    else:
        print()
        print(f"Crossref mailto: {crossref_mailto}")

    print_help(pdf_paths, metadata_mode)

    while True:
        print()

        try:
            user_input = input(f"metadata ({metadata_mode})> ").strip()
        except EOFError:
            print()
            return

        if user_input in QUIT_COMMANDS:
            return

        if not user_input:
            continue

        if user_input == "help":
            print_help(pdf_paths, metadata_mode)
            continue

        if user_input in METADATA_MODES:
            metadata_mode = METADATA_MODES[user_input]
            print(f"Metadata mode: {metadata_mode}")
            continue

        if not user_input.isdigit():
            print("Please enter a PDF number, local, crossref, help, or done.")
            continue

        index = int(user_input)

        if not 1 <= index <= len(pdf_paths):
            print(f"Please enter a number from 1 to {len(pdf_paths)}.")
            continue

        pdf_path = pdf_paths[index - 1]

        print()
        print(f"PDF: {pdf_path.name}")

        if metadata_mode == "crossref":
            print(
                "Using Crossref-enhanced metadata mode. Extra wait time here "
                "is Crossref lookup, not slow kurrent ingestion."
            )

        ingest_and_print_document(
            pdf_path=pdf_path,
            store=store,
            metadata_mode=metadata_mode,
            crossref_mailto=crossref_mailto,
        )


if __name__ == "__main__":

    if len(sys.argv) > 1:
        root_or_pdf = Path(sys.argv[1])
    else:
        root_or_pdf = DEFAULT_ROOT_DIR

    from kurrent.config import get_crossref_mailto
    crossref_mailto = get_crossref_mailto()

    PLAYGROUND_DIR.mkdir(parents=True, exist_ok=True)

    db_path = PLAYGROUND_DIR / "kurrent.db"
    prepare_fresh_playground_database(db_path)

    store = StateStore(db_path)

    try:
        pdf_paths = discover_pdfs(root_or_pdf)

        print(f"PDF source:     {root_or_pdf}")
        print(f"Database path:  {db_path}")
        print(f"PDFs found:     {len(pdf_paths)}")

        metadata_loop(
            pdf_paths=pdf_paths,
            store=store,
            crossref_mailto=crossref_mailto,
        )
    finally:
        cleanup_playground_database(db_path)
