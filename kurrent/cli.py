"""Command-line interface for kurrent.

Currently supported:

    kurrent ingest file.pdf
    kurrent ingest --local-metadata file.pdf
    kurrent ingest -r directoryOfPdfs
    kurrent ingest -y -r directoryOfPdfs

The default metadata mode is Crossref-enhanced metadata lookup. Use
--local-metadata to avoid network lookups.

The -y/--yes flag skips interactive metadata and heading review.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
import re
import sys
import time


QUIT_COMMANDS = {":q", ":quit", "done", "quit", "exit"}
CROSSREF_REQUEST_INTERVAL_SECONDS = 1.0

COMMON_SECTION_HEADINGS = {
    "abstract",
    "introduction",
    "background",
    "related work",
    "literature review",
    "theory",
    "model",
    "models",
    "methods",
    "method",
    "materials and methods",
    "data",
    "results",
    "analysis",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "future work",
    "acknowledgments",
    "acknowledgements",
    "references",
    "bibliography",
    "appendix",
    "supplementary material",
    "supporting information",
}

BAD_HEADING_PATTERNS = [
    r"^doi\b",
    r"^https?://",
    r"^www\.",
    r"^copyright\b",
    r"^©",
    r"^received\b",
    r"^accepted\b",
    r"^published\b",
    r"^author contributions\b",
    r"^the authors declare\b",
    r"^to whom correspondence\b",
    r"^this article",
    r"^pnas\b",
    r"^\d+\s+of\s+\d+$",
    r"^\d+$",
    r"^physical review\b",
    r"^journal of\b",
    r"^proceedings of\b",
    r"^proc\.\b",
    r"^vol\.\b",
    r"^volume\b",
    r"^no\.\b",
    r"^school of\b",
    r"^department of\b",
    r"^university of\b",
    r"^institute of\b",
    r"^college of\b",
    r"^faculty of\b",
    r"^center for\b",
    r"^centre for\b",
    r"^pacs number",
    r"^\(received\b",
    r"^\(revised\b",
    r"^\(accepted\b",
    r"^nih public access$",
    r"^author manuscript$",
    r"^nih-pa author manuscript$",
    r"^pmc\b",
    r"^available in pmc\b",
    r"^published in final edited form\b",
    r"^as:$",
    r"^corresponding author\b",
    r"^email:",
    r"^e-mail:",
    r"^tel:",
    r"^fax:",
    r"^\*",
    r"^†",
    r"^‡",
    r"^\d+\s*(program|center|centre|department|school|college|faculty|"
    r"institute|laboratory|lab)\b",
    r"^\(?ministry of education\)?",
    r"^canada\s+[a-z]\d[a-z]\s*\d[a-z]\d$",
    r"^usa$",
]


@dataclass(slots=True)
class IngestResult:
    """Result of one CLI ingest attempt."""

    pdf_path: Path
    doc_id: str | None
    already_existed: bool = False
    error: str | None = None


@dataclass(slots=True)
class IngestOutcome:
    """Successful result of ingesting one PDF."""

    doc_id: str
    already_existed: bool


def normalize_line(line: str) -> str:
    """Normalize one extracted-text line for heading review."""

    return re.sub(r"\s+", " ", line).strip()


def looks_like_bad_heading(line: str) -> bool:
    """Return True for obvious header/footer/citation junk."""

    lowered = line.lower().strip()

    if not lowered:
        return True

    return any(re.search(pattern, lowered) for pattern in BAD_HEADING_PATTERNS)


def looks_like_author_or_affiliation_line(line: str) -> bool:
    """Return whether a line looks like authors/affiliations, not a heading."""

    line = normalize_line(line)

    if re.search(r"\b[A-Z][A-Za-z.-]+[0-9,*†‡]", line):
        if "," in line or " and " in line:
            return True

    if re.match(
        r"^\d+\s*(program|center|centre|department|school|college|faculty|"
        r"institute|laboratory|lab)\b",
        line,
        flags=re.IGNORECASE,
    ):
        return True

    address_words = {
        "university",
        "college",
        "school",
        "department",
        "institute",
        "center",
        "centre",
        "laboratory",
        "cambridge",
        "beijing",
        "china",
        "canada",
        "usa",
    }
    words = set(re.findall(r"[A-Za-z]+", line.lower()))

    return len(words & address_words) >= 2


def looks_like_numbered_heading(line: str) -> bool:
    """Return whether a line looks like a numbered section heading."""

    line = normalize_line(line)

    patterns = [
        r"^[IVXLCDM]+\.\s+[A-Z][A-Za-z0-9 ,;:/()&\-]+$",
        r"^\d+(\.\d+)*\.?\s+[A-Z][A-Za-z0-9 ,;:/()&\-]+$",
        r"^[A-Z]\.\s+[A-Z][A-Za-z0-9 ,;:/()&\-]+$",
    ]

    return any(re.match(pattern, line) for pattern in patterns)


def looks_like_heading(line: str) -> bool:
    """Return whether a line looks like a plausible section heading.

    This intentionally favors precision over recall. False section headings are
    more irritating in an interactive ingest flow than missed headings.
    """

    line = normalize_line(line)

    if looks_like_bad_heading(line):
        return False

    if looks_like_author_or_affiliation_line(line):
        return False

    if len(line) < 3 or len(line) > 120:
        return False

    if line.lower() in COMMON_SECTION_HEADINGS:
        return True

    if looks_like_numbered_heading(line):
        return True

    return False


def dedupe_preserving_order(values: Iterable[str]) -> list[str]:
    """Return unique values while preserving first occurrence order."""

    seen = set()
    unique_values = []

    for value in values:
        key = value.lower()

        if key in seen:
            continue

        seen.add(key)
        unique_values.append(value)

    return unique_values


def extract_heading_candidates(
    pdf_path: Path,
    max_pages: int = 8,
) -> list[str]:
    """Return plausible section-heading candidates from early PDF text."""

    import pymupdf

    from kurrent.file_utils import silence_mupdf_messages

    silence_mupdf_messages()

    candidates: list[str] = []

    with pymupdf.open(pdf_path) as pdf:
        pages_examined = min(len(pdf), max_pages)

        for page_index in range(pages_examined):
            page = pdf.load_page(page_index)
            text = page.get_text("text", sort=True)

            for raw_line in text.splitlines():
                line = normalize_line(raw_line)

                if looks_like_heading(line):
                    candidates.append(line)

    candidates = dedupe_preserving_order(candidates)

    if len(candidates) < 2:
        return []

    return candidates


def print_metadata(metadata) -> None:
    """Print extracted metadata in a compact review format."""

    print()
    print("Metadata")
    print("--------")
    print(f"title:   {metadata.title}")
    print(f"authors: {metadata.authors}")
    print(f"year:    {metadata.year}")
    print(f"doi:     {metadata.doi}")


def prompt_text_field(label: str, current: str | None) -> str | None:
    """Prompt for one optional text metadata field."""

    shown = "" if current is None else current
    value = input(f"{label} [{shown}]: ").strip()

    if not value:
        return current

    return value


def prompt_year_field(current: int | None) -> int | None:
    """Prompt for an optional integer year field."""

    shown = "" if current is None else str(current)

    while True:
        value = input(f"year [{shown}]: ").strip()

        if not value:
            return current

        try:
            return int(value)
        except ValueError:
            print("Please enter a four-digit year, or press Enter to keep it.")


def review_metadata(metadata):
    """Let the user accept or correct extracted metadata."""

    from kurrent.schema import ExtractedMetadata

    print_metadata(metadata)
    print()
    print("Press Enter to keep a field unchanged.")
    print("Type corrected values where needed.")

    return ExtractedMetadata(
        title=prompt_text_field("title", metadata.title),
        authors=prompt_text_field("authors", metadata.authors),
        year=prompt_year_field(metadata.year),
        doi=prompt_text_field("doi", metadata.doi),
    )


def print_heading_candidates(headings: list[str]) -> None:
    """Print numbered heading candidates."""

    print()
    print("Section heading candidates")
    print("--------------------------")

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


def review_section_headings(pdf_path: Path) -> list[str]:
    """Let the user remove bogus section-heading candidates.

    These reviewed headings are not yet consumed by the current fixed-size
    chunker. This is a UI/development scaffold for the upcoming
    section-aware chunker.
    """

    headings = extract_heading_candidates(pdf_path)
    print_heading_candidates(headings)

    if not headings:
        return []

    print()
    print("Enter comma-separated numbers to remove bogus headings.")
    print("Press Enter to keep all headings.")

    while True:
        raw = input("remove headings> ").strip()

        if raw.lower() in QUIT_COMMANDS:
            raise KeyboardInterrupt("Ingest cancelled by user.")

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
        print("Accepted section headings:")
        if accepted:
            for heading in accepted:
                print(f"  - {heading}")
        else:
            print("  (none)")

        return accepted


def print_accepted_section_headings(pdf_path: Path) -> list[str]:
    """Print headings that -y/--yes accepts without interactive review."""

    headings = extract_heading_candidates(pdf_path)

    print_heading_candidates(headings)

    if headings:
        print()
        print("Accepted section headings without review because -y/--yes was used.")

    return headings


def metadata_update_kwargs(metadata) -> dict:
    """Return update_document_metadata kwargs for non-None metadata fields."""

    return {
        key: value
        for key, value in {
            "title": metadata.title,
            "authors": metadata.authors,
            "year": metadata.year,
            "doi": metadata.doi,
        }.items()
        if value is not None
    }


def ingest_pdf_with_metadata(
    pdf_path: Path,
    store,
    embedder,
    metadata,
    metadata_was_reviewed: bool,
) -> IngestOutcome:
    """Ingest one PDF using already-extracted metadata.

    This avoids doing Crossref lookup twice during interactive ingestion.
    The returned outcome records whether the document row already existed in
    kurrent state before this ingest command.
    """

    from kurrent.chunker import chunk_document
    from kurrent.file_utils import is_pdf, normalize_path, sha256_file
    from kurrent.schema import Document

    pdf_path = normalize_path(pdf_path)

    if not is_pdf(pdf_path):
        raise ValueError(f"No such PDF file {pdf_path}")

    pdf_sha256 = sha256_file(pdf_path)
    existing = store.get_document_by_sha256(pdf_sha256)
    already_existed = existing is not None

    if existing is None:
        document = Document.for_pdf(
            pdf_path=pdf_path,
            pdf_sha256=pdf_sha256,
            metadata=metadata,
        )
        store.insert_document(document)
        doc_id = document.doc_id
    else:
        doc_id = existing.doc_id

        if metadata_was_reviewed:
            updates = metadata_update_kwargs(metadata)

            if updates:
                store.update_document_metadata(doc_id, **updates)

    chunk_document(doc_id, store)
    embedder.index_chunks(doc_id, store)

    return IngestOutcome(
        doc_id=doc_id,
        already_existed=already_existed,
    )


def ingest_one_pdf(
    pdf_path: Path,
    store,
    embedder,
    doi_lookup: bool,
    crossref_mailto: str | None,
    assume_yes: bool,
) -> IngestOutcome:
    """Ingest one PDF through the CLI workflow."""

    from kurrent.metadata_extractor import extract_metadata

    print()
    print(f"PDF: {pdf_path}", flush=True)

    metadata = extract_metadata(
        pdf_path,
        doi_lookup=doi_lookup,
        crossref_mailto=crossref_mailto,
    )

    metadata_was_reviewed = False

    if assume_yes:
        print_metadata(metadata)
        print_accepted_section_headings(pdf_path)
    else:
        metadata = review_metadata(metadata)
        metadata_was_reviewed = True
        review_section_headings(pdf_path)

    outcome = ingest_pdf_with_metadata(
        pdf_path=pdf_path,
        store=store,
        embedder=embedder,
        metadata=metadata,
        metadata_was_reviewed=metadata_was_reviewed,
    )

    print()

    if outcome.already_existed:
        print(
            "Already in kurrent state; using existing kurrent ID: "
            f"{outcome.doc_id}",
            flush=True,
        )
    else:
        print(f"Created new kurrent ID: {outcome.doc_id}", flush=True)

    return outcome


def ingest_targets(path: Path, recursive: bool) -> list[Path]:
    """Return PDF paths selected by CLI arguments."""

    from kurrent.file_utils import is_pdf, normalize_path

    path = normalize_path(path)

    if recursive:
        if not path.is_dir():
            raise NotADirectoryError(f"Recursive ingest requires a directory: {path}")

        return sorted(
            candidate
            for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix.lower() == ".pdf"
        )

    if path.is_dir():
        raise IsADirectoryError(
            "Directory ingest requires -r/--recursive. "
            f"Got directory: {path}"
        )

    if not is_pdf(path):
        raise ValueError(f"Not a PDF file: {path}")

    return [path]


def run_ingest(args: argparse.Namespace) -> int:
    """Run the kurrent ingest command."""

    print("Starting kurrent ingest...", flush=True)

    from kurrent.config import get_crossref_mailto, get_kurrent_state_paths

    state_paths = get_kurrent_state_paths(args.state_dir)

    if state_paths.state_dir.exists():
        print(f"kurrent state directory: {state_paths.state_dir}", flush=True)
    else:
        print(
            "kurrent state directory does not exist; creating it now: "
            f"{state_paths.state_dir}",
            flush=True,
        )
        state_paths.state_dir.mkdir(parents=True, exist_ok=True)

    print("Finding PDFs...", flush=True)

    pdf_paths = ingest_targets(args.path, recursive=args.recursive)

    if not pdf_paths:
        print(f"No PDFs found under: {args.path}")
        return 0

    doi_lookup = args.metadata_mode == "crossref"
    crossref_mailto = get_crossref_mailto()

    print(f"PDFs selected:           {len(pdf_paths)}", flush=True)

    if state_paths.sqlite_path.exists():
        print(f"SQLite database:         {state_paths.sqlite_path}", flush=True)
    else:
        print(
            "SQLite database does not exist; it will be created: "
            f"{state_paths.sqlite_path}",
            flush=True,
        )

    if state_paths.chroma_path.exists():
        print(f"Chroma directory:        {state_paths.chroma_path}", flush=True)
    else:
        print(
            "Chroma directory does not exist; it will be created: "
            f"{state_paths.chroma_path}",
            flush=True,
        )

    print(f"Metadata mode:           {args.metadata_mode}", flush=True)

    if doi_lookup and crossref_mailto is None:
        print()
        print(
            "No Crossref mailto address configured. Crossref lookup can still "
            "run, but setting KURRENT_CROSSREF_MAILTO is more polite.",
            flush=True,
        )

    print()
    print("Loading kurrent state store...", flush=True)
    from kurrent.state_store import StateStore

    print("Loading embedding model / Chroma index...", flush=True)
    from kurrent.embedder import Embedder

    store = StateStore(state_paths.sqlite_path)
    embedder = Embedder(chroma_path=state_paths.chroma_path)

    print("Ready. Beginning PDF ingest.", flush=True)

    results: list[IngestResult] = []

    try:
        for i, pdf_path in enumerate(pdf_paths, start=1):
            print()
            print(f"[{i}/{len(pdf_paths)}] {pdf_path}", flush=True)

            try:
                outcome = ingest_one_pdf(
                    pdf_path=pdf_path,
                    store=store,
                    embedder=embedder,
                    doi_lookup=doi_lookup,
                    crossref_mailto=crossref_mailto,
                    assume_yes=args.assume_yes,
                )
                results.append(
                    IngestResult(
                        pdf_path=pdf_path,
                        doc_id=outcome.doc_id,
                        already_existed=outcome.already_existed,
                    )
                )
            except KeyboardInterrupt:
                print()
                print("Cancelled.")
                return 130
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"Could not ingest {pdf_path}: {message}")
                results.append(
                    IngestResult(
                        pdf_path=pdf_path,
                        doc_id=None,
                        error=message,
                    )
                )

            if doi_lookup and i < len(pdf_paths):
                time.sleep(CROSSREF_REQUEST_INTERVAL_SECONDS)
    finally:
        store.close()

    succeeded = [result for result in results if result.doc_id is not None]
    created = [
        result
        for result in succeeded
        if not result.already_existed
    ]
    already_ingested = [
        result
        for result in succeeded
        if result.already_existed
    ]
    failed = [result for result in results if result.error is not None]

    print()
    print("Ingest summary")
    print("--------------")
    print(f"PDFs selected:     {len(pdf_paths)}")
    print(f"New documents:     {len(created)}")
    print(f"Already ingested:  {len(already_ingested)}")
    print(f"Failed:            {len(failed)}")

    if failed:
        print()
        print("Failures:")
        for result in failed:
            print(f"  {result.pdf_path}: {result.error}")

    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level kurrent CLI parser."""

    parser = argparse.ArgumentParser(
        prog="kurrent",
        description="kurrent command-line research-literature manager.",
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing kurrent.db and Chroma state. If omitted, "
            "KURRENT_STATE_DIR from .env is used."
        ),
    )

    subparsers = parser.add_subparsers(
        title="commands",
        dest="command",
        metavar="command",
        required=True,
    )

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="ingest PDFs into kurrent state",
    )
    ingest_parser.add_argument(
        "path",
        type=Path,
        help="PDF file, or directory when -r/--recursive is supplied.",
    )
    ingest_parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="recursively ingest PDFs under a directory.",
    )
    ingest_parser.add_argument(
        "-y",
        "--yes",
        dest="assume_yes",
        action="store_true",
        help="accept extracted metadata and section headings without prompts.",
    )

    metadata_group = ingest_parser.add_mutually_exclusive_group()
    metadata_group.add_argument(
        "--local-metadata",
        action="store_const",
        const="local",
        dest="metadata_mode",
        help="use local PDF metadata/text only; do not query Crossref.",
    )
    metadata_group.add_argument(
        "--crossref-metadata",
        action="store_const",
        const="crossref",
        dest="metadata_mode",
        help=(
            "use Crossref-enhanced metadata lookup when a DOI is found "
            "(default)."
        ),
    )
    ingest_parser.set_defaults(
        func=run_ingest,
        metadata_mode="crossref",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""

    parser = build_parser()
    args = parser.parse_args(argv)

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
